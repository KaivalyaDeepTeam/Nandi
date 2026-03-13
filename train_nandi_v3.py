"""
Train Nandi V3 — Discrete-Action Trading System.

Supports two algorithms:
  --algo dqn  (default): Rainbow-IQN-DQN (3-phase pipeline)
  --algo ppo:            PPO with discrete actions (2-phase pipeline)

PPO advantage: outputs action probabilities directly, avoiding Q-value
dominance where DQN's argmax always picks HOLD.

Usage:
    python train_nandi_v3.py --algo ppo --pairs eurusd gbpusd
    python train_nandi_v3.py --algo ppo --timesteps 500000
    python train_nandi_v3.py --algo dqn --timesteps 100000
    python train_nandi_v3.py --algo ppo --skip-hoa --timesteps 500000
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nandi")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_envs(pair_data, pairs, lookback, timeframe, episode_length,
                reward_fn=None, is_eval=False):
    """Create discrete-action environments for all pairs."""
    from nandi.environment.discrete_env import MultiEpisodeDiscreteEnv
    from nandi.environment.market_sim import MarketSimulator

    envs = []
    for pair in pairs:
        if pair not in pair_data:
            continue

        data = pair_data[pair]
        if is_eval:
            features = data["test_features"]
            prices = data["test_prices"]
            ep_len = len(prices) - lookback - 2
            if ep_len < 10:
                logger.warning(f"Skipping {pair} eval — insufficient test data")
                continue
        else:
            features = data["train_features"]
            prices = data["train_prices"]
            ep_len = episode_length

        market_sim = MarketSimulator(pair_name=pair, timeframe=timeframe)
        env = MultiEpisodeDiscreteEnv(
            features=features,
            prices=prices,
            lookback=lookback,
            episode_length=ep_len,
            pair_name=pair,
            market_sim=market_sim,
            timeframe=timeframe,
            reward_fn=reward_fn,
        )
        envs.append(env)

    return envs


def main():
    parser = argparse.ArgumentParser(
        description="Train Nandi V3 — Discrete-Action Trading",
    )
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["dqn", "ppo"],
                        help="Algorithm: ppo (recommended) or dqn")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Training steps (Phase 2)")
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="Pairs to train (default: all 8)")
    parser.add_argument("--timeframe", type=str, default="M5",
                        choices=["M5", "M1", "H1", "D1"],
                        help="Trading timeframe (default: M5)")
    parser.add_argument("--encoder", type=str, default="msfan",
                        choices=["msfan", "tft", "ssm"],
                        help="Encoder architecture (default: msfan)")
    parser.add_argument("--skip-hoa", action="store_true",
                        help="Skip Phase 1 (HOA pre-training)")
    parser.add_argument("--skip-hardening", action="store_true",
                        help="Skip Phase 3 (risk hardening, DQN only)")
    parser.add_argument("--hardening-steps", type=int, default=50_000,
                        help="Phase 3 training steps (DQN only)")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Load agent from checkpoint path")
    args = parser.parse_args()

    from nandi.config import (
        MODEL_DIR, PAIRS, TIMEFRAME_PROFILES,
        DQN_CONFIG, PPO_CONFIG, PPO_CONFIG_H1,
        HOA_CONFIG, HOA_CONFIG_H1, TRAINING_CONFIG,
    )
    from nandi.data.manager import DataManager
    from nandi.environment.mfe_reward import MFEMAEReward
    from nandi.training.hoa_pretrainer import HOAPretrainer

    use_ppo = args.algo == "ppo"

    pairs = args.pairs or PAIRS
    profile = TIMEFRAME_PROFILES.get(args.timeframe, TIMEFRAME_PROFILES["M5"])
    lookback = profile["lookback_bars"]
    episode_length = profile["episode_bars"]
    device = get_device()
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Config overrides — select per-timeframe configs
    dqn_config = dict(DQN_CONFIG)
    dqn_config["total_steps"] = args.timesteps
    dqn_config["hardening_steps"] = args.hardening_steps

    ppo_config = dict(PPO_CONFIG_H1 if args.timeframe == "H1" else PPO_CONFIG)

    hoa_config = dict(HOA_CONFIG_H1 if args.timeframe == "H1" else HOA_CONFIG)

    training_config = dict(TRAINING_CONFIG)
    training_config["total_timesteps"] = args.timesteps

    algo_name = "PPO" if use_ppo else "DQN"
    logger.info("=" * 60)
    logger.info(f"  NANDI V3 — Discrete-Action {algo_name} Trading System")
    logger.info("=" * 60)
    logger.info(f"  Algorithm:    {algo_name}")
    logger.info(f"  Timeframe:    {args.timeframe}")
    logger.info(f"  Encoder:      {args.encoder.upper()}")
    logger.info(f"  Device:       {device}")
    logger.info(f"  Pairs:        {', '.join(p.upper() for p in pairs)}")
    logger.info(f"  Steps:        {args.timesteps:,}")
    logger.info(f"  Lookback:     {lookback} bars")
    logger.info(f"  Episode:      {episode_length} bars")
    logger.info(f"  Phase 1 HOA:  {'SKIP' if args.skip_hoa else 'ON'}")
    if not use_ppo:
        logger.info(f"  Phase 3 Risk: {'SKIP' if args.skip_hardening else 'ON'}")
    logger.info("=" * 60)

    # ═══════════════════════════════════════════════════════════
    # Step 1: Download & prepare data for all pairs
    # ═══════════════════════════════════════════════════════════
    logger.info("\n[1/4] Preparing data...")
    dm = DataManager(
        pairs=pairs, years=args.years, test_months=args.test_months,
        timeframe=args.timeframe,
    )
    pair_data = dm.prepare_all()

    if not pair_data:
        logger.error("No pair data available. Check network and retry.")
        sys.exit(1)

    first_pair = next(iter(pair_data))
    n_features = pair_data[first_pair]["n_features"]
    logger.info(f"Features: {n_features} | Pairs with data: {len(pair_data)}")

    # ═══════════════════════════════════════════════════════════
    # Step 2: Create agent
    # ═══════════════════════════════════════════════════════════
    if use_ppo:
        from nandi.models.ppo_agent import NandiPPOAgent
        logger.info(f"\n[2/4] Creating NandiPPOAgent...")
        agent = NandiPPOAgent(
            n_features=n_features,
            dqn_config=dqn_config,
            encoder_type=args.encoder,
        )
    else:
        from nandi.models.dqn_agent import NandiDQNAgent
        logger.info(f"\n[2/4] Creating NandiDQNAgent...")
        agent = NandiDQNAgent(
            n_features=n_features,
            dqn_config=dqn_config,
            encoder_type=args.encoder,
        )

    if args.load_checkpoint:
        logger.info(f"Loading checkpoint: {args.load_checkpoint}")
        agent.load_agent(args.load_checkpoint)

    # Warmup forward pass
    dummy_ms = torch.zeros(1, lookback, n_features)
    dummy_pi = torch.zeros(1, DQN_CONFIG["position_dim"])
    dummy_pair = torch.zeros(1, dtype=torch.long)
    if use_ppo:
        agent.get_policy_and_value(dummy_ms, dummy_pi, dummy_pair)
    else:
        agent(dummy_ms, dummy_pi, dummy_pair)

    n_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    logger.info(f"Agent: {n_params:,} parameters")

    # ═══════════════════════════════════════════════════════════
    # Phase 1: HOA Pre-training
    # ═══════════════════════════════════════════════════════════
    if not args.skip_hoa and not args.load_checkpoint:
        logger.info("\n[Phase 1] HOA Pre-training...")
        pretrainer = HOAPretrainer(
            agent=agent,
            pair_data=pair_data,
            lookback=lookback,
            timeframe=args.timeframe,
            hoa_config=hoa_config,
            device=device,
            position_aware=not use_ppo,   # PPO: flat labels (more entry signal)
            price_flip_augment=use_ppo,   # PPO: augment to prevent directional bias
        )
        hoa_metrics = pretrainer.train()
        logger.info(f"HOA metrics: {hoa_metrics}")

        # Save HOA checkpoint
        hoa_name = "ppo_agent_hoa.pt" if use_ppo else "dqn_agent_hoa.pt"
        hoa_path = os.path.join(MODEL_DIR, hoa_name)
        agent.save_agent(hoa_path)
        logger.info(f"HOA checkpoint saved: {hoa_path}")
    else:
        logger.info("\n[Phase 1] Skipped")

    # ═══════════════════════════════════════════════════════════
    # Phase 2: RL Training (PPO or DQN)
    # ═══════════════════════════════════════════════════════════
    logger.info(f"\n[Phase 2] {algo_name} Training...")

    # Create reward function
    reward_fn = MFEMAEReward(
        max_hold_bars=profile.get("max_hold_bars", 36),
    )

    # Create training and eval environments
    train_envs = create_envs(
        pair_data, pairs, lookback, args.timeframe, episode_length,
        reward_fn=reward_fn, is_eval=False,
    )
    eval_envs = create_envs(
        pair_data, pairs, lookback, args.timeframe, episode_length,
        reward_fn=reward_fn, is_eval=True,
    )

    if not train_envs:
        logger.error("No training environments created. Check data.")
        sys.exit(1)

    logger.info(f"Training envs: {len(train_envs)} | Eval envs: {len(eval_envs)}")

    if use_ppo:
        from nandi.training.ppo_trainer import PPOTrainer
        trainer = PPOTrainer(
            agent=agent,
            train_envs=train_envs,
            eval_envs=eval_envs,
            ppo_config=ppo_config,
            training_config=training_config,
            device=device,
            freeze_encoder=not args.skip_hoa,
        )
    else:
        from nandi.training.dqn_trainer import DQNTrainer
        trainer = DQNTrainer(
            agent=agent,
            train_envs=train_envs,
            eval_envs=eval_envs,
            dqn_config=dqn_config,
            training_config=training_config,
            device=device,
            freeze_encoder=not args.skip_hoa,
        )

    rl_stats = trainer.train()

    # Save Phase 2 checkpoint
    p2_name = "ppo_agent_phase2.pt" if use_ppo else "dqn_agent_phase2.pt"
    p2_path = os.path.join(MODEL_DIR, p2_name)
    agent.save_agent(p2_path)
    logger.info(f"Phase 2 checkpoint saved: {p2_path}")

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Risk Hardening (DQN only)
    # ═══════════════════════════════════════════════════════════
    if not use_ppo and not args.skip_hardening:
        logger.info("\n[Phase 3] Risk Hardening...")
        from nandi.training.risk_hardening import RiskHardeningTrainer
        hardener = RiskHardeningTrainer(
            agent=agent,
            train_envs=train_envs,
            eval_envs=eval_envs,
            dqn_config=dqn_config,
            training_config=training_config,
            device=device,
        )
        hardening_stats = hardener.train()

        final_path = os.path.join(MODEL_DIR, "dqn_agent_final.pt")
        agent.save_agent(final_path)
        logger.info(f"Final checkpoint saved: {final_path}")
    elif not use_ppo:
        logger.info("\n[Phase 3] Skipped")

    # ═══════════════════════════════════════════════════════════
    # Save final model and metadata
    # ═══════════════════════════════════════════════════════════
    final_name = "ppo_agent.pt" if use_ppo else "dqn_agent.pt"
    agent.save_agent(os.path.join(MODEL_DIR, final_name))

    # Save per-pair scalers and metadata
    for pair in pairs:
        if pair not in pair_data:
            continue
        data = pair_data[pair]
        pair_dir = os.path.join(MODEL_DIR, pair)
        os.makedirs(pair_dir, exist_ok=True)
        joblib.dump(data["scaler"], os.path.join(pair_dir, "scaler.pkl"))
        joblib.dump(data["feature_names"],
                    os.path.join(pair_dir, "feature_names.pkl"))
        joblib.dump({
            "algo": args.algo, "encoder": args.encoder,
            "timeframe": args.timeframe, "version": "v3",
        }, os.path.join(pair_dir, "meta.pkl"))

    joblib.dump(list(pair_data.keys()),
                os.path.join(MODEL_DIR, "trained_pairs.pkl"))
    joblib.dump({
        "algo": args.algo, "encoder": args.encoder,
        "timeframe": args.timeframe, "version": "v3",
        "n_features": n_features,
    }, os.path.join(MODEL_DIR, "training_meta.pkl"))

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  NANDI V3 TRAINING COMPLETE ({algo_name})")
    logger.info(f"  Model saved to: {MODEL_DIR}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
