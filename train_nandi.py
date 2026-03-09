"""
Train Nandi V2 — Multi-Pair Portfolio Training.

Downloads 20 years of data for 8 forex pairs, trains RL agents
per pair using the selected algorithm and encoder, then evaluates
portfolio-level performance.

Usage:
    python train_nandi.py
    python train_nandi.py --algo sac --encoder tft --timesteps 500000
    python train_nandi.py --algo ensemble --pairs eurusd gbpusd usdjpy
    python train_nandi.py --pairs eurusd --years 10 --curriculum
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


def create_trainer(algo, agent, train_env, eval_env, training_config):
    """Create the appropriate trainer for the selected algorithm."""
    if algo == "ppo":
        from nandi.training.ppo_trainer import NandiTrainer
        return NandiTrainer(
            agent=agent, train_env=train_env, eval_env=eval_env,
            training_config=training_config,
        )
    elif algo == "sac":
        from nandi.training.sac_trainer import SACTrainer
        return SACTrainer(
            agent=agent, train_env=train_env, eval_env=eval_env,
            training_config=training_config,
        )
    elif algo == "td3":
        from nandi.training.td3_trainer import TD3Trainer
        return TD3Trainer(
            agent=agent, train_env=train_env, eval_env=eval_env,
            config=training_config,
        )
    elif algo == "aegis":
        from nandi.training.aegis_trainer import AEGISTrainer
        from nandi.config import AEGIS_CONFIG
        # Filter out agent-only params (regime_dim is set on AEGISAgent, not trainer)
        trainer_keys = {
            "cvar_alpha", "n_quantiles", "asymmetry_factor", "kl_coef",
            "edge_coef", "edge_util_coef", "batch_size", "buffer_capacity",
            "tau_soft", "gamma", "warmup_steps",
        }
        trainer_params = {k: v for k, v in AEGIS_CONFIG.items() if k in trainer_keys}
        # Map learning_rate -> lr (trainer uses 'lr')
        trainer_params["lr"] = AEGIS_CONFIG.get("learning_rate", 3e-4)
        return AEGISTrainer(
            agent=agent, train_env=train_env, eval_env=eval_env,
            training_config=training_config,
            **trainer_params,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Use ppo/sac/td3/aegis/ensemble")


def train_single_pair(pair, data, algo, encoder_type, training_config,
                      use_curriculum, lookback, timeframe="D1"):
    """Train a single pair and return (agent, stats)."""
    from nandi.config import PPO_CONFIG, TIMEFRAME_PROFILES
    from nandi.environment.single_pair_env import MultiEpisodeEnv
    from nandi.environment.market_sim import MarketSimulator
    from nandi.models.agent import NandiAgent

    profile = TIMEFRAME_PROFILES.get(timeframe, TIMEFRAME_PROFILES["D1"])

    logger.info(f"\n{'─' * 60}")
    logger.info(f"  Training {pair.upper()} | {timeframe} | algo={algo} | encoder={encoder_type} | "
                f"{data['n_features']} features")
    logger.info(f"{'─' * 60}")

    market_sim = MarketSimulator(pair_name=pair, timeframe=timeframe)
    episode_length = profile["episode_bars"]

    train_env = MultiEpisodeEnv(
        features=data["train_features"],
        prices=data["train_prices"],
        lookback=lookback,
        episode_length=episode_length,
        pair_name=pair,
        market_sim=market_sim,
        timeframe=timeframe,
    )

    test_len = len(data["test_prices"]) - lookback - 2
    if test_len < 10:
        logger.warning(f"Skipping {pair} — insufficient test data")
        return None, None

    eval_env = MultiEpisodeEnv(
        features=data["test_features"],
        prices=data["test_prices"],
        lookback=lookback,
        episode_length=test_len,
        pair_name=pair,
        market_sim=market_sim,
        timeframe=timeframe,
    )

    # Create agent with selected encoder
    if algo == "aegis":
        from nandi.models.aegis import AEGISAgent
        from nandi.config import AEGIS_CONFIG
        agent = AEGISAgent(
            n_features=data["n_features"],
            encoder_type=encoder_type,
            regime_dim=AEGIS_CONFIG["regime_dim"],
            cvar_alpha=AEGIS_CONFIG["cvar_alpha"],
        )
    else:
        agent = NandiAgent(n_features=data["n_features"], encoder_type=encoder_type)
    dummy_ms = torch.zeros(1, lookback, data["n_features"])
    dummy_pi = torch.zeros(1, 4)
    agent(dummy_ms, dummy_pi)

    n_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    logger.info(f"Agent: {n_params:,} params | Device: {get_device()}")

    # Apply curriculum learning if requested
    if use_curriculum:
        from nandi.training.curriculum import CurriculumScheduler
        curriculum = CurriculumScheduler()
        logger.info(f"Curriculum learning enabled — starting at stage: {curriculum.stage_name}")

    trainer = create_trainer(algo, agent, train_env, eval_env, training_config)
    stats = trainer.train()

    return agent, stats


def train_ensemble_pair(pair, data, encoder_type, training_config, lookback,
                        timeframe="D1"):
    """Train PPO + SAC + TD3 independently, return EnsembleAgent."""
    from nandi.training.ensemble_trainer import EnsembleAgent

    agents_list = []
    for algo in ["ppo", "sac", "td3"]:
        logger.info(f"\n  [{pair.upper()}] Training {algo.upper()} for ensemble...")
        agent, stats = train_single_pair(
            pair, data, algo, encoder_type, training_config,
            use_curriculum=False, lookback=lookback, timeframe=timeframe,
        )
        if agent is not None:
            agents_list.append(agent)

    if not agents_list:
        return None, None

    ensemble = EnsembleAgent(agents_list)
    logger.info(f"  [{pair.upper()}] Ensemble created with {len(agents_list)} agents")
    return ensemble, {"algo": "ensemble", "n_agents": len(agents_list)}


def main():
    parser = argparse.ArgumentParser(description="Train Nandi V2 Multi-Pair RL Agents")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="Pairs to train (default: all 8)")
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=["ppo", "sac", "td3", "aegis", "ensemble"],
                        help="RL algorithm (default: ppo)")
    parser.add_argument("--encoder", type=str, default="msfan",
                        choices=["msfan", "tft", "ssm"],
                        help="Encoder architecture (default: msfan)")
    parser.add_argument("--timeframe", type=str, default="D1",
                        choices=["D1", "M5"],
                        help="Trading timeframe (default: D1)")
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable curriculum learning")
    args = parser.parse_args()

    from nandi.config import MODEL_DIR, TRAINING_CONFIG, PAIRS, TIMEFRAME_PROFILES
    from nandi.data.manager import DataManager
    from nandi.training.evaluator import BacktestEvaluator, PortfolioEvaluator

    pairs = args.pairs or PAIRS
    profile = TIMEFRAME_PROFILES.get(args.timeframe, TIMEFRAME_PROFILES["D1"])
    lookback = profile["lookback_bars"]
    os.makedirs(MODEL_DIR, exist_ok=True)

    device = get_device()
    logger.info("=" * 60)
    logger.info("  NANDI V2 — Multi-Pair Portfolio Training")
    logger.info("=" * 60)
    logger.info(f"  Timeframe:  {args.timeframe}")
    logger.info(f"  Algorithm:  {args.algo.upper()}")
    logger.info(f"  Encoder:    {args.encoder.upper()}")
    logger.info(f"  Device:     {device}")
    logger.info(f"  Pairs:      {', '.join(p.upper() for p in pairs)}")
    logger.info(f"  Timesteps:  {args.timesteps:,} per pair")
    logger.info(f"  Leverage:   {profile['leverage']}x")
    logger.info(f"  MaxPos:     {profile['max_position']}")
    logger.info(f"  Lookback:   {lookback} bars")
    logger.info(f"  Curriculum: {'ON' if args.curriculum else 'OFF'}")
    logger.info("=" * 60)

    # ── Step 1: Download & prepare data for all pairs ──
    dm = DataManager(pairs=pairs, years=args.years, test_months=args.test_months,
                     timeframe=args.timeframe)
    pair_data = dm.prepare_all()

    if not pair_data:
        logger.error("No pair data available. Check network and retry.")
        sys.exit(1)

    # ── Step 2: Train each pair ──
    agents = {}
    training_results = {}

    training_config = TRAINING_CONFIG.copy()
    training_config["total_timesteps"] = args.timesteps

    for pair in pairs:
        if pair not in pair_data:
            logger.warning(f"Skipping {pair} — no data available")
            continue

        data = pair_data[pair]

        if args.algo == "ensemble":
            agent, stats = train_ensemble_pair(
                pair, data, args.encoder, training_config, lookback,
                timeframe=args.timeframe,
            )
        else:
            agent, stats = train_single_pair(
                pair, data, args.algo, args.encoder, training_config,
                args.curriculum, lookback, timeframe=args.timeframe,
            )

        if agent is None:
            continue

        # Save per-pair model and scaler
        pair_model_dir = os.path.join(MODEL_DIR, pair)
        os.makedirs(pair_model_dir, exist_ok=True)
        agent.save_agent(os.path.join(pair_model_dir, "agent.pt"))
        joblib.dump(data["scaler"], os.path.join(pair_model_dir, "scaler.pkl"))
        joblib.dump(data["feature_names"], os.path.join(pair_model_dir, "feature_names.pkl"))
        joblib.dump({"algo": args.algo, "encoder": args.encoder,
                     "timeframe": args.timeframe},
                    os.path.join(pair_model_dir, "meta.pkl"))

        agents[pair] = agent
        training_results[pair] = stats
        logger.info(f"[{pair.upper()}] Training complete — model saved")

    if not agents:
        logger.error("No agents trained successfully.")
        sys.exit(1)

    # ── Step 3: Portfolio evaluation ──
    logger.info(f"\n{'=' * 60}")
    logger.info("  PORTFOLIO EVALUATION ON UNSEEN TEST DATA")
    logger.info(f"{'=' * 60}")

    pair_evaluators = {}
    for pair in agents:
        pair_evaluators[pair] = BacktestEvaluator(agents[pair], pair_name=pair)

    portfolio_eval = PortfolioEvaluator(pair_evaluators)
    results = portfolio_eval.evaluate_portfolio(pair_data)

    pm = results["portfolio_metrics"]
    logger.info(f"\n  Portfolio Return:  {pm['total_return_pct']:+.2f}%")
    logger.info(f"  Portfolio Sharpe:  {pm['sharpe']:.2f}")
    logger.info(f"  Portfolio Sortino: {pm['sortino']:.2f}")
    logger.info(f"  Portfolio Calmar:  {pm['calmar']:.2f}")
    logger.info(f"  Portfolio MaxDD:   {pm['max_drawdown']:.2%}")

    logger.info(f"\n  Per-Pair Results:")
    for pair, metrics in results["pair_metrics"].items():
        logger.info(
            f"    {pair.upper():>8s}: "
            f"Return={metrics['total_return_pct']:+7.2f}% | "
            f"Sharpe={metrics['sharpe']:5.2f} | "
            f"MaxDD={metrics['max_drawdown']:.2%} | "
            f"WR={metrics['win_rate']:.1%} | "
            f"Trades={metrics['n_trades']}"
        )

    # Save portfolio config
    joblib.dump(list(agents.keys()), os.path.join(MODEL_DIR, "trained_pairs.pkl"))
    joblib.dump({"algo": args.algo, "encoder": args.encoder,
                 "timeframe": args.timeframe},
                os.path.join(MODEL_DIR, "training_meta.pkl"))

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  NANDI V2 TRAINING COMPLETE — {args.algo.upper()}/{args.encoder.upper()}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
