"""
Train Nandi V4 — SPIN: Stochastic Path Intelligence Network.

55K-param model with hard-wired risk management on M5 data.
2-phase pipeline: HOA pre-training (stop-loss-aware) -> PPO fine-tuning.

Usage:
    python train_nandi_v4.py --algo spin --pairs eurusd --timeframe M5 --timesteps 200000
    python train_nandi_v4.py --algo spin --timeframe M5 --timesteps 1000000
    python train_nandi_v4.py --algo spin --skip-hoa --timesteps 500000
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


def create_spin_envs(pair_data, pairs, lookback, episode_length,
                     risk_config=None, reward_fn=None, is_eval=False):
    """Create SPIN trading environments for all pairs."""
    from nandi.environment.spin_env import MultiEpisodeSPINEnv

    envs = []
    for pair in pairs:
        if pair not in pair_data:
            continue

        data = pair_data[pair]
        if is_eval:
            features = data["test_features"]
            prices = data["test_prices"]
            atr = data.get("atr_test")
            h1_trend = data.get("h1_trend_test")
            ep_len = min(episode_length, len(prices) - lookback - 2)
            if ep_len < 10:
                logger.warning(f"Skipping {pair} eval — insufficient test data")
                continue
        else:
            features = data["train_features"]
            prices = data["train_prices"]
            atr = data.get("atr_train")
            h1_trend = data.get("h1_trend_train")
            ep_len = episode_length

        env = MultiEpisodeSPINEnv(
            features=features,
            prices=prices,
            lookback=lookback,
            episode_length=ep_len,
            pair_name=pair,
            timeframe="M5",
            atr_series=atr,
            h1_trend_series=h1_trend,
            risk_config=risk_config,
            reward_fn=reward_fn,
        )
        envs.append(env)

    return envs


def main():
    parser = argparse.ArgumentParser(
        description="Train Nandi V4 — SPIN Trading System",
    )
    parser.add_argument("--algo", type=str, default="spin",
                        choices=["spin"],
                        help="Algorithm: spin")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Training steps (Phase 2)")
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="Pairs to train (default: all 8)")
    parser.add_argument("--timeframe", type=str, default="M5",
                        choices=["M5"],
                        help="Trading timeframe (M5 only for SPIN)")
    parser.add_argument("--skip-hoa", action="store_true",
                        help="Skip Phase 1 (HOA pre-training)")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Load agent from checkpoint path")
    args = parser.parse_args()

    from nandi.config import (
        MODEL_DIR, PAIRS, SPIN_CONFIG, SPIN_PPO_CONFIG,
        SPIN_HOA_CONFIG, SPIN_RISK_CONFIG, TRAINING_CONFIG,
    )
    from nandi.data.manager import DataManager
    from nandi.models.spin_agent import SPINAgent
    from nandi.environment.spin_reward import SPINReward
    from nandi.training.hoa_pretrainer import HOAPretrainer, compute_spin_hoa_labels

    pairs = args.pairs or PAIRS
    lookback = SPIN_CONFIG["lookback_bars"]
    episode_length = 2016  # M5 episode
    device = get_device()
    os.makedirs(MODEL_DIR, exist_ok=True)

    ppo_config = dict(SPIN_PPO_CONFIG)
    hoa_config = dict(SPIN_HOA_CONFIG)
    risk_config = dict(SPIN_RISK_CONFIG)
    training_config = dict(TRAINING_CONFIG)
    training_config["total_timesteps"] = args.timesteps

    logger.info("=" * 60)
    logger.info("  NANDI V4 — SPIN: Stochastic Path Intelligence Network")
    logger.info("=" * 60)
    logger.info(f"  Device:       {device}")
    logger.info(f"  Pairs:        {', '.join(p.upper() for p in pairs)}")
    logger.info(f"  Steps:        {args.timesteps:,}")
    logger.info(f"  Lookback:     {lookback} bars")
    logger.info(f"  Episode:      {episode_length} bars")
    logger.info(f"  Risk gates:   SL={risk_config['stop_loss_atr_mult']}×ATR  "
                f"MaxHold={risk_config['max_hold_bars']}  "
                f"Cooldown={risk_config['cooldown_bars']}  "
                f"TrendFilter={'ON' if risk_config['trend_filter'] else 'OFF'}")
    logger.info(f"  Phase 1 HOA:  {'SKIP' if args.skip_hoa else 'ON'}")
    logger.info("=" * 60)

    # ═══════════════════════════════════════════════════════════
    # Step 1: Download & prepare data (M5 with SPIN features)
    # ═══════════════════════════════════════════════════════════
    logger.info("\n[1/3] Preparing SPIN data (M5 + path signatures + HTF context)...")
    dm = DataManager(
        pairs=pairs, years=args.years, test_months=args.test_months,
        timeframe="M5_SPIN",
    )
    pair_data = dm.prepare_all()

    if not pair_data:
        logger.error("No pair data available. Check network and retry.")
        sys.exit(1)

    first_pair = next(iter(pair_data))
    n_features = pair_data[first_pair]["n_features"]
    logger.info(f"Features: {n_features} | Pairs with data: {len(pair_data)}")

    # ═══════════════════════════════════════════════════════════
    # Step 2: Create SPIN agent
    # ═══════════════════════════════════════════════════════════
    logger.info(f"\n[2/3] Creating SPINAgent...")
    agent = SPINAgent(n_features=n_features, spin_config=SPIN_CONFIG)

    if args.load_checkpoint:
        logger.info(f"Loading checkpoint: {args.load_checkpoint}")
        agent.load_agent(args.load_checkpoint)

    # Warmup forward pass
    dummy_ms = torch.zeros(1, lookback, n_features)
    dummy_pi = torch.zeros(1, SPIN_CONFIG["position_dim"])
    dummy_pair = torch.zeros(1, dtype=torch.long)
    agent.get_policy_and_value(dummy_ms, dummy_pi, dummy_pair)

    n_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    logger.info(f"SPINAgent: {n_params:,} parameters")

    # ═══════════════════════════════════════════════════════════
    # Phase 1: HOA Pre-training (stop-loss-aware)
    # ═══════════════════════════════════════════════════════════
    if not args.skip_hoa and not args.load_checkpoint:
        logger.info("\n[Phase 1] SPIN HOA Pre-training (stop-loss-aware labels)...")

        # Compute SPIN-specific HOA labels (30K per pair to fit in RAM)
        # 8 pairs × 30K × (120×65×4 bytes) ≈ 7.5GB — fits in 31GB RAM
        all_ms, all_pi, all_labels, all_pairs = [], [], [], []

        for pair_name, data in pair_data.items():
            logger.info(f"Computing SPIN HOA labels for {pair_name.upper()}...")
            ms, pi, labels_arr, pair_idx = compute_spin_hoa_labels(
                prices=data["train_prices"],
                features=data["train_features"],
                lookback=lookback,
                pair_name=pair_name,
                horizon=hoa_config["horizon"],
                cost_threshold_mult=hoa_config["cost_threshold_mult"],
                flat_hold_pct=hoa_config.get("flat_hold_pct", 0.55),
                atr_series=data.get("atr_train"),
                h1_trend_series=data.get("h1_trend_train"),
                risk_config=risk_config,
                max_samples=30_000,
            )
            if len(labels_arr) > 0:
                all_ms.append(ms)
                all_pi.append(pi)
                all_labels.append(labels_arr)
                all_pairs.append(pair_idx)

        if not all_ms:
            logger.error("No SPIN HOA labels computed — check data")
            sys.exit(1)

        # Use HOAPretrainer machinery for training
        from nandi.training.hoa_pretrainer import HOADataset
        from torch.utils.data import DataLoader
        import torch.nn as nn

        ms_all = np.concatenate(all_ms)
        pi_all = np.concatenate(all_pi)
        labels_all = np.concatenate(all_labels)
        pairs_all = np.concatenate(all_pairs)
        logger.info(f"Total SPIN HOA samples: {len(labels_all):,}")

        # Class weights
        n_classes = 4
        class_counts = np.bincount(labels_all, minlength=n_classes).astype(np.float32)
        class_counts = np.maximum(class_counts, 1.0)
        class_weights = 1.0 / np.sqrt(class_counts)
        class_weights = class_weights / class_weights.min()
        class_weights = np.minimum(class_weights, 10.0)
        class_weights[3] = 0.0  # no CLOSE in flat HOA
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
        logger.info(f"Class weights: {class_weights}")

        dataset = HOADataset(ms_all, pi_all, labels_all, pairs_all)
        dataloader = DataLoader(dataset, batch_size=hoa_config["batch_size"],
                                shuffle=True, drop_last=True, num_workers=0)

        agent.to(device)
        agent.train()
        optimizer = torch.optim.Adam(agent.parameters(), lr=hoa_config["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=hoa_config["epochs"],
        )
        criterion = nn.CrossEntropyLoss(
            weight=class_weights_t,
            label_smoothing=hoa_config["label_smoothing"],
        )

        best_w_acc = 0.0
        best_state = None

        for epoch in range(hoa_config["epochs"]):
            total_loss = 0.0
            correct = np.zeros(4)
            total = np.zeros(4)
            n_batches = 0

            for batch_ms, batch_pi, batch_labels, batch_pairs in dataloader:
                batch_ms = batch_ms.to(device)
                batch_pi = batch_pi.to(device)
                batch_labels = batch_labels.to(device)
                batch_pairs = batch_pairs.to(device)

                logits = agent.get_classification_logits(batch_ms, batch_pi, batch_pairs)
                loss = criterion(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

                preds = logits.argmax(dim=-1)
                for c in range(4):
                    mask = batch_labels == c
                    total[c] += mask.sum().item()
                    correct[c] += (preds[mask] == c).sum().item()

            scheduler.step()

            avg_loss = total_loss / max(1, n_batches)
            per_class_acc = np.where(total > 0, correct / total, 0.0)
            active = total > 0
            w_acc = np.mean(per_class_acc[active]) if active.sum() > 0 else 0.0

            logger.info(
                f"HOA Epoch {epoch + 1:>2}/{hoa_config['epochs']} | "
                f"Loss: {avg_loss:.4f} | "
                f"H={per_class_acc[0]:.3f} L={per_class_acc[1]:.3f} "
                f"S={per_class_acc[2]:.3f} | Macro={w_acc:.3f}"
            )

            if w_acc > best_w_acc:
                best_w_acc = w_acc
                best_state = {k: v.cpu().clone() for k, v in agent.state_dict().items()}

        if best_state is not None:
            agent.load_state_dict(best_state)
            agent.to(device)

        logger.info(f"SPIN HOA complete. Best macro accuracy: {best_w_acc:.3f}")

        # Initialize CLOSE bias = HOLD bias for HOA head -> entry/exit heads
        # The hoa_head was trained, now transfer bias insight
        with torch.no_grad():
            # Entry head: set biases to reasonable starting point
            entry_last = agent.entry_head[-1]  # Linear(32, 3)
            exit_last = agent.exit_head[-1]    # Linear(32, 2)
            hoa_last = agent.hoa_head[-1]      # Linear(32, 4)

            # Copy HOA head biases to entry/exit heads
            entry_last.bias.data[0] = hoa_last.bias.data[0]  # HOLD
            entry_last.bias.data[1] = hoa_last.bias.data[1]  # LONG
            entry_last.bias.data[2] = hoa_last.bias.data[2]  # SHORT
            exit_last.bias.data[0] = hoa_last.bias.data[0]   # HOLD
            exit_last.bias.data[1] = hoa_last.bias.data[0]   # CLOSE = HOLD bias

        hoa_path = os.path.join(MODEL_DIR, "spin_agent_hoa.pt")
        agent.save_agent(hoa_path)
        logger.info(f"HOA checkpoint saved: {hoa_path}")
    else:
        logger.info("\n[Phase 1] Skipped")

    # ═══════════════════════════════════════════════════════════
    # Phase 2: PPO Training with SPIN environment
    # ═══════════════════════════════════════════════════════════
    logger.info(f"\n[Phase 2] SPIN PPO Training...")

    reward_fn = SPINReward()

    train_envs = create_spin_envs(
        pair_data, pairs, lookback, episode_length,
        risk_config=risk_config, reward_fn=reward_fn, is_eval=False,
    )
    eval_envs = create_spin_envs(
        pair_data, pairs, lookback, episode_length,
        risk_config=risk_config, reward_fn=reward_fn, is_eval=True,
    )

    if not train_envs:
        logger.error("No training environments created. Check data.")
        sys.exit(1)

    logger.info(f"Training envs: {len(train_envs)} | Eval envs: {len(eval_envs)}")

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

    rl_stats = trainer.train()

    # Save Phase 2 checkpoint
    p2_path = os.path.join(MODEL_DIR, "spin_agent_phase2.pt")
    agent.save_agent(p2_path)
    logger.info(f"Phase 2 checkpoint saved: {p2_path}")

    # ═══════════════════════════════════════════════════════════
    # Save final model and metadata
    # ═══════════════════════════════════════════════════════════
    agent.save_agent(os.path.join(MODEL_DIR, "spin_agent.pt"))

    for pair in pairs:
        if pair not in pair_data:
            continue
        data = pair_data[pair]
        pair_dir = os.path.join(MODEL_DIR, pair)
        os.makedirs(pair_dir, exist_ok=True)
        joblib.dump(data["scaler"], os.path.join(pair_dir, "scaler_spin.pkl"))
        joblib.dump(data["feature_names"],
                    os.path.join(pair_dir, "feature_names_spin.pkl"))

    joblib.dump(list(pair_data.keys()),
                os.path.join(MODEL_DIR, "trained_pairs_spin.pkl"))
    joblib.dump({
        "algo": "spin", "timeframe": "M5", "version": "v4",
        "n_features": n_features, "n_params": n_params,
    }, os.path.join(MODEL_DIR, "training_meta_spin.pkl"))

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  NANDI V4 SPIN TRAINING COMPLETE")
    logger.info(f"  Model saved to: {MODEL_DIR}")
    logger.info(f"  Parameters: {n_params:,}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
