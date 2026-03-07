"""
Train Nandi — The Conscious Trading Agent.

Downloads 20 years of EURUSD data, trains a PPO RL agent with
a Multi-Scale Fractal Attention Network encoder.

Usage:
    python train_nandi.py
    python train_nandi.py --timesteps 1000000 --years 25
"""

import argparse
import logging
import os
import sys

import numpy as np
import tensorflow as tf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nandi")

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def main():
    parser = argparse.ArgumentParser(description="Train Nandi RL Agent")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument("--test-months", type=int, default=6)
    args = parser.parse_args()

    from nandi.config import MODEL_DIR, LOOKBACK_WINDOW, TRAINING_CONFIG, PPO_CONFIG
    from nandi.data import prepare_data
    from nandi.environment import MultiEpisodeEnv, ForexTradingEnv
    from nandi.model import NandiAgent
    from nandi.trainer import NandiTrainer

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Step 1: Download & prepare data ──
    logger.info("=" * 60)
    logger.info("  NANDI — Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Downloading {args.years} years of EURUSD daily data...")

    data = prepare_data(
        lookback_window=LOOKBACK_WINDOW,
        test_months=args.test_months,
        years=args.years,
    )

    logger.info(f"Total features: {data['n_features']}")

    # ── Step 2: Create environments ──
    train_env = MultiEpisodeEnv(
        features=data["train_features"],
        prices=data["train_prices"],
        lookback=LOOKBACK_WINDOW,
        episode_length=PPO_CONFIG["rollout_length"],
    )

    eval_env = MultiEpisodeEnv(
        features=data["test_features"],
        prices=data["test_prices"],
        lookback=LOOKBACK_WINDOW,
        episode_length=len(data["test_prices"]) - LOOKBACK_WINDOW - 2,
    )

    # ── Step 3: Create agent ──
    agent = NandiAgent(n_features=data["n_features"])

    # Build the model by running a dummy forward pass
    dummy_ms = np.zeros((1, LOOKBACK_WINDOW, data["n_features"]), dtype=np.float32)
    dummy_pi = np.zeros((1, train_env.position_info_dim), dtype=np.float32)
    agent(dummy_ms, dummy_pi)

    n_params = sum(np.prod(v.shape) for v in agent.trainable_variables)
    logger.info(f"Nandi agent created: {n_params:,} trainable parameters")

    # ── Step 4: Train ──
    training_config = TRAINING_CONFIG.copy()
    training_config["total_timesteps"] = args.timesteps

    trainer = NandiTrainer(
        agent=agent,
        train_env=train_env,
        eval_env=eval_env,
        training_config=training_config,
    )

    stats = trainer.train()

    # ── Step 5: Final evaluation ──
    logger.info("\n" + "=" * 60)
    logger.info("  FINAL EVALUATION ON UNSEEN TEST DATA")
    logger.info("=" * 60)

    final_eval = trainer.evaluate(n_episodes=10)

    logger.info(f"\n  Return:   {final_eval['eval_return']:+.2f}%")
    logger.info(f"  MaxDD:    {final_eval['eval_drawdown']:.2%}")
    logger.info(f"  Trades:   {final_eval['eval_trades']:.0f}")
    logger.info(f"  Win Rate: {final_eval['eval_win_rate']:.1%}")

    # Save scaler for live trading
    import joblib
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(data["scaler"], os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(data["feature_names"], os.path.join(MODEL_DIR, "feature_names.pkl"))
    logger.info(f"Scaler and feature names saved to {MODEL_DIR}")

    logger.info("\n" + "=" * 60)
    logger.info("  NANDI TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
