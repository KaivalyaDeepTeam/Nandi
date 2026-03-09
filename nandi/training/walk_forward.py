"""
Walk-Forward Training Orchestrator — rolling window retraining.

Phase 3 implementation: monthly rolling retraining with ensemble.
"""

import os
import logging

import numpy as np
import torch

from nandi.config import WALK_FORWARD_CONFIG, MODEL_DIR, LOOKBACK_WINDOW, PPO_CONFIG
from nandi.models.agent import NandiAgent
from nandi.environment.single_pair_env import MultiEpisodeEnv
from nandi.training.ppo_trainer import NandiTrainer

logger = logging.getLogger(__name__)


class WalkForwardOrchestrator:
    """Rolling window retrain orchestrator.

    For each window:
        1. Train on train_days of data
        2. Validate on val_days (for alpha weight calibration)
        3. Test on test_days (for OOS metrics)
        4. Step forward by step_days and repeat
    """

    def __init__(self, config=None):
        self.config = config or WALK_FORWARD_CONFIG
        self.checkpoints = []  # list of (window_id, model_path, val_metrics)

    def run(self, pair_name, features, prices, n_features):
        """Run walk-forward for a single pair.

        Args:
            pair_name: string identifier.
            features: (N, n_features) full feature array.
            prices: (N,) full price array.
            n_features: number of features.

        Returns:
            list of per-window results.
        """
        train_days = self.config["train_days"]
        val_days = self.config["val_days"]
        test_days = self.config["test_days"]
        step_days = self.config["step_days"]
        timesteps = self.config["timesteps_per_window"]

        total_needed = train_days + val_days + test_days
        n_windows = max(1, (len(features) - total_needed) // step_days + 1)

        logger.info(
            f"[{pair_name}] Walk-forward: {n_windows} windows, "
            f"train={train_days}d, val={val_days}d, test={test_days}d, step={step_days}d"
        )

        results = []

        for w in range(n_windows):
            start = w * step_days
            train_end = start + train_days
            val_end = train_end + val_days
            test_end = val_end + test_days

            if test_end > len(features):
                break

            train_feat = features[start:train_end]
            train_prices = prices[start:train_end]
            val_feat = features[train_end:val_end]
            val_prices = prices[train_end:val_end]

            # Create agent and train
            agent = NandiAgent(n_features=n_features)

            train_env = MultiEpisodeEnv(
                train_feat, train_prices,
                lookback=LOOKBACK_WINDOW,
                episode_length=PPO_CONFIG["rollout_length"],
                pair_name=pair_name,
            )
            eval_env = MultiEpisodeEnv(
                val_feat, val_prices,
                lookback=LOOKBACK_WINDOW,
                episode_length=len(val_prices) - LOOKBACK_WINDOW - 2,
                pair_name=pair_name,
            )

            trainer = NandiTrainer(
                agent=agent, train_env=train_env, eval_env=eval_env,
                training_config={"total_timesteps": timesteps, "eval_interval": 10_000,
                                 "save_interval": timesteps, "n_eval_episodes": 3, "seed": 42},
            )

            stats = trainer.train()
            val_metrics = trainer.evaluate(n_episodes=5)

            # Save checkpoint
            ckpt_path = os.path.join(
                MODEL_DIR, f"wf_{pair_name}_w{w}.pt"
            )
            agent.save_agent(ckpt_path)
            self.checkpoints.append((w, ckpt_path, val_metrics))

            results.append({
                "window": w,
                "val_metrics": val_metrics,
                "checkpoint": ckpt_path,
            })

            logger.info(
                f"[{pair_name}] Window {w}: val_return={val_metrics['eval_return']:.2f}%"
            )

        return results

    def get_ensemble_agent(self, n_models=3, n_features=None):
        """Average weights of last N checkpoints for Polyak-style ensemble."""
        if len(self.checkpoints) < 1:
            logger.warning("No checkpoints available for ensemble")
            return None

        n_models = min(n_models, len(self.checkpoints))
        recent = self.checkpoints[-n_models:]

        agents = []
        for _, path, _ in recent:
            agent = NandiAgent(n_features=n_features)
            if agent.load_agent(path):
                agents.append(agent)

        if not agents:
            return None

        # Average parameters
        avg_state = {}
        for key in agents[0].state_dict():
            tensors = [a.state_dict()[key].float() for a in agents]
            avg_state[key] = torch.stack(tensors).mean(0)

        ensemble = NandiAgent(n_features=n_features)
        ensemble.load_state_dict(avg_state)
        logger.info(f"Created ensemble from {len(agents)} checkpoints")
        return ensemble
