"""Quick eval script to test DQN checkpoint with different strategies.

Runs independently of the main training process — loads checkpoint from disk.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

import logging
logging.basicConfig(level=logging.WARNING)  # suppress data loading noise

from nandi.config import (
    PAIRS, PAIR_TO_IDX, TIMEFRAME_PROFILES, DQN_CONFIG,
    INITIAL_BALANCE, MODEL_DIR,
)
from nandi.data.manager import DataManager
from nandi.models.dqn_agent import NandiDQNAgent
from nandi.environment.discrete_env import DiscreteActionEnv
from nandi.environment.market_sim import MarketSimulator


def eval_checkpoint(pair_data, pairs, profile, timeframe,
                    agent, device, strategy="argmax",
                    temperature=0.1, n_episodes=3):
    """Evaluate checkpoint with a specific action strategy."""
    lookback = profile["lookback_bars"]
    episode_length = profile["episode_bars"]

    if strategy == "noisy":
        agent.train()
        agent.reset_noise()
    else:
        agent.eval()

    all_returns = []
    all_trades = []
    all_wr = []
    all_actions = np.zeros(4)
    q_spreads = []
    q_vals_all = []

    for pair in pairs:
        if pair not in pair_data:
            continue
        data = pair_data[pair]
        features = data["test_features"]
        prices = data["test_prices"]

        for ep in range(n_episodes):
            market_sim = MarketSimulator(pair_name=pair, timeframe=timeframe)
            env = DiscreteActionEnv(
                features=features, prices=prices,
                lookback=lookback,
                pair_name=pair, market_sim=market_sim, timeframe=timeframe,
            )

            max_start = len(features) - episode_length - lookback - 1
            if max_start <= lookback:
                start = lookback
            else:
                start = np.random.randint(lookback, max_start)

            state = env.reset(start_idx=start)
            done = False
            ep_actions = []
            step_count = 0

            while not done and step_count < episode_length:
                ms, pi = state
                mask = env.get_action_mask()

                with torch.no_grad():
                    _, q_values = agent.get_action(
                        ms, pi, env.pair_idx, action_mask=mask,
                    )

                # Track Q-value diagnostics
                valid_q = q_values[mask]
                if len(valid_q) > 1:
                    q_spreads.append(valid_q.max() - valid_q.min())
                    q_vals_all.append(q_values.copy())

                # Action selection
                if strategy == "argmax":
                    q_masked = q_values.copy()
                    q_masked[~mask] = -1e8
                    action = int(np.argmax(q_masked))

                elif strategy == "softmax":
                    q_masked = q_values.copy()
                    q_masked[~mask] = -1e8
                    q_shifted = (q_masked - q_masked[mask].max()) / max(temperature, 1e-6)
                    exp_q = np.exp(np.clip(q_shifted, -20, 0))
                    exp_q[~mask] = 0.0
                    probs = exp_q / (exp_q.sum() + 1e-10)
                    action = int(np.random.choice(4, p=probs))

                elif strategy == "noisy":
                    q_masked = q_values.copy()
                    q_masked[~mask] = -1e8
                    action = int(np.argmax(q_masked))

                else:  # random
                    action = int(np.random.choice(np.where(mask)[0]))

                ep_actions.append(action)
                state, _, done, info = env.step(action)
                step_count += 1

            ret = info.get("return_pct", 0)
            trades = info.get("total_trades", 0)
            wr = info.get("win_rate", 0)
            dd = info.get("drawdown", 0)

            all_returns.append(ret)
            all_trades.append(trades)
            all_wr.append(wr)

            act_counts = np.bincount(ep_actions, minlength=4)
            all_actions += act_counts

            print(f"  {pair} ep{ep}: Ret={ret:+.2f}% Trades={trades} "
                  f"WR={wr:.1%} DD={dd:.2%} "
                  f"H/L/S/C: {act_counts[0]}/{act_counts[1]}/{act_counts[2]}/{act_counts[3]}")

    total_a = max(1, all_actions.sum())
    q_spread_mean = np.mean(q_spreads) if q_spreads else 0
    q_spread_p50 = np.median(q_spreads) if q_spreads else 0
    q_spread_p90 = np.percentile(q_spreads, 90) if q_spreads else 0

    # Q-value diagnostics
    if q_vals_all:
        q_arr = np.array(q_vals_all[-1000:])  # last 1000 observations
        print(f"\n  Q-value diagnostics (last 1000 steps):")
        print(f"    Mean Q per action: HOLD={q_arr[:,0].mean():.4f} "
              f"LONG={q_arr[:,1].mean():.4f} "
              f"SHORT={q_arr[:,2].mean():.4f} "
              f"CLOSE={q_arr[:,3].mean():.4f}")
        print(f"    Q spread: mean={q_spread_mean:.4f} "
              f"median={q_spread_p50:.4f} p90={q_spread_p90:.4f}")

    print(f"\n  {'─'*50}")
    print(f"  RESULT ({strategy}, temp={temperature}):")
    print(f"    Return: {np.mean(all_returns):+.2f}% ± {np.std(all_returns):.2f}%")
    print(f"    Trades: {np.mean(all_trades):.0f} | WR: {np.mean(all_wr):.1%}")
    print(f"    Actions: H={all_actions[0]/total_a*100:.0f}% "
          f"L={all_actions[1]/total_a*100:.0f}% "
          f"S={all_actions[2]/total_a*100:.0f}% "
          f"C={all_actions[3]/total_a*100:.0f}%")
    print(f"  {'─'*50}\n")


def main():
    pairs = ["eurusd", "gbpusd"]
    timeframe = "M5"
    profile = TIMEFRAME_PROFILES[timeframe]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    print("Loading data...")
    dm = DataManager(pairs=pairs, years=20, test_months=6, timeframe=timeframe)
    pair_data = dm.prepare_all()

    if not pair_data:
        print("No data available")
        return

    first_pair = next(iter(pair_data))
    n_features = pair_data[first_pair]["n_features"]

    # Load agent
    ckpt_path = os.path.join(MODEL_DIR, "dqn_agent.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    print(f"  (modified: {os.path.getmtime(ckpt_path):.0f})")

    agent = NandiDQNAgent(n_features=n_features, dqn_config=DQN_CONFIG)
    agent.load_agent(ckpt_path)
    agent = agent.to(device)

    n_params = sum(p.numel() for p in agent.parameters())
    print(f"Agent: {n_params:,} params, {n_features} features")

    # Run evaluations
    strategies = [
        ("argmax", 0.0),
        ("softmax", 0.1),
        ("softmax", 0.01),
        ("noisy", 0.0),
        ("random", 0.0),
    ]

    for strat, temp in strategies:
        print(f"\n{'='*55}")
        print(f"  Strategy: {strat} (temp={temp})")
        print(f"{'='*55}")
        eval_checkpoint(
            pair_data, pairs, profile, timeframe,
            agent, device,
            strategy=strat, temperature=temp, n_episodes=3,
        )


if __name__ == "__main__":
    main()
