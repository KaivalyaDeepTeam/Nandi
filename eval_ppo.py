"""Evaluate PPO checkpoint with multiple episodes for stable statistics.

Uses stochastic sampling (the learned policy) — not argmax.
Runs 10 episodes per pair with random start positions.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

import logging
logging.basicConfig(level=logging.WARNING)

from nandi.config import (
    PAIRS, TIMEFRAME_PROFILES, DQN_CONFIG, MODEL_DIR,
)
from nandi.data.manager import DataManager
from nandi.models.ppo_agent import NandiPPOAgent
from nandi.environment.discrete_env import DiscreteActionEnv
from nandi.environment.market_sim import MarketSimulator


def eval_ppo(pair_data, pairs, profile, timeframe, agent, device,
             deterministic=False, n_episodes=10):
    """Evaluate PPO agent over multiple episodes."""
    lookback = profile["lookback_bars"]
    episode_length = profile["episode_bars"]
    agent.eval()

    all_results = []

    for pair in pairs:
        if pair not in pair_data:
            continue
        data = pair_data[pair]
        features = data["test_features"]
        prices = data["test_prices"]

        pair_returns = []
        pair_trades = []
        pair_wins = []

        for ep in range(n_episodes):
            market_sim = MarketSimulator(pair_name=pair, timeframe=timeframe)
            env = DiscreteActionEnv(
                features=features, prices=prices, lookback=lookback,
                pair_name=pair, market_sim=market_sim, timeframe=timeframe,
            )

            max_start = len(features) - episode_length - lookback - 1
            if max_start <= lookback:
                start = lookback
            else:
                start = np.random.randint(lookback, max_start)

            state = env.reset(start_idx=start)
            done = False
            step_count = 0
            action_counts = np.zeros(4, dtype=int)
            trades = 0
            wins = 0

            while not done and step_count < episode_length:
                ms, pi = state
                mask = env.get_action_mask()

                action, log_prob, value, probs = agent.get_action(
                    ms, pi, env.pair_idx,
                    action_mask=mask,
                    deterministic=deterministic,
                )

                state, reward, done, info = env.step(action)
                action_counts[action] += 1
                step_count += 1

                if info.get("trade_closed", False):
                    trades += 1
                    if info.get("raw_pnl", 0.0) > 0:
                        wins += 1

            ret_pct = info.get("return_pct", 0.0)
            wr = wins / max(1, trades)
            total_a = max(1, action_counts.sum())
            hold_pct = action_counts[0] / total_a * 100

            pair_returns.append(ret_pct)
            pair_trades.append(trades)
            pair_wins.append(wr)

            print(f"  {pair} ep{ep:>2d}: ret={ret_pct:+7.2f}% "
                  f"trades={trades:>4d} WR={wr:.1%} "
                  f"hold={hold_pct:.0f}% "
                  f"H/L/S/C={action_counts[0]}/{action_counts[1]}/"
                  f"{action_counts[2]}/{action_counts[3]}")

        mean_ret = np.mean(pair_returns)
        std_ret = np.std(pair_returns)
        mean_trades = np.mean(pair_trades)
        mean_wr = np.mean(pair_wins)

        print(f"  {pair} AVG: ret={mean_ret:+.2f}% ± {std_ret:.2f}%  "
              f"trades={mean_trades:.0f}  WR={mean_wr:.1%}")
        print()

        all_results.append({
            "pair": pair,
            "returns": pair_returns,
            "mean_return": mean_ret,
            "std_return": std_ret,
            "mean_trades": mean_trades,
            "mean_wr": mean_wr,
        })

    # Overall
    all_returns = [r for res in all_results for r in res["returns"]]
    print(f"{'=' * 55}")
    mode = "DETERMINISTIC" if deterministic else "STOCHASTIC"
    print(f"  OVERALL ({mode}, {n_episodes} eps × {len(all_results)} pairs):")
    print(f"    Return: {np.mean(all_returns):+.2f}% ± {np.std(all_returns):.2f}%")
    print(f"    Median: {np.median(all_returns):+.2f}%")
    print(f"    Min/Max: {np.min(all_returns):+.2f}% / {np.max(all_returns):+.2f}%")
    for res in all_results:
        print(f"    {res['pair']}: {res['mean_return']:+.2f}% ± {res['std_return']:.2f}%  "
              f"trades={res['mean_trades']:.0f}  WR={res['mean_wr']:.1%}")
    print(f"{'=' * 55}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoint")
    parser.add_argument("--pairs", nargs="+", default=["eurusd", "gbpusd"],
                        help="Pairs to evaluate")
    parser.add_argument("--timeframe", type=str, default="M5",
                        choices=["M5", "M1", "H1", "D1"],
                        help="Trading timeframe")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint filename (e.g. ppo_agent_best.pt)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of eval episodes per pair")
    args = parser.parse_args()

    pairs = args.pairs
    timeframe = args.timeframe
    profile = TIMEFRAME_PROFILES[timeframe]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    print(f"Loading data ({timeframe})...")
    dm = DataManager(pairs=pairs, years=20, test_months=6, timeframe=timeframe)
    pair_data = dm.prepare_all()
    if not pair_data:
        print("No data available")
        return

    first_pair = next(iter(pair_data))
    n_features = pair_data[first_pair]["n_features"]

    # Try best checkpoint first, then phase2, then default
    if args.checkpoint:
        ckpt_path = os.path.join(MODEL_DIR, args.checkpoint)
    else:
        for ckpt_name in ["ppo_agent_best.pt", "ppo_agent_phase2.pt", "ppo_agent.pt"]:
            ckpt_path = os.path.join(MODEL_DIR, ckpt_name)
            if os.path.exists(ckpt_path):
                break

    print(f"Loading: {ckpt_path}")
    agent = NandiPPOAgent(n_features=n_features, dqn_config=DQN_CONFIG)
    agent.load_agent(ckpt_path)
    agent = agent.to(device)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"Agent: {n_params:,} params\n")

    n_episodes = args.episodes

    # Stochastic eval (the real policy)
    print(f"{'=' * 55}")
    print(f"  STOCHASTIC POLICY (learned distribution)")
    print(f"{'=' * 55}")
    eval_ppo(pair_data, pairs, profile, timeframe, agent, device,
             deterministic=False, n_episodes=n_episodes)

    print()

    # Deterministic eval (argmax — for comparison)
    print(f"\n{'=' * 55}")
    print(f"  DETERMINISTIC (argmax — for comparison)")
    print(f"{'=' * 55}")
    eval_ppo(pair_data, pairs, profile, timeframe, agent, device,
             deterministic=True, n_episodes=n_episodes)


if __name__ == "__main__":
    main()
