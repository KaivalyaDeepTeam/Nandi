"""Evaluate HOA/PPO model with confidence-threshold strategy.

Instead of stochastic sampling (too many random trades) or argmax (no trades),
use a threshold on action probabilities:
  - Enter LONG only when P(LONG) > threshold
  - Enter SHORT only when P(SHORT) > threshold
  - CLOSE when P(CLOSE) > exit_threshold (or when in trade and P(CLOSE) > P(HOLD))
  - Otherwise HOLD

The insight: EURUSD ep0 got +13.77% with only 61 trades (94% HOLD).
Selective, high-conviction entries are the key to profitability.
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

import logging
logging.basicConfig(level=logging.WARNING)

from nandi.config import TIMEFRAME_PROFILES, DQN_CONFIG, MODEL_DIR
from nandi.data.manager import DataManager
from nandi.models.ppo_agent import NandiPPOAgent
from nandi.environment.discrete_env import DiscreteActionEnv
from nandi.environment.market_sim import MarketSimulator

HOLD, LONG, SHORT, CLOSE = 0, 1, 2, 3


def eval_threshold(pair_data, pairs, profile, timeframe, agent, device,
                   entry_threshold=0.3, exit_threshold=0.2,
                   temperature=1.0, n_episodes=10):
    """Evaluate with confidence thresholds."""
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

        for ep in range(n_episodes):
            market_sim = MarketSimulator(pair_name=pair, timeframe=timeframe)
            env = DiscreteActionEnv(
                features=features, prices=prices, lookback=lookback,
                pair_name=pair, market_sim=market_sim, timeframe=timeframe,
            )

            max_start = len(features) - episode_length - lookback - 1
            start = (lookback if max_start <= lookback
                     else np.random.randint(lookback, max_start))

            state = env.reset(start_idx=start)
            done = False
            step_count = 0
            action_counts = np.zeros(4, dtype=int)
            trades = 0
            wins = 0

            while not done and step_count < episode_length:
                ms, pi = state
                mask = env.get_action_mask()

                # Get probabilities from the model
                with torch.no_grad():
                    ms_t = torch.tensor(ms[np.newaxis], dtype=torch.float32,
                                        device=device)
                    pi_t = torch.tensor(pi[np.newaxis], dtype=torch.float32,
                                        device=device)
                    pid_t = torch.tensor([env.pair_idx], dtype=torch.long,
                                         device=device)
                    mask_t = torch.tensor(mask[np.newaxis], dtype=torch.bool,
                                          device=device)

                    dist, value = agent.get_policy_and_value(
                        ms_t, pi_t, pid_t, action_mask=mask_t,
                    )
                    # Apply temperature
                    logits = dist.logits[0]
                    if temperature != 1.0:
                        logits = logits / temperature
                        logits[~mask_t[0]] = -1e8
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()

                # Threshold-based action selection
                in_trade = env._position_state != 0

                if in_trade:
                    # In trade: binary decision (HOLD or CLOSE)
                    if mask[CLOSE] and probs[CLOSE] > exit_threshold:
                        action = CLOSE
                    else:
                        action = HOLD
                else:
                    # Flat: check if any entry signal exceeds threshold
                    if mask[LONG] and probs[LONG] > entry_threshold:
                        if probs[LONG] >= probs[SHORT]:
                            action = LONG
                        else:
                            action = SHORT if mask[SHORT] else LONG
                    elif mask[SHORT] and probs[SHORT] > entry_threshold:
                        action = SHORT
                    else:
                        action = HOLD

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
            print(f"  {pair} ep{ep:>2d}: ret={ret_pct:+7.2f}% "
                  f"trades={trades:>4d} WR={wr:.1%} hold={hold_pct:.0f}% "
                  f"H/L/S/C={action_counts[0]}/{action_counts[1]}/"
                  f"{action_counts[2]}/{action_counts[3]}")

        mean_ret = np.mean(pair_returns)
        std_ret = np.std(pair_returns)
        print(f"  {pair} AVG: ret={mean_ret:+.2f}% ± {std_ret:.2f}%  "
              f"({n_episodes} eps)")
        print()
        all_results.append({
            "pair": pair, "returns": pair_returns,
            "mean": mean_ret, "std": std_ret,
        })

    all_rets = [r for res in all_results for r in res["returns"]]
    print(f"  OVERALL: {np.mean(all_rets):+.2f}% ± {np.std(all_rets):.2f}%  "
          f"median={np.median(all_rets):+.2f}%  "
          f"min={np.min(all_rets):+.2f}% max={np.max(all_rets):+.2f}%")
    return np.mean(all_rets)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate with confidence thresholds")
    parser.add_argument("--pairs", nargs="+", default=["eurusd", "gbpusd"],
                        help="Pairs to evaluate")
    parser.add_argument("--timeframe", type=str, default="M5",
                        choices=["M5", "M1", "H1", "D1"],
                        help="Trading timeframe")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint filename")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of eval episodes per config")
    args = parser.parse_args()

    pairs = args.pairs
    timeframe = args.timeframe
    profile = TIMEFRAME_PROFILES[timeframe]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading data ({timeframe})...")
    dm = DataManager(pairs=pairs, years=20, test_months=6, timeframe=timeframe)
    pair_data = dm.prepare_all()
    if not pair_data:
        print("No data"); return

    first_pair = next(iter(pair_data))
    n_features = pair_data[first_pair]["n_features"]

    if args.checkpoint:
        ckpt_path = os.path.join(MODEL_DIR, args.checkpoint)
    else:
        for ckpt_name in ["ppo_agent_best.pt", "ppo_agent_hoa.pt", "ppo_agent.pt"]:
            ckpt_path = os.path.join(MODEL_DIR, ckpt_name)
            if os.path.exists(ckpt_path):
                break

    print(f"Loading: {ckpt_path}")
    agent = NandiPPOAgent(n_features=n_features, dqn_config=DQN_CONFIG)
    agent.load_agent(ckpt_path)
    agent = agent.to(device)
    print(f"Agent: {sum(p.numel() for p in agent.parameters()):,} params\n")

    # Grid search over thresholds — fine grid around sweet spot
    configs = [
        # (entry_threshold, exit_threshold, temperature, label)
        (0.15, 0.10, 1.0, "t=0.15 (moderate)"),
        (0.20, 0.12, 1.0, "t=0.20"),
        (0.22, 0.14, 1.0, "t=0.22"),
        (0.25, 0.15, 1.0, "t=0.25"),
        (0.28, 0.18, 1.0, "t=0.28"),
        (0.30, 0.20, 1.0, "t=0.30 (selective)"),
        (0.20, 0.12, 0.7, "t=0.20 temp=0.7"),
        (0.25, 0.15, 0.7, "t=0.25 temp=0.7"),
    ]

    best_ret = -999
    best_cfg = None

    for entry_t, exit_t, temp, label in configs:
        print(f"\n{'=' * 60}")
        print(f"  entry>{entry_t:.2f}  exit>{exit_t:.2f}  temp={temp}  ({label})")
        print(f"{'=' * 60}")
        avg_ret = eval_threshold(
            pair_data, pairs, profile, timeframe, agent, device,
            entry_threshold=entry_t, exit_threshold=exit_t,
            temperature=temp, n_episodes=args.episodes,
        )
        if avg_ret > best_ret:
            best_ret = avg_ret
            best_cfg = (entry_t, exit_t, temp, label)

    print(f"\n{'#' * 60}")
    print(f"  BEST: entry>{best_cfg[0]:.2f} exit>{best_cfg[1]:.2f} "
          f"temp={best_cfg[2]} ({best_cfg[3]})")
    print(f"  Return: {best_ret:+.2f}%")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()
