"""Evaluate SPIN checkpoint with multiple episodes for stable statistics.

Uses stochastic sampling (learned policy) on SPIN environment with
hard-wired risk management (ATR stop-loss, max hold, cooldown, trend filter).
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

import logging
logging.basicConfig(level=logging.WARNING)

from nandi.config import (
    PAIRS, SPIN_CONFIG, SPIN_RISK_CONFIG, MODEL_DIR,
)
from nandi.data.manager import DataManager
from nandi.models.spin_agent import SPINAgent
from nandi.environment.spin_env import SPINTradingEnv
from nandi.environment.spin_reward import SPINReward


def eval_spin(pair_data, pairs, agent, device,
              deterministic=False, n_episodes=10):
    """Evaluate SPIN agent over multiple episodes."""
    lookback = SPIN_CONFIG["lookback_bars"]
    episode_length = 2016
    risk_config = SPIN_RISK_CONFIG
    agent.eval()

    all_results = []

    for pair in pairs:
        if pair not in pair_data:
            continue
        data = pair_data[pair]
        features = data["test_features"]
        prices = data["test_prices"]
        atr = data.get("atr_test")
        h1_trend = data.get("h1_trend_test")

        pair_returns = []
        pair_trades = []
        pair_wins = []
        pair_sl_hits = []
        pair_pf = []

        for ep in range(n_episodes):
            reward_fn = SPINReward()
            env = SPINTradingEnv(
                features=features, prices=prices, lookback=lookback,
                pair_name=pair, timeframe="M5",
                atr_series=atr, h1_trend_series=h1_trend,
                risk_config=risk_config, reward_fn=reward_fn,
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
            sl_hits = 0
            gross_profit = 0.0
            gross_loss = 0.0

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
                    nr = info.get("net_return", 0.0)
                    if nr > 0:
                        wins += 1
                        gross_profit += nr
                    else:
                        gross_loss += abs(nr)
                    if info.get("stop_loss_hit", False):
                        sl_hits += 1

            ret_pct = info.get("return_pct", 0.0)
            wr = wins / max(1, trades)
            pf = gross_profit / max(gross_loss, 1e-8)
            total_a = max(1, action_counts.sum())
            hold_pct = action_counts[0] / total_a * 100

            pair_returns.append(ret_pct)
            pair_trades.append(trades)
            pair_wins.append(wr)
            pair_sl_hits.append(sl_hits)
            pair_pf.append(pf)

            print(f"  {pair} ep{ep:>2d}: ret={ret_pct:+7.2f}% "
                  f"trades={trades:>3d} WR={wr:.1%} PF={pf:.2f} "
                  f"SL={sl_hits:>2d} hold={hold_pct:.0f}% "
                  f"H/L/S/C={action_counts[0]}/{action_counts[1]}/"
                  f"{action_counts[2]}/{action_counts[3]}")

        mean_ret = np.mean(pair_returns)
        std_ret = np.std(pair_returns)
        mean_trades = np.mean(pair_trades)
        mean_wr = np.mean(pair_wins)
        mean_pf = np.mean(pair_pf)
        mean_sl = np.mean(pair_sl_hits)

        print(f"  {pair} AVG: ret={mean_ret:+.2f}% +/- {std_ret:.2f}%  "
              f"trades={mean_trades:.0f}  WR={mean_wr:.1%}  PF={mean_pf:.2f}  "
              f"SL/ep={mean_sl:.1f}")
        print()

        all_results.append({
            "pair": pair,
            "returns": pair_returns,
            "mean_return": mean_ret,
            "std_return": std_ret,
            "mean_trades": mean_trades,
            "mean_wr": mean_wr,
            "mean_pf": mean_pf,
            "mean_sl": mean_sl,
        })

    # Overall
    all_returns = [r for res in all_results for r in res["returns"]]
    print(f"{'=' * 65}")
    mode = "DETERMINISTIC" if deterministic else "STOCHASTIC"
    print(f"  SPIN OVERALL ({mode}, {n_episodes} eps x {len(all_results)} pairs):")
    print(f"    Return: {np.mean(all_returns):+.2f}% +/- {np.std(all_returns):.2f}%")
    print(f"    Median: {np.median(all_returns):+.2f}%")
    print(f"    Min/Max: {np.min(all_returns):+.2f}% / {np.max(all_returns):+.2f}%")

    all_wr = np.mean([r["mean_wr"] for r in all_results])
    all_pf = np.mean([r["mean_pf"] for r in all_results])
    all_trades = np.mean([r["mean_trades"] for r in all_results])
    print(f"    Avg WR: {all_wr:.1%}  Avg PF: {all_pf:.2f}  Avg trades/ep: {all_trades:.0f}")

    for res in all_results:
        print(f"    {res['pair']}: {res['mean_return']:+.2f}% +/- {res['std_return']:.2f}%  "
              f"trades={res['mean_trades']:.0f}  WR={res['mean_wr']:.1%}  PF={res['mean_pf']:.2f}")
    print(f"{'=' * 65}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate SPIN checkpoint")
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="Pairs to evaluate (default: all 8)")
    parser.add_argument("--timeframe", type=str, default="M5",
                        choices=["M5"],
                        help="Trading timeframe (M5 only)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint filename (e.g. spin_agent_best.pt)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of eval episodes per pair")
    args = parser.parse_args()

    pairs = args.pairs or PAIRS
    device = get_device()

    # Load data
    print(f"Loading SPIN data (M5 + path signatures + HTF)...")
    dm = DataManager(pairs=pairs, years=20, test_months=6, timeframe="M5_SPIN")
    pair_data = dm.prepare_all()
    if not pair_data:
        print("No data available")
        return

    first_pair = next(iter(pair_data))
    n_features = pair_data[first_pair]["n_features"]

    # Try best checkpoint first
    if args.checkpoint:
        ckpt_path = os.path.join(MODEL_DIR, args.checkpoint)
    else:
        for ckpt_name in ["spin_agent_best.pt", "spin_agent_phase2.pt", "spin_agent.pt"]:
            ckpt_path = os.path.join(MODEL_DIR, ckpt_name)
            if os.path.exists(ckpt_path):
                break

    print(f"Loading: {ckpt_path}")
    agent = SPINAgent(n_features=n_features, spin_config=SPIN_CONFIG)
    agent.load_agent(ckpt_path)
    agent = agent.to(device)
    n_params = sum(p.numel() for p in agent.parameters())
    print(f"SPINAgent: {n_params:,} params\n")

    n_episodes = args.episodes

    # Stochastic eval
    print(f"{'=' * 65}")
    print(f"  STOCHASTIC POLICY (learned distribution)")
    print(f"{'=' * 65}")
    eval_spin(pair_data, pairs, agent, device,
              deterministic=False, n_episodes=n_episodes)

    print()

    # Deterministic eval
    print(f"\n{'=' * 65}")
    print(f"  DETERMINISTIC (argmax — for comparison)")
    print(f"{'=' * 65}")
    eval_spin(pair_data, pairs, agent, device,
              deterministic=True, n_episodes=n_episodes)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


if __name__ == "__main__":
    main()
