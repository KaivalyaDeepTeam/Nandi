#!/usr/bin/env python3
"""
Nandi Training Dashboard — Live visualization of AEGIS training.

Panel 1: Trade Chart — Price with BUY/SELL markers, position size as color intensity
Panel 2: Equity Curve — Agent's simulated equity over time
Panel 3: Training Metrics — R̄, Actor/Critic loss, α over steps

Usage:
    python dashboard.py              # show latest training data
    python dashboard.py --live       # auto-refresh every 30s
    python dashboard.py --episode 5  # show specific episode
"""

import os
import sys
import json
import argparse
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

DASHBOARD_LOG = os.path.join("models", "nandi", "training_dashboard.jsonl")


def load_dashboard_data(log_path=None):
    """Load all training data from the JSONL dashboard log."""
    log_path = log_path or DASHBOARD_LOG
    if not os.path.exists(log_path):
        print(f"No dashboard log found at {log_path}")
        print("Start training first: caffeinate -dims .venv/bin/python train_nandi.py --algo aegis --timeframe M5 --timesteps 500000")
        return None, None

    metrics = []
    episodes = []

    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("type") == "metrics":
                    metrics.append(entry)
                elif entry.get("type") == "episode":
                    episodes.append(entry)
            except json.JSONDecodeError:
                continue

    return metrics, episodes


def plot_dashboard(metrics, episodes, episode_idx=-1, save_path=None):
    """Create 3-panel dashboard."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    fig = plt.figure(figsize=(18, 14), facecolor="#1a1a2e")
    fig.suptitle("NANDI AEGIS — Training Dashboard", fontsize=16,
                 color="white", fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1.5, 1.2], hspace=0.35,
                           left=0.07, right=0.95, top=0.94, bottom=0.05)

    # Dark theme for all axes
    ax_style = {"facecolor": "#16213e", "grid_alpha": 0.2}

    # ════════════════════════════════════════════════════════════════
    # PANEL 1: Trade Chart — Price with BUY/SELL markers
    # ════════════════════════════════════════════════════════════════
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(ax_style["facecolor"])
    ax1.set_title("Trade Chart — Latest Episode", color="white", fontsize=13, pad=10)

    if episodes:
        ep = episodes[episode_idx] if abs(episode_idx) <= len(episodes) else episodes[-1]
        trades = ep.get("trades", [])
        equity_snaps = ep.get("equity", [])

        if equity_snaps:
            steps = [e["step"] for e in equity_snaps]
            prices = [e["price"] for e in equity_snaps if e.get("price", 0) > 0]
            price_steps = [e["step"] for e in equity_snaps if e.get("price", 0) > 0]

            if prices:
                ax1.plot(price_steps, prices, color="#4cc9f0", linewidth=1.0,
                         alpha=0.8, label="Price")

                # BUY/SELL markers
                if trades:
                    buy_steps = [t["step"] for t in trades if t["action"] > 0.05]
                    buy_prices = [t["price"] for t in trades if t["action"] > 0.05]
                    buy_sizes = [abs(t["action"]) for t in trades if t["action"] > 0.05]

                    sell_steps = [t["step"] for t in trades if t["action"] < -0.05]
                    sell_prices = [t["price"] for t in trades if t["action"] < -0.05]
                    sell_sizes = [abs(t["action"]) for t in trades if t["action"] < -0.05]

                    # Size = position intensity, capped for visibility
                    if buy_steps:
                        sizes_b = np.array(buy_sizes) * 300 + 20
                        ax1.scatter(buy_steps, buy_prices, c="#00ff88",
                                    s=sizes_b, marker="^", alpha=0.8,
                                    edgecolors="white", linewidths=0.5,
                                    label=f"BUY ({len(buy_steps)})", zorder=5)

                    if sell_steps:
                        sizes_s = np.array(sell_sizes) * 300 + 20
                        ax1.scatter(sell_steps, sell_prices, c="#ff4466",
                                    s=sizes_s, marker="v", alpha=0.8,
                                    edgecolors="white", linewidths=0.5,
                                    label=f"SELL ({len(sell_steps)})", zorder=5)

                    # Profitable vs losing trades (background coloring)
                    for t in trades:
                        color = "#00ff8833" if t.get("reward", 0) > 0 else "#ff446633"
                        ax1.axvline(t["step"], color=color, linewidth=0.5, alpha=0.3)

                ax1.legend(loc="upper left", fontsize=9, facecolor="#0f3460",
                           edgecolor="gray", labelcolor="white")

        ep_info = f"Episode {ep.get('episode', '?')} | Step {ep.get('step', '?'):,} | R̄: {ep.get('reward', 0):.1f}"
        ax1.text(0.99, 0.97, ep_info, transform=ax1.transAxes,
                 ha="right", va="top", fontsize=10, color="#aaa",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f3460", alpha=0.8))
    else:
        ax1.text(0.5, 0.5, "Waiting for episode data...", transform=ax1.transAxes,
                 ha="center", va="center", fontsize=14, color="#666")

    ax1.tick_params(colors="white", labelsize=9)
    ax1.grid(True, alpha=0.15, color="white")
    ax1.set_ylabel("Price", color="white", fontsize=10)

    # ════════════════════════════════════════════════════════════════
    # PANEL 2: Equity Curve
    # ════════════════════════════════════════════════════════════════
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(ax_style["facecolor"])
    ax2.set_title("Equity Curve — $5,000 Initial", color="white", fontsize=13, pad=10)

    if episodes:
        # Plot equity from all episodes
        all_rewards = [ep.get("reward", 0) for ep in episodes]
        all_steps = [ep.get("step", 0) for ep in episodes]

        # Cumulative equity approximation from episode rewards
        cumulative_equity = [5000]
        for r in all_rewards:
            cumulative_equity.append(cumulative_equity[-1] + r)

        # Color by profit/loss
        colors = ["#00ff88" if cumulative_equity[i+1] >= cumulative_equity[i]
                  else "#ff4466" for i in range(len(cumulative_equity)-1)]

        for i in range(len(all_steps)):
            ax2.plot([all_steps[max(0, i-1)] if i > 0 else 0, all_steps[i]],
                     [cumulative_equity[i], cumulative_equity[i+1]],
                     color=colors[i], linewidth=1.5, alpha=0.8)

        # Starting balance line
        ax2.axhline(5000, color="#ffffff44", linewidth=1, linestyle="--", label="Starting $5,000")

        # Fill profit/loss
        ax2.fill_between(
            [0] + all_steps,
            cumulative_equity,
            5000,
            where=[e >= 5000 for e in cumulative_equity],
            color="#00ff88", alpha=0.1,
        )
        ax2.fill_between(
            [0] + all_steps,
            cumulative_equity,
            5000,
            where=[e < 5000 for e in cumulative_equity],
            color="#ff4466", alpha=0.1,
        )

        final_eq = cumulative_equity[-1]
        pct = (final_eq / 5000 - 1) * 100
        color = "#00ff88" if pct >= 0 else "#ff4466"
        ax2.text(0.99, 0.97, f"${final_eq:,.0f} ({pct:+.1f}%)",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=12, color=color, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f3460", alpha=0.8))

        ax2.legend(loc="upper left", fontsize=9, facecolor="#0f3460",
                   edgecolor="gray", labelcolor="white")
    else:
        ax2.text(0.5, 0.5, "Waiting for episode data...", transform=ax2.transAxes,
                 ha="center", va="center", fontsize=14, color="#666")

    ax2.tick_params(colors="white", labelsize=9)
    ax2.grid(True, alpha=0.15, color="white")
    ax2.set_ylabel("Equity ($)", color="white", fontsize=10)

    # ════════════════════════════════════════════════════════════════
    # PANEL 3: Training Metrics
    # ════════════════════════════════════════════════════════════════
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(ax_style["facecolor"])
    ax3.set_title("Training Metrics", color="white", fontsize=13, pad=10)

    if metrics:
        steps = [m["step"] for m in metrics]
        rewards = [m["reward"] for m in metrics]
        actor_loss = [m["actor_loss"] for m in metrics]
        critic_loss = [m["critic_loss"] for m in metrics]
        alphas = [m["alpha"] for m in metrics]

        ax3.plot(steps, rewards, color="#f72585", linewidth=1.5, label="R̄ (reward)", alpha=0.9)
        ax3.plot(steps, critic_loss, color="#4cc9f0", linewidth=1.0, label="Critic", alpha=0.7)

        # Alpha on secondary axis
        ax3b = ax3.twinx()
        ax3b.plot(steps, alphas, color="#ffd700", linewidth=1.0, label="α (entropy)", alpha=0.7, linestyle="--")
        ax3b.set_ylabel("α", color="#ffd700", fontsize=10)
        ax3b.tick_params(colors="#ffd700", labelsize=9)

        ax3.legend(loc="upper left", fontsize=9, facecolor="#0f3460",
                   edgecolor="gray", labelcolor="white")
        ax3b.legend(loc="upper right", fontsize=9, facecolor="#0f3460",
                    edgecolor="gray", labelcolor="white")

        # Latest stats annotation
        latest = metrics[-1]
        info = f"Step {latest['step']:,} | Gate: {latest.get('gate', 0):.2f}"
        ax3.text(0.5, 0.97, info, transform=ax3.transAxes,
                 ha="center", va="top", fontsize=10, color="#aaa",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f3460", alpha=0.8))
    else:
        ax3.text(0.5, 0.5, "Waiting for metrics...", transform=ax3.transAxes,
                 ha="center", va="center", fontsize=14, color="#666")

    ax3.tick_params(colors="white", labelsize=9)
    ax3.grid(True, alpha=0.15, color="white")
    ax3.set_ylabel("Loss / Reward", color="white", fontsize=10)
    ax3.set_xlabel("Training Step", color="white", fontsize=10)

    # Save
    out = save_path or "training_dashboard.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Dashboard saved: {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Nandi Training Dashboard")
    parser.add_argument("--live", action="store_true", help="Auto-refresh every 30s")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval (seconds)")
    parser.add_argument("--episode", type=int, default=-1, help="Episode index to show (-1=latest)")
    parser.add_argument("--output", type=str, default="training_dashboard.png", help="Output file")
    args = parser.parse_args()

    if args.live:
        print(f"Live dashboard mode — refreshing every {args.interval}s (Ctrl+C to stop)")
        while True:
            try:
                metrics, episodes = load_dashboard_data()
                if metrics or episodes:
                    plot_dashboard(metrics or [], episodes or [],
                                   episode_idx=args.episode, save_path=args.output)
                else:
                    print("No data yet, waiting...")
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped.")
                break
    else:
        metrics, episodes = load_dashboard_data()
        if metrics or episodes:
            plot_dashboard(metrics or [], episodes or [],
                           episode_idx=args.episode, save_path=args.output)
        else:
            print("No training data available yet.")


if __name__ == "__main__":
    main()
