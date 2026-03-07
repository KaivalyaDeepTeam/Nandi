"""
Backtest Nandi on unseen test data.

Runs the trained agent on held-out data it has NEVER seen during training,
simulating realistic trading with transaction costs and risk limits.

Usage:
    python backtest_nandi.py
    python backtest_nandi.py --test-months 3
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nandi")

np.random.seed(42)
tf.random.set_seed(42)


def main():
    parser = argparse.ArgumentParser(description="Backtest Nandi on unseen data")
    parser.add_argument("--test-months", type=int, default=2)
    parser.add_argument("--years", type=int, default=20)
    args = parser.parse_args()

    from nandi.config import MODEL_DIR, LOOKBACK_WINDOW, INITIAL_BALANCE, LEVERAGE
    from nandi.config import TRANSACTION_COST_BPS, RISK_LIMITS
    from nandi.data import prepare_data
    from nandi.model import NandiAgent

    # ── Load data ──
    data = prepare_data(
        lookback_window=LOOKBACK_WINDOW,
        test_months=args.test_months,
        years=args.years,
    )

    test_features = data["test_features"]
    test_prices = data["test_prices"]
    test_dates = data["test_dates"]

    # ── Load trained agent ──
    agent = NandiAgent(n_features=data["n_features"])
    dummy_ms = np.zeros((1, LOOKBACK_WINDOW, data["n_features"]), dtype=np.float32)
    dummy_pi = np.zeros((1, 4), dtype=np.float32)
    agent(dummy_ms, dummy_pi)

    if not agent.load_agent():
        logger.error("No trained model found. Run train_nandi.py first.")
        sys.exit(1)

    logger.info("Loaded trained Nandi agent")

    # ── Run backtest ──
    logger.info(f"\nBacktesting on {len(test_prices)} unseen days...")
    logger.info(f"Period: {test_dates[0].date()} → {test_dates[-1].date()}")

    equity = INITIAL_BALANCE
    peak_equity = INITIAL_BALANCE
    position = 0.0
    trades = []

    for i in range(LOOKBACK_WINDOW, len(test_features) - 1):
        # Get state
        market_state = test_features[i - LOOKBACK_WINDOW:i]
        dd = (peak_equity - equity) / peak_equity
        position_info = np.array([
            position,
            equity / INITIAL_BALANCE - 1,
            dd,
            float(np.std(test_features[max(0, i - 10):i, 0])) if i > 10 else 1.0,
        ], dtype=np.float32)

        # Get action from agent
        action, _, _, uncertainty = agent.get_action(
            market_state, position_info, deterministic=True
        )

        # Uncertainty gating: reduce position if uncertain
        if uncertainty > 0.7:
            action *= 0.3

        # Risk limit: scale down at drawdown threshold
        if dd > RISK_LIMITS["scale_down_threshold"]:
            action *= 0.5

        # Force flat at max drawdown
        if dd > RISK_LIMITS["max_drawdown"]:
            action = 0.0

        action = float(np.clip(action, -1.0, 1.0))

        # Simulate
        price_now = test_prices[i]
        price_next = test_prices[i + 1]
        market_ret = (price_next - price_now) / price_now

        # P&L
        pnl = position * market_ret * equity * LEVERAGE
        cost = abs(action - position) * equity * TRANSACTION_COST_BPS / 10000

        equity += pnl - cost
        peak_equity = max(peak_equity, equity)

        # Track position changes as trades
        if abs(action - position) > 0.05:
            trades.append({
                "date": test_dates[i],
                "action": "LONG" if action > 0.1 else ("SHORT" if action < -0.1 else "FLAT"),
                "position": round(action, 3),
                "price": price_now,
                "pnl": round(pnl - cost, 2),
                "equity": round(equity, 2),
                "uncertainty": round(uncertainty, 3),
                "drawdown": round(dd, 4),
            })

        position = action

    # ── Results ──
    total_return = (equity / INITIAL_BALANCE - 1) * 100
    max_dd = (peak_equity - min(t["equity"] for t in trades)) / peak_equity * 100 if trades else 0

    tdf = pd.DataFrame(trades)

    if len(tdf) == 0:
        logger.info("No trades generated.")
        return

    profitable = tdf[tdf["pnl"] > 0]
    losing = tdf[tdf["pnl"] <= 0]
    n_trades = len(tdf)
    n_wins = len(profitable)
    n_losses = len(losing)
    win_rate = n_wins / n_trades * 100

    avg_win = profitable["pnl"].mean() if n_wins > 0 else 0
    avg_loss = losing["pnl"].mean() if n_losses > 0 else 0
    gross_profit = profitable["pnl"].sum() if n_wins > 0 else 0
    gross_loss = abs(losing["pnl"].sum()) if n_losses > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe ratio (daily)
    daily_returns = tdf["pnl"].values / INITIAL_BALANCE
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(252)

    # Annualized return
    test_days = len(test_prices) - LOOKBACK_WINDOW
    annual_return = ((equity / INITIAL_BALANCE) ** (252 / max(test_days, 1)) - 1) * 100

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  NANDI BACKTEST RESULTS — EURUSD (UNSEEN DATA)")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Test period:      {test_dates[LOOKBACK_WINDOW].date()} → {test_dates[-1].date()}")
    logger.info(f"  Start balance:    ${INITIAL_BALANCE:,.2f}")
    logger.info(f"  End balance:      ${equity:,.2f}")
    logger.info(f"{'─' * 60}")
    logger.info(f"  Total trades:     {n_trades}")
    logger.info(f"  Win / Loss:       {n_wins} / {n_losses}")
    logger.info(f"  Win rate:         {win_rate:.1f}%")
    logger.info(f"{'─' * 60}")
    logger.info(f"  Total return:     {total_return:+.2f}%")
    logger.info(f"  Annualized (est): {annual_return:+.1f}%")
    logger.info(f"  Sharpe ratio:     {sharpe:.2f}")
    logger.info(f"  Profit factor:    {profit_factor:.2f}")
    logger.info(f"  Max drawdown:     {max_dd:.2f}%")
    logger.info(f"{'─' * 60}")
    logger.info(f"  Avg win:          ${avg_win:+,.2f}")
    logger.info(f"  Avg loss:         ${avg_loss:+,.2f}")
    logger.info(f"{'=' * 60}")

    # Save trades
    os.makedirs("data", exist_ok=True)
    tdf.to_csv("data/nandi_backtest.csv", index=False)
    logger.info(f"Trades saved to data/nandi_backtest.csv")


if __name__ == "__main__":
    main()
