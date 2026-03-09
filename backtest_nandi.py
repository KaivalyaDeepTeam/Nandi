"""
Backtest Nandi V2 — Multi-pair portfolio backtesting on unseen data.

Uses event-driven backtester with realistic execution simulation,
Monte Carlo permutation testing, stress testing, and OOD detection.

Usage:
    python backtest_nandi.py
    python backtest_nandi.py --timeframe M5
    python backtest_nandi.py --test-months 3
    python backtest_nandi.py --pairs eurusd gbpusd --monte-carlo
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nandi")

np.random.seed(42)
torch.manual_seed(42)


def report_scalping_metrics(pair_results, pair_data, timeframe):
    """Report M5-specific metrics: trades/day, avg hold time, profit per trade."""
    if timeframe == "D1":
        return

    logger.info(f"\n{'─' * 60}")
    logger.info("  SCALPING METRICS")
    logger.info(f"{'─' * 60}")

    for pair, result in pair_results.items():
        trades = result.get("trades", [])
        if not trades:
            logger.info(f"  {pair.upper():>8s}: No trades")
            continue

        n_trades = len(trades)
        # Estimate trading days from test data length
        if pair in pair_data:
            n_bars = len(pair_data[pair]["test_prices"])
            n_days = max(1, n_bars / 288)  # 288 M5 bars per day
        else:
            n_days = 1

        trades_per_day = n_trades / n_days

        # Avg hold time (in bars -> hours for M5)
        hold_times = [t.get("hold_bars", 0) for t in trades if "hold_bars" in t]
        avg_hold_bars = np.mean(hold_times) if hold_times else 0
        avg_hold_hours = avg_hold_bars * 5 / 60  # M5 = 5 min per bar

        # Profit per trade
        profits = [t.get("pnl", 0) for t in trades if "pnl" in t]
        avg_profit = np.mean(profits) if profits else 0
        win_rate = sum(1 for p in profits if p > 0) / max(len(profits), 1)

        logger.info(
            f"  {pair.upper():>8s} | "
            f"Trades/day: {trades_per_day:.1f} | "
            f"Avg hold: {avg_hold_hours:.1f}h ({avg_hold_bars:.0f} bars) | "
            f"Avg P/L: ${avg_profit:.2f} | "
            f"WR: {win_rate:.1%}"
        )


def main():
    parser = argparse.ArgumentParser(description="Backtest Nandi V2 Portfolio")
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument("--pairs", nargs="+", default=None)
    parser.add_argument("--timeframe", type=str, default="D1",
                        choices=["D1", "M5"],
                        help="Trading timeframe (default: D1)")
    parser.add_argument("--monte-carlo", action="store_true",
                        help="Run Monte Carlo significance test")
    parser.add_argument("--stress-test", action="store_true",
                        help="Run stress tests against historical crises")
    parser.add_argument("--ood", action="store_true",
                        help="Run OOD detection on test data")
    parser.add_argument("--full", action="store_true",
                        help="Run all tests (Monte Carlo + stress + OOD)")
    args = parser.parse_args()

    if args.full:
        args.monte_carlo = True
        args.stress_test = True
        args.ood = True

    from nandi.config import MODEL_DIR, INITIAL_BALANCE, PAIRS, TIMEFRAME_PROFILES
    from nandi.data.manager import DataManager
    from nandi.models.agent import NandiAgent
    from nandi.backtest.event_engine import EventDrivenBacktester

    profile = TIMEFRAME_PROFILES.get(args.timeframe, TIMEFRAME_PROFILES["D1"])
    lookback = profile["lookback_bars"]

    # Determine which pairs have trained models
    trained_pairs_path = os.path.join(MODEL_DIR, "trained_pairs.pkl")
    if os.path.exists(trained_pairs_path):
        available_pairs = joblib.load(trained_pairs_path)
    else:
        available_pairs = PAIRS

    pairs = args.pairs or available_pairs

    # Load training metadata for encoder type
    meta_path = os.path.join(MODEL_DIR, "training_meta.pkl")
    encoder_type = "msfan"
    if os.path.exists(meta_path):
        meta = joblib.load(meta_path)
        encoder_type = meta.get("encoder", "msfan")

    # Load agents
    agents = {}
    for pair in pairs:
        pair_dir = os.path.join(MODEL_DIR, pair)
        agent_path = os.path.join(pair_dir, "agent.pt")
        fn_path = os.path.join(pair_dir, "feature_names.pkl")

        if not os.path.exists(agent_path):
            logger.warning(f"No trained model for {pair}, skipping")
            continue

        # Check per-pair encoder type
        pair_meta_path = os.path.join(pair_dir, "meta.pkl")
        pair_encoder = encoder_type
        if os.path.exists(pair_meta_path):
            pair_meta = joblib.load(pair_meta_path)
            pair_encoder = pair_meta.get("encoder", encoder_type)

        feature_names = joblib.load(fn_path)

        # Check if this was trained with AEGIS
        pair_algo = "ppo"
        if os.path.exists(pair_meta_path):
            pair_algo = pair_meta.get("algo", "ppo")

        if pair_algo == "aegis":
            from nandi.models.aegis import AEGISAgent
            agent = AEGISAgent(n_features=len(feature_names), encoder_type=pair_encoder)
        else:
            agent = NandiAgent(n_features=len(feature_names), encoder_type=pair_encoder)

        dummy_ms = torch.zeros(1, lookback, len(feature_names))
        dummy_pi = torch.zeros(1, 4)
        agent(dummy_ms, dummy_pi)

        if agent.load_agent(agent_path):
            agents[pair] = agent
            logger.info(f"Loaded {pair.upper()} agent ({pair_algo}/{pair_encoder}, {len(feature_names)} features)")
        else:
            logger.warning(f"Failed to load {pair} agent")

    if not agents:
        logger.error("No trained models found. Run train_nandi.py first.")
        sys.exit(1)

    # Prepare data
    dm = DataManager(pairs=list(agents.keys()), years=args.years,
                     test_months=args.test_months, timeframe=args.timeframe)
    pair_data = dm.prepare_all()

    # ── Event-Driven Backtest ──
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  NANDI V2 EVENT-DRIVEN BACKTEST ({args.timeframe})")
    logger.info(f"  Pairs: {', '.join(p.upper() for p in agents.keys())}")
    logger.info(f"  Test window: {args.test_months} months")
    logger.info(f"  Lookback: {lookback} bars")
    logger.info(f"{'=' * 60}")

    pair_results = {}
    portfolio_equity_curves = []
    portfolio_daily_returns = []

    for pair, agent in agents.items():
        if pair not in pair_data:
            continue

        data = pair_data[pair]
        backtester = EventDrivenBacktester(
            agent=agent,
            pair_name=pair,
            initial_balance=INITIAL_BALANCE,
        )

        result = backtester.run(
            features=data["test_features"],
            prices=data["test_prices"],
            lookback=lookback,
        )

        pair_results[pair] = result
        portfolio_equity_curves.append(result["equity_curve"])
        portfolio_daily_returns.append(result["daily_returns"])

        m = result["metrics"]
        logger.info(
            f"  {pair.upper():>8s} | "
            f"Return: {m['total_return_pct']:+7.2f}% | "
            f"Sharpe: {m['sharpe']:5.2f} | "
            f"MaxDD: {m['max_drawdown']:.2%} | "
            f"WR: {m['win_rate']:.1%} | "
            f"PF: {m['profit_factor']:.2f} | "
            f"Trades: {m['n_trades']} | "
            f"Avg R: {m['avg_r_multiple']:.3f}"
        )

    # ── Scalping Metrics (M5 only) ──
    report_scalping_metrics(pair_results, pair_data, args.timeframe)

    # ── Portfolio Aggregation ──
    if portfolio_daily_returns:
        # Equal-weight portfolio returns
        min_len = min(len(r) for r in portfolio_daily_returns)
        trimmed = [r[:min_len] for r in portfolio_daily_returns]
        portfolio_returns = np.mean(trimmed, axis=0)

        from nandi.utils.metrics import sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown
        port_equity = INITIAL_BALANCE * np.cumprod(1 + portfolio_returns)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"  PORTFOLIO RESULTS — {len(pair_results)} Pairs ({args.timeframe})")
        logger.info(f"{'=' * 60}")
        logger.info(f"  Total Return:     {(port_equity[-1] / INITIAL_BALANCE - 1) * 100:+.2f}%")
        logger.info(f"  Sharpe Ratio:     {sharpe_ratio(portfolio_returns):.2f}")
        logger.info(f"  Sortino Ratio:    {sortino_ratio(portfolio_returns):.2f}")
        logger.info(f"  Calmar Ratio:     {calmar_ratio(portfolio_returns):.2f}")
        logger.info(f"  Max Drawdown:     {max_drawdown(portfolio_returns):.2%}")

    # ── Monte Carlo Significance Test ──
    if args.monte_carlo and portfolio_daily_returns:
        logger.info(f"\n{'─' * 60}")
        logger.info("  MONTE CARLO SIGNIFICANCE TEST (1000 permutations)")
        logger.info(f"{'─' * 60}")

        from nandi.backtest.monte_carlo import MonteCarloValidator
        mc = MonteCarloValidator(n_permutations=1000)

        # Test portfolio
        mc_result = mc.test_significance(portfolio_returns)
        logger.info(f"  Portfolio Sharpe: {mc_result['real_sharpe']:.3f}")
        logger.info(f"  p-value:          {mc_result['p_value']:.4f} {mc_result['significance_level']}")
        logger.info(f"  Null mean/std:    {mc_result['null_mean']:.3f} +/- {mc_result['null_std']:.3f}")
        logger.info(f"  Significant:      {'YES' if mc_result['is_significant'] else 'NO'}")

        # Test each pair
        for pair, result in pair_results.items():
            mc_pair = mc.test_significance(result["daily_returns"])
            logger.info(
                f"  {pair.upper():>8s}: Sharpe={mc_pair['real_sharpe']:.3f}, "
                f"p={mc_pair['p_value']:.4f} {mc_pair['significance_level']}"
            )

    # ── Stress Testing ──
    if args.stress_test:
        logger.info(f"\n{'─' * 60}")
        logger.info("  STRESS TESTING — Historical Crisis Periods")
        logger.info(f"{'─' * 60}")

        from nandi.backtest.stress_test import StressTester
        stress_tester = StressTester()

        # Need full date range for stress testing
        for pair, result in pair_results.items():
            if pair in pair_data and "test_dates" in pair_data[pair]:
                dates = pair_data[pair]["test_dates"]
                if len(dates) == len(result["daily_returns"]):
                    logger.info(f"\n  {pair.upper()} Crisis Performance:")
                    crises = stress_tester.test_all_crises(
                        result["daily_returns"], dates
                    )
                    if not crises:
                        logger.info("    (no crisis periods in test window)")

    # ── OOD Detection ──
    if args.ood:
        logger.info(f"\n{'─' * 60}")
        logger.info("  OUT-OF-DISTRIBUTION DETECTION")
        logger.info(f"{'─' * 60}")

        from nandi.backtest.ood_detector import OODDetector
        ood = OODDetector()

        for pair in pair_results:
            if pair not in pair_data:
                continue

            data = pair_data[pair]
            # Fit on training data
            ood.fit(data["train_features"])

            # Score test data
            scores = ood.batch_score(data["test_features"])
            n_ood = int(np.sum(scores > ood.threshold))
            pct_ood = n_ood / len(scores) * 100

            logger.info(
                f"  {pair.upper():>8s}: "
                f"{n_ood}/{len(scores)} bars OOD ({pct_ood:.1f}%) | "
                f"Mean score: {np.mean(scores):.2f} | "
                f"Max score: {np.max(scores):.2f} | "
                f"Threshold: {ood.threshold:.2f}"
            )

    # ── Save Results ──
    os.makedirs("data", exist_ok=True)

    suffix = f"_{args.timeframe}" if args.timeframe != "D1" else ""
    for pair, result in pair_results.items():
        if result["trades"]:
            tdf = pd.DataFrame(result["trades"])
            tdf.to_csv(f"data/nandi_bt_{pair}{suffix}.csv", index=False)

    if portfolio_daily_returns:
        eq_df = pd.DataFrame({
            "portfolio_equity": port_equity,
        })
        eq_df.to_csv(f"data/nandi_portfolio_equity{suffix}.csv", index=False)

    logger.info(f"\nResults saved to data/")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    main()
