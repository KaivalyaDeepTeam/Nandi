"""
Walk-forward backtesting engine.

Usage:
    python backtest.py --symbol EURUSD
    python backtest.py --symbol EURUSD --bars 5000
"""

import argparse
import logging
import sys
import os

import numpy as np
import pandas as pd

from config.settings import (
    HISTORY_BARS, PREDICTION_HORIZON, ENSEMBLE_CONFIG,
    RISK_CONFIG, LSTM_CONFIG,
)
from src.mt5_connector import MT5Connector
from src.feature_engineer import add_all_features, prepare_features
from src.models.ensemble import EnsemblePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def simulate_trade(df, entry_idx, horizon, signal, sl_pips, tp_pips):
    """Simulate a trade checking each candle for SL/TP hit."""
    entry_price = df.iloc[entry_idx]["close"]
    point = 0.0001

    sl_dist = sl_pips * point
    tp_dist = tp_pips * point

    if signal == "BUY":
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
    else:
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist

    for j in range(1, min(horizon + 1, len(df) - entry_idx)):
        candle = df.iloc[entry_idx + j]
        if signal == "BUY":
            if candle["low"] <= sl_price:
                return -sl_pips, "SL"
            if candle["high"] >= tp_price:
                return tp_pips, "TP"
        else:
            if candle["high"] >= sl_price:
                return -sl_pips, "SL"
            if candle["low"] <= tp_price:
                return tp_pips, "TP"

    exit_price = df.iloc[min(entry_idx + horizon, len(df) - 1)]["close"]
    if signal == "BUY":
        pips = (exit_price - entry_price) / point
    else:
        pips = (entry_price - exit_price) / point
    return round(pips, 1), "TIMEOUT"


def backtest(symbol: str, bars: int, use_mt5: bool = True):
    """Run walk-forward backtest."""

    if use_mt5:
        connector = MT5Connector()
        if not connector.connect():
            logger.error("Cannot connect to MT5")
            sys.exit(1)
        df = connector.get_historical_data(symbol, "M5", bars)
        connector.disconnect()
    else:
        data_path = f"data/{symbol}_history.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path, index_col="time", parse_dates=True)
        else:
            logger.error(f"No data at {data_path}")
            sys.exit(1)

    logger.info(f"Data: {len(df)} bars | {df.index[0]} to {df.index[-1]}")

    # Prepare all features
    X, y, feature_names, df_feat = prepare_features(df, PREDICTION_HORIZON)
    logger.info(f"Features: {len(feature_names)} | Samples: {len(X)} | Target balance: {y.mean():.2%} bullish")

    # Use pre-trained model from train.py (more realistic)
    ensemble = EnsemblePredictor()
    if ensemble.load(symbol):
        logger.info("Using pre-trained model from train.py")
        # Use last 30% of data as out-of-sample test
        split = int(len(X) * 0.7)
        X_test = X[split:]
        y_test = y[split:]
    else:
        # Fallback: train a new model
        logger.info("No pre-trained model found, training new one...")
        split = int(len(X) * 0.7)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        ensemble.train(X_train, y_train, feature_names)

    logger.info(f"Test period: {len(X_test)} samples")

    # Backtest on test period
    logger.info("\nSimulating trades on test period...")

    seq_len = LSTM_CONFIG["sequence_length"]
    sl_pips = RISK_CONFIG["stop_loss_pips"]
    tp_pips = RISK_CONFIG["take_profit_pips"]
    risk_pct = RISK_CONFIG["max_risk_per_trade"]
    pip_value_per_lot = 10.0

    balance = 5000.0
    initial_balance = balance
    trades = []
    cooldown = 0

    for i in range(seq_len, len(X_test) - PREDICTION_HORIZON):
        if cooldown > 0:
            cooldown -= 1
            continue

        X_window = X_test[:i + 1]
        try:
            # Get raw advanced features for filtering
            feat_idx = split + i
            row = df_feat.iloc[feat_idx]
            raw_features = {
                "market_quality": row.get("market_quality", 0.5),
                "regime": row.get("regime", -1),
                "hurst": row.get("hurst", 0.5),
                "entropy": row.get("entropy", 1.0),
            }
            prediction = ensemble.predict_latest(
                ensemble.scaler.transform(X_window),
                raw_features
            )
        except Exception:
            continue

        if not prediction["should_trade"]:
            continue

        entry_idx = split + i

        # Apply trade filters (session, trend, pattern)
        from src.trade_filters import apply_all_filters
        hour = df_feat.index[entry_idx].hour if hasattr(df_feat.index, 'hour') else 12
        filter_result = apply_all_filters(
            prediction["signal"], prediction,
            df_feat.iloc[max(0, entry_idx-60):entry_idx+1], hour
        )
        if not filter_result["filters_pass"]:
            continue

        # Adaptive SL/TP based on ATR
        atr_val = df_feat.iloc[entry_idx]["atr"] if "atr" in df_feat.columns else 0.0005
        adaptive_sl = atr_val / 0.0001 * 1.5  # 1.5 ATR in pips
        adaptive_tp = atr_val / 0.0001 * 2.5  # 2.5 ATR in pips
        adaptive_sl = max(5, min(adaptive_sl, 25))  # clamp 5-25 pips
        adaptive_tp = max(8, min(adaptive_tp, 50))   # clamp 8-50 pips

        # Position sizing
        risk_amount = balance * risk_pct
        lot_size = risk_amount / (adaptive_sl * pip_value_per_lot)
        lot_size = max(0.01, round(lot_size, 2))

        # Simulate with adaptive levels
        pips, exit_type = simulate_trade(
            df_feat, entry_idx, PREDICTION_HORIZON,
            prediction["signal"], adaptive_sl, adaptive_tp
        )

        pnl = pips * pip_value_per_lot * lot_size
        balance += pnl

        trades.append({
            "time": df_feat.index[entry_idx],
            "signal": prediction["signal"],
            "confidence": round(prediction["confidence"], 3),
            "agree": prediction["models_agree"],
            "pips": pips,
            "lot": lot_size,
            "pnl": round(pnl, 2),
            "exit": exit_type,
            "balance": round(balance, 2),
        })

        cooldown = PREDICTION_HORIZON

        if balance <= 500:
            logger.warning("Account critically low, stopping.")
            break

    # Print results
    n = len(trades)
    if n == 0:
        logger.info("No trades generated.")
        return

    tdf = pd.DataFrame(trades)
    wins = len(tdf[tdf["pnl"] > 0])
    losses = len(tdf[tdf["pnl"] <= 0])
    total_pnl = balance - initial_balance
    win_rate = wins / n * 100
    avg_win = tdf[tdf["pnl"] > 0]["pnl"].mean() if wins > 0 else 0
    avg_loss = tdf[tdf["pnl"] <= 0]["pnl"].mean() if losses > 0 else 0
    max_dd = (tdf["balance"].cummax() - tdf["balance"]).max()
    tp_count = len(tdf[tdf["exit"] == "TP"])
    sl_count = len(tdf[tdf["exit"] == "SL"])
    to_count = len(tdf[tdf["exit"] == "TIMEOUT"])

    gross_profit = tdf[tdf["pnl"] > 0]["pnl"].sum()
    gross_loss = abs(tdf[tdf["pnl"] < 0]["pnl"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Annualized return estimate
    test_bars = len(X_test)
    test_days = test_bars * 5 / (24 * 60 / 5)  # approximate trading days
    if test_days > 0:
        daily_return = (balance / initial_balance) ** (1 / max(test_days, 1)) - 1
        annual_return = ((1 + daily_return) ** 252 - 1) * 100
    else:
        annual_return = 0

    logger.info(f"\n{'='*60}")
    logger.info(f"  BACKTEST RESULTS — {symbol}")
    logger.info(f"{'='*60}")
    logger.info(f"  Period:           {tdf['time'].iloc[0]} → {tdf['time'].iloc[-1]}")
    logger.info(f"  Start balance:    ${initial_balance:,.2f}")
    logger.info(f"  End balance:      ${balance:,.2f}")
    logger.info(f"  Peak balance:     ${tdf['balance'].max():,.2f}")
    logger.info(f"{'─'*60}")
    logger.info(f"  Total trades:     {n}")
    logger.info(f"  Win / Loss:       {wins} / {losses}")
    logger.info(f"  Win rate:         {win_rate:.1f}%")
    logger.info(f"  TP / SL / Timeout:{tp_count} / {sl_count} / {to_count}")
    logger.info(f"{'─'*60}")
    logger.info(f"  Total P&L:        ${total_pnl:+,.2f}")
    logger.info(f"  Avg win:          ${avg_win:+,.2f}")
    logger.info(f"  Avg loss:         ${avg_loss:+,.2f}")
    logger.info(f"  Profit factor:    {pf:.2f}")
    logger.info(f"  Max drawdown:     ${max_dd:,.2f}")
    logger.info(f"{'─'*60}")
    logger.info(f"  Return:           {(total_pnl/initial_balance)*100:+.2f}%")
    logger.info(f"  Annualized (est): {annual_return:+.1f}%")
    logger.info(f"{'='*60}")

    os.makedirs("data", exist_ok=True)
    tdf.to_csv(f"data/backtest_{symbol}.csv", index=False)
    logger.info(f"Trades saved to data/backtest_{symbol}.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--bars", type=int, default=HISTORY_BARS)
    parser.add_argument("--no-mt5", action="store_true")
    args = parser.parse_args()
    backtest(args.symbol, args.bars, use_mt5=not args.no_mt5)


if __name__ == "__main__":
    main()
