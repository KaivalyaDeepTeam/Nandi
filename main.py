"""
Main trading loop - runs the forex prediction bot in real-time.

Usage:
    python main.py --symbol EURUSD
    python main.py --symbol EURUSD --retrain
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timedelta

from config.settings import (
    PREDICTION_HORIZON, RETRAIN_INTERVAL, LSTM_CONFIG
)
from src.mt5_connector import MT5Connector
from src.feature_engineer import add_all_features, prepare_features
from src.models.ensemble import EnsemblePredictor
from src.risk_manager import RiskManager
from src.trade_executor import TradeExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RUNNING = True


def signal_handler(sig, frame):
    global RUNNING
    logger.info("Shutdown signal received...")
    RUNNING = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def main():
    parser = argparse.ArgumentParser(description="Forex Prediction Trading Bot")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Trading symbol")
    parser.add_argument("--retrain", action="store_true", help="Retrain model before starting")
    parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds (default: 300 = 5min)")
    args = parser.parse_args()

    symbol = args.symbol
    check_interval = args.interval

    # Initialize components
    connector = MT5Connector()
    risk_manager = RiskManager()
    ensemble = EnsemblePredictor()

    # Connect to MT5
    logger.info(f"Connecting to MetaTrader 5...")
    if not connector.connect():
        logger.error("Failed to connect to MT5. Ensure the EA is running.")
        sys.exit(1)

    executor = TradeExecutor(connector, risk_manager)

    # Load or train model
    if args.retrain or not ensemble.load(symbol):
        logger.info("Training models (this may take a few minutes)...")
        from train import train_symbol
        train_symbol(connector, symbol, 5000)
        ensemble.load(symbol)

    if not ensemble.is_trained:
        logger.error("No trained model available. Run train.py first.")
        connector.disconnect()
        sys.exit(1)

    # Show account info
    status = executor.get_status()
    logger.info(f"Account: balance={status.get('balance', 0):.2f} equity={status.get('equity', 0):.2f}")

    candle_count = 0
    last_day = datetime.now().date()

    logger.info(f"\n{'='*60}")
    logger.info(f"Bot started | Symbol: {symbol} | Interval: {check_interval}s")
    logger.info(f"{'='*60}\n")

    try:
        while RUNNING:
            try:
                now = datetime.now()

                # Reset daily stats at midnight
                if now.date() != last_day:
                    risk_manager.reset_daily()
                    last_day = now.date()
                    logger.info("Daily stats reset")

                # Fetch recent data for prediction
                seq_len = LSTM_CONFIG["sequence_length"]
                bars_needed = seq_len + 100  # extra for indicator warmup
                df = connector.get_historical_data(symbol, "M5", bars_needed)

                # Add features
                df_feat = add_all_features(df)
                df_feat.dropna(inplace=True)

                # Use only the features the model was trained on
                feature_cols = ensemble.feature_names
                X = df_feat[feature_cols].values

                # Extract raw advanced features for trade filtering
                last_row = df_feat.iloc[-1]
                raw_features = {
                    "market_quality": last_row.get("market_quality", 0.5),
                    "regime": last_row.get("regime", -1),
                    "hurst": last_row.get("hurst", 0.5),
                    "entropy": last_row.get("entropy", 1.0),
                }

                # Get prediction
                prediction = ensemble.predict_latest(X, raw_features)

                regime_names = {0: "RANGE", 1: "TREND", 2: "BREAKOUT", 3: "CHOPPY"}
                logger.info(
                    f"[{symbol}] Signal: {prediction['signal']} | "
                    f"Conf: {prediction['confidence']:.1%} | "
                    f"Regime: {regime_names.get(prediction['regime'], '?')} | "
                    f"Quality: {prediction['market_quality']:.2f} | "
                    f"Hurst: {prediction['hurst']:.2f} | "
                    f"Trade: {'YES' if prediction['should_trade'] else 'NO'}"
                )

                # Execute with all filters
                current_hour = df_feat.index[-1].hour if hasattr(df_feat.index, 'hour') else now.hour
                result = executor.execute_signal(symbol, prediction, df_feat, current_hour)
                if result["action"] == "EXECUTED":
                    logger.info(f">>> Trade opened: {result}")

                # Manage existing positions
                executor.manage_positions(symbol)

                # Print status
                status = executor.get_status()
                logger.info(
                    f"Status: balance={status.get('balance', 0):.2f} | "
                    f"equity={status.get('equity', 0):.2f} | "
                    f"positions={status.get('open_positions', 0)} | "
                    f"daily_pnl={status.get('daily_pnl', 0):.2f}"
                )

                # Check if retraining is needed
                candle_count += 1
                if candle_count >= RETRAIN_INTERVAL:
                    logger.info("Retraining models...")
                    from train import train_symbol
                    train_symbol(connector, symbol, 5000)
                    ensemble.load(symbol)
                    candle_count = 0

            except ConnectionError as e:
                logger.error(f"Connection lost: {e}. Reconnecting...")
                time.sleep(5)
                connector.connect()
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)

            # Wait for next candle
            time.sleep(check_interval)

    finally:
        logger.info("Shutting down...")
        status = executor.get_status()
        logger.info(f"Final status: {status}")
        connector.disconnect()
        logger.info("Bot stopped.")


if __name__ == "__main__":
    main()
