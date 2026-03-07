"""
Training script - fetches data from MT5 and trains the ensemble model.

Usage:
    python train.py --symbol EURUSD
    python train.py --symbol EURUSD --bars 10000
"""

import argparse
import logging
import sys
import os

import numpy as np

from config.settings import HISTORY_BARS, PREDICTION_HORIZON, SYMBOLS
from src.mt5_connector import MT5Connector
from src.feature_engineer import prepare_features
from src.models.ensemble import EnsemblePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def train_symbol(connector: MT5Connector, symbol: str, bars: int):
    """Train the ensemble model for a single symbol."""
    logger.info(f"{'='*60}")
    logger.info(f"Training model for {symbol}")
    logger.info(f"{'='*60}")

    # Fetch historical data
    logger.info(f"Fetching {bars} bars of history...")
    df = connector.get_historical_data(symbol, "M5", bars)
    logger.info(f"Got {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Prepare features
    logger.info("Engineering features...")
    X, y, feature_names, df_feat = prepare_features(df, PREDICTION_HORIZON)
    logger.info(f"Features: {len(feature_names)} | Samples: {len(X)} | Target balance: {y.mean():.2%} bullish")

    # Train ensemble
    ensemble = EnsemblePredictor()
    results = ensemble.train(X, y, feature_names)

    logger.info(f"\nResults for {symbol}:")
    logger.info(f"  LSTM      - {results['lstm']}")
    logger.info(f"  XGBoost   - {results['xgboost']}")
    logger.info(f"  Ensemble  - val_acc={results['ensemble_val_accuracy']:.4f}")

    # Save models
    os.makedirs("models", exist_ok=True)
    ensemble.save(symbol)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train forex prediction models")
    parser.add_argument("--symbol", type=str, default=None, help="Symbol to train (default: all configured)")
    parser.add_argument("--bars", type=int, default=HISTORY_BARS, help="Number of historical bars")
    args = parser.parse_args()

    symbols = [args.symbol] if args.symbol else SYMBOLS

    # Connect to MT5
    connector = MT5Connector()
    if not connector.connect():
        logger.error("Cannot connect to MT5. Make sure the Expert Advisor is running.")
        sys.exit(1)

    try:
        for symbol in symbols:
            train_symbol(connector, symbol, args.bars)
    finally:
        connector.disconnect()

    logger.info("\nTraining complete for all symbols!")


if __name__ == "__main__":
    main()
