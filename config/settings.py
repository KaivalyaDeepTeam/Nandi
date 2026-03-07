"""Configuration settings for the Forex Predictor system."""

import os

# ── MT5 File Bridge ──────────────────────────────────────────────────
# Path to MT5's MQL5/Files folder (where EA reads/writes CSV files)
MT5_FILES_DIR = os.path.expanduser(
    "~/Library/Application Support/net.metaquotes.wine.metatrader5/"
    "drive_c/Program Files/MetaTrader 5/MQL5/Files"
)

# ── Trading Pairs (configurable) ─────────────────────────────────────
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"]
DEFAULT_SYMBOL = "EURUSD"

# ── Timeframes ───────────────────────────────────────────────────────
TIMEFRAME = "M5"  # 5-minute candles for prediction
HISTORY_BARS = 20000  # bars to fetch for training (~70 trading days)

# ── Model Parameters ────────────────────────────────────────────────
LSTM_CONFIG = {
    "sequence_length": 30,
    "units_1": 64,
    "units_2": 32,
    "dropout": 0.4,
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 0.0005,
}

XGBOOST_CONFIG = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "min_child_weight": 10,
    "reg_alpha": 1.0,
    "reg_lambda": 5.0,
    "gamma": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
}

ENSEMBLE_CONFIG = {
    "lstm_weight": 0.4,
    "xgboost_weight": 0.6,
    "confidence_threshold": 0.20,  # low threshold, rely on model agreement filter
}

# ── Risk Management ─────────────────────────────────────────────────
# Targeting 30-40% annual = ~$7/day on $5K
# Math: 58% WR * 12pip TP - 42% * 8pip SL = ~3.6 pips/trade
# 3 trades/day * 3.6 pips * $0.50/pip = ~$5.4/day = ~34% annual
RISK_CONFIG = {
    "max_risk_per_trade": 0.02,   # 2% risk per trade
    "max_open_trades": 2,
    "stop_loss_pips": 8,          # tight SL matching M5 moves
    "take_profit_pips": 12,       # 1.5:1 RR, reachable in 1-2 hours
    "trailing_stop_pips": 6,
    "max_daily_loss": 0.05,       # 5% max daily drawdown
    "default_lot_size": 0.05,
}

# ── Feature Engineering ──────────────────────────────────────────────
TECHNICAL_INDICATORS = [
    "rsi", "macd", "bollinger", "ema", "sma",
    "atr", "stochastic", "cci", "williams_r", "adx",
]

# ── Data Paths ───────────────────────────────────────────────────────
MODEL_DIR = "models"
DATA_DIR = "data"

# ── Prediction ───────────────────────────────────────────────────────
PREDICTION_HORIZON = 12  # predict 12 candles ahead (1 hour on M5)
RETRAIN_INTERVAL = 500  # retrain every N new candles
