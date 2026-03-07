"""Nandi Configuration."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "nandi")
MODEL_DIR = os.path.join(BASE_DIR, "models", "nandi")

# ── Data ──────────────────────────────────────────────────────────────
SYMBOL = "EURUSD=X"          # yfinance format
SYMBOL_MT5 = "EURUSD"        # MT5 format
LOOKBACK_YEARS = 20           # Training data span
TEST_MONTHS = 6               # Held-out unseen test period (~130 trading days)
LOOKBACK_WINDOW = 30          # Days of history the agent sees

# ── Environment ──────────────────────────────────────────────────────
INITIAL_BALANCE = 10_000.0
TRANSACTION_COST_BPS = 3.0    # Spread + commission in basis points
LEVERAGE = 100
MAX_POSITION = 1.0            # Max position as fraction of equity

# ── Encoder: Multi-Scale Fractal Attention Network ───────────────────
ENCODER_CONFIG = {
    "d_model": 128,
    "n_scales": 3,
    "kernel_sizes": [3, 7, 15],
    "dilations": [1, 4, 16],
    "n_heads": 4,
    "dropout": 0.15,
}

# ── PPO Agent ────────────────────────────────────────────────────────
PPO_CONFIG = {
    "gamma": 0.99,
    "lambda_gae": 0.95,
    "clip_ratio": 0.2,
    "entropy_coef": 0.02,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "learning_rate": 3e-4,
    "n_epochs": 10,
    "batch_size": 64,
    "rollout_length": 252,      # ~1 trading year per rollout
}

# ── Training ─────────────────────────────────────────────────────────
TRAINING_CONFIG = {
    "total_timesteps": 500_000,
    "eval_interval": 10_000,
    "save_interval": 25_000,
    "n_eval_episodes": 5,
    "seed": 42,
}

# ── Risk (hard limits — not learned, always enforced) ────────────────
RISK_LIMITS = {
    "max_drawdown": 0.15,           # 15% DD → force flat
    "max_daily_loss": 0.03,         # 3% daily loss → stop
    "scale_down_threshold": 0.08,   # 8% DD → halve position
}

# ── Live Trading ─────────────────────────────────────────────────────
LIVE_CONFIG = {
    "lot_size_base": 0.1,
    "check_interval": 300,          # seconds
    "timeframe": "D1",
}

# ── MT5 Bridge ───────────────────────────────────────────────────────
MT5_FILES_DIR = os.path.expanduser(
    "~/Library/Application Support/net.metaquotes.wine.metatrader5/"
    "drive_c/Program Files/MetaTrader 5/MQL5/Files"
)
