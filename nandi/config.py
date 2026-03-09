"""Nandi V2 Configuration — Multi-Asset Multi-Strategy Portfolio Trading System."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "nandi")
MODEL_DIR = os.path.join(BASE_DIR, "models", "nandi")

# ── Multi-Pair Configuration ─────────────────────────────────────
PAIRS = [
    "eurusd", "gbpusd", "usdjpy", "audusd",
    "nzdusd", "usdchf", "usdcad", "eurjpy",
]

PAIRS_MT5 = {
    "eurusd": "EURUSD", "gbpusd": "GBPUSD", "usdjpy": "USDJPY",
    "audusd": "AUDUSD", "nzdusd": "NZDUSD", "usdchf": "USDCHF",
    "usdcad": "USDCAD", "eurjpy": "EURJPY",
}

# OctaFX may use symbol suffixes (auto-detected by NandiBridge EA)
BROKER_CONFIG = {
    "name": "OctaFX",
    "suffix": "",               # auto-detected: "", "m", ".pro", etc.
    "timezone": "Asia/Kolkata",  # IST (UTC+5:30)
    "utc_offset_hours": 5.5,
    "currency": "USD",           # account currency
    "min_lot": 0.01,             # micro lots
    "max_lot_paper": 0.1,        # conservative for paper trading
    "max_lot_live": 0.05,        # even more conservative for live
}

# Correlated pair groups for stat arb
PAIR_GROUPS = {
    "eur_gbp": ("eurusd", "gbpusd"),
    "aud_nzd": ("audusd", "nzdusd"),
    "usd_chf_cad": ("usdchf", "usdcad"),
}

# USD-denominated pairs (for DXY proxy)
USD_PAIRS = ["eurusd", "gbpusd", "audusd", "nzdusd"]  # invert these for DXY
USD_PAIRS_DIRECT = ["usdjpy", "usdchf", "usdcad"]     # direct USD strength

# Legacy single-pair (backward compat)
SYMBOL = "EURUSD=X"
SYMBOL_MT5 = "EURUSD"

# ── Data ──────────────────────────────────────────────────────────────
LOOKBACK_YEARS = 20
TEST_MONTHS = 6
LOOKBACK_WINDOW = 30

# ── Timeframes ────────────────────────────────────────────────────
TIMEFRAMES = {
    "D1": {"bars": 500, "mt5_tf": "D1"},
    "H4": {"bars": 500, "mt5_tf": "H4"},
    "H1": {"bars": 500, "mt5_tf": "H1"},
    "M5": {"bars": 2016, "mt5_tf": "M5"},
}

# ── Timeframe Profiles ───────────────────────────────────────────
TIMEFRAME_PROFILES = {
    "D1": {
        "lookback_bars": 30,
        "leverage": 20,
        "episode_bars": 252,
        "bars_per_session": 1,
        "max_position": 1.0,
        "feature_windows": {"returns": [1, 2, 5, 10, 20], "vol": [5, 10, 20, 60]},
        "spread_multiplier": 1.0,
        "data_source": "stooq",
    },
    "M5": {
        "lookback_bars": 120,
        "leverage": 5,
        "episode_bars": 2016,
        "bars_per_session": 288,
        "max_position": 0.3,
        "feature_windows": {"returns": [1, 3, 6, 12, 36], "vol": [12, 36, 72, 288]},
        "spread_multiplier": 0.8,
        "data_source": "mt5",
        "max_hold_bars": 36,
        "min_edge_bps": 2.0,
    },
}

SCALPING_CONFIG = {
    "max_spread_pips": 3.0,
    "session_filter": True,
    "london_open_utc": 7,       # 12:30 PM IST
    "london_close_utc": 16,     # 9:30 PM IST
    "ny_open_utc": 12,          # 5:30 PM IST
    "ny_close_utc": 21,         # 2:30 AM IST (next day)
    "overlap_start_utc": 12,    # London+NY overlap: 5:30-9:30 PM IST (best window)
    "overlap_end_utc": 16,
    "trade_all_pairs": True,
    "min_confidence": 0.6,
}

# India-specific trading schedule (IST = UTC+5:30)
# Best scalping windows for India-based traders:
#   Evening session: 5:30 PM - 9:30 PM IST (London+NY overlap, highest liquidity)
#   Night session:   9:30 PM - 2:30 AM IST (NY session, good liquidity)
#   Avoid: 2:30 AM - 12:30 PM IST (Asian session, wide spreads, low volume)
INDIA_SCHEDULE = {
    "prime_start_ist": "17:30",   # 5:30 PM IST = London+NY overlap start
    "prime_end_ist": "21:30",     # 9:30 PM IST = London close
    "secondary_start_ist": "21:30",
    "secondary_end_ist": "02:30",  # next day
    "avoid_start_ist": "02:30",
    "avoid_end_ist": "12:30",
}

# ── News Intelligence ────────────────────────────────────────────
# Free API keys (get yours — all free, no credit card):
#   Finnhub:       https://finnhub.io/register  (economic calendar + news)
#   Alpha Vantage: https://www.alphavantage.co/support/#api-key  (news sentiment)
#   FRED:          https://fred.stlouisfed.org/docs/api/api_key.html  (Fed rates)
NEWS_CONFIG = {
    "enabled": True,
    "finnhub_key": os.environ.get("FINNHUB_API_KEY", ""),
    "alpha_vantage_key": os.environ.get("ALPHA_VANTAGE_API_KEY", ""),
    "fred_key": os.environ.get("FRED_API_KEY", ""),
    "calendar_gate_enabled": True,    # hard gate: reduce positions before NFP/Fed
    "sentiment_features_enabled": True,
    "rate_features_enabled": True,
    "gate_minutes_before": 30,        # start reducing 30 min before high-impact event
    "gate_minutes_after": 15,         # stay reduced 15 min after event
    "n_news_features": 12,            # total news features appended to state
}

# ── Environment ──────────────────────────────────────────────────────
INITIAL_BALANCE = 5_000.0
TRANSACTION_COST_BPS = 3.0
LEVERAGE = 20  # 20x for daily bars (100x was suicidal — 1% daily move = 100% equity swing)
MAX_POSITION = 1.0
SLIPPAGE_BPS = 1.0  # additional slippage cost in basis points

# Per-pair typical spreads in pips
PAIR_SPREADS = {
    "eurusd": 1.0, "gbpusd": 1.2, "usdjpy": 0.8,
    "audusd": 1.5, "nzdusd": 2.0, "usdchf": 1.5,
    "usdcad": 1.5, "eurjpy": 2.0,
}

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
    "entropy_coef": 0.10,           # doubled: prevent entropy collapse
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "learning_rate": 3e-4,
    "n_epochs": 10,
    "batch_size": 64,
    "rollout_length": 252,
    "min_entropy": 0.5,             # entropy floor to maintain exploration
}

# ── AEGIS: Adaptive Edge-Gated Intelligent Scalper ──────────────────
AEGIS_CONFIG = {
    "cvar_alpha": 0.50,           # optimize worst 50% (less fearful, will actually trade)
    "n_quantiles": 32,            # quantile resolution for return distribution
    "asymmetry_factor": 1.5,     # reduced from 2.0 — less pessimistic critic
    "regime_dim": 8,              # latent regime dimensions
    "kl_coef": 0.01,              # regime VAE regularization weight
    "edge_coef": 1.0,             # edge gate loss weight
    "edge_util_coef": 0.1,        # edge utilization bonus weight
    "batch_size": 256,
    "buffer_capacity": 200_000,
    "tau_soft": 0.005,            # Polyak averaging coefficient
    "gamma": 0.95,                # lower for M5 scalping (short horizons)
    "learning_rate": 1e-4,        # slower LR for stability
    "warmup_steps": 1000,
}

# ── Training ─────────────────────────────────────────────────────────
TRAINING_CONFIG = {
    "total_timesteps": 500_000,
    "eval_interval": 25_000,        # less frequent eval (was 10K)
    "save_interval": 50_000,
    "n_eval_episodes": 10,          # more episodes for stable eval (was 5)
    "early_stop_patience": 10,      # much more patient (was 5)
    "seed": 42,
}

# ── Per-Pair Risk (hard limits — not learned, always enforced) ───────
RISK_LIMITS = {
    "max_drawdown": 0.15,
    "max_daily_loss": 0.03,
    "scale_down_threshold": 0.08,
}

# M5 scalping uses tighter risk limits (capital preservation first)
RISK_LIMITS_M5 = {
    "max_drawdown": 0.10,           # 10% DD → done (tighter than D1's 15%)
    "max_daily_loss": 0.02,         # 2% session loss → force flat (tighter than D1's 3%)
    "scale_down_threshold": 0.05,   # 5% DD → half position (tighter than D1's 8%)
}

# ── Portfolio Risk ───────────────────────────────────────────────────
PORTFOLIO_RISK = {
    "max_portfolio_dd": 0.12,       # 12% portfolio DD → close all
    "max_daily_loss": 0.025,        # 2.5% daily loss → stop trading
    "max_total_exposure": 3.0,      # sum of |positions| across all pairs
    "max_single_pair": 1.0,         # max position for any single pair
    "scale_down_dd": 0.06,          # 6% portfolio DD → reduce all positions
    "scale_down_factor": 0.5,       # multiply all positions by this when scaling down
    "max_correlated_exposure": 2.0, # max combined exposure for correlated pairs
}

# ── Walk-Forward Config ──────────────────────────────────────────────
WALK_FORWARD_CONFIG = {
    "train_days": 504,      # ~2 years
    "val_days": 63,         # ~3 months
    "test_days": 63,        # ~3 months
    "step_days": 21,        # ~1 month
    "n_ensemble": 3,        # average last 3 checkpoints
    "timesteps_per_window": 200_000,
}

# ── Live Trading ─────────────────────────────────────────────────────
LIVE_CONFIG = {
    "lot_size_base": 0.1,
    "check_interval": 300,
    "timeframe": "D1",
}

# ── MT5 Bridge ───────────────────────────────────────────────────────
_MT5_PATHS = [
    # FILE_COMMON path (NandiBridge EA default — most reliable)
    os.path.expanduser(
        "~/Library/Application Support/net.metaquotes.wine.metatrader5/"
        "drive_c/users/Public/AppData/Roaming/MetaQuotes/Terminal/Common/Files"
    ),
    # Standard MQL5/Files path
    os.path.expanduser(
        "~/Library/Application Support/net.metaquotes.wine.metatrader5/"
        "drive_c/Program Files/MetaTrader 5/MQL5/Files"
    ),
    # OctaFX custom installer
    os.path.expanduser(
        "~/Library/Application Support/com.octafx.metatrader5/"
        "drive_c/Program Files/OctaFX MetaTrader 5/MQL5/Files"
    ),
    # Wine prefix (manual install)
    os.path.expanduser(
        "~/.wine/drive_c/Program Files/MetaTrader 5/MQL5/Files"
    ),
]

def _find_mt5_dir():
    for p in _MT5_PATHS:
        if os.path.isdir(p):
            return p
    return _MT5_PATHS[0]

MT5_FILES_DIR = _find_mt5_dir()
