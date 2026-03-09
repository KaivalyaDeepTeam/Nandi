"""
Run Nandi V2 live — multi-pair portfolio trading via MT5.

Uses TradingOrchestrator to wire: data -> features -> regime -> alphas -> optimize -> risk.
Synchronizes positions with MT5 via file bridge.

Usage:
    python run_nandi.py
    python run_nandi.py --paper
    python run_nandi.py --timeframe M5 --paper
    python run_nandi.py --pairs eurusd gbpusd --interval 300
"""

import argparse
import logging
import signal
import sys
import os
import time
import json
from datetime import datetime

import numpy as np
import torch
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nandi")

np.random.seed(42)
torch.manual_seed(42)

RUNNING = True

# Polling intervals by timeframe
POLL_INTERVALS = {
    "D1": 300,   # 5 minutes
    "M5": 30,    # 30 seconds (check all 8 pairs, take best setups)
    "M1": 5,     # 5 seconds
}


def signal_handler(sig, frame):
    global RUNNING
    logger.info("Shutdown signal received...")
    RUNNING = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def main():
    parser = argparse.ArgumentParser(description="Run Nandi V2 Live Trading")
    parser.add_argument("--paper", action="store_true", help="Paper trading mode")
    parser.add_argument("--interval", type=int, default=None,
                        help="Check interval in seconds (auto-set by timeframe if omitted)")
    parser.add_argument("--timeframe", type=str, default="D1",
                        choices=["D1", "M5"],
                        help="Trading timeframe (default: D1)")
    parser.add_argument("--pairs", nargs="+", default=None)
    args = parser.parse_args()

    # Auto-set polling interval based on timeframe
    if args.interval is None:
        args.interval = POLL_INTERVALS.get(args.timeframe, 300)

    from nandi.config import (
        MODEL_DIR, LIVE_CONFIG, MT5_FILES_DIR, PAIRS,
        TIMEFRAME_PROFILES, NEWS_CONFIG,
    )
    from nandi.data.manager import download_forex_data
    from nandi.data.features import compute_features
    from nandi.data.advanced_features import compute_advanced_features
    from nandi.models.agent import NandiAgent
    from nandi.orchestrator import TradingOrchestrator
    from nandi.regime.hmm_detector import HMMRegimeDetector
    from nandi.risk.tail_hedge import TailRiskHedger
    from nandi.backtest.ood_detector import OODDetector

    profile = TIMEFRAME_PROFILES.get(args.timeframe, TIMEFRAME_PROFILES["D1"])
    lookback = profile["lookback_bars"]

    # Load trained pairs
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

    # Load agents and scalers
    agents = {}
    scalers = {}
    feature_names_map = {}

    for pair in pairs:
        pair_dir = os.path.join(MODEL_DIR, pair)
        agent_path = os.path.join(pair_dir, "agent.pt")
        scaler_path = os.path.join(pair_dir, "scaler.pkl")
        fn_path = os.path.join(pair_dir, "feature_names.pkl")

        if not all(os.path.exists(p) for p in [agent_path, scaler_path, fn_path]):
            logger.warning(f"Missing model files for {pair}, skipping")
            continue

        # Check per-pair meta for encoder type override
        pair_meta_path = os.path.join(pair_dir, "meta.pkl")
        pair_encoder = encoder_type
        if os.path.exists(pair_meta_path):
            pair_meta = joblib.load(pair_meta_path)
            pair_encoder = pair_meta.get("encoder", encoder_type)

        feature_names = joblib.load(fn_path)
        scaler = joblib.load(scaler_path)

        # Check if trained with AEGIS
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
            scalers[pair] = scaler
            feature_names_map[pair] = feature_names
            logger.info(f"Loaded {pair.upper()} agent ({pair_algo}/{pair_encoder})")
        else:
            logger.warning(f"Failed to load {pair} agent")

    if not agents:
        logger.error("No trained models found. Run train_nandi.py first.")
        sys.exit(1)

    active_pairs = list(agents.keys())

    # Initialize orchestrator (wires regime -> alphas -> optimizer -> risk)
    regime_detector = HMMRegimeDetector()
    orchestrator = TradingOrchestrator(
        pairs=active_pairs,
        agents=agents,
        regime_detector=regime_detector,
    )

    # Initialize tail risk hedger and OOD detector
    tail_hedger = TailRiskHedger()
    ood_detector = OODDetector()

    # Initialize news intelligence gate
    news_gate = None
    if NEWS_CONFIG.get("enabled", False):
        try:
            from nandi.data.news.gate import NewsGate
            news_gate = NewsGate(
                finnhub_key=NEWS_CONFIG.get("finnhub_key", ""),
                alpha_vantage_key=NEWS_CONFIG.get("alpha_vantage_key", ""),
                fred_key=NEWS_CONFIG.get("fred_key", ""),
            )
            news_gate.refresh()
            logger.info("News intelligence gate initialized")
            logger.info(news_gate.get_status_display())
        except Exception as e:
            logger.warning(f"News gate init failed (continuing without): {e}")
            news_gate = None

    # Connect to MT5
    connector = None
    position_sync = None
    if not args.paper:
        from nandi.execution.mt5_bridge import MT5Connector
        from nandi.execution.position_sync import PositionSynchronizer
        connector = MT5Connector(MT5_FILES_DIR)
        if not connector.connect():
            logger.error("Cannot connect to MT5. Make sure EA is running.")
            sys.exit(1)
        position_sync = PositionSynchronizer(connector)
        logger.info("Connected to MT5")

    # State tracking
    positions = {pair: 0.0 for pair in active_pairs}
    equity = 10000.0

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  NANDI V2 — Live Trading {'(PAPER)' if args.paper else '(LIVE)'}")
    logger.info(f"  Timeframe: {args.timeframe}")
    logger.info(f"  Pairs: {', '.join(p.upper() for p in active_pairs)}")
    logger.info(f"  Interval: {args.interval}s")
    logger.info(f"  Leverage: {profile['leverage']}x | MaxPos: {profile['max_position']}")
    logger.info(f"  Orchestrator: regime + 4 alphas + Kelly + correlation")
    logger.info(f"{'=' * 60}\n")

    # Decision log
    log_dir = os.path.join("data", "nandi", "live_logs")
    os.makedirs(log_dir, exist_ok=True)

    # Feature computation function depends on timeframe
    is_intraday = args.timeframe != "D1"

    try:
        while RUNNING:
            try:
                now = datetime.now()

                # Build features for all pairs
                features_by_pair = {}
                decision_log = {
                    "timestamp": now.isoformat(),
                    "timeframe": args.timeframe,
                    "signals": {},
                }

                # For M5 intraday: collect all close prices first for cross-pair features
                pair_close_cache = {}

                for pair in active_pairs:
                    try:
                        if is_intraday:
                            feat_vals, close_series = _get_intraday_features(
                                pair, args, connector, lookback,
                                feature_names_map[pair], scalers[pair],
                                args.timeframe, return_closes=True,
                            )
                            if close_series is not None:
                                pair_close_cache[pair] = close_series
                        else:
                            feat_vals = _get_daily_features(
                                pair, args, connector, lookback,
                                feature_names_map[pair], scalers[pair],
                            )

                        if feat_vals is None:
                            continue

                        # Build state tuple for orchestrator
                        portfolio_dd = (orchestrator.current_equity - equity) / orchestrator.current_equity if orchestrator.current_equity > 0 else 0
                        pos_info = np.array([
                            positions.get(pair, 0),
                            equity / 10000 - 1,
                            max(0, portfolio_dd),
                            float(np.std(feat_vals[-10:, 0])),
                        ], dtype=np.float32)

                        features_by_pair[pair] = (feat_vals, pos_info)

                        # OOD detection
                        try:
                            if ood_detector.is_fitted:
                                ood_score = ood_detector.score(feat_vals[-1])
                                if ood_score > ood_detector.threshold:
                                    decision_log["signals"][pair] = {
                                        "ood_warning": True,
                                        "ood_score": float(ood_score),
                                    }
                        except Exception:
                            pass

                    except Exception as e:
                        logger.error(f"Error processing {pair}: {e}")

                # Inject cross-pair lead-lag context for M5
                if is_intraday and len(pair_close_cache) >= 2:
                    try:
                        from nandi.data.cross_pair_scalping import compute_cross_pair_scalping_features
                        for pair in list(features_by_pair.keys()):
                            cross_feats = compute_cross_pair_scalping_features(
                                pair, pair_close_cache
                            )
                            if len(cross_feats) > 0:
                                # Append cross-pair signal summary to decision log
                                last_row = cross_feats.iloc[-1]
                                usd_flow_cols = [c for c in cross_feats.columns if "usd_flow" in c]
                                if usd_flow_cols:
                                    decision_log["signals"].setdefault(pair, {})
                                    decision_log["signals"][pair]["usd_flow"] = float(last_row.get("usd_flow_3b", 0))
                    except Exception as e:
                        logger.debug(f"Cross-pair live features: {e}")

                if not features_by_pair:
                    time.sleep(args.interval)
                    continue

                # Generate signals via orchestrator
                target_positions = orchestrator.generate_signals(
                    features_by_pair,
                    equity=equity,
                    positions=positions,
                )

                # News gate: hard scaling before high-impact events
                if news_gate and NEWS_CONFIG.get("calendar_gate_enabled", True):
                    try:
                        news_gate.refresh()
                        for pair_key in list(target_positions.keys()):
                            scale = news_gate.get_position_scale(pair_key)
                            if scale < 1.0:
                                target_positions[pair_key] *= scale
                                decision_log["signals"].setdefault(pair_key, {})
                                decision_log["signals"][pair_key]["news_gate_scale"] = scale
                    except Exception as e:
                        logger.debug(f"News gate error: {e}")

                # Tail risk hedging
                try:
                    port_ret = sum(
                        positions.get(p, 0) * features_by_pair[p][0][-1, 0]
                        for p in active_pairs if p in features_by_pair
                    ) / max(len(active_pairs), 1)
                    tail_hedger.update(port_ret)
                    scale = tail_hedger.compute_hedge_ratio()
                    if scale < 1.0:
                        logger.warning(f"Tail risk hedger scaling positions by {scale:.2f}")
                        target_positions = {p: v * scale for p, v in target_positions.items()}
                except Exception:
                    pass

                # Log decisions
                decision_log["target_positions"] = target_positions
                decision_log["equity"] = equity

                # Execute
                if not args.paper and position_sync:
                    actions = position_sync.sync(target_positions)
                    if actions:
                        decision_log["executions"] = actions

                # Update state
                positions = target_positions

                # Update equity from MT5
                if not args.paper and connector:
                    try:
                        account = connector.get_account_info()
                        equity = account["equity"]
                    except Exception:
                        pass

                # Print status
                total_exposure = sum(abs(v) for v in positions.values())
                active_positions = {p: v for p, v in positions.items() if abs(v) > 0.05}

                logger.info(
                    f"[{now.strftime('%H:%M:%S')}] {args.timeframe} | "
                    f"Equity: ${equity:,.0f} | "
                    f"Exposure: {total_exposure:.2f} | "
                    f"Active: {len(active_positions)}/{len(active_pairs)}"
                )

                for p, v in active_positions.items():
                    sig = "LONG" if v > 0 else "SHORT"
                    logger.info(f"  {p.upper():>8s}: {sig} {abs(v):.3f}")

                # Save decision log
                log_file = os.path.join(
                    log_dir, f"decisions_{args.timeframe}_{now.strftime('%Y%m%d')}.jsonl"
                )
                with open(log_file, "a") as f:
                    f.write(json.dumps(decision_log) + "\n")

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)

            time.sleep(args.interval)

    finally:
        logger.info(f"\nShutting down Nandi V2...")
        if not args.paper and connector:
            connector.disconnect()
        logger.info("Nandi stopped.")


def _get_daily_features(pair, args, connector, lookback, feature_names, scaler):
    """Build D1 features for a pair (original pipeline)."""
    from nandi.data.manager import download_forex_data
    from nandi.data.features import compute_features
    from nandi.data.advanced_features import compute_advanced_features

    if args.paper:
        df = download_forex_data(symbol=pair, years=1)
        df = df.tail(lookback + 100)
    else:
        from nandi.config import PAIRS_MT5
        mt5_sym = PAIRS_MT5.get(pair, pair.upper())
        df = connector.get_historical_data(mt5_sym, "D1", lookback + 100)

    features = compute_features(df)
    try:
        adv = compute_advanced_features(df)
        features = features.join(adv, how='left')
    except Exception:
        pass

    features.dropna(inplace=True)
    if len(features) < lookback:
        return None

    # Pad missing columns
    for c in feature_names:
        if c not in features.columns:
            features[c] = 0.0

    feat_vals = scaler.transform(
        features[feature_names].values[-lookback:]
    ).astype(np.float32)

    return feat_vals


def _get_intraday_features(pair, args, connector, lookback, feature_names,
                           scaler, timeframe, return_closes=False):
    """Build M5 features for a pair (scalping pipeline).

    Args:
        return_closes: if True, also return the close price Series
                       (used for cross-pair lead-lag computation).
    """
    from nandi.data.mt5_data import MT5DataFetcher
    from nandi.data.scalping_features import compute_scalping_features
    from nandi.config import TIMEFRAME_PROFILES

    profile = TIMEFRAME_PROFILES.get(timeframe, TIMEFRAME_PROFILES["M5"])

    if args.paper:
        # In paper mode, use synthetic data from daily
        from nandi.data.manager import download_forex_data
        from nandi.data.mt5_data import generate_synthetic_m5
        daily_df = download_forex_data(symbol=pair, years=1)
        daily_df = daily_df.tail(5)  # last 5 days of synthetic M5
        df = generate_synthetic_m5(daily_df, pair_name=pair)
    else:
        # Live: read from MT5 bridge
        fetcher = MT5DataFetcher()
        df = fetcher.fetch(pair, bars=lookback + 200)

    if df is None or len(df) < lookback:
        return (None, None) if return_closes else None

    # Save close prices before feature computation (for cross-pair)
    close_series = df["close"].copy() if return_closes else None

    features = compute_scalping_features(df, profile=profile)
    features.dropna(inplace=True)

    if len(features) < lookback:
        return (None, None) if return_closes else None

    # Pad missing columns
    for c in feature_names:
        if c not in features.columns:
            features[c] = 0.0

    feat_vals = scaler.transform(
        features[feature_names].values[-lookback:]
    ).astype(np.float32)

    if return_closes:
        return feat_vals, close_series
    return feat_vals


if __name__ == "__main__":
    main()
