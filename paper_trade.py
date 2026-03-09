"""
Quick Paper Trading — Start paper trading immediately.

If no trained AEGIS model exists, trains a quick model (50K steps on synthetic/cached data)
then starts paper trading loop. If a trained model exists, uses it directly.

The full training (500K steps on real data) can continue in parallel.

Usage:
    python paper_trade.py                          # Quick start with EURUSD
    python paper_trade.py --pairs eurusd gbpusd    # Multiple pairs
    python paper_trade.py --skip-train             # Use existing model only
    python paper_trade.py --timesteps 100000       # Longer quick train
"""

import argparse
import logging
import os
import sys
import signal
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
logger = logging.getLogger("nandi.paper")

RUNNING = True


def signal_handler(sig, frame):
    global RUNNING
    logger.info("\nShutdown signal received...")
    RUNNING = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def has_trained_model(pair, model_dir):
    """Check if a trained model exists for a pair."""
    pair_dir = os.path.join(model_dir, pair)
    return all(
        os.path.exists(os.path.join(pair_dir, f))
        for f in ["agent.pt", "scaler.pkl", "feature_names.pkl"]
    )


def quick_train(pairs, timesteps, model_dir):
    """Quick-train AEGIS on synthetic/cached data for immediate paper trading."""
    from nandi.config import TIMEFRAME_PROFILES, TRAINING_CONFIG, AEGIS_CONFIG
    from nandi.data.manager import DataManager
    from nandi.environment.single_pair_env import MultiEpisodeEnv
    from nandi.environment.market_sim import MarketSimulator
    from nandi.models.aegis import AEGISAgent

    timeframe = "M5"
    profile = TIMEFRAME_PROFILES[timeframe]
    lookback = profile["lookback_bars"]

    logger.info("=" * 60)
    logger.info("  QUICK TRAIN — Preparing model for paper trading")
    logger.info(f"  Pairs: {', '.join(p.upper() for p in pairs)}")
    logger.info(f"  Timesteps: {timesteps:,} (quick mode)")
    logger.info("=" * 60)

    # Use 2 years of daily data -> synthetic M5 (fast, no download needed if cached)
    dm = DataManager(pairs=pairs, years=2, test_months=3, timeframe=timeframe)
    pair_data = dm.prepare_all()

    if not pair_data:
        logger.error("No data available for quick training")
        return False

    training_config = TRAINING_CONFIG.copy()
    training_config["total_timesteps"] = timesteps
    training_config["eval_interval"] = max(timesteps // 10, 1000)
    training_config["save_interval"] = timesteps  # save at end

    for pair in pairs:
        if pair not in pair_data:
            continue

        data = pair_data[pair]
        logger.info(f"\n  Quick training {pair.upper()} ({timesteps:,} steps)...")

        market_sim = MarketSimulator(pair_name=pair, timeframe=timeframe)
        episode_length = profile["episode_bars"]

        train_env = MultiEpisodeEnv(
            features=data["train_features"],
            prices=data["train_prices"],
            lookback=lookback,
            episode_length=episode_length,
            pair_name=pair,
            market_sim=market_sim,
            timeframe=timeframe,
        )

        test_len = len(data["test_prices"]) - lookback - 2
        if test_len < 10:
            continue

        eval_env = MultiEpisodeEnv(
            features=data["test_features"],
            prices=data["test_prices"],
            lookback=lookback,
            episode_length=test_len,
            pair_name=pair,
            market_sim=market_sim,
            timeframe=timeframe,
        )

        agent = AEGISAgent(
            n_features=data["n_features"],
            encoder_type="msfan",
            regime_dim=AEGIS_CONFIG["regime_dim"],
            cvar_alpha=AEGIS_CONFIG["cvar_alpha"],
        )

        # Initialize
        dummy_ms = torch.zeros(1, lookback, data["n_features"])
        dummy_pi = torch.zeros(1, 4)
        agent(dummy_ms, dummy_pi)

        # Train
        from nandi.training.aegis_trainer import AEGISTrainer
        trainer_keys = {
            "cvar_alpha", "n_quantiles", "asymmetry_factor", "kl_coef",
            "edge_coef", "edge_util_coef", "batch_size", "buffer_capacity",
            "tau_soft", "gamma", "warmup_steps",
        }
        trainer_params = {k: v for k, v in AEGIS_CONFIG.items() if k in trainer_keys}
        trainer_params["lr"] = AEGIS_CONFIG.get("learning_rate", 3e-4)

        trainer = AEGISTrainer(
            agent=agent, train_env=train_env, eval_env=eval_env,
            training_config=training_config, **trainer_params,
        )
        trainer.train()

        # Save
        pair_dir = os.path.join(model_dir, pair)
        os.makedirs(pair_dir, exist_ok=True)
        agent.save_agent(os.path.join(pair_dir, "agent.pt"))
        joblib.dump(data["scaler"], os.path.join(pair_dir, "scaler.pkl"))
        joblib.dump(data["feature_names"], os.path.join(pair_dir, "feature_names.pkl"))
        joblib.dump({"algo": "aegis", "encoder": "msfan", "timeframe": "M5"},
                    os.path.join(pair_dir, "meta.pkl"))

        logger.info(f"  [{pair.upper()}] Quick model saved")

    # Save portfolio config
    joblib.dump(pairs, os.path.join(model_dir, "trained_pairs.pkl"))
    joblib.dump({"algo": "aegis", "encoder": "msfan", "timeframe": "M5"},
                os.path.join(model_dir, "training_meta.pkl"))

    return True


def run_paper_trading(pairs, model_dir, interval=30):
    """Run paper trading loop using trained models."""
    from nandi.config import TIMEFRAME_PROFILES, INITIAL_BALANCE, SCALPING_CONFIG
    from nandi.data.mt5_data import MT5DataFetcher, generate_synthetic_m5
    from nandi.data.manager import download_forex_data
    from nandi.data.scalping_features import compute_scalping_features
    from nandi.models.aegis import AEGISAgent

    timeframe = "M5"
    profile = TIMEFRAME_PROFILES[timeframe]
    lookback = profile["lookback_bars"]

    # Load models
    agents = {}
    scalers = {}
    feature_names_map = {}

    for pair in pairs:
        pair_dir = os.path.join(model_dir, pair)
        if not has_trained_model(pair, model_dir):
            logger.warning(f"No model for {pair}, skipping")
            continue

        feature_names = joblib.load(os.path.join(pair_dir, "feature_names.pkl"))
        scaler = joblib.load(os.path.join(pair_dir, "scaler.pkl"))

        agent = AEGISAgent(n_features=len(feature_names), encoder_type="msfan")
        dummy_ms = torch.zeros(1, lookback, len(feature_names))
        dummy_pi = torch.zeros(1, 4)
        agent(dummy_ms, dummy_pi)

        if agent.load_agent(os.path.join(pair_dir, "agent.pt")):
            agents[pair] = agent
            scalers[pair] = scaler
            feature_names_map[pair] = feature_names
            logger.info(f"  Loaded {pair.upper()} AEGIS agent")

    if not agents:
        logger.error("No models loaded. Run with --skip-train=False first.")
        return

    active_pairs = list(agents.keys())

    # Initialize news gate
    news_gate = None
    try:
        from nandi.config import NEWS_CONFIG
        if NEWS_CONFIG.get("enabled", False):
            from nandi.data.news.gate import NewsGate
            news_gate = NewsGate(
                finnhub_key=NEWS_CONFIG.get("finnhub_key", ""),
                alpha_vantage_key=NEWS_CONFIG.get("alpha_vantage_key", ""),
                fred_key=NEWS_CONFIG.get("fred_key", ""),
            )
            news_gate.refresh()
            logger.info("News intelligence gate active")
            logger.info(news_gate.get_status_display())
    except Exception as e:
        logger.info(f"News gate not available (continuing without): {e}")

    # Paper trading state
    equity = INITIAL_BALANCE
    peak_equity = equity
    positions = {pair: 0.0 for pair in active_pairs}
    trade_log = []  # all trades
    n_trades = 0
    total_pnl = 0.0

    # Simulated M5 bar index (we'll cycle through synthetic data)
    sim_data = {}
    sim_idx = {}

    for pair in active_pairs:
        daily_df = download_forex_data(symbol=pair, years=1)
        daily_df = daily_df.tail(30)  # last 30 days
        m5_df = generate_synthetic_m5(daily_df, pair_name=pair)
        features = compute_scalping_features(m5_df, profile=profile)
        features.dropna(inplace=True)

        if len(features) >= lookback:
            sim_data[pair] = {
                "m5": m5_df,
                "features": features,
                "prices": m5_df.loc[features.index, "close"].values,
            }
            sim_idx[pair] = lookback  # start after lookback
            logger.info(f"  [{pair.upper()}] Loaded {len(features)} simulated M5 bars")

    if not sim_data:
        logger.error("No simulation data available")
        return

    # Also try to load real MT5 data if available
    fetcher = MT5DataFetcher()
    for pair in active_pairs:
        real_df = fetcher.fetch(pair)
        if real_df is not None and len(real_df) > lookback + 100:
            features = compute_scalping_features(real_df, profile=profile)
            features.dropna(inplace=True)
            if len(features) >= lookback:
                sim_data[pair] = {
                    "m5": real_df,
                    "features": features,
                    "prices": real_df.loc[features.index, "close"].values,
                }
                sim_idx[pair] = lookback
                logger.info(f"  [{pair.upper()}] Using REAL M5 data from MT5 bridge!")

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  NANDI PAPER TRADING — AEGIS M5 SCALPER")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Pairs:    {', '.join(p.upper() for p in active_pairs)}")
    logger.info(f"  Balance:  ${equity:,.0f}")
    logger.info(f"  Leverage: {profile['leverage']}x")
    logger.info(f"  MaxPos:   {profile['max_position']}")
    logger.info(f"  Interval: {interval}s per M5 bar simulation")
    logger.info(f"{'=' * 60}\n")

    log_dir = os.path.join("data", "nandi", "paper_logs")
    os.makedirs(log_dir, exist_ok=True)

    bar_count = 0

    while RUNNING:
        try:
            bar_count += 1
            now = datetime.now()
            bar_signals = {}

            for pair in active_pairs:
                if pair not in sim_data:
                    continue

                data = sim_data[pair]
                idx = sim_idx[pair]

                # Check if we've exhausted the data
                if idx >= len(data["features"]):
                    sim_idx[pair] = lookback  # loop back
                    idx = lookback

                # Get feature window
                feat_names = feature_names_map[pair]
                raw_features = data["features"].iloc[idx - lookback:idx]

                # Pad/align columns
                for c in feat_names:
                    if c not in raw_features.columns:
                        raw_features[c] = 0.0

                feat_vals = scalers[pair].transform(
                    raw_features[feat_names].values
                ).astype(np.float32)

                # Current price
                price = data["prices"][idx]

                # Build state
                pos_info = np.array([
                    positions.get(pair, 0),
                    equity / INITIAL_BALANCE - 1,
                    max(0, (peak_equity - equity) / peak_equity),
                    float(np.std(feat_vals[-10:, 0])) if feat_vals.shape[0] >= 10 else 0,
                ], dtype=np.float32)

                # Agent decision
                ms_tensor = torch.tensor(feat_vals).unsqueeze(0)
                pi_tensor = torch.tensor(pos_info).unsqueeze(0)

                with torch.no_grad():
                    action, log_prob, value, edge_score = agents[pair].get_action(
                        ms_tensor, pi_tensor
                    )

                action_val = action.item()
                edge_val = edge_score.item() if edge_score is not None else 1.0

                # Apply edge gate
                if edge_val < SCALPING_CONFIG["min_confidence"]:
                    action_val = 0.0  # skip — not confident enough

                # News gate: scale down before high-impact events
                news_scale = 1.0
                if news_gate:
                    try:
                        news_scale = news_gate.get_position_scale(pair)
                        action_val *= news_scale
                    except Exception:
                        pass

                # Clip to max position
                action_val = np.clip(action_val, -profile["max_position"], profile["max_position"])

                # Calculate PnL from position change
                old_pos = positions.get(pair, 0)
                if abs(old_pos) > 0.01 and abs(action_val - old_pos) > 0.01:
                    # Position changed — calculate PnL
                    prev_price = data["prices"][max(0, idx - 1)]
                    price_change = (price - prev_price) / prev_price
                    pnl = old_pos * price_change * equity * profile["leverage"]
                    total_pnl += pnl
                    equity += pnl

                    if abs(action_val) < 0.01 or np.sign(action_val) != np.sign(old_pos):
                        n_trades += 1
                        trade_log.append({
                            "bar": bar_count,
                            "pair": pair,
                            "direction": "LONG" if old_pos > 0 else "SHORT",
                            "pnl": pnl,
                            "edge": edge_val,
                        })

                positions[pair] = action_val
                peak_equity = max(peak_equity, equity)

                bar_signals[pair] = {
                    "action": round(action_val, 4),
                    "edge": round(edge_val, 4),
                    "price": round(price, 5),
                    "position": round(action_val, 4),
                }

                sim_idx[pair] = idx + 1

            # Status log
            dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
            total_return = (equity / INITIAL_BALANCE - 1) * 100
            active_pos = {p: v for p, v in positions.items() if abs(v) > 0.05}

            if bar_count % 5 == 0 or active_pos:
                logger.info(
                    f"[Bar {bar_count}] Equity: ${equity:,.2f} ({total_return:+.2f}%) | "
                    f"DD: {dd:.2f}% | Trades: {n_trades} | "
                    f"Active: {len(active_pos)}/{len(active_pairs)}"
                )
                for p, v in active_pos.items():
                    sig = "LONG" if v > 0 else "SHORT"
                    edge = bar_signals.get(p, {}).get("edge", 0)
                    logger.info(f"  {p.upper():>8s}: {sig} {abs(v):.3f} (edge={edge:.2f})")

            # Save log
            log_entry = {
                "bar": bar_count,
                "timestamp": now.isoformat(),
                "equity": round(equity, 2),
                "return_pct": round(total_return, 4),
                "drawdown_pct": round(dd, 4),
                "n_trades": n_trades,
                "signals": bar_signals,
            }
            log_file = os.path.join(log_dir, f"paper_{now.strftime('%Y%m%d')}.jsonl")
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Risk check — stop if DD > 10%
            if dd > 10:
                logger.warning(f"DRAWDOWN LIMIT HIT: {dd:.1f}% > 10%. Flattening all positions.")
                positions = {pair: 0.0 for pair in active_pairs}

        except Exception as e:
            logger.error(f"Paper trading error: {e}", exc_info=True)

        time.sleep(interval)

    # Final summary
    total_return = (equity / INITIAL_BALANCE - 1) * 100
    max_dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > INITIAL_BALANCE else 0
    win_trades = [t for t in trade_log if t["pnl"] > 0]
    win_rate = len(win_trades) / max(n_trades, 1) * 100

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  PAPER TRADING SESSION SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Final Equity:  ${equity:,.2f}")
    logger.info(f"  Total Return:  {total_return:+.2f}%")
    logger.info(f"  Max Drawdown:  {max_dd:.2f}%")
    logger.info(f"  Total Trades:  {n_trades}")
    logger.info(f"  Win Rate:      {win_rate:.1f}%")
    logger.info(f"  Total PnL:     ${total_pnl:+,.2f}")
    logger.info(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Nandi Quick Paper Trading")
    parser.add_argument("--pairs", nargs="+", default=["eurusd", "gbpusd", "usdjpy"],
                        help="Pairs to paper trade (default: eurusd gbpusd usdjpy)")
    parser.add_argument("--timesteps", type=int, default=50_000,
                        help="Quick training timesteps if no model exists (default: 50K)")
    parser.add_argument("--interval", type=int, default=5,
                        help="Seconds between simulated M5 bars (default: 5)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, use existing models only")
    args = parser.parse_args()

    from nandi.config import MODEL_DIR
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Check if models exist
    need_train = not args.skip_train and any(
        not has_trained_model(pair, MODEL_DIR) for pair in args.pairs
    )

    if need_train:
        logger.info("No trained models found — starting quick training...")
        pairs_to_train = [p for p in args.pairs if not has_trained_model(p, MODEL_DIR)]
        success = quick_train(pairs_to_train, args.timesteps, MODEL_DIR)
        if not success:
            logger.error("Quick training failed")
            sys.exit(1)

    # Start paper trading
    run_paper_trading(args.pairs, MODEL_DIR, interval=args.interval)


if __name__ == "__main__":
    main()
