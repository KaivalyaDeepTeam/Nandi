"""Live SPIN Trading Bot — M5 scalping on MT5 via NandiBridge.

Usage:
    python live_spin.py
    python live_spin.py --pairs eurusd gbpusd audusd nzdusd usdchf usdcad
    python live_spin.py --lot-size 0.01 --dry-run
    python live_spin.py --interval 10

Ctrl+C to gracefully close all positions and exit.
"""

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from nandi.config import (
    PAIRS, PAIR_TO_IDX, PAIRS_MT5, SPIN_CONFIG, MODEL_DIR,
    SCALPING_CONFIG,
)
from nandi.models.spin_agent import SPINAgent
from nandi.live.bridge import NandiBridgeClient
from nandi.live.feature_engine import LiveFeatureEngine
from nandi.live.risk_manager import LiveRiskManager

# ── Defaults ────────────────────────────────────────────────────
DEFAULT_PAIRS = ["eurusd", "gbpusd", "audusd", "nzdusd", "usdchf", "usdcad"]
DEFAULT_CHECKPOINT = "ppo_agent_best.pt"
DEFAULT_LOT_SIZE = 0.01
DEFAULT_INTERVAL = 10  # seconds

ACTION_NAMES = {0: "HOLD", 1: "LONG", 2: "SHORT", 3: "CLOSE"}

# Session hours (UTC)
LONDON_OPEN = SCALPING_CONFIG["london_open_utc"]   # 7
NY_CLOSE = SCALPING_CONFIG["ny_close_utc"]         # 21

# ── Logging ─────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/live_spin.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("live_spin")


class LiveSPINTrader:
    """Main live trading controller."""

    def __init__(self, pairs, checkpoint, lot_size, dry_run, interval):
        self.pairs = pairs
        self.lot_size = lot_size
        self.dry_run = dry_run
        self.interval = interval
        self.running = True

        # Track last processed bar timestamp per pair
        self.last_bar_time = {pair: None for pair in pairs}

        # Track open MT5 tickets per pair
        self.open_tickets = {pair: None for pair in pairs}

        # Session tracking
        self.in_session = False
        self.total_trades = 0
        self.total_pnl = 0.0

        # Dashboard timer
        self._last_dashboard = 0

        # ── Load model ──
        logger.info("Loading SPINAgent...")
        self.device = self._get_device()
        self.agent = SPINAgent(
            n_features=SPIN_CONFIG["n_features"],
            spin_config=SPIN_CONFIG,
        )
        ckpt_path = os.path.join(MODEL_DIR, checkpoint)
        if not os.path.exists(ckpt_path):
            logger.error(f"Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        self.agent.load_agent(ckpt_path)
        self.agent = self.agent.to(self.device)
        self.agent.eval()
        n_params = sum(p.numel() for p in self.agent.parameters())
        logger.info(f"SPINAgent loaded: {n_params:,} params, device={self.device}")

        # ── Bridge ──
        self.bridge = NandiBridgeClient()

        # ── Feature engines (per pair) ──
        self.engines = {}
        for pair in pairs:
            try:
                self.engines[pair] = LiveFeatureEngine(pair)
                logger.info(f"[{pair}] Feature engine ready")
            except Exception as e:
                logger.error(f"[{pair}] Failed to load scaler: {e}")
                sys.exit(1)

        # ── Risk manager ──
        self.risk = LiveRiskManager(pairs)

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def startup(self):
        """Verify bridge and build initial feature state."""
        logger.info("=" * 60)
        logger.info("SPIN Live Trader starting...")
        logger.info(f"  Pairs: {self.pairs}")
        logger.info(f"  Lot size: {self.lot_size}")
        logger.info(f"  Dry run: {self.dry_run}")
        logger.info(f"  Interval: {self.interval}s")
        logger.info("=" * 60)

        # Check bridge
        logger.info("Pinging NandiBridge EA...")
        if not self.bridge.ping():
            logger.warning("EA not responding — will retry in main loop")

        # Read initial M5 data and build features
        for pair in self.pairs:
            self._update_features(pair)

        # Check readiness
        ready = [p for p in self.pairs if self.engines[p].ready]
        not_ready = [p for p in self.pairs if not self.engines[p].ready]
        logger.info(f"Feature engines ready: {ready}")
        if not_ready:
            logger.warning(f"Not ready (need more bars): {not_ready}")

        # Sync with existing positions
        self._sync_positions()

        logger.info("Startup complete. Entering main loop.")

    def _update_features(self, pair):
        """Read M5 bars and update feature engine for a pair."""
        df = self.bridge.read_m5_bars(pair)
        if df.empty:
            logger.debug(f"[{pair}] No M5 data available")
            return False
        try:
            self.engines[pair].update(df)
            return True
        except Exception as e:
            logger.error(f"[{pair}] Feature update failed: {e}")
            return False

    def _sync_positions(self):
        """Check MT5 for existing positions and sync risk manager state."""
        positions = self.bridge.read_positions()
        for pos in positions:
            sym = str(pos.get("symbol", "")).upper()
            # Find matching pair
            for pair, mt5_sym in PAIRS_MT5.items():
                if sym.startswith(mt5_sym) and pair in self.pairs:
                    ticket = int(pos.get("ticket", 0))
                    pos_type = pos.get("type", "")
                    direction = 1 if str(pos_type) in ("0", "BUY", "0.0") else -1
                    price = float(pos.get("price_open", 0))
                    sl = float(pos.get("sl", 0))

                    self.open_tickets[pair] = ticket
                    # Approximate entry state in risk manager
                    atr = self.engines[pair].get_atr() if self.engines[pair].ready else 0.001
                    self.risk.on_entry(pair, direction, price, atr)
                    if sl > 0:
                        self.risk.states[pair].stop_price = sl
                    logger.info(
                        f"[{pair}] Synced existing position: ticket={ticket} "
                        f"dir={'LONG' if direction == 1 else 'SHORT'} @ {price:.5f}"
                    )
                    break

    def _is_session_active(self):
        """Check if current UTC time is within trading sessions."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        # Weekday check (Mon=0, Fri=4)
        if now.weekday() > 4:
            return False
        return LONDON_OPEN <= hour < NY_CLOSE

    def run(self):
        """Main trading loop."""
        self.startup()

        while self.running:
            try:
                self._loop_iteration()
                time.sleep(self.interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                time.sleep(self.interval)

        self.shutdown()

    def _loop_iteration(self):
        """Single iteration of the main loop."""
        now = datetime.now(timezone.utc)
        session_active = self._is_session_active()

        # Session transition: close all when session ends
        if self.in_session and not session_active:
            logger.info("Session ended — closing all positions")
            self._close_all_positions()
            for pair in self.pairs:
                self.risk.new_session(pair)
            self.in_session = False
            return

        if not session_active:
            return

        if not self.in_session:
            logger.info(f"Session started @ {now.strftime('%H:%M UTC')}")
            self.in_session = True

        # Process each pair
        for pair in self.pairs:
            self._process_pair(pair)

        # Dashboard
        if time.time() - self._last_dashboard > 60:
            self._print_dashboard()
            self._last_dashboard = time.time()

    def _process_pair(self, pair):
        """Process one pair: detect new bar, compute features, decide action."""
        engine = self.engines[pair]

        # Read latest M5 bars
        df = self.bridge.read_m5_bars(pair)
        if df.empty:
            return

        # Detect new bar
        current_last = df.index[-1]
        prev_last = self.last_bar_time[pair]

        if prev_last is not None and current_last <= prev_last:
            # No new bar — but still check SL intra-bar via tick
            self._check_intrabar_sl(pair)
            return

        # New bar detected
        self.last_bar_time[pair] = current_last
        logger.info(f"[{pair}] New M5 bar: {current_last}")

        # Update features
        try:
            engine.update(df)
        except Exception as e:
            logger.error(f"[{pair}] Feature update error: {e}")
            return

        if not engine.ready:
            return

        # Tick risk manager (update bars_in_trade, excursion, cooldown)
        current_price = float(df["close"].iloc[-1])
        self.risk.tick_bar(pair, current_price)

        # Check risk gate closures first
        if self.risk.get_position_state(pair) != 0:
            # Check stop-loss
            if self.risk.check_stop_loss(pair, current_price):
                logger.info(f"[{pair}] STOP-LOSS triggered @ {current_price:.5f}")
                self._execute_close(pair, current_price, is_stop_loss=True)
                return

            # Check max hold
            if self.risk.check_max_hold(pair):
                logger.info(f"[{pair}] MAX HOLD reached — force close")
                self._execute_close(pair, current_price, is_stop_loss=False)
                return

        # Get model decision
        market_state = engine.get_market_state()
        atr = engine.get_atr()
        h1_trend = engine.get_h1_trend()

        position_info = self.risk.get_position_info(pair, atr, current_price)
        action_mask = self.risk.get_action_mask(pair, atr, current_price, h1_trend)
        pair_idx = PAIR_TO_IDX.get(pair, 0)

        action, log_prob, value, probs = self.agent.get_action(
            market_state, position_info, pair_idx,
            action_mask=action_mask,
            deterministic=True,
        )

        action_name = ACTION_NAMES.get(action, "?")
        if action != 0:  # Log non-HOLD actions at INFO
            logger.info(
                f"[{pair}] >> {action_name}  probs=[{probs[0]:.2f},{probs[1]:.2f},"
                f"{probs[2]:.2f},{probs[3]:.2f}]  value={value:.3f}"
            )
        else:
            # Log HOLD with mask and h1_trend at INFO so we can diagnose
            logger.info(
                f"[{pair}] HOLD  p=[{probs[0]:.2f},{probs[1]:.2f},"
                f"{probs[2]:.2f},{probs[3]:.2f}]  v={value:.3f}  "
                f"mask={action_mask.astype(int).tolist()}  h1={h1_trend}"
            )

        # Execute action
        if action == 1 and self.risk.get_position_state(pair) == 0:
            self._execute_entry(pair, 1, current_price, atr)
        elif action == 2 and self.risk.get_position_state(pair) == 0:
            self._execute_entry(pair, -1, current_price, atr)
        elif action == 3 and self.risk.get_position_state(pair) != 0:
            self._execute_close(pair, current_price, is_stop_loss=False)

    def _check_intrabar_sl(self, pair):
        """Check stop-loss using tick data between M5 bars."""
        if self.risk.get_position_state(pair) == 0:
            return

        ticks = self.bridge.read_ticks()
        mt5_sym = PAIRS_MT5.get(pair, pair.upper())

        # Try with and without suffix
        tick = None
        for key in ticks:
            if key.startswith(mt5_sym):
                tick = ticks[key]
                break

        if tick is None:
            return

        # Use bid for long SL check, ask for short SL check
        pos_state = self.risk.get_position_state(pair)
        check_price = tick["bid"] if pos_state == 1 else tick["ask"]

        if self.risk.check_stop_loss(pair, check_price):
            logger.info(
                f"[{pair}] INTRA-BAR STOP-LOSS @ {check_price:.5f} "
                f"(SL={self.risk.get_stop_price(pair):.5f})"
            )
            self._execute_close(pair, check_price, is_stop_loss=True)

    def _execute_entry(self, pair, direction, price, atr):
        """Open a new trade."""
        # Compute SL price
        sl_dist = atr * self.risk.risk["stop_loss_atr_mult"]
        if direction == 1:
            sl_price = price - sl_dist
        else:
            sl_price = price + sl_dist

        dir_name = "LONG" if direction == 1 else "SHORT"
        logger.info(
            f"[{pair}] SIGNAL: {dir_name} @ {price:.5f}  "
            f"SL={sl_price:.5f}  ATR={atr:.5f}"
        )

        if self.dry_run:
            logger.info(f"[{pair}] DRY RUN — not executing")
            # Still update risk manager for tracking
            self.risk.on_entry(pair, direction, price, atr)
            return

        # Send order
        if direction == 1:
            resp = self.bridge.buy(pair, self.lot_size, sl=sl_price, comment="SPIN")
        else:
            resp = self.bridge.sell(pair, self.lot_size, sl=sl_price, comment="SPIN")

        if resp.startswith("OK"):
            parts = resp.split(",")
            ticket = int(parts[1]) if len(parts) > 1 else 0
            self.open_tickets[pair] = ticket
            self.risk.on_entry(pair, direction, price, atr)
            self.total_trades += 1
            logger.info(f"[{pair}] ORDER FILLED: ticket={ticket}")
        else:
            logger.error(f"[{pair}] ORDER FAILED: {resp}")

    def _execute_close(self, pair, price, is_stop_loss=False):
        """Close existing position."""
        sl_tag = " (SL)" if is_stop_loss else ""
        logger.info(f"[{pair}] CLOSING{sl_tag} @ {price:.5f}")

        if not self.dry_run:
            ticket = self.open_tickets.get(pair)
            if ticket:
                resp = self.bridge.close(ticket)
                if resp.startswith("OK"):
                    logger.info(f"[{pair}] CLOSE OK: ticket={ticket}")
                else:
                    logger.error(f"[{pair}] CLOSE FAILED: {resp}")
            else:
                logger.warning(f"[{pair}] No ticket to close")

        net_return = self.risk.on_close(pair, price, is_stop_loss=is_stop_loss)
        self.total_pnl += net_return * 100.0
        self.open_tickets[pair] = None

    def _close_all_positions(self):
        """Close all open positions."""
        for pair in self.pairs:
            if self.risk.get_position_state(pair) != 0:
                # Get latest price
                ticks = self.bridge.read_ticks()
                mt5_sym = PAIRS_MT5.get(pair, pair.upper())
                price = 0.0
                for key in ticks:
                    if key.startswith(mt5_sym):
                        price = ticks[key]["bid"]
                        break
                if price > 0:
                    self._execute_close(pair, price, is_stop_loss=False)

        if not self.dry_run:
            self.bridge.close_all()

    def _print_dashboard(self):
        """Print status dashboard."""
        now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        account = self.bridge.read_account()
        balance = account.get("balance", "?")
        equity = account.get("equity", "?")

        lines = [
            f"\n{'=' * 55}",
            f"  SPIN Dashboard @ {now}",
            f"  Balance: {balance}  Equity: {equity}",
            f"  Total trades: {self.total_trades}  Session PnL: {self.total_pnl:+.2f}%",
            f"{'─' * 55}",
        ]

        for pair in self.pairs:
            s = self.risk.states[pair]
            engine = self.engines[pair]
            status = "FLAT"
            if s.position_state == 1:
                status = f"LONG  bars={s.bars_in_trade} exc={s.current_excursion*100:+.2f}%"
            elif s.position_state == -1:
                status = f"SHORT bars={s.bars_in_trade} exc={s.current_excursion*100:+.2f}%"

            ready = "OK" if engine.ready else "WAIT"
            cd = f"cd={s.cooldown_remaining}" if s.cooldown_remaining > 0 else ""
            lines.append(f"  {pair:8s} [{ready}] {status} {cd}")

        lines.append(f"{'=' * 55}\n")
        dashboard = "\n".join(lines)
        logger.info(dashboard)

    def shutdown(self):
        """Graceful shutdown: close all positions."""
        logger.info("\nShutting down...")
        self._close_all_positions()
        logger.info(f"Final stats: {self.total_trades} trades, PnL: {self.total_pnl:+.2f}%")
        logger.info("Goodbye.")


def main():
    parser = argparse.ArgumentParser(description="SPIN Live Trader for MT5")
    parser.add_argument(
        "--pairs", nargs="+", default=DEFAULT_PAIRS,
        help="Pairs to trade (default: 6 profitable pairs, excludes USDJPY/EURJPY)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
        help=f"Model checkpoint filename (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--lot-size", type=float, default=DEFAULT_LOT_SIZE,
        help=f"Position size in lots (default: {DEFAULT_LOT_SIZE})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Log decisions but don't send commands to MT5",
    )
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL,
        help=f"Check interval in seconds (default: {DEFAULT_INTERVAL})",
    )
    args = parser.parse_args()

    # Validate pairs
    for p in args.pairs:
        if p not in PAIRS:
            logger.error(f"Unknown pair: {p}. Valid: {PAIRS}")
            sys.exit(1)

    trader = LiveSPINTrader(
        pairs=args.pairs,
        checkpoint=args.checkpoint,
        lot_size=args.lot_size,
        dry_run=args.dry_run,
        interval=args.interval,
    )

    # Handle SIGTERM gracefully
    def handle_signal(signum, frame):
        logger.info(f"Signal {signum} received")
        trader.running = False

    signal.signal(signal.SIGTERM, handle_signal)

    trader.run()


if __name__ == "__main__":
    main()
