"""NandiBridge Client — File-based IPC with MT5 Expert Advisor.

Reads market data and sends trade commands via CSV files in FILE_COMMON.
"""

import csv
import logging
import os
import time

import pandas as pd

from nandi.config import MT5_FILES_DIR, PAIRS_MT5

logger = logging.getLogger(__name__)

# Polling config
RESPONSE_TIMEOUT = 5.0     # seconds to wait for EA response
RESPONSE_POLL_MS = 100      # poll interval in ms

# MQL5 writes CSV in UTF-16 LE with BOM
MT5_ENCODING = "utf-16"


class NandiBridgeClient:
    """File-bridge client for NandiBridge.mq5 EA."""

    def __init__(self, files_dir=None):
        self.dir = files_dir or MT5_FILES_DIR
        logger.info(f"Bridge dir: {self.dir}")

    # ── Paths ──────────────────────────────────────────────────────

    def _path(self, filename):
        return os.path.join(self.dir, filename)

    # ── Read market data ───────────────────────────────────────────

    def read_ticks(self):
        """Read current tick data for all pairs.

        Returns:
            dict: {symbol: {bid, ask, spread, point, digits, ...}}
        """
        path = self._path("fx_tick.csv")
        if not os.path.exists(path):
            return {}
        ticks = {}
        try:
            with open(path, "r", encoding=MT5_ENCODING) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sym = row.get("symbol", "").strip()
                    if sym:
                        ticks[sym] = {
                            "bid": float(row.get("bid", 0)),
                            "ask": float(row.get("ask", 0)),
                            "spread": float(row.get("spread", 0)),
                            "point": float(row.get("point", 0.00001)),
                            "digits": int(row.get("digits", 5)),
                            "time": row.get("time", ""),
                        }
        except Exception as e:
            logger.warning(f"read_ticks error: {e}")
        return ticks

    def read_m5_bars(self, pair):
        """Read M5 OHLCV bars for a pair.

        Args:
            pair: lowercase pair name (e.g. "eurusd")

        Returns:
            DataFrame with DatetimeIndex and OHLCV columns, or empty DataFrame.
        """
        mt5_sym = PAIRS_MT5.get(pair, pair.upper())
        path = self._path(f"fx_m5_{mt5_sym}.csv")
        if not os.path.exists(path):
            logger.debug(f"No M5 file: {path}")
            return pd.DataFrame()
        try:
            df = pd.read_csv(path, encoding=MT5_ENCODING)
            df.columns = [c.strip().lower() for c in df.columns]
            # Ensure standard OHLCV columns
            for col in ["time", "open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    logger.warning(f"Missing column {col} in M5 data for {pair}")
                    return pd.DataFrame()
            # MT5 exports time as Unix epoch seconds (UTC).
            # Parse as UTC, strip tz to naive (training uses naive UTC index).
            df.index = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
            df.index = df.index.tz_localize(None)
            df.index.name = "time"
            df.drop(columns=["time"], inplace=True)
            # Keep real tick_volume (training now uses MT5 data with real volume)
            df.sort_index(inplace=True)
            return df
        except Exception as e:
            logger.warning(f"read_m5_bars({pair}) error: {e}")
            return pd.DataFrame()

    def read_account(self):
        """Read account state.

        Returns:
            dict: {balance, equity, margin, ...} or empty dict.
        """
        path = self._path("fx_account.csv")
        if not os.path.exists(path):
            return {}
        try:
            account = {}
            with open(path, "r", encoding=MT5_ENCODING) as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        key = row[0].strip().lower()
                        val = row[1].strip()
                        try:
                            account[key] = float(val)
                        except ValueError:
                            account[key] = val
            return account
        except Exception as e:
            logger.warning(f"read_account error: {e}")
            return {}

    def read_positions(self):
        """Read open positions.

        Returns:
            list of dicts: [{ticket, symbol, type, volume, price_open, sl, tp, profit, ...}]
        """
        path = self._path("fx_positions.csv")
        if not os.path.exists(path):
            return []
        try:
            positions = []
            with open(path, "r", encoding=MT5_ENCODING) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pos = {}
                    for k, v in row.items():
                        k = k.strip().lower()
                        try:
                            pos[k] = float(v)
                        except (ValueError, TypeError):
                            pos[k] = v.strip() if isinstance(v, str) else v
                    if pos.get("ticket"):
                        pos["ticket"] = int(pos["ticket"])
                        positions.append(pos)
            return positions
        except Exception as e:
            logger.warning(f"read_positions error: {e}")
            return []

    # ── Send commands ──────────────────────────────────────────────

    def send_command(self, cmd_str):
        """Write command to fx_command.csv and poll fx_response.csv.

        Args:
            cmd_str: command string (e.g. "BUY,EURUSD,0.01,1.0850")

        Returns:
            str: response string or "TIMEOUT"
        """
        cmd_path = self._path("fx_command.csv")
        resp_path = self._path("fx_response.csv")

        # Clean stale files
        for p in [cmd_path, resp_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

        # Write command (UTF-16 LE for MQL5 to read)
        try:
            with open(cmd_path, "w", encoding=MT5_ENCODING) as f:
                f.write(cmd_str + "\n")
            logger.info(f"CMD -> {cmd_str}")
        except Exception as e:
            logger.error(f"Failed to write command: {e}")
            return f"ERROR,write_failed,{e}"

        # Poll for response
        deadline = time.time() + RESPONSE_TIMEOUT
        while time.time() < deadline:
            if os.path.exists(resp_path):
                try:
                    with open(resp_path, "r", encoding=MT5_ENCODING) as f:
                        response = f.read().strip()
                    if response:
                        # Clean up
                        try:
                            os.remove(resp_path)
                        except OSError:
                            pass
                        try:
                            os.remove(cmd_path)
                        except OSError:
                            pass
                        logger.info(f"RSP <- {response}")
                        return response
                except Exception:
                    pass
            time.sleep(RESPONSE_POLL_MS / 1000.0)

        logger.warning(f"Command timed out: {cmd_str}")
        return "TIMEOUT"

    # ── Trade commands ─────────────────────────────────────────────

    def buy(self, pair, lots, sl=0, tp=0, comment=""):
        """Send BUY order.

        Args:
            pair: lowercase pair name
            lots: position size
            sl: stop-loss price (0 = none)
            tp: take-profit price (0 = none)

        Returns:
            str: response
        """
        sym = PAIRS_MT5.get(pair, pair.upper())
        parts = ["BUY", sym, f"{lots:.2f}"]
        if sl or tp:
            parts.extend([f"{sl:.5f}", f"{tp:.5f}"])
        if comment:
            parts.append(comment)
        return self.send_command(",".join(parts))

    def sell(self, pair, lots, sl=0, tp=0, comment=""):
        """Send SELL order."""
        sym = PAIRS_MT5.get(pair, pair.upper())
        parts = ["SELL", sym, f"{lots:.2f}"]
        if sl or tp:
            parts.extend([f"{sl:.5f}", f"{tp:.5f}"])
        if comment:
            parts.append(comment)
        return self.send_command(",".join(parts))

    def close(self, ticket):
        """Close position by ticket number."""
        return self.send_command(f"CLOSE,{int(ticket)}")

    def close_all(self):
        """Close all open positions."""
        return self.send_command("CLOSE_ALL")

    def modify(self, ticket, sl=0, tp=0):
        """Modify position SL/TP."""
        return self.send_command(f"MODIFY,{int(ticket)},{sl:.5f},{tp:.5f}")

    def ping(self):
        """Check if EA is alive.

        Returns:
            bool: True if EA responded.
        """
        resp = self.send_command("PING")
        return resp.startswith("OK")
