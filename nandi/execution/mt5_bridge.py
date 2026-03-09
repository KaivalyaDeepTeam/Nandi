"""
MT5 File Bridge Connector — communicates with MetaTrader 5 via shared CSV files.
"""

import os
import time
import logging

import pandas as pd

from nandi.config import MT5_FILES_DIR

logger = logging.getLogger(__name__)


class MT5Connector:
    """Connects to MetaTrader 5 via shared file bridge."""

    def __init__(self, files_dir=MT5_FILES_DIR):
        self.files_dir = files_dir
        self.connected = False

        self.data_file = os.path.join(files_dir, "fx_data.csv")
        self.tick_file = os.path.join(files_dir, "fx_tick.csv")
        self.account_file = os.path.join(files_dir, "fx_account.csv")
        self.positions_file = os.path.join(files_dir, "fx_positions.csv")
        self.cmd_file = os.path.join(files_dir, "fx_command.csv")
        self.resp_file = os.path.join(files_dir, "fx_response.csv")

    def connect(self):
        if not os.path.isdir(self.files_dir):
            logger.error(f"MT5 Files directory not found: {self.files_dir}")
            return False

        if os.path.exists(self.tick_file):
            self.connected = True
            logger.info(f"Connected to MT5 via files at {self.files_dir}")
            return True
        else:
            logger.error("EA files not found. Make sure ForexPredictor EA is running.")
            return False

    def disconnect(self):
        self.connected = False
        logger.info("Disconnected from MT5")

    def _wait_for_response(self, timeout=30.0):
        start = time.time()
        while time.time() - start < timeout:
            if os.path.exists(self.resp_file):
                time.sleep(0.1)
                with open(self.resp_file, "r") as f:
                    response = f.read().strip()
                os.remove(self.resp_file)
                return response
            time.sleep(0.1)
        raise TimeoutError("No response from MT5 EA within timeout")

    def _send_command(self, *args):
        if not self.connected:
            raise ConnectionError("Not connected to MT5")
        if os.path.exists(self.resp_file):
            os.remove(self.resp_file)
        with open(self.cmd_file, "w") as f:
            f.write(",".join(str(a) for a in args))
        return self._wait_for_response()

    def get_historical_data(self, symbol, timeframe, bars):
        try:
            resp = self._send_command("REFRESH_HISTORY", symbol, bars, timeframe)
            if not resp.startswith("OK"):
                logger.warning(f"History refresh response: {resp}")
        except TimeoutError:
            logger.warning("History refresh timed out, using existing data file")

        if not os.path.exists(self.data_file):
            raise RuntimeError("No history data file found")

        df = pd.read_csv(self.data_file)
        df["time"] = pd.to_datetime(df["time"].astype(int), unit="s")
        df.set_index("time", inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df

    def get_tick(self, symbol):
        if not os.path.exists(self.tick_file):
            raise RuntimeError("No tick data file found")

        df = pd.read_csv(self.tick_file)
        tick = {}
        for _, row in df.iterrows():
            key = str(row["field"])
            val = row["value"]
            try:
                tick[key] = float(val)
            except (ValueError, TypeError):
                tick[key] = str(val)
        return tick

    def get_account_info(self):
        if not os.path.exists(self.account_file):
            raise RuntimeError("No account data file found")

        df = pd.read_csv(self.account_file)
        account = {}
        for _, row in df.iterrows():
            key = str(row["field"])
            val = row["value"]
            try:
                account[key] = float(val)
            except (ValueError, TypeError):
                account[key] = str(val)
        return account

    def open_trade(self, symbol, order_type, lot_size, sl_price=0, tp_price=0, comment=""):
        resp = self._send_command(order_type, symbol, lot_size, sl_price, tp_price, comment)
        parts = resp.split(",")
        if parts[0] != "OK":
            raise RuntimeError(f"Failed to open trade: {resp}")
        result = {"status": "OK", "ticket": int(parts[1]) if len(parts) > 1 else 0}
        if len(parts) > 2:
            result["price"] = float(parts[2])
        logger.info(f"Opened {order_type} {lot_size} lots {symbol} | ticket={result.get('ticket')}")
        return result

    def close_trade(self, ticket):
        resp = self._send_command("CLOSE", ticket)
        if not resp.startswith("OK"):
            raise RuntimeError(f"Failed to close trade: {resp}")
        logger.info(f"Closed trade ticket={ticket}")
        return {"status": "OK"}

    def get_open_positions(self):
        if not os.path.exists(self.positions_file):
            return []

        df = pd.read_csv(self.positions_file)
        if df.empty:
            return []

        positions = []
        for _, row in df.iterrows():
            positions.append({
                "ticket": int(row["ticket"]),
                "symbol": str(row["symbol"]),
                "type": int(row["type"]),
                "volume": float(row["volume"]),
                "price_open": float(row["price_open"]),
                "sl": float(row["sl"]),
                "tp": float(row["tp"]),
                "profit": float(row["profit"]),
                "comment": str(row.get("comment", "")),
            })
        return positions

    def modify_trade(self, ticket, sl_price=0, tp_price=0):
        resp = self._send_command("MODIFY", ticket, sl_price, tp_price)
        if not resp.startswith("OK"):
            raise RuntimeError(f"Failed to modify trade: {resp}")
        return {"status": "OK"}

    def get_symbol_info(self, symbol):
        tick = self.get_tick(symbol)
        return {
            "point": tick.get("point", 0.0001),
            "digits": int(tick.get("digits", 5)),
            "trade_tick_value": tick.get("trade_tick_value", 10),
            "trade_tick_size": tick.get("trade_tick_size", 0.00001),
            "volume_min": tick.get("volume_min", 0.01),
            "volume_max": tick.get("volume_max", 100),
            "volume_step": tick.get("volume_step", 0.01),
            "spread": tick.get("spread", 15),
        }
