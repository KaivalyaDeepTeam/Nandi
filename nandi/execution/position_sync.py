"""
Multi-pair position synchronization — maps RL positions to MT5 orders.
"""

import logging
from nandi.config import LIVE_CONFIG, PAIRS_MT5

logger = logging.getLogger(__name__)

# Adaptive SL/TP ATR multipliers (ported from V1)
_SL_ATR_MULT = 1.5
_TP_ATR_MULT = 2.5


class PositionSynchronizer:
    """Synchronizes target positions from the portfolio optimizer with MT5 positions."""

    def __init__(self, connector):
        self.connector = connector
        self.current_positions = {}  # pair -> {"ticket": ..., "direction": ..., "volume": ...}

    def _compute_sl_tp(self, entry_price, order_type, atr):
        """Compute adaptive SL and TP prices from ATR.

        Args:
            entry_price: fill/entry price of the trade.
            order_type: "BUY" or "SELL".
            atr: current ATR value for the pair.

        Returns:
            (sl_price, tp_price) tuple.
        """
        sl_distance = _SL_ATR_MULT * atr
        tp_distance = _TP_ATR_MULT * atr
        if order_type == "BUY":
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:  # SELL
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
        return sl_price, tp_price

    def sync(self, target_positions, lot_size_base=None, atr_values=None):
        """Synchronize MT5 positions to match target positions.

        Args:
            target_positions: {pair: float} target position in [-1, 1] per pair.
            lot_size_base: base lot size for position sizing.
            atr_values: optional {pair: float} ATR per pair. When provided,
                adaptive SL/TP are computed (SL=1.5*ATR, TP=2.5*ATR) and
                passed to open_trade so MT5 enforces them server-side.

        Returns:
            list of executed actions.
        """
        lot_base = lot_size_base or LIVE_CONFIG["lot_size_base"]
        atr_map = atr_values or {}
        actions = []

        # Get current MT5 positions
        mt5_positions = self._get_mt5_positions()

        for pair, target in target_positions.items():
            mt5_symbol = PAIRS_MT5.get(pair, pair.upper())
            current = mt5_positions.get(mt5_symbol, {"direction": 0, "volume": 0, "tickets": []})

            target_direction = 1 if target > 0.05 else (-1 if target < -0.05 else 0)
            target_volume = round(abs(target) * lot_base, 2)
            target_volume = max(0.01, target_volume) if target_direction != 0 else 0

            current_direction = current["direction"]
            current_volume = current["volume"]

            # No change needed
            if target_direction == current_direction and abs(target_volume - current_volume) < 0.01:
                continue

            try:
                # Close existing if direction changes or going flat
                if current_direction != 0 and (target_direction != current_direction or target_direction == 0):
                    for ticket in current["tickets"]:
                        self.connector.close_trade(ticket)
                        actions.append({
                            "pair": pair, "action": "CLOSE", "ticket": ticket,
                        })

                # Open new if needed
                if target_direction != 0 and target_direction != current_direction:
                    order_type = "BUY" if target_direction > 0 else "SELL"
                    tick = self.connector.get_tick(mt5_symbol)
                    entry = tick["ask"] if order_type == "BUY" else tick["bid"]

                    # Compute adaptive SL/TP when ATR is available
                    sl_price = None
                    tp_price = None
                    atr = atr_map.get(pair)
                    if atr is not None and atr > 0:
                        sl_price, tp_price = self._compute_sl_tp(entry, order_type, atr)
                        logger.debug(
                            f"{pair} {order_type} entry={entry:.5f} atr={atr:.5f} "
                            f"sl={sl_price:.5f} tp={tp_price:.5f}"
                        )

                    result = self.connector.open_trade(
                        mt5_symbol, order_type, target_volume,
                        sl=sl_price, tp=tp_price,
                        comment=f"NANDI_V2|{pair}"
                    )
                    action_record = {
                        "pair": pair, "action": order_type,
                        "volume": target_volume, "result": result,
                    }
                    if sl_price is not None:
                        action_record["sl"] = sl_price
                        action_record["tp"] = tp_price
                    actions.append(action_record)

            except Exception as e:
                logger.error(f"Position sync error for {pair}: {e}")
                actions.append({"pair": pair, "action": "ERROR", "error": str(e)})

        return actions

    def _get_mt5_positions(self):
        """Get current MT5 positions grouped by symbol."""
        try:
            positions = self.connector.get_open_positions()
        except Exception:
            return {}

        grouped = {}
        for pos in positions:
            symbol = pos["symbol"]
            if symbol not in grouped:
                grouped[symbol] = {"direction": 0, "volume": 0, "tickets": []}

            direction = 1 if pos["type"] == 0 else -1  # 0=BUY, 1=SELL
            grouped[symbol]["direction"] = direction
            grouped[symbol]["volume"] += pos["volume"]
            grouped[symbol]["tickets"].append(pos["ticket"])

        return grouped

    def close_all(self):
        """Close all open positions."""
        try:
            positions = self.connector.get_open_positions()
            for pos in positions:
                self.connector.close_trade(pos["ticket"])
                logger.info(f"Closed {pos['symbol']} ticket={pos['ticket']}")
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
