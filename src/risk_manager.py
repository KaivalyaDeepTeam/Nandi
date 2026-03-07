"""
Risk management module for position sizing and trade validation.
"""

import logging
from config.settings import RISK_CONFIG

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages trade risk, position sizing, and daily loss limits."""

    def __init__(self, config: dict = None):
        self.config = config or RISK_CONFIG
        self.daily_pnl = 0.0
        self.trade_count_today = 0

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.trade_count_today = 0

    def calculate_lot_size(self, balance: float, sl_pips: float, pip_value: float) -> float:
        """Calculate position size based on risk percentage and stop loss."""
        risk_amount = balance * self.config["max_risk_per_trade"]
        if sl_pips <= 0 or pip_value <= 0:
            return self.config["default_lot_size"]
        lot_size = risk_amount / (sl_pips * pip_value)
        lot_size = max(0.01, round(lot_size, 2))
        return lot_size

    def calculate_sl_tp(self, entry_price: float, signal: str, symbol_info: dict) -> tuple:
        """Calculate stop loss and take profit prices."""
        point = symbol_info.get("point", 0.0001)
        digits = symbol_info.get("digits", 5)

        sl_distance = self.config["stop_loss_pips"] * point * 10
        tp_distance = self.config["take_profit_pips"] * point * 10

        if signal == "BUY":
            sl_price = round(entry_price - sl_distance, digits)
            tp_price = round(entry_price + tp_distance, digits)
        else:
            sl_price = round(entry_price + sl_distance, digits)
            tp_price = round(entry_price - tp_distance, digits)

        return sl_price, tp_price

    def can_open_trade(self, open_positions: list, balance: float) -> tuple:
        """Check if a new trade is allowed based on risk rules."""
        # Check max open trades
        if len(open_positions) >= self.config["max_open_trades"]:
            return False, "Max open trades reached"

        # Check daily loss limit
        max_daily_loss = balance * self.config["max_daily_loss"]
        if self.daily_pnl < -max_daily_loss:
            return False, f"Daily loss limit reached ({self.daily_pnl:.2f})"

        return True, "OK"

    def update_daily_pnl(self, pnl: float):
        self.daily_pnl += pnl
        self.trade_count_today += 1
        logger.info(f"Daily P&L: {self.daily_pnl:.2f} | Trades today: {self.trade_count_today}")

    def should_close_position(self, position: dict, current_price: float) -> tuple:
        """Check if a position should be closed based on trailing stop logic."""
        entry_price = position.get("price_open", 0)
        position_type = position.get("type", "")
        profit = position.get("profit", 0)

        # Let MT5 handle SL/TP hits; this is for additional logic
        if profit > 0:
            # Could implement trailing stop updates here
            pass

        return False, ""
