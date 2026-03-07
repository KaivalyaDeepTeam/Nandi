"""
Trade execution module with advanced filtering.
"""

import logging
from src.mt5_connector import MT5Connector
from src.risk_manager import RiskManager
from src.trade_filters import apply_all_filters, calculate_adaptive_sl_tp

logger = logging.getLogger(__name__)


class TradeExecutor:
    """Executes trades on MT5 with multi-layer filtering."""

    def __init__(self, connector: MT5Connector, risk_manager: RiskManager):
        self.mt5 = connector
        self.risk = risk_manager

    def execute_signal(self, symbol: str, prediction: dict,
                       df_feat=None, current_hour: int = 12) -> dict:
        """Execute a trade with all filters applied."""

        if not prediction["should_trade"]:
            return {"action": "SKIP", "reason": "Model says no"}

        # Apply advanced filters
        if df_feat is not None:
            filter_result = apply_all_filters(
                prediction["signal"], prediction, df_feat, current_hour
            )
            if not filter_result["filters_pass"]:
                reasons = ", ".join(filter_result["reasons"])
                logger.info(f"Filtered out: {reasons}")
                return {"action": "FILTERED", "reason": reasons}
        else:
            filter_result = {"confidence_boost": 0}

        # Get current state
        account = self.mt5.get_account_info()
        positions = self.mt5.get_open_positions()
        balance = account["balance"]

        # Check risk rules
        can_trade, reason = self.risk.can_open_trade(positions, balance)
        if not can_trade:
            logger.info(f"Trade blocked by risk manager: {reason}")
            return {"action": "BLOCKED", "reason": reason}

        # Check for conflicting position on same symbol
        for pos in positions:
            if pos.get("symbol") == symbol:
                pos_type = "BUY" if pos.get("type") == 0 else "SELL"
                if pos_type == prediction["signal"]:
                    return {"action": "SKIP", "reason": "Existing same-direction position"}

        # Get symbol info and current price
        symbol_info = self.mt5.get_symbol_info(symbol)
        tick = self.mt5.get_tick(symbol)

        entry_price = tick["ask"] if prediction["signal"] == "BUY" else tick["bid"]

        # Adaptive SL/TP based on ATR
        atr = df_feat.iloc[-1]["atr"] if df_feat is not None and "atr" in df_feat.columns else 0.0005
        digits = int(symbol_info.get("digits", 5))
        sl_price, tp_price, sl_pips, tp_pips = calculate_adaptive_sl_tp(
            atr, prediction["signal"], entry_price, digits
        )

        # Position sizing based on adaptive SL
        pip_value = symbol_info.get("trade_tick_value", 10)
        lot_size = self.risk.calculate_lot_size(balance, sl_pips, pip_value)

        # Execute trade
        try:
            conf = prediction["confidence"] + filter_result["confidence_boost"]
            result = self.mt5.open_trade(
                symbol=symbol,
                order_type=prediction["signal"],
                lot_size=lot_size,
                sl_price=sl_price,
                tp_price=tp_price,
                comment=f"FX|c={conf:.0%}|R{prediction.get('regime','-')}",
            )
            logger.info(
                f"TRADE: {prediction['signal']} {lot_size} {symbol} @ {entry_price} "
                f"SL={sl_price}({sl_pips:.0f}p) TP={tp_price}({tp_pips:.0f}p) "
                f"conf={conf:.0%}"
            )
            return {
                "action": "EXECUTED",
                "signal": prediction["signal"],
                "lot_size": lot_size,
                "entry_price": entry_price,
                "sl": sl_price,
                "tp": tp_price,
                "sl_pips": sl_pips,
                "tp_pips": tp_pips,
                "ticket": result.get("ticket"),
            }
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {"action": "ERROR", "reason": str(e)}

    def manage_positions(self, symbol: str):
        """Monitor and manage open positions."""
        positions = self.mt5.get_open_positions()
        for pos in positions:
            if pos.get("symbol") != symbol:
                continue
            tick = self.mt5.get_tick(symbol)
            current_price = tick["bid"] if pos.get("type") == 0 else tick["ask"]
            should_close, reason = self.risk.should_close_position(pos, current_price)
            if should_close:
                self.mt5.close_trade(pos["ticket"])
                self.risk.update_daily_pnl(pos.get("profit", 0))
                logger.info(f"Closed position {pos['ticket']}: {reason}")

    def close_all(self, symbol: str = None):
        """Close all open positions."""
        positions = self.mt5.get_open_positions()
        for pos in positions:
            if symbol and pos.get("symbol") != symbol:
                continue
            try:
                self.mt5.close_trade(pos["ticket"])
                self.risk.update_daily_pnl(pos.get("profit", 0))
                logger.info(f"Closed position {pos['ticket']} P&L={pos.get('profit', 0):.2f}")
            except Exception as e:
                logger.error(f"Failed to close {pos['ticket']}: {e}")

    def get_status(self) -> dict:
        try:
            account = self.mt5.get_account_info()
            positions = self.mt5.get_open_positions()
            return {
                "balance": account["balance"],
                "equity": account["equity"],
                "margin": account.get("margin", 0),
                "free_margin": account.get("margin_free", 0),
                "profit": account.get("profit", 0),
                "open_positions": len(positions),
                "daily_pnl": self.risk.daily_pnl,
            }
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {}
