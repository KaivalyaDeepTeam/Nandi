"""ATR-based trailing stop manager for live positions."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class TrailingStopManager:
    """Manages ATR-based trailing stops for open positions."""

    def __init__(self, atr_multiplier=2.0):
        self.atr_multiplier = atr_multiplier
        self.trails = {}  # ticket -> {"best_price": float, "trail_distance": float, "direction": int}

    def register(self, ticket, entry_price, direction, atr):
        """Register a new position for trailing."""
        trail_distance = atr * self.atr_multiplier
        self.trails[ticket] = {
            "best_price": entry_price,
            "trail_distance": trail_distance,
            "direction": direction,  # +1 long, -1 short
        }
        logger.debug(f"Registered trailing stop for ticket {ticket}: "
                     f"distance={trail_distance:.5f}")

    def update(self, ticket, current_price, atr=None):
        """Update trailing stop for a position. Returns new SL price or None."""
        if ticket not in self.trails:
            return None

        state = self.trails[ticket]
        if atr is not None:
            state["trail_distance"] = atr * self.atr_multiplier

        direction = state["direction"]
        if direction > 0:  # long position
            state["best_price"] = max(state["best_price"], current_price)
            new_sl = state["best_price"] - state["trail_distance"]
            return new_sl
        else:  # short position
            state["best_price"] = min(state["best_price"], current_price)
            new_sl = state["best_price"] + state["trail_distance"]
            return new_sl

    def should_close(self, ticket, current_price):
        """Check if trailing stop has been hit."""
        if ticket not in self.trails:
            return False
        state = self.trails[ticket]
        sl = self.update(ticket, current_price)
        if sl is None:
            return False
        if state["direction"] > 0:
            return current_price <= sl
        else:
            return current_price >= sl

    def remove(self, ticket):
        """Remove trailing stop tracking for a closed position."""
        self.trails.pop(ticket, None)

    def get_all_stops(self):
        """Get current stop levels for all tracked positions."""
        stops = {}
        for ticket, state in self.trails.items():
            if state["direction"] > 0:
                stops[ticket] = state["best_price"] - state["trail_distance"]
            else:
                stops[ticket] = state["best_price"] + state["trail_distance"]
        return stops
