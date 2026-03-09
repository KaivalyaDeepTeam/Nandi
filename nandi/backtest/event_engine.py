"""Event-driven backtester with realistic execution simulation."""

import logging
import numpy as np
from nandi.config import LEVERAGE, LOOKBACK_WINDOW, INITIAL_BALANCE
from nandi.environment.market_sim import MarketSimulator
from nandi.utils.metrics import sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown, profit_factor, win_rate

logger = logging.getLogger(__name__)


class EventDrivenBacktester:
    """Processes bars one at a time with MarketSimulator for realistic fills."""

    def __init__(self, agent, pair_name="eurusd", initial_balance=INITIAL_BALANCE):
        self.agent = agent
        self.pair_name = pair_name
        self.initial_balance = initial_balance
        self.market_sim = MarketSimulator(pair_name=pair_name)

    def run(self, features, prices, lookback=LOOKBACK_WINDOW):
        """Run backtest on features/prices arrays.

        Returns dict with equity_curve, daily_returns, trades, and metrics.
        """
        equity = self.initial_balance
        peak_equity = equity
        position = 0.0

        equity_curve = [equity]
        daily_returns = []
        trades = []
        trade_entry = None

        for i in range(lookback, len(features) - 1):
            market_state = features[i - lookback:i]
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            vol_regime = float(np.std(features[max(0, i-10):i, 0])) if i > 10 else 1.0

            pos_info = np.array([
                position,
                equity / self.initial_balance - 1.0,
                dd,
                vol_regime,
            ], dtype=np.float32)

            # Get agent decision
            action, _, _, uncertainty = self.agent.get_action(
                market_state, pos_info, deterministic=True
            )

            # Uncertainty gating
            if uncertainty > 0.7:
                action *= 0.3
            # Risk scaling
            if dd > 0.08:
                action *= 0.5
            if dd > 0.15:
                action = 0.0

            action = float(np.clip(action, -1.0, 1.0))

            # Compute PnL
            price_now = prices[i]
            price_next = prices[i + 1]
            market_ret = (price_next - price_now) / price_now

            pnl = position * market_ret * equity * LEVERAGE

            # Realistic cost from MarketSimulator
            cost = self.market_sim.get_total_cost(
                price_now, position, action, abs(market_ret)
            ) * equity

            # Track trades
            pos_change = abs(action - position)
            if pos_change > 0.05:
                # Close previous trade
                if trade_entry is not None:
                    trade_pnl = equity - trade_entry["equity"]
                    trades.append({
                        "entry_idx": trade_entry["idx"],
                        "exit_idx": i,
                        "direction": trade_entry["direction"],
                        "pnl": trade_pnl,
                        "r_multiple": trade_pnl / self.initial_balance if self.initial_balance > 0 else 0,
                    })
                # Open new trade
                if abs(action) > 0.05:
                    trade_entry = {
                        "idx": i,
                        "direction": 1 if action > 0 else -1,
                        "equity": equity,
                    }
                else:
                    trade_entry = None

            equity += pnl - cost
            peak_equity = max(peak_equity, equity)

            daily_ret = (pnl - cost) / (equity - pnl + cost) if (equity - pnl + cost) > 0 else 0
            daily_returns.append(daily_ret)
            equity_curve.append(equity)

            position = action

        # Close last trade if open
        if trade_entry is not None:
            trades.append({
                "entry_idx": trade_entry["idx"],
                "exit_idx": len(features) - 1,
                "direction": trade_entry["direction"],
                "pnl": equity - trade_entry["equity"],
                "r_multiple": (equity - trade_entry["equity"]) / self.initial_balance,
            })

        returns = np.array(daily_returns)
        trade_returns = [t["pnl"] for t in trades]

        metrics = {
            "total_return_pct": (equity / self.initial_balance - 1) * 100,
            "sharpe": sharpe_ratio(returns),
            "sortino": sortino_ratio(returns),
            "calmar": calmar_ratio(returns),
            "max_drawdown": max_drawdown(returns),
            "profit_factor": profit_factor(np.array(trade_returns)) if trade_returns else 0,
            "win_rate": win_rate(np.array(trade_returns)) if trade_returns else 0,
            "n_trades": len(trades),
            "avg_r_multiple": float(np.mean([t["r_multiple"] for t in trades])) if trades else 0,
        }

        return {
            "equity_curve": np.array(equity_curve),
            "daily_returns": returns,
            "trades": trades,
            "metrics": metrics,
        }
