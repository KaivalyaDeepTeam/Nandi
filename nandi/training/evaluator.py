"""
Portfolio Backtest Evaluator — runs OOS evaluation across all pairs with portfolio metrics.
"""

import logging
import numpy as np

from nandi.config import (
    LOOKBACK_WINDOW, INITIAL_BALANCE, LEVERAGE,
    TRANSACTION_COST_BPS, RISK_LIMITS,
)
from nandi.utils.metrics import (
    sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown_from_equity,
    profit_factor, win_rate,
)

logger = logging.getLogger(__name__)


class BacktestEvaluator:
    """Run OOS backtest for a single pair and compute detailed metrics."""

    def __init__(self, agent, pair_name="unknown"):
        self.agent = agent
        self.pair_name = pair_name

    def evaluate(self, test_features, test_prices, test_dates=None):
        """Run backtest on unseen test data.

        Returns:
            dict with equity curve, trades, and performance metrics.
        """
        lookback = LOOKBACK_WINDOW
        equity = INITIAL_BALANCE
        peak_equity = INITIAL_BALANCE
        position = 0.0
        trades = []
        equity_curve = [equity]
        daily_returns = []

        for i in range(lookback, len(test_features) - 1):
            market_state = test_features[i - lookback:i]
            dd = (peak_equity - equity) / peak_equity
            position_info = np.array([
                position,
                equity / INITIAL_BALANCE - 1,
                dd,
                float(np.std(test_features[max(0, i - 10):i, 0])) if i > 10 else 1.0,
            ], dtype=np.float32)

            action, _, _, uncertainty = self.agent.get_action(
                market_state, position_info, deterministic=True
            )

            if uncertainty > 0.7:
                action *= 0.3
            if dd > RISK_LIMITS["scale_down_threshold"]:
                action *= 0.5
            if dd > RISK_LIMITS["max_drawdown"]:
                action = 0.0

            action = float(np.clip(action, -1.0, 1.0))

            price_now = test_prices[i]
            price_next = test_prices[i + 1]
            market_ret = (price_next - price_now) / price_now

            pnl = position * market_ret * equity * LEVERAGE
            cost = abs(action - position) * equity * TRANSACTION_COST_BPS / 10000

            prev_equity = equity
            equity += pnl - cost
            peak_equity = max(peak_equity, equity)
            equity_curve.append(equity)

            daily_ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            daily_returns.append(daily_ret)

            if abs(action - position) > 0.05:
                trade_info = {
                    "action": "LONG" if action > 0.1 else ("SHORT" if action < -0.1 else "FLAT"),
                    "position": round(action, 3),
                    "price": price_now,
                    "pnl": round(pnl - cost, 2),
                    "equity": round(equity, 2),
                    "uncertainty": round(uncertainty, 3),
                    "drawdown": round(dd, 4),
                }
                if test_dates is not None:
                    trade_info["date"] = test_dates[i]
                trades.append(trade_info)

            position = action

        # Compute metrics
        equity_arr = np.array(equity_curve)
        returns_arr = np.array(daily_returns) if daily_returns else np.array([0.0])
        trade_pnls = np.array([t["pnl"] for t in trades]) if trades else np.array([0.0])

        metrics = {
            "pair": self.pair_name,
            "total_return_pct": (equity / INITIAL_BALANCE - 1) * 100,
            "sharpe": sharpe_ratio(returns_arr),
            "sortino": sortino_ratio(returns_arr),
            "calmar": calmar_ratio(returns_arr),
            "max_drawdown": max_drawdown_from_equity(equity_arr),
            "profit_factor": profit_factor(trade_pnls),
            "win_rate": win_rate(trade_pnls),
            "n_trades": len(trades),
            "final_equity": equity,
        }

        return {
            "metrics": metrics,
            "trades": trades,
            "equity_curve": equity_curve,
            "daily_returns": daily_returns,
        }


class PortfolioEvaluator:
    """Evaluate portfolio of multiple pairs together."""

    def __init__(self, pair_evaluators):
        """
        Args:
            pair_evaluators: {pair_name: BacktestEvaluator}
        """
        self.evaluators = pair_evaluators

    def evaluate_portfolio(self, pair_data_dict):
        """Run all pair backtests and compute portfolio-level metrics.

        Args:
            pair_data_dict: {pair: {"test_features": ..., "test_prices": ..., "test_dates": ...}}
        """
        pair_results = {}
        all_equity_curves = []

        for pair, evaluator in self.evaluators.items():
            data = pair_data_dict[pair]
            result = evaluator.evaluate(
                data["test_features"], data["test_prices"],
                data.get("test_dates"),
            )
            pair_results[pair] = result

            # Normalize equity curve to returns for aggregation
            eq = np.array(result["equity_curve"])
            daily_rets = np.diff(eq) / eq[:-1]
            all_equity_curves.append(daily_rets)

        # Portfolio = average of per-pair daily returns
        min_len = min(len(r) for r in all_equity_curves)
        aligned = np.array([r[:min_len] for r in all_equity_curves])
        portfolio_returns = aligned.mean(axis=0)

        # Portfolio equity curve
        portfolio_equity = INITIAL_BALANCE * np.cumprod(1 + portfolio_returns)
        portfolio_equity = np.insert(portfolio_equity, 0, INITIAL_BALANCE)

        portfolio_metrics = {
            "total_return_pct": (portfolio_equity[-1] / INITIAL_BALANCE - 1) * 100,
            "sharpe": sharpe_ratio(portfolio_returns),
            "sortino": sortino_ratio(portfolio_returns),
            "calmar": calmar_ratio(portfolio_returns),
            "max_drawdown": max_drawdown_from_equity(portfolio_equity),
            "n_pairs": len(pair_results),
        }

        # Per-pair summary
        pair_summaries = {
            pair: result["metrics"] for pair, result in pair_results.items()
        }

        return {
            "portfolio_metrics": portfolio_metrics,
            "pair_metrics": pair_summaries,
            "portfolio_equity_curve": portfolio_equity.tolist(),
            "pair_results": pair_results,
        }
