"""Portfolio performance metrics — Sharpe, Sortino, Calmar, Information Ratio."""

import numpy as np


def sharpe_ratio(returns, risk_free_rate=0.0, annualize=252):
    """Annualized Sharpe ratio from daily returns array."""
    if len(returns) < 2:
        return 0.0
    excess = np.asarray(returns) - risk_free_rate / annualize
    std = np.std(excess, ddof=1)
    if std < 1e-10:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(annualize))


def sortino_ratio(returns, risk_free_rate=0.0, annualize=252):
    """Annualized Sortino ratio (downside deviation only)."""
    if len(returns) < 2:
        return 0.0
    excess = np.asarray(returns) - risk_free_rate / annualize
    downside = excess[excess < 0]
    if len(downside) < 1:
        return float("inf") if np.mean(excess) > 0 else 0.0
    downside_std = np.sqrt(np.mean(downside ** 2))
    if downside_std < 1e-10:
        return 0.0
    return float(np.mean(excess) / downside_std * np.sqrt(annualize))


def calmar_ratio(returns, annualize=252):
    """Calmar ratio: annualized return / max drawdown."""
    if len(returns) < 2:
        return 0.0
    dd = max_drawdown(returns)
    if dd < 1e-10:
        return float("inf") if np.mean(returns) > 0 else 0.0
    ann_ret = np.mean(returns) * annualize
    return float(ann_ret / dd)


def max_drawdown(returns):
    """Maximum drawdown from a returns series."""
    if len(returns) < 1:
        return 0.0
    equity = np.cumprod(1 + np.asarray(returns))
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    return float(np.max(dd))


def max_drawdown_from_equity(equity):
    """Maximum drawdown from an equity curve."""
    equity = np.asarray(equity)
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    return float(np.max(dd))


def information_ratio(returns, benchmark_returns, annualize=252):
    """Information ratio: excess return vs benchmark / tracking error."""
    if len(returns) < 2:
        return 0.0
    excess = np.asarray(returns) - np.asarray(benchmark_returns)
    te = np.std(excess, ddof=1)
    if te < 1e-10:
        return 0.0
    return float(np.mean(excess) / te * np.sqrt(annualize))


def profit_factor(returns):
    """Gross profit / gross loss."""
    returns = np.asarray(returns)
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    if gross_loss < 1e-10:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def win_rate(returns):
    """Percentage of positive returns."""
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0
    return float(np.sum(returns > 0) / len(returns))
