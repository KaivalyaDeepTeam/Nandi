"""Stress testing against historical crisis periods."""

import logging
import numpy as np
import pandas as pd
from nandi.utils.metrics import sharpe_ratio, max_drawdown

logger = logging.getLogger(__name__)


class StressTester:
    """Tests strategy performance during historical crisis periods."""

    CRISIS_PERIODS = {
        "gfc_2008": ("2008-09-01", "2009-03-31"),
        "flash_crash_2010": ("2010-05-01", "2010-06-30"),
        "snb_2015": ("2015-01-15", "2015-02-28"),
        "brexit_2016": ("2016-06-20", "2016-07-15"),
        "covid_2020": ("2020-03-01", "2020-04-30"),
        "rate_hike_2022": ("2022-06-01", "2022-10-31"),
    }

    def test_crisis(self, daily_returns, dates, crisis_name):
        """Test performance during a specific crisis period."""
        if crisis_name not in self.CRISIS_PERIODS:
            return None

        start, end = self.CRISIS_PERIODS[crisis_name]
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))

        if mask.sum() < 5:
            return None

        crisis_returns = daily_returns[mask]
        return {
            "period": f"{start} to {end}",
            "n_days": int(mask.sum()),
            "total_return_pct": float((np.prod(1 + crisis_returns) - 1) * 100),
            "sharpe": sharpe_ratio(crisis_returns),
            "max_drawdown": max_drawdown(crisis_returns),
            "worst_day": float(np.min(crisis_returns)),
            "best_day": float(np.max(crisis_returns)),
            "vol": float(np.std(crisis_returns) * np.sqrt(252)),
        }

    def test_all_crises(self, daily_returns, dates):
        """Test performance during all known crisis periods."""
        results = {}
        for name in self.CRISIS_PERIODS:
            result = self.test_crisis(daily_returns, dates, name)
            if result is not None:
                results[name] = result
                logger.info(f"  {name}: return={result['total_return_pct']:+.2f}%, "
                           f"DD={result['max_drawdown']:.2%}")

        if not results:
            logger.warning("No crisis periods found in date range")

        return results
