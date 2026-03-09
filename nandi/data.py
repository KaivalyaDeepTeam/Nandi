"""Backward compatibility — imports from new locations."""
from nandi.data.features import compute_features  # noqa: F401
from nandi.data.manager import download_forex_data, DataManager  # noqa: F401


def prepare_data(lookback_window=60, test_months=6, years=20):
    """Legacy single-pair data preparation."""
    dm = DataManager(pairs=["eurusd"], years=years, test_months=test_months,
                     lookback_window=lookback_window)
    data = dm.prepare_pair("eurusd")
    return data
