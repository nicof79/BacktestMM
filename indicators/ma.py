"""
indicators/ma.py
Implementation of moving averages: SMA, EMA, WMA, HMA.
Public API:
    calculate_ma(series: pd.Series, period: int, ma_type: str) -> pd.Series
"""
from typing import Optional
import numpy as np
import pandas as pd


def _wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    def apply_wma(window):
        if len(window) < period:
            return np.nan
        return float(np.dot(window, weights) / weights.sum())
    return series.rolling(window=period, min_periods=period).apply(lambda x: apply_wma(x), raw=True)


def calculate_ma(series: pd.Series, period: int, ma_type: str = "SMA") -> pd.Series:
    """
    Calculate a moving average series.

    Args:
        series: pd.Series of prices (indexed by date).
        period: int >= 1
        ma_type: one of "SMA", "EMA", "WMA", "HMA" (case-insensitive)

    Returns:
        pd.Series of same index with moving average values (NaN where undefined).

    Raises:
        ValueError on invalid period or unknown ma_type.
    """
    if not isinstance(period, int) or period < 1:
        raise ValueError("period must be integer >= 1")

    ma_type_norm = (ma_type or "SMA").upper()

    if ma_type_norm == "SMA":
        return series.rolling(window=period, min_periods=period).mean()
    if ma_type_norm == "EMA":
        # span=period produces an EMA with the same effective window behavior
        return series.ewm(span=period, adjust=False).mean()
    if ma_type_norm == "WMA":
        return _wma(series, period)
    if ma_type_norm == "HMA":
        # HMA: WMA(2*WMA(period/2) - WMA(period)), sqrt(period)
        half = max(1, int(period / 2))
        sq = max(1, int(np.sqrt(period)))
        wma_half = _wma(series, half)
        wma_full = _wma(series, period)
        diff = 2 * wma_half - wma_full
        # diff may contain NaNs; calculate WMA on diff
        return _wma(diff, sq)
    raise ValueError(f"Unknown MA type: {ma_type}")