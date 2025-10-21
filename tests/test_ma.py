import pandas as pd
import numpy as np
import pytest
from indicators.ma import calculate_ma

@pytest.fixture
def simple_series():
    # simple increasing series 1..20
    return pd.Series(np.arange(1, 21), index=pd.date_range("2020-01-01", periods=20))

def test_sma_basic(simple_series):
    s = simple_series
    sma_5 = calculate_ma(s, 5, "SMA")
    # first 4 values must be NaN, 5th must equal mean(1..5)=3
    assert np.isnan(sma_5.iloc[0])
    assert np.isnan(sma_5.iloc[3])
    assert sma_5.iloc[4] == pytest.approx(3.0)

def test_ema_basic(simple_series):
    s = simple_series
    ema_3 = calculate_ma(s, 3, "EMA")
    # EMA should be defined from the first value (pandas ewm returns non-NaN)
    assert not np.isnan(ema_3.iloc[0])
    # Check monotonicity on strictly increasing series
    assert ema_3.is_monotonic_increasing

def test_wma_basic(simple_series):
    s = simple_series
    wma_4 = calculate_ma(s, 4, "WMA")
    # first 3 values NaN, 4th equals weighted average of 1,2,3,4
    expected = (1*1 + 2*2 + 3*3 + 4*4) / (1+2+3+4)
    assert np.isnan(wma_4.iloc[2])
    assert wma_4.iloc[3] == pytest.approx(expected)

def test_hma_basic(simple_series):
    s = simple_series
    hma_9 = calculate_ma(s, 9, "HMA")
    # HMA requires multiple steps; ensure result is a pd.Series and has correct index
    assert isinstance(hma_9, pd.Series)
    assert list(hma_9.index) == list(s.index)

def test_invalid_period():
    with pytest.raises(ValueError):
        calculate_ma(pd.Series([1, 2, 3]), 0, "SMA")

def test_unknown_type(simple_series):
    with pytest.raises(ValueError):
        calculate_ma(simple_series, 5, "UNKNOWN")