import pandas as pd
import numpy as np
from engine.vector_backtest import generate_cross_signals, vector_scan
from indicators.ma import calculate_ma

def make_price_series():
    # Construct a price series with clear up then down then up moves to create several crossings
    prices = [
        10, 11, 12, 13, 14,  # uptrend
        13, 12, 11,          # downtrend
        12, 13, 14, 15,      # uptrend again
        14, 13, 12           # downtrend
    ]
    idx = pd.date_range("2021-01-01", periods=len(prices))
    return pd.DataFrame({"Close": prices}, index=idx)

def test_generate_cross_signals_basic():
    df = make_price_series()
    # create simple moving averages columns to test generator directly
    df['SMA2'] = calculate_ma(df['Close'], 2, "SMA")
    df['SMA3'] = calculate_ma(df['Close'], 3, "SMA")
    signals = generate_cross_signals(df, 'SMA2', 'SMA3')
    assert isinstance(signals, pd.Series)
    # values limited to -1/0/1
    assert set(np.unique(signals.values)).issubset({-1,0,1})
    # expect at least one buy and one sell in this crafted series
    assert (signals == 1).sum() >= 1
    assert (signals == -1).sum() >= 1

def test_vector_scan_returns_dataframe_and_filters_min_trades():
    df = make_price_series()
    periods = [2,3,5]
    ma_types = ["SMA"]
    metrics_cfg = {"metric": "mean_return_per_trade", "min_trades": 1}
    res = vector_scan(df, periods, ma_types, ratio_min=1.1, ratio_max=10.0, metrics_config=metrics_cfg)
    assert isinstance(res, pd.DataFrame)
    # Since min_trades=1, expect some combos returned
    assert len(res) > 0

    # test filtering by raising min_trades to a high value that excludes all combos
    metrics_cfg2 = {"metric": "mean_return_per_trade", "min_trades": 99}
    res2 = vector_scan(df, periods, ma_types, ratio_min=1.1, ratio_max=10.0, metrics_config=metrics_cfg2)
    assert isinstance(res2, pd.DataFrame)
    assert res2.empty

def test_vector_scan_content_consistency():
    df = make_price_series()
    periods = [2,3]
    ma_types = ["SMA"]
    metrics_cfg = {"metric": "mean_return_per_trade", "min_trades": 0}
    res = vector_scan(df, periods, ma_types, ratio_min=1.0, ratio_max=10.0, metrics_config=metrics_cfg)
    # For SMA2/SMA3 there should be one combination p_short=2 p_long=3
    combs = res['combination'].tolist()
    assert any("SMA2/SMA3" in c or "SMA2/SMA3" == c for c in combs) or any("SMA2/SMA3" == c.replace(" ", "") for c in combs)
    # nb_trades non-negative
    assert (res['nb_trades'] >= 0).all()