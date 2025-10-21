import pytest
import pandas as pd
import numpy as np
from metrics.performance import trade_stats, equity_curve_metrics, format_currency

def test_trade_stats_empty():
    res = trade_stats([])
    assert res["nb_trades"] == 0
    assert res["win_rate_pct"] == 0.0

def test_trade_stats_basic():
    trades = [
        {"profit": 10.0, "buy_price": 100.0, "sell_price": 110.0},
        {"profit": -5.0, "buy_price": 200.0, "sell_price": 190.0},
        {"profit": 20.0, "buy_price": 50.0, "sell_price": 60.0}
    ]
    res = trade_stats(trades)
    assert res["nb_trades"] == 3
    assert res["nb_wins"] == 2
    assert res["win_rate_pct"] == pytest.approx(2/3 * 100.0)

def test_equity_curve_metrics_basic():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    eq = pd.Series([1000.0, 1100.0, 1050.0, 1200.0, 1150.0], index=idx)
    res = equity_curve_metrics(eq)
    assert "total_return_pct" in res
    assert res["total_return_pct"] == pytest.approx((1150.0 / 1000.0 - 1.0) * 100.0)

def test_format_currency():
    assert format_currency(1234.56) == "1 234,56€"
    assert format_currency(1000000) == "1 000 000,00€"