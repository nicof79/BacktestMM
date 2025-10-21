import pandas as pd
import numpy as np
from engine.backtester import backtest_signals, full_backtest
from engine.money_manager import MoneyManager
from indicators.ma import calculate_ma
from engine.vector_backtest import generate_cross_signals

def make_simple_price_df():
    # price series with enough bars to allow next-row execution
    prices = [10, 11, 12, 11, 10, 9, 10, 11, 12, 13]
    opens =  [10, 11, 11.5, 12, 11, 10, 9.5, 10, 11, 12]
    idx = pd.date_range("2021-01-01", periods=len(prices))
    return pd.DataFrame({"Open": opens, "Close": prices}, index=idx)

def money_manager_factory(cfg):
    # simple factory for tests using small allocation so multiple buys possible
    return MoneyManager(initial_capital=1000.0, max_alloc_pct=0.5, commission=0.0, slippage=0.0)

def test_backtest_signals_basic_buy_sell_sequence():
    df = make_simple_price_df()
    # create a signal series: buy at index 1 (exec at 2 open), sell at index 3 (exec at 4 open)
    signals = pd.Series(0, index=df.index)
    signals.iloc[1] = 1
    signals.iloc[3] = -1
    mm = money_manager_factory({})
    outcome = backtest_signals(df, signals, mm, exec_price_col="Open", exec_offset=1)
    # After buy at open idx2 and sell at open idx4 with allocation rules, expect at least one trade in history
    trades = outcome["trades"]
    assert isinstance(trades, list)
    assert len(trades) >= 1
    # final capital should be a finite number
    assert np.isfinite(outcome["final_capital"])

def test_full_backtest_with_precomputed_mas():
    df = make_simple_price_df()
    # precompute SMA2 and SMA3 columns
    df["MA_SMA2"] = calculate_ma(df["Close"], 2, "SMA")
    df["MA_SMA3"] = calculate_ma(df["Close"], 3, "SMA")
    combos = [{"type_short":"SMA","type_long":"SMA","p_short":2,"p_long":3}]
    cfg = {}
    res_df = full_backtest(df, combos, cfg, money_manager_factory, exec_price_col="Open", exec_offset=1)
    assert isinstance(res_df, pd.DataFrame)
    assert len(res_df) == 1
    row = res_df.iloc[0]
    assert "final_capital" in row and np.isfinite(row["final_capital"])
    assert row["nb_trades"] >= 0