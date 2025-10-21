import pytest
from engine.money_manager import MoneyManager

def test_buy_and_sell_sequence():
    mm = MoneyManager(initial_capital=1000.0, max_alloc_pct=0.5, commission=0.0, slippage=0.0)
    # max allocation = 50% -> 500€ per order, price 100 -> qty 5
    pos = mm.try_buy("2020-01-02", 100.0)
    assert pos is not None
    assert pos["quantity"] == 5
    # cash should be reduced by 500 (no fees/slippage)
    assert pytest.approx(mm.cash, rel=1e-6) == 500.0

    # open another position: 50% of remaining cash = 250 -> qty 2 at price 100
    pos2 = mm.try_buy("2020-01-03", 100.0)
    assert pos2 is not None
    assert pos2["quantity"] == 2
    # cash now 500 - 200 = 300
    assert pytest.approx(mm.cash, rel=1e-6) == 300.0

    # sell oldest (FIFO) at price 110
    trade = mm.sell_oldest("2020-01-04", 110.0)
    assert trade is not None
    assert trade["quantity"] == 5
    # proceeds 5*110 = 550 -> cash was 300 -> now 850
    assert pytest.approx(mm.cash, rel=1e-6) == 850.0

def test_insufficient_cash_for_min_share():
    mm = MoneyManager(initial_capital=50.0, max_alloc_pct=0.5)
    # max allocation 25€, price 60 -> cannot buy and cash < price -> None
    res = mm.try_buy("2020-01-01", 60.0)
    assert res is None

def test_minimum_one_share_if_affordable():
    mm = MoneyManager(initial_capital=100.0, max_alloc_pct=0.1)  # max order 10€
    # price 20 less than cash -> allow single share despite allocation
    pos = mm.try_buy("2020-01-01", 20.0)
    assert pos is not None
    assert pos["quantity"] == 1
    assert mm.cash < 100.0

def test_commission_and_slippage_apply():
    mm = MoneyManager(initial_capital=1000.0, max_alloc_pct=0.5, commission=1.0, slippage=0.01)
    pos = mm.try_buy("2020-01-02", 100.0)
    # expected invested = qty*price + slip + commission = 5*100 + 0.01*100*5 + 1 = 500 + 5 + 1 = 506
    assert pos is not None
    assert pytest.approx(pos["invested"], rel=1e-6) == 506.0
    # cash reduced accordingly
    assert pytest.approx(mm.cash, rel=1e-6) == 494.0

    # sell at 110 -> proceeds = 5*110 - slip - commission = 550 - 5.5 - 1 = 543.5
    trade = mm.sell_oldest("2020-01-10", 110.0)
    assert pytest.approx(trade["proceeds"], rel=1e-6) == 543.5
    # profit = proceeds - invested = 543.5 - 506 = 37.5
    assert pytest.approx(trade["profit"], rel=1e-6) == 37.5

def test_close_all_and_mark_to_market():
    mm = MoneyManager(initial_capital=1000.0, max_alloc_pct=0.5)
    mm.try_buy("2020-01-02", 100.0)  # qty 5
    mm.try_buy("2020-01-03", 50.0)   # qty floor(250/50)=5
    # mark to market at price 60: open value = (5+5)*60 = 600, cash = initial - (5*100 + 5*50) = depends on allocation but mtm should be finite
    equity = mm.mark_to_market(60.0)
    assert equity > 0
    trades = mm.close_all("2020-12-31", 60.0)
    # after close_all there should be no open positions
    assert mm.snapshot()["open_positions_count"] == 0
    assert isinstance(trades, list)