"""
engine/backtester.py

Backtester pas-a-pas (stateful) qui exécute les signaux en utilisant MoneyManager.
Public API:
- backtest_signals(df, signals, money_manager, exec_price_col='Open', exec_offset=1) -> dict
- full_backtest(df, combos, config, money_manager_factory) -> pd.DataFrame

Notes:
- signals: pd.Series indexed like df with values {1, -1, 0}. Signal is read at index t and execution happens
  at row t + exec_offset using df[exec_price_col].
- money_manager_factory: callable(config) -> MoneyManager instance; permet d'injecter différents managers (tests).
- This module avoids IO; returns dicts / DataFrames for caller to persist or display.
"""
from typing import Dict, Any, List, Callable
import pandas as pd
import numpy as np

def _safe_price_at(df: pd.DataFrame, idx_pos: int, price_col: str) -> float | None:
    """Return float price at idx_pos if exists and finite, otherwise None."""
    if idx_pos < 0 or idx_pos >= len(df):
        return None
    val = df.iloc[idx_pos].get(price_col, None)
    try:
        p = float(val)
        if not np.isfinite(p) or p <= 0:
            return None
        return p
    except Exception:
        return None

def backtest_signals(df: pd.DataFrame,
                     signals: pd.Series,
                     money_manager,
                     exec_price_col: str = "Open",
                     exec_offset: int = 1) -> Dict[str, Any]:
    """
    Simulate trades given signals and a MoneyManager instance.

    Returns:
      {
        "trades": list of trade dicts from MoneyManager.trade_history,
        "equity_curve": pd.Series indexed by dates (daily equity snapshot using mark_to_market with Close),
        "final_capital": float
      }

    Behavior:
      - Read signals at index i (value in {1,-1,0}).
      - Execution happens at index i + exec_offset using df[exec_price_col].
      - For buy (1): call money_manager.try_buy(date_exec, price_exec)
      - For sell (-1): call money_manager.sell_oldest(date_exec, price_exec)
      - At the end, call money_manager.close_all(last_date, last_close) to liquidate remaining positions.
      - Equity snapshot collected after each execution event (date_exec) using mark_to_market with df['Close'] at that date (if Close exists).
    """
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column for equity snapshots.")

    trades_out: List[Dict[str, Any]] = []
    equity_dates = []
    equity_vals = []

    # Align signals index with df index; ensure same length and labels
    # signals may be a Series derived from df; we'll iterate by integer positions
    for pos in range(len(df)):
        sig = 0
        try:
            sig = int(signals.iloc[pos])
        except Exception:
            sig = 0

        if sig == 0:
            continue

        exec_pos = pos + exec_offset
        price_exec = _safe_price_at(df, exec_pos, exec_price_col)
        if price_exec is None:
            # skip this signal if execution price not available
            continue
        date_exec = df.index[exec_pos]

        if sig == 1:
            _ = money_manager.try_buy(date_exec, price_exec)
        elif sig == -1:
            _ = money_manager.sell_oldest(date_exec, price_exec)

        # snapshot equity using Close at exec_pos if available, otherwise last Close
        close_price = _safe_price_at(df, exec_pos, 'Close')
        if close_price is not None:
            equity = money_manager.mark_to_market(close_price)
            equity_dates.append(date_exec)
            equity_vals.append(equity)

    # final liquidation at last available Close
    last_date = df.index[-1]
    last_close = _safe_price_at(df, len(df)-1, 'Close')
    if last_close is not None:
        final_trades = money_manager.close_all(last_date, last_close)
        trades_out.extend(final_trades)
        # final equity after liquidation is simply cash
        equity_dates.append(last_date)
        equity_vals.append(money_manager.cash)

    # collect trade history from manager (may contain both intermediate sells and final sells)
    # some money_managers append to trade_history automatically; adapt if different implementation is used
    if hasattr(money_manager, "trade_history"):
        trades_out = getattr(money_manager, "trade_history", trades_out)

    equity_series = pd.Series(data=equity_vals, index=pd.to_datetime(equity_dates)) if equity_dates else pd.Series(dtype=float)

    final_cap = float(money_manager.cash) if hasattr(money_manager, "cash") else (equity_series.iloc[-1] if not equity_series.empty else 0.0)

    return {"trades": trades_out, "equity_curve": equity_series, "final_capital": final_cap}

def full_backtest(df: pd.DataFrame,
                  combos: List[Dict[str, Any]],
                  config: Dict[str, Any],
                  money_manager_factory: Callable[[Dict[str, Any]], Any],
                  exec_price_col: str = "Open",
                  exec_offset: int = 1) -> pd.DataFrame:
    """
    Iterate combos (each combo dict should include keys: type_short, type_long, p_short, p_long).
    For each combo:
      - compute MAs externally (caller may provide columns) OR this function expects columns named
        f"MA_{type}{period}" to already exist in df.
      - generate signals using short and long column names and call backtest_signals
      - collect metrics into a results list and return pd.DataFrame

    Returned DataFrame columns include:
      ['combination','nb_trades','final_capital','total_profit','total_return_pct']
    """
    import engine.vector_backtest as vmod  # local import to avoid top-level dependency cycles
    results = []
    for combo in combos:
        t_short = combo['type_short']
        t_long = combo['type_long']
        p_short = combo['p_short']
        p_long = combo['p_long']
        short_col = f"MA_{t_short}{p_short}"
        long_col = f"MA_{t_long}{p_long}"

        if short_col not in df.columns or long_col not in df.columns:
            # skip combos that do not have precomputed columns
            continue

        signals = vmod.generate_cross_signals(df, short_col, long_col)
        mm = money_manager_factory(config)
        outcome = backtest_signals(df, signals, mm, exec_price_col=exec_price_col, exec_offset=exec_offset)
        trades = outcome.get("trades", [])
        final_cap = outcome.get("final_capital", mm.cash)
        total_profit = final_cap - mm.initial_capital if hasattr(mm, "initial_capital") else final_cap
        nb_trades = len(trades) if isinstance(trades, list) else 0
        total_return_pct = (final_cap / mm.initial_capital - 1.0) * 100.0 if hasattr(mm, "initial_capital") else 0.0

        results.append({
            "combination": f"{t_short}{p_short}/{t_long}{p_long}",
            "nb_trades": int(nb_trades),
            "final_capital": float(final_cap),
            "total_profit": float(total_profit),
            "total_return_pct": float(total_return_pct),
            "trades": trades
        })

    return pd.DataFrame(results)