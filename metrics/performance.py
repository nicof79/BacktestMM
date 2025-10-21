"""
metrics/performance.py

Calculs de métriques pour trades et séries d'equity.

API publique :
- trade_stats(trades: list[dict]) -> dict
- equity_curve_metrics(equity_series: pd.Series) -> dict
- format_currency(value: float, decimals: int = 2) -> str
"""
from typing import List, Dict, Any
import numpy as np
import pandas as pd

def trade_stats(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calcule statistiques basiques à partir d'une liste de trades.
    Trade attendu avec champs numériques 'profit' ou 'profit_€' ou 'pct_change'.

    Retour:
      {
        nb_trades: int,
        nb_wins: int,
        nb_losses: int,
        win_rate_pct: float,
        avg_gain_pct: float,
        avg_loss_pct: float,
        avg_profit_absolute: float,
        total_profit_absolute: float
      }
    """
    if not trades:
        return {
            "nb_trades": 0,
            "nb_wins": 0,
            "nb_losses": 0,
            "win_rate_pct": 0.0,
            "avg_gain_pct": 0.0,
            "avg_loss_pct": 0.0,
            "avg_profit_absolute": 0.0,
            "total_profit_absolute": 0.0
        }

    profits_abs = []
    profits_pct = []
    for t in trades:
        if "profit" in t:
            profits_abs.append(float(t["profit"]))
        elif "profit_€" in t:
            profits_abs.append(float(t["profit_€"]))
        else:
            profits_abs.append(0.0)

        if "pct_change" in t:
            profits_pct.append(float(t["pct_change"]))
        else:
            # fallback: compute pct if buy/sell price available
            bp = t.get("buy_price")
            sp = t.get("sell_price")
            if bp and sp and bp != 0:
                profits_pct.append((float(sp) - float(bp)) / float(bp) * 100.0)
            else:
                profits_pct.append(0.0)

    profits_abs = np.array(profits_abs, dtype=float)
    profits_pct = np.array(profits_pct, dtype=float)

    wins = profits_pct[profits_pct > 0]
    losses = profits_pct[profits_pct <= 0]

    nb_trades = int(len(profits_pct))
    nb_wins = int(len(wins))
    nb_losses = nb_trades - nb_wins
    win_rate = (nb_wins / nb_trades) * 100.0 if nb_trades > 0 else 0.0
    avg_gain = float(np.mean(wins)) if wins.size > 0 else 0.0
    avg_loss = float(np.mean(losses)) if losses.size > 0 else 0.0
    avg_profit_abs = float(np.mean(profits_abs)) if profits_abs.size > 0 else 0.0
    total_profit_abs = float(np.sum(profits_abs)) if profits_abs.size > 0 else 0.0

    return {
        "nb_trades": nb_trades,
        "nb_wins": nb_wins,
        "nb_losses": nb_losses,
        "win_rate_pct": float(win_rate),
        "avg_gain_pct": avg_gain,
        "avg_loss_pct": avg_loss,
        "avg_profit_absolute": avg_profit_abs,
        "total_profit_absolute": total_profit_abs
    }

def equity_curve_metrics(equity_series: pd.Series) -> Dict[str, Any]:
    """
    Calcule metrics sur la série d'equity (valeurs absolues).
    equity_series: pd.Series index datetime, valeurs equity (float).

    Retour:
      {
        total_return_pct: float,
        annualized_return_pct: float,
        max_drawdown_pct: float,
        max_drawdown_period: (start, trough, end) or None
      }
    """
    if equity_series is None or equity_series.empty:
        return {
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "max_drawdown_period": None
        }

    ser = equity_series.dropna().astype(float)
    start_val = ser.iloc[0]
    end_val = ser.iloc[-1]
    total_return = (end_val / start_val - 1.0) * 100.0 if start_val != 0 else 0.0

    # annualized return (approx) based on calendar days
    days = (ser.index[-1] - ser.index[0]).days
    if days > 0 and start_val > 0:
        years = days / 365.25
        annualized = ((end_val / start_val) ** (1.0 / years) - 1.0) * 100.0
    else:
        annualized = 0.0

    # drawdown
    roll_max = ser.cummax()
    drawdown = (ser - roll_max) / roll_max
    max_dd = drawdown.min() * 100.0
    if np.isfinite(max_dd):
        # compute period: find peak before trough and recovery point after trough (simple)
        trough_idx = drawdown.idxmin()
        trough_val = drawdown.min()
        # find peak start
        peak_series = ser.loc[:trough_idx]
        peak_idx = peak_series.idxmax()
        # find recovery (first index after trough where ser >= previous peak)
        recovery_idx = None
        for i in ser.loc[trough_idx:].index:
            if ser.loc[i] >= ser.loc[peak_idx]:
                recovery_idx = i
                break
        max_dd_period = (peak_idx, trough_idx, recovery_idx)
    else:
        max_dd = 0.0
        max_dd_period = None

    return {
        "total_return_pct": float(total_return),
        "annualized_return_pct": float(annualized),
        "max_drawdown_pct": float(max_dd),
        "max_drawdown_period": max_dd_period
    }

def format_currency(value: float, decimals: int = 2) -> str:
    """
    Format a float value into a human readable euro-like string "1 234,56".
    Simple implementation uses space thousands separator and comma decimal.
    """
    try:
        v = float(value)
    except Exception:
        return str(value)
    s = f"{v:,.{decimals}f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", " ")
    return f"{s}€"