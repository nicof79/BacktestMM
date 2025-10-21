"""
metrics.py
-----------
Module de calcul des métriques de performance (rendement, drawdown, Sharpe...).
"""

import pandas as pd
import numpy as np

def compute_drawdown(equity_curve: pd.Series) -> float:
    """Calcule le drawdown maximal."""
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    return dd.min()

def compute_sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Calcule le ratio de Sharpe."""
    return (returns.mean() - risk_free) / returns.std() if returns.std() != 0 else 0.0

def compute_performance(results: list) -> dict:
    """
    Calcule les métriques globales du backtest à partir des résultats.
    """
    df = pd.DataFrame(results)
    best = df.loc[df["final_capital"].idxmax()]
    return {
        "best_short": best["short"],
        "best_long": best["long"],
        "best_capital": best["final_capital"],
        "avg_trades": df["nb_trades"].mean()
    }
