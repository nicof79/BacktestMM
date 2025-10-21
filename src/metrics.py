"""
metrics.py
-----------
Calcul des métriques de performance à partir des résultats du backtest.
"""

from typing import List, Dict
import pandas as pd
import numpy as np

def compute_total_return(initial: float, final: float) -> float:
    return (final - initial) / initial * 100.0

def compute_performance(results: List[Dict]) -> Dict:
    """Analyse les résultats et retourne les stats principales."""
    if not results:
        return {"summary_df": pd.DataFrame(), "best": None, "overall": {}}

    df = pd.DataFrame(results)
    df["total_return_pct"] = (df["final_capital"] - df["initial_capital"]) / df["initial_capital"] * 100

    best_idx = df["final_capital"].idxmax()
    best = df.loc[best_idx].to_dict()

    overall = {
        "n_combinations": len(df),
        "avg_final_capital": float(df["final_capital"].mean()),
        "median_nb_trades": int(df["nb_trades"].median())
    }

    return {"summary_df": df, "best": best, "overall": overall}

