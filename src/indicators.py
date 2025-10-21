"""
indicators.py
--------------
Module de calcul des indicateurs techniques : SMA, EMA, WMA, HMA.
"""

import pandas as pd
import numpy as np
from typing import List


def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Calcule une moyenne mobile simple (SMA)."""
    return series.rolling(window=window, min_periods=1).mean()


def compute_ema(series: pd.Series, window: int) -> pd.Series:
    """Calcule une moyenne mobile exponentielle (EMA)."""
    return series.ewm(span=window, adjust=False, min_periods=1).mean()


def compute_wma(series: pd.Series, window: int) -> pd.Series:
    """Calcule une moyenne mobile pondérée (WMA)."""
    weights = np.arange(1, window + 1)
    return series.rolling(window=window).apply(
        lambda prices: np.dot(prices, weights) / weights.sum(),
        raw=True
    )


def compute_hma(series: pd.Series, window: int) -> pd.Series:
    """Calcule une moyenne mobile de Hull (HMA)."""
    if window < 2:
        return series.copy()
    half_len = int(window / 2)
    sqrt_len = int(np.sqrt(window))
    wma_half = compute_wma(series, half_len)
    wma_full = compute_wma(series, window)
    diff = 2 * wma_half - wma_full
    return compute_wma(diff, sqrt_len)


def compute_moving_averages(df: pd.DataFrame, ma_types: List[str], ma_periods: List[int]) -> pd.DataFrame:
    """
    Calcule toutes les moyennes mobiles spécifiées (SMA, EMA, WMA, HMA).
    Ajoute les colonnes correspondantes au DataFrame et logge le résultat.
    """
    if not ma_types or not ma_periods:
        print("⚠️  Aucun type ou période de moyenne mobile fourni.")
        print("ma_types:", ma_types, "ma_periods:", ma_periods)
        return df

    created_cols = []
    for ma_type in ma_types:
        t = ma_type.upper().strip()
        for period in ma_periods:
            col_name = f"{t}_{period}"
            if col_name in df.columns:
                continue
            try:
                if t == "SMA":
                    df[col_name] = compute_sma(df["Close"], period)
                elif t == "EMA":
                    df[col_name] = compute_ema(df["Close"], period)
                elif t == "WMA":
                    df[col_name] = compute_wma(df["Close"], period)
                elif t == "HMA":
                    df[col_name] = compute_hma(df["Close"], period)
                else:
                    print(f"⚠️  Type de moyenne inconnu : {t}")
                    continue
                created_cols.append(col_name)
            except Exception as e:
                print(f"❌ Erreur sur {col_name}: {e}")

    print(f"Moyennes calculées ({len(created_cols)}) : {created_cols}")
    return df
