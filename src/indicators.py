"""
indicators.py
--------------
Module de calcul des indicateurs techniques (SMA, EMA, etc.)
"""

import pandas as pd

def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Calcule une moyenne mobile simple."""
    return series.rolling(window=window).mean()

def compute_ema(series: pd.Series, window: int) -> pd.Series:
    """Calcule une moyenne mobile exponentielle."""
    return series.ewm(span=window, adjust=False).mean()

def compute_moving_averages(df: pd.DataFrame, ma_types: list, ma_periods: list) -> pd.DataFrame:
    """
    Calcule plusieurs types de moyennes mobiles selon les paramètres.
    Renvoie le DataFrame enrichi.
    """
    for ma_type in ma_types:
        for period in ma_periods:
            col_name = f"{ma_type}_{period}"
            if ma_type.upper() == "SMA":
                df[col_name] = compute_sma(df["Close"], period)
            elif ma_type.upper() == "EMA":
                df[col_name] = compute_ema(df["Close"], period)
    return df
