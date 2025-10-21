"""
data_loader.py
---------------
Téléchargement et préparation des données de marché.
"""

import yfinance as yf
import pandas as pd

def load_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Télécharge les données depuis Yahoo Finance et nettoie le DataFrame."""
    df = yf.download(symbol, start=start, end=end)
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    return df
