"""
data_loader.py
---------------
Téléchargement et préparation des données de marché via yfinance.
"""

import yfinance as yf
import pandas as pd
import logging

def load_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Télécharge les données depuis Yahoo Finance et prépare le DataFrame.

    Renvoie un DataFrame indexé par la date (DatetimeIndex) contenant les colonnes
    Open, High, Low, Close, Adj Close, Volume.
    """
    logging.info(f"Downloading data for {symbol} from {start} to {end}")
    df = yf.download(symbol, start=start, end=end, progress=False)

    if df is None or df.empty:
        raise ValueError(f"Aucune donnée récupérée pour {symbol} entre {start} et {end}.")

    # Drop rows with NaNs (ensures indicators can be calculated)
    df = df.dropna(how="any")
    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.reset_index(inplace=True)
        df.set_index("Date", inplace=True)
        df.index = pd.to_datetime(df.index)

    logging.info(f"Data loaded: {len(df)} rows")
    return df
