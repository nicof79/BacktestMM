"""
money_management.py
-------------------
Gestion du capital et exécution des signaux.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def execute_trades(df: pd.DataFrame, bull_cross: pd.Series, bear_cross: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Exécute les trades selon les signaux de croisement.
    Achat à l'ouverture du jour suivant un croisement haussier.
    Vente à l'ouverture du jour suivant un croisement baissier.
    """
    capital = float(config.get("initial_capital", 10000))
    max_alloc = float(config.get("max_allocation", 0.2))
    position = 0.0
    entry_price = 0.0
    trades = []

    df_iter = df.reset_index(drop=True)
    bull_cross = bull_cross.astype(bool).reset_index(drop=True)
    bear_cross = bear_cross.astype(bool).reset_index(drop=True)

    for i in range(len(df_iter) - 1):
        pos = float(position)

        if bool(bull_cross.iloc[i]) and pos == 0.0:
            next_open = float(df_iter.loc[i + 1, "Open"].iloc[0] if isinstance(df_iter.loc[i + 1, "Open"], pd.Series) else df_iter.loc[i + 1, "Open"])
            amount_to_invest = capital * max_alloc
            position = amount_to_invest / next_open
            entry_price = next_open
            capital -= amount_to_invest
            trades.append({
                "date": df_iter.loc[i + 1, "Date"] if "Date" in df_iter.columns else i + 1,
                "type": "BUY",
                "price": next_open,
                "capital": capital
            })
            print(f"🟢 BUY {df_iter.loc[i + 1, 'Date'] if 'Date' in df_iter.columns else i + 1} @ {next_open:.2f} | Capital restant: {capital:.2f}")

        elif bool(bear_cross.iloc[i]) and pos > 0.0:
            next_open = float(df_iter.loc[i + 1, "Open"].iloc[0] if isinstance(df_iter.loc[i + 1, "Open"], pd.Series) else df_iter.loc[i + 1, "Open"])
            capital += pos * next_open
            trades.append({
                "date": df_iter.loc[i + 1, "Date"] if "Date" in df_iter.columns else i + 1,
                "type": "SELL",
                "price": next_open,
                "capital": capital
            })
            print(f"🔴 SELL {df_iter.loc[i + 1, 'Date'] if 'Date' in df_iter.columns else i + 1} @ {next_open:.2f} | Capital: {capital:.2f}")
            position = 0.0
            entry_price = 0.0

    # Clôture à la fin si position ouverte
    if position > 0.0:
        last_close = float(df_iter["Close"].iloc[-1])
        capital += position * last_close
        trades.append({
            "date": df_iter.iloc[-1]["Date"] if "Date" in df_iter.columns else len(df_iter) - 1,
            "type": "SELL_END",
            "price": last_close,
            "capital": capital
        })
        print(f"⚪ SELL_END (Clôture finale) @ {last_close:.2f} | Capital final: {capital:.2f}")


    return {
        "initial_capital": float(config.get("initial_capital", 10000)),
        "final_capital": float(capital),
        "nb_trades": len(trades),
        "trades": trades
    }
