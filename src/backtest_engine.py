"""
backtest_engine.py
------------------
Moteur du backtest : génération automatique des combinaisons et détection des croisements.
"""

from typing import List, Dict, Tuple
import pandas as pd
from src.money_management import execute_trades
import logging
import itertools


def detect_crossovers(df: pd.DataFrame, col1: str, col2: str) -> Tuple[pd.Series, pd.Series]:
    """
    Détecte les croisements haussiers (bull) et baissiers (bear) entre deux moyennes mobiles.
    """
    # Correction des warnings : nouvelle syntaxe pandas
    s1 = df[col1].bfill().ffill()
    s2 = df[col2].bfill().ffill()

    delta = s1 - s2
    delta_prev = delta.shift(1)

    bull_cross = (delta_prev <= 0) & (delta > 0)
    bear_cross = (delta_prev >= 0) & (delta < 0)

    if bull_cross.any() or bear_cross.any():
        print(f"Croisements détectés pour {col1}/{col2}")
    return bull_cross.fillna(False), bear_cross.fillna(False)



def generate_combinations(ma_types: List[str], ma_periods: List[int], ratio_min: float, ratio_max: float):
    """
    Génère toutes les combinaisons valides de moyennes mobiles selon les contraintes de ratio.
    """
    combos = []
    for type1, type2 in itertools.product(ma_types, ma_types):
        for p1 in ma_periods:
            for p2 in ma_periods:
                if p2 <= p1:
                    continue
                ratio = p2 / p1
                if ratio_min <= ratio <= ratio_max:
                    combos.append((type1.upper(), p1, type2.upper(), p2))
    return combos


def run_backtest(df: pd.DataFrame, config: Dict) -> List[Dict]:
    """
    Exécute le backtest sur toutes les combinaisons générées automatiquement.
    """
    ma_types = [t.upper() for t in config.get("ma_types", ["SMA"])]
    ma_periods = config.get("ma_periods", [])
    ratio_min = float(config.get("ratio_min", 1.5))
    ratio_max = float(config.get("ratio_max", 13.0))

    all_combos = generate_combinations(ma_types, ma_periods, ratio_min, ratio_max)
    logging.info(f"Génération de {len(all_combos)} combinaisons valides de moyennes mobiles.")

    results = []
    for (t1, p1, t2, p2) in all_combos:
        col1, col2 = f"{t1}_{p1}", f"{t2}_{p2}"
        if col1 not in df.columns or col2 not in df.columns:
            continue

        bull, bear = detect_crossovers(df, col1, col2)
        # Ne teste que si au moins un croisement détecté
        if not bull.any() and not bear.any():
            continue

        trades = execute_trades(df, bull, bear, config)
        results.append({
            "type1": t1,
            "p1": p1,
            "type2": t2,
            "p2": p2,
            "final_capital": float(trades["final_capital"]),
            "initial_capital": float(trades["initial_capital"]),
            "nb_trades": len(trades["trades"]),
            "trades": trades["trades"]
        })

    logging.info(f"{len(results)} combinaisons testées (croisements détectés).")
    return results
