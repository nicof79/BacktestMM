"""
backtest_engine.py
------------------
Module principal du moteur de backtest.
Gère la logique des signaux, l’exécution des trades et la collecte des résultats.
"""

from src.money_management import execute_trades

def detect_crossovers(df, short_col, long_col):
    """Détecte les croisements haussiers et baissiers entre deux MMs."""
    delta = df[short_col] - df[long_col]
    delta_prev = delta.shift(1)
    bull_cross = (delta_prev <= 0) & (delta > 0)
    bear_cross = (delta_prev >= 0) & (delta < 0)
    return bull_cross, bear_cross

def run_backtest(df, config):
    """
    Exécute le backtest sur les combinaisons de moyennes mobiles définies.
    Renvoie la liste des résultats.
    """
    results = []
    for short in config["short_periods"]:
        for long in config["long_periods"]:
            if long <= short:
                continue
            short_col, long_col = f"SMA_{short}", f"SMA_{long}"
            bull, bear = detect_crossovers(df, short_col, long_col)
            trades = execute_trades(df, bull, bear, config)
            results.append({
                "short": short, "long": long,
                "final_capital": trades["final_capital"],
                "nb_trades": len(trades["positions"])
            })
    return results
