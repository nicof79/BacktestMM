"""
utils.py
---------
Fonctions utilitaires : configuration, logs, affichage, etc.
"""

import json
import logging

def load_config(path="config/config.json") -> dict:
    """Charge le fichier de configuration JSON."""
    with open(path, "r") as f:
        return json.load(f)

def init_logger(log_path="logs/backtest.log"):
    """Initialise le système de logs."""
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def display_results(metrics: dict):
    """Affiche les résultats principaux du backtest."""
    print("\n=== Résultats du backtest ===")
    print(f"Meilleure combinaison : {metrics['best_short']}/{metrics['best_long']}")
    print(f"Capital final : {metrics['best_capital']:.2f} €")
    print(f"Nombre moyen de trades : {metrics['avg_trades']:.1f}")
