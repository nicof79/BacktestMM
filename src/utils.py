"""
utils.py
---------
Fonctions utilitaires : configuration, logs, affichage, sauvegarde.
"""

import json
import logging
import os
from datetime import datetime

def load_config(path="config/config.json") -> dict:
    """Charge le fichier de configuration JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def init_logger(log_path="logs/backtest.log"):
    """Initialise le système de logs (création du dossier si besoin)."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.getLogger().addHandler(logging.StreamHandler())  # also print to stdout

def display_results(metrics: dict):
    """Affiche le résumé des métriques calculées."""
    best = metrics.get("best")
    overall = metrics.get("overall", {})
    summary_df = metrics.get("summary_df")

    print("\n" + "="*80)
    print("RÉSUMÉ GÉNÉRAL DU BACKTEST")
    print("="*80)
    print(f"Combinaisons testées : {overall.get('n_combinations', 0)}")
    print(f"Capital final moyen : {overall.get('avg_final_capital', 0.0):.2f} €")
    print(f"Nombre moyen/median de trades : {overall.get('median_nb_trades', 0)}")
    if best:
        print("\nMeilleure combinaison (par capital final) :")
        print(f"  Type : {best.get('ma_type')}  MM : {best.get('short')}/{best.get('long')}")
        print(f"  Capital initial : {best.get('initial_capital'):.2f} €")
        print(f"  Capital final   : {best.get('final_capital'):.2f} €")
        print(f"  Rendement total : {best.get('total_return_pct'):.2f} %")
        print(f"  Nb trades       : {best.get('nb_trades')}")
    print("="*80)

def save_results_df(df, filename=None):
    """Sauvegarde le DataFrame de synthèse dans /results/ avec timestamp."""
    import pandas as pd
    os.makedirs("results", exist_ok=True)
    if filename is None:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"results/results_{ts}.csv"
    df.to_csv(filename, index=False)
    logging.info(f"Saved results to {filename}")
    return filename
