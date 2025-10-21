"""
CrossMM_Backtest.py
-------------------
Point d’entrée principal du projet Cross Moving Averages Backtest.

Version : 1.3 (structuration du code)
Auteur  : Nicolas F.
"""

from src.utils import load_config, init_logger, display_results
from src.data_loader import load_data
from src.indicators import compute_moving_averages
from src.backtest_engine import run_backtest
from src.metrics import compute_performance

def main():
    """Point d’entrée du programme : exécute un backtest complet."""
    config = load_config()
    init_logger()

    print(f"\n=== Lancement du backtest sur {config['symbol']} ===")
    data = load_data(config["symbol"], config["start_date"], config["end_date"])
    data = compute_moving_averages(data, config["ma_types"], config["ma_periods"])
    results = run_backtest(data, config)
    metrics = compute_performance(results)
    display_results(metrics)

if __name__ == "__main__":
    main()
