"""
CrossMM_Backtest.py
-------------------
Point d’entrée principal du projet Cross Moving Averages Backtest (V1.3).
"""

from src.utils import load_config, init_logger, display_results, save_results_df
from src.data_loader import load_data
from src.indicators import compute_moving_averages
from src.backtest_engine import run_backtest
from src.metrics import compute_performance
import logging
import pandas as pd

def main():
    # Load configuration
    config = load_config()
    
    #DEBUG 
    config.setdefault("ma_types", ["SMA"])
    config.setdefault("ma_periods", [5, 10, 20, 50, 100])
    
    # Init logger
    init_logger()
    logging.info("Starting CrossMM_Backtest (V1.3)")

    # Load data
    df = load_data(config.get("symbol"), config.get("start_date"), config.get("end_date"))

    # DEBUG : vérifier les données chargées
    print("DEBUG ma_types:", config.get("ma_types"))
    print("DEBUG ma_periods:", config.get("ma_periods"))
    
    # Compute indicators
    df = compute_moving_averages(df, config.get("ma_types", ["SMA"]), config.get("ma_periods", []))

    # DEBUG : vérifier les colonnes créées et leurs valeurs
    print("\nColonnes disponibles :", [c for c in df.columns if "SMA" in c])
    print(df[[c for c in df.columns if "SMA" in c]].tail(5))

    # Run backtest engine
    results = run_backtest(df, config)

    # Compute performance metrics
    metrics = compute_performance(results)

    # Display
    display_results(metrics)

    # Optionally save summary dataframe
    summary_df = metrics.get("summary_df")
    if summary_df is not None and not summary_df.empty:
        # ensure pandas DataFrame
        if isinstance(summary_df, pd.DataFrame):
            save_results_df(summary_df)

if __name__ == "__main__":
    main()
