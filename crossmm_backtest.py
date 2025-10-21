#!/usr/bin/env python
"""
crossmm_backtest.py

CLI orchestration:
- charge config JSON
- télécharge les données via yfinance pour chaque symbole (option d'utiliser les prix ajustés)
- exécute vector_scan si demandé
- sélectionne top N combos si pipeline
- pré-calcul des MAs nécessaires pour les combos retenus
- exécute full_backtest sur les combos sélectionnés
- sauvegarde fichiers CSV: vector_results.csv, full_results.csv
- affiche résumé minimal
"""
import argparse
from pathlib import Path
import json
import logging
import sys

import yfinance as yf
import pandas as pd

from utils.utils import load_config, get_run_plan
from utils.io import ensure_dir, save_results_csv
import engine.vector_backtest as vmod
import engine.backtester as bmod
from engine.money_manager import MoneyManager
from metrics.performance import trade_stats, equity_curve_metrics

LOG = logging.getLogger("crossmm_backtest")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
LOG.addHandler(handler)


def money_manager_factory(cfg):
    return MoneyManager(initial_capital=cfg.get("initial_capital", 10000.0),
                        max_alloc_pct=cfg.get("max_allocation_pct", 0.2),
                        commission=cfg.get("commission", 0.0),
                        slippage=cfg.get("slippage", 0.0))


def download_symbol(symbol: str, start: str, end: str, progress: bool = False, use_adjusted: bool = True) -> pd.DataFrame:
    LOG.info(f"Downloading {symbol} {start} -> {end}")
    try:
        df = yf.download(symbol, start=start, end=end, progress=progress, auto_adjust=False)
    except Exception as e:
        LOG.warning(f"yf.download failed for {symbol}: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        LOG.warning(f"No data for {symbol}")
        return pd.DataFrame()

    # Normalize MultiIndex columns (yfinance sometimes returns tuples like ('Close','MC.PA'))
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) and len(col) > 0 else col for col in df.columns]

    cols = list(df.columns)
    LOG.debug(f"Downloaded columns for {symbol}: {cols}")

    # If Adj Close present and user wants adjusted prices
    if use_adjusted and "Adj Close" in df.columns:
        if "Close" not in df.columns:
            df["Close"] = df["Adj Close"]
        else:
            df["_Close_raw"] = df["Close"]
            if "Open" in df.columns: df["_Open_raw"] = df["Open"]
            if "High" in df.columns: df["_High_raw"] = df["High"]
            if "Low" in df.columns: df["_Low_raw"] = df["Low"]

            adj_ratio = df["Adj Close"] / df["Close"]
            adj_ratio = adj_ratio.replace([float("inf"), float("-inf")], float("nan"))
            adj_ratio = adj_ratio.ffill().fillna(1.0)

            if "Open" in df.columns:
                df["Open"] = df["Open"] * adj_ratio
            if "High" in df.columns:
                df["High"] = df["High"] * adj_ratio
            if "Low" in df.columns:
                df["Low"] = df["Low"] * adj_ratio
            df["Close"] = df["Adj Close"]

    # If Close not available at this point, log and return empty
    if "Close" not in df.columns:
        LOG.warning(f"Downloaded data for {symbol} has no Close column after normalization; available columns: {list(df.columns)}")
        return pd.DataFrame()

    # Drop rows without Close values (defensive)
    try:
        df = df.dropna(subset=["Close"])
    except KeyError:
        LOG.warning(f"Unexpected missing Close column after processing for {symbol}")
        return pd.DataFrame()

    return df


def precompute_mas_for_combos(df: pd.DataFrame, combos):
    """
    Ensure df has MA columns for the combos list (dicts with type_short/p_short/type_long/p_long).
    Uses indicators.calculate_ma to create columns named MA_{type}{period}
    """
    from indicators.ma import calculate_ma
    cols_done = set()
    for c in combos:
        for t, p in [(c["type_short"], c["p_short"]), (c["type_long"], c["p_long"])]:
            col = f"MA_{t}{p}"
            if col in cols_done:
                continue
            df[col] = calculate_ma(df["Close"], int(p), t)
            cols_done.add(col)
    return df


def run_for_symbol(symbol: str, cfg: dict, out_dir: Path, use_adjusted: bool = True):
    df = download_symbol(symbol,
                         cfg["start_date"],
                         cfg.get("end_date"),
                         progress=cfg.get("download", {}).get("progress", False),
                         use_adjusted=use_adjusted)
    if df.empty:
        return

    plan = get_run_plan(cfg)
    vector_results = None
    full_results = None

    # VECTOR SCAN
    if plan["vector"]:
        LOG.info("Starting vector scan")
        vector_results = vmod.vector_scan(
            df=df,
            periods=cfg["periods"],
            ma_types=cfg["ma_types"],
            ratio_min=cfg["ratio_min"],
            ratio_max=cfg["ratio_max"],
            metrics_config=cfg.get("vector_metrics", {"metric": "mean_return_per_trade", "min_trades": 0})
        )
        if not vector_results.empty:
            vr_path = out_dir / f"{symbol}_vector_results.csv"
            save_results_csv(vector_results, vr_path)
            LOG.info(f"Vector results saved to {vr_path}")
        else:
            LOG.info("Vector scan returned no combos")

    # FULL BACKTEST (pipeline behavior)
    if plan["full"]:
        combos_to_run = []
        if plan["vector"] and vector_results is not None and not vector_results.empty and plan.get("pipeline_top_n", 0) > 0:
            top_n = int(plan["pipeline_top_n"])
            selected = vector_results.head(top_n)
            for _, row in selected.iterrows():
                combos_to_run.append({
                    "type_short": row["type_short"],
                    "type_long": row["type_long"],
                    "p_short": int(row["p_short"]),
                    "p_long": int(row["p_long"])
                })
        elif plan["vector"] and (vector_results is not None and not vector_results.empty) and plan.get("pipeline_top_n", 0) == 0:
            # run on all vector results
            for _, row in vector_results.iterrows():
                combos_to_run.append({
                    "type_short": row["type_short"],
                    "type_long": row["type_long"],
                    "p_short": int(row["p_short"]),
                    "p_long": int(row["p_long"])
                })
        else:
            # no vector or vector empty — build combos from config directly (may be heavy)
            for ts in cfg["ma_types"]:
                for tl in cfg["ma_types"]:
                    for ps in cfg["periods"]:
                        for pl in cfg["periods"]:
                            if ps >= pl:
                                continue
                            ratio = pl / ps
                            if ratio < cfg["ratio_min"] or ratio > cfg["ratio_max"]:
                                continue
                            combos_to_run.append({"type_short": ts, "type_long": tl, "p_short": ps, "p_long": pl})

        if not combos_to_run:
            LOG.info("No combos to run in full backtest")
        else:
            LOG.info(f"Running full backtest on {len(combos_to_run)} combos")
            df = precompute_mas_for_combos(df, combos_to_run)
            # run full_backtest
            full_results = bmod.full_backtest(df, combos_to_run, cfg, money_manager_factory, exec_price_col="Open", exec_offset=1)
            if not full_results.empty:
                # compute basic metrics per combo
                metrics_rows = []
                for _, r in full_results.iterrows():
                    trades = r.get("trades", [])
                    ts = trade_stats(trades)
                    metrics_rows.append({**r, **ts})
                fr_path = out_dir / f"{symbol}_full_results.csv"
                save_results_csv(full_results, fr_path)
                LOG.info(f"Full backtest results saved to {fr_path}")

    LOG.info("Done for symbol: %s", symbol)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config.json", help="Path to config.json")
    parser.add_argument("--out", "-o", default="results", help="Output directory")
    parser.add_argument("--symbol", "-s", help="Override symbol (single)")
    # CLI option to control adjusted price behavior
    parser.add_argument("--use-adjusted", dest="use_adjusted", action="store_true", help="Use adjusted prices (Adj Close) and rescale OHLC")
    parser.add_argument("--no-adjusted", dest="use_adjusted", action="store_false", help="Do not use adjusted prices")
    parser.set_defaults(use_adjusted=True)

    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    plan = get_run_plan(cfg)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    symbols = [args.symbol] if args.symbol else cfg.get("symbols", ["MC.PA"])
    for sym in symbols:
        run_for_symbol(sym, cfg, out_dir, use_adjusted=args.use_adjusted)

    LOG.info("All done")


if __name__ == "__main__":
    main()