"""
utils/utils.py
Config loader, validation, et helpers.

Public API:
- load_config(path: str | Path) -> dict
- validate_config(cfg: dict) -> None (lève ValueError si invalide)
- get_run_plan(cfg: dict) -> dict
"""
from pathlib import Path
from typing import Any, Dict
import json
import datetime

DEFAULT_CONFIG = {
    "symbols": ["MC.PA"],
    "start_date": "2020-01-01",
    "end_date": None,
    "initial_capital": 10000.0,
    "max_allocation_pct": 0.2,
    "periods": [2, 3, 5, 8, 10, 12, 13, 15, 20, 21, 26, 30, 34, 50, 55, 89, 100, 144, 200],
    "ma_types": ["SMA"],
    "ratio_min": 1.5,
    "ratio_max": 13.0,
    "min_data_days": 200,
    "mode": {"vector_scan": True, "full_backtest": False, "pipeline_filter_top_n": 50},
    "vector_metrics": {"metric": "mean_return_per_trade", "min_trades": 10},
    "download": {"source": "yfinance", "progress": False}
}


def _parse_date(s: Any) -> str | None:
    if s is None:
        return None
    if isinstance(s, str):
        try:
            # normalize format YYYY-MM-DD
            datetime.datetime.strptime(s, "%Y-%m-%d")
            return s
        except Exception:
            raise ValueError(f"Invalid date format, expected YYYY-MM-DD: {s}")
    raise ValueError(f"Invalid date value: {s}")


def load_config(path) -> Dict[str, Any]:
    """
    Load config JSON from path, shallow-merge defaults and validate.

    Returns normalized config dict ready for use.
    Raises FileNotFoundError or ValueError on invalid content.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Merge defaults (shallow)
    merged = DEFAULT_CONFIG.copy()
    for k, v in cfg.items():
        # if value is a dict, merge shallowly
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = merged[k].copy()
            merged[k].update(v)
        else:
            merged[k] = v

    # Normalize dates
    merged["start_date"] = _parse_date(merged.get("start_date"))
    merged["end_date"] = _parse_date(merged.get("end_date"))

    # validate config
    validate_config(merged)

    return merged


def validate_config(cfg: Dict[str, Any]) -> None:
    """
    Validate config dict in-place; raise ValueError on invalid values.
    Checks presence/types and simple logical constraints.
    """
    # symbols
    symbols = cfg.get("symbols")
    if not isinstance(symbols, list) or len(symbols) == 0:
        raise ValueError("config['symbols'] must be a non-empty list of ticker strings")

    # dates
    start = cfg.get("start_date")
    end = cfg.get("end_date")
    if start is None:
        raise ValueError("config['start_date'] must be provided")
    # end can be None (interpreted as today), otherwise validate order
    if end is not None:
        start_dt = datetime.datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.datetime.strptime(end, "%Y-%m-%d")
        if end_dt <= start_dt:
            raise ValueError("config['end_date'] must be after start_date")

    # numeric fields
    if not isinstance(cfg.get("initial_capital"), (int, float)) or cfg["initial_capital"] <= 0:
        raise ValueError("config['initial_capital'] must be a positive number")
    if not isinstance(cfg.get("max_allocation_pct"), (int, float)) or not (0 < cfg["max_allocation_pct"] <= 1):
        raise ValueError("config['max_allocation_pct'] must be a float in (0, 1]")

    # periods
    periods = cfg.get("periods")
    if not isinstance(periods, list) or not all(isinstance(x, int) and x >= 1 for x in periods):
        raise ValueError("config['periods'] must be a list of integers >= 1")

    # ma_types
    ma_types = cfg.get("ma_types")
    if not isinstance(ma_types, list) or not all(isinstance(t, str) for t in ma_types):
        raise ValueError("config['ma_types'] must be a list of strings")

    # ratios
    rmin = cfg.get("ratio_min")
    rmax = cfg.get("ratio_max")
    if not isinstance(rmin, (int, float)) or not isinstance(rmax, (int, float)) or rmin <= 0 or rmax <= 0:
        raise ValueError("ratio_min and ratio_max must be positive numbers")
    if rmin >= rmax:
        raise ValueError("ratio_min must be less than ratio_max")

    # min_data_days
    if not isinstance(cfg.get("min_data_days"), int) or cfg["min_data_days"] < 10:
        raise ValueError("config['min_data_days'] must be integer >= 10")

    # mode
    mode = cfg.get("mode")
    if not isinstance(mode, dict):
        raise ValueError("config['mode'] must be a dict")
    if not any(bool(mode.get(k)) for k in ("vector_scan", "full_backtest")):
        # allow none -> default pipeline behavior, but require explicit in config pipeline ; here we allow but warn via exception is avoided
        pass
    top = mode.get("pipeline_filter_top_n")
    if top is not None and (not isinstance(top, int) or top < 0):
        raise ValueError("mode.pipeline_filter_top_n must be a non-negative integer or null")

    # vector_metrics
    vm = cfg.get("vector_metrics")
    if not isinstance(vm, dict):
        raise ValueError("config['vector_metrics'] must be a dict")
    metric = vm.get("metric")
    if metric is not None and not isinstance(metric, str):
        raise ValueError("vector_metrics.metric must be a string")
    min_trades = vm.get("min_trades")
    if min_trades is not None and (not isinstance(min_trades, int) or min_trades < 0):
        raise ValueError("vector_metrics.min_trades must be a non-negative integer")


def get_run_plan(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Determine what to run from config.
    Returns dict with keys: vector (bool), full (bool), pipeline_top_n (int or 0)
    """
    mode = cfg.get("mode", {})
    vector = bool(mode.get("vector_scan", False))
    full = bool(mode.get("full_backtest", False))
    top_n = int(mode.get("pipeline_filter_top_n", 0)) if mode.get("pipeline_filter_top_n") is not None else 0

    # If neither specified true, default to vector only
    if not vector and not full:
        vector = True
        full = False

    return {"vector": vector, "full": full, "pipeline_top_n": top_n}