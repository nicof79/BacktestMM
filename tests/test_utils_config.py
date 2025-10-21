import json
import tempfile
from pathlib import Path
import pytest
from utils.utils import load_config, get_run_plan, validate_config

VALID_MINIMAL = {
    "symbols": ["MC.PA"],
    "start_date": "2020-01-01"
}

VALID_FULL = {
    "symbols": ["AAPL", "MSFT"],
    "start_date": "2020-01-01",
    "end_date": "2022-01-01",
    "initial_capital": 5000,
    "max_allocation_pct": 0.1,
    "periods": [5, 10, 20],
    "ma_types": ["SMA", "EMA"],
    "ratio_min": 1.5,
    "ratio_max": 10.0,
    "min_data_days": 50,
    "mode": {"vector_scan": True, "full_backtest": True, "pipeline_filter_top_n": 20},
    "vector_metrics": {"metric": "mean_return_per_trade", "min_trades": 5}
}

def _write_tmp(cfg):
    fd, fname = tempfile.mkstemp(suffix=".json")
    p = Path(fname)
    p.write_text(json.dumps(cfg))
    return p

def test_load_config_minimal(tmp_path):
    p = _write_tmp(VALID_MINIMAL)
    cfg = load_config(p)
    assert "symbols" in cfg and cfg["symbols"] == ["MC.PA"]
    assert cfg["start_date"] == "2020-01-01"
    # defaults filled
    assert "initial_capital" in cfg
    assert isinstance(cfg["periods"], list)
    plan = get_run_plan(cfg)
    assert plan["vector"] is True and plan["full"] is False

def test_load_config_full(tmp_path):
    p = _write_tmp(VALID_FULL)
    cfg = load_config(p)
    assert cfg["symbols"] == ["AAPL", "MSFT"]
    assert cfg["end_date"] == "2022-01-01"
    plan = get_run_plan(cfg)
    assert plan["vector"] is True and plan["full"] is True and plan["pipeline_top_n"] == 20

def test_invalid_date():
    bad = {"symbols": ["MC.PA"], "start_date": "2020-13-01"}
    p = _write_tmp(bad)
    with pytest.raises(ValueError):
        load_config(p)

def test_invalid_periods():
    bad = {"symbols": ["X"], "start_date": "2020-01-01", "periods": [0, -1, "a"]}
    p = _write_tmp(bad)
    with pytest.raises(ValueError):
        load_config(p)

def test_invalid_alloc_pct():
    bad = {"symbols": ["X"], "start_date": "2020-01-01", "max_allocation_pct": 1.5}
    p = _write_tmp(bad)
    with pytest.raises(ValueError):
        load_config(p)