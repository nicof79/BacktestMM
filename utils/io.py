"""
utils/io.py

Small IO helpers for saving DataFrames and ensuring directories.
"""
from pathlib import Path
import pandas as pd

def ensure_dir(p: Path):
    p = Path(p)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)

def save_results_csv(df, path: Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # If DataFrame contains nested lists/dicts (e.g., trades), convert to JSON strings for CSV
    df_out = df.copy()
    for col in df_out.columns:
        if df_out[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df_out[col] = df_out[col].apply(lambda x: pd.io.json.dumps(x, force_ascii=False) if x is not None else "")
    df_out.to_csv(p, index=False)