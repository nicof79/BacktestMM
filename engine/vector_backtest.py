"""
engine/vector_backtest.py

Vectorized helpers to generate crossover signals and to scan many MA combinations quickly.

Public API:
- generate_cross_signals(df, short_col, long_col) -> pd.Series
- vector_scan(df, periods, ma_types, ratio_min, ratio_max, metrics_config) -> pd.DataFrame

Notes:
- This module does quick, portfolio-free metrics based on price series and signals.
- It pairs buy and sell signals in FIFO order and computes pct change per closed trade.
- Execution prices are taken as the next-row Close (i.e., execution at Close of next bar).
"""
from typing import List, Dict, Any
import itertools
import pandas as pd
import numpy as np
from indicators.ma import calculate_ma


def generate_cross_signals(df: pd.DataFrame, short_col: str, long_col: str) -> pd.Series:
    """
    Generate crossover signals from two columns in df.
    Returns a pd.Series indexed like df with values {1, -1, 0}.
    1 indicates a bullish crossing (short crosses above long), -1 a bearish crossing.
    """
    delta = df[short_col] - df[long_col]
    delta_prev = delta.shift(1)
    buy = (delta_prev <= 0) & (delta > 0)
    sell = (delta_prev >= 0) & (delta < 0)
    signals = pd.Series(0, index=df.index)
    signals.loc[buy] = 1
    signals.loc[sell] = -1
    return signals.astype(int)


def _pair_trades_from_signals(df: pd.DataFrame, signals: pd.Series) -> List[float]:
    """
    Robust pairing of buys and sells using positional indexing to get next-row execution prices.
    Execution price for buy and sell is taken from Close at next row (i_pos + 1).
    Unclosed buys at the end are ignored.
    """
    closes = df['Close'].values
    index = df.index
    entry_idxs = list(signals[signals == 1].index)
    exit_idxs = list(signals[signals == -1].index)

    # helper to get next-row close given a timestamp index; returns None if next row missing or NaN
    def next_close(ts):
        try:
            pos = df.index.get_loc(ts)
            next_pos = pos + 1
            if next_pos >= len(closes):
                return None
            val = float(closes[next_pos])
            return val if np.isfinite(val) else None
        except Exception:
            return None

    pct_changes = []
    ei = 0
    xi = 0
    while ei < len(entry_idxs) and xi < len(exit_idxs):
        if entry_idxs[ei] < exit_idxs[xi]:
            buy_ts = entry_idxs[ei]
            sell_ts = exit_idxs[xi]
            buy_price = next_close(buy_ts)
            sell_price = next_close(sell_ts)
            if buy_price is not None and sell_price is not None and buy_price > 0:
                pct = (sell_price - buy_price) / buy_price * 100.0
                pct_changes.append(pct)
            ei += 1
            xi += 1
        else:
            xi += 1
    return pct_changes

def vector_scan(df: pd.DataFrame,
                periods: List[int],
                ma_types: List[str],
                ratio_min: float,
                ratio_max: float,
                metrics_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Scan many MA short/long combinations and compute quick metrics.

    Args:
      df: DataFrame with at least column 'Close'.
      periods: list of int periods to combine.
      ma_types: list of strings ["SMA","EMA",...]
      ratio_min, ratio_max: filter on long/short ratio.
      metrics_config: dict with keys:
        - metric: string name to sort by (supported: 'mean_return_per_trade', 'win_rate', 'nb_trades')
        - min_trades: int minimum trades to keep the combo

    Returns:
      pd.DataFrame with one row per tested combination and the following columns:
       ['type_short','type_long','p_short','p_long','combination','nb_trades','win_rate',
        'mean_return_per_trade','median_return_per_trade','pct_positive_trades']
      Sorted descending by metrics_config['metric'].
    """
    if 'Close' not in df.columns:
        raise ValueError("df must contain 'Close' column")

    results = []
    metric = metrics_config.get('metric', 'mean_return_per_trade')
    min_trades = int(metrics_config.get('min_trades', 0))

    # Precompute nothing here; compute MAs per combo to keep memory small
    combos = []
    for t_short, t_long in itertools.product(ma_types, ma_types):
        for p_short, p_long in itertools.product(periods, periods):
            if p_short >= p_long:
                continue
            ratio = p_long / p_short
            if ratio < ratio_min or ratio > ratio_max:
                continue
            # require at least p_long + 1 bars for next-row execution
            if len(df) < p_long + 1:
                continue
            combos.append((t_short, t_long, p_short, p_long))

    for t_short, t_long, p_short, p_long in combos:
        short_col = f"MA_{t_short}{p_short}"
        long_col = f"MA_{t_long}{p_long}"

        # compute MAs (calculate_ma handles NaNs properly)
        df_short = calculate_ma(df['Close'], p_short, t_short)
        df_long = calculate_ma(df['Close'], p_long, t_long)

        temp = df.copy()
        temp[short_col] = df_short
        temp[long_col] = df_long

        signals = generate_cross_signals(temp, short_col, long_col)
        pct_changes = _pair_trades_from_signals(temp, signals)

        nb_trades = len(pct_changes)
        if nb_trades < min_trades:
            continue

        pct_arr = np.array(pct_changes) if pct_changes else np.array([])
        mean_ret = float(np.mean(pct_arr)) if pct_arr.size > 0 else 0.0
        median_ret = float(np.median(pct_arr)) if pct_arr.size > 0 else 0.0
        pct_positive = float((pct_arr > 0).sum() / pct_arr.size * 100.0) if pct_arr.size > 0 else 0.0
        win_rate = pct_positive  # same as pct_positive

        results.append({
            'type_short': t_short,
            'type_long': t_long,
            'p_short': int(p_short),
            'p_long': int(p_long),
            'combination': f"{t_short}{p_short}/{t_long}{p_long}",
            'nb_trades': int(nb_trades),
            'win_rate': float(win_rate),
            'mean_return_per_trade': float(mean_ret),
            'median_return_per_trade': float(median_ret),
            'pct_positive_trades': float(pct_positive)
        })

    if not results:
        return pd.DataFrame(columns=[
            'type_short','type_long','p_short','p_long','combination','nb_trades','win_rate',
            'mean_return_per_trade','median_return_per_trade','pct_positive_trades'
        ])

    results_df = pd.DataFrame(results)
    if metric not in results_df.columns:
        metric = 'mean_return_per_trade'
    results_df = results_df.sort_values(by=metric, ascending=False).reset_index(drop=True)
    return results_df