from __future__ import annotations

import pandas as pd


def filter_time_window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        start_ts, end_ts = end_ts, start_ts
    return df[(df["Timestamp"] >= start_ts) & (df["Timestamp"] <= end_ts)].copy()


def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def selected_time_range(plot_event):
    """Extract min/max x from a Streamlit Plotly selection event.

    Returns (start, end) as pandas Timestamps, or None when no points are selected.
    """
    selection = _get(plot_event, "selection", {})
    points = _get(selection, "points", []) or []

    x_values = []
    for point in points:
        x = _get(point, "x")
        if x is not None:
            parsed = pd.to_datetime(x, errors="coerce")
            if pd.notna(parsed):
                x_values.append(parsed)

    if not x_values:
        return None
    return pd.Timestamp(min(x_values)), pd.Timestamp(max(x_values))
