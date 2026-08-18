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


def _parse_timestamp(value):
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _range_from_box(selection):
    """Try to extract the exact x-range from Streamlit's Plotly box metadata.

    Streamlit exposes box-selection metadata, but the exact nested shape may vary
    with Plotly/Streamlit versions. Support the common representations and fall
    back to selected point x-values when necessary.
    """
    boxes = _get(selection, "box", []) or []
    for box in boxes:
        # Common representation: {"x": [x0, x1], "y": [y0, y1]}
        x_range = _get(box, "x")
        if isinstance(x_range, (list, tuple)) and len(x_range) >= 2:
            x0 = _parse_timestamp(x_range[0])
            x1 = _parse_timestamp(x_range[1])
            if x0 is not None and x1 is not None:
                return (min(x0, x1), max(x0, x1))

        # Alternate representation: {"x0": ..., "x1": ...}
        x0 = _parse_timestamp(_get(box, "x0"))
        x1 = _parse_timestamp(_get(box, "x1"))
        if x0 is not None and x1 is not None:
            return (min(x0, x1), max(x0, x1))

        # Alternate nested representation: {"range": {"x": [x0, x1]}}
        box_range = _get(box, "range", {}) or {}
        x_range = _get(box_range, "x")
        if isinstance(x_range, (list, tuple)) and len(x_range) >= 2:
            x0 = _parse_timestamp(x_range[0])
            x1 = _parse_timestamp(x_range[1])
            if x0 is not None and x1 is not None:
                return (min(x0, x1), max(x0, x1))

    return None


def selected_time_range(plot_event):
    """Extract a time interval from a Streamlit Plotly selection event.

    Preference order:
      1. Exact box x-coordinates.
      2. Min/max x of selected points.

    Returns (start, end) as pandas Timestamps, or None when no valid time
    selection is present.
    """
    selection = _get(plot_event, "selection", {}) or {}

    box_range = _range_from_box(selection)
    if box_range is not None:
        return box_range

    points = _get(selection, "points", []) or []
    x_values = []
    for point in points:
        parsed = _parse_timestamp(_get(point, "x"))
        if parsed is not None:
            x_values.append(parsed)

    if not x_values:
        return None
    return pd.Timestamp(min(x_values)), pd.Timestamp(max(x_values))


def clamp_time_range(start, end, minimum, maximum):
    """Clamp an interval to available data bounds."""
    start_ts = max(pd.Timestamp(start), pd.Timestamp(minimum))
    end_ts = min(pd.Timestamp(end), pd.Timestamp(maximum))
    if start_ts > end_ts:
        return pd.Timestamp(minimum), pd.Timestamp(maximum)
    return start_ts, end_ts
