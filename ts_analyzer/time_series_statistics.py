from __future__ import annotations

import numpy as np
import pandas as pd


def sampling_statistics(df: pd.DataFrame) -> dict[str, float]:
    if len(df) < 2:
        return {"median_s": np.nan, "mean_s": np.nan, "std_s": np.nan, "min_s": np.nan, "max_s": np.nan}
    dt = df["Timestamp"].diff().dt.total_seconds().dropna()
    return {
        "median_s": float(dt.median()),
        "mean_s": float(dt.mean()),
        "std_s": float(dt.std()),
        "min_s": float(dt.min()),
        "max_s": float(dt.max()),
    }


def derivative(series: pd.Series, timestamps: pd.Series) -> np.ndarray:
    y = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if len(y) < 3:
        return np.full(len(y), np.nan)
    t = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    return np.gradient(y, t)


def rolling_statistics(series: pd.Series, window: int) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce")
    return pd.DataFrame(
        {
            "Rolling mean": s.rolling(window, center=True).mean(),
            "Rolling std": s.rolling(window, center=True).std(),
            "Rolling min": s.rolling(window, center=True).min(),
            "Rolling max": s.rolling(window, center=True).max(),
        },
        index=series.index,
    )
