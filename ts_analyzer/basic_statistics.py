from __future__ import annotations

import numpy as np
import pandas as pd


SUMMARY_COLUMNS = [
    "Signal",
    "Count",
    "Missing %",
    "Min",
    "Mean",
    "Max",
    "Std",
    "P05",
    "P50",
    "P95",
    "Range",
]


def numeric_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Descriptive statistics for the requested signals."""
    rows = []
    for column in columns:
        s = pd.to_numeric(df[column], errors="coerce")
        minimum = s.min()
        maximum = s.max()
        rows.append(
            {
                "Signal": column,
                "Count": int(s.notna().sum()),
                "Missing %": float(100 * s.isna().mean()),
                "Min": minimum,
                "Mean": s.mean(),
                "Max": maximum,
                "Std": s.std(),
                "P05": s.quantile(0.05),
                "P50": s.quantile(0.50),
                "P95": s.quantile(0.95),
                "Range": maximum - minimum if pd.notna(maximum) and pd.notna(minimum) else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame()
    return df[columns].corr(numeric_only=True)
