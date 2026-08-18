from __future__ import annotations

import pandas as pd


def clean_name(value) -> str:
    return str(value).strip() if pd.notna(value) else ""


def detect_data_start(raw: pd.DataFrame, scan_rows: int = 100) -> int:
    """Return the first row that looks like the beginning of time-series data."""
    for i in range(min(len(raw), scan_rows)):
        ts = pd.to_datetime(raw.iloc[i, 0], errors="coerce")
        if pd.isna(ts):
            continue
        numeric_count = pd.to_numeric(raw.iloc[i, 1:], errors="coerce").notna().sum()
        if numeric_count >= 1:
            return i
    return 0


def normalize_historian_export(raw: pd.DataFrame):
    """Normalize a historian export into one Timestamp column plus numeric signal columns.

    Supported layout:
        timestamp | tag | value | tag | value | ...

    The tag-name columns may repeat the same tag name on every row. Numeric-only columns
    are also preserved as generic signals.
    """
    start = detect_data_start(raw)
    df = raw.iloc[start:].copy().reset_index(drop=True)

    timestamps = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    valid = timestamps.notna()
    df = df.loc[valid].reset_index(drop=True)
    timestamps = timestamps.loc[valid].reset_index(drop=True)

    output = pd.DataFrame({"Timestamp": timestamps})
    tag_map: list[tuple[int | None, int, str]] = []

    c = 1
    while c < df.shape[1]:
        column = df.iloc[:, c]
        non_null = column.dropna()
        text_ratio = 0.0 if non_null.empty else non_null.map(lambda x: isinstance(x, str)).mean()

        if text_ratio > 0.5 and c + 1 < df.shape[1]:
            names = [clean_name(x) for x in non_null.head(200)]
            names = [x for x in names if x]
            tag = pd.Series(names).mode().iloc[0] if names else f"Signal_{c + 1}"
            values = pd.to_numeric(df.iloc[:, c + 1], errors="coerce")

            final_name = tag
            suffix = 2
            while final_name in output.columns:
                final_name = f"{tag}_{suffix}"
                suffix += 1

            output[final_name] = values.to_numpy()
            tag_map.append((c, c + 1, final_name))
            c += 2
        else:
            values = pd.to_numeric(column, errors="coerce")
            if values.notna().sum() > 0:
                name = f"Signal_{c + 1}"
                output[name] = values.to_numpy()
                tag_map.append((None, c, name))
            c += 1

    output = (
        output.dropna(subset=["Timestamp"])
        .sort_values("Timestamp")
        .drop_duplicates(subset=["Timestamp"], keep="last")
        .reset_index(drop=True)
    )
    return output, tag_map, start
