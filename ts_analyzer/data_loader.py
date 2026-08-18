from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from .normalizer import normalize_historian_export


def workbook_sheet_names(file_bytes: bytes) -> list[str]:
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


def load_excel_timeseries(file_bytes: bytes, sheet_name=0):
    raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=None)
    return normalize_historian_export(raw)


def find_bundled_workbook(candidates: list[str | Path]) -> tuple[bytes | None, str | None]:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path.read_bytes(), path.name
    return None, None
