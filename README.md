# Modular Time-Series Analyzer

A Streamlit application for interactive historian/Excel trend analysis.

## Architecture

- `app.py` — UI orchestration only
- `ts_analyzer/data_loader.py` — workbook/sheet loading
- `ts_analyzer/normalizer.py` — historian export normalization
- `ts_analyzer/selection.py` — time-window filtering and Plotly selection parsing
- `ts_analyzer/basic_statistics.py` — Min/Mean/Max/Std/P05/P50/P95/etc.
- `ts_analyzer/time_series_statistics.py` — sampling, derivative and rolling statistics
- `ts_analyzer/visualization.py` — reusable Plotly trend, boxplot, correlation and dynamics figures

PID/PI-specific functions are intentionally not included in this version.

## Dynamic Trend Explorer

The Trend Explorer uses Plotly box selection. Drag over the desired time interval on the main trend. The following components recalculate from the selected interval:

- sample count and selected duration
- Min / Mean / Max
- standard deviation
- P05 / P50 / P95
- range and missing percentage
- box plot
- correlation matrix

If no box selection is active, the sidebar time window is used.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud

Put `app.py`, `requirements.txt`, and the complete `ts_analyzer/` folder in the same repository. Use `app.py` as the Streamlit entry point.
