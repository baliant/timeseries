# Time-Series & PI Loop Analyzer

A Streamlit application for PI/Excel historian exports and process-control trend analysis.

## Run

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run timeseries_pid_analyzer.py
```

Open the URL shown by Streamlit, normally `http://localhost:8501`.

## Workbook handling

The supplied workbook uses a layout like:

`Timestamp | Tag name | Value | Tag name | Value | ...`

The app automatically detects the first data row, extracts each repeated tag name, and converts the export into a normal wide time-series table.

## Functions

- interactive multi-signal trend plotting
- time-window selection
- signal statistics, percentiles, missing-data overview
- correlation matrix and histograms
- PV/SP/OUT assignment for a PI loop
- error trend, IAE, ISE, ITAE, RMSE, MAE
- time-inside-error-band calculation
- controller output / valve travel metric
- PI interpretation calculator for Kp, Ti and constant error
- experimental regression-based Kp/Ti estimation from trend data
- rate-of-change and largest-movement analysis
- CSV export of the selected window

## Important engineering limitation

A closed-loop trend alone generally does **not** uniquely identify the controller tuning. The experimental Kp/Ti estimator is diagnostic only. Controller action, configured PV span, OUT scaling, filtering, output limits, manual/cascade states, saturation, split-range logic and disturbances must be considered before using any inferred parameters.
