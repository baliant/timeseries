import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Time-Series & PI Loop Analyzer", page_icon="📈", layout="wide")

st.title("Time-Series & PI Loop Analyzer")
st.caption("For PI/Excel historian exports and general process-control trend analysis")


def _clean_name(x):
    return str(x).strip() if pd.notna(x) else ""


def detect_data_start(raw: pd.DataFrame) -> int:
    """Find first row where col 0 looks like datetime and at least one later numeric value exists."""
    for i in range(min(len(raw), 100)):
        ts = pd.to_datetime(raw.iloc[i, 0], errors="coerce")
        if pd.isna(ts):
            continue
        nums = pd.to_numeric(raw.iloc[i, 1:], errors="coerce").notna().sum()
        if nums >= 1:
            return i
    return 0


def normalize_historian_export(raw: pd.DataFrame):
    """
    Accepts the uploaded workbook pattern:
      timestamp | tag | value | tag | value | ...
    where tag-name columns repeat the same tag on each row.
    Returns a wide dataframe indexed by timestamp.
    """
    start = detect_data_start(raw)
    df = raw.iloc[start:].copy().reset_index(drop=True)
    ts = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    valid = ts.notna()
    df = df.loc[valid].reset_index(drop=True)
    ts = ts.loc[valid].reset_index(drop=True)

    out = pd.DataFrame({"Timestamp": ts})
    tag_map = []

    c = 1
    while c < df.shape[1]:
        # If current column looks mostly textual, assume tag-name + next value pair.
        col = df.iloc[:, c]
        non_null = col.dropna()
        text_ratio = 0 if len(non_null) == 0 else non_null.map(lambda x: isinstance(x, str)).mean()

        if text_ratio > 0.5 and c + 1 < df.shape[1]:
            names = [_clean_name(x) for x in non_null.head(200)]
            names = [x for x in names if x]
            tag = pd.Series(names).mode().iloc[0] if names else f"Signal_{c+1}"
            vals = pd.to_numeric(df.iloc[:, c + 1], errors="coerce")
            final_name = tag
            suffix = 2
            while final_name in out.columns:
                final_name = f"{tag}_{suffix}"
                suffix += 1
            out[final_name] = vals.values
            tag_map.append((c, c + 1, final_name))
            c += 2
        else:
            vals = pd.to_numeric(col, errors="coerce")
            if vals.notna().sum() > 0:
                name = f"Signal_{c+1}"
                out[name] = vals.values
                tag_map.append((None, c, name))
            c += 1

    out = out.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    out = out.drop_duplicates(subset=["Timestamp"], keep="last").reset_index(drop=True)
    return out, tag_map, start


def read_uploaded_excel(file_bytes: bytes, sheet_name=0):
    raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, header=None)
    return normalize_historian_export(raw)


def sampling_stats(df):
    if len(df) < 2:
        return np.nan, np.nan, np.nan
    dt = df["Timestamp"].diff().dt.total_seconds().dropna()
    return dt.median(), dt.mean(), dt.std()


def numeric_summary(df, cols):
    rows = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        rows.append({
            "Signal": c,
            "Count": int(s.notna().sum()),
            "Missing %": 100 * s.isna().mean(),
            "Min": s.min(),
            "Mean": s.mean(),
            "Max": s.max(),
            "Std": s.std(),
            "P05": s.quantile(0.05),
            "P50": s.quantile(0.50),
            "P95": s.quantile(0.95),
        })
    return pd.DataFrame(rows)


def infer_candidates(columns, tokens):
    result = []
    for c in columns:
        uc = c.upper()
        if any(t in uc for t in tokens):
            result.append(c)
    return result


def derivative(series, time_s):
    y = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    t = np.asarray(time_s, dtype=float)
    if len(y) < 3:
        return np.full(len(y), np.nan)
    return np.gradient(y, t)


def calc_pid_metrics(df, pv, sp, out, band=None):
    tmp = df[["Timestamp", pv, sp, out]].copy().dropna()
    if tmp.empty:
        return {}, tmp
    tmp["Error"] = tmp[sp] - tmp[pv]
    tmp["AbsError"] = tmp["Error"].abs()
    tmp["SquaredError"] = tmp["Error"] ** 2
    dt = tmp["Timestamp"].diff().dt.total_seconds().fillna(0)
    tmp["dt_s"] = dt
    duration = max((tmp["Timestamp"].iloc[-1] - tmp["Timestamp"].iloc[0]).total_seconds(), 1)
    ise = float((tmp["SquaredError"] * tmp["dt_s"]).sum())
    iae = float((tmp["AbsError"] * tmp["dt_s"]).sum())
    itae = float((tmp["AbsError"] * (tmp["Timestamp"] - tmp["Timestamp"].iloc[0]).dt.total_seconds() * tmp["dt_s"]).sum())
    d_out = tmp[out].diff().abs()
    metrics = {
        "Mean error": float(tmp["Error"].mean()),
        "MAE": float(tmp["AbsError"].mean()),
        "RMSE": float(np.sqrt(tmp["SquaredError"].mean())),
        "Max |error|": float(tmp["AbsError"].max()),
        "IAE": iae,
        "ISE": ise,
        "ITAE": itae,
        "Valve travel Σ|ΔOUT|": float(d_out.sum()),
        "OUT std": float(tmp[out].std()),
        "Duration h": duration / 3600,
    }
    if band is not None:
        metrics[f"Time in ±{band:g} band %"] = 100 * float((tmp["AbsError"] <= band).mean())
    return metrics, tmp


def estimate_pi_from_data(tmp, pv, sp, out, pv_span=None, out_span=100.0, min_error_change=1e-9):
    """
    Experimental linear least-squares estimate from positional PI equation:
       du/dt = Kp * de/dt + (Kp/Ti) * e
    This is only meaningful if signal scaling/action is correct and loop isn't saturated/noisy.
    """
    if len(tmp) < 20:
        return None
    t = (tmp["Timestamp"] - tmp["Timestamp"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    e_eng = (tmp[sp] - tmp[pv]).to_numpy(dtype=float)
    u = tmp[out].to_numpy(dtype=float)

    if pv_span is None or pv_span == 0:
        pv_span = np.nanpercentile(np.r_[tmp[pv].to_numpy(), tmp[sp].to_numpy()], 95) - np.nanpercentile(np.r_[tmp[pv].to_numpy(), tmp[sp].to_numpy()], 5)
    if not np.isfinite(pv_span) or pv_span == 0:
        return None

    e_pct = e_eng / pv_span * 100.0
    u_pct = u / out_span * 100.0 if out_span != 100 else u.copy()
    de_dt = derivative(e_pct, t)
    du_dt = derivative(u_pct, t)

    X = np.column_stack([de_dt, e_pct])
    mask = np.isfinite(X).all(axis=1) & np.isfinite(du_dt)
    # Exclude obviously flat, uninformative rows.
    mask &= (np.abs(e_pct) + np.abs(de_dt)) > min_error_change
    if mask.sum() < 20:
        return None
    X2 = X[mask]
    y2 = du_dt[mask]
    beta, *_ = np.linalg.lstsq(X2, y2, rcond=None)
    kp = beta[0]
    ki = beta[1]
    ti = kp / ki if abs(ki) > 1e-12 else np.nan
    yhat = X2 @ beta
    ss_res = np.sum((y2-yhat)**2)
    ss_tot = np.sum((y2-np.mean(y2))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return {"Kp_est": kp, "Ki_est_per_s": ki, "Ti_est_s": ti, "R2": r2, "PV_span_used": pv_span}


with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload Excel time-series", type=["xlsx", "xls"])
    use_example = st.checkbox("Use bundled workbook if available", value=uploaded is None)

file_bytes = None
source_name = None
if uploaded is not None:
    file_bytes = uploaded.getvalue()
    source_name = uploaded.name
elif use_example:
    local = Path("PG_21_keszulek_07-20-07-22.xlsx")
    if local.exists():
        file_bytes = local.read_bytes()
        source_name = local.name
    else:
        st.info("Upload the Excel file to begin. If you run this app beside the supplied workbook, it can load it automatically.")

if file_bytes is None:
    st.stop()

try:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    with st.sidebar:
        sheet = st.selectbox("Sheet", xls.sheet_names)
    df, tag_map, start_row = read_uploaded_excel(file_bytes, sheet)
except Exception as exc:
    st.error(f"Could not read workbook: {exc}")
    st.stop()

signals = [c for c in df.columns if c != "Timestamp"]
if not signals:
    st.error("No numeric time-series signals detected.")
    st.stop()

with st.sidebar:
    st.success(f"Loaded: {source_name}")
    st.caption(f"Detected data start: Excel row {start_row+1}")
    st.divider()
    st.header("Time window")
    tmin, tmax = df["Timestamp"].min(), df["Timestamp"].max()
    start_dt = st.datetime_input("From", value=tmin.to_pydatetime(), min_value=tmin.to_pydatetime(), max_value=tmax.to_pydatetime())
    end_dt = st.datetime_input("To", value=tmax.to_pydatetime(), min_value=tmin.to_pydatetime(), max_value=tmax.to_pydatetime())

filtered = df[(df["Timestamp"] >= pd.Timestamp(start_dt)) & (df["Timestamp"] <= pd.Timestamp(end_dt))].copy()

median_dt, mean_dt, std_dt = sampling_stats(filtered)
span = filtered["Timestamp"].max() - filtered["Timestamp"].min() if len(filtered) else pd.Timedelta(0)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Samples", f"{len(filtered):,}")
c2.metric("Signals", len(signals))
c3.metric("Median sample time", f"{median_dt:.1f} s" if np.isfinite(median_dt) else "—")
c4.metric("Time span", str(span).split(".")[0])

if np.isfinite(std_dt) and std_dt > max(1.0, 0.05 * median_dt):
    st.warning(f"Sampling is not perfectly uniform: mean {mean_dt:.2f} s, median {median_dt:.2f} s, std {std_dt:.2f} s.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Trend Explorer", "Signal Statistics", "PI Loop Analyzer", "Dynamics", "Data / Export"])

with tab1:
    st.subheader("Interactive trends")
    default = signals[:min(4, len(signals))]
    selected = st.multiselect("Signals", signals, default=default)
    normalize = st.checkbox("Normalize selected signals (z-score)", value=False)
    if selected:
        fig = go.Figure()
        for c in selected:
            y = filtered[c]
            if normalize:
                sd = y.std()
                y = (y-y.mean())/sd if sd and np.isfinite(sd) else y*0
            fig.add_trace(go.Scattergl(x=filtered["Timestamp"], y=y, mode="lines", name=c))
        fig.update_layout(height=600, hovermode="x unified", xaxis_title="Time", yaxis_title="z-score" if normalize else "Engineering value", legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Correlation")
        corr = filtered[selected].corr(numeric_only=True)
        st.dataframe(corr.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1), use_container_width=True)

with tab2:
    st.subheader("Signal statistics")
    summary = numeric_summary(filtered, signals)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Distribution")
    sig = st.selectbox("Signal for histogram", signals, key="hist_sig")
    bins = st.slider("Bins", 10, 150, 50)
    hist = go.Figure(go.Histogram(x=filtered[sig].dropna(), nbinsx=bins))
    hist.update_layout(height=400, xaxis_title=sig, yaxis_title="Count")
    st.plotly_chart(hist, use_container_width=True)

with tab3:
    st.subheader("PI loop performance")
    st.caption("Assign the actual process variable (PV), setpoint (SP), and controller output (OUT). The app does not assume tag naming is correct.")

    pv_candidates = infer_candidates(signals, ["PV", "TT", "PT", "TIT", "PIT"])
    sp_candidates = infer_candidates(signals, ["SP", "SETPOINT"])
    out_candidates = infer_candidates(signals, ["OUT", "OP", "CV", "VALVE", "AO"])

    col1,col2,col3 = st.columns(3)
    pv = col1.selectbox("PV", signals, index=signals.index(pv_candidates[0]) if pv_candidates else 0)
    sp_options = ["<none>"] + signals
    sp_idx = 0
    if sp_candidates:
        sp_idx = sp_options.index(sp_candidates[0])
    sp = col2.selectbox("SP", sp_options, index=sp_idx)
    out_idx = signals.index(out_candidates[-1]) if out_candidates else min(1, len(signals)-1)
    out = col3.selectbox("OUT / valve position", signals, index=out_idx)

    if sp == "<none>":
        st.info("Select an SP signal to calculate control error and PI performance metrics.")
    else:
        pv_range = float(pd.concat([filtered[pv], filtered[sp]]).max() - pd.concat([filtered[pv], filtered[sp]]).min())
        default_band = max(pv_range * 0.01, 0.01)
        band = st.number_input("Acceptable absolute PV error band (engineering units)", min_value=0.0, value=float(default_band), format="%.6g")
        metrics, loop = calc_pid_metrics(filtered, pv, sp, out, band)

        mcols = st.columns(5)
        for idx, key in enumerate(["Mean error","MAE","RMSE","Max |error|",f"Time in ±{band:g} band %"]):
            if key in metrics:
                mcols[idx].metric(key, f"{metrics[key]:.4g}")
        m2 = st.columns(4)
        for idx, key in enumerate(["IAE","ISE","Valve travel Σ|ΔOUT|","OUT std"]):
            m2[idx].metric(key, f"{metrics[key]:.5g}")

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.45,0.25,0.30])
        fig.add_trace(go.Scattergl(x=loop["Timestamp"], y=loop[pv], name=f"PV: {pv}"), row=1,col=1)
        fig.add_trace(go.Scattergl(x=loop["Timestamp"], y=loop[sp], name=f"SP: {sp}"), row=1,col=1)
        fig.add_trace(go.Scattergl(x=loop["Timestamp"], y=loop["Error"], name="Error SP-PV"), row=2,col=1)
        fig.add_hline(y=band, line_dash="dot", row=2,col=1)
        fig.add_hline(y=-band, line_dash="dot", row=2,col=1)
        fig.add_trace(go.Scattergl(x=loop["Timestamp"], y=loop[out], name=f"OUT: {out}"), row=3,col=1)
        fig.update_layout(height=780, hovermode="x unified", legend=dict(orientation="h"))
        fig.update_yaxes(title_text="PV / SP", row=1,col=1)
        fig.update_yaxes(title_text="Error", row=2,col=1)
        fig.update_yaxes(title_text="OUT", row=3,col=1)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### PI interpretation calculator")
        a,b,c = st.columns(3)
        kp = a.number_input("Controller gain Kp", value=2.0, step=0.1)
        ti = b.number_input("Reset / integral time Ti [s]", value=60.0, min_value=0.001)
        err_pct = c.number_input("Constant error [% of PV span]", value=1.0, step=0.1)
        p_action = kp * err_pct
        i_rate = kp * err_pct / ti
        st.write(f"For a sustained **{err_pct:g}%** error: proportional action = **{p_action:g}% OUT immediately**, integral ramp = **{i_rate:g}% OUT/s** = **{i_rate*60:g}% OUT/min**. After one Ti ({ti:g} s), the integral contribution equals **{p_action:g}% OUT**.")

        with st.expander("Experimental: estimate PI parameters from PV/SP/OUT data"):
            st.warning("Use this only as a diagnostic. A closed-loop trend generally cannot identify tuning reliably unless signal scaling, controller action, output limits, filtering, and disturbances are understood.")
            pvr = st.number_input("Configured PV span (URV-LRV)", min_value=0.0, value=float(max(pv_range, 0.001)), format="%.6g")
            outr = st.number_input("Configured OUT span", min_value=0.001, value=100.0)
            if st.button("Estimate Kp and Ti"):
                est = estimate_pi_from_data(loop, pv, sp, out, pvr, outr)
                if est is None:
                    st.error("Not enough informative variation to estimate the PI parameters.")
                else:
                    ec1,ec2,ec3,ec4 = st.columns(4)
                    ec1.metric("Estimated Kp", f"{est['Kp_est']:.4g}")
                    ec2.metric("Estimated Ti", f"{est['Ti_est_s']:.4g} s")
                    ec3.metric("Estimated Ki", f"{est['Ki_est_per_s']:.4g} 1/s")
                    ec4.metric("Fit R²", f"{est['R2']:.3f}")
                    if not np.isfinite(est['Ti_est_s']) or est['Ti_est_s'] <= 0 or est['R2'] < 0.3:
                        st.warning("The estimate is weak/non-physical. Do not use it as controller tuning without engineering review.")

with tab4:
    st.subheader("Dynamics and movement")
    sig = st.selectbox("Signal", signals, key="dyn_sig")
    tmp = filtered[["Timestamp", sig]].dropna().copy()
    if len(tmp) > 2:
        tsec = (tmp["Timestamp"] - tmp["Timestamp"].iloc[0]).dt.total_seconds().to_numpy()
        tmp["Rate_per_s"] = derivative(tmp[sig], tsec)
        win = st.slider("Rolling window [samples]", 2, min(200, max(2,len(tmp)//5)), min(20, max(2,len(tmp)//10)))
        tmp["RollingMean"] = tmp[sig].rolling(win, center=True).mean()
        tmp["RollingStd"] = tmp[sig].rolling(win, center=True).std()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scattergl(x=tmp["Timestamp"], y=tmp[sig], name=sig), row=1,col=1)
        fig.add_trace(go.Scattergl(x=tmp["Timestamp"], y=tmp["RollingMean"], name=f"Rolling mean ({win})"), row=1,col=1)
        fig.add_trace(go.Scattergl(x=tmp["Timestamp"], y=tmp["Rate_per_s"], name="Rate of change /s"), row=2,col=1)
        fig.update_layout(height=650, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Largest movements")
        moves = tmp.assign(AbsRate=tmp["Rate_per_s"].abs()).nlargest(20, "AbsRate")[["Timestamp",sig,"Rate_per_s"]]
        st.dataframe(moves, use_container_width=True, hide_index=True)

with tab5:
    st.subheader("Normalized data")
    st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)
    st.caption("Preview limited to first 500 rows; downloads contain the full selected time window.")

    csv = filtered.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download selected window as CSV", csv, file_name="timeseries_selected.csv", mime="text/csv")

    st.markdown("#### Detected tag/value mapping")
    mapping_df = pd.DataFrame(tag_map, columns=["Tag-name column (0-based)", "Value column (0-based)", "Detected signal"])
    st.dataframe(mapping_df, use_container_width=True, hide_index=True)

st.divider()
st.caption("Engineering note: controller performance metrics are only meaningful when PV, SP and OUT are correctly assigned and scaled. Saturation, cascade/RCAS mode, split-range logic, filtering, bad-status periods and manual mode can invalidate tuning conclusions.")
