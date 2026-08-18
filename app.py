from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from ts_analyzer.basic_statistics import correlation_matrix, numeric_summary
from ts_analyzer.data_loader import find_bundled_workbook, load_excel_timeseries, workbook_sheet_names
from ts_analyzer.selection import filter_time_window, selected_time_range
from ts_analyzer.time_series_statistics import derivative, rolling_statistics, sampling_statistics
from ts_analyzer.visualization import (
    build_boxplot,
    build_correlation_heatmap,
    build_dynamics_figure,
    build_overlay_multi_axis_trend,
    build_shared_axis_trend,
    build_stacked_trend,
)

st.set_page_config(page_title="Time-Series Analyzer", page_icon="📈", layout="wide")
st.title("Time-Series Analyzer")
st.caption("Interactive historian / Excel trend exploration with dynamic statistics")


# -----------------------------------------------------------------------------
# Data source
# -----------------------------------------------------------------------------
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
    file_bytes, source_name = find_bundled_workbook(
        [
            "PG_21_keszulek_07-20-07-22.xlsx",
            "/mnt/data/PG_21_keszulek_07-20-07-22.xlsx",
        ]
    )

if file_bytes is None:
    st.info("Upload an Excel file to begin.")
    st.stop()

try:
    sheets = workbook_sheet_names(file_bytes)
    with st.sidebar:
        sheet = st.selectbox("Sheet", sheets)
    df, tag_map, start_row = load_excel_timeseries(file_bytes, sheet)
except Exception as exc:
    st.error(f"Could not read workbook: {exc}")
    st.stop()

signals = [column for column in df.columns if column != "Timestamp"]
if not signals:
    st.error("No numeric time-series signals were detected.")
    st.stop()

with st.sidebar:
    st.success(f"Loaded: {source_name}")
    st.caption(f"Detected data start: Excel row {start_row + 1}")
    st.divider()
    st.header("Base time window")
    tmin, tmax = df["Timestamp"].min(), df["Timestamp"].max()
    base_start = st.datetime_input(
        "From",
        value=tmin.to_pydatetime(),
        min_value=tmin.to_pydatetime(),
        max_value=tmax.to_pydatetime(),
    )
    base_end = st.datetime_input(
        "To",
        value=tmax.to_pydatetime(),
        min_value=tmin.to_pydatetime(),
        max_value=tmax.to_pydatetime(),
    )

base_df = filter_time_window(df, base_start, base_end)
sampling = sampling_statistics(base_df)
base_span = base_df["Timestamp"].max() - base_df["Timestamp"].min() if len(base_df) else pd.Timedelta(0)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Samples", f"{len(base_df):,}")
m2.metric("Signals", len(signals))
m3.metric("Median sample time", f"{sampling['median_s']:.1f} s" if np.isfinite(sampling["median_s"]) else "—")
m4.metric("Base time span", str(base_span).split(".")[0])

if np.isfinite(sampling["std_s"]) and np.isfinite(sampling["median_s"]):
    if sampling["std_s"] > max(1.0, 0.05 * sampling["median_s"]):
        st.warning(
            "Sampling is not perfectly uniform: "
            f"mean {sampling['mean_s']:.2f} s, median {sampling['median_s']:.2f} s, "
            f"std {sampling['std_s']:.2f} s."
        )


# -----------------------------------------------------------------------------
# Main UI
# -----------------------------------------------------------------------------
trend_tab, ts_tab, data_tab = st.tabs(["Trend Explorer", "Time-Series Statistics", "Data / Export"])

with trend_tab:
    st.subheader("Trend Explorer")
    st.caption(
        "Use the box-select tool on the trend to define the analysis interval. "
        "The table, box plot and correlation below update from that selected time window."
    )

    default_signals = signals[: min(4, len(signals))]
    selected_signals = st.multiselect("Signals", signals, default=default_signals)

    option_col1, option_col2 = st.columns([1, 2])
    with option_col1:
        normalize_trend = st.checkbox("Normalize trend (z-score)", value=False)
    with option_col2:
        trend_mode = st.radio(
            "Trend display",
            ["Shared y-axis", "Separate overlay y-axes", "Stacked subplots"],
            horizontal=True,
        )

    if not selected_signals:
        st.info("Select at least one signal.")
    else:
        axis_map = {}
        if trend_mode == "Separate overlay y-axes":
            max_groups = min(4, len(selected_signals))
            st.caption("Assign signals with comparable engineering units to the same axis group.")
            axis_columns = st.columns(min(4, len(selected_signals)))
            for idx, signal in enumerate(selected_signals):
                with axis_columns[idx % len(axis_columns)]:
                    axis_map[signal] = st.selectbox(
                        f"Axis: {signal}",
                        list(range(1, max_groups + 1)),
                        index=min(idx, max_groups - 1),
                        key=f"axis_group_{signal}",
                    )
            trend_fig = build_overlay_multi_axis_trend(base_df, selected_signals, axis_map, normalize_trend)
        elif trend_mode == "Stacked subplots":
            trend_fig = build_stacked_trend(base_df, selected_signals, normalize_trend)
        else:
            trend_fig = build_shared_axis_trend(base_df, selected_signals, normalize_trend)

        event = st.plotly_chart(
            trend_fig,
            use_container_width=True,
            key="main_trend",
            on_select="rerun",
            selection_mode=("box",),
        )

        selected_range = selected_time_range(event)
        if selected_range is None:
            analysis_df = base_df
            selection_label = "Base sidebar time window"
        else:
            analysis_df = filter_time_window(base_df, selected_range[0], selected_range[1])
            selection_label = f"{selected_range[0]} → {selected_range[1]}"

        if analysis_df.empty:
            st.warning("The current chart selection contains no samples. Using the base sidebar time window instead.")
            analysis_df = base_df
            selection_label = "Base sidebar time window"

        sel_span = analysis_df["Timestamp"].max() - analysis_df["Timestamp"].min() if len(analysis_df) else pd.Timedelta(0)
        s1, s2, s3 = st.columns(3)
        s1.metric("Analyzed samples", f"{len(analysis_df):,}")
        s2.metric("Analyzed span", str(sel_span).split(".")[0])
        s3.metric("Selected signals", len(selected_signals))
        st.caption(f"Statistics window: **{selection_label}**")

        st.markdown("### Dynamic signal statistics")
        summary = numeric_summary(analysis_df, selected_signals)
        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Missing %": st.column_config.NumberColumn(format="%.2f"),
                "Min": st.column_config.NumberColumn(format="%.6g"),
                "Mean": st.column_config.NumberColumn(format="%.6g"),
                "Max": st.column_config.NumberColumn(format="%.6g"),
                "Std": st.column_config.NumberColumn(format="%.6g"),
                "P05": st.column_config.NumberColumn(format="%.6g"),
                "P50": st.column_config.NumberColumn(format="%.6g"),
                "P95": st.column_config.NumberColumn(format="%.6g"),
                "Range": st.column_config.NumberColumn(format="%.6g"),
            },
        )

        st.markdown("### Dynamic box plot")
        normalize_box = st.checkbox("Normalize box plot (z-score)", value=False)
        st.plotly_chart(
            build_boxplot(analysis_df, selected_signals, normalize_box),
            use_container_width=True,
            key="dynamic_boxplot",
        )

        if len(selected_signals) >= 2:
            with st.expander("Correlation for selected interval", expanded=False):
                corr = correlation_matrix(analysis_df, selected_signals)
                st.plotly_chart(build_correlation_heatmap(corr), use_container_width=True, key="dynamic_corr")
                st.dataframe(corr.round(4), use_container_width=True)

with ts_tab:
    st.subheader("Time-Series Statistics")
    st.caption("Reusable time-series diagnostics independent of PID/controller semantics.")

    signal = st.selectbox("Signal", signals, key="ts_signal")
    signal_df = base_df[["Timestamp", signal]].dropna().copy()

    if len(signal_df) > 2:
        max_window = min(500, max(2, len(signal_df) // 4))
        default_window = min(20, max_window)
        window = st.slider("Rolling window [samples]", 2, max_window, default_window)
        rate = derivative(signal_df[signal], signal_df["Timestamp"])
        rolling = rolling_statistics(signal_df[signal], window)
        st.plotly_chart(
            build_dynamics_figure(signal_df, signal, rate, rolling),
            use_container_width=True,
            key="ts_dynamics",
        )

        stats = sampling_statistics(signal_df)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Median Δt", f"{stats['median_s']:.3g} s")
        c2.metric("Mean Δt", f"{stats['mean_s']:.3g} s")
        c3.metric("Std Δt", f"{stats['std_s']:.3g} s")
        c4.metric("Min Δt", f"{stats['min_s']:.3g} s")
        c5.metric("Max Δt", f"{stats['max_s']:.3g} s")

        movements = signal_df.copy()
        movements["Rate per s"] = rate
        movements["Absolute rate"] = np.abs(rate)
        st.markdown("### Largest movements")
        st.dataframe(
            movements.nlargest(20, "Absolute rate")[["Timestamp", signal, "Rate per s"]],
            use_container_width=True,
            hide_index=True,
        )

with data_tab:
    st.subheader("Data / Export")
    st.dataframe(base_df.head(500), use_container_width=True, hide_index=True)
    st.caption("Preview is limited to 500 rows. The download contains the complete base time window.")

    csv = base_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download base time window as CSV",
        data=csv,
        file_name="timeseries_selected.csv",
        mime="text/csv",
    )

    st.markdown("### Detected tag/value mapping")
    mapping = pd.DataFrame(
        tag_map,
        columns=["Tag-name column (0-based)", "Value column (0-based)", "Detected signal"],
    )
    st.dataframe(mapping, use_container_width=True, hide_index=True)
