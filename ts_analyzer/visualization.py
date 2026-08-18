from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _plot_series(series: pd.Series, normalize: bool) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if not normalize:
        return s
    std = s.std()
    return (s - s.mean()) / std if std and np.isfinite(std) else s * 0


def build_shared_axis_trend(df: pd.DataFrame, signals: list[str], normalize: bool = False) -> go.Figure:
    fig = go.Figure()
    for signal in signals:
        fig.add_trace(
            go.Scattergl(
                x=df["Timestamp"],
                y=_plot_series(df[signal], normalize),
                mode="lines",
                name=signal,
            )
        )
    fig.update_layout(
        height=650,
        hovermode="x unified",
        dragmode="select",
        xaxis_title="Time",
        yaxis_title="z-score" if normalize else "Engineering value",
        legend=dict(orientation="h"),
        margin=dict(l=70, r=70, t=45, b=60),
    )
    return fig


def build_overlay_multi_axis_trend(
    df: pd.DataFrame,
    signals: list[str],
    axis_map: dict[str, int],
    normalize: bool = False,
) -> go.Figure:
    fig = go.Figure()
    axis_groups = sorted({axis_map[s] for s in signals})
    side_cycle = {1: "left", 2: "right", 3: "left", 4: "right"}
    positions = {1: 0.0, 2: 1.0, 3: 0.06, 4: 0.94}

    for signal in signals:
        axis_num = axis_map[signal]
        axis_ref = "y" if axis_num == 1 else f"y{axis_num}"
        fig.add_trace(
            go.Scattergl(
                x=df["Timestamp"],
                y=_plot_series(df[signal], normalize),
                mode="lines",
                name=signal,
                yaxis=axis_ref,
            )
        )

    layout = dict(
        height=650,
        hovermode="x unified",
        dragmode="select",
        xaxis=dict(title="Time"),
        legend=dict(orientation="h"),
        margin=dict(l=80, r=80, t=45, b=60),
    )

    for axis_num in axis_groups:
        key = "yaxis" if axis_num == 1 else f"yaxis{axis_num}"
        group_signals = [s for s in signals if axis_map[s] == axis_num]
        title = " / ".join(group_signals[:2]) + (" ..." if len(group_signals) > 2 else "")
        config = dict(
            title="z-score" if normalize else title,
            side=side_cycle.get(axis_num, "left"),
            showgrid=axis_num == 1,
            zeroline=False,
        )
        if axis_num != 1:
            config.update(overlaying="y", anchor="free", position=positions.get(axis_num, 1.0))
        layout[key] = config

    fig.update_layout(**layout)
    return fig


def build_stacked_trend(df: pd.DataFrame, signals: list[str], normalize: bool = False) -> go.Figure:
    fig = make_subplots(rows=len(signals), cols=1, shared_xaxes=True, vertical_spacing=0.025)
    for row, signal in enumerate(signals, start=1):
        fig.add_trace(
            go.Scattergl(
                x=df["Timestamp"],
                y=_plot_series(df[signal], normalize),
                mode="lines",
                name=signal,
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text="z-score" if normalize else signal, row=row, col=1)
    fig.update_layout(
        height=max(450, 220 * len(signals)),
        hovermode="x unified",
        dragmode="select",
        legend=dict(orientation="h"),
    )
    fig.update_xaxes(title_text="Time", row=len(signals), col=1)
    return fig


def build_boxplot(df: pd.DataFrame, signals: list[str], normalize: bool = False) -> go.Figure:
    fig = go.Figure()
    for signal in signals:
        values = _plot_series(df[signal], normalize).dropna()
        fig.add_trace(
            go.Box(
                y=values,
                name=signal,
                boxmean=True,
                boxpoints="outliers",
            )
        )
    fig.update_layout(
        height=480,
        yaxis_title="z-score" if normalize else "Engineering value",
        xaxis_title="Signal",
        showlegend=False,
    )
    return fig


def build_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    z = corr.to_numpy(dtype=float)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            zmin=-1,
            zmax=1,
            colorbar=dict(title="r"),
            text=np.round(z, 3),
            texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}<br>r=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(height=max(380, 70 + 35 * len(corr.columns)))
    return fig


def build_dynamics_figure(df: pd.DataFrame, signal: str, rate, rolling: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07)
    fig.add_trace(go.Scattergl(x=df["Timestamp"], y=df[signal], name=signal), row=1, col=1)
    fig.add_trace(go.Scattergl(x=df["Timestamp"], y=rolling["Rolling mean"], name="Rolling mean"), row=1, col=1)
    fig.add_trace(go.Scattergl(x=df["Timestamp"], y=rate, name="Rate of change /s"), row=2, col=1)
    fig.update_yaxes(title_text=signal, row=1, col=1)
    fig.update_yaxes(title_text="Rate / s", row=2, col=1)
    fig.update_layout(height=650, hovermode="x unified", legend=dict(orientation="h"))
    return fig
