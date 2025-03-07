import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_classification_dataset(series: np.ndarray, labels: np.ndarray):
    assert series.shape[0] == labels.shape[0]
    num_samples, ts_len, num_vars = series.shape
    series = np.reshape(series, (num_samples * ts_len, num_vars))

    fig = make_subplots(
        rows=num_vars,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        figure=go.Figure(
            layout=dict(
                height=num_vars * 100,
                showlegend=True,
                margin=dict(l=15, r=15, t=30, b=30),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )
        ),
    )

    for i in range(num_vars):
        name = f"Var {i}"
        fig.add_trace(
            go.Scatter(
                x=np.arange(series.shape[0]),
                y=series[:, i],
                mode="lines",
                name=name,
                showlegend=False,
            ),
            row=i + 1,
            col=1,
        )
        fig.update_yaxes(title_text=name, row=i + 1, col=1)
        if i == num_vars - 1:
            fig.update_xaxes(title_text="Time", row=i + 1, col=1)

    colors = px.colors.qualitative.Pastel
    legend_entries = set()
    for j, label in enumerate(labels):
        name = f"Class {int(label)}"
        start, end = j * ts_len, (j + 1) * ts_len
        color = colors[int(label) % len(colors)]

        if name not in legend_entries:
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(color=color),
                    name=name,
                    showlegend=True,
                )
            )
            legend_entries.add(name)

        fig.add_vrect(
            x0=start,
            x1=end,
            name=name,
            fillcolor=color,
            opacity=0.4,
            layer="below",
            line_width=0,
            showlegend=False,
        )
        if 0 < j:
            fig.add_vline(x=start, line_dash="dash", line_color="gray")
    return fig


def plot_ts(df: pd.DataFrame):
    fig = make_subplots(
        rows=len(df.columns),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02
    )

    for i, col in enumerate(df.columns):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, showlegend=False), row=i+1, col=1)
        fig.update_yaxes(title_text=col, row=i+1, col=1)

    return fig.update_layout(
        height=150*len(df.columns),
        showlegend=False,
        margin=dict(l=15, r=15, t=30, b=30)
    )


def imbalance_ratio(labels: np.ndarray) -> float:
    _, class_counts = np.unique(labels, return_counts=True)
    return float(class_counts.max() / class_counts.min())
