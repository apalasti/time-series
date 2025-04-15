from typing import Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix


def visualize_attention(attention_values):
    attention_mean = np.mean(attention_values, axis=0)
    attention_std = np.std(attention_values, axis=0)
    n_layers, n_heads, n_tokens = attention_mean.shape

    fig = make_subplots(
        rows=n_layers, cols=2, subplot_titles=[f"Layer {i+1}" for i in range(n_layers)]
    )

    for i in range(n_layers):
        layer_mean = attention_mean[i, :, :]
        layer_std = attention_std[i, :, :]
        row_idx, col_idx = (i // 2) + 1, (i % 2) + 1

        text = np.char.add(
            np.char.mod('%.2f', layer_mean),
            np.char.add('<br>±', np.char.mod('%.2f', layer_std))
        )

        fig.add_trace(
            go.Heatmap(
                z=layer_mean, showscale=False,
                x=[f"Token {j+1}" for j in range(n_tokens)],
                y=[f"Head {j+1}" for j in range(n_heads)],
                colorscale="Viridis", zmin=0.0, zmax=1.0,
                text=text, texttemplate="%{text}",
            ),
            row=row_idx,
            col=col_idx,
        )

    height_per_row = n_heads * 60
    width_per_col = n_tokens * 60
    num_rows = (n_layers + 1) // 2 # Calculate number of rows needed

    fig.update_layout(
        height=num_rows * height_per_row,
        width=2 * width_per_col, # Always 2 columns
    )

    return fig


def visualize_pca_components(
    time_series_data: np.ndarray,
    timestamp_embeddings: np.ndarray,
    n_components: int = 15,
):
    pca = PCA(n_components=min(n_components, *timestamp_embeddings.shape))
    timestamp_components = pca.fit_transform(timestamp_embeddings)

    fig_timeseries = px.line(
        time_series_data,
        labels=dict(index="Time Step", value="Value", variable="Channel"),
    )

    fig_heatmap = px.imshow(
        np.transpose(timestamp_components),
        labels=dict(x="Time", y="Principal Components", color="Value"),
        y=[
            f"PC{i+1:02d} ({var:.1%})"
            for i, var in enumerate(pca.explained_variance_ratio_)
        ],
    )

    fig = make_subplots(
        subplot_titles=("Input Time Series - All Channels", "Timestamp PCA Components"),
        rows=2, cols=1,
    )

    fig.add_traces(fig_timeseries.data, rows=1, cols=1)
    fig.add_traces(fig_heatmap.data, rows=2, cols=1)
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    fig.update_layout(coloraxis_colorscale='Viridis', showlegend=False)

    return fig


def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray):
    """Generates and returns a confusion matrix plot."""
    cm = confusion_matrix(labels, preds)
    fig = px.imshow(
        cm / cm.sum(axis=1, keepdims=True), zmin=0, zmax=1,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=[f"Class {i}" for i in range(cm.shape[0])],
        y=[f"Class {i}" for i in range(cm.shape[1])],
        color_continuous_scale="Blues", text_auto=True,
    )
    return fig


def visualize_weights(weights: np.ndarray):
    num_classes, num_weights = weights.shape
    fig = px.bar(
        x=[i % num_weights for i in range(num_classes * num_weights)],
        y=weights.flatten(),
        color=[f"Class {c}" for i in range(num_classes) for c in [i] * num_weights],
        labels={
            "x": "Time Step",
            "y": "Mean Absolute Weight",
            "color": "Class",
        },
    )
    fig.update_traces(opacity=0.8, marker_line_width=1, marker_line_color="black")
    return fig


def visualize_predictions(past: np.ndarray, future: np.ndarray, preds: np.ndarray):
    """Creates a multi-channel time series plot."""
    num_channels = past.shape[-1]
    fig = make_subplots(rows=num_channels, cols=1, shared_xaxes=True)

    past_len = past.shape[0]
    future_len = future.shape[0]
    for channel_idx in range(num_channels):
        fig.add_scatter(
            x=np.arange(past_len),
            y=past[:, channel_idx],
            mode="lines",
            line_color="blue",
            name=f"Past (Channel {channel_idx})",
            row=channel_idx + 1,
            col=1,
            showlegend=False,
        )
        fig.add_scatter(
            x=np.arange(past_len, past_len + future_len),
            y=future[:, channel_idx],
            mode="lines",
            line_color="blue",
            name=f"Future (Channel {channel_idx})",
            row=channel_idx + 1,
            col=1,
            showlegend=False,
        )
        fig.add_scatter(
            x=np.arange(past_len, past_len + future_len),
            y=preds[:, channel_idx],
            mode="lines",
            line_color="red",
            line_dash="dash",
            name=f"Predictions (Channel {channel_idx})",
            row=channel_idx + 1,
            col=1,
            showlegend=False,
        )

    return fig


def visualize_cls_embeddings_2d(
    embeddings: np.ndarray, labels: Union[np.ndarray, None] = None
):
    if labels is None:
        labels = np.zeros(len(embeddings))
    assert embeddings.shape[0] == labels.shape[0]

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    fig = px.scatter(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        color=["Class " + str(label) for label in labels.astype(str)],
        labels={
            "x": f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
            "y": f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
        },
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def visualize_patch_reconstruction(original: np.ndarray, reconstructed: np.ndarray):
    """Compare original patched time series with reconstructed version."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Original Patches", "Reconstructed Patches"),
        shared_xaxes=True, vertical_spacing=0.1,
    )

    zmin = min(original.min(), reconstructed.min())
    zmax = max(original.max(), reconstructed.max())
    fig.add_trace(go.Heatmap(
        z=original.T, zmin=zmin, zmax=zmax,
        colorscale="Viridis",
        text=original.T,
        texttemplate="%{text:.2f}",
        textfont={"size": 8},
        colorbar=dict(len=0.40, y=0.77, yanchor="middle")
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=reconstructed.T, zmin=zmin, zmax=zmax,
        colorscale="Viridis",
        text=reconstructed.T,
        texttemplate="%{text:.2f}",
        textfont={"size": 8},
        colorbar=dict(len=0.40, y=0.22, yanchor="middle"),
    ), row=2, col=1)

    return fig


def cosine_similarity_matrix(embeddings: np.ndarray, labels: np.ndarray):
    """
    Calculates and visualizes the cosine similarity matrix between class embeddings.
    """
    assert embeddings.shape[0] == labels.shape[0]
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    similarities_mean = np.zeros((num_classes, num_classes))
    similarities_std = np.zeros((num_classes, num_classes))

    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    for i in range(len(unique_labels)):
        embeddings_i = normalized_embeddings[labels == unique_labels[i]]
        for j in range(i, len(unique_labels)):
            embeddings_j = normalized_embeddings[labels == unique_labels[j]]
            similarities = np.dot(embeddings_i, embeddings_j.T)
            similarities_mean[i, j] = similarities_mean[j, i] = np.mean(similarities)
            similarities_std[i, j] = similarities_std[j, i] = np.std(similarities)

    text = [
        [f"Mean: {m:.2f}<br>Std: {s:.2f}" for m, s in zip(row_m, row_s)]
        for row_m, row_s in zip(similarities_mean, similarities_std)
    ]
    fig = go.Figure(data=go.Heatmap(
        z=similarities_mean,
        x=[f"Class {l}" for l in unique_labels],
        y=[f"Class {l}" for l in unique_labels],
        colorscale="thermal", zmin=-1.0, zmax=1.0,
        text=text, texttemplate="%{text}",
        textfont={"size": 10},
    ))
    fig.update_layout(
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_classification_dataset(series: np.ndarray, labels: np.ndarray):
    assert series.shape[0] == labels.shape[0]
    num_samples, ts_len, num_vars = series.shape
    series = np.reshape(series, (num_samples * ts_len, num_vars))

    fig = make_subplots(
        rows=num_vars, cols=1,
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
                x=np.arange(series.shape[0]), y=series[:, i],
                mode="lines", name=name, showlegend=False,
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
                    x=[None], y=[None], mode="markers",
                    marker=dict(color=color), name=name,
                    showlegend=True,
                )
            )
            legend_entries.add(name)

        fig.add_vrect(
            x0=start, x1=end, name=name, fillcolor=color,
            opacity=0.4, layer="below", line_width=0,
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
