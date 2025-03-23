from typing import Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch.nn.functional as F
from einops import rearrange
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


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


def create_patches(
    x: Tensor, 
    patch_len: int, 
    stride: int, 
    enable_channel_independence: bool = True
) -> Tensor:
    """Create patches from input tensor.
    
    Args:
        x: Input tensor of shape (B, T, C)
        patch_len: Number of consecutive time steps to combine into a single patch
        stride: Step size between consecutive patches
        enable_channel_independence: If True, treats each channel independently by creating
            separate patches for each channel. If False, combines channels within each patch
            
    Returns:
        Patched tensor with shape:
        - (B * C, num_patches, patch_len) if channel independence enabled
        - (B, num_patches, C * patch_len) otherwise
    """
    assert stride <= patch_len, (
        f"Patch length ({patch_len}) must be greater than or equal to stride ({stride}). "
        "A patch length smaller than stride would result in skipped time steps and "
        "incomplete coverage of the input sequence."
    )

    # Rearrange and pad input
    x = rearrange(x, "B T C -> B C T")
    # NOTE: The padding appends stride number of values to the sequence, I think
    # this is wrong as there's no reason why if stride < patch_len the first
    # values should be fewer times in the patches as the last ones.
    x = F.pad(x, (0, stride), "replicate")

    # Create patches using unfold
    # Number of patches: (T + stride - patch_len) // stride + 1
    x = x.unfold(dimension=-1, size=patch_len, step=stride)

    # Rearrange based on channel independence setting
    if enable_channel_independence:
        return rearrange(x, "B C T_p P -> (B C) T_p P")
    return rearrange(x, "B C T_p P -> B T_p (C P)")


def load_lr_scheduler(optimizer: Optimizer, scheduler_type: str):
    if scheduler_type == "lambda": # type 3
        lr_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (
                1.0 if (epoch + 1) < 3 else (0.9 ** (((epoch + 1) - 3) // 1))
            ),
        )
    elif scheduler_type == "constant":
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    elif scheduler_type == "exponential_decay": # type 1
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** ((epoch - 1) // 1))
    elif scheduler_type == "warmup":
        lr_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (epoch + 1) / 5 if epoch < 5 else (0.9 ** ((epoch - 5) // 1))
        )
    else:
        raise ValueError(
            f"Unknown lr_scheduler: {scheduler_type}. "
            "Supported values are 'exponential_decay', 'warmup', 'lambda' and 'constant'"
        )
    return lr_scheduler


def visualize_embeddings_2d(
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
    similarity_matrix = np.zeros((num_classes, num_classes))

    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    for i in range(len(unique_labels)):
        embeddings_i = normalized_embeddings[labels == unique_labels[i]]
        for j in range(i, len(unique_labels)):
            embeddings_j = normalized_embeddings[labels == unique_labels[j]]
            similarities = np.dot(embeddings_i, embeddings_j.T)
            similarity_matrix[i, j] = similarity_matrix[j, i] = np.mean(similarities)

    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=[f"Class {l}" for l in unique_labels],
        y=[f"Class {l}" for l in unique_labels],
        colorscale="thermal", zmin=-1.0, zmax=1.0,
        text=similarity_matrix,
        texttemplate="%{text:.2f}",
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
