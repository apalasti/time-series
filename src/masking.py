from typing import Literal, Union, List

import numpy as np
import shap
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from src.linear_models import LinearClassifier


def evaluate_masking_strategies(
    classifier: LinearClassifier,
    test_timesteps: np.ndarray,
    test_labels: np.ndarray,
    shap_values: List[np.ndarray],
    metric = "accuracy"
):
    # assert train_timesteps.shape[1:] == test_timesteps.shape[1:]
    assert test_labels.ndim == 1 and test_labels.shape[0] == test_timesteps.shape[0]

    masked_percents = np.linspace(0, 0.9, 10)
    # train_preds = classifier.predict(train_timesteps)
    # test_preds = classifier.predict(test_timesteps)

    shap_result = np.array([
        masking_impact(
            classifier, test_timesteps, test_labels, 
            shapley_order(shap_values)[test_labels],
            masked_percents, metric=metric
        )
        for shap_values in shap_values
    ])

    random_result = np.array([
        masking_impact(
            classifier, test_timesteps, test_labels,
            masked_percentages=masked_percents,
            order="random", metric=metric
        )
        for _ in range(15)
    ])
    return {f"shap": shap_result, "random": random_result}


def masking_impact(
    classifier: LinearClassifier,
    timesteps: np.ndarray,
    labels: np.ndarray,
    order: Union[np.ndarray, Literal["random"]],
    masked_percentages=None,
    metric: Literal["accuracy", "f1", "auc_ovo", "auc_ovr"] = "accuracy",
):
    assert metric in ("accuracy", "f1", "auc_ovo", "auc_ovr")
    N, T, _ = timesteps.shape
    if isinstance(order, str) and order == "random":
        order = np.array(
            [np.random.choice(np.arange(T), size=T, replace=False) for _ in range(N)]
        )

    assert (order.ndim == 1 and order.shape == (T,)) or (
        order.ndim == 2 and order.shape == (N, T)
    )
    if masked_percentages is None:
        masked_percentages = np.linspace(0, 0.9, 10)

    metrics = []
    for prob in masked_percentages:
        masked_count = int(T * prob)

        masked_ixs = order[..., :masked_count]

        if metric == "accuracy":
            masked_timesteps = mask_timesteps(timesteps, masked_ixs)
            preds = classifier.predict(masked_timesteps)
            metrics.append(accuracy_score(labels, preds))
        elif metric == "f1":
            masked_timesteps = mask_timesteps(timesteps, masked_ixs)
            preds = classifier.predict(masked_timesteps)
            metrics.append(f1_score(labels, preds, average="macro"))
        elif metric == "auc_ovr":
            masked_timesteps = mask_timesteps(timesteps, masked_ixs)
            probs = classifier.predict_proba(masked_timesteps)
            metrics.append(
                roc_auc_score(
                    labels,
                    probs[:, 1] if classifier.n_classes_ == 2 else probs,
                    average=None,
                    multi_class="ovr" if classifier.n_classes_ > 2 else "raise"
                )
            )
            if classifier.n_classes_ == 2:
                metrics[-1] = np.stack([metrics[-1], metrics[-1]], axis=-1)
        elif metric == "auc_ovo":
            masked_timesteps = mask_timesteps(timesteps, masked_ixs)
            probs = classifier.predict_proba(masked_timesteps)
            metrics.append(
                roc_auc_score(labels, probs, average="macro", multi_class="ovo")
                if classifier.n_classes_ > 2 else
                roc_auc_score(labels, probs[:, 1])
            )

    return np.array(metrics)


def calculate_shapley_values(model: LinearClassifier, samples: np.ndarray, seed=None):
    N, T, D = samples.shape
    sequences = np.concatenate((samples, np.zeros((1, T, D))))
    sample_ixs = np.column_stack([np.arange(N)] * T)

    def f(sample_ix: np.ndarray):
        input = sequences[sample_ix, np.arange(T)[np.newaxis, :]]
        pred = model.predict_proba(input)
        return pred

    def masker(mask: np.ndarray, sample_ix: np.ndarray):
        masked = sample_ix.copy()
        masked[~mask] = -1
        return masked[np.newaxis, :]

    explainer = shap.Explainer(f, masker, seed=seed)
    shap_values = explainer(sample_ixs, batch_size=100)
    return shap_values


def shapley_order(shap_values):
    # assert (
        # labels.ndim == 1
        # and shap_values.shape[0] == labels.shape[0]
    # ), (
        # f"`labels` must be 1D (received shape: {labels.shape}), "
        # f"and its first dimension must match the first dimension of `shap_values` "
        # f"(got {shap_values.shape[0]} and {labels.shape[0]})."
    # )
    shap_values = np.transpose(shap_values.values, (0, 2, 1))
    importance = np.abs(shap_values).mean(axis=0)
    # importance = np.array([
    # np.abs(shap_values.values[pred == preds, :, pred]).mean(axis=0)
    # for pred in np.unique(preds)
    # ])
    # importance = np.array([
        # np.max(np.abs(shap_values[label == labels]).mean(axis=0), axis=0)
        # for label in np.unique(labels)
    # ])
    return np.argsort(importance, axis=-1)
    # return np.argsort(np.abs(shap_values).mean(axis=0).T)


def mask_timesteps(samples: np.ndarray, masked_cols: np.ndarray) -> np.ndarray:
    """Mask specified timesteps in time series samples by setting them to zero.
    
    Args:
        samples: Input time series data of shape (n_samples, n_timesteps, n_features)
        masked_cols: Indices of timesteps to mask. Can be:
            - 1D array: Same timesteps masked for all samples
            - 2D array: Per-sample masking with shape (n_samples, n_masked_timesteps)
            
    Returns:
        np.ndarray: Masked time series data with same shape as input
        
    Raises:
        ValueError: If masked_cols is not 1D or 2D, or if 2D shape doesn't match samples
    """
    masked = samples.copy()
    if masked_cols.size == 0:
        return masked

    if masked_cols.ndim == 1:
        masked[:, masked_cols] = 0.0
    elif masked_cols.ndim == 2:
        if masked_cols.shape[0] != samples.shape[0]:
            raise ValueError(
                f"If masked_cols is 2D, its first dimension "
                f"({masked_cols.shape[0]}) must match the number of samples "
                f"({samples.shape[0]}) for per-sample masking."
            )
        masked[np.arange(samples.shape[0])[:, np.newaxis], masked_cols] = 0.0
    else:
        raise ValueError("masked_cols must be a 1D or 2D numpy array.")
    return masked


def diversity_of_unmasked(timesteps: np.ndarray, masks: np.ndarray):
    (N, T, _) = timesteps.shape
    avg_cos_sim = np.ones((N,))
    for i in range(N):
        unmasked_indices = np.setdiff1d(np.arange(T), masks[i if masks.ndim == 2 else ()])
        if len(unmasked_indices) == 1: 
            continue
           
        cos_sim = cosine_similarity(timesteps[i, unmasked_indices])
        avg_cos_sim[i] = cos_sim[np.tril_indices(len(unmasked_indices), k=-1)].mean()
    return avg_cos_sim
