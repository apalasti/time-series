from typing import Literal, List, Optional, Dict, Union

import numpy as np
import shap
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from src.linear_models import LinearClassifier


def evaluate_masking_strategies(
    classifier: LinearClassifier,
    test_timesteps: np.ndarray,
    test_labels: np.ndarray,
    shap_values: List[np.ndarray] = None,
    metric = "accuracy"
):
    assert test_labels.ndim == 1 and test_labels.shape[0] == test_timesteps.shape[0]

    masked_percents = np.linspace(0, 0.9, 10)
    result_dict = {}

    if shap_values:
        result_dict["shap"] = np.array([
            masking_impact(
                classifier, test_timesteps, test_labels, 
                shapley_order(shap_values)[test_labels],
                masked_percents, metric=metric
            )
            for shap_values in shap_values
        ])

    result_dict["diversity"] = np.array([
        masking_impact(
            classifier, test_timesteps, test_labels,
            diversity_order(test_timesteps),
            masked_percents, metric=metric
        ),
    ])

    result_dict["random"] = np.array([
        masking_impact(
            classifier, test_timesteps, test_labels,
            # TODO: Try a different approach of random order
            random_order(test_timesteps),
            masked_percents, metric=metric
        )
        for _ in range(15)
    ])
    return result_dict


def evaluate_masking_strategies_v2(
    classifier: LinearClassifier,
    test_timesteps: np.ndarray,
    test_labels: np.ndarray,
    strategies: Dict[str, Union[np.ndarray, List[np.ndarray]]],
    metric = "accuracy"
):
    assert test_labels.ndim == 1 and test_labels.shape[0] == test_timesteps.shape[0]

    masked_percents = np.linspace(0, 0.9, 10)
    result_dict = {}
    for strategy, orders in strategies.items():
        orders = orders if isinstance(orders, list) else [orders]
        result_dict[strategy] = np.array([
            masking_impact(
                classifier, test_timesteps, test_labels, order,
                masked_percents, metric=metric,
            )
            for order in orders
        ])
    return result_dict


def masking_impact(
    classifier: LinearClassifier,
    timesteps: np.ndarray,
    labels: np.ndarray,
    order: np.ndarray,
    masked_percentages=None,
    metric: Literal["accuracy", "f1", "auc_ovo", "auc_ovr"] = "accuracy",
    mask=None,
):
    assert metric in {"accuracy", "f1", "auc_ovo", "auc_ovr"}
    N, T, _ = timesteps.shape

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
            masked_timesteps = mask_timesteps(timesteps, masked_ixs, mask)
            preds = classifier.predict(masked_timesteps)
            metrics.append(accuracy_score(labels, preds))
        elif metric == "f1":
            masked_timesteps = mask_timesteps(timesteps, masked_ixs, mask)
            preds = classifier.predict(masked_timesteps)
            metrics.append(f1_score(labels, preds, average="macro"))
        elif metric == "auc_ovr":
            masked_timesteps = mask_timesteps(timesteps, masked_ixs, mask)
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
            masked_timesteps = mask_timesteps(timesteps, masked_ixs, mask)
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
    shap_values = np.transpose(shap_values.values, (0, 2, 1))
    importance = np.abs(shap_values).mean(axis=0)
    return np.argsort(importance, axis=-1)


def random_order(timesteps: np.ndarray, all_unique=True):
    N, T, _ = timesteps.shape
    if all_unique:
        # Different order for each sample
        return np.array(
            [np.random.choice(np.arange(T), size=T, replace=False) for _ in range(N)]
        )

    order = np.random.choice(np.arange(T), size=T, replace=False)
    return np.repeat(order[np.newaxis], repeats=N, axis=0)


def mask_timesteps(
    samples: np.ndarray, masked_cols: np.ndarray, mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Mask specified timesteps in samples (shape: n_samples, n_timesteps, n_features).

    Args:
        samples: Input array.
        masked_cols: Timesteps to replace (1D: same for all, 2D: per-sample).
        mask: Replacement values (default zeros).
    """
    _, T, D = samples.shape
    masked = samples.copy()
    if masked_cols.size == 0:
        return masked

    if mask is None:
        mask = np.zeros((T, D))

    assert mask.shape == (T, D), f"mask should be of shape ({T}, {D}), given: {mask.shape}"
    assert masked_cols.ndim in {1, 2}, f"masked_cols must be a 1D or 2D numpy array, shape given: {masked_cols.shape}"

    if masked_cols.ndim == 1:
        masked[:, masked_cols] = mask[np.newaxis, masked_cols]
    elif masked_cols.ndim == 2:
        assert samples.shape[0] == masked_cols.shape[0]
        row = np.arange(samples.shape[0])[:, np.newaxis]
        masked[row, masked_cols] = mask[masked_cols]
    return masked


def diversity_order(timesteps: np.ndarray):
    return np.argsort(-diversity(timesteps), axis=-1)


def diversity(timesteps: np.ndarray):
    (N, T, _) = timesteps.shape
    # For each sample, compute the mean cosine similarity of each timestep to all others (excluding self)
    mean_cos_sim = np.empty((N, T))
    for i in range(N):
        cos_sim = cosine_similarity(timesteps[i])
        # Exclude diagonal (self-similarity) for each timestep
        for j in range(T):
            mean_cos_sim[i, j] = (np.sum(cos_sim[j]) - cos_sim[j, j]) / (T - 1)
    return mean_cos_sim


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
