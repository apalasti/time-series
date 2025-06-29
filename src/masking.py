import numpy as np
import shap
from sklearn.metrics.pairwise import cosine_similarity

from src.linear_models import LinearClassifier


def shapley_explainer(model: LinearClassifier, samples: np.ndarray, seed=None):
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
    shap_values = explainer(sample_ixs)

    avg_shap_magnitudes: np.ndarray = np.mean(np.abs(shap_values.values), axis=0)
    order = np.argsort(
        (avg_shap_magnitudes / avg_shap_magnitudes.sum(axis=0)).max(axis=1)
    )
    return order, shap_values


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
