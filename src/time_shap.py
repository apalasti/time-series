import numpy as np
import shap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .datasets import ClassificationDataset


def calculate_shapley_values(model: nn.Module, dataset: ClassificationDataset, layer_name: str, seed=None):
    layer_to_patch = next(
        (module for name, module in model.named_modules() if name == layer_name), 
        None
    )
    if layer_to_patch is None:
        raise ValueError(f"Could not find layer with name in model: '{layer_name}'")

    device = model.device
    orig_forward = layer_to_patch.forward
    N, _, _ = dataset.samples.shape

    model.eval()

    n_channels, n_timesteps, n_dims = 0, 0, 0
    with torch.no_grad(), torch.inference_mode():
        try:
            def raising_forward(x: Tensor, *args, **kwargs):
                raise Exception(*x.shape)
            layer_to_patch.forward = raising_forward
            model(torch.unsqueeze(dataset[0][0], dim=0).to(device=device))
        except Exception as e:
            if len(e.args) == 3 and all(isinstance(arg, int) for arg in e.args):
                n_channels, n_timesteps, n_dims = e.args
            else:
                # If exception was something else, restore and re-raise
                layer_to_patch.forward = orig_forward
                raise e
        finally:
            layer_to_patch.forward = orig_forward

    blank = torch.zeros(n_timesteps, n_dims).to(device=device)
    def f(timestep_ixs: np.ndarray):
        sample_ix = timestep_ixs.max()
        x, _ = dataset[sample_ix]

        cond_tensor = torch.from_numpy(timestep_ixs != -1).to(device=device).unsqueeze(1).unsqueeze(-1)
        def patched_forward(x: Tensor, *args, **kwargs):
            x = torch.where(cond_tensor, x, blank).reshape(-1, n_timesteps, n_dims)
            return orig_forward(x, *args, **kwargs)

        layer_to_patch.forward = patched_forward
        with torch.no_grad(), torch.inference_mode():
            out = model(torch.unsqueeze(x, dim=0).to(device=device))
            return F.softmax(out, dim=-1).cpu().numpy()

    def masker(mask: np.ndarray, sample_ix: np.ndarray):
        return np.where(mask == 0, sample_ix, -1)[np.newaxis, ...]

    sample_ixs = np.column_stack([np.arange(N)] * n_timesteps)
    explainer = shap.Explainer(f, masker, seed=seed)
    try:
        shap_values = explainer(sample_ixs, batch_size=n_timesteps+1)
    finally:
        layer_to_patch.forward = orig_forward

    return shap_values


if __name__ == "__main__":
    from .tsl.model import TSLModel
    from scripts.utils import load_datasets
    train_ds, _, _ = load_datasets("Epilepsy", {})
    model = TSLModel.load_from_checkpoint("models/PatchTST_Epilepsy.ckpt")

    train_ds.samples = train_ds.samples[:120]
    shap_values = calculate_shapley_values(model, train_ds, "model.encoder.attn_layers.0")
