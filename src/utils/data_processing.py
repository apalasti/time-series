import numpy as np
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


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


def imbalance_ratio(labels: np.ndarray) -> float:
    _, class_counts = np.unique(labels, return_counts=True)
    return float(class_counts.max() / class_counts.min())