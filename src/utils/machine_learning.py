from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def load_lr_scheduler(optimizer: Optimizer, scheduler_type: str):
    """
    Creates a learning rate scheduler for the given optimizer.

    Args:
        optimizer: PyTorch optimizer instance.
        scheduler_type: One of 'constant', 'exponential_decay', 'warmup', 'lambda'.

    Returns:
        LambdaLR scheduler instance.

    Raises:
        ValueError: If scheduler_type is unsupported.
    """
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
