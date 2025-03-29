from typing import Literal

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor

from src.utils import load_lr_scheduler


class BaseModule(L.LightningModule):
    def __init__(self):
        super().__init__()

    def _log_figure(self, name, fig):
        if isinstance(self.logger, WandbLogger):
            logger: WandbLogger = self.logger
            logger.log_metrics({name: fig})
        else:
            fig.update_layout(title=name)
            fig.show()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler_type = self.hparams.get("lr_scheduler", "constant")
        lr_scheduler = load_lr_scheduler(optimizer, scheduler_type)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_start(self):
        self.log(
            "lr", self.optimizers().param_groups[0]["lr"], on_step=False, on_epoch=True
        )


class InstanceNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super(InstanceNorm, self).__init__()
        self.eps = eps

    def forward(self, x, mode: Literal["norm", "denorm"] = "norm"):
        if mode == "norm":
            self._get_statistics(x)  # Set mean and std
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x: Tensor):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.std = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.std
        return x

    def _denormalize(self, x):
        x = x * self.std
        x = x + self.mean
        return x
