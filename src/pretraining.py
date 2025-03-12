from typing import List, Tuple

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import Tensor

from src.time_drl import TimeDRL
from src.utils import create_patches, visualize_embeddings_2d


class PretrainedTimeDRL(L.LightningModule):
    def __init__(self, **config):
        super().__init__()

        self.save_hyperparameters()

        self.create_patches = lambda x: create_patches(
            x,
            config["patch_len"],
            config["patch_stride"],
            enable_channel_independence=False,
        )

        _, patched_seq_len, patched_channels = self.create_patches(
            torch.randn((1, config["sequence_len"], config["input_channels"]))
        ).shape

        self.model = TimeDRL(
            sequence_len=patched_seq_len,
            input_channels=patched_channels,
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            token_embedding_kernel_size=config["token_embedding_kernel_size"],
            dropout=config["dropout"],
        )

    def _get_representations(self, x_patched: Tensor) -> Tuple[Tensor, Tensor]:
        representations = self.model(x_patched)
        return representations[:, 0], representations[:, 1:]

    def training_step(self, batch, batch_idx):
        x, _ = batch

        B, T, C = x.shape
        assert (
            B == self.hparams.batch_size
            and T == self.hparams.sequence_len
            and C == self.hparams.input_channels
        )

        x_patched = self.create_patches(x)

        cls_a, timestamps_a = self._get_representations(x_patched)
        cls_b, timestamps_b = self._get_representations(x_patched)

        reconstruction_loss = (
            F.mse_loss(self.model.reconstructor(timestamps_a), x_patched)
            + F.mse_loss(self.model.reconstructor(timestamps_b), x_patched)
        ) / 2.0

        # NOTE: Seems as if the authors try to reconstruct the other instance
        # embedding by bringing them closer with cosine similarity
        contrastive_loss = (
            F.cosine_similarity(
                cls_a.detach(), self.model.contrastive_predictor(cls_b)
            ).mean()
            + F.cosine_similarity(
                cls_b.detach(), self.model.contrastive_predictor(cls_a)
            ).mean()
        ) / 2.0

        loss = (
            self.hparams.get("reconstruction_weight", 1.0) * reconstruction_loss
            + self.hparams.get("contrastive_weight", 1.0) * contrastive_loss
        )

        self.log_dict({"lr": self.optimizers().param_groups[0]["lr"]}, on_epoch=True)
        self.log_dict(
            {
                "train_loss": loss,
                "reconstruction_loss": reconstruction_loss,
                "contrastive_loss": contrastive_loss,
            },
            prog_bar=True,
            on_step=True,
        )

        return loss

    def on_validation_start(self):
        self.cls_embeddings: List[np.ndarray] = []
        self.cls_labels: List[np.ndarray] = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_patched = self.create_patches(x)
        cls, timestamps = self._get_representations(x_patched)
        reconstruction_loss = F.mse_loss(
            self.model.reconstructor(timestamps), x_patched
        )

        self.log("val_reconstruction_loss", reconstruction_loss, on_epoch=True)
        self.cls_embeddings.append(cls.detach().cpu().numpy())
        if y.ndim == 1:  # This means a classification dataset
            self.cls_labels.append(y.detach().cpu().numpy())

    def on_validation_end(self):
        cls_embeddings = np.concatenate(self.cls_embeddings, axis=0)
        fig = visualize_embeddings_2d(
            cls_embeddings,
            np.concatenate(self.cls_labels, axis=0) if self.cls_labels else None,
        )
        wandb.log({"[CLS] Embeddings": fig}, step=self.global_step)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        lr_scheduler = None
        if self.hparams.get("lr_scheduler", None) == "lambda":
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda epoch: (
                    self.hparams.learning_rate
                    if epoch < 3
                    else self.hparams.learning_rate * (0.9 ** ((epoch - 3) // 1))
                ),
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": (
                {
                    "scheduler": lr_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                }
                if lr_scheduler is not None
                else None
            ),
        }
