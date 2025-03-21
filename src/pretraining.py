from typing import List, Tuple

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR

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

        # self.instance_norm = RevIN(num_features=0, affine=False)
        self.instance_norm = lambda x: F.instance_norm(
            torch.transpose(x, 1, 2)
        ).transpose(1, 2)
        self.model = TimeDRL(
            sequence_len=patched_seq_len,
            input_channels=patched_channels,
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            token_embedding_kernel_size=config["token_embedding_kernel_size"],
            dropout=config["dropout"],
        )

    def get_representations(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.instance_norm(x)
        x = self.create_patches(x)
        representations = self.model(x)
        return representations[:, 0], representations[:, 1:]

    def training_step(self, batch, batch_idx):
        x, _ = batch
        B, T, C = x.shape
        assert (
            B == self.hparams.batch_size and T == self.hparams.sequence_len and C == self.hparams.input_channels
        ), f"Shape mismatch. Expected ({self.hparams.batch_size}, {self.hparams.sequence_len}, {self.hparams.input_channels}), got ({B}, {T}, {C})"

        cls_a, timestamps_a = self.get_representations(x)
        cls_b, timestamps_b = self.get_representations(x)

        reconstructed_a = self.model.reconstructor(timestamps_a)
        reconstructed_b = self.model.reconstructor(timestamps_b)

        reconstruction_loss = (
            F.mse_loss(reconstructed_a, self.create_patches(x))
            + F.mse_loss(reconstructed_b, self.create_patches(x))
        ) / 2.0

        # NOTE: Seems as if the authors try to reconstruct the other instance
        # embedding by bringing them closer with cosine similarity
        contrastive_loss = -(
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

        self.log_dict(
            {"lr": self.optimizers().param_groups[0]["lr"]},
            on_step=False,
            on_epoch=True,
        )
        self.log_dict(
            {
                "train/loss": loss,
                "train/reconstruction_loss": reconstruction_loss,
                "train/contrastive_loss": contrastive_loss,
            },
            prog_bar=True,
            on_step=True,
        )

        return loss

    def on_validation_epoch_start(self):
        self.cls_embeddings: List[np.ndarray] = []
        self.cls_labels: List[np.ndarray] = []

    def validation_step(self, batch, batch_idx):
        x, y = batch
        cls, timestamps = self.get_representations(x)
        reconstruction_loss = F.mse_loss(
            self.model.reconstructor(timestamps), self.create_patches(x)
        )

        self.log("val/reconstruction_loss", reconstruction_loss, on_epoch=True)
        self.cls_embeddings.append(cls.detach().cpu().numpy())
        if y.ndim == 1:  # This means a classification dataset
            self.cls_labels.append(y.detach().cpu().numpy())

    def on_validation_epoch_end(self):
        cls_embeddings = np.concatenate(self.cls_embeddings, axis=0)
        fig = visualize_embeddings_2d(
            cls_embeddings,
            np.concatenate(self.cls_labels, axis=0) if self.cls_labels else None,
        )
        if isinstance(self.logger, WandbLogger):
            logger: WandbLogger = self.logger
            logger.log_metrics({"val/[CLS] Embeddings": fig})
        else:
            fig.show()

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        cls_embeddings, timestamps = self.get_representations(x)
        return cls_embeddings, timestamps

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        scheduler_type = self.hparams.get("lr_scheduler", "constant")
        if scheduler_type == "lambda":
            lr_scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: (
                    1.0 if (epoch + 1) < 3 else (0.9 ** (((epoch + 1) - 3) // 1))
                ),
            )
        elif scheduler_type == "constant":
            lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        else:
            raise ValueError(
                f"Unknown lr_scheduler: {scheduler_type}. "
                "Supported values are 'lambda' and 'constant'"
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }
