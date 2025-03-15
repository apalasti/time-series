from typing import List, Tuple

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor

from src.time_drl import TimeDRL
from src.instance_norm import RevIN
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

        self.instance_norm = RevIN(num_features=0, affine=False)
        self.model = TimeDRL(
            sequence_len=patched_seq_len,
            input_channels=patched_channels,
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            token_embedding_kernel_size=config["token_embedding_kernel_size"],
            dropout=config["dropout"],
        )

        # torch.manual_seed(2023)
        # for name, param in sorted(self.model.named_parameters()):
            # torch.nn.init.normal_(param)

    def _get_representations(self, x_patched: Tensor) -> Tuple[Tensor, Tensor]:
        # x_normed = self.instance_norm(x, "norm")
        # x_normed_patches = self.create_patches(x_normed)
        representations = self.model(x_patched)
        return representations[:, 0], representations[:, 1:]

    def training_step(self, batch, batch_idx):
        # for name, param in sorted(self.model.named_parameters()):
            # print(f"{name}: {param.data[:2]}")
            # if param.grad is not None:
                # print(f"{name} grad: {param.grad.data[:2]}")
            # else:
                # print(f"{name} grad: None")

        x, _ = batch

        B, T, C = x.shape
        assert (
            B == self.hparams.batch_size
            and T == self.hparams.sequence_len
            and C == self.hparams.input_channels
        )

        x_normed = self.instance_norm(x, "norm")
        x_normed_patches = self.create_patches(x_normed)

        # self.model.eval()
        cls_a, timestamps_a = self._get_representations(x_normed_patches)
        cls_b, timestamps_b = self._get_representations(x_normed_patches)

        reconstructed_a = self.model.reconstructor(timestamps_a)
        reconstructed_b = self.model.reconstructor(timestamps_b)

        if batch_idx == 0:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Convert tensors to numpy for visualization
            x_patched_np = self.create_patches(x)[0].detach().cpu().numpy()
            cls_a_np = cls_a[0].detach().cpu().numpy()
            timestamps_a_np = reconstructed_a[0].detach().cpu().numpy()

            # Create subplots for each component
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=("Patched Input", "CLS Embedding", "Timestamp Embeddings"),
                shared_yaxes=True
            )

            # Plot patched input
            fig.add_trace(go.Heatmap(
                z=x_patched_np.T,
                colorscale="Viridis",
                colorbar=dict(title="Value")
            ), row=1, col=1)

            # Plot CLS embedding
            fig.add_trace(go.Heatmap(
                z=cls_a_np[None, :],  # Add dimension for heatmap
                colorscale="Viridis",
                colorbar=dict(title="Value")
            ), row=2, col=1)

            # Plot timestamp embeddings
            fig.add_trace(go.Heatmap(
                z=timestamps_a_np.T,
                colorscale="Viridis",
                colorbar=dict(title="Value")
            ), row=3, col=1)

            fig.update_layout(
                title_text="Model Embeddings Visualization",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            fig.show()

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
        x_norm = self.instance_norm(x, "norm")
        x_patched = self.create_patches(x_norm)
        cls, timestamps = self._get_representations(x_patched)
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
        x_norm = self.instance_norm(x, "norm")
        x_patched = self.create_patches(x_norm)
        cls_embeddings, timestamps = self._get_representations(x_patched)
        return cls_embeddings, timestamps

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        return { "optimizer": optimizer, }
