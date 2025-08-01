from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.base import BaseModule, InstanceNorm
from src.time_drl import TimeDRL, AttentionInspector
from src.linear_models import LinearClassifier
from src.utils import (cosine_similarity_matrix, create_patches,
                       plot_confusion_matrix, visualize_cls_embeddings_2d,
                       visualize_patch_reconstruction, visualize_attention,
                       visualize_pca_components)

TOP_N_COMPONENTS = 15
KNN_N_NEIGHBORS = 5


class PretrainedTimeDRL(BaseModule):
    def __init__(self, **config):
        super().__init__()
        self.save_hyperparameters()

        self.create_patches = lambda x: create_patches(
            x,
            config["patch_len"],
            config["patch_stride"],
            config["enable_channel_independence"],
        )

        _, patched_seq_len, patched_channels = self.create_patches(
            torch.randn((1, config["sequence_len"], config["input_channels"]))
        ).shape

        self.instance_norm = InstanceNorm()
        self.model = TimeDRL(
            sequence_len=patched_seq_len,
            input_channels=patched_channels,
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            n_layers=config["n_layers"],
            token_embedding_kernel_size=config["token_embedding_kernel_size"],
            dropout=config["dropout"],
            pos_embed_type=config["pos_embed_type"],
        )

        # self.cls_reconstructor = nn.Sequential(
        # nn.Flatten(start_dim=1),
        # nn.Dropout(float(config.get("cls_reconstructor_dropout", 0.0))),
        # nn.Linear(patched_seq_len * config["d_model"], config["d_model"]),
        # )

    def get_representations(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.instance_norm(x)
        x = self.create_patches(x)
        # x = F.dropout1d(x, p=0.2, training=self.training)
        representations = self.model(x)
        return representations[:, 0], representations[:, 1:]

    def training_step(self, batch, batch_idx):
        x, _ = batch
        B, T, C = x.shape
        assert (
            T == self.hparams.sequence_len and C == self.hparams.input_channels
        ), f"Shape mismatch. Expected (N, {self.hparams.sequence_len}, {self.hparams.input_channels}), got (N, {T}, {C})"

        cls_a, timestamps_a = self.get_representations(x)
        cls_b, timestamps_b = self.get_representations(x)

        reconstructed_a = self.model.reconstructor(
            # torch.cat([cls_a.unsqueeze(1).expand(-1, timestamps_a.shape[1], -1), timestamps_a], dim=-1)
            timestamps_a
        )
        reconstructed_b = self.model.reconstructor(
            # torch.cat([cls_b.unsqueeze(1).expand(-1, timestamps_b.shape[1], -1), timestamps_b], dim=-1)
            timestamps_b
        )

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

        # reconstructed_cls_a = self.cls_reconstructor(timestamps_a.detach())
        # reconstructed_cls_b = self.cls_reconstructor(timestamps_b.detach())
        # cls_reconstruction_loss = (
        # F.mse_loss(reconstructed_cls_a, cls_a.detach())
        # + F.mse_loss(reconstructed_cls_b, cls_b.detach())
        # ) / 2.0

        loss = (
            self.hparams.get("reconstruction_weight", 1.0) * reconstruction_loss
            + self.hparams.get("contrastive_weight", 1.0) * contrastive_loss
        )

        self.log_dict({
            "train/loss": loss,
            "train/reconstruction_loss": reconstruction_loss,
            "train/contrastive_loss": contrastive_loss,
            # "train/cls_reconstruction_loss": cls_reconstruction_loss,
        }, prog_bar=True, on_step=True)

        return loss # + cls_reconstruction_loss

    # def on_after_backward(self):
    # for param in self.cls_reconstructor.parameters():
    # if param.grad is not None:
    # param.grad = -param.grad

    def on_validation_epoch_start(self):
        self.cls_embeddings: List[np.ndarray] = []
        self.cls_labels: List[np.ndarray] = []
        self.inspector = AttentionInspector(self.model)

    def validation_step(self, batch, batch_idx):
        x, y = batch

        cls, timestamps = self.get_representations(x)
        reconstructions = self.model.reconstructor(
            # torch.cat([cls.unsqueeze(1).expand(-1, timestamps.shape[1], -1), timestamps], dim=-1)
            timestamps
        )
        reconstruction_loss = F.mse_loss(reconstructions, self.create_patches(x))

        # reconstructed_cls = self.cls_reconstructor(timestamps)
        # cls_reconstruction_loss = F.cosine_similarity(reconstructed_cls, cls).mean()

        if batch_idx == 0:
            fig = visualize_pca_components(
                x[0].detach().cpu().numpy(),
                timestamps[0].detach().cpu().numpy(),
                n_components=TOP_N_COMPONENTS,
            )
            self._log_figure("val/Timestep PCA Components", fig)

            reconstructions_np = reconstructions.detach().cpu().numpy()
            patches_np = self.create_patches(x).detach().cpu().numpy()
            fig = visualize_patch_reconstruction(patches_np[0], reconstructions_np[0])
            self._log_figure("val/Reconstruction Comparison", fig)

        self.log_dict({
            "val/reconstruction_loss": reconstruction_loss,
            # "val/cls_reconstruction_loss": cls_reconstruction_loss,
        }, on_epoch=True)
        self.cls_embeddings.append(cls.detach().cpu().numpy())
        if y.ndim == 1:  # This means a classification dataset
            self.cls_labels.append(y.detach().cpu().numpy())

    def on_validation_epoch_end(self):
        attention_values = self.inspector.get_attention_values()
        cls_attention_values = attention_values[:, :, :, 0, :] # (N, Layer, Head, From, To)
        fig = visualize_attention(cls_attention_values.cpu().numpy())
        self._log_figure("val/[CLS] Attention Values", fig)
        self.inspector.unsubscribe()

        cls_embeddings = np.concatenate(self.cls_embeddings, axis=0)

        fig = visualize_cls_embeddings_2d(
            cls_embeddings,
            np.concatenate(self.cls_labels, axis=0) if self.cls_labels else None,
            method="pca",
        )
        self._log_figure("val/[CLS] Embeddings (PCA)", fig)

        fig = visualize_cls_embeddings_2d(
            cls_embeddings,
            np.concatenate(self.cls_labels, axis=0) if self.cls_labels else None,
            method="tsne",
        )
        self._log_figure("val/[CLS] Embeddings (TSNE)", fig)

        if self.cls_labels:
            labels = np.concatenate(self.cls_labels, axis=0)
            fig_confusion = cosine_similarity_matrix(cls_embeddings, labels)
            self._log_figure("val/Cosine Similarity Confusion Matrix", fig_confusion)

        if self.cls_labels and self.trainer.train_dataloader:
            # Perform kNN classification
            labels = np.concatenate(self.cls_labels, axis=0)
            train_cls_embeddings, _, train_labels = (
                self.get_representations_from_dataloader(self.trainer.train_dataloader)
            )
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS, n_jobs=-1)),
            ])
            pipeline.fit(train_cls_embeddings, train_labels)
            preds = pipeline.predict(cls_embeddings)

            fig = plot_confusion_matrix(labels, preds)
            self._log_figure(f"val/kNN Confusion Matrix", fig)
            self.log_dict({
                "val/knn_accuracy":accuracy_score(labels, preds),
                "val/knn_mf1": f1_score(labels, preds, average="macro"),
                "val/knn_kappa": cohen_kappa_score(labels, preds)
            }, on_epoch=True)

            # Perform linear classfication as well
            # classifier = LinearClassifier(
            # learning_rate=self.hparams["learning_rate"],
            # weight_decay=self.hparams["weight_decay"],
            # epochs=30,
            # device=self.device,
            # )
            # with torch.enable_grad():
            # classifier.fit(train_cls_embeddings, train_labels)
            # preds = classifier.predict(cls_embeddings)
            # self.log_dict({
            # "val/linear_accuracy":accuracy_score(labels, preds),
            # "val/linear_mf1": f1_score(labels, preds, average="macro"),
            # "val/linear_kappa": cohen_kappa_score(labels, preds)
            # }, on_epoch=True)

    def get_representations_from_dataloader(self, dataloader: DataLoader):
        self.eval()
        cls_embeddings = []
        timestamp_embeddings = []
        labels = []

        with torch.inference_mode():
            with tqdm(
                dataloader, total=len(dataloader), unit="batch", leave=True,
                desc="Extracting representations from dataloader",
            ) as pbar:
                for batch in pbar:
                    x, y = batch
                    x = x.to(self.device)  # Ensure data is on the correct device
                    cls, timestamps = self.get_representations(x)
                    cls_embeddings.append(cls.cpu().numpy())
                    timestamp_embeddings.append(timestamps.cpu().numpy())
                    labels.append(y.cpu().numpy())

        return (
            np.concatenate(cls_embeddings, axis=0), 
            np.concatenate(timestamp_embeddings, axis=0),
            np.concatenate(labels, axis=0)
        )
