import lightning as L
import numpy as np
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error)

from src.base import BaseModule
from src.pretraining import PretrainedTimeDRL
from src.utils import visualize_predictions, visualize_weights, plot_confusion_matrix


class ClassificationFineTune(BaseModule):
    def __init__(
        self,
        pretrained: PretrainedTimeDRL,
        **config
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained"])

        self.pretrained = pretrained
        self._create_classifiers()

    def _create_classifiers(self):
        dummy = torch.randn(
            (1, self.hparams["sequence_len"], self.hparams["input_channels"])
        )
        _, patched_seq_len, _ = self.pretrained.create_patches(dummy).shape

        self.classifier = nn.Linear(
            self.hparams["d_model"], self.hparams["num_classes"]
        )
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        self.timestamp_classifier = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(
                self.hparams["d_model"] * patched_seq_len, self.hparams["num_classes"]
            ),
        )
        nn.init.xavier_uniform_(self.timestamp_classifier[1].weight)
        nn.init.zeros_(self.timestamp_classifier[1].bias)

    def training_step(self, batch, batch_idx):
        x, y = batch

        self.pretrained.freeze()
        self.pretrained.eval()
        cls_embeddings, timestamps = self.pretrained.get_representations(x)

        cls_logits = self.classifier(cls_embeddings)
        timestamp_logits = self.timestamp_classifier(timestamps)

        cls_loss = F.cross_entropy(cls_logits, y)
        timestamp_loss = F.cross_entropy(timestamp_logits, y)

        self.log_dict(
            {"train/cls_loss": cls_loss, "train/timestamp_loss": timestamp_loss},
            prog_bar=True, on_step=True,
        )

        return cls_loss + timestamp_loss

    def on_validation_epoch_start(self):
        self.val_cls_preds = []
        self.val_timestamp_preds = []
        self.val_labels = []

    def validation_step(self, batch, batch_idx):
        x, y = batch

        cls_embeddings, timestamps = self.pretrained.get_representations(x)
        cls_logits = self.classifier(cls_embeddings)
        timestamp_logits = self.timestamp_classifier(timestamps)

        cls_loss = F.cross_entropy(cls_logits, y)
        timestamp_loss = F.cross_entropy(timestamp_logits, y)

        cls_preds = torch.argmax(cls_logits, dim=1)
        timestamp_preds = torch.argmax(timestamp_logits, dim=1)

        self.val_cls_preds.append(cls_preds.cpu().numpy())
        self.val_timestamp_preds.append(timestamp_preds.cpu().numpy())
        self.val_labels.append(y.cpu().numpy())

        self.log_dict(
            {"val/loss": cls_loss, "val/cls_loss": cls_loss, "val/timestamp_loss": timestamp_loss},
            prog_bar=True, on_epoch=True
        )

    def on_validation_epoch_end(self):
        cls_preds = np.concatenate(self.val_cls_preds)
        timestamp_preds = np.concatenate(self.val_timestamp_preds)
        labels = np.concatenate(self.val_labels)

        timestamp_weights: np.ndarray = (
            self.timestamp_classifier[1].weight.detach().cpu().numpy()
            .reshape((self.hparams["num_classes"], -1, self.hparams["d_model"]))
            .mean(axis=-1)
        )
        fig = visualize_weights(timestamp_weights)
        self._log_figure("val/Timestamp Classifier Weights", fig)

        self._log_classification_metrics(cls_preds, labels, "val", "cls")
        self._log_classification_metrics(timestamp_preds, labels, "val", "timestamp")

        del self.val_cls_preds
        del self.val_timestamp_preds
        del self.val_labels

    def on_test_epoch_start(self) -> None:
        self.test_cls_preds = []
        self.test_timestamp_preds = []
        self.test_labels = []

    def test_step(self, batch, batch_idx):
        x, y = batch

        cls_embeddings, timestamps = self.pretrained.get_representations(x)
        cls_logits = self.classifier(cls_embeddings)
        timestamp_logits = self.timestamp_classifier(timestamps)

        cls_preds = torch.argmax(cls_logits, dim=1)
        timestamp_preds = torch.argmax(timestamp_logits, dim=1)

        self.test_cls_preds.append(cls_preds.cpu().numpy())
        self.test_timestamp_preds.append(timestamp_preds.cpu().numpy())
        self.test_labels.append(y.cpu().numpy())

    def on_test_epoch_end(self):
        cls_preds = np.concatenate(self.test_cls_preds)
        timestamp_preds = np.concatenate(self.test_timestamp_preds)
        labels = np.concatenate(self.test_labels)

        self._log_classification_metrics(cls_preds, labels, "test", "cls")
        self._log_classification_metrics(timestamp_preds, labels, "test", "timestamp")

        del self.test_cls_preds
        del self.test_timestamp_preds
        del self.test_labels

    def _log_classification_metrics(
        self, preds: np.ndarray, labels: np.ndarray, prefix: str, classifier_type: str
    ):
        self.log_dict(
            {
                f"{prefix}/{classifier_type}_acc": accuracy_score(labels, preds),
                f"{prefix}/{classifier_type}_MF1": f1_score(
                    labels, preds, average="macro"
                ),
                f"{prefix}/{classifier_type}_Kappa": cohen_kappa_score(labels, preds),
            }
        )

        fig = plot_confusion_matrix(labels, preds)
        self._log_figure(f"{prefix}/{classifier_type}_Confusion Matrix", fig)


class ForecastingFineTune(BaseModule):
    def __init__(
        self,
        pretrained: PretrainedTimeDRL,
        **config
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained"])

        self.pretrained = pretrained
        self.predictor = self._create_predictor()

    def _create_predictor(self):
        pred_len = self.hparams["prediction_len"]
        in_channels = self.hparams["input_channels"]

        _, patched_seq_len, _ = self.pretrained.create_patches(
            torch.randn((1, self.hparams["sequence_len"], in_channels))
        ).shape

        if self.hparams["enable_channel_independence"]:
            self.rearrange = lambda x: torch.transpose(x, 2, 1)
            return nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(patched_seq_len * self.hparams["d_model"], pred_len),
                nn.Unflatten(0, (-1, self.hparams["input_channels"])),
            )
        else:
            self.rearrange = lambda x: torch.reshape(x, (-1, pred_len, in_channels))
            return nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(
                    patched_seq_len * self.hparams["d_model"], pred_len * in_channels
                ),
            )

    def get_predictions(self, x: torch.Tensor):
        _, timestamps = self.pretrained.get_representations(x)
        predictions = self.rearrange(self.predictor(timestamps))
        predictions = self.pretrained.instance_norm(predictions, "denorm")
        return predictions

    def training_step(self, batch, batch_idx):
        past, future = batch

        self.pretrained.freeze()
        self.pretrained.eval()
        predictions = self.get_predictions(past)

        loss = F.mse_loss(predictions, future)
        self.log("train/loss", loss, prog_bar=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        past, future = batch

        self.pretrained.freeze()
        self.pretrained.eval()
        predictions = self.get_predictions(past)

        if batch_idx == 0:
            sample_idx = 0
            fig = visualize_predictions(
                past[sample_idx].detach().cpu().numpy(),
                future[sample_idx].detach().cpu().numpy(),
                predictions[sample_idx].detach().cpu().numpy(),
            )
            self._log_figure("val/Predictions", fig)

        loss = F.mse_loss(predictions, future)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_start(self):
        self.preds = []
        self.targets = []

    def test_step(self, batch, batch_idx):
        past, future = batch

        self.pretrained.freeze()
        self.pretrained.eval()
        predictions = self.get_predictions(past)

        self.preds.append(predictions.detach().cpu().numpy())
        self.targets.append(future.detach().cpu().numpy())

    def on_test_epoch_end(self):
        preds = np.concatenate(self.preds)
        targets = np.concatenate(self.targets)

        preds = preds.reshape((preds.shape[0], -1))
        targets = targets.reshape((targets.shape[0], -1))

        self.log_dict({
            "test/mse": mean_squared_error(preds, targets),
            "test/mae": mean_absolute_error(preds, targets),
        })
