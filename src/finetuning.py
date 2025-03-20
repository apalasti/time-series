import lightning as L
import numpy as np
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    cohen_kappa_score,
)

from src.pretraining import PretrainedTimeDRL
from src.utils import load_lr_scheduler


class ClassificationFineTune(L.LightningModule):
    def __init__(
        self,
        pretrained: PretrainedTimeDRL,
        **config
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_model"])

        self.pretrained = pretrained
        self.classifier = nn.Linear(config["d_model"], config["num_classes"])

    def training_step(self, batch, batch_idx):
        x, y = batch

        self.pretrained.freeze()
        self.pretrained.eval()
        cls_embeddings, _ = self.pretrained.get_representations(x)
        logits = self.classifier(cls_embeddings)

        loss = F.cross_entropy(logits, y)

        self.log("train/loss", loss, prog_bar=True, on_step=True)

        return loss

    def on_validation_epoch_start(self):
        self.preds = []
        self.labels = []

    def validation_step(self, batch, batch_idx):
        x, y = batch

        cls_embeddings, _ = self.pretrained.get_representations(x)
        logits = self.classifier(cls_embeddings)

        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)

        self.preds.append(preds.cpu().numpy())
        self.labels.append(y.cpu().numpy())

        self.log_dict({"val/loss": loss}, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self):
        preds = np.concatenate(self.preds)
        labels = np.concatenate(self.labels)

        self._log_classification_metrics(preds, labels, "val")

        del self.preds
        del self.labels

    def on_test_epoch_start(self) -> None:
        self.preds = []
        self.labels = []

    def test_step(self, batch, batch_idx):
        x, y = batch

        cls_embeddings, _ = self.pretrained.get_representations(x)
        logits = self.classifier(cls_embeddings)
        preds = torch.argmax(logits, dim=1)

        self.preds.append(preds.cpu().numpy())
        self.labels.append(y.cpu().numpy())

    def on_test_epoch_end(self):
        preds = np.concatenate(self.preds)
        labels = np.concatenate(self.labels)

        self._log_classification_metrics(preds, labels, "test")

        del self.preds
        del self.labels

    def _log_classification_metrics(self, preds: np.ndarray, labels: np.ndarray, prefix: str):
        self.log_dict({
            f"{prefix}/acc": accuracy_score(labels, preds),
            f"{prefix}/MF1": f1_score(labels, preds, average="macro"),
            f"{prefix}/Kappa": cohen_kappa_score(labels, preds),
        })

        cm = confusion_matrix(labels, preds)
        fig = px.imshow(
            cm / cm.sum(axis=1, keepdims=True),
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=[f"Class {i}" for i in range(cm.shape[0])],
            y=[f"Class {i}" for i in range(cm.shape[1])],
            color_continuous_scale="Blues", text_auto=True,
        )
        self.__log_figure(f"{prefix}/Confusion Matrix", fig)

    def __log_figure(self, name, fig):
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
