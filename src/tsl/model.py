import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from ..base import BaseModule
from .patch_tst import PatchTST
from .informer import Informer
from .itransformer import iTransformer
from .ns_transformer import NSTransformer


TSL_MODELS = {
    "PatchTST": PatchTST,
    "Informer": Informer,
    "iTransformer": iTransformer,
    "NSTransformer": NSTransformer,
}


class TSLModel(BaseModule):
    def __init__(self, **config):
        super().__init__()
        self.save_hyperparameters()

        self.model = TSL_MODELS[config["model"]](config)

    def training_step(self, batch, batch_idx):
        x, y = batch
        B, T, C = x.shape
        assert (
            T == self.hparams.sequence_len and C == self.hparams.input_channels
        ), f"Shape mismatch. Expected (N, {self.hparams.sequence_len}, {self.hparams.input_channels}), got (N, {T}, {C})"

        outputs = self.model(x, None, None, None)
        loss = F.cross_entropy(outputs, y, reduction="mean")

        self.log_dict({
            "train/loss": loss,
        })

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        outputs = self.model(x, None, None, None)
        loss = F.cross_entropy(outputs, y)

        preds = F.softmax(outputs, dim=-1).argmax(dim=-1).cpu().numpy()
        targets = y.detach().cpu().numpy()

        self.log_dict({
            "val/loss": loss,
            "val/accuracy": accuracy_score(targets, preds),
            "val/mf1": f1_score(targets, preds, average="macro"),
        }, on_epoch=True)

    def __call__(self, x_batch):
        with torch.no_grad(), torch.inference_mode():
            return self.model(x_batch, None, None, None)
