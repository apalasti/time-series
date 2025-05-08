import pathlib

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from src.pretraining import PretrainedTimeDRL
from scripts.utils import load_datasets, get_config
from src.linear_models import LinearReconstructor, LinearClassifier


MODELS_DIR = pathlib.Path(__file__).parent.parent / "models"
DATASETS = ["FingerMovements", "Epilepsy", "HAR", "PenDigits", "WISDM"]

if torch.cuda.is_available(): 
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

results = []
for dataset_name in DATASETS:
    config = get_config(dataset_name, "pretraining")
    train_ds, _, test_ds = load_datasets(dataset_name, config)

    model = PretrainedTimeDRL.load_from_checkpoint(
        MODELS_DIR / f"{dataset_name}_pretrained.ckpt", device, strict=False
    )

    train_cls, train_timestamp, train_labels = model.get_representations_from_dataloader(
        DataLoader(train_ds, batch_size=10, shuffle=False)
    )
    test_cls, test_timestamp, test_labels = model.get_representations_from_dataloader(
        DataLoader(test_ds, batch_size=10, shuffle=False)
    )

    cls_classifier = LinearClassifier(
        dropout_rate=0.0,
        learning_rate=config["finetuning"]["learning_rate"],
        weight_decay=config["finetuning"]["weight_decay"],
        epochs=30,
        device=device,
        # random_state=849213,
    )
    cls_classifier.fit(train_cls, train_labels)
    cls_preds = cls_classifier.predict(test_cls)

    ts_classifier = LinearClassifier(
        dropout_rate=0.0,
        learning_rate=config["finetuning"]["learning_rate"],
        weight_decay=config["finetuning"]["weight_decay"],
        epochs=30,
        device=device,
        # random_state=849213,
    )
    ts_classifier.fit(train_timestamp, train_labels)
    ts_preds = ts_classifier.predict(test_timestamp)

    result = {
        "dataset": dataset_name,
        "instance_acc": accuracy_score(test_labels, cls_preds),
        "instance_mf1": f1_score(test_labels, cls_preds, average="macro"),
        "instance_kappa": cohen_kappa_score(test_labels, cls_preds),
        "timestep_acc": accuracy_score(test_labels, ts_preds),
        "timestep_mf1": f1_score(test_labels, ts_preds, average="macro"),
        "timestep_kappa": cohen_kappa_score(test_labels, ts_preds),
    }

    cls_reconstructor = LinearReconstructor(
        dropout_rate=0.2,
        learning_rate=config["learning_rate"],
        epochs=30,
        device=device,
        # random_state=849213,
    )
    cls_reconstructor.fit(train_timestamp, train_cls)

    dropout_probs = np.linspace(0, 0.9, 10)
    for prob in dropout_probs:
        dropped_timestamps = (
            F.dropout1d(torch.from_numpy(test_timestamp), p=prob)
            .numpy()
        )
        rec_cls_predictions = cls_reconstructor.predict(dropped_timestamps)

        numerator = np.sum(rec_cls_predictions * test_cls, axis=1)
        denominator = np.linalg.norm(rec_cls_predictions, axis=1) * np.linalg.norm(test_cls, axis=1)
        cos_sim = numerator / denominator

        preds = cls_classifier.predict(rec_cls_predictions)
        result[f"reconstructed_instance_cos_similarity(missing={prob:.2f})"] = cos_sim.mean()
        result[f"reconstructed_instance_acc(missing={prob:.2f})"] = accuracy_score(test_labels, preds)
        result[f"reconstructed_instance_mf1(missing={prob:.2f})"] = f1_score(test_labels, preds, average="macro")
        result[f"reconstructed_instance_kappa(missing={prob:.2f})"] = cohen_kappa_score(test_labels, preds)

    results.append(result)


results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)
print(results_df)


# def plot_history(history):
# fig = plt.figure(figsize=(8, 4))
# for i, (name, data) in enumerate(history.items()):
# ax = fig.add_subplot(len(history) // 2 + 1, 2, i + 1)
# ax.plot(data)
# ax.set_title(f"{name.title().replace('_', ' ')} (Final: {data[-1]:.4f})")
# ax.grid(True)
# fig.tight_layout()
# return fig
