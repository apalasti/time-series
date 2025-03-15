from typing import Dict

import argparse
import json
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# from lightning.pytorch.loggers import WandbLogger
# from torch.utils.data import random_split

from src.datasets import ClassificationDataset
from src.pretraining import PretrainedTimeDRL


DATASETS_DIR = Path(__file__).parent.parent / "datasets"
CONFIGS_PATH = Path(__file__).parent.parent / "configs"
MODELS_PATH = Path(__file__).parent.parent / "models"


def get_config(dataset_name) -> Dict:
    config_path = CONFIGS_PATH / f"{dataset_name}.json"
    try:
        with open(config_path, "r") as f:
            dataset_config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Config file not found for dataset: {dataset_name}. Expected path: {config_path}"
        )
    return dataset_config


def load_datasets(dataset_name):
    # Search for dataset folder
    dataset_path = next(
        (p for p in DATASETS_DIR.glob(f"**/{dataset_name}") if p.is_dir()), None
    )
    if dataset_path is None:
        raise FileNotFoundError(
            f"Dataset folder for {dataset_name} not found in {DATASETS_DIR}"
        )

    if str(dataset_path.parent.name).lower() == "classification":
        train_ds = ClassificationDataset(dataset_path / "train.pt")

        mean, std = train_ds.mean(), train_ds.std()
        transform = lambda sample, label: (
            ((sample - mean) / std).float(),
            label.long(),
        )

        train_ds.transform = transform
        val_ds = ClassificationDataset(dataset_path / "val.pt", transform=transform)
        test_ds = ClassificationDataset(dataset_path / "test.pt", transform=transform)
    else:
        raise ValueError("Currently only classification datasets are supported.")

    return train_ds, val_ds, test_ds


def main():
    L.seed_everything(seed=2023)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to use (e.g., Epilepsy)",
    )
    args = parser.parse_args()

    config = get_config(args.dataset)
    print(
        f"Loaded configuration for dataset '{args.dataset}':"
        + json.dumps(config, indent=4)
    )

    train_ds, val_ds, _ = load_datasets(args.dataset)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=False
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False
    )

    MODELS_PATH.mkdir(exist_ok=True)
    config.update(**config.get("pretraining", {}))

    model = PretrainedTimeDRL(**config)
    # wandb_logger = WandbLogger(
        # name=f"{args.dataset}:Pretrained", project="TimeSeries", group=args.dataset
    # )
    trainer = L.Trainer(
        max_epochs=config["epochs"],
        # logger=wandb_logger,
        callbacks=[
            EarlyStopping("train/loss", patience=config["patience"]),
            ModelCheckpoint(
                dirpath=MODELS_PATH,
                filename=f"{args.dataset}_pretrained",
                monitor="train/loss",
                every_n_epochs=1,
            ),
        ],
        log_every_n_steps=1,
        num_sanity_val_steps=0
    )
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
