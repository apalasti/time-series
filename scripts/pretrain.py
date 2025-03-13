import argparse
import json
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import random_split

from src.datasets import ClassificationDataset
from src.pretraining import PretrainedTimeDRL

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
CONFIGS_PATH = Path(__file__).parent.parent / "configs"
MODELS_PATH = Path(__file__).parent.parent / "models"


def get_config(dataset_name):
    config_path = CONFIGS_PATH / f"{dataset_name}.json"
    try:
        with open(config_path, "r") as f:
            dataset_config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Config file not found for dataset: {dataset_name}. Expected path: {config_path}"
        )
    return dataset_config


def get_data_loaders(dataset_name):
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
        val_ds = (
            ClassificationDataset(dataset_path / "train.pt")
            if (dataset_path / "val.pt").exists()
            else None
        )
        test_ds = ClassificationDataset(dataset_path / "test.pt")
    else:
        raise ValueError("Currently only classification datasets are supported.")

    if val_ds is None:
        # NOTE: For forecasting this will not pass
        val_ds = random_split(train_ds, [0.8, 0.2])

    return train_ds, val_ds, test_ds


def main():
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

    train_ds, val_ds, _ = get_data_loaders(args.dataset)

    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False
    )

    MODELS_PATH.mkdir(exist_ok=True)

    model = PretrainedTimeDRL(**config)
    wandb_logger = WandbLogger(
        name=f"{args.dataset}:Pretrained", project="TimeSeries", group=args.dataset
    )
    trainer = L.Trainer(
        max_epochs=config["pretrain_epochs"],
        logger=wandb_logger,
        callbacks=[
            EarlyStopping("train/loss", patience=config["patience"]),
            ModelCheckpoint(
                dirpath=MODELS_PATH,
                filename=f"{args.dataset}_pretrained",
                monitor="train/loss",
                every_n_epochs=1,
            ),
        ],
    )
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
