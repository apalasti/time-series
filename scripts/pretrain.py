import argparse
import json
from pathlib import Path
from typing import Dict

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from src.datasets import ClassificationDataset, load_forecasting_dataset
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


def load_datasets(dataset_name, config):
    # Search for dataset folder
    dataset_path = [
        p for p in DATASETS_DIR.glob(f"**/{dataset_name}") if p.is_dir()
    ] + [p for p in DATASETS_DIR.glob(f"**/{dataset_name}.csv")]
    if not dataset_path:
        raise FileNotFoundError(
            f"Dataset folder for {dataset_name} not found in {DATASETS_DIR}"
        )

    dataset_path = dataset_path[0]
    if "classification" in str(dataset_path.absolute()).lower():
        train_ds = ClassificationDataset(dataset_path / "train.pt")

        mean, std = train_ds.mean(), train_ds.std()
        transform = lambda sample, label: (
            ((sample - mean) / std).float(),
            label.long(),
        )
        train_ds.transform = transform

        if (dataset_path / "val.pt").exists():
            val_ds = ClassificationDataset(dataset_path / "val.pt", transform=transform)
        else:
            # NOTE: For pretraining use the test dataset as validation.  This
            # could be wrong, since we use the best performing (on the test
            # dataset) pretrained model.
            val_ds = ClassificationDataset(dataset_path / "test.pt", transform=transform)
        test_ds = ClassificationDataset(dataset_path / "test.pt", transform=transform)
    else:
        train_ds = load_forecasting_dataset(
            dataset_path, "train", config["sequence_len"], config["prediction_len"]
        )

        mean, std = train_ds.mean(), train_ds.std()
        transform = lambda past, future: (
            ((past - mean) / std).float(),
            ((future - mean) / std).float(),
        )
        train_ds.transform = transform

        val_ds = load_forecasting_dataset(
            dataset_path,
            "validation",
            config["sequence_len"],
            config["prediction_len"],
            transform=transform,
        )
        test_ds = load_forecasting_dataset(
            dataset_path,
            "validation",
            config["sequence_len"],
            config["prediction_len"],
            transform=transform,
        )

    return train_ds, val_ds, test_ds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to use (e.g., Epilepsy)",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    return parser.parse_args()


def main():
    L.seed_everything(seed=2023)
    MODELS_PATH.mkdir(exist_ok=True)

    args = parse_args()
    config = get_config(args.dataset)
    print(
        f"Loaded configuration for dataset '{args.dataset}':"
        + json.dumps(config, indent=4)
    )

    train_ds, val_ds, _ = load_datasets(args.dataset, config)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        # num_workers=2, pin_memory=True,
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False
    )

    logger = None
    if args.use_wandb:
        logger = WandbLogger(
            name=f"{args.dataset}:Pretrained", project="TimeSeries", group=args.dataset
        )

    config.update(**config.get("pretraining", {}))
    model = PretrainedTimeDRL(**config)
    trainer = L.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=[
            EarlyStopping("train/loss", patience=config["patience"]),
            ModelCheckpoint(
                dirpath=MODELS_PATH,
                filename=f"{args.dataset}_pretrained",
                monitor="train/loss",
                every_n_epochs=1,
            ),
        ],
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dl, val_dl)


if __name__ == "__main__":
    main()
