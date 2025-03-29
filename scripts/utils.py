import json
from pathlib import Path
from typing import Dict, Literal

import torch

from src.datasets import ClassificationDataset, load_forecasting_dataset

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
CONFIGS_PATH = Path(__file__).parent.parent / "configs"
MODELS_PATH = Path(__file__).parent.parent / "models"


def get_config(dataset_name, job_type: Literal["pretraining", "finetuning"]) -> Dict:
    """Load configuration file for a given dataset."""
    config_path = CONFIGS_PATH / f"{dataset_name}.json"
    try:
        with open(config_path, "r") as f:
            config: Dict = json.load(f)
            job_specific = config.pop(job_type, None)
            config.pop("finetuning" if job_type == "pretraining" else "pretraining")
            return {**config, **job_specific}
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Config file not found for dataset: {dataset_name}. Expected path: {config_path}"
        )


def load_datasets(dataset_name: str, config: Dict):
    """Load datasets based on dataset type (classification or forecasting)."""
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

        val_path = dataset_path / "val.pt"
        val_ds = (
            ClassificationDataset(val_path, transform=transform)
            if val_path.exists()
            else ClassificationDataset(dataset_path / "test.pt", transform=transform)
        )
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
            "test",
            config["sequence_len"],
            config["prediction_len"],
            transform=transform,
        )

    return train_ds, val_ds, test_ds


def create_data_loaders(dataset: str, config: Dict):
    train_ds, val_ds, test_ds = load_datasets(dataset, config)
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False
    )
    test_dl = torch.utils.data.DataLoader(
        test_ds, batch_size=config["batch_size"], shuffle=False
    )
    return train_dl, val_dl, test_dl
