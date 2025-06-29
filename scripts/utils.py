import json
from pathlib import Path
from typing import Dict, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.linear_models import LinearClassifier
from src.pretraining import PretrainedTimeDRL
from src.datasets import ClassificationDataset, load_forecasting_dataset
from src.time_drl import AttentionInspector


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
            # config.pop("finetuning" if job_type == "pretraining" else "pretraining")
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


def load_finetuned_model_data(models_path: Path, dataset_name: str, device: torch.device = None):
    # Define cache path and check if cached representations exist
    cache_path = models_path / "cached" / f"{dataset_name}.npy"
    if cache_path.exists():
        # Load cached representations
        cached_data = np.load(cache_path, allow_pickle=True)[()]
        cached_data["classifier"] = LinearClassifier.load_state_dict(cached_data["classifier"])
        return cached_data

    # Compute and cache representations
    config = get_config(dataset_name, "pretraining")
    train_ds, _, test_ds = load_datasets(dataset_name, config)
    model = PretrainedTimeDRL.load_from_checkpoint(
        models_path / f"{dataset_name}_pretrained.ckpt", device, strict=False
    )

    # inspector = AttentionInspector(model)
    train_cls, train_timesteps, train_labels = model.get_representations_from_dataloader(
        DataLoader(train_ds, batch_size=10, shuffle=False)
    )
    # train_attn_values = inspector.get_attention_values().cpu().numpy()
    # inspector.clear()

    test_cls, test_timesteps, test_labels = model.get_representations_from_dataloader(
        DataLoader(test_ds, batch_size=10, shuffle=False)
    )
    # test_attn_values = inspector.get_attention_values().cpu().numpy()
    # inspector.clear()

    # Train classifier on top of timesteps
    classifier = LinearClassifier(
        dropout_rate=0.2,
        learning_rate=config["finetuning"]["learning_rate"],
        weight_decay=config["finetuning"]["weight_decay"],
        epochs=30,
        device=device,
        random_state=849213,
    )
    classifier.fit(train_timesteps, train_labels)

    # Save representations to cache
    cache_path.parent.mkdir(exist_ok=True, parents=True)
    cached_data = {
        "classifier": classifier.state_dict(),
        "train": {
            "cls": train_cls,
            "timesteps": train_timesteps,
            "labels": train_labels,
            # "attn_values": train_attn_values,
        },
        "test": {
            "cls": test_cls,
            "timesteps": test_timesteps,
            "labels": test_labels,
            # "attn_values": test_attn_values,
        },
    }
    np.save(cache_path, cached_data)

    cached_data["classifier"] = classifier
    return cached_data
