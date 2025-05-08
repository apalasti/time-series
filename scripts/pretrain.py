import argparse
import json

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from src.pretraining import PretrainedTimeDRL

from .utils import MODELS_PATH, create_data_loaders, get_config


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
    parser.add_argument(
        "--params",
        nargs="*",
        default=[],
        help="Override config parameters, e.g. --params epochs=20 lr=0.001",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2023,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def update_config_from_args(config, params_from_args):
    for item in params_from_args:
        if "=" not in item:
            raise ValueError(f"Invalid format for --params: '{item}'. Use key=value.")
        key, value = item.split("=", 1)
        # Try to infer type from existing config
        if key in config:
            orig_type = type(config[key])
            try:
                value = orig_type(value)
            except Exception:
                raise ValueError(
                    f"Failed to convert '{value}' to {orig_type.__name__} for key '{key}'. "
                    f"Original value was {config[key]} of type {orig_type.__name__}."
                )
        config[key] = value
    return config


def main():
    args = parse_args()

    L.seed_everything(seed=args.seed)
    MODELS_PATH.mkdir(exist_ok=True)

    config = get_config(args.dataset, "pretraining")
    config = update_config_from_args(config, args.params)
    print(
        f"Loaded configuration for dataset '{args.dataset}':"
        + json.dumps(config, indent=4)
    )

    train_dl, val_dl, test_dl = create_data_loaders(args.dataset, config)

    logger = None
    if args.use_wandb:
        logger = WandbLogger(
            name=f"{args.dataset}:Pretrained", project="TimeSeries", group=args.dataset
        )

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
    trainer.fit(model, train_dl, test_dl)


if __name__ == "__main__":
    main()
