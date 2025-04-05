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
    return parser.parse_args()


def main():
    L.seed_everything(seed=2023)
    MODELS_PATH.mkdir(exist_ok=True)

    args = parse_args()
    config = get_config(args.dataset, "pretraining")
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
