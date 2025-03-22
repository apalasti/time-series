import argparse
import json

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.finetuning import ClassificationFineTune, ForecastingFineTune
from src.pretraining import PretrainedTimeDRL

from .utils import MODELS_PATH, get_config, create_data_loaders


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
    config = get_config(args.dataset, "finetuning")
    print(
        f"Loaded configuration for dataset '{args.dataset}':"
        + json.dumps(config, indent=4)
    )

    # Load pretrained model
    checkpoint_path = MODELS_PATH / f"{args.dataset}_pretrained.ckpt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Pretrained model not found at {checkpoint_path}. Please run pretraining first."
        )
    pretrained = PretrainedTimeDRL.load_from_checkpoint(checkpoint_path)

    train_dl, val_dl, test_dl = create_data_loaders(args.dataset, config)

    if config["task_type"] == "forecasting":
        finetune_model_class = ForecastingFineTune
    elif config["task_type"] == "classification":
        finetune_model_class = ClassificationFineTune
    else:
        raise ValueError(
            f"Unsupported task type: {config['task_type']}. Must be either 'forecasting' or 'classification'"
        )

    logger = None
    if args.use_wandb:
        logger = WandbLogger(
            name=f"{args.dataset}:Finetuned", project="TimeSeries", group=args.dataset
        )

    model = finetune_model_class(pretrained, **config)
    trainer = L.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=[
            EarlyStopping("val/loss", patience=config["patience"], mode="max"),
            ModelCheckpoint(
                dirpath=MODELS_PATH,
                filename=f"{args.dataset}_finetuned",
                monitor="val/loss", mode="max",
                every_n_epochs=1,
            ),
        ],
        log_every_n_steps=10,
    )
    trainer.fit(model, train_dl, val_dl)

    model = finetune_model_class.load_from_checkpoint(
        MODELS_PATH / f"{args.dataset}_finetuned.ckpt", pretrained=pretrained
    )
    trainer.test(model, test_dl)


if __name__ == "__main__":
    main()
