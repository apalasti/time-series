import re
import argparse
from pathlib import Path

import numpy as np

from src.tsl.model import TSLModel
from src.time_shap import calculate_shapley_values
from .utils import load_datasets, OUT_PATH


def parse_args():
    def model_file_type(file_path):
        pattern = r".*/?([^_/]+)_([^_/]+)\.ckpt$"
        if not re.match(pattern, file_path):
            raise argparse.ArgumentTypeError(
                "The model file must match the <MODEL>_<DATASET>.ckpt pattern"
            )
        return file_path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_file",
        type=model_file_type,
        help="Path to the model checkpoint file matching the <MODEL>_<DATASET>.ckpt pattern",
    )
    parser.add_argument("--layer", type=str, required=True)
    parser.add_argument(
        "--max_samples", type=int, default=1500, 
        help="Maximum number of samples to calculate SHAP values for (default: 1500)"
    )
    return parser.parse_args()


def main(args):
    model_file = Path(args.model_file)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file '{model_file}' does not exist. Provide a valid path to a model checkpoint file."
        )

    model_name, dataset_name = model_file.stem.split("_")
    model = TSLModel.load_from_checkpoint(model_file)
    train_ds, _, _ = load_datasets(dataset_name, {})
    print(f"Loaded model: {model_name}(device={model.device}), Dataset: {dataset_name}")

    # Only calculate for args.max_samples samples at max
    n_select = min(args.max_samples, len(train_ds.samples))
    selected_indices = np.random.choice(len(train_ds.samples), size=n_select, replace=False)
    train_ds.samples = train_ds.samples[selected_indices]

    shapley_values = calculate_shapley_values(model, train_ds, args.layer)

    safe_layer_name = args.layer.replace(".", "-")
    suffix, fname = 1, f"{model_name}_{dataset_name}_{safe_layer_name}.npy"
    while (OUT_PATH / fname).exists():
        fname = f"{model_name}_{dataset_name}_{safe_layer_name}_{suffix}.npy"
        suffix += 1
    np.save(OUT_PATH / fname, { "selected_indicies": selected_indices, "shapley_values": shapley_values.values})


if __name__ == "__main__":
    main(parse_args())
