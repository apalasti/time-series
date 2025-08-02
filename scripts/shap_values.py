import argparse
from pathlib import Path

import numpy as np
import torch

from .utils import MODELS_PATH, load_finetuned_model_data, get_device
from src.masking import calculate_shapley_values
from src.linear_models import LinearClassifier


NUM_SHAP_TRIALS = 3
NUM_RANDOM_TRIALS = 15


def main():
    device = get_device()
    print(f"Using device: {device}")

    parser = argparse.ArgumentParser(description="Analyze masking effects on time series classification.")
    parser.add_argument("--models_dir", type=str, default=MODELS_PATH,
                       help="Path to the directory containing pretrained models.")
    parser.add_argument("--output_dir", type=str, default=MODELS_PATH,
                       help="Path to the directory where results will be saved.")
    args = parser.parse_args()

    models_path = Path(args.models_dir)
    output_path = Path(args.output_dir)

    # Get available models
    available_models = {
        p.name.rsplit("_", maxsplit=1)[0] for p in models_path.glob("*_pretrained.ckpt")
    }
    print(f"Available pretrained models ({models_path}): {available_models}")

    results = {}
    for dataset_name in available_models:
        print(f"Analyzing dataset: {dataset_name}")

        # Load data
        data = load_finetuned_model_data(models_path, dataset_name, device)
        train_timesteps, train_labels = data["train"]["timesteps"], data["train"]["labels"]
        test_timesteps, test_labels = data["test"]["timesteps"], data["test"]["labels"]
        classifier: LinearClassifier = data["classifier"]

        # Calculate Shapley importance
        print("Calculating Shapley values...")
        shap_values = [
            calculate_shapley_values(classifier, train_timesteps)
            for _ in range(NUM_SHAP_TRIALS)
        ]
        # Store results
        results[dataset_name] = {
            "shap_values": shap_values,
        }

    # Save results
    results_path = output_path / "shap_values.npy"
    np.save(results_path, results, allow_pickle=True)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
