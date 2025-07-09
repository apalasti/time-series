import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .utils import MODELS_PATH, load_finetuned_model_data
from src.masking import mask_timesteps, shapley_explainer, diversity_of_unmasked
from src.linear_models import LinearClassifier


NUM_SHAP_TRIALS = 3
NUM_RANDOM_TRIALS = 15


def get_device() -> torch.device:
    """Determine the best available device for computation."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_masking_strategies(
    train_timesteps: np.ndarray,
    test_timesteps: np.ndarray,
    classifier: LinearClassifier,
):
    """Evaluate accuracy as a function of masking percentage."""
    (_, T, _) = test_timesteps.shape
    masked_percentages = np.linspace(0.0, 0.9, 10)

    shap_orders = [
        shapley_explainer(classifier, train_timesteps)[0]
        for _ in range(NUM_SHAP_TRIALS)
    ]

    shap_preds, shap_diversity = [], []
    random_preds, random_diversity = [], []

    for percentage in tqdm(masked_percentages, desc="Evaluating masking strategies"):
        masked_count = int(T * percentage)

        # SHAP-based masking
        shap_preds.append([])
        shap_diversity.append([])
        for shap_order in shap_orders:
            timesteps_shap = mask_timesteps(test_timesteps, shap_order[:masked_count])
            shap_preds[-1].append(classifier.predict(timesteps_shap))
            shap_diversity[-1].append(diversity_of_unmasked(test_timesteps, shap_order[:masked_count]))

        # Random masking trials
        random_preds.append([])
        random_diversity.append([])
        for _ in range(NUM_RANDOM_TRIALS):
            masks = np.array([
                np.random.choice(np.arange(T), size=masked_count, replace=False)
                for _ in range(len(test_timesteps))
            ])
            timesteps_random = mask_timesteps(test_timesteps, masks)

            random_preds[-1].append(classifier.predict(timesteps_random))
            random_diversity[-1].append(diversity_of_unmasked(test_timesteps, masks))

    return {
        "masked_percentage": masked_percentages,
        "shap_orders": shap_orders,
        "shap_preds": shap_preds,
        "shap_diversity": shap_diversity,
        "random_preds": random_preds,
        "random_diversity": random_diversity
    }


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
        print("Calculating importance of each timestep feature based on Shapley values")

        # Evaluate masking strategies
        masking_results = evaluate_masking_strategies(train_timesteps, test_timesteps, classifier)

        # Store results
        results[dataset_name] = {
            "classifier_history": classifier.history_,
            "labels": test_labels,
            **masking_results
        }

    # Save results
    results_path = output_path / "collected_masking_results.npy"
    np.save(results_path, results, allow_pickle=True)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()
