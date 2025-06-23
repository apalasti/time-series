import shap
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import argparse
from pathlib import Path

from .utils import MODELS_PATH, get_config, load_datasets
from src.pretraining import PretrainedTimeDRL
from src.linear_models import LinearClassifier


MAX_SAMPLES_FOR_SHAPLEY = 2000
NUM_RANDOM_TRIALS = 15


def shapley_explainer(model: LinearClassifier, samples: np.ndarray, seed=None):
    N, T, D = samples.shape
    sequences = np.concatenate((samples, np.zeros((1, T, D))))
    sample_ixs = np.column_stack([np.arange(N)] * T)

    def f(sample_ix: np.ndarray):
        input = sequences[sample_ix, np.arange(T)[np.newaxis, :]]
        pred = model.predict_proba(input)
        return pred

    def masker(mask: np.ndarray, sample_ix: np.ndarray):
        masked = sample_ix.copy()
        masked[~mask] = -1
        return masked[np.newaxis, :]

    explainer = shap.Explainer(f, masker, seed=seed)
    shap_values = explainer(sample_ixs)
    return shap_values


def mask_timesteps(samples: np.ndarray, masked_cols: np.ndarray):
    masked = samples.copy()
    if masked_cols.size == 0:
        return masked

    if masked_cols.ndim == 1:
        masked[:, masked_cols] = 0.0
    elif masked_cols.ndim == 2:
        if masked_cols.shape[0] != samples.shape[0]:
            raise ValueError(
                f"If masked_cols is 2D, its first dimension "
                f"({masked_cols.shape[0]}) must match the number of samples "
                f"({samples.shape[0]}) for per-sample masking."
            )
        masked[np.arange(samples.shape[0])[:, np.newaxis], masked_cols] = 0.0
    else:
        raise ValueError("masked_cols must be a 1D or 2D numpy array.")
    return masked


def get_samples(data: np.ndarray, max_samples: int) -> np.ndarray:
    all_indices = np.arange(len(data))
    sample_indices = np.random.choice(
        all_indices,
        size=min(max_samples, len(data)),
        replace=False
    )
    return data[sample_indices]


def main():
    parser = argparse.ArgumentParser(description="Analyze masking effects on time series classification.")
    parser.add_argument(
        "--models_dir",
        type=str,
        default=MODELS_PATH,
        help="Path to the directory containing pretrained models."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=MODELS_PATH,
        help="Path to the directory where results will be saved."
    )
    args = parser.parse_args()

    models_path = Path(args.models_dir)
    output_path = Path(args.output_dir)

    available_models = {
        path.name.rstrip("_pretrained.ckpt")
        for path in models_path.glob("*_pretrained.ckpt")
    }
    print(f"Available pretrained models ({models_path}): {available_models}")

    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Using device: {device}")

    results = dict()
    for dataset_name in available_models:
        results[dataset_name] = dict()
        
        print(f"Loading {dataset_name}...")
        config = get_config(dataset_name, "pretraining")
        train_ds, _, test_ds = load_datasets(dataset_name, config)
        model = PretrainedTimeDRL.load_from_checkpoint(
            models_path / f"{dataset_name}_pretrained.ckpt", device, strict=False
        )

        train_cls, train_timesteps, train_labels = model.get_representations_from_dataloader(
            DataLoader(train_ds, batch_size=10, shuffle=False)
        )
        test_cls, test_timesteps, test_labels = model.get_representations_from_dataloader(
            DataLoader(test_ds, batch_size=10, shuffle=False)
        )

        classifier = LinearClassifier(
            dropout_rate=0.2,
            learning_rate=config["finetuning"]["learning_rate"],
            weight_decay=config["finetuning"]["weight_decay"],
            epochs=30,
            device=device,
            random_state=849213,
        )
        classifier.fit(train_timesteps, train_labels)
        test_preds = classifier.predict(test_timesteps)
        results[dataset_name]["classifier_history"] = classifier.history_
        results[dataset_name]["classifier_mestrics"] = {
            "accuracy": accuracy_score(test_labels, test_preds),
            "macro_f1": f1_score(test_labels, test_preds, average="macro"),
            "kappa": cohen_kappa_score(test_labels, test_preds)
        }

        print("Calculate importance of each timestep feature based on Shapley values")
        sampled_timesteps = get_samples(train_timesteps, MAX_SAMPLES_FOR_SHAPLEY)
        shap_values = shapley_explainer(classifier, sampled_timesteps)
        results[dataset_name]["shap_explainer"] = shap_values

        # Determine the importance order
        avg_shap_magnitudes: np.ndarray = np.mean(np.abs(shap_values.values), axis=0)
        order = np.argsort((avg_shap_magnitudes / avg_shap_magnitudes.sum(axis=0)).max(axis=1))

        accuracy_random = []
        accuracy_shap = []

        masked_percentages = np.linspace(0.0, 0.9, 10)
        for percentage in tqdm(masked_percentages, desc="Evaluating accuracy determined as a function of masking"):
            masked_count = int(len(order) * (percentage))

            # SHAP-based dropout
            timesteps_shap = mask_timesteps(test_timesteps, order[:masked_count])
            preds_shap = classifier.predict(timesteps_shap)
            accuracy_shap.append(accuracy_score(test_labels, preds_shap))

            # Randomly mask out the timestep features of the test samples
            avg_accuracy = 0
            for _ in range(NUM_RANDOM_TRIALS):
                timesteps_random = mask_timesteps(test_timesteps, np.array([
                    np.random.choice(order, size=masked_count, replace=False)
                    for _ in range(len(test_timesteps))
                ]))
                preds_random = classifier.predict(timesteps_random)
                avg_accuracy += accuracy_score(test_labels, preds_random)
            avg_accuracy /= NUM_RANDOM_TRIALS
            accuracy_random.append(avg_accuracy)
        
        results[dataset_name]["accuracy_shap"] = accuracy_shap
        results[dataset_name]["accuracy_random"] = accuracy_random
        results[dataset_name]["masked_percentage"] = masked_percentages

    results_path = output_path / f"collected_masking_results.npy"
    np.save(results_path, results, allow_pickle=True)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
