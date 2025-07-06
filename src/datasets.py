from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

FORECASTING_SPLIT_CONFIG = {
    "ETTh1": [0, 12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24],
    "ETTh2": [0, 12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24],
    "ETTm1": [0, 12 * 30 * 24 * 4, 16 * 30 * 24 * 4, 20 * 30 * 24 * 4],
    "ETTm2": [0, 12 * 30 * 24 * 4, 16 * 30 * 24 * 4, 20 * 30 * 24 * 4],
    "exchange_rate": [0, 5311, 6071, 7588],
    "weather": [0, 36_887, 42_157, 52_696],
}


def load_forecasting_dataset(dataset_path, split: str, sequence_len: int, prediction_len: int, transform=None):
    dataset_name = Path(dataset_path).stem

    idx = {"train": 1, "validation": 2, "test": 3}[split]
    start = max(0, FORECASTING_SPLIT_CONFIG[dataset_name][idx - 1] - sequence_len)
    end = FORECASTING_SPLIT_CONFIG[dataset_name][idx]

    return ForecastingDataset(
        dataset_path, sequence_len, prediction_len, start, end, transform
    )


class ClassificationDataset(Dataset):
    def __init__(self, dataset_path, transform=None, map_location=None) -> None:
        super().__init__()

        self.transform = transform
        self.dataset_path = Path(dataset_path)
        if self.dataset_path.suffix != ".pt":
            raise ValueError(f"Dataset file must have .pt extension, got {self.dataset_path.suffix}")

        self.dataset: Dict[str, Tensor] = torch.load(self.dataset_path, map_location, weights_only=True)

        # The samples is of shape: (N, T, C) where:
        #   - N = Number of samples
        #   - T = Time steps (sequence length)
        #   - C = Channels/features per time step
        self.samples = self.dataset["samples"]
        self.labels = self.dataset["labels"].long()
        assert self.samples.shape[0] == self.labels.shape[0], (
            f"Number of samples ({self.samples.shape[0]}) does not match number of labels "
            f"({self.labels.shape[0]}) in dataset {self.dataset_path}"
        )

    def mean(self):
        return self.samples.mean(dim=(0, 1))

    def std(self):
        return self.samples.std(dim=(0, 1))

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        if self.transform is not None:
            return self.transform(self.samples[index], self.labels[index])

        return self.samples[index].float(), self.labels[index].long()


class ForecastingDataset(Dataset):

    def __init__(
        self,
        dataset_path,
        sequence_len: int,
        prediction_len: int,
        start: Optional[int] = None,
        end: Optional[int] = None,
        transform=None,
    ):
        super().__init__()

        self.transform = transform
        self.dataset_path = Path(dataset_path)
        if self.dataset_path.suffix != ".csv":
            raise ValueError(f"Dataset file must have .csv extension, got {self.dataset_path.suffix}")

        df = pd.read_csv(self.dataset_path, index_col=0, parse_dates=True, dtype=float)
        if start is not None and end is not None:
            df = df.iloc[start:end]
        elif start is not None:
            df = df.iloc[start:]
        elif end is not None:
            df = df.iloc[:end]

        self.timestamps = df.index.values
        self.time_series = torch.from_numpy(df.values)

        assert 0 < sequence_len and 0 < prediction_len
        assert sequence_len + prediction_len <= len(df)

        self.sequence_len = sequence_len
        self.prediction_len = prediction_len

    def mean(self):
        return self.time_series.mean(dim=(0, 1))

    def std(self):
        return self.time_series.std(dim=(0, 1))

    def __len__(self):
        return self.time_series.shape[0] - (self.sequence_len + self.prediction_len) + 1

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index, return_timestamps = index
        else:
            return_timestamps = False

        start, end = index, index + self.sequence_len
        x = self.time_series[start:end]
        y = self.time_series[end : end + self.prediction_len]

        x_timestamps = self.timestamps[start:end]
        y_timestamps = self.timestamps[end : end + self.prediction_len]

        return_value = (x, y, x_timestamps, y_timestamps) if return_timestamps else (x, y)

        if self.transform is not None:
            return self.transform(*return_value)
        return return_value


class SyntheticDataset(ClassificationDataset):
    def __init__(self, n_samples=1000, seq_length=128, n_channels=4, n_classes=4, seed=None):
        self.n_samples = n_samples
        self.seq_length = seq_length
        self.n_channels = n_channels
        self.n_classes = n_classes

        if seed is not None:
            np.random.seed(seed)
        
        # Initialize dataset parameters
        self.t = np.linspace(0, 4 * np.pi, self.seq_length)
        self.frequencies = np.random.uniform(0.1, 2.0, size=self.n_channels)
        self.interference_times = np.random.uniform(0, 4 * np.pi, size=(self.n_classes, self.n_channels))
        
        # Generate dataset
        samples, labels = self._generate_dataset()
        self.samples = torch.from_numpy(samples)
        self.labels = torch.from_numpy(labels).long()
        
    def _apply_interference(self, signal: np.ndarray, class_idx: int):
        """Apply class-specific interference to the signal."""
        time = np.linspace(0, 4 * np.pi, self.seq_length)
        center_time = self.interference_times[class_idx].reshape(-1, 1)
        DURATION, AMPLITUDE = 0.2, 1.5

        interference = (
            AMPLITUDE
            * np.exp(-((time - center_time) ** 2) / (2 * DURATION**2))
            * np.random.choice([-1, 1], size=center_time.shape)
        )
        interference = np.transpose(interference, (1, 0))
        return signal + interference

    def _generate_dataset(self):
        samples = np.zeros((self.n_samples, self.seq_length, self.n_channels))
        labels = np.zeros(self.n_samples, dtype=int)
        
        for i in range(self.n_samples):
            class_idx = np.random.randint(self.n_classes)
            labels[i] = class_idx

            for c in range(self.n_channels):
                phase_shift = np.random.uniform(0, 4 * np.pi)
                samples[i, :, c] = np.sin(self.frequencies[c] * self.t + phase_shift)
            samples[i] = self._apply_interference(samples[i], class_idx)
        
        # Add noise
        samples += 0.1 * np.random.normal(size=samples.shape)
        return samples, labels


if __name__ == "__main__":
    dataset = ClassificationDataset(
        dataset_path=Path(__file__).parent.parent / "datasets/classification/WISDM/train.pt"
    )
    for i in range(2):
        print(dataset[i])

    dataset = load_forecasting_dataset(
        dataset_path=Path(__file__).parent.parent / "datasets/forecasting/ETT-small/ETTh1.csv",
        split="train",
        sequence_len=4,
        prediction_len=2,
    )
    for i in range(2):
        print(dataset[i])
    print(f"Forecasting dataset length: {len(dataset)}")
