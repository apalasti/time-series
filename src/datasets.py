from pathlib import Path
from typing import Dict

import torch
from torch import Tensor
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(self, dataset_path, apply=None, map_location=None) -> None:
        super().__init__()

        self.apply = apply
        self.dataset_path = Path(dataset_path)
        if self.dataset_path.suffix != ".pt":
            raise ValueError(f"Dataset file must have .pt extension, got {self.dataset_path.suffix}")

        self.dataset: Dict[str, Tensor] = torch.load(self.dataset_path, map_location)

        # The samples is of shape: (N, T, C) where:
        #   - N = Number of samples
        #   - T = Time steps (sequence length)
        #   - C = Channels/features per time step
        self.samples = self.dataset["samples"].float()
        self.labels = self.dataset["labels"].float()
        assert self.samples.shape[0] == self.labels.shape[0], (
            f"Number of samples ({self.samples.shape[0]}) does not match number of labels "
            f"({self.labels.shape[0]}) in dataset {self.dataset_path}"
        )

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        if self.apply is not None:
            return self.apply(self.samples[index], self.labels[index])

        return self.samples[index], self.labels[index]


if __name__ == "__main__":
    dataset = ClassificationDataset(
        dataset_path=Path(__file__).parent.parent / "datasets/classification/WISDM/train.pt"
    )
    for i in range(2):
        print(dataset[i])
