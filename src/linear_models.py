import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange


class LinearReconstructor(RegressorMixin, BaseEstimator):
    def __init__(self, dropout_rate=0.1, learning_rate=1e-3,
                 epochs=100, batch_size=32, device="cpu", random_state=None):
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        self._device = torch.device(self.device)

    def _build_model(self, input_dim, output_dim):
        self.model_ = nn.Sequential(
            nn.Dropout1d(self.dropout_rate),
            nn.Flatten(start_dim=1),
            nn.Linear(input_dim, output_dim),
        ).to(self._device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_timesteps, n_features = X.shape
        if y.shape != (n_samples, n_features):
            raise ValueError(
                f"Expected y to have shape ({n_samples}, {n_features}), but got {y.shape}"
            )

        self._build_model(n_timesteps * n_features, n_features)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self._device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        self.model_.train()
        self.history_ = {"loss": [], "cosine_similarity": []}

        for _ in trange(
            0, self.epochs,
            desc="Training LinearReconstructor",
            unit="epoch",
            leave=True,
        ):
            epoch_loss = 0.0
            epoch_sim = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)
                epoch_sim += F.cosine_similarity(
                    outputs.detach(), batch_y.detach()
                ).sum().item()

            avg_epoch_loss = epoch_loss / len(dataset)
            avg_epoch_sim = epoch_sim / len(dataset)
            self.history_["loss"].append(avg_epoch_loss)
            self.history_["cosine_similarity"].append(avg_epoch_sim)

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)
        with torch.inference_mode():
            predictions = self.model_(X_tensor)

        return predictions.cpu().numpy()
