import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange


class LinearReconstructor(RegressorMixin, BaseEstimator):
    def __init__(self, dropout_rate=0.1, learning_rate=1e-3, weight_decay=0.0,
                 epochs=100, batch_size=32, device="cpu", random_state=None):
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
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
        optimizer = optim.AdamW(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        self.model_.train()
        self.history_ = {"loss": [], "cosine_similarity": [], "lr": []}

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
            self.history_["lr"].append(optimizer.param_groups[0]['lr'])

            scheduler.step()

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)
        with torch.inference_mode():
            predictions = self.model_(X_tensor)

        return predictions.cpu().numpy()


class LinearClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, dropout_rate=0.0, learning_rate=1e-3, weight_decay=0.0,
                 epochs=100, batch_size=32, device="cpu", random_state=None):
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available() and self.device == "cuda":
                torch.cuda.manual_seed_all(self.random_state)

        self._device = torch.device(self.device)

    def _build_model(self, input_dim, output_dim):
        """Builds the neural network model."""
        self.model_ = nn.Sequential(
            nn.Dropout1d(self.dropout_rate),
            nn.Flatten(start_dim=1),
            nn.Linear(input_dim, output_dim),
        ).to(self._device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)

        label_map = {label: i for i, label in enumerate(self.classes_)}
        y_mapped = np.array([label_map[label] for label in y])

        if X.ndim == 2:
            n_samples, n_features = X.shape
            n_timesteps = 1
        else:
            n_samples, n_timesteps, n_features = X.shape
        if y_mapped.shape != (n_samples,):
            raise ValueError(
                 f"Expected y to have shape ({n_samples},), but got {y.shape} after mapping"
             )

        # Build model based on input dimensions and number of classes
        self.input_features = n_timesteps * n_features
        self._build_model(n_timesteps * n_features, self.n_classes_)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)
        y_tensor = torch.tensor(y_mapped, dtype=torch.long).to(self._device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Use CrossEntropyLoss for classification
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        self.model_.train()
        self.history_ = {"loss": [], "accuracy": [], "lr": []}

        for _ in trange(
            0, self.epochs,
            desc="Training LinearClassifier",
            unit="epoch",
            leave=True,
        ):
            epoch_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                # Get model outputs (logits)
                outputs = self.model_(batch_X)
                # Calculate loss
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_X.size(0)
                # Calculate accuracy
                _, predicted_labels = torch.max(outputs.detach(), 1)
                correct_predictions += (predicted_labels == batch_y).sum().item()
                total_samples += batch_y.size(0)

            avg_epoch_loss = epoch_loss / total_samples
            epoch_accuracy = correct_predictions / total_samples
            self.history_["loss"].append(avg_epoch_loss)
            self.history_["accuracy"].append(epoch_accuracy)
            self.history_["lr"].append(optimizer.param_groups[0]["lr"])

            scheduler.step()

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)
        with torch.inference_mode():
            logits = self.model_(X_tensor)
            probabilities = F.softmax(logits, dim=1)

        return probabilities.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        indices = np.argmax(probabilities, axis=1)
        return self.classes_[indices]

    def state_dict(self):
        if not self.is_fitted_:
            raise RuntimeError("Cannot save unfitted model. Call fit() first.")

        return {
            "model_state_dict": self.model_.state_dict(),
            "init_params": {
                "dropout_rate": self.dropout_rate,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "device": self.device,
                "random_state": self.random_state,
            },
            "input_features": self.input_features,
            "history": self.history_,
            "classes_": self.classes_,
            "n_classes_": self.n_classes_,
        }

    @classmethod
    def load_state_dict(cls, checkpoint):
        model = cls(**checkpoint["init_params"])
        model._build_model(checkpoint["input_features"], checkpoint['n_classes_'])

        model.model_.load_state_dict(checkpoint["model_state_dict"])
        model.model_.eval() # Set model to evaluation mode

        model.history_ = checkpoint["history"]
        model.classes_ = checkpoint["classes_"]
        model.n_classes_ = checkpoint["n_classes_"]
        model.is_fitted_ = True

        return model
