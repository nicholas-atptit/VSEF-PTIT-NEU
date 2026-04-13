"""PyTorch LSTM models for time-series classification and regression."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseModel


class _RecurrentNet(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_dim: int,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        direction_factor = 2 if bidirectional else 1
        self.head = nn.Linear(hidden_size * direction_factor, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.rnn(x)
        return self.head(outputs[:, -1, :])


class LstmModel(BaseModel):
    """Trainable LSTM model used by the unified ML registry."""

    algorithm_name = "lstm"
    bidirectional = False

    def __init__(
        self,
        *,
        task: str = "classification",
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 30,
        patience: int = 5,
        random_state: int = 42,
        **_: Any,
    ) -> None:
        self.task = task
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler: StandardScaler | None = None
        self.input_dim: int | None = None
        self.output_dim: int | None = None
        self.classes_: np.ndarray | None = None
        self.model: _RecurrentNet | None = None
        self.params = {
            "task": task,
            "sequence_length": sequence_length,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "random_state": random_state,
        }

    def _build_model(self) -> None:
        if self.input_dim is None or self.output_dim is None:
            raise RuntimeError("input_dim and output_dim must be set before building the LSTM")
        self.model = _RecurrentNet(
            input_dim=self.input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_dim=self.output_dim,
            bidirectional=self.bidirectional,
        ).to(self.device)

    def _check_3d(self, X: np.ndarray) -> np.ndarray:
        array = np.asarray(X, dtype=float)
        if array.ndim != 3:
            raise ValueError(
                f"{self.algorithm_name} expects a 3D array of shape "
                f"(samples, sequence_length, features); got {array.shape}"
            )
        return array

    def _fit_label_encoder(self, y_train: np.ndarray) -> np.ndarray:
        if self.task == "regression":
            self.classes_ = None
            self.output_dim = 1
            return np.asarray(y_train, dtype=np.float32)
        self.classes_ = np.unique(y_train)
        self.output_dim = int(len(self.classes_))
        inverse = np.searchsorted(self.classes_, y_train)
        return inverse.astype(np.int64)

    def _transform_labels(self, y: np.ndarray) -> np.ndarray:
        if self.task == "regression":
            return np.asarray(y, dtype=np.float32)
        if self.classes_ is None:
            raise RuntimeError("Classification labels are not initialized")
        return np.searchsorted(self.classes_, y).astype(np.int64)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("Scaler is not fitted")
        reshaped = X.reshape(-1, X.shape[2])
        scaled = self.scaler.transform(reshaped)
        return scaled.reshape(X.shape)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_train = self._check_3d(X_train)
        self.input_dim = int(X_train.shape[2])
        y_train_encoded = self._fit_label_encoder(np.asarray(y_train))
        self._build_model()

        self.scaler = StandardScaler()
        self.scaler.fit(X_train.reshape(-1, self.input_dim))
        X_train_scaled = self._scale(X_train)

        X_val_scaled = None
        y_val_encoded = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            X_val = self._check_3d(X_val)
            X_val_scaled = self._scale(X_val)
            y_val_encoded = self._transform_labels(np.asarray(y_val))

        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train_encoded),
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        val_loader = None
        if X_val_scaled is not None and y_val_encoded is not None:
            val_dataset = TensorDataset(
                torch.tensor(X_val_scaled, dtype=torch.float32),
                torch.tensor(y_val_encoded),
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        criterion: nn.Module
        if self.task == "regression":
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_state = copy.deepcopy(self.model.state_dict())
        best_val_loss = float("inf")
        patience_counter = 0

        for _ in range(self.epochs):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                if self.task == "regression":
                    loss = criterion(preds.squeeze(-1), yb.float())
                else:
                    loss = criterion(preds, yb.long())
                loss.backward()
                optimizer.step()

            if val_loader is None:
                continue

            self.model.eval()
            losses: list[float] = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    preds = self.model(xb)
                    if self.task == "regression":
                        loss = criterion(preds.squeeze(-1), yb.float())
                    else:
                        loss = criterion(preds, yb.long())
                    losses.append(float(loss.item()))

            mean_val_loss = float(np.mean(losses)) if losses else float("inf")
            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        self.model.load_state_dict(best_state)
        self.model.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._check_3d(X)
        probabilities = self.predict_proba(X) if self.task == "classification" else None
        if probabilities is not None:
            if self.classes_ is None:
                raise RuntimeError("Classification labels are not initialized")
            return self.classes_[np.argmax(probabilities, axis=1)]

        if self.model is None:
            raise RuntimeError("Model is not loaded")
        X_scaled = self._scale(X)
        with torch.no_grad():
            outputs = self.model(torch.tensor(X_scaled, dtype=torch.float32, device=self.device))
        return outputs.squeeze(-1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise NotImplementedError("Regression LSTM models do not expose predict_proba")
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        X = self._check_3d(X)
        X_scaled = self._scale(X)
        with torch.no_grad():
            outputs = self.model(torch.tensor(X_scaled, dtype=torch.float32, device=self.device))
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()

    @classmethod
    def get_model_capabilities(cls) -> Dict[str, Any]:
        return {
            "algorithm": cls.algorithm_name,
            "model_family": "deep_learning",
            "requires_sequence_data": True,
            "supports_exogenous_features": True,
            "artifact_type": "torch",
        }

    def save(self, artifact_path: Path) -> None:
        if self.model is None or self.scaler is None or self.input_dim is None or self.output_dim is None:
            raise RuntimeError("Cannot save an untrained LSTM model")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), artifact_path)
        joblib.dump(self.scaler, artifact_path.with_suffix(".scaler.joblib"))
        joblib.dump(self.get_artifact_metadata(), artifact_path.with_suffix(".meta.joblib"))

    @classmethod
    def load(cls, artifact_path: Path) -> "LstmModel":
        metadata: Dict[str, Any] = joblib.load(artifact_path.with_suffix(".meta.joblib"))
        instance = cls(**metadata["params"])
        instance.input_dim = metadata["input_dim"]
        instance.output_dim = metadata["output_dim"]
        classes = metadata.get("classes")
        if classes is not None:
            instance.classes_ = np.asarray(classes)
        instance.scaler = joblib.load(artifact_path.with_suffix(".scaler.joblib"))
        instance._build_model()
        state_dict = torch.load(artifact_path, map_location=instance.device)
        instance.model.load_state_dict(state_dict)
        instance.model.eval()
        return instance

    def get_artifact_metadata(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm_name,
            "task": self.task,
            "params": self.params,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "classes": None if self.classes_ is None else self.classes_.tolist(),
        }
