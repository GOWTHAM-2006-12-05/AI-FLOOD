"""
lstm_model.py — LSTM-based rainfall time-series forecasting for flood prediction.

Responsibilities:
    • Convert hourly rainfall sequences into sliding-window tensors
    • Build a configurable LSTM architecture (stacked layers + dropout)
    • Train with early stopping and learning-rate scheduling
    • Forecast future rainfall accumulation probabilities
    • Export P(flood) from the temporal pathway

Architecture:
    Input  → (batch, seq_len, n_features) — e.g. (N, 24, 6) for 24h window
    Layer1 → LSTM(64, return_sequences=True) + Dropout
    Layer2 → LSTM(32) + Dropout
    Dense  → Dense(16, relu) → Dense(1, sigmoid)
    Output → P(flood) ∈ [0, 1]

This module uses TensorFlow/Keras for the LSTM.
If TensorFlow is unavailable, a lightweight numpy-only fallback is provided
that uses a simple exponential-weighted moving average (for demo/testing).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Suppress TF warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default features to include in the time-series window
SEQUENCE_FEATURES: List[str] = [
    "rain_1hr",
    "soil_moisture",
    "surface_pressure",
    "wind_speed_10m",
    "temperature_2m",
    "relative_humidity",
]

DEFAULT_SEQ_LEN = 24       # 24 hours lookback
DEFAULT_LSTM_UNITS = [64, 32]
DEFAULT_DROPOUT = 0.3
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 0.001


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def create_sequences(
    data: np.ndarray,
    labels: np.ndarray,
    seq_len: int = DEFAULT_SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 2-D time-series data into sliding-window 3-D tensors.

    Parameters
    ----------
    data : np.ndarray, shape (n_timesteps, n_features)
        Chronologically ordered feature matrix.
    labels : np.ndarray, shape (n_timesteps,)
        Binary flood labels for each timestep.
    seq_len : int
        Number of past timesteps per sample.

    Returns
    -------
    X : np.ndarray, shape (n_samples, seq_len, n_features)
    y : np.ndarray, shape (n_samples,)
        Label at the END of each window.
    """
    X_seq, y_seq = [], []
    for i in range(seq_len, len(data)):
        X_seq.append(data[i - seq_len : i])
        y_seq.append(labels[i])
    return np.array(X_seq), np.array(y_seq)


def generate_synthetic_sequences(
    n_timesteps: int = 5000,
    n_features: int = 6,
    seq_len: int = DEFAULT_SEQ_LEN,
    flood_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic temporal data for LSTM training/testing.

    Returns (X_train_seq, y_train_seq, X_val_seq, y_val_seq).
    """
    rng = np.random.RandomState(seed)

    # Simulate hourly data
    rain = rng.exponential(3.0, n_timesteps).clip(0, 100)
    soil = rng.uniform(0.1, 0.9, n_timesteps)
    pressure = np.cumsum(rng.normal(0, 0.5, n_timesteps)) + 1013
    wind = rng.uniform(0, 30, n_timesteps)
    temp = 28 + np.sin(np.arange(n_timesteps) * 2 * np.pi / 24) * 5 + rng.normal(0, 1, n_timesteps)
    humidity = rng.uniform(50, 95, n_timesteps)

    data = np.column_stack([rain, soil, pressure, wind, temp, humidity])

    # Labels: flood when rolling 6h rain > threshold AND soil > 0.6
    rolling_rain = np.convolve(rain, np.ones(6), mode="same")
    flood_score = rolling_rain * soil - 10
    prob = 1.0 / (1.0 + np.exp(-flood_score * 0.1))
    labels = (rng.rand(n_timesteps) < prob).astype(int)

    # Normalise features (per-feature z-score)
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-8
    data_norm = (data - mean) / std

    # Create sequences
    X, y = create_sequences(data_norm, labels, seq_len)

    # Split 80/20
    split = int(0.8 * len(X))
    return X[:split], y[:split], X[split:], y[split:]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class LSTMTrainResult:
    """Container for LSTM training results."""

    model: Any  # tf.keras.Model or None
    accuracy: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    val_loss: float = 0.0
    history: Dict[str, List[float]] = field(default_factory=dict)
    epochs_trained: int = 0
    using_fallback: bool = False

    def summary(self) -> Dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
            "val_loss": round(self.val_loss, 4),
            "epochs_trained": self.epochs_trained,
            "using_fallback": self.using_fallback,
        }


# ---------------------------------------------------------------------------
# Keras LSTM builder
# ---------------------------------------------------------------------------


def _build_keras_lstm(
    seq_len: int,
    n_features: int,
    lstm_units: List[int] = DEFAULT_LSTM_UNITS,
    dropout: float = DEFAULT_DROPOUT,
    learning_rate: float = DEFAULT_LR,
) -> Any:
    """
    Build a Keras LSTM model for binary classification.

    Architecture:
        Input(seq_len, n_features)
        → LSTM(64, return_sequences=True) → Dropout(0.3)
        → LSTM(32) → Dropout(0.3)
        → Dense(16, relu) → Dropout(0.2)
        → Dense(1, sigmoid)
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(name="flood_lstm")

    # First LSTM layer
    model.add(
        layers.LSTM(
            lstm_units[0],
            return_sequences=len(lstm_units) > 1,
            input_shape=(seq_len, n_features),
            kernel_regularizer=keras.regularizers.l2(1e-4),
        )
    )
    model.add(layers.Dropout(dropout))

    # Additional LSTM layers
    for i, units in enumerate(lstm_units[1:], 1):
        return_seq = i < len(lstm_units) - 1
        model.add(
            layers.LSTM(
                units,
                return_sequences=return_seq,
                kernel_regularizer=keras.regularizers.l2(1e-4),
            )
        )
        model.add(layers.Dropout(dropout))

    # Dense head
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dropout(dropout * 0.6))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    lstm_units: List[int] = DEFAULT_LSTM_UNITS,
    dropout: float = DEFAULT_DROPOUT,
    learning_rate: float = DEFAULT_LR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    patience: int = 10,
) -> LSTMTrainResult:
    """
    Train the LSTM model with early stopping and LR scheduling.

    Parameters
    ----------
    X_train : shape (n_samples, seq_len, n_features)
    y_train : shape (n_samples,)
    X_val, y_val : validation data
    lstm_units : list of int
        Units per LSTM layer.
    dropout : float
        Dropout rate.
    learning_rate : float
        Initial learning rate.
    epochs : int
        Max epochs.
    batch_size : int
        Mini-batch size.
    patience : int
        Early-stopping patience.

    Returns
    -------
    LSTMTrainResult
    """
    try:
        import tensorflow as tf
        from tensorflow import keras

        HAS_TF = True
    except ImportError:
        HAS_TF = False

    if not HAS_TF:
        logger.warning(
            "TensorFlow not installed — using fallback EWMA predictor. "
            "Install with: pip install tensorflow"
        )
        return _train_fallback(X_train, y_train, X_val, y_val)

    seq_len = X_train.shape[1]
    n_features = X_train.shape[2]

    model = _build_keras_lstm(
        seq_len=seq_len,
        n_features=n_features,
        lstm_units=lstm_units,
        dropout=dropout,
        learning_rate=learning_rate,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # Handle class imbalance via sample weights
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    if n_pos > 0:
        weight_for_0 = 1.0
        weight_for_1 = n_neg / n_pos
        class_weight = {0: weight_for_0, 1: weight_for_1}
    else:
        class_weight = None

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=0,
    )

    # Evaluate
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    y_prob = model.predict(X_val, verbose=0).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_val, y_prob)
    except ValueError:
        auc = 0.0

    val_loss = min(history.history.get("val_loss", [0.0]))
    epochs_trained = len(history.history.get("loss", []))

    logger.info(
        "LSTM trained — acc=%.4f  f1=%.4f  auc=%.4f  epochs=%d",
        acc, f1, auc, epochs_trained,
    )

    return LSTMTrainResult(
        model=model,
        accuracy=acc,
        f1=f1,
        roc_auc=auc,
        val_loss=val_loss,
        history={k: [float(v) for v in vals] for k, vals in history.history.items()},
        epochs_trained=epochs_trained,
        using_fallback=False,
    )


# ---------------------------------------------------------------------------
# Fallback predictor (no TensorFlow)
# ---------------------------------------------------------------------------


class _EWMAPredictor:
    """
    Lightweight exponential-weighted moving average predictor.

    Used when TensorFlow is not available. It simply looks at the
    mean rain intensity in the sequence window and applies a sigmoid.
    NOT production-quality — placeholder only.
    """

    def __init__(self, threshold: float = 0.5, alpha: float = 0.3):
        self.threshold = threshold
        self.alpha = alpha
        self._rain_mean = 0.0
        self._rain_std = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # X shape: (n_samples, seq_len, n_features)
        # Assume feature 0 is rainfall
        rain_sums = X[:, :, 0].sum(axis=1)
        self._rain_mean = rain_sums.mean()
        self._rain_std = rain_sums.std() + 1e-8

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        rain_sums = X[:, :, 0].sum(axis=1)
        z = (rain_sums - self._rain_mean) / self._rain_std
        prob = 1.0 / (1.0 + np.exp(-z * self.alpha))
        return prob

    def predict(self, X: np.ndarray, verbose: int = 0) -> np.ndarray:
        """Keras-like predict interface."""
        return self.predict_proba(X).reshape(-1, 1)


def _train_fallback(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> LSTMTrainResult:
    """Train fallback EWMA predictor."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    predictor = _EWMAPredictor()
    predictor.fit(X_train, y_train)

    y_prob = predictor.predict_proba(X_val)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_val, y_prob)
    except ValueError:
        auc = 0.0

    return LSTMTrainResult(
        model=predictor,
        accuracy=acc,
        f1=f1,
        roc_auc=auc,
        val_loss=0.0,
        history={},
        epochs_trained=0,
        using_fallback=True,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_flood_proba_lstm(
    model: Any,
    X: np.ndarray,
) -> np.ndarray:
    """
    Get P(flood) from the LSTM model.

    Parameters
    ----------
    model : keras.Model or _EWMAPredictor
    X : shape (n_samples, seq_len, n_features)

    Returns
    -------
    np.ndarray, shape (n_samples,) with values in [0, 1]
    """
    preds = model.predict(X, verbose=0)
    return preds.flatten()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_lstm(model: Any, path: str | Path, is_fallback: bool = False) -> None:
    """Save model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if is_fallback:
        import joblib
        joblib.dump(model, str(path))
    else:
        model.save(str(path))
    logger.info("LSTM model saved → %s", path)


def load_lstm(path: str | Path, is_fallback: bool = False) -> Any:
    """Load a previously saved LSTM."""
    if is_fallback:
        import joblib
        return joblib.load(str(path))
    else:
        import tensorflow as tf
        return tf.keras.models.load_model(str(path))
