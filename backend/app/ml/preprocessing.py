"""
preprocessing.py — Data preprocessing pipeline for flood prediction.

Handles:
    • Feature construction from weather + terrain data
    • Train / validation / test splitting (stratified)
    • SMOTE class-balancing for the minority class
    • Robust feature scaling
    • k-fold cross-validation generator
    • Drift detection helpers (PSI — Population Stability Index)

Target label
    flood_occurred : 1 = flood event, 0 = no flood

Feature vector (8 core + extended):
    Rain_1hr, Rain_3hr, Rain_6hr, Rain_24hr,
    Elevation, Soil_moisture, Drainage_capacity, Urbanization

All public functions return plain numpy arrays or pandas DataFrames so they
remain framework-agnostic (work with XGBoost, scikit-learn, Keras, etc.).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CORE_FEATURES: List[str] = [
    "rain_1hr",
    "rain_3hr",
    "rain_6hr",
    "rain_24hr",
    "elevation",
    "soil_moisture",
    "drainage_capacity",
    "urbanization",
]

EXTENDED_FEATURES: List[str] = [
    # From weather_features.py composite signals
    "rain_x_soil",
    "wind_x_pressure_deficit",
    "flood_compound",
    "pressure_drop_3hr",
    "pressure_drop_6hr",
    "rain_intensity_ratio",
    "temperature_2m",
    "wind_speed_10m",
    "surface_pressure",
    "relative_humidity",
    "cloud_cover",
    "monsoon_flag",
]

ALL_FEATURES: List[str] = CORE_FEATURES + EXTENDED_FEATURES

TARGET_COL = "flood_occurred"

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class FloodDataset:
    """Immutable container returned by the preprocessing pipeline."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    scaler: RobustScaler
    smote_applied: bool = False
    original_class_dist: Dict[int, int] = field(default_factory=dict)
    resampled_class_dist: Dict[int, int] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "train_shape": self.X_train.shape,
            "val_shape": self.X_val.shape,
            "test_shape": self.X_test.shape,
            "n_features": len(self.feature_names),
            "smote_applied": self.smote_applied,
            "original_class_dist": self.original_class_dist,
            "resampled_class_dist": self.resampled_class_dist,
        }


# ---------------------------------------------------------------------------
# Synthetic data generator (for demo / bootstrapping)
# ---------------------------------------------------------------------------


def generate_synthetic_flood_data(
    n_samples: int = 5000,
    flood_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate realistic synthetic flood data for training / testing.

    The generator encodes domain knowledge:
        • Higher rainfall → higher flood probability
        • Low elevation + high soil moisture → flood-prone
        • High urbanization + poor drainage → flood-prone
        • Monsoon months multiply flood likelihood

    Parameters
    ----------
    n_samples : int
        Total number of samples to generate.
    flood_ratio : float
        Approximate fraction of flood=1 labels.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with all feature columns + 'flood_occurred' target.
    """
    rng = np.random.RandomState(seed)

    # ----- Base features -----
    rain_1hr = rng.exponential(scale=5.0, size=n_samples).clip(0, 120)
    rain_3hr = rain_1hr * rng.uniform(1.5, 3.5, n_samples)
    rain_6hr = rain_3hr * rng.uniform(1.2, 2.5, n_samples)
    rain_24hr = rain_6hr * rng.uniform(1.3, 3.0, n_samples)

    elevation = rng.uniform(0, 500, n_samples)  # metres above sea level
    soil_moisture = rng.uniform(0.05, 0.95, n_samples)  # fraction
    drainage_capacity = rng.uniform(0.1, 1.0, n_samples)  # 0=poor, 1=good
    urbanization = rng.uniform(0.0, 1.0, n_samples)  # 0=rural, 1=dense

    # Extended features
    rain_x_soil = rain_24hr * soil_moisture
    wind_speed = rng.uniform(0, 40, n_samples)
    pressure = rng.normal(1013, 8, n_samples)
    pressure_deficit = np.clip(1013.25 - pressure, 0, None)
    wind_x_pressure = wind_speed * pressure_deficit
    flood_compound = (rain_24hr * soil_moisture) / (drainage_capacity + 0.01)
    pressure_drop_3hr = rng.normal(0, 3, n_samples)
    pressure_drop_6hr = pressure_drop_3hr * rng.uniform(1.0, 2.0, n_samples)
    rain_intensity = np.where(rain_24hr > 0, rain_1hr / rain_24hr, 0)
    temperature = rng.normal(28, 5, n_samples)
    humidity = rng.uniform(40, 100, n_samples)
    cloud_cover = rng.uniform(0, 100, n_samples)
    monsoon = rng.choice([0, 1], n_samples, p=[0.5, 0.5])

    # ----- Flood probability (logistic model with domain priors) -----
    logit = (
        -4.0
        + 0.04 * rain_1hr
        + 0.02 * rain_3hr
        + 0.015 * rain_6hr
        + 0.008 * rain_24hr
        - 0.005 * elevation
        + 2.0 * soil_moisture
        - 1.5 * drainage_capacity
        + 1.2 * urbanization
        + 0.8 * monsoon
        + 0.001 * flood_compound
        + 0.05 * pressure_drop_3hr
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    flood_occurred = (rng.rand(n_samples) < prob).astype(int)

    # Build DataFrame
    df = pd.DataFrame(
        {
            "rain_1hr": rain_1hr,
            "rain_3hr": rain_3hr,
            "rain_6hr": rain_6hr,
            "rain_24hr": rain_24hr,
            "elevation": elevation,
            "soil_moisture": soil_moisture,
            "drainage_capacity": drainage_capacity,
            "urbanization": urbanization,
            "rain_x_soil": rain_x_soil,
            "wind_x_pressure_deficit": wind_x_pressure,
            "flood_compound": flood_compound,
            "pressure_drop_3hr": pressure_drop_3hr,
            "pressure_drop_6hr": pressure_drop_6hr,
            "rain_intensity_ratio": rain_intensity,
            "temperature_2m": temperature,
            "wind_speed_10m": wind_speed,
            "surface_pressure": pressure,
            "relative_humidity": humidity,
            "cloud_cover": cloud_cover,
            "monsoon_flag": monsoon.astype(float),
            TARGET_COL: flood_occurred,
        }
    )

    actual_ratio = flood_occurred.mean()
    logger.info(
        "Generated %d synthetic samples (flood ratio=%.2f%%)",
        n_samples,
        actual_ratio * 100,
    )
    return df


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess(
    df: pd.DataFrame,
    *,
    feature_cols: Optional[List[str]] = None,
    target_col: str = TARGET_COL,
    test_size: float = 0.15,
    val_size: float = 0.15,
    apply_smote: bool = True,
    smote_strategy: str = "auto",
    random_state: int = 42,
) -> FloodDataset:
    """
    Full preprocessing pipeline.

    Steps:
        1. Select features present in both df columns and feature_cols
        2. Handle missing values (median imputation)
        3. Stratified train / val / test split (70 / 15 / 15 default)
        4. Fit RobustScaler on train only → transform all sets
        5. Apply SMOTE on train only (never on val/test)
        6. Package into FloodDataset

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with feature columns and target column.
    feature_cols : list of str or None
        Features to use. Defaults to ALL_FEATURES filtered to df columns.
    target_col : str
        Name of the binary target column.
    test_size : float
        Fraction for test set.
    val_size : float
        Fraction for validation set (computed from remaining after test).
    apply_smote : bool
        Whether to apply SMOTE to balance training data.
    smote_strategy : str
        SMOTE sampling_strategy (default 'auto' → balance to majority).
    random_state : int
        Global random seed for reproducibility.

    Returns
    -------
    FloodDataset
        Ready-to-train dataset container.
    """
    # 1. Feature selection
    if feature_cols is None:
        feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.warning("Dropping missing columns: %s", missing_cols)
        feature_cols = [c for c in feature_cols if c not in missing_cols]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in DataFrame")

    X = df[feature_cols].copy()
    y = df[target_col].values.astype(int)

    # 2. Impute missing values — median (robust to outliers)
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            logger.info("Imputed %d NaN in '%s' with median=%.4f",
                        X[col].isna().sum(), col, median_val)

    # Replace infinities
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0.0, inplace=True)

    # 3. Stratified split: train → (train + val), test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X.values, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # From the remaining data, split again for validation
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=relative_val_size,
        stratify=y_temp,
        random_state=random_state,
    )

    original_dist = {
        int(k): int(v)
        for k, v in zip(*np.unique(y_train, return_counts=True))
    }

    # 4. Scale — fit on train only
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # 5. SMOTE — on train only
    smote_applied = False
    resampled_dist = original_dist.copy()
    if apply_smote:
        try:
            from imblearn.over_sampling import SMOTE

            sm = SMOTE(
                sampling_strategy=smote_strategy,
                random_state=random_state,
                k_neighbors=min(5, original_dist.get(1, 1) - 1)
                if original_dist.get(1, 1) > 1
                else 1,
            )
            X_train, y_train = sm.fit_resample(X_train, y_train)
            smote_applied = True
            resampled_dist = {
                int(k): int(v)
                for k, v in zip(*np.unique(y_train, return_counts=True))
            }
            logger.info(
                "SMOTE applied: %s → %s", original_dist, resampled_dist
            )
        except ImportError:
            logger.warning(
                "imbalanced-learn not installed — skipping SMOTE. "
                "Install with: pip install imbalanced-learn"
            )

    # 6. Package
    return FloodDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_cols,
        scaler=scaler,
        smote_applied=smote_applied,
        original_class_dist=original_dist,
        resampled_class_dist=resampled_dist,
    )


# ---------------------------------------------------------------------------
# Cross-validation helper
# ---------------------------------------------------------------------------


def create_cv_folds(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate stratified k-fold index pairs.

    Returns
    -------
    list of (train_idx, val_idx) tuples
    """
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )
    return list(skf.split(X, y))


# ---------------------------------------------------------------------------
# Drift detection — Population Stability Index (PSI)
# ---------------------------------------------------------------------------


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """
    Population Stability Index between two 1-D distributions.

    PSI < 0.10  → no significant shift
    PSI 0.10–0.25 → moderate shift, monitor
    PSI > 0.25  → significant drift, retrain model

    Parameters
    ----------
    reference : np.ndarray
        Training-time feature distribution.
    current : np.ndarray
        Inference-time feature distribution.
    n_bins : int
        Number of equal-frequency bins.
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    float
        PSI value ≥ 0.
    """
    # Use reference quantiles to define bins
    breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts = np.histogram(reference, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(current, bins=breakpoints)[0].astype(float)

    ref_pct = ref_counts / ref_counts.sum() + eps
    cur_pct = cur_counts / cur_counts.sum() + eps

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def check_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    threshold: float = 0.25,
) -> Dict[str, Dict[str, Any]]:
    """
    Check PSI-based drift for each feature.

    Returns
    -------
    dict
        {feature_name: {"psi": float, "drifted": bool, "severity": str}}
    """
    if feature_cols is None:
        feature_cols = [
            c for c in reference_df.columns if c in ALL_FEATURES
        ]

    results: Dict[str, Dict[str, Any]] = {}
    for col in feature_cols:
        if col not in current_df.columns:
            continue
        psi = compute_psi(
            reference_df[col].dropna().values,
            current_df[col].dropna().values,
        )
        if psi < 0.10:
            severity = "none"
        elif psi < threshold:
            severity = "moderate"
        else:
            severity = "significant"

        results[col] = {
            "psi": round(psi, 6),
            "drifted": psi >= threshold,
            "severity": severity,
        }

    return results


# ---------------------------------------------------------------------------
# Scaler persistence
# ---------------------------------------------------------------------------


def save_scaler(scaler: RobustScaler, path: str | Path) -> None:
    """Save the fitted scaler to disk via joblib."""
    import joblib

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, str(path))
    logger.info("Scaler saved to %s", path)


def load_scaler(path: str | Path) -> RobustScaler:
    """Load a previously saved scaler."""
    import joblib

    return joblib.load(str(path))
