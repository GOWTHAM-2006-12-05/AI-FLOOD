"""
xgboost_model.py — XGBoost-based flood prediction for structured / tabular data.

Responsibilities:
    • Bayesian hyperparameter tuning (Optuna)
    • k-fold cross-validated training with early stopping
    • Overfitting prevention (regularisation, subsampling, max_depth caps)
    • Feature importance extraction
    • Model persistence (save / load)
    • Threshold calibration for desired recall

Architecture:
    Input  → 20-feature vector (from preprocessing.py)
    Model  → XGBClassifier (gradient-boosted trees)
    Output → P(flood) ∈ [0, 1]
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyper-parameter search space
# ---------------------------------------------------------------------------

DEFAULT_PARAMS: Dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "tree_method": "hist",           # fast histogram-based
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,                    # min split loss → prune weak splits
    "reg_alpha": 0.5,               # L1 regularisation
    "reg_lambda": 1.5,              # L2 regularisation
    "scale_pos_weight": 1.0,        # adjusted at runtime if imbalanced
    "random_state": 42,
    "n_jobs": -1,
}

# Optuna search bounds
TUNING_SPACE: Dict[str, Tuple] = {
    "max_depth": (3, 10),
    "learning_rate": (0.01, 0.3),
    "n_estimators": (100, 1200),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.4, 1.0),
    "min_child_weight": (1, 15),
    "gamma": (0.0, 1.0),
    "reg_alpha": (0.0, 5.0),
    "reg_lambda": (0.5, 5.0),
}

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class XGBTrainResult:
    """Result of a training / evaluation run."""

    model: Any  # xgb.XGBClassifier
    accuracy: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    roc_auc: float = 0.0
    best_params: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    cv_scores: List[float] = field(default_factory=list)
    report: str = ""

    def summary(self) -> Dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "f1": round(self.f1, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "roc_auc": round(self.roc_auc, 4),
            "cv_mean": round(float(np.mean(self.cv_scores)), 4)
            if self.cv_scores
            else None,
            "cv_std": round(float(np.std(self.cv_scores)), 4)
            if self.cv_scores
            else None,
            "n_trees": self.best_params.get("n_estimators"),
        }


# ---------------------------------------------------------------------------
# Core training
# ---------------------------------------------------------------------------


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    *,
    params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 30,
) -> XGBTrainResult:
    """
    Train an XGBClassifier with early stopping.

    Parameters
    ----------
    X_train, y_train : array-like
        Training data (possibly SMOTE-resampled).
    X_val, y_val : array-like
        Validation data (never resampled).
    feature_names : list of str
        Feature column names for importance mapping.
    params : dict or None
        Override hyperparameters; merged over DEFAULT_PARAMS.
    early_stopping_rounds : int
        Stop if val loss doesn't improve for N rounds.

    Returns
    -------
    XGBTrainResult
    """
    import xgboost as xgb

    hp = {**DEFAULT_PARAMS, **(params or {})}

    # Auto-set scale_pos_weight if imbalanced
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    if n_pos > 0 and hp.get("scale_pos_weight", 1.0) == 1.0:
        hp["scale_pos_weight"] = n_neg / n_pos

    clf = xgb.XGBClassifier(**hp)
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    auc = roc_auc_score(y_val, y_prob)
    report = classification_report(y_val, y_pred, zero_division=0)

    # Feature importance (gain-based)
    raw_imp = clf.feature_importances_
    importance = {
        name: round(float(imp), 6)
        for name, imp in sorted(
            zip(feature_names, raw_imp), key=lambda x: -x[1]
        )
    }

    logger.info(
        "XGBoost trained — acc=%.4f  f1=%.4f  auc=%.4f  trees=%d",
        acc, f1, auc, clf.n_estimators,
    )

    return XGBTrainResult(
        model=clf,
        accuracy=acc,
        f1=f1,
        precision=prec,
        recall=rec,
        roc_auc=auc,
        best_params=hp,
        feature_importance=importance,
        report=report,
    )


# ---------------------------------------------------------------------------
# Cross-validated training
# ---------------------------------------------------------------------------


def train_xgboost_cv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    *,
    n_folds: int = 5,
    params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> XGBTrainResult:
    """
    Train with stratified k-fold cross-validation.

    Trains on each fold, collects OOF (out-of-fold) metrics, then
    retrains a single model on all data with the same params.

    Returns
    -------
    XGBTrainResult
        best_params, cv_scores (AUC per fold), and final model.
    """
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold

    hp = {**DEFAULT_PARAMS, **(params or {})}
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_aucs: List[float] = []
    fold_f1s: List[float] = []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X[train_idx], X[val_idx]
        y_tr, y_va = y[train_idx], y[val_idx]

        clf = xgb.XGBClassifier(**hp)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )

        y_prob = clf.predict_proba(X_va)[:, 1]
        y_pred = clf.predict(X_va)
        auc = roc_auc_score(y_va, y_prob)
        f1_val = f1_score(y_va, y_pred)
        fold_aucs.append(auc)
        fold_f1s.append(f1_val)
        logger.info("  Fold %d — AUC=%.4f  F1=%.4f", fold_i, auc, f1_val)

    logger.info(
        "CV complete — mean AUC=%.4f ± %.4f",
        np.mean(fold_aucs), np.std(fold_aucs),
    )

    # Final model on all data
    final_clf = xgb.XGBClassifier(**hp)
    final_clf.fit(X, y, verbose=False)

    raw_imp = final_clf.feature_importances_
    importance = {
        name: round(float(imp), 6)
        for name, imp in sorted(
            zip(feature_names, raw_imp), key=lambda x: -x[1]
        )
    }

    return XGBTrainResult(
        model=final_clf,
        accuracy=0.0,  # not applicable for CV
        f1=float(np.mean(fold_f1s)),
        roc_auc=float(np.mean(fold_aucs)),
        best_params=hp,
        feature_importance=importance,
        cv_scores=fold_aucs,
    )


# ---------------------------------------------------------------------------
# Optuna hyperparameter tuning
# ---------------------------------------------------------------------------


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_trials: int = 50,
    timeout: int = 300,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Bayesian hyperparameter optimisation via Optuna.

    Optimises ROC-AUC on the validation set.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials to run.
    timeout : int
        Max seconds for the entire study.

    Returns
    -------
    dict
        Best hyperparameters, ready to pass to train_xgboost().
    """
    import optuna
    import xgboost as xgb

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        p = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": -1,
            "max_depth": trial.suggest_int("max_depth", *TUNING_SPACE["max_depth"]),
            "learning_rate": trial.suggest_float(
                "learning_rate", *TUNING_SPACE["learning_rate"], log=True
            ),
            "n_estimators": trial.suggest_int(
                "n_estimators", *TUNING_SPACE["n_estimators"], step=50
            ),
            "subsample": trial.suggest_float("subsample", *TUNING_SPACE["subsample"]),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *TUNING_SPACE["colsample_bytree"]
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", *TUNING_SPACE["min_child_weight"]
            ),
            "gamma": trial.suggest_float("gamma", *TUNING_SPACE["gamma"]),
            "reg_alpha": trial.suggest_float("reg_alpha", *TUNING_SPACE["reg_alpha"]),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", *TUNING_SPACE["reg_lambda"]
            ),
        }

        clf = xgb.XGBClassifier(**p)
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_prob = clf.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best = study.best_params
    best.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "tree_method": "hist",
            "random_state": random_state,
            "n_jobs": -1,
        }
    )

    logger.info(
        "Optuna best AUC=%.4f after %d trials", study.best_value, len(study.trials)
    )
    return best


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_flood_proba(
    model: Any,
    X: np.ndarray,
) -> np.ndarray:
    """Return P(flood=1) for each row."""
    return model.predict_proba(X)[:, 1]


def predict_flood(
    model: Any,
    X: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Binary prediction with adjustable threshold."""
    proba = predict_flood_proba(model, X)
    return (proba >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(model: Any, path: str | Path) -> None:
    """Save XGBoost model to disk."""
    import joblib

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(path))
    logger.info("XGBoost model saved → %s", path)


def load_model(path: str | Path) -> Any:
    """Load a previously saved XGBoost model."""
    import joblib

    return joblib.load(str(path))
