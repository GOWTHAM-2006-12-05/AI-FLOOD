"""
ensemble.py — Weighted ensemble blending of XGBoost + LSTM flood predictions.

Strategy:
    P_ensemble = α · P_xgb + (1 - α) · P_lstm

    Default α = 0.65 (XGBoost weighted higher for structured data).
    α is tuneable: can be optimised on validation set via grid search.

Additional capabilities:
    • Stacking meta-learner (logistic regression on top of base predictions)
    • Confidence-based dynamic weighting
    • Disagreement detection (when models strongly conflict)
    • Risk categorisation from ensemble probability
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Risk categories
# ---------------------------------------------------------------------------


class FloodRisk(str, Enum):
    """Flood risk level derived from ensemble probability."""
    MINIMAL = "minimal"         # P < 0.15
    LOW = "low"                 # 0.15 ≤ P < 0.35
    MODERATE = "moderate"       # 0.35 ≤ P < 0.60
    HIGH = "high"               # 0.60 ≤ P < 0.80
    CRITICAL = "critical"       # P ≥ 0.80


def classify_risk(prob: float) -> FloodRisk:
    """Map a probability to a risk category."""
    if prob < 0.15:
        return FloodRisk.MINIMAL
    elif prob < 0.35:
        return FloodRisk.LOW
    elif prob < 0.60:
        return FloodRisk.MODERATE
    elif prob < 0.80:
        return FloodRisk.HIGH
    else:
        return FloodRisk.CRITICAL


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class EnsembleResult:
    """Output of the ensemble prediction."""

    flood_probability: float
    risk_level: FloodRisk
    xgb_probability: float
    lstm_probability: float
    alpha: float  # blending weight for XGBoost
    confidence: float  # 1 - model disagreement
    models_agree: bool  # True if both models on same side of 0.5
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flood_probability": round(self.flood_probability, 4),
            "risk_level": self.risk_level.value,
            "xgb_probability": round(self.xgb_probability, 4),
            "lstm_probability": round(self.lstm_probability, 4),
            "alpha": round(self.alpha, 3),
            "confidence": round(self.confidence, 4),
            "models_agree": self.models_agree,
            **self.details,
        }


# ---------------------------------------------------------------------------
# Core blending
# ---------------------------------------------------------------------------


def blend_predictions(
    xgb_proba: np.ndarray | float,
    lstm_proba: np.ndarray | float,
    alpha: float = 0.65,
) -> np.ndarray:
    """
    Weighted average blending.

    P_ensemble = α · P_xgb + (1 - α) · P_lstm

    Parameters
    ----------
    xgb_proba : array or scalar
        XGBoost flood probabilities.
    lstm_proba : array or scalar
        LSTM flood probabilities.
    alpha : float
        Weight for XGBoost (0–1). Default 0.65.

    Returns
    -------
    np.ndarray
        Blended probabilities.
    """
    xgb = np.asarray(xgb_proba, dtype=float)
    lstm = np.asarray(lstm_proba, dtype=float)
    blended = alpha * xgb + (1 - alpha) * lstm
    return np.clip(blended, 0.0, 1.0)


def predict_ensemble(
    xgb_proba: float,
    lstm_proba: float,
    alpha: float = 0.65,
) -> EnsembleResult:
    """
    Full ensemble prediction with metadata for a single sample.

    Parameters
    ----------
    xgb_proba : float
        XGBoost P(flood).
    lstm_proba : float
        LSTM P(flood).
    alpha : float
        Blending weight for XGBoost.

    Returns
    -------
    EnsembleResult
    """
    prob = float(blend_predictions(xgb_proba, lstm_proba, alpha))
    risk = classify_risk(prob)

    # Confidence: inversely proportional to disagreement
    disagreement = abs(xgb_proba - lstm_proba)
    confidence = 1.0 - disagreement

    # Models agree if both on same side of 0.5
    agree = (xgb_proba >= 0.5) == (lstm_proba >= 0.5)

    return EnsembleResult(
        flood_probability=prob,
        risk_level=risk,
        xgb_probability=xgb_proba,
        lstm_probability=lstm_proba,
        alpha=alpha,
        confidence=confidence,
        models_agree=agree,
    )


# ---------------------------------------------------------------------------
# Optimal alpha search
# ---------------------------------------------------------------------------


def optimise_alpha(
    xgb_proba: np.ndarray,
    lstm_proba: np.ndarray,
    y_true: np.ndarray,
    metric: str = "roc_auc",
    n_steps: int = 101,
) -> Tuple[float, float]:
    """
    Grid-search for the optimal blending weight α on validation data.

    Parameters
    ----------
    xgb_proba : array
        XGBoost predicted probabilities on validation set.
    lstm_proba : array
        LSTM predicted probabilities on validation set.
    y_true : array
        True labels.
    metric : str
        Optimisation target: 'roc_auc', 'f1', or 'accuracy'.
    n_steps : int
        Number of α values to try (0.00, 0.01, ..., 1.00).

    Returns
    -------
    (best_alpha, best_score)
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    best_alpha = 0.5
    best_score = -1.0

    for i in range(n_steps):
        a = i / (n_steps - 1)
        blended = blend_predictions(xgb_proba, lstm_proba, alpha=a)

        if metric == "roc_auc":
            try:
                score = roc_auc_score(y_true, blended)
            except ValueError:
                score = 0.0
        elif metric == "f1":
            y_pred = (blended >= 0.5).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "accuracy":
            y_pred = (blended >= 0.5).astype(int)
            score = accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_alpha = a

    logger.info(
        "Optimal α=%.3f  %s=%.4f", best_alpha, metric, best_score
    )
    return best_alpha, best_score


# ---------------------------------------------------------------------------
# Stacking meta-learner (optional advanced ensemble)
# ---------------------------------------------------------------------------


@dataclass
class StackingEnsemble:
    """
    Level-2 stacking: train a logistic regression on base-model outputs.

    This can capture non-linear interactions between the two models.
    """

    meta_model: Any = None
    is_fitted: bool = False

    def fit(
        self,
        xgb_proba: np.ndarray,
        lstm_proba: np.ndarray,
        y_true: np.ndarray,
    ) -> None:
        """
        Train logistic regression stacker on base predictions.
        """
        from sklearn.linear_model import LogisticRegression

        X_meta = np.column_stack([xgb_proba, lstm_proba])
        self.meta_model = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000
        )
        self.meta_model.fit(X_meta, y_true)
        self.is_fitted = True
        logger.info(
            "Stacking meta-learner fitted — coefs: XGB=%.4f  LSTM=%.4f",
            self.meta_model.coef_[0][0],
            self.meta_model.coef_[0][1],
        )

    def predict_proba(
        self,
        xgb_proba: np.ndarray,
        lstm_proba: np.ndarray,
    ) -> np.ndarray:
        """Return P(flood) from the stacker."""
        if not self.is_fitted:
            raise RuntimeError("StackingEnsemble not fitted; call .fit() first")
        X_meta = np.column_stack([xgb_proba, lstm_proba])
        return self.meta_model.predict_proba(X_meta)[:, 1]


# ---------------------------------------------------------------------------
# Confidence-based dynamic weighting
# ---------------------------------------------------------------------------


def dynamic_blend(
    xgb_proba: float,
    lstm_proba: float,
    xgb_confidence: float = 1.0,
    lstm_confidence: float = 1.0,
) -> float:
    """
    Dynamically weight models based on their confidence.

    Confidence can be derived from:
        • XGBoost: tree-based prediction variance across estimators
        • LSTM: MC-Dropout multiple forward passes

    Parameters
    ----------
    xgb_proba, lstm_proba : float
        Base predictions.
    xgb_confidence, lstm_confidence : float
        Confidence scores (0–1). Higher = more confident.

    Returns
    -------
    float
        Dynamically weighted probability.
    """
    total = xgb_confidence + lstm_confidence + 1e-8
    w_xgb = xgb_confidence / total
    w_lstm = lstm_confidence / total
    return float(np.clip(w_xgb * xgb_proba + w_lstm * lstm_proba, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Batch ensemble processing
# ---------------------------------------------------------------------------


def batch_ensemble_predict(
    xgb_proba: np.ndarray,
    lstm_proba: np.ndarray,
    alpha: float = 0.65,
) -> List[EnsembleResult]:
    """
    Run ensemble prediction for multiple samples.

    Returns
    -------
    list of EnsembleResult
    """
    results = []
    blended = blend_predictions(xgb_proba, lstm_proba, alpha)
    for i in range(len(blended)):
        results.append(
            predict_ensemble(
                float(xgb_proba[i]),
                float(lstm_proba[i]),
                alpha=alpha,
            )
        )
    return results
