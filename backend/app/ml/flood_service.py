"""
flood_service.py — Modular inference service for real-time flood prediction.

Connects the full pipeline:
    weather_service.fetch_weather()
    → weather_features.build_features()
    → preprocessing (scale)
    → XGBoost predict
    → LSTM predict
    → ensemble blend
    → risk level + alert recommendation

This is the single entry-point for the API layer. It manages model loading,
feature transformation, and caching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from backend.app.ml.ensemble import (
    EnsembleResult,
    FloodRisk,
    blend_predictions,
    classify_risk,
    predict_ensemble,
)
from backend.app.ml.preprocessing import (
    ALL_FEATURES,
    CORE_FEATURES,
    RobustScaler,
    load_scaler,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert recommendation
# ---------------------------------------------------------------------------

ALERT_RECOMMENDATIONS: Dict[FloodRisk, Dict[str, str]] = {
    FloodRisk.MINIMAL: {
        "action": "No action required",
        "message": "No flood risk detected in your area.",
        "color": "#4CAF50",  # green
    },
    FloodRisk.LOW: {
        "action": "Stay informed",
        "message": "Low flood risk. Monitor weather updates.",
        "color": "#8BC34A",  # light green
    },
    FloodRisk.MODERATE: {
        "action": "Prepare",
        "message": "Moderate flood risk. Prepare emergency supplies and identify evacuation routes.",
        "color": "#FF9800",  # orange
    },
    FloodRisk.HIGH: {
        "action": "Be ready to evacuate",
        "message": "HIGH flood risk! Move valuables to higher ground. Follow local authority instructions.",
        "color": "#F44336",  # red
    },
    FloodRisk.CRITICAL: {
        "action": "EVACUATE IMMEDIATELY",
        "message": "CRITICAL flood risk! Immediate evacuation recommended. Seek higher ground NOW.",
        "color": "#B71C1C",  # dark red
    },
}


# ---------------------------------------------------------------------------
# Flood prediction result
# ---------------------------------------------------------------------------


@dataclass
class FloodPrediction:
    """Complete flood prediction output for a location."""

    latitude: float
    longitude: float
    ensemble_result: EnsembleResult
    alert: Dict[str, str]
    features_used: Dict[str, float] = field(default_factory=dict)
    weather_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "prediction": self.ensemble_result.to_dict(),
            "alert": self.alert,
            "features_used": {
                k: round(v, 4) for k, v in self.features_used.items()
            },
            "weather_summary": self.weather_summary,
        }


# ---------------------------------------------------------------------------
# Service class
# ---------------------------------------------------------------------------


class FloodPredictionService:
    """
    Stateful service that holds loaded models and scaler.

    Usage:
        service = FloodPredictionService()
        service.load_models("models/")
        prediction = service.predict(lat=13.08, lon=80.27)
    """

    def __init__(self):
        self._xgb_model: Any = None
        self._lstm_model: Any = None
        self._scaler: Optional[RobustScaler] = None
        self._alpha: float = 0.65
        self._feature_names: List[str] = list(ALL_FEATURES)
        self._is_ready: bool = False
        self._lstm_is_fallback: bool = False

    # ---- Model management ----

    def load_models(
        self,
        model_dir: str | Path,
        xgb_filename: str = "xgb_flood.joblib",
        lstm_filename: str = "lstm_flood",
        scaler_filename: str = "scaler.joblib",
    ) -> None:
        """
        Load pre-trained models and scaler from disk.
        """
        model_dir = Path(model_dir)

        # XGBoost
        xgb_path = model_dir / xgb_filename
        if xgb_path.exists():
            from backend.app.ml.xgboost_model import load_model
            self._xgb_model = load_model(xgb_path)
            logger.info("XGBoost model loaded from %s", xgb_path)
        else:
            logger.warning("XGBoost model not found at %s", xgb_path)

        # LSTM
        lstm_path = model_dir / lstm_filename
        lstm_joblib = model_dir / f"{lstm_filename}.joblib"
        if lstm_joblib.exists():
            from backend.app.ml.lstm_model import load_lstm
            self._lstm_model = load_lstm(lstm_joblib, is_fallback=True)
            self._lstm_is_fallback = True
            logger.info("LSTM fallback model loaded from %s", lstm_joblib)
        elif lstm_path.exists():
            from backend.app.ml.lstm_model import load_lstm
            self._lstm_model = load_lstm(lstm_path, is_fallback=False)
            logger.info("LSTM model loaded from %s", lstm_path)
        else:
            logger.warning("LSTM model not found at %s", lstm_path)

        # Scaler
        scaler_path = model_dir / scaler_filename
        if scaler_path.exists():
            self._scaler = load_scaler(scaler_path)
            logger.info("Scaler loaded from %s", scaler_path)
        else:
            logger.warning("Scaler not found at %s", scaler_path)

        self._is_ready = (
            self._xgb_model is not None or self._lstm_model is not None
        )

    def set_models(
        self,
        xgb_model: Any = None,
        lstm_model: Any = None,
        scaler: Optional[RobustScaler] = None,
        alpha: float = 0.65,
        feature_names: Optional[List[str]] = None,
        lstm_is_fallback: bool = False,
    ) -> None:
        """
        Directly inject models (e.g. after training).
        """
        if xgb_model is not None:
            self._xgb_model = xgb_model
        if lstm_model is not None:
            self._lstm_model = lstm_model
            self._lstm_is_fallback = lstm_is_fallback
        if scaler is not None:
            self._scaler = scaler
        self._alpha = alpha
        if feature_names:
            self._feature_names = feature_names
        self._is_ready = (
            self._xgb_model is not None or self._lstm_model is not None
        )

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    # ---- Prediction ----

    def predict_from_features(
        self,
        features: Dict[str, float],
        latitude: float = 0.0,
        longitude: float = 0.0,
    ) -> FloodPrediction:
        """
        Run flood prediction from a pre-built feature dictionary.

        Parameters
        ----------
        features : dict
            Feature name → value mapping (from weather_features.build_features).
        latitude, longitude : float
            Location for the prediction record.

        Returns
        -------
        FloodPrediction
        """
        if not self._is_ready:
            raise RuntimeError(
                "FloodPredictionService not ready. Call load_models() or set_models() first."
            )

        # Build feature vector in correct order
        feature_vec = np.array(
            [features.get(f, 0.0) for f in self._feature_names],
            dtype=float,
        ).reshape(1, -1)

        # Scale
        if self._scaler is not None:
            feature_vec_scaled = self._scaler.transform(feature_vec)
        else:
            feature_vec_scaled = feature_vec

        # --- XGBoost prediction ---
        xgb_prob = 0.5  # fallback
        if self._xgb_model is not None:
            from backend.app.ml.xgboost_model import predict_flood_proba
            xgb_prob = float(predict_flood_proba(self._xgb_model, feature_vec_scaled)[0])

        # --- LSTM prediction ---
        lstm_prob = 0.5  # fallback
        if self._lstm_model is not None:
            # For LSTM we need a sequence; create a "static" sequence
            # by repeating the current features across the sequence length
            from backend.app.ml.lstm_model import (
                DEFAULT_SEQ_LEN,
                SEQUENCE_FEATURES,
                predict_flood_proba_lstm,
            )

            seq_features = np.array(
                [features.get(f, 0.0) for f in SEQUENCE_FEATURES],
                dtype=float,
            )
            # Repeat to form (1, seq_len, n_seq_features)
            seq_input = np.tile(seq_features, (DEFAULT_SEQ_LEN, 1)).reshape(
                1, DEFAULT_SEQ_LEN, len(SEQUENCE_FEATURES)
            )
            lstm_prob = float(predict_flood_proba_lstm(self._lstm_model, seq_input)[0])

        # --- Ensemble ---
        ensemble = predict_ensemble(xgb_prob, lstm_prob, alpha=self._alpha)

        # Alert
        alert = ALERT_RECOMMENDATIONS[ensemble.risk_level]

        return FloodPrediction(
            latitude=latitude,
            longitude=longitude,
            ensemble_result=ensemble,
            alert=alert,
            features_used={
                f: features.get(f, 0.0)
                for f in self._feature_names[:8]  # core features only
            },
        )

    def predict(
        self,
        latitude: float,
        longitude: float,
    ) -> FloodPrediction:
        """
        End-to-end: fetch weather → build features → predict flood risk.

        Parameters
        ----------
        latitude, longitude : float
            GPS coordinates.

        Returns
        -------
        FloodPrediction
        """
        # 1. Fetch weather
        from backend.app.ingestion.weather_service import fetch_weather
        from backend.app.features.weather_features import build_features

        weather_result = fetch_weather(latitude, longitude)
        if weather_result.status.value != "success":
            logger.error("Weather fetch failed: %s", weather_result.status)
            # Return a "no-data" prediction
            return FloodPrediction(
                latitude=latitude,
                longitude=longitude,
                ensemble_result=EnsembleResult(
                    flood_probability=0.0,
                    risk_level=FloodRisk.MINIMAL,
                    xgb_probability=0.0,
                    lstm_probability=0.0,
                    alpha=self._alpha,
                    confidence=0.0,
                    models_agree=True,
                    details={"error": f"Weather fetch failed: {weather_result.status.value}"},
                ),
                alert=ALERT_RECOMMENDATIONS[FloodRisk.MINIMAL],
                weather_summary={"error": weather_result.status.value},
            )

        # 2. Build features
        features = build_features(weather_result)

        # 3. Predict
        prediction = self.predict_from_features(
            features,
            latitude=latitude,
            longitude=longitude,
        )

        # 4. Attach weather summary
        prediction.weather_summary = {
            "temperature": features.get("temperature_2m", None),
            "humidity": features.get("relative_humidity", None),
            "rain_1hr": features.get("rain_1hr", None),
            "rain_24hr": features.get("rain_24hr", None),
            "wind_speed": features.get("wind_speed_10m", None),
            "pressure": features.get("surface_pressure", None),
        }

        return prediction


# ---------------------------------------------------------------------------
# Singleton-style convenience
# ---------------------------------------------------------------------------

_service_instance: Optional[FloodPredictionService] = None


def get_flood_service() -> FloodPredictionService:
    """Get or create the global flood prediction service."""
    global _service_instance
    if _service_instance is None:
        _service_instance = FloodPredictionService()
    return _service_instance
