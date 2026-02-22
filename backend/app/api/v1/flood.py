"""
FastAPI flood prediction endpoints.

Endpoints:
    POST /api/v1/flood/predict       — Predict flood risk for a location
    POST /api/v1/flood/predict-batch  — Batch prediction for multiple locations
    POST /api/v1/flood/train          — Trigger training pipeline
    GET  /api/v1/flood/model-info     — Get model status and metrics
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.ml.flood_service import (
    FloodPredictionService,
    get_flood_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/flood", tags=["flood-prediction"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class FloodPredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude in degrees")
    # Optional overrides for terrain features (not available from weather API)
    elevation: Optional[float] = Field(None, description="Elevation in metres")
    drainage_capacity: Optional[float] = Field(
        None, ge=0, le=1, description="Drainage capacity factor (0=poor, 1=good)"
    )
    urbanization: Optional[float] = Field(
        None, ge=0, le=1, description="Urbanization factor (0=rural, 1=dense)"
    )


class FloodPredictResponse(BaseModel):
    latitude: float
    longitude: float
    flood_probability: float
    risk_level: str
    xgb_probability: float
    lstm_probability: float
    confidence: float
    models_agree: bool
    alert_action: str
    alert_message: str
    alert_color: str
    weather_summary: Dict[str, Any] = {}
    features_used: Dict[str, float] = {}


class BatchPredictRequest(BaseModel):
    locations: List[FloodPredictRequest] = Field(
        ..., max_length=20, description="Up to 20 locations"
    )


class TrainRequest(BaseModel):
    n_samples: int = Field(5000, ge=500, le=50000)
    do_tune: bool = Field(False, description="Run Optuna hyperparameter tuning")
    tune_trials: int = Field(30, ge=5, le=200)


class ModelInfoResponse(BaseModel):
    is_ready: bool
    message: str
    metrics: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/predict", response_model=FloodPredictResponse)
async def predict_flood(req: FloodPredictRequest):
    """
    Predict flood risk for a single location.

    Fetches live weather data from Open-Meteo, builds ML features,
    runs XGBoost + LSTM ensemble, and returns risk assessment.
    """
    service = get_flood_service()

    if not service.is_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "Models not loaded. Train first via POST /api/v1/flood/train "
                "or load pre-trained models."
            ),
        )

    try:
        prediction = service.predict(
            latitude=req.latitude,
            longitude=req.longitude,
        )

        ens = prediction.ensemble_result
        return FloodPredictResponse(
            latitude=prediction.latitude,
            longitude=prediction.longitude,
            flood_probability=round(ens.flood_probability, 4),
            risk_level=ens.risk_level.value,
            xgb_probability=round(ens.xgb_probability, 4),
            lstm_probability=round(ens.lstm_probability, 4),
            confidence=round(ens.confidence, 4),
            models_agree=ens.models_agree,
            alert_action=prediction.alert["action"],
            alert_message=prediction.alert["message"],
            alert_color=prediction.alert["color"],
            weather_summary=prediction.weather_summary,
            features_used=prediction.features_used,
        )

    except Exception as e:
        logger.exception("Flood prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-batch")
async def predict_flood_batch(req: BatchPredictRequest):
    """Predict flood risk for multiple locations (max 20)."""
    service = get_flood_service()

    if not service.is_ready:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    results = []
    for loc in req.locations:
        try:
            prediction = service.predict(
                latitude=loc.latitude,
                longitude=loc.longitude,
            )
            results.append(prediction.to_dict())
        except Exception as e:
            results.append({
                "latitude": loc.latitude,
                "longitude": loc.longitude,
                "error": str(e),
            })

    return {"predictions": results, "count": len(results)}


@router.post("/train")
async def train_model(req: TrainRequest):
    """
    Trigger model training pipeline.

    This trains XGBoost + LSTM on synthetic data and loads the
    models into the prediction service.

    Note: In production, training would run as a background job.
    """
    try:
        from backend.app.ml.train_pipeline import run_pipeline

        metrics = run_pipeline(
            n_samples=req.n_samples,
            do_tune=req.do_tune,
            tune_trials=req.tune_trials,
            save=True,
        )

        # Load trained models into the service
        service = get_flood_service()
        from pathlib import Path
        model_dir = Path(__file__).resolve().parent.parent.parent.parent / "models"
        service.load_models(model_dir)

        return {
            "status": "success",
            "message": "Training complete. Models loaded.",
            "metrics": {
                "xgboost_accuracy": metrics.get("xgboost", {}).get("accuracy"),
                "lstm_accuracy": metrics.get("lstm", {}).get("accuracy"),
                "test_accuracy": metrics.get("test", {}).get("accuracy"),
                "test_auc": metrics.get("test", {}).get("roc_auc"),
                "elapsed_seconds": metrics.get("elapsed_seconds"),
            },
        }

    except Exception as e:
        logger.exception("Training pipeline failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get current model status and available metrics."""
    service = get_flood_service()

    if not service.is_ready:
        return ModelInfoResponse(
            is_ready=False,
            message="No models loaded. Train via POST /api/v1/flood/train",
        )

    # Try to load saved metrics
    metrics = {}
    try:
        import json
        from pathlib import Path

        metrics_path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "models"
            / "training_metrics.json"
        )
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
    except Exception:
        pass

    return ModelInfoResponse(
        is_ready=True,
        message="Models loaded and ready for prediction.",
        metrics=metrics,
    )
