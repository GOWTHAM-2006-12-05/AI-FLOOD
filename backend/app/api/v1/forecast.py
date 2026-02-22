"""
FastAPI endpoints for sub-hour flood forecasting.

Routes:
    POST /api/v1/forecast/predict     — Full multi-horizon forecast
    POST /api/v1/forecast/burst       — Rainfall burst detection only
    POST /api/v1/forecast/trend       — Rainfall trend analysis only
    POST /api/v1/forecast/train       — Train the forecast LSTM
    GET  /api/v1/forecast/horizons    — List available forecast horizons
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.ml.forecast_engine import (
    ALL_HORIZONS,
    HORIZON_LABELS,
    ForecastResult,
    Observation,
    _FallbackForecaster,
    build_input_window,
    detect_burst,
    estimate_trend,
    generate_forecast_training_data,
    run_forecast,
    train_forecast_model,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/forecast",
    tags=["forecast"],
)


# ── State ──────────────────────────────────────────────────────────

_forecast_model: Any = None
_is_fallback: bool = True


# ── Schemas ────────────────────────────────────────────────────────


class ObservationInput(BaseModel):
    """Single 10-minute observation."""
    timestamp_unix: float = Field(..., description="Unix epoch seconds")
    rainfall_mm: float = Field(0.0, ge=0, description="Rainfall in this 10-min interval (mm)")
    soil_moisture: float = Field(0.3, ge=0, le=1, description="Volumetric water content")
    temperature_c: float = Field(28.0, description="Temperature (°C)")
    relative_humidity: float = Field(70.0, ge=0, le=100, description="Relative humidity (%)")
    surface_pressure_hpa: float = Field(1013.0, description="Surface pressure (hPa)")
    wind_speed_ms: float = Field(3.0, ge=0, description="Wind speed (m/s)")


class ForecastRequest(BaseModel):
    """Input for multi-horizon forecast."""
    observations: List[ObservationInput] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="10-minute observations, oldest first. At least 1 required, 18 ideal.",
    )
    xgb_probability: float = Field(
        0.0, ge=0, le=1,
        description="XGBoost snapshot flood probability for ensemble blending",
    )
    alpha: float = Field(
        0.55, ge=0, le=1,
        description="Base blending weight for XGBoost (0=LSTM only, 1=XGB only)",
    )


class BurstRequest(BaseModel):
    """Input for burst detection."""
    observations: List[ObservationInput] = Field(
        ..., min_length=1, max_length=100,
    )
    rate_threshold_mm_hr: float = Field(20.0, ge=1, description="Burst rate threshold")
    gradient_threshold: float = Field(5.0, ge=0.5, description="Rate gradient threshold")
    sustained_steps: int = Field(2, ge=1, le=10, description="Min consecutive steps")


class TrendRequest(BaseModel):
    """Input for trend analysis."""
    observations: List[ObservationInput] = Field(
        ..., min_length=3, max_length=100,
    )
    lookback_steps: int = Field(6, ge=3, le=18, description="Steps for trend fitting")


class TrainRequest(BaseModel):
    """Parameters for training the forecast model."""
    n_samples: int = Field(3000, ge=500, le=50000, description="Training samples")
    epochs: int = Field(80, ge=10, le=500, description="Max epochs")
    batch_size: int = Field(32, ge=8, le=256, description="Mini-batch size")
    patience: int = Field(10, ge=3, le=50, description="Early stopping patience")
    seed: int = Field(42, description="Random seed")


# ── Helpers ────────────────────────────────────────────────────────


def _to_observations(inputs: List[ObservationInput]) -> List[Observation]:
    """Convert Pydantic models to engine Observation objects."""
    return [
        Observation(
            timestamp_unix=o.timestamp_unix,
            rainfall_mm=o.rainfall_mm,
            soil_moisture=o.soil_moisture,
            temperature_c=o.temperature_c,
            relative_humidity=o.relative_humidity,
            surface_pressure_hpa=o.surface_pressure_hpa,
            wind_speed_ms=o.wind_speed_ms,
        )
        for o in inputs
    ]


# ── Endpoints ──────────────────────────────────────────────────────


@router.post("/predict", summary="Multi-horizon flood forecast")
async def forecast_predict(req: ForecastRequest) -> Dict[str, Any]:
    """
    Run a complete sub-hour flood forecast.

    Accepts 10-minute observations and returns flood probability
    predictions for T+30min, T+1hr, and T+3hr with confidence intervals.
    """
    global _forecast_model, _is_fallback

    observations = _to_observations(req.observations)

    result = run_forecast(
        observations=observations,
        model=_forecast_model,
        xgb_probability=req.xgb_probability,
        alpha=req.alpha,
        is_fallback=_is_fallback,
    )

    return result.to_dict()


@router.post("/burst", summary="Rainfall burst detection")
async def detect_rainfall_burst(req: BurstRequest) -> Dict[str, Any]:
    """
    Analyse observations for rainfall burst patterns.
    """
    observations = _to_observations(req.observations)
    window = build_input_window(observations)

    burst = detect_burst(
        window,
        rate_threshold=req.rate_threshold_mm_hr,
        gradient_threshold=req.gradient_threshold,
        sustained_steps=req.sustained_steps,
    )

    return {
        "burst_detection": burst.to_dict(),
        "observations_analysed": len(observations),
    }


@router.post("/trend", summary="Rainfall trend analysis")
async def analyse_trend(req: TrendRequest) -> Dict[str, Any]:
    """
    Estimate rainfall acceleration / deceleration trend.
    """
    observations = _to_observations(req.observations)
    window = build_input_window(observations)

    trend = estimate_trend(window, lookback_steps=req.lookback_steps)

    return {
        "trend_estimate": trend.to_dict(),
        "observations_analysed": len(observations),
    }


@router.post("/train", summary="Train the forecast LSTM model")
async def train_model(req: TrainRequest) -> Dict[str, Any]:
    """
    Train (or retrain) the forecast model on synthetic data.

    In production, this would accept real historical observations.
    """
    global _forecast_model, _is_fallback

    t0 = time.time()

    # Generate training data
    X, y = generate_forecast_training_data(
        n_samples=req.n_samples,
        seed=req.seed,
    )

    # Train
    result = train_forecast_model(
        X=X,
        y=y,
        epochs=req.epochs,
        batch_size=req.batch_size,
        patience=req.patience,
    )

    _forecast_model = result.model
    _is_fallback = result.using_fallback

    elapsed = time.time() - t0

    return {
        "status": "trained",
        "elapsed_seconds": round(elapsed, 2),
        "model_type": "fallback" if _is_fallback else "lstm_multi_horizon",
        **result.summary(),
    }


@router.get("/horizons", summary="List forecast horizons")
async def list_horizons() -> Dict[str, Any]:
    """
    Return metadata about available forecast horizons.
    """
    return {
        "horizons": [
            {
                "id": h.value,
                "label": HORIZON_LABELS[h],
                "minutes": h.value,
                "hours": round(h.value / 60.0, 2),
            }
            for h in ALL_HORIZONS
        ],
        "input_window": {
            "interval_minutes": 10,
            "timesteps": 18,
            "total_duration_minutes": 180,
            "features": [
                "rainfall_mm",
                "rainfall_cumulative",
                "rainfall_rate_mm_hr",
                "soil_moisture",
                "temperature_c",
                "relative_humidity",
                "surface_pressure_hpa",
                "wind_speed_ms",
            ],
        },
    }
