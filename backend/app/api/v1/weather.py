"""
FastAPI route: Weather Data — ingestion endpoints for Open-Meteo weather data.

Provides endpoints to:
    - Fetch full weather data for a coordinate
    - Get rainfall accumulation summary
    - Get ML-ready feature vector for a location
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from backend.app.ingestion.weather_service import (
    FetchStatus,
    fetch_rainfall_summary,
    fetch_weather,
)
from backend.app.features.weather_features import build_features, print_features

router = APIRouter(prefix="/api/v1/weather", tags=["weather"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class WeatherRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, examples=[13.0827])
    longitude: float = Field(..., ge=-180, le=180, examples=[80.2707])
    forecast_days: int = Field(default=2, ge=1, le=16)
    past_days: int = Field(default=1, ge=0, le=92)


class RainfallOut(BaseModel):
    rain_1hr: float
    rain_3hr: float
    rain_6hr: float
    rain_24hr: float
    flood_risk: str
    intensity_ratio: float


class CurrentConditionsOut(BaseModel):
    temperature_c: float
    humidity_pct: float
    wind_speed_kmh: float
    wind_gusts_kmh: float
    pressure_hpa: float
    cloud_cover_pct: float
    soil_moisture: float
    timestamp: str


class WeatherResponse(BaseModel):
    success: bool
    status: str
    latitude: float
    longitude: float
    elevation_m: float = 0.0
    current: Optional[CurrentConditionsOut] = None
    rainfall: Optional[RainfallOut] = None
    has_data_gaps: bool = False
    fetch_duration_ms: int = 0
    error_message: str = ""


class MLFeaturesResponse(BaseModel):
    success: bool
    latitude: float
    longitude: float
    feature_count: int = 0
    features: Dict[str, float] = {}
    error_message: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/fetch",
    response_model=WeatherResponse,
    summary="Fetch full weather data",
    description=(
        "Fetches hourly weather data from Open-Meteo for a coordinate. "
        "Returns current conditions, rainfall accumulations, and metadata."
    ),
)
async def weather_fetch(body: WeatherRequest):
    """
    Full weather data fetch.

    **Flow:**
    1. Validate coordinates
    2. Call Open-Meteo API with retry logic
    3. Parse response, compute rainfall accumulations
    4. Return structured result
    """
    result = fetch_weather(
        body.latitude,
        body.longitude,
        forecast_days=body.forecast_days,
        past_days=body.past_days,
    )

    response = WeatherResponse(
        success=result.success,
        status=result.status.value,
        latitude=result.latitude,
        longitude=result.longitude,
        elevation_m=result.elevation_m,
        has_data_gaps=result.has_data_gaps,
        fetch_duration_ms=result.fetch_duration_ms,
        error_message=result.error_message,
    )

    if result.current:
        response.current = CurrentConditionsOut(
            temperature_c=result.current.temperature_c,
            humidity_pct=result.current.humidity_pct,
            wind_speed_kmh=result.current.wind_speed_kmh,
            wind_gusts_kmh=result.current.wind_gusts_kmh,
            pressure_hpa=result.current.pressure_hpa,
            cloud_cover_pct=result.current.cloud_cover_pct,
            soil_moisture=result.current.soil_moisture,
            timestamp=result.current.timestamp,
        )

    if result.rainfall:
        response.rainfall = RainfallOut(
            rain_1hr=result.rainfall.rain_1hr,
            rain_3hr=result.rainfall.rain_3hr,
            rain_6hr=result.rainfall.rain_6hr,
            rain_24hr=result.rainfall.rain_24hr,
            flood_risk=result.rainfall.flood_risk_category,
            intensity_ratio=result.rainfall.intensity_ratio,
        )

    return response


@router.get(
    "/rainfall",
    summary="Quick rainfall summary",
    description="Lightweight endpoint returning only rainfall accumulation data.",
)
async def rainfall_summary(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
):
    """Returns rain_1hr, rain_3hr, rain_6hr, rain_24hr + flood risk category."""
    return fetch_rainfall_summary(lat, lon)


@router.post(
    "/ml-features",
    response_model=MLFeaturesResponse,
    summary="Get ML-ready feature vector",
    description=(
        "Fetches weather data and transforms it into a flat feature dictionary "
        "suitable for model.predict(). Includes current conditions, rainfall "
        "accumulations, trends, temporal encoding, and composite features."
    ),
)
async def ml_features(body: WeatherRequest):
    """
    End-to-end: fetch weather → engineer features → return feature vector.

    This is the endpoint the ML inference pipeline calls.
    """
    result = fetch_weather(
        body.latitude,
        body.longitude,
        forecast_days=body.forecast_days,
        past_days=body.past_days,
    )

    if not result.success:
        return MLFeaturesResponse(
            success=False,
            latitude=body.latitude,
            longitude=body.longitude,
            error_message=result.error_message,
        )

    features = build_features(result)

    return MLFeaturesResponse(
        success=True,
        latitude=result.latitude,
        longitude=result.longitude,
        feature_count=len(features),
        features=features,
    )
