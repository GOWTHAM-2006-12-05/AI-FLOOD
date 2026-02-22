"""
FastAPI cyclone monitoring endpoints.

Endpoints:
    POST /api/v1/cyclone/detect        — Detect cyclonic conditions from weather data
    POST /api/v1/cyclone/assess        — Full cyclone risk assessment
    POST /api/v1/cyclone/filter        — Radius-based cyclone event filtering
    POST /api/v1/cyclone/escalation    — Compute escalation level
    GET  /api/v1/cyclone/categories    — IMD cyclone category reference
    GET  /api/v1/cyclone/thresholds    — Current threshold info
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.ml.cyclone_service import (
    DEFAULT_PRESSURE_BASELINE_HPA,
    DEFAULT_RAINFALL_THRESHOLD_MM,
    DEFAULT_WIND_THRESHOLD_KMH,
    CycloneCategory,
    CycloneEvent,
    EscalationLevel,
    assess_cyclone_risk,
    classify_cyclone,
    compute_cyclone_risk,
    detect_cyclone_conditions,
    determine_escalation,
    filter_cyclones_by_radius,
    get_cyclone_alert,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/cyclone", tags=["cyclone-monitoring"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    wind_speed_kmh: float = Field(..., ge=0, description="Sustained wind speed in km/h")
    rainfall_mm: float = Field(0.0, ge=0, description="Accumulated rainfall in mm")
    pressure_hpa: float = Field(1013.25, description="Sea-level pressure in hPa")
    wind_gust_kmh: Optional[float] = Field(None, ge=0, description="Peak wind gust")
    wind_threshold: float = Field(60.0, ge=0, description="Wind threshold for detection")
    rainfall_threshold: float = Field(50.0, ge=0, description="Rainfall threshold")


class AssessRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    wind_speed_kmh: float = Field(..., ge=0)
    rainfall_mm: float = Field(0.0, ge=0)
    pressure_hpa: float = Field(1013.25)
    wind_gust_kmh: Optional[float] = Field(None, ge=0)
    wind_threshold: float = Field(60.0, ge=0)
    rainfall_threshold: float = Field(50.0, ge=0)


class EscalationRequest(BaseModel):
    wind_speed_kmh: float = Field(..., ge=0)
    distance_km: float = Field(..., ge=0)
    rainfall_mm: float = Field(0.0, ge=0)
    pressure_hpa: float = Field(1013.25)
    wind_threshold: float = Field(60.0, ge=0)


class CycloneEventInput(BaseModel):
    event_id: str
    name: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    wind_speed_kmh: float = Field(..., ge=0)
    pressure_hpa: float = Field(1013.25)
    rainfall_24h_mm: float = Field(0.0, ge=0)
    timestamp: str = Field(..., description="ISO-8601 timestamp")


class FilterRequest(BaseModel):
    user_latitude: float = Field(..., ge=-90, le=90)
    user_longitude: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(500.0, gt=0, le=5000)
    wind_threshold: float = Field(60.0, ge=0)
    sort_by: str = Field("distance", description="distance or risk")
    max_results: int = Field(20, ge=1, le=100)
    events: List[CycloneEventInput]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/detect")
async def detect_conditions(req: DetectRequest):
    """
    Detect cyclonic conditions from current weather observations.

    Checks wind speed, rainfall, and pressure against thresholds.
    Returns classification, heavy rainfall flag, and risk score.
    """
    result = detect_cyclone_conditions(
        wind_speed_kmh=req.wind_speed_kmh,
        rainfall_mm=req.rainfall_mm,
        pressure_hpa=req.pressure_hpa,
        wind_gust_kmh=req.wind_gust_kmh,
        wind_threshold=req.wind_threshold,
        rainfall_threshold=req.rainfall_threshold,
    )

    return result.to_dict()


@router.post("/assess")
async def assess_risk(req: AssessRequest):
    """
    Full cyclone risk assessment for a location.

    Combines detection, classification, escalation, and alert generation.
    """
    assessment = assess_cyclone_risk(
        latitude=req.latitude,
        longitude=req.longitude,
        wind_speed_kmh=req.wind_speed_kmh,
        rainfall_mm=req.rainfall_mm,
        pressure_hpa=req.pressure_hpa,
        wind_gust_kmh=req.wind_gust_kmh,
        wind_threshold=req.wind_threshold,
        rainfall_threshold=req.rainfall_threshold,
    )

    return assessment.to_dict()


@router.post("/filter")
async def filter_by_radius(req: FilterRequest):
    """
    Filter cyclone events by distance from user.

    Computes risk scores and escalation levels for each event within radius.
    """
    from datetime import datetime, timezone

    events = []
    for e in req.events:
        try:
            ts = datetime.fromisoformat(e.timestamp.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.now(timezone.utc)

        events.append(CycloneEvent(
            event_id=e.event_id,
            name=e.name,
            latitude=e.latitude,
            longitude=e.longitude,
            wind_speed_kmh=e.wind_speed_kmh,
            pressure_hpa=e.pressure_hpa,
            rainfall_24h_mm=e.rainfall_24h_mm,
            timestamp=ts,
        ))

    result = filter_cyclones_by_radius(
        events=events,
        user_lat=req.user_latitude,
        user_lon=req.user_longitude,
        radius_km=req.radius_km,
        wind_threshold=req.wind_threshold,
        sort_by=req.sort_by,
        max_results=req.max_results,
    )

    return result.to_dict()


@router.post("/escalation")
async def compute_escalation(req: EscalationRequest):
    """
    Compute the escalation level for a cyclone at a given distance.

    Returns the escalation level and corresponding alert recommendation.
    """
    level = determine_escalation(
        wind_speed_kmh=req.wind_speed_kmh,
        distance_km=req.distance_km,
        rainfall_mm=req.rainfall_mm,
        pressure_hpa=req.pressure_hpa,
        wind_threshold=req.wind_threshold,
    )

    alert = get_cyclone_alert(level)

    return {
        "escalation_level": level.value,
        "alert": alert,
        "inputs": {
            "wind_speed_kmh": req.wind_speed_kmh,
            "distance_km": req.distance_km,
            "rainfall_mm": req.rainfall_mm,
            "pressure_hpa": req.pressure_hpa,
        },
    }


@router.get("/categories")
async def get_categories():
    """
    Return the IMD tropical cyclone classification table.
    """
    return {
        "classification": "India Meteorological Department (IMD)",
        "scale": [
            {"category": "Low Pressure Area", "wind_kmh": "< 31", "wind_knots": "< 17"},
            {"category": "Depression (D)", "wind_kmh": "31–49", "wind_knots": "17–27"},
            {"category": "Deep Depression (DD)", "wind_kmh": "50–61", "wind_knots": "28–33"},
            {"category": "Cyclonic Storm (CS)", "wind_kmh": "62–88", "wind_knots": "34–47"},
            {"category": "Severe Cyclonic Storm (SCS)", "wind_kmh": "89–117", "wind_knots": "48–63"},
            {"category": "Very Severe CS (VSCS)", "wind_kmh": "118–166", "wind_knots": "64–89"},
            {"category": "Extremely Severe CS (ESCS)", "wind_kmh": "167–221", "wind_knots": "90–119"},
            {"category": "Super Cyclonic Storm (SuCS)", "wind_kmh": "> 221", "wind_knots": "> 119"},
        ],
        "note": "Based on 3-minute sustained wind speed. WMO/Atlantic uses 1-minute sustained.",
    }


@router.get("/thresholds")
async def get_thresholds():
    """
    Return current cyclone threshold configuration and tuning guidance.
    """
    return {
        "defaults": {
            "wind_threshold_kmh": DEFAULT_WIND_THRESHOLD_KMH,
            "rainfall_threshold_mm": DEFAULT_RAINFALL_THRESHOLD_MM,
            "pressure_baseline_hpa": DEFAULT_PRESSURE_BASELINE_HPA,
        },
        "escalation_radii": {
            "watch": "< 500 km",
            "warning": "< 300 km",
            "alert": "< 150 km",
            "critical": "< 50 km",
        },
        "risk_weights": {
            "wind": 0.35,
            "rainfall": 0.25,
            "distance": 0.25,
            "pressure": 0.15,
        },
        "tuning_guidance": [
            "Coastal areas: lower wind threshold (e.g., 50 km/h)",
            "Inland areas: higher threshold acceptable (e.g., 90 km/h)",
            "Monsoon season: lower rainfall threshold (easier saturation)",
            "Weak structures: lower thresholds across all parameters",
            "Target: false alarm rate < 30%, miss rate < 5%",
        ],
    }
