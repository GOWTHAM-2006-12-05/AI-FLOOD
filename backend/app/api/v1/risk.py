"""
FastAPI route: Unified Disaster Risk Aggregation endpoint.

Provides a single API call to aggregate flood, earthquake, and cyclone
risks into an overall risk score (0–100%) and risk level.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.ml.risk_aggregator import (
    ALERT_MESSAGES,
    AggregatedRisk,
    OverallRiskLevel,
    aggregate_risk,
)

router = APIRouter(prefix="/api/v1/risk", tags=["risk-aggregation"])


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class RiskAggregationRequest(BaseModel):
    """Input for the unified risk aggregation."""

    latitude: float = Field(
        default=0.0, ge=-90.0, le=90.0,
        description="User latitude in decimal degrees",
        examples=[13.0827],
    )
    longitude: float = Field(
        default=0.0, ge=-180.0, le=180.0,
        description="User longitude in decimal degrees",
        examples=[80.2707],
    )

    # Per-hazard inputs
    flood_probability: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Flood probability from ensemble model (0–1)",
        examples=[0.65],
    )
    earthquake_magnitude: float = Field(
        default=0.0, ge=0.0, le=12.0,
        description="Earthquake magnitude (0 = no earthquake)",
        examples=[5.5],
    )
    earthquake_depth_km: float = Field(
        default=10.0, ge=0.0,
        description="Earthquake focal depth in km",
        examples=[15.0],
    )
    cyclone_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Cyclone composite risk score (0–1)",
        examples=[0.45],
    )

    # Optional: previous level for hysteresis
    previous_level: Optional[str] = Field(
        default=None,
        description="Previous risk level for hysteresis (safe/watch/warning/severe)",
        examples=["watch"],
    )


class HazardBreakdownOut(BaseModel):
    hazard_type: str
    raw_value: float
    normalised_score: float
    weight: float
    weighted_contribution: float
    is_active: bool
    is_critical: bool
    priority: int
    details: Dict[str, Any] = {}


class FormulaComponentsOut(BaseModel):
    R_avg: float
    R_max: float
    beta: float
    R_hybrid: float
    amplifier: float
    weights: Dict[str, float]


class AlertInfoOut(BaseModel):
    title: str
    message: str
    color: str
    icon: str


class RiskAggregationResponse(BaseModel):
    """Full response from the unified risk aggregation."""
    overall_risk_score: float = Field(
        ..., description="Overall risk score 0–100"
    )
    overall_risk_score_pct: str = Field(
        ..., description="Human-readable percentage string"
    )
    overall_risk_level: str = Field(
        ..., description="Risk level: safe / watch / warning / severe"
    )
    alert_action: str = Field(
        ..., description="Recommended action"
    )
    dominant_hazard: str
    active_hazard_count: int
    alert_triggered: bool
    alert_reasons: List[str]
    alert_info: AlertInfoOut
    hazard_breakdown: List[HazardBreakdownOut]
    formula_components: FormulaComponentsOut
    location: Dict[str, float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/aggregate",
    response_model=RiskAggregationResponse,
    summary="Unified Disaster Risk Aggregation",
    description=(
        "Combines flood probability, earthquake severity, and cyclone risk "
        "into a single overall_risk_score (0–100%) and overall_risk_level."
    ),
)
async def aggregate_disaster_risk(req: RiskAggregationRequest):
    """
    Compute the unified disaster risk score.

    Accepts individual hazard values and returns:
    - overall_risk_score (0–100)
    - overall_risk_level (safe / watch / warning / severe)
    - per-hazard breakdown with normalised scores
    - alert trigger information
    - formula components for transparency
    """
    # Parse previous level if provided
    prev_level: Optional[OverallRiskLevel] = None
    if req.previous_level:
        try:
            prev_level = OverallRiskLevel(req.previous_level.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid previous_level: '{req.previous_level}'. "
                       f"Must be one of: safe, watch, warning, severe.",
            )

    result: AggregatedRisk = aggregate_risk(
        flood_probability=req.flood_probability,
        earthquake_magnitude=req.earthquake_magnitude,
        earthquake_depth_km=req.earthquake_depth_km,
        cyclone_score=req.cyclone_score,
        latitude=req.latitude,
        longitude=req.longitude,
        previous_level=prev_level,
    )

    # Build response
    result_dict = result.to_dict()
    result_dict["alert_info"] = ALERT_MESSAGES[result.overall_risk_level]

    return result_dict


@router.get(
    "/thresholds",
    summary="Get Risk Thresholds",
    description="Returns the current escalation thresholds and weight configuration.",
)
async def get_thresholds():
    """Return the tuning parameters used by the aggregation engine."""
    from backend.app.ml.risk_aggregator import (
        ACTIVE_THRESHOLD,
        BETA,
        CYCLONE_CRITICAL,
        EARTHQUAKE_CRITICAL,
        FLOOD_CRITICAL,
        GAMMA,
        HYSTERESIS_BUFFER,
        PRIORITY_MAP,
        THRESHOLD_SAFE_UPPER,
        THRESHOLD_WARNING_UPPER,
        THRESHOLD_WATCH_UPPER,
        W_CYCLONE,
        W_EARTHQUAKE,
        W_FLOOD,
    )

    return {
        "weights": {
            "flood": W_FLOOD,
            "earthquake": W_EARTHQUAKE,
            "cyclone": W_CYCLONE,
        },
        "blending": {
            "beta": BETA,
            "description": "β·R_max + (1−β)·R_avg",
        },
        "concurrency": {
            "gamma": GAMMA,
            "active_threshold": ACTIVE_THRESHOLD,
            "description": "amplifier = 1 + γ·(n_active − 1)",
        },
        "escalation_thresholds": {
            "safe": f"0 – {THRESHOLD_SAFE_UPPER}%",
            "watch": f"{THRESHOLD_SAFE_UPPER} – {THRESHOLD_WATCH_UPPER}%",
            "warning": f"{THRESHOLD_WATCH_UPPER} – {THRESHOLD_WARNING_UPPER}%",
            "severe": f"{THRESHOLD_WARNING_UPPER} – 100%",
        },
        "hysteresis_buffer": HYSTERESIS_BUFFER,
        "critical_thresholds": {
            "flood": FLOOD_CRITICAL,
            "earthquake": EARTHQUAKE_CRITICAL,
            "cyclone": CYCLONE_CRITICAL,
        },
        "priority_order": PRIORITY_MAP,
        "alert_messages": {
            level.value: msg for level, msg in ALERT_MESSAGES.items()
        },
    }


@router.get(
    "/health",
    summary="Risk Aggregation Health Check",
)
async def risk_health():
    return {
        "status": "ok",
        "module": "risk-aggregation",
        "version": "1.0.0",
    }
