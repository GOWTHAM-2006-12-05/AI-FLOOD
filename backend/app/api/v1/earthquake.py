"""
FastAPI earthquake monitoring endpoints.

Endpoints:
    GET  /api/v1/earthquake/feed       — Fetch USGS earthquake feed
    POST /api/v1/earthquake/query      — Custom USGS query with filters
    POST /api/v1/earthquake/nearby     — Radius-based nearby earthquakes
    POST /api/v1/earthquake/monitor    — Full monitoring pipeline
    POST /api/v1/earthquake/impact     — Estimate impact radius for a quake
    POST /api/v1/earthquake/depth      — Depth analysis for a set of events
    GET  /api/v1/earthquake/thresholds — Current threshold information
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.ml.earthquake_service import (
    DEFAULT_MAX_RADIUS_KM,
    DEFAULT_MIN_MAGNITUDE,
    DepthClass,
    EarthquakeRisk,
    FeedType,
    TimePeriod,
    analyze_depth_distribution,
    estimate_impact_radius,
    fetch_earthquakes_feed,
    fetch_earthquakes_query,
    filter_earthquakes_by_radius,
    monitor_earthquakes,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/earthquake", tags=["earthquake-monitoring"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class NearbyRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="User latitude")
    longitude: float = Field(..., ge=-180, le=180, description="User longitude")
    radius_km: float = Field(500.0, gt=0, le=20000, description="Search radius in km")
    min_magnitude: float = Field(4.0, ge=0, le=10, description="Minimum magnitude")
    period: str = Field("day", description="Time period: hour, day, week, month")
    feed_type: str = Field("4.5", description="Feed type: significant, 4.5, 2.5, 1.0, all")
    depth_class: Optional[str] = Field(None, description="Filter by depth: shallow, intermediate, deep")
    sort_by: str = Field("distance", description="Sort: distance or magnitude")
    max_results: int = Field(50, ge=1, le=500)


class MonitorRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    radius_km: float = Field(500.0, gt=0, le=20000)
    min_magnitude: float = Field(4.0, ge=0, le=10)
    period: str = Field("day")
    feed_type: str = Field("4.5")


class ImpactRequest(BaseModel):
    magnitude: float = Field(..., ge=0, le=10, description="Earthquake magnitude")
    depth_km: float = Field(..., ge=0, le=700, description="Focal depth in km")


class QueryRequest(BaseModel):
    start_time: Optional[str] = Field(None, description="ISO-8601 start date")
    end_time: Optional[str] = Field(None, description="ISO-8601 end date")
    min_magnitude: float = Field(4.0, ge=0, le=10)
    max_magnitude: Optional[float] = Field(None, ge=0, le=10)
    min_depth: Optional[float] = Field(None, ge=0, le=700)
    max_depth: Optional[float] = Field(None, ge=0, le=700)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    max_radius_km: float = Field(500.0, gt=0)
    limit: int = Field(100, ge=1, le=20000)
    order_by: str = Field("time", description="Sort: time or magnitude")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/feed")
async def get_earthquake_feed(
    period: str = Query("day", description="hour, day, week, or month"),
    feed_type: str = Query("4.5", description="significant, 4.5, 2.5, 1.0, or all"),
):
    """
    Fetch the USGS earthquake summary feed.

    Returns GeoJSON-parsed earthquake events for the specified time window
    and magnitude band.
    """
    try:
        tp = TimePeriod(period)
    except ValueError:
        raise HTTPException(400, f"Invalid period: {period}. Use: hour, day, week, month")

    try:
        ft = FeedType(feed_type)
    except ValueError:
        raise HTTPException(400, f"Invalid feed_type: {feed_type}. Use: significant, 4.5, 2.5, 1.0, all")

    result = fetch_earthquakes_feed(tp, ft)
    if not result.success:
        raise HTTPException(502, f"USGS API error: {result.error}")

    return result.to_dict()


@router.post("/query")
async def query_earthquakes(req: QueryRequest):
    """
    Custom query against the USGS FDSN Event Web Service.

    Supports temporal, spatial, magnitude, and depth filters.
    """
    result = fetch_earthquakes_query(
        start_time=req.start_time,
        end_time=req.end_time,
        min_magnitude=req.min_magnitude,
        max_magnitude=req.max_magnitude,
        min_depth=req.min_depth,
        max_depth=req.max_depth,
        latitude=req.latitude,
        longitude=req.longitude,
        max_radius_km=req.max_radius_km,
        limit=req.limit,
        order_by=req.order_by,
    )
    if not result.success:
        raise HTTPException(502, f"USGS API error: {result.error}")

    return result.to_dict()


@router.post("/nearby")
async def get_nearby_earthquakes(req: NearbyRequest):
    """
    Find earthquakes near a user's location.

    Fetches from USGS feed, then applies Haversine radius filtering,
    magnitude threshold, and optional depth class filter.
    """
    try:
        tp = TimePeriod(req.period)
    except ValueError:
        raise HTTPException(400, f"Invalid period: {req.period}")

    try:
        ft = FeedType(req.feed_type)
    except ValueError:
        raise HTTPException(400, f"Invalid feed_type: {req.feed_type}")

    # Parse depth class filter
    depth_filter = None
    if req.depth_class:
        try:
            depth_filter = DepthClass(req.depth_class)
        except ValueError:
            raise HTTPException(400, f"Invalid depth_class: {req.depth_class}")

    # Fetch
    feed = fetch_earthquakes_feed(tp, ft)
    if not feed.success:
        raise HTTPException(502, f"USGS API error: {feed.error}")

    # Filter
    filtered = filter_earthquakes_by_radius(
        events=feed.events,
        user_lat=req.latitude,
        user_lon=req.longitude,
        radius_km=req.radius_km,
        min_magnitude=req.min_magnitude,
        depth_class_filter=depth_filter,
        sort_by=req.sort_by,
        max_results=req.max_results,
    )

    return filtered.to_dict()


@router.post("/monitor")
async def monitor(req: MonitorRequest):
    """
    Full earthquake monitoring pipeline.

    Fetches, filters, analyses depth distribution, identifies highest-risk
    event, and returns a complete monitoring report.
    """
    try:
        tp = TimePeriod(req.period)
    except ValueError:
        raise HTTPException(400, f"Invalid period: {req.period}")

    try:
        ft = FeedType(req.feed_type)
    except ValueError:
        raise HTTPException(400, f"Invalid feed_type: {req.feed_type}")

    result = monitor_earthquakes(
        user_lat=req.latitude,
        user_lon=req.longitude,
        radius_km=req.radius_km,
        min_magnitude=req.min_magnitude,
        period=tp,
        feed_type=ft,
    )

    return result.to_dict()


@router.post("/impact")
async def estimate_impact(req: ImpactRequest):
    """
    Estimate the impact radius for a given earthquake magnitude and depth.

    Returns three concentric zones:
    - Felt radius (MMI ≥ IV)
    - Damage radius (MMI ≥ VI)
    - Severe radius (MMI ≥ VIII)
    """
    impact = estimate_impact_radius(req.magnitude, req.depth_km)

    return {
        "magnitude": req.magnitude,
        "depth_km": req.depth_km,
        "impact": impact.to_dict(),
        "explanation": {
            "formula": "R_felt = 10^((M − 1.0) / 2.0) × depth_factor",
            "depth_factor_formula": "max(0.3, 1.0 − (depth − 10) / 100)",
            "note": "Empirical model calibrated from USGS DYFI data",
        },
    }


@router.get("/thresholds")
async def get_thresholds():
    """
    Return current threshold configuration and tuning guidance.
    """
    return {
        "defaults": {
            "min_magnitude": DEFAULT_MIN_MAGNITUDE,
            "max_radius_km": DEFAULT_MAX_RADIUS_KM,
        },
        "magnitude_scale": {
            "micro": "< 2.5 — generally not felt",
            "minor": "2.5–4.0 — felt locally, rarely causes damage",
            "light": "4.0–5.0 — noticeable shaking, minor damage near epicentre",
            "moderate": "5.0–6.0 — damage to weak structures within ~50 km",
            "strong": "6.0–7.0 — significant damage within ~100 km",
            "major": "7.0–8.0 — serious damage over hundreds of km",
            "great": "≥ 8.0 — devastating over very large areas",
        },
        "depth_classification": {
            "shallow": "0–70 km (most destructive at surface)",
            "intermediate": "70–300 km (subduction zones)",
            "deep": "300–700 km (minimal surface damage)",
        },
        "tuning_guidance": [
            "High-seismicity zones: raise threshold to avoid alert fatigue",
            "Weak building stock: lower threshold for earlier warnings",
            "Urban areas: lower threshold due to higher consequence",
            "Farther events: only relevant at higher magnitudes",
        ],
    }
