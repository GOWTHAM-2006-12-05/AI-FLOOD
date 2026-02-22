"""
FastAPI route: Nearby Disasters — radius-based filtering endpoint.

Integrates the radius_utils module with the FastAPI HTTP layer.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException, Query

from backend.app.api.schemas import (
    CoordinateOut,
    DisasterOut,
    HealthResponse,
    NearbyDisastersRequest,
    NearbyDisastersResponse,
)
from backend.app.spatial.radius_utils import (
    Coordinate,
    DisasterEvent,
    filter_disasters,
    format_distance,
    haversine,
    is_inside_radius,
)

router = APIRouter(prefix="/api/v1", tags=["disasters"])


# ---------------------------------------------------------------------------
# Mock data source (replace with real DB query in production)
# ---------------------------------------------------------------------------

_MOCK_DISASTERS: List[DisasterEvent] = [
    DisasterEvent(
        id="EQ-2026-0041",
        title="Minor Tremor near Tambaram",
        hazard_type="earthquake",
        location=Coordinate(12.9249, 80.1000),
        severity=2,
        timestamp="2026-02-22T06:14:00Z",
        description="M 3.1 tremor detected at 8 km depth.",
        source="USGS",
    ),
    DisasterEvent(
        id="FL-2026-0112",
        title="Adyar River Flooding",
        hazard_type="flood",
        location=Coordinate(13.0067, 80.2565),
        severity=4,
        timestamp="2026-02-22T09:30:00Z",
        description="Water level crossed danger mark at Adyar bridge.",
        source="CWC",
    ),
    DisasterEvent(
        id="CY-2026-0007",
        title="Cyclone DANA — Category 2",
        hazard_type="cyclone",
        location=Coordinate(11.5000, 82.0000),
        severity=5,
        timestamp="2026-02-22T03:00:00Z",
        description="Eye located 300 km SE of Chennai, moving NW at 15 km/h.",
        source="IMD",
    ),
    DisasterEvent(
        id="HW-2026-0023",
        title="Severe Heatwave Warning",
        hazard_type="heatwave",
        location=Coordinate(13.0878, 80.2785),
        severity=3,
        timestamp="2026-02-22T12:00:00Z",
        description="Temperature forecast: 44°C. Stay hydrated.",
        source="IMD",
    ),
    DisasterEvent(
        id="LS-2026-0005",
        title="Landslide Risk — Nilgiris",
        hazard_type="landslide",
        location=Coordinate(11.4102, 76.6950),
        severity=4,
        timestamp="2026-02-22T07:45:00Z",
        description="Heavy rainfall + steep slope. Avoid hill roads.",
        source="GSI",
    ),
    DisasterEvent(
        id="FL-2026-0113",
        title="Waterlogging at T. Nagar",
        hazard_type="flood",
        location=Coordinate(13.0418, 80.2341),
        severity=2,
        timestamp="2026-02-22T10:15:00Z",
        description="Knee-deep water on Usman Road due to blocked drains.",
        source="GCC",
    ),
]


def _get_disasters() -> List[DisasterEvent]:
    """
    In production, this queries PostgreSQL/PostGIS.
    For now it returns the mock list.
    """
    return _MOCK_DISASTERS


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/disasters/nearby",
    response_model=NearbyDisastersResponse,
    summary="Get disasters near a location",
    description=(
        "Accepts a user location (from GPS or manual input) and a radius, "
        "returns all active disasters within that radius sorted by distance."
    ),
)
async def get_nearby_disasters(body: NearbyDisastersRequest):
    """
    Core radius-filtering endpoint.

    **Flow:**
    1. Parse user location + radius from request body
    2. Fetch all active disasters (mock data or DB)
    3. Run `filter_disasters()` — bounding-box pre-filter → Haversine
    4. Return matched disasters with distances
    """
    user_loc = Coordinate(
        latitude=body.location.latitude,
        longitude=body.location.longitude,
    )

    hazard_set = (
        {h.value for h in body.hazard_types}
        if body.hazard_types
        else None
    )

    result = filter_disasters(
        user_location=user_loc,
        disasters=_get_disasters(),
        radius_km=body.radius_km.value,
        sort_by_distance=True,
        max_results=body.max_results,
        min_severity=body.min_severity,
        hazard_types=hazard_set,
    )

    disaster_list = [
        DisasterOut(
            id=e.id,
            title=e.title,
            hazard_type=e.hazard_type,
            severity=e.severity,
            location=CoordinateOut(
                latitude=e.location.latitude,
                longitude=e.location.longitude,
            ),
            distance_km=round(e.distance_km, 2),
            distance_display=format_distance(e.distance_km),
            timestamp=e.timestamp,
            description=e.description,
            source=e.source,
        )
        for e in result.matched
    ]

    return NearbyDisastersResponse(
        user_location=CoordinateOut(
            latitude=user_loc.latitude,
            longitude=user_loc.longitude,
        ),
        radius_km=result.radius_km,
        total_checked=result.total_checked,
        results_count=result.count,
        excluded_count=result.excluded,
        disasters=disaster_list,
    )


@router.get(
    "/disasters/distance",
    summary="Calculate distance between two points",
    description="Utility endpoint to compute Haversine distance.",
)
async def get_distance(
    lat1: float = Query(..., ge=-90, le=90, description="Origin latitude"),
    lon1: float = Query(..., ge=-180, le=180, description="Origin longitude"),
    lat2: float = Query(..., ge=-90, le=90, description="Target latitude"),
    lon2: float = Query(..., ge=-180, le=180, description="Target longitude"),
):
    """Return the great-circle distance in km between two coordinates."""
    p1 = Coordinate(lat1, lon1)
    p2 = Coordinate(lat2, lon2)
    dist = haversine(p1, p2)
    return {
        "from": {"latitude": lat1, "longitude": lon1},
        "to": {"latitude": lat2, "longitude": lon2},
        "distance_km": dist,
        "distance_display": format_distance(dist),
    }


@router.get(
    "/disasters/check-radius",
    summary="Check if a point is inside a radius",
)
async def check_radius(
    user_lat: float = Query(..., ge=-90, le=90),
    user_lon: float = Query(..., ge=-180, le=180),
    target_lat: float = Query(..., ge=-90, le=90),
    target_lon: float = Query(..., ge=-180, le=180),
    radius_km: float = Query(10.0, gt=0),
):
    """Check whether a target point falls within a radius from the user."""
    user = Coordinate(user_lat, user_lon)
    target = Coordinate(target_lat, target_lon)
    inside, dist = is_inside_radius(user, target, radius_km)
    return {
        "inside_radius": inside,
        "distance_km": dist,
        "distance_display": format_distance(dist),
        "radius_km": radius_km,
    }


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()
