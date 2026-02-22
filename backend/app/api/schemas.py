"""
Pydantic schemas for the radius-filtering API.

Separated from the route handler so they are reusable across
the codebase (WebSocket handlers, background workers, tests).
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RadiusOption(float, Enum):
    """Radius presets that mirror the frontend selector."""
    KM_5  = 5.0
    KM_10 = 10.0
    KM_20 = 20.0
    KM_50 = 50.0


class HazardType(str, Enum):
    FLOOD      = "flood"
    EARTHQUAKE = "earthquake"
    CYCLONE    = "cyclone"
    HEATWAVE   = "heatwave"
    LANDSLIDE  = "landslide"
    WILDFIRE   = "wildfire"


class Severity(int, Enum):
    LOW      = 1
    MODERATE = 2
    HIGH     = 3
    SEVERE   = 4
    CRITICAL = 5


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class LocationInput(BaseModel):
    """
    Accepts location from either browser geolocation or manual entry.
    The frontend sends whichever the user chose — the API treats them
    identically.
    """
    latitude: float = Field(
        ..., ge=-90.0, le=90.0,
        description="Latitude in decimal degrees",
        examples=[13.0827],
    )
    longitude: float = Field(
        ..., ge=-180.0, le=180.0,
        description="Longitude in decimal degrees",
        examples=[80.2707],
    )
    source: str = Field(
        default="manual",
        description="How the location was obtained: 'gps' | 'manual'",
        examples=["gps"],
    )


class NearbyDisastersRequest(BaseModel):
    """Request body for POST /api/v1/disasters/nearby."""
    location: LocationInput
    radius_km: RadiusOption = Field(
        default=RadiusOption.KM_10,
        description="Alert radius in kilometers",
    )
    min_severity: int = Field(
        default=1, ge=1, le=5,
        description="Minimum severity to include (1-5)",
    )
    hazard_types: Optional[List[HazardType]] = Field(
        default=None,
        description="Filter to these hazard types only (null = all)",
    )
    max_results: Optional[int] = Field(
        default=50, ge=1, le=200,
        description="Maximum number of results to return",
    )


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class CoordinateOut(BaseModel):
    latitude: float
    longitude: float


class DisasterOut(BaseModel):
    """A single disaster event in the response."""
    id: str
    title: str
    hazard_type: HazardType
    severity: int = Field(..., ge=1, le=5)
    location: CoordinateOut
    distance_km: float = Field(
        ..., description="Distance from user in km"
    )
    distance_display: str = Field(
        ..., description="Human-readable distance string"
    )
    timestamp: str
    description: str = ""
    source: str = ""


class NearbyDisastersResponse(BaseModel):
    """Response for POST /api/v1/disasters/nearby."""
    user_location: CoordinateOut
    radius_km: float
    total_checked: int
    results_count: int
    excluded_count: int
    disasters: List[DisasterOut]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    module: str = "radius-filter"
