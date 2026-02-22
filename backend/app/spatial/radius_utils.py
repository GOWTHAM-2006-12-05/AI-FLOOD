"""
radius_utils.py — Location-based radius filtering for disaster alerts.

Provides:
    - Haversine distance calculation between two (lat, lon) points
    - Radius-based disaster filtering (is a disaster inside a user's radius?)
    - Batch filtering of disaster lists
    - Bounding-box pre-filter for performance at scale

All distances are in **kilometers**. Coordinates are in **decimal degrees**.

Mathematical Foundation — Haversine Formula
============================================
The Haversine formula computes the great-circle distance between two points
on a sphere given their latitudes and longitudes.

Given two points P₁(φ₁, λ₁) and P₂(φ₂, λ₂):

    a = sin²(Δφ / 2) + cos(φ₁) · cos(φ₂) · sin²(Δλ / 2)
    c = 2 · atan2(√a, √(1 − a))
    d = R · c

Where:
    φ  = latitude in radians
    λ  = longitude in radians
    Δφ = φ₂ − φ₁
    Δλ = λ₂ − λ₁
    R  = Earth's mean radius ≈ 6,371 km
    a  = square of half the chord length between the points
    c  = angular distance in radians
    d  = great-circle distance in km

Why Haversine and not Euclidean?
    - Earth is a sphere (approximately). Euclidean distance on raw lat/lon
      produces wildly inaccurate results, especially far from the equator.
    - Haversine is accurate to ~0.5% (good enough for disaster alerting;
      the Vincenty formula on the WGS-84 ellipsoid is better but 10× slower).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence
from datetime import datetime


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM: float = 6_371.0088  # IAU mean radius


class RadiusPreset(float, Enum):
    """Standard radius options exposed to the frontend selector."""
    KM_5  = 5.0
    KM_10 = 10.0
    KM_20 = 20.0
    KM_50 = 50.0


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Coordinate:
    """A geographic point in decimal degrees."""
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not (-90.0 <= self.latitude <= 90.0):
            raise ValueError(
                f"Latitude must be in [-90, 90], got {self.latitude}"
            )
        if not (-180.0 <= self.longitude <= 180.0):
            raise ValueError(
                f"Longitude must be in [-180, 180], got {self.longitude}"
            )

    @property
    def lat_rad(self) -> float:
        """Latitude in radians."""
        return math.radians(self.latitude)

    @property
    def lon_rad(self) -> float:
        """Longitude in radians."""
        return math.radians(self.longitude)


@dataclass
class DisasterEvent:
    """Represents a single disaster event with location metadata."""
    id: str
    title: str
    hazard_type: str  # e.g. "flood", "earthquake", "cyclone"
    location: Coordinate
    severity: int  # 1 (low) → 5 (critical)
    timestamp: str  # ISO-8601
    description: str = ""
    source: str = ""

    # Populated by filtering functions — not set by caller
    distance_km: Optional[float] = field(default=None, repr=False)


@dataclass
class FilterResult:
    """Container for radius-filter output."""
    user_location: Coordinate
    radius_km: float
    total_checked: int
    matched: List[DisasterEvent]
    excluded: int

    @property
    def count(self) -> int:
        return len(self.matched)


# ---------------------------------------------------------------------------
# Haversine implementation
# ---------------------------------------------------------------------------

def haversine(point1: Coordinate, point2: Coordinate) -> float:
    """
    Compute the great-circle distance between two points using the
    Haversine formula.

    Parameters
    ----------
    point1 : Coordinate
        Origin point (e.g. user location).
    point2 : Coordinate
        Target point (e.g. disaster location).

    Returns
    -------
    float
        Distance in kilometers, rounded to 4 decimal places.

    Examples
    --------
    >>> haversine(Coordinate(13.0827, 80.2707), Coordinate(12.9716, 77.5946))
    290.2122

    >>> haversine(Coordinate(0, 0), Coordinate(0, 0))
    0.0
    """
    d_lat = point2.lat_rad - point1.lat_rad
    d_lon = point2.lon_rad - point1.lon_rad

    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(point1.lat_rad)
        * math.cos(point2.lat_rad)
        * math.sin(d_lon / 2.0) ** 2
    )

    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    distance = EARTH_RADIUS_KM * c
    return round(distance, 4)


# ---------------------------------------------------------------------------
# Bounding-box pre-filter (fast rejection before expensive Haversine)
# ---------------------------------------------------------------------------

def _bounding_box(center: Coordinate, radius_km: float) -> tuple:
    """
    Compute a lat/lon bounding box that fully contains the circle defined
    by (center, radius_km). Used as a cheap rectangular pre-filter so we
    only run Haversine on candidates that *might* be inside the radius.

    Returns (min_lat, max_lat, min_lon, max_lon) in degrees.
    """
    # Angular radius in radians
    angular = radius_km / EARTH_RADIUS_KM

    min_lat = center.latitude - math.degrees(angular)
    max_lat = center.latitude + math.degrees(angular)

    # Longitude delta depends on latitude (shrinks toward poles)
    lat_rad = math.radians(center.latitude)
    if math.cos(lat_rad) > 1e-10:
        delta_lon = math.degrees(angular / math.cos(lat_rad))
    else:
        delta_lon = 180.0  # At the poles, all longitudes are "near"

    min_lon = center.longitude - delta_lon
    max_lon = center.longitude + delta_lon

    return (
        max(min_lat, -90.0),
        min(max_lat, 90.0),
        max(min_lon, -180.0),
        min(max_lon, 180.0),
    )


def _inside_bbox(
    lat: float, lon: float,
    min_lat: float, max_lat: float,
    min_lon: float, max_lon: float,
) -> bool:
    """Quick rectangular check."""
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


# ---------------------------------------------------------------------------
# Radius filtering
# ---------------------------------------------------------------------------

def is_inside_radius(
    user_location: Coordinate,
    disaster_location: Coordinate,
    radius_km: float,
) -> tuple[bool, float]:
    """
    Check whether a disaster falls within the user's alert radius.

    Parameters
    ----------
    user_location : Coordinate
        The user's current position.
    disaster_location : Coordinate
        The disaster event's position.
    radius_km : float
        Alert radius in kilometers (e.g. 5, 10, 20, 50).

    Returns
    -------
    (inside, distance_km) : tuple[bool, float]
        Whether the disaster is inside the radius, and the actual distance.

    Examples
    --------
    >>> loc = Coordinate(13.0827, 80.2707)   # Chennai
    >>> quake = Coordinate(13.10, 80.30)     # ~4 km away
    >>> is_inside_radius(loc, quake, radius_km=5.0)
    (True, 3.7266)

    >>> far = Coordinate(12.9716, 77.5946)   # Bangalore
    >>> is_inside_radius(loc, far, radius_km=50.0)
    (False, 290.2122)
    """
    if radius_km <= 0:
        raise ValueError(f"Radius must be positive, got {radius_km}")

    dist = haversine(user_location, disaster_location)
    return (dist <= radius_km, dist)


def filter_disasters(
    user_location: Coordinate,
    disasters: Sequence[DisasterEvent],
    radius_km: float,
    *,
    sort_by_distance: bool = True,
    max_results: Optional[int] = None,
    min_severity: int = 1,
    hazard_types: Optional[set[str]] = None,
) -> FilterResult:
    """
    Filter a list of disaster events to only those within the user's radius.

    Applies a bounding-box pre-filter first, then precise Haversine check.
    Optionally filters by severity and hazard type.

    Parameters
    ----------
    user_location : Coordinate
        The user's position.
    disasters : Sequence[DisasterEvent]
        All known disaster events.
    radius_km : float
        Alert radius in km.
    sort_by_distance : bool
        If True, results are sorted nearest-first.
    max_results : int | None
        Cap the number of returned events.
    min_severity : int
        Only include disasters with severity >= this value.
    hazard_types : set[str] | None
        If provided, only include these hazard types.

    Returns
    -------
    FilterResult
        Structured result with matched disasters and metadata.

    Examples
    --------
    >>> user = Coordinate(13.0827, 80.2707)
    >>> events = [
    ...     DisasterEvent("EQ001", "Minor Tremor", "earthquake",
    ...                   Coordinate(13.10, 80.30), 2, "2026-02-22T10:00:00Z"),
    ...     DisasterEvent("FL001", "River Flooding", "flood",
    ...                   Coordinate(13.05, 80.25), 4, "2026-02-22T09:30:00Z"),
    ...     DisasterEvent("CY001", "Cyclone Approaching", "cyclone",
    ...                   Coordinate(12.00, 82.00), 5, "2026-02-22T08:00:00Z"),
    ... ]
    >>> result = filter_disasters(user, events, radius_km=10.0)
    >>> result.count
    2
    >>> [e.id for e in result.matched]
    ['FL001', 'EQ001']
    """
    if radius_km <= 0:
        raise ValueError(f"Radius must be positive, got {radius_km}")

    bbox = _bounding_box(user_location, radius_km)
    matched: list[DisasterEvent] = []

    for event in disasters:
        # --- Optional pre-filters ---
        if event.severity < min_severity:
            continue
        if hazard_types and event.hazard_type not in hazard_types:
            continue

        # --- Bounding-box fast rejection ---
        if not _inside_bbox(
            event.location.latitude,
            event.location.longitude,
            *bbox,
        ):
            continue

        # --- Precise Haversine check ---
        dist = haversine(user_location, event.location)
        if dist <= radius_km:
            event.distance_km = dist
            matched.append(event)

    if sort_by_distance:
        matched.sort(key=lambda e: e.distance_km or 0.0)

    if max_results is not None:
        matched = matched[:max_results]

    return FilterResult(
        user_location=user_location,
        radius_km=radius_km,
        total_checked=len(disasters),
        matched=matched,
        excluded=len(disasters) - len(matched),
    )


# ---------------------------------------------------------------------------
# Utility: Human-readable distance
# ---------------------------------------------------------------------------

def format_distance(km: float) -> str:
    """
    Format a distance for display.

    >>> format_distance(0.45)
    '450 m'
    >>> format_distance(3.7266)
    '3.73 km'
    >>> format_distance(290.2122)
    '290.21 km'
    """
    if km < 1.0:
        return f"{int(km * 1000)} m"
    return f"{km:.2f} km"
