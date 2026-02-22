"""
earthquake_service.py — Earthquake monitoring via USGS Earthquake API.

Integrates the USGS Earthquake Hazards Program real-time feed to provide:
    • Live earthquake event ingestion (past hour / day / week / month)
    • Magnitude threshold filtering (configurable)
    • Depth analysis with seismological classification
    • Radius-based spatial filtering (Haversine)
    • Impact radius estimation (empirical + physics-based)
    • Risk level classification

═══════════════════════════════════════════════════════════════════════════
WHY EARTHQUAKES CANNOT BE PREDICTED
═══════════════════════════════════════════════════════════════════════════

Unlike floods or cyclones, earthquakes are fundamentally **unpredictable**
with current science. Here is why:

1. CHAOTIC FAULT DYNAMICS
   Earthquakes originate from sudden slip on faults. The stress state along
   a fault is governed by highly nonlinear friction laws (rate-and-state
   friction). Tiny perturbations — a pressure change in pore fluid, a
   micro-crack propagation — can trigger a cascade whose final magnitude
   is unknowable until it stops. This is a classic example of **self-
   organised criticality**: the system is always near a critical state,
   and the same initial perturbation can produce M2 or M8.

2. INACCESSIBILITY OF FAULT ZONES
   Major faults are 5–700 km deep. We cannot directly observe the stress
   tensor, pore pressure, or frictional properties at depth. Borehole
   measurements sample only pinpoints; faults extend hundreds of km.

3. NO RELIABLE PRECURSORS
   Despite decades of research, no consistent precursor has been found:
   - Foreshocks: Only ~5% of large earthquakes have identifiable foreshocks.
   - Radon gas: Anomalies are inconsistent and non-reproducible.
   - Animal behaviour: Anecdotal; fails controlled studies.
   - GPS strain: Shows long-term loading but not when release occurs.
   - Electromagnetic signals: Contested; lab results don't scale.

4. STATISTICAL LIMITS (Gutenberg-Richter & Omori)
   We CAN say:
   - log₁₀(N) = a − bM  (Gutenberg-Richter: how often each magnitude occurs)
   - n(t) = K / (t + c)^p  (Omori: aftershock rate decays as power law)
   These are statistical, not predictive. They tell us *average rates*,
   not *when* the next event will occur.

5. WHAT WE DO INSTEAD → MONITORING + EARLY WARNING
   - This module: rapid detection & alerting after an earthquake occurs
   - ShakeAlert (USGS): ~5–20 seconds warning via P-wave detection
   - Seismic hazard maps: probabilistic (10% in 50 years) not predictive

Therefore this module is a **monitoring** system, not a prediction system.
We detect earthquakes ASAP after they occur and estimate impact.

═══════════════════════════════════════════════════════════════════════════
IMPACT RADIUS ESTIMATION FORMULA
═══════════════════════════════════════════════════════════════════════════

We estimate the radius of significant shaking (Modified Mercalli Intensity
≥ V, "felt strongly / light damage") using an empirical attenuation model.

The general intensity attenuation relationship:

    I(r) = I₀ − k₁ · log₁₀(r / h) − k₂ · (r − h)

Where:
    I₀   = epicentral intensity (derived from magnitude)
    r    = hypocentral distance = √(Δ² + h²)
    h    = focal depth (km)
    Δ    = epicentral distance (km)
    k₁   = geometric spreading coefficient ≈ 3.0
    k₂   = anelastic attenuation coefficient ≈ 0.0036 (per km)

Epicentral intensity from magnitude (Gutenberg-Richter, 1956):

    I₀ ≈ 1.5 · M − 1.0     (approximate; varies by region)

We solve for the epicentral distance Δ where I(r) drops below threshold
(MMI V ≈ 4.5) numerically. In practice, we use a simplified empirical
formula widely used in seismology:

    R_felt (km) ≈ 10^((M − 1.0) / 2.0)            (rough felt radius)
    R_damage(km) ≈ 10^((M − 3.5) / 1.8)            (structural damage)
    R_severe(km) ≈ 10^((M − 5.0) / 1.5)            (severe / collapse)

These are calibrated from historical earthquake catalogues (e.g., DYFI data).
Depth correction: shallower earthquakes produce larger impact radii:

    depth_factor = max(0.3, 1.0 − (depth_km − 10) / 100)
    R_corrected  = R · depth_factor

═══════════════════════════════════════════════════════════════════════════
MAGNITUDE THRESHOLD TUNING LOGIC
═══════════════════════════════════════════════════════════════════════════

Why configurable thresholds?

    M < 2.5   — Generally not felt. Instruments only. Thousands/day globally.
    2.5 ≤ M < 4.0 — Felt locally. Usually no damage. Hundreds/day.
    4.0 ≤ M < 5.0 — Light shaking. Minor damage possible near epicentre.
    5.0 ≤ M < 6.0 — Moderate. Damage to weak structures within ~50 km.
    6.0 ≤ M < 7.0 — Strong. Significant damage within ~100 km.
    7.0 ≤ M < 8.0 — Major. Serious damage, hundreds of km.
    M ≥ 8.0       — Great. Devastating over very large areas.

Defaults:
    - Public alerts:  min_magnitude=4.0 (damage possible)
    - Research/seismology: min_magnitude=2.5 (felt)
    - Global monitoring: min_magnitude=5.5 (significant worldwide)

Threshold should be tuned based on:
    1. Local seismicity — high-seismicity zones (California, Japan) may use
       higher thresholds to avoid alert fatigue.
    2. Building codes — regions with poor infrastructure lower thresholds.
    3. Population density — urban areas warrant lower thresholds.
    4. Distance — farther events need higher magnitude to be relevant.

USGS API Reference:
    https://earthquake.usgs.gov/fdsnws/event/1/
    GeoJSON feed: https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend.app.spatial.radius_utils import (
    Coordinate,
    haversine,
    EARTH_RADIUS_KM,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

USGS_API_BASE = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_FEED_BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"

# Attenuation model coefficients
K1_GEOMETRIC = 3.0        # geometric spreading
K2_ANELASTIC = 0.0036     # anelastic attenuation (km⁻¹)

# Default thresholds
DEFAULT_MIN_MAGNITUDE = 4.0
DEFAULT_MAX_RADIUS_KM = 500.0

# Retry configuration
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5  # seconds × attempt


# ═══════════════════════════════════════════════════════════════════════════
# Enums & Data Structures
# ═══════════════════════════════════════════════════════════════════════════

class TimePeriod(str, Enum):
    """USGS feed time windows."""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class FeedType(str, Enum):
    """USGS feed magnitude bands."""
    SIGNIFICANT = "significant"  # magnitude ≥ 5.5 (curated)
    M4_5_PLUS   = "4.5"         # M ≥ 4.5
    M2_5_PLUS   = "2.5"         # M ≥ 2.5
    M1_0_PLUS   = "1.0"         # M ≥ 1.0
    ALL         = "all"         # everything


class DepthClass(str, Enum):
    """
    Seismological depth classification.

    Shallow events produce stronger surface shaking for the same magnitude.
    Deep events attenuate more before reaching the surface.
    """
    SHALLOW      = "shallow"       # 0–70 km   (crustal: most destructive)
    INTERMEDIATE = "intermediate"  # 70–300 km (subduction zones)
    DEEP         = "deep"          # 300–700 km (minimal surface damage)


class EarthquakeRisk(str, Enum):
    """Risk level for alerting."""
    NEGLIGIBLE = "negligible"   # M < 3.0 or very deep/far
    LOW        = "low"          # M 3–4.5 or moderate distance
    MODERATE   = "moderate"     # M 4.5–5.5 within region
    HIGH       = "high"         # M 5.5–7.0 nearby
    CRITICAL   = "critical"     # M ≥ 7.0 nearby


def classify_depth(depth_km: float) -> DepthClass:
    """
    Classify earthquake depth.

    Parameters
    ----------
    depth_km : float
        Focal depth in kilometres (always ≥ 0; USGS sometimes reports
        negative values for very shallow events → clamped to 0).

    Returns
    -------
    DepthClass

    Examples
    --------
    >>> classify_depth(10)
    <DepthClass.SHALLOW: 'shallow'>
    >>> classify_depth(150)
    <DepthClass.INTERMEDIATE: 'intermediate'>
    >>> classify_depth(500)
    <DepthClass.DEEP: 'deep'>
    """
    depth_km = max(0.0, depth_km)
    if depth_km <= 70.0:
        return DepthClass.SHALLOW
    elif depth_km <= 300.0:
        return DepthClass.INTERMEDIATE
    else:
        return DepthClass.DEEP


def depth_severity_factor(depth_km: float) -> float:
    """
    Compute a severity multiplier based on depth.

    Shallow earthquakes (< 20 km) are most dangerous.
    Deep earthquakes (> 300 km) rarely cause surface damage.

    Returns a factor in [0.2, 1.5]:
        - Very shallow (< 10 km): 1.5×
        - Shallow (10–70 km): 1.0×
        - Intermediate: 0.6×
        - Deep: 0.2×
    """
    depth_km = max(0.0, depth_km)
    if depth_km < 10.0:
        return 1.5
    elif depth_km <= 70.0:
        return 1.0
    elif depth_km <= 300.0:
        return 0.6
    else:
        return 0.2


# ═══════════════════════════════════════════════════════════════════════════
# Impact Radius Estimation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ImpactEstimate:
    """
    Estimated impact radii for an earthquake.

    Three concentric zones:
        felt_radius_km    — MMI ≥ IV  (felt by most people indoors)
        damage_radius_km  — MMI ≥ VI  (light-to-moderate structural damage)
        severe_radius_km  — MMI ≥ VIII (heavy damage / partial collapse)
    """
    felt_radius_km: float
    damage_radius_km: float
    severe_radius_km: float
    depth_factor: float
    epicentral_intensity: float  # estimated MMI at epicentre

    def to_dict(self) -> Dict[str, Any]:
        return {
            "felt_radius_km": round(self.felt_radius_km, 2),
            "damage_radius_km": round(self.damage_radius_km, 2),
            "severe_radius_km": round(self.severe_radius_km, 2),
            "depth_factor": round(self.depth_factor, 3),
            "epicentral_intensity_mmi": round(self.epicentral_intensity, 1),
        }


def estimate_impact_radius(
    magnitude: float,
    depth_km: float,
) -> ImpactEstimate:
    """
    Estimate concentric impact zones from magnitude and depth.

    Uses empirical power-law relations calibrated from USGS DYFI data:

        R_felt   = 10^((M − 1.0) / 2.0)
        R_damage = 10^((M − 3.5) / 1.8)
        R_severe = 10^((M − 5.0) / 1.5)

    Depth correction (shallower → broader impact):

        depth_factor = max(0.3, 1.0 − (depth_km − 10) / 100)

    Parameters
    ----------
    magnitude : float
        Earthquake magnitude (Richter / moment magnitude).
    depth_km : float
        Focal depth in km.

    Returns
    -------
    ImpactEstimate
        Estimated radii for felt / damage / severe zones.

    Examples
    --------
    >>> est = estimate_impact_radius(6.0, 10.0)
    >>> est.felt_radius_km > est.damage_radius_km > est.severe_radius_km
    True
    """
    depth_km = max(0.0, depth_km)

    # Depth correction: shallow = larger area of effect
    depth_factor = max(0.3, 1.0 - (depth_km - 10.0) / 100.0)
    # Cap at 1.5 for very shallow events
    depth_factor = min(1.5, depth_factor)

    # Empirical radius formulas
    r_felt = 10.0 ** ((magnitude - 1.0) / 2.0) * depth_factor
    r_damage = max(0.0, 10.0 ** ((magnitude - 3.5) / 1.8) * depth_factor)
    r_severe = max(0.0, 10.0 ** ((magnitude - 5.0) / 1.5) * depth_factor)

    # Epicentral intensity (Gutenberg-Richter relation)
    I0 = 1.5 * magnitude - 1.0

    return ImpactEstimate(
        felt_radius_km=round(r_felt, 2),
        damage_radius_km=round(r_damage, 2),
        severe_radius_km=round(r_severe, 2),
        depth_factor=round(depth_factor, 3),
        epicentral_intensity=I0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Earthquake Event
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EarthquakeEvent:
    """Parsed earthquake event from USGS."""

    event_id: str
    title: str
    magnitude: float
    depth_km: float
    latitude: float
    longitude: float
    timestamp: datetime
    place: str
    url: str
    felt_reports: int  # "felt" DYFI count
    tsunami_flag: bool
    alert_level: Optional[str]  # USGS PAGER: green/yellow/orange/red
    status: str  # "automatic" or "reviewed"
    magnitude_type: str  # "ml", "mb", "mw", etc.

    # Computed fields
    depth_class: DepthClass = field(init=False)
    impact: ImpactEstimate = field(init=False)
    risk_level: EarthquakeRisk = field(init=False)
    distance_km: Optional[float] = None  # populated by radius filtering

    def __post_init__(self):
        self.depth_class = classify_depth(self.depth_km)
        self.impact = estimate_impact_radius(self.magnitude, self.depth_km)
        self.risk_level = _classify_earthquake_risk(
            self.magnitude, self.depth_km
        )

    @property
    def coordinate(self) -> Coordinate:
        return Coordinate(self.latitude, self.longitude)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "title": self.title,
            "magnitude": self.magnitude,
            "magnitude_type": self.magnitude_type,
            "depth_km": self.depth_km,
            "depth_class": self.depth_class.value,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timestamp": self.timestamp.isoformat(),
            "place": self.place,
            "url": self.url,
            "felt_reports": self.felt_reports,
            "tsunami_flag": self.tsunami_flag,
            "alert_level": self.alert_level,
            "status": self.status,
            "risk_level": self.risk_level.value,
            "impact": self.impact.to_dict(),
            "distance_km": (
                round(self.distance_km, 2) if self.distance_km else None
            ),
        }


def _classify_earthquake_risk(
    magnitude: float,
    depth_km: float,
) -> EarthquakeRisk:
    """
    Classify earthquake risk from magnitude and depth.

    The classification combines magnitude with depth severity factor:

        effective_magnitude = magnitude × depth_severity_factor(depth)

    Thresholds:
        eff_M < 2.5  → NEGLIGIBLE
        2.5 ≤ eff_M < 4.0  → LOW
        4.0 ≤ eff_M < 5.5  → MODERATE
        5.5 ≤ eff_M < 7.0  → HIGH
        eff_M ≥ 7.0  → CRITICAL
    """
    dsf = depth_severity_factor(depth_km)
    eff_mag = magnitude * dsf

    if eff_mag < 2.5:
        return EarthquakeRisk.NEGLIGIBLE
    elif eff_mag < 4.0:
        return EarthquakeRisk.LOW
    elif eff_mag < 5.5:
        return EarthquakeRisk.MODERATE
    elif eff_mag < 7.0:
        return EarthquakeRisk.HIGH
    else:
        return EarthquakeRisk.CRITICAL


# ═══════════════════════════════════════════════════════════════════════════
# USGS API Integration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class USGSQueryResult:
    """Result container for USGS earthquake query."""
    success: bool
    events: List[EarthquakeEvent]
    total_count: int
    query_params: Dict[str, Any]
    fetch_time_ms: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "total_count": self.total_count,
            "events_returned": len(self.events),
            "query_params": self.query_params,
            "fetch_time_ms": round(self.fetch_time_ms, 1),
            "error": self.error,
            "events": [e.to_dict() for e in self.events],
        }


def _parse_usgs_feature(feature: Dict[str, Any]) -> Optional[EarthquakeEvent]:
    """
    Parse a single GeoJSON feature from the USGS API into an EarthquakeEvent.

    USGS GeoJSON format:
        feature = {
            "type": "Feature",
            "properties": { "mag": 5.2, "place": "...", "time": 1708617600000, ... },
            "geometry": { "type": "Point", "coordinates": [lon, lat, depth_km] },
            "id": "us7000m..."
        }
    """
    try:
        props = feature["properties"]
        geom = feature["geometry"]["coordinates"]  # [lon, lat, depth]

        magnitude = float(props.get("mag", 0.0) or 0.0)
        depth_km = float(geom[2]) if len(geom) > 2 else 0.0
        latitude = float(geom[1])
        longitude = float(geom[0])

        # Timestamp: USGS gives milliseconds since epoch
        ts_ms = props.get("time", 0)
        timestamp = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)

        return EarthquakeEvent(
            event_id=str(feature.get("id", "")),
            title=str(props.get("title", f"M{magnitude} Earthquake")),
            magnitude=magnitude,
            depth_km=max(0.0, depth_km),
            latitude=latitude,
            longitude=longitude,
            timestamp=timestamp,
            place=str(props.get("place", "Unknown")),
            url=str(props.get("url", "")),
            felt_reports=int(props.get("felt", 0) or 0),
            tsunami_flag=bool(props.get("tsunami", 0)),
            alert_level=props.get("alert"),
            status=str(props.get("status", "automatic")),
            magnitude_type=str(props.get("magType", "ml")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Failed to parse USGS feature: %s", exc)
        return None


def fetch_earthquakes_feed(
    period: TimePeriod = TimePeriod.DAY,
    feed_type: FeedType = FeedType.M4_5_PLUS,
    timeout: float = 15.0,
) -> USGSQueryResult:
    """
    Fetch earthquakes from the USGS GeoJSON summary feed.

    This is the simplest approach: pre-built feeds updated every ~5 minutes.

    Parameters
    ----------
    period : TimePeriod
        Time window: hour, day, week, or month.
    feed_type : FeedType
        Magnitude band: significant, 4.5, 2.5, 1.0, or all.
    timeout : float
        HTTP request timeout in seconds.

    Returns
    -------
    USGSQueryResult

    Examples
    --------
    >>> result = fetch_earthquakes_feed(TimePeriod.DAY, FeedType.M4_5_PLUS)
    >>> result.success
    True
    """
    url = f"{USGS_FEED_BASE}/{feed_type.value}_{period.value}.geojson"
    params = {"period": period.value, "feed_type": feed_type.value}

    start = time.monotonic()
    events: List[EarthquakeEvent] = []

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            # Parse features array
            for feat in data.get("features", []):
                event = _parse_usgs_feature(feat)
                if event is not None:
                    events.append(event)

            elapsed = (time.monotonic() - start) * 1000
            return USGSQueryResult(
                success=True,
                events=events,
                total_count=len(events),
                query_params=params,
                fetch_time_ms=elapsed,
            )

        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            logger.warning(
                "USGS feed attempt %d/%d failed: %s",
                attempt, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)

    elapsed = (time.monotonic() - start) * 1000
    return USGSQueryResult(
        success=False,
        events=[],
        total_count=0,
        query_params=params,
        fetch_time_ms=elapsed,
        error="All retry attempts exhausted",
    )


def fetch_earthquakes_query(
    *,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    min_magnitude: float = DEFAULT_MIN_MAGNITUDE,
    max_magnitude: Optional[float] = None,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    max_radius_km: float = DEFAULT_MAX_RADIUS_KM,
    limit: int = 100,
    order_by: str = "time",
    timeout: float = 15.0,
) -> USGSQueryResult:
    """
    Query the USGS FDSN Event Web Service with fine-grained parameters.

    This endpoint offers full control over spatial, temporal, and magnitude
    filters, unlike the pre-built summary feeds.

    Parameters
    ----------
    start_time, end_time : str | None
        ISO-8601 date strings (e.g. "2026-02-01").
    min_magnitude : float
        Minimum magnitude to include (default: 4.0).
    max_magnitude : float | None
        Maximum magnitude.
    min_depth, max_depth : float | None
        Depth range in km.
    latitude, longitude : float | None
        Centre point for spatial query.
    max_radius_km : float
        Search radius in km (requires lat/lon).
    limit : int
        Maximum number of events to return (USGS max: 20000).
    order_by : str
        "time" (newest first) or "magnitude" (largest first).
    timeout : float
        HTTP timeout in seconds.

    Returns
    -------
    USGSQueryResult
    """
    params: Dict[str, Any] = {
        "format": "geojson",
        "minmagnitude": min_magnitude,
        "limit": min(limit, 20000),
        "orderby": order_by,
    }
    if start_time:
        params["starttime"] = start_time
    if end_time:
        params["endtime"] = end_time
    if max_magnitude is not None:
        params["maxmagnitude"] = max_magnitude
    if min_depth is not None:
        params["mindepth"] = min_depth
    if max_depth is not None:
        params["maxdepth"] = max_depth
    if latitude is not None and longitude is not None:
        params["latitude"] = latitude
        params["longitude"] = longitude
        params["maxradiuskm"] = max_radius_km

    # Build URL
    query_string = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{USGS_API_BASE}?{query_string}"

    start = time.monotonic()
    events: List[EarthquakeEvent] = []

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            for feat in data.get("features", []):
                event = _parse_usgs_feature(feat)
                if event is not None:
                    events.append(event)

            elapsed = (time.monotonic() - start) * 1000
            return USGSQueryResult(
                success=True,
                events=events,
                total_count=len(events),
                query_params=params,
                fetch_time_ms=elapsed,
            )

        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            logger.warning(
                "USGS query attempt %d/%d failed: %s",
                attempt, MAX_RETRIES, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)

    elapsed = (time.monotonic() - start) * 1000
    return USGSQueryResult(
        success=False,
        events=[],
        total_count=0,
        query_params=params,
        fetch_time_ms=elapsed,
        error="All retry attempts exhausted",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Radius Filtering & Local Filtering
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EarthquakeFilterResult:
    """Result of radius-based earthquake filtering."""
    user_location: Coordinate
    radius_km: float
    total_checked: int
    matched: List[EarthquakeEvent]
    excluded: int
    min_magnitude_used: float

    @property
    def count(self) -> int:
        return len(self.matched)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_location": {
                "latitude": self.user_location.latitude,
                "longitude": self.user_location.longitude,
            },
            "radius_km": self.radius_km,
            "total_checked": self.total_checked,
            "matched_count": self.count,
            "excluded": self.excluded,
            "min_magnitude_used": self.min_magnitude_used,
            "events": [e.to_dict() for e in self.matched],
        }


def filter_earthquakes_by_radius(
    events: List[EarthquakeEvent],
    user_lat: float,
    user_lon: float,
    radius_km: float = DEFAULT_MAX_RADIUS_KM,
    min_magnitude: float = DEFAULT_MIN_MAGNITUDE,
    depth_class_filter: Optional[DepthClass] = None,
    sort_by: str = "distance",
    max_results: int = 50,
) -> EarthquakeFilterResult:
    """
    Filter earthquake events by distance from a user's location.

    Uses Haversine distance for accurate km-based filtering.

    Parameters
    ----------
    events : list[EarthquakeEvent]
        Events to filter.
    user_lat, user_lon : float
        User's GPS coordinates.
    radius_km : float
        Maximum distance from user to include.
    min_magnitude : float
        Minimum magnitude threshold.
    depth_class_filter : DepthClass | None
        If given, only include earthquakes of this depth class.
    sort_by : str
        "distance" (nearest first) or "magnitude" (largest first).
    max_results : int
        Cap on returned events.

    Returns
    -------
    EarthquakeFilterResult
    """
    user_coord = Coordinate(user_lat, user_lon)
    matched: List[EarthquakeEvent] = []

    for event in events:
        # Magnitude filter
        if event.magnitude < min_magnitude:
            continue

        # Depth class filter
        if depth_class_filter and event.depth_class != depth_class_filter:
            continue

        # Distance filter (Haversine)
        dist = haversine(user_coord, event.coordinate)
        if dist <= radius_km:
            event.distance_km = dist
            matched.append(event)

    # Sort
    if sort_by == "magnitude":
        matched.sort(key=lambda e: e.magnitude, reverse=True)
    else:
        matched.sort(key=lambda e: e.distance_km or 0.0)

    matched = matched[:max_results]

    return EarthquakeFilterResult(
        user_location=user_coord,
        radius_km=radius_km,
        total_checked=len(events),
        matched=matched,
        excluded=len(events) - len(matched),
        min_magnitude_used=min_magnitude,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Depth Analysis
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DepthAnalysis:
    """Statistical depth analysis for a set of earthquakes."""
    total_events: int
    shallow_count: int
    intermediate_count: int
    deep_count: int
    mean_depth_km: float
    median_depth_km: float
    max_depth_km: float
    min_depth_km: float
    depth_std_km: float
    shallow_fraction: float
    depth_magnitude_correlation: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_events": self.total_events,
            "distribution": {
                "shallow": self.shallow_count,
                "intermediate": self.intermediate_count,
                "deep": self.deep_count,
            },
            "shallow_fraction": round(self.shallow_fraction, 3),
            "mean_depth_km": round(self.mean_depth_km, 1),
            "median_depth_km": round(self.median_depth_km, 1),
            "max_depth_km": round(self.max_depth_km, 1),
            "min_depth_km": round(self.min_depth_km, 1),
            "depth_std_km": round(self.depth_std_km, 1),
            "depth_magnitude_correlation": (
                round(self.depth_magnitude_correlation, 3)
                if self.depth_magnitude_correlation is not None
                else None
            ),
        }


def analyze_depth_distribution(
    events: List[EarthquakeEvent],
) -> DepthAnalysis:
    """
    Perform statistical analysis on earthquake depth distribution.

    This is useful for characterising a seismic zone:
        - Subduction zones → mix of shallow + intermediate + deep
        - Mid-ocean ridges → almost entirely shallow (< 20 km)
        - Intraplate → shallow, sparse

    Parameters
    ----------
    events : list[EarthquakeEvent]
        Earthquakes to analyse.

    Returns
    -------
    DepthAnalysis

    Examples
    --------
    >>> events = [EarthquakeEvent(..., depth_km=15), ...]
    >>> analysis = analyze_depth_distribution(events)
    >>> analysis.shallow_count
    1
    """
    if not events:
        return DepthAnalysis(
            total_events=0,
            shallow_count=0, intermediate_count=0, deep_count=0,
            mean_depth_km=0, median_depth_km=0,
            max_depth_km=0, min_depth_km=0, depth_std_km=0,
            shallow_fraction=0, depth_magnitude_correlation=None,
        )

    depths = [e.depth_km for e in events]
    mags = [e.magnitude for e in events]
    n = len(depths)

    # Classification counts
    shallow = sum(1 for d in depths if d <= 70)
    intermediate = sum(1 for d in depths if 70 < d <= 300)
    deep = sum(1 for d in depths if d > 300)

    # Statistics
    mean_d = sum(depths) / n
    sorted_d = sorted(depths)
    median_d = sorted_d[n // 2] if n % 2 else (sorted_d[n // 2 - 1] + sorted_d[n // 2]) / 2
    std_d = (sum((d - mean_d) ** 2 for d in depths) / n) ** 0.5

    # Pearson correlation between depth and magnitude
    corr = None
    if n >= 3:
        mean_m = sum(mags) / n
        std_m = (sum((m - mean_m) ** 2 for m in mags) / n) ** 0.5
        if std_d > 0 and std_m > 0:
            cov = sum((d - mean_d) * (m - mean_m) for d, m in zip(depths, mags)) / n
            corr = cov / (std_d * std_m)

    return DepthAnalysis(
        total_events=n,
        shallow_count=shallow,
        intermediate_count=intermediate,
        deep_count=deep,
        mean_depth_km=mean_d,
        median_depth_km=median_d,
        max_depth_km=max(depths),
        min_depth_km=min(depths),
        depth_std_km=std_d,
        shallow_fraction=shallow / n,
        depth_magnitude_correlation=corr,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: Combined Fetch + Filter + Analyse
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EarthquakeMonitorResult:
    """Full monitoring result combining fetch, filter, depth analysis, impact."""
    query_result: USGSQueryResult
    filter_result: EarthquakeFilterResult
    depth_analysis: DepthAnalysis
    highest_risk_event: Optional[EarthquakeEvent]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "query": {
                "success": self.query_result.success,
                "total_fetched": self.query_result.total_count,
                "fetch_time_ms": round(self.query_result.fetch_time_ms, 1),
            },
            "filter": self.filter_result.to_dict(),
            "depth_analysis": self.depth_analysis.to_dict(),
            "highest_risk": (
                self.highest_risk_event.to_dict()
                if self.highest_risk_event else None
            ),
        }


def monitor_earthquakes(
    user_lat: float,
    user_lon: float,
    radius_km: float = DEFAULT_MAX_RADIUS_KM,
    min_magnitude: float = DEFAULT_MIN_MAGNITUDE,
    period: TimePeriod = TimePeriod.DAY,
    feed_type: FeedType = FeedType.M4_5_PLUS,
) -> EarthquakeMonitorResult:
    """
    End-to-end earthquake monitoring: fetch → filter → analyse → rank.

    Parameters
    ----------
    user_lat, user_lon : float
        User's location.
    radius_km : float
        Alert radius in km.
    min_magnitude : float
        Minimum magnitude to report.
    period : TimePeriod
        Time window for USGS feed.
    feed_type : FeedType
        Magnitude band.

    Returns
    -------
    EarthquakeMonitorResult
    """
    # 1. Fetch from USGS
    query = fetch_earthquakes_feed(period, feed_type)

    # 2. Radius filter
    filtered = filter_earthquakes_by_radius(
        events=query.events,
        user_lat=user_lat,
        user_lon=user_lon,
        radius_km=radius_km,
        min_magnitude=min_magnitude,
    )

    # 3. Depth analysis
    depth = analyze_depth_distribution(filtered.matched)

    # 4. Highest risk event
    highest: Optional[EarthquakeEvent] = None
    if filtered.matched:
        risk_order = list(EarthquakeRisk)
        highest = max(
            filtered.matched,
            key=lambda e: (risk_order.index(e.risk_level), e.magnitude),
        )

    # 5. Summary
    summary = {
        "status": "ok" if query.success else "fetch_failed",
        "location": {"lat": user_lat, "lon": user_lon},
        "radius_km": radius_km,
        "period": period.value,
        "events_nearby": filtered.count,
        "highest_magnitude": (
            max(e.magnitude for e in filtered.matched)
            if filtered.matched else None
        ),
        "highest_risk": highest.risk_level.value if highest else "none",
    }

    return EarthquakeMonitorResult(
        query_result=query,
        filter_result=filtered,
        depth_analysis=depth,
        highest_risk_event=highest,
        summary=summary,
    )
