"""
cyclone_service.py — Cyclone / tropical storm monitoring and alert module.

Provides:
    • Wind speed threshold logic (configurable; default >60 km/h)
    • Heavy rainfall condition detection
    • Radius-based filtering (distance from eye to user)
    • Multi-level escalation logic (depression → storm → cyclone → super)
    • Risk scoring and alert generation
    • Weather-data-driven cyclone detection (from Open-Meteo integration)

═══════════════════════════════════════════════════════════════════════════
CYCLONE CLASSIFICATION — THRESHOLD LOGIC
═══════════════════════════════════════════════════════════════════════════

Tropical cyclone classification varies by basin. We use the India
Meteorological Department (IMD) scale by default (most relevant for
the Bay of Bengal / Arabian Sea), with Saffir-Simpson as alternative.

IMD Classification (sustained wind over 3-minute average):
    ┌─────────────────────────────────┬──────────────┬─────────────┐
    │ Category                        │ Wind (km/h)  │ Wind (knots)│
    ├─────────────────────────────────┼──────────────┼─────────────┤
    │ Low Pressure Area               │ < 31         │ < 17        │
    │ Depression (D)                  │ 31–49        │ 17–27       │
    │ Deep Depression (DD)            │ 50–61        │ 28–33       │
    │ Cyclonic Storm (CS)             │ 62–88        │ 34–47       │
    │ Severe Cyclonic Storm (SCS)     │ 89–117       │ 48–63       │
    │ Very Severe CS (VSCS)           │ 118–166      │ 64–89       │
    │ Extremely Severe CS (ESCS)      │ 167–221      │ 90–119      │
    │ Super Cyclonic Storm (SuCS)     │ > 221        │ > 119       │
    └─────────────────────────────────┴──────────────┴─────────────┘

Why 60 km/h as our default alert threshold?
    - Below 31: normal weather — no alert needed
    - 31–61: depression/deep depression — mostly mariners' concern
    - 62+: cyclonic storm — inland impact, evacuations begin
    - We use 60 km/h (≈ start of "cyclonic storm") as the default
      because this is when structural damage and flooding start.
    - Users in coastal areas may want a LOWER threshold (e.g. 50).
    - Inland users may want a HIGHER threshold (e.g. 90).

═══════════════════════════════════════════════════════════════════════════
ESCALATION LOGIC
═══════════════════════════════════════════════════════════════════════════

The escalation system triggers progressively stronger alerts based on:

    1. WIND SPEED — primary classification driver
    2. RAINFALL INTENSITY — flooding risk (orthogonal to wind damage)
    3. DISTANCE TO USER — spatial urgency
    4. RATE OF APPROACH — temporal urgency (time-to-impact)

Escalation levels and actions:

    WATCH    → Cyclone detected within 500 km. Monitor updates.
    WARNING  → Cyclone within 300 km OR intensifying toward user.
    ALERT    → Cyclone within 150 km. Prepare for impact.
    CRITICAL → Cyclone within 50 km. Immediate protective action.

Combined risk score (0–1):

    R_cyclone = w₁ · f_wind(V) + w₂ · f_rain(P) + w₃ · f_dist(D) + w₄ · f_press(ΔP)

Where:
    f_wind(V)   = min(1, (V − V_min) / (V_max − V_min))   ← normalised wind
    f_rain(P)   = min(1, P / P_extreme)                     ← normalised rainfall
    f_dist(D)   = max(0, 1 − D / D_max)                     ← inverse distance
    f_press(ΔP) = min(1, max(0, (1013 − P_msl) / 60))      ← pressure deficit

    Weights: w₁=0.35, w₂=0.25, w₃=0.25, w₄=0.15
    (wind is dominant; rainfall and distance equally important; pressure supplements)

═══════════════════════════════════════════════════════════════════════════
THRESHOLD TUNING LOGIC
═══════════════════════════════════════════════════════════════════════════

Thresholds should be tuned based on:

    1. Geographic basin — Bay of Bengal storms differ from Atlantic hurricanes.
    2. Coastal vs. inland — coastal populations need earlier/lower thresholds.
    3. Building stock — reinforced concrete tolerates higher winds than tin roofs.
    4. Historical calibration — compare alert levels against past damage reports.
    5. False alarm tolerance — too many alerts → alert fatigue → people ignore.
       Typical target: false alarm rate < 30%, miss rate < 5%.

The module exposes all thresholds as parameters, allowing:
    - Per-region profiles (e.g., Chennai coastal vs. Delhi inland)
    - Time-of-year adjustment (monsoon season → lower rainfall threshold)
    - Progressive refinement via Bayesian optimisation on historical data
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from backend.app.spatial.radius_utils import (
    Coordinate,
    haversine,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Constants & Defaults
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_WIND_THRESHOLD_KMH = 60.0       # Alert starts at cyclonic storm
DEFAULT_RAINFALL_THRESHOLD_MM = 50.0    # Heavy rainfall per 24h
DEFAULT_PRESSURE_BASELINE_HPA = 1013.25  # Standard sea-level pressure
DEFAULT_MAX_RADIUS_KM = 500.0           # Max monitoring radius
DEFAULT_EXTREME_RAINFALL_MM = 200.0     # Normalisation ceiling for rainfall
DEFAULT_EXTREME_WIND_KMH = 250.0        # Normalisation ceiling for wind

# Risk weight vector
W_WIND = 0.35
W_RAIN = 0.25
W_DIST = 0.25
W_PRESS = 0.15


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class CycloneCategory(str, Enum):
    """
    IMD tropical cyclone classification.
    Based on 3-minute sustained wind speed.
    """
    LOW_PRESSURE       = "low_pressure"        # < 31 km/h
    DEPRESSION         = "depression"          # 31–49 km/h
    DEEP_DEPRESSION    = "deep_depression"     # 50–61 km/h
    CYCLONIC_STORM     = "cyclonic_storm"      # 62–88 km/h
    SEVERE_CS          = "severe_cyclonic"     # 89–117 km/h
    VERY_SEVERE_CS     = "very_severe"         # 118–166 km/h
    EXTREMELY_SEVERE   = "extremely_severe"    # 167–221 km/h
    SUPER_CYCLONE      = "super_cyclone"       # > 221 km/h


class EscalationLevel(str, Enum):
    """Progressive escalation for alerts."""
    NONE     = "none"       # Below monitoring threshold
    WATCH    = "watch"      # Detected but distant
    WARNING  = "warning"    # Approaching or intensifying
    ALERT    = "alert"      # Imminent impact expected
    CRITICAL = "critical"   # Direct impact underway / imminent


class CycloneRisk(str, Enum):
    """Overall cyclone risk level."""
    NONE     = "none"
    LOW      = "low"
    MODERATE = "moderate"
    HIGH     = "high"
    EXTREME  = "extreme"


# ═══════════════════════════════════════════════════════════════════════════
# Classification Functions
# ═══════════════════════════════════════════════════════════════════════════

def classify_cyclone(wind_speed_kmh: float) -> CycloneCategory:
    """
    Classify a tropical system by sustained wind speed (km/h).

    Uses the India Meteorological Department (IMD) scale.

    Parameters
    ----------
    wind_speed_kmh : float
        3-minute sustained wind speed in km/h.

    Returns
    -------
    CycloneCategory

    Examples
    --------
    >>> classify_cyclone(45)
    <CycloneCategory.DEPRESSION: 'depression'>
    >>> classify_cyclone(130)
    <CycloneCategory.VERY_SEVERE_CS: 'very_severe'>
    >>> classify_cyclone(250)
    <CycloneCategory.SUPER_CYCLONE: 'super_cyclone'>
    """
    if wind_speed_kmh < 31:
        return CycloneCategory.LOW_PRESSURE
    elif wind_speed_kmh < 50:
        return CycloneCategory.DEPRESSION
    elif wind_speed_kmh < 62:
        return CycloneCategory.DEEP_DEPRESSION
    elif wind_speed_kmh < 89:
        return CycloneCategory.CYCLONIC_STORM
    elif wind_speed_kmh < 118:
        return CycloneCategory.SEVERE_CS
    elif wind_speed_kmh < 167:
        return CycloneCategory.VERY_SEVERE_CS
    elif wind_speed_kmh < 222:
        return CycloneCategory.EXTREMELY_SEVERE
    else:
        return CycloneCategory.SUPER_CYCLONE


def classify_risk(score: float) -> CycloneRisk:
    """
    Map a composite risk score [0, 1] to a risk level.

    Thresholds:
        [0.0, 0.15) → NONE
        [0.15, 0.35) → LOW
        [0.35, 0.60) → MODERATE
        [0.60, 0.80) → HIGH
        [0.80, 1.0]  → EXTREME
    """
    if score < 0.15:
        return CycloneRisk.NONE
    elif score < 0.35:
        return CycloneRisk.LOW
    elif score < 0.60:
        return CycloneRisk.MODERATE
    elif score < 0.80:
        return CycloneRisk.HIGH
    else:
        return CycloneRisk.EXTREME


def is_heavy_rainfall(
    rainfall_mm: float,
    threshold_mm: float = DEFAULT_RAINFALL_THRESHOLD_MM,
) -> bool:
    """
    Determine if rainfall qualifies as "heavy" for cyclone risk.

    IMD classification for 24-hour rainfall:
        Very light:  0.1 – 2.4 mm
        Light:       2.5 – 15.5 mm
        Moderate:    15.6 – 64.4 mm
        Heavy:       64.5 – 115.5 mm
        Very heavy:  115.6 – 204.4 mm
        Extremely heavy: ≥ 204.5 mm

    We default to 50 mm/24h as the threshold (moderate-heavy boundary)
    to capture flood-risk conditions early.

    Parameters
    ----------
    rainfall_mm : float
        Accumulated rainfall in mm (typically 24-hour total).
    threshold_mm : float
        Threshold in mm. Default: 50.

    Returns
    -------
    bool
    """
    return rainfall_mm >= threshold_mm


# ═══════════════════════════════════════════════════════════════════════════
# Escalation Logic
# ═══════════════════════════════════════════════════════════════════════════

def determine_escalation(
    wind_speed_kmh: float,
    distance_km: float,
    rainfall_mm: float = 0.0,
    pressure_hpa: float = DEFAULT_PRESSURE_BASELINE_HPA,
    *,
    wind_threshold: float = DEFAULT_WIND_THRESHOLD_KMH,
    max_watch_radius: float = 500.0,
    max_warning_radius: float = 300.0,
    max_alert_radius: float = 150.0,
    max_critical_radius: float = 50.0,
) -> EscalationLevel:
    """
    Determine the escalation level for a cyclone event relative to a user.

    The escalation combines:
        1. Wind speed — must exceed threshold to trigger any escalation
        2. Distance — closer → higher escalation
        3. Rainfall — heavy rain can upgrade escalation by one level
        4. Pressure — very low pressure can upgrade escalation by one level

    Parameters
    ----------
    wind_speed_kmh : float
        Sustained wind speed near the cyclone.
    distance_km : float
        Distance from cyclone centre to user location.
    rainfall_mm : float
        24-hour accumulated rainfall.
    pressure_hpa : float
        Sea-level pressure.
    wind_threshold : float
        Minimum wind speed to trigger monitoring.
    max_watch_radius : float
        Maximum distance for WATCH level.
    max_warning_radius : float
        Maximum distance for WARNING level.
    max_alert_radius : float
        Maximum distance for ALERT level.
    max_critical_radius : float
        Maximum distance for CRITICAL level.

    Returns
    -------
    EscalationLevel

    Examples
    --------
    >>> determine_escalation(80, 400)     # cyclone, 400km away
    <EscalationLevel.WATCH: 'watch'>
    >>> determine_escalation(120, 100)    # severe cyclone, 100km
    <EscalationLevel.ALERT: 'alert'>
    >>> determine_escalation(200, 30)     # extremely severe, 30km
    <EscalationLevel.CRITICAL: 'critical'>
    """
    # Below wind threshold — no escalation
    if wind_speed_kmh < wind_threshold:
        return EscalationLevel.NONE

    # Beyond maximum watch radius — no escalation (too far)
    if distance_km > max_watch_radius:
        return EscalationLevel.NONE

    # Base level from distance
    if distance_km <= max_critical_radius:
        level = EscalationLevel.CRITICAL
    elif distance_km <= max_alert_radius:
        level = EscalationLevel.ALERT
    elif distance_km <= max_warning_radius:
        level = EscalationLevel.WARNING
    else:
        level = EscalationLevel.WATCH

    # Upgrade conditions: heavy rainfall or deep pressure deficit
    upgrade = False
    if is_heavy_rainfall(rainfall_mm, threshold_mm=100.0):
        upgrade = True
    pressure_deficit = DEFAULT_PRESSURE_BASELINE_HPA - pressure_hpa
    if pressure_deficit > 30:  # > 30 hPa drop → intense system
        upgrade = True

    if upgrade:
        level_order = list(EscalationLevel)
        idx = level_order.index(level)
        if idx < len(level_order) - 1:
            level = level_order[idx + 1]

    return level


# ═══════════════════════════════════════════════════════════════════════════
# Cyclone Risk Score
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CycloneRiskScore:
    """Composite cyclone risk assessment."""

    wind_component: float       # f_wind normalised [0, 1]
    rainfall_component: float   # f_rain normalised [0, 1]
    distance_component: float   # f_dist normalised [0, 1]
    pressure_component: float   # f_press normalised [0, 1]
    composite_score: float      # weighted combination [0, 1]
    risk_level: CycloneRisk
    category: CycloneCategory
    escalation: EscalationLevel

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composite_score": round(self.composite_score, 4),
            "risk_level": self.risk_level.value,
            "category": self.category.value,
            "escalation": self.escalation.value,
            "components": {
                "wind": round(self.wind_component, 4),
                "rainfall": round(self.rainfall_component, 4),
                "distance": round(self.distance_component, 4),
                "pressure": round(self.pressure_component, 4),
            },
            "weights": {
                "wind": W_WIND,
                "rainfall": W_RAIN,
                "distance": W_DIST,
                "pressure": W_PRESS,
            },
        }


def compute_cyclone_risk(
    wind_speed_kmh: float,
    rainfall_mm: float,
    distance_km: float,
    pressure_hpa: float = DEFAULT_PRESSURE_BASELINE_HPA,
    *,
    wind_threshold: float = DEFAULT_WIND_THRESHOLD_KMH,
    max_radius_km: float = DEFAULT_MAX_RADIUS_KM,
    extreme_rainfall_mm: float = DEFAULT_EXTREME_RAINFALL_MM,
    extreme_wind_kmh: float = DEFAULT_EXTREME_WIND_KMH,
) -> CycloneRiskScore:
    """
    Compute composite cyclone risk score.

    Formula:
        R = w₁·f_wind + w₂·f_rain + w₃·f_dist + w₄·f_press

    Where each component is normalised to [0, 1]:
        f_wind  = clamp((V − V_min) / (V_max − V_min), 0, 1)
        f_rain  = clamp(P / P_extreme, 0, 1)
        f_dist  = clamp(1 − D / D_max, 0, 1)
        f_press = clamp((1013.25 − P) / 60, 0, 1)

    Parameters
    ----------
    wind_speed_kmh : float
        Sustained wind speed in km/h.
    rainfall_mm : float
        Accumulated rainfall in mm (typically 24h).
    distance_km : float
        Distance from cyclone centre to user.
    pressure_hpa : float
        Sea-level pressure at cyclone centre.
    wind_threshold : float
        Minimum wind for cyclone consideration.
    max_radius_km : float
        Maximum relevant distance.
    extreme_rainfall_mm : float
        Rainfall ceiling for normalisation.
    extreme_wind_kmh : float
        Wind ceiling for normalisation.

    Returns
    -------
    CycloneRiskScore

    Examples
    --------
    >>> score = compute_cyclone_risk(120, 80, 200, 990)
    >>> 0 <= score.composite_score <= 1
    True
    """
    # Normalise components
    wind_range = extreme_wind_kmh - wind_threshold
    if wind_range > 0:
        f_wind = max(0.0, min(1.0, (wind_speed_kmh - wind_threshold) / wind_range))
    else:
        f_wind = 1.0 if wind_speed_kmh >= wind_threshold else 0.0

    f_rain = max(0.0, min(1.0, rainfall_mm / extreme_rainfall_mm))

    if max_radius_km > 0:
        f_dist = max(0.0, min(1.0, 1.0 - distance_km / max_radius_km))
    else:
        f_dist = 1.0

    pressure_deficit = DEFAULT_PRESSURE_BASELINE_HPA - pressure_hpa
    f_press = max(0.0, min(1.0, pressure_deficit / 60.0))

    # Weighted composite
    composite = (
        W_WIND * f_wind
        + W_RAIN * f_rain
        + W_DIST * f_dist
        + W_PRESS * f_press
    )
    composite = max(0.0, min(1.0, composite))

    # Classify
    category = classify_cyclone(wind_speed_kmh)
    risk = classify_risk(composite)
    escalation = determine_escalation(
        wind_speed_kmh, distance_km, rainfall_mm, pressure_hpa,
        wind_threshold=wind_threshold,
    )

    return CycloneRiskScore(
        wind_component=f_wind,
        rainfall_component=f_rain,
        distance_component=f_dist,
        pressure_component=f_press,
        composite_score=composite,
        risk_level=risk,
        category=category,
        escalation=escalation,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Cyclone Event
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CycloneEvent:
    """Represents a cyclone / tropical storm event."""

    event_id: str
    name: str
    latitude: float
    longitude: float
    wind_speed_kmh: float
    pressure_hpa: float
    rainfall_24h_mm: float
    timestamp: datetime
    category: CycloneCategory = field(init=False)
    source: str = "weather_analysis"
    heading_degrees: Optional[float] = None  # direction of movement
    speed_of_movement_kmh: Optional[float] = None

    # Populated by filtering
    distance_km: Optional[float] = None
    risk_score: Optional[CycloneRiskScore] = None

    def __post_init__(self):
        self.category = classify_cyclone(self.wind_speed_kmh)

    @property
    def coordinate(self) -> Coordinate:
        return Coordinate(self.latitude, self.longitude)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "wind_speed_kmh": self.wind_speed_kmh,
            "pressure_hpa": self.pressure_hpa,
            "rainfall_24h_mm": self.rainfall_24h_mm,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "heading_degrees": self.heading_degrees,
            "speed_of_movement_kmh": self.speed_of_movement_kmh,
            "distance_km": (
                round(self.distance_km, 2) if self.distance_km else None
            ),
            "risk_score": (
                self.risk_score.to_dict() if self.risk_score else None
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Weather-Based Cyclone Detection
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CycloneDetectionResult:
    """Result of weather-based cyclone condition detection."""

    is_cyclonic: bool
    wind_speed_kmh: float
    rainfall_mm: float
    pressure_hpa: float
    category: CycloneCategory
    heavy_rainfall: bool
    conditions_met: List[str]
    risk_score: Optional[CycloneRiskScore] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_cyclonic": self.is_cyclonic,
            "wind_speed_kmh": self.wind_speed_kmh,
            "rainfall_mm": self.rainfall_mm,
            "pressure_hpa": self.pressure_hpa,
            "category": self.category.value,
            "heavy_rainfall": self.heavy_rainfall,
            "conditions_met": self.conditions_met,
            "risk_score": (
                self.risk_score.to_dict() if self.risk_score else None
            ),
        }


def detect_cyclone_conditions(
    wind_speed_kmh: float,
    rainfall_mm: float = 0.0,
    pressure_hpa: float = DEFAULT_PRESSURE_BASELINE_HPA,
    wind_gust_kmh: Optional[float] = None,
    *,
    wind_threshold: float = DEFAULT_WIND_THRESHOLD_KMH,
    rainfall_threshold: float = DEFAULT_RAINFALL_THRESHOLD_MM,
    pressure_drop_threshold: float = 10.0,
    distance_km: float = 0.0,
) -> CycloneDetectionResult:
    """
    Detect if current weather conditions indicate cyclonic activity.

    Checks three independent criteria:
        1. Wind speed ≥ threshold (default: 60 km/h)
        2. Heavy rainfall ≥ threshold (default: 50 mm)
        3. Pressure deficit ≥ threshold (default: 10 hPa below normal)

    Any ONE condition triggers is_cyclonic=True (they are OR-ed).
    All met conditions are listed in conditions_met.

    Parameters
    ----------
    wind_speed_kmh : float
        Current sustained wind speed.
    rainfall_mm : float
        Accumulated rainfall.
    pressure_hpa : float
        Sea-level pressure.
    wind_gust_kmh : float | None
        Peak wind gust (if available; treated as max wind if > sustained).
    wind_threshold : float
        Wind speed for cyclone detection.
    rainfall_threshold : float
        Rainfall threshold for heavy rain flag.
    pressure_drop_threshold : float
        Pressure deficit from 1013.25 hPa to flag.
    distance_km : float
        Distance from user (used for risk score; 0 = co-located).

    Returns
    -------
    CycloneDetectionResult
    """
    effective_wind = wind_speed_kmh
    if wind_gust_kmh is not None and wind_gust_kmh > wind_speed_kmh:
        effective_wind = max(wind_speed_kmh, wind_gust_kmh * 0.85)
        # Gust factor: sustained ≈ 85% of peak gust (typical ratio)

    conditions: List[str] = []

    # Check wind
    if effective_wind >= wind_threshold:
        conditions.append(
            f"wind_speed={effective_wind:.1f} km/h ≥ {wind_threshold}"
        )

    # Check rainfall
    heavy_rain = is_heavy_rainfall(rainfall_mm, rainfall_threshold)
    if heavy_rain:
        conditions.append(
            f"rainfall={rainfall_mm:.1f} mm ≥ {rainfall_threshold}"
        )

    # Check pressure
    pressure_deficit = DEFAULT_PRESSURE_BASELINE_HPA - pressure_hpa
    if pressure_deficit >= pressure_drop_threshold:
        conditions.append(
            f"pressure_deficit={pressure_deficit:.1f} hPa ≥ {pressure_drop_threshold}"
        )

    is_cyclonic = len(conditions) > 0
    category = classify_cyclone(effective_wind)

    # Compute risk if cyclonic
    risk = None
    if is_cyclonic:
        risk = compute_cyclone_risk(
            effective_wind, rainfall_mm, distance_km, pressure_hpa,
            wind_threshold=wind_threshold,
        )

    return CycloneDetectionResult(
        is_cyclonic=is_cyclonic,
        wind_speed_kmh=effective_wind,
        rainfall_mm=rainfall_mm,
        pressure_hpa=pressure_hpa,
        category=category,
        heavy_rainfall=heavy_rain,
        conditions_met=conditions,
        risk_score=risk,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Radius Filtering
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CycloneFilterResult:
    """Result of radius-based cyclone filtering."""
    user_location: Coordinate
    radius_km: float
    total_checked: int
    matched: List[CycloneEvent]
    excluded: int
    highest_escalation: EscalationLevel

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
            "highest_escalation": self.highest_escalation.value,
            "events": [e.to_dict() for e in self.matched],
        }


def filter_cyclones_by_radius(
    events: List[CycloneEvent],
    user_lat: float,
    user_lon: float,
    radius_km: float = DEFAULT_MAX_RADIUS_KM,
    wind_threshold: float = DEFAULT_WIND_THRESHOLD_KMH,
    sort_by: str = "distance",
    max_results: int = 20,
) -> CycloneFilterResult:
    """
    Filter cyclone events by distance from user, compute risk scores,
    and determine escalation levels.

    Parameters
    ----------
    events : list[CycloneEvent]
        Cyclone events to filter.
    user_lat, user_lon : float
        User's GPS coordinates.
    radius_km : float
        Alert radius in km.
    wind_threshold : float
        Minimum wind speed for consideration.
    sort_by : str
        "distance" or "risk" (highest risk first).
    max_results : int
        Cap on returned events.

    Returns
    -------
    CycloneFilterResult
    """
    user_coord = Coordinate(user_lat, user_lon)
    matched: List[CycloneEvent] = []
    highest_esc = EscalationLevel.NONE

    for event in events:
        if event.wind_speed_kmh < wind_threshold:
            continue

        dist = haversine(user_coord, event.coordinate)
        if dist > radius_km:
            continue

        event.distance_km = dist
        event.risk_score = compute_cyclone_risk(
            event.wind_speed_kmh,
            event.rainfall_24h_mm,
            dist,
            event.pressure_hpa,
            wind_threshold=wind_threshold,
        )

        # Track highest escalation
        esc_order = list(EscalationLevel)
        if esc_order.index(event.risk_score.escalation) > esc_order.index(highest_esc):
            highest_esc = event.risk_score.escalation

        matched.append(event)

    # Sort
    if sort_by == "risk" and matched:
        matched.sort(
            key=lambda e: e.risk_score.composite_score if e.risk_score else 0,
            reverse=True,
        )
    else:
        matched.sort(key=lambda e: e.distance_km or 0.0)

    matched = matched[:max_results]

    return CycloneFilterResult(
        user_location=user_coord,
        radius_km=radius_km,
        total_checked=len(events),
        matched=matched,
        excluded=len(events) - len(matched),
        highest_escalation=highest_esc,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Alert Recommendations
# ═══════════════════════════════════════════════════════════════════════════

CYCLONE_ALERTS: Dict[EscalationLevel, Dict[str, str]] = {
    EscalationLevel.NONE: {
        "action": "No action required",
        "message": "No cyclonic threat detected.",
        "color": "#4CAF50",
    },
    EscalationLevel.WATCH: {
        "action": "Stay informed",
        "message": "Cyclone detected in region. Monitor official updates from IMD/NDMA.",
        "color": "#2196F3",
    },
    EscalationLevel.WARNING: {
        "action": "Prepare",
        "message": "Cyclone approaching. Stock emergency supplies, charge devices, "
                   "identify nearest shelter.",
        "color": "#FF9800",
    },
    EscalationLevel.ALERT: {
        "action": "Take protective action",
        "message": "Cyclone impact expected within hours. Move to sturdy structures. "
                   "Secure loose objects. Avoid coastal areas.",
        "color": "#F44336",
    },
    EscalationLevel.CRITICAL: {
        "action": "SEEK SHELTER IMMEDIATELY",
        "message": "CRITICAL: Cyclone making landfall. Stay indoors in reinforced "
                   "structures away from windows. Follow evacuation orders.",
        "color": "#B71C1C",
    },
}


def get_cyclone_alert(
    escalation: EscalationLevel,
    event: Optional[CycloneEvent] = None,
) -> Dict[str, Any]:
    """
    Generate a cyclone alert message for a given escalation level.

    Parameters
    ----------
    escalation : EscalationLevel
        Current escalation.
    event : CycloneEvent | None
        The cyclone event (for enriched messaging).

    Returns
    -------
    dict
        Alert details with action, message, and metadata.
    """
    base = dict(CYCLONE_ALERTS[escalation])
    base["escalation_level"] = escalation.value

    if event is not None:
        base["cyclone_name"] = event.name
        base["category"] = event.category.value
        base["wind_speed_kmh"] = event.wind_speed_kmh
        base["distance_km"] = round(event.distance_km, 1) if event.distance_km else None

    return base


# ═══════════════════════════════════════════════════════════════════════════
# End-to-End: Weather-Based Cyclone Assessment
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CycloneAssessment:
    """Complete cyclone assessment for a location."""

    latitude: float
    longitude: float
    detection: CycloneDetectionResult
    escalation: EscalationLevel
    alert: Dict[str, Any]
    weather_inputs: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "detection": self.detection.to_dict(),
            "escalation": self.escalation.value,
            "alert": self.alert,
            "weather_inputs": self.weather_inputs,
        }


def assess_cyclone_risk(
    latitude: float,
    longitude: float,
    wind_speed_kmh: float,
    rainfall_mm: float = 0.0,
    pressure_hpa: float = DEFAULT_PRESSURE_BASELINE_HPA,
    wind_gust_kmh: Optional[float] = None,
    *,
    wind_threshold: float = DEFAULT_WIND_THRESHOLD_KMH,
    rainfall_threshold: float = DEFAULT_RAINFALL_THRESHOLD_MM,
) -> CycloneAssessment:
    """
    End-to-end cyclone risk assessment from weather observations.

    Combines detection, classification, escalation, and alert generation.

    Parameters
    ----------
    latitude, longitude : float
        Location coordinates.
    wind_speed_kmh : float
        Sustained wind speed.
    rainfall_mm : float
        Accumulated rainfall.
    pressure_hpa : float
        Sea-level pressure.
    wind_gust_kmh : float | None
        Peak gust.
    wind_threshold : float
        Wind threshold for detection.
    rainfall_threshold : float
        Rainfall threshold for heavy rain.

    Returns
    -------
    CycloneAssessment
    """
    detection = detect_cyclone_conditions(
        wind_speed_kmh,
        rainfall_mm=rainfall_mm,
        pressure_hpa=pressure_hpa,
        wind_gust_kmh=wind_gust_kmh,
        wind_threshold=wind_threshold,
        rainfall_threshold=rainfall_threshold,
        distance_km=0.0,  # assessing at the observation point
    )

    # Escalation at distance=0 (observation point)
    escalation = determine_escalation(
        detection.wind_speed_kmh,
        distance_km=0.0,
        rainfall_mm=rainfall_mm,
        pressure_hpa=pressure_hpa,
        wind_threshold=wind_threshold,
    )

    alert = get_cyclone_alert(escalation)

    return CycloneAssessment(
        latitude=latitude,
        longitude=longitude,
        detection=detection,
        escalation=escalation,
        alert=alert,
        weather_inputs={
            "wind_speed_kmh": wind_speed_kmh,
            "wind_gust_kmh": wind_gust_kmh or 0.0,
            "rainfall_mm": rainfall_mm,
            "pressure_hpa": pressure_hpa,
        },
    )
