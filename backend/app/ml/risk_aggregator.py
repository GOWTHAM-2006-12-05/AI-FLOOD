"""
risk_aggregator.py — Unified Disaster Risk Aggregation Engine.

Combines independent hazard assessments into a single composite risk profile:

    • Flood probability       (0–1)   from XGBoost + LSTM ensemble
    • Earthquake severity     (0–1)   from magnitude × depth factor
    • Cyclone risk score      (0–1)   from wind / rain / distance / pressure

Produces:
    • overall_risk_score      (0–100%)
    • overall_risk_level      (Safe / Watch / Warning / Severe)
    • per-hazard breakdown
    • dominant hazard identification
    • alert triggering decisions

═══════════════════════════════════════════════════════════════════════════
MATHEMATICAL WEIGHTING FORMULA
═══════════════════════════════════════════════════════════════════════════

The aggregation uses a **max-dominated weighted hybrid** strategy rather
than a pure weighted average.  Pure averaging is dangerous because it
dilutes a single catastrophic hazard when the others are calm.

    Step 1 — Normalise each hazard to [0, 1]:

        S_flood     = flood_probability                          ∈ [0, 1]
        S_earthquake = mag × depth_severity_factor / 10          ∈ [0, 1]
                       (capped at 1.0; M10 × 1.5 → 1.5 → 1.0)
        S_cyclone   = cyclone_composite_score                    ∈ [0, 1]

    Step 2 — Weighted average component (captures multi-hazard exposure):

        R_avg = w_f · S_flood + w_e · S_earthquake + w_c · S_cyclone

        Default weights:
            w_f = 0.40  (floods are the most frequent natural disaster)
            w_e = 0.30  (earthquakes: rare but devastating)
            w_c = 0.30  (cyclones: seasonal but high-impact)

        Weights are configurable per-region.  Coastal areas may increase
        w_c; seismic zones increase w_e; floodplains increase w_f.

    Step 3 — Max component (captures single-hazard dominance):

        R_max = max(S_flood, S_earthquake, S_cyclone)

    Step 4 — Hybrid blending:

        R_hybrid = β · R_max + (1 − β) · R_avg

        Default β = 0.60  (max-dominated: a single extreme hazard should
                           still produce a high overall score even if the
                           other two are zero)

    Step 5 — Amplification for concurrent hazards:

        If two or more hazards exceed their "active" threshold (≥ 0.30),
        apply a concurrency bonus:

            n_active = count of hazards with S ≥ 0.30
            amplifier = 1.0 + γ · (n_active − 1)     γ = 0.10

        So 2 concurrent hazards → ×1.10, 3 → ×1.20.
        This reflects the compounding reality: earthquake + rain → landslide.

    Step 6 — Final score:

        overall_risk_score = clamp(R_hybrid × amplifier × 100, 0, 100)

═══════════════════════════════════════════════════════════════════════════
ESCALATION THRESHOLDS
═══════════════════════════════════════════════════════════════════════════

    overall_risk_score    overall_risk_level    Action
    ─────────────────     ──────────────────    ────────────────────────
      0 – 20              Safe                 No action.  Monitor only.
     20 – 45              Watch                Stay informed; review plans.
     45 – 70              Warning              Prepare; secure property.
     70 – 100             Severe               Evacuate / shelter NOW.

Transitions are **hysteresis-aware**: an escalation from Watch → Warning
requires score ≥ 45, but de-escalation from Warning → Watch requires
score ≤ 38 (7-point buffer) to prevent oscillation.

═══════════════════════════════════════════════════════════════════════════
ALERT TRIGGERING LOGIC
═══════════════════════════════════════════════════════════════════════════

An alert is triggered when:

    1. overall_risk_level changes (escalation or de-escalation), OR
    2. Any single hazard crosses its individual CRITICAL threshold:
           flood_probability ≥ 0.80
           earthquake severity ≥ CRITICAL (effective M ≥ 7.0)
           cyclone composite  ≥ 0.80 (EXTREME)
    3. Two or more hazards are simultaneously ≥ WARNING level.

Alert priority order (highest → lowest):
    1. Earthquake  — no lead time; immediate structural danger
    2. Cyclone     — hours of lead time; widespread area impact
    3. Flood       — minutes-to-hours lead; localised but persistent

This priority drives which alert message is displayed first in the UI
when multiple hazards fire simultaneously.

═══════════════════════════════════════════════════════════════════════════
PRIORITY ORDERING OF DISASTERS
═══════════════════════════════════════════════════════════════════════════

Priority is assigned by **immediacy × lethality**:

    Priority 1 — Earthquake
        • Zero warning time (seconds at best via ShakeAlert)
        • Structural collapse is immediately lethal
        • Triggers secondary hazards (tsunami, landslide, fire)
        • Cannot be "prepared for" in the moment

    Priority 2 — Cyclone
        • Hours of warning (satellite / radar detection)
        • Large area of impact (100s of km)
        • Storm surge + wind damage + flooding combined
        • Disrupts infrastructure for days–weeks

    Priority 3 — Flood
        • Minutes to hours of lead time (gauge / rainfall data)
        • Usually localised (river basin / urban catchment)
        • Progressive onset allows staged evacuation
        • Highest frequency globally — chronic risk

When two disasters have equal risk scores, the higher-priority type
is displayed first and its alert takes precedence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Constants — Tunable Parameters
# ═══════════════════════════════════════════════════════════════════════════

# Hazard weights (must sum to 1.0)
W_FLOOD = 0.40
W_EARTHQUAKE = 0.30
W_CYCLONE = 0.30

# Max-dominance blending factor
BETA = 0.60  # 60% max, 40% weighted average

# Concurrency amplification factor
GAMMA = 0.10  # bonus per additional active hazard
ACTIVE_THRESHOLD = 0.30  # hazard considered "active" above this

# Earthquake normalisation ceiling
EQ_NORMALISATION_CEILING = 10.0  # effective magnitude / ceiling → [0,1]

# Escalation thresholds (0–100 scale)
THRESHOLD_SAFE_UPPER = 20.0
THRESHOLD_WATCH_UPPER = 45.0
THRESHOLD_WARNING_UPPER = 70.0
# Severe is anything ≥ 70

# Hysteresis buffer for de-escalation
HYSTERESIS_BUFFER = 7.0

# Individual hazard critical thresholds (normalised 0–1)
FLOOD_CRITICAL = 0.80
EARTHQUAKE_CRITICAL = 0.80  # normalised score (≈ M8 shallow)
CYCLONE_CRITICAL = 0.80

# Disaster priority ranking (lower number = higher priority)
PRIORITY_MAP = {
    "earthquake": 1,
    "cyclone": 2,
    "flood": 3,
}


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class OverallRiskLevel(str, Enum):
    """Unified risk level across all disaster types."""
    SAFE = "safe"           # 0–20%
    WATCH = "watch"         # 20–45%
    WARNING = "warning"     # 45–70%
    SEVERE = "severe"       # 70–100%


class AlertAction(str, Enum):
    """Recommended action for each risk level."""
    MONITOR = "monitor"
    STAY_INFORMED = "stay_informed"
    PREPARE = "prepare"
    EVACUATE = "evacuate"


# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HazardScore:
    """
    Normalised score for a single hazard type.

    Attributes
    ----------
    hazard_type : str
        One of 'flood', 'earthquake', 'cyclone'.
    raw_value : float
        The original value from the hazard service (probability, magnitude, score).
    normalised_score : float
        Score mapped to [0, 1] for aggregation.
    weight : float
        Weight assigned in the aggregation formula.
    is_active : bool
        True if normalised_score ≥ ACTIVE_THRESHOLD.
    is_critical : bool
        True if normalised_score exceeds the hazard's critical threshold.
    priority : int
        Disaster priority (1 = highest).
    details : dict
        Additional metadata from the source service.
    """
    hazard_type: str
    raw_value: float
    normalised_score: float
    weight: float
    is_active: bool
    is_critical: bool
    priority: int
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hazard_type": self.hazard_type,
            "raw_value": round(self.raw_value, 4),
            "normalised_score": round(self.normalised_score, 4),
            "weight": self.weight,
            "weighted_contribution": round(self.normalised_score * self.weight, 4),
            "is_active": self.is_active,
            "is_critical": self.is_critical,
            "priority": self.priority,
            "details": self.details,
        }


@dataclass
class AggregatedRisk:
    """
    Complete output of the unified disaster risk aggregation.

    Contains the overall score, level, per-hazard breakdown, alert
    information, and the mathematical components used in computation.
    """
    # ---- Core outputs ----
    overall_risk_score: float          # 0–100
    overall_risk_level: OverallRiskLevel
    alert_action: AlertAction

    # ---- Per-hazard breakdown ----
    hazard_scores: List[HazardScore]
    dominant_hazard: str               # hazard type with highest normalised score
    active_hazard_count: int           # number of hazards ≥ ACTIVE_THRESHOLD

    # ---- Aggregation internals (for transparency / debugging) ----
    r_avg: float                       # weighted average component
    r_max: float                       # max component
    r_hybrid: float                    # blended before amplification
    amplifier: float                   # concurrency bonus multiplier
    alert_triggered: bool              # True if an alert should fire
    alert_reasons: List[str]           # human-readable trigger reasons

    # ---- Location context ----
    latitude: float = 0.0
    longitude: float = 0.0

    # ---- Previous level for hysteresis ----
    previous_level: Optional[OverallRiskLevel] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for API response."""
        return {
            "overall_risk_score": round(self.overall_risk_score, 2),
            "overall_risk_score_pct": f"{self.overall_risk_score:.1f}%",
            "overall_risk_level": self.overall_risk_level.value,
            "alert_action": self.alert_action.value,
            "dominant_hazard": self.dominant_hazard,
            "active_hazard_count": self.active_hazard_count,
            "alert_triggered": self.alert_triggered,
            "alert_reasons": self.alert_reasons,
            "hazard_breakdown": [h.to_dict() for h in self.hazard_scores],
            "formula_components": {
                "R_avg": round(self.r_avg, 4),
                "R_max": round(self.r_max, 4),
                "beta": BETA,
                "R_hybrid": round(self.r_hybrid, 4),
                "amplifier": round(self.amplifier, 4),
                "weights": {
                    "flood": W_FLOOD,
                    "earthquake": W_EARTHQUAKE,
                    "cyclone": W_CYCLONE,
                },
            },
            "thresholds": {
                "safe":    f"0 – {THRESHOLD_SAFE_UPPER}%",
                "watch":   f"{THRESHOLD_SAFE_UPPER} – {THRESHOLD_WATCH_UPPER}%",
                "warning": f"{THRESHOLD_WATCH_UPPER} – {THRESHOLD_WARNING_UPPER}%",
                "severe":  f"{THRESHOLD_WARNING_UPPER} – 100%",
            },
            "location": {
                "latitude": self.latitude,
                "longitude": self.longitude,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════
# Normalisation Functions
# ═══════════════════════════════════════════════════════════════════════════

def normalise_flood(flood_probability: float) -> float:
    """
    Normalise flood probability to [0, 1].

    The ensemble already outputs a calibrated probability ∈ [0, 1],
    so this is largely a passthrough with clamping.

    Parameters
    ----------
    flood_probability : float
        Flood probability from ensemble (0–1).

    Returns
    -------
    float
        Clamped to [0, 1].
    """
    return max(0.0, min(1.0, flood_probability))


def normalise_earthquake(
    magnitude: float,
    depth_km: float,
    ceiling: float = EQ_NORMALISATION_CEILING,
) -> float:
    """
    Normalise earthquake severity to [0, 1].

    Uses effective magnitude = magnitude × depth_severity_factor(depth):

        S_earthquake = clamp(effective_magnitude / ceiling, 0, 1)

    The depth_severity_factor amplifies shallow quakes (×1.5) and
    attenuates deep ones (×0.2), matching real-world damage patterns.

    Parameters
    ----------
    magnitude : float
        Earthquake magnitude (Richter / moment magnitude).
    depth_km : float
        Focal depth in km.
    ceiling : float
        Maximum "effective magnitude" for normalisation. Default 10.0
        (theoretical max is ~9.5; with 1.5× depth factor → 14.25
        but we cap at 1.0 anyway).

    Returns
    -------
    float
        Normalised earthquake severity ∈ [0, 1].

    Examples
    --------
    >>> normalise_earthquake(6.0, 10.0)   # shallow M6
    0.9       # 6.0 × 1.5 / 10 = 0.9
    >>> normalise_earthquake(4.0, 200.0)  # intermediate M4
    0.24      # 4.0 × 0.6 / 10 = 0.24
    >>> normalise_earthquake(3.0, 400.0)  # deep M3
    0.06      # 3.0 × 0.2 / 10 = 0.06
    """
    from backend.app.ml.earthquake_service import depth_severity_factor

    dsf = depth_severity_factor(depth_km)
    effective = magnitude * dsf
    return max(0.0, min(1.0, effective / ceiling))


def normalise_cyclone(cyclone_composite_score: float) -> float:
    """
    Normalise cyclone composite score to [0, 1].

    The cyclone service already produces a composite ∈ [0, 1]
    (wind + rain + distance + pressure weighted), so this is
    a passthrough with clamping.

    Parameters
    ----------
    cyclone_composite_score : float
        Composite score from compute_cyclone_risk().

    Returns
    -------
    float
        Clamped to [0, 1].
    """
    return max(0.0, min(1.0, cyclone_composite_score))


# ═══════════════════════════════════════════════════════════════════════════
# Risk Level Classification
# ═══════════════════════════════════════════════════════════════════════════

def classify_overall_risk(
    score: float,
    previous_level: Optional[OverallRiskLevel] = None,
) -> OverallRiskLevel:
    """
    Map overall risk score (0–100) to a risk level with hysteresis.

    Hysteresis prevents oscillation at boundaries.  To escalate, the
    score must exceed the upper threshold.  To de-escalate, the score
    must drop below (threshold − HYSTERESIS_BUFFER).

    Parameters
    ----------
    score : float
        Overall risk score ∈ [0, 100].
    previous_level : OverallRiskLevel | None
        The previous risk level (for hysteresis). If None, no hysteresis.

    Returns
    -------
    OverallRiskLevel

    Examples
    --------
    >>> classify_overall_risk(15)
    <OverallRiskLevel.SAFE: 'safe'>
    >>> classify_overall_risk(50)
    <OverallRiskLevel.WARNING: 'warning'>
    >>> classify_overall_risk(85)
    <OverallRiskLevel.SEVERE: 'severe'>
    """
    # Determine the "raw" level without hysteresis
    if score >= THRESHOLD_WARNING_UPPER:
        raw_level = OverallRiskLevel.SEVERE
    elif score >= THRESHOLD_WATCH_UPPER:
        raw_level = OverallRiskLevel.WARNING
    elif score >= THRESHOLD_SAFE_UPPER:
        raw_level = OverallRiskLevel.WATCH
    else:
        raw_level = OverallRiskLevel.SAFE

    if previous_level is None:
        return raw_level

    # Hysteresis: resist de-escalation unless score is clearly below threshold
    level_order = [
        OverallRiskLevel.SAFE,
        OverallRiskLevel.WATCH,
        OverallRiskLevel.WARNING,
        OverallRiskLevel.SEVERE,
    ]
    prev_idx = level_order.index(previous_level)
    raw_idx = level_order.index(raw_level)

    if raw_idx < prev_idx:
        # Attempting de-escalation — apply hysteresis
        de_escalation_thresholds = [
            0.0,                                          # below SAFE (N/A)
            THRESHOLD_SAFE_UPPER - HYSTERESIS_BUFFER,     # WATCH→SAFE: 13
            THRESHOLD_WATCH_UPPER - HYSTERESIS_BUFFER,    # WARNING→WATCH: 38
            THRESHOLD_WARNING_UPPER - HYSTERESIS_BUFFER,  # SEVERE→WARNING: 63
        ]
        # Check if score is below the de-escalation threshold of current level
        de_thresh = de_escalation_thresholds[prev_idx]
        if score > de_thresh:
            return previous_level  # stick to previous (resist de-escalation)

    return raw_level


def risk_level_to_action(level: OverallRiskLevel) -> AlertAction:
    """Map risk level to recommended action."""
    return {
        OverallRiskLevel.SAFE: AlertAction.MONITOR,
        OverallRiskLevel.WATCH: AlertAction.STAY_INFORMED,
        OverallRiskLevel.WARNING: AlertAction.PREPARE,
        OverallRiskLevel.SEVERE: AlertAction.EVACUATE,
    }[level]


# ═══════════════════════════════════════════════════════════════════════════
# Alert Triggering
# ═══════════════════════════════════════════════════════════════════════════

ALERT_MESSAGES = {
    OverallRiskLevel.SAFE: {
        "title": "All Clear",
        "message": "No significant disaster risk detected. Continue normal activities.",
        "color": "#4CAF50",
        "icon": "check_circle",
    },
    OverallRiskLevel.WATCH: {
        "title": "Risk Watch",
        "message": "Elevated risk detected. Stay informed and review your emergency plan.",
        "color": "#FF9800",
        "icon": "visibility",
    },
    OverallRiskLevel.WARNING: {
        "title": "Risk Warning",
        "message": "Significant risk! Prepare emergency supplies. Secure property. Be ready to act.",
        "color": "#F44336",
        "icon": "warning",
    },
    OverallRiskLevel.SEVERE: {
        "title": "SEVERE — Immediate Action Required",
        "message": "CRITICAL multi-hazard risk! Follow evacuation orders. Seek shelter immediately.",
        "color": "#B71C1C",
        "icon": "emergency",
    },
}


def determine_alerts(
    hazard_scores: List[HazardScore],
    overall_level: OverallRiskLevel,
    previous_level: Optional[OverallRiskLevel] = None,
) -> Tuple[bool, List[str]]:
    """
    Determine whether an alert should fire and collect reasons.

    Alert triggers:
        1. Risk level escalated from previous level
        2. Any hazard crossed its individual critical threshold
        3. Two or more hazards are simultaneously active (≥ 0.30)

    Parameters
    ----------
    hazard_scores : list of HazardScore
    overall_level : OverallRiskLevel
    previous_level : OverallRiskLevel | None

    Returns
    -------
    (triggered, reasons)
        triggered : bool
        reasons : list of str — human-readable explanations
    """
    reasons: List[str] = []

    # 1. Level escalation
    if previous_level is not None:
        level_order = [
            OverallRiskLevel.SAFE,
            OverallRiskLevel.WATCH,
            OverallRiskLevel.WARNING,
            OverallRiskLevel.SEVERE,
        ]
        if level_order.index(overall_level) > level_order.index(previous_level):
            reasons.append(
                f"Risk escalated: {previous_level.value} → {overall_level.value}"
            )

    # 2. Individual critical thresholds
    critical_thresholds = {
        "flood": FLOOD_CRITICAL,
        "earthquake": EARTHQUAKE_CRITICAL,
        "cyclone": CYCLONE_CRITICAL,
    }
    for hs in hazard_scores:
        threshold = critical_thresholds.get(hs.hazard_type, 0.80)
        if hs.normalised_score >= threshold:
            reasons.append(
                f"{hs.hazard_type.upper()} at CRITICAL level "
                f"(score={hs.normalised_score:.2f} ≥ {threshold})"
            )

    # 3. Concurrent active hazards
    active = [hs for hs in hazard_scores if hs.is_active]
    if len(active) >= 2:
        names = ", ".join(hs.hazard_type for hs in active)
        reasons.append(
            f"Multiple concurrent hazards active: {names} "
            f"({len(active)} hazards ≥ {ACTIVE_THRESHOLD})"
        )

    triggered = len(reasons) > 0
    return triggered, reasons


# ═══════════════════════════════════════════════════════════════════════════
# Core Aggregation Engine
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_risk(
    flood_probability: float = 0.0,
    earthquake_magnitude: float = 0.0,
    earthquake_depth_km: float = 10.0,
    cyclone_score: float = 0.0,
    *,
    latitude: float = 0.0,
    longitude: float = 0.0,
    previous_level: Optional[OverallRiskLevel] = None,
    w_flood: float = W_FLOOD,
    w_earthquake: float = W_EARTHQUAKE,
    w_cyclone: float = W_CYCLONE,
    beta: float = BETA,
    gamma: float = GAMMA,
    flood_details: Optional[Dict[str, Any]] = None,
    earthquake_details: Optional[Dict[str, Any]] = None,
    cyclone_details: Optional[Dict[str, Any]] = None,
) -> AggregatedRisk:
    """
    Compute the unified disaster risk aggregation.

    This is the main entry-point.  It normalises each hazard, applies
    the weighted-max hybrid formula with concurrency amplification,
    classifies the result, and determines alert triggers.

    Parameters
    ----------
    flood_probability : float
        Flood probability from ensemble model (0–1).
    earthquake_magnitude : float
        Earthquake magnitude (0–10+). Pass 0 if no earthquake.
    earthquake_depth_km : float
        Earthquake focal depth in km. Default 10 (shallow).
    cyclone_score : float
        Cyclone composite risk score (0–1). Pass 0 if no cyclone.
    latitude, longitude : float
        User location for context.
    previous_level : OverallRiskLevel | None
        Previous risk level for hysteresis de-escalation.
    w_flood, w_earthquake, w_cyclone : float
        Per-hazard weights (should sum to 1.0).
    beta : float
        Blending factor between max and average (0 = pure avg, 1 = pure max).
    gamma : float
        Concurrency amplification per additional active hazard.
    flood_details, earthquake_details, cyclone_details : dict | None
        Optional metadata to attach to each hazard score.

    Returns
    -------
    AggregatedRisk
        Complete aggregation result with score, level, breakdown, alerts.

    Examples
    --------
    >>> result = aggregate_risk(flood_probability=0.7, earthquake_magnitude=0, cyclone_score=0)
    >>> result.overall_risk_score  # ~42  (flood-only, no amplification)
    >>> result.overall_risk_level  # WATCH

    >>> result = aggregate_risk(flood_probability=0.9, earthquake_magnitude=7.5,
    ...                         earthquake_depth_km=8, cyclone_score=0.85)
    >>> result.overall_risk_score  # ~100 (all critical, amplified)
    >>> result.overall_risk_level  # SEVERE
    """
    # ---- Step 1: Normalise ----
    s_flood = normalise_flood(flood_probability)
    s_earthquake = normalise_earthquake(earthquake_magnitude, earthquake_depth_km)
    s_cyclone = normalise_cyclone(cyclone_score)

    scores = {
        "flood": s_flood,
        "earthquake": s_earthquake,
        "cyclone": s_cyclone,
    }

    # ---- Step 2: Weighted average ----
    r_avg = w_flood * s_flood + w_earthquake * s_earthquake + w_cyclone * s_cyclone

    # ---- Step 3: Max ----
    r_max = max(s_flood, s_earthquake, s_cyclone)

    # ---- Step 4: Hybrid blend ----
    r_hybrid = beta * r_max + (1.0 - beta) * r_avg

    # ---- Step 5: Concurrency amplification ----
    n_active = sum(1 for s in scores.values() if s >= ACTIVE_THRESHOLD)
    amplifier = 1.0 + gamma * max(0, n_active - 1)

    # ---- Step 6: Final score ----
    raw_score = r_hybrid * amplifier * 100.0
    overall_score = max(0.0, min(100.0, raw_score))

    # ---- Build per-hazard breakdown ----
    hazard_scores = []
    for hazard_type, norm_score in scores.items():
        weight = {"flood": w_flood, "earthquake": w_earthquake, "cyclone": w_cyclone}[hazard_type]
        critical_thresh = {"flood": FLOOD_CRITICAL, "earthquake": EARTHQUAKE_CRITICAL, "cyclone": CYCLONE_CRITICAL}[hazard_type]
        details_map = {"flood": flood_details, "earthquake": earthquake_details, "cyclone": cyclone_details}[hazard_type]
        raw_map = {"flood": flood_probability, "earthquake": earthquake_magnitude, "cyclone": cyclone_score}[hazard_type]

        hazard_scores.append(HazardScore(
            hazard_type=hazard_type,
            raw_value=raw_map,
            normalised_score=norm_score,
            weight=weight,
            is_active=norm_score >= ACTIVE_THRESHOLD,
            is_critical=norm_score >= critical_thresh,
            priority=PRIORITY_MAP[hazard_type],
            details=details_map or {},
        ))

    # Sort by priority (earthquake first, then cyclone, then flood)
    hazard_scores.sort(key=lambda h: h.priority)

    # ---- Dominant hazard ----
    # When all scores are equal (especially 0), prefer flood as it's the most common disaster
    # Use positive priority (higher priority number = flood) when scores are tied
    dominant = max(hazard_scores, key=lambda h: (h.normalised_score, h.priority))
    dominant_hazard = dominant.hazard_type

    # ---- Classify level ----
    overall_level = classify_overall_risk(overall_score, previous_level)
    action = risk_level_to_action(overall_level)

    # ---- Alert triggers ----
    alert_triggered, alert_reasons = determine_alerts(
        hazard_scores, overall_level, previous_level
    )

    return AggregatedRisk(
        overall_risk_score=overall_score,
        overall_risk_level=overall_level,
        alert_action=action,
        hazard_scores=hazard_scores,
        dominant_hazard=dominant_hazard,
        active_hazard_count=n_active,
        r_avg=r_avg,
        r_max=r_max,
        r_hybrid=r_hybrid,
        amplifier=amplifier,
        alert_triggered=alert_triggered,
        alert_reasons=alert_reasons,
        latitude=latitude,
        longitude=longitude,
        previous_level=previous_level,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: Aggregate from Service Objects
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_from_services(
    flood_prediction: Optional[Any] = None,
    earthquake_event: Optional[Any] = None,
    cyclone_risk_score: Optional[Any] = None,
    *,
    latitude: float = 0.0,
    longitude: float = 0.0,
    previous_level: Optional[OverallRiskLevel] = None,
) -> AggregatedRisk:
    """
    Convenience wrapper that extracts values from the actual service objects.

    Accepts:
        - FloodPrediction from flood_service.py
        - EarthquakeEvent from earthquake_service.py
        - CycloneRiskScore from cyclone_service.py

    Falls back to zero-risk for any missing hazard.

    Parameters
    ----------
    flood_prediction : FloodPrediction | None
    earthquake_event : EarthquakeEvent | None
    cyclone_risk_score : CycloneRiskScore | None
    latitude, longitude : float
    previous_level : OverallRiskLevel | None

    Returns
    -------
    AggregatedRisk
    """
    # Extract flood probability
    flood_prob = 0.0
    flood_details = None
    if flood_prediction is not None:
        flood_prob = getattr(
            getattr(flood_prediction, "ensemble_result", None),
            "flood_probability", 0.0
        )
        flood_details = {
            "risk_level": getattr(
                getattr(flood_prediction, "ensemble_result", None),
                "risk_level", None
            ),
            "confidence": getattr(
                getattr(flood_prediction, "ensemble_result", None),
                "confidence", None
            ),
        }
        # Convert enum to string if needed
        if flood_details.get("risk_level") and hasattr(flood_details["risk_level"], "value"):
            flood_details["risk_level"] = flood_details["risk_level"].value

    # Extract earthquake data
    eq_mag = 0.0
    eq_depth = 10.0
    eq_details = None
    if earthquake_event is not None:
        eq_mag = getattr(earthquake_event, "magnitude", 0.0)
        eq_depth = getattr(earthquake_event, "depth_km", 10.0)
        eq_details = {
            "magnitude": eq_mag,
            "depth_km": eq_depth,
            "depth_class": getattr(earthquake_event, "depth_class", None),
            "risk_level": getattr(earthquake_event, "risk_level", None),
            "place": getattr(earthquake_event, "place", None),
        }
        # Convert enums
        for key in ("depth_class", "risk_level"):
            if eq_details.get(key) and hasattr(eq_details[key], "value"):
                eq_details[key] = eq_details[key].value

    # Extract cyclone composite score
    cyc_score = 0.0
    cyc_details = None
    if cyclone_risk_score is not None:
        cyc_score = getattr(cyclone_risk_score, "composite_score", 0.0)
        cyc_details = {
            "composite_score": cyc_score,
            "risk_level": getattr(cyclone_risk_score, "risk_level", None),
            "category": getattr(cyclone_risk_score, "category", None),
            "escalation": getattr(cyclone_risk_score, "escalation", None),
        }
        for key in ("risk_level", "category", "escalation"):
            if cyc_details.get(key) and hasattr(cyc_details[key], "value"):
                cyc_details[key] = cyc_details[key].value

    return aggregate_risk(
        flood_probability=flood_prob,
        earthquake_magnitude=eq_mag,
        earthquake_depth_km=eq_depth,
        cyclone_score=cyc_score,
        latitude=latitude,
        longitude=longitude,
        previous_level=previous_level,
        flood_details=flood_details,
        earthquake_details=eq_details,
        cyclone_details=cyc_details,
    )
