"""
hydrology.py — Hydrological computation engine for flood modeling.

Implements the physics-based relationships between:
    • Soil moisture → saturation → infiltration capacity
    • Terrain slope → surface runoff velocity
    • Elevation → gravitational drainage & ponding risk
    • Runoff coefficient → effective rainfall-to-runoff ratio
    • Drainage efficiency → channel capacity vs. water volume

All functions operate on scalars or numpy arrays so they can be
applied per-cell or vectorised across an entire spatial grid.

Reference equations:
    Green-Ampt infiltration (simplified)
    Manning's equation for overland flow
    SCS Curve Number method (adapted)
    Topographic Wetness Index (TWI)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

WATER_DENSITY = 1000.0          # kg/m³
GRAVITY = 9.81                  # m/s²
POROSITY_RANGE = (0.30, 0.55)   # typical soil porosity (sand → clay)
WILTING_POINT = 0.10            # residual moisture fraction
FIELD_CAPACITY = 0.35           # typical field capacity


# ---------------------------------------------------------------------------
# Soil type definitions
# ---------------------------------------------------------------------------


class SoilType(str, Enum):
    """Hydrological soil groups (USDA classification)."""
    SAND = "sand"               # High infiltration, low runoff
    SANDY_LOAM = "sandy_loam"
    LOAM = "loam"               # Moderate
    CLAY_LOAM = "clay_loam"
    CLAY = "clay"               # Low infiltration, high runoff
    URBAN_IMPERVIOUS = "urban"  # Near-zero infiltration


# Saturated hydraulic conductivity Ksat (mm/hr) by soil type
KSAT: Dict[SoilType, float] = {
    SoilType.SAND: 120.0,
    SoilType.SANDY_LOAM: 65.0,
    SoilType.LOAM: 25.0,
    SoilType.CLAY_LOAM: 8.0,
    SoilType.CLAY: 2.5,
    SoilType.URBAN_IMPERVIOUS: 0.5,
}

# Porosity by soil type
POROSITY: Dict[SoilType, float] = {
    SoilType.SAND: 0.43,
    SoilType.SANDY_LOAM: 0.45,
    SoilType.LOAM: 0.47,
    SoilType.CLAY_LOAM: 0.49,
    SoilType.CLAY: 0.52,
    SoilType.URBAN_IMPERVIOUS: 0.10,
}

# SCS Curve Number (AMC-II) by soil type — higher = more runoff
CURVE_NUMBER: Dict[SoilType, float] = {
    SoilType.SAND: 40.0,
    SoilType.SANDY_LOAM: 55.0,
    SoilType.LOAM: 65.0,
    SoilType.CLAY_LOAM: 78.0,
    SoilType.CLAY: 88.0,
    SoilType.URBAN_IMPERVIOUS: 98.0,
}

# Manning's roughness coefficient
MANNINGS_N: Dict[SoilType, float] = {
    SoilType.SAND: 0.020,
    SoilType.SANDY_LOAM: 0.025,
    SoilType.LOAM: 0.035,
    SoilType.CLAY_LOAM: 0.040,
    SoilType.CLAY: 0.050,
    SoilType.URBAN_IMPERVIOUS: 0.015,
}


# ===================================================================
# 1. SOIL MOISTURE & SATURATION
# ===================================================================


def soil_saturation_fraction(
    soil_moisture: float,
    porosity: float = 0.47,
) -> float:
    """
    Compute how saturated the soil is (0 = dry, 1 = fully saturated).

    Saturation fraction:
        S = (θ - θ_wp) / (φ - θ_wp)

    Where:
        θ    = current volumetric moisture content
        θ_wp = wilting point (residual moisture, ~0.10)
        φ    = porosity (total pore space)

    Physical meaning:
        S < 0.4  → soil can absorb significant rainfall
        S 0.4–0.7 → moderate absorption capacity
        S 0.7–0.9 → limited absorption, runoff begins
        S > 0.9  → near-saturated, most rain becomes runoff

    Parameters
    ----------
    soil_moisture : float
        Current volumetric water content (0–1 fraction).
    porosity : float
        Total pore space fraction.

    Returns
    -------
    float
        Saturation fraction ∈ [0, 1].
    """
    effective_range = porosity - WILTING_POINT
    if effective_range <= 0:
        return 1.0  # Impervious
    s = (soil_moisture - WILTING_POINT) / effective_range
    return float(np.clip(s, 0.0, 1.0))


def infiltration_capacity(
    soil_moisture: float,
    ksat: float = 25.0,
    porosity: float = 0.47,
) -> float:
    """
    Simplified Green-Ampt infiltration rate (mm/hr).

    As soil saturates, infiltration drops exponentially:
        f = Ksat × (1 - S)^β

    Where:
        Ksat = saturated hydraulic conductivity (mm/hr)
        S    = saturation fraction
        β    = shape parameter (typically 3.0)

    When S→1, f→0: the soil can no longer absorb water.

    Parameters
    ----------
    soil_moisture : float
        Current volumetric moisture (0–1).
    ksat : float
        Saturated hydraulic conductivity (mm/hr).
    porosity : float
        Soil porosity.

    Returns
    -------
    float
        Infiltration capacity in mm/hr. f ∈ [0, Ksat].
    """
    beta = 3.0  # non-linearity exponent
    s = soil_saturation_fraction(soil_moisture, porosity)
    f = ksat * (1.0 - s) ** beta
    return float(max(f, 0.0))


def saturation_flood_probability(
    soil_moisture: float,
    porosity: float = 0.47,
) -> float:
    """
    Probability modifier from soil saturation.

    Uses a logistic curve centered at S = 0.75:
        P_sat = 1 / (1 + exp(-k × (S - S₀)))

    Physical interpretation:
        • Below S=0.5: soil absorbs most rainfall → low flood risk
        • S=0.75: inflection point — infiltration declining rapidly
        • Above S=0.9: soil essentially impervious → flood likely

    Returns
    -------
    float
        Flood probability factor from saturation ∈ [0, 1].
    """
    s = soil_saturation_fraction(soil_moisture, porosity)
    k = 10.0     # steepness
    s0 = 0.75    # inflection point
    return float(1.0 / (1.0 + math.exp(-k * (s - s0))))


# ===================================================================
# 2. TERRAIN SLOPE
# ===================================================================


def compute_slope_degrees(
    elevation_center: float,
    elevation_neighbors: List[float],
    cell_size_m: float = 1000.0,
) -> float:
    """
    Compute terrain slope from a cell and its neighbours.

    Uses the maximum gradient method:
        slope = arctan(max |Δh| / Δx)

    Parameters
    ----------
    elevation_center : float
        Elevation at the grid cell centre (metres).
    elevation_neighbors : list of float
        Elevations of adjacent cells (4- or 8-connected).
    cell_size_m : float
        Grid resolution in metres (default 1000 = 1 km).

    Returns
    -------
    float
        Slope angle in degrees.
    """
    if not elevation_neighbors:
        return 0.0
    max_gradient = max(
        abs(elevation_center - n) / cell_size_m
        for n in elevation_neighbors
    )
    return float(math.degrees(math.atan(max_gradient)))


def slope_runoff_factor(slope_deg: float) -> float:
    """
    How terrain slope affects surface runoff.

    Steeper slopes → faster runoff → water concentrates downstream.
    Flat terrain → ponding but slower drainage.

    Relationship (empirical fit):
        R_slope = 1 + α × sin(slope)  for slope effect on velocity
        But for FLOOD RISK at a cell:
        - Flat (0–1°): HIGH ponding risk, water cannot drain
        - Gentle (1–5°): MODERATE, water moves but slowly
        - Moderate (5–15°): LOWER risk at this cell (water drains away)
        - Steep (>15°): LOW risk here but HIGH risk DOWNSTREAM

    For local cell risk:
        f_slope = 1.0 - 0.6 × tanh(slope / 8)  + 0.2 × flat_bonus

    Returns
    -------
    float
        Slope-based local flood risk factor ∈ [0.2, 1.2].
    """
    # Flat terrain bonus: water pools
    flat_bonus = math.exp(-slope_deg ** 2 / 2.0)  # Gaussian peaked at 0°

    # Slope drainage benefit: steeper = drains faster
    drainage_benefit = 0.6 * math.tanh(slope_deg / 8.0)

    factor = 1.0 - drainage_benefit + 0.2 * flat_bonus
    return float(np.clip(factor, 0.2, 1.2))


def downstream_runoff_contribution(
    slope_deg: float,
    rainfall_mm: float,
    runoff_coeff: float = 0.5,
) -> float:
    """
    Runoff that flows FROM this cell TO its downslope neighbour.

    Uses simplified Manning's-derived velocity:
        Q_out = C × R × sin(θ)^0.5

    Where C = runoff coefficient, R = rainfall, θ = slope.

    Returns
    -------
    float
        Outgoing runoff depth in mm.
    """
    slope_rad = math.radians(max(slope_deg, 0.01))
    velocity_factor = math.sqrt(math.sin(slope_rad))
    return float(runoff_coeff * rainfall_mm * velocity_factor)


# ===================================================================
# 3. ELEVATION RISK SCORE
# ===================================================================


def elevation_risk_score(
    elevation_m: float,
    regional_min: float = 0.0,
    regional_max: float = 500.0,
) -> float:
    """
    Risk score based on relative elevation within the region.

    Lower elevation → higher risk (water flows downhill).

    Score:
        E_risk = 1 - (h - h_min) / (h_max - h_min)

    So elevation=h_min → risk=1.0, elevation=h_max → risk=0.0.

    Parameters
    ----------
    elevation_m : float
        Cell elevation in metres.
    regional_min, regional_max : float
        Min/max elevation across the study area.

    Returns
    -------
    float
        Elevation risk score ∈ [0, 1].
    """
    elev_range = regional_max - regional_min
    if elev_range <= 0:
        return 0.5  # flat terrain, uniform risk
    normalised = (elevation_m - regional_min) / elev_range
    return float(np.clip(1.0 - normalised, 0.0, 1.0))


def topographic_wetness_index(
    contributing_area_m2: float,
    slope_deg: float,
    cell_size_m: float = 1000.0,
) -> float:
    """
    Topographic Wetness Index (TWI) — Beven & Kirkby (1979).

    TWI = ln(a / tan(β))

    Where:
        a = specific upslope contributing area (m²/m)
        β = local slope in radians

    Higher TWI → more water accumulation → higher flood risk.
    Typical range: 5–20 for natural terrain.

    Parameters
    ----------
    contributing_area_m2 : float
        Upslope area draining through this cell (m²).
    slope_deg : float
        Local slope in degrees.
    cell_size_m : float
        Grid cell width for computing specific area.

    Returns
    -------
    float
        TWI value (higher = wetter / more flood-prone).
    """
    specific_area = contributing_area_m2 / cell_size_m
    slope_rad = math.radians(max(slope_deg, 0.1))  # avoid tan(0)
    return float(math.log(specific_area / math.tan(slope_rad)))


# ===================================================================
# 4. RUNOFF COEFFICIENT
# ===================================================================


def scs_runoff_depth(
    rainfall_mm: float,
    curve_number: float = 65.0,
) -> float:
    """
    SCS Curve Number method for runoff estimation.

    Q = (P - Ia)² / (P - Ia + S)    when P > Ia
    Q = 0                            when P ≤ Ia

    Where:
        P  = rainfall depth (mm)
        S  = potential maximum retention = 25400/CN - 254  (mm)
        Ia = initial abstraction ≈ 0.2 × S

    Higher CN → more runoff (urbanised, clay soils).
    Lower CN → more infiltration (sandy soils, forested).

    Returns
    -------
    float
        Runoff depth Q in mm.
    """
    if curve_number >= 100:
        return rainfall_mm  # fully impervious
    s = (25400.0 / curve_number) - 254.0
    ia = 0.2 * s
    if rainfall_mm <= ia:
        return 0.0
    q = (rainfall_mm - ia) ** 2 / (rainfall_mm - ia + s)
    return float(max(q, 0.0))


def composite_runoff_coefficient(
    soil_type: SoilType,
    urbanization: float,
    soil_moisture: float,
) -> float:
    """
    Composite runoff coefficient C ∈ [0, 1].

    Combines:
        • Soil type base coefficient
        • Urbanization impervious-fraction boost
        • Antecedent moisture adjustment

    C = C_base × (1 + 0.3 × urban) × (1 + 0.5 × S_sat)

    Where C_base derives from the curve number.

    Returns
    -------
    float
        Effective runoff coefficient ∈ [0, 1].
    """
    cn = CURVE_NUMBER[soil_type]
    c_base = cn / 100.0  # normalise to 0–1 range
    porosity = POROSITY[soil_type]
    s_sat = soil_saturation_fraction(soil_moisture, porosity)

    c = c_base * (1.0 + 0.3 * urbanization) * (1.0 + 0.5 * s_sat)
    return float(np.clip(c, 0.0, 1.0))


# ===================================================================
# 5. DRAINAGE EFFICIENCY
# ===================================================================


@dataclass
class DrainageProfile:
    """Drainage system characterisation for a grid cell."""
    channel_capacity_mm_hr: float = 30.0   # max throughput
    pipe_coverage_fraction: float = 0.5    # fraction of cell with drains
    maintenance_factor: float = 0.8        # 1.0 = perfect, 0.0 = blocked
    natural_drainage_slope: float = 2.0    # degrees

    @property
    def effective_capacity(self) -> float:
        """Effective drainage in mm/hr accounting for coverage and maintenance."""
        return (
            self.channel_capacity_mm_hr
            * self.pipe_coverage_fraction
            * self.maintenance_factor
        )


def drainage_efficiency_score(
    rainfall_rate_mm_hr: float,
    drainage: DrainageProfile,
) -> float:
    """
    Ratio of drainage capacity to demand.

    η = D_effective / max(R, ε)

    Where:
        D_effective = channel_capacity × coverage × maintenance
        R = rainfall rate (mm/hr)

    η > 1.0 → drainage can handle the rain
    η < 1.0 → system overwhelmed, flooding likely
    η < 0.5 → severe flooding

    Returns
    -------
    float
        Drainage efficiency ratio (capped at 2.0).
    """
    demand = max(rainfall_rate_mm_hr, 0.01)
    efficiency = drainage.effective_capacity / demand
    return float(min(efficiency, 2.0))


def drainage_overflow_mm(
    rainfall_rate_mm_hr: float,
    drainage: DrainageProfile,
    duration_hr: float = 1.0,
) -> float:
    """
    Excess water that cannot be drained (mm).

    Overflow = max(0, R - D_eff) × t

    Returns
    -------
    float
        Accumulated overflow in mm over the duration.
    """
    excess_rate = max(0.0, rainfall_rate_mm_hr - drainage.effective_capacity)
    return float(excess_rate * duration_hr)


# ===================================================================
# 6. COMPOSITE CELL RISK SCORING
# ===================================================================


@dataclass
class CellRiskFactors:
    """All risk factor components for a single grid cell."""
    saturation_risk: float = 0.0       # from soil moisture
    slope_risk: float = 0.0            # from terrain slope
    elevation_risk: float = 0.0        # from relative elevation
    runoff_risk: float = 0.0           # from runoff coefficient
    drainage_risk: float = 0.0         # from drainage deficiency
    rainfall_intensity: float = 0.0    # mm/hr current/forecast

    def to_dict(self) -> Dict[str, float]:
        return {
            "saturation_risk": round(self.saturation_risk, 4),
            "slope_risk": round(self.slope_risk, 4),
            "elevation_risk": round(self.elevation_risk, 4),
            "runoff_risk": round(self.runoff_risk, 4),
            "drainage_risk": round(self.drainage_risk, 4),
            "rainfall_intensity": round(self.rainfall_intensity, 4),
        }


# Weights for composite risk score
RISK_WEIGHTS: Dict[str, float] = {
    "saturation": 0.25,
    "slope": 0.10,
    "elevation": 0.20,
    "runoff": 0.20,
    "drainage": 0.25,
}


def compute_cell_risk(
    soil_moisture: float,
    slope_deg: float,
    elevation_m: float,
    rainfall_rate_mm_hr: float,
    soil_type: SoilType = SoilType.LOAM,
    urbanization: float = 0.5,
    drainage: Optional[DrainageProfile] = None,
    regional_min_elev: float = 0.0,
    regional_max_elev: float = 500.0,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, CellRiskFactors]:
    """
    Compute composite flood risk for a single 1km×1km grid cell.

    The final risk score is a weighted combination:
        R = w₁·S_sat + w₂·F_slope + w₃·E_risk + w₄·C_runoff + w₅·D_def

    Where each factor ∈ [0, 1] and weights sum to 1.0.

    Parameters
    ----------
    soil_moisture : float
        Volumetric water content (0–1).
    slope_deg : float
        Average terrain slope in degrees.
    elevation_m : float
        Mean elevation of the cell in metres.
    rainfall_rate_mm_hr : float
        Current or forecast rainfall intensity.
    soil_type : SoilType
        Hydrological soil group.
    urbanization : float
        Urban impervious fraction (0–1).
    drainage : DrainageProfile or None
        Drainage infrastructure. Defaults to moderate system.
    regional_min_elev, regional_max_elev : float
        Min/max elevation of the study region.
    weights : dict or None
        Override risk-factor weights.

    Returns
    -------
    (risk_score, CellRiskFactors)
        risk_score ∈ [0, 1], plus the individual components.
    """
    if drainage is None:
        drainage = DrainageProfile()

    w = weights or RISK_WEIGHTS
    porosity = POROSITY[soil_type]

    # --- Factor 1: Saturation risk ---
    sat_risk = saturation_flood_probability(soil_moisture, porosity)

    # --- Factor 2: Slope risk (local ponding) ---
    slope_factor = slope_runoff_factor(slope_deg)
    # Normalise to 0–1 (slope_runoff_factor range is 0.2–1.2)
    slope_risk = (slope_factor - 0.2) / 1.0

    # --- Factor 3: Elevation risk ---
    elev_risk = elevation_risk_score(elevation_m, regional_min_elev, regional_max_elev)

    # --- Factor 4: Runoff risk ---
    c_runoff = composite_runoff_coefficient(soil_type, urbanization, soil_moisture)
    # Runoff risk proportional to how much rain becomes surface flow
    runoff_risk = c_runoff  # already ∈ [0, 1]

    # --- Factor 5: Drainage deficiency risk ---
    eta = drainage_efficiency_score(rainfall_rate_mm_hr, drainage)
    # Invert: low efficiency → high risk
    drainage_risk = float(np.clip(1.0 - eta, 0.0, 1.0))

    # --- Composite score ---
    risk_score = (
        w["saturation"] * sat_risk
        + w["slope"] * slope_risk
        + w["elevation"] * elev_risk
        + w["runoff"] * runoff_risk
        + w["drainage"] * drainage_risk
    )

    # Rainfall intensity multiplier (amplifies risk when it's raining hard)
    if rainfall_rate_mm_hr > 0:
        rain_mult = min(1.0 + 0.3 * math.log1p(rainfall_rate_mm_hr / 10.0), 2.0)
        risk_score *= rain_mult

    risk_score = float(np.clip(risk_score, 0.0, 1.0))

    factors = CellRiskFactors(
        saturation_risk=sat_risk,
        slope_risk=slope_risk,
        elevation_risk=elev_risk,
        runoff_risk=runoff_risk,
        drainage_risk=drainage_risk,
        rainfall_intensity=rainfall_rate_mm_hr,
    )

    return risk_score, factors
