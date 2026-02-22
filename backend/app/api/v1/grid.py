"""
FastAPI endpoints for hyper-local grid-based flood simulation.

Endpoints:
    POST /api/v1/grid/simulate         — Run full grid simulation
    GET  /api/v1/grid/simulate-default — Quick simulation with defaults
    POST /api/v1/grid/cell-risk        — Risk for a single cell
    GET  /api/v1/grid/soil-types       — List available soil types
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/grid", tags=["grid-simulation"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class GridSimulateRequest(BaseModel):
    latitude: float = Field(13.08, ge=-90, le=90, description="Centre latitude")
    longitude: float = Field(80.27, ge=-180, le=180, description="Centre longitude")
    grid_size: int = Field(10, ge=3, le=30, description="Grid NxN dimension")
    rainfall_mm_hr: float = Field(40.0, ge=0, le=300, description="Rainfall intensity")
    terrain_style: str = Field(
        "valley",
        description="Terrain style: valley, coastal, flat, urban",
    )
    rainfall_pattern: str = Field(
        "uniform",
        description="Rainfall pattern: uniform, gradient, storm_cell",
    )
    seed: int = Field(42, description="Random seed for reproducibility")


class CellRiskRequest(BaseModel):
    soil_moisture: float = Field(0.45, ge=0, le=1, description="Volumetric moisture (0–1)")
    slope_deg: float = Field(2.0, ge=0, le=90, description="Terrain slope in degrees")
    elevation_m: float = Field(50.0, ge=0, description="Elevation in metres")
    rainfall_mm_hr: float = Field(30.0, ge=0, description="Rainfall intensity mm/hr")
    soil_type: str = Field("loam", description="Soil type: sand, sandy_loam, loam, clay_loam, clay, urban")
    urbanization: float = Field(0.3, ge=0, le=1, description="Urbanization fraction")
    drainage_capacity_mm_hr: float = Field(30.0, ge=0, description="Drain capacity mm/hr")
    drainage_coverage: float = Field(0.5, ge=0, le=1, description="Drain pipe coverage fraction")


class CellRiskResponse(BaseModel):
    risk_score: float
    risk_level: str
    factors: Dict[str, float]
    infiltration_mm_hr: float
    surface_excess_mm_hr: float
    runoff_depth_mm: float
    drainage_efficiency: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/simulate")
async def simulate_grid_endpoint(req: GridSimulateRequest):
    """
    Run a hyper-local 1 km × 1 km flood simulation.

    Generates synthetic terrain, applies rainfall, propagates runoff
    downslope, and returns per-cell risk scores with hotspot analysis.
    """
    valid_terrains = {"valley", "coastal", "flat", "urban"}
    if req.terrain_style not in valid_terrains:
        raise HTTPException(400, f"terrain_style must be one of {valid_terrains}")

    valid_patterns = {"uniform", "gradient", "storm_cell"}
    if req.rainfall_pattern not in valid_patterns:
        raise HTTPException(400, f"rainfall_pattern must be one of {valid_patterns}")

    try:
        from backend.app.ml.grid_model import run_simulation

        result = run_simulation(
            center_lat=req.latitude,
            center_lon=req.longitude,
            grid_size=req.grid_size,
            rainfall_mm_hr=req.rainfall_mm_hr,
            terrain_style=req.terrain_style,
            rainfall_pattern=req.rainfall_pattern,
            seed=req.seed,
        )

        return result.to_dict()

    except Exception as e:
        logger.exception("Grid simulation failed")
        raise HTTPException(500, detail=str(e))


@router.get("/simulate-default")
async def simulate_default(
    lat: float = Query(13.08, ge=-90, le=90),
    lon: float = Query(80.27, ge=-180, le=180),
    rainfall: float = Query(40.0, ge=0, le=300),
    size: int = Query(10, ge=3, le=30),
):
    """Quick simulation with default valley terrain and uniform rainfall."""
    from backend.app.ml.grid_model import run_simulation

    result = run_simulation(
        center_lat=lat,
        center_lon=lon,
        grid_size=size,
        rainfall_mm_hr=rainfall,
        terrain_style="valley",
        rainfall_pattern="uniform",
    )
    return result.to_dict()


@router.post("/cell-risk", response_model=CellRiskResponse)
async def single_cell_risk(req: CellRiskRequest):
    """
    Compute flood risk factors for a single 1 km × 1 km cell.

    Useful for inspecting how individual parameters affect flood risk.
    """
    from backend.app.ml.hydrology import (
        KSAT,
        POROSITY,
        DrainageProfile,
        SoilType,
        compute_cell_risk,
        drainage_efficiency_score,
        infiltration_capacity,
        scs_runoff_depth,
    )

    try:
        soil = SoilType(req.soil_type)
    except ValueError:
        raise HTTPException(400, f"Unknown soil_type: {req.soil_type}")

    ksat = KSAT[soil]
    porosity = POROSITY[soil]

    drainage = DrainageProfile(
        channel_capacity_mm_hr=req.drainage_capacity_mm_hr,
        pipe_coverage_fraction=req.drainage_coverage,
        maintenance_factor=0.8,
    )

    risk, factors = compute_cell_risk(
        soil_moisture=req.soil_moisture,
        slope_deg=req.slope_deg,
        elevation_m=req.elevation_m,
        rainfall_rate_mm_hr=req.rainfall_mm_hr,
        soil_type=soil,
        urbanization=req.urbanization,
        drainage=drainage,
    )

    infil = infiltration_capacity(req.soil_moisture, ksat, porosity)
    excess = max(0.0, req.rainfall_mm_hr - infil)
    runoff = scs_runoff_depth(req.rainfall_mm_hr, curve_number=80)
    drain_eff = drainage_efficiency_score(req.rainfall_mm_hr, drainage)

    risk_label = "minimal"
    if risk >= 0.8:
        risk_label = "critical"
    elif risk >= 0.6:
        risk_label = "high"
    elif risk >= 0.35:
        risk_label = "moderate"
    elif risk >= 0.15:
        risk_label = "low"

    return CellRiskResponse(
        risk_score=round(risk, 4),
        risk_level=risk_label,
        factors=factors.to_dict(),
        infiltration_mm_hr=round(infil, 2),
        surface_excess_mm_hr=round(excess, 2),
        runoff_depth_mm=round(runoff, 2),
        drainage_efficiency=round(drain_eff, 3),
    )


@router.get("/soil-types")
async def list_soil_types():
    """List available soil types with their hydrological properties."""
    from backend.app.ml.hydrology import (
        CURVE_NUMBER,
        KSAT,
        MANNINGS_N,
        POROSITY,
        SoilType,
    )

    types = []
    for st in SoilType:
        types.append({
            "soil_type": st.value,
            "ksat_mm_hr": KSAT[st],
            "porosity": POROSITY[st],
            "curve_number": CURVE_NUMBER[st],
            "mannings_n": MANNINGS_N[st],
            "description": {
                SoilType.SAND: "High infiltration, low runoff. Sandy beaches / deserts.",
                SoilType.SANDY_LOAM: "Good drainage. Agricultural land.",
                SoilType.LOAM: "Moderate infiltration. Mixed-use land.",
                SoilType.CLAY_LOAM: "Poor drainage. Retains water.",
                SoilType.CLAY: "Very low infiltration. High runoff potential.",
                SoilType.URBAN_IMPERVIOUS: "Concrete/asphalt. Near-zero infiltration.",
            }[st],
        })
    return {"soil_types": types}
