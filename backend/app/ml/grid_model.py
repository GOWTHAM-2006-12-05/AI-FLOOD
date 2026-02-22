"""
grid_model.py — 1 km × 1 km hyper-local spatial flood grid simulation.

Simulates a 2-D grid of terrain cells, each carrying:
    • Elevation (metres)
    • Slope (degrees)
    • Soil type & moisture
    • Urbanization fraction
    • Drainage profile
    • Rainfall intensity

The simulator:
    1. Generates (or loads) a terrain grid
    2. Applies per-cell hydrology calculations
    3. Propagates runoff downslope between cells
    4. Computes composite risk scores per cell
    5. Returns a risk heat-map & ranked hotspot list

Extension path:
    Replace the synthetic terrain generator with a real DEM
    (SRTM / Copernicus 30 m tiles) and satellite soil-moisture
    (SMAP / Sentinel-1) for production-grade resolution.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.app.ml.hydrology import (
    CURVE_NUMBER,
    KSAT,
    POROSITY,
    CellRiskFactors,
    DrainageProfile,
    SoilType,
    composite_runoff_coefficient,
    compute_cell_risk,
    downstream_runoff_contribution,
    drainage_overflow_mm,
    elevation_risk_score,
    infiltration_capacity,
    scs_runoff_depth,
    slope_runoff_factor,
    soil_saturation_fraction,
    topographic_wetness_index,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Grid cell
# ===================================================================


@dataclass
class GridCell:
    """Single 1 km × 1 km cell in the spatial grid."""

    row: int
    col: int
    latitude: float                 # centre-point lat
    longitude: float                # centre-point lon
    elevation_m: float = 50.0       # metres above sea level
    slope_deg: float = 2.0          # terrain gradient
    soil_type: SoilType = SoilType.LOAM
    soil_moisture: float = 0.35     # volumetric fraction (0–1)
    urbanization: float = 0.3       # impervious fraction (0–1)
    drainage: DrainageProfile = field(default_factory=DrainageProfile)
    rainfall_mm_hr: float = 0.0     # current/forecast rainfall

    # ---- Computed results (filled after simulation) ----
    risk_score: float = 0.0
    risk_factors: Optional[CellRiskFactors] = None
    runoff_received_mm: float = 0.0  # inflow from upslope neighbours
    water_depth_mm: float = 0.0      # accumulated standing water

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row": self.row,
            "col": self.col,
            "lat": round(self.latitude, 6),
            "lon": round(self.longitude, 6),
            "elevation_m": round(self.elevation_m, 1),
            "slope_deg": round(self.slope_deg, 2),
            "soil_type": self.soil_type.value,
            "soil_moisture": round(self.soil_moisture, 3),
            "urbanization": round(self.urbanization, 2),
            "rainfall_mm_hr": round(self.rainfall_mm_hr, 1),
            "risk_score": round(self.risk_score, 4),
            "risk_factors": self.risk_factors.to_dict() if self.risk_factors else None,
            "runoff_received_mm": round(self.runoff_received_mm, 2),
            "water_depth_mm": round(self.water_depth_mm, 2),
        }


# ===================================================================
# Grid result
# ===================================================================


@dataclass
class GridSimulationResult:
    """Output of a complete grid simulation."""

    grid_rows: int
    grid_cols: int
    cell_size_km: float
    center_lat: float
    center_lon: float
    cells: List[List[GridCell]]
    risk_heatmap: Optional[np.ndarray] = None  # (rows, cols) risk scores
    hotspots: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for API response (cell data + stats, no numpy)."""
        flat_cells = []
        for row in self.cells:
            for cell in row:
                flat_cells.append(cell.to_dict())

        return {
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
            "cell_size_km": self.cell_size_km,
            "center": {"lat": self.center_lat, "lon": self.center_lon},
            "total_cells": self.grid_rows * self.grid_cols,
            "stats": self.stats,
            "hotspots": self.hotspots,
            "cells": flat_cells,
        }


# ===================================================================
# Synthetic terrain generator
# ===================================================================


def generate_terrain(
    center_lat: float,
    center_lon: float,
    rows: int = 10,
    cols: int = 10,
    cell_size_km: float = 1.0,
    seed: int = 42,
    terrain_style: str = "valley",
) -> List[List[GridCell]]:
    """
    Generate a synthetic terrain grid centred on (lat, lon).

    Terrain styles:
        'valley'  — Central low-lying area surrounded by hills
        'coastal' — Elevation decreases toward one edge (coast)
        'flat'    — Mostly flat with random micro-variations
        'urban'   — Flat + high urbanization in centre

    Each cell gets:
        • Computed lat/lon based on offset from centre
        • Elevation from the terrain model
        • Slope computed from neighbouring elevations
        • Randomised soil type, moisture, urbanization, drainage

    Parameters
    ----------
    center_lat, center_lon : float
        Geographic centre of the grid.
    rows, cols : int
        Grid dimensions (default 10×10 = 100 cells = 100 km²).
    cell_size_km : float
        Side length of each cell in km.
    seed : int
        Random seed for reproducibility.
    terrain_style : str
        One of 'valley', 'coastal', 'flat', 'urban'.

    Returns
    -------
    list of list of GridCell
        2-D grid [row][col].
    """
    rng = np.random.RandomState(seed)

    # --- 1. Generate elevation matrix ---
    elev = _generate_elevation(rows, cols, terrain_style, rng)

    # --- 2. Soil type distribution ---
    soil_types = list(SoilType)
    soil_weights = [0.10, 0.20, 0.30, 0.20, 0.15, 0.05]
    soil_grid = rng.choice(
        len(soil_types), size=(rows, cols), p=soil_weights
    )

    # --- 3. Build grid cells ---
    # 1° latitude ≈ 111 km → 1 km ≈ 0.009°
    km_to_deg_lat = 1.0 / 111.0
    km_to_deg_lon = 1.0 / (111.0 * math.cos(math.radians(center_lat)))

    offset_lat = (rows - 1) / 2.0 * cell_size_km * km_to_deg_lat
    offset_lon = (cols - 1) / 2.0 * cell_size_km * km_to_deg_lon

    grid: List[List[GridCell]] = []

    for r in range(rows):
        row_cells: List[GridCell] = []
        for c in range(cols):
            lat = center_lat + (r * cell_size_km * km_to_deg_lat) - offset_lat
            lon = center_lon + (c * cell_size_km * km_to_deg_lon) - offset_lon

            st = soil_types[soil_grid[r, c]]

            # Soil moisture — wetter in valleys (low elevation)
            elev_norm = (elev[r, c] - elev.min()) / (elev.max() - elev.min() + 1e-8)
            base_moisture = 0.25 + 0.45 * (1.0 - elev_norm)
            soil_moist = float(np.clip(
                base_moisture + rng.normal(0, 0.08), 0.05, 0.95
            ))

            # Urbanization — higher near grid centre
            dist_from_center = math.sqrt(
                ((r - rows / 2) / rows) ** 2 + ((c - cols / 2) / cols) ** 2
            )
            if terrain_style == "urban":
                urban = float(np.clip(0.9 - 1.2 * dist_from_center + rng.normal(0, 0.05), 0, 1))
            else:
                urban = float(np.clip(0.2 + rng.exponential(0.15) - 0.5 * dist_from_center, 0, 0.9))

            # Drainage — better in urban areas (engineered), worse in rural
            drain = DrainageProfile(
                channel_capacity_mm_hr=20.0 + 30.0 * urban + rng.normal(0, 5),
                pipe_coverage_fraction=0.2 + 0.6 * urban,
                maintenance_factor=float(np.clip(0.5 + rng.normal(0.3, 0.1), 0.2, 1.0)),
                natural_drainage_slope=max(0.1, elev_norm * 5.0),
            )

            cell = GridCell(
                row=r,
                col=c,
                latitude=lat,
                longitude=lon,
                elevation_m=float(elev[r, c]),
                slope_deg=0.0,  # computed below
                soil_type=st,
                soil_moisture=soil_moist,
                urbanization=urban,
                drainage=drain,
            )
            row_cells.append(cell)
        grid.append(row_cells)

    # --- 4. Compute slopes from the elevation matrix ---
    for r in range(rows):
        for c in range(cols):
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbors.append(float(elev[nr, nc]))
            grid[r][c].slope_deg = _slope_from_neighbors(
                float(elev[r, c]), neighbors, cell_size_km * 1000
            )

    return grid


def _generate_elevation(
    rows: int,
    cols: int,
    style: str,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generate a 2-D elevation surface (metres)."""
    r_axis = np.linspace(-1, 1, rows)
    c_axis = np.linspace(-1, 1, cols)
    R, C = np.meshgrid(r_axis, c_axis, indexing="ij")

    if style == "valley":
        # Bowl shape: high edges, low centre
        elev = 80 + 200 * (R ** 2 + C ** 2) + rng.normal(0, 8, (rows, cols))

    elif style == "coastal":
        # Slope from west (high) to east (low/coast)
        elev = 150 - 100 * C + rng.normal(0, 10, (rows, cols))

    elif style == "urban":
        # Mostly flat with slight variations
        elev = 20 + 15 * np.abs(R) + rng.normal(0, 3, (rows, cols))

    else:  # "flat"
        elev = 30 + rng.normal(0, 5, (rows, cols))

    return np.clip(elev, 0, 1000).astype(float)


def _slope_from_neighbors(
    center_elev: float,
    neighbor_elevs: List[float],
    cell_size_m: float,
) -> float:
    """Compute slope in degrees from maximum gradient to neighbours."""
    if not neighbor_elevs:
        return 0.0
    max_grad = max(abs(center_elev - n) / cell_size_m for n in neighbor_elevs)
    return float(math.degrees(math.atan(max_grad)))


# ===================================================================
# Grid simulation engine
# ===================================================================


def simulate_grid(
    grid: List[List[GridCell]],
    rainfall_mm_hr: float = 30.0,
    rainfall_pattern: str = "uniform",
    propagation_steps: int = 3,
    cell_size_km: float = 1.0,
    seed: int = 42,
) -> GridSimulationResult:
    """
    Run the full hyper-local flood simulation on a terrain grid.

    Pipeline per cell:
        1. Apply rainfall (uniform or spatial pattern)
        2. Compute infiltration → excess surface water
        3. Compute per-cell risk factors
        4. Propagate runoff downslope (iterative)
        5. Re-evaluate water depth & risk with accumulated runoff
        6. Build risk heat-map & extract hotspots

    Parameters
    ----------
    grid : list[list[GridCell]]
        2-D terrain grid from generate_terrain().
    rainfall_mm_hr : float
        Base rainfall intensity (mm/hr).
    rainfall_pattern : str
        'uniform' — same rain everywhere.
        'gradient' — decreases from NW to SE.
        'storm_cell' — intense centre, lighter edges.
    propagation_steps : int
        Number of downslope runoff propagation iterations.
    cell_size_km : float
        Grid resolution in km.
    seed : int
        Random seed for stochastic elements.

    Returns
    -------
    GridSimulationResult
    """
    rng = np.random.RandomState(seed)
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # --- Step 1: Apply rainfall spatial pattern ---
    _apply_rainfall(grid, rainfall_mm_hr, rainfall_pattern, rng)

    # Get elevation range for relative scoring
    all_elevs = [grid[r][c].elevation_m for r in range(rows) for c in range(cols)]
    elev_min, elev_max = min(all_elevs), max(all_elevs)

    # --- Step 2 & 3: Per-cell hydrology + risk ---
    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]

            # Infiltration capacity
            ksat = KSAT[cell.soil_type]
            porosity = POROSITY[cell.soil_type]
            infil = infiltration_capacity(cell.soil_moisture, ksat, porosity)

            # Surface water = rainfall - infiltration
            surface_water = max(0.0, cell.rainfall_mm_hr - infil)
            cell.water_depth_mm = surface_water

            # Compute composite risk
            risk, factors = compute_cell_risk(
                soil_moisture=cell.soil_moisture,
                slope_deg=cell.slope_deg,
                elevation_m=cell.elevation_m,
                rainfall_rate_mm_hr=cell.rainfall_mm_hr,
                soil_type=cell.soil_type,
                urbanization=cell.urbanization,
                drainage=cell.drainage,
                regional_min_elev=elev_min,
                regional_max_elev=elev_max,
            )
            cell.risk_score = risk
            cell.risk_factors = factors

    # --- Step 4: Downslope runoff propagation ---
    for step in range(propagation_steps):
        _propagate_runoff(grid, cell_size_km)

    # --- Step 5: Recalculate risk with accumulated water ---
    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]
            if cell.runoff_received_mm > 0:
                # Effective rainfall includes received runoff
                effective_rain = cell.rainfall_mm_hr + cell.runoff_received_mm
                risk, factors = compute_cell_risk(
                    soil_moisture=min(cell.soil_moisture + 0.05 * step, 0.95),
                    slope_deg=cell.slope_deg,
                    elevation_m=cell.elevation_m,
                    rainfall_rate_mm_hr=effective_rain,
                    soil_type=cell.soil_type,
                    urbanization=cell.urbanization,
                    drainage=cell.drainage,
                    regional_min_elev=elev_min,
                    regional_max_elev=elev_max,
                )
                cell.risk_score = max(cell.risk_score, risk)
                cell.risk_factors = factors
                cell.water_depth_mm += cell.runoff_received_mm

    # --- Step 6: Build heat-map & hotspots ---
    risk_heatmap = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            risk_heatmap[r, c] = grid[r][c].risk_score

    # Extract hotspot cells (risk > 0.6)
    hotspots = []
    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]
            if cell.risk_score >= 0.6:
                hotspots.append({
                    "row": r,
                    "col": c,
                    "lat": round(cell.latitude, 6),
                    "lon": round(cell.longitude, 6),
                    "risk_score": round(cell.risk_score, 4),
                    "water_depth_mm": round(cell.water_depth_mm, 1),
                    "elevation_m": round(cell.elevation_m, 1),
                    "risk_level": _risk_label(cell.risk_score),
                })

    hotspots.sort(key=lambda h: h["risk_score"], reverse=True)

    # --- Stats ---
    risk_flat = risk_heatmap.flatten()
    stats = {
        "mean_risk": round(float(risk_flat.mean()), 4),
        "max_risk": round(float(risk_flat.max()), 4),
        "min_risk": round(float(risk_flat.min()), 4),
        "std_risk": round(float(risk_flat.std()), 4),
        "cells_high_risk": int((risk_flat >= 0.6).sum()),
        "cells_critical": int((risk_flat >= 0.8).sum()),
        "total_cells": rows * cols,
        "pct_high_risk": round(float((risk_flat >= 0.6).mean()) * 100, 1),
        "rainfall_mm_hr": rainfall_mm_hr,
        "rainfall_pattern": rainfall_pattern,
    }

    # Compute centre from grid
    center_lat = grid[rows // 2][cols // 2].latitude
    center_lon = grid[rows // 2][cols // 2].longitude

    return GridSimulationResult(
        grid_rows=rows,
        grid_cols=cols,
        cell_size_km=cell_size_km,
        center_lat=center_lat,
        center_lon=center_lon,
        cells=grid,
        risk_heatmap=risk_heatmap,
        hotspots=hotspots,
        stats=stats,
    )


# ===================================================================
# Rainfall patterns
# ===================================================================


def _apply_rainfall(
    grid: List[List[GridCell]],
    base_mm_hr: float,
    pattern: str,
    rng: np.random.RandomState,
) -> None:
    """Set per-cell rainfall according to the spatial pattern."""
    rows = len(grid)
    cols = len(grid[0])

    for r in range(rows):
        for c in range(cols):
            if pattern == "uniform":
                rain = base_mm_hr + rng.normal(0, base_mm_hr * 0.05)

            elif pattern == "gradient":
                # NW corner gets 1.5× base, SE gets 0.5×
                frac_r = r / max(rows - 1, 1)
                frac_c = c / max(cols - 1, 1)
                factor = 1.5 - 1.0 * (frac_r + frac_c) / 2.0
                rain = base_mm_hr * factor + rng.normal(0, 2)

            elif pattern == "storm_cell":
                # Gaussian storm centred on grid
                dist = math.sqrt(
                    ((r - rows / 2) / (rows / 2)) ** 2
                    + ((c - cols / 2) / (cols / 2)) ** 2
                )
                rain = base_mm_hr * math.exp(-dist ** 2 / 0.5) + rng.normal(0, 2)

            else:
                rain = base_mm_hr

            grid[r][c].rainfall_mm_hr = max(0.0, rain)


# ===================================================================
# Runoff propagation
# ===================================================================


def _propagate_runoff(
    grid: List[List[GridCell]],
    cell_size_km: float,
) -> None:
    """
    Propagate excess water downslope using D8 flow direction.

    Each cell sends a fraction of its surface water to its
    lowest-elevation neighbour. The fraction depends on the
    slope between them and the runoff coefficient.
    """
    rows = len(grid)
    cols = len(grid[0])

    # Store outflows to avoid read-write conflicts
    inflows = np.zeros((rows, cols))

    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]
            if cell.water_depth_mm <= 0.1:
                continue

            # Find steepest downslope neighbour (D8 algorithm)
            best_drop = 0.0
            best_nr, best_nc = -1, -1

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    drop = cell.elevation_m - grid[nr][nc].elevation_m
                    if drop > best_drop:
                        best_drop = drop
                        best_nr, best_nc = nr, nc

            if best_nr < 0:
                continue  # local minimum — water ponds

            # Runoff fraction depends on slope
            slope_rad = math.atan(best_drop / (cell_size_km * 1000))
            transfer_frac = min(0.6 * math.sin(slope_rad) ** 0.5, 0.8)

            outflow = cell.water_depth_mm * transfer_frac
            inflows[best_nr, best_nc] += outflow
            cell.water_depth_mm -= outflow

    # Apply inflows
    for r in range(rows):
        for c in range(cols):
            grid[r][c].runoff_received_mm += inflows[r, c]


def _risk_label(score: float) -> str:
    """Human-readable risk level from 0–1 score."""
    if score < 0.15:
        return "minimal"
    elif score < 0.35:
        return "low"
    elif score < 0.60:
        return "moderate"
    elif score < 0.80:
        return "high"
    else:
        return "critical"


# ===================================================================
# Quick-run convenience
# ===================================================================


def run_simulation(
    center_lat: float = 13.08,
    center_lon: float = 80.27,
    grid_size: int = 10,
    rainfall_mm_hr: float = 40.0,
    terrain_style: str = "valley",
    rainfall_pattern: str = "uniform",
    seed: int = 42,
) -> GridSimulationResult:
    """
    One-call convenience: generate terrain → simulate → return results.

    Defaults to Chennai (13.08°N, 80.27°E) with 10×10 km grid,
    valley terrain, and 40 mm/hr rainfall.
    """
    grid = generate_terrain(
        center_lat=center_lat,
        center_lon=center_lon,
        rows=grid_size,
        cols=grid_size,
        cell_size_km=1.0,
        seed=seed,
        terrain_style=terrain_style,
    )

    result = simulate_grid(
        grid=grid,
        rainfall_mm_hr=rainfall_mm_hr,
        rainfall_pattern=rainfall_pattern,
        propagation_steps=3,
        cell_size_km=1.0,
        seed=seed,
    )

    logger.info(
        "Grid simulation complete: %dx%d cells, %d hotspots, mean_risk=%.3f",
        result.grid_rows,
        result.grid_cols,
        len(result.hotspots),
        result.stats["mean_risk"],
    )

    return result
