# Hyper-Local Flood Modeling at 1 km × 1 km Resolution

## System Overview

This module implements a **simulated spatial flood model** operating at 1 km × 1 km
grid resolution. Each grid cell carries terrain, soil, drainage, and rainfall
attributes that feed into physics-based hydrological equations to produce a
composite flood risk score.

```
┌──────────────────────────────────────────────────────────────┐
│                  SPATIAL FLOOD GRID (10×10)                  │
│                                                              │
│   ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐                │
│   │0.1│0.2│0.2│0.3│0.3│0.3│0.2│0.2│0.1│0.1│  ← Elevation  │
│   ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤     ridge      │
│   │0.2│0.3│0.4│0.5│0.5│0.5│0.4│0.3│0.2│0.1│                │
│   ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤                │
│   │0.3│0.4│0.6│0.7│0.8│0.8│0.7│0.5│0.3│0.2│                │
│   ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤  ← Valley     │
│   │0.3│0.5│0.7│0.9│1.0│1.0│0.8│0.6│0.4│0.3│     floor     │
│   ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤   (HIGH RISK) │
│   │0.2│0.4│0.6│0.8│0.9│0.9│0.7│0.5│0.3│0.2│                │
│   ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤                │
│   │0.2│0.3│0.4│0.5│0.5│0.5│0.4│0.3│0.2│0.1│                │
│   └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘                │
│                                                              │
│   Values = composite flood risk score (0–1)                  │
│   Water flows from ridges → valley → accumulates at centre   │
└──────────────────────────────────────────────────────────────┘
```

---

## 1. How Soil Saturation Increases Flood Probability

### The Physics

Soil acts as a **sponge** — it can absorb rainfall up to its pore capacity.
Once saturated, every additional drop becomes surface runoff.

```
                INFILTRATION vs. SATURATION
    
    Infiltration
    Rate (mm/hr)
    ▲
    │  ████
    │  ████████
    │  ████████████
    │  ████████████████
    │  ████████████████████
    │  ████████████████████████▄▄▄▄
    │  ████████████████████████████████▄▄▄▄▄▄▄▄▄▄
    └──────────────────────────────────────────────►
       0%    20%   40%   60%   80%   100%
                 Soil Saturation (S)
    
    As saturation → 1.0, infiltration → 0
    The relationship is NON-LINEAR (exponential decay)
```

### Mathematical Model

**Saturation fraction:**

$$S = \frac{\theta - \theta_{wp}}{\phi - \theta_{wp}}$$

Where:
- $\theta$ = current volumetric moisture content
- $\theta_{wp}$ = wilting point (residual moisture, ≈ 0.10)
- $\phi$ = porosity (total pore space, varies by soil type)

**Infiltration capacity** (simplified Green-Ampt):

$$f = K_{sat} \times (1 - S)^{\beta}$$

Where $K_{sat}$ is saturated hydraulic conductivity and $\beta = 3.0$.

**Flood probability from saturation** (logistic):

$$P_{sat} = \frac{1}{1 + e^{-k(S - S_0)}}$$

With inflection $S_0 = 0.75$ and steepness $k = 10$.

### Key Thresholds

| Saturation Range | Infiltration State | Flood Risk |
|---|---|---|
| S < 0.4 | Soil absorbs most rainfall | Minimal |
| 0.4 ≤ S < 0.7 | Absorption declining | Low–Moderate |
| 0.7 ≤ S < 0.9 | Most rain becomes runoff | High |
| S ≥ 0.9 | Effectively impervious | Critical |

### Soil Type Impact

| Soil Type | K_sat (mm/hr) | Porosity | Curve Number | Character |
|---|---|---|---|---|
| Sand | 120.0 | 0.43 | 40 | Rapid drainage, low runoff |
| Sandy Loam | 65.0 | 0.45 | 55 | Good drainage |
| Loam | 25.0 | 0.47 | 65 | Moderate |
| Clay Loam | 8.0 | 0.49 | 78 | Poor drainage |
| Clay | 2.5 | 0.52 | 88 | Very slow, high runoff |
| Urban | 0.5 | 0.10 | 98 | Near-impervious |

---

## 2. How Terrain Slope Impacts Runoff

### Dual Effect of Slope

Slope affects flooding in **two opposing ways**:

```
    SLOPE vs. LOCAL FLOOD RISK

    Local
    Risk
    ▲
    │ ████
    │ ████████
    │ ██████████████
    │   ██████████████████
    │      ████████████████████
    │           ████████████████████
    │                  ██████████████████
    └──────────────────────────────────────►
      0°    5°    10°   15°   20°   25°
                 Terrain Slope

    FLAT terrain: HIGH local risk (water POOLS)
    STEEP terrain: LOW local risk but HIGH DOWNSTREAM risk
```

**At the cell itself:**
- Flat (0–1°): Water cannot drain → **ponding** → high local risk
- Moderate (5–15°): Water drains at moderate speed
- Steep (>15°): Water runs off quickly → low local risk

**For downstream cells:**
- Steep upslope → MORE runoff arrives faster
- The D8 flow algorithm routes water to the steepest downslope neighbour

### Mathematical Model

**Local slope risk factor:**

$$f_{slope} = 1.0 - 0.6 \times \tanh\left(\frac{\theta}{8}\right) + 0.2 \times e^{-\theta^2/2}$$

The Gaussian term adds a flat-terrain ponding bonus.

**Downstream runoff contribution:**

$$Q_{out} = C \times R \times \sqrt{\sin(\theta)}$$

Where $C$ = runoff coefficient, $R$ = rainfall, $\theta$ = slope angle.

---

## 3. Risk Scoring Logic

### Composite Risk Formula

Each grid cell's flood risk is a **weighted sum of five factors**:

$$R_{cell} = w_1 \cdot P_{sat} + w_2 \cdot F_{slope} + w_3 \cdot E_{elev} + w_4 \cdot C_{runoff} + w_5 \cdot D_{drain}$$

Amplified by a rainfall intensity multiplier:

$$R_{final} = R_{cell} \times \min\left(1 + 0.3 \times \ln\left(1 + \frac{R_{rain}}{10}\right),\ 2.0\right)$$

### Default Weights

| Factor | Weight | What It Measures |
|---|---|---|
| **Saturation** ($P_{sat}$) | 0.25 | Soil's remaining absorption capacity |
| **Slope** ($F_{slope}$) | 0.10 | Water ponding vs. drainage potential |
| **Elevation** ($E_{elev}$) | 0.20 | Position relative to regional terrain |
| **Runoff** ($C_{runoff}$) | 0.20 | Fraction of rain becoming surface flow |
| **Drainage** ($D_{drain}$) | 0.25 | Infrastructure capacity vs. demand |

### Factor Computation Details

**Elevation Risk Score:**

$$E_{risk} = 1 - \frac{h - h_{min}}{h_{max} - h_{min}}$$

Low elevation relative to the region → higher risk (water flows downhill).

**Runoff Coefficient** (SCS Curve Number method):

$$Q = \frac{(P - I_a)^2}{P - I_a + S} \quad \text{where } S = \frac{25400}{CN} - 254, \quad I_a = 0.2S$$

Higher CN (urban, clay) → more runoff. Lower CN (sandy, forested) → more infiltration.

**Drainage Deficiency:**

$$\eta = \frac{D_{capacity} \times f_{coverage} \times f_{maintenance}}{R_{rainfall}}$$

$\eta > 1$ means drainage can handle the rain. $\eta < 0.5$ → severe flooding.

### Risk Levels

| Score Range | Level | Action |
|---|---|---|
| 0.00 – 0.15 | Minimal | No action |
| 0.15 – 0.35 | Low | Monitor |
| 0.35 – 0.60 | Moderate | Prepare |
| 0.60 – 0.80 | High | Ready to evacuate |
| 0.80 – 1.00 | Critical | Evacuate immediately |

---

## 4. Grid Simulation Strategy

### Step-by-Step Pipeline

```
  ┌────────────────────┐
  │ 1. GENERATE TERRAIN │  Elevation, slope, soil type
  │    (synthetic/DEM)  │  per 1 km × 1 km cell
  └─────────┬──────────┘
            ▼
  ┌────────────────────┐
  │ 2. APPLY RAINFALL   │  Uniform / gradient / storm cell
  │    PATTERN          │  pattern across the grid
  └─────────┬──────────┘
            ▼
  ┌────────────────────┐
  │ 3. PER-CELL         │  Infiltration capacity
  │    HYDROLOGY        │  Surface excess → water depth
  └─────────┬──────────┘
            ▼
  ┌────────────────────┐
  │ 4. COMPUTE RISK     │  5-factor weighted composite
  │    FACTORS          │  per cell
  └─────────┬──────────┘
            ▼
  ┌────────────────────┐
  │ 5. PROPAGATE RUNOFF │  D8 flow direction algorithm
  │    DOWNSLOPE (×3)   │  Water → lowest neighbour
  └─────────┬──────────┘
            ▼
  ┌────────────────────┐
  │ 6. RECALCULATE RISK │  Include accumulated runoff
  │    WITH INFLOWS     │  from upslope cells
  └─────────┬──────────┘
            ▼
  ┌────────────────────┐
  │ 7. OUTPUT           │  Risk heat-map (NxN matrix)
  │    RESULTS          │  Hotspot list (risk ≥ 0.6)
  └────────────────────┘  Statistics & cell details
```

### Terrain Styles

| Style | Pattern | Best For |
|---|---|---|
| **valley** | Bowl shape: high edges, low centre | River basin flooding |
| **coastal** | West-to-east elevation decrease | Coastal storm surge |
| **flat** | Uniform with micro-variations | Urban drainage problems |
| **urban** | Flat + high central urbanization | City flood planning |

### Rainfall Patterns

| Pattern | Distribution | Simulates |
|---|---|---|
| **uniform** | Same intensity everywhere (±5%) | Widespread monsoon |
| **gradient** | NW heavy → SE light | Frontal system passage |
| **storm_cell** | Gaussian peak at centre | Convective thunderstorm |

### D8 Runoff Propagation

The D8 (deterministic eight-neighbour) algorithm routes water from each cell
to its **steepest downslope neighbour**:

```
    ┌───┬───┬───┐
    │NW │ N │NE │     For each cell, compute:
    ├───┼───┼───┤       Δh = h_center - h_neighbor
    │ W │ X │ E │       gradient = Δh / distance
    ├───┼───┼───┤     
    │SW │ S │SE │     Route water to steepest
    └───┴───┴───┘     downslope neighbour
    
    Transfer fraction = 0.6 × sin(slope)^0.5
    Capped at 80% to retain some ponding
```

Propagation runs for 3 iterations, allowing water to cascade
through multiple cells downstream.

---

## 5. Extending to Real Satellite DEM Data

### Data Sources for Production

| Data Product | Resolution | Coverage | Access |
|---|---|---|---|
| **SRTM** | 30 m | Global 60°N–56°S | USGS EarthExplorer (free) |
| **Copernicus DEM** | 30 m | Global | Copernicus Open Access Hub |
| **ALOS PALSAR** | 12.5 m | Global | Alaska Satellite Facility |
| **SMAP** (soil moisture) | 9 km → downscaled | Global | NASA Earthdata |
| **Sentinel-1** (soil moisture) | 1 km | Global | ESA Copernicus |

### Integration Steps

```python
# STEP 1: Replace generate_terrain() with DEM loader
def load_dem_grid(
    geotiff_path: str,
    center_lat: float,
    center_lon: float,
    grid_size_km: int = 10,
) -> List[List[GridCell]]:
    """
    Load real elevation data from a GeoTIFF DEM.
    
    Libraries: rasterio, pyproj
    
    Pipeline:
        1. Open GeoTIFF with rasterio
        2. Compute bounding box from centre + grid_size
        3. Window-read the elevation band
        4. Resample to 1 km resolution (aggregate 30m pixels)
        5. Compute slope from kernel convolution
        6. Build GridCell objects with real elevation + slope
    """
    import rasterio
    from rasterio.windows import from_bounds
    
    with rasterio.open(geotiff_path) as src:
        # Compute bounding box
        half_deg = (grid_size_km / 2) / 111.0
        window = from_bounds(
            center_lon - half_deg,
            center_lat - half_deg,
            center_lon + half_deg,
            center_lat + half_deg,
            src.transform,
        )
        elevation = src.read(1, window=window)
    
    # Resample 30m → 1km by block-averaging
    block = int(1000 / 30)  # ~33 pixels per km
    rows_km = elevation.shape[0] // block
    cols_km = elevation.shape[1] // block
    elev_1km = elevation[:rows_km*block, :cols_km*block].reshape(
        rows_km, block, cols_km, block
    ).mean(axis=(1, 3))
    
    # Build grid cells from real data...
    # (same GridCell structure, real elevations)
```

```python
# STEP 2: Replace soil moisture with satellite data
def load_soil_moisture_smap(
    hdf5_path: str,
    grid: List[List[GridCell]],
) -> None:
    """
    Overlay NASA SMAP soil moisture on the grid.
    
    SMAP provides volumetric water content at 9 km resolution.
    Downscale to 1 km using bilinear interpolation + terrain
    correction (valleys wetter, ridges drier).
    """
    import h5py
    from scipy.interpolate import RegularGridInterpolator
    
    with h5py.File(hdf5_path) as f:
        sm = f['Soil_Moisture_Retrieval_Data/soil_moisture'][:]
        lat = f['Soil_Moisture_Retrieval_Data/latitude'][:]
        lon = f['Soil_Moisture_Retrieval_Data/longitude'][:]
    
    interp = RegularGridInterpolator((lat[:, 0], lon[0, :]), sm)
    
    for row in grid:
        for cell in row:
            cell.soil_moisture = float(
                interp((cell.latitude, cell.longitude))
            )
```

```python
# STEP 3: Real-time rainfall from weather radar / API
def overlay_live_rainfall(
    grid: List[List[GridCell]],
) -> None:
    """
    Fetch per-cell rainfall from Open-Meteo for each
    grid cell's lat/lon. Rate-limited to avoid API abuse.
    """
    from backend.app.ingestion.weather_service import fetch_weather
    
    for row in grid:
        for cell in row:
            result = fetch_weather(cell.latitude, cell.longitude)
            if result.current:
                cell.rainfall_mm_hr = result.current.precipitation
```

### Recommended Production Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  SRTM DEM   │────▶│  DEM Loader  │────▶│              │
│  GeoTIFF    │     │  (rasterio)  │     │              │
└─────────────┘     └──────────────┘     │              │
                                          │   Grid       │
┌─────────────┐     ┌──────────────┐     │   Simulation │
│ SMAP / S-1  │────▶│ Soil Moisture│────▶│   Engine     │──▶ Risk Map
│ HDF5 / TIFF │     │  Downscaler  │     │              │
└─────────────┘     └──────────────┘     │              │
                                          │              │
┌─────────────┐     ┌──────────────┐     │              │
│ Open-Meteo  │────▶│  Rainfall    │────▶│              │
│ Weather API │     │  Interpolator│     │              │
└─────────────┘     └──────────────┘     └──────────────┘
                                                │
                                                ▼
                                         ┌──────────────┐
                                         │  Hotspot      │
                                         │  Alerting     │
                                         └──────────────┘
```

### Key Libraries for Extension

| Library | Purpose |
|---|---|
| `rasterio` | Read/write GeoTIFF elevation data |
| `pyproj` | Coordinate system transformations |
| `scipy.ndimage` | Slope computation via Sobel filters |
| `h5py` | Read HDF5 satellite data (SMAP) |
| `xarray` | Multi-dimensional geospatial arrays |
| `geopandas` | Vector flood zone boundaries |

---

## 6. API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/v1/grid/simulate` | Full grid simulation with custom params |
| `GET` | `/api/v1/grid/simulate-default` | Quick simulation (query params) |
| `POST` | `/api/v1/grid/cell-risk` | Single-cell risk computation |
| `GET` | `/api/v1/grid/soil-types` | List soil types + properties |

### Example Request

```json
POST /api/v1/grid/simulate
{
    "latitude": 13.08,
    "longitude": 80.27,
    "grid_size": 10,
    "rainfall_mm_hr": 50,
    "terrain_style": "valley",
    "rainfall_pattern": "storm_cell",
    "seed": 42
}
```

### Example Response (abridged)

```json
{
    "grid_rows": 10,
    "grid_cols": 10,
    "total_cells": 100,
    "stats": {
        "mean_risk": 0.4521,
        "max_risk": 0.9876,
        "cells_high_risk": 12,
        "cells_critical": 4,
        "pct_high_risk": 12.0
    },
    "hotspots": [
        {
            "row": 4, "col": 4,
            "lat": 13.08, "lon": 80.27,
            "risk_score": 0.9876,
            "water_depth_mm": 42.3,
            "risk_level": "critical"
        }
    ],
    "cells": [ ... ]
}
```

---

## 7. Module File Map

| File | Role |
|---|---|
| `backend/app/ml/hydrology.py` | Physics equations: saturation, infiltration, slope, runoff, drainage |
| `backend/app/ml/grid_model.py` | Grid generator, simulation engine, D8 propagation |
| `backend/app/api/v1/grid.py` | FastAPI REST endpoints |
| `docs/HYPER_LOCAL_FLOOD_MODELING.md` | This document |
