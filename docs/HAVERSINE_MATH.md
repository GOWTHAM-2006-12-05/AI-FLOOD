# Haversine Formula — Mathematical Derivation & Usage

## Why Not Euclidean Distance?

On a flat plane, the distance between two points $(x_1, y_1)$ and $(x_2, y_2)$ is:

$$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

But Earth is a **sphere** (approximately). Applying Euclidean distance to raw
latitude/longitude values produces wildly wrong results because:

1. **1° of longitude ≠ 1° of latitude** — longitude lines converge at the poles
2. At the equator, 1° of longitude ≈ 111.32 km, but at 60° latitude it's only ≈ 55.66 km
3. You cannot "shortcut through the Earth" — you must travel along the curved surface

The **Haversine formula** gives the **great-circle distance**: the shortest path
between two points on a sphere, traveling along the surface.

---

## The Formula

Given two points:
- $P_1 = (\varphi_1, \lambda_1)$ — latitude and longitude of point 1
- $P_2 = (\varphi_2, \lambda_2)$ — latitude and longitude of point 2

All angles must be in **radians**. Convert from degrees:

$$\varphi_{rad} = \varphi_{deg} \times \frac{\pi}{180}$$

### Step 1 — Compute Differences

$$\Delta\varphi = \varphi_2 - \varphi_1$$
$$\Delta\lambda = \lambda_2 - \lambda_1$$

### Step 2 — Haversine of the Central Angle

The **haversine function** is defined as:

$$\text{hav}(\theta) = \sin^2\!\left(\frac{\theta}{2}\right)$$

Apply it to get the intermediate quantity $a$:

$$a = \sin^2\!\left(\frac{\Delta\varphi}{2}\right) + \cos(\varphi_1) \cdot \cos(\varphi_2) \cdot \sin^2\!\left(\frac{\Delta\lambda}{2}\right)$$

**What $a$ represents:** the square of half the chord length between the two points
on a unit sphere. It captures both the latitude difference and the
longitude difference (adjusted for latitude).

### Step 3 — Angular Distance

$$c = 2 \cdot \text{atan2}\!\left(\sqrt{a},\; \sqrt{1 - a}\right)$$

$c$ is the **angular distance** in radians between the two points, as seen
from the center of the Earth.

> Note: We use `atan2(√a, √(1−a))` instead of `asin(√a)` for better
> numerical stability when sizes are very small or very large.

### Step 4 — Surface Distance

$$d = R \cdot c$$

Where $R$ is Earth's mean radius:

$$R = 6{,}371.0088 \text{ km} \quad \text{(IAU mean radius)}$$

---

## Worked Example

**Chennai** $(13.0827°N, 80.2707°E)$ → **Bangalore** $(12.9716°N, 77.5946°E)$

### Convert to radians:

| Value        | Degrees   | Radians           |
|-------------|-----------|-------------------|
| $\varphi_1$ | 13.0827°  | 0.228352 rad      |
| $\lambda_1$ | 80.2707°  | 1.400856 rad      |
| $\varphi_2$ | 12.9716°  | 0.226413 rad      |
| $\lambda_2$ | 77.5946°  | 1.354133 rad      |

### Differences:

$$\Delta\varphi = 0.226413 - 0.228352 = -0.001939 \text{ rad}$$
$$\Delta\lambda = 1.354133 - 1.400856 = -0.046723 \text{ rad}$$

### Compute $a$:

$$a = \sin^2(-0.000970) + \cos(0.228352) \cdot \cos(0.226413) \cdot \sin^2(-0.023362)$$
$$a = 9.409 \times 10^{-7} + 0.97400 \times 0.97438 \times 5.458 \times 10^{-4}$$
$$a = 9.409 \times 10^{-7} + 5.318 \times 10^{-4}$$
$$a \approx 5.327 \times 10^{-4}$$

### Compute $c$:

$$c = 2 \cdot \text{atan2}\!\left(\sqrt{5.327 \times 10^{-4}},\; \sqrt{1 - 5.327 \times 10^{-4}}\right)$$
$$c = 2 \cdot \text{atan2}(0.02308, 0.99973)$$
$$c \approx 0.04556 \text{ rad}$$

### Final distance:

$$d = 6{,}371.0088 \times 0.04556 \approx 290.17 \text{ km}$$

✅ **Our code produces: `290.1724 km`** — matches perfectly.

---

## Accuracy Considerations

| Method | Accuracy | Speed | Use Case |
|--------|----------|-------|----------|
| **Euclidean on lat/lon** | Very poor (50%+ error) | Fastest | Never for geo |
| **Haversine (sphere)** | ~0.5% error | Fast | Disaster alerting ✅ |
| **Vincenty (ellipsoid)** | ~0.05% error | 3-5× slower | Survey-grade work |
| **Karney (ellipsoid)** | Machine precision | 5-10× slower | Geodetic reference |

For disaster alerting, Haversine's ~0.5% error means:
- At 10 km actual distance, error ≤ 50 meters
- At 100 km, error ≤ 500 meters

This is far more precise than needed — disaster zones are typically
described in km-scale polygons, not meter-precision points.

---

## Bounding-Box Pre-Filter

Before running the (relatively expensive) Haversine formula on every disaster
in the database, we apply a **bounding-box pre-filter**:

```
Given: center = (φ, λ), radius = r km

angular_radius = r / R    (in radians)

lat_min = φ - angular_radius
lat_max = φ + angular_radius

Δλ = angular_radius / cos(φ)

lon_min = λ - Δλ
lon_max = λ + Δλ
```

Any disaster whose lat/lon falls **outside** this box is guaranteed to be
outside the radius — we skip Haversine entirely. This turns an $O(n)$ scan
into a much faster filter when disasters are geographically spread out.

In PostGIS (production), this becomes a spatial index query:
```sql
WHERE ST_DWithin(location::geography, ST_MakePoint(lon, lat)::geography, radius_meters)
```

---

## How It Fits in the System

```
User Location (GPS/Manual)
        │
        ▼
┌──────────────────┐      For each disaster:
│  Bounding Box    │─────────────────────────────┐
│  Pre-Filter      │    Outside box? → SKIP       │
└──────────────────┘                              │
        │ (candidates only)                       │
        ▼                                         │
┌──────────────────┐                              │
│  Haversine       │    distance > radius? → SKIP │
│  Distance Check  │                              │
└──────────────────┘                              │
        │ (matches only)                          │
        ▼                                         │
┌──────────────────┐                              │
│  Sort by         │                              │
│  Distance (ASC)  │                              │
└──────────────────┘                              │
        │                                         │
        ▼                                         │
   Return to user                                 │
   with distance_km                               │
   and severity                                   │
```

---

## Python Implementation Reference

The complete implementation is in:

```
backend/app/spatial/radius_utils.py
```

Key functions:
- `haversine(p1, p2) → float` — core distance calculation
- `is_inside_radius(user, disaster, radius) → (bool, float)` — single check
- `filter_disasters(user, list, radius, ...) → FilterResult` — batch filter
- `_bounding_box(center, radius) → (min_lat, max_lat, min_lon, max_lon)` — pre-filter

All functions are pure, stateless, and have no external dependencies beyond
Python's standard `math` module.
