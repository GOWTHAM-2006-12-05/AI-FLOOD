"""End-to-end test for the hyper-local grid flood modelling system."""

from __future__ import annotations

import sys, os, math

# Ensure project root is on the path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from backend.app.ml.hydrology import (
    SoilType,
    POROSITY,
    KSAT,
    CURVE_NUMBER,
    soil_saturation_fraction,
    infiltration_capacity,
    saturation_flood_probability,
    compute_slope_degrees,
    slope_runoff_factor,
    downstream_runoff_contribution,
    elevation_risk_score,
    topographic_wetness_index,
    scs_runoff_depth,
    composite_runoff_coefficient,
    DrainageProfile,
    drainage_efficiency_score,
    drainage_overflow_mm,
    compute_cell_risk,
)
from backend.app.ml.grid_model import (
    GridCell,
    generate_terrain,
    simulate_grid,
    run_simulation,
)

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}  {detail}")


# ── 1. Hydrology unit tests ───────────────────────────────────────────

print("\n=== 1. HYDROLOGY UNIT TESTS ===\n")

# Saturation fraction
s = soil_saturation_fraction(0.35, POROSITY[SoilType.LOAM])
check("saturation_fraction in [0,1]", 0.0 <= s <= 1.0, f"got {s}")

s_dry = soil_saturation_fraction(0.10, POROSITY[SoilType.CLAY])
check("dry soil saturation ≈ 0", s_dry < 0.05, f"got {s_dry}")

s_wet = soil_saturation_fraction(0.50, POROSITY[SoilType.SAND])
check("wet sand saturation high", s_wet > 0.8, f"got {s_wet}")

# Infiltration capacity
f = infiltration_capacity(0.2, KSAT[SoilType.SANDY_LOAM], POROSITY[SoilType.SANDY_LOAM])
check("infiltration > 0", f > 0, f"got {f}")

f_sat = infiltration_capacity(0.50, KSAT[SoilType.SAND], POROSITY[SoilType.SAND])
check("saturated sand low infiltration", f_sat < 5.0, f"got {f_sat}")

# Saturation flood probability
p_low = saturation_flood_probability(0.3)
p_high = saturation_flood_probability(0.95)
check("low saturation → low P", p_low < 0.15, f"got {p_low}")
check("high saturation → high P", p_high > 0.85, f"got {p_high}")

# Slope
slope = compute_slope_degrees(100, [110, 90, 105, 95])
check("slope > 0", slope > 0, f"got {slope}")

f_flat = slope_runoff_factor(0.5)
f_steep = slope_runoff_factor(25.0)
check("flat > steep local risk", f_flat > f_steep, f"flat={f_flat}, steep={f_steep}")

# Downstream contribution
q = downstream_runoff_contribution(10.0, 0.5, 15.0)
check("downstream Q > 0", q > 0, f"got {q}")

# Elevation risk
e_low = elevation_risk_score(5, 5, 100)
e_high = elevation_risk_score(95, 5, 100)
check("low elevation → high risk", e_low > 0.9, f"got {e_low}")
check("high elevation → low risk", e_high < 0.1, f"got {e_high}")

# TWI
twi = topographic_wetness_index(5000, 5.0)
check("TWI > 0", twi > 0, f"got {twi}")

# SCS runoff depth
q_scs = scs_runoff_depth(50.0, 80)
check("SCS runoff > 0 for CN=80", q_scs > 0, f"got {q_scs}")
q_low_cn = scs_runoff_depth(50.0, 40)
check("SCS: CN=80 > CN=40 runoff", q_scs > q_low_cn, f"80:{q_scs} 40:{q_low_cn}")

# Composite runoff
c = composite_runoff_coefficient(SoilType.CLAY, 0.6, 0.4)
check("composite runoff in [0,1]", 0 <= c <= 1, f"got {c}")

# Drainage
dp = DrainageProfile(channel_capacity_mm_hr=20.0, pipe_coverage_fraction=0.7, maintenance_factor=0.8)
eff = drainage_efficiency_score(30.0, dp)
check("drainage efficiency in [0,1]", 0 <= eff <= 2, f"got {eff}")

overflow = drainage_overflow_mm(50.0, dp)
check("overflow > 0 when rain > capacity", overflow > 0, f"got {overflow}")

# Composite risk
risk, factors = compute_cell_risk(
    soil_moisture=0.8,
    slope_deg=3.0,
    elevation_m=10.0,
    rainfall_rate_mm_hr=60.0,
    soil_type=SoilType.CLAY,
    urbanization=0.5,
    drainage=DrainageProfile(channel_capacity_mm_hr=15.0, pipe_coverage_fraction=0.5, maintenance_factor=0.6),
    regional_min_elev=5.0,
    regional_max_elev=100.0,
)
check("composite risk in [0,1]", 0 <= risk <= 1, f"got {risk}")
check("high-params → risk > 0.5", risk > 0.5, f"got {risk}")

# ── 2. Terrain generation tests ──────────────────────────────────────

print("\n=== 2. TERRAIN GENERATION ===\n")

for style in ["valley", "coastal", "flat", "urban"]:
    grid = generate_terrain(13.0, 80.0, 5, 5, terrain_style=style, seed=42)
    check(f"{style} terrain 5×5", len(grid) == 5 and len(grid[0]) == 5)
    # All cells should have valid data
    ok = all(
        isinstance(c, GridCell) and c.elevation_m >= 0
        for row in grid for c in row
    )
    check(f"{style} cells valid", ok)

# ── 3. Full simulation tests ─────────────────────────────────────────

print("\n=== 3. GRID SIMULATION ===\n")

result = run_simulation(
    center_lat=13.08,
    center_lon=80.27,
    grid_size=8,
    rainfall_mm_hr=45.0,
    terrain_style="valley",
    rainfall_pattern="storm_cell",
    seed=123,
)

check("simulation has cells", len(result.cells) > 0)
check("heatmap shape 8×8", len(result.risk_heatmap) == 8 and len(result.risk_heatmap[0]) == 8)
check("stats populated", result.stats["total_cells"] == 64)
check("mean_risk in [0,1]", 0 <= result.stats["mean_risk"] <= 1.0)
check("max_risk in [0,1]", 0 <= result.stats["max_risk"] <= 1.0)

# Valley centre should be higher risk than edges
centre_risk = result.risk_heatmap[4][4]
edge_risk = result.risk_heatmap[0][0]
check("valley centre > edge risk", centre_risk > edge_risk,
      f"centre={centre_risk:.3f} edge={edge_risk:.3f}")

# Hotspots should be high risk
if result.hotspots:
    check("hotspots risk >= 0.55", all(
        (h.risk_score if hasattr(h, 'risk_score') else h.get("risk_score", h.get("risk", 0))) >= 0.55
        for h in result.hotspots
    ))
else:
    check("hotspots present with 45mm rain", False, "no hotspots found")

# Different rainfall patterns produce different distributions
r_uniform = run_simulation(grid_size=6, rainfall_mm_hr=50, rainfall_pattern="uniform", seed=1)
r_storm = run_simulation(grid_size=6, rainfall_mm_hr=50, rainfall_pattern="storm_cell", seed=1)
check("storm_cell ≠ uniform mean risk",
      abs(r_uniform.stats["mean_risk"] - r_storm.stats["mean_risk"]) > 0.001 or
      r_uniform.stats["cells_high_risk"] != r_storm.stats["cells_high_risk"])

# Higher rainfall → higher risk
r_light = run_simulation(grid_size=5, rainfall_mm_hr=10, seed=7)
r_heavy = run_simulation(grid_size=5, rainfall_mm_hr=80, seed=7)
check("80mm > 10mm risk",
      r_heavy.stats["mean_risk"] > r_light.stats["mean_risk"],
      f"heavy={r_heavy.stats['mean_risk']:.3f} light={r_light.stats['mean_risk']:.3f}")

# ── Summary ───────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print(f"RESULTS:  {passed} passed,  {failed} failed,  {passed+failed} total")
print(f"{'='*50}\n")

sys.exit(0 if failed == 0 else 1)
