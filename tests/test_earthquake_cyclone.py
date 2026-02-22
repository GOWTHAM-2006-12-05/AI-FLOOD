"""
tests/test_earthquake_cyclone.py — Comprehensive tests for earthquake and cyclone modules.

Runs as a standalone script (no pytest dependency required).
Tests:
    1. Earthquake depth classification
    2. Earthquake depth severity factor
    3. Earthquake impact radius estimation
    4. Earthquake risk classification
    5. USGS GeoJSON parsing
    6. Earthquake radius filtering
    7. Earthquake depth analysis
    8. Cyclone classification (IMD scale)
    9. Heavy rainfall detection
    10. Cyclone escalation logic
    11. Cyclone risk score computation
    12. Cyclone condition detection
    13. Cyclone radius filtering
    14. Cyclone assessment pipeline
    15. Edge cases and boundary conditions
"""

from __future__ import annotations

import sys
import math
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Bootstrap path
# ---------------------------------------------------------------------------
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Imports — Earthquake
# ---------------------------------------------------------------------------
from backend.app.ml.earthquake_service import (
    DepthClass,
    EarthquakeRisk,
    EarthquakeEvent,
    ImpactEstimate,
    classify_depth,
    depth_severity_factor,
    estimate_impact_radius,
    _classify_earthquake_risk,
    _parse_usgs_feature,
    filter_earthquakes_by_radius,
    analyze_depth_distribution,
)

# ---------------------------------------------------------------------------
# Imports — Cyclone
# ---------------------------------------------------------------------------
from backend.app.ml.cyclone_service import (
    CycloneCategory,
    CycloneRisk,
    EscalationLevel,
    CycloneEvent as CycloneEventClass,
    classify_cyclone,
    classify_risk as classify_cyclone_risk,
    is_heavy_rainfall,
    determine_escalation,
    compute_cyclone_risk,
    detect_cyclone_conditions,
    filter_cyclones_by_radius,
    assess_cyclone_risk,
    get_cyclone_alert,
)

# ---------------------------------------------------------------------------
# Imports — Spatial
# ---------------------------------------------------------------------------
from backend.app.spatial.radius_utils import Coordinate, haversine

# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------
_pass = 0
_fail = 0
_section = ""


def section(name: str):
    global _section
    _section = name
    print(f"\n=== {name} ===\n")


def check(label: str, condition: bool, detail: str = ""):
    global _pass, _fail
    if condition:
        _pass += 1
        print(f"  [PASS] {label}")
    else:
        _fail += 1
        msg = f"  [FAIL] {label}"
        if detail:
            msg += f"  — {detail}"
        print(msg)


# ═══════════════════════════════════════════════════════════════════════════
# 1. EARTHQUAKE DEPTH CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

section("1. EARTHQUAKE DEPTH CLASSIFICATION")

check("shallow at 10 km", classify_depth(10) == DepthClass.SHALLOW)
check("shallow at 70 km (boundary)", classify_depth(70) == DepthClass.SHALLOW)
check("intermediate at 71 km", classify_depth(71) == DepthClass.INTERMEDIATE)
check("intermediate at 300 km", classify_depth(300) == DepthClass.INTERMEDIATE)
check("deep at 301 km", classify_depth(301) == DepthClass.DEEP)
check("deep at 600 km", classify_depth(600) == DepthClass.DEEP)
check("negative depth → shallow", classify_depth(-5) == DepthClass.SHALLOW)
check("zero depth → shallow", classify_depth(0) == DepthClass.SHALLOW)


# ═══════════════════════════════════════════════════════════════════════════
# 2. DEPTH SEVERITY FACTOR
# ═══════════════════════════════════════════════════════════════════════════

section("2. DEPTH SEVERITY FACTOR")

check("very shallow (5km) = 1.5", depth_severity_factor(5) == 1.5)
check("shallow (30km) = 1.0", depth_severity_factor(30) == 1.0)
check("intermediate (200km) = 0.6", depth_severity_factor(200) == 0.6)
check("deep (500km) = 0.2", depth_severity_factor(500) == 0.2)
check("negative → 1.5 (clamped to 0)", depth_severity_factor(-10) == 1.5)


# ═══════════════════════════════════════════════════════════════════════════
# 3. IMPACT RADIUS ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════

section("3. IMPACT RADIUS ESTIMATION")

imp = estimate_impact_radius(6.0, 10.0)
check("impact is ImpactEstimate", isinstance(imp, ImpactEstimate))
check("felt > damage > severe",
      imp.felt_radius_km > imp.damage_radius_km > imp.severe_radius_km)
check("felt > 0", imp.felt_radius_km > 0)
check("damage > 0", imp.damage_radius_km > 0)
check("severe > 0", imp.severe_radius_km > 0)
check("epicentral intensity = 8.0", imp.epicentral_intensity == 8.0,
      f"got {imp.epicentral_intensity}")

# Depth correction: shallow < deep → larger impact
shallow_imp = estimate_impact_radius(6.0, 5.0)
deep_imp = estimate_impact_radius(6.0, 300.0)
check("shallow felt > deep felt",
      shallow_imp.felt_radius_km > deep_imp.felt_radius_km,
      f"shallow={shallow_imp.felt_radius_km}, deep={deep_imp.felt_radius_km}")

# Magnitude scaling
m5 = estimate_impact_radius(5.0, 10.0)
m7 = estimate_impact_radius(7.0, 10.0)
check("M7 felt > M5 felt", m7.felt_radius_km > m5.felt_radius_km)
check("M7 damage > M5 damage", m7.damage_radius_km > m5.damage_radius_km)

# Small magnitude: severe should be tiny
m3 = estimate_impact_radius(3.0, 10.0)
check("M3 severe ≈ small", m3.severe_radius_km < 5.0,
      f"got {m3.severe_radius_km}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. EARTHQUAKE RISK CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

section("4. EARTHQUAKE RISK CLASSIFICATION")

check("M2 shallow → negligible/low",
      _classify_earthquake_risk(2.0, 10) in (EarthquakeRisk.NEGLIGIBLE, EarthquakeRisk.LOW))
check("M5 shallow → moderate/high",
      _classify_earthquake_risk(5.0, 10) in (EarthquakeRisk.MODERATE, EarthquakeRisk.HIGH))
check("M7 shallow → high/critical",
      _classify_earthquake_risk(7.0, 5) in (EarthquakeRisk.HIGH, EarthquakeRisk.CRITICAL))
check("M8 shallow → critical",
      _classify_earthquake_risk(8.0, 5) == EarthquakeRisk.CRITICAL)
check("M6 deep → lower risk (attenuated)",
      _classify_earthquake_risk(6.0, 500) in (EarthquakeRisk.NEGLIGIBLE, EarthquakeRisk.LOW))


# ═══════════════════════════════════════════════════════════════════════════
# 5. USGS GEOJSON PARSING
# ═══════════════════════════════════════════════════════════════════════════

section("5. USGS GEOJSON PARSING")

sample_feature = {
    "type": "Feature",
    "properties": {
        "mag": 5.2,
        "place": "42 km NW of Chennai, India",
        "time": 1708617600000,
        "url": "https://earthquake.usgs.gov/earthquakes/eventpage/us7000test",
        "title": "M 5.2 - 42 km NW of Chennai",
        "felt": 150,
        "tsunami": 0,
        "alert": "green",
        "status": "reviewed",
        "magType": "mw",
    },
    "geometry": {
        "type": "Point",
        "coordinates": [80.15, 13.25, 12.5],
    },
    "id": "us7000test",
}

parsed = _parse_usgs_feature(sample_feature)
check("parsed is EarthquakeEvent", parsed is not None)
check("event_id correct", parsed.event_id == "us7000test")
check("magnitude = 5.2", parsed.magnitude == 5.2)
check("depth = 12.5", parsed.depth_km == 12.5)
check("lat = 13.25", parsed.latitude == 13.25)
check("lon = 80.15", parsed.longitude == 80.15)
check("felt_reports = 150", parsed.felt_reports == 150)
check("tsunami = False", parsed.tsunami_flag == False)
check("alert = green", parsed.alert_level == "green")
check("depth class = shallow", parsed.depth_class == DepthClass.SHALLOW)
check("magType = mw", parsed.magnitude_type == "mw")

# Malformed feature
bad_feature = {"type": "Feature", "properties": {}}
bad_parsed = _parse_usgs_feature(bad_feature)
check("malformed feature → None", bad_parsed is None)


# ═══════════════════════════════════════════════════════════════════════════
# 6. EARTHQUAKE RADIUS FILTERING
# ═══════════════════════════════════════════════════════════════════════════

section("6. EARTHQUAKE RADIUS FILTERING")

# Create test events
test_events = [
    EarthquakeEvent(
        event_id="eq1", title="Nearby M5", magnitude=5.0, depth_km=10,
        latitude=13.10, longitude=80.30, timestamp=datetime.now(timezone.utc),
        place="Near Chennai", url="", felt_reports=50, tsunami_flag=False,
        alert_level=None, status="reviewed", magnitude_type="mw",
    ),
    EarthquakeEvent(
        event_id="eq2", title="Far M6", magnitude=6.0, depth_km=15,
        latitude=28.61, longitude=77.20, timestamp=datetime.now(timezone.utc),
        place="Delhi", url="", felt_reports=200, tsunami_flag=False,
        alert_level="yellow", status="reviewed", magnitude_type="mw",
    ),
    EarthquakeEvent(
        event_id="eq3", title="Small M2", magnitude=2.0, depth_km=5,
        latitude=13.05, longitude=80.25, timestamp=datetime.now(timezone.utc),
        place="Very near", url="", felt_reports=5, tsunami_flag=False,
        alert_level=None, status="automatic", magnitude_type="ml",
    ),
]

# Filter near Chennai with radius 50km, min magnitude 3.0
filtered = filter_earthquakes_by_radius(
    test_events, user_lat=13.08, user_lon=80.27,
    radius_km=50.0, min_magnitude=3.0,
)
check("filter matched 1 event (nearby M5)", filtered.count == 1,
      f"got {filtered.count}")
check("matched event is eq1", filtered.matched[0].event_id == "eq1")
check("distance populated", filtered.matched[0].distance_km is not None)
check("distance < 50 km", filtered.matched[0].distance_km < 50)

# Filter with larger radius, lower magnitude
filtered_wide = filter_earthquakes_by_radius(
    test_events, user_lat=13.08, user_lon=80.27,
    radius_km=2000.0, min_magnitude=1.0,
)
check("wide filter includes all 3", filtered_wide.count == 3,
      f"got {filtered_wide.count}")

# Sort by magnitude
filtered_mag = filter_earthquakes_by_radius(
    test_events, user_lat=13.08, user_lon=80.27,
    radius_km=2000.0, min_magnitude=1.0, sort_by="magnitude",
)
check("magnitude sort: largest first",
      filtered_mag.matched[0].magnitude >= filtered_mag.matched[1].magnitude)

# Depth class filter
filtered_deep = filter_earthquakes_by_radius(
    test_events, user_lat=13.08, user_lon=80.27,
    radius_km=2000.0, min_magnitude=1.0,
    depth_class_filter=DepthClass.INTERMEDIATE,
)
check("depth class filter: no intermediate events", filtered_deep.count == 0)


# ═══════════════════════════════════════════════════════════════════════════
# 7. EARTHQUAKE DEPTH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

section("7. EARTHQUAKE DEPTH ANALYSIS")

analysis = analyze_depth_distribution(test_events)
check("total events = 3", analysis.total_events == 3)
check("all shallow (3)", analysis.shallow_count == 3)
check("no intermediate", analysis.intermediate_count == 0)
check("no deep", analysis.deep_count == 0)
check("mean depth > 0", analysis.mean_depth_km > 0)
check("min depth = 5", analysis.min_depth_km == 5)
check("max depth = 15", analysis.max_depth_km == 15)
check("shallow fraction = 1.0", analysis.shallow_fraction == 1.0)
check("correlation exists", analysis.depth_magnitude_correlation is not None)

# Empty analysis
empty_analysis = analyze_depth_distribution([])
check("empty: total = 0", empty_analysis.total_events == 0)
check("empty: mean = 0", empty_analysis.mean_depth_km == 0)

# to_dict coverage
d = analysis.to_dict()
check("to_dict has distribution", "distribution" in d)
check("to_dict has shallow_fraction", "shallow_fraction" in d)


# ═══════════════════════════════════════════════════════════════════════════
# 8. CYCLONE CLASSIFICATION (IMD SCALE)
# ═══════════════════════════════════════════════════════════════════════════

section("8. CYCLONE CLASSIFICATION (IMD SCALE)")

check("20 km/h → low pressure", classify_cyclone(20) == CycloneCategory.LOW_PRESSURE)
check("40 km/h → depression", classify_cyclone(40) == CycloneCategory.DEPRESSION)
check("55 km/h → deep depression", classify_cyclone(55) == CycloneCategory.DEEP_DEPRESSION)
check("75 km/h → cyclonic storm", classify_cyclone(75) == CycloneCategory.CYCLONIC_STORM)
check("100 km/h → severe CS", classify_cyclone(100) == CycloneCategory.SEVERE_CS)
check("140 km/h → very severe CS", classify_cyclone(140) == CycloneCategory.VERY_SEVERE_CS)
check("200 km/h → extremely severe", classify_cyclone(200) == CycloneCategory.EXTREMELY_SEVERE)
check("250 km/h → super cyclone", classify_cyclone(250) == CycloneCategory.SUPER_CYCLONE)

# Boundary tests
check("31 km/h → depression (boundary)", classify_cyclone(31) == CycloneCategory.DEPRESSION)
check("62 km/h → cyclonic storm (boundary)", classify_cyclone(62) == CycloneCategory.CYCLONIC_STORM)
check("222 km/h → super cyclone (boundary)", classify_cyclone(222) == CycloneCategory.SUPER_CYCLONE)


# ═══════════════════════════════════════════════════════════════════════════
# 9. HEAVY RAINFALL DETECTION
# ═══════════════════════════════════════════════════════════════════════════

section("9. HEAVY RAINFALL DETECTION")

check("60 mm heavy (default 50)", is_heavy_rainfall(60.0))
check("40 mm not heavy (default 50)", not is_heavy_rainfall(40.0))
check("50 mm exactly = heavy", is_heavy_rainfall(50.0))
check("0 mm not heavy", not is_heavy_rainfall(0.0))
check("custom threshold: 30mm, rain=35", is_heavy_rainfall(35.0, 30.0))
check("custom threshold: 30mm, rain=25", not is_heavy_rainfall(25.0, 30.0))


# ═══════════════════════════════════════════════════════════════════════════
# 10. CYCLONE ESCALATION LOGIC
# ═══════════════════════════════════════════════════════════════════════════

section("10. CYCLONE ESCALATION LOGIC")

# Below wind threshold → none
check("below threshold → NONE",
      determine_escalation(50, 200) == EscalationLevel.NONE)

# Beyond max watch radius → none
check("too far → NONE",
      determine_escalation(80, 600) == EscalationLevel.NONE)

# Distance-based escalation
check("400km → WATCH",
      determine_escalation(80, 400) == EscalationLevel.WATCH)
check("250km → WARNING",
      determine_escalation(80, 250) == EscalationLevel.WARNING)
check("100km → ALERT",
      determine_escalation(80, 100) == EscalationLevel.ALERT)
check("30km → CRITICAL",
      determine_escalation(80, 30) == EscalationLevel.CRITICAL)

# Upgrade via heavy rainfall
esc_base = determine_escalation(80, 400, rainfall_mm=50)
esc_rain = determine_escalation(80, 400, rainfall_mm=150)
check("heavy rain upgrades WATCH → WARNING", esc_rain == EscalationLevel.WARNING,
      f"got {esc_rain.value}")

# Upgrade via pressure deficit
esc_press = determine_escalation(80, 400, pressure_hpa=970)
check("pressure deficit upgrades WATCH → WARNING",
      esc_press == EscalationLevel.WARNING,
      f"got {esc_press.value}")

# Critical cannot upgrade further
esc_crit = determine_escalation(200, 20, rainfall_mm=200, pressure_hpa=950)
check("critical stays critical",
      esc_crit == EscalationLevel.CRITICAL)


# ═══════════════════════════════════════════════════════════════════════════
# 11. CYCLONE RISK SCORE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

section("11. CYCLONE RISK SCORE COMPUTATION")

score = compute_cyclone_risk(120, 80, 200, 990)
check("composite in [0, 1]", 0 <= score.composite_score <= 1)
check("risk level exists", score.risk_level in list(CycloneRisk))
check("category = very severe CS (120 >= 118)", score.category == CycloneCategory.VERY_SEVERE_CS)
check("wind component > 0", score.wind_component > 0)
check("rain component > 0", score.rainfall_component > 0)
check("distance component > 0", score.distance_component > 0)
check("pressure component > 0", score.pressure_component > 0)

# Higher wind → higher risk
score_low = compute_cyclone_risk(70, 20, 200, 1010)
score_high = compute_cyclone_risk(200, 150, 50, 960)
check("extreme > mild risk",
      score_high.composite_score > score_low.composite_score)

# to_dict
d = score.to_dict()
check("to_dict has composite_score", "composite_score" in d)
check("to_dict has components", "components" in d)
check("to_dict has weights", "weights" in d)

# Zero wind below threshold → risk components reflect no wind
score_calm = compute_cyclone_risk(30, 0, 100, 1013)
check("calm: wind component = 0", score_calm.wind_component == 0)


# ═══════════════════════════════════════════════════════════════════════════
# 12. CYCLONE CONDITION DETECTION
# ═══════════════════════════════════════════════════════════════════════════

section("12. CYCLONE CONDITION DETECTION")

# Cyclonic wind
det_wind = detect_cyclone_conditions(80)
check("80 km/h is cyclonic", det_wind.is_cyclonic)
check("category = cyclonic storm", det_wind.category == CycloneCategory.CYCLONIC_STORM)
check("conditions list non-empty", len(det_wind.conditions_met) > 0)
check("risk score attached", det_wind.risk_score is not None)

# Heavy rain only (no wind)
det_rain = detect_cyclone_conditions(20, rainfall_mm=100)
check("heavy rain only: is_cyclonic", det_rain.is_cyclonic)
check("heavy_rainfall flag", det_rain.heavy_rainfall)

# Pressure only
det_press = detect_cyclone_conditions(20, pressure_hpa=990)
check("pressure deficit only: is_cyclonic", det_press.is_cyclonic)

# Calm conditions
det_calm = detect_cyclone_conditions(15, rainfall_mm=10, pressure_hpa=1015)
check("calm: not cyclonic", not det_calm.is_cyclonic)
check("calm: no conditions", len(det_calm.conditions_met) == 0)

# Gust factor
det_gust = detect_cyclone_conditions(40, wind_gust_kmh=90)
check("gust 90 → effective ≈ 76.5",
      det_gust.wind_speed_kmh > 70,
      f"got {det_gust.wind_speed_kmh}")
check("gust triggers cyclonic", det_gust.is_cyclonic)

# to_dict
d = det_wind.to_dict()
check("to_dict has is_cyclonic", "is_cyclonic" in d)
check("to_dict has category", "category" in d)


# ═══════════════════════════════════════════════════════════════════════════
# 13. CYCLONE RADIUS FILTERING
# ═══════════════════════════════════════════════════════════════════════════

section("13. CYCLONE RADIUS FILTERING")

cyclone_events = [
    CycloneEventClass(
        event_id="cy1", name="Cyclone Alpha",
        latitude=12.0, longitude=82.0,
        wind_speed_kmh=120, pressure_hpa=980,
        rainfall_24h_mm=100,
        timestamp=datetime.now(timezone.utc),
    ),
    CycloneEventClass(
        event_id="cy2", name="Storm Beta",
        latitude=13.5, longitude=80.5,
        wind_speed_kmh=70, pressure_hpa=1000,
        rainfall_24h_mm=40,
        timestamp=datetime.now(timezone.utc),
    ),
    CycloneEventClass(
        event_id="cy3", name="Weak Depression",
        latitude=14.0, longitude=81.0,
        wind_speed_kmh=45, pressure_hpa=1008,
        rainfall_24h_mm=20,
        timestamp=datetime.now(timezone.utc),
    ),
]

# Filter from Chennai (13.08, 80.27) with default wind threshold 60
cfilt = filter_cyclones_by_radius(
    cyclone_events, user_lat=13.08, user_lon=80.27,
    radius_km=500.0, wind_threshold=60.0,
)
check("filtered count = 2 (wind ≥ 60)", cfilt.count == 2,
      f"got {cfilt.count}")
check("weak depression excluded",
      all(e.event_id != "cy3" for e in cfilt.matched))
check("distances populated",
      all(e.distance_km is not None for e in cfilt.matched))
check("risk scores attached",
      all(e.risk_score is not None for e in cfilt.matched))
check("highest escalation ≥ WATCH",
      list(EscalationLevel).index(cfilt.highest_escalation) >= 1)

# Sort by risk
cfilt_risk = filter_cyclones_by_radius(
    cyclone_events, user_lat=13.08, user_lon=80.27,
    radius_km=500.0, sort_by="risk",
)
if cfilt_risk.count >= 2:
    check("risk sort: highest risk first",
          cfilt_risk.matched[0].risk_score.composite_score
          >= cfilt_risk.matched[1].risk_score.composite_score)

# to_dict
d = cfilt.to_dict()
check("filter to_dict has matched_count", "matched_count" in d)
check("filter to_dict has highest_escalation", "highest_escalation" in d)


# ═══════════════════════════════════════════════════════════════════════════
# 14. CYCLONE ASSESSMENT PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

section("14. CYCLONE ASSESSMENT PIPELINE")

assessment = assess_cyclone_risk(
    latitude=13.08, longitude=80.27,
    wind_speed_kmh=130, rainfall_mm=80,
    pressure_hpa=975, wind_gust_kmh=180,
)
check("assessment has detection", assessment.detection is not None)
check("assessment is cyclonic", assessment.detection.is_cyclonic)
check("assessment escalation ≥ ALERT",
      list(EscalationLevel).index(assessment.escalation) >= 3,
      f"got {assessment.escalation.value}")
check("alert has action", "action" in assessment.alert)
check("alert has message", "message" in assessment.alert)

# Calm assessment
calm_assess = assess_cyclone_risk(
    latitude=13.08, longitude=80.27,
    wind_speed_kmh=20, rainfall_mm=5, pressure_hpa=1015,
)
check("calm: not cyclonic", not calm_assess.detection.is_cyclonic)
check("calm: escalation = NONE",
      calm_assess.escalation == EscalationLevel.NONE)

# to_dict
d = assessment.to_dict()
check("assessment to_dict has detection", "detection" in d)
check("assessment to_dict has alert", "alert" in d)


# ═══════════════════════════════════════════════════════════════════════════
# 15. EDGE CASES & BOUNDARY CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════

section("15. EDGE CASES & BOUNDARY CONDITIONS")

# Impact radius at M0
m0 = estimate_impact_radius(0.0, 10.0)
check("M0: felt radius small", m0.felt_radius_km < 1.0,
      f"got {m0.felt_radius_km}")

# Impact at extreme depth
deep = estimate_impact_radius(7.0, 700.0)
check("M7 at 700km: reduced impact",
      deep.felt_radius_km < m7.felt_radius_km,
      f"deep={deep.felt_radius_km}, shallow_m7={m7.felt_radius_km}")

# Cyclone risk at distance=0
score_d0 = compute_cyclone_risk(150, 100, 0.0, 970)
check("distance=0: dist component = 1.0",
      score_d0.distance_component == 1.0)

# Cyclone risk at max distance
score_dmax = compute_cyclone_risk(150, 100, 500.0, 970)
check("distance=max: dist component = 0.0",
      score_dmax.distance_component == 0.0)

# Cyclone risk classify boundaries
check("risk 0.0 → NONE", classify_cyclone_risk(0.0) == CycloneRisk.NONE)
check("risk 0.14 → NONE", classify_cyclone_risk(0.14) == CycloneRisk.NONE)
check("risk 0.15 → LOW", classify_cyclone_risk(0.15) == CycloneRisk.LOW)
check("risk 0.35 → MODERATE", classify_cyclone_risk(0.35) == CycloneRisk.MODERATE)
check("risk 0.60 → HIGH", classify_cyclone_risk(0.60) == CycloneRisk.HIGH)
check("risk 0.80 → EXTREME", classify_cyclone_risk(0.80) == CycloneRisk.EXTREME)

# Alert generation
alert_crit = get_cyclone_alert(EscalationLevel.CRITICAL)
check("critical alert has SHELTER", "SHELTER" in alert_crit["action"])
alert_none = get_cyclone_alert(EscalationLevel.NONE)
check("none alert has no action", "No action" in alert_none["action"])

# Earthquake event to_dict
eq_dict = test_events[0].to_dict()
check("eq to_dict has impact", "impact" in eq_dict)
check("eq to_dict has risk_level", "risk_level" in eq_dict)
check("eq to_dict has depth_class", "depth_class" in eq_dict)

# CycloneEvent to_dict
cy_dict = cyclone_events[0].to_dict()
check("cy to_dict has category", "category" in cy_dict)
check("cy to_dict has wind_speed_kmh", "wind_speed_kmh" in cy_dict)


# ═══════════════════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{'='*50}")
print(f"RESULTS:  {_pass} passed,  {_fail} failed,  {_pass + _fail} total")
print(f"{'='*50}")

sys.exit(0 if _fail == 0 else 1)
