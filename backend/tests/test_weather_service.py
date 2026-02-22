"""
test_weather_service.py — Demonstrates and tests the weather ingestion module.

Run:
    cd "AI disaster predition"
    python -m backend.tests.test_weather_service

Tests against the LIVE Open-Meteo API (free, no key needed).
Shows:
    1. Full weather fetch for Chennai
    2. Rainfall accumulation breakdown
    3. ML feature vector generation
    4. Error handling for invalid inputs
    5. Multiple city comparison
"""

from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.app.ingestion.weather_service import (
    FetchStatus,
    RainfallAccumulation,
    fetch_rainfall_summary,
    fetch_weather,
)
from backend.app.features.weather_features import build_features, print_features


def separator(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


# =========================================================================
# TEST 1 — Full Weather Fetch (Chennai)
# =========================================================================

def test_full_fetch():
    separator("TEST 1 — Full Weather Fetch (Chennai)")

    result = fetch_weather(13.0827, 80.2707)

    print(f"  Success:       {result.success}")
    print(f"  Status:        {result.status.value}")
    print(f"  Latitude:      {result.latitude}")
    print(f"  Longitude:     {result.longitude}")
    print(f"  Elevation:     {result.elevation_m} m")
    print(f"  Timezone:      {result.timezone_name}")
    print(f"  Data gaps:     {result.has_data_gaps}")
    print(f"  Fetch time:    {result.fetch_duration_ms} ms")

    if result.current:
        print(f"\n  Current Conditions:")
        print(f"    Temperature:   {result.current.temperature_c}°C")
        print(f"    Humidity:      {result.current.humidity_pct}%")
        print(f"    Wind speed:    {result.current.wind_speed_kmh} km/h")
        print(f"    Wind gusts:    {result.current.wind_gusts_kmh} km/h")
        print(f"    Pressure:      {result.current.pressure_hpa} hPa")
        print(f"    Cloud cover:   {result.current.cloud_cover_pct}%")
        print(f"    Soil moisture: {result.current.soil_moisture}")
        print(f"    Timestamp:     {result.current.timestamp}")

    if result.hourly:
        n = len(result.hourly.timestamps)
        print(f"\n  Hourly data:     {n} hours retrieved")
        print(f"    First hour:    {result.hourly.timestamps[0]}")
        print(f"    Last hour:     {result.hourly.timestamps[-1]}")

    return result


# =========================================================================
# TEST 2 — Rainfall Accumulation Detail
# =========================================================================

def test_rainfall(result=None):
    separator("TEST 2 — Rainfall Accumulation")

    if result is None:
        result = fetch_weather(13.0827, 80.2707)

    if not result.success:
        print("  ❌ Fetch failed, skipping rainfall test")
        return

    rf = result.rainfall
    print(f"  Rain_1hr:      {rf.rain_1hr} mm")
    print(f"  Rain_3hr:      {rf.rain_3hr} mm")
    print(f"  Rain_6hr:      {rf.rain_6hr} mm")
    print(f"  Rain_24hr:     {rf.rain_24hr} mm")
    print(f"  Intensity ratio: {rf.intensity_ratio}")
    print(f"  Flood risk:    {rf.flood_risk_category}")

    print(f"\n  How accumulation was computed:")
    print(f"    Rain_1hr  = precip at current hour")
    print(f"    Rain_3hr  = sum of last 3 hourly precip values")
    print(f"    Rain_6hr  = sum of last 6 hourly precip values")
    print(f"    Rain_24hr = sum of last 24 hourly precip values")

    # Show the actual hourly precipitation values used
    if result.hourly and result.hourly.precipitation:
        from backend.app.ingestion.weather_service import _find_current_index
        idx = _find_current_index(result.hourly.timestamps)
        start_24 = max(0, idx - 23)
        recent = result.hourly.precipitation[start_24:idx + 1]
        print(f"\n  Last {len(recent)} hourly precip values (mm):")

        # Print in groups of 6 for readability
        for i in range(0, len(recent), 6):
            chunk = recent[i:i + 6]
            hours = [f"{v:5.1f}" for v in chunk]
            print(f"    [{', '.join(hours)}]")

        computed_24 = round(sum(recent), 2)
        print(f"\n  Manual sum check: {computed_24} mm (should match Rain_24hr = {rf.rain_24hr} mm)")


# =========================================================================
# TEST 3 — ML Feature Vector
# =========================================================================

def test_ml_features(result=None):
    separator("TEST 3 — ML Feature Vector")

    if result is None:
        result = fetch_weather(13.0827, 80.2707)

    if not result.success:
        print("  ❌ Fetch failed, skipping feature test")
        return

    features = build_features(result)
    print_features(features, title="Chennai — ML Feature Vector")

    # Show which features go to which model
    print("\n  Feature → Model mapping:")
    flood_features = [k for k in features if any(
        w in k for w in ["rain", "soil", "flood", "pressure", "elevation"]
    )]
    heat_features = [k for k in features if any(
        w in k for w in ["temp", "heat", "humidity"]
    )]
    cyclone_features = [k for k in features if any(
        w in k for w in ["wind", "pressure", "cyclonic"]
    )]

    print(f"    Flood model:   {flood_features}")
    print(f"    Heatwave model:{heat_features}")
    print(f"    Cyclone model: {cyclone_features}")


# =========================================================================
# TEST 4 — Error Handling
# =========================================================================

def test_error_handling():
    separator("TEST 4 — Error Handling")

    # Invalid latitude
    print("  Test 4a: Invalid latitude (lat=999)")
    r = fetch_weather(999, 80.0)
    print(f"    success={r.success}, status={r.status.value}")
    print(f"    error: {r.error_message}")

    # Invalid longitude
    print("\n  Test 4b: Invalid longitude (lon=999)")
    r = fetch_weather(13.0, 999)
    print(f"    success={r.success}, status={r.status.value}")
    print(f"    error: {r.error_message}")

    # Edge case: North Pole
    print("\n  Test 4c: North Pole (90, 0)")
    r = fetch_weather(90.0, 0.0)
    print(f"    success={r.success}, status={r.status.value}")
    if r.success:
        print(f"    temperature: {r.current.temperature_c}°C")
        print(f"    elevation: {r.elevation_m} m")

    # Edge case: 0,0 (Gulf of Guinea)
    print("\n  Test 4d: Null Island (0, 0)")
    r = fetch_weather(0.0, 0.0)
    print(f"    success={r.success}, status={r.status.value}")
    if r.success:
        print(f"    temperature: {r.current.temperature_c}°C")
        print(f"    elevation: {r.elevation_m} m")


# =========================================================================
# TEST 5 — Multi-City Rainfall Comparison
# =========================================================================

def test_multi_city():
    separator("TEST 5 — Multi-City Rainfall Comparison")

    cities = [
        ("Chennai",    13.0827,  80.2707),
        ("Mumbai",     19.0760,  72.8777),
        ("Kolkata",    22.5726,  88.3639),
        ("Bangalore",  12.9716,  77.5946),
        ("Delhi",      28.6139,  77.2090),
    ]

    print(f"  {'City':<12} {'Elev(m)':>8} {'Rain1h':>8} {'Rain3h':>8} "
          f"{'Rain6h':>8} {'Rain24h':>8} {'Risk':<12} {'ms':>5}")
    print(f"  {'─' * 12} {'─' * 8} {'─' * 8} {'─' * 8} "
          f"{'─' * 8} {'─' * 8} {'─' * 12} {'─' * 5}")

    for name, lat, lon in cities:
        r = fetch_weather(lat, lon)
        if r.success and r.rainfall:
            rf = r.rainfall
            print(
                f"  {name:<12} {r.elevation_m:>8.1f} {rf.rain_1hr:>8.1f} "
                f"{rf.rain_3hr:>8.1f} {rf.rain_6hr:>8.1f} {rf.rain_24hr:>8.1f} "
                f"{rf.flood_risk_category:<12} {r.fetch_duration_ms:>5}"
            )
        else:
            print(f"  {name:<12} FETCH FAILED: {r.error_message}")


# =========================================================================
# TEST 6 — Quick Rainfall Summary (convenience function)
# =========================================================================

def test_rainfall_summary():
    separator("TEST 6 — Quick Rainfall Summary")

    summary = fetch_rainfall_summary(13.0827, 80.2707)
    print("  fetch_rainfall_summary(13.0827, 80.2707) →")
    print(f"  {json.dumps(summary, indent=4)}")


# =========================================================================
# TEST 7 — Simulated API Request/Response
# =========================================================================

def test_api_simulation():
    separator("TEST 7 — Simulated API Request → Response")

    request = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "forecast_days": 2,
        "past_days": 1,
    }
    print("  REQUEST (POST /api/v1/weather/fetch):")
    print(f"  {json.dumps(request, indent=4)}")

    result = fetch_weather(**request)

    response = {
        "success": result.success,
        "status": result.status.value,
        "latitude": result.latitude,
        "longitude": result.longitude,
        "elevation_m": result.elevation_m,
        "has_data_gaps": result.has_data_gaps,
        "fetch_duration_ms": result.fetch_duration_ms,
    }

    if result.current:
        response["current"] = {
            "temperature_c": result.current.temperature_c,
            "humidity_pct": result.current.humidity_pct,
            "wind_speed_kmh": result.current.wind_speed_kmh,
            "pressure_hpa": result.current.pressure_hpa,
            "timestamp": result.current.timestamp,
        }

    if result.rainfall:
        response["rainfall"] = {
            "rain_1hr": result.rainfall.rain_1hr,
            "rain_3hr": result.rainfall.rain_3hr,
            "rain_6hr": result.rainfall.rain_6hr,
            "rain_24hr": result.rainfall.rain_24hr,
            "flood_risk": result.rainfall.flood_risk_category,
            "intensity_ratio": result.rainfall.intensity_ratio,
        }

    print("\n  RESPONSE:")
    print(f"  {json.dumps(response, indent=4)}")


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("  WEATHER SERVICE — TEST & DEMO SUITE")
    print("  (Calling LIVE Open-Meteo API — no API key needed)")
    print("█" * 70)

    result = test_full_fetch()
    test_rainfall(result)
    test_ml_features(result)
    test_error_handling()
    test_multi_city()
    test_rainfall_summary()
    test_api_simulation()

    separator("ALL TESTS COMPLETE ✅")
