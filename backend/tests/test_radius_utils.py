"""
test_radius_utils.py — Demonstrates and tests the radius filtering system.

Run:
    cd "AI disaster predition"
    python -m backend.tests.test_radius_utils

Outputs clear, annotated examples showing:
    1. Haversine distance calculations
    2. Point-in-radius checks
    3. Full disaster list filtering
    4. Edge cases
"""

from __future__ import annotations

import json
import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.app.spatial.radius_utils import (
    Coordinate,
    DisasterEvent,
    FilterResult,
    RadiusPreset,
    filter_disasters,
    format_distance,
    haversine,
    is_inside_radius,
)


def separator(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


# =========================================================================
# EXAMPLE 1 — Haversine Distance Calculations
# =========================================================================

def example_haversine():
    separator("EXAMPLE 1 — Haversine Distance Calculations")

    pairs = [
        ("Chennai Central", Coordinate(13.0827, 80.2707),
         "Chennai Airport",  Coordinate(12.9941, 80.1709)),

        ("Chennai",          Coordinate(13.0827, 80.2707),
         "Bangalore",        Coordinate(12.9716, 77.5946)),

        ("Mumbai",           Coordinate(19.0760, 72.8777),
         "Delhi",            Coordinate(28.6139, 77.2090)),

        ("New York",         Coordinate(40.7128, -74.0060),
         "London",           Coordinate(51.5074, -0.1278)),

        ("Same Point",       Coordinate(0.0, 0.0),
         "Same Point",       Coordinate(0.0, 0.0)),
    ]

    print(f"  {'From':<20} {'To':<20} {'Distance':>12}")
    print(f"  {'─' * 20} {'─' * 20} {'─' * 12}")

    for name1, p1, name2, p2 in pairs:
        dist = haversine(p1, p2)
        print(f"  {name1:<20} {name2:<20} {format_distance(dist):>12}")

    # Show raw values for documentation
    print("\n  Raw numeric outputs:")
    chennai  = Coordinate(13.0827, 80.2707)
    airport  = Coordinate(12.9941, 80.1709)
    dist = haversine(chennai, airport)
    print(f"    haversine(Chennai, Airport) = {dist} km")

    bangalore = Coordinate(12.9716, 77.5946)
    dist2 = haversine(chennai, bangalore)
    print(f"    haversine(Chennai, Bangalore) = {dist2} km")


# =========================================================================
# EXAMPLE 2 — Point-in-Radius Check
# =========================================================================

def example_radius_check():
    separator("EXAMPLE 2 — Is Disaster Inside Radius?")

    user = Coordinate(13.0827, 80.2707)  # Chennai
    print(f"  User location: Chennai ({user.latitude}, {user.longitude})\n")

    checks = [
        ("Nearby tremor",   Coordinate(13.10,   80.30),   5.0),
        ("Nearby tremor",   Coordinate(13.10,   80.30),  10.0),
        ("Flood at Adyar",  Coordinate(13.0067, 80.2565), 10.0),
        ("Cyclone at sea",  Coordinate(11.50,   82.00),   50.0),
        ("Bangalore quake", Coordinate(12.9716, 77.5946), 50.0),
    ]

    print(f"  {'Event':<20} {'Radius':>8} {'Inside':>8} {'Distance':>12}")
    print(f"  {'─' * 20} {'─' * 8} {'─' * 8} {'─' * 12}")

    for name, loc, radius in checks:
        inside, dist = is_inside_radius(user, loc, radius)
        mark = "✅ YES" if inside else "❌ NO"
        print(f"  {name:<20} {radius:>6.0f}km {mark:>8} {format_distance(dist):>12}")


# =========================================================================
# EXAMPLE 3 — Full Disaster List Filtering
# =========================================================================

def example_filter():
    separator("EXAMPLE 3 — Filter Disaster List by Radius")

    user = Coordinate(13.0827, 80.2707)  # Chennai

    disasters = [
        DisasterEvent(
            id="EQ-001", title="Minor Tremor near Tambaram",
            hazard_type="earthquake",
            location=Coordinate(12.9249, 80.1000),
            severity=2, timestamp="2026-02-22T06:14:00Z",
        ),
        DisasterEvent(
            id="FL-001", title="Adyar River Flooding",
            hazard_type="flood",
            location=Coordinate(13.0067, 80.2565),
            severity=4, timestamp="2026-02-22T09:30:00Z",
        ),
        DisasterEvent(
            id="CY-001", title="Cyclone DANA — Cat 2",
            hazard_type="cyclone",
            location=Coordinate(11.5000, 82.0000),
            severity=5, timestamp="2026-02-22T03:00:00Z",
        ),
        DisasterEvent(
            id="HW-001", title="Severe Heatwave",
            hazard_type="heatwave",
            location=Coordinate(13.0878, 80.2785),
            severity=3, timestamp="2026-02-22T12:00:00Z",
        ),
        DisasterEvent(
            id="LS-001", title="Landslide Risk — Nilgiris",
            hazard_type="landslide",
            location=Coordinate(11.4102, 76.6950),
            severity=4, timestamp="2026-02-22T07:45:00Z",
        ),
        DisasterEvent(
            id="FL-002", title="Waterlogging at T. Nagar",
            hazard_type="flood",
            location=Coordinate(13.0418, 80.2341),
            severity=2, timestamp="2026-02-22T10:15:00Z",
        ),
    ]

    for radius in [5.0, 10.0, 20.0, 50.0]:
        result = filter_disasters(user, disasters, radius_km=radius)
        print(f"  Radius: {radius:.0f} km → {result.count} match(es), "
              f"{result.excluded} excluded")
        for e in result.matched:
            print(f"    • [{e.id}] {e.title} — {format_distance(e.distance_km)}")
        print()


# =========================================================================
# EXAMPLE 4 — Filtered by Severity + Hazard Type
# =========================================================================

def example_advanced_filter():
    separator("EXAMPLE 4 — Filter with Severity + Hazard Type")

    user = Coordinate(13.0827, 80.2707)

    disasters = [
        DisasterEvent("A", "Low flood",  "flood",      Coordinate(13.05, 80.25), 1, "2026-02-22T00:00:00Z"),
        DisasterEvent("B", "Med flood",  "flood",      Coordinate(13.06, 80.26), 3, "2026-02-22T00:00:00Z"),
        DisasterEvent("C", "Quake",      "earthquake", Coordinate(13.07, 80.28), 4, "2026-02-22T00:00:00Z"),
        DisasterEvent("D", "Heatwave",   "heatwave",   Coordinate(13.08, 80.27), 2, "2026-02-22T00:00:00Z"),
    ]

    # Only severity >= 3
    result = filter_disasters(user, disasters, radius_km=20.0, min_severity=3)
    print(f"  Filter: radius=20km, min_severity=3")
    print(f"  Results: {result.count}")
    for e in result.matched:
        print(f"    • [{e.id}] {e.title} (sev={e.severity}) — {format_distance(e.distance_km)}")

    print()

    # Only floods
    result2 = filter_disasters(user, disasters, radius_km=20.0, hazard_types={"flood"})
    print(f"  Filter: radius=20km, hazard_types={{flood}}")
    print(f"  Results: {result2.count}")
    for e in result2.matched:
        print(f"    • [{e.id}] {e.title} — {format_distance(e.distance_km)}")


# =========================================================================
# EXAMPLE 5 — Edge Cases
# =========================================================================

def example_edge_cases():
    separator("EXAMPLE 5 — Edge Cases")

    # Antipodal points (maximum distance ~20,015 km)
    north_pole = Coordinate(90.0, 0.0)
    south_pole = Coordinate(-90.0, 0.0)
    dist = haversine(north_pole, south_pole)
    print(f"  North Pole ↔ South Pole: {format_distance(dist)}")

    # International date line
    p1 = Coordinate(0.0, 179.9)
    p2 = Coordinate(0.0, -179.9)
    dist2 = haversine(p1, p2)
    print(f"  Across date line (179.9° ↔ -179.9°): {format_distance(dist2)}")

    # Invalid coordinates
    print("\n  Validation tests:")
    try:
        Coordinate(91.0, 0.0)
        print("    ❌ Should have raised ValueError for lat=91")
    except ValueError as e:
        print(f"    ✅ Caught: {e}")

    try:
        Coordinate(0.0, 181.0)
        print("    ❌ Should have raised ValueError for lon=181")
    except ValueError as e:
        print(f"    ✅ Caught: {e}")

    try:
        is_inside_radius(Coordinate(0, 0), Coordinate(1, 1), radius_km=-5)
        print("    ❌ Should have raised ValueError for negative radius")
    except ValueError as e:
        print(f"    ✅ Caught: {e}")

    # Empty disaster list
    result = filter_disasters(Coordinate(0, 0), [], radius_km=10.0)
    print(f"\n  Empty list filter: count={result.count}, excluded={result.excluded}")


# =========================================================================
# EXAMPLE 6 — Simulated API Request/Response
# =========================================================================

def example_api_simulation():
    separator("EXAMPLE 6 — Simulated API Request → Response")

    # This is exactly what the FastAPI endpoint receives and returns

    request_body = {
        "location": {
            "latitude": 13.0827,
            "longitude": 80.2707,
            "source": "gps"
        },
        "radius_km": 10,
        "min_severity": 1,
        "hazard_types": None,
        "max_results": 50,
    }
    print("  REQUEST (POST /api/v1/disasters/nearby):")
    print(f"  {json.dumps(request_body, indent=4)}")

    # Simulate the processing
    user = Coordinate(
        request_body["location"]["latitude"],
        request_body["location"]["longitude"],
    )
    disasters = [
        DisasterEvent("FL-001", "Adyar Flooding", "flood",
                       Coordinate(13.0067, 80.2565), 4, "2026-02-22T09:30:00Z"),
        DisasterEvent("HW-001", "Heatwave", "heatwave",
                       Coordinate(13.0878, 80.2785), 3, "2026-02-22T12:00:00Z"),
        DisasterEvent("CY-001", "Cyclone DANA", "cyclone",
                       Coordinate(11.5, 82.0), 5, "2026-02-22T03:00:00Z"),
    ]
    result = filter_disasters(user, disasters, radius_km=10.0)

    response = {
        "user_location": {"latitude": user.latitude, "longitude": user.longitude},
        "radius_km": result.radius_km,
        "total_checked": result.total_checked,
        "results_count": result.count,
        "excluded_count": result.excluded,
        "disasters": [
            {
                "id": e.id,
                "title": e.title,
                "hazard_type": e.hazard_type,
                "severity": e.severity,
                "distance_km": round(e.distance_km, 2),
                "distance_display": format_distance(e.distance_km),
            }
            for e in result.matched
        ],
    }

    print("\n  RESPONSE:")
    print(f"  {json.dumps(response, indent=4)}")


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("  RADIUS UTILS — TEST & DEMO SUITE")
    print("█" * 70)

    example_haversine()
    example_radius_check()
    example_filter()
    example_advanced_filter()
    example_edge_cases()
    example_api_simulation()

    separator("ALL EXAMPLES COMPLETE ✅")
