"""
geo_fence.py — Spatial targeting for alert broadcasting.

Determines which recipients fall within an alert's geographic zone
using the same Haversine mathematics as the radius filtering module.

═══════════════════════════════════════════════════════════════════════════
GEO-FENCE DESIGN
═══════════════════════════════════════════════════════════════════════════

An alert defines a circular geo-fence:

    centre:  (alert.latitude, alert.longitude)   — event epicentre
    radius:  alert.radius_km                      — impact zone

A recipient is targeted if:

    haversine(event_location, recipient_location) ≤ alert_radius_km

For large recipient lists, a bounding-box pre-filter eliminates most
candidates before running the O(1)-but-expensive trig in Haversine:

    Step 1 — Compute bounding box (lat_min, lat_max, lon_min, lon_max)
    Step 2 — Reject recipients outside the box (simple float comparison)
    Step 3 — Run Haversine only on candidates inside the box

This reduces a 100 000-recipient broadcast from 100 000 Haversine calls
to ~500 (the ones actually near the event), a 200× speedup.

═══════════════════════════════════════════════════════════════════════════
DYNAMIC RADIUS EXPANSION
═══════════════════════════════════════════════════════════════════════════

For CRITICAL events, the geo-fence radius is automatically expanded:

    effective_radius = base_radius × expansion_factor

    Priority          expansion_factor
    ──────────        ────────────────
    INFORMATIONAL     1.0×
    ADVISORY          1.0×
    URGENT            1.25×
    CRITICAL          1.50×

This ensures that severe events reach a wider audience even beyond
the calculated impact zone, accounting for:
    - Evacuation routes that extend beyond the zone
    - Secondary effects (e.g. tsunami from earthquake)
    - Error margins in impact radius estimation
"""

from __future__ import annotations

import logging
import math
from typing import List, Tuple

from backend.app.alerts.models import AlertPayload, AlertPriority, AlertRecipient
from backend.app.spatial.radius_utils import Coordinate, haversine, EARTH_RADIUS_KM

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

EXPANSION_FACTORS = {
    AlertPriority.INFORMATIONAL: 1.0,
    AlertPriority.ADVISORY: 1.0,
    AlertPriority.URGENT: 1.25,
    AlertPriority.CRITICAL: 1.50,
}


# ═══════════════════════════════════════════════════════════════════════════
# Bounding-box Pre-filter
# ═══════════════════════════════════════════════════════════════════════════

def _bounding_box(
    lat: float, lon: float, radius_km: float,
) -> Tuple[float, float, float, float]:
    """
    Compute (min_lat, max_lat, min_lon, max_lon) bounding box.

    Uses spherical approximation — identical maths to radius_utils
    but inlined here to avoid coupling on internal function.
    """
    angular = radius_km / EARTH_RADIUS_KM  # radians

    min_lat = lat - math.degrees(angular)
    max_lat = lat + math.degrees(angular)

    cos_lat = math.cos(math.radians(lat))
    if cos_lat > 1e-10:
        delta_lon = math.degrees(angular / cos_lat)
    else:
        delta_lon = 180.0

    min_lon = lon - delta_lon
    max_lon = lon + delta_lon

    return (
        max(min_lat, -90.0),
        min(max_lat, 90.0),
        max(min_lon, -180.0),
        min(max_lon, 180.0),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Core Geo-fence Filtering
# ═══════════════════════════════════════════════════════════════════════════

def compute_effective_radius(payload: AlertPayload) -> float:
    """
    Compute the effective geo-fence radius after priority expansion.

    Parameters
    ----------
    payload : AlertPayload
        The alert being broadcast.

    Returns
    -------
    float
        Effective radius in km.
    """
    factor = EXPANSION_FACTORS.get(payload.priority, 1.0)
    return payload.radius_km * factor


def filter_recipients_by_geofence(
    payload: AlertPayload,
    recipients: List[AlertRecipient],
) -> Tuple[List[AlertRecipient], List[AlertRecipient]]:
    """
    Partition recipients into (inside, outside) the alert geo-fence.

    Uses bounding-box pre-filter + Haversine for precision.

    Parameters
    ----------
    payload : AlertPayload
        Alert with epicentre coordinates and radius.
    recipients : list of AlertRecipient
        All potential recipients.

    Returns
    -------
    (targeted, excluded)
        targeted : recipients inside the geo-fence
        excluded : recipients outside

    Examples
    --------
    >>> payload = AlertPayload(latitude=13.08, longitude=80.27, radius_km=10)
    >>> r1 = AlertRecipient("U1", "User1", 13.08, 80.27)  # same location
    >>> r2 = AlertRecipient("U2", "User2", 20.0, 70.0)    # far away
    >>> targeted, excluded = filter_recipients_by_geofence(payload, [r1, r2])
    >>> len(targeted), len(excluded)
    (1, 1)
    """
    effective_radius = compute_effective_radius(payload)
    event_coord = Coordinate(payload.latitude, payload.longitude)

    # Bounding box for fast rejection
    min_lat, max_lat, min_lon, max_lon = _bounding_box(
        payload.latitude, payload.longitude, effective_radius
    )

    targeted: List[AlertRecipient] = []
    excluded: List[AlertRecipient] = []

    for recipient in recipients:
        # Fast rejection via bounding box
        if not (min_lat <= recipient.latitude <= max_lat
                and min_lon <= recipient.longitude <= max_lon):
            excluded.append(recipient)
            continue

        # Precise Haversine check
        r_coord = Coordinate(recipient.latitude, recipient.longitude)
        dist = haversine(event_coord, r_coord)

        if dist <= effective_radius:
            targeted.append(recipient)
        else:
            excluded.append(recipient)

    logger.info(
        "Geo-fence filter: %d targeted, %d excluded (radius=%.1f km, "
        "effective=%.1f km, priority=%s)",
        len(targeted), len(excluded),
        payload.radius_km, effective_radius, payload.priority.name,
    )

    return targeted, excluded
