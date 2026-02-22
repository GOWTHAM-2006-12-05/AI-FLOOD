"""
siren_api.py — Physical siren / outdoor warning system integration.

Delivery mechanism:
    • HTTP API calls to municipal siren control systems
    • Activates outdoor sirens within the alert geo-fence
    • Siren patterns indicate severity (steady / wail / pulse)

═══════════════════════════════════════════════════════════════════════════
SIREN API DESIGN
═══════════════════════════════════════════════════════════════════════════

Physical sirens are only activated at CRITICAL priority because:
    1. Sirens cannot be "unseen" — they affect the entire neighbourhood
    2. False siren activations erode public trust rapidly
    3. Operational cost is high (power, maintenance, noise ordinances)
    4. Sirens serve as the "last resort" channel for unreachable populations

Siren patterns (modelled on the International Alert System):

    Pattern        Duration    Meaning
    ──────────     ────────    ─────────────────────────────────
    STEADY_TONE    3 min       General emergency — take cover
    WAIL           3 min       Imminent danger — evacuate
    PULSE          1 min       All-clear / test

Siren control API (simulated):

    POST /api/sirens/activate
    {
        "zone_id": "ZONE-CHN-001",
        "pattern": "WAIL",
        "duration_seconds": 180,
        "lat": 13.08, "lon": 80.27, "radius_km": 10,
        "alert_id": "ALR-3A7B...",
        "authorized_by": "disaster-prediction-ai"
    }
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any

from backend.app.alerts.models import (
    AlertChannel,
    AlertPayload,
    AlertRecipient,
    DeliveryAttempt,
    DeliveryStatus,
)

logger = logging.getLogger(__name__)


class SirenPattern(str, Enum):
    STEADY_TONE = "steady_tone"     # general emergency
    WAIL        = "wail"            # imminent danger / evacuate
    PULSE       = "pulse"           # all-clear or test


def _select_pattern(payload: AlertPayload) -> SirenPattern:
    """Select siren pattern based on hazard type and risk score."""
    if payload.source_risk_score >= 85:
        return SirenPattern.WAIL          # highest threat → evacuate signal
    elif payload.source_risk_score >= 70:
        return SirenPattern.STEADY_TONE   # take cover
    else:
        return SirenPattern.PULSE         # informational


def send(
    payload: AlertPayload,
    recipient: AlertRecipient,
    *,
    siren_api_url: str = "http://localhost:9090/api/sirens/activate",
    zone_id: Optional[str] = None,
    timeout_seconds: float = 10.0,
) -> DeliveryAttempt:
    """
    Activate a physical siren for the alert zone.

    NOTE: The "recipient" parameter is used to determine the siren zone.
    In practice, sirens cover geographic zones, not individual users.
    One siren activation per zone covers all recipients in that zone.

    Parameters
    ----------
    payload : AlertPayload
    recipient : AlertRecipient
        Used for zone determination (latitude/longitude).
    siren_api_url : str
        URL of the siren control API.
    zone_id : str | None
        Override zone ID; auto-generated from coordinates if None.
    timeout_seconds : float

    Returns
    -------
    DeliveryAttempt
    """
    attempt = DeliveryAttempt(
        channel=AlertChannel.SIREN_API,
        recipient_id=recipient.recipient_id,
        status=DeliveryStatus.SENDING,
    )

    try:
        # ── Only CRITICAL priority activates sirens ──
        if payload.priority.value < 4:
            attempt.status = DeliveryStatus.SKIPPED
            attempt.completed_at = datetime.now(timezone.utc)
            attempt.error_message = (
                f"Siren only for CRITICAL; current priority={payload.priority.name}"
            )
            return attempt

        # ── Determine zone ──
        if zone_id is None:
            # Generate zone from lat/lon grid (0.1° ≈ 11 km cells)
            lat_grid = round(payload.latitude, 1)
            lon_grid = round(payload.longitude, 1)
            zone_id = f"ZONE-{lat_grid:.1f}-{lon_grid:.1f}"

        pattern = _select_pattern(payload)

        siren_request: Dict[str, Any] = {
            "zone_id": zone_id,
            "pattern": pattern.value,
            "duration_seconds": 180,
            "lat": payload.latitude,
            "lon": payload.longitude,
            "radius_km": payload.radius_km,
            "alert_id": payload.alert_id,
            "authorized_by": "disaster-prediction-ai",
            "hazard_type": payload.hazard_type,
        }

        # ── Simulated activation ──
        # Production: httpx.post(siren_api_url, json=siren_request, timeout=timeout_seconds)
        logger.warning(
            "[SIREN] 🔊 Activating siren zone %s | pattern=%s | alert=%s | "
            "hazard=%s | risk=%.1f%%",
            zone_id, pattern.value, payload.alert_id,
            payload.hazard_type, payload.source_risk_score,
        )

        attempt.status = DeliveryStatus.DELIVERED
        attempt.completed_at = datetime.now(timezone.utc)
        attempt.provider_response = {
            "mode": "simulated",
            "zone_id": zone_id,
            "pattern": pattern.value,
            "duration_seconds": 180,
            "siren_request": siren_request,
        }

    except Exception as exc:
        logger.error("[SIREN] Activation failed for zone %s: %s", zone_id, exc)
        attempt.status = DeliveryStatus.FAILED
        attempt.completed_at = datetime.now(timezone.utc)
        attempt.error_message = str(exc)

    return attempt
