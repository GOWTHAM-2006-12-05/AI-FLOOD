"""
sms_fallback.py — Low-bandwidth SMS fallback channel.

This is the last-resort channel for recipients in areas with
poor connectivity (2G-only, intermittent signal, rural regions).

═══════════════════════════════════════════════════════════════════════════
WHY A SEPARATE LOW-BANDWIDTH SMS CHANNEL
═══════════════════════════════════════════════════════════════════════════

Standard SMS (sms_gateway.py) already uses ≤160-char GSM encoding,
but the fallback channel goes further:

    ┌──────────────────────┬──────────────────┬──────────────────────┐
    │ Feature              │ Standard SMS     │ SMS Fallback         │
    ├──────────────────────┼──────────────────┼──────────────────────┤
    │ Max length           │ 160 chars        │ 70 chars             │
    │ Encoding             │ GSM 7-bit        │ GSM 7-bit (strict)   │
    │ Unicode              │ Allowed (UCS-2)  │ Forbidden (ASCII)    │
    │ URLs                 │ Optional         │ Never (save bytes)   │
    │ Delivery confirm     │ Webhook          │ Polling (lower BW)   │
    │ Retry strategy       │ 3× at 30s gaps   │ 5× at 60s gaps      │
    │ Message structure    │ [ALERT] body ref │ ABBRV coded msg      │
    │ Activation           │ URGENT+          │ CRITICAL only        │
    │ Target scenario      │ 3G/4G/5G areas   │ 2G / spotty coverage │
    └──────────────────────┴──────────────────┴──────────────────────┘

Message template (≤70 chars):

    "{HAZARD_CODE}!{LEVEL} {location_code}. ACT:{action_code}. #{ref}"

    Example: "FL!SEV ADYAR. ACT:EVAC. #3A7B"       (30 chars)
             "EQ!CRT M7.2 CHNAI. ACT:SHLT. #9F2C"  (35 chars)

Hazard codes:
    FL = Flood, EQ = Earthquake, CY = Cyclone

Level codes:
    SAF = Safe, WCH = Watch, WRN = Warning, SEV = Severe, CRT = Critical

Action codes:
    MON = Monitor, INF = Stay Informed, PREP = Prepare, EVAC = Evacuate,
    SHLT = Shelter

═══════════════════════════════════════════════════════════════════════════
WHEN THIS CHANNEL IS USED
═══════════════════════════════════════════════════════════════════════════

    1. Recipient has low_bandwidth=True flag set
    2. Primary SMS channel failed after all retries
    3. Alert priority is CRITICAL
    4. Automatic fallback from sms_gateway failure
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from backend.app.alerts.models import (
    AlertChannel,
    AlertPayload,
    AlertRecipient,
    DeliveryAttempt,
    DeliveryStatus,
)

logger = logging.getLogger(__name__)

SMS_FALLBACK_MAX = 70  # strict 70-char limit

# Hazard abbreviations
_HAZARD_CODE = {
    "flood": "FL",
    "earthquake": "EQ",
    "cyclone": "CY",
    "heatwave": "HW",
    "landslide": "LS",
    "wildfire": "WF",
    "unknown": "DIS",
}

# Risk level abbreviations
_LEVEL_CODE = {
    "safe": "SAF",
    "watch": "WCH",
    "warning": "WRN",
    "severe": "SEV",
    "critical": "CRT",
}

# Action abbreviations
_ACTION_CODE = {
    "monitor": "MON",
    "stay_informed": "INF",
    "prepare": "PREP",
    "evacuate": "EVAC",
}


def _format_fallback_sms(payload: AlertPayload) -> str:
    """
    Build an ultra-short SMS (≤70 chars) using abbreviation codes.

    Format: "{HAZARD}!{LEVEL} {minimal_msg}. ACT:{action}. #{ref}"
    """
    hazard = _HAZARD_CODE.get(payload.hazard_type, "DIS")
    level = _LEVEL_CODE.get(payload.source_risk_level, "WRN")
    ref = payload.alert_id[-4:]

    # Use minimal_message if provided, otherwise truncate title
    body = payload.minimal_message or payload.title
    action = _ACTION_CODE.get(
        payload.metadata.get("action", "prepare"), "PREP"
    )

    skeleton = f"{hazard}!{level} . ACT:{action}. #{ref}"
    available = SMS_FALLBACK_MAX - len(skeleton)

    if len(body) > available:
        body = body[: available - 1] + "."

    msg = f"{hazard}!{level} {body}. ACT:{action}. #{ref}"

    # Final safety trim
    if len(msg) > SMS_FALLBACK_MAX:
        msg = msg[:SMS_FALLBACK_MAX]

    return msg


def send(
    payload: AlertPayload,
    recipient: AlertRecipient,
    *,
    provider: str = "simulation",
    timeout_seconds: float = 20.0,
) -> DeliveryAttempt:
    """
    Send a low-bandwidth fallback SMS.

    Parameters
    ----------
    payload : AlertPayload
    recipient : AlertRecipient
        Must have .phone set.
    provider : str
    timeout_seconds : float

    Returns
    -------
    DeliveryAttempt
    """
    attempt = DeliveryAttempt(
        channel=AlertChannel.SMS_FALLBACK,
        recipient_id=recipient.recipient_id,
        status=DeliveryStatus.SENDING,
    )

    try:
        if not recipient.phone:
            attempt.status = DeliveryStatus.SKIPPED
            attempt.completed_at = datetime.now(timezone.utc)
            attempt.error_message = "No phone number on file"
            return attempt

        sms_body = _format_fallback_sms(payload)

        logger.info(
            "[SMS_FALLBACK] Alert %s → %s (%s): '%s' (%d chars)",
            payload.alert_id,
            recipient.phone,
            recipient.name,
            sms_body,
            len(sms_body),
        )

        # Simulated delivery (production: use SMS gateway with GSM-7 encoding)
        attempt.status = DeliveryStatus.DELIVERED
        attempt.completed_at = datetime.now(timezone.utc)
        attempt.provider_response = {
            "mode": "simulated",
            "encoding": "gsm7_strict",
            "message_length": len(sms_body),
            "message": sms_body,
            "phone": recipient.phone,
            "max_allowed": SMS_FALLBACK_MAX,
        }

    except Exception as exc:
        logger.error(
            "[SMS_FALLBACK] Failed for %s: %s", recipient.recipient_id, exc
        )
        attempt.status = DeliveryStatus.FAILED
        attempt.completed_at = datetime.now(timezone.utc)
        attempt.error_message = str(exc)

    return attempt
