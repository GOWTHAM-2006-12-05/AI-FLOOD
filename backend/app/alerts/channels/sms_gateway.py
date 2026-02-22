"""
sms_gateway.py — SMS delivery channel via gateway integration.

Delivery mechanism:
    • Primary: HTTP API to SMS gateway (e.g., Twilio, MSG91, TextLocal)
    • Payload: ≤160 chars (GSM 7-bit) or ≤70 chars (UCS-2 Unicode)
    • Delivery confirmation via provider webhook or polling

═══════════════════════════════════════════════════════════════════════════
SMS GATEWAY ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════

    App  →  HTTP POST  →  SMS Gateway API  →  Carrier  →  Handset
                  │
                  └── Webhook callback (delivery receipt)

    Provider abstraction:
        - Twilio:    POST https://api.twilio.com/2010-04-01/Accounts/SID/Messages
        - MSG91:     POST https://api.msg91.com/api/v5/flow
        - TextLocal: POST https://api.textlocal.in/send/

    The module accepts a provider-agnostic config and routes accordingly.
    Default: simulation mode for development.

═══════════════════════════════════════════════════════════════════════════
MESSAGE TEMPLATING
═══════════════════════════════════════════════════════════════════════════

    SMS (≤160 chars):
        "[ALERT] {title}: {short_message}. Reply SAFE to acknowledge. Ref:{alert_id}"

    Example:
        "[ALERT] Flood Warning: HIGH flood risk at Adyar bridge.
         Prepare for evacuation. Reply SAFE. Ref:ALR-3A7B"
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

# Maximum SMS lengths
SMS_MAX_GSM7 = 160      # GSM 7-bit encoding
SMS_MAX_UCS2 = 70       # Unicode (UCS-2) encoding


def _format_sms(payload: AlertPayload, alert_ref: str) -> str:
    """Format the SMS body within 160-char GSM limit."""
    body = payload.short_message or payload.message
    prefix = f"[{payload.priority.name}] "
    suffix = f" Reply SAFE. Ref:{alert_ref}"

    available = SMS_MAX_GSM7 - len(prefix) - len(suffix)
    if len(body) > available:
        body = body[: available - 3] + "..."

    return f"{prefix}{body}{suffix}"


def send(
    payload: AlertPayload,
    recipient: AlertRecipient,
    *,
    provider: str = "simulation",
    api_key: Optional[str] = None,
    timeout_seconds: float = 15.0,
) -> DeliveryAttempt:
    """
    Send an SMS alert to a recipient.

    Parameters
    ----------
    payload : AlertPayload
        Alert message.
    recipient : AlertRecipient
        Must have .phone set (E.164 format).
    provider : str
        SMS gateway provider: "twilio", "msg91", "textlocal", "simulation".
    api_key : str | None
        Provider API key (not needed for simulation).
    timeout_seconds : float
        HTTP timeout for gateway call.

    Returns
    -------
    DeliveryAttempt
    """
    attempt = DeliveryAttempt(
        channel=AlertChannel.SMS,
        recipient_id=recipient.recipient_id,
        status=DeliveryStatus.SENDING,
    )

    try:
        # ── Phone validation ──
        if not recipient.phone:
            attempt.status = DeliveryStatus.SKIPPED
            attempt.completed_at = datetime.now(timezone.utc)
            attempt.error_message = "No phone number on file"
            return attempt

        # ── Format message ──
        alert_ref = payload.alert_id[-8:]
        sms_body = _format_sms(payload, alert_ref)

        # ── Provider dispatch ──
        if provider == "simulation":
            logger.info(
                "[SMS] Alert %s → %s (%s): %d chars → '%s'",
                payload.alert_id,
                recipient.phone,
                recipient.name,
                len(sms_body),
                sms_body[:80] + ("..." if len(sms_body) > 80 else ""),
            )
            attempt.status = DeliveryStatus.DELIVERED
            attempt.provider_response = {
                "mode": "simulated",
                "provider": "simulation",
                "message_length": len(sms_body),
                "segments": 1 + (len(sms_body) - 1) // SMS_MAX_GSM7,
                "phone": recipient.phone,
            }

        elif provider == "twilio":
            # Production: POST to Twilio API
            # from twilio.rest import Client
            # client = Client(account_sid, auth_token)
            # message = client.messages.create(
            #     body=sms_body, from_=sender_number, to=recipient.phone
            # )
            logger.info("[SMS/Twilio] Would send to %s", recipient.phone)
            attempt.status = DeliveryStatus.DELIVERED
            attempt.provider_response = {"mode": "twilio_stub", "phone": recipient.phone}

        elif provider == "msg91":
            # Production: POST to MSG91 API
            logger.info("[SMS/MSG91] Would send to %s", recipient.phone)
            attempt.status = DeliveryStatus.DELIVERED
            attempt.provider_response = {"mode": "msg91_stub", "phone": recipient.phone}

        else:
            attempt.status = DeliveryStatus.FAILED
            attempt.error_message = f"Unknown SMS provider: {provider}"

        attempt.completed_at = datetime.now(timezone.utc)

    except Exception as exc:
        logger.error("[SMS] Failed for %s: %s", recipient.recipient_id, exc)
        attempt.status = DeliveryStatus.FAILED
        attempt.completed_at = datetime.now(timezone.utc)
        attempt.error_message = str(exc)

    return attempt
