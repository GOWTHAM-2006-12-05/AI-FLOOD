"""
web_push.py — Web push notification channel.

Delivery mechanism:
    • Uses Web Push Protocol (RFC 8030) via VAPID keys
    • Payload: JSON with title, body, icon, action URL
    • Delivery confirmation via push service receipt

In production, this would use:
    - pywebpush library with VAPID authentication
    - Firebase Cloud Messaging (FCM) as push service
    - Service Worker on the client to display the notification

This module provides a simulation for development / testing that
logs the notification and returns a simulated delivery result.

═══════════════════════════════════════════════════════════════════════════
WHY WEB PUSH IS ALWAYS THE FIRST CHANNEL
═══════════════════════════════════════════════════════════════════════════

    1. Zero marginal cost    — no per-message charge (unlike SMS)
    2. Rich content          — supports icons, images, action buttons
    3. Instant delivery      — sub-second latency via persistent connection
    4. No phone number needed — works with anonymous users
    5. Clickable             — user can tap to open detailed alert page

Limitations:
    - Requires browser/app to have granted notification permission
    - Doesn't work if user is offline (queued by push service, may expire)
    - Desktop browser must be open (mobile works even when app is closed)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from backend.app.alerts.models import (
    AlertPayload,
    AlertRecipient,
    DeliveryAttempt,
    DeliveryStatus,
    AlertChannel,
)

logger = logging.getLogger(__name__)


def send(
    payload: AlertPayload,
    recipient: AlertRecipient,
    *,
    timeout_seconds: float = 10.0,
) -> DeliveryAttempt:
    """
    Send a web push notification to a recipient.

    Parameters
    ----------
    payload : AlertPayload
        The alert message to deliver.
    recipient : AlertRecipient
        Target user (must have web_push_token for real delivery).
    timeout_seconds : float
        HTTP timeout for push service call.

    Returns
    -------
    DeliveryAttempt
        Result of the send operation.
    """
    attempt = DeliveryAttempt(
        channel=AlertChannel.WEB_NOTIFICATION,
        recipient_id=recipient.recipient_id,
        status=DeliveryStatus.SENDING,
    )

    try:
        # ── Token validation ──
        if not recipient.web_push_token:
            logger.warning(
                "No web push token for recipient %s — simulating delivery",
                recipient.recipient_id,
            )
            # In dev mode, simulate success for testing
            # In production, this would be SKIPPED
            attempt.status = DeliveryStatus.DELIVERED
            attempt.completed_at = datetime.now(timezone.utc)
            attempt.provider_response = {
                "mode": "simulated",
                "reason": "no_push_token — dev simulation",
            }
            return attempt

        # ── Build push payload ──
        push_data = {
            "notification": {
                "title": payload.title,
                "body": payload.message,
                "icon": "/icons/disaster-alert.png",
                "badge": "/icons/badge.png",
                "tag": payload.alert_id,
                "data": {
                    "alert_id": payload.alert_id,
                    "priority": payload.priority.name,
                    "hazard_type": payload.hazard_type,
                    "risk_score": payload.source_risk_score,
                    "url": f"/alerts/{payload.alert_id}",
                },
                "actions": [
                    {"action": "acknowledge", "title": "I'm Safe"},
                    {"action": "details", "title": "View Details"},
                ],
                "requireInteraction": payload.priority.value >= 3,
                "vibrate": [200, 100, 200] if payload.priority.value >= 3 else [100],
            },
        }

        # ── Simulated push delivery ──
        # In production: webpush.send(subscription_info, data, vapid_claims)
        logger.info(
            "[WEB_PUSH] Alert %s → %s (%s): %s",
            payload.alert_id,
            recipient.recipient_id,
            recipient.name,
            payload.title,
        )

        attempt.status = DeliveryStatus.DELIVERED
        attempt.completed_at = datetime.now(timezone.utc)
        attempt.provider_response = {
            "mode": "simulated",
            "push_payload_size": len(str(push_data)),
            "token_prefix": recipient.web_push_token[:12] + "...",
        }

    except Exception as exc:
        logger.error(
            "[WEB_PUSH] Failed for %s: %s", recipient.recipient_id, exc
        )
        attempt.status = DeliveryStatus.FAILED
        attempt.completed_at = datetime.now(timezone.utc)
        attempt.error_message = str(exc)

    return attempt
