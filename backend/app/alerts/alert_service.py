"""
alert_service.py — Core alert broadcasting orchestration engine.

This is the central coordinator that:
    1. Receives an alert payload from the risk aggregator
    2. Applies geo-fence filtering to identify targeted recipients
    3. Selects channels based on alert priority and recipient capabilities
    4. Dispatches to each channel with retry logic
    5. Tracks delivery status and acknowledgments
    6. Handles fallback escalation (SMS fail → SMS fallback)
    7. Produces a delivery report

═══════════════════════════════════════════════════════════════════════════
ORCHESTRATION FLOW
═══════════════════════════════════════════════════════════════════════════

    ┌─────────────────────┐
    │  Risk Aggregator    │
    │  triggers alert     │
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  1. Build Payload   │  Map risk level → AlertPriority
    │     from risk data  │  Generate message variants (full / SMS / fallback)
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  2. Geo-fence       │  Filter recipients by distance from event
    │     Targeting       │  Apply priority-based radius expansion
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  3. Channel         │  Select channels from CHANNELS_BY_PRIORITY
    │     Selection       │  Honour per-recipient capability (has phone? email?)
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  4. Dispatch        │  Send via each channel in priority order:
    │     with Retry      │    web → email → SMS → siren → SMS fallback
    │                     │  Retry failed channels (exponential backoff)
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  5. Acknowledgment  │  Track ack status per recipient
    │     Tracking        │  Escalate channel if no ack within timeout
    └─────────┬───────────┘
              │
              ▼
    ┌─────────────────────┐
    │  6. Delivery        │  Compile per-recipient and aggregate report
    │     Report          │  Log for audit trail
    └─────────────────────┘

═══════════════════════════════════════════════════════════════════════════
RETRY & REDUNDANCY STRATEGY
═══════════════════════════════════════════════════════════════════════════

Per-channel retry configuration:

    Channel          Max Retries    Backoff Base    Backoff Type
    ──────────       ───────────    ────────────    ──────────────
    Web Push         2              1.0s            Exponential
    Email            3              2.0s            Exponential
    SMS              3              5.0s            Exponential
    Siren            2              3.0s            Exponential
    SMS Fallback     5              10.0s           Linear (2G latency)

Backoff formula (exponential):
    delay = base × 2^(attempt - 1) + jitter

    Example for SMS (base=5s):
        Attempt 1: 5s, Attempt 2: 10s, Attempt 3: 20s

Redundancy policy:
    • If primary SMS fails after all retries → auto-trigger SMS Fallback
    • If ALL channels for a recipient fail → log as UNREACHABLE
    • Each channel operates independently (failure in one doesn't block others)

═══════════════════════════════════════════════════════════════════════════
LOW CONNECTIVITY HANDLING
═══════════════════════════════════════════════════════════════════════════

Recipients flagged with low_bandwidth=True receive:
    1. SMS Fallback INSTEAD OF standard SMS (not in addition to)
    2. Web Push is still attempted (may be queued by push service)
    3. Email is downgraded to plain-text only
    4. Retry intervals are extended (×2 of normal)
    5. Delivery confirmation uses polling instead of webhooks

Detection heuristics for low-bandwidth areas:
    - Recipient manually sets low_bandwidth=True
    - Previous delivery failure rate > 50% for SMS
    - Geographic mapping of 2G-only coverage areas
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.app.alerts.models import (
    AckStatus,
    AlertChannel,
    AlertPayload,
    AlertPriority,
    AlertRecipient,
    AlertReport,
    CHANNELS_BY_PRIORITY,
    DeliveryAttempt,
    DeliveryStatus,
    RecipientDeliveryRecord,
)
from backend.app.alerts.geo_fence import (
    compute_effective_radius,
    filter_recipients_by_geofence,
)
from backend.app.alerts.channels import (
    web_push,
    email_alert,
    sms_gateway,
    siren_api,
    sms_fallback,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Retry Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RetryConfig:
    """Per-channel retry parameters."""
    max_retries: int
    backoff_base_seconds: float
    backoff_type: str  # "exponential" or "linear"
    low_bandwidth_multiplier: float = 2.0


RETRY_CONFIGS: Dict[AlertChannel, RetryConfig] = {
    AlertChannel.WEB_NOTIFICATION: RetryConfig(2, 1.0, "exponential"),
    AlertChannel.EMAIL:            RetryConfig(3, 2.0, "exponential"),
    AlertChannel.SMS:              RetryConfig(3, 5.0, "exponential"),
    AlertChannel.SIREN_API:        RetryConfig(2, 3.0, "exponential"),
    AlertChannel.SMS_FALLBACK:     RetryConfig(5, 10.0, "linear"),
}


def _compute_backoff(config: RetryConfig, attempt: int, low_bw: bool = False) -> float:
    """
    Compute delay before next retry.

    Parameters
    ----------
    config : RetryConfig
    attempt : int
        Current attempt number (1-based).
    low_bw : bool
        If True, multiply delay by low_bandwidth_multiplier.

    Returns
    -------
    float
        Delay in seconds.
    """
    if config.backoff_type == "exponential":
        delay = config.backoff_base_seconds * (2 ** (attempt - 1))
    else:  # linear
        delay = config.backoff_base_seconds * attempt

    if low_bw:
        delay *= config.low_bandwidth_multiplier

    return delay


# ═══════════════════════════════════════════════════════════════════════════
# Channel Dispatcher Registry
# ═══════════════════════════════════════════════════════════════════════════

# Maps each channel to its send function
_CHANNEL_DISPATCHERS: Dict[AlertChannel, Callable] = {
    AlertChannel.WEB_NOTIFICATION: web_push.send,
    AlertChannel.EMAIL:            email_alert.send,
    AlertChannel.SMS:              sms_gateway.send,
    AlertChannel.SIREN_API:        siren_api.send,
    AlertChannel.SMS_FALLBACK:     sms_fallback.send,
}


def _can_use_channel(channel: AlertChannel, recipient: AlertRecipient) -> bool:
    """
    Check if a recipient has the capability for a given channel.

    Returns False if the recipient lacks the required contact info.
    """
    if channel == AlertChannel.WEB_NOTIFICATION:
        return True  # always attempt (simulated if no token)
    elif channel == AlertChannel.EMAIL:
        return recipient.email is not None
    elif channel == AlertChannel.SMS:
        return recipient.phone is not None and not recipient.low_bandwidth
    elif channel == AlertChannel.SIREN_API:
        return True  # zone-based, not per-recipient
    elif channel == AlertChannel.SMS_FALLBACK:
        return recipient.phone is not None
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Single-Channel Delivery with Retry
# ═══════════════════════════════════════════════════════════════════════════

def _deliver_via_channel(
    channel: AlertChannel,
    payload: AlertPayload,
    recipient: AlertRecipient,
) -> DeliveryAttempt:
    """
    Attempt delivery via a single channel with retries.

    Parameters
    ----------
    channel : AlertChannel
    payload : AlertPayload
    recipient : AlertRecipient

    Returns
    -------
    DeliveryAttempt
        Final attempt after retries.
    """
    dispatcher = _CHANNEL_DISPATCHERS.get(channel)
    if dispatcher is None:
        return DeliveryAttempt(
            channel=channel,
            recipient_id=recipient.recipient_id,
            status=DeliveryStatus.FAILED,
            error_message=f"No dispatcher for channel: {channel.value}",
        )

    config = RETRY_CONFIGS.get(channel, RetryConfig(1, 1.0, "exponential"))
    low_bw = recipient.low_bandwidth

    last_attempt: Optional[DeliveryAttempt] = None

    for attempt_num in range(1, config.max_retries + 2):  # +1 for initial + retries
        result = dispatcher(payload, recipient)
        result.retry_count = attempt_num - 1
        last_attempt = result

        if result.status in (DeliveryStatus.DELIVERED, DeliveryStatus.SKIPPED):
            return result

        # Failed — retry if attempts remain
        if attempt_num <= config.max_retries:
            delay = _compute_backoff(config, attempt_num, low_bw)
            logger.info(
                "Retry %d/%d for %s via %s in %.1fs",
                attempt_num, config.max_retries,
                recipient.recipient_id, channel.value, delay,
            )
            time.sleep(delay)

    # All retries exhausted
    if last_attempt:
        last_attempt.status = DeliveryStatus.FAILED
    return last_attempt or DeliveryAttempt(
        channel=channel,
        recipient_id=recipient.recipient_id,
        status=DeliveryStatus.FAILED,
        error_message="All retries exhausted",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Channel Delivery for One Recipient
# ═══════════════════════════════════════════════════════════════════════════

def _deliver_to_recipient(
    payload: AlertPayload,
    recipient: AlertRecipient,
    channels: List[AlertChannel],
) -> RecipientDeliveryRecord:
    """
    Deliver alert to one recipient via all applicable channels.

    Channels are attempted in order. If SMS fails, SMS fallback
    is automatically triggered (redundancy escalation).

    Parameters
    ----------
    payload : AlertPayload
    recipient : AlertRecipient
    channels : list of AlertChannel

    Returns
    -------
    RecipientDeliveryRecord
    """
    record = RecipientDeliveryRecord(
        recipient_id=recipient.recipient_id,
        name=recipient.name,
    )

    # Determine ack requirement
    if payload.require_acknowledgment and payload.priority.value >= 3:
        record.acknowledgment = AckStatus.AWAITING

    sms_failed = False

    for channel in channels:
        # Skip channels the recipient can't receive
        if not _can_use_channel(channel, recipient):
            logger.debug(
                "Skipping %s for %s (not capable)",
                channel.value, recipient.recipient_id,
            )
            continue

        # For low-bandwidth recipients, replace SMS with SMS Fallback
        if channel == AlertChannel.SMS and recipient.low_bandwidth:
            logger.info(
                "Redirecting %s from SMS to SMS_FALLBACK (low_bandwidth=True)",
                recipient.recipient_id,
            )
            continue  # SMS_FALLBACK will be handled later in the channel list

        record.channels_attempted.append(channel)
        attempt = _deliver_via_channel(channel, payload, recipient)
        record.attempts.append(attempt)

        if attempt.status == DeliveryStatus.DELIVERED:
            record.channels_delivered.append(channel)
        elif attempt.status == DeliveryStatus.FAILED:
            record.channels_failed.append(channel)
            if channel == AlertChannel.SMS:
                sms_failed = True

    # ── Redundancy: SMS failed → trigger SMS Fallback ──
    if sms_failed and AlertChannel.SMS_FALLBACK not in record.channels_attempted:
        if _can_use_channel(AlertChannel.SMS_FALLBACK, recipient):
            logger.info(
                "SMS failed for %s — falling back to SMS_FALLBACK",
                recipient.recipient_id,
            )
            record.channels_attempted.append(AlertChannel.SMS_FALLBACK)
            fallback_attempt = _deliver_via_channel(
                AlertChannel.SMS_FALLBACK, payload, recipient
            )
            record.attempts.append(fallback_attempt)

            if fallback_attempt.status == DeliveryStatus.DELIVERED:
                record.channels_delivered.append(AlertChannel.SMS_FALLBACK)
            else:
                record.channels_failed.append(AlertChannel.SMS_FALLBACK)

    return record


# ═══════════════════════════════════════════════════════════════════════════
# Acknowledgment Management
# ═══════════════════════════════════════════════════════════════════════════

# In-memory ack store (production: Redis or database)
_ack_store: Dict[str, Dict[str, RecipientDeliveryRecord]] = {}


def record_acknowledgment(
    alert_id: str,
    recipient_id: str,
    channel: AlertChannel,
) -> bool:
    """
    Record that a recipient acknowledged an alert.

    Parameters
    ----------
    alert_id : str
    recipient_id : str
    channel : AlertChannel
        Channel via which the ack was received.

    Returns
    -------
    bool
        True if the ack was recorded; False if alert/recipient not found.
    """
    alert_records = _ack_store.get(alert_id)
    if not alert_records:
        return False

    record = alert_records.get(recipient_id)
    if not record:
        return False

    record.acknowledgment = AckStatus.ACKNOWLEDGED
    record.acknowledged_at = datetime.now(timezone.utc)
    record.acknowledged_via = channel

    logger.info(
        "Ack received: alert=%s recipient=%s via=%s",
        alert_id, recipient_id, channel.value,
    )
    return True


def get_ack_status(alert_id: str) -> Dict[str, Any]:
    """
    Get acknowledgment summary for an alert.

    Returns
    -------
    dict
        Summary of ack statuses for all recipients.
    """
    records = _ack_store.get(alert_id, {})
    total = len(records)
    acked = sum(
        1 for r in records.values()
        if r.acknowledgment == AckStatus.ACKNOWLEDGED
    )
    awaiting = sum(
        1 for r in records.values()
        if r.acknowledgment == AckStatus.AWAITING
    )

    return {
        "alert_id": alert_id,
        "total_recipients": total,
        "acknowledged": acked,
        "awaiting": awaiting,
        "ack_rate": f"{(acked / total * 100) if total else 0:.1f}%",
        "recipients": {
            rid: {
                "status": r.acknowledgment.value,
                "acknowledged_at": (
                    r.acknowledged_at.isoformat() if r.acknowledged_at else None
                ),
                "via": r.acknowledged_via.value if r.acknowledged_via else None,
            }
            for rid, r in records.items()
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Payload Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_payload_from_risk(
    risk_data: Dict[str, Any],
    *,
    latitude: float = 0.0,
    longitude: float = 0.0,
    radius_km: float = 50.0,
) -> AlertPayload:
    """
    Convert a risk aggregation result into an AlertPayload.

    Maps:
        overall_risk_level → AlertPriority
        risk_score + hazard info → message variants

    Parameters
    ----------
    risk_data : dict
        Output from risk_aggregator.aggregate_risk().to_dict()
    latitude, longitude : float
        Event / user location.
    radius_km : float
        Alert zone radius.

    Returns
    -------
    AlertPayload
    """
    level = risk_data.get("overall_risk_level", "safe")
    score = risk_data.get("overall_risk_score", 0.0)
    dominant = risk_data.get("dominant_hazard", "unknown")
    reasons = risk_data.get("alert_reasons", [])

    # Map level → priority
    priority_map = {
        "safe": AlertPriority.INFORMATIONAL,
        "watch": AlertPriority.ADVISORY,
        "warning": AlertPriority.URGENT,
        "severe": AlertPriority.CRITICAL,
    }
    priority = priority_map.get(level, AlertPriority.INFORMATIONAL)

    # Generate messages
    title = f"{dominant.upper()} {level.upper()} — Risk {score:.0f}%"
    message = (
        f"Disaster risk level: {level.upper()} ({score:.1f}%). "
        f"Dominant hazard: {dominant}. "
        + (f"Reasons: {'; '.join(reasons)}. " if reasons else "")
        + f"Follow emergency guidance for your area."
    )
    short_message = (
        f"{dominant.upper()} {level.upper()} risk {score:.0f}%. "
        f"Follow emergency instructions."
    )
    minimal_message = f"{dominant.upper()} {level.upper()} {score:.0f}%"

    # Require ack for URGENT and CRITICAL
    require_ack = priority.value >= AlertPriority.URGENT.value

    return AlertPayload(
        priority=priority,
        hazard_type=dominant,
        title=title,
        message=message,
        short_message=short_message,
        minimal_message=minimal_message,
        source_risk_score=score,
        source_risk_level=level,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
        require_acknowledgment=require_ack,
        metadata={
            "action": risk_data.get("alert_action", "monitor"),
            "active_hazard_count": risk_data.get("active_hazard_count", 0),
            "alert_reasons": reasons,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main Broadcast Function
# ═══════════════════════════════════════════════════════════════════════════

def broadcast_alert(
    payload: AlertPayload,
    recipients: List[AlertRecipient],
    *,
    apply_geofence: bool = True,
) -> AlertReport:
    """
    Broadcast an alert to all targeted recipients via appropriate channels.

    This is the top-level orchestration function.

    Steps:
        1. Geo-fence filter recipients
        2. Determine channels for this priority
        3. Dispatch to each recipient via each channel (with retries)
        4. Compile delivery report
        5. Store ack records for tracking

    Parameters
    ----------
    payload : AlertPayload
        The alert to broadcast.
    recipients : list of AlertRecipient
        All potential recipients (pre-filter or full user list).
    apply_geofence : bool
        If True, filter recipients by geo-fence. If False, send to all.

    Returns
    -------
    AlertReport
        Complete delivery report.
    """
    started = datetime.now(timezone.utc)
    logger.info(
        "Broadcasting alert %s [%s] — %s — to %d potential recipients",
        payload.alert_id, payload.priority.name,
        payload.title, len(recipients),
    )

    # ── Step 1: Geo-fence filtering ──
    if apply_geofence and payload.latitude != 0.0:
        targeted, excluded = filter_recipients_by_geofence(payload, recipients)
    else:
        targeted = recipients
        excluded = []

    if not targeted:
        logger.warning(
            "Alert %s: no recipients within geo-fence (radius=%.1f km)",
            payload.alert_id, payload.radius_km,
        )

    # ── Step 2: Channel selection ──
    channels = CHANNELS_BY_PRIORITY.get(
        payload.priority,
        [AlertChannel.WEB_NOTIFICATION],
    )

    logger.info(
        "Channels for %s: %s",
        payload.priority.name,
        [c.value for c in channels],
    )

    # ── Step 3: Dispatch to each recipient ──
    delivery_records: List[RecipientDeliveryRecord] = []
    total_attempts = 0

    for recipient in targeted:
        record = _deliver_to_recipient(payload, recipient, channels)
        delivery_records.append(record)
        total_attempts += len(record.attempts)

    # ── Step 4: Store ack records ──
    alert_ack_records = {}
    for record in delivery_records:
        alert_ack_records[record.recipient_id] = record
    _ack_store[payload.alert_id] = alert_ack_records

    # ── Step 5: Compile report ──
    completed = datetime.now(timezone.utc)
    reached = sum(1 for r in delivery_records if r.is_reached)
    failed = sum(1 for r in delivery_records if not r.is_reached)
    acked = sum(
        1 for r in delivery_records
        if r.acknowledgment == AckStatus.ACKNOWLEDGED
    )

    report = AlertReport(
        alert_id=payload.alert_id,
        payload=payload,
        total_recipients=len(targeted),
        recipients_reached=reached,
        recipients_failed=failed,
        recipients_acknowledged=acked,
        delivery_records=delivery_records,
        broadcast_started_at=started,
        broadcast_completed_at=completed,
        total_attempts=total_attempts,
    )

    logger.info(
        "Alert %s broadcast complete: %d/%d reached (%.1f%%), %d attempts, %.1fs",
        payload.alert_id,
        reached, len(targeted),
        report.reach_rate * 100,
        total_attempts,
        (completed - started).total_seconds(),
    )

    return report


# ═══════════════════════════════════════════════════════════════════════════
# In-Memory Alert Store (production: database)
# ═══════════════════════════════════════════════════════════════════════════

_alert_reports: Dict[str, AlertReport] = {}


def get_alert_report(alert_id: str) -> Optional[AlertReport]:
    """Retrieve a stored alert report by ID."""
    return _alert_reports.get(alert_id)


def store_alert_report(report: AlertReport) -> None:
    """Store an alert report."""
    _alert_reports[report.alert_id] = report
