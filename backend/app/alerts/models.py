"""
models.py — Shared data structures for the alert broadcasting system.

Defines:
    • AlertPriority — escalation levels
    • AlertChannel  — delivery channel enum
    • DeliveryStatus — per-recipient delivery tracking
    • AlertRecipient — a target user with location + contact info
    • AlertPayload   — the broadcast message
    • DeliveryAttempt — single send attempt record
    • AlertReport     — final delivery summary

═══════════════════════════════════════════════════════════════════════════
ALERT ESCALATION LEVELS
═══════════════════════════════════════════════════════════════════════════

Level mapping from the risk aggregator:

    Risk Level    Alert Priority    Channels Activated
    ──────────    ──────────────    ──────────────────────────────────
    Safe          INFORMATIONAL     Web only (passive dashboard update)
    Watch         ADVISORY          Web + Email
    Warning       URGENT            Web + Email + SMS
    Severe        CRITICAL          Web + Email + SMS + Siren + SMS-Fallback

Higher priorities activate more channels and demand acknowledgment.

═══════════════════════════════════════════════════════════════════════════
CHANNEL SELECTION LOGIC
═══════════════════════════════════════════════════════════════════════════

Channels are selected by priority AND connectivity context:

    1. Web Notification  — always attempted (cheapest, fastest)
    2. Email             — Advisory+ (good for detailed info, non-urgent)
    3. SMS               — Urgent+   (high delivery rate, 160-char limit)
    4. Siren API         — Critical  (physical sirens in infrastructure)
    5. SMS Fallback      — Critical  (bare-minimum 70-char GSM encoding
                           for areas with 2G-only / intermittent coverage)

If the primary SMS channel fails, the system automatically falls back
to the low-bandwidth SMS variant which uses:
    - GSM 7-bit encoding (no Unicode → 160 chars/segment)
    - Abbreviated message templates
    - No URL links (bandwidth saving)
    - Delivery receipt polling instead of webhooks
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class AlertPriority(IntEnum):
    """
    Alert escalation levels — integer ordering enables comparison.

    Higher value = more severe = more channels activated.
    """
    INFORMATIONAL = 1   # Safe — web-only passive update
    ADVISORY      = 2   # Watch — web + email
    URGENT        = 3   # Warning — web + email + SMS
    CRITICAL      = 4   # Severe — all channels + siren + fallback


class AlertChannel(str, Enum):
    """Available broadcasting channels."""
    WEB_NOTIFICATION = "web_notification"
    EMAIL            = "email"
    SMS              = "sms"
    SIREN_API        = "siren_api"
    SMS_FALLBACK     = "sms_fallback"  # low-bandwidth GSM 7-bit


class DeliveryStatus(str, Enum):
    """Delivery state machine per recipient per channel."""
    PENDING      = "pending"        # queued, not yet sent
    SENDING      = "sending"        # send in progress
    DELIVERED    = "delivered"       # confirmed delivery
    FAILED       = "failed"         # all retries exhausted
    ACKNOWLEDGED = "acknowledged"   # recipient confirmed receipt
    SKIPPED      = "skipped"        # channel not applicable


class AckStatus(str, Enum):
    """Acknowledgment tracking states."""
    NOT_REQUIRED   = "not_required"    # informational — no ack needed
    AWAITING       = "awaiting"        # ack requested, not yet received
    ACKNOWLEDGED   = "acknowledged"    # recipient confirmed
    TIMED_OUT      = "timed_out"       # ack deadline passed
    ESCALATED      = "escalated"       # no ack → escalated to next channel


# ═══════════════════════════════════════════════════════════════════════════
# Channel-Priority Mapping
# ═══════════════════════════════════════════════════════════════════════════

# Which channels are activated at each priority level
CHANNELS_BY_PRIORITY: Dict[AlertPriority, List[AlertChannel]] = {
    AlertPriority.INFORMATIONAL: [
        AlertChannel.WEB_NOTIFICATION,
    ],
    AlertPriority.ADVISORY: [
        AlertChannel.WEB_NOTIFICATION,
        AlertChannel.EMAIL,
    ],
    AlertPriority.URGENT: [
        AlertChannel.WEB_NOTIFICATION,
        AlertChannel.EMAIL,
        AlertChannel.SMS,
    ],
    AlertPriority.CRITICAL: [
        AlertChannel.WEB_NOTIFICATION,
        AlertChannel.EMAIL,
        AlertChannel.SMS,
        AlertChannel.SIREN_API,
        AlertChannel.SMS_FALLBACK,
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════

def _generate_id() -> str:
    return f"ALR-{uuid.uuid4().hex[:12].upper()}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class AlertRecipient:
    """
    A target user/device for alert delivery.

    Attributes
    ----------
    recipient_id : str
        Unique identifier for the recipient.
    name : str
        Display name.
    latitude, longitude : float
        Recipient's current/registered location.
    phone : str | None
        Phone number for SMS channels (E.164 format: +91XXXXXXXXXX).
    email : str | None
        Email address.
    web_push_token : str | None
        Push notification token (FCM / VAPID).
    preferred_language : str
        BCP-47 language code for localised messages.
    low_bandwidth : bool
        If True, prefer SMS fallback over rich channels.
    """
    recipient_id: str
    name: str
    latitude: float
    longitude: float
    phone: Optional[str] = None
    email: Optional[str] = None
    web_push_token: Optional[str] = None
    preferred_language: str = "en"
    low_bandwidth: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recipient_id": self.recipient_id,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "phone": self.phone,
            "email": self.email,
            "low_bandwidth": self.low_bandwidth,
        }


@dataclass
class AlertPayload:
    """
    The alert message to broadcast.

    Carries the disaster context and pre-rendered messages for
    each channel (channels have different length/format constraints).
    """
    alert_id: str = field(default_factory=_generate_id)
    priority: AlertPriority = AlertPriority.INFORMATIONAL
    hazard_type: str = "unknown"
    title: str = ""
    message: str = ""                              # full-length message
    short_message: str = ""                        # SMS (≤160 chars)
    minimal_message: str = ""                      # SMS fallback (≤70 chars)
    source_risk_score: float = 0.0                 # 0–100 from risk aggregator
    source_risk_level: str = "safe"                # from OverallRiskLevel
    latitude: float = 0.0                          # epicentre / event location
    longitude: float = 0.0
    radius_km: float = 50.0                        # geo-fence radius
    created_at: datetime = field(default_factory=_now)
    expires_at: Optional[datetime] = None          # alert validity window
    require_acknowledgment: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "priority": self.priority.name,
            "priority_value": int(self.priority),
            "hazard_type": self.hazard_type,
            "title": self.title,
            "message": self.message,
            "short_message": self.short_message,
            "minimal_message": self.minimal_message,
            "source_risk_score": round(self.source_risk_score, 2),
            "source_risk_level": self.source_risk_level,
            "location": {
                "latitude": self.latitude,
                "longitude": self.longitude,
            },
            "radius_km": self.radius_km,
            "created_at": self.created_at.isoformat(),
            "expires_at": (
                self.expires_at.isoformat() if self.expires_at else None
            ),
            "require_acknowledgment": self.require_acknowledgment,
            "metadata": self.metadata,
        }


@dataclass
class DeliveryAttempt:
    """Record of a single delivery attempt to one recipient via one channel."""
    attempt_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    channel: AlertChannel = AlertChannel.WEB_NOTIFICATION
    recipient_id: str = ""
    status: DeliveryStatus = DeliveryStatus.PENDING
    attempted_at: datetime = field(default_factory=_now)
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    provider_response: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "channel": self.channel.value,
            "recipient_id": self.recipient_id,
            "status": self.status.value,
            "attempted_at": self.attempted_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "retry_count": self.retry_count,
            "error_message": self.error_message,
        }


@dataclass
class RecipientDeliveryRecord:
    """Aggregated delivery status for one recipient across all channels."""
    recipient_id: str
    name: str
    channels_attempted: List[AlertChannel] = field(default_factory=list)
    channels_delivered: List[AlertChannel] = field(default_factory=list)
    channels_failed: List[AlertChannel] = field(default_factory=list)
    acknowledgment: AckStatus = AckStatus.NOT_REQUIRED
    acknowledged_at: Optional[datetime] = None
    acknowledged_via: Optional[AlertChannel] = None
    attempts: List[DeliveryAttempt] = field(default_factory=list)

    @property
    def is_reached(self) -> bool:
        """True if at least one channel succeeded."""
        return len(self.channels_delivered) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recipient_id": self.recipient_id,
            "name": self.name,
            "is_reached": self.is_reached,
            "channels_attempted": [c.value for c in self.channels_attempted],
            "channels_delivered": [c.value for c in self.channels_delivered],
            "channels_failed": [c.value for c in self.channels_failed],
            "acknowledgment": self.acknowledgment.value,
            "acknowledged_at": (
                self.acknowledged_at.isoformat()
                if self.acknowledged_at else None
            ),
            "acknowledged_via": (
                self.acknowledged_via.value
                if self.acknowledged_via else None
            ),
            "attempt_count": len(self.attempts),
        }


@dataclass
class AlertReport:
    """Final delivery report for a broadcast alert."""
    alert_id: str
    payload: AlertPayload
    total_recipients: int = 0
    recipients_reached: int = 0
    recipients_failed: int = 0
    recipients_acknowledged: int = 0
    delivery_records: List[RecipientDeliveryRecord] = field(default_factory=list)
    broadcast_started_at: Optional[datetime] = None
    broadcast_completed_at: Optional[datetime] = None
    total_attempts: int = 0

    @property
    def reach_rate(self) -> float:
        if self.total_recipients == 0:
            return 0.0
        return self.recipients_reached / self.total_recipients

    @property
    def ack_rate(self) -> float:
        if self.total_recipients == 0:
            return 0.0
        return self.recipients_acknowledged / self.total_recipients

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "priority": self.payload.priority.name,
            "hazard_type": self.payload.hazard_type,
            "total_recipients": self.total_recipients,
            "recipients_reached": self.recipients_reached,
            "recipients_failed": self.recipients_failed,
            "recipients_acknowledged": self.recipients_acknowledged,
            "reach_rate": f"{self.reach_rate:.1%}",
            "ack_rate": f"{self.ack_rate:.1%}",
            "total_attempts": self.total_attempts,
            "broadcast_started_at": (
                self.broadcast_started_at.isoformat()
                if self.broadcast_started_at else None
            ),
            "broadcast_completed_at": (
                self.broadcast_completed_at.isoformat()
                if self.broadcast_completed_at else None
            ),
            "delivery_records": [r.to_dict() for r in self.delivery_records],
        }
