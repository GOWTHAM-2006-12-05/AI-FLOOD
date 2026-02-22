"""
FastAPI route: Multi-channel alert broadcasting endpoint.

Provides endpoints to:
    POST /api/v1/alerts/broadcast     — trigger a broadcast
    POST /api/v1/alerts/{id}/ack      — acknowledge an alert
    GET  /api/v1/alerts/{id}/status   — get delivery report
    GET  /api/v1/alerts/channels      — list available channels
    GET  /api/v1/alerts/health        — service health
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.app.alerts.models import (
    AckStatus,
    AlertChannel,
    AlertPayload,
    AlertPriority,
    AlertRecipient,
    CHANNELS_BY_PRIORITY,
    DeliveryStatus,
)
from backend.app.alerts.alert_service import (
    broadcast_alert,
    build_payload_from_risk,
    get_ack_status,
    get_alert_report,
    record_acknowledgment,
    store_alert_report,
)

router = APIRouter(prefix="/api/v1/alerts", tags=["alert-broadcasting"])


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class RecipientInput(BaseModel):
    """A single alert recipient."""
    recipient_id: str = Field(..., description="Unique identifier", examples=["R001"])
    name: str = Field(..., description="Display name", examples=["Arun"])
    latitude: float = Field(..., ge=-90, le=90, examples=[13.0])
    longitude: float = Field(..., ge=-180, le=180, examples=[80.27])
    phone: Optional[str] = Field(None, examples=["+919876543210"])
    email: Optional[str] = Field(None, examples=["user@example.com"])
    web_push_token: Optional[str] = Field(None)
    low_bandwidth: bool = Field(False, description="Use low-bandwidth channels")


class RiskBasedBroadcastRequest(BaseModel):
    """Trigger an alert broadcast based on risk aggregation results."""
    # Location of the disaster event
    latitude: float = Field(..., ge=-90, le=90, examples=[13.0827])
    longitude: float = Field(..., ge=-180, le=180, examples=[80.2707])
    radius_km: float = Field(50.0, ge=1.0, le=500.0, examples=[50.0])

    # Risk data (from risk aggregator output)
    overall_risk_score: float = Field(
        ..., ge=0, le=100, examples=[72.5],
        description="Overall risk score (0–100%)",
    )
    overall_risk_level: str = Field(
        ..., examples=["severe"],
        description="Risk level: safe/watch/warning/severe",
    )
    dominant_hazard: str = Field(
        "unknown", examples=["flood"],
        description="Primary hazard type",
    )
    alert_action: str = Field(
        "monitor", examples=["evacuate"],
        description="Recommended action",
    )
    alert_reasons: List[str] = Field(
        default_factory=list,
        examples=[["Flood probability 0.80", "Earthquake M5.5"]],
    )
    active_hazard_count: int = Field(0, ge=0, examples=[2])

    # Recipients
    recipients: List[RecipientInput] = Field(
        ..., min_length=1,
        description="List of recipients to target",
    )

    # Options
    apply_geofence: bool = Field(
        True,
        description="If True, filter recipients by geo-fence distance",
    )


class DirectBroadcastRequest(BaseModel):
    """Trigger a broadcast with explicit payload control."""
    priority: str = Field(
        ..., examples=["URGENT"],
        description="INFORMATIONAL / ADVISORY / URGENT / CRITICAL",
    )
    hazard_type: str = Field("unknown", examples=["earthquake"])
    title: str = Field(..., examples=["Earthquake Warning"])
    message: str = Field(..., examples=["A magnitude 5.5 earthquake detected."])
    short_message: str = Field("", examples=["EQ M5.5 near Chennai. Take cover."])
    minimal_message: str = Field("", examples=["EQ!SEV CHENNAI"])
    latitude: float = Field(0.0, ge=-90, le=90)
    longitude: float = Field(0.0, ge=-180, le=180)
    radius_km: float = Field(50.0, ge=1.0, le=500.0)
    require_acknowledgment: bool = Field(False)
    recipients: List[RecipientInput] = Field(..., min_length=1)
    apply_geofence: bool = Field(True)


class AcknowledgeRequest(BaseModel):
    """Request body for acknowledging an alert."""
    recipient_id: str = Field(..., examples=["R001"])
    channel: str = Field(
        "web_notification", examples=["sms"],
        description="Channel via which ack was received",
    )


class BroadcastResponse(BaseModel):
    """Broadcast result summary."""
    alert_id: str
    priority: str
    hazard_type: str
    total_recipients: int
    recipients_reached: int
    recipients_failed: int
    reach_rate: str
    total_attempts: int
    broadcast_started_at: Optional[str]
    broadcast_completed_at: Optional[str]
    delivery_records: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _parse_priority(priority_str: str) -> AlertPriority:
    """Parse a priority string to AlertPriority enum."""
    mapping = {
        "informational": AlertPriority.INFORMATIONAL,
        "advisory": AlertPriority.ADVISORY,
        "urgent": AlertPriority.URGENT,
        "critical": AlertPriority.CRITICAL,
    }
    result = mapping.get(priority_str.lower())
    if result is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid priority '{priority_str}'. "
                   f"Must be one of: {list(mapping.keys())}",
        )
    return result


def _parse_channel(channel_str: str) -> AlertChannel:
    """Parse a channel string to AlertChannel enum."""
    try:
        return AlertChannel(channel_str)
    except ValueError:
        valid = [c.value for c in AlertChannel]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid channel '{channel_str}'. Must be one of: {valid}",
        )


def _to_recipient(r: RecipientInput) -> AlertRecipient:
    """Convert Pydantic model to dataclass."""
    return AlertRecipient(
        recipient_id=r.recipient_id,
        name=r.name,
        latitude=r.latitude,
        longitude=r.longitude,
        phone=r.phone,
        email=r.email,
        web_push_token=r.web_push_token,
        low_bandwidth=r.low_bandwidth,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/broadcast/risk",
    response_model=BroadcastResponse,
    summary="Broadcast alert from risk aggregation data",
    description=(
        "Takes risk aggregator output + recipients, builds an alert payload, "
        "applies geo-fence filtering, and dispatches via appropriate channels."
    ),
)
async def broadcast_from_risk(request: RiskBasedBroadcastRequest):
    """Broadcast alert based on risk aggregation output."""
    risk_data = {
        "overall_risk_score": request.overall_risk_score,
        "overall_risk_level": request.overall_risk_level,
        "dominant_hazard": request.dominant_hazard,
        "alert_action": request.alert_action,
        "alert_reasons": request.alert_reasons,
        "active_hazard_count": request.active_hazard_count,
    }

    payload = build_payload_from_risk(
        risk_data,
        latitude=request.latitude,
        longitude=request.longitude,
        radius_km=request.radius_km,
    )

    recipients = [_to_recipient(r) for r in request.recipients]

    report = broadcast_alert(
        payload, recipients,
        apply_geofence=request.apply_geofence,
    )

    store_alert_report(report)

    return BroadcastResponse(
        alert_id=report.alert_id,
        priority=report.payload.priority.name,
        hazard_type=report.payload.hazard_type,
        total_recipients=report.total_recipients,
        recipients_reached=report.recipients_reached,
        recipients_failed=report.recipients_failed,
        reach_rate=f"{report.reach_rate:.1%}",
        total_attempts=report.total_attempts,
        broadcast_started_at=(
            report.broadcast_started_at.isoformat()
            if report.broadcast_started_at else None
        ),
        broadcast_completed_at=(
            report.broadcast_completed_at.isoformat()
            if report.broadcast_completed_at else None
        ),
        delivery_records=[r.to_dict() for r in report.delivery_records],
    )


@router.post(
    "/broadcast/direct",
    response_model=BroadcastResponse,
    summary="Broadcast alert with explicit payload",
    description="Full control over alert payload — specify priority, messages, etc.",
)
async def broadcast_direct(request: DirectBroadcastRequest):
    """Broadcast alert with direct payload specification."""
    priority = _parse_priority(request.priority)

    payload = AlertPayload(
        priority=priority,
        hazard_type=request.hazard_type,
        title=request.title,
        message=request.message,
        short_message=request.short_message or request.message[:155],
        minimal_message=request.minimal_message or request.message[:65],
        latitude=request.latitude,
        longitude=request.longitude,
        radius_km=request.radius_km,
        require_acknowledgment=request.require_acknowledgment,
    )

    recipients = [_to_recipient(r) for r in request.recipients]

    report = broadcast_alert(
        payload, recipients,
        apply_geofence=request.apply_geofence,
    )

    store_alert_report(report)

    return BroadcastResponse(
        alert_id=report.alert_id,
        priority=report.payload.priority.name,
        hazard_type=report.payload.hazard_type,
        total_recipients=report.total_recipients,
        recipients_reached=report.recipients_reached,
        recipients_failed=report.recipients_failed,
        reach_rate=f"{report.reach_rate:.1%}",
        total_attempts=report.total_attempts,
        broadcast_started_at=(
            report.broadcast_started_at.isoformat()
            if report.broadcast_started_at else None
        ),
        broadcast_completed_at=(
            report.broadcast_completed_at.isoformat()
            if report.broadcast_completed_at else None
        ),
        delivery_records=[r.to_dict() for r in report.delivery_records],
    )


@router.post(
    "/{alert_id}/acknowledge",
    summary="Acknowledge an alert",
    description="Record that a recipient acknowledged receipt of an alert.",
)
async def acknowledge_alert(alert_id: str, request: AcknowledgeRequest):
    """Record alert acknowledgment from a recipient."""
    channel = _parse_channel(request.channel)
    success = record_acknowledgment(alert_id, request.recipient_id, channel)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Alert '{alert_id}' or recipient '{request.recipient_id}' "
                   f"not found.",
        )

    return {
        "alert_id": alert_id,
        "recipient_id": request.recipient_id,
        "status": "acknowledged",
        "channel": channel.value,
    }


@router.get(
    "/{alert_id}/status",
    summary="Get alert delivery status",
    description="Retrieve the full delivery report for a broadcast alert.",
)
async def get_alert_status(alert_id: str):
    """Get delivery report for an alert."""
    report = get_alert_report(alert_id)
    if report:
        return report.to_dict()

    # Try ack store for partial info
    ack_data = get_ack_status(alert_id)
    if ack_data.get("total_recipients", 0) > 0:
        return ack_data

    raise HTTPException(
        status_code=404,
        detail=f"Alert '{alert_id}' not found.",
    )


@router.get(
    "/channels",
    summary="List available channels",
    description="Show channel configuration and priority mapping.",
)
async def list_channels():
    """List all channels and their priority-based activation."""
    return {
        "channels": [
            {
                "name": channel.value,
                "activated_at_priorities": [
                    priority.name
                    for priority, channels in CHANNELS_BY_PRIORITY.items()
                    if channel in channels
                ],
            }
            for channel in AlertChannel
        ],
        "priority_levels": [
            {"name": p.name, "value": int(p)} for p in AlertPriority
        ],
        "priority_channel_map": {
            p.name: [c.value for c in ch]
            for p, ch in CHANNELS_BY_PRIORITY.items()
        },
    }


@router.get(
    "/health",
    summary="Alert service health check",
)
async def health():
    """Check alert broadcasting service health."""
    return {
        "status": "healthy",
        "service": "alert-broadcasting",
        "channels_available": len(AlertChannel),
        "priority_levels": len(AlertPriority),
    }
