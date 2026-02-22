"""
test_alert_service.py — Comprehensive tests for the multi-channel alert
broadcasting system.

Covers:
    • Data models (AlertPayload, AlertRecipient, enums, records)
    • Geo-fence targeting (effective radius, filtering, edge cases)
    • Channel backends (web push, SMS, email, siren, SMS fallback)
    • Alert service orchestration (broadcast, retry, fallback, ack)
    • Payload builder (risk data → AlertPayload)
    • API endpoint schemas and responses
    • Edge cases (no recipients, empty geo-fence, all channels fail)

Run with:
    pytest tests/test_alert_service.py -v
"""

from __future__ import annotations

import math
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

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
    EXPANSION_FACTORS,
)
from backend.app.alerts.channels import (
    web_push,
    email_alert,
    sms_gateway,
    siren_api,
    sms_fallback,
)
from backend.app.alerts.alert_service import (
    RETRY_CONFIGS,
    RetryConfig,
    _can_use_channel,
    _compute_backoff,
    _deliver_to_recipient,
    _deliver_via_channel,
    broadcast_alert,
    build_payload_from_risk,
    get_ack_status,
    record_acknowledgment,
    store_alert_report,
    get_alert_report,
    _ack_store,
    _alert_reports,
)


# ═══════════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════════

# Chennai central (13.0827°N, 80.2707°E)
CHENNAI_LAT = 13.0827
CHENNAI_LON = 80.2707


def _make_recipient(
    rid: str = "R001",
    name: str = "Test User",
    lat: float = CHENNAI_LAT,
    lon: float = CHENNAI_LON,
    phone: str = "+919876543210",
    email: str = "test@example.com",
    web_push_token: str = "token_abc",
    low_bandwidth: bool = False,
) -> AlertRecipient:
    """Create a test recipient."""
    return AlertRecipient(
        recipient_id=rid,
        name=name,
        latitude=lat,
        longitude=lon,
        phone=phone,
        email=email,
        web_push_token=web_push_token,
        low_bandwidth=low_bandwidth,
    )


def _make_payload(
    priority: AlertPriority = AlertPriority.URGENT,
    lat: float = CHENNAI_LAT,
    lon: float = CHENNAI_LON,
    radius_km: float = 50.0,
    require_ack: bool = False,
    hazard_type: str = "flood",
) -> AlertPayload:
    """Create a test payload."""
    return AlertPayload(
        priority=priority,
        hazard_type=hazard_type,
        title=f"Test {priority.name} Alert",
        message="This is a test alert with full details for the given hazard.",
        short_message="Test alert short message for SMS.",
        minimal_message="TST!ALR",
        source_risk_score=72.5,
        source_risk_level="warning",
        latitude=lat,
        longitude=lon,
        radius_km=radius_km,
        require_acknowledgment=require_ack,
    )


@pytest.fixture(autouse=True)
def _clear_stores():
    """Clean in-memory stores before each test."""
    _ack_store.clear()
    _alert_reports.clear()
    yield
    _ack_store.clear()
    _alert_reports.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Section 1: Data Model Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAlertPriority:
    """Test AlertPriority enum."""

    def test_ordering(self):
        assert AlertPriority.INFORMATIONAL < AlertPriority.ADVISORY
        assert AlertPriority.ADVISORY < AlertPriority.URGENT
        assert AlertPriority.URGENT < AlertPriority.CRITICAL

    def test_values(self):
        assert AlertPriority.INFORMATIONAL.value == 1
        assert AlertPriority.ADVISORY.value == 2
        assert AlertPriority.URGENT.value == 3
        assert AlertPriority.CRITICAL.value == 4

    def test_comparison_with_int(self):
        assert AlertPriority.CRITICAL >= 4
        assert AlertPriority.INFORMATIONAL <= 1


class TestAlertChannel:
    """Test AlertChannel enum."""

    def test_all_channels_present(self):
        channels = set(AlertChannel)
        assert AlertChannel.WEB_NOTIFICATION in channels
        assert AlertChannel.EMAIL in channels
        assert AlertChannel.SMS in channels
        assert AlertChannel.SIREN_API in channels
        assert AlertChannel.SMS_FALLBACK in channels
        assert len(channels) == 5

    def test_string_values(self):
        assert AlertChannel.WEB_NOTIFICATION.value == "web_notification"
        assert AlertChannel.SMS_FALLBACK.value == "sms_fallback"


class TestChannelsByPriority:
    """Test the channel-priority mapping."""

    def test_informational_web_only(self):
        assert CHANNELS_BY_PRIORITY[AlertPriority.INFORMATIONAL] == [
            AlertChannel.WEB_NOTIFICATION,
        ]

    def test_advisory_web_and_email(self):
        channels = CHANNELS_BY_PRIORITY[AlertPriority.ADVISORY]
        assert AlertChannel.WEB_NOTIFICATION in channels
        assert AlertChannel.EMAIL in channels
        assert len(channels) == 2

    def test_urgent_three_channels(self):
        channels = CHANNELS_BY_PRIORITY[AlertPriority.URGENT]
        assert AlertChannel.WEB_NOTIFICATION in channels
        assert AlertChannel.EMAIL in channels
        assert AlertChannel.SMS in channels
        assert len(channels) == 3

    def test_critical_all_five_channels(self):
        channels = CHANNELS_BY_PRIORITY[AlertPriority.CRITICAL]
        assert len(channels) == 5
        for ch in AlertChannel:
            assert ch in channels


class TestAlertPayload:
    """Test AlertPayload dataclass."""

    def test_default_id_generated(self):
        p1 = _make_payload()
        p2 = _make_payload()
        assert p1.alert_id != p2.alert_id
        assert p1.alert_id.startswith("ALR-")

    def test_to_dict_has_required_keys(self):
        payload = _make_payload()
        d = payload.to_dict()
        assert "alert_id" in d
        assert "priority" in d
        assert "hazard_type" in d
        assert "location" in d
        assert d["location"]["latitude"] == CHENNAI_LAT
        assert d["source_risk_score"] == 72.5

    def test_created_at_is_utc(self):
        payload = _make_payload()
        assert payload.created_at.tzinfo == timezone.utc


class TestAlertRecipient:
    """Test AlertRecipient dataclass."""

    def test_to_dict(self):
        r = _make_recipient()
        d = r.to_dict()
        assert d["recipient_id"] == "R001"
        assert d["latitude"] == CHENNAI_LAT
        assert d["low_bandwidth"] is False

    def test_optional_fields_none(self):
        r = AlertRecipient(
            recipient_id="R002", name="NoContact",
            latitude=0, longitude=0,
        )
        assert r.phone is None
        assert r.email is None
        assert r.web_push_token is None


class TestRecipientDeliveryRecord:
    """Test RecipientDeliveryRecord."""

    def test_is_reached_when_at_least_one_delivered(self):
        record = RecipientDeliveryRecord(
            recipient_id="R001", name="User",
            channels_delivered=[AlertChannel.WEB_NOTIFICATION],
        )
        assert record.is_reached is True

    def test_not_reached_when_all_failed(self):
        record = RecipientDeliveryRecord(
            recipient_id="R001", name="User",
            channels_failed=[AlertChannel.SMS],
        )
        assert record.is_reached is False

    def test_to_dict(self):
        record = RecipientDeliveryRecord(
            recipient_id="R001", name="User",
            channels_attempted=[AlertChannel.WEB_NOTIFICATION],
            channels_delivered=[AlertChannel.WEB_NOTIFICATION],
            acknowledgment=AckStatus.AWAITING,
        )
        d = record.to_dict()
        assert d["is_reached"] is True
        assert d["acknowledgment"] == "awaiting"


class TestAlertReport:
    """Test AlertReport."""

    def test_reach_rate_computed(self):
        payload = _make_payload()
        report = AlertReport(
            alert_id=payload.alert_id,
            payload=payload,
            total_recipients=10,
            recipients_reached=7,
        )
        assert abs(report.reach_rate - 0.70) < 0.001

    def test_reach_rate_zero_recipients(self):
        payload = _make_payload()
        report = AlertReport(
            alert_id=payload.alert_id,
            payload=payload,
            total_recipients=0,
        )
        assert report.reach_rate == 0.0

    def test_ack_rate_computed(self):
        payload = _make_payload()
        report = AlertReport(
            alert_id=payload.alert_id,
            payload=payload,
            total_recipients=10,
            recipients_acknowledged=5,
        )
        assert abs(report.ack_rate - 0.50) < 0.001


# ═══════════════════════════════════════════════════════════════════════════
# Section 2: Geo-fence Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestExpandEffectiveRadius:
    """Test compute_effective_radius."""

    def test_informational_no_expansion(self):
        payload = _make_payload(priority=AlertPriority.INFORMATIONAL, radius_km=50.0)
        assert compute_effective_radius(payload) == 50.0

    def test_advisory_no_expansion(self):
        payload = _make_payload(priority=AlertPriority.ADVISORY, radius_km=50.0)
        assert compute_effective_radius(payload) == 50.0

    def test_urgent_25pct_expansion(self):
        payload = _make_payload(priority=AlertPriority.URGENT, radius_km=50.0)
        assert abs(compute_effective_radius(payload) - 62.5) < 0.01

    def test_critical_50pct_expansion(self):
        payload = _make_payload(priority=AlertPriority.CRITICAL, radius_km=50.0)
        assert abs(compute_effective_radius(payload) - 75.0) < 0.01


class TestGeofenceFiltering:
    """Test filter_recipients_by_geofence."""

    def test_recipient_within_radius_targeted(self):
        payload = _make_payload(radius_km=100.0)
        recipient = _make_recipient(lat=CHENNAI_LAT + 0.1, lon=CHENNAI_LON + 0.1)
        targeted, excluded = filter_recipients_by_geofence(payload, [recipient])
        assert len(targeted) == 1
        assert len(excluded) == 0

    def test_recipient_outside_radius_excluded(self):
        payload = _make_payload(radius_km=10.0)
        # ~111 km away (1° latitude ≈ 111 km)
        recipient = _make_recipient(lat=CHENNAI_LAT + 1.0, lon=CHENNAI_LON)
        targeted, excluded = filter_recipients_by_geofence(payload, [recipient])
        assert len(targeted) == 0
        assert len(excluded) == 1

    def test_exact_same_location(self):
        payload = _make_payload(radius_km=1.0)
        recipient = _make_recipient(lat=CHENNAI_LAT, lon=CHENNAI_LON)
        targeted, excluded = filter_recipients_by_geofence(payload, [recipient])
        assert len(targeted) == 1

    def test_multiple_recipients_mixed(self):
        payload = _make_payload(radius_km=20.0)
        r_near = _make_recipient(rid="R_NEAR", lat=CHENNAI_LAT + 0.05, lon=CHENNAI_LON)
        r_far = _make_recipient(rid="R_FAR", lat=CHENNAI_LAT + 2.0, lon=CHENNAI_LON)
        targeted, excluded = filter_recipients_by_geofence(payload, [r_near, r_far])
        assert len(targeted) == 1
        assert targeted[0].recipient_id == "R_NEAR"

    def test_critical_expansion_reaches_further(self):
        """CRITICAL priority expands radius by 50%, targeting more recipients."""
        # Place recipient at ~60 km (just outside 50 km base radius)
        r = _make_recipient(lat=CHENNAI_LAT + 0.55, lon=CHENNAI_LON)

        # Urgent expands 50 km → 62.5 km
        p_urgent = _make_payload(priority=AlertPriority.URGENT, radius_km=50.0)
        t_urgent, _ = filter_recipients_by_geofence(p_urgent, [r])

        # Critical expands 50 km → 75 km
        p_critical = _make_payload(priority=AlertPriority.CRITICAL, radius_km=50.0)
        t_critical, _ = filter_recipients_by_geofence(p_critical, [r])

        assert len(t_critical) >= len(t_urgent)

    def test_empty_recipient_list(self):
        payload = _make_payload()
        targeted, excluded = filter_recipients_by_geofence(payload, [])
        assert len(targeted) == 0
        assert len(excluded) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Section 3: Channel Backend Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestWebPushChannel:
    """Test web_push.send."""

    def test_sends_successfully_with_token(self):
        payload = _make_payload()
        recipient = _make_recipient(web_push_token="valid_token")
        result = web_push.send(payload, recipient)
        assert result.channel == AlertChannel.WEB_NOTIFICATION
        assert result.status == DeliveryStatus.DELIVERED
        assert result.recipient_id == "R001"

    def test_sends_simulated_without_token(self):
        payload = _make_payload()
        recipient = _make_recipient(web_push_token=None)
        result = web_push.send(payload, recipient)
        # Should still succeed in dev mode (simulated)
        assert result.status == DeliveryStatus.DELIVERED
        assert result.provider_response.get("mode") == "simulated"


class TestSmsGatewayChannel:
    """Test sms_gateway.send."""

    def test_sends_successfully_with_phone(self):
        payload = _make_payload()
        recipient = _make_recipient(phone="+919876543210")
        result = sms_gateway.send(payload, recipient)
        assert result.channel == AlertChannel.SMS
        assert result.status == DeliveryStatus.DELIVERED

    def test_skips_without_phone(self):
        payload = _make_payload()
        recipient = _make_recipient(phone=None)
        result = sms_gateway.send(payload, recipient)
        assert result.status == DeliveryStatus.SKIPPED

    def test_message_within_160_chars(self):
        """Verify formatted SMS fits in GSM 7-bit limit."""
        payload = _make_payload()
        ref = payload.alert_id[-8:]
        body = sms_gateway._format_sms(payload, ref)
        assert len(body) <= 160


class TestEmailAlertChannel:
    """Test email_alert.send."""

    def test_sends_successfully_with_email(self):
        payload = _make_payload()
        recipient = _make_recipient(email="user@test.com")
        result = email_alert.send(payload, recipient)
        assert result.channel == AlertChannel.EMAIL
        assert result.status == DeliveryStatus.DELIVERED

    def test_skips_without_email(self):
        payload = _make_payload()
        recipient = _make_recipient(email=None)
        result = email_alert.send(payload, recipient)
        assert result.status == DeliveryStatus.SKIPPED


class TestSirenApiChannel:
    """Test siren_api.send."""

    def test_activates_for_critical_priority(self):
        payload = _make_payload(priority=AlertPriority.CRITICAL)
        recipient = _make_recipient()
        result = siren_api.send(payload, recipient)
        assert result.channel == AlertChannel.SIREN_API
        assert result.status == DeliveryStatus.DELIVERED

    def test_skipped_for_non_critical(self):
        payload = _make_payload(priority=AlertPriority.URGENT)
        recipient = _make_recipient()
        result = siren_api.send(payload, recipient)
        assert result.status == DeliveryStatus.SKIPPED


class TestSmsFallbackChannel:
    """Test sms_fallback.send."""

    def test_sends_with_phone(self):
        payload = _make_payload()
        recipient = _make_recipient(phone="+919876543210")
        result = sms_fallback.send(payload, recipient)
        assert result.channel == AlertChannel.SMS_FALLBACK
        assert result.status == DeliveryStatus.DELIVERED

    def test_skips_without_phone(self):
        payload = _make_payload()
        recipient = _make_recipient(phone=None)
        result = sms_fallback.send(payload, recipient)
        assert result.status == DeliveryStatus.SKIPPED


# ═══════════════════════════════════════════════════════════════════════════
# Section 4: Retry & Backoff Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRetryConfig:
    """Test retry configuration and backoff computation."""

    def test_all_channels_have_config(self):
        for channel in AlertChannel:
            assert channel in RETRY_CONFIGS

    def test_exponential_backoff(self):
        config = RetryConfig(3, 5.0, "exponential")
        assert _compute_backoff(config, 1) == 5.0
        assert _compute_backoff(config, 2) == 10.0
        assert _compute_backoff(config, 3) == 20.0

    def test_linear_backoff(self):
        config = RetryConfig(5, 10.0, "linear")
        assert _compute_backoff(config, 1) == 10.0
        assert _compute_backoff(config, 2) == 20.0
        assert _compute_backoff(config, 3) == 30.0

    def test_low_bandwidth_multiplier(self):
        config = RetryConfig(3, 5.0, "exponential", low_bandwidth_multiplier=2.0)
        assert _compute_backoff(config, 1, low_bw=True) == 10.0
        assert _compute_backoff(config, 2, low_bw=True) == 20.0

    def test_sms_fallback_has_most_retries(self):
        sms_fb = RETRY_CONFIGS[AlertChannel.SMS_FALLBACK]
        for ch, config in RETRY_CONFIGS.items():
            if ch != AlertChannel.SMS_FALLBACK:
                assert sms_fb.max_retries >= config.max_retries


# ═══════════════════════════════════════════════════════════════════════════
# Section 5: Channel Capability Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCanUseChannel:
    """Test _can_use_channel recipient capability checking."""

    def test_web_always_available(self):
        r = _make_recipient(phone=None, email=None, web_push_token=None)
        assert _can_use_channel(AlertChannel.WEB_NOTIFICATION, r) is True

    def test_email_requires_email(self):
        r_with = _make_recipient(email="a@b.com")
        r_without = _make_recipient(email=None)
        assert _can_use_channel(AlertChannel.EMAIL, r_with) is True
        assert _can_use_channel(AlertChannel.EMAIL, r_without) is False

    def test_sms_requires_phone(self):
        r_with = _make_recipient(phone="+91123")
        r_without = _make_recipient(phone=None)
        assert _can_use_channel(AlertChannel.SMS, r_with) is True
        assert _can_use_channel(AlertChannel.SMS, r_without) is False

    def test_sms_not_for_low_bandwidth(self):
        r_low_bw = _make_recipient(phone="+91123", low_bandwidth=True)
        assert _can_use_channel(AlertChannel.SMS, r_low_bw) is False

    def test_sms_fallback_requires_phone(self):
        r_with = _make_recipient(phone="+91123")
        r_without = _make_recipient(phone=None)
        assert _can_use_channel(AlertChannel.SMS_FALLBACK, r_with) is True
        assert _can_use_channel(AlertChannel.SMS_FALLBACK, r_without) is False

    def test_siren_always_available(self):
        r = _make_recipient(phone=None, email=None)
        assert _can_use_channel(AlertChannel.SIREN_API, r) is True


# ═══════════════════════════════════════════════════════════════════════════
# Section 6: Single-Channel Delivery Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDeliverViaChannel:
    """Test _deliver_via_channel with retry logic."""

    def test_successful_first_attempt(self):
        payload = _make_payload()
        recipient = _make_recipient()
        result = _deliver_via_channel(AlertChannel.WEB_NOTIFICATION, payload, recipient)
        assert result.status == DeliveryStatus.DELIVERED
        assert result.retry_count == 0

    def test_skipped_channel_returns_skipped(self):
        """Siren API skips non-CRITICAL alerts."""
        payload = _make_payload(priority=AlertPriority.ADVISORY)
        recipient = _make_recipient()
        result = _deliver_via_channel(AlertChannel.SIREN_API, payload, recipient)
        assert result.status == DeliveryStatus.SKIPPED


# ═══════════════════════════════════════════════════════════════════════════
# Section 7: Multi-Channel Recipient Delivery Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestDeliverToRecipient:
    """Test _deliver_to_recipient multi-channel dispatch."""

    def test_informational_uses_web_only(self):
        payload = _make_payload(priority=AlertPriority.INFORMATIONAL)
        recipient = _make_recipient()
        channels = CHANNELS_BY_PRIORITY[AlertPriority.INFORMATIONAL]
        record = _deliver_to_recipient(payload, recipient, channels)
        assert AlertChannel.WEB_NOTIFICATION in record.channels_attempted
        assert len(record.channels_attempted) == 1
        assert record.is_reached

    def test_urgent_uses_three_channels(self):
        payload = _make_payload(priority=AlertPriority.URGENT)
        recipient = _make_recipient()
        channels = CHANNELS_BY_PRIORITY[AlertPriority.URGENT]
        record = _deliver_to_recipient(payload, recipient, channels)
        assert AlertChannel.WEB_NOTIFICATION in record.channels_attempted
        assert AlertChannel.EMAIL in record.channels_attempted
        assert AlertChannel.SMS in record.channels_attempted

    def test_critical_uses_all_channels(self):
        payload = _make_payload(priority=AlertPriority.CRITICAL)
        recipient = _make_recipient()
        channels = CHANNELS_BY_PRIORITY[AlertPriority.CRITICAL]
        record = _deliver_to_recipient(payload, recipient, channels)
        assert len(record.channels_attempted) >= 4  # at least web+email+sms+siren

    def test_low_bandwidth_skips_sms_for_fallback(self):
        """Low-bandwidth recipients should use SMS_FALLBACK instead of SMS."""
        payload = _make_payload(priority=AlertPriority.CRITICAL)
        recipient = _make_recipient(low_bandwidth=True)
        channels = CHANNELS_BY_PRIORITY[AlertPriority.CRITICAL]
        record = _deliver_to_recipient(payload, recipient, channels)
        assert AlertChannel.SMS not in record.channels_attempted
        assert AlertChannel.SMS_FALLBACK in record.channels_attempted

    def test_no_email_skips_email_channel(self):
        payload = _make_payload(priority=AlertPriority.ADVISORY)
        recipient = _make_recipient(email=None)
        channels = CHANNELS_BY_PRIORITY[AlertPriority.ADVISORY]
        record = _deliver_to_recipient(payload, recipient, channels)
        assert AlertChannel.EMAIL not in record.channels_attempted

    def test_ack_set_for_urgent(self):
        payload = _make_payload(priority=AlertPriority.URGENT, require_ack=True)
        recipient = _make_recipient()
        channels = CHANNELS_BY_PRIORITY[AlertPriority.URGENT]
        record = _deliver_to_recipient(payload, recipient, channels)
        assert record.acknowledgment == AckStatus.AWAITING

    def test_ack_not_required_for_informational(self):
        payload = _make_payload(priority=AlertPriority.INFORMATIONAL, require_ack=True)
        recipient = _make_recipient()
        channels = CHANNELS_BY_PRIORITY[AlertPriority.INFORMATIONAL]
        record = _deliver_to_recipient(payload, recipient, channels)
        assert record.acknowledgment == AckStatus.NOT_REQUIRED


# ═══════════════════════════════════════════════════════════════════════════
# Section 8: Payload Builder Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildPayloadFromRisk:
    """Test build_payload_from_risk."""

    def test_safe_maps_to_informational(self):
        risk_data = {
            "overall_risk_score": 10.0,
            "overall_risk_level": "safe",
            "dominant_hazard": "none",
        }
        payload = build_payload_from_risk(risk_data)
        assert payload.priority == AlertPriority.INFORMATIONAL

    def test_watch_maps_to_advisory(self):
        risk_data = {
            "overall_risk_score": 35.0,
            "overall_risk_level": "watch",
            "dominant_hazard": "earthquake",
        }
        payload = build_payload_from_risk(risk_data)
        assert payload.priority == AlertPriority.ADVISORY

    def test_warning_maps_to_urgent(self):
        risk_data = {
            "overall_risk_score": 60.0,
            "overall_risk_level": "warning",
            "dominant_hazard": "cyclone",
        }
        payload = build_payload_from_risk(risk_data)
        assert payload.priority == AlertPriority.URGENT
        assert payload.require_acknowledgment is True

    def test_severe_maps_to_critical(self):
        risk_data = {
            "overall_risk_score": 85.0,
            "overall_risk_level": "severe",
            "dominant_hazard": "flood",
            "alert_reasons": ["Flood probability 0.90"],
        }
        payload = build_payload_from_risk(risk_data)
        assert payload.priority == AlertPriority.CRITICAL
        assert payload.require_acknowledgment is True

    def test_messages_generated(self):
        risk_data = {
            "overall_risk_score": 72.5,
            "overall_risk_level": "warning",
            "dominant_hazard": "flood",
            "alert_reasons": ["High water level"],
        }
        payload = build_payload_from_risk(risk_data, latitude=13.0, longitude=80.0)
        assert "FLOOD" in payload.title
        assert "WARNING" in payload.title
        assert payload.latitude == 13.0
        assert payload.longitude == 80.0
        assert len(payload.message) > 0
        assert len(payload.short_message) > 0
        assert len(payload.minimal_message) > 0

    def test_location_passed_through(self):
        risk_data = {
            "overall_risk_score": 50.0,
            "overall_risk_level": "warning",
            "dominant_hazard": "earthquake",
        }
        payload = build_payload_from_risk(
            risk_data, latitude=28.6, longitude=77.2, radius_km=30.0,
        )
        assert payload.latitude == 28.6
        assert payload.longitude == 77.2
        assert payload.radius_km == 30.0

    def test_metadata_includes_action(self):
        risk_data = {
            "overall_risk_score": 85.0,
            "overall_risk_level": "severe",
            "dominant_hazard": "flood",
            "alert_action": "evacuate",
            "active_hazard_count": 2,
        }
        payload = build_payload_from_risk(risk_data)
        assert payload.metadata["action"] == "evacuate"
        assert payload.metadata["active_hazard_count"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# Section 9: Broadcast Integration Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBroadcastAlert:
    """Test broadcast_alert full orchestration."""

    def test_basic_broadcast_reaches_recipients(self):
        payload = _make_payload()
        recipients = [_make_recipient()]
        report = broadcast_alert(payload, recipients, apply_geofence=False)
        assert report.total_recipients == 1
        assert report.recipients_reached == 1
        assert report.reach_rate == 1.0

    def test_broadcast_with_geofence_filters(self):
        payload = _make_payload(radius_km=10.0)
        r_near = _make_recipient(rid="NEAR", lat=CHENNAI_LAT, lon=CHENNAI_LON)
        r_far = _make_recipient(rid="FAR", lat=CHENNAI_LAT + 2.0, lon=CHENNAI_LON)
        report = broadcast_alert(payload, [r_near, r_far], apply_geofence=True)
        assert report.total_recipients == 1
        assert report.recipients_reached == 1

    def test_broadcast_no_geofence(self):
        payload = _make_payload(radius_km=1.0)
        r_far = _make_recipient(rid="FAR", lat=CHENNAI_LAT + 5.0, lon=CHENNAI_LON)
        # Without geofence, even far recipients are included
        report = broadcast_alert(payload, [r_far], apply_geofence=False)
        assert report.total_recipients == 1

    def test_broadcast_stores_ack_records(self):
        payload = _make_payload(require_ack=True)
        recipients = [_make_recipient()]
        broadcast_alert(payload, recipients, apply_geofence=False)
        assert payload.alert_id in _ack_store
        assert "R001" in _ack_store[payload.alert_id]

    def test_broadcast_multiple_recipients(self):
        payload = _make_payload()
        recipients = [
            _make_recipient(rid=f"R{i:03d}", lat=CHENNAI_LAT, lon=CHENNAI_LON)
            for i in range(5)
        ]
        report = broadcast_alert(payload, recipients, apply_geofence=False)
        assert report.total_recipients == 5
        assert report.recipients_reached == 5

    def test_broadcast_report_timing(self):
        payload = _make_payload()
        recipients = [_make_recipient()]
        report = broadcast_alert(payload, recipients, apply_geofence=False)
        assert report.broadcast_started_at is not None
        assert report.broadcast_completed_at is not None
        assert report.broadcast_completed_at >= report.broadcast_started_at

    def test_broadcast_empty_recipients_geofence(self):
        payload = _make_payload(radius_km=1.0)
        # All recipients far away
        recipients = [
            _make_recipient(rid="FAR", lat=CHENNAI_LAT + 10.0, lon=CHENNAI_LON)
        ]
        report = broadcast_alert(payload, recipients, apply_geofence=True)
        assert report.total_recipients == 0
        assert report.recipients_reached == 0

    def test_broadcast_critical_uses_all_channels(self):
        payload = _make_payload(priority=AlertPriority.CRITICAL)
        recipient = _make_recipient()
        report = broadcast_alert(payload, [recipient], apply_geofence=False)
        record = report.delivery_records[0]
        # Should have attempted web + email + SMS + siren + SMS_fallback
        assert len(record.channels_attempted) >= 4


# ═══════════════════════════════════════════════════════════════════════════
# Section 10: Acknowledgment Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAcknowledgment:
    """Test acknowledgment tracking."""

    def test_record_ack_success(self):
        payload = _make_payload(require_ack=True)
        recipients = [_make_recipient()]
        broadcast_alert(payload, recipients, apply_geofence=False)

        success = record_acknowledgment(
            payload.alert_id, "R001", AlertChannel.WEB_NOTIFICATION,
        )
        assert success is True

    def test_record_ack_updates_status(self):
        payload = _make_payload(require_ack=True)
        recipients = [_make_recipient()]
        broadcast_alert(payload, recipients, apply_geofence=False)

        record_acknowledgment(payload.alert_id, "R001", AlertChannel.SMS)

        record = _ack_store[payload.alert_id]["R001"]
        assert record.acknowledgment == AckStatus.ACKNOWLEDGED
        assert record.acknowledged_via == AlertChannel.SMS
        assert record.acknowledged_at is not None

    def test_record_ack_unknown_alert(self):
        success = record_acknowledgment("FAKE_ID", "R001", AlertChannel.WEB_NOTIFICATION)
        assert success is False

    def test_record_ack_unknown_recipient(self):
        payload = _make_payload()
        recipients = [_make_recipient()]
        broadcast_alert(payload, recipients, apply_geofence=False)
        success = record_acknowledgment(payload.alert_id, "FAKE_R", AlertChannel.SMS)
        assert success is False

    def test_get_ack_status(self):
        payload = _make_payload(require_ack=True)
        recipients = [
            _make_recipient(rid="R001"),
            _make_recipient(rid="R002", lat=CHENNAI_LAT, lon=CHENNAI_LON),
        ]
        broadcast_alert(payload, recipients, apply_geofence=False)

        record_acknowledgment(payload.alert_id, "R001", AlertChannel.WEB_NOTIFICATION)

        status = get_ack_status(payload.alert_id)
        assert status["total_recipients"] == 2
        assert status["acknowledged"] == 1
        assert status["awaiting"] == 1

    def test_get_ack_status_empty(self):
        status = get_ack_status("NONEXISTENT")
        assert status["total_recipients"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Section 11: Report Store Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAlertReportStore:
    """Test in-memory alert report storage."""

    def test_store_and_retrieve(self):
        payload = _make_payload()
        report = AlertReport(
            alert_id=payload.alert_id,
            payload=payload,
            total_recipients=5,
            recipients_reached=3,
        )
        store_alert_report(report)
        retrieved = get_alert_report(payload.alert_id)
        assert retrieved is not None
        assert retrieved.total_recipients == 5

    def test_retrieve_nonexistent(self):
        assert get_alert_report("NOSUCH") is None


# ═══════════════════════════════════════════════════════════════════════════
# Section 12: Edge Case Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_recipient_no_contact_info_at_all(self):
        """Recipient with no phone, email, or push token."""
        payload = _make_payload(priority=AlertPriority.CRITICAL)
        recipient = AlertRecipient(
            recipient_id="GHOST", name="Ghost User",
            latitude=CHENNAI_LAT, longitude=CHENNAI_LON,
        )
        report = broadcast_alert(payload, [recipient], apply_geofence=False)
        assert report.total_recipients == 1
        # Web always works (simulated), siren always works (zone-based)
        assert report.recipients_reached >= 1

    def test_zero_radius_geofence(self):
        """Radius of 1 km — only very close recipients."""
        payload = _make_payload(radius_km=1.0)
        r_exact = _make_recipient(lat=CHENNAI_LAT, lon=CHENNAI_LON)
        r_far = _make_recipient(rid="FAR", lat=CHENNAI_LAT + 0.1, lon=CHENNAI_LON)
        report = broadcast_alert(payload, [r_exact, r_far], apply_geofence=True)
        assert report.total_recipients == 1  # only the exact match

    def test_payload_with_long_message(self):
        """Long messages should be truncated for SMS channels."""
        payload = AlertPayload(
            priority=AlertPriority.URGENT,
            title="Long Alert",
            message="A" * 1000,
            short_message="B" * 200,
            minimal_message="C" * 100,
            latitude=CHENNAI_LAT,
            longitude=CHENNAI_LON,
        )
        recipient = _make_recipient()
        report = broadcast_alert(payload, [recipient], apply_geofence=False)
        assert report.recipients_reached == 1

    def test_broadcast_to_many_recipients(self):
        """Broadcast to 20 recipients."""
        payload = _make_payload(priority=AlertPriority.INFORMATIONAL)
        recipients = [
            _make_recipient(
                rid=f"R{i:03d}", lat=CHENNAI_LAT, lon=CHENNAI_LON,
            )
            for i in range(20)
        ]
        report = broadcast_alert(payload, recipients, apply_geofence=False)
        assert report.total_recipients == 20
        assert report.recipients_reached == 20

    def test_to_dict_roundtrip(self):
        """AlertReport.to_dict() produces valid JSON-serialisable dict."""
        payload = _make_payload()
        recipients = [_make_recipient()]
        report = broadcast_alert(payload, recipients, apply_geofence=False)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["alert_id"] == payload.alert_id
        assert isinstance(d["delivery_records"], list)
        assert d["total_recipients"] == 1

    def test_all_expansion_factors_defined(self):
        """Every priority has an expansion factor."""
        for p in AlertPriority:
            assert p in EXPANSION_FACTORS
