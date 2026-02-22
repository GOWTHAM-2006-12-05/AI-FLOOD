# Multi-Channel Alert Broadcasting System

## Overview

The alert broadcasting system delivers disaster warnings to affected populations through **five channels**, with intelligent routing based on alert severity, recipient capabilities, and network conditions.

It sits atop the [risk aggregation engine](RISK_AGGREGATION.md) and converts risk scores into actionable alerts that reach the right people through the right channels at the right time.

---

## Architecture Diagram

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │                     RISK AGGREGATOR                                  │
 │                                                                      │
 │   Flood ────┐                                                        │
 │   Earthquake ┼──► aggregate_risk() ──► overall_risk_score (0–100%)  │
 │   Cyclone ──┘                          overall_risk_level            │
 └──────────────────────┬───────────────────────────────────────────────┘
                        │
                        ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │                   ALERT SERVICE (alert_service.py)                   │
 │                                                                      │
 │   1. build_payload_from_risk()                                       │
 │      risk_level → AlertPriority                                      │
 │      Generate message variants (full / SMS / minimal)                │
 │                                                                      │
 │   2. Geo-fence Targeting (geo_fence.py)                              │
 │      ┌─────────────────────────────────────┐                         │
 │      │  Bounding-box pre-filter (fast)     │                         │
 │      │  Haversine precise filter           │                         │
 │      │  Priority-based radius expansion    │                         │
 │      │    URGENT:  ×1.25                   │                         │
 │      │    CRITICAL: ×1.50                  │                         │
 │      └─────────────────────────────────────┘                         │
 │                                                                      │
 │   3. Channel Selection (CHANNELS_BY_PRIORITY)                        │
 │      INFORMATIONAL → [Web]                                           │
 │      ADVISORY      → [Web, Email]                                    │
 │      URGENT        → [Web, Email, SMS]                               │
 │      CRITICAL      → [Web, Email, SMS, Siren, SMS Fallback]         │
 │                                                                      │
 │   4. Per-channel dispatch with retry                                 │
 │   5. Acknowledgment tracking                                         │
 │   6. Delivery report compilation                                     │
 └────────┬──────┬──────┬──────┬──────┬─────────────────────────────────┘
          │      │      │      │      │
          ▼      ▼      ▼      ▼      ▼
 ┌──────┐┌──────┐┌─────┐┌─────┐┌──────────┐
 │ Web  ││Email ││ SMS ││Siren││SMS       │
 │ Push ││Alert ││ GW  ││ API ││Fallback  │
 │      ││      ││     ││     ││(2G/GSM)  │
 └──┬───┘└──┬───┘└──┬──┘└──┬──┘└──┬───────┘
    │       │       │      │      │
    ▼       ▼       ▼      ▼      ▼
  Browser  Inbox  Phone  Siren  Basic
  /App             SMS   Tower  Phone
```

---

## Alert Priority Mapping

| Risk Level | Alert Priority   | Value | Channels Activated                        | Ack Required |
|------------|------------------|-------|-------------------------------------------|--------------|
| Safe       | INFORMATIONAL    | 1     | Web only                                  | No           |
| Watch      | ADVISORY         | 2     | Web + Email                               | No           |
| Warning    | URGENT           | 3     | Web + Email + SMS                         | Yes          |
| Severe     | CRITICAL         | 4     | Web + Email + SMS + Siren + SMS Fallback  | Yes          |

---

## Channel Details

### 1. Web Push Notification (`web_push.py`)

| Property        | Value                                |
|-----------------|--------------------------------------|
| **Protocol**    | Web Push (RFC 8030) + VAPID          |
| **Cost**        | Zero marginal cost                   |
| **Latency**     | Sub-second                           |
| **Payload**     | JSON (title, body, icon, actions)    |
| **Requires**    | Browser notification permission      |
| **Always sent** | Yes (first channel for all levels)   |

Features:
- Action buttons: "I'm Safe" (ack) + "View Details"
- Vibration pattern for URGENT+: `[200, 100, 200]`
- `requireInteraction: true` for priority ≥ 3

### 2. Email Alert (`email_alert.py`)

| Property     | Value                             |
|--------------|-----------------------------------|
| **Protocol** | SMTP / API (SendGrid, SES)        |
| **Payload**  | HTML + plain-text multipart       |
| **Subject**  | Emoji-coded: `🚨 [URGENT] ...`   |
| **Activated**| ADVISORY and above                |

Features:
- Priority-coded colour headers (green/amber/orange/red)
- Plain-text fallback for low-bandwidth email clients
- Inline CSS for maximum client compatibility

### 3. SMS Gateway (`sms_gateway.py`)

| Property     | Value                              |
|--------------|------------------------------------|
| **Encoding** | GSM 7-bit (≤160 chars)             |
| **Providers**| Twilio, MSG91, TextLocal           |
| **Template** | `[PRIORITY] body Reply SAFE. Ref:` |
| **Activated**| URGENT and above                   |

Features:
- Provider-agnostic abstraction layer
- Delivery receipt via webhook callback
- Acknowledgment via SMS reply (`SAFE`)

### 4. Siren API (`siren_api.py`)

| Property     | Value                              |
|--------------|------------------------------------|
| **Protocol** | HTTP POST to IoT siren controllers |
| **Patterns** | STEADY_TONE, WAIL, PULSE           |
| **Zone ID**  | Auto-generated from lat/lon grid   |
| **Activated**| CRITICAL only (priority ≥ 4)       |

Features:
- 0.1° grid zones (~11 km cells)
- Pattern selection based on hazard type
- Duration proportional to severity

### 5. SMS Fallback (`sms_fallback.py`)

| Property     | Value                              |
|--------------|------------------------------------|
| **Encoding** | Strict GSM 7-bit (≤70 chars)       |
| **No URLs**  | Bandwidth saving                   |
| **Format**   | `FL!SEV ADYAR. ACT:EVAC. #3A7B`   |
| **Activated**| CRITICAL, or SMS failure fallback  |

Abbreviation codes:
- Hazard: `FL` (Flood), `EQ` (Earthquake), `CY` (Cyclone)
- Level: `SAF`, `WCH`, `WRN`, `SEV`, `CRT`
- Action: `MON`, `INF`, `PREP`, `EVAC`, `SHLT`

---

## Retry & Redundancy Strategy

### Per-Channel Retry Configuration

| Channel       | Max Retries | Backoff Base | Backoff Type  |
|---------------|-------------|-------------|---------------|
| Web Push      | 2           | 1.0s        | Exponential   |
| Email         | 3           | 2.0s        | Exponential   |
| SMS           | 3           | 5.0s        | Exponential   |
| Siren         | 2           | 3.0s        | Exponential   |
| SMS Fallback  | 5           | 10.0s       | Linear        |

### Backoff Formula

**Exponential:**
$$\text{delay} = \text{base} \times 2^{(\text{attempt} - 1)}$$

Example for SMS (base = 5s):
- Attempt 1: 5s → Attempt 2: 10s → Attempt 3: 20s

**Linear** (for SMS Fallback — 2G latency):
$$\text{delay} = \text{base} \times \text{attempt}$$

**Low-bandwidth modifier:**
$$\text{delay}_{\text{low\_bw}} = \text{delay} \times 2.0$$

### Redundancy Rules

1. **SMS fails → auto-trigger SMS Fallback** (if not already attempted)
2. **Each channel operates independently** (failure in one doesn't block others)
3. **All channels fail → recipient logged as UNREACHABLE**

---

## Geo-Fenced Targeting

### Algorithm

```
1. Compute effective_radius = base_radius × expansion_factor
2. Compute lat/lon bounding box (O(1) arithmetic)
3. Pre-filter recipients outside bounding box (simple float comparison)
4. Run Haversine on remaining candidates (precise distance)
5. Return (targeted, excluded) lists
```

### Expansion Factors

| Priority       | Factor | 50 km base → Effective |
|----------------|--------|------------------------|
| INFORMATIONAL  | 1.0×   | 50.0 km                |
| ADVISORY       | 1.0×   | 50.0 km                |
| URGENT         | 1.25×  | 62.5 km                |
| CRITICAL       | 1.50×  | 75.0 km                |

### Performance

For a 100,000-recipient database with a 50 km radius:
- Bounding box eliminates ~99.5% of candidates
- Only ~500 Haversine computations needed
- **200× speedup** over brute-force

---

## Acknowledgment Tracking

### Flow

```
Alert sent → AckStatus.AWAITING
    │
    ├── Recipient clicks "I'm Safe" (Web Push)     → ACKNOWLEDGED
    ├── Recipient replies "SAFE" (SMS)              → ACKNOWLEDGED
    ├── Recipient clicks ack link (Email)           → ACKNOWLEDGED
    │
    └── No response after timeout                   → TIMED_OUT → ESCALATED
```

### API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/alerts/broadcast/risk` | POST | Broadcast from risk data |
| `/api/v1/alerts/broadcast/direct` | POST | Broadcast with explicit payload |
| `/api/v1/alerts/{id}/acknowledge` | POST | Record ack from recipient |
| `/api/v1/alerts/{id}/status` | GET | Full delivery report |
| `/api/v1/alerts/channels` | GET | Channel configuration |
| `/api/v1/alerts/health` | GET | Service health check |

---

## Low Connectivity Handling

Recipients flagged with `low_bandwidth=True` receive:

1. **SMS Fallback INSTEAD OF standard SMS** (not in addition to)
2. **Web Push still attempted** (queued by push service)
3. **Email downgraded** to plain-text only
4. **Retry intervals doubled** (×2.0 multiplier)
5. **Delivery confirmation** via polling (not webhooks)

Detection heuristics:
- Recipient manually sets `low_bandwidth=True`
- Previous SMS delivery failure rate > 50%
- Geographic mapping of 2G-only coverage areas

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                             │
│                                                              │
│  POST /broadcast/risk ──────────────────────────────────┐    │
│                                                          │    │
│  POST /broadcast/direct ─────────────────────────────┐  │    │
│                                                       │  │    │
│  POST /{id}/acknowledge ──► record_acknowledgment()   │  │    │
│                                                       │  │    │
│  GET  /{id}/status ──────► get_alert_report()         │  │    │
│                                                       ▼  ▼    │
│                                               broadcast_alert()│
└─────────────────────────────────┬────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                             │
│                                                              │
│  build_payload_from_risk() ──► AlertPayload                  │
│                                                              │
│  filter_recipients_by_geofence() ──► (targeted, excluded)    │
│                                                              │
│  CHANNELS_BY_PRIORITY[priority] ──► [channel_list]           │
│                                                              │
│  For each recipient:                                         │
│    For each channel:                                         │
│      _can_use_channel() ──► bool                             │
│      _deliver_via_channel() ──► DeliveryAttempt              │
│        └── retry with exponential backoff                    │
│                                                              │
│  SMS failed? ──► auto SMS Fallback                           │
│                                                              │
│  Compile AlertReport                                         │
│  Store in _ack_store for tracking                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
backend/app/alerts/
├── __init__.py              # Package init
├── models.py                # AlertPriority, AlertChannel, AlertPayload,
│                            # AlertRecipient, DeliveryAttempt, AlertReport
├── geo_fence.py             # Spatial targeting with Haversine + expansion
├── alert_service.py         # Core orchestration, retry, ack tracking
└── channels/
    ├── __init__.py
    ├── web_push.py          # Web Push (RFC 8030)
    ├── sms_gateway.py       # SMS via Twilio/MSG91/TextLocal
    ├── email_alert.py       # HTML + plain-text email
    ├── siren_api.py         # IoT siren activation (CRITICAL only)
    └── sms_fallback.py      # ≤70 char GSM for 2G networks

backend/app/api/v1/
└── alerts.py                # FastAPI endpoints

tests/
└── test_alert_service.py    # 90 tests (models, channels, geo-fence,
                             # service, ack, edge cases)
```

---

## API Examples

### Broadcast from Risk Data

```bash
curl -X POST http://localhost:8000/api/v1/alerts/broadcast/risk \
  -H "Content-Type: application/json" \
  -d '{
    "latitude": 13.0827,
    "longitude": 80.2707,
    "radius_km": 50.0,
    "overall_risk_score": 72.5,
    "overall_risk_level": "warning",
    "dominant_hazard": "flood",
    "alert_action": "prepare",
    "alert_reasons": ["Flood probability 0.80", "Heavy rainfall"],
    "active_hazard_count": 1,
    "recipients": [
      {
        "recipient_id": "R001",
        "name": "Arun",
        "latitude": 13.05,
        "longitude": 80.25,
        "phone": "+919876543210",
        "email": "arun@example.com"
      }
    ]
  }'
```

### Acknowledge an Alert

```bash
curl -X POST http://localhost:8000/api/v1/alerts/ALR-3A7B12345678/acknowledge \
  -H "Content-Type: application/json" \
  -d '{
    "recipient_id": "R001",
    "channel": "sms"
  }'
```

### Check Delivery Status

```bash
curl http://localhost:8000/api/v1/alerts/ALR-3A7B12345678/status
```

---

## Test Coverage

**90 tests** across 12 test classes:

| Test Class | Count | Coverage |
|------------|-------|----------|
| AlertPriority | 3 | Enum ordering, values, comparison |
| AlertChannel | 2 | All channels present, string values |
| ChannelsByPriority | 4 | Correct channel sets per priority |
| AlertPayload | 3 | ID generation, to_dict, UTC timestamps |
| AlertRecipient | 2 | Serialisation, optional fields |
| RecipientDeliveryRecord | 3 | is_reached logic, serialisation |
| AlertReport | 3 | Reach/ack rates, zero-division safety |
| Geo-fence (expand/filter) | 11 | Expansion factors, filtering, mixed, empty |
| Channel Backends | 11 | All 5 channels: success, skip, format |
| Retry & Backoff | 5 | Exponential, linear, low-BW, config |
| Channel Capability | 6 | Per-channel capability checks |
| Delivery (single/multi) | 9 | Channel routing, low-BW redirect, ack |
| Payload Builder | 7 | All 4 risk levels, messages, metadata |
| Broadcast Integration | 8 | End-to-end, geofence, timing, critical |
| Acknowledgment | 6 | Record, update, unknown, status query |
| Report Store | 2 | Store/retrieve, nonexistent |
| Edge Cases | 6 | No contact, zero radius, long msg, many recipients |
