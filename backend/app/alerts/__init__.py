"""
alerts — Multi-channel disaster alert broadcasting system.

Sub-modules:
    channels/       — Per-channel delivery backends (web, SMS, email, siren)
    alert_service   — Core orchestration: routing, retry, geo-fence, ack tracking
    geo_fence       — Spatial targeting of alert zones
    models          — Data structures shared across the system
"""
