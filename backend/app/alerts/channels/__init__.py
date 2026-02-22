"""
channels — Per-channel delivery backends.

Each channel module exposes:
    send(payload, recipient) → DeliveryAttempt

Channels are stateless functions. Retry logic lives in alert_service.
"""
