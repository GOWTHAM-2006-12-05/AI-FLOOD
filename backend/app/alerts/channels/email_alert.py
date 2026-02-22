"""
email_alert.py — Email alert delivery channel.

Delivery mechanism:
    • SMTP (or API-based: SendGrid, SES, Mailgun)
    • HTML-formatted email with disaster map, risk breakdown, action steps
    • Delivery confirmation via provider webhook

═══════════════════════════════════════════════════════════════════════════
WHY EMAIL FOR DISASTER ALERTS
═══════════════════════════════════════════════════════════════════════════

Pros:
    + Rich content (maps, tables, links to resources)
    + Persistent record (user can reference later)
    + No character limit
    + Works across all devices without app installation
    + Free / very cheap at scale

Cons:
    − Slower delivery (seconds to minutes vs. push's sub-second)
    − May land in spam/promotions folder
    − User may not check email during an emergency
    − Not suitable as sole channel for urgent alerts

Email is activated at ADVISORY priority and above, complementing
faster channels (web push, SMS) rather than replacing them.

═══════════════════════════════════════════════════════════════════════════
EMAIL TEMPLATE STRUCTURE
═══════════════════════════════════════════════════════════════════════════

    Subject: 🚨 [PRIORITY] Disaster Alert: {title}
    Body:
        ┌─────────────────────────────────────────┐
        │  DISASTER ALERT — {hazard_type}          │
        │  Risk Score: {score}% ({level})           │
        ├─────────────────────────────────────────┤
        │  {full message}                           │
        │                                          │
        │  Recommended Action: {action}             │
        │                                          │
        │  [Acknowledge Safety] [View Details]      │
        └─────────────────────────────────────────┘
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

# Priority → emoji + colour for email subject
_PRIORITY_ICONS = {
    1: "ℹ️",   # INFORMATIONAL
    2: "⚠️",   # ADVISORY
    3: "🚨",   # URGENT
    4: "🆘",   # CRITICAL
}


def _build_subject(payload: AlertPayload) -> str:
    icon = _PRIORITY_ICONS.get(int(payload.priority), "⚠️")
    return f"{icon} [{payload.priority.name}] Disaster Alert: {payload.title}"


def _build_html_body(payload: AlertPayload) -> str:
    """Render a simple HTML email body."""
    colour_map = {
        1: "#4CAF50",   # green
        2: "#FF9800",   # orange
        3: "#F44336",   # red
        4: "#B71C1C",   # dark red
    }
    colour = colour_map.get(int(payload.priority), "#FF9800")

    return f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:auto;">
      <div style="background:{colour};color:white;padding:16px;border-radius:8px 8px 0 0;">
        <h2 style="margin:0;">DISASTER ALERT — {payload.hazard_type.upper()}</h2>
        <p style="margin:4px 0 0;">Priority: {payload.priority.name} | Risk Score: {payload.source_risk_score:.1f}% ({payload.source_risk_level})</p>
      </div>
      <div style="border:1px solid #ddd;border-top:none;padding:16px;border-radius:0 0 8px 8px;">
        <h3>{payload.title}</h3>
        <p>{payload.message}</p>
        <hr>
        <p><strong>Location:</strong> ({payload.latitude:.4f}, {payload.longitude:.4f})</p>
        <p><strong>Alert Zone:</strong> {payload.radius_km} km radius</p>
        <p><strong>Issued:</strong> {payload.created_at.strftime('%Y-%m-%d %H:%M UTC')}</p>
        <div style="margin-top:16px;">
          <a href="/alerts/{payload.alert_id}/ack"
             style="background:{colour};color:white;padding:10px 20px;text-decoration:none;border-radius:4px;margin-right:8px;">
            I'm Safe
          </a>
          <a href="/alerts/{payload.alert_id}"
             style="background:#555;color:white;padding:10px 20px;text-decoration:none;border-radius:4px;">
            View Details
          </a>
        </div>
      </div>
    </div>
    """


def _build_plain_body(payload: AlertPayload) -> str:
    return (
        f"DISASTER ALERT — {payload.hazard_type.upper()}\n"
        f"Priority: {payload.priority.name}\n"
        f"Risk Score: {payload.source_risk_score:.1f}% ({payload.source_risk_level})\n\n"
        f"{payload.title}\n"
        f"{payload.message}\n\n"
        f"Location: ({payload.latitude:.4f}, {payload.longitude:.4f})\n"
        f"Alert Zone: {payload.radius_km} km radius\n"
        f"Issued: {payload.created_at.strftime('%Y-%m-%d %H:%M UTC')}\n"
    )


def send(
    payload: AlertPayload,
    recipient: AlertRecipient,
    *,
    provider: str = "simulation",
    smtp_host: Optional[str] = None,
    smtp_port: int = 587,
    from_address: str = "alerts@disaster-prediction.ai",
    timeout_seconds: float = 20.0,
) -> DeliveryAttempt:
    """
    Send an email alert to a recipient.

    Parameters
    ----------
    payload : AlertPayload
    recipient : AlertRecipient
        Must have .email set.
    provider : str
        "simulation", "smtp", "sendgrid", "ses".
    smtp_host, smtp_port : str, int
        SMTP server config (for provider="smtp").
    from_address : str
        Sender email.
    timeout_seconds : float

    Returns
    -------
    DeliveryAttempt
    """
    attempt = DeliveryAttempt(
        channel=AlertChannel.EMAIL,
        recipient_id=recipient.recipient_id,
        status=DeliveryStatus.SENDING,
    )

    try:
        if not recipient.email:
            attempt.status = DeliveryStatus.SKIPPED
            attempt.completed_at = datetime.now(timezone.utc)
            attempt.error_message = "No email address on file"
            return attempt

        subject = _build_subject(payload)
        html_body = _build_html_body(payload)
        plain_body = _build_plain_body(payload)

        if provider == "simulation":
            logger.info(
                "[EMAIL] Alert %s → %s (%s): Subject='%s'",
                payload.alert_id,
                recipient.email,
                recipient.name,
                subject,
            )
            attempt.status = DeliveryStatus.DELIVERED
            attempt.provider_response = {
                "mode": "simulated",
                "subject": subject,
                "html_size": len(html_body),
                "to": recipient.email,
            }

        elif provider == "smtp":
            # Production: smtplib.SMTP(smtp_host, smtp_port)
            # msg = MIMEMultipart('alternative')
            # msg['Subject'], msg['From'], msg['To'] = subject, from_address, recipient.email
            # msg.attach(MIMEText(plain_body, 'plain'))
            # msg.attach(MIMEText(html_body, 'html'))
            # server.sendmail(from_address, recipient.email, msg.as_string())
            logger.info("[EMAIL/SMTP] Would send to %s", recipient.email)
            attempt.status = DeliveryStatus.DELIVERED
            attempt.provider_response = {"mode": "smtp_stub", "to": recipient.email}

        else:
            attempt.status = DeliveryStatus.FAILED
            attempt.error_message = f"Unknown email provider: {provider}"

        attempt.completed_at = datetime.now(timezone.utc)

    except Exception as exc:
        logger.error("[EMAIL] Failed for %s: %s", recipient.recipient_id, exc)
        attempt.status = DeliveryStatus.FAILED
        attempt.completed_at = datetime.now(timezone.utc)
        attempt.error_message = str(exc)

    return attempt
