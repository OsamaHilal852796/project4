"""Warnings to student / guardian.
 - Email : SMTP (Gmail app-password, Outlook, SendGrid SMTP ...)
 - SMS   : Twilio REST API (optional)
 - WhatsApp: free click-to-chat link (wa.me), no account needed
Credentials come from environment variables or Streamlit secrets - never from code."""
import os
import re
import smtplib
import urllib.parse
from email.message import EmailMessage
import requests

DEFAULT_TEMPLATE = (
    "Dear {recipient},\n\n"
    "This is an early-warning notice from the Student Performance Warning System.\n\n"
    "Student : {name} (ID {sid})\n"
    "Subject : {subject}\n"
    "Projected final exam score : {pred_final}\n"
    "Projected total grade      : {pred_total}\n"
    "Risk level                 : {risk}  (confidence {conf}%)\n\n"
    "We recommend extra revision and contacting the course staff as soon as possible.\n\n"
    "Regards,\nCourse Staff"
)


def _get(key, secrets=None):
    v = os.environ.get(key)
    if not v and secrets is not None:
        try:
            v = secrets.get(key)
        except Exception:
            v = None
    return v


def config_status(secrets=None) -> dict:
    return {
        "email": all(_get(k, secrets) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD")),
        "sms": all(_get(k, secrets) for k in ("TWILIO_SID", "TWILIO_TOKEN", "TWILIO_FROM")),
    }


def render(template, row, subject, recipient="Parent/Guardian"):
    g = lambda k, d="-": row.get(k, d) if hasattr(row, "get") else d
    return template.format(
        recipient=recipient, name=g("Name"), sid=g("Student ID"), subject=subject,
        pred_final=g("Predicted Final"), pred_total=g("Predicted Total"),
        risk=g("Risk Class"), conf=g("Confidence %"))


def send_email(to, subject, body, secrets=None):
    host, user, pwd = (_get(k, secrets) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"))
    if not (host and user and pwd):
        raise RuntimeError("Email is not configured (set SMTP_HOST, SMTP_USER, SMTP_PASSWORD).")
    port = int(_get("SMTP_PORT", secrets) or 587)
    msg = EmailMessage()
    msg["From"] = _get("SMTP_FROM", secrets) or user
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP(host, port, timeout=20) as s:
        s.starttls()
        s.login(user, pwd)
        s.send_message(msg)


def send_sms(to, body, secrets=None):
    sid, tok, frm = (_get(k, secrets) for k in ("TWILIO_SID", "TWILIO_TOKEN", "TWILIO_FROM"))
    if not (sid and tok and frm):
        raise RuntimeError("SMS is not configured (set TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM).")
    r = requests.post(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
                      data={"To": to, "From": frm, "Body": body}, auth=(sid, tok), timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"Twilio error {r.status_code}: {r.text[:200]}")


def whatsapp_link(phone, body):
    digits = re.sub(r"\D", "", str(phone))
    return f"https://wa.me/{digits}?text={urllib.parse.quote(body)}"
