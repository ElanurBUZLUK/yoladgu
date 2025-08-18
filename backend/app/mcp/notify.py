import os, smtplib, json
from email.mime.text import MIMEText
import httpx

SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL")
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))

def notify_slack(text: str):
    if not SLACK_WEBHOOK:
        return
    with httpx.Client(timeout=5.0) as c:
        c.post(SLACK_WEBHOOK, json={"text": text[:4000]})

def send_email(to_addr: str, subject: str, body: str):
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS):
        return
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = to_addr
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)