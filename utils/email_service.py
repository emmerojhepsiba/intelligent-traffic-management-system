import os
import sys
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta

# In-memory OTP store
_otp_store = {}

# SMTP Configuration - loaded from environment or .env
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_EMAIL = os.environ.get("SMTP_EMAIL", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")


def load_smtp_config():
    """Load SMTP credentials from .env file if not set in environment."""
    global SMTP_EMAIL, SMTP_PASSWORD
    if SMTP_EMAIL and SMTP_PASSWORD:
        return
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key == "SMTP_EMAIL":
                        SMTP_EMAIL = val
                    elif key == "SMTP_PASSWORD":
                        SMTP_PASSWORD = val


def generate_otp(email: str) -> str:
    """Generate a 6-digit OTP for the given email."""
    otp = "".join(random.choices(string.digits, k=6))
    _otp_store[email] = {
        "otp": otp,
        "expires": datetime.utcnow() + timedelta(minutes=5),
        "attempts": 0
    }
    return otp


def verify_otp(email: str, otp: str) -> bool:
    """Verify the OTP for the given email."""
    record = _otp_store.get(email)
    if not record:
        return False
    if datetime.utcnow() > record["expires"]:
        del _otp_store[email]
        return False
    record["attempts"] += 1
    if record["attempts"] > 5:
        del _otp_store[email]
        return False
    if record["otp"] == otp:
        del _otp_store[email]
        return True
    return False


def send_otp_email(email: str, otp: str, app=None):
    """Send OTP via Gmail SMTP. Falls back to console if SMTP is not configured."""
    load_smtp_config()

    if SMTP_EMAIL and SMTP_PASSWORD:
        try:
            msg = MIMEMultipart("alternative")
            msg["From"] = f"TrafficCommand <{SMTP_EMAIL}>"
            msg["To"] = email
            msg["Subject"] = "TrafficCommand - Your Login OTP"

            # Plain text version
            text = f"""Your One-Time Password (OTP) for TrafficCommand login:

    {otp}

This code expires in 5 minutes.
Do not share this code with anyone.

- TrafficCommand Smart Traffic Management System"""

            # HTML version
            html = f"""
<div style="font-family:Arial,sans-serif;max-width:480px;margin:0 auto;background:#0f172a;border-radius:12px;overflow:hidden;border:1px solid #1e293b">
    <div style="padding:24px 32px;background:linear-gradient(135deg,#1e3a5f,#0f172a);border-bottom:1px solid #1e293b">
        <h1 style="color:#748ffc;margin:0;font-size:20px">TrafficCommand</h1>
        <p style="color:#64748b;margin:4px 0 0;font-size:12px">Smart Traffic Management System</p>
    </div>
    <div style="padding:32px">
        <p style="color:#e2e8f0;font-size:15px;margin:0 0 8px">Your login verification code:</p>
        <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:20px;text-align:center;margin:16px 0">
            <span style="font-size:32px;font-weight:bold;letter-spacing:8px;color:#748ffc;font-family:monospace">{otp}</span>
        </div>
        <p style="color:#94a3b8;font-size:13px;margin:16px 0 0">This code expires in <strong style="color:#e2e8f0">5 minutes</strong>.</p>
        <p style="color:#64748b;font-size:12px;margin:8px 0 0">If you didn't request this code, please ignore this email.</p>
    </div>
    <div style="padding:16px 32px;background:#0a0f1d;border-top:1px solid #1e293b">
        <p style="color:#475569;font-size:11px;margin:0;text-align:center">TrafficCommand &mdash; AI-Powered Traffic Intelligence</p>
    </div>
</div>"""

            msg.attach(MIMEText(text, "plain"))
            msg.attach(MIMEText(html, "html"))

            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(SMTP_EMAIL, SMTP_PASSWORD)
                server.sendmail(SMTP_EMAIL, email, msg.as_string())

            print(f"[MAIL] OTP sent to {email}", flush=True)
            return True

        except smtplib.SMTPAuthenticationError:
            print(f"[MAIL ERROR] SMTP auth failed. Check App Password.", flush=True)
        except Exception as e:
            print(f"[MAIL ERROR] {e}", flush=True)

    # Fallback: print to console
    print(f"\n{'='*50}", flush=True)
    print(f"  OTP for {email}: {otp}", flush=True)
    print(f"  (Configure .env with SMTP credentials to send real emails)", flush=True)
    print(f"{'='*50}\n", flush=True)
    sys.stdout.flush()
    return True