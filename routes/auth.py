import os
import jwt
import bcrypt
from datetime import datetime, timedelta
from functools import wraps
from flask import Blueprint, request, jsonify, session, redirect, url_for, render_template
from utils.db import get_db
from utils.email_service import generate_otp, verify_otp, send_otp_email
from config import Config

auth_bp = Blueprint("auth", __name__)

def create_token(email: str, name: str) -> str:
    payload = {
        "email": email,
        "name": name,
        "exp": datetime.utcnow() + timedelta(hours=Config.JWT_EXPIRY_HOURS)
    }
    return jwt.encode(payload, Config.SECRET_KEY, algorithm="HS256")

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = session.get("token")
        if not token:
            if request.headers.get("Accept", "").startswith("application/json"):
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("auth.login_page"))
        try:
            data = jwt.decode(token, Config.SECRET_KEY, algorithms=["HS256"])
            request.admin = data
        except jwt.ExpiredSignatureError:
            session.pop("token", None)
            return redirect(url_for("auth.login_page"))
        except jwt.InvalidTokenError:
            session.pop("token", None)
            return redirect(url_for("auth.login_page"))
        return f(*args, **kwargs)
    return decorated

@auth_bp.route("/login")
def login_page():
    if session.get("token"):
        try:
            jwt.decode(session["token"], Config.SECRET_KEY, algorithms=["HS256"])
            return redirect(url_for("dashboard.dashboard_page"))
        except Exception:
            session.pop("token", None)
    return render_template("login.html")

@auth_bp.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.get_json()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    db = get_db()
    admin = db.admins.find_one({"email": email})
    if not admin:
        return jsonify({"error": "Invalid credentials"}), 401

    if not bcrypt.checkpw(password.encode("utf-8"), admin["password"]):
        return jsonify({"error": "Invalid credentials"}), 401

    # Generate and send OTP
    otp = generate_otp(email)
    send_otp_email(email, otp)

    # Store email in session for OTP verification
    session["pending_email"] = email
    session["pending_name"] = admin.get("name", "Admin")

    return jsonify({"message": "OTP sent to your email", "requires_otp": True})

@auth_bp.route("/otp")
def otp_page():
    if not session.get("pending_email"):
        return redirect(url_for("auth.login_page"))
    return render_template("otp.html", email=session["pending_email"])

@auth_bp.route("/api/auth/verify-otp", methods=["POST"])
def api_verify_otp():
    data = request.get_json()
    otp = data.get("otp", "").strip()
    email = session.get("pending_email")

    if not email:
        return jsonify({"error": "Session expired. Please login again."}), 401

    if not otp:
        return jsonify({"error": "OTP is required"}), 400

    if verify_otp(email, otp):
        name = session.pop("pending_name", "Admin")
        session.pop("pending_email", None)
        token = create_token(email, name)
        session["token"] = token

        # Log login event
        db = get_db()
        db.traffic_logs.insert_one({
            "type": "admin_login",
            "email": email,
            "timestamp": datetime.utcnow(),
            "ip": request.remote_addr
        })

        return jsonify({"message": "Login successful", "redirect": "/dashboard"})
    else:
        return jsonify({"error": "Invalid or expired OTP"}), 401

@auth_bp.route("/api/auth/resend-otp", methods=["POST"])
def api_resend_otp():
    email = session.get("pending_email")
    if not email:
        return jsonify({"error": "Session expired"}), 401

    otp = generate_otp(email)
    send_otp_email(email, otp)
    return jsonify({"message": "OTP resent successfully"})

@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login_page"))