import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "traffic-cmd-secret-key-change-in-production")
    MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/traffic_management")
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "uploads")
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB max upload
    ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}
    ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv", "webm"}
    YOLO_MODEL = "yolov8n.pt"
    YOLO_CONFIDENCE = 0.35
    # Fine-tuned emergency vehicle model (ambulance / fire truck / police)
    # Place your own trained model as "emergency.pt" in the project root,
    # or run: python download_emergency_model.py
    EMERGENCY_MODEL = "emergency.pt"

    # Mail config (for OTP)
    MAIL_SERVER = os.environ.get("MAIL_SERVER", "smtp.gmail.com")
    MAIL_PORT = int(os.environ.get("MAIL_PORT", 587))
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.environ.get("MAIL_USERNAME", "")
    MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", "")
    MAIL_DEFAULT_SENDER = os.environ.get("MAIL_DEFAULT_SENDER", "trafficcommand@system.com")

    # JWT
    JWT_EXPIRY_HOURS = 8