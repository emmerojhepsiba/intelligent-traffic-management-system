import os
from flask import Flask, render_template
from flask_socketio import SocketIO
from flask_cors import CORS
from config import Config
from utils.db import init_db, seed_default_admin

socketio = SocketIO()

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config["SESSION_TYPE"] = "filesystem"

    CORS(app)

    # Ensure upload directory exists
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

    # Initialize database
    init_db(app)
    seed_default_admin()

    # Initialize SocketIO
    socketio.init_app(app, cors_allowed_origins="*", async_mode="threading")

    # Register blueprints
    from routes.auth import auth_bp
    from routes.dashboard import dashboard_bp
    from routes.detection import detection_bp
    from routes.search import search_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(detection_bp)
    app.register_blueprint(search_bp)

    # Home page
    @app.route("/")
    def home():
        return render_template("index.html")

    # WebSocket events
    @socketio.on("connect")
    def handle_connect():
        print("[WS] Client connected")

    @socketio.on("disconnect")
    def handle_disconnect():
        print("[WS] Client disconnected")

    @socketio.on("signal_update")
    def handle_signal_update(data):
        socketio.emit("signal_changed", data, broadcast=True)

    @socketio.on("emergency_alert")
    def handle_emergency(data):
        socketio.emit("emergency_notification", data, broadcast=True)

    return app

if __name__ == "__main__":
    app = create_app()
    print("\n" + "=" * 55)
    print("  TrafficCommand - Smart Traffic Management System")
    print("  Running at: http://localhost:5000")
    print("=" * 55 + "\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)