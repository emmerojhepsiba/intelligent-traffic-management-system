import cv2
import base64
import numpy as np
from flask import Blueprint, request, jsonify
from routes.auth import login_required
from utils.detection import detect_vehicles, draw_detections

detection_bp = Blueprint("detection", __name__)


@detection_bp.route("/api/detect/frame", methods=["POST"])
@login_required
def detect_frame():
    """Process a single frame from webcam (base64 encoded)."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data received"}), 400

        frame_data = data.get("frame")
        location   = data.get("location", "Live Camera Feed")

        if not frame_data:
            return jsonify({"error": "No frame data"}), 400

        # Decode base64 image
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]

        try:
            img_bytes = base64.b64decode(frame_data)
        except Exception:
            return jsonify({"error": "Invalid base64 frame data"}), 400

        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            return jsonify({"error": "Could not decode frame"}), 400

        result    = detect_vehicles(image_array=frame)
        annotated = draw_detections(frame, result["detections"])

        _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        result["annotated_frame"] = f"data:image/jpeg;base64,{annotated_b64}"
        result["location"]        = location

        return jsonify(result)

    except Exception as e:
        print(f"[Detection] Frame detection error: {e}")
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500
