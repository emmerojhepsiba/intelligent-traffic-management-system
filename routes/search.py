from datetime import datetime
from flask import Blueprint, request, jsonify, render_template
from utils.db import get_db

search_bp = Blueprint("search", __name__)

# Simulated traffic data for demonstration
TRAFFIC_DATA = {
    "mangamur road, ongole": {"lat": 15.5057, "lng": 80.0499, "base_density": "medium"},
    "kurnool road, ongole": {"lat": 15.5100, "lng": 80.0450, "base_density": "low"},
    "trunk road, ongole": {"lat": 15.5020, "lng": 80.0520, "base_density": "high"},
    "court center, ongole": {"lat": 15.5035, "lng": 80.0480, "base_density": "medium"},
    "bhagya nagar, ongole": {"lat": 15.5080, "lng": 80.0510, "base_density": "low"},
    "main street": {"lat": 15.5045, "lng": 80.0495, "base_density": "medium"},
    "highway 101": {"lat": 15.5120, "lng": 80.0550, "base_density": "high"},
    "station road": {"lat": 15.5000, "lng": 80.0470, "base_density": "medium"},
}

@search_bp.route("/search")
def search_page():
    return render_template("search.html")

@search_bp.route("/api/search/traffic", methods=["POST"])
def search_traffic():
    data = request.get_json()
    location = data.get("location", "").strip()

    if not location:
        return jsonify({"error": "Location is required"}), 400

    # Try to match location in our data
    location_lower = location.lower()
    matched = None
    for key, val in TRAFFIC_DATA.items():
        if key in location_lower or location_lower in key:
            matched = {"name": key.title(), **val}
            break

    if not matched:
        # Default coordinates (Ongole center) for unknown locations
        matched = {
            "name": location,
            "lat": 15.5057,
            "lng": 80.0499,
            "base_density": "medium"
        }

    # Predict density based on time
    now = datetime.now()
    hour = now.hour
    day = now.weekday()  # 0=Monday, 6=Sunday

    density = _predict_density(matched["base_density"], hour, day)

    # Check DB for recent detection data at this location
    db = get_db()

    # Aggregate all detections for this location (last 7 days)
    from datetime import timedelta
    since = datetime.utcnow() - timedelta(days=7)
    recent_detections = list(db.detections.find(
        {
            "location": {"$regex": location, "$options": "i"},
            "timestamp": {"$gte": since}
        },
        sort=[("timestamp", -1)],
        limit=20
    ))

    real_data = None
    if recent_detections:
        # Map density to number for averaging
        density_map = {"low": 1, "medium": 2, "high": 3}
        reverse_map = {1: "low", 2: "medium", 3: "high"}
        scores = [density_map.get(d.get("density", "low"), 1) for d in recent_detections]
        avg_score = round(sum(scores) / len(scores))
        density = reverse_map.get(avg_score, "medium")

        total_vehicles_avg = round(
            sum(d.get("total_vehicles", 0) for d in recent_detections) / len(recent_detections)
        )
        emergency_count = sum(1 for d in recent_detections if d.get("emergency_detected"))
        last_updated = recent_detections[0].get("timestamp")

        real_data = {
            "detections_used": len(recent_detections),
            "avg_vehicles": total_vehicles_avg,
            "emergency_count": emergency_count,
            "last_updated": last_updated.isoformat() if last_updated else None,
        }
    else:
        # No real data — fall back to time-based prediction
        density = _predict_density(matched["base_density"], hour, day)

    result = {
        "location": matched["name"],
        "lat": matched["lat"],
        "lng": matched["lng"],
        "density": density,
        "hour": hour,
        "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day],
        "suggestion": _get_suggestion(density),
        "timestamp": datetime.utcnow().isoformat(),
        "data_source": "real" if real_data else "predicted",
        "real_data": real_data,
    }

    # Log search
    db.traffic_logs.insert_one({
        "type": "public_search",
        "location": location,
        "density": density,
        "timestamp": datetime.utcnow()
    })

    return jsonify(result)

@search_bp.route("/api/search/history")
def get_area_history():
    """Get traffic density history for a location."""
    location = request.args.get("location", "")
    db = get_db()
    logs = list(
        db.traffic_logs.find(
            {"type": "public_search", "location": {"$regex": location, "$options": "i"}},
            {"_id": 0}
        ).sort("timestamp", -1).limit(20)
    )
    return jsonify(logs)

def _predict_density(base: str, hour: int, day: int) -> str:
    """Predict traffic density based on time patterns."""
    # Peak hours
    is_morning_peak = 8 <= hour <= 11
    is_evening_peak = 17 <= hour <= 21
    is_weekend = day >= 5

    if is_weekend:
        if is_evening_peak:
            return "medium" if base == "low" else base
        return base

    if is_morning_peak or is_evening_peak:
        if base == "low":
            return "medium"
        elif base == "medium":
            return "high"
        else:
            return "high"
    elif 0 <= hour <= 5:
        return "low"
    else:
        return base

def _get_suggestion(density: str) -> str:
    if density == "high":
        return "Heavy traffic detected. Consider alternate routes or delay your travel."
    elif density == "medium":
        return "Moderate traffic. Normal travel time expected with slight delays."
    else:
        return "Light traffic. Roads are clear for travel."