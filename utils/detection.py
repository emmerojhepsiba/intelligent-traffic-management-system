import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from config import Config

# ── Model handles ────────────────────────────────────────────────────────────
_traffic_model = None   # YOLOv8n on COCO  – counts all vehicles
_emerg_model   = None   # Fine-tuned ambulance detection model

# COCO vehicle class ids
VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def get_traffic_model():
    global _traffic_model
    if _traffic_model is None:
        _traffic_model = YOLO(Config.YOLO_MODEL)
    return _traffic_model


def get_emergency_model():
    """Load the fine-tuned ambulance model. Falls back to color heuristics if not found."""
    global _emerg_model
    if _emerg_model is False:
        return None
    if _emerg_model is not None:
        return _emerg_model
    try:
        import os
        local = os.path.join(os.path.dirname(os.path.dirname(__file__)), Config.EMERGENCY_MODEL)
        if os.path.exists(local):
            _emerg_model = YOLO(local)
            print(f"[Detection] Loaded emergency model: {local}")
            print(f"[Detection] Emergency model classes: {_emerg_model.names}")
        else:
            print(f"[Detection] No {Config.EMERGENCY_MODEL} found — using color heuristics only.")
            _emerg_model = False
            return None
    except Exception as e:
        print(f"[Detection] Emergency model load failed: {e}")
        _emerg_model = False
        return None
    return _emerg_model


def _nms(boxes, iou_threshold=0.30, img_w=0, img_h=0):
    """Three-stage suppression to collapse duplicate detections of the same vehicle."""
    if not boxes:
        return []

    # Stage 1: standard IoU NMS
    boxes = sorted(boxes, key=lambda b: b[5], reverse=True)
    kept = []
    while boxes:
        best = boxes.pop(0)
        kept.append(best)
        boxes = [b for b in boxes if _iou(best[:4], b[:4]) < iou_threshold]

    # Stage 2: containment suppression
    def _cont(a, b):
        """Fraction of box b inside box a."""
        ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / area_b

    def _area(b):
        return max(0, (b[2] - b[0])) * max(0, (b[3] - b[1]))

    final_stage2 = []
    for i, box in enumerate(kept):
        dominated = False
        for j, other in enumerate(kept):
            if i == j:
                continue
            if other[5] >= box[5]:
                cont = _cont(other[:4], box[:4])
                size_ratio = _area(other) / max(1, _area(box))
                if cont > 0.45:
                    dominated = True; break
                if size_ratio > 3 and cont > 0.15:
                    dominated = True; break
        if not dominated:
            final_stage2.append(box)

    # Stage 3: center-distance suppression
    img_diag = ((img_w ** 2 + img_h ** 2) ** 0.5) if img_w and img_h else 0

    def _center(b):
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    def _bdiag(b):
        return ((b[2] - b[0]) ** 2 + (b[3] - b[1]) ** 2) ** 0.5

    final = []
    for box in final_stage2:
        cx, cy = _center(box)
        duplicate = False
        for kb in final:
            kx, ky = _center(kb)
            dist = ((cx - kx) ** 2 + (cy - ky) ** 2) ** 0.5
            avg_diag = (_bdiag(box) + _bdiag(kb)) / 2
            if avg_diag > 0 and dist < avg_diag * 0.55:
                duplicate = True; break
            if img_diag > 0 and dist < img_diag * 0.20:
                duplicate = True; break
        if not duplicate:
            final.append(box)

    return final


def detect_vehicles(image_path: str = None, image_array: np.ndarray = None):
    """
    Two-pass detection:
      Pass 1 – YOLOv8n (COCO)  → count all vehicles
      Pass 2 – Emergency model  → detect ambulance / fire truck / police
    Returns a safe result dict — never raises, never returns None.
    """
    # ── Input validation ────────────────────────────────────────────────────
    try:
        if image_array is not None:
            source = image_array
            img = image_array
        elif image_path:
            source = image_path
            img = cv2.imread(image_path)
        else:
            return _empty_result("No input provided")

        if img is None or img.size == 0:
            return _empty_result("Could not read image")

        img_h, img_w = img.shape[:2]

    except Exception as e:
        print(f"[Detection] Input error: {e}")
        return _empty_result(str(e))

    # ── Pass 1: COCO vehicle detection ──────────────────────────────────────
    detections = []
    vehicle_counts = {}
    total_vehicles = 0

    try:
        traffic_model = get_traffic_model()
        results = traffic_model(source, conf=Config.YOLO_CONFIDENCE, verbose=False)
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            # Clamp to image bounds
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))
            if cls_id in VEHICLE_CLASSES and x2 > x1 and y2 > y1:
                vtype = VEHICLE_CLASSES[cls_id]
                total_vehicles += 1
                vehicle_counts[vtype] = vehicle_counts.get(vtype, 0) + 1
                detections.append({
                    "type": vtype,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2],
                    "is_emergency": False,
                })
    except Exception as e:
        print(f"[Detection] COCO model error: {e}")

    # ── Pass 2: emergency vehicle detection ─────────────────────────────────
    emergency_detected = False
    emergency_count    = 0
    emergency_vehicles = []

    try:
        raw_emerg  = _run_emergency_model(source, img)
        emerg_boxes = _nms(raw_emerg, iou_threshold=0.25, img_w=img_w, img_h=img_h)

        if emerg_boxes:
            matched_det_indices = set()

            for eb in emerg_boxes:
                ex1, ey1, ex2, ey2, etype, econf = eb
                # Clamp emergency bbox to image bounds
                ex1 = max(0, min(int(ex1), img_w - 1))
                ey1 = max(0, min(int(ey1), img_h - 1))
                ex2 = max(0, min(int(ex2), img_w))
                ey2 = max(0, min(int(ey2), img_h))
                if ex2 <= ex1 or ey2 <= ey1:
                    continue
                emerg_bbox = [ex1, ey1, ex2, ey2]

                # Find the best-matching COCO detection using IoU + containment.
                # This is ONLY used to avoid double-counting in total_vehicles.
                # The emergency model's own bbox is always used for drawing.
                best_idx, best_score = -1, 0.0
                for i, det in enumerate(detections):
                    if i in matched_det_indices:
                        continue
                    iou   = _iou(det["bbox"], emerg_bbox)
                    cont  = _containment_of_a_in_b(emerg_bbox, det["bbox"])
                    score = max(iou, cont)
                    if score > best_score:
                        best_score, best_idx = score, i

                if best_score >= 0.50 and best_idx not in matched_det_indices:
                    # Good COCO match — suppress the COCO box (avoid double count)
                    # but always draw using the emergency model's own bbox
                    matched_det_indices.add(best_idx)
                    detections[best_idx]["is_emergency"]   = True
                    detections[best_idx]["emergency_type"] = etype
                    detections[best_idx]["type"]           = etype
                    detections[best_idx]["bbox"]           = emerg_bbox
                    detections[best_idx]["confidence"]     = round(float(econf), 2)
                    emergency_detected = True
                    emergency_count   += 1
                else:
                    # No confident COCO match — add emergency as its own detection
                    # Also suppress any weak COCO match to avoid double-counting
                    if best_score >= 0.25 and best_idx >= 0 and best_idx not in matched_det_indices:
                        matched_det_indices.add(best_idx)
                        detections[best_idx]["_suppressed"] = True
                    total_vehicles += 1
                    vehicle_counts[etype] = vehicle_counts.get(etype, 0) + 1
                    detections.append({
                        "type": etype,
                        "confidence": round(float(econf), 2),
                        "bbox": emerg_bbox,
                        "is_emergency": True,
                        "emergency_type": etype,
                    })
                    emergency_detected = True
                    emergency_count   += 1

            emergency_vehicles = list({
                det.get("emergency_type", "ambulance")
                for det in detections if det["is_emergency"]
            })

        else:
            # Fallback: color + shape heuristics on each vehicle ROI.
            # Pre-pass: mark detections that are heavily contained inside a larger
            # detection — these are partial/duplicate views and should be skipped
            # to avoid the same vehicle being flagged multiple times.
            def _area(b):
                return max(0, (b[2] - b[0])) * max(0, (b[3] - b[1]))

            dominated_indices = set()
            for i, di in enumerate(detections):
                for j, dj in enumerate(detections):
                    if i == j:
                        continue
                    # If di is >60% contained inside dj AND dj is larger, skip di
                    if _containment_of_a_in_b(di["bbox"], dj["bbox"]) > 0.60 and \
                            _area(dj["bbox"]) > _area(di["bbox"]):
                        dominated_indices.add(i)
                        break

            for idx, det in enumerate(detections):
                if idx in dominated_indices:
                    continue
                try:
                    x1, y1, x2, y2 = det["bbox"]
                    is_e, etype = _check_emergency_heuristic(img, x1, y1, x2, y2, det["type"])
                    if is_e:
                        det["is_emergency"]   = True
                        det["emergency_type"] = etype
                        emergency_detected    = True
                        emergency_vehicles.append(etype)
                        emergency_count      += 1
                except Exception:
                    pass

    except Exception as e:
        print(f"[Detection] Emergency detection error: {e}")

    return {
        "total_vehicles":     int(total_vehicles),
        "vehicle_counts":     {k: int(v) for k, v in vehicle_counts.items()},
        "density":            _calculate_density(total_vehicles),
        "emergency_detected": bool(emergency_detected),
        "emergency_vehicles": list(set(emergency_vehicles)),
        "emergency_count":    int(emergency_count),
        "detections":         [d for d in detections if not d.get("_suppressed")],
        "timestamp":          datetime.utcnow().isoformat(),
    }


def _empty_result(reason=""):
    """Return a safe zero-count result when detection cannot run."""
    if reason:
        print(f"[Detection] Skipped: {reason}")
    return {
        "total_vehicles":     0,
        "vehicle_counts":     {},
        "density":            "low",
        "emergency_detected": False,
        "emergency_vehicles": [],
        "emergency_count":    0,
        "detections":         [],
        "timestamp":          datetime.utcnow().isoformat(),
    }


def _run_emergency_model(source, img):
    """Run the fine-tuned emergency model. Returns list of (x1,y1,x2,y2,label,conf)."""
    try:
        model = get_emergency_model()
        if model is None:
            return []
        input_src = img if img is not None else source
        if input_src is None:
            return []
        results = model(input_src, conf=0.60, verbose=False)
        img_area = img.shape[0] * img.shape[1] if img is not None else 0
        min_area = img_area * 0.018  # box must cover at least 1.8% of image — kills tiny noise
        max_area = img_area * 0.80   # ignore huge background detections
        boxes = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label  = results[0].names[cls_id].lower().strip()
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            box_area = max(0, x2 - x1) * max(0, y2 - y1)
            if img_area > 0 and (box_area < min_area or box_area > max_area):
                continue
            boxes.append((x1, y1, x2, y2, _normalise_etype(label), float(box.conf[0])))

        # If multiple boxes found, only keep those within 15% confidence of the best
        # This removes weak false positives while preserving genuine multi-ambulance scenes
        if len(boxes) > 1:
            best_conf = max(b[5] for b in boxes)
            boxes = [b for b in boxes if b[5] >= best_conf * 0.85]

        return boxes
    except Exception as e:
        print(f"[Detection] Emergency model inference error: {e}")
        return []


def _normalise_etype(label: str) -> str:
    if "ambulance" in label:
        return "ambulance"
    if "fire" in label:
        return "fire truck"
    if "police" in label:
        return "police"
    return label


def _iou(a, b):
    """Intersection over Union for two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


def _containment_of_a_in_b(a, b):
    """Fraction of box a's area that lies inside box b."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area_a


# ── Color / shape heuristics (fallback only) ────────────────────────────────

def _check_emergency_heuristic(image, x1, y1, x2, y2, vehicle_type):
    if vehicle_type not in ("car", "truck", "bus"):
        return False, None
    try:
        roi = image[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if roi is None or roi.size == 0:
            return False, None

        total = roi.shape[0] * roi.shape[1]
        if total == 0:
            return False, None

        if _has_red_cross(roi):
            return True, "ambulance"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        red_m    = cv2.inRange(hsv, np.array([0,  130, 70]),  np.array([10, 255, 255])) | \
                   cv2.inRange(hsv, np.array([155, 130, 70]), np.array([180, 255, 255]))
        white_m  = cv2.inRange(hsv, np.array([0,   0, 195]), np.array([180,  35, 255]))
        blue_m   = cv2.inRange(hsv, np.array([100, 130, 80]), np.array([130, 255, 255]))
        yellow_m = cv2.inRange(hsv, np.array([20, 140, 100]), np.array([35,  255, 255]))
        orange_m = cv2.inRange(hsv, np.array([10, 140, 100]), np.array([20,  255, 255]))
        green_m  = cv2.inRange(hsv, np.array([40,  90,  60]), np.array([85,  255, 255]))

        r = np.sum(red_m    > 0) / total
        w = np.sum(white_m  > 0) / total
        b = np.sum(blue_m   > 0) / total
        y = np.sum(yellow_m > 0) / total
        o = np.sum(orange_m > 0) / total
        g = np.sum(green_m  > 0) / total

        top_h = max(1, int(roi.shape[0] * 0.15))
        top   = hsv[:top_h, :, :]
        tp    = top.shape[0] * top.shape[1]
        if tp > 0:
            tr = (np.sum(cv2.inRange(top, np.array([0,   160, 160]), np.array([10,  255, 255])) > 0) +
                  np.sum(cv2.inRange(top, np.array([155, 160, 160]), np.array([180, 255, 255])) > 0)) / tp
            tb = np.sum(cv2.inRange(top, np.array([100, 160, 160]), np.array([130, 255, 255])) > 0) / tp
            if tr > 0.35:
                return True, "ambulance"
            if tb > 0.35:
                return True, "police"

        if r > 0.38 or (r > 0.22 and o > 0.10):        return True, "fire truck"
        if w > 0.50 and r > 0.018:                      return True, "ambulance"  # white body + red (cross/stripe) — high white threshold avoids partial-view false positives
        if y > 0.32 and r > 0.06:                       return True, "ambulance"
        if g > 0.32 and r > 0.06:                       return True, "ambulance"
        if b > 0.22 and w > 0.28:                       return True, "police"
        if w > 0.65 and vehicle_type in ("truck", "bus"): return True, "ambulance"

    except Exception:
        pass

    return False, None


def _has_red_cross(roi):
    """Detect a red medical cross on the vehicle."""
    try:
        if roi.shape[0] < 30 or roi.shape[1] < 30:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Use sat>=100 to catch real red paint including rear-view/shaded angles
        mask = cv2.inRange(hsv, np.array([0,   100, 60]), np.array([10,  255, 255])) | \
               cv2.inRange(hsv, np.array([155, 100, 60]), np.array([180, 255, 255]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = roi.shape[0] * roi.shape[1]

        for c in cnts:
            area = cv2.contourArea(c)
            # Cross must be at least 0.4% and at most 25% of the ROI
            # (0.4% handles large close-up images where cross is small relative to vehicle)
            if area < roi_area * 0.004 or area > roi_area * 0.25:
                continue
            x, y, w, h = cv2.boundingRect(c)
            asp = w / h if h > 0 else 0
            if not (0.60 < asp < 1.60):   # slightly relaxed for perspective distortion
                continue
            sol = area / (w * h) if w * h > 0 else 0
            if not (0.40 < sol < 0.72):   # slightly relaxed solidity range
                continue
            pad = 4
            bx1 = max(0, x - pad); by1 = max(0, y - pad)
            bx2 = min(roi.shape[1], x + w + pad); by2 = min(roi.shape[0], y + h + pad)
            surround = hsv[by1:by2, bx1:bx2]
            if surround.size == 0:
                continue
            light_mask  = cv2.inRange(surround, np.array([0, 0, 140]), np.array([180, 60, 255]))
            light_ratio = np.sum(light_mask > 0) / max(1, surround.shape[0] * surround.shape[1])
            if light_ratio < 0.40:   # cross must sit on a clearly light background
                continue
            return True
    except Exception:
        pass
    return False


def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes on image. Never raises — returns original image on error."""
    try:
        annotated = image.copy()
        h, w = annotated.shape[:2]
        for det in detections:
            try:
                x1, y1, x2, y2 = det["bbox"]
                # Clamp to image bounds
                x1 = max(0, min(int(x1), w - 1))
                y1 = max(0, min(int(y1), h - 1))
                x2 = max(0, min(int(x2), w))
                y2 = max(0, min(int(y2), h))
                if x2 <= x1 or y2 <= y1:
                    continue

                is_emerg  = det.get("is_emergency", False)
                color     = (0, 0, 255) if is_emerg else (0, 255, 0)
                thickness = 3           if is_emerg else 2

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

                label = (f"EMERGENCY: {det.get('emergency_type', 'unknown')}"
                         if is_emerg else f"{det['type']} {det['confidence']:.0%}")

                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = max(lh + 10, y1)
                cv2.rectangle(annotated, (x1, label_y - lh - 10), (x1 + lw, label_y), color, -1)
                cv2.putText(annotated, label, (x1, label_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except Exception:
                continue
        return annotated
    except Exception:
        return image


def _calculate_density(n: int) -> str:
    if n <= 5:   return "low"
    if n <= 15:  return "medium"
    return "high"


def process_video_frame(frame: np.ndarray):
    return detect_vehicles(image_array=frame)
