import sys; sys.path.insert(0, '.')
from utils.detection import detect_vehicles, _run_emergency_model, _nms
import cv2, os

folder = 'static/uploads'
images = [f for f in os.listdir(folder)
          if not f.startswith('det_') and f.lower().endswith(('.jpeg','.jpg','.png','.webp'))]

all_ok = True
for img_file in sorted(images):
    path = os.path.join(folder, img_file)
    img = cv2.imread(path)
    h, w = img.shape[:2]
    raw = _run_emergency_model(path, img)
    after_nms = _nms(raw, img_w=w, img_h=h)
    r = detect_vehicles(image_path=path)
    false_pos = [d for d in r['detections'] if d['is_emergency'] and
                 d.get('emergency_type', d['type']) not in ('ambulance', 'fire truck', 'police')]
    ok = r['emergency_count'] <= max(len(after_nms), 1) and len(false_pos) == 0
    status = "OK" if ok else "ISSUE"
    if not ok: all_ok = False
    fp_note = f" FALSE_POS={[d['type'] for d in false_pos]}" if false_pos else ""
    print(f"[{status}] {img_file}  raw={len(raw)} nms={len(after_nms)} count={r['emergency_count']}{fp_note}")

print("\nAll OK!" if all_ok else "\nIssues found!")
