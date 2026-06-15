"""
Ambulance model retrainer.
Downloads dataset from Roboflow using SDK, trains YOLOv8s, replaces emergency.pt

Usage:
    pip install roboflow ultralytics pyyaml
    python retrain.py
"""
import os, shutil, yaml, zipfile, io, requests, time

DATASET_DIR  = "ambulance_dataset"
OUTPUT_MODEL = "emergency.pt"

ROBOFLOW_API_KEY = "MwdibrTjxfHcaS6VcV8L"
ROBOFLOW_PROJECT = "ambulance-r71d0-zgfnq"
ROBOFLOW_VERSION = 1

# ── Step 1: Download via Roboflow REST API ────────────────────────────────────
print("\n[1/4] Downloading dataset via Roboflow API...")

import requests, zipfile, io

def _get_workspace(api_key):
    """Resolve the workspace slug for this API key."""
    r = requests.get(f"https://api.roboflow.com/?api_key={api_key}", timeout=30)
    r.raise_for_status()
    data = r.json()
    print(f"     API root keys: {list(data.keys())}")
    # Try common locations for workspace slug
    if isinstance(data.get("workspace"), dict):
        ws = data["workspace"]
        return ws.get("handle") or ws.get("url") or ws.get("name")
    # Sometimes it's a flat string or nested differently
    if isinstance(data.get("workspace"), str):
        return data["workspace"]
    # Fall back: look for any key that looks like a workspace handle
    for key in ("handle", "url", "slug"):
        if key in data:
            return data[key]
    return None

try:
    print("     Resolving workspace...")
    workspace = _get_workspace(ROBOFLOW_API_KEY)
    if not workspace:
        raise ValueError("Could not resolve workspace from API key")
    print(f"     Workspace: {workspace}")

    # Request a YOLOv8 export
    export_url = (
        f"https://api.roboflow.com/{workspace}/{ROBOFLOW_PROJECT}/{ROBOFLOW_VERSION}/yolov8"
        f"?api_key={ROBOFLOW_API_KEY}"
    )
    print(f"     Requesting export (may take up to 60s to generate)...")
    er = requests.get(export_url, timeout=60)
    er.raise_for_status()
    export_data = er.json()

    # Poll until the export link is ready (Roboflow generates it async)
    link = (export_data.get("export") or {}).get("link") or export_data.get("link")
    if not link:
        print(f"     Export response: {export_data}")
        raise ValueError("No download link returned — check project name and version.")

    # Retry the ZIP download — it may take a few seconds to appear on GCS
    print(f"     Waiting for export to be ready...")
    zr = None
    for attempt in range(8):
        zr = requests.get(link, stream=True, timeout=120)
        if zr.status_code == 200:
            break
        print(f"     Not ready yet (attempt {attempt+1}/8), waiting 10s...")
        time.sleep(10)
    zr.raise_for_status()

    # Verify it's a ZIP
    content = zr.content
    if not content[:4] == b'PK\x03\x04':
        raise ValueError("Downloaded file is not a ZIP — export may still be processing. Wait 1 min and retry.")

    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    os.makedirs(DATASET_DIR)

    with zipfile.ZipFile(io.BytesIO(content)) as z:
        z.extractall(DATASET_DIR)
    print(f"     Extracted to {DATASET_DIR}/")

except Exception as e:
    print(f"\n     Download failed: {e}")
    print("\n     ── MANUAL FALLBACK ──")
    print("     1. Log in to https://roboflow.com")
    print(f"     2. Open project: {ROBOFLOW_PROJECT}")
    print("     3. Versions → Export → YOLOv8 format → Download ZIP")
    print(f"     4. Extract into folder named '{DATASET_DIR}' inside C:\\Traffic 1\\")
    print("     5. Re-run: python retrain.py")
    raise SystemExit(1)

# ── Step 2: Find data.yaml ────────────────────────────────────────────────────
print("\n[2/4] Locating data.yaml...")
data_yaml = None
for root, dirs, files in os.walk(DATASET_DIR):
    for f in files:
        if f == "data.yaml":
            data_yaml = os.path.join(root, f)
            break

if not data_yaml:
    raise FileNotFoundError("data.yaml not found in downloaded dataset!")

print(f"     Found: {data_yaml}")

with open(data_yaml, 'r') as f:
    cfg = yaml.safe_load(f)

dataset_root = os.path.dirname(os.path.abspath(data_yaml))
print(f"     Dataset root : {dataset_root}")
print(f"     Classes      : {cfg.get('names')}")
print(f"     Num classes  : {cfg.get('nc')}")

# Fix paths to absolute so YOLO can find them regardless of cwd
cfg['path'] = dataset_root
with open(data_yaml, 'w') as f:
    yaml.dump(cfg, f)

# ── Step 3: Train ─────────────────────────────────────────────────────────────
print("\n[3/4] Training YOLOv8s (10-30 mins on GPU)...")
print("      yolov8s.pt — larger and more accurate than the original yolov8n.pt\n")

from ultralytics import YOLO
model = YOLO("yolov8s.pt")

results = model.train(
    data=data_yaml,
    epochs=80,
    imgsz=640,
    batch=4,
    patience=20,
    device='cpu',
    project="models",
    name="ambulance_retrain",
    exist_ok=True,
    hsv_h=0.02, hsv_s=0.7, hsv_v=0.4,
    fliplr=0.5, mosaic=1.0, mixup=0.1,
    degrees=5.0, translate=0.1, scale=0.5,
)

# ── Step 4: Save best weights ─────────────────────────────────────────────────
print("\n[4/4] Saving model...")
best = "models/ambulance_retrain/weights/best.pt"
if os.path.exists(best):
    if os.path.exists(OUTPUT_MODEL):
        shutil.copy(OUTPUT_MODEL, "emergency_backup.pt")
        print(f"     Old model backed up → emergency_backup.pt")
    shutil.copy(best, OUTPUT_MODEL)
    print(f"     New model saved → {OUTPUT_MODEL}")
else:
    print(f"     WARNING: best.pt not found at {best}")

# ── Final metrics ─────────────────────────────────────────────────────────────
print("\n=== Training Complete ===")
m = results.results_dict
print(f"  Precision : {m.get('metrics/precision(B)', 0):.3f}")
print(f"  Recall    : {m.get('metrics/recall(B)', 0):.3f}")
print(f"  mAP@50    : {m.get('metrics/mAP50(B)', 0):.3f}")
print(f"  mAP@50-95 : {m.get('metrics/mAP50-95(B)', 0):.3f}")
print("\nNext steps:")
print("  1. Restart Flask server to load the new model")
print("  2. Run: python test_nms.py")
print("  3. Delete ambulance_dataset/ and emergency_backup.pt when satisfied")
