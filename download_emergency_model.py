"""
Downloads a YOLOv8 emergency vehicle model from Hugging Face.
Run once:  python download_emergency_model.py
"""
import urllib.request
import os
import sys

OUTPUT = "emergency.pt"

# Working models from Hugging Face Hub (ambulance / fire truck / police)
SOURCES = [
    {
        "url": "https://huggingface.co/keremberke/yolov8n-emergency-vehicle-detection/resolve/main/best.pt",
        "desc": "keremberke/yolov8n-emergency-vehicle-detection (HuggingFace)"
    },
    {
        "url": "https://huggingface.co/keremberke/yolov8s-emergency-vehicle-detection/resolve/main/best.pt",
        "desc": "keremberke/yolov8s-emergency-vehicle-detection (HuggingFace)"
    },
]


def download(url, dest):
    print(f"  Trying: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = r.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Progress: {pct:.1f}%  ({downloaded//1024} KB)", end="", flush=True)
        print()
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  Saved: {dest}  ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"\n  Failed: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


if __name__ == "__main__":
    if os.path.exists(OUTPUT):
        print(f"'{OUTPUT}' already exists. Delete it to re-download.")
        sys.exit(0)

    print("Downloading emergency vehicle detection model...\n")
    for src in SOURCES:
        print(f"Source: {src['desc']}")
        if download(src["url"], OUTPUT):
            print(f"\nDone. Model saved as '{OUTPUT}'")
            print("Restart your Flask app — it will load automatically.")
            sys.exit(0)

    print("\n" + "="*60)
    print("All automatic downloads failed.")
    print("="*60)
    print("""
Manual steps (2 minutes):

1. Open this URL in your browser:
   https://huggingface.co/keremberke/yolov8n-emergency-vehicle-detection

2. Click the 'Files and versions' tab

3. Download 'best.pt'

4. Rename it to 'emergency.pt' and place it in:
   """ + os.path.abspath(".") + """

5. Restart the Flask app.

Classes detected: ambulance, fire truck, police car
""")
