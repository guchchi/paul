"""
Model Weights Downloader & Setup Utility
Downloads lightweight pretrained YOLO weights for fire/smoke detection.
"""
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_URLS = {
    # Standard official YOLOv8 nano model as base/benchmark
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
}

def download_weights(target_filename="yolov8n.pt"):
    target_path = MODELS_DIR / target_filename
    if target_path.exists():
        print(f"[+] Model weights already exist at: {target_path}")
        return str(target_path)
        
    url = WEIGHTS_URLS.get(target_filename)
    if not url:
        print(f"[!] No direct URL configured for {target_filename}.")
        return None
        
    print(f"[*] Downloading {target_filename} from {url}...")
    try:
        urllib.request.urlretrieve(url, str(target_path))
        print(f"[+] Successfully downloaded: {target_path}")
        return str(target_path)
    except Exception as e:
        print(f"[!] Download failed: {e}")
        return None

if __name__ == "__main__":
    download_weights()
