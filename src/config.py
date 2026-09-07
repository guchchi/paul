"""
Central Configuration for SIH Fire & Smoke Detection System
"""
import os
from dataclasses import dataclass
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
INCIDENTS_DIR = BASE_DIR / "incidents"
MODELS_DIR = BASE_DIR / "models"

# Ensure runtime directories exist
INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class AppConfig:
    # Camera Settings
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps_limit: int = 30
    
    # Feature Toggles (Requested by User)
    enable_smoke: bool = True
    enable_audio_alarm: bool = True
    enable_voice_alert: bool = True
    enable_snapshots: bool = True
    
    # Detection Thresholds
    confidence_threshold: float = 0.40
    min_fire_area_pixels: int = 300
    persistence_frames: int = 5          # Consecutive positive frames before raising alarm
    alarm_cooldown_seconds: float = 6.0  # Cooldown between voice/incident snapshots
    
    # Engine Mode: "HYBRID", "DL_ONLY", "CV_ONLY"
    engine_mode: str = "HYBRID"
    
    # Model Weights (Auto-checked in models/)
    yolo_model_path: str = str(MODELS_DIR / "yolov8n_fire.pt")
    
    # Alert Dispatchers (Optional integrations)
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    serial_port: str = ""                # e.g., "COM3" for Arduino/Relay
    serial_baud: int = 9600

# Global active configuration instance
config = AppConfig()
