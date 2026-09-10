"""
Unified Fire & Smoke Detector Interface
Integrates Deep Learning (YOLO) with Classical CV Dynamics and Temporal Filtering.
Supports dynamic toggles for Smoke Detection, engine selection, and sensitivity tuning.
"""
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2

from src.detection.cv_fire_filter import CVFireSmokeFilter
from src.detection.temporal_filter import TemporalFilter
from src.config import config

class FireDetector:
    def __init__(self, mode: str = None, enable_smoke: bool = None):
        self.mode = mode or config.engine_mode
        self.enable_smoke = config.enable_smoke if enable_smoke is None else enable_smoke
        
        # Classical CV Filter (always available)
        self.cv_filter = CVFireSmokeFilter(min_area=config.min_fire_area_pixels)
        
        # Temporal persistence filter
        self.temporal_filter = TemporalFilter(persistence_frames=config.persistence_frames)
        
        # YOLO model (lazy loaded if available)
        self.yolo_model = None
        self.yolo_initialized = False
        self._init_yolo()
        
        # Performance tracking
        self.last_process_time = time.time()
        self.current_fps = 0.0
        
    def _init_yolo(self):
        """Attempts to load YOLO model if ultralytics is available and weights exist."""
        try:
            from ultralytics import YOLO
            model_path = Path(config.yolo_model_path)
            
            # Check for customized fire weights or fallback to standard weights
            if model_path.exists():
                print(f"[+] Loading Fire Detection YOLO model from: {model_path}")
                self.yolo_model = YOLO(str(model_path))
                self.yolo_initialized = True
            else:
                # If custom fire weights not found, we can load yolov8n base or let CV handle it
                print(f"[*] Custom YOLO weights not at {model_path}. Using high-precision CV Dynamics Engine.")
                self.yolo_model = None
                self.yolo_initialized = False
        except ImportError:
            print("[*] Ultralytics not installed. Defaulting to high-performance CV Dynamics Engine.")
            self.yolo_model = None
            self.yolo_initialized = False
        except Exception as e:
            print(f"[!] Warning initializing YOLO: {e}. Using CV Dynamics Engine.")
            self.yolo_model = None
            self.yolo_initialized = False

    def set_smoke_detection(self, enabled: bool):
        """Dynamic toggle for smoke detection."""
        self.enable_smoke = enabled
        print(f"[+] Smoke detection toggled to: {'ENABLED' if enabled else 'DISABLED'}")

    def set_persistence_frames(self, frames: int):
        """Adjust temporal persistence sensitivity."""
        self.temporal_filter.persistence_frames = max(1, frames)

    def process_frame(self, frame: cv2.Mat) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        Processes a single camera frame.
        Returns:
            (system_state, detections_list, fps)
        """
        start_t = time.time()
        raw_detections = []
        
        # 1. Classical CV Filter Analysis
        if self.mode in ("CV_ONLY", "HYBRID") or not self.yolo_initialized:
            cv_dets = self.cv_filter.detect(frame, detect_smoke=self.enable_smoke)
            raw_detections.extend(cv_dets)
            
        # 2. YOLO Deep Learning Analysis (if initialized and active)
        if self.yolo_initialized and self.mode in ("DL_ONLY", "HYBRID"):
            try:
                results = self.yolo_model.predict(
                    source=frame,
                    conf=config.confidence_threshold,
                    verbose=False,
                    imgsz=480
                )
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = r.names.get(cls_id, str(cls_id)).lower()
                        conf = float(box.conf[0])
                        
                        # Map classes
                        if "fire" in cls_name or "flame" in cls_name:
                            label = "fire"
                        elif "smoke" in cls_name:
                            if not self.enable_smoke:
                                continue
                            label = "smoke"
                        else:
                            continue
                            
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        raw_detections.append({
                            "box": (x1, y1, x2 - x1, y2 - y1),
                            "label": label,
                            "confidence": round(conf, 2),
                            "source": "yolo"
                        })
            except Exception as e:
                pass
                
        # 3. Apply Smoke Toggle Filter (in case any raw detection had smoke)
        if not self.enable_smoke:
            raw_detections = [d for d in raw_detections if d["label"] != "smoke"]
            
        # 4. Filter by confidence threshold
        filtered_detections = [
            d for d in raw_detections 
            if d.get("confidence", 0.0) >= config.confidence_threshold
        ]
        
        # 5. Temporal Persistence Verification
        system_state, validated_detections = self.temporal_filter.update(filtered_detections)
        
        # Calculate instantaneous FPS
        elapsed = time.time() - start_t
        self.current_fps = round(1.0 / elapsed, 1) if elapsed > 0 else 30.0
        
        return system_state, validated_detections, self.current_fps

    def detect(self, frame: cv2.Mat) -> Tuple[cv2.Mat, List[Dict[str, Any]], str]:
        """
        High-level detection API used by the dashboard.
        Processes the frame, draws bounding boxes, and returns:
            (annotated_frame, detections_list, system_state)
        """
        system_state, detections, _ = self.process_frame(frame)

        annotated = frame.copy()

        # Color map for detection labels
        color_map = {
            "fire":  (0, 0, 255),    # Red (BGR)
            "smoke": (200, 200, 0),  # Cyan-ish (BGR)
        }

        for det in detections:
            box = det.get("box")
            if not box:
                continue
            x, y, w, h = box
            label = det.get("label", "fire")
            conf = det.get("confidence", 0.0)
            color = color_map.get(label, (0, 255, 255))

            # Bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Label background + text
            text = f"{label.upper()} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x, y - th - 6), (x + tw + 4, y), color, -1)
            cv2.putText(annotated, text, (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated, detections, system_state
