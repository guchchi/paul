"""
Temporal Persistence Filter
Tracks sliding window of detections over consecutive frames to eliminate false positives.
Distinguishes transient artifacts (flashes, orange garments, camera noise) from real continuous combustion.
"""
import time
from collections import deque
from typing import List, Dict, Any, Tuple

class TemporalFilter:
    def __init__(self, persistence_frames: int = 5, history_length: int = 15):
        self.persistence_frames = persistence_frames
        self.history = deque(maxlen=history_length)
        self.consecutive_fire_count = 0
        self.consecutive_smoke_count = 0
        self.state = "NORMAL"  # "NORMAL", "VERIFYING", "ALARM"
        self.last_alarm_time = 0.0
        
    def update(self, detections: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Updates the tracker with frame detections.
        Returns:
            (state, validated_detections)
            state can be 'NORMAL', 'VERIFYING', or 'ALARM'
        """
        has_fire = any(d["label"] == "fire" for d in detections)
        has_smoke = any(d["label"] == "smoke" for d in detections)
        
        # Track consecutive counts
        if has_fire:
            self.consecutive_fire_count += 1
        else:
            self.consecutive_fire_count = max(0, self.consecutive_fire_count - 1)
            
        if has_smoke:
            self.consecutive_smoke_count += 1
        else:
            self.consecutive_smoke_count = max(0, self.consecutive_smoke_count - 1)
            
        self.history.append({"has_fire": has_fire, "has_smoke": has_smoke, "time": time.time()})
        
        # Determine system alarm state
        active_count = max(self.consecutive_fire_count, self.consecutive_smoke_count)
        
        if active_count >= self.persistence_frames:
            self.state = "ALARM"
        elif active_count > 0:
            self.state = "VERIFYING"
        else:
            self.state = "NORMAL"
            
        # Filter detections with stability confidence boost if persistent
        validated = []
        for det in detections:
            det_copy = dict(det)
            if self.state == "ALARM":
                det_copy["confidence"] = min(0.99, det_copy["confidence"] + 0.15)
                det_copy["status"] = "CONFIRMED"
            elif self.state == "VERIFYING":
                det_copy["status"] = "VERIFYING"
            else:
                det_copy["status"] = "PROBABLE"
            validated.append(det_copy)
            
        return self.state, validated

    def reset(self):
        self.consecutive_fire_count = 0
        self.consecutive_smoke_count = 0
        self.state = "NORMAL"
        self.history.clear()
