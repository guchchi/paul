"""
Alert and Incident Response Manager
Handles non-blocking audio siren alarms, pyttsx3 voice announcements,
and automated incident snapshot evidence logging.
"""
import os
import time
import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2

from src.config import config, INCIDENTS_DIR

class AlertManager:
    def __init__(self):
        self.enable_audio = config.enable_audio_alarm
        self.enable_voice = config.enable_voice_alert
        self.enable_snapshots = config.enable_snapshots
        
        self.last_alarm_time = 0.0
        self.last_voice_time = 0.0
        self.last_snapshot_time = 0.0
        
        self.is_sounding_alarm = False
        self.alert_queue = queue.Queue(maxsize=10)
        
        # Incident history (in-memory log of captured events)
        self.incident_history: List[Dict[str, Any]] = []
        
        # Start background worker thread for audio & voice alerts
        self._worker_thread = threading.Thread(target=self._alert_worker, daemon=True)
        self._worker_thread.start()
        
    def set_audio(self, enabled: bool):
        self.enable_audio = enabled
        print(f"[+] Audio siren alerts: {'ENABLED' if enabled else 'DISABLED'}")
        
    def set_voice(self, enabled: bool):
        self.enable_voice = enabled
        print(f"[+] Voice announcements: {'ENABLED' if enabled else 'DISABLED'}")
        
    def set_snapshots(self, enabled: bool):
        self.enable_snapshots = enabled
        print(f"[+] Incident snapshots: {'ENABLED' if enabled else 'DISABLED'}")

    def trigger_alert(self, frame: cv2.Mat, detections: List[Dict[str, Any]], system_state: str):
        """
        Invoked on each frame when an alert condition occurs.
        Respects cooldown timers to prevent alert flooding.
        """
        if system_state != "ALARM" or not detections:
            return
            
        current_time = time.time()
        labels = [d["label"] for d in detections]
        primary_threat = "fire" if "fire" in labels else "smoke"
        max_conf = max(d.get("confidence", 0.5) for d in detections)
        
        # 1. Schedule Audio / Voice via background thread
        if (current_time - self.last_alarm_time) >= config.alarm_cooldown_seconds:
            self.last_alarm_time = current_time
            try:
                self.alert_queue.put_nowait({
                    "type": primary_threat,
                    "confidence": max_conf,
                    "timestamp": current_time
                })
            except queue.Full:
                pass
                
        # 2. Automated Incident Snapshot Logging
        if self.enable_snapshots and (current_time - self.last_snapshot_time) >= config.alarm_cooldown_seconds:
            self.last_snapshot_time = current_time
            self._save_incident_snapshot(frame, primary_threat, max_conf)

    def trigger(self, hazard_type: str = "fire", conf: float = 0.0, frame: cv2.Mat = None):
        """Dashboard compatibility method for triggering alarms."""
        if frame is None:
            return
        detections = [{"label": hazard_type, "confidence": conf, "box": (0, 0, 10, 10)}]
        self.trigger_alert(frame, detections, "ALARM")

    def _alert_worker(self):
        """Background worker thread to handle sound and speech without blocking video FPS."""
        # Initialize pyttsx3 engine safely inside worker
        tts_engine = None
        try:
            import pyttsx3
            tts_engine = pyttsx3.init()
            tts_engine.setProperty("rate", 160)
        except Exception as e:
            print(f"[!] Voice TTS initialization notice: {e}")
            
        while True:
            try:
                alert_item = self.alert_queue.get()
                threat = alert_item.get("type", "fire")
                
                # A. Sound Siren Alarm (Windows winsound)
                if self.enable_audio:
                    try:
                        import winsound
                        # Two-tone hazard warble
                        for freq in (1200, 1800, 1400, 2000):
                            winsound.Beep(freq, 120)
                    except Exception:
                        pass
                        
                # B. Voice Announcement
                if self.enable_voice and tts_engine:
                    try:
                        msg = f"Warning! {threat.capitalize()} hazard detected! Please inspect camera zone immediately."
                        tts_engine.say(msg)
                        tts_engine.runAndWait()
                    except Exception as e:
                        pass
                        
                self.alert_queue.task_done()
            except Exception as e:
                time.sleep(0.1)

    def _save_incident_snapshot(self, frame: cv2.Mat, threat_type: str, confidence: float):
        """Saves a timestamped snapshot of the incident with visual telemetry stamped."""
        try:
            now = datetime.now()
            timestamp_str = now.strftime("%Y%m%d_%H%M%S")
            filename = f"incident_{timestamp_str}_{threat_type}.jpg"
            filepath = INCIDENTS_DIR / filename
            
            # Create a copy with visual annotation
            annotated = frame.copy()
            h, w = annotated.shape[:2]
            
            # Overlay banner
            cv2.rectangle(annotated, (0, h - 45), (w, h), (0, 0, 0), -1)
            watermark = f"SIH ALERT | {threat_type.upper()} DETECTED | Conf: {confidence:.2f} | {now.strftime('%Y-%m-%d %H:%M:%S')}"
            cv2.putText(annotated, watermark, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255) if threat_type == 'fire' else (200, 200, 200), 2)
            
            cv2.imwrite(str(filepath), annotated)
            print(f"[+] Incident evidence recorded: {filepath}")
            
            # Add to in-memory history log
            self.incident_history.insert(0, {
                "filename": filename,
                "filepath": str(filepath),
                "timestamp": now.strftime("%H:%M:%S"),
                "date": now.strftime("%Y-%m-%d"),
                "threat": threat_type,
                "confidence": confidence
            })
        except Exception as e:
            print(f"[!] Error saving snapshot: {e}")

# Global AlertManager instance
alert_manager = AlertManager()
