"""
Automated Test Suite for SIH Fire & Smoke Detection System
Validates color-space segmentation, temporal persistence filter, feature toggles,
and end-to-end detection pipeline with synthetic frames.
"""
import time
import unittest
import numpy as np
import cv2

from src.config import config, INCIDENTS_DIR
from src.detection.cv_fire_filter import CVFireSmokeFilter
from src.detection.temporal_filter import TemporalFilter
from src.detection.fire_detector import FireDetector
from src.alerts.alert_manager import alert_manager

class TestFireDetectionPipeline(unittest.TestCase):
    def setUp(self):
        # Create synthetic 480x640 frames
        self.blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Synthetic fire pattern: bright orange-yellow flame region in BGR (B=20, G=120, R=255)
        self.fire_frame = self.blank_frame.copy()
        cv2.circle(self.fire_frame, (320, 240), 45, (15, 130, 255), -1)
        # Inner flame core (yellow-white: B=80, G=220, R=255)
        cv2.circle(self.fire_frame, (320, 240), 20, (80, 220, 255), -1)
        
        # Synthetic smoke pattern: diffuse gray plume in BGR (B=160, G=160, R=160)
        self.smoke_frame = self.blank_frame.copy()
        cv2.circle(self.smoke_frame, (200, 180), 60, (160, 160, 160), -1)

    def test_01_cv_filter_fire_detection(self):
        cv_filter = CVFireSmokeFilter(min_area=100)
        # Run two consecutive frames to establish flicker baseline
        _ = cv_filter.detect(self.fire_frame, detect_smoke=False)
        # Add slight intensity variation to mimic flame flicker
        flickering_frame = self.fire_frame.copy()
        flickering_frame[230:250, 310:330] = (25, 140, 255)
        detections = cv_filter.detect(flickering_frame, detect_smoke=False)
        
        fire_dets = [d for d in detections if d["label"] == "fire"]
        self.assertTrue(len(fire_dets) > 0, "Fire region should be detected by CV filter")
        self.assertGreaterEqual(fire_dets[0]["confidence"], 0.40)
        print(f"[+] test_01 passed: Fire detected with confidence {fire_dets[0]['confidence']}")

    def test_02_temporal_filter_state_progression(self):
        t_filter = TemporalFilter(persistence_frames=3)
        mock_det = [{"box": (10, 10, 50, 50), "label": "fire", "confidence": 0.8}]
        
        # Frame 1: Should enter VERIFYING
        state1, _ = t_filter.update(mock_det)
        self.assertEqual(state1, "VERIFYING")
        
        # Frame 2: Should still be VERIFYING
        state2, _ = t_filter.update(mock_det)
        self.assertEqual(state2, "VERIFYING")
        
        # Frame 3: Reaches persistence threshold -> ALARM
        state3, dets3 = t_filter.update(mock_det)
        self.assertEqual(state3, "ALARM")
        self.assertEqual(dets3[0]["status"], "CONFIRMED")
        print("[+] test_02 passed: Temporal state progression verified (NORMAL -> VERIFYING -> ALARM)")

    def test_03_smoke_toggle(self):
        detector = FireDetector(mode="CV_ONLY", enable_smoke=True)
        # With smoke enabled
        _, dets_enabled, _ = detector.process_frame(self.smoke_frame)
        smoke_count_enabled = sum(1 for d in dets_enabled if d["label"] == "smoke")
        
        # Disable smoke
        detector.set_smoke_detection(False)
        _, dets_disabled, _ = detector.process_frame(self.smoke_frame)
        smoke_count_disabled = sum(1 for d in dets_disabled if d["label"] == "smoke")
        
        self.assertEqual(smoke_count_disabled, 0, "When smoke is disabled, zero smoke detections should appear")
        print(f"[+] test_03 passed: Smoke toggle effectively suppressed smoke detections")

    def test_04_alert_manager_snapshot_logging(self):
        # Test manual snapshot writing
        test_frame = self.fire_frame.copy()
        mock_dets = [{"box": (300, 220, 40, 40), "label": "fire", "confidence": 0.88}]
        
        # Trigger alert with snapshots enabled
        alert_manager.set_snapshots(True)
        alert_manager.trigger_alert(test_frame, mock_dets, "ALARM")
        
        # Verify file creation in incidents/
        files = list(INCIDENTS_DIR.glob("incident_*_fire.jpg"))
        self.assertTrue(len(files) > 0, "Incident snapshot file should be created on disk")
        print(f"[+] test_04 passed: Incident snapshot logged successfully to {files[0].name}")

if __name__ == "__main__":
    unittest.main()
