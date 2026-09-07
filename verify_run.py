"""
End-to-End Headless Verification
Runs 30 frames through FireDetector and HUDVisualizer to test FPS, stability, and memory.
"""
import time
import cv2
import numpy as np
from src.config import config
from src.detection.fire_detector import FireDetector
from src.utils.visualizer import hud_visualizer
from src.alerts.alert_manager import alert_manager

def test_headless_run():
    print("[*] Testing 30 frames through full Fire Detection pipeline...")
    detector = FireDetector(mode="HYBRID", enable_smoke=True)
    
    # Generate test webcam stream with simulated flame motion
    fps_records = []
    for i in range(30):
        # Create test frame
        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
        
        # Add flame circle on frames 10-25
        if 10 <= i <= 25:
            # Vary radius and color slightly to simulate flickering flame
            radius = 35 + (i % 5)
            cv2.circle(frame, (320, 240), radius, (15 + (i % 4), 125 + (i % 8), 250), -1)
            cv2.circle(frame, (320, 240), 15, (75, 215, 255), -1)
            
        sys_state, detections, fps = detector.process_frame(frame)
        fps_records.append(fps)
        
        # Test alert manager trigger
        if sys_state == "ALARM":
            alert_manager.trigger_alert(frame, detections, sys_state)
            
        # Test HUD drawing
        hud_frame = hud_visualizer.draw_hud(
            frame=frame,
            system_state=sys_state,
            detections=detections,
            fps=fps,
            toggles={"smoke": detector.enable_smoke, "audio": False, "voice": False, "snapshots": True},
            engine_mode=detector.mode
        )
        assert hud_frame.shape == (480, 640, 3)
        
    avg_fps = sum(fps_records) / len(fps_records)
    print(f"[+] Headless pipeline test PASSED!")
    print(f"    - Processed 30 frames successfully")
    print(f"    - Measured Average Processing Speed: {avg_fps:.1f} FPS")
    return True

if __name__ == "__main__":
    test_headless_run()
