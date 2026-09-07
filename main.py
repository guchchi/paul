"""
Main Application Entry Point (Desktop HUD)
Real-time Fire & Smoke Detection from Laptop Webcam.
Features live keyboard toggles, non-blocking alerts, and HUD overlays.
"""
import sys
import time
import cv2

from src.config import config, INCIDENTS_DIR
from src.detection.fire_detector import FireDetector
from src.alerts.alert_manager import alert_manager
from src.utils.visualizer import hud_visualizer

def main():
    print("=" * 65)
    print("      SIH SMART MINE HAZARD SURVEILLANCE: FIRE & SMOKE DETECTOR")
    print("=" * 65)
    print(f"[*] Starting Video Stream from Camera Index: {config.camera_index}")
    print("[*] Keyboard Controls:")
    print("    [M] : Toggle Smoke Detection (ON / OFF)")
    print("    [A] : Toggle Audio Siren Alarm (ON / OFF)")
    print("    [V] : Toggle Voice Announcements (ON / OFF)")
    print("    [S] : Save Manual Snapshot")
    print("    [E] : Cycle Engine Mode (HYBRID / CV_ONLY / DL_ONLY)")
    print("    [Q] : Quit Application")
    print("=" * 65)

    # Initialize Webcam with DirectShow backend for fast startup on Windows
    cap = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[!] DirectShow failed, attempting standard camera backend...")
        cap = cv2.VideoCapture(config.camera_index)
        
    if not cap.isOpened():
        print(f"[x] Critical Error: Unable to open webcam at index {config.camera_index}!")
        print("    Ensure no other application is using the camera.")
        sys.exit(1)
        
    # Configure camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
    
    # Initialize Detection Engine
    detector = FireDetector(mode=config.engine_mode, enable_smoke=config.enable_smoke)
    
    window_name = "SIH Fire & Smoke Detection System (Press Q to Exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.frame_width, config.frame_height)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[!] Frame read failed. Retrying...")
                time.sleep(0.05)
                continue
                
            # Run detection pipeline
            system_state, detections, fps = detector.process_frame(frame)
            
            # Handle alarms and snapshot logging
            if system_state == "ALARM":
                alert_manager.trigger_alert(frame, detections, system_state)
                
            # Compile current active toggles for HUD display
            active_toggles = {
                "smoke": detector.enable_smoke,
                "audio": alert_manager.enable_audio,
                "voice": alert_manager.enable_voice,
                "snapshots": alert_manager.enable_snapshots
            }
            
            # Draw presentation HUD
            display_frame = hud_visualizer.draw_hud(
                frame=frame,
                system_state=system_state,
                detections=detections,
                fps=fps,
                toggles=active_toggles,
                engine_mode=detector.mode
            )
            
            # Render to desktop window
            cv2.imshow(window_name, display_frame)
            
            # Keyboard interaction handler (wait 1ms)
            key = cv2.waitKey(1) & 0xFF
            
            if key in (ord('q'), ord('Q'), 27):  # 'q' or ESC
                print("[*] Exit signal received. Shutting down...")
                break
            elif key in (ord('m'), ord('M')):     # Toggle Smoke
                new_state = not detector.enable_smoke
                detector.set_smoke_detection(new_state)
            elif key in (ord('a'), ord('A')):     # Toggle Audio
                alert_manager.set_audio(not alert_manager.enable_audio)
            elif key in (ord('v'), ord('V')):     # Toggle Voice
                alert_manager.set_voice(not alert_manager.enable_voice)
            elif key in (ord('s'), ord('S')):     # Manual snapshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                snap_path = INCIDENTS_DIR / f"manual_snapshot_{timestamp}.jpg"
                cv2.imwrite(str(snap_path), display_frame)
                print(f"[+] Manual snapshot saved to: {snap_path}")
            elif key in (ord('e'), ord('E')):     # Cycle Engine
                modes = ["HYBRID", "CV_ONLY", "DL_ONLY"]
                cur_idx = modes.index(detector.mode) if detector.mode in modes else 0
                detector.mode = modes[(cur_idx + 1) % len(modes)]
                print(f"[+] Switched Engine Mode to: {detector.mode}")
                
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[+] Camera released. System terminated cleanly.")

if __name__ == "__main__":
    main()
