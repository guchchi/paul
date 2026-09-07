"""
Hardware Diagnostic Script
Tests laptop webcam connectivity, frame acquisition speed, and resolution.
"""
import sys
import time
import cv2

def test_webcam(camera_index=0):
    print(f"[*] Testing Webcam at Index {camera_index}...")
    
    # Try DirectShow first on Windows for faster initialization
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[!] CAP_DSHOW failed, falling back to default backend...")
        cap = cv2.VideoCapture(camera_index)
        
    if not cap.isOpened():
        print(f"[x] Error: Unable to access camera at index {camera_index}!")
        return False
        
    # Read properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[+] Camera connected successfully!")
    print(f"    - Native Resolution: {int(width)}x{int(height)}")
    print(f"    - Reported FPS: {fps}")
    
    # Test reading 10 frames to measure real acquisition speed
    print("[*] Benchmarking frame grab latency (10 frames)...")
    start_time = time.time()
    frames_read = 0
    for _ in range(10):
        ret, frame = cap.read()
        if ret:
            frames_read += 1
            
    elapsed = time.time() - start_time
    measured_fps = frames_read / elapsed if elapsed > 0 else 0
    print(f"[+] Read {frames_read}/10 frames in {elapsed:.2f}s (~{measured_fps:.1f} FPS)")
    
    cap.release()
    return frames_read > 0

if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    success = test_webcam(idx)
    sys.exit(0 if success else 1)
