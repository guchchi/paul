import cv2
import numpy as np
from src.detection.zone_classifier import zone_classifier

def test_zone_classifier():
    print("[*] Testing ZoneVisionClassifier AI Model...")
    
    # 1. Create a dummy image representing a "SAFE" zone (plain gray, no cracks)
    safe_img = np.ones((500, 500, 3), dtype=np.uint8) * 150
    is_success, buffer = cv2.imencode(".jpg", safe_img)
    safe_bytes = buffer.tobytes()
    
    res_safe = zone_classifier.analyze_image(safe_bytes)
    assert res_safe["rating"] == "SAFE", f"Expected SAFE, got {res_safe['rating']}"
    assert res_safe["anomaly_pct"] < 2.0
    print("[+] SAFE image classification passed.")
    
    # 2. Create a dummy image representing a "DANGER" zone (black background with many white lines to simulate cracks)
    danger_img = np.zeros((500, 500, 3), dtype=np.uint8)
    for i in range(0, 500, 20):
        cv2.line(danger_img, (0, i), (500, i+20), (255, 255, 255), 5)
        
    is_success, buffer = cv2.imencode(".jpg", danger_img)
    danger_bytes = buffer.tobytes()
    
    res_danger = zone_classifier.analyze_image(danger_bytes)
    assert res_danger["rating"] == "DANGER", f"Expected DANGER, got {res_danger['rating']}"
    assert res_danger["anomaly_pct"] >= 6.0
    print("[+] DANGER image classification passed.")
    
    print("[*] All tests passed!")

if __name__ == "__main__":
    test_zone_classifier()
