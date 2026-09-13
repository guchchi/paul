import pytest
import numpy as np
import cv2
from src.detection.vision_pipeline.pipeline import vision_pipeline

def generate_test_image(size=(800, 800), blur=False, dark=False, anomalies=False):
    # Base uniform image with noise to pass contrast/blur checks
    img = np.random.randint(120, 180, (size[0], size[1], 3), dtype=np.uint8)
    
    if dark:
        img = np.random.randint(10, 30, (size[0], size[1], 3), dtype=np.uint8)
        
    if anomalies:
        # Add crack-like structures
        cv2.line(img, (100, 100), (300, 400), (0, 0, 0), 8)
        cv2.line(img, (300, 400), (500, 600), (0, 0, 0), 6)
        cv2.circle(img, (600, 200), 50, (0, 0, 0), -1)
        
    if blur:
        img = cv2.GaussianBlur(img, (25, 25), 0)
        
    is_success, buffer = cv2.imencode(".jpg", img)
    return buffer.tobytes()

def test_pipeline_low_quality_gate():
    bad_img = generate_test_image(blur=True, dark=True)
    res = vision_pipeline.process("TEST_Z1", "Fixed Ground Camera", bad_img)
    
    assert res["image_quality"] == "LOW QUALITY / RESCAN REQUIRED"
    assert "Image is too blurry" in res["quality_issues"]
    assert "Image is under-exposed (too dark)" in res["quality_issues"]
    assert res["confidence"] < 50.0

def test_pipeline_safe_image():
    safe_img = generate_test_image()
    res = vision_pipeline.process("TEST_Z2", "Satellite / Aerial", safe_img)
    
    assert "image_quality" in res
    assert "risk_level" in res
    assert "sensor_verification_required" in res
    assert type(res["hotspots"]) == list

def test_pipeline_anomaly_detection():
    danger_img = generate_test_image(anomalies=True)
    res = vision_pipeline.process("TEST_Z3", "Drone / Orthographic", danger_img)
    
    assert res["crack_score"] > 0
    assert len(res["hotspots"]) > 0
    
    # Depending on thresholds, it should be at least MODERATE or HIGH
    assert res["risk_level"] in ["MODERATE", "HIGH RISK", "CRITICAL VISUAL ALERT"]
    
def test_pipeline_temporal_change():
    before_img = generate_test_image()
    after_img = generate_test_image(anomalies=True)
    
    res = vision_pipeline.process("TEST_Z4", "Temporal (Before/After)", after_img, before_img)
    
    assert res["change_score"] > 0.0
    assert res["temporal_registration_quality"] == "ACCEPTABLE"
