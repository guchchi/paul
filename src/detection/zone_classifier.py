"""
Module 4: Zone Vision Classifier
Simulates an AI/ML Computer Vision model to analyze uploaded satellite/camera 
images for subsidence cracks, fissures, and anomalies using advanced OpenCV heuristics.
Returns a SAFE / MODERATE / DANGER rating based on detected anomaly density.
"""
import cv2
import numpy as np
from typing import Dict, Any, Tuple
import io
from PIL import Image

class ZoneVisionClassifier:
    def __init__(self):
        # Thresholds for classification
        self.moderate_threshold_pct = 2.0  # 2% of area covered by anomalies
        self.danger_threshold_pct = 6.0    # 6% of area covered by anomalies

    def analyze_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Processes an uploaded image to detect cracks/fissures.
        Returns a dictionary with the rating, score, anomalies, and annotated image bytes.
        """
        # Convert bytes to numpy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Invalid image provided to ZoneVisionClassifier")

        # Keep a copy for annotation
        annotated_img = img.copy()
        
        # Preprocessing: Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge Detection (Canny) to find cracks/fissures
        # Adaptive thresholds could be used, but static works for prototype
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect broken segments of cracks
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours of the anomalies
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_pixels = img.shape[0] * img.shape[1]
        anomaly_area = 0
        detected_anomalies = []
        
        # Filter small noise contours
        min_contour_area = total_pixels * 0.0005 # at least 0.05% of image
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_contour_area:
                anomaly_area += area
                
                # Draw bounding box on annotated image
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Classify the type of anomaly heuristically
                aspect_ratio = float(w)/h
                if aspect_ratio > 3 or aspect_ratio < 0.33:
                    anomaly_type = "Linear Fissure"
                    color = (0, 0, 255) # Red for fissures
                else:
                    anomaly_type = "Subsidence Pooling / Sinkhole"
                    color = (0, 165, 255) # Orange
                    
                cv2.rectangle(annotated_img, (x, y), (x+w, y+h), color, 3)
                cv2.putText(annotated_img, anomaly_type, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if anomaly_type not in detected_anomalies:
                    detected_anomalies.append(anomaly_type)

        # Calculate Risk Score (Percentage of anomaly area)
        anomaly_pct = (anomaly_area / total_pixels) * 100
        
        # Boost score slightly to make it look realistic for a 0-100 scale
        risk_score = min(100.0, anomaly_pct * 12.5) 
        
        # Classification Logic
        if anomaly_pct >= self.danger_threshold_pct:
            rating = "DANGER"
            if "Critical Rupture" not in detected_anomalies:
                detected_anomalies.append("Critical Rupture")
        elif anomaly_pct >= self.moderate_threshold_pct:
            rating = "MODERATE"
        else:
            rating = "SAFE"
            if not detected_anomalies:
                detected_anomalies = ["Normal Strata (No anomalies)"]
                
        # Convert annotated image back to bytes for Streamlit
        is_success, buffer = cv2.imencode(".jpg", annotated_img)
        annotated_bytes = buffer.tobytes() if is_success else image_bytes

        return {
            "rating": rating,
            "risk_score": round(risk_score, 1),
            "anomaly_pct": round(anomaly_pct, 2),
            "anomalies": detected_anomalies,
            "annotated_image_bytes": annotated_bytes
        }

# Singleton instance
zone_classifier = ZoneVisionClassifier()
