"""
Classical Computer Vision Fire & Smoke Filter
Implements multi-color space segmentation (HSV + YCrCb) with temporal flicker dynamics.
Functions as both a standalone high-FPS detector and a verification layer for DL models.
"""
import cv2
import numpy as np
from typing import List, Dict, Any

class CVFireSmokeFilter:
    def __init__(self, min_area: int = 300):
        self.min_area = min_area
        self.prev_gray = None
        self.prev_fire_mask = None
        
    def detect(self, frame: cv2.Mat, detect_smoke: bool = True) -> List[Dict[str, Any]]:
        """
        Analyze frame for fire and smoke regions.
        Returns list of detections: [{'box': (x, y, w, h), 'label': 'fire'/'smoke', 'confidence': float}]
        """
        if frame is None:
            return []
            
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # -------------------------------------------------------------
        # 1. Fire Color Segmentation
        # -------------------------------------------------------------
        # HSV fire thresholds (Hue 0-25 and 165-180 for reds, oranges, and warm yellow flame)
        lower_fire_hsv1 = np.array([0, 90, 140], dtype=np.uint8)
        upper_fire_hsv1 = np.array([28, 255, 255], dtype=np.uint8)
        mask_hsv1 = cv2.inRange(hsv, lower_fire_hsv1, upper_fire_hsv1)
        
        lower_fire_hsv2 = np.array([168, 90, 140], dtype=np.uint8)
        upper_fire_hsv2 = np.array([180, 255, 255], dtype=np.uint8)
        mask_hsv2 = cv2.inRange(hsv, lower_fire_hsv2, upper_fire_hsv2)
        mask_fire_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
        
        # YCrCb fire condition: Flame rule: Y >= Cb, Cr >= Cb, and Cr >= 135
        Y, Cr, Cb = cv2.split(ycrcb)
        fire_ycrcb_cond = (Y >= Cb) & (Cr >= Cb) & (Cr >= 135) & (Y >= 110)
        mask_fire_ycrcb = np.uint8(fire_ycrcb_cond * 255)
        
        # Combined Fire Mask
        fire_mask = cv2.bitwise_and(mask_fire_hsv, mask_fire_ycrcb)
        
        # Morphological operations to remove isolated speckle noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_DILATE, kernel, iterations=2)
        
        # -------------------------------------------------------------
        # 2. Flicker / Dynamic Verification
        # -------------------------------------------------------------
        flicker_score = 0.5  # default if no previous frame
        if self.prev_gray is not None:
            frame_diff = cv2.absdiff(gray, self.prev_gray)
            # Check movement specifically inside candidate fire regions
            motion_in_fire = cv2.mean(frame_diff, mask=fire_mask)[0]
            # Flames exhibit temporal variation between 3.0 and 35.0 intensity units
            if motion_in_fire > 2.0:
                flicker_score = min(0.95, 0.5 + (motion_in_fire / 25.0) * 0.45)
            else:
                flicker_score = 0.35  # static warm object (e.g. orange book, desk lamp)
                
        self.prev_gray = gray.copy()
        
        detections = []
        
        # Find contours for Fire
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_area:
                x, y, cw, ch = cv2.boundingRect(cnt)
                # Compute aspect ratio and fill factor
                extent = area / float(cw * ch)
                if 0.15 < extent < 0.95:  # Flame shapes have jagged contours
                    conf = round(float(flicker_score * min(1.0, 0.4 + (area / 1500.0) * 0.5)), 2)
                    detections.append({
                        "box": (x, y, cw, ch),
                        "label": "fire",
                        "confidence": max(0.40, conf),
                        "area": area
                    })
                    
        # -------------------------------------------------------------
        # 3. Smoke Segmentation (If enabled)
        # -------------------------------------------------------------
        if detect_smoke:
            # Smoke rule: Low saturation (grayish/white), mid-to-high luminance
            H, S, V = cv2.split(hsv)
            smoke_cond = (S < 65) & (V >= 90) & (V <= 235) & (Cr >= 115) & (Cr <= 145) & (Cb >= 115) & (Cb <= 145)
            smoke_mask = np.uint8(smoke_cond * 255)
            smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_DILATE, kernel, iterations=2)
            
            # Discard smoke overlapping with fire
            smoke_mask = cv2.bitwise_and(smoke_mask, cv2.bitwise_not(fire_mask))
            
            smoke_contours, _ = cv2.findContours(smoke_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in smoke_contours:
                area = cv2.contourArea(cnt)
                # Smoke plumes are generally larger and more diffuse
                if area >= (self.min_area * 2):
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    # Filter out whole-frame background
                    if cw < w * 0.9 and ch < h * 0.9:
                        detections.append({
                            "box": (x, y, cw, ch),
                            "label": "smoke",
                            "confidence": 0.55,
                            "area": area
                        })
                        
        return detections
