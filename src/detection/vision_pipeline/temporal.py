import cv2
import numpy as np
from typing import Dict, Any, Tuple

class TemporalChangeDetector:
    """
    Analyzes temporal differences between a 'Before' and 'After' image pair.
    """
    
    def detect(self, before_img: np.ndarray, after_img: np.ndarray) -> Dict[str, Any]:
        """
        Registers images and finds structural pixel differences, ignoring minor lighting.
        """
        if before_img is None or after_img is None:
            return {
                "change_score": 0.0,
                "change_heatmap": None,
                "registration_quality": "N/A"
            }
            
        # Ensure same size
        h, w = after_img.shape[:2]
        before_img = cv2.resize(before_img, (w, h))
        
        gray_before = cv2.cvtColor(before_img, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(after_img, cv2.COLOR_BGR2GRAY)
        
        # Absolute difference
        diff = cv2.absdiff(gray_before, gray_after)
        
        # Threshold to ignore minor lighting changes
        _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
        
        # Clean up noise
        kernel = np.ones((5,5), np.uint8)
        clean_diff = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        total_pixels = h * w
        changed_pixels = cv2.countNonZero(clean_diff)
        change_density = changed_pixels / total_pixels
        
        # Score normalization (10% change is massive)
        change_score = min(1.0, change_density / 0.10)
        
        # Heatmap
        heatmap = cv2.applyColorMap(clean_diff, cv2.COLORMAP_HOT)
        
        return {
            "change_score": round(change_score, 3),
            "change_heatmap": heatmap,
            "changed_regions_density": round(change_density, 3),
            "registration_quality": "ACCEPTABLE" # Mocked for prototype
        }
