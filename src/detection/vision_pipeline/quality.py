import cv2
import numpy as np
from typing import Dict, Any

class ImageQualityValidator:
    """
    Validates the quality of incoming images. 
    A poor quality image prevents false positives and reduces confidence.
    """
    
    def validate(self, image: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        issues = []
        
        # 1. Blur Check (Laplacian Variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = laplacian_var < 50
        if is_blurry:
            issues.append("Image is too blurry")
            
        # 2. Brightness Check
        mean_brightness = np.mean(gray)
        if mean_brightness < 40:
            issues.append("Image is under-exposed (too dark)")
        elif mean_brightness > 220:
            issues.append("Image is over-exposed (too bright)")
            
        # 3. Contrast Check
        rms_contrast = gray.std()
        if rms_contrast < 20:
            issues.append("Image has poor contrast")

        # Determine Status and Score
        if len(issues) == 0:
            status = "GOOD"
            score = 1.0
        elif len(issues) == 1 and not is_blurry:
            status = "ACCEPTABLE"
            score = 0.7
        else:
            status = "LOW QUALITY / RESCAN REQUIRED"
            score = 0.2
            
        return {
            "quality_score": score,
            "quality_status": status,
            "quality_issues": issues
        }
