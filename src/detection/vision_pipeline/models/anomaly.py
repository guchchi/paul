import cv2
import numpy as np
from typing import Dict, Any
from .base import BaseAnomalyModel

class CVPrototypeAnomalyBackend(BaseAnomalyModel):
    """
    Prototype Vision Heuristic backend for visual anomaly detection.
    Computes a mock anomaly heatmap using structural texture deviation (Laplacian variation)
    to mimic an Autoencoder's reconstruction error map.
    """
    
    def get_backend_name(self) -> str:
        return "Prototype Vision Heuristic (Texture Deviation)"
        
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture deviation using Scharr Gradient Magnitude
        # This highlights surface roughness and deep fissures much better than Laplacian
        grad_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        grad_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        magnitude = cv2.magnitude(grad_x, grad_y)
        
        # Smooth out the variance to create regional hotspots rather than pixel noise
        variance_map = cv2.GaussianBlur(magnitude, (45, 45), 0)
        
        # Clip to remove extreme outliers which would squash the rest of the map to blue
        p98 = np.percentile(variance_map, 98.0)
        if p98 > 0:
            variance_map = np.clip(variance_map, 0, p98)
        
        # Normalize to 0-255 for heatmap
        normalized_variance = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX)
        anomaly_heatmap = np.uint8(normalized_variance)
        
        # Threshold to find areas of high deviation
        _, thresh = cv2.threshold(anomaly_heatmap, 150, 255, cv2.THRESH_BINARY)
        
        # Calculate anomaly score (percentage of highly variant pixels)
        total_pixels = image.shape[0] * image.shape[1]
        anomaly_pixels = cv2.countNonZero(thresh)
        anomaly_density = anomaly_pixels / total_pixels
        
        # Normalize score between 0.0 and 1.0 (Assume 15% highly variant is max anomaly)
        anomaly_score = min(1.0, anomaly_density / 0.15)
        
        # Color map for UI visualization
        heatmap_colored = cv2.applyColorMap(anomaly_heatmap, cv2.COLORMAP_JET)
        
        return {
            "heatmap": heatmap_colored,
            "anomaly_score": float(round(anomaly_score, 3))
        }

class VisualAnomalyDetector:
    """
    Adapter class for Visual Anomaly Detection. 
    Can hot-swap between CVPrototypeBackend and a Deep Learning Autoencoder.
    """
    def __init__(self, use_deep_learning: bool = False):
        if use_deep_learning:
            raise NotImplementedError("Deep Learning Autoencoder backend is not yet trained/available.")
        else:
            self.backend = CVPrototypeAnomalyBackend()
            
    def get_backend_name(self) -> str:
        return self.backend.get_backend_name()
        
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        return self.backend.predict(image)
