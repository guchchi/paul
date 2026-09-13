import cv2
import numpy as np
from typing import Dict, Any
from .base import BaseVisionModel
import os

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class CVPrototypeCrackBackend(BaseVisionModel):
    """
    OpenCV Blackhat Morphology backend for crack localization.
    Extracts dark, narrow structures (cracks) from lighter backgrounds
    and returns bounding boxes around detected regions.
    """
    
    def get_backend_name(self) -> str:
        return "OpenCV Blackhat Morphology (Heuristic)"
        
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        total_pixels = h * w
        
        # ─── Feature 1: Edge Density Ratio ───
        # Cracked surfaces have significantly higher edge density than smooth surfaces
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        edge_density = cv2.countNonZero(edges) / total_pixels
        
        # ─── Feature 2: Blackhat Morphology for crack-like structures ───
        kernel_size = (15, 15)
        blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, blackhat_kernel)
        
        # Use OTSU threshold instead of adaptive - less sensitive to gentle textures
        _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        blackhat_density = cv2.countNonZero(thresh) / total_pixels
        
        # ─── Feature 3: Local Contrast Variance ───
        # Cracked ground has high local contrast; smooth fields have low local contrast
        local_std = cv2.GaussianBlur(gray.astype(np.float32), (31, 31), 0)
        local_sq = cv2.GaussianBlur((gray.astype(np.float32))**2, (31, 31), 0)
        local_variance = np.mean(np.sqrt(np.maximum(local_sq - local_std**2, 0)))
        # Normalize: smooth surface ~5-15, cracked surface ~25-50+
        contrast_score = min(1.0, max(0.0, (local_variance - 10) / 35))
        
        # ─── Feature 4: Uniform region detection ───
        # Safe images (fields, sky) have large uniform regions
        # Cracked images have very few uniform regions
        blur_heavy = cv2.GaussianBlur(gray, (21, 21), 0)
        diff = cv2.absdiff(gray, blur_heavy)
        uniform_ratio = 1.0 - (cv2.countNonZero(cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)[1]) / total_pixels)
        
        # ─── Combine Features into crack_score ───
        # Weight the features: high edge density + high blackhat + high contrast + low uniformity = cracks
        raw_score = (
            0.25 * min(1.0, edge_density / 0.12) +     # Edge density (cracked > 0.12)
            0.30 * min(1.0, blackhat_density / 0.08) +  # Blackhat density 
            0.25 * contrast_score +                      # Local contrast
            0.20 * (1.0 - uniform_ratio)                 # Inverse uniformity
        )
        
        crack_score = round(min(1.0, max(0.0, raw_score)), 3)
        
        # ─── Hotspot Detection (only if score is significant) ───
        hotspots = []
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(thresh, dilate_kernel, iterations=2)
        
        if crack_score > 0.15:
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_bbox_area = total_pixels * 0.002
            
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                bbox_area = bw * bh
                if bbox_area > min_bbox_area:
                    aspect_ratio = float(bw) / max(bh, 1)
                    severity = "High" if bbox_area > (min_bbox_area * 5) else "Moderate"
                    
                    hotspots.append({
                        "bbox": (x, y, bw, bh),
                        "type": "Network/Fissure" if (aspect_ratio > 3 or aspect_ratio < 0.33) else "Subsidence Region",
                        "severity": severity,
                        "area": bbox_area
                    })
                    
            hotspots.sort(key=lambda hs: hs["area"], reverse=True)
            hotspots = hotspots[:15]
            
            for hs in hotspots:
                del hs["area"]
            
        crack_density = blackhat_density
        
        return {
            "mask": dilated,
            "crack_density": float(crack_density),
            "crack_score": float(crack_score),
            "hotspots": hotspots,
            "connected_components": int(len(hotspots)),
            "features": {
                "edge_density": float(round(edge_density, 4)),
                "blackhat_density": float(round(blackhat_density, 4)),
                "contrast_score": float(round(contrast_score, 3)),
                "uniform_ratio": float(round(uniform_ratio, 3))
            }
        }


class YOLOClassifierBackend(BaseVisionModel):
    """
    YOLOv8 Classification backend.
    Uses a trained YOLOv8-cls model to classify the entire image as crack/no_crack.
    The classification confidence drives the crack_score.
    """
    def __init__(self, model_path: str = None):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics is required for YOLOClassifierBackend")
        
        # Search for custom trained weights
        if model_path is None:
            # Walk up from this file: models/ -> vision_pipeline/ -> detection/ -> src/ -> project_root/
            this_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(this_dir))))
            candidates = [
                os.path.join(project_root, "models", "crack_classifier_best.pt"),
                os.path.join(project_root, "models", "crack_classifier", "weights", "best.pt"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_path = c
                    break
        
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            self.model_path = model_path
            self.has_custom_weights = True
        else:
            # No custom model yet — will return neutral scores
            self.model = None
            self.model_path = "none (not trained yet)"
            self.has_custom_weights = False
        
    def get_backend_name(self) -> str:
        if self.has_custom_weights:
            return f"Deep Learning (YOLOv8-cls - {os.path.basename(self.model_path)})"
        return "YOLOv8-cls (awaiting training)"
        
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        if not self.has_custom_weights or self.model is None:
            return {
                "crack_probability": 0.5,
                "classification": "unknown",
                "model_ready": False
            }
        
        results = self.model(image, verbose=False)
        
        if len(results) > 0:
            result = results[0]
            probs = result.probs
            
            if probs is not None:
                class_names = result.names
                top1_idx = int(probs.top1)
                top1_conf = float(probs.top1conf)
                top1_name = class_names.get(top1_idx, "unknown")
                
                # Determine crack probability
                # Look for class named 'crack', 'Positive', 'cracked', etc.
                crack_prob = 0.0
                for idx, name in class_names.items():
                    name_lower = name.lower()
                    if any(k in name_lower for k in ["crack", "positive", "damage", "fissure"]):
                        crack_prob = float(probs.data[idx])
                        break
                
                return {
                    "crack_probability": round(crack_prob, 4),
                    "classification": top1_name,
                    "confidence": round(top1_conf, 4),
                    "model_ready": True
                }
        
        return {
            "crack_probability": 0.5,
            "classification": "unknown",
            "model_ready": False
        }


class HybridCrackBackend(BaseVisionModel):
    """
    HYBRID backend: Combines YOLOv8 Classification + OpenCV Blackhat Morphology.
    
    - YOLOv8-cls: Provides a global crack probability (is this image showing cracks?)
    - OpenCV Blackhat: Provides spatial localization (WHERE are the cracks?)
    
    The final crack_score is a weighted combination of both signals.
    If the YOLO model isn't trained yet, it falls back to pure OpenCV.
    """
    def __init__(self):
        self.cv_backend = CVPrototypeCrackBackend()
        
        try:
            self.yolo_backend = YOLOClassifierBackend()
            self.yolo_available = True
        except Exception:
            self.yolo_available = False
    
    def get_backend_name(self) -> str:
        if self.yolo_available and self.yolo_backend.has_custom_weights:
            return f"Hybrid (YOLOv8-cls + OpenCV Blackhat)"
        return "OpenCV Blackhat Morphology (YOLO awaiting training)"
        
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        # Always run OpenCV for spatial localization
        cv_result = self.cv_backend.predict(image)
        
        # Run YOLO classification if available
        if self.yolo_available:
            yolo_result = self.yolo_backend.predict(image)
            
            if yolo_result.get("model_ready", False):
                yolo_prob = yolo_result["crack_probability"]
                cv_score = cv_result["crack_score"]
                
                # Weighted fusion: YOLO classification confidence + OpenCV localization
                # YOLO is the "brain" (global understanding), OpenCV is the "eyes" (local detail)
                combined_score = float((0.6 * yolo_prob) + (0.4 * cv_score))
                combined_score = float(round(min(1.0, combined_score), 3))
                
                # Boost hotspot severity based on YOLO confidence
                for hs in cv_result["hotspots"]:
                    if yolo_prob > 0.7:
                        hs["severity"] = "High"
                    elif yolo_prob > 0.4:
                        hs["severity"] = "Moderate"
                
                cv_result["crack_score"] = combined_score
                cv_result["yolo_classification"] = str(yolo_result.get("classification", "N/A"))
                cv_result["yolo_confidence"] = float(yolo_result.get("confidence", 0.0))
                cv_result["yolo_crack_probability"] = float(yolo_prob)
                
                return cv_result
        
        # Fallback: pure OpenCV
        return cv_result


class CrackSegmentationModel:
    """
    Adapter class for Crack Segmentation. 
    Uses the Hybrid backend by default (YOLO + OpenCV).
    Falls back gracefully if YOLO is not available.
    """
    def __init__(self, use_deep_learning: bool = True):
        if use_deep_learning:
            self.backend = HybridCrackBackend()
        else:
            self.backend = CVPrototypeCrackBackend()
            
    def get_backend_name(self) -> str:
        return self.backend.get_backend_name()
        
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        return self.backend.predict(image)
