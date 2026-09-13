import cv2
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from .quality import ImageQualityValidator
from .preprocessing import ImagePreprocessor
from .models.crack_segmentation import CrackSegmentationModel
from .models.anomaly import VisualAnomalyDetector
from .temporal import TemporalChangeDetector
from .tracking import ReferenceFeatureTracker
from .aggregator import EvidenceAggregator
from .classifier import RiskClassifier
from .history import ScanHistoryManager

class VisionPipeline:
    """
    Orchestrates the modular AI Vision Pipeline for wide-area screening.
    """
    def __init__(self):
        self.quality_validator = ImageQualityValidator()
        self.preprocessor = ImagePreprocessor()
        self.crack_model = CrackSegmentationModel(use_deep_learning=True)
        self.anomaly_model = VisualAnomalyDetector()
        self.temporal_detector = TemporalChangeDetector()
        self.tracker = ReferenceFeatureTracker()
        self.aggregator = EvidenceAggregator()
        self.classifier = RiskClassifier()
        self.history_mgr = ScanHistoryManager()

    def process(self, 
                zone_id: str, 
                image_source: str, 
                primary_image_bytes: bytes, 
                reference_image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Executes the full vision pipeline.
        """
        scan_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # 1. Decode Images
        img = self._decode_image(primary_image_bytes)
        ref_img = self._decode_image(reference_image_bytes) if reference_image_bytes else None
        
        if img is None:
            raise ValueError("Invalid primary image provided.")
            
        annotated_img = img.copy()
        evidence_list = []
        
        # 2. Quality Validation
        quality = self.quality_validator.validate(img)
        quality_score = quality["quality_score"]
        
        # 3. Preprocessing
        processed_img = self.preprocessor.process(img)
        
        # 4. Feature Extraction Models
        scores = {}
        hotspots = []
        
        # Crack Segmentation
        crack_res = self.crack_model.predict(processed_img)
        scores["crack_score"] = crack_res["crack_score"]
        if crack_res["crack_score"] > 0.2:
            evidence_list.append("Crack/Fissure structures detected")
            
        # Draw crack hotspots on annotated image
        for hs in crack_res["hotspots"]:
            x, y, w, h = hs["bbox"]
            color = (0, 0, 255) if hs["severity"] == "High" else (0, 165, 255)
            cv2.rectangle(annotated_img, (x, y), (x+w, y+h), color, 3)
            cv2.putText(annotated_img, hs["type"], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            hs["location"] = f"Zone relative ({x},{y})"
            hotspots.append(hs)
            
        # Anomaly Detection
        anomaly_res = self.anomaly_model.predict(processed_img)
        scores["anomaly_score"] = anomaly_res["anomaly_score"]
        if anomaly_res["anomaly_score"] > 0.3:
            evidence_list.append("Abnormal texture/surface variation detected")
            
        # Temporal & Tracking (if reference image provided)
        if ref_img is not None:
            processed_ref = self.preprocessor.process(ref_img)
            
            temporal_res = self.temporal_detector.detect(processed_ref, processed_img)
            scores["change_score"] = temporal_res["change_score"]
            if temporal_res["change_score"] > 0.2:
                evidence_list.append("Significant temporal structural change detected")
                
            tracking_res = self.tracker.track(processed_ref, processed_img)
            scores["tracking_score"] = tracking_res["tracking_score"]
        else:
            temporal_res = {"change_heatmap": None, "registration_quality": "N/A"}
            
        # 5. Evidence Aggregation
        visual_evidence_score = self.aggregator.aggregate(scores, quality_score)
        
        # 6. Persistence Tracking
        persistence_score = self.history_mgr.get_persistence_score(zone_id, visual_evidence_score)
        if persistence_score > 0.4:
            evidence_list.append("Anomaly persists across multiple scans")
            
        # 7. Risk Classification
        classification = self.classifier.classify(
            visual_evidence_score, 
            quality_score, 
            persistence_score,
            evidence_list
        )
        
        # Encode output images
        is_success, buffer = cv2.imencode(".jpg", annotated_img)
        annotated_bytes = buffer.tobytes() if is_success else primary_image_bytes
        
        anomaly_heatmap = anomaly_res.get("heatmap")
        if anomaly_heatmap is not None:
            _, hm_buf = cv2.imencode(".jpg", anomaly_heatmap)
            heatmap_bytes = hm_buf.tobytes()
        else:
            heatmap_bytes = None
            
        # Construct final Scan Result
        scan_result = {
            "zone_id": zone_id,
            "scan_id": scan_id,
            "timestamp": timestamp,
            "image_source": image_source,
            "model_backend": self.crack_model.get_backend_name(),
            
            "image_quality": quality["quality_status"],
            "quality_issues": quality["quality_issues"],
            
            "crack_score": float(scores.get("crack_score", 0.0)),
            "anomaly_score": float(scores.get("anomaly_score", 0.0)),
            "change_score": float(scores.get("change_score", 0.0)),
            "persistence_score": float(persistence_score),
            "visual_evidence_score": float(visual_evidence_score),
            
            "risk_level": classification["risk_level"],
            "confidence": float(classification["confidence"]),
            "evidence": classification["evidence"],
            "recommended_action": classification["recommended_action"],
            "sensor_verification_required": classification["sensor_verification_required"],
            
            "hotspots": hotspots,
            "annotated_image_bytes": annotated_bytes,
            "heatmap_bytes": heatmap_bytes,
            "temporal_registration_quality": temporal_res.get("registration_quality")
        }
        
        # Save to history
        self.history_mgr.add_scan(zone_id, scan_result)
        
        return scan_result
        
    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img

# Singleton orchestrator
vision_pipeline = VisionPipeline()
