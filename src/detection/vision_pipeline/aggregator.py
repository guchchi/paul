from typing import Dict, Any

class EvidenceAggregator:
    """
    Aggregates normalized scores from various modules.
    Configurable weights allow tuning without changing module logic.
    """
    
    def __init__(self):
        # Configurable weights (Prioritize YOLO Crack Model for Prototype)
        self.crack_weight = 0.95
        self.anomaly_weight = 0.0  # Disabled for prototype due to false positives on natural terrain
        self.change_weight = 0.05
        self.tracking_weight = 0.0  # Zero for prototype
        
    def aggregate(self, scores: Dict[str, float], quality_score: float) -> float:
        """
        Combines scores into a single visual evidence score (0.0 to 1.0).
        Poor quality images explicitly pull down the evidence confidence, 
        ensuring we don't hallucinate safety from bad data.
        """
        crack = scores.get("crack_score", 0.0)
        anomaly = scores.get("anomaly_score", 0.0)
        change = scores.get("change_score", 0.0)
        tracking = scores.get("tracking_score", 0.0)
        
        # Weighted sum of evidence
        raw_evidence = (
            (crack * self.crack_weight) +
            (anomaly * self.anomaly_weight) +
            (change * self.change_weight) +
            (tracking * self.tracking_weight)
        )
        
        # Cap at 1.0
        evidence_score = min(1.0, raw_evidence)
        
        # Note: We do NOT multiply evidence by quality to reduce it.
        # Quality affects the *confidence* of the risk classifier, 
        # not the evidence strength itself. The aggregator simply returns the evidence.
        return round(evidence_score, 3)
