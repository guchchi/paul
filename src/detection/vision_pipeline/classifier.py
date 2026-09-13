from typing import Dict, Any, List

class RiskClassifier:
    """
    Classifies the aggregated evidence into a prototype decision-support category
    and generates appropriate Recommended Actions.
    """
    
    def classify(self, 
                 evidence_score: float, 
                 quality_score: float, 
                 persistence_score: float,
                 evidence_list: List[str]) -> Dict[str, Any]:
        """
        Calculates Risk Level, Confidence, and Action based on rules.
        """
        # 1. Base Confidence is driven by Image Quality
        confidence = quality_score * 100.0
        
        # 2. Adjust Risk Level combining current evidence + persistence
        # Persistence acts as an amplifier
        effective_risk_score = evidence_score + (persistence_score * 0.2)
        
        if effective_risk_score >= 0.85:
            risk_level = "CRITICAL VISUAL ALERT"
            action = "Immediate ground-sensor verification and professional inspection required"
            sensor_req = True
        elif effective_risk_score >= 0.70:
            risk_level = "HIGH RISK"
            action = "Verify with nearby ground sensors and inspect zone"
            sensor_req = True
        elif effective_risk_score >= 0.55:
            risk_level = "MODERATE"
            action = "Increase monitoring frequency and rescan"
            sensor_req = False
        else:
            risk_level = "SAFE"
            action = "Continue routine monitoring"
            sensor_req = False
            
        # 3. Handle Low Quality edge cases
        if quality_score < 0.5:
            if risk_level in ["SAFE", "MODERATE"]:
                risk_level = "LOW CONFIDENCE / RESCAN REQUIRED"
                action = "Rescan zone with higher quality imagery"
                sensor_req = False
            else:
                # If we see high risk even in low quality, maintain alert but drop confidence
                confidence -= 20.0
                
        # Constrain confidence
        confidence = max(0.0, min(99.9, confidence))
        
        return {
            "risk_level": risk_level,
            "confidence": round(confidence, 1),
            "recommended_action": action,
            "sensor_verification_required": sensor_req,
            "evidence": evidence_list if evidence_score > 0.15 else ["Visual indicators within normal parameters"]
        }
