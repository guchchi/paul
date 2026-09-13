import json
import os
from datetime import datetime
from typing import Dict, Any

class ScanHistoryManager:
    """
    Manages persistence of scans to prevent one-off false alarms from triggering critical alerts.
    """
    def __init__(self, history_file: str = "data/scan_history.json"):
        self.history_file = history_file
        self.history = self._load()
        
    def _load(self) -> Dict:
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
        
    def _save(self):
        import numpy as np
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=4, cls=NumpyEncoder)
            
    def get_persistence_score(self, zone_id: str, current_evidence_score: float) -> float:
        """
        Calculates how persistent the anomaly is based on previous scans.
        """
        if zone_id not in self.history:
            return 0.0
            
        zone_scans = self.history[zone_id]
        if not zone_scans:
            return 0.0
            
        # Get the most recent scan
        last_scan = zone_scans[-1]
        last_evidence = last_scan.get("visual_evidence_score", 0.0)
        
        if current_evidence_score > 0.4 and last_evidence > 0.4:
            # Anomaly persists
            if current_evidence_score > last_evidence:
                return 0.8 # Anomaly is increasing
            return 0.5 # Anomaly is stable
            
        return 0.0
        
    def add_scan(self, zone_id: str, scan_data: Dict[str, Any]):
        """
        Records a new scan into history.
        """
        if zone_id not in self.history:
            self.history[zone_id] = []
            
        # Keep only the last 10 scans per zone to manage size
        self.history[zone_id].append({
            "timestamp": datetime.now().isoformat(),
            "risk_level": scan_data.get("risk_level", "UNKNOWN"),
            "visual_evidence_score": scan_data.get("visual_evidence_score", 0.0)
        })
        self.history[zone_id] = self.history[zone_id][-10:]
        
        self._save()
