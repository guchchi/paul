"""
Module 4: Mine Project & Zone Management
Handles the creation of Mine Projects, automatic distribution into Zones, 
and state persistence for the AI-driven Zone Classification flow.
"""
import os
import json
import math
from typing import Dict, List, Any
from pathlib import Path

from src.config import BASE_DIR

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROJECTS_FILE = DATA_DIR / "projects.json"

class ProjectManager:
    def __init__(self):
        self.projects: Dict[str, dict] = {}
        self.load_state()

    def load_state(self):
        if PROJECTS_FILE.exists():
            try:
                with open(PROJECTS_FILE, 'r') as f:
                    self.projects = json.load(f)
            except Exception as e:
                print(f"Failed to load projects: {e}")
                self.projects = {}
        else:
            # Initialize with dummy project if empty to show something immediately
            self.add_mine_project("Jharia Block II", 1200, 1200)

    def save_state(self):
        with open(PROJECTS_FILE, 'w') as f:
            json.dump(self.projects, f, indent=4)

    def add_mine_project(self, name: str, width_m: float, height_m: float) -> str:
        project_id = f"PROJ_{len(self.projects) + 1:03d}"
        
        # Determine grid size based on dimensions (assume ~400x400m per zone)
        cols = max(1, math.ceil(width_m / 400))
        rows = max(1, math.ceil(height_m / 400))
        
        zones = []
        for r in range(rows):
            for c in range(cols):
                zone_id = f"{chr(65 + r)}{c + 1}" # A1, A2, B1, etc.
                zones.append({
                    "zone_id": zone_id,
                    "row": r,
                    "col": c,
                    "status": "UNSCANNED",
                    "risk_score": 0.0,
                    "last_scan": None,
                    "anomalies": []
                })

        self.projects[project_id] = {
            "project_id": project_id,
            "name": name,
            "width_m": width_m,
            "height_m": height_m,
            "grid_rows": rows,
            "grid_cols": cols,
            "zones": zones,
            "created_at": "2026-09-12" # Static for demo
        }
        
        self.save_state()
        return project_id

    def get_all_projects(self) -> List[dict]:
        return list(self.projects.values())

    def get_project(self, project_id: str) -> dict:
        return self.projects.get(project_id, None)

    def update_zone_status(self, project_id: str, zone_id: str, status: str, score: float, anomalies: List[str]):
        if project_id in self.projects:
            for zone in self.projects[project_id]["zones"]:
                if zone["zone_id"] == zone_id:
                    zone["status"] = status
                    zone["risk_score"] = score
                    zone["anomalies"] = anomalies
                    import datetime
                    zone["last_scan"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    break
            self.save_state()

# Singleton instance
project_manager = ProjectManager()
