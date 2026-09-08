"""
Overground Ground Subsidence & Geotechnical Mesh Telemetry Module
Simulates distributed surface sensor nodes (Inclinometers, Micro-seismic Geophones,
Surface Crack Gauges) and computes Peck's Gaussian Subsidence Profile.
"""
import time
import math
import random
import numpy as np

class SubsidenceMesh:
    def __init__(self):
        self.scenario = "NORMAL"  # 'NORMAL', 'STRATA_STRAIN', 'CRITICAL_SUBSIDENCE'
        self.tick = 0
        
        # Geotechnical Node Coordinates (x in meters across Panel-04 cross-section)
        self.nodes = {
            "NODE-01": {"name": "Goaf Center Trough", "x": 0, "zone": "Active Depletion Center"},
            "NODE-02": {"name": "Shear Margin / Rib Pillar", "x": 55, "zone": "Maximum Tensile Strain"},
            "NODE-03": {"name": "Shaft Infrastructure Buffer", "x": 110, "zone": "Critical Asset Zone"},
            "NODE-04": {"name": "Far-Field Benchmark Node", "x": 180, "zone": "Undisturbed Stable Ground"}
        }

    def set_scenario(self, scenario: str):
        """Set demonstration scenario: NORMAL, STRATA_STRAIN, or CRITICAL_SUBSIDENCE."""
        self.scenario = scenario

    def calculate_peck_profile(self, s_max: float, inflection_i: float = 60.0):
        """
        Computes Peck's Gaussian Settlement Profile across the transverse panel line.
        S(x) = S_max * exp( -x^2 / (2 * i^2) )
        Tilt T(x) = - (x / i^2) * S(x)
        """
        x_pts = np.linspace(-150, 150, 61)
        settlement = s_max * np.exp(- (x_pts**2) / (2 * (inflection_i**2)))
        
        # Tilt is the first spatial derivative of settlement (mm/m)
        tilt = - (x_pts / (inflection_i**2)) * settlement * 10.0
        
        return x_pts.tolist(), settlement.tolist(), tilt.tolist()

    def get_mesh_data(self) -> dict:
        """
        Generates real-time telemetry across all 4 surface geotechnical nodes,
        evaluates progressive risk level, and computes the active settlement curve.
        """
        self.tick += 1
        t = self.tick * 0.15
        
        # Base multiplier based on scenario
        if self.scenario == "CRITICAL_SUBSIDENCE":
            risk_level = "LEVEL 4: CRITICAL (EVACUATION REQUIRED)"
            risk_color = "#FF1744"
            s_max = 145.0 + 3.0 * math.sin(t * 0.8)   # mm of sag
            vib_base = 0.42 + random.uniform(-0.04, 0.06) # g
            crack_base = 8.5 + 0.5 * math.sin(t)      # mm
            tilt_base = 12.4                          # mm/m
            alert_header = "🚨 CRITICAL HAZARD: RAPID SUBSIDENCE & STRATA COLLAPSE IMMINENT!"
            alert_sub = "Severe ground fissure detected at Node-02. Immediate withdrawal of personnel advised."
        elif self.scenario == "STRATA_STRAIN":
            risk_level = "LEVEL 2: ADVISORY (ACTIVE DEFORMATION)"
            risk_color = "#FFD600"
            s_max = 48.0 + 1.5 * math.sin(t * 0.5)
            vib_base = 0.12 + random.uniform(-0.02, 0.02)
            crack_base = 2.4 + random.uniform(-0.1, 0.1)
            tilt_base = 4.8
            alert_header = "▲ ADVISORY: ACCELERATED STRAIN DETECTED ACROSS PANEL-04"
            alert_sub = "Micro-seismic activity detected. Rate of tilt exceeds 3.0 mm/m baseline."
        else: # NORMAL
            risk_level = "LEVEL 1: NORMAL (SECURE BASELINE)"
            risk_color = "#00E676"
            s_max = 8.2 + 0.4 * math.sin(t * 0.3)
            vib_base = 0.012 + random.uniform(-0.002, 0.003)
            crack_base = 0.4 + random.uniform(-0.02, 0.02)
            tilt_base = 0.8
            alert_header = "● GROUND STRATA SECURE"
            alert_sub = "Surface deformation and seismic vibrations are within permissible DGMS norms."

        # Compute Peck's profile curve
        x_coords, settlement_curve, tilt_curve = self.calculate_peck_profile(s_max)

        # Build telemetry for individual nodes
        node_telemetry = {}
        for nid, info in self.nodes.items():
            dist = abs(info["x"])
            # Decay factors based on distance from goaf center
            dist_factor = math.exp(- (dist**2) / (2 * (60.0**2)))
            
            # Node specific readings
            node_settlement = round(s_max * dist_factor, 1)
            # Tilt is highest at inflection point (around 50-60m)
            node_tilt = round(tilt_base * (dist / 60.0) * math.exp(- (dist**2) / (2 * (60.0**2))) * 2.2, 2)
            if nid == "NODE-01":
                node_tilt = round(0.4 + random.uniform(-0.05, 0.05), 2)  # Zero tilt at symmetry apex
                
            node_vib = round(max(0.008, vib_base * (0.4 + 0.6 * dist_factor) + random.uniform(-0.003, 0.003)), 3)
            node_crack = round(max(0.1, crack_base * (0.2 + 0.8 * dist_factor) + random.uniform(-0.05, 0.05)), 1)
            
            node_telemetry[nid] = {
                "name": info["name"],
                "zone": info["zone"],
                "distance_m": info["x"],
                "settlement_mm": node_settlement,
                "tilt_mm_m": node_tilt,
                "vibration_g": node_vib,
                "crack_aperture_mm": node_crack,
                "battery_pct": 92 if nid != "NODE-03" else 88,
                "rf_rssi_dbm": -64 - int(dist * 0.08)
            }

        return {
            "timestamp": time.strftime("%H:%M:%S"),
            "scenario": self.scenario,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "alert_header": alert_header,
            "alert_sub": alert_sub,
            "max_settlement_mm": round(s_max, 1),
            "max_tilt_mm_m": round(tilt_base, 1),
            "peak_vibration_g": round(vib_base, 3),
            "nodes": node_telemetry,
            "profile_curve": {
                "x": x_coords,
                "settlement": settlement_curve,
                "tilt": tilt_curve
            }
        }

# Singleton instance
subsidence_mesh = SubsidenceMesh()
