"""
Underground Robotic Rover Telemetry Module
Handles communication with the ESP8266 edge sensor node and simulates
deep-mine environmental metrics (MQ gas matrix, DHT11, and FLIR vision feed).
"""
import time
import math
import random
import urllib.request
import json
import numpy as np
import cv2

class RoverTelemetry:
    def __init__(self):
        self.node_ip = "192.168.4.1"
        self.is_connected = False
        self.last_poll_time = 0
        self.tick = 0
        
        # Simulation scenario: 'NORMAL', 'GAS_LEAK', 'LOW_O2'
        self.scenario = "NORMAL"
        
        # Baseline environmental values
        self.base_temp = 30.8
        self.base_humidity = 81.5
        self.base_ch4 = 0.24
        self.base_co = 12.0
        self.base_co2 = 640.0
        self.base_o2 = 20.8
        self.base_h2s = 0.8
        
        # History for charts (last 30 points)
        self.history = {
            "timestamp": [],
            "temp": [],
            "humidity": [],
            "ch4": [],
            "co": [],
            "co2": [],
            "o2": []
        }

    def set_scenario(self, scenario: str):
        """Set demonstration scenario: NORMAL, GAS_LEAK, or LOW_O2."""
        self.scenario = scenario

    def poll_esp8266(self, ip: str = None) -> dict:
        """Attempt to fetch real telemetry from ESP8266 REST endpoint."""
        target_ip = ip or self.node_ip
        url = f"http://{target_ip}/data"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'MineShield-Gateway'})
            with urllib.request.urlopen(req, timeout=0.35) as response:
                if response.status == 200:
                    raw_data = response.read().decode('utf-8')
                    parsed = json.loads(raw_data)
                    self.is_connected = True
                    return parsed
        except Exception:
            self.is_connected = False
        return None

    def get_readings(self, esp_ip: str = None) -> dict:
        """
        Get current telemetry readings.
        If ESP8266 is reachable, incorporates real hardware DHT11 data.
        Otherwise, runs realistic deep-mine stochastic physics model.
        """
        self.tick += 1
        t = self.tick * 0.2
        
        # Check physical hardware first
        real_hw = self.poll_esp8266(esp_ip)
        
        # Stochastic micro-jitter (Brownian drift + sine harmonics)
        jitter_temp = 0.25 * math.sin(t * 0.7) + random.uniform(-0.08, 0.08)
        jitter_hum = 0.6 * math.cos(t * 0.5) + random.uniform(-0.15, 0.15)
        
        if real_hw and "temperature" in real_hw:
            temp = float(real_hw["temperature"])
            humidity = float(real_hw.get("humidity", self.base_humidity))
            hw_status = "ESP8266 HARDWARE SYNCED"
            node_type = "Physical ESP8266 Edge Node"
        else:
            temp = self.base_temp + jitter_temp
            humidity = self.base_humidity + jitter_hum
            hw_status = "AUTONOMOUS SIMULATION ACTIVE"
            node_type = "Virtual Edge Node (Subsurface CH-04)"
            
        # Apply Scenario Adjustments
        if self.scenario == "GAS_LEAK":
            ch4 = 1.45 + 0.15 * math.sin(t * 1.2) + random.uniform(-0.02, 0.02)
            co = 42.0 + 3.0 * math.sin(t * 0.9) + random.uniform(-0.5, 0.5)
            co2 = 1250.0 + random.uniform(-20, 20)
            o2 = 19.1 + random.uniform(-0.1, 0.1)
            h2s = 4.2 + random.uniform(-0.2, 0.2)
            gas_status = "CRITICAL"
            alert_msg = "HAZARD: Elevated Methane & Carbon Monoxide Levels!"
        elif self.scenario == "LOW_O2":
            ch4 = self.base_ch4 + random.uniform(-0.01, 0.01)
            co = 18.0 + random.uniform(-0.5, 0.5)
            co2 = 1650.0 + random.uniform(-30, 30)
            o2 = 17.6 + 0.2 * math.sin(t * 0.5)
            h2s = 1.1 + random.uniform(-0.1, 0.1)
            gas_status = "WARNING"
            alert_msg = "ADVISORY: Depleted Oxygen (Hypoxia Risk)!"
        else: # NORMAL
            ch4 = max(0.05, self.base_ch4 + 0.03 * math.sin(t * 0.4) + random.uniform(-0.01, 0.01))
            co = max(4.0, self.base_co + 1.2 * math.cos(t * 0.3) + random.uniform(-0.3, 0.3))
            co2 = self.base_co2 + 15.0 * math.sin(t * 0.6) + random.uniform(-5, 5)
            o2 = max(19.8, min(20.95, self.base_o2 + 0.08 * math.sin(t * 0.2) + random.uniform(-0.02, 0.02)))
            h2s = max(0.1, self.base_h2s + 0.08 * math.sin(t * 0.8) + random.uniform(-0.02, 0.02))
            gas_status = "NORMAL"
            alert_msg = "All Atmospheric Gas Levels Within DGMS Safe Thresholds."

        # Compile full telemetry packet
        readings = {
            "timestamp": time.strftime("%H:%M:%S"),
            "node_status": hw_status,
            "node_type": node_type,
            "is_hardware_connected": self.is_connected,
            "scenario": self.scenario,
            "gas_status": gas_status,
            "alert_message": alert_msg,
            # Environmental (DHT11)
            "temperature": round(temp, 1),
            "humidity": round(humidity, 1),
            # Gas Matrix (MQ Series)
            "ch4_pct": round(ch4, 2),        # % Vol (MQ-4)
            "co_ppm": round(co, 1),           # PPM (MQ-7)
            "co2_ppm": round(co2, 0),         # PPM (MQ-135)
            "o2_pct": round(o2, 1),           # % Vol (Electrochemical O2)
            "h2s_ppm": round(h2s, 1),         # PPM (MQ-136)
            # Rover Chassis Metrics
            "depth_m": -142.5,                # Subsurface shaft level 3
            "battery_pct": max(15, 88 - int(self.tick * 0.01)),
            "rssi_dbm": -68 + random.randint(-2, 2),
            "pitch_deg": round(2.3 + 0.5 * math.sin(t * 0.5), 1),
            "roll_deg": round(1.1 + 0.4 * math.cos(t * 0.6), 1),
            "rover_speed_mps": 0.22
        }
        
        # Maintain history buffer for live graphs
        self.history["timestamp"].append(readings["timestamp"])
        self.history["temp"].append(readings["temperature"])
        self.history["humidity"].append(readings["humidity"])
        self.history["ch4"].append(readings["ch4_pct"])
        self.history["co"].append(readings["co_ppm"])
        self.history["co2"].append(readings["co2_ppm"])
        self.history["o2"].append(readings["o2_pct"])
        
        # Keep window size at 30
        max_pts = 30
        for k in self.history:
            if len(self.history[k]) > max_pts:
                self.history[k] = self.history[k][-max_pts:]
                
        return readings

    def generate_flir_frame(self, readings: dict) -> np.ndarray:
        """
        Synthesizes a realistic subterranean thermal/night-vision inspection frame
        simulating the rover's forward FLIR camera inside a coal mine roadway.
        """
        h, w = 360, 480
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 1. Base gradient to simulate an arched mine tunnel perspective
        center_x, center_y = w // 2, h // 2 - 20
        y_indices, x_indices = np.indices((h, w))
        dist_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
        
        # Tunnel wall shading
        intensity = np.clip(180 - (dist_from_center * 0.7), 20, 200).astype(np.uint8)
        
        # Apply thermal color mapping (Inferno / Ironbow)
        flir_base = cv2.applyColorMap(intensity, cv2.COLORMAP_INFERNO)
        
        # Add subtle noise and strata texture
        noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
        flir_noisy = np.clip(flir_base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frame = flir_noisy
        
        # 2. Draw tunnel floor track lines (Perspective rails)
        cv2.line(frame, (center_x - 15, center_y + 35), (60, h - 5), (40, 40, 220), 2)
        cv2.line(frame, (center_x + 15, center_y + 35), (w - 60, h - 5), (40, 40, 220), 2)
        
        # 3. Draw Crosshair / Reticle in center
        cv2.drawMarker(frame, (center_x, center_y), (0, 255, 255), cv2.MARKER_CROSS, 28, 1)
        cv2.circle(frame, (center_x, center_y), 38, (0, 255, 255), 1)
        
        # 4. HUD Telemetry Overlay (Rugged scientific style)
        cv2.rectangle(frame, (8, 8), (w - 8, 48), (0, 0, 0), -1)
        cv2.rectangle(frame, (8, 8), (w - 8, 48), (0, 200, 255), 1)
        
        # Header text
        cv2.putText(frame, "ROVER-01 [THERMAL FLIR IR-850nm]", (15, 26), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(frame, f"SHAFT LEVEL: -142.5m | SEAM 03", (15, 42), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        
        time_str = readings.get("timestamp", time.strftime("%H:%M:%S"))
        cv2.putText(frame, f"UTC {time_str}", (w - 110, 26), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)
        cv2.putText(frame, f"BAT: {readings.get('battery_pct', 86)}%", (w - 90, 42), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

        # Bottom telemetry bar
        cv2.rectangle(frame, (8, h - 45), (w - 8, h - 8), (0, 0, 0), -1)
        cv2.rectangle(frame, (8, h - 45), (w - 8, h - 8), (0, 200, 255), 1)
        
        p = readings.get('pitch_deg', 0.0)
        r = readings.get('roll_deg', 0.0)
        ch4 = readings.get('ch4_pct', 0.0)
        co = readings.get('co_ppm', 0.0)
        
        cv2.putText(frame, f"PITCH: {p:+.1f}*  ROLL: {r:+.1f}*", (15, h - 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        cv2.putText(frame, f"CH4: {ch4}%  CO: {co}ppm", (15, h - 14), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255) if ch4 < 1.0 else (0, 0, 255), 1)
        cv2.putText(frame, "LIDAR: CLEAR", (w - 115, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)
                    
        # Subtle horizontal scanlines
        frame[::4, :, :] = (frame[::4, :, :] * 0.85).astype(np.uint8)
        
        return frame

# Singleton instance
rover_telemetry = RoverTelemetry()
