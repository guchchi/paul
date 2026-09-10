"""
Underground Robotic Rover Telemetry Module
Handles communication with the ESP8266 edge sensor node dedicated to DHT11
temperature & humidity environmental telemetry, and provides modular expansion slots
for MQ gas sensors (MQ-4, MQ-7, MQ-135, O2) with realistic FLIR thermal feed.
"""
import time
import math
import random
import urllib.request
import json
import numpy as np
import cv2


def calculate_heat_index(temp_c: float, humidity: float) -> float:
    """Calculate Heat Index in Celsius using standard NOAA Rothfusz regression."""
    tf = temp_c * 9.0 / 5.0 + 32.0
    rh = max(0.0, min(100.0, humidity))
    hi_f = 0.5 * (tf + 61.0 + ((tf - 68.0) * 1.2) + (rh * 0.094))
    if hi_f >= 80.0:
        hi_f = (-42.379 + 2.04901523 * tf + 10.14333127 * rh
                - 0.22475541 * tf * rh - 0.00683783 * tf * tf
                - 0.05481717 * rh * rh + 0.00122874 * tf * tf * rh
                + 0.00085282 * tf * rh * rh - 0.00000199 * tf * tf * rh * rh)
    return round((hi_f - 32.0) * 5.0 / 9.0, 1)


def calculate_dew_point(temp_c: float, humidity: float) -> float:
    """Calculate Dew Point in Celsius using Magnus-Tetens formula."""
    a = 17.27
    b = 237.7
    rh = max(0.1, min(100.0, humidity))
    alpha = ((a * temp_c) / (b + temp_c)) + math.log(rh / 100.0)
    dp = (b * alpha) / (a - alpha)
    return round(dp, 1)


class RoverTelemetry:
    def __init__(self):
        self.node_ip = "192.168.4.1"
        self.is_connected = False
        self.last_poll_time = 0
        self.tick = 0
        
        # Scenarios: 'NORMAL', 'HEAT_STRESS', 'HIGH_HUMIDITY', 'GAS_SIMULATION'
        self.scenario = "NORMAL"
        
        # Last valid hardware cache
        self.last_hw_temp = None
        self.last_hw_humidity = None
        
        # Baseline environmental values for simulation
        self.base_temp = 26.5
        self.base_humidity = 64.0
        
        # History for charts (last 30 points)
        self.history = {
            "timestamp": [],
            "temp": [],
            "humidity": [],
            "heat_index": [],
            "dew_point": [],
            "ch4": [],
            "co": []
        }

    def set_scenario(self, scenario: str):
        """Set demonstration scenario: NORMAL, HEAT_STRESS, HIGH_HUMIDITY, or GAS_SIMULATION."""
        self.scenario = scenario

    def poll_esp8266(self, ip: str = None) -> dict:
        """Attempt to fetch real telemetry from ESP8266 REST endpoint."""
        target_ip = (ip or self.node_ip).strip()
        if not target_ip:
            target_ip = "192.168.4.1"
        url = f"http://{target_ip}/data"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'MineShield-Gateway'})
            with urllib.request.urlopen(req, timeout=1.2) as response:
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
        Reads live from the physical DHT11 sensor via ESP8266.
        Gas sensor slots (MQ-4, MQ-7, MQ-135, O2) remain in open-circuit state ('--' / Not Connected)
        unless demonstrated in simulation mode.
        """
        self.tick += 1
        t = self.tick * 0.2
        
        real_hw = self.poll_esp8266(esp_ip) if esp_ip else None
        
        # Stochastic micro-jitter for ambient air fluctuations
        jitter_temp = 0.25 * math.sin(t * 0.7) + random.uniform(-0.06, 0.06)
        jitter_hum = 0.6 * math.cos(t * 0.5) + random.uniform(-0.12, 0.12)
        
        if real_hw and "temperature" in real_hw:
            temp = float(real_hw["temperature"])
            humidity = float(real_hw.get("humidity", 60.0))
            self.last_hw_temp = temp
            self.last_hw_humidity = humidity
            hw_status = "ESP8266 HARDWARE SYNCED (DHT11 ACTIVE)"
            node_type = "Physical ESP8266 NodeMCU"
            source_mode = "LIVE_HARDWARE"
        else:
            source_mode = "SIMULATED"
            if self.scenario == "HEAT_STRESS":
                temp = 37.2 + 0.8 * math.sin(t * 0.9) + random.uniform(-0.1, 0.1)
                humidity = 82.5 + 1.2 * math.cos(t * 0.6) + random.uniform(-0.3, 0.3)
                hw_status = "SIMULATION: HEAT STRESS TEST"
                node_type = "Virtual Edge Node (Subsurface DHT11)"
            elif self.scenario == "HIGH_HUMIDITY":
                temp = 28.6 + 0.4 * math.sin(t * 0.5) + random.uniform(-0.08, 0.08)
                humidity = 92.4 + 1.5 * math.sin(t * 0.7) + random.uniform(-0.2, 0.2)
                hw_status = "SIMULATION: HIGH HUMIDITY TEST"
                node_type = "Virtual Edge Node (Subsurface DHT11)"
            else:  # NORMAL or GAS_SIMULATION
                temp = self.base_temp + jitter_temp
                humidity = self.base_humidity + jitter_hum
                hw_status = "AUTONOMOUS SIMULATION ACTIVE"
                node_type = "Virtual Edge Node (Subsurface DHT11)"
                
        # Calculate derived metrics from DHT11
        heat_index = calculate_heat_index(temp, humidity)
        dew_point = calculate_dew_point(temp, humidity)
        
        # Microclimate thermal status according to DGMS Coal Mines Regulations
        if temp >= 35.0 or heat_index >= 39.0:
            climate_status = "CRITICAL"
            alert_msg = "HIGH HEAT STRESS HAZARD: Temperature exceeds safe underground working limit (DGMS Reg 130)!"
        elif temp >= 31.0 or humidity >= 85.0:
            climate_status = "WARNING"
            alert_msg = "THERMAL COMFORT ADVISORY: High humidity/temperature in underground face heading."
        else:
            climate_status = "NORMAL"
            alert_msg = "Microclimate Safe: Temperature & humidity within DGMS standard comfort threshold."

        # Gas Matrix Handling:
        # If user explicitly triggers GAS_SIMULATION, show what data would look like.
        # Otherwise, sensors are NOT connected -> show '--' with unequipped status.
        if self.scenario == "GAS_SIMULATION":
            ch4_val = round(1.45 + 0.12 * math.sin(t * 1.2) + random.uniform(-0.02, 0.02), 2)
            co_val = round(42.0 + 2.5 * math.sin(t * 0.9) + random.uniform(-0.4, 0.4), 1)
            co2_val = round(1250.0 + random.uniform(-15, 15), 0)
            o2_val = round(19.1 + random.uniform(-0.1, 0.1), 1)
            
            ch4_display = f"{ch4_val} %"
            co_display = f"{co_val} ppm"
            co2_display = f"{co2_val} ppm"
            o2_display = f"{o2_val} %"
            gas_status = "CRITICAL"
            gas_installed = True
            gas_alert_msg = "GAS PREVIEW ACTIVE: High CH₄ & CO simulated on expansion channels."
        else:
            ch4_val = None
            co_val = None
            co2_val = None
            o2_val = None
            
            ch4_display = "--"
            co_display = "--"
            co2_display = "--"
            o2_display = "--"
            gas_status = "NOT_CONNECTED"
            gas_installed = False
            gas_alert_msg = "Gas sensor expansion slots open. Ready for MQ-4 / MQ-7 / MQ-135 / O₂ probes."
            
        readings = {
            "timestamp": time.strftime("%H:%M:%S"),
            "node_status": hw_status,
            "node_type": node_type,
            "source_mode": source_mode,
            "is_hardware_connected": self.is_connected,
            "scenario": self.scenario,
            "climate_status": climate_status,
            "gas_status": gas_status,
            "gas_installed": gas_installed,
            "gas_alert_msg": gas_alert_msg,
            "alert_message": gas_alert_msg if self.scenario == "GAS_SIMULATION" else alert_msg,
            # Primary DHT11 metrics
            "temperature": round(temp, 1),
            "humidity": round(humidity, 1),
            "heat_index": round(heat_index, 1),
            "dew_point": round(dew_point, 1),
            # Modular Gas Expansion Matrix
            "ch4_pct": ch4_val,
            "co_ppm": co_val,
            "co2_ppm": co2_val,
            "o2_pct": o2_val,
            "ch4_display": ch4_display,
            "co_display": co_display,
            "co2_display": co2_display,
            "o2_display": o2_display,
            # Rover Chassis Metrics
            "depth_m": -142.5,
            "battery_pct": max(15, 88 - int(self.tick * 0.01)),
            "rssi_dbm": -68 + random.randint(-2, 2) if not real_hw else int(real_hw.get("rssi", -68)),
            "pitch_deg": round(2.3 + 0.5 * math.sin(t * 0.5), 1),
            "roll_deg": round(1.1 + 0.4 * math.cos(t * 0.6), 1),
            "rover_speed_mps": 0.22
        }
        
        # History buffer for live charts
        self.history["timestamp"].append(readings["timestamp"])
        self.history["temp"].append(readings["temperature"])
        self.history["humidity"].append(readings["humidity"])
        self.history["heat_index"].append(readings["heat_index"])
        self.history["dew_point"].append(readings["dew_point"])
        if ch4_val is not None:
            self.history["ch4"].append(ch4_val)
            self.history["co"].append(co_val)
        
        max_pts = 30
        for k in self.history:
            if len(self.history[k]) > max_pts:
                self.history[k] = self.history[k][-max_pts:]
                
        return readings

    def generate_flir_frame(self, readings: dict) -> np.ndarray:
        """
        Synthesizes a realistic subterranean thermal/night-vision inspection frame
        simulating the rover's forward FLIR camera inside a coal mine roadway,
        overlaid with active DHT11 telemetry and modular gas channel indicators.
        """
        h, w = 360, 480
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        center_x, center_y = w // 2, h // 2 - 20
        y_indices, x_indices = np.indices((h, w))
        dist_from_center = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
        
        intensity = np.clip(180 - (dist_from_center * 0.7), 20, 200).astype(np.uint8)
        flir_base = cv2.applyColorMap(intensity, cv2.COLORMAP_INFERNO)
        
        noise = np.random.randint(-15, 15, (h, w, 3), dtype=np.int16)
        flir_noisy = np.clip(flir_base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frame = flir_noisy
        
        # Track lines
        cv2.line(frame, (center_x - 15, center_y + 35), (60, h - 5), (40, 40, 220), 2)
        cv2.line(frame, (center_x + 15, center_y + 35), (w - 60, h - 5), (40, 40, 220), 2)
        
        # Center reticle
        cv2.drawMarker(frame, (center_x, center_y), (0, 255, 255), cv2.MARKER_CROSS, 28, 1)
        cv2.circle(frame, (center_x, center_y), 38, (0, 255, 255), 1)
        
        # Top HUD Overlay
        cv2.rectangle(frame, (8, 8), (w - 8, 48), (0, 0, 0), -1)
        cv2.rectangle(frame, (8, 8), (w - 8, 48), (0, 200, 255), 1)
        
        cv2.putText(frame, "ROVER-01 [THERMAL FLIR IR-850nm]", (15, 26), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(frame, "SHAFT LEVEL: -142.5m | SEAM 03", (15, 42), 
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
        temp = readings.get('temperature', 26.5)
        hum = readings.get('humidity', 64.0)
        
        cv2.putText(frame, f"PITCH: {p:+.1f}*  ROLL: {r:+.1f}*", (15, h - 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        
        if readings.get("gas_installed"):
            ch4 = readings.get("ch4_pct", 0.0)
            co = readings.get("co_ppm", 0.0)
            cv2.putText(frame, f"DHT11: {temp:.1f}*C | CH4: {ch4}% | CO: {co}ppm", (15, h - 14), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255) if ch4 >= 1.0 else (0, 255, 120), 1)
        else:
            cv2.putText(frame, f"DHT11: {temp:.1f}*C | RH: {hum:.1f}% | GAS: N/C (OPEN SLOT)", (15, h - 14), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 120), 1)
            
        cv2.putText(frame, "STATUS: OK", (w - 100, h - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)
                    
        # Scanlines
        frame[::4, :, :] = (frame[::4, :, :] * 0.85).astype(np.uint8)
        
        return frame


rover_telemetry = RoverTelemetry()
