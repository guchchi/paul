"""
Overground Ground Subsidence & Geotechnical Mesh Telemetry Module
Integrates real hardware Arduino MPU6050 (Accelerometer + Gyroscope) on COM7
with Peck's analytical subsidence formula and geotechnical mesh telemetry.
"""
import time
import math
import random
import re
import threading
import numpy as np
import serial
import serial.tools.list_ports


class SubsidenceMesh:
    def __init__(self, default_port: str = "COM7", baudrate: int = 9600):
        self.port = default_port
        self.baud = baudrate
        self.running = True
        
        # Connection status
        self.is_connected = False
        self.status_message = "Initializing serial reader..."
        self.last_read_time = 0
        self.tick = 0
        
        # Scenarios for presentation / simulation fallback
        self.scenario = "NORMAL"  # 'NORMAL', 'STRATA_STRAIN', 'CRITICAL_SUBSIDENCE'
        
        # Latest parsed readings from MPU6050
        self.ax_raw = 0
        self.ay_raw = 0
        self.az_raw = 16384
        self.gx_raw = 0
        self.gy_raw = 0
        self.gz_raw = 0
        
        self.accel_x_g = 0.0
        self.accel_y_g = 0.0
        self.accel_z_g = 1.0
        
        self.pitch_deg = 0.0
        self.roll_deg = 0.0
        self.tilt_deg = 0.0
        self.tilt_mm_m = 0.0
        self.vibration_g = 0.008
        
        self.gyro_x_dps = 0.0
        self.gyro_y_dps = 0.0
        self.gyro_z_dps = 0.0
        self.gyro_total_dps = 0.0
        
        # History for real-time charts (last 35 points)
        self.history = {
            "timestamp": [],
            "tilt_mm_m": [],
            "pitch_deg": [],
            "roll_deg": [],
            "vibration_g": [],
            "gyro_dps": []
        }
        
        # Geotechnical Node Coordinates (x in meters across Panel-04)
        self.nodes = {
            "NODE-01": {"name": "Goaf Center (MPU6050 Live)", "x": 0, "zone": "Active Depletion Center"},
            "NODE-02": {"name": "Shear Margin / Rib Pillar", "x": 55, "zone": "Maximum Tensile Strain"},
            "NODE-03": {"name": "Shaft Infrastructure Buffer", "x": 110, "zone": "Critical Asset Zone"},
            "NODE-04": {"name": "Far-Field Benchmark Node", "x": 180, "zone": "Undisturbed Stable Ground"}
        }
        
        self.ser = None
        self.lock = threading.Lock()
        
        # Start background reader thread
        self.thread = threading.Thread(target=self._serial_worker, daemon=True)
        self.thread.start()

    def set_scenario(self, scenario: str):
        """Set demonstration scenario fallback: NORMAL, STRATA_STRAIN, or CRITICAL_SUBSIDENCE."""
        self.scenario = scenario

    def set_port(self, port: str):
        """Change serial COM port and reconnect."""
        if port != self.port:
            with self.lock:
                self.port = port
                if self.ser:
                    try:
                        self.ser.close()
                    except Exception:
                        pass
                    self.ser = None
                self.is_connected = False
                self.status_message = f"Connecting to {port}..."

    def _serial_worker(self):
        """Continuous background worker reading from Arduino serial."""
        while self.running:
            if self.ser is None or not self.ser.is_open:
                try:
                    self.ser = serial.Serial(self.port, self.baud, timeout=1.0)
                    time.sleep(0.1)
                    with self.lock:
                        self.is_connected = True
                        self.status_message = f"Connected to {self.port} (MPU6050 Live Sync)"
                except serial.SerialException as e:
                    with self.lock:
                        self.is_connected = False
                        err_str = str(e)
                        if "Access is denied" in err_str or "PermissionError" in err_str:
                            self.status_message = f"Port {self.port} is busy! Please close Arduino IDE Serial Monitor."
                        else:
                            self.status_message = f"Waiting for {self.port} to connect..."
                    self.ser = None
                    time.sleep(1.5)
                    continue
                except Exception as e:
                    with self.lock:
                        self.is_connected = False
                        self.status_message = f"Serial error: {str(e)[:40]}"
                    self.ser = None
                    time.sleep(1.5)
                    continue

            # Read line
            try:
                raw_line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if raw_line:
                    self._parse_line(raw_line)
            except Exception:
                with self.lock:
                    self.is_connected = False
                    self.status_message = f"Connection lost on {self.port}. Retrying..."
                try:
                    if self.ser:
                        self.ser.close()
                except Exception:
                    pass
                self.ser = None
                time.sleep(1.0)

    def _parse_line(self, line: str):
        """
        Parses serial line formatted as:
        Accel: <ax>  <ay>  <az>    Gyro: <gx>  <gy>  <gz>
        """
        match = re.search(
            r"Accel:\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)\s+Gyro:\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)",
            line
        )
        if not match:
            return
            
        try:
            ax = float(match.group(1))
            ay = float(match.group(2))
            az = float(match.group(3))
            gx = float(match.group(4))
            gy = float(match.group(5))
            gz = float(match.group(6))
        except (ValueError, IndexError):
            return

        with self.lock:
            self.ax_raw = int(ax)
            self.ay_raw = int(ay)
            self.az_raw = int(az)
            self.gx_raw = int(gx)
            self.gy_raw = int(gy)
            self.gz_raw = int(gz)
            
            # Accelerometer in g (sensitivity ±2g = 16384 LSB/g)
            self.accel_x_g = ax / 16384.0
            self.accel_y_g = ay / 16384.0
            self.accel_z_g = az / 16384.0 if az != 0 else 0.001
            
            # Tilt Angles in degrees using standard Euler pitch/roll
            self.pitch_deg = round(math.degrees(math.atan2(-self.accel_x_g, math.sqrt(self.accel_y_g**2 + self.accel_z_g**2))), 2)
            self.roll_deg = round(math.degrees(math.atan2(self.accel_y_g, self.accel_z_g)), 2)
            self.tilt_deg = round(math.sqrt(self.pitch_deg**2 + self.roll_deg**2), 2)
            
            # Geotechnical Tilt in mm/m (DGMS coal mining regulation metric)
            self.tilt_mm_m = round(math.tan(math.radians(min(85.0, self.tilt_deg))) * 1000.0, 1)
            
            # Vibration / Dynamic Tremor in g
            tot_accel = math.sqrt(self.accel_x_g**2 + self.accel_y_g**2 + self.accel_z_g**2)
            self.vibration_g = round(abs(tot_accel - 1.0), 3)
            
            # Gyroscope in degrees per second (sensitivity ±250°/s = 131 LSB/(°/s))
            self.gyro_x_dps = round(gx / 131.0, 1)
            self.gyro_y_dps = round(gy / 131.0, 1)
            self.gyro_z_dps = round(gz / 131.0, 1)
            self.gyro_total_dps = round(math.sqrt(self.gyro_x_dps**2 + self.gyro_y_dps**2 + self.gyro_z_dps**2), 1)
            
            self.last_read_time = time.time()
            self.is_connected = True
            
            # History buffer update
            now_str = time.strftime("%H:%M:%S")
            self.history["timestamp"].append(now_str)
            self.history["tilt_mm_m"].append(self.tilt_mm_m)
            self.history["pitch_deg"].append(self.pitch_deg)
            self.history["roll_deg"].append(self.roll_deg)
            self.history["vibration_g"].append(self.vibration_g)
            self.history["gyro_dps"].append(self.gyro_total_dps)
            
            max_pts = 35
            for k in self.history:
                if len(self.history[k]) > max_pts:
                    self.history[k] = self.history[k][-max_pts:]

    def calculate_peck_profile(self, s_max: float, inflection_i: float = 60.0):
        """
        Computes Peck's Gaussian Settlement Profile across the transverse panel line.
        S(x) = S_max * exp( -x^2 / (2 * i^2) )
        Tilt T(x) = - (x / i^2) * S(x)
        """
        x_pts = np.linspace(-150, 150, 61)
        settlement = s_max * np.exp(- (x_pts**2) / (2 * (inflection_i**2)))
        tilt = - (x_pts / (inflection_i**2)) * settlement * 10.0
        return x_pts.tolist(), settlement.tolist(), tilt.tolist()

    def get_mesh_data(self) -> dict:
        """
        Returns live telemetry for Module 3.
        Incorporates physical MPU6050 readings if hardware is streaming,
        otherwise uses calibrated geotechnical physics model.
        """
        self.tick += 1
        t = self.tick * 0.15
        
        now = time.time()
        with self.lock:
            hw_live = self.is_connected and (now - self.last_read_time < 3.0)
            status_msg = self.status_message
            port = self.port
            
            ax_r = self.ax_raw
            ay_r = self.ay_raw
            az_r = self.az_raw
            gx_r = self.gx_raw
            gy_r = self.gy_raw
            gz_r = self.gz_raw
            
            pitch = self.pitch_deg
            roll = self.roll_deg
            tilt_deg = self.tilt_deg
            tilt_mm_m = self.tilt_mm_m
            vib_g = self.vibration_g
            gyro_tot = self.gyro_total_dps
            
        if hw_live:
            source = "LIVE_HARDWARE_COM7"
            # Calibrate settlement dynamically based on physical tilt and vibration
            s_max = round(max(5.0, 8.0 + (tilt_mm_m * 0.8) + (vib_g * 60.0)), 1)
            
            if tilt_mm_m > 12.0 or vib_g > 0.30:
                risk_level = "LEVEL 4: CRITICAL (IMMINENT COLLAPSE)"
                alert_header = "CRITICAL HAZARD: SEVERE STRATA DISPLACEMENT"
                alert_sub = f"Live MPU6050 Inclinometer: Tilt {tilt_mm_m:.1f} mm/m, Vibration {vib_g:.3f} g!"
            elif tilt_mm_m > 4.5 or vib_g > 0.08:
                risk_level = "LEVEL 2: ADVISORY (ACTIVE STRAIN)"
                alert_header = "ADVISORY: ELEVATED GROUND STRAIN DETECTED"
                alert_sub = f"Live MPU6050 Inclinometer: Tilt {tilt_mm_m:.1f} mm/m, Gyro {gyro_tot:.1f}°/s."
            else:
                risk_level = "LEVEL 1: NORMAL (STRATA SECURE)"
                alert_header = "GROUND STRATA STABLE (LIVE MPU6050)"
                alert_sub = f"Physical MPU6050 active on {port}. Inclinometer & vibration within safe norms."
        else:
            source = "AUTONOMOUS_SIMULATION"
            # Fallback based on scenario
            if self.scenario == "CRITICAL_SUBSIDENCE":
                risk_level = "LEVEL 4: CRITICAL (IMMINENT COLLAPSE)"
                s_max = 145.0 + 3.0 * math.sin(t * 0.8)
                vib_g = 0.42 + random.uniform(-0.04, 0.06)
                tilt_mm_m = 12.4 + random.uniform(-0.2, 0.2)
                tilt_deg = 0.71
                pitch = 0.5
                roll = 0.5
                gyro_tot = 18.5
                alert_header = "CRITICAL HAZARD: RAPID SUBSIDENCE DETECTED"
                alert_sub = "Severe ground fissure imminent at Node-02. Evacuation threshold reached."
            elif self.scenario == "STRATA_STRAIN":
                risk_level = "LEVEL 2: ADVISORY (ACTIVE STRAIN)"
                s_max = 48.0 + 1.5 * math.sin(t * 0.5)
                vib_g = 0.12 + random.uniform(-0.02, 0.02)
                tilt_mm_m = 4.8 + random.uniform(-0.1, 0.1)
                tilt_deg = 0.27
                pitch = 0.2
                roll = 0.2
                gyro_tot = 6.2
                alert_header = "ADVISORY: ACCELERATED STRAIN DETECTED"
                alert_sub = "Micro-seismic activity registered across Panel 4 shear zone."
            else:  # NORMAL
                risk_level = "LEVEL 1: NORMAL (STRATA SECURE)"
                s_max = 8.2 + 0.4 * math.sin(t * 0.3)
                vib_g = 0.012 + random.uniform(-0.002, 0.003)
                tilt_mm_m = 0.8 + random.uniform(-0.05, 0.05)
                tilt_deg = 0.05
                pitch = 0.03
                roll = 0.04
                gyro_tot = 0.5
                alert_header = "GROUND STRATA STABLE"
                alert_sub = "Surface deformation and seismic vibrations are within permissible DGMS norms."

        # Compute Peck's profile curve
        x_coords, settlement_curve, tilt_curve = self.calculate_peck_profile(s_max)
        
        # Build telemetry for 4 surface nodes
        node_telemetry = {}
        for nid, info in self.nodes.items():
            dist = abs(info["x"])
            dist_factor = math.exp(- (dist**2) / (2 * (60.0**2)))
            
            if nid == "NODE-01":
                # Primary hardware node directly mapped to physical MPU6050
                node_settlement = round(s_max, 1)
                node_tilt = round(tilt_mm_m, 2)
                node_vib = round(vib_g, 3)
            else:
                node_settlement = round(s_max * dist_factor, 1)
                node_tilt = round(tilt_mm_m * (dist / 60.0) * dist_factor * 1.8, 2)
                node_vib = round(max(0.006, vib_g * (0.3 + 0.7 * dist_factor)), 3)
                
            node_telemetry[nid] = {
                "name": info["name"],
                "zone": info["zone"],
                "distance_m": info["x"],
                "settlement_mm": node_settlement,
                "tilt_mm_m": node_tilt,
                "vibration_g": node_vib,
                "battery_pct": 94 if nid != "NODE-03" else 89,
                "rf_rssi_dbm": -62 - int(dist * 0.07)
            }
            
        return {
            "timestamp": time.strftime("%H:%M:%S"),
            "port": port,
            "hw_live": hw_live,
            "status_message": status_msg,
            "source": source,
            "risk_level": risk_level,
            "alert_header": alert_header,
            "alert_sub": alert_sub,
            "max_settlement_mm": round(s_max, 1),
            "inflection_point_i": 60.0,
            # Live MPU6050 Telemetry
            "tilt_deg": round(tilt_deg, 2),
            "tilt_mm_m": round(tilt_mm_m, 1),
            "vibration_g": round(vib_g, 3),
            "pitch_deg": round(pitch, 1),
            "roll_deg": round(roll, 1),
            "gyro_total_dps": round(gyro_tot, 1),
            # Raw hardware readings
            "ax_raw": ax_r,
            "ay_raw": ay_r,
            "az_raw": az_r,
            "gx_raw": gx_r,
            "gy_raw": gy_r,
            "gz_raw": gz_r,
            # Node mesh and curves
            "nodes": node_telemetry,
            "profile_curve": {
                "x": x_coords,
                "settlement": settlement_curve,
                "tilt": tilt_curve
            }
        }


subsidence_mesh = SubsidenceMesh(default_port="COM7", baudrate=9600)
