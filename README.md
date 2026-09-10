# SIH Smart Mine Safety, Subsidence & Hazard Early-Warning System

> **Designed for the Smart India Hackathon (SIH 2026)**  
> *Project: Indigenous Real-Time Mine Subsidence & Subsurface Hazard Command Portal*  
> *Team: Student Engineering Team (Module: MineShield-SCADA)*

---

## 🌟 Executive Summary

Underground coal extraction triggers rock strata movement, leading to **surface subsidence troughs, fissures, structural damage, and subsurface spontaneous combustion (coal seam fires)**. 

Conventional monitoring methods (intermittent leveling, GNSS surveys, or satellite InSAR) are periodic and cannot provide immediate early-warning before a critical failure occurs.

This project delivers a **low-cost, real-time, indigenous mine safety command system** integrating three vital safety tiers into a unified SCADA portal:
1. **Surface Optical Hazard Surveillance (Webcam AI)**: Early detection of fire and smoke initiation using OpenCV flame flicker dynamics + YOLOv8 + temporal persistence.
2. **Subsurface Robotic Rover Telemetry (ESP8266 + DHT11 + MQ Gas Matrix)**: Deep-mine exploration at Seam-03 (-142.5m) monitoring environmental heat, humidity, and explosive/toxic gases ($CH_4, CO, CO_2, O_2, H_2S$), backed by a synthetic FLIR thermal roadway camera.
3. **Overground Subsidence Mesh (Distributed Geotechnical Grid)**: 4-Node surface network monitoring inclinometer tilt, micro-seismic tremors, and fissure opening, integrated with analytical **Peck's Gaussian Settlement Profile** and 4-tier progressive risk warnings (Normal ➔ Advisory ➔ Warning ➔ Evacuation).

---

## 🚀 The 3 Core Operational Modules

```
                        ┌─────────────────────────────────────────────────┐
                        │      🔐 DGMS INDUSTRIAL SCADA LOGIN GATEWAY      │
                        │                 (Operator Auth)                 │
                        └────────────────────────┬────────────────────────┘
                                                 │
                     ┌───────────────────────────┴───────────────────────────┐
                     ▼                                                       ▼
      ┌─────────────────────────────┐                         ┌─────────────────────────────┐
      │ [OPTION 01] OPTICAL HAZARD  │                         │ [OPTION 02] SUBSURFACE ROVER│
      │ • Laptop Webcam OpenCV/YOLO │                         │ • ESP8266 + DHT11 Hardware  │
      │ • Flame Dynamics + Smoke    │                         │ • MQ Gas Matrix (CH4, CO...)│
      │ • Forensic Evidence Gallery │                         │ • Forward FLIR Night-Vision │
      └─────────────────────────────┘                         └─────────────────────────────┘
                                                 │
                                                 ▼
                              ┌─────────────────────────────────────┐
                              │ [OPTION 03] OVERGROUND SUBSIDENCE   │
                              │ • 4 Surface Geotechnical Nodes      │
                              │ • Inclinometer Tilt & Vibration (g) │
                              │ • Peck's Analytical Trough Curve    │
                              │ • Progressive Risk Evacuation Alert │
                              └─────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
paul/
├── app_dashboard.py          # Central Industrial SCADA Web Station (Login + 3 Modules)
├── hardware/
│   └── esp8266_dht11_node.ino# Student Arduino C++ sketch for ESP8266 NodeMCU + DHT11
├── src/
│   ├── config.py             # System thresholds, camera parameters, and paths
│   ├── telemetry/
│   │   ├── rover_telemetry.py      # ESP8266 HTTP poller + dynamic MQ/DHT11 sensor model + FLIR view
│   │   └── subsidence_telemetry.py # Surface geotechnical mesh + Peck's profile calculator
│   ├── detection/
│   │   ├── fire_detector.py   # Hybrid CV + YOLOv8 flame and smoke detector
│   │   ├── cv_fire_filter.py  # HSV/YCrCb color segmentation + flicker dynamics
│   │   └── temporal_filter.py # Temporal persistence tracker (eliminates false positives)
│   ├── alerts/
│   │   └── alert_manager.py   # Acoustic siren, pyttsx3 speech synthesizer, and snapshot logger
│   └── utils/
│       └── visualizer.py      # Real-time HUD overlay renderer
├── incidents/                 # Timestamped forensic evidence snapshots
├── models/                    # Model weights storage
├── main.py                    # Standalone Desktop OpenCV HUD window
├── test_pipeline.py           # Automated unit verification test suite
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation & pitch guide
```

---

## ⚡ Quickstart Guide

### 1. Run Automated Test Verification
Ensure all detection engines and telemetry generators pass:
```bash
python test_pipeline.py
```

### 2. Launch the SCADA Command Web Station
Start the unified browser application:
```bash
streamlit run app_dashboard.py
```
*Tip: Once the browser opens, press **`F11`** for Fullscreen Mode to give judges the complete industrial control-room feel!*

### 3. Login to the Terminal
- **Operator ID**: `MSO-ZONE4-INCHARGE` (or `operator_01`)
- **Security Key**: `sih2026`

---

## 🔌 Hardware Demonstration (ESP8266 + DHT11)

For live evaluation, place your breadboard containing the **ESP8266 NodeMCU** and **DHT11** on the table.

1. The embedded C++ firmware is located in `hardware/esp8266_dht11_node.ino`.
2. Connect the DHT11 data pin to **Pin D4 (GPIO 2)** and ground/VCC.
3. The NodeMCU connects to Wi-Fi (or broadcasts SoftAP `MineRover_Node_AP`) and serves telemetry on `http://<IP>/data`.
4. On **Module 2**, enter the Node IP (default: `192.168.4.1`) and select `Auto-Detect Hardware`.
5. **Fail-Safe Protection**: If Wi-Fi is unavailable in the competition hall, the system automatically runs the **Autonomous Realistic Simulation** so your live presentation never fails or freezes!

---

## 🏆 Presentation & Pitch Guide for Evaluators

When presenting this prototype to SIH judges:

1. **System Architecture (Edge-to-Gateway)**:
   - *"Sir, underground we have the ESP8266 edge sensor node mounted on our scout rover streaming high-frequency environmental telemetry to our surface laptop gateway. The surface gateway hosts this central SCADA command station, performs AI hazard detection, and synchronizes the geotechnical mesh."*

2. **Demonstrate Module 1 (Optical Surveillance)**:
   - Click **▶️ Launch Optical Hazard Camera**.
   - Show how the temporal persistence filter prevents false alarms from yellow lamps or clothing.
   - Click **Test Siren** or **Test Voice** to demonstrate multi-modal audio evacuation warnings.

3. **Demonstrate Module 2 (Underground Rover)**:
   - Point to your breadboard (ESP8266 + DHT11).
   - Show the live **DHT11** temperature and humidity alongside the **MQ gas matrix** ($CH_4, CO, CO_2, O_2$).
   - Click **⚠️ Inject Gas Leak Hazard**: Watch the Methane and Carbon Monoxide levels spike and the status switch to `🚨 CRITICAL MINE GAS ALARM` with live trend line charts.

4. **Demonstrate Module 3 (Overground Subsidence Mesh)**:
   - Explain how 4 surface nodes track ground movement across extraction Panel-04.
   - Show the **Peck's Gaussian Settlement Profile** curve calculated in real-time.
   - Click **🚨 Severe Collapse Hazard**: Show how ground tilt exceeds DGMS thresholds ($>6.0\text{ mm/m}$), triggering immediate Level-4 Evacuation alerts.