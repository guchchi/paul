"""
SIH 2026 - Mine Subsidence & Safety Early Warning System
A clean, student-built prototype monitoring:
1. Surface Fire & Smoke Camera (Webcam + OpenCV/YOLO)
2. Underground Rover Sensors (ESP8266 + DHT11 + MQ Gases)
3. Surface Subsidence Mesh (Panel 4 Geotechnical Nodes)
"""
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# System modules
from src.config import config, INCIDENTS_DIR
from src.detection.fire_detector import FireDetector
from src.alerts.alert_manager import alert_manager
from src.utils.visualizer import hud_visualizer
from src.telemetry.rover_telemetry import rover_telemetry
from src.telemetry.subsidence_telemetry import subsidence_mesh

# -------------------------------------------------------------
# Streamlit Page Setup
# -------------------------------------------------------------
st.set_page_config(
    page_title="Mine Safety & Subsidence System",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------------------
# Clean, Neutral Styling
# -------------------------------------------------------------
st.markdown("""
<style>
    body, .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Session State
# -------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "current_view" not in st.session_state:
    st.session_state["current_view"] = "hub"  # 'hub', 'camera', 'rover', 'subsidence'
if "esp_ip" not in st.session_state:
    st.session_state["esp_ip"] = "192.168.4.1"
if "operator_name" not in st.session_state:
    st.session_state["operator_name"] = "Safety Officer"

# -------------------------------------------------------------
# Top Navigation Header (Clean & Minimal)
# -------------------------------------------------------------
def render_header():
    col_left, col_right = st.columns([3, 1])
    with col_left:
        st.markdown(
            f"### Mine Subsidence & Safety Early Warning System\n"
            f"<span style='color: #8b949e; font-size: 0.88rem;'>Smart India Hackathon 2026 Prototype &nbsp;|&nbsp; Active Area: <b>Panel 4</b> &nbsp;|&nbsp; Operator: <b>{st.session_state['operator_name']}</b></span>", 
            unsafe_allow_html=True
        )
    with col_right:
        c1, c2 = st.columns(2)
        if st.session_state["current_view"] != "hub":
            if c1.button("⬅️ Home", use_container_width=True):
                st.session_state["current_view"] = "hub"
                st.rerun()
        if c2.button("Logout", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["current_view"] = "hub"
            st.rerun()
            
    st.divider()

# -------------------------------------------------------------
# SCREEN 0: Simple Login Form
# -------------------------------------------------------------
def render_login():
    st.markdown("<br>", unsafe_allow_html=True)
    _, col_center, _ = st.columns([1, 1.4, 1])
    
    with col_center:
        with st.container(border=True):
            st.subheader("Mine Safety Portal Login")
            st.caption("Enter credentials to access monitoring dashboard")
            
            with st.form("login_form"):
                username = st.text_input("User ID", value="operator_01")
                password = st.text_input("Password", value="sih2026", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True)
                
                if submitted:
                    if username and password:
                        st.session_state["authenticated"] = True
                        st.session_state["operator_name"] = username
                        st.rerun()
                    else:
                        st.error("Please enter both username and password")
                        
            st.markdown("<div style='text-align: center; color: #888; margin: 0.5rem 0;'>or</div>", unsafe_allow_html=True)
            if st.button("Quick Demo Login (for Presentation)", use_container_width=True):
                st.session_state["authenticated"] = True
                st.session_state["operator_name"] = "SIH Judge / Evaluator"
                st.rerun()

# -------------------------------------------------------------
# SCREEN 1: Simple 3-Module Selection (Hub)
# -------------------------------------------------------------
def render_hub():
    render_header()
    
    st.write("#### Monitoring Modules")
    st.caption("Select a module below to view live sensor telemetry and alerts:")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.markdown("#### 📹 1. Fire & Smoke Camera")
            st.write("Real-time optical fire and smoke detection on mine surface infrastructure using the laptop webcam.")
            st.markdown("---")
            st.markdown("- **Sensor:** Laptop Camera (Index 0)")
            st.markdown("- **Detection:** Flame Dynamics & Smoke")
            st.markdown("- **Status:** Ready")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Open Camera Module", key="btn_open_cam", use_container_width=True):
                st.session_state["current_view"] = "camera"
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown("#### 🤖 2. Underground Rover")
            st.write("Subsurface environmental data from mobile rover: temperature, humidity, and toxic mine gases (CH4, CO).")
            st.markdown("---")
            st.markdown("- **Hardware:** ESP8266 + DHT11")
            st.markdown("- **Gases:** CH4, CO, CO2, O2")
            st.markdown("- **Depth:** -142.5m (Seam 3)")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Open Rover Module", key="btn_open_rover", use_container_width=True):
                st.session_state["current_view"] = "rover"
                st.rerun()

    with col3:
        with st.container(border=True):
            st.markdown("#### 📡 3. Surface Subsidence")
            st.write("Distributed geotechnical surface mesh tracking ground tilt, micro-seismic tremors, and settlement curve.")
            st.markdown("---")
            st.markdown("- **Coverage:** Panel 4 (4 Nodes)")
            st.markdown("- **Parameters:** Tilt, Vibration, Crack")
            st.markdown("- **Profile:** Peck's Formula")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Open Subsidence Module", key="btn_open_sub", use_container_width=True):
                st.session_state["current_view"] = "subsidence"
                st.rerun()

# -------------------------------------------------------------
# SCREEN 2: OPTION 1 - Fire & Smoke Camera
# -------------------------------------------------------------
def render_module1_camera():
    render_header()
    
    st.subheader("Option 1: Surface Fire & Smoke Camera")
    st.caption("Webcam feed with optical flame dynamics and smoke detection")
    
    # Clean, direct toggle switches right on screen
    with st.container(border=True):
        st.write("**Feature Toggles**")
        t1, t2, t3, t4 = st.columns(4)
        toggle_smoke = t1.toggle("Smoke Detection", value=config.enable_smoke, key="c_t_smoke")
        toggle_audio = t2.toggle("Audio Siren", value=config.enable_audio_alarm, key="c_t_audio")
        toggle_voice = t3.toggle("Voice Alerts", value=config.enable_voice_alert, key="c_t_voice")
        toggle_snap = t4.toggle("Auto-Save Photos", value=config.enable_snapshots, key="c_t_snap")
        
        config.enable_smoke = toggle_smoke
        config.enable_audio_alarm = toggle_audio
        config.enable_voice_alert = toggle_voice
        config.enable_snapshots = toggle_snap
        alert_manager.set_audio(toggle_audio)
        alert_manager.set_voice(toggle_voice)
        alert_manager.set_snapshots(toggle_snap)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab_stream, tab_evidence = st.tabs(["Live Camera Feed", "Saved Incident Photos"])
    
    with tab_evidence:
        st.write("Recent snapshots saved during fire or smoke detection:")
        snapshot_files = sorted(list(INCIDENTS_DIR.glob("*.jpg")), key=lambda p: p.stat().st_mtime, reverse=True)
        if not snapshot_files:
            st.info("No incident photos recorded yet.")
        else:
            cols = st.columns(3)
            for i, snap_path in enumerate(snapshot_files[:6]):
                col = cols[i % 3]
                try:
                    img = Image.open(snap_path)
                    col.image(img, caption=snap_path.name, use_container_width=True)
                except Exception:
                    pass

    with tab_stream:
        col_video, col_controls = st.columns([2.5, 1])
        
        with col_controls:
            with st.container(border=True):
                st.write("**Camera Controls**")
                start_btn = st.button("Start Camera", type="primary", use_container_width=True)
                stop_btn = st.button("Stop Camera", use_container_width=True)
                
                st.markdown("---")
                st.write("**Detection Settings**")
                engine_mode = st.selectbox("Engine", ["HYBRID", "CV_ONLY", "DL_ONLY"], index=0)
                conf_val = st.slider("Confidence", 0.20, 0.80, float(config.confidence_threshold), 0.05)
                config.engine_mode = engine_mode
                config.confidence_threshold = conf_val
                
                st.markdown("---")
                st.write("**Test Alarms**")
                c_test1, c_test2 = st.columns(2)
                if c_test1.button("Test Siren"):
                    try:
                        import winsound
                        winsound.Beep(1800, 200)
                    except Exception:
                        pass
                if c_test2.button("Test Voice"):
                    try:
                        import pyttsx3
                        tts = pyttsx3.init()
                        tts.say("Warning! Fire detection operational.")
                        tts.runAndWait()
                    except Exception:
                        pass

        with col_video:
            video_placeholder = st.empty()
            
            # Simple clean metric row
            m1, m2, m3, m4 = st.columns(4)
            metric_status = m1.metric("Status", "Standby")
            metric_fps = m2.metric("FPS", "0.0")
            metric_threat = m3.metric("Threat", "None")
            metric_count = m4.metric("Logged Incidents", len(list(INCIDENTS_DIR.glob("*.jpg"))))

        if start_btn:
            cap = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(config.camera_index)
                
            if not cap.isOpened():
                st.error("Could not open laptop webcam. Please check camera connection.")
            else:
                detector = FireDetector(mode=config.engine_mode, enable_smoke=config.enable_smoke)
                detector.set_persistence_frames(config.persistence_frames)
                
                active_toggles = {
                    "smoke": config.enable_smoke,
                    "audio": config.enable_audio_alarm,
                    "voice": config.enable_voice_alert,
                    "snapshots": config.enable_snapshots
                }
                
                while not stop_btn:
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.03)
                        continue
                        
                    sys_state, detections, fps = detector.process_frame(frame)
                    
                    if sys_state == "ALARM":
                        alert_manager.trigger_alert(frame, detections, sys_state)
                        
                    hud_frame = hud_visualizer.draw_hud(
                        frame=frame,
                        system_state=sys_state,
                        detections=detections,
                        fps=fps,
                        toggles=active_toggles,
                        engine_mode=config.engine_mode
                    )
                    
                    rgb_frame = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                    
                    # Update metrics
                    threat_label = "None"
                    if detections:
                        threat_label = ", ".join(set(d["label"].upper() for d in detections))
                        
                    m1.metric("Status", sys_state)
                    m2.metric("FPS", f"{fps:.1f}")
                    m3.metric("Threat", threat_label)
                    
                    time.sleep(0.01)
                    
                cap.release()

# -------------------------------------------------------------
# SCREEN 3: OPTION 2 - Underground Rover
# -------------------------------------------------------------
def render_module2_rover():
    render_header()
    
    st.subheader("Option 2: Underground Rover Telemetry")
    st.caption("Subsurface environmental data from mobile rover (DHT11 & MQ gas sensors)")
    
    # Clean hardware connection row
    with st.container(border=True):
        hw_col1, hw_col2, hw_col3 = st.columns([2, 1.5, 1])
        with hw_col1:
            esp_ip = st.text_input("ESP8266 Node IP", value=st.session_state["esp_ip"])
            st.session_state["esp_ip"] = esp_ip
        with hw_col2:
            st.write("**Mode**")
            mode = st.radio("Mode", ["Auto-Detect Hardware", "Simulation"], horizontal=True, label_visibility="collapsed")
        with hw_col3:
            st.write("")
            st.write("")
            st.button("Ping Node", use_container_width=True)
            
    # Simple test scenario buttons for judges
    st.write("**Presentation Scenarios:**")
    scen_col1, scen_col2, scen_col3 = st.columns(3)
    if scen_col1.button("Normal Mine Conditions", use_container_width=True):
        rover_telemetry.set_scenario("NORMAL")
    if scen_col2.button("Simulate Gas Leak (Methane & CO)", use_container_width=True):
        rover_telemetry.set_scenario("GAS_LEAK")
    if scen_col3.button("Simulate Low Oxygen (Hypoxia)", use_container_width=True):
        rover_telemetry.set_scenario("LOW_O2")
        
    readings = rover_telemetry.get_readings(st.session_state["esp_ip"] if "Hardware" in mode else None)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Status Banner (Native Streamlit Alerts)
    if readings["gas_status"] == "CRITICAL":
        st.error(f"Critical Gas Alert: {readings['alert_message']} (Threshold Exceeded)")
    elif readings["gas_status"] == "WARNING":
        st.warning(f"Advisory Warning: {readings['alert_message']}")
    else:
        st.success(f"Atmosphere Safe: {readings['alert_message']} (ESP8266 Status: {readings['node_status']})")
        
    # Sensor Metrics
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("DHT11 Temp", f"{readings['temperature']} °C")
    k2.metric("DHT11 Humidity", f"{readings['humidity']} %")
    k3.metric("Methane (MQ-4)", f"{readings['ch4_pct']} %", delta=None if readings['ch4_pct'] < 1.0 else "High")
    k4.metric("CO (MQ-7)", f"{readings['co_ppm']} ppm", delta=None if readings['co_ppm'] < 25.0 else "High")
    k5.metric("CO2 (MQ-135)", f"{readings['co2_ppm']} ppm")
    k6.metric("Oxygen (O2)", f"{readings['o2_pct']} %", delta=None if readings['o2_pct'] >= 19.5 else "Low")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # FLIR Camera view and live sensor graphs
    col_left, col_right = st.columns([1.2, 1.8])
    
    with col_left:
        with st.container(border=True):
            st.write("**Rover FLIR Camera View (Seam 3)**")
            flir_frame = rover_telemetry.generate_flir_frame(readings)
            rgb_flir = cv2.cvtColor(flir_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_flir, channels="RGB", use_container_width=True)
            st.caption(f"Depth: {readings['depth_m']}m | Battery: {readings['battery_pct']}% | Signal: {readings['rssi_dbm']} dBm")

    with col_right:
        with st.container(border=True):
            st.write("**Sensor Readings Over Time**")
            hist = rover_telemetry.history
            if len(hist["temp"]) > 1:
                df_env = pd.DataFrame({
                    "Temperature (°C)": hist["temp"],
                    "Humidity (%)": hist["humidity"]
                })
                st.line_chart(df_env, height=150)
                
                df_gas = pd.DataFrame({
                    "Methane (% Vol)": hist["ch4"],
                    "CO (ppm / 10)": [x/10.0 for x in hist["co"]]
                })
                st.line_chart(df_gas, height=150)

# -------------------------------------------------------------
# SCREEN 4: OPTION 3 - Surface Subsidence Mesh
# -------------------------------------------------------------
def render_module3_subsidence():
    render_header()
    
    st.subheader("Option 3: Surface Subsidence Mesh")
    st.caption("Surface sensor nodes monitoring tilt, vibrations, and ground deformation over active coal panel")
    
    # Presentation Scenarios
    st.write("**Presentation Scenarios:**")
    sub_c1, sub_c2, sub_c3 = st.columns(3)
    if sub_c1.button("Stable Ground (Normal)", use_container_width=True):
        subsidence_mesh.set_scenario("NORMAL")
    if sub_c2.button("Simulate Ground Strain (Advisory)", use_container_width=True):
        subsidence_mesh.set_scenario("STRATA_STRAIN")
    if sub_c3.button("Simulate Subsidence Collapse (Alarm)", use_container_width=True):
        subsidence_mesh.set_scenario("CRITICAL_SUBSIDENCE")
        
    mesh_data = subsidence_mesh.get_mesh_data()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Status Banner
    if "CRITICAL" in mesh_data["risk_level"]:
        st.error(f"🚨 {mesh_data['alert_header']} — {mesh_data['alert_sub']}")
    elif "ADVISORY" in mesh_data["risk_level"]:
        st.warning(f"▲ {mesh_data['alert_header']} — {mesh_data['alert_sub']}")
    else:
        st.success(f"● {mesh_data['alert_header']} — {mesh_data['alert_sub']}")
        
    # 4 Surface Nodes
    st.write("**Surface Sensor Nodes across Panel 4:**")
    col_n1, col_n2, col_n3, col_n4 = st.columns(4)
    nodes = mesh_data["nodes"]
    
    for col, (nid, ndata) in zip([col_n1, col_n2, col_n3, col_n4], nodes.items()):
        with col:
            with st.container(border=True):
                st.write(f"**{nid}** ({ndata['name']})")
                st.caption(f"Position: {ndata['distance_m']}m from center")
                st.markdown("---")
                st.write(f"**Settlement:** {ndata['settlement_mm']} mm")
                st.write(f"**Tilt Angle:** {ndata['tilt_mm_m']} mm/m")
                st.write(f"**Vibration:** {ndata['vibration_g']} g")
                st.write(f"**Crack Width:** {ndata['crack_aperture_mm']} mm")
                st.caption(f"Battery: {ndata['battery_pct']}% | RF: {ndata['rf_rssi_dbm']} dBm")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Peck's Subsidence Trough Curve
    with st.container(border=True):
        st.write("**Ground Subsidence Profile (Peck's Formula)**")
        st.caption("Transverse distance (-150m to +150m) vs vertical settlement (mm)")
        
        prof = mesh_data["profile_curve"]
        df_prof = pd.DataFrame({
            "Distance (m)": prof["x"],
            "Settlement (mm)": prof["settlement"],
            "Tilt (mm/m)": prof["tilt"]
        }).set_index("Distance (m)")
        
        st.line_chart(df_prof, height=220)

# -------------------------------------------------------------
# Main Navigation Router
# -------------------------------------------------------------
def main():
    if not st.session_state["authenticated"]:
        render_login()
    else:
        view = st.session_state["current_view"]
        if view == "hub":
            render_hub()
        elif view == "camera":
            render_module1_camera()
        elif view == "rover":
            render_module2_rover()
        elif view == "subsidence":
            render_module3_subsidence()
        else:
            render_hub()

if __name__ == "__main__":
    main()
