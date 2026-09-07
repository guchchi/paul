"""
SIH Smart Mine Safety Monitoring Station - Web Dashboard
Interactive Streamlit Dashboard with live webcam feed, feature toggles,
confidence sliders, and incident evidence gallery.
"""
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.config import config, INCIDENTS_DIR
from src.detection.fire_detector import FireDetector
from src.alerts.alert_manager import alert_manager
from src.utils.visualizer import hud_visualizer

# Streamlit Page Config
st.set_page_config(
    page_title="SIH Smart Mine Hazard & Fire Surveillance",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Hackathon Aesthetic
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FF4B4B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }
    .status-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #333;
        background-color: #111;
        text-align: center;
    }
    .status-normal { color: #00E676; font-weight: bold; font-size: 1.3rem; }
    .status-verifying { color: #FFD600; font-weight: bold; font-size: 1.3rem; }
    .status-alarm { color: #FF1744; font-weight: bold; font-size: 1.3rem; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Sidebar: Control Panel & Toggles
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/color/96/fire-element.png", width=64)
st.sidebar.title("Surveillance Controls")
st.sidebar.markdown("---")

# Feature Toggles (Requested by User)
st.sidebar.subheader("Feature Toggles")
enable_smoke = st.sidebar.checkbox(
    "Enable Smoke Detection",
    value=config.enable_smoke,
    help="Detect diffuse smoke plumes in addition to direct flames"
)

enable_audio = st.sidebar.checkbox(
    "Enable Siren Alarm",
    value=config.enable_audio_alarm,
    help="Play high-frequency audio warble when hazard is confirmed"
)

enable_voice = st.sidebar.checkbox(
    "Enable Voice Announcements",
    value=config.enable_voice_alert,
    help="Synthesize TTS voice warnings via pyttsx3"
)

enable_snapshots = st.sidebar.checkbox(
    "Enable Auto-Snapshots",
    value=config.enable_snapshots,
    help="Automatically record timestamped evidence to incidents/"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Detection Parameters")

engine_mode = st.sidebar.selectbox(
    "Detection Engine",
    options=["HYBRID", "CV_ONLY", "DL_ONLY"],
    index=["HYBRID", "CV_ONLY", "DL_ONLY"].index(config.engine_mode)
)

confidence_thresh = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.20,
    max_value=0.90,
    value=float(config.confidence_threshold),
    step=0.05
)

persistence_frames = st.sidebar.slider(
    "Temporal Persistence (Frames)",
    min_value=1,
    max_value=15,
    value=int(config.persistence_frames),
    step=1,
    help="Frames fire must persist continuously before triggering alarm"
)

camera_idx = st.sidebar.number_input("Webcam Index", min_value=0, max_value=5, value=config.camera_index, step=1)

# Apply settings to global configuration
config.enable_smoke = enable_smoke
config.enable_audio_alarm = enable_audio
config.enable_voice_alert = enable_voice
config.enable_snapshots = enable_snapshots
config.confidence_threshold = confidence_thresh
config.persistence_frames = persistence_frames
config.engine_mode = engine_mode
config.camera_index = int(camera_idx)

# Update managers
alert_manager.set_audio(enable_audio)
alert_manager.set_voice(enable_voice)
alert_manager.set_snapshots(enable_snapshots)

# -------------------------------------------------------------
# Main Dashboard
# -------------------------------------------------------------
st.markdown('<div class="main-header">🔥 SIH Smart Mine Hazard & Fire Surveillance</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Video-Based Combustion & Smoke Detection Module for Mine Surface Infrastructure</div>', unsafe_allow_html=True)

# Tabs
tab_live, tab_gallery, tab_architecture = st.tabs([
    "📹 Live Camera Feed",
    "📸 Incident Evidence Gallery",
    "🏗️ SIH System Context"
])

# -------------------------------------------------------------
# Tab 1: Live Video Feed
# -------------------------------------------------------------
with tab_live:
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    kpi_status = col_kpi1.empty()
    kpi_fps = col_kpi2.empty()
    kpi_threat = col_kpi3.empty()
    kpi_incidents = col_kpi4.empty()

    col_video, col_side_info = st.columns([3, 1])
    
    with col_side_info:
        st.markdown("### Active Toggles")
        st.write(f"- **Smoke Detection:** {'🟢 ON' if enable_smoke else '🔴 OFF'}")
        st.write(f"- **Audio Siren:** {'🟢 ON' if enable_audio else '🔴 OFF'}")
        st.write(f"- **Voice Alert:** {'🟢 ON' if enable_voice else '🔴 OFF'}")
        st.write(f"- **Incident Logger:** {'🟢 ON' if enable_snapshots else '🔴 OFF'}")
        st.write(f"- **Active Engine:** `{engine_mode}`")
        
        st.markdown("### Safety Actions")
        if st.button("🚨 Manual Test Siren"):
            import winsound
            winsound.Beep(1500, 250)
            st.success("Siren hardware check passed.")
            
        if st.button("🗣️ Manual Test Voice"):
            try:
                import pyttsx3
                tts = pyttsx3.init()
                tts.say("Fire detection system operational.")
                tts.runAndWait()
                st.success("Voice synthesizer operational.")
            except Exception as e:
                st.error(f"Voice test error: {e}")

    with col_video:
        video_placeholder = st.empty()
        stream_btn_col1, stream_btn_col2 = st.columns(2)
        start_stream = stream_btn_col1.button("▶️ Start Live Webcam Feed", type="primary")
        stop_stream = stream_btn_col2.button("⏹️ Stop Stream")

    if start_stream:
        # Initialize camera
        cap = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(config.camera_index)
            
        if not cap.isOpened():
            st.error(f"Cannot access webcam at index {config.camera_index}! Check connection.")
        else:
            detector = FireDetector(mode=engine_mode, enable_smoke=enable_smoke)
            detector.set_persistence_frames(persistence_frames)
            
            st.toast("Webcam initialized successfully!", icon="📹")
            
            active_toggles = {
                "smoke": enable_smoke,
                "audio": enable_audio,
                "voice": enable_voice,
                "snapshots": enable_snapshots
            }
            
            while not stop_stream:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Frame read timeout. Retrying...")
                    time.sleep(0.05)
                    continue
                    
                # Process frame
                sys_state, detections, fps = detector.process_frame(frame)
                
                # Alerts
                if sys_state == "ALARM":
                    alert_manager.trigger_alert(frame, detections, sys_state)
                    
                # Render HUD
                hud_frame = hud_visualizer.draw_hud(
                    frame=frame,
                    system_state=sys_state,
                    detections=detections,
                    fps=fps,
                    toggles=active_toggles,
                    engine_mode=engine_mode
                )
                
                # Convert BGR to RGB for Streamlit
                rgb_frame = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
                # Update KPIs
                status_html = {
                    "NORMAL": '<div class="status-card"><span class="status-normal">● SECURE</span><br><small>No Fire Detected</small></div>',
                    "VERIFYING": '<div class="status-card"><span class="status-verifying">▲ VERIFYING</span><br><small>Combustion Dynamics</small></div>',
                    "ALARM": '<div class="status-card"><span class="status-alarm">⚠ CRITICAL ALARM</span><br><small>Fire / Smoke Confirmed</small></div>'
                }.get(sys_state, sys_state)
                
                kpi_status.markdown(status_html, unsafe_allow_html=True)
                kpi_fps.metric("Inference Rate", f"{fps:.1f} FPS")
                
                threat_label = "None"
                if detections:
                    threat_label = ", ".join(set(d["label"].upper() for d in detections))
                kpi_threat.metric("Active Threat", threat_label)
                
                snapshot_count = len(list(INCIDENTS_DIR.glob("*.jpg")))
                kpi_incidents.metric("Recorded Incidents", f"{snapshot_count} Events")
                
                time.sleep(0.01)
                
            cap.release()

# -------------------------------------------------------------
# Tab 2: Incident Evidence Gallery
# -------------------------------------------------------------
with tab_gallery:
    st.subheader("Automated Incident Snapshots")
    st.markdown("All high-confidence hazard events captured by the webcam are automatically preserved here.")
    
    snapshot_files = sorted(list(INCIDENTS_DIR.glob("*.jpg")), key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not snapshot_files:
        st.info("No fire incidents recorded yet. System has operated with zero alarms.")
    else:
        st.write(f"Total Logged Incidents: **{len(snapshot_files)}**")
        cols = st.columns(3)
        for i, snap_path in enumerate(snapshot_files[:12]):
            col = cols[i % 3]
            try:
                img = Image.open(snap_path)
                col.image(img, caption=snap_path.name, use_container_width=True)
                with open(snap_path, "rb") as file_bytes:
                    col.download_button(
                        label="⬇️ Download Snapshot",
                        data=file_bytes,
                        file_name=snap_path.name,
                        mime="image/jpeg",
                        key=f"dl_{snap_path.name}"
                    )
            except Exception as e:
                col.error(f"Error loading {snap_path.name}")

# -------------------------------------------------------------
# Tab 3: SIH System Architecture & Context
# -------------------------------------------------------------
with tab_architecture:
    st.subheader("Integration with SIH Smart Mine Subsidence & Safety Framework")
    st.markdown("""
    ### System Context
    Underground coal extraction triggers subsidence, surface fissures, and spontaneous subsurface combustion (coal seam fires).
    
    This module serves as the **Autonomous Vision Surveillance Node** operating alongside the surface wireless mesh network:
    
    1. **Dual Combustion Verification**:
       - **Classical CV Filter**: Real-time HSV and YCrCb color space segmentation coupled with flame flicker frequency analysis.
       - **Deep Learning Stream**: YOLOv8/11 object classification identifying flames and smoke plumes.
       - **Temporal Persistence Tracking**: Eliminates transient light reflections and yellow/orange ambient objects by requiring continuous detection across consecutive frames.
       
    2. **Multi-Modal Alert Dispatch**:
       - Local audio siren & synthesized speech warnings for workers on-site.
       - Automated timestamped forensic snapshot logging.
       - IoT Serial integration ready for Arduino / ESP32 mesh relay triggers.
    """)
