"""
SIH 2026 - Mine Subsidence & Safety Early Warning System
Modernized SCADA Command Station using streamlit-shadcn-ui components.
Three Core Modules:
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

# Shadcn UI Library (0.1.19 standard Streamlit component architecture)
import streamlit_shadcn_ui as ui

# System modules
from src.config import config, INCIDENTS_DIR
from src.detection.fire_detector import FireDetector
from src.alerts.alert_manager import alert_manager
from src.utils.visualizer import hud_visualizer
from src.telemetry.rover_telemetry import rover_telemetry
from src.telemetry.subsidence_telemetry import subsidence_mesh
from src.telemetry.ground_scanner import ground_scanner

# -------------------------------------------------------------
# Streamlit Page Setup
# -------------------------------------------------------------
st.set_page_config(
    page_title="Mine Safety & Subsidence System",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── SCADA Dashboard Theme (Light) ─────────────────────────────
st.markdown("""
<style>
    /* ── Import Neo-Brutalist Font (Plus Jakarta Sans) ── */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800;900&family=Space+Grotesk:wght@600;700&display=swap');

    /* ── Global App Shell (Warm Canvas) ── */
    body, .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
        background-color: #faf8f0 !important;
        color: #000000 !important;
    }

    /* Hide default Streamlit chrome */
    header[data-testid="stHeader"] {
        background: rgba(250,248,240,0.95) !important;
        border-bottom: 2.5px solid #000000 !important;
    }
    #MainMenu, footer { visibility: hidden; }

    /* ── Typography ── */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #000000 !important;
        font-weight: 900 !important;
        letter-spacing: -0.03em !important;
    }
    p, span, label, .stMarkdown p, .stMarkdown span {
        color: #111827;
        font-weight: 600;
    }

    /* ── Neo-Brutalist Card Containers ── */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 2.5px solid #000000 !important;
        border-radius: 16px !important;
        padding: 1.15rem !important;
        box-shadow: 5px 5px 0px #000000 !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translate(-2px, -2px) !important;
        box-shadow: 7px 7px 0px #000000 !important;
    }

    /* ── Spec Grid ── */
    .spec-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.65rem;
        margin: 0.85rem 0 1.05rem 0;
    }
    .spec-cell {
        background: #fafafa;
        border: 2px solid #000000;
        border-radius: 9px;
        box-shadow: 2px 2px 0px #000000;
        padding: 0.48rem 0.75rem;
    }
    .spec-label {
        font-size: 0.62rem;
        font-weight: 900;
        color: #000000;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: block;
        margin-bottom: 2px;
    }
    .spec-val {
        font-size: 0.80rem;
        font-weight: 800;
        color: #111827;
        display: block;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* ── Text Inputs ── */
    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2.5px solid #000000 !important;
        border-radius: 10px !important;
        box-shadow: 3px 3px 0px #000000 !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        transition: all 0.12s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #000000 !important;
        box-shadow: 5px 5px 0px #000000 !important;
        outline: none !important;
    }
    .stTextInput label {
        color: #000000 !important;
        font-size: 0.78rem !important;
        font-weight: 900 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    /* ── Select / Dropdowns ── */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 2.5px solid #000000 !important;
        border-radius: 10px !important;
        box-shadow: 3px 3px 0px #000000 !important;
        color: #000000 !important;
        font-weight: 700 !important;
    }
    .stSelectbox label {
        color: #000000 !important;
        font-size: 0.78rem !important;
        font-weight: 900 !important;
        text-transform: uppercase;
    }

    /* ── Slider ── */
    .stSlider label {
        color: #000000 !important;
        font-size: 0.78rem !important;
        font-weight: 900 !important;
    }
    .stSlider > div > div > div[role="slider"] {
        background-color: #000000 !important;
        border: 2px solid #000000 !important;
    }

    /* ── Radio Buttons ── */
    .stRadio label {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    /* ── Streamlit native buttons (Neo-Brutalist) ── */
    button[kind="primary"], .stButton > button[kind="primary"],
    button[kind="secondary"], .stButton > button[kind="secondary"],
    .stFormSubmitButton > button {
        border: 2.5px solid #000000 !important;
        border-radius: 10px !important;
        box-shadow: 4px 4px 0px #000000 !important;
        font-weight: 900 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.02em !important;
        padding: 0.55rem 1.1rem !important;
        transition: transform 0.1s ease, box-shadow 0.1s ease !important;
    }
    button[kind="primary"], .stButton > button[kind="primary"] {
        background: #facc15 !important;
        color: #000000 !important;
    }
    button[kind="primary"] p, .stButton > button[kind="primary"] p,
    button[kind="primary"] span, .stButton > button[kind="primary"] span {
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 0.85rem !important;
    }
    button[kind="secondary"], .stButton > button[kind="secondary"] {
        background: #ffffff !important;
        color: #000000 !important;
    }
    button[kind="secondary"] p, .stButton > button[kind="secondary"] p,
    button[kind="secondary"] span, .stButton > button[kind="secondary"] span {
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 0.82rem !important;
    }
    button:hover, .stButton > button:hover, .stFormSubmitButton > button:hover {
        transform: translate(2px, 2px) !important;
        box-shadow: 2px 2px 0px #000000 !important;
    }

    /* ── Hub Grid Module Buttons (Scoped to Markers) ── */
    div[data-testid="stColumn"]:has(.hub-marker-m1) .stButton > button {
        background: #4ade80 !important;
    }
    div[data-testid="stColumn"]:has(.hub-marker-m1) .stButton > button:hover {
        background: #22c55e !important;
    }

    div[data-testid="stColumn"]:has(.hub-marker-m2) .stButton > button {
        background: #fde047 !important;
    }
    div[data-testid="stColumn"]:has(.hub-marker-m2) .stButton > button:hover {
        background: #eab308 !important;
    }

    div[data-testid="stColumn"]:has(.hub-marker-m3) .stButton > button {
        background: #c084fc !important;
    }
    div[data-testid="stColumn"]:has(.hub-marker-m3) .stButton > button:hover {
        background: #a855f7 !important;
    }

    div[data-testid="stColumn"]:has(.hub-marker-m4) .stButton > button {
        background: #fb923c !important;
    }
    div[data-testid="stColumn"]:has(.hub-marker-m4) .stButton > button:hover {
        background: #f97316 !important;
    }

    .stFormSubmitButton > button {
        background: #facc15 !important;
        color: #000000 !important;
    }
    .stFormSubmitButton > button p {
        color: #000000 !important;
        font-weight: 900 !important;
    }

    /* ── Streamlit Divider ── */
    hr {
        border-color: #000000 !important;
        border-width: 1.5px !important;
        opacity: 1;
    }

    /* ── Alerts ── */
    .stAlert > div {
        border: 2.5px solid #000000 !important;
        border-radius: 12px !important;
        box-shadow: 4px 4px 0px #000000 !important;
        font-weight: 800 !important;
        color: #000000 !important;
    }

    /* ── Metric cards (native st.metric) ── */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 2.5px solid #000000 !important;
        border-radius: 12px !important;
        box-shadow: 3.5px 3.5px 0px #000000 !important;
        padding: 0.75rem 0.95rem !important;
    }
    div[data-testid="stMetric"] label {
        color: #000000 !important;
        font-size: 0.72rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        font-weight: 900 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: 900 !important;
        font-size: 1.35rem !important;
    }

    /* ── Form container styling ── */
    div[data-testid="stForm"] {
        background-color: #ffffff !important;
        border: 3px solid #000000 !important;
        border-radius: 18px !important;
        box-shadow: 6px 6px 0px #000000 !important;
        padding: 1.6rem !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab"] {
        font-weight: 800 !important;
        color: #000000 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #000000 !important;
        border-bottom-color: #000000 !important;
        border-bottom-width: 3px !important;
    }

    /* ── Checkbox ── */
    .stCheckbox label {
        color: #000000 !important;
        font-weight: 800 !important;
    }

    /* ── Column gap ── */
    div[data-testid="stHorizontalBlock"] {
        gap: 0.85rem;
    }

    /* ── Images ── */
    .stImage img {
        border: 2.5px solid #000000;
        border-radius: 12px;
        box-shadow: 4px 4px 0px #000000;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "current_view" not in st.session_state:
    st.session_state["current_view"] = "hub"  # 'hub', 'camera', 'rover', 'subsidence'
if "esp_ip" not in st.session_state:
    st.session_state["esp_ip"] = "192.168.4.1"
if "operator_name" not in st.session_state:
    st.session_state["operator_name"] = "Safety Officer"
if "camera_active" not in st.session_state:
    st.session_state["camera_active"] = False
if "cap" not in st.session_state:
    st.session_state["cap"] = None
if "detector" not in st.session_state:
    st.session_state["detector"] = None

# -------------------------------------------------------------
# Top Navigation Header (Neo-Brutalist Single-Row Header)
# -------------------------------------------------------------
def render_header():
    now_str = datetime.now().strftime("%H:%M:%S")
    date_str = datetime.now().strftime("%d %b %Y")
    is_hub = (st.session_state.get("current_view", "hub") == "hub")

    # ── Single-Row Neo-Brutalist Header ──
    if is_hub:
        col_brand, col_clock, col_actions = st.columns([5.2, 3.2, 1.2], vertical_alignment="center")
    else:
        col_brand, col_clock, col_actions = st.columns([4.8, 2.8, 2.0], vertical_alignment="center")

    with col_brand:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:0.75rem;">
            <div style="width:40px; height:40px; border-radius:10px; background:#ffffff; border:2.5px solid #000000; box-shadow:3px 3px 0px #000000; display:flex; align-items:center; justify-content:center; font-size:1.25rem; flex-shrink:0;" title="System Settings">
                ⚙️
            </div>
            <div style="width:40px; height:40px; border-radius:10px; background:#facc15; border:2.5px solid #000000; box-shadow:3px 3px 0px #000000; display:flex; align-items:center; justify-content:center; font-size:1.25rem; flex-shrink:0;">
                ⛏️
            </div>
            <div>
                <div style="display:flex; align-items:center; gap:0.55rem;">
                    <span style="font-size:1.18rem; font-weight:900; color:#000000; letter-spacing:-0.03em;">MineShield EWS</span>
                    <span style="display:inline-flex; align-items:center; gap:0.35rem; padding:0.18rem 0.6rem; background:#4ade80; border:2px solid #000000; border-radius:999px; font-size:0.66rem; font-weight:900; color:#000000; text-transform:uppercase; letter-spacing:0.04em; box-shadow:2px 2px 0px #000000;">
                        <span style="width:6px; height:6px; border-radius:50%; background:#000000;"></span>
                        LIVE ONLINE
                    </span>
                </div>
                <div style="font-size:0.74rem; color:#4b5563; font-weight:700; margin-top:2px;">
                    DGMS Coal Mines Regulations Early-Warning Gateway &middot; SIH 2026
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_clock:
        st.markdown(f"""
        <div style="display:flex; align-items:center; justify-content:flex-end; gap:0.6rem;">
            <div style="display:flex; flex-direction:column; align-items:flex-end; justify-content:center;">
                <div style="display:flex; align-items:center; gap:0.45rem; font-family:'Space Grotesk', monospace; font-size:0.84rem; font-weight:800; color:#000000; background:#ffffff; padding:0.3rem 0.75rem; border-radius:10px; border:2.5px solid #000000; box-shadow:3px 3px 0px #000000;">
                    <span>⏱</span> {now_str} <span style="color:#9ca3af;">|</span> <span>{date_str}</span>
                </div>
                <div style="font-size:0.68rem; color:#000000; font-weight:800; margin-top:3px; text-transform:uppercase; letter-spacing:0.03em;">
                    STATION: SINKHOLE-Z4 &middot; PANEL 4
                </div>
            </div>
            <div style="display:flex; align-items:center; gap:0.4rem;">
                <div style="width:38px; height:38px; border-radius:10px; background:#ffffff; border:2.5px solid #000000; box-shadow:2.5px 2.5px 0px #000000; display:flex; align-items:center; justify-content:center; font-size:1.15rem; position:relative;" title="Logs & Reports">
                    📁<span style="position:absolute; top:-3px; right:-3px; width:9px; height:9px; border-radius:50%; background:#ff5388; border:1.5px solid #000;"></span>
                </div>
                <div style="width:38px; height:38px; border-radius:10px; background:#ffffff; border:2.5px solid #000000; box-shadow:2.5px 2.5px 0px #000000; display:flex; align-items:center; justify-content:center; font-size:1.15rem;" title="Search Nodes">
                    🔍
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_actions:
        if is_hub:
            if st.button("Log Out", key="hdr_btn_logout", use_container_width=True):
                if st.session_state.get("cap") is not None:
                    try:
                        st.session_state["cap"].release()
                    except Exception:
                        pass
                    st.session_state["cap"] = None
                st.session_state["camera_active"] = False
                st.session_state["authenticated"] = False
                st.session_state["current_view"] = "hub"
                st.rerun()
        else:
            b_col1, b_col2 = st.columns([1.2, 1], vertical_alignment="center")
            with b_col1:
                if st.button("← Hub", key="hdr_btn_home", use_container_width=True):
                    if st.session_state.get("cap") is not None:
                        try:
                            st.session_state["cap"].release()
                        except Exception:
                            pass
                        st.session_state["cap"] = None
                    st.session_state["camera_active"] = False
                    st.session_state["current_view"] = "hub"
                    st.rerun()
            with b_col2:
                if st.button("Exit", key="hdr_btn_logout", use_container_width=True):
                    if st.session_state.get("cap") is not None:
                        try:
                            st.session_state["cap"].release()
                        except Exception:
                            pass
                        st.session_state["cap"] = None
                    st.session_state["camera_active"] = False
                    st.session_state["authenticated"] = False
                    st.session_state["current_view"] = "hub"
                    st.rerun()

    # Hairline divider
    st.markdown("""
    <div style="height:2.5px; background:#000000; margin:0.5rem 0 1.2rem 0; border-radius:2px;"></div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# Neo-Brutalist Reusable Components
# -------------------------------------------------------------
def render_neo_metric(title: str, value: str, desc: str = "", icon: str = "", bg: str = "#ffffff", badge: str = "", badge_bg: str = "#fde047") -> str:
    badge_html = f"<span style='background:{badge_bg}; border:1.5px solid #000; border-radius:6px; font-size:0.65rem; font-weight:900; color:#000; padding:0.12rem 0.45rem; box-shadow:1.5px 1.5px 0px #000;'>{badge}</span>" if badge else ""
    icon_html = f"<span style='font-size:1.15rem;'>{icon}</span>" if icon else ""
    desc_html = f"<div style='font-size:0.72rem; font-weight:700; color:#4b5563; margin-top:0.3rem;'>{desc}</div>" if desc else ""
    
    return f"""
    <div style="background:{bg}; border:2.5px solid #000000; border-radius:14px; box-shadow:3.5px 3.5px 0px #000000; padding:0.85rem 1.05rem; margin-bottom:0.75rem; min-height:102px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem;">
            <span style="font-size:0.68rem; font-weight:900; color:#000000; text-transform:uppercase; letter-spacing:0.04em;">{title}</span>
            <div style="display:flex; align-items:center; gap:0.4rem;">{badge_html}{icon_html}</div>
        </div>
        <div style="font-size:1.75rem; font-weight:900; color:#000000; letter-spacing:-0.03em; line-height:1.1;">{value}</div>
        {desc_html}
    </div>
    """

# -------------------------------------------------------------
# SCREEN 0: Neo-Brutalist Login Form
# -------------------------------------------------------------
def render_login():
    st.markdown("<br><br>", unsafe_allow_html=True)
    _, col_center, _ = st.columns([1.2, 1, 1.2])
    
    with col_center:
        # Branded header block
        st.markdown("""
        <div style="text-align:center; margin-bottom:1.5rem;">
            <div style="width:58px; height:58px; border-radius:16px; background:#facc15; border:3px solid #000000; box-shadow:4px 4px 0px #000000; display:flex; align-items:center; justify-content:center; font-size:1.8rem; margin:0 auto 0.8rem auto;">
                ⛏️
            </div>
            <div style="font-size:1.6rem; font-weight:900; color:#000000; letter-spacing:-0.03em;">Mine Safety Portal</div>
            <div style="font-size:0.8rem; color:#4b5563; margin-top:0.2rem; font-weight:700;">DGMS Early Warning Gateway &middot; SIH 2026</div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            username = st.text_input("User ID", value="operator_01")
            password = st.text_input("Password", value="sih2026", type="password")
            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Sign In →", use_container_width=True)
            
            if submitted:
                if username and password:
                    st.session_state["authenticated"] = True
                    st.session_state["operator_name"] = username
                    st.rerun()
                else:
                    st.error("Please enter both username and password")
        
        st.markdown("""
        <div style="text-align:center; margin-top:1rem;">
            <span style="font-size:0.76rem; font-weight:800; color:#4b5563;">Zone-04 Secure &nbsp;·&nbsp; Panel 4 &nbsp;·&nbsp; SIH 2026</span>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# SCREEN 1: 3-Module Selection (Neo-Brutalist Folder Hub)
# -------------------------------------------------------------
def render_hub():
    render_header()
    
    # ── Neo-Brutalist Hero Headline (Matching "Save now. Find anytime.") ──
    st.markdown("""
    <div style="margin:0.2rem 0 1.1rem 0;">
        <div style="font-size:2.4rem; font-weight:900; color:#000000; line-height:1.05; letter-spacing:-0.035em;">
            Save now.<br>Protect anytime.
        </div>
        <div style="font-size:0.88rem; font-weight:700; color:#4b5563; margin-top:0.35rem;">
            Real-time optical vision &bull; Subsurface rover telemetry &bull; Geotechnical subsidence mesh
        </div>
    </div>
    
    <!-- Neo-Brutalist Quick Bar (Matching search bar + yellow button) -->
    <div style="display:flex; gap:0.75rem; align-items:stretch; margin-bottom:1.4rem;">
        <div style="flex:1; background:#ffffff; border:2.5px solid #000000; border-radius:12px; box-shadow:4px 4px 0px #000000; padding:0.7rem 1.1rem; display:flex; align-items:center; justify-content:space-between;">
            <div style="display:flex; align-items:center; gap:0.65rem;">
                <span style="font-size:1.1rem; font-weight:900; color:#000000;">#</span>
                <span style="font-size:0.88rem; font-weight:800; color:#000000;">Live Sensor Stream Active &bull; Station Sinkhole-Z4 (Panel 4)</span>
                <span style="font-size:0.75rem; color:#9ca3af;">|</span>
                <span style="font-size:0.8rem; font-weight:700; color:#4b5563;">5 Hz Microclimate &amp; 6-DOF Inertial Link</span>
            </div>
            <div style="display:flex; align-items:center; gap:0.4rem;">
                <span style="background:#f4f4f5; border:1.5px solid #000; border-radius:6px; padding:0.18rem 0.5rem; font-size:0.72rem; font-weight:900; box-shadow:1.5px 1.5px 0px #000; color:#000;">📋 COPY</span>
            </div>
        </div>
        <div style="width:52px; min-width:52px; background:#facc15; border:2.5px solid #000000; border-radius:12px; box-shadow:4px 4px 0px #000000; display:flex; align-items:center; justify-content:center; font-size:1.8rem; font-weight:900; color:#000000; cursor:pointer;" title="Add Hardware Stream / Live Sync">
            +
        </div>
    </div>
    
    <div style="font-size:1.3rem; font-weight:900; color:#000000; letter-spacing:-0.02em; margin-bottom:0.75rem;">
        My Subsystems
    </div>
    """, unsafe_allow_html=True)
    
    # ── 2x2 Grid Layout for Subsystems ──
    # ── ROW 1: Module 01 (Camera) & Module 02 (Rover) ──
    row1_col1, row1_col2 = st.columns(2, gap="large")
    
    # ── Module 1: Camera (Green Folder Tab) ──
    with row1_col1:
        st.markdown("""
        <div style="margin-bottom:-4px; margin-left:14px; position:relative; z-index:2; display:inline-block;">
            <div style="background:#22c55e; border:2.5px solid #000000; border-bottom:2.5px solid #22c55e; border-radius:10px 10px 0 0; padding:0.3rem 0.9rem; font-size:0.72rem; font-weight:900; color:#000000; letter-spacing:0.04em; text-transform:uppercase; box-shadow:2px -2px 0px #000000;">
                📁 MODULE 01 &middot; SURFACE AI
            </div>
        </div>
        """, unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("""
            <span class="hub-marker-m1" style="display:none;"></span>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                <div style="width:48px; height:48px; border-radius:12px; background:#4ade80; border:2.5px solid #000000; box-shadow:2.5px 2.5px 0px #000000; display:flex; align-items:center; justify-content:center; font-size:1.45rem;">
                    📹
                </div>
                <div style="text-align:right;">
                    <span style="font-size:0.74rem; font-weight:900; color:#000000; background:#f4f4f5; border:2px solid #000000; border-radius:6px; padding:0.2rem 0.55rem; box-shadow:2px 2px 0px #000000;">
                        30 FPS LIVE
                    </span>
                    <div style="font-size:0.72rem; font-weight:700; color:#4b5563; margin-top:4px;">Surface Pit & Incline</div>
                </div>
            </div>
            <div style="font-size:1.3rem; font-weight:900; color:#000000; margin:0.75rem 0 0.25rem 0; letter-spacing:-0.02em;">
                Fire & Smoke Vision
            </div>
            <div style="font-size:0.84rem; font-weight:600; color:#374151; line-height:1.45; min-height:46px; margin-bottom:0.75rem;">
                Optical surveillance utilizing hybrid OpenCV flame kinetics and YOLOv8 plume dispersion to detect combustion risks early.
            </div>
            <div class="spec-grid">
                <div class="spec-cell">
                    <span class="spec-label">SENSOR INPUT</span>
                    <span class="spec-val">Optical CMOS (Cam 0)</span>
                </div>
                <div class="spec-cell">
                    <span class="spec-label">ENGINE PIPELINE</span>
                    <span class="spec-val">OpenCV + YOLO Hybrid</span>
                </div>
                <div class="spec-cell">
                    <span class="spec-label">TARGET CLASSES</span>
                    <span class="spec-val">Flame Core & Smoke</span>
                </div>
                <div class="spec-cell">
                    <span class="spec-label">DISPATCH ACTION</span>
                    <span class="spec-val">Voice Siren + Snap</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open Camera →", key="hub_open_cam", use_container_width=True, type="primary"):
                st.session_state["current_view"] = "camera"
                st.rerun()

    # ── Module 2: Rover (Yellow Folder Tab) ──
    with row1_col2:
        st.markdown("""
        <div style="margin-bottom:-4px; margin-left:14px; position:relative; z-index:2; display:inline-block;">
            <div style="background:#facc15; border:2.5px solid #000000; border-bottom:2.5px solid #facc15; border-radius:10px 10px 0 0; padding:0.3rem 0.9rem; font-size:0.72rem; font-weight:900; color:#000000; letter-spacing:0.04em; text-transform:uppercase; box-shadow:2px -2px 0px #000000;">
                📁 MODULE 02 &middot; SUBSURFACE IOT
            </div>
        </div>
        """, unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("""
            <span class="hub-marker-m2" style="display:none;"></span>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                <div style="width:48px; height:48px; border-radius:12px; background:#fde047; border:2.5px solid #000000; box-shadow:2.5px 2.5px 0px #000000; display:flex; align-items:center; justify-content:center; font-size:1.45rem;">
                    🤖
                </div>
                <div style="text-align:right;">
                    <span style="font-size:0.74rem; font-weight:900; color:#000000; background:#f4f4f5; border:2px solid #000000; border-radius:6px; padding:0.2rem 0.55rem; box-shadow:2px 2px 0px #000000;">
                        ESP8266 WI-FI
                    </span>
                    <div style="font-size:0.72rem; font-weight:700; color:#4b5563; margin-top:4px;">Seam 3 (-142.5m)</div>
                </div>
            </div>
            <div style="font-size:1.3rem; font-weight:900; color:#000000; margin:0.75rem 0 0.25rem 0; letter-spacing:-0.02em;">
                Underground Rover
            </div>
            <div style="font-size:0.84rem; font-weight:600; color:#374151; line-height:1.45; min-height:46px; margin-bottom:0.75rem;">
                Mobile wireless exploratory telemetry streaming ambient temperature, relative humidity, thermal heat index, and gas expansion matrix.
            </div>
            <div class="spec-grid">
                <div class="spec-cell">
                    <span class="spec-label">ACTIVE SENSOR</span>
                    <span class="spec-val">DHT11 Environmental</span>
                </div>
                <div class="spec-cell">
                    <span class="spec-label">EXPANSION PORTS</span>
                    <span class="spec-val">MQ-4 &middot; MQ-7 &middot; O2</span>
                </div>
                <div class="spec-cell">
                    <span class="spec-label">LOCATION DEPTH</span>
                    <span class="spec-val">-142.5m (Working Seam 3)</span>
                </div>
                <div class="spec-cell">
                    <span class="spec-label">NETWORK LINK</span>
                    <span class="spec-val">ESP8266 HTTP / REST</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open Rover →", key="hub_open_rover", use_container_width=True, type="primary"):
                st.session_state["current_view"] = "rover"
                st.rerun()

    st.markdown("<div style='height: 1.0rem;'></div>", unsafe_allow_html=True)

    # ── ROW 2: Module 03 (Subsidence) & Module 04 (Ground Scanner) ──
    row2_col1, row2_col2 = st.columns(2, gap="large")

    # ── Module 3: Subsidence (Purple Folder Tab) ──
    with row2_col1:
        st.markdown("""
        <div style="margin-bottom:-4px; margin-left:14px; position:relative; z-index:2; display:inline-block;">
            <div style="background:#c084fc; border:2.5px solid #000000; border-bottom:2.5px solid #c084fc; border-radius:10px 10px 0 0; padding:0.3rem 0.9rem; font-size:0.72rem; font-weight:900; color:#000000; letter-spacing:0.04em; text-transform:uppercase; box-shadow:2px -2px 0px #000000;">
                📁 MODULE 03 &middot; GEOTECHNICAL
            </div>
        </div>
        """, unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("""
            <span class="hub-marker-m3" style="display:none;"></span>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                <div style="width:48px; height:48px; border-radius:12px; background:#d8b4fe; border:2.5px solid #000000; box-shadow:2.5px 2.5px 0px #000000; display:flex; align-items:center; justify-content:center; font-size:1.45rem;">
                    📡
                </div>
                <div style="text-align:right;">
                    <span style="font-size:0.74rem; font-weight:900; color:#000000; background:#f4f4f5; border:2px solid #000000; border-radius:6px; padding:0.2rem 0.55rem; box-shadow:2px 2px 0px #000000; margin-right:4px;">
                        COM7 9600 BAUD
                    </span>
                    <span style="font-size:0.65rem; font-weight:900; color:#ffffff; background:#ff5388; border:2px solid #000000; border-radius:6px; padding:0.18rem 0.45rem; box-shadow:2px 2px 0px #000000;">
                        NEW
                    </span>
                    <div style="font-size:0.72rem; font-weight:700; color:#4b5563; margin-top:4px;">Overburden Strata</div>
                </div>
            </div>
            <div style="font-size:1.3rem; font-weight:900; color:#000000; margin:0.75rem 0 0.25rem 0; letter-spacing:-0.02em;">
                Surface Subsidence Mesh
            </div>
            <div style="font-size:0.84rem; font-weight:600; color:#374151; line-height:1.45; min-height:46px; margin-bottom:0.75rem;">
                Physical 6-DOF inertial sensor streaming live ground tilt (mm/m) and vibration to model sinkhole settlement basins via Peck's formulation.
            </div>
            <div class="spec-grid">
                <div class="spec-cell">
                    <span class="spec-label">HARDWARE NODE</span>
                    <span class="spec-val">Arduino Nano + MPU6050</span>
                </div>
                <div class="spec-cell">
                    <span class="spec-label">SERIAL PORT</span>
                    <span class="spec-val">COM7 @ 9600 Baud (8N1)</span>
                </div>
                <div class="spec-cell">
                    <span class="spec-label">CORE METRICS</span>
                    <span class="spec-val">Tilt (mm/m) &middot; Vib &middot; Roll</span>
                </div>
                <div class="spec-cell">
                    <span class="spec-label">PREDICTIVE MODEL</span>
                    <span class="spec-val">Peck's Gaussian Trough</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open Subsidence →", key="hub_open_sub", use_container_width=True, type="primary"):
                st.session_state["current_view"] = "subsidence"
                st.rerun()

    # ── Module 4: Ground Scanner & Dynamic Mesh (Orange Folder Tab) ──
    with row2_col2:
        st.markdown("""
        <div style="margin-bottom:-4px; margin-left:14px; position:relative; z-index:2; display:inline-block;">
            <div style="background:#fb923c; border:2.5px solid #000000; border-bottom:2.5px solid #fb923c; border-radius:10px 10px 0 0; padding:0.3rem 0.9rem; font-size:0.72rem; font-weight:900; color:#000000; letter-spacing:0.04em; text-transform:uppercase; box-shadow:2px -2px 0px #000000;">
                📁 MODULE 04 &middot; SPATIAL SCAN
            </div>
        </div>
        """, unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("""
            <span class="hub-marker-m4" style="display:none;"></span>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                <div style="width:48px; height:48px; border-radius:12px; background:#fdba74; border:2.5px solid #000000; box-shadow:2.5px 2.5px 0px #000000; display:flex; align-items:center; justify-content:center; font-size:1.45rem;">
                    🗺️
                </div>
                <div style="text-align:right;">
                    <span style="font-size:0.74rem; font-weight:900; color:#000000; background:#f4f4f5; border:2px solid #000000; border-radius:6px; padding:0.2rem 0.55rem; box-shadow:2px 2px 0px #000000; margin-right:4px;">
                        2D CONTOUR
                    </span>
                    <span style="font-size:0.65rem; font-weight:900; color:#000000; background:#fde047; border:2px solid #000000; border-radius:6px; padding:0.18rem 0.45rem; box-shadow:2px 2px 0px #000000;">
                        ADAPTIVE
                    </span>
                    <div style="font-size:0.72rem; font-weight:700; color:#4b5563; margin-top:4px;">Zone Separation</div>
                </div>
            </div>
            <div style="font-size:1.3rem; font-weight:900; color:#000000; margin:0.75rem 0 0.25rem 0; letter-spacing:-0.02em;">
                Ground Scan &amp; Mesh
            </div>
            <div style="font-size:0.84rem; font-weight:600; color:#374151; line-height:1.45; min-height:46px; margin-bottom:0.75rem;">
                Autonomous concession area survey, DGMS zone risk classification, and dynamic sensor node density scaling in critical danger zones.
            </div>
            <div class="spec-grid">
                <div class="spec-cell">
                    <span class="spec-label">SURVEY INPUT</span>
                    <span class="spec-val">Concession Grid 2D</span>
                </div>
                <div class="spec-cell">
                    <span class="spec-label">RISK ZONES</span>
                    <span class="spec-val">4 Partitioned Sectors</span>
                </div>
                <div class="spec-cell">
                    <span class="spec-label">DANGER DENSITY</span>
                    <span class="spec-val">Auto-Node Allocation</span>
                </div>
                <div class="spec-cell">
                    <span class="spec-label">EARLY WARNING</span>
                    <span class="spec-val">+45 min Lead Time</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Open Scanner →", key="hub_open_scan", use_container_width=True, type="primary"):
                st.session_state["current_view"] = "scanner"
                st.rerun()


# -------------------------------------------------------------
# SCREEN 2: OPTION 1 - Fire & Smoke Camera
# -------------------------------------------------------------
def render_module1_camera():
    render_header()
    
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; background:#ffffff; border:2.5px solid #000000; border-radius:14px; padding:0.8rem 1.2rem; margin-bottom:1.1rem; box-shadow:4px 4px 0px #000000;">
        <div style="display:flex; align-items:center; gap:0.85rem;">
            <div style="width:44px; height:44px; border-radius:12px; background:#4ade80; border:2.5px solid #000000; box-shadow:2.5px 2.5px 0px #000000; display:flex; align-items:center; justify-content:center; font-size:1.3rem;">📹</div>
            <div>
                <div style="font-size:1.1rem; font-weight:900; color:#000000; letter-spacing:-0.02em;">Module 01 &middot; Surface Optical Fire & Smoke Sentinel</div>
                <div style="font-size:0.78rem; font-weight:700; color:#4b5563;">Live optical stream with OpenCV dynamic flame flicker & YOLOv8 plume classifier</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:0.5rem;">
            <span style="font-size:0.75rem; font-weight:900; color:#000000; background:#4ade80; border:2px solid #000000; border-radius:8px; padding:0.25rem 0.65rem; box-shadow:2px 2px 0px #000000;">WEBCAM ACTIVE</span>
            <span style="font-size:0.75rem; font-weight:900; color:#000000; background:#f4f4f5; border:2px solid #000000; border-radius:8px; padding:0.25rem 0.65rem; box-shadow:2px 2px 0px #000000;">HYBRID AI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Toggles
    with st.container(border=True):
        st.markdown("<span style='font-size:0.7rem; font-weight:500; color:#9ca3af; text-transform:uppercase; letter-spacing:0.06em;'>Feature Controls</span>", unsafe_allow_html=True)
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            toggle_smoke = ui.switch(default_checked=config.enable_smoke, label="Smoke Detection", key="sw_smoke")
        with t2:
            toggle_audio = ui.switch(default_checked=config.enable_audio_alarm, label="Audio Siren", key="sw_audio")
        with t3:
            toggle_voice = ui.switch(default_checked=config.enable_voice_alert, label="Voice Alerts", key="sw_voice")
        with t4:
            toggle_snap = ui.switch(default_checked=config.enable_snapshots, label="Auto-Save Photos", key="sw_snap")
        
        config.enable_smoke = toggle_smoke
        config.enable_audio_alarm = toggle_audio
        config.enable_voice_alert = toggle_voice
        config.enable_snapshots = toggle_snap
        alert_manager.set_audio(toggle_audio)
        alert_manager.set_voice(toggle_voice)
        alert_manager.set_snapshots(toggle_snap)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Shadcn Tabs
    selected_tab = ui.tabs(options=["Live Camera Feed", "Saved Incident Photos"], default_value="Live Camera Feed", key="cam_tabs_sel")
    
    if selected_tab == "Saved Incident Photos":
        if st.session_state.get("camera_active", False):
            st.session_state["camera_active"] = False
            if st.session_state.get("cap") is not None:
                try:
                    st.session_state["cap"].release()
                except Exception:
                    pass
                st.session_state["cap"] = None
        st.write("Recent snapshots recorded during fire or smoke detection:")
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

    elif selected_tab == "Live Camera Feed":
        col_video, col_controls = st.columns([2.5, 1])
        
        with col_controls:
            with st.container(border=True):
                st.markdown("<span style='font-size:0.75rem; font-weight:900; color:#000000; text-transform:uppercase; letter-spacing:0.04em;'>Camera Optical Stream</span>", unsafe_allow_html=True)
                start_btn = st.button("▶ Start Camera", key="btn_start_cam", type="primary", use_container_width=True)
                stop_btn = st.button("■ Stop Camera", key="btn_stop_cam", use_container_width=True)
                
                if start_btn:
                    st.session_state["camera_active"] = True
                if stop_btn:
                    st.session_state["camera_active"] = False
                    if st.session_state.get("cap") is not None:
                        try:
                            st.session_state["cap"].release()
                        except Exception:
                            pass
                        st.session_state["cap"] = None
                    st.rerun()

                if st.session_state.get("camera_active", False):
                    st.markdown("<div style='margin:0.4rem 0 0.2rem 0; display:flex; align-items:center; gap:0.4rem;'><span style='width:8px; height:8px; border-radius:50%; background:#22c55e; border:1px solid #000; display:inline-block;'></span><span style='color:#000000; font-size:0.76rem; font-weight:800;'>SENSOR RUNNING</span></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='margin:0.4rem 0 0.2rem 0; display:flex; align-items:center; gap:0.4rem;'><span style='width:8px; height:8px; border-radius:50%; background:#9ca3af; border:1px solid #000; display:inline-block;'></span><span style='color:#4b5563; font-size:0.76rem; font-weight:700;'>OPTICAL STANDBY</span></div>", unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("<span style='font-size:0.75rem; font-weight:900; color:#000000; text-transform:uppercase; letter-spacing:0.04em;'>AI Detection Model</span>", unsafe_allow_html=True)
                engine_mode = st.selectbox("Engine", ["HYBRID", "CV_ONLY", "DL_ONLY"], index=0)
                conf_val = st.slider("Confidence Threshold", 0.20, 0.80, float(config.confidence_threshold), 0.05)
                config.engine_mode = engine_mode
                config.confidence_threshold = conf_val
                
                st.markdown("---")
                st.markdown("<span style='font-size:0.75rem; font-weight:900; color:#000000; text-transform:uppercase; letter-spacing:0.04em;'>Audio Alerts</span>", unsafe_allow_html=True)
                c_test1, c_test2 = st.columns(2)
                with c_test1:
                    if st.button("🚨 Siren", key="btn_test_siren", use_container_width=True):
                        try:
                            import winsound
                            winsound.Beep(1800, 200)
                        except Exception:
                            pass
                with c_test2:
                    if st.button("🔊 Voice", key="btn_test_voice", use_container_width=True):
                        try:
                            import pyttsx3
                            tts = pyttsx3.init()
                            tts.say("Warning! Fire detection operational.")
                            tts.runAndWait()
                        except Exception:
                            pass

        with col_video:
            video_placeholder = st.empty()
            
            # Metric Placeholders for live updates
            m1, m2, m3, m4 = st.columns(4)
            m1_ph = m1.empty()
            m2_ph = m2.empty()
            m3_ph = m3.empty()
            m4_ph = m4.empty()
            with m1_ph:
                st.markdown(render_neo_metric("Status", "Standby" if not st.session_state.get("camera_active", False) else "Running", "Optical Sentinel", "📹", "#ffffff", "LIVE" if st.session_state.get("camera_active", False) else "IDLE", "#4ade80" if st.session_state.get("camera_active", False) else "#e2e8f0"), unsafe_allow_html=True)
            with m2_ph:
                st.markdown(render_neo_metric("FPS", "0.0", "Frame capture rate", "⚡", "#ffffff"), unsafe_allow_html=True)
            with m3_ph:
                st.markdown(render_neo_metric("Threat", "None", "Combustion detector", "🛡️", "#ffffff", "CLEAR", "#4ade80"), unsafe_allow_html=True)
            with m4_ph:
                st.markdown(render_neo_metric("Incidents", str(len(list(INCIDENTS_DIR.glob("*.jpg")))), "Logged snapshots", "📁", "#ffffff"), unsafe_allow_html=True)

        if st.session_state.get("camera_active", False):
            cap = st.session_state.get("cap")
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(config.camera_index, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(config.camera_index)
                st.session_state["cap"] = cap
                
            if not cap.isOpened():
                st.error("Could not open laptop webcam. Please check camera connection.")
                st.session_state["camera_active"] = False
                st.session_state["cap"] = None
            else:
                if st.session_state.get("detector") is None:
                    st.session_state["detector"] = FireDetector(mode=config.engine_mode, enable_smoke=config.enable_smoke)
                
                detector = st.session_state["detector"]
                detector.mode = config.engine_mode
                detector.enable_smoke = config.enable_smoke
                detector.confidence_threshold = config.confidence_threshold
                detector.set_persistence_frames(config.persistence_frames)
                
                active_toggles = {
                    "smoke": config.enable_smoke,
                    "audio": config.enable_audio_alarm,
                    "voice": config.enable_voice_alert,
                    "snapshots": config.enable_snapshots
                }
                
                fps_time = time.time()
                frame_count = 0
                fps = 0.0
                
                st.info("Webcam active. Click Stop Camera to end session.")
                
                ret = True
                while cap.isOpened() and st.session_state.get("camera_active", False):
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    frame_count += 1
                    now = time.time()
                    if now - fps_time >= 1.0:
                        fps = frame_count / (now - fps_time)
                        frame_count = 0
                        fps_time = now
                        
                    detector.mode = config.engine_mode
                    detector.enable_smoke = config.enable_smoke
                    detector.confidence_threshold = config.confidence_threshold
                    active_toggles["smoke"] = config.enable_smoke
                    active_toggles["audio"] = config.enable_audio_alarm
                    active_toggles["voice"] = config.enable_voice_alert
                    active_toggles["snapshots"] = config.enable_snapshots
                    
                    annotated_frame, detections, sys_state = detector.detect(frame)
                    
                    if sys_state == "ALARM":
                        best_det = max(detections, key=lambda d: d.get("confidence", 0.0)) if detections else {}
                        hazard_type = best_det.get("label", "fire")
                        conf = best_det.get("confidence", 0.0)
                        alert_manager.trigger(hazard_type, conf, frame)
                        
                    hud_frame = hud_visualizer.draw_hud(
                        annotated_frame,
                        fps=fps,
                        system_state=sys_state,
                        active_toggles=active_toggles,
                        engine_mode=config.engine_mode
                    )
                    
                    rgb_frame = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                    
                    threat_label = "None"
                    if detections:
                        threat_label = ", ".join(set(d["label"].upper() for d in detections))
                        
                    threat_bg = "#ff4d4d" if threat_label != "None" else "#4ade80"
                    m1_ph.markdown(render_neo_metric("Status", sys_state, "Optical Sentinel", "📹", "#ffffff", "LIVE", "#4ade80"), unsafe_allow_html=True)
                    m2_ph.markdown(render_neo_metric("FPS", f"{fps:.1f}", "Frame capture rate", "⚡", "#ffffff"), unsafe_allow_html=True)
                    m3_ph.markdown(render_neo_metric("Threat", threat_label, "Combustion detector", "🔥" if threat_label != "None" else "🛡️", "#ffffff", "ALERT" if threat_label != "None" else "CLEAR", threat_bg), unsafe_allow_html=True)
                    
                    time.sleep(0.01)
                    
                if not cap.isOpened() or not ret:
                    if st.session_state.get("cap") is not None:
                        try:
                            st.session_state["cap"].release()
                        except Exception:
                            pass
                        st.session_state["cap"] = None
                    st.session_state["camera_active"] = False

# -------------------------------------------------------------
# Underground Rover Knowledge Base & Safety Guidelines Panel
# -------------------------------------------------------------
def render_rover_knowledge_base():
    st.markdown("""
    <!-- Protruding Folder Tab for Knowledge Base -->
    <div style="margin-top:1.6rem; margin-bottom:-4px; margin-left:14px; position:relative; z-index:2; display:inline-block;">
        <div style="background:#fde047; border:2.5px solid #000000; border-bottom:2.5px solid #fde047; border-radius:10px 10px 0 0; padding:0.35rem 1rem; font-size:0.75rem; font-weight:900; color:#000000; letter-spacing:0.04em; text-transform:uppercase; box-shadow:2px -2px 0px #000000;">
            📁 STATUTORY MINE CODES &middot; PHYSIOLOGICAL SAFETY STANDARDS (DGMS / CMR 2017)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1rem; flex-wrap:wrap; gap:0.5rem;">
            <div>
                <div style="font-size:1.3rem; font-weight:900; color:#000000; letter-spacing:-0.02em;">
                    Underground Safety Limits &amp; Detection Diagnostic Standards
                </div>
                <div style="font-size:0.8rem; font-weight:700; color:#4b5563; margin-top:2px;">
                    Statutory benchmarks under Coal Mines Regulations (CMR 2017), sensor detection physics, physiological human thresholds &amp; mandatory emergency protocols.
                </div>
            </div>
            <div style="display:flex; gap:0.4rem; align-items:center;">
                <span style="font-size:0.72rem; font-weight:900; background:#facc15; border:1.5px solid #000; border-radius:6px; padding:0.18rem 0.5rem; box-shadow:1.5px 1.5px 0px #000; color:#000;">DGMS APPROVED</span>
                <span style="font-size:0.72rem; font-weight:900; background:#4ade80; border:1.5px solid #000; border-radius:6px; padding:0.18rem 0.5rem; box-shadow:1.5px 1.5px 0px #000; color:#000;">CMR 2017 COMPLIANT</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        tab_limits, tab_physics, tab_health, tab_sop = st.tabs([
            "📊 Statutory Safe Limits Matrix",
            "🔬 How Sensors Detect (Physics & Math)",
            "🩺 Miner Physiological & Health Impact",
            "🚨 DGMS Evacuation & Action SOP"
        ])
        
        with tab_limits:
            st.markdown("""
            <div style="overflow-x:auto; margin-bottom:0.5rem;">
                <table style="width:100%; border-collapse:collapse; font-size:0.82rem; font-weight:600; text-align:left; border:2px solid #000000; background:#ffffff;">
                    <thead>
                        <tr style="background:#fafafa; border-bottom:2.5px solid #000000;">
                            <th style="padding:0.65rem 0.8rem; border-right:2px solid #000; font-weight:900; color:#000; text-transform:uppercase; font-size:0.72rem;">Parameter</th>
                            <th style="padding:0.65rem 0.8rem; border-right:2px solid #000; font-weight:900; color:#000; text-transform:uppercase; font-size:0.72rem;">Hardware / Source</th>
                            <th style="padding:0.65rem 0.8rem; border-right:2px solid #000; font-weight:900; color:#000; text-transform:uppercase; font-size:0.72rem;">Human Safe Zone</th>
                            <th style="padding:0.65rem 0.8rem; border-right:2px solid #000; font-weight:900; color:#000; text-transform:uppercase; font-size:0.72rem;">Advisory / Warning</th>
                            <th style="padding:0.65rem 0.8rem; border-right:2px solid #000; font-weight:900; color:#000; text-transform:uppercase; font-size:0.72rem;">Critical Evacuation</th>
                            <th style="padding:0.65rem 0.8rem; font-weight:900; color:#000; text-transform:uppercase; font-size:0.72rem;">Regulatory Standard</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom:1.5px solid #000;">
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000; font-weight:800; color:#000;">🌡️ Ambient Temp</td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;">DHT11 Thermistor</td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#4ade80; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">18°C – 28°C</span></td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#fde047; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">29°C – 33.5°C</span></td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#ff4d4d; color:#fff; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">&gt; 35.0°C</span></td>
                            <td style="padding:0.6rem 0.8rem; font-weight:700;">DGMS Circular 04 &middot; CMR 153</td>
                        </tr>
                        <tr style="border-bottom:1.5px solid #000; background:#fafafa;">
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000; font-weight:800; color:#000;">💧 Relative Humidity</td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;">DHT11 Capacitive</td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#4ade80; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">40% – 70% RH</span></td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#fde047; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">71% – 80% RH</span></td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#ff4d4d; color:#fff; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">&gt; 85% with High Temp</span></td>
                            <td style="padding:0.6rem 0.8rem; font-weight:700;">DGMS Wet-Bulb &lt; 30.5°C Limit</td>
                        </tr>
                        <tr style="border-bottom:1.5px solid #000;">
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000; font-weight:800; color:#000;">🔥 Heat Index (Apparent)</td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;">NOAA Rothfusz Math</td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#4ade80; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">&lt; 29.0°C</span></td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#fde047; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">30.0°C – 38.0°C</span></td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#ff4d4d; color:#fff; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">&ge; 39.0°C (Heat Stroke)</span></td>
                            <td style="padding:0.6rem 0.8rem; font-weight:700;">OSHA / NIOSH Heat Stress Standard</td>
                        </tr>
                        <tr style="border-bottom:1.5px solid #000; background:#fafafa;">
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000; font-weight:800; color:#000;">🌫️ Dew Point</td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;">Magnus-Tetens Model</td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#4ade80; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">10°C – 18°C</span></td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#fde047; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">19°C – 23°C</span></td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#ff4d4d; color:#fff; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">&ge; 24.0°C (Oppressive)</span></td>
                            <td style="padding:0.6rem 0.8rem; font-weight:700;">ASHRAE Confined Space</td>
                        </tr>
                        <tr style="border-bottom:1.5px solid #000;">
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000; font-weight:800; color:#000;">⚡ Methane (CH4)</td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;">MQ-4 Catalytic Bead</td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#4ade80; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">&lt; 0.50% vol</span></td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#fde047; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">0.75% (Airflow Boost)</span></td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#ff4d4d; color:#fff; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">&ge; 1.25% (Mandatory Evac)</span></td>
                            <td style="padding:0.6rem 0.8rem; font-weight:700;">CMR 2017 Reg 153 &amp; 169 (LEL: 5.0%)</td>
                        </tr>
                        <tr style="border-bottom:1.5px solid #000; background:#fafafa;">
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000; font-weight:800; color:#000;">☠️ Carbon Monoxide (CO)</td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;">MQ-7 Micro-Hotplate</td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#4ade80; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">&lt; 25 ppm (0.0025%)</span></td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#fde047; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">25 – 50 ppm (TWA 8hr)</span></td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#ff4d4d; color:#fff; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">&gt; 50 ppm (Immediate Evac)</span></td>
                            <td style="padding:0.6rem 0.8rem; font-weight:700;">DGMS Circular 02 &middot; CMR 153</td>
                        </tr>
                        <tr style="border-bottom:1.5px solid #000;">
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000; font-weight:800; color:#000;">🌿 Carbon Dioxide (CO2)</td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;">MQ-135 Metal Oxide</td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#4ade80; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">&lt; 0.50% (5,000 ppm)</span></td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#fde047; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">0.50% – 1.00%</span></td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#ff4d4d; color:#fff; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">&gt; 1.25% in Return Airway</span></td>
                            <td style="padding:0.6rem 0.8rem; font-weight:700;">CMR 2017 Reg 153 Blackdamp Standard</td>
                        </tr>
                        <tr style="background:#fafafa;">
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000; font-weight:800; color:#000;">🫁 Oxygen (O2)</td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;">Electrochemical Cell</td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#4ade80; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">19.5% – 21.0%</span></td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#fde047; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">17.0% – 19.0% (Deficit)</span></td>
                            <td style="padding:0.6rem 0.8rem; border-right:2px solid #000;"><span style="background:#ff4d4d; color:#fff; border:1px solid #000; border-radius:4px; padding:0.1rem 0.4rem; font-weight:800; font-size:0.75rem;">&lt; 19.0% (Statutory Violation)</span></td>
                            <td style="padding:0.6rem 0.8rem; font-weight:700;">Mines Act 1952 Sec 19 &middot; CMR 153</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
        with tab_physics:
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("""
                <div style="background:#ffffff; border:2px solid #000; border-radius:10px; padding:0.85rem; box-shadow:3px 3px 0px #000; margin-bottom:0.8rem;">
                    <div style="font-size:0.88rem; font-weight:900; color:#000; display:flex; justify-content:space-between; align-items:center;">
                        <span>🌡️ DHT11 Environmental Transducer</span>
                        <span style="background:#4ade80; border:1.5px solid #000; border-radius:4px; font-size:0.65rem; padding:0.1rem 0.4rem; font-weight:900;">ACTIVE TELEMETRY</span>
                    </div>
                    <div style="font-size:0.78rem; font-weight:600; color:#374151; margin-top:0.4rem; line-height:1.45;">
                        <b>• Temperature:</b> Utilizes a Negative Temperature Coefficient (NTC) semiconductor thermistor bead. As air temperature warms, electron-hole generation increases, driving down electrical resistance according to the Steinhart-Hart equation.<br>
                        <b>• Relative Humidity:</b> Uses a resistive-capacitive substrate where a moisture-absorptive polymer changes dielectric permittivity as water vapor adsorbs, converting moisture into a calibrated digital pulse train.
                    </div>
                </div>
                
                <div style="background:#ffffff; border:2px solid #000; border-radius:10px; padding:0.85rem; box-shadow:3px 3px 0px #000; margin-bottom:0.8rem;">
                    <div style="font-size:0.88rem; font-weight:900; color:#000; display:flex; justify-content:space-between; align-items:center;">
                        <span>🔥 NOAA Heat Index &amp; Magnus Dew Point</span>
                        <span style="background:#fde047; border:1.5px solid #000; border-radius:4px; font-size:0.65rem; padding:0.1rem 0.4rem; font-weight:900;">MATH FORMULATION</span>
                    </div>
                    <div style="font-size:0.78rem; font-weight:600; color:#374151; margin-top:0.4rem; line-height:1.45;">
                        <b>• Heat Index Formulation:</b> Evaluated via Rothfusz polynomial regression:
                        <div style="background:#f4f4f5; border:1px solid #000; border-radius:6px; padding:0.3rem 0.5rem; font-family:monospace; font-size:0.72rem; margin:0.3rem 0; font-weight:800;">
                            HI = -42.379 + 2.049·T + 10.143·RH - 0.224·T·RH ...
                        </div>
                        Models the reduced evaporative sweat transfer from human skin in saturated mine air.<br>
                        <b>• Dew Point (Magnus-Tetens):</b> Computes saturation vapor pressure curve to predict condensation on mine roof rock bolts.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_p2:
                st.markdown("""
                <div style="background:#ffffff; border:2px solid #000; border-radius:10px; padding:0.85rem; box-shadow:3px 3px 0px #000; margin-bottom:0.8rem;">
                    <div style="font-size:0.88rem; font-weight:900; color:#000; display:flex; justify-content:space-between; align-items:center;">
                        <span>⚡ MQ-4 Methane Catalytic Surface</span>
                        <span style="background:#e2e8f0; border:1.5px solid #000; border-radius:4px; font-size:0.65rem; padding:0.1rem 0.4rem; font-weight:900;">EXPANSION READY</span>
                    </div>
                    <div style="font-size:0.78rem; font-weight:600; color:#374151; margin-top:0.4rem; line-height:1.45;">
                        <b>• Detection Chemistry:</b> Micro-machined alumina ceramic coated with Tin Dioxide (SnO2). In oxygen-rich clean air, negative oxygen ions (O⁻, O²⁻) adsorb onto the semiconductor crystal, creating an electrostatic potential barrier.<br>
                        <b>• Firedamp Interaction:</b> When CH4 gas passes over the 300°C heated surface, catalytic oxidation releases trapped electrons back to the conduction band, decreasing sensor surface resistance (Rs) proportionally to volumetric methane percentage.
                    </div>
                </div>
                
                <div style="background:#ffffff; border:2px solid #000; border-radius:10px; padding:0.85rem; box-shadow:3px 3px 0px #000; margin-bottom:0.8rem;">
                    <div style="font-size:0.88rem; font-weight:900; color:#000; display:flex; justify-content:space-between; align-items:center;">
                        <span>☠️ MQ-7 Dual-Thermal Cycle for CO</span>
                        <span style="background:#e2e8f0; border:1.5px solid #000; border-radius:4px; font-size:0.65rem; padding:0.1rem 0.4rem; font-weight:900;">EXPANSION READY</span>
                    </div>
                    <div style="font-size:0.78rem; font-weight:600; color:#374151; margin-top:0.4rem; line-height:1.45;">
                        <b>• Micro-Hotplate Duty Cycle:</b> CO detection requires discriminating toxic Whitedamp from background hydrocarbons:<br>
                        1. <i>Regeneration Cycle (60s @ 5.0V):</i> Burns off residual impurities &amp; desorbs VOCs.<br>
                        2. <i>Measurement Cycle (90s @ 1.4V):</i> Low-temperature chemisorption where CO selectively oxidizes to CO2, generating measurable conductivity proportional to ppm.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        with tab_health:
            c_h1, c_h2, c_h3 = st.columns(3)
            with c_h1:
                st.markdown("""
                <div style="background:#ffffff; border:2px solid #000; border-radius:10px; padding:0.85rem; box-shadow:3px 3px 0px #000; height:100%;">
                    <div style="font-size:0.86rem; font-weight:900; color:#000; display:flex; align-items:center; gap:0.4rem;">
                        <span>☠️ Carbon Monoxide Toxicity</span>
                    </div>
                    <div style="font-size:0.76rem; font-weight:700; color:#dc2626; margin-top:0.25rem;">"The Silent Killer" (Whitedamp)</div>
                    <div style="font-size:0.77rem; font-weight:600; color:#374151; margin-top:0.35rem; line-height:1.4;">
                        <b>• Mechanism:</b> CO binds to human hemoglobin with <b>210x higher affinity</b> than oxygen, synthesizing Carboxyhemoglobin (HbCO).<br>
                        <b>• 25–50 ppm:</b> TWA safe ceiling. Mild frontal headache after 2 hours.<br>
                        <b>• 100–200 ppm:</b> Throbbing headache, dizziness, nausea.<br>
                        <b>• 400–800 ppm:</b> Motor collapse, confusion, loss of consciousness within 45 min.<br>
                        <b>• &gt; 1200 ppm:</b> Fatal anoxia within 5 to 15 minutes.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c_h2:
                st.markdown("""
                <div style="background:#ffffff; border:2px solid #000; border-radius:10px; padding:0.85rem; box-shadow:3px 3px 0px #000; height:100%;">
                    <div style="font-size:0.86rem; font-weight:900; color:#000; display:flex; align-items:center; gap:0.4rem;">
                        <span>🫁 Oxygen Deficit (Hypoxia)</span>
                    </div>
                    <div style="font-size:0.76rem; font-weight:700; color:#2563eb; margin-top:0.25rem;">Blackdamp Displacement Hazard</div>
                    <div style="font-size:0.77rem; font-weight:600; color:#374151; margin-top:0.35rem; line-height:1.4;">
                        <b>• 19.5% – 21.0%:</b> Optimal respiration in mining workings.<br>
                        <b>• 16.0% – 19.0%:</b> Increased respiratory rate, tachycardia, impaired emotional judgment and manual dexterity.<br>
                        <b>• 12.0% – 16.0%:</b> Peripheral cyanosis (blue skin), rapid exhaustion, dizziness.<br>
                        <b>• 6.0% – 10.0%:</b> Immediate nausea, convulsions, syncope; irreversible brain necrosis occurs after 4 minutes.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with c_h3:
                st.markdown("""
                <div style="background:#ffffff; border:2px solid #000; border-radius:10px; padding:0.85rem; box-shadow:3px 3px 0px #000; height:100%;">
                    <div style="font-size:0.86rem; font-weight:900; color:#000; display:flex; align-items:center; gap:0.4rem;">
                        <span>🔥 Deep Seam Heat Strain</span>
                    </div>
                    <div style="font-size:0.76rem; font-weight:700; color:#d97706; margin-top:0.25rem;">Geothermal &amp; Humidity Stress</div>
                    <div style="font-size:0.77rem; font-weight:600; color:#374151; margin-top:0.35rem; line-height:1.4;">
                        <b>• Wet-Bulb Benchmark:</b> DGMS mandates that when wet-bulb temperature exceeds <b>30.5°C</b>, airflow must be boosted to 1.0 m/s minimum.<br>
                        <b>• Evaporative Failure:</b> Above 80% RH, sweat drips off without evaporating, preventing body cooling.<br>
                        <b>• Core Temp &gt; 38.5°C:</b> Causes heat cramps, hyperthermia, circulatory collapse, and potentially fatal heat stroke.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        with tab_sop:
            st.markdown("""
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.75rem;">
                <div style="background:#ffffff; border:2px solid #000; border-radius:10px; padding:0.85rem; box-shadow:3px 3px 0px #000;">
                    <div style="display:flex; align-items:center; gap:0.45rem; font-size:0.88rem; font-weight:900; color:#000;">
                        <span style="background:#fde047; border:1.5px solid #000; border-radius:6px; padding:0.15rem 0.45rem; font-size:0.7rem;">STAGE 1</span>
                        <span>DGMS Advisory Action Protocol (Code Yellow)</span>
                    </div>
                    <div style="font-size:0.77rem; font-weight:600; color:#374151; margin-top:0.45rem; line-height:1.45;">
                        <b>Trigger:</b> CH4 &ge; 0.75% | Temp &ge; 30°C | CO &ge; 25 ppm | O2 &le; 19.5%<br>
                        <b>1. Fan Regulation:</b> Shift in-charge must immediately adjust intake ventilation regulators to increase air velocity above 1.5 m/s across the working face.<br>
                        <b>2. Manual Verification:</b> Competent mining sirdar conducts flame safety lamp &amp; multi-gas sniffer cross-check at roof &amp; rib pockets.<br>
                        <b>3. Telemetry Log:</b> SCADA system flags warning state in DGMS statutory shift register.
                    </div>
                </div>
                
                <div style="background:#ffffff; border:2px solid #000; border-radius:10px; padding:0.85rem; box-shadow:3px 3px 0px #000;">
                    <div style="display:flex; align-items:center; gap:0.45rem; font-size:0.88rem; font-weight:900; color:#000;">
                        <span style="background:#ff4d4d; color:#fff; border:1.5px solid #000; border-radius:6px; padding:0.15rem 0.45rem; font-size:0.7rem;">STAGE 2</span>
                        <span>Statutory Emergency Evacuation (Code Red)</span>
                    </div>
                    <div style="font-size:0.77rem; font-weight:600; color:#374151; margin-top:0.45rem; line-height:1.45;">
                        <b>Trigger:</b> CH4 &ge; 1.25% | CO &gt; 50 ppm | O2 &lt; 19.0% | Heat Index &gt; 42°C<br>
                        <b>1. Immediate Power Cut:</b> CMR Reg 169 mandates immediate de-energization of all non-intrinsically safe electrical power to the underground district.<br>
                        <b>2. Klaxon &amp; Voice Siren:</b> Audible surface &amp; subsurface sirens sound continuously.<br>
                        <b>3. SCSR Donning:</b> Miners immediately unlatch and don 30-min Self-Contained Self-Rescuers (SCSR oxygen breathing apparatus).<br>
                        <b>4. Intake Roadway Egress:</b> Evacuation along pre-marked fresh air intake roadways to pit-bottom cage shaft.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# -------------------------------------------------------------
# SCREEN 3: OPTION 2 - Underground Rover
# -------------------------------------------------------------
def render_module2_rover():
    render_header()
    
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; background:#ffffff; border:2.5px solid #000000; border-radius:14px; padding:0.8rem 1.2rem; margin-bottom:1.1rem; box-shadow:4px 4px 0px #000000;">
        <div style="display:flex; align-items:center; gap:0.85rem;">
            <div style="width:44px; height:44px; border-radius:12px; background:#fde047; border:2.5px solid #000000; box-shadow:2.5px 2.5px 0px #000000; display:flex; align-items:center; justify-content:center; font-size:1.3rem;">🤖</div>
            <div>
                <div style="font-size:1.1rem; font-weight:900; color:#000000; letter-spacing:-0.02em;">Module 02 &middot; Underground Mobile Rover Telemetry</div>
                <div style="font-size:0.78rem; font-weight:700; color:#4b5563;">Seam 3 (-142.5m) Environmental Microclimate & Modular Gas Expansion Array</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:0.5rem;">
            <span style="font-size:0.75rem; font-weight:900; color:#000000; background:#fde047; border:2px solid #000000; border-radius:8px; padding:0.25rem 0.65rem; box-shadow:2px 2px 0px #000000;">ESP8266 REST</span>
            <span style="font-size:0.75rem; font-weight:900; color:#000000; background:#4ade80; border:2px solid #000000; border-radius:8px; padding:0.25rem 0.65rem; box-shadow:2px 2px 0px #000000;">DHT11 ACTIVE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hardware connection row
    with st.container(border=True):
        hw_col1, hw_col2 = st.columns([3.5, 1])
        with hw_col1:
            esp_ip = st.text_input("ESP8266 Node IP Address", value=st.session_state["esp_ip"])
            st.session_state["esp_ip"] = esp_ip
        with hw_col2:
            st.write("")
            st.write("")
            if st.button("Sync Node ⚡", key="btn_ping_node", use_container_width=True):
                rover_telemetry.poll_esp8266(esp_ip)
                st.rerun()

    readings = rover_telemetry.get_readings(st.session_state["esp_ip"])
    
    # Show alert only if there is a hazard
    clim_status = readings.get("climate_status", "NORMAL")
    gas_installed = readings.get("gas_installed", False)
    if gas_installed or clim_status == "CRITICAL":
        st.error(f"🚨 {readings['alert_message']}")
        st.markdown("<br>", unsafe_allow_html=True)
    elif clim_status == "WARNING":
        st.warning(f"⚠️ {readings['alert_message']}")
        st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Unified Sensor Telemetry Matrix (Row 1: Microclimate, Row 2: Atmospheric Gases)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(render_neo_metric("DHT11 Temperature", f"{readings['temperature']} °C", "Ambient mine air temp", "🌡️", "#ffffff", "NORMAL" if clim_status=="NORMAL" else "ALERT", "#4ade80" if clim_status=="NORMAL" else "#ff4d4d"), unsafe_allow_html=True)
    with k2:
        st.markdown(render_neo_metric("DHT11 Humidity", f"{readings['humidity']} %", "Relative air moisture", "💧", "#ffffff", "64.6%", "#fde047"), unsafe_allow_html=True)
    with k3:
        st.markdown(render_neo_metric("Calculated Heat Index", f"{readings['heat_index']} °C", "Apparent thermal strain", "🔥", "#ffffff"), unsafe_allow_html=True)
    with k4:
        st.markdown(render_neo_metric("Dew Point", f"{readings['dew_point']} °C", "Condensation threshold", "🌫️", "#ffffff"), unsafe_allow_html=True)

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown(render_neo_metric("Methane (MQ-4)", readings["ch4_display"], "Safety threshold: < 1.0%", "⚡", "#ffffff", "UNWIRED", "#e2e8f0"), unsafe_allow_html=True)
    with g2:
        st.markdown(render_neo_metric("Carbon Monoxide (MQ-7)", readings["co_display"], "Toxic threshold: < 25 ppm", "☠️", "#ffffff", "UNWIRED", "#e2e8f0"), unsafe_allow_html=True)
    with g3:
        st.markdown(render_neo_metric("Carbon Dioxide (MQ-135)", readings["co2_display"], "Air purity index", "🌿", "#ffffff", "UNWIRED", "#e2e8f0"), unsafe_allow_html=True)
    with g4:
        st.markdown(render_neo_metric("Oxygen (Electrochemical O2)", readings["o2_display"], "Safe minimum: >= 19.5%", "🫁", "#ffffff", "UNWIRED", "#e2e8f0"), unsafe_allow_html=True)

    # ── MINE SAFETY KNOWLEDGE BASE & DIAGNOSTIC STANDARDS PANEL ──
    render_rover_knowledge_base()



# -------------------------------------------------------------
# SCREEN 4: OPTION 3 - Surface Subsidence Mesh
# -------------------------------------------------------------
def render_module3_subsidence():
    render_header()
    
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; background:#ffffff; border:2.5px solid #000000; border-radius:14px; padding:0.8rem 1.2rem; margin-bottom:1.1rem; box-shadow:4px 4px 0px #000000;">
        <div style="display:flex; align-items:center; gap:0.85rem;">
            <div style="width:44px; height:44px; border-radius:12px; background:#d8b4fe; border:2.5px solid #000000; box-shadow:2.5px 2.5px 0px #000000; display:flex; align-items:center; justify-content:center; font-size:1.3rem;">📡</div>
            <div>
                <div style="font-size:1.1rem; font-weight:900; color:#000000; letter-spacing:-0.02em;">Module 03 &middot; Geotechnical Subsidence & Inclinometer Mesh</div>
                <div style="font-size:0.78rem; font-weight:700; color:#4b5563;">Overburden strata tilt (mm/m), 6-DOF vibration, and Peck's sinkhole profile modeling</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:0.5rem;">
            <span style="font-size:0.75rem; font-weight:900; color:#000000; background:#c084fc; border:2px solid #000000; border-radius:8px; padding:0.25rem 0.65rem; box-shadow:2px 2px 0px #000000;">ARDUINO COM7</span>
            <span style="font-size:0.75rem; font-weight:900; color:#000000; background:#f4f4f5; border:2px solid #000000; border-radius:8px; padding:0.25rem 0.65rem; box-shadow:2px 2px 0px #000000;">9600 BAUD</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hardware Connection Row
    with st.container(border=True):
        hw_c1, hw_c2, hw_c3 = st.columns([3, 1, 1.2])
        with hw_c1:
            com_port = st.text_input("Arduino MPU6050 Serial Port", value=st.session_state.get("mpu_port", "COM7"))
            if com_port != st.session_state.get("mpu_port", "COM7"):
                st.session_state["mpu_port"] = com_port
                subsidence_mesh.set_port(com_port)
        with hw_c2:
            st.write("")
            st.write("")
            if st.button("Sync COM7 ⚡", key="btn_sync_mpu", use_container_width=True):
                subsidence_mesh.set_port(com_port)
                st.rerun()
        with hw_c3:
            st.write("")
            st.write("")
            live_stream = st.checkbox("Live Stream (5Hz)", value=st.session_state.get("mpu_live_stream", False), key="cb_mpu_live")
            st.session_state["mpu_live_stream"] = live_stream
            
    mesh_data = subsidence_mesh.get_mesh_data()
    
    # Status Banner
    if not mesh_data["hw_live"]:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; padding:0.6rem 1rem; background:#ffffff; border:2.5px solid #000000; border-radius:12px; box-shadow:3px 3px 0px #000000; margin:0.4rem 0 0.8rem 0;">
            <span style="font-size:0.8rem; color:#b45309; font-weight:800;">⚠️ {mesh_data['status_message']}</span>
            <span style="font-size:0.72rem; background:#fde047; border:1.5px solid #000; border-radius:6px; padding:0.12rem 0.45rem; font-weight:900; color:#000; box-shadow:1.5px 1.5px 0px #000;">BAUD: 9600 &middot; {mesh_data['port']}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; padding:0.6rem 1rem; background:#ffffff; border:2.5px solid #000000; border-radius:12px; box-shadow:3px 3px 0px #000000; margin:0.4rem 0 0.8rem 0;">
            <span style="font-size:0.82rem; color:#000000; font-weight:800;">● Live Hardware Stream Active (Arduino MPU6050 @ {mesh_data['port']})</span>
            <span style="font-size:0.72rem; background:#4ade80; border:1.5px solid #000; border-radius:6px; padding:0.12rem 0.45rem; font-weight:900; color:#000; box-shadow:1.5px 1.5px 0px #000;">9600 BAUD &middot; SYNCHRONIZED</span>
        </div>
        """, unsafe_allow_html=True)
        
    if "CRITICAL" in mesh_data["risk_level"]:
        st.error(f"🚨 {mesh_data['alert_header']} — {mesh_data['alert_sub']}")
    elif "ADVISORY" in mesh_data["risk_level"]:
        st.warning(f"▲ {mesh_data['alert_header']} — {mesh_data['alert_sub']}")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 1: Primary Geotechnical Inclinometer & Tremor Metrics
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(render_neo_metric("Ground Tilt", f"{mesh_data['tilt_mm_m']} mm/m", f"Angle: {mesh_data['tilt_deg']:.2f}° · DGMS Limit: 10 mm/m", "📐", "#ffffff", "LIVE", "#4ade80"), unsafe_allow_html=True)
    with k2:
        st.markdown(render_neo_metric("Micro-Seismic Vibration", f"{mesh_data['vibration_g']:.3f} g", "Dynamic ground tremor magnitude", "〰️", "#ffffff"), unsafe_allow_html=True)
    with k3:
        st.markdown(render_neo_metric("Pitch Angle", f"{mesh_data['pitch_deg']:+.1f}°", "Transverse axis tilt", "🧭", "#ffffff"), unsafe_allow_html=True)
    with k4:
        st.markdown(render_neo_metric("Roll Angle", f"{mesh_data['roll_deg']:+.1f}°", "Longitudinal axis tilt", "🔄", "#ffffff"), unsafe_allow_html=True)

    # Row 2: 6-DOF Inertial Dynamics & Subsidence Metrics
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown(render_neo_metric("Angular Velocity (Gyro)", f"{mesh_data['gyro_total_dps']} °/s", f"Gx:{mesh_data['gx_raw']} Gy:{mesh_data['gy_raw']} Gz:{mesh_data['gz_raw']}", "⚡", "#ffffff"), unsafe_allow_html=True)
    with g2:
        st.markdown(render_neo_metric("Raw Accelerometer", f"Z: {mesh_data['az_raw']}", f"Ax:{mesh_data['ax_raw']} | Ay:{mesh_data['ay_raw']}", "📊", "#ffffff"), unsafe_allow_html=True)
    with g3:
        st.markdown(render_neo_metric("Calculated Sag (S_max)", f"{mesh_data['max_settlement_mm']} mm", "Peak trough subsidence", "📉", "#ffffff"), unsafe_allow_html=True)
    with g4:
        risk_short = mesh_data['risk_level'].split(':')[1].split('(')[0].strip() if ':' in mesh_data['risk_level'] else mesh_data['risk_level']
        risk_bg = "#ff4d4d" if "CRITICAL" in mesh_data["risk_level"] else ("#fde047" if "ADVISORY" in mesh_data["risk_level"] else "#4ade80")
        st.markdown(render_neo_metric("Strata Risk Level", risk_short, "Geotechnical DGMS standard", "🛡️", "#ffffff", "STATUS", risk_bg), unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Real-Time Telemetry Trends and Peck's Analytical Curve
    c_left, c_right = st.columns([1.1, 1.3])
    
    with c_left:
        with st.container(border=True):
            st.write("**MPU6050 Live Telemetry Trends**")
            hist = subsidence_mesh.history
            if len(hist["tilt_mm_m"]) > 1:
                df_tilt = pd.DataFrame({
                    "Tilt (mm/m)": hist["tilt_mm_m"],
                    "Pitch (°)": hist["pitch_deg"],
                    "Roll (°)": hist["roll_deg"]
                })
                st.line_chart(df_tilt, height=140)
                
                df_vib = pd.DataFrame({
                    "Vibration (g)": hist["vibration_g"],
                    "Gyro (°/s)": [x / 10.0 for x in hist["gyro_dps"]]
                })
                st.line_chart(df_vib, height=140)
            else:
                st.caption("Waiting for MPU6050 historical points...")

    with c_right:
        with st.container(border=True):
            st.write("**Ground Subsidence Profile (Peck's Analytical Formula)**")
            st.caption("Transverse distance (-150m to +150m) vs vertical settlement (mm)")
            
            prof = mesh_data["profile_curve"]
            df_prof = pd.DataFrame({
                "Distance (m)": prof["x"],
                "Settlement (mm)": prof["settlement"],
                "Tilt (mm/m)": prof["tilt"]
            }).set_index("Distance (m)")
            
            st.line_chart(df_prof, height=220)

    # Cross-link banner to Module 04 Ground Scanner & Dynamic Mesh
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        c_sub1, c_sub2 = st.columns([3.8, 1.2], vertical_alignment="center")
        with c_sub1:
            st.markdown("""
            <div style="display:flex; align-items:center; gap:0.75rem;">
                <span style="font-size:1.6rem;">🗺️</span>
                <div>
                    <div style="font-size:0.95rem; font-weight:900; color:#000000;">Continuous Concession 2D Ground Scanner &amp; Dynamic Risk Mesh</div>
                    <div style="font-size:0.78rem; font-weight:600; color:#4b5563;">Perform full-area spatial scans, partition sectors into danger tiers, and dynamically deploy auxiliary micro-nodes into high-strain zones.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c_sub2:
            if st.button("Open Ground Scanner →", key="btn_sub_to_scanner", use_container_width=True, type="primary"):
                st.session_state["current_view"] = "scanner"
                st.rerun()

    # If Live Stream active, auto-refresh for real-time telemetry
    if live_stream:
        time.sleep(0.18)
        st.rerun()

# -------------------------------------------------------------
# SCREEN 5: OPTION 4 - Ground Scanner & Dynamic Risk Mesh
# -------------------------------------------------------------
# -------------------------------------------------------------
# SCREEN 5: MODULE 04 - Wide Area Satellite & Closed-Loop Adaptive Mesh
# -------------------------------------------------------------
def render_module4_ground_scanner():
    render_header()
    scan_data = ground_scanner.get_scanner_data()
    
    # ── Top Title & Badge Banner ──
    st.markdown("""
    <div style="display:flex; justify-content:space-between; align-items:center; background:#ffffff; border:2.5px solid #000000; border-radius:14px; padding:0.8rem 1.2rem; margin-bottom:0.9rem; box-shadow:4px 4px 0px #000000;">
        <div style="display:flex; align-items:center; gap:0.85rem;">
            <div style="width:44px; height:44px; border-radius:12px; background:#fdba74; border:2.5px solid #000000; box-shadow:2.5px 2.5px 0px #000000; display:flex; align-items:center; justify-content:center; font-size:1.3rem;">🗺️</div>
            <div>
                <div style="font-size:1.1rem; font-weight:900; color:#000000; letter-spacing:-0.02em;">Module 04 &middot; Wide Area Camera / Satellite Scanner &amp; Closed-Loop Mesh</div>
                <div style="font-size:0.76rem; font-weight:700; color:#4b5563;">Wide Area Surveillance ➔ Zone Detection (Safe/Moderate/Danger) ➔ 4-Sensor Telemetry ➔ Anomaly Analysis ➔ Closed-Loop Densification ↺</div>
            </div>
        </div>
        <div style="display:flex; align-items:center; gap:0.5rem;">
            <span style="font-size:0.72rem; font-weight:900; color:#000000; background:#fdba74; border:2px solid #000000; border-radius:8px; padding:0.25rem 0.55rem; box-shadow:2px 2px 0px #000000;">SATELLITE InSAR</span>
            <span style="font-size:0.72rem; font-weight:900; color:#000000; background:#fde047; border:2px solid #000000; border-radius:8px; padding:0.25rem 0.55rem; box-shadow:2px 2px 0px #000000;">CLOSED-LOOP FEEDBACK</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Flowchart Pipeline Stage Ribbon (Direct Architectural Flowchart Representation) ──
    stages = [
        ("1. Wide Area", "🛰️", "Camera / Satellite Simulation"),
        ("2. Zone Detect", "🗺️", "Safe / Moderate / Danger"),
        ("3. Sensor Mesh", "📡", "Few / More / Dense Nodes"),
        ("4. Real-Time", "📈", "Tilt / Disp / Vib / Crack"),
        ("5. Anomaly", "🔍", "Rate Spikes & Rupture"),
        ("6. Risk Score", "⚖️", "0 - 100% Composite"),
        ("7. Alert", "🚨", "Siren & Voice Alarm"),
        ("8. Loop Update ↺", "🔄", "Risk Zone Updated")
    ]
    ribbon_items = []
    for i, (name, icon, sub) in enumerate(stages):
        is_curr = (i == ground_scanner.current_stage_idx)
        bg = "#fef08a" if is_curr else "#f8fafc"
        border = "2.5px solid #000" if is_curr else "1.5px solid #cbd5e1"
        shadow = "2px 2px 0px #000" if is_curr else "none"
        ribbon_items.append(
            f'<div style="flex:1; text-align:center; padding:0.35rem 0.4rem; background:{bg}; border:{border}; border-radius:8px; box-shadow:{shadow};">'
            f'<div style="font-size:0.95rem;">{icon}</div>'
            f'<div style="font-size:0.68rem; font-weight:900; color:#000000; margin-top:2px; white-space:nowrap;">{name}</div>'
            f'<div style="font-size:0.58rem; font-weight:700; color:#6b7280; white-space:nowrap;">{sub}</div>'
            f'</div>'
        )
        if i < len(stages) - 1:
            ribbon_items.append('<div style="font-size:0.85rem; font-weight:900; color:#000000;">➔</div>')

    ribbon_html = (
        '<div style="background:#ffffff; border:2.5px solid #000000; border-radius:12px; padding:0.6rem 0.8rem; margin-bottom:1.0rem; box-shadow:3px 3px 0px #000000; overflow-x:auto;">'
        '<div style="display:flex; align-items:center; justify-content:space-between; gap:0.35rem; min-width:820px;">'
        + "".join(ribbon_items)
        + '</div></div>'
    )
    st.markdown(ribbon_html, unsafe_allow_html=True)

    # ── Interactive Closed-Loop Simulation Toolbar ──
    with st.container(border=True):
        st.markdown("<div style='font-size:0.75rem; font-weight:900; text-transform:uppercase; margin-bottom:0.4rem;'>🕹️ Closed-Loop Simulation &amp; Hazard Injection Console</div>", unsafe_allow_html=True)
        tb_c1, tb_c2, tb_c3, tb_c4, tb_c5 = st.columns([1.2, 1.3, 1.3, 1.3, 1.0], vertical_alignment="center")
        
        with tb_c1:
            if st.button("🟢 Safe Baseline", key="scen_baseline_btn", use_container_width=True, help="All sectors stable within statutory DGMS limits (Few Nodes)."):
                ground_scanner.set_scenario("NORMAL")
                st.toast("✅ Restored Safe Strata Baseline (All Sectors Secure)", icon="🟢")
                st.rerun()

        with tb_c2:
            if st.button("🟡 Inject Fissure Strain", key="scen_moderate_btn", use_container_width=True, help="Triggers tensile crack widening (4.8mm) in Zone-B. Transitions to Moderate (More Nodes)."):
                ground_scanner.set_scenario("MODERATE_STRAIN")
                st.toast("⚠️ Tensile Fissures Active in Zone-B! More Nodes Provisioned.", icon="🟡")
                st.rerun()

        with tb_c3:
            if st.button("🔴 Inject Void Rupture", key="scen_danger_btn", use_container_width=True, help="Triggers deep extraction cavity collapse in Zone-A (Sag: 148mm, Tilt: 12.5mm/m, Crack: 8.6mm). Triggers Evacuation Siren & Dense Nodes!"):
                ground_scanner.set_scenario("CRITICAL_SUBSIDENCE")
                st.toast("🚨 Critical Rupture Hazard in Zone-A! Dense Micro-Mesh Activated.", icon="🔴")
                st.rerun()

        with tb_c4:
            if st.button("⚡ Auto-Deploy Nodes", key="btn_auto_deploy_nodes", use_container_width=True, help="Dynamically densifies danger zones with secondary micro-nodes to eliminate blind spots."):
                n_count, n_ids = ground_scanner.auto_deploy_danger_nodes()
                st.toast(f"⚡ Deployed {n_count} Dense Micro-Nodes into High-Risk Zones!", icon="📡")
                st.rerun()

        with tb_c5:
            if st.button("🔄 Reset", key="btn_reset_scanner_mesh", use_container_width=True, help="Restores mesh back to statutory 4 baseline inclinometers."):
                ground_scanner.reset_to_baseline()
                st.toast("Reset to 4 statutory baseline nodes.", icon="🔄")
                st.rerun()

    # Re-fetch data after any user button action
    scan_data = ground_scanner.get_scanner_data()

    # ── Dynamic Multi-Tier Alert Banner & Audio Action ──
    b_col1, b_col2 = st.columns([4.0, 1.2], vertical_alignment="center")
    with b_col1:
        if scan_data["danger_zones_count"] > 0:
            st.error(f"🚨 **{scan_data['global_status']}**: Sector **{scan_data['highest_risk_zone']}** exhibits **{scan_data['highest_risk_score']}% Composite Risk** with active tensile fissures and rapid displacement! Recommended: Evacuate surface infrastructure and activate high-density micro-mesh.")
        elif scan_data["moderate_zones_count"] > 0:
            st.warning(f"▲ **{scan_data['global_status']}**: Secondary shear movement registered in Sector **{scan_data['highest_risk_zone']}** ({scan_data['highest_risk_score']}% Risk). Additional sentry nodes dispatched.")
        else:
            st.success(f"● **{scan_data['global_status']}**: Geotechnical tilt, sag, vibration, and fissure apertures across all sectors comply with DGMS CMR 2017 standards.")

    with b_col2:
        if st.button("🔊 Sound Evac Alarm", key="btn_sound_geotech_alarm", use_container_width=True):
            alert_manager.trigger_geotechnical_alarm(scan_data["highest_risk_zone"], scan_data["highest_risk_score"])
            st.toast(f"🚨 Siren & Voice Evacuation Broadcast Sent for {scan_data['highest_risk_zone']}!", icon="📢")

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

    # ── Top 4 KPI Metrics ──
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        risk_color = "#ff4d4d" if scan_data["danger_zones_count"] > 0 else ("#fde047" if scan_data["moderate_zones_count"] > 0 else "#4ade80")
        st.markdown(render_neo_metric("Highest Risk Sector", f"{scan_data['highest_risk_zone']} ({scan_data['highest_risk_score']}%)", "Composite 4-sensor geotechnical strain", "⚠️", "#ffffff", "RISK", risk_color), unsafe_allow_html=True)
    with k2:
        z_badge = "DANGER" if scan_data['danger_zones_count']>0 else ("MODERATE" if scan_data['moderate_zones_count']>0 else "ALL SAFE")
        z_badge_col = "#ff4d4d" if scan_data['danger_zones_count']>0 else ("#fde047" if scan_data['moderate_zones_count']>0 else "#4ade80")
        st.markdown(render_neo_metric("Zone Separation", f"{scan_data['danger_zones_count']}D · {scan_data['moderate_zones_count']}M · {scan_data['safe_zones_count']}S", "Safe (Few) · Mod (More) · Danger (Dense)", "🗺️", "#ffffff", z_badge, z_badge_col), unsafe_allow_html=True)
    with k3:
        st.markdown(render_neo_metric("Active Mesh Nodes", f"{scan_data['total_nodes']} Nodes", f"Statutory: {scan_data['base_nodes_count']} | Adaptive: {scan_data['dynamic_nodes_count']}", "📡", "#ffffff", "DENSIFIED" if scan_data['dynamic_nodes_count']>0 else "BASELINE", "#fb923c" if scan_data['dynamic_nodes_count']>0 else "#e2e8f0"), unsafe_allow_html=True)
    with k4:
        st.markdown(render_neo_metric("Early-Warning Lead Time", f"{scan_data['lead_time_mins']} Mins", f"Mesh Coverage: {scan_data['mesh_coverage_pct']}% | Blind spots closed", "⏱️", "#ffffff", "+48M BOOST" if scan_data['dynamic_nodes_count']>0 else "STANDARD", "#4ade80" if scan_data['dynamic_nodes_count']>0 else "#e2e8f0"), unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Main 2-Column Command Workspace ──
    col_map, col_sectors = st.columns([1.35, 1.0])

    with col_map:
        with st.container(border=True):
            v_head_col, v_sel_col = st.columns([1.2, 1.8], vertical_alignment="center")
            with v_head_col:
                st.markdown("<div style='font-size:0.95rem; font-weight:900;'>Wide-Area Remote Sensing</div>", unsafe_allow_html=True)
            with v_sel_col:
                view_mode = st.radio(
                    "Select Remote Sensing Feed",
                    options=["🛰️ Satellite InSAR Fringes", "📷 Aerial Camera Feed", "🗺️ 2D Subsidence Contour"],
                    horizontal=True,
                    label_visibility="collapsed",
                    key="scanner_view_mode_select"
                )

            # Generate and render the selected wide-area perspective
            if "InSAR" in view_mode:
                st.caption("Satellite Differential InSAR (Phase Wrapped Interferogram): Each 2π fringe cycle represents 2.80 cm ground subsidence.")
                insar_bytes = ground_scanner.generate_insar_fringe_view()
                st.image(insar_bytes, use_container_width=True)
            elif "Aerial" in view_mode:
                st.caption("High-Altitude Multi-Spectral Optical Feed: Autonomous aerial/drone concession scan with telemetric HUD crosshairs.")
                sat_bytes = ground_scanner.generate_satellite_simulation_view()
                st.image(sat_bytes, use_container_width=True)
            else:
                st.caption("2D Geotechnical Ground Settlement Field S(x, y) overlaid with active permanent inclinometers & adaptive micro-nodes.")
                map_bytes = ground_scanner.generate_spatial_risk_map()
                st.image(map_bytes, use_container_width=True)

            # Map Legend & Specs
            st.markdown("""
            <div style="display:flex; justify-content:space-between; align-items:center; background:#f8fafc; border:1.5px solid #000; border-radius:8px; padding:0.4rem 0.75rem; margin-top:0.4rem; font-size:0.72rem; font-weight:800;">
                <div style="display:flex; align-items:center; gap:0.5rem;">
                    <span style="color:#2563eb; font-size:0.95rem;">♦</span>
                    <span>Statutory Baseline Nodes (Sparse: 1 per 60m)</span>
                </div>
                <div style="display:flex; align-items:center; gap:0.5rem;">
                    <span style="color:#dc2626; font-size:1.0rem;">★</span>
                    <span>Adaptive Micro-Mesh Nodes (Dense: 1 per 18m)</span>
                </div>
                <div style="color:#4b5563;">
                    Grid: 300m &times; 200m &middot; DGMS Grid
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_sectors:
        with st.container(border=True):
            st.markdown("<div style='font-size:0.95rem; font-weight:900; margin-bottom:0.2rem;'>Zone Detection &amp; Densification Status</div>", unsafe_allow_html=True)
            st.caption("3 Tiers: SAFE (Few Nodes) | MODERATE (More Nodes) | DANGER (Dense Nodes)")

            for zid, zdata in scan_data["zones"].items():
                r_level = zdata["risk_level"]
                if r_level == "DANGER":
                    b_edge = "#b91c1c"
                    b_bg = "#fecaca"
                    badge_title = "DANGER (Dense Nodes)"
                elif r_level == "MODERATE":
                    b_edge = "#d97706"
                    b_bg = "#fef3c7"
                    badge_title = "MODERATE (More Nodes)"
                else:
                    b_edge = "#15803d"
                    b_bg = "#dcfce7"
                    badge_title = "SAFE (Few Nodes)"

                # Anomaly tag badges
                anom_badges = "".join([
                    f"<span style='background:#fee2e2; color:#991b1b; border:1px solid #b91c1c; border-radius:4px; font-size:0.62rem; font-weight:900; padding:0.08rem 0.35rem; margin-right:4px;'>⚠️ {a.replace('_', ' ')}</span>"
                    for a in zdata.get("active_anomalies", [])
                ]) if zdata.get("active_anomalies") else "<span style='color:#16a34a; font-size:0.68rem; font-weight:800;'>✓ No Active Anomalies</span>"

                crack_col = '#dc2626' if zdata['crack_mm'] > 3.0 else '#000000'
                dens_col = '#dc2626' if zdata['node_count'] < zdata['recommended_nodes'] else '#16a34a'
                dens_msg = '⚠️ Deficit: Add Nodes' if zdata['node_count'] < zdata['recommended_nodes'] else '✓ Target Saturated'

                card_html = (
                    f'<div style="background:#ffffff; border:2.5px solid {b_edge}; border-radius:12px; padding:0.65rem 0.85rem; margin-bottom:0.6rem; box-shadow:2.5px 2.5px 0px #000000;">'
                    f'<div style="display:flex; justify-content:space-between; align-items:center;">'
                    f'<span style="font-size:0.86rem; font-weight:900; color:#000000;">{zdata["zone_id"]}: {zdata["name"]}</span>'
                    f'<span style="background:{b_bg}; border:1.5px solid #000; border-radius:6px; padding:0.12rem 0.45rem; font-size:0.66rem; font-weight:900; color:#000;">{badge_title} &middot; {zdata["risk_score"]}%</span>'
                    f'</div>'
                    f'<div style="display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:0.35rem; margin:0.45rem 0; font-size:0.72rem; font-weight:800; background:#f8fafc; border:1.5px solid #000; border-radius:6px; padding:0.35rem 0.5rem;">'
                    f'<div><span style="color:#64748b; display:block; font-size:0.60rem;">TILT</span><b>{zdata["avg_tilt_mm_m"]} mm/m</b></div>'
                    f'<div><span style="color:#64748b; display:block; font-size:0.60rem;">SAG (S)</span><b>{zdata["max_settlement_mm"]} mm</b></div>'
                    f'<div><span style="color:#64748b; display:block; font-size:0.60rem;">TREMOR</span><b>{zdata["vibration_g"]} g</b></div>'
                    f'<div><span style="color:#64748b; display:block; font-size:0.60rem;">CRACK</span><b style="color:{crack_col};">{zdata["crack_mm"]} mm</b></div>'
                    f'</div>'
                    f'<div style="margin-bottom:0.35rem;">{anom_badges}</div>'
                    f'<div style="display:flex; justify-content:space-between; align-items:center; font-size:0.72rem; color:#4b5563; font-weight:800;">'
                    f'<span>Allocated Nodes: <b>{zdata["node_count"]}</b> (Target: <b>{zdata["recommended_nodes"]}</b>)</span>'
                    f'<span style="color:{dens_col}; font-weight:900;">{dens_msg}</span>'
                    f'</div>'
                    f'</div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

                # Manual deployment button per zone
                _, b_col = st.columns([2.8, 1.2])
                with b_col:
                    if st.button(f"+ Add Node", key=f"btn_add_node_{zid}", use_container_width=True):
                        new_node_id = ground_scanner.deploy_manual_node(zid, "CRACK_METER")
                        st.toast(f"✅ Sentry Node {new_node_id} deployed into {zid}!", icon="📍")
                        st.rerun()

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # ── Sensor Network & Real-Time Data (4 Sensors: Tilt, Displacement, Vibration, Crack) ──
    with st.container(border=True):
        st.markdown("<div style='font-size:1.0rem; font-weight:900;'>Sensor Network &amp; 4-Parameter Real-Time Telemetry Stream</div>", unsafe_allow_html=True)
        st.caption("Live streaming 4-sensor arrays: Inclinometer Tilt (mm/m), Settlement Sag (mm), Micro-Seismic Vibration (g), and Surface Crack Meter (mm).")

        all_node_ids = [n["node_id"] for n in scan_data["nodes"]]
        sel_node_id = st.selectbox(
            "Select Sensor Node for Real-Time Telemetry Inspection",
            options=all_node_ids,
            format_func=lambda nid: f"{nid} - {next((n['name'] for n in scan_data['nodes'] if n['node_id']==nid), nid)} ({next((n['zone_id'] for n in scan_data['nodes'] if n['node_id']==nid), '')})",
            key="scanner_node_inspect_select"
        )
        sel_node = next((n for n in scan_data["nodes"] if n["node_id"] == sel_node_id), scan_data["nodes"][0])

        # 4 Mini Metric Cards for Selected Node
        n_m1, n_m2, n_m3, n_m4 = st.columns(4)
        with n_m1:
            st.markdown(render_neo_metric("Inclinometer Tilt", f"{sel_node['tilt_mm_m']} mm/m", "Limit: 8.0 mm/m (DGMS)", "📐", "#ffffff", "TILT", "#e0f2fe"), unsafe_allow_html=True)
        with n_m2:
            st.markdown(render_neo_metric("Settlement Sag", f"{sel_node['settlement_mm']} mm", "Limit: 80.0 mm (CMR)", "📉", "#ffffff", "DISP", "#fef3c7"), unsafe_allow_html=True)
        with n_m3:
            st.markdown(render_neo_metric("Micro-Tremor", f"{sel_node['vibration_g']} g", "Limit: 0.25 g", "📳", "#ffffff", "VIB", "#dcfce7"), unsafe_allow_html=True)
        with n_m4:
            crack_bg = "#fee2e2" if sel_node['crack_mm'] >= 4.0 else "#f4f4f5"
            st.markdown(render_neo_metric("Crack Aperture", f"{sel_node['crack_mm']} mm", f"Rate: {sel_node['crack_rate_mm_min']} mm/min", "⚡", "#ffffff", "CRACK", crack_bg), unsafe_allow_html=True)

        # Real-time multi-metric line chart
        if len(sel_node.get("history_tilt", [])) > 0:
            df_hist = pd.DataFrame({
                "Tilt (mm/m)": sel_node["history_tilt"],
                "Displacement (mm)": sel_node["history_disp"],
                "Tremor (g x100)": [v * 100.0 for v in sel_node["history_vib"]],
                "Crack Aperture (mm)": sel_node["history_crack"]
            })
            st.markdown("<div style='font-size:0.75rem; font-weight:800; color:#4b5563; margin-top:0.4rem; margin-bottom:0.2rem;'>LIVE ROLLING TIME-SERIES (LAST 30 DATA PACKETS &bull; 5 HZ TELEMETRY STREAM)</div>", unsafe_allow_html=True)
            st.line_chart(df_hist, use_container_width=True, height=220)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # ── Detailed Inspector Tabs ──
    tab_nodes, tab_standards, tab_audit = st.tabs([
        "📡 Active Sensor Node Roster",
        "📋 DGMS Strata Guidelines & Anomaly Rules",
        "📜 Closed-Loop Dispatch & Audit Trail"
    ])

    with tab_nodes:
        with st.container(border=True):
            st.write("**Current Active Geotechnical Sensor Node Roster**")
            st.caption("Live parameters transmitted from base statutory inclinometers and dynamically provisioned danger-zone micro-nodes.")
            
            nodes_list = scan_data["nodes"]
            if nodes_list:
                df_nodes = pd.DataFrame(nodes_list)
                df_nodes = df_nodes[[
                    "node_id", "name", "zone_id", "node_type", "is_dynamic",
                    "x", "y", "tilt_mm_m", "settlement_mm", "vibration_g", "crack_mm",
                    "battery_pct", "rssi_dbm", "status", "deployed_at"
                ]]
                df_nodes.columns = [
                    "Node ID", "Sensor Tag", "Sector", "Type", "Dynamic?",
                    "X (m)", "Y (m)", "Tilt (mm/m)", "Subsidence (mm)", "Tremor (g)", "Crack (mm)",
                    "Battery (%)", "RSSI (dBm)", "Status", "Deployment"
                ]
                st.dataframe(df_nodes, use_container_width=True, hide_index=True)
            else:
                st.caption("No nodes active.")

    with tab_standards:
        st.markdown("""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.85rem;">
            <div style="background:#ffffff; border:2.5px solid #000; border-radius:12px; padding:1.0rem; box-shadow:3.5px 3.5px 0px #000;">
                <div style="font-size:0.95rem; font-weight:900; color:#000;">
                    🛡️ DGMS Statutory Strata Management Plan (SMP)
                </div>
                <div style="font-size:0.78rem; font-weight:600; color:#374151; margin-top:0.45rem; line-height:1.5;">
                    <b>Coal Mines Regulations (CMR) 2017 &middot; Regulation 123 &amp; 124:</b><br>
                    • Continuous strata deformation monitoring is statutory over extracted extraction panels (Goaf).<br>
                    • <b>Critical Tilt Threshold:</b> Any ground inclination exceeding <b>8.0 mm/m</b> triggers immediate red alert and surface infrastructure evacuation.<br>
                    • <b>Surface Crack Aperture:</b> Tensile fissure opening &gt; <b>6.0 mm</b> or expansion velocity &gt; <b>0.4 mm/min</b> indicates progressive strata shear rupture.<br>
                    • <b>Dynamic Tremors:</b> Micro-seismic vibrations &gt; <b>0.25 g</b> indicate strata shearing prior to breach.
                </div>
            </div>
            <div style="background:#ffffff; border:2.5px solid #000; border-radius:12px; padding:1.0rem; box-shadow:3.5px 3.5px 0px #000;">
                <div style="font-size:0.95rem; font-weight:900; color:#000;">
                    ⚡ Closed-Loop Feedback &amp; Dynamic Densification
                </div>
                <div style="font-size:0.78rem; font-weight:600; color:#374151; margin-top:0.45rem; line-height:1.5;">
                    <b>Why adaptive closed-loop density scaling?</b><br>
                    • <b>Blind-Spot Elimination:</b> Single baseline sensors (e.g. 1 per 60m) miss localized tensile shear fissures developing between nodes.<br>
                    • <b>Feedback Loop Rebalancing:</b> Anomaly detection and risk scoring feed back into zone classification; when risk drops (via backfilling), node density rebalances.<br>
                    • <b>Lead-Time Multiplication:</b> Dense micro-nodes extend warning lead time from <b>~12 minutes to 60+ minutes</b>, enabling orderly workforce evacuation.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab_audit:
        with st.container(border=True):
            st.write("**Ground Scanner Event & Dispatch Audit Log**")
            logs = scan_data["audit_log"]
            for l in logs:
                badge_bg = "#ff4d4d" if l.get("type") == "ACTION" else ("#facc15" if l.get("type") == "MANUAL" else "#e2e8f0")
                badge_lbl = l.get("type", "INFO")
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:0.6rem; padding:0.35rem 0.6rem; border-bottom:1px solid #e5e7eb; font-size:0.78rem;">
                    <span style="font-family:'Space Grotesk', monospace; font-weight:800; color:#6b7280;">[{l['time']}]</span>
                    <span style="background:{badge_bg}; border:1px solid #000; border-radius:4px; padding:0.08rem 0.35rem; font-size:0.65rem; font-weight:900; color:#000;">{badge_lbl}</span>
                    <span style="font-weight:700; color:#111827;">{l['event']}</span>
                </div>
                """, unsafe_allow_html=True)


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
        elif view == "scanner":
            render_module4_ground_scanner()
        else:
            render_hub()

if __name__ == "__main__":
    main()
