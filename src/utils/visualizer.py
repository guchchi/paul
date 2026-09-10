"""
Presentation-Grade HUD Visualizer for SIH Evaluation
Overlays real-time bounding boxes, danger status pills, FPS telemetry,
and active feature toggle indicators.
"""
import time
from typing import List, Dict, Any
import cv2
import numpy as np

class HUDVisualizer:
    def __init__(self):
        # Color definitions (BGR format)
        self.COLOR_FIRE = (0, 69, 255)       # Vibrant Red-Orange
        self.COLOR_SMOKE = (220, 180, 70)    # Cyan / Cool Light Gray
        self.COLOR_SAFE = (50, 205, 50)      # Forest Green
        self.COLOR_WARN = (0, 215, 255)      # Amber / Yellow-Gold
        self.COLOR_ALERT = (0, 0, 255)       # Emergency Red
        self.COLOR_DARK_BG = (20, 20, 20)
        self.COLOR_WHITE = (245, 245, 245)
        
    def draw_hud(
        self,
        frame: cv2.Mat,
        system_state: str = "SECURE",
        detections: List[Dict[str, Any]] = None,
        fps: float = 0.0,
        toggles: Dict[str, bool] = None,
        engine_mode: str = "HYBRID",
        active_toggles: Dict[str, bool] = None,
        **kwargs
    ) -> cv2.Mat:
        """
        Draws complete HUD overlay on frame.
        """
        if frame is None:
            return frame

        if toggles is None:
            toggles = active_toggles if active_toggles is not None else {}
        if detections is None:
            detections = []
            
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # 1. Pulsing Alarm Perimeter Vignette if state is ALARM
        if system_state == "ALARM":
            border_thickness = 8
            # Pulsing intensity based on clock
            pulse = int((np.sin(time.time() * 8) + 1.0) * 0.5 * 100) + 155
            alert_color = (0, 0, pulse)
            cv2.rectangle(annotated, (0, 0), (w, h), alert_color, border_thickness)
            
        # 2. Draw Target Bounding Boxes
        for det in detections:
            x, y, bw, bh = det["box"]
            label = det.get("label", "fire")
            conf = det.get("confidence", 0.5)
            status = det.get("status", "VERIFYING")
            
            # Select color based on object label
            box_color = self.COLOR_FIRE if label == "fire" else self.COLOR_SMOKE
            
            # Bounding box
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), box_color, 2)
            
            # Label tag with background pill
            tag_text = f"{label.upper()} {int(conf * 100)}% [{status}]"
            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (x, max(0, y - th - 8)), (x + tw + 8, y), box_color, -1)
            cv2.putText(
                annotated,
                tag_text,
                (x + 4, max(th + 2, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255) if label == "fire" else (20, 20, 20),
                1,
                cv2.LINE_AA
            )
            
        # 3. Top HUD Status Bar (Semi-transparent overlay)
        top_bar_height = 42
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, top_bar_height), self.COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.82, annotated, 0.18, 0, annotated)
        
        # Determine Status Pill Color and Text
        if system_state == "ALARM":
            status_color = self.COLOR_ALERT
            status_str = "CRITICAL: FIRE DETECTED"
        elif system_state == "VERIFYING":
            status_color = self.COLOR_WARN
            status_str = "ANALYZING COMBUSTION"
        else:
            status_color = self.COLOR_SAFE
            status_str = "ALL ZONES SECURE"
            
        # Draw status pill indicator circle
        cv2.circle(annotated, (18, 21), 7, status_color, -1)
        cv2.putText(
            annotated,
            status_str,
            (32, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            status_color,
            2,
            cv2.LINE_AA
        )
        
        # Telemetry info (FPS + Engine) on the right side of top bar
        telemetry_str = f"FPS: {fps:.1f} | Engine: {engine_mode}"
        (tw, _), _ = cv2.getTextSize(telemetry_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(
            annotated,
            telemetry_str,
            (w - tw - 15, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            self.COLOR_WHITE,
            1,
            cv2.LINE_AA
        )
        
        # 4. Bottom Feature Toggles Bar
        bot_bar_height = 30
        overlay_bot = annotated.copy()
        cv2.rectangle(overlay_bot, (0, h - bot_bar_height), (w, h), self.COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay_bot, 0.85, annotated, 0.15, 0, annotated)
        
        # Format toggle badges: Smoke, Audio, Voice, Snapshots
        badge_x = 12
        badges = [
            ("SMOKE [M]", toggles.get("smoke", True)),
            ("SIREN [A]", toggles.get("audio", True)),
            ("VOICE [V]", toggles.get("voice", True)),
            ("SNAPS [S]", toggles.get("snapshots", True)),
        ]
        
        for name, active in badges:
            text = f"{name}: {'ON' if active else 'OFF'}"
            color = self.COLOR_SAFE if active else (120, 120, 120)
            cv2.putText(
                annotated,
                text,
                (badge_x, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                color,
                1,
                cv2.LINE_AA
            )
            (bw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            badge_x += bw + 18
            
        # Hotkey guide hint on right of bottom bar
        quit_hint = "Quit: [Q]"
        (qw, _), _ = cv2.getTextSize(quit_hint, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        cv2.putText(
            annotated,
            quit_hint,
            (w - qw - 12, h - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (170, 170, 170),
            1,
            cv2.LINE_AA
        )
        
        return annotated

# Global visualizer instance
hud_visualizer = HUDVisualizer()
