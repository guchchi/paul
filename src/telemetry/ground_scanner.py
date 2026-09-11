"""
Overground Ground Area Scanner & Dynamic Risk-Based Sensor Mesh Module.
Provides 2D spatial strata subsidence modeling, zone risk classification (DGMS norms),
and an adaptive node allocation engine that provisions high-density sensor nodes
into identified danger zones to eliminate coverage blind spots.
"""
import io
import time
import math
import random
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend safe for web dashboards
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap


@dataclass
class GroundNode:
    """Represents a surface or borehole geotechnical telemetry node."""
    node_id: str
    name: str
    node_type: str        # 'BASE_INCLINOMETER', 'ADAPTIVE_TILTMETER', 'MICRO_SEISMIC', 'STRAIN_EXTENSOMETER'
    x: float              # Coordinate in meters (-150 to 150)
    y: float              # Coordinate in meters (-100 to 100)
    zone_id: str
    is_dynamic: bool = False
    status: str = "ONLINE"
    tilt_mm_m: float = 0.0
    settlement_mm: float = 0.0
    vibration_g: float = 0.008
    battery_pct: int = 95
    rssi_dbm: int = -64
    deployed_at: str = "Baseline"


@dataclass
class GroundZone:
    """Represents an operational mine surface/overburden concession sector."""
    zone_id: str
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    description: str
    max_settlement_mm: float = 0.0
    avg_tilt_mm_m: float = 0.0
    vibration_g: float = 0.010
    void_index: float = 0.05       # 0.0 to 1.0 (underground cavity / fracture likelihood)
    risk_score: float = 10.0       # 0 to 100%
    risk_level: str = "SAFE_STABLE"  # 'SAFE_STABLE', 'ADVISORY_STRAIN', 'CRITICAL_DANGER'
    assigned_nodes: List[str] = field(default_factory=list)
    recommended_nodes: int = 1


class GroundScanner:
    """
    Continuous Ground Area Scanner & Adaptive Geotechnical Mesh Engine.
    Computes 2D subsidence basins, separates mine sectors into risk tiers,
    and dynamically allocates secondary sensor nodes into critical danger zones.
    """

    def __init__(self):
        # Concession spatial bounds (in meters)
        self.x_bounds = (-150.0, 150.0)
        self.y_bounds = (-100.0, 100.0)

        # Active survey scenario: 'NORMAL', 'STRATA_STRAIN', 'CRITICAL_SUBSIDENCE'
        self.scenario = "NORMAL"

        # Scan state
        self.last_scan_time = time.time()
        self.scan_count = 1
        self.is_scanning = False

        # Concession Sectors
        self.zones: Dict[str, GroundZone] = {
            "ZONE-A": GroundZone(
                zone_id="ZONE-A",
                name="Goaf Extraction Basin",
                x_min=-150.0, x_max=-20.0,
                y_min=-100.0, y_max=20.0,
                description="Active extraction goaf area with high void collapse potential",
                assigned_nodes=["NODE-01"]
            ),
            "ZONE-B": GroundZone(
                zone_id="ZONE-B",
                name="Rib Pillar Shear Margin",
                x_min=-20.0, x_max=70.0,
                y_min=-100.0, y_max=20.0,
                description="Tensile strain concentration & pillar boundary cleavage zone",
                assigned_nodes=["NODE-02"]
            ),
            "ZONE-C": GroundZone(
                zone_id="ZONE-C",
                name="Haulage Drift & Incline Corridor",
                x_min=-150.0, x_max=150.0,
                y_min=20.0, y_max=100.0,
                description="Main underground conveyor roadway & transport shaft entry",
                assigned_nodes=["NODE-03"]
            ),
            "ZONE-D": GroundZone(
                zone_id="ZONE-D",
                name="Shaft Infrastructure Buffer",
                x_min=70.0, x_max=150.0,
                y_min=-100.0, y_max=20.0,
                description="Statutory shaft protection pillar & surface winding house",
                assigned_nodes=["NODE-04"]
            ),
        }

        # Active telemetry nodes (Base + Dynamically provisioned)
        self.nodes: Dict[str, GroundNode] = {}
        self.reset_to_baseline()

        # Audit history log
        self.audit_log: List[Dict[str, str]] = [
            {"time": time.strftime("%H:%M:%S"), "event": "Ground Scanner initialized. Baseline mesh active (4 nodes).", "type": "INFO"}
        ]

        # Precompute initial scan
        self.run_ground_scan(force=True)

    def reset_to_baseline(self):
        """Restores the sensor mesh back to the default 4 statutory base nodes."""
        self.nodes = {
            "NODE-01": GroundNode(
                node_id="NODE-01",
                name="Goaf Center (MPU6050 Primary)",
                node_type="BASE_INCLINOMETER",
                x=-80.0, y=-40.0,
                zone_id="ZONE-A",
                is_dynamic=False,
                status="ONLINE",
                battery_pct=96,
                rssi_dbm=-62,
                deployed_at="Permanent Benchmark"
            ),
            "NODE-02": GroundNode(
                node_id="NODE-02",
                name="Shear Boundary Extensometer",
                node_type="BASE_INCLINOMETER",
                x=25.0, y=-40.0,
                zone_id="ZONE-B",
                is_dynamic=False,
                status="ONLINE",
                battery_pct=92,
                rssi_dbm=-68,
                deployed_at="Permanent Benchmark"
            ),
            "NODE-03": GroundNode(
                node_id="NODE-03",
                name="Haulage Drift Acoustic Geophone",
                node_type="MICRO_SEISMIC",
                x=0.0, y=60.0,
                zone_id="ZONE-C",
                is_dynamic=False,
                status="ONLINE",
                battery_pct=88,
                rssi_dbm=-71,
                deployed_at="Permanent Benchmark"
            ),
            "NODE-04": GroundNode(
                node_id="NODE-04",
                name="Shaft Pillar Reference Tiltmeter",
                node_type="BASE_INCLINOMETER",
                x=110.0, y=-40.0,
                zone_id="ZONE-D",
                is_dynamic=False,
                status="ONLINE",
                battery_pct=98,
                rssi_dbm=-59,
                deployed_at="Permanent Benchmark"
            )
        }
        # Re-assign nodes to zones
        for z in self.zones.values():
            z.assigned_nodes = [nid for nid, n in self.nodes.items() if n.zone_id == z.zone_id]

    def set_scenario(self, scenario: str):
        """Set simulation scenario: NORMAL, STRATA_STRAIN, or CRITICAL_SUBSIDENCE."""
        self.scenario = scenario
        self.run_ground_scan(force=True)

    def _compute_2d_subsidence_field(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Calculates 2D continuous subsidence settlement field S(x, y) using
        Peck's 2D Gaussian basin superposition with geological fault vectors.
        """
        if self.scenario == "CRITICAL_SUBSIDENCE":
            s_peak = 152.0  # mm
            center_x, center_y = -85.0, -42.0
            inflection_ix, inflection_iy = 55.0, 42.0
            secondary_s = 65.0
            sec_x, sec_y = 15.0, -35.0
        elif self.scenario == "STRATA_STRAIN":
            s_peak = 52.0   # mm
            center_x, center_y = -80.0, -40.0
            inflection_ix, inflection_iy = 60.0, 50.0
            secondary_s = 28.0
            sec_x, sec_y = 20.0, -35.0
        else:  # NORMAL
            s_peak = 8.5    # mm
            center_x, center_y = -80.0, -40.0
            inflection_ix, inflection_iy = 70.0, 60.0
            secondary_s = 4.0
            sec_x, sec_y = 25.0, -35.0

        # Primary Goaf subsidence trough
        r2_primary = ((X - center_x) ** 2) / (2 * (inflection_ix ** 2)) + ((Y - center_y) ** 2) / (2 * (inflection_iy ** 2))
        trough_primary = s_peak * np.exp(-r2_primary)

        # Secondary pillar shear subsidence
        r2_sec = ((X - sec_x) ** 2) / (2 * (40.0 ** 2)) + ((Y - sec_y) ** 2) / (2 * (35.0 ** 2))
        trough_secondary = secondary_s * np.exp(-r2_sec)

        # Combined continuous field
        field = trough_primary + trough_secondary
        return field

    def run_ground_scan(self, force: bool = False) -> Dict[str, Any]:
        """
        Executes a spatial ground area scan across all concession zones.
        Evaluates subsidence magnitude, tilt gradients, micro-vibrations, and void anomalies.
        Classifies each zone into SAFE_STABLE, ADVISORY_STRAIN, or CRITICAL_DANGER.
        """
        self.is_scanning = True
        self.scan_count += 1
        self.last_scan_time = time.time()

        # Grid resolution for analysis
        x_pts = np.linspace(self.x_bounds[0], self.x_bounds[1], 61)
        y_pts = np.linspace(self.y_bounds[0], self.y_bounds[1], 41)
        X, Y = np.meshgrid(x_pts, y_pts)
        S_field = self._compute_2d_subsidence_field(X, Y)

        # Compute gradient (tilt magnitude in mm/m)
        grad_y, grad_x = np.gradient(S_field, y_pts[1] - y_pts[0], x_pts[1] - x_pts[0])
        Tilt_field = np.sqrt(grad_x ** 2 + grad_y ** 2) * 10.0  # Scale to mm/m equivalent

        # Evaluate each zone's spatial metrics
        danger_zones_count = 0
        for zid, zone in self.zones.items():
            mask = (X >= zone.x_min) & (X <= zone.x_max) & (Y >= zone.y_min) & (Y <= zone.y_max)
            zone_subsidence = S_field[mask]
            zone_tilt = Tilt_field[mask]

            max_sub = float(np.max(zone_subsidence)) if len(zone_subsidence) > 0 else 0.0
            avg_tilt = float(np.mean(zone_tilt)) if len(zone_tilt) > 0 else 0.0

            # Estimate vibration and void cavity index based on scenario and subsidence
            if zid == "ZONE-A":
                vib = 0.42 if self.scenario == "CRITICAL_SUBSIDENCE" else (0.12 if self.scenario == "STRATA_STRAIN" else 0.012)
                void_idx = 0.88 if self.scenario == "CRITICAL_SUBSIDENCE" else (0.45 if self.scenario == "STRATA_STRAIN" else 0.08)
            elif zid == "ZONE-B":
                vib = 0.28 if self.scenario == "CRITICAL_SUBSIDENCE" else (0.09 if self.scenario == "STRATA_STRAIN" else 0.009)
                void_idx = 0.65 if self.scenario == "CRITICAL_SUBSIDENCE" else (0.32 if self.scenario == "STRATA_STRAIN" else 0.05)
            elif zid == "ZONE-C":
                vib = 0.08 if self.scenario == "CRITICAL_SUBSIDENCE" else 0.02
                void_idx = 0.20 if self.scenario == "CRITICAL_SUBSIDENCE" else 0.04
            else:  # ZONE-D
                vib = 0.015 if self.scenario == "CRITICAL_SUBSIDENCE" else 0.005
                void_idx = 0.05

            # Composite DGMS Risk Score (0-100%):
            # 40% Subsidence depth (rel to 100mm threshold)
            # 30% Tilt gradient (rel to 10 mm/m DGMS threshold)
            # 15% Dynamic tremor vibration (rel to 0.3g threshold)
            # 15% Void cavity index
            sub_factor = min(1.0, max_sub / 120.0)
            tilt_factor = min(1.0, avg_tilt / 8.0)
            vib_factor = min(1.0, vib / 0.30)
            void_factor = void_idx

            risk_score = round((0.40 * sub_factor + 0.30 * tilt_factor + 0.15 * vib_factor + 0.15 * void_factor) * 100.0, 1)

            # Assign DGMS Risk Category
            if risk_score >= 68.0 or max_sub > 90.0 or avg_tilt > 7.0:
                risk_level = "CRITICAL_DANGER"
                rec_nodes = 4  # High-density mesh required
                danger_zones_count += 1
            elif risk_score >= 38.0 or max_sub > 30.0 or avg_tilt > 3.0:
                risk_level = "ADVISORY_STRAIN"
                rec_nodes = 2
            else:
                risk_level = "SAFE_STABLE"
                rec_nodes = 1

            zone.max_settlement_mm = round(max_sub, 1)
            zone.avg_tilt_mm_m = round(avg_tilt, 2)
            zone.vibration_g = round(vib, 3)
            zone.void_index = round(void_idx, 2)
            zone.risk_score = risk_score
            zone.risk_level = risk_level
            zone.recommended_nodes = rec_nodes

        # Update telemetry for all placed nodes according to their coordinates
        for node in self.nodes.values():
            if node.zone_id == "ZONE-A":
                node.tilt_mm_m = round(12.4 if self.scenario == "CRITICAL_SUBSIDENCE" else (4.6 if self.scenario == "STRATA_STRAIN" else 0.8) + random.uniform(-0.1, 0.1), 2)
                node.vibration_g = round(0.38 if self.scenario == "CRITICAL_SUBSIDENCE" else (0.11 if self.scenario == "STRATA_STRAIN" else 0.012), 3)
                node.settlement_mm = round(145.0 if self.scenario == "CRITICAL_SUBSIDENCE" else (48.0 if self.scenario == "STRATA_STRAIN" else 8.0), 1)
            elif node.zone_id == "ZONE-B":
                node.tilt_mm_m = round(7.8 if self.scenario == "CRITICAL_SUBSIDENCE" else (3.2 if self.scenario == "STRATA_STRAIN" else 0.6), 2)
                node.vibration_g = round(0.24 if self.scenario == "CRITICAL_SUBSIDENCE" else (0.07 if self.scenario == "STRATA_STRAIN" else 0.009), 3)
                node.settlement_mm = round(65.0 if self.scenario == "CRITICAL_SUBSIDENCE" else (24.0 if self.scenario == "STRATA_STRAIN" else 4.5), 1)
            else:
                node.tilt_mm_m = round(1.2 if self.scenario == "CRITICAL_SUBSIDENCE" else 0.4, 2)
                node.vibration_g = round(0.04 if self.scenario == "CRITICAL_SUBSIDENCE" else 0.007, 3)
                node.settlement_mm = round(12.0 if self.scenario == "CRITICAL_SUBSIDENCE" else 2.0, 1)

        self.is_scanning = False
        return self.get_scanner_data()

    def get_danger_zones(self) -> List[GroundZone]:
        """Returns list of zones currently marked as CRITICAL_DANGER."""
        return [z for z in self.zones.values() if z.risk_level == "CRITICAL_DANGER"]

    def auto_deploy_danger_nodes(self) -> Tuple[int, List[str]]:
        """
        Dynamic Node Deployment Engine:
        Automatically provisions dense telemetry nodes into identified CRITICAL_DANGER zones.
        Calculates optimal spatial coordinates to eliminate coverage blind spots.
        """
        danger_zones = self.get_danger_zones()
        if not danger_zones:
            # Fallback: if no critical zone, check advisory zones
            advisory = [z for z in self.zones.values() if z.risk_level == "ADVISORY_STRAIN"]
            target_zones = advisory if advisory else [self.zones["ZONE-A"]]
        else:
            target_zones = danger_zones

        deployed_ids = []
        for zone in target_zones:
            current_count = len(zone.assigned_nodes)
            needed = max(0, zone.recommended_nodes - current_count)
            if needed == 0:
                # If already at baseline recommended count, add at least 2 dense micro-nodes for high resolution
                needed = 2

            # Candidate strategic coordinates in the zone (perimeter shear & center basin points)
            offsets = [
                (-35.0, 15.0, "High-Gradient North Flank", "ADAPTIVE_TILTMETER"),
                (25.0, -18.0, "East Tensile Crack Point", "STRAIN_EXTENSOMETER"),
                (-20.0, -32.0, "South Goaf Margin Geophone", "MICRO_SEISMIC"),
                (10.0, 20.0, "Subsidence Trough Crest Inclinometer", "ADAPTIVE_TILTMETER"),
            ]

            for i in range(needed):
                idx = len(self.nodes) + 1
                new_id = f"DN-{idx:02d}"
                if new_id in self.nodes:
                    new_id = f"DN-{idx + 10:02d}"

                # Calculate placement coords inside zone bounds
                center_x = (zone.x_min + zone.x_max) / 2.0
                center_y = (zone.y_min + zone.y_max) / 2.0
                off_x, off_y, loc_label, ntype = offsets[i % len(offsets)]
                target_x = max(zone.x_min + 5.0, min(zone.x_max - 5.0, center_x + off_x))
                target_y = max(zone.y_min + 5.0, min(zone.y_max - 5.0, center_y + off_y))

                new_node = GroundNode(
                    node_id=new_id,
                    name=f"{zone.name} - {loc_label}",
                    node_type=ntype,
                    x=round(target_x, 1),
                    y=round(target_y, 1),
                    zone_id=zone.zone_id,
                    is_dynamic=True,
                    status="DEPLOYED",
                    battery_pct=random.randint(92, 99),
                    rssi_dbm=random.randint(-65, -55),
                    deployed_at=time.strftime("%H:%M:%S")
                )

                self.nodes[new_id] = new_node
                zone.assigned_nodes.append(new_id)
                deployed_ids.append(new_id)

                self.audit_log.insert(0, {
                    "time": time.strftime("%H:%M:%S"),
                    "event": f"⚡ Dynamic Node {new_id} ({ntype}) deployed in {zone.zone_id} at ({target_x:.1f}m, {target_y:.1f}m).",
                    "type": "ACTION"
                })

        # Recalculate readings for new nodes
        self.run_ground_scan(force=True)
        return len(deployed_ids), deployed_ids

    def deploy_manual_node(self, zone_id: str, node_type: str = "ADAPTIVE_TILTMETER") -> Optional[str]:
        """Manually provisions an extra node into a chosen zone."""
        if zone_id not in self.zones:
            return None

        zone = self.zones[zone_id]
        idx = len(self.nodes) + 1
        new_id = f"DN-{idx:02d}"
        if new_id in self.nodes:
            new_id = f"DN-{idx + 10:02d}"

        # Jitter position inside zone
        x = round(random.uniform(zone.x_min + 10.0, zone.x_max - 10.0), 1)
        y = round(random.uniform(zone.y_min + 10.0, zone.y_max - 10.0), 1)

        new_node = GroundNode(
            node_id=new_id,
            name=f"{zone.name} Manual Sentry",
            node_type=node_type,
            x=x, y=y,
            zone_id=zone_id,
            is_dynamic=True,
            status="DEPLOYED",
            battery_pct=99,
            rssi_dbm=-58,
            deployed_at=time.strftime("%H:%M:%S")
        )
        self.nodes[new_id] = new_node
        zone.assigned_nodes.append(new_id)

        self.audit_log.insert(0, {
            "time": time.strftime("%H:%M:%S"),
            "event": f"👤 Operator deployed node {new_id} ({node_type}) into {zone_id} at ({x}m, {y}m).",
            "type": "MANUAL"
        })

        self.run_ground_scan(force=True)
        return new_id

    def generate_spatial_risk_map(self) -> bytes:
        """
        Renders a high-resolution 2D spatial contour & telemetry mesh map using Matplotlib.
        Visualizes continuous subsidence gradient, zone risk boundaries, base nodes,
        and dynamically deployed danger-zone sensor nodes.
        Returns PNG image bytes.
        """
        fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=130)
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#f8fafc')

        # Generate coordinate mesh
        x_pts = np.linspace(self.x_bounds[0], self.x_bounds[1], 120)
        y_pts = np.linspace(self.y_bounds[0], self.y_bounds[1], 80)
        X, Y = np.meshgrid(x_pts, y_pts)
        S_field = self._compute_2d_subsidence_field(X, Y)

        # Custom high-contrast SCADA geotechnical colormap:
        # Stable Teal/Green -> Advisory Amber/Gold -> Danger Crimson
        colors = [
            (0.12, 0.72, 0.53),  # Safe Emerald
            (0.35, 0.80, 0.35),  # Lime Green
            (0.98, 0.85, 0.25),  # Advisory Amber
            (0.98, 0.55, 0.15),  # Warning Orange
            (0.92, 0.18, 0.22),  # Critical Crimson
            (0.65, 0.05, 0.12),  # Deep Rupture Red
        ]
        cmap = LinearSegmentedColormap.from_list("mine_risk_cmap", colors, N=100)

        # Filled contours for continuous subsidence settlement
        levels = np.linspace(0, max(160.0, float(np.max(S_field)) + 10.0), 32)
        contour_plot = ax.contourf(X, Y, S_field, levels=levels, cmap=cmap, alpha=0.88)

        # Draw contour lines
        line_levels = [10.0, 30.0, 60.0, 90.0, 120.0, 140.0]
        line_levels = [l for l in line_levels if l <= np.max(S_field)]
        if line_levels:
            cs = ax.contour(X, Y, S_field, levels=line_levels, colors='#1e293b', linewidths=1.1, linestyles='--')
            ax.clabel(cs, inline=True, fmt='%1.0f mm', fontsize=8, colors='#000000')

        # Draw Zone Boundaries & Neo-Brutalist Badges
        zone_styles = {
            "CRITICAL_DANGER": {"edge": "#b91c1c", "bg": "#fecaca", "badge": "CRITICAL DANGER", "badge_col": "#7f1d1d"},
            "ADVISORY_STRAIN": {"edge": "#d97706", "bg": "#fef3c7", "badge": "ACTIVE STRAIN", "badge_col": "#78350f"},
            "SAFE_STABLE":     {"edge": "#15803d", "bg": "#dcfce7", "badge": "STABLE SECURE", "badge_col": "#14532d"},
        }

        for zid, zone in self.zones.items():
            st_info = zone_styles.get(zone.risk_level, zone_styles["SAFE_STABLE"])
            width = zone.x_max - zone.x_min
            height = zone.y_max - zone.y_min

            # Boundary rectangle
            rect = patches.Rectangle(
                (zone.x_min, zone.y_min), width, height,
                linewidth=2.4, edgecolor=st_info["edge"], facecolor='none',
                linestyle='-', zorder=4
            )
            ax.add_patch(rect)

            # Zone label badge inside the sector
            label_x = zone.x_min + 6.0
            label_y = zone.y_max - 14.0
            zone_desc = f"{zone.zone_id}: {zone.name}\n[{st_info['badge']} · {zone.risk_score}% RISK · {len(zone.assigned_nodes)} NODES]"
            ax.text(
                label_x, label_y, zone_desc,
                fontsize=8.5, fontweight='bold', color='#000000',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#ffffff', edgecolor='#000000', linewidth=1.8, alpha=0.92),
                zorder=7
            )

        # Plot Base Nodes (Solid Blue Diamonds with black border)
        base_nodes = [n for n in self.nodes.values() if not n.is_dynamic]
        if base_nodes:
            bx = [n.x for n in base_nodes]
            by = [n.y for n in base_nodes]
            ax.scatter(
                bx, by, s=150, marker='D', facecolor='#2563eb', edgecolor='#000000',
                linewidth=2.0, label='Base Statutory Inclinometers (Permanent)', zorder=10
            )
            for n in base_nodes:
                ax.text(
                    n.x + 3.0, n.y - 1.0, f"{n.node_id}\n({n.tilt_mm_m:.1f}mm/m)",
                    fontsize=7.8, fontweight='bold', color='#000000',
                    bbox=dict(boxstyle='square,pad=0.18', facecolor='#e0f2fe', edgecolor='#000000', linewidth=1.2),
                    zorder=12
                )

        # Plot Dynamically Deployed Nodes in Danger Zones (Pulsing Gold/Crimson Stars)
        dyn_nodes = [n for n in self.nodes.values() if n.is_dynamic]
        if dyn_nodes:
            dx = [n.x for n in dyn_nodes]
            dy = [n.y for n in dyn_nodes]
            # Outer halo
            ax.scatter(dx, dy, s=360, marker='o', facecolor='#facc15', edgecolor='#b91c1c', linewidth=2.5, alpha=0.5, zorder=13)
            # Center star
            ax.scatter(
                dx, dy, s=190, marker='*', facecolor='#ff0033', edgecolor='#000000',
                linewidth=1.8, label='Dynamic Danger-Zone Micro-Nodes (Active)', zorder=14
            )
            for n in dyn_nodes:
                ax.text(
                    n.x + 3.2, n.y - 1.0, f"⚡ {n.node_id}\n({n.tilt_mm_m:.1f}mm/m)",
                    fontsize=8.0, fontweight='heavy', color='#990000',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#fef08a', edgecolor='#b91c1c', linewidth=1.6),
                    zorder=15
                )

        # Colorbar
        cbar = fig.colorbar(contour_plot, ax=ax, orientation='vertical', fraction=0.035, pad=0.02)
        cbar.set_label('Surface Subsidence Settlement S(x,y) [mm]', fontsize=9.5, fontweight='bold', color='#000000')
        cbar.ax.tick_params(labelsize=8.5)

        # Axis styling
        ax.set_xlim(self.x_bounds[0], self.x_bounds[1])
        ax.set_ylim(self.y_bounds[0], self.y_bounds[1])
        ax.set_xlabel('Mine Concession Transverse Axis X (Meters)', fontsize=9.5, fontweight='bold', color='#000000')
        ax.set_ylabel('Mine Concession Longitudinal Axis Y (Meters)', fontsize=9.5, fontweight='bold', color='#000000')
        ax.set_title('SURFACE CONCESSION SPATIAL RISK MAP & DYNAMIC NODE ALLOCATION\nDGMS Early-Warning Strata Monitoring & Automated Micro-Mesh Densification',
                     fontsize=11.0, fontweight='bold', pad=12, color='#000000')

        ax.grid(True, linestyle=':', alpha=0.45, color='#64748b')
        ax.legend(loc='lower left', framealpha=0.95, facecolor='#ffffff', edgecolor='#000000', prop={'weight': 'bold', 'size': 8.5})

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    def get_scanner_data(self) -> Dict[str, Any]:
        """Returns unified ground scanner telemetry, zone evaluations, and mesh density KPIs."""
        total_nodes = len(self.nodes)
        dynamic_nodes_count = sum(1 for n in self.nodes.values() if n.is_dynamic)
        base_nodes_count = total_nodes - dynamic_nodes_count

        # Evaluate global status
        danger_zones = self.get_danger_zones()
        advisory_zones = [z for z in self.zones.values() if z.risk_level == "ADVISORY_STRAIN"]

        if danger_zones:
            global_status = "CRITICAL HAZARD: DANGER ZONE DETECTED"
            highest_risk_zone = max(self.zones.values(), key=lambda z: z.risk_score)
            status_color = "#ff4d4d"
        elif advisory_zones:
            global_status = "ADVISORY: ELEVATED STRATA STRAIN"
            highest_risk_zone = max(self.zones.values(), key=lambda z: z.risk_score)
            status_color = "#fde047"
        else:
            global_status = "ALL ZONES SECURE & STABLE"
            highest_risk_zone = max(self.zones.values(), key=lambda z: z.risk_score)
            status_color = "#4ade80"

        # Calculate Danger Zone Node Density
        danger_zone_nodes = len(self.zones["ZONE-A"].assigned_nodes)
        danger_area_sqm = (self.zones["ZONE-A"].x_max - self.zones["ZONE-A"].x_min) * (self.zones["ZONE-A"].y_max - self.zones["ZONE-A"].y_min)
        danger_density_per_1000m2 = round((danger_zone_nodes / danger_area_sqm) * 1000.0, 2)

        # Early-Warning Lead Time calculation based on node count in high-risk zones
        # 1 node = ~12 mins lead time; 4+ dense nodes = ~55 mins lead time
        lead_time_mins = min(60, int(12 + (danger_zone_nodes - 1) * 14.5)) if danger_zone_nodes > 0 else 10
        mesh_coverage_pct = min(99.4, round(60.0 + (total_nodes / 8.0) * 39.4, 1))

        return {
            "timestamp": time.strftime("%H:%M:%S"),
            "scan_count": self.scan_count,
            "scenario": self.scenario,
            "global_status": global_status,
            "status_color": status_color,
            "highest_risk_zone": highest_risk_zone.zone_id,
            "highest_risk_score": highest_risk_zone.risk_score,
            "danger_zones_count": len(danger_zones),
            "advisory_zones_count": len(advisory_zones),
            "total_nodes": total_nodes,
            "base_nodes_count": base_nodes_count,
            "dynamic_nodes_count": dynamic_nodes_count,
            "danger_node_density": danger_density_per_1000m2,
            "lead_time_mins": lead_time_mins,
            "mesh_coverage_pct": mesh_coverage_pct,
            "zones": {
                zid: {
                    "zone_id": z.zone_id,
                    "name": z.name,
                    "description": z.description,
                    "max_settlement_mm": z.max_settlement_mm,
                    "avg_tilt_mm_m": z.avg_tilt_mm_m,
                    "vibration_g": z.vibration_g,
                    "void_index": z.void_index,
                    "risk_score": z.risk_score,
                    "risk_level": z.risk_level,
                    "node_count": len(z.assigned_nodes),
                    "recommended_nodes": z.recommended_nodes,
                    "assigned_nodes": z.assigned_nodes
                }
                for zid, z in self.zones.items()
            },
            "nodes": [
                {
                    "node_id": n.node_id,
                    "name": n.name,
                    "node_type": n.node_type,
                    "zone_id": n.zone_id,
                    "x": n.x,
                    "y": n.y,
                    "is_dynamic": n.is_dynamic,
                    "status": n.status,
                    "tilt_mm_m": n.tilt_mm_m,
                    "settlement_mm": n.settlement_mm,
                    "vibration_g": n.vibration_g,
                    "battery_pct": n.battery_pct,
                    "rssi_dbm": n.rssi_dbm,
                    "deployed_at": n.deployed_at
                }
                for n in self.nodes.values()
            ],
            "audit_log": self.audit_log[:15]
        }


# Global singleton ground scanner instance
ground_scanner = GroundScanner()
