"""
Comprehensive Automated Test Suite for Module 4:
Wide Area Satellite/Camera Scanner & Closed-Loop Adaptive Geotechnical Mesh.
Verifies all 8 stages:
1. Wide Area Camera / Satellite Simulation
2. Zone Detection (SAFE / MODERATE / DANGER)
3. Sensor Network node density scaling (Few / More / Dense Nodes)
4. Real-Time Data (Tilt, Displacement, Vibration, Crack)
5. Analysis + Anomaly Detection (Rate of crack, tilt jerk, micro-tremor)
6. Risk Score formulation
7. Alert levels
8. Risk Zone Updated (Closed-Loop Feedback ↺)
"""
import sys
from src.telemetry.ground_scanner import ground_scanner

def test_ground_scanner():
    print("=================================================================")
    print("[STAGE 1 & 2] Testing Baseline Ground Scanner & Zone Classification...")
    ground_scanner.reset_to_baseline()
    ground_scanner.set_scenario("NORMAL")
    data = ground_scanner.get_scanner_data()
    
    assert data["total_nodes"] == 4, f"Expected 4 baseline nodes, got {data['total_nodes']}"
    assert data["dynamic_nodes_count"] == 0, f"Expected 0 dynamic nodes, got {data['dynamic_nodes_count']}"
    assert data["danger_zones_count"] == 0, f"Expected 0 danger zones in NORMAL mode, got {data['danger_zones_count']}"
    assert data["safe_zones_count"] >= 3, "Expected majority zones to be SAFE in baseline"
    print(f"  [OK] Baseline verified: {data['total_nodes']} nodes, Global Status: '{data['global_status']}'")

    print("\n[STAGE 3 & 4] Validating 4 Geotechnical Sensor Streams (Tilt, Displacement, Vibration, Crack)...")
    for n in data["nodes"]:
        assert "tilt_mm_m" in n, "Missing tilt stream"
        assert "settlement_mm" in n, "Missing displacement stream"
        assert "vibration_g" in n, "Missing vibration stream"
        assert "crack_mm" in n, "Missing crack stream"
        assert len(n["history_tilt"]) > 0, "Missing rolling history for tilt"
        assert len(n["history_crack"]) > 0, "Missing rolling history for crack"
    print(f"  [OK] 4 Geotechnical sensor streams verified across all {len(data['nodes'])} active nodes.")

    print("\n[STAGE 5 & 6] Testing Moderate Strain & Anomaly Detection (Fissures & Angular Shear)...")
    ground_scanner.set_scenario("MODERATE_STRAIN")
    data_mod = ground_scanner.get_scanner_data()
    assert data_mod["moderate_zones_count"] >= 1, "Expected at least 1 MODERATE zone"
    mod_zone = data_mod["zones"]["ZONE-B"]
    assert mod_zone["crack_mm"] > 2.0, f"Expected crack opening > 2.0mm, got {mod_zone['crack_mm']}mm"
    assert len(data_mod["all_anomalies"]) > 0, "Expected anomalies to be detected under moderate strain"
    print(f"  [OK] Moderate strain anomalies verified: {data_mod['all_anomalies']}")

    print("\n[STAGE 6 & 7] Testing Critical Danger Classification & Evacuation Alert Dispatch...")
    ground_scanner.set_scenario("CRITICAL_SUBSIDENCE")
    data_crit = ground_scanner.get_scanner_data()
    
    assert data_crit["danger_zones_count"] >= 1, "Expected at least 1 danger zone under CRITICAL_SUBSIDENCE"
    assert data_crit["highest_risk_zone"] == "ZONE-A", f"Expected ZONE-A to be highest risk, got {data_crit['highest_risk_zone']}"
    assert data_crit["highest_risk_score"] > 70.0, f"Expected risk > 70%, got {data_crit['highest_risk_score']}%"
    assert data_crit["alert_tier"] == "CRITICAL_EVACUATION", f"Expected CRITICAL_EVACUATION, got {data_crit['alert_tier']}"
    print(f"  [OK] Critical Danger verified: Sector {data_crit['highest_risk_zone']} Risk = {data_crit['highest_risk_score']}% (Alert: {data_crit['alert_tier']})")

    print("\n[STAGE 8] Testing Closed-Loop Dynamic Node Allocation (Dense Nodes in Danger Zone)...")
    prev_nodes = data_crit["total_nodes"]
    count_deployed, deployed_ids = ground_scanner.auto_deploy_danger_nodes()
    data_deployed = ground_scanner.get_scanner_data()
    
    assert count_deployed > 0, "Expected at least 1 node to be dynamically deployed"
    assert data_deployed["dynamic_nodes_count"] == count_deployed, f"Dynamic node count mismatch: {data_deployed['dynamic_nodes_count']} vs {count_deployed}"
    assert data_deployed["total_nodes"] == prev_nodes + count_deployed, "Total nodes count mismatch"
    assert data_deployed["lead_time_mins"] > data_crit["lead_time_mins"], "Expected lead time to increase after densification"
    print(f"  [OK] Closed-loop allocation verified: {count_deployed} dense nodes deployed ({deployed_ids})")
    print(f"  [OK] Early-Warning Lead Time boosted from {data_crit['lead_time_mins']} mins to {data_deployed['lead_time_mins']} mins")
    print(f"  [OK] Danger sector node density: {data_deployed['danger_node_density']} nodes / 1000m^2")

    print("\n[STAGE 1 SURVEILLANCE] Testing Satellite InSAR & Aerial Surveillance View Generation...")
    sat_bytes = ground_scanner.generate_satellite_simulation_view()
    assert isinstance(sat_bytes, bytes) and len(sat_bytes) > 20000, "Satellite simulation view generation failed"
    print(f"  [OK] Satellite Concession Surveillance view generated: {len(sat_bytes)} bytes")

    insar_bytes = ground_scanner.generate_insar_fringe_view()
    assert isinstance(insar_bytes, bytes) and len(insar_bytes) > 20000, "D-InSAR interferogram generation failed"
    print(f"  [OK] Differential InSAR Phase Fringe view generated: {len(insar_bytes)} bytes")

    map_bytes = ground_scanner.generate_spatial_risk_map()
    assert isinstance(map_bytes, bytes) and len(map_bytes) > 20000, "2D Geotechnical Spatial Map generation failed"
    print(f"  [OK] 2D Spatial Geotechnical Map generated: {len(map_bytes)} bytes")

    print("\n[RESET & REBALANCE] Testing Return to Statutory Baseline Mesh...")
    ground_scanner.reset_to_baseline()
    data_reset = ground_scanner.get_scanner_data()
    assert data_reset["total_nodes"] == 4, f"Expected 4 nodes after reset, got {data_reset['total_nodes']}"
    assert data_reset["dynamic_nodes_count"] == 0, "Expected 0 dynamic nodes after reset"
    print(f"  [OK] Baseline restored cleanly: {data_reset['total_nodes']} permanent statutory nodes active.")

    print("\n=================================================================")
    print("ALL MODULE 4 CLOSED-LOOP STAGES VERIFIED SUCCESSFULLY!")
    print("=================================================================")

if __name__ == "__main__":
    test_ground_scanner()
