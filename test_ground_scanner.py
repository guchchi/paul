"""
Comprehensive Automated Test Suite for Ground Scanner & Dynamic Zone Risk Mesh.
Verifies spatial 2D modeling, zone risk separation, dynamic node allocation in danger zones,
and 2D contour map image generation.
"""
import sys
from src.telemetry.ground_scanner import ground_scanner

def test_ground_scanner():
    print("=================================================================")
    print("[STEP 1] Testing Baseline Ground Scanner Initialization...")
    ground_scanner.reset_to_baseline()
    ground_scanner.set_scenario("NORMAL")
    data = ground_scanner.get_scanner_data()
    
    assert data["total_nodes"] == 4, f"Expected 4 baseline nodes, got {data['total_nodes']}"
    assert data["dynamic_nodes_count"] == 0, f"Expected 0 dynamic nodes, got {data['dynamic_nodes_count']}"
    assert data["danger_zones_count"] == 0, f"Expected 0 danger zones in NORMAL mode, got {data['danger_zones_count']}"
    print(f"  [OK] Baseline verified: {data['total_nodes']} nodes, Global Status: '{data['global_status']}'")

    print("\n[STEP 2] Testing Strata Strain & Critical Danger Classification...")
    ground_scanner.set_scenario("CRITICAL_SUBSIDENCE")
    data_crit = ground_scanner.get_scanner_data()
    
    assert data_crit["danger_zones_count"] >= 1, "Expected at least 1 danger zone under CRITICAL_SUBSIDENCE"
    assert data_crit["highest_risk_zone"] == "ZONE-A", f"Expected ZONE-A to be highest risk, got {data_crit['highest_risk_zone']}"
    assert data_crit["highest_risk_score"] > 70.0, f"Expected risk > 70%, got {data_crit['highest_risk_score']}%"
    print(f"  [OK] Danger classification verified: Sector {data_crit['highest_risk_zone']} Risk = {data_crit['highest_risk_score']}% (CRITICAL)")

    print("\n[STEP 3] Testing Dynamic Danger-Zone Node Allocation...")
    prev_nodes = data_crit["total_nodes"]
    count_deployed, deployed_ids = ground_scanner.auto_deploy_danger_nodes()
    data_deployed = ground_scanner.get_scanner_data()
    
    assert count_deployed > 0, "Expected at least 1 node to be dynamically deployed"
    assert data_deployed["dynamic_nodes_count"] == count_deployed, f"Dynamic node count mismatch: {data_deployed['dynamic_nodes_count']} vs {count_deployed}"
    assert data_deployed["total_nodes"] == prev_nodes + count_deployed, "Total nodes count mismatch"
    assert data_deployed["lead_time_mins"] > data_crit["lead_time_mins"], "Expected lead time to increase after densification"
    print(f"  [OK] Dynamic allocation verified: {count_deployed} nodes deployed ({deployed_ids})")
    print(f"  [OK] Lead time boosted from {data_crit['lead_time_mins']} mins to {data_deployed['lead_time_mins']} mins")
    print(f"  [OK] Danger sector node density: {data_deployed['danger_node_density']} nodes / 1000m^2")

    print("\n[STEP 4] Testing Manual Operator Node Placement...")
    manual_id = ground_scanner.deploy_manual_node("ZONE-C", "STRAIN_EXTENSOMETER")
    assert manual_id is not None, "Failed to deploy manual node"
    assert manual_id in ground_scanner.nodes, f"Node {manual_id} not in scanner nodes"
    assert ground_scanner.nodes[manual_id].zone_id == "ZONE-C"
    print(f"  [OK] Manual node placement verified: {manual_id} provisioned in ZONE-C")

    print("\n[STEP 5] Testing Reset to Statutory Baseline...")
    ground_scanner.reset_to_baseline()
    data_reset = ground_scanner.get_scanner_data()
    assert data_reset["total_nodes"] == 4, f"Expected 4 nodes after reset, got {data_reset['total_nodes']}"
    assert data_reset["dynamic_nodes_count"] == 0, "Expected 0 dynamic nodes after reset"
    print(f"  [OK] Reset verified: Mesh restored cleanly to {data_reset['total_nodes']} baseline nodes")

    print("\n[STEP 6] Testing 2D Spatial Map Generation...")
    map_bytes = ground_scanner.generate_spatial_risk_map()
    assert isinstance(map_bytes, bytes), "Expected bytes output"
    assert len(map_bytes) > 20000, f"Map PNG bytes too small: {len(map_bytes)} bytes"
    print(f"  [OK] 2D Spatial Map generated successfully: {len(map_bytes)} PNG bytes")

    print("\n=================================================================")
    print("ALL TEST ASSERTIONS PASSED SUCCESSFULLY!")
    print("=================================================================")

if __name__ == "__main__":
    test_ground_scanner()
