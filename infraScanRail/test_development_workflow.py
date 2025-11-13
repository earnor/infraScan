"""Test script for the development capacity workflow.

This script demonstrates how to process a development network:
1. Loads development edges
2. Filters stations to baseline + new stations
3. Filters edges to those with all nodes in filtered stations
4. Enriches from baseline prep workbook
5. Generates capacity workbook and plots
"""

from pathlib import Path
from capacity_calculator import export_capacity_workbook
from network_plot import network_current_map, plot_capacity_network, plot_speed_profile_network, plot_service_network

# Get the script directory to use absolute paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR  # Script is in infraScanRail/

# Example: Process development 101023
# Note: Development files have .0 suffix (e.g., 101023.0.gpkg)
dev_id = "101032"
dev_edges_filename = f"{dev_id}.0.gpkg"
dev_edges_path = PROJECT_ROOT / "data" / "Network" / "processed" / "developments" / dev_edges_filename
network_label = f"AK_2035_dev_{dev_id}"

print(f"\n{'='*80}")
print(f"TESTING DEVELOPMENT WORKFLOW: {network_label}")
print(f"{'='*80}\n")

# Check if development file exists
if not dev_edges_path.exists():
    print(f"[ERROR] Development edges file not found: {dev_edges_path}")
    print("Please ensure development files exist before running this test.")
    exit(1)

print(f"[INFO] Development edges file: {dev_edges_path}")
print(f"[INFO] Network label: {network_label}")

# Step 1: Generate capacity workbook
# This will:
# - Load development edges
# - Filter stations to baseline NR column + new stations (flagged with [NEW])
# - Filter edges to those with all nodes (FromNode, ToNode, Via) in filtered stations
# - Auto-enrich from baseline prep workbook
# - Save to: data/Network/capacity/developments/{dev_id}/
print(f"\n[STEP 1] Generating capacity workbook...")

try:
    capacity_workbook = export_capacity_workbook(
        edges_path=dev_edges_path,
        network_label=network_label,
        # enrichment_source is auto-detected from settings.rail_network
        # output_dir is auto-detected from network_label
        skip_manual_checkpoint=False,  # Will prompt if new stations exist
    )
    print(f"[SUCCESS] Capacity workbook saved: {capacity_workbook}")
except FileNotFoundError as e:
    print(f"[ERROR] {e}")
    print("\nPlease ensure:")
    print("  1. Baseline prep workbook exists (run baseline workflow first)")
    print("  2. points.gpkg exists in data/Network/processed/")
    exit(1)
except ValueError as e:
    print(f"[ERROR] {e}")
    exit(1)

# Step 2: Generate network plots
# Plots will be saved to: plots/network/developments/{dev_id}/
print(f"\n[STEP 2] Generating network plots...")

try:
    # Infrastructure plot
    infra_plot = network_current_map(network_label=network_label)
    print(f"[SUCCESS] Infrastructure plot: {infra_plot}")

    # Capacity plot (requires sections workbook)
    try:
        base_plot, capacity_plot = plot_capacity_network(network_label=network_label)
        print(f"[SUCCESS] Capacity plots: {base_plot}, {capacity_plot}")
    except Exception as e:
        print(f"[WARNING] Could not generate capacity plot: {e}")

    # Speed profile plot
    speed_plot = plot_speed_profile_network(network_label=network_label)
    print(f"[SUCCESS] Speed plot: {speed_plot}")

    # Service plot
    service_plot = plot_service_network(network_label=network_label)
    print(f"[SUCCESS] Service plot: {service_plot}")

except FileNotFoundError as e:
    print(f"[ERROR] {e}")
    print("Prep workbook not found. Please complete manual enrichment first.")
    exit(1)

print(f"\n{'='*80}")
print(f"DEVELOPMENT WORKFLOW TEST COMPLETED")
print(f"{'='*80}\n")
print("Output locations:")
print(f"  - Capacity workbooks: data/Network/capacity/developments/{dev_id}/")
print(f"  - Network plots: plots/network/developments/{dev_id}/")
