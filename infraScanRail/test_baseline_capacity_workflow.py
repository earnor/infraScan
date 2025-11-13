"""
Test script for baseline network capacity calculation.

This script demonstrates the complete baseline capacity workflow in a SINGLE RUN:
1. Generate raw capacity workbook from baseline network
2. Prompt user to manually enrich the workbook
3. After user confirmation:
   - Generate sections from enriched data
   - Calculate capacity and utilization
   - Create all network plots (infrastructure, speed, service, capacity)

WORKFLOW (Single Run):
----------------------
1. Script generates capacity_[network]_network.xlsx with empty tracks/platforms/speed
2. Script PAUSES and prompts user to:
   - Open workbook in Excel
   - Fill tracks/platforms for all stations
   - Fill tracks/speed/length_m for all segments
   - Save as capacity_[network]_network_prep.xlsx
3. User types "y" when done
4. Script continues automatically:
   - Generates sections workbook with capacity analysis
   - Creates all 4 network visualization plots
5. Complete!

Usage:
    python test_baseline_capacity.py
"""

from pathlib import Path
from capacity_calculator import export_capacity_workbook, _build_sections_dataframe
from network_plot import plot_capacity_network, plot_speed_profile_network, plot_service_network
import pandas as pd
import settings

# Get the script directory to build absolute paths
SCRIPT_DIR = Path(__file__).parent
DATA_ROOT = SCRIPT_DIR / "data" / "Network"

# Paths
OUTPUT_DIR = DATA_ROOT / "capacity"


def test_baseline_network():
    """Test capacity calculation for baseline network (standard mode)"""

    print("=" * 80)
    print("Testing Baseline Network Capacity Calculation (STANDARD MODE)")
    print("=" * 80)
    print(f"\nSettings.rail_network: {settings.rail_network}")

    # Temporarily set to standard mode for this test
    original_network = settings.rail_network
    settings.rail_network = "AK_2035"  # Remove _extended suffix for standard mode

    print(f"Testing with: {settings.rail_network}")
    print(f"\n{'─' * 80}")
    print("Expected behavior:")
    print("  - Load edges from: edges_in_corridor.gpkg")
    print("  - Remove edges that also appear in: edges_on_corridor_border.gpkg")
    print("  - Load stations from: points.gpkg (master)")
    print("  - Generate empty workbook for manual enrichment")
    print(f"{'─' * 80}\n")

    try:
        output_path = export_capacity_workbook(
            # No edges_path → BASELINE MODE
            # No enrichment_source → Manual enrichment required
            skip_manual_checkpoint=True  # Skip prompt for testing
        )

        print(f"\n{'=' * 80}")
        print(f"✓ SUCCESS! Baseline capacity workbook created at:")
        print(f"  {output_path}")
        print(f"{'=' * 80}\n")

        prep_path = output_path.with_name(f'{output_path.stem}_prep{output_path.suffix}')

        # Prompt user to enrich the workbook
        print(f"\n{'─' * 80}")
        print("MANUAL ENRICHMENT REQUIRED")
        print(f"{'─' * 80}")
        print("Please complete the following steps:")
        print("  1. Open the workbook in Excel")
        print("  2. Fill 'tracks' and 'platforms' for all stations")
        print("  3. Fill 'tracks', 'speed', 'length_m' for all segments")
        print(f"  4. Save as: {prep_path.name}")
        print(f"{'─' * 80}\n")

        response = input("Have you completed the manual enrichment? (y/n): ").strip().lower()

        if response in {"y", "yes"}:
            if not prep_path.exists():
                print(f"\n❌ ERROR: Prep workbook not found at {prep_path}")
                print("Please save the enriched workbook with the correct name and location.\n")
            else:
                print(f"\n{'─' * 80}")
                print(f"Found prep workbook: {prep_path.name}")
                print("Generating sections and capacity analysis...")
                print(f"{'─' * 80}\n")

                try:
                    # Load prep workbook
                    stations_df = pd.read_excel(prep_path, sheet_name="Stations")
                    segments_df = pd.read_excel(prep_path, sheet_name="Segments")

                    # Generate sections
                    print("[INFO] Building sections from enriched data...")
                    sections_df = _build_sections_dataframe(stations_df, segments_df)

                    if not sections_df.empty:
                        # Round float columns
                        float_columns = sections_df.select_dtypes(include=["float"]).columns
                        if len(float_columns) > 0:
                            sections_df[float_columns] = sections_df[float_columns].round(3)

                        # Export sections workbook
                        sections_path = output_path.with_name(f"{output_path.stem}_sections.xlsx")
                        with pd.ExcelWriter(sections_path, engine="openpyxl") as writer:
                            stations_df.to_excel(writer, sheet_name="Stations", index=False)
                            segments_df.to_excel(writer, sheet_name="Segments", index=False)
                            sections_df.to_excel(writer, sheet_name="Sections", index=False)

                        print(f"✓ Sections workbook created: {sections_path.name}")
                        print(f"  - {len(sections_df)} sections generated")

                        # Generate plots
                        print(f"\n[INFO] Generating network plots...")

                        # Infrastructure map
                        print("  - Creating infrastructure map...")
                        from network_plot import network_current_map
                        network_current_map(network_label=settings.rail_network)

                        # Speed profile map
                        print("  - Creating speed profile map...")
                        plot_speed_profile_network(network_label=settings.rail_network)

                        # Service frequency map
                        print("  - Creating service frequency map...")
                        plot_service_network(network_label=settings.rail_network)

                        # Capacity utilization map
                        print("  - Creating capacity utilization map...")
                        plot_capacity_network(
                            sections_workbook_path=str(sections_path),
                            network_label=settings.rail_network
                        )

                        print(f"\n✓ All plots generated successfully!")

                    else:
                        print("⚠ Warning: No sections could be generated from the data")

                except Exception as e:
                    print(f"\n❌ ERROR during sections/plotting:")
                    print(f"  {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("\nSkipping sections generation and plotting.")
            print("Re-run this script when you have completed the manual enrichment.\n")

    except Exception as e:
        print(f"\n❌ ERROR during capacity calculation:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original setting
        settings.rail_network = original_network


def test_baseline_network_extended():
    """Test capacity calculation for baseline network (extended mode)"""

    print("\n" + "=" * 80)
    print("Testing Baseline Network Capacity Calculation (EXTENDED MODE)")
    print("=" * 80)

    # Temporarily set to extended mode for this test
    original_network = settings.rail_network
    settings.rail_network = "AK_2035_extended"  # Add _extended suffix

    print(f"\nSettings.rail_network: {settings.rail_network}")
    print(f"\n{'─' * 80}")
    print("Expected behavior:")
    print("  - Load edges from: edges_in_corridor.gpkg + edges_on_corridor_border.gpkg")
    print("  - Load stations from: points.gpkg (master)")
    print("  - Generate empty workbook for manual enrichment")
    print(f"{'─' * 80}\n")

    try:
        output_path = export_capacity_workbook(
            # No edges_path → BASELINE MODE (extended)
            # No enrichment_source → Manual enrichment required
            skip_manual_checkpoint=True  # Skip prompt for testing
        )

        print(f"\n{'=' * 80}")
        print(f"✓ SUCCESS! Extended baseline capacity workbook created at:")
        print(f"  {output_path}")
        print(f"{'=' * 80}\n")

        prep_path = output_path.with_name(f'{output_path.stem}_prep{output_path.suffix}')

        # Prompt user to enrich the workbook
        print(f"\n{'─' * 80}")
        print("MANUAL ENRICHMENT REQUIRED")
        print(f"{'─' * 80}")
        print("Please complete the following steps:")
        print("  1. Open the workbook in Excel")
        print("  2. Fill 'tracks' and 'platforms' for all stations")
        print("  3. Fill 'tracks', 'speed', 'length_m' for all segments")
        print(f"  4. Save as: {prep_path.name}")
        print(f"{'─' * 80}\n")

        response = input("Have you completed the manual enrichment? (y/n): ").strip().lower()

        if response in {"y", "yes"}:
            if not prep_path.exists():
                print(f"\n❌ ERROR: Prep workbook not found at {prep_path}")
                print("Please save the enriched workbook with the correct name and location.\n")
            else:
                print(f"\n{'─' * 80}")
                print(f"Found prep workbook: {prep_path.name}")
                print("Generating sections and capacity analysis...")
                print(f"{'─' * 80}\n")

                try:
                    # Load prep workbook
                    stations_df = pd.read_excel(prep_path, sheet_name="Stations")
                    segments_df = pd.read_excel(prep_path, sheet_name="Segments")

                    # Generate sections
                    print("[INFO] Building sections from enriched data...")
                    sections_df = _build_sections_dataframe(stations_df, segments_df)

                    if not sections_df.empty:
                        # Round float columns
                        float_columns = sections_df.select_dtypes(include=["float"]).columns
                        if len(float_columns) > 0:
                            sections_df[float_columns] = sections_df[float_columns].round(3)

                        # Export sections workbook
                        sections_path = output_path.with_name(f"{output_path.stem}_sections.xlsx")
                        with pd.ExcelWriter(sections_path, engine="openpyxl") as writer:
                            stations_df.to_excel(writer, sheet_name="Stations", index=False)
                            segments_df.to_excel(writer, sheet_name="Segments", index=False)
                            sections_df.to_excel(writer, sheet_name="Sections", index=False)

                        print(f"✓ Sections workbook created: {sections_path.name}")
                        print(f"  - {len(sections_df)} sections generated")

                        # Generate plots
                        print(f"\n[INFO] Generating network plots...")

                        # Infrastructure map
                        print("  - Creating infrastructure map...")
                        from network_plot import network_current_map
                        network_current_map(network_label=settings.rail_network)

                        # Speed profile map
                        print("  - Creating speed profile map...")
                        plot_speed_profile_network(network_label=settings.rail_network)

                        # Service frequency map
                        print("  - Creating service frequency map...")
                        plot_service_network(network_label=settings.rail_network)

                        # Capacity utilization map
                        print("  - Creating capacity utilization map...")
                        plot_capacity_network(
                            sections_workbook_path=str(sections_path),
                            network_label=settings.rail_network
                        )

                        print(f"\n✓ All plots generated successfully!")

                    else:
                        print("⚠ Warning: No sections could be generated from the data")

                except Exception as e:
                    print(f"\n❌ ERROR during sections/plotting:")
                    print(f"  {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print("\nSkipping sections generation and plotting.")
            print("Re-run this script when you have completed the manual enrichment.\n")

    except Exception as e:
        print(f"\n❌ ERROR during extended capacity calculation:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original setting
        settings.rail_network = original_network


if __name__ == "__main__":
    # Test standard mode
    #test_baseline_network()

    # Test extended mode
    test_baseline_network_extended()

    print("\n" + "=" * 80)
    print("All baseline tests complete!")
    print("=" * 80)
