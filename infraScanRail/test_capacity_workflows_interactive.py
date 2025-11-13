"""
Interactive end-to-end test script for all capacity calculation workflows.

This script provides FULL INTERACTIVE testing for:
1. Baseline (Standard) - edges_in_corridor.gpkg + points_corridor.gpkg
2. Baseline Extended - edges_in_corridor.gpkg + points.gpkg (no corridor filtering)
3. Development - custom development edges + auto-enrichment from baseline

Each test includes ALL MANUAL PROMPTS as they exist in the real workflow:
- Workbook generation (Stations + Segments sheets)
- **MANUAL PROMPT**: User asked to fill NA values in Excel
- User confirmation before proceeding to sections calculation
- Sections calculation (groups continuous segments, calculates capacity)
- **INTERACTIVE PROMPTS**: For sections with multiple capacity strategies
- Visualization (network plots for capacity, speed, services)

Usage:
    python test_capacity_workflows_interactive.py

    Then uncomment the workflow you want to test in the main section.
    Follow the prompts during execution to simulate the real workflow.
"""

from pathlib import Path
import pandas as pd
from capacity_calculator import (
    export_capacity_workbook,
    _derive_prep_path,
    _derive_sections_path
)
import settings

# Get the script directory to build absolute paths
SCRIPT_DIR = Path(__file__).parent
DATA_ROOT = SCRIPT_DIR / "data" / "Network"


def test_baseline_standard_interactive():
    """
    INTERACTIVE Test 1: Baseline Standard Workflow with ALL MANUAL PROMPTS

    Tests the full pipeline with manual prompts:
    1. Generate raw workbook (NA infrastructure fields)
    2. **PROMPT**: User asked to manually fill infrastructure in Excel
    3. **PROMPT**: User confirms completion of manual enrichment
    4. Calculate sections (group segments, calculate capacity)
    5. **PROMPTS**: For sections with multiple capacity strategies (if applicable)
    6. Visualize results (capacity, speed, service maps)
    """
    print("\n" + "=" * 80)
    print("INTERACTIVE TEST 1: BASELINE STANDARD WORKFLOW")
    print("=" * 80)

    # Temporarily set to standard mode
    original_network = settings.rail_network
    settings.rail_network = "AK_2035"  # Standard mode

    print(f"\nSettings.rail_network: {settings.rail_network}")
    print(f"\n{'─' * 80}")
    print("Pipeline steps (INTERACTIVE):")
    print("  1. Generate raw workbook (edges_in_corridor.gpkg + points_corridor.gpkg)")
    print("  2. **YOU** manually fill tracks, platforms, speed, length in Excel")
    print("  3. Confirm completion before proceeding")
    print("  4. Calculate sections (group segments, calculate capacity)")
    print("  5. Interactive prompts for capacity strategy selection (if needed)")
    print("  6. Generate visualizations automatically")
    print(f"{'─' * 80}\n")

    try:
        # Step 1 & 2 & 3: Generate raw workbook with MANUAL PROMPTS ENABLED
        print("STEP 1-3: Generating raw capacity workbook with MANUAL ENRICHMENT...")
        print("\nNOTE: You will be prompted to:")
        print("  - Open the workbook in Excel")
        print("  - Fill all NA values for tracks, platforms, speed, length_m")
        print("  - Save the file as *_prep.xlsx")
        print("  - Return here and confirm completion")
        print("\nStarting workflow...\n")

        output_path = export_capacity_workbook(
            edges_path=None,  # BASELINE MODE
            network_label=None,
            enrichment_source=None,  # Manual enrichment required
            skip_manual_checkpoint=False  # *** ENABLE MANUAL PROMPTS ***
        )

        print(f"\n✓ Workbook pipeline completed at: {output_path}")

        # At this point, if user confirmed, sections workbook exists
        sections_path = _derive_sections_path(output_path)

        # Step 4: Visualize results (if sections exist)
        if sections_path.exists():
            print("\nSTEP 4: Generating visualizations...")
            try:
                from network_plot import (
                    plot_capacity_network,
                    plot_speed_profile_network,
                    plot_service_network
                )

                print(f"  Generating capacity network plot...")
                plot_capacity_network(
                    workbook_path=sections_path,
                    network_label="AK_2035"
                )
                print(f"    ✓ Capacity network plot generated")

                print(f"  Generating speed profile network plot...")
                plot_speed_profile_network(
                    workbook_path=sections_path,
                    network_label="AK_2035"
                )
                print(f"    ✓ Speed profile network plot generated")

                print(f"  Generating service network plot...")
                plot_service_network(
                    workbook_path=sections_path,
                    network_label="AK_2035"
                )
                print(f"    ✓ Service network plot generated")

                print(f"✓ Visualizations complete")
            except ImportError as e:
                print(f"  ⚠ WARNING: Could not import network_plot module: {e}")
                print(f"    Skipping visualization step")

        # Final summary
        print(f"\n{'=' * 80}")
        print(f"✓ INTERACTIVE SUCCESS! Full baseline standard pipeline executed")
        print(f"{'=' * 80}\n")
        print("Output files:")
        print(f"  Raw workbook:      {output_path}")
        prep_path = _derive_prep_path(output_path)
        if prep_path.exists():
            print(f"  Prep workbook:     {prep_path}")
        if sections_path.exists():
            print(f"  Sections workbook: {sections_path}")
        print(f"\nVerification checklist:")
        print(f"  1. Open sections workbook in Excel (if generated)")
        print(f"  2. Check Stations sheet: all tracks/platforms filled")
        print(f"  3. Check Segments sheet: all tracks/speed/length filled")
        print(f"  4. Check Sections sheet: capacity and utilization calculated")
        print(f"  5. Check plots folder for network visualization PNG files")
        print(f"\n{'─' * 80}\n")

    except Exception as e:
        print(f"\n❌ ERROR during baseline standard interactive workflow:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original setting
        settings.rail_network = original_network


def test_baseline_extended_interactive():
    """
    INTERACTIVE Test 2: Baseline Extended Workflow with ALL MANUAL PROMPTS

    Tests the full pipeline with manual prompts (extended mode):
    1. Generate raw workbook (edges_in_corridor.gpkg + points.gpkg, no filtering)
    2. **PROMPT**: User asked to manually fill infrastructure in Excel
    3. **PROMPT**: User confirms completion of manual enrichment
    4. Calculate sections (group segments, calculate capacity)
    5. **PROMPTS**: For sections with multiple capacity strategies (if applicable)
    6. Visualize results (capacity, speed, service maps)
    """
    print("\n" + "=" * 80)
    print("INTERACTIVE TEST 2: BASELINE EXTENDED WORKFLOW")
    print("=" * 80)

    # Temporarily set to extended mode
    original_network = settings.rail_network
    settings.rail_network = "AK_2035_extended"  # Extended mode

    print(f"\nSettings.rail_network: {settings.rail_network}")
    print(f"\n{'─' * 80}")
    print("Pipeline steps (INTERACTIVE):")
    print("  1. Generate raw workbook (edges_in_corridor.gpkg + points.gpkg, no filtering)")
    print("  2. **YOU** manually fill tracks, platforms, speed, length in Excel")
    print("  3. Confirm completion before proceeding")
    print("  4. Calculate sections (group segments, calculate capacity)")
    print("  5. Interactive prompts for capacity strategy selection (if needed)")
    print("  6. Generate visualizations automatically")
    print(f"{'─' * 80}\n")

    try:
        # Step 1 & 2 & 3: Generate raw workbook with MANUAL PROMPTS ENABLED
        print("STEP 1-3: Generating raw capacity workbook with MANUAL ENRICHMENT...")
        print("\nNOTE: You will be prompted to:")
        print("  - Open the workbook in Excel")
        print("  - Fill all NA values for tracks, platforms, speed, length_m")
        print("  - Save the file as *_prep.xlsx")
        print("  - Return here and confirm completion")
        print("\nStarting workflow...\n")

        output_path = export_capacity_workbook(
            edges_path=None,  # BASELINE MODE (extended)
            network_label=None,
            enrichment_source=None,  # Manual enrichment required
            skip_manual_checkpoint=False  # *** ENABLE MANUAL PROMPTS ***
        )

        print(f"\n✓ Workbook pipeline completed at: {output_path}")

        # At this point, if user confirmed, sections workbook exists
        sections_path = _derive_sections_path(output_path)

        # Step 4: Visualize results (if sections exist)
        if sections_path.exists():
            print("\nSTEP 4: Generating visualizations...")
            try:
                from network_plot import (
                    plot_capacity_network,
                    plot_speed_profile_network,
                    plot_service_network
                )

                print(f"  Generating capacity network plot...")
                plot_capacity_network(
                    workbook_path=sections_path,
                    network_label="AK_2035_extended"
                )
                print(f"    ✓ Capacity network plot generated")

                print(f"  Generating speed profile network plot...")
                plot_speed_profile_network(
                    workbook_path=sections_path,
                    network_label="AK_2035_extended"
                )
                print(f"    ✓ Speed profile network plot generated")

                print(f"  Generating service network plot...")
                plot_service_network(
                    workbook_path=sections_path,
                    network_label="AK_2035_extended"
                )
                print(f"    ✓ Service network plot generated")

                print(f"✓ Visualizations complete")
            except ImportError as e:
                print(f"  ⚠ WARNING: Could not import network_plot module: {e}")
                print(f"    Skipping visualization step")

        # Final summary
        print(f"\n{'=' * 80}")
        print(f"✓ INTERACTIVE SUCCESS! Full baseline extended pipeline executed")
        print(f"{'=' * 80}\n")
        print("Output files:")
        print(f"  Raw workbook:      {output_path}")
        prep_path = _derive_prep_path(output_path)
        if prep_path.exists():
            print(f"  Prep workbook:     {prep_path}")
        if sections_path.exists():
            print(f"  Sections workbook: {sections_path}")
        print(f"\nVerification checklist:")
        print(f"  1. Compare station count with standard mode (should be MORE)")
        print(f"  2. Check Sections sheet: MORE sections than standard mode")
        print(f"  3. Verify Via nodes outside corridor are included")
        print(f"  4. Check visualizations show extended network coverage")
        print(f"\n{'─' * 80}\n")

    except Exception as e:
        print(f"\n❌ ERROR during baseline extended interactive workflow:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original setting
        settings.rail_network = original_network


def test_development_interactive():
    """
    INTERACTIVE Test 3: Development Workflow with MANUAL PROMPTS (if needed)

    Tests the full pipeline with prompts for NEW infrastructure:
    1. Generate auto-enriched workbook (development edges + baseline prep)
    2. **PROMPT** (if NEW infrastructure): User fills NA values for NEW segments/stations
    3. **PROMPT**: User confirms completion (if manual enrichment was needed)
    4. Calculate sections (group segments, calculate capacity)
    5. **PROMPTS**: For sections with multiple capacity strategies (if applicable)
    6. Visualize results (capacity, speed, service maps)

    NOTE: This test uses AK_2035_extended as the baseline network.
    """
    print("\n" + "=" * 80)
    print("INTERACTIVE TEST 3: DEVELOPMENT WORKFLOW (AK_2035_extended)")
    print("=" * 80)

    # Temporarily set to extended mode for proper baseline detection
    original_network = settings.rail_network
    settings.rail_network = "AK_2035_extended"  # Extended baseline

    # Configure development parameters
    # Note: Geopackage files have .0 before .gpkg (e.g., 101032.0.gpkg)
    DEV_ID = "101032.0"  # Example development ID with .0 suffix
    DEV_EDGES_PATH = DATA_ROOT / "processed" / "developments" / f"{DEV_ID}.gpkg"
    BASELINE_PREP_PATH = DATA_ROOT / "capacity" / "AK_2035_extended" / "capacity_AK_2035_extended_network_prep.xlsx"

    print(f"\nSettings.rail_network: {settings.rail_network}")
    print(f"Development ID: {DEV_ID}")
    print(f"Development edges: {DEV_EDGES_PATH}")
    print(f"Baseline network: AK_2035_extended")
    print(f"Baseline prep: {BASELINE_PREP_PATH}")
    print(f"\n{'─' * 80}")
    print("Pipeline steps (INTERACTIVE):")
    print("  1. Generate auto-enriched workbook (inherits from baseline prep)")
    print("  2. **YOU** manually fill NEW infrastructure only (if any)")
    print("  3. Confirm completion before proceeding (if manual enrichment needed)")
    print("  4. Calculate sections (group segments, calculate capacity)")
    print("  5. Interactive prompts for capacity strategy selection (if needed)")
    print("  6. Generate visualizations automatically")
    print(f"{'─' * 80}\n")

    # Check if files exist
    if not DEV_EDGES_PATH.exists():
        print(f"⚠ WARNING: Development edges file not found:")
        print(f"  {DEV_EDGES_PATH}")
        print(f"\nPlease update DEV_EDGES_PATH in this script to point to your development file.")
        print("Skipping development workflow test.\n")
        settings.rail_network = original_network  # Restore setting
        return

    if not BASELINE_PREP_PATH.exists():
        print(f"⚠ WARNING: Baseline prep workbook not found:")
        print(f"  {BASELINE_PREP_PATH}")
        print(f"\nPlease run the baseline workflow first and complete manual enrichment.")
        print("Skipping development workflow test.\n")
        settings.rail_network = original_network  # Restore setting
        return

    try:
        # Step 1 & 2 & 3: Generate auto-enriched workbook with MANUAL PROMPTS ENABLED
        print("STEP 1-3: Generating auto-enriched workbook with INTERACTIVE PROMPTS...")
        print("\nNOTE: If there are NEW stations or segments:")
        print("  - You will be prompted to open the workbook in Excel")
        print("  - Fill NA values for NEW infrastructure only")
        print("  - Existing infrastructure is already enriched from baseline")
        print("  - Return here and confirm completion")
        print("\nStarting workflow...\n")

        output_path = export_capacity_workbook(
            edges_path=DEV_EDGES_PATH,  # DEVELOPMENT MODE
            network_label=f"AK_2035_extended_dev_{DEV_ID}",
            enrichment_source=BASELINE_PREP_PATH,  # Auto-enrichment
            skip_manual_checkpoint=False  # *** ENABLE MANUAL PROMPTS ***
        )

        print(f"\n✓ Workbook pipeline completed at: {output_path}")

        # At this point, if user confirmed, sections workbook exists
        sections_path = _derive_sections_path(output_path)

        # Step 4: Visualize results (if sections exist)
        if sections_path.exists():
            print("\nSTEP 4: Generating visualizations...")
            try:
                from network_plot import (
                    plot_capacity_network,
                    plot_speed_profile_network,
                    plot_service_network
                )

                print(f"  Generating capacity network plot...")
                plot_capacity_network(
                    workbook_path=sections_path,
                    network_label=f"AK_2035_extended_dev_{DEV_ID}"
                )
                print(f"    ✓ Capacity network plot generated")

                print(f"  Generating speed profile network plot...")
                plot_speed_profile_network(
                    workbook_path=sections_path,
                    network_label=f"AK_2035_extended_dev_{DEV_ID}"
                )
                print(f"    ✓ Speed profile network plot generated")

                print(f"  Generating service network plot...")
                plot_service_network(
                    workbook_path=sections_path,
                    network_label=f"AK_2035_extended_dev_{DEV_ID}"
                )
                print(f"    ✓ Service network plot generated")

                print(f"✓ Visualizations complete")
            except ImportError as e:
                print(f"  ⚠ WARNING: Could not import network_plot module: {e}")
                print(f"    Skipping visualization step")

        # Final summary
        print(f"\n{'=' * 80}")
        print(f"✓ INTERACTIVE SUCCESS! Full development pipeline executed")
        print(f"{'=' * 80}\n")
        print("Output files:")
        print(f"  Auto-enriched workbook: {output_path}")
        prep_path = _derive_prep_path(output_path)
        if prep_path.exists():
            print(f"  Prep workbook:          {prep_path}")
        if sections_path.exists():
            print(f"  Sections workbook:      {sections_path}")
        print(f"\nVerification checklist:")
        print(f"  1. Check for [NEW] flags in station names")
        print(f"  2. Verify existing infrastructure inherited from baseline")
        print(f"  3. Check NEW infrastructure has appropriate values")
        print(f"  4. Compare sections with baseline (should show new capacity)")
        print(f"  5. Check visualizations highlight development changes")
        print(f"\n{'─' * 80}\n")

    except Exception as e:
        print(f"\n❌ ERROR during development interactive workflow:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original setting
        settings.rail_network = original_network


def test_all_workflows_interactive():
    """
    Run all three interactive workflow tests sequentially.

    WARNING: This will require manual intervention for EACH workflow!
    - You will need to fill Excel workbooks 3 times (once per workflow)
    - Each workflow waits for your confirmation before proceeding
    - Consider testing workflows individually instead
    """
    print("\n" + "=" * 80)
    print("TESTING ALL INTERACTIVE CAPACITY WORKFLOWS")
    print("=" * 80)
    print("\n⚠ WARNING: This will require manual Excel work for ALL THREE workflows!")
    print("Consider testing workflows individually for easier management.\n")

    response = input("Continue with all workflows? (y/n): ").strip().lower()
    if response not in {"y", "yes"}:
        print("Cancelled. Please uncomment individual test functions instead.\n")
        return

    test_baseline_standard_interactive()
    test_baseline_extended_interactive()
    test_development_interactive()

    print("\n" + "=" * 80)
    print("ALL INTERACTIVE WORKFLOW TESTS FINISHED!")
    print("=" * 80)


if __name__ == "__main__":
    # Uncomment the workflow you want to test:

    # Test individual interactive workflows
    test_baseline_standard_interactive()
    #test_baseline_extended_interactive()
    #test_development_interactive()

    # Or test all interactive workflows (requires manual work for each!)
    # test_all_workflows_interactive()
