"""
Unified CLI tool for rail network capacity analysis.

This script provides a single entry point for all capacity workflows:
1. Baseline - Standard baseline workflow (corridor-filtered stations)
2. Baseline Extended - Extended baseline workflow (all stations, no corridor filtering)
3. Development - Development network workflow with auto-enrichment
4. Enhanced - Phase 4 capacity interventions (enhanced baseline)

Usage:
    # Baseline workflows
    python run_capacity_analysis.py baseline
    python run_capacity_analysis.py baseline-extended --network 2024_extended

    # Development workflow
    python run_capacity_analysis.py development --dev-id 101032.0

    # Enhanced workflow (Phase 4)
    python run_capacity_analysis.py enhanced --network 2024_extended --threshold 2.0

    # Get help
    python run_capacity_analysis.py --help
    python run_capacity_analysis.py baseline --help
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Import capacity modules
from capacity_calculator import export_capacity_workbook, _derive_prep_path, _derive_sections_path
from capacity_interventions import run_phase_four, visualize_enhanced_network
import settings
import paths

# Define absolute paths (consistent with capacity_calculator.py)
DATA_ROOT = Path(paths.MAIN) / "data" / "Network"
CAPACITY_ROOT = DATA_ROOT / "capacity"


def run_baseline_workflow(network_label: str = None, visualize: bool = True) -> int:
    """
    Execute baseline (standard) capacity workflow.

    Uses corridor-filtered stations (points_corridor.gpkg).

    Args:
        network_label: Network label (e.g., "2024_extended"). If None, uses settings.rail_network.
        visualize: Whether to generate plots after sections calculation

    Returns:
        Exit code (0 for success, 1 for error)
    """
    print("\n" + "=" * 70)
    print("BASELINE WORKFLOW - Standard (Corridor-Filtered)")
    print("=" * 70)

    # Use settings.rail_network if no network label provided
    if network_label is None:
        network_label = settings.rail_network

    print(f"\nNetwork: {network_label}")
    print(f"Stations: points_corridor.gpkg (corridor boundary only)")
    print(f"Edges: edges_in_corridor.gpkg")
    print("\n" + "-" * 70 + "\n")

    # Temporarily set network label
    original_network = settings.rail_network
    settings.rail_network = network_label

    try:
        # Generate capacity workbook with manual enrichment prompts
        print("Step 1: Generating capacity workbook...")
        print("NOTE: You will be prompted to manually enrich the workbook.\n")

        output_path = export_capacity_workbook(
            edges_path=None,  # Baseline mode
            network_label=None,
            enrichment_source=None,
            skip_manual_checkpoint=False  # Enable manual prompts
        )

        print(f"\n✓ Capacity workbook generated: {output_path}")

        # Check if sections were generated
        sections_path = _derive_sections_path(output_path)

        if not sections_path.exists():
            print("\n⚠ Sections workbook not generated (manual enrichment may be incomplete)")
            print("Please complete manual enrichment and re-run if needed.")
            return 0

        print(f"✓ Sections workbook generated: {sections_path}")

        # Visualization
        if visualize:
            print("\nStep 2: Generating visualizations...")
            try:
                from network_plot import (
                    plot_capacity_network,
                    plot_speed_profile_network,
                    plot_service_network
                )

                plot_capacity_network(workbook_path=str(sections_path), network_label=network_label)
                print("  ✓ Capacity network plot")

                plot_speed_profile_network(workbook_path=str(sections_path), network_label=network_label)
                print("  ✓ Speed profile plot")

                plot_service_network(workbook_path=str(sections_path), network_label=network_label)
                print("  ✓ Service network plot")

            except Exception as e:
                print(f"  ⚠ Warning: Visualization failed: {e}")

        print("\n" + "=" * 70)
        print("✓ BASELINE WORKFLOW COMPLETE")
        print("=" * 70)
        print(f"\nOutput files:")
        print(f"  - {output_path}")
        print(f"  - {_derive_prep_path(output_path)}")
        print(f"  - {sections_path}")
        print(f"\nPlots saved to: plots/network/{network_label}/\n")

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        settings.rail_network = original_network


def run_baseline_extended_workflow(network_label: str = None, visualize: bool = True) -> int:
    """
    Execute baseline extended capacity workflow.

    Uses all stations (points.gpkg), no corridor filtering.

    Args:
        network_label: Network label (e.g., "2024_extended").
                      If None, uses settings.rail_network (appends "_extended" if not present).
        visualize: Whether to generate plots after sections calculation

    Returns:
        Exit code (0 for success, 1 for error)
    """
    print("\n" + "=" * 70)
    print("BASELINE EXTENDED WORKFLOW - All Stations (No Corridor Filter)")
    print("=" * 70)

    # Use settings.rail_network + "_extended" if no network label provided
    # Check if _extended suffix already exists to avoid duplication
    if network_label is None:
        base = settings.rail_network
        network_label = base if base.endswith('_extended') else f"{base}_extended"

    print(f"\nNetwork: {network_label}")
    print(f"Stations: points.gpkg (all stations)")
    print(f"Edges: edges_in_corridor.gpkg")
    print("\n" + "-" * 70 + "\n")

    # Temporarily set network label
    original_network = settings.rail_network
    settings.rail_network = network_label

    try:
        # Generate capacity workbook with manual enrichment prompts
        print("Step 1: Generating capacity workbook...")
        print("NOTE: You will be prompted to manually enrich the workbook.\n")

        output_path = export_capacity_workbook(
            edges_path=None,  # Baseline mode
            network_label=None,
            enrichment_source=None,
            skip_manual_checkpoint=False  # Enable manual prompts
        )

        print(f"\n✓ Capacity workbook generated: {output_path}")

        # Check if sections were generated
        sections_path = _derive_sections_path(output_path)

        if not sections_path.exists():
            print("\n⚠ Sections workbook not generated (manual enrichment may be incomplete)")
            print("Please complete manual enrichment and re-run if needed.")
            return 0

        print(f"✓ Sections workbook generated: {sections_path}")

        # Visualization
        if visualize:
            print("\nStep 2: Generating visualizations...")
            try:
                from network_plot import (
                    plot_capacity_network,
                    plot_speed_profile_network,
                    plot_service_network
                )

                plot_capacity_network(workbook_path=str(sections_path), network_label=network_label)
                print("  ✓ Capacity network plot")

                plot_speed_profile_network(workbook_path=str(sections_path), network_label=network_label)
                print("  ✓ Speed profile plot")

                plot_service_network(workbook_path=str(sections_path), network_label=network_label)
                print("  ✓ Service network plot")

            except Exception as e:
                print(f"  ⚠ Warning: Visualization failed: {e}")

        print("\n" + "=" * 70)
        print("✓ BASELINE EXTENDED WORKFLOW COMPLETE")
        print("=" * 70)
        print(f"\nOutput files:")
        print(f"  - {output_path}")
        print(f"  - {_derive_prep_path(output_path)}")
        print(f"  - {sections_path}")
        print(f"\nPlots saved to: plots/network/{network_label}/\n")

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        settings.rail_network = original_network


def run_development_workflow(
    dev_id: str,
    base_network: str = None,
    visualize: bool = True
) -> int:
    """
    Execute development capacity workflow with auto-enrichment.

    Args:
        dev_id: Development ID (e.g., "101032.0") - Required
        base_network: Base network label (e.g., "2024_extended").
                     If None, uses settings.rail_network.
        visualize: Whether to generate plots after sections calculation

    Returns:
        Exit code (0 for success, 1 for error)
    """
    print("\n" + "=" * 70)
    print("DEVELOPMENT WORKFLOW - Selective Enhanced Enrichment")
    print("=" * 70)

    # Use settings.rail_network if no base network provided
    if base_network is None:
        base_network = settings.rail_network

    print(f"\nBase network: {base_network}")
    print(f"Development ID: {dev_id}")

    # Construct paths
    dev_edges_path = DATA_ROOT / "processed" / "developments" / f"{dev_id}.gpkg"
    baseline_prep_path = CAPACITY_ROOT / "Baseline" / base_network / f"capacity_{base_network}_network_prep.xlsx"

    # Fallback to old structure if new structure doesn't exist
    if not baseline_prep_path.exists():
        baseline_prep_path = CAPACITY_ROOT / base_network / f"capacity_{base_network}_network_prep.xlsx"

    print(f"\nDevelopment edges: {dev_edges_path}")
    print(f"Baseline prep: {baseline_prep_path}")
    print(f"Enhanced baseline: Auto-detected (selective enrichment based on capacity demand increases)")
    print("\n" + "-" * 70 + "\n")

    # Validation
    if not dev_edges_path.exists():
        print(f"❌ ERROR: Development edges file not found:")
        print(f"  {dev_edges_path}")
        print("\nPlease ensure the development file exists.")
        return 1

    if not baseline_prep_path.exists():
        print(f"❌ ERROR: Baseline prep workbook not found:")
        print(f"  {baseline_prep_path}")
        print("\nPlease run the baseline workflow first and complete manual enrichment.")
        return 1

    # Temporarily set network label
    original_network = settings.rail_network
    settings.rail_network = base_network

    try:
        network_label = f"{base_network}_dev_{dev_id}"

        print("Step 1: Generating auto-enriched capacity workbook...")
        print("NOTE: Baseline infrastructure used for unchanged services.")
        print("      Enhanced baseline infrastructure used for segments/stations with capacity demand increases.")
        print("      You may be prompted to fill NEW infrastructure only.\n")

        output_path = export_capacity_workbook(
            edges_path=dev_edges_path,
            network_label=network_label,
            enrichment_source=baseline_prep_path,
            skip_manual_checkpoint=False  # Enable prompts for NEW infrastructure
        )

        print(f"\n✓ Capacity workbook generated: {output_path}")

        # Check if sections were generated
        sections_path = _derive_sections_path(output_path)

        if not sections_path.exists():
            print("\n⚠ Sections workbook not generated (manual enrichment may be incomplete)")
            print("Please complete manual enrichment for NEW infrastructure and re-run if needed.")
            return 0

        print(f"✓ Sections workbook generated: {sections_path}")

        # Visualization
        if visualize:
            print("\nStep 2: Generating visualizations...")
            try:
                from network_plot import (
                    plot_capacity_network,
                    plot_speed_profile_network,
                    plot_service_network
                )

                plot_capacity_network(workbook_path=str(sections_path), network_label=network_label)
                print("  ✓ Capacity network plot")

                plot_speed_profile_network(workbook_path=str(sections_path), network_label=network_label)
                print("  ✓ Speed profile plot")

                plot_service_network(workbook_path=str(sections_path), network_label=network_label)
                print("  ✓ Service network plot")

            except Exception as e:
                print(f"  ⚠ Warning: Visualization failed: {e}")

        print("\n" + "=" * 70)
        print("✓ DEVELOPMENT WORKFLOW COMPLETE")
        print("=" * 70)
        print(f"\nOutput files:")
        print(f"  - {output_path}")
        print(f"  - {_derive_prep_path(output_path)}")
        print(f"  - {sections_path}")
        print(f"\nPlots saved to: plots/network/Developments/{dev_id}/\n")

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        settings.rail_network = original_network


def run_enhanced_workflow(
    network_label: str = None,
    threshold: float = 2.0,
    max_iterations: int = 10
) -> int:
    """
    Execute Phase 4 enhanced capacity workflow (capacity interventions).

    Args:
        network_label: Base network label (e.g., "2024_extended"). If None, uses settings.rail_network.
        threshold: Minimum required available capacity in tphpd (default: 2.0)
        max_iterations: Maximum intervention iterations (default: 10)

    Returns:
        Exit code (0 for success, 1 for error)
    """
    print("\n" + "=" * 70)
    print("ENHANCED WORKFLOW - Phase 4 Capacity Interventions")
    print("=" * 70)

    # Use settings.rail_network if no network label provided
    if network_label is None:
        network_label = settings.rail_network

    print(f"\nBase network: {network_label}")
    print(f"Threshold: ≥{threshold} tphpd available capacity")
    print(f"Max iterations: {max_iterations}")

    # Input paths from Baseline subdirectory
    base_capacity_dir = CAPACITY_ROOT / "Baseline" / network_label
    prep_workbook_path = base_capacity_dir / f"capacity_{network_label}_network_prep.xlsx"
    sections_workbook_path = base_capacity_dir / f"capacity_{network_label}_network_sections.xlsx"

    # Fallback to old structure
    if not prep_workbook_path.exists():
        base_capacity_dir = CAPACITY_ROOT / network_label
        prep_workbook_path = base_capacity_dir / f"capacity_{network_label}_network_prep.xlsx"
        sections_workbook_path = base_capacity_dir / f"capacity_{network_label}_network_sections.xlsx"

    # Output paths to Enhanced subdirectory
    output_dir = CAPACITY_ROOT / "Enhanced" / f"{network_label}_enhanced"

    print(f"\nInput:")
    print(f"  Prep workbook: {prep_workbook_path}")
    print(f"  Sections: {sections_workbook_path}")
    print(f"\nOutput directory: {output_dir}")
    print("\n" + "-" * 70 + "\n")

    # Validation
    if not prep_workbook_path.exists():
        print(f"❌ ERROR: Baseline prep workbook not found:")
        print(f"  {prep_workbook_path}")
        print("\nPlease run the baseline workflow first.")
        return 1

    if not sections_workbook_path.exists():
        print(f"❌ ERROR: Sections workbook not found:")
        print(f"  {sections_workbook_path}")
        print("\nPlease complete Phase 3 (sections calculation) first.")
        return 1

    # User confirmation
    response = input("Proceed with Phase 4 execution? (y/n): ")
    if response.lower() not in ['y', 'yes']:
        print("Aborted by user.")
        return 0

    print()

    try:
        # Load Phase 3 outputs
        print("Loading Phase 3 outputs...")
        stations_df = pd.read_excel(prep_workbook_path, sheet_name='Stations')
        segments_df = pd.read_excel(prep_workbook_path, sheet_name='Segments')
        sections_df = pd.read_excel(sections_workbook_path, sheet_name='Sections')

        print(f"  ✓ Loaded {len(stations_df)} stations")
        print(f"  ✓ Loaded {len(segments_df)} segments")
        print(f"  ✓ Loaded {len(sections_df)} sections\n")

        # Execute Phase 4
        output_dir.mkdir(parents=True, exist_ok=True)

        interventions_catalog, enhanced_prep_path, final_sections_df = run_phase_four(
            original_sections_df=sections_df,
            original_segments_df=segments_df,
            original_stations_df=stations_df,
            prep_workbook_path=prep_workbook_path,
            output_dir=output_dir,
            network_label=network_label,
            threshold_tphpd=threshold,
            max_iterations=max_iterations
        )

        # Reload the ENHANCED stations and segments data from the enhanced prep workbook
        print("\nReloading enhanced infrastructure data...")
        enhanced_stations_df = pd.read_excel(enhanced_prep_path, sheet_name='Stations')
        enhanced_segments_df = pd.read_excel(enhanced_prep_path, sheet_name='Segments')
        print(f"  ✓ Loaded {len(enhanced_stations_df)} enhanced stations")
        print(f"  ✓ Loaded {len(enhanced_segments_df)} enhanced segments")

        # Update the sections workbook with enhanced data
        enhanced_sections_path = output_dir / f"capacity_{network_label}_enhanced_network_sections.xlsx"
        
        print(f"\nUpdating sections workbook with enhanced infrastructure...")
        with pd.ExcelWriter(enhanced_sections_path, engine='openpyxl') as writer:
            enhanced_stations_df.to_excel(writer, sheet_name='Stations', index=False)
            enhanced_segments_df.to_excel(writer, sheet_name='Segments', index=False)
            final_sections_df.to_excel(writer, sheet_name='Sections', index=False)
        
        print(f"  ✓ Saved enhanced sections workbook: {enhanced_sections_path}")

        # Generate visualizations
        print("\n" + "-" * 70)
        print("Generating visualizations...")

        infrastructure_plot, capacity_plot = visualize_enhanced_network(
            enhanced_prep_path=enhanced_prep_path,
            enhanced_sections_path=enhanced_sections_path,
            interventions_list=interventions_catalog,
            network_label=f"{network_label}_enhanced"
        )

        print("✓ Visualizations complete")

        print("\n" + "=" * 70)
        print("✓ ENHANCED WORKFLOW COMPLETE (PHASE 4)")
        print("=" * 70)
        print(f"\nOutput files:")
        print(f"  - {enhanced_prep_path}")
        print(f"  - {enhanced_sections_path}")
        print(f"  - {output_dir / 'capacity_interventions.csv'}")
        print(f"\nPlots saved to: plots/network/{network_label}_enhanced/\n")

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main_interactive():
    """Interactive menu for workflow selection."""
    print("\n" + "=" * 70)
    print("RAIL NETWORK CAPACITY ANALYSIS TOOL")
    print("=" * 70)
    print("\nAvailable Workflows:")
    print("  1. Baseline - Standard baseline (corridor-filtered stations)")
    print("  2. Baseline Extended - All stations (no corridor filtering)")
    print("  3. Development - Development network with auto-enrichment")
    print("  4. Enhanced - Phase 4 capacity interventions")
    print("  0. Exit")
    print("=" * 70)

    while True:
        choice = input("\nSelect workflow (0-4): ").strip()

        if choice == '0':
            print("Exiting...")
            return 0

        elif choice == '1':
            # Baseline workflow
            print("\n--- Baseline Workflow Configuration ---")
            network = input(f"Network label [{settings.rail_network}]: ").strip() or settings.rail_network
            visualize_input = input("Generate visualizations? (y/n) [y]: ").strip().lower()
            visualize = visualize_input != 'n'

            return run_baseline_workflow(
                network_label=network,
                visualize=visualize
            )

        elif choice == '2':
            # Baseline Extended workflow
            print("\n--- Baseline Extended Workflow Configuration ---")
            base = settings.rail_network
            default_extended = base if base.endswith('_extended') else f"{base}_extended"
            network = input(f"Network label [{default_extended}]: ").strip() or default_extended
            visualize_input = input("Generate visualizations? (y/n) [y]: ").strip().lower()
            visualize = visualize_input != 'n'

            return run_baseline_extended_workflow(
                network_label=network,
                visualize=visualize
            )

        elif choice == '3':
            # Development workflow
            print("\n--- Development Workflow Configuration ---")
            dev_id = input("Development ID (e.g., 101032.0): ").strip()

            if not dev_id:
                print("❌ ERROR: Development ID is required")
                continue

            default_base = settings.rail_network
            base_network = input(f"Base network label [{default_base}]: ").strip() or default_base
            visualize_input = input("Generate visualizations? (y/n) [y]: ").strip().lower()
            visualize = visualize_input != 'n'

            return run_development_workflow(
                dev_id=dev_id,
                base_network=base_network,
                visualize=visualize
            )

        elif choice == '4':
            # Enhanced workflow (Phase 4)
            print("\n--- Enhanced Workflow (Phase 4) Configuration ---")
            network = input(f"Base network label [{settings.rail_network}]: ").strip() or settings.rail_network

            threshold_input = input("Minimum available capacity threshold in tphpd [2.0]: ").strip()
            try:
                threshold = float(threshold_input) if threshold_input else 2.0
            except ValueError:
                print("⚠ Invalid threshold, using default: 2.0")
                threshold = 2.0

            max_iter_input = input("Maximum iterations [10]: ").strip()
            try:
                max_iterations = int(max_iter_input) if max_iter_input else 10
            except ValueError:
                print("⚠ Invalid max iterations, using default: 10")
                max_iterations = 10

            return run_enhanced_workflow(
                network_label=network,
                threshold=threshold,
                max_iterations=max_iterations
            )

        else:
            print("❌ Invalid choice. Please select 0-4.")


def main():
    """Main entry point - supports both interactive and CLI modes."""
    # Check if command-line arguments were provided
    if len(sys.argv) > 1:
        # CLI mode with argparse
        parser = argparse.ArgumentParser(
            description="Rail network capacity analysis tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Interactive mode (no arguments)
  %(prog)s

  # CLI mode with arguments
  %(prog)s baseline
  %(prog)s baseline-extended --network 2024_extended
  %(prog)s development --dev-id 101032.0 --base-network 2024_extended
  %(prog)s enhanced --network 2024_extended --threshold 2.0
"""
        )

        # Subcommands
        subparsers = parser.add_subparsers(dest='workflow', help='Workflow type', required=True)

        # Baseline workflow
        baseline_parser = subparsers.add_parser(
            'baseline',
            help='Baseline workflow (corridor-filtered stations)'
        )
        baseline_parser.add_argument(
            '--network',
            default=None,
            help=f'Network label (default: {settings.rail_network})'
        )
        baseline_parser.add_argument(
            '--no-visualize',
            action='store_true',
            help='Skip visualization generation'
        )

        # Baseline extended workflow
        baseline_ext_parser = subparsers.add_parser(
            'baseline-extended',
            help='Baseline extended workflow (all stations, no corridor filter)'
        )
        baseline_ext_parser.add_argument(
            '--network',
            default=None,
            help=f'Network label (default: {settings.rail_network if settings.rail_network.endswith("_extended") else settings.rail_network + "_extended"})'
        )
        baseline_ext_parser.add_argument(
            '--no-visualize',
            action='store_true',
            help='Skip visualization generation'
        )

        # Development workflow
        dev_parser = subparsers.add_parser(
            'development',
            help='Development workflow with auto-enrichment'
        )
        dev_parser.add_argument(
            '--dev-id',
            required=True,
            help='Development ID (e.g., 101032.0)'
        )
        dev_parser.add_argument(
            '--base-network',
            default=None,
            help=f'Base network label (default: {settings.rail_network})'
        )
        dev_parser.add_argument(
            '--no-visualize',
            action='store_true',
            help='Skip visualization generation'
        )

        # Enhanced workflow (Phase 4)
        enhanced_parser = subparsers.add_parser(
            'enhanced',
            help='Enhanced workflow - Phase 4 capacity interventions'
        )
        enhanced_parser.add_argument(
            '--network',
            default=None,
            help=f'Base network label (default: {settings.rail_network})'
        )
        enhanced_parser.add_argument(
            '--threshold',
            type=float,
            default=2.0,
            help='Minimum required available capacity in tphpd (default: 2.0)'
        )
        enhanced_parser.add_argument(
            '--max-iterations',
            type=int,
            default=10,
            help='Maximum intervention iterations (default: 10)'
        )

        args = parser.parse_args()

        # Dispatch to appropriate workflow
        if args.workflow == 'baseline':
            exit_code = run_baseline_workflow(
                network_label=args.network,
                visualize=not args.no_visualize
            )

        elif args.workflow == 'baseline-extended':
            exit_code = run_baseline_extended_workflow(
                network_label=args.network,
                visualize=not args.no_visualize
            )

        elif args.workflow == 'development':
            exit_code = run_development_workflow(
                dev_id=args.dev_id,
                base_network=args.base_network,
                visualize=not args.no_visualize
            )

        elif args.workflow == 'enhanced':
            exit_code = run_enhanced_workflow(
                network_label=args.network,
                threshold=args.threshold,
                max_iterations=args.max_iterations
            )

        else:
            print(f"Unknown workflow: {args.workflow}")
            exit_code = 1

        sys.exit(exit_code)

    else:
        # Interactive mode (no command-line arguments)
        exit_code = main_interactive()
        sys.exit(exit_code)


if __name__ == "__main__":
    main()