# import packages
import paths
import scoring
import settings
import re
from TT_Delay import *
from catchment_pt import *
from display_results import *
from generate_infrastructure import *
from paths import get_rail_services_path
from scenarios import *
from scoring import *
from scoring import create_cost_and_benefit_df
from traveltime_delay import *
from random_scenarios import get_random_scenarios
from plots import *
from run_capacity_analysis import (
    run_baseline_workflow,
    run_baseline_extended_workflow,
    run_enhanced_workflow,
    run_development_workflow,
    CAPACITY_ROOT
)
import geopandas as gpd
import pandas as pd
import os
import warnings
import cost_parameters as cp
import plot_parameter as pp
import json
import time
import pickle
from pathlib import Path


# ================================================================================
# GLOBAL PIPELINE CONFIGURATION
# ================================================================================

class PipelineConfig:
    """Global configuration for pipeline execution."""
    
    def __init__(self):
        self.visualization_mode = None  # 'manual', 'none', 'all'
        self.grouping_strategy = None   # 'manual', 'conservative', 'baseline', 'optimal'
        self._original_input = None
    
    def should_generate_plots(self, default_yes: bool = False) -> bool:
        """
        Determine if plots should be generated based on global setting.
        
        Args:
            default_yes: Default answer if mode is manual
        
        Returns:
            bool: True if plots should be generated, None if should prompt
        """
        if self.visualization_mode == 'all':
            return True
        elif self.visualization_mode == 'none':
            return False
        else:  # manual
            return None  # Signal to prompt user
    
    def get_grouping_choice(self, prompt: str) -> str:
        """
        Get grouping strategy choice based on global setting.
        
        Detects available choices from the prompt and selects appropriately.
        
        Args:
            prompt: The original prompt text
        
        Returns:
            str: The choice to make
        """
        if self.grouping_strategy == 'manual':
            # Use original input
            return self._original_input(prompt)
        
        # Detect available choices from prompt
        # Look for patterns like "(1-2)" or "(1-3)" in the prompt      
        choice_match = re.search(r'\(1-(\d+)\)', prompt)
        if choice_match:
            max_choice = int(choice_match.group(1))
            available_choices = [str(i) for i in range(1, max_choice + 1)]
        else:
            # Fallback: assume 1-2 if we can't detect
            available_choices = ['1', '2']
        
        # Select based on strategy
        if self.grouping_strategy == 'conservative':
            # Always choose lowest option (1)
            choice = '1'
            print(prompt + f"{choice}  [AUTO-SELECTED: Conservative]")
        elif self.grouping_strategy == 'baseline':
            # Always choose option 2 if available, otherwise lowest
            choice = '2' if '2' in available_choices else '1'
            print(prompt + f"{choice}  [AUTO-SELECTED: Baseline]")
        elif self.grouping_strategy == 'optimal':
            # Always choose highest option
            choice = max(available_choices, key=int)
            print(prompt + f"{choice}  [AUTO-SELECTED: Optimal]")
        else:
            # Fallback to manual
            return self._original_input(prompt)
        
        return choice


# Global configuration instance
PIPELINE_CONFIG = PipelineConfig()


# ================================================================================
# PHASE FUNCTIONS - Modular Pipeline Components
# ================================================================================

def phase_1_initialization(runtimes: dict) -> tuple:
    """
    Phase 1: Initialize workspace and study area boundaries.
    Also configures global pipeline settings.

    Args:
        runtimes: Dictionary to track phase execution times

    Returns:
        tuple: (innerboundary, outerboundary) - Study area polygons
    """
    print("\n" + "="*80)
    print("PHASE 1: INITIALIZE VARIABLES & PIPELINE CONFIGURATION")
    print("="*80 + "\n")
    st = time.time()

    # ============================================================================
    # GLOBAL CONFIGURATION PROMPTS
    # ============================================================================
    print("PIPELINE CONFIGURATION")
    print("-" * 80)
    
    # A. Visualization Strategy
    print("\nA. VISUALIZATION STRATEGY")
    print("   How should the pipeline handle optional plot generation?")
    print("   1) Manual  - Prompt for each visualization decision (default)")
    print("   2) None    - Skip all optional visualizations")
    print("   3) All     - Generate all optional visualizations")
    
    while True:
        viz_choice = input("\n   Select visualization mode (1-3) [1]: ").strip() or "1"

        if viz_choice in ['1', '2', '3']:
            break
        print("   Invalid selection. Please enter 1, 2, or 3.")
    
    viz_modes = {'1': 'manual', '2': 'none', '3': 'all'}
    PIPELINE_CONFIG.visualization_mode = viz_modes[viz_choice]
    print(f"   → Visualization mode: {PIPELINE_CONFIG.visualization_mode.upper()}")
    
    # B. Grouping Strategy
    print("\nB. CAPACITY GROUPING STRATEGY")
    print("   How should capacity grouping decisions be made?")
    print("   1) Manual       - Prompt for each grouping decision (default)")
    print("   2) Conservative - Always choose lowest capacity option")
    print("   3) Baseline     - Always choose middle option (2)")
    print("   4) Optimal      - Always choose highest capacity option")
    
    while True:
        group_choice = input("\n   Select grouping strategy (1-4) [1]: ").strip() or "1"
        if group_choice in ['1', '2', '3', '4']:
            break
        print("   Invalid selection. Please enter 1, 2, 3, or 4.")
    
    group_modes = {'1': 'manual', '2': 'conservative', '3': 'baseline', '4': 'optimal'}
    PIPELINE_CONFIG.grouping_strategy = group_modes[group_choice]
    print(f"   → Grouping strategy: {PIPELINE_CONFIG.grouping_strategy.upper()}")
    
    print("-" * 80)
    print("✓ Pipeline configuration complete\n")

    # ============================================================================
    # WORKSPACE INITIALIZATION
    # ============================================================================
    innerboundary, outerboundary = create_focus_area()

    runtimes["Initialize variables"] = time.time() - st
    return innerboundary, outerboundary


def phase_2_data_import(runtimes: dict) -> None:
    """
    Phase 2: Import raw geographic data (lakes, cities).

    Args:
        runtimes: Dictionary to track phase execution times

    Side Effects:
        - Writes lake_data_zh.gpkg
        - Writes cities.shp
    """
    print("\n" + "="*80)
    print("PHASE 2: IMPORT RAW DATA")
    print("="*80 + "\n")
    st = time.time()

    # Import shapes of lake for plots
    get_lake_data()

    # Import the file containing the locations to be plotted
    import_cities()

    # Define area that is protected for constructing railway links
    #   get_protected_area(limits=limits_corridor)
    #   get_unproductive_area(limits=limits_corridor)
    #   landuse(limits=limits_corridor)

    # Tif file of all unsuitable land cover and protected areas
    # File is stored to 'data\\landuse_landcover\\processed\\zone_no_infra\\protected_area_{suffix}.tif'

    # all_protected_area_to_raster(suffix="corridor")

    runtimes["Import land use and land cover data"] = time.time() - st


def phase_3_baseline_capacity_analysis(runtimes: dict) -> tuple:
    """
    Phase 3: Baseline capacity analysis (3 sub-steps).

    Sub-steps:
        3.1: Import and process base network
        3.2: Establish baseline capacity
        3.3: Enhance baseline network (Phase 4 interventions)

    Args:
        runtimes: Dictionary to track phase execution times

    Returns:
        tuple: (points, baseline_prep_path, baseline_sections_path, enhanced_network_label)
            - points: Station points GeoDataFrame
            - baseline_prep_path: Path to baseline prep workbook
            - baseline_sections_path: Path to baseline sections workbook
            - enhanced_network_label: Label for enhanced network
    """
    print("\n" + "="*80)
    print("PHASE 3: BASELINE CAPACITY ANALYSIS")
    print("="*80 + "\n")

    # ============================================================================
    # STEP 3.1: IMPORT AND PROCESS BASE NETWORK
    # ============================================================================
    print("\n--- Step 3.1: Import and Process Base Network ---\n")
    st = time.time()

    points = import_process_network(settings.use_cache_network)

    runtimes["Preprocess the network"] = time.time() - st

    # ============================================================================
    # STEP 3.2: ESTABLISH BASELINE CAPACITY 
    # ============================================================================
    print("\n--- Step 3.2: Establish Baseline Capacity ---\n")
    st = time.time()

    # Determine visualization setting
    visualize_baseline = PIPELINE_CONFIG.should_generate_plots(default_yes=True)
    if visualize_baseline is None:  # Manual mode
        response = input("Generate baseline capacity visualizations? (y/n) [y]: ").strip().lower()
        visualize_baseline = response != 'n'
    
    if visualize_baseline:
        print("  → Baseline visualizations will be generated")
    else:
        print("  → Skipping baseline visualizations")

    # Auto-select workflow based on network label
    if '_extended' in settings.rail_network:
        print(f"  Using Baseline Extended workflow (all stations) for {settings.rail_network}")
        baseline_exit_code = run_baseline_extended_workflow(
            network_label=settings.rail_network,
            visualize=visualize_baseline
        )
    else:
        print(f"  Using Baseline workflow (corridor-filtered) for {settings.rail_network}")
        baseline_exit_code = run_baseline_workflow(
            network_label=settings.rail_network,
            visualize=visualize_baseline
        )

    if baseline_exit_code != 0:
        raise RuntimeError(
            "Baseline capacity analysis failed. "
            "Please check manual enrichment steps and ensure prep workbook is complete."
        )

    # Store paths for later use
    baseline_capacity_dir = CAPACITY_ROOT / "Baseline" / settings.rail_network
    baseline_prep_path = baseline_capacity_dir / f"capacity_{settings.rail_network}_network_prep.xlsx"
    baseline_sections_path = baseline_capacity_dir / f"capacity_{settings.rail_network}_network_sections.xlsx"

    # Fallback to old structure if new structure doesn't exist
    if not baseline_prep_path.exists():
        baseline_capacity_dir = CAPACITY_ROOT / settings.rail_network
        baseline_prep_path = baseline_capacity_dir / f"capacity_{settings.rail_network}_network_prep.xlsx"
        baseline_sections_path = baseline_capacity_dir / f"capacity_{settings.rail_network}_network_sections.xlsx"

    # Validate files exist
    if not baseline_prep_path.exists() or not baseline_sections_path.exists():
        raise FileNotFoundError(
            f"Baseline capacity files not found in {baseline_capacity_dir}\n"
            f"Expected:\n  {baseline_prep_path}\n  {baseline_sections_path}"
        )

    print(f"  ✓ Baseline capacity established: {baseline_sections_path}")

    runtimes["Establish baseline capacity"] = time.time() - st

    # ============================================================================
    # STEP 3.3: ENHANCE BASELINE NETWORK (PHASE 4 INTERVENTIONS)
    # ============================================================================
    print("\n--- Step 3.3: Enhance Baseline Network (Phase 4) ---\n")
    st = time.time()

    # Prompt user for capacity enhancement parameters
    print(f"  Network to enhance: {settings.rail_network}")
    print(f"\n  Configure capacity enhancement parameters:")
    print(f"  Default threshold: {settings.capacity_threshold} tphpd")
    print(f"  Default max iterations: {settings.max_enhancement_iterations}")

    # Get threshold from user
    threshold_input = input(f"\n  Enter capacity threshold (tphpd) or press Enter for default [{settings.capacity_threshold}]: ").strip()
    if threshold_input:
        try:
            capacity_threshold = float(threshold_input)
            print(f"  → Using threshold: {capacity_threshold} tphpd")
        except ValueError:
            print(f"  ⚠ Invalid input. Using default: {settings.capacity_threshold} tphpd")
            capacity_threshold = settings.capacity_threshold
    else:
        capacity_threshold = settings.capacity_threshold
        print(f"  → Using default threshold: {capacity_threshold} tphpd")

    # Get max iterations from user
    iterations_input = input(f"  Enter max iterations or press Enter for default [{settings.max_enhancement_iterations}]: ").strip()
    if iterations_input:
        try:
            max_iterations = int(iterations_input)
            print(f"  → Using max iterations: {max_iterations}")
        except ValueError:
            print(f"  ⚠ Invalid input. Using default: {settings.max_enhancement_iterations}")
            max_iterations = settings.max_enhancement_iterations
    else:
        max_iterations = settings.max_enhancement_iterations
        print(f"  → Using default max iterations: {max_iterations}")

    # Determine visualization setting for enhanced network
    visualize_enhanced = PIPELINE_CONFIG.should_generate_plots(default_yes=True)
    if visualize_enhanced is None:  # Manual mode
        response = input("Generate enhanced network visualizations? (y/n) [y]: ").strip().lower()
        visualize_enhanced = response != 'n'
    
    if visualize_enhanced:
        print("  → Enhanced network visualizations will be generated")
    else:
        print("  → Skipping enhanced network visualizations")

    # Run Phase 4 iterative capacity enhancement
    print(f"\n  Running Phase 4 enhancement workflow for {settings.rail_network}...")
    print(f"  Threshold: {capacity_threshold} tphpd")
    print(f"  Max iterations: {max_iterations}\n")

    enhanced_exit_code = run_enhanced_workflow(
        network_label=settings.rail_network,
        threshold=capacity_threshold,
        max_iterations=max_iterations,
    )

    if enhanced_exit_code != 0:
        raise RuntimeError(
            "Phase 4 enhancement workflow failed. "
            "Check intervention design and manual enrichment steps."
        )

    # Determine enhanced network label
    enhanced_network_label = f"{settings.rail_network}_enhanced"

    # NOTE: Development workflow uses the BASELINE network for enrichment
    # (run_capacity_analysis.py only looks in Baseline/ directory, not Enhanced/)
    # The enhanced baseline is for reference/validation purposes only
    settings.baseline_network_for_developments = settings.rail_network  # Use base, not enhanced

    print(f"\n  ✓ Baseline enhancement complete")
    print(f"  Enhanced network: {enhanced_network_label}")
    print(f"  → Developments will use baseline network ({settings.rail_network}) for enrichment\n")

    runtimes["Enhance baseline network"] = time.time() - st

    return points, baseline_prep_path, baseline_sections_path, enhanced_network_label


def phase_4_infrastructure_developments(points: gpd.GeoDataFrame, runtimes: dict) -> tuple:
    """
    Phase 4: Infrastructure developments (4 sub-steps).

    Sub-steps:
        4.1: Generate infrastructure developments
        4.2: Analyze development capacity requirements
        4.3: Extract capacity intervention costs
        4.4: Public transit catchment (optional)

    Args:
        points: Station points from Phase 3
        runtimes: Dictionary to track phase execution times

    Returns:
        tuple: (dev_id_lookup, capacity_analysis_results)
            - dev_id_lookup: Development ID lookup table DataFrame
            - capacity_analysis_results: Dict with capacity analysis results for each development

    Side Effects:
        - Writes capacity_intervention_costs.csv to data/costs/
    """
    print("\n" + "="*80)
    print("PHASE 4: INFRASTRUCTURE DEVELOPMENTS")
    print("="*80 + "\n")

    # ============================================================================
    # STEP 4.1: GENERATE INFRASTRUCTURE DEVELOPMENTS
    # ============================================================================
    print("\n--- Step 4.1: Generate Infrastructure Developments ---\n")
    st = time.time()

    # Determine if infrastructure generation plots should be created
    generate_infra_plots = PIPELINE_CONFIG.should_generate_plots(default_yes=False)
    if generate_infra_plots is None:  # Manual mode
        response = input("Generate infrastructure development plots (network graphs, missing connections)? (y/n) [n]: ").strip().lower()
        generate_infra_plots = response in {'y', 'yes'}

    if generate_infra_plots:
        print("  → Infrastructure development plots will be generated")
    else:
        print("  → Skipping infrastructure development plots")

    generate_infra_development(
        use_cache=settings.use_cache_developments,
        mod_type=settings.infra_generation_modification_type,
        generate_plots=generate_infra_plots
    )

    # Create lookup table for developments
    dev_id_lookup = create_dev_id_lookup_table()
    print(f"  ✓ Generated {len(dev_id_lookup)} infrastructure developments")
    runtimes["Generate infrastructure developments"] = time.time() - st

    # ============================================================================
    # STEP 4.2: ANALYZE DEVELOPMENT CAPACITY
    # ============================================================================
    print("\n--- Step 4.2: Analyze Development Capacity (Workflow 3) ---\n")

    # STEP 4.2: Determine if development visualizations should be generated
    generate_dev_plots = PIPELINE_CONFIG.should_generate_plots(default_yes=True)
    if generate_dev_plots is None:  # Manual mode
        print(f"Found {len(dev_id_lookup)} developments to analyze.")
        print("Each development can generate capacity, speed profile, and service network plots.")
        response = input("\nGenerate visualizations for all developments? (y/n) [y]: ").strip().lower()
        generate_dev_plots = response != 'n'
    
    if generate_dev_plots:
        print("  → Visualizations will be generated for each development")
    else:
        print("  → Visualizations will be skipped for all developments")

    print()
    st = time.time()

    capacity_analysis_results = {}
    failed_developments = []

    for idx, row in dev_id_lookup.iterrows():
        dev_id = row['dev_id']
        print(f"\n  [{idx+1}/{len(dev_id_lookup)}] Analyzing development {dev_id}...")

        try:
            # Run Workflow 3 (development capacity analysis with auto-enrichment)
            dev_exit_code = run_development_workflow(
                dev_id=dev_id,
                base_network=settings.baseline_network_for_developments,
                visualize=generate_dev_plots
            )

            # Handle workflow failure
            if dev_exit_code != 0:
                print(f"    ⚠ Capacity analysis workflow failed for {dev_id}")
                print(f"    → Development will proceed with base infrastructure costs only")

                capacity_analysis_results[dev_id] = {
                    'status': 'workflow_failed',
                    'use_base_costs': True
                }
                failed_developments.append(dev_id)
                continue

            # Load capacity results
            # Development workflow creates network_label as f"{base_network}_dev_{dev_id}"
            # BUT: File system omits the .0 decimal suffix from dev_id
            # AND: Sections file uses underscore format: dev_XXXXX_Y_ instead of dev_XXXXX.Y_

            # Remove .0 suffix from dev_id for file system paths
            dev_id_for_path = str(dev_id).replace('.0', '')

            # Convert dev_id format for sections filename: 101025.0 -> 100001_0
            if '.' in str(dev_id):
                dev_id_parts = str(dev_id).split('.')
                dev_id_for_sections = f"{dev_id_parts[0]}_{dev_id_parts[1]}"
            else:
                dev_id_for_sections = str(dev_id)

            # Construct paths with corrected naming
            dev_capacity_dir = CAPACITY_ROOT / "Developments" / dev_id_for_path
            dev_network_label = f"{settings.baseline_network_for_developments}_dev_{dev_id_for_sections}"
            dev_sections_path = dev_capacity_dir / f"capacity_{dev_network_label}_network_sections.xlsx"

            # DEBUG: Show what we're looking for
            print(f"    Looking for sections file: {dev_sections_path}")

            # Validate output files exist
            if not dev_sections_path.exists():
                print(f"    ⚠ Sections file not found: {dev_sections_path}")

                # Try alternate naming patterns
                alternate_patterns = [
                    dev_capacity_dir / f"capacity_{settings.baseline_network_for_developments}_dev_{dev_id}_network_sections.xlsx",
                    dev_capacity_dir / f"capacity_{settings.baseline_network_for_developments}_dev_{dev_id_for_path}_network_sections.xlsx",
                ]

                found = False
                for alt_path in alternate_patterns:
                    if alt_path.exists():
                        print(f"    ✓ Found alternate: {alt_path}")
                        dev_sections_path = alt_path
                        found = True
                        break

                if not found:
                    capacity_analysis_results[dev_id] = {'status': 'missing_sections'}
                    failed_developments.append(dev_id)
                    continue

            # Store successful results
            capacity_analysis_results[dev_id] = {
                'status': 'success',
                'sections_path': str(dev_sections_path),
                'base_network': settings.baseline_network_for_developments
            }

            print(f"    ✓ Capacity analysis complete for {dev_id}")

        except Exception as e:
            print(f"    ❌ Unexpected error analyzing {dev_id}: {e}")
            capacity_analysis_results[dev_id] = {'status': 'error', 'error': str(e)}
            failed_developments.append(dev_id)

    # Save results
    capacity_results_path = Path(paths.MAIN) / "data" / "Network" / "capacity" / "capacity_analysis_results.json"
    capacity_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(capacity_results_path, 'w') as f:
        json.dump(capacity_analysis_results, f, indent=2)

    # Summary
    successful = sum(1 for r in capacity_analysis_results.values() if r.get('status') == 'success')
    print(f"\n  {'='*70}")
    print(f"  CAPACITY ANALYSIS SUMMARY")
    print(f"  {'='*70}")
    print(f"  • Total developments:       {len(dev_id_lookup)}")
    print(f"  • Successfully analyzed:    {successful}")
    print(f"  • Failed analysis:          {len(failed_developments)}")
    print(f"  {'='*70}\n")

    runtimes["Analyze development capacity"] = time.time() - st

    # ============================================================================
    # STEP 4.3: CAPACITY INTERVENTION COST EXTRACTION 
    # ============================================================================
    print("\n--- Step 4.3: Extract Capacity Intervention Costs ---\n")
    st = time.time()

    _ = extract_capacity_intervention_costs(
        capacity_analysis_results=capacity_analysis_results,
        baseline_network_label=settings.baseline_network_for_developments
    )

    # Manual verification checkpoint
    output_csv_path = Path(paths.MAIN) / "data" / "costs" / "capacity_intervention_costs.csv"
    print("\n" + "="*80)
    print("MANUAL VERIFICATION CHECKPOINT")
    print("="*80)
    print(f"\nCapacity intervention costs have been extracted to:")
    print(f"  {output_csv_path}")
    print("\nPlease review the following:")
    print("  1. Check that intervention costs are correctly matched to developments")
    print("  2. Verify construction and maintenance costs are reasonable")
    print("  3. Make any necessary corrections directly in the CSV file")
    print("  4. Save the file and return here to continue")
    print("="*80)

    response = input("\nHave you reviewed and (if needed) corrected the intervention costs (y/n)? ").strip().lower()
    if response not in {"y", "yes"}:
        print("\nPipeline paused. Please review the intervention costs and re-run when ready.")
        print("You can resume from this point by running the pipeline again.\n")
        return dev_id_lookup, capacity_analysis_results

    runtimes["Extract capacity intervention costs"] = time.time() - st

    # ============================================================================
    # STEP 4.4: PUBLIC TRANSIT CATCHMENT (OPTIONAL)
    # ============================================================================
    if settings.OD_type == 'pt_catchment_perimeter':
        print("\n--- Step 4.4: Public Transit Catchment Analysis ---\n")
        st = time.time()
        
        get_catchment(use_cache=settings.use_cache_pt_catchment)
        
        # Use global visualization setting
        generate_catchment_plots = PIPELINE_CONFIG.should_generate_plots(default_yes=False)
        if generate_catchment_plots is None:  # Manual mode
            response = input("\nGenerate catchment visualization plots? (y/n) [n]: ").strip().lower()
            generate_catchment_plots = response in {'y', 'yes'}
        
        if generate_catchment_plots:
            create_plot_catchement()
            create_catchement_plot_time()
        
        runtimes["Public transit catchment"] = time.time() - st

    return dev_id_lookup, capacity_analysis_results


def phase_5_demand_analysis(points: gpd.GeoDataFrame, runtimes: dict) -> None:
    """
    Phase 5: Generate origin-destination demand matrix.

    Args:
        points: Station points from Phase 3
        runtimes: Dictionary to track phase execution times

    Side Effects:
        - Writes OD matrix CSV to paths.OD_STATIONS_KT_ZH_PATH
    """
    print("\n" + "="*80)
    print("PHASE 5: DEMAND ANALYSIS (OD MATRIX)")
    print("="*80 + "\n")
    st = time.time()

    if settings.OD_type == 'canton_ZH':
        # Filter points within demand perimeter
        points_in_perimeter = points[points.apply(lambda row: settings.perimeter_demand.contains(row.geometry), axis=1)]
        perimeter_stations = points_in_perimeter[['ID_point', 'NAME']].values.tolist()
        getStationOD(settings.use_cache_stationsOD, perimeter_stations, settings.only_demand_from_to_perimeter)

    elif settings.OD_type == 'pt_catchment_perimeter':
        GetCatchmentOD(settings.use_cache_catchmentOD)
    else:
        raise ValueError("OD_type must be either 'canton_ZH' or 'pt_catchment_perimeter'")

    runtimes["Generate OD matrix"] = time.time() - st


def phase_6_travel_time_computation(dev_id_lookup: pd.DataFrame, runtimes: dict) -> tuple:
    """
    Phase 6: Calculate baseline and development travel times.

    Args:
        dev_id_lookup: Development ID lookup table from Phase 4
        runtimes: Dictionary to track phase execution times

    Returns:
        tuple: (od_times_dev, od_times_status_quo, G_status_quo, G_development)
            - od_times_dev: OD times for all developments (Dict)
            - od_times_status_quo: OD times for status quo (Dict)
            - G_status_quo: NetworkX graph for status quo (List)
            - G_development: List of NetworkX graphs for developments (List)
    """
    print("\n" + "="*80)
    print("PHASE 6: TRAVEL TIME COMPUTATION")
    print("="*80 + "\n")
    st = time.time()

    od_times_dev, od_times_status_quo, G_status_quo, G_development = create_travel_time_graphs(
        settings.rail_network,
        settings.use_cache_traveltime_graph,
        dev_id_lookup
    )

    runtimes["Calculate Traveltimes for all developments"] = time.time() - st
    return od_times_dev, od_times_status_quo, G_status_quo, G_development


def phase_7_passenger_flow_visualization(G_development: list, G_status_quo: list, dev_id_lookup: pd.DataFrame, runtimes: dict) -> None:
    """Phase 7: Visualize passenger flows (optional)."""
    print("\n" + "="*80)
    print("PHASE 7: PASSENGER FLOW VISUALIZATION")
    print("="*80 + "\n")
    
    # Use global visualization setting
    generate_flow_plots = PIPELINE_CONFIG.should_generate_plots(default_yes=False)
    if generate_flow_plots is None:  # Manual mode
        print(f"Found {len(G_development)} developments for passenger flow visualization.")
        response = input("\nGenerate passenger flow plots? (y/n) [n]: ").strip().lower()
        generate_flow_plots = response in {'y', 'yes'}
    
    if not generate_flow_plots:
        print("  → Skipping passenger flow visualization")
        return
    
    print("  → Generating passenger flow plots for all developments...")
    st = time.time()

    plot_passenger_flows_on_network(G_development, G_status_quo, dev_id_lookup)

    runtimes["Compute and visualize passenger flows on network"] = time.time() - st


def phase_8_scenario_generation(runtimes: dict) -> None:
    """
    Phase 8: Generate future demand scenarios.

    Args:
        runtimes: Dictionary to track phase execution times

    Side Effects:
        - Writes scenario cache files
        - Writes scenario plots (if do_plot=True)
    """
    print("\n" + "="*80)
    print("PHASE 8: SCENARIO GENERATION")
    print("="*80 + "\n")
    st = time.time()

    # Determine if scenario plots should be generated
    generate_scenario_plots = PIPELINE_CONFIG.should_generate_plots(default_yes=False)
    if generate_scenario_plots is None:  # Manual mode
        response = input("Generate scenario visualization plots? (y/n) [n]: ").strip().lower()
        generate_scenario_plots = response in {'y', 'yes'}

    if generate_scenario_plots:
        print("  → Scenario plots will be generated")
    else:
        print("  → Skipping scenario plots")

    get_random_scenarios(
        start_year=2018,
        end_year=2100,
        num_of_scenarios=settings.amount_of_scenarios,
        use_cache=settings.use_cache_scenarios,
        do_plot=generate_scenario_plots
    )

    runtimes["Generate the scenarios"] = time.time() - st


def phase_9_travel_time_savings(dev_id_lookup: pd.DataFrame, od_times_dev: dict, od_times_status_quo: dict, runtimes: dict) -> tuple:
    """
    Phase 9: Monetize travel time savings.

    Args:
        dev_id_lookup: Development ID lookup table from Phase 4
        od_times_dev: OD times for developments from Phase 6
        od_times_status_quo: OD times for status quo from Phase 6
        runtimes: Dictionary to track phase execution times

    Returns:
        tuple: (dev_list, monetized_tt, scenario_list)
            - dev_list: List of development IDs
            - monetized_tt: Monetized travel time savings DataFrame
            - scenario_list: List of scenarios
    """
    print("\n" + "="*80)
    print("PHASE 9: TRAVEL TIME SAVINGS")
    print("="*80 + "\n")
    st = time.time()

    dev_list, monetized_tt, scenario_list = compute_tts(
        dev_id_lookup=dev_id_lookup,
        od_times_dev=od_times_dev,
        od_times_status_quo=od_times_status_quo,
        use_cache=settings.use_cache_tts_calc
    )

    runtimes["Calculate the TTT Savings"] = time.time() - st
    return dev_list, monetized_tt, scenario_list


def phase_10_old_construction_maintenance_costs(monetized_tt: pd.DataFrame, runtimes: dict) -> pd.DataFrame:
    """
    Phase 10 OLD: Calculate construction and maintenance costs using OLD method.

    Uses 8 trains/track capacity check for EXTEND_LINES.
    Outputs: construction_cost_old.csv

    Args:
        monetized_tt: Monetized travel time savings from Phase 9
        runtimes: Dictionary to track phase execution times

    Returns:
        pd.DataFrame: Construction and maintenance costs with old capacity logic
    """
    print("\n" + "="*80)
    print("PHASE 10 OLD: CONSTRUCTION & MAINTENANCE COSTS (8 trains/track)")
    print("="*80 + "\n")
    st = time.time()

    file_path = "data/Network/Rail-Service_Link_construction_cost.csv"

    construction_and_maintenance_costs_old = construction_costs(
        file_path=file_path,
        cost_per_meter=cp.track_cost_per_meter,
        tunnel_cost_per_meter=cp.tunnel_cost_per_meter,
        bridge_cost_per_meter=cp.bridge_cost_per_meter,
        track_maintenance_cost=cp.track_maintenance_cost,
        tunnel_maintenance_cost=cp.tunnel_maintenance_cost,
        bridge_maintenance_cost=cp.bridge_maintenance_cost,
        duration=cp.duration,
        use_old_capacity_logic=True,
        output_suffix="_old"
    )

    runtimes["Compute construction costs (OLD)"] = time.time() - st
    return construction_and_maintenance_costs_old


def phase_10_new_construction_maintenance_costs(monetized_tt: pd.DataFrame, runtimes: dict) -> pd.DataFrame:
    """
    Phase 10 NEW: Calculate construction and maintenance costs using NEW method.

    Uses capacity interventions from Phase 4 for EXTEND_LINES.
    Outputs: construction_cost.csv

    Args:
        monetized_tt: Monetized travel time savings from Phase 9
        runtimes: Dictionary to track phase execution times

    Returns:
        pd.DataFrame: Construction and maintenance costs with capacity interventions
    """
    print("\n" + "="*80)
    print("PHASE 10 NEW: CONSTRUCTION & MAINTENANCE COSTS (Capacity Interventions)")
    print("="*80 + "\n")
    st = time.time()

    file_path = "data/Network/Rail-Service_Link_construction_cost.csv"

    construction_and_maintenance_costs_new = construction_costs(
        file_path=file_path,
        cost_per_meter=cp.track_cost_per_meter,
        tunnel_cost_per_meter=cp.tunnel_cost_per_meter,
        bridge_cost_per_meter=cp.bridge_cost_per_meter,
        track_maintenance_cost=cp.track_maintenance_cost,
        tunnel_maintenance_cost=cp.tunnel_maintenance_cost,
        bridge_maintenance_cost=cp.bridge_maintenance_cost,
        duration=cp.duration,
        use_old_capacity_logic=False,
        output_suffix=""
    )

    runtimes["Compute construction costs (NEW)"] = time.time() - st
    return construction_and_maintenance_costs_new


def phase_11_old_cost_benefit_integration(runtimes: dict) -> pd.DataFrame:
    """
    Phase 11 OLD: Integrate costs with benefits and apply discounting (OLD method).

    Loads: construction_cost_old.csv
    Outputs: costs_and_benefits_old_discounted.csv

    Args:
        runtimes: Dictionary to track phase execution times

    Returns:
        pd.DataFrame: Discounted costs and benefits (OLD method)
    """
    print("\n" + "="*80)
    print("PHASE 11 OLD: COST-BENEFIT INTEGRATION (8 trains/track)")
    print("="*80 + "\n")
    st = time.time()

    # Create cost-benefit dataframe
    print("  → Creating cost-benefit dataframe for OLD method...")
    costs_and_benefits_old, _ = create_cost_and_benefit_df(
        settings.start_year_scenario,
        settings.end_year_scenario,
        settings.start_valuation_year,
        cost_file_path="data/costs/construction_cost_old.csv"
    )

    # Apply discounting
    print("  → Applying discounting to OLD method costs...")
    costs_and_benefits_old_discounted = discounting(
        costs_and_benefits_old,
        discount_rate=cp.discount_rate,
        base_year=settings.start_valuation_year
    )

    old_discounted_path = "data/costs/costs_and_benefits_old_discounted.csv"
    costs_and_benefits_old_discounted.to_csv(old_discounted_path)
    print(f"  ✓ Saved to: {old_discounted_path}")

    runtimes["Cost-benefit integration (OLD)"] = time.time() - st
    return costs_and_benefits_old_discounted


def phase_11_new_cost_benefit_integration(runtimes: dict) -> pd.DataFrame:
    """
    Phase 11 NEW: Integrate costs with benefits and apply discounting (NEW method).

    Loads: construction_cost.csv
    Outputs: costs_and_benefits_discounted.csv

    Args:
        runtimes: Dictionary to track phase execution times

    Returns:
        pd.DataFrame: Discounted costs and benefits (NEW method)
    """
    print("\n" + "="*80)
    print("PHASE 11 NEW: COST-BENEFIT INTEGRATION (Capacity Interventions)")
    print("="*80 + "\n")
    st = time.time()

    # Create cost-benefit dataframe
    print("  → Creating cost-benefit dataframe for NEW method...")
    _, costs_and_benefits = create_cost_and_benefit_df(
        settings.start_year_scenario,
        settings.end_year_scenario,
        settings.start_valuation_year,
        cost_file_path="data/costs/construction_cost.csv"
    )

    # Apply discounting
    print("  → Applying discounting to NEW method costs...")
    costs_and_benefits_discounted = discounting(
        costs_and_benefits,
        discount_rate=cp.discount_rate,
        base_year=settings.start_valuation_year
    )

    discounted_path = "data/costs/costs_and_benefits_discounted.csv"
    costs_and_benefits_discounted.to_csv(discounted_path)
    print(f"  ✓ Saved to: {discounted_path}")

    # Optional: Generate plots
    generate_cb_plots = PIPELINE_CONFIG.should_generate_plots(default_yes=False)
    if generate_cb_plots is None:  # Manual mode
        print("\n" + "="*80)
        print("VISUALIZATION OPTION")
        print("="*80)
        response = input("\nGenerate cost-benefit plots for all developments? (y/n) [n]: ").strip().lower()
        generate_cb_plots = response == 'y'

    if generate_cb_plots:
        print("\n  Generating plots for all developments...")
        output_dir = os.path.join("plots", "Discounted Costs")

        # Get all unique development IDs from the discounted dataframe
        dev_ids = costs_and_benefits_discounted.index.get_level_values('development').unique()

        for i, dev_id in enumerate(dev_ids, 1):
            print(f"    [{i}/{len(dev_ids)}] Plotting development {dev_id}...")
            plot_costs_benefits(costs_and_benefits_discounted, line=dev_id, output_dir=output_dir)

        print(f"\n  ✓ All plots saved to: {output_dir}")
    else:
        print("  → Skipping visualizations")

    runtimes["Cost-benefit integration (NEW)"] = time.time() - st
    return costs_and_benefits_discounted


def phase_12_old_cost_aggregation(runtimes: dict) -> None:
    """
    Phase 12 OLD: Aggregate cost elements (OLD method).

    Loads: costs_and_benefits_old_discounted.csv
    Outputs: total_costs_old.csv, total_costs_summary_old.csv (CSV only, no geometry)

    Args:
        runtimes: Dictionary to track phase execution times
    """
    print("\n" + "="*80)
    print("PHASE 12 OLD: COST AGGREGATION (8 trains/track)")
    print("="*80 + "\n")
    st = time.time()

    # Load the discounted cost-benefit dataframe from Phase 11 OLD
    costs_old_path = "data/costs/costs_and_benefits_old_discounted.csv"
    print(f"  Loading OLD method costs: {costs_old_path}")
    costs_and_benefits_old_discounted = pd.read_csv(costs_old_path)

    # Process old version - CSV only
    print("  → Processing OLD method costs (CSV only)...")
    rearange_costs(costs_and_benefits_old_discounted, output_prefix="_old", csv_only=True)

    runtimes["Aggregate costs (OLD)"] = time.time() - st


def phase_12_new_cost_aggregation(runtimes: dict) -> None:
    """
    Phase 12 NEW: Aggregate cost elements (NEW method).

    Loads: costs_and_benefits_discounted.csv
    Outputs: total_costs.csv, total_costs_with_geometry.gpkg, total_costs_summary.csv

    Args:
        runtimes: Dictionary to track phase execution times
    """
    print("\n" + "="*80)
    print("PHASE 12 NEW: COST AGGREGATION (Capacity Interventions)")
    print("="*80 + "\n")
    st = time.time()

    # Load the discounted cost-benefit dataframe from Phase 11 NEW
    costs_path = "data/costs/costs_and_benefits_discounted.csv"
    print(f"  Loading NEW method costs: {costs_path}")
    costs_and_benefits_discounted = pd.read_csv(costs_path)

    # Process new version - full outputs with geometry
    print("  → Processing NEW method costs (full outputs with geometry)...")
    rearange_costs(costs_and_benefits_discounted, output_prefix="")

    runtimes["Aggregate costs (NEW)"] = time.time() - st


def phase_13_results_visualization(runtimes: dict) -> None:
    """Phase 13: Generate all result visualizations."""
    print("\n" + "="*80)
    print("PHASE 13: RESULTS VISUALIZATION")
    print("="*80 + "\n")
    st = time.time()

    # Use global visualization setting
    generate_result_plots = PIPELINE_CONFIG.should_generate_plots(default_yes=True)

    if generate_result_plots is None:  # Manual mode - show detailed prompts
        print("Select which benefit plot categories to generate:")
        print("─" * 60)

        plot_small_developments = input("  Generate plots for small developments (Expand 1 Stop)? (y/n): ").strip().lower() == 'y'
        plot_grouped_by_connection = input("  Generate plots grouped by missing connection? (y/n): ").strip().lower() == 'y'
        plot_ranked_groups = input("  Generate ranked group plots (by net benefit)? (y/n): ").strip().lower() == 'y'
        plot_combined_with_maps = input("  Generate combined plots (charts + network maps)? (y/n): ").strip().lower() == 'y'
        plot_pipeline_comparison = input("  Generate pipeline comparison plots (Old vs New)? (y/n): ").strip().lower() == 'y'

        print("─" * 60)

        plot_preferences = {
            'small_developments': plot_small_developments,
            'grouped_by_connection': plot_grouped_by_connection,
            'ranked_groups': plot_ranked_groups,
            'combined_with_maps': plot_combined_with_maps,
            'pipeline_comparison': plot_pipeline_comparison
        }
    elif generate_result_plots:  # All mode
        print("  → Generating all result visualizations")
        plot_preferences = {
            'small_developments': True,
            'grouped_by_connection': True,
            'ranked_groups': True,
            'combined_with_maps': True,
            'pipeline_comparison': True
        }
    else:  # None mode
        print("  → Skipping result visualizations")
        plot_preferences = {
            'small_developments': False,
            'grouped_by_connection': False,
            'ranked_groups': False,
            'combined_with_maps': False,
            'pipeline_comparison': False
        }

    visualize_results(clear_plot_directory=False, plot_preferences=plot_preferences)

    # Generate pipeline comparison plots (Old vs New)
    if plot_preferences.get('pipeline_comparison', False):
        create_ranked_pipeline_comparison_plots(plot_directory="plots", top_n_list=[5, 10])
        create_all_developments_pipeline_comparison_plots(plot_directory="plots")

        # Export comprehensive report statistics for pipeline comparison
        export_pipeline_comparison_report_statistics()

    runtimes["Visualize results"] = time.time() - st


def export_pipeline_comparison_report_statistics():
    """
    Export comprehensive report-ready statistics for old vs new pipeline comparison.

    Generates 5 CSV files with detailed comparative statistics suitable for
    academic reporting, similar to sensitivity analysis output.

    Files created in plots/Benefits_Pipeline_Comparison/:
    - report_pipeline_viability_statistics.csv
    - report_pipeline_cost_statistics.csv
    - report_pipeline_bcr_statistics.csv
    - report_pipeline_net_benefit_statistics.csv
    - report_pipeline_top10_performers.csv
    """
    print("\n" + "="*80)
    print("EXPORTING PIPELINE COMPARISON REPORT STATISTICS")
    print("="*80 + "\n")

    # Load data from both pipelines
    new_data_path = "data/costs/total_costs_raw.csv"
    old_data_path = "data/costs/total_costs_raw_old.csv"

    if not os.path.exists(new_data_path) or not os.path.exists(old_data_path):
        print(f"  ⚠ Data files not found - skipping report statistics export")
        return

    # Output directory
    output_dir = os.path.join("plots", "Benefits_Pipeline_Comparison", "report_statistics")
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare data using the helper function from plots.py
    from plots import _prepare_pipeline_data

    print(f"  Loading and preparing pipeline data...")
    df_new = _prepare_pipeline_data(new_data_path, pipeline='new')
    df_old = _prepare_pipeline_data(old_data_path, pipeline='old')

    # Combine datasets
    df_combined = pd.concat([df_new, df_old], ignore_index=True)

    # Mark viable scenarios (net benefit > 0)
    df_combined['is_viable'] = df_combined['total_net_benefit'] > 0

    # Get all developments
    all_developments = sorted(df_combined['development'].unique())

    print(f"  Processing {len(all_developments)} developments across both pipelines...\n")

    # ========================================================================
    # 1. VIABILITY STATISTICS
    # ========================================================================
    print("  [1/5] Generating viability statistics...")

    viability_stats = []

    for dev_id in all_developments:
        dev_data = df_combined[df_combined['development'] == dev_id]
        line_name = dev_data['line_name'].iloc[0] if 'line_name' in dev_data.columns else dev_id

        new_data = dev_data[dev_data['pipeline'] == 'new']
        old_data = dev_data[dev_data['pipeline'] == 'old']

        # Count viable scenarios
        viable_new = new_data['is_viable'].sum() if len(new_data) > 0 else 0
        viable_old = old_data['is_viable'].sum() if len(old_data) > 0 else 0
        total_scenarios = new_data['scenario'].nunique() if len(new_data) > 0 else 0

        viability_stats.append({
            'development': dev_id,
            'line_name': line_name,
            'total_scenarios': total_scenarios,
            'viable_scenarios_old': int(viable_old),
            'viable_scenarios_new': int(viable_new),
            'viable_scenarios_change': int(viable_new - viable_old),
            'viable_pct_old': (viable_old / total_scenarios * 100) if total_scenarios > 0 else 0,
            'viable_pct_new': (viable_new / total_scenarios * 100) if total_scenarios > 0 else 0,
            'viable_pct_change': ((viable_new - viable_old) / total_scenarios * 100) if total_scenarios > 0 else 0,
            'viability_improved': viable_new > viable_old,
            'viability_worsened': viable_new < viable_old,
            'viability_unchanged': viable_new == viable_old
        })

    viability_df = pd.DataFrame(viability_stats)
    viability_path = os.path.join(output_dir, "report_pipeline_viability_statistics.csv")
    viability_df.to_csv(viability_path, index=False)
    print(f"      ✓ Saved: {viability_path}")

    # ========================================================================
    # 2. COST STATISTICS
    # ========================================================================
    print("  [2/5] Generating cost statistics...")

    cost_stats = []

    for dev_id in all_developments:
        dev_data = df_combined[df_combined['development'] == dev_id]
        line_name = dev_data['line_name'].iloc[0] if 'line_name' in dev_data.columns else dev_id

        new_data = dev_data[dev_data['pipeline'] == 'new']
        old_data = dev_data[dev_data['pipeline'] == 'old']

        # Calculate mean costs
        mean_cost_new = new_data['total_costs'].mean() if len(new_data) > 0 else 0
        mean_cost_old = old_data['total_costs'].mean() if len(old_data) > 0 else 0
        cost_change_chf = mean_cost_new - mean_cost_old
        cost_change_pct = (cost_change_chf / mean_cost_old * 100) if mean_cost_old != 0 else 0

        # Cost component breakdown (if available)
        construction_new = new_data['TotalConstructionCost'].mean() if 'TotalConstructionCost' in new_data.columns and len(new_data) > 0 else 0
        construction_old = old_data['TotalConstructionCost'].mean() if 'TotalConstructionCost' in old_data.columns and len(old_data) > 0 else 0

        maintenance_new = new_data['TotalMaintenanceCost'].mean() if 'TotalMaintenanceCost' in new_data.columns and len(new_data) > 0 else 0
        maintenance_old = old_data['TotalMaintenanceCost'].mean() if 'TotalMaintenanceCost' in old_data.columns and len(old_data) > 0 else 0

        operating_new = new_data['TotalUncoveredOperatingCost'].mean() if 'TotalUncoveredOperatingCost' in new_data.columns and len(new_data) > 0 else 0
        operating_old = old_data['TotalUncoveredOperatingCost'].mean() if 'TotalUncoveredOperatingCost' in old_data.columns and len(old_data) > 0 else 0

        cost_stats.append({
            'development': dev_id,
            'line_name': line_name,
            'mean_total_cost_old_chf': mean_cost_old,
            'mean_total_cost_new_chf': mean_cost_new,
            'total_cost_change_chf': cost_change_chf,
            'total_cost_change_pct': cost_change_pct,
            'construction_cost_old_chf': construction_old,
            'construction_cost_new_chf': construction_new,
            'construction_cost_change_chf': construction_new - construction_old,
            'maintenance_cost_old_chf': maintenance_old,
            'maintenance_cost_new_chf': maintenance_new,
            'maintenance_cost_change_chf': maintenance_new - maintenance_old,
            'operating_cost_old_chf': operating_old,
            'operating_cost_new_chf': operating_new,
            'operating_cost_change_chf': operating_new - operating_old,
            'cost_reduced': cost_change_chf < 0,
            'cost_increased': cost_change_chf > 0,
            'cost_unchanged': cost_change_chf == 0
        })

    cost_df = pd.DataFrame(cost_stats)
    cost_path = os.path.join(output_dir, "report_pipeline_cost_statistics.csv")
    cost_df.to_csv(cost_path, index=False)
    print(f"      ✓ Saved: {cost_path}")

    # ========================================================================
    # 3. BCR/CBA STATISTICS
    # ========================================================================
    print("  [3/5] Generating BCR/CBA statistics...")

    bcr_stats = []

    for dev_id in all_developments:
        dev_data = df_combined[df_combined['development'] == dev_id]
        line_name = dev_data['line_name'].iloc[0] if 'line_name' in dev_data.columns else dev_id

        new_data = dev_data[dev_data['pipeline'] == 'new']
        old_data = dev_data[dev_data['pipeline'] == 'old']

        # BCR/CBA statistics
        if 'cba_ratio' in new_data.columns and len(new_data) > 0:
            mean_bcr_new = new_data['cba_ratio'].mean()
            median_bcr_new = new_data['cba_ratio'].median()
            min_bcr_new = new_data['cba_ratio'].min()
            max_bcr_new = new_data['cba_ratio'].max()
            std_bcr_new = new_data['cba_ratio'].std()
            bcr_above_1_new = (new_data['cba_ratio'] >= 1.0).sum()
        else:
            mean_bcr_new = median_bcr_new = min_bcr_new = max_bcr_new = std_bcr_new = bcr_above_1_new = 0

        if 'cba_ratio' in old_data.columns and len(old_data) > 0:
            mean_bcr_old = old_data['cba_ratio'].mean()
            median_bcr_old = old_data['cba_ratio'].median()
            min_bcr_old = old_data['cba_ratio'].min()
            max_bcr_old = old_data['cba_ratio'].max()
            std_bcr_old = old_data['cba_ratio'].std()
            bcr_above_1_old = (old_data['cba_ratio'] >= 1.0).sum()
        else:
            mean_bcr_old = median_bcr_old = min_bcr_old = max_bcr_old = std_bcr_old = bcr_above_1_old = 0

        bcr_stats.append({
            'development': dev_id,
            'line_name': line_name,
            'mean_bcr_old': mean_bcr_old,
            'mean_bcr_new': mean_bcr_new,
            'bcr_change': mean_bcr_new - mean_bcr_old,
            'median_bcr_old': median_bcr_old,
            'median_bcr_new': median_bcr_new,
            'min_bcr_old': min_bcr_old,
            'min_bcr_new': min_bcr_new,
            'max_bcr_old': max_bcr_old,
            'max_bcr_new': max_bcr_new,
            'std_bcr_old': std_bcr_old,
            'std_bcr_new': std_bcr_new,
            'scenarios_bcr_above_1_old': int(bcr_above_1_old),
            'scenarios_bcr_above_1_new': int(bcr_above_1_new),
            'bcr_viability_change': int(bcr_above_1_new - bcr_above_1_old),
            'bcr_improved': mean_bcr_new > mean_bcr_old,
            'bcr_worsened': mean_bcr_new < mean_bcr_old,
            'bcr_unchanged': mean_bcr_new == mean_bcr_old
        })

    bcr_df = pd.DataFrame(bcr_stats)
    bcr_path = os.path.join(output_dir, "report_pipeline_bcr_statistics.csv")
    bcr_df.to_csv(bcr_path, index=False)
    print(f"      ✓ Saved: {bcr_path}")

    # ========================================================================
    # 4. NET BENEFIT STATISTICS
    # ========================================================================
    print("  [4/5] Generating net benefit statistics...")

    net_benefit_stats = []

    for dev_id in all_developments:
        dev_data = df_combined[df_combined['development'] == dev_id]
        line_name = dev_data['line_name'].iloc[0] if 'line_name' in dev_data.columns else dev_id

        new_data = dev_data[dev_data['pipeline'] == 'new']
        old_data = dev_data[dev_data['pipeline'] == 'old']

        # Net benefit statistics
        mean_nb_new = new_data['total_net_benefit'].mean() if len(new_data) > 0 else 0
        mean_nb_old = old_data['total_net_benefit'].mean() if len(old_data) > 0 else 0
        median_nb_new = new_data['total_net_benefit'].median() if len(new_data) > 0 else 0
        median_nb_old = old_data['total_net_benefit'].median() if len(old_data) > 0 else 0

        total_nb_new = new_data['total_net_benefit'].sum() if len(new_data) > 0 else 0
        total_nb_old = old_data['total_net_benefit'].sum() if len(old_data) > 0 else 0

        positive_nb_new = (new_data['total_net_benefit'] > 0).sum() if len(new_data) > 0 else 0
        positive_nb_old = (old_data['total_net_benefit'] > 0).sum() if len(old_data) > 0 else 0

        negative_nb_new = (new_data['total_net_benefit'] < 0).sum() if len(new_data) > 0 else 0
        negative_nb_old = (old_data['total_net_benefit'] < 0).sum() if len(old_data) > 0 else 0

        net_benefit_stats.append({
            'development': dev_id,
            'line_name': line_name,
            'mean_net_benefit_old_chf': mean_nb_old,
            'mean_net_benefit_new_chf': mean_nb_new,
            'net_benefit_change_chf': mean_nb_new - mean_nb_old,
            'median_net_benefit_old_chf': median_nb_old,
            'median_net_benefit_new_chf': median_nb_new,
            'total_net_benefit_old_chf': total_nb_old,
            'total_net_benefit_new_chf': total_nb_new,
            'scenarios_positive_nb_old': int(positive_nb_old),
            'scenarios_positive_nb_new': int(positive_nb_new),
            'scenarios_negative_nb_old': int(negative_nb_old),
            'scenarios_negative_nb_new': int(negative_nb_new),
            'net_benefit_improved': mean_nb_new > mean_nb_old,
            'net_benefit_worsened': mean_nb_new < mean_nb_old,
            'net_benefit_unchanged': mean_nb_new == mean_nb_old
        })

    net_benefit_df = pd.DataFrame(net_benefit_stats)
    net_benefit_path = os.path.join(output_dir, "report_pipeline_net_benefit_statistics.csv")
    net_benefit_df.to_csv(net_benefit_path, index=False)
    print(f"      ✓ Saved: {net_benefit_path}")

    # ========================================================================
    # 5. TOP 10 PERFORMERS BY PIPELINE
    # ========================================================================
    print("  [5/5] Generating top 10 performers...")

    # Rank by mean net benefit for each pipeline
    new_rankings = df_new.groupby('development')['total_net_benefit'].mean().sort_values(ascending=False).head(10)
    old_rankings = df_old.groupby('development')['total_net_benefit'].mean().sort_values(ascending=False).head(10)

    top10_data = []

    # Add NEW pipeline top 10
    for rank, (dev_id, mean_nb) in enumerate(new_rankings.items(), 1):
        dev_data = df_new[df_new['development'] == dev_id]
        line_name = dev_data['line_name'].iloc[0] if 'line_name' in dev_data.columns else dev_id
        mean_cost = dev_data['total_costs'].mean()
        mean_bcr = dev_data['cba_ratio'].mean() if 'cba_ratio' in dev_data.columns else 0
        viable_scenarios = (dev_data['total_net_benefit'] > 0).sum()

        top10_data.append({
            'pipeline': 'new',
            'rank': rank,
            'development': dev_id,
            'line_name': line_name,
            'mean_net_benefit_chf': mean_nb,
            'mean_total_cost_chf': mean_cost,
            'mean_bcr': mean_bcr,
            'viable_scenarios': int(viable_scenarios)
        })

    # Add OLD pipeline top 10
    for rank, (dev_id, mean_nb) in enumerate(old_rankings.items(), 1):
        dev_data = df_old[df_old['development'] == dev_id]
        line_name = dev_data['line_name'].iloc[0] if 'line_name' in dev_data.columns else dev_id
        mean_cost = dev_data['total_costs'].mean()
        mean_bcr = dev_data['cba_ratio'].mean() if 'cba_ratio' in dev_data.columns else 0
        viable_scenarios = (dev_data['total_net_benefit'] > 0).sum()

        top10_data.append({
            'pipeline': 'old',
            'rank': rank,
            'development': dev_id,
            'line_name': line_name,
            'mean_net_benefit_chf': mean_nb,
            'mean_total_cost_chf': mean_cost,
            'mean_bcr': mean_bcr,
            'viable_scenarios': int(viable_scenarios)
        })

    top10_df = pd.DataFrame(top10_data)
    top10_path = os.path.join(output_dir, "report_pipeline_top10_performers.csv")
    top10_df.to_csv(top10_path, index=False)
    print(f"      ✓ Saved: {top10_path}")

    # ========================================================================
    # SUMMARY OUTPUT
    # ========================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPARISON REPORT STATISTICS - SUMMARY")
    print("="*80)
    print(f"\nGenerated 5 comprehensive report files in:")
    print(f"  {output_dir}")
    print(f"\nFiles created:")
    print(f"  1. report_pipeline_viability_statistics.csv")
    print(f"  2. report_pipeline_cost_statistics.csv")
    print(f"  3. report_pipeline_bcr_statistics.csv")
    print(f"  4. report_pipeline_net_benefit_statistics.csv")
    print(f"  5. report_pipeline_top10_performers.csv")

    # Summary statistics
    print(f"\nKey Findings:")
    print(f"  • Total developments analyzed: {len(all_developments)}")
    print(f"  • Developments with improved viability (NEW): {viability_df['viability_improved'].sum()}")
    print(f"  • Developments with reduced costs (NEW): {cost_df['cost_reduced'].sum()}")
    print(f"  • Developments with improved BCR (NEW): {bcr_df['bcr_improved'].sum()}")
    print(f"  • Developments with improved net benefit (NEW): {net_benefit_df['net_benefit_improved'].sum()}")
    print(f"  • Average cost change (NEW vs OLD): CHF {cost_df['total_cost_change_chf'].mean():,.0f}")
    print(f"  • Average BCR change (NEW vs OLD): {bcr_df['bcr_change'].mean():.3f}")
    print(f"  • Average net benefit change (NEW vs OLD): CHF {net_benefit_df['net_benefit_change_chf'].mean():,.0f}")
    print("="*80 + "\n")


def infrascanrail_cap():
    """
    Enhanced InfraScanRail main pipeline with integrated capacity analysis.

    This orchestrator sequentially calls all 13 phases of the capacity-enhanced pipeline.
    Each phase is now encapsulated as a separate function for easier debugging and testing.
    """
    os.chdir(paths.MAIN)
    warnings.filterwarnings("ignore")  # TODO: No warnings should be ignored
    runtimes = {}

    # ============================================================================
    # CONFIGURABLE AUTO-RESPONSE SETUP
    # ============================================================================
    # Store original input function
    if isinstance(__builtins__, dict):
        _original_input = __builtins__['input']
    else:
        _original_input = __builtins__.input
    
    # Store in config for later use
    PIPELINE_CONFIG._original_input = _original_input

    def smart_input(prompt=""):
        """
        Intelligently handle input based on global configuration.
        - Capacity grouping prompts: Use grouping strategy
        - Visualization prompts: Use visualization strategy
        - All other prompts: Use original input
        """
        # Check if this is a capacity grouping prompt
        if ("Enter choice (1-" in prompt or 
            "Select the strategy number" in prompt or
            "Select track allocation strategy" in prompt):
            return PIPELINE_CONFIG.get_grouping_choice(prompt)
        
        # For all other prompts, use original input
        return _original_input(prompt)

    # Replace built-in input with our smart input function
    if isinstance(__builtins__, dict):
        __builtins__['input'] = smart_input
    else:
        __builtins__.input = smart_input

    ##################################################################################
    # PHASE 1: INITIALIZATION & CONFIGURATION
    ##################################################################################
    innerboundary, outerboundary = phase_1_initialization(runtimes)

    ##################################################################################
    # PHASE 2: DATA IMPORT
    ##################################################################################
    phase_2_data_import(runtimes)

    ##################################################################################
    # PHASE 3: BASELINE CAPACITY ANALYSIS
    ##################################################################################
    points, baseline_prep_path, baseline_sections_path, enhanced_network_label = \
        phase_3_baseline_capacity_analysis(runtimes)
 
    ##################################################################################
    # PHASE 4: INFRASTRUCTURE DEVELOPMENTS
    ##################################################################################
    dev_id_lookup, capacity_analysis_results = \
        phase_4_infrastructure_developments(points, runtimes)

    ##################################################################################
    # PHASE 5: DEMAND ANALYSIS (OD MATRIX)
    ##################################################################################
    phase_5_demand_analysis(points, runtimes)

    ##################################################################################
    # PHASE 6: TRAVEL TIME COMPUTATION
    ##################################################################################
    od_times_dev, od_times_status_quo, G_status_quo, G_development = \
        phase_6_travel_time_computation(dev_id_lookup, runtimes)

    ##################################################################################
    # PHASE 7: PASSENGER FLOW VISUALIZATION
    ##################################################################################
    if settings.plot_passenger_flow:
        phase_7_passenger_flow_visualization(
            G_development, G_status_quo, dev_id_lookup, runtimes
        )

    ##################################################################################
    # PHASE 8: SCENARIO GENERATION
    ##################################################################################
    if settings.OD_type == 'canton_ZH':
        phase_8_scenario_generation(runtimes)

    ##################################################################################
    # PHASE 9: TRAVEL TIME SAVINGS
    ##################################################################################
    dev_list, monetized_tt, scenario_list = \
        phase_9_travel_time_savings(
            dev_id_lookup, od_times_dev, od_times_status_quo, runtimes
        )

    ##################################################################################
    # PHASE 10 OLD: CONSTRUCTION & MAINTENANCE COSTS (8 trains/track)
    ##################################################################################
    phase_10_old_construction_maintenance_costs(monetized_tt, runtimes)

    ##################################################################################
    # PHASE 11 OLD: COST-BENEFIT INTEGRATION (8 trains/track)
    ##################################################################################
    phase_11_old_cost_benefit_integration(runtimes)

    ##################################################################################
    # PHASE 12 OLD: COST AGGREGATION (8 trains/track)
    ##################################################################################
    phase_12_old_cost_aggregation(runtimes)

    ##################################################################################
    # PHASE 10 NEW: CONSTRUCTION & MAINTENANCE COSTS (Capacity Interventions)
    ##################################################################################
    phase_10_new_construction_maintenance_costs(monetized_tt, runtimes)

    ##################################################################################
    # PHASE 11 NEW: COST-BENEFIT INTEGRATION (Capacity Interventions)
    ##################################################################################
    phase_11_new_cost_benefit_integration(runtimes)

    ##################################################################################
    # PHASE 12 NEW: COST AGGREGATION (Capacity Interventions)
    ##################################################################################
    phase_12_new_cost_aggregation(runtimes)

    ##################################################################################
    # PHASE 13: RESULTS VISUALIZATION
    ##################################################################################
    phase_13_results_visualization(runtimes)
  
    ##################################################################################
    # SAVE RUNTIMES
    ##################################################################################
    with open(r'runtimes_cap.txt', 'w') as file:
        file.write("=" * 80 + "\n")
        file.write("INFRASCANRAIL CAPACITY-ENHANCED PIPELINE RUNTIMES\n")
        file.write("=" * 80 + "\n\n")
        total_time = sum(runtimes.values())
        for part, runtime in runtimes.items():
            mins = int(runtime // 60)
            secs = int(runtime % 60)
            file.write(f"{part:.<50} {mins}m {secs}s ({runtime:.2f}s)\n")
        file.write("\n" + "=" * 80 + "\n")
        total_mins = int(total_time // 60)
        total_secs = int(total_time % 60)
        file.write(f"{'TOTAL TIME':.<50} {total_mins}m {total_secs}s ({total_time:.2f}s)\n")
        file.write("=" * 80 + "\n")

    # Restore original input function
    if isinstance(__builtins__, dict):
        __builtins__['input'] = _original_input
    else:
        __builtins__.input = _original_input

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nTotal runtime: {int(total_time // 60)}m {int(total_time % 60)}s")
    print(f"Runtimes saved to: runtimes_cap.txt\n")


# ================================================================================
# SUPPORTING FUNCTIONS (from original main.py)
# ================================================================================

def create_focus_area():
    """Define spatial limits of the research corridor."""
    e_min, e_max = 2687000, 2708000
    n_min, n_max = 1237000, 1254000
    margin = 3000  # meters
    innerboundary, outerboundary = save_focus_area_shapefile(e_min, e_max, n_min, n_max, margin)
    return innerboundary, outerboundary


def create_dev_id_lookup_table():
    """
    Creates a lookup table (DataFrame) of development filenames.
    DataFrame index starts at 1 and filenames are listed without extensions.
    """
    dev_dir = paths.DEVELOPMENT_DIRECTORY
    all_files = [
        f for f in os.listdir(dev_dir)
        if os.path.isfile(os.path.join(dev_dir, f))
    ]
    dev_ids = sorted(os.path.splitext(f)[0] for f in all_files)
    df = pd.DataFrame({'dev_id': dev_ids}, index=range(1, len(dev_ids) + 1))
    return df


def extract_capacity_intervention_costs(
    capacity_analysis_results: dict,
    baseline_network_label: str
) -> pd.DataFrame:
    """
    Extract capacity intervention costs for each development by comparing to baseline.

    Compares development networks (Stations/Segments) to baseline network and identifies
    capacity interventions (added tracks/platforms) that match the enhanced baseline
    intervention catalog.

    Args:
        capacity_analysis_results: Dict from Phase 4.2 with development analysis results
        baseline_network_label: Baseline network label (e.g., "2024_extended")

    Returns:
        DataFrame with columns: dev_id, int_id, construction_cost, maintenance_cost
    """
    print("\n  Extracting capacity intervention costs for developments...")

    # Load baseline network (original, not enhanced)
    baseline_capacity_dir = CAPACITY_ROOT / "Baseline" / baseline_network_label
    baseline_prep_path = baseline_capacity_dir / f"capacity_{baseline_network_label}_network_prep.xlsx"

    # Fallback to old structure
    if not baseline_prep_path.exists():
        baseline_capacity_dir = CAPACITY_ROOT / baseline_network_label
        baseline_prep_path = baseline_capacity_dir / f"capacity_{baseline_network_label}_network_prep.xlsx"

    if not baseline_prep_path.exists():
        print(f"    ⚠ Baseline prep workbook not found: {baseline_prep_path}")
        print(f"    → Skipping capacity intervention cost extraction")
        return pd.DataFrame(columns=['dev_id', 'int_id', 'construction_cost', 'maintenance_cost'])

    # Load enhanced baseline intervention catalog
    enhanced_network_label = f"{baseline_network_label}_enhanced"
    interventions_catalog_path = (
        CAPACITY_ROOT / "Enhanced" / enhanced_network_label / "capacity_interventions.csv"
    )

    if not interventions_catalog_path.exists():
        print(f"    ⚠ Interventions catalog not found: {interventions_catalog_path}")
        print(f"    → Skipping capacity intervention cost extraction")
        return pd.DataFrame(columns=['dev_id', 'int_id', 'construction_cost', 'maintenance_cost'])

    # Load baseline network
    print(f"    Loading baseline network: {baseline_prep_path}")
    baseline_stations = pd.read_excel(baseline_prep_path, sheet_name='Stations')
    baseline_segments = pd.read_excel(baseline_prep_path, sheet_name='Segments')

    # Load intervention catalog
    print(f"    Loading intervention catalog: {interventions_catalog_path}")
    interventions_catalog = pd.read_csv(interventions_catalog_path)

    # Results storage
    results = []

    # Process each development
    for dev_id, dev_result in capacity_analysis_results.items():
        if dev_result.get('status') != 'success':
            # No successful capacity analysis - record zero costs
            results.append({
                'dev_id': dev_id,
                'int_id': '',
                'construction_cost': 0.0,
                'maintenance_cost': 0.0
            })
            continue

        # Load development sections workbook
        dev_sections_path = Path(dev_result['sections_path'])

        if not dev_sections_path.exists():
            print(f"    ⚠ Development sections file not found: {dev_sections_path}")
            results.append({
                'dev_id': dev_id,
                'int_id': '',
                'construction_cost': 0.0,
                'maintenance_cost': 0.0
            })
            continue

        try:
            dev_stations = pd.read_excel(dev_sections_path, sheet_name='Stations')
            dev_segments = pd.read_excel(dev_sections_path, sheet_name='Segments')
        except Exception as e:
            print(f"    ⚠ Error loading development {dev_id} workbook: {e}")
            results.append({
                'dev_id': dev_id,
                'int_id': '',
                'construction_cost': 0.0,
                'maintenance_cost': 0.0
            })
            continue

        # Track matched interventions
        matched_interventions = []
        total_construction_cost = 0.0
        total_maintenance_cost = 0.0

        # Compare stations (tracks and platforms)
        for _, dev_station in dev_stations.iterrows():
            node_id = dev_station['NR']

            # Find matching baseline station
            baseline_station = baseline_stations[baseline_stations['NR'] == node_id]

            if len(baseline_station) == 0:
                # New station in development (not in baseline) - skip
                continue

            baseline_station = baseline_station.iloc[0]

            # Check for track increases
            dev_tracks = dev_station.get('tracks', 0)
            baseline_tracks = baseline_station.get('tracks', 0)

            # Check for platform increases
            dev_platforms = dev_station.get('platforms', 0)
            baseline_platforms = baseline_station.get('platforms', 0)

            if dev_tracks > baseline_tracks or dev_platforms > baseline_platforms:
                # Look for matching intervention in catalog
                station_interventions = interventions_catalog[
                    (interventions_catalog['type'] == 'station_track') &
                    (interventions_catalog['node_id'] == node_id)
                ]

                if len(station_interventions) > 0:
                    # Use first matching intervention
                    intervention = station_interventions.iloc[0]
                    matched_interventions.append(intervention['intervention_id'])
                    total_construction_cost += intervention['construction_cost_chf']
                    total_maintenance_cost += intervention['maintenance_cost_annual_chf']
                else:
                    print(f"    ⚠ Warning: Station {node_id} in dev {dev_id} has increased tracks/platforms but no matching intervention found")

        # Compare segments (tracks only)
        for _, dev_segment in dev_segments.iterrows():
            from_node = dev_segment['from_node']
            to_node = dev_segment['to_node']

            # Find matching baseline segment
            baseline_segment = baseline_segments[
                (baseline_segments['from_node'] == from_node) &
                (baseline_segments['to_node'] == to_node)
            ]

            if len(baseline_segment) == 0:
                # New segment in development (not in baseline) - skip
                continue

            baseline_segment = baseline_segment.iloc[0]

            # Check for track increases
            dev_tracks = dev_segment.get('tracks', 0)
            baseline_tracks = baseline_segment.get('tracks', 0)

            if dev_tracks > baseline_tracks:
                # Look for matching intervention in catalog
                segment_id = f"{from_node}-{to_node}"
                segment_interventions = interventions_catalog[
                    (interventions_catalog['type'] == 'segment_passing_siding') &
                    (interventions_catalog['segment_id'] == segment_id)
                ]

                if len(segment_interventions) > 0:
                    # Use first matching intervention
                    intervention = segment_interventions.iloc[0]
                    matched_interventions.append(intervention['intervention_id'])
                    total_construction_cost += intervention['construction_cost_chf']
                    total_maintenance_cost += intervention['maintenance_cost_annual_chf']
                else:
                    print(f"    ⚠ Warning: Segment {segment_id} in dev {dev_id} has increased tracks but no matching intervention found")

        # Record results for this development
        if matched_interventions:
            int_id_str = '|'.join(matched_interventions)
        else:
            int_id_str = ''

        results.append({
            'dev_id': dev_id,
            'int_id': int_id_str,
            'construction_cost': total_construction_cost,
            'maintenance_cost': total_maintenance_cost
        })

        if matched_interventions:
            print(f"    ✓ Dev {dev_id}: {len(matched_interventions)} interventions, "
                  f"CHF {total_construction_cost:,.0f} construction, "
                  f"CHF {total_maintenance_cost:,.0f} annual maintenance")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    output_path = Path(paths.MAIN) / "data" / "costs" / "capacity_intervention_costs.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\n    ✓ Capacity intervention costs saved to: {output_path}")
    print(f"    • Total developments processed: {len(results_df)}")
    print(f"    • Developments with interventions: {(results_df['construction_cost'] > 0).sum()}")
    print(f"    • Total construction cost: CHF {results_df['construction_cost'].sum():,.0f}")
    print(f"    • Total annual maintenance: CHF {results_df['maintenance_cost'].sum():,.0f}\n")

    return results_df


def import_process_network(use_cache):
    """Import and process railway network data."""
    if use_cache:
        print("Using cached rail network data...")
        return gpd.read_file(r'data\Network\processed\points.gpkg')
    reformat_rail_nodes()
    network_ak2035, points = create_railway_services_AK2035()
    create_railway_services_AK2035_extended(network_ak2035, points)
    create_railway_services_2024_extended()
    reformat_rail_edges(settings.rail_network)
    add_construction_info_to_network()
    network_in_corridor(poly=settings.perimeter_infra_generation)
    return points


def getStationOD(use_cache, stations_in_perimeter, only_demand_from_to_corridor=False):
    """Generate station-level OD matrix from commune-level data."""
    if use_cache:
        return
    else:
        communalOD = scoring.GetOevDemandPerCommune(tau=1)
        communes_to_stations = pd.read_excel(paths.COMMUNE_TO_STATION_PATH)
        railway_station_OD = aggregate_commune_od_to_station_od(communalOD, communes_to_stations)
        if only_demand_from_to_corridor:
            railway_station_OD = filter_od_matrix_by_stations(railway_station_OD, stations_in_perimeter)
        railway_station_OD.to_csv(paths.OD_STATIONS_KT_ZH_PATH)


def add_construction_info_to_network():
    """Add construction cost information to network edges."""
    const_cost_path = r"data/Network/Rail-Service_Link_construction_cost.csv"
    rows = ['NumOfTracks', 'Bridges m', 'Tunnel m', 'TunnelTrack',
            'tot length m', 'length of 1', 'length of 2 ', 'length of 3 and more']
    df_railway_network = gpd.read_file(paths.RAIL_SERVICES_AK2035_PATH)
    df_const_costs = pd.read_csv(const_cost_path, sep=";", decimal=",")
    # In add_construction_info_to_network(), line 1496:
    df_const_costs_grouped = df_const_costs.groupby(['FromNode', 'ToNode'], as_index=False)[rows].mean()
    new_columns = [col for col in rows if col not in df_railway_network.columns]
    if new_columns:
        df_railway_network[new_columns] = 0
    df_railway_network = df_railway_network.merge(df_const_costs_grouped, on=['FromNode', 'ToNode'], how='left',
                                                  suffixes=('', '_new'))
    for col in rows:
        df_railway_network[col] = df_railway_network[col + '_new'].fillna(df_railway_network[col])
        df_railway_network.drop(columns=[col + '_new'], inplace=True)
    df_railway_network.to_file(paths.RAIL_SERVICES_AK2035_PATH)


def create_travel_time_graphs(network_selection, use_cache, dev_id_lookup_table):
    """Create travel time graphs for status quo and all developments."""
    cache_file = 'data/Network/travel_time/cache/od_times.pkl'

    if use_cache and os.path.exists(cache_file):
        print(f"Loading travel time graphs from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            od_times_dev = cached_data['od_times_dev']
            od_times_status_quo = cached_data['od_times_status_quo']
            G_status_quo = cached_data['G_status_quo']
            G_development = cached_data['G_developments']
        return od_times_dev, od_times_status_quo, G_status_quo, G_development

    # Compute travel time graphs using functions from TT_Delay
    rail_network_path = get_rail_services_path(network_selection)
    network_status_quo = [rail_network_path]
    G_status_quo = create_graphs_from_directories(network_status_quo)
    od_times_status_quo = calculate_od_pairs_with_times_by_graph(G_status_quo)
    
    # Get paths of all developments
    directories_dev = [
        os.path.join(paths.DEVELOPMENT_DIRECTORY, filename)
        for filename in os.listdir(paths.DEVELOPMENT_DIRECTORY) 
        if filename.endswith(".gpkg")
    ]
    directories_dev = [path.replace("\\", "/") for path in directories_dev]
    
    G_development = create_graphs_from_directories(directories_dev)
    od_times_dev = calculate_od_pairs_with_times_by_graph(G_development)

    # Cache results
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'od_times_dev': od_times_dev,
            'od_times_status_quo': od_times_status_quo,
            'G_status_quo': G_status_quo,
            'G_developments': G_development
        }, f)
    print("OD-times saved to cache.")

    return od_times_dev, od_times_status_quo, G_status_quo, G_development


def plot_passenger_flows_on_network(G_development, G_status_quo, dev_id_lookup):
    """Plot passenger flows for status quo and all developments."""
    def calculate_flow_difference(status_quo_graph, development_graph, OD_matrix_flow, points):
        """
        Calculate the difference in passenger flows between status quo and a development

        Args:
            status_quo_graph: Graph of status quo
            development_graph: Graph of a development
            OD_matrix_flow: OD matrix with passenger flows
            points: GeoDataFrame with station points

        Returns:
            difference_flows: GeoDataFrame with flow differences, same structure as flows_on_edges
        """
        # Calculate status quo and development flows
        flows_sq_graph, _ = calculate_flow_on_edges(status_quo_graph, OD_matrix_flow, points)
        flows_dev_graph, _ = calculate_flow_on_edges(development_graph, OD_matrix_flow, points)

        # Extract flow data from graphs
        flows_sq_data = []
        for u, v, data in flows_sq_graph.edges(data=True):
            flow = data.get('flow', 0)
            flows_sq_data.append({'u': u, 'v': v, 'flow': flow})
        flows_sq = pd.DataFrame(flows_sq_data)

        flows_dev_data = []
        for u, v, data in flows_dev_graph.edges(data=True):
            flow = data.get('flow', 0)
            flows_dev_data.append({'u': u, 'v': v, 'flow': flow})
        flows_dev = pd.DataFrame(flows_dev_data)

        # Merge all edges
        all_edges = pd.concat([flows_sq[['u', 'v']], flows_dev[['u', 'v']]]).drop_duplicates()

        # Merge with both flows
        merged = all_edges.merge(flows_sq[['u', 'v', 'flow']], on=['u', 'v'], how='left', suffixes=('', '_sq'))
        merged = merged.merge(flows_dev[['u', 'v', 'flow']], on=['u', 'v'], how='left', suffixes=('', '_dev'))

        # Replace NaN values with 0
        merged['flow'].fillna(0, inplace=True)
        merged['flow_dev'].fillna(0, inplace=True)

        # Calculate difference
        merged['flow_diff'] = merged['flow_dev'] - merged['flow']

        # Create difference graph with same structure as original flow_on_edges graph
        difference_graph = nx.DiGraph()

        # Copy geometry data from original graphs
        for index, row in merged.iterrows():
            u = row['u']
            v = row['v']
            flow_diff = row['flow_diff']

            # Add nodes if not present
            if not difference_graph.has_node(u) and flows_sq_graph.has_node(u):
                # Copy attributes from status quo graph
                node_attrs = flows_sq_graph.nodes[u]
                difference_graph.add_node(u, **node_attrs)
            elif not difference_graph.has_node(u) and flows_dev_graph.has_node(u):
                # If only in development graph
                node_attrs = flows_dev_graph.nodes[u]
                difference_graph.add_node(u, **node_attrs)

            if not difference_graph.has_node(v) and flows_sq_graph.has_node(v):
                node_attrs = flows_sq_graph.nodes[v]
                difference_graph.add_node(v, **node_attrs)
            elif not difference_graph.has_node(v) and flows_dev_graph.has_node(v):
                node_attrs = flows_dev_graph.nodes[v]
                difference_graph.add_node(v, **node_attrs)

            # Add edge with difference flow
            if difference_graph.has_node(u) and difference_graph.has_node(v):
                # Copy geometry from SQ or Dev
                if flows_sq_graph.has_edge(u, v):
                    edge_attrs = flows_sq_graph.get_edge_data(u, v)
                    # Overwrite flow with difference
                    edge_attrs['flow'] = flow_diff
                    difference_graph.add_edge(u, v, **edge_attrs)
                elif flows_dev_graph.has_edge(u, v):
                    edge_attrs = flows_dev_graph.get_edge_data(u, v)
                    edge_attrs['flow'] = flow_diff
                    difference_graph.add_edge(u, v, **edge_attrs)

        return difference_graph

    # Compute passenger flow on network
    OD_matrix_flow = pd.read_csv(paths.OD_STATIONS_KT_ZH_PATH)
    points = gpd.read_file(paths.RAIL_POINTS_PATH)

    # Calculate and visualize passenger flow for status quo (G_status_quo[0])
    flows_on_edges_sq, flows_on_railway_lines_sq = calculate_flow_on_edges(G_status_quo[0], OD_matrix_flow, points)
    plot_flow_graph(flows_on_edges_sq,
                    output_path="plots/passenger_flows/passenger_flow_map_status_quo.png",
                    edge_scale=0.0007,
                    selected_stations=pp.selected_stations,
                    plot_perimeter=True,
                    title="Passenger flow - Status Quo",
                    style="absolute")

    # Calculate and visualize passenger flow for all development scenarios
    for i, graph in enumerate(G_development):
        # Get development ID from lookup table (if available, otherwise use index)
        dev_id = dev_id_lookup.loc[
            i + 1, 'dev_id'] if 'dev_id_lookup' in locals() and i + 1 in dev_id_lookup.index else f"dev_{i + 1}"

        # Calculate passenger flow
        flows_on_edges, flows_on_railway_lines = calculate_flow_on_edges(graph, OD_matrix_flow, points)

        # Create visualizations
        plot_flow_graph(flows_on_edges,
                        output_path=f"plots/passenger_flows/passenger_flow_map_{dev_id}.png",
                        edge_scale=0.0007,
                        selected_stations=pp.selected_stations,
                        plot_perimeter=True,
                        title=f"Passenger flow - Development {dev_id}",
                        style="absolute")

        # Calculate and visualize passenger flow differences for all development scenarios
        dev_id = dev_id_lookup.loc[
            i + 1, 'dev_id'] if 'dev_id_lookup' in locals() and i + 1 in dev_id_lookup.index else f"dev_{i + 1}"

        # Calculate flow difference to status quo
        flow_difference = calculate_flow_difference(G_status_quo[0], graph, OD_matrix_flow, points)

        # Create difference visualization
        plot_flow_graph(flow_difference,
                        output_path=f"plots/passenger_flows/passenger_flow_diff_{dev_id}.png",
                        edge_scale=0.003,
                        selected_stations=pp.selected_stations,
                        plot_perimeter=True,
                        title=f"Passenger flow difference - Development {dev_id}",
                        style="difference")


def compute_tts(dev_id_lookup, od_times_dev, od_times_status_quo, use_cache=False):
    """Compute total travel times and monetize savings."""
    cache_file = paths.TTS_CACHE

    if use_cache:
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file not found: {cache_file!r}")
        with open(cache_file, "rb") as f_in:
            dev_list, monetized_tt, scenario_list = pickle.load(f_in)
        print(f"[compute_tts] Loaded results from cache: {cache_file}")
        return dev_list, monetized_tt, scenario_list

    df_access = pd.read_csv(
        r"data/Network/Rail_Node.csv",
        sep=";",
        decimal=",",
        encoding="ISO-8859-1"
    )

    TTT_status_quo = calculate_total_travel_times(
        od_times_status_quo,
        paths.RANDOM_SCENARIO_CACHE_PATH,
        df_access
    )

    TTT_developments = calculate_total_travel_times(
        od_times_dev,
        paths.RANDOM_SCENARIO_CACHE_PATH,
        df_access
    )

    output_path = "data/costs/traveltime_savings.csv"
    monetized_tt, scenario_list, dev_list = calculate_monetized_tt_savings(
        TTT_status_quo,
        TTT_developments,
        cp.VTTS,
        output_path,
        dev_id_lookup
    )

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "wb") as f_out:
        pickle.dump((dev_list, monetized_tt, scenario_list), f_out)

    print(f"[compute_tts] Computation complete; results written to cache: {cache_file}")
    return dev_list, monetized_tt, scenario_list


def generate_infra_development(use_cache, mod_type, generate_plots=True):
    """Generate infrastructure development scenarios."""
    if use_cache:
        print("  ⚠ Using cached developments - skipping generation and plotting")
        return

    print(f"  → Generating infrastructure developments (mod_type='{mod_type}')")

    if mod_type in ('ALL', 'EXTEND_LINES'):
        # Identifies railway service endpoints, creates a buffer around them, and selects nearby stations
        generate_rail_edges(n=5, radius=20)
        # Filter out unnecessary links
        filter_unnecessary_links(settings.rail_network)
        # Filter links connecting to corridor access points
        only_links_to_corridor()
        calculate_new_service_time()

    if mod_type in ('ALL', 'NEW_DIRECT_CONNECTIONS'):
        print(f"\n  ✓ Generating NEW_DIRECT_CONNECTIONS (mod_type={mod_type})")
        df_network = gpd.read_file(settings.infra_generation_rail_network)
        df_points = gpd.read_file(r'data\Network\processed\points.gpkg')
        G, pos = prepare_Graph(df_network, df_points)

        # Analyze the railway network to find missing connections
        print("\n=== New Direct connections ===")
        print("Identifying missing connections...")
        missing_connections = get_missing_connections(G, pos, print_results=True,
                                                      polygon=settings.perimeter_infra_generation)

        # Conditional plotting based on global visualization config
        if generate_plots:
            print(f"  → Plotting graph to {paths.PLOT_DIRECTORY}")
            plot_graph(G, pos, highlight_centers=True, missing_links=missing_connections,
                       directory=paths.PLOT_DIRECTORY,
                       polygon=settings.perimeter_infra_generation)
            print(f"  ✓ Graph plot complete")
        else:
            print(f"  → Skipping graph plot (visualization disabled)")

        # Generate potential new railway lines
        print("\n=== GENERATING NEW RAILWAY LINES ===")
        new_railway_lines = generate_new_railway_lines(G, missing_connections)

        # Print detailed information about the new lines
        print("\n=== NEW RAILWAY LINES DETAILS ===")
        print_new_railway_lines(new_railway_lines)

        # Export to GeoPackage
        export_new_railway_lines(new_railway_lines, pos, paths.NEW_RAILWAY_LINES_PATH)
        print("\nNew railway lines exported to paths.NEW_RAILWAY_LINES_PATH")

        # Conditional visualization based on global config
        if generate_plots:
            print("\n=== VISUALIZATION ===")
            print("Creating visualization of the network with highlighted missing connections...")

            plots_dir = "plots/missing_connections"
            print(f"  → Creating individual plots in {plots_dir}/")
            plot_lines_for_each_missing_connection(new_railway_lines, G, pos, plots_dir)
            print(f"  ✓ Individual plots complete")
        else:
            print("\n  → Skipping missing connections visualization (visualization disabled)")

        add_railway_lines_to_new_links(paths.NEW_RAILWAY_LINES_PATH, mod_type,
                                       paths.NEW_LINKS_UPDATED_PATH, settings.rail_network)

    combined_gdf = update_network_with_new_links(settings.rail_network, paths.NEW_LINKS_UPDATED_PATH)
    update_stations(combined_gdf, paths.NETWORK_WITH_ALL_MODIFICATIONS)
    create_network_foreach_dev()


def rearange_costs(cost_and_benefits, output_prefix="", csv_only=False):
    """
    Aggregate the single cost elements to one dataframe and create summary.

    Args:
        cost_and_benefits: DataFrame with cost and benefit data
        output_prefix: Prefix for output files (e.g., "_old" for old version)
        csv_only: If True, only generate CSV output (skip .gpkg files)

    Outputs:
        - "data/costs/total_costs_raw{output_prefix}.csv" (without redundant columns)
        - "data/costs/total_costs{output_prefix}.csv"
        - "data/costs/total_costs{output_prefix}_with_geometry.gpkg" (unless csv_only=True)
        - "data/costs/total_costs_summary{output_prefix}.csv" (new summary file)

    Convert all costs in million CHF
    """
    print(f" -> Aggregate costs (output_prefix='{output_prefix}', csv_only={csv_only})")
    aggregate_costs(cost_and_benefits, cp.tts_valuation_period, output_prefix=output_prefix, csv_only=csv_only)
    transform_and_reshape_cost_df(output_prefix=output_prefix, csv_only=csv_only)
    
    # Create standalone summary CSV
    print(f"\n -> Creating cost summary{output_prefix}...")
    include_geometry = not csv_only  # Include geometry only for full version
    create_cost_summary(output_prefix=output_prefix, include_geometry=include_geometry)


def visualize_results(clear_plot_directory=False, plot_preferences=None):
    """Generate all result visualizations."""
    # Define the plot directory
    plot_dir = "plots"

    # Clear only files in the main plot directory if requested
    if clear_plot_directory:
        print(f"Clearing files in plot directory: {plot_dir}")
        for filename in os.listdir(plot_dir):
            file_path = os.path.join(plot_dir, filename)
            try:
                # Only delete files, not directories
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error while clearing {file_path}: {e}")

    # Check if any plots should be generated at all
    if plot_preferences is None:
        plot_preferences = {}

    any_plots_enabled = any(plot_preferences.values()) if plot_preferences else False

    # Generate core visualizations if any plot type is enabled
    if any_plots_enabled:
        # Generate all visualizations (data processing + plotting)
        plotting(input_file="data/costs/total_costs_with_geometry.gpkg",
                 output_file="data/costs/processed_costs.gpkg",
                 node_file="data/Network/Rail_Node.xlsx")

        # Make a plot of the developments
        if plot_preferences.get('small_developments', False):
            plot_developments_expand_by_one_station()

        # Load the dataset and generate plots with user preferences
        results_raw = pd.read_csv("data/costs/total_costs_raw.csv")
        railway_lines = gpd.read_file(paths.NEW_RAILWAY_LINES_PATH)
        create_and_save_plots(df=results_raw, railway_lines=railway_lines, plot_preferences=plot_preferences)

        # Plot cumulative cost distribution (always include if any plots enabled)
        plot_cumulative_cost_distribution(results_raw, "plots/cumulative_cost_distribution.png")
    else:
        print("  → Skipping all result visualizations (no plot types selected)")


# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == '__main__':
    infrascanrail_cap()
