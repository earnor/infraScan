# import packages
import paths
import scoring
import settings
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
from visualization_costs import (
    plot_infrastructure_costs_with_capacity,
    plot_capacity_surplus_by_development,
    export_intervention_details,
    export_cost_breakdown_table,
    plot_bcr_comparison_scatter,
    plot_bcr_by_development,
    export_viability_results
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
# PHASE FUNCTIONS - Modular Pipeline Components
# ================================================================================

def phase_1_initialization(runtimes: dict) -> tuple:
    """
    Phase 1: Initialize workspace and study area boundaries.

    Args:
        runtimes: Dictionary to track phase execution times

    Returns:
        tuple: (innerboundary, outerboundary) - Study area polygons
    """
    print("\n" + "="*80)
    print("PHASE 1: INITIALIZE VARIABLES")
    print("="*80 + "\n")
    st = time.time()

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

    # Auto-select workflow based on network label
    if '_extended' in settings.rail_network:
        print(f"  Using Baseline Extended workflow (all stations) for {settings.rail_network}")
        baseline_exit_code = run_baseline_extended_workflow(
            network_label=settings.rail_network,
            visualize=settings.visualize_capacity_analysis
        )
    else:
        print(f"  Using Baseline workflow (corridor-filtered) for {settings.rail_network}")
        baseline_exit_code = run_baseline_workflow(
            network_label=settings.rail_network,
            visualize=settings.visualize_capacity_analysis
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

    # Run Phase 4 iterative capacity enhancement
    print(f"\n  Running Phase 4 enhancement workflow for {settings.rail_network}...")
    print(f"  Threshold: {capacity_threshold} tphpd")
    print(f"  Max iterations: {max_iterations}\n")

    enhanced_exit_code = run_enhanced_workflow(
        network_label=settings.rail_network,
        threshold=capacity_threshold,
        max_iterations=max_iterations
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

    generate_infra_development(
        use_cache=settings.use_cache_developments,
        mod_type=settings.infra_generation_modification_type
    )

    # Create lookup table for developments
    dev_id_lookup = create_dev_id_lookup_table()
    print(f"  ✓ Generated {len(dev_id_lookup)} infrastructure developments")
    runtimes["Generate infrastructure developments"] = time.time() - st

    # ============================================================================
    # STEP 4.2: ANALYZE DEVELOPMENT CAPACITY
    # ============================================================================
    print("\n--- Step 4.2: Analyze Development Capacity (Workflow 3) ---\n")

    # Ask user if they want to generate plots for all developments
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
        
        # Ask user if they want catchment plots
        response = input("\nGenerate catchment visualization plots? (y/n) [n]: ").strip().lower()
        if response in {'y', 'yes'}:
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
    
    # Ask user if they want to generate passenger flow plots
    print(f"Found {len(G_development)} developments for passenger flow visualization.")
    response = input("\nGenerate passenger flow plots? (y/n) [n]: ").strip().lower()
    
    if response not in {'y', 'yes'}:
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

    get_random_scenarios(
        start_year=2018,
        end_year=2100,
        num_of_scenarios=settings.amount_of_scenarios,
        use_cache=settings.use_cache_scenarios,
        do_plot=True
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


def phase_10_construction_maintenance_costs(monetized_tt: pd.DataFrame, runtimes: dict) -> pd.DataFrame:
    """
    Phase 10: Calculate construction and maintenance costs.

    Args:
        monetized_tt: Monetized travel time savings from Phase 9
        runtimes: Dictionary to track phase execution times

    Returns:
        construction_and_maintenance_costs: DataFrame with infrastructure costs
    """
    print("\n" + "="*80)
    print("PHASE 10: CONSTRUCTION & MAINTENANCE COSTS")
    print("="*80 + "\n")
    st = time.time()

    # Compute construction costs
    file_path = "data/Network/Rail-Service_Link_construction_cost.csv"
    construction_and_maintenance_costs = construction_costs(
        file_path=file_path,
        cost_per_meter=cp.track_cost_per_meter,
        tunnel_cost_per_meter=cp.tunnel_cost_per_meter,
        bridge_cost_per_meter=cp.bridge_cost_per_meter,
        track_maintenance_cost=cp.track_maintenance_cost,
        tunnel_maintenance_cost=cp.tunnel_maintenance_cost,
        bridge_maintenance_cost=cp.bridge_maintenance_cost,
        duration=cp.duration
    )

    runtimes["Compute construction costs"] = time.time() - st
    return construction_and_maintenance_costs


def phase_11_cost_benefit_integration(construction_and_maintenance_costs: pd.DataFrame, runtimes: dict) -> tuple:
    """
    Phase 11: Integrate costs with benefits and apply discounting.

    Creates TWO versions:
    1. Old (WITHOUT capacity interventions)
    2. Current (WITH capacity interventions)

    Args:
        construction_and_maintenance_costs: Construction costs from Phase 10
        runtimes: Dictionary to track phase execution times

    Returns:
        tuple: (costs_and_benefits_old_discounted, costs_and_benefits_discounted)
    """
    print("\n" + "="*80)
    print("PHASE 11: COST-BENEFIT INTEGRATION")
    print("="*80 + "\n")
    st = time.time()

    # Create both versions of cost-benefit dataframes
    costs_and_benefits_old, costs_and_benefits = create_cost_and_benefit_df(
        settings.start_year_scenario,
        settings.end_year_scenario,
        settings.start_valuation_year
    )

    # Apply discounting to OLD version (WITHOUT capacity interventions)
    print("\n  Applying discounting to costs WITHOUT capacity interventions...")
    costs_and_benefits_old_discounted = discounting(
        costs_and_benefits_old,
        discount_rate=cp.discount_rate,
        base_year=settings.start_valuation_year
    )
    old_discounted_path = "data/costs/costs_and_benefits_old_discounted.csv"
    costs_and_benefits_old_discounted.to_csv(old_discounted_path)
    print(f"  ✓ Saved to: {old_discounted_path}")

    # Apply discounting to current version (WITH capacity interventions)
    print("\n  Applying discounting to costs WITH capacity interventions...")
    costs_and_benefits_discounted = discounting(
        costs_and_benefits,
        discount_rate=cp.discount_rate,
        base_year=settings.start_valuation_year
    )
    discounted_path = "data/costs/costs_and_benefits_discounted.csv"
    costs_and_benefits_discounted.to_csv(discounted_path)
    print(f"  ✓ Saved to: {discounted_path}")

    # Ask user if they want visualizations
    print("\n" + "="*80)
    print("VISUALIZATION OPTION")
    print("="*80)
    response = input("\nGenerate cost-benefit plots for all developments? (y/n) [n]: ").strip().lower()

    if response == 'y':
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

    runtimes["Cost-benefit integration"] = time.time() - st
    return costs_and_benefits_old_discounted, costs_and_benefits_discounted


def phase_12_cost_aggregation(runtimes: dict) -> None:
    """
    Phase 12: Aggregate cost elements.

    Processes both old (without capacity) and new (with capacity) cost-benefit files.

    Args:
        runtimes: Dictionary to track phase execution times

    Side Effects:
        - Writes total_costs.gpkg (with capacity)
        - Writes total_costs.csv (with capacity)
        - Writes total_costs_with_geometry.gpkg (with capacity)
        - Writes total_costs_old.csv (without capacity)
    """
    print("\n" + "="*80)
    print("PHASE 12: COST AGGREGATION")
    print("="*80 + "\n")
    st = time.time()

    # Load the discounted cost-benefit dataframes from Phase 11
    costs_path = "data/costs/costs_and_benefits_discounted.csv"
    costs_old_path = "data/costs/costs_and_benefits_old_discounted.csv"

    print(f"  Loading WITH capacity interventions: {costs_path}")
    costs_and_benefits_discounted = pd.read_csv(costs_path)

    print(f"  Loading WITHOUT capacity interventions: {costs_old_path}")
    costs_and_benefits_old_discounted = pd.read_csv(costs_old_path)

    # Process new version (WITH capacity) - full outputs
    print("\n  → Processing costs WITH capacity interventions (full outputs)...")
    rearange_costs(costs_and_benefits_discounted, output_prefix="")

    # Process old version (WITHOUT capacity) - CSV only
    print("\n  → Processing costs WITHOUT capacity interventions (CSV only)...")
    rearange_costs(costs_and_benefits_old_discounted, output_prefix="_old", csv_only=True)

    runtimes["Aggregate costs"] = time.time() - st


def phase_13_results_visualization(runtimes: dict) -> None:
    """
    Phase 13: Generate all result visualizations.

    Args:
        runtimes: Dictionary to track phase execution times

    Side Effects:
        - Writes multiple plot files to plots/
        - Writes processed_costs.gpkg
    """
    print("\n" + "="*80)
    print("PHASE 13: RESULTS VISUALIZATION")
    print("="*80 + "\n")
    st = time.time()

    # User prompts for benefit plot categories
    print("Select which benefit plot categories to generate:")
    print("─" * 60)
    
    plot_small_developments = input("  Generate plots for small developments (Expand 1 Stop)? (y/n): ").strip().lower() == 'y'
    plot_grouped_by_connection = input("  Generate plots grouped by missing connection? (y/n): ").strip().lower() == 'y'
    plot_ranked_groups = input("  Generate ranked group plots (by net benefit)? (y/n): ").strip().lower() == 'y'
    plot_combined_with_maps = input("  Generate combined plots (charts + network maps)? (y/n): ").strip().lower() == 'y'
    
    print("─" * 60)
    
    # Store plot preferences in a dictionary to pass to visualization function
    plot_preferences = {
        'small_developments': plot_small_developments,
        'grouped_by_connection': plot_grouped_by_connection,
        'ranked_groups': plot_ranked_groups,
        'combined_with_maps': plot_combined_with_maps
    }

    visualize_results(clear_plot_directory=False, plot_preferences=plot_preferences)

    runtimes["Visualize results"] = time.time() - st


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
    # AUTO-RESPONSE SETUP: Automatically answer capacity grouping prompts with "1"
    # ============================================================================
    # Store original input function
    if isinstance(__builtins__, dict):
        _original_input = __builtins__['input']
    else:
        _original_input = __builtins__.input

    def auto_input(prompt=""):
        """
        Automatically respond with "1" to capacity grouping prompts,
        otherwise use normal input.
        """
        # Check if this is a capacity grouping prompt
        if "Enter choice (1-2):" in prompt or "Select the strategy number" in prompt:
            print(prompt + "1  [AUTO-SELECTED]")
            return "1"
        # For all other prompts, use original input
        return _original_input(prompt)

    # Replace built-in input with our auto-input function
    if isinstance(__builtins__, dict):
        __builtins__['input'] = auto_input
    else:
        __builtins__.input = auto_input

    ##################################################################################
    # PHASE 1: INITIALIZATION
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
    # PHASE 10: CONSTRUCTION & MAINTENANCE COSTS
    ##################################################################################
    construction_and_maintenance_costs = \
        phase_10_construction_maintenance_costs(monetized_tt, runtimes)

    ##################################################################################
    # PHASE 11: COST-BENEFIT INTEGRATION
    ##################################################################################
    costs_and_benefits_old_discounted, costs_and_benefits_discounted = \
        phase_11_cost_benefit_integration(construction_and_maintenance_costs, runtimes)

    ##################################################################################
    # PHASE 12: COST AGGREGATION
    ##################################################################################
    phase_12_cost_aggregation(runtimes)

    ##################################################################################
    # PHASE 13: RESULTS VISUALIZATION
    ##################################################################################
    # phase_13_results_visualization(runtimes)
 
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
    df_const_costs_grouped = df_const_costs.groupby(['FromNode', 'ToNode'], as_index=False)[rows].sum()
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
                    title="Passagierfluss - Status Quo",
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
                        title=f"Passagierfluss - Entwicklung {dev_id}",
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
                        title=f"Passagierfluss Differenz - Entwicklung {dev_id}",
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


def generate_infra_development(use_cache, mod_type):
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
        print(f"  → Plotting graph to {paths.PLOT_DIRECTORY}")
        plot_graph(G, pos, highlight_centers=True, missing_links=missing_connections,
                   directory=paths.PLOT_DIRECTORY,
                   polygon=settings.perimeter_infra_generation)
        print(f"  ✓ Graph plot complete")

        # Generate potential new railway lines
        print("\n=== GENERATING NEW RAILWAY LINES ===")
        new_railway_lines = generate_new_railway_lines(G, missing_connections)

        # Print detailed information about the new lines
        print("\n=== NEW RAILWAY LINES DETAILS ===")
        print_new_railway_lines(new_railway_lines)

        # Export to GeoPackage
        export_new_railway_lines(new_railway_lines, pos, paths.NEW_RAILWAY_LINES_PATH)
        print("\nNew railway lines exported to paths.NEW_RAILWAY_LINES_PATH")

        # Visualize the new railway lines
        print("\n=== VISUALIZATION ===")
        print("Creating visualization of the network with highlighted missing connections...")

        plots_dir = "plots/missing_connections"
        print(f"  → Creating individual plots in {plots_dir}/")
        plot_lines_for_each_missing_connection(new_railway_lines, G, pos, plots_dir)
        print(f"  ✓ Individual plots complete")
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

    # Generate all visualizations
    plotting(input_file="data/costs/total_costs_with_geometry.gpkg",
             output_file="data/costs/processed_costs.gpkg",
             node_file="data/Network/Rail_Node.xlsx")
    
    # Make a plot of the developments
    plot_developments_expand_by_one_station()
    
    # Load the dataset and generate plots
    results_raw = pd.read_csv("data/costs/total_costs_raw.csv")
    railway_lines = gpd.read_file(paths.NEW_RAILWAY_LINES_PATH)
    create_and_save_plots(df=results_raw, railway_lines=railway_lines)
    
    # Plot cumulative cost distribution
    plot_cumulative_cost_distribution(results_raw, "plots/cumulative_cost_distribution.png")


# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == '__main__':
    infrascanrail_cap()
