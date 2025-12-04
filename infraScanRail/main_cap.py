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
from plots import plot_cumulative_cost_distribution, plot_flow_graph
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


def infrascanrail_cap():
    """
    Enhanced InfraScanRail main pipeline with integrated capacity analysis.

    This version implements the full capacity-aware workflow:
    - Phase 3.2: Establish Baseline Capacity
    - Phase 3.3: Enhance Baseline Network (Phase 4 Interventions)
    - Phase 3.5: Analyze Development Capacity
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
    print("\n" + "="*80)
    print("PHASE 1: INITIALIZE VARIABLES")
    print("="*80 + "\n")
    st = time.time()

    innerboundary, outerboundary = create_focus_area()

    runtimes["Initialize variables"] = time.time() - st

    ##################################################################################
    # PHASE 2: DATA IMPORT
    ##################################################################################
    print("\n" + "="*80)
    print("PHASE 2: IMPORT RAW DATA")
    print("="*80 + "\n")
    st = time.time()

    # Import shapes of lake for plots
    get_lake_data()

    # Import the file containing the locations to be plotted
    import_cities()

    runtimes["Import land use and land cover data"] = time.time() - st

    ##################################################################################
    # PHASE 3: BASELINE CAPACITY ANALYSIS
    ##################################################################################
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
    # STEP 3.2: ESTABLISH BASELINE CAPACITY ⭐ NEW
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
    # STEP 3.3: ENHANCE BASELINE NETWORK (PHASE 4 INTERVENTIONS) ⭐ NEW
    # ============================================================================
    print("\n--- Step 3.3: Enhance Baseline Network (Phase 4) ---\n")
    st = time.time()

    # Run Phase 4 iterative capacity enhancement
    print(f"  Running Phase 4 enhancement workflow for {settings.rail_network}...")
    print(f"  Threshold: {settings.capacity_threshold} tphpd")
    print(f"  Max iterations: {settings.max_enhancement_iterations}\n")

    enhanced_exit_code = run_enhanced_workflow(
        network_label=settings.rail_network,
        threshold=settings.capacity_threshold,
        max_iterations=settings.max_enhancement_iterations
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

    ##################################################################################
    # PHASE 4: INFRASTRUCTURE DEVELOPMENTS
    ##################################################################################
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

    # ============================================================================
    # Via Column Modification for EXTEND_LINES Developments
    # ============================================================================
    # For EXTEND_LINES developments (dev_id 100001-100999):
    # Check new_dev column and set Via = "-99" for new developments
    # This signals no intermediate stations, direct connection
    print("\n  Applying Via column modifications for EXTEND_LINES developments...")

    # Get list of development .gpkg files (same as used in Workflow 3 below)
    dev_dir = Path(paths.DEVELOPMENT_DIRECTORY)
    if dev_dir.exists() and dev_dir.is_dir():
        # Get all .gpkg files in development directory
        gpkg_files = [
            os.path.join(paths.DEVELOPMENT_DIRECTORY, filename)
            for filename in os.listdir(paths.DEVELOPMENT_DIRECTORY)
            if filename.endswith(".gpkg")
        ]
        
        if not gpkg_files:
            print(f"  ⚠ No .gpkg files found in {paths.DEVELOPMENT_DIRECTORY}")
        else:
            total_modifications = 0
            
            for gpkg_file in gpkg_files:
                try:
                    developments_gdf = gpd.read_file(gpkg_file)
                    
                    # Check if new_dev column exists
                    if 'new_dev' not in developments_gdf.columns:
                        continue
                    
                    modifications_count = 0
                    
                    for idx, row in developments_gdf.iterrows():
                        # Only process EXTEND_LINES developments (100001-100999)
                        dev_id = row.get('dev_id', None)
                        
                        # Handle both string and numeric dev_id
                        if dev_id is not None:
                            # Convert to int if it's a float or string
                            try:
                                if isinstance(dev_id, str):
                                    dev_id_num = int(dev_id)
                                elif isinstance(dev_id, (int, float)):
                                    dev_id_num = int(dev_id)
                                else:
                                    continue
                            except (ValueError, TypeError):
                                continue
                            
                            # Check if dev_id is in EXTEND_LINES range (100001-100999)
                            if settings.dev_id_start_extended_lines <= dev_id_num < settings.dev_id_start_new_direct_connections:
                                if row.get('new_dev') == 'Yes':
                                    developments_gdf.at[idx, 'Via'] = '-99'
                                    modifications_count += 1
                    
                    # Save modified geopackage if any modifications were made
                    if modifications_count > 0:
                        developments_gdf.to_file(gpkg_file, driver='GPKG')
                        print(f"  ✓ Modified Via column for {modifications_count} records in {Path(gpkg_file).name}")
                        total_modifications += modifications_count
                        
                except Exception as e:
                    print(f"  ⚠ Error processing {Path(gpkg_file).name}: {e}")
            
            if total_modifications > 0:
                print(f"\n  ✓ Total Via column modifications: {total_modifications}")
            else:
                print(f"\n  ⓘ No EXTEND_LINES developments with new_dev='Yes' found")
    else:
        print(f"  ⚠ Development directory not found or is not a directory: {paths.DEVELOPMENT_DIRECTORY}")
    runtimes["Generate infrastructure developments"] = time.time() - st

    # ============================================================================
    # STEP 4.2: ANALYZE DEVELOPMENT CAPACITY ⭐ NEW
    # ============================================================================
    print("\n--- Step 4.2: Analyze Development Capacity (Workflow 3) ---\n")
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
                visualize=settings.visualize_capacity_analysis
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
    # STEP 3.6: PUBLIC TRANSIT CATCHMENT (OPTIONAL)
    # ============================================================================
    if settings.OD_type == 'pt_catchment_perimeter':
        print("\n--- Step 3.6: Public Transit Catchment Analysis ---\n")
        st = time.time()
        
        get_catchment(use_cache=settings.use_cache_pt_catchment)
        
        runtimes["Public transit catchment"] = time.time() - st

    ##################################################################################
    # PHASE 5: DEMAND ANALYSIS (OD MATRIX)
    ##################################################################################
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

    ##################################################################################
    # PHASE 6: TRAVEL TIME COMPUTATION
    ##################################################################################
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

    ##################################################################################
    # PHASE 7: PASSENGER FLOW VISUALIZATION
    ##################################################################################
    if settings.plot_passenger_flow:
        print("\n" + "="*80)
        print("PHASE 7: PASSENGER FLOW VISUALIZATION")
        print("="*80 + "\n")
        st = time.time()
        
        plot_passenger_flows_on_network(G_development, G_status_quo, dev_id_lookup)
        
        runtimes["Compute and visualize passenger flows on network"] = time.time() - st

    ##################################################################################
    # PHASE 8: SCENARIO GENERATION
    ##################################################################################
    print("\n" + "="*80)
    print("PHASE 8: SCENARIO GENERATION")
    print("="*80 + "\n")
    st = time.time()

    if settings.OD_type == 'canton_ZH':
        get_random_scenarios(
            start_year=2018, 
            end_year=2100, 
            num_of_scenarios=settings.amount_of_scenarios,
            use_cache=settings.use_cache_scenarios, 
            do_plot=True
        )

    runtimes["Generate the scenarios"] = time.time() - st

    ##################################################################################
    # PHASE 9: TRAVEL TIME SAVINGS
    ##################################################################################
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

    ##################################################################################
    # PHASE 10: INFRASTRUCTURE COSTS WITH CAPACITY INTERVENTIONS
    ##################################################################################
    print("\n" + "="*80)
    print("PHASE 10: INFRASTRUCTURE COSTS WITH CAPACITY INTERVENTIONS")
    print("="*80 + "\n")
    st = time.time()

    # Step 10.1: Calculate base infrastructure costs
    print("  Step 10.1: Calculate base infrastructure costs...")
    file_path = "data/Network/Rail-Service_Link_construction_cost.csv"
    
    # Import construction_costs from scoring module
    from scoring import construction_costs
    
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
    
    # DEBUG: Check what columns exist
    print(f"  DEBUG: construction_and_maintenance_costs columns: {construction_and_maintenance_costs.columns.tolist()}")
    print(f"  DEBUG: construction_and_maintenance_costs shape: {construction_and_maintenance_costs.shape}")
    if len(construction_and_maintenance_costs) > 0:
        print(f"  DEBUG: First row:\n{construction_and_maintenance_costs.iloc[0]}")
    
    # Identify or create dev_id column
    if 'dev_id' not in construction_and_maintenance_costs.columns:
        # Check for alternative column names
        possible_dev_columns = ['Line', 'line', 'development', 'Development', 'filename', 'dev', 'id']
        dev_col_found = None
        
        for col in possible_dev_columns:
            if col in construction_and_maintenance_costs.columns:
                print(f"  ⓘ Found development ID column: '{col}', renaming to 'dev_id'")
                construction_and_maintenance_costs = construction_and_maintenance_costs.rename(columns={col: 'dev_id'})
                dev_col_found = True
                break
        
        if not dev_col_found:
            # Check if index contains dev_ids
            if construction_and_maintenance_costs.index.name in ['dev_id', 'Line', 'development']:
                print(f"  ⓘ Using index '{construction_and_maintenance_costs.index.name}' as 'dev_id' column")
                construction_and_maintenance_costs = construction_and_maintenance_costs.reset_index()
                if construction_and_maintenance_costs.columns[0] != 'dev_id':
                    construction_and_maintenance_costs = construction_and_maintenance_costs.rename(
                        columns={construction_and_maintenance_costs.columns[0]: 'dev_id'}
                    )
            else:
                # Last resort: use index as dev_id
                print(f"  ⚠ No dev_id column found. Using index as dev_id.")
                construction_and_maintenance_costs = construction_and_maintenance_costs.reset_index()
                construction_and_maintenance_costs = construction_and_maintenance_costs.rename(
                    columns={'index': 'dev_id'}
                )
    
    print(f"  ✓ Base costs calculated for {len(construction_and_maintenance_costs)} developments\n")

    # Step 10.2: Calculate capacity intervention costs
    print("  Step 10.2: Calculate capacity intervention costs...")
    from development_interventions import calculate_intervention_costs_per_development

    # Path to Phase 3.3 baseline interventions (enhanced network)
    enhanced_network_label = f"{settings.rail_network}_enhanced"
    baseline_interventions_path = CAPACITY_ROOT / "Enhanced" / enhanced_network_label / "capacity_interventions.csv"

    if baseline_interventions_path.exists():
        print(f"    Using interventions from: {baseline_interventions_path}")

        # Use development_interventions module to match and calculate costs
        intervention_costs_df = calculate_intervention_costs_per_development(
            dev_id_lookup=dev_id_lookup,
            baseline_interventions_path=str(baseline_interventions_path),
            capacity_analysis_results=capacity_analysis_results,
            development_directory=paths.DEVELOPMENT_DIRECTORY
        )
        
        print(f"  ✓ Calculated intervention costs for {len(intervention_costs_df)} developments")
        print(f"    Total intervention construction cost: CHF {intervention_costs_df['intervention_construction_cost'].sum():,.0f}")
        print(f"    Total intervention maintenance (annual): CHF {intervention_costs_df['intervention_maintenance_annual'].sum():,.0f}\n")

        # Merge intervention costs with base infrastructure costs
        # Ensure dev_id is the correct type for merging
        print("  Merging intervention costs with base costs...")
        construction_and_maintenance_costs['dev_id'] = construction_and_maintenance_costs['dev_id'].astype(str)
        intervention_costs_df['dev_id'] = intervention_costs_df['dev_id'].astype(str)
        
        # DEBUG: Show sample dev_ids from both dataframes
        print(f"  DEBUG: Sample dev_ids from base costs: {construction_and_maintenance_costs['dev_id'].head().tolist()}")
        print(f"  DEBUG: Sample dev_ids from interventions: {intervention_costs_df['dev_id'].head().tolist()}")
        
        construction_and_maintenance_costs = construction_and_maintenance_costs.merge(
            intervention_costs_df,
            on='dev_id',
            how='left'
        )
        
        # Fill NaN values with 0 for developments without interventions
        construction_and_maintenance_costs['intervention_construction_cost'] = \
            construction_and_maintenance_costs['intervention_construction_cost'].fillna(0.0)
        construction_and_maintenance_costs['intervention_maintenance_annual'] = \
            construction_and_maintenance_costs['intervention_maintenance_annual'].fillna(0.0)
        construction_and_maintenance_costs['intervention_count'] = \
            construction_and_maintenance_costs['intervention_count'].fillna(0).astype(int)
        construction_and_maintenance_costs['intervention_ids'] = \
            construction_and_maintenance_costs['intervention_ids'].fillna('')
        
        # Add total costs (base + interventions)
        construction_and_maintenance_costs['total_construction_cost'] = \
            construction_and_maintenance_costs['TotalConstructionCost'] + \
            construction_and_maintenance_costs['intervention_construction_cost']

        construction_and_maintenance_costs['total_maintenance_cost'] = \
            construction_and_maintenance_costs['YearlyMaintenanceCost'] + \
            construction_and_maintenance_costs['intervention_maintenance_annual']
        
        print(f"  ✓ Intervention costs merged with base infrastructure costs\n")
        
    else:
        print(f"  ⚠ Baseline interventions file not found: {baseline_interventions_path}")
        print(f"    Proceeding with base infrastructure costs only\n")
        
        # Add zero-cost intervention columns for consistency
        construction_and_maintenance_costs['intervention_construction_cost'] = 0.0
        construction_and_maintenance_costs['intervention_maintenance_annual'] = 0.0
        construction_and_maintenance_costs['intervention_count'] = 0
        construction_and_maintenance_costs['intervention_ids'] = ''
        construction_and_maintenance_costs['total_construction_cost'] = \
            construction_and_maintenance_costs['TotalConstructionCost']
        construction_and_maintenance_costs['total_maintenance_cost'] = \
            construction_and_maintenance_costs['YearlyMaintenanceCost']

    runtimes["Compute costs"] = time.time() - st

    ##################################################################################
    # PHASE 12: COST-BENEFIT INTEGRATION, AGGREGATION & VISUALIZATION
    ##################################################################################
    print("\n" + "="*80)
    print("PHASE 12: COST-BENEFIT INTEGRATION, AGGREGATION & VISUALIZATION")
    print("="*80 + "\n")
    st = time.time()

    # Step 12.1: Cost-Benefit Integration
    print("  Step 12.1: Cost-benefit integration...")
    
    # Save enhanced costs to temporary CSV for cost-benefit analysis
    enhanced_costs_path = Path(paths.MAIN) / "data" / "costs" / "construction_costs_with_interventions.csv"
    enhanced_costs_path.parent.mkdir(parents=True, exist_ok=True)
    construction_and_maintenance_costs.to_csv(enhanced_costs_path, index=False)
    
    # Use enhanced costs if interventions were calculated, otherwise use base costs
    if baseline_interventions_path.exists() and 'intervention_construction_cost' in construction_and_maintenance_costs.columns:
        print("    Using intervention-enhanced costs for cost-benefit analysis...")
        cost_file_for_cba = str(enhanced_costs_path)
    else:
        print("    Using base costs for cost-benefit analysis...")
        cost_file_for_cba = None  # Will default to paths.CONSTRUCTION_COSTS

    cost_and_benefits_dev = create_cost_and_benefit_df(
        settings.start_year_scenario,
        settings.end_year_scenario,
        settings.start_valuation_year,
        cost_file_path=cost_file_for_cba
    )
    costs_and_benefits_dev_discounted = discounting(
        cost_and_benefits_dev,
        discount_rate=cp.discount_rate,
        base_year=settings.start_valuation_year
    )
    costs_and_benefits_dev_discounted.to_csv(paths.COST_AND_BENEFITS_DISCOUNTED)
    
    # Generate example cost-benefit plot if visualization function exists
    try:
        from display_results import plot_costs_benefit_example
        plot_costs_benefit_example(costs_and_benefits_dev_discounted, line='101032.0')
    except ImportError:
        print("    ⓘ plot_costs_benefit_example not available, skipping example plot")
    except Exception as e:
        print(f"    ⚠ Could not generate cost-benefit example plot: {e}")
    
    print("    ✓ Cost-benefit integration complete\n")

    runtimes["Cost-benefit integration"] = time.time() - st

    ##################################################################################
    # PHASE 11: VIABILITY ASSESSMENT
    ##################################################################################
    print("\n" + "="*80)
    print("PHASE 11: VIABILITY ASSESSMENT")
    print("="*80 + "\n")
    st = time.time()

    # Calculate viability with and without capacity intervention costs
    viability_results = {}
    bcr_threshold = getattr(settings, 'bcr_threshold', 1.0)

    print("  Calculating benefit-cost ratios with/without capacity interventions...\n")

    # Extract benefits from cost_and_benefits dataframe
    # Aggregate NPV benefits by development
    benefits_by_dev = costs_and_benefits_dev_discounted.groupby('dev_id')['NPV_TT_Savings'].sum().to_dict()

    for dev_id in dev_id_lookup['dev_id']:
        # Get benefits
        benefits = benefits_by_dev.get(dev_id, 0.0)

        # Get costs from construction_and_maintenance_costs
        cost_row = construction_and_maintenance_costs[
            construction_and_maintenance_costs['dev_id'] == dev_id
        ]

        if cost_row.empty:
            continue

        cost_row = cost_row.iloc[0]

        # Costs WITHOUT capacity interventions
        costs_without_capacity = (
            cost_row['TotalConstructionCost'] +
            cost_row['YearlyMaintenanceCost'] * cp.duration
        )

        # Costs WITH capacity interventions
        costs_with_capacity = cost_row['TotalCostWithInterventions']

        # Calculate BCRs
        bcr_without_capacity = benefits / costs_without_capacity if costs_without_capacity > 0 else 0.0
        bcr_with_capacity = benefits / costs_with_capacity if costs_with_capacity > 0 else 0.0

        # Viability checks
        viable_without = bcr_without_capacity >= bcr_threshold
        viable_with = bcr_with_capacity >= bcr_threshold

        # Store results
        viability_results[dev_id] = {
            'benefits': benefits,
            'costs_without_capacity': costs_without_capacity,
            'costs_with_capacity': costs_with_capacity,
            'bcr_without_capacity': bcr_without_capacity,
            'bcr_with_capacity': bcr_with_capacity,
            'viable_without_capacity': viable_without,
            'viable_with_capacity': viable_with,
            'capacity_impact_on_bcr': bcr_without_capacity - bcr_with_capacity
        }

    # Print viability assessment report
    print("\n" + "="*80)
    print("VIABILITY ASSESSMENT REPORT")
    print("="*80 + "\n")

    for dev_id in sorted(viability_results.keys()):
        v = viability_results[dev_id]

        print(f"Development: {dev_id}")
        print(f"  Benefits (PV):                 CHF {v['benefits']/1e6:.1f} M")
        print(f"  Costs WITHOUT Capacity (PV):   CHF {v['costs_without_capacity']/1e6:.1f} M")
        print(f"  Costs WITH Capacity (PV):      CHF {v['costs_with_capacity']/1e6:.1f} M")
        print(f"  ")
        print(f"  BCR WITHOUT Capacity:          {v['bcr_without_capacity']:.2f}  {'✓ VIABLE' if v['viable_without_capacity'] else '✗ NOT VIABLE'}")
        print(f"  BCR WITH Capacity:             {v['bcr_with_capacity']:.2f}  {'✓ VIABLE' if v['viable_with_capacity'] else '✗ NOT VIABLE'}")
        print(f"  Capacity Impact on BCR:        {v['capacity_impact_on_bcr']:.2f}")
        print("-"*80 + "\n")

    # Summary statistics
    total_developments = len(viability_results)
    viable_without_count = sum([v['viable_without_capacity'] for v in viability_results.values()])
    viable_with_count = sum([v['viable_with_capacity'] for v in viability_results.values()])

    print(f"SUMMARY:")
    print(f"  Total Developments:                    {total_developments}")
    print(f"  Viable WITHOUT Capacity Costs:         {viable_without_count} ({viable_without_count/total_developments*100:.1f}%)")
    print(f"  Viable WITH Capacity Costs:            {viable_with_count} ({viable_with_count/total_developments*100:.1f}%)")
    print(f"  Developments Made Unviable by Capacity: {viable_without_count - viable_with_count}")
    print("="*80 + "\n")

    # Generate viability visualizations
    print("  Generating viability visualizations...")
    try:
        plot_bcr_comparison_scatter(viability_results, bcr_threshold=bcr_threshold)
        plot_bcr_by_development(viability_results, bcr_threshold=bcr_threshold)
        export_viability_results(viability_results)
        print(f"  ✓ Viability visualizations generated\n")
    except Exception as e:
        print(f"  ⚠ Error generating viability visualizations: {e}\n")

    runtimes["Viability assessment"] = time.time() - st

    ##################################################################################
    # PHASE 12 CONTINUED: COST AGGREGATION & VISUALIZATION
    ##################################################################################
    print("\n" + "="*80)
    print("PHASE 12 CONTINUED: COST AGGREGATION & VISUALIZATION")
    print("="*80 + "\n")
    st = time.time()

    # Step 12.2: Cost Aggregation
    print("  Step 12.2: Cost aggregation...")
    rearange_costs(costs_and_benefits_dev_discounted)
    print("    ✓ Costs aggregated\n")

    # Step 12.3: Results Visualization
    print("  Step 12.3: Results visualization...")
    visualize_results(clear_plot_directory=False)
    print("    ✓ Visualizations generated\n")

    runtimes["Cost aggregation & visualization"] = time.time() - st

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
        print("use cache for developments")
        return

    if mod_type in ('ALL', 'EXTEND_LINES'):
        # Identifies railway service endpoints, creates a buffer around them, and selects nearby stations
        generate_rail_edges(n=5, radius=20)
        # Filter out unnecessary links
        filter_unnecessary_links(settings.rail_network)
        # Filter links connecting to corridor access points
        only_links_to_corridor()
        calculate_new_service_time()

    if mod_type in ('ALL', 'NEW_DIRECT_CONNECTIONS'):
        df_network = gpd.read_file(settings.infra_generation_rail_network)
        df_points = gpd.read_file(r'data\Network\processed\points.gpkg')
        G, pos = prepare_Graph(df_network, df_points)

        # Analyze the railway network to find missing connections
        print("\n=== New Direct connections ===")
        print("Identifying missing connections...")
        missing_connections = get_missing_connections(G, pos, print_results=True,
                                                      polygon=settings.perimeter_infra_generation)
        plot_graph(G, pos, highlight_centers=True, missing_links=missing_connections, 
                   directory=paths.PLOT_DIRECTORY,
                   polygon=settings.perimeter_infra_generation)

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
        plot_lines_for_each_missing_connection(new_railway_lines, G, pos, plots_dir)
        add_railway_lines_to_new_links(paths.NEW_RAILWAY_LINES_PATH, mod_type, 
                                       paths.NEW_LINKS_UPDATED_PATH, settings.rail_network)

    combined_gdf = update_network_with_new_links(settings.rail_network, paths.NEW_LINKS_UPDATED_PATH)
    update_stations(combined_gdf, paths.NETWORK_WITH_ALL_MODIFICATIONS)
    create_network_foreach_dev()


def rearange_costs(cost_and_benefits):
    """
    Aggregate the single cost elements to one dataframe.
    New dataframe is stored in "data/costs/total_costs.gpkg" and "data/costs/total_costs.csv"
    Convert all costs in million CHF
    """
    print(" -> Aggregate costs")
    aggregate_costs(cost_and_benefits, cp.tts_valuation_period)
    transform_and_reshape_cost_df()


def visualize_results(clear_plot_directory=False):
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
