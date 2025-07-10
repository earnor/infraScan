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
import geopandas as gpd
import os
import warnings
import cost_parameters as cp
import plot_parameter as pp


def infrascanrail():
    os.chdir(paths.MAIN)
    warnings.filterwarnings("ignore") #TODO:No warnings should be ignored, but this is necessary for the current code to have a clean output
    runtimes = {}

    ##################################################################################
    # Initializing global variables
    print("\nINITIALIZE VARIABLES \n")
    st = time.time()

    innerboundary, outerboundary = create_focus_area()

    runtimes["Initialize variables"] = time.time() - st
    st = time.time()

    ##################################################################################
    # Import and prepare raw data
    print("\nIMPORT RAW DATA \n")

    # Import shapes of lake for plots
    get_lake_data()

    # Import the file containing the locations to be ploted
    import_cities()

    # Define area that is protected for constructing railway links
    #   get_protected_area(limits=limits_corridor)
    #   get_unproductive_area(limits=limits_corridor)
    #   landuse(limits=limits_corridor)

    # Tif file of all unsuitable land cover and protected areas
    # File is stored to 'data\landuse_landcover\processed\zone_no_infra\protected_area_{suffix}.tif'

    # all_protected_area_to_raster(suffix="corridor")

    runtimes["Import land use and land cover data"] = time.time() - st
    st = time.time()

    print("\nINFRASTRUCTURE NETWORK \n")

    ##################################################################################
    # 2) Process network
    ##################################################################################
    ##################################################################################
    # INFRASTRUCTURE NETWORK
    # 1) Import&Process network

    # Import the railway network and preprocess it

    points = import_process_network(settings.use_cache_network)

    runtimes["Preprocess the network"] = time.time() - st
    st = time.time()
    ##################################################################################
    # 2) Generate developments (new connections)

    generate_infra_development(use_cache=settings.use_cache_developments, mod_type=settings.infra_generation_modification_type)

    runtimes["Generate infrastructure developments"] = time.time() - st
    st = time.time()

    # Compute the catchement area for the status quo and for all developments based on access time to train station
    if settings.OD_type == 'pt_catchment_perimeter':
        get_catchment(use_cache=settings.use_cache_pt_catchment)

    #runtimes["Generate The Catchement based on the Bus network"] = time.time() - st
    #st = time.time()

    # here would code be needed to get all catchements for the different developments, if access point are added


    if settings.OD_type == 'canton_ZH':
        # Filtere Punkte innerhalb der innerboundary
        points_in_inner_boundary = points[points.apply(lambda row: innerboundary.contains(row.geometry), axis=1)]

        # Liste der Einträge erstellen (z.B. die ID_point und NAME)
        inner_boundary_stations = points_in_inner_boundary[['ID_point', 'NAME']].values.tolist()
        #stationOD also saved as a file
        getStationOD(settings.use_cache_stationsOD, inner_boundary_stations, settings.only_demand_from_to_perimeter)
        #getScenarios(od_directory_scenario, pd.read_csv(paths.OD_STATIONS_KT_ZH_PATH))


    elif settings.OD_type == 'pt_catchment_perimeter':
        od_directory_scenario = r"data/traffic_flow/od/rail"
        GetCatchmentOD(settings.use_cache_catchmentOD)
    else:
        raise ValueError("OD_type must be either 'canton_ZH' or 'pt_catchment_perimeter'")

    runtimes["Generate OD matrix"] = time.time() - st
    st = time.time()

    ##################################################################################
    ##################################################################################
    # SCENARIO
    # 1) Define scenario based on cantonal predictions

    print("\nSCENARIO \n")
    ##################################################################################
    # 1) Define scenario based on cantonal predictions



    ##################################################################################
    ##################################################################################
    # IMPLEMENT THE SCORING
    # 1) Compute construction and maintenance costs
    # 2) Compute Traveltime Savings

    print("\nIMPLEMENT SCORING \n")

    ##################################################################################
    # 1) Calculate Traveltimes for all OD_ for all developments
    # Constructs a directed graph from the railway network GeoPackage, 
    # adding nodes (stations) and edges (connections) with travel and service data.
    # Computes an OD matrix using Dijkstra's algorithm, 
    # calculates travel times with penalties for line changes, and stores full path geometries.
    # Returns the graph (nx.DiGraph) and a DataFrame with OD travel data including adjusted travel times and geometries.

    # network of status quo

    dev_id_lookup = create_dev_id_lookup_table()
    od_times_dev, od_times_status_quo, G_status_quo, G_development = create_travel_time_graphs(settings.rail_network, settings.use_cache_traveltime_graph, dev_id_lookup)

    #Compute Passenger flow on network
    OD_matrix_flow = pd.read_csv(paths.OD_STATIONS_KT_ZH_PATH)
    points = gpd.read_file(paths.RAIL_POINTS_PATH)
    flows_on_edges, flows_on_railway_lines = calculate_flow_on_edges(G_status_quo[0], OD_matrix_flow, points)
    plot_flow_graph(flows_on_edges, output_path="plots/passenger_flows/passenger_flow_map.png", edge_scale=0.0007, selected_stations=pp.selected_stations, plot_perimeter = True)
    plot_flow_graph(flows_on_railway_lines, output_path="plots/passenger_flows/passenger_flow_map2.png", edge_scale=0.0007, selected_stations=pp.selected_stations)
    plot_line_flows(flows_on_railway_lines, paths.RAIL_SERVICES_AK2035_EXTENDED_PATH, output_path="plots/passenger_flows/railway_line_load.png")
    runtimes["Calculate Traveltimes for all developments"] = time.time() - st
    st = time.time()

    runtimes["Compute and visualize passenger flows on network"] = time.time() - st
    st = time.time()

    #################################################################################

    # Compute the OD matrix for the current infrastructure under all scenarios
    if settings.OD_type == 'canton_ZH':
        get_random_scenarios(start_year=2018, end_year=2100, num_of_scenarios=settings.amount_of_scenarios,
                             use_cache=settings.use_cache_scenarios, do_plot=True)

    elif settings.OD_type == 'pt_catchment_perimeter':runtimes["Generate the scenarios"] = time.time() - st
    st = time.time()

    dev_list, monetized_tt, scenario_list = compute_tts(dev_id_lookup=dev_id_lookup, od_times_dev= od_times_dev,
                                                        od_times_status_quo=od_times_status_quo, use_cache = settings.use_cache_tts_calc)


    runtimes["Calculate the TTT Savings"] = time.time() - st
    st = time.time()

    # 2) Compute construction costs

    ##here a check for capacity could be added
    # Compute the construction costs for each development
    file_path = "data/Network/Rail-Service_Link_construction_cost.csv"
    construction_and_maintenance_costs = construction_costs(file_path=file_path,
                                                            cost_per_meter=cp.track_cost_per_meter,
                                                            tunnel_cost_per_meter=cp.tunnel_cost_per_meter,
                                                            bridge_cost_per_meter=cp.bridge_cost_per_meter,
                                                            track_maintenance_cost=cp.track_maintenance_cost,
                                                            tunnel_maintenance_cost=cp.tunnel_maintenance_cost,
                                                            bridge_maintenance_cost=cp.bridge_maintenance_cost,
                                                            duration=cp.duration)


    cost_and_benefits_dev = create_cost_and_benefit_df(settings.start_year_scenario, settings.end_year_scenario)
    costs_and_benefits_dev_discounted = discounting(cost_and_benefits_dev, discount_rate=cp.discount_rate, base_year=settings.start_year_scenario)
    costs_and_benefits_dev_discounted.to_csv(paths.COST_AND_BENEFITS_DISCOUNTED)

    runtimes["Compute costs"] = time.time() - st
    st = time.time()


    rearange_costs(costs_and_benefits_dev_discounted)

    runtimes["Aggregate costs"] = time.time() - st

    # Write runtimes to a file


    ##################################################################################
    # VISUALIZE THE RESULTS

    print("\nVISUALIZE THE RESULTS \n")

    visualize_results(clear_plot_directory=False)
    plot_costs_benefits_example(costs_and_benefits_dev_discounted)  # only plots cost&benefits for the dev with highest tts

    runtimes["Visualize results"] = time.time() - st
    st = time.time()

    with open(r'runtimes.txt', 'w') as file:
        for part, runtime in runtimes.items():
            file.write(f"{part}: {runtime}/n")

    # Run the display results function to launch the GUI
    # Call the function to create_scenario_analysis_viewerreate and display the GUI
    #create_scenario_analysis_viewer(paths.TOTAL_COST_WITH_GEOMETRY)


def compute_tts(dev_id_lookup,
                od_times_dev,
                od_times_status_quo,
                use_cache = False):
    """
    Computes total travel times (TTT) for status quo and developments,
    monetizes travel-time savings, and either saves to or loads from cache.

    Args:
        dev_id_lookup:            (whatever your code expects)
        od_directory_scenario:    directory or mapping for OD files
        od_times_dev:             OD times for developments
        od_times_status_quo:      OD times for status quo
        use_cache (bool):         if True, load result from cache file;
                                  if False, do full computation and then write cache.
    Returns:
        dev_list, monetized_tt, scenario_list
    """
    cache_file = paths.TTS_CACHE

    if use_cache:
        # 2) If use_cache=True, try to load from disk
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Cache file not found: {cache_file!r}")
        with open(cache_file, "rb") as f_in:
            dev_list, monetized_tt, scenario_list = pickle.load(f_in)
        print(f"[compute_tts] Loaded results from cache: {cache_file}")
        return dev_list, monetized_tt, scenario_list

    # 3) If we reach here, use_cache=False ⇒ do the “full” computation
    df_access = pd.read_csv(
        r"data/Network/Rail_Node.csv",
        sep=";",
        decimal=",",
        encoding="ISO-8859-1"
    )

    # Compute TTT for status quo
    TTT_status_quo = calculate_total_travel_times(
        od_times_status_quo,
        paths.RANDOM_SCENARIO_CACHE_PATH,
        df_access
    )

    # Compute TTT for developments
    TTT_developments = calculate_total_travel_times(
        od_times_dev,
        paths.RANDOM_SCENARIO_CACHE_PATH,
        df_access
    )

    print("TTT_status_quo:", TTT_status_quo)
    print("TTT_developments:", TTT_developments)

    # Monetize travel‐time savings
    output_path = "data/costs/traveltime_savings.csv"
    monetized_tt, scenario_list, dev_list = calculate_monetized_tt_savings(
        TTT_status_quo,
        TTT_developments,
        cp.VTTS,
        output_path,
        dev_id_lookup
    )

    # 4) Once computed, ensure the cache directory exists and write out the tuple
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "wb") as f_out:
        pickle.dump((dev_list, monetized_tt, scenario_list), f_out)

    print(f"[compute_tts] Computation complete; results written to cache: {cache_file}")
    return dev_list, monetized_tt, scenario_list


def import_process_network(use_cache):
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
    if use_cache:
        return
    else:
        communalOD = scoring.GetOevDemandPerCommune(tau=1)
        communes_to_stations = pd.read_excel(paths.COMMUNE_TO_STATION_PATH)
        railway_station_OD = aggregate_commune_od_to_station_od(communalOD, communes_to_stations)
        if only_demand_from_to_corridor:
            railway_station_OD = filter_od_matrix_by_stations(railway_station_OD, stations_in_perimeter)
        railway_station_OD.to_csv(paths.OD_STATIONS_KT_ZH_PATH)




def getScenarios(od_directory_scenario, railway_station_OD):
    # create dummy scenarios
    # TODO: remove this part, when the scenarios are defined
    scenario_list_dummy = dummy_generate_scenarios(settings.amount_of_scenarios, 12)
    # Für jedes Szenario eine neue OD-Matrix erstellen und speichern
    for i, scenario in enumerate(scenario_list_dummy):
        # Erstelle eine neue OD-Matrix basierend auf der Status-quo OD-Matrix
        scenario_od = railway_station_OD.copy() * scenario['general_factor']

        file_path = os.path.join(od_directory_scenario, f"od_matrix_stations_ktzh_future_{i + 1}.csv")
        scenario_od.to_csv(file_path)


def add_construction_info_to_network():
    const_cost_path = r"data/Network/Rail-Service_Link_construction_cost.csv"
    rows = ['NumOfTracks', 'Bridges m', 'Tunnel m', 'TunnelTrack',
            'tot length m', 'length of 1', 'length of 2 ', 'length of 3 and more']
    df_railway_network = gpd.read_file(paths.RAIL_SERVICES_AK2035_PATH)
    df_const_costs = pd.read_csv(const_cost_path, sep=";", decimal=",")
    # Aggregate costs in case of duplicates
    df_const_costs_grouped = df_const_costs.groupby(['FromNode', 'ToNode'], as_index=False)[rows].sum()
    # Add missing columns to the main df
    new_columns = [col for col in rows if col not in df_railway_network.columns]
    if new_columns:
        df_railway_network[new_columns] = 0
    # Merge on FromNode and ToNode
    df_railway_network = df_railway_network.merge(df_const_costs_grouped, on=['FromNode', 'ToNode'], how='left',
                                                  suffixes=('', '_new'))
    # Update values only if not already present
    for col in rows:
        df_railway_network[col] = df_railway_network[col + '_new'].fillna(df_railway_network[col])
        df_railway_network.drop(columns=[col + '_new'], inplace=True)
    # Save the updated DataFrame to a new file
    df_railway_network.to_file(paths.RAIL_SERVICES_AK2035_PATH)


def create_travel_time_graphs(network_selection, use_cache, dev_id_lookup_table):
    # Define cache file for pickle
    cache_file = 'data/Network/travel_time/cache/od_times.pkl'

    od_nodes = [
        'Rüti ZH', 'Nänikon-Greifensee', 'Uster', 'Wetzikon ZH',
        'Zürich Altstetten', 'Schwerzenbach ZH', 'Fehraltorf',
        'Bubikon', 'Zürich HB', 'Kempten', 'Pfäffikon ZH',
        'Zürich Oerlikon', 'Zürich Stadelhofen', 'Hinwil', 'Aathal',
        'Winterthur', 'Effretikon', 'Dübendorf', 'Rapperswil'
    ]

    if use_cache:
        if os.path.exists(cache_file):
            print("Load OD-times from cache...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                od_times_dev = cache_data['od_times_dev']
                od_times_status_quo = cache_data['od_times_status_quo']
                G_status_quo = cache_data['G_status_quo']
                G_developments = cache_data['G_developments']
            analyze_travel_times(od_times_status_quo, od_times_dev, od_nodes,
                                 dev_id_lookup_table)  # output of this is not used!
            return od_times_dev, od_times_status_quo, G_status_quo, G_developments
        else:
            print("Cache not found. Calculate OD-times...")

    network_status_quo = [get_rail_services_path(network_selection)]
    G_status_quo = create_graphs_from_directories(network_status_quo)
    od_times_status_quo = calculate_od_pairs_with_times_by_graph(G_status_quo)
    # Example usage Test1
    origin_station = "Uster"
    destination_station = "Zürich HB"
    find_fastest_path(G_status_quo[0], origin_station, destination_station)
    # Example usage Test2
    origin_station = "Uster"
    destination_station = "Pfäffikon ZH"
    find_fastest_path(G_status_quo[0], origin_station, destination_station)
    # networks with all developments
    # get the paths of all developments
    directories_dev = [os.path.join(paths.DEVELOPMENT_DIRECTORY, filename)
                       for filename in os.listdir(paths.DEVELOPMENT_DIRECTORY) if filename.endswith(".gpkg")]
    directories_dev = [path.replace("\\", "/") for path in directories_dev]
    G_developments = create_graphs_from_directories(directories_dev)
    od_times_dev = calculate_od_pairs_with_times_by_graph(G_developments)  # OD-time for each development
    # Example usage Test1 for development 1007 (New Link Uster-Pfäffikon)
    origin_station = "Uster"
    destination_station = "Zürich HB"
    find_fastest_path(G_developments[5], origin_station, destination_station)
    # Example usage Test2
    origin_station = "Uster"
    destination_station = "Pfäffikon ZH"
    find_fastest_path(G_developments[5], origin_station, destination_station)
    # Example usage Development 8 (Wetikon to Hinwil (S3))
    origin_station = "Kempten"
    destination_station = "Hinwil"
    find_fastest_path(G_status_quo[0], origin_station, destination_station)
    # Example usage Test2
    origin_station = "Kempten"
    destination_station = "Hinwil"
    find_fastest_path(G_developments[7], origin_station, destination_station)
    #selected_indices = [0, 1, 2, 3, 4, 5, 6, 7]  # Indices of selected developments


    # Analyse der Delta-Reisezeiten

    # Ergebnis anzeigen
    print("\nFinal travel times:")

    # Cache-Ausgabe mit Pickle (kann Listen von DataFrames verarbeiten)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'od_times_dev': od_times_dev,
            'od_times_status_quo': od_times_status_quo,
            'G_status_quo': G_status_quo,
            'G_developments': G_developments
        }, f)
    print("OD-times saved to cache.")
    analyze_travel_times(od_times_status_quo, od_times_dev, od_nodes,
                         dev_id_lookup_table)  # output of this is not used!


    return od_times_dev, od_times_status_quo, G_status_quo, G_developments


def rearange_costs(cost_and_benefits):
    ##################################################################################
    # Aggregate the single cost elements to one dataframe
    # New dataframe is stored in "data/costs/total_costs.gpkg"
    # New dataframe also stored in "data/costs/total_costs.csv"
    # Convert all costs in million CHF
    print(" -> Aggregate costs")
    aggregate_costs(cost_and_benefits, cp.tts_valuation_period)
    transform_and_reshape_cost_df()


def generate_scenarios():
    # Import the predicted scenario defined by the canton of Zürich
    # Define the relative growth per scenario and district
    # The growth rates are stored in "data/temp/data_scenario_n.shp"
    # future_scenario_zuerich_2022(scenario_zh)
    # Plot the growth rates as computed above for population and employment and over three scenarios
    # plot_2x3_subplots(scenario_polygon, outerboundary, network, location)
    # Calculates population growth allocation across nx3 scenarios for municipalities within a defined corridor.
    # For each scenario, adjusts total growth and distributes it among municipalities with urban, equal, and rural biases.
    # Merges growth results with spatial boundaries to form a GeoDataFrame of growth projections for mapping.
    # Saves the resulting GeoDataFrame to a shapefile.
    limits_variables = [2680600, 1227700, 2724300, 1265600]
    future_scenario_pop(n=3)
    future_scenario_empl(n=3)
    # Compute the predicted amount of population and employment in each raster cell (hectar) for each scenario
    # The resulting raster data are stored in "data/independent_variables/scenario/{col}.tif" with col being pop or empl and the scenario
    scenario_to_raster_pop(limits_variables)
    scenario_to_raster_emp(limits_variables)
    # Aggregate the the scenario data to over the voronoi polygons, here euclidian polygons
    # Store the resulting file to "data/Voronoi/voronoi_developments_euclidian_values.shp"
    # scenario_to_voronoi(polygons_gdf, euclidean=True)
    # Convert multiple tif files to one same tif with multiple bands
    stack_tif_files(var="empl")
    stack_tif_files(var="pop")


def visualize_results(clear_plot_directory=False):
    # Define the plot directory
    plot_dir = "plots"

    # Clear only files in the main plot directory if requested
    if clear_plot_directory:
        print(f"Clearing files in plot directory: {plot_dir}")
        for filename in os.listdir(plot_dir):
            file_path = os.path.join(plot_dir, filename)
            try:
                # Nur Dateien löschen, keine Verzeichnisse
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error while clearing {file_path}: {e}")

    # Generate all visualizations

    plotting(input_file="data/costs/total_costs_with_geometry.gpkg",
             output_file="data/costs/processed_costs.gpkg",
             node_file="data/Network/Rail_Node.xlsx")
    # make a plot of the developments
    plot_developments_expand_by_one_station()
    # plot the scenarios
    #plot_scenarios()
    # make a plot of the catchement with id and times
    #create_plot_catchement()
    #create_catchement_plot_time()
    # plot the empl and pop with the comunal boarders and the catchment
    # to visualize the OD-Transformation 
    # plot_catchment_and_distributions(
    #     s_bahn_lines_path="data/Network/processed/split_s_bahn_lines.gpkg",
    #     water_bodies_path="data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp",
    #     catchment_raster_path="data/catchment_pt/catchement.tif",
    #     communal_borders_path="data/_basic_data/Gemeindegrenzen/UP_GEMEINDEN_F.shp",
    #     population_raster_path="data/independent_variable/processed/raw/pop20.tif",
    #     employment_raster_path="data/independent_variable/processed/raw/empl20.tif",
    #     extent_path="data/_basic_data/innerboundary.shp"
    # )
    # Load the dataset and generate plots:
    # - Enhanced boxplot and strip plot for monetized savings by development.
    # Plots are saved in the 'plots' directory.
    results_raw = pd.read_csv("data/costs/total_costs_raw.csv")
    railway_lines = gpd.read_file(paths.NEW_RAILWAY_LINES_PATH)
    create_and_save_plots(df=results_raw, railway_lines=railway_lines)
    # Mit dem vorhandenen DataFrame
    plot_cumulative_cost_distribution(results_raw, "plots/cumulative_cost_distribution.png")
    '''
    plot_developments_and_table_for_scenarios(
    osm_file="data/osm_map.gpkg",  # Use the converted GeoPackage file
    input_dir="data/costs",
    output_dir="data/plots")
    '''


def generate_infra_development(use_cache, mod_type):
    if use_cache:
        print("use cache for developments")
        return

    if mod_type in ('ALL', 'EXTEND_LINES'):
        # Identifies railway service endpoints, creates a buffer around them, and selects nearby stations within a specified radius and count (n).
        # It then generates new edges between these points and saves the resulting datasets for further use.
        # Then it calculates Traveltime, using only the existing infrastructure
        # Then it creates a new Network for each development and saves them as a GPGK
        generate_rail_edges(n=5, radius=20)
        # Filter out unnecessary links in the new_links GeoDataFrame by ensuring the connection is not redundant
        # by ensuring the connection is not redundant within the existing Sline routes
        filter_unnecessary_links(settings.rail_network)
        # Import the generated points as dataframe
        # Filter the generated links that connect to one of the access point within the considered corridor
        # These access points are defined in the manually defined list of access points
        # The links to corridor are stored in "data/Network/processed/developments_to_corridor_attribute.gpkg"
        # The generated points with link to access point in the corridor are stored in "data/Network/processed/generated_nodes_connecting_corridor.gpkg"
        # The end point [ID_new] of developments_to_corridor_attribute are equivalent to the points in generated_nodes_connecting_corridor
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
        plot_graph(G, pos, highlight_centers=True, missing_links=missing_connections, directory=paths.PLOT_DIRECTORY,
                   polygon=settings.perimeter_infra_generation)

        # Generate potential new railway lines
        print("\n=== GENERATING NEW RAILWAY LINES ===")
        new_railway_lines = generate_new_railway_lines(G, missing_connections)

        # Print detailed information about the new lines
        print("\n=== NEW RAILWAY LINES DETAILS ===")
        print_new_railway_lines(new_railway_lines)

        # Export to GeoPackage for further analysis and visualization in GIS software
        export_new_railway_lines(new_railway_lines, pos, paths.NEW_RAILWAY_LINES_PATH)
        print("\nNew railway lines exported to paths.NEW_RAILWAY_LINES_PATH")

        # Visualize the new railway lines on the network graph
        print("\n=== VISUALIZATION ===")
        print("Creating visualization of the network with highlighted missing connections...")

        # Create a directory for individual connection plots if it doesn't exist

        plots_dir = "plots/missing_connections"
        plot_lines_for_each_missing_connection(new_railway_lines, G, pos, plots_dir)
        add_railway_lines_to_new_links(paths.NEW_RAILWAY_LINES_PATH, mod_type, paths.NEW_LINKS_UPDATED_PATH, settings.rail_network)

    combined_gdf = update_network_with_new_links(settings.rail_network, paths.NEW_LINKS_UPDATED_PATH)
    update_stations(combined_gdf, paths.NETWORK_WITH_ALL_MODIFICATIONS)
    create_network_foreach_dev()



def create_focus_area():
    # Define spatial limits of the research corridor
    # The coordinates must end with 000 in order to match the coordinates of the input raster data
    e_min, e_max = 2687000, 2708000  # 2688000, 2704000 - 2688000, 2705000
    n_min, n_max = 1237000, 1254000  # 1238000, 1252000 - 1237000, 1252000
    # For global operation a margin is added to the boundary
    margin = 3000  # meters
    # Define the size of the resolution of the raster to 100 meter
    # save spatial limits as shp
    innerboundary, outerboundary = save_focus_area_shapefile(e_min, e_max, n_min, n_max, margin)
    return innerboundary, outerboundary

def create_dev_id_lookup_table():
    """
    Creates a lookup table (DataFrame) of development filenames from the directory
    specified by paths.DEVELOPMENT_DIRECTORY. The DataFrame index starts at 1
    and the filenames are listed without their file extensions.
    """
    # Get the directory path
    dev_dir = paths.DEVELOPMENT_DIRECTORY

    # List all files in the directory and filter out subdirectories
    all_files = [
        f for f in os.listdir(dev_dir)
        if os.path.isfile(os.path.join(dev_dir, f))
    ]

    # Strip file extensions and sort the filenames
    dev_ids = sorted(os.path.splitext(f)[0] for f in all_files)

    # Create DataFrame with index starting at 1
    df = pd.DataFrame({'dev_id': dev_ids}, index=range(1, len(dev_ids) + 1))

    return df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #infrascanrail()
    os.chdir(paths.MAIN)
    results_raw = pd.read_csv("data/costs/total_costs_raw.csv")
    railway_lines = gpd.read_file(paths.NEW_RAILWAY_LINES_PATH)
    create_and_save_plots(df=results_raw, railway_lines=railway_lines)

