# import packages
import shutil
import time

import paths
import scoring
import settings
from TT_Delay import *
from catchment_pt import *
from display_results import *
from generate_infrastructure import *
from scenarios import *
from scoring import *
from scoring import create_cost_and_benefit_df
from traveltime_delay import *
import geopandas as gpd
import networkx as nx

def infrascanrail():



    os.chdir(paths.MAIN)
    runtimes = {}

    ##################################################################################
    # Initializing global variables
    print("\nINITIALIZE VARIABLES \n")
    st = time.time()

    outerboundary = create_focus_area()

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

    ##################################################################################
    ##################################################################################
    # INFRASTRUCTURE NETWORK
    # 1) Import&Process network

    # Import the railway network and preprocess it
    # Data are stored as "data/temp/???.gpkg" ## To DO
    #load_nw()

    # 2) Generate developments (new access points) and connection to existing infrastructure

    print("\nINFRASTRUCTURE NETWORK \n")

    ##################################################################################
    # 2) Process network

    reformat_rail_nodes()
    network_ak2035, points = create_railway_services_AK2035()
    create_railway_services_AK2035_extended(network_ak2035, points)
    reformat_rail_edges()

    add_construction_info_to_network()

    network_in_corridor(poly=outerboundary)

    runtimes["Preprocess the network"] = time.time() - st
    st = time.time()

    ##################################################################################
    # 3) Generate developments (new connections) 

    generate_infra_development(use_cache=settings.use_cache_developments)

    ##here insert other network generations and save them also as a GPGK at: data/Network/processed/developments/

    runtimes["Generate infrastructure developments"] = time.time() - st
    st = time.time()

    # Compute the catchement area for the status quo and for all developments based on access time to train station
    get_catchment(use_cache=settings.use_cache_pt_catchment)

    runtimes["Generate The Catchement based on the Bus network"] = time.time() - st
    st = time.time()

    # here would code be needed to get all catchements for the different developments, if access point are added



    ##################################################################################
    ##################################################################################
    # SCENARIO
    # 1) Define scenario based on cantonal predictions

    print("\nSCENARIO \n")
    ##################################################################################
    # 1) Define scenario based on cantonal predictions

    generate_scenarios()
    runtimes["Generate the scenarios"] = time.time() - st
    st = time.time()

    ##################################################################################
    ##################################################################################
    # IMPLEMENT THE SCORING
    # 1) Compute construction and maintenancecosts
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
    od_times_dev, od_times_status_quo = create_travel_time_graphs(settings.rail_network)

    # osm_nw_to_raster(limits_variables)
    runtimes["Calculate Traveltimes for all OD_ for all developments"] = time.time() - st
    st = time.time()

    ##################################################################################
    # 2) Compute construction costs

    ##here a check for capacity could be added

    # Compute the construction costs for each development

    runtimes["Compute construction and maintenance costs"] = time.time() - st
    st = time.time()

    #################################################################################
    # Travel time delay on rail

    # Compute the OD matrix for the current infrastructure under all scenarios
    od_directory_stat_quo = r"data/traffic_flow/od/rail/stat_quo"
    od_directory_scenario = r"data/traffic_flow/od/rail"

    if settings.OD_type == 'canton_ZH':
        railway_station_OD = getStationOD(settings.use_cache_stationsOD)
    elif settings.OD_type == 'pt_catchment_perimeter':
        GetCatchmentOD(settings.use_cache_catchmentOD)
    else:
        raise ValueError("OD_type must be either 'canton_ZH' or 'pt_catchment_perimeter'")
    #combine_and_save_od_matrices(od_directory_scenario, od_directory_stat_quo)

    # Compute the OD matrix for the infrastructure developments under all scenarios
    # GetVoronoiOD_multi()

    runtimes["Reallocate OD matrices to Catchement polygons"] = time.time() - st
    st = time.time()
    # compute_TT()
    df_access = pd.read_csv(r"data/Network/Rail_Node.csv", sep=";", decimal=",", encoding="ISO-8859-1")
    TTT_status_quo = calculate_total_travel_times(od_times_status_quo, od_directory_scenario, df_access)

    # TTT for developments (trips in Peak hour * OD-Times) [in hour]
    TTT_developments = calculate_total_travel_times(od_times_dev, od_directory_scenario, df_access)
    print(TTT_status_quo)
    print(TTT_developments)
    # Monetize travel time savings ()
    output_path = "data/costs/traveltime_savings.csv"
    monetized_tt, scenario_list, dev_list = calculate_monetized_tt_savings(TTT_status_quo, TTT_developments, cp.VTTS, cp.tts_valuation_period,
                                                  output_path)
    

    file_path = "data/Network/Rail-Service_Link_construction_cost.csv"
    construction_and_maintenance_costs = construction_costs(file_path=file_path,
                                                            cost_per_meter=cp.cost_per_meter,
                                                            tunnel_cost_per_meter=cp.tunnel_cost_per_meter,
                                                            bridge_cost_per_meter=cp.bridge_cost_per_meter,
                                                            track_maintenance_cost=cp.track_maintenance_cost,
                                                            tunnel_maintenance_cost=cp.tunnel_maintenance_cost,
                                                            bridge_maintenance_cost=cp.bridge_maintenance_cost,
                                                            duration=cp.duration)
    
    # check if flow are possible
    #scenario_list = [item.replace("od_matrix_combined_", "") for item in scenario_list]

    cost_and_benefits_dev = create_cost_and_benefit_df(construction_and_maintenance_costs, dev_list, monetized_tt, scenario_list)
    costs_and_benefits_dev_discounted = discounting(cost_and_benefits_dev, discount_rate=0.03)
    discounted_costs_benefits_csv_path = "data/costs/costs_and_benefits_dev_discounted.csv"
    costs_and_benefits_dev_discounted.to_csv(discounted_costs_benefits_csv_path)



    link_traffic_to_map() #only makes a nice graph, not necessary for functioning of tool

    runtimes["Calculate the TTT Savings"] = time.time() - st
    st = time.time()

    rearange_costs(costs_and_benefits_dev_discounted)

    runtimes["Aggregate costs"] = time.time() - st

    # Write runtimes to a file
    with open(r'runtimes.txt', 'w') as file:
        for part, runtime in runtimes.items():
            file.write(f"{part}: {runtime}/n")

    ##################################################################################
    # VISUALIZE THE RESULTS

    print("\nVISUALIZE THE RESULTS \n")

    visualize_results(clear_plot_directory=True)
    plot_costs_benefits_example(
    costs_and_benefits_dev_discounted)  # only plots cost&benefits for the dev with highest tts


    # Run the display results function to launch the GUI
    # Specify the path to your CSV file
    csv_file_path = "data/costs/total_costs_with_geometry.csv"
    # Call the function to create_scenario_analysis_viewerreate and display the GUI
    create_scenario_analysis_viewer(csv_file_path)


def getStationOD(use_cache):
    if use_cache:
        railway_station_OD = pd.read_excel(paths.OD_STATIONS_KT_ZH_PATH)
    else:
        communalOD = scoring.GetOevDemandPerCommune(tau=1)
        communes_to_stations = pd.read_excel(paths.COMMUNE_TO_STATION_PATH)
        railway_station_OD = aggregate_commune_od_to_station_od(communalOD, communes_to_stations)
        railway_station_OD.to_excel(paths.OD_STATIONS_KT_ZH_PATH, index=False)

    return railway_station_OD


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


def create_travel_time_graphs(network_selection):
    if network_selection == 'current':
        network_status_quo = [paths.RAIL_SERVICES_2024_PATH]
    elif network_selection == 'AK_2035':
        network_status_quo = [paths.RAIL_SERVICES_AK2035_PATH]
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
    directory_path = r"data/Network/processed/developments"  # Define the target directory
    directories_dev = [os.path.join(directory_path, filename)
                       for filename in os.listdir(directory_path) if filename.endswith(".gpkg")]
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

    od_nodes = [
        'main_Rüti ZH', 'main_Nänikon-Greifensee', 'main_Uster', 'main_Wetzikon ZH',
        'main_Zürich Altstetten', 'main_Schwerzenbach ZH', 'main_Fehraltorf',
        'main_Bubikon', 'main_Zürich HB', 'main_Kempten', 'main_Pfäffikon ZH',
        'main_Zürich Oerlikon', 'main_Zürich Stadelhofen', 'main_Hinwil', 'main_Aathal'
    ]
    # Analyze the Delta TT
    analyze_travel_times(od_times_status_quo, od_times_dev, od_nodes) #output of this is not used!
    # Display the result
    print("\nFinal Travel Times and Delta Times:")
    return od_times_dev, od_times_status_quo


def rearange_costs(cost_and_benefits):
    ##################################################################################
    # Aggregate the single cost elements to one dataframe
    # New dataframe is stored in "data/costs/total_costs.gpkg"
    # New dataframe also stored in "data/costs/total_costs.csv"
    # Convert all costs in million CHF
    print(" -> Aggregate costs")
    aggregate_costs(cost_and_benefits)
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

    # Clear the plot directory if requested
    if clear_plot_directory:
        print(f"Clearing plot directory: {plot_dir}")
        for filename in os.listdir(plot_dir):
            file_path = os.path.join(plot_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Error while clearing {file_path}: {e}")

    # Generate all visualizations

    plotting(input_file="data/costs/total_costs_with_geometry.gpkg",
             output_file="data/costs/processed_costs.gpkg",
             node_file="data/Network/Rail_Node.xlsx")
    # make a plot of the developments
    plot_develompments_rail()
    # plot the scenarios
    plot_scenarios()
    # make a plot of the catchement with id and times
    create_plot_catchement()
    create_catchement_plot_time()
    # plot the empl and pop with the comunal boarders and the catchment
    # to visualize the OD-Transformation 
    plot_catchment_and_distributions(
        s_bahn_lines_path="data/Network/processed/split_s_bahn_lines.gpkg",
        water_bodies_path="data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp",
        catchment_raster_path="data/catchment_pt/catchement.tif",
        communal_borders_path="data/_basic_data/Gemeindegrenzen/UP_GEMEINDEN_F.shp",
        population_raster_path="data/independent_variable/processed/raw/pop20.tif",
        employment_raster_path="data/independent_variable/processed/raw/empl20.tif",
        extent_path="data/_basic_data/innerboundary.shp"
    )
    # Load the dataset and generate plots:
    # - Enhanced boxplot and strip plot for monetized savings by development.
    # Plots are saved in the 'plots' directory.
    results_raw = pd.read_csv("data/costs/total_costs_raw.csv")
    create_and_save_plots(results_raw)

    '''
    plot_developments_and_table_for_scenarios(
    osm_file="data/osm_map.gpkg",  # Use the converted GeoPackage file
    input_dir="data/costs",
    output_dir="data/plots")
    '''


def generate_infra_development(use_cache):
    if use_cache:
        print("use cache for developments")
        return
    # Identifies railway service endpoints, creates a buffer around them, and selects nearby stations within a specified radius and count (n).
    # It then generates new edges between these points and saves the resulting datasets for further use.
    # Then it calculates Traveltime, using only the existing infrastructure
    # Then it creates a new Network for each development and saves them as a GPGK
    generate_rail_edges(n=5, radius=20)
    # Filter out unnecessary links in the new_links GeoDataFrame by ensuring the connection is not redundant
    # by ensuring the connection is not redundant within the existing Sline routes
    filter_unnecessary_links()
    # Import the generated points as dataframe
    # Filter the generated links that connect to one of the access point within the considered corridor
    # These access points are defined in the manually defined list of access points
    # The links to corridor are stored in "data/Network/processed/developments_to_corridor_attribute.gpkg"
    # The generated points with link to access point in the corridor are stored in "data/Network/processed/generated_nodes_connecting_corridor.gpkg"
    # The end point [ID_new] of developments_to_corridor_attribute are equivalent to the points in generated_nodes_connecting_corridor
    only_links_to_corridor()
    calculate_new_service_time()

    new_links_updated_path = r"data\Network\processed\updated_new_links.gpkg"
    output_path = r"data\Network\processed\combined_network_with_new_links.gpkg"

    # combined_gdf = delete_connections_back(file_path_updated=r"data\Network\processed\new_links.gpkg",
    #                                        file_path_raw_edges=r"data/temp/network_railway-services.gpkg",
    #                                        output_path=r"data/Network/processed/updated_new_links_cleaned.gpkg")

    combined_gdf = update_network_with_new_links(paths.RAIL_SERVICES_AK2035_PATH, new_links_updated_path)
    update_stations(combined_gdf, output_path)


    create_network_foreach_dev()



def create_focus_area():
    # Define spatial limits of the research corridor
    # The coordinates must end with 000 in order to match the coordinates of the input raster data
    e_min, e_max = 2687000, 2708000  # 2688000, 2704000 - 2688000, 2705000
    n_min, n_max = 1237000, 1254000  # 1238000, 1252000 - 1237000, 1252000
    # For global operation a margin is added to the boundary
    margin = 3000  # meters
    outerboundary = polygon_from_points(e_min=e_min, e_max=e_max, n_min=n_min, n_max=n_max, margin=margin)
    # Define the size of the resolution of the raster to 100 meter
    # save spatial limits as shp
    save_focus_area_shapefile(e_min, e_max, n_min, n_max)
    return outerboundary


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    infrascanrail()