# import packages
import os
import math
import time

import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

from data_import import *
from catchement_pt import *
from scenarios import *
from plots import *
from generate_infrastructure import *
from scoring import *
from traveltime_delay import *
from TT_Delay import *
from display_results import *



def print_hi(name):
    # Welcome
    # Use a breakpoint in the code line below to debug your script.
    #os.chdir(r"C:\Users\Fabrice\Desktop\HS23\Thesis\Code")
    #os.chdir(r"G:\IM\09 - Teaching\11 - Masters - Projects\2023 - FS\Marggi\04_Submission\Submission\FS2023 - MScProject - Marggi\06 - Developments\01 - Code\01 - Python")
    #os.chdir(r"C:/Users/spadmin/PycharmProjects/infraScan/infraScanRail")
    #os.chdir(r"C:\Users\phili\polybox\ETH_RE&IS\Master Thesis\06-Developments\01-Code\infraScanRail")
    #os.chdir(r"/local/home/earnor/infraScan/")
    #os.chdir(r"/home/earnor/infraScan/")
    os.chdir(r"D:/OneDrive/ETH/FS25/01_Master_Project/20_Code/infraScan/infraScanRail")

    runtimes = {}


    ##################################################################################
    # Initializing global variables
    print("\nINITIALIZE VARIABLES \n")
    st = time.time()

    # Define spatial limits of the research corridor
    # The coordinates must end with 000 in order to match the coordinates of the input raster data
    e_min, e_max = 2687000, 2708000     # 2688000, 2704000 - 2688000, 2705000
    n_min, n_max = 1237000, 1254000     # 1238000, 1252000 - 1237000, 1252000
    limits_corridor = [e_min, n_min, e_max, n_max]

    # Boudary for plot
    boundary_plot = polygon_from_points(e_min=e_min+1000, e_max=e_max-500, n_min=n_min+1000, n_max=n_max-2000)

    # Get a polygon as limits for teh corridor
    innerboundary = polygon_from_points(e_min=e_min, e_max=e_max, n_min=n_min, n_max=n_max)

    # For global operation a margin is added to the boundary
    margin = 3000 # meters
    outerboundary = polygon_from_points(e_min=e_min, e_max=e_max, n_min=n_min, n_max=n_max, margin=margin)

    # Define the size of the resolution of the raster to 100 meter
    raster_size = 100 # meters

    #save spatial limits as shp
    save_focus_area_shapefile(e_min, e_max, n_min, n_max)

    ##################################################################################
    # Define variables for monetisation

    # Value of travel time savings (VTTS)
    VTTS = 14.8 # CHF/h 
    

    # Construction costs
    cost_per_meter = 33250  # CHF per meter
    tunnel_cost_per_meter = 104000  # CHF per meter per track (From Schweizer())
    bridge_cost_per_meter = 70000  # CHF per meter per track
    

    track_maintenance_cost = 132 # CHF per meter per track per year
    tunnel_maintenance_cost = 132 # CHF/m/a
    bridge_maintenance_cost = 368.8 # CHF/m/a
    duration = 50  # 50 years
    travel_time_duration = 50
    


    runtimes["Initialize variables"] = time.time() - st
    st = time.time()

    ##################################################################################
    # Import and prepare raw data
    print("\nIMPORT RAW DATA \n")

    # Import the required data for the analysis
    #import_data(limits=limits_corridor)

    # Import shapes of lake for plots
    #get_lake_data()

    # Import the file containing the locations to be ploted
    #import_locations()


    # Define area that is protected for constructing railway links
    get_protected_area(limits=limits_corridor)
    get_unproductive_area(limits=limits_corridor)
    landuse(limits=limits_corridor)

    # Tif file of all unsuitable land cover and protected areas
    # File is stored to 'data\landuse_landcover\processed\zone_no_infra\protected_area_{suffix}.tif'

    #all_protected_area_to_raster(suffix="corridor")

    runtimes["Import land use and land cover data"] = time.time() - st
    st = time.time()

    ##################################################################################
    ##################################################################################
    # INFRASTRUCTURE NETWORK
    # 1) Import network
    # 2) Process network
    # 3) Generate developments (new access points) and connection to existing infrastructure

    print("\nINFRASTRUCTURE NETWORK \n")
    ##################################################################################
    # 1) Import network
    # Import the railway network and preprocess it
    # Data are stored as "data/temp/???.gpkg" ## To DO
    #load_nw()

    # Read the network dataset to avoid running the function above
    network = gpd.read_file(r"data/temp/network_railway-services.gpkg")
    
    
    # Import manually gathered access points and map them on the highway infrastructure
    # The same point but with adjusted coordinate are saved to "data\access_highway_matched.gpkg"
    #df_access = pd.read_csv(r"data/manually_gathered_data/highway_access.csv", sep=";")
    df_access = pd.read_csv(r"data/Network/Rail_Node.csv", sep=";",decimal=",", encoding = "ISO-8859-1")
    df_construction_cost = pd.read_csv(r"data/Network/Rail-Service_Link_construction_cost.csv", sep=";",decimal=",", encoding = "utf-8-sig")
    
    '''
    map_access_points_on_network(current_points=df_access, network=network)
    current_access_points = df_acces
    '''

    runtimes["Import network data"] = time.time() - st
    st = time.time()
    """
    # Plot the highway network with the access points (adjusted coordinates)
    current_points = gpd.read_file(r"data\access_highway_matched.shp")
    map_2 = CustomBasemap(boundary=polygon_from_points(current_points.total_bounds), network=network, access_points=current_points, frame=innerboundary)
    map_2.show()
    """

    ##################################################################################
    # 2) Process network

    # Simplify the physical topology of the network
    # One distinct edge between two nodes (currently multiple edges between nodes)
    # Edges are stored in "data\Network\processed\edges.gpkg"
    # Points in simplified network can be intersections ("intersection"==1) or access points ("intersection"==0)
    # Points are stored in "data\Network\processed\points.gpkg"
    #reformat_highway_network()
    reformat_rail_network()


    # Filter the infrastructure elements that lie within a given polygon
    # Points within the corridor are stored in "data\Network\processed\points_corridor.gpkg"
    # Edges within the corridor are stored in "data\Network\processed\edges_corridor.gpkg"
    # Edges crossing the corridor border are stored in "data\Network\processed\edges_on_corridor.gpkg"
    network_in_corridor(poly=outerboundary)



    # Add attributes to nodes within the corridor (mainly access point T/F)
    # Points with attributes saved as "data\Network\processed\points_attribute.gpkg"
    #map_values_to_nodes()

    # Add attributes to the edges
    get_edge_attributes()

    # Add specific elements to the network
    #required_manipulations_on_network()

    runtimes["Preprocess the network"] = time.time() - st
    st = time.time()


    ##################################################################################
    # 3) Generate developments (new connections) 

    #Identifies railway service endpoints, creates a buffer around them, and selects nearby stations within a specified radius and count (n). 
    #It then generates new edges between these points and saves the resulting datasets for further use.
    #Then it calculates Traveltime, using only the existing infrastructure
    #Then it creates a new Network for each development and saves them as a GPGK

    generate_rail_edges(n=5,radius=20)

   
    #Filter out unnecessary links in the new_links GeoDataFrame by ensuring the connection is not redundant
    #by ensuring the connection is not redundant within the existing Sline routes
    filter_unnecessary_links()


    #filtered_gdf.to_file(r"data/Network/processed/generated_nodes.gpkg")


    # Import the generated points as dataframe

    # Filter the generated links that connect to one of the access point within the considered corridor
    # These access points are defined in the manually defined list of access points
    # The links to corridor are stored in "data/Network/processed/developments_to_corridor_attribute.gpkg"
    # The generated points with link to access point in the corridor are stored in "data/Network/processed/generated_nodes_connecting_corridor.gpkg"
    # The end point [ID_new] of developments_to_corridor_attribute are equivlent to the points in generated_nodes_connecting_corridor
    only_links_to_corridor(poly=outerboundary)

    calculate_new_service_time()

    network_railway_service_path = r"data\temp\network_railway-services.gpkg"
    new_links_updated_path = r"data\Network\processed\updated_new_links.gpkg"
    output_path = r"data\Network\processed\combined_network_with_new_links.gpkg"
    

    #combined_gdf = delete_connections_back(file_path_updated=r"data\Network\processed\new_links.gpkg",
    #                                        file_path_raw_edges=r"data/temp/network_railway-services.gpkg",
    #                                        output_path=r"data/Network/processed/updated_new_links_cleaned.gpkg")


    combined_gdf = update_network_with_new_links(network_railway_service_path, new_links_updated_path)
    combined_gdf = update_stations(combined_gdf, output_path)

    
    create_network_foreach_dev()

    ##here insert other network generations and save them also as a GPGK at: data/Network/processed/developments/

    runtimes["Generate infrastructure developments"] = time.time() - st
    st = time.time()


    
    ## PRO did comment the lines below raster and routing_raster() out
    '''
    # Find a routing for the generated links that considers protected areas
    # The new links are stored in "data/Network/processed/new_links_realistic.gpkg"
    # If a point is not accessible due to the banned zoned it is stored in "data/Network/processed/points_inaccessible.csv"
    raster = r'data/landuse_landcover/processed/zone_no_infra/protected_area_corridor.tif'
    routing_raster(raster_path=raster)
    '''


    """
    #plot_corridor(network, limits=limits_corridor, location=location, new_nodes=filtered_rand_gdf, access_link=True)
    map_3 = CustomBasemap(boundary=outerboundary, network=network, frame=innerboundary)
    map_3.new_development(new_nodes=filtered_rand_gdf)
    map_3.show()
    """

    # Compute the catchement polygons for the status quo and for all developments based on access time to train station
    # Dataframe with the voronoi polygons for the status quo is stored in "data/Voronoi/voronoi_status_quo_euclidian.gpkg"
    # Dataframe with the voronoi polygons for the all developments is stored in "data/Voronoi/voronoi_developments_euclidian.gpkg"
    
    get_catchement(limits_corridor, outerboundary)

    runtimes["Generate The Catchement based on the Bus network"] = time.time() - st
    st = time.time()
    

    # here would code be needed to get all catchements for the different developments, if access point are added
    
    limits_variables = [2680600, 1227700, 2724300, 1265600]
    
    ##################################################################################
    ##################################################################################
    # SCENARIO
    # 1) Define scenario based on cantonal predictions

    print("\nSCENARIO \n")
    ##################################################################################
    # 1) Define scenario based on cantonal predictions

    # Import the predicted scenario defined by the canton of Zürich
    scenario_zh = pd.read_csv(r"data/Scenario/KTZH_00000705_00001741.csv", sep=";")

    
    # Define the relative growth per scenario and district
    # The growth rates are stored in "data/temp/data_scenario_n.shp"
    #future_scenario_zuerich_2022(scenario_zh)
    # Plot the growth rates as computed above for population and employment and over three scenarios
    #plot_2x3_subplots(scenario_polygon, outerboundary, network, location)


    # Calculates population growth allocation across nx3 scenarios for municipalities within a defined corridor.
    # For each scenario, adjusts total growth and distributes it among municipalities with urban, equal, and rural biases.
    # Merges growth results with spatial boundaries to form a GeoDataFrame of growth projections for mapping.
    # Saves the resulting GeoDataFrame to a shapefile.
    # !!!!!!!!!!!!!!!!!!! data missing:
    #future_scenario_pop(n=3)
    #future_scenario_empl(n=3)


    # Compute the predicted amount of population and employment in each raster cell (hectar) for each scenario
    # The resulting raster data are stored in "data/independent_variables/scenario/{col}.tif" with col being pop or empl and the scenario
    scenario_to_raster_pop(limits_variables)
    scenario_to_raster_emp(limits_variables)
    

    # Aggregate the the scenario data to over the voronoi polygons, here euclidian polygons
    # Store the resulting file to "data/Voronoi/voronoi_developments_euclidian_values.shp"
    polygons_gdf = gpd.read_file(r"data/Voronoi/voronoi_developments_euclidian.gpkg")
    #scenario_to_voronoi(polygons_gdf, euclidean=True)

    # Convert multiple tif files to one same tif with multiple bands
    stack_tif_files(var="empl")
    stack_tif_files(var="pop")
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

    #network of status quo

    network_status_quo = [r"data/temp/network_railway-services.gpkg"]
    G_status_quo = create_graphs_from_directories(network_status_quo)
    od_times_status_quo = calculate_od_pairs_with_times_by_graph(G_status_quo)

    #Example usage Test1
    origin_station = "Uster"
    destination_station = "Zürich HB"
    find_fastest_path(G_status_quo[0], origin_station, destination_station)
    #Example usage Test2
    origin_station = "Uster"
    destination_station = "Pfäffikon ZH"
    find_fastest_path(G_status_quo[0], origin_station, destination_station)

    #networks with all developments

    # get the paths of all developments
    directory_path = r"data/Network/processed/developments" # Define the target directory
    directories_dev = [os.path.join(directory_path, filename) 
             for filename in os.listdir(directory_path) if filename.endswith(".gpkg")]
    directories_dev = [path.replace("\\", "/") for path in directories_dev]

    G_developments = create_graphs_from_directories(directories_dev)
    od_times_dev = calculate_od_pairs_with_times_by_graph(G_developments)

    #Example usage Test1 for development 1007 (New Link Uster-Pfäffikon)
    origin_station = "Uster"
    destination_station = "Zürich HB"
    find_fastest_path(G_developments[5], origin_station, destination_station)

    #Example usage Test2
    origin_station = "Uster"
    destination_station = "Pfäffikon ZH"
    find_fastest_path(G_developments[5], origin_station, destination_station)

    
    #Example usage Development 8 (Wetikon to Hinwil (S3))
    origin_station = "Kempten"
    destination_station = "Hinwil"
    find_fastest_path(G_status_quo[0], origin_station, destination_station)
    #Example usage Test2
    origin_station = "Kempten"
    destination_station = "Hinwil"
    find_fastest_path(G_developments[7], origin_station, destination_station)

    selected_indices = [0,1,2,3,4, 5,6, 7]  # Indices of selected developments
    od_nodes = [
        'main_Rüti ZH', 'main_Nänikon-Greifensee', 'main_Uster', 'main_Wetzikon ZH',
        'main_Zürich Altstetten', 'main_Schwerzenbach ZH', 'main_Fehraltorf', 
        'main_Bubikon', 'main_Zürich HB', 'main_Kempten', 'main_Pfäffikon ZH', 
        'main_Zürich Oerlikon', 'main_Zürich Stadelhofen', 'main_Hinwil', 'main_Aathal'
    ]

    # Analyze the Delta TT
    analyze_travel_times(od_times_status_quo, od_times_dev, selected_indices, od_nodes)
    final_result = analyze_travel_times(od_times_status_quo, od_times_dev, selected_indices, od_nodes)

    # Display the result
    print("\nFinal Travel Times and Delta Times:")
    print(final_result)


    '''
    G_status_quo = create_directed_graph(network_status_quo)
    network_status_quo = [r"data/temp/network_railway-services.gpkg"]
    G_status_quo = define_rail_network(network_status_quo)
    plot_rail_network(G_status_quo)
    od_matrix_stat_quo = calculate_od_matrices_with_penalties(G_status_quo)
    '''

    '''
    G_developments = define_rail_network(directories_dev)
    #plot_rail_network(G_developments)
    od_matrix_dev = calculate_od_matrices_with_penalties(G_developments)
    '''
    # calculate traveltimes with google maps to compare and check accurancy
    '''
    # Assuming get_google_travel_time is a defined function that retrieves travel time from Google API.
    # Ensure your API key is correctly set up.
    api_key = 'AIzaSyCFByVXpNNrVY_HATr7NaJk2m3Tuix1u2Y'  # Replace with your actual API key

    # Calculate the travel times and update od_matrix
    od_matrix = calculate_travel_times(od_matrix, api_key)
    '''

   
    #osm_nw_to_raster(limits_variables)
    runtimes["Calculate Traveltimes for all OD_ for all developments"] = time.time() - st
    st = time.time()

    # Write runtimes to a file
    with open(r'runtimes.txt', 'w') as file:
        for part, runtime in runtimes.items():
            file.write(f"{part}: {runtime}\n")
        

    ##################################################################################
    # 2) Compute construction costs

    ##here a check for capacity could be added

    # Compute the construction costs for each development 
    print(" -> Construction costs")

    file_path = "data/Network/Rail-Service_Link_construction_cost.csv"
    developments = read_development_files('data/Network/processed/developments')


    construction_and_maintenance_costs = construction_costs(file_path,
                                                            developments,
                                                            cost_per_meter,
                                                            tunnel_cost_per_meter,
                                                            bridge_cost_per_meter,
                                                            track_maintenance_cost,
                                                            tunnel_maintenance_cost,
                                                            bridge_maintenance_cost,
                                                            duration)
    

    runtimes["Compute construction and maintenance costs"] = time.time() - st
    st = time.time()


    #################################################################################
    # Travel time delay on rail

    # Compute the OD matrix for the current infrastructure under all scenarios
    directory = "data/traffic_flow/od/rail/"
    status_quo_directory = "data/traffic_flow/od/rail/stat_quo"
    od_directory_stat_quo = r"data/traffic_flow/od/rail/stat_quo"
    od_directory_dev = r"data/traffic_flow/od/rail"

    GetCatchmentOD()
    combine_and_save_od_matrices(od_directory_dev, od_directory_stat_quo)
    #compute_TT()

    # Compute the OD matrix for the infrastructure developments under all scenarios
    #GetVoronoiOD_multi()

    runtimes["Reallocate OD matrices to Catchement polygons"] = time.time() - st
    st = time.time()


    #TTT for status quo (trips in Peak hour * OD-Times) [in hour]
    TTT_status_quo = calculate_total_travel_times(od_times_status_quo, od_directory_stat_quo, df_access)

    #TTT for developments (trips in Peak hour * OD-Times) [in hour]
    TTT_developments = calculate_total_travel_times(od_times_dev, od_directory_dev, df_access)

    # Monetize travel time savings ()
    output_path = "data/costs/traveltime_savings.csv"
    monetized_tt = calculate_monetized_tt_savings(TTT_status_quo, TTT_developments, VTTS, travel_time_duration, output_path)

    '''
    #tt_optimization_status_quo()
    # check if flow are possible
    link_traffic_to_map()
    print('Flag: link_traffic_to_map is complete')
    # Run travel time optimization for infrastructure developments and all scenarios
    tt_optimization_all_developments()
    print('Flag: tt_optimization_all_developments is complete')
    # Monetize travel time savings
    monetize_tts(VTTS=VTTS, duration=travel_time_duration)
    '''
    runtimes["Calculate the TTT Savings"] = time.time() - st
    st = time.time()

    ##################################################################################
    # Aggregate the single cost elements to one dataframe
    # New dataframe is stored in "data/costs/total_costs.gpkg"
    # New dataframe also stored in "data/costs/total_costs.csv"
    # Convert all costs in million CHF
    print(" -> Aggregate costs")

    aggregate_costs()
    transform_and_reshape_dataframe()


    runtimes["Aggregate costs"] = time.time() - st

    # Write runtimes to a file
    with open(r'runtimes_2.txt', 'w') as file:
        for part, runtime in runtimes.items():
            file.write(f"{part}: {runtime}/n")
    
    #####code until here runs fine

    ##################################################################################
    ##################################################################################
    # VISUALIZE THE RESULTS

    print("\nVISUALIZE THE RESULTS \n")


    plotting(input_file="data/costs/total_costs_with_geometry.gpkg",
             output_file="data/costs/processed_costs.gpkg",
             node_file="data/Network/Rail_Node.xlsx")


    #make a plot of the developments
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

    # Run the display results function to launch the GUI
    # Specify the path to your CSV file
    csv_file_path = "data/costs/total_costs_with_geometry.csv"
    # Call the function to create and display the GUI
    create_scenario_analysis_viewer(csv_file_path)

    

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('You did a good job ;)')
