# Import Package
import os # For file operation
import sys # For system operation
import time # For time operation

import pandas as pd # For data manipulation
import geopandas as gpd # For geospatial data manipulation
import json # For JSON operation

import icecream as ic # For debugging
import logging # For logging and debugging

import warnings # For warning
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning) # Ignore specific pandas warning

# Import Module
import data_import
# import catchement_pt
import scenarios
import plots
import generate_infrastructure
import scoring
import traveltime_delay
# import TT_Delay
# import display_results


# Main Class
class Rail:
    # Class Variable
    # Constructor
    def __init__(self, config: dict):
        # Instance Variable
        self.config = config # Configuration JSON
        self.wd = os.path.join(self.config["General"].get("working_directory", ""), "InfraScanRail") # Working Directory
        os.chdir(self.wd) # Change working directory

        # runtime
        self.runtimes = {} # For saving runtime of each step

        # Set Logging Level Dynamically
        logging_level = self.config["General"]["Logging"].get("log_level", logging.INFO)

        if not isinstance(logging_level, int):
            raise ValueError(f'Invalid log level: {logging_level}')

        # Ensure the logger is properly configured
        logger = logging.getLogger()
        logger.setLevel(logging_level)  # Dynamically change log level

        # Remove existing handlers to avoid duplicate logs
        if logger.hasHandlers():
            logger.handlers.clear()

        # Reconfigure logging with new settings
        logging.basicConfig(level=logging_level, format='%(levelname)s: %(message)s')

        logging.info("Logging level set to: %s", logging_level)

    def run(self):
        # Run the process based on the configuration

        self.initialize_variables()
        self.import_raw_data()
        self.infrastructure_network_import()

   
    def initialize_variables (self):
        # Initialize Variables
        logging.info("INITIALIZE VARIABLES")
        st = time.time()

        # Define variables for monetisation
        # Value of travel time savings (VTTS)
        self.VTTS = self.config["Rail"]["Value of Travel Time Savings"]["VTTS"]["value"]

        # Construction costs
        self.cost_per_meter =  self.config["Rail"]["Construction Costs"]["cost_per_meter"]["value"]
        self.tunnel_cost_per_meter = self.config["Rail"]["Construction Costs"]["tunnel_cost_per_meter"]["value"]
        self.bridge_cost_per_meter = self.config["Rail"]["Construction Costs"]["bridge_cost_per_meter"]["value"]

        self.track_maintenance_cost = self.config["Rail"]["Maintenance Costs"]["track_maintenance_cost"]["value"]
        self.tunnel_maintenance_cost = self.config["Rail"]["Maintenance Costs"]["tunnel_maintenance_cost"]["value"]
        self.bridge_maintenance_cost = self.config["Rail"]["Maintenance Costs"]["bridge_maintenance_cost"]["value"]
        
        self.duration = self.config["Rail"]["Durations"]["duration"]["value"]
        self.travel_time_duration = self.config["Rail"]["Durations"]["travel_time_duration"]["value"]

        # Define spatial limits of the research corridor
        # The coordinates must end with 000 in order to match the coordinates of the input raster data
        self.e_min = self.config["General"]["Spatial Limits"]["e_min"]
        self.e_max = self.config["General"]["Spatial Limits"]["e_max"]
        self.n_min = self.config["General"]["Spatial Limits"]["n_min"]
        self.n_max = self.config["General"]["Spatial Limits"]["n_max"]
        self.limits_corridor = [self.e_min, self.n_min, self.e_max, self.n_max]
        self.margin = 3000 # meters (for gloabl operation)

        # Boundry for plot
        self.boundary_plot = data_import.polygon_from_points(
            e_min=self.e_min+1000,
            e_max=self.e_max-500, 
            n_min=self.n_min+1000, 
            n_max=self.n_max-2000)
        # Get a polygon as limits for the corridor
        self.innerboundary = data_import.polygon_from_points(
            e_min=self.e_min, 
            e_max=self.e_max, 
            n_min=self.n_min, 
            n_max=self.n_max)
        # For global operation a margin is added to the boundary
        self.outerboundary = data_import.polygon_from_points(
            e_min=self.e_min, 
            e_max=self.e_max, 
            n_min=self.n_min, 
            n_max=self.n_max, 
            margin=self.margin)
        # Dont know what this is:
        self.limits_variables = [2680600, 1227700, 2724300, 1265600]
        
        # Define the size of the resolution of the raster to 100 meter
        self.raster_size = 100 # meters
        # Maybe deprecated

        #save spatial limits as shp
        data_import.save_focus_area_shapefile(self.e_min, self.e_max, self.n_min, self.n_max)

        # Save runtime
        self.runtimes["initialize variables"] = time.time() - st
        logging.info(f"Runtime: {self.runtimes['initialize variables']} seconds")

    def import_raw_data(self):
        # Import raw data
        logging.info("IMPORT RAW DATA")
        st = time.time()

        # Define area that is protected for constructing railway links
        data_import.get_protected_area(limits=self.limits_corridor)
        data_import.get_unproductive_area(limits=self.limits_corridor)
        data_import.landuse(limits=self.limits_corridor)

        #data_import.all_protected_area_to_raster(suffix="corridor")
        # maybe deprecated

        # Save runtime
        self.runtimes["Import land use and land cover data"] = time.time() - st
        logging.info(f"Runtime: {self.runtimes['import_raw_data']} seconds")

    def infrastructure_network_import(self):
        # Import infrastructure network
        logging.info("INFRASTRUCTURE NETWORK IMPORT")
        st = time.time()

        # Import the railway network and preprocess it
        #data_import.load_nw()

        # Read the network dataset to avoid running the function above
        self.network = gpd.read_file(r"data/temp/network_railway-services.gpkg")

        # Import manually gathered access points and map them on the highway infrastructure
        # The same point but with adjusted coordinate are saved to "data\access_highway_matched.gpkg"
        #df_access = pd.read_csv(r"data/manually_gathered_data/highway_access.csv", sep=";")
        self.df_access = pd.read_csv(r"data/Network/Rail_Node.csv", sep=";",decimal=",", encoding = "ISO-8859-1")
        self.df_construction_cost = pd.read_csv(r"data/Network/Rail-Service_Link_construction_cost.csv", sep=";",decimal=",", encoding = "utf-8-sig")
        
        #data_import.map_access_points_on_network(current_points=self.df_access, network=self.network)
        #self.current_access_points = self.df_acces

        # Save runtime
        self.runtimes["Import network data"] = time.time() - st
        logging.info(f"Runtime: {self.runtimes['Import network data']} seconds")

    def infrastructure_network_process(self):
        # Process infrastructure network
        logging.info("INFRASTRUCTURE NETWORK PROCESS")
        st = time.time()

        # Simplify the physical topology of the network
        # One distinct edge between two nodes (currently multiple edges between nodes)
        # Edges are stored in "data\Network\processed\edges.gpkg"
        # Points in simplified network can be intersections ("intersection"==1) or access points ("intersection"==0)
        # Points are stored in "data\Network\processed\points.gpkg"
        data_import.reformat_rail_network()


        # Filter the infrastructure elements that lie within a given polygon
        # Points within the corridor are stored in "data\Network\processed\points_corridor.gpkg"
        # Edges within the corridor are stored in "data\Network\processed\edges_corridor.gpkg"
        # Edges crossing the corridor border are stored in "data\Network\processed\edges_on_corridor.gpkg"
        data_import.network_in_corridor(poly=self.outerboundary)



        # Add attributes to nodes within the corridor (mainly access point T/F)
        # Points with attributes saved as "data\Network\processed\points_attribute.gpkg"
        #dataimport.map_values_to_nodes()

        # Add attributes to the edges
        data_import.get_edge_attributes()

        # Add specific elements to the network
        #data_import.required_manipulations_on_network()

        # Save runtime
        self.runtimes["Preprocess the network"] = time.time() - st
        logging.info(f"Runtime: {self.runtimes['Preprocess the network']} seconds")

    def infrastructure_network_generate_development(self):
        # Generate infrastructure network development
        logging.info("INFRASTRUCTURE NETWORK GENERATE DEVELOPMENT")
        st = time.time()

        #Identifies railway service endpoints, creates a buffer around them, and selects nearby stations within a specified radius and count (n). 
        #It then generates new edges between these points and saves the resulting datasets for further use.
        #Then it calculates Traveltime, using only the existing infrastructure
        #Then it creates a new Network for each development and saves them as a GPGK

        generate_infrastructure.generate_rail_edges(n=5,radius=20)

    
        #Filter out unnecessary links in the new_links GeoDataFrame by ensuring the connection is not redundant
        #by ensuring the connection is not redundant within the existing Sline routes
        generate_infrastructure.filter_unnecessary_links()


        #filtered_gdf.to_file(r"data/Network/processed/generated_nodes.gpkg")


        # Import the generated points as dataframe

        # Filter the generated links that connect to one of the access point within the considered corridor
        # These access points are defined in the manually defined list of access points
        # The links to corridor are stored in "data/Network/processed/developments_to_corridor_attribute.gpkg"
        # The generated points with link to access point in the corridor are stored in "data/Network/processed/generated_nodes_connecting_corridor.gpkg"
        # The end point [ID_new] of developments_to_corridor_attribute are equivlent to the points in generated_nodes_connecting_corridor
        data_import.only_links_to_corridor(poly=self.outerboundary)

        generate_infrastructure.calculate_new_service_time()

        network_railway_service_path = r"data\temp\network_railway-services.gpkg"
        new_links_updated_path = r"data\Network\processed\updated_new_links.gpkg"
        output_path = r"data\Network\processed\combined_network_with_new_links.gpkg"
        

        #combined_gdf = delete_connections_back(file_path_updated=r"data\Network\processed\new_links.gpkg",
        #                                        file_path_raw_edges=r"data/temp/network_railway-services.gpkg",
        #                                        output_path=r"data/Network/processed/updated_new_links_cleaned.gpkg")


        combined_gdf = generate_infrastructure.update_network_with_new_links(network_railway_service_path, new_links_updated_path)
        combined_gdf = generate_infrastructure.update_stations(combined_gdf, output_path)

        
        generate_infrastructure.create_network_foreach_dev()

        ##here insert other network generations and save them also as a GPGK at: data/Network/processed/developments/


        # Save runtime
        self.runtimes["Generate infrastructure developments"] = time.time() - st
        logging.info(f"Runtime: {self.runtimes['Generate infrastructure developments']} seconds")

        def infrastructure_network_catchment(self):
            # Generate catchment area
            logging.info("INFRASTRUCTURE NETWORK CATCHMENT")
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
        
        catchement_pt.get_catchement(self.limits_corridor, self.outerboundary)

        # Save runtime
        self.runtimes["Generate The Catchement based on the Bus network"] = time.time() - st
        logging.info(f"Runtime: {self.runtimes['Generate The Catchement based on the Bus network']} seconds")

    def scenarios_cantonal_predictions(self):
        # Define scenario based on cantonal predictions
        logging.info("SCORING CANTONAL PREDICTIONS")
        st = time.time()

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
        scenarios.future_scenario_pop(n=3)
        scenarios.future_scenario_empl(n=3)


        # Compute the predicted amount of population and employment in each raster cell (hectar) for each scenario
        # The resulting raster data are stored in "data/independent_variables/scenario/{col}.tif" with col being pop or empl and the scenario
        scenarios.scenario_to_raster_pop(self.limits_variables)
        scenarios.scenario_to_raster_emp(self.limits_variables)
        

        # Aggregate the the scenario data to over the voronoi polygons, here euclidian polygons
        # Store the resulting file to "data/Voronoi/voronoi_developments_euclidian_values.shp"
        self.polygons_gdf = gpd.read_file(r"data/Voronoi/voronoi_developments_euclidian.gpkg")
        #scenario_to_voronoi(polygons_gdf, euclidean=True)

        # Convert multiple tif files to one same tif with multiple bands
        traveltime_delay.stack_tif_files(var="empl")
        traveltime_delay.stack_tif_files(var="pop")

        # Save runtime
        self.runtimes["Generate the scenarios"] = time.time() - st
        logging.info(f"Runtime: {self.runtimes['Generate the scenarios']} seconds")

    def scoring_traveltime_savings(self):
        # Scoring travel time savings
        logging.info("SCORING TRAVEL TIME SAVINGS")
        st = time.time()

        # 1) Calculate Traveltimes for all OD_ for all developments
        # Constructs a directed graph from the railway network GeoPackage, 
        # adding nodes (stations) and edges (connections) with travel and service data.
        # Computes an OD matrix using Dijkstra's algorithm, 
        # calculates travel times with penalties for line changes, and stores full path geometries.
        # Returns the graph (nx.DiGraph) and a DataFrame with OD travel data including adjusted travel times and geometries.

        #network of status quo

        network_status_quo = [r"data/temp/network_railway-services.gpkg"]
        G_status_quo = TT_Delay.create_graphs_from_directories(network_status_quo)
        od_times_status_quo = TT_Delay.calculate_od_pairs_with_times_by_graph(G_status_quo)

        #Example usage Test1
        origin_station = "Uster"
        destination_station = "Zürich HB"
        TT_Delay.find_fastest_path(G_status_quo[0], origin_station, destination_station)
        #Example usage Test2
        origin_station = "Uster"
        destination_station = "Pfäffikon ZH"
        TT_Delay.find_fastest_path(G_status_quo[0], origin_station, destination_station)

        #networks with all developments

        # get the paths of all developments
        directory_path = r"data/Network/processed/developments" # Define the target directory
        directories_dev = [os.path.join(directory_path, filename) 
                for filename in os.listdir(directory_path) if filename.endswith(".gpkg")]
        directories_dev = [path.replace("\\", "/") for path in directories_dev]

        G_developments = TT_Delay.create_graphs_from_directories(directories_dev)
        od_times_dev = TT_Delay.calculate_od_pairs_with_times_by_graph(G_developments)

        #Example usage Test1 for development 1007 (New Link Uster-Pfäffikon)
        origin_station = "Uster"
        destination_station = "Zürich HB"
        TT_Delay.find_fastest_path(G_developments[5], origin_station, destination_station)

        #Example usage Test2
        origin_station = "Uster"
        destination_station = "Pfäffikon ZH"
        TT_Delay.find_fastest_path(G_developments[5], origin_station, destination_station)

        
        #Example usage Development 8 (Wetikon to Hinwil (S3))
        origin_station = "Kempten"
        destination_station = "Hinwil"
        TT_Delay.find_fastest_path(G_status_quo[0], origin_station, destination_station)
        #Example usage Test2
        origin_station = "Kempten"
        destination_station = "Hinwil"
        TT_Delay.find_fastest_path(G_developments[7], origin_station, destination_station)

        selected_indices = [0,1,2,3,4, 5,6, 7]  # Indices of selected developments
        od_nodes = [
            'main_Rüti ZH', 'main_Nänikon-Greifensee', 'main_Uster', 'main_Wetzikon ZH',
            'main_Zürich Altstetten', 'main_Schwerzenbach ZH', 'main_Fehraltorf', 
            'main_Bubikon', 'main_Zürich HB', 'main_Kempten', 'main_Pfäffikon ZH', 
            'main_Zürich Oerlikon', 'main_Zürich Stadelhofen', 'main_Hinwil', 'main_Aathal'
        ]

        # Analyze the Delta TT
        TT_Delay.analyze_travel_times(od_times_status_quo, od_times_dev, selected_indices, od_nodes)
        final_result = TT_Delay.analyze_travel_times(od_times_status_quo, od_times_dev, selected_indices, od_nodes)

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

        # Save runtime
        self.runtimes["Scoring travel time savings"] = time.time() - st
        logging.info(f"Runtime: {self.runtimes['Scoring travel time savings']} seconds")

    def scoring_construction_cost(self):
        logging.info("SCORING CONSTRUCTION COST")
        st = time.time()

        ##here a check for capacity could be added

        file_path = "data/Network/Rail-Service_Link_construction_cost.csv"
        developments = TT_Delay.read_development_files('data/Network/processed/developments')


        construction_and_maintenance_costs = TT_Delay.construction_costs(file_path,
                                                                developments,
                                                                self.cost_per_meter,
                                                                self.tunnel_cost_per_meter,
                                                                self.bridge_cost_per_meter,
                                                                self.track_maintenance_cost,
                                                                self.tunnel_maintenance_cost,
                                                                self.bridge_maintenance_cost,
                                                                self.duration)
        
        # Save runtime
        self.runtimes["Scoring construction cost"] = time.time() - st
        logging.info(f"Runtime: {self.runtimes['Scoring construction cost']} seconds")


if __name__ == "__main__":
    logging.info("Starting InfraScanRail")
    logging.debug("sys.argv: %s", len(sys.argv))
    if len(sys.argv) == 1: # No arguments passed which means no JSON input
        logging.warning("No JSON input, Running with default configuration")
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(cwd, "GUI", "base_config.json"), "r") as f:
            config_data = json.load(f)
    else:
        try: # Read JSON from stdin
            config_data = json.load(sys.stdin)  # Read JSON from stdin
        except json.JSONDecodeError:
            print("Failed to decode JSON")
            sys.exit(1)

    logging.debug("config_data: %s", config_data)

    scanner = Rail(config_data)
    scanner.run()