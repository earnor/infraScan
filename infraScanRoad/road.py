# import packages
import os
import sys
import math
import time
import logging
import json

import pandas as pd
import geopandas as gpd
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

import tkinter as tk
from tkinter import ttk

# Get the parent directory of GUI (i.e., InfraScan)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)  # Add InfraScan to Python's module search path
from logging_config import logger  # Import central logger

import infraScanRoad.data_import as road_data_import
import infraScanRoad.voronoi_tiling as road_voronoi_tiling
import infraScanRoad.scenarios as road_scenarios
import infraScanRoad.plots as road_plots
import infraScanRoad.generate_infrastructure as road_generate_infrastructure
import infraScanRoad.scoring as road_scoring
import infraScanRoad.OSM_network as road_OSM_network
import infraScanRoad.traveltime_delay as road_traveltime_delay


# Main Class
class Road:
    # Class Variable
    # Constructor
    def __init__(self, config: dict):
        # Instance Variable
        self.config = config # Configuration JSON
        self.wd = os.path.join(self.config["General"].get("working_directory", ""), "infraScanRoad") # Working Directory
        os.chdir(self.wd) # Change working directory
        logger.road(f"Working Directory: {self.wd}")

        # runtime
        self.runtimes = {} # For saving runtime of each step        

        # Start GUI
        self.progress_bar_init("Road Modul", 100) # Initialize Progress Bar

    def progress_bar_init(self, title, total: int):
        # Initialize Progress Bar in tkinter GUI
        self.root = tk.Tk()
        self.root.title(title)

        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=10)

        self.label_config = tk.Label(self.root, text=f"Config: {self.config['General']['config_file_name']}")
        self.label_config.pack(pady=5)

        self.label_start_time = tk.Label(self.root, text=f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.label_start_time.pack(pady=5)

        self.label_estimated_completion_time = tk.Label(self.root, text="Estimated Completion Time: Calculating...")
        self.label_estimated_completion_time.pack(pady=5)

        self.label_current_process = tk.Label(self.root, text="Current Process: Initializing...")
        self.label_current_process.pack(pady=5)

        self.total_steps = total
        self.progress["maximum"] = total

        # cancle button
        self.button = tk.Button(self.root, text="Cancel", command=self.root.quit)
        self.button.pack(pady=5)

        self.root.update()
        pass

    def progress_bar_update(self, process: str, current: int = None):
        # Update Progress Bar in tkinter GUI
        if current is not None:
            self.progress["value"] = current
        self.label_current_process["text"] = f"Current Process: {process}"
        self.root.update()
        pass

    def run(self):
        # Run the process based on the configuration

        # Initialize
        self.progress_bar_update("Initialize Variables", 1)
        self.initialize_variables()
        
        # Import Data
        if self.config["Process"]["Import"]:
            self.progress_bar_update("Import Raw Data", 2)
            self.import_raw_data()

        # Network
        if self.config["Process"]["Network"]:
            self.progress_bar_update("Infrastructure Network Import", 3)
            self.infrastructure_network_import()

            self.progress_bar_update("Infrastructure Network Process", 4)
            self.infrastructure_network_process()

            self.progress_bar_update("Infrastructure Network Generate Development", 5)
            self.infrastructure_network_generate_development()

        # Scenario
        if self.config["Process"]["Scenario"]:
            self.progress_bar_update("Import Scenario Variables", 6)
            self.import_scenario_variables()

            self.progress_bar_update("Generate Scenarios", 7)
            self.generate_scenarios()

        # Scoring
        if self.config["Process"]["Scoring"]:
            self.progress_bar_update("Redefine Protected Area", 8)
            self.redefine_protected_area()

            self.progress_bar_update("Import Road Network OSM", 9)
            self.import_road_network_osm()

            self.progress_bar_update("Compute Construction Cost", 10)
            self.compute_construction_cost()

            self.progress_bar_update("Compute Cost Externalities", 11)
            self.compute_cost_externalities()

            self.progress_bar_update("Get Voronoi Tilling Travel Time", 12)
            self.get_voronoi_tilling_tt()

            self.progress_bar_update("Compute Accessibility Cost", 13)
            self.compute_accessiblity_cost()

            self.progress_bar_update("Rellocate OD Matrices", 14)
            self.rellocate_OD_mattrices()

            self.progress_bar_update("Aggregate Costs", 15)
            self.aggregate_costs()

        # Visualize Results
        if self.config["Process"]["Visualize"]:
            self.progress_bar_update("Visualize Results", 99)
            self.visualize_results()

        self.progress_bar_update("Finish", 100)

    def initialize_variables(self):
        # Initialize 
        logger.road("INITIALIZE VARIABLES \n")
        st = time.time()

        # Define variables for monetisation

        # Construction costs
        self.c_openhighway = self.config["Road"]["Construction Costs"]["c_openhighway"]["value"]
        self.c_tunnel = self.config["Road"]["Construction Costs"]["c_tunnel"]["value"]
        self.c_bridge = self.config["Road"]["Construction Costs"]["c_bridge"]["value"]
        self.ramp = self.config["Road"]["Construction Costs"]["ramp"]["value"]

        # Maintenance costs
        self.c_structural_maint = self.config["Road"]["Maintenance Costs"]["c_structural_maint"]["value"]
        self.c_om_openhighway = self.config["Road"]["Maintenance Costs"]["c_om_openhighway"]["value"]
        self.c_om_tunnel = self.config["Road"]["Maintenance Costs"]["c_om_tunnel"]["value"]
        self.c_om_bridge = self.config["Road"]["Maintenance Costs"]["c_om_bridge"]["value"]
        self.maintenance_duration = self.config["Road"]["Maintenance Costs"]["maintenance_duration"]["value"]

        # Value of travel time savings (VTTS)
        self.VTTS = self.config["Road"]["Value of Travel Time Savings"]["VTTS"]["value"]
        self.travel_time_duration = self.config["Road"]["Value of Travel Time Savings"]["travel_time_duration"]["value"]

        # Noise costs
        self.noise_distance = self.config["Road"]["Noise Costs"]["noise_distance"]["value"]
        self.noise_values = self.config["Road"]["Noise Costs"]["noise_values"]["value"]
        self.noise_duration = self.config["Road"]["Noise Costs"]["noise_duration"]["value"]

        # Climate effects
        self.co2_highway = self.config["Road"]["Climate Effects"]["co2_highway"]["value"]
        self.co2_tunnel = self.config["Road"]["Climate Effects"]["co2_tunnel"]["value"]

        # Nature and Landscape
        self.fragmentation = self.config["Road"]["Nature and Landscape"]["fragmentation"]["value"]
        self.fragmentation_duration = self.config["Road"]["Nature and Landscape"]["fragmentation_duration"]["value"]
        self.habitat_loss = self.config["Road"]["Nature and Landscape"]["habitat_loss"]["value"]
        self.habitat_loss_duration = self.config["Road"]["Nature and Landscape"]["habitat_loss_duration"]["value"]

        # Land reallocation
        self.forest_reallocation = self.config["Road"]["Land Reallocation"]["forest_reallocation"]["value"]
        self.meadow_reallocation = self.config["Road"]["Land Reallocation"]["meadow_reallocation"]["value"]
        self.reallocation_duration = self.config["Road"]["Land Reallocation"]["reallocation_duration"]["value"]

        # For infrastructure development generation
        self.num_rand = 1000
        
        # Define spatial limits of the research corridor
        # The coordinates must end with 000 in order to match the coordinates of the input raster data
        self.e_min = self.config["General"]["Spatial Limits"]["e_min"]
        self.e_max = self.config["General"]["Spatial Limits"]["e_max"]
        self.n_min = self.config["General"]["Spatial Limits"]["n_min"]
        self.n_max = self.config["General"]["Spatial Limits"]["n_max"]
        self.limits_corridor = [self.e_min, self.n_min, self.e_max, self.n_max]
        self.margin = 3000 # meters (for gloabl operation)

        # Boundry for plot
        self.boundary_plot = road_data_import.polygon_from_points(
            e_min=self.e_min+1000,
            e_max=self.e_max-500, 
            n_min=self.n_min+1000, 
            n_max=self.n_max-2000)
        # Get a polygon as limits for the corridor
        self.innerboundary = road_data_import.polygon_from_points(
            e_min=self.e_min, 
            e_max=self.e_max, 
            n_min=self.n_min, 
            n_max=self.n_max)
        # For global operation a margin is added to the boundary
        self.outerboundary = road_data_import.polygon_from_points(
            e_min=self.e_min, 
            e_max=self.e_max, 
            n_min=self.n_min, 
            n_max=self.n_max, 
            margin=self.margin)
        # Dont know what this is:
        self.limits_variables = [2680600, 1227700, 2724300, 1265600]

        # Define the size of the resolution of the raster to 100 meter
        self.raster_size = 100 # meters
        # Maybe deprecat

        self.runtimes["Initialize variables"] = time.time() - st

    def import_raw_data(self):
        # Import and prepare raw data
        logger.road("IMPORT RAW DATA \n")
        st = time.time()

        # Import shapes of lake for plots
        #get_lake_data()

        # Import the file containing the locations to be ploted
        #import_locations()


        # Define area that is protected for constructing highway links

        road_data_import.get_protected_area(limits=self.limits_corridor)
        road_data_import.get_unproductive_area(limits=self.limits_corridor)
        road_data_import.landuse(limits=self.limits_corridor)

        # Tif file of all unsuitable land cover and protected areas
        # File is stored to 'data\landuse_landcover\processed\zone_no_infra\protected_area_{suffix}.tif'

        #all_protected_area_to_raster(suffix="corridor")

        self.runtimes["Import land use and land cover data"] = time.time() - st

    def infrastructure_network_import(self):
        # Import infrastructure network
        logger.road("INFRASTRUCTURE NETWORK IMPORT")
        st = time.time()

        # Import the highway network and preprocess it
        # Data are stored as "data/temp/network_highway.gpkg"
        #load_nw()

        # Read the network dataset to avoid running the function above
        self.network = gpd.read_file(r"data/temp/network_highway.gpkg")

        # Import manually gathered access points and map them on the highway infrastructure
        # The same point but with adjusted coordinate are saved to "data\access_highway_matched.gpkg"
        self.df_access = pd.read_csv(r"data/manually_gathered_data/highway_access.csv", sep=";")
        road_data_import.map_access_points_on_network(current_points=self.df_access, network=self.network)
        self.runtimes["Import network data"] = time.time() - st

    def infrastructure_network_process(self):
        # Process infrastructure network
        logger.road("INFRASTRUCTURE NETWORK PROCESS")
        st = time.time()

        # Simplify the physical topology of the network
        # One distinct edge between two nodes (currently multiple edges between nodes)
        # Edges are stored in "data\Network\processed\edges.gpkg"
        # Points in simplified network can be intersections ("intersection"==1) or access points ("intersection"==0)
        # Points are stored in "data\Network\processed\points.gpkg"
        #reformat_network()


        # Filter the infrastructure elements that lie within a given polygon
        # Points within the corridor are stored in "data\Network\processed\points_corridor.gpkg"
        # Edges within the corridor are stored in "data\Network\processed\edges_corridor.gpkg"
        # Edges crossing the corridor border are stored in "data\Network\processed\edges_on_corridor.gpkg"
        #network_in_corridor(polygon=outerboundary)



        # Add attributes to nodes within the corridor (mainly access point T/F)
        # Points with attributes saved as "data\Network\processed\points_attribute.gpkg"
        #map_values_to_nodes()

        # Add attributes to the edges
        #get_edge_attributes()

        # Add specific elements to the network
        #required_manipulations_on_network()

        self.runtimes["Preprocess the network"] = time.time() - st

    def infrastructure_network_generate_development(self):
        # Generate infrastructure network development
        logger.road("INFRASTRUCTURE NETWORK GENERATE DEVELOPMENT")
        st = time.time()

        # Make random points within the perimeter (extent) and filter them, so they do not fall within protected or
        # unsuitable area
        # The resulting dataframe of generated nodes is stored in "data\Network\processed\generated_nodes.gpkg"
        
        random_gdf = road_generate_infrastructure.generated_access_points(extent=self.innerboundary, number=self.num_rand)

        road_generate_infrastructure.filter_access_points(random_gdf)

        #filtered_gdf.to_file(r"data/Network/processed/generated_nodes.gpkg")

        # Import the generated points as dataframe
        self.generated_points = gpd.read_file(r"data/Network/processed/generated_nodes.gpkg")

        # Import current points as dataframe and filter only access points (no intersection points)
        self.current_points = gpd.read_file(r"data/Network/processed/points_corridor_attribute.gpkg")
        self.current_access_points = self.current_points.loc[self.current_points["intersection"] == 0]

        # Connect the generated points to the existing access points
        # New lines are stored in "data/Network/processed/new_links.gpkg"
        self.filtered_rand_temp = road_generate_infrastructure.connect_points_to_network(self.generated_points, self.current_access_points)
        self.nearest_gdf = road_generate_infrastructure.create_nearest_gdf(self.filtered_rand_temp)
        road_generate_infrastructure.create_lines(self.generated_points, self.nearest_gdf)

        # Filter the generated links that connect to one of the access point within the considered corridor
        # These access points are defined in the manually defined list of access points
        # The links to corridor are stored in "data/Network/processed/developments_to_corridor_attribute.gpkg"
        # The generated points with link to access point in the corridor are stored in "data/Network/processed/generated_nodes_connecting_corridor.gpkg"
        # The end point [ID_new] of developments_to_corridor_attribute are equivlent to the points in generated_nodes_connecting_corridor
        road_data_import.only_links_to_corridor()

        # Find a routing for the generated links that considers protected areas
        # The new links are stored in "data/Network/processed/new_links_realistic.gpkg"
        # If a point is not accessible due to the banned zoned it is stored in "data/Network/processed/points_inaccessible.csv"
        raster = r'data/landuse_landcover/processed/zone_no_infra/protected_area_corridor.tif'

        road_generate_infrastructure.routing_raster(raster_path=raster)

        """
        #plot_corridor(network, limits=limits_corridor, location=location, new_nodes=filtered_rand_gdf, access_link=True)
        map_3 = CustomBasemap(boundary=outerboundary, network=network, frame=innerboundary)
        map_3.new_development(new_nodes=filtered_rand_gdf)
        map_3.show()
        """

        # Compute the voronoi polygons for the status quo and for alle developments based on euclidean distance
        # Dataframe with the voronoi polygons for the status quo is stored in "data/Voronoi/voronoi_status_quo_euclidian.gpkg"
        # Dataframe with the voronoi polygons for the all developments is stored in "data/Voronoi/voronoi_developments_euclidian.gpkg"
        road_voronoi_tiling.get_voronoi_status_quo()
        self.limits_variables = road_voronoi_tiling.get_voronoi_all_developments()
        # !!!!!!!!!!!!!!!! find why this is needed
        self.limits_variables = [2680600, 1227700, 2724300, 1265600]
        
        # Compute the area covered by the voronoi polygons of all developments. This is required to know on which area the
        # scenario must be developed
        """
        voronoi_gdf_status_quo = gpd.read_file(r"data/Voronoi/voronoi_status_quo_euclidian.gpkg")
        limits_voronoi_raw = get_voronoi_frame(voronoi_gdf_status_quo)
        limits_voronoi = [round(math.floor(limits_voronoi_raw[0]), -2), round(math.floor(limits_voronoi_raw[1]), -2),
                        round(math.ceil(limits_voronoi_raw[2]), -2), round(math.ceil(limits_voronoi_raw[3]), -2)]
        logger.road(f"LIMITS VORONOI {limits_voronoi}")
        
        voronoi_gdf = gpd.read_file(r"data/Voronoi/voronoi_developments_euclidian.shp")
        limits = voronoi_gdf.total_bounds
        limits_variables = [round(math.floor(limits[0]), -2), round(math.floor(limits[1]), -2),
                            round(math.ceil(limits[2]), -2), round(math.ceil(limits[3]), -2)]
        logger.road(f"Limits for scenarios: {limits_variables}")
        """

        self.runtimes["Generate infrastructure developments"] = time.time() - st


    def import_scenario_variables(self):
        # Import scenario variables
        logger.road("IMPORT SCENARIO VARIABLES")
        st = time.time()

        # Import the raw data, reshape it partially and store it as tif
        # Tif are stored to "data/independent_variable/processed/raw/pop20.tif"
        # File name indicates population (pop) and employment (empl), the year (20), and the extent swisswide (_ch) or only for corridor (no suffix)

        road_data_import.import_data(self.limits_variables)

        self.runtimes["Import variable for scenario (population and employment)"] = time.time() - st


    def generate_scenarios(self):
        # Define scenario
        logger.road("GENERATE SCENARIOS")
        st = time.time()

        # Import the predicted scenario defined by the canton of Zürich
        scenario_zh = pd.read_csv(r"data/Scenario/KTZH_00000705_00001741.csv", sep=";")

        # Define the relative growth per scenario and district
        # The growth rates are stored in "data/temp/data_scenario_n.shp"

        road_scenarios.future_scenario_zuerich_2022(scenario_zh)
        
        # Plot the growth rates as computed above for population and employment and over three scenarios
        #plot_2x3_subplots(scenario_polygon, outerboundary, network, location)

        # Compute the predicted amount of population and employment in each raster cell (hectar) for each scenario
        # The resulting raster data are stored in "data/independent_variables/scenario/{col}.tif" with col being pop or empl and the scenario
        road_scenarios.scenario_to_raster(self.limits_variables)

        # Aggregate the the scenario data to over the voronoi polygons, here euclidian polygons
        # Store the resulting file to "data/Voronoi/voronoi_developments_euclidian_values.shp"
        polygons_gdf = gpd.read_file(r"data/Voronoi/voronoi_developments_euclidian.gpkg")
        road_scenarios.scenario_to_voronoi(polygons_gdf, euclidean=True)

        # Convert multiple tif files to one same tif with multiple bands
        road_traveltime_delay.stack_tif_files(var="empl")
        road_traveltime_delay.stack_tif_files(var="pop")
        self.runtimes["Generate the scenarios"] = time.time() - st

    def redefine_protected_area(self):
        # Redefine protected area
        logger.road("REDEFINE PROTECTED AREA")
        st = time.time()

        # This operation has already been done above for the corridor limits, here it is applied to the voronoi polygon limits which are bigger than the corridor limits
        #get_protected_area(limits=limits_variables)
        #get_unproductive_area(limits=limits_variables)
        #landuse(limits=limits_variables)

        # Find possible links considering land cover and protected areas
        road_data_import.all_protected_area_to_raster(suffix="variables")

        self.runtimes["Redefine protected area"] = time.time() - st

    def import_road_network_osm(self):
        # Import road network from OSM
        logger.road("IMPORT ROAD NETWORK OSM")
        st = time.time()

        # Import the road network from OSM and rasterize it
        road_voronoi_tiling.nw_from_osm(self.limits_variables) #todo this requires data under data/Network/OSM_road that is not available.
        road_voronoi_tiling.osm_nw_to_raster(self.limits_variables)
        self.runtimes["Import and rasterize local road network from OSM"] = time.time() - st
        st = time.time()

        # Write runtimes to a file
        with open(r'runtimes.txt', 'w') as file:
            for part, runtime in self.runtimes.items():
                file.write(f"{part}: {runtime}\n")

        self.runtimes["Import road network from OSM"] = time.time() - st
    
    def compute_construction_cost(self):
        # Compute construction cost
        logger.road("COMPUTE CONSTRUCTION COST")
        st = time.time()

        # Compute the elevation profile for each routing to assess the amount
        # First import the elevation model downscale the resolution and store it as raster data to 'data/elevation_model/elevation.tif'
        #resolution = 50 # meter
        #import_elevation_model(new_resolution=resolution)
        self.runtimes["Import elevation model in 50 meter resolution"] = time.time() - st
        st = time.time()

        # Compute the elevation profile for each generated highway routing based on the elevation model
        links_temp = road_generate_infrastructure.get_road_elevation_profile()
        #links_temp.to_csv(r"data/Network/processed/new_links_realistic_woTunnel.csv")

        # Based on the elevation profile of each links compute the required amount of bridges and tunnels
        # Safe the dataset to "data/Network/processed/new_links_realistic_tunnel.gpkg"
        #get_tunnel_candidates(links_temp)
        road_generate_infrastructure.tunnel_bridges(links_temp)

        self.runtimes["Optimize eleavtion profile of links to find need for tunnel and bridges"] = time.time() - st
        st = time.time()

        # Compute the construction costs for each development (generated points with according link to existing access point)
        # Not including tunnels and bridges with regards to the elevation profile of a section yet
        # Result stored to "data/costs/construction.gpkg"
        logger.road(" -> Construction costs")

        road_scoring.construction_costs(highway=self.c_openhighway, tunnel=self.c_tunnel, bridge=self.c_bridge, ramp=self.ramp)
        road_scoring.maintenance_costs(duration=self.maintenance_duration, highway=self.c_om_openhighway, tunnel=self.c_om_tunnel, bridge=self.c_om_bridge, structural=self.c_structural_maint)


        self.runtimes["Compute construction and maintenance costs"] = time.time() - st

    def compute_cost_externalities(self):
        # Compute cost of externalities
        # Compute the costs arrising from externalities for each development (generated points with according link to existing access point)
        # Result stored to "data/Network/processed/new_links_externalities_costs.gpkg"

        logger.road("COMPUTE COST OF EXTERNALITIES")
        st = time.time()

        road_scoring.externalities_costs(ce_highway=self.co2_highway, ce_tunnel=self.co2_tunnel,
                            realloc_forest=self.forest_reallocation ,realloc_FFF=self.meadow_reallocation,
                            realloc_dry_meadow=self.meadow_reallocation, realloc_period=self.reallocation_duration,
                            nat_fragmentation=self.fragmentation, fragm_period=self.fragmentation_duration,
                            nat_loss_habitat=self.habitat_loss, habitat_period=self.habitat_loss_duration)


        logger.road(" -> Noise")
        road_scoring.noise_costs(years=self.noise_duration, boundaries=self.noise_distance, unit_costs=self.noise_values)

        # r"data/costs/externalities.gpkg"
        # r"data/costs/noise.gpkg"

        # Add geospatial link to the table with costs
        # Result stored to "data/costs/building_externalities.gpkg"
        #map_coordinates_to_developments()

        # Plot individual cost elements on map
        #gdf_extern_costs = gpd.read_file(r"data/Network/processed/new_links_externalities_costs.gpkg")
        #gdf_constr_costs = gpd.read_file(r"data/Network/processed/new_links_construction_costs.gpkg")
        #gdf_costs = gpd.read_file(r"data/costs/building_externalities.gpkg")
        #tif_path_plot = r"data/landuse_landcover/processed/zone_no_infra/protected_area_corridor.tif"
        #plot_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, boundary=innerboundary, network=network, access_points=current_access_points)

        self.runtimes["Compute Externalities"] = time.time() - st


    def get_voronoi_tilling_tt(self):
        # Get voronoi tilling travel time
        logger.road("GET VORONOI TILLING TRAVEL TIME")
        st = time.time()

        # Based on the rasterised road network from OSM, compute the travel time required to access the closest existing
        # highway access point from each cell in the perimeter. As result, it is also known for each cell which current
        # access points is the closest (its ID)
        # The raster file showing the travel time to the next access point is stored to 'data/Network/travel_time/travel_time_raster.tif'
        # The raster file showing the ID of the closest access point is stored in 'data/Network/travel_time/source_id_raster.tif'
        # Aggregating all cells with same closest access point is equivalent to a travel time based voronoi tiling. This is
        # stored as vector file in "data/Network/travel_time/Voronoi_statusquo.gpkg"
        road_OSM_network.travel_cost_polygon(self.limits_variables)

        self.voronoi_status_quo = gpd.read_file(r"data/Voronoi/voronoi_status_quo_euclidian.gpkg")
        self.voronoi_tt = gpd.read_file(r"data/Network/travel_time/Voronoi_statusquo.gpkg")

        # Same operation is made for all developments
        # These are store similarily than above, with id_new beeing the id of the development (ID of generated point)
        # The raster file showing the travel time to the next access point is stored to 'data/Network/travel_time/developments/dev{id_new}_travel_time_raster.tif'
        # The raster file showing the ID of the closest access point is stored in 'data/Network/travel_time/developments/dev{id_new}_source_id_raster.tif'
        # Aggregating all cells with same closest access point is equivalent to a travel time based voronoi tiling. This is
        # stored as vector file in "data/Network/travel_time/developments/dev{id_new}_Voronoi.gpkg"
        road_OSM_network.travel_cost_developments(self.limits_variables)

        self.runtimes["Voronoi tiling: Compute travel time from each raster cell to the closest access point"] = time.time() - st
        st = time.time()

        # Generate one dataframe containing the Voronoi polygons for all developments and all access points within the
        # perimeter. Before the polygons are store in an individual dataset for each development
        # The resulting dataframe is stored to "data/Voronoi/combined_developments.gpkg"
        folder_path = "data/Network/travel_time/developments"
        road_generate_infrastructure.single_tt_voronoi_ton_one(folder_path)

        # Based on the scenario and the travel time based Voronoi tiling, compute the predicted population and employment
        # in each polygon and for each scenario
        # Resulting dataset is stored to "data/Voronoi/voronoi_developments_tt_values.shp"
        polygon_gdf = gpd.read_file(r"data/Voronoi/combined_developments.gpkg")
        road_scenarios.scenario_to_voronoi(polygon_gdf, euclidean=False)

        self.runtimes["Aggregate scenarios by Voronoi polygons"] = time.time() - st


    def compute_accessiblity_cost(self):
        # Compute accessibility cost
        logger.road("COMPUTE ACCESSIBILITY COST")
        st = time.time()

        # Compute the accessibility for status quo for scenarios
        accessib_status_quo = road_scoring.accessibility_status_quo(VTT_h=self.VTTS, duration=self.travel_time_duration)

        # Compute the benefit in accessibility for each development compared to the status quo
        # The accessibility for each polygon for every development is store in "data/Voronoi/voronoi_developments_local_accessibility.gpkg"
        # The benefit of each development compared to the status quo is stored in 'data/costs/local_accessibility.csv'
        road_scoring.accessibility_developments(accessib_status_quo, VTT_h=self.VTTS, duration=self.travel_time_duration)  # make this more efficient in terms of for loops and open files

        self.runtimes["Compute highway access time benefits"] = time.time() - st

    def rellocate_OD_mattrices(self):
        # Rellocate OD matrices
        logger.road("RELLOCATE OD MATRICES")
        st = time.time()

        # Compute the OD matrix for the current infrastructure under all scenarios
        road_scoring.GetVoronoiOD()
        # od = GetVoronoiOD()

        # Compute the OD matrix for the infrastructure developments under all scenarios
        road_scoring.GetVoronoiOD_multi()

        self.runtimes["Reallocate OD matrices to Voronoi polygons"] = time.time() - st

    def aggregate_costs(self):
        # Aggregate costs
        logger.road("AGGREGATE COSTS")
        st = time.time()

        road_scoring.tt_optimization_status_quo()

        # check if flow are possible
        road_scoring.link_traffic_to_map()
        logger.road('Flag: link_traffic_to_map is complete')
        # Run travel time optimization for infrastructure developments and all scenarios
        road_scoring.tt_optimization_all_developments()
        logger.road('Flag: tt_optimization_all_developments is complete')
        # Monetize travel time savings
        road_scoring.monetize_tts(VTTS=self.VTTS, duration=self.travel_time_duration)

        ##################################################################################
        # Aggregate the single cost elements to one dataframe
        # New dataframe is stored in "data/costs/total_costs.gpkg"
        # New dataframe also stored in "data/costs/total_costs.csv"
        logger.road(" -> Aggregate costs")
        road_scoring.aggregate_costs()

        # Import to the overall cost dataframe
        self.gdf_costs = gpd.read_file(r"data/costs/total_costs.gpkg")
        # Convert all costs in million CHF
        self.gdf_costs["total_low"] = (self.gdf_costs["total_low"] / 1000000).astype(int)
        self.gdf_costs["total_medium"] = (self.gdf_costs["total_medium"] / 1000000).astype(int)
        self.gdf_costs["total_high"] = (self.gdf_costs["total_high"] / 1000000).astype(int)

        self.runtimes["Aggregate costs"] = time.time() - st

    def visualize_results(self):
        # Visualize results
        logger.road("VISUALIZE RESULTS")
        st = time.time()

        # Import layers to plot
        tif_path_plot = r"data/landuse_landcover/processed/zone_no_infra/protected_area_corridor.tif"

        links_beeline = gpd.read_file(r"data/Network/processed/new_links.gpkg")
        links_realistic = gpd.read_file(r"data/Network/processed/new_links_realistic.gpkg")
        print(links_realistic.head(5).to_string())
        # Plot the net benefits for each generated point and interpolate the area in between
        generated_points = gpd.read_file(r"data/Network/processed/generated_nodes.gpkg")
        # Get a gpd df with points have an ID_new that is not in links_realistic ID_new
        filtered_rand_gdf = generated_points[~generated_points["ID_new"].isin(links_realistic["ID_new"])]
        #plot_points_gen(points=generated_points, edges=links_beeline, banned_area=tif_path_plot, boundary=boundary_plot, network=network, all_zones=True, plot_name="gen_nodes_beeline")
        #plot_points_gen(points=generated_points, points_2=filtered_rand_gdf, edges=links_realistic, banned_area=tif_path_plot, boundary=boundary_plot, network=network, all_zones=False, plot_name="gen_links_realistic")

        voronoi_dev_2 = gpd.read_file(r"data/Network/travel_time/developments/dev779_Voronoi.gpkg")
        road_plots.plot_voronoi_development(self.voronoi_tt, voronoi_dev_2, generated_points, boundary=self.innerboundary, network=self.network, access_points=self.current_access_points, plot_name="new_voronoi")

        #plot_voronoi_comp(voronoi_status_quo, voronoi_tt, boundary=boundary_plot, network=network, access_points=current_access_points, plot_name="voronoi")


        # Plot the net benefits for each generated point and interpolate the area in between
        # if plot_name is not False, then the plot is stored in "plot/results/{plot_name}.png"
        road_plots.plot_cost_result(df_costs=self.gdf_costs, banned_area=tif_path_plot, title_bar="scenario low growth", boundary=self.boundary_plot, network=self.network,
                        access_points=self.current_access_points, plot_name="total_costs_low",col="total_low")
        road_plots.plot_cost_result(df_costs=self.gdf_costs, banned_area=tif_path_plot, title_bar="scenario medium growth", boundary=self.boundary_plot, network=self.network,
                        access_points=self.current_access_points, plot_name="total_costs_medium",col="total_medium")
        road_plots.plot_cost_result(df_costs=self.gdf_costs, banned_area=tif_path_plot, title_bar="scenario high growth", boundary=self.boundary_plot, network=self.network,
                        access_points=self.current_access_points, plot_name="total_costs_high",col="total_high")

        # Plot single cost element

        road_plots.plot_single_cost_result(df_costs=self.gdf_costs, banned_area=tif_path_plot, title_bar="construction",
                                boundary=self.boundary_plot, network=self.network, access_points=self.current_access_points,
                                plot_name="construction and maintenance", col="construction_maintenance")
        # Due to erros when plotting convert values to integer
        self.gdf_costs["local_s1"] = self.gdf_costs["local_s1"].astype(int)
        road_plots.plot_single_cost_result(df_costs=self.gdf_costs, banned_area=tif_path_plot, title_bar="access time to highway",
                                boundary=self.boundary_plot, network=self.network, access_points=self.current_access_points,
                                plot_name="access_costs",col="local_s1")
        road_plots.plot_single_cost_result(df_costs=self.gdf_costs, banned_area=tif_path_plot, title_bar="highway travel time",
                                boundary=self.boundary_plot, network=self.network, access_points=self.current_access_points,
                                plot_name="tt_costs",col="tt_medium")
        road_plots.plot_single_cost_result(df_costs=self.gdf_costs, banned_area=tif_path_plot, title_bar="noise emissions",
                                boundary=self.boundary_plot, network=self.network, access_points=self.current_access_points,
                                plot_name="externalities_costs", col="externalities_s1")

        # Plot uncertainty
        self.gdf_costs['mean_costs'] = self.gdf_costs[["total_low", "total_medium", "total_high"]].mean(axis=1)
        self.gdf_costs["std"] = self.gdf_costs[["total_low", "total_medium", "total_high"]].std(axis=1)
        self.gdf_costs['cv'] = self.gdf_costs[["total_low", "total_medium", "total_high"]].std(axis=1) / abs(self.gdf_costs['mean_costs'])
        self.gdf_costs['cv'] = self.gdf_costs['cv'] * 10000000

        road_plots.plot_cost_uncertainty(df_costs=self.gdf_costs, banned_area=tif_path_plot,
                            boundary=self.boundary_plot, network=self.network, col="std",
                            legend_title="Standard deviation\n[Mio. CHF]",
                            access_points=self.current_access_points, plot_name="uncertainty")

        road_plots.plot_cost_uncertainty(df_costs=self.gdf_costs, banned_area=tif_path_plot,
                            boundary=self.boundary_plot, network=self.network, col="cv",
                            legend_title="Coefficient of variation/n[0/0'000'000]",
                            access_points=self.current_access_points, plot_name="cv")

        # Plot the uncertainty of the nbr highest ranked developments as boxplot
        road_plots.boxplot(self.gdf_costs, 15)

        road_plots.plot_benefit_distribution_bar_single(df_costs=self.gdf_costs, column="total_medium")

        road_plots.plot_benefit_distribution_line_multi(df_costs=self.gdf_costs, columns=["total_low", "total_medium", "total_high"],
                                            labels=["low growth", "medium growth",
                                                    "high growth"], plot_name="overall", legend_title="Tested scenario")

        single_components = ["construction_maintenance", "local_s1", "tt_low", "externalities_s1"]
        for i in single_components:
            self.gdf_costs[i] = (self.gdf_costs[i] / 1000000).astype(int)
        # Plot benefit distribution for all cost elements
        road_plots.plot_benefit_distribution_line_multi(df_costs=self.gdf_costs,
                                            columns=["construction_maintenance", "local_s1", "tt_low", "externalities_s1"],
                                            labels=["construction and maintenance", "access costs", "highway travel time",
                                                    "external costs"], plot_name="single_components",
                                            legend_title="Scoring components")
        #todo plot the uncertainty
        #plot_best_worse(df=gdf_costs)


        # Plot influence of discounting
        """
        map_vor = CustomBasemap(boundary=outerboundary, network=network)
        map_vor.single_development(id=2, new_nodes=filtered_rand_gdf, new_links=new_links)
        map_vor.voronoi(id=2, gdf_voronoi=voronoi_gdf)
        map_vor.show()


        for i in voronoi_gdf["ID"].unique():
            map_vor = CustomBasemap(boundary=outerboundary, network=network)
            map_vor.single_development(id=i, new_nodes=filtered_rand_gdf, new_links=new_links)
            map_vor.voronoi(id=i, gdf_voronoi=voronoi_gdf)
            del map_vor
        

        map_development = CustomBasemap(boundary=outerboundary, network=network, access_points=current_access_points, frame=innerboundary)
        map_development.new_development(new_nodes=filtered_rand_gdf, new_links=lines_gdf)
        map_development.show()
        """
        self.runtimes["Visualize results"] = time.time() - st


def has_stdin_input():
    """Check if there's data available in sys.stdin (to detect GUI mode)."""
    return not sys.stdin.isatty()  # True if input is piped (GUI mode)

if __name__ == "__main__":
    logger.road("Starting InfraScanRoad")
    logger.road("sys.argv: %s", sys.argv)

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, BASE_DIR)  # Add InfraScan to Python's module search path

    try:
        if has_stdin_input():
            logger.road("Reading configuration from GUI (stdin)...")
            config_data = json.load(sys.stdin)
        else:
            logger.road("No valid JSON received, using default configuration.")
            cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            with open(os.path.join(cwd, "GUI", "base_config.json"), "r") as f:
                config_data = json.load(f)
    except json.JSONDecodeError:
        logger.road("Failed to parse JSON. Using default configuration.")
        cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(cwd, "GUI", "base_config.json"), "r") as f:
            config_data = json.load(f)

    scanner = Road(config_data)
    scanner.run()