# import packages
import os
import math
import time
import pandas as pd
import geopandas as gpd
import sys
import ast # Added for the elevation profile fix

from data_import import *
from voronoi_tiling import *
from scenarios import *
from plots import *
from generate_infrastructure import *
from scoring import *
from OSM_network import *
from traveltime_delay import *
from data_converter import *



def print_hi(name):
    os.chdir(r'/Users/ruki/PycharmProjects/infraScan/infraScanCycle')  # TODO: implement the same code for data_converter
    sys.setrecursionlimit(2000)
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

    # Boundary for plot
    boundary_plot = polygon_from_points(e_min=e_min+1000, e_max=e_max-500, n_min=n_min+1000, n_max=n_max-2000)

    # Get a polygon as limits for the corridor
    innerboundary = polygon_from_points(e_min=e_min, e_max=e_max, n_min=n_min, n_max=n_max)

    # For global operation a margin is added to the boundary
    margin = 3000 # meters TODO: change margin
    outerboundary = polygon_from_points(e_min=e_min, e_max=e_max, n_min=n_min, n_max=n_max, margin=margin)

    # Define the size of the resolution of the raster to 25-50 meter, before 100 was too coarse
    raster_size = 50 # meters

    ##################################################################################
    # Define variables for monetisation

    # ---------------------------------------------------------------------------------
    # PLACEHOLDER VALUES — all costs below must be updated with cycling-specific values
    # Sources to consult: ARE Wegleitung Kosten-Nutzen-Analyse, VSS norms for cycling
    # ---------------------------------------------------------------------------------

    # Construction costs [CHF/m] — highway values, replace with cycle path costs
    # Typical cycle path: ~500–2000 CHF/m depending on surface and segregation
    c_openhighway = None  # TODO: set cycle path construction cost [CHF/m]
    c_tunnel = None  # TODO: set or remove (tunnels rare for cycling)
    c_bridge = None  # TODO: set cycle bridge cost [CHF/m] (much cheaper than highway)
    ramp = None  # TODO: remove ramp cost (not applicable for cycling)

    # Value of Travel Time Savings [CHF/h]
    # Highway: 32.2 CHF/h — cycling VTTS is lower (~10–15 CHF/h for leisure, higher for commute)
    VTTS = None  # TODO: set cycling VTTS [CHF/h]
    travel_time_duration = 50  # years — can stay the same

    # Noise costs — cycling produces no relevant noise, remove from scoring
    # noise_distance and noise_values left as comments, not used
    # noise_distance = [0, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]
    # noise_values = [7254, 5536, 4055, 2812, 1799, 1019, 467, 130, 20]
    # noise_duration = 50

    # Climate effects [CHF/m/50a] — cycling emits no CO2 during use
    # Construction CO2 of cycle path is much lower than highway
    co2_cycle = None  # TODO: set cycle path construction CO2 cost [CHF/m/50a]

    # Nature and Landscape — reduce significantly for cycling (smaller footprint)
    fragmentation = None  # TODO: reduce or set to 0 for cycle paths [CHF/m2/a]
    fragmentation_duration = 50
    habitat_loss = None  # TODO: reduce or set to 0 for cycle paths [CHF/m2/a]
    habitat_loss_duration = 30

    # Land reallocation — may apply for new dedicated cycle paths through farmland
    forest_reallocation = None  # TODO: set or 0 if Wald is not reallocated [CHF/m2/a]
    meadow_reallocation = None  # TODO: set for cycle paths through Fruchtfolgeflaeche [CHF/m2/a]
    reallocation_duration = 50

    runtimes["Initialize variables"] = time.time() - st
    st = time.time()

    ##################################################################################
    # Import and prepare raw data
    print("\nIMPORT RAW DATA \n") #TODO: raw data will be the converted network vector->nodes and edges)

    # Import shapes of lake for plots
    get_lake_data() #ok

    # Import the file containing the locations to be plotted
    import_locations() #ok

    # Define area that is protected for constructing highway links, TODO: Where can I find this, is it the same as for InfraScanRoad?
    get_protected_area(limits=limits_corridor)
    get_unproductive_area(limits=limits_corridor)
    landuse(limits=limits_corridor)

    # Tif file of all unsuitable land cover and protected areas
    # File is stored to 'data/landuse_landcover/processed/zone_no_infra/protected_area_{suffix}.tif'
    all_protected_area_to_raster(suffix="corridor")

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
    # Import the cycling network and preprocess it ALLTAG.GIS
    # higher network layer (like highway for cycling) stored in infraScanCycle/data/raw/ALLTAG/Velonetz_Alltag_-OGD.gpkg
    # gaps in higher level network stored in infraScanCycle/data/raw/SCHWACHSTELLEN/TBA_VNP_SCHWACHSTELLEN_L.shp
    # make option to join them together

    network = import_network_GIS_ALLTAG()  # already EPSG:2056, already clean

    # CLEAN DATA
    network = network[network.geometry.notnull()]

    # Import and match destination points
    df_access = pd.read_csv(r"data/raw/VELOPARKIERANLAGEN/OGD_VELOPARKIERANLAGEN_P.csv", sep=",")
    df_access.columns = df_access.columns.str.strip()  # Crucial for OGD data

    #map destination points to network
    access_points, hubs_snapped = map_access_points_on_network(network, df_access)

    runtimes["Import network data"] = time.time() - st
    st = time.time()


    ##################################################################################
    # 2) Process network

    # Simplify the physical topology of the network
    # One distinct edge between two nodes (currently multiple edges between nodes)
    # Edges are stored in r"data/Network/processed/edges.gpkg"
    # Points in simplified network can be intersections ("intersection"==1) or access points ("intersection"==0)
    # Points are stored in r"data/Network/processed/points.gpkg"
    nodes_gdf, edges_final = reformat_network()

    # Filter the infrastructure elements that lie within a given polygon
    # Points within the corridor are stored in r"data/Network/processed/points_corridor.gpkg"
    # Edges within the corridor are stored in r"data/Network/processed/edges_corridor.gpkg"
    # Edges crossing the corridor border are stored in r"data/Network/processed/edges_on_corridor.gpkg"
    points_corridor, edges_corridor, edges_border = network_in_corridor(polygon=outerboundary)


    # Add attributes to nodes within the corridor (mainly access point T/F)
    # Points with attributes saved as "data/Network/processed/points_attribute.gpkg"
    nodes_all, nodes_corridor = map_values_to_nodes()

    # Add attributes to the edges
    edges = get_edge_attributes()



    runtimes["Preprocess the network"] = time.time() - st
    st = time.time()

    # --- NETWORK QUALITY CHECK ---
    print("\n--- NETWORK QUALITY CHECK ---")
    print(
        f"Edges missing ROUTENTYP:       {edges['ROUTENTYP'].isna().sum() if 'ROUTENTYP' in edges.columns else 'col missing'}")
    print(f"One-way edges:                 {edges['oneway'].sum()}")
    print(
        f"Corridor edge coverage:        {len(edges_corridor)}/{len(edges)} ({100 * len(edges_corridor) / len(edges):.0f}%)")
    print(f"Total nodes:                   {len(nodes_all)}")
    print(f"Corridor nodes:                {len(nodes_corridor)}")

    # is_destination lives in points.gpkg (reformat_network output), not in map_values_to_nodes output
    points_all = gpd.read_file('data/Network/processed/points.gpkg')
    print(
        f"Nodes flagged as destinations: {points_all['is_destination'].sum() if 'is_destination' in points_all.columns else 'col missing'}")
    print(
        f"Nodes flagged as intersections:{points_all['is_intersection'].sum() if 'is_intersection' in points_all.columns else 'col missing'}")
    print("-----------------------------\n")

    ##################################################################################
    # 3) Generate developments (new access points) and connection to existing infrastructure

    # Make random points within the perimeter (extent) and filter them
    num_rand = 500
    random_gdf = generated_access_points(extent=innerboundary, number=num_rand)


    # Assign the return value of the function to the variable name
    filtered_gdf = filter_access_points(random_gdf)
    filtered_gdf.to_file(r"data/Network/processed/generated_nodes.gpkg")

    # Import the generated points as dataframe
    generated_points = filtered_gdf  # TODO 3c

    # Import current points as dataframe and filter only access points (no intersection points)
    current_access_points = gpd.read_file(r"data/Network/processed/points_corridor_attribute.gpkg")

    # Filter to corridor nodes only (spatial join with corridor polygon)
    poly_gdf = gpd.GeoDataFrame({'geometry': [outerboundary]}, crs="EPSG:2056")
    current_access_points = gpd.sjoin(
        current_access_points, poly_gdf, how='inner', predicate='within'
    ).drop(columns=['index_right'], errors='ignore').reset_index(drop=True)

    print(f"  Current access points in corridor: {len(current_access_points)}")
    print(f"  Columns: {current_access_points.columns.tolist()}")

    # Connect the generated points to the existing access points
    filtered_rand_temp = connect_points_to_network(generated_points, current_access_points)

    # Filter out links that are too long to be realistic cycling connections
    max_link_dist = 1500  # meters — adjust based on context
    new_links_realistic = filtered_rand_temp[
        filtered_rand_temp['dist_to_node'] <= max_link_dist
        ].copy().reset_index(drop=True)

    print(f"  -> {len(new_links_realistic)} links within {max_link_dist}m threshold "
          f"(dropped {len(filtered_rand_temp) - len(new_links_realistic)})")

    new_links_realistic.to_file('data/Network/processed/new_links_corridor.gpkg', driver='GPKG')


    # Find a routing for the generated links that considers protected areas
    raster = r'data/landuse_landcover/processed/zone_no_infra/protected_area_corridor.tif'
    routing_raster(raster_path=raster)  # TODO:Ensure the routing algorithm penalizes slope (gradient)

    # After routing_raster():
    inaccessible = pd.read_csv('data/Network/processed/points_inaccessible.csv')
    print(f"  Inaccessible point E range: {inaccessible['x'].min():.0f}–{inaccessible['x'].max():.0f}")
    print(f"  Inaccessible point N range: {inaccessible['y'].min():.0f}–{inaccessible['y'].max():.0f}")
    # Compare with raster extent using rasterio.open(raster).bounds

    # Compute the Voronoi polygons for status quo
    voronoi_sq = get_voronoi_status_quo(corridor_polygon=innerboundary)
    limits_variables = [2680600, 1227700, 2724300, 1265600]


    runtimes["Generate infrastructure developments"] = time.time() - st
    st = time.time()

    # Import the raw data, reshape it partially and store it as tif

    runtimes["Import variable for scenario (population and employment)"] = time.time() - st
    st = time.time()


    ##################################################################################
    ##################################################################################
    # SCENARIO
    print("\nSCENARIO \n")
    ##################################################################################
    # 1) Define scenario based on cantonal predictions
    # Import the predicted scenario defined by the canton of Zürich
    scenario_zh = pd.read_csv(r"data/Scenario/KTZH_00000705_00001741.csv", sep=";")

    # Define the relative growth per scenario and district
    # The growth rates are stored in "data/temp/data_scenario_n.shp"
    future_scenario_zuerich_2022(scenario_zh)
    # Plot the growth rates as computed above for population and employment and over three scenarios


    # Compute the predicted amount of population and employment in each raster cell (hectar) for each scenario
    # The resulting raster data are stored in "data/independent_variables/scenario/{col}.tif" with col being pop or empl and the scenario
    scenario_to_raster(limits_variables)

    # Aggregate the scenario data to over the voronoi polygons, here euclidian polygons
    # Store the resulting file to "data/Voronoi/voronoi_developments_euclidian_values.shp"
    polygons_gdf = gpd.read_file(r"data/Voronoi/voronoi_developments_euclidian.gpkg")
    scenario_to_voronoi(polygons_gdf, euclidean=True)

    # Convert multiple tif files to one same tif with multiple bands
    stack_tif_files(var="empl")
    stack_tif_files(var="pop")
    runtimes["Generate the scenarios"] = time.time() - st
    st = time.time()
    """
    ##################################################################################
    ##################################################################################
    # IMPLEMENT THE SCORING
    # 1) Redefine protected area for scoring perimeter
    # 2) Import road network from OSM and rasterize it
    # 3) Compute construction costs
    # 4) Compute costs of externalities
    # 5) Get Voronoi tiling based on travel time
    # 6) Compute accessibility costs

    print("\nIMPLEMENT SCORING \n")

    ##################################################################################
    # 1) Redefine protected area for scoring perimeter

    # This operation has already been done above for the corridor limits, here it is applied to the voronoi polygon limits which are bigger than the corridor limits
    get_protected_area(limits=limits_variables)
    get_unproductive_area(limits=limits_variables)
    landuse(limits=limits_variables)

    # Find possible links considering land cover and protected areas
    all_protected_area_to_raster(suffix="variables")

    ##################################################################################
    # 2) Import road network from OSM and rasterize it
    # Import the road network from OSM and rasterize it
    nw_from_osm(limits_variables)
    osm_nw_to_raster(limits_variables)
    runtimes["Import and rasterize local road network from OSM"] = time.time() - st
    st = time.time()

    # Write runtimes to a file
    with open(r'runtimes.txt', 'w') as file:
        for part, runtime in runtimes.items():
            file.write(f"{part}: {runtime}\n")
    ##################################################################################
    # 3) Compute construction costs

    # Compute the elevation profile for each routing to assess the amount
    # First import the elevation model downscale the resolution and store it as raster data to 'data/elevation_model/elevation.tif'
    resolution = 50 # meter

    runtimes["Import elevation model in 50 meter resolution"] = time.time() - st
    st = time.time()

    # Compute the elevation profile for each generated highway routing based on the elevation model
    #TODO: For cycling, slope > 4% should significantly increase the "perceived" travel time cost.


    # Based on the elevation profile of each links compute the required amount of bridges and tunnels
    # Safe the dataset to "data/Network/processed/new_links_realistic_tunnel.gpkg"

    # Convert the string representation of lists back into actual Python lists


    runtimes["Optimize eleavtion profile of links to find need for tunnel and bridges"] = time.time() - st
    st = time.time()

    # Compute the construction costs for each development (generated points with according link to existing access point)
    # Not including tunnels and bridges with regards to the elevation profile of a section yet
    # Result stored to "data/costs/construction.gpkg"
    print(" -> Construction costs")

    c_structural_maint = 1.2 / 100 # % of construction costs
    c_om_openhighway = 89.7 # CHF/m/a
    c_om_tunnel = 89.7 # CHF/m/a
    c_om_bridge = 368.8 # CHF/m/a
    maintenance_duration = 50 # years


    runtimes["Compute construction and maintenance costs"] = time.time() - st
    st = time.time()


    ##################################################################################
    # 4) Compute costs of externalities
    # Compute the costs arising from externalities for each development (generated points with according link to existing access point)
    # Result stored to "data/Network/processed/new_links_externalities_costs.gpkg"

    print(" -> Externalities")


    # Add geospatial link to the table with costs
    # Result stored to "data/costs/building_externalities.gpkg"


    # Plot individual cost elements on map

    runtimes["Compute Externalities"] = time.time() - st
    st = time.time()

    ##################################################################################
    # 5) Get Voronoi tiling based on travel time
    # Based on the rasterized road network from OSM, compute the travel time required to access the closest existing
    # cycling network access point from each cell in the perimeter. As result, it is also known for each cell which current
    # access points is the closest (its ID)
    # The raster file showing the travel time to the next access point is stored to 'data/Network/travel_time/travel_time_raster.tif'
    # The raster file showing the ID of the closest access point is stored in 'data/Network/travel_time/source_id_raster.tif'
    # Aggregating all cells with same closest access point is equivalent to a travel time based voronoi tiling. This is
    # stored as vector file in "data/Network/travel_time/Voronoi_statusquo.gpkg"




    # Same operation is made for all developments
    # These are store similarly than above, with id_new being the id of the development (ID of generated point)
    # The raster file showing the travel time to the next access point is stored to 'data/Network/travel_time/developments/dev{id_new}_travel_time_raster.tif'
    # The raster file showing the ID of the closest access point is stored in 'data/Network/travel_time/developments/dev{id_new}_source_id_raster.tif'
    # Aggregating all cells with same closest access point is equivalent to a travel time based voronoi tiling. This is
    # stored as vector file in "data/Network/travel_time/developments/dev{id_new}_Voronoi.gpkg"


    runtimes["Voronoi tiling: Compute travel time from each raster cell to the closest access point"] = time.time() - st
    st = time.time()

    # Generate one dataframe containing the Voronoi polygons for all developments and all access points within the
    # perimeter. Before the polygons are store in an individual dataset for each development
    # The resulting dataframe is stored to "data/Voronoi/combined_developments.gpkg"


    # Based on the scenario and the travel time based Voronoi tiling, compute the predicted population and employment
    # in each polygon and for each scenario
    # Resulting dataset is stored to "data/Voronoi/voronoi_developments_tt_values.shp"


    runtimes["Aggregate scenarios by Voronoi polygons"] = time.time() - st
    st = time.time()

    ##################################################################################
    # 6) Compute access time costs

    # Compute the accessibility for status quo for scenarios


    # Compute the benefit in accessibility for each development compared to the status quo
    # The accessibility for each polygon for every development is store in "data/Voronoi/voronoi_developments_local_accessibility.gpkg"
    # The benefit of each development compared to the status quo is stored in 'data/costs/local_accessibility.csv'

    runtimes["Compute network access time benefits"] = time.time() - st
    st = time.time()

    #################################################################################
    # Travel time delay on network

    # Compute the OD matrix for the current infrastructure under all scenarios


    # Compute the OD matrix for the infrastructure developments under all scenarios


    runtimes["Reallocate OD matrices to Voronoi polygons"] = time.time() - st
    st = time.time()


    # check if flow are possible

    print('Flag: link_traffic_to_map is complete')

    # Run travel time optimization for infrastructure developments and all scenarios

    print('Flag: tt_optimization_all_developments is complete')

    # Monetize travel time savings


    ##################################################################################
    # Aggregate the single cost elements to one dataframe
    # New dataframe is stored in "data/costs/total_costs.gpkg"
    # New dataframe also stored in "data/costs/total_costs.csv"
    print(" -> Aggregate costs")


    # Import to the overall cost dataframe


    # Convert all costs in million CHF


    runtimes["Aggregate costs"] = time.time() - st

    # Write runtimes to a file
    with open(r'runtimes_2.txt', 'w') as file:
        for part, runtime in runtimes.items():
            file.write(f"{part}: {runtime}/n")

    ##################################################################################
    ##################################################################################
    # VISUALIZE THE RESULTS

    print("\nVISUALIZE THE RESULTS \n")

    # Import layers to plot



    # Plot the net benefits for each generated point and interpolate the area in between


    # Get a gpd df with points have an ID_new that is not in links_realistic ID_new


    # Plot the net benefits for each generated point and interpolate the area in between
    # if plot_name is not False, then the plot is stored in "plot/results/{plot_name}.png"

    # Plot single cost element


    # Plot uncertainty

    # Plot the uncertainty of the nbr highest ranked developments as boxplot

    # Plot benefit distribution for all cost elements

    #plot the uncertainty


    # Plot influence of discounting
    """


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('You did a good job ;)')