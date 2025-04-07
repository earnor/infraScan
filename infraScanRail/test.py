

    # Compute the elevation profile for each routing to assess the amount
    # First import the elevation model downscale the resolution and store it as raster data to 'data/elevation_model/elevation.tif'
    #resolution = 50 # meter
    #import_elevation_model(new_resolution=resolution)
    runtimes["Import elevation model in 50 meter resolution"] = time.time() - st
    st = time.time()

    # Compute the elevation profile for each generated highway routing based on the elevation model
    #links_temp = get_road_elevation_profile()
    #links_temp.to_csv(r"data/Network/processed/new_links_realistic_woTunnel.csv")

    # Based on the elevation profile of each links compute the required amount of bridges and tunnels
    # Safe the dataset to "data/Network/processed/new_links_realistic_tunnel.gpkg"
    #get_tunnel_candidates(links_temp)
    #tunnel_bridges(links_temp)

    runtimes["Optimize eleavtion profile of links to find need for tunnel and bridges"] = time.time() - st
    st = time.time()


################################################################################################################################################
#Fabrice's Model for scenarios
'''
def future_scenario_zuerich_2022(df_input):
    """
    This function represents the changes in population and employment based on data from the canton of Zürich.
    :param df: DataFrame defining the growth by region
    :param lim: List of coordinates defining the perimeter investigated in the analysis
    :return:
    """

    # import boundaries of each region
    boundaries = gpd.read_file(r"data\Scenario\Boundaries\Gemeindegrenzen\UP_BEZIRKE_F.shp")

    # group per location and time
    df = df_input.copy()
    df = df.groupby(["bezirk", "jahr"]).sum("anzahl")
    df = df.reset_index()

    # filter regions of interest
    df = df[(df["jahr"] >= 2020)] # & (df["bezirk"].isin(['Bülach', 'Hinwil', 'Meilen', 'Pfäffikon', 'Uster', 'Zürich', 'Winterthur']))]
    df = df.pivot(index='bezirk', columns='jahr', values='anzahl')


    def relative_number(x):
        return x / df.iloc[:, 0]
    # Compute the relative growth in each district and for each year from 2020 to 2050
    df_rel = df.apply(relative_number,args=(), axis=0)

    # plot development per region in one
    boundaries = boundaries.merge(df_rel, left_on="BEZIRK", right_on="bezirk", how="right")

    df_scenario = boundaries[["BEZIRK", "geometry", 2050]]
    df_scenario.rename(columns={2050: 's1_pop'}, inplace=True)

    # import boundaries of each region
    empl_dev = pd.read_csv(r"data\Scenario\KANTON_ZUERICH_596.csv", sep=";", encoding='unicode_escape')
    empl_dev = empl_dev[["BFS_NR", "GEBIET_NAME", "INDIKATOR_JAHR", "INDIKATOR_VALUE"]]

    bfs_nr = gpd.read_file(r"data\Scenario\Boundaries\Gemeindegrenzen\UP_GEMEINDEN_F.shp")
    bfs_nr = bfs_nr[["BFS", "BEZIRKSNAM"]]

    empl_dev = empl_dev.merge(right=bfs_nr, left_on="BFS_NR", right_on="BFS", how="left")
    empl_dev = empl_dev.drop_duplicates(subset=['BFS_NR', 'GEBIET_NAME', 'INDIKATOR_JAHR'], keep='first')
    empl_dev = empl_dev.rename(columns={"BEZIRKSNAM":"bezirk", "INDIKATOR_JAHR":"jahr", "INDIKATOR_VALUE":"anzahl"})

    # group per location and time
    empl_dev = empl_dev.groupby(["bezirk", "jahr"]).sum("anzahl")
    empl_dev = empl_dev.reset_index()
    #print(empl_dev.head(10).to_string())

    empl_dev = empl_dev[(empl_dev["jahr"] == 2011) | (empl_dev["jahr"] == 2021)]
    empl_dev = empl_dev.pivot(index='bezirk', columns='jahr', values='anzahl').reset_index()

    # Rename the columns
    empl_dev.columns.name = None
    empl_dev.columns = ['bezirk', '2011', '2021']
    empl_dev["rel_10y"] = empl_dev["2021"] / empl_dev["2011"] - 1
    #empl_dev["empl50"] = (empl_dev["2021"] * (1 + empl_dev["rel_10y"] * 2.9)).astype(int)
    empl_dev["s1_empl"] = (1 + empl_dev["rel_10y"] * 2.9)
    empl_dev = empl_dev[["bezirk", "s1_empl"]]

    print(empl_dev.head(10).to_string())

    # plot development per region in one
    df_scenario = df_scenario.merge(empl_dev, left_on="BEZIRK", right_on="bezirk", how="right")
    #print(df_scenraio.head(10).to_string())

    # df_scenraio = boundaries[["BEZIRK", "geometry", 2050]]

    df_scenario["s2_pop"] = df_scenario["s1_pop"] - (df_scenario["s1_pop"] -1) / 3
    df_scenario["s3_pop"] = df_scenario["s1_pop"] + (df_scenario["s1_pop"] - 1) / 3

    df_scenario["s2_empl"] = df_scenario["s1_empl"] - (df_scenario["s1_empl"] -1) / 3
    df_scenario["s3_empl"] = df_scenario["s1_empl"] + (df_scenario["s1_empl"] - 1) / 3

    print(df_scenario.columns)

    """
    scen_2_pop  = [1.199, 1.261, 1.192, 1.215, 1.32, 1.32, 1.215]
    scen_2_empl = [1.169, 1.231, 1.162, 1.185, 1.29, 1.35, 1.185]

    scen_3_pop  = [1.279, 1.35, 1.272, 1.295, 1.40, 1.40, 1.295]
    scen_3_empl = [1.245, 1.35, 1.242, 1.265, 1.40, 1.45, 1.265]


    df_scenraio["scen_2_empl"] = scen_2_empl
    df_scenraio["scen_2_pop"] = scen_2_pop

    df_scenraio["scen_3_empl"] = scen_3_empl
    df_scenraio["scen_3_pop"] = scen_3_pop

    print(df_scenraio.columns)
    plot_2x3_subplots(df_scenraio, lim, network, location)
    """
    df_scenario.to_file(r"data\temp\data_scenario_n.shp")
    return



def scenario_to_raster(frame=False):
    # Load the shapefile
    scenario_polygon = gpd.read_file(r"data\temp\data_scenario_n.shp")

    if frame != False:
        # Create a bounding box polygon
        bounding_poly = box(frame[0], frame[1], frame[2], frame[3])
        len = (frame[2]-frame[0])/100
        width = (frame[3]-frame[1])/100
        print(f"frame: {len, width} it should be 377, 437")

        # Calculate the difference polygon
        # This will be the area in the bounding box not covered by existing polygons
        difference_poly = bounding_poly
        for geom in scenario_polygon['geometry']:
            difference_poly = difference_poly.difference(geom)

        # Calculate the mean values for the three columns
        #mean_values = scenario_polygon.mean()

        # Create a new row for the difference polygon
        #new_row = {'geometry': difference_poly, 's1_pop': mean_values['s1_pop'], 's2_pop': mean_values['s2_pop'],
        #           's3_pop': mean_values['s3_pop'], 's1_empl': mean_values['s1_empl'], 's2_empl': mean_values['s2_empl'], 's3_empl': mean_values['s3_empl']}
        new_row = {'geometry': difference_poly, 's1_pop': scenario_polygon['s1_pop'].mean(),
                   's2_pop': scenario_polygon['s2_pop'].mean(), 's3_pop': scenario_polygon['s3_pop'].mean(),
                   's1_empl': scenario_polygon['s1_empl'].mean(), 's2_empl': scenario_polygon['s2_empl'].mean(),
                   's3_empl': scenario_polygon['s3_empl'].mean()}
        print("New row added")
        #scenario_polygon = scenario_polygon.append(new_row, ignore_index=True)
        scenario_polygon = gpd.GeoDataFrame(pd.concat([pd.DataFrame(scenario_polygon), pd.DataFrame(pd.Series(new_row)).T], ignore_index=True))

    growth_rate_columns_pop = ["s1_pop", "s2_pop", "s3_pop"]
    path_pop = r"data\independent_variable\processed\raw\pop20.tif"

    growth_rate_columns_empl = ["s1_empl", "s2_empl", "s3_empl"]
    path_empl = r"data\independent_variable\processed\raw\empl20.tif"

    growth_to_tif(scenario_polygon, path=path_pop, columns=growth_rate_columns_pop)
    growth_to_tif(scenario_polygon, path=path_empl, columns=growth_rate_columns_empl)
    print('Scenario_To_Raster complete')
    return


def growth_to_tif(polygons, path, columns):
    # Load the raster data
    aws_session = AWSSession(requester_pays=True)
    with rio.Env(aws_session):
        with rasterio.open(path) as src:
            raster = src.read(1)  # Assuming a single band raster

            # Iterate over each growth rate column
            for col in columns:
                # Create a new copy of the original raster to apply changes for each column
                modified_raster = raster.copy()

                for index, row in polygons.iterrows():
                    polygon = row['geometry']
                    change_rate = row[col]  # Use the current growth rate column

                    # Create a mask to identify raster cells within the polygon
                    mask = geometry_mask([polygon], out_shape=modified_raster.shape, transform=src.transform, invert=True)

                    # Apply the change rate to the cells within the polygon
                    modified_raster[mask] *= (change_rate)  # You may need to adjust this based on how your change rates are stored

                # Save the modified raster data to a new TIFF file
                output_tiff = f'data\independent_variable\processed\scenario\{col}.tif'
                with rasterio.open(output_tiff, 'w', **src.profile) as dst:
                    dst.write(modified_raster, 1)
    return
'''

def scenario_to_voronoi(polygons_gdf, euclidean=False):
    print('Scenario_To_Voronoi started')
    # List of your raster files
    raster_path = r"data\independent_variable\processed\scenario"
    raster_files = ['s1_empl.tif', 's2_empl.tif', 's3_empl.tif', 's1_pop.tif', 's2_pop.tif', 's3_pop.tif']

    # Loop over the raster files and calculate zonal stats for each
    for i, raster_file in enumerate(raster_files, start=1):
        with rasterio.open(raster_path+'\\' + raster_file) as src:
            affine = src.transform
            array = src.read(1)
            nodata = src.nodata
            nodata = -999
            #polygons_gdf[(raster_file).removesuffix('.tif')] = pd.DataFrame(zonal_stats(vectors=polygons_gdf['geometry'], raster=src, stats='mean'))['mean']
            # Calculate zonal statistics
            stats = zonal_stats(polygons_gdf, array, affine=affine, stats=['sum'], nodata=nodata, all_touched=True)


            # Extract the 'sum' values and assign them to a new column in the geodataframe
            polygons_gdf[(raster_file).removesuffix('.tif')] = [stat['sum'] for stat in stats]

    # Now polygons_gdf has new columns with the sum of raster values for each polygon
    # You can save this geodataframe to a new file if desired
    #print(polygons_gdf.head(10).to_string())
    if euclidean:
        polygons_gdf.to_file(r"data\Voronoi\voronoi_developments_euclidian_values.shp")
    else:
        polygons_gdf.to_file(r"data\Voronoi\voronoi_developments_tt_values.shp")
    print('Scenario_To_Voronoi complete')
    return






















import pandas as pd
from shapely.geometry import LineString

def calculate_od_matrix_with_penalties(G):
    # Initialize an empty list to collect OD records
    od_records = []

    # Loop over each pair of nodes in the graph
    for origin in G.nodes:
        origin_station = G.nodes[origin]['station']  # Get the station name for the origin
        # Use Dijkstra's algorithm to find shortest paths from the origin node
        paths = nx.single_source_dijkstra_path(G, origin, weight='weight')
        travel_times = nx.single_source_dijkstra_path_length(G, origin, weight='weight')
        
        for destination, path in paths.items():
            if origin != destination:
                destination_station = G.nodes[destination]['station']  # Get the station name for the destination
                
                # Calculate the number of unique lines used along the path
                lines_used = []
                total_travel_time = 0
                path_geometry = []

                for i in range(len(path) - 1):
                    edge_data = G.get_edge_data(path[i], path[i + 1])
                    total_travel_time += edge_data['weight']
                    lines_used.append(edge_data['service'])

                    # Collect the geometry of each edge in the path
                    path_geometry.append(edge_data['geometry'])
                
                # Combine geometries into a single LineString
                full_path_geometry = LineString([pt for geom in path_geometry for pt in geom.coords])

                # Calculate the total travel time, adding 5 minutes for each line change beyond the first
                num_lines_used = len(set(lines_used))
                penalty_time = (num_lines_used - 1) * 5
                adjusted_travel_time = total_travel_time + penalty_time

                # Add to the list of OD records
                od_records.append({
                    'Origin': origin,
                    'OriginStation': origin_station,
                    'Destination': destination,
                    'DestinationStation': destination_station,
                    'FastestTravelTime': total_travel_time,
                    'NumLinesUsed': num_lines_used,
                    'TotalTravelTime': adjusted_travel_time,
                    'Geometry': full_path_geometry
                })

    # Convert the list of records to a DataFrame
    od_matrix = pd.DataFrame(od_records)
    return od_matrix

























import geopandas as gpd
import networkx as nx

def define_rail_network():
    # Load the main railway services dataset
    nw_gdf = gpd.read_file(r"data/temp/network_railway-services.gpkg")
    
    # Load filtered links with dev_id and Time from the GeoPackage
    filtered_links_gdf = gpd.read_file(r"data/Network/processed/filtered_new_links_in_corridor.gpkg")

    # Initialize the base directed graph for the rail network
    G_base = nx.DiGraph()

    # Populate the base network graph from nw_gdf
    for _, row in nw_gdf.iterrows():
        # Define nodes and edge attributes
        from_node = row['FromNode']
        to_node = row['ToNode']
        from_station = row['FromStation']
        to_station = row['ToStation']
        service = row['Service']
        direction = row['Direction']
        weight = row['TravelTime'] + row['InVehWait']
        frequency = row['Frequency']
        line_geom = row['geometry']
        
        # Add nodes and edges to the base graph
        if not G_base.has_node(from_node):
            G_base.add_node(from_node, station=from_station, geometry=line_geom.coords[0])
        if not G_base.has_node(to_node):
            G_base.add_node(to_node, station=to_station, geometry=line_geom.coords[-1])
        
        G_base.add_edge(from_node, to_node, service=service, direction=direction, weight=weight, frequency=frequency, geometry=line_geom)

    # Initialize a dictionary to store graphs by dev_id
    dev_networks = {}

    # Iterate over each unique dev_id in the filtered links
    for dev_id, link_row in filtered_links_gdf.groupby('dev_id'):
        # Clone the base network for each development ID
        G_dev = G_base.copy()
        
        # Add the specific link for the current dev_id
        for _, row in link_row.iterrows():
            from_node = row['from_ID_new']
            to_node = row['to_ID_new']
            time = row['Time']
            line_geom = row['geometry']

            # Ensure nodes exist before adding an edge
            if not G_dev.has_node(from_node):
                G_dev.add_node(from_node, geometry=line_geom.coords[0])
            if not G_dev.has_node(to_node):
                G_dev.add_node(to_node, geometry=line_geom.coords[-1])

            # Add the link with the specified Time as the weight
            G_dev.add_edge(from_node, to_node, service='new_service', weight=time, geometry=line_geom)

        # Store the graph with the development ID
        dev_networks[dev_id] = G_dev

    return dev_networks


##only_links_to_corridor(poly=outerboundary) calculate for this here the time for each development and then try the function above













































import os
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString, GeometryCollection
from shapely.ops import split
import matplotlib.pyplot as plt



# Set up working directory and file paths
os.chdir(r"C:\Users\phili\polybox\ETH_RE&IS\Master Thesis\06-Developments\01-Code\infraScanRail")
s_bahn_lines_path = r"data/Network/Buslines/Linien_des_offentlichen_Verkehrs_-OGD.gpkg"
layer_name_segmented = 'ZVV_S_BAHN_Linien_L'
stops_path = r"data/Network/Buslines/Haltestellen_des_offentlichen_Verkehrs_-OGD.gpkg"

# Load S-Bahn lines and stops data
s_bahn_lines = gpd.read_file(s_bahn_lines_path, layer=layer_name_segmented)
stops = gpd.read_file(stops_path)

# Run the function to split lines at stops
split_lines_gdf = split_multilinestrings_at_stops(s_bahn_lines, stops)

# Plot the result
fig, ax = plt.subplots(figsize=(10, 10))
s_bahn_lines.plot(ax=ax, color='gray', linewidth=1, label='Original S-Bahn Lines')
stops.plot(ax=ax, color='blue', marker='o', label='Stops', markersize=5)
split_lines_gdf.plot(ax=ax, color='red', linewidth=1, label='Split Line Segments')
plt.legend()
plt.show()

# Save the result to a new file if needed
split_lines_gdf.to_file(r"C:\Users\phili\polybox\ETH_RE&IS\Master Thesis\06-Developments\01-Code\infraScanRail\split_s_bahn_lines.gpkg", driver="GPKG")










def calculate_od_pairs_with_status(G_bus):
    # Extract nodes of type 'Train' and 'Bus'
    train_stations = [node for node, attr in G_bus.nodes(data=True) if attr['type'] == 'Train']
    bus_stops = [node for node, attr in G_bus.nodes(data=True) if attr['type'] == 'Bus']
    
    # Initialize a list to store OD pairs and travel times
    od_pairs = []

    # Iterate over each train station
    for train_id in train_stations:
        # For each train station, iterate over all bus stops
        for bus_id in bus_stops:
            try:
                # Calculate the shortest path and its length
                shortest_path_length = nx.shortest_path_length(
                    G_bus, source=train_id, target=bus_id, weight='weight'
                )
                # Append the OD pair, travel time, and status to the result
                od_pairs.append({
                    'origin': train_id,
                    'destination': bus_id,
                    'travel_time': shortest_path_length,
                    'status': 'connected'
                })
            except nx.NetworkXNoPath:
                # Handle cases where no path exists between the nodes
                od_pairs.append({
                    'origin': train_id,
                    'destination': bus_id,
                    'travel_time': float('inf'),  # No connection
                    'status': 'not connected'
                })

    # Convert the result into a DataFrame for better handling
    od_df = pd.DataFrame(od_pairs)

    return od_df


def find_closest_destinations_with_status(od_df):
    # Filter out rows where the status is 'connected'
    connected_df = od_df[od_df['status'] == 'connected']

    # Find the closest destination for each origin based on minimum travel time
    closest_connected = connected_df.loc[connected_df.groupby('origin')['travel_time'].idxmin()]

    # Identify origins with no connections (status == 'not connected' for all destinations)
    all_origins = od_df['origin'].unique()
    connected_origins = closest_connected['origin'].unique()
    not_connected_origins = set(all_origins) - set(connected_origins)

    # Create a DataFrame for not connected origins
    not_connected_df = pd.DataFrame({
        'origin': list(not_connected_origins),
        'destination': None,
        'travel_time': float('inf'),
        'status': 'not connected'
    })

    # Combine the closest connected and not connected DataFrames
    result_df = pd.concat([closest_connected, not_connected_df], ignore_index=True)

    # Sort by origin for better readability
    result_df.sort_values(by='origin', inplace=True)

    return result_df



# Dictionary to hold the closest train station for each bus stop
closest_train_stations = {}

# Loop through all nodes in the graph
for stop_id in G_bus.nodes:
    stop_type = G_bus.nodes[stop_id]['type']
    
    if stop_type == 'Bus':  # Only process bus stops
        min_distance = float('inf')  # Initialize minimum distance
        closest_station = None  # Initialize closest station
        
        # Get the position of the current bus stop
        pos_bus = G_bus.nodes[stop_id]['pos']
        point_bus = Point(pos_bus)

        # Loop through all nodes to find the closest train station
        for other_stop_id in G_bus.nodes:
            if G_bus.nodes[other_stop_id]['type'] == 'Train':  # Check for train stops
                pos_train = G_bus.nodes[other_stop_id]['pos']
                point_train = Point(pos_train)
                
                # Calculate distance between the bus stop and the train station
                distance = point_bus.distance(point_train)
                
                # Update the closest train station if this one is closer
                if distance < min_distance:
                    min_distance = distance
                    closest_station = other_stop_id
        
        # Assign "not connected to bus" if no train station is found
        if closest_station is None:
            closest_train_stations[stop_id] = 'not connected to bus'
        else:
            closest_train_stations[stop_id] = closest_station

# Output the results
for bus_stop, train_station in closest_train_stations.items():
    if train_station == 'not connected to bus':
        print(f"Bus Stop {bus_stop} -> Not connected to any train station")
    else:
        print(f"Bus Stop {bus_stop} -> Closest Train Station {train_station}")

# Check which bus stations are closest to a trainstation

# Define the specific train station ID to find closest bus stops for
target_train_station_id = 10008 #aathal
##12882

# List to hold closest bus stops and their names
closest_bus_stops = []

# Loop through the closest_train_stations dictionary
for bus_stop, closest_station in closest_train_stations.items():
    if closest_station == target_train_station_id:  # Check if the closest station is the target train station
        # Get the bus stop name from the stops_filtered DataFrame
        bus_stop_row = stops_filtered[stops_filtered['DIVA_NR'] == bus_stop]
        if not bus_stop_row.empty:
            bus_stop_name = bus_stop_row.iloc[0]['CHSTNAME']  # Assuming CHSTNAME is the column name
            closest_bus_stops.append((bus_stop, bus_stop_name))

# Convert the results into a DataFrame for better readability
closest_bus_stops_df = pd.DataFrame(closest_bus_stops, columns=['Bus Stop ID', 'Bus Stop Name'])

# Print the DataFrame with closest bus stops to the specified train station
print(closest_bus_stops_df)


# Function to calculate total travel time (walking + bus)
def calculate_total_travel_time(bus_times_within_buffer, G_bus, closest_train_stations):
    total_times = []
    
    for index, row in bus_times_within_buffer.iterrows():
        grid_point = row['geometry']
        nearest_bus_stop = row['nearest_bus_stop']
        walking_time = row['walking_time']

        if nearest_bus_stop not in closest_train_stations:
            total_times.append({'grid_point': grid_point, 'closest_train_station': None, 'total_time': float('inf')})
            continue

        closest_train_station = closest_train_stations[nearest_bus_stop]

        try:
            bus_travel_time = nx.shortest_path_length(G_bus, source=nearest_bus_stop, target=closest_train_station, weight='weight')
            total_time = walking_time + bus_travel_time
            total_times.append({'grid_point': grid_point, 'closest_train_station': closest_train_station, 'total_time': total_time})

        except nx.NetworkXNoPath:
            total_times.append({'grid_point': grid_point, 'closest_train_station': closest_train_station, 'total_time': float('inf')})

    return pd.DataFrame(total_times)


def save_points_as_raster(df):
    # Extract coordinates from 'grid_point'
    df[['x', 'y']] = df['grid_point'].str.extract(r'POINT \((\d+) (\d+)\)').astype(float)

    # Define grid extent
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()
    resolution = 100  # Assuming grid spacing is 100 meters

    # Create raster grid dimensions
    x_range = np.arange(x_min, x_max + resolution, resolution)
    y_range = np.arange(y_min, y_max + resolution, resolution)
    nrows, ncols = len(y_range), len(x_range)

    # Initialize raster arrays
    total_time_raster = np.full((nrows, ncols), 99999, dtype=float)
    station_raster = np.full((nrows, ncols), -1, dtype=float)  # Use -1 for 'noPT'

    # Map DataFrame values to raster
    for _, row in df.iterrows():
        col = int((row['x'] - x_min) / resolution)
        row_idx = int((y_max - row['y']) / resolution)
        total_time_raster[row_idx, col] = row['total_time']
        station_raster[row_idx, col] = row['closest_train_station']

    # Define raster metadata
    transform = from_origin(x_min, y_max, resolution, resolution)
    metadata = {
        'driver': 'GTiff',
        'height': nrows,
        'width': ncols,
        'count': 2,  # Two bands: total_time and closest_train_station
        'dtype': 'float32',
        'crs': 'EPSG:4326',  # Example CRS, adjust as needed
        'transform': transform,
    }

    # Save as GeoTIFF
    with rasterio.open('output.tif', 'w', **metadata) as dst:
        dst.write(total_time_raster, 1)  # First band: total_time
        dst.write(station_raster, 2)    # Second band: closest_train_station 
    
    return

















def correct_rasters_to_extent(
    empl_path, pop_path, 
    output_empl_path, output_pop_path,
    reference_boundary, resolution=100, crs="EPSG:2056"):

    """
    Corrects the raster files to match the given boundary extent and resolution.

    Args:
        empl_path (str): Path to the employment raster file.
        pop_path (str): Path to the population raster file.
        output_empl_path (str): Path to save the corrected employment raster.
        output_pop_path (str): Path to save the corrected population raster.
        reference_boundary (shapely.geometry.Polygon): Boundary polygon for cropping and masking.
        resolution (int): Resolution of the output raster in meters. Default is 100.
        crs (str): Coordinate Reference System for the output rasters. Default is "EPSG:2056".
    """
    # Determine the bounds and raster grid size
    xmin, ymin, xmax, ymax = reference_boundary.bounds
    width = int((xmax - xmin) / resolution)
    height = int((ymax - ymin) / resolution)
    transform = from_origin(xmin, ymax, resolution, resolution)

    # Convert the boundary to GeoJSON-like format for masking
    boundary_geom = [mapping(reference_boundary)]

    def process_raster(input_path, output_path):
        with rasterio.open(input_path) as src:
            # Initialize an empty array for the corrected raster
            data_corrected = np.zeros((height, width), dtype=src.dtypes[0])

            # Reproject and resample
            reproject(
                source=rasterio.band(src, 1),
                destination=data_corrected,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.bilinear,
            )

            # Mask the raster to fit within the boundary
            mask = geometry_mask(boundary_geom, transform=transform, invert=True, out_shape=(height, width))
            data_corrected[~mask] = np.nan  # Set values outside the boundary to NaN

            # Save the corrected raster
            profile = src.profile
            profile.update(
                driver="GTiff",
                height=height,
                width=width,
                transform=transform,
                crs=crs,
                nodata=np.nan,
            )
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(data_corrected, 1)

    # Process both employment and population rasters
    process_raster(empl_path, output_empl_path)
    process_raster(pop_path, output_pop_path)


    def define_rail_network():
    # Load the GeoDataFrame from the GeoPackage
    nw_gdf = gpd.read_file(r"data/temp/network_railway-services.gpkg")

    # Initialize a directed graph
    G = nx.DiGraph()

    # Iterate over each row in the GeoDataFrame to add nodes and edges
    for _, row in nw_gdf.iterrows():
        # Define the nodes (stations) and edge attributes
        from_node = row['FromNode']
        to_node = row['ToNode']
        from_station = row['FromStation']
        to_station = row['ToStation']
        service = row['Service']
        direction = row['Direction']
        weight = row['TravelTime'] + row['InVehWait']
        frequency = row['Frequency']
    
        # Extract start and end points of the LineString geometry
        line_geom = row['geometry']
        from_geometry = line_geom.coords[0]  # Start point of the LineString
        to_geometry = line_geom.coords[-1]   # End point of the LineString

        # Add the nodes if they don't already exist in the graph
        if not G.has_node(from_node):
            G.add_node(from_node, station=from_station, geometry=from_geometry)
        if not G.has_node(to_node):
            G.add_node(to_node, station=to_station, geometry=to_geometry)
        
        # Add a directed edge between the nodes with the specified attributes
        G.add_edge(from_node, to_node, service=service, direction=direction, weight=weight, frequency=frequency, geometry=line_geom)
    
    return G

def plot_rail_network(graphs):
    """
    Plot a list of rail network graphs.

    Args:
        graphs (list): A list of NetworkX graphs (DiGraph objects).
    """
    for i, G in enumerate(graphs):
        # Create a dictionary for positions using the node geometries in G
        pos = {node: (data['geometry'][0], data['geometry'][1]) for node, data in G.nodes(data=True)}

        # Plot the graph with NetworkX, using the geographic coordinates for positioning
        plt.figure(figsize=(10, 10))
        
        # Draw nodes and labels
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue', alpha=0.7)
        nx.draw_networkx_labels(G, pos, labels={node: data['station'] for node, data in G.nodes(data=True)}, font_size=5)
        
        # Draw edges with weights as edge labels
        nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=10, edge_color="gray", width=0.5)
        edge_labels = {(u, v): f"{d['service']}, {d['weight']}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5)

        # Show the plot for this graph
        plt.title(f"Rail Network Graph {i + 1}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.show()

def create_network_foreach_dev():
    # Load the GPK file
    input_gpkg = "data/Network/processed/combined_network_with_new_links.gpkg"
    output_directory = "data/Network/processed/developments/"  # Directory to save output files
    os.makedirs(output_directory, exist_ok=True)  # Ensure the output directory exists

    # Read the GeoPackage
    gdf = gpd.read_file(input_gpkg)

    # Separate rows based on `new_dev`
    base_gdf = gdf[gdf['new_dev'] == "No"]  # Base: Rows where `new_dev` is "No"
    new_dev_rows = gdf[gdf['new_dev'] == "Yes"]  # Rows where `new_dev` is "Yes"

    # Iterate through unique dev_id values in the `new_dev == "Yes"` rows
    for dev_id in new_dev_rows['dev_id'].dropna().unique():
        # Select one row for the current dev_id
        selected_row = new_dev_rows[new_dev_rows['dev_id'] == dev_id].iloc[0]  # Choose the first row for the dev_id

        # Create a GeoDataFrame for the selected row
        selected_row_gdf = gpd.GeoDataFrame([selected_row], crs=gdf.crs)

        # Combine base GeoDataFrame with the selected row (new_dev == "Yes")
        combined_gdf = pd.concat([base_gdf, selected_row_gdf], ignore_index=True)

        # Save to the specified directory, naming the file after dev_id
        output_gpkg = os.path.join(output_directory, f"{dev_id}.gpkg")
        combined_gdf.to_file(output_gpkg, driver="GPKG")
        print(f"Saved: {output_gpkg}")

    print("Processing complete.")


    def update_network_with_new_links(network_railway_service_path, new_links_updated_path, output_path):
    """
    Add new links to the network_railway_service GeoDataFrame with new columns indicating
    the source of each link (existing or new).
    
    Parameters:
    - network_railway_service_path (str): File path to the existing network_railway_service GeoDataFrame.
    - new_links_updated_path (str): File path to the new links to be added (new_links_updated).
    - output_path (str): File path to save the combined GeoDataFrame with additional columns.
    
    Returns:
    - combined_gdf (GeoDataFrame): Combined GeoDataFrame with updated columns.
    """
    os.chdir(r"C:\Users\phili\polybox\ETH_RE&IS\Master Thesis\06-Developments\01-Code\infraScanRail")
    network_railway_service_path = r"data/temp/network_railway-services.gpkg"
    new_links_updated_path = r"data/Network/processed/updated_new_links.gpkg"
    output_path = r"data/Network/processed/combined_network_with_new_links.gpkg"


    # Load the existing railway network and new links GeoDataFrames
    network_railway_service = gpd.read_file(network_railway_service_path)
    new_links_updated = gpd.read_file(new_links_updated_path)

    # Add required columns to new_links_updated
    new_links_updated["new_dev"] = "Yes"  # Mark all new links with "Yes"
    new_links_updated["FromNode"] = new_links_updated["from_ID_new"]  # Rename from_ID_new to FromNode
    new_links_updated["ToNode"] = new_links_updated["to_ID"]  # Rename to_ID to ToNode
    new_links_updated["TravelTime"] = new_links_updated["time"]  # Copy 'time' column to 'TravelTime'
    new_links_updated["InVehWait"] = 0  # Set 'InVehWait' to 0 for new links

    # Add required columns to network_railway_service
    #network_railway_service["dev_id"] = None  # No 'dev_id' for existing links
    #network_railway_service["new_dev"] = "No"  # Mark all existing links with "No"
    #network_railway_service["InVehWait"] = None  # Add 'InVehWait' column to match
    #network_railway_service["TravelTime"] = network_railway_service["TravelTime"]  # Keep TravelTime column

    # Ensure all columns from the original are present in the new links
    for column in network_railway_service.columns:
        if column not in new_links_updated.columns:
            new_links_updated[column] = None  # Add missing columns with default values

    # Reorder columns in new_links_updated to match network_railway_service
    new_links_updated = new_links_updated[network_railway_service.columns]

    # Combine the two GeoDataFrames
    combined_gdf = pd.concat([network_railway_service, new_links_updated], ignore_index=True)

    # Save the combined GeoDataFrame to the specified output path
    combined_gdf.to_file(output_path, driver="GPKG")
    
    return combined_gdf




def update_network_with_new_links(network_railway_service_path, new_links_updated_path, output_path):
    """
    Add new links to the network_railway_service GeoDataFrame with new columns indicating
    the source of each link (existing or new).
    
    Parameters:
    - network_railway_service_path (str): File path to the existing network_railway_service GeoDataFrame.
    - new_links_updated_path (str): File path to the new links to be added (new_links_updated).
    - output_path (str): File path to save the combined GeoDataFrame with additional columns.
    
    Returns:
    - combined_gdf (GeoDataFrame): Combined GeoDataFrame with updated columns.
    """
    os.chdir(r"C:\Users\phili\polybox\ETH_RE&IS\Master Thesis\06-Developments\01-Code\infraScanRail")
    network_railway_service_path = r"data/temp/network_railway-services.gpkg"
    new_links_updated_path = r"data/Network/processed/updated_new_links.gpkg"
    output_path = r"data/Network/processed/combined_network_with_new_links.gpkg"
    # Load the existing railway network and new links GeoDataFrames
    network_railway_service = gpd.read_file(network_railway_service_path)
    new_links_updated = gpd.read_file(new_links_updated_path)

    # Ensure that new_links_updated has the same columns as network_railway_service
    for column in network_railway_service.columns:
        if column not in new_links_updated.columns:
            new_links_updated[column] = None  # Add missing columns with default values

    # Add required columns to new_links_updated
    new_links_updated["new_dev"] = "Yes"  # Mark all new links with "Yes"
    new_links_updated["FromNode"] = new_links_updated["from_ID_new"]  # Rename from_ID_new to FromNode
    new_links_updated["ToNode"] = new_links_updated["to_ID"]  # Rename to_ID to ToNode
    new_links_updated["TravelTime"] = new_links_updated["time"]  # Copy 'time' column to 'TravelTime'
    new_links_updated["InVehWait"] = 0  # Set 'InVehWait' to 0 for new links

    # Ensure all other columns from network_railway_service are in new_links_updated
    # Copy over values where appropriate for new links
    for column in network_railway_service.columns:
        if column not in new_links_updated.columns:
            new_links_updated[column] = None  # Add missing columns if any
        else:
            if column == "FromNode":
                new_links_updated[column] = new_links_updated["FromNode"]
            elif column == "ToNode":
                new_links_updated[column] = new_links_updated["ToNode"]
            elif column == "TravelTime":
                new_links_updated[column] = new_links_updated["TravelTime"]
            elif column == "InVehWait":
                new_links_updated[column] = new_links_updated["InVehWait"]
            # Handle other necessary columns (like 'Peak', 'OffPeak', etc.)
            # You can add specific logic to populate these based on your data.

    # Reorder columns to match the original dataframe
    new_links_updated = new_links_updated[network_railway_service.columns]

    # Combine the two GeoDataFrames
    combined_gdf = pd.concat([network_railway_service, new_links_updated], ignore_index=True)

    # Save the combined GeoDataFrame to the specified output path
    combined_gdf.to_file(output_path, driver="GPKG")
    
    return combined_gdf

def update_network_with_new_links(network_railway_service_path, new_links_updated_path, output_path):
    """
    Add new links to the network_railway_service GeoDataFrame with new columns indicating
    the source of each link (existing or new). Add FromStation and ToStation names for
    new links based on FromNode and ToNode matches.
    
    Parameters:
    - network_railway_service_path (str): File path to the existing network_railway_service GeoDataFrame.
    - new_links_updated_path (str): File path to the new links to be added (new_links_updated).
    - output_path (str): File path to save the combined GeoDataFrame with additional columns.
    
    Returns:
    - combined_gdf (GeoDataFrame): Combined GeoDataFrame with updated columns.
    """

    network_railway_service_path = r"data/temp/network_railway-services.gpkg"
    new_links_updated_path = r"data/Network/processed/updated_new_links.gpkg"
    output_path = r"data/Network/processed/combined_network_with_new_links.gpkg"

    network_railway_service_path = r"data/temp/network_railway-services.gpkg"
    new_links_updated_path = r"data/Network/processed/updated_new_links.gpkg"
    rail_node = pd.read_csv(r"data/Network/Rail_Node.csv", sep=";",decimal=",", encoding = "ISO-8859-1")
    output_path = r"data/Network/processed/combined_network_with_new_links.gpkg"

    # Load the existing railway network and new links GeoDataFrames
    network_railway_service = gpd.read_file(network_railway_service_path)
    new_links_updated = gpd.read_file(new_links_updated_path)

    # Ensure that new_links_updated has the same columns as network_railway_service
    for column in network_railway_service.columns:
        if column not in new_links_updated.columns:
            new_links_updated[column] = None  # Add missing columns with default values

    # Add required columns to new_links_updated
    new_links_updated["new_dev"] = "Yes"
    new_links_updated["FromNode"] = new_links_updated["from_ID_new"]
    new_links_updated["ToNode"] = new_links_updated["to_ID"]
    new_links_updated["TravelTime"] = new_links_updated["time"]
    new_links_updated["InVehWait"] = 0
    new_links_updated["Service"] = "S99"  # Set Service to "S99"
    new_links_updated["Frequency"] = 2
    new_links_updated["TotalPeakCapacity"] = 690
    new_links_updated["Capacity"] = 345

    # Add "new_dev" column to the original network if not present
    if "new_dev" not in network_railway_service.columns:
        network_railway_service["new_dev"] = "No"
    
    # Fill missing FromStation and ToStation in network_railway_service
    network_railway_service["FromStation"] = network_railway_service.apply(
        lambda row: f"Unknown_Station_{int(row['FromNode'])}" if pd.isna(row["FromStation"]) and pd.notna(row["FromNode"]) else row["FromStation"], axis=1
    )
    network_railway_service["ToStation"] = network_railway_service.apply(
        lambda row: f"Unknown_Station_{int(row['ToNode'])}" if pd.isna(row["ToStation"]) and pd.notna(row["ToNode"]) else row["ToStation"], axis=1
    )

    # Create a mapping from FromNode/ToNode to Station names
    node_to_station = {}
    for idx, row in network_railway_service.iterrows():
        if pd.notna(row["FromNode"]) and pd.notna(row["FromStation"]):
            node_to_station[row["FromNode"]] = row["FromStation"]
        if pd.notna(row["ToNode"]) and pd.notna(row["ToStation"]):
            node_to_station[row["ToNode"]] = row["ToStation"]


    # Add FromStation and ToStation for new_dev links based on FromNode and ToNode
    new_links_updated["FromStation"] = new_links_updated["FromNode"].map(node_to_station)
    new_links_updated["ToStation"] = new_links_updated["ToNode"].map(node_to_station)

    # Fill missing FromStation or ToStation with placeholder names if needed
    new_links_updated["FromStation"].fillna("Unknown_Station", inplace=True)
    new_links_updated["ToStation"].fillna("Unknown_Station", inplace=True)

    # Duplicate new links: one with Direction A, one with Direction B
    new_links_with_direction = []
    for idx, row in new_links_updated.iterrows():
        # Row with Direction A
        row_a = row.copy()
        row_a["Direction"] = "A"

        # Row with Direction B (swapping FromNode and ToNode)
        row_b = row.copy()
        row_b["Direction"] = "B"
        row_b["FromNode"], row_b["ToNode"] = row_b["ToNode"], row_b["FromNode"]
        row_b["FromStation"], row_b["ToStation"] = row_b["ToStation"], row_b["FromStation"]  # Swap station names

        # Append to the list
        new_links_with_direction.extend([row_a, row_b])

    # Convert the list of rows back into a GeoDataFrame
    new_links_updated_with_direction = gpd.GeoDataFrame(new_links_with_direction, geometry="geometry")

    # Combine the original network with the updated new links
    combined_gdf = pd.concat([network_railway_service, new_links_updated_with_direction], ignore_index=True)

    # Save the combined GeoDataFrame to the specified output path
    combined_gdf.to_file(output_path, driver="GPKG")
    
    return combined_gdf


def update_network_with_new_links(network_railway_service_path, new_links_updated_path, output_path):
    """
    Add new links to the network_railway_service GeoDataFrame with new columns indicating
    the source of each link (existing or new). Add FromStation and ToStation names for
    new links based on FromNode and ToNode matches, using Rail_Node data for lookup.
    
    Parameters:
    - network_railway_service_path (str): File path to the existing network_railway_service GeoDataFrame.
    - new_links_updated_path (str): File path to the new links to be added (new_links_updated).
    - rail_node_path (str): File path to the Rail_Node CSV file.
    - output_path (str): File path to save the combined GeoDataFrame with additional columns.
    
    Returns:
    - combined_gdf (GeoDataFrame): Combined GeoDataFrame with updated columns.
    """

    # Load the existing railway network, new links, and rail node data
    network_railway_service = gpd.read_file(network_railway_service_path)
    new_links_updated = gpd.read_file(new_links_updated_path)
    rail_node = pd.read_csv(r"data/Network/Rail_Node.csv", sep=";",decimal=",", encoding = "ISO-8859-1")

    # Ensure Rail_Node has the required mapping
    rail_node_mapping = rail_node.set_index("NR")["NAME"].to_dict()

    # Manual addition of missing station names for nodes 2298 and 1018
    # Debugging showed that these nodes were consistently missing, so we add them manually.
    rail_node_mapping[2298] = "Uster"  # Node 2298 corresponds to Uster
    rail_node_mapping[1018] = "Hinwil"  # Node 1018 corresponds to Hinwil

    # Add required columns to new_links_updated
    new_links_updated["new_dev"] = "Yes"
    new_links_updated["FromNode"] = new_links_updated["from_ID_new"]
    new_links_updated["ToNode"] = new_links_updated["to_ID"]
    new_links_updated["TravelTime"] = new_links_updated["time"]
    new_links_updated["InVehWait"] = 0
    new_links_updated["Service"] = "S99"
    new_links_updated["Frequency"] = 2
    new_links_updated["TotalPeakCapacity"] = 690
    new_links_updated["Capacity"] = 345

    # Fill missing FromStation and ToStation in network_railway_service
    network_railway_service["FromStation"] = network_railway_service["FromStation"].fillna(
        network_railway_service["FromNode"].apply(lambda x: rail_node_mapping.get(x, f"Unknown_Station_{int(x)}") if pd.notna(x) else None)
    )
    network_railway_service["ToStation"] = network_railway_service["ToStation"].fillna(
        network_railway_service["ToNode"].apply(lambda x: rail_node_mapping.get(x, f"Unknown_Station_{int(x)}") if pd.notna(x) else None)
    )

    # Create a mapping from nodes to station names
    node_to_station = {}
    for idx, row in network_railway_service.iterrows():
        if pd.notna(row["FromNode"]) and pd.notna(row["FromStation"]):
            node_to_station[row["FromNode"]] = row["FromStation"]
        if pd.notna(row["ToNode"]) and pd.notna(row["ToStation"]):
            node_to_station[row["ToNode"]] = row["ToStation"]

    # Add FromStation and ToStation for new_dev links based on FromNode and ToNode
    new_links_updated["FromStation"] = new_links_updated["FromNode"].map(node_to_station)
    new_links_updated["ToStation"] = new_links_updated["ToNode"].map(node_to_station)

    # Use Rail_Node.csv to fill remaining missing FromStation and ToStation
    new_links_updated["FromStation"] = new_links_updated.apply(
        lambda row: rail_node_mapping.get(row["FromNode"], f"Unknown_Station_{int(row['FromNode'])}") 
        if pd.isna(row["FromStation"]) and pd.notna(row["FromNode"]) else row["FromStation"], axis=1
    )
    new_links_updated["ToStation"] = new_links_updated.apply(
        lambda row: rail_node_mapping.get(row["ToNode"], f"Unknown_Station_{int(row['ToNode'])}") 
        if pd.isna(row["ToStation"]) and pd.notna(row["ToNode"]) else row["ToStation"], axis=1
    )

    # Duplicate new links: one with Direction A, one with Direction B
    new_links_with_direction = []
    for idx, row in new_links_updated.iterrows():
        # Row with Direction A
        row_a = row.copy()
        row_a["Direction"] = "A"

        # Row with Direction B (swapping FromNode and ToNode)
        row_b = row.copy()
        row_b["Direction"] = "B"
        row_b["FromNode"], row_b["ToNode"] = row_b["ToNode"], row_b["FromNode"]
        row_b["FromStation"], row_b["ToStation"] = row_b["ToStation"], row_b["FromStation"]

        # Append to the list
        new_links_with_direction.extend([row_a, row_b])

    # Convert the list of rows back into a GeoDataFrame
    new_links_updated_with_direction = gpd.GeoDataFrame(new_links_with_direction, geometry="geometry")

    # Combine the original network with the updated new links
    combined_gdf = pd.concat([network_railway_service, new_links_updated_with_direction], ignore_index=True)

    # Save the combined GeoDataFrame
    combined_gdf.to_file(output_path, driver="GPKG")
    return combined_gdf


def GetCatchmentOD():

    # Define spatial limits of the research corridor
    # The coordinates must end with 000 in order to match the coordinates of the input raster data
    e_min, e_max = 2687000, 2708000     # 2688000, 2704000 - 2688000, 2705000
    n_min, n_max = 1237000, 1254000     # 1238000, 1252000 - 1237000, 1252000
    limits_corridor = [e_min, n_min, e_max, n_max]
    # Get a polygon as limits for teh corridor
    innerboundary = polygon_from_points(e_min=e_min, e_max=e_max, n_min=n_min, n_max=n_max)


    # Import the required data or define the path to access it
    catchement_tif_path = r'data/catchment_pt/catchement.tif'
    catchmentdf = gpd.read_file(r"data/catchment_pt/catchement.gpkg")

    # File paths for population and employment combined raster files
    pop_combined_file = r"data/independent_variable/processed/scenario/pop_combined.tif"
    empl_combined_file = r"data/independent_variable/processed/scenario/empl_combined.tif"

    correct_rasters_to_extent(pop_combined_file,
        empl_combined_file,
        output_empl_path="data/independent_variable/processed/scenario/empl20_corrected.tif",
        output_pop_path="data/independent_variable/processed/scenario/pop20_corrected.tif",
        reference_boundary=innerboundary,
        resolution=100,
        crs="EPSG:2056")
    
    pop_combined_file = r"data/independent_variable/processed/scenario/empl20_corrected.tif"
    empl_combined_file = r"data/independent_variable/processed/scenario/pop20_corrected.tif"

    # define dev (=ID of the polygons of a development)
    dev = 0

    # Get voronoidf crs
    print(catchmentdf.crs)

    # todo When we iterate over devs and scens, maybe we can check if the VoronoiDF already has the communal data and then skip the following five lines
    popvec = GetCommunePopulation(y0="2021")
    jobvec = GetCommuneEmployment(y0=2021)
    od = GetHighwayPHDemandPerCommune() ## check tau values for PT
    odmat = GetODMatrix(od)

    # This function returns a np array of raster data storing the bfs number of the commune in each cell
    commune_raster, commune_df = GetCommuneShapes(raster_path=catchement_tif_path)

    if jobvec.shape[0] != odmat.shape[0]:
        print(
            "Error: The number of communes in the OD matrix and the number of communes in the employment data do not match.")
    # com_idx = np.unique(od['quelle_code']) # previously od_mat
    # 1. Define a new raster file that stores the Commune's BFS ID as cell value
    # Think if new band or new tif makes more sense
    # using communeShapes

    # I guess here iterate over all developments
    # voronoidf = voronoidf.loc[(voronoidf['ID_develop'] == dev)] # Work with temp gdf of voronoi
    # If possible simplify all the amount of developments

    # Define scenario names for population and employment
    pop_scenarios = [
        "pop_urban_", "pop_equal_", "pop_rural_",
        "pop_urba_1", "pop_equa_1", "pop_rura_1",
        "pop_urba_2", "pop_equa_2", "pop_rura_2"]

    empl_scenarios = [
        "empl_urban", "empl_equal", "empl_rural",
        "empl_urb_1", "empl_equ_1", "empl_rur_1",
        "empl_urb_2", "empl_equ_2", "empl_rur_2"]

    # Create dictionaries to store the raster data for each scenario
    pop_raster_data = {}
    empl_raster_data = {}

    # Read population scenarios from the combined raster file
    with rasterio.open(pop_combined_file) as src:
        for idx, scenario in enumerate(pop_scenarios, start=1):  # Start from band 1
            pop_raster_data[scenario] = src.read(idx)  # Read each band

    # Read employment scenarios from the combined raster file
    with rasterio.open(empl_combined_file) as src:
        for idx, scenario in enumerate(empl_scenarios, start=1):  # Start from band 1
            empl_raster_data[scenario] = src.read(idx)  # Read each band

    correct_rasters_to_extent(r"data/independent_variable/processed/raw/empl20.tif",
                              r"data/independent_variable/processed/raw/pop20.tif",
        output_empl_path="data/independent_variable/processed/raw/empl20_corrected.tif",
        output_pop_path="data/independent_variable/processed/raw/pop20_corrected.tif",
        reference_boundary=innerboundary,
        resolution=100,
        crs="EPSG:2056")
    
    # Open status quo
    with rasterio.open(r"data/independent_variable/processed/raw/empl20_corrected.tif") as src:
        scen_empl_20_tif = src.read(1)

    with rasterio.open(r"data/independent_variable/processed/raw/pop20_corrected.tif") as src:
        scen_pop_20_tif = src.read(1)

    # Load the catchment raster data
    with rasterio.open(catchement_tif_path) as src:
        # Read the raster data
        catchment_tif = src.read(2)  # Read the second band, which holds id information
        bounds = src.bounds  # Get the spatial bounds of the raster
        catchment_transform = src.transform  # Get the affine transform for spatial reference

    # Identify unique catchment IDs
    unique_catchment_id = np.sort(np.unique(catchment_tif))
    catch_idx = unique_catchment_id.size  # Total number of unique catchments

    # Filter commune_df based on catchment raster bounds
    commune_df_filtered = commune_df.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]

    # Extract "BFS" values (unique commune IDs) within the bounds
    commune_df_filtered = commune_df_filtered["BFS"].to_numpy()

    # Ensure the OD matrix corresponds only to filtered communes
    odmat_frame = odmat.loc[commune_df_filtered, commune_df_filtered]

    # Initialize an OD matrix for catchments
    # Shape is [number of unique catchments, number of unique catchments]
    od_mn = np.zeros([catch_idx, catch_idx])


    # Assume vectorized functions are defined for the below operations
    def compute_cont_r(odmat, popvec, jobvec):
        # Convert popvec and jobvec to 2D arrays for broadcasting
        pop_matrix = np.array(popvec)[:, np.newaxis]
        job_matrix = np.array(jobvec)[np.newaxis, :]

        # Ensure odmat is a NumPy array
        odmat = np.array(odmat)

        # Perform the vectorized operation
        cont_r = odmat / (pop_matrix * job_matrix)
        return cont_r

    def compute_cont_v(cont_r, pop_m, job_n):
        # Sum over the cont_r matrix, multiply by pop_m and job_n
        cont_v = np.sum(cont_r)
        return cont_v

    ###############################################################################################################################
    # Step 1: generate unit_flow matrix from each commune to each other commune
    cout_r = odmat / np.outer(popvec, jobvec)
    ###############################################################################################################################
    # Step 2: Get all pairs of combinations from communes to polygons
    unique_commune_id = np.sort(np.unique(commune_raster))
    pairs = pd.DataFrame(columns=['commune_id', 'catchement_id'])
    pop_empl = pd.DataFrame(columns=['commune_id', 'catchement_id', "empl", "pop"])

    # Initialize the DataFrame for storing results
    pop_empl = gpd.GeoDataFrame()
    pairs = gpd.GeoDataFrame()

    # Process each unique catchment and commune
    for i in tqdm(unique_catchment_id, desc='Processing Catchment IDs'):
        # Mask for the current catchment
        mask_catchment = catchment_tif == i

        for j in unique_commune_id:
            if j > 0:
                # Mask for the current commune
                mask_commune = commune_raster == j

                # Combined mask to find overlap
                mask = mask_commune & mask_catchment

                if np.nansum(mask) > 0:
                    # Record the commune and catchment pair
                    temp = pd.Series({'commune_id': j, 'catchment_id': i})
                    pairs = gpd.GeoDataFrame(
                        pd.concat([pairs, pd.DataFrame(temp).T], ignore_index=True)
                    )

                    # Extract population and employment data for all scenarios
                    temp_dict = {'commune_id': j, 'catchment_id': i}

                    # Status quo
                    temp_dict['pop_20'] = np.nansum(scen_pop_20_tif[mask])
                    temp_dict['empl_20'] = np.nansum(scen_empl_20_tif[mask])

                    # Loop through each scenario for population and employment
                    for scenario in pop_scenarios:
                        temp_dict[f'{scenario}'] = np.nansum(pop_raster_data[scenario][mask])

                    for scenario in empl_scenarios:
                        temp_dict[f'{scenario}'] = np.nansum(empl_raster_data[scenario][mask])

                    # Append the result for the current pair
                    temp = pd.Series(temp_dict)
                    pop_empl = gpd.GeoDataFrame(
                        pd.concat([pop_empl, pd.DataFrame(temp).T], ignore_index=True)
                    )


    ###############################################################################################################################
    # Step 3 complete exploded matrix
    # Initialize the OD matrix DataFrame with zeros or NaNs
    tuples = list(zip(pairs['catchment_id'], pairs['commune_id']))
    multi_index = pd.MultiIndex.from_tuples(tuples, names=['catchment_id', 'commune_id'])
    temp_df = pd.DataFrame(index=multi_index, columns=multi_index).fillna(0).to_numpy('float')
    od_matrix = pd.DataFrame(data=temp_df, index=multi_index, columns=multi_index)

    # Handle raster without values
    # Drop pairs with 0 pop or empl

    set_id_destination = [col[1] for col in od_matrix.columns]

    # Get unique values from the second level of the index
    unique_values_second_index = od_matrix.index.get_level_values(1).unique()

    # Iterate over each cell in the od_matrix to fill it with corresponding values from other_matrix
    for commune_id_origin in unique_values_second_index:
        # for (polygon_id_o, commune_id_o), _ in tqdm(od_matrix.index.to_series().iteritems(), desc='Allocating unit_values to OD matrix'):

        # Extract the row for commune_id_o
        row_values = cout_r.loc[commune_id_origin]

        # Use the valid columns to extract values
        extracted_values = row_values[set_id_destination].to_numpy('float')

        # Create a boolean mask for rows where the second element of the index matches commune_id_o
        mask = od_matrix.index.get_level_values(1) == commune_id_origin

        # Update the rows in od_matrix where the mask is True
        od_matrix.loc[mask] = extracted_values  # .to_numpy('float')

    ####################################################################################################3
    # todo Filling happens here

    # Check for scenario based on column names in pop_empl
    # Sceanrio are defined like pop_XX and empl_XX get a list of all these endings (only XX)
    # Get the column names of pop_empl
    pop_empl_columns = pop_empl.columns
    # Get the column names that end with XX
    '''
    pop_empl_scenarios = [col.split("_")[1] for col in pop_empl_columns if col.startswith("pop_")]
    print(pop_empl_scenarios)

    # SEt index of df to access its single components
    pop_empl = pop_empl.set_index(['catchment_id', 'commune_id'])

    # Extract unique scenario identifiers by removing the prefix ('pop_' or 'empl_') and trailing underscores
    pop_empl_scenarios = sorted(set(col.split('_', 1)[1].rstrip('_') for col in pop_empl.columns if col.startswith(('pop_', 'empl_'))))
    pop_empl = pop_empl[sorted(pop_empl.columns)]
    '''
    # SEt index of df to access its single components
    pop_empl = pop_empl.set_index(['catchment_id', 'commune_id'])
    pop_empl_scenarios = pop_empl.columns.tolist()

    # for each of these scenarios make an own copy of od_matrix named od_matrix+scen
    for scen in pop_empl_scenarios:
        print(f"Processing scenario {scen}")
        od_matrix_temp = od_matrix.copy()


        for polygon_id, row in tqdm(pop_empl.iterrows(), desc='Allocating pop and empl to OD matrix'):
            # Multiply all values in the row/column
            od_matrix_temp.loc[polygon_id] *= row[f'{scen}']
            od_matrix_temp.loc[:, polygon_id] *= row[f'{scen}']

        ###############################################################################################################################
        # Step 4: Group the OD matrix by polygon_id
        # Reset the index to turn the MultiIndex into columns
        od_matrix_reset = od_matrix_temp.reset_index()

        # Sum the values by 'polygon_id' for both the rows and columns
        od_grouped = od_matrix_reset.groupby('catchment_id').sum()

        # Now od_grouped has 'polygon_id' as the index, but we still need to group the columns
        # First, transpose the DataFrame to apply the same operation on the columns
        od_grouped = od_grouped.T

        # Again group by 'polygon_id' and sum, then transpose back
        od_grouped = od_grouped.groupby('catchment_id').sum().T

        # Drop column commune_id
        od_grouped = od_grouped.drop(columns='commune_id')

        # Set diagonal values to 0
        temp_sum = od_grouped.sum().sum()
        np.fill_diagonal(od_grouped.values, 0)
        # Compute the sum after changing the diagonal
        temp_sum2 = od_grouped.sum().sum()
        # Print difference
        print(f"Sum of OD matrix before {temp_sum} and after {temp_sum2} removing diagonal values")

        # Save pd df to csv
        od_grouped.to_csv(fr"data/traffic_flow/od/rail/od_matrix_{scen}.csv")
        # odmat.to_csv(r"data/traffic_flow/od/od_matrix_raw.csv")

        # Print sum of all values in od df
        # Sum over all values in pd df
        sum_com = odmat.sum().sum()
        sum_poly = od_grouped.sum().sum()
        sum_com_frame = odmat_frame.sum().sum()
        print(
            f"Total trips before {sum_com_frame} ({odmat_frame.shape} communes) and after {sum_poly} ({od_grouped.shape} polygons)")
        print(
            f"Total trips before {sum_com} ({odmat.shape} communes) and after {sum_poly} ({od_grouped.shape} polygons)")

        # Sum all columns of od_grouped
        origin = od_grouped.sum(axis=1).reset_index()
        origin.colum = ["catchment_id", "origin"]
        # Sum all rows of od_grouped
        destination = od_grouped.sum(axis=0)
        destination = destination.reset_index()

        # merge origin and destination to catchmentdf based on catchment_id
        # Make a copy of catchmentdf
        catchmentdf_temp = catchmentdf.copy()
        catchmentdf_temp.rename(columns={'id': 'ID_point'}, inplace=True)
        catchmentdf_temp = catchmentdf_temp.merge(origin, how='left', left_on='ID_point', right_on='catchment_id')
        catchmentdf_temp = catchmentdf_temp.merge(destination, how='left', left_on='ID_point', right_on='catchment_id')
        catchmentdf_temp = catchmentdf_temp.rename(columns={'0_x': 'origin', '0_y': 'destination'})
        catchmentdf_temp.to_file(fr"data/traffic_flow/od/catchment_id_{scen}.gpkg", driver="GPKG")

        # Same for odmat and commune_df
        if scen == "20":
            origin_commune = odmat_frame.sum(axis=1).reset_index()
            origin_commune.colum = ["commune_id", "origin"]
            destination_commune = odmat_frame.sum(axis=0).reset_index()
            destination_commune.colum = ["commune_id", "destination"]
            commune_df = commune_df.merge(origin_commune, how='left', left_on='BFS', right_on='quelle_code')
            commune_df = commune_df.merge(destination_commune, how='left', left_on='BFS', right_on='ziel_code')
            commune_df = commune_df.rename(columns={'0_x': 'origin', '0_y': 'destination'})
            commune_df.to_file(r"data/traffic_flow/od/OD_commune_filtered.gpkg", driver="GPKG")
    return

def move_original_files():
    """
    Moves all files containing "original" in their name from the source directory
    to the target directory.

    Parameters:
        source_directory (str): The directory to search for "original" files.
        target_directory (str): The directory to move "original" files to.
    """

    source_directory = "data/traffic_flow/od"
    target_directory = "data/traffic_flow/od/original"
    # Ensure the target directory exists
    os.makedirs(target_directory, exist_ok=True)

    # Iterate through files in the source directory
    for file_name in os.listdir(source_directory):
        if "original" in file_name:
            source_path = os.path.join(source_directory, file_name)
            target_path = os.path.join(target_directory, file_name)

            # Move the file
            shutil.move(source_path, target_path)
            print(f"Moved: {source_path} -> {target_path}")


def calculate_total_travel_times(od_times_list, traffic_flow_dir, df_access):
    """
    Calculate total travel times for each development and scenario.
    
    Parameters:
        od_times_list (list): List of DataFrames with OD travel times for each scenario.
        traffic_flow_dir (str): Directory containing CSV files with traffic flow data for each development.
        df_access (pd.DataFrame): Rail node DataFrame for mapping IDs to station names.
    
    Returns:
        list: A list of lists, where each sub-list corresponds to total travel times for each scenario.
    """
    # Mapping of IDs to station names
    id_to_name = df_access.set_index("NR")["NAME"].to_dict()
    
    # Filter traffic flow files and prepare for processing
    traffic_flow_files = [file for file in os.listdir(traffic_flow_dir) if file.endswith('.csv')]
    total_travel_times = []  # This will store the results for all developments
    
    for dev_file in traffic_flow_files:
        dev_total_times = []  # Store results for this development
        
        # Load traffic flow matrix
        traffic_flow_df = pd.read_csv(os.path.join(traffic_flow_dir, dev_file), index_col=0)
        traffic_flow_df.index = traffic_flow_df.index.astype(int)
        traffic_flow_df.columns = traffic_flow_df.columns.astype(float).astype(int)
        
        # Map station names to IDs in traffic flow matrix
        traffic_flow_df = traffic_flow_df.rename(index=id_to_name, columns=id_to_name)
        
        # Remove rows and columns with -1.0 (outside catchment area)
        if -1 in traffic_flow_df.index:
            traffic_flow_df = traffic_flow_df.drop(index=-1, errors='ignore')
        if -1 in traffic_flow_df.columns:
            traffic_flow_df = traffic_flow_df.drop(columns=-1, errors='ignore')
        
        # Process each scenario in od_times_list
        for scenario_od_df in od_times_list:
            # Merge OD times with traffic flows
            scenario_od_df["from_name"] = scenario_od_df["from_id"].str.replace("main_", "")
            scenario_od_df["to_name"] = scenario_od_df["to_id"].str.replace("main_", "")
            
            # Merge trips from traffic flow matrix
            scenario_od_df["trips"] = scenario_od_df.apply(
                lambda row: traffic_flow_df.at[row["from_name"], row["to_name"]]
                if row["from_name"] in traffic_flow_df.index and row["to_name"] in traffic_flow_df.columns
                else 0,
                axis=1
            )
            
            # Calculate weighted travel times (trips * time)
            scenario_od_df["weighted_time"] = scenario_od_df["trips"] * scenario_od_df["time"]
            
            # Sum total weighted time for this scenario
            total_time = scenario_od_df["weighted_time"].sum()
            dev_total_times.append(total_time)
        
        total_travel_times.append(dev_total_times)
    
    return total_travel_times


def aggregate_costs():
    """
    Aggregate and calculate total costs for each development and scenario.

    Parameters:
        construction_path (str): Path to the construction costs GeoPackage.
        maintenance_path (str): Path to the maintenance costs GeoPackage.
        access_time_path (str): Path to the local accessibility CSV.
        travel_time_path (str): Path to the travel time savings CSV.
        externalities_path (str): Path to the externalities GeoPackage.
        noise_path (str): Path to the noise costs GeoPackage.
        nodes_path (str): Path to the generated nodes GeoPackage.
        total_costs_csv_path (str): Path to save the total costs CSV.
        total_costs_gpkg_path (str): Path to save the total costs GeoPackage.
    """

    #construction_path="data/costs/construction.gpkg",
    #maintenance_path="data/costs/maintenance.gpkg",
    #access_time_path="data/costs/local_accessibility.csv",
    travel_time_path="data/costs/traveltime_savings.csv",
    #externalities_path="data/costs/externalities.gpkg",
    #noise_path="data/costs/noise.gpkg",
    #nodes_path="data/Network/processed/generated_nodes.gpkg",
    total_costs_csv_path="data/costs/total_costs.csv",
    total_costs_gpkg_path="data/costs/total_costs.gpkg

    # Define scenarios for population and employment
    pop_scenarios = [
        "pop_urban_", "pop_equal_", "pop_rural_",
        "pop_urba_1", "pop_equa_1", "pop_rura_1",
        "pop_urba_2", "pop_equa_2", "pop_rura_2"]

    empl_scenarios = [
        "empl_urban", "empl_equal", "empl_rural",
        "empl_urb_1", "empl_equ_1", "empl_rur_1",
        "empl_urb_2", "empl_equ_2", "empl_rur_2"]

    # Load other cost components
    #c_construction = gpd.read_file(construction_path)
    #c_maintenance = gpd.read_file(maintenance_path)
    #c_access_time = pd.read_csv(access_time_path)
    #c_travel_time = pd.read_csv(travel_time_path)
    #c_externalities = gpd.read_file(externalities_path)
    #c_noise = gpd.read_file(noise_path)

    # Rename columns for consistency
    #c_access_time = c_access_time.rename(columns={'ID_develop': 'ID_new'})
    c_travel_time = c_travel_time.rename(columns={'development': 'ID_new'})

    # Merge data components
    total_costs = c_construction.drop("geometry", axis=1).merge(
        c_maintenance.drop("geometry", axis=1), how='inner', on='ID_new')
    #total_costs = total_costs.merge(c_access_time, how='inner', on='ID_new')
    total_costs = total_costs.merge(c_travel_time, how='inner', on='ID_new')
    #total_costs = total_costs.merge(c_externalities.drop("geometry", axis=1), how='inner', on='ID_new')
    #total_costs = total_costs.merge(c_noise.drop("geometry", axis=1), how='inner', on='ID_new')

    # Multiply cost columns by -1
    cost_columns = ['cost_path', 'cost_bridge', 'cost_tunnel', 'building_costs', 'climate_cost', 'land_realloc',
                    'nature', 'noise_s1', 'noise_s2', 'noise_s3', "maintenance"]
    for column in cost_columns:
        if column in total_costs.columns:
            total_costs[column] = total_costs[column] * -1

    # Dynamically compute costs for each scenario
    for pop_scenario, empl_scenario in zip(pop_scenarios, empl_scenarios):
        total_costs[f"total_{pop_scenario}"] = (
            total_costs["building_costs"] +
            total_costs["maintenance"] +
            total_costs.get(f"local_{pop_scenario}", 0) +
            total_costs.get(f"tt_{pop_scenario}", 0) +
            total_costs.get(f"externalities_{pop_scenario}", 0)
        )

    # Save results to CSV
    total_costs.to_csv(total_costs_csv_path, index=False)

    # Save results as GeoPackage
    points = gpd.read_file(nodes_path)
    total_costs = total_costs.merge(right=points, how="left", on="ID_new")
    total_costs = gpd.GeoDataFrame(total_costs, geometry="geometry")
    total_costs.to_file(total_costs_gpkg_path, driver="GPKG")

    print(f"Total costs saved to {total_costs_csv_path} and {total_costs_gpkg_path}")



def construction_costs(highway, tunnel, bridge, ramp):
    """
    highway = 11000 # CHF / m
    tunnel = 300000 # CHF / m
    bridge = 2600 * 22 # CHF / m
    ramp = 100000000 # CHF
    """

    bridge_small_river = 0  # m
    bridge_medium_river = 25  # m
    bridge_big_river = 50  # m
    bridge_rail = 25  # m

    # generated_links_gdf = gpd.read_file(r"data/Network/processed/new_links.shp")
    # generated_links_gdf = gpd.read_file(r"data/Network/processed/new_links_realistic.gpkg")
    generated_links_gdf = gpd.read_file(r"data/Network/processed/new_links_realistic_tunnel_adjusted.gpkg")
    # generated_links_gdf = gpd.read_file(r"data/Network/processed/new_links_realistic_tunnel.gpkg")

    # Aggreagte by development over all tunnels and bridges
    generated_links_gdf = generated_links_gdf.fillna(0)
    generated_links_gdf = generated_links_gdf.groupby(by="ID_new").agg(
        {"ID_current": "first", "total_tunnel_length": "sum", "total_bridge_length": "sum", "geometry": "first"})
    # Convert the index into a column
    generated_links_gdf = generated_links_gdf.reset_index()
    # Convert the DataFrame back to a GeoDataFrame
    generated_links_gdf = gpd.GeoDataFrame(generated_links_gdf, geometry='geometry', crs="epsg:2056")

    # Costs due to bridges to cross water
    generated_links_gdf = bridges_crossing_water(generated_links_gdf)

    # Costs due to bridges to cross railways
    generated_links_gdf = rail_crossing(generated_links_gdf)

    # Replace nan values by 0
    generated_links_gdf = generated_links_gdf.fillna(0)

    generated_links_gdf["bridge"] = generated_links_gdf["count_rail"] * bridge_rail + generated_links_gdf[
        "klein"] * bridge_small_river + generated_links_gdf["mittel"] * bridge_medium_river + generated_links_gdf[
                                        "gross"] * bridge_big_river

    # Sum amount of tunnel and bridges
    generated_links_gdf["bridge_len"] = generated_links_gdf["total_bridge_length"] + generated_links_gdf["bridge"]
    generated_links_gdf["tunnel_len"] = generated_links_gdf["total_tunnel_length"]
    generated_links_gdf["hw_len"] = generated_links_gdf.geometry.length - generated_links_gdf["bridge_len"] - \
                                    generated_links_gdf["tunnel_len"]

    # Drop unseless columns
    generated_links_gdf = generated_links_gdf.drop(
        columns=["gross", "klein", "mittel", "count_rail", "bridge", "total_bridge_length", "total_bridge_length"])
    generated_links_gdf.to_file(r"data/Network/processed/links_with_geometry_attributes.gpkg")

    generated_links_gdf["cost_path"] = generated_links_gdf["hw_len"] * highway
    generated_links_gdf["cost_bridge"] = generated_links_gdf["bridge_len"] * bridge
    generated_links_gdf["cost_tunnel"] = generated_links_gdf["tunnel_len"] * tunnel
    generated_links_gdf["building_costs"] = generated_links_gdf["cost_path"] + generated_links_gdf["cost_bridge"] + \
                                            generated_links_gdf["cost_tunnel"] + ramp

    # Only keep relevant columns
    generated_links_gdf = generated_links_gdf[
        ["ID_current", "ID_new", "geometry", "cost_path", "cost_bridge", "cost_tunnel", "building_costs"]]
    generated_links_gdf.to_file(r"data/costs/construction.gpkg")

    return




def construction_costs(file_path):
    """
    Process the rail network data to calculate construction costs and services per edge.
    Includes checks for capacity adequacy.

    Parameters:
        file_path (str): Path to the CSV file containing the rail network data.

    Returns:
        pd.DataFrame: Processed DataFrame with expanded edges and calculated service metrics.
    """
    import pandas as pd

    try:
        # Load the data
        df_construction_cost = pd.read_csv(file_path, sep=";", decimal=",", encoding="utf-8-sig")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the file: {e}")

    # Split the lines with a Via column
    df_split = split_via_nodes(df_construction_cost)

    # Group by unique edges and count distinct services
    services_per_edge = (
        df_split.groupby(['FromNode', 'ToNode'])['Service']
        .nunique()
        .reset_index(name='NumServices')
    )
    
    # Merge the result back to the split DataFrame
    df_split = pd.merge(df_split, services_per_edge, on=['FromNode', 'ToNode'], how='left')

    # Calculate NumServicesPerH as the product of NumServices and Frequency
    df_split['NumServicesPerH'] = df_split['NumServices'] * df_split['Frequency']

    # Calculate MinTrack as the smallest digit from the NumOfTracks column
    df_split['MinTrack'] = df_split['NumOfTracks'].astype(str).str[0].astype(int)

    # Calculate ServicesPerTrack
    df_split['ServicesPerTrack'] = df_split['NumServicesPerH'] / df_split['MinTrack']

    # Add a new column 'enoughCap' based on ServicesPerTrack < 8
    df_split['enoughCap'] = df_split['ServicesPerTrack'].apply(lambda x: 'Yes' if x < 8 else 'No')
    
    return df_split


def merge_lines(df):
    """
    Merge lines with the same combination of start and stop nodes (ignoring direction).
    Aggregate specific columns while summing up 'Frequency', and summing up
    'TravelTime' and 'InVehWait' to calculate 'TotalTravelTime'.

    Parameters:
        df (pd.DataFrame): Input DataFrame with columns including 'FromNode', 'ToNode', 'Frequency', 'TravelTime', and 'InVehWait'.

    Returns:
        pd.DataFrame: Merged DataFrame with aggregated information.
    """

    # Create a directionless edge column to group by
    df['Edge'] = df.apply(
        lambda row: tuple(sorted([row['FromNode'], row['ToNode']])), axis=1
    )

    # Columns to aggregate with identical values for the combined lines
    common_columns = [
        'FromNode', 'ToNode', 'FromStation', 'ToStation', 'NumOfTracks',
        'Bridges m', 'Tunnel m', 'TunnelTrack', 'tot length m', 'length of 1',
        'length of 2 ', 'length of 3 and more'
    ]

    # Function to extract common values for a group
    def extract_common_values(group):
        row = group.iloc[0]  # Take the first row as representative
        return {col: row[col] for col in common_columns}

    # Group by the directionless edge
    grouped = df.groupby('Edge')

    # Aggregate the data
    merged_data = []
    for _, group in grouped:
        common_data = extract_common_values(group)  # Common columns
        total_frequency = group['Frequency'].sum()  # Sum of 'Frequency'
        total_travel_time = group['TravelTime'].sum() + group['InVehWait'].sum()  # Sum TravelTime and InVehWait
        merged_row = {
            **common_data,
            'TotalFrequency': total_frequency,
            'TotalTravelTime': total_travel_time
        }
        merged_data.append(merged_row)

    # Convert the list of merged rows to a new DataFrame
    merged_df = pd.DataFrame(merged_data)

    return merged_df




def update_network_with_new_links(network_railway_service_path, new_links_updated_path):
    """
    Add new links to the railway network, marking them as new and generating both directions.
    Ensure FromStation and ToStation are mapped correctly using Rail_Node data.
    """
    # Load data
    network_railway_service = gpd.read_file(network_railway_service_path)
    new_links_updated = gpd.read_file(new_links_updated_path)
    rail_node = pd.read_csv(r"data/Network/Rail_Node.csv", sep=";", decimal=",", encoding="ISO-8859-1")

    # Ensure Rail_Node has required columns
    if not {"NR", "NAME"}.issubset(rail_node.columns):
        raise ValueError("Rail_Node file must contain 'NR' and 'NAME' columns.")
    
    # Map NR to NAME for station names
    rail_node_mapping = rail_node.set_index("NR")["NAME"].to_dict()

    # Populate required columns for new links
    new_links_updated = new_links_updated.assign(
        new_dev="Yes",
        FromNode=new_links_updated["from_ID_new"],
        ToNode=new_links_updated["to_ID"],
        FromStation=new_links_updated["from_ID_new"].map(rail_node_mapping),
        ToStation=new_links_updated["to_ID"].map(rail_node_mapping),
        Direction="A",  # Default direction
    )

    # Ensure `new_dev` in the original network remains unchanged
    network_railway_service["new_dev"] = network_railway_service.get("new_dev", "No")

    # Assign additional columns directly
    new_links_updated["TravelTime"] = new_links_updated["time"]
    new_links_updated["InVehWait"] = 0
    new_links_updated["Service"] = new_links_updated["Sline"]
    new_links_updated["Frequency"] = 2
    new_links_updated["TotalPeakCapacity"] = 690
    new_links_updated["Capacity"] = 345

    # Calculate the Via nodes for all the new connections
    via_df = get_via(new_links_updated)

    # Merge the 'via_nodes' from 'via_df' into 'new_links_updated' based on 'from_ID_new' and 'to_ID'
    new_links_updated = pd.merge(
        new_links_updated, 
        via_df[['from_ID_new', 'to_ID', 'via_nodes']], 
        left_on=['from_ID_new', 'to_ID'], 
        right_on=['from_ID_new', 'to_ID'], 
        how='left'
    )

    # Rename the 'via_nodes' column to 'Via' for clarity
    new_links_updated.rename(columns={'via_nodes': 'Via'}, inplace=True)

    # Ensure all Via values are strings or -99 for empty paths
    new_links_updated['Via'] = new_links_updated['Via'].apply(
        lambda x: '-99' if not x or x == [-99] else ','.join(map(str, x))
    )

    # Identify and report missing node mappings
    missing_from_nodes = new_links_updated["FromNode"][new_links_updated["FromStation"].isna()].unique()
    missing_to_nodes = new_links_updated["ToNode"][new_links_updated["ToStation"].isna()].unique()

    if len(missing_from_nodes) > 0 or len(missing_to_nodes) > 0:
        print("Warning: Missing mappings for the following nodes:")
        if len(missing_from_nodes) > 0:
            print(f"FromNodes: {missing_from_nodes}")
        if len(missing_to_nodes) > 0:
            print(f"ToNodes: {missing_to_nodes}")

    # Generate rows for Direction B while preserving dev_id
    direction_b = new_links_updated.copy()
    direction_b["Direction"] = "B"
    direction_b["FromNode"], direction_b["ToNode"] = direction_b["ToNode"], direction_b["FromNode"]
    direction_b["FromStation"], direction_b["ToStation"] = direction_b["ToStation"], direction_b["FromStation"]

    # Combine A and B directions, preserving the same dev_id
    combined_new_links = pd.concat([new_links_updated, direction_b], ignore_index=True)

    # Ensure GeoDataFrame compatibility
    combined_new_links_gdf = gpd.GeoDataFrame(combined_new_links, geometry=new_links_updated.geometry)

    # Combine with original network
    combined_network = pd.concat([network_railway_service, combined_new_links_gdf], ignore_index=True)

    return combined_network

def get_via(new_connections):
    """
    Calculate the list of nodes traversed for each new connection based on the existing connections.

    Parameters:
        new_connections (pd.DataFrame): New connections with columns 'from_ID_new' and 'to_ID'.

    Returns:
        pd.DataFrame: A DataFrame with the new connections and a list of nodes traversed for each connection,
                      represented as a string or an integer (-99 if no path exists).
    """
    # File path for the construction cost data
    file_path = r"data/Network/Rail-Service_Link_construction_cost.csv"

    try:
        # Load the data
        df_construction_cost = pd.read_csv(file_path, sep=";", decimal=",", encoding="utf-8-sig")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the file: {e}")

    # Create an undirected graph
    G = nx.Graph()

    # Split the lines with a Via column
    df_split = split_via_nodes(df_construction_cost)
    df_split = merge_lines(df_split)

    # Add edges to the graph
    for _, row in df_split.iterrows():
        G.add_edge(row['FromNode'], row['ToNode'], weight=row['TotalTravelTime'])

    # Ensure nodes and connections IDs are integers
    G = nx.relabel_nodes(G, {n: int(n) for n in G.nodes})
    new_connections['from_ID_new'] = new_connections['from_ID_new'].astype(int)
    new_connections['to_ID'] = new_connections['to_ID'].astype(int)

    # Compute the routes
    results = []
    for _, row in new_connections.iterrows():
        from_node = row['from_ID_new']
        to_node = row['to_ID']

        # Find the shortest path based on TravelTime
        try:
            path = nx.shortest_path(G, source=from_node, target=to_node, weight='weight')
            # Convert path to a string
            path_str = ",".join(map(str, path))  # Convert list to comma-separated string
        except nx.NetworkXNoPath:
            path_str = -99  # No path exists

        # Add the result to the list
        results.append({
            'from_ID_new': from_node,
            'to_ID': to_node,
            'via_nodes': path_str  # Path as string or -99
        })

    # Convert results to a DataFrame
    result_df = pd.DataFrame(results)

    return result_df


def construction_costs(file_path, cost_per_meter, tunnel_cost_per_meter, bridge_cost_per_meter):
    """
    Process the rail network data to calculate construction costs and services per edge.
    Includes checks for capacity adequacy, considers both directions (A and B) using the same tracks,
    and calculates the cost of building additional tracks where capacity is insufficient.

    Parameters:
        file_path (str): Path to the CSV file containing the rail network data.
        cost_per_meter (float): Cost of building a new track per meter.
        tunnel_cost_per_meter (float): Cost of updating tunnels per meter per track.
        bridge_cost_per_meter (float): Cost of updating bridges per meter per track.

    Returns:
        pd.DataFrame: Processed DataFrame with expanded edges, calculated service metrics, and costs.
    """
    try:
        # Load the data
        df_construction_cost = pd.read_csv(file_path, sep=";", decimal=",", encoding="utf-8-sig")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the file: {e}")

    # Split the lines with a Via column
    df_split = split_via_nodes(df_construction_cost)
    df_split = merge_lines(df_split)

    # Calculate MinTrack as the smallest digit from the NumOfTracks column
    df_split['MinTrack'] = df_split['NumOfTracks'].apply(lambda x: int(min(str(x))))

    # Calculate ServicesPerTrack
    df_split['ServicesPerTrack'] = df_split['TotalFrequency'] / df_split['MinTrack']

    # Add a new column 'enoughCap' based on ServicesPerTrack < 8
    df_split['enoughCap'] = df_split['ServicesPerTrack'].apply(lambda x: 'Yes' if x < 8 else 'No')

    # Calculate costs for connections with insufficient capacity
    insufficient_capacity = df_split[df_split['enoughCap'] == 'No'].copy()

    # Initialize cost columns
    insufficient_capacity['NewTrackCost'] = insufficient_capacity['length of 1'] * cost_per_meter
    insufficient_capacity['NewTunnelCost'] = (
        insufficient_capacity['Tunnel m'] * (tunnel_cost_per_meter / insufficient_capacity['NumOfTracks'])
    )
    insufficient_capacity['NewBridgeCost'] = (
        insufficient_capacity['Bridges m'] * (bridge_cost_per_meter / insufficient_capacity['NumOfTracks'])
    )

    # Update TunnelTrack if needed
    insufficient_capacity['UpdatedTunnelTrack'] = insufficient_capacity['TunnelTrack'] + 1

    # Calculate total cost
    insufficient_capacity['TotalCost'] = (
        insufficient_capacity['NewTrackCost'] +
        insufficient_capacity['NewTunnelCost'] +
        insufficient_capacity['NewBridgeCost']
    )

    # Merge the calculated costs back into the original DataFrame
    df_split = pd.merge(
        df_split,
        insufficient_capacity[['FromNode', 'ToNode', 'NewTrackCost', 'NewTunnelCost', 'NewBridgeCost', 'TotalCost']],
        on=['FromNode', 'ToNode'],
        how='left'
    )

    return df_split


def aggregate_costs():
    """
    Aggregate and calculate total and average costs for each development, with one line per development.

    Parameters:
        construction_path (str): Path to the construction costs GeoPackage.
        maintenance_path (str): Path to the maintenance costs GeoPackage.
        access_time_path (str): Path to the local accessibility CSV.
        travel_time_path (str): Path to the travel time savings CSV.
        externalities_path (str): Path to the externalities GeoPackage.
        noise_path (str): Path to the noise costs GeoPackage.
        nodes_path (str): Path to the generated nodes GeoPackage.
        total_costs_csv_path (str): Path to save the total costs CSV.
        total_costs_gpkg_path (str): Path to save the total costs GeoPackage.
    """

    import pandas as pd
    import geopandas as gpd

    # Paths for components
    travel_time_path = "data/costs/traveltime_savings.csv"
    total_costs_csv_path = "data/costs/total_costs.csv"
    total_costs_gpkg_path = "data/costs/total_costs.gpkg"

    # Define scenarios for population and employment
    pop_scenarios = [
        "pop_urban_", "pop_equal_", "pop_rural_",
        "pop_urba_1", "pop_equa_1", "pop_rura_1",
        "pop_urba_2", "pop_equa_2", "pop_rura_2"]

    # Load travel time savings
    c_travel_time = pd.read_csv(travel_time_path)

    # Initialize total costs with zeros for missing components
    total_costs = c_travel_time.rename(columns={'development': 'ID_new'})
    total_costs["building_costs"] = 0
    total_costs["maintenance"] = 0
    total_costs["climate_cost"] = 0
    total_costs["land_realloc"] = 0
    total_costs["nature"] = 0
    total_costs["noise_s1"] = 0
    total_costs["noise_s2"] = 0
    total_costs["noise_s3"] = 0

    # Calculate costs for each scenario and add them as columns
    for pop_scenario in pop_scenarios:
        total_costs[f"total_{pop_scenario}"] = (
            total_costs["building_costs"] +
            total_costs["maintenance"] +
            total_costs.get(f"local_{pop_scenario}", 0) +
            total_costs.get(f"tt_{pop_scenario}", 0) +
            total_costs.get(f"externalities_{pop_scenario}", 0)
        )

    # Aggregate total and average costs per development
    scenario_columns = [f"total_{pop_scenario}" for pop_scenario in pop_scenarios]
    total_costs_aggregated = total_costs.groupby("ID_new")[scenario_columns].sum().reset_index()
    total_costs_aggregated["total_all_scenarios"] = total_costs_aggregated[scenario_columns].sum(axis=1)
    total_costs_aggregated["average_cost"] = total_costs_aggregated[scenario_columns].mean(axis=1)

    # Save results to CSV
    total_costs_aggregated.to_csv(total_costs_csv_path, index=False)

    # Save results as GeoPackage (geometry initialization for now set to None)
    total_costs_aggregated = gpd.GeoDataFrame(total_costs_aggregated, geometry=None)
    total_costs_aggregated.to_file(total_costs_gpkg_path, driver="GPKG")

    print(f"Total costs saved to {total_costs_csv_path} and {total_costs_gpkg_path}")


def construction_costs(file_path, developments, cost_per_meter, tunnel_cost_per_meter, bridge_cost_per_meter):
    """
    Process the rail network data to calculate construction costs and services per edge.
    Includes checks for capacity adequacy, considers both directions (A and B) using the same tracks,
    and calculates the cost of building additional tracks where capacity is insufficient for multiple developments.

    Parameters:
        file_path (str): Path to the CSV file containing the rail network data.
        developments (list of pd.DataFrame): List of DataFrames, each representing new connections for a development.
        cost_per_meter (float): Cost of building a new track per meter.
        tunnel_cost_per_meter (float): Cost of updating tunnels per meter per track.
        bridge_cost_per_meter (float): Cost of updating bridges per meter per track.

    Returns:
        pd.DataFrame: Processed DataFrame with expanded edges, calculated service metrics, and costs.
        pd.DataFrame: Summary DataFrame containing total costs for each development.
    """
    try:
        # Load the base construction cost data
        df_construction_cost = pd.read_csv(file_path, sep=";", decimal=",", encoding="utf-8-sig")
        #df_construction_cost = pd.read_csv(file_path, sep=";", decimal=",", encoding="ISO-8859-1")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the file: {e}")

    development_costs = []  # To store costs for each development
    developments = read_development_files('data/Network/processed/developments')
    developments = [process_via_column(df) for df in developments]

    
    for i, dev_df in enumerate(developments):
        # Add the development lines to the construction cost data
        combined_df = pd.concat([df_construction_cost, dev_df], ignore_index=True)

        # Split the lines with a Via column
        df_split = split_via_nodes(combined_df)
        df_split = merge_lines(df_split)
        df_split = df_split.dropna(subset=['NumOfTracks'])


        # Calculate MinTrack as the smallest digit from the NumOfTracks column
        # Convert 'NumOfTracks' to integer
        df_split['NumOfTracks'] = df_split['NumOfTracks'].astype(int)
        df_split['MinTrack'] = df_split['NumOfTracks'].apply(lambda x: int(min(str(x))))

        # Calculate ServicesPerTrack
        df_split['ServicesPerTrack'] = df_split['TotalFrequency'] / df_split['MinTrack']

        # Add a new column 'enoughCap' based on ServicesPerTrack < 8
        df_split['enoughCap'] = df_split['ServicesPerTrack'].apply(lambda x: 'Yes' if x < 8 else 'No')

        # Calculate costs for connections with insufficient capacity
        insufficient_capacity = df_split[df_split['enoughCap'] == 'No'].copy()

        # Initialize cost columns
        insufficient_capacity['NewTrackCost'] = insufficient_capacity['length of 1'] * cost_per_meter
        insufficient_capacity['NewTunnelCost'] = (
            insufficient_capacity['Tunnel m'] * (tunnel_cost_per_meter / insufficient_capacity['NumOfTracks'])
        )
        insufficient_capacity['NewBridgeCost'] = (
            insufficient_capacity['Bridges m'] * (bridge_cost_per_meter / insufficient_capacity['NumOfTracks'])
        )

        # Update TunnelTrack if needed
        insufficient_capacity['UpdatedTunnelTrack'] = insufficient_capacity['TunnelTrack'] + 1

        # Calculate total cost
        insufficient_capacity['TotalCost'] = (
            insufficient_capacity['NewTrackCost'] +
            insufficient_capacity['NewTunnelCost'] +
            insufficient_capacity['NewBridgeCost']
        )

        # Summarize total cost for the current development
        total_cost = insufficient_capacity['TotalCost'].sum()
        development_costs.append({"Development": f"Development_{i+1}", "TotalCost": total_cost})

        # Update the base construction cost data for the next iteration
        df_construction_cost = pd.concat([df_construction_cost, dev_df], ignore_index=True)

    # Create a summary DataFrame for development costs
    development_costs_df = pd.DataFrame(development_costs)
    pd.DataFrame(development_costs).to_csv("data/costs/construction_cost.csv", index=False)

    return development_costs_df



def calculate_monetized_tt_savings(TTT_status_quo, TTT_developments, VTTS, duration, output_path):
    """
    Calculate and monetize travel time savings for each development scenario compared to the status quo.

    Parameters:
        TTT_status_quo (dict): Dictionary of total travel times for the status quo.
        TTT_developments (dict): Dictionary of total travel times for each development scenario.
        VTTS (float): Value of Travel Time Savings (CHF/h).
        duration (float): Duration factor (e.g., years).
        output_path (str): Path to save the monetized travel time savings CSV.

    Returns:
        pd.DataFrame: DataFrame containing monetized travel time savings for each development and scenario.
    """
    import pandas as pd

    # Monetization factor of travel time (peak hour * CHF/h * 365 d/a * duration)
    mon_factor = VTTS * 365 * duration

    # Prepare a list to store the results
    results = []

    # Iterate over each development
    for dev_name, scenarios in TTT_developments.items():
        for scenario_name, dev_tt in scenarios.items():
            # Get the corresponding status quo travel time
            status_quo_tt = TTT_status_quo.get(dev_name, {}).get(scenario_name, 0)

            # Calculate travel time savings (negative if no savings)
            tt_savings = status_quo_tt - dev_tt

            # Monetize the travel time savings
            monetized_savings = tt_savings * mon_factor

            # Append the results
            results.append({
                "development": dev_name,
                "scenario": scenario_name,
                "status_quo_tt": status_quo_tt,
                "development_tt": dev_tt,
                "tt_savings": tt_savings,
                "monetized_savings": monetized_savings
            })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Monetized travel time savings saved to: {output_path}")

    return results_df



import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import contextily as ctx
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import os

# Define output directory
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
output_pdf_path = os.path.join(output_dir, "output_map_and_table.pdf")

# Load the GeoPackage file
gpk_file = "data/costs/processed_costs_Urban_High.gpkg"
scenario_name = os.path.splitext(os.path.basename(gpk_file))[0]

try:
    # Read and process the GeoPackage
    gdf = gpd.read_file(gpk_file)

    # Reproject to Web Mercator for compatibility with basemaps
    gdf = gdf.to_crs(epsg=3857)

    # Create a multipage PDF
    with PdfPages(output_pdf_path) as pdf:
        # ------------------ Page 1: Map ------------------
        # Calculate bounds for a larger area
        bounds = gdf.total_bounds
        buffer_factor = 2.0  # Increase buffer size for a larger area
        x_min, y_min, x_max, y_max = bounds
        x_range, y_range = (x_max - x_min) * buffer_factor, (y_max - y_min) * buffer_factor
        x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2

        # Plot the GeoDataFrame on a basemap
        fig, ax = plt.subplots(figsize=(15, 15))  # Increase figure size
        gdf.plot(ax=ax, alpha=0.6, edgecolor='black')

        # Add a basemap with contextily
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.7)

        # Add a scale bar
        scalebar = ScaleBar(
            dx=1, units="m", location="lower left", scale_loc="bottom",
            length_fraction=0.2, font_properties={"size": 10}, border_pad=0.5,
            box_alpha=0.8, color="black"
        )
        ax.add_artist(scalebar)

        # Add labels for each development
        if 'development' in gdf.columns:
            for _, row in gdf.iterrows():
                centroid = row.geometry.centroid
                ax.text(
                    centroid.x, centroid.y, row['development'], fontsize=8,
                    ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.6)
                )

        # Set limits for the larger area
        ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
        ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)

        # Remove coordinates from the axes
        ax.set_axis_off()

        # Add title to the map page
        ax.set_title(f"Possible Developments for the {scenario_name} Scenario", fontsize=18, pad=20)

        # Save the map to the first page of the PDF
        pdf.savefig(fig)
        plt.close()

        # ------------------ Page 2: Table ------------------
        # Prepare the table DataFrame
        columns = {
            'Source_Name': 'Current End Stop',
            'Target_Name': 'New End Station',
            'Sline': 'S- Line Service',
            'Construction and Maintenance Cost in Mio. CHF': 'Construction & Maintenance Cost (Mio. CHF)',
            'monetized_savings_od_matrix_combined_pop_urba_2': 'Monetized Savings (Mio. CHF)',
            'Net Benefit Urban High [in Mio. CHF]': 'Net Benefit (Mio. CHF)'
        }
        table_df = gdf[list(columns.keys())].rename(columns=columns)
        table_df = table_df.round(2)

        # Create a new figure for the table
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.axis('tight')
        ax.axis('off')

        # Create a table
        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(table_df.columns))))

        # Highlight the header
        for (row, col), cell in table.get_celld().items():
            if row == 0:  # Header row
                cell.set_fontsize(12)
                cell.set_facecolor('#d3d3d3')
                cell.set_text_props(weight='bold')

        # Add title to the table page
        fig.suptitle(f"Table of Possible Developments for the {scenario_name} Scenario", fontsize=18, y=0.95)

        # Save the table to the second page of the PDF
        pdf.savefig(fig)
        plt.close()

    print(f"PDF saved successfully to {output_pdf_path}")

except Exception as e:
    print(f"An error occurred: {e}")




from shapely.geometry import box
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import os
from shapely.affinity import translate

# File paths
s_bahn_lines_path = r"data/Network/Buslines/Linien_des_offentlichen_Verkehrs_-OGD.gpkg"
layer_name_segmented = 'ZVV_S_BAHN_LINIEN_L'

# Output directory for plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)
output_image_path = os.path.join(output_dir, "s_bahn_lines_map.png")

try:
    # Load the bus lines layer
    s_bahn_lines = gpd.read_file(s_bahn_lines_path, layer=layer_name_segmented)

    # Reproject to Web Mercator for compatibility with contextily
    s_bahn_lines = s_bahn_lines.to_crs(epsg=3857)

    # Define refined bounds for Zürich (focus on Zürich Lake and surroundings)
    zurich_bounds = {
        "west": 8.49,  # Exclude far west areas
        "south": 47.35,  # Include more south of Zürich Lake
        "east": 8.65,  # Extend a bit further east
        "north": 47.5  # Cover the entire Zürich area
    }

    # Create a bounding box for Zürich
    bbox = gpd.GeoDataFrame(
        geometry=[box(zurich_bounds["west"], zurich_bounds["south"],
                      zurich_bounds["east"], zurich_bounds["north"])],
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    # Filter the S-Bahn lines within the Zürich region
    s_bahn_lines_filtered = s_bahn_lines[s_bahn_lines.intersects(bbox.iloc[0].geometry)]

    # Offset the lines slightly for visibility (to avoid overlap)
    def offset_geometry(geometry, dx, dy):
        """Offset geometry by dx and dy."""
        return translate(geometry, xoff=dx, yoff=dy)

    unique_lines = s_bahn_lines_filtered["LINIESBAHN"].unique()
    offset_step = 500  # Adjust the offset distance
    line_offsets = {line: i * offset_step for i, line in enumerate(unique_lines)}

    s_bahn_lines_filtered["geometry"] = s_bahn_lines_filtered.apply(
        lambda row: offset_geometry(row.geometry, dx=line_offsets[row["LINIESBAHN"]], dy=0), axis=1
    )

    # Plot the S-Bahn lines
    fig, ax = plt.subplots(figsize=(12, 12))
    s_bahn_lines_filtered.plot(
        ax=ax,
        column="LINIESBAHN",  # Use the LINIESBAHN column for coloring
        legend=True,
        legend_kwds={"title": "S-Bahn Lines"},
        cmap="tab10",  # Use a color map for distinct colors
        linewidth=2,
        alpha=0.8
    )

    # Add the basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.7)

    # Add a title
    ax.set_title("S-Bahn Lines in the Zürich Region", fontsize=16, pad=15)

    # Remove axes
    ax.set_axis_off()

    # Save the plot to a file
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Plot saved successfully to {output_image_path}")

except Exception as e:
    print(f"An error occurred: {e}")


def create_lines(gen_pts_gdf, nearest_infra_pt_gdf):
    """
    Create lines connecting generated points to their nearest infrastructure points
    and include the Sline (service) in the generated links.

    Parameters:
        gen_pts_gdf (GeoDataFrame): Generated points GeoDataFrame with Sline information.
        nearest_infra_pt_gdf (GeoDataFrame): Nearest infrastructure points GeoDataFrame.

    Returns:
        None: Saves the generated links as a GeoPackage file.
    """

    # Sort the generated points by their unique IDs to ensure correct order
    gen_pts_gdf = gen_pts_gdf.sort_values(by="To_ID-point")
    points = gen_pts_gdf.geometry  # Extract geometries (points) from the generated points dataframe

    # Sort the nearest infrastructure points by their unique IDs for the same reason as above
    nearest_infra_pt_gdf = nearest_infra_pt_gdf.sort_values(by="ID_point")
    nearest_points = nearest_infra_pt_gdf.geometry  # Extract geometries (points) from the infrastructure dataframe

    # Create a list of LineString objects by connecting each point from gen_pts_gdf to its nearest point
    line_geometries = [LineString([points.iloc[i], nearest_points.iloc[i]]) for i in range(len(gen_pts_gdf))]

    # Create a new GeoDataFrame to store these lines
    line_gdf = gpd.GeoDataFrame(geometry=line_geometries)

    # Add the ID_new from the generated points (gen_pts_gdf) to the new GeoDataFrame (line_gdf)
    line_gdf["from_ID_new"] = gen_pts_gdf["ID_point"]

    # Add the corresponding ID_point (nearest infrastructure points) to line_gdf
    line_gdf["to_ID"] = nearest_infra_pt_gdf["ID_point"]

    # Add the Sline (service) information from gen_pts_gdf to line_gdf
    line_gdf["Sline"] = gen_pts_gdf["Service"]

    # Identify self-loops (lines where from_ID_new and to_ID are the same)
    self_loops = line_gdf[line_gdf["from_ID_new"] == line_gdf["to_ID"]]

    # Drop the self-loops from the main line_gdf
    line_gdf = line_gdf[line_gdf["from_ID_new"] != line_gdf["to_ID"]].reset_index(drop=True)

    # Set the coordinate reference system (CRS) of the line geometries to EPSG:2056 (Swiss national grid)
    line_gdf = line_gdf.set_crs("epsg:2056")

    # Save the resulting GeoDataFrame to a GeoPackage file
    line_gdf.to_file(r"data\Network\processed\new_links.gpkg")
    return 


def create_lines(gen_pts_gdf, nearest_infra_pt_gdf):
    """
    Create lines connecting generated points to their nearest infrastructure points
    and include the Sline (service) in the generated links.

    Parameters:
        gen_pts_gdf (GeoDataFrame): Generated points GeoDataFrame with Sline information.
        nearest_infra_pt_gdf (GeoDataFrame): Nearest infrastructure points GeoDataFrame.

    Returns:
        None: Saves the generated links as a GeoPackage file.
    """
    # Ensure CRS match
    if gen_pts_gdf.crs != nearest_infra_pt_gdf.crs:
        nearest_infra_pt_gdf = nearest_infra_pt_gdf.to_crs(gen_pts_gdf.crs)

    # Merge datasets to align rows
    merged = pd.merge(
        gen_pts_gdf[['ID_point', 'geometry', 'Service']],
        nearest_infra_pt_gdf[['TO_ID_new', 'geometry_current']],
        left_on='ID_point', right_on='TO_ID_new',
        how='inner'
    )

    # Create LineString geometries
    merged['geometry'] = merged.apply(
        lambda row: LineString([row['geometry'], row['geometry_current']]), axis=1
    )

    # Create a GeoDataFrame for lines
    line_gdf = gpd.GeoDataFrame(merged, geometry='geometry')

    # Drop unnecessary columns
    line_gdf = line_gdf[['ID_point', 'TO_ID_new', 'Service', 'geometry']]

    # Set CRS to EPSG:2056
    line_gdf.set_crs("EPSG:2056", inplace=True)

    # Save the resulting GeoDataFrame
    line_gdf.to_file(r"data/Network/processed/new_links.gpkg", driver="GPKG")
    print(line_gdf)

    return


def assign_services_to_generated_points(raw_edges, generated_points):
    """
    Assign S services to generated points based on the terminating S lines at end nodes.

    Parameters:
        raw_edges (GeoDataFrame): Original raw edges containing 'FromNode', 'ToNode', and 'Service'.
        generated_points (GeoDataFrame): Newly generated points with 'To_ID-point' and other attributes.

    Returns:
        GeoDataFrame: Updated generated_points with the assigned 'Service'.
    """
    # Step 1: Identify end nodes
    end_nodes = set(raw_edges[raw_edges['FromEnd'] == True]['FromNode']) | \
                set(raw_edges[raw_edges['ToEnd'] == True]['ToNode'])

    # Step 2: Map end nodes to their terminating S services
    end_node_services = (
        raw_edges[raw_edges['ToNode'].isin(end_nodes)]
        .groupby('ToNode')['Service']
        .apply(lambda x: random.choice(list(x)) if len(x) > 1 else x.iloc[0])  # Select one random service if multiple
        .to_dict()
    )

    # Step 3: Assign services to generated points based on To_ID-point
    generated_points['Service'] = generated_points['To_ID-point'].map(end_node_services)

    return generated_points

def generate_rail_edges(n,radius):
    radius = 20
    n = 5

    # Step 1 : Identify all service end point nodes
    current_points = gpd.read_file(r"data/Network/processed/points.gpkg")
    #the rows with ID_point values 112, 113, 720, and 2200 have been manually deleted because they are either not active at the moment or are not part of the study area
    current_points = current_points[~current_points['ID_point'].isin([112, 113, 720, 2200])]
    raw_edges = gpd.read_file(r"data/temp/network_railway-services.gpkg")
    endpoints=[]
    for index, edge in raw_edges[raw_edges['FromEnd'] == True].iterrows():
        value = edge.FromNode
        endpoints.append(value)
    for index, edge in raw_edges[raw_edges['ToEnd'] == True].iterrows():
        value = edge.ToNode
        endpoints.append(value)
    endpoints = list(set(endpoints)) #to get unique values
    endnodes_gdf = current_points[current_points['ID_point'].isin(endpoints)]

    # Step 2 : for each end point, make a r km buffer and include the set of all other rail stations
    radius = radius*1000 #converting to m instead of km
    set_gdf = endnodes_gdf.head(0)
    set_gdf['current'] = None

    # Step 2: Iterate over all rows in endnodes_gdf
    for idx, endnode in endnodes_gdf.iterrows():
        # Create a buffer of r km around the endnode
        buffer = endnode.geometry.buffer(radius)  # using 0.5 degrees for simplicity, convert if necessary
        temp_gdf = current_points[current_points.within(buffer)]
        temp_gdf['current'] = endnode['ID_point']
        temp_gdf['geometry_current'] = endnode['geometry']

        # Step 3: If there are more than n stations_gdf in the buffer, only select the n closest
        if len(temp_gdf) > n:
            temp_gdf['distance'] = temp_gdf.geometry.apply(lambda x: endnode.geometry.distance(x))
            temp_gdf = temp_gdf.nsmallest(5, 'distance').drop(columns=['distance'])

        # Append to set_gdf
        set_gdf = gpd.GeoDataFrame(
            pd.concat([set_gdf, temp_gdf], ignore_index=True))
        #set_gdf = set_gdf.append(temp_gdf)

    #set_gdf.to_file(r"data\Network\processed\generated_nodeset.gpkg")
    generated_points = set_gdf[['NAME','ID_point', 'current' ,'XKOORD','YKOORD','HST','geometry']]
    # Renaming columns
    generated_points = generated_points.rename(columns={'current': 'To_ID-point','HST':'index'})
    #generated_points = generated_points.rename(columns={'ID_point':'ID_new','HST':'index'})
    #generated_points['index'].values[:] = 0
    nearest_gdf = gpd.GeoDataFrame(set_gdf[['ID_point', 'current', 'geometry_current']],geometry = 'geometry_current')
    nearest_gdf=nearest_gdf.rename(columns={'ID_point': 'TO_ID_new'})
    nearest_gdf=nearest_gdf.rename(columns={'current': 'ID_point'})

    nearest_gdf = gpd.GeoDataFrame(set_gdf[['ID_point', 'current', 'geometry_current']], geometry='geometry_current')
    nearest_gdf = nearest_gdf.rename(columns={'ID_point': 'TO_ID_new', 'current': 'ID_point'})

    # Set the CRS to EPSG:2056
    nearest_gdf.set_crs("EPSG:2056", inplace=True)


    # Assign services to generated points
    generated_points = assign_services_to_generated_points(raw_edges, generated_points)

    generated_points.to_file(r"data\Network\processed\generated_nodeset.gpkg")
    nearest_gdf.to_file(r"data\Network\processed\endnodes.gpkg")

    create_lines(generated_points, nearest_gdf)

    return

def filter_unnecessary_links():
    """
    Filter out unnecessary links in the new_links GeoDataFrame by ensuring the connection 
    is not redundant within the existing Sline routes. Saves the filtered links as a 
    GeoPackage file.
    """
    # Load data
    raw_edges = gpd.read_file(r"data/temp/network_railway-services.gpkg")
    line_gdf = gpd.read_file(r"data/Network/processed/new_links.gpkg")

    # Step 1: Build Sline routes
    sline_routes = (
        raw_edges.groupby('Service')
        .apply(lambda df: set(df['FromNode']).union(set(df['ToNode'])))
        .to_dict()
    )

    # Step 2: Filter new_links
    filtered_links = []
    for _, row in line_gdf.iterrows():
        sline = row['Sline']
        to_id = row['to_ID']
        from_id = row['from_ID_new']
        
        # Check if the connection is redundant
        if from_id in sline_routes.get(sline, set()) and to_id in sline_routes.get(sline, set()):
            continue  # Skip redundant links
        else:
            filtered_links.append(row)

    # Step 3: Create a GeoDataFrame for filtered links
    filtered_gdf = gpd.GeoDataFrame(filtered_links, geometry='geometry', crs=line_gdf.crs)

    # Save filtered links
    filtered_gdf.to_file(r"data/Network/processed/filtered_new_links.gpkg", driver="GPKG")
    print("Filtered new links saved successfully!")


def filter_unnecessary_links():
    """
    Filter out unnecessary links in the new_links GeoDataFrame by ensuring the connection 
    is not redundant within the existing Sline routes. Saves the filtered links as a 
    GeoPackage file.
    """
    # Load data
    raw_edges = gpd.read_file(r"data/temp/network_railway-services.gpkg")  # Use raw string
    line_gdf = gpd.read_file(r"data/Network/processed/new_links.gpkg")      # Use raw string

    # Step 1: Build Sline routes
    sline_routes = (
        raw_edges.groupby('Service')
        .apply(lambda df: set(df['FromNode']).union(set(df['ToNode'])))
        .to_dict()
    )

    # Step 2: Filter new_links
    filtered_links = []
    for _, row in line_gdf.iterrows():
        sline = row['Sline']
        to_id = row['to_ID']
        from_id = row['from_ID_new']
        
        # Check if the connection is redundant
        if from_id in sline_routes.get(sline, set()) and to_id in sline_routes.get(sline, set()):
            continue  # Skip redundant links
        else:
            filtered_links.append(row)

    # Step 3: Create a GeoDataFrame for filtered links
    filtered_gdf = gpd.GeoDataFrame(filtered_links, geometry='geometry', crs=line_gdf.crs)

    # Save filtered links
    filtered_gdf.to_file(r"data/Network/processed/filtered_new_links.gpkg", driver="GPKG")  # Use raw string
    print("Filtered new links saved successfully!")


def create_merged_trainstation_buffers(closest_trainstations_df, stops, output_path):
    '''
    Create merged polygons for train station catchment areas based on bus stop buffers.

    Parameters:
        closest_trainstations_df (GeoDataFrame): DataFrame containing bus stops and their closest train stations.
        stops (GeoDataFrame): GeoDataFrame containing bus stop geometries.
        output_path (str): Path to save the resulting GeoPackage.

    Returns:
        None
    '''

    # Step 1: Merge `closest_trainstations_df` with `stops` to add geometries
    merged_df = closest_trainstations_df.merge(
        stops[['DIVA_NR', 'geometry']],
        left_on='bus_stop',
        right_on='DIVA_NR',
        how='left'
    )

    # Ensure connected_df is a GeoDataFrame with valid geometries
    connected_df = gpd.GeoDataFrame(merged_df, geometry='geometry', crs='EPSG:2056')

    # Check and remove invalid or missing geometries
    invalid_geometries = connected_df[connected_df['geometry'].is_empty | connected_df['geometry'].isna()]
    if not invalid_geometries.empty:
        print("Warning: Invalid geometries found. These rows will be excluded:")
        print(invalid_geometries)
        connected_df = connected_df[~(connected_df['geometry'].is_empty | connected_df['geometry'].isna())]

    # Apply buffer operation
    connected_df['buffer'] = connected_df['geometry'].buffer(650)

    # Step 5: Group buffers by `train_station` and merge them
    grouped_buffers = (
        connected_df.groupby('train_station')
        .agg({'buffer': lambda x: unary_union(x)})
        .reset_index()
    )

    # Step 6: Create a GeoDataFrame for the merged polygons
    merged_polygons = gpd.GeoDataFrame(grouped_buffers, geometry='buffer', crs='EPSG:2056')

    # Step 7: Save the merged polygons to a GeoPackage
    merged_polygons.to_file(output_path, driver="GPKG")

    # Print completion message
    print(f"Merged polygons have been saved to {output_path}")


def process_raster(input_path, output_path):
        with rasterio.open(input_path) as src:
            # Number of bands in the raster
            num_bands = src.count

            # Initialize a 3D array for corrected raster data (bands, height, width)
            data_corrected = np.zeros((num_bands, height, width), dtype=src.dtypes[0])

            # Process each band
            for band_idx in range(1, num_bands + 1):
                band_data = np.zeros((height, width), dtype=src.dtypes[0])

                # Reproject and resample
                rasterio.warp.reproject(
                    source=rasterio.band(src, band_idx),
                    destination=band_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=rasterio.warp.Resampling.bilinear,
                )

                # Mask the raster to fit within the boundary
                mask = rasterio.features.geometry_mask(
                    boundary_geom, transform=transform, invert=True, out_shape=(height, width)
                )
                band_data[~mask] = np.nan  # Set values outside the boundary to NaN

                # Store the corrected band
                data_corrected[band_idx - 1] = band_data

            # Save the corrected raster
            profile = src.profile
            profile.update(
                driver="GTiff",
                count=num_bands,
                height=height,
                width=width,
                transform=transform,
                crs=crs,
                nodata=np.nan,
            )
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(data_corrected)

    # Process both employment and population rasters
    process_raster(empl_path, output_empl_path)
    process_raster(pop_path, output_pop_path)


def correct_rasters_to_extent(
    empl_path, pop_path, 
    output_empl_path, output_pop_path,
    reference_boundary, resolution=100, crs="EPSG:2056"):

    """
    Corrects the raster files to match the given boundary extent and resolution for all bands.

    Args:
        empl_path (str): Path to the employment raster file.
        pop_path (str): Path to the population raster file.
        output_empl_path (str): Path to save the corrected employment raster.
        output_pop_path (str): Path to save the corrected population raster.
        reference_boundary (shapely.geometry.Polygon): Boundary polygon for cropping and masking.
        resolution (int): Resolution of the output raster in meters. Default is 100.
        crs (str): Coordinate Reference System for the output rasters. Default is "EPSG:2056".
    """
    # Determine the bounds and raster grid size
    xmin, ymin, xmax, ymax = reference_boundary.bounds
    width = int((xmax - xmin) / resolution)
    height = int((ymax - ymin) / resolution)
    transform = rasterio.transform.from_origin(xmin, ymax, resolution, resolution)

    # Convert the boundary to GeoJSON-like format for masking
    boundary_geom = [mapping(reference_boundary)]


# Function to create graphs from a list of DataFrames
def create_graphs_from_directories(directories):
    """
    Create a list of directed graphs from a list of file directories.
    
    Parameters:
        directories (list): List of file paths to GeoPackage or CSV files.
        
    Returns:
        list: A list of NetworkX directed graphs.
    """
    graphs = []
    for i, directory in enumerate(directories):
        try:
            print(f"Reading file {i+1}/{len(directories)}: {directory}...")
            # Read the file into a GeoDataFrame
            if directory.endswith('.gpkg'):
                df = gpd.read_file(directory)
            elif directory.endswith('.csv'):
                df = pd.read_csv(directory)
            else:
                print(f"Unsupported file format: {directory}")
                continue
            
            # Create the graph
            graph = create_directed_graph(df)
            graphs.append(graph)
            print(f"Graph {i+1} created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error processing file {directory}: {e}")
    return graphs


def create_directed_graph(df):
    """
    Create a directed graph from a DataFrame of travel connections.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing travel connections with columns
                           like 'FromStation', 'ToStation', 'TravelTime', 'InVehWait', etc.
    
    Returns:
        nx.DiGraph: A NetworkX directed graph.
    """
    G = nx.DiGraph()
    
    # Add main nodes for each station
    for station in set(df['FromStation']).union(set(df['ToStation'])):
        G.add_node(f"main_{station}", type="main_node", station=station)
    
    # Add sub-nodes and connections between them
    for idx, row in df.iterrows():
        # Skip rows with critical missing data
        if pd.isna(row['FromStation']) or pd.isna(row['ToStation']) or pd.isna(row['Service']) or pd.isna(row['Direction']):
            print(f"Skipping row {idx} due to missing critical data.")
            continue
        
        # Create sub-nodes
        from_sub_node = f"sub_{row['FromStation']}_{row['Service']}_{row['Direction']}"
        to_sub_node = f"sub_{row['ToStation']}_{row['Service']}_{row['Direction']}"
        
        G.add_node(from_sub_node, type="sub_node", station=row['FromStation'], service=row['Service'], direction=row['Direction'])
        G.add_node(to_sub_node, type="sub_node", station=row['ToStation'], service=row['Service'], direction=row['Direction'])
        
        # Validate TravelTime and InVehWait
        travel_time = row['TravelTime'] if pd.notna(row['TravelTime']) else None
        in_veh_wait = row['InVehWait'] if pd.notna(row['InVehWait']) else 0  # Default to 0 if missing

        if travel_time is not None:
            travel_time = round(travel_time)  # Convert to integer for consistency
            weight = travel_time + int(in_veh_wait)
            G.add_edge(from_sub_node, to_sub_node, weight=weight)
        else:
            print(f"Skipping edge from {from_sub_node} to {to_sub_node} due to missing TravelTime.")
    
    # Connect sub-nodes to their corresponding main nodes in both directions
    for node in G.nodes:
        if G.nodes[node]["type"] == "sub_node":
            station = G.nodes[node]["station"]
            main_node = f"main_{station}"
            if main_node in G.nodes:
                G.add_edge(node, main_node, weight=3)  # Sub-to-Main
                G.add_edge(main_node, node, weight=3)  # Main-to-Sub

    # Add direct edges between sub-nodes of the same service, direction, and station
    sub_nodes = [node for node, data in G.nodes(data=True) if data["type"] == "sub_node"]
    for sub1 in sub_nodes:
        for sub2 in sub_nodes:
            if sub1 != sub2:
                data1 = G.nodes[sub1]
                data2 = G.nodes[sub2]
                # Add edge only if the same service, direction, and station
                if (data1["service"] == data2["service"] and 
                    data1["direction"] == data2["direction"] and 
                    data1["station"] == data2["station"]):
                    G.add_edge(sub1, sub2, weight=0)  # Forward connection
                    G.add_edge(sub2, sub1, weight=0)  # Reverse connection
    
    return G




# Function to create a directed graph
def create_directed_graph(df):
    G = nx.DiGraph()
    
    # Add main nodes for each station
    for station in set(df['FromStation']).union(set(df['ToStation'])):
        G.add_node(f"main_{station}", type="main_node", station=station)
    
    # Add sub-nodes and connections between them
    for idx, row in df.iterrows():
        from_sub_node = f"sub_{row['FromStation']}_{row['Service']}_{row['Direction']}"
        to_sub_node = f"sub_{row['ToStation']}_{row['Service']}_{row['Direction']}"
        
        # Add sub-nodes with attributes
        G.add_node(from_sub_node, type="sub_node", station=row['FromStation'], service=row['Service'], direction=row['Direction'])
        G.add_node(to_sub_node, type="sub_node", station=row['ToStation'], service=row['Service'], direction=row['Direction'])
        
        # Add edges between sub-nodes if the line exists in the dataframe
        if pd.notna(row['TravelTime']) and pd.notna(row['InVehWait']):
            weight = row['TravelTime'] + row['InVehWait']
            G.add_edge(from_sub_node, to_sub_node, weight=weight)
    
    # Connect sub-nodes to their corresponding main nodes in both directions
    for node in G.nodes:
        if G.nodes[node]["type"] == "sub_node":
            station = G.nodes[node]["station"]
            main_node = f"main_{station}"
            if main_node in G.nodes:
                G.add_edge(node, main_node, weight=3)  # Sub-to-Main
                G.add_edge(main_node, node, weight=3)  # Main-to-Sub

    # Add direct edges between sub-nodes of the same service, direction, and stations
    sub_nodes = [node for node, data in G.nodes(data=True) if data["type"] == "sub_node"]
    for sub1 in sub_nodes:
        for sub2 in sub_nodes:
            if sub1 != sub2:
                data1 = G.nodes[sub1]
                data2 = G.nodes[sub2]
                # Add edge only if the same service, direction, and station
                if (data1["service"] == data2["service"] and 
                    data1["direction"] == data2["direction"] and 
                    data1["station"] == data2["station"]):
                    G.add_edge(sub1, sub2, weight=0)  # Forward connection
                    G.add_edge(sub2, sub1, weight=0)  # Reverse connection
    
    return G


def update_stations(combined_gdf, output_path):
    """
    Update the FromStation and ToStation columns based on FromNode and ToNode values.
    Handles potential data type mismatches and missing values.

    Parameters:
    combined_gdf (pd.DataFrame): Input DataFrame with columns FromNode, ToNode, FromStation, ToStation.

    Returns:
    pd.DataFrame: Updated DataFrame.
    """
    # Ensure FromNode and ToNode are numeric (convert if necessary)
    combined_gdf["FromNode"] = pd.to_numeric(combined_gdf["FromNode"], errors="coerce")
    combined_gdf["ToNode"] = pd.to_numeric(combined_gdf["ToNode"], errors="coerce")

    # Drop rows where FromNode or ToNode are NaN after conversion
    combined_gdf = combined_gdf.dropna(subset=["FromNode", "ToNode"])
    
    # Convert FromNode and ToNode to integer type
    combined_gdf["FromNode"] = combined_gdf["FromNode"].astype(int)
    combined_gdf["ToNode"] = combined_gdf["ToNode"].astype(int)

    # Define mapping for nodes to stations
    node_to_station = {1008: "Hinwil", 2298: "Uster"}

    # Update ToStation based on ToNode
    combined_gdf.loc[combined_gdf["ToNode"] == 1018, "ToStation"] = "Hinwil"
    combined_gdf.loc[combined_gdf["ToNode"] == 2298, "ToStation"] = "Uster"

    # Update FromStation based on FromNode
    combined_gdf.loc[combined_gdf["FromNode"] == 1018, "FromStation"] = "Hinwil"
    combined_gdf.loc[combined_gdf["FromNode"] == 2298, "FromStation"] = "Uster"

    # Save the output
    combined_gdf.to_file(output_path, driver="GPKG")
    print("Combined network with new links saved successfully!")

    return combined_gdf

def analyze_travel_times(od_times_status_quo, od_times_dev, selected_indices, od_nodes):
    """
    Analyze travel times for the status quo and selected developments.

    Parameters:
    - od_times_status_quo: list of DataFrames, first element contains status quo data
    - od_times_dev: list of DataFrames, contains development data
    - selected_indices: list of int, indices of developments to analyze
    - od_nodes: list of str, OD nodes to consider

    Returns:
    - DataFrame containing status quo times, delta times for selected developments, and OD pairs
    """

    # Extract the status quo DataFrame
    status_quo_df = od_times_status_quo[0]

    # Filter the required developments
    selected_developments = [od_times_dev[i] for i in selected_indices]

    # Generate OD pairs using the provided nodes
    od_pairs = [(origin, destination) for origin in od_nodes for destination in od_nodes if origin != destination]

    # Function to extract travel times for specified OD pairs
    def extract_travel_times(od_matrix, od_pairs):
        extracted_data = []
        for origin, destination in od_pairs:
            filtered_data = od_matrix[(od_matrix['from_id'] == origin) & 
                                      (od_matrix['to_id'] == destination)]
            if not filtered_data.empty:
                extracted_data.append({
                    "origin": origin,
                    "destination": destination,
                    "time": filtered_data.iloc[0]['time']
                })
        return pd.DataFrame(extracted_data)

    # Extract travel times for the status quo
    status_quo_times = extract_travel_times(status_quo_df, od_pairs)
    status_quo_times = status_quo_times.rename(columns={"time": "status_quo_time"})

    # Extract travel times for each selected development
    development_times = []
    for i, dev_data in enumerate(selected_developments):
        dev_times = extract_travel_times(dev_data, od_pairs)
        dev_times = dev_times.rename(columns={"time": f"dev_{selected_indices[i] + 1}_time"})
        development_times.append(dev_times)

    # Merge all data into a single DataFrame
    merged = status_quo_times
    for dev_times in development_times:
        merged = pd.merge(merged, dev_times, on=["origin", "destination"], how="left")

    # Calculate delta times for each development
    for i, index in enumerate(selected_indices):
        merged[f"delta_dev_{index + 1}"] = merged[f"dev_{index + 1}_time"] - merged["status_quo_time"]

    # Select and reorder columns
    columns_to_display = ["origin", "destination", "status_quo_time"] + \
                         [f"delta_dev_{index + 1}" for index in selected_indices]
    final_result = merged[columns_to_display]

    return final_result



import rasterio
from rasterio.mask import mask
import geopandas as gpd

# File paths
pop_combined_file = r"data/independent_variable/processed/scenario/pop_combined.tif"
empl_combined_file = r"data/independent_variable/processed/scenario/empl_combined.tif"
boundary_file = r"data/_basic_data/innerboundary.shp"
output_pop_path = r"data/independent_variable/processed/scenario/pop20_corrected.tif"
output_empl_path = r"data/independent_variable/processed/scenario/empl20_corrected.tif"

# Load the boundary shapefile
boundary_gdf = gpd.read_file(boundary_file).to_crs("EPSG:2056")
boundary_geometry = [geom for geom in boundary_gdf.geometry]

# Function to clip a raster
def clip_raster(input_raster_path, output_raster_path, geometry, crs="EPSG:2056"):
    with rasterio.open(input_raster_path) as src:
        # Clip the raster
        out_image, out_transform = mask(src, geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": crs
        })
        # Save the clipped raster
        with rasterio.open(output_raster_path, "w", **out_meta) as dest:
            dest.write(out_image)

# Clip the population raster
clip_raster(pop_combined_file, output_pop_path, boundary_geometry)

# Clip the employment raster
clip_raster(empl_combined_file, output_empl_path, boundary_geometry)


# Import layers to plot
    tif_path_plot = r"data/landuse_landcover/processed/zone_no_infra/protected_area_corridor.tif"

    links_beeline = gpd.read_file(r"data/Network/processed/new_links.gpkg")
    links_realistic = gpd.read_file(r"data/Network/processed/new_links_realistic.gpkg")
    print(links_realistic.head(5).to_string())
    # Plot the net benefits for each generated point and interpolate the area in between
    generated_points = gpd.read_file(r"data/Network/processed/generated_nodes.gpkg")
    # Get a gpd df with points have an ID_new that is not in links_realistic ID_new
    filtered_rand_gdf = generated_points[~generated_points["ID_new"].isin(links_realistic["ID_new"])]
    plot_points_gen(points=generated_points, edges=links_beeline, banned_area=tif_path_plot, boundary=boundary_plot, network=network, all_zones=True, plot_name="gen_nodes_beeline")
    #plot_points_gen(points=generated_points, points_2=filtered_rand_gdf, edges=links_realistic, banned_area=tif_path_plot, boundary=boundary_plot, network=network, all_zones=False, plot_name="gen_links_realistic")




    # Plot the net benefits for each generated point and interpolate the area in between
    # if plot_name is not False, then the plot is stored in "plot/results/{plot_name}.png"
    plot_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="scenario low growth", boundary=boundary_plot, network=network,
                     access_points=current_access_points, plot_name="total_costs_low",col="total_low")
    plot_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="scenario medium growth", boundary=boundary_plot, network=network,
                     access_points=current_access_points, plot_name="total_costs_medium",col="total_medium")
    plot_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="scenario high growth", boundary=boundary_plot, network=network,
                     access_points=current_access_points, plot_name="total_costs_high",col="total_high")

    # Plot single cost element

    plot_single_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="construction",
                            boundary=boundary_plot, network=network, access_points=current_access_points,
                            plot_name="construction and maintenance", col="construction_maintenance")
    # Due to erros when plotting convert values to integer
    gdf_costs["local_s1"] = gdf_costs["local_s1"].astype(int)
    plot_single_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="access time to highway",
                            boundary=boundary_plot, network=network, access_points=current_access_points,
                            plot_name="access_costs",col="local_s1")
    plot_single_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="highway travel time",
                            boundary=boundary_plot, network=network, access_points=current_access_points,
                            plot_name="tt_costs",col="tt_medium")
    plot_single_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="noise emissions",
                            boundary=boundary_plot, network=network, access_points=current_access_points,
                            plot_name="externalities_costs", col="externalities_s1")


# Assuming od_times_status_quo is a list of DataFrames
# Using the first DataFrame in the list (adjust index if needed)
status_quo_df = od_times_status_quo[0]

# Filter the required developments: indices 3, 6, 7 (Development 4, 7, 8)
selected_indices = [3, 6, 7]
selected_developments = [od_times_dev[i] for i in selected_indices]

# Define the OD pairs to analyze
od_pairs = [("main_Aathal", "main_Uster"), 
            ("main_Aathal", "main_Wetzikon ZH"), 
            ("main_Hinwil", "main_Wetzikon ZH"),
            ("main_Pfäffikon ZH", "main_Uster"),
            ("main_Pfäffikon ZH", "main_Aathal")]

# Function to extract travel times for specified OD pairs
def extract_travel_times_debug(od_matrix, od_pairs):
    extracted_data = []
    for origin, destination in od_pairs:
        filtered_data = od_matrix[(od_matrix['from_id'] == origin) & 
                                  (od_matrix['to_id'] == destination)]
        if not filtered_data.empty:
            extracted_data.append({
                "origin": origin,
                "destination": destination,
                "time": filtered_data.iloc[0]['time']
            })
        else:
            print(f"No match found for OD pair: {origin} -> {destination}")
    return pd.DataFrame(extracted_data)

# Extract and print travel times for the status quo
status_quo_times = extract_travel_times_debug(status_quo_df, od_pairs)
print("\nTravel Times for Status Quo:")
print(status_quo_times)

# Extract and print travel times for each selected development
results = {}
for i, dev_data in enumerate(selected_developments, start=1):
    dev_times = extract_travel_times_debug(dev_data, od_pairs)
    results[f"Development {selected_indices[i-1] + 1}"] = dev_times

    print(f"\nTravel Times for Development {selected_indices[i-1] + 1}:")
    print(dev_times)


import pandas as pd

# Assuming od_times_status_quo is a list of DataFrames
# Using the first DataFrame in the list (adjust index if needed)
status_quo_df = od_times_status_quo[0]

# Filter the required developments: indices 3, 6, 7 (Development 4, 7, 8)
selected_indices = [3, 6, 5, 7]
selected_developments = [od_times_dev[i] for i in selected_indices]

# Define the provided OD nodes
od_nodes = [
    'main_Rüti ZH', 'main_Nänikon-Greifensee', 'main_Uster', 'main_Wetzikon ZH',
    'main_Zürich Altstetten', 'main_Schwerzenbach ZH', 'main_Fehraltorf', 
    'main_Bubikon', 'main_Zürich HB', 'main_Kempten', 'main_Pfäffikon ZH', 
    'main_Zürich Oerlikon', 'main_Zürich Stadelhofen', 'main_Hinwil', 'main_Aathal'
]

# Generate OD pairs using the provided nodes
od_pairs = [(origin, destination) for origin in od_nodes for destination in od_nodes if origin != destination]
print(f"Number of OD pairs: {len(od_pairs)}")

# Function to extract travel times for specified OD pairs
def extract_travel_times_debug(od_matrix, od_pairs):
    extracted_data = []
    for origin, destination in od_pairs:
        filtered_data = od_matrix[(od_matrix['from_id'] == origin) & 
                                  (od_matrix['to_id'] == destination)]
        if not filtered_data.empty:
            extracted_data.append({
                "origin": origin,
                "destination": destination,
                "time": filtered_data.iloc[0]['time']
            })
        else:
            print(f"No match found for OD pair: {origin} -> {destination}")
    return pd.DataFrame(extracted_data)

# Extract travel times for the status quo
status_quo_times = extract_travel_times_debug(status_quo_df, od_pairs)
status_quo_times = status_quo_times.rename(columns={"time": "status_quo_time"})

# Extract travel times for each selected development
development_times = []
for i, dev_data in enumerate(selected_developments):
    dev_times = extract_travel_times_debug(dev_data, od_pairs)
    dev_times = dev_times.rename(columns={"time": f"dev_{selected_indices[i] + 1}_time"})
    development_times.append(dev_times)

# Merge all data into a single DataFrame
merged = status_quo_times
for dev_times in development_times:
    merged = pd.merge(merged, dev_times, on=["origin", "destination"], how="left")

# Calculate delta times for each development
for i, index in enumerate(selected_indices):
    merged[f"delta_dev_{index + 1}"] = merged[f"dev_{index + 1}_time"] - merged["status_quo_time"]

# Select and reorder columns
columns_to_display = ["origin", "destination", "status_quo_time"] + \
                     [f"delta_dev_{index + 1}" for index in selected_indices]
final_result = merged[columns_to_display]

# Display the final DataFrame
print("\nFinal Travel Times and Delta Times:")
print(final_result)

import os
import glob
import time
import geopandas as gpd
import pandas as pd
import networkx as nx
import rasterio
import rasterio.features
import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon


def travel_cost_polygon(frame):

    points_all = gpd.read_file(r"data\Network\processed\points_attribute.gpkg")
    # Need the node id as ID_point
    points_all = points_all[points_all["intersection"] == 0]
    points_all_frame = points_all.cx[frame[0]:frame[2], frame[1]:frame[3]]
    # print(points_all_frame.head(10).to_string())

    # travel speed
    raster_file = r"data\Network\OSM_tif\speed_limit_raster.tif"
    # should change lake speed to 0
    # and other area to slightly higher speed to other land covers
    with rasterio.open(raster_file) as dataset:
        raster_data = dataset.read(1)  # Assumes forbidden cells are marked with 1 or another distinct value
        transform = dataset.transform

        # Convert real-world coordinates to raster indices
        sources_indices = [~transform * (x, y) for x, y in zip(points_all_frame.geometry.x, points_all_frame.geometry.y)]
        sources_indices = [(int(y), int(x)) for x, y in sources_indices]
        """
        # Calculate path lengths using Dijkstra's algorithm
        start = time.time()
        path_lengths, source_index = nx.multi_source_dijkstra_path_length(graph, sources_indices, weight='weight')
        end = time.time()
        print(f"Time dijkstra: {end-start} sec.")

        # Initialize an empty raster for path lengths
        path_length_raster = np.full(raster_data.shape, np.nan)
        source_raster = np.full(raster_data.shape, np.nan)

        # Populate the raster with path lengths
        for node, length in path_lengths.items():
            y, x = node
            path_length_raster[y, x] = length
        """

        sources_indices, idx_correct = match_access_point_on_highway(sources_indices, raster_data)
        # Remove all cells that contain highway
        #raster_data[raster_data > 90] = 50


        start = time.time()
        # Convert raster to graph
        graph = raster_to_graph(raster_data)
        end = time.time()
        print(f"Time to initialize graph: {end-start} sec.")

        start = time.time()
        # Get both path lengths and paths
        distances, paths = nx.multi_source_dijkstra(G=graph, sources=sources_indices, weight='weight')
        end = time.time()
        print(f"Time dijkstra: {end - start} sec.")

        # Initialize empty rasters for path lengths and source coordinates
        path_length_raster = np.full(raster_data.shape, np.nan)

        # Initialize an empty raster with np.nan and dtype float
        temp_raster = np.full(raster_data.shape, np.nan, dtype=float)
        # Change the dtype to object
        source_coord_raster = temp_raster.astype(object)

        # Populate the rasters
        for node, path in paths.items():
            y, x = node
            path_length_raster[y, x] = distances[node]

            if path:  # Check if path is not empty
                source_y, source_x = path[0]  # First element of the path is the source
                source_coord_raster[y, x] = (source_y, source_x)


    # Save the path length raster
    with rasterio.open(
            r'data\Network\travel_time\travel_time_raster.tif', 'w',
            driver='GTiff',
            height=path_length_raster.shape[0],
            width=path_length_raster.shape[1],
            count=1,
            dtype=path_length_raster.dtype,
            crs=dataset.crs,
            transform=transform
    ) as new_dataset:
        new_dataset.write(path_length_raster, 1)

    # Inverse transform to convert CRS coordinates to raster indices
    inv_transform = ~transform

    # Convert the geometry coordinates to raster indices
    points_all_frame['raster_x'], points_all_frame['raster_y'] = zip(*points_all_frame['geometry'].apply(lambda geom: inv_transform * (geom.x, geom.y)))
    points_all_frame["raster_y"] = points_all_frame["raster_y"].apply(lambda x: int(np.floor(x)))
    points_all_frame["raster_x"] = points_all_frame["raster_x"].apply(lambda x: int(np.floor(x)))

    # Create a dictionary to map raster indices to ID_point
    index_to_id = {(row['raster_y'], row['raster_x']): row['ID_point'] for _, row in points_all_frame.iterrows()}

    # Iterate over the source_coord_raster and replace coordinates with ID_point
    # Assuming new_array is your 2D array of coordinates and matched_dict is your dictionary


    for (y, x), coord in np.ndenumerate(source_coord_raster):
        if coord in idx_correct:
            source_coord_raster[y, x] = idx_correct[coord]

    for (y, x), source_coord in np.ndenumerate(source_coord_raster):
        if source_coord in index_to_id:
            source_coord_raster[y, x] = index_to_id[source_coord]

    # Convert the array to a float data type
    source_coord_raster = source_coord_raster.astype(float)
    # Set NaN values to a specific NoData value, e.g., -1
    source_coord_raster[np.isnan(source_coord_raster)] = -1

    path_id_raster = r'data\Network\travel_time\source_id_raster.tif'
    with rasterio.open(path_id_raster, 'w',
        driver='GTiff',
        height=source_coord_raster.shape[0],
        width=source_coord_raster.shape[1],
        count=1,
        dtype=source_coord_raster.dtype,
        crs=dataset.crs,
        transform=transform
        ) as new_dataset:
            new_dataset.write(source_coord_raster, 1)

    # get Voronoi polygons in vector data as gpd df
    gdf_polygon = raster_to_polygons(path_id_raster)
    #print(gdf_polygon.head(10).to_string())
    gdf_polygon.to_file(r"data\Network\travel_time\Voronoi_statusquo.gpkg")

        # how to get the inputs? nodes in which reference system, weights automatically?
        # how to get the coordinates of the closest point?

        # tif with travel time
        # tif with closest point


    return


def raster_to_polygons___(tif_path):
    # Read the raster data
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        data = data.astype('int32')
        transform = src.transform

    # Find unique values in the raster
    unique_values = np.unique(data[data >= 0])  # Assuming negative values are no-data

    # Initialize list to store polygons and their values
    polygons = []

    # Iterate over unique values and create polygons
    for val in unique_values:
        # Create mask for the current value
        mask = data == val

        # Generate shapes (polygons) from the mask
        shapes = rasterio.features.shapes(data, mask=mask, transform=transform)
        for shape, value in shapes:
            if value == val:
                # Convert shape to a Shapely Polygon and add to list
                polygons.append({
                    'geometry': Polygon(shape['coordinates'][0]),
                    'ID_point': val
                })

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(polygons, crs=src.crs)
    #gdf_dissolved = gdf.dissolve(by='ID_point')
    gdf_dissolved = groupby_multipoly(gdf, by="ID_point")

    return gdf_dissolved


def raster_to_polygons(tif_path):
    # Read the raster data
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        data = data.astype('int32')
        transform = src.transform

    # Find unique positive values in the raster
    unique_values = np.unique(data[data >= 0])

    # Create a mask for negative values (holes)
    negative_mask = data < 0

    # Initialize list to store polygons and their values
    polygons = []

    # Iterate over unique values and create polygons
    for val in unique_values:
        # Create mask for the current value
        positive_mask = data == val

        # Generate shapes (polygons) for positive values
        positive_shapes = rasterio.features.shapes(data, mask=positive_mask, transform=transform)

        # Generate shapes for negative values (holes)
        hole_shapes = rasterio.features.shapes(data, mask=negative_mask, transform=transform)

        # Combine positive shapes and holes
        combined_polygons = []
        for shape, value in positive_shapes:
            if value == val:
                outer_polygon = Polygon(shape['coordinates'][0])

                # Create holes
                holes = [Polygon(hole_shape['coordinates'][0]) for hole_shape, hole_value in hole_shapes if hole_value < 0]
                holes_union = unary_union(holes)

                # Combine outer polygon with holes
                if holes_union.is_empty:
                    combined_polygons.append(outer_polygon)
                else:
                    combined_polygon = outer_polygon.difference(holes_union)
                    combined_polygons.append(combined_polygon)

        # Add combined polygons to list
        polygons.extend([{'geometry': poly, 'ID_point': val} for poly in combined_polygons if not poly.is_empty])

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(polygons, crs=src.crs)
    gdf_dissolved = gdf.dissolve(by='ID_point')

    return gdf_dissolved


def groupby_multipoly(df, by, aggfunc="first"):
    data = df.drop(labels=df.geometry.name, axis=1)
    aggregated_data = data.groupby(by=by).agg(aggfunc)

    # Process spatial component
    def merge_geometries(block):
        return MultiPolygon(block.values)

    g = df.groupby(by=by, group_keys=False)[df.geometry.name].agg(
        merge_geometries
    )

    # Aggregate
    aggregated_geometry = gpd.GeoDataFrame(g, geometry=df.geometry.name, crs=df.crs)
    # Recombine
    aggregated = aggregated_geometry.join(aggregated_data)
    return aggregated


def raster_to_graph(raster_data):
    #high_weight = 90 # sec  is the time required to cross 100 with 4 km/h
    raster_cell = 100 # m

    # convert travel speed from km/h to m/s
    raster_data = raster_data * 1000 / 3600

    rows, cols = raster_data.shape
    graph = nx.grid_2d_graph(rows, cols)

    nodes_to_remove = []
    for node in graph.nodes:
        y, x = node
        if raster_data[y, x] == 0:
            nodes_to_remove.append(node)

    graph.remove_nodes_from(nodes_to_remove)

    # Add weights for existing edges in the grid_2d_graph
    for (node1, node2) in graph.edges:
        y1, x1 = node1
        y2, x2 = node2
        if raster_data[y1, x1] == 0 or raster_data[y2, x2] == 0:
            # Assign a high weight to this edge
            weight = None
        else:
            # Calculate weight normally
            weight = (raster_cell / raster_data[y1, x1] + raster_cell / raster_data[y2, x2]) / 2

        #weight = (0.1 / raster_data[y1, x1] + 0.1 / raster_data[y2, x2]) / 2 * 3600
        graph[node1][node2]['weight'] = weight

    # Add diagonal edges (from 4 to 8 neighbors)
    new_edges = []
    for x in range(cols - 1):
        for y in range(rows - 1):
            # Check for zero values in raster data for diagonal neighbors
            if raster_data[y, x] == 0 or raster_data[y + 1, x + 1] == 0:
                weight = None
            else:
                weight = 1.4 * (raster_cell / raster_data[y, x] + raster_cell / raster_data[y + 1, x + 1]) / 2

            new_edges.append(((y, x), (y + 1, x + 1), {'weight': weight}))
            
            if raster_data[y, x + 1] == 0 or raster_data[y + 1, x] == 0:
                weight = None
            else:
                weight = 1.4 * (raster_cell / raster_data[y, x + 1] + raster_cell / raster_data[y + 1, x]) / 2

            new_edges.append(((y, x + 1), (y + 1, x), {'weight': weight}))

    # Add new diagonal edges with calculated weights
    graph.add_edges_from(new_edges)

    # iterate over all options
    # get the closest point
    return graph


def match_access_point_on_highway(idx, raster):
    # get value of all idx in raster cell
    # initialise dict
    # for i in idx
    #   if value of i < 120
    #       if there is a cell A with raster value == 120 in 8 neighbors of i
    #           replace idx of i = idx of cell A
    #           dict.add(idx of A: idx of i)
    #       elif value of i < 100
    #           if there is a cell B with raster value == 100 in 8 neighbors of i
    #               replace idx of i = idx of cell B
    #               dict.add(idx of B: idx of i)
    #       elif value of i < 80
    #           if there is a cell C with raster value == 80 in 8 neighbors of i
    #               replace idx of i = idx of cell C
    #               dict.add(idx of C: idx of i)
    # return idx, dict

    matched_dict = {}
    updated_idx = []

    for i in idx:
        y, x = i
        value = raster[y, x]
        match_found = False  # Flag to indicate if a match is found

        if value < 80:
            # First search in the immediate neighborhood
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < raster.shape[0] and 0 <= nx < raster.shape[1]:
                        if raster[ny, nx] >= 100:
                            matched_dict[(ny, nx)] = i
                            i = (ny, nx)
                            match_found = True
                            break
                        elif raster[ny, nx] >= 80:
                            matched_dict[(ny, nx)] = i
                            i = (ny, nx)
                            match_found = True
                            break
                        elif raster[ny, nx] >= 50:
                            matched_dict[(ny, nx)] = i
                            i = (ny, nx)
                            match_found = True
                            break
                        elif raster[ny, nx] >= 30:
                            matched_dict[(ny, nx)] = i
                            i = (ny, nx)
                            match_found = True
                            break
                if match_found:
                    break

            # If no match found, expand search to wider range
            if not match_found:
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dy == 0 and dx == 0:  # Skip the cell itself
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < raster.shape[0] and 0 <= nx < raster.shape[1]:
                            if raster[ny, nx] >= 100:
                                matched_dict[(ny, nx)] = i
                                i = (ny, nx)
                                break
                            elif raster[ny, nx] >= 80:
                                matched_dict[(ny, nx)] = i
                                i = (ny, nx)
                                break
                            elif raster[ny, nx] >= 50:
                                matched_dict[(ny, nx)] = i
                                i = (ny, nx)
                                match_found = True
                                break
                            elif raster[ny, nx] >= 30:
                                matched_dict[(ny, nx)] = i
                                i = (ny, nx)
                                match_found = True
                                break
                    if match_found:
                        print("No point found to match on network")
                        break

        updated_idx.append(i)

    return updated_idx, matched_dict


def travel_cost_developments(frame):
    # First delete all elements that are in the folder where the files are stored to avoid doubling

    files = glob.glob(r'data\Network\travel_time\developments/*')
    for f in files:
        os.remove(f)

    points = gpd.read_file(r"data\Network\processed\points_attribute.gpkg")
    # Need the node id as ID_point
    #points = points[points["intersection"] == 0]
    points = points.cx[frame[0]:frame[2], frame[1]:frame[3]]

    generated_points = gpd.read_file(r"data\Network\processed\generated_nodeset.gpkg")

    # travel speed
    raster_file = r"data\Network\OSM_tif\speed_limit_raster.tif"
    # should change lake speed to 0
    # and other area to slightly higher speed to other land covers
    with rasterio.open(raster_file) as dataset:
        raster_data = dataset.read(1)  # Assumes forbidden cells are marked with 1 or another distinct value
        transform = dataset.transform

        # Iterate over all developments
        for index, row in generated_points.iterrows():
            geometry = row.geometry  # Access geometry
            id_new = row['ID_new']  # Access value in ID_new column
            print(f"Development {id_new}")

            # Create a new row to add to the target GeoDataFrame
            new_row = {'geometry': geometry, 'intersection': 0, 'ID_point': 9999} # , 'ID_new': id_new

            # Append new row to target_gdf
            temp_points = points.copy()
            #temp_points = temp_points.append(new_row, ignore_index=True)
            # geometries.append(tempgeom)
            temp_points = gpd.GeoDataFrame(pd.concat([temp_points, pd.DataFrame(pd.Series(new_row)).T], ignore_index=True))

            # Convert real-world coordinates to raster indices
            sources_indices = [~transform * (x, y) for x, y in zip(temp_points.geometry.x, temp_points.geometry.y)]
            sources_indices = [(int(y), int(x)) for x, y in sources_indices]

            sources_indices, idx_correct = match_access_point_on_highway(sources_indices, raster_data)
            # Remove all cells that contain highway
            # raster_data[raster_data > 90] = 50

            start = time.time()
            # Convert raster to graph
            graph = raster_to_graph(raster_data)

            # Get both path lengths and paths
            distances, paths = nx.multi_source_dijkstra(G=graph, sources=sources_indices, weight='weight')
            end = time.time()
            print(f"Initialize graph and running dijkstra: {end - start} sec.")

            # Initialize empty rasters for path lengths and source coordinates
            path_length_raster = np.full(raster_data.shape, np.nan)

            # Initialize an empty raster with np.nan and dtype float
            temp_raster = np.full(raster_data.shape, np.nan, dtype=float)
            # Change the dtype to object
            source_coord_raster = temp_raster.astype(object)

            # Populate the raster
            for node, path in paths.items():
                y, x = node
                path_length_raster[y, x] = distances[node]

                if path:  # Check if path is not empty
                    source_y, source_x = path[0]  # First element of the path is the source
                    source_coord_raster[y, x] = (source_y, source_x)

            # Save the path length raster
            with rasterio.open(
                    fr'data\Network\travel_time\developments\dev{id_new}_travel_time_raster.tif', 'w',
                    driver='GTiff',
                    height=path_length_raster.shape[0],
                    width=path_length_raster.shape[1],
                    count=1,
                    dtype=path_length_raster.dtype,
                    crs=dataset.crs,
                    transform=transform
            ) as new_dataset:
                new_dataset.write(path_length_raster, 1)

            # Inverse transform to convert CRS coordinates to raster indices
            inv_transform = ~transform

            # Convert the geometry coordinates to raster indices
            temp_points['raster_x'], temp_points['raster_y'] = zip(*temp_points['geometry'].apply(lambda geom: inv_transform * (geom.x, geom.y)))
            temp_points["raster_y"] = temp_points["raster_y"].apply(lambda x: int(np.floor(x)))
            temp_points["raster_x"] = temp_points["raster_x"].apply(lambda x: int(np.floor(x)))

            # Create a dictionary to map raster indices to ID_point
            index_to_id = {(row['raster_y'], row['raster_x']): row['ID_point'] for _, row in temp_points.iterrows()}

            # Iterate over the source_coord_raster and replace coordinates with ID_point
            # Assuming new_array is your 2D array of coordinates and matched_dict is your dictionary
            for (y, x), coord in np.ndenumerate(source_coord_raster):
                if coord in idx_correct:
                    source_coord_raster[y, x] = idx_correct[coord]

            for (y, x), source_coord in np.ndenumerate(source_coord_raster):
                if source_coord in index_to_id:
                    source_coord_raster[y, x] = index_to_id[source_coord]

            # Make sure there are no tuples as point ID
            for (y, x), value in np.ndenumerate(source_coord_raster):
                # Check if the value is a tuple or an array (or another iterable except strings)
                if isinstance(value, (tuple, list, np.ndarray)):
                    # Keep only the first value of the tuple/array
                    source_coord_raster[y, x] = value[0]
                    print(f"Index ({value}) replace by {value[0]}")
                elif np.isnan(value):
                    source_coord_raster[y, x] = -1
                    pass

            # Convert the array to a float data type
            source_coord_raster = source_coord_raster.astype(float)
            # Set NaN values to a specific NoData value, e.g., -1
            source_coord_raster[np.isnan(source_coord_raster)] = -1

            path_id_raster = fr'data\Network\travel_time\developments\dev{id_new}_source_id_raster.tif'
            with rasterio.open(
                path_id_raster, 'w',
                driver='GTiff',
                height=source_coord_raster.shape[0],
                width=source_coord_raster.shape[1],
                count=1,
                dtype=source_coord_raster.dtype,
                crs=dataset.crs,
                transform=transform
                ) as new_dataset:
                    new_dataset.write(source_coord_raster, 1)

            # get Voronoi polygons in vector data as gpd df
            gdf_polygon = raster_to_polygons(path_id_raster)
            # print(gdf_polygon.head(10).to_string())
            gdf_polygon.to_file(fr"data\Network\travel_time\developments\dev{id_new}_Voronoi.gpkg")
                # how to get the inputs? nodes in which reference system, weights automatically?
                # how to get the coordinates of the closest point?

                # tif with travel time
                # tif with closest point
    return


def get_voronoi_frame(polygons_gdf):
    margin = 100
    points_gdf = gpd.read_file(r"data\Network\processed\points_corridor_attribute.gpkg")
    points_gdf = points_gdf[points_gdf["intersection"] == 0]

    points_all = gpd.read_file(r"data\Network\processed\points.gpkg")
    points_all.crs = "epsg:2056"
    points_all = points_all[points_all["intersection"] == 0]

    # union of all polygons from points
    # get all polygons touching it
    # get its extrem values

    # Step 1: Identify polygons containing points
    points_gdf = points_gdf.drop(columns=["index_right"])
    polygons_with_points = gpd.sjoin(polygons_gdf, points_gdf, predicate='contains').drop_duplicates(
        subset=polygons_gdf.index.name)
    polygons_with_points = polygons_with_points[["ID_point", "geometry"]]
    polygons_with_points = polygons_with_points.drop_duplicates()
    # Use unary_union to union all geometries into a single geometry
    #polygons_with_points = unary_union(polygons_with_points['geometry'])
    #polygons_with_points = gpd.GeoDataFrame(geometry=[polygons_with_points], crs="epsg:2056")
    #polygons_with_points.to_file(r"data\Network\processed\ppg.gpkg")

    # Step 2: Find polygons touching the identified set
    # Add custom suffixes to avoid naming conflicts
    touching_polygons = gpd.sjoin(polygons_gdf, polygons_with_points, how='inner', predicate='touches', lsuffix='left',
                                  rsuffix='_right')

    # Combine the identified polygons and the ones touching them
    #combined_polygons = pd.concat([polygons_with_points, touching_polygons]).drop_duplicates(subset=polygons_gdf.index.name)

    # Step 3: Extract points contained in the combined set of polygons

    points_in_polygons = gpd.sjoin(points_all, touching_polygons, predicate='within', lsuffix='_l',
                                  rsuffix='r')
    points_in_polygons = points_in_polygons[["geometry", "index_r"]]
    points_in_polygons = points_in_polygons.drop_duplicates()

    # Step 4: Calculate extreme values
    xmin, ymin, xmax, ymax = points_in_polygons.total_bounds

    return [xmin-margin, ymin-margin, xmax+margin, ymax+margin]



































































import math
import sys
import os
import zipfile
import timeit
import time
from data_import import *

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
import scipy.io
from scipy.interpolate import griddata
from scipy.optimize import minimize, Bounds, least_squares
import rasterio
from rasterio.transform import from_origin
from rasterio.features import geometry_mask, shapes, rasterize
from shapely.geometry import Point, Polygon, box, shape, MultiPolygon, mapping
from shapely.ops import unary_union
from pyproj import Transformer
from rasterio.mask import mask
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import networkx as nx
from itertools import islice
import requests
from shapely.geometry import LineString
import pyproj  # For coordinate transformation
from googlemaps import Client as GoogleMapsClient
from rasterio.warp import reproject, Resampling
from shapely.geometry import mapping

# Initialize UTM to Lat/Lng transformer (replace with your UTM zone if needed)
utm_proj = pyproj.CRS("EPSG:32632")  # For UTM zone 32N (adjust if needed)
latlng_proj = pyproj.CRS("EPSG:4326")  # WGS84 Lat/Lng

def define_rail_network():
    # Load the GeoDataFrame from the GeoPackage
    nw_gdf = gpd.read_file(r"data/temp/network_railway-services.gpkg")

    # Initialize a directed graph
    G = nx.DiGraph()

    # Iterate over each row in the GeoDataFrame to add nodes and edges
    for _, row in nw_gdf.iterrows():
        # Define the nodes (stations) and edge attributes
        from_node = row['FromNode']
        to_node = row['ToNode']
        from_station = row['FromStation']
        to_station = row['ToStation']
        service = row['Service']
        direction = row['Direction']
        weight = row['TravelTime'] + row['InVehWait']
        frequency = row['Frequency']
    
        # Extract start and end points of the LineString geometry
        line_geom = row['geometry']
        from_geometry = line_geom.coords[0]  # Start point of the LineString
        to_geometry = line_geom.coords[-1]   # End point of the LineString

        # Add the nodes if they don't already exist in the graph
        if not G.has_node(from_node):
            G.add_node(from_node, station=from_station, geometry=from_geometry)
        if not G.has_node(to_node):
            G.add_node(to_node, station=to_station, geometry=to_geometry)
        
        # Add a directed edge between the nodes with the specified attributes
        G.add_edge(from_node, to_node, service=service, direction=direction, weight=weight, frequency=frequency, geometry=line_geom)
    
    return G

def calculate_od_matrix_with_penalties(G):
    # Initialize an empty list to collect OD records
    od_records = []

    # Loop over each pair of nodes in the graph
    for origin in G.nodes:
        origin_station = G.nodes[origin]['station']  # Get the station name for the origin
        # Use Dijkstra's algorithm to find shortest paths from the origin node
        paths = nx.single_source_dijkstra_path(G, origin, weight='weight')
        travel_times = nx.single_source_dijkstra_path_length(G, origin, weight='weight')
        
        for destination, path in paths.items():
            if origin != destination:
                destination_station = G.nodes[destination]['station']  # Get the station name for the destination
                
                # Calculate the number of unique lines used along the path
                lines_used = []
                total_travel_time = 0
                path_geometry = []
                for i in range(len(path) - 1):
                    edge_data = G.get_edge_data(path[i], path[i + 1])
                    total_travel_time += edge_data['weight']
                    lines_used.append(edge_data['service'])

                    # Collect the geometry of each edge in the path
                    path_geometry.append(edge_data['geometry'])
                
                # Combine geometries into a single LineString
                full_path_geometry = LineString([pt for geom in path_geometry for pt in geom.coords])
                
                # Calculate the total travel time, adding 5 minutes for each line change beyond the first
                num_lines_used = len(set(lines_used))
                penalty_time = (num_lines_used - 1) * 5
                adjusted_travel_time = total_travel_time + penalty_time

                # Add to the list of OD records
                od_records.append({
                    'Origin': origin,
                    'OriginStation': origin_station,
                    'Destination': destination,
                    'DestinationStation': destination_station,
                    'FastestTravelTime': total_travel_time,
                    'NumLinesUsed': num_lines_used,
                    'TotalTravelTime': adjusted_travel_time,
                    'Geometry': full_path_geometry
                })

    # Convert the list of records to a DataFrame
    od_matrix = pd.DataFrame(od_records)
    return od_matrix

def utm_to_latlng(easting, northing):
    """Convert UTM coordinates (easting, northing) to latitude and longitude."""
    lon, lat = pyproj.transform(utm_proj, latlng_proj, easting, northing)
    return lat, lon


# Function to calculate travel time from Google Maps API
def get_google_travel_time(origin_coords, destination_coords, api_key, mode="transit"):
    """Fetch the travel time using Google Maps API."""
    gmaps = GoogleMapsClient(key=api_key)
    try:
        # Request travel time via train (transit mode)
        directions = gmaps.directions(origin_coords, destination_coords, mode=mode, transit_mode="train")
        
        if directions:
            # Extract travel time from the API response (convert seconds to minutes)
            travel_time = directions[0]['legs'][0]['duration']['value'] / 60
            return travel_time
        else:
            return None
    except Exception as e:
        return None

# Function to calculate travel times for all rows in od_matrix
def calculate_travel_times(od_matrix, api_key):

    # Initialize UTM to Lat/Lng transformer (replace with your UTM zone if needed)
    utm_proj = pyproj.Proj(init="epsg:32632")  # For UTM zone 32N (adjust if needed)
    latlng_proj = pyproj.Proj(init="epsg:4326")  # WGS84 Lat/Lng
    """Calculate and update travel times in the GeoDataFrame using Google Maps API."""
    od_matrix['GoogleTravelTime'] = None  # Initialize the column for travel times

    # Iterate through each row to calculate travel time
    for idx, row in od_matrix.iterrows():
        line_geom = row['Geometry']  # Geometry column, ensure correct name

        if isinstance(line_geom, LineString):
            # Extract the origin (start) and destination (end) coordinates (Easting, Northing)
            origin_coords_utm = line_geom.coords[0]
            destination_coords_utm = line_geom.coords[-1]

            # Convert UTM coordinates to Lat/Lng
            origin_coords_latlng = utm_to_latlng(origin_coords_utm[0], origin_coords_utm[1])
            destination_coords_latlng = utm_to_latlng(destination_coords_utm[0], destination_coords_utm[1])

            # Format the coordinates as 'lat,lng'
            origin_coords = f"{origin_coords_latlng[0]},{origin_coords_latlng[1]}"
            destination_coords = f"{destination_coords_latlng[0]},{destination_coords_latlng[1]}"

            # Fetch the travel time using the Google Maps API
            travel_time = get_google_travel_time(origin_coords, destination_coords, api_key)

            # Update the GeoDataFrame with the calculated travel time if available
            if travel_time is not None:
                od_matrix.at[idx, 'GoogleTravelTime'] = travel_time

            # Adjust sleep to prevent hitting API rate limits (Google recommends 1 request per second)
            time.sleep(1)

    return od_matrix


def import_elevation_model_old():
    # Replace with your actual file path list
    file_paths = ['path/to/zip1', 'path/to/zip2', ...]
    # "data/elevation_model/ch.swisstopo.swissalti3d-pivq0Jb7.csv"

    # Temporary directory for extracted files
    temp_dir = "temp_xyz"
    os.makedirs(temp_dir, exist_ok=True)

    # Loop through your file paths
    for file_path in file_paths:
        # Here you would download the file if 'file_path' is a URL
        # For example, using requests.get if it's an HTTP link
        # Extract the ZIP file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Now read the extracted XYZ file (assuming there's only one file in the ZIP)
        xyz_file = os.listdir(temp_dir)[0]  # This is not robust - only works if there's one file in the ZIP
        xyz_path = os.path.join(temp_dir, xyz_file)
        data = pd.read_csv(xyz_path, delim_whitespace=True, names=['X', 'Y', 'Z'])

        # Perform your data processing here
        # For example, creating a grid to interpolate onto
        # Define your grid spacing for the raster (this is where you coarsen the resolution)
        grid_x, grid_y = np.mgrid[data['X'].min():data['X'].max():100,
                         data['Y'].min():data['Y'].max():100]  # 100 can be replaced with the desired spacing

        # Interpolate using griddata - this creates the raster from the point data
        grid_z = griddata((data['X'], data['Y']), data['Z'], (grid_x, grid_y), method='nearest')

        # The rest of the code goes here to create and resample the raster using rasterio...

    # Clean up the temporary directory
    os.rmdir(temp_dir)
    return


def construction_costs(highway, tunnel, bridge, ramp):
    """
    highway = 11000 # CHF / m
    tunnel = 300000 # CHF / m
    bridge = 2600 * 22 # CHF / m
    ramp = 100000000 # CHF
    """

    bridge_small_river = 0  # m
    bridge_medium_river = 25  # m
    bridge_big_river = 50  # m
    bridge_rail = 25  # m

    # generated_links_gdf = gpd.read_file(r"data/Network/processed/new_links.shp")
    # generated_links_gdf = gpd.read_file(r"data/Network/processed/new_links_realistic.gpkg")
    generated_links_gdf = gpd.read_file(r"data/Network/processed/new_links_realistic_tunnel_adjusted.gpkg")
    # generated_links_gdf = gpd.read_file(r"data/Network/processed/new_links_realistic_tunnel.gpkg")

    # Aggreagte by development over all tunnels and bridges
    generated_links_gdf = generated_links_gdf.fillna(0)
    generated_links_gdf = generated_links_gdf.groupby(by="ID_new").agg(
        {"ID_current": "first", "total_tunnel_length": "sum", "total_bridge_length": "sum", "geometry": "first"})
    # Convert the index into a column
    generated_links_gdf = generated_links_gdf.reset_index()
    # Convert the DataFrame back to a GeoDataFrame
    generated_links_gdf = gpd.GeoDataFrame(generated_links_gdf, geometry='geometry', crs="epsg:2056")

    # Costs due to bridges to cross water
    generated_links_gdf = bridges_crossing_water(generated_links_gdf)

    # Costs due to bridges to cross railways
    generated_links_gdf = rail_crossing(generated_links_gdf)

    # Replace nan values by 0
    generated_links_gdf = generated_links_gdf.fillna(0)

    generated_links_gdf["bridge"] = generated_links_gdf["count_rail"] * bridge_rail + generated_links_gdf[
        "klein"] * bridge_small_river + generated_links_gdf["mittel"] * bridge_medium_river + generated_links_gdf[
                                        "gross"] * bridge_big_river

    # Sum amount of tunnel and bridges
    generated_links_gdf["bridge_len"] = generated_links_gdf["total_bridge_length"] + generated_links_gdf["bridge"]
    generated_links_gdf["tunnel_len"] = generated_links_gdf["total_tunnel_length"]
    generated_links_gdf["hw_len"] = generated_links_gdf.geometry.length - generated_links_gdf["bridge_len"] - \
                                    generated_links_gdf["tunnel_len"]

    # Drop unseless columns
    generated_links_gdf = generated_links_gdf.drop(
        columns=["gross", "klein", "mittel", "count_rail", "bridge", "total_bridge_length", "total_bridge_length"])
    generated_links_gdf.to_file(r"data/Network/processed/links_with_geometry_attributes.gpkg")

    generated_links_gdf["cost_path"] = generated_links_gdf["hw_len"] * highway
    generated_links_gdf["cost_bridge"] = generated_links_gdf["bridge_len"] * bridge
    generated_links_gdf["cost_tunnel"] = generated_links_gdf["tunnel_len"] * tunnel
    generated_links_gdf["building_costs"] = generated_links_gdf["cost_path"] + generated_links_gdf["cost_bridge"] + \
                                            generated_links_gdf["cost_tunnel"] + ramp

    # Only keep relevant columns
    generated_links_gdf = generated_links_gdf[
        ["ID_current", "ID_new", "geometry", "cost_path", "cost_bridge", "cost_tunnel", "building_costs"]]
    generated_links_gdf.to_file(r"data/costs/construction.gpkg")

    return


def maintenance_costs(duration, highway, tunnel, bridge, structural):
    generated_links_gdf = gpd.read_file(r"data/Network/processed/links_with_geometry_attributes.gpkg")
    # print(generated_links_gdf.head(10).to_string())

    generated_links_gdf["operational_maint"] = duration * (
                generated_links_gdf["hw_len"] * highway + generated_links_gdf["tunnel_len"] * tunnel +
                generated_links_gdf["bridge_len"] * bridge)

    costs_links = gpd.read_file(r"data/costs/construction.gpkg")
    costs_links["structural_maint"] = costs_links["building_costs"] * structural * duration

    # generated_links_gdf["structural_maint"] = duration * generated_links_gdf["bridge_len"] * structural

    # Merge column "structural_maint" to generated links using ID_new
    generated_links_gdf = generated_links_gdf.merge(costs_links[["ID_new", "structural_maint"]], on="ID_new",
                                                    how="left")
    generated_links_gdf["maintenance"] = generated_links_gdf["operational_maint"] + generated_links_gdf[
        "structural_maint"]

    # Only keep df with ID_new and maintenance costs
    generated_links_gdf = generated_links_gdf[["ID_new", "geometry", "maintenance"]]

    # Store the modified GeoDataFrame
    generated_links_gdf.to_file(r"data/costs/maintenance.gpkg", driver='GPKG')
    print(generated_links_gdf.head(10).to_string())
    return


def bridges_crossing_water(links):
    # crosing things as water
    rivers = gpd.read_file(r"data/landuse_landcover/landcover/water_ch/Typisierung_LV95/typisierung.gpkg")
    rivers = rivers[["ABFLUSS", "geometry"]]

    # Use spatial join to find crossings - this will add an index to each street where it intersects a river
    intersections = gpd.sjoin(links, rivers, how="left", predicate='intersects')

    # Now, count the number of intersections for each street
    # Assuming the 'streets_gdf' has a unique identifier for each street in the 'street_id' column
    crossing_counts = intersections.groupby(['ID_new', "ABFLUSS"]).count()
    crossing_counts = crossing_counts[["ID_current"]].rename(columns={"ID_current": "count"})
    crossing_counts = crossing_counts.reset_index()
    # Now pivot 'Abfluss' to become columns and 'count' as values
    pivot_df = crossing_counts.pivot(index='ID_new', columns='ABFLUSS', values='count')
    # Replace NaN with 0 since you want counts to default to 0 where there's no data
    pivot_df = pivot_df.fillna(0)

    links = links.merge(pivot_df, on='ID_new', how='left')

    return links


def rail_crossing(links):
    # Get all the layers from the .gdb file
    # layers = fiona.listlayers(r"data/landuse_landcover/landcover/railway/schienennetz_2056_de.gdb")
    # print(layers)
    rail = gpd.read_file(r"data/landuse_landcover/landcover/railway/schienennetz_2056_de.gdb", layer='Netzsegment')

    # Use spatial join to find crossings - this will add an index to each street where it intersects a river
    intersections = gpd.sjoin(links, rail, how="left", predicate='intersects')

    # Now, count the number of intersections for each street
    # Assuming the 'streets_gdf' has a unique identifier for each street in the 'street_id' column
    crossing_counts = intersections.groupby(['ID_new']).count()
    crossing_counts = crossing_counts[["ID_current"]].rename(columns={"ID_current": "count_rail"})

    links = links.merge(crossing_counts, on='ID_new', how='left')
    links["count_rail"] = links["count_rail"].fillna(0)

    return links


def land_tb_reallocated(links, buffer_distance):
    zones = gpd.read_file(r"data/landuse_landcover/processed/partly_protected.gpkg")
    print("Zones", zones.name.unique())

    buffer = links.copy()
    # Create a buffer around each line
    links['buffer'] = buffer.geometry.buffer(buffer_distance)
    # links = links.set_geometry(col="buffer")

    # Initialize the columns for the areas of overlap
    for mp_id in zones['name'].unique():
        links[f'{mp_id}_area'] = 0.0

    # Calculate the overlapping area for each polygon with each multipolygon
    for idx, multipolygon in zones.iterrows():
        # Get the current multipolygon_id
        mp_id = multipolygon['name']

        # Calculate the intersection with each polygon in A
        # This returns a GeoSeries of the intersecting geometries
        intersections = links['buffer'].intersection(multipolygon['geometry'])

        # Calculate the area of each intersection
        links[f'{mp_id}_area'] = intersections.area

    links = links.drop(columns="buffer")

    return links


def externalities_costs(ce_highway, ce_tunnel, realloc_forest, realloc_FFF, realloc_dry_meadow, realloc_period,
                        nat_fragmentation, fragm_period, nat_loss_habitat, habitat_period):
    # Import dataframe with links geometries
    generated_links_gdf = gpd.read_file(r"data/Network/processed/links_with_geometry_attributes.gpkg")
    # Replace nan values by 0
    generated_links_gdf = generated_links_gdf.fillna(0)
    ########################################3
    # Climate effects
    """
    highway = 2325 # CHF/m/50a
    tunnel = 3137 # CHF/m/50a
    """
    ce_bridge = ce_tunnel

    generated_links_gdf["climate_cost"] = generated_links_gdf["hw_len"] * ce_highway + generated_links_gdf[
        "tunnel_len"] * ce_tunnel + generated_links_gdf["bridge_len"] * ce_bridge

    ############################
    # Land reallocation
    """
    periode_ecosystem = 50
    realloc_forest = 0.889  # CHF/m2/a
    FFF = 0.075  # CHF/m2/a
    dry_meadow = 0.075  # CHF/m2/a
    """

    # Import generated tunnels
    tunnels_gdf = gpd.read_file(r"data/Network/processed/edges_tunnels.gpkg")

    # Remove tunnel from link geometry
    buffer_distance = 20

    # Iterate over the links
    for idx, link in generated_links_gdf.iterrows():
        # Find the corresponding tunnel
        corresponding_tunnels = tunnels_gdf[tunnels_gdf['link_id'] == link['ID_new']]

        if not corresponding_tunnels.empty:
            # Create a buffer around each tunnel geometry and then combine them
            all_tunnel_buffers = corresponding_tunnels.geometry.buffer(buffer_distance).unary_union

            # Subtract the combined tunnel buffers from the link geometry
            new_link_geometry = link.geometry.difference(all_tunnel_buffers)

            # Update the link geometry
            generated_links_gdf.at[idx, 'geometry'] = new_link_geometry

    # Reallocation of land
    buffer_distance = 25
    generated_links_gdf = land_tb_reallocated(generated_links_gdf, buffer_distance)

    generated_links_gdf["land_realloc"] = realloc_period * (
            generated_links_gdf["wald_area"] * realloc_forest + generated_links_gdf[
        "fruchtfolgeflaeche_area"] * realloc_FFF + (
                    generated_links_gdf["trockenweiden_area"] + generated_links_gdf[
                "trockenlandschaften_area"] * realloc_dry_meadow))

    ###########################################
    # Nature and landscape
    """
    nat_fragmentation = 155.6  # CHF/m/a
    nat_loss_habitat = 31.6  # CHF/m/a
    """
    generated_links_gdf["nature"] = generated_links_gdf["hw_len"] * (
                nat_fragmentation * fragm_period + nat_loss_habitat * habitat_period)

    # df_temp["externality_costs"] = df_temp["climate_cost"] + df_temp["nature"]
    # df_temp["building_costs"] = df_temp["building_costs"] + df_temp["land_realloc"]

    generated_links_gdf = generated_links_gdf[
        ["ID_new", "ID_current", "geometry", "climate_cost", "land_realloc", "nature"]]
    # print(generated_links_gdf.head(10).to_string())
    generated_links_gdf.to_file(r"data/costs/externalities.gpkg")

    return


def noise_costs(years, unit_costs, boundaries):
    # Input data with generated edges as linestrings
    edges = gpd.read_file(r"data/Network/processed/links_with_geometry_attributes.gpkg")

    # For each edge do a buffer around the linestring with distances 0-10, 10-20, 20-40, 40-80, 80-160, 160-320, 320-640, 640-1280, 1280-2560 meters
    # Define variables of boundaries
    """
    boundaries = [0, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]
    # Define unit costs for each buffer zone (CHF/p/a)
    unit_costs = [7615, 5878, 4371, 3092, 2039, 1210, 604, 212, 19]
    """
    # Calculate the amount of inhabitants in each buffer zone
    # Iterate over all scenarios
    # Input data of scenarios import directly tif files
    scenario_path = ['s1_pop.tif', 's2_pop.tif', 's3_pop.tif']
    for path in scenario_path:
        with rasterio.open(fr"data/independent_variable/processed/scenario/{path}") as scenario_tif:
            trip_tif = scenario_tif.read(1)

            edges_temp = gpd.GeoDataFrame()

            for i in range(len(boundaries) - 1):
                outer_buffer = edges.geometry.buffer(boundaries[i + 1])
                inner_buffer = edges.geometry.buffer(boundaries[i])
                edges_temp[f'noise_{i}'] = outer_buffer.difference(inner_buffer)

            total_costs = []

            for index, row in edges_temp.iterrows():
                cost_per_edge = 0

                for i, unit_cost in zip(range(len(boundaries) - 1), unit_costs):
                    if not row[f'noise_{i}'].is_empty:
                        geom = [mapping(row[f'noise_{i}'])]
                        out_image, out_transform = mask(scenario_tif, geom, crop=True)
                        # Replace nan by 0
                        out_image = np.nan_to_num(out_image)
                        population_sum = out_image.sum()
                        cost_per_edge += population_sum * unit_cost
                total_costs.append(cost_per_edge)

            edges[f"noise_{path[:2]}"] = total_costs  # costs per link and year
            edges[f"noise_{path[:2]}"] = edges[f"noise_{path[:2]}"] * years  # costs per link and all years

    edges = edges[["ID_current", "ID_new", "geometry", "noise_s1", "noise_s2", "noise_s3"]]
    # Store the modified GeoDataFrame
    edges.to_file(r"data/costs/noise.gpkg", driver='GPKG')
    return


def accessibility_developments(costs, VTT_h, duration):
    """
    # Import travel time from each cell
    tt = 0
    # Import closest access point for each cell and development
    nearest_access = 0
    # Import scenario values
    scen = 0

    # Get amount of highway trips per day and inhabitant
    trip_generation = 1.14 # trip/p/d
    duration = 50 #years
    VTT = 30 # CHF/h

    # Get amount of trips per cell
    trip_cell = trip_prob * scen * duration * tt

    # Get travel time per cell
    time_cell = trip_cell * VTT

    # Aggregate over entire area
    poly = time_cell.agg(nearest_access)
    """

    # File paths and trip_generation
    # travel_time_path = r"data/Network/travel_time/travel_time_raster.tif"
    scenario_path = ['s1_pop.tif', 's2_pop.tif', 's3_pop.tif']
    voronoi_path = r"data/Voronoi/voronoi_developments_tt_values.shp"

    trip_generation_day_cell = 1.14  # trip/p/d
    # duration = 30  # years
    duration_d = duration * 365
    trip_generation = trip_generation_day_cell * duration_d
    # VTT_h = 29.9  # CHF/h
    VTT = VTT_h / 60 / 60  # CHF/sec
    print(f"VTT: {VTT}")

    # Load TIF B and polygons
    voronoi_gdf = gpd.read_file(voronoi_path)
    # print(voronoi_gdf["ID_develop"].unique())

    # Process each TIF A
    for path in scenario_path:
        with rasterio.open(fr"data/independent_variable/processed/scenario/{path}") as scenario_tif:
            print(path)
            # Multiply TIF A by trip_generation
            trip_tif = scenario_tif.read(1) * trip_generation

            for index, row in voronoi_gdf.iterrows():
                # print(row["ID_develop"])
                id_development = row["ID_develop"]

                # If the geometry is a MultiPolygon, convert it to a list of Polygons
                if isinstance(row['geometry'], MultiPolygon):
                    polygons = [poly for poly in row['geometry'].geoms]
                else:
                    polygons = [row['geometry']]

                # Extract data for the polygon area from TIF A
                trip_mask = geometry_mask(polygons, transform=scenario_tif.transform, invert=True,
                                          out_shape=(scenario_tif.height, scenario_tif.width))
                # Apply the mask to the raster data

                # trip_filled = np.full(trip_mask.shape, 1.3)
                # Overlay the raster data onto the 1.3-filled array
                # Only replace where tt_mask is False (i.e., within the raster extent)
                # trip_filled[~trip_mask] = trip_tif[~trip_mask]

                trip_masked = trip_tif * trip_mask

                with rasterio.open(
                        fr"data/Network/travel_time/developments/dev{id_development}_travel_time_raster.tif") as travel_time:
                    # data/Network/travel_time/developments/dev2_travel_time_raster.tif"
                    ###################################################################################
                    # travel_time = rasterio.open(travel_time_path)
                    tt_tif = travel_time.read(1)
                    tt_mask = geometry_mask(polygons, transform=travel_time.transform, invert=True,
                                            out_shape=(travel_time.height, travel_time.width))

                    tt_masked = tt_tif * tt_mask

                    # Extract data for the polygon area from TIF B
                    # data_B_polygon = get_data_from_tif(tif_B, row['geometry'])

                    # Multiply values of TIF A and B
                    total_tt = tt_masked * trip_masked

                    # Sum values in the polygon area
                    sum_value = np.nansum(total_tt)

                    # Store the sum in the GeoDataFrame
                    column_name = path.split('.')[0]  # Adjust as needed
                    voronoi_gdf.at[index, column_name] = sum_value * VTT

    # Save the modified GeoDataFrame
    voronoi_gdf.to_file(r"data/Voronoi/voronoi_developments_local_accessibility.gpkg", driver='GPKG')
    # print(voronoi_gdf.head(50).to_string())
    voronoi_gdf = voronoi_gdf.drop(columns=['geometry'])
    grouped_sum = voronoi_gdf.groupby('ID_develop').sum()
    grouped_sum = grouped_sum[["s1_pop", "s2_pop", "s3_pop"]]
    costs = costs[["s1_pop", "s2_pop", "s3_pop"]]
    print(grouped_sum.head().to_string())

    grouped_sum["local_s1"] = costs["s1_pop"] - grouped_sum["s1_pop"]
    grouped_sum["local_s2"] = costs["s2_pop"] - grouped_sum["s2_pop"]
    grouped_sum["local_s3"] = costs["s3_pop"] - grouped_sum["s3_pop"]
    print(grouped_sum.head().to_string())
    # print(costs.head().to_string())
    grouped_sum = grouped_sum.reset_index().rename(columns={'index': 'ID_development'})

    # Optionally, you can rename the new column (which will be named 'index' by default)
    # gr = df.reset_index()

    # Save the DataFrame as a CSV file
    grouped_sum.to_csv('data/costs/local_accessibility.csv', index=False)
    # grouped_sum.to(r"data/costs/local_accessibility.gpkg", driver='GPKG')

    return


def accessibility_status_quo(VTT_h, duration):
    # File paths and trip_generation
    travel_time_path = r"data/Network/travel_time/travel_time_raster.tif"
    scenario_path = ['s1_pop.tif', 's2_pop.tif', 's3_pop.tif']
    voronoi_path = r"data/Network/travel_time/Voronoi_statusquo.gpkg"

    trip_generation_day_cell = 1.14  # trip/p/d
    # duration = 30  # years
    duration_d = duration * 365
    trip_generation = trip_generation_day_cell * duration_d
    # VTT_h = 30.6  # CHF/h
    VTT = VTT_h / 60 / 60  # CHF/h

    # Load TIF B and polygons
    voronoi_gdf = gpd.read_file(voronoi_path)

    # Process each TIF A
    for path in scenario_path:
        with rasterio.open(fr"data/independent_variable/processed/scenario/{path}") as scenario_tif:
            # Multiply TIF A by trip_generation
            trip_tif = scenario_tif.read(1) * trip_generation
            print(trip_tif.shape)

            for index, row in voronoi_gdf.iterrows():

                # If the geometry is a MultiPolygon, convert it to a list of Polygons
                if isinstance(row['geometry'], MultiPolygon):
                    polygons = [poly for poly in row['geometry'].geoms]
                else:
                    polygons = [row['geometry']]

                # Extract data for the polygon area from TIF A
                trip_mask = geometry_mask(polygons, transform=scenario_tif.transform, invert=True,
                                          out_shape=(scenario_tif.height, scenario_tif.width))
                # Apply the mask to the raster data

                # trip_filled = np.full(trip_mask.shape, 1.3)
                # Overlay the raster data onto the 1.3-filled array
                # Only replace where tt_mask is False (i.e., within the raster extent)
                # trip_filled[~trip_mask] = trip_tif[~trip_mask]

                trip_masked = trip_tif * trip_mask

                with rasterio.open(travel_time_path) as travel_time:
                    # travel_time = rasterio.open(travel_time_path)
                    tt_tif = travel_time.read(1)
                    tt_mask = geometry_mask(polygons, transform=travel_time.transform, invert=True,
                                            out_shape=(travel_time.height, travel_time.width))

                    tt_masked = tt_tif * tt_mask

                    # Extract data for the polygon area from TIF B
                    # data_B_polygon = get_data_from_tif(tif_B, row['geometry'])

                    # Multiply values of TIF A and B
                    total_tt = tt_masked * trip_masked

                    # Sum values in the polygon area
                    sum_value = np.nansum(total_tt)

                    # Store the sum in the GeoDataFrame
                    column_name = path.split('.')[0]  # Adjust as needed
                    voronoi_gdf.at[index, column_name] = sum_value * VTT

    # Save the modified GeoDataFrame
    voronoi_gdf.to_file(r"data/Voronoi/voronoi_developments_local_accessibility.gpkg", driver='GPKG')
    # print(voronoi_gdf.head(50).to_string())
    # print(voronoi_gdf.sum()["s1_pop"])
    voronoi_gdf = voronoi_gdf.drop(columns=['geometry'])
    return voronoi_gdf.sum()
"""

def nw_from_osm(limits):
    # Split the area into smaller polygons
    num_splits = 10  # Adjust this to get 1/10th of the area (e.g., 3 for a 1/9th split)
    sub_polygons = split_area(limits, num_splits)

    # Initialize the transformer between LV95 and WGS 84
    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    for i, lv95_sub_polygon in enumerate(sub_polygons):

        # Convert the coordinates of the sub-polygon to lat/lon
        lat_lon_frame = Polygon([transformer.transform(*point) for point in lv95_sub_polygon.exterior.coords])

        try:
            # Attempt to process the OSM data for the sub-polygon
            print(f"Processing sub-polygon {i + 1}/{len(sub_polygons)}")
            # G = ox.graph_from_polygon(lat_lon_frame, network_type="drive", simplify=True, truncate_by_edge=True)
            # Define a custom filter to exclude highways
            # This example excludes motorways, motorway_links, trunks, and trunk_links
            # custom_filter = '["highway"!~"motorway|motorway_link|trunk|trunk_link"]'
            # Create the graph using the custom filter
            G = ox.graph_from_polygon(lat_lon_frame, network_type="drive", simplify=True,
                                      truncate_by_edge=True)  # custom_filter=custom_filter,
            G = ox.add_edge_speeds(G)

            # Convert the graph to a GeoDataFrame
            gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
            gdf_edges = gdf_edges[["geometry", "speed_kph"]]
            gdf_edges = gdf_edges[gdf_edges["speed_kph"] <= 80]

            # Project the edges GeoDataFrame to the desired CRS (if necessary)
            gdf_edges = gdf_edges.to_crs("EPSG:2056")

            # Save only the edges GeoDataFrame to a GeoPackage
            output_filename = f"data/Network/OSM_road/sub_area_edges_{i + 1}.gpkg"
            gdf_edges.to_file(output_filename, driver="GPKG")
            print(f"Sub-polygon {i + 1} processed and saved.")

        except ValueError as e:
            # Handle areas with no nodes by logging or printing an error message
            print(f"Skipping graph in sub-polygon {i + 1} due to error: {e}")
            # Optionally, continue with the next sub-polygon or perform other error handling
            continue


def split_area(limits, num_splits):
    
    Split the given area defined by 'limits' into 'num_splits' smaller polygons.

    :param limits: Tuple of (min_x, max_x, min_y, max_y) in LV95 coordinates.
    :param num_splits: The number of splits along each axis (total areas = num_splits^2).
    :return: List of shapely Polygon objects representing the smaller areas.
    min_x, min_y, max_x, max_y = limits
    width = (max_x - min_x) / num_splits
    height = (max_y - min_y) / num_splits

    sub_polygons = []
    for i in range(num_splits):
        for j in range(num_splits):
            # Calculate the corners of the sub-polygon
            sub_min_x = min_x + i * width
            sub_max_x = sub_min_x + width
            sub_min_y = min_y + j * height
            sub_max_y = sub_min_y + height

            # Create the sub-polygon and add it to the list
            sub_polygon = box(sub_min_x, sub_min_y, sub_max_x, sub_max_y)
            sub_polygons.append(sub_polygon)

    return sub_polygons

def osm_nw_to_raster(limits):
    # Add comment

    # Folder containing all the geopackages
    gpkg_folder = "data/Network/OSM_road"

    # List all geopackage files in the folder
    gpkg_files = [os.path.join(gpkg_folder, f) for f in os.listdir(gpkg_folder) if f.endswith('.gpkg')]

    # Combine all geopackages into one GeoDataFrame
    gdf_combined = gpd.GeoDataFrame(pd.concat([gpd.read_file(f) for f in gpkg_files], ignore_index=True))
    # Assuming 'speed' is the column with speed limits
    # Convert speeds to numeric, handling non-numeric values
    gdf_combined['speed'] = pd.to_numeric(gdf_combined['speed_kph'], errors='coerce')

    # Drop NaN values or replace them with 0, depending on how you want to handle them
    # gdf_combined.dropna(subset=['speed_kph'], inplace=True)
    gdf_combined['speed_kph'].fillna(30, inplace=True)
    # print(gdf_combined.crs)
    # print(gdf_combined.head(10).to_string())
    gdf_combined.to_file('data/Network/OSM_tif/nw_speed_limit.gpkg')
    print("file stored")

    gdf_combined = gpd.read_file('data/Network/OSM_tif/nw_speed_limit.gpkg')

    # Define the resolution
    resolution = 100

    # Define the bounds of the raster (aligned with your initial limits)
    minx, miny, maxx, maxy = limits
    print(limits)

    # Compute the number of rows and columns
    num_cols = int((maxx - minx) / resolution)
    num_rows = int((maxy - miny) / resolution)

    # Initialize the raster with 4 = minimal travel speed (or np.nan for no-data value)
    # raster = np.zeros((num_rows, num_cols), dtype=np.float32)
    raster = np.full((num_rows, num_cols), 4, dtype=np.float32)

    # Define the transform
    transform = from_origin(west=minx, north=maxy, xsize=resolution, ysize=resolution)

    # lake = gpd.read_file(r"data/landuse_landcover/landcover/water_ch/Typisierung_LV95/typisierung.gpkg")
    ###############################################################################################################

    print("ready to fill")

    tot_num = num_cols * num_cols
    count = 0

    for row in range(num_rows):
        for col in range(num_cols):

            # print(row, " - ", col)
            # Find the bounds of the cell
            cell_bounds = box(minx + col * resolution,
                              maxy - row * resolution,
                              minx + (col + 1) * resolution,
                              maxy - (row + 1) * resolution)

            # Find the roads that intersect with this cell
            # print(gdf_combined.head(10).to_string())
            intersecting_roads = gdf_combined[gdf_combined.intersects(cell_bounds)]

            # Debugging print
            # print(f"Cell {row},{col} intersects with {len(intersecting_roads)} roads")

            # If there are any intersecting roads, find the maximum speed limit
            if not intersecting_roads.empty:
                max_speed = intersecting_roads['speed_kph'].max()
                raster[row, col] = max_speed

            # Print the progress
            count += 1
            progress_percentage = (count / tot_num) * 100
            sys.stdout.write(f"\rProgress: {progress_percentage:.2f}%")
            sys.stdout.flush()

    # Check for spatial overlap with the second raster and update values if necessary
    with rasterio.open(r"data/landuse_landcover/processed/unproductive_area.tif") as src2:
        unproductive_area = src2.read(1)
        if raster.shape == unproductive_area.shape:
            print("Network raster and unproductive area are overalpping")
            mask = np.logical_and(unproductive_area > 0, unproductive_area < 100)
            raster[mask] = 0
        else:
            print("Network raster and unproductive area are not overalpping!!!!!")

    with rasterio.open(
            'data/Network/OSM_tif/speed_limit_raster.tif',
            'w',
            driver='GTiff',
            height=raster.shape[0],
            width=raster.shape[1],
            count=1,
            dtype=str(raster.dtype),
            crs="EPSG:2056",
            transform=transform,
    ) as dst:
        dst.write(raster, 1)
"""

def tif_to_vector(raster_path, vector_path):
    # Step 1: Read the raster data
    with rasterio.open(raster_path) as src:
        image = src.read(1)  # Read the first band

    # Step 2: Apply threshold or classification
    # This is an example where we create a mask for all values above a threshold
    mask = image >= 0  # Define your own threshold value

    # Step 3: Convert the masked raster to vector shapes
    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) in enumerate(
        shapes(image, mask=mask, transform=src.transform)))

    # Step 4: Create Shapely polygons and generate a GeoDataFrame
    geometries = [shape(result['geometry']) for result in results]
    gdf = gpd.GeoDataFrame.from_features([
        {"geometry": geom, "properties": {"value": val}}
        for geom, val in zip(geometries, mask)
    ])

    # Save to a new Shapefile, if desired
    # gdf.to_file(vector_path)
    return gdf


def map_coordinates_to_developments():
    df_temp = gpd.read_file(r"data/Network/processed/new_links_realistic_costs.gpkg")
    points = gpd.read_file(r"data/Network/processed/generated_nodes.gpkg")
    # print(points.columns)
    # print(points.head(10).to_string())
    # print(points["ID_new"].unique())

    df_temp = df_temp.merge(points, how='left', left_on='ID_new', right_on='ID_new')
    # print(df_temp["ID_new"].unique())
    # print(df_temp.head(10).to_string())
    df_temp['geometry'] = df_temp['geometry_y']
    # todo buildin and externality not in index
    df_temp = df_temp[['ID_current', 'ID_new', 'building_costs', 'externality_costs', 'geometry']]
    df_temp["total_cost"] = df_temp["building_costs"] + df_temp["externality_costs"]
    # df_temp['geometry'] = df_temp['geometry_y'].replace('geometry_y', 'geometry')

    # df_temp = df_temp.rename({"geometry_y":"geometry"})
    # print(df_temp.head(10).to_string())
    df_temp = gpd.GeoDataFrame(df_temp, geometry="geometry")
    df_temp.to_file(r"data/costs/building_externalities.gpkg")
    return


def aggregate_costs():
    # Construction costs
    c_construction = gpd.read_file(r"data/costs/construction.gpkg")
    # Maintenance costs
    c_maintenance = gpd.read_file(r"data/costs/maintenance.gpkg")
    # Access time costs
    c_acces_time = pd.read_csv(r"data/costs/local_accessibility.csv")
    c_acces_time = c_acces_time[["ID_develop", "local_s1", "local_s2", "local_s3"]]
    # Import travel time costs
    c_tt = pd.read_csv(r"data/costs/traveltime_savings.csv")
    # Import externalities
    c_externalities = gpd.read_file(r"data/costs/externalities.gpkg")
    # Import noise costs
    c_noise = gpd.read_file(r"data/costs/noise.gpkg")

    # Rename columns to simplify further steps
    c_acces_time = c_acces_time.rename(columns={'ID_develop': 'ID_new'})
    c_tt = c_tt.rename(columns={'development': 'ID_new'}).drop(columns=["Unnamed: 0"])

    # Find common values
    common_values = set(c_construction["ID_new"]).intersection(c_acces_time["ID_new"]).intersection(
        c_tt["ID_new"]).intersection(c_externalities["ID_new"]).intersection(c_noise["ID_new"])
    print(f"Number of developments: {len(common_values)}")

    # Merge construction costs and maintenance costs
    # c_construction = c_construction.merge(c_maintenance, how='inner', on='ID_new')
    # Add acccess time costs
    # total_costs = c_construction.merge(c_acces_time, how='inner', on='ID_new')
    # Add travel time
    # total_costs = total_costs.merge(c_tt, how='inner', on='ID_new')
    # Add externalities costs
    # total_costs = total_costs.merge(c_externalities, how='inner', on='ID_new')
    # Add noise costs
    # total_costs = total_costs.merge(c_noise, how='inner', on='ID_new')

    # geom_id_map = c_maintenance.drop("maintenance",axis=1)
    total_costs = c_construction.drop("geometry", axis=1).merge(c_maintenance.drop(["geometry"], axis=1), how='inner',
                                                                on='ID_new')
    # Add acccess time costs
    total_costs = total_costs.merge(c_acces_time, how='inner', on='ID_new')
    # Add travel time
    total_costs = total_costs.merge(c_tt, how='inner', on='ID_new')
    # Add externalities costs
    total_costs = total_costs.merge(c_externalities.drop("geometry", axis=1), how='inner', on='ID_new')
    # Add noise costs
    total_costs = total_costs.merge(c_noise.drop("geometry", axis=1), how='inner', on='ID_new')

    total_costs = total_costs[['ID_new', 'cost_path', 'cost_bridge', 'cost_tunnel', 'building_costs',
                               'local_s1', 'local_s2', 'local_s3', 'tt_low', 'tt_medium', 'tt_high', 'climate_cost',
                               'land_realloc', 'nature', 'noise_s1', 'noise_s2', 'noise_s3', "maintenance"]]
    cost_columns = ['cost_path', 'cost_bridge', 'cost_tunnel', 'building_costs', 'climate_cost', 'land_realloc',
                    'nature', 'noise_s1', 'noise_s2', 'noise_s3', "maintenance"]

    # Multiply the values in these columns by -1
    for column in cost_columns:
        total_costs[column] = total_costs[column] * -1

    # Compute costs of externalities
    total_costs["externalities_s1"] = total_costs["climate_cost"] + total_costs["land_realloc"] + total_costs[
        "nature"] + total_costs["noise_s1"]
    total_costs["externalities_s2"] = total_costs["climate_cost"] + total_costs["land_realloc"] + total_costs[
        "nature"] + total_costs["noise_s2"]
    total_costs["externalities_s3"] = total_costs["climate_cost"] + total_costs["land_realloc"] + total_costs[
        "nature"] + total_costs["noise_s3"]
    total_costs["construction_maintenance"] = total_costs["building_costs"] + total_costs["maintenance"]

    # Sum externality costs
    # total_costs["externalities"] = total_costs['climate_cost'] + total_costs['land_realloc'] + total_costs['nature']
    print(total_costs.head(10).to_string())
    # Compute net benefit for each development
    total_costs["total_low"] = total_costs[["construction_maintenance", "local_s2", "tt_low", "externalities_s2"]].sum(
        axis=1)
    total_costs["total_medium"] = total_costs[
        ["construction_maintenance", "local_s1", "tt_medium", "externalities_s1"]].sum(
        axis=1)
    total_costs["total_high"] = total_costs[
        ["construction_maintenance", "local_s3", "tt_high", "externalities_s3"]].sum(
        axis=1)

    # print(total_costs.sort_values(by="total_medium", ascending=False).head(7).to_string())

    # Filter dataframe columns to store the data as csv
    total_costs[["ID_new", "total_low", "total_medium", "total_high"]].to_csv(r"data/costs/total_costs.csv")

    # Save Results a geodata
    # Map point geometries
    points = gpd.read_file(r"data/Network/processed/generated_nodes.gpkg")
    total_costs = total_costs.merge(right=points, how="left", on="ID_new")
    total_costs = gpd.GeoDataFrame(total_costs, geometry="geometry")

    # Store as file
    gpd.GeoDataFrame(total_costs).to_file(r"data/costs/total_costs.gpkg")


#######################################################################################################################
# From here on the code is destinated to compute the travel time on the highway network

def stack_tif_files(var):
    # List of your TIFF file paths
    tiff_files = [f"/s1_{var}.tif", f"/s2_{var}.tif", f"/s3_{var}.tif"]

    # Open the first file to retrieve the metadata
    with rasterio.open(r"data/independent_variable/processed/scenario" + tiff_files[0]) as src0:
        meta = src0.meta

    # Update metadata to reflect the number of layers
    meta.update(count=len(tiff_files))

    out_fp = fr"data/independent_variable/processed/scenario/scen_{var}.tif"
    # Read each layer and write it to stack
    with rasterio.open(out_fp, 'w', **meta) as dst:
        for id, layer in enumerate(tiff_files, start=1):
            with rasterio.open(r"data/independent_variable/processed/scenario" + layer) as src1:
                dst.write_band(id, src1.read(1))


# # 0 Who will drive by car
# We assume peak hour demand is generated by population residence at origin and employment opportunites at destination.
def GetCommunePopulation(y0):  # We find population of each commune.
    y0="2021"
    rawpop = pd.read_excel('data/_basic_data/KTZH_00000127_00001245.xlsx', sheet_name='Gemeinden', header=None)
    rawpop.columns = rawpop.iloc[5]
    rawpop = rawpop.drop([0, 1, 2, 3, 4, 5, 6])
    pop = pd.DataFrame(data=rawpop, columns=['BFS-NR', 'TOTAL_' + str(y0)]).sort_values(by='BFS-NR')
    popvec = np.array(pop['TOTAL_' + str(y0)])
    return popvec


def GetCommuneEmployment(y0):  # we find employment in each commune.
    rawjob = pd.read_excel('data/_basic_data/KANTON_ZUERICH_596.xlsx')
    rawjob = rawjob.loc[(rawjob['INDIKATOR_JAHR'] == y0) & (rawjob['BFS_NR'] > 0) & (rawjob['BFS_NR'] != 291)]

    # rawjob=rawjob.loc[(rawjob['INDIKATOR_JAHR']==y0)&(rawjob['BFS_NR']>0)&(rawjob['BFS_NR']!=291)]
    job = pd.DataFrame(data=rawjob, columns=['BFS_NR', 'INDIKATOR_VALUE']).sort_values(by='BFS_NR')
    jobvec = np.array(job['INDIKATOR_VALUE'])
    return jobvec


def GetHighwayPHDemandPerCommune():
    # now we extract an od matrix for oev tripps from year 2019
    # we then modify the OD matrix to fit our needs of expressing peak hour travel demand
    y0 = 2019
    rawod = pd.read_excel('data/_basic_data/KTZH_00001982_00003903.xlsx')
    communalOD = rawod.loc[
        (rawod['jahr'] == 2018) & (rawod['kategorie'] == 'Verkehrsaufkommen') & (rawod['verkehrsmittel'] == 'oev')]
    # communalOD = data.drop(['jahr','quelle_name','quelle_gebietart','ziel_name','ziel_gebietart',"kategorie","verkehrsmittel","einheit","gebietsstand_jahr","zeit_dimension"],axis=1)
    # sum(communalOD['wert'])
    # 1 Who will go on highway?
    # # # Not binnenverkehr ... removes about 50% of trips
    communalOD['wert'].loc[(communalOD['quelle_code'] == communalOD['ziel_code'])] = 0
    # sum(communalOD['wert'])
    # # Take share of OD
    # todo adapt this value
    tau = 0.13  # Data is in trips per OD combination per day. Now we assume the number of trips gone in peak hour
    # This ratio explains the interzonal trips made in peak hour as a ratio of total interzonal trips made per day.
    # communalOD['wert'] = (communalOD['wert']*tau)
    communalOD.loc[:, 'wert'] = communalOD['wert'] * tau
    # # # Not those who travel < 15 min ?  Not yet implemented.
    return communalOD


def GetODMatrix(od):
    od_ext = od.loc[(od['quelle_code'] > 9999) | (od[
                                                      'ziel_code'] > 9999)]  # here we separate the parts of the od matrix that are outside the canton. We can add them later.
    od_int = od.loc[(od['quelle_code'] < 9999) & (od['ziel_code'] < 9999)]
    odmat = od_int.pivot(index='quelle_code', columns='ziel_code', values='wert')
    return odmat


def GetCommuneShapes(raster_path):  # todo this might be unnecessary if you already have these shapes.
    communalraw = gpd.read_file(r"data/_basic_data/Gemeindegrenzen/UP_GEMEINDEN_F.shp")
    communalraw = communalraw.loc[(communalraw['ART_TEXT'] == 'Gemeinde')]
    communedf = gpd.GeoDataFrame(data=communalraw, geometry=communalraw['geometry'], columns=['BFS', 'GEMEINDENA'],
                                 crs="epsg:2056").sort_values(by='BFS')

    # Read the reference TIFF file
    with rasterio.open(raster_path) as src:
        profile = src.profile
        profile.update(count=1)
        crs = src.crs

    # Rasterize
    with rasterio.open('data/_basic_data/Gemeindegrenzen/gemeinde_zh.tif', 'w', **profile) as dst:
        rasterized_image = rasterize(
            [(shape, value) for shape, value in zip(communedf.geometry, communedf['BFS'])],
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            all_touched=False,
            dtype=rasterio.int32
        )
        dst.write(rasterized_image, 1)

    # Convert the rasterized image to a numpy array
    commune_raster = np.array(rasterized_image)

    return commune_raster, communedf

def correct_rasters_to_extent(
    empl_path, pop_path, 
    output_empl_path, output_pop_path,
    reference_boundary, resolution=100, crs="EPSG:2056"):

    """
    Corrects the raster files to match the given boundary extent and resolution for all bands.

    Args:
        empl_path (str): Path to the employment raster file.
        pop_path (str): Path to the population raster file.
        output_empl_path (str): Path to save the corrected employment raster.
        output_pop_path (str): Path to save the corrected population raster.
        reference_boundary (shapely.geometry.Polygon): Boundary polygon for cropping and masking.
        resolution (int): Resolution of the output raster in meters. Default is 100.
        crs (str): Coordinate Reference System for the output rasters. Default is "EPSG:2056".
    """
    # Determine the bounds and raster grid size
    xmin, ymin, xmax, ymax = reference_boundary.bounds
    width = int((xmax - xmin) / resolution)
    height = int((ymax - ymin) / resolution)
    transform = rasterio.transform.from_origin(xmin, ymax, resolution, resolution)

    # Convert the boundary to GeoJSON-like format for masking
    boundary_geom = [mapping(reference_boundary)]

    def process_raster(input_path, output_path):
        with rasterio.open(input_path) as src:
            # Number of bands in the raster
            num_bands = src.count

            # Initialize a 3D array for corrected raster data (bands, height, width)
            data_corrected = np.zeros((num_bands, height, width), dtype=src.dtypes[0])

            # Process each band
            for band_idx in range(1, num_bands + 1):
                band_data = np.zeros((height, width), dtype=src.dtypes[0])

                # Reproject and resample
                rasterio.warp.reproject(
                    source=rasterio.band(src, band_idx),
                    destination=band_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=rasterio.warp.Resampling.bilinear,
                )

                # Mask the raster to fit within the boundary
                mask = rasterio.features.geometry_mask(
                    boundary_geom, transform=transform, invert=True, out_shape=(height, width)
                )
                band_data[~mask] = np.nan  # Set values outside the boundary to NaN

                # Store the corrected band
                data_corrected[band_idx - 1] = band_data

            # Save the corrected raster
            profile = src.profile
            profile.update(
                driver="GTiff",
                count=num_bands,
                height=height,
                width=width,
                transform=transform,
                crs=crs,
                nodata=np.nan,
            )
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(data_corrected)

    # Process both employment and population rasters
    process_raster(empl_path, output_empl_path)
    process_raster(pop_path, output_pop_path)



###################################################################################################################
###################################################################################################################
###################################################################################################################
###################################################################################################################

def GetVoronoiOD():
   
    # Define spatial limits of the research corridor
    # The coordinates must end with 000 in order to match the coordinates of the input raster data
    e_min, e_max = 2687000, 2708000     # 2688000, 2704000 - 2688000, 2705000
    n_min, n_max = 1237000, 1254000     # 1238000, 1252000 - 1237000, 1252000
    limits_corridor = [e_min, n_min, e_max, n_max]
    # Get a polygon as limits for teh corridor
    innerboundary = polygon_from_points(e_min=e_min, e_max=e_max, n_min=n_min, n_max=n_max)


    # Import the required data or define the path to access it
    catchement_tif_path = r'data/catchment_pt/catchement.tif'
    catchmentdf = gpd.read_file(r"data/catchment_pt/catchement.gpkg")

    # File paths for population and employment combined raster files
    pop_combined_file = r"data/independent_variable/processed/scenario/pop_combined.tif"
    empl_combined_file = r"data/independent_variable/processed/scenario/empl_combined.tif"

    correct_rasters_to_extent(pop_combined_file,
        empl_combined_file,
        output_empl_path="data/independent_variable/processed/scenario/empl20_corrected.tif",
        output_pop_path="data/independent_variable/processed/scenario/pop20_corrected.tif",
        reference_boundary=innerboundary,
        resolution=100,
        crs="EPSG:2056")
    
    pop_combined_file = r"data/independent_variable/processed/scenario/empl20_corrected.tif"
    empl_combined_file = r"data/independent_variable/processed/scenario/pop20_corrected.tif"

    # define dev (=ID of the polygons of a development)
    dev = 0

    # Get voronoidf crs
    print(catchmentdf.crs)

    # todo When we iterate over devs and scens, maybe we can check if the VoronoiDF already has the communal data and then skip the following five lines
    popvec = GetCommunePopulation(y0="2021")
    jobvec = GetCommuneEmployment(y0=2021)
    od = GetHighwayPHDemandPerCommune() ## check tau values for PT
    odmat = GetODMatrix(od)

    # This function returns a np array of raster data storing the bfs number of the commune in each cell
    commune_raster, commune_df = GetCommuneShapes(raster_path=catchement_tif_path)

    if jobvec.shape[0] != odmat.shape[0]:
        print(
            "Error: The number of communes in the OD matrix and the number of communes in the employment data do not match.")
    # com_idx = np.unique(od['quelle_code']) # previously od_mat
    # 1. Define a new raster file that stores the Commune's BFS ID as cell value
    # Think if new band or new tif makes more sense
    # using communeShapes


###################################################################################################################

    # Open all scenario raster data (only low scenarios) 
    with rasterio.open(pop_combined_file) as src:
        # Read the raster into a NumPy array (assuming you want the first band)
        #"pop_urban_", "pop_equal_", "pop_rural_"
        scen_pop_urban_tif = src.read(1)
        scen_pop_equal_tif = src.read(2)
        scen_pop_rural_tif = src.read(3)

    with rasterio.open(empl_combined_file) as src:
        # Read the raster into a NumPy array (assuming you want the first band)
        scen_empl_urban_tif = src.read(1)
        scen_empl_equal_tif = src.read(2)
        scen_empl_rural_tif = src.read(3)

    # Open status quo
    with rasterio.open(r"data/independent_variable/processed/raw/empl20_corrected.tif") as src:
        scen_empl_20_tif = src.read(1)

    with rasterio.open(r"data/independent_variable/processed/raw/pop20_corrected.tif") as src:
        scen_pop_20_tif = src.read(1)

    # Load the catchment raster data
    with rasterio.open(catchement_tif_path) as src:
        # Read the raster data
        catchment_tif = src.read(2)  # Read the second band, which holds id information
    # Identify unique catchment IDs
    unique_catchment_id = np.sort(np.unique(catchment_tif))
    catch_idx = unique_catchment_id.size  # Total number of unique catchments

    bounds = src.bounds  # Get the spatial bounds of the raster
    # Filter commune_df based on catchment raster bounds
    commune_df_filtered = commune_df.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]

    # Extract "BFS" values (unique commune IDs) within the bounds
    commune_df_filtered = commune_df_filtered["BFS"].to_numpy()

    # Ensure the OD matrix corresponds only to filtered communes
    odmat_frame = odmat.loc[commune_df_filtered, commune_df_filtered]

    # Initialize an OD matrix for catchments
    # Shape is [number of unique catchments, number of unique catchments]
    od_mn = np.zeros([catch_idx, catch_idx])


    # Assume vectorized functions are defined for the below operations
    def compute_cont_r(odmat, popvec, jobvec):
        # Convert popvec and jobvec to 2D arrays for broadcasting
        pop_matrix = np.array(popvec)[:, np.newaxis]
        job_matrix = np.array(jobvec)[np.newaxis, :]

        # Ensure odmat is a NumPy array
        odmat = np.array(odmat)

        # Perform the vectorized operation
        cont_r = odmat / (pop_matrix * job_matrix)
        return cont_r

    def compute_cont_v(cont_r, pop_m, job_n):
        # Sum over the cont_r matrix, multiply by pop_m and job_n
        cont_v = np.sum(cont_r)
        return cont_v

    # Step 1: generate unit_flow matrix from each commune to each other commune
    cout_r = odmat / np.outer(popvec, jobvec)

    # Step 2: Get all pairs of combinations from communes to polygons
    unique_commune_id = np.sort(np.unique(commune_raster))
    pairs = pd.DataFrame(columns=['commune_id', 'catchment_id'])
    pop_empl = pd.DataFrame(columns=['commune_id', 'catchment_id', "empl", "pop"])

    for i in tqdm(unique_catchment_id_id, desc='Processing catchment IDs'):
        # Get the catchment raster
        mask_catchment = catchment_tif == i
        for j in unique_commune_id:
            if j > 0:
                # Get the commune raster
                mask_commune = commune_raster == j
                # Combined mask
                mask = mask_commune & mask_catchment
                # Check if there are overlaying values
                if np.nansum(mask) > 0:
                    # pairs = pairs.append({'commune_id': j, 'catchment_id': i}, ignore_index=True)
                    temp = pd.Series({'commune_id': j, 'catchment_id': i})
                    pairs = gpd.GeoDataFrame(
                        pd.concat([pairs, pd.DataFrame(temp).T], ignore_index=True))

                    # Get the population and employment values for multiple scenarios
                    pop20 = scen_pop_20_tif[mask]
                    empl20 = scen_empl_20_tif[mask]
                    pop_low = scen_pop_low_tif[mask]
                    empl_low = scen_empl_low_tif[mask]
                    pop_medium = scen_pop_medium_tif[mask]
                    empl_medium = scen_empl_medium_tif[mask]
                    pop_high = scen_pop_high_tif[mask]
                    empl_high = scen_empl_high_tif[mask]

                    temp = pd.Series({'commune_id': j, 'voronoi_id': i,
                                      'pop_20': np.nansum(pop20), 'empl_20': np.nansum(empl20),
                                      'pop_low': np.nansum(pop_low), 'empl_low': np.nansum(empl_low),
                                      'pop_medium': np.nansum(pop_medium), 'empl_medium': np.nansum(empl_medium),
                                      'pop_high': np.nansum(pop_high), 'empl_high': np.nansum(empl_high)})
                    pop_empl = gpd.GeoDataFrame(
                        pd.concat([pop_empl, pd.DataFrame(temp).T], ignore_index=True))
                    # pop_empl = pop_empl.append({'commune_id': j, 'voronoi_id': i,
                    #                            'pop_20': np.nansum(pop20), 'empl_20': np.nansum(empl20),
                    #                            'pop_low': np.nansum(pop_low), 'empl_low': np.nansum(empl_low),
                    #                            'pop_medium': np.nansum(pop_medium), 'empl_medium': np.nansum(empl_medium),
                    #                            'pop_high': np.nansum(pop_high), 'empl_high': np.nansum(empl_high)},
                    #                            ignore_index=True)

            else:
                continue

    # Print array shapes to compare
    print(f"cout_r: {cout_r.shape}")
    print(f"pairs: {pairs.shape}")
    print(f"pop_empl: {pop_empl.shape}")

    # Step 3 complete exploded matrix
    # Initialize the OD matrix DataFrame with zeros or NaNs
    tuples = list(zip(pairs['voronoi_id'], pairs['commune_id']))
    multi_index = pd.MultiIndex.from_tuples(tuples, names=['voronoi_id', 'commune_id'])
    temp_df = pd.DataFrame(index=multi_index, columns=multi_index).fillna(0).to_numpy('float')
    od_matrix = pd.DataFrame(data=temp_df, index=multi_index, columns=multi_index)

    # Handle raster without values
    # Drop pairs with 0 pop or empl

    set_id_destination = [col[1] for col in od_matrix.columns]

    # Get unique values from the second level of the index
    unique_values_second_index = od_matrix.index.get_level_values(1).unique()

    # Iterate over each cell in the od_matrix to fill it with corresponding values from other_matrix
    for commune_id_origin in unique_values_second_index:
        # for (polygon_id_o, commune_id_o), _ in tqdm(od_matrix.index.to_series().iteritems(), desc='Allocating unit_values to OD matrix'):

        # Extract the row for commune_id_o
        row_values = cout_r.loc[commune_id_origin]

        # Use the valid columns to extract values
        extracted_values = row_values[set_id_destination].to_numpy('float')

        # Create a boolean mask for rows where the second element of the index matches commune_id_o
        mask = od_matrix.index.get_level_values(1) == commune_id_origin

        # Update the rows in od_matrix where the mask is True
        od_matrix.loc[mask] = extracted_values  # .to_numpy('float')

    ####################################################################################################3
    # todo Filling happens here

    # Check for scenario based on column names in pop_empl
    # Sceanrio are defined like pop_XX and empl_XX get a list of all these endings (only XX)
    # Get the column names of pop_empl
    pop_empl_columns = pop_empl.columns
    # Get the column names that end with XX
    pop_empl_scenarios = [col.split("_")[1] for col in pop_empl_columns if col.startswith("pop_")]
    print(pop_empl_scenarios)

    # SEt index of df to access its single components
    pop_empl = pop_empl.set_index(['voronoi_id', 'commune_id'])

    # for each of these scenarios make an own copy of od_matrix named od_matrix+scen
    for scen in pop_empl_scenarios:
        print(f"Processing scenario {scen}")
        od_matrix_temp = od_matrix.copy()

        for polygon_id, row in tqdm(pop_empl.iterrows(), desc='Allocating pop and empl to OD matrix'):
            # Multiply all values in the row/column
            od_matrix_temp.loc[polygon_id] *= row[f'pop_{scen}']
            od_matrix_temp.loc[:, polygon_id] *= row[f'empl_{scen}']

        # Step 4: Group the OD matrix by polygon_id
        # Reset the index to turn the MultiIndex into columns
        od_matrix_reset = od_matrix_temp.reset_index()

        # Sum the values by 'polygon_id' for both the rows and columns
        od_grouped = od_matrix_reset.groupby('voronoi_id').sum()

        # Now od_grouped has 'polygon_id' as the index, but we still need to group the columns
        # First, transpose the DataFrame to apply the same operation on the columns
        od_grouped = od_grouped.T

        # Again group by 'polygon_id' and sum, then transpose back
        od_grouped = od_grouped.groupby('voronoi_id').sum().T

        # Drop column commune_id
        od_grouped = od_grouped.drop(columns='commune_id')

        # Set diagonal values to 0
        temp_sum = od_grouped.sum().sum()
        np.fill_diagonal(od_grouped.values, 0)
        # Compute the sum after changing the diagonal
        temp_sum2 = od_grouped.sum().sum()
        # Print difference
        print(f"Sum of OD matrix before {temp_sum} and after {temp_sum2} removing diagonal values")

        # Save pd df to csv
        od_grouped.to_csv(fr"data/traffic_flow/od/od_matrix_{scen}.csv")
        # odmat.to_csv(r"data/traffic_flow/od/od_matrix_raw.csv")

        # Print sum of all values in od df
        # Sum over all values in pd df
        sum_com = odmat.sum().sum()
        sum_poly = od_grouped.sum().sum()
        sum_com_frame = odmat_frame.sum().sum()
        print(
            f"Total trips before {sum_com_frame} ({odmat_frame.shape} communes) and after {sum_poly} ({od_grouped.shape} polygons)")
        print(
            f"Total trips before {sum_com} ({odmat.shape} communes) and after {sum_poly} ({od_grouped.shape} polygons)")

        # Sum all columns of od_grouped
        origin = od_grouped.sum(axis=1).reset_index()
        origin.colum = ["voronoi_id", "origin"]
        # Sum all rows of od_grouped
        destination = od_grouped.sum(axis=0)
        destination = destination.reset_index()

        # merge origin and destination to voronoidf based on voronoi_id
        # Make a copy of voronoidf
        voronoidf_temp = voronoidf.copy()
        voronoidf_temp = voronoidf_temp.merge(origin, how='left', left_on='ID_point', right_on='voronoi_id')
        voronoidf_temp = voronoidf_temp.merge(destination, how='left', left_on='ID_point', right_on='voronoi_id')
        voronoidf_temp = voronoidf_temp.rename(columns={'0_x': 'origin', '0_y': 'destination'})
        voronoidf_temp.to_file(fr"data/traffic_flow/od/OD_voronoidf_{scen}.gpkg", driver="GPKG")

        # Same for odmat and commune_df
        if scen == "20":
            origin_commune = odmat_frame.sum(axis=1).reset_index()
            origin_commune.colum = ["commune_id", "origin"]
            destination_commune = odmat_frame.sum(axis=0).reset_index()
            destination_commune.colum = ["commune_id", "destination"]
            commune_df = commune_df.merge(origin_commune, how='left', left_on='BFS', right_on='quelle_code')
            commune_df = commune_df.merge(destination_commune, how='left', left_on='BFS', right_on='ziel_code')
            commune_df = commune_df.rename(columns={'0_x': 'origin', '0_y': 'destination'})
            commune_df.to_file(r"data/traffic_flow/od/OD_commune_filtered.gpkg", driver="GPKG")

    return





