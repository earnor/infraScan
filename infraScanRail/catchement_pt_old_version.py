import networkx as nx
import os
import numpy as np
from shapely.geometry import Point, Polygon
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Point
from geopy.distance import geodesic
from scipy.spatial import Voronoi
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
# Additional imports for grid creation
from shapely.geometry import box
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


# 1.) Define Access Points (Train Stations):
# 2.) Prepare Bus Network:
# 3.) Calculate Fastest Path from Each Node (Bus Stop) to All Train Stations:
# 4.) Make a Grid with all Busstops and a Buffer of 650m
# 5.) Calculate Closest Busstop from each Gridpoint and Calculate Corresponding walking Time
# 6.) Calculate Total Time to Closest Acces Point fom each GridPoint and Visualize it


###############################################################################################################################################################################################
# 1.) Define Access Points (Train Stations):
# 2.) Prepare Bus Network:
###############################################################################################################################################################################################



# Function to generate a polygon for the corridor limits
def polygon_from_points(e_min, e_max, n_min, n_max, margin=0):
    points = [
        (e_min - margin, n_min - margin),
        (e_max + margin, n_min - margin),
        (e_max + margin, n_max + margin),
        (e_min - margin, n_max + margin),
        (e_min - margin, n_min - margin)
    ]
    return Polygon(points)

# Set your working directory
os.chdir(r"C:\Users\phili\polybox\ETH_RE&IS\Master Thesis\06-Developments\01-Code\infraScanRail")

# Define spatial limits of the research corridor
e_min, e_max = 2687000, 2708000
n_min, n_max = 1237000, 1254000
limits_corridor = [e_min, n_min, e_max, n_max]

# Get polygons for the corridor
boundary_plot = polygon_from_points(e_min=e_min+1000, e_max=e_max-500, n_min=n_min+1000, n_max=n_max-2000)
innerboundary = polygon_from_points(e_min=e_min, e_max=e_max, n_min=n_min, n_max=n_max)
margin = 3000
outerboundary = polygon_from_points(e_min=e_min, e_max=e_max, n_min=n_min, n_max=n_max, margin=margin)

# Load the GeoPackage for bus lines and stops
bus_lines_path = r"data/Network/Buslines/Linien_des_offentlichen_Verkehrs_-OGD.gpkg"

# Load the bus lines and bus stops layers
layer_name_segmented = 'ZVV_LINIEN_L'
bus_lines_segmented = gpd.read_file(bus_lines_path, layer=layer_name_segmented)
stops = gpd.read_file(r"data/Network/Buslines/Haltestellen_des_offentlichen_Verkehrs_-OGD.gpkg")

# Filter bus stops and bus lines within the boundary
stops_filtered = stops[stops.within(innerboundary)]
bus_lines_segmented_filtered = bus_lines_segmented[bus_lines_segmented.within(innerboundary)]

# Create a directed graph for the bus network
G_bus = nx.Graph()

# Add filtered bus stops as nodes with positions from the 'geometry' column
for idx, row in stops_filtered.iterrows():
    stop_id = row['DIVA_NR']
    stop_position = row['geometry']
    

    # Determine the type of stop (Bus, Train, or Other)
    if pd.isna(row['VTYP']) or row['VTYP'] == '':
        stop_type = 'Other'  # Set to 'Other' if VTYP is empty or NaN
    elif row['VTYP'] == 'S-Bahn':
        stop_type = 'Train'
    else:
        stop_type = 'Bus'

    # Determine the type of stop (Bus or Train)
    stop_type = 'Bus'  # Default to Bus
    if row['VTYP'] == 'S-Bahn':
        stop_type = 'Train'
    
    # Add the node with position and type
    G_bus.add_node(stop_id, pos=(stop_position.x, stop_position.y), type=stop_type)


# Add edges from the filtered bus lines segment
for idx, row in bus_lines_segmented_filtered.iterrows():
    from_stop = row['VONHALTESTELLENNR']
    to_stop = row['BISHALTESTELLENNR']
    travel_time = row['SHAPE_LEN'] / 6.945  # Assuming average bus speed in m/s
    
    # Check if both stops exist in the graph
    if from_stop in G_bus.nodes and to_stop in G_bus.nodes:
        # Determine the types of the stops
        from_stop_type = G_bus.nodes[from_stop]['type']
        to_stop_type = G_bus.nodes[to_stop]['type']
        
        # Add an edge to the graph with travel time as weight and stop types
        G_bus.add_edge(from_stop, to_stop, weight=travel_time, from_type=from_stop_type, to_type=to_stop_type)


# Define a threshold distance (e.g., 100 meters)
threshold_distance = 100

# Create edges with a fixed travel time of 60 seconds for all nodes closer than 100m
for stop_id_1 in G_bus.nodes:
    pos_1 = G_bus.nodes[stop_id_1]['pos']
    point_1 = Point(pos_1)  # Create a Point from the position

    for stop_id_2 in G_bus.nodes:
        if stop_id_1 != stop_id_2:  # Avoid self-comparison
            pos_2 = G_bus.nodes[stop_id_2]['pos']
            point_2 = Point(pos_2)  # Create a Point for the second stop
            
            # Calculate the distance between the two stops
            dist = point_1.distance(point_2)

            # If the distance is less than the threshold, add an edge with a fixed travel time
            if dist < threshold_distance:
                G_bus.add_edge(stop_id_1, stop_id_2, weight=83)  # Adding an edge with 83 seconds (walking Time for 100m, At 1.2 meters/second)


# Extract positions of bus stops for plotting
pos = nx.get_node_attributes(G_bus, 'pos')

# Plot the bus network
plt.figure(figsize=(12, 12))
nx.draw(G_bus, pos, node_size=10, node_color='red', with_labels=False, edge_color='blue', linewidths=1, font_size=8)

###############################################################################################################################################################################################
# 3.) Calculate Fastest Path from Each Node (Bus Stop) to All Train Stations:
###############################################################################################################################################################################################

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
        
        # Store the closest train station for the bus stop
        closest_train_stations[stop_id] = closest_station

# Output the results
for bus_stop, train_station in closest_train_stations.items():
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


###############################################################################################################################################################################################
# 4.) Make a Grid (100x100m) with all Busstops and a Buffer of 650m
# 5.) Calculate Closest Busstop from each Gridpoint and Calculate Corresponding walking Time
# 6.) Calculate Total Time to Closest Acces Point fom each GridPoint and Visualize it
###############################################################################################################################################################################################

# Set the cell size to 100 meters
cell_size = 100

# Generate x and y coordinates spaced 100 meters apart
x_coords = np.arange(e_min, e_max, cell_size)
y_coords = np.arange(n_min, n_max, cell_size)

# Create grid points by iterating over x and y coordinates
grid_points = [Point(x, y) for x in x_coords for y in y_coords]

# Create a GeoDataFrame for the grid
grid_gdf = gpd.GeoDataFrame(geometry=grid_points, crs="EPSG:2056")  # Swiss CH1903+ / LV95 coordinate system

# Generate a 650m buffer around each bus stop
stops_filtered['buffer'] = stops_filtered.geometry.buffer(650)

# Merge the individual buffers into a single geometry (union of all buffers)
merged_buffer = stops_filtered['buffer'].unary_union

# Filter grid points that are within the 650m bus stop buffers
grid_points_within_buffer = grid_gdf[grid_gdf.geometry.within(merged_buffer)]

#def to return the grid_points_within_buffer
def get_grid(grid_points_within_buffer):
    return grid_points_within_buffer

# Function to calculate walking time to the nearest bus stop
def calculate_nearest_bus_stop_time(grid_points_within_buffer, stops_filtered):
    # Filter bus stops only and reset index for safe access
    bus_stops = stops_filtered[stops_filtered['VTYP'] == 'Bus'].reset_index(drop=True)
    
    walking_times = []
    
    for index, row in grid_points_within_buffer.iterrows():
        grid_point = row['geometry']
        distances = bus_stops['geometry'].distance(grid_point)
        
        if distances.empty or distances.isna().all():
            walking_times.append({'geometry': grid_point, 'nearest_bus_stop': None, 'walking_time': float('inf')})
            continue
        
        nearest_bus_stop_idx = distances.idxmin()
        nearest_bus_stop = bus_stops.iloc[nearest_bus_stop_idx]['DIVA_NR']
        walking_time = distances.min() * 60 / 5000  # Convert distance to walking time (m/s)
        
        walking_times.append({
            'geometry': grid_point,
            'nearest_bus_stop': nearest_bus_stop,
            'walking_time': walking_time
        })
    
    return pd.DataFrame(walking_times)


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

# Calculate nearest bus stop time and then total travel time
bus_times_within_buffer = calculate_nearest_bus_stop_time(grid_points_within_buffer, stops_filtered)
total_times_within_buffer = calculate_total_travel_time(bus_times_within_buffer, G_bus, closest_train_stations)


# Print results
print(total_times_within_buffer)

#Store the total times and the closest access point (trainstation) in a gpkg file 


# Ensure the data has the right CRS if not set
total_times_within_buffer = gpd.GeoDataFrame(total_times_within_buffer, geometry='grid_point', crs='epsg:2056')

# Group points by closest_train_station and merge into polygons
polygons_by_station = total_times_within_buffer.dissolve(by='closest_train_station', aggfunc='sum')

# Create a convex hull for each group
polygons_by_station['polygons'] = polygons_by_station.geometry.apply(lambda g: g.convex_hull)

# Set the 'polygon' column as the geometry column
polygons_by_station.set_geometry("polygons", inplace=True)
polygons_by_station = polygons_by_station.drop(columns=["grid_point"])


# Specify the output path for the GeoPackage file
output_path = r"data/catchment_pt/merged_polygons_by_station.gpkg"

# Save the GeoDataFrame to a GeoPackage file
polygons_by_station.to_file(output_path, driver="GPKG")

###############################################################################################################################################################################################
###############################################################################################################################################################################################
''''
####Plot the Network

import contextily as ctx

# Extract positions of bus stops for plotting
pos = nx.get_node_attributes(G_bus, 'pos')

# Create a Matplotlib figure
fig, ax = plt.subplots(figsize=(12, 12))

# Plot the bus network
nx.draw(G_bus, pos, node_size=10, node_color='red', with_labels=False, edge_color='blue', linewidths=1, font_size=8, ax=ax)

# Set the axis limits based on your research corridor
ax.set_xlim(e_min, e_max)
ax.set_ylim(n_min, n_max)

# Add OpenStreetMap background
ctx.add_basemap(ax, crs="EPSG:2056", source=ctx.providers.OpenStreetMap.Mapnik)

# Show the plot
plt.show()






# Define the station IDs
source_station = 10008  # Starting station
target_station = 2608    # Destination station

# Check if both stations exist in the graph
if source_station in G_bus.nodes and target_station in G_bus.nodes:
    try:
        # Find the shortest path
        fastest_path = nx.shortest_path(G_bus, source=source_station, target=target_station, weight='weight')
        
        # Calculate the total travel time for the fastest path
        total_travel_time = nx.path_weight(G_bus, fastest_path, weight='weight')
        
        # Print the results
        print(f"Fastest path from station {source_station} to station {target_station}:")
        print(" -> ".join(map(str, fastest_path)))
        print(f"Total travel time: {total_travel_time:.2f} seconds")
        
    except nx.NetworkXNoPath:
        print(f"No path exists between station {source_station} and station {target_station}.")
else:
    print("One or both of the specified stations do not exist in the graph.")

'''





























































'''

import contextily as ctx
import matplotlib.pyplot as plt

# Plot grid points within the buffer on an OSM background
fig, ax = plt.subplots(figsize=(12, 12))
grid_points_within_buffer.plot(ax=ax, color='red', markersize=5, label='Grid Points Within Buffer')

# Set the extent to the bounds of grid_points_within_buffer for proper zoom
minx, miny, maxx, maxy = grid_points_within_buffer.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

# Add OSM basemap
ctx.add_basemap(ax, crs=grid_points_within_buffer.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

# Add labels and legend
ax.set_title("Grid Points Within 650m Buffer on OSM Background")
ax.legend()

plt.show()



# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

# Plot train stations
train_stations.plot(ax=ax, color='black', marker='^', markersize=50, label='Train Stations')

# Check if total_times_within_buffer is a GeoDataFrame
if isinstance(total_times_within_buffer, gpd.GeoDataFrame):
    # Extract x and y coordinates
    x_coords = total_times_within_buffer.geometry.x
    y_coords = total_times_within_buffer.geometry.y
else:
    # If not a GeoDataFrame, convert it
    total_times_within_buffer = gpd.GeoDataFrame(total_times_within_buffer, geometry='grid_point')
    x_coords = total_times_within_buffer.geometry.x
    y_coords = total_times_within_buffer.geometry.y

# Plot grid points with color based on the closest train station
sc = ax.scatter(
    x_coords,
    y_coords,
    c=total_times_within_buffer['color_idx'],
    cmap='viridis',
    s=10,
    alpha=0.6
)

# Add a colorbar
plt.colorbar(sc, ax=ax, label='Color Index')

# Add OSM background
# Make sure the plot is in the right coordinate reference system (CRS)
total_times_within_buffer = total_times_within_buffer.to_crs(epsg=3857)  # Web Mercator for OSM
ax.set_xlim(total_times_within_buffer.total_bounds[[0, 2]])  # Set x limits
ax.set_ylim(total_times_within_buffer.total_bounds[[1, 3]])  # Set y limits
ctx.add_basemap(ax, source=ctx.providers.Stamen.Terrain, crs=total_times_within_buffer.crs)  # Black and White OSM Background

# Add labels and title
ax.set_xlabel('Easting')
ax.set_ylabel('Northing')
ax.set_title('Closest Train Stations and Walking Times')

# Show legend
ax.legend()
plt.show()

'''

