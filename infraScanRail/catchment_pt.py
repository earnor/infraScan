import networkx as nx
import os
import numpy as np
from shapely.geometry import Point, Polygon
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from geopy.distance import geodesic
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
# Additional imports for grid creation
from shapely.geometry import box
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from plots import *
import alphashape
from data_import import *
from shapely.geometry import MultiPoint
from rasterio.features import rasterize
from shapely import wkt
import rasterio
from rasterio.transform import from_origin
from rasterio.transform import from_bounds



# 1.) Define Access Points (Train Stations):
# 2.) Prepare Bus Network:
# 3.) Calculate Fastest Path from Each Node (Bus Stop) to All Train Stations:
# 4.) Make a Grid with all Busstops and a Buffer of 650m
# 5.) Calculate Closest Busstop from each Gridpoint and Calculate Corresponding walking Time
# 6.) Calculate Total Time to Closest Acces Point fom each GridPoint and Visualize it

###############################################################################################################################################################################################

def create_bus_buffers(closest_trainstations_df, stops, output_path):
    '''
    Create merged bus stop buffers grouped by train station.

    Parameters:
        closest_trainstations_df (GeoDataFrame): DataFrame containing bus stops and their closest train stations.
        stops (GeoDataFrame): GeoDataFrame containing bus stop geometries.
        output_path (str): Path to save the resulting GeoPackage.

    Returns:
        GeoDataFrame: Merged bus buffers.
    '''
    merged_df = closest_trainstations_df.merge(
        stops[['DIVA_NR', 'geometry']],
        left_on='bus_stop',
        right_on='DIVA_NR',
        how='left'
    )

    connected_df = gpd.GeoDataFrame(merged_df, geometry='geometry', crs='EPSG:2056')
    connected_df = connected_df[~(connected_df['geometry'].is_empty | connected_df['geometry'].isna())]
    connected_df['buffer'] = connected_df['geometry'].buffer(650)

    grouped_buffers = (
        connected_df.groupby('train_station')
        .agg({'buffer': lambda x: unary_union(x)})
        .reset_index()
    )

    bus_buffers = gpd.GeoDataFrame(grouped_buffers, geometry='buffer', crs='EPSG:2056')
    bus_buffers.to_file(output_path, driver="GPKG")
    print(f"Bus buffers saved to {output_path}")
    return bus_buffers


def create_train_buffers(stops, output_path):
    '''
    Create merged train station buffers for VTYP='S-Bahn'.

    Parameters:
        stops (GeoDataFrame): GeoDataFrame containing train station geometries.
        output_path (str): Path to save the resulting GeoPackage.

    Returns:
        GeoDataFrame: Train station buffers.
    '''
    # Filter stops to include only train stations (VTYP='S-Bahn')
    train_stations = stops[stops['VTYP'] == 'S-Bahn'].drop_duplicates(subset=['DIVA_NR']).copy()
    
    # Copy DIVA_NR to train_station
    train_stations['train_station'] = train_stations['DIVA_NR']
    
    # Apply 1000m buffer to the train stations
    train_stations['buffer'] = train_stations['geometry'].buffer(1000)

    # Create a GeoDataFrame with the buffer
    train_buffers = gpd.GeoDataFrame(train_stations[['train_station', 'buffer']], geometry='buffer', crs='EPSG:2056')

    # Save the train buffers to a GeoPackage
    train_buffers.to_file(output_path, driver="GPKG")
    print(f"Train buffers saved to {output_path}")

    return train_buffers

def resolve_overlaps(bus_buffers, train_buffers, output_path):
    '''
    Resolve overlaps between train and bus buffers, prioritizing train buffers, 
    and merge final polygons by train station.

    Parameters:
        bus_buffers (GeoDataFrame): GeoDataFrame containing bus buffers.
        train_buffers (GeoDataFrame): GeoDataFrame containing train buffers.
        output_path (str): Path to save the resulting GeoPackage.

    Returns:
        None
    '''
    # Debug: Print initial DataFrames
    print("Initial Train Buffers:")
    print(train_buffers.head())
    print(train_buffers.columns)
    print("Train Buffers CRS:", train_buffers.crs)
    
    print("Initial Bus Buffers:")
    print(bus_buffers.head())
    print(bus_buffers.columns)
    print("Bus Buffers CRS:", bus_buffers.crs)

    # Resolve overlaps: subtract train buffers from bus buffers
    for train_idx, train_row in train_buffers.iterrows():
        train_geom = train_row['buffer']
        for bus_idx, bus_row in bus_buffers.iterrows():
            bus_geom = bus_row['buffer']
            if train_geom.intersects(bus_geom):
                intersection = train_geom.intersection(bus_geom)
                bus_buffers.at[bus_idx, 'buffer'] = bus_geom.difference(intersection)

    # Combine train and bus buffers
    train_buffers['type'] = 'train'
    bus_buffers['type'] = 'bus'

    # Debug: Print buffers before concatenation
    print("Train Buffers Before Concatenation:")
    print(train_buffers.head())
    print("Bus Buffers Before Concatenation:")
    print(bus_buffers.head())

    # Combine buffers
    combined_buffers = pd.concat([train_buffers, bus_buffers], ignore_index=True)

    # Debug: Print combined DataFrame
    print("Combined Buffers DataFrame:")
    print(combined_buffers.head())
    print(combined_buffers.columns)

    # Create GeoDataFrame and set geometry
    final_buffers = gpd.GeoDataFrame(combined_buffers, geometry='buffer', crs='EPSG:2056')

    # Group by train_station and merge polygons
    grouped_buffers = (
        final_buffers.groupby('train_station')
        .agg({'buffer': lambda x: unary_union(x)})
        .reset_index()
    )

    # Debug: Print final grouped buffers
    print("Grouped Buffers DataFrame:")
    print(grouped_buffers.head())

    # Create the final GeoDataFrame
    final_buffers = gpd.GeoDataFrame(grouped_buffers, geometry='buffer', crs='EPSG:2056')

    # Save the final combined polygons
    final_buffers.to_file(output_path, driver="GPKG")
    print(f"Final merged polygons saved to {output_path}")

import geopandas as gpd
from shapely.geometry import Polygon

def clip_and_fill_polygons(merged_buffers_path, innerboundary_path, output_path):
    '''
    Clip polygons to the inner boundary, assign a value of -1 to uncovered areas,
    and save the result with non-overlapping polygons.

    Parameters:
        merged_buffers_path (str): Path to the merged polygons GeoPackage.
        innerboundary_path (str): Path to the shapefile defining the inner boundary.
        output_path (str): Path to save the final processed GeoPackage.

    Returns:
        None
    '''
    # Step 1: Load the merged buffers and inner boundary
    merged_buffers = gpd.read_file(merged_buffers_path)
    inner_boundary = gpd.read_file(innerboundary_path)

    # Ensure CRS consistency
    if merged_buffers.crs != inner_boundary.crs:
        inner_boundary = inner_boundary.to_crs(merged_buffers.crs)

    # Fix invalid geometries in merged buffers
    if not merged_buffers.geometry.is_valid.all():
        print("Fixing invalid geometries in merged buffers...")
        merged_buffers['geometry'] = merged_buffers.geometry.buffer(0)

    # Step 2: Clip polygons to the inner boundary
    merged_buffers['geometry'] = merged_buffers.geometry.intersection(inner_boundary.unary_union)

    # Step 3: Identify uncovered areas
    merged_area = merged_buffers.unary_union  # Combine all polygons
    uncovered_area = inner_boundary.unary_union.difference(merged_area)

    # Step 4: Create a single polygon for uncovered areas and assign -1
    if not uncovered_area.is_empty:
        uncovered_polygon = gpd.GeoDataFrame(
            [{'train_station': -1, 'geometry': uncovered_area}],
            geometry='geometry',
            crs=merged_buffers.crs
        )
    else:
        uncovered_polygon = gpd.GeoDataFrame(
            [{'train_station': -1, 'geometry': Polygon()}],  # Empty polygon
            geometry='geometry',
            crs=merged_buffers.crs
        )

    # Step 5: Combine the merged buffers with the uncovered polygon
    final_buffers = gpd.GeoDataFrame(
        pd.concat([merged_buffers, uncovered_polygon], ignore_index=True),
        geometry='geometry',
        crs=merged_buffers.crs
    )

    # Step 6: Save the finalized polygons
    final_buffers.to_file(output_path, driver='GPKG')
    print(f"Final processed polygons saved to {output_path}")

def add_diva_nr_to_points_with_buffer(points_path, stops_path, output_path, buffer_distance=100):
    """
    Adds DIVA_NR information to points by performing a spatial join with S-Bahn stops using a buffer.

    Parameters:
        points_path (str): Path to the points GeoPackage.
        stops_path (str): Path to the bus stops GeoPackage.
        output_path (str): Path to save the updated points GeoPackage.
        buffer_distance (int): Buffer distance (in meters) around points for the spatial join.

    Returns:
        None
    """
    # Load the points and stops GeoDataFrames
    points_gdf = gpd.read_file(points_path)
    stops_gdf = gpd.read_file(stops_path)

    # Filter stops for S-Bahn
    s_bahn_stops = stops_gdf[stops_gdf['VTYP'] == 'S-Bahn']

    # Create a buffer around each point
    points_gdf['buffer'] = points_gdf.geometry.buffer(buffer_distance)

    # Use the buffer geometries for spatial join
    points_with_buffer = gpd.GeoDataFrame(points_gdf, geometry='buffer', crs=points_gdf.crs)
    joined_gdf = gpd.sjoin(points_with_buffer, s_bahn_stops[['DIVA_NR', 'geometry']], how='left', predicate='intersects')

    # Add DIVA_NR to the original points GeoDataFrame
    points_gdf['DIVA_NR'] = joined_gdf['DIVA_NR']

    # Drop the buffer column
    points_gdf.drop(columns=['buffer'], inplace=True)

    # Save the updated points GeoDataFrame
    points_gdf.to_file(output_path, driver="GPKG")
    print(f"Updated points with DIVA_NR saved to {output_path}")

def process_polygons_with_mapping(polygons_file_path, points_file_path, output_path):
    """
    Processes polygons GeoDataFrame by adding a column (ID_point) based on matching train_station with DIVA_NR.
    Adds a manual mapping for train_station = 13728, renames the column to 'id', and fills NULL values with -1.

    Args:
        polygons_file_path (str): Path to the polygons GeoPackage file.
        points_file_path (str): Path to the points GeoPackage file.
        output_path (str): Path to save the updated GeoPackage file.

    Returns:
        None
    """
    # Load the data
    polygons_gdf = gpd.read_file(polygons_file_path)
    points_gdf = gpd.read_file(points_file_path)

    # Convert columns to integers for consistent matching
    polygons_gdf['train_station'] = pd.to_numeric(polygons_gdf['train_station'], errors='coerce').astype('Int64')
    points_gdf['DIVA_NR'] = pd.to_numeric(points_gdf['DIVA_NR'], errors='coerce').astype('Int64')

    # Drop NaN values from both datasets after conversion
    polygons_gdf = polygons_gdf.dropna(subset=['train_station'])
    points_gdf = points_gdf.dropna(subset=['DIVA_NR'])

    # Perform the merge to match polygons_gdf with points_gdf based on train_station and DIVA_NR
    merged = polygons_gdf.merge(
        points_gdf[['DIVA_NR', 'ID_point']],
        left_on='train_station',
        right_on='DIVA_NR',
        how='left'
    )

    # Add the ID_point column to polygons_gdf
    polygons_gdf['ID_point'] = merged['ID_point']

    # Apply manual mapping for train_station = 13728
    polygons_gdf.loc[polygons_gdf['train_station'] == 13728, 'ID_point'] = 1663

    # Rename ID_point to id
    polygons_gdf.rename(columns={'ID_point': 'id'}, inplace=True)

    # Replace NULL values in 'id' with -1
    polygons_gdf['id'] = polygons_gdf['id'].fillna(-1).astype(int)

    # Save the result to a new file
    polygons_gdf.to_file(output_path, driver="GPKG")

    print(f"Processed data saved to {output_path}")

def create_raster_from_gpkg(input_gpkg, output_tif, raster_size=(100, 100)):
    """
    Creates a raster from a GeoPackage file based on polygon IDs.
    If multiple polygons overlap, assigns one ID randomly to the raster cell.
    NULL values in the 'id' column are replaced with -1.

    Args:
        input_gpkg (str): Path to the GeoPackage file containing polygons with an 'id' column.
        output_tif (str): Path to save the resulting raster file.
        raster_size (tuple): Tuple of (height, width) for the output raster dimensions.

    Returns:
        None
    """
    # Load the GeoPackage file
    gdf = gpd.read_file(input_gpkg)

    # Ensure the 'id' column exists and replace NULLs with -1
    if 'id' not in gdf.columns:
        raise ValueError("The GeoPackage file must contain an 'id' column.")
    gdf['id'] = pd.to_numeric(gdf['id'], errors='coerce').fillna(-1).astype(int)

    # Define the bounds and transform for the raster
    minx, miny, maxx, maxy = gdf.total_bounds
    transform = from_bounds(minx, miny, maxx, maxy, raster_size[1], raster_size[0])

    # Prepare shapes for rasterization (geometry and id as value)
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf['id']))

    # Rasterize the polygons
    raster = rasterize(
        shapes=shapes,
        out_shape=raster_size,
        transform=transform,
        fill=-1,  # Default value for no data
        dtype='int32'
    )

    # Define raster metadata
    meta = {
        'driver': 'GTiff',
        'height': raster_size[0],
        'width': raster_size[1],
        'count': 1,
        'dtype': 'int32',
        'crs': gdf.crs.to_string(),
        'transform': transform,
        'nodata': -1
    }

    # Save the raster to file
    with rasterio.open(output_tif, 'w', **meta) as dst:
        dst.write(raster, 1)

    print(f"Raster saved to {output_tif}")


def calculate_fastest_connections_to_trains(G_bus):
    # Extract nodes of type 'Train' and 'Bus'
    train_stations = [node for node, attr in G_bus.nodes(data=True) if attr['type'] == 'Train']
    bus_stops = [node for node, attr in G_bus.nodes(data=True) if attr['type'] == 'Bus']
    
    # Initialize a list to store OD pairs and travel times
    od_pairs = []

    # Iterate over each bus stop
    for bus_id in bus_stops:
        # For each bus stop, iterate over all train stations
        for train_id in train_stations:
            try:
                # Calculate the shortest path and its length
                shortest_path_length = nx.shortest_path_length(
                    G_bus, source=bus_id, target=train_id, weight='weight'
                )
                # Append the OD pair, travel time, and status to the result
                od_pairs.append({
                    'bus_stop': bus_id,
                    'train_station': train_id,
                    'travel_time': shortest_path_length,
                    'status': 'connected'
                })
            except nx.NetworkXNoPath:
                # Handle cases where no path exists between the nodes
                od_pairs.append({
                    'bus_stop': bus_id,
                    'train_station': train_id,
                    'travel_time': float('inf'),  # No connection
                    'status': 'not connected'
                })

    # Convert the result into a DataFrame for better handling
    od_df = pd.DataFrame(od_pairs)

    return od_df


def find_closest_train_station(od_df):
    """
    Find the closest train station for each bus stop based on travel time.

    Args:
    od_df (pd.DataFrame): A DataFrame containing columns 'bus_stop', 'train_station',
                          'travel_time', and 'status'.

    Returns:
    pd.DataFrame: A DataFrame with the closest train station for each bus stop, 
                  or marked as 'not connected' if no connection exists.
    """
    # Filter out rows where the status is 'not connected'
    connected_df = od_df[od_df['status'] == 'connected']

    # Find the train station with the minimum travel time for each bus stop
    closest_pairs = connected_df.loc[connected_df.groupby('bus_stop')['travel_time'].idxmin()]

    # Ensure all bus stops are included in the result
    all_bus_stops = od_df['bus_stop'].unique()
    connected_bus_stops = closest_pairs['bus_stop'].unique()
    not_connected_bus_stops = set(all_bus_stops) - set(connected_bus_stops)

    # Create a DataFrame for bus stops with no connection
    not_connected_df = pd.DataFrame({
        'bus_stop': list(not_connected_bus_stops),
        'train_station': None,
        'travel_time': float('inf'),
        'status': 'not connected'
    })

    # Combine connected and not connected results
    result_df = pd.concat([closest_pairs, not_connected_df], ignore_index=True)

    # Sort the results by bus_stop for readability
    result_df.sort_values(by='bus_stop', inplace=True)

    return result_df



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

import geopandas as gpd
from shapely.ops import unary_union

import geopandas as gpd
from shapely.ops import unary_union, split
from shapely.geometry import Polygon

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



def calculate_total_travel_time(bus_times_within_buffer, closest_trainstations_df):
    """
    Calculate total travel time (walking + bus) for grid points to train stations.

    Args:
    - bus_times_within_buffer (pd.DataFrame): DataFrame with columns:
        - 'geometry': Geometry of grid points
        - 'nearest_bus_stop': Nearest bus stop ID for each grid point
        - 'walking_time': Walking time to the nearest bus stop
    - closest_trainstations_df (pd.DataFrame): DataFrame with columns:
        - 'bus_stop': Bus stop ID
        - 'train_station': Closest train station ID
        - 'travel_time': Travel time from bus stop to train station
        - 'status': Connection status ('connected' or 'not connected')

    Returns:
    - pd.DataFrame: DataFrame with total travel times and closest train stations.
    """
    total_times = []

    # Convert closest_trainstations_df to a dictionary for quick lookup
    bus_to_train_time = closest_trainstations_df.set_index('bus_stop')['travel_time'].to_dict()
    bus_to_train_station = closest_trainstations_df.set_index('bus_stop')['train_station'].to_dict()

    for index, row in bus_times_within_buffer.iterrows():
        grid_point = row['geometry']
        nearest_bus_stop = row['nearest_bus_stop']
        walking_time = row['walking_time']

        # Check if the nearest bus stop has a connection to a train station
        if nearest_bus_stop not in bus_to_train_time or pd.isna(bus_to_train_time[nearest_bus_stop]):
            total_times.append({
                'grid_point': grid_point,
                'closest_train_station': None,
                'total_time': float('inf')
            })
            continue

        # Get the travel time and train station for the nearest bus stop
        bus_travel_time = bus_to_train_time[nearest_bus_stop]
        closest_train_station = bus_to_train_station[nearest_bus_stop]

        # Calculate the total travel time (walking + bus)
        total_time = walking_time + bus_travel_time

        total_times.append({
            'grid_point': grid_point,
            'closest_train_station': closest_train_station,
            'total_time': total_time
        })

    return pd.DataFrame(total_times)


def save_points_as_raster(df, output_path=r'data/catchment_pt/catchement.tif', resolution=100, crs='EPSG:2056'):
    """
    Save points from a DataFrame to a GeoTIFF raster file.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'grid_point', 'total_time', and 'closest_train_station' columns.
        output_path (str): Path to save the output GeoTIFF file.
        resolution (int): Resolution of the raster in meters.
        crs (str): Coordinate Reference System in EPSG format.
    """
    # Validate input DataFrame
    required_columns = {'grid_point', 'total_time', 'closest_train_station'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Parse grid_point into x and y coordinates
    # Extract x and y coordinates directly from the object
    df['x'] = df['grid_point'].apply(lambda point: point.x)
    df['y'] = df['grid_point'].apply(lambda point: point.y)

    # Drop rows with NaN values in x or y
    if df[['x', 'y']].isna().any().any():
        print("Warning: Dropping rows with invalid 'grid_point' coordinates.")
        df = df.dropna(subset=['x', 'y'])

    # Define grid extent
    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()

    # Validate grid extent
    if np.isnan(x_min) or np.isnan(x_max) or np.isnan(y_min) or np.isnan(y_max):
        raise ValueError("Grid extent contains NaN values. Check the 'grid_point' column for invalid entries.")

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
        'crs': crs,
        'transform': transform,
    }

    # Save as GeoTIFF
    with rasterio.open(output_path, 'w', **metadata) as dst:
        dst.write(total_time_raster, 1)  # First band: total_time
        dst.write(station_raster, 2)    # Second band: closest_train_station
    
    print(f"Raster saved to {output_path}")

###############################################################################################################################################################################################
# 1.) Define Access Points (Train Stations):
# 2.) Prepare Bus Network:
###############################################################################################################################################################################################

def get_catchement():


    # Define spatial limits of the research corridor
    # The coordinates must end with 000 in order to match the coordinates of the input raster data
    e_min, e_max = 2687000, 2708000     # 2688000, 2704000 - 2688000, 2705000
    n_min, n_max = 1237000, 1254000     # 1238000, 1252000 - 1237000, 1252000

    # Boudary for plot

    # Get a polygon as limits for teh corridor

    # For global operation a margin is added to the boundary
    margin = 3000 # meters
    outerboundary = polygon_from_points(e_min=e_min, e_max=e_max, n_min=n_min, n_max=n_max, margin=margin)
    
    # Load the GeoPackage for bus lines and stops
    bus_lines_path = r"data/Network/Buslines/Linien_des_offentlichen_Verkehrs_-OGD.gpkg"

    # Load the bus lines and bus stops layers
    layer_name_segmented = 'ZVV_LINIEN_L'
    bus_lines_segmented = gpd.read_file(bus_lines_path, layer=layer_name_segmented)
    stops = gpd.read_file(r"data/Network/Buslines/Haltestellen_des_offentlichen_Verkehrs_-OGD.gpkg")
    

    # Filter bus stops and bus lines within the boundary
    stops_filtered = stops[stops.within(outerboundary)]
    bus_lines_segmented_filtered = bus_lines_segmented[bus_lines_segmented.within(outerboundary)]

    # Create a directed graph for the bus network
    G_bus = nx.Graph()

    # Add filtered bus stops as nodes with positions from the 'geometry' column
    for idx, row in stops_filtered.iterrows():
        stop_id = row['DIVA_NR']
        stop_position = row['geometry']
        

        # Determine the type of stop (Bus, Train, or Other)
        if pd.isna(row['VTYP']) or row['VTYP'] == '':
            pass
        elif row['VTYP'] == 'S-Bahn':
            pass
        else:
            pass

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

    # Create edges with a fixed travel time of 83 seconds for all nodes closer than 100m
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

    '''
    # Plot the bus network
    plt.figure(figsize=(12, 12))
    nx.draw(G_bus, pos, node_size=10, node_color='red', with_labels=False, edge_color='blue', linewidths=1, font_size=8)
    '''

    ###############################################################################################################################################################################################
    # 3.) Calculate Fastest Path from Each Node (Bus Stop) to All Train Stations:
    ###############################################################################################################################################################################################

    # calculate od_matrix for the bus metwork
    od_df_busTOtrain = calculate_fastest_connections_to_trains(G_bus)
    #find the closest trainstations for each busstop
    closest_trainstations_df = find_closest_train_station(od_df_busTOtrain)

    # These lines of code create a 650m buffer around bus stops and assign each bus stop 
    # to the closest train station based on their service direction toward a train station.
    # requires assessing changes in access times.
    # Specify the output file path
    #create_merge is an old function, which does not include the trainbuffers
    #create_merged_trainstation_buffers(closest_trainstations_df, stops, output_path)
    # File paths
    bus_buffers_path = "data/catchment_pt/bus_buffers.gpkg"
    train_buffers_path = "data/catchment_pt/train_buffers.gpkg"
    final_output_path = "data/catchment_pt/final_buffers.gpkg"

    # SubStep 1: Create Bus Buffers
    bus_buffers = create_bus_buffers(closest_trainstations_df, stops, bus_buffers_path)

    # SubStep 2: Create Train Buffers (only VTYP='S-Bahn')
    train_buffers = create_train_buffers(stops, train_buffers_path)

    # SubStep 3: Resolve Overlaps and Merge Polygons by Train Station
    resolve_overlaps(bus_buffers, train_buffers, final_output_path)
    clip_and_fill_polygons(
    merged_buffers_path="data/catchment_pt/final_buffers.gpkg",
    innerboundary_path="data/_basic_data/innerboundary.shp",
    output_path="data/catchment_pt/final_clipped_buffers.gpkg")

    #prepare the nodes for matching the DIVANR and the rail node number
    # File paths
    points_file_path = "data/Network/processed/points.gpkg"
    stops_file_path = "data/Network/Buslines/Haltestellen_des_offentlichen_Verkehrs_-OGD.gpkg"
    output_file_path = "data/Network/processed/points_with_diva_nr_buffered.gpkg"

    # Mapping
    add_diva_nr_to_points_with_buffer(points_file_path, stops_file_path, output_file_path)
    process_polygons_with_mapping(
        polygons_file_path="data/catchment_pt/final_clipped_buffers.gpkg",
        points_file_path="data/Network/processed/points_with_diva_nr_buffered.gpkg",
        output_path="data/catchment_pt/catchement.gpkg")
    
    #save .gkpk also as .tif raster
    create_raster_from_gpkg(
    input_gpkg="data/catchment_pt/catchement.gpkg",
    output_tif="data/catchment_pt/catchement.tif",
    raster_size=(100, 100))  # Raster size set to 100x100

    return
