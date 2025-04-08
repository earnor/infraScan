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
import shutil
import glob


# Initialize UTM to Lat/Lng transformer (replace with your UTM zone if needed)
utm_proj = pyproj.CRS("EPSG:32632")  # For UTM zone 32N (adjust if needed)
latlng_proj = pyproj.CRS("EPSG:4326")  # WGS84 Lat/Lng

def define_rail_network(paths):
    # Initialize a dictionary to store graphs for each path
    graphs = {}

    # Iterate over each path in the list
    for path in paths:
        # Load the GeoDataFrame from the GeoPackage
        nw_gdf = gpd.read_file(path)

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
        
        # Store the graph in the dictionary using the path as the key
        graphs[path] = G

    return graphs

def calculate_od_matrices_with_penalties(graph_dict):
    """
    Calculate OD matrices with penalties for multiple graphs.

    Args:
        graph_dict (dict): A dictionary where keys are graph identifiers (e.g., paths or names)
                           and values are NetworkX graph objects.

    Returns:
        dict: A dictionary where keys are graph identifiers and values are the corresponding OD matrices (as DataFrames).
    """
    od_matrices = {}

    for graph_name, G in graph_dict.items():
        # Initialize an empty list to collect OD records for the current graph
        od_records = []

        # Loop over each pair of nodes in the graph
        for origin in G.nodes:
            origin_station = G.nodes[origin]['station']  # Get the station name for the origin
            # Use Dijkstra's algorithm to find shortest paths from the origin node
            paths = nx.single_source_dijkstra_path(G, origin, weight='weight')

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

        # Convert the list of records to a DataFrame and store it in the result dictionary
        od_matrix = pd.DataFrame(od_records)
        od_matrices[graph_name] = od_matrix

    # Directory where you want to save the CSV files
    output_dir = r'data\Network\travel_time\developments'

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over the dictionary and save each DataFrame as a CSV
    for file_path, df in od_matrices.items():
        # Extract the file name from the file path (remove directories and use only the base name)
        file_name = os.path.basename(file_path)

        # Remove potential .gpkg from the file name
        file_name = file_name.replace(".gpkg", "")
        
        # Construct the full path for the output CSV
        output_csv_path = os.path.join(output_dir, f"{file_name}.csv")
        
        # Save the DataFrame as a CSV file
        df.to_csv(output_csv_path)
        print(f"Saved: {output_csv_path}")


    return od_matrices


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

        # The rest of the code goes here to create and resample the raster using rasterio...

    # Clean up the temporary directory
    os.rmdir(temp_dir)
    return

def split_via_nodes(df):
    """
    Split rows where the 'Via' column contains intermediate nodes.
    Each new row will represent a sub-edge from 'FromNode' to 'ToNode' including intermediate nodes.
    Station names for 'FromStation' and 'ToStation' are updated based on corresponding nodes.

    Parameters:
        df (pd.DataFrame): Original DataFrame containing 'FromNode', 'ToNode', 'FromStation', 'ToStation', and 'Via' columns.
        
    Returns:
        pd.DataFrame: Expanded DataFrame with all sub-edges and updated station names.
    """
    import pandas as pd
    import ast  # For safely evaluating string representations of lists

    # Ensure '-99' strings in the Via column are converted to an integer -99
    df['Via'] = df['Via'].apply(lambda x: str(x).replace("[-99]", "-99"))

    # Define a helper function to parse the 'Via' column
    def parse_via_column(via):
        if via == '-99':  # Special case: no intermediate nodes
            return []
        try:
            return [int(x) for x in ast.literal_eval(via)]
        except (ValueError, SyntaxError):
            return []

    # Parse the 'Via' column into lists of integers
    df['Via'] = df['Via'].apply(parse_via_column)

    # Create a mapping of node numbers to station names
    node_to_station = pd.concat([
        df[['FromNode', 'FromStation']].rename(columns={'FromNode': 'Node', 'FromStation': 'Station'}),
        df[['ToNode', 'ToStation']].rename(columns={'ToNode': 'Node', 'ToStation': 'Station'})
    ]).drop_duplicates().set_index('Node')['Station'].to_dict()

    # List to hold the expanded rows
    expanded_rows = []

    for _, row in df.iterrows():
        # Extract FromNode, ToNode, and parsed Via
        from_node = row['FromNode']
        to_node = row['ToNode']
        via_nodes = row['Via']
        
        # Create a complete path of nodes: FromNode -> ViaNode1 -> ... -> ViaNodeN -> ToNode
        all_nodes = [from_node] + via_nodes + [to_node]
        
        # Create sub-edges for each consecutive pair of nodes
        for i in range(len(all_nodes) - 1):
            new_row = row.copy()
            new_row['FromNode'] = all_nodes[i]
            new_row['ToNode'] = all_nodes[i + 1]
            new_row['FromStation'] = node_to_station.get(all_nodes[i], f"Unknown Node {all_nodes[i]}")
            new_row['ToStation'] = node_to_station.get(all_nodes[i + 1], f"Unknown Node {all_nodes[i + 1]}")
            expanded_rows.append(new_row)
    
    # Create a new DataFrame from the expanded rows
    expanded_df = pd.DataFrame(expanded_rows)
    return expanded_df

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
        'FromNode', 'ToNode', 'FromStation', 'ToStation'
    ]
    
    # Columns to take the first non-null value (existing information)
    first_non_null_columns = [
        'NumOfTracks', 'Bridges m', 'Tunnel m', 'TunnelTrack',
        'tot length m', 'length of 1', 'length of 2 ', 'length of 3 and more'
    ]

    # Function to extract common values for a group
    def extract_common_values(group):
        row = group.iloc[0]  # Take the first row as representative
        return {col: row[col] for col in common_columns}

    # Function to extract the first non-null value for specific columns
    def extract_first_non_null_values(group):
        return {col: group[col].dropna().iloc[0] if not group[col].dropna().empty else None
                for col in first_non_null_columns}

    # Group by the directionless edge
    grouped = df.groupby('Edge')

    # Aggregate the data
    merged_data = []
    for _, group in grouped:
        common_data = extract_common_values(group)  # Common columns
        non_null_data = extract_first_non_null_values(group)  # First non-null values
        total_frequency = group['Frequency'].sum()  # Sum of 'Frequency'
        total_travel_time = group['TravelTime'].sum() + group['InVehWait'].sum()  # Sum TravelTime and InVehWait
        merged_row = {
            **common_data,
            **non_null_data,
            'TotalFrequency': total_frequency,
            'TotalTravelTime': total_travel_time
        }
        merged_data.append(merged_row)

    # Convert the list of merged rows to a new DataFrame
    merged_df = pd.DataFrame(merged_data)

    return merged_df


def read_development_files():
    """
    Read all development files from the specified directory and filter lines with new_dev = 'Yes'.

    Returns:
        list of pd.DataFrame: A list of DataFrames, each containing filtered data for a development.
    """
    development_dir = 'data/Network/processed/developments'

    # List to store filtered DataFrames for each development
    developments = []

    # Read all `.gpkg` files in the directory
    for file_path in glob.glob(f"{development_dir}/*.gpkg"):
        try:
            # Load the GeoPackage file
            dev_gdf = gpd.read_file(file_path)

            # Filter rows where new_dev is 'Yes'
            filtered_gdf = dev_gdf[dev_gdf['new_dev'] == 'Yes']

            # Convert to DataFrame and append to the list
            developments.append(filtered_gdf)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    return developments


def process_via_column(df):
    """
    Process the Via column in a DataFrame.
    Converts strings like '1,8,8,5,,,3,5,1,,,2,4,9,7,,,1,0,1,8,' or '[2526]'
    in the Via column to lists of integers like [1885, 351, 2497, 1018].
    If no Via nodes are present, replaces it with -99.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing a Via column.

    Returns:
        pd.DataFrame: The modified DataFrame with an updated Via column.
    """
    import ast  # For safely evaluating string representations of Python literals

    def convert_to_int_list(via):
        if not isinstance(via, str) or via.strip() == "":
            return -99
        try:
            # If via is a list-like string, evaluate it safely
            if via.startswith("[") and via.endswith("]"):
                via_list = ast.literal_eval(via)
                if isinstance(via_list, list) and all(isinstance(x, int) for x in via_list):
                    return via_list
            # If via is a custom delimited string (e.g., "1,8,8,5,,,3,5,1,,,2,4,9,7,,,1,0,1,8,")
            chunks = via.split(",,,")
            int_list = [int("".join(chunk.split(","))) for chunk in chunks if chunk]
            return int_list if int_list else -99
        except (ValueError, SyntaxError):
            return -99

    if "Via" in df.columns:
        df["Via"] = df["Via"].apply(convert_to_int_list)
    else:
        print("The 'Via' column is not present in the DataFrame.")

    return df

def construction_costs(file_path, cost_per_meter, tunnel_cost_per_meter, bridge_cost_per_meter,
                       track_maintenance_cost, tunnel_maintenance_cost, bridge_maintenance_cost, duration):
    """
    Process the rail network data to calculate construction and maintenance costs for each segment.
    Includes checks for capacity adequacy, considers both directions (A and B) using the same tracks,
    and calculates costs for building additional tracks and maintaining them.

    Parameters:
        file_path (str): Path to the CSV file containing the rail network data.
        cost_per_meter (float): Cost of building a new track per meter.
        tunnel_cost_per_meter (float): Cost of updating tunnels per meter per track.
        bridge_cost_per_meter (float): Cost of updating bridges per meter per track.
        track_maintenance_cost (float): Annual maintenance cost per meter of track.
        tunnel_maintenance_cost (float): Annual maintenance cost per meter of tunnel.
        bridge_maintenance_cost (float): Annual maintenance cost per meter of bridge.
        duration (int): The duration (in years) for which maintenance costs are calculated.

    Returns:
        pd.DataFrame: Summary DataFrame containing total construction and maintenance costs for each development.
    """
    try:
        # Load the base construction cost data
        df_construction_cost = pd.read_csv(file_path, sep=";", decimal=",", encoding="utf-8-sig")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the file: {e}")

    development_costs = []  # To store costs for each development
    developments = read_development_files()
    #Process the Via column in a DataFrame.
    #Converts strings in the Via column to lists of integers
    #If no Via nodes are present, replaces it with -99.
    developments = [process_via_column(df) for df in developments]

    for i, dev_df in enumerate(developments):
        # Add the development lines to the construction cost data
        combined_df = pd.concat([df_construction_cost, dev_df], ignore_index=True)

        # Split the lines with a Via column
        df_split = split_via_nodes(combined_df)
        df_split = merge_lines(df_split)
        df_split = df_split.dropna(subset=['NumOfTracks'])

        # Calculate MinTrack as the smallest digit from the NumOfTracks column
        df_split['NumOfTracks'] = df_split['NumOfTracks'].astype(int)
        df_split['MinTrack'] = df_split['NumOfTracks'].apply(lambda x: int(min(str(x))))

        # Calculate ServicesPerTrack
        df_split['ServicesPerTrack'] = df_split['TotalFrequency'] / df_split['MinTrack']

        # Add a new column 'enoughCap' based on ServicesPerTrack < 8
        df_split['enoughCap'] = df_split['ServicesPerTrack'].apply(lambda x: 'Yes' if x < 8 else 'No')

        # Calculate costs for connections with insufficient capacity
        insufficient_capacity = df_split[df_split['enoughCap'] == 'No'].copy()

        # Generate line segments from development DataFrame
        def get_development_segments(dev_df):
            segments = []
            for _, row in dev_df.iterrows():
                from_node = row['FromNode']
                to_node = row['ToNode']
                via_nodes = row['Via'] if isinstance(row['Via'], list) else []
                
                # Create segments from FromNode -> Via -> ToNode
                prev_node = from_node
                for via_node in via_nodes:
                    segments.append((prev_node, via_node))
                    prev_node = via_node
                segments.append((prev_node, to_node))
            return segments

        # Generate segments for the current development
        development_segments = get_development_segments(dev_df)

        # Filter insufficient_capacity to include only lines in the current development
        def is_in_development(row, segments):
            return (row['FromNode'], row['ToNode']) in segments

        insufficient_capacity = insufficient_capacity[
            insufficient_capacity.apply(lambda row: is_in_development(row, development_segments), axis=1)
        ]

        # Initialize cost columns
        insufficient_capacity['NewTrackCost'] = insufficient_capacity['length of 1'] * cost_per_meter
        insufficient_capacity['NewTunnelCost'] = (
            insufficient_capacity['Tunnel m'] * (tunnel_cost_per_meter)
        )
        insufficient_capacity['NewBridgeCost'] = (
            insufficient_capacity['Bridges m'] * (bridge_cost_per_meter)
        )

        # Calculate total construction cost
        insufficient_capacity['construction_cost'] = (
            insufficient_capacity['NewTrackCost'] +
            insufficient_capacity['NewTunnelCost'] +
            insufficient_capacity['NewBridgeCost']
        )

        # Calculate maintenance cost for each segment
        insufficient_capacity['maintenance_cost'] = duration * (
            insufficient_capacity['length of 1'] * track_maintenance_cost +
            insufficient_capacity['Tunnel m'] * tunnel_maintenance_cost +
            insufficient_capacity['Bridges m'] * bridge_maintenance_cost
        )

        # Summarize total construction and maintenance costs for the current development
        total_construction_cost = insufficient_capacity['construction_cost'].sum()
        total_maintenance_cost = insufficient_capacity['maintenance_cost'].sum()
        development_costs.append({
            "Development": f"Development_{i+1}",
            "TotalConstructionCost": total_construction_cost,
            "TotalMaintenanceCost": total_maintenance_cost
        })

        # Update the base construction cost data for the next iteration
        df_construction_cost = pd.concat([df_construction_cost, dev_df], ignore_index=True)

    # Create a summary DataFrame for development costs
    development_costs_df = pd.DataFrame(development_costs)
    development_costs_df.to_csv("data/costs/construction_cost.csv", index=False)

    return development_costs_df




def maintenance_costs(duration, highway, tunnel, bridge, structural):
    generated_links_gdf = gpd.read_file(r"data/Network/processed/links_with_geometry_attributes.gpkg")
    # print(generated_links_gdf.head(10).to_string())

    generated_links_gdf["operational_maint"] = duration * (
                generated_links_gdf["hw_len"] * highway + generated_links_gdf["tunnel_len"] * tunnel +
                generated_links_gdf["bridge_len"] * bridge)

    costs_links = gpd.read_file(r"data/costs/construction.gpkg")
    costs_links["structural_maint"] = costs_links["construction_cost"] * structural * duration

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

def tif_to_vector(raster_path):
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

    # Paths for components
    travel_time_path = "data/costs/traveltime_savings.csv"
    construction_maintenance_cost = "data/costs/construction_cost.csv"
    total_costs_csv_path = "data/costs/total_costs_raw.csv"

    # Define scenarios for population and employment
    pop_scenarios = [
        "pop_urban_", "pop_equal_", "pop_rural_",
        "pop_urba_1", "pop_equa_1", "pop_rura_1",
        "pop_urba_2", "pop_equa_2", "pop_rura_2"]

    empl_scenarios = [
        "empl_urban", "empl_equal", "empl_rural",
        "empl_urb_1", "empl_equ_1", "empl_rur_1",
        "empl_urb_2", "empl_equ_2", "empl_rur_2"]

    # Load travel time savings
    c_travel_time = pd.read_csv(travel_time_path)
    c_construction =  pd.read_csv(construction_maintenance_cost)

    # Initialize total costs with zeros for missing components
    total_costs = c_travel_time.rename(columns={'development': 'ID_new'})
    total_costs["construction_cost"] = 0
    total_costs["maintenance"] = 0
    total_costs["climate_cost"] = 0
    total_costs["land_realloc"] = 0
    total_costs["nature"] = 0
    total_costs["noise_s1"] = 0
    total_costs["noise_s2"] = 0
    total_costs["noise_s3"] = 0

    # Merge TotalCost from c_construction into total_costs
    c_construction = c_construction.rename(columns={"Development": "scenario"})
    total_costs = total_costs.merge(
        c_construction[["scenario", "TotalConstructionCost"]],
        how="left",
        on="scenario"
    )

    # Update construction_cost with TotalConstructionCost values
    total_costs["construction_cost"] = total_costs["TotalConstructionCost"].fillna(0)

    # Merge TotalCost from c_construction (maintenance) into total_costs
    c_construction = c_construction.rename(columns={"Development": "scenario"})
    total_costs = total_costs.merge(
        c_construction[["scenario", "TotalMaintenanceCost"]],
        how="left",
        on="scenario"
    )

    # Update construction_cost with TotalMaintenanceCost values
    total_costs["maintenance"] = total_costs["TotalMaintenanceCost"].fillna(0)

    # Dynamically compute costs for each scenario
    for pop_scenario, empl_scenario in zip(pop_scenarios, empl_scenarios):
        total_costs[f"total_{pop_scenario}"] = (
            total_costs["construction_cost"] +
            total_costs["maintenance"] +
            total_costs.get(f"local_{pop_scenario}", 0) +
            total_costs.get(f"tt_{pop_scenario}", 0) +
            total_costs.get(f"externalities_{pop_scenario}", 0)
        )

    # Save results to CSV
    total_costs.to_csv(total_costs_csv_path, index=False)

    print(f"Total costs raw saved to {total_costs_csv_path}")

def transform_and_reshape_dataframe():
    """
    Transform, reshape, and enhance the dataframe:
    - Drop specific columns.
    - Rename 'scenario' to 'development'.
    - Calculate total costs for each scenario.
    - Add geometry and additional information for each development.
    - Save results as both CSV and GeoPackage.

    Returns:
        gpd.GeoDataFrame: Transformed GeoDataFrame with geometry column.
    """
    # Load the dataframe
    df = pd.read_csv("data/costs/total_costs_raw.csv")

    # Drop the specified columns
    columns_to_drop = [
        'status_quo_tt', 'development_tt',
        'total_pop_urban_', 'total_pop_equal_', 'total_pop_rural_',
        'total_pop_urba_1', 'total_pop_equa_1', 'total_pop_rura_1',
        'total_pop_urba_2', 'total_pop_equa_2', 'total_pop_rura_2'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Rename 'scenario' to 'development'
    df = df.rename(columns={'scenario': 'development'})

    # Reshaping the dataframe
    reshaped_df = df.pivot_table(
        index='development',
        columns='ID_new',
        values=['monetized_savings', 'construction_cost','maintenance' ],
        aggfunc='first'
    )

    # Flatten the MultiIndex columns
    reshaped_df.columns = [f"{col[0]}_{col[1]}" for col in reshaped_df.columns]

    # Reset index to have 'development' as a column
    reshaped_df = reshaped_df.reset_index()

    # Calculate total costs for each scenario
    savings_columns = [col for col in reshaped_df.columns if col.startswith("monetized_savings_")]
    
    for savings_col in savings_columns:
        scenario_name = savings_col.replace("monetized_savings_", "")
        construction_cost_col = f"construction_cost_{scenario_name}"
        maintenance_cost_col = f"maintenance_{scenario_name}"
        if construction_cost_col in reshaped_df.columns:
            reshaped_df[f"total_cost_{scenario_name}"] = reshaped_df[savings_col] + reshaped_df[construction_cost_col] + reshaped_df[maintenance_cost_col]

    # Identify columns to keep
    construction_cost_col = [col for col in reshaped_df.columns if col.startswith('construction_cost_od_matrix_combined_')][0]
    maintenance_cost_col = [col for col in reshaped_df.columns if col.startswith('maintenance_od_matrix_combined_')][0]

    # Create new columns with scaled values
    reshaped_df['Construction Cost [in Mio. CHF]'] = reshaped_df[construction_cost_col] / 1_000_000
    reshaped_df['Maintenance Costs [in Mio. CHF]'] = reshaped_df[maintenance_cost_col] / 1_000_000

    # Drop all original construction and maintenance columns
    columns_to_drop = [col for col in reshaped_df.columns if col.startswith('construction_cost_od_matrix_combined_') or col.startswith('maintenance_od_matrix_combined_')]
    reshaped_df = reshaped_df.drop(columns=columns_to_drop, errors='ignore')

    # Apply renaming function to all columns
    reshaped_df.columns = [rename_total_cost_columns(col) for col in reshaped_df.columns]

    # Adjust Net Benefit columns
    for col in reshaped_df.columns:
        if col.startswith("Net Benefit "):
            reshaped_df[col] = -reshaped_df[col] / 1_000_000
            reshaped_df = reshaped_df.rename(columns={col: f"{col} [in Mio. CHF]"})

    # Temporarily transform 'development' to integer for merging
    reshaped_df['dev_id'] = reshaped_df['development'].str.replace('Development_', '').astype(int)

    # Load the geometry and additional data
    geometry_data = gpd.read_file("data/Network/processed/updated_new_links.gpkg")[['dev_id', 'geometry','Sline']]
    geometry_data['dev_id'] = geometry_data['dev_id'] - 100000 + 1  # Adjust dev_id by subtracting 100000 and starting with ID 1

    # Merge geometry and additional information into reshaped_df
    reshaped_df = reshaped_df.merge(geometry_data, on='dev_id', how='left')

    # Restore 'Development_' prefix in the development column for CSV
    reshaped_df['development'] = 'Development_' + reshaped_df['dev_id'].astype(str)

    # Drop the temporary 'dev_id' column
    reshaped_df = reshaped_df.drop(columns=['dev_id'])

    # Rename columns
    reshaped_df.columns = [rename_monetized_savings_columns(col) for col in reshaped_df.columns]

    # Round monetized savings columns to the nearest CHF
    monetized_savings_columns = [col for col in reshaped_df.columns if col.startswith("Monetized Savings")]
    reshaped_df[monetized_savings_columns] = reshaped_df[monetized_savings_columns].round(0)

    # Reorder columns
    columns_order = (
        ['development', 'Sline', 'Construction Cost [in Mio. CHF]', 'Maintenance Costs [in Mio. CHF]'] +
        [col for col in reshaped_df.columns if col.startswith("Monetized Savings")] +
        [col for col in reshaped_df.columns if col.startswith("Net Benefit")]
    )

    # Ensure 'geometry' is placed at the very end
    columns_order += [col for col in reshaped_df.columns if col not in columns_order and col != 'geometry']
    columns_order.append('geometry')

    # Reorder the dataframe
    reshaped_df = reshaped_df[columns_order]

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(reshaped_df, geometry='geometry', crs=geometry_data.crs)

    # Save results to CSV and GeoPackage
    reshaped_df.to_csv("data/costs/total_costs_with_geometry.csv", index=False)
    gdf.to_file("data/costs/total_costs_with_geometry.gpkg", driver="GPKG")

    print("Transformed dataframe saved to:")
    print("- CSV: 'data/costs/total_costs_with_geometry.csv'")
    print("- GeoPackage: 'data/costs/total_costs_with_geometry.gpkg'")
    
    return gdf

# Rename the columns
def rename_total_cost_columns(col):
    if col.startswith("total_cost_od_matrix_combined_pop_"):
        # Extract scenario type and number
        scenario_name = col.replace("total_cost_od_matrix_combined_pop_", "")
        scenario_type = ""
        level = ""

        # Map scenario type abbreviations to full names
        if "urb" in scenario_name:
            scenario_type = "Urban"
        elif "equ" in scenario_name:
            scenario_type = "Equal"
        elif "rur" in scenario_name:
            scenario_type = "Rural"

        # Determine benefit level based on the ending
        if scenario_name.endswith("_"):
            level = "Low"
        elif scenario_name.endswith("1"):
            level = "Medium"
        elif scenario_name.endswith("2"):
            level = "High"

        # Construct the new column name
        return f"Net Benefit {scenario_type} {level}"
    return col  # Leave other columns unchanged

def rename_monetized_savings_columns(col):
    if col.startswith("monetized_savings_od_matrix_combined_pop_"):
        # Extract scenario type and number
        scenario_name = col.replace("monetized_savings_od_matrix_combined_pop_", "")
        scenario_type = ""
        level = ""

        # Map scenario type abbreviations to full names
        if "urba" in scenario_name:
            scenario_type = "Urban"
        elif "equa" in scenario_name:
            scenario_type = "Equal"
        elif "rura" in scenario_name:
            scenario_type = "Rural"

        # Determine savings level based on the ending
        if scenario_name.endswith("_"):
            level = "Low"
        elif scenario_name.endswith("1"):
            level = "Medium"
        elif scenario_name.endswith("2"):
            level = "High"

        # Construct the new column name
        return f"Monetized Savings {scenario_type} {level} [in CHF]"
    return col  # Leave other columns unchanged


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




import rasterio
from rasterio.enums import Resampling
import numpy as np

def correct_rasters_to_extent(
    empl_path, pop_path, 
    output_empl_path, output_pop_path,
    reference_raster_path):

    """
    Corrects the raster files to match the given reference raster's extent, resolution, and transform.

    Args:
        empl_path (str): Path to the employment raster file.
        pop_path (str): Path to the population raster file.
        output_empl_path (str): Path to save the corrected employment raster.
        output_pop_path (str): Path to save the corrected population raster.
        reference_raster_path (str): Path to the reference raster file.
    """
    # Read the reference raster
    with rasterio.open(reference_raster_path) as ref_src:
        ref_transform = ref_src.transform
        ref_width = ref_src.width
        ref_height = ref_src.height
        ref_crs = ref_src.crs

    # Function to process each raster
    def process_raster(input_path, output_path):
        with rasterio.open(input_path) as src:
            # Create an empty array with the reference dimensions
            data_corrected = np.zeros((src.count, ref_height, ref_width), dtype=src.dtypes[0])

            # Reproject and resample each band
            for band_idx in range(1, src.count + 1):
                rasterio.warp.reproject(
                    source=rasterio.band(src, band_idx),
                    destination=data_corrected[band_idx - 1],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear
                )

            # Update metadata and write the corrected raster
            profile = src.profile
            profile.update(
                driver="GTiff",
                height=ref_height,
                width=ref_width,
                transform=ref_transform,
                crs=ref_crs,
                nodata=np.nan
            )
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(data_corrected)

    # Process employment and population rasters
    process_raster(empl_path, output_empl_path)
    process_raster(pop_path, output_pop_path)

def GetCatchmentOD():

    # Define spatial limits of the research corridor
    # The coordinates must end with 000 in order to match the coordinates of the input raster data
    e_min, e_max = 2687000, 2708000     # 2688000, 2704000 - 2688000, 2705000
    n_min, n_max = 1237000, 1254000     # 1238000, 1252000 - 1237000, 1252000
    # Get a polygon as limits for teh corridor

    # Import the required data or define the path to access it
    catchement_tif_path = r'data/catchment_pt/catchement.tif'
    catchmentdf = gpd.read_file(r"data/catchment_pt/catchement.gpkg")

    # File paths for population and employment combined raster files

    '''
    correct_rasters_to_extent(pop_combined_file,
        empl_combined_file,
        output_empl_path="data/independent_variable/processed/scenario/empl20_corrected.tif",
        output_pop_path="data/independent_variable/processed/scenario/pop20_corrected.tif",
        reference_boundary=innerboundary,
        resolution=100,
        crs="EPSG:2056")
    '''
    # Paths to input and output files
    pop_combined_file = r"data/independent_variable/processed/scenario/pop_combined.tif"
    empl_combined_file = r"data/independent_variable/processed/scenario/empl_combined.tif"
    catchment_tif_path = r"data/catchment_pt/catchement.tif"

    output_pop_path = r"data/independent_variable/processed/scenario/pop20_corrected.tif"
    output_empl_path = r"data/independent_variable/processed/scenario/empl20_corrected.tif"

    # Correct rasters using the catchment raster as reference
    correct_rasters_to_extent(
        empl_path=empl_combined_file,
        pop_path=pop_combined_file,
        output_empl_path=output_empl_path,
        output_pop_path=output_pop_path,
        reference_raster_path=catchment_tif_path)



    
    pop_combined_file = r"data/independent_variable/processed/scenario/empl20_corrected.tif"
    empl_combined_file = r"data/independent_variable/processed/scenario/pop20_corrected.tif"

    # define dev (=ID of the polygons of a development)

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

    '''
    correct_rasters_to_extent(r"data/independent_variable/processed/raw/empl20.tif",
                              r"data/independent_variable/processed/raw/pop20.tif",
        output_empl_path="data/independent_variable/processed/raw/empl20_corrected.tif",
        output_pop_path="data/independent_variable/processed/raw/pop20_corrected.tif",
        reference_boundary=innerboundary,
        resolution=100,
        crs="EPSG:2056")
    '''
    # Paths to input and output files
    empl_raster_path = r"data/independent_variable/processed/raw/empl20.tif"
    pop_raster_path = r"data/independent_variable/processed/raw/pop20.tif"
    catchment_tif_path = r"data/catchment_pt/catchement.tif"

    output_empl_path = r"data/independent_variable/processed/raw/empl20_corrected.tif"
    output_pop_path = r"data/independent_variable/processed/raw/pop20_corrected.tif"

    # Correct rasters using the catchment raster as reference
    correct_rasters_to_extent(
        empl_path=empl_raster_path,
        pop_path=pop_raster_path,
        output_empl_path=output_empl_path,
        output_pop_path=output_pop_path,
        reference_raster_path=catchment_tif_path
    )

    
    # Open status quo
    with rasterio.open(r"data/independent_variable/processed/raw/empl20_corrected.tif") as src:
        scen_empl_20_tif = src.read(1)

    with rasterio.open(r"data/independent_variable/processed/raw/pop20_corrected.tif") as src:
        scen_pop_20_tif = src.read(1)

    #Load the catchment raster data
    with rasterio.open(catchement_tif_path) as src:
        # Read the raster data
        catchment_tif = src.read(1)  # Read the first band, which holds id information
        bounds = src.bounds  # Get the spatial bounds of the raster

    # Identify unique catchment IDs
    unique_catchment_id = np.sort(np.unique(catchment_tif))

    # Filter commune_df based on catchment raster bounds
    commune_df_filtered = commune_df.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]

    # Extract "BFS" values (unique commune IDs) within the bounds
    commune_df_filtered = commune_df_filtered["BFS"].to_numpy()

    # Ensure the OD matrix corresponds only to filtered communes
    odmat_frame = odmat.loc[commune_df_filtered, commune_df_filtered]

    # Initialize an OD matrix for catchments
    # Shape is [number of unique catchments, number of unique catchments]

    # Assume vectorized functions are defined for the below operations

    ###############################################################################################################################
    # Step 1: generate unit_flow matrix from each commune to each other commune
    cout_r = odmat / np.outer(popvec, jobvec)
    ###############################################################################################################################
    # Step 2: Get all pairs of combinations from communes to polygons
    unique_commune_id = np.sort(np.unique(commune_raster))

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
            # Debug: Print the current polygon_id and row data
            print(f"Processing polygon_id: {polygon_id}, scenario: {scen}")
            print(f"Row data: {row}")

            # Debug: Print the sum before scaling
            print(f"Sum before scaling: {od_matrix_temp.sum().sum()}")

            # Multiply all values in the row/column
            scaling_factor = row[f'{scen}']  # Extract scaling factor
            print(f"Scaling factor: {scaling_factor}")  # Debug: Check scaling factor

            od_matrix_temp.loc[polygon_id] *= scaling_factor
            od_matrix_temp.loc[:, polygon_id] *= scaling_factor

            # Debug: Print the sum after scaling
            print(f"Sum after scaling: {od_matrix_temp.sum().sum()}")

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

import os
import pandas as pd

def combine_and_save_od_matrices(directory, status_quo_directory):
    """
    Combines population and employment OD matrices for each scenario into a single file.
    Renames the original files with '_original' and saves the combined file in the same directory.
    For status quo scenarios (ending in '20'), saves them in a different directory.

    Parameters:
        directory (str): Directory containing the OD matrix files.
        status_quo_directory (str): Directory to save status quo OD matrices.
    """
    # Ensure the status quo directory exists
    os.makedirs(status_quo_directory, exist_ok=True)

    # Define scenarios for population and employment
    pop_scenarios = [
        "pop_urban_", "pop_equal_", "pop_rural_",
        "pop_urba_1", "pop_equa_1", "pop_rura_1",
        "pop_urba_2", "pop_equa_2", "pop_rura_2"]

    empl_scenarios = [
        "empl_urban", "empl_equal", "empl_rural",
        "empl_urb_1", "empl_equ_1", "empl_rur_1",
        "empl_urb_2", "empl_equ_2", "empl_rur_2"]

    # Process status quo files separately
    status_quo_pop_file = os.path.join(directory, "od_matrix_pop_20.csv")
    status_quo_empl_file = os.path.join(directory, "od_matrix_empl_20.csv")

    if os.path.exists(status_quo_pop_file) and os.path.exists(status_quo_empl_file):
        # Load status quo files
        pop_df = pd.read_csv(status_quo_pop_file)
        empl_df = pd.read_csv(status_quo_empl_file)

        # Combine the two matrices
        combined_df = pop_df.set_index("catchment_id") + empl_df.set_index("catchment_id")
        combined_df.reset_index(inplace=True)

        # Save the combined status quo file
        save_path = os.path.join(status_quo_directory, "od_matrix_combined_20.csv")
        combined_df.to_csv(save_path, index=False)
        print(f"Saved combined status quo OD matrix: {save_path}")

        # Rename the original files
        pop_renamed_path = os.path.join(directory, "od_matrix_pop_20_original.csv")
        empl_renamed_path = os.path.join(directory, "od_matrix_empl_20_original.csv")

        if os.path.exists(pop_renamed_path):
            os.remove(pop_renamed_path)  # Remove the existing file
        if os.path.exists(empl_renamed_path):
            os.remove(empl_renamed_path)  # Remove the existing file

        os.rename(status_quo_pop_file, pop_renamed_path)
        os.rename(status_quo_empl_file, empl_renamed_path)
        print(f"Renamed original status quo files.")
    else:
        print("Missing status quo files: od_matrix_pop_20.csv, od_matrix_empl_20.csv")

    # Iterate through other scenarios to combine files
    for pop_scenario, empl_scenario in zip(pop_scenarios, empl_scenarios):
        pop_file = f"od_matrix_{pop_scenario}.csv"
        empl_file = f"od_matrix_{empl_scenario}.csv"

        pop_path = os.path.join(directory, pop_file)
        empl_path = os.path.join(directory, empl_file)

        # Check if both files exist
        if os.path.exists(pop_path) and os.path.exists(empl_path):
            # Load the population and employment files
            pop_df = pd.read_csv(pop_path)
            empl_df = pd.read_csv(empl_path)

            # Combine the two matrices by summing their values
            combined_df = pop_df.set_index("catchment_id") + empl_df.set_index("catchment_id")
            combined_df.reset_index(inplace=True)

            # Save the combined file
            save_path = os.path.join(directory, f"od_matrix_combined_{pop_scenario}.csv")
            combined_df.to_csv(save_path, index=False)
            print(f"Saved combined OD matrix: {save_path}")

            # Rename the original files
            pop_renamed_path = os.path.join(directory, pop_file.replace(".csv", "_original.csv"))
            empl_renamed_path = os.path.join(directory, empl_file.replace(".csv", "_original.csv"))

            if os.path.exists(pop_renamed_path):
                os.remove(pop_renamed_path)  # Remove the existing file
            if os.path.exists(empl_renamed_path):
                os.remove(empl_renamed_path)  # Remove the existing file

            os.rename(pop_path, pop_renamed_path)
            os.rename(empl_path, empl_renamed_path)
            print(f"Renamed original files: {pop_path} -> {pop_renamed_path}, {empl_path} -> {empl_renamed_path}")

    # Remove all original files
    for file_name in os.listdir(directory):
        if "_original.csv" in file_name:
            file_path = os.path.join(directory, file_name)
            os.remove(file_path)
            print(f"Deleted original file: {file_path}")



def GetVoronoiOD_multi():
    voronoi_tif_path = r"data/Network/travel_time/source_id_raster.tif"
    scen_empl_path = r"data/independent_variable/processed/scenario/scen_empl.tif"
    scen_pop_path = r"data/independent_variable/processed/scenario/scen_pop.tif"

    popvec = GetCommunePopulation(y0="2021")
    jobvec = GetCommuneEmployment(y0=2021)
    od = GetHighwayPHDemandPerCommune()
    odmat = GetODMatrix(od)

    # This function returns a np array of raster data storing the bfs number of the commune in each cell
    commune_raster, commune_df = GetCommuneShapes(raster_path=voronoi_tif_path)

    if jobvec.shape[0] != odmat.shape[0]:
        print(
            "Error: The number of communes in the OD matrix and the number of communes in the employment data do not match.")

    # Open scenario (medium) raster data    (low = band 2, high = band 3)
    with rasterio.open(scen_pop_path) as src:
        # Read the raster into a NumPy array (assuming you want the first band)
        scen_pop_medium_tif = src.read(1)
        scen_pop_low_tif = src.read(2)
        scen_pop_high_tif = src.read(3)

    with rasterio.open(scen_empl_path) as src:
        # Read the raster into a NumPy array (assuming you want the first band)
        scen_empl_medium_tif = src.read(1)
        scen_empl_low_tif = src.read(2)
        scen_empl_high_tif = src.read(3)

    # Step 1: generate unit_flow matrix from each commune to each other commune
    cout_r = odmat / np.outer(popvec, jobvec)

    # Directory path to developments
    directory_path = "data/Network/travel_time/developments/"

    # List to hold extracted values
    xx_values = []

    # Iterate through files in the directory
    for filename in os.listdir(directory_path):
        # Check if the filename matches the pattern 'devXX_source_id_raster.tif'
        match = re.match(r'dev(\d+)_source_id_raster\.tif', filename)
        if match:
            # Extract XX value and add to the list
            xx = match.group(1)
            xx_values.append(xx)

    # Convert values to integers if needed
    xx_values = [int(xx) for xx in xx_values]
    print(len(xx_values))

    for xx in tqdm(xx_values, desc='Processing Voronoi IDs'):
        # Construct the file path
        file_path = f"{directory_path}dev{xx}_source_id_raster.tif"

        # Open the file with rasterio
        with rasterio.open(file_path) as src:
            # Read the raster data
            voronoi_tif = src.read(1)

        unique_voronoi_id = np.sort(np.unique(voronoi_tif))

        # Step 2: Get all pairs of combinations from communes to polygons
        unique_commune_id = np.sort(np.unique(commune_raster))
        pairs = pd.DataFrame(columns=['commune_id', 'voronoi_id'])
        pop_empl = pd.DataFrame(columns=['commune_id', 'voronoi_id', "empl", "pop"])

        for i in unique_voronoi_id:
            # Get the voronoi raster
            mask_voronoi = voronoi_tif == i
            for j in unique_commune_id:
                if j > 0:
                    # Get the commune raster
                    mask_commune = commune_raster == j
                    # Combined mask
                    mask = mask_commune & mask_voronoi
                    # Check if there are overlaying values
                    if np.nansum(mask) > 0:
                        # pairs = pairs.append({'commune_id': j, 'voronoi_id': i}, ignore_index=True)
                        temp = pd.Series({'commune_id': j, 'voronoi_id': i})
                        pairs = gpd.GeoDataFrame(
                            pd.concat([pairs, pd.DataFrame(temp).T], ignore_index=True))

                        # Get the population and employment values for multiple scenarios
                        pop_low = scen_pop_low_tif[mask]
                        empl_low = scen_empl_low_tif[mask]
                        pop_medium = scen_pop_medium_tif[mask]
                        empl_medium = scen_empl_medium_tif[mask]
                        pop_high = scen_pop_high_tif[mask]
                        empl_high = scen_empl_high_tif[mask]

                        temp = pd.Series({'commune_id': j, 'voronoi_id': i,
                                          'pop_low': np.nansum(pop_low), 'empl_low': np.nansum(empl_low),
                                          'pop_medium': np.nansum(pop_medium),
                                          'empl_medium': np.nansum(empl_medium),
                                          'pop_high': np.nansum(pop_high), 'empl_high': np.nansum(empl_high)})
                        pop_empl = gpd.GeoDataFrame(
                            pd.concat([pop_empl, pd.DataFrame(temp).T], ignore_index=True))
                        # pop_empl = pop_empl.append({'commune_id': j, 'voronoi_id': i,
                        #                            'pop_low': np.nansum(pop_low), 'empl_low': np.nansum(empl_low),
                        #                            'pop_medium': np.nansum(pop_medium),
                        #                            'empl_medium': np.nansum(empl_medium),
                        #                            'pop_high': np.nansum(pop_high), 'empl_high': np.nansum(empl_high)},
                        #                           ignore_index=True)
                else:
                    continue

        # Step 3 complete exploded matrix
        # Initialize the OD matrix DataFrame with zeros or NaNs
        tuples = list(zip(pairs['voronoi_id'], pairs['commune_id']))
        multi_index = pd.MultiIndex.from_tuples(tuples, names=['voronoi_id', 'commune_id'])
        temp_df = pd.DataFrame(index=multi_index, columns=multi_index).fillna(0).to_numpy('float')
        od_matrix = pd.DataFrame(data=temp_df, index=multi_index, columns=multi_index)

        # Handle raster without values
        # Drop pairs with 0 pop or empl

        # Get the set of destination commune id
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

        # todo fill with according values

        # Check for scenario based on column names in pop_empl
        # Sceanrio are defined like pop_XX and empl_XX get a list of all these endings (only XX)
        # Get the column names of pop_empl
        pop_empl_columns = pop_empl.columns
        # Get the column names that end with XX
        pop_empl_scenarios = [col.split("_")[1] for col in pop_empl_columns if col.startswith("pop_")]

        # SEt index of df to access its single components
        pop_empl = pop_empl.set_index(['voronoi_id', 'commune_id'])

        # for each of these scenarios make an own copy of od_matrix named od_matrix+scen
        for scen in pop_empl_scenarios:
            # print(f"Processing scenario {scen} and development {xx}")
            od_matrix_temp = od_matrix.copy()

            for polygon_id, row in pop_empl.iterrows():
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
            np.fill_diagonal(od_grouped.values, 0)

            # Save pd df to csv
            od_grouped.to_csv(fr"data/traffic_flow/od/developments/od_matrix_dev{xx}_{scen}.csv")
            # odmat.to_csv(r"data/traffic_flow/od/od_matrix_raw.csv")

    return


def link_traffic_to_map():
    # Import travel flows from matrix to df, no index, set column name to flow
    # flow = pd.read_csv(r"data/traffic_flow/Xi_sum.csv", header=None, index_col=False)
    flow = pd.read_csv(r"data/traffic_flow/developments/D_i/Xi_sum_status_quo_20.csv", header=None, index_col=False)
    flow.columns = ['flow']
    print(flow.head(10).to_string())

    # Import data with links
    edges = gpd.read_file(r"data/Network/processed/edges_with_attribute.gpkg")
    print(edges.head(10).to_string())

    # Compare lenght of dataframes
    print(f"Length of edges df: {len(edges)}")
    print(f"Length of flow df: {len(flow)}")

    # Sort edges by edge_ID
    edges["ID_edge"] = edges["ID_edge"].astype(int)
    edges = edges.sort_values(by=['ID_edge'])

    # Add flow column to edges df
    edges['flow'] = flow['flow']

    print(edges.head(10).to_string())

    # Only keep column capacity, flow and geometry
    edges = edges[['ID_edge', 'geometry', 'flow']]
    # Safe file
    edges.to_file(r"data/Network/processed/edges_only_flow.gpkg")

    # Compare values to calibrate to tau value when creating the OD matrix
    # Edge ID 94 -> Tagesverkehr 3028 (DTV 54014)
    # Edge ID 95 -> Tagesverkehr  3034 (DTV 53867)
    # Edge ID 88 -> Tagesverkehr  1103 (DTV 18852)
    # Edge ID 90 -> Tagesverkehr 1087 (DTV 18547)
    # Print a table comparing the flow (edges["flow"] values in edges for ID mentioned above and the Tagesverkehr values

    # print(f"Link 94 - modelled flow: {edges.loc[edges['ID_edge'] == 94, 'flow'].iloc[0]} and measured flow: 3028")
    # print(f"Link 95 - modelled flow: {edges.loc[edges['ID_edge'] == 95, 'flow'].iloc[0]} and measured flow: 3034")
    # print(f"Link 88 - modelled flow: {edges.loc[edges['ID_edge'] == 88, 'flow'].iloc[0]} and measured flow: 1103")
    # print(f"Link 90 - modelled flow: {edges.loc[edges['ID_edge'] == 90, 'flow'].iloc[0]} and measured flow: 1087")


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################


def convert_data_to_input(points, edges):
    # Set crs for points and edges to epsg:2056
    points = points.set_crs("epsg:2056", allow_override=True)
    edges = edges.set_crs("epsg:2056", allow_override=True)
    # Print crs of points and edges
    # print(f"Points crs: {points.crs}")
    # print(f"Edges crs: {edges.crs}")

    # Change "corridor_border" to False if "within_corridor" is True
    points.loc[points["within_corridor"] == True, "on_corridor_border"] = False

    # Define new column to state if node generates traffic in model
    # When point is in corridor or on border, it generates traffic
    # points["generate_traffic"] = points["within_corridor"] | points["on_corridor_border"]
    # points["generate_traffic"] = np.logical_or(points['within_corridor'], points['on_corridor_border'])
    points["generate_traffic"] = np.logical_or(np.array(points['within_corridor']) == '1',
                                               np.array(points['on_corridor_border']) == '1')

    #######################################################################################################################
    # Store values as needed for the model

    # Assert nodes and edges are sorted by ID
    # points["ID_point"] = points["ID_point"].astype(int)
    # points = points.sort_values(by=["ID_point"])
    points.index = points.index.astype(int)
    points = points.sort_index()
    edges["ID_edge"] = edges["ID_edge"].astype(int)
    edges = edges.sort_values(by=["ID_edge"])

    # Nodes: store dict of coordinates of all nodes nodes = [[x1, y1], [x2, y2], ...] <class 'numpy.ndarray'>
    nodes_lv95 = points[["geometry"]].to_numpy()
    # Same with coordinates converted to wgs84
    nodes_wgs84 = points[["geometry"]].to_crs("epsg:4326").to_numpy()

    # Edges: store dict of coordinates of all edges links = [[id_start_node_1, id_end_node_1], [id_start_node_2, id_end_node_2], ...] <class 'numpy.ndarray'>
    links = edges[["start", "end"]].to_numpy(int)

    # Length of edges stored a array link_length_i = [[length_1], [length_2], ...] <class 'numpy.ndarray'>
    link_length_i = edges["geometry"].length.to_numpy(float)

    # Number of edges
    nlinks = len(edges)

    # Travel time on each link assuming free flow speed
    # Get edge length
    edges["length"] = edges["geometry"].length / 1000  # in kilometers
    # Calculate free flow travel time on all edges
    edges["fftt_i"] = edges["length"] / pd.to_numeric(edges["ffs"])  # in hours
    # Store these values in a dict as fftt_i = [[fftt_1], [fftt_2], ...] <class 'numpy.ndarray'>
    fftt_i = edges[["fftt_i"]].to_numpy(float)

    # Capacity on each link
    # Store these values in a dict as capacity_i = [[capacity_1], [capacity_2], ...] <class 'numpy.ndarray'>
    Xmax_i = edges[["capacity"]].to_numpy(float)

    # Store same alpha and gamma for all links as alpha_i = [[alpha_1], [alpha_2], ...] <class 'numpy.ndarray'>
    alpha = 0.25
    gamma = 2.4
    alpha_i = np.tile(alpha, Xmax_i.shape)
    gamma_i = np.tile(gamma, Xmax_i.shape)

    par = {"fftt_i": fftt_i, "Xmax_i": Xmax_i, "alpha_i": alpha_i, "gamma_i": gamma_i}

    return nodes_lv95, nodes_wgs84, links, link_length_i, nlinks, par

def create_rail_graph(points,edges,changes):
    # Create an empty NetworkX graph
    G = nx.MultiGraph()
    # print(points.head(5).to_string())
    # Add nodes with IDs to the graph
    for node_id, row in points.iterrows():
        G.add_node(node_id, pos=(row['geometry'].x, row['geometry'].y), demand=row['generate_demand'])
    # For every node, we will create sub-nodes A and B for every service Sx (e.g., S5).
    # In direction A, transit passengers will arrive at sub-node A and depart from sub-node A. The edge between
    # A and B will be the in-vehicle wait time (default 0) or change time between services (default 6).
    # Passengers accessing or egressing the service at the station are allocated to the original node defined above.
    # Access/egress time is defined as 12 minutes, to ensure routing logic, whereby 24 minutes are subtracted at the end.

    for idx, row in edges.iterrows():
        temp_from_x = str(row['FromNode']) + "_" + str(row['Service']) + "_" + row['Direction'] +"_dep"
        G.add_node(temp_from_x, pos=(row['FromGeometry'].x, row['FromGeometry'].y), demand=False)
        G.add_edge(row['FromNode'], temp_from_x, key=(str(row['ID_edge'])+temp_from_x), fftt=.2)
        temp_to_y = str(row['ToNode']) + "_" + str(row['Service']) + "_" + row['Direction'] +"_arr"
        G.add_node( temp_to_y, pos=(row['ToGeometry'].x, row['ToGeometry'].y), demand=False)
        G.add_edge(temp_to_y, row['ToNode'], key=(str(row['ID_edge'])+temp_to_y), fftt=.2)

        G.add_edge(temp_from_x, temp_to_y, key=row['ID_edge'], fftt=pd.to_numeric(row['TravelTime']))

    for idx, row in changes.iterrows():
        temp_arr_x = str(row['FromNode']) + "_" + str(row['Service']) + "_" + row['Direction'] +"_arr"
        dep = row[6:11][row[9:14] != '9000']
        for depidx, depvalue in dep.items():
            temp_dep_i = str(row['FromNode']) + "_" + str(depidx.split("-")[0]) + "_A"+ "_dep"
            G.add_edge(temp_arr_x, temp_dep_i, key=(temp_arr_x + "_" + temp_dep_i), fftt=depvalue)
            temp_dep_i = str(row['FromNode']) + "_" + str(depidx.split("-")[0]) + "_B" + "_dep"
            G.add_edge(temp_arr_x, temp_dep_i, key=(temp_arr_x + "_" + temp_dep_i), fftt=depvalue)

    return G

def get_nw_data(OD_matrix, points, voronoi_gdf, edges):
    # Adapt OD matrix
    # nodes within perimeter and on border
    ####################################################
    ### Define zones
    ### Check for 2050

    # Filter points to only keep those within the corridor or on border
    # points_in = points[(points["within_corridor"] == True) | (points["on_corridor_border"] == True)]
    points_in = points[
        np.logical_or(np.array(points['within_corridor']) == '1', np.array(points['on_corridor_border']) == '1')]
    # print(f"Points in corridor or border: {points_in.shape}")
    # print(points_in.head(5).to_string())
    # print(voronoi_gdf.head(5).to_string())
    # Get common ID_point and voronoi_ID as list
    # common_ID = list(set(points_in["ID_point"]).intersection(set(voronoi_gdf["ID_point"])))
    common_ID = list(set(pd.to_numeric(points_in["ID_point"])) & set(voronoi_gdf["ID_point"]))

    # print(f"\n\n\n max value in ID_point: {max(voronoi_gdf['ID_point'])}")
    # print(f"Point in polygon and with voronoi: {len(common_ID)}")

    # new column "generate_demand" where ID_point is in common_ID
    points["generate_demand"] = points["ID_point"].astype(int).isin(common_ID)

    # Filter OD matrix to only keep common ID in rows and columns
    # Convert ID elements to the appropriate type if necessary
    common_ID = [int(id) for id in common_ID]
    OD_matrix.index = OD_matrix.index.map(lambda x: int(float(x)))
    OD_matrix.columns = OD_matrix.columns.map(lambda x: int(float(x)))

    OD_matrix = OD_matrix.loc[common_ID, common_ID]
    # print(f"Shape OD matrix: {OD_matrix.shape}")

    # flatten OD matrix to 1D array as D_od
    D_od = OD_matrix.to_numpy().flatten()

    # Get the amount of values in the OD matrix as nOD
    nOD = len(D_od)
    # print(nOD)

    # Map the single zones of the OD to actual nodes in the network
    # nodes within perimeter and on border
    # Get nodes in zones -> then build network with these nodes

    # Build network using networkx
    # Map the coordinates to the edges DataFrame
    # Set the index of the nodes DataFrame to be the 'ID_point' column
    points.set_index('ID_point', inplace=True)
    #todo automate the way that these changes are generated.

    G = create_rail_graph(points,edges,changes)
    """
    # Plot graph small points with coordinates as position and color based on demand attribute
    nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), node_size=3, node_color=list(nx.get_node_attributes(G, 'demand').values()), edge_color='black', width=0.5)
    #nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), node_size=1, node_color='black', edge_color='black', width=0.5)
    plt.show()
    """

    # Compute the route for each OD pair (maybe limit to max 5)
    # Get routes for all points with demand = True in the network G
    # Get all nodes with demand = True
    # nx.all_simple_edge_paths() # (u,v,k) -> u,v are nodes, k is the key of the edge

    # Step 1: Identify nodes with demand
    demand_nodes = [n for n, attr in G.nodes(data=True) if attr.get('demand') == True]
    if len(demand_nodes) < 29:
        print(f"Number of points considered: {len(demand_nodes)} ({len(demand_nodes) * len(demand_nodes)})")
    """
    # Find connected components
    connected_components = list(nx.connected_components(G))
    # Filter components to include only those with at least one node having 'demand' == True
    #connected_components_with_demand = [comp for comp in connected_components if any(G.nodes[node]['demand'] for node in comp)]

    print(connected_components)

    # Convert the generator to a list to get its length
    connected_components_list = list(connected_components)

    # To plot each connected component in a different color, we can use a color map
    colors = plt.cm.rainbow(np.linspace(0, 1, len(connected_components_list)))

    # Create a plot
    plt.figure(figsize=(8, 6))

    # For each connected component, using a different color for each
    for comp, color in zip(connected_components_list, colors):
        # Separate nodes with demand and without demand
        nodes_with_demand = [node for node in comp if G.nodes[node]['demand']]
        nodes_without_demand = [node for node in comp if not G.nodes[node]['demand']]

        # Get positions for all nodes in the component
        pos = nx.get_node_attributes(G, 'pos')  # Use existing positions if available

        # Draw nodes with demand
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_with_demand, node_color=[color], node_size=20)
        # Draw nodes without demand
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_without_demand, node_color=[color], node_size=3)
        # Draw edges
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges() if u in comp and v in comp])

        # Draw node labels in small size
        nx.draw_networkx_labels(G, pos, font_size=2)

    # Show the plot
    plt.title("Connected Components with Individual Colors")
    plt.axis('off')  # Turn off axis
    # safe plot
    plt.savefig(fr"plot/results/connected_components.png", dpi=500)
    plt.show()
    """

    index_routes = 0
    index_OD_pair = 0
    delta_odr = np.zeros((nOD, 1000000))
    routelinks_list = []
    # delta_ir = np.zero((nlinks, 10000)

    # Convert MultiDiGraph or MultiGraph to DiGraph or Graph
    G_simple = nx.Graph(G)

    def k_shortest_paths_edge_ids(G, source, target, k, weight=None):
        return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

    for i in range(len(demand_nodes)):
        for j in range(len(demand_nodes)):
            source = demand_nodes[i]
            target = demand_nodes[j]
            unique_paths_ij = []

            paths = k_shortest_paths_edge_ids(G_simple, source, target, 2, weight='fftt')

            for path in paths:
                edge_ids = [list(G[u][v])[0] for u, v in zip(path[:-1], path[1:])]  # Assuming G is a DiGraph or Graph

                # Check if this path is already in unique_paths
                if edge_ids not in unique_paths_ij:
                    unique_paths_ij.append(edge_ids)
            # print(unique_paths_ij)
            for edge_ids in unique_paths_ij:
                routelinks_list.append(edge_ids)
                delta_odr[index_OD_pair][index_routes] = 1
                index_routes += 1

            index_OD_pair += 1
            # print(f"Number of routes: {index_routes} and OD pairs: {index_OD_pair}")

    # print(f"Number of routes: {index_routes}")
    # print(f"Number of OD pairs: {index_OD_pair}")

    max_length = max(len(arr) for arr in routelinks_list)
    # Replace all 0s with -1
    routelinks_list = [[-1 if element == 0 else element for element in sublist] for sublist in routelinks_list]

    # Create a 2D array, padding shorter arrays with zeros
    routelinks_same_len = [arr + [0] * (max_length - len(arr)) for arr in routelinks_list]
    routelinks = np.array(routelinks_same_len)

    # Get the number of rows in array1
    n_routes = routelinks.shape[0]

    ## Assert the algorithm works well
    column_sums = np.sum(delta_odr, axis=0)
    # print(f"The number of routes {np.sum(column_sums >= 1)} and {n_routes}")
    if np.sum(column_sums >= 1) == n_routes:
        delta_odr = delta_odr[:, :n_routes]
    else:
        print("not same amount of routes in computation - check code for errors")

    # Example edge keys and 2D numpy array
    edge_keys = [k for u, v, k in G.edges(keys=True)]  # Replace with your actual edge keys
    # Replace all 0s with -1
    edge_keys = [-1 if element == 0 else element for element in edge_keys]

    # Initialize DataFrame with edge IDs as columns and zeros
    delta_ir_df = pd.DataFrame(0, index=np.arange(len(routelinks)), columns=edge_keys)

    # Fill the DataFrame
    ####################################################################################################################
    ######## This is not filling correctly

    # From a matrix with all edge IDs for each route (routelinks) route ID (x) and edge ID (y) just in a list
    # to a matrix with all routes (x) and all edges (y) as delta_ir (binary if edge in route)
    for row_idx, path in enumerate(routelinks):
        # print(f"row_idx: {row_idx} and path: {path}")
        for edge in path:
            if edge != 0:
                delta_ir_df.at[row_idx, edge] = 1
        # print entire row
        # print(delta_ir_df.iloc[row_idx])

    # Sort in ascending edge_id and store it as array
    # print(delta_ir_df.sort_index(axis=1).transpose().head(10).to_string())
    delta_ir = delta_ir_df.sort_index(axis=1).transpose().to_numpy()
    """
    print(f"Amount of links used: {np.sum(delta_ir > 0)} (delta_ir) and {np.sum(routelinks > 0)} (routelinks)")

    print(f"Shape delta_odr {delta_odr.shape} (OD pairs x #routes)")
    print(f"Shape delta_ir {delta_ir.shape} (#links (edge ID sorted ascending) x #routes)")
    print(f"Shape routelinks {routelinks.shape} (#routes x amount of links of longest route)")
    print(f"Shape D_od {D_od.shape[0]} ({math.sqrt(D_od.shape[0])} OD zones)")
    print(f"nOD number of OD pairs {nOD}")
    print(f"number of routes (n_routes) {n_routes}")
    """
    # paths = list(k_shortest_paths(G, source, target, 3))

    # Store a matrix with edge_ID for each route (routelinks) route ID (x) and edge ID (y)

    # Store a matrix with all OD pairs (x) and all route (y) as delta_odr (binary if route serves OD pair)

    # Store a matrix with all links (x) and all edges (y) as delta_ir (binary if edge in route)
    return delta_ir, delta_odr, routelinks, D_od, nOD, n_routes


def CostFun(Xi, par):
    # Computes the cost function for the flow Xi, with the parameters 'par', Xi has the adequate size

    # BPR
    s1 = np.power(np.divide(Xi, par['Xmax_i']), par['gamma_i'])
    s2 = np.multiply(par['alpha_i'], s1)
    Ci = np.multiply(par['fftt_i'], np.add(1, s2))
    # Ci=par.fftt_i.*(1+par.alpha_i.*(Xi./par.Xmax_i).^par.gamma_i);
    return Ci


def IntCostFun(par):
    # Computes the integral of the cost function for the flow Xi, with the
    # parameters 'par'. Xi has the adequate size
    Xi = 2
    # BPR
    # s1 = (Xi./par.Xmax_i).^par.gamma_i            (Flow/MaxCapacity)^gamma
    # s2 = Xi*par.alpha_i.*par.fftt_i.*s1           Flow*alpha*fftt*s1
    # Ci = Xi.*par.fftt_i + s2./(par.gamma_i + 1)   Flow*fftt + s2/(gamma+1)

    s1 = np.power(np.divide(Xi, par['Xmax_i']), par['gamma_i'])  # (Xi./par.Xmax_i).^par.gamma_i
    s2 = np.multiply(Xi,
                     np.multiply(par['alpha_i'], np.multiply(par['fftt_i'], s1)))  # (Xi.*par.alpha_i.*par.fftt_i.*s1)
    Ci = np.multiply(Xi, par['fftt_i']) + np.divide(s2, (
        np.add(par['gamma_i'], 1)))  # Xi.*par.fftt_i + s2./(par.gamma_i + 1);
    # Ci=Xi.*par.fftt_i + (Xi.*par.alpha_i.*par.fftt_i.*(Xi./par.Xmax_i).^par.gamma_i)./(par.gamma_i + 1);

    # returns travel cost for each link
    return Ci


def SUE_C_Logit(D_od, par, delta_ir, delta_odr, cf_r):
    # De acuerdo con C logit SUE_Zhou (2010)

    # --- Optimizacion NO lineal
    # objfun = @(D_r)( sum(IntLinksTimes(D_r)) + 1/theta*sum(D_r.*log(D_r)) + sum(D_r.*cf_r) );
    # A=[];b=[];  # A x <= b
    # Aeq=delta_odr;beq=D_od; # Aeq x = beq
    # lb=zeros(nroutes,1);ub=max(D_od)*ones(nroutes,1);
    # D_r0=delta_odr.H*(D_od./sum(delta_odr,2)); #estimacion: reparto equitativamente #zeros(nroutes,1);#
    ##options=optimset('Algorithm','interior-point','MaxFunEvals',1e5);#,'Display','off'); #matlab viejo
    # options=optimoptions('fmincon','Algorithm','sqp','MaxFunEvals',1e5,'Display','off');
    # [D_r,fval,exitflag,output]=fmincon(objfun,D_r0,A,b,Aeq,beq,lb,ub,[],options);
    # x_i=delta_ir*D_r

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- Non-linear optimisation using Sequential least squares programming

    def IntLinksTimes(D_r):
        # Demand on each link from demand on routes
        intTrec_i = IntCostFun(par)  # integral de la funcion de coste
        return intTrec_i

    # def bpr(x):

    def fun(x):
        ############3 Thy not theta?
        thetavec = np.ones_like(x)
        # eqval = np.sum(IntLinksTimes(x),axis=0) + 1/theta*np.sum(np.multiply(x,np.log(x)),axis=1) + np.sum((np.multiply(x,cf_r)),axis=1)

        # X contains 2150 zeros

        # print(np.multiply(x,np.log(x)))
        # print(np.sum(np.divide(np.multiply(x,np.log(x)),thetavec)))

        # IntLinksTimes()           Travel time on each link

        # np.multiply(x,cf_r)       Commonality factor on each route with optimize flow
        ############################################################################################################
        ### Wha to divide by 1?
        # print(np.sum(IntLinksTimes(x)) + np.sum(np.divide(np.multiply(x, np.log(x)), thetavec)) + np.sum((np.multiply(x, cf_r))))
        # print(np.multiply(x, np.log(x)))
        # Count the amount of nan in x
        # print(np.sum(np.isnan(np.multiply(x, np.log(x)))))  # This creates nan (like 30)
        # print(np.sum(np.isnan(np.multiply(x,cf_r))))
        # print(cf_r)
        # print(np.sum(np.isinf(cf_r)))
        # print(np.multiply(x,cf_r))

        # Check values in np.multiply(x,cf_r), number of nan and number of inf
        # print(f"Number of nan in np.multiply(x,cf_r): {np.sum(np.isnan(np.multiply(x,cf_r)))} and number of inf: {np.sum(np.isinf(np.multiply(x,cf_r)))}")
        # Same for np.multiply(x,np.log(x))
        # print(f"Number of nan in np.multiply(x,np.log(x)): {np.sum(np.isnan(np.multiply(x,np.log(x))))} and number of inf: {np.sum(np.isinf(np.multiply(x,np.log(x))))}")
        # Same for x
        # print(f"Number of nan in x: {np.sum(np.isnan(x))}, number of inf: {np.sum(np.isinf(x))}, number of zeros: {np.sum(x==0)}")
        # And for log(x)
        # print(f"Number of nan in log(x): {np.sum(np.isnan(np.log(x)))}, number of inf: {np.sum(np.isinf(np.log(x)))}, number of zeros: {np.sum(np.log(x)==0)}")

        """
        #Check if 0 in x if so print the optimization iteration
        if 0 in x:
            #print("0 in x")
            print(f"There are 0 in x   {np.sum(x==0)}")

        # Check for NaN or Inf in x
        if np.any(np.isnan(x)) or np.any(np.isinf(x)):
            print("NaN or Inf found in x:", x)
        """
        # If x is smaller than 0 replace with 0.001
        x[x <= 0] = 0.0001
        temp_log = np.log(x)

        # replace inf with 0
        temp_log[np.isinf(temp_log)] = 0.1
        temp_log[np.isnan(temp_log)] = 0.1
        """
        # Check if inf or nan in np.log(x)
        if np.sum(np.isinf(np.log(x))) > 0:
            print(f"inf in log(x)  {np.sum(np.isinf(np.log(x)))}")
        if np.sum(np.isnan(np.log(x))) > 0:
            print(f"nan in log(x)  {np.sum(np.isnan(np.log(x)))}")
        """

        # Replace all nan values in x with 0
        # x[np.isnan(x)] = 0

        result = np.sum(IntLinksTimes(x)) + np.sum(np.divide(np.multiply(x, temp_log), thetavec)) + np.sum(
            (np.multiply(x, cf_r)))
        if np.sum(IntLinksTimes(x) < 0) > 0:
            print(f"Neagtive numbers in IntLinksTimes(x): {np.sum(IntLinksTimes(x) > 0)}")
        if np.isnan(np.sum(IntLinksTimes(x))):
            print("NaN in IntLinksTimes:", result)
        if np.isnan(np.sum(np.divide(np.multiply(x, temp_log), thetavec))):
            print("NaN in np.divide(np.multiply(x,np.log(x)),thetavec):",
                  np.sum(np.isnan(np.divide(np.multiply(x, temp_log), thetavec))))
            print("NaN in np.multiply(x,np.log(x))):", np.sum(np.isnan(np.multiply(x, temp_log))))
            print("NaN in x:", np.sum(np.isnan(x)))
            # Print amount of negative values in x
            print(f"Negative values in x: {np.sum(x < 0)}")
        if np.isnan(np.sum((np.multiply(x, cf_r)))):
            print("NaN in np.multiply(x,cf_r):", result)
        # same for inf
        if np.isinf(np.sum(IntLinksTimes(x))):
            print("Inf in IntLinksTimes:", result)
        if np.isinf(np.sum(np.divide(np.multiply(x, temp_log), thetavec))):
            print("Inf in np.divide(np.multiply(x,np.log(x)),thetavec):", result)
        if np.isinf(np.sum((np.multiply(x, cf_r)))):
            print("Inf in np.multiply(x,cf_r):", result)
        return result

        # s=np.squeeze(eqval)
        # val=float(s)
        # val=float(eqval)
        # return eqval

    # def fun_der(x):
    #     der = np.zeros_like(x)
    #     s0 = np.matmul(delta_ir,x)
    #     s1 = np.power(np.divide(s0,par['Xmax_i']),par['gamma_i'])
    #     s2 = np.multiply(par['alpha_i'],s1)
    #     s3 = np.multiply(par['fftt_i'],s2)
    #     s4 = np.matmul(delta_ir.transpose(),s3)
    #     a1 = np.matmul(delta_ir.transpose(),par['fftt_i'])
    #     der = np.divide((np.log(x)+1),theta)+cf_r+a1+s4
    #     return der.flatten()

    ineq_cons = {'type': 'ineq',
                 'fun': lambda x: x}  # ,
    # 'jac' : lambda x: np.array([])}
    eq_cons = {'type': 'eq',
               'fun': lambda x: (np.matmul(delta_odr, x) - (D_od)).flatten()}  # ,
    # 'jac' : lambda x: -delta_odr.flatten()}

    ###################################################################################################################
    # Check if there are nan values in the matrix
    # replace nan values with 0
    tt = (np.divide(D_od.transpose(), np.sum(delta_odr, axis=1))).transpose()
    tt[np.isinf(tt)] = tt.max() * 10

    D_r0 = np.matmul(delta_odr.transpose(), tt)

    # D_r0=np.matmul(delta_odr.transpose(),(np.divide(D_od.transpose(),np.sum(delta_odr,axis=1))).transpose())
    lb = np.zeros((np.shape(D_r0))).flatten()
    # Add a very small value to lb to avoid 0 in x
    lb = lb + 0.01
    ub = (max(D_od) * np.ones(np.shape(D_r0))).flatten()
    ub = ub * 5
    bounds = Bounds(lb, ub)
    # print(f"lb: {lb}")
    # print(f"ub: {ub}")
    # res=least_squares(fun, D_r0.flatten(),jac=fun_der,bounds=bounds)

    # D_r to be optimized -> demand on each route
    res = minimize(fun, D_r0.flatten(),
                   method='trust-constr',
                   # method='SLSQP',
                   # jac=fun_der,
                   constraints=[eq_cons, ineq_cons],
                   options={  # 'ftol': 1e5,
                       'maxiter': 2,
                       'verbose': 0,
                       'disp': True},
                   bounds=bounds
                   )
    # callback=callback_function
    # )
    # Describe variables
    # D_r is the demand on each route
    D_r = res.x
    D_r[D_r <= 0] = 0.001
    # check if values in D_r are negative if so print the amount
    if np.sum(D_r < 0) > 0:
        print(f"Negative values in D_r: {np.sum(D_r < 0)}")
    # Same for 0
    if np.sum(D_r == 0) > 0:
        print(f"0 in D_r: {np.sum(D_r == 0)}")

    # x_i is the demand on each link
    x_i = delta_ir * D_r
    # Get travel time on each route
    intTrec_i = IntCostFun(par)

    thetavec = np.ones_like(D_r)

    # fval is the objective function value
    fval = np.sum(IntLinksTimes(D_r)) + np.sum(np.divide(np.multiply(D_r, np.log(D_r)), thetavec)) + np.sum(
        (np.multiply(D_r, cf_r)))
    # fval = sum(IntLinksTimes(D_r)) + 1/theta*sum(np.multiply(D_r,np.log(D_r))) + sum(np.multiply(D_r,cf_r))
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    return [x_i, D_r, intTrec_i, fval]


def Commonality(betaCom, delta_ir, delta_odr, fftt_i, fftt_r):
    [nOD, nroutes] = delta_odr.shape
    # print(f"nOD {nOD} and nroutes {nroutes}")

    # Initialize commonality factor (per route)
    cf_r = np.zeros((nroutes, 1))

    # Iterate over all pairs of OD
    for od in range(0, nOD):
        # Find routes for OD pair
        routes = np.argwhere(np.ravel(delta_odr[od, :]) == 1)
        # print(routes)

        # Check if there is more than  route for the OD pair
        if len(routes) <= 1:
            # In this case there is no commonality factor
            continue
        else:
            # routes=find(delta_odr[od,:]==1)
            for r1 in routes:
                # Freeflow travel time for route
                t0_1 = fftt_r[r1]
                for r2 in routes:
                    # Freeflow travel time for route
                    t0_2 = fftt_r[r2]
                    aa = np.argwhere(np.ravel(delta_ir[:, r1]) == 1)
                    bb = np.argwhere(np.ravel(delta_ir[:, r2]) == 1)
                    # Get common links among routes routes compared
                    common = np.array(list(set(aa.flatten()).intersection(bb.flatten())))
                    # common=intersect(find(delta_ir(:,r1)==1),find(delta_ir(:,r2)==1));

                    # Check if common is empty
                    if len(common) == 0:
                        t0_1_2 = 0

                    else:
                        # Sum of freeflow travel time for common links
                        t0_1_2 = sum(fftt_i[common])

                    # Update commonality factor
                    cf_r[r1] = cf_r[r1] + t0_1_2 / (t0_1 ** .5 * t0_2 ** .5)
                    """
                    if (t0_1**.5*t0_2**.5) == 0:
                        print(f"t0_1 or t0_2 is 0 for OD pair {od} and routes {r1} and {r2}")
                    """
    cf_r[cf_r <= 0] = 0.0001
    cf_r = betaCom * np.log(cf_r)
    cf_r[np.isnan(cf_r)] = 0  # for routes with just one link -->cf_r=0;
    cf_r[np.isinf(cf_r)] = 0  # for routes with just one link -->cf_r=0;
    return cf_r


def travel_flow_optimization(OD_matrix, points, edges, voronoi, dev, scen):
    ## READING DATA AND FITTING THE MODEL


    # print for all variable ist daty structure and the data itselve
    print("nodes: ", type(nodes), nodes)
    print("links: ", type(links), links)
    print("nlinks: ", type(nlinks), nlinks)
    print("linklength_i: ", type(linklength_i), linklength_i)
    print("par: ", type(par), par)
    print("typeroad_i: ", type(typeroad_i), typeroad_i)


    # Same with own data
    nodes_lv95, nodes_wgs84, links, link_length_i, nlinks, par = convert_data_to_input(points=points, edges=edges)
    delta_ir, delta_odr, routelinks, D_od, nOD, nroutes = get_nw_data(OD_matrix=OD_matrix, points=points,
                                                                      voronoi_gdf=voronoi, edges=edges)

    # Remove all trip from origin to same destination
    # Delete all diagonals in OD matrix
    # Get the amount of values in the OD matrix as nOD
    OD_single = int(math.sqrt(nOD))
    # Remove every first and then every (OD_single + 1)th element
    idx = np.arange(0, nOD, OD_single + 1)
    # Delete every (OD_single + 1)th element
    D_od = np.delete(D_od, idx)
    # same for columns of delta_odr
    delta_odr = np.delete(delta_odr, idx, axis=0)

    # print(f"Shape delta_odr {delta_odr.shape} (OD pairs x #routes)")
    # print(f" Shape D_od {D_od.shape} (OD pairs x 1)")

    # --- Definition of parameters for SUE

    # Freeflow traval time per route
    fftt_r = (np.matmul(par['fftt_i'].transpose(), delta_ir)).transpose()
    #######################################################################################################################
    # No travel time when OD same point (intra cellular move)
    # fftt_r[fftt_r == 0] = 0.0001

    betaCom = 1
    # Get the common links among routes
    cf_r = Commonality(betaCom, delta_ir, delta_odr, par['fftt_i'], fftt_r)
    # print amount of nan in cf_r
    # print(f"Amount of nan in cf_r: {np.sum(np.isnan(cf_r))}")
    # print(f"Amount of inf in cf_r: {np.sum(np.isinf(cf_r))}")

    ## INPUT DATA ANALYSIS

    """
    for i in range(0, nlinks):
        var_labs[i] = print("L_{%3.0f}" % i)
        # var_labs{i}=sprintf('L_{#-.0f}',i);
    """

    done = 1  # calculate iterations  ## changed this to 1 for rail, because no increase in travel time due to congestion

    ## CALCULATION OF INCREASE OF TOTAL TRAVEL COST
    if not done:
        # t = time.process_time()

        # --- Reference value: network with no damage

        [Xi, D_r1, intTrec_i, ref] = SUE_C_Logit(D_od, par, delta_ir, delta_odr, cf_r)

        # Sum values of Xi for each row
        Xi_sum = np.sum(Xi, axis=1)

        # Store Xi_sum as csv but using number through pd df (demand link)
        pd.DataFrame(Xi_sum).to_csv(f"data/traffic_flow/developments/D_i/Xi_sum_{dev}_{scen}.csv", header=False,
                                    index=False)

        # Store D_r1 as csv but using number through pd df (demand route)
        pd.DataFrame(D_r1).to_csv(f"data/traffic_flow/developments/D_r/D_r1_{dev}_{scen}.csv", header=False,
                                  index=False)

        # Store intTrec_i as csv using pd df (travel time link)
        pd.DataFrame(intTrec_i).to_csv(f"data/traffic_flow/developments/tt_i/intTrec_i_{dev}_{scen}.csv", header=False,
                                       index=False)

        # Compute total travel time
        # Multiplying travel time on each link with demand on each link
        travel_time = np.matmul(intTrec_i.transpose(), Xi_sum).flatten()

        # --- Damaging link by link
        # Results=zeros(nlinks,1) #sol for each i
        # tic = time.time()
        # parfor i=1:nlinks
        #    #Reduction of capacity and ffftt of the affected link
        #    par2=par;
        #    par2.fftt_i(i)=par2.fftt_i(i)/factor;
        #    par2.Xmax_i(i)=par2.Xmax_i(i)*factor;
        #    # cost evaluation
        #    [~,~,fval]=SUE_C_Logit(nroutes,D_od,par2,delta_ir,delta_odr,cf_r,theta);
        #    fprintf('Evaluation i=#-3.0f  -> #10.2f (#10.2f ##)\n',i,fval, (fval-ref)/ref*100)
        #    Results(i)=fval;
        # end
        # toc = time.time()
        # print(toc-tic, ' sec elapsed')
        # print(f'CPU time (seconds): {comptime}')

        return travel_time  # .item(0)


def tt_optimization_all_developments():
    # Run travel time optimization for infrastructure developments and all scenarios
    # Scenario = OD matrix
    # Development = new network
    scenario = ["low", "medium", "high"]

    directory_path = r"data/traffic_flow/od/developments"
    # Get the developments
    developments = []
    for filename in os.listdir(directory_path):
        # Check if the filename matches the pattern 'devXX_source_id_raster.tif'
        match = re.match(r'od_matrix_dev(\d+)_medium\.csv', filename)
        if match:
            # Extract XX value and add to the list
            xx = match.group(1)
            developments.append(xx)

    # Convert values to integers if needed
    developments = [int(xx) for xx in developments]
    # print development and its length
    # print(f"Developments: {developments} and length: {len(developments)}")

    costs_travel_time = pd.DataFrame(columns=["development", "low", "medium", "high"])

    for dev in tqdm(developments):
        # Import generated links
        # links_developments = gpd.read_file(fr"data/Network/processed/new_links_realistic_costs.gpkg")
        links_developments = gpd.read_file(fr"data/costs/construction.gpkg") #todo is this correcT?
        # Check "dev" is in links_developments['ID_new']
        if dev not in links_developments['ID_new'].values:
            print(f"Development {dev} not in links_developments['ID_new'] - skipping")
            # continue with next for loop
            continue

        results = {}
        for scen in scenario:
            # if scen == "medium":
            #    results[scen] = 0
            #    continue

            # if scen == "high":
            #    results[scen] = 0
            #    continue

            print(f"Development: {dev} in scenario: {scen}")

            # Import generated points
            points_developments = gpd.read_file(fr"data/Network/processed/generated_nodes.gpkg")
            # print(points_developments.head(5).to_string())

            # Import points of current network
            points_current = gpd.read_file(fr"data/Network/processed/points_with_attribute.gpkg")

            # Filter generated point of development using ID_new
            point_temp = points_developments[points_developments["ID_new"] == dev]
            # Check if point_temp is empty, if so continur with next scen
            if point_temp.empty:
                print(f"Development {dev} not in points_developments['ID_new'] - skipping")
                continue
            # try geometry[0] otherwise geometry
            points = points_current.copy()
            # Add point of development to network
            new_point_row = {"intersection": 0,
                             "ID_point": 9999,
                             "geometry": point_temp.geometry.iloc[0],
                             "open_ends": None,
                             "within_corridor": True,
                             "on_corridor_border": False,
                             "generate_traffic": 0
                             }
            # points = points.append(new_point_row, ignore_index=True)
            temp = pd.Series(new_point_row)
            points = gpd.GeoDataFrame(
                pd.concat([points, pd.DataFrame(temp).T], ignore_index=True))
            # sort points in ascending point ID
            points.index = points.index.astype(int)
            points = points.sort_index()
            # points = points.sort_values(by=["ID_point"])
            points['id_dummy'] = points.index.values

            # Import current links
            links_current = gpd.read_file(fr"data/Network/processed/edges_with_attribute.gpkg")
            # print(links_current.head(5).to_string())

            # Filter edge of development
            edge_temp = links_developments[links_developments["ID_new"] == dev]
            edges = links_current.copy()
            # Get ID for new edge
            # edge_ID_max = edges["ID_edge"].max()
            edge_ID_max = edges["ID_edge"].astype(int).max()
            # Get index of point with ID_points = 999
            # index_point_9999 = points[points["id_dummy"] == 9999].index[0]
            index_point_start = points[points["id_dummy"] == edge_temp["ID_current"].values[0]].index[0]

            # Add edge of development to network
            new_edge_row = {"start": index_point_start,
                            "end": 9999,
                            "geometry": edge_temp["geometry"].iloc[0],
                            "ffs": 120,
                            "capacity": 2200,
                            "start_access": False,
                            "end_access": True,
                            "polygon_border": False,
                            "ID_edge": edge_ID_max + 1
                            }
            # edges = edges.append(new_edge_row, ignore_index=True)
            temp = pd.Series(new_edge_row)
            edges = gpd.GeoDataFrame(
                pd.concat([edges, pd.DataFrame(temp).T], ignore_index=True))

            # sort edges in ascending edge ID
            edges["ID_edge"] = edges["ID_edge"].astype(int)
            edges = edges.sort_values(by=["ID_edge"])
            # sort points in ascending point ID
            # points = points.sort_values(by=["ID_point"])
            points.index = points.index.astype(int)
            points = points.sort_index()

            # print last 3 rows of edges
            # print(edges.tail(3).to_string())
            # print(points.tail(3).to_string())

            # Generates traffic?
            # Print stats for values in point["generate_traffic"]
            # print(points["generate_traffic"].value_counts())
            # ID of developments?
            # Count amount of rows in generated_points
            # print(f"Amount of rows in points: {len(points_developments)}, links: {len(links_developments)} and developments in OD matrix: {len(developments)}")
            # Get amount number of overlapping points in links_development["new_ID"] and develpments
            # print(f"Amount of overlapping points in links_development['new_ID'] and developments: {len(set(links_developments['ID_new']).intersection(developments))}")

            # Get path like data/traffic_flow/od/od_matrix_i.csv
            OD_matrix = pd.read_csv(fr"data/traffic_flow/od/developments/od_matrix_dev{dev}_{scen}.csv", sep=",",
                                    index_col=0)

            # Import voronoi_df based on development
            voronoi_df = gpd.read_file(fr"data/Network/travel_time/developments/dev{dev}_Voronoi.gpkg")

            tt = travel_flow_optimization(OD_matrix=OD_matrix, points=points, edges=edges, voronoi=voronoi_df, dev=dev,
                                          scen=scen)
            # add tt to dict with key dev
            results[scen] = tt
        # print(results)

        # Append the result dict and add dev as values for thats row development column
        # costs_travel_time = costs_travel_time.append({"development":dev, "low":results["low"], "medium":results["medium"], "high":results["high"]}, ignore_index=True)
        temp = pd.Series(
            {"development": dev, "low": results["low"], "medium": results["medium"], "high": results["high"]})
        costs_travel_time = pd.concat([costs_travel_time, pd.DataFrame(temp).T], ignore_index=True)
        # print(costs_travel_time.head(5).to_string())
        costs_travel_time.to_csv(fr"data/traffic_flow/travel_time.csv", index=False)

    # Save results as csv
    costs_travel_time.to_csv(fr"data/traffic_flow/travel_time_2.csv", index=False)


def tt_optimization_status_quo():
    # Run travel time optimization for current infrastructure and all scenarios
    scenario = ["20", "low", "medium", "high"]
    dev = "status_quo"

    # Define dict to store results
    results_status_quo = {}
    for scen in scenario:
        # GEt path like data/traffic_flow/od/od_matrix_i.csv
        OD_matrix = pd.read_csv(r"data/traffic_flow/od/od_matrix_" + scen + ".csv", sep=",", index_col=0)

        # Import polygons of the corridor
        # voronoi_OD = gpd.read_file(r"data/traffic_flow/od/OD_voronoidf.gpkg")
        voronoi_df = gpd.read_file(r"data/Network/travel_time/Voronoi_statusquo.gpkg")

        # Import gpkg file with the network points
        points = gpd.read_file(r"data/Network/processed/points_with_attribute.gpkg")
        # points["ID_point"] = points["ID_point"].astype(int)
        # points = points.sort_values(by=["ID_point"])
        points.index = points.index.astype(int)
        points = points.sort_index()
        edges = gpd.read_file(r"data/Network/processed/edges_with_attribute.gpkg")
        edges["ID_edge"] = edges["ID_edge"].astype(int)
        edges = edges.sort_values(by=["ID_edge"])

        tt = travel_flow_optimization(OD_matrix=OD_matrix, points=points, edges=edges, voronoi=voronoi_df, dev=dev,
                                      scen=scen)
        # Append tt to dict with key scen
        results_status_quo[scen] = tt
    pd.DataFrame(results_status_quo).to_csv(r"data/traffic_flow/travel_time_status_quo.csv", index=False)


def monetize_tts(VTTS, duration):
    # Import total travel time for each scenario and each development
    tt_total = pd.read_csv(r"data/traffic_flow/travel_time.csv")
    # tt_total_low = pd.read_csv(r"data/traffic_flow/travel_time_low.csv")
    # tt_total["low"] = tt_total_low["low"]
    # Some values are stored in list format, convert them to float

    # convert columns object to float
    # for i in [1,len(tt_total["low"])]:
    #	tt_total["low"][i-1]=tt_total["low"][i-1].strip('][')
    #	tt_total["medium"][i-1]=tt_total["medium"][i-1].strip('][')
    #	tt_total["high"][i-1]=tt_total["high"][i-1].strip('][')
    # tt_total["low"] = tt_total["low"].astype(float)
    # tt_total["medium"] = tt_total["medium"].astype(float)
    # tt_total["high"] = tt_total["high"].astype(float)
    tt_total["low"] = tt_total["low"].apply(lambda x: float(x[1:-1]))
    tt_total["medium"] = tt_total["medium"].apply(lambda x: float(x[1:-1]))
    tt_total["high"] = tt_total["high"].apply(lambda x: float(x[1:-1]))

    # Import reference travel time for each scenario and current infrastructure
    tt_status_quo = pd.read_csv(fr"data/traffic_flow/travel_time_status_quo.csv")

    # monetization factor of travel time (peak hour * CHF/h * 365 d/a * 30 a)
    mon_factor = VTTS * 365 * duration
    # mon_factor = VTTS * 2.5 * 250 * duration
    # Compute difference in travel time for each scenario and each development
    tt_total["tt_low"] = (tt_status_quo["low"].iloc[0] - tt_total["low"]) * mon_factor
    tt_total["tt_medium"] = (tt_status_quo["medium"].iloc[0] - tt_total["medium"]) * mon_factor
    tt_total["tt_high"] = (tt_status_quo["high"].iloc[0] - tt_total["high"]) * mon_factor

    # Change presign of all psitive values to negative
    columns_to_negate = ['tt_low', 'tt_medium', 'tt_high']
    for col in columns_to_negate:
        tt_total[col] = tt_total[col].apply(lambda x: -abs(x))

    # drop useless columns
    tt_total = tt_total.drop(columns=["low", "medium", "high"])
    tt_total.to_csv(r"data/costs/traveltime_savings.csv")







