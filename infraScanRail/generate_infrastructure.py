import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from rasterio.features import geometry_mask
from scipy.stats.qmc import LatinHypercube
import re
import glob
import tkinter as tk
from tkinter.simpledialog import Dialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from shapely.geometry import shape, LineString, GeometryCollection
from shapely.ops import nearest_points, split
import fiona
from scipy.optimize import minimize
from tqdm import tqdm
import pulp
import requests
import zipfile
import rasterio
from shapely.geometry import Point, LineString, MultiLineString, GeometryCollection
from shapely.ops import split
import random
import os
import geopandas as gpd
import re
import gc

from data_import import *
from scoring import *


def generated_access_points(extent,number):
    e_min, n_min, e_max, n_max = extent.bounds
    e = int(e_max - e_min)
    n = int(n_max+100 - n_min+100)

    N = number

    engine = LatinHypercube(d=2, seed=42)  # seed=42
    sample = engine.random(n=N)

    n_sample = np.asarray(list(sample[:, 0]))
    e_sample = np.asarray(list(sample[:, 1]))

    n_gen = np.add(np.multiply(n_sample, n), int(n_min))
    e_gen = np.add(np.multiply(e_sample, e), int(e_min))

    idlist = list(range(0,N))
    gen_df = pd.DataFrame({"ID": idlist, "XKOORD": e_gen,"YKOORD":n_gen})
    gen_gdf = gpd.GeoDataFrame(gen_df,geometry=gpd.points_from_xy(gen_df.XKOORD,gen_df.YKOORD),crs="epsg:2056")

    return gen_gdf


def filter_access_points(gdf):
    newgdf = gdf.copy()
    print("All")
    print(len(newgdf))
    # idx = list(np.zeros(N))
    """
    print("Lake")
    idx = get_idx_todrop(newgdf, r"data\landuse_landcover\landcover\lake\WB_GEWAESSERRAUM_F.shp")
    newgdf.loc[:, "index"] = idx
    keepidx = newgdf['index'] == 0 # 1 is the value of the columns that should be dropped
    newgdf = newgdf.loc[keepidx,:]

    idx = get_idx_todrop(newgdf,r"data\landuse_landcover\landcover\lake\WB_STEHGEWAESSER_F.shp")
    newgdf.loc[:, "index"] = idx
    keepidx = newgdf['index'] == 0 # 1 is the value of the columns that should be dropped
    newgdf = newgdf.loc[keepidx,:]
    print(len(newgdf))
    """


    """
    # Perform a spatial join
    FFF_gdf = gpd.read_file(r"data\landuse_landcover\Schutzzonen\Fruchtfolgeflachen_-OGD\FFF_F.shp")
    print(FFF_gdf.head().to_string())
    joined = gpd.sjoin(newgdf, FFF_gdf, how="left", predicate="within")
    print(joined.head().to_string())
    # Filter points that are within polygons
    newgdf = newgdf[~joined["index_right"].isna()]
    print(newgdf.head().to_string())

    #newgdf.loc[:, "index"] = idx
    #print(newgdf.head().to_string())
    #newgdf.drop(newgdf.loc[newgdf['index'] == 1],inplace=True)
    print(len(newgdf))
    """

    print("Schutzanordnung Natur und Landschaft")
    idx = get_idx_todrop(newgdf,r"data\landuse_landcover\Schutzzonen\Schutzanordnungen_Natur_und_Landschaft_-SAO-_-OGD\FNS_SCHUTZZONE_F.shp")
    newgdf.loc[:, "index"] = idx
    keepidx = newgdf['index'] == 0  # 1 is the value of the columns that should be dropped
    newgdf = newgdf.loc[keepidx, :]
    print(len(newgdf))
    """
    print("Naturschutzobjekte")
    idx = get_idx_todrop(newgdf, r"data\landuse_landcover\Schutzzonen\Inventar_der_Natur-_und_Landsch...uberkommunaler_Bedeutung_-OGD\INV80_NATURSCHUTZOBJEKTE_F.shp")
    newgdf.loc[:, "index"] = idx
    keepidx = newgdf['index'] == 0  # 1 is the value of the columns that should be dropped
    newgdf = newgdf.loc[keepidx, :]

    idx = get_idx_todrop(newgdf, r"data\landuse_landcover\Schutzzonen\Inventar_der_Natur-_und_Landsch...uberkommunaler_Bedeutung_-OGD\INVERG_NATURSCHUTZOBJEKTE_F.shp")
    newgdf.loc[:, "index"] = idx
    keepidx = newgdf['index'] == 0  # 1 is the value of the columns that should be dropped
    newgdf = newgdf.loc[keepidx, :]
    print(len(newgdf))
    """

    print("Forest")
    idx = get_idx_todrop(newgdf,r"data\landuse_landcover\Schutzzonen\Waldareal_-OGD\WALD_WALDAREAL_F.shp")
    newgdf.loc[:, "index"] = idx
    keepidx = newgdf['index'] == 0 # 1 is the value of the columns that should be dropped
    newgdf = newgdf.loc[keepidx,:]
    print(len(newgdf))

    ###########################################################################3
    """
    print("Wetlands")
    idx = get_idx_todrop(newgdf,r"data\landuse_landcover\landcover\lake\WB_STEHGEWAESSER_F.shp")
    newgdf.loc[:, "index"] = idx
    keepidx = newgdf['index'] == 0 # 1 is the value of the columns that should be dropped
    newgdf = newgdf.loc[keepidx,:]
    print(len(newgdf))
    """

    print("Network buffer")
    network_gdf = gpd.read_file(r"data\Network\processed\edges.gpkg")
    network_gdf['geometry'] = network_gdf['geometry'].buffer(1000)
    network_gdf.to_file(r"data\temp\buffered_network.gpkg")

    idx = get_idx_todrop(newgdf, r"data\temp\buffered_network.gpkg")
    newgdf.loc[:, "index"] = idx
    keepidx = newgdf['index'] == 0 # 1 is the value of the columns that should be dropped
    newgdf = newgdf.loc[keepidx,:]
    print(len(newgdf))
    """
    print("Residential area")
    idx = get_idx_todrop(newgdf,r"data\landuse_landcover\landcover\Quartieranalyse_-OGD\QUARTIERE_F.shp")
    newgdf.loc[:, "index"] = idx
    keepidx = newgdf['index'] == 0 # 1 is the value of the columns that should be dropped
    newgdf = newgdf.loc[keepidx,:]
    print(len(newgdf))
    """

    print("Protected zones")
    # List to store indices to drop
    indices_to_drop = []

    with rasterio.open("data\landuse_landcover\processed\zone_no_infra\protected_area_corridor.tif") as src:
        # Read the raster data once outside the loop
        raster_data = src.read(1)

        # Loop through each point in the GeoDataFrame
        for index, row in newgdf.iterrows():
            # Convert the point geometry to raster space
            row_x, row_y = row['geometry'].x, row['geometry'].y
            row_col, row_row = src.index(row_x, row_y)

            if 0 <= row_col < raster_data.shape[0] and 0 <= row_row < raster_data.shape[1]:
                # Read the value of the corresponding raster cell
                value = raster_data[row_col, row_row]

                # If the value is not NaN, mark the index for dropping
                if not np.isnan(value):
                    indices_to_drop.append(index)

            else:
                print(f"Point outside the polygon {row_x, row_y}")
                indices_to_drop.append(index)

        # Drop the points
    newgdf = newgdf.drop(indices_to_drop)
    print(len(newgdf))

    print("FFF")
    idx = get_idx_todrop(newgdf, r"data\landuse_landcover\Schutzzonen\Fruchtfolgeflachen_-OGD\FFF_F.shp")
    newgdf.loc[:, "index"] = idx
    keepidx = newgdf['index'] == 0  # 1 is the value of the columns that should be dropped
    newgdf = newgdf.loc[keepidx, :]
    print(len(newgdf))

    newgdf = newgdf.rename(columns={"ID": "ID_new"})
    newgdf = newgdf.to_crs("epsg:2056")

    return newgdf

def generate_highway_access_points(n,filter=False):
    num_rand=n
    random_gdf = generated_access_points(extent=innerboundary, number=num_rand)
    if filter ==False:
        random_gdf.to_file(r"data\Network\processed\generated_nodes.gpkg")
    else:
        filtered_gdf = filter_access_points(random_gdf)
        filtered_gdf.to_file(r"data\Network\processed\generated_nodes.gpkg")
    generated_points = gpd.read_file(r"data/Network/processed/generated_nodes.gpkg")
    # Import current points as dataframe and filter only access points (no intersection points)
    current_points = gpd.read_file(r"data/Network/processed/points_corridor_attribute.gpkg")
    current_access_points = current_points.loc[current_points["intersection"] == 0]



    # Connect the generated points to the existing access points
    # New lines are stored in "data/Network/processed/new_links.gpkg"
    filtered_rand_temp = connect_points_to_network(generated_points, current_access_points)
    filtered_rand_temp.to_file(r"data\Network\temp\filtered_nodes.gpkg")
    link_new_access_points_to_highway(generated_points,filtered_rand_temp)
    return

def link_new_access_points_to_highway(generated_points,filtered_rand_temp):
    nearest_gdf = create_nearest_gdf(filtered_rand_temp)
    create_lines(generated_points, nearest_gdf)
    return

def generate_rail_edges(n, radius):
    """
    Generate rail edges by connecting generated points to nearest infrastructure points.
    
    Parameters:
        n (int): Maximum number of points to include in a buffer zone.
        radius (int): Buffer radius in kilometers.

    Returns:
        None
    """
    radius = radius * 1000  # Convert radius to meters
    
    # Step 1: Load data and filter
    current_points = gpd.read_file(r"data/Network/processed/points.gpkg")
    current_points = current_points[~current_points['ID_point'].isin([112, 113, 720, 2200])]
    raw_edges = gpd.read_file(r"data/temp/network_railway-services.gpkg")
    
    # Identify endpoint nodes
    endpoints = set(
        raw_edges.loc[raw_edges['FromEnd'] == True, 'FromNode']
    ).union(
        raw_edges.loc[raw_edges['ToEnd'] == True, 'ToNode']
    )
    endnodes_gdf = current_points[current_points['ID_point'].isin(endpoints)]

    # Step 2: Buffer and find nearest points
    set_gdf = endnodes_gdf.head(0)
    set_gdf['current'] = None
    
    for idx, endnode in endnodes_gdf.iterrows():
        buffer = endnode.geometry.buffer(radius)
        temp_gdf = current_points[current_points.within(buffer)]
        temp_gdf['current'] = endnode['ID_point']
        temp_gdf['geometry_current'] = endnode['geometry']
        
        if len(temp_gdf) > n:
            temp_gdf['distance'] = temp_gdf.geometry.apply(lambda x: endnode.geometry.distance(x))
            temp_gdf = temp_gdf.nsmallest(n, 'distance').drop(columns=['distance'])
        
        set_gdf = pd.concat([set_gdf, temp_gdf], ignore_index=True)

    # Prepare generated points and nearest points GeoDataFrames
    generated_points = set_gdf[['NAME', 'ID_point', 'current', 'XKOORD', 'YKOORD', 'HST', 'geometry']]
    generated_points = generated_points.rename(columns={'current': 'To_ID-point', 'HST': 'index'})
    nearest_gdf = gpd.GeoDataFrame(set_gdf[['ID_point', 'current', 'geometry_current']], geometry='geometry_current')
    nearest_gdf = nearest_gdf.rename(columns={'ID_point': 'TO_ID_new', 'current': 'ID_point'})

    # Set CRS to EPSG:2056
    nearest_gdf.set_crs("EPSG:2056", inplace=True)
    generated_points.set_crs("EPSG:2056", inplace=True)

    # Assign services to generated points
    generated_points = assign_services_to_generated_points(raw_edges, generated_points)
    
    # Save intermediate files
    generated_points.to_file(r"data/Network/processed/generated_nodeset.gpkg", driver="GPKG")
    nearest_gdf.to_file(r"data/Network/processed/endnodes.gpkg", driver="GPKG")

    # Create lines
    create_lines(generated_points, nearest_gdf)

def assign_services_to_generated_points(raw_edges, generated_points):
    """
    Assign services to generated points based on raw edges data, specifically for endpoints (ToEnd=True).

    Parameters:
        raw_edges (GeoDataFrame): Raw edges GeoDataFrame.
        generated_points (GeoDataFrame): Generated points GeoDataFrame.

    Returns:
        GeoDataFrame: Updated generated points with assigned services.
    """
    # Filter raw_edges to include only those with ToEnd=True
    endpoint_services = raw_edges[raw_edges['ToEnd'] == True]

    # Create a mapping of ToNode to its terminating services
    service_mapping = endpoint_services.groupby('ToNode')['Service'].apply(list).to_dict()

    # Map services to generated points based on To_ID-point
    generated_points['Service'] = generated_points['To_ID-point'].map(
        lambda to_id: ','.join(service_mapping.get(to_id, []))  # Join multiple services as a string
    )

    return generated_points

def filter_unnecessary_links():
    """
    Filter out unnecessary links in the new_links GeoDataFrame.
    Saves the filtered links as a GeoPackage file.
    """
    try:
        # Load raw edges and new links
        raw_edges = gpd.read_file(r"data\temp\network_railway-services.gpkg")
        time.sleep(1)  # Ensure file access is sequential
        line_gdf = gpd.read_file(r"data\Network\processed\new_links.gpkg")
    except Exception as e:
        print(f"Error loading files: {e}")
        return

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
        
        if from_id in sline_routes.get(sline, set()) and to_id in sline_routes.get(sline, set()):
            continue  # Skip redundant links
        else:
            filtered_links.append(row)

    # Step 3: Create GeoDataFrame for filtered links
    filtered_gdf = gpd.GeoDataFrame(filtered_links, geometry='geometry', crs=line_gdf.crs)

    # Save filtered links
    try:
        filtered_gdf.to_file(r"data\Network\processed\filtered_new_links.gpkg", driver="GPKG")
        print("Filtered new links saved successfully!")
    except Exception as e:
        print(f"Error saving filtered new links: {e}")
    
    # Cleanup
    del filtered_gdf, line_gdf, raw_edges
    gc.collect()


def delete_connections_back(file_path_updated, file_path_raw_edges, output_path):
    """
    Deletes rows from `updated_new_links` if connections in `raw_edges` lead back to existing nodes.

    Parameters:
        file_path_updated (str): Path to the GeoPackage file `updated_new_links`.
        file_path_raw_edges (str): Path to the GeoPackage file `raw_edges`.
        output_path (str): Path to save the updated file.

    Returns:
        None: Saves the updated file as a GeoPackage.
    """
    # Read the GeoPackage files
    updated_new_links = gpd.read_file(file_path_updated)
    raw_edges = gpd.read_file(file_path_raw_edges)

    # List to keep track of rows to be removed
    rows_to_remove = []

    # Iterate through each row in `updated_new_links`
    for idx, row in updated_new_links.iterrows():
        from_id_new = row["from_ID_new"]
        sline = row["Sline"]

        # Filter `raw_edges` where Service matches and FromNode/ToNode corresponds to from_ID_NEW
        matching_edges = raw_edges[
            (raw_edges["Service"] == sline) &
            ((raw_edges["FromNode"] == from_id_new) | (raw_edges["ToNode"] == from_id_new))
        ]

        # If matching edges are found, mark this row for removal
        if not matching_edges.empty:
            rows_to_remove.append(idx)

    # Remove the marked rows
    updated_new_links = updated_new_links.drop(index=rows_to_remove)

    # Save the updated GeoDataFrame
    updated_new_links.to_file(output_path, driver="GPKG")
    print(f"Updated file saved to: {output_path}")
    return updated_new_links



def calculate_new_service_time():
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

    # Save the split lines for future use
    split_lines_gdf.to_file(r"data\Network\processed\split_s_bahn_lines.gpkg", driver="GPKG")

    # Load split lines and corridor line data
    corridor_path = r"data\Network\processed\filtered_new_links_in_corridor.gpkg"
    new_links = gpd.read_file(corridor_path)

    # Create the graph from split line segments with weights
    G = create_graph_from_lines(split_lines_gdf)

    # Define the average speed in km/h
    average_speed_kmh = 60

    # Initialize lists to store lengths and times for each shortest path
    path_lengths = []
    path_times = []

    # Calculate the shortest path for each dev_id in new_links
    for _, row in new_links.iterrows():
        line_geometry = row.geometry
        start_point = line_geometry.coords[0]
        end_point = line_geometry.coords[-1]
        
        # Calculate the shortest path in the graph
        shortest_path_coords = calculate_shortest_path(G, start_point, end_point)
        
        # Check if the path is valid (contains more than one point)
        if len(shortest_path_coords) > 1:
            # Convert the shortest path coordinates to a LineString
            shortest_path_line = LineString(shortest_path_coords)
            
            # Calculate the length of the shortest path in kilometers
            path_length_km = shortest_path_line.length / 1000  # convert from meters to kilometers

            # Calculate the time needed at 60 km/h in minutes
            path_time_minutes = (path_length_km / average_speed_kmh) * 60  # convert hours to minutes
            
            # Append the length and time to the lists
            path_lengths.append(path_length_km * 1000)  # convert back to meters for consistency
            path_times.append(path_time_minutes)
        else:
            # No valid path found, append None or 0 as desired
            path_lengths.append(None)  # or 0 if you prefer
            path_times.append(None)

    # Add the path length and time as new columns in new_links
    new_links['shortest_path_length'] = path_lengths
    new_links['time'] = path_times

    # Save the updated new_links with the shortest path information
    new_links.to_file("data/Network/processed/updated_new_links.gpkg", driver="GPKG")

    return

def split_multilinestrings_at_stops(s_bahn_lines, stops, buffer_distance=30):
    """
    Split MultiLineStrings in `s_bahn_lines` at each Point in `stops` and calculate lengths.
    
    Parameters:
    - s_bahn_lines (GeoDataFrame): GeoDataFrame with MultiLineString geometries.
    - stops (GeoDataFrame): GeoDataFrame with Point geometries.
    - buffer_distance (float): Buffer distance (in meters) around points to ensure precision in intersections.
    
    Returns:
    - GeoDataFrame containing the split LineStrings with a length column.
    """
    
    # Buffer each stop point by the specified buffer distance
    stops_buffered = stops.copy()
    stops_buffered['geometry'] = stops_buffered.geometry.buffer(buffer_distance)
    
    # Initialize a list to collect the split line segments and their lengths
    split_lines = []
    lengths = []
    
    # Iterate through each MultiLineString geometry in s_bahn_lines
    for mls in s_bahn_lines.geometry:
        if isinstance(mls, (MultiLineString, LineString)):
            segments = [mls]
            
            # Iterate over each buffered stop point
            for stop_buffer in stops_buffered.geometry:
                # Split each segment at the intersection points with the buffered stop
                new_segments = []
                for segment in segments:
                    if segment.intersects(stop_buffer):
                        split_result = split(segment, stop_buffer)
                        for part in split_result.geoms:
                            if isinstance(part, LineString):
                                new_segments.append(part)
                    else:
                        new_segments.append(segment)
                segments = new_segments
            
            # Add the resulting segments and their lengths to the list
            for seg in segments:
                split_lines.append(seg)
                lengths.append(seg.length)
    
    # Create a GeoDataFrame from the split lines and add the length column
    split_lines_gdf = gpd.GeoDataFrame(geometry=split_lines, crs=s_bahn_lines.crs)
    split_lines_gdf['length'] = lengths  # Add length column to GeoDataFrame
    
    return split_lines_gdf


def create_graph_from_lines(split_lines_gdf, max_distance=30):
    """
    Create a NetworkX graph from a GeoDataFrame with LineString geometries, connecting points 
    closer than `max_distance` with a straight line.
    
    Parameters:
    - split_lines_gdf (GeoDataFrame): GeoDataFrame containing LineString geometries and a 'length' column.
    - max_distance (float): Maximum distance to connect points with a straight line (in meters).
    
    Returns:
    - NetworkX Graph with edges weighted by length.
    """
    # Initialize the graph
    G = nx.Graph()
    
    # Iterate over each LineString in the GeoDataFrame
    for _, row in split_lines_gdf.iterrows():
        line = row.geometry
        length = row['length']
        
        # Get the start and end points of the LineString as tuples
        start_point = (line.coords[0][0], line.coords[0][1])
        end_point = (line.coords[-1][0], line.coords[-1][1])
        
        # Add the edge or update it if it exists with a shorter length
        if G.has_edge(start_point, end_point):
            existing_length = G[start_point][end_point]['weight']
            if length < existing_length:
                G[start_point][end_point]['weight'] = length
        else:
            G.add_edge(start_point, end_point, weight=length)
    
    # Get a list of all nodes for distance checking
    nodes = list(G.nodes)
    
    # Connect nodes within the max_distance if they are not already connected
    for i, node1 in enumerate(nodes):
        for node2 in nodes[i+1:]:
            distance = Point(node1).distance(Point(node2))
            if distance < max_distance and not G.has_edge(node1, node2):
                G.add_edge(node1, node2, weight=distance)  # Add edge with straight-line distance
    
    return G

def calculate_shortest_path(graph, start_point, end_point):
    """
    Calculate the shortest path between two points in a weighted graph.
    
    Parameters:
    - graph (NetworkX Graph): Graph with weighted edges.
    - start_point (tuple): Starting point coordinates (x, y).
    - end_point (tuple): Ending point coordinates (x, y).
    
    Returns:
    - List of coordinates representing the shortest path.
    """
    # Find nearest nodes in the graph to the start and end points
    nearest_start = min(graph.nodes, key=lambda node: Point(node).distance(Point(start_point)))
    nearest_end = min(graph.nodes, key=lambda node: Point(node).distance(Point(end_point)))
    
    # Calculate the shortest path based on the weight (length)
    shortest_path = nx.shortest_path(graph, source=nearest_start, target=nearest_end, weight='weight')
    
    return shortest_path


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
        Direction="B",  # Default direction
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

    # Generate rows for Direction A while preserving dev_id
    direction_A = new_links_updated.copy()
    direction_A["Direction"] = "A"
    direction_A["FromNode"], direction_A["ToNode"] = direction_A["ToNode"], direction_A["FromNode"]
    direction_A["FromStation"], direction_A["ToStation"] = direction_A["ToStation"], direction_A["FromStation"]

    # Combine A and B directions, preserving the same dev_id
    combined_new_links = pd.concat([new_links_updated, direction_A], ignore_index=True)

    # Ensure GeoDataFrame compatibility
    combined_new_links_gdf = gpd.GeoDataFrame(combined_new_links, geometry=new_links_updated.geometry)

    # Standardize station names in FromStation and ToStation
    standardize_station_names = {
        "Wetzikon": "Wetzikon ZH",
        # Add more mappings here if needed
    }
    new_links_updated["FromStation"] = new_links_updated["FromStation"].replace(standardize_station_names)
    new_links_updated["ToStation"] = new_links_updated["ToStation"].replace(standardize_station_names)

    combined_new_links["FromStation"] = combined_new_links["FromStation"].replace(standardize_station_names)
    combined_new_links["ToStation"] = combined_new_links["ToStation"].replace(standardize_station_names)


    # Combine with original network
    combined_network = pd.concat([network_railway_service, combined_new_links_gdf], ignore_index=True)
    

    return combined_network

def create_network_foreach_dev():
    """
    Creates individual GeoPackages for each unique development (identified by dev_id),
    combining the entire old network with the corresponding development in both directions.
    """

    # Load the GPK file
    input_gpkg = "data/Network/processed/combined_network_with_new_links.gpkg"
    output_directory = "data/Network/processed/developments/"  # Directory to save output files
    os.makedirs(output_directory, exist_ok=True)  # Ensure the output directory exists

    # Read the GeoPackage
    gdf = gpd.read_file(input_gpkg)

    # Separate rows based on `new_dev`
    base_gdf = gdf[gdf['new_dev'] == "No"]  # Old network: Rows where `new_dev` is "No"
    new_dev_rows = gdf[gdf['new_dev'] == "Yes"]  # Rows where `new_dev` is "Yes"

    # Group new development rows by dev_id
    grouped_new_dev_rows = new_dev_rows.groupby("dev_id")

    # Iterate through unique dev_id groups
    for dev_id, group in grouped_new_dev_rows:
        if pd.isna(dev_id):
            print("Skipping group with NULL dev_id")
            continue  # Skip groups where dev_id is NULL (unlikely for "Yes")

        # Combine both directions for the current dev_id
        new_dev_gdf = gpd.GeoDataFrame(group, crs=gdf.crs)

        # Combine the entire old network with the current development rows
        combined_gdf_new = gpd.GeoDataFrame(pd.concat([base_gdf, new_dev_gdf], ignore_index=True), crs=gdf.crs)

        # Save to the specified directory, naming the file after dev_id
        output_gpkg = os.path.join(output_directory, f"{dev_id}.gpkg")
        combined_gdf_new.to_file(output_gpkg, driver="GPKG")
        print(f"Saved: {output_gpkg}")

    print("Processing complete.")


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
    combined_gdf.loc[combined_gdf["ToNode"] == 2497, "ToStation"] = "Wetzikon ZH"

    # Update FromStation based on FromNode
    combined_gdf.loc[combined_gdf["FromNode"] == 1018, "FromStation"] = "Hinwil"
    combined_gdf.loc[combined_gdf["FromNode"] == 2298, "FromStation"] = "Uster"
    combined_gdf.loc[combined_gdf["FromNode"] == 2497, "FromStation"] = "Wetzikon ZH"

    # Save the output
    combined_gdf.to_file(output_path, driver="GPKG")
    print("Combined network with new links saved successfully!")

    return combined_gdf



def get_idx_todrop(pt, filename):
    #with fiona.open(r"data\landuse_landcover\landcover\lake\WB_STEHGEWAESSER_F.shp") as input:
    with fiona.open(filename, crs="epsg:2056") as input:
        #pt = newgdf.copy() #for testing
        idx = np.ones(len(pt))
        for feat in input:
            geom = shape(feat['geometry'])
            temptempidx = pt.within(geom)
            temptempidx = np.multiply(np.array(temptempidx), 1)
            tempidx = [i ^ 1 for i in temptempidx]
            #tempidx = np.multiply(np.array(tempidx),1)
            idx = np.multiply(idx, tempidx)
        intidx = [int(i) for i in idx]
        newidx = [i ^ 1 for i in intidx]
        #print(newidx)
    return newidx


def nearest(row, geom_union, df1, df2, geom1_col='geometry', geom2_col='geometry', src_column=None):
    """Find the nearest point and return the corresponding value from specified column."""

    # Find the geometry that is closest
    #nearest = df2[geom2_col] == nearest_points(row[geom1_col], geom_union)[1]
    nearest = df2[geom2_col] == nearest_points(geom_union,row[geom1_col])[1]

    # Get the corresponding value from df2 (matching is based on the geometry)
    value = df2[nearest][src_column].values()[0]

    return value


def near(point, network_gdf,pts):
    # find the nearest point and return the corresponding Place value
    nearest = network_gdf.geometry == nearest_points(point, pts)[1]
    return network_gdf[nearest].geometry.values()[0]


def connect_points_to_network(new_point_gdf, network_gdf):
    #unary_union = network_gdf.unary_union
    #new_gdf=point_gdf.copy()
    ###
    #network_gdf = network_gdf.rename(columns={'geometry': 'geometry_current'})
    #network_gdf = network_gdf.set_geometry("geometry_current")
    network_gdf["geometry_current"] = network_gdf["geometry"]
    network_gdf = network_gdf[['intersection', 'ID_point', 'name', 'end', 'cor_1',
       'geometry', 'geometry_current']]
    new_gdf = gpd.sjoin_nearest(new_point_gdf,network_gdf,distance_col="distances")[["ID_new","XKOORD","YKOORD","geometry","distances","geometry_current", "ID_point"]] # "geometry",
    ###
    #new_gdf['straight_line'] = new_gdf.apply(lambda row: LineString([row['geometry'], row['nearest_node']]), axis=1) #Create a linestring column
    return new_gdf


def create_nearest_gdf(filtered_rand_gdf):
    nearest_gdf = filtered_rand_gdf[["ID_new", "ID_point", "geometry_current"]].set_geometry("geometry_current")
    #nearest_gdf = nearest_gdf.rename({"ID":"PointID", "index_right":"NearestAccID"})
    #nearest_df = filtered_rand_gdf.assign(PointID=filtered_rand_gdf["ID"],NearestAccID=filtered_rand_gdf["index_right"],x=filtered_rand_gdf["x"],y=filtered_rand_gdf["y"])
    #nearest_gdf = gpd.GeoDataFrame(nearest_df,geometry=gpd.points_from_xy(nearest_df.x,nearest_df.y),crs="epsg:2056")
    return nearest_gdf



def create_lines(gen_pts_gdf, nearest_infra_pt_gdf):
    """
    Create lines connecting generated points to their single nearest infrastructure point.

    Parameters:
        gen_pts_gdf (GeoDataFrame): Generated points GeoDataFrame.
        nearest_infra_pt_gdf (GeoDataFrame): Nearest infrastructure points GeoDataFrame.

    Returns:
        None: Saves the generated links as a GeoPackage file.
    """
    # Ensure CRS match
    if gen_pts_gdf.crs != nearest_infra_pt_gdf.crs:
        nearest_infra_pt_gdf = nearest_infra_pt_gdf.to_crs(gen_pts_gdf.crs)

    # Validate geometry columns
    gen_pts_gdf = gen_pts_gdf[gen_pts_gdf['geometry'].notnull()]
    nearest_infra_pt_gdf = nearest_infra_pt_gdf[nearest_infra_pt_gdf['geometry_current'].notnull()]

    # Create connections by finding the nearest match for each generated point
    connections = []
    for _, gen_row in gen_pts_gdf.iterrows():
        # Filter rows in `nearest_infra_pt_gdf` that correspond to the current generated point
        potential_matches = nearest_infra_pt_gdf[nearest_infra_pt_gdf['ID_point'] == gen_row['To_ID-point']]

        # If multiple matches exist, find the closest one
        if not potential_matches.empty:
            closest_row = potential_matches.loc[
                potential_matches['geometry_current'].distance(gen_row['geometry']).idxmin()
            ]

            # Create the connection
            connections.append({
                'from_ID_new': gen_row['ID_point'],
                'to_ID': closest_row['TO_ID_new'],
                'Sline' : gen_row['Service'],
                'geometry': LineString([gen_row['geometry'], closest_row['geometry_current']])
            })

    # Create GeoDataFrame for lines
    line_gdf = gpd.GeoDataFrame(connections, geometry='geometry', crs=gen_pts_gdf.crs)

    # Save the resulting GeoDataFrame
    line_gdf.to_file(r"data/Network/processed/new_links.gpkg", driver="GPKG")

    print("New links saved successfully!")



def plot_lines_to_network(points_gdf,lines_gdf):
    points_gdf.plot(marker='*', color='green', markersize=5)
    base = lines_gdf.plot(edgecolor='black')
    points_gdf.plot(ax=base, marker='o', color='red', markersize=5)
    plt.savefig(r"plot\predict\230822_network-generation.png", dpi=300)
    return None


def line_scoring(lines_gdf,raster_location):
    # Load your raster file using rasterio
    raster_path = raster_location
    with rasterio.open(raster_path) as src:
        raster = src.read(1)  # Assuming it's a single-band raster

    # Create an empty list to store the sums
    sums = []

    # Iterate over each line geometry in the GeoDataFrame
    for idx, line in lines_gdf.iterrows():
        mask = geometry_mask([line['geometry']], out_shape=raster.shape, transform=src.transform, invert=False)
        line_sum = raster[mask].sum()
        sums.append(line_sum)

    # Add the sums as a new column to the GeoDataFrame
    lines_gdf['raster_sum'] = sums

    return lines_gdf


def routing_raster(raster_path):
    # Process LineStrings
    generated_links = gpd.read_file(r"data\Network\processed\new_links.gpkg")
    print(generated_links["ID_new"].unique())

    #print(generated_links.head(10))
    new_lines = []
    generated_points_unaccessible = []

    with rasterio.open(raster_path) as dataset:
        raster_data = dataset.read(1)  # Assumes forbidden cells are marked with 1 or another distinct value

        transform = dataset.transform

        for i, line in enumerate(generated_links.geometry):
            # Get the start and end points from the linestring
            start_point = line.coords[0]
            end_point = line.coords[-1]

            # Convert real-world coordinates to raster indices
            start_index = rasterio.transform.rowcol(transform, xs=start_point[0], ys=start_point[1])
            end_index = rasterio.transform.rowcol(transform, xs=end_point[0], ys=end_point[1])

            # Convert raster to graph
            graph = raster_to_graph(raster_data)

            # Calculate the shortest path avoiding forbidden cells
            try:
                path, generated_points_unaccessible = find_path(graph, start_index, end_index, generated_points_unaccessible, end_point)
            except Exception as e:
                path=None
                print(e)
                #print(generated_links.iloc[i])


            if path:
                # If you need to convert back the path to real-world coordinates, you would use the raster's transform
                # Here's a stub for that process
                real_world_path = [rasterio.transform.xy(transform, cols=point[1], rows=point[0], offset='center') for point in path]
                #print("Start ", real_world_path[0], " ersetzt durch ", line.coords[0])
                #print("Ende ", real_world_path[-1], " ersetzt durch ", line.coords[-1])
                real_world_path[0] = line.coords[0]
                real_world_path[-1] = line.coords[-1]
                new_lines.append(real_world_path)
            else:
                new_lines.append(None)

    # Update GeoDataFrame
    generated_links['new_geometry'] = new_lines

    # Save the updated GeoDataFrame
    #generated_links.to_csv(r'data\Network\processed\generated_links_updated.csv', index=False)

    df_links = generated_links.dropna(subset=['new_geometry'])

    df_links = gpd.GeoDataFrame(df_links)
    listattempt = df_links['new_geometry'].apply(lambda x: len(x)) > 1
    df_links = df_links[listattempt]
    # Assuming 'df' is your DataFrame and it has a column 'coords' with coordinate arrays
    # Step 1: Convert to LineStrings
    #df_links['geometry'] = df_links['new_geometry'].apply(lambda x: LineString(x))
    #df_links2 = df_links
    tempgeom = df_links['new_geometry'].head(0)

    for index, row in df_links.iterrows():
        try:
            tempgeom = df_links['new_geometry'].apply(lambda x: LineString(x))
        except Exception as e:
            df_links.drop(index, inplace=True)
            #print(e)
    df_links['geometry'] = tempgeom
    df_links = df_links.drop(columns="new_geometry")
    df_links = df_links.set_geometry("geometry")
    df_links = df_links.set_crs(epsg=2056)

    # df_links.to_file(r"data\Network\processed\01_linestring_links.gpkg")

    # Step 2: Simplify LineStrings (Retaining corners)
    #tolerance = 0.01  # Adjust tolerance to your needs
    #df_links['geometry'] = df_links['geometry'].apply(lambda x: x.simplify(tolerance))

    df_links.to_file(r"data\Network\processed\new_links_realistic.gpkg")

    # Also store the point which are not joinable due to banned land cover
    # Writing to the CSV file with a header
    #df_inaccessible_points = pd.DataFrame(generated_points_unaccessible, columns=["point_id"])
    #df_inaccessible_points.to_csv(r"data\Network\processed\points_inaccessible.csv", index=False)

    return


def raster_to_graph(raster_data):
    rows, cols = raster_data.shape
    graph = nx.grid_2d_graph(rows, cols)
    graph.add_edges_from([
                         ((x, y), (x + 1, y + 1))
                         for x in range(cols)
                         for y in range(rows)
                     ] + [
                         ((x + 1, y), (x, y + 1))
                         for x in range(cols)
                         for y in range(rows)
                     ], weight=1.4)

    # Remove edges to forbidden cells (assuming forbidden cells are marked with value 1)
    for y in range(rows):
        for x in range(cols):
            if raster_data[y, x] > 0:
                graph.remove_node((y, x))

    return graph


def find_path(graph, start, end, list_no_path, point_end):
    # Find the shortest path using A* algorithm or dijkstra
    # You might want to include a heuristic function for A*
    try:
        #path = nx.astar_path(graph, start, end)
        path = nx.dijkstra_path(graph, start, end)
        return path, list_no_path
    except nx.NetworkXNoPath:
        list_no_path.append(point_end)
        print("No path found ", point_end)
        return None, list_no_path


def plot_corridor(network, limits, location, current_nodes=False, new_nodes=False, new_links=False, access_link=False):

    fig, ax = plt.subplots(figsize=(10, 10))

    network = network[(network["Rank"] == 1) & (network["Opening Ye"] < 2023) & (network["NAME"] != 'Freeway Tunnel planned') & (
                network["NAME"] != 'Freeway planned')]

    # Define square to show perimeter of investigation
    square = Polygon([(limits[0], limits[2]), (limits[1], limits[2]), (limits[1], limits[3]), (limits[0], limits[3])])
    frame = gpd.GeoDataFrame(geometry=[square], crs=network.crs)

    #df_voronoi.plot(ax=ax, facecolor='none', alpha=0.2, edgecolor='k')

    if access_link==True:
        access = network[network["NAME"] == "Freeway access"]
        access["point"] = access.representative_point()
        access.plot(ax=ax, color="red", markersize=50)

    if isinstance(new_links, gpd.GeoDataFrame):
        new_links.plot(ax=ax, color="darkgray")

    if isinstance(new_nodes, gpd.GeoDataFrame):
        new_nodes.plot(ax=ax, color="blue", markersize=50)

    network.plot(ax=ax, color="black", lw=4)

    if isinstance(current_nodes, gpd.GeoDataFrame):
        current_nodes.plot(ax=ax, color="black", markersize=50)

    # Plot the location as points
    location.plot(ax=ax, color="black", markersize=75)
    # Add city names to the plot
    for idx, row in location.iterrows():
        plt.annotate(row['location'], xy=row["geometry"].coords[0], ha='left', va="bottom", fontsize=15)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])

    #plt.title("Voronoi polygons to each highway access point")
    plt.savefig(r"plot\network_base_generated.png", dpi=300)
    plt.show()

    return


def single_tt_voronoi_ton_one(folder_path):

    # List all gpkg files in the folder
    gpkg_files = [f for f in os.listdir(folder_path) if f.endswith('Voronoi.gpkg')]

    # Initialize an empty list to store dataframes
    dataframes = []

    for file in gpkg_files:
        # Read the gpkg file
        gdf = gpd.read_file(os.path.join(folder_path, file))

        # Use regular expression to extract the XXX number from the filename
        id_development = re.search(r'dev(\d+)_Voronoi', file)
        if id_development:
            id_development = int(id_development.group(1))
        else:
            print("Error in predict >> 394")
            continue  # Skip file if no match is found

        # Add the ID_development as a new column
        gdf['ID_development'] = id_development

        # Append the dataframe to the list
        dataframes.append(gdf)

    # Concatenate all dataframes into one
    combined_gdf = pd.concat(dataframes)

    # Save the combined dataframe as a new gpkg file
    combined_gdf.to_file("data\Voronoi\combined_developments.gpkg", driver="GPKG")


def import_elevation_model(new_resolution):

    # Read CSV file containing the ZIP file links
    csv_file = r"data\elevation_model\ch.swisstopo.swissalti3d-pivq0Jb7.csv"
    df = pd.read_csv(csv_file, names=["url"], header=None)

    # Download and extract ZIP files
    for url in df["url"]:
        r = requests.get(url)
        zip_path = r"data\elevation_model\zip_files\temp.zip"
        with open(zip_path, 'wb') as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(r"data\elevation_model\extracted_xyz_files")

    # Find all XYZ files
    xyz_files = glob.glob(r"data\elevation_model\extracted_xyz_files\*.xyz")

    # Initialize an empty DataFrame for the results
    concatenated_data = pd.DataFrame(columns=["X", "Y", "Z"])

    # Calculate the minimum coordinates based on the first file
    sample_data = pd.read_csv(xyz_files[0], sep=" ")
    min_x, min_y = sample_data['X'].min(), sample_data['Y'].min()

    # Process each file
    for i, file in enumerate(xyz_files, start=1):
        downsampled_data = downsample_elevation_xyz_file(file, min_x, min_y, resolution=new_resolution)
        concatenated_data = pd.concat([concatenated_data, downsampled_data])
        # Print progress
        print(f"Processed file {i}/{len(xyz_files)}: {file}")

    # Reset index
    concatenated_data.reset_index(drop=True, inplace=True)
    print(concatenated_data.shape)

    # Convert the DataFrame to a 2D grid
    min_x, max_x = concatenated_data['X'].min(), concatenated_data['X'].max()
    min_y, max_y = concatenated_data['Y'].min(), concatenated_data['Y'].max()

    # Calculate the number of rows and columns
    cols = int((max_x - min_x) / new_resolution) + 1
    rows = int((max_y - min_y) / new_resolution) + 1

    # Create an empty grid
    raster = np.full((rows, cols), np.nan)

    # Populate the grid with Z values
    for _, row in concatenated_data.iterrows():
        col_idx = int((row['X'] - min_x) / new_resolution)
        row_idx = int((max_y - row['Y']) / new_resolution)
        raster[row_idx, col_idx] = row['Z']

    # Define the georeferencing transform
    transform = from_origin(min_x, max_y, new_resolution, new_resolution)

    # Write the data to a GeoTIFF file
    with rasterio.open(r'data\elevation_model\elevation.tif', 'w', driver='GTiff',
                       height=raster.shape[0], width=raster.shape[1],
                       count=1, dtype=str(raster.dtype),
                       crs='EPSG:2056', transform=transform) as dst:
        dst.write(raster, 1)

    return


def downsample_elevation_xyz_file(file_path, min_x, min_y, resolution):
    # Read the file
    data = pd.read_csv(file_path, sep=" ")

    # Filter the data
    filtered_data = data[((data['X'] - min_x) % resolution == 0) & ((data['Y'] - min_y) % resolution == 0)]

    return filtered_data


def get_road_elevation_profile():
    # Import the dataframe containing the rounting of the highway links
    links = gpd.read_file(r"data\Network\processed\new_links_realistic.gpkg")

    # Open the GeoTIFF file
    elevation_raster = r"data\elevation_model\elevation.tif"

    def interpolate_linestring(linestring, interval):
        length = linestring.length
        num_points = int(np.ceil(length / interval))
        points = [linestring.interpolate(distance) for distance in np.linspace(0, length, num_points)]
        return points

    def sample_raster_at_points(points, raster):
        values = []
        for point in points:
            row, col = raster.index(point.x, point.y)
            value = raster.read(1)[row, col]
            values.append(value)
        return values

    # Define the sampling interval (e.g., every 10 meters)
    sampling_interval = 50

    with rasterio.open(elevation_raster) as raster:
        print(raster.crs)
        print(links.crs)

        # Interpolate points and extract raster values for each linestring
        links['elevation_profile'] = links['geometry'].apply(
            lambda x: sample_raster_at_points(
                interpolate_linestring(x, sampling_interval), raster))


    # Somehow find how to investigate the need for tunnels based on the elevation profile
    # Assuming you have a DataFrame named 'df' with a column 'altitude'
    # Calculate the elevation difference between successive values

    # Iterate through the DataFrame using iterrows and calculate the elevation difference
    links['elevation_difference'] = links.apply(lambda row: np.diff(np.array(row['elevation_profile'])), axis=1)

    # Compute absolute elevation
    links["elevation_absolute"] = links.apply(lambda row: np.absolute(row["elevation_difference"]), axis=1)


    links["slope"] = links.apply(lambda row: row["elevation_absolute"] / 50 * 100, axis=1)

    # Compute mean elevation
    links['slope_mean'] = links.apply(lambda row: np.mean(row['slope']), axis=1)

    # Compute number of values bigger than thresshold
    links["steep_section"] = links.apply(lambda  row: (row["slope"] < 5).sum(), axis=1)

    links["check_needed"] = (links['slope_mean'] > 5) | (links["steep_section"] > 40)
    links = links.drop(columns=["elevation_difference", "elevation_absolute", "slope", "slope_mean", "steep_section"])
    #links["elevation_profile"] = links["elevation_profile"].astype("string")
    #links.to_file(r"data\Network\processed\new_links_realistic_elevation.gpkg")
    return links


def get_tunnel_candidates(df):
    print("You will have to define the needed tunnels and bridges for ", df["check_needed"].sum() , " section.")

    df["elevation_profile"] = df["elevation_profile"].astype("object")
    # Custom dialog class for pop-up
    class CustomDialog(Dialog):
        def __init__(self, parent, row):
            self.row = row
            Dialog.__init__(self, parent)

        def body(self, master):
            # Create a figure for the plot
            self.fig, self.ax = plt.subplots()
            x_values = np.arange(0, len(self.row['elevation_profile'])) * 50
            self.ax.plot(x_values, self.row['elevation_profile'])
            self.ax.set_title('Elevation profile')
            self.ax.set_xlabel('Distance (m)')
            self.ax.set_ylabel('Elevation (m. asl.)')

            # Create labels and input fields for questions
            tk.Label(master, text="How much tunnel is required in meters:").pack()
            self.tunnel_len_entry = tk.Entry(master)
            self.tunnel_len_entry.pack()

            tk.Label(master, text="How much bridge is required in meters:").pack()
            self.bridge_len_entry = tk.Entry(master)
            self.bridge_len_entry.pack()

            # Create a canvas to display the plot
            canvas = FigureCanvasTkAgg(self.fig, master=master)
            canvas.get_tk_widget().pack()

        def apply(self):
            # Get the user's input values
            tunnel_len = int(self.tunnel_len_entry.get())
            bridge_len = int(self.bridge_len_entry.get())

            # Update DataFrame with user's input
            df.at[self.row.name, 'tunnel_len'] = tunnel_len
            df.at[self.row.name, 'bridge_len'] = bridge_len
    # Create new columns for user input
    df['tunnel_len'] = None
    df['bridge_len'] = None


    # Iterate through the DataFrame and show the custom pop-up for rows with 'check_needed' set to True
    for index, row in df.iterrows():
        if row['check_needed']:
            root = tk.Tk()
            root.withdraw()
            dlg = CustomDialog(root, row)
            #dlg.wait_window()
    df["elevation_profile"]=df["elevation_profile"].astype('string')
    print(df)
    #df.to_file(r"data\Network\processed\new_links_realistic_tunnel.gpkg")
    df.to_file(r"data\Network\processed\new_links_realistic_tunnel-terminal.gpkg")


def tunnel_bridges(df):
    # The aim is to estimate the need of tunnels and bridge based on the elevation profile of each link
    print(df.head().to_string())

    # Define max slope allowed on a highway
    max_slope = 7  # in percent
    max_slope = max_slope / 100

    """
    # Based on the first and the last element of the elevation profile, the elevation difference is calculated
    # The length of the total link = nbr elements in elevation profile * 50
    # Get first an last element of the elevation profile
    df["total_dif"] = df.apply(lambda row: row["elevation_profile"][-1] - row["elevation_profile"][0], axis=1)
    df["total_length"] = df.apply(lambda row: len(row["elevation_profile"]) * 50, axis=1)
    df["total_slope"] = df.apply(lambda row: row["total_dif"] / row["total_length"], axis=1)
    # Check if total slope is bigger than 5%
    df["too_steep"] = df.apply(lambda row: row["total_slope"] > max_slope, axis=1)
    # Print the amount of too_steep = True
    print("There are ", df["too_steep"].sum(), " links that are too steep.")

    # Check how big the elevation difference is between each consecutive point and store that as new list
    # Thus for each row check elevation_i with elevation_i+1 knowing distance is 50m
    df["single_elevation_difference"] = df.apply(lambda row: np.diff(np.array(row["elevation_profile"])), axis=1)
    df["single_slope"] = df.apply(lambda row: np.absolute(row["single_elevation_difference"]) / 50, axis=1)

    # Check if there there are slopes with more slope than 5%, return True, False
    df["too_steep_single"] = df['single_slope'].apply(lambda x: any(np.array(x) > max_slope))

    # Print the amount of too_steep_single = True, and print the entire amount of links
    print("There are ", df["too_steep_single"].sum(), " (",len(df),") links that are too steep.")

    #print("There are ", df["too_steep_single"].sum(), " links that are too steep.")
    """
    """
    def adjust_elevation(elevation, max_slope=0.05):
        n = len(elevation)
        x = np.arange(n) * 50  # Assuming each point is 50m apart

        # Define the objective function for optimization
        def objective(new_elevation):
            # Count the number of changes
            changes = np.sum(new_elevation != elevation)
            return changes

        # Define constraints for the slope
        def slope_constraint(new_elevation, i):
            if i < n - 1:
                return max_slope * 50 - np.abs(new_elevation[i + 1] - new_elevation[i])
            return 0

        cons = [{'type': 'ineq', 'fun': slope_constraint, 'args': (i,)} for i in range(n - 1)]

        # Run the optimization
        result = minimize(objective, elevation, constraints=cons, method='SLSQP')

        new_elevation = result.x
        changes = new_elevation != elevation
        return new_elevation, changes
    """
    """
    def adjust_elevation(elevation, max_slope=max_slope):
        n = len(elevation)

        def objective(new_elevation):
            return np.sum(new_elevation != elevation)

        def slope_constraint(new_elevation, i):
            if i > 0:
                slope = np.abs(new_elevation[i] - new_elevation[i - 1]) / 50
                return max_slope - slope
            return 0

        cons = [{'type': 'ineq', 'fun': slope_constraint, 'args': (i,)} for i in range(1, n)]
        # Bounds: First and last points remain the same, others have +/- 50m range
        bounds = [(elevation[0], elevation[0])] + [(val - 200, val + 200) for val in elevation[1:-1]] + [
            (elevation[-1], elevation[-1])]

        result = minimize(objective, elevation, method='SLSQP', bounds=bounds, constraints=cons, options={'disp': True})

        if not result.success:
            print(f"Optimization failed: {result.message}")

        new_elevation = result.x
        changes = new_elevation != elevation
        return new_elevation, changes


    new_profiles = []
    change_flags = []
    # iterate over all rows of the dataframe and print process bar

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        elevation = np.array(row['elevation_profile'])
        new_elevation, changes = adjust_elevation(elevation)
        new_profiles.append(new_elevation)
        change_flags.append(changes)

        # Check how big the elevation difference is between each consecutive point and store that as new list
        # Thus for each row check elevation_i with elevation_i+1 knowing distance is 50m
        single_slope = np.diff(np.array(new_elevation)) / 50
        too_steep_single = any(np.abs(single_slope) > max_slope)
        if too_steep_single:
            print("There are links that are too steep.", np.where(single_slope > max_slope), changes)

    df['new_elevation'] = new_profiles
    df['changes'] = change_flags
    """

    def optimize_values_min_changes(values, max_slope):
        max_diff = max_slope * 50

        # Initialize the LP problem
        prob = pulp.LpProblem("SlopeOptimizationMinChanges", pulp.LpMinimize)

        # Decision variables
        lp_vars = {i: pulp.LpVariable(f"v_{i}") for i in range(len(values))}
        change_vars = {i: pulp.LpVariable(f"c_{i}", 0, 1, cat='Binary') for i in range(len(values))}

        # Objective function: minimize the number of points that are changed
        prob += pulp.lpSum(change_vars[i] for i in range(len(values)))

        # Constraints for slope and changes
        for i in range(len(values)):
            if i > 0:
                prob += lp_vars[i] - lp_vars[i - 1] <= max_diff
                prob += lp_vars[i - 1] - lp_vars[i] <= max_diff
            # Change indicator constraints
            # If change_var is 0, lp_var must be equal to the original value
            prob += lp_vars[i] - values[i] <= 1e9 * change_vars[i]
            prob += values[i] - lp_vars[i] <= 1e9 * change_vars[i]

        # Constraints for keeping first and last values unchanged
        # Enforce first and last values remain unchanged
        prob += lp_vars[0] == values[0]
        prob += change_vars[0] == 0  # No change for the first element
        prob += lp_vars[len(values) - 1] == values[len(values) - 1]
        prob += change_vars[len(values) - 1] == 0  # No change for the last element

        # Solve the problem without printing messages
        #prob.solve(pulp.PULP_CBC_CMD(msg=False))
        prob.solve(pulp.PULP_CBC_CMD(msg=True))

        # Check if the problem is infeasible
        if prob.status != pulp.LpStatusOptimal:
            print("Infeasible Problem")
            return None

        # Get the optimized values
        optimized_values = [pulp.value(lp_vars[i]) for i in range(len(values))]
        return optimized_values

    # Add new column with optimized elevation profile
    tqdm.pandas(desc="Optimizing elevation profiles")
    df['new_elevation'] = df.progress_apply(
        lambda row: optimize_values_min_changes(row["elevation_profile"], max_slope) if row['check_needed'] else row[
            'elevation_profile'],
        axis=1
    )

    # Drop with "new_elevation" == None
    df = df.dropna(subset=['new_elevation'])

    # Add new column showing the difference between the old and new elevation profile, 0 if not - 1 if yes
    df['changes'] = df.apply(lambda row: np.array(row['elevation_profile']) != np.array(row['new_elevation']), axis=1)

    print(df.head(20).to_string())

    def check_for_bridge_tunnel(elevation, new_elevation, changes):
        elevation = np.array(elevation)
        new_elevation = np.array(new_elevation)
        changes = np.array(changes)

        flags = np.zeros(len(elevation))
        height_diff = new_elevation - elevation

        for i in range(len(elevation) - 1):
            # This check ensures that tunnel are longer than 50m and that elevation difference is at least 10m
            # It is assumed that otherwise there is no need for tunnel or bridge
            if changes[i]:
                if height_diff[i] <= -10 and height_diff[i + 1] <= -10:
                    flags[i] = -1  # Tunnel
                elif height_diff[i] >= 10 and height_diff[i + 1] >= 10:
                    flags[i] = 1  # Bridge

        return list(flags)  # Convert back to list for DataFrame storage

    df['bridge_tunnel_flags'] = df.apply(
        lambda row: check_for_bridge_tunnel(row['elevation_profile'], row['new_elevation'], row['changes']), axis=1)
    """
    def create_linestrings(elevation_profile, flags, original_linestring):
        tunnel_linestrings = []
        bridge_linestrings = []
        current_line = []
        current_flag = flags[0]

        for i, flag in enumerate(flags):
            # Adjust the point position by 25 meters
            point_position = max(i * 50 - 25, 0)

            # Check for the end of a current structure or the last flag
            if (flag != current_flag or i == len(flags) - 1) and current_line:
                # Extend the current line by 25 meters if possible
                end_position = min((i + 1) * 50 - 25, len(elevation_profile) * 50)
                current_line.append(original_linestring.interpolate(end_position / original_linestring.length))
                if current_flag == -1:
                    tunnel_linestrings.append(LineString(current_line))
                elif current_flag == 1:
                    bridge_linestrings.append(LineString(current_line))
                current_line = []

            # Check for the start of a new structure
            if (current_flag in [0, 1] and flag == -1) or (current_flag in [0, -1] and flag == 1):
                current_line.append(original_linestring.interpolate(point_position / original_linestring.length))

            current_flag = flag

        return tunnel_linestrings, bridge_linestrings

    tunnel_df = pd.DataFrame(columns=['link_id', 'tunnel_linestring'])
    bridge_df = pd.DataFrame(columns=['link_id', 'bridge_linestring'])

    for index, row in df.iterrows():
        original_linestring = row["geometry"]
        tunnels, bridges = create_linestrings(row['elevation_profile'], row['bridge_tunnel_flags'], original_linestring)
        tunnel_df = tunnel_df.append({'link_id': index, 'tunnel_linestring': tunnels}, ignore_index=True)
        bridge_df = bridge_df.append({'link_id': index, 'bridge_linestring': bridges}, ignore_index=True)
    """
    """
    def process_row(row):
        original_linestring = row["geometry"]
        flags = row['bridge_tunnel_flags']
        elevation_profile = row['elevation_profile']

        tunnel_linestrings = []
        bridge_linestrings = []
        current_line = []
        current_type = 0  # 0 for road, -1 for tunnel, 1 for bridge

        for i, flag in enumerate(flags):
            # Interpolate the point on the linestring
            point_position = i * 50  # Adjust as per your requirement
            point = original_linestring.interpolate(point_position / original_linestring.length)

            if flag != current_type:
                if current_line:
                    # Complete the current linestring
                    current_line.append(point)
                    if current_type == -1:
                        tunnel_linestrings.append(LineString(current_line))
                    elif current_type == 1:
                        bridge_linestrings.append(LineString(current_line))

                current_line = [] if flag != 0 else [point]
                current_type = flag
            elif flag != 0:
                current_line.append(point)

        # Handle the last segment
        if current_line:
            if current_type == -1:
                tunnel_linestrings.append(LineString(current_line))
            elif current_type == 1:
                bridge_linestrings.append(LineString(current_line))

        return tunnel_linestrings, bridge_linestrings

    # Processing each row and storing the results
    tunnel_data = []
    bridge_data = []
    
    
        for index, row in df.iterrows():
        tunnels, bridges = process_row(row)
        for tunnel in tunnels:
            tunnel_data.append({'link_id': index, 'tunnel_linestring': tunnel})
        for bridge in bridges:
            bridge_data.append({'link_id': index, 'bridge_linestring': bridge})
    """


    # Make a lineplot of both lists in elevation profile and new elevation profile on the same plot
    # Plot the elevation profile

    # df_to_plot = df[df["ID"] == 103]
    df_to_plot = df[df["ID_new"] == 990]
    for index, row in df_to_plot.iterrows():
        # initialize flat figure
        plt.figure(figsize=(10, 3))
        plt.plot(row["elevation_profile"], label="Original", color="gray", linewidth=3, zorder=2)
        # Plot the new elevation profile
        plt.plot(row["new_elevation"], label="Optimized", color="black", zorder=3)
        # Multiply x ticks by 50 to get distance in meters
        plt.xticks(np.arange(0, len(row["elevation_profile"]), step=10), np.arange(0, len(row["elevation_profile"]) * 50, step=500))
        # Add labels
        plt.xlabel("Link distance (m)")
        plt.ylabel("Elevation (m. asl.)")
        # Mark where tunnel and where bridge based on flags
        for i, flag in enumerate(row['bridge_tunnel_flags']):
            if flag == -1:
                plt.axvline(x=i+0.5, color='lightgray', linestyle='solid', linewidth=12, zorder=1, alpha=0.7)
            elif flag == 1:
                plt.axvline(x=i+0.5, color='lightblue', linestyle='solid', linewidth=12, zorder=1, alpha=0.7)

        # Create custom patches for legend
        original_line = mlines.Line2D([], [], color='gray', linewidth=3, label='Original')
        optimized_line = mlines.Line2D([], [], color='black', label='Optimized')
        tunnel_patch = mpatches.Patch(color='lightgray', alpha=0.7, label='Required tunnel')
        bridge_patch = mpatches.Patch(color='lightblue', alpha=0.7, label='Required bridge')

        # Modify the legend to include custom patches
        legend = plt.legend(handles=[original_line, optimized_line, tunnel_patch, bridge_patch],
                   title="Elevation profile", loc="lower left", bbox_to_anchor=(1.04, 0), frameon=False)
        legend.get_title().set_horizontalalignment('left')
        plt.tight_layout()
        plt.savefig(fr"plot\network\elevation\new_profile{row['ID_new']}.png", dpi=300)
        plt.show()





    def split_linestring_at_distance(linestring, distance):
        """Split a LineString at a specified distance."""
        if distance <= 0.0 or distance >= linestring.length:
            return [linestring]
        split_point = linestring.interpolate(distance)
        split_result = split(linestring, split_point)
        return list(split_result.geoms)

    def process_flags(original_linestring, flags):
        road_linestrings = []
        tunnel_linestrings = []
        bridge_linestrings = []

        current_line = original_linestring
        last_split = 0

        for i in range(1, len(flags)):
            flag = flags[i]
            prev_flag = flags[i - 1]

            if flag != prev_flag:
                # Determine split point
                if flag == 0:
                    split_point = i * 50 + 25
                elif prev_flag == 0:
                    split_point = max(0, i * 50 - 25)
                else:
                    split_point = i * 50

                # Ensure split_point is within the linestring's length
                split_point = min(split_point, current_line.length)

                # Split the linestring
                split_segments = split_linestring_at_distance(current_line, split_point - last_split)

                if len(split_segments) > 1:
                    segment, current_line = split_segments
                    last_split = split_point

                    # Assign segment to the appropriate list
                    if prev_flag == -1:
                        tunnel_linestrings.append(segment)
                    elif prev_flag == 1:
                        bridge_linestrings.append(segment)
                    else:
                        road_linestrings.append(segment)

        # Handle the last segment
        if current_line:
            last_flag = flags[-1]
            if last_flag == -1:
                tunnel_linestrings.append(current_line)
            elif last_flag == 1:
                bridge_linestrings.append(current_line)
            else:
                road_linestrings.append(current_line)

        return road_linestrings, tunnel_linestrings, bridge_linestrings

    tunnel_data = []
    bridge_data = []
    road_data = []

    for index, row in df.iterrows():
        road_linestrings, tunnel_linestrings, bridge_linestrings = process_flags(row["geometry"], row['bridge_tunnel_flags'])
        for tunnel in tunnel_linestrings:
            tunnel_data.append({'link_id': index, 'tunnel_linestring': tunnel})
        for bridge in bridge_linestrings:
            bridge_data.append({'link_id': index, 'bridge_linestring': bridge})
        for road in road_linestrings:
            road_data.append({'link_id': index, 'road_linestring': road})


    # Creating DataFrames
    tunnel_df = pd.DataFrame(tunnel_data)
    bridge_df = pd.DataFrame(bridge_data)
    road_df = pd.DataFrame(road_data)


    # Calculate Lengths for Each Linestring
    #tunnel_df['length'] = tunnel_df['tunnel_linestring'].apply(lambda x: sum([line.length for line in x]))
    #bridge_df['length'] = bridge_df['bridge_linestring'].apply(lambda x: sum([line.length for line in x]))

    """
    def convert_to_multilinestring(linestrings):
        # Filter out None values and ensure that linestrings is not empty
        valid_linestrings = [ls for ls in linestrings if ls is not None]
        if valid_linestrings:
            return MultiLineString(valid_linestrings)
        return None
    """

    # Convert tunnel DataFrame to GeoDataFrame

    tunnel_gdf = gpd.GeoDataFrame(tunnel_df, geometry='tunnel_linestring')
    bridge_gdf = gpd.GeoDataFrame(bridge_df, geometry='bridge_linestring')
    road_gdf = gpd.GeoDataFrame(road_df, geometry='road_linestring')


    #tunnel_gdf['geometry'] = tunnel_gdf['tunnel_linestring'].apply(convert_to_multilinestring)

    # Convert bridge DataFrame to GeoDataFrame

    #bridge_gdf['geometry'] = bridge_gdf['bridge_linestring'].apply(convert_to_multilinestring)

    tunnel_gdf.set_crs(epsg=2056, inplace=True)
    bridge_gdf.set_crs(epsg=2056, inplace=True)
    road_gdf.set_crs(epsg=2056, inplace=True)

    # Calculate total length of tunnels for each link
    tunnel_gdf['total_tunnel_length'] = tunnel_gdf['tunnel_linestring'].apply(lambda x: x.length if x is not None else 0)
    # Calculate total length of bridges for each link
    bridge_gdf['total_bridge_length'] = bridge_gdf['bridge_linestring'].apply(lambda x: x.length if x is not None else 0)
    # Calculate total length of road for each link
    road_gdf['total_road_length'] = road_gdf['road_linestring'].apply(lambda x: x.length if x is not None else 0)

    # Join tunnel lengths
    df = df.join(tunnel_gdf.set_index('link_id')['total_tunnel_length'])
    # Join bridge lengths
    df = df.join(bridge_gdf.set_index('link_id')['total_bridge_length'])
    # Aggregate Lengths for Each Link
    #total_tunnel_lengths = tunnel_df.groupby('link_id')['length'].sum()
    #total_bridge_lengths = bridge_df.groupby('link_id')['length'].sum()

    # Join these lengths back to the original DataFrame
    #df = df.join(total_tunnel_lengths, rsuffix='_tunnel')
    #df = df.join(total_bridge_lengths, rsuffix='_bridge')

    # Drop column with list from df DataFrame
    df = df.drop(columns=["bridge_tunnel_flags", "new_elevation", "changes", "elevation_profile"])
    #tunnel_gdf = tunnel_gdf.drop(columns=["tunnel_linestring"])
    #bridge_gdf = bridge_gdf.drop(columns=["bridge_linestring"])

    #print(bridge_gdf.head(10).to_string())
    #print(df.head().to_string())
    # safe file as geopackage
    df.to_file(r"data\Network\processed\new_links_realistic_tunnel_adjusted.gpkg")
    tunnel_gdf.to_file(r"data\Network\processed\edges_tunnels.gpkg")
    bridge_gdf.to_file(r"data\Network\processed\edges_bridges.gpkg")
    road_gdf.to_file(r"data\Network\processed\edges_roads.gpkg")

    """
    def slope_constrained_curve_fit(x, y):
        # Define an objective function for optimization
        def objective_function(coeffs):
            # Calculate the polynomial values
            y_pred = np.polyval(coeffs, x)
            # Calculate the slope and enforce the slope constraint (5%)
            slopes = np.diff(y_pred) / np.diff(x)
            slope_penalty = np.sum(np.maximum(0, np.abs(slopes) - 0.05))
            # Objective: Minimize the sum of squared differences and slope penalty
            return np.sum((y_pred[:-1] - y[:-1]) ** 2) + slope_penalty

        # Initial guess for polynomial coefficients
        initial_guess = np.polyfit(x, y, deg=15)
        # Run the optimization
        result = minimize(objective_function, initial_guess, method='SLSQP')
        return result.x

    for index, row in df.iterrows():
        elevation = row['elevation_profile']
        x = np.arange(len(elevation)) * 50  # Assuming each point is 50m apart
        coefficients = slope_constrained_curve_fit(x, elevation)
        fitted_curve = np.polyval(coefficients, x)

        # Identify sections for bridges or tunnels
        # (where the difference between actual and fitted curve is more than 10m)
        bridge_tunnel_sections = np.abs(fitted_curve - elevation) > 10

        # Visualization for analysis
        plt.figure()
        plt.plot(x, elevation, label='Actual Elevation')
        plt.plot(x, fitted_curve, label='Fitted Curve')
        plt.fill_between(x, elevation, fitted_curve, where=bridge_tunnel_sections,
                         color='red', alpha=0.3, label='Bridge/Tunnel Sections')
        plt.title(f'Elevation Profile {index}')
        plt.xlabel('Distance (m)')
        plt.ylabel('Elevation (m)')
        plt.legend()
        plt.show()
    """
    return

