import os
import time

import geopandas as gpd
import networkx as nx
import pandas as pd
import pyproj
from googlemaps import Client as GoogleMapsClient
from shapely import LineString


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


utm_proj = pyproj.CRS("EPSG:32632")  # For UTM zone 32N (adjust if needed)
latlng_proj = pyproj.CRS("EPSG:4326")  # WGS84 Lat/Lng


def utm_to_latlng(easting, northing):
    """Convert UTM coordinates (easting, northing) to latitude and longitude."""
    lon, lat = pyproj.transform(utm_proj, latlng_proj, easting, northing)
    return lat, lon


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
