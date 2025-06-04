import networkx
from shapely.geometry import MultiLineString
from shapely.ops import split
import gc

import time

from sympy.polys.subresultants_qq_zz import final_touches

import paths
import settings
from scoring import *
from scoring import split_via_nodes, merge_lines


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

    network_railway_service_path = paths.get_rail_services_path(settings.rail_network)
    raw_edges = gpd.read_file(network_railway_service_path)

    #raw_edges['FromEnd'] = raw_edges['FromEnd'].astype(bool)
    #raw_edges['ToEnd'] = raw_edges['ToEnd'].astype(bool)

    raw_edges['FromEnd'] = raw_edges['FromEnd'].astype(str).map({'1': True, '0': False})
    raw_edges['ToEnd'] = raw_edges['ToEnd'].astype(str).map({'1': True, '0': False})

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

def filter_unnecessary_links(rail_network):
    """
    Filter out unnecessary links in the new_links GeoDataFrame.
    Saves the filtered links as a GeoPackage file.
    """
    network_railway_service_path = paths.get_rail_services_path(rail_network)
    raw_edges = gpd.read_file(network_railway_service_path)
    time.sleep(1)  # Ensure file access is sequential
    line_gdf = gpd.read_file(r"data\Network\processed\new_links.gpkg")

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


def calculate_new_service_time():
    # Set up working directory and file paths
    os.chdir(paths.MAIN)
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
    new_links.to_file(paths.NEW_LINKS_UPDATED_PATH, driver="GPKG")

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


def create_network_foreach_dev():
    """
    Creates individual GeoPackages for each unique development (identified by dev_id),
    combining the entire old network with the corresponding development in both directions.
    """

    # Load the GPK file
    output_directory = paths.DEVELOPMENT_DIRECTORY
    os.makedirs(output_directory, exist_ok=True)  # Ensure the output directory exists

    # Read the GeoPackage
    gdf = gpd.read_file(paths.NETWORK_WITH_ALL_MODIFICATIONS)
    # Fill NaN with 0 for all columns except 'geometry'
    gdf.loc[:, gdf.columns != 'geometry'] = gdf.loc[:, gdf.columns != 'geometry'].fillna(0)
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

        # Remove rows where ID_point equals To_ID-point
        potential_matches = potential_matches[potential_matches['TO_ID_new'] != gen_row['ID_point']]

        # If multiple matches exist, find the closest one
        if not potential_matches.empty:
            closest_row = potential_matches.loc[
                potential_matches['geometry_current'].distance(gen_row['geometry']).idxmin()
            ]

            # Create the connection
            tolerance = 1e-3  # or another small value depending on your needs
            if gen_row['geometry'].distance(closest_row['geometry_current']) < tolerance:
                # If the geometries are almost equal, skip this connection
                continue
            else:
                # Create a connection with the geometry of the generated point and the closest infrastructure point
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

    try:
        # Load the data
        df_network = gpd.read_file(paths.RAIL_SERVICES_AK2035_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {paths.RAIL_SERVICES_AK2035_PATH}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the file: {e}")

    # Create an undirected graph
    G = nx.Graph()

    # Split the lines with a Via column
    df_split = split_via_nodes(df_network)
    df_split = merge_lines(df_split)

    # Add edges to the graph
    for _, row in df_split.iterrows():
        G.add_edge(row['FromNode'], row['ToNode'], weight=row['TotalTravelTime'])

    # Ensure nodes and connections IDs are integers
    G = nx.relabel_nodes(G, {n: int(n) for n in G.nodes})
    new_connections['from_ID_new'] = new_connections['from_ID_new'].astype(int)
    new_connections['to_ID'] = new_connections['to_ID'].astype(int)

    # Get all nodes available in the graph
    available_nodes = set(G.nodes())

    # Compute the routes
    results = []
    for _, row in new_connections.iterrows():
        from_node = row['from_ID_new']
        to_node = row['to_ID']

        if from_node not in available_nodes or to_node not in available_nodes:
            # Skip this iteration if either node is not in the network
            """            
                results.append({
                'from_ID_new': from_node,
                'to_ID': to_node,
                'via_nodes': -99  # No path exists because nodes aren't in the network
            })
            """
            continue

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


def update_network_with_new_links(rail_network_selection, new_links_updated_path):
    """
    Add new links to the railway network, marking them as new and generating both directions.
    Ensure FromStation and ToStation are mapped correctly using Rail_Node data.
    """
    # Load data
    network_railway_service_path = paths.get_rail_services_path(rail_network_selection)

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

    # Calculate the Via nodes for the new connections (extended lines)
    new_links_extended_lines = new_links_updated[new_links_updated['dev_id'] < settings.dev_id_start_new_direct_connections]
    via_df = get_via(new_links_extended_lines)

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
        lambda x: '-99' if pd.isna(x) or x == [-99] else ','.join(map(str, x))
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
    combined_new_links_gdf = gpd.GeoDataFrame(combined_new_links, geometry=combined_new_links.geometry)

    # Standardize station names in FromStation and ToStation
    standardize_station_names = {
        "Wetzikon": "Wetzikon ZH",
        # Add more mappings here if needed
    }

    combined_new_links["FromStation"] = combined_new_links["FromStation"].replace(standardize_station_names)
    combined_new_links["ToStation"] = combined_new_links["ToStation"].replace(standardize_station_names)


    # Combine with original network
    combined_network = pd.concat([network_railway_service, combined_new_links_gdf], ignore_index=True)


    return combined_network


def split_via_nodes_mod(df,delete_via_edges=False):
    """
    Split rows where the 'Via' column contains intermediate nodes.
    Each new row will represent a sub-edge from 'FromNode' to 'ToNode' including intermediate nodes.
    Station names for 'FromStation' and 'ToStation' are updated based on corresponding nodes.

    Parameters:
        df (pd.DataFrame): Original DataFrame containing 'FromNode', 'ToNode', 'FromStation', 'ToStation', and 'Via' columns.

    Returns:
        pd.DataFrame: Expanded DataFrame with all sub-edges and updated station names.
    """

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
    expanded_rows = pd.DataFrame()
    edges_to_remove = []

    for _, row in df.iterrows():
        # Extract FromNode, ToNode, and parsed Via
        from_node = row['FromNode']
        to_node = row['ToNode']
        via_nodes = row['Via']

        # Create a complete path of nodes: FromNode -> ViaNode1 -> ... -> ViaNodeN -> ToNode
        all_nodes = [from_node] + via_nodes + [to_node]

        # Create sub-edges for each consecutive pair of nodes
        new_rows = []

        for i in range(len(all_nodes) - 1):
            new_row = row.copy()
            new_row['FromNode'] = all_nodes[i]
            new_row['ToNode'] = all_nodes[i + 1]
            new_row['FromStation'] = node_to_station.get(all_nodes[i], f"Unknown Node {all_nodes[i]}")
            new_row['ToStation'] = node_to_station.get(all_nodes[i + 1], f"Unknown Node {all_nodes[i + 1]}")
            new_row['Via'] = []
            new_rows.append(new_row)

        expanded_rows = pd.concat([expanded_rows, pd.DataFrame(new_rows)], ignore_index=True)

        if delete_via_edges and len(via_nodes) > 0:
            edges_to_remove.append((from_node, to_node))

    if edges_to_remove:
        expanded_rows = expanded_rows.loc[~expanded_rows.apply(
            lambda x: (x['FromNode'], x['ToNode']) in edges_to_remove, axis=1
        )]

    # Create a new DataFrame from the expanded rows
    expanded_df = pd.DataFrame(expanded_rows)
    return expanded_df


def safe_parse_via(x):
    if x == '-99':
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            result = ast.literal_eval(x)
            if isinstance(result, list):
                return [int(n) for n in result]
            else:
                return []
        except (ValueError, SyntaxError):
            return []
    return []


def identify_end_stations(graph, df_network):
    """
    Identifies and marks end stations of existing railway lines in the graph.

    Args:
        graph (networkx.Graph): The railway network graph
        df_network (GeoDataFrame): DataFrame containing railway network information

    Returns:
        set: Set of node IDs that are end stations
    """
    # Convert FromEnd and ToEnd to boolean if they're string values
    if 'FromEnd' in df_network.columns and 'ToEnd' in df_network.columns:
        df_network['FromEnd'] = df_network['FromEnd'].astype(str).map({'1': True, '0': False, 'True': True, 'False': False})
        df_network['ToEnd'] = df_network['ToEnd'].astype(str).map({'1': True, '0': False, 'True': True, 'False': False})

        # Collect all end stations
        end_stations = set()

        # Add nodes marked as FromEnd
        from_end_nodes = df_network.loc[df_network['FromEnd'] == True, 'FromNode'].unique()
        end_stations.update(from_end_nodes)

        # Add nodes marked as ToEnd
        to_end_nodes = df_network.loc[df_network['ToEnd'] == True, 'ToNode'].unique()
        end_stations.update(to_end_nodes)

        # Mark end stations in the graph
        for node_id in end_stations:
            if node_id in graph.nodes:
                graph.nodes[node_id]['end_station'] = True
                station_name = graph.nodes[node_id].get('station_name', f"Unknown Station {node_id}")
                print(f"Marked end station: {station_name} (ID: {node_id})")

        # Mark non-end stations
        for node_id in graph.nodes:
            if node_id not in end_stations:
                graph.nodes[node_id]['end_station'] = False

        return end_stations
    else:
        # If FromEnd/ToEnd not available, use degree-based approach
        end_stations = set()
        for node, degree in graph.degree():
            if degree == 1:  # Nodes with only one connection are likely end stations
                graph.nodes[node]['end_station'] = True
                end_stations.add(node)
                station_name = graph.nodes[node].get('station_name', f"Unknown Station {node}")
                print(f"Identified terminal station by degree: {station_name} (ID: {node})")
            else:
                graph.nodes[node]['end_station'] = False

        return end_stations


def get_node_positions(df_split, df_points, G):
    node_coords = {}
    for _, row in df_split.iterrows():
        # Get the nodes
        from_node = row['FromNode']
        to_node = row['ToNode']

        # Get coordinates from df_points using geometry
        if from_node in df_points['ID_point'].values:
            from_geom = df_points.loc[df_points['ID_point'] == from_node, 'geometry'].iloc[0]
            from_coords = (from_geom.x, from_geom.y)
            if from_coords != (0.0, 0.0):  # Skip if coordinates are (0,0)
                node_coords[from_node] = from_coords

        if to_node in df_points['ID_point'].values:
            to_geom = df_points.loc[df_points['ID_point'] == to_node, 'geometry'].iloc[0]
            to_coords = (to_geom.x, to_geom.y)
            if to_coords != (0.0, 0.0):  # Skip if coordinates are (0,0)
                node_coords[to_node] = to_coords
    # Create position dictionary for networkx, only including nodes with valid coordinates
    pos = {node: node_coords[node] for node in G.nodes() if node in node_coords}
    return pos


def get_missing_connections(G, pos, print_results=False, polygon=None):
    """
    Identifies center nodes and their border nodes, checking for connections with same service.
    Two border points are connected if there is any service which serves both of the border points.
    A connection is missing only if there's no service that serves both border nodes.

    Args:
        G (networkx.Graph): Input graph with 'service' edge attributes and 'station_name' node attributes.
        print_results (bool): Whether to print the results.
        polygon (shapely.geometry.Polygon, optional): Geographic area to filter nodes. Only connections
            where at least one border node is inside the polygon are considered.

    Returns:
        list: List of dictionaries containing center nodes, their borders, and missing connections.
    """
    results = []
    processed_nodes = set()

    service_to_nodes = {}
    for u, v, data in G.edges(data=True):
        service = data.get('service')
        if service:
            service_to_nodes.setdefault(service, set()).update([u, v])

    node_to_services = {}
    for service, nodes in service_to_nodes.items():
        for node in nodes:
            node_to_services.setdefault(node, set()).add(service)

    for node in G.nodes():
        if node_to_services.get(node) and print_results:
            station_name = G.nodes[node].get('station_name', f"Unknown Station {node}")
            print(f"Node {node} ({station_name}) is served by: {node_to_services[node]}")

    for node in G.nodes():
        if node in processed_nodes:
            continue

        neighbors = list(G.neighbors(node))
        if len(neighbors) > 2:
            center_name = G.nodes[node].get('station_name', f"Unknown Station {node}")

            border_names = []
            for border in neighbors:
                border_name = G.nodes[border].get('station_name', f"Unknown Station {border}")
                border_names.append(border_name)

            center_info = {
                'center': node,
                'center_name': center_name,
                'borders': neighbors,
                'border_names': border_names,
                'missing_connections': []
            }

            G.nodes[node]['type'] = 'center'
            for border in neighbors:
                G.nodes[border]['type'] = 'border'

            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    node1, node2 = neighbors[i], neighbors[j]
                    name1 = G.nodes[node1].get('station_name', f"Unknown Station {node1}")
                    name2 = G.nodes[node2].get('station_name', f"Unknown Station {node2}")

                    services1 = node_to_services.get(node1, set())
                    services2 = node_to_services.get(node2, set())
                    common_services = services1.intersection(services2)

                    if not common_services:
                        # Filter by polygon if provided
                        if polygon is not None:
                            pos1 = pos.get(node1)
                            pos2 = pos.get(node2)
                            if pos1 is None and pos2 is None:
                                continue
                            point1_inside = Point(pos1).within(polygon) if pos1 else False
                            point2_inside = Point(pos2).within(polygon) if pos2 else False
                            if not (point1_inside or point2_inside):
                                continue  # skip this connection

                        services = [data['service'] for data in G[node][node1].values()]
                        service1 = services[0] if services else None

                        center_info['missing_connections'].append({
                            'nodes': (node1, node2),
                            'node_names': (name1, name2),
                            'service': service1
                        })

            if center_info['missing_connections']:
                results.append(center_info)

            processed_nodes.add(node)

    if print_results:
        for result in results:
            print(f"\nCenter Node: {result['center']} ({result['center_name']})")
            print("Border Nodes:")
            for i, (node_id, node_name) in enumerate(zip(result['borders'], result['border_names'])):
                print(f"  {i + 1}. ID: {node_id} ({node_name})")

            if result['missing_connections']:
                print("Missing Connections:")
                for conn in result['missing_connections']:
                    print(f"  Between {conn['node_names'][0]} and {conn['node_names'][1]} (Service: {conn['service']})")
                    print(f"  (Node IDs: {conn['nodes'][0]} and {conn['nodes'][1]})")

    return results


def generate_new_railway_lines(G, center_analysis):
    """
    Generates new railway lines starting from missing connections.
    Ensures lines only end at existing end stations of the railway network.

    Args:
        G (networkx.Graph): Input graph with service attributes and end_station node attributes
        center_analysis (list): Results from find_center_and_borders function

    Returns:
        list: List of new railway lines with their paths and names
    """

    def find_path_continuation(current_node, visited_nodes, forbidden_nodes, target_is_end_station=False):
        """
        Recursively finds all possible path continuations from a node.
        If target_is_end_station is True, only returns paths that end at an end station.
        Returns list of possible paths, each path is a list of nodes.
        """
        # Base case - if we reached an end station and that's what we're looking for
        if target_is_end_station and G.nodes[current_node].get('end_station', False):
            return [[current_node]]

        visited_nodes.add(current_node)

        neighbors = list(G.neighbors(current_node))
        valid_neighbors = [n for n in neighbors if n not in visited_nodes and n not in forbidden_nodes]

        paths = []

        # If we're looking for end stations and this node is not an end station or we have neighbors,
        # continue exploration
        if not target_is_end_station or not G.nodes[current_node].get('end_station', False) or valid_neighbors:
            for next_node in valid_neighbors:
                # Recursively find all paths from the next node
                next_paths = find_path_continuation(next_node, visited_nodes.copy(), forbidden_nodes, target_is_end_station)
                # Add current node to the beginning of each path
                for path in next_paths:
                    paths.append([current_node] + path)

        # If we're not specifically looking for end stations, or if this is an end station,
        # include this node as a single-node path
        if not target_is_end_station or G.nodes[current_node].get('end_station', False):
            paths.append([current_node])

        return paths

    new_lines = []
    service_counter = 1

    # Iterate through each center and its missing connections
    for center_info in center_analysis:
        for missing in center_info['missing_connections']:
            node1, node2 = missing['nodes']
            name1, name2 = missing['node_names']
            forbidden_nodes = set(center_info['borders'])  # Don't use any border nodes

            # Check if both missing connection nodes are already end stations
            node1_is_end = G.nodes[node1].get('end_station', False)
            node2_is_end = G.nodes[node2].get('end_station', False)

            print(f"Processing missing connection: {name1} - {name2}")
            print(f"  Node {node1} ({name1}) is end station: {node1_is_end}")
            print(f"  Node {node2} ({name2}) is end station: {node2_is_end}")

            # Start from first node
            # If node1 is not an end station, find paths to end stations
            # Otherwise, just use the node itself
            if node1_is_end:
                paths_from_node1 = [[node1]]
            else:
                paths_from_node1 = find_path_continuation(node1, set(), forbidden_nodes - {node1}, True)
                # Remove paths that don't end at an end station
                paths_from_node1 = [path for path in paths_from_node1
                                   if G.nodes[path[-1]].get('end_station', False)]

            # Start from second node
            # Same logic as for node1
            if node2_is_end:
                paths_from_node2 = [[node2]]
            else:
                paths_from_node2 = find_path_continuation(node2, set(), forbidden_nodes - {node2}, True)
                # Remove paths that don't end at an end station
                paths_from_node2 = [path for path in paths_from_node2
                                   if G.nodes[path[-1]].get('end_station', False)]

            print(f"  Found {len(paths_from_node1)} possible paths from {name1}")
            print(f"  Found {len(paths_from_node2)} possible paths from {name2}")

            # Combine paths from both ends to create complete lines
            valid_lines_created = 0
            for path1 in paths_from_node1:
                for path2 in paths_from_node2:
                    # Check if paths don't overlap (except possibly at endpoints)
                    path1_nodes = set(path1[:-1])  # Exclude last node
                    path2_nodes = set(path2[:-1])  # Exclude last node

                    if not (path1_nodes & path2_nodes):  # Ensure paths don't overlap
                        # Create complete path
                        complete_path = path2[::-1] + path1

                        # Get station names for the path
                        stations = [G.nodes[n].get('station_name', f"Unknown Station {n}")
                                   for n in complete_path]

                        # Create new service line
                        new_line = {
                            'name': f'X{service_counter}',
                            'path': complete_path,
                            'stations': stations,
                            'original_missing_connection': {
                                'nodes': (node1, node2),
                                'stations': (name1, name2)
                            },
                            'endpoints': {
                                'start': {
                                    'node': path2[-1],
                                    'station': G.nodes[path2[-1]].get('station_name', f"Unknown Station {path2[-1]}")
                                },
                                'end': {
                                    'node': path1[-1],
                                    'station': G.nodes[path1[-1]].get('station_name', f"Unknown Station {path1[-1]}")
                                }
                            }
                        }
                        new_lines.append(new_line)
                        service_counter += 1
                        valid_lines_created += 1

            print(f"  Created {valid_lines_created} valid new railway lines for this missing connection")

    return new_lines


def export_new_railway_lines(new_lines, pos, file_path="new_railway_lines.gpkg"):
    """
    Exports the generated railway lines to a GeoPackage file.

    Args:
        new_lines (list): List of new railway line dictionaries
        file_path (str): Path to save the GeoPackage file
    """
    # Create lists to store data
    rows = []

    for line in new_lines:
        # Get path nodes and convert to a LineString geometry
        path_nodes = line['path']

        # Get coordinates for each node in the path
        path_coords = []
        for node_id in path_nodes:
            if node_id in pos:
                path_coords.append(pos[node_id])

        # Skip if we don't have coordinates for all nodes
        if len(path_coords) != len(path_nodes):
            print(f"Warning: Missing coordinates for some nodes in line {line['name']}")
            continue

        # Create a LineString from the coordinates
        if len(path_coords) >= 2:
            line_geom = LineString(path_coords)

            # Create a row for this line
            row = {
                'name': line['name'],
                'start_station': line['endpoints']['start']['station'],
                'end_station': line['endpoints']['end']['station'],
                'start_node': line['endpoints']['start']['node'],
                'end_node': line['endpoints']['end']['node'],
                'missing_connection': f"{line['original_missing_connection']['stations'][0]} - {line['original_missing_connection']['stations'][1]}",
                'station_count': len(line['stations']),
                'stations': ','.join(line['stations']),
                'path': ','.join(str(p) for p in line['path']),
                'geometry': line_geom
            }
            rows.append(row)

    # Create GeoDataFrame if we have any valid rows
    if rows:
        gdf = gpd.GeoDataFrame(rows, crs="epsg:2056")
        gdf.to_file(file_path, driver="GPKG")
        print(f"Successfully exported {len(rows)} new railway lines to {file_path}")
        return gdf
    else:
        print("No valid railway lines to export")
        return None


def prepare_Graph(df_network, df_points):
    # Create an undirected graph
    G = nx.MultiGraph()
    # Split the lines with a Via column
    df_split = split_via_nodes_mod(df_network, delete_via_edges=True)
    df_split['Via'] = df_split['Via'].apply(safe_parse_via)
    # Create a set of (FromNode, ToNode) pairs that should be removed (old connections which skip via station)
    pairs_to_remove = set()
    for _, row in df_split.iterrows():
        via_nodes = row['Via']
        if len(via_nodes) >= 2:
            # Add the pair of first and last Via nodes
            pairs_to_remove.add((via_nodes[0], via_nodes[-1]))
    # Filter out rows where FromNode, ToNode matches any of the pairs to remove
    df_split = df_split[~df_split.apply(lambda row: (row['FromNode'], row['ToNode']) in pairs_to_remove, axis=1)]
    unique_edges = df_split[['FromNode', 'ToNode', 'FromStation', 'ToStation', 'Service']].drop_duplicates()
    # Add nodes first with their station names as attributes
    nodes_with_names = pd.concat([
        unique_edges[['FromNode', 'FromStation']].rename(columns={'FromNode': 'Node', 'FromStation': 'Station'}),
        unique_edges[['ToNode', 'ToStation']].rename(columns={'ToNode': 'Node', 'ToStation': 'Station'})
    ]).drop_duplicates()
    # Add nodes with station names as attributes
    for _, row in nodes_with_names.iterrows():
        G.add_node(row['Node'], station_name=row['Station'])
    # Initialize node_coords dictionary
    node_coords = {}
    # Add edges
    for _, row in unique_edges.iterrows():
        G.add_edge(
            row['FromNode'],
            row['ToNode'],
            service=row['Service'])  # Add Service as edge attribute
    # Mark end stations in the graph
    end_stations = identify_end_stations(G, df_network)
    print(f"Total end stations identified: {len(end_stations)}")
    pos = get_node_positions(df_split, df_points, G)
    # First plot the regular network
    return G, pos

def add_railway_lines_to_new_links(new_railway_lines_path, mod_type, new_links_path, rail_network_path):
    # Load data
    new_railway_lines = gpd.read_file(new_railway_lines_path)
    rail_network = gpd.read_file(paths.get_rail_services_path(rail_network_path))

    # Initialize container for new links
    new_links_list = []
    dev_id = 101000

    for _, row in new_railway_lines.iterrows():
        try:
            path_nodes = list(map(int, row["path"].split(",")))
        except Exception as e:
            print(f"Skipping row due to path parsing error: {e}")
            continue

        full_geom = row["geometry"]
        line_name = row["name"]

        for i in range(len(path_nodes) - 1):
            from_node = path_nodes[i]
            to_node = path_nodes[i + 1]

            # Estimate geometry using interpolation
            segment_geom = LineString([
                full_geom.interpolate(i / (len(path_nodes) - 1), normalized=True),
                full_geom.interpolate((i + 1) / (len(path_nodes) - 1), normalized=True)
            ])

            # Try to find a matching travel time in the existing network
            match = rail_network[
                ((rail_network["FromNode"] == from_node) & (rail_network["ToNode"] == to_node)) |
                ((rail_network["FromNode"] == to_node) & (rail_network["ToNode"] == from_node))
            ]

            if not match.empty and not match["TravelTime"].isna().all():
                travel_time = match["TravelTime"].dropna().astype(float).iloc[0]
            else:
                length_m = segment_geom.length
                travel_time = length_m / 1000 #/60 *60  # speed = 60 km/h, 60 min/h

            new_links_list.append({
                "from_ID_new": from_node,
                "to_ID": to_node,
                "Sline": line_name,
                "dev_id": dev_id,
                "shortest_path_length": segment_geom.length,
                "time": travel_time,
                "geometry": segment_geom
            })

        dev_id += 1

    # Convert to GeoDataFrame
    updated_new_links = gpd.read_file(new_links_path)
    new_links_gdf = gpd.GeoDataFrame(new_links_list, geometry="geometry", crs=updated_new_links.crs)

    # Append and save
    if mod_type == 'ALL':
        #join new railway lines with existing new links
        final_links = pd.concat([updated_new_links, new_links_gdf], ignore_index=True)
    else:
        final_links = new_links_gdf
    final_links.to_file(new_links_path, driver="GPKG")
