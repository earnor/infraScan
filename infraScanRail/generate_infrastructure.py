from shapely.geometry import MultiLineString
from shapely.ops import split
import gc

from scoring import *


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
    if settings.rail_network == 'current':
        raw_edges = gpd.read_file(r"data/temp/network_railway-services.gpkg")
    elif settings.rail_network == 'AK_2035':
        raw_edges = gpd.read_file(paths.RAIL_SERVICES_AK2035_PATH)
    else:
        exit("No rail network specified.")
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

