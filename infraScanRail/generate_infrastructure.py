import pandas as pd
from scipy.stats.qmc import LatinHypercube
from shapely.ops import nearest_points
import fiona
from shapely.geometry import MultiLineString
from shapely.ops import split
import gc

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


def nearest(row, geom_union, df2, geom1_col='geometry', geom2_col='geometry', src_column=None):
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


