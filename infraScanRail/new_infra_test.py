import ast
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import paths
from scoring import *
from scoring import split_via_nodes, merge_lines

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

def plot_graph(graph, positions, highlight_centers=True, highlight_missing=None):
    """
    Plot the railway network graph with optional highlighting of center nodes and missing connections.
    
    Args:
        graph (networkx.Graph): The railway network graph
        positions (dict): Dictionary mapping node IDs to (x,y) coordinates
        highlight_centers (bool): Whether to highlight center nodes
        highlight_missing (list): List of missing connections to highlight
    """
    # Create the plot
    plt.figure(figsize=(20, 16), dpi=300)
    
    # Draw edges
    nx.draw_networkx_edges(graph, positions, edge_color='gray', width=0.5, alpha=0.6)
    
    # Draw nodes with different colors based on type
    node_colors = []
    node_sizes = []
    for node in graph.nodes():
        if graph.nodes[node].get('type') == 'center' and highlight_centers:
            node_colors.append('red')
            node_sizes.append(150)
        elif graph.nodes[node].get('type') == 'border' and highlight_centers:
            node_colors.append('orange')
            node_sizes.append(100)
        elif graph.nodes[node].get('end_station', False):
            node_colors.append('green')
            node_sizes.append(100)
        else:
            node_colors.append('lightblue')
            node_sizes.append(50)
    
    # Draw nodes
    nx.draw_networkx_nodes(graph, positions, node_color=node_colors, node_size=node_sizes)
    
    # Add labels with station names
    labels = nx.get_node_attributes(graph, 'station_name')
    nx.draw_networkx_labels(graph, positions, labels, font_size=8)
    
    # Highlight missing connections if provided
    if highlight_missing:
        for conn in highlight_missing:
            node1, node2 = conn['nodes']
            if node1 in positions and node2 in positions:
                plt.plot([positions[node1][0], positions[node2][0]], 
                         [positions[node1][1], positions[node2][1]], 
                         'r--', linewidth=2, alpha=0.7)
    
    # Add legend
    if highlight_centers:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Center Node'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Border Node'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='End Station'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Regular Node')
        ]
        if highlight_missing:
            legend_elements.append(Line2D([0], [0], color='r', linestyle='--', label='Missing Connection'))
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    plt.title("Railway Network Analysis")
    plt.axis('on')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('network_graph.png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.2,
                format='png')
    #plt.show()






######## Remove the express edges which jump stations

# First, ensure Via is properly parsed into lists
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



# Identify and mark end stations in the graph
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



# Extract node positions from the line geometries
"""for _, row in df_split.iterrows():
    # Get the coordinates of the first and last points of the LineString
    line_coords = list(row['geometry'].coords)

    # First point coordinates for FromNode
    node_coords[row['FromNode']] = line_coords[0]

    # Last point coordinates for ToNode
    node_coords[row['ToNode']] = line_coords[-1]

# Create position dictionary for networkx
pos = {node: node_coords[node] for node in G.nodes()}"""


def get_node_positions(df_split, df_points):
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


def get_missing_connections(G):
    """
    Identifies center nodes and their border nodes, checking for connections with same service.
    Two border points are connected if there is any service which serves both of the border points.
    A connection is missing only if there's no service that serves both border nodes.

    Args:
        G (networkx.Graph): Input graph with 'service' edge attributes and 'station_name' node attributes

    Returns:
        list: List of dictionaries containing center nodes, their borders, and missing connections
    """
    results = []
    processed_nodes = set()

    # Create a service-to-nodes mapping to track which nodes each service passes through
    service_to_nodes = {}

    # For each edge in the graph, record the service and the nodes it connects
    for u, v, data in G.edges(data=True):
        service = data.get('service')
        if service:
            if service not in service_to_nodes:
                service_to_nodes[service] = set()
            service_to_nodes[service].add(u)
            service_to_nodes[service].add(v)

    # Create a node-to-services mapping for easier lookup
    node_to_services = {}
    for service, nodes in service_to_nodes.items():
        for node in nodes:
            if node not in node_to_services:
                node_to_services[node] = set()
            node_to_services[node].add(service)

    # For debugging: print information about services at nodes
    for node in G.nodes():
        if node in node_to_services:
            station_name = G.nodes[node].get('station_name', f"Unknown Station {node}")
            print(f"Node {node} ({station_name}) is served by: {node_to_services[node]}")

    # Process each center node
    for node in G.nodes():
        # Skip if node already processed
        if node in processed_nodes:
            continue

        # Check if node has more than 2 different neighbors
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 2:
            # Get station name for center node
            center_name = G.nodes[node].get('station_name', f"Unknown Station {node}")

            # Get station names for border nodes
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

            # Mark the center node with attribute
            G.nodes[node]['type'] = 'center'

            # Mark border nodes with attribute
            for border in neighbors:
                G.nodes[border]['type'] = 'border'

            # Check connections between border nodes
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    node1, node2 = neighbors[i], neighbors[j]
                    name1 = G.nodes[node1].get('station_name', f"Unknown Station {node1}")
                    name2 = G.nodes[node2].get('station_name', f"Unknown Station {node2}")

                    # Get services for both nodes
                    services1 = node_to_services.get(node1, set())
                    services2 = node_to_services.get(node2, set())

                    # Check if they share any common services
                    common_services = services1.intersection(services2)

                    # If no common services exist, add to missing connections
                    if not common_services:
                        services = [data['service'] for data in G[node][node1].values()]
                        service1 = services[0] if services else None  # Use first service if available

                        center_info['missing_connections'].append({
                            'nodes': (node1, node2),
                            'node_names': (name1, name2),
                            'service': service1})

            # Only add to results if there are actually missing connections
            if center_info['missing_connections']:
                results.append(center_info)

            processed_nodes.add(node)
            #processed_nodes.update(neighbors)

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


def print_new_railway_lines(new_lines):
    """
    Prints the generated railway lines in a readable format.
    Includes information about end stations.
    """
    for line in new_lines:
        print(f"\nNew Railway Line {line['name']}:")
        print(f"Original missing connection: {line['original_missing_connection']['stations'][0]} - "
              f"{line['original_missing_connection']['stations'][1]}")
        
        # Print endpoints information
        if 'endpoints' in line:
            print(f"Terminal stations:")
            print(f"  Start: {line['endpoints']['start']['station']} (ID: {line['endpoints']['start']['node']}) - Terminal Station")
            print(f"  End: {line['endpoints']['end']['station']} (ID: {line['endpoints']['end']['node']}) - Terminal Station")
        
        print("Route:")
        for i, (node_id, station) in enumerate(zip(line['path'], line['stations'])):
            station_type = ""
            if i == 0:
                prefix = "  Start:"
                station_type = "Terminal Station"
            elif i == len(line['path']) - 1:
                prefix = "  End:"
                station_type = "Terminal Station"
            else:
                prefix = "  Via:"
            
            print(f"{prefix} {station} (ID: {node_id}) {station_type}")
        
        print(f"Total stations: {len(line['path'])}")


def export_new_railway_lines(new_lines, file_path="new_railway_lines.gpkg"):
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
            if node_id in node_coords:
                path_coords.append(node_coords[node_id])
        
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
                'geometry': line_geom
            }
            rows.append(row)
    
    # Create GeoDataFrame if we have any valid rows
    if rows:
        gdf = gpd.GeoDataFrame(rows, crs="epsg:2056")
        gdf.to_file(file_path, driver="GPKG")
        print(f"Successfully exported {len(rows)} new railway lines to {file_path}")
    else:
        print("No valid railway lines to export")


df_network = gpd.read_file(paths.RAIL_SERVICES_AK2035_PATH)
df_points = gpd.read_file(r'data\Network\processed\points.gpkg')

# Create an undirected graph
G = nx.MultiGraph()


# Split the lines with a Via column
df_split = split_via_nodes_mod(df_network,delete_via_edges=True)

df_split['Via'] = df_split['Via'].apply(safe_parse_via)

# Create a set of (FromNode, ToNode) pairs that should be removed
pairs_to_remove = set()
for _, row in df_split.iterrows():
    via_nodes = row['Via']
    if len(via_nodes) >= 2:
        # Add the pair of first and last Via nodes
        pairs_to_remove.add((via_nodes[0], via_nodes[-1]))

# Filter out rows where FromNode, ToNode matches any of the pairs to remove
df_split = df_split[~df_split.apply(lambda row: (row['FromNode'], row['ToNode']) in pairs_to_remove, axis=1)]

unique_edges = df_split[['FromNode', 'ToNode', 'FromStation', 'ToStation','Service']].drop_duplicates()

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
        service=row['Service'] ) # Add Service as edge attribute

# Mark end stations in the graph
end_stations = identify_end_stations(G, df_network)
print(f"Total end stations identified: {len(end_stations)}")

pos = get_node_positions(df_split, df_points)
# First plot the regular network
plot_graph(G, pos)

# After analysis, plot again with highlights
center_analysis = get_missing_connections(G)

# Print results
for result in center_analysis:
    print(f"\nCenter Node: {result['center']} ({result['center_name']})")
    print("Border Nodes:")
    for i, (node_id, node_name) in enumerate(zip(result['borders'], result['border_names'])):
        print(f"  {i + 1}. ID: {node_id} ({node_name})")

    if result['missing_connections']:
        print("Missing Connections:")
        for conn in result['missing_connections']:
            print(f"  Between {conn['node_names'][0]} and {conn['node_names'][1]} (Service: {conn['service']})")
            print(f"  (Node IDs: {conn['nodes'][0]} and {conn['nodes'][1]})")

missing_connections = []
for center in center_analysis:
    missing_connections.extend(center['missing_connections'])

# Plot with highlights for centers and missing connections
plot_graph(G, pos, highlight_centers=True, highlight_missing=missing_connections)


# Analyze the railway network to find missing connections
print("\n=== RAILWAY NETWORK ANALYSIS ===")
print("Identifying center nodes and missing connections...")
center_analysis = get_missing_connections(G)

# Generate potential new railway lines
print("\n=== GENERATING NEW RAILWAY LINES ===")
new_railway_lines = generate_new_railway_lines(G, center_analysis)

# Print detailed information about the new lines
print("\n=== NEW RAILWAY LINES DETAILS ===")
print_new_railway_lines(new_railway_lines)

# Export to GeoPackage for further analysis and visualization in GIS software
if node_coords:
    export_new_railway_lines(new_railway_lines, "data/Network/processed/new_railway_lines.gpkg")
    print("\nNew railway lines exported to data/Network/processed/new_railway_lines.gpkg")
else:
    print("\nWARNING: Could not export railway lines because node coordinates are missing")

# Visualize the new railway lines on the network graph
print("\n=== VISUALIZATION ===")
print("Creating visualization of the network with highlighted missing connections...")

# Create a new figure to visualize the proposed new railway lines
plt.figure(figsize=(20, 16), dpi=300)

# Draw the existing network
nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.5, alpha=0.4)
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=30, alpha=0.4)

# Draw the new railway lines with different colors
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
for i, line in enumerate(new_railway_lines):
    path = line['path']
    color = colors[i % len(colors)]
    label = line['name']
    
    # Draw the path segments
    for j in range(len(path) - 1):
        if path[j] in pos and path[j+1] in pos:
            plt.plot([pos[path[j]][0], pos[path[j+1]][0]], 
                     [pos[path[j]][1], pos[path[j+1]][1]], 
                     color=color, linewidth=2)
    
    # Mark the first occurrence to generate the legend entry
    if path[0] in pos:
        plt.plot(pos[path[0]][0], pos[path[0]][1], 'o', color=color, markersize=8, label=label)

# Add legend and title
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
plt.title("Proposed New Railway Lines")
plt.grid(True)
plt.tight_layout()
plt.savefig('proposed_railway_lines.png',
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.2,
            format='png')
#plt.show()

print("\nAnalysis and visualization complete!")

# Export the new railway lines to a GeoPackage file if coordinates are available
if node_coords:
    export_new_railway_lines(new_railway_lines, "data/Network/processed/new_railway_lines.gpkg")