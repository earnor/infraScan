from shapely.geometry import MultiLineString
from shapely.ops import split
import gc

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

def plot_graph(graph, positions):
    # Create the plot
    plt.figure(figsize=(20, 16), dpi=300)
    # Draw edges
    nx.draw_networkx_edges(graph, positions, edge_color='gray', width=0.5)
    # Draw nodes
    nx.draw_networkx_nodes(graph, positions, node_color='lightblue', node_size=50)
    # Add labels with station names
    labels = nx.get_node_attributes(graph, 'station_name')
    nx.draw_networkx_labels(graph, positions, labels, font_size=8)
    plt.title("Railway Network")
    plt.axis('on')
    plt.grid(True)
    plt.savefig('network_graph.png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.2,
                format='png')
    plt.show()




df_network = gpd.read_file(paths.RAIL_SERVICES_AK2035_PATH)
df_points = gpd.read_file(r'data\Network\processed\points.gpkg')

# Create an undirected graph
G = nx.Graph()

# Split the lines with a Via column
df_split = split_via_nodes_mod(df_network,delete_via_edges=True)

######## Remove the express edges which jump stations

# First, ensure Via is properly parsed into lists
df_split['Via'] = df_split['Via'].apply(lambda x: [] if x == '-99' else [int(n) for n in ast.literal_eval(x)] if isinstance(x, str) else [])

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

# Add edges
for _, row in unique_edges.iterrows():
    G.add_edge(
        row['FromNode'],
        row['ToNode'],
        service=row['Service'] ) # Add Service as edge attribute

    node_coords = {}

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

pos = get_node_positions(df_split, df_points)
plot_graph(G,pos)


def find_center_and_borders(G):
    """
    Identifies center nodes and their border nodes, checking for connections with same service.
    Includes station names in the results for better readability, while using node IDs for operations.

    Args:
        G (networkx.Graph): Input graph with 'service' edge attributes and 'station_name' node attributes

    Returns:
        list: List of dictionaries containing center nodes, their borders, and missing connections
    """
    results = []
    processed_nodes = set()

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

                    # Get the service type between center and both border nodes
                    service1 = G[node][node1]['service']
                    service2 = G[node][node2]['service']

                    # Check if there's a path between border nodes with same service
                    path_exists = False
                    if nx.has_path(G, node1, node2):
                        paths = nx.all_simple_paths(G, node1, node2)
                        for path in paths:
                            # Check if all edges in path have the same service
                            services_in_path = set()
                            for k in range(len(path) - 1):
                                services_in_path.add(G[path[k]][path[k + 1]]['service'])

                            # If there's a path with all same service type
                            if len(services_in_path) == 1:
                                path_exists = True
                                break

                    # If no valid path exists, add to missing connections
                    if not path_exists:
                        center_info['missing_connections'].append({
                            'nodes': (node1, node2),
                            'node_names': (name1, name2),
                            'service': service1  # Using service of first border node
                        })

            results.append(center_info)
            processed_nodes.add(node)
            processed_nodes.update(neighbors)

    return results


# Example usage:
center_analysis = find_center_and_borders(G)

# Print results
for result in center_analysis:
    print(f"\nCenter Node: {result['center']} ({result['center_name']})")
    print("Border Nodes:")
    for i, (node_id, node_name) in enumerate(zip(result['borders'], result['border_names'])):
        print(f"  {i+1}. ID: {node_id} ({node_name})")
        
    if result['missing_connections']:
        print("Missing Connections:")
        for conn in result['missing_connections']:
            print(f"  Between {conn['node_names'][0]} and {conn['node_names'][1]} (Service: {conn['service']})")
            print(f"  (Node IDs: {conn['nodes'][0]} and {conn['nodes'][1]})")


def generate_new_railway_lines(G, center_analysis):
    """
    Generates new railway lines starting from missing connections.

    Args:
        G (networkx.Graph): Input graph with service attributes
        center_analysis (list): Results from find_center_and_borders function

    Returns:
        list: List of new railway lines with their paths and names
    """

    def find_path_continuation(current_node, visited_nodes, forbidden_nodes):
        """
        Recursively finds all possible path continuations from a node.
        Returns list of possible paths, each path is a list of nodes.
        """
        paths = [[current_node]]
        visited_nodes.add(current_node)

        neighbors = list(G.neighbors(current_node))
        valid_neighbors = [n for n in neighbors if n not in visited_nodes and n not in forbidden_nodes]

        for next_node in valid_neighbors:
            # Recursively find all paths from the next node
            next_paths = find_path_continuation(next_node, visited_nodes.copy(), forbidden_nodes)
            # Add current node to the beginning of each path
            for path in next_paths:
                paths.append([current_node] + path)

        return paths

    new_lines = []
    service_counter = 1

    # Iterate through each center and its missing connections
    for center_info in center_analysis:
        for missing in center_info['missing_connections']:
            node1, node2 = missing['nodes']
            name1, name2 = missing['node_names']
            forbidden_nodes = set(center_info['borders'])  # Don't use any border nodes

            # Start from first node
            paths_from_node1 = find_path_continuation(node1, set(), forbidden_nodes - {node1})

            # Start from second node
            paths_from_node2 = find_path_continuation(node2, set(), forbidden_nodes - {node2})

            # Combine paths from both ends to create complete lines
            for path1 in paths_from_node1:
                for path2 in paths_from_node2:
                    if not (set(path1) & set(path2)):  # Ensure paths don't overlap
                        # Create new service line
                        new_line = {
                            'name': f'X{service_counter}',
                            'path': path2[::-1] + path1,  # Reverse path2 and combine
                            'stations': [G.nodes[n].get('station_name', f"Unknown Station {n}")
                                         for n in (path2[::-1] + path1)],
                            'original_missing_connection': {
                                'nodes': (node1, node2),
                                'stations': (name1, name2)
                            }
                        }
                        new_lines.append(new_line)
                        service_counter += 1

    return new_lines


def print_new_railway_lines(new_lines):
    """
    Prints the generated railway lines in a readable format.
    """
    for line in new_lines:
        print(f"\nNew Railway Line {line['name']}:")
        print(f"Original missing connection: {line['original_missing_connection']['stations'][0]} - "
              f"{line['original_missing_connection']['stations'][1]}")
        print("Route:")
        for i, (node_id, station) in enumerate(zip(line['path'], line['stations'])):
            if i == 0:
                print(f"  Start: {station} (ID: {node_id})")
            elif i == len(line['path']) - 1:
                print(f"  End: {station} (ID: {node_id})")
            else:
                print(f"  Via: {station} (ID: {node_id})")
        print(f"Total stations: {len(line['path'])}")


# Usage example:
center_analysis = find_center_and_borders(G)
new_railway_lines = generate_new_railway_lines(G, center_analysis)
print_new_railway_lines(new_railway_lines)