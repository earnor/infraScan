import pandas as pd
from tqdm import tqdm  # Stellen Sie sicher, dass tqdm importiert wird
from joblib import Parallel, delayed
# from scipy.spatial import cKDTree
# Additional imports for grid creation
from data_import import *
from random_scenarios import load_scenarios_from_cache
import numba
import cost_parameters as cp


def create_directed_graph(df, change_time):
    G = nx.DiGraph()

    stations = set(df['FromStation']).union(set(df['ToStation']))

    # Add entry and exit nodes for each station with node_id
    for station in stations:
        # Finde die node_id für diese Station
        node_id = None
        station_rows_from = df[df['FromStation'] == station]
        station_rows_to = df[df['ToStation'] == station]
        node_id = station_rows_from.iloc[0]['FromNode']

        G.add_node(f"entry_{station}", type="entry_node", station=station, node_id=node_id)
        G.add_node(f"exit_{station}", type="exit_node", station=station, node_id=node_id)

    # Add sub-nodes and S-Bahn edges
    for idx, row in df.iterrows():
        from_sub = f"sub_{row['FromStation']}_{row['Service']}_{row['Direction']}"
        to_sub = f"sub_{row['ToStation']}_{row['Service']}_{row['Direction']}"

        # Add sub-nodes with attributes
        G.add_node(from_sub, type="sub_node", station=row['FromStation'], service=row['Service'],
                   direction=row['Direction'])
        G.add_node(to_sub, type="sub_node", station=row['ToStation'], service=row['Service'],
                   direction=row['Direction'])

        # S-Bahn connection
        if pd.notna(row['TravelTime']):
            weight = int(round(row['TravelTime']))
            G.add_edge(from_sub, to_sub, weight=weight)

    # Add boarding and alighting edges
    for node, data in G.nodes(data=True):
        if data['type'] == 'sub_node':
            station = data['station']
            entry_node = f"entry_{station}"
            exit_node = f"exit_{station}"

            # Boarding: entry → sub_node
            G.add_edge(entry_node, node, weight=3)
            # Alighting: sub_node → exit
            G.add_edge(node, exit_node, weight=3)

    # Add intra-station change edges between sub-nodes
    for station in stations:
        station_sub_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'sub_node' and d['station'] == station]
        for i, sub1 in enumerate(station_sub_nodes):
            for j, sub2 in enumerate(station_sub_nodes):
                if i != j:
                    G.add_edge(sub1, sub2, weight=change_time)

    return G


def create_graphs_from_directories(directories, n_jobs=-1):
    """
    Create a list of directed graphs from a list of file directories using parallel processing.

    Parameters:
        directories (list): List of file paths to GeoPackage or CSV files.
        n_jobs (int): Number of parallel jobs. Default -1 uses all available cores.

    Returns:
        list: A list of NetworkX directed graphs.
    """

    def process_file(directory):
        try:
            print(f"Processing: {directory}...")
            # Read the file into a DataFrame
            if directory.endswith('.gpkg'):
                df = gpd.read_file(directory)
            elif directory.endswith('.csv'):
                df = pd.read_csv(directory)
            else:
                print(f"Unsupported file format: {directory}")
                return None

            # Convert TravelTime to integers for consistent weight usage
            if "TravelTime" in df.columns:
                df["TravelTime"] = df["TravelTime"].round().astype(int)

            # Create the graph
            graph = create_directed_graph(df, cp.comfort_weighted_change_time)
            print(f"Graph created for {directory}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
            return graph
        except Exception as e:
            print(f"Error processing file {directory}: {e}")
            return None

    print(f"Processing {len(directories)} files in parallel with {n_jobs} jobs...")
    results = Parallel(n_jobs=n_jobs)(delayed(process_file)(directory) for directory in directories)

    # Filter out None values (failed processing)
    graphs = [graph for graph in results if graph is not None]
    print(f"Successfully created {len(graphs)} graphs out of {len(directories)} files.")

    return graphs


# Function to calculate the fastest travel time between two main nodes
def calculate_fastest_travel_time(graph, origin, destination):
    """
    Calculate the fastest travel time between two stations using entry and exit nodes.

    Parameters:
        graph (networkx.DiGraph): The graph representing the railway network.
        origin (str): The name of the origin station.
        destination (str): The name of the destination station.

    Returns:
        tuple: Shortest path and total travel time.
    """
    source_node = f"entry_{origin}"
    target_node = f"exit_{destination}"

    # Check if the path exists and calculate shortest path
    if nx.has_path(graph, source_node, target_node):
        shortest_path = nx.shortest_path(graph, source=source_node, target=target_node, weight='weight')
        total_weight = nx.shortest_path_length(graph, source=source_node, target=target_node, weight='weight')
        print(f"Fastest path from {origin} to {destination}: {shortest_path}")
        print(f"Total travel time: {total_weight} minutes")
        return shortest_path, total_weight
    else:
        print(f"No path exists between {origin} and {destination}.")
        return None, None


# Wrapper function to input origin and destination
def find_fastest_path(graph, origin, destination):
    """
    Wrapper function to compute the fastest path between two stations.
    
    Parameters:
        origin (str): The origin station name.
        destination (str): The destination station name.
    """
    calculate_fastest_travel_time(graph, origin, destination)


def calculate_od_pairs_with_times_by_graph(graphs):
    """
    Create all OD pairs for stations across multiple graphs and calculate travel times using optimized methods.
    Returns a list of DataFrames, one for each graph_id.

    Parameters:
        graphs (list): List of NetworkX directed graphs representing railway networks.

    Returns:
        list: A list of Pandas DataFrames, one for each graph_id.
    """
    graph_dataframes = []

    for graph_id, graph in enumerate(tqdm(graphs, desc="Processing developments")):
        print(f"→ Calculating OD pairs for development {graph_id}...")
        od_data = []

        # Extrahiere die Stationen aus den entry/exit Nodes
        entry_nodes = [node for node, data in graph.nodes(data=True) if data.get("type") == "entry_node"]
        exit_nodes = [node for node, data in graph.nodes(data=True) if data.get("type") == "exit_node"]

        # Erstelle Zuordnung von Stationsnamen zu entry/exit nodes
        station_to_entry = {}
        station_to_exit = {}

        for node in entry_nodes:
            station = graph.nodes[node]["station"]
            station_to_entry[station] = node

        for node in exit_nodes:
            station = graph.nodes[node]["station"]
            station_to_exit[station] = node

        # Liste aller Stationen erstellen
        stations = list(station_to_entry.keys())

        # Use all_pairs_dijkstra_path_length für optimierte Distanzberechnung
        all_lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="weight"))

        for origin_station in stations:
            origin_entry = station_to_entry[origin_station]
            origin_lengths = all_lengths.get(origin_entry, {})

            for destination_station in stations:
                if origin_station != destination_station:
                    destination_exit = station_to_exit[destination_station]
                    travel_time = origin_lengths.get(destination_exit, None)

                    od_data.append({
                        "from_id": origin_entry,
                        "to_id": destination_exit,
                        "time": travel_time,
                        "from_station": origin_station,
                        "to_station": destination_station
                    })

        # Konvertiere die OD-Daten für diesen Graphen in ein DataFrame
        od_df = pd.DataFrame(od_data)
        od_df["graph_id"] = graph_id  # Graph-ID als Spalte hinzufügen
        graph_dataframes.append(od_df)  # DataFrame zur Liste hinzufügen

    return graph_dataframes


@numba.njit
def compute_weighted_times(from_idx, to_idx, times, OD_matrix):
    weighted_sum = 0.0
    for i in range(len(from_idx)):
        f = from_idx[i]
        t = to_idx[i]
        if f >= 0 and t >= 0:
            trips = OD_matrix[f, t]
            weighted_sum += trips * times[i] / 60.0
    return weighted_sum


def preprocess_OD_matrix(OD, id_to_name):
    # Convert index and columns from ID to station names
    OD.index = OD.index.astype(int)
    OD.columns = OD.columns.astype(float).astype(int)
    OD = OD.rename(index=id_to_name, columns=id_to_name)

    # Keep only stations that are in both index and columns
    common_stations = list(set(OD.index) & set(OD.columns))
    OD = OD.loc[common_stations, common_stations]

    # Convert to NumPy array and build index mapping
    OD_matrix = OD.values
    station_to_index = {name: idx for idx, name in enumerate(OD.index)}
    return OD_matrix, station_to_index


def process_scenario_year_numba(OD_matrix, station_to_index, preprocessed_od_times_list, scenario_name, year):
    dev_total_times = {}
    for dev_name, from_names, to_names, times in preprocessed_od_times_list:
        from_idx = np.array([station_to_index.get(name, -1) for name in from_names])
        to_idx = np.array([station_to_index.get(name, -1) for name in to_names])
        total_time = compute_weighted_times(from_idx, to_idx, times, OD_matrix)
        dev_total_times[dev_name] = total_time
    return (scenario_name, year, dev_total_times)


def calculate_total_travel_times(od_times_list, scenario_ODs_dir, df_access):
    scenario_ods = load_scenarios_from_cache(scenario_ODs_dir)
    id_to_name = df_access.set_index("NR")["NAME"].to_dict()

    # Preprocess OD time data once
    preprocessed_od_times_list = []
    for idx, od_df in enumerate(od_times_list):
        dev_name = f"Development_{idx + 1}"
        from_names = od_df["from_station"].values
        to_names = od_df["to_station"].values
        times = od_df["time"].values.astype(np.float64)
        preprocessed_od_times_list.append((dev_name, from_names, to_names, times))

    # Prepare parallel tasks: preprocess OD matrix and pass it
    tasks = []
    for scenario_name, OD_dict in scenario_ods.items():
        for year, OD in OD_dict.items():
            OD_matrix, station_to_index = preprocess_OD_matrix(OD, id_to_name)
            tasks.append((OD_matrix, station_to_index, preprocessed_od_times_list, scenario_name, year))

    # Parallel execution
    results = Parallel(n_jobs=-1)(
        delayed(process_scenario_year_numba)(OD_matrix, station_to_index, preprocessed_od_times_list, scenario_name, year)
        for OD_matrix, station_to_index, preprocessed_od_times_list, scenario_name, year in tqdm(tasks, desc="Parallel Processing")
    )

    # Organize results into final dictionary
    total_travel_times = {}
    for scenario_name, year, dev_total_times in results:
        if scenario_name not in total_travel_times:
            total_travel_times[scenario_name] = {}
        total_travel_times[scenario_name][year] = dev_total_times

    return total_travel_times

def calculate_monetized_tt_savings(TTT_status_quo, TTT_developments, VTTS, output_path, dev_id_lookup_table):
    """
    Calculate and monetize travel time savings for each development scenario compared to the status quo,
    scaling peak hour data to daily trips using a fixed tau value.

    Parameters:
        TTT_status_quo (dict): Dictionary of total travel times for the status quo.
        TTT_developments (dict): Dictionary of total travel times for each development scenario.
        VTTS (float): Value of Travel Time Savings (CHF/h).
        duration (float): Duration factor (e.g., years).
        output_path (str): Path to save the monetized travel time savings CSV.

    Returns:
        pd.DataFrame: DataFrame containing monetized travel time savings for each development and scenario.
    """
  

    # Define tau (fraction of trips occurring in the peak hour)
    tau = 0.13  # Assumes 13% of daily trips occur in the peak hour

    # Monetization factor of travel time (CHF/h * 365 d/a * duration)
    mon_factor = VTTS * 365

    # Prepare a list to store the results
    results = []

    # Iterate over each development
    for scenario_name, development in tqdm(TTT_developments.items(), desc="Saving travel time savings"):
        for year, year_tt in development.items():
            for dev_id, dev_tt in year_tt.items():
                # Get the corresponding status quo travel time
                status_quo_tt = TTT_status_quo.get(scenario_name, {}).get(year, {}).get('Development_1', 0)

                # Calculate travel time savings (negative if no savings), scaled to daily trips
                tt_savings_daily = (status_quo_tt - dev_tt) #again scaling with tau?
                monetized_savings_yearly = tt_savings_daily * 365 * VTTS
                # Monetize the travel time savings
                dev_id_lookup = dev_id_lookup_table.loc[int(dev_id.removeprefix("Development_")), "dev_id"]
                # Append the results
                results.append({
                    "development": dev_id_lookup,
                    "scenario": scenario_name,
                    "year": year,
                    "status_quo_tt": status_quo_tt,
                    "development_tt": dev_tt,
                    "tt_savings_daily": tt_savings_daily,
                    "monetized_savings_yearly": monetized_savings_yearly
                })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    scenario_list = sorted(results_df["scenario"].unique().tolist())
    dev_list = sorted(results_df["development"].unique().tolist())

    # Save the results to CSV
    results_df.to_csv(output_path, index=False)
    print(f"Monetized travel time savings saved to: {output_path}")

    return results_df, scenario_list, dev_list


def analyze_travel_times(od_times_status_quo, od_times_dev, od_nodes, dev_id_lookup_table):
    """
    Analyze travel times for the status quo and selected developments.

    Parameters:
    - od_times_status_quo: list of DataFrames, first element contains status quo data
    - od_times_dev: list of DataFrames, contains development data
    - selected_indices: list of int, indices of developments to analyze
    - od_nodes: list of str, OD nodes to consider

    Saves:
    - Individual CSVs for each development sorted by delta time.
    - Top 20 OD pairs for each development.
    """

    # Define file paths
    savings_path = "data/Network/travel_time/TravelTime_Savings"
    report_path = os.path.join(savings_path, "for_report")

    # Ensure directories exist
    os.makedirs(savings_path, exist_ok=True)
    os.makedirs(report_path, exist_ok=True)

    # Extract the status quo DataFrame
    status_quo_df = od_times_status_quo[0]
    selected_indices = [dev_id_lookup_table.loc[i+1, "dev_id"] for i in range(len(od_times_dev))]  # Exclude the first element (status quo)
    # Filter the required developments
    selected_developments = [od_times_dev[i] for i in range(len(od_times_dev))]

    # Generate OD pairs using the provided nodes
    od_pairs = [(origin, destination) for origin in od_nodes for destination in od_nodes if origin != destination]

    # Function to extract travel times for specified OD pairs
    # Function to extract travel times for specified OD pairs
    def extract_travel_times(od_matrix, od_pairs):
        extracted_data = []
        for origin, destination in od_pairs:
            filtered_data = od_matrix[(od_matrix['from_station'] == origin) &
                                      (od_matrix['to_station'] == destination)]
            if not filtered_data.empty:
                extracted_data.append({
                    "origin": origin,
                    "destination": destination,
                    "time": filtered_data.iloc[0]['time'],
                    "from_id": filtered_data.iloc[0]['from_id'],
                    "to_id": filtered_data.iloc[0]['to_id']
                })
        return pd.DataFrame(extracted_data)

    # Extract travel times for the status quo
    status_quo_times = extract_travel_times(status_quo_df, od_pairs)
    status_quo_times = status_quo_times.rename(columns={"time": "status_quo_time"})

    # Process each selected development
    for i, dev_data in enumerate(selected_developments):
        # Extract travel times for the current development
        dev_times = extract_travel_times(dev_data, od_pairs)
        dev_times = dev_times.rename(columns={"time": "new_time"})

        # Merge with status quo times
        merged = pd.merge(status_quo_times, dev_times, on=["origin", "destination"], how="left")

        # Calculate delta time
        merged["delta_time"] = merged["new_time"] - merged["status_quo_time"]

        # Sort by delta time (descending)
        merged_sorted = merged.sort_values(by="delta_time", ascending=True)

        # Save the full CSV for this development
        dev_file = os.path.join(savings_path, f"TravelTime_Savings_Dev_{int(float(selected_indices[i]))}.csv")
        merged_sorted.to_csv(dev_file, index=False)

        # Extract top 20 OD pairs by delta time
        top_20 = merged_sorted.head(20)

        # Save the top 20 OD pairs to a separate file
        top_20_file = os.path.join(report_path, f"TravelTime_Savings_Dev_{int(float(selected_indices[i]))}_Top20.csv")
        top_20.to_csv(top_20_file, index=False)

    return "Analysis completed and files saved."


def calculate_flow_on_edges(graph, OD_matrix, points):
    """
    Berechnet die Anzahl der Personen auf jeder Kante eines Bahn-Graphen basierend auf einer OD-Matrix.
    Verwendet node_id anstatt Stationsnamen und erstellt zwei Graphen:
    1. Pro Station nur einen allgemeinen Knoten (stationsbasiert)
    2. Pro Bahnlinie separate Kanten (linienbasiert)

    Parameters:
        graph (nx.DiGraph): NetworkX-Graph des Bahn-Netzwerks
        OD_matrix (pd.DataFrame): OD-Matrix mit Personen zwischen Stationspaaren
        points (pd.DataFrame): DataFrame mit Geometriedaten der Stationen

    Returns:
        tuple: (station_flow_graph, line_flow_graph) - Zwei NetworkX-DiGraph-Objekte:
               1. Graph mit Stationsknoten und aggregierten Flüssen
               2. Graph mit linienspezifischen Kanten und Flüssen
    """
    # Erstelle zwei neue Graphen für die Flüsse
    station_flow_graph = nx.DiGraph()
    line_flow_graph = nx.DiGraph()
    OD_matrix = OD_matrix.set_index('from_station')

    # Extrahiere die Stationen aus den entry/exit Nodes
    entry_nodes = [node for node, data in graph.nodes(data=True) if data.get("type") == "entry_node"]
    exit_nodes = [node for node, data in graph.nodes(data=True) if data.get("type") == "exit_node"]

    # Erstelle Zuordnungen
    station_to_entry = {}
    station_to_exit = {}
    node_id_to_entry = {}
    node_id_to_exit = {}
    node_id_to_station = {}  # Zuordnung von node_id zu Stationsname

    for node in entry_nodes:
        station = graph.nodes[node]["station"]
        node_id = graph.nodes[node]["node_id"]
        station_to_entry[station] = node
        node_id_to_entry[node_id] = node
        node_id_to_station[node_id] = station

    for node in exit_nodes:
        station = graph.nodes[node]["station"]
        node_id = graph.nodes[node]["node_id"]
        station_to_exit[station] = node
        node_id_to_exit[node_id] = node
        node_id_to_station[node_id] = station

    # Liste aller Knoten-IDs erstellen
    node_ids = list(node_id_to_entry.keys())

    # Dictionaries zur Speicherung der Kanten-Flüsse
    station_edge_flows = {}  # Für stationsbasierte Flüsse
    line_edge_flows = {}  # Für linienbasierte Flüsse

    # Iteration über alle OD-Paare in der OD-Matrix
    for origin_id in tqdm(node_ids, desc="Verarbeite Stationen"):
        for dest_id in node_ids:
            if origin_id != dest_id:
                if origin_id in OD_matrix.index and str(dest_id) in OD_matrix.columns:
                    flow = OD_matrix.loc[origin_id, str(dest_id)]

                    if flow > 0:
                        # Ermittle den kürzesten Pfad für dieses OD-Paar
                        path = nx.shortest_path(graph,
                                                source=node_id_to_entry[origin_id],
                                                target=node_id_to_exit[dest_id],
                                                weight='weight')

                        # Extrahiere nur die S-Bahn-Kanten
                        for i in range(len(path) - 1):
                            if path[i].startswith('sub_') and path[i + 1].startswith('sub_'):
                                # Extrahiere Stationsnamen aus den sub_nodes
                                # Format: "sub_StationName_Service_Direction"
                                source_parts = path[i].split('_')
                                target_parts = path[i + 1].split('_')

                                source_station = source_parts[1]
                                target_station = target_parts[1]

                                # Für linienbasierten Graphen: Extrahiere Service (z.B. S5) und Richtung
                                service = source_parts[2]  # z.B. "S5"
                                direction = source_parts[3]  # Richtung

                                # Aktualisiere stationsbasierte Flüsse
                                station_edge = (source_station, target_station)
                                if station_edge in station_edge_flows:
                                    station_edge_flows[station_edge] += flow
                                else:
                                    station_edge_flows[station_edge] = flow

                                # Aktualisiere linienbasierte Flüsse mit Service und Richtung
                                line_edge = (source_station, target_station, service, direction)
                                if line_edge in line_edge_flows:
                                    line_edge_flows[line_edge] += flow
                                else:
                                    line_edge_flows[line_edge] = flow

    # Füge die Kanten mit Flüssen zum stationsbasierten Graphen hinzu
    for (source_station, target_station), flow in station_edge_flows.items():
        # Finde node_ids für Quell- und Zielstation
        source_node_id = None
        target_node_id = None

        for node_id, station in node_id_to_station.items():
            if station == source_station:
                source_node_id = node_id
            if station == target_station:
                target_node_id = node_id

            if source_node_id and target_node_id:
                break

        # Füge die Knoten hinzu, wenn node_ids gefunden wurden
        if source_node_id and not station_flow_graph.has_node(source_station):
            # Füge Knoten zum stationsbasierten Graphen hinzu
            point_row = points[points['ID_point'] == source_node_id]
            if not point_row.empty:
                geometry = point_row.iloc[0]['geometry']
                position = (geometry.x, geometry.y)
                station_flow_graph.add_node(source_station, position=position)

        if target_node_id and not station_flow_graph.has_node(target_station):
            # Füge Knoten zum stationsbasierten Graphen hinzu
            point_row = points[points['ID_point'] == target_node_id]
            if not point_row.empty:
                geometry = point_row.iloc[0]['geometry']
                position = (geometry.x, geometry.y)
                station_flow_graph.add_node(target_station, position=position)

        # Füge die Kante mit Fluss zum stationsbasierten Graphen hinzu
        if station_flow_graph.has_node(source_station) and station_flow_graph.has_node(target_station):
            station_flow_graph.add_edge(source_station, target_station, flow=flow)

    # Füge die Knoten und Kanten zum linienbasierten Graphen hinzu
    for (source_station, target_station, service, direction), flow in line_edge_flows.items():
        # Finde node_ids für Quell- und Zielstation
        source_node_id = None
        target_node_id = None

        for node_id, station in node_id_to_station.items():
            if station == source_station:
                source_node_id = node_id
            if station == target_station:
                target_node_id = node_id

            if source_node_id and target_node_id:
                break

        # Knoten im linienbasierten Graphen anlegen
        for station, node_id in [(source_station, source_node_id), (target_station, target_node_id)]:
            if node_id and not line_flow_graph.has_node(station):
                point_row = points[points['ID_point'] == node_id]
                if not point_row.empty:
                    geometry = point_row.iloc[0]['geometry']
                    position = (geometry.x, geometry.y)
                    line_flow_graph.add_node(station, position=position)

        # Füge die Kante zum linienbasierten Graphen hinzu
        if (line_flow_graph.has_node(source_station) and
                line_flow_graph.has_node(target_station)):
            # Erstelle eine eindeutige Kanten-ID für die Linie
            edge_key = f"{service}_{direction}"

            # Füge die Kante mit Fluss und Liniendaten hinzu
            line_flow_graph.add_edge(source_station, target_station,
                                     key=edge_key,
                                     flow=flow,
                                     service=service,
                                     direction=direction)

    return station_flow_graph, line_flow_graph

