import pandas as pd
from tqdm import tqdm  # Stellen Sie sicher, dass tqdm importiert wird
from joblib import Parallel, delayed
# from scipy.spatial import cKDTree
# Additional imports for grid creation
from data_import import *
from random_scenarios import load_scenarios_from_cache
import numba

def create_directed_graph(df):
    G = nx.DiGraph()
    
    # Add main nodes for each station
    for station in set(df['FromStation']).union(set(df['ToStation'])):
        G.add_node(f"main_{station}", type="main_node", station=station)
    
    # Add sub-nodes and connections between them
    for idx, row in df.iterrows():
        from_sub_node = f"sub_{row['FromStation']}_{row['Service']}_{row['Direction']}"
        to_sub_node = f"sub_{row['ToStation']}_{row['Service']}_{row['Direction']}"
        
        # Add sub-nodes
        G.add_node(from_sub_node, type="sub_node", station=row['FromStation'], Service=row['Service'], direction=row['Direction'])
        G.add_node(to_sub_node, type="sub_node", station=row['ToStation'], Service=row['Service'], direction=row['Direction'])
        
        # Add direct edge with TravelTime weight
        if pd.notna(row['TravelTime']):
            weight = int(round(row['TravelTime']))  # Ensure integer weights
            G.add_edge(from_sub_node, to_sub_node, weight=weight)
    
    # Add sub-to-main and main-to-sub switching with penalties
    for node in G.nodes:
        if G.nodes[node]["type"] == "sub_node":
            station = G.nodes[node]["station"]
            main_node = f"main_{station}"
            if main_node in G.nodes:
                G.add_edge(node, main_node, weight=3)  # Sub-to-Main
                G.add_edge(main_node, node, weight=3)  # Main-to-Sub
    
    # Ensure sub-to-sub direct edges within the same line and direction
    sub_nodes = [node for node, data in G.nodes(data=True) if data["type"] == "sub_node"]
    for sub1 in sub_nodes:
        for sub2 in sub_nodes:
            if sub1 != sub2:
                data1 = G.nodes[sub1]
                data2 = G.nodes[sub2]
                # Add edge only if the same service, direction, and station
                if (data1["Service"] == data2["Service"] and 
                    data1["direction"] == data2["direction"] and 
                    data1["station"] != data2["station"]):
                    # Get travel time from DataFrame for this connection
                    row = df[
                        (df['FromStation'] == data1["station"]) &
                        (df['ToStation'] == data2["station"]) &
                        (df['Service'] == data1["Service"]) &
                        (df['Direction'] == data1["direction"])
                    ]
                    if not row.empty:
                        travel_time = int(row.iloc[0]['TravelTime'])
                        G.add_edge(sub1, sub2, weight=travel_time)
    
    return G



def create_graphs_from_directories(directories):
    """
    Create a list of directed graphs from a list of file directories.
    
    Parameters:
        directories (list): List of file paths to GeoPackage or CSV files.
        
    Returns:
        list: A list of NetworkX directed graphs.
    """
    graphs = []
    for i, directory in enumerate(directories):
        try:
            print(f"Reading file {i+1}/{len(directories)}: {directory}...")
            # Read the file into a GeoDataFrame
            if directory.endswith('.gpkg'):
                df = gpd.read_file(directory)
            elif directory.endswith('.csv'):
                df = pd.read_csv(directory)
            else:
                print(f"Unsupported file format: {directory}")
                continue

            # Convert TravelTime to integers for consistent weight usage
            if "TravelTime" in df.columns:
                df["TravelTime"] = df["TravelTime"].round().astype(int)
            
            # Create the graph
            graph = create_directed_graph(df)
            graphs.append(graph)
            print(f"Graph {i+1} created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error processing file {directory}: {e}")
    return graphs


# Function to calculate the fastest travel time between two main nodes
def calculate_fastest_travel_time(graph, origin, destination):
    """
    Calculate the fastest travel time between two main nodes.
    
    Parameters:
        graph (networkx.DiGraph): The graph representing the railway network.
        origin (str): The name of the origin station.
        destination (str): The name of the destination station.
        
    Returns:
        tuple: Shortest path and total travel time.
    """
    source_node = f"main_{origin}"
    target_node = f"main_{destination}"
    
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
    Create all OD pairs for main stations across multiple graphs and calculate travel times using optimized methods.
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

        # Extract main station nodes for the current graph
        main_nodes = [node for node, data in graph.nodes(data=True) if data.get("type") == "main_node"]

        # Use all_pairs_dijkstra_path_length for optimized distance calculation
        all_lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="weight"))

        for origin in main_nodes:
            origin_lengths = all_lengths.get(origin, {})
            for destination in main_nodes:
                if origin != destination:
                    travel_time = origin_lengths.get(destination, None)
                    od_data.append({
                        "from_id": origin,
                        "to_id": destination,
                        "time": travel_time
                    })

        # Convert the OD data for this graph to a DataFrame
        od_df = pd.DataFrame(od_data)
        od_df["graph_id"] = graph_id  # Add graph_id as a column
        graph_dataframes.append(od_df)  # Append the DataFrame to the list

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
        df = od_df.copy()
        df["from_name"] = df["from_id"].str.replace("main_", "")
        df["to_name"] = df["to_id"].str.replace("main_", "")
        from_names = df["from_name"].values
        to_names = df["to_name"].values
        times = df["time"].values.astype(np.float64)
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
    def extract_travel_times(od_matrix, od_pairs):
        extracted_data = []
        for origin, destination in od_pairs:
            filtered_data = od_matrix[(od_matrix['from_id'] == origin) & 
                                      (od_matrix['to_id'] == destination)]
            if not filtered_data.empty:
                extracted_data.append({
                    "origin": origin,
                    "destination": destination,
                    "time": filtered_data.iloc[0]['time']
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
