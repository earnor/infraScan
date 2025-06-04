import pandas

import cost_parameters as cp
import paths

from data_import import *
from plots import plot_costs_benefits_example
import os
import pandas as pd

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
from rasterio.features import geometry_mask, rasterize
from shapely.ops import unary_union
from tqdm import tqdm
from rasterio.warp import reproject
import glob
import settings
import ast  # For safely evaluating string representations of lists

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
            "TotalMaintenanceCost": total_maintenance_cost,
            "YearlyMaintenanceCost": total_maintenance_cost / duration
        })

        # Update the base construction cost data for the next iteration
        df_construction_cost = pd.concat([df_construction_cost, dev_df], ignore_index=True)

    # Create a summary DataFrame for development costs
    development_costs_df = pd.DataFrame(development_costs)
    development_costs_df.to_csv("data/costs/construction_cost.csv", index=False)

    return development_costs_df


def aggregate_costs(cost_and_benefits):
    """
    Aggregate and calculate total costs for each development and scenario.
    Uses the cost_and_benefits DataFrame to sum up construction costs,
    maintenance costs, and benefits over the entire 50-year period.

    Parameters:
        cost_and_benefits (pd.DataFrame): DataFrame with costs and benefits for each development,
                                          scenario, and year, with MultiIndex (development, scenario, year).
    """

    # Define scenarios for population and employment
    #pop_scenarios = settings.pop_scenarios
    #empl_scenarios = settings.empl_scenarios
    # Wenn cost_and_benefits einen MultiIndex hat, diesen zurücksetzen
    if isinstance(cost_and_benefits.index, pd.MultiIndex):
        cost_and_benefits = cost_and_benefits.reset_index()

    scenarios = cost_and_benefits["scenario"].unique().tolist()

    # Group by development and scenario, summing costs and benefits across all years
    aggregated = cost_and_benefits.groupby(['development', 'scenario']).agg({
        'const_cost': 'sum',
        'maint_cost': 'sum',
        'benefit': 'sum'
    }).reset_index()

    # Load travel time savings for compatibility with existing structure
    # This provides the necessary structure for the output DataFrame
    travel_time_path = "data/costs/traveltime_savings.csv"
    c_travel_time = pd.read_csv(travel_time_path)
    total_costs = c_travel_time
    # Initialize total costs DataFrame with the existing structure
    # but using the aggregated values from cost_and_benefits
    
    # Initialize columns with zeros
    total_costs["construction_cost"] = 0
    total_costs["maintenance"] = 0
    total_costs["climate_cost"] = 0
    total_costs["land_realloc"] = 0
    total_costs["nature"] = 0
    total_costs["noise_s1"] = 0
    total_costs["noise_s2"] = 0
    total_costs["noise_s3"] = 0
    
    # Update the construction and maintenance costs from the aggregated data
    for _, row in aggregated.iterrows():
        dev = row['development']
        scenario = row['scenario']
        mask = (total_costs['development'] == dev) & (total_costs['scenario'] == scenario)
        
        if any(mask):
            total_costs.loc[mask, 'construction_cost'] = row['const_cost']
            total_costs.loc[mask, 'maintenance'] = row['maint_cost']
            # Use negative of the benefit as monetized_savings
            # Assuming benefits are positive when they save money, so costs should be negative
            total_costs.loc[mask, 'monetized_savings'] = row['benefit']

    # Dynamically compute total costs for each scenario
    for scenario in scenarios:
        total_costs[f"total_{scenario}"] = (
            total_costs["construction_cost"] +
            total_costs["maintenance"] +
            total_costs.get(f"local_{scenario}", 0) +
            total_costs.get(f"tt_{scenario}", 0) +
            total_costs.get(f"externalities_{scenario}", 0)
        )

    total_costs["TotalConstructionCost"] = total_costs["construction_cost"]
    total_costs["TotalMaintenanceCost"] = total_costs["maintenance"]
    # Save results to CSV
    total_costs.to_csv(paths.TOTAL_COST_RAW, index=False)

    print(f"Total costs raw saved to {paths.TOTAL_COST_RAW}")

def transform_and_reshape_cost_df():
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
    df = pd.read_csv(paths.TOTAL_COST_RAW)

    # Drop the specified columns
    columns_to_drop = [
        'status_quo_tt', 'development_tt',
        'total_pop_urban_', 'total_pop_equal_', 'total_pop_rural_',
        'total_pop_urba_1', 'total_pop_equa_1', 'total_pop_rura_1',
        'total_pop_urba_2', 'total_pop_equa_2', 'total_pop_rura_2'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Reshaping the dataframe
    reshaped_df = df.pivot_table(
        index='development',
        columns='scenario',
        values=['monetized_savings', 'construction_cost','maintenance' ],
        aggfunc='first'
    )

    # Flatten the MultiIndex columns
    reshaped_df.columns = [f"{col[0]}_{col[1]}" for col in reshaped_df.columns]

    # Reset index to have 'development' as a column
    reshaped_df = reshaped_df.reset_index()

    # Calculate total costs for each scenario
    construction_cost_columns = [col for col in reshaped_df.columns if col.startswith("construction_cost_")]
    savings_columns = [col for col in reshaped_df.columns if col.startswith("monetized_savings_")]
    
    for savings_col in savings_columns:
        scenario_name = savings_col.replace("monetized_savings_", "")
        construction_cost_col = f"construction_cost_{scenario_name}"
        maintenance_cost_col = f"maintenance_{scenario_name}"
        if construction_cost_col in reshaped_df.columns:
            reshaped_df[f"total_cost_{scenario_name}"] = reshaped_df[savings_col] + reshaped_df[construction_cost_col] + reshaped_df[maintenance_cost_col]

    # Identify columns to keep
    construction_cost_col = [col for col in reshaped_df.columns if col.startswith('construction_cost_od_matrix_')][0]
    maintenance_cost_col = [col for col in reshaped_df.columns if col.startswith('maintenance_od_matrix_')][0]

    # Create new columns with scaled values
    reshaped_df['Construction Cost [in Mio. CHF]'] = reshaped_df[construction_cost_col] / 1_000_000
    reshaped_df['Maintenance Costs [in Mio. CHF]'] = reshaped_df[maintenance_cost_col] / 1_000_000

    # Drop all original construction and maintenance columns
    columns_to_drop = [col for col in reshaped_df.columns if col.startswith('construction_cost_od_matrix_') or col.startswith('maintenance_od_matrix_combined_')]
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
    reshaped_df.to_csv(paths.TOTAL_COST_WITH_GEOMETRY, index=False)
    gdf.to_file("data/costs/total_costs_with_geometry.gpkg", driver="GPKG")

    print("Transformed dataframe saved to:")
    print(f"- CSV: '{paths.TOTAL_COST_WITH_GEOMETRY}'")
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


def GetOevDemandPerCommune(tau = 0.13): # Data is in trips per OD combination per day. Now we assume the number of trips gone in peak hour
    # now we extract an od matrix for oev tripps from year 2019
    # we then modify the OD matrix to fit our needs of expressing peak hour travel demand
    rawOD = pd.read_excel(paths.OD_KT_ZH_PATH)
    communalOD = rawOD.loc[
        (rawOD['jahr'] == 2018) & (rawOD['kategorie'] == 'Verkehrsaufkommen') & (rawOD['verkehrsmittel'] == 'oev')]
    # communalOD = data.drop(['jahr','quelle_name','quelle_gebietart','ziel_name','ziel_gebietart',"kategorie","verkehrsmittel","einheit","gebietsstand_jahr","zeit_dimension"],axis=1)
    # sum(communalOD['wert'])
    # # # Not binnenverkehr ... removes about 50% of trips
    communalOD['wert'].loc[(communalOD['quelle_code'] == communalOD['ziel_code'])] = 0
    # sum(communalOD['wert'])
    # # Take share of OD
    # todo adapt this value
    # This ratio explains the interzonal trips made in peak hour as a ratio of total interzonal trips made per day.
    # communalOD['wert'] = (communalOD['wert']*tau)
    communalOD.loc[:, 'wert'] = communalOD['wert'] * tau
    # # # Not those who travel < 15 min ?  Not yet implemented.
    return communalOD


def GetODMatrix(od):
    od_int = od.loc[(od['quelle_code'] < 9999) & (od['ziel_code'] < 9999)]
    od_ext = od.loc[(od['quelle_code'] > 9999) | (od[                                                      'ziel_code'] > 9999)]  # here we separate the parts of the od matrix that are outside the canton. We can add them later.
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

def GetCatchmentOD(use_cache = False):
    """
    Defines the `GetCatchmentOD` function, which performs operations to analyze spatial and raster
    data for a defined research corridor. This includes managing raster data, defining spatial
    boundaries, calculating interactions between communes and catchment areas, and handling multiple
    scenarios of population and employment distributions. Raster data is corrected and processed
    against a catchment raster for spatial alignment.

    This function leverages several geospatial operations, including reading and filtering geospatial
    data files, processing rasters for employment and population distribution scenarios, and creating
    origin-destination (OD) matrices for interaction analysis. It utilizes external functions and
    data sources like commune population, employment levels, and demand per commune for processing.

    Attributes:
        The key attributes are contained within function-specific variables. Key spatial
        and raster data inputs include:
        - Catchment raster paths.
        - Scenario-related raster paths for employment and population.

        The function also calculates raster data corrections and scenario-specific matrices stored in
        dictionaries for downstream analysis.

    Raises:
        The function handles errors during matrix construction, ensuring the data's consistency
        across populations, employment data, and origin-destination matrices. Errors are shown
        uniquely for OD matrix shapes.
    """
    if use_cache == True:
        return
    # Import the required data or define the path to access it
    catchment_tif_path = r'data/catchment_pt/catchement.tif'
    catchmentdf = gpd.read_file(r"data/catchment_pt/catchement.gpkg")

    # Paths to input and output files
    pop_combined_file = r"data/independent_variable/processed/scenario/pop_combined.tif"
    empl_combined_file = r"data/independent_variable/processed/scenario/empl_combined.tif"

    output_pop_path = r"data/independent_variable/processed/scenario/pop20_corrected.tif"
    output_empl_path = r"data/independent_variable/processed/scenario/empl20_corrected.tif"

    # Correct rasters using the catchment raster as reference
    correct_rasters_to_extent(
        empl_path=empl_combined_file,
        pop_path=pop_combined_file,
        output_empl_path=output_empl_path,
        output_pop_path=output_pop_path,
        reference_raster_path=catchment_tif_path)



    # todo When we iterate over devs and scens, maybe we can check if the VoronoiDF already has the communal data and then skip the following five lines
    popvec = GetCommunePopulation(y0="2021")
    jobvec = GetCommuneEmployment(y0=2021)
    od = GetOevDemandPerCommune(tau=1) ## check tau values for PT
    #OD is list with from;to;traffic_volume
    odmat = GetODMatrix(od) #Attention!!!! The demand from outer zones still must be added

    # This function returns a np array of raster data storing the bfs number of the commune in each cell
    commune_raster, commune_df = GetCommuneShapes(raster_path=catchment_tif_path)

    if jobvec.shape[0] != odmat.shape[0]:
        print(
            "Error: The number of communes in the OD matrix and the number of communes in the employment data do not match.")

    # Define scenario names for population and employment
    pop_scenarios = settings.pop_scenarios
    empl_scenarios = settings.empl_scenarios

    # Create dictionaries to store the raster data for each scenario
    pop_raster_data = {}
    empl_raster_data = {}

    # Read population scenarios from the combined raster file
    with rasterio.open(output_pop_path) as src:
        for idx, scenario in enumerate(pop_scenarios, start=1):  # Start from band 1
            pop_raster_data[scenario] = src.read(idx)  # Read each band

    # Read employment scenarios from the combined raster file
    with rasterio.open(output_empl_path) as src:
        for idx, scenario in enumerate(empl_scenarios, start=1):  # Start from band 1
            empl_raster_data[scenario] = src.read(idx)  # Read each band

    # Paths to input and output files
    empl_raster_path = r"data/independent_variable/processed/raw/empl20.tif"
    pop_raster_path = r"data/independent_variable/processed/raw/pop20.tif"

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
    with rasterio.open(output_empl_path) as src:
        scen_empl_20_tif = src.read(1)

    with rasterio.open(output_pop_path) as src:
        scen_pop_20_tif = src.read(1)

    #Load the catchment raster data
    with rasterio.open(catchment_tif_path) as src:
        # Read the raster data
        catchment_tif = src.read(1)  # Read the first band, which holds id information
        bounds = src.bounds  # Get the spatial bounds of the raster


    # Filter commune_df based on catchment raster bounds
    commune_df_filtered = commune_df.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]

    # Extract "BFS" values (unique commune IDs) within the bounds
    commune_df_filtered = commune_df_filtered["BFS"].to_numpy()

    # Ensure the OD matrix corresponds only to filtered communes
    odmat_frame = odmat.loc[commune_df_filtered, commune_df_filtered]
    print(f"Total sum of the od matrix communes: {np.nansum(odmat_frame)}")

    # Initialize an OD matrix for catchments
    # Assume vectorized functions are defined for the below operations
    def compute_cont_r(odmat, popvec, jobvec):
        # Convert popvec and jobvec to 2D arrays for broadcasting
        pop_matrix = np.array(popvec)[:, np.newaxis]
        job_matrix = np.array(jobvec)[np.newaxis, :]

        # Ensure odmat is a NumPy array
        odmat = np.array(odmat)

        # Perform the vectorized operation
        cont_r = odmat / (pop_matrix * job_matrix)
        return cont_r

    def compute_cont_v(cont_r, pop_m, job_n):
        # Sum over the cont_r matrix, multiply by pop_m and job_n
        cont_v = np.sum(cont_r)
        return cont_v
    ###############################################################################################################################
    # Step 1: generate unit_flow matrix from each commune to each other commune
    outer = np.outer(popvec, jobvec)
    cout_r = odmat / outer
    ###############################################################################################################################
    # Step 2: Get all pairs of combinations from communes to polygons

    pairs, pop_empl = pop_empl_to_catchment_commune_pairs(
        catchment_tif=catchment_tif,
        commune_raster=commune_raster,
        empl_raster_data=empl_raster_data,
        empl_scenarios=empl_scenarios,
        pop_raster_data=pop_raster_data,
        pop_scenarios=pop_scenarios,
        scen_empl_20_tif=scen_empl_20_tif,
        scen_pop_20_tif=scen_pop_20_tif)

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
    pop_scenarios = [col for col in pop_empl.columns if 'pop' in col.lower()]   #only keep population columns
    empl_scenarios = [col for col in pop_empl.columns if 'empl' in col.lower()]
    # for each of these scenarios make an own copy of od_matrix named od_matrix+scen
    for i in range(len(pop_scenarios)):
        print(f"Processing scenario {pop_scenarios[i],empl_scenarios[i]}")
        od_matrix_temp = od_matrix.copy()


        for polygon_id, row in tqdm(pop_empl.iterrows(), desc='Allocating pop and empl to OD matrix'):
            # Debug: Print the current polygon_id and row data
            print(f"Processing polygon_id: {polygon_id}, scenario: {pop_scenarios[i],empl_scenarios[i]}")
            print(f"Row data: {row}")

            # Debug: Print the sum before scaling
            print(f"Sum before scaling: {od_matrix_temp.sum().sum()}")

            # Multiply all values in the row/column
            scaling_factor_pop = row[pop_scenarios[i]]  # Extract scaling factor
            scaling_factor_empl = row[empl_scenarios[i]]  # Extract scaling factor
            print(f"Scaling factor: {scaling_factor_pop,scaling_factor_empl}")  # Debug: Check scaling factor

            od_matrix_temp.loc[polygon_id] *= scaling_factor_pop
            od_matrix_temp.loc[:, polygon_id] *= scaling_factor_empl

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

        od_grouped.to_csv(fr"data/traffic_flow/od/rail/od_matrix_{pop_scenarios[i],empl_scenarios[i]}.csv")
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
        catchmentdf_temp.to_file(fr"data/traffic_flow/od/catchment_id_{pop_scenarios[i]}.gpkg", driver="GPKG") #only output

        """        # Same for odmat and commune_df
        if scen == "20":
            origin_commune = odmat_frame.sum(axis=1).reset_index()
            origin_commune.colum = ["commune_id", "origin"]
            destination_commune = odmat_frame.sum(axis=0).reset_index()
            destination_commune.colum = ["commune_id", "destination"]
            commune_df = commune_df.merge(origin_commune, how='left', left_on='BFS', right_on='quelle_code')
            commune_df = commune_df.merge(destination_commune, how='left', left_on='BFS', right_on='ziel_code')
            commune_df = commune_df.rename(columns={'0_x': 'origin', '0_y': 'destination'})
            commune_df.to_file(r"data/traffic_flow/od/OD_commune_filtered.gpkg", driver="GPKG")
            """
    return


def pop_empl_to_catchment_commune_pairs(catchment_tif, commune_raster, empl_raster_data, empl_scenarios,
                                        pop_raster_data, pop_scenarios, scen_empl_20_tif, scen_pop_20_tif):
    # Identify unique catchment and commune IDs
    unique_catchment_id = np.sort(np.unique(catchment_tif))
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
    return pairs, pop_empl


def combine_and_save_od_matrices(directory, status_quo_directory):
    """
    Combines Origin-Destination (OD) matrices for population and employment data
    for multiple scenarios and saves the combined results. This function also
    ensures the proper handling of status quo and scenario-specific files
    by renaming them as original files to avoid overwriting.

    The function processes OD matrix files for both the status quo and specific
    scenarios, combines their data by summing population and employment matrices,
    saves the combined files, and removes or renames original files to maintain
    clarity in scenario versions.

    Args:
        directory (str): The directory containing the population and employment
            OD matrix files for various scenarios and status quo.
        status_quo_directory (str): The target directory where the combined
            OD matrix for the status quo will be saved.
    """
    # Ensure the status quo directory exists
    os.makedirs(status_quo_directory, exist_ok=True)

    # Define scenarios for population and employment
    pop_scenarios = settings.pop_scenarios

    empl_scenarios = settings.empl_scenarios

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


def link_traffic_to_map():
    """
    Links traffic flow data to a geospatial map representation and processes it for analysis. The
    function imports traffic flow data and geospatial link data, integrates them, and outputs a file
    with combined attributes for further use. It ensures data consistency by comparing lengths of
    datasets and sorts the geospatial link data for correct assignment of traffic flow values.
    The resulting dataset is saved in the specified file format, keeping only relevant columns such
    as geometric and traffic flow information.

    Raises:
        FileNotFoundError: If the specified CSV or geopackage files are not found or improperly specified.

    Imports:
        - pandas: Used for handling the traffic flow dataset.
        - geopandas: Used for handling geospatial data structures and operations.

    Side Effects:
        - Prints the first few rows of datasets for debugging or verification.
        - Outputs a file containing the geospatial links enriched with traffic flow data.

    File Output:
        Generates a processed geopackage file at "data/Network/processed/edges_only_flow.gpkg".
    """
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


def monetize_tts(VTTS, duration):
    """
    Processes travel time data from multiple scenarios, monetizes the time differences
    from a reference scenario, and outputs the calculated cost savings to a new file.

    Args:
        VTTS (float): Value of travel time savings expressed in CHF/h.
        duration (int): Duration over which the travel time is monetized in years.

    Raises:
        FileNotFoundError: If the required CSV files are not found.
        KeyError: If specified columns do not exist in the input data files.
        ValueError: If input data cannot be converted to the required types.
    """
    # Import total travel time for each scenario and each development
    tt_total = pd.read_csv(r"data/traffic_flow/travel_time.csv")

    tt_total["low"] = tt_total["low"].apply(lambda x: float(x[1:-1]))
    tt_total["medium"] = tt_total["medium"].apply(lambda x: float(x[1:-1]))
    tt_total["high"] = tt_total["high"].apply(lambda x: float(x[1:-1]))

    # Import reference travel time for each scenario and current infrastructure
    tt_status_quo = pd.read_csv(fr"data/traffic_flow/travel_time_status_quo.csv")

    # monetization factor of travel time (peak hour * CHF/h * 365 d/a * 50 a)
    mon_factor = VTTS * 365 * duration

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

def discounting(df, discount_rate):
    """
    Apply discounting to costs and benefits

    Args:
        df: DataFrame with multi-index (development, scenario, year)
        discount_rate: Annual discount rate (default 2%)

    Returns:
        DataFrame with discounted values
    """
    # Create a copy to avoid modifying the original
    df_discounted = df.copy()

    # Calculate discount factors for each year
    years = df.index.get_level_values('year').unique()
    discount_factors = {year: 1 / ((1 + discount_rate) ** (year - 1)) for year in years}

    # Apply discounting to each column
    columns_to_discount = ['maint_cost', 'const_cost', 'benefit']
    for col in columns_to_discount:
        for year in years:
            mask = df_discounted.index.get_level_values('year') == year
            df_discounted.loc[mask, col] *= discount_factors[year]

    return df_discounted


def create_cost_and_benefit_df(construction_and_maintenance_costs, dev_list, monetized_tt, scenario_list):
    # Create full index for the complete DataFrame
    full_index = pd.MultiIndex.from_product(
        [dev_list, scenario_list, list(range(1, cp.duration + 1))],
        names=["development", "scenario", "year"]
    )
    # Create an index for costs (which only vary by development and year)
    cost_index = pd.MultiIndex.from_product(
        [dev_list, list(range(1, cp.duration + 1))],
        names=["development", "year"]
    )
    # Create an empty DataFrame for costs with the development-year index
    cost_df = pd.DataFrame(index=cost_index, columns=["const_cost", "maint_cost"])
    # Fill the cost_df DataFrame with construction and maintenance costs
    for _, row in construction_and_maintenance_costs.iterrows():
        dev_name = row["Development"]
        total_construction_cost = row["TotalConstructionCost"]
        yearly_maintenance_cost = row["YearlyMaintenanceCost"]

        # Add construction cost only in year 1
        cost_df.loc[(dev_name, 1), "const_cost"] = total_construction_cost

        # Set maintenance cost to 0 for year 1
        cost_df.loc[(dev_name, 1), "maint_cost"] = 0

        # Add yearly maintenance costs for years 2 through duration
        for year in range(2, cp.duration + 1):
            cost_df.loc[(dev_name, year), "const_cost"] = 0  # No construction costs after year 1
            cost_df.loc[(dev_name, year), "maint_cost"] = yearly_maintenance_cost
    # Create the full DataFrame with all columns
    costs_and_benefits_dev = pd.DataFrame(index=full_index, columns=["const_cost", "maint_cost",
                                                                     "benefit"])  # contains benefits and costs for each year for every scenario and development
    # Convert monetized benefits directly to a dictionary for safer lookup
    monetized_benefits_dict = monetized_tt.set_index(["development", "scenario"])["monetized_savings_yearly"].to_dict()
    # Safely assign benefits using loop to avoid index mismatches
    for idx in costs_and_benefits_dev.index:
        dev, scenario, year = idx
        key = (dev, scenario)
        if key in monetized_benefits_dict:
            costs_and_benefits_dev.loc[idx, "benefit"] = monetized_benefits_dict[key]
    # Manual filling of costs for each scenario-development-year combination
    # This approach ensures no index mismatches
    print("Filling in costs for each development and year...")
    for idx in costs_and_benefits_dev.index:
        dev, scenario, year = idx
        dev_year_key = (dev, year)

        if dev_year_key in cost_df.index:
            # Assign construction cost for this development-year
            costs_and_benefits_dev.loc[idx, "const_cost"] = cost_df.loc[dev_year_key, "const_cost"]

            # Assign maintenance cost for this development-year
            costs_and_benefits_dev.loc[idx, "maint_cost"] = cost_df.loc[dev_year_key, "maint_cost"]
    # Save the costs and benefits DataFrame to CSV
    costs_benefits_csv_path = "data/costs/costs_and_benefits_dev.csv"
    print(f"Saving costs and benefits to {costs_benefits_csv_path}")
    costs_and_benefits_dev.to_csv(costs_benefits_csv_path)
    # Save a reset_index version for easier analysis if needed
    costs_benefits_flat_csv_path = "data/costs/costs_and_benefits_flat.csv"
    costs_and_benefits_dev.reset_index().to_csv(costs_benefits_flat_csv_path, index=False)
    print(f"Saving flattened version to {costs_benefits_flat_csv_path}")
    # Apply discounting to the DataFrame
    return costs_and_benefits_dev


def aggregate_commune_od_to_station_od(commune_od_df, commune_station_df):
    # Rename for clarity
    mapping_df = commune_station_df.rename(columns={'Commune_BFS_code': 'quelle_code'})

    # Map origin communes to stations
    merged_df = commune_od_df.merge(mapping_df, on='quelle_code', how='left')
    merged_df = merged_df.rename(columns={'ID_point': 'from_station'})

    # Map destination communes to stations
    mapping_df = commune_station_df.rename(columns={'Commune_BFS_code': 'ziel_code'})
    merged_df = merged_df.merge(mapping_df, on='ziel_code', how='left')
    merged_df = merged_df.rename(columns={'ID_point': 'to_station'})

    # Drop any rows with missing station mappings
    merged_df = merged_df.dropna(subset=['from_station', 'to_station'])

    # Group by station pairs and sum the demand
    station_od_df = merged_df.groupby(['from_station', 'to_station'], as_index=False)['wert'].sum()

    # Pivot to wide matrix format (optional)
    station_od_matrix = station_od_df.pivot(index='from_station', columns='to_station', values='wert').fillna(0)

    return station_od_matrix


def filter_od_matrix_by_stations(railway_station_OD, stations_in_perimeter):
    """
    Filtert die Einträge der OD-Matrix. Setzt Werte auf 0, wo weder die Zeilen-ID noch
    die Spalten-ID in der Liste stations_in_perimeter enthalten ist.

    Parameters:
        railway_station_OD (pd.DataFrame): OD-Matrix mit Stationen als Indizes
        stations_in_perimeter (list): Liste von Station-IDs, die im Perimeter liegen

    Returns:
        pd.DataFrame: OD-Matrix mit gefilterten Einträgen (auf 0 gesetzt)
    """
    # Kopie der Matrix erstellen, um die Originaldaten nicht zu verändern
    filtered_od = railway_station_OD.copy()

    # Extrahiere die IDs aus den Stationseinträgen (falls die Liste Tupel mit [ID, NAME] enthält)
    station_ids = [station[0] if isinstance(station, list) else station for station in stations_in_perimeter]

    # Erstelle einen Filter für Zeilen, die NICHT in station_ids sind
    rows_not_in_filter = ~filtered_od.index.isin(station_ids)

    # Erstelle einen Filter für Spalten, die NICHT in station_ids sind
    cols_not_in_filter = ~filtered_od.columns.isin(station_ids)

    # Erstelle eine Maske, wo weder Zeilen- noch Spalten-ID in station_ids ist
    # (Verwendet Broadcasting für elementweise Logik)
    mask = np.outer(rows_not_in_filter, cols_not_in_filter)

    # Setze Werte auf 0, wo die Maske True ist
    filtered_od.values[mask] = 0

    print(f"Originale OD-Matrix-Summe: {railway_station_OD.values.sum()}")
    print(f"Gefilterte OD-Matrix-Summe: {filtered_od.values.sum()}")
    print(f"Anzahl der Stationen im Perimeter: {len(station_ids)}")

    return filtered_od