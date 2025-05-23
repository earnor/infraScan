import glob
import os
import re
import tkinter as tk
import zipfile

import fiona
import pulp
import rasterio
import requests
from rasterio.features import geometry_mask
from scipy.stats._qmc import LatinHypercube
from shapely import LineString
from shapely.geometry import shape
from shapely.ops import split, nearest_points
from tqdm import tqdm

from generate_infrastructure import create_lines


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
    # Create new columns for user input
    df['tunnel_len'] = None
    df['bridge_len'] = None


    # Iterate through the DataFrame and show the custom pop-up for rows with 'check_needed' set to True
    for index, row in df.iterrows():
        if row['check_needed']:
            root = tk.Tk()
            root.withdraw()
            #dlg.wait_window()
    df["elevation_profile"]=df["elevation_profile"].astype('string')
    print(df)
    #df.to_file(r"data\Network\processed\new_links_realistic_tunnel.gpkg")
    df.to_file(r"data\Network\processed\new_links_realistic_tunnel-terminal.gpkg")


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
