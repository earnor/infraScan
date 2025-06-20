# Core Libraries
import os
import math
import re

import networkx
import numpy
import numpy as np
import pandas as pd

# Geospatial Libraries
import geopandas as gpd
import seaborn
from shapely.geometry import Point, LineString, box
from shapely import make_valid
# Raster Data Libraries
import rasterio
from rasterio.plot import show
from rasterio.mask import mask

# Network Analysis Libraries
import networkx as nx

# Visualization Libraries
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import contextily as ctx
import contextily as cx
from geo_northarrow import add_north_arrow
from matplotlib import patches as mpatches, pyplot
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
import matplotlib.lines as mlines
import matplotlib.colors as mcolors




# Pyrosm for OpenStreetMap Data
from pyrosm import get_data, OSM

import settings


def plotting(input_file, output_file, node_file):

    # Read the GeoPackage file
    gdf = gpd.read_file(input_file)

    # Read the mapping file
    mapping_file = gpd.read_file("data/Network/processed/updated_new_links.gpkg")

    # Correct regex for extracting numeric part of 'ID_new'
    gdf['dev_numeric'] = gdf['development'].str.extract(r'Development_(\d+)', expand=False)

    # Handle NaN values by replacing them with a placeholder
    gdf['dev_numeric'] = gdf['dev_numeric'].fillna(-1).astype(int)

    # Step 2: Normalize `dev_id` in mapping_file
    mapping_file['dev_numeric'] = mapping_file['dev_id'].astype(int)

    # Debug: Print unique numeric values before merging
    print("Numeric values in gdf['dev_numeric']:", gdf['dev_numeric'].unique())
    print("Numeric values in mapping_file['dev_numeric']:", mapping_file['dev_numeric'].unique())

    # Step 3: Merge `from_ID_new` and `to_ID` into `gdf`
    gdf = gdf.merge(
        mapping_file[['dev_numeric', 'from_ID_new', 'to_ID']], 
        on='dev_numeric', 
        how='left'
    )

    # Step 4: Rename ID columns for clarity
    gdf.rename(columns={
        'from_ID_new': 'Source_Node_ID',
        'to_ID': 'Target_Node_ID'
    }, inplace=True)

    # Step 5: Drop the temporary 'dev_numeric' column
    gdf.drop(columns=['dev_numeric'], inplace=True)

    # Step 6: Read the node file and prepare mapping
    node_data = pd.read_excel(node_file)

    # Convert columns to the same type for merging
    gdf['Source_Node_ID'] = gdf['Source_Node_ID'].astype(int)
    gdf['Target_Node_ID'] = gdf['Target_Node_ID'].astype(int)
    node_data['NR'] = node_data['NR'].astype(int)

    # Merge to get Source and Target Names
    gdf = gdf.merge(
        node_data[['NR', 'NAME']], 
        left_on='Source_Node_ID', 
        right_on='NR', 
        how='left'
    ).rename(columns={'NAME': 'Source_Name'}).drop(columns=['NR'])

    gdf = gdf.merge(
        node_data[['NR', 'NAME']], 
        left_on='Target_Node_ID', 
        right_on='NR', 
        how='left'
    ).rename(columns={'NAME': 'Target_Name'}).drop(columns=['NR'])

    gdf = gdf.loc[:, ~gdf.columns.duplicated()]

    gdf.rename(columns={
    'Source_Node_ID': 'Source_ID',
    'Target_Node_ID': 'Target_ID',
    'Source_Name': 'Source_Name',
    'Target_Name': 'Target_Name'}, inplace=True)

    # Debug: Check if the merge succeeded
    print("After merge, NULL values in 'Source_ID':", gdf['Source_ID'].isnull().sum())
    print("After merge, NULL values in 'Target_ID':", gdf['Target_ID'].isnull().sum())

    # Filter columns containing 'construction_cost' in their name
    construction_costs_columns = [col for col in gdf.columns if 'construction_cost' in col]

    if construction_costs_columns:
        # Keep only the first column with 'construction_cost' in the name
        first_column = construction_costs_columns[0]

        # Retain all other columns and geometry
        other_columns = [col for col in gdf.columns if col not in construction_costs_columns or col == first_column]
        gdf = gdf[other_columns]

        # Rename the first 'construction_cost' column
        gdf.rename(columns={first_column: "Construction and Maintenance Cost in Mio. CHF"}, inplace=True)

        # Convert the first 'construction_cost' column values to millions (divide by 1,000,000)
        gdf["Construction and Maintenance Cost in Mio. CHF"] = gdf["Construction and Maintenance Cost in Mio. CHF"] / 1_000_000

    # Define pairings of monetized savings and net benefit columns
    pairings = [
        ("monetized_savings_total_od_matrix_combined_pop_equa_1", "Net Benefit Equal Medium [in Mio. CHF]"),
        ("monetized_savings_total_od_matrix_combined_pop_equa_2", "Net Benefit Equal High [in Mio. CHF]"),
        ("monetized_savings_total_od_matrix_combined_pop_equal_", "Net Benefit Equal Low [in Mio. CHF]"),
        ("monetized_savings_total_od_matrix_combined_pop_rura_1", "Net Benefit Rural Medium [in Mio. CHF]"),
        ("monetized_savings_total_od_matrix_combined_pop_rura_2", "Net Benefit Rural High [in Mio. CHF]"),
        ("monetized_savings_total_od_matrix_combined_pop_rural_", "Net Benefit Rural Low [in Mio. CHF]"),
        ("monetized_savings_total_od_matrix_combined_pop_urba_1", "Net Benefit Urban Medium [in Mio. CHF]"),
        ("monetized_savings_total_od_matrix_combined_pop_urba_2", "Net Benefit Urban High [in Mio. CHF]"),
        ("monetized_savings_total_od_matrix_combined_pop_urban_", "Net Benefit Urban Low [in Mio. CHF]")
    ]

    # Create and save new DataFrames for each pairing
    for savings_col, net_benefit_col in pairings:
        if savings_col in gdf.columns and net_benefit_col in gdf.columns:
            scenario_df = gdf[[
                "development",
                "Source_ID",  # Use renamed column
                "Target_ID",  # Use renamed column
                "Source_Name",  # Add Source_Name
                "Target_Name",  # Add Target_Name
                "Construction and Maintenance Cost in Mio. CHF",
                savings_col,
                net_benefit_col,
                "geometry",
                "Sline"
            ]].copy()

            # Round all values to 1 decimal place
            scenario_df = scenario_df.round(1)

            # Ensure only one geometry column exists
            scenario_df = scenario_df.loc[:, ~scenario_df.columns.duplicated()]
            scenario_df.set_geometry("geometry", inplace=True)

            # Generate a scenario name from the net benefit column
            scenario_name = net_benefit_col.replace("Net Benefit ", "").replace("[in Mio. CHF]", "").strip().replace(" ", "_")
            scenario_output_file = f"{output_file.replace('.gpkg', '')}_{scenario_name}.gpkg"

            # Save the scenario DataFrame to a file
            if not scenario_df.empty:
                scenario_df.to_file(scenario_output_file, driver='GPKG')
                print(f"Saved: {scenario_output_file}")
            else:
                print(f"No data to save for {scenario_name}")
    

def plot_developments_and_table_for_scenarios(input_dir, output_dir):
    """
    Plots all developments on a map with OSM as the background and labels them.
    Creates a corresponding table with development details for each scenario.
    
    Parameters:
        input_dir (str): Directory containing GeoPackage files with processed costs.
        output_dir (str): Directory to save the map and table images.
    """
    # Define the path to your .osm.pbf file
    pbf_file = "data/_basic_data/planet_8.4,47.099_9.376,47.492.osm.pbf"

    # Load the OSM data
    osm = OSM(pbf_file)

    # Extract desired network data (e.g., roads, paths, waterways)
    roads = osm.get_network(network_type="all")  # Options: "driving", "walking", etc.

    # Save the roads data as a GeoPackage
    output_gpkg = "data/osm_map.gpkg"
    roads.to_file(output_gpkg, driver="GPKG")
    print(f"Converted OSM data saved to {output_gpkg}")
    
    # Set a grey theme for the OSM map
    osm_color = '#d9d9d9'

    # Loop through all GeoPackage files in the input directory
    for file in os.listdir(input_dir):
        if file.endswith(".gpkg") and "processed_costs" in file:
            # Read the GeoPackage file
            gdf = gpd.read_file(os.path.join(input_dir, file))
            
            # Extract scenario name from the file name
            scenario_name = os.path.splitext(file)[0]
            
            # Set up the plot for the map
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            osm.plot(ax=ax, color=osm_color, edgecolor='white', linewidth=0.5)

            # Plot developments
            gdf.plot(ax=ax, color='red', edgecolor='black', linewidth=1, alpha=0.8)

            # Add labels for developments
            for x, y, label in zip(gdf.geometry.centroid.x, gdf.geometry.centroid.y, gdf['ID_new']):
                ax.text(x, y, label, fontsize=8, color='black', ha='center', va='center', weight='bold')

            # Add scalebar
            scalebar = ScaleBar(1, location='lower right', units='m', scale_loc='bottom')
            ax.add_artist(scalebar)

            # Remove axes for a cleaner map
            ax.axis('off')

            # Save the map
            output_map = os.path.join(output_dir, f"{scenario_name}_map.png")
            plt.title(f"Developments and OSM Map: {scenario_name}", fontsize=14)
            plt.tight_layout()
            plt.savefig(output_map, dpi=300)
            plt.close()

            # Plot the corresponding table
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Prepare data for the table
            table_data = gdf[['ID_new', 'Source_Name', 'Target_Name']]
            divider = make_axes_locatable(ax)
            table_ax = divider.append_axes("bottom", size="75%", pad=0.1)

            # Remove map axes and add table
            ax.axis('off')
            table_ax.axis('tight')
            table_ax.axis('off')
            table = table_ax.table(
                cellText=table_data.values,
                colLabels=table_data.columns,
                loc='center',
                cellLoc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.auto_set_column_width(col=list(range(len(table_data.columns))))

            # Save the table
            output_table = os.path.join(output_dir, f"{scenario_name}_table.png")
            plt.tight_layout()
            plt.savefig(output_table, dpi=300)
            plt.close()

            print(f"Map saved to {output_map}")
            print(f"Table saved to {output_table}")


def plot_bus_network(G, pos, e_min, e_max, n_min, n_max):
    """
    Plots the bus network on a map with an OpenStreetMap background.
    
    Parameters:
    - G: NetworkX graph representing the bus network.
    - pos: Dictionary of positions for each node in the format {node: (x, y)}.
    - e_min, e_max, n_min, n_max: Floats defining the extent of the map to plot.
    """
    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot the network
    nx.draw(
        G, pos,
        node_size=10,
        node_color='red',
        with_labels=False,
        edge_color='blue',
        linewidths=1,
        font_size=8,
        ax=ax
    )

    # Set the axis limits
    ax.set_xlim(e_min, e_max)
    ax.set_ylim(n_min, n_max)

    # Add OpenStreetMap background
    ctx.add_basemap(ax, crs="EPSG:2056", source=ctx.providers.OpenStreetMap.Mapnik)

    # Display the plot
    plt.show()

# Example usage:
# Define the limits of your research corridor
# e_min, e_max, n_min, n_max = <appropriate values>

# Call the function
# plot_bus_network(G_bus, pos, e_min, e_max, n_min, n_max)





class CustomBasemap:
    def __init__(self, boundary=None, network=None, access_points=None, frame=None, canton=False):
        # Create a figure and axis
        self.fig, self.ax = plt.subplots(figsize=(15, 10))

        # Plot cantonal border
        if canton==True:
            canton = gpd.read_file(r"data/Scenario/Boundaries/Gemeindegrenzen/UP_KANTON_F.shp")
            canton[canton["KANTON"] == 'Zürich'].boundary.plot(ax=self.ax, color="black", lw=2)

        # Plot lakes
        lakes = gpd.read_file(r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp")
        lakes.plot(ax=self.ax, color="lightblue")

        # Add scale bar
        self.ax.add_artist(ScaleBar(1, location="lower right"))

        if isinstance(network, gpd.GeoDataFrame):
            network.plot(ax=self.ax, color="black", lw=2)

        if isinstance(access_points, gpd.GeoDataFrame):
            access_points.plot(ax=self.ax, color="black", markersize=50)

        location = gpd.read_file(r'data/manually_gathered_data/Cities.shp', crs="epsg:2056")
        # Plot the location as points
        location.plot(ax=self.ax, color="black", markersize=75)
        # Add city names to the plot
        for idx, row in location.iterrows():
            self.ax.annotate(row['location'], xy=row["geometry"].coords[0], ha="center", va="top", xytext=(0, -6),
                        textcoords='offset points', fontsize=15)

        self.ax.set_xticks([])
        self.ax.set_yticks([])

        if boundary:
            min_x, min_y, max_x, max_y = boundary.bounds
            self.ax.set_xlim(min_x, max_x)
            self.ax.set_ylim(min_y, max_y)

        if frame:
            x, y = frame.exterior.xy  # Extract the exterior coordinates
            self.ax.plot(x, y, color='b', alpha=0.7, linewidth=2)


    def savefig(self, path):
        plt.savefig(path+".png", ax=self.ax, dpi=500)


    def show(self):
        plt.show()

    def new_development(self, new_links=None, new_nodes=None):
        if isinstance(new_links, gpd.GeoDataFrame):
            print("ploting links")
            new_links.plot(ax=self.ax, color="darkgray", lw=2)

        if isinstance(new_nodes, gpd.GeoDataFrame):
            print("ploting nodes")
            new_nodes.plot(ax=self.ax, color="blue", markersize=50)


    def single_development(self, id ,new_links=None, new_nodes=None):
        if isinstance(new_links, gpd.GeoDataFrame):
            #print("ploting links")
            new_links[new_links["ID_new"] == id].plot(ax=self.ax, color="darkgray", lw=2)

        if isinstance(new_nodes, gpd.GeoDataFrame):
            #print("ploting nodes")
            new_nodes[new_nodes["ID"] == id].plot(ax=self.ax, color="blue", markersize=50)

    def voronoi(self, id, gdf_voronoi):
        gdf_voronoi["ID"] = gdf_voronoi["ID"].astype(int)
        #print(gdf_voronoi[gdf_voronoi["ID"] == id].head(9).to_string())
        gdf_voronoi[gdf_voronoi["ID"] == id].plot(ax=self.ax, edgecolor='red', facecolor='none' , lw=2)
        plt.savefig(r"plot/Voronoi/developments/dev_" + str(id) + ".png", dpi=400)


def plot_cost_result(df_costs, banned_area, title_bar, boundary=None, network=None, access_points=None, plot_name=False, col="total_medium"):
    # cmap = "viridis"

    # Determine the range of your data
    min_val = df_costs[col].min()
    max_val = df_costs[col].max()

    # Number of color intervals
    n_intervals = 256
    # Define a gray color for the zero point
    gray_color = [0.83, 0.83, 0.83, 1]  # RGBA for gray

    if (min_val < 0) & (max_val > 0):
        total_range = abs(min_val) + abs(max_val)
        neg_proportion = abs(min_val) / total_range
        pos_proportion = abs(max_val) / total_range

        # Generate colors for negative (red) and positive (blue) ranges
        neg_colors = plt.cm.Reds_r(np.linspace(0.15, 0.8, int(n_intervals * neg_proportion)))
        pos_colors = plt.cm.Blues(np.linspace(0.3, 0.95, int(n_intervals * pos_proportion)))

        # Create a transition array from reds to gray and from gray to blues
        transition_length = int(n_intervals * 0.2)  # Length of the transition zone
        reds_to_gray = np.linspace(neg_colors[-1], gray_color, transition_length)
        gray_to_blues = np.linspace(gray_color, pos_colors[0], transition_length)

        # Create an array that combines the colors with a smooth transition
        all_colors = np.vstack((neg_colors[:-1], reds_to_gray, gray_to_blues, pos_colors[1:]))

    elif min_val >= 0:
        # Case with only positive values
        pos_colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_intervals))
        gray_to_blues = np.linspace(gray_color, pos_colors[0], int(n_intervals * 0.3))
        all_colors = np.vstack((gray_to_blues, pos_colors[1:]))

    elif max_val <= 0:
        # Case with only negative values
        neg_colors = plt.cm.Reds_r(np.linspace(0.2, 0.8, n_intervals))
        reds_to_gray = np.linspace(neg_colors[-1], gray_color, int(n_intervals * 0.3))
        all_colors = np.vstack((neg_colors[:-1], reds_to_gray))

    # Create the new colormap
    cmap = LinearSegmentedColormap.from_list("custom_colormap", all_colors)
    fig, ax = plt.subplots(figsize=(15, 10))
    # Plot lakes
    lakes = gpd.read_file(r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp")
    lakes.plot(ax=ax, color="lightblue", zorder=9)

    # Add scale bar
    ax.add_artist(ScaleBar(1, location="lower right"))

    if isinstance(network, gpd.GeoDataFrame):
        network.plot(ax=ax, color="black", lw=2, zorder=11)

    if isinstance(access_points, gpd.GeoDataFrame):
        access_points.plot(ax=ax, color="black", markersize=50, zorder=12)

    location = gpd.read_file(r'data/manually_gathered_data/Cities.shp', crs="epsg:2056")
    # Plot the location as points
    location.plot(ax=ax, color="black", markersize=75, zorder=13)
    # Add city names to the plot
    for idx, row in location.iterrows():
        ax.annotate(row['location'], xy=row["geometry"].coords[0], ha="center", va="top", xytext=(0, -6),
                         textcoords='offset points', fontsize=15, zorder=13)

    # Get min max values of point coordinates
    bounds = df_costs.total_bounds  # returns (xmin, ymin, xmax, ymax)
    xmin, ymin, xmax, ymax = bounds

    # Interpolating values for heatmap
    grid_x, grid_y = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]  # Adjust grid size as needed
    points = np.array([df_costs.geometry.x, df_costs.geometry.y]).T
    values = df_costs[col]
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear') # cubic

    # Plot heatmap
    heatmap = ax.imshow(grid_z.T, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap=cmap, alpha=0.8, zorder=2)

    # Plot points
    df_costs.plot(ax=ax, column=col, cmap=cmap, zorder=4, edgecolor='black', linewidth=1)

    # Create an axis for the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.5)

    # Create the colorbar in the new axis
    cbar = plt.colorbar(heatmap, cax=cax)

    # Add and rotate the colorbar title
    cbar.set_label(f'Net benefits in {title_bar} [Mio. CHF]/n(Construction, maintenance, highway travel time, access time and external effects)', rotation=90,
                    labelpad=30, fontsize=16)

    # Set the font size of the colorbar's tick labels
    cbar.ax.tick_params(labelsize=14)

    raster = rasterio.open(banned_area)
    cmap_raster = ListedColormap(["white", "white"])
    rasterio.plot.show(raster, ax=ax, cmap=cmap_raster, zorder=3)

    # Create custom legend elements

    # Create the legend below the plot

    # Add a north arrow
    # Add the letter "N"
    ax.text(0.96, 0.92, "N", fontsize=28, weight=1, ha='center', va='center', transform=ax.transAxes, zorder=1000)

    # Add a custom north arrow
    arrow = FancyArrowPatch((0.96, 0.89), (0.96, 0.97), color='black', lw=2, arrowstyle='->', mutation_scale=25, transform=ax.transAxes, zorder=1000)
    ax.add_patch(arrow)

    ax.set_xticks([])
    ax.set_yticks([])

    # Get plot limits
    min_x, min_y, max_x, max_y = boundary.bounds
    ax.set_xlim(min_x - 100, max_x + 100)
    ax.set_ylim(min_y - 100, max_y + 100)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1)  # Adjust linewidth for frame thickness
        spine.set_zorder(1000)

    if plot_name != False:
        plt.tight_layout()
        plt.savefig(fr"plot/results/04_{plot_name}.png", dpi=300)

    plt.show()
    return

def plot_single_cost_result(df_costs, banned_area , title_bar, boundary=None, network=None, access_points=None, plot_name=False, col="total_medium"):
    #cmap = "viridis"
    df_costs[col] = df_costs[col] / 10**6

    # Determine the range of your data
    min_val = df_costs[col].min()
    max_val = df_costs[col].max()

    print(f"min: {min_val}, max: {max_val}")

    # Number of color intervals
    n_intervals = 256
    # Define a gray color for the zero point
    gray_color = [0.83, 0.83, 0.83, 1]  # RGBA for gray

    if (min_val < 0) & (max_val > 0):
        total_range = abs(min_val) + abs(max_val)
        neg_proportion = abs(min_val) / total_range
        pos_proportion = abs(max_val) / total_range

        # Generate colors for negative (red) and positive (blue) ranges
        neg_colors = plt.cm.Reds_r(np.linspace(0.15, 0.8, int(n_intervals * neg_proportion)))
        pos_colors = plt.cm.Blues(np.linspace(0.3, 0.95, int(n_intervals * pos_proportion)))

        # Create a transition array from reds to gray and from gray to blues
        transition_length = int(n_intervals * 0.2)  # Length of the transition zone
        reds_to_gray = np.linspace(neg_colors[-1], gray_color, transition_length)
        gray_to_blues = np.linspace(gray_color, pos_colors[0], transition_length)

        # Create an array that combines the colors with a smooth transition
        all_colors = np.vstack((neg_colors[:-1], reds_to_gray, gray_to_blues, pos_colors[1:]))

    elif min_val >= 0:
        # Case with only positive values
        pos_colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_intervals))
        gray_to_blues = np.linspace(gray_color, pos_colors[0], int(n_intervals * 0.3))
        all_colors = np.vstack((gray_to_blues, pos_colors[1:]))

    elif max_val <= 0:
        # Case with only negative values
        neg_colors = plt.cm.Reds_r(np.linspace(0.2, 0.8, n_intervals))
        reds_to_gray = np.linspace(neg_colors[-1], gray_color, int(n_intervals * 0.3))
        all_colors = np.vstack((neg_colors[:-1], reds_to_gray))

    # Create the new colormap
    cmap = LinearSegmentedColormap.from_list("custom_colormap", all_colors)

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(15, 10))
    # Plot lakes
    lakes = gpd.read_file(r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp")
    lakes.plot(ax=ax, color="lightblue", zorder=9)

    # Add scale bar
    ax.add_artist(ScaleBar(1, location="lower right"))

    if isinstance(network, gpd.GeoDataFrame):
        network.plot(ax=ax, color="black", lw=2, zorder=11)

    if isinstance(access_points, gpd.GeoDataFrame):
        access_points.plot(ax=ax, color="black", markersize=50, zorder=12)

    location = gpd.read_file(r'data/manually_gathered_data/Cities.shp', crs="epsg:2056")
    # Plot the location as points
    location.plot(ax=ax, color="black", markersize=75, zorder=13)
    # Add city names to the plot
    for idx, row in location.iterrows():
        ax.annotate(row['location'], xy=row["geometry"].coords[0], ha="center", va="top", xytext=(0, -8),
                         textcoords='offset points', fontsize=15, zorder=13)

    # Get min max values of point coordinates
    bounds = df_costs.total_bounds  # returns (xmin, ymin, xmax, ymax)
    xmin, ymin, xmax, ymax = bounds

    # Interpolating values for heatmap
    grid_x, grid_y = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]  # Adjust grid size as needed
    points = np.array([df_costs.geometry.x, df_costs.geometry.y]).T
    values = df_costs[col]
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear') # cubic

    # Plot heatmap
    heatmap = ax.imshow(grid_z.T, extent=(xmin, xmax, ymin, ymax), origin='lower', cmap=cmap, alpha=0.8, zorder=2)

    # Plot points
    df_costs.plot(ax=ax, column=col, cmap=cmap, zorder=4, edgecolor='black', linewidth=1)

    # Create an axis for the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.5)

    # Create the colorbar in the new axis
    cbar = plt.colorbar(heatmap, cax=cax)

    # Add and rotate the colorbar title
    cbar.set_label(f'Net benefit of {title_bar} [Mio. CHF]', rotation=90, labelpad=30, fontsize=16)

    # Set the font size of the colorbar's tick labels
    cbar.ax.tick_params(labelsize=14)

    raster = rasterio.open(banned_area)
    cmap_raster = ListedColormap(["white", "white"])
    rasterio.plot.show(raster, ax=ax, cmap=cmap_raster, zorder=3)

    # Create custom legend elements

    # Create the legend below the plot

    ax.set_xticks([])
    ax.set_yticks([])

    # Get plot limits
    min_x, min_y, max_x, max_y = boundary.bounds
    ax.set_xlim(min_x, max_x+100)
    ax.set_ylim(min_y, max_y)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1)  # Adjust linewidth for frame thickness
        spine.set_zorder(1000)

    if plot_name != False:
        plt.tight_layout()
        plt.savefig(fr"plot/results/04_{plot_name}.png", dpi=300)

    plt.show()
    return

def plot_cost_uncertainty(df_costs, banned_area, col, legend_title, boundary=None, network=None, access_points=None, plot_name=False):

    # Determine the range of your data
    min_val = df_costs["mean_costs"].min()
    max_val = df_costs["mean_costs"].max()

    # Number of color intervals
    n_intervals = 256
    # Define a gray color for the zero point
    gray_color = [0.83, 0.83, 0.83, 1]  # RGBA for gray

    if (min_val < 0) & (max_val > 0):
        total_range = abs(min_val) + abs(max_val)
        neg_proportion = abs(min_val) / total_range
        pos_proportion = abs(max_val) / total_range

        # Generate colors for negative (red) and positive (blue) ranges
        neg_colors = plt.cm.Reds_r(np.linspace(0.15, 0.8, int(n_intervals * neg_proportion)))
        pos_colors = plt.cm.Blues(np.linspace(0.3, 0.95, int(n_intervals * pos_proportion)))

        # Create a transition array from reds to gray and from gray to blues
        transition_length = int(n_intervals * 0.2)  # Length of the transition zone
        reds_to_gray = np.linspace(neg_colors[-1], gray_color, transition_length)
        gray_to_blues = np.linspace(gray_color, pos_colors[0], transition_length)

        # Create an array that combines the colors with a smooth transition
        all_colors = np.vstack((neg_colors[:-1], reds_to_gray, gray_to_blues, pos_colors[1:]))

    elif min_val >= 0:
        # Case with only positive values
        pos_colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_intervals))
        gray_to_blues = np.linspace(gray_color, pos_colors[0], int(n_intervals * 0.3))
        all_colors = np.vstack((gray_to_blues, pos_colors[1:]))

    elif max_val <= 0:
        # Case with only negative values
        neg_colors = plt.cm.Reds_r(np.linspace(0.2, 0.8, n_intervals))
        reds_to_gray = np.linspace(neg_colors[-1], gray_color, int(n_intervals * 0.3))
        all_colors = np.vstack((neg_colors[:-1], reds_to_gray))

    # Create the new colormap
    cmap = LinearSegmentedColormap.from_list("custom_colormap", all_colors)
    fig, ax = plt.subplots(figsize=(20, 10))
    # Plot lakes
    lakes = gpd.read_file(r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp")
    lakes.plot(ax=ax, color="lightblue", zorder=9)

    # Add scale bar
    ax.add_artist(ScaleBar(1, location="lower right"))

    if isinstance(network, gpd.GeoDataFrame):
        network.plot(ax=ax, color="black", lw=2, zorder=11)

    if isinstance(access_points, gpd.GeoDataFrame):
        access_points.plot(ax=ax, color="black", markersize=50, zorder=12)

    location = gpd.read_file(r'data/manually_gathered_data/Cities.shp', crs="epsg:2056")
    # Plot the location as points
    location.plot(ax=ax, color="black", markersize=75, zorder=13)
    # Add city names to the plot
    for idx, row in location.iterrows():
        ax.annotate(row['location'], xy=row["geometry"].coords[0], ha="center", va="top", xytext=(0, -6),
                         textcoords='offset points', fontsize=15, zorder=13)
    """
    # Comopute markersize based on cv value but they should range within 2 - 50
    # Assuming 'df' is your DataFrame and 'value_column' is the column you want to normalize
    min_val, max_val = df_costs['std'].min(), df_costs['std'].max()
    scale_min, scale_max = 10, 400
    # Normalize the column
    df_costs['markersize'] = scale_max - (((df_costs['std'] - min_val) / (max_val - min_val)) * (scale_max - scale_min))
    # Plot points
    """
    scale_min, scale_max = 30, 500
    # Apply a non-linear transformation (e.g., logarithm) to the 'std' column
    df_costs[f'log_{col}'] = np.log(df_costs[col])  # You can use np.log10 for base 10 logarithm if needed
    # Normalize the transformed column
    min_val = df_costs[f'log_{col}'].min()
    max_val = df_costs[f'log_{col}'].max()
    df_costs['markersize'] = scale_max - (((df_costs[f'log_{col}'] - min_val) / (max_val - min_val)) * (scale_max - scale_min))

    df_costs_sorted = df_costs.sort_values(by='mean_costs')
    df_costs_sorted.plot(ax=ax, column="mean_costs", markersize="markersize", cmap=cmap, zorder=4, edgecolor='black', linewidth=1)

    # Get the position of the current plot
    pos = ax.get_position()

    # Create a new axes for the colorbar on the right of the plot
    y_start = 0.25
    cbar_ax = fig.add_axes([pos.x1 + 0.1, pos.y0 + y_start, 0.01, pos.y1 - y_start - 0.005])

    # Add the colorbar
    cbar_gdf = fig.colorbar(ax.collections[4], cax=cbar_ax)

    cbar_gdf.set_label(
        f'Mean Net benefits [Mio. CHF]\n(Construction, maintenance, highway travel\ntime, access time and external effects)',
        rotation=90, labelpad=30, fontsize=16)
    cbar_gdf.ax.tick_params(labelsize=14)


    raster = rasterio.open(banned_area)
    gray_brigth = (0.88, 0.88, 0.88)
    cmap_raster = ListedColormap([gray_brigth, gray_brigth])
    rasterio.plot.show(raster, ax=ax, cmap=cmap_raster, zorder=3)

    # Create custom legend elements

    # Create the legend below the plot
    # legend = ax.legend(handles=[water_body_patch, protected_area_patch], loc='lower center',bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=16, frameon=False)
    """
    # Create actual scatter points on the plot for the legend
    # Choose a range of std values for the legend
    original_std_values = np.linspace(min_val, max_val, 6)
    # Calculate corresponding marker sizes for these std values
    legend_sizes = scale_max - ((original_std_values - min_val) / (max_val - min_val)) * (scale_max - scale_min)
    # Create scatter plot handles for the legend
    legend_handles = [mlines.Line2D([], [], color='white', marker='o', linestyle='solid', linewidth=1, markerfacecolor='white', markeredgecolor='black',
                                    markersize=np.sqrt(size), label=f'{std_val:.1f}')
                      for size, std_val in zip(legend_sizes, original_std_values)]
    """
    # Choose a range of std values for the legend
    original_std_values = np.linspace(df_costs[col].min(), df_costs[col].max(), 6)  # Use original std values

    # Calculate corresponding marker sizes for these std values (reversed mapping)
    legend_sizes = scale_max - (
                ((np.log(original_std_values) - min_val) / (max_val - min_val)) * (scale_max - scale_min))

    # Create scatter plot handles for the legend with labels as original std values
    legend_handles = [
        mlines.Line2D([], [], color='white', marker='o', linestyle='solid', linewidth=1, markerfacecolor='white',
                      markeredgecolor='black',
                      markersize=np.sqrt(size), label=f'{std_val:.0f}')
        for size, std_val in zip(legend_sizes, original_std_values)]

    # Create patch elements for the legend
    water_body_patch = mpatches.Patch(facecolor="lightblue", label='Water bodies', edgecolor='black', linewidth=1)
    protected_area_patch = mpatches.Patch(facecolor=gray_brigth, label='Protected area', edgecolor='black', linewidth=1)

    # Combine scatter handles and patch elements
    combined_legend_elements = legend_handles + [water_body_patch, protected_area_patch]

    # Create a single combined legend below the plot
    combined_legend = ax.legend(handles=combined_legend_elements, loc='lower left', bbox_to_anchor=(1.015, 0),
                                fontsize=14, frameon=False, title=f'{legend_title}\n',title_fontsize=16)

    # Add the combined legend to the plot
    ax.add_artist(combined_legend)

    # Add a north arrow
    # Add the letter "N"
    ax.text(0.96, 0.92, "N", fontsize=28, weight=1, ha='center', va='center', transform=ax.transAxes, zorder=1000)

    # Add a custom north arrow
    arrow = FancyArrowPatch((0.96, 0.89), (0.96, 0.97), color='black', lw=2, arrowstyle='->', mutation_scale=25, transform=ax.transAxes, zorder=1000)
    ax.add_patch(arrow)

    ax.set_xticks([])
    ax.set_yticks([])

    # Get plot limits
    min_x, min_y, max_x, max_y = boundary.bounds
    ax.set_xlim(min_x - 100, max_x + 100)
    ax.set_ylim(min_y - 100, max_y + 100)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1)  # Adjust linewidth for frame thickness
        spine.set_zorder(1000)

    if plot_name != False:
        plt.tight_layout()
        plt.savefig(fr"plot/results/04_{plot_name}.png", dpi=300)

    plt.show()
    return


def plot_benefit_distribution_bar_single(df_costs, column):
    # Define bin width
    bin_width = 100
    # Automatically calculate bin edges and create a new column 'bin'
    # Calculate the desired bin edges
    min_value = df_costs[column].min()
    min_value = math.floor(min_value / bin_width) * bin_width
    if min_value % (2*bin_width) != 0:
        min_value = min_value - bin_width
    max_value = df_costs[column].max()
    max_value = math.ceil(max_value / bin_width) * bin_width

    # Calculate the number of bins based on the bin width
    num_bins = int((max_value - min_value) / bin_width)
    # Create bin edges that end with "00"
    bin_edges = [min_value + bin_width *i for i in range(num_bins + 1)]
    df_costs['bin'] = pd.cut(df_costs[column], bins=bin_edges, include_lowest=True)

    # Count occurrences in each bin
    bin_counts = df_costs['bin'].value_counts().sort_index()

    # Create a bar plot
    plt.bar(bin_counts.index.astype(str), bin_counts.values, color="black", zorder=3)

    # Set labels and title
    plt.xlabel('Net benefit [Mio CHF]', fontsize=12)
    plt.ylabel('Occurrence' , fontsize=12)
    # Define custom x-axis tick positions and labels based on bin boundaries
    bin_boundaries = [bin.left for bin in bin_counts.index] + [bin_counts.index[-1].right]
    custom_ticks = np.arange(len(bin_boundaries)) - 0.5  # One tick per bin boundary
    custom_labels = [f"{int(boundary)}" if i % 2 == 0 else '' for i, boundary in enumerate(bin_boundaries)]
    # Apply custom ticks and labels to the x-axis
    plt.xticks(custom_ticks, custom_labels, rotation=90)

    # Determine the appropriate y-axis tick step size dynamically
    max_occurrence = bin_counts.max()
    y_tick_step = 1
    while max_occurrence > 10 * y_tick_step:
        y_tick_step *= 2

    # Set y-axis ticks as integer multiples of the determined step size
    y_ticks = np.arange(0, max_occurrence + y_tick_step, y_tick_step)
    plt.yticks(y_ticks)

    # Calculate the actual bin boundaries for the shaded region
    min_shaded_region = next((i for i, val in enumerate(bin_edges) if val >= 0), None)
    plt.axvspan(min_shaded_region-0.5, custom_ticks.max()+0.5, color='lightgray', alpha=0.5)

    # Set x-axis limits
    plt.xlim(custom_ticks.min()-0.5, custom_ticks.max()+0.5)

    # Add light horizontal grid lines for each y-axis tick
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)

    plt.tight_layout()

    # Safe figure
    plt.savefig(r"plot/results/benefit_distribution.png", dpi=500)

    # Show the plot
    plt.show()


def plot_benefit_distribution_line_multi(df_costs, columns, labels, plot_name, legend_title):
    # Define bin width
    bin_width = 100
    # Automatically calculate bin edges and create a new column 'bin'
    # Calculate the desired bin edges
    min_value = df_costs[columns].min().min()
    min_value = math.floor(min_value / bin_width) * bin_width - bin_width * 2
    max_value = df_costs[columns].max().max()
    max_value = math.ceil(max_value / bin_width) * bin_width + bin_width*4

    num_bins = int((max_value - min_value) / bin_width)
    bin_edges = [min_value + bin_width * i for i in range(num_bins + 1)]

    for column in columns:
        df_costs[f'bin_{column}'] = pd.cut(df_costs[column], bins=bin_edges, include_lowest=True)

    # Initialize an empty DataFrame for bin_counts
    bin_counts = pd.DataFrame(index=bin_edges[:-1], columns=columns)

    # Count occurrences in each bin for each column
    for column in columns:
        column_counts = df_costs.groupby(f'bin_{column}')[column].count()
        bin_counts[f'bin_{column}'] = column_counts

    print(bin_counts.head(10).to_string())
    # Define labels
    # Check if labels len is same as columns len
    if len(labels) != len(columns):
        print("Labels and columns length are not the same")
    else:
        # Create a dict with column names as keys and labels as values
        legend_labels = dict(zip(columns, labels))

    linestyles = ['solid', 'dashdot', 'dashed', 'dotted']
    line_colors = ['darkgray', 'gray', 'dimgray', 'black'] # 'gray', 'lightgray',

    fig, ax = plt.subplots(figsize=(13, 6))

    # Create a line plot with legends
    for i, column in enumerate(columns):
        ax.plot(bin_counts.index.astype(str), bin_counts[f'bin_{column}'], label=legend_labels[column], color=line_colors[i], linestyle=linestyles[i])
    ax.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0., title=legend_title, fontsize=12, title_fontsize=14, frameon=False)

    plt.xlabel('Net benefit [Mio CHF]', fontsize=14)
    plt.ylabel('Occurrence', fontsize=14)
    plt.xticks(rotation=90)

    # Locate the legend right beside the plot
    #legend = ax.legend(title=legend_title, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10, title_fontsize=12, bbox_transform=ax.transAxes)

    # Shift the x-tick positions by 0.5 to the left
    current_xticks = plt.xticks()[0]  # Get current x-tick locations
    new_xtick_locations = [x + 0.5 for x in current_xticks]
    # only keep every second x-tick
    # Generate labels: keep every second label, replace others with empty strings
    current_labels = [label.get_text() for label in plt.gca().get_xticklabels()]
    new_labels = [label if i % 2 == 0 else '' for i, label in enumerate(current_labels)]

    # Set new x-tick positions with adjusted labels
    plt.xticks(ticks=new_xtick_locations, labels=new_labels)

    # Add vertical lines to all ticks with low linewidth and alpha
    plt.grid(axis='x', linestyle='-', linewidth=0.5, alpha=0.5)

    # Increase the size of the ticks with labels
    plt.tick_params(axis='x', which='major', length=6, width=1, labelsize=12)  # Adjust length and labelsize as needed

    max_occurrence = bin_counts.max().max()
    y_tick_step = 1
    while max_occurrence > 10 * y_tick_step:
        y_tick_step *= 2

    y_ticks = np.arange(0, max_occurrence + y_tick_step, y_tick_step)
    plt.yticks(y_ticks, fontsize=12)

    min_shaded_region = next((i for i, val in enumerate(bin_edges) if val >= 0), None)
    plt.axvspan(-0.5, min_shaded_region + 0.5, color='lightgray', alpha=0.5)

    plt.xlim(-0.5, len(bin_counts.index) - 0.5)
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)

    plt.tight_layout()
    plt.savefig(fr"plot/results/04_distribution_line_{plot_name}.png", dpi=500)
    plt.show()


def plot_best_worse(df):

    # Sort the DataFrame by "total_medium" in ascending and descending order
    df_top5 = df.nlargest(5, 'total_medium')
    df_bottom5 = df.nsmallest(5, 'total_medium')

    # Specify the columns to plot
    print(df.columns)
    columns_to_plot = ['building_costs', 'local_s1', 'externalities', 'tt_medium', 'noise_s1']

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Function to dynamically determine costs and benefits
    def categorize_values(row):
        costs = [val if val < 0 else 0 for val in row]
        benefits = [val if val >= 0 else 0 for val in row]
        return costs, benefits

    # Plot the top 5 rows in the first subplot
    for i, row in df_top5.iterrows():
        observation = row['ID_new']
        costs, benefits = categorize_values(row[columns_to_plot])

        axs[0].bar(columns_to_plot, costs, color='red', label=f'{observation} - Costs')
        axs[0].bar(columns_to_plot, benefits, bottom=costs, color='blue', label=f'{observation} - Benefits')

    # Plot the bottom 5 rows in the second subplot
    for i, row in df_bottom5.iterrows():
        observation = row['ID_new']
        costs, benefits = categorize_values(row[columns_to_plot])

        axs[1].bar(columns_to_plot, costs, color='red', label=f'{observation} - Costs')
        axs[1].bar(columns_to_plot, benefits, bottom=costs, color='blue', label=f'{observation} - Benefits')

    # Set labels and legend for each subplot
    axs[0].set_title('Top 5 Rows')
    axs[1].set_title('Bottom 5 Rows')
    axs[0].set_xlabel('Categories (Costs/Benefits)')
    axs[1].set_xlabel('Categories (Costs/Benefits)')
    axs[0].set_ylabel('Value')
    axs[0].legend(title='Legend', loc='upper left', bbox_to_anchor=(1, 1))
    axs[1].legend(title='Legend', loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def boxplot(df, nbr):
    # Calculate the mean for each development
    df["mean"] = df[['total_low', 'total_medium', 'total_high']].mean(axis=1)
    #mean_values = df.groupby('ID_new')['total_low', 'total_medium', 'total_high'].mean()

    # sort df by mean and keep to nbr rows
    df = df.sort_values(by=['mean'], ascending=False)
    df_top = df.head(nbr)

    df_top = df_top[['ID_new', 'total_low', 'total_medium', 'total_high']]
    # set ID_new as index and transpose df
    df_top = df_top.set_index('ID_new').T

    # Plotting the boxplot
    plt.figure(figsize=(20, 8))
    df_top.boxplot()

    # Color area 0f y<0 with light grey
    # Get min y value
    ymin, ymax = plt.ylim()
    plt.axhspan(ymin, 0, color='lightgrey', alpha=0.5)
    # Set y limit
    plt.ylim(ymin, ymax)

    plt.xlabel("Development ID", fontsize=22)
    plt.ylabel("Net benefits over all scenarios \n [Mio. CHF]", fontsize=22)
    # Increse fontsize for all ticks
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(r"plot/results/04_boxplot.png", dpi=500)
    plt.show()


def plot_2x3_subplots(gdf, network, location):
    """
    This function plots the relative population and employment development for all districts considered and for all
    three scenarios defined
    :param gdf: Geopandas DataFrame containing the growth values
    :param lim: List of coordinates defining the perimeter investigated
    :return:
    """
    lim = gdf.total_bounds
    vmin, vmax = 1, 1.75

    # Create a figure with 6 subplots arranged in two rows and three columns
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))

    # Loop through each column of the dataframe and plot it on its corresponding subplot
    index = [0, 1, 2, 3, 4, 5]
    columns = ["s2_pop", "s1_pop", "s3_pop", "s2_empl", "s1_empl", "s3_empl"]
    title = ["Population - low", "Population - medium", "Population - high",
             "Employment - low", "Employment - medium", "Employment - high"]
    for i in range(6):
        row = index[i] // 3
        col = index[i] % 3
        ax = axs[row, col]
        gdf.plot(column=columns[i], ax=ax, cmap='summer_r', edgecolor = "gray", vmin=vmin, vmax=vmax, lw=0.2)
        network.plot(ax=ax, color="black", linewidth=0.5)
        # Plot the location as points
        location.plot(ax=ax, color="black", markersize=20, zorder=7)
        for idx, row in location.iterrows():
            ax.annotate(row['location'], xy=row["geometry"].coords[0], ha="right", va="top", xytext=(0, -4),
                            textcoords='offset points', fontsize=7.5)

        ax.set_ylim(lim[1], lim[3])
        ax.set_xlim(lim[0], lim[2])
        ax.axis('off')
        ax.set_title(title[i], fontsize=9)

    # Set a common colorbar for all subplots
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap='summer_r', norm=norm)
    sm.set_array([])

    # Add the colorbar to the figure
    title_ax = fig.add_axes([0.97, 0.45, 0.05, 0.1])
    # cbar.ax.set_title("Relative population increase", rotation=90)
    # cbar.ax.yaxis.set_label_position('right')
    title_ax.axis('off')  # Hide the frame around the title axis
    title_ax.text(0.5, 0.5, 'Relative population and employment increase compared to 2020', rotation=90,
                  horizontalalignment='center', verticalalignment='center')

    # Show the plot
    plt.savefig(r"plot/Scenario/5_all_scen.png", dpi=450, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_points_gen(points, edges, banned_area, points_2=None, boundary=None, network=None, access_points=None, plot_name=False, all_zones=False):

    # Import other zones
    schutzzonen = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Schutzanordnungen_Natur_und_Landschaft_-SAO-_-OGD/FNS_SCHUTZZONE_F.shp")
    forest = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Waldareal_-OGD/WALD_WALDAREAL_F.shp")

    fig, ax = plt.subplots(figsize=(13,9))
    # Plot lakes
    lakes = gpd.read_file(r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp")
    lakes.plot(ax=ax, color="lightblue", zorder=9)

    # Add scale bar
    ax.add_artist(ScaleBar(1, location="lower right"))

    if isinstance(network, gpd.GeoDataFrame):
        network.plot(ax=ax, color="black", lw=2, zorder=11)

    if isinstance(access_points, gpd.GeoDataFrame):
        access_points.plot(ax=ax, color="black", markersize=50, zorder=12)

    location = gpd.read_file(r'data/manually_gathered_data/Cities.shp', crs="epsg:2056")
    # Plot the location as points
    location.plot(ax=ax, color="black", markersize=75, zorder=200)
    # Add city names to the plot
    for idx, row in location.iterrows():
        ax.annotate(row['location'], xy=row["geometry"].coords[0], ha="center", va="top", xytext=(0, -6),
                         textcoords='offset points', fontsize=15, zorder=200)

    # Plot points
    points.plot(ax=ax, zorder=100, edgecolor='darkslateblue', linewidth=2, color='white', markersize=70)

    # Plot edges
    edges.plot(ax=ax, zorder=90, linewidth=1, color='darkslateblue')

    if all_zones:
        # Plot other zones in lightgray
        schutzzonen.plot(ax=ax, color="lightgray", zorder=5)
        forest.plot(ax=ax, color="lightgray", zorder=5)
        #fff.plot(ax=ax, color="lightgray", zorder=5)


    raster = rasterio.open(banned_area)
    cmap_raster = ListedColormap(["lightgray", "lightgray"])
    rasterio.plot.show(raster, ax=ax, cmap=cmap_raster, zorder=3)

    # Create custom legend elements
    water_body_patch = mpatches.Patch(facecolor="lightblue", label='Water bodies', edgecolor='black', linewidth=1)
    protected_area_patch = mpatches.Patch(facecolor='lightgray', label='Infeasible area',
                                          edgecolor='black', linewidth=1)
    # Add existing network, generated points and generated links to the legend
    network_line = mlines.Line2D([], [], color='black', label='Current highway\nnetwork', linewidth=2)
    points_marker = mlines.Line2D([], [], color='white', marker='o', markersize=10, label='Generated points',
                                  markeredgecolor='darkslateblue', linestyle='None', linewidth=3)
    edges_line = mlines.Line2D([], [], color='darkslateblue', label='Generated links', linewidth=1.5)

    legend_handles = [network_line, points_marker, edges_line, water_body_patch, protected_area_patch]

    if isinstance(points_2, gpd.GeoDataFrame):
        points_2.plot(ax=ax, zorder=101, color='lightseagreen', markersize=70, edgecolor='black', linewidth=1.5)
        deleted_points_marker = mlines.Line2D([], [], color='lightseagreen', marker='o', markersize=10,
                                              label='Deleted points',markeredgecolor='black', linestyle='None', linewidth=1)
        legend_handles.insert(2, deleted_points_marker)

    # Create the legend below the plot
    legend = ax.legend(handles=legend_handles, loc='lower left', bbox_to_anchor=(1.02, 0), fontsize=16, frameon=False,
                       title="Legend", title_fontsize=20)
    legend._legend_box.align = "left"

    # Add a north arrow
    # Add the letter "N"
    ax.text(0.96, 0.925, "N", fontsize=20, weight=1, ha='center', va='center', transform=ax.transAxes, zorder=1000)

    # Add a custom north arrow
    arrow = FancyArrowPatch((0.96, 0.90), (0.96, 0.975), color='black', lw=2, arrowstyle='->', mutation_scale=20, transform=ax.transAxes, zorder=1000)
    ax.add_patch(arrow)

    ax.set_xticks([])
    ax.set_yticks([])

    # Get plot limits
    min_x, min_y, max_x, max_y = boundary.bounds
    ax.set_xlim(min_x - 100, max_x + 100)
    ax.set_ylim(min_y - 100, max_y + 100)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1)  # Adjust linewidth for frame thickness
        spine.set_zorder(1000)

    plt.tight_layout()
    if plot_name != False:
        plt.tight_layout()
        plt.savefig(fr"plot/results/04_{plot_name}.png", dpi=500, bbox_inches='tight')

    plt.show()
    return


def plot_voronoi_comp(eucledian, traveltime, boundary=None, network=None, access_points=None, plot_name=False):
    fig, ax = plt.subplots(figsize=(13, 9))
    # Plot lakes
    lakes = gpd.read_file(r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp")
    lakes.plot(ax=ax, color="lightblue", zorder=4)

    # Add scale bar
    ax.add_artist(ScaleBar(1, location="lower right"))

    if isinstance(network, gpd.GeoDataFrame):
        network.plot(ax=ax, color="black", lw=2, zorder=11)

    if isinstance(access_points, gpd.GeoDataFrame):
        access_points.plot(ax=ax, color="black", markersize=50, zorder=12)

    location = gpd.read_file(r'data/manually_gathered_data/Cities.shp', crs="epsg:2056")
    # Plot the location as points
    location.plot(ax=ax, color="black", markersize=75, zorder=200)
    # Add city names to the plot
    for idx, row in location.iterrows():
        ax.annotate(row['location'], xy=row["geometry"].coords[0], ha="center", va="top", xytext=(0, -6),
                         textcoords='offset points', fontsize=15, zorder=200)

    # Plot boundaries of eucledian
    eucledian.boundary.plot(ax=ax, color="lightgray", linewidth=3, zorder=4)
    # Plot boundaries of traveltime
    traveltime.boundary.plot(ax=ax, color="darkslateblue", linewidth=1.5, zorder=5)


    # Create custom legend elements
    water_body_patch = mpatches.Patch(facecolor="lightblue", label='Water bodies', edgecolor='black', linewidth=1)
    eucledian_patch = mpatches.Patch(facecolor='white', label='Euclidian Voronoi tiling',
                                          edgecolor='lightgray', linewidth=3)
    traveltime_patch = mpatches.Patch(facecolor='white', label='Travel time Voronoi tiling',
                                          edgecolor='darkslateblue', linewidth=1)
    # Add existing network, generated points and generated links to the legend
    network_line = mlines.Line2D([], [], color='black', label='Current highway\nnetwork', linewidth=2)


    legend_handles = [network_line, eucledian_patch, traveltime_patch, water_body_patch]


    # Create the legend below the plot
    legend = ax.legend(handles=legend_handles, loc='lower left', bbox_to_anchor=(1.02, 0), fontsize=16, frameon=False,
                       title="Legend", title_fontsize=20)
    legend._legend_box.align = "left"

    # Add a north arrow
    # Add the letter "N"
    ax.text(0.96, 0.925, "N", fontsize=16, weight=1, ha='center', va='center', transform=ax.transAxes, zorder=1000)

    # Add a custom north arrow
    arrow = FancyArrowPatch((0.96, 0.90), (0.96, 0.975), color='black', lw=1.5, arrowstyle='->', mutation_scale=14, transform=ax.transAxes, zorder=1000)
    ax.add_patch(arrow)

    ax.set_xticks([])
    ax.set_yticks([])

    # Get plot limits
    min_x, min_y, max_x, max_y = boundary.bounds
    ax.set_xlim(min_x - 100, max_x + 100)
    ax.set_ylim(min_y - 100, max_y + 100)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1)  # Adjust linewidth for frame thickness
        spine.set_zorder(1000)


    if plot_name != False:
        plt.tight_layout()
        plt.savefig(fr"plot/results/04_{plot_name}.png", dpi=500)

    plt.show()
    return


def plot_voronoi_development(statusquo, development_voronoi, development_point, boundary=None, network=None, access_points=None, plot_name=False):
    fig, ax = plt.subplots(figsize=(13, 9))
    # Plot lakes
    lakes = gpd.read_file(r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp")
    lakes.plot(ax=ax, color="lightblue", zorder=4)

    # Add scale bar
    ax.add_artist(ScaleBar(1, location="lower right"))

    if isinstance(network, gpd.GeoDataFrame):
        network.plot(ax=ax, color="black", lw=2, zorder=11)

    if isinstance(access_points, gpd.GeoDataFrame):
        access_points.plot(ax=ax, color="black", markersize=50, zorder=12)

    location = gpd.read_file(r'data/manually_gathered_data/Cities.shp', crs="epsg:2056")
    # Plot the location as points
    location.plot(ax=ax, color="black", markersize=75, zorder=200)
    # Add city names to the plot
    for idx, row in location.iterrows():
        ax.annotate(row['location'], xy=row["geometry"].coords[0], ha="center", va="top", xytext=(0, -6),
                         textcoords='offset points', fontsize=15, zorder=200)

    # Plot boundaries of eucledian
    statusquo.boundary.plot(ax=ax, color="darkgray", linewidth=2, zorder=4)
    # Plot boundaries of traveltime

    # Filter development we want
    # Plot according point and polygon
    i = 779
    ii = development_voronoi["ID_point"].max()
    development_point[development_point["ID_new"] == i].plot(ax=ax, color="darkslateblue", markersize=80, zorder=12)
    development_voronoi[development_voronoi["ID_point"] == ii].plot(ax=ax, facecolor="darkslateblue", alpha=0.3, edgecolor="black", linewidth=2, zorder=11)

    # Create custom legend elements
    water_body_patch = mpatches.Patch(facecolor="lightblue", label='Water bodies', edgecolor='black', linewidth=1)
    current_patch = mpatches.Patch(facecolor='white', label='Voronoi tiling for\ncurrent access points',
                                          edgecolor='darkgray', linewidth=3)
    newpoly_patch = mpatches.Patch(facecolor='darkslateblue', alpha=0.3, label='Voronoi polygon of the\ngenerated access point',
                                          edgecolor='black', linewidth=1)
    # Add existing network, generated points and generated links to the legend
    newpoint_path = mlines.Line2D([], [], color='darkslateblue', marker='o', markersize=15,
                                  label='Generated access point', linestyle='None')
    # Add existing network, generated points and generated links to the legend
    network_line = mlines.Line2D([], [], color='black', marker='o', markersize=10, label='Current highway\nnetwork', linewidth=2)


    legend_handles = [network_line, current_patch, newpoint_path, newpoly_patch, water_body_patch]


    # Create the legend below the plot
    legend = ax.legend(handles=legend_handles, loc='lower left', bbox_to_anchor=(1.02, 0), fontsize=16, frameon=False,
                       title="Legend", title_fontsize=20)
    legend._legend_box.align = "left"

    # Add a north arrow
    # Add the letter "N"
    ax.text(0.96, 0.925, "N", fontsize=24, weight=1, ha='center', va='center', transform=ax.transAxes, zorder=1000)

    # Add a custom north arrow
    arrow = FancyArrowPatch((0.96, 0.90), (0.96, 0.975), color='black', lw=1.5, arrowstyle='->', mutation_scale=18, transform=ax.transAxes, zorder=1000)
    ax.add_patch(arrow)

    ax.set_xticks([])
    ax.set_yticks([])

    # Get plot limits
    min_x, min_y, max_x, max_y = boundary.bounds
    ax.set_xlim(min_x - 100, max_x + 1000)
    ax.set_ylim(min_y - 100, max_y)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1)  # Adjust linewidth for frame thickness
        spine.set_zorder(1000)


    if plot_name != False:
        plt.tight_layout()
        plt.savefig(fr"plot/results/04_{plot_name}.png", dpi=500)

    plt.show()
    return


def plot_rail_network(graph_dict):
    """
    Plot multiple graphs from a dictionary of graphs.

    Args:
        graph_dict (dict): A dictionary where keys are identifiers (e.g., file paths or names) and 
                           values are NetworkX graph objects.
    """
    for graph_name, G in graph_dict.items():
        # Create a dictionary for positions using node geometries in G
        pos = {node: (data['geometry'][0], data['geometry'][1]) for node, data in G.nodes(data=True)}

        # Set up the plot
        plt.figure(figsize=(10, 10))
        plt.title(f"Graph: {graph_name}")

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=50, node_color='blue', alpha=0.7)

        # Draw edges
        nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=10, edge_color="gray", width=0.5)

        # Draw node labels
        nx.draw_networkx_labels(G, pos, labels={node: data['station'] for node, data in G.nodes(data=True)}, font_size=5)

        # Draw edge labels
        edge_labels = {(u, v): f"{d['service']}, {d['weight']}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5)

        # Show plot for the current graph
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.show()


def plot_scenarios():
    # File paths
    pop_file = "data/temp/data_scenario_pop.shp"
    empl_file = "data/temp/data_scenario_empl.shp"
    cities_file = "data/manually_gathered_data/cities.shp"
    output_path = "plots/scenarios.png"

    # Load data
    pop_data = gpd.read_file(pop_file)
    empl_data = gpd.read_file(empl_file)
    cities_data = gpd.read_file(cities_file)

    # Columns to plot (reordered: rural, equal, urban)
    pop_columns = ['pop_rural_', 'pop_equal_', 'pop_urban_']
    empl_columns = ['empl_rural', 'empl_equal', 'empl_urban']

    # Create the figure and axes
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Shared color scheme
    cmap = 'Reds'
    norm = Normalize(vmin=1.15, vmax=1.26)  # Adjusted color scale range

    # Function to generate plot titles
    def generate_title(data_type, column):
        if "rural" in column.lower():
            return f"{data_type}: Rural"
        elif "equal" in column.lower():
            return f"{data_type}: Equal"
        elif "urban" in column.lower():
            return f"{data_type}: Urban"

    # Function to plot a single map
    def plot_map(ax, gdf, column, title, cities_data):
        # Plot the main data layer with enhanced polygon boundaries
        gdf.plot(column=column, cmap=cmap, norm=norm, ax=ax, edgecolor='black', linewidth=0.2)
        
        # Plot the cities layer
        cities_data.plot(ax=ax, color='black', markersize=10)
        
        # Add city labels
        for _, row in cities_data.iterrows():
            ax.text(row.geometry.x, row.geometry.y, row['location'], fontsize=11, ha='center')
        
        ax.set_title(title, fontsize=16)
        ax.axis('off')

    # Plot population data
    for i, col in enumerate(pop_columns):
        title = generate_title("Population", col)
        plot_map(axes[i], pop_data, col, title, cities_data)

    # Plot employment data
    for i, col in enumerate(empl_columns):
        title = generate_title("Employment", col)
        plot_map(axes[i+3], empl_data, col, title, cities_data)

    # Add a single colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position: [left, bottom, width, height]
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Modelled growth rates between 2021 and 2050', fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for the colorbar

    # Save the plot to file
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")



def create_plot_catchement():
    # File paths
    raster_tif = "data/catchment_pt/catchement.tif"
    water_bodies_path = "data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp"
    location_path = "data/manually_gathered_data/Cities.shp"
    points_path = "data/Network/processed/points.gpkg"
    s_bahn_lines_path = "data/Network/processed/split_s_bahn_lines.gpkg"
    output_path = "plots/catchement.png"

    # Load data
    lakes = gpd.read_file(water_bodies_path)
    locations = gpd.read_file(location_path, crs="epsg:2056")
    points = gpd.read_file(points_path)
    s_bahn_lines = gpd.read_file(s_bahn_lines_path)

    # Open raster data
    with rasterio.open(raster_tif) as raster:
        # Get the raster extent
        raster_bounds = raster.bounds
        raster_extent = [raster_bounds.left, raster_bounds.right, raster_bounds.bottom, raster_bounds.top]

        # Read raster data
        raster_data = raster.read(1)

        # Extract unique values, excluding NoData (-1)
        unique_values = np.unique(raster_data)
        unique_ids = [val for val in unique_values if val != -1]  # Exclude NoData
        unique_ids.sort()  # Ensure the IDs are sorted

        print("Unique values in the raster:", unique_values)  # Debugging
        print("Unique values (excluding NoData):", unique_ids)  # Debugging

        # Define specific colors
        nodata_color = (0.678, 0.847, 0.902, 1.0)  # Soft blue for NoData
        orange_color = (1.0, 0.5, 0.0, 1.0)  # Orange for ID 6

        # Create a colormap for unique IDs
        colors = plt.cm.get_cmap("tab10", len(unique_ids)).colors
        colors = list(colors)
        custom_cmap = colors.copy()

        # Assign specific colors
        for idx, unique_id in enumerate(unique_ids):
            if unique_id == 6:  # Assign orange to ID 6
                custom_cmap[idx] = orange_color

        # Add NoData color (optional, transparent)
        custom_cmap.append(nodata_color)

        # Create colormap and normalization
        cmap = ListedColormap(custom_cmap)
        norm = BoundaryNorm(unique_ids + [unique_ids[-1] + 1], len(unique_ids))

        # Replace NoData values for visualization
        raster_display = np.where(raster_data == -1, np.nan, raster_data)

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot raster with 70% transparency
        show(raster_display, ax=ax, cmap=cmap, norm=norm, alpha=0.7, extent=raster_extent, zorder=1)

        # Clip and plot water bodies within raster extent
        lakes_in_extent = lakes.cx[raster_bounds.left:raster_bounds.right, raster_bounds.bottom:raster_bounds.top]
        lakes_in_extent.plot(ax=ax, color="lightblue", zorder=2, edgecolor="blue", linewidth=0.5)

        # Plot locations as points
        locations.plot(ax=ax, color="black", markersize=75, zorder=3)

        # Add city names to the plot
        for idx, row in locations.iterrows():
            ax.annotate(row['location'], xy=row["geometry"].coords[0], ha="center", va="top", xytext=(0, -6),
                        textcoords='offset points', fontsize=15)

        # Plot additional layers
        points.plot(ax=ax, color="red", markersize=30, zorder=4)
        s_bahn_lines.plot(ax=ax, color="red", linewidth=1, zorder=5)

        # Add north arrow
        ax.text(0.96, 0.92, "N", fontsize=20, weight="bold", ha="center", va="center", transform=ax.transAxes)
        arrow = FancyArrowPatch((0.96, 0.89), (0.96, 0.97), color="black", lw=2, arrowstyle="->", mutation_scale=20, transform=ax.transAxes)
        ax.add_patch(arrow)

        # Add a scale bar for 5 km
        scale_length = 5000  # 5 km in meters
        scale_bar_x = raster_bounds.left + 0.1 * (raster_bounds.right - raster_bounds.left)
        scale_bar_y = raster_bounds.bottom + 0.05 * (raster_bounds.top - raster_bounds.bottom)
        ax.add_patch(
            mpatches.Rectangle(
                (scale_bar_x, scale_bar_y),
                scale_length,
                0.02 * (raster_bounds.top - raster_bounds.bottom),
                color="black",
            )
        )
        ax.text(
            scale_bar_x + scale_length / 2,
            scale_bar_y - 0.02 * (raster_bounds.top - raster_bounds.bottom),
            "5 km",
            ha="center",
            va="center",
            fontsize=12,
            color="black",
        )

        # Set extent to raster bounds
        ax.set_xlim(raster_bounds.left, raster_bounds.right)
        ax.set_ylim(raster_bounds.bottom, raster_bounds.top)

        # Remove axes for a cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

        # Add legend outside the plot
        legend_elements = [
            Patch(facecolor="lightblue", edgecolor="blue", label="Water Bodies"),
            Patch(facecolor="red", edgecolor="red", label="S-Bahn"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Trainstations"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=3,
            frameon=False,
            fontsize=12,
            title="Legend",
            title_fontsize=14,
        )

        # Save the plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Plot saved at {output_path}")

def create_catchement_plot_time():
    # File paths
    raster_path = "data/catchment_pt/old_catchements/catchement.tif"
    cities_path = "data/manually_gathered_data/cities.shp"
    s_bahn_path = "data/Network/processed/split_s_bahn_lines.gpkg"
    lakes_path = "data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp"
    points_path = "data/Network/processed/points.gpkg"
    boundary_path = "data/_basic_data/innerboundary.shp"
    output_path = "plots/Catchement_Time.png"

    # Load the boundary shapefile
    boundary = gpd.read_file(boundary_path)

    # Ensure all layers are in the same CRS as the boundary
    def reproject_to_boundary(layer, boundary):
        if layer.crs != boundary.crs:
            return layer.to_crs(boundary.crs)
        return layer

    # Load and reproject vector layers
    cities = gpd.read_file(cities_path)
    cities = reproject_to_boundary(cities, boundary)
    cities['geometry'] = cities['geometry'].apply(make_valid)

    s_bahn = gpd.read_file(s_bahn_path)
    s_bahn = reproject_to_boundary(s_bahn, boundary)
    s_bahn['geometry'] = s_bahn['geometry'].apply(make_valid)

    lakes = gpd.read_file(lakes_path)
    lakes = reproject_to_boundary(lakes, boundary)
    lakes['geometry'] = lakes['geometry'].apply(make_valid)

    points = gpd.read_file(points_path)
    points = reproject_to_boundary(points, boundary)
    points['geometry'] = points['geometry'].apply(make_valid)

    # Clip vector layers to the boundary
    cities_clipped = gpd.clip(cities, boundary)
    s_bahn_clipped = gpd.clip(s_bahn, boundary)
    lakes_clipped = gpd.clip(lakes, boundary)
    points_clipped = gpd.clip(points, boundary)

    # Open the raster file and clip it to the boundary
    with rasterio.open(raster_path) as src:
        # Clip the raster to the boundary
        boundary_geometry = [boundary.geometry.unary_union]
        clipped_raster, clipped_transform = mask(src, boundary_geometry, crop=True, nodata=99999)

        # Update metadata for plotting
        clipped_meta = src.meta.copy()
        clipped_meta.update({
            "height": clipped_raster.shape[1],
            "width": clipped_raster.shape[2],
            "transform": clipped_transform
        })

    # Calculate the extent of the clipped raster
    extent = (
        clipped_transform[2],  # left
        clipped_transform[2] + clipped_transform[0] * clipped_meta["width"],  # right
        clipped_transform[5] + clipped_transform[4] * clipped_meta["height"],  # bottom
        clipped_transform[5],  # top
    )

    # Process the raster data
    masked_data = np.where(clipped_raster[0] == 99999, np.nan, clipped_raster[0])  # Mask 99999 as NaN
    masked_data = np.clip(masked_data, 0, 1500)  # Clip values above 1500

    # Define a colormap with white for low values, orange for middle, and red starting at values around 500
    cmap = LinearSegmentedColormap.from_list("custom_cmap", [(0, "white"), (500 / 1500, "orange"), (1, "red")])

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))
    raster_plot = ax.imshow(masked_data, cmap=cmap, extent=extent, zorder=1)
    cbar = plt.colorbar(raster_plot, ax=ax)
    cbar.set_label("Time to closest train station in seconds", fontsize=12)

    # Clip and plot water bodies within raster extent
    lakes_clipped.plot(ax=ax, color="lightblue", zorder=2, edgecolor="blue", linewidth=0.5)

    # Plot locations as points
    cities_clipped.plot(ax=ax, color="black", markersize=75, zorder=3)

    # Add city names to the plot
    for idx, row in cities_clipped.iterrows():
        ax.annotate(
            row['location'],
            xy=row["geometry"].coords[0],
            ha="center",
            va="top",
            xytext=(0, -6),
            textcoords='offset points',
            fontsize=15,
        )

    # Plot additional layers
    points_clipped.plot(ax=ax, color="red", markersize=30, zorder=4)
    s_bahn_clipped.plot(ax=ax, color="red", linewidth=1, zorder=5)

    # Add north arrow
    ax.text(0.96, 0.92, "N", fontsize=20, weight="bold", ha="center", va="center", transform=ax.transAxes)
    arrow = FancyArrowPatch((0.96, 0.89), (0.96, 0.97), color="black", lw=2, arrowstyle="->", mutation_scale=20, transform=ax.transAxes)
    ax.add_patch(arrow)

    # Add a scale bar
    scale_length_meters = 5000  # 5 km scale bar
    scale_bar_x = extent[0] + 0.1 * (extent[1] - extent[0])
    scale_bar_y = extent[2] + 0.05 * (extent[3] - extent[2])
    ax.add_patch(
        mpatches.Rectangle(
            (scale_bar_x, scale_bar_y),
            scale_length_meters,
            0.02 * (extent[3] - extent[2]),
            color="black",
        )
    )
    ax.text(
        scale_bar_x + scale_length_meters / 2,
        scale_bar_y - 0.02 * (extent[3] - extent[2]),
        "5 km",
        ha="center",
        va="center",
        fontsize=10,
        color="black",
    )

    # Remove axes for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    # Add legend outside the plot
    legend_elements = [
        Patch(facecolor="lightblue", edgecolor="blue", label="Water Bodies"),
        Patch(facecolor="red", edgecolor="red", label="S-Bahn"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Trainstations"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        frameon=False,
        fontsize=12,
        title="Legend",
        title_fontsize=14,
    )

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved at {output_path}")


def plot_developments_expand_by_one_station():
    # File paths
    trainstations_path = "data/Network/processed/points.gpkg"
    lakes_path = "data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp"
    s_bahn_lines_path = "data/Network/processed/split_s_bahn_lines.gpkg"
    developments_path = "data/costs/total_costs_with_geometry.gpkg"
    endnodes_path = "data/Network/processed/endnodes.gpkg"
    boundary_path = "data/_basic_data/outerboundary.shp"
    output_path = "plots/developments.png"

    # Load data
    trainstations = gpd.read_file(trainstations_path)
    trainstations["geometry"] = trainstations["geometry"].apply(make_valid)

    lakes = gpd.read_file(lakes_path)
    lakes["geometry"] = lakes["geometry"].apply(make_valid)

    s_bahn_lines = gpd.read_file(s_bahn_lines_path)
    s_bahn_lines["geometry"] = s_bahn_lines["geometry"].apply(make_valid)

    developments = gpd.read_file(developments_path)
    developments["development"] = developments["development"].str.replace("Development_", "", regex=True)
    developments = developments[developments["development"].astype(int) < settings.dev_id_start_new_direct_connections]
    developments["geometry"] = developments["geometry"].apply(make_valid)

    endnodes = gpd.read_file(endnodes_path)
    endnodes["geometry"] = endnodes["geometry"].apply(make_valid)

    boundary = gpd.read_file(boundary_path)
    boundary["geometry"] = boundary["geometry"].apply(make_valid)


    # Ensure all layers use the same CRS
    layers = [trainstations, lakes, s_bahn_lines, developments, endnodes, boundary]
    for layer in layers:
        if layer.crs != "epsg:2056":
            layer.to_crs("epsg:2056", inplace=True)

    # Clip data to the boundary extent
    clipped_layers = {
        "trainstations": gpd.clip(trainstations, boundary),
        "lakes": gpd.clip(lakes, boundary),
        "s_bahn_lines": gpd.clip(s_bahn_lines, boundary),
        "developments": developments,#gpd.clip(developments, boundary),
        "endnodes": gpd.clip(endnodes, boundary)
    }

    # Filter trainstations for specific names
    station_names = ["Aathal", "Wetzikon", "Uster", "Schwerzenbach", "Rüti ZH", "Pfäffikon ZH", 
                     "Nänikon-Greifensee", "Kempten", "Illnau", "Hinwil", "Fehraltorf", "Effretikon", 
                     "Dübendorf", "Dietlikon", "Bubikon","Saland","Bauma", "Esslingen", "Forch", "Männedorf", "Küsnacht ZH", "Glattbrugg","Kloten", "Kemptthal", "Zürich Rehalp", "Herrliberg-Feldmeilen" , "Horgen", "Thalwil", "Wila", "Schwerzenbach" ]
    labeled_trainstations = clipped_layers["trainstations"][clipped_layers["trainstations"]["NAME"].isin(station_names)]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot lakes (in the back)
    clipped_layers["lakes"].plot(ax=ax, color="lightblue", edgecolor="blue", zorder=1)

    # Plot S-Bahn lines (above lakes)
    clipped_layers["s_bahn_lines"].plot(ax=ax, color="red", linewidth=1.5, zorder=2)

    # Plot all trainstations (above railway lines)
    clipped_layers["trainstations"].plot(ax=ax, color="red", markersize=30, zorder=3)

    # Plot endnodes (above trainstations)
    clipped_layers["endnodes"].plot(ax=ax, color="orange", markersize=250, zorder=8)

    # Korrigierter Code für die Farbzuweisung
    # Korrigierter Code für die Farbzuweisung
    def generate_color_scheme(n_developments):
        # Use different colormaps based on number of developments
        if n_developments <= 10:
            cmap = plt.cm.tab10
        elif n_developments <= 20:
            cmap = plt.cm.tab20
        else:
            cmap = plt.cm.nipy_spectral

        # Generate colors
        colors = [mcolors.rgb2hex(cmap(i / n_developments)) for i in range(n_developments)]

        # Create development color dictionary - ohne "Development_" Präfix
        return {str(i + 1): color for i, color in enumerate(colors)}

    # Generate color scheme for developments
    unique_developments = clipped_layers["developments"]["development"].unique()
    development_colors = generate_color_scheme(len(unique_developments))

    # Plot developments (above endnodes) with colors
    for idx, row in clipped_layers["developments"].iterrows():
        dev_id = row["development"]  # Jetzt ohne "Development_" Präfix
        color = development_colors.get(str(int(dev_id) - 100000) if dev_id.isdigit() and int(dev_id) > 100000 else dev_id, "black")
        ax.plot(*row.geometry.xy, color=color, linewidth=4, zorder=5)

        # Add station labels for specific names (on top of all other layers)
    for idx, row in labeled_trainstations.iterrows():
            ax.annotate(row["NAME"], xy=row.geometry.coords[0], ha="center", va="top", xytext=(0, -10),
                        textcoords="offset points", fontsize=12, color="black", zorder=7)

    # Add north arrow
    add_north_arrow(ax, scale=.75, xlim_pos=.9025, ylim_pos=.835, color='#000', text_scaler=4, text_yT=-1.25)

    # Add a scale bar
    scalebar = ScaleBar(dx=1, units="m", location="lower left", scale_loc="bottom")
    ax.add_artist(scalebar)

    # Create legend
    legend_elements = [
            Patch(facecolor="lightblue", edgecolor="blue", label="Water Bodies"),
            Line2D([0], [0], color="red", marker="o", markersize=10, label="Trainstations"),
            Line2D([0], [0], color="orange", marker="o", markersize=10, label="Endnodes"),
            Line2D([0], [0], color="red", lw=1.5, label="S-Bahn"),
    ]

    # Auch die Legende entsprechend anpassen
    for dev, color in development_colors.items():
        legend_elements.append(Line2D([0], [0], color=color, lw=4, label=f"Development {dev}"))

    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=2,
        frameon=False,
        fontsize=12,
        title="Legend",
        title_fontsize=14
    )

    # Add a terrain basemap
    #cx.add_basemap(ax, crs=trainstations.crs, source=cx.providers.Stamen.Terrain)

    # Set extent to the boundary
    ax.set_xlim(boundary.total_bounds[0], boundary.total_bounds[2])
    ax.set_ylim(boundary.total_bounds[1], boundary.total_bounds[3])

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved at {output_path}")

def create_and_save_plots(df, plot_directory="plots"):
    """
    Creates and saves enhanced boxplots and strip plots of monetized savings,
    total net benefits, and CBA ratio by development.
    """
    # Ensure the plot directory exists
    os.makedirs(plot_directory, exist_ok=True)

    # Rename the column 'ID_new' to 'Scenario'
    df.rename(columns={'ID_new': 'Scenario'}, inplace=True)

    # Ensure all monetized savings are positive
    df['monetized_savings_total'] = df['monetized_savings_total'].abs()

    # Calculate total net benefits and CBA ratio
    df['total_costs'] = df['TotalConstructionCost'] + df['TotalMaintenanceCost'] + df['TotalUncoveredOperatingCost']
    df['total_net_benefit'] = df['monetized_savings_total'] - df['total_costs']
    df['cba_ratio'] = df['monetized_savings_total'] / df['total_costs']

    # Map consistent colors based on scenario type
    def assign_color_by_type(scenario):
        if "Urban" in scenario:
            return 'orange'
        elif "Equal" in scenario:
            return 'blue'
        elif "Rural" in scenario:
            return 'green'
        else:
            return 'gray'

    df['Color'] = df['scenario'].apply(assign_color_by_type)

    # First create plots for developments < 101000
    small_dev_data = df[df['development'] < settings.dev_id_start_new_direct_connections]

    # MONETIZED SAVINGS PLOTS

    # Boxplot for small developments - showing only outliers, not individual points
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=small_dev_data,
        x='development',
        y='monetized_savings_total',
        palette="Set2",
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 6,
        },
        fliersize=3,  # Smaller outlier points
        showfliers=True  # Show outliers
    )
    plt.xlabel('Development', fontsize=12)
    plt.ylabel('Monetized Savings in CHF', fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    handles = [plt.Line2D([0], [0], marker='o', color='black', label='Mean', markersize=6)]
    plt.legend(handles=handles, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_directory, "enhanced_boxplot_small_developments.png"))
    plt.close()

    # Strip plot for small developments with smaller point size
    plt.figure(figsize=(14, 8))
    sns.stripplot(
        data=small_dev_data,
        x='development',
        y='monetized_savings_total',
        hue='scenario',
        palette=dict(zip(small_dev_data['scenario'], small_dev_data['Color'])),
        jitter=True,
        dodge=True,
        size=3,  # Much smaller point size
        alpha=0.7
    )
    plt.xlabel('Development', fontsize=12)
    plt.ylabel('Monetized Savings in CHF', fontsize=12)
    plt.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_directory, "strip_plot_with_scenarios_small_developments.png"))
    plt.close()

    # NET BENEFIT BOXPLOT FOR SMALL DEVELOPMENTS
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=small_dev_data,
        x='development',
        y='total_net_benefit',
        palette="Set2",
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 6,
        },
        fliersize=3,
        showfliers=True
    )
    plt.xlabel('Development', fontsize=12)
    plt.ylabel('Total Net Benefit in CHF', fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Add horizontal line at y=0 to highlight positive vs negative net benefits
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.5)
    handles = [plt.Line2D([0], [0], marker='o', color='black', label='Mean', markersize=6)]
    plt.legend(handles=handles, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_directory, "net_benefit_boxplot_small_developments.png"))
    plt.close()

    # CBA RATIO BOXPLOT FOR SMALL DEVELOPMENTS
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=small_dev_data,
        x='development',
        y='cba_ratio',
        palette="Set2",
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 6,
        },
        fliersize=3,
        showfliers=True
    )
    plt.xlabel('Development', fontsize=12)
    plt.ylabel('CBA Ratio (Benefits/Costs)', fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Add horizontal line at y=1 to highlight beneficial vs non-beneficial ratio
    plt.axhline(y=1, color='red', linestyle='-', alpha=0.5)
    handles = [plt.Line2D([0], [0], marker='o', color='black', label='Mean', markersize=6)]
    plt.legend(handles=handles, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_directory, "cba_ratio_boxplot_small_developments.png"))
    plt.close()

    # Create separate plots for every 15 developments (only for developments >= 101000)
    large_dev_data = df[df['development'] >= settings.dev_id_start_new_direct_connections]
    developments = sorted(large_dev_data['development'].unique())
    num_plots = len(developments) // 15 + (1 if len(developments) % 15 > 0 else 0)

    for i in range(num_plots):
        start_idx = i * 15
        end_idx = min((i + 1) * 15, len(developments))
        plot_developments = developments[start_idx:end_idx]

        plot_data = large_dev_data[large_dev_data['development'].isin(plot_developments)]

        # MONETIZED SAVINGS PLOTS

        # Boxplot for each group - showing only outliers
        plt.figure(figsize=(14, 8))
        sns.boxplot(
            data=plot_data,
            x='development',
            y='monetized_savings_total',
            palette="Set2",
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": 6,
            },
            fliersize=3,  # Smaller outlier points
            showfliers=True  # Show outliers
        )
        plt.xlabel('Development', fontsize=12)
        plt.ylabel('Monetized Savings in CHF', fontsize=12)
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        handles = [plt.Line2D([0], [0], marker='o', color='black', label='Mean', markersize=6)]
        plt.legend(handles=handles, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, f"enhanced_boxplot_group_{i + 1}.png"))
        plt.close()

        # Strip plot for each group with smaller point size
        plt.figure(figsize=(14, 8))
        sns.stripplot(
            data=plot_data,
            x='development',
            y='monetized_savings_total',
            hue='scenario',
            palette=dict(zip(plot_data['scenario'], plot_data['Color'])),
            jitter=True,
            dodge=True,
            size=3,  # Much smaller point size
            alpha=0.7
        )
        plt.xlabel('Development', fontsize=12)
        plt.ylabel('Monetized Savings in CHF', fontsize=12)
        plt.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, f"strip_plot_with_scenarios_group_{i + 1}.png"))
        plt.close()

        # NET BENEFIT BOXPLOT FOR GROUP
        plt.figure(figsize=(14, 8))
        sns.boxplot(
            data=plot_data,
            x='development',
            y='total_net_benefit',
            palette="Set2",
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": 6,
            },
            fliersize=3,
            showfliers=True
        )
        plt.xlabel('Development', fontsize=12)
        plt.ylabel('Total Net Benefit in CHF', fontsize=12)
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='-', alpha=0.5)
        handles = [plt.Line2D([0], [0], marker='o', color='black', label='Mean', markersize=6)]
        plt.legend(handles=handles, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, f"net_benefit_boxplot_group_{i + 1}.png"))
        plt.close()

        # CBA RATIO BOXPLOT FOR GROUP
        plt.figure(figsize=(14, 8))
        sns.boxplot(
            data=plot_data,
            x='development',
            y='cba_ratio',
            palette="Set2",
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "black",
                "markeredgecolor": "black",
                "markersize": 6,
            },
            fliersize=3,
            showfliers=True
        )
        plt.xlabel('Development', fontsize=12)
        plt.ylabel('CBA Ratio (Benefits/Costs)', fontsize=12)
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.axhline(y=1, color='red', linestyle='-', alpha=0.5)
        handles = [plt.Line2D([0], [0], marker='o', color='black', label='Mean', markersize=6)]
        plt.legend(handles=handles, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, f"cba_ratio_boxplot_group_{i + 1}.png"))
        plt.close()

    # Add stacked bar plots with costs below and savings above the middle axis
    # For small developments first
    plt.figure(figsize=(14, 8))

    # Prepare data for small developments
    small_dev_summary = small_dev_data.groupby('development').agg({
        'TotalConstructionCost': 'mean',
        'TotalMaintenanceCost': 'mean',
        'TotalUncoveredOperatingCost': 'mean',
        'monetized_savings_total': 'mean'
    }).reset_index()

    # Calculate total width for all bars
    dev_count = len(small_dev_summary)
    bar_width = 0.8
    x_positions = np.arange(dev_count)

    # Plot costs as negative values (below axis)
    plt.bar(x_positions, -small_dev_summary['TotalConstructionCost'],
            width=bar_width, color='#ff9999', label='Baukosten')
    plt.bar(x_positions, -small_dev_summary['TotalMaintenanceCost'],
            width=bar_width, bottom=-small_dev_summary['TotalConstructionCost'],
            color='#ffcc99', label='Wartungskosten')
    plt.bar(x_positions, -small_dev_summary['TotalUncoveredOperatingCost'],
            width=bar_width,
            bottom=-(small_dev_summary['TotalConstructionCost'] + small_dev_summary['TotalMaintenanceCost']),
            color='#ffff99', label='Betriebskosten')

    # Plot savings as positive values (above axis)
    plt.bar(x_positions, small_dev_summary['monetized_savings_total'],
            width=bar_width, color='#99ccff', label='Monetarisierte Einsparungen')

    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-')

    # Set x-axis labels to development IDs
    plt.xticks(x_positions, small_dev_summary['development'], rotation=90)
    plt.xlabel('Entwicklung', fontsize=12)
    plt.ylabel('CHF', fontsize=12)
    plt.title('Kosten und Einsparungen für kleine Entwicklungen', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_directory, "costs_savings_comparison_small_developments.png"))
    plt.close()

    # Now create similar plots for groups of larger developments
    for i in range(num_plots):
        start_idx = i * 15
        end_idx = min((i + 1) * 15, len(developments))
        plot_developments = developments[start_idx:end_idx]

        plot_data = large_dev_data[large_dev_data['development'].isin(plot_developments)]

        # Prepare data summary
        plot_summary = plot_data.groupby('development').agg({
            'TotalConstructionCost': 'mean',
            'TotalMaintenanceCost': 'mean',
            'TotalUncoveredOperatingCost': 'mean',
            'monetized_savings_total': 'mean'
        }).reset_index()

        plt.figure(figsize=(14, 8))

        # Calculate positions
        dev_count = len(plot_summary)
        x_positions = np.arange(dev_count)

        # Plot costs as negative values (below axis)
        plt.bar(x_positions, -plot_summary['TotalConstructionCost'],
                width=bar_width, color='#ff9999', label='Baukosten')
        plt.bar(x_positions, -plot_summary['TotalMaintenanceCost'],
                width=bar_width, bottom=-plot_summary['TotalConstructionCost'],
                color='#ffcc99', label='Wartungskosten')
        plt.bar(x_positions, -plot_summary['TotalUncoveredOperatingCost'],
                width=bar_width,
                bottom=-(plot_summary['TotalConstructionCost'] + plot_summary['TotalMaintenanceCost']),
                color='#ffff99', label='Betriebskosten')

        # Plot savings as positive values (above axis)
        plt.bar(x_positions, plot_summary['monetized_savings_total'],
                width=bar_width, color='#99ccff', label='Monetarisierte Einsparungen')

        # Add horizontal line at y=0
        plt.axhline(y=0, color='black', linestyle='-')

        # Set x-axis labels to development IDs
        plt.xticks(x_positions, plot_summary['development'], rotation=90)
        plt.xlabel('Entwicklung', fontsize=12)
        plt.ylabel('CHF', fontsize=12)
        plt.title(f'Kosten und Einsparungen für Entwicklungsgruppe {i + 1}', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend(title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_directory, f"costs_savings_comparison_group_{i + 1}.png"))
        plt.close()



def plot_catchment_and_distributions(
    s_bahn_lines_path,
    water_bodies_path,
    catchment_raster_path,
    communal_borders_path,
    population_raster_path,
    employment_raster_path,
    extent_path,
    output_dir="plots/"
):
    extent = gpd.read_file(extent_path)
    extent["geometry"] = extent["geometry"].apply(make_valid)
    extent_bounds = extent.total_bounds  # Get the bounding box as [xmin, ymin, xmax, ymax]

    # Load vector data
    s_bahn_lines = gpd.read_file(s_bahn_lines_path)
    s_bahn_lines["geometry"] = s_bahn_lines["geometry"].apply(make_valid)

    water_bodies = gpd.read_file(water_bodies_path)
    water_bodies["geometry"] = water_bodies["geometry"].apply(make_valid)

    communal_borders = gpd.read_file(communal_borders_path)
    communal_borders["geometry"] = communal_borders["geometry"].apply(make_valid)

    # Clip vector data to the extent
    s_bahn_lines = gpd.clip(s_bahn_lines, extent)
    water_bodies = gpd.clip(water_bodies, extent)
    communal_borders = gpd.clip(communal_borders, extent)

    # Load raster data and crop to the extent
    with rasterio.open(population_raster_path) as pop_src:
        pop_raster, pop_transform = mask(pop_src, [box(*extent_bounds)], crop=True)

    with rasterio.open(employment_raster_path) as empl_src:
        empl_raster, empl_transform = mask(empl_src, [box(*extent_bounds)], crop=True)

    with rasterio.open(catchment_raster_path) as catchment_src:
        catchment_raster, catchment_transform = mask(catchment_src, [box(*extent_bounds)], crop=True)

    # Handle -1 (NoData) values in the catchment raster
    catchment_raster = np.where(catchment_raster == -1, np.nan, catchment_raster)

    # Create the figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    # Shared basemap function
    def plot_basemap(ax, background_color="white"):
        # Set background color
        ax.set_facecolor(background_color)
        # Plot water bodies
        water_bodies.plot(ax=ax, color="lightblue", edgecolor="none", zorder=1)
        # Plot S-Bahn lines with thicker lines
        s_bahn_lines.plot(ax=ax, color="red", linewidth=2.5, zorder=2)

    # Custom colormap for inverted raster cells
    def create_white_cmap():
        return ListedColormap(["white", "black"])

    # Plot 1: Population Raster (Top Left)
    ax1 = axes[0, 0]
    plot_basemap(ax1, background_color="black")
    pop_raster_data = np.where(pop_raster[0] > 0, 1, 0)  # Binary: 1 for data, 0 for NaN
    white_cmap = create_white_cmap()
    show(pop_raster_data, transform=pop_transform, ax=ax1, cmap=white_cmap)
    ax1.set_title("Population Distribution",  fontweight="bold", fontsize=16)  # Larger title
    ax1.set_xlim(extent_bounds[0], extent_bounds[2])
    ax1.set_ylim(extent_bounds[1], extent_bounds[3])

    # Plot 2: Employment Raster (Top Right)
    ax2 = axes[0, 1]
    plot_basemap(ax2, background_color="black")
    empl_raster_data = np.where(empl_raster[0] > 0, 1, 0)  # Binary: 1 for data, 0 for NaN
    show(empl_raster_data, transform=empl_transform, ax=ax2, cmap=white_cmap)
    ax2.set_title("Employment Distribution", fontweight="bold", fontsize=16)  # Larger title
    ax2.set_xlim(extent_bounds[0], extent_bounds[2])
    ax2.set_ylim(extent_bounds[1], extent_bounds[3])

    # Plot 3: Communal Borders (Bottom Left)
    ax3 = axes[1, 0]
    plot_basemap(ax3, background_color="white")
    communal_borders.plot(ax=ax3, color="none", edgecolor="black", linewidth=0.7, zorder=3)
    ax3.set_title("Communal Borders", fontweight="bold", fontsize=16)  # Larger and bold title
    ax3.set_xlim(extent_bounds[0], extent_bounds[2])
    ax3.set_ylim(extent_bounds[1], extent_bounds[3])

    # Plot 4: Catchment Raster (Bottom Right)
    ax4 = axes[1, 1]
    plot_basemap(ax4, background_color="white")
    unique_values = np.unique(catchment_raster)
    unique_values = unique_values[unique_values != -1]  # Remove NoData values
    cmap = ListedColormap(plt.cm.tab10.colors[:len(unique_values)])
    norm = Normalize(vmin=np.nanmin(catchment_raster), vmax=np.nanmax(catchment_raster))
    show(catchment_raster[0], transform=catchment_transform, ax=ax4, cmap=cmap, norm=norm)
    ax4.set_title("Catchment Areas", fontweight="bold", fontsize=16)  # Larger title
    ax4.set_xlim(extent_bounds[0], extent_bounds[2])
    ax4.set_ylim(extent_bounds[1], extent_bounds[3])

    # Save the figure
    plt.tight_layout()
    output_path = f"{output_dir}catchment_and_distributions_larger_titles.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plots saved to {output_path}")


def plot_costs_benefits_example(cost_benefit_df):
    """
    Create a scientific-looking bar chart of costs and benefits over years
    for the development with the highest benefit in year 1.

    Args:
        cost_benefit_df: DataFrame with discounted costs and benefits
    """
    # Finde das niedrigste Jahr im DataFrame
    min_year = cost_benefit_df.index.get_level_values('year').min()
    # Extrahiere Daten für das niedrigste Jahr
    year1_benefits = cost_benefit_df.xs(min_year, level='year')
    max_benefit_dev = year1_benefits.groupby('development')['benefit'].mean().idxmax()

    # Get data for the selected development
    dev_data = cost_benefit_df.xs(max_benefit_dev, level='development')

    # Calculate mean values across scenarios for each year
    plot_data = dev_data.groupby('year').mean()

    # Create a figure with two subplots sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.1})

    # Set style for scientific plot
    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'serif'

    width = 0.8

    # Top subplot for costs (using appropriate scale)
    # Plot stacked bars for costs
    costs_bars1 = ax1.bar(plot_data.index, plot_data['const_cost'], width,
                          label='Construction Costs', color='#1f77b4', alpha=0.8)
    costs_bars2 = ax1.bar(plot_data.index, plot_data['maint_cost'], width,
                          bottom=plot_data['const_cost'], label='Maintenance Costs',
                          color='#ff7f0e', alpha=0.8)

    # Add the operating costs as a third stacked bar
    sum_prev_costs = plot_data['const_cost'] + plot_data['maint_cost']
    costs_bars3 = ax1.bar(plot_data.index, plot_data['uncovered_op_cost'], width,
                          bottom=sum_prev_costs, label='Operating Costs',
                          color='#d62728', alpha=0.8)

    # Customize top subplot
    ax1.set_ylabel('Costs (CHF)', fontsize=12)
    ax1.set_title(f'Discounted Costs and Benefits Over Time\nDevelopment: {max_benefit_dev}',
                  fontsize=14, pad=20)
    ax1.legend(loc='upper right', frameon=True, edgecolor='black')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.set_yscale('log')

    # If zero or negative values exist, handle them for log scale
    min_positive_cost = plot_data[['const_cost', 'maint_cost', 'uncovered_op_cost']].stack().replace(0,
                                                                                                     float('nan')).min()
    ax1.set_ylim(bottom=min_positive_cost * 0.1 if pd.notna(min_positive_cost) else 1)

    # Bottom subplot for benefits (using linear scale)
    benefit_bars = ax2.bar(plot_data.index, plot_data['benefit'], width,
                           label='Benefits', color='#2ca02c', alpha=0.8)

    # Customize bottom subplot
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Benefits (CHF)', fontsize=12)
    ax2.legend(loc='upper right', frameon=True, edgecolor='black')
    ax2.grid(True, which="both", ls="-", alpha=0.2)

    # Auto-adjust y-limits for benefits based on data
    if plot_data['benefit'].max() > 0:
        ax2.set_ylim(bottom=0, top=plot_data['benefit'].max() * 1.2)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)

    # Create filename with development name
    filename = f'plots/discounted_costs_benefits_{max_benefit_dev}.png'

    # Delete the file if it already exists
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Deleted existing file: {filename}")

    # Save the plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved for development {max_benefit_dev} as {filename}")


def plot_lines_to_network(points_gdf,lines_gdf):
    points_gdf.plot(marker='*', color='green', markersize=5)
    base = lines_gdf.plot(edgecolor='black')
    points_gdf.plot(ax=base, marker='o', color='red', markersize=5)
    plt.savefig(r"plot\predict\230822_network-generation.png", dpi=300)
    return None


def plot_graph(graph, positions, highlight_centers=True, missing_links=None, directory='data/plots',polygon=None):
    """
    Plot the railway network graph with optional highlighting of center nodes and missing connections.

    Args:
        graph (networkx.Graph): The railway network graph
        positions (dict): Dictionary mapping node IDs to (x,y) coordinates
        highlight_centers (bool): Whether to highlight center nodes
        missing_connections (list): List of missing connections to highlight
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

    if missing_links:
        missing_connections = []
        for center in missing_links:
            missing_connections.extend(center['missing_connections'])
        for conn in missing_connections:
            node1, node2 = conn['nodes']
            if node1 in positions and node2 in positions:
                plt.plot([positions[node1][0], positions[node2][0]],
                         [positions[node1][1], positions[node2][1]],
                         'r--', linewidth=2, alpha=0.7)

    if polygon:
        x, y = polygon.exterior.xy
        plt.plot(x, y, 'b-', linewidth=2, alpha=0.8)

    # Add legend
    if highlight_centers:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Center Node'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Border Node'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='End Station'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Regular Node')
        ]
        if missing_links:
            legend_elements.append(Line2D([0], [0], color='r', linestyle='--', label='Missing Connection'))
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.title("Railway Network Analysis")
    plt.axis('on')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'network_graph.png'),
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.2,
                format='png')


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


def plot_railway_lines_with_offset(G, pos, new_railway_lines, output_file='proposed_railway_lines.png'):
    """
    Creates a visualization of railway lines with offset paths to prevent overlapping.

    Args:
        G (networkx.Graph): Input graph
        pos (dict): Node positions
        new_railway_lines (list): List of new railway lines to visualize
        output_file (str): Output file path for the visualization
    """
    plt.figure(figsize=(20, 16), dpi=300)

    # Draw the existing network
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.5, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=30, alpha=0.4)

    # Create a dictionary to track how many lines pass between each pair of nodes
    edge_count = {}

    # First pass: count how many lines use each edge
    for line in new_railway_lines:
        path = line['path']
        for j in range(len(path) - 1):
            if path[j] in pos and path[j + 1] in pos:
                edge = tuple(sorted([path[j], path[j + 1]]))
                edge_count[edge] = edge_count.get(edge, 0) + 1

    # Draw the new railway lines with different colors and offsets
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    legend_handles = []

    for i, line in enumerate(new_railway_lines):
        path = line['path']
        color = colors[i % len(colors)]
        label = line['name']

        # Draw the path segments with offsets
        for j in range(len(path) - 1):
            if path[j] in pos and path[j + 1] in pos:
                start_pos = pos[path[j]]
                end_pos = pos[path[j + 1]]

                # Calculate perpendicular offset
                dx = end_pos[0] - start_pos[0]
                dy = end_pos[1] - start_pos[1]
                length = np.sqrt(dx * dx + dy * dy)

                if length > 0:
                    # Calculate perpendicular vector
                    perpx = -dy / length
                    perpy = dx / length

                    # Determine offset based on line index and total lines
                    edge = tuple(sorted([path[j], path[j + 1]]))
                    total_lines = edge_count[edge]
                    line_index = i % total_lines

                    # Calculate offset distance (adjust multiplier as needed)
                    offset = (line_index - (total_lines - 1) / 2) * 100  # Adjust 100 to change spacing

                    # Apply offset to coordinates
                    start_offset = (
                        start_pos[0] + perpx * offset,
                        start_pos[1] + perpy * offset
                    )
                    end_offset = (
                        end_pos[0] + perpx * offset,
                        end_pos[1] + perpy * offset
                    )

                    # Draw the offset line segment
                    line_handle, = plt.plot(
                        [start_offset[0], end_offset[0]],
                        [start_offset[1], end_offset[1]],
                        color=color,
                        linewidth=2,
                        label=label if j == 0 else "_nolegend_"
                    )

                    if j == 0:
                        legend_handles.append(line_handle)

    # Add legend and title
    plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.title("Proposed New Railway Lines")
    plt.grid(True)
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_file,
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.2,
                format='png')
    plt.close()


def plot_missing_connection_lines(G, pos, new_railway_lines, connection_nodes, output_file):
    """
    Creates a visualization of railway lines for a specific missing connection.
    Railway lines are drawn with offsets to prevent overlapping when they share the same path segments.

    Args:
        G (networkx.Graph): Input graph
        pos (dict): Node positions
        new_railway_lines (list): List of new railway lines to visualize
        connection_nodes (tuple): The missing connection node pair (node1, node2)
        output_file (str): Output file path for the visualization
    """
    # Filter lines that belong to this specific missing connection
    connection_lines = [line for line in new_railway_lines
                        if line['original_missing_connection']['nodes'] == connection_nodes]

    if not connection_lines:
        print(f"No railway lines found for connection {connection_nodes}")
        return

    node1, node2 = connection_nodes
    station1 = G.nodes[node1].get('station_name', f"Node {node1}")
    station2 = G.nodes[node2].get('station_name', f"Node {node2}")

    plt.figure(figsize=(20, 16), dpi=300)

    # Draw the existing network with reduced opacity
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.5, alpha=0.3)

    # Draw nodes with different sizes and colors
    node_colors = []
    node_sizes = []
    node_labels = {}

    for node in G.nodes():
        # Get station name for label if it's an important node
        station_name = G.nodes[node].get('station_name', '')

        # Assign colors and sizes based on node type
        if node in connection_nodes:  # Highlight the missing connection nodes
            node_colors.append('red')
            node_sizes.append(150)
            node_labels[node] = station_name
        elif G.nodes[node].get('type') == 'center':
            node_colors.append('orange')
            node_sizes.append(80)
            node_labels[node] = station_name
        elif G.nodes[node].get('end_station', False):
            node_colors.append('green')
            node_sizes.append(100)
            node_labels[node] = station_name
        elif any(node in line['path'] for line in connection_lines):  # Nodes in the new lines
            node_colors.append('blue')
            node_sizes.append(80)
            node_labels[node] = station_name
        else:
            node_colors.append('lightgray')
            node_sizes.append(30)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7)

    # Add labels for important nodes
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)

    # Create a dictionary to track how many lines pass between each pair of nodes
    edge_count = {}

    # First pass: count how many lines use each edge
    for line in connection_lines:
        path = line['path']
        for j in range(len(path) - 1):
            if path[j] in pos and path[j + 1] in pos:
                edge = tuple(sorted([path[j], path[j + 1]]))
                edge_count[edge] = edge_count.get(edge, 0) + 1

    # Draw the new railway lines with different colors
    colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink']
    legend_handles = []

    # First, draw the missing connection as a dashed line
    if node1 in pos and node2 in pos:
        missing_handle, = plt.plot([pos[node1][0], pos[node2][0]],
                                  [pos[node1][1], pos[node2][1]],
                                  'r--', linewidth=2.5, alpha=0.8,
                                  label=f"Missing Connection: {station1} - {station2}")
        legend_handles.append(missing_handle)

    # Then draw each proposed railway line with offset
    for i, line in enumerate(connection_lines):
        path = line['path']
        color = colors[i % len(colors)]  # Use colors other than red
        label = f"{line['name']}: {line['endpoints']['start']['station']} - {line['endpoints']['end']['station']}"
        line_segments = []

        # Draw the path segments with offsets to prevent overlapping
        for j in range(len(path) - 1):
            if path[j] in pos and path[j + 1] in pos:
                start_pos = pos[path[j]]
                end_pos = pos[path[j + 1]]

                # Calculate perpendicular offset
                dx = end_pos[0] - start_pos[0]
                dy = end_pos[1] - start_pos[1]
                length = np.sqrt(dx * dx + dy * dy)

                if length > 0:
                    # Calculate perpendicular vector
                    perpx = -dy / length
                    perpy = dx / length

                    # Determine offset based on line index and total lines for this edge
                    edge = tuple(sorted([path[j], path[j + 1]]))
                    total_lines = edge_count[edge]

                    # Determine this line's position among those using this edge
                    # Find position of current line among all lines using this edge
                    line_position = 0
                    for k, other_line in enumerate(connection_lines):
                        if k == i:  # Found our line
                            break
                        # Check if other line uses this edge
                        other_path = other_line['path']
                        for m in range(len(other_path) - 1):
                            if tuple(sorted([other_path[m], other_path[m + 1]])) == edge:
                                line_position += 1
                                break

                    # Calculate offset - distribute lines evenly
                    if total_lines > 1:
                        offset = (line_position - (total_lines - 1) / 2) * 150  # Increased spacing
                    else:
                        offset = 0

                    # Apply offset to coordinates
                    start_offset = (
                        start_pos[0] + perpx * offset,
                        start_pos[1] + perpy * offset
                    )
                    end_offset = (
                        end_pos[0] + perpx * offset,
                        end_pos[1] + perpy * offset
                    )

                    # Draw the offset line segment
                    line_segment, = plt.plot(
                        [start_offset[0], end_offset[0]],
                        [start_offset[1], end_offset[1]],
                        color=color,
                        linewidth=3,
                        label="_nolegend_"  # Don't add individual segments to legend
                    )

                    line_segments.append(line_segment)

        # Add a legend entry for this line
        if line_segments:
            # Create a custom legend entry with the proper color
            from matplotlib.lines import Line2D
            legend_line = Line2D([0], [0], color=color, lw=3, label=label)
            legend_handles.append(legend_line)

    # Add legend and title with larger font size and better positioning
    plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1),
               fontsize=10, framealpha=0.8)
    plt.title(f"Proposed Railway Lines for Missing Connection: {station1} - {station2}",
              fontsize=14, pad=20)

    # Add a grid and other formatting
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Add a scale bar or axis labels if appropriate for geographic data
    if all(isinstance(pos[node], tuple) and len(pos[node]) == 2 for node in pos):
        plt.xlabel('X Coordinate (m)', fontsize=10)
        plt.ylabel('Y Coordinate (m)', fontsize=10)

    # Save the figure
    plt.savefig(output_file,
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.2,
                format='png')
    plt.close()


def plot_lines_for_each_missing_connection(new_railway_lines, G, pos, plots_dir):

    os.makedirs(plots_dir, exist_ok=True)
    # Create a set of all unique missing connections from the new railway lines
    unique_connections = set()
    for line in new_railway_lines:
        conn = line['original_missing_connection']['nodes']
        unique_connections.add(conn)
    print(f"\nCreating individual plots for {len(unique_connections)} missing connections...")
    # Create individual plots for each missing connection
    for i, connection in enumerate(unique_connections):
        node1, node2 = connection
        station1 = G.nodes[node1].get('station_name', f"Node {node1}")
        station2 = G.nodes[node2].get('station_name', f"Node {node2}")

        # Create a filename based on the connection
        filename = f"{plots_dir}/connection_{i + 1}_{station1.replace(' ', '_')}_to_{station2.replace(' ', '_')}.png"

        print(f"  Creating plot for missing connection: {station1} - {station2}")
        plot_missing_connection_lines(G, pos, new_railway_lines, connection, filename)


def plot_cumulative_cost_distribution(df, output_path="plot/cumulative_cost_distribution.png"):
    """
    Erstellt eine kumulative Wahrscheinlichkeitsverteilung der Kosten für die 5 Entwicklungen
    mit dem höchsten Nutzen.

    Args:
        df: DataFrame mit den Kostendaten (monetized_savings_total)
        output_path: Pfad zum Speichern der Abbildung

    """
    # Berechnung des Mittelwerts für jede Entwicklung
    mean_by_dev = df.groupby('development')['monetized_savings_total'].mean().reset_index()

    # Sortieren nach Mittelwert und Auswahl der 5 Entwicklungen mit dem höchsten Nutzen
    top_5_devs = mean_by_dev.sort_values(by=['monetized_savings_total'], ascending=False).head(5)

    # Extrahieren der relevanten Entwicklungs-IDs
    top_dev_ids = top_5_devs['development'].unique()

    # Datensatz auf die Top-5-Entwicklungen filtern
    df_top = df[df['development'].isin(top_dev_ids)]

    # Erstellen einer Figur
    plt.figure(figsize=(10, 6))

    # Farbpalette für die Entwicklungen
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Für jede Entwicklung eine kumulative Verteilung plotten
    for i, dev_id in enumerate(top_dev_ids):
        dev_data = df_top[df_top['development'] == dev_id]

        # Sammeln aller Nutzenwerte für diese Entwicklung und Umrechnung in Millionen CHF
        values = dev_data['monetized_savings_total'].dropna().values / 1_000_000

        # Sortieren der Werte für die kumulative Verteilung
        values = np.sort(values)

        # Berechnung der kumulativen Wahrscheinlichkeiten
        y_values = np.arange(1, len(values) + 1) / len(values)

        # Label für die Legende erstellen
        dev_name = f"Entwicklung {dev_id}"
        mean_value = dev_data['monetized_savings_total'].mean() / 1_000_000
        label = f"{dev_name}: {mean_value:.1f} Mio. CHF"

        # Kumulative Verteilung plotten
        plt.plot(values, y_values, '-', color=colors[i], linewidth=2, label=label)

    # Nulllinie hinzufügen
    plt.axvline(x=0, color='lightgray', linestyle='-', linewidth=1)

    # Beschriftungen und Formatierung
    plt.xlabel('Nettonutzen [Mio. CHF]', fontsize=12)
    plt.ylabel('Kumulative Wahrscheinlichkeit', fontsize=12)
    plt.title('Kumulative Wahrscheinlichkeitsverteilung des Nettonutzens\nfür die 5 besten Entwicklungen', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Entwicklungen mit mittlerem Nettonutzen', fontsize=10, title_fontsize=11)

    # Y-Achse von 0 bis 1
    plt.ylim(0, 1.05)

    # X-Achsen-Grenzen definieren
    x_min = df_top['monetized_savings_total'].min() / 1_000_000
    x_max = df_top['monetized_savings_total'].max() / 1_000_000
    plt.xlim(x_min - 5, x_max + 5)

    # Füge Markierungen bei 25%, 50% und 75% hinzu
    plt.axhline(y=0.25, color='darkgray', linestyle=':', alpha=0.7)
    plt.axhline(y=0.5, color='darkgray', linestyle=':', alpha=0.7)
    plt.axhline(y=0.75, color='darkgray', linestyle=':', alpha=0.7)

    # Beschriftung der horizontalen Linien
    plt.text(x_max + 3, 0.25, '25%', va='center', fontsize=9, color='darkgray')
    plt.text(x_max + 3, 0.5, '50%', va='center', fontsize=9, color='darkgray')
    plt.text(x_max + 3, 0.75, '75%', va='center', fontsize=9, color='darkgray')

    plt.tight_layout()

    # Verzeichnis erstellen, falls nicht vorhanden
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Speichern und Anzeigen
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Kumulative Kostenverteilung gespeichert unter: {output_path}")