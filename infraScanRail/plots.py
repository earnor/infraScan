# Core Libraries
import math
import os
import pickle
import re

# Data Libraries
import numpy as np
import pandas as pd

# Geospatial Libraries
import contextily as ctx
import geopandas as gpd
import rasterio
import seaborn as sns
#from geo_northarrow import add_north_arrow
from PIL import Image
from pyrosm import OSM, get_data
from rasterio.mask import mask
from rasterio.plot import show
from shapely import make_valid
from shapely.geometry import LineString, Point, box

# Network Analysis Libraries
import networkx as nx

# Visualization Libraries
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches, pyplot
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Polygon as plotpolygon
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Interpolation
from scipy.interpolate import griddata

# Project-Specific Modules
import paths
import settings
import importlib
import plot_parameter as pp


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
    #add_north_arrow(ax, scale=.75, xlim_pos=.9025, ylim_pos=.835, color='#000', text_scaler=4, text_yT=-1.25)

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

def plot_selected_lines(selected_lines, color_dict = None):
    railway_lines = gpd.read_file(paths.NEW_RAILWAY_LINES_PATH)
    zvv_colors = pp.zvv_colors
    df_network = gpd.read_file(settings.infra_generation_rail_network)
    df_points = gpd.read_file(r'data\Network\processed\points.gpkg')
    generate_infrastructure = importlib.import_module('generate_infrastructure')
    G, pos = generate_infrastructure.prepare_Graph(df_network, df_points)

    # Liniennamen aus selected_lines verwenden und Subset erzeugen
    filtered_lines = railway_lines[railway_lines["name"].isin(selected_lines)]
    filtered_lines['path'] = filtered_lines['path'].str.split(',')
    filtered_lines = filtered_lines.rename(columns={"missing_connection": "original_missing_connection"})



    # Linienfarben in filtered_lines hinzufügen
    filtered_lines_dict = filtered_lines.to_dict(orient='records')

    # Netzgrafik erzeugen
    filename = f"railway_lines_very_special_plot.png"
    output_file_name = os.path.join("plots", filename)

    # Die Funktion plot_railway_lines_only muss angepasst werden, um die Farben zu berücksichtigen
    plot_railway_lines_only(
        G, pos, filtered_lines_dict, output_file_name, color_dict=color_dict, selected_stations=pp.selected_stations
    )

def create_and_save_plots(df, railway_lines, plot_directory="plots", plot_preferences=None):
    """
    Create and save benefit plots for infrastructure developments.
    
    Args:
        df: DataFrame with cost/benefit data
        railway_lines: GeoDataFrame with railway line geometries
        plot_directory: Base directory for plots (default: "plots")
        plot_preferences: Dict with keys:
            - 'small_developments': bool
            - 'grouped_by_connection': bool
            - 'ranked_groups': bool
            - 'combined_with_maps': bool
    """
    # Default preferences: generate all plots if not specified
    if plot_preferences is None:
        plot_preferences = {
            'small_developments': True,
            'grouped_by_connection': True,
            'ranked_groups': True,
            'combined_with_maps': True
        }
    
    # Create subdirectories for new structure
    benefits_dir = os.path.join(plot_directory, "Benefits")
    benefits_combined_dir = os.path.join(plot_directory, "Benefits_Combined")
    benefits_ranked_dir = os.path.join(plot_directory, "Benefits_Ranked")
    benefits_ranked_combined_dir = os.path.join(benefits_ranked_dir, "combined")
    
    os.makedirs(benefits_dir, exist_ok=True)
    if plot_preferences['combined_with_maps']:
        os.makedirs(benefits_combined_dir, exist_ok=True)
        os.makedirs(benefits_ranked_combined_dir, exist_ok=True)

    # ZVV-Farbpalette definieren
    zvv_colors = pp.zvv_colors

    df.rename(columns={'ID_new': 'Scenario'}, inplace=True)
    df['monetized_savings_total'] = df['monetized_savings_total'].abs()
    df['total_costs'] = df['TotalConstructionCost'] + df['TotalMaintenanceCost'] + df['TotalUncoveredOperatingCost']
    df['total_net_benefit'] = df['monetized_savings_total'] - df['total_costs']
    df['cba_ratio'] = df['monetized_savings_total'] / df['total_costs']
    df['Color'] = 'gray'

    # Mapping: df['development'] (float) → railway_lines['name'] (string)
    dev_to_conn = railway_lines.set_index('name')['missing_connection'].to_dict()
    df['missing_connection'] = df['development'].map(lambda x: dev_to_conn.get(f"X{int(x) - 101000}", "unknown"))

    # Load Sline data if not already present in df
    if 'Sline' not in df.columns:
        try:
            # Load Sline from updated_new_links.gpkg
            sline_data = gpd.read_file("data/Network/processed/updated_new_links.gpkg")[['dev_id', 'Sline']]
            # Get unique Sline per dev_id (they should all be the same for each dev_id)
            sline_data = sline_data.groupby('dev_id')['Sline'].first().reset_index()
            # Merge into df
            df = df.merge(sline_data, left_on='development', right_on='dev_id', how='left')
            # Drop the dev_id column if it was created
            if 'dev_id' in df.columns:
                df = df.drop(columns=['dev_id'])
            print("  → Sline data loaded from updated_new_links.gpkg")
        except Exception as e:
            print(f"  ⚠ Warning: Could not load Sline data: {e}")

    df['line_name'] = None  # Initialisierung

    # Für IDs, die mit 100... beginnen → ohne X, aber mit Sline suffix (e.g., "2_G")
    # Check if Sline column exists for small developments
    if 'Sline' in df.columns:
        df.loc[df['development'] < 101000, 'line_name'] = \
            df.loc[df['development'] < 101000].apply(
                lambda row: f"{int(row['development'] - 100000)}_{row['Sline']}", axis=1)
    else:
        # Fallback if Sline is not available
        df.loc[df['development'] < 101000, 'line_name'] = \
            df['development'].loc[df['development'] < 101000].map(
                lambda x: str(int(x - 100000)))

    # Für IDs, die mit 101... beginnen → mit X
    df.loc[df['development'] >= 101000, 'line_name'] = \
        df['development'].loc[df['development'] >= 101000].map(
            lambda x: f"X{int(x - 101000)}")

    # Kleine Entwicklungen
    small_dev_data = df[df['development'] < settings.dev_id_start_new_direct_connections]

    def plot_basic_charts(data, filename_prefix, plot_directory, line_colors=None):
        order = data.groupby('line_name')['total_net_benefit'].mean().sort_values(ascending=False).index.tolist()
        n_lines = len(order)

        # Farbzuordnung für Linien erstellen
        if line_colors is None:
            # Erstelle Farbpalette aus ZVV-Farben (zyklisch wiederholen falls nötig)
            line_colors = {line_name: zvv_colors[i % len(zvv_colors)] for i, line_name in enumerate(order)}

        # Farben für Kosten/Nutzen
        kosten_farben = {
            'TotalConstructionCost': '#a6bddb',
            'TotalMaintenanceCost': '#3690c0',
            'TotalUncoveredOperatingCost': '#034e7b',
            'monetized_savings_total': '#31a354'
        }

        # --- Boxplot Einsparungen ---
        plt.figure(figsize=(7, 5), dpi=300)
        colors = [line_colors[line] for line in order]
        ax = sns.boxplot(
            data=data,
            x='line_name',
            y=data['monetized_savings_total'] / 1e6,
            order=order,
            palette=colors,
            width=0.4,
            linewidth=0.8,
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5},
            fliersize=3,
            showfliers=True
        )
        ax.set_xlim(-0.5, n_lines - 0.5)
        plt.xlabel('Line', fontsize=12)
        plt.ylabel('Monetised travel time savings in million CHF', fontsize=12)
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        legend_handles = [
            mlines.Line2D([0], [0], marker='o', color='black', label='Mean', markersize=5),
            mpatches.Patch(color=colors[0], label='Line Colour')
        ]
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)

        plt.tight_layout(rect=[0, 0, 0.95, 1])
        plt.savefig(os.path.join(plot_directory, f"{filename_prefix}_boxplot_savings.png"), dpi=600)
        plt.close()

        # --- Violinplot Einsparungen mit Datenpunkten ---
        plt.figure(figsize=(7, 5), dpi=300)
        ax = sns.violinplot(
            data=data,
            x='line_name',
            y=data['monetized_savings_total'] / 1e6,
            order=order,
            palette=colors,
            width=0.7,
            inner=None,
            linewidth=0.8,
            cut=0,
            scale='width'
        )

        unique_data = data.drop_duplicates(subset=['line_name', 'scenario', 'monetized_savings_total'])

        sns.stripplot(
            data=unique_data,
            x='line_name',
            y=unique_data['monetized_savings_total'] / 1e6,
            order=order,
            color='black',
            alpha=0.4,
            jitter=True,
            size=2,
            dodge=False
        )
        ax.set_xlim(-0.5, n_lines - 0.5)
        plt.xlabel('Line', fontsize=12)
        plt.ylabel('Monetised travel time savings in million CHF', fontsize=12)
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        legend_handles = [
            mlines.Line2D([], [], marker='o', color='black', alpha=0.4, linestyle='None', markersize=3,
                          label='Individual Values'),
            mpatches.Patch(color=colors[0], label='Line Colour')
        ]
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)

        plt.tight_layout(rect=[0, 0, 0.95, 1])
        plt.savefig(os.path.join(plot_directory, f"{filename_prefix}_violinplot_savings.png"), dpi=600)
        plt.close()

        # --- Boxplot Nettonutzen ---
        plt.figure(figsize=(7, 5), dpi=300)
        ax = sns.boxplot(
            data=data,
            x='line_name',
            y=data['total_net_benefit'] / 1e6,
            order=order,
            palette=colors,
            width=0.4,
            linewidth=0.8,
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5},
            fliersize=3,
            showfliers=True
        )
        ax.set_xlim(-0.5, n_lines - 0.5)
        plt.xlabel('Line', fontsize=12)
        plt.ylabel('Net benefit in CHF million', fontsize=12)
        plt.axhline(y=0, color='red', linestyle='-', alpha=0.5)
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        legend_handles = [
            mlines.Line2D([0], [0], marker='o', color='black', label='Mean', markersize=5),
            mpatches.Patch(color=colors[0], label='Line Colour')
        ]
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)

        plt.tight_layout(rect=[0, 0, 0.95, 1])
        plt.savefig(os.path.join(plot_directory, f"{filename_prefix}_boxplot_net_benefit.png"), dpi=600)
        plt.close()

        # --- Boxplot CBA ---
        plt.figure(figsize=(7, 5), dpi=300)
        ax = sns.boxplot(
            data=data,
            x='line_name',
            y='cba_ratio',
            order=order,
            palette=colors,
            width=0.4,
            linewidth=0.8,
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5},
            fliersize=3,
            showfliers=True
        )
        ax.set_xlim(-0.5, n_lines - 0.5)
        plt.xlabel('Line', fontsize=12)
        plt.ylabel('Cost-benefit ratio', fontsize=12)
        plt.axhline(y=1, color='red', linestyle='-', alpha=0.5)
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        legend_handles = [
            mlines.Line2D([0], [0], marker='o', color='black', label='Mean', markersize=5),
            mpatches.Patch(color=colors[0], label='Line Colour')
        ]
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)

        plt.tight_layout(rect=[0, 0, 0.95, 1])
        plt.savefig(os.path.join(plot_directory, f"{filename_prefix}_boxplot_cba.png"), dpi=600)
        plt.close()

        # --- Gestapeltes Balkendiagramm ---
        summary = data.groupby(['development', 'line_name']).agg({
            'TotalConstructionCost': 'mean',
            'TotalMaintenanceCost': 'mean',
            'TotalUncoveredOperatingCost': 'mean',
            'monetized_savings_total': 'mean'
        }).loc[(slice(None), order), :].reset_index().set_index('line_name').loc[order].reset_index()

        x_pos = np.arange(n_lines)
        bar_width = 0.6
        plt.figure(figsize=(7, 5), dpi=300)

        plt.bar(x_pos, -summary['TotalConstructionCost'] / 1e6, width=bar_width,
                color=kosten_farben['TotalConstructionCost'],
                label='Construction costs')
        plt.bar(x_pos, -summary['TotalMaintenanceCost'] / 1e6, width=bar_width,
                bottom=-summary['TotalConstructionCost'] / 1e6, color=kosten_farben['TotalMaintenanceCost'],
                label='Uncovered maintenance costs')
        plt.bar(x_pos, -summary['TotalUncoveredOperatingCost'] / 1e6, width=bar_width,
                bottom=-(summary['TotalConstructionCost'] + summary['TotalMaintenanceCost']) / 1e6,
                color=kosten_farben['TotalUncoveredOperatingCost'], label='Uncovered operating costs')
        
        for i, line_name in enumerate(order):
            plt.bar(x_pos[i], summary[summary['line_name'] == line_name]['monetized_savings_total'].values[0] / 1e6,
                    width=bar_width, color=line_colors[line_name], hatch='////', edgecolor='black')

        plt.axhline(y=0, color='black', linestyle='-')
        plt.xticks(x_pos, summary['line_name'], rotation=90)
        plt.xlabel('Line', fontsize=12)
        plt.ylabel('Value in CHF million', fontsize=12)
        plt.title('Costs and benefits per modification', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        kosten_handles = [
            mpatches.Patch(color=kosten_farben['TotalConstructionCost'], label='Construction costs'),
            mpatches.Patch(color=kosten_farben['TotalMaintenanceCost'], label='Uncovered maintenance costs'),
            mpatches.Patch(color=kosten_farben['TotalUncoveredOperatingCost'], label='Uncovered operating costs'),
            mpatches.Patch(facecolor="none", hatch='////', edgecolor='black', label='Travel time savings'),
        ]

        plt.legend(handles=kosten_handles, bbox_to_anchor=(1.01, 1))

        plt.tight_layout(rect=[0, 0, 0.95, 1])
        plt.savefig(os.path.join(plot_directory, f"{filename_prefix}_cost_savings.png"), dpi=600)
        plt.close()

        # NEU: Kumulatives Verteilungsdiagramm für diese Datengruppe
        # Use line_name directly as the grouping column
        cumulative_output_path = os.path.join(plot_directory, f"{filename_prefix}_cumulative_cost_distribution.png")
        plot_cumulative_cost_distribution(data, cumulative_output_path,
                                          color_dict=line_colors,
                                          group_by='line_name')

        return line_colors

    # ============================================================================
    # SMALL DEVELOPMENTS (Expand 1 Stop)
    # ============================================================================
    line_colors_small = None
    if plot_preferences['small_developments'] and not small_dev_data.empty:
        print("  → Generating plots for small developments (Expand 1 Stop)...")
        order = small_dev_data.groupby('line_name')['total_net_benefit'].mean().sort_values(ascending=False).index.tolist()
        # Apply ZVV color palette to small developments
        line_colors_small = {line_name: zvv_colors[i % len(zvv_colors)] for i, line_name in enumerate(order)}
        line_colors_small = plot_basic_charts(small_dev_data, "Expand_1_Stop", plot_directory=benefits_dir, line_colors=line_colors_small)

    # ============================================================================
    # RANKED GROUPS - EXPAND LINES ONLY
    # ============================================================================
    if plot_preferences['ranked_groups'] and not small_dev_data.empty:
        print("  → Generating ranked plot for expand lines...")

        # Use existing Benefits_Ranked directory
        os.makedirs(benefits_ranked_dir, exist_ok=True)
        os.makedirs(benefits_ranked_combined_dir, exist_ok=True)

        # Rank all small developments by mean net benefit (all in one plot named "ranked_group_expand")
        ranked_small_dev = small_dev_data.groupby('line_name')['total_net_benefit'].mean().sort_values(ascending=False)
        ranked_line_names = ranked_small_dev.index.tolist()

        # Use all expand lines in a single plot
        ranked_filename_prefix = "ranked_group_expand"

        # Generate chart plots with consistent colors (preserve ranking order)
        ranked_line_colors = {line_name: zvv_colors[i % len(zvv_colors)]
                             for i, line_name in enumerate(ranked_line_names)}

        plot_basic_charts(small_dev_data, ranked_filename_prefix,
                         plot_directory=benefits_ranked_dir,
                         line_colors=ranked_line_colors)

        # Generate network maps and combined plots if requested
        if plot_preferences['combined_with_maps']:

            # Load network data if not already loaded
            if 'G' not in locals() or 'pos' not in locals():
                df_network = gpd.read_file(settings.infra_generation_rail_network)
                df_points = gpd.read_file(r'data\Network\processed\points.gpkg')
                generate_infrastructure = importlib.import_module('generate_infrastructure')
                G, pos = generate_infrastructure.prepare_Graph(df_network, df_points)

            # Get all line names for expand developments
            expand_line_names = small_dev_data['line_name'].unique()

            # Filter railway_lines to get geometries for expand developments
            # For expand lines, railway_lines["name"] should match the base development number
            # e.g., development 100002 → railway_lines name "2"
            # Extract just the base numbers from line_name (e.g., "2_G" → "2")
            expand_base_numbers = list(set([line_name.split('_')[0] for line_name in expand_line_names]))

            # Filter railway_lines by these base numbers
            filtered_lines = railway_lines[railway_lines["name"].isin(expand_base_numbers)]

            print(f"    → Expand base numbers: {expand_base_numbers}")
            print(f"    → Found {len(filtered_lines)} railway lines matching expand developments")

            if not filtered_lines.empty:
                filtered_lines = filtered_lines.copy()
                filtered_lines['path'] = filtered_lines['path'].str.split(',')

                # Create records dict with colors
                filtered_lines_dict = filtered_lines.to_dict(orient='records')

                # Map colors: railway_lines uses "name" (e.g., "2"), but we need to map to "line_name" (e.g., "2_G")
                # Create a mapping from base line number to line_name with Sline
                line_num_to_full_name = {}
                for line_name in expand_line_names:
                    # Extract base number from line_name (e.g., "2_G" → "2")
                    base_num = line_name.split('_')[0]
                    if base_num not in line_num_to_full_name:
                        line_num_to_full_name[base_num] = []
                    line_num_to_full_name[base_num].append(line_name)

                # Assign colors to records based on the first matching line_name
                for record in filtered_lines_dict:
                    base_line_name = record["name"]
                    if base_line_name in line_num_to_full_name:
                        # Use the first full line_name for this base number
                        full_line_name = line_num_to_full_name[base_line_name][0]
                        if full_line_name in ranked_line_colors:
                            record["color"] = ranked_line_colors[full_line_name]

                # Generate network map
                map_filename = f"railway_lines_{ranked_filename_prefix}.png"
                map_output_path = os.path.join("plots", map_filename)

                plot_railway_lines_only(
                    G, pos, filtered_lines_dict, map_output_path,
                    color_dict=ranked_line_colors, selected_stations=pp.selected_stations
                )

                # Combine images for each chart type
                for suffix in [
                    "boxplot_savings",
                    "violinplot_savings",
                    "boxplot_net_benefit",
                    "boxplot_cba",
                    "cost_savings",
                    "cumulative_cost_distribution"
                ]:
                    chart_path = os.path.join(benefits_ranked_dir, f"{ranked_filename_prefix}_{suffix}.png")
                    map_path = os.path.join("plots", f"railway_lines_{ranked_filename_prefix}.png")
                    combined_path = os.path.join(benefits_ranked_combined_dir, f"{ranked_filename_prefix}_{suffix}_combined.png")

                    try:
                        if not os.path.exists(chart_path):
                            print(f"    ⚠ Chart not found: {chart_path}")
                            continue
                        if not os.path.exists(map_path):
                            print(f"    ⚠ Map not found: {map_path}")
                            continue

                        map_image = Image.open(map_path)
                        chart_image = Image.open(chart_path)

                        target_height = max(map_image.height, chart_image.height)

                        def resize_to_height(img, target_h):
                            w, h = img.size
                            new_w = int(w * (target_h / h))
                            return img.resize((new_w, target_h), Image.LANCZOS)

                        map_image_resized = resize_to_height(map_image, target_height)
                        chart_image_resized = resize_to_height(chart_image, target_height)

                        total_width = map_image_resized.width + chart_image_resized.width
                        combined = Image.new("RGB", (total_width, target_height), (255, 255, 255))
                        combined.paste(map_image_resized, (0, 0))
                        combined.paste(chart_image_resized, (map_image_resized.width, 0))
                        combined.save(combined_path)

                    except Exception as e:
                        print(f"    ⚠ Error combining images for {ranked_filename_prefix}_{suffix}: {e}")
            else:
                print(f"    ⚠ No railway lines found for expand developments")

    # ============================================================================
    # GROUPED BY MISSING CONNECTION
    # ============================================================================
    df['plot_nr'] = None
    large_dev_data = df[df['development'] >= settings.dev_id_start_new_direct_connections]

    if plot_preferences['grouped_by_connection'] and not large_dev_data.empty:
        print("  → Generating plots grouped by missing connection...")
        
        df_network = gpd.read_file(settings.infra_generation_rail_network)
        df_points = gpd.read_file(r'data\Network\processed\points.gpkg')
        generate_infrastructure = importlib.import_module('generate_infrastructure')
        G, pos = generate_infrastructure.prepare_Graph(df_network, df_points)

        for conn in large_dev_data['missing_connection'].dropna().unique():
            sub_df = large_dev_data[large_dev_data['missing_connection'] == conn].copy()

            mean_benefit = sub_df.groupby('development')['total_net_benefit'].mean().sort_values(ascending=False)
            unique_devs = mean_benefit.index.tolist()

            num_plots = len(unique_devs) // 6 + int(len(unique_devs) % 6 > 0)

            for i in range(num_plots):
                devs_in_plot = unique_devs[i * 6: (i + 1) * 6]
                selected = sub_df[sub_df['development'].isin(devs_in_plot)]
                df.loc[selected.index, 'plot_nr'] = f"{conn}_{i + 1}"
                filename_prefix = f"{conn}_group_{i + 1}"
                filename_prefix = filename_prefix.replace(" - ", "_").replace(" ", "_")

                # Generate chart plots
                line_colors = plot_basic_charts(selected, filename_prefix, plot_directory=benefits_dir)

                # Generate network map (if combined plots requested)
                if plot_preferences['combined_with_maps']:
                    line_names = selected['line_name'].unique()
                    filtered_lines = railway_lines[railway_lines["name"].isin(line_names)]
                    filtered_lines['path'] = filtered_lines['path'].str.split(',')
                    filtered_lines = filtered_lines.rename(columns={"missing_connection": "original_missing_connection"})

                    filtered_lines_dict = filtered_lines.to_dict(orient='records')
                    for record in filtered_lines_dict:
                        line_name = record["name"]
                        if line_name in line_colors:
                            record["color"] = line_colors[line_name]

                    # Network map saved to plots/ (root level, as per Q4)
                    filename = f"railway_lines_{filename_prefix}.png"
                    output_file_name = os.path.join("plots", filename)

                    plot_railway_lines_only(
                        G, pos, filtered_lines_dict, output_file_name, 
                        color_dict=line_colors, selected_stations=pp.selected_stations
                    )

                    # Combine images
                    for suffix in [
                        "boxplot_savings",
                        "violinplot_savings",
                        "boxplot_net_benefit",
                        "boxplot_cba",
                        "cost_savings",
                        "cumulative_cost_distribution"
                    ]:
                        chart_path = os.path.join(benefits_dir, f"{filename_prefix}_{suffix}.png")
                        map_path = os.path.join("plots", f"railway_lines_{filename_prefix}.png")
                        combined_path = os.path.join(benefits_combined_dir, f"{filename_prefix}_{suffix}_combined.png")

                        try:
                            if not os.path.exists(chart_path):
                                raise FileNotFoundError(f"Chart not found: {chart_path}")
                            if not os.path.exists(map_path):
                                raise FileNotFoundError(f"Map not found: {map_path}")

                            map_image = Image.open(map_path)
                            chart_image = Image.open(chart_path)

                            target_height = max(map_image.height, chart_image.height)

                            def resize_to_height(img, target_h):
                                w, h = img.size
                                new_w = int(w * (target_h / h))
                                return img.resize((new_w, target_h), Image.LANCZOS)

                            map_image_resized = resize_to_height(map_image, target_height)
                            chart_image_resized = resize_to_height(chart_image, target_height)

                            total_width = map_image_resized.width + chart_image_resized.width
                            combined = Image.new("RGB", (total_width, target_height), (255, 255, 255))
                            combined.paste(map_image_resized, (0, 0))
                            combined.paste(chart_image_resized, (map_image_resized.width, 0))
                            combined.save(combined_path)

                        except Exception as e:
                            print(f"Error combining images for {filename_prefix}_{suffix}: {e}")

    # ============================================================================
    # RANKED GROUPS (Global ranking)
    # ============================================================================
    if plot_preferences['ranked_groups'] and not large_dev_data.empty:
        print("  → Generating ranked group plots...")
        
        os.makedirs(benefits_ranked_dir, exist_ok=True)
        
        # Load network data if not already loaded (might be loaded from grouped section)
        if not plot_preferences['grouped_by_connection']:
            df_network = gpd.read_file(settings.infra_generation_rail_network)
            df_points = gpd.read_file(r'data\Network\processed\points.gpkg')
            generate_infrastructure = importlib.import_module('generate_infrastructure')
            G, pos = generate_infrastructure.prepare_Graph(df_network, df_points)

        global_mean_benefit = large_dev_data.groupby('development')['total_net_benefit'].mean().sort_values(ascending=False)
        global_unique_devs = global_mean_benefit.index.tolist()

        num_global_plots = len(global_unique_devs) // 6 + int(len(global_unique_devs) % 6 > 0)

        for i in range(num_global_plots):
            devs_in_plot = global_unique_devs[i * 6: (i + 1) * 6]
            global_selected = large_dev_data[large_dev_data['development'].isin(devs_in_plot)]

            global_filename_prefix = f"ranked_group_{i + 1}"

            # Generate chart plots
            global_line_colors = plot_basic_charts(global_selected, global_filename_prefix,
                                                   plot_directory=benefits_ranked_dir)

            # Generate network map (if combined plots requested)
            if plot_preferences['combined_with_maps']:
                global_line_names = global_selected['line_name'].unique()
                global_filtered_lines = railway_lines[railway_lines["name"].isin(global_line_names)]
                global_filtered_lines['path'] = global_filtered_lines['path'].str.split(',')
                global_filtered_lines = global_filtered_lines.rename(
                    columns={"missing_connection": "original_missing_connection"})

                global_filtered_lines_dict = global_filtered_lines.to_dict(orient='records')
                for record in global_filtered_lines_dict:
                    line_name = record["name"]
                    if line_name in global_line_colors:
                        record["color"] = global_line_colors[line_name]

                # Network map saved to plots/ (root level)
                global_filename = f"railway_lines_{global_filename_prefix}.png"
                global_output_file_name = os.path.join("plots", global_filename)

                plot_railway_lines_only(
                    G, pos, global_filtered_lines_dict, global_output_file_name,
                    color_dict=global_line_colors, selected_stations=pp.selected_stations
                )

                # Combine images
                for suffix in [
                    "boxplot_savings",
                    "violinplot_savings",
                    "boxplot_net_benefit",
                    "boxplot_cba",
                    "cost_savings",
                    "cumulative_cost_distribution"
                ]:
                    chart_path = os.path.join(benefits_ranked_dir, f"{global_filename_prefix}_{suffix}.png")
                    map_path = os.path.join("plots", f"railway_lines_{global_filename_prefix}.png")
                    combined_path = os.path.join(benefits_ranked_combined_dir, f"{global_filename_prefix}_{suffix}_combined.png")

                    try:
                        if not os.path.exists(chart_path):
                            raise FileNotFoundError(f"Chart not found: {chart_path}")
                        if not os.path.exists(map_path):
                            raise FileNotFoundError(f"Map not found: {map_path}")

                        map_image = Image.open(map_path)
                        chart_image = Image.open(chart_path)

                        target_height = max(map_image.height, chart_image.height)

                        def resize_to_height(img, target_h):
                            w, h = img.size
                            new_w = int(w * (target_h / h))
                            return img.resize((new_w, target_h), Image.LANCZOS)

                        map_image_resized = resize_to_height(map_image, target_height)
                        chart_image_resized = resize_to_height(chart_image, target_height)

                        total_width = map_image_resized.width + chart_image_resized.width
                        combined = Image.new("RGB", (total_width, target_height), (255, 255, 255))
                        combined.paste(map_image_resized, (0, 0))
                        combined.paste(chart_image_resized, (map_image_resized.width, 0))
                        combined.save(combined_path)

                    except Exception as e:
                        print(f"Error combining images for {global_filename_prefix}_{suffix}: {e}")

    return df, railway_lines



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


def plot_costs_benefits(cost_benefit_df, line=None, output_dir="plots"):
    """
    Erstellt ein wissenschaftlich aussehendes Balkendiagramm von Kosten und Nutzen über die Jahre
    für die Entwicklung mit dem höchsten Nutzen im Jahr 1.

    Args:
        cost_benefit_df: DataFrame mit diskontierten Kosten und Nutzen
    """
    # Finde das niedrigste Jahr im DataFrame
    min_year = cost_benefit_df.index.get_level_values('year').min()
    if line:
        try:
            dev_data = cost_benefit_df.xs(line, level='development')
        except KeyError:
            # Falls der angegebene Wert nicht existiert, verwende die Entwicklung mit dem höchsten Nutzen
            print(f"Development {line} not found. Use development with the highest benefit.")
            year1_benefits = cost_benefit_df.xs(min_year, level='year')
            max_benefit_dev = year1_benefits.groupby('development')['benefit'].mean().idxmax()
            dev_data = cost_benefit_df.xs(max_benefit_dev, level='development')
            line = max_benefit_dev
    else:
        # Extrahiere Daten für das niedrigste Jahr
        year1_benefits = cost_benefit_df.xs(min_year, level='year')
        max_benefit_dev = year1_benefits.groupby('development')['benefit'].mean().idxmax()
        # Daten für die ausgewählte Entwicklung abrufen
        dev_data = cost_benefit_df.xs(max_benefit_dev, level='development')
        line = max_benefit_dev

    # Berechne Mittelwerte über alle Szenarien für jedes Jahr
    plot_data = dev_data.groupby('year').mean()

    # Farben für die Kosten definieren
    kosten_farben = {
        'const_cost': '#a6bddb',  # TotalConstructionCost
        'maint_cost': '#3690c0',  # TotalMaintenanceCost
        'uncovered_op_cost': '#034e7b'  # TotalUncoveredOperatingCost
    }

    # Negation der Kosten für das gemeinsame Diagramm
    plot_data['const_cost_neg'] = -plot_data['const_cost']
    plot_data['maint_cost_neg'] = -plot_data['maint_cost']
    plot_data['uncovered_op_neg'] = -plot_data['uncovered_op_cost']

    # Abbildung erstellen
    fig, ax = plt.subplots(figsize=(7, 5))

    # Wissenschaftlicher Stil
    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'serif'

    width = 0.8

    # Gestapelte Balken für negative Kosten (nach unten)
    costs_bars1 = ax.bar(plot_data.index, plot_data['const_cost_neg'], width,
                         label='Construction costs', color=kosten_farben['const_cost'], alpha=0.8)
    costs_bars2 = ax.bar(plot_data.index, plot_data['maint_cost_neg'], width,
                         bottom=plot_data['const_cost_neg'], label='Maintenance costs',
                         color=kosten_farben['maint_cost'], alpha=0.8)
    sum_prev_costs = plot_data['const_cost_neg'] + plot_data['maint_cost_neg']
    costs_bars3 = ax.bar(plot_data.index, plot_data['uncovered_op_neg'], width,
                         bottom=sum_prev_costs, label='Operating costs',
                         color=kosten_farben['uncovered_op_cost'], alpha=0.8)

    # Balken für Reisezeitersparnisse (nach oben)
    benefit_bars = ax.bar(plot_data.index, plot_data['benefit'], width,
                          label='Travel time savings', color='#2ca02c', alpha=0.8)

    # Diagramm anpassen
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Values in CHF million', fontsize=12)
    ax.set_title(f'Discounted costs and benefits over time\nDevelopment: {line}',
                 fontsize=14, pad=20)
    ax.legend(loc='lower right', frameon=True, edgecolor='black')
    ax.grid(True, which="both", ls="-", alpha=0.2)

    # Y-Achsengrenzen anhand der Kosten im Jahr 2 festlegen
    year_2_costs = (plot_data['const_cost_neg'] + plot_data['maint_cost_neg'] +
                    plot_data['uncovered_op_neg']).iloc[1] if len(plot_data) > 1 else 0
    y_limit_bottom = year_2_costs * 1.5  # 150% der Kosten im Jahr 2

    max_value = max(plot_data['benefit'])
    y_limit_top = max_value * 1.2  # Obere Grenze bleibt wie zuvor

    ax.set_ylim(bottom=y_limit_bottom, top=y_limit_top)

    # Formatter für Millionenwerte ohne wissenschaftliche Notation
    def millions_formatter(x, pos):
        return f'{x / 1e6:.1f}'

    ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))

    # Beschriftung für abgeschnittene Construction Costs hinzufügen
    # Finde die höchsten Construction Costs
    total_costs_by_year = (plot_data['const_cost_neg'] + plot_data['maint_cost_neg'] +
                           plot_data['uncovered_op_neg'])
    max_cost_year = total_costs_by_year.idxmin()
    max_cost_value = total_costs_by_year.min()

    # Annotiere den gesamten abgeschnittenen Stapel
    ax.annotate(f'{-max_cost_value / 1e6:.1f} Mio. CHF',
                xy=(max_cost_year, y_limit_bottom * 0.95),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=10)

    # Layout anpassen
    plt.tight_layout()

    # Stelle sicher, dass das Verzeichnis existiert
    os.makedirs(output_dir, exist_ok=True)

    # Dateiname mit Entwicklungsname erstellen
    filename = os.path.join(output_dir, f'cost_benefit_{line}.png')

    # Lösche die Datei, wenn sie bereits existiert
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Bestehende Datei gelöscht: {filename}")

    # Plot speichern
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot for development {line} saved as {filename}")


def plot_lines_to_network(points_gdf,lines_gdf):
    points_gdf.plot(marker='*', color='green', markersize=5)
    base = lines_gdf.plot(edgecolor='black')
    points_gdf.plot(ax=base, marker='o', color='red', markersize=5)
    plt.savefig(r"plot\predict\230822_network-generation.png", dpi=300)
    return None


def plot_graph(graph, positions, highlight_centers=True, missing_links=None, directory='data/plots', polygon=None):
    """
    Plot the railway network graph with optional highlighting of center nodes and missing connections.

    Args:
        graph (networkx.Graph): The railway network graph
        positions (dict): Dictionary mapping node IDs to (x,y) coordinates
        highlight_centers (bool): Whether to highlight center nodes
        missing_links (list): List of missing connections to highlight
        directory (str): Directory to save the plot
        polygon (shapely.geometry.Polygon): Optional polygon to draw around the network
    """
    # Create the plot
    plt.figure(figsize=(10, 8), dpi=600)
    ax = plt.gca()

    # Setze das Achsenverhältnis auf "gleich", damit X und Y die gleiche Skala haben
    ax.set_aspect('equal')

    # Zeichne das Polygon zuerst im Hintergrund
    if polygon:
        x, y = polygon.exterior.xy
        ax.plot(x, y, 'b-', linewidth=5, alpha=0.4, zorder=0)

    # Draw edges
    nx.draw_networkx_edges(graph, positions, edge_color='gray', width=0.5, alpha=0.6, ax=ax)

    # Wenn missing_links vorhanden sind
    nodes_in_missing_connections = set()
    missing_connections_coords = []  # Für den Zoom

    if missing_links:
        missing_connections = []

        for center in missing_links:
            missing_connections.extend(center['missing_connections'])

        for conn in missing_connections:
            node1, node2 = conn['nodes']
            nodes_in_missing_connections.add(node1)
            nodes_in_missing_connections.add(node2)

            if node1 in positions and node2 in positions:
                x1, y1 = positions[node1]
                x2, y2 = positions[node2]
                # Hier gepunktete Linie (:) statt gestrichelte Linie (--)
                # Mit feinerer Kontrolle über die Punktdichte
                ax.plot([x1, x2], [y1, y2], 'r', linestyle=':', linewidth=3.0, alpha=0.9, zorder=2,
                        dashes=[2, 1])  # [Punktlänge, Zwischenraum] - kleinere Werte = engere Punkte
                missing_connections_coords.append((x1, y1))
                missing_connections_coords.append((x2, y2))

    # Draw nodes with different colors based on type
    node_colors = []
    node_sizes = []
    important_nodes = set()  # Für Knoten mit Labels

    # Liste für Stationen mit Label unter dem Knoten
    below_label_stations = ["Zürich Oerlikon", "Stettbach", "Zürich HB", "Zürich Stadelhofen", "Zürich Altstetten"]

    for node in graph.nodes():
        station_name = graph.nodes[node].get('station_name', '')

        # Bestimme, welche Knoten als wichtig gelten und beschriftet werden
        if graph.nodes[node].get('type') == 'center' and highlight_centers:
            node_colors.append('red')
            node_sizes.append(150)
            important_nodes.add(node)
        # Prüfe zuerst auf fehlende Verbindungen vor der Prüfung auf Grenzknoten
        elif node in nodes_in_missing_connections:
            node_colors.append('orange')  # Knoten einer missing connection in Orange
            node_sizes.append(100)        # Größe 100
            important_nodes.add(node)
        elif graph.nodes[node].get('type') == 'border' and highlight_centers:
            node_colors.append('orange')
            node_sizes.append(100)
        elif graph.nodes[node].get('end_station', False):
            node_colors.append('green')
            node_sizes.append(100)
            important_nodes.add(node)
        else:
            node_colors.append('lightblue')
            node_sizes.append(50)

    # Füge alle Knoten mit Stationsnamen in below_label_stations zu important_nodes hinzu
    important_nodes.update(
        [node for node in graph.nodes() if graph.nodes[node].get('station_name') in below_label_stations])

    # Draw nodes
    nx.draw_networkx_nodes(graph, positions, node_color=node_colors, node_size=node_sizes, ax=ax)

    # Add labels only for important nodes
    labels = {node: graph.nodes[node].get('station_name', '') for node in important_nodes}

    # Text-Positionen für Labels berechnen (doppelter Abstand vom Knoten)
    # Text-Positionen für Labels berechnen (mit vergrößertem Abstand)
    label_pos = {}
    for node in labels:
        if node in positions:
            station_name = graph.nodes[node].get('station_name', '')
            if station_name in below_label_stations:
                # Für Stationen in below_label_stations UNTER dem Knoten mit doppeltem Abstand
                label_pos[node] = (positions[node][0], positions[node][1] - 400)  # Doppelter Abstand nach unten (-2)
            else:
                # Für alle anderen ÜBER dem Knoten mit doppeltem Abstand
                label_pos[node] = (positions[node][0], positions[node][1] + 400)  # Doppelter Abstand nach oben (+2)

    # Erstelle zwei separate Gruppen für Knoten, die oben bzw. unten beschriftet werden sollen
    top_labels = {node: labels[node] for node in labels if
                  graph.nodes[node].get('station_name', '') not in below_label_stations}
    bottom_labels = {node: labels[node] for node in labels if
                     graph.nodes[node].get('station_name', '') in below_label_stations}

    # Erstelle die entsprechenden Positionsdictionaries
    top_label_pos = {node: label_pos[node] for node in top_labels}
    bottom_label_pos = {node: label_pos[node] for node in bottom_labels}

    # Zeichne die oberen Labels (mit korrekter verticalalignment)
    nx.draw_networkx_labels(graph, top_label_pos, top_labels, font_size=10, font_weight='normal',
                            verticalalignment='bottom', horizontalalignment='center', ax=ax)

    # Zeichne die unteren Labels (mit korrekter verticalalignment)
    nx.draw_networkx_labels(graph, bottom_label_pos, bottom_labels, font_size=10, font_weight='normal',
                            verticalalignment='top', horizontalalignment='center', ax=ax)

    # Zoom auf den Bereich mit fehlenden Verbindungen, falls vorhanden
    if missing_connections_coords:
        x_coords = [coord[0] for coord in missing_connections_coords]
        y_coords = [coord[1] for coord in missing_connections_coords]

        # Berechne den Bereich der fehlenden Verbindungen mit 7km Puffer (7000 Meter)
        # Annahme: 1 Einheit in Koordinaten = 1 Meter
        buffer_size = 7000
        min_x, max_x = min(x_coords) - buffer_size, max(x_coords) + buffer_size
        min_y, max_y = min(y_coords) - buffer_size, max(y_coords) + buffer_size

        # Setze die Achsengrenzen auf den Bereich der fehlenden Verbindungen
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

    # === Nordpfeil oben rechts positionieren ===
    arrow_pos_x = 0.92  # X-Position im Achsenkoordinatensystem
    arrow_pos_y = 0.92  # Y-Position im Achsenkoordinatensystem

    # Pfeil zeichnen (näher zur Beschriftung)
    arrow = FancyArrowPatch((arrow_pos_x, arrow_pos_y - 0.010),
                            (arrow_pos_x, arrow_pos_y + 0.05),
                            color='black',
                            lw=2.5,  # Etwas dicker
                            arrowstyle='->',
                            mutation_scale=20,  # Größerer Pfeilkopf
                            transform=ax.transAxes,
                            zorder=1000)
    ax.add_patch(arrow)

    # "N" Beschriftung näher zum Pfeil platzieren und größer
    ax.text(arrow_pos_x, arrow_pos_y - 0.035, "N",
            fontsize=18, weight='bold',  # Größere Schrift
            ha='center', va='center',
            transform=ax.transAxes,
            zorder=1000,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # === Maßstab oben rechts neben dem Nordpfeil hinzufügen ===
    scale_pos_x = 0.8  # Weiter links vom Nordpfeil
    scale_pos_y = 0.92  # Gleiche Höhe wie der Nordpfeil
    scale_width = 0.1  # Breite des Maßstabsbalkens in Achsenkoordinaten
    scale_height = 0.01  # Höhe des Maßstabsbalkens

    # Maßstabsbalken als schwarzes Rechteck
    scale_rect = plt.Rectangle((scale_pos_x - scale_width / 2, scale_pos_y - scale_height / 2),
                               scale_width, scale_height,
                               facecolor='black', edgecolor='black',
                               transform=ax.transAxes, zorder=1000)
    ax.add_patch(scale_rect)

    # Maßstabstext (5 km) unter dem Balken, größer
    ax.text(scale_pos_x, scale_pos_y - 0.04, "5 km",
            fontsize=12, ha='center', va='center',  # Größere Schrift
            transform=ax.transAxes, zorder=1000,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # Add legend
    if highlight_centers:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Zentrumsknoten'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Grenzknoten'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Endstation'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Station')
        ]
        # Polygon zur Legende hinzufügen, falls vorhanden
        if polygon:
            legend_elements.append(
                Line2D([0], [0], color='blue', linewidth=5, alpha=0.4, label='Fallstudienkorridor'))
        if missing_links:
            legend_elements.append(
                Line2D([0], [0], color='r', linestyle=':', linewidth=3, label='Fehlende Direktverbindung'))
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.title("Fehlende Direktverbindungen von Korridoren")
    plt.axis('on')
    plt.grid(True)
    plt.tight_layout()

    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)

    plt.savefig(os.path.join(directory, 'network_graph.png'),
                dpi=600,
                bbox_inches='tight',
                pad_inches=0.2,
                format='png')

    plt.close()  # Schließe die Figur, um Speicher freizugeben


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


def plot_railway_lines_only(G, pos, railway_lines, output_file, color_dict=None, selected_stations=None):
            """
            Plot a railway graph with proposed railway lines,
            zoomed to line extent with correct aspect ratio and compact figure size.

            Args:
                G: NetworkX graph containing the railway network
                pos: Dictionary mapping node IDs to coordinates
                railway_lines: List of railway line dictionaries
                output_file: Path where to save the plot
                color_dict: Optional dictionary mapping line names to colors
                selected_stations: Optional list of station names to display on the map
            """
            zvv_colors = pp.zvv_colors

            # Falls keine Liste ausgewählter Stationen übergeben wurde, leere Liste verwenden
            if selected_stations is None:
                selected_stations = []

            # === Validierungen ===
            if not isinstance(G, nx.Graph):
                raise TypeError("'G' muss ein NetworkX-Graph sein.")
            if not isinstance(pos, dict) or not pos:
                raise ValueError("'pos' muss ein nicht-leeres Dictionary mit Koordinaten sein.")
            if not isinstance(railway_lines, list) or len(railway_lines) == 0:
                raise ValueError("'railway_lines' muss eine nicht-leere Liste von Dictionaries sein.")
            if not output_file or not isinstance(output_file, str):
                raise ValueError("'output_file' muss ein gültiger Pfadstring sein.")

            plt.figure(figsize=(8, 6), dpi=300)

            nx.draw_networkx_edges(G, pos, edge_color='gray', width=0.5, alpha=0.3)

            node_colors = []
            node_sizes = []
            node_labels = {}

            # Bereich für Zoom berechnen - wird für die Bestimmung sichtbarer Stationen benötigt
            line_nodes = set(int(n) for line in railway_lines for n in line['path'] if int(n) in pos)
            if not line_nodes:
                raise ValueError("Keine gültigen Knoten in 'railway_lines' für Zoom gefunden.")

            x_coords = [pos[node][0] for node in line_nodes]
            y_coords = [pos[node][1] for node in line_nodes]
            padding = 2000
            x_min, x_max = min(x_coords) - padding, max(x_coords) + padding
            y_min, y_max = min(y_coords) - padding, max(y_coords) + padding
            x_range = x_max - x_min
            y_range = y_max - y_min

            # Stationen nach Kriterien filtern
            for node in G.nodes():
                station_name = G.nodes[node].get('station_name', '')
                is_main_node = G.nodes[node].get('end_station', False)
                is_selected = station_name in selected_stations

                # Prüfen, ob der Knoten auf einer der aktuellen Linien liegt
                is_on_current_line = int(node) in line_nodes if node in G else False

                # Prüfen, ob der Knoten innerhalb des sichtbaren Bereichs liegt
                if node in pos:
                    node_x, node_y = pos[node]
                    is_visible = (x_min <= node_x <= x_max) and (y_min <= node_y <= y_max)
                else:
                    is_visible = False

                # Farbzuweisungen
                if G.nodes[node].get('type') == 'center':
                    node_colors.append('orange')
                    node_sizes.append(80)
                    # Nur Labels für ausgewählte Stationen, wenn sie auf einer aktuellen Linie liegen
                    if is_visible and (is_main_node or (is_selected and is_on_current_line)):
                        node_labels[node] = station_name
                elif is_main_node:
                    node_colors.append('green')
                    node_sizes.append(100)
                    # Nur Labels für Hauptknoten innerhalb des sichtbaren Bereichs
                    if is_visible:
                        node_labels[node] = station_name
                elif any(int(node) in [int(n) for n in line.get('path', [])] for line in railway_lines):
                    node_colors.append('blue')
                    node_sizes.append(80)
                    # Nur Labels für ausgewählte Stationen, wenn sie auf einer aktuellen Linie liegen
                    if is_visible and is_selected and is_on_current_line:
                        node_labels[node] = station_name
                else:
                    node_colors.append('lightgray')
                    node_sizes.append(30)
                    # Nur Labels für ausgewählte Stationen, wenn sie auf einer aktuellen Linie liegen
                    if is_visible and is_selected and is_on_current_line:
                        node_labels[node] = station_name

            # Knoten zeichnen
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7)

            scale_factor = max(x_range, y_range) / 100  # dynamischer Offset-Basiswert

            # Kanten zählen für Offset-Berechnung
            edge_count = {}
            for line in railway_lines:
                path = [int(n) for n in line['path']]
                for j in range(len(path) - 1):
                    edge = tuple(sorted([path[j], path[j + 1]]))
                    edge_count[edge] = edge_count.get(edge, 0) + 1

            legend_handles = []

            # Linien zeichnen
            for i, line in enumerate(railway_lines):
                path = [int(n) for n in line['path']]
                name = line.get('name', f'Linie {i + 1}')

                if color_dict and name in color_dict:
                    color = color_dict[name]
                elif 'color' in line:
                    color = line['color']
                else:
                    color = zvv_colors[i % len(zvv_colors)]

                label = f"{name}:\n{line.get('start_station', '?')} – {line.get('end_station', '?')}"
                line_segments = []

                for j in range(len(path) - 1):
                    node_a, node_b = path[j], path[j + 1]
                    if node_a not in pos or node_b not in pos:
                        continue

                    start_pos = pos[node_a]
                    end_pos = pos[node_b]

                    dx = end_pos[0] - start_pos[0]
                    dy = end_pos[1] - start_pos[1]
                    length = np.hypot(dx, dy)
                    if length == 0:
                        continue

                    perpx = -dy / length
                    perpy = dx / length
                    edge = tuple(sorted([node_a, node_b]))
                    total_lines = edge_count[edge]

                    line_position = 0
                    for k, other_line in enumerate(railway_lines):
                        if k >= i:
                            break
                        other_path = [int(n) for n in other_line['path']]
                        for l in range(len(other_path) - 1):
                            if tuple(sorted([other_path[l], other_path[l + 1]])) == edge:
                                line_position += 1
                                break

                    offset = (line_position - (total_lines - 1) / 2) * scale_factor

                    start_offset = (
                        start_pos[0] + perpx * offset,
                        start_pos[1] + perpy * offset
                    )
                    end_offset = (
                        end_pos[0] + perpx * offset,
                        end_pos[1] + perpy * offset
                    )

                    segment, = plt.plot(
                        [start_offset[0], end_offset[0]],
                        [start_offset[1], end_offset[1]],
                        color=color, linewidth=2.5, zorder=10
                    )
                    line_segments.append(segment)

                if line_segments:
                    legend_line = Line2D([0], [0], color=color, lw=2, label=label)
                    legend_handles.append(legend_line)

            # Stationsnamen mit weißlichem Hintergrund platzieren und innerhalb der Karte halten
            ax = plt.gca()
            for node, label in node_labels.items():
                x, y = pos[node]

                # Bestimme Position basierend auf Quadranten (relative Position im Bild)
                # Standardwerte setzen
                ha = 'center'  # horizontale Ausrichtung
                va = 'bottom'  # vertikale Ausrichtung
                x_offset = 0
                y_offset = y_range * 0.01  # Standard y-Offset über dem Knoten

                # X-Position: Links, Mitte oder Rechts des Bildes
                x_rel_pos = (x - x_min) / x_range  # relative x-Position (0 bis 1)

                if x_rel_pos < 0.33:  # Links im Bild
                    ha = 'left'
                    x_offset = y_range * 0.01
                elif x_rel_pos > 0.67:  # Rechts im Bild
                    ha = 'right'
                    x_offset = -y_range * 0.01

                # Verhindere, dass Labels am unteren Rand nach unten zeigen
                if y < y_min + y_range * 0.1:  # Nahe am unteren Rand
                    va = 'bottom'
                    y_offset = abs(y_offset)

                # Text mit angepasster Ausrichtung und Position
                text = ax.text(
                    x + x_offset,
                    y + y_offset,
                    label,
                    fontsize=9,
                    ha=ha,
                    va=va,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                    zorder=1000
                )

            # Legende mit größerer Schrift
            plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.02, 1),
                       fontsize=12, framealpha=0.8)

            plt.title("Generated S-Bahn lines", fontsize=14, pad=20)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.xlabel('X-Koordinate', fontsize=10)
            plt.ylabel('Y-Koordinate', fontsize=10)

            # Zoom auf Linienbereich
            plt.axis('equal')
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

            # === Nordpfeil überarbeitet ===
            arrow_pos_x = 0.92  # Position oben rechts
            arrow_pos_y = 0.92  # Position oben rechts

            # "N" Beschriftung unter dem Pfeil platzieren
            ax.text(arrow_pos_x, arrow_pos_y - 0.04, "N",
                    fontsize=16, weight='bold',
                    ha='center', va='center',
                    transform=ax.transAxes,
                    zorder=1000,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

            # Pfeil zeichnen (oberhalb der Beschriftung)
            arrow = FancyArrowPatch((arrow_pos_x, arrow_pos_y - 0.01),
                                   (arrow_pos_x, arrow_pos_y + 0.03),
                                   color='black',
                                   lw=2,
                                   arrowstyle='->',
                                   mutation_scale=15,
                                   transform=ax.transAxes,
                                   zorder=1000)
            ax.add_patch(arrow)

            # === Maßstab in der Mitte des unteren Kartenrandes hinzufügen ===
            scale_length = 5000  # 5 km
            scale_x = x_min + (x_range / 2) - (scale_length / 2)  # Zentriert am unteren Rand
            scale_y = y_min + 0.05 * y_range  # Nahe dem unteren Rand
            scale_height = y_range * 0.01

            # Maßstabsbalken mit weißem Hintergrund für bessere Sichtbarkeit
            scale_bg = plt.Rectangle((scale_x - scale_length*0.05, scale_y - scale_height*1.5),
                                    scale_length*1.1, scale_height*5,
                                    facecolor='white', alpha=0.7, zorder=999)
            ax.add_patch(scale_bg)

            scale_rect = plt.Rectangle((scale_x, scale_y), scale_length, scale_height,
                                      facecolor='black', edgecolor='black', zorder=1000)
            ax.add_patch(scale_rect)

            # Maßstabstext
            ax.text(scale_x + scale_length / 2, scale_y + scale_height * 2,
                   "5 km", ha='center', va='bottom', fontsize=12, zorder=1000)

            plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.2, format='png')
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


def plot_cumulative_cost_distribution(df, output_path="plot/cumulative_cost_distribution.png", color_dict=None, group_by='development'):
    """
    Erstellt eine kumulative Wahrscheinlichkeitsverteilung des Nettonutzens
    für alle im DataFrame enthaltenen Entwicklungen.

    Args:
        df: DataFrame mit den Kostendaten (monetized_savings_total)
        output_path: Pfad zum Speichern der Abbildung
        color_dict: Optional color dictionary for developments/lines
        group_by: Column to group by ('development' or 'line_name')
    """
    # Mittelwert für jede Entwicklung/Linie berechnen
    mean_by_dev = df.groupby(group_by)['monetized_savings_total'].mean().reset_index()

    # Sortieren nach Mittelwert für konsistente Farben/Legende
    sorted_devs = mean_by_dev.sort_values(by='monetized_savings_total', ascending=False)
    dev_ids_sorted = sorted_devs[group_by].tolist()

    # Erstellen einer Figur
    plt.figure(figsize=(10, 6))

    # Farbpalette generieren (ausreichend viele Farben)
    if color_dict is None:
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i % 20) for i in range(len(dev_ids_sorted))]
        color_dict = {dev_id: colors[i] for i, dev_id in enumerate(dev_ids_sorted)}

    # Kumulative Verteilungen plotten
    for i, dev_id in enumerate(dev_ids_sorted):
        dev_data = df[df[group_by] == dev_id]
        values = dev_data['monetized_savings_total'].dropna().values / 1_000_000  # in Mio. CHF
        values = np.sort(values)
        y_values = np.arange(1, len(values) + 1) / len(values)

        # Legendenlabel
        mean_value = values.mean()
        label = f"Line {dev_id}: {mean_value:.1f} Mio. CHF"

        # Linie zeichnen
        color = color_dict.get(dev_id, f"C{i % 10}")  # Fallback zu matplotlib-Standardfarben
        plt.plot(values, y_values, '-', color=color, linewidth=2, label=label)

    # Nulllinie
    plt.axvline(x=0, color='lightgray', linestyle='-', linewidth=1)

    # Beschriftungen und Formatierung
    plt.xlabel('Monetised travel time savings [CHF million]', fontsize=12)
    plt.ylabel('Cumulative probability', fontsize=12)
    plt.title('Cumulative probability distribution of the net benefit \nof all developments considered', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Lines with moderate utility', fontsize=9, title_fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

    # Y-Achse 0–1
    plt.ylim(0, 1.05)

    # X-Achse skalieren
    x_min = df['monetized_savings_total'].min() / 1_000_000
    x_max = df['monetized_savings_total'].max() / 1_000_000
    plt.xlim(x_min - 5, x_max + 5)

    # Zusätzliche horizontale Linien
    for q in [0.25, 0.5, 0.75]:
        plt.axhline(y=q, color='darkgray', linestyle=':', alpha=0.7)
        plt.text(x_max + 3, q, f'{int(q * 100)}%', va='center', fontsize=9, color='darkgray')

    plt.tight_layout()

    # Verzeichnis anlegen
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Speichern
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_flow_graph(flow_graph, output_path=None, title="Passenger flows on the rail network",
                    node_size=100, node_color='skyblue', edge_scale=0.001, figsize=(20, 16),
                    selected_stations=None, plot_perimeter=False, style='absolute'):
    """
    Visualisiert einen Graph mit Passagierflüssen, wobei die Liniendicke proportional zum Fluss ist.
    Optional können nur bestimmte Stationen mit Namen beschriftet werden.

    Parameters:
        flow_graph (nx.DiGraph): Der Graph mit Flussattributen an den Kanten
        output_path (str, optional): Pfad zum Speichern der Grafik, wenn None wird die Grafik angezeigt
        title (str): Titel der Grafik
        node_size (int): Größe der Knoten
        node_color (str): Farbe der Knoten
        edge_scale (float): Skalierungsfaktor für die Kantendicke
        figsize (tuple): Größe der Abbildung (Breite, Höhe)
        selected_stations (list of str, optional): Liste von Stationsnamen, die beschriftet werden sollen
        plot_perimeter (bool): Ob der Perimeter-Polygon gezeichnet werden soll
        style (str): 'absolute' für absolute Flüsse oder 'difference' für Differenzdarstellung

    Returns:
        matplotlib.figure.Figure: Die erstellte Abbildung
    """

    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, LinearSegmentedColormap

    fig, ax = plt.subplots(figsize=figsize)
    if plot_perimeter:
        # Dein Polygon (leicht angepasst, damit es geschlossen ist)
        polygon_coords = [
            (2700989.862, 1235663.403),
            (2708491.515, 1239608.529),
            (2694972.602, 1255514.900),
            (2687415.817, 1251056.404),
            (2700989.862, 1235663.403)  # schliesst das Polygon explizit
        ]

        # Polygon hinzufügen – nur Umrandung, keine Füllung
        outline_polygon = plotpolygon(
            polygon_coords,
            closed=True,
            facecolor='none',
            edgecolor='gray',
            linewidth=1.5,
            zorder=-10  # ganz im Hintergrund
        )

        # Polygon ins Axes-Objekt einfügen
        ax.add_patch(outline_polygon)
    pos = nx.get_node_attributes(flow_graph, 'position')

    if len(pos) != flow_graph.number_of_nodes():
        raise ValueError(f"Es fehlen Positionen für einige Knoten: {len(pos)} von {flow_graph.number_of_nodes()}")

    combined_flows = {}
    for source, target, data in flow_graph.edges(data=True):
        edge_key = tuple(sorted([source, target]))
        flow = data.get('flow', 0)
        combined_flows[edge_key] = combined_flows.get(edge_key, 0) + flow

    flows = list(combined_flows.values())
    if not flows:
        raise ValueError("No passenger flows were found in the graph")

    min_flow = min(flows)
    max_flow = max(flows)

    # Farbskala je nach Stil auswählen
    if style == 'absolute':
        cmap = plt.cm.viridis_r
        norm = Normalize(vmin=min_flow, vmax=max_flow)
        cbar_label = 'Combined passenger flow'
    else:  # 'difference' style
        # Erstelle eine eigene Rot-Grau-Grün Farbskala für Differenzwerte
        colors = [(0.8, 0.0, 0.0), (0.85, 0.85, 0.85), (0.0, 0.6, 0.0)]  # rot, grau, grün
        cmap = LinearSegmentedColormap.from_list('RdGrGn', colors)

        # Finde das Maximum der absoluten Differenzwerte für symmetrische Skala
        max_abs_flow = max(abs(min_flow), abs(max_flow))
        norm = Normalize(vmin=-max_abs_flow, vmax=max_abs_flow)
        cbar_label = 'Flow change (difference)'

    nx.draw_networkx_nodes(flow_graph, pos,
                           node_size=node_size,
                           node_color=node_color,
                           edgecolors='black',
                           alpha=0.7,
                           ax=ax)

    # Wenn style=='difference', sammle alle Flows > 200 und zeige nur die größten an
    if style == 'difference':
        # Sammle alle Kanten mit Flows >= 200
        edges_with_flows = [(edge_key, total_flow) for edge_key, total_flow in combined_flows.items()
                            if abs(total_flow) >= 200]

        # Sortiere nach absolutem Flow-Wert (absteigend)
        edges_with_flows.sort(key=lambda x: abs(x[1]), reverse=True)

        # Bereite Speicher für bereits belegte Positionen vor
        label_positions = []

        # Minimale Distanz zwischen Beschriftungen, um Überlappungen zu vermeiden
        min_distance = 800  # Anpassbar je nach Skalierung

    # Zeichne die Kanten
    for edge_key, total_flow in combined_flows.items():
        source, target = edge_key
        # Die Liniendicke basiert auf dem Absolutwert (in beiden Modi)
        abs_flow = abs(total_flow)
        width = max(0.5, min(10, abs_flow * edge_scale))

        # Die Farbe basiert auf dem Stil
        edge_color = cmap(norm(total_flow))

        nx.draw_networkx_edges(flow_graph, pos,
                               edgelist=[(source, target)],
                               width=width,
                               edge_color=[edge_color],
                               alpha=0.7,
                               arrows=False,
                               ax=ax)

    # Wenn style=='difference', zeige Werte auf den Kanten
    if style == 'difference':
        for edge_key, total_flow in edges_with_flows:
            source, target = edge_key
            # Mittelpunkt der Kante berechnen
            x1, y1 = pos[source]
            x2, y2 = pos[target]
            x_mid = (x1 + x2) / 2
            y_mid = (y1 + y2) / 2

            # Prüfen, ob Position zu nah an bereits vorhandenen Labels ist
            too_close = False
            for x_pos, y_pos in label_positions:
                if ((x_mid - x_pos) ** 2 + (y_mid - y_pos) ** 2) < min_distance ** 2:
                    too_close = True
                    break

            # Nur zeichnen, wenn genügend Abstand zu anderen Labels
            if not too_close:
                ax.text(x_mid, y_mid, f"{int(total_flow)}", fontsize=8, ha='center', va='center',
                        bbox=dict(facecolor='white', alpha=0.7, pad=0.1, boxstyle='round', edgecolor='none'),
                        zorder=5)
                # Position merken
                label_positions.append((x_mid, y_mid))

    # Knotennamen anzeigen, mit leichtem weißem Hintergrund ohne Rahmen
    if selected_stations:
        labels = {node: node for node in flow_graph.nodes if node in selected_stations}
        # Verschiebe die Beschriftung nach unten links
        label_offset = (-1000, -600)  # x- und y-Verschiebung

        for node, label in labels.items():
            x, y = pos[node][0] + label_offset[0], pos[node][1] + label_offset[1]
            ax.text(x, y, label, fontsize=12, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, pad=0.1, boxstyle='round', edgecolor='none'),
                    zorder=10)
    else:
        # Für alle Knoten einen leichten weißen Hintergrund hinzufügen
        for node in flow_graph.nodes():
            x, y = pos[node]
            ax.text(x, y, node, fontsize=8, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, pad=0.1, boxstyle='round', edgecolor='none'),
                    zorder=10)

    # Kleinere Legende + 5k Tickformat
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Manuell positionierte, kürzere Farbskala (gleich breit)
    cbar_ax = fig.add_axes([0.92, 0.25, 0.07, 0.3])  # [left, bottom, width, height]
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label(cbar_label, fontsize=10)
    ticks = cbar.get_ticks()
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f'{int(t / 1000)}k' for t in ticks])

    total_flow = sum(flows)
    if style == 'absolute':
        text_info = (f"Total flow: {total_flow:.0f} passengers\n"
                     f"Max. flow: {max_flow:.0f}\n"
                     f"Min. flow: {min_flow:.0f}\n"
                     f"Number of edges: {len(combined_flows)}")
    else:
        pos_sum = sum(flow for flow in flows if flow > 0)
        neg_sum = sum(flow for flow in flows if flow < 0)
        text_info = (f"Net change: {total_flow:.0f} passengers\n"
                     f"Positive change: +{pos_sum:.0f}\n"
                     f"Negative change: {neg_sum:.0f}\n"
                     f"Number of edges: {len(combined_flows)}")

    plt.figtext(0.01, 0.01, text_info, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

    ax.set_title(title, fontsize=16)
    ax.set_aspect('equal')  # ← wichtige Zeile für unverzerrte Darstellung
    ax.axis('off')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Grafik wurde gespeichert unter: {output_path}")
    else:
        plt.show()

    return fig

def plot_line_flows(line_flow_graph, s_bahn_geopackage_path, output_path):
    """
    Erstellt für jede S-Bahn-Linie einen separaten Barplot, der den Fluss zwischen
    aufeinanderfolgenden Stationspaaren in der richtigen Reihenfolge anzeigt.

    Parameters:
        line_flow_graph (nx.DiGraph): Der linienbasierte Graph mit Flussattributen
        s_bahn_geopackage_path (str): Pfad zum GeoPackage mit den S-Bahn-Linien
        output_path (str): Pfad zum Speichern der Ausgabedatei
    """
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import os

    # Lade das GeoPackage mit den S-Bahn-Linien
    s_bahn_lines = gpd.read_file(s_bahn_geopackage_path)

    # Dictionary zur Sammlung der kombinierten Flüsse pro Linie und Stationspaar
    flows = {}

    # Sammle alle gerichteten Flüsse aus dem Graph
    for source, target, data in line_flow_graph.edges(data=True):
        if 'service' in data and 'flow' in data:
            service = data['service']
            flow = data['flow']

            # Erstelle einen eindeutigen Schlüssel für dieses Stationspaar
            key = (service, source, target)

            # Speichere den Fluss
            if key not in flows:
                flows[key] = flow
            else:
                flows[key] += flow

    # Dictionary für die Linien und ihre geordneten Stationen
    lines_data = {}

    # Verarbeite die S-Bahn-Linien aus dem GeoPackage
    for service in s_bahn_lines['Service'].unique():
        # Filtere nur Einträge für diese Linie mit Direction "A"
        line_segments = s_bahn_lines[(s_bahn_lines['Service'] == service) &
                                     (s_bahn_lines['Direction'] == 'A')]

        if len(line_segments) == 0:
            continue

        # Finde den Startpunkt (FromEnd = '1')
        start_segment = line_segments[line_segments['FromEnd'] == '1']

        if len(start_segment) == 0:
            # Falls kein expliziter Startpunkt, nimm einfach den ersten Eintrag
            ordered_segments = [line_segments.iloc[0]]
        else:
            ordered_segments = [start_segment.iloc[0]]

        # Baue die Sequenz der Segmente auf
        current_station = ordered_segments[0]['ToStation']
        remaining_segments = line_segments[~line_segments.index.isin([ordered_segments[0].name])]

        while len(remaining_segments) > 0:
            # Finde das nächste Segment, das mit der aktuellen Station beginnt
            next_segment = remaining_segments[remaining_segments['FromStation'] == current_station]

            if len(next_segment) == 0:
                # Kein weiteres Segment gefunden
                break

            # Füge das nächste Segment hinzu
            ordered_segments.append(next_segment.iloc[0])

            # Aktualisiere die aktuelle Station und entferne das verarbeitete Segment
            current_station = ordered_segments[-1]['ToStation']
            remaining_segments = remaining_segments[~remaining_segments.index.isin([ordered_segments[-1].name])]

        # Erstelle eine Liste von Kanten für diese Linie in der richtigen Reihenfolge
        edges = []
        for segment in ordered_segments:
            source = segment['FromStation']
            target = segment['ToStation']

            # Kombiniere die Flüsse in beide Richtungen
            flow_forward = flows.get((service, source, target), 0)
            flow_backward = flows.get((service, target, source), 0)
            total_flow = flow_forward + flow_backward

            edges.append({
                'source': source,
                'target': target,
                'flow': total_flow
            })

        lines_data[service] = edges

    # Anzahl der S-Bahn-Linien bestimmen
    n_lines = len(lines_data)
    if n_lines == 0:
        print("Keine S-Bahn-Linien im Graph gefunden.")
        return

    # Figurgröße basierend auf Anzahl der Linien anpassen
    fig, axes = plt.subplots(n_lines, 1, figsize=(12, 4 * n_lines))
    if n_lines == 1:
        axes = [axes]  # Für konsistente Indexierung bei nur einer Linie

    # Sortierte Liste der Linien (S1, S2, S3, ...)
    sorted_lines = sorted(lines_data.keys())

    # Für jede Linie einen Barplot erstellen
    for i, service in enumerate(sorted_lines):
        edges = lines_data[service]

        # Edge-Labels und Flows extrahieren
        edge_labels = [f"{edge['source']} → {edge['target']}" for edge in edges]
        flows = [edge['flow'] / 1000 for edge in edges]  # Werte in Tausend

        # Plot für diese Linie erstellen
        ax = axes[i]
        bars = ax.bar(edge_labels, flows, color='steelblue')

        # Achsenbeschriftung
        ax.set_xlabel('Station connection', fontsize=10)
        ax.set_ylabel('Utilisation (in thousands)', fontsize=10)
        ax.set_title(f'Line utilisation {service}', fontsize=12)

        # Werte über den Balken anzeigen
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}k',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        # X-Achsenbeschriftungen rotieren für bessere Lesbarkeit
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)

        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Layout optimieren
    plt.tight_layout()

    # Sicherstellen, dass das Verzeichnis existiert
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Plot speichern
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Linienauslastungs-Plots gespeichert unter: {output_path}")

    return fig


# ================================================================================
# PIPELINE COMPARISON PLOTTING (OLD vs NEW)
# ================================================================================

def create_ranked_pipeline_comparison_plots(
    plot_directory="plots",
    top_n_list=[5, 10]
):
    """
    Create comparison plots showing top N developments ranked by net benefit,
    comparing old pipeline (8 trains/track) vs new pipeline (capacity interventions).

    Args:
        plot_directory: Base directory for plots (default: "plots")
        top_n_list: List of top N values to generate plots for (default: [5, 10])

    Outputs:
        For each N in top_n_list, generates 3 plot types in plots/Benefits_Pipeline_Comparison/:
        - ranked_topN_boxplot_cba_comparison.png
        - ranked_topN_boxplot_net_benefit_comparison.png
        - ranked_topN_cost_savings_comparison.png
    """
    print("\n" + "="*80)
    print("GENERATING PIPELINE COMPARISON PLOTS (OLD vs NEW)")
    print("="*80 + "\n")

    # Create output directory
    comparison_dir = os.path.join(plot_directory, "Benefits_Pipeline_Comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Load data from both pipelines
    new_data_path = "data/costs/total_costs_raw.csv"
    old_data_path = "data/costs/total_costs_raw_old.csv"

    if not os.path.exists(new_data_path):
        print(f"  ⚠ New pipeline data not found: {new_data_path}")
        print(f"  → Skipping pipeline comparison plots")
        return

    if not os.path.exists(old_data_path):
        print(f"  ⚠ Old pipeline data not found: {old_data_path}")
        print(f"  → Skipping pipeline comparison plots")
        return

    print(f"  Loading new pipeline data: {new_data_path}")
    print(f"  Loading old pipeline data: {old_data_path}")

    # Load and prepare both datasets
    df_new = _prepare_pipeline_data(new_data_path, pipeline='new')
    df_old = _prepare_pipeline_data(old_data_path, pipeline='old')

    # Rank developments by NEW pipeline mean net benefit
    print(f"\n  Ranking developments by new pipeline mean net benefit...")
    rankings = df_new.groupby('development')['total_net_benefit'].mean().sort_values(ascending=False)

    print(f"  Total developments: {len(rankings)}")
    print(f"  Top development: {rankings.index[0]} (mean net benefit: {rankings.iloc[0]/1e6:.2f} M CHF)")

    # Get ZVV colors
    zvv_colors = pp.zvv_colors

    # Generate plots for each top N
    for N in top_n_list:
        print(f"\n  {'='*70}")
        print(f"  Generating plots for TOP {N} developments")
        print(f"  {'='*70}")

        # Get top N development IDs
        top_dev_ids = rankings.head(N).index.tolist()

        print(f"  Top {N} developments: {top_dev_ids}")

        # Filter both datasets to top N
        df_new_top = df_new[df_new['development'].isin(top_dev_ids)].copy()
        df_old_top = df_old[df_old['development'].isin(top_dev_ids)].copy()

        # Check for missing developments in old pipeline
        new_devs = set(df_new_top['development'].unique())
        old_devs = set(df_old_top['development'].unique())
        missing_in_old = new_devs - old_devs

        if missing_in_old:
            print(f"  ⚠ Warning: {len(missing_in_old)} developments in new pipeline not found in old pipeline")
            print(f"    Missing: {missing_in_old}")
            print(f"    These will be skipped in comparison plots")
            # Filter out missing developments
            top_dev_ids = [dev for dev in top_dev_ids if dev in old_devs]
            df_new_top = df_new_top[df_new_top['development'].isin(top_dev_ids)]

        # Combine datasets
        df_combined = pd.concat([df_new_top, df_old_top], ignore_index=True)

        # Assign colors to developments (consistent across both pipelines)
        color_map = {dev: zvv_colors[i % len(zvv_colors)]
                     for i, dev in enumerate(top_dev_ids)}

        # Generate each plot type
        print(f"\n  Generating boxplot CBA comparison...")
        _plot_boxplot_cba_comparison(df_combined, color_map, top_dev_ids, N, comparison_dir)

        print(f"  Generating boxplot net benefit comparison...")
        _plot_boxplot_net_benefit_comparison(df_combined, color_map, top_dev_ids, N, comparison_dir)

        print(f"  Generating cost savings comparison...")
        _plot_cost_savings_comparison(df_combined, color_map, top_dev_ids, N, comparison_dir)

    print(f"\n  {'='*70}")
    print(f"  ✓ All pipeline comparison plots saved to: {comparison_dir}")
    print(f"  {'='*70}\n")


def _prepare_pipeline_data(file_path, pipeline='new'):
    """
    Load and prepare pipeline data with consistent structure.

    Args:
        file_path: Path to CSV file
        pipeline: 'new' or 'old' to label the data source

    Returns:
        DataFrame with columns: development, line_name, scenario, pipeline,
                                monetized_savings_total, total_costs, total_net_benefit, cba_ratio
    """
    df = pd.read_csv(file_path)

    # Rename columns to match expected format
    if 'ID_new' in df.columns:
        df.rename(columns={'ID_new': 'scenario'}, inplace=True)

    # Calculate derived metrics
    df['monetized_savings_total'] = df['monetized_savings_total'].abs()
    df['total_costs'] = (df['TotalConstructionCost'] +
                         df['TotalMaintenanceCost'] +
                         df['TotalUncoveredOperatingCost'])
    df['total_net_benefit'] = df['monetized_savings_total'] - df['total_costs']
    df['cba_ratio'] = df['monetized_savings_total'] / df['total_costs']

    # Add pipeline identifier
    df['pipeline'] = pipeline

    # Load Sline data for line_name generation
    try:
        sline_data = gpd.read_file("data/Network/processed/updated_new_links.gpkg")[['dev_id', 'Sline']]
        sline_data = sline_data.groupby('dev_id')['Sline'].first().reset_index()
        df = df.merge(sline_data, left_on='development', right_on='dev_id', how='left')
        if 'dev_id' in df.columns:
            df = df.drop(columns=['dev_id'])
    except Exception as e:
        print(f"    ⚠ Warning: Could not load Sline data: {e}")

    # Generate line_name
    df['line_name'] = None

    # Small developments (100xxx): format as "N_Sline"
    if 'Sline' in df.columns:
        df.loc[df['development'] < 101000, 'line_name'] = \
            df.loc[df['development'] < 101000].apply(
                lambda row: f"{int(row['development'] - 100000)}_{row['Sline']}"
                if pd.notna(row['Sline']) else str(int(row['development'] - 100000)),
                axis=1)
    else:
        df.loc[df['development'] < 101000, 'line_name'] = \
            df['development'].loc[df['development'] < 101000].map(
                lambda x: str(int(x - 100000)))

    # Large developments (101xxx): format as "XN"
    df.loc[df['development'] >= 101000, 'line_name'] = \
        df['development'].loc[df['development'] >= 101000].map(
            lambda x: f"X{int(x - 101000)}")

    return df


def _plot_boxplot_cba_comparison(df_combined, color_map, dev_order, N, output_dir):
    """Generate boxplot comparing CBA ratios between old and new pipelines."""

    # Prepare data for plotting
    plot_data = []
    for dev_id in dev_order:
        dev_data = df_combined[df_combined['development'] == dev_id]
        line_name = dev_data['line_name'].iloc[0]

        # New pipeline data
        new_data = dev_data[dev_data['pipeline'] == 'new']
        for _, row in new_data.iterrows():
            plot_data.append({
                'development': dev_id,
                'line_name': line_name,
                'cba_ratio': row['cba_ratio'],
                'pipeline': 'New',
                'position': dev_order.index(dev_id) * 2  # Even positions
            })

        # Old pipeline data
        old_data = dev_data[dev_data['pipeline'] == 'old']
        for _, row in old_data.iterrows():
            plot_data.append({
                'development': dev_id,
                'line_name': line_name,
                'cba_ratio': row['cba_ratio'],
                'pipeline': 'Old',
                'position': dev_order.index(dev_id) * 2 + 0.5  # Odd positions (offset)
            })

    plot_df = pd.DataFrame(plot_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(max(7, N * 1.2), 5), dpi=300)

    # Plot boxes for each development
    for i, dev_id in enumerate(dev_order):
        dev_subset = plot_df[plot_df['development'] == dev_id]
        line_name = dev_subset['line_name'].iloc[0]
        color = color_map[dev_id]

        # New pipeline box
        new_subset = dev_subset[dev_subset['pipeline'] == 'New']
        if not new_subset.empty:
            bp_new = ax.boxplot(
                [new_subset['cba_ratio'].values],
                positions=[i * 2.5],
                widths=0.4,
                patch_artist=True,
                showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5},
                flierprops={"markersize": 3},
                boxprops={"facecolor": color, "alpha": 1.0, "edgecolor": "black", "linewidth": 0.8},
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"color": "black", "linewidth": 0.8},
                capprops={"color": "black", "linewidth": 0.8}
            )

        # Old pipeline box
        old_subset = dev_subset[dev_subset['pipeline'] == 'Old']
        if not old_subset.empty:
            bp_old = ax.boxplot(
                [old_subset['cba_ratio'].values],
                positions=[i * 2.5 + 0.6],
                widths=0.4,
                patch_artist=True,
                showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5},
                flierprops={"markersize": 3},
                boxprops={"facecolor": color, "alpha": 0.4, "edgecolor": "black", "linewidth": 0.8},
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"color": "black", "linewidth": 0.8},
                capprops={"color": "black", "linewidth": 0.8}
            )

    # Set x-axis labels
    line_names = [plot_df[plot_df['development'] == dev]['line_name'].iloc[0] for dev in dev_order]
    ax.set_xticks([i * 2.5 + 0.3 for i in range(len(dev_order))])
    ax.set_xticklabels(line_names, rotation=90)

    # Labels and formatting
    ax.set_xlabel('Development', fontsize=12)
    ax.set_ylabel('Cost-benefit ratio', fontsize=12)
    ax.axhline(y=1, color='red', linestyle='-', alpha=0.5, label='Break-even (CBA = 1)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor='gray', alpha=1.0, edgecolor='black', label='New Pipeline'),
        mpatches.Patch(facecolor='gray', alpha=0.4, edgecolor='black', label='Old Pipeline'),
        mlines.Line2D([0], [0], marker='o', color='black', label='Mean', markersize=5, linestyle='None'),
        mlines.Line2D([0], [0], color='red', linestyle='-', alpha=0.5, label='Break-even (CBA = 1)')
    ]
    ax.legend(handles=legend_handles, loc='best', frameon=True, fontsize=10, fancybox=True, shadow=True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"ranked_top{N}_boxplot_cba_comparison.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {output_path}")


def _plot_boxplot_net_benefit_comparison(df_combined, color_map, dev_order, N, output_dir):
    """Generate boxplot comparing net benefits between old and new pipelines."""

    # Prepare data for plotting
    plot_data = []
    for dev_id in dev_order:
        dev_data = df_combined[df_combined['development'] == dev_id]
        line_name = dev_data['line_name'].iloc[0]

        # New pipeline data
        new_data = dev_data[dev_data['pipeline'] == 'new']
        for _, row in new_data.iterrows():
            plot_data.append({
                'development': dev_id,
                'line_name': line_name,
                'net_benefit': row['total_net_benefit'] / 1e6,  # Convert to millions
                'pipeline': 'New'
            })

        # Old pipeline data
        old_data = dev_data[dev_data['pipeline'] == 'old']
        for _, row in old_data.iterrows():
            plot_data.append({
                'development': dev_id,
                'line_name': line_name,
                'net_benefit': row['total_net_benefit'] / 1e6,  # Convert to millions
                'pipeline': 'Old'
            })

    plot_df = pd.DataFrame(plot_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(max(7, N * 1.2), 5), dpi=300)

    # Plot boxes for each development
    for i, dev_id in enumerate(dev_order):
        dev_subset = plot_df[plot_df['development'] == dev_id]
        color = color_map[dev_id]

        # New pipeline box
        new_subset = dev_subset[dev_subset['pipeline'] == 'New']
        if not new_subset.empty:
            bp_new = ax.boxplot(
                [new_subset['net_benefit'].values],
                positions=[i * 2.5],
                widths=0.4,
                patch_artist=True,
                showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5},
                flierprops={"markersize": 3},
                boxprops={"facecolor": color, "alpha": 1.0, "edgecolor": "black", "linewidth": 0.8},
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"color": "black", "linewidth": 0.8},
                capprops={"color": "black", "linewidth": 0.8}
            )

        # Old pipeline box
        old_subset = dev_subset[dev_subset['pipeline'] == 'Old']
        if not old_subset.empty:
            bp_old = ax.boxplot(
                [old_subset['net_benefit'].values],
                positions=[i * 2.5 + 0.6],
                widths=0.4,
                patch_artist=True,
                showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5},
                flierprops={"markersize": 3},
                boxprops={"facecolor": color, "alpha": 0.4, "edgecolor": "black", "linewidth": 0.8},
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"color": "black", "linewidth": 0.8},
                capprops={"color": "black", "linewidth": 0.8}
            )

    # Set x-axis labels
    line_names = [plot_df[plot_df['development'] == dev]['line_name'].iloc[0] for dev in dev_order]
    ax.set_xticks([i * 2.5 + 0.3 for i in range(len(dev_order))])
    ax.set_xticklabels(line_names, rotation=90)

    # Labels and formatting
    ax.set_xlabel('Development', fontsize=12)
    ax.set_ylabel('Net benefit in CHF million', fontsize=12)
    ax.axhline(y=0, color='red', linestyle='-', alpha=0.5, label='Break-even (Net benefit = 0)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor='gray', alpha=1.0, edgecolor='black', label='New Pipeline'),
        mpatches.Patch(facecolor='gray', alpha=0.4, edgecolor='black', label='Old Pipeline'),
        mlines.Line2D([0], [0], marker='o', color='black', label='Mean', markersize=5, linestyle='None'),
        mlines.Line2D([0], [0], color='red', linestyle='-', alpha=0.5, label='Break-even')
    ]
    ax.legend(handles=legend_handles, loc='best', frameon=True, fontsize=10, fancybox=True, shadow=True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"ranked_top{N}_boxplot_net_benefit_comparison.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {output_path}")


def _plot_boxplot_savings_comparison(df_combined, color_map, dev_order, N, output_dir):
    """Generate boxplot comparing travel time savings between old and new pipelines."""

    # Prepare data for plotting
    plot_data = []
    for dev_id in dev_order:
        dev_data = df_combined[df_combined['development'] == dev_id]
        line_name = dev_data['line_name'].iloc[0]

        # New pipeline data
        new_data = dev_data[dev_data['pipeline'] == 'new']
        for _, row in new_data.iterrows():
            plot_data.append({
                'development': dev_id,
                'line_name': line_name,
                'savings': row['monetized_savings_total'] / 1e6,  # Convert to millions
                'pipeline': 'New'
            })

        # Old pipeline data
        old_data = dev_data[dev_data['pipeline'] == 'old']
        for _, row in old_data.iterrows():
            plot_data.append({
                'development': dev_id,
                'line_name': line_name,
                'savings': row['monetized_savings_total'] / 1e6,  # Convert to millions
                'pipeline': 'Old'
            })

    plot_df = pd.DataFrame(plot_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(max(7, N * 1.2), 5), dpi=300)

    # Plot boxes for each development
    for i, dev_id in enumerate(dev_order):
        dev_subset = plot_df[plot_df['development'] == dev_id]
        color = color_map[dev_id]

        # New pipeline box
        new_subset = dev_subset[dev_subset['pipeline'] == 'New']
        if not new_subset.empty:
            bp_new = ax.boxplot(
                [new_subset['savings'].values],
                positions=[i * 2.5],
                widths=0.4,
                patch_artist=True,
                showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5},
                flierprops={"markersize": 3},
                boxprops={"facecolor": color, "alpha": 1.0, "edgecolor": "black", "linewidth": 0.8},
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"color": "black", "linewidth": 0.8},
                capprops={"color": "black", "linewidth": 0.8}
            )

        # Old pipeline box
        old_subset = dev_subset[dev_subset['pipeline'] == 'Old']
        if not old_subset.empty:
            bp_old = ax.boxplot(
                [old_subset['savings'].values],
                positions=[i * 2.5 + 0.6],
                widths=0.4,
                patch_artist=True,
                showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5},
                flierprops={"markersize": 3},
                boxprops={"facecolor": color, "alpha": 0.4, "edgecolor": "black", "linewidth": 0.8},
                medianprops={"color": "black", "linewidth": 1.5},
                whiskerprops={"color": "black", "linewidth": 0.8},
                capprops={"color": "black", "linewidth": 0.8}
            )

    # Set x-axis labels
    line_names = [plot_df[plot_df['development'] == dev]['line_name'].iloc[0] for dev in dev_order]
    ax.set_xticks([i * 2.5 + 0.3 for i in range(len(dev_order))])
    ax.set_xticklabels(line_names, rotation=90)

    # Labels and formatting
    ax.set_xlabel('Development', fontsize=12)
    ax.set_ylabel('Monetised travel time savings in million CHF', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor='gray', alpha=1.0, edgecolor='black', label='New Pipeline (Capacity Interventions)'),
        mpatches.Patch(facecolor='gray', alpha=0.4, edgecolor='black', label='Old Pipeline (8 trains/track)'),
        mlines.Line2D([0], [0], marker='o', color='black', label='Mean', markersize=5, linestyle='None')
    ]
    ax.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"ranked_top{N}_boxplot_savings_comparison.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {output_path}")


def _plot_violinplot_savings_comparison(df_combined, color_map, dev_order, N, output_dir):
    """Generate violinplot comparing travel time savings between old and new pipelines."""

    # Prepare data for plotting
    plot_data = []
    for dev_id in dev_order:
        dev_data = df_combined[df_combined['development'] == dev_id]
        line_name = dev_data['line_name'].iloc[0]

        # New pipeline data
        new_data = dev_data[dev_data['pipeline'] == 'new']
        for _, row in new_data.iterrows():
            plot_data.append({
                'development': dev_id,
                'line_name': line_name,
                'savings': row['monetized_savings_total'] / 1e6,
                'pipeline': 'New',
                'scenario': row['scenario']
            })

        # Old pipeline data
        old_data = dev_data[dev_data['pipeline'] == 'old']
        for _, row in old_data.iterrows():
            plot_data.append({
                'development': dev_id,
                'line_name': line_name,
                'savings': row['monetized_savings_total'] / 1e6,
                'pipeline': 'Old',
                'scenario': row['scenario']
            })

    plot_df = pd.DataFrame(plot_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(max(7, N * 1.2), 5), dpi=300)

    # Plot violins for each development
    for i, dev_id in enumerate(dev_order):
        dev_subset = plot_df[plot_df['development'] == dev_id]
        color = color_map[dev_id]

        # New pipeline violin
        new_subset = dev_subset[dev_subset['pipeline'] == 'New']
        if not new_subset.empty:
            parts_new = ax.violinplot(
                [new_subset['savings'].values],
                positions=[i * 2.5],
                widths=0.5,
                showmeans=False,
                showmedians=False,
                showextrema=False
            )
            for pc in parts_new['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(1.0)
                pc.set_edgecolor('black')
                pc.set_linewidth(0.8)

        # Old pipeline violin
        old_subset = dev_subset[dev_subset['pipeline'] == 'Old']
        if not old_subset.empty:
            parts_old = ax.violinplot(
                [old_subset['savings'].values],
                positions=[i * 2.5 + 0.6],
                widths=0.5,
                showmeans=False,
                showmedians=False,
                showextrema=False
            )
            for pc in parts_old['bodies']:
                pc.set_facecolor(color)
                pc.set_alpha(0.4)
                pc.set_edgecolor('black')
                pc.set_linewidth(0.8)

        # Add stripplot for individual points
        # New pipeline points
        if not new_subset.empty:
            unique_new = new_subset.drop_duplicates(subset=['scenario'])
            ax.scatter(
                [i * 2.5] * len(unique_new),
                unique_new['savings'].values,
                color='black',
                alpha=0.4,
                s=10,
                zorder=3
            )

        # Old pipeline points
        if not old_subset.empty:
            unique_old = old_subset.drop_duplicates(subset=['scenario'])
            ax.scatter(
                [i * 2.5 + 0.6] * len(unique_old),
                unique_old['savings'].values,
                color='black',
                alpha=0.4,
                s=10,
                zorder=3
            )

    # Set x-axis labels
    line_names = [plot_df[plot_df['development'] == dev]['line_name'].iloc[0] for dev in dev_order]
    ax.set_xticks([i * 2.5 + 0.3 for i in range(len(dev_order))])
    ax.set_xticklabels(line_names, rotation=90)

    # Labels and formatting
    ax.set_xlabel('Development', fontsize=12)
    ax.set_ylabel('Monetised travel time savings in million CHF', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor='gray', alpha=1.0, edgecolor='black', label='New Pipeline (Capacity Interventions)'),
        mpatches.Patch(facecolor='gray', alpha=0.4, edgecolor='black', label='Old Pipeline (8 trains/track)'),
        mlines.Line2D([], [], marker='o', color='black', alpha=0.4, linestyle='None', markersize=4, label='Individual Values')
    ]
    ax.legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"ranked_top{N}_violinplot_savings_comparison.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {output_path}")


def _plot_cost_savings_comparison(df_combined, color_map, dev_order, N, output_dir):
    """Generate stacked bar chart comparing costs and savings between old and new pipelines."""

    # Color scheme for cost components
    kosten_farben = {
        'TotalConstructionCost': '#a6bddb',
        'TotalMaintenanceCost': '#3690c0',
        'TotalUncoveredOperatingCost': '#034e7b',
        'monetized_savings_total': '#31a354'
    }

    # Aggregate data by development and pipeline
    summary_data = []
    for dev_id in dev_order:
        dev_data = df_combined[df_combined['development'] == dev_id]
        line_name = dev_data['line_name'].iloc[0]

        # New pipeline summary
        new_data = dev_data[dev_data['pipeline'] == 'new']
        if not new_data.empty:
            summary_data.append({
                'development': dev_id,
                'line_name': line_name,
                'pipeline': 'New',
                'TotalConstructionCost': new_data['TotalConstructionCost'].mean(),
                'TotalMaintenanceCost': new_data['TotalMaintenanceCost'].mean(),
                'TotalUncoveredOperatingCost': new_data['TotalUncoveredOperatingCost'].mean(),
                'monetized_savings_total': new_data['monetized_savings_total'].mean()
            })

        # Old pipeline summary
        old_data = dev_data[dev_data['pipeline'] == 'old']
        if not old_data.empty:
            summary_data.append({
                'development': dev_id,
                'line_name': line_name,
                'pipeline': 'Old',
                'TotalConstructionCost': old_data['TotalConstructionCost'].mean(),
                'TotalMaintenanceCost': old_data['TotalMaintenanceCost'].mean(),
                'TotalUncoveredOperatingCost': old_data['TotalUncoveredOperatingCost'].mean(),
                'monetized_savings_total': old_data['monetized_savings_total'].mean()
            })

    summary_df = pd.DataFrame(summary_data)

    # Create figure
    n_devs = len(dev_order)
    fig, ax = plt.subplots(figsize=(max(7, n_devs * 1.2), 5), dpi=300)

    bar_width = 0.4
    x_positions = []

    # Plot bars for each development
    for i, dev_id in enumerate(dev_order):
        dev_subset = summary_df[summary_df['development'] == dev_id]
        color = color_map[dev_id]

        # New pipeline bar (left)
        new_subset = dev_subset[dev_subset['pipeline'] == 'New']
        if not new_subset.empty:
            row = new_subset.iloc[0]
            x_pos = i * 2.5
            x_positions.append(x_pos)

            # Costs (negative)
            construction = -row['TotalConstructionCost'] / 1e6
            maintenance = -row['TotalMaintenanceCost'] / 1e6
            operating = -row['TotalUncoveredOperatingCost'] / 1e6

            ax.bar(x_pos, construction, width=bar_width,
                   color=kosten_farben['TotalConstructionCost'], edgecolor='black', linewidth=0.5)
            ax.bar(x_pos, maintenance, width=bar_width, bottom=construction,
                   color=kosten_farben['TotalMaintenanceCost'], edgecolor='black', linewidth=0.5)
            ax.bar(x_pos, operating, width=bar_width, bottom=construction + maintenance,
                   color=kosten_farben['TotalUncoveredOperatingCost'], edgecolor='black', linewidth=0.5)

            # Savings (positive) - with hatching to show new pipeline
            ax.bar(x_pos, row['monetized_savings_total'] / 1e6, width=bar_width,
                   color=color, hatch='////', edgecolor='black', linewidth=0.5, alpha=1.0)

        # Old pipeline bar (right)
        old_subset = dev_subset[dev_subset['pipeline'] == 'Old']
        if not old_subset.empty:
            row = old_subset.iloc[0]
            x_pos = i * 2.5 + 0.6

            # Costs (negative)
            construction = -row['TotalConstructionCost'] / 1e6
            maintenance = -row['TotalMaintenanceCost'] / 1e6
            operating = -row['TotalUncoveredOperatingCost'] / 1e6

            ax.bar(x_pos, construction, width=bar_width,
                   color=kosten_farben['TotalConstructionCost'], edgecolor='black', linewidth=0.5, alpha=0.4)
            ax.bar(x_pos, maintenance, width=bar_width, bottom=construction,
                   color=kosten_farben['TotalMaintenanceCost'], edgecolor='black', linewidth=0.5, alpha=0.4)
            ax.bar(x_pos, operating, width=bar_width, bottom=construction + maintenance,
                   color=kosten_farben['TotalUncoveredOperatingCost'], edgecolor='black', linewidth=0.5, alpha=0.4)

            # Savings (positive) - lighter color for old pipeline
            ax.bar(x_pos, row['monetized_savings_total'] / 1e6, width=bar_width,
                   color=color, hatch='...', edgecolor='black', linewidth=0.5, alpha=0.4)

    # Set x-axis labels
    line_names = [summary_df[summary_df['development'] == dev]['line_name'].iloc[0] for dev in dev_order]
    ax.set_xticks([i * 2.5 + 0.3 for i in range(len(dev_order))])
    ax.set_xticklabels(line_names, rotation=90)

    # Labels and formatting
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Development', fontsize=12)
    ax.set_ylabel('Value in CHF million', fontsize=12)
    ax.set_title('Costs and benefits comparison (Old vs New Pipeline)', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Legend - placed outside plot area
    legend_handles = [
        mpatches.Patch(color=kosten_farben['TotalConstructionCost'], label='Construction costs'),
        mpatches.Patch(color=kosten_farben['TotalMaintenanceCost'], label='Uncovered maintenance costs'),
        mpatches.Patch(color=kosten_farben['TotalUncoveredOperatingCost'], label='Uncovered operating costs'),
        mpatches.Patch(facecolor='gray', hatch='////', edgecolor='black', alpha=1.0, label='Travel time savings (New Pipeline)'),
        mpatches.Patch(facecolor='gray', hatch='...', edgecolor='black', alpha=0.4, label='Travel time savings (Old Pipeline)')
    ]
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=True, fontsize=10, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legend on the right
    output_path = os.path.join(output_dir, f"ranked_top{N}_cost_savings_comparison.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {output_path}")


# ================================================================================
# ALL DEVELOPMENTS PIPELINE COMPARISON (OLD vs NEW)
# ================================================================================

def create_all_developments_pipeline_comparison_plots(plot_directory="plots"):
    """
    Create comparison plots for ALL developments comparing old vs new pipelines.

    Generates:
    1. CBA comparison (grouped bars)
    2. Total cost change (diverging bars with red/green + hatching)
    3. Scenario viability comparison (grouped bars with percentages)
    4. Summary CSV with all metrics

    Args:
        plot_directory: Base directory for plots (default: "plots")

    Outputs:
        - all_developments_cba_comparison.png
        - all_developments_cost_change_comparison.png
        - all_developments_scenario_viability_comparison.png
        - all_developments_pipeline_comparison_summary.csv
    """
    print("\n" + "="*80)
    print("GENERATING ALL DEVELOPMENTS PIPELINE COMPARISON (OLD vs NEW)")
    print("="*80 + "\n")

    # Create output directory
    comparison_dir = os.path.join(plot_directory, "Benefits_Pipeline_Comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    # Load data from both pipelines
    new_data_path = "data/costs/total_costs_raw.csv"
    old_data_path = "data/costs/total_costs_raw_old.csv"

    if not os.path.exists(new_data_path):
        print(f"  ⚠ New pipeline data not found: {new_data_path}")
        print(f"  → Skipping all developments comparison plots")
        return

    if not os.path.exists(old_data_path):
        print(f"  ⚠ Old pipeline data not found: {old_data_path}")
        print(f"  → Skipping all developments comparison plots")
        return

    print(f"  Loading new pipeline data: {new_data_path}")
    print(f"  Loading old pipeline data: {old_data_path}")

    # Load and prepare both datasets
    df_new = _prepare_pipeline_data(new_data_path, pipeline='new')
    df_old = _prepare_pipeline_data(old_data_path, pipeline='old')

    # Get all developments
    all_dev_ids = sorted(df_new['development'].unique())
    print(f"\n  Total developments: {len(all_dev_ids)}")

    # Rank by NEW pipeline mean net benefit
    rankings = df_new.groupby('development')['total_net_benefit'].mean().sort_values(ascending=False)
    dev_order = rankings.index.tolist()

    # Get ZVV colors
    zvv_colors = pp.zvv_colors

    # Assign colors to developments
    color_map = {dev: zvv_colors[i % len(zvv_colors)] for i, dev in enumerate(dev_order)}

    # Combine datasets
    df_combined = pd.concat([df_new, df_old], ignore_index=True)

    # Generate plots
    print(f"\n  Generating CBA comparison plot...")
    _plot_all_cba_comparison(df_combined, color_map, dev_order, comparison_dir)

    print(f"  Generating total cost change plot...")
    _plot_all_cost_change(df_combined, color_map, dev_order, comparison_dir)

    print(f"  Generating scenario viability comparison plot...")
    _plot_all_scenario_viability(df_combined, color_map, dev_order, comparison_dir)

    print(f"  Generating summary CSV...")
    _generate_comparison_summary_csv(df_combined, dev_order, comparison_dir)

    print(f"\n  {'='*70}")
    print(f"  ✓ All developments comparison plots and CSV saved to: {comparison_dir}")
    print(f"  {'='*70}\n")


def _plot_all_cba_comparison(df_combined, color_map, dev_order, output_dir):
    """Generate CBA comparison plot for all developments (vertical grouped bars)."""

    # Calculate mean CBA for each development and pipeline
    summary = df_combined.groupby(['development', 'pipeline'])['cba_ratio'].mean().reset_index()

    # Create figure - dynamic width based on number of developments
    n_devs = len(dev_order)
    fig_width = max(12, n_devs * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)

    bar_width = 0.35
    x_positions = np.arange(n_devs)

    # Prepare data for plotting
    new_cbas = []
    old_cbas = []
    line_names = []

    for dev_id in dev_order:
        dev_data = summary[summary['development'] == dev_id]
        line_name = df_combined[df_combined['development'] == dev_id]['line_name'].iloc[0]
        line_names.append(line_name)

        new_cba = dev_data[dev_data['pipeline'] == 'new']['cba_ratio'].values
        old_cba = dev_data[dev_data['pipeline'] == 'old']['cba_ratio'].values

        new_cbas.append(new_cba[0] if len(new_cba) > 0 else 0)
        old_cbas.append(old_cba[0] if len(old_cba) > 0 else 0)

    # Plot bars
    for i, dev_id in enumerate(dev_order):
        color = color_map[dev_id]

        # New pipeline bar with hatching
        ax.bar(x_positions[i] - bar_width/2, new_cbas[i], bar_width,
               color=color, alpha=1.0, edgecolor='black', linewidth=0.5, hatch='////')

        # Old pipeline bar with lighter hatching
        ax.bar(x_positions[i] + bar_width/2, old_cbas[i], bar_width,
               color=color, alpha=0.4, edgecolor='black', linewidth=0.5, hatch='...')

    # Formatting
    ax.set_xlabel('Development', fontsize=12)
    ax.set_ylabel('Mean Cost-Benefit Ratio', fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(line_names, rotation=90, fontsize=8)
    ax.axhline(y=1, color='red', linestyle='-', alpha=0.5, linewidth=1.5)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor='gray', hatch='////', alpha=1.0, edgecolor='black', label='New Pipeline'),
        mpatches.Patch(facecolor='gray', hatch='...', alpha=0.4, edgecolor='black', label='Old Pipeline'),
        mlines.Line2D([0], [0], color='red', linestyle='-', alpha=0.5, label='Break-even (CBA = 1)')
    ]
    ax.legend(handles=legend_handles, loc='best', frameon=True, fontsize=10, fancybox=True, shadow=True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "all_developments_cba_comparison.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {output_path}")


def _plot_all_cost_change(df_combined, color_map, dev_order, output_dir):
    """Generate total cost change plot for all developments (vertical diverging bars)."""

    # Calculate mean total costs for each development and pipeline
    summary = df_combined.groupby(['development', 'pipeline'])['total_costs'].mean().reset_index()

    # Create figure - dynamic width based on number of developments
    n_devs = len(dev_order)
    fig_width = max(12, n_devs * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)

    # Prepare data for plotting
    cost_changes_abs = []  # Absolute change in CHF millions
    cost_changes_pct = []  # Percentage change
    line_names = []

    for dev_id in dev_order:
        dev_data = summary[summary['development'] == dev_id]
        line_name = df_combined[df_combined['development'] == dev_id]['line_name'].iloc[0]
        line_names.append(line_name)

        new_cost = dev_data[dev_data['pipeline'] == 'new']['total_costs'].values
        old_cost = dev_data[dev_data['pipeline'] == 'old']['total_costs'].values

        if len(new_cost) > 0 and len(old_cost) > 0:
            change_abs = (new_cost[0] - old_cost[0]) / 1e6  # Convert to millions
            change_pct = ((new_cost[0] - old_cost[0]) / old_cost[0]) * 100 if old_cost[0] != 0 else 0
        else:
            change_abs = 0
            change_pct = 0

        cost_changes_abs.append(change_abs)
        cost_changes_pct.append(change_pct)

    # Plot bars with colors and hatching for colorblind accessibility
    x_positions = np.arange(n_devs)
    bar_width = 0.6

    for i, (change_abs, change_pct) in enumerate(zip(cost_changes_abs, cost_changes_pct)):
        if change_abs >= 0:
            # Cost increase - red with diagonal hatching
            ax.bar(x_positions[i], change_abs, bar_width,
                   color='#d62728', alpha=0.7, edgecolor='black', linewidth=0.5,
                   hatch='\\\\\\')
        else:
            # Cost decrease - green with opposite diagonal hatching
            ax.bar(x_positions[i], change_abs, bar_width,
                   color='#2ca02c', alpha=0.7, edgecolor='black', linewidth=0.5,
                   hatch='///')

    # Formatting
    ax.set_xlabel('Development', fontsize=12)
    ax.set_ylabel('Total Cost Change (New - Old) in CHF million', fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(line_names, rotation=90, fontsize=8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add percentage change as secondary y-axis
    ax2 = ax.twinx()
    ax2.set_ylabel('Percentage Change (%)', fontsize=12)
    max_abs_cost = max(abs(x) for x in cost_changes_abs) if cost_changes_abs else 1
    max_abs_pct = max(abs(x) for x in cost_changes_pct) if cost_changes_pct else 1
    if max_abs_cost > 0:
        scale_factor = max_abs_pct / max_abs_cost
        ax2.set_ylim(ax.get_ylim()[0] * scale_factor, ax.get_ylim()[1] * scale_factor)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor='#d62728', hatch='\\\\\\', edgecolor='black', alpha=0.7, label='Cost Increase'),
        mpatches.Patch(facecolor='#2ca02c', hatch='///', edgecolor='black', alpha=0.7, label='Cost Decrease'),
        mlines.Line2D([0], [0], color='black', linestyle='-', label='No Change')
    ]
    ax.legend(handles=legend_handles, loc='best', frameon=True, fontsize=10, fancybox=True, shadow=True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "all_developments_cost_change_comparison.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {output_path}")


def _plot_all_scenario_viability(df_combined, color_map, dev_order, output_dir):
    """Generate scenario viability comparison plot for all developments (vertical grouped bars with percentages).

    Only includes developments with at least one viable scenario in either pipeline.
    """

    # Calculate viability (net benefit > 0) for each development, pipeline, and scenario
    df_combined['is_viable'] = df_combined['total_net_benefit'] > 0

    # Count viable scenarios per development and pipeline
    viability_summary = df_combined.groupby(['development', 'pipeline']).agg({
        'is_viable': 'sum',  # Count of viable scenarios
        'scenario': 'count'   # Total scenarios
    }).reset_index()

    viability_summary.columns = ['development', 'pipeline', 'viable_count', 'total_count']
    viability_summary['viable_pct'] = (viability_summary['viable_count'] / viability_summary['total_count']) * 100

    # Filter to only include developments with at least one viable scenario in either pipeline
    # Get max viable count per development across both pipelines
    max_viable_per_dev = viability_summary.groupby('development')['viable_count'].max()
    devs_with_viable = max_viable_per_dev[max_viable_per_dev > 0].index.tolist()

    # Filter dev_order to only include developments with viable scenarios
    filtered_dev_order = [dev for dev in dev_order if dev in devs_with_viable]

    if len(filtered_dev_order) == 0:
        print(f"    ⚠ No developments with viable scenarios found - skipping viability plot")
        return

    # Create figure - dynamic width based on number of developments
    n_devs = len(filtered_dev_order)
    fig_width = max(12, n_devs * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)

    bar_width = 0.35
    x_positions = np.arange(n_devs)

    # Prepare data for plotting (only for filtered developments)
    new_viability_pct = []
    old_viability_pct = []
    line_names = []

    for dev_id in filtered_dev_order:
        dev_data = viability_summary[viability_summary['development'] == dev_id]
        line_name = df_combined[df_combined['development'] == dev_id]['line_name'].iloc[0]
        line_names.append(line_name)

        new_pct = dev_data[dev_data['pipeline'] == 'new']['viable_pct'].values
        old_pct = dev_data[dev_data['pipeline'] == 'old']['viable_pct'].values

        new_viability_pct.append(new_pct[0] if len(new_pct) > 0 else 0)
        old_viability_pct.append(old_pct[0] if len(old_pct) > 0 else 0)

    # Plot bars
    for i, dev_id in enumerate(filtered_dev_order):
        color = color_map[dev_id]

        # New pipeline bar with hatching
        ax.bar(x_positions[i] - bar_width/2, new_viability_pct[i], bar_width,
               color=color, alpha=1.0, edgecolor='black', linewidth=0.5, hatch='////')

        # Old pipeline bar with lighter hatching
        ax.bar(x_positions[i] + bar_width/2, old_viability_pct[i], bar_width,
               color=color, alpha=0.4, edgecolor='black', linewidth=0.5, hatch='...')

    # Formatting
    ax.set_xlabel('Development', fontsize=12)
    ax.set_ylabel('Viable Scenarios (%)', fontsize=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(line_names, rotation=90, fontsize=8)
    ax.set_ylim(0, 105)  # 0-100% with slight margin
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor='gray', hatch='////', alpha=1.0, edgecolor='black', label='New Pipeline'),
        mpatches.Patch(facecolor='gray', hatch='...', alpha=0.4, edgecolor='black', label='Old Pipeline')
    ]
    ax.legend(handles=legend_handles, loc='best', frameon=True, fontsize=10, fancybox=True, shadow=True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "all_developments_scenario_viability_comparison.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {output_path}")
    print(f"    • Developments shown: {n_devs} (filtered from {len(dev_order)} total - only showing developments with ≥1 viable scenario)")


def _generate_comparison_summary_csv(df_combined, dev_order, output_dir):
    """Generate CSV summary of all comparison metrics."""

    # Calculate all metrics
    summary_data = []

    # Mark viable scenarios
    df_combined['is_viable'] = df_combined['total_net_benefit'] > 0

    for dev_id in dev_order:
        dev_data = df_combined[df_combined['development'] == dev_id]
        line_name = dev_data['line_name'].iloc[0]

        # Separate new and old pipeline data
        new_data = dev_data[dev_data['pipeline'] == 'new']
        old_data = dev_data[dev_data['pipeline'] == 'old']

        # CBA metrics
        mean_cba_new = new_data['cba_ratio'].mean() if len(new_data) > 0 else 0
        mean_cba_old = old_data['cba_ratio'].mean() if len(old_data) > 0 else 0
        cba_change = mean_cba_new - mean_cba_old

        # Cost metrics
        total_cost_new = new_data['total_costs'].mean() if len(new_data) > 0 else 0
        total_cost_old = old_data['total_costs'].mean() if len(old_data) > 0 else 0
        total_cost_change_chf = total_cost_new - total_cost_old
        total_cost_change_pct = (total_cost_change_chf / total_cost_old * 100) if total_cost_old != 0 else 0

        # Viability metrics
        viable_scenarios_new = new_data['is_viable'].sum() if len(new_data) > 0 else 0
        viable_scenarios_old = old_data['is_viable'].sum() if len(old_data) > 0 else 0
        total_scenarios = new_data['scenario'].nunique() if len(new_data) > 0 else 0
        viable_scenarios_change = viable_scenarios_new - viable_scenarios_old
        viable_scenarios_new_pct = (viable_scenarios_new / total_scenarios * 100) if total_scenarios > 0 else 0
        viable_scenarios_old_pct = (viable_scenarios_old / total_scenarios * 100) if total_scenarios > 0 else 0

        # Net benefit (for sorting reference)
        mean_net_benefit_new = new_data['total_net_benefit'].mean() if len(new_data) > 0 else 0

        summary_data.append({
            'development': dev_id,
            'line_name': line_name,
            'mean_cba_new': mean_cba_new,
            'mean_cba_old': mean_cba_old,
            'cba_change': cba_change,
            'total_cost_new_chf': total_cost_new,
            'total_cost_old_chf': total_cost_old,
            'total_cost_change_chf': total_cost_change_chf,
            'total_cost_change_pct': total_cost_change_pct,
            'viable_scenarios_new': int(viable_scenarios_new),
            'viable_scenarios_old': int(viable_scenarios_old),
            'viable_scenarios_change': int(viable_scenarios_change),
            'viable_scenarios_new_pct': viable_scenarios_new_pct,
            'viable_scenarios_old_pct': viable_scenarios_old_pct,
            'total_scenarios': int(total_scenarios),
            'mean_net_benefit_new': mean_net_benefit_new
        })

    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    output_path = os.path.join(output_dir, "all_developments_pipeline_comparison_summary.csv")
    summary_df.to_csv(output_path, index=False)

    print(f"    ✓ Saved: {output_path}")
    print(f"    • Total developments: {len(summary_df)}")
    print(f"    • Developments with improved CBA: {(summary_df['cba_change'] > 0).sum()}")
    print(f"    • Developments with reduced costs: {(summary_df['total_cost_change_chf'] < 0).sum()}")
    print(f"    • Developments with more viable scenarios: {(summary_df['viable_scenarios_change'] > 0).sum()}")
