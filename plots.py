import numpy as np
import rasterio
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib_scalebar.scalebar import ScaleBar
import rasterio.plot
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import math
import matplotlib.lines as mlines


class CustomBasemap:
    def __init__(self, boundary=None, network=None, access_points=None, frame=None, canton=False):
        # Create a figure and axis
        self.fig, self.ax = plt.subplots(figsize=(15, 10))

        # Plot cantonal border
        if canton==True:
            canton = gpd.read_file(r"data\Scenario\Boundaries\Gemeindegrenzen\UP_KANTON_F.shp")
            canton[canton["KANTON"] == 'Zürich'].boundary.plot(ax=self.ax, color="black", lw=2)

        # Plot lakes
        lakes = gpd.read_file(r"data\landuse_landcover\landcover\lake\WB_STEHGEWAESSER_F.shp")
        lakes.plot(ax=self.ax, color="lightblue")

        # Add scale bar
        self.ax.add_artist(ScaleBar(1, location="lower right"))

        if isinstance(network, gpd.GeoDataFrame):
            network.plot(ax=self.ax, color="black", lw=2)

        if isinstance(access_points, gpd.GeoDataFrame):
            access_points.plot(ax=self.ax, color="black", markersize=50)

        location = gpd.read_file(r'data\manually_gathered_data\cities.shp', crs="epsg:2056")
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
        plt.savefig(r"plot\Voronoi\developments\dev_" + str(id) + ".png", dpi=400)


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
    lakes = gpd.read_file(r"data\landuse_landcover\landcover\lake\WB_STEHGEWAESSER_F.shp")
    lakes.plot(ax=ax, color="lightblue", zorder=9)

    # Add scale bar
    ax.add_artist(ScaleBar(1, location="lower right"))

    if isinstance(network, gpd.GeoDataFrame):
        network.plot(ax=ax, color="black", lw=2, zorder=11)

    if isinstance(access_points, gpd.GeoDataFrame):
        access_points.plot(ax=ax, color="black", markersize=50, zorder=12)

    location = gpd.read_file(r'data\manually_gathered_data\cities.shp', crs="epsg:2056")
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
    cbar.set_label(f'Net benefits in {title_bar} [Mio. CHF]\n(Construction, maintenance, highway travel time, access time and external effects)', rotation=90,
                    labelpad=30, fontsize=16)

    # Set the font size of the colorbar's tick labels
    cbar.ax.tick_params(labelsize=14)

    raster = rasterio.open(banned_area)
    cmap_raster = ListedColormap(["white", "white"])
    rasterio.plot.show(raster, ax=ax, cmap=cmap_raster, zorder=3)

    # Create custom legend elements
    water_body_patch = mpatches.Patch(facecolor="lightblue", label='Water bodies', edgecolor='black', linewidth=1)
    protected_area_patch = mpatches.Patch(facecolor='white', label='Protected area or no interpolated value available',
                                          edgecolor='black', linewidth=1)

    # Create the legend below the plot
    legend = ax.legend(handles=[water_body_patch, protected_area_patch], loc='upper center',
                       bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=16, frameon=False)

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
        plt.savefig(fr"plot\results\04_{plot_name}.png", dpi=300)

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
    lakes = gpd.read_file(r"data\landuse_landcover\landcover\lake\WB_STEHGEWAESSER_F.shp")
    lakes.plot(ax=ax, color="lightblue", zorder=9)

    # Add scale bar
    ax.add_artist(ScaleBar(1, location="lower right"))

    if isinstance(network, gpd.GeoDataFrame):
        network.plot(ax=ax, color="black", lw=2, zorder=11)

    if isinstance(access_points, gpd.GeoDataFrame):
        access_points.plot(ax=ax, color="black", markersize=50, zorder=12)

    location = gpd.read_file(r'data\manually_gathered_data\cities.shp', crs="epsg:2056")
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
    water_body_patch = mpatches.Patch(facecolor="lightblue", label='Water bodies', edgecolor='black', linewidth=1)
    protected_area_patch = mpatches.Patch(facecolor='white', label='Protected area or no interpolated value available',
                                          edgecolor='black', linewidth=1)

    # Create the legend below the plot
    legend = ax.legend(handles=[water_body_patch, protected_area_patch], loc='upper center',
                       bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=16, frameon=False)

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
        plt.savefig(fr"plot\results\04_{plot_name}.png", dpi=300)

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
    lakes = gpd.read_file(r"data\landuse_landcover\landcover\lake\WB_STEHGEWAESSER_F.shp")
    lakes.plot(ax=ax, color="lightblue", zorder=9)

    # Add scale bar
    ax.add_artist(ScaleBar(1, location="lower right"))

    if isinstance(network, gpd.GeoDataFrame):
        network.plot(ax=ax, color="black", lw=2, zorder=11)

    if isinstance(access_points, gpd.GeoDataFrame):
        access_points.plot(ax=ax, color="black", markersize=50, zorder=12)

    location = gpd.read_file(r'data\manually_gathered_data\cities.shp', crs="epsg:2056")
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
    water_body_patch = mpatches.Patch(facecolor="lightblue", label='Water bodies', edgecolor='black', linewidth=1)
    protected_area_patch = mpatches.Patch(facecolor='lightgray', label='Protected area',
                                          edgecolor='black', linewidth=1)

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
        plt.savefig(fr"plot\results\04_{plot_name}.png", dpi=300)

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
    plt.savefig(r"plot\results\benefit_distribution.png", dpi=500)

    # Show the plot
    plt.show()


def plot_benefit_distribution_line_multi(df_costs, columns, labels, plot_name, legend_title):
    # Define bin width
    bin_width = 100
    # Automatically calculate bin edges and create a new column 'bin'
    # Calculate the desired bin edges
    min_value = df_costs[columns].min().min().astype(int)
    min_value = math.floor(min_value / bin_width) * bin_width - bin_width * 2
    max_value = df_costs[columns].max().max().astype(int)
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
    plt.savefig(fr"plot\results\04_distribution_line_{plot_name}.png", dpi=500)
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
    plt.savefig(r"plot\results\04_boxplot.png", dpi=500)
    plt.show()


def plot_2x3_subplots(gdf, limits, network, location):
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
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    title_ax = fig.add_axes([0.97, 0.45, 0.05, 0.1])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    # cbar.ax.set_title("Relative population increase", rotation=90)
    # cbar.ax.yaxis.set_label_position('right')
    title_ax.axis('off')  # Hide the frame around the title axis
    title_ax.text(0.5, 0.5, 'Relative population and employment increase compared to 2020', rotation=90,
                  horizontalalignment='center', verticalalignment='center')

    # Show the plot
    plt.savefig(r"plot\Scenario\5_all_scen.png", dpi=450, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_points_gen(points, edges, banned_area, points_2=None, boundary=None, network=None, access_points=None, plot_name=False, all_zones=False):

    # Import other zones
    schutzzonen = gpd.read_file(r"data\landuse_landcover\Schutzzonen\Schutzanordnungen_Natur_und_Landschaft_-SAO-_-OGD\FNS_SCHUTZZONE_F.shp")
    forest = gpd.read_file(r"data\landuse_landcover\Schutzzonen\Waldareal_-OGD\WALD_WALDAREAL_F.shp")
    fff = gpd.read_file(r"data\landuse_landcover\Schutzzonen\Fruchtfolgeflachen_-OGD\FFF_F.shp")

    fig, ax = plt.subplots(figsize=(13,9))
    # Plot lakes
    lakes = gpd.read_file(r"data\landuse_landcover\landcover\lake\WB_STEHGEWAESSER_F.shp")
    lakes.plot(ax=ax, color="lightblue", zorder=9)

    # Add scale bar
    ax.add_artist(ScaleBar(1, location="lower right"))

    if isinstance(network, gpd.GeoDataFrame):
        network.plot(ax=ax, color="black", lw=2, zorder=11)

    if isinstance(access_points, gpd.GeoDataFrame):
        access_points.plot(ax=ax, color="black", markersize=50, zorder=12)

    location = gpd.read_file(r'data\manually_gathered_data\cities.shp', crs="epsg:2056")
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
        plt.savefig(fr"plot\results\04_{plot_name}.png", dpi=500, bbox_inches='tight')

    plt.show()
    return


def plot_voronoi_comp(eucledian, traveltime, boundary=None, network=None, access_points=None, plot_name=False, all_zones=False):
    fig, ax = plt.subplots(figsize=(13, 9))
    # Plot lakes
    lakes = gpd.read_file(r"data\landuse_landcover\landcover\lake\WB_STEHGEWAESSER_F.shp")
    lakes.plot(ax=ax, color="lightblue", zorder=4)

    # Add scale bar
    ax.add_artist(ScaleBar(1, location="lower right"))

    if isinstance(network, gpd.GeoDataFrame):
        network.plot(ax=ax, color="black", lw=2, zorder=11)

    if isinstance(access_points, gpd.GeoDataFrame):
        access_points.plot(ax=ax, color="black", markersize=50, zorder=12)

    location = gpd.read_file(r'data\manually_gathered_data\cities.shp', crs="epsg:2056")
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
        plt.savefig(fr"plot\results\04_{plot_name}.png", dpi=500)

    plt.show()
    return


def plot_voronoi_development(statusquo, development_voronoi, development_point, boundary=None, network=None, access_points=None, plot_name=False, all_zones=False):
    fig, ax = plt.subplots(figsize=(13, 9))
    # Plot lakes
    lakes = gpd.read_file(r"data\landuse_landcover\landcover\lake\WB_STEHGEWAESSER_F.shp")
    lakes.plot(ax=ax, color="lightblue", zorder=4)

    # Add scale bar
    ax.add_artist(ScaleBar(1, location="lower right"))

    if isinstance(network, gpd.GeoDataFrame):
        network.plot(ax=ax, color="black", lw=2, zorder=11)

    if isinstance(access_points, gpd.GeoDataFrame):
        access_points.plot(ax=ax, color="black", markersize=50, zorder=12)

    location = gpd.read_file(r'data\manually_gathered_data\cities.shp', crs="epsg:2056")
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
        plt.savefig(fr"plot\results\04_{plot_name}.png", dpi=500)

    plt.show()
    return
