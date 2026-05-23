import numpy as np
import rasterio
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
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


def _make_diverging_cmap(min_val, max_val):
    """
    Red→grey→blue diverging colormap anchored at zero.

    The proportions of red and blue are scaled to the magnitude of the
    negative and positive ranges respectively, so the grey midpoint always
    falls at NB=0 regardless of how asymmetric the range is.

    max(1, ...) guards against an empty colour array when one side of the
    range is extremely dominant (e.g. all-negative or near-zero positive).
    """
    n = 256
    gray = [0.83, 0.83, 0.83, 1.0]
    if min_val < 0 and max_val > 0:
        total = abs(min_val) + abs(max_val)
        neg_p, pos_p = abs(min_val) / total, abs(max_val) / total
        neg_c = plt.cm.Reds_r(np.linspace(0.15, 0.8,  max(1, int(n * neg_p))))
        pos_c = plt.cm.Blues( np.linspace(0.3,  0.95, max(1, int(n * pos_p))))
        tl = int(n * 0.2)
        all_c = np.vstack((neg_c[:-1],
                           np.linspace(neg_c[-1], gray, tl),
                           np.linspace(gray, pos_c[0], tl),
                           pos_c[1:]))
    elif min_val >= 0:
        pos_c = plt.cm.Blues(np.linspace(0.3, 0.9, n))
        all_c = np.vstack((np.linspace(gray, pos_c[0], int(n * 0.3)), pos_c[1:]))
    else:
        neg_c = plt.cm.Reds_r(np.linspace(0.2, 0.8, n))
        all_c = np.vstack((neg_c[:-1], np.linspace(neg_c[-1], gray, int(n * 0.3))))
    return LinearSegmentedColormap.from_list("div_cmap", all_c)


def _base_map(ax, network, access_points, boundary):
    """Draw lakes, network, access points, city labels, scale bar, north arrow."""
    lakes = gpd.read_file(r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp")
    lakes.plot(ax=ax, color="lightblue", zorder=9)
    ax.add_artist(ScaleBar(1, location="lower right"))

    if isinstance(network, gpd.GeoDataFrame):
        network.plot(ax=ax, color="#888888", lw=1.0, zorder=10, alpha=0.55)
    if isinstance(access_points, gpd.GeoDataFrame):
        access_points.plot(ax=ax, color="black", markersize=40, zorder=12)

    loc = gpd.read_file(r"data/manually_gathered_data/Cities.shp", crs="epsg:2056")
    loc.plot(ax=ax, color="black", markersize=60, zorder=13)
    for _, row in loc.iterrows():
        ax.annotate(row["location"], xy=row.geometry.coords[0],
                    ha="center", va="top", xytext=(0, -6),
                    textcoords="offset points", fontsize=13, zorder=13)

    ax.text(0.96, 0.92, "N", fontsize=26, weight="bold", ha="center",
            va="center", transform=ax.transAxes, zorder=1000)
    ax.add_patch(FancyArrowPatch((0.96, 0.89), (0.96, 0.97), color="black", lw=2,
                                 arrowstyle="->", mutation_scale=25,
                                 transform=ax.transAxes, zorder=1000))
    ax.set_xticks([]); ax.set_yticks([])
    if boundary is not None:
        xmin, ymin, xmax, ymax = boundary.bounds
        ax.set_xlim(xmin - 100, xmax + 100)
        ax.set_ylim(ymin - 100, ymax + 100)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor("black")
        sp.set_linewidth(1); sp.set_zorder(1000)


def _add_duebendorf_inset(ax, network=None, pos=(0.62, 0.03, 0.36, 0.44)):
    """
    Add a zoomed detail inset of the Dübendorf area (NW corner of the
    corridor, approx. E 2 687 000–2 697 500 / N 1 247 000–1 254 000 LV95).
    A zoom-indicator rectangle is drawn on the parent axes.
    Returns the inset axes so the caller can layer the coloured feature.
    pos – (x0, y0, width, height) in axes-fraction coordinates.
    """
    DUB_E_MIN, DUB_E_MAX = 2_687_000, 2_697_500
    DUB_N_MIN, DUB_N_MAX = 1_247_000, 1_254_000

    ax_ins = ax.inset_axes(pos)
    ax_ins.set_xlim(DUB_E_MIN, DUB_E_MAX)
    ax_ins.set_ylim(DUB_N_MIN, DUB_N_MAX)
    ax_ins.set_xticks([])
    ax_ins.set_yticks([])

    lakes_path = r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp"
    if os.path.exists(lakes_path):
        gpd.read_file(lakes_path).plot(ax=ax_ins, color="lightblue", zorder=9)

    if isinstance(network, gpd.GeoDataFrame):
        network.plot(ax=ax_ins, color="#888888", lw=0.8, zorder=10, alpha=0.55)

    for sp in ax_ins.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("black")
        sp.set_linewidth(2)
        sp.set_zorder(1000)

    ax_ins.set_title("Dübendorf (detail)", fontsize=8, pad=3, fontweight="bold")

    try:
        ax.indicate_inset_zoom(ax_ins, edgecolor="black", alpha=0.6, linewidth=1.5)
    except Exception:
        pass  # older matplotlib versions

    return ax_ins


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


def plot_cost_result(df_costs, banned_area, title_bar, boundary=None, network=None,
                     access_points=None, plot_name=False, col="total_medium"):
    # Join NB values to actual LineString edge geometries (development candidates)
    dev_geom = gpd.read_file("data/Network/processed/development_candidates.gpkg")[["ID_new", "geometry"]]
    dev_geom["ID_new"] = dev_geom["ID_new"].astype(int)
    df_costs = df_costs.copy()
    df_costs["ID_new"] = df_costs["ID_new"].astype(int)
    df_plot = dev_geom.merge(df_costs.drop(columns=["geometry"], errors="ignore"),
                             on="ID_new", how="inner")
    df_plot = gpd.GeoDataFrame(df_plot, geometry="geometry", crs="EPSG:2056")
    df_plot = df_plot.dropna(subset=[col])

    min_val, max_val = df_plot[col].min(), df_plot[col].max()
    cmap = _make_diverging_cmap(min_val, max_val)
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

    fig, ax = plt.subplots(figsize=(15, 10))
    _base_map(ax, network, access_points, boundary)

    # Development lines coloured by net-benefit value
    df_plot.plot(ax=ax, column=col, cmap=cmap, norm=norm,
                 linewidth=6, zorder=11, legend=False, capstyle="round")

    # ID label at each line's midpoint
    for _, row in df_plot.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        mid = row.geometry.interpolate(0.5, normalized=True)
        ax.annotate(str(int(row["ID_new"])), xy=(mid.x, mid.y),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="black", zorder=16,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, ec="none"))

    # Colorbar
    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.5)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(f"{title_bar} [Mio. CHF]", rotation=90, labelpad=20, fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    raster = rasterio.open(banned_area)
    rasterio.plot.show(raster, ax=ax, cmap=ListedColormap(["white", "white"]), zorder=3)

    water_patch = mpatches.Patch(facecolor="lightblue", label="Water bodies",
                                 edgecolor="black", linewidth=1)
    ax.legend(handles=[water_patch], loc="upper center",
              bbox_to_anchor=(0.5, -0.02), fontsize=13, frameon=False)

    if plot_name:
        plt.tight_layout()
        plt.savefig(fr"plot/results/04_{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_single_cost_result(df_costs, banned_area, title_bar, boundary=None, network=None,
                            access_points=None, plot_name=False, col="total_medium"):
    # Join component values to actual LineString edge geometries
    dev_geom = gpd.read_file("data/Network/processed/development_candidates.gpkg")[["ID_new", "geometry"]]
    dev_geom["ID_new"] = dev_geom["ID_new"].astype(int)
    df_costs = df_costs.copy()
    df_costs["ID_new"] = df_costs["ID_new"].astype(int)
    df_costs[col] = df_costs[col] / 1e6   # scale to Mio. CHF
    df_plot = dev_geom.merge(df_costs.drop(columns=["geometry"], errors="ignore"),
                             on="ID_new", how="inner")
    df_plot = gpd.GeoDataFrame(df_plot, geometry="geometry", crs="EPSG:2056")
    df_plot = df_plot.dropna(subset=[col])

    min_val, max_val = df_plot[col].min(), df_plot[col].max()
    cmap = _make_diverging_cmap(min_val, max_val)
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

    fig, ax = plt.subplots(figsize=(15, 10))
    _base_map(ax, network, access_points, boundary)

    df_plot.plot(ax=ax, column=col, cmap=cmap, norm=norm,
                 linewidth=6, zorder=11, legend=False, capstyle="round")

    for _, row in df_plot.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        mid = row.geometry.interpolate(0.5, normalized=True)
        ax.annotate(str(int(row["ID_new"])), xy=(mid.x, mid.y),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="black", zorder=16,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, ec="none"))

    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.5)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(f"{title_bar} [Mio. CHF]", rotation=90, labelpad=20, fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    raster = rasterio.open(banned_area)
    rasterio.plot.show(raster, ax=ax, cmap=ListedColormap(["white", "white"]), zorder=3)

    water_patch = mpatches.Patch(facecolor="lightblue", label="Water bodies",
                                 edgecolor="black", linewidth=1)
    ax.legend(handles=[water_patch], loc="upper center",
              bbox_to_anchor=(0.5, -0.02), fontsize=13, frameon=False)

    if plot_name:
        plt.tight_layout()
        plt.savefig(fr"plot/results/04_{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

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
    # Apply a non-linear transformation (log) — clip to avoid log(0) = -inf
    df_costs[f'log_{col}'] = np.log(df_costs[col].clip(lower=1e-6))
    # Normalize the transformed column; guard against all-identical values
    min_val = df_costs[f'log_{col}'].min()
    max_val = df_costs[f'log_{col}'].max()
    log_range = max_val - min_val if max_val > min_val else 1.0
    df_costs['markersize'] = scale_max - (((df_costs[f'log_{col}'] - min_val) / log_range) * (scale_max - scale_min))

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
                ((np.log(np.maximum(original_std_values, 1e-6)) - min_val) / log_range) * (scale_max - scale_min))

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
        plt.savefig(fr"plot/results/04_{plot_name}.png", dpi=300, bbox_inches='tight')

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
    bin_width = 5
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

    # Count occurrences per bin for each column, indexed by left bin edge
    bin_counts = pd.DataFrame(index=bin_edges[:-1])
    for column in columns:
        column_counts = df_costs.groupby(f'bin_{column}', observed=False)[column].count()
        column_counts.index = column_counts.index.map(lambda iv: iv.left)
        bin_counts[f'bin_{column}'] = column_counts

    # Define labels
    # Check if labels len is same as columns len
    if len(labels) != len(columns):
        print("Labels and columns length are not the same")
    else:
        # Create a dict with column names as keys and labels as values
        legend_labels = dict(zip(columns, labels))

    bar_colors = ['#2980b9', '#8e44ad', '#27ae60', '#e67e22', '#c0392b']

    fig, ax = plt.subplots(figsize=(13, 6))
    x_pos      = np.arange(len(bin_counts))
    n_cols     = len(columns)
    bar_w      = 0.8 / n_cols          # width of each individual bar

    # Grouped bars: each series sits next to the others within the same bin
    for i, column in enumerate(columns):
        offset = (i - n_cols / 2 + 0.5) * bar_w
        ax.bar(x_pos + offset, bin_counts[f'bin_{column}'],
               width=bar_w, label=legend_labels[column],
               color=bar_colors[i % len(bar_colors)],
               alpha=0.9, zorder=3)

    ax.legend(bbox_to_anchor=(1.02, 0), loc="lower left", borderaxespad=0.,
              title=legend_title, fontsize=12, title_fontsize=14, frameon=False)

    plt.xlabel('Net benefit [Mio CHF]', fontsize=14)
    plt.ylabel('Occurrence', fontsize=14)

    # x-tick labels: every 2nd bin edge shown
    tick_labels = [str(int(v)) if j % 2 == 0 else ''
                   for j, v in enumerate(bin_edges[:-1])]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=12)

    plt.grid(axis='x', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.tick_params(axis='x', which='major', length=6, width=1, labelsize=12)

    max_occurrence = bin_counts.max().max()
    y_tick_step = 1
    while max_occurrence > 10 * y_tick_step:
        y_tick_step *= 2
    y_ticks = np.arange(0, max_occurrence + y_tick_step, y_tick_step)
    plt.yticks(y_ticks, fontsize=12)

    min_shaded_region = next((i for i, val in enumerate(bin_edges) if val >= 0), None)
    ax.axvspan(-0.5, min_shaded_region - 0.5, color='lightgray', alpha=0.4, zorder=1)

    plt.xlim(-0.5, len(bin_counts.index) - 0.5)
    plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)

    plt.tight_layout()
    plt.savefig(fr"plot/results/04_distribution_line_{plot_name}.png", dpi=500)
    plt.show()


def plot_scenario_grouped_bar(plot_name="scenario_grouped_bar"):
    """
    Grouped horizontal bar chart: for each development (sorted by NB_s2) three
    bars are shown side by side — one per growth scenario (s1 / s2 / s3).
    Makes the scenario spread directly readable per development.

    Reads:  data/costs/net_benefits.csv
            data/Network/processed/development_candidates.gpkg
    Saves:  plot/results/{plot_name}.png
    """
    nb_path   = "data/costs/net_benefits.csv"
    devs_path = "data/Network/processed/development_candidates.gpkg"
    for p in [nb_path, devs_path]:
        if not os.path.exists(p):
            print(f"[plot_scenario_grouped_bar] Missing: {p} — skipping")
            return

    nb   = pd.read_csv(nb_path)
    devs = gpd.read_file(devs_path)[["ID_new", "dev_type"]]
    nb   = nb.merge(devs, on="ID_new", how="left")
    nb   = nb.sort_values("NB_s2", ascending=True).reset_index(drop=True)

    for c in ["NB_s1", "NB_s2", "NB_s3"]:
        nb[c] = nb[c] / 1e6

    n     = len(nb)
    y     = np.arange(n)
    bar_h = 0.25

    scenarios = [
        ("NB_s1", "#74b9ff", "Low growth (s1)"),
        ("NB_s2", "#0984e3", "Medium growth (s2)"),
        ("NB_s3", "#2d3436", "High growth (s3)"),
    ]

    fig, ax = plt.subplots(figsize=(11, max(6, n * 0.55)))

    for i, (col, color, label) in enumerate(scenarios):
        offset = (i - 1) * bar_h   # –1 / 0 / +1
        ax.barh(y + offset, nb[col], height=bar_h,
                color=color, alpha=0.85, label=label, zorder=3)

    ax.axvline(0, color="black", lw=1.0, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(nb["ID_new"].astype(int).astype(str), fontsize=9)
    ax.set_xlabel("Net Benefit [Mio. CHF]", fontsize=11)
    ax.set_title("Net Benefit per Development — all growth scenarios\n(sorted by medium scenario)",
                 fontsize=12, pad=8)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)

    plt.tight_layout()
    os.makedirs("plot/results", exist_ok=True)
    plt.savefig(f"plot/results/{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[plot_scenario_grouped_bar] saved → plot/results/{plot_name}.png")


def plot_cost_benefit_scatter(plot_name="cost_benefit_scatter"):
    """
    Scatter plot: X = total costs |C+M|, Y = total benefits T+R+S (medium
    scenario). The dashed diagonal is the break-even line (NB = 0).
    Points above = positive NB (green), below = negative NB (red).
    Shows immediately whether a development fails due to high costs or low
    benefits.

    Reads:  data/costs/net_benefits.gpkg
    Saves:  plot/results/{plot_name}.png
    """
    nb_path = "data/costs/net_benefits.gpkg"
    if not os.path.exists(nb_path):
        print(f"[plot_cost_benefit_scatter] Missing: {nb_path} — skipping")
        return

    nb = gpd.read_file(nb_path)
    nb["costs_m"]    = (nb["C"].abs() + nb["M"].abs()) / 1e6
    nb["benefits_m"] = (nb["T_s2"] + nb["R_s2"] + nb["S_s2"]) / 1e6
    nb["NB_m"]       = nb["NB_s2"] / 1e6

    fig, ax = plt.subplots(figsize=(9, 7))

    # Break-even diagonal
    max_val = max(nb["costs_m"].max(), nb["benefits_m"].max()) * 1.12
    ax.plot([0, max_val], [0, max_val], color="black", lw=1.2,
            linestyle="--", alpha=0.45, label="Break-even  (NB = 0)", zorder=1)

    colors = nb["NB_m"].apply(lambda v: "#27ae60" if v >= 0 else "#e74c3c")
    ax.scatter(nb["costs_m"], nb["benefits_m"],
               c=colors, s=90, alpha=0.85, zorder=3, edgecolors="white", linewidths=0.5)

    for _, row in nb.iterrows():
        ax.annotate(str(int(row["ID_new"])),
                    (row["costs_m"], row["benefits_m"]),
                    fontsize=8, ha="left", va="bottom",
                    xytext=(4, 3), textcoords="offset points")

    # Shaded region: above diagonal = positive NB
    ax.fill_between([0, max_val], [0, max_val], max_val,
                    color="#27ae60", alpha=0.04, zorder=0)
    ax.fill_between([0, max_val], 0, [0, max_val],
                    color="#e74c3c", alpha=0.04, zorder=0)

    legend_handles = [
        mpatches.Patch(color="#27ae60", label="Positive NB  (benefits > costs)"),
        mpatches.Patch(color="#e74c3c", label="Negative NB  (costs > benefits)"),
        plt.Line2D([0], [0], color="black", lw=1.2, linestyle="--",
                   alpha=0.6, label="Break-even  (NB = 0)"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper left")

    ax.set_xlabel("Total Costs  |C + M|  [Mio. CHF]", fontsize=11)
    ax.set_ylabel("Total Benefits  T + R + S  [Mio. CHF]", fontsize=11)
    ax.set_title("Cost-Benefit Balance per Development\n(medium growth scenario)",
                 fontsize=12, pad=8)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(linestyle="--", alpha=0.3, zorder=0)

    plt.tight_layout()
    os.makedirs("plot/results", exist_ok=True)
    plt.savefig(f"plot/results/{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[plot_cost_benefit_scatter] saved → plot/results/{plot_name}.png")


def plot_best_worse(df):

    # Sort the DataFrame by "total_medium" in ascending and descending order
    df_top5 = df.nlargest(5, 'total_medium')
    df_bottom5 = df.nsmallest(5, 'total_medium')

    # Specify the columns to plot
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
    df["mean"] = df[['NB_s1', 'NB_s2', 'NB_s3']].mean(axis=1)
    df = df.sort_values(by=['mean'], ascending=False)
    df_top = df.head(nbr)
    df_top = df_top[['ID_new', 'NB_s1', 'NB_s2', 'NB_s3']]
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
    plt.savefig(r"plot/Scenario/5_all_scen.png", dpi=450, bbox_inches='tight', pad_inches=0.1)
    plt.show()


def plot_points_gen(points, edges, banned_area, points_2=None, boundary=None, network=None, access_points=None, plot_name=False, all_zones=False):

    # Import other zones
    schutzzonen = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Schutzanordnungen_Natur_und_Landschaft_-SAO-_-OGD/FNS_SCHUTZZONE_F.shp")
    forest = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Waldareal_-OGD/WALD_WALDAREAL_F.shp")
    fff = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Fruchtfolgeflachen_-OGD/FFF_F.shp")

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


def plot_voronoi_comp(eucledian, traveltime, boundary=None, network=None, access_points=None, plot_name=False, all_zones=False):
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


def plot_voronoi_development(statusquo, development_voronoi, development_point, boundary=None, network=None, access_points=None, plot_name=False, all_zones=False):
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

def plot_example_od_path(boundary=None, network=None, access_points=None,
                         plot_name="example_od_path"):
    """
    Plot one example fastest path from data/costs/od_fastest_paths.csv.

    Picks the status-quo row with the highest travel time whose average speed
    implies it uses real network edges (≥ 8 km/h — excludes pure 5 km/h gap
    penalty paths).  Rebuilds the same graph as od_fastest_paths (real edges +
    connectivity bridges + gap links) so Dijkstra finds the identical route.

    Segments are coloured by type:
      solid red   — real surveyed edge
      dashed orange — connectivity bridge or gap penalty edge
    """
    import networkx as nx
    from shapely.geometry import LineString, MultiLineString
    from scipy.spatial import cKDTree

    paths_csv = "data/costs/od_fastest_paths.csv"
    if not os.path.exists(paths_csv):
        print("[plot_example_od_path] od_fastest_paths.csv not found — skipping")
        return

    df = pd.read_csv(paths_csv)
    sq = df[df["scenario"] == "status_quo"].copy()
    if sq.empty:
        print("[plot_example_od_path] No status-quo rows — skipping")
        return

    # Filter out gap-dominated paths (avg speed < 8 km/h → pure penalty path)
    sq["speed_kmh"] = (sq["length_m"] / 1000) / (sq["tt_min"] / 60)
    real_paths = sq[sq["speed_kmh"] >= 8.0]
    pool = real_paths if not real_paths.empty else sq
    row    = pool.loc[pool["tt_min"].idxmax()]
    origin = int(row["origin"])
    dest   = int(row["dest"])
    print(f"[plot_example_od_path] node {origin} → {dest}  "
          f"({row['tt_min']:.1f} min, {row['length_m']/1000:.2f} km, "
          f"{row['speed_kmh']:.1f} km/h avg)")

    # ── Rebuild graph (real edges + bridges + gap links) ──────────────────────
    points = gpd.read_file("data/Network/processed/points_with_attribute.gpkg")
    edges  = gpd.read_file("data/Network/processed/edges_with_attribute.gpkg")
    edges  = edges.set_crs("epsg:2056", allow_override=True)
    edges["length_m"] = edges.geometry.length

    G = nx.Graph()
    for _, pt in points.iterrows():
        G.add_node(int(pt["ID_point"]), x=pt.geometry.x, y=pt.geometry.y)
    edges_clean = edges.dropna(subset=["start", "end"])
    for _, e in edges_clean.iterrows():
        u, v = int(e["start"]), int(e["end"])
        tt = float(e["tt_min"])
        if G.has_edge(u, v):
            if tt < G[u][v].get("tt", 1e9):
                G[u][v].update({"tt": tt, "length_m": float(e["length_m"])})
        else:
            G.add_edge(u, v, tt=tt, length_m=float(e["length_m"]), is_real=True)

    # Connectivity bridges
    conn_path = "data/Network/processed/connectivity_developments.gpkg"
    if os.path.exists(conn_path):
        node_list = list(G.nodes())
        node_arr  = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in node_list])
        kd = cKDTree(node_arr)
        for _, br in gpd.read_file(conn_path).iterrows():
            coords = list(br.geometry.coords)
            _, i_u = kd.query([coords[0][0],  coords[0][1]])
            _, i_v = kd.query([coords[-1][0], coords[-1][1]])
            u2, v2 = node_list[i_u], node_list[i_v]
            if u2 != v2 and not G.has_edge(u2, v2):
                lm = br.geometry.length
                G.add_edge(u2, v2, tt=(lm/1000)/5.0*60, length_m=lm,
                           is_bridge=True, is_real=False)

    # Gap links between disconnected OD nodes (mirrors od_fastest_paths logic)
    all_od = set(df["origin"].astype(int)) | set(df["dest"].astype(int))
    all_od.discard(9999)
    od_valid = [n for n in all_od if n in G]
    comp_map = {n: i for i, c in enumerate(nx.connected_components(G)) for n in c}
    seen_gap = set()
    for o in od_valid:
        for d in od_valid:
            if o == d:
                continue
            pair = (min(o, d), max(o, d))
            if pair in seen_gap or comp_map.get(o) == comp_map.get(d):
                continue
            seen_gap.add(pair)
            ox2 = G.nodes[o].get("x", 0); oy2 = G.nodes[o].get("y", 0)
            dx2 = G.nodes[d].get("x", 0); dy2 = G.nodes[d].get("y", 0)
            gap_dist = ((ox2-dx2)**2 + (oy2-dy2)**2)**0.5 * 1.4
            G.add_edge(o, d, tt=gap_dist/5000*60*3.6,
                       length_m=gap_dist, is_gap=True, is_real=False)

    if origin not in G or dest not in G:
        print("[plot_example_od_path] Origin or destination not in graph — skipping")
        return
    try:
        _, node_path = nx.single_source_dijkstra(G, origin, dest, weight="tt")
    except nx.NetworkXNoPath:
        print("[plot_example_od_path] No path found — skipping")
        return

    # ── Reconstruct geometry, split real vs penalty segments ─────────────────
    edge_lookup = {}
    for _, e in edges_clean.iterrows():
        u, v = int(e["start"]), int(e["end"])
        edge_lookup[(u, v)] = e.geometry
        edge_lookup[(v, u)] = e.geometry

    real_segs, gap_segs = [], []
    for u, v in zip(node_path[:-1], node_path[1:]):
        attr = G[u][v]
        geom = edge_lookup.get((u, v))
        if geom is None:
            geom = LineString([(G.nodes[u]["x"], G.nodes[u]["y"]),
                               (G.nodes[v]["x"], G.nodes[v]["y"])])
        if attr.get("is_real", False):
            real_segs.append(geom)
        else:
            gap_segs.append(geom)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 10))
    lakes = gpd.read_file(r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp")
    lakes.plot(ax=ax, color="lightblue", zorder=1)
    ax.add_artist(ScaleBar(1, location="lower right"))

    if isinstance(network, gpd.GeoDataFrame):
        network.plot(ax=ax, color="#aaaaaa", lw=1.2, zorder=2, label="Network")
    if isinstance(access_points, gpd.GeoDataFrame):
        access_points.plot(ax=ax, color="#555555", markersize=20, zorder=3)

    if real_segs:
        gpd.GeoDataFrame(geometry=real_segs, crs="epsg:2056").plot(
            ax=ax, color="#e63946", lw=3, zorder=10, label="Real network path"
        )
    if gap_segs:
        gpd.GeoDataFrame(geometry=gap_segs, crs="epsg:2056").plot(
            ax=ax, color="#f4a261", lw=2.5, zorder=10,
            linestyle="dashed", label="Gap / bridge hop"
        )

    ox, oy = G.nodes[origin]["x"], G.nodes[origin]["y"]
    dx, dy = G.nodes[dest]["x"],   G.nodes[dest]["y"]
    ax.scatter([ox], [oy], s=160, color="#2a9d8f", zorder=12,
               marker="o", label=f"Origin ({origin})")
    ax.scatter([dx], [dy], s=160, color="#e76f51", zorder=12,
               marker="s", label=f"Destination ({dest})")
    ax.annotate(f"O: {origin}", xy=(ox, oy), xytext=(6, 6),
                textcoords="offset points", fontsize=9, fontweight="bold",
                color="#2a9d8f", zorder=13)
    ax.annotate(f"D: {dest}", xy=(dx, dy), xytext=(6, 6),
                textcoords="offset points", fontsize=9, fontweight="bold",
                color="#e76f51", zorder=13)

    location = gpd.read_file(r"data/manually_gathered_data/Cities.shp", crs="epsg:2056")
    location.plot(ax=ax, color="black", markersize=60, zorder=11)
    for _, loc_row in location.iterrows():
        ax.annotate(loc_row["location"], xy=loc_row["geometry"].coords[0],
                    ha="center", va="top", xytext=(0, -6),
                    textcoords="offset points", fontsize=13, zorder=11)

    if boundary:
        xmin, ymin, xmax, ymax = boundary.bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Example fastest path  ·  node {origin} → {dest}"
        f"  ·  {row['tt_min']:.1f} min  ·  {row['length_m']/1000:.2f} km",
        fontsize=13, pad=10,
    )
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    os.makedirs("plot/results", exist_ok=True)
    plt.savefig(f"plot/results/{plot_name}.png", dpi=400)
    plt.show()
    return


# ─────────────────────────────────────────────────────────────────────────────
# NEW RESULT VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_priority_ranking(plot_name="priority_ranking"):
    """
    Horizontal bar chart: one bar per development, sorted by NB_s2 (medium
    scenario), with error bars spanning NB_s1 → NB_s3 as scenario uncertainty.
    Bars are coloured by dev_type (red = Netzlücke, orange = Schwachstelle).
    A vertical line at 0 separates net-positive from net-negative candidates.

    Reads:  data/costs/net_benefits.csv
            data/Network/processed/developments_list.csv
    Saves:  plot/results/{plot_name}.png
    """
    nb_path   = "data/costs/net_benefits.csv"
    devs_path = "data/Network/processed/development_candidates.gpkg"
    for p in [nb_path, devs_path]:
        if not os.path.exists(p):
            print(f"[plot_priority_ranking] Missing: {p} — skipping")
            return

    nb   = pd.read_csv(nb_path)
    devs = gpd.read_file(devs_path)[["ID_new", "dev_type"]]
    nb   = nb.merge(devs, on="ID_new", how="left")
    nb   = nb.sort_values("NB_s2").reset_index(drop=True)

    # Convert to Mio. CHF
    for c in ["NB_s1", "NB_s2", "NB_s3"]:
        nb[c] = nb[c] / 1e6

    err_lo = (nb["NB_s2"] - nb["NB_s1"]).clip(lower=0)
    err_hi = (nb["NB_s3"] - nb["NB_s2"]).clip(lower=0)

    colors = nb["dev_type"].map(
        {"netzluecke": "#c0392b", "schwachstelle": "#e67e22", "connectivity": "#7f8c8d"}
    ).fillna("#95a5a6")

    n = len(nb)
    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.38)))
    y = np.arange(n)

    ax.barh(y, nb["NB_s2"], xerr=[err_lo, err_hi],
            color=colors, capsize=3, alpha=0.85, error_kw={"lw": 1.2, "ecolor": "#333333"})
    ax.axvline(0, color="black", lw=1.0, zorder=5)

    labels = nb["ID_new"].astype(int).astype(str)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Net Benefit — medium scenario [Mio. CHF]", fontsize=11)
    ax.set_title("Development Ranking by Net Benefit\n(NB_s2 ± scenario range s1–s3)",
                 fontsize=12, pad=8)
    ax.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)

    present_types = nb["dev_type"].dropna().unique().tolist()
    type_meta = {
        "netzluecke":    ("#c0392b", "Netzlücke"),
        "schwachstelle": ("#e67e22", "Schwachstelle"),
        "connectivity":  ("#7f8c8d", "Connectivity bridge"),
    }
    legend_handles = [
        mpatches.Patch(color=col, label=lbl)
        for key, (col, lbl) in type_meta.items()
        if key in present_types
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    plt.tight_layout()
    os.makedirs("plot/results", exist_ok=True)
    plt.savefig(f"plot/results/{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[plot_priority_ranking] saved → plot/results/{plot_name}.png")


def plot_nb_components_waterfall(plot_name="nb_components_waterfall"):
    """
    Stacked horizontal bar chart: for each development (sorted by NB_s2)
    the negative side shows Construction (C) + Maintenance (M) and the
    positive side shows Travel-time savings (T_s2) + Comfort (R) + Safety (S_s2).
    Makes it immediately clear what component drives each result.

    Reads:  data/costs/net_benefits.gpkg
            data/Network/processed/developments_list.csv
    Saves:  plot/results/{plot_name}.png
    """
    nb_path   = "data/costs/net_benefits.gpkg"
    devs_path = "data/Network/processed/developments_list.csv"
    for p in [nb_path, devs_path]:
        if not os.path.exists(p):
            print(f"[plot_nb_components_waterfall] Missing: {p} — skipping")
            return

    nb   = gpd.read_file(nb_path)
    devs = pd.read_csv(devs_path)
    nb   = nb.merge(devs[["ID_new", "dev_type"]], on="ID_new", how="left")
    nb   = nb.sort_values("NB_s2").reset_index(drop=True)

    M   = 1e6
    ids = nb["ID_new"].astype(int).astype(str)
    n   = len(nb)
    y   = np.arange(n)

    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.38)))

    # ── Costs (extend left, stacked) ─────────────────────────────────────────
    ax.barh(y, nb["C"] / M,  color="#c0392b", alpha=0.85, label="Construction (C)")
    ax.barh(y, nb["M"] / M,  left=nb["C"] / M,
            color="#e74c3c", alpha=0.85, label="Maintenance (M)")

    # ── Benefits (extend right, stacked) ─────────────────────────────────────
    ax.barh(y, nb["T_s2"] / M, color="#27ae60", alpha=0.85, label="Travel-time savings (T)")
    ax.barh(y, nb["R_s2"] / M,
            left=nb["T_s2"] / M,
            color="#2ecc71", alpha=0.85, label="Route comfort (R)")
    ax.barh(y, nb["S_s2"] / M,
            left=(nb["T_s2"] + nb["R_s2"]) / M,
            color="#3498db", alpha=0.85, label="Safety (S)")

    ax.axvline(0, color="black", lw=1.0, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(ids, fontsize=9)
    ax.set_xlabel("CHF [Mio.]", fontsize=11)
    ax.set_title("NB Components per Development (medium scenario)",
                 fontsize=12, pad=8)
    ax.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(loc="lower right", fontsize=9, frameon=True)

    plt.tight_layout()
    os.makedirs("plot/results", exist_ok=True)
    plt.savefig(f"plot/results/{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[plot_nb_components_waterfall] saved → plot/results/{plot_name}.png")


def plot_tt_improvement_map(boundary=None, network=None, top_n=3,
                             plot_name="tt_improvement_map"):
    """
    Spatial raster-diff map: for the top_n developments by NB_s2, shows the
    per-pixel travel-time improvement (status_quo − dev) in minutes.
    Green = shorter travel time with the new link.  The corridor network is
    overlaid as a grey backdrop.

    Reads:  data/costs/net_benefits.csv
            data/Network/travel_time/travel_time_raster.tif
            data/Network/travel_time/developments/dev{id}_travel_time_raster.tif
            data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp
    Saves:  plot/results/{plot_name}.png
    """
    import rasterio as _rio

    nb_path = "data/costs/net_benefits.csv"
    sq_path = "data/Network/travel_time/travel_time_raster.tif"
    for p in [nb_path, sq_path]:
        if not os.path.exists(p):
            print(f"[plot_tt_improvement_map] Missing: {p} — skipping")
            return

    nb     = pd.read_csv(nb_path).sort_values("NB_s2", ascending=False)
    top_ids = nb["ID_new"].astype(int).head(top_n).tolist()

    # Filter to developments that actually have a raster file
    available = [
        i for i in top_ids
        if os.path.exists(
            f"data/Network/travel_time/developments/dev{i}_travel_time_raster.tif"
        )
    ]
    if not available:
        print("[plot_tt_improvement_map] No development rasters found — skipping")
        return

    with _rio.open(sq_path) as src:
        sq_arr = src.read(1).astype(float)  # seconds
        ext    = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]

    lakes_path = r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp"

    n_plots = len(available)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 8),
                              squeeze=False)

    for ax, dev_id in zip(axes[0], available):
        dev_path = (
            f"data/Network/travel_time/developments/dev{dev_id}_travel_time_raster.tif"
        )
        with _rio.open(dev_path) as src2:
            dev_arr = src2.read(1).astype(float)

        # Improvement in minutes (positive = better with new link)
        diff_min = np.where(
            np.isnan(sq_arr) | np.isnan(dev_arr),
            np.nan,
            (sq_arr - dev_arr) / 60.0
        )

        vmax = max(0.5, float(np.nanpercentile(np.abs(diff_min), 95)))
        im   = ax.imshow(diff_min, cmap="RdYlGn",
                         vmin=-vmax, vmax=0,
                         extent=ext, origin="upper", alpha=0.85, zorder=2)

        if os.path.exists(lakes_path):
            gpd.read_file(lakes_path).plot(
                ax=ax, color="lightblue", zorder=3, alpha=0.7)

        if isinstance(network, gpd.GeoDataFrame):
            network.plot(ax=ax, color="#555555", lw=0.6, alpha=0.5, zorder=4)

        nb_val = nb.loc[nb["ID_new"] == dev_id, "NB_s2"].values
        nb_str = f"NB = {nb_val[0]/1e6:+.1f} Mio. CHF" if len(nb_val) else ""
        ax.set_title(f"Dev {dev_id}\n{nb_str}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        cbar = plt.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
        cbar.set_label("ΔTT [min]", fontsize=9)

        if boundary:
            xmin, ymin, xmax, ymax = boundary.bounds
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

    fig.suptitle(
        f"Travel-Time Improvement vs. Status Quo — Top {len(available)} Developments",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    os.makedirs("plot/results", exist_ok=True)
    plt.savefig(f"plot/results/{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[plot_tt_improvement_map] saved → plot/results/{plot_name}.png")


def plot_nb_on_network(boundary=None, network=None, access_points=None,
                       plot_name="nb_network_map"):
    """
    Map with development candidate **edge geometries** coloured by NB_s2
    (medium scenario net benefit).  Line width encodes construction cost
    so cheap high-NB links stand out.  The status-quo network is shown
    as a grey backdrop.  Uses the same custom red–grey–blue colormap as
    plot_cost_result() so colours are consistent across all result maps.

    Unlike plot_cost_result() (which uses centroid points + interpolation),
    this function draws the actual LineString geometry of each candidate —
    more accurate for infrastructure that spans several hundred metres.

    Reads:  data/Network/processed/development_candidates.gpkg
            data/costs/net_benefits.gpkg
            data/Network/processed/developments_list.csv
            data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp
    Saves:  plot/results/{plot_name}.png
    """
    cands_path = "data/Network/processed/development_candidates.gpkg"
    nb_path    = "data/costs/net_benefits.gpkg"
    devs_path  = "data/Network/processed/developments_list.csv"
    lakes_path = r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp"

    for p in [cands_path, nb_path]:
        if not os.path.exists(p):
            print(f"[plot_nb_on_network] Missing: {p} — skipping")
            return

    cands = gpd.read_file(cands_path)
    nb    = gpd.read_file(nb_path)[["ID_new", "NB_s2", "C"]]
    nb["NB_s2"] = pd.to_numeric(nb["NB_s2"], errors="coerce")
    nb["C"]     = pd.to_numeric(nb["C"],     errors="coerce")

    # Attach NB scores to the actual edge geometry (not centroids)
    merged = cands.merge(nb, on="ID_new", how="inner")
    if merged.empty:
        print("[plot_nb_on_network] No matching rows after merge — skipping")
        return

    # Attach description for labels if available
    if os.path.exists(devs_path):
        devs   = pd.read_csv(devs_path)[["ID_new", "dev_type"]]
        merged = merged.merge(devs, on="ID_new", how="left")

    nb_vals = merged["NB_s2"] / 1e6
    vabs    = float(nb_vals.abs().quantile(0.95)) or 1.0
    min_val, max_val = nb_vals.min(), nb_vals.max()

    # Build same Red–Grey–Blue colormap used throughout plots.py
    n_intervals = 256
    gray_color  = [0.83, 0.83, 0.83, 1]
    if min_val < 0 and max_val > 0:
        total_range   = abs(min_val) + abs(max_val)
        neg_colors    = plt.cm.Reds_r(
            np.linspace(0.15, 0.8, int(n_intervals * abs(min_val) / total_range))
        )
        pos_colors    = plt.cm.Blues(
            np.linspace(0.3, 0.95, int(n_intervals * abs(max_val) / total_range))
        )
        tr = int(n_intervals * 0.2)
        all_colors = np.vstack((
            neg_colors[:-1],
            np.linspace(neg_colors[-1], gray_color, tr),
            np.linspace(gray_color, pos_colors[0], tr),
            pos_colors[1:],
        ))
    elif min_val >= 0:
        pos_colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_intervals))
        all_colors = np.vstack((
            np.linspace(gray_color, pos_colors[0], int(n_intervals * 0.3)),
            pos_colors[1:],
        ))
    else:
        neg_colors = plt.cm.Reds_r(np.linspace(0.2, 0.8, n_intervals))
        all_colors = np.vstack((
            neg_colors[:-1],
            np.linspace(neg_colors[-1], gray_color, int(n_intervals * 0.3)),
        ))

    cmap = LinearSegmentedColormap.from_list("nb_edge_cmap", all_colors)
    norm = plt.Normalize(vmin=-vabs, vmax=vabs)

    fig, ax = plt.subplots(figsize=(15, 10))

    # ── Base layers ───────────────────────────────────────────────────────────
    if os.path.exists(lakes_path):
        gpd.read_file(lakes_path).plot(ax=ax, color="lightblue", zorder=1)

    if isinstance(network, gpd.GeoDataFrame):
        network.plot(ax=ax, color="#bbbbbb", lw=1.0, zorder=2, label="Status-quo network")

    # ── Candidate edges coloured by NB_s2 ────────────────────────────────────
    c_min = merged["C"].abs().min() or 1.0
    c_max = merged["C"].abs().max() or 1.0

    for _, row in merged.iterrows():
        nb_m  = row["NB_s2"] / 1e6
        color = cmap(norm(nb_m))
        # Line width 1.5–5.5 proportional to construction cost
        c_abs = abs(row["C"]) if pd.notna(row["C"]) else c_min
        lw    = 1.5 + 4.0 * (c_abs - c_min) / max(c_max - c_min, 1)
        gpd.GeoDataFrame([row], crs=cands.crs).plot(
            ax=ax, color=[color], lw=lw, zorder=5)
        # ID label at midpoint
        try:
            mid = row.geometry.interpolate(0.5, normalized=True)
            ax.annotate(
                str(int(row["ID_new"])),
                xy=(mid.x, mid.y), xytext=(4, 4),
                textcoords="offset points",
                fontsize=8, fontweight="bold", color="black", zorder=10
            )
        except Exception:
            pass

    if isinstance(access_points, gpd.GeoDataFrame):
        access_points.plot(ax=ax, color="black", markersize=25, zorder=6)

    # ── City labels ───────────────────────────────────────────────────────────
    cities_path = r"data/manually_gathered_data/Cities.shp"
    if os.path.exists(cities_path):
        cities = gpd.read_file(cities_path, crs="epsg:2056")
        cities.plot(ax=ax, color="black", markersize=60, zorder=7)
        for _, loc_row in cities.iterrows():
            ax.annotate(
                loc_row["location"],
                xy=loc_row["geometry"].coords[0],
                ha="center", va="top", xytext=(0, -6),
                textcoords="offset points", fontsize=13, zorder=7
            )

    # ── Colorbar ──────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="2%", pad=0.5)
    cbar    = plt.colorbar(sm, cax=cax)
    cbar.set_label("Net Benefit NB_s2 [Mio. CHF]\n(line width ∝ construction cost)",
                   rotation=90, labelpad=16, fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    # ── North arrow ───────────────────────────────────────────────────────────
    ax.text(0.96, 0.92, "N", fontsize=28, weight="bold",
            ha="center", va="center", transform=ax.transAxes, zorder=1000)
    ax.add_patch(FancyArrowPatch(
        (0.96, 0.89), (0.96, 0.97), color="black", lw=2,
        arrowstyle="->", mutation_scale=25,
        transform=ax.transAxes, zorder=1000
    ))
    ax.add_artist(ScaleBar(1, location="lower right"))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "Development Candidates — Net Benefit on Network\n"
        "(colour = NB_s2, line width = construction cost)",
        fontsize=13, pad=8
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(1)
        spine.set_zorder(1000)

    if boundary:
        xmin, ymin, xmax, ymax = boundary.bounds
        ax.set_xlim(xmin - 100, xmax + 100)
        ax.set_ylim(ymin - 100, ymax + 100)

    plt.tight_layout()
    os.makedirs("plot/results", exist_ok=True)
    plt.savefig(f"plot/results/{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[plot_nb_on_network] saved → plot/results/{plot_name}.png")


def plot_duebendorf_zoom(df_costs, banned_area, title_bar, network=None,
                         access_points=None, plot_name="duebendorf_zoom",
                         col="NB_s2"):
    """
    Stand-alone zoomed map of the Dübendorf sub-area
    (E 2 687 000–2 697 500 / N 1 247 000–1 254 000, LV95/EPSG:2056).
    Same colour scheme as plot_cost_result so the two maps can be read
    side by side.  Development IDs are labelled at each line's midpoint.

    Reads:  data/Network/processed/development_candidates.gpkg
            data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp
    Saves:  plot/results/{plot_name}.png
    """
    DUB_E_MIN, DUB_E_MAX = 2_687_000, 2_697_500
    DUB_N_MIN, DUB_N_MAX = 1_247_000, 1_254_000

    # ── Join NB values to LineString edge geometries ──────────────────────────
    dev_geom = gpd.read_file(
        "data/Network/processed/development_candidates.gpkg")[["ID_new", "geometry"]]
    dev_geom["ID_new"] = dev_geom["ID_new"].astype(int)
    df_costs = df_costs.copy()
    df_costs["ID_new"] = df_costs["ID_new"].astype(int)
    df_plot = dev_geom.merge(
        df_costs.drop(columns=["geometry"], errors="ignore"),
        on="ID_new", how="inner")
    df_plot = gpd.GeoDataFrame(df_plot, geometry="geometry", crs="EPSG:2056")
    df_plot = df_plot.dropna(subset=[col])

    # ── Clip to Dübendorf extent ──────────────────────────────────────────────
    df_dub = df_plot.cx[DUB_E_MIN:DUB_E_MAX, DUB_N_MIN:DUB_N_MAX].copy()
    if df_dub.empty:
        print(f"[plot_duebendorf_zoom] No developments in Dübendorf extent — skipping")
        return

    # Same diverging colormap as the full corridor maps
    min_val, max_val = df_plot[col].min(), df_plot[col].max()
    cmap = _make_diverging_cmap(min_val, max_val)
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

    fig, ax = plt.subplots(figsize=(10, 8))

    # ── Base layers ───────────────────────────────────────────────────────────
    lakes_path = r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp"
    if os.path.exists(lakes_path):
        gpd.read_file(lakes_path).plot(ax=ax, color="lightblue", zorder=9)

    if isinstance(network, gpd.GeoDataFrame):
        network.plot(ax=ax, color="#888888", lw=1.0, zorder=10, alpha=0.55)

    if isinstance(access_points, gpd.GeoDataFrame):
        access_points.plot(ax=ax, color="black", markersize=40, zorder=12)

    # ── Coloured development lines ────────────────────────────────────────────
    df_dub.plot(ax=ax, column=col, cmap=cmap, norm=norm,
                linewidth=7, zorder=11, legend=False, capstyle="round")

    # ── ID labels ─────────────────────────────────────────────────────────────
    for _, row in df_dub.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        mid = row.geometry.interpolate(0.5, normalized=True)
        ax.annotate(str(int(row["ID_new"])), xy=(mid.x, mid.y),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=10, fontweight="bold", color="black", zorder=16,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              alpha=0.8, ec="none"))

    # ── Raster overlay ────────────────────────────────────────────────────────
    raster = rasterio.open(banned_area)
    rasterio.plot.show(raster, ax=ax,
                       cmap=ListedColormap(["white", "white"]), zorder=3)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.4)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(f"{title_bar} [Mio. CHF]", rotation=90, labelpad=16, fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    # ── Map extent, scale bar, north arrow ───────────────────────────────────
    ax.set_xlim(DUB_E_MIN, DUB_E_MAX)
    ax.set_ylim(DUB_N_MIN, DUB_N_MAX)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_artist(ScaleBar(1, location="lower right"))
    ax.text(0.96, 0.92, "N", fontsize=22, weight="bold",
            ha="center", va="center", transform=ax.transAxes, zorder=1000)
    ax.add_patch(FancyArrowPatch((0.96, 0.89), (0.96, 0.97), color="black", lw=2,
                                 arrowstyle="->", mutation_scale=20,
                                 transform=ax.transAxes, zorder=1000))
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor("black")
        sp.set_linewidth(1); sp.set_zorder(1000)

    water_patch = mpatches.Patch(facecolor="lightblue", label="Water bodies",
                                 edgecolor="black", linewidth=1)
    ax.legend(handles=[water_patch], loc="upper center",
              bbox_to_anchor=(0.5, -0.02), fontsize=12, frameon=False)

    ax.set_title(f"Dübendorf detail — {title_bar}", fontsize=13, pad=8)

    plt.tight_layout()
    os.makedirs("plot/results", exist_ok=True)
    plt.savefig(f"plot/results/{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[plot_duebendorf_zoom] saved → plot/results/{plot_name}.png")


def plot_bcr_bar(plot_name="bcr_bar"):
    """
    Horizontal bar chart of Benefit-Cost Ratio (BCR) per development,
    sorted ascending.  BCR = (T_s2 + R_s2 + S_s2) / |C + M|.
    A vertical dashed line at BCR = 1 marks the break-even threshold.
    Bars are green (BCR ≥ 1) or red (BCR < 1).

    Reads:  data/costs/net_benefits.gpkg
    Saves:  plot/results/{plot_name}.png
    """
    nb_path = "data/costs/net_benefits.gpkg"
    if not os.path.exists(nb_path):
        print(f"[plot_bcr_bar] Missing: {nb_path} — skipping")
        return

    nb = gpd.read_file(nb_path)
    for c in ["C", "M", "T_s2", "R_s2", "S_s2"]:
        nb[c] = pd.to_numeric(nb[c], errors="coerce")

    nb["total_costs"]    = nb["C"].abs() + nb["M"].abs()
    nb["total_benefits"] = nb["T_s2"] + nb["R_s2"] + nb["S_s2"]
    nb = nb[nb["total_costs"] > 0].copy()
    nb["BCR"] = nb["total_benefits"] / nb["total_costs"]
    nb = nb.sort_values("BCR", ascending=True).reset_index(drop=True)

    colors = ["#27ae60" if v >= 1 else "#e74c3c" for v in nb["BCR"]]

    fig, ax = plt.subplots(figsize=(9, max(5, len(nb) * 0.45)))
    y = np.arange(len(nb))
    ax.barh(y, nb["BCR"], color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(1.0, color="black", lw=1.5, linestyle="--", zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(nb["ID_new"].astype(int).astype(str), fontsize=9)
    ax.set_xlabel("Benefit-Cost Ratio  (T + R + S) / |C + M|", fontsize=11)
    ax.set_title("Benefit-Cost Ratio per Development\n"
                 "(medium growth scenario  ·  BCR > 1 = net positive)",
                 fontsize=12, pad=8)
    ax.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)

    legend_handles = [
        mpatches.Patch(color="#27ae60", label="BCR ≥ 1  (benefits exceed costs)"),
        mpatches.Patch(color="#e74c3c", label="BCR < 1  (costs exceed benefits)"),
        plt.Line2D([0], [0], color="black", lw=1.5, linestyle="--",
                   label="Break-even  (BCR = 1)"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="lower right")

    plt.tight_layout()
    os.makedirs("plot/results", exist_ok=True)
    plt.savefig(f"plot/results/{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[plot_bcr_bar] saved → plot/results/{plot_name}.png")
