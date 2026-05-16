import pandas as pd
from scipy.spatial import Voronoi
import osmnx as ox
from pyproj import Transformer
import sys
import numpy as np
from data_import import *
from plots import *
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import Voronoi
import os


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : scipy.spatial.Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : ndarray
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    Source: https://stackoverflow.com/a/20678647
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges.get(p1, [])
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            # Append to the new region
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # store region
        new_regions.append(new_region.tolist())

    # Remove infinite vertices
    # finite_vertices = [v for i, v in enumerate(new_vertices) if i not in vor.vertices]

    return new_regions, np.asarray(new_vertices)


def get_voronoi_status_quo(corridor_polygon=None, nodes_gdf=None):
    """
    Computes Euclidean Voronoi polygons for corridor cycling network nodes.

    Every node is an access point and gets a Voronoi cell representing the
    area for which it is the closest entry point to the cycling network.
    Always recomputed each run to stay consistent with the current network.

    Parameters
    ----------
    corridor_polygon : shapely.geometry.Polygon, optional
        If provided and nodes_gdf is None, restricts nodes to those
        intersecting this polygon.
    nodes_gdf : GeoDataFrame, optional
        Pre-filtered node GeoDataFrame to use directly (e.g. points_corridor).
        When supplied, corridor_polygon is only used for clipping, not filtering.

    Output: data/Voronoi/voronoi_status_quo_euclidian.gpkg
    """
    os.makedirs('data/Voronoi', exist_ok=True)

    # ------------------------------------------------------------------
    # 1. LOAD corridor nodes
    # ------------------------------------------------------------------
    if nodes_gdf is not None:
        # Use the caller-supplied node set directly (exact same nodes as the run)
        access_nodes = nodes_gdf.copy().reset_index(drop=True)
        if access_nodes.crs is None:
            access_nodes = access_nodes.set_crs("EPSG:2056")
    else:
        nodes = gpd.read_file('data/Network/processed/points.gpkg')
        if nodes.crs is None:
            nodes = nodes.set_crs("EPSG:2056")
        access_nodes = nodes.copy().reset_index(drop=True)

        # Optionally restrict to corridor
        if corridor_polygon is not None:
            poly_gdf = gpd.GeoDataFrame({'geometry': [corridor_polygon]}, crs="EPSG:2056")
            access_nodes = gpd.sjoin(
                access_nodes, poly_gdf, how='inner', predicate='intersects'
            ).drop(columns=['index_right'], errors='ignore').reset_index(drop=True)

    print(f"  Computing Voronoi for {len(access_nodes)} access points...")

    if len(access_nodes) < 4:
        raise ValueError("Need at least 4 access points for Voronoi tessellation")

    # ------------------------------------------------------------------
    # 2. COMPUTE Voronoi
    # ------------------------------------------------------------------
    coords = np.array([(geom.x, geom.y) for geom in access_nodes.geometry])
    vor    = Voronoi(coords)
    regions, vertices = voronoi_finite_polygons_2d(vor, radius=50000)  # 50km — always enough

    # ------------------------------------------------------------------
    # 3. BUILD GeoDataFrame — one polygon per access point
    # ------------------------------------------------------------------
    polygons = [Polygon(vertices[region]) for region in regions]

    voronoi_gdf = gpd.GeoDataFrame(
        {'ID_point': access_nodes['ID_point'].values},
        geometry=polygons,
        crs="EPSG:2056"
    )

    # Clip to corridor if provided
    if corridor_polygon is not None:
        voronoi_gdf['geometry'] = voronoi_gdf.geometry.intersection(corridor_polygon)
        voronoi_gdf = voronoi_gdf[~voronoi_gdf.geometry.is_empty].copy()

    voronoi_gdf.to_file('data/Voronoi/voronoi_status_quo_euclidian.gpkg', driver='GPKG')

    print(f"  -> {len(voronoi_gdf)} Voronoi polygons saved")

    return voronoi_gdf


def plot_voronoi_status_quo(voronoi_gdf, access_nodes=None, edges_gdf=None, corridor_polygon=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import geopandas as gpd
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, color_by_area in zip(axes, [False, True]):
        # Corridor boundary
        if corridor_polygon is not None:
            gpd.GeoDataFrame({'geometry': [corridor_polygon]}, crs="EPSG:2056").boundary.plot(
                ax=ax, color='black', linewidth=1.5, linestyle='--', zorder=4)

        if color_by_area:
            # Colour cells by area — large = underserved
            voronoi_gdf = voronoi_gdf.copy()
            voronoi_gdf['area_m2'] = voronoi_gdf.geometry.area
            norm = mcolors.Normalize(vmin=voronoi_gdf['area_m2'].min(),
                                     vmax=voronoi_gdf['area_m2'].max())
            cmap = cm.RdYlGn_r  # red = large catchment = underserved
            voronoi_gdf.plot(ax=ax, column='area_m2', cmap='RdYlGn_r',
                             edgecolor='white', linewidth=0.5, alpha=0.7, zorder=1)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label='Catchment area (m²)', shrink=0.6)
            ax.set_title('Voronoi Cells — coloured by catchment size\n(red = underserved)', fontsize=11)
        else:
            voronoi_gdf.plot(ax=ax, facecolor='lightyellow', edgecolor='steelblue',
                             linewidth=0.8, alpha=0.7, zorder=1)
            ax.set_title('Voronoi Status Quo — catchment areas', fontsize=11)

        # Network edges
        if edges_gdf is not None:
            edges_gdf.plot(ax=ax, color='steelblue', linewidth=0.8, alpha=0.6, zorder=2)

        # Access points
        if access_nodes is not None:
            ax.scatter(access_nodes.geometry.x, access_nodes.geometry.y,
                       s=15, color='black', alpha=0.8, zorder=3, label='Access points')
            ax.legend(fontsize=8)

        ax.set_aspect('equal')

    plt.suptitle('Voronoi Tessellation — Cycling Network Status Quo', fontsize=13)
    plt.tight_layout()
    plt.savefig('data/Voronoi/voronoi_status_quo_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved → data/Voronoi/voronoi_status_quo_plot.png")
