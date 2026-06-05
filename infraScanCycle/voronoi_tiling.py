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
    """
    2 × 3 choropleth of Voronoi catchments coloured by scenario growth.

    Rows    : Population (top) · Employment (bottom)
    Columns : S2 — Low · S1 — Medium · S3 — High   (ordered low → high)

    Requires voronoi_gdf to have columns s1_pop, s2_pop, s3_pop,
    s1_empl, s2_empl, s3_empl (added in-place by scenario_to_voronoi).
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import box

    # ── Zoom to corridor bounds (+ 5 % padding) ───────────────────────────────
    if corridor_polygon is not None:
        minx, miny, maxx, maxy = corridor_polygon.bounds
    else:
        minx, miny, maxx, maxy = voronoi_gdf.total_bounds
    pad_x = (maxx - minx) * 0.05
    pad_y = (maxy - miny) * 0.05
    xlim = (minx - pad_x, maxx + pad_x)
    ylim = (miny - pad_y, maxy + pad_y)

    clip_box     = box(xlim[0], ylim[0], xlim[1], ylim[1])
    voronoi_clip = voronoi_gdf.clip(clip_box)
    edges_clip   = edges_gdf.clip(clip_box) if edges_gdf is not None else None
    pts_in       = (access_nodes[access_nodes.geometry.within(clip_box)]
                    if access_nodes is not None else None)

    # Scenario columns ordered low → medium → high
    scenario_cols = [
        ("s2_pop",  "s2_empl",  "S2 — Low growth"),
        ("s1_pop",  "s1_empl",  "S1 — Medium growth"),
        ("s3_pop",  "s3_empl",  "S3 — High growth"),
    ]
    row_specs = [
        ("pop",  ["s2_pop",  "s1_pop",  "s3_pop"],  "Population",  "YlOrRd"),
        ("empl", ["s2_empl", "s1_empl", "s3_empl"], "Employment",  "YlGnBu"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(19, 11),
                             facecolor="white", constrained_layout=True)

    for row_idx, (_, cols, var_label, cmap_name) in enumerate(row_specs):
        # Shared colour scale across the three scenario panels for this variable
        valid = voronoi_clip[cols].replace(0, np.nan)
        vmin  = valid.min().min()
        vmax  = valid.max().max()
        norm  = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap  = plt.cm.get_cmap(cmap_name)

        for col_idx, (pop_col, empl_col, scen_title) in enumerate(scenario_cols):
            ax  = axes[row_idx, col_idx]
            col = cols[col_idx]   # the right column for this row & scenario

            # Voronoi choropleth
            voronoi_clip.plot(
                column=col, ax=ax, cmap=cmap, norm=norm,
                edgecolor="#BDBDBD", linewidth=0.4, alpha=0.90,
                missing_kwds={"color": "#EEEEEE"},
            )

            # Network edges
            if edges_clip is not None:
                edges_clip.plot(ax=ax, color="#444444", linewidth=0.7,
                                alpha=0.5, zorder=3)

            # Corridor boundary
            if corridor_polygon is not None:
                gpd.GeoDataFrame(
                    {"geometry": [corridor_polygon]}, crs="EPSG:2056"
                ).boundary.plot(ax=ax, color="black", linewidth=1.5,
                                linestyle="--", zorder=5)

            # Access-point nodes
            if pts_in is not None:
                ax.scatter(pts_in.geometry.x, pts_in.geometry.y,
                           s=8, color="black", alpha=0.7, zorder=6)

            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.axis("off")

            # Column title (top row only)
            if row_idx == 0:
                ax.set_title(scen_title, fontsize=11,
                             fontweight="bold" if col_idx == 1 else "normal", pad=5)

            # Scenario total as subtitle (bottom row)
            total = voronoi_clip[col].sum()
            ax.annotate(f"total: {total:,.0f}", xy=(0.5, 0.02),
                        xycoords="axes fraction", ha="center", fontsize=8,
                        color="#333333")

        # Row label
        axes[row_idx, 0].set_ylabel(var_label, fontsize=10)

        # Shared colour bar for this row
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axes[row_idx, :], shrink=0.50,
                          pad=0.01, aspect=28)
        cb.set_label(f"{var_label} per Voronoi catchment", fontsize=9)
        cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Voronoi Tessellation — Scenario Growth (corridor close-up)\n"
        "columns: Low (S2) · Medium (S1) · High (S3)   ·   "
        "grey lines = network   ·   dots = nodes",
        fontsize=13, fontweight="bold",
    )

    plt.savefig("data/Voronoi/voronoi_status_quo_plot.png", dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.show()
    print("Plot saved → data/Voronoi/voronoi_status_quo_plot.png")
