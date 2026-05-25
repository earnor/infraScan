import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from rasterio.features import geometry_mask, rasterize as rio_rasterize
from scipy.stats.qmc import LatinHypercube
import re
import glob
import tkinter as tk
from tkinter.simpledialog import Dialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from shapely.geometry import shape, LineString, GeometryCollection
from shapely.ops import nearest_points, split
import fiona
from scipy.optimize import minimize
from tqdm import tqdm
import pulp
import os
import requests
import zipfile
import rasterio
import geopandas as gpd
from shapely.validation import make_valid
from shapely import wkt
from data_import import *







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

    newgdf = gdf.copy().reset_index(drop=True)
    print(f"  Total points generated: {len(newgdf)}")

    # Ensure CRS is EPSG:2056 before any spatial operation
    if newgdf.crs is None:
        newgdf = newgdf.set_crs("EPSG:2056")
    elif newgdf.crs.to_epsg() != 2056:
        newgdf = newgdf.to_crs("EPSG:2056")

    # ------------------------------------------------------------------
    # Helper: filter points that fall INSIDE a vector polygon layer
    # Returns the input GDF with points inside the polygon removed
    # ------------------------------------------------------------------
    def filter_by_polygon(points_gdf, shapefile_path, label):
        print(f"  Filtering: {label}")
        if not os.path.exists(shapefile_path):
            print(f"    Warning: {shapefile_path} not found — skipping")
            return points_gdf

        poly_gdf = gpd.read_file(shapefile_path)
        if poly_gdf.crs is None:
            poly_gdf = poly_gdf.set_crs("EPSG:2056")
        elif poly_gdf.crs.to_epsg() != 2056:
            poly_gdf = poly_gdf.to_crs("EPSG:2056")


        poly_gdf['geometry'] = poly_gdf.geometry.apply(lambda g: make_valid(g) if g is not None else g)
        poly_gdf = poly_gdf[poly_gdf.geometry.notna() & ~poly_gdf.geometry.is_empty].reset_index(drop=True)

        # sjoin: keep only points NOT inside the polygon
        joined = gpd.sjoin(points_gdf, poly_gdf[['geometry']], how='left', predicate='within')
        mask = joined['index_right'].isna()
        # Deduplicate in case a point touches multiple polygons
        mask = mask[~mask.index.duplicated(keep='first')]
        result = points_gdf[mask.values].copy().reset_index(drop=True)
        print(f"    Remaining: {len(result)}")
        return result

    # ------------------------------------------------------------------
    # 1. Schutzanordnung Natur und Landschaft
    # ------------------------------------------------------------------
    newgdf = filter_by_polygon(
        newgdf,
        "data/landuse_landcover/Schutzzonen/Schutzanordnungen_Natur_und_Landschaft_-SAO-_-OGD/FNS_SCHUTZZONE_F.shp",
        "Schutzanordnung Natur und Landschaft"
    )

    # ------------------------------------------------------------------
    # 2. Forest (Waldareal)
    # ------------------------------------------------------------------
    newgdf = filter_by_polygon(
        newgdf,
        "data/landuse_landcover/Schutzzonen/Waldareal_-OGD/WALD_WALDAREAL_F.shp",
        "Forest (Waldareal)"
    )

    # ------------------------------------------------------------------
    # 3. Network buffer — keep only points within 2500m of existing network
    #    Logic inverted: points OUTSIDE the buffer are dropped
    # ------------------------------------------------------------------
    print("  Filtering: Network proximity (within 2500m of existing network)")
    network_path = "data/Network/processed/edges.gpkg"
    if os.path.exists(network_path):
        network_gdf = gpd.read_file(network_path)
        if network_gdf.crs.to_epsg() != 2056:
            network_gdf = network_gdf.to_crs("EPSG:2056")

        network_buf = network_gdf.copy()
        network_buf['geometry'] = network_gdf.geometry.buffer(1000)


        network_buf['geometry'] = network_buf.geometry.apply(lambda g: make_valid(g) if g is not None else g)
        network_buf = network_buf[['geometry']].dissolve()

        # Keep points that ARE within the buffer (inside = good here)
        joined = gpd.sjoin(newgdf, network_buf, how='left', predicate='within')
        mask = joined['index_right'].notna()
        mask = mask[~mask.index.duplicated(keep='first')]
        newgdf = newgdf[mask.values].copy().reset_index(drop=True)
        print(f"    Remaining: {len(newgdf)}")
    else:
        print(f"    Warning: {network_path} not found — skipping")

    # ------------------------------------------------------------------
    # 4. Protected zones (Raster check)
    #    Drop points that land on protected raster cells (value > 0, not nodata)
    # ------------------------------------------------------------------
    print("  Filtering: Protected zones (Raster)")
    raster_path = "data/landuse_landcover/processed/zone_no_infra/protected_area_corridor.tif"

    if os.path.exists(raster_path):
        indices_to_drop = []
        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)
            nodata = src.nodata

            for idx, row in newgdf.iterrows():
                x, y = row.geometry.x, row.geometry.y
                try:
                    row_i, col_i = src.index(x, y)  # returns (row, col)
                    if 0 <= row_i < raster_data.shape[0] and 0 <= col_i < raster_data.shape[1]:
                        value = raster_data[row_i, col_i]
                        # Drop if protected (value != nodata and value > 0)
                        if nodata is not None and value == nodata:
                            pass  # nodata = not protected, keep
                        elif np.isnan(float(value)):
                            pass  # NaN = not protected, keep
                        elif value > 0:
                            indices_to_drop.append(idx)
                    else:
                        indices_to_drop.append(idx)  # outside raster extent = drop
                except Exception:
                    indices_to_drop.append(idx)

        newgdf = newgdf.drop(index=indices_to_drop).reset_index(drop=True)
        print(f"    Remaining: {len(newgdf)}")
    else:
        print(f"    Warning: {raster_path} not found — skipping")


    # ------------------------------------------------------------------
    # Cleanup and export
    # ------------------------------------------------------------------
    if 'ID' in newgdf.columns:
        newgdf = newgdf.rename(columns={"ID": "ID_new"})

    # Reassign clean sequential IDs after all filtering
    newgdf['ID_new'] = range(len(newgdf))
    newgdf = newgdf.drop(columns=['index'], errors='ignore')
    newgdf = newgdf.set_crs("EPSG:2056", allow_override=True)

    print(f"  Final count after all filters: {len(newgdf)}")
    return newgdf


def get_idx_todrop(pt, filename):
    #with fiona.open(r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp") as input:
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


def nearest(row, geom_union, df1, df2, geom1_col='geometry', geom2_col='geometry', src_column=None):
    """Find the nearest point and return the corresponding value from specified column."""

    # Find the geometry that is closest
    #nearest = df2[geom2_col] == nearest_points(row[geom1_col], geom_union)[1]
    nearest = df2[geom2_col] == nearest_points(geom_union,row[geom1_col])[1]

    # Get the corresponding value from df2 (matching is based on the geometry)
    value = df2[nearest][src_column].iloc[0]

    return value


def near(point, network_gdf,pts):
    # find the nearest point and return the corresponding Place value
    nearest = network_gdf.geometry == nearest_points(point, pts)[1]
    return network_gdf[nearest].geometry.iloc[0]


def connect_points_to_network(new_point_gdf, network_gdf, edges_gdf=None):
    import geopandas as gpd
    from shapely.geometry import LineString, Point
    from shapely.ops import nearest_points
    import os

    print("  Connecting candidate points to nearest cycling network nodes...")

    access_nodes = network_gdf.copy().reset_index(drop=True)
    if 'ID_point' not in access_nodes.columns:
        access_nodes['ID_point'] = access_nodes.index

    print(f"    {len(access_nodes)} access nodes available as connection targets")

    for df in [new_point_gdf, access_nodes]:
        drop_cols = [c for c in ['index_right', 'index_left'] if c in df.columns]
        df.drop(columns=drop_cols, inplace=True)

    # Nearest node join
    joined = gpd.sjoin_nearest(
        new_point_gdf.reset_index(drop=True),
        access_nodes[['ID_point', 'geometry']].reset_index(drop=True),
        how='left',
        distance_col='dist_to_node'
    ).drop(columns=['index_right'], errors='ignore')

    joined = joined[~joined.index.duplicated(keep='first')].reset_index(drop=True)

    node_lookup = access_nodes.set_index('ID_point')['geometry']
    joined['node_id']   = joined['ID_point']
    joined['node_geom'] = joined['node_id'].map(node_lookup)

    # Straight-line geometry (used for graph/routing)
    joined['geometry'] = joined.apply(
        lambda row: LineString([row.geometry, row['node_geom']])
        if row['node_geom'] is not None else None,
        axis=1
    )
    joined = joined[joined['geometry'].notnull()].copy()

    # Realistic visual path: new point → snap on nearest edge → node
    if edges_gdf is not None:
        from shapely.validation import make_valid
        edges_gdf = edges_gdf.copy()
        edges_gdf['geometry'] = edges_gdf.geometry.apply(lambda g: make_valid(g) if g is not None else g)
        edges_gdf = edges_gdf[edges_gdf.geometry.notna() & ~edges_gdf.geometry.is_empty]
        edges_union = edges_gdf.geometry.unary_union

        def make_visual_geom(orig_pt, node_geom):
            snap_pt = nearest_points(orig_pt, edges_union)[1]
            return LineString([orig_pt, snap_pt, node_geom])

        joined['visual_geom'] = joined.apply(
            lambda row: make_visual_geom(
                Point(row.geometry.coords[0]),
                row['node_geom']
            ).wkt,
            axis=1
        )
        print("    visual_geom: new point → edge snap → node (stored as WKT)")
    else:
        joined['visual_geom'] = joined['geometry'].apply(lambda g: g.wkt)
        print("    visual_geom: no edges_gdf provided — fallback to straight line")

    links = gpd.GeoDataFrame(joined, geometry='geometry', crs="EPSG:2056")
    links = links.drop(columns=['node_geom', 'ID_point'], errors='ignore')
    links['ID_link'] = range(len(links))

    os.makedirs('data/Network/processed', exist_ok=True)
    links.to_file('data/Network/processed/new_links.gpkg', driver='GPKG')

    print(f"  -> {len(links)} candidate links created")
    print(f"     distance range: {links['dist_to_node'].min():.0f}–{links['dist_to_node'].max():.0f} m")

    return links


def plot_connections(new_point_gdf, network_gdf, links, polygon=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, show_dist in zip(axes, [False, True]):
        if polygon:
            import geopandas as gpd
            gpd.GeoDataFrame({'geometry': [polygon]}, crs="EPSG:2056").boundary.plot(
                ax=ax, color='black', linewidth=1.5, linestyle='--', zorder=1)

        # Network nodes
        network_gdf.plot(ax=ax, color='steelblue', markersize=8, alpha=0.6, zorder=2)

        # Connector links
        if show_dist:
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            norm = mcolors.Normalize(vmin=links['dist_to_node'].min(),
                                     vmax=links['dist_to_node'].max())
            cmap = cm.RdYlGn_r
            for _, row in links.iterrows():
                gpd.GeoDataFrame([row], crs=links.crs).plot(
                    ax=ax, color=[cmap(norm(row['dist_to_node']))], linewidth=1, alpha=0.7, zorder=3)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label='Distance to network (m)', shrink=0.6)
            ax.set_title('Connector Links — coloured by distance', fontsize=11)
        else:
            links.plot(ax=ax, color='orange', linewidth=0.8, alpha=0.7, zorder=3)
            ax.set_title('Connector Links — new points to network', fontsize=11)

        # New development points
        new_point_gdf.plot(ax=ax, color='red', markersize=15, marker='*',
                           alpha=0.9, zorder=4)

        legend = [
            Line2D([0], [0], color='steelblue', marker='o', linestyle='None',
                   markersize=6, label=f'Network nodes ({len(network_gdf)})'),
            Line2D([0], [0], color='red', marker='*', linestyle='None',
                   markersize=10, label=f'New points ({len(new_point_gdf)})'),
            Line2D([0], [0], color='orange', linewidth=1.5,
                   label=f'Connector links ({len(links)})'),
        ]
        ax.legend(handles=legend, fontsize=8)
        ax.set_aspect('equal')

    plt.suptitle('Last-Mile Connections to Cycling Network', fontsize=13)
    plt.tight_layout()
    plt.savefig('data/Network/processed/connections_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved → data/Network/processed/connections_plot.png")


def build_combined_network(nodes_gdf, edges_gdf, generated_points, routed_links):
    """
    Saves the status-quo ALLTAG network and builds a combined network
    that merges existing nodes/edges with generated points and new routed links.

    Inputs:
        nodes_gdf        – existing network nodes (from reformat_network)
        edges_gdf        – existing edges with attributes (from get_edge_attributes)
        generated_points – filtered generated development points
        routed_links     – new routed connector links (from routing_raster, filtered)

    Outputs (saved to data/Network/):
        status_quo/nodes.gpkg + edges.gpkg   – original network
        combined/nodes.gpkg  + edges.gpkg    – merged network
    """
    import geopandas as gpd
    import pandas as pd
    import os

    os.makedirs('data/Network/status_quo', exist_ok=True)
    os.makedirs('data/Network/combined', exist_ok=True)

    # ------------------------------------------------------------------
    # 1. SAVE STATUS QUO as-is
    # ------------------------------------------------------------------
    nodes_gdf.to_file('data/Network/status_quo/nodes.gpkg', driver='GPKG')
    edges_gdf.to_file('data/Network/status_quo/edges.gpkg', driver='GPKG')
    print(f"  Status quo saved: {len(nodes_gdf)} nodes, {len(edges_gdf)} edges")

    # ------------------------------------------------------------------
    # 2. ASSIGN EDGE ATTRIBUTES TO NEW LINKS
    #    New links = Nebenverbindung equivalent (shared paths, low speed)
    # ------------------------------------------------------------------
    NEW_LINK_ATTRS = {'ffs': 12, 'capacity': 200, 'oneway': 0}

    new_edges = routed_links.copy()

    new_edges['ffs'] = NEW_LINK_ATTRS['ffs']
    new_edges['capacity'] = NEW_LINK_ATTRS['capacity']
    new_edges['oneway'] = NEW_LINK_ATTRS['oneway']
    new_edges['ROUTENTYP'] = 'New Link'

    # Travel time from routed length (prefer length_routed_m, fallback to geometry)
    if 'length_routed_m' in new_edges.columns:
        new_edges['length_m'] = new_edges['length_routed_m']
    else:
        new_edges['length_m'] = new_edges.geometry.length

    new_edges['tt_min'] = (new_edges['length_m'] / 1000) / new_edges['ffs'] * 60

    # Use visual_geom as geometry if available (routed path shape)
    if 'visual_geom' in new_edges.columns:
        from shapely import wkt
        new_edges['geometry'] = new_edges['visual_geom'].apply(
            lambda g: wkt.loads(g) if isinstance(g, str) else g
        )
        new_edges = gpd.GeoDataFrame(new_edges, geometry='geometry', crs="EPSG:2056")

    # ------------------------------------------------------------------
    # 3. COMBINE NODES: existing + generated points
    # ------------------------------------------------------------------
    gen_nodes = generated_points.copy()
    gen_nodes['x'] = gen_nodes.geometry.x
    gen_nodes['y'] = gen_nodes.geometry.y
    gen_nodes['is_intersection'] = 0
    gen_nodes['is_endpoint'] = 1
    gen_nodes['degree'] = 1

    id_offset = int(nodes_gdf['ID_point'].max()) + 1 if 'ID_point' in nodes_gdf.columns else len(nodes_gdf)
    gen_nodes['ID_point'] = range(id_offset, id_offset + len(gen_nodes))
    gen_nodes['source'] = 'generated'

    existing_nodes = nodes_gdf.copy()
    existing_nodes['source'] = 'existing'

    # Deduplicate columns before concat
    existing_nodes = existing_nodes.loc[:, ~existing_nodes.columns.duplicated()]
    gen_nodes = gen_nodes.loc[:, ~gen_nodes.columns.duplicated()]

    shared_cols = list(dict.fromkeys(
        c for c in existing_nodes.columns if c in gen_nodes.columns
    ))  # preserves order, deduped

    combined_nodes = pd.concat(
        [existing_nodes[shared_cols], gen_nodes[shared_cols]],
        ignore_index=True
    )
    combined_nodes = gpd.GeoDataFrame(combined_nodes, geometry='geometry', crs="EPSG:2056")

    # ------------------------------------------------------------------
    # 4. COMBINE EDGES: existing + new links
    # ------------------------------------------------------------------
    # Keep only columns present in both (+ fill missing with NaN)
    existing_edges = edges_gdf.copy()
    existing_edges['source'] = 'existing'
    new_edges['source'] = 'new_link'

    combined_edges = pd.concat([existing_edges, new_edges], ignore_index=True)
    combined_edges = gpd.GeoDataFrame(combined_edges, geometry='geometry', crs="EPSG:2056")
    combined_edges['ID_edge'] = range(len(combined_edges))

    # ------------------------------------------------------------------
    # 5. SAVE COMBINED NETWORK
    # ------------------------------------------------------------------
    combined_nodes.to_file('data/Network/combined/nodes.gpkg', driver='GPKG')
    combined_edges.to_file('data/Network/combined/edges.gpkg', driver='GPKG')

    combined_nodes.drop(columns='geometry').to_csv('data/Network/combined/nodes.csv', index=False)
    combined_edges.drop(columns='geometry').to_csv('data/Network/combined/edges.csv', index=False)

    print(f"  Combined network saved:")
    print(f"    Nodes: {len(existing_nodes)} existing + {len(gen_nodes)} generated = {len(combined_nodes)}")
    print(f"    Edges: {len(existing_edges)} existing + {len(new_edges)} new links = {len(combined_edges)}")
    print(f"    New link ffs: {NEW_LINK_ATTRS['ffs']} km/h | capacity: {NEW_LINK_ATTRS['capacity']} bikes/h")
    print(f"    New link tt_min range: {new_edges['tt_min'].min():.1f}–{new_edges['tt_min'].max():.1f} min")

    return combined_nodes, combined_edges


def create_nearest_gdf(filtered_rand_gdf):
    nearest_gdf = filtered_rand_gdf[["ID_new", "ID_point", "geometry_current"]].set_geometry("geometry_current")
    #nearest_gdf = nearest_gdf.rename({"ID":"PointID", "index_right":"NearestAccID"})
    #nearest_df = filtered_rand_gdf.assign(PointID=filtered_rand_gdf["ID"],NearestAccID=filtered_rand_gdf["index_right"],x=filtered_rand_gdf["x"],y=filtered_rand_gdf["y"])
    #nearest_gdf = gpd.GeoDataFrame(nearest_df,geometry=gpd.points_from_xy(nearest_df.x,nearest_df.y),crs="epsg:2056")
    return nearest_gdf


def create_lines(rand_pts_gdf, nearest_highway_pt_gdf):
    rand_pts_gdf = rand_pts_gdf.sort_values(by="ID_new")
    points = rand_pts_gdf.geometry
    nearest_highway_pt_gdf = nearest_highway_pt_gdf.sort_values(by="ID_new")
    nearest_points = nearest_highway_pt_gdf.geometry

    line_geometries = [LineString([points.iloc[i], nearest_points.iloc[i]]) for i in range(len(rand_pts_gdf))]
    line_gdf = gpd.GeoDataFrame(geometry=line_geometries)
    line_gdf["ID_new"] = rand_pts_gdf["ID_new"]
    line_gdf["ID_current"] = nearest_highway_pt_gdf["ID_point"]

    line_gdf = line_gdf.set_crs("epsg:2056")
    line_gdf.to_file(r"data/Network/processed/new_links.gpkg")
    return


def plot_lines_to_network(points_gdf,lines_gdf):
    points_gdf.plot(marker='*', color='green', markersize=5)
    base = lines_gdf.plot(edgecolor='black')
    points_gdf.plot(ax=base, marker='o', color='red', markersize=5)
    plt.savefig(r"plot/predict/230822_network-generation.png", dpi=300)
    return None


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


def routing_raster(raster_path, links_path='data/Network/processed/new_links_corridor.gpkg'):

    if not os.path.exists(links_path):
        raise FileNotFoundError(f"Missing: {links_path} — run connect_points_to_network() first")

    generated_links = gpd.read_file(links_path)
    print(f"  Routing {len(generated_links)} candidate links...")

    new_geometries      = []
    inaccessible_points = []

    with rasterio.open(raster_path) as dataset:
        raster_data = dataset.read(1)
        transform   = dataset.transform

        print("  Building routing graph from raster...")
        graph = raster_to_graph(raster_data)

        for i, row in generated_links.iterrows():
            line        = row.geometry
            start_point = line.coords[0]
            end_point   = line.coords[-1]

            start_idx = rasterio.transform.rowcol(
                transform, xs=start_point[0], ys=start_point[1]
            )
            end_idx = rasterio.transform.rowcol(
                transform, xs=end_point[0], ys=end_point[1]
            )

            path = None
            try:
                path, inaccessible_points = find_path(
                    graph, start_idx, end_idx, inaccessible_points, end_point
                )
            except Exception as e:
                path = None

            # ----------------------------------------------------------
            # GUARD: path must have at least 2 points for a LineString
            # A single-point path means start == end (same raster cell)
            # or find_path returned a degenerate result
            # ----------------------------------------------------------
            if path and len(path) >= 2:
                coords = [
                    rasterio.transform.xy(transform, rows=p[0], cols=p[1], offset='center')
                    for p in path
                ]
                # Snap to exact original coordinates (avoids 25m raster offset)
                coords[0]  = start_point
                coords[-1] = end_point

                # Final guard: deduplicate consecutive identical points
                coords = [c for j, c in enumerate(coords)
                          if j == 0 or c != coords[j - 1]]

                if len(coords) >= 2:
                    new_geometries.append(LineString(coords))
                else:
                    new_geometries.append(None)

            elif path and len(path) == 1:
                # Start and end in the same raster cell — use straight line
                if start_point != end_point:
                    new_geometries.append(LineString([start_point, end_point]))
                else:
                    new_geometries.append(None)
            else:
                new_geometries.append(None)

    # ------------------------------------------------------------------
    # Update geometries and clean up
    # ------------------------------------------------------------------
    generated_links = generated_links.copy()
    generated_links['geometry'] = new_geometries

    routed  = generated_links.dropna(subset=['geometry']).copy()
    routed  = gpd.GeoDataFrame(routed, geometry='geometry', crs="EPSG:2056")
    routed['length_routed_m'] = routed.geometry.length

    dropped = len(generated_links) - len(routed)
    print(f"  -> {len(routed)} routed links "
          f"({dropped} dropped — no valid path found)")
    if len(routed) > 0:
        print(f"     routed length range: "
              f"{routed['length_routed_m'].min():.0f}–"
              f"{routed['length_routed_m'].max():.0f} m")

    os.makedirs('data/Network/processed', exist_ok=True)
    routed.to_file('data/Network/processed/new_links_realistic.gpkg', driver='GPKG')

    if inaccessible_points:
        pd.DataFrame(inaccessible_points, columns=['x', 'y']) \
          .to_csv('data/Network/processed/points_inaccessible.csv', index=False)
        print(f"  -> {len(inaccessible_points)} inaccessible points logged")

    return routed


def raster_to_graph(raster_data):
    rows, cols = raster_data.shape
    graph = nx.grid_2d_graph(rows, cols)

    # Add diagonal edges
    graph.add_edges_from([
        ((x, y), (x + 1, y + 1))
        for x in range(cols - 1)
        for y in range(rows - 1)
    ] + [
        ((x + 1, y), (x, y + 1))
        for x in range(cols - 1)
        for y in range(rows - 1)
    ], weight=1.4)

    # Remove nodes for blocked cells entirely — hard barrier, A* cannot pass through
    for y in range(rows):
        for x in range(cols):
            if raster_data[y, x] > 0 and graph.has_node((y, x)):
                graph.remove_node((y, x))

    return graph


def find_path(graph, start, end, list_no_path, point_end):
    try:
        def heuristic(a, b):
            return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        path = nx.astar_path(graph, start, end, heuristic=heuristic, weight='weight')
        return path, list_no_path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        list_no_path.append(point_end)
        print(f"  No obstacle-free path: {point_end}")
        return None, list_no_path



def plot_corridor(network, limits, location, current_nodes=False, new_nodes=False, new_links=False, access_link=False):

    fig, ax = plt.subplots(figsize=(10, 10))

    network = network[(network["Rank"] == 1) & (network["Opening Ye"] < 2023) & (network["NAME"] != 'Freeway Tunnel planned') & (
                network["NAME"] != 'Freeway planned')]

    # Define square to show perimeter of investigation
    square = Polygon([(limits[0], limits[2]), (limits[1], limits[2]), (limits[1], limits[3]), (limits[0], limits[3])])
    frame = gpd.GeoDataFrame(geometry=[square], crs=network.crs)

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
    plt.savefig(r"plot/network_base_generated.png", dpi=300)
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
    combined_gdf.to_file("data/Voronoi/combined_developments.gpkg", driver="GPKG")


def import_elevation_model(new_resolution):

    # Read CSV file containing the ZIP file links
    csv_file = r"data/elevation_model/ch.swisstopo.swissalti3d-pivq0Jb7.csv"
    df = pd.read_csv(csv_file, names=["url"], header=None)

    # Download and extract ZIP files
    for url in df["url"]:
        r = requests.get(url)
        zip_path = r"data/elevation_model/zip_files/temp.zip"
        with open(zip_path, 'wb') as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(r"data/elevation_model/extracted_xyz_files")

    # Find all XYZ files
    xyz_files = glob.glob(r"data/elevation_model/extracted_xyz_files/*.xyz")

    # Calculate the minimum coordinates based on the first file
    sample_data = pd.read_csv(xyz_files[0], sep=" ")
    min_x, min_y = sample_data['X'].min(), sample_data['Y'].min()

    # Process each file — collect into list, concat once to avoid O(n²) copies
    chunks = []
    for i, file in enumerate(xyz_files, start=1):
        chunks.append(downsample_elevation_xyz_file(file, min_x, min_y, resolution=new_resolution))
        print(f"Processed file {i}/{len(xyz_files)}: {file}")
    concatenated_data = pd.concat(chunks, ignore_index=True)
    del chunks
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
    with rasterio.open(r'data/elevation_model/elevation.tif', 'w', driver='GTiff',
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






