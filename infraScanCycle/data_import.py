import os
os.environ['USE_PYGEOS'] = '0'

import math
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point, MultiPoint, shape, box, Polygon
from shapely.ops import split, snap, linemerge, unary_union
from rasterio import crs

from rasterio.features import shapes, rasterize
from geopandas.tools import sjoin
from plots import *
import requests
import zipfile
import glob
import numpy as np
import rasterio
from rasterio.transform import from_origin

from shapely.validation import make_valid

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import geopandas as gpd


from collections import Counter
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D



def import_data(limits):

    # Arealstatistik - 1985, 1997, 2009, 2018
    #areal_stat = pd.read_csv(r'data/landuse_landcover/landcover/ag-b-00.03-37-area-csv.csv', sep=";")
    #areal_stat = areal_stat.drop(areal_stat.columns[[3,4,5,6,7,8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31,32,33,34,35]], axis=1)
    #areal_stat = areal_stat[["E", "N", "AS18_17", "AS18_4", "LU18_10", "LU18_4"]]
    #print(areal_stat.head(50).to_string())
    # AS85_17 17 Klassen gemäss Standardnomenklatur der Arealstatistik 1979/85
    # AS85_4  4 Hauptbereiche gemäss Standardnomenklatur der Arealstatistik 1979/85
    # LU85_10 10 Klassen der Bodennutzung der Arealstatistik 1979/85
    # LU85_4  4 Hauptbereiche der Bodennutzung der Arealstatistik 1979/85


    betriebzaehlung20 = pd.read_csv(r"data/independent_variable/statent/ag-b-00.03-22-STATENT2020/STATENT_2020.csv", sep=";")
    # BXXS2: Arbeitsstätte Sektor 2, BxxVZAS2: Vollzeitäquivalent Sektor 2
    betriebzaehlung20 = betriebzaehlung20[["B08VZAT", "E_KOORD", "N_KOORD"]].rename(
        {'B08VZAT': 'empl20'}, axis=1)
    empl20_ch = fill_raster_dataframe(betriebzaehlung20)
    csv_to_tiff(empl20_ch, attribute="empl20", path=r"data/independent_variable/processed/raw/empl20_ch.tif")

    empl20 = empl20_ch[(empl20_ch["E_COORD"] >= limits[0]) & (empl20_ch["E_COORD"] <= limits[2] - 100) & (empl20_ch["N_COORD"] >= limits[1]) & (empl20_ch["N_COORD"] <= limits[3] - 100)]
    csv_to_tiff(empl20, attribute="empl20", path=r"data/independent_variable/processed/raw/empl20.tif")

    population20 = pd.read_csv(r"data/independent_variable/statpop/ag-b-00.03-vz2020statpop/STATPOP2020.csv", sep=";")
    population20 = population20[["B20BTOT", "E_KOORD", "N_KOORD"]].rename({"B20BTOT": "pop20"}, axis=1)
    pop20_ch = fill_raster_dataframe(population20)
    csv_to_tiff(pop20_ch, attribute="pop20", path=r"data/independent_variable/processed/raw/pop20_ch.tif")
    pop20 = pop20_ch[
        (pop20_ch["E_COORD"] >= limits[0]) & (pop20_ch["E_COORD"] <= limits[2] - 100) & (pop20_ch["N_COORD"] >= limits[1]) & (
                    pop20_ch["N_COORD"] <= limits[3] - 100)]
    csv_to_tiff(pop20, attribute="pop20", path=r"data/independent_variable/processed/raw/pop20.tif")

    # Store the restructured dataset as csv file
    #population20.to_csv(r"data/temp/pop_filtered.csv")
    #betriebzaehlung20.to_csv(r"data/temp/empl_filtered.csv")
    return




def fill_raster_dataframe(df, rastersize=100):
    # Determine the minX, maxX, minY, maxY for consistent coverage
    try:
        df = df.rename(columns={"E_KOORD" : "E_COORD"})
        df = df.rename(columns={"N_KOORD" : "N_COORD"})
    except:
        print("")
    minX, maxX = df['E_COORD'].min(), df['E_COORD'].max()
    minX, maxX = round(math.floor(minX), -2), round(math.ceil(maxX), -2)
    minY, maxY = df['N_COORD'].min(), df['N_COORD'].max()
    minY, maxY = round(math.floor(minY), -2), round(math.ceil(maxY), -2)

    # Create a regular grid within the specified bounds
    x_grid = np.arange(minX, maxX, rastersize)
    y_grid = np.arange(minY, maxY, rastersize)

    # Create a new DataFrame with all combinations of X and Y
    new_x, new_y = np.meshgrid(x_grid, y_grid)
    new_data = pd.DataFrame({'E_COORD': new_x.ravel(), 'N_COORD': new_y.ravel()})

    # Merge the new DataFrame with the existing data and fill missing values with NaN
    merged_data = pd.merge(new_data, df, on=['E_COORD', 'N_COORD'], how='left')

    return merged_data


def csv_to_tiff(data_table, attribute, path, rastersize = 100):
    # Define the geospatial attributes
    crs_value = "epsg:2056"  # Define your desired CRS
    width = len(data_table['E_COORD'].unique())  # Match width to the number of unique X coordinates
    height = len(data_table['N_COORD'].unique())  # Match height to the number of unique Y coordinates

    x_min = min(data_table["E_COORD"])
    x_min = round(math.floor(x_min), -2)
    x_max = max(data_table["E_COORD"])
    x_max = round(math.ceil(x_max), -2)
    y_min = min(data_table["N_COORD"])
    y_min = round(math.floor(y_min), -2)
    y_max = max(data_table["N_COORD"])
    y_max = round(math.ceil(y_max), -2)

    #width = int((x_max - x_min) / rastersize)
    #height = int((y_max - y_min) / rastersize)
    #width = int(width)
    #height = int(height)
    transform = from_origin(x_min, y_max+100, rastersize, rastersize)

    #print(data_table[attribute].values.shape)
    #print(width, "   -   ", height, "   -   ", width*height)
    #sorted_df = df.sort_values(by=['Age', 'Salary'], ascending=[True, False])
    data_table_sorted = data_table.sort_values(by=["N_COORD", "E_COORD"], ascending=[False, True])
    # Create the GeoTIFF file
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=1,
                       dtype=data_table_sorted[attribute].dtype, crs=crs.CRS.from_string(crs_value), transform=transform) as dst:
        dst.write(data_table_sorted[attribute].values.reshape(height, width), 1)

    return


def import_locations():
    """
    This functions converts a csv files of location with coordinate to a geopandas DataFrame
    :return: GeoPandas DataFrame containing the locations as points
    """
    # Read csv file into pandas DataFrame
    df_cities = pd.read_csv(r"data/manually_gathered_data/City_map.csv", sep=";")

    # Convert single values into coordinates of geopandas DataFrame and initialize the coordinate reference system
    gdf_cities = gpd.GeoDataFrame(df_cities, geometry=gpd.points_from_xy(df_cities["x"], df_cities["y"]),
                                  crs="epsg:2056")

    gdf_cities.crs = "epsg:2056"
    gdf_cities.to_file('data/manually_gathered_data/cities.shp')
    return

def get_lake_data():
    gdf = gpd.read_file(r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp")
    gdf = gdf[gdf["GEWAESSERN"].isin(["Zürichsee", "Greifensee", "Pfäffikersee"])]
    # Set the CRS on the object first, then save without the crs argument
    gdf.crs = "epsg:2056"
    out = 'data/landuse_landcover/processed/lake_data_zh.gpkg'
    if os.path.exists(out):
        os.remove(out)
    gdf.to_file(out)
    return


def polygon_from_points(bounds=None, e_min=None, e_max=None, n_min=None, n_max=None, margin=0):
    """
    This function returns a square as polygon
    :param bounds: all limits of a polygon given as one element
    :param e_min: single limit values for polygon (same for e_max, n_min, n_max)
    :param margin: define if polygon should be bigger than the limits feede in
    :return:
    """
    if isinstance(bounds, np.ndarray):
        e_min, n_min, e_max, n_max = bounds
    if e_min is not None and e_max is not None and n_min is not None and n_max is not None:
        print("")
    else:
        print("No suitable coords for polygon")

    return Polygon([(e_min - margin, n_min - margin), (e_max + margin, n_min - margin), (e_max + margin, n_max + margin),
                 (e_min - margin, n_max + margin)])






def _connect_dead_ends_to_nodes(edges_gdf, threshold=50.0):
    """
    For each remaining dead-end endpoint, find the nearest other endpoint
    (any node) within `threshold` metres that it is NOT already directly
    connected to.  Add a short straight connector edge between them.

    This handles the case where a dead-end stub nearly reaches an existing
    junction node but falls just outside the merge tolerance.
    """
    from scipy.spatial import cKDTree

    # Count degrees and collect all unique endpoints
    ep_count = {}
    for geom in edges_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        for xy in [(geom.coords[0][0], geom.coords[0][1]),
                   (geom.coords[-1][0], geom.coords[-1][1])]:
            ep_count[xy] = ep_count.get(xy, 0) + 1

    dead_ends = [xy for xy, cnt in ep_count.items() if cnt == 1]
    all_nodes = list(ep_count.keys())
    if not dead_ends or len(all_nodes) < 2:
        return edges_gdf

    # Build direct-connection lookup: which node pairs already share an edge?
    connected_pairs = set()
    for geom in edges_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        s = (geom.coords[0][0],  geom.coords[0][1])
        e = (geom.coords[-1][0], geom.coords[-1][1])
        connected_pairs.add((s, e))
        connected_pairs.add((e, s))

    arr  = np.array(all_nodes)
    tree = cKDTree(arr)
    new_rows  = []
    connected = set()   # dead ends already handled this pass

    template = edges_gdf.iloc[0].copy()   # attribute template for stubs

    for dead_xy in dead_ends:
        if dead_xy in connected:
            continue
        dists, idxs = tree.query(np.array(dead_xy), k=min(10, len(all_nodes)))
        dists = np.atleast_1d(dists)
        idxs  = np.atleast_1d(idxs)
        for dist, idx in zip(dists, idxs):
            if dist < 0.1:
                continue   # same point
            if dist > threshold:
                break
            target_xy = tuple(arr[idx])
            if (dead_xy, target_xy) in connected_pairs:
                continue
            # Add connector
            stub = template.copy()
            stub['geometry']         = LineString([dead_xy, target_xy])
            stub['length_m']         = stub['geometry'].length
            stub['is_connector']     = 1
            stub['is_development']   = 0
            stub['is_schwachstelle'] = 0
            stub['ROUTENTYP']        = 'Connector'
            new_rows.append(stub)
            connected_pairs.add((dead_xy, target_xy))
            connected_pairs.add((target_xy, dead_xy))
            connected.add(dead_xy)
            break   # one connection per dead end is enough

    if not new_rows:
        print(f"  Dead-end→node connect: 0 stubs added (threshold={threshold} m)")
        return edges_gdf

    additions = gpd.GeoDataFrame(new_rows, crs=edges_gdf.crs)
    result    = pd.concat([edges_gdf, additions], ignore_index=True)
    print(f"  Dead-end→node connect: {len(new_rows)} stub(s) added (threshold={threshold} m)")
    return result


def _snap_dead_ends_to_edges(edges_gdf, threshold=30.0):
    """
    For every degree-1 (dead-end) node, find the nearest edge whose interior
    is within `threshold` metres.  Project the dead-end onto that edge,
    split the edge at the projected point, and add a short connector stub
    from the dead-end to the projected point.

    This connects dangling stubs that are close to—but not touching—a
    neighbouring route, without requiring exact coordinate overlap.
    """
    # Count endpoint degrees
    ep_count = {}
    ep_to_edges = {}   # (x,y) → list of row indices whose endpoint is here
    for idx, row in edges_gdf.iterrows():
        for pt_xy in [(row.geometry.coords[0][0], row.geometry.coords[0][1]),
                      (row.geometry.coords[-1][0], row.geometry.coords[-1][1])]:
            ep_count[pt_xy] = ep_count.get(pt_xy, 0) + 1
            ep_to_edges.setdefault(pt_xy, []).append(idx)

    dead_end_pts = {xy for xy, cnt in ep_count.items() if cnt == 1}
    if not dead_end_pts:
        print("  Dead-end snap: no dead-end nodes found")
        return edges_gdf

    # STRtree over edges for fast proximity lookup
    edge_list  = list(edges_gdf.itertuples())
    edge_geoms = [row.geometry for row in edge_list]
    edge_tree  = STRtree(edge_geoms)

    new_rows  = []
    drop_idx  = set()
    n_snapped = 0

    for dead_xy in dead_end_pts:
        dead_pt   = Point(dead_xy)
        # Edges that already touch this dead end (don't snap to own edge)
        own_edges = set(ep_to_edges.get(dead_xy, []))

        # Candidate edges within threshold
        cands = edge_tree.query(dead_pt.buffer(threshold))
        best_dist  = threshold
        best_idx   = None
        best_proj  = None

        for ci in cands:
            orig_idx = edge_list[ci].Index
            if orig_idx in own_edges or orig_idx in drop_idx:
                continue
            geom = edge_geoms[ci]
            dist = geom.distance(dead_pt)
            if dist < best_dist:
                # Projected point must be in the interior (not at an endpoint)
                proj_frac = geom.project(dead_pt, normalized=True)
                if 0.01 < proj_frac < 0.99:
                    best_dist = dist
                    best_idx  = orig_idx
                    best_proj = geom.interpolate(proj_frac, normalized=True)

        if best_idx is None or best_proj is None:
            continue

        # Split the target edge at the projected point
        target_row  = edges_gdf.loc[best_idx]
        target_geom = target_row.geometry
        try:
            snapped_geom = snap(target_geom, best_proj, best_dist + 0.1)
            parts = list(split(snapped_geom, best_proj).geoms)
        except Exception:
            continue
        if len(parts) < 2:
            continue

        drop_idx.add(best_idx)
        for part in parts:
            if part.length > 0.1:
                r = target_row.copy()
                r['geometry'] = part
                r['length_m'] = part.length
                new_rows.append(r)

        # Connector stub: dead-end → projected point
        if best_dist > 0.1:
            stub_geom = LineString([dead_xy, (best_proj.x, best_proj.y)])
            stub = target_row.copy()
            stub['geometry']         = stub_geom
            stub['length_m']         = stub_geom.length
            stub['is_connector']     = 1
            stub['is_development']   = 0
            stub['is_schwachstelle'] = 0
            stub['ROUTENTYP']        = 'Connector'
            new_rows.append(stub)

        n_snapped += 1

    if n_snapped == 0:
        print(f"  Dead-end snap: 0 dead-ends connected (threshold={threshold} m)")
        return edges_gdf

    # Rebuild: keep all rows not dropped, add new split/stub rows
    kept = edges_gdf[~edges_gdf.index.isin(drop_idx)]
    additions = gpd.GeoDataFrame(new_rows, crs=edges_gdf.crs)
    result = pd.concat([kept, additions], ignore_index=True)
    result = result[result.geometry.length > 0.1].reset_index(drop=True)
    print(f"  Dead-end snap: {n_snapped} dead-end(s) connected to nearest edge "
          f"(threshold={threshold} m) → {len(edges_gdf)} → {len(result)} edges")
    return result


def _split_edges_at_nodes(edges_gdf, snap_tol=1.0):
    """
    Find nodes that lie ON the interior of an existing edge (within snap_tol metres)
    but are not connected to it, and split the edge at that point.

    This catches the case where edge A runs from node 1 → node 2 passing
    directly through the location of node 3, but the graph has no edge
    from 1→3 or 3→2.  After splitting, node 3 is a proper junction.
    """
    # Collect all unique endpoints — these are the nodes we want to test
    endpoint_set = set()
    for geom in edges_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        endpoint_set.add((coords[0][0],  coords[0][1]))
        endpoint_set.add((coords[-1][0], coords[-1][1]))

    ep_points = [Point(x, y) for x, y in endpoint_set]
    if not ep_points:
        return edges_gdf

    # STRtree over endpoints for fast per-edge lookup
    ep_tree = STRtree(ep_points)

    new_rows   = []
    drop_idx   = set()
    n_splits   = 0

    for idx, edge in edges_gdf.iterrows():
        geom = edge.geometry
        if geom is None or geom.is_empty or geom.length < 0.1:
            new_rows.append(edge)
            continue

        # Find endpoints near this edge (within snap_tol of the LINE, not just the bbox)
        candidate_idx = ep_tree.query(geom.buffer(snap_tol))
        split_pts = []
        for ci in candidate_idx:
            pt = ep_points[ci]
            # Skip the edge's own endpoints
            if pt.distance(Point(geom.coords[0])) < snap_tol:
                continue
            if pt.distance(Point(geom.coords[-1])) < snap_tol:
                continue
            # Only keep points that actually lie on the edge interior
            if geom.distance(pt) <= snap_tol:
                split_pts.append(pt)

        if not split_pts:
            new_rows.append(edge)
            continue

        # Split the edge at all matching interior points
        try:
            working = snap(geom, MultiPoint(split_pts), snap_tol)
            result  = split(working, MultiPoint(split_pts))
            parts   = list(result.geoms)
        except Exception:
            new_rows.append(edge)
            continue

        if len(parts) <= 1:
            new_rows.append(edge)
            continue

        drop_idx.add(idx)
        for part in parts:
            if part.length > 0.1:
                row = edge.copy()
                row['geometry'] = part
                row['length_m'] = part.length
                new_rows.append(row)
        n_splits += 1

    if n_splits == 0:
        print(f"  Node-on-edge split: no interior nodes found (tol={snap_tol} m)")
        return edges_gdf

    result_gdf = gpd.GeoDataFrame(new_rows, crs=edges_gdf.crs).reset_index(drop=True)
    print(f"  Node-on-edge split: {n_splits} edge(s) split → "
          f"{len(edges_gdf)} → {len(result_gdf)} edges (tol={snap_tol} m)")
    return result_gdf


def _connect_components(nodes_gdf, edges_gdf, coord_round=2):
    """
    Find all disconnected components in the network and connect them by adding
    a synthetic straight-line edge between the closest pair of nodes from each
    component pair.  Repeats until the network is fully connected.

    Synthetic edges are marked with is_connector=1 so they can be handled
    differently during scoring (never scored, always present for routing).
    """
    import networkx as nx
    from scipy.spatial import cKDTree

    # Build undirected graph from edge start/end node IDs
    G = nx.Graph()
    G.add_nodes_from(nodes_gdf['ID_point'].tolist())
    for _, e in edges_gdf.iterrows():
        G.add_edge(int(e['start']), int(e['end']))

    components = list(nx.connected_components(G))
    if len(components) == 1:
        print(f"  Component connect: already fully connected ({len(nodes_gdf)} nodes)")
        return nodes_gdf, edges_gdf

    print(f"  Component connect: {len(components)} disconnected components → bridging …")

    # Index: node ID → (x, y)
    id_to_xy = {int(row['ID_point']): (row.geometry.x, row.geometry.y)
                for _, row in nodes_gdf.iterrows()}

    new_edges = []
    # Iteratively bridge the two closest components until fully connected
    while True:
        components = list(nx.connected_components(G))
        if len(components) == 1:
            break

        best_dist  = float('inf')
        best_pair  = None   # (node_id_a, node_id_b)

        # Build KDTree per component, query against all others
        comp_pts   = [np.array([id_to_xy[n] for n in c if n in id_to_xy]) for c in components]
        comp_ids   = [list(c) for c in components]

        for i in range(len(components)):
            if len(comp_pts[i]) == 0:
                continue
            tree_i = cKDTree(comp_pts[i])
            for j in range(i + 1, len(components)):
                if len(comp_pts[j]) == 0:
                    continue
                dists, idxs = tree_i.query(comp_pts[j])
                k = np.argmin(dists)
                if dists[k] < best_dist:
                    best_dist  = dists[k]
                    best_pair  = (comp_ids[i][idxs[k]], comp_ids[j][k])

        if best_pair is None:
            break

        na, nb = best_pair
        xa, ya = id_to_xy[na]
        xb, yb = id_to_xy[nb]
        geom = LineString([(xa, ya), (xb, yb)])

        # Build synthetic row inheriting columns from edges_gdf, with safe defaults
        row = {col: None for col in edges_gdf.columns}
        row['geometry']        = geom
        row['length_m']        = geom.length
        row['start']           = na
        row['end']             = nb
        row['ID_edge']         = int(edges_gdf['ID_edge'].max()) + len(new_edges) + 1
        row['is_connector']    = 1
        row['is_development']  = 0
        row['is_schwachstelle']= 0
        row['ROUTENTYP']       = 'Connector'
        new_edges.append(row)
        G.add_edge(na, nb)
        print(f"    Bridge added: node {na} ↔ {nb}  ({best_dist:.0f} m)")

    if new_edges:
        additions  = gpd.GeoDataFrame(new_edges, crs=edges_gdf.crs)
        edges_gdf  = pd.concat([edges_gdf, additions], ignore_index=True)

        # Recompute node flags from updated edge set
        def _xy(coord):
            return (round(coord[0], coord_round), round(coord[1], coord_round))  # noqa: B023

        ep_counts = Counter()
        for geom in edges_gdf.geometry:
            ep_counts[_xy(geom.coords[0])]  += 1
            ep_counts[_xy(geom.coords[-1])] += 1

        nodes_gdf['is_intersection']  = nodes_gdf.geometry.apply(
            lambda g: int(ep_counts[_xy(g.coords[0])] >= 3)).astype(np.int8)
        nodes_gdf['is_through_point'] = nodes_gdf.geometry.apply(
            lambda g: int(ep_counts[_xy(g.coords[0])] == 2)).astype(np.int8)
        nodes_gdf['is_endpoint']      = nodes_gdf.geometry.apply(
            lambda g: int(ep_counts[_xy(g.coords[0])] == 1)).astype(np.int8)

        # Persist updated files
        os.makedirs('data/Network/processed', exist_ok=True)
        lc = Counter(c.lower() for c in edges_gdf.columns)
        drop_upper = [c for c in edges_gdf.columns if c != c.lower() and lc[c.lower()] > 1]
        for col in drop_upper + ['visual_geom']:
            if col in edges_gdf.columns:
                edges_gdf = edges_gdf.drop(columns=[col])
        nodes_gdf.to_file('data/Network/processed/points.gpkg',  driver='GPKG')
        edges_gdf.to_file('data/Network/processed/edges.gpkg',   driver='GPKG')

        print(f"  Component connect: {len(new_edges)} bridge(s) added → network now fully connected")

    return nodes_gdf, edges_gdf


def reformat_network():
    print("\nreformat_network: start\n")

    MERGE_TOL   = 15.0  # m — nodes closer than this are merged into one
    COORD_ROUND = 2     # decimal places for endpoint rounding (0.01 mm)

    # ── Step 1: Load edges and nodes ─────────────────────────────────────────
    edges_path = 'data/Network/processed/edges.gpkg'
    nodes_path = 'data/Network/processed/nodes.gpkg'

    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"Missing {edges_path} — run import_network_GIS_ALLTAG() first")

    edges_gdf = gpd.read_file(edges_path)
    if edges_gdf.crs.to_epsg() != 2056:
        edges_gdf = edges_gdf.to_crs("EPSG:2056")

    if os.path.exists(nodes_path):
        nodes_raw = gpd.read_file(nodes_path)
        if nodes_raw.crs.to_epsg() != 2056:
            nodes_raw = nodes_raw.to_crs("EPSG:2056")
    else:
        nodes_raw = None

    print(f"  Loaded {len(edges_gdf)} edges")

    # ── Step 2: Merge nearby nodes ────────────────────────────────────────────
    # Build a KDTree over all unique endpoints.  Any pair of endpoints within
    # MERGE_TOL metres is clustered (Union-Find); the cluster centroid becomes
    # the single shared node.  Edge start/end coordinates are remapped so every
    # edge in the cluster connects to exactly one point.
    from scipy.spatial import cKDTree

    def _collect_endpoints(gdf):
        pts = set()
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            coords = list(geom.coords)
            pts.add((coords[0][0],  coords[0][1]))
            pts.add((coords[-1][0], coords[-1][1]))
        return list(pts)

    pt_list = _collect_endpoints(edges_gdf)
    arr     = np.array(pt_list)
    tree    = cKDTree(arr)
    pairs   = tree.query_pairs(MERGE_TOL)

    parent = list(range(len(pt_list)))
    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in pairs:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj

    clusters = {}
    for i in range(len(pt_list)):
        clusters.setdefault(_find(i), []).append(i)

    remap = {}
    for members in clusters.values():
        pts_m = arr[members]
        cx = round(pts_m[:, 0].mean() / 5) * 5
        cy = round(pts_m[:, 1].mean() / 5) * 5
        for m in members:
            remap[pt_list[m]] = (cx, cy)

    def _remap_geom(geom):
        if geom is None or geom.is_empty:
            return None
        coords  = list(geom.coords)
        s = remap.get((coords[0][0],  coords[0][1]),  (coords[0][0],  coords[0][1]))
        e = remap.get((coords[-1][0], coords[-1][1]), (coords[-1][0], coords[-1][1]))
        if s == e:
            return None
        interior = [(x, y) for x, y in coords[1:-1]]
        return LineString([s] + interior + [e])

    edges_gdf = edges_gdf.copy()
    edges_gdf['geometry'] = edges_gdf.geometry.apply(_remap_geom)
    edges_gdf = edges_gdf[edges_gdf.geometry.notna()].reset_index(drop=True)

    n_remapped = sum(1 for o, r in remap.items() if o != r)
    print(f"  Node merge: {n_remapped} endpoint(s) remapped into {len(clusters)} clusters "
          f"(tol={MERGE_TOL} m) → {len(edges_gdf)} edges remain")

    # ── Step 3: No additional splitting needed ────────────────────────────────
    # import_network_GIS_ALLTAG already uses momepy.gdf_to_nx (primal approach)
    # which nodes all LineString crossings into shared endpoints.  Re-splitting
    # here is redundant and causes a combinatorial explosion when the buffer
    # tolerance pulls in junction points from neighbouring edges.
    edges_split = edges_gdf.copy()
    print(f"  Edges after node merge: {len(edges_split)}")

    # ── Step 3b: Split edges where a node lies on an edge interior ────────────
    edges_split = _split_edges_at_nodes(edges_split, snap_tol=1.0)

    # ── Step 3c: Snap dead-end nodes to nearby edge interiors ─────────────────
    # Handles the case where a dead-end is 5–30 m from a parallel route —
    # projects it onto the nearest edge, splits there, and adds a short stub.
    edges_split = _snap_dead_ends_to_edges(edges_split, threshold=30.0)

    # ── Step 3d: Connect dead-ends to nearest existing node ───────────────────
    # Handles the case where a dead-end stub nearly reaches an existing junction
    # but the nearest point is at an edge endpoint (proj_frac ≈ 0 or 1), which
    # step 3c deliberately skips to avoid degenerate splits.
    edges_split = _connect_dead_ends_to_nodes(edges_split, threshold=50.0)

    def _xy(coord):
        return (round(coord[0], COORD_ROUND), round(coord[1], COORD_ROUND))

    # ── Step 4: Topology cleanup ──────────────────────────────────────────────
    # a) Drop zero/near-zero length edges
    edges_split = edges_split[edges_split.geometry.length > 0.1].reset_index(drop=True)

    # b) Assign fresh sequential edge IDs
    edges_split['ID_edge'] = range(len(edges_split))
    edges_split['length_m'] = edges_split.geometry.length

    # c) Build deduplicated node table from actual edge endpoints
    ep_counts   = Counter()
    coord_to_id = {}
    node_rows   = []

    for geom in edges_split.geometry:
        ep_counts[_xy(geom.coords[0])]  += 1
        ep_counts[_xy(geom.coords[-1])] += 1

    def _get_node(coord):
        key = _xy(coord)
        if key not in coord_to_id:
            nid = len(coord_to_id)
            coord_to_id[key] = nid
            node_rows.append({'ID_point': nid, 'geometry': Point(key)})
        return coord_to_id[key]

    edges_split['start'] = [_get_node(g.coords[0])  for g in edges_split.geometry]
    edges_split['end']   = [_get_node(g.coords[-1]) for g in edges_split.geometry]

    nodes_gdf = gpd.GeoDataFrame(node_rows, crs="EPSG:2056")

    # d) Flag node types from endpoint degree
    nodes_gdf['is_intersection'] = nodes_gdf.geometry.apply(
        lambda g: int(ep_counts[_xy(g.coords[0])] >= 3)).astype(np.int8)
    nodes_gdf['is_through_point'] = nodes_gdf.geometry.apply(
        lambda g: int(ep_counts[_xy(g.coords[0])] == 2)).astype(np.int8)
    nodes_gdf['is_endpoint'] = nodes_gdf.geometry.apply(
        lambda g: int(ep_counts[_xy(g.coords[0])] == 1)).astype(np.int8)


    # e) Save — drop columns that would cause GPKG case-collision or are too large
    lc = Counter(c.lower() for c in edges_split.columns)
    drop_upper = [c for c in edges_split.columns if c != c.lower() and lc[c.lower()] > 1]
    for col in drop_upper + ['visual_geom']:
        if col in edges_split.columns:
            edges_split = edges_split.drop(columns=[col])

    os.makedirs('data/Network/processed', exist_ok=True)
    nodes_gdf.to_file('data/Network/processed/points.gpkg',  driver='GPKG')
    edges_split.to_file('data/Network/processed/edges.gpkg', driver='GPKG')
    nodes_gdf.drop(columns='geometry').to_csv('data/Network/processed/points_export.csv', index=False)
    edges_split.drop(columns='geometry').to_csv('data/Network/processed/edges_export.csv', index=False)

    print(f"\n  → {len(edges_split)} edges, {len(nodes_gdf)} nodes")
    print(f"     {nodes_gdf['is_intersection'].sum()} intersections  "
          f"{nodes_gdf['is_through_point'].sum()} through-nodes  "
          f"{nodes_gdf['is_endpoint'].sum()} dead-ends")
    if 'is_development' in edges_split.columns:
        print(f"     {edges_split['is_development'].sum()} Netzlücken  "
              f"{edges_split['is_schwachstelle'].sum()} Schwachstellen")
    # ── Step 5: Connect disconnected components ───────────────────────────────
    nodes_gdf, edges_split = _connect_components(nodes_gdf, edges_split, COORD_ROUND)

    print("\nreformat_network: end\n")

    return nodes_gdf, edges_split



def plot_network_classified(nodes_gdf, edges_gdf, figsize=(14, 10)):
    fig, ax = plt.subplots(figsize=figsize)

    # --- Edges ---
    edges_gdf.plot(ax=ax, color='steelblue', linewidth=0.8, alpha=0.6, zorder=1)

    has_endpoint     = 'is_endpoint'     in nodes_gdf.columns
    has_intersection = 'is_intersection' in nodes_gdf.columns

    # --- Intersections (degree >= 3) ---
    if has_intersection:
        intersections = nodes_gdf[nodes_gdf['is_intersection'] == 1]
        non_inter     = nodes_gdf[nodes_gdf['is_intersection'] == 0]
    else:
        intersections = nodes_gdf.iloc[0:0]
        non_inter     = nodes_gdf

    # --- Dead ends (degree == 1) ---
    if has_endpoint:
        endpoints = non_inter[non_inter['is_endpoint'] == 1]
        regular   = non_inter[non_inter['is_endpoint'] == 0]
    else:
        endpoints = non_inter.iloc[0:0]
        regular   = non_inter

    if len(regular):
        ax.scatter(regular.geometry.x, regular.geometry.y,
                   s=8, color='gray', alpha=0.5, zorder=2, label=f'Through node ({len(regular)})')
    if len(intersections):
        ax.scatter(intersections.geometry.x, intersections.geometry.y,
                   s=30, color='orange', alpha=0.85, zorder=3, label=f'Intersection ({len(intersections)})')
    if len(endpoints):
        ax.scatter(endpoints.geometry.x, endpoints.geometry.y,
                   s=20, color='red', alpha=0.85, zorder=4, label=f'Dead end ({len(endpoints)})')

    ax.set_title('Cycling Network — Node Classification', fontsize=14)
    ax.set_xlabel('Easting (EPSG:2056)')
    ax.set_ylabel('Northing (EPSG:2056)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('data/Network/processed/network_plot.png', dpi=150)
    plt.show()
    print("Plot saved → data/Network/processed/network_plot.png")

def plot_edge_attributes(edges=None):
    if edges is None:
        edges = gpd.read_file('data/Network/processed/edges.gpkg')

    edges = edges.copy()

    # ── Derive ffs and tt_min from ROUTENTYP if not already present ───────────
    _FFS = {
        'Velobahn':                       20,
        'Veloschnellroute':               20,
        'Hauptverbindung':                18,
        'Nebenverbindung':                16,
        'Zusätzliche Freizeitverbindung': 18,
    }
    if 'ffs' not in edges.columns:
        edges['ffs'] = edges['ROUTENTYP'].map(lambda rt: _FFS.get(rt, 20))
    if 'tt_min' not in edges.columns:
        edges['tt_min'] = (edges['length_m'] / 1000) / edges['ffs'] * 60

    # ── Derive avg_incline_pct — batch all sample points in one raster read ───
    if 'avg_incline_pct' not in edges.columns:
        _elev_path = 'data/elevation_model/elevation.tif'
        if os.path.exists(_elev_path):
            N_SAMPLES = 10
            with rasterio.open(_elev_path) as _src:
                _elev_data      = _src.read(1).astype(float)
                _elev_transform = _src.transform
                _elev_nodata    = _src.nodata
            if _elev_nodata is not None:
                _elev_data[_elev_data == _elev_nodata] = np.nan
            h, w = _elev_data.shape

            inclines = []
            for geom in edges.geometry:
                if geom is None or geom.length == 0:
                    inclines.append(np.nan)
                    continue
                pts  = [geom.interpolate(f, normalized=True) for f in np.linspace(0, 1, N_SAMPLES)]
                rows, cols = rasterio.transform.rowcol(
                    _elev_transform, [p.x for p in pts], [p.y for p in pts])
                rows = np.clip(rows, 0, h - 1)
                cols = np.clip(cols, 0, w - 1)
                elevs = _elev_data[rows, cols]          # vectorised array index — no Python loop
                valid = elevs[~np.isnan(elevs)]
                if len(valid) < 2:
                    inclines.append(np.nan)
                else:
                    seg_len = geom.length / (N_SAMPLES - 1)
                    inclines.append(float(np.abs(np.diff(valid)).mean() / seg_len * 100))
            edges['avg_incline_pct'] = inclines
        else:
            edges['avg_incline_pct'] = np.nan

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # --- 1. Free-flow speed ---
    ax = axes[0]
    speed_colors = {20: '#3498db', 18: '#9b59b6', 15: '#e74c3c'}
    for speed, color in speed_colors.items():
        subset = edges[edges['ffs'] == speed]
        if len(subset):
            subset.plot(ax=ax, color=color, linewidth=1.5, alpha=0.8,
                        label=f'{speed} km/h ({len(subset)})')
    ax.set_title('Free-flow Speed (km/h)', fontsize=12)
    ax.legend(fontsize=8)
    ax.set_aspect('equal')

    # --- 2. Average incline — single vectorised plot call with colour array ---
    ax = axes[1]
    incline_vals = edges['avg_incline_pct'].fillna(0).values
    norm = mcolors.Normalize(vmin=0, vmax=max(incline_vals.max(), 8))
    cmap = cm.RdYlGn_r
    colors = [cmap(norm(v)) for v in incline_vals]
    edges.plot(ax=ax, color=colors, linewidth=1.5)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Avg incline (%)', shrink=0.6)
    ax.set_title('Average Incline (%)', fontsize=12)
    ax.set_aspect('equal')

    # --- 3. Travel time — single vectorised plot call with colour array ---
    ax = axes[2]
    tt_vals = edges['tt_min'].values
    norm2 = mcolors.Normalize(vmin=tt_vals.min(), vmax=tt_vals.max())
    cmap2 = cm.Blues
    colors2 = [cmap2(norm2(v)) for v in tt_vals]
    edges.plot(ax=ax, color=colors2, linewidth=1.5)
    sm2 = cm.ScalarMappable(cmap=cmap2, norm=norm2)
    sm2.set_array([])
    plt.colorbar(sm2, ax=ax, label='Travel time (min)', shrink=0.6)
    ax.set_title('Travel Time (min)', fontsize=12)
    ax.set_aspect('equal')

    plt.suptitle('Edge Attributes — Cycling Network', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('data/Network/processed/edge_attributes_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved → data/Network/processed/edge_attributes_plot.png")

def network_in_corridor(polygon):
    print(f"network_in_corridor(): start")
    os.makedirs('data/Network/processed', exist_ok=True)

    # ------------------------------------------------------------------
    # 1. LOAD inputs from reformat_network() outputs
    # ------------------------------------------------------------------
    edges  = gpd.read_file('data/Network/processed/edges.gpkg')
    points = gpd.read_file('data/Network/processed/points.gpkg')

    if edges.crs is None:
        edges  = edges.set_crs("EPSG:2056")
    if points.crs is None:
        points = points.set_crs("EPSG:2056")

    if edges.crs.to_epsg() != 2056:
        edges = edges.to_crs("EPSG:2056")
    if points.crs.to_epsg() != 2056:
        points = points.to_crs("EPSG:2056")

    print(f"  Loaded {len(edges)} edges, {len(points)} nodes")
    print(f"  Node columns: {points.columns.tolist()}")

    poly_gdf  = gpd.GeoDataFrame({'geometry': [polygon]}, crs="EPSG:2056")
    poly_geom = polygon

    # ------------------------------------------------------------------
    # 2. NODES inside corridor
    # ------------------------------------------------------------------
    points_corridor = gpd.sjoin(points, poly_gdf, how='inner', predicate='within') \
                         .drop(columns=['index_right'], errors='ignore') \
                         .reset_index(drop=True)
    points_corridor.to_file('data/Network/processed/points_corridor.gpkg', driver='GPKG')
    points_corridor.drop(columns='geometry').to_csv(
        'data/Network/processed/points_corridor.csv', index=False
    )
    print(f"  Nodes in corridor → points_corridor.gpkg + .csv  ({len(points_corridor)} rows)")

    # ------------------------------------------------------------------
    # 3. ACCESS POINTS inside corridor — all corridor nodes are access points
    # ------------------------------------------------------------------
    access_points_corridor = points_corridor.copy().reset_index(drop=True)
    access_points_corridor['ID_access'] = range(len(access_points_corridor))
    access_points_corridor.to_file('data/Network/processed/access_points_corridor.gpkg', driver='GPKG')
    access_points_corridor.drop(columns='geometry').to_csv(
        'data/Network/processed/access_points_corridor.csv', index=False
    )
    print(f"  Access points in corridor → access_points_corridor.gpkg + .csv  ({len(access_points_corridor)} rows)")

    # ------------------------------------------------------------------
    # 4. FLAG edges with corridor membership
    # ------------------------------------------------------------------
    def one_endpoint_inside(geom, poly):
        return poly.contains(Point(geom.coords[0])) != poly.contains(Point(geom.coords[-1]))

    edges['within_corridor'] = edges.geometry.apply(lambda g: poly_geom.contains(g))
    edges['on_border']       = edges.geometry.apply(lambda g: one_endpoint_inside(g, poly_geom))

    edges_corridor = edges[edges['within_corridor']].copy().reset_index(drop=True)
    edges_border   = edges[edges['on_border']].copy().reset_index(drop=True)

    # ------------------------------------------------------------------
    # 5. FLAG nodes with corridor membership
    # ------------------------------------------------------------------
    points['within_corridor'] = points.geometry.apply(lambda g: poly_geom.contains(g))

    # on_corridor_border: node touches a border-crossing edge
    points_buf = points.copy()
    points_buf['geometry'] = points.buffer(1e-6)
    points_buf = points_buf.reset_index().rename(columns={'index': 'orig_idx'})

    border_join = gpd.sjoin(
        points_buf[['orig_idx', 'geometry']],
        edges_border[['geometry']].reset_index(drop=True),
        how='left',
        predicate='intersects'
    )
    on_border_flag = border_join.groupby('orig_idx')['index_right'].apply(
        lambda x: x.notnull().any()
    )
    points['on_corridor_border'] = on_border_flag.reindex(points.index, fill_value=False).values

    # ------------------------------------------------------------------
    # 8. SAVE enriched full datasets
    # ------------------------------------------------------------------
    points.to_file('data/Network/processed/points_with_attribute.gpkg', driver='GPKG')
    points.drop(columns='geometry').to_csv(
        'data/Network/processed/points_with_attribute.csv', index=False
    )
    edges.to_file('data/Network/processed/edges_with_attribute.gpkg', driver='GPKG')
    edges.drop(columns='geometry').to_csv(
        'data/Network/processed/edges_with_attribute.csv', index=False
    )
    print(f"  Full attributed nodes/edges saved (with within_corridor + on_corridor_border flags)")

    # ------------------------------------------------------------------
    # 9. SAVE corridor points with attributes
    # ------------------------------------------------------------------
    points_corridor_attribute = points[points['within_corridor']].copy().reset_index(drop=True)
    points_corridor_attribute.to_file(
        'data/Network/processed/points_corridor_attribute.gpkg', driver='GPKG'
    )
    points_corridor_attribute.drop(columns='geometry').to_csv(
        'data/Network/processed/points_corridor_attribute.csv', index=False
    )
    print(f"  Corridor nodes with attributes → points_corridor_attribute.gpkg  ({len(points_corridor_attribute)} rows)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n  Summary:")
    print(f"    {len(points_corridor)} / {len(points)} nodes inside corridor (all are access points)")
    for flag in ('is_intersection', 'is_through_point', 'is_endpoint'):
        if flag in points_corridor.columns:
            print(f"    {points_corridor[flag].sum()} {flag.replace('is_', '')} in corridor")
    print(f"    {edges['within_corridor'].sum()} / {len(edges)} edges inside corridor")
    print(f"    {edges['on_border'].sum()} edges crossing corridor border")

    print(f"network_in_corridor(): end")
    return points_corridor, edges_corridor, edges_border


def plot_corridor_network(polygon, points_corridor, edges_corridor, edges_border, points_full=None, edges_full=None):
    import numpy as np

    fig, ax = plt.subplots(figsize=(22, 16))

    # --- Corridor polygon background only (no full-canton context) ---
    poly_gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs="EPSG:2056")
    poly_gdf.plot(ax=ax, facecolor='#F7F9FC', edgecolor='none', zorder=0)
    poly_gdf.boundary.plot(ax=ax, color='#333333', linewidth=2.0, linestyle='--', zorder=5)

    # --- Edges inside corridor ---
    if len(edges_corridor):
        edges_corridor.plot(ax=ax, color='#2166AC', linewidth=1.8, alpha=0.85, zorder=3)

    # --- Border-crossing edges ---
    if len(edges_border):
        edges_border.plot(ax=ax, color='#D6604D', linewidth=2.0, linestyle=':', alpha=0.9, zorder=4)

    # --- Nodes: intersections / dead-ends / through-nodes ---
    id_col = 'ID_point' if 'ID_point' in points_corridor.columns else None

    def _label_nodes(subset, fontsize=6, color='#1A1A1A', offset=3):
        """Annotate each node with its ID_point value."""
        if id_col is None:
            return
        for _, row in subset.iterrows():
            ax.annotate(
                str(int(row[id_col])),
                xy=(row.geometry.x, row.geometry.y),
                xytext=(offset, offset),
                textcoords='offset points',
                fontsize=fontsize,
                color=color,
                fontweight='bold',
                zorder=9,
            )

    if 'is_intersection' in points_corridor.columns and 'is_endpoint' in points_corridor.columns:
        intersections = points_corridor[points_corridor['is_intersection'] == 1]
        endpoints     = points_corridor[points_corridor['is_endpoint']     == 1]
        through       = points_corridor[
            (points_corridor['is_intersection'] == 0) &
            (points_corridor['is_endpoint']     == 0)
        ]
        if len(through):
            ax.scatter(through.geometry.x, through.geometry.y,
                       s=18, color='#888888', alpha=0.6, zorder=6, linewidths=0)
            _label_nodes(through, fontsize=5, color='#555555')
        if len(intersections):
            ax.scatter(intersections.geometry.x, intersections.geometry.y,
                       s=60, color='#F4A800', alpha=0.95, zorder=7,
                       edgecolors='#5C3A00', linewidths=0.5)
            _label_nodes(intersections, fontsize=6, color='#3A2000')
        if len(endpoints):
            ax.scatter(endpoints.geometry.x, endpoints.geometry.y,
                       s=40, color='#D73027', alpha=0.95, zorder=7,
                       edgecolors='#7A0000', linewidths=0.5)
            _label_nodes(endpoints, fontsize=6, color='#7A0000')

        n_inter   = len(intersections)
        n_ends    = len(endpoints)
        n_through = len(through)
    else:
        points_corridor.plot(ax=ax, color='#2166AC', markersize=12, zorder=6)
        if id_col:
            _label_nodes(points_corridor, fontsize=6, color='#1A1A1A')
        n_inter = n_ends = n_through = 0

    # --- Tight view: clip axes to corridor bounding box + small margin ---
    xmin, ymin, xmax, ymax = polygon.bounds
    margin = max((xmax - xmin), (ymax - ymin)) * 0.03
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)

    # --- Reference grid: major lines every 1000 m, minor every 500 m (labelled in km) ---
    grid_major = 1000   # 1 km major
    grid_minor = 500    # 500 m minor

    x0 = int(np.floor((xmin - margin) / grid_major)) * grid_major
    x1 = int(np.ceil ((xmax + margin) / grid_major)) * grid_major
    y0 = int(np.floor((ymin - margin) / grid_major)) * grid_major
    y1 = int(np.ceil ((ymax + margin) / grid_major)) * grid_major

    for xg in np.arange(x0, x1 + 1, grid_minor):
        lw = 0.6 if xg % grid_major == 0 else 0.25
        alpha = 0.35 if xg % grid_major == 0 else 0.18
        ax.axvline(xg, color='#4A4A4A', linewidth=lw, alpha=alpha, zorder=1)
    for yg in np.arange(y0, y1 + 1, grid_minor):
        lw = 0.6 if yg % grid_major == 0 else 0.25
        alpha = 0.35 if yg % grid_major == 0 else 0.18
        ax.axhline(yg, color='#4A4A4A', linewidth=lw, alpha=alpha, zorder=1)

    # Axis tick labels at every km, formatted as integers
    xticks = np.arange(x0, x1 + 1, grid_major)
    yticks = np.arange(y0, y1 + 1, grid_major)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f'{int(x):,}' for x in xticks], fontsize=9, rotation=30, ha='right')
    ax.set_yticklabels([f'{int(y):,}' for y in yticks], fontsize=9)

    # --- Legend ---
    legend_elements = [
        Line2D([0], [0], color='#2166AC', linewidth=2.0,
               label=f'Edges in corridor ({len(edges_corridor)})'),
        Line2D([0], [0], color='#D6604D', linewidth=2.0, linestyle=':',
               label=f'Border-crossing edges ({len(edges_border)})'),
        Line2D([0], [0], color='#333333', linewidth=2.0, linestyle='--',
               label='Corridor boundary'),
        mpatches.Patch(facecolor='#F4A800', edgecolor='#5C3A00', label=f'Intersections ({n_inter})'),
        mpatches.Patch(facecolor='#D73027', edgecolor='#7A0000', label=f'Dead ends ({n_ends})'),
        mpatches.Patch(facecolor='#888888',                      label=f'Through-nodes ({n_through})'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=10, framealpha=0.92, edgecolor='#CCCCCC')

    n_pts = len(points_corridor)
    ax.set_title(
        f'Cycling Network — Corridor  '
        f'({len(edges_corridor)} edges · {n_pts} nodes · '
        f'{len(edges_border)} border-crossing)\n'
        f'{n_inter} intersections · {n_ends} dead-ends · {n_through} through-nodes',
        fontsize=14, fontweight='bold', pad=12,
    )
    ax.set_xlabel('Easting LV95 [m]', fontsize=11)
    ax.set_ylabel('Northing LV95 [m]', fontsize=11)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('data/Network/processed/corridor_plot.png', dpi=250, bbox_inches='tight')
    plt.show()
    print("Plot saved → data/Network/processed/corridor_plot.png")


def only_links_to_corridor():
    # 1. Load data
    all_links = gpd.read_file(r"data/Network/processed/new_links.gpkg")
    all_access_points = gpd.read_file(r"data/Network/processed/points_corridor_attribute.gpkg")

    # 2. Determine available columns to prevent KeyError
    # We always need ID_point for the join
    cols_to_use = ["ID_point"]

    if "cor_1" in all_access_points.columns:
        # Filter for corridor points if attribute exists
        access_corridor = all_access_points[all_access_points["cor_1"] == "1"].copy()
        cols_to_use.append("cor_1")
    else:
        print("Warning: 'cor_1' column not found. Skipping attribute filter and using all points.")
        access_corridor = all_access_points.copy()

    # 3. Join links to corridor points
    # We join all_links.ID_current to access_corridor.ID_point
    all_links["ID_current"] = all_links["ID_current"].astype(str)
    access_corridor["ID_point"] = access_corridor["ID_point"].astype(str)

    links_corridor = all_links.merge(
        right=access_corridor[cols_to_use],
        left_on="ID_current",
        right_on="ID_point"
    )

    print(f"Links connected to points within the corridor: {links_corridor.shape[0]} of {all_links.shape[0]}")

    # 4. Save processed links
    if "ID_current" in links_corridor.columns:
        links_corridor = links_corridor.drop(columns=["ID_current"])

    links_corridor.to_file(r"data/Network/processed/developments_to_corridor_attribute.gpkg")

    # 5. Process generated access points
    generated_points = gpd.read_file(r"data/Network/processed/generated_nodes.gpkg")

    # Ensure ID_new is also treated as string for the merge
    generated_points["ID_new"] = generated_points["ID_new"].astype(str)
    links_corridor["ID_new"] = links_corridor["ID_new"].astype(str)

    # Merge to filter only generated points that successfully found a link
    temp = generated_points.merge(right=links_corridor, on="ID_new")

    # geometry_x is the location of the NEW generated point
    generated_points_corridor = temp[["ID_new", "geometry_x", "ID_point"]].copy()

    # Rename geometry column to standard 'geometry'
    generated_points_corridor = generated_points_corridor.rename(columns={"geometry_x": "geometry"})

    # Convert back to a proper GeoDataFrame
    gdf_nodes_out = gpd.GeoDataFrame(
        generated_points_corridor,
        geometry="geometry",
        crs=generated_points.crs
    )

    gdf_nodes_out.to_file(r"data/Network/processed/generated_nodes_connecting_corridor.gpkg")
    print("Infrastructure generation step complete.")

def get_protected_area(limits):
    bln = gpd.read_file(r"data/landuse_landcover/Schutzzonen/BLN/N2017_Revision_landschaftnaturdenkmal_20170727_20221110.shp")
    wildkorridore = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Wildtierkorridore/Wildtierkorridore.gpkg")
    trockenweiden = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Trockenwiesen/TWW_LV95/trockenwiesenweiden.shp")
    trockenlandschaften = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Moorlandschaft/Moorlandschaft_LV95/moorlandschaft.shp")
    flachmoore = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Flachmoore/Flachmoor_LV95/flachmoor_20210701.shp")
    hochmoore = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Hochmoor/Hochmoor_LV95/hochmoor.shp")
    bundesinventar_auen = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Bundesinventar_auen/N2017_Revision_Auengebiete_20171101_20221122.shp")
    ramsar = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Ramsar/Ramsar_LV95/ra.shp")
    naturschutz = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Inventar_der_Natur-_und_Landsch...uberkommunaler_Bedeutung_-OGD/INV80_NATURSCHUTZOBJEKTE_F.shp")
    wald = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Waldareal_-OGD/WALD_WALDAREAL_F.shp")
    fruchtfolgeflaeche = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Fruchtfolgeflachen_-OGD/FFF_F.shp")

    # Cycling: BLN, Moores, Auen, Ramsar and Naturschutz remain fully protected (legally binding).
    # Wildtierkorridore removed from fully_protected: a narrow cycle path has a small footprint
    # and may be permissible — treat as partly protected instead.
    gdf_fully_protected = [
        bln,
        flachmoore,
        hochmoore,
        bundesinventar_auen,
        ramsar,
        naturschutz
    ]
    names_fully_protected = [
        'bln',
        'flachmoore',
        'hochmoore',
        'bundesinventar_auen',
        'ramsar',
        'naturschutz'
    ]

    # Cycling: Wald removed from partly_protected — cycle paths through forests are common and
    # legally permitted. Wildtierkorridore added here as a soft constraint.
    gdf_partly_protected = [
        wildkorridore,
        fruchtfolgeflaeche,
        trockenweiden,
        trockenlandschaften
    ]
    names_partly_protected = [
        "wildkorridore",
        "fruchtfolgeflaeche",
        "trockenweiden",
        "trockenlandschaften"
    ]

    multiple_shp_to_one(gdf_fully_protected, names_fully_protected, "fully_protected", limits)
    multiple_shp_to_one(gdf_partly_protected, names_partly_protected, "partly_protected", limits)

    return


def multiple_shp_to_one(gdf_list, names_list, path, limits):
    # Initialize an empty list to store the dissolved geometries
    dissolved_geometries = []

    for gdf in gdf_list:
        # Dissolve all features within the GeoDataFrame into a single geometry
        gdf = gdf.copy()
        gdf['geometry'] = gdf.geometry.apply(lambda g: make_valid(g) if g is not None else g)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)
        dissolved = gdf.dissolve()
        # Append the dissolved geometry to the list

        valid_geoms = dissolved.geometry.apply(make_valid)
        dissolved_geometries.append(valid_geoms.unary_union)

    # Now create a new DataFrame with the dissolved geometries
    # Use the 'unary_union' attribute to ensure that the geometry is merged into one
    combined_gdf = gpd.GeoDataFrame({'geometry': dissolved_geometries})

    combined_gdf["name"] = names_list
    combined_gdf.crs = "epsg:2056"

    combined_gdf.to_file(fr"data/landuse_landcover/Schutzzonen/{path}.gpkg", driver="GPKG")

    # Create a bounding box as a shapely object
    frame_box = box(limits[0], limits[1], limits[2], limits[3])

    # Clip the GeoDataFrame using the bounding box
    combined_gdf_frame = gpd.clip(combined_gdf, frame_box)
    #combined_gdf_frame.to_file(fr"data/landuse_landcover/Schutzzonen/{path}_frame.gpkg")
    combined_gdf_frame.to_file(fr"data/landuse_landcover/processed/{path}_frame.gpkg")


def all_protected_area_to_raster(suffix=""):
    # Load your shapefile with geopandas
    #shp_file = r"data/landuse_landcover/processed/fully_protected.gpkg"
    shp_file = r"data/landuse_landcover/Schutzzonen/fully_protected.gpkg" #correction by Arnor
    shapes = gpd.read_file(shp_file)

    # Load your raster file with rasterio
    tif_file = r"data/landuse_landcover/processed/protected_area.tif"

    try:
        # Load the CSV file with the coordinates
        csv_file = r"data/manually_gathered_data/cell_to_remove.csv"
        coords_df = pd.read_csv(csv_file, sep=";")
        print(coords_df.head().to_string())
    except:
        pass

    # Open the existing TIFF file
    with rasterio.open(tif_file) as src:
        meta = src.meta.copy()
        # Read the existing data
        existing_data = src.read(1)
        nodata_value = src.nodata or -9999
        meta.update(nodata=nodata_value)

        # Rasterize the shapes to the same dimension as the raster data
        burned = rasterize(
            [(shape, 1) for shape in shapes.geometry],
            out_shape=src.shape,
            transform=src.transform,
            fill=0,  # the default fill value
            all_touched=True  # mark all cells touched by polygons
        )

        # Merge the burned raster with the existing data
        # Where the burned data is 1, we update the existing data
        updated_data = np.where(burned == 1, 39, existing_data)

        try:
            # Process coordinates from the CSV file
            for _, row in coords_df.iterrows():
                row_x, row_y = row['x'], row['y']
                row_col, row_row = src.index(row_x, row_y)
                updated_data[row_col, row_row] = -9999
        except:
            print("No cell to remove")

        # Write the updated data to a new raster file
        with rasterio.open(fr'data/landuse_landcover/processed/zone_no_infra/protected_area_{suffix}.tif', 'w', **meta) as dst:
            dst.write(updated_data, 1)


def _write_zero_raster(path, limits, rastersize=100):
    """Write an all-zero GeoTIFF placeholder when no data exists for a corridor."""
    width  = max(1, int((limits[2] - limits[0]) / rastersize))
    height = max(1, int((limits[3] - limits[1]) / rastersize))
    transform = rasterio.transform.from_bounds(
        limits[0], limits[1], limits[2], limits[3], width, height
    )
    profile = dict(driver='GTiff', dtype='float64', width=width, height=height,
                   count=1, crs='epsg:2056', transform=transform)
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(np.zeros((height, width), dtype=np.float64), 1)
    print(f"  Written zero-placeholder raster → {path}")


def landuse(limits):
    # Read the CSV file into a Pandas DataFrame
    # Arealstatistik - 1985, 1997, 2009, 2018
    areal_stat = pd.read_csv(r'data/landuse_landcover/landcover/ag-b-00.03-37-area-csv.csv', sep=";")
    if limits:
        areal_stat = areal_stat[(areal_stat["E_COORD"] >= limits[0]) &
                                (areal_stat["E_COORD"] <= limits[2]) &
                                (areal_stat["N_COORD"] >= limits[1]) &
                                (areal_stat["N_COORD"] <= limits[3])]

    areal_stat = areal_stat[["E_COORD", "N_COORD", "AS18_27"]]

    # Cycling: much smaller footprint than a highway — fewer land-use categories are true barriers.
    # Removed: 4 (Parks/Grünanlagen), 5 (Strassen/Verkehr), 8 (Sportanlagen), 10 (Camping),
    #          17 (Obstgärten), 18 (Reben), 23 (Wald), 27 (Gewässer).
    # 4 removed: cycle paths through parks are common and desirable.
    # 5 removed: cycle paths almost always run alongside existing roads — blocking cat. 5 would
    #            prevent nearly all realistic cycle route generation.
    # Wald (23): cycle paths through forests are standard practice.
    # Gewässer (27): cycle bridges are cheap and small; handle via elevation model.
    # Kept: residential/industrial buildings, special buildings, airports, cemeteries.
    protected_categories = [1, 2, 3, 7, 9]

    protected_area = areal_stat[areal_stat["AS18_27"].isin(protected_categories)]

    out_path = r"data/landuse_landcover/processed/protected_area.tif"
    protected_area_full = fill_raster_dataframe(protected_area) if not protected_area.empty else pd.DataFrame()
    if protected_area_full.empty or protected_area_full["E_COORD"].dropna().empty:
        _write_zero_raster(out_path, limits)
        return

    protected_area_full["N_COORD"] = protected_area_full["N_COORD"] + 100
    csv_to_tiff(protected_area_full, attribute="AS18_27", path=out_path)
    # print(areal_stat.head(50).to_string())


def get_unproductive_area(limits):
    areal_stat = pd.read_csv(r'data/landuse_landcover/landcover/ag-b-00.03-37-area-csv.csv', sep=";")
    if limits:
        areal_stat = areal_stat[(areal_stat["E_COORD"] >= limits[0]) &
                                (areal_stat["E_COORD"] <= limits[2]) &
                                (areal_stat["N_COORD"] >= limits[1]) &
                                (areal_stat["N_COORD"] <= limits[3])]

    areal_stat = areal_stat[["E_COORD", "N_COORD", "AS18_27"]]
    #print(areal_stat.shape)
    # Cycling: only true water bodies are impassable. Forest (23), shrub forest (24), and
    # woody vegetation (25) are removed — cycle paths through these are common and permitted.
    unproductive_zones = [26, 27]
    unproductive_area = areal_stat[areal_stat["AS18_27"].isin(unproductive_zones)]

    out_path = r"data/landuse_landcover/processed/unproductive_area.tif"
    unproductive_area_full = fill_raster_dataframe(unproductive_area) if not unproductive_area.empty else pd.DataFrame()
    if unproductive_area_full.empty or unproductive_area_full["E_COORD"].dropna().empty:
        _write_zero_raster(out_path, limits)
        return

    # Correction of the reference of each raster cell from bottom left to top left
    unproductive_area_full["N_COORD"] = unproductive_area_full["N_COORD"] + 100
    csv_to_tiff(unproductive_area_full, attribute="AS18_27", path=out_path)

    # AS85_17 17 Klassen gemäss Standardnomenklatur der Arealstatistik 1979/85
    # AS85_4  4 Hauptbereiche gemäss Standardnomenklatur der Arealstatistik 1979/85
    # LU85_10 10 Klassen der Bodennutzung der Arealstatistik 1979/85
    # LU85_4  4 Hauptbereiche der Bodennutzung der Arealstatistik 1979/85





