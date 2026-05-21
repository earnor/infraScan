from operator import truediv
from turtledemo.chaos import plot

import momepy
import osmnx
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
import os
import pandas as pd
import ast
import networkx as nx
from typing import Optional, Any
from shapely.validation import make_valid
from OSM_network import build_network_from_shapefile, check_network_connectivity

# TODO: hardcoded os.chdir() — this file is imported by main.py which already
# sets the working directory.  Calling chdir() here again is redundant and
# will break if the module is imported from a different entry point.
# Remove this line and rely on main.py to set the working directory, or
# use pathlib.Path(__file__).parent for all relative paths in this module.
#os.chdir(r'/Users/ruki/PycharmProjects/infraScan/infraScanCycle')
os.chdir(r'/Users/ninablattler/PycharmProjects/infraScan/infraScanCycle')


def _merge_close_endpoints(df, tolerance=5.0):
    """
    Merge edge endpoints that are within `tolerance` metres of each other.

    After grid-snapping two endpoints may still land on adjacent 5 m cells
    (e.g. 2.4 m and 2.6 m from the same grid line → cells 0 m and 5 m apart).
    This function collects every unique snapped endpoint, uses a KDTree to find
    all pairs within `tolerance`, clusters them via Union-Find, and replaces
    each cluster with a single grid-snapped centroid coordinate.  Edges that
    become degenerate (start == end after merging) are dropped.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    # Collect all unique endpoints from the already-snapped geometries
    pt_set = set()
    for geom in df.geometry:
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        pt_set.add((coords[0][0],  coords[0][1]))
        pt_set.add((coords[-1][0], coords[-1][1]))

    if len(pt_set) < 2:
        return df

    pt_list = list(pt_set)
    arr     = np.array(pt_list)

    # Find all pairs within tolerance
    tree  = cKDTree(arr)
    pairs = tree.query_pairs(tolerance)

    # Union-Find: cluster nearby endpoints
    parent = list(range(len(pt_list)))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj

    # Build cluster → representative (grid-snapped centroid of all members)
    clusters = {}
    for i in range(len(pt_list)):
        clusters.setdefault(_find(i), []).append(i)

    remap = {}
    for members in clusters.values():
        pts = arr[members]
        cx = round(pts[:, 0].mean() / 5) * 5
        cy = round(pts[:, 1].mean() / 5) * 5
        for m in members:
            remap[pt_list[m]] = (cx, cy)

    merged = sum(1 for orig, rep in remap.items() if orig != rep)
    print(f"  _merge_close_endpoints: {merged} endpoint(s) remapped "
          f"({len(clusters)} clusters from {len(pt_list)} unique points, tol={tolerance} m)")

    def _apply(geom):
        if geom is None or geom.is_empty:
            return None
        coords  = list(geom.coords)
        s = remap.get((coords[0][0],  coords[0][1]),  (coords[0][0],  coords[0][1]))
        e = remap.get((coords[-1][0], coords[-1][1]), (coords[-1][0], coords[-1][1]))
        if s == e:
            return None  # degenerate after merge — drop
        interior = [(x, y) for x, y in coords[1:-1]]
        return LineString([s] + interior + [e])

    df = df.copy()
    df['geometry'] = df.geometry.apply(_apply)
    df = df[df.geometry.notna()].reset_index(drop=True)
    return df


def _connect_nearby_dead_ends(df, tolerance=20.0):
    """
    Add short synthetic edges between pairs of degree-1 (dead-end) endpoints
    that are within `tolerance` metres of each other but not yet connected.

    Works at the coordinate level (before momepy), so it runs after
    _merge_close_endpoints and before gdf_to_nx.  Synthetic edges inherit
    column values from the edge whose endpoint is closest to the new connector.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    # Count how many times each endpoint coordinate appears across all edges
    counts = {}
    for geom in df.geometry:
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        s = (coords[0][0],  coords[0][1])
        e = (coords[-1][0], coords[-1][1])
        counts[s] = counts.get(s, 0) + 1
        counts[e] = counts.get(e, 0) + 1

    dead_ends = [pt for pt, n in counts.items() if n == 1]
    if len(dead_ends) < 2:
        return df

    arr  = np.array(dead_ends)
    tree = cKDTree(arr)
    pairs = tree.query_pairs(tolerance)

    if not pairs:
        print(f"  _connect_nearby_dead_ends: 0 connections added (no pairs within {tolerance} m)")
        return df

    # Build index: endpoint → row index in df for attribute inheritance
    ep_to_row = {}
    for idx, geom in enumerate(df.geometry):
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        s = (coords[0][0],  coords[0][1])
        e = (coords[-1][0], coords[-1][1])
        ep_to_row[s] = idx
        ep_to_row[e] = idx

    # Greedy connect: sort by distance, skip if either endpoint is already
    # claimed by a new edge (keeps the graph clean — avoids multi-star clusters)
    pair_list = sorted(pairs, key=lambda p: np.linalg.norm(arr[p[0]] - arr[p[1]]))
    claimed   = set()
    new_rows  = []
    for i, j in pair_list:
        if i in claimed or j in claimed:
            continue
        claimed.add(i)
        claimed.add(j)
        pt_i = dead_ends[i]
        pt_j = dead_ends[j]
        # Inherit attributes from whichever row owns endpoint i
        src_row = df.iloc[ep_to_row.get(pt_i, ep_to_row.get(pt_j, 0))].copy()
        src_row['geometry'] = LineString([pt_i, pt_j])
        src_row['length_m'] = src_row['geometry'].length
        new_rows.append(src_row)

    if not new_rows:
        print(f"  _connect_nearby_dead_ends: 0 connections added (all pairs claimed)")
        return df

    additions = gpd.GeoDataFrame(new_rows, crs=df.crs).reset_index(drop=True)
    result    = pd.concat([df, additions], ignore_index=True)
    print(f"  _connect_nearby_dead_ends: {len(new_rows)} synthetic edge(s) added "
          f"(tol={tolerance} m, {len(dead_ends)} dead-ends found)")
    return result


def import_network_GIS_ALLTAG() -> Any:
    """
    Load the ALLTAG cantonal cycling network shapefile, attach Schwachstellen
    and Netzlücken flags, snap/merge endpoints to a 5 m grid, connect nearby
    dead ends, build a primal graph, and save edges + nodes to GPKG/CSV.

    Returns the edge GeoDataFrame (gdf_edges) with all attributes.
    """


    # ── Step 1: Load ALLTAG shapefile ─────────────────────────────────────────
    path_alltag = 'data/raw/ALLTAG/OGD_VELO_ALLTAG_NETZ_L_M.shp'
    print("\n[import_network_GIS_ALLTAG] Loading ALLTAG shapefile …")
    df = gpd.read_file(path_alltag, engine='pyogrio')

    if df.crs.to_epsg() != 2056:
        df = df.to_crs("EPSG:2056")

    # Rename truncated DBF column names to full descriptive names
    df = df.rename(columns={
        'VERBINDUNG': 'VERBINDUNGSNAME',
        'FAHRRICHTU': 'FAHRRICHTUNGSTYP',
        'PLANUNGSTY': 'PLANUNGSTYP',
    })

    # Keep only the columns we need; visual_geom stashes the original winding shape
    keep_cols = ['geometry', 'RW_KEY', 'RW_KEY_NR', 'KMMIN', 'KMMAX',
                 'VERBINDUNGSNAME', 'ZWECK', 'ROUTENTYP', 'FAHRRICHTUNGSTYP', 'PLANUNGSTYP', 'SHAPE_LEN']
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    df = df.rename(columns={'SHAPE_LEN': 'LAENGE_m'})

    df['visual_geom'] = df['geometry'].apply(lambda g: g.wkt if g is not None else None)
    print(f"  Loaded {len(df)} raw features, route types: {sorted(df['ROUTENTYP'].dropna().unique())}")

    # ── Step 2: Geometry cleanup ──────────────────────────────────────────────
    df['geometry'] = df.geometry.apply(lambda g: make_valid(g) if g is not None else g)
    df = df[df.geometry.notna() & ~df.geometry.is_empty].reset_index(drop=True)

    # ── Step 3: Netzlücken flag ───────────────────────────────────────────────
    # Planned edges (PLANUNGSTYP == 'geplant') are network gaps (Netzlücken).
    # 'Variante' rows are alternative alignments — treated as gaps too.
    # Flag BEFORE explode so each original feature stays as one edge.
    df['is_development'] = df['PLANUNGSTYP'].isin(['geplant', 'Variante']).astype(int)

    # Separate: Netzlücken keep their full geometry (linemerge collapses any
    # MultiLineString into a single LineString); normal edges are exploded.
    df_netzluecken = df[df['is_development'] == 1].copy()
    df_normal      = df[df['is_development'] == 0].copy()

    def _ensure_single(g):
        if g.geom_type == 'MultiLineString':
            merged = linemerge(g)
            return merged  # returns LineString if connected, else MultiLineString
        return g

    df_netzluecken['geometry'] = df_netzluecken.geometry.apply(_ensure_single)
    df_netzluecken = df_netzluecken[df_netzluecken.geometry.notna() & ~df_netzluecken.geometry.is_empty].reset_index(drop=True)

    # Explode only normal (existing) edges
    df_normal = df_normal.explode(index_parts=False).reset_index(drop=True)

    df = pd.concat([df_normal, df_netzluecken], ignore_index=True)

    n_dev = df['is_development'].sum()
    print(f"  After explode: {len(df)} edges  ({len(df_netzluecken)} Netzlücken kept whole)")
    print(f"  Netzlücken (is_development=1): {n_dev} edges")

    # ── Step 4: Schwachstellen flag ───────────────────────────────────────────
    # Load Schwachstellen from data/raw/SCHWACHSTELLEN/. Each feature is kept as
    # a single whole geometry (no explode) so one Schwachstelle = one spatial unit.
    # Any ALLTAG edge whose geometry intersects a buffered Schwachstelle gets
    # is_schwachstelle = 1. Only existing (bestehend) edges can be weak spots.
    path_sw = 'data/raw/SCHWACHSTELLEN/TBA_VNP_SCHWACHSTELLEN_L.shp'
    df['is_schwachstelle'] = 0
    try:
        sw = gpd.read_file(path_sw, engine='pyogrio')
        if sw.crs.to_epsg() != 2056:
            sw = sw.to_crs("EPSG:2056")

        # Buffer each Schwachstelle line by 10 m (accounts for digitising offsets).
        # Features are NOT exploded — each remains one spatial entity.
        sw_buf = sw[['geometry']].copy()
        sw_buf['geometry'] = sw_buf.geometry.buffer(10.0)
        sw_buf = sw_buf.reset_index(drop=True)

        # Spatial join: mark ALLTAG edges that intersect any Schwachstelle buffer
        joined = gpd.sjoin(
            df[['geometry']].reset_index(),
            sw_buf,
            how='left',
            predicate='intersects'
        )
        hit_indices = joined[joined['index_right'].notna()]['index'].unique()
        df.loc[hit_indices, 'is_schwachstelle'] = 1

        # Schwachstellen apply only to existing edges, not planned ones
        df.loc[df['is_development'] == 1, 'is_schwachstelle'] = 0

        n_sw = df['is_schwachstelle'].sum()
        print(f"  Schwachstellen (is_schwachstelle=1): {n_sw} edges "
              f"(from {len(sw)} Schwachstellen features, 10 m buffer)")
    except Exception as exc:
        print(f"    Schwachstellen join failed: {exc} — is_schwachstelle set to 0 for all edges")

    # ── Step 5: Endpoint snapping to 5 m grid ────────────────────────────────
    # Forces topologically shared nodes to identical coordinates so momepy
    # merges them into one graph node.
    # MultiLineStrings that survived linemerge (disconnected parts) are exploded
    # here so every row passed to momepy is a plain LineString.
    if df.geometry.geom_type.isin(['MultiLineString']).any():
        df = df.explode(index_parts=False).reset_index(drop=True)

    def _snap_endpoints(geom):
        if geom is None or geom.is_empty:
            return None
        coords = list(geom.coords)
        if len(coords) < 2:
            return None
        s = (round(coords[0][0]  / 5) * 5, round(coords[0][1]  / 5) * 5)
        e = (round(coords[-1][0] / 5) * 5, round(coords[-1][1] / 5) * 5)
        if s == e:
            return None
        interior = [(x, y) for x, y in coords[1:-1]]
        return LineString([s] + interior + [e])

    df['geometry'] = df.geometry.apply(_snap_endpoints)
    df = df[df.geometry.notna()].reset_index(drop=True)

    # ── Step 6: KDTree merge of remaining close endpoints (≤ 5 m) ─────────────
    df = _merge_close_endpoints(df, tolerance=5.0)


    # ── Step 7: Extract start/end node coordinates ────────────────────────────
    # These are the snapped, merged endpoint coordinates — used by routing and
    # for building the explicit node table.
    def _start(geom):
        c = list(geom.coords)
        return c[0][0], c[0][1]

    def _end(geom):
        c = list(geom.coords)
        return c[-1][0], c[-1][1]

    df['start_x'] = df.geometry.apply(lambda g: _start(g)[0])
    df['start_y'] = df.geometry.apply(lambda g: _start(g)[1])
    df['end_x']   = df.geometry.apply(lambda g: _end(g)[0])
    df['end_y']   = df.geometry.apply(lambda g: _end(g)[1])

    # Build a unique node table from all endpoint coordinates
    coord_to_id = {}
    node_records = []

    def _node_id(x, y):
        key = (x, y)
        if key not in coord_to_id:
            nid = len(coord_to_id)
            coord_to_id[key] = nid
            node_records.append({'node_id': nid, 'x': x, 'y': y})
        return coord_to_id[key]

    df['source_id'] = df.apply(lambda r: _node_id(r['start_x'], r['start_y']), axis=1)
    df['target_id'] = df.apply(lambda r: _node_id(r['end_x'],   r['end_y']),   axis=1)

    nodes_gdf = gpd.GeoDataFrame(
        node_records,
        geometry=gpd.points_from_xy(
            [n['x'] for n in node_records],
            [n['y'] for n in node_records],
        ),
        crs="EPSG:2056",
    )
    print(f"  Unique nodes: {len(nodes_gdf)}")

    # ── Step 8: Build primal graph via momepy ─────────────────────────────────
    H = momepy.gdf_to_nx(df, approach='primal')
    gdf_nodes, gdf_edges = momepy.nx_to_gdf(H, points=True)
    check_network_connectivity(H, label="ALLTAG momepy primal graph")

    # ── Step 9: Save outputs ─────────────────────────────────────────────────
    os.makedirs('data/Network/processed', exist_ok=True)

    nodes_gdf.to_file('data/Network/processed/nodes.gpkg', driver='GPKG')
    nodes_csv = nodes_gdf.drop(columns='geometry').copy()
    nodes_csv.to_csv('data/Network/processed/nodes_export.csv', index=False)
    print(f"  Nodes → nodes.gpkg + nodes_export.csv  ({len(nodes_gdf)} rows)")

    gdf_edges.to_file('data/Network/processed/edges.gpkg', driver='GPKG')
    edges_csv = gdf_edges.drop(columns='geometry').copy()
    edges_csv.to_csv('data/Network/processed/edges_export.csv', index=False)
    print(f"  Edges → edges.gpkg + edges_export.csv  ({len(gdf_edges)} rows)")

    # Summary by route type
    print("\n  Route type summary:")
    if 'ROUTENTYP' in gdf_edges.columns:
        for rt, grp in gdf_edges.groupby('ROUTENTYP', dropna=False):
            n_dev_rt = grp['is_development'].sum() if 'is_development' in grp.columns else '?'
            n_sw_rt  = grp['is_schwachstelle'].sum() if 'is_schwachstelle' in grp.columns else '?'
            print(f"    {rt or '(none)':35s}  {len(grp):4d} edges  "
                  f"{n_dev_rt} Netzlücken  {n_sw_rt} Schwachstellen")

    return gdf_edges



def import_osmnx_feeder_data():
    # Dübendorf - Hinwil corridor (bounding box in WGS84)
    left, bottom, right, top = 8.5800, 47.2700, 8.8700, 47.4200
    G = osmnx.graph_from_bbox(bbox=(left, bottom, right, top), network_type='bike', simplify=True)

    # Convert OSMnx graph to GeoDataFrames and reproject to Swiss LV95
    gdf_nodes, gdf_edges = osmnx.graph_to_gdfs(G)
    gdf_nodes = gdf_nodes.to_crs(epsg=2056)
    gdf_edges = gdf_edges.to_crs(epsg=2056)

    # Reset index so node IDs and edge u/v/key become regular columns
    gdf_nodes = gdf_nodes.reset_index()
    gdf_edges = gdf_edges.reset_index()

    # Keep only relevant columns for the feeder network
    node_cols = ['osmid', 'x', 'y', 'geometry']
    edge_cols = [
        'u', 'v', 'key',
        'osmid',
        'name',
        'highway',        # road/path type (cycleway, residential, etc.)
        'oneway',
        'length',         # edge length in meters (from OSMnx, WGS84-based)
        'maxspeed',
        'geometry',
    ]

    # Only keep columns that actually exist in the data (OSMnx output varies by area)
    gdf_nodes = gdf_nodes[[c for c in node_cols if c in gdf_nodes.columns]]
    gdf_edges = gdf_edges[[c for c in edge_cols if c in gdf_edges.columns]]

    # Recompute length in meters using projected CRS (more accurate than OSMnx default)
    gdf_edges['length_m'] = gdf_edges.geometry.length

    # Store original winding geometry as WKT for later visualization
    gdf_edges['visual_geom'] = gdf_edges.geometry.apply(lambda g: g.wkt)

    # Save outputs
    os.makedirs('data/Network/processed', exist_ok=True)
    gdf_nodes.to_file('data/Network/processed/osmnx_feeder_nodes.gpkg', driver='GPKG')
    gdf_edges.to_file('data/Network/processed/osmnx_feeder_edges.gpkg', driver='GPKG')

    return gdf_nodes, gdf_edges


def plot_network(
    network,
    source_col: str = None,
    target_col: str = None,
    directed: bool = False,
    node_color: str = "#4C9BE8",
    edge_color: str = "#888888",
    node_size: int = 10,          # small — there are thousands of nodes
    with_labels: bool = False,    # off by default for geo networks
    title: str = "Network Graph",
    figsize: tuple = (12, 9),
    layout: str = "spring",
) -> None:

    G = nx.DiGraph() if directed else nx.Graph()
    pos = {}

    # ── Case 1: GeoDataFrame with geometry ────────────────────────────────
    if isinstance(network, gpd.GeoDataFrame) and "geometry" in network.columns:

        if isinstance(source_col, str) and isinstance(target_col, str):
            # Explicit node-ID columns — use those
            for _, row in network.iterrows():
                G.add_edge(row[source_col], row[target_col])
            layout_fn = {
                "spring": nx.spring_layout, "circular": nx.circular_layout,
                "kamada_kawai": nx.kamada_kawai_layout, "spectral": nx.spectral_layout,
                "shell": nx.shell_layout, "random": nx.random_layout,
            }.get(layout, nx.spring_layout)
            pos = layout_fn(G, seed=42)

        else:
            # Derive nodes from LineString endpoints, snap to integer grid
            # to merge nearby points into the same node
            SNAP = 10  # meters — adjust if network has gaps

            def snap(xy):
                return (round(xy[0] / SNAP) * SNAP, round(xy[1] / SNAP) * SNAP)

            for _, row in network.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
                for line in lines:
                    coords = list(line.coords)
                    u = snap(coords[0])
                    v = snap(coords[-1])
                    if u != v:
                        G.add_edge(u, v)
                        pos[u] = u  # x, y coords used directly as position
                        pos[v] = v

    # ── Case 2: plain list of tuples ──────────────────────────────────────
    elif isinstance(network, list):
        for edge in network:
            if len(edge) == 3:
                u, v, w = edge
                G.add_edge(u, v, weight=w)
            else:
                G.add_edge(*edge)
        pos = nx.spring_layout(G, seed=42)

    # ── Case 3: plain DataFrame with source/target columns ────────────────
    elif isinstance(network, pd.DataFrame) and source_col and target_col:
        for _, row in network.iterrows():
            G.add_edge(row[source_col], row[target_col])
        pos = nx.spring_layout(G, seed=42)

    else:
        raise TypeError(
            f"Cannot build graph from type {type(network)}. "
            "Pass a GeoDataFrame with geometry, a list of tuples, "
            "or a DataFrame with source_col/target_col."
        )

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Draw ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_color,
        width=0.8,
        alpha=0.6,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_color,
        node_size=node_size,
        alpha=0.85,
        ax=ax,
    )
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")   # keep geographic proportions correct
    ax.axis("off")
    plt.tight_layout()
    plt.show()