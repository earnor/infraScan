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

# Set working directory to the folder that contains this file so relative
# paths work regardless of where the script is launched from.
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def _merge_close_endpoints(df, tolerance=5.0):
    """
    Merge edge endpoints that are within `tolerance` metres of each other.
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
    # Load Schwachstellen from data/raw/SCHWACHSTELLEN/
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





