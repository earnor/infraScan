import os
import glob
import time
import geopandas as gpd
import pandas as pd
import networkx as nx
import rasterio
import rasterio.features
import numpy as np
from scipy.spatial import cKDTree
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import make_valid


_EDGES_CORRIDOR_PATH = r'data/Network/processed/edges_corridor.gpkg'
_EDGES_BORDER_PATH   = r'data/Network/processed/edges_corridor_border.gpkg'

# Placeholder speeds for gap edges in the base network.
# Netzlücken / Schwachstellen: present but below quality — slow cycling path
# Connectivity bridges:         auto-generated cycling links — same degraded speed
BAD_FFS   = 13.0  # km/h
WORST_FFS = 13.0  # km/h


# TODO: check_parallel_edges_gdf() is defined but never called in the pipeline.
# After _build_base_gdf() or _build_graph_direct(), call it to detect duplicate
# node-pairs that create parallel edges in the bidirectional graph.  Parallel
# edges can distort Dijkstra weights and inflate the edge count silently.
def check_parallel_edges_gdf(gdf):
    """
    Report how many undirected node-pairs share more than one edge in *gdf*.

    An edge is identified by its endpoint coordinates rounded to 0.1 m (the
    same precision used everywhere in the routing code).  Two edges that
    connect the same pair of nodes in either direction are counted as one
    parallel pair.

    Returns
    -------
    n_parallel : int   — number of node-pairs with > 1 edge
    counts     : pd.Series — node-pair → edge count (only pairs with count > 1)
    """
    def _endpoints(geom):
        coords = list(geom.coords)
        u = (round(coords[0][0],  1), round(coords[0][1],  1))
        v = (round(coords[-1][0], 1), round(coords[-1][1], 1))
        return tuple(sorted([u, v]))  # undirected

    ep = gdf.geometry.apply(_endpoints)
    counts = ep.value_counts()
    parallel = counts[counts > 1]
    print(f"[check_parallel_edges] {len(parallel)} node-pairs with >1 edge "
          f"({parallel.sum()} edges involved out of {len(gdf)} total)")
    return len(parallel), parallel


def _load_corridor_gdf(only_existing=True):
    """Load the pre-filtered corridor edges from edges_corridor.gpkg and
    edges_corridor_border.gpkg, concatenate them, and explode MultiLineStrings.

    only_existing: if True, drop rows where is_development == 1 (Netzlücken).
    """
    inside = gpd.read_file(_EDGES_CORRIDOR_PATH)
    border = gpd.read_file(_EDGES_BORDER_PATH)
    gdf = pd.concat([inside, border], ignore_index=True)
    if only_existing and 'is_development' in gdf.columns:
        gdf = gdf[gdf['is_development'] == False].copy()
    return gdf.explode(index_parts=False).reset_index(drop=True)


def _build_graph_direct(gdf, cycling_speed_kmh=15):
    """
    Build a routable DiGraph from a GeoDataFrame of LineStrings by iterating
    edges directly.  Uses the stored start/end coordinates of each edge as
    nodes — no topology splitting.  Use this for networks (like the corridor
    files) whose topology is already correct.

    # TODO: no explicit connectivity validation after graph construction.
    # If two edges share a junction whose coordinates differ by >0.1 m before
    # snapping, they produce two separate nodes and leave a gap.
    # After building G, check nx.number_connected_components(G.to_undirected())
    # and print a warning if > 1 so gaps are caught early.

    When the GDF has a `tt_min` column the stored per-edge travel time (in
    minutes) is used as the edge weight (converted to seconds).  Otherwise
    the weight is computed from length and cycling_speed_kmh.
    """
    speed_ms   = cycling_speed_kmh * 1000 / 3600
    G          = nx.DiGraph()
    length_col = 'length_m' if 'length_m' in gdf.columns else None
    tt_col     = 'tt_min'   if 'tt_min'   in gdf.columns else None

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        u = (round(coords[0][0],  1), round(coords[0][1],  1))
        v = (round(coords[-1][0], 1), round(coords[-1][1], 1))
        length = row[length_col] if length_col else geom.length
        if tt_col is not None:
            raw = row[tt_col]
            try:
                tt_sec = float(raw) * 60
                tt = tt_sec if tt_sec > 0 else length / speed_ms
            except (TypeError, ValueError):
                tt = length / speed_ms
        else:
            tt = length / speed_ms
        G.add_node(u, x=u[0], y=u[1])
        G.add_node(v, x=v[0], y=v[1])
        G.add_edge(u, v, weight=tt, length=length)
        G.add_edge(v, u, weight=tt, length=length)

    return G


def _build_graph_from_gdf(gdf, cycling_speed_kmh=15):
    """
    Build a routable DiGraph splitting all edges at mutual intersections via
    unary_union.  Use this only when adding new geometry (e.g. a Netzlücke)
    that may cross existing edges and needs new junction nodes.
    """
    merged = unary_union(gdf.geometry.values)
    segments = list(merged.geoms)

    speed_ms = cycling_speed_kmh * 1000 / 3600
    G = nx.DiGraph()

    for seg in segments:
        coords = list(seg.coords)
        u = (round(coords[0][0], 1), round(coords[0][1], 1))
        v = (round(coords[-1][0], 1), round(coords[-1][1], 1))
        length = seg.length
        tt = length / speed_ms
        G.add_node(u, x=u[0], y=u[1])
        G.add_node(v, x=v[0], y=v[1])
        G.add_edge(u, v, weight=tt, length=length)
        G.add_edge(v, u, weight=tt, length=length)

    return G


# ---------------------------------------------------------------------------
# Public helpers: load-from-file + connectivity check
# ---------------------------------------------------------------------------

def build_network_from_shapefile(path, cycling_speed_kmh=15, snap_tolerance=0.1):
    """
    Load a line shapefile or GeoPackage, node all lines at their mutual
    intersections (via shapely unary_union), and return a bidirectional DiGraph.

    Parameters
    ----------
    path              : str  — path to .shp, .gpkg, or any fiona-readable file
    cycling_speed_kmh : float — default speed used to compute edge weights [km/h]
    snap_tolerance    : float — coordinate rounding precision in metres (default 0.1 m)

    Returns
    -------
    G   : nx.DiGraph   with node keys (x, y) rounded to snap_tolerance
    gdf : GeoDataFrame of the noded line segments (useful for export / QA)
    """
    gdf_raw = gpd.read_file(path)

    # Keep only LineString / MultiLineString rows; drop empties
    gdf_raw = gdf_raw[~gdf_raw.geometry.is_empty & gdf_raw.geometry.notna()].copy()
    gdf_raw = gdf_raw.explode(index_parts=False).reset_index(drop=True)

    print(f"[build_network] Loaded {len(gdf_raw)} line segments from {path}")
    print(f"[build_network] CRS: {gdf_raw.crs}")

    # ── Node the lines: unary_union splits every line at all intersection points ──
    merged = unary_union(gdf_raw.geometry.values)
    if merged.geom_type == 'LineString':
        segments = [merged]
    else:
        segments = [g for g in merged.geoms if not g.is_empty]

    print(f"[build_network] {len(segments)} segments after noding (was {len(gdf_raw)})")

    # ── Build graph ──────────────────────────────────────────────────────────────
    prec = int(round(-np.log10(snap_tolerance))) if snap_tolerance > 0 else 1
    speed_ms = cycling_speed_kmh * 1000 / 3600
    G = nx.DiGraph()

    noded_records = []
    for seg in segments:
        coords = list(seg.coords)
        u = (round(coords[0][0],  prec), round(coords[0][1],  prec))
        v = (round(coords[-1][0], prec), round(coords[-1][1], prec))
        length = seg.length
        tt     = length / speed_ms
        G.add_node(u, x=u[0], y=u[1])
        G.add_node(v, x=v[0], y=v[1])
        G.add_edge(u, v, weight=tt, length=length)
        G.add_edge(v, u, weight=tt, length=length)
        noded_records.append({'geometry': seg, 'length_m': length, 'tt_sec': tt})

    gdf_noded = gpd.GeoDataFrame(noded_records, crs=gdf_raw.crs)

    print(f"[build_network] Graph: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} directed edges")

    return G, gdf_noded


def check_network_connectivity(G, label="network", edges_gdf=None, nodes_gdf=None):
    """
    Run a full connectivity analysis on *G* and print a human-readable report.

    Works on both directed and undirected graphs.  For directed graphs the check
    uses the underlying undirected topology (weak connectivity) so that a
    bidirectional edge network is not falsely reported as disconnected.

    When edges_gdf / nodes_gdf are supplied and the graph is disconnected, both
    GeoDataFrames are filtered in-place to the largest connected component and
    returned in the result dict.  Nodes are matched by rounding coordinates to
    1 decimal place (matching the build_graph_direct convention).

    Parameters
    ----------
    G         : nx.Graph or nx.DiGraph
    label     : str — name printed in the report header
    edges_gdf : GeoDataFrame or None — edge table to filter (optional)
    nodes_gdf : GeoDataFrame or None — node table to filter (optional)

    Returns
    -------
    report : dict with keys
        is_connected        bool
        num_components      int
        largest_component   int  (node count)
        isolated_nodes      int
        component_sizes     list[int]  (sorted descending)
        edges_gdf           GeoDataFrame or None  (filtered if disconnected)
        nodes_gdf           GeoDataFrame or None  (filtered if disconnected)
    """
    Gu = G.to_undirected() if G.is_directed() else G

    n_nodes = Gu.number_of_nodes()
    n_edges = Gu.number_of_edges()

    if n_nodes == 0:
        print(f"[connectivity:{label}] Graph is empty — nothing to check.")
        return {'is_connected': False, 'num_components': 0,
                'largest_component': 0, 'isolated_nodes': 0, 'component_sizes': [],
                'edges_gdf': edges_gdf, 'nodes_gdf': nodes_gdf}

    components     = list(nx.connected_components(Gu))
    num_components = len(components)
    sizes          = sorted([len(c) for c in components], reverse=True)
    largest        = sizes[0]
    isolated       = sum(1 for s in sizes if s == 1)
    is_connected   = num_components == 1

    status = "CONNECTED" if is_connected else f"DISCONNECTED — {num_components} component(s)"

    print(f"\n{'─'*60}")
    print(f"  Connectivity report: {label}")
    print(f"{'─'*60}")
    print(f"  Nodes            : {n_nodes}")
    print(f"  Edges (undirected): {n_edges}")
    print(f"  Components       : {num_components}  →  {status}")
    print(f"  Largest component: {largest} nodes "
          f"({100 * largest / n_nodes:.1f}% of network)")
    if not is_connected:
        print(f"  Isolated nodes   : {isolated}")
        print(f"  Component sizes  : {sizes[:10]}"
              f"{'…' if len(sizes) > 10 else ''}")
        print(f"  Action required  : add connectivity bridges or snap dangling endpoints")
    print(f"{'─'*60}\n")

    if not is_connected and (edges_gdf is not None or nodes_gdf is not None):
        largest_nodes = max(nx.connected_components(Gu), key=len)

        def _coords_of(geom):
            c = list(geom.coords)
            return ((round(c[0][0], 1), round(c[0][1], 1)),
                    (round(c[-1][0], 1), round(c[-1][1], 1)))

        if edges_gdf is not None:
            edge_mask = edges_gdf.geometry.apply(
                lambda g: all(n in largest_nodes for n in _coords_of(g))
            )
            edges_gdf = edges_gdf[edge_mask].reset_index(drop=True)

        if nodes_gdf is not None:
            pt_mask = nodes_gdf.geometry.apply(
                lambda g: (round(g.x, 1), round(g.y, 1)) in largest_nodes
            )
            nodes_gdf = nodes_gdf[pt_mask].reset_index(drop=True)

        n_dropped = n_nodes - len(largest_nodes)
        n_edges_kept = len(edges_gdf) if edges_gdf is not None else '?'
        print(f"  Kept largest component: {len(largest_nodes)} nodes, "
              f"{n_edges_kept} edges  ({n_dropped} nodes dropped)")

    return {
        'is_connected':      is_connected,
        'num_components':    num_components,
        'largest_component': largest,
        'isolated_nodes':    isolated,
        'component_sizes':   sizes,
        'edges_gdf':         edges_gdf,
        'nodes_gdf':         nodes_gdf,
    }


def _build_base_gdf():
    """Assemble the full base network used for both status-quo scoring and as
    the fixed backdrop in every per-development Dijkstra run.

    Layer            Source                              tt_min
    ─────────────────────────────────────────────────────────────────────
    Good existing    edges_corridor{,_border}.gpkg       ROUTENTYP-based ffs
    Netzlücken       same files (is_development == 1)    BAD_FFS
    Schwachstellen   same files (is_schwachstelle == 1)  BAD_FFS
    Connectivity     connectivity_developments.gpkg      WORST_FFS
    bridges

    The corridor files may lack ffs/tt_min (saved before get_edge_attributes
    runs).  In that case ffs is merged from edges_with_attribute.gpkg by
    ID_edge, and tt_min is computed per-segment from (length/ffs).
    """
    inside = gpd.read_file(_EDGES_CORRIDOR_PATH)
    border = gpd.read_file(_EDGES_BORDER_PATH)
    gdf = pd.concat([inside, border], ignore_index=True).explode(index_parts=False).reset_index(drop=True)

    # Merge ffs from edges_with_attribute.gpkg when corridor files lack it
    if 'ffs' not in gdf.columns and 'ID_edge' in gdf.columns:
        attr_path = 'data/Network/processed/edges_with_attribute.gpkg'
        if os.path.exists(attr_path):
            attrs = gpd.read_file(attr_path)[['ID_edge', 'ffs']].drop_duplicates('ID_edge')
            gdf = gdf.merge(attrs, on='ID_edge', how='left')
            print(f"  Base GDF: merged ffs from edges_with_attribute.gpkg "
                  f"({gdf['ffs'].notna().sum()}/{len(gdf)} edges matched)")

    # TODO: default ffs 15.0 km/h is hardcoded in three places (here, in
    # _build_graph_direct fallback, and in travel_cost_developments fallback).
    # Define a module-level constant FFS_DEFAULT = 15.0 and reference it
    # everywhere so a single change propagates.  Also add a validation that
    # no ffs value is <= 0 (would produce infinite or negative travel times).
    seg_len = gdf['length_m'] if 'length_m' in gdf.columns else gdf.geometry.length
    if 'ffs' in gdf.columns:
        ffs_col = gdf['ffs'].fillna(20.0)
    else:
        ffs_col = pd.Series(20.0, index=gdf.index)
    gdf['tt_min'] = (seg_len / 1000) / ffs_col * 60

    # Override Netzlücken to BAD_FFS — Schwachstellen keep their ROUTENTYP-based ffs
    bad_mask = pd.Series(False, index=gdf.index)
    if 'is_development' in gdf.columns:
        bad_mask |= gdf['is_development'].astype(bool)

    if bad_mask.any():
        gdf.loc[bad_mask, 'tt_min'] = (seg_len[bad_mask] / 1000) / BAD_FFS * 60
        print(f"  Base GDF: {bad_mask.sum()} edge(s) set to BAD_FFS ({BAD_FFS} km/h) — Netzlücken only")

    # Append connectivity bridges at WORST_FFS (never scored, always present)
    conn_path = 'data/Network/processed/connectivity_developments.gpkg'
    if os.path.exists(conn_path):
        bridges = gpd.read_file(conn_path).explode(index_parts=False).reset_index(drop=True)
        bridge_len = bridges.geometry.length
        bridges['tt_min'] = (bridge_len / 1000) / WORST_FFS * 60
        if 'length_m' not in bridges.columns:
            bridges['length_m'] = bridge_len
        keep = ['geometry', 'tt_min', 'length_m']
        for col in ('ID_edge', 'is_development', 'is_schwachstelle'):
            if col in bridges.columns:
                keep.append(col)
        gdf = pd.concat([gdf, bridges[keep]], ignore_index=True)
        print(f"  Base GDF: {len(bridges)} connectivity bridge(s) added at WORST_FFS ({WORST_FFS} km/h)")

    return gdf


build_graph_direct = _build_graph_direct   # public alias — use this outside this module


def augment_with_netzluecken(edges_corridor, points_corridor,
                              c_cycle_path_new=1000,
                              c_om_cycle_path=100,
                              c_structural_maint=0.012):
    """
    Label Netzlücken (is_development==1) with ROUTENTYP='Netzlücke' and BAD_FFS,
    store the original ROUTENTYP and ffs as routentyp_built / ffs_built so
    scoring functions can use the correct per-edge built speed, rebuild
    connectivity, and write a scoring CSV skeleton to
    data/Network/processed/netzluecken_scoring.csv.

    CSV columns: length, type/speed before (unbuilt) & after (built),
    construction and maintenance costs.  Scoring columns (comfort,
    accessibility, safety) are left blank to be filled by the respective
    scoring functions.

    Returns (edges_aug, points_corridor).
    """
    edges_aug = edges_corridor.copy()

    # Derive ffs from ROUTENTYP if the corridor edges were saved before
    # get_edge_attributes / plot_edge_attributes ran (edges.gpkg lacks ffs).
    if 'ffs' not in edges_aug.columns:
        _FFS_MAP = {
            'Velobahn':                       20.0,
            'Veloschnellroute':               20.0,
            'Hauptverbindung':                18.0,
            'Nebenverbindung':                16.0,
            'Zusätzliche Freizeitverbindung': 18.0,
        }
        edges_aug['ffs'] = edges_aug['ROUTENTYP'].map(
            lambda rt: _FFS_MAP.get(rt, 20.0))
        print(f"  augment_with_netzluecken: ffs derived from ROUTENTYP "
              f"(column was missing from edges_corridor)")

    if 'tt_min' not in edges_aug.columns:
        edges_aug['tt_min'] = (edges_aug['length_m'] / 1000) / edges_aug['ffs'] * 60

    nl_mask = edges_aug['is_development'] == 1

    # Snapshot original values before overwriting — these become the "after"
    # columns (what the edge looks like once the Netzlücke is built).
    nl_orig = edges_aug.loc[nl_mask, ['length_m', 'ROUTENTYP', 'ffs']].copy()

    edges_aug.loc[nl_mask, 'ffs_built']       = nl_orig['ffs'].values
    edges_aug.loc[nl_mask, 'routentyp_built'] = nl_orig['ROUTENTYP'].values
    edges_aug.loc[nl_mask, 'ROUTENTYP'] = 'Netzlücke'
    edges_aug.loc[nl_mask, 'ffs']       = BAD_FFS
    edges_aug.loc[nl_mask, 'tt_min']    = (
        edges_aug.loc[nl_mask, 'length_m'] / 1000) / BAD_FFS * 60
    print(f"\n  Added {nl_mask.sum()} Netzlücken at BAD_FFS ({BAD_FFS} km/h) to corridor graph")

    G_with_nl = build_graph_direct(edges_aug)
    conn_nl   = check_network_connectivity(G_with_nl, label="corridor + Netzlücken",
                                           edges_gdf=edges_aug, nodes_gdf=points_corridor)
    edges_aug       = conn_nl['edges_gdf']
    points_corridor = conn_nl['nodes_gdf']

    # Build scoring CSV: before = current unbuilt state, after = once built
    nl_scoring = pd.DataFrame({
        'length_m':                    nl_orig['length_m'].values,
        'type_before':                 'Netzlücke',
        'type_after':                  nl_orig['ROUTENTYP'].values,
        'speed_before_kmh':            BAD_FFS,
        'speed_after_kmh':             nl_orig['ffs'].values,
        'route_comfort_index_before':  None,
        'route_comfort_index_after':   None,
        'accessibility_scoring':       None,
        'route_comfort_scoring':       None,
        'safety_index_before':         None,
        'safety_index_after':          None,
        'safety_scoring':              None,
        'construction_cost_chf':       nl_orig['length_m'].values * c_cycle_path_new,
        'maintenance_cost_chf_yr':     nl_orig['length_m'].values * (
            c_om_cycle_path + c_structural_maint * c_cycle_path_new),
    })

    os.makedirs('data/Network/processed', exist_ok=True)
    nl_scoring.to_csv('data/Network/processed/netzluecken_scoring.csv', index=False)
    print(f"  Netzlücken scoring CSV → netzluecken_scoring.csv ({len(nl_scoring)} rows)")

    return edges_aug, points_corridor


_MAX_DETOUR_FACTOR = 2.5   # drop OD pairs where network path > 2.5× straight-line distance


def compute_od_matrix(max_dist_m=25_000, cycling_speed_kmh=15):
    """
    Build the full base network (existing edges + Netzlücken at BAD_FFS +
    connectivity bridges at WORST_FFS), snap all corridor access points to
    graph nodes, then run single-source Dijkstra from each origin to find the
    fastest path to every other access point.

    OD pairs are dropped when:
      • path distance > max_dist_m (default 25 km), OR
      • detour factor = path_dist / straight_line_dist > _MAX_DETOUR_FACTOR (2.5×)
        — flags paths routed through long connectivity bridges rather than
          real cycling infrastructure.
    """
    access_pts = gpd.read_file('data/Network/processed/points_corridor.gpkg')

    # Precompute straight-line lookup: ID_point → (x, y) in LV95
    id_to_xy = {
        int(r['ID_point']): (r.geometry.x, r.geometry.y)
        for _, r in access_pts.iterrows()
    }

    t0 = time.time()
    gdf_base = _build_base_gdf()
    G = _build_graph_direct(gdf_base, cycling_speed_kmh)
    print(f"  Base graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"in {time.time() - t0:.1f}s")

    snapped    = _snap_points_to_graph(access_pts, G)
    node_to_id = {node: id_pt for node, id_pt in snapped}
    origin_nodes = list(node_to_id.keys())

    od_rows  = []
    n_drop_dist   = 0
    n_drop_detour = 0
    t0 = time.time()

    for origin_node in origin_nodes:
        id_origin = node_to_id[origin_node]
        ox, oy = id_to_xy.get(id_origin, (None, None))
        tt_dict, path_dict = nx.single_source_dijkstra(G, origin_node, weight='weight')

        for dest_node, id_dest in node_to_id.items():
            if dest_node == origin_node:
                continue
            if dest_node not in path_dict:
                continue

            path = path_dict[dest_node]
            dist_m = sum(
                G[path[k]][path[k + 1]].get('length', 0.0)
                for k in range(len(path) - 1)
            )
            if dist_m > max_dist_m:
                n_drop_dist += 1
                continue

            air_dist_m  = None
            detour_ratio = None
            if ox is not None:
                dx, dy = id_to_xy.get(id_dest, (None, None))
                if dx is not None:
                    air_dist_m = ((ox - dx) ** 2 + (oy - dy) ** 2) ** 0.5
                    if air_dist_m > 0:
                        detour_ratio = dist_m / air_dist_m
                        if detour_ratio > _MAX_DETOUR_FACTOR:
                            n_drop_detour += 1
                            continue

            od_rows.append({
                'origin_id':    id_origin,
                'dest_id':      id_dest,
                'tt_sec':       tt_dict[dest_node],
                'dist_m':       dist_m,
                'air_dist_m':   air_dist_m,
                'detour_ratio': detour_ratio,
            })

    od_df = pd.DataFrame(od_rows)
    print(f"  OD matrix: {len(od_df)} pairs kept  "
          f"[dropped {n_drop_dist} >{max_dist_m/1000:.0f} km, "
          f"{n_drop_detour} detour >{_MAX_DETOUR_FACTOR}×]  "
          f"[{time.time() - t0:.1f}s, {len(origin_nodes)} origins]")

    os.makedirs('data/OD', exist_ok=True)
    od_df.to_csv('data/OD/od_fastest_paths.csv', index=False)
    print(f"  Saved → data/OD/od_fastest_paths.csv")
    return od_df


def build_project_network_graph(cycling_speed_kmh=15, only_existing=True):
    """Build a routable DiGraph from the saved corridor edge files."""
    gdf = _load_corridor_gdf(only_existing=only_existing)
    G = _build_graph_direct(gdf, cycling_speed_kmh)
    label = 'existing' if only_existing else 'all'
    print(f"Corridor graph ({label}: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges)")
    check_network_connectivity(G, label=f"corridor ({label})")
    return G


def _snap_points_to_graph(points_gdf, G):
    """
    Snap each point in points_gdf to the nearest node in G.
    Returns a list of (snap_node, id_point) tuples (one per row).
    """
    nodes_list = list(G.nodes())
    node_arr = np.array(nodes_list)
    tree = cKDTree(node_arr)

    result = []
    for _, row in points_gdf.iterrows():
        xy = np.array([row.geometry.x, row.geometry.y])
        _, idx = tree.query(xy)
        result.append((nodes_list[idx], row['ID_point']))
    return result


def _rasterize_network_to_grid(node_times, node_srcs, G, raster_shape, transform):
    """
    Convert per-node Dijkstra results to a raster grid via nearest-node assignment.
    Pixels further than 2 km from any reachable node are marked as unreachable.

    Returns: (travel_time_arr float32, source_id_arr float32)
    """
    nodes = list(node_times.keys())
    coords = np.array([(G.nodes[n]['x'], G.nodes[n]['y']) for n in nodes])
    tt_vals  = np.array([node_times[n]           for n in nodes], dtype=np.float32)
    src_vals = np.array([node_srcs.get(n, -1.0)  for n in nodes], dtype=np.float32)

    tree = cKDTree(coords)

    h, w = raster_shape
    col_grid, row_grid = np.meshgrid(np.arange(w), np.arange(h))
    xs, ys = rasterio.transform.xy(transform, row_grid.ravel(), col_grid.ravel())
    pixel_coords = np.column_stack([xs, ys])

    dists, idxs = tree.query(pixel_coords)

    tt_arr  = tt_vals[idxs].reshape(h, w)
    src_arr = src_vals[idxs].reshape(h, w)

    # TODO: 2000 m hardcoded threshold for "too far from any reachable node".
    # Pixels beyond this become NaN (unreachable).  Should be a named constant
    # or parameter tied to the expected network density of the corridor.
    far = dists.reshape(h, w) > 2000
    tt_arr[far]  = np.nan
    src_arr[far] = -1.0

    return tt_arr, src_arr


def make_cycling_speed_raster(cycling_speed_kmh=15):
    """
    Creates a uniform cycling speed raster from the existing speed_limit_raster.
    All passable cells (speed > 0) are replaced with cycling_speed_kmh.
    Impassable cells (speed == 0, e.g. lakes) remain 0.
    Result saved to data/Network/OSM_tif/cycling_speed_raster.tif
    """
    raster_file = r"data/Network/OSM_tif/speed_limit_raster.tif"
    output_file = r"data/Network/OSM_tif/cycling_speed_raster.tif"

    with rasterio.open(raster_file) as src:
        raster_data = src.read(1).astype(float)
        profile = src.profile

    # Replace all passable cells with the cycling speed
    cycling_raster = np.where(raster_data > 0, cycling_speed_kmh, 0).astype(float)

    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(cycling_raster, 1)

    print(f"Cycling speed raster created at {output_file} ({cycling_speed_kmh} km/h)")
    return output_file


def travel_cost_polygon(frame, raster_file=r"data/Network/OSM_tif/cycling_speed_raster.tif",
                        cycling_speed_kmh=15):
    """
    Compute status-quo travel time from all access points using the FULL base
    network: existing edges at their surveyed speeds, Netzlücken / Schwachstellen
    at BAD_FFS, and connectivity bridges at WORST_FFS.
    """
    points_all = gpd.read_file(r"data/Network/processed/points_corridor.gpkg")

    # Build full base graph
    t0 = time.time()
    gdf_base = _build_base_gdf()
    G = _build_graph_direct(gdf_base, cycling_speed_kmh)
    print(f"Base graph built in {time.time() - t0:.1f}s  "
          f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    check_network_connectivity(G, label="base graph (travel_cost_polygon)")

    # Snap access points to nearest network node; track which node → which ID_point
    snapped = _snap_points_to_graph(points_all, G)
    source_nodes = list({node for node, _ in snapped})
    node_to_id = {}
    for node, id_pt in snapped:
        node_to_id[node] = id_pt  # last write wins for duplicates

    # Multi-source Dijkstra on Alltag network
    t0 = time.time()
    distances, paths = nx.multi_source_dijkstra(G=G, sources=source_nodes, weight='weight')
    print(f"Dijkstra: {len(distances)} reachable nodes in {time.time() - t0:.1f}s")

    # Build per-node source-ID map
    node_srcs = {}
    for node, path in paths.items():
        src = path[0] if path else node
        node_srcs[node] = float(node_to_id.get(src, -1))

    # Rasterize onto reference raster grid
    with rasterio.open(raster_file) as ref:
        transform    = ref.transform
        raster_shape = (ref.height, ref.width)
        profile      = ref.profile.copy()

    tt_arr, src_arr = _rasterize_network_to_grid(distances, node_srcs, G, raster_shape, transform)

    os.makedirs(r'data/Network/travel_time', exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(dtype='float32', count=1)

    with rasterio.open(r'data/Network/travel_time/travel_time_raster.tif', 'w', **out_profile) as dst:
        dst.write(tt_arr, 1)

    src_arr[np.isnan(tt_arr)] = -1.0
    path_id_raster = r'data/Network/travel_time/source_id_raster.tif'
    with rasterio.open(path_id_raster, 'w', **out_profile) as dst:
        dst.write(src_arr, 1)

    gdf_polygon = raster_to_polygons(path_id_raster)
    gdf_polygon.to_file(r"data/Network/travel_time/Voronoi_statusquo.gpkg")

    return





def raster_to_polygons(tif_path):
    # Read the raster data
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        data = data.astype('int32')
        transform = src.transform

    # Find unique positive values in the raster
    unique_values = np.unique(data[data >= 0])

    # Create a mask for negative values (holes)
    negative_mask = data < 0

    # Initialize list to store polygons and their values
    polygons = []

    # Iterate over unique values and create polygons
    for val in unique_values:
        # Create mask for the current value
        positive_mask = data == val

        # Generate shapes (polygons) for positive values
        positive_shapes = rasterio.features.shapes(data, mask=positive_mask, transform=transform)

        # Generate shapes for negative values (holes)
        hole_shapes = rasterio.features.shapes(data, mask=negative_mask, transform=transform)

        # Combine positive shapes and holes
        combined_polygons = []
        for shape, value in positive_shapes:
            if value == val:
                outer_polygon = make_valid(Polygon(shape['coordinates'][0]))
                holes = [make_valid(Polygon(hole_shape['coordinates'][0])) for hole_shape, hole_value in hole_shapes if
                         hole_value < 0]
                holes = [h for h in holes if not h.is_empty]
                holes_union = unary_union(holes) if holes else outer_polygon.__class__()

                # Combine outer polygon with holes
                if holes_union.is_empty:
                    combined_polygons.append(outer_polygon)
                else:
                    combined_polygon = outer_polygon.difference(holes_union)
                    combined_polygons.append(combined_polygon)

        # Add combined polygons to list
        polygons.extend([{'geometry': poly, 'ID_point': val} for poly in combined_polygons if not poly.is_empty])

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(polygons, crs=src.crs)
    gdf_dissolved = gdf.dissolve(by='ID_point')

    return gdf_dissolved


def groupby_multipoly(df, by, aggfunc="first"):
    data = df.drop(labels=df.geometry.name, axis=1)
    aggregated_data = data.groupby(by=by).agg(aggfunc)

    # Process spatial component
    def merge_geometries(block):
        return MultiPolygon(block.values)

    g = df.groupby(by=by, group_keys=False)[df.geometry.name].agg(
        merge_geometries
    )

    # Aggregate
    aggregated_geometry = gpd.GeoDataFrame(g, geometry=df.geometry.name, crs=df.crs)
    # Recombine
    aggregated = aggregated_geometry.join(aggregated_data)
    return aggregated


def raster_to_graph(raster_data, raster_cell=50):


    # convert travel speed from km/h to m/s
    raster_data = raster_data * 1000 / 3600

    rows, cols = raster_data.shape
    graph = nx.grid_2d_graph(rows, cols)

    nodes_to_remove = []
    for node in graph.nodes:
        y, x = node
        if raster_data[y, x] == 0:
            nodes_to_remove.append(node)

    graph.remove_nodes_from(nodes_to_remove)

    # Add weights for existing edges in the grid_2d_graph
    for (node1, node2) in graph.edges:
        y1, x1 = node1
        y2, x2 = node2
        if raster_data[y1, x1] == 0 or raster_data[y2, x2] == 0:
            # Assign a high weight to this edge
            weight = None
        else:
            # Calculate weight normally
            weight = (raster_cell / raster_data[y1, x1] + raster_cell / raster_data[y2, x2]) / 2

        #weight = (0.1 / raster_data[y1, x1] + 0.1 / raster_data[y2, x2]) / 2 * 3600
        graph[node1][node2]['weight'] = weight

    # Add diagonal edges (from 4 to 8 neighbors)
    new_edges = []
    for x in range(cols - 1):
        for y in range(rows - 1):
            # Check for zero values in raster data for diagonal neighbors
            if raster_data[y, x] == 0 or raster_data[y + 1, x + 1] == 0:
                weight = None
            else:
                weight = 1.4 * (raster_cell / raster_data[y, x] + raster_cell / raster_data[y + 1, x + 1]) / 2

            new_edges.append(((y, x), (y + 1, x + 1), {'weight': weight}))
            
            if raster_data[y, x + 1] == 0 or raster_data[y + 1, x] == 0:
                weight = None
            else:
                weight = 1.4 * (raster_cell / raster_data[y, x + 1] + raster_cell / raster_data[y + 1, x]) / 2

            new_edges.append(((y, x + 1), (y + 1, x), {'weight': weight}))

    # Add new diagonal edges with calculated weights
    graph.add_edges_from(new_edges)

    # iterate over all options
    # get the closest point
    return graph





def travel_cost_developments(frame, raster_file=r"data/Network/OSM_tif/cycling_speed_raster.tif",
                              cycling_speed_kmh=15):
    """
    Score each official Netzlücke / Schwachstelle development candidate by
    measuring travel-time improvement against a fixed base network.

    Base network (same as travel_cost_polygon):
      • Existing edges       → stored ffs / tt_min
      • Netzlücken           → BAD_FFS  (gap present but hard to traverse)
      • Schwachstellen       → BAD_FFS  (below quality standard)
      • Connectivity bridges → WORST_FFS (bare link, never scored)

    Per-development scenario:
      Find the one edge in the base that matches this candidate (by ID_edge),
      restore its planned ffs-derived tt_min, re-run Dijkstra, compare to base.
      All other Netzlücken / Schwachstellen stay at BAD_FFS.
      Connectivity bridges are NEVER scored — they are excluded from the loop.
    """
    os.makedirs('data/Network/travel_time/developments', exist_ok=True)
    for f in glob.glob(r'data/Network/travel_time/developments/*'):
        os.remove(f)

    points = gpd.read_file(r"data/Network/processed/points_corridor.gpkg")

    # Official candidates only — connectivity bridges are not scored
    all_candidates = gpd.read_file(r"data/Network/processed/development_candidates.gpkg")
    dev_candidates = all_candidates[
        (all_candidates['within_corridor'] | all_candidates['on_border']) &
        (all_candidates['dev_type'] != 'connectivity')
    ].copy()
    print(f"Scoring {len(dev_candidates)} official development candidate(s) "
          f"(connectivity bridges excluded)")

    with rasterio.open(raster_file) as ref:
        transform    = ref.transform
        raster_shape = (ref.height, ref.width)
        out_profile  = ref.profile.copy()
    out_profile.update(dtype='float32', count=1)

    # ── Build the shared base network ────────────────────────────────────────
    t0 = time.time()
    gdf_base = _build_base_gdf()
    G_base   = _build_graph_direct(gdf_base, cycling_speed_kmh)
    print(f"Base graph: {G_base.number_of_nodes()} nodes, "
          f"{G_base.number_of_edges()} edges in {time.time()-t0:.1f}s")
    check_network_connectivity(G_base, label="base graph (travel_cost_developments)")

    snapped_base    = _snap_points_to_graph(points, G_base)
    base_sources    = list({n for n, _ in snapped_base})
    base_node_to_id = {n: id_pt for n, id_pt in snapped_base}

    t0 = time.time()
    base_distances, base_paths = nx.multi_source_dijkstra(
        G=G_base, sources=base_sources, weight='weight'
    )
    print(f"Baseline Dijkstra: {len(base_distances)} reachable nodes in {time.time()-t0:.1f}s")

    base_node_srcs = {
        n: float(base_node_to_id.get(path[0] if path else n, -1))
        for n, path in base_paths.items()
    }
    base_tt_arr, base_src_arr = _rasterize_network_to_grid(
        base_distances, base_node_srcs, G_base, raster_shape, transform
    )
    base_tt_inf = np.where(np.isnan(base_tt_arr), np.inf, base_tt_arr)

    has_id_edge = 'ID_edge' in gdf_base.columns

    # Status-quo Voronoi is reused for every development — access points never change,
    # so polygon geometries are identical across all developments.
    sq_voronoi_path = r'data/Network/travel_time/Voronoi_statusquo.gpkg'
    sq_voronoi = gpd.read_file(sq_voronoi_path) if os.path.exists(sq_voronoi_path) else None

    improvements = []  # list of {ID_new, tt_improvement_h}

    # ── Per-development loop ─────────────────────────────────────────────────
    for _, dev_row in dev_candidates.iterrows():
        id_new   = dev_row['ID_new']
        dev_type = str(dev_row.get('dev_type', ''))
        desc     = str(dev_row.get('description', f'dev {id_new}'))[:70]
        print(f"\nDevelopment {id_new} [{dev_type}]: {desc}")

        # Resolve planned ffs (used to compute per-segment tt_min when improved)
        if 'ffs' in dev_row.index and pd.notna(dev_row.get('ffs')):
            planned_ffs = float(dev_row['ffs'])
        elif ('tt_min' in dev_row.index and pd.notna(dev_row.get('tt_min'))
              and 'length_m' in dev_row.index and float(dev_row['length_m']) > 0):
            # Back-compute ffs from total tt_min and total length
            planned_ffs = (float(dev_row['length_m']) / 1000) / (float(dev_row['tt_min']) / 60)
        else:
            planned_ffs = 15.0  # fallback

        # Find matching edge(s) in base by ID_edge
        id_edge = dev_row.get('ID_edge', None) if 'ID_edge' in dev_row.index else None
        if not has_id_edge or id_edge is None:
            print(f"  Warning: no ID_edge available — skipping development {id_new}")
            continue

        mask = gdf_base['ID_edge'] == id_edge
        n_matched = mask.sum()
        if n_matched == 0:
            print(f"  Warning: no edge matched ID_edge={id_edge} in base GDF — skipping")
            continue

        # Build development GDF: swap matched edge(s) to planned speed
        gdf_dev = gdf_base.copy()
        length_col_name = 'length_m' if 'length_m' in gdf_dev.columns else None
        if length_col_name:
            seg_lens = gdf_dev.loc[mask, length_col_name]
        else:
            seg_lens = gdf_dev.loc[mask].geometry.length
        gdf_dev.loc[mask, 'tt_min'] = (seg_lens / 1000) / planned_ffs * 60

        print(f"  Swapped {n_matched} segment(s) with ID_edge={id_edge} "
              f"→ planned ffs={planned_ffs:.1f} km/h")

        t0 = time.time()
        G_dev = _build_graph_direct(gdf_dev, cycling_speed_kmh)
        print(f"  Dev graph: {G_dev.number_of_nodes()} nodes, "
              f"{G_dev.number_of_edges()} edges in {time.time()-t0:.1f}s")

        snapped_dev = _snap_points_to_graph(points, G_dev)
        dev_sources = list({n for n, _ in snapped_dev})

        t0 = time.time()
        dev_distances, dev_paths = nx.multi_source_dijkstra(
            G=G_dev, sources=dev_sources, weight='weight'
        )
        print(f"  Dijkstra: {len(dev_distances)} reachable nodes in {time.time()-t0:.1f}s")

        # TODO: source-node mapping uses base_node_to_id (keyed on G_base nodes).
        # dev_paths comes from G_dev, whose node set is identical to G_base
        # because _build_graph_direct() uses the same rounded coordinates.
        # This works correctly as long as no new access points are injected into
        # G_dev.  If that ever changes, create a separate dev_node_to_id here.
        dev_node_srcs = {
            n: float(base_node_to_id.get(path[0] if path else n, -1))
            for n, path in dev_paths.items()
        }
        dev_tt_arr, dev_src_arr = _rasterize_network_to_grid(
            dev_distances, dev_node_srcs, G_dev, raster_shape, transform
        )
        dev_tt_inf = np.where(np.isnan(dev_tt_arr), np.inf, dev_tt_arr)

        # Element-wise minimum vs. baseline
        dev_wins  = dev_tt_inf < base_tt_inf
        merged_tt = np.where(dev_wins, dev_tt_inf, base_tt_inf).astype(np.float32)
        merged_tt[np.isinf(merged_tt)] = np.nan

        merged_src = np.where(dev_wins, 9999.0, base_src_arr).astype(np.float32)
        merged_src[np.isnan(merged_tt)] = -1.0

        improvement = np.nansum(base_tt_inf[~np.isinf(base_tt_inf)] -
                                merged_tt[~np.isinf(base_tt_inf)])
        improvement_h = improvement / 3600
        print(f"  Total TT improvement: {improvement_h:.1f} node-hours")
        improvements.append({'ID_new': id_new, 'tt_improvement_h': improvement_h})

        with rasterio.open(
            fr'data/Network/travel_time/developments/dev{id_new}_travel_time_raster.tif',
            'w', **out_profile
        ) as dst:
            dst.write(merged_tt, 1)

        # Reuse status-quo Voronoi geometry — access points don't change between
        # developments, so the catchment polygons are identical for every dev.
        if sq_voronoi is not None:
            sq_voronoi.to_file(
                fr"data/Network/travel_time/developments/dev{id_new}_Voronoi.gpkg"
            )
            # Write status-quo source_id raster for this development so that
            # GetVoronoiOD_multi() can discover and process it.  Since Voronoi
            # zones are unchanged (same access points), base_src_arr is correct.
            path_id_raster = fr'data/Network/travel_time/developments/dev{id_new}_source_id_raster.tif'
            with rasterio.open(path_id_raster, 'w', **out_profile) as dst:
                dst.write(base_src_arr, 1)
        else:
            path_id_raster = (
                fr'data/Network/travel_time/developments/dev{id_new}_source_id_raster.tif'
            )
            with rasterio.open(path_id_raster, 'w', **out_profile) as dst:
                dst.write(merged_src, 1)
            gdf_polygon = raster_to_polygons(path_id_raster)
            gdf_polygon.to_file(
                fr"data/Network/travel_time/developments/dev{id_new}_Voronoi.gpkg"
            )

    os.makedirs('data/costs', exist_ok=True)
    pd.DataFrame(improvements).to_csv('data/costs/dijkstra_improvements.csv', index=False)
    print(f"\n  Dijkstra improvements saved → data/costs/dijkstra_improvements.csv")
    return

def match_access_point_on_cycling_network(idx, raster):
    matched_dict = {}
    updated_idx = []
    for i in idx:
        y, x = i
        match_found = raster[y, x] > 0
        for radius in range(1, 4):
            if match_found:
                break
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < raster.shape[0] and 0 <= nx < raster.shape[1]:
                        if raster[ny, nx] > 0:
                            matched_dict[(ny, nx)] = i
                            i = (ny, nx)
                            match_found = True
                            break
                if match_found:
                    break
        updated_idx.append(i)
    return updated_idx, matched_dict


def get_voronoi_frame(polygons_gdf):
    margin = 100
    points_gdf = gpd.read_file(r"data/Network/processed/points_with_attribute.gpkg")
    points_gdf = points_gdf[points_gdf["intersection"] == 0]

    points_all = gpd.read_file(r"data/Network/processed/points.gpkg")
    points_all.crs = "epsg:2056"
    points_all = points_all[points_all["intersection"] == 0]

    # union of all polygons from points
    # get all polygons touching it
    # get its extrem values

    # Step 1: Identify polygons containing points
    # TODO: drop(columns=["index_right"]) will raise KeyError if the sjoin
    # did not produce that column (e.g. when points_gdf has no matching column).
    # Replace with .drop(columns=["index_right"], errors='ignore').
    points_gdf = points_gdf.drop(columns=["index_right"])
    polygons_with_points = gpd.sjoin(polygons_gdf, points_gdf, predicate='contains').drop_duplicates(
        subset=polygons_gdf.index.name)
    polygons_with_points = polygons_with_points[["ID_point", "geometry"]]
    polygons_with_points = polygons_with_points.drop_duplicates()
    # Use unary_union to union all geometries into a single geometry
    #polygons_with_points = unary_union(polygons_with_points['geometry'])
    #polygons_with_points = gpd.GeoDataFrame(geometry=[polygons_with_points], crs="epsg:2056")
    #polygons_with_points.to_file(r"data/Network/processed/ppg.gpkg")

    # Step 2: Find polygons touching the identified set
    # Add custom suffixes to avoid naming conflicts
    touching_polygons = gpd.sjoin(polygons_gdf, polygons_with_points, how='inner', predicate='touches', lsuffix='left',
                                  rsuffix='_right')

    # Combine the identified polygons and the ones touching them
    #combined_polygons = pd.concat([polygons_with_points, touching_polygons]).drop_duplicates(subset=polygons_gdf.index.name)

    # Step 3: Extract points contained in the combined set of polygons

    points_in_polygons = gpd.sjoin(points_all, touching_polygons, predicate='within', lsuffix='_l',
                                  rsuffix='r')
    points_in_polygons = points_in_polygons[["geometry", "index_r"]]
    points_in_polygons = points_in_polygons.drop_duplicates()

    # Step 4: Calculate extreme values
    xmin, ymin, xmax, ymax = points_in_polygons.total_bounds

    return [xmin-margin, ymin-margin, xmax+margin, ymax+margin]



