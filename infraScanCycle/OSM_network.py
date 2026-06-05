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


BAD_FFS   = 13.0  # km/h — base speed for Netzlücken / Schwachstellen
WORST_FFS = 13.0  # km/h — base speed for auto-generated connectivity bridges



def check_parallel_edges_gdf(gdf):

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

    inside = gpd.read_file(_EDGES_CORRIDOR_PATH)
    border = gpd.read_file(_EDGES_BORDER_PATH)
    gdf = pd.concat([inside, border], ignore_index=True)
    if only_existing and 'is_development' in gdf.columns:
        gdf = gdf[gdf['is_development'] == False].copy()
    return gdf.explode(index_parts=False).reset_index(drop=True)


def _build_graph_direct(gdf, cycling_speed_kmh=15):

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




def build_network_from_shapefile(path, cycling_speed_kmh=15, snap_tolerance=0.1):
    """

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
    scoring functions can use the correct per-edge built speed, check
    network connectivity, and write a scoring CSV skeleton to
    data/Network/processed/netzluecken_scoring.csv.

    Workflow:
    1. Derive ffs from ROUTENTYP if the column is missing (edges.gpkg lacks it).
    2. Snapshot routentyp_built / ffs_built from the original (pre-override) values.
    3. Override ROUTENTYP → 'Netzlücke' and ffs → BAD_FFS for all development edges.
    4. Recompute tt_min from length/ffs for all edges.
    5. Build graph and call check_network_connectivity — if the network is
       disconnected, small isolated components are dropped and the GDF is
       reset-indexed.  This means ID_edge values in the returned GDF match
       those in edges.gpkg (the source), not a renumbered sequence.
    6. Write netzluecken_scoring.csv with cost estimates for each Netzlücke.

    NOTE: Netzlücken that are not traversed by any OD-optimal path produce zero
    comfort, safety, and travel-time benefits.  This is a valid model result
    indicating the edge does not create a useful shortcut for any demand pair.

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


    far = dists.reshape(h, w) > 2000
    tt_arr[far]  = np.nan
    src_arr[far] = -1.0

    return tt_arr, src_arr


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

