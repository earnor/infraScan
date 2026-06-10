"""Phase-6D passenger flows: unroll routed loads onto the real infrastructure.
Last modified: 2026-06-09

Joins the persisted routing primitive (service-link hops, unprojected stop ids)
to the projected service links via station NAMES (the projection re-keys
interchange stations to platform-group Betriebspunkte, e.g. Zürich HB 8503000 →
Zürich HB Löwenstrasse 8516144, so ids do not bridge the two spaces), then
unrolls each link's load over its Via_Segment node-pair chain onto the composed
infra version's segments. Produces per-network infra-segment and node load
tables (local vs passing split) plus flow / diff maps. Replaces the legacy
main_cap Phase 7 flow plotting.

Entry points:
    build_passenger_flows(svc_network, infra_version, method, ...) -> dict
    build_flow_diff(base_network, dev_network, infra_version, method, ...)
Standalone CLI in __main__ (module-CLI convention); main_new passes settings.
"""

import os

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pyogrio
from matplotlib.lines import Line2D
from shapely.geometry import LineString

import cache_manifest
import ints_core
import paths
import settings

CODEBASE_CRS = 'EPSG:2056'

_FLOW_FILES = ('flow_segments.gpkg', 'flow_segments_by_service.csv',
               'flow_nodes.gpkg', 'flow_nodes.csv')


def build_passenger_flows(svc_network: str, infra_version: str, method: str = '',
                          make_plots: bool = True,
                          use_cache: bool = False,
                          links_infra_version: str = '') -> dict:
    """Build the Phase-6D flow tables (+ map) for one routed network.

    Args:
        svc_network:   network folder name WITH the '_network' suffix — the
                       baseline ('AK_2026_S18_network') or a per-svc-int
                       network (combo-keyed, e.g.
                       'Developments/<combo>/ext_100001_network').
        infra_version: infra version whose segments/nodes the loads are
                       unrolled onto — the svc-int's CC-only composed version
                       (real or Developments/Derived), the base for the
                       baseline.
        method:        'shortest_path' | 'logit' ('' -> settings; 'both' falls
                       back to 'logit').
        make_plots:    render the flow map (data outputs always written).
        use_cache:     skip when all four flow outputs already exist.
        links_infra_version: dirname the projected links / 5C merged gpkg live
                       under — apply_svc_int writes them under the BASE infra
                       name even when projected on the composed infra; ''
                       falls back to infra_version (baseline case).

    Returns:
        dict(segments=GeoDataFrame, nodes=GeoDataFrame, trips_total=float,
             n_synthetic=int) — or dict(cached=True) on a cache hit.
    """
    method = method or settings.ROUTING_ASSIGNMENT_METHOD
    if method == 'both':
        method = 'logit'
    flow_dir = paths.get_flow_dir(svc_network, method)
    if (use_cache
            and all(os.path.exists(os.path.join(flow_dir, f))
                    for f in _FLOW_FILES)
            and cache_manifest.check_manifest(
                flow_dir, 'flows_6d',
                {'svc_network': svc_network, 'infra_version': infra_version})):
        print(f"  [flows] cached outputs at {flow_dir} — reuse")
        return {'cached': True}

    print("=" * 70)
    print(f"PHASE 6D PASSENGER FLOWS — {svc_network}")
    print(f"  Method : {method}  |  Infra: {infra_version}")
    print("=" * 70)

    prim_path = paths.get_routing_primitive_path(svc_network, method, 'segments')
    if not os.path.exists(prim_path):
        raise FileNotFoundError(
            f"Routing primitive missing at {prim_path}. Run Phase 4C "
            f"(baseline) or Phase 6C (svc-int) for '{svc_network}'/{method} first.")
    prim_seg = pd.read_parquet(prim_path)
    prim_ev = pd.read_parquet(
        paths.get_routing_primitive_path(svc_network, method, 'events'))
    trips_total = float(prim_seg['trips'].sum())

    # Service-link hops: every primitive row is one consecutive-scheduled-stop
    # hop of its variant ('x' prefix only marks out-of-catchment stations).
    hops = prim_seg.copy()
    hops['from_nr'] = hops['from_id'].astype(str).str.lstrip('x')
    hops['to_nr'] = hops['to_id'].astype(str).str.lstrip('x')
    hops = (hops.groupby(['variant_key', 'from_nr', 'to_nr'], as_index=False)
            ['trips'].sum())
    vk_parts = hops['variant_key'].astype(str).str.rsplit('_', n=2, expand=True)
    hops['route_id'] = vk_parts[0]
    hops['did'] = vk_parts[1]
    hops['vr'] = vk_parts[2]

    stop_names, stop_pts = _stop_lookup(svc_network)
    hops['from_name'] = hops['from_nr'].map(stop_names)
    hops['to_name'] = hops['to_nr'].map(stop_names)

    links = _load_projected_links(svc_network,
                                  links_infra_version or infra_version)
    joined = hops.merge(
        links, how='left',
        left_on=['route_id', 'did', 'vr', 'from_name', 'to_name'],
        right_on=['_rid', '_did', '_vr', 'from_stop_name', 'to_stop_name'])
    n_unmatched = int(joined['Via_Segment'].isna().sum())
    unmatched_trips = float(joined.loc[joined['Via_Segment'].isna(), 'trips'].sum())
    print(f"  hops: {len(hops):,} (variant, link) loads, {trips_total:,.1f} "
          f"trip-legs; {n_unmatched:,} hop(s) with no projected link "
          f"({unmatched_trips:,.1f} trips) -> synthetic")

    seg_by_pair, node_names, node_pts, adjacency, junctions = _infra_lookup(
        infra_version)

    seg_total: dict = {}
    seg_service: dict = {}
    passing_interior: dict = {}
    synthetic_rows = []
    n_empty_via = n_missing_pair = n_walked_pairs = 0
    for r in joined.itertuples(index=False):
        trips, vk = float(r.trips), str(r.variant_key)
        via = '' if (not isinstance(r.Via_Segment, str)) else r.Via_Segment
        if not via:
            if isinstance(r.Via_Segment, str):
                n_empty_via += 1
            synthetic_rows.append((r.from_nr, r.to_nr, r.from_name, r.to_name,
                                   vk, trips))
            continue
        # Resolve each Via pair to composed sub-pairs: a pair authored on the
        # pre-split infra (base links over a CC-split host) is walked through
        # the pass-through junction chain instead of falling synthetic.
        raw_pairs = [tuple(int(t) for t in p.split('-')) for p in via.split('|')]
        pairs = []
        chain_ok = True
        for (u, v) in raw_pairs:
            if (u, v) in seg_by_pair or (v, u) in seg_by_pair:
                pairs.append((u, v))
                continue
            walked = ints_core.walk_segment_chain(adjacency, u, v,
                                                  pass_nodes=junctions)
            if walked is None:
                n_missing_pair += 1
                chain_ok = False
                break
            n_walked_pairs += 1
            pairs.extend(zip(walked[:-1], walked[1:]))
        if not chain_ok:
            synthetic_rows.append((r.from_nr, r.to_nr, r.from_name, r.to_name,
                                   vk, trips))
            continue
        for (u, v) in pairs:
            sid = seg_by_pair.get((u, v)) or seg_by_pair.get((v, u))
            seg_total[sid] = seg_total.get(sid, 0.0) + trips
            seg_service[(sid, vk)] = seg_service.get((sid, vk), 0.0) + trips
        for n in [p[0] for p in pairs[1:]]:        # interior chain nodes
            passing_interior[n] = passing_interior.get(n, 0.0) + trips

    segments_gdf = _build_segments_gdf(seg_total, infra_version, synthetic_rows,
                                       stop_pts)
    by_service = pd.DataFrame(
        [{'Segment_ID': sid, 'variant_key': vk, 'trips': t}
         for (sid, vk), t in sorted(seg_service.items())])
    if not by_service.empty:
        by_service['route_id'] = (by_service['variant_key'].astype(str)
                                  .str.rsplit('_', n=2).str[0])

    nodes_gdf = _build_nodes_gdf(hops, prim_ev, passing_interior, stop_names,
                                 stop_pts, node_names, node_pts)

    os.makedirs(flow_dir, exist_ok=True)
    segments_gdf.to_file(os.path.join(flow_dir, 'flow_segments.gpkg'),
                         driver='GPKG')
    by_service.to_csv(os.path.join(flow_dir, 'flow_segments_by_service.csv'),
                      index=False, encoding='utf-8-sig')
    nodes_gdf.to_file(os.path.join(flow_dir, 'flow_nodes.gpkg'), driver='GPKG')
    pd.DataFrame(nodes_gdf.drop(columns='geometry')).to_csv(
        os.path.join(flow_dir, 'flow_nodes.csv'), index=False,
        encoding='utf-8-sig')
    print(f"  flow tables -> {flow_dir}")
    cache_manifest.write_manifest(flow_dir, 'flows_6d',
                                  {'svc_network': svc_network,
                                   'infra_version': infra_version})

    # Reconciliation: every hop's trips land exactly once per traversed infra
    # segment, so the conserved quantity is hop trips (unrolled + synthetic),
    # not the per-segment sum (that is a passenger-km-like measure).
    syn_trips = float(sum(t for *_x, t in synthetic_rows))
    unrolled_trips = float(joined['trips'].sum()) - syn_trips
    print(f"  reconciliation: unrolled {unrolled_trips:,.1f} + synthetic "
          f"{syn_trips:,.1f} = {unrolled_trips + syn_trips:,.1f} hop trips "
          f"(primitive total {trips_total:,.1f})")
    print(f"  synthetic legs: {len(synthetic_rows):,} "
          f"(empty Via_Segment: {n_empty_via:,}, no projected link: "
          f"{n_unmatched:,}, unmatched infra pair: {n_missing_pair:,}); "
          f"{n_walked_pairs:,} pair(s) walked onto split sections")

    if make_plots:
        _plot_flow_map(segments_gdf, svc_network, method)

    return {'segments': segments_gdf, 'nodes': nodes_gdf,
            'trips_total': trips_total, 'n_synthetic': len(synthetic_rows)}


def build_flow_diff(base_network: str, dev_network: str, infra_version: str,
                    method: str = '', make_plots: bool = True,
                    dev_infra_version: str = '') -> None:
    """Diff the developed network's infra-segment loads against the baseline.

    Writes flow_segments_diff.gpkg under the DEV network's flow dir and (gated)
    renders the red/green diff map: green = load increase, red = decrease,
    width ∝ |Δ|, synthetic legs dashed — the legacy main_cap Phase-7 semantics.

    Args:
        infra_version:     the BASELINE's infra version.
        dev_infra_version: the dev network's (composed) infra version when it
            differs — baseline host segments split by a CC/CAP are then
            re-keyed onto the dev split pieces before the join, so the diff
            lands at piece granularity instead of a spurious host decrease
            plus piece increases.
    """
    method = method or settings.ROUTING_ASSIGNMENT_METHOD
    if method == 'both':
        method = 'logit'
    base = gpd.read_file(paths.get_flow_table_path(base_network, method,
                                                   'flow_segments.gpkg'))
    dev = gpd.read_file(paths.get_flow_table_path(dev_network, method,
                                                  'flow_segments.gpkg'))
    if dev_infra_version and dev_infra_version != infra_version:
        base = _replicate_base_onto_split_pieces(base, infra_version,
                                                 dev_infra_version)
    key = 'Segment_ID'
    merged = dev.merge(
        pd.DataFrame(base.drop(columns='geometry'))[[key, 'trips']]
        .rename(columns={'trips': 'trips_base'}), on=key, how='outer')
    base_only = merged['geometry'].isna()
    if base_only.any():
        geo = base.set_index(key)['geometry']
        merged.loc[base_only, 'geometry'] = (
            merged.loc[base_only, key].map(geo).values)
        merged.loc[base_only, 'synthetic'] = (
            merged.loc[base_only, key].map(
                base.set_index(key)['synthetic']).values)
    merged['trips'] = merged['trips'].fillna(0.0)
    merged['trips_base'] = merged['trips_base'].fillna(0.0)
    merged['delta'] = merged['trips'] - merged['trips_base']
    merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=dev.crs)

    out = paths.get_flow_table_path(dev_network, method,
                                    'flow_segments_diff.gpkg')
    merged.to_file(out, driver='GPKG')
    changed = merged[merged['delta'].abs() > 1e-6]
    print(f"  [flow diff] {dev_network} vs {base_network}: "
          f"{len(changed):,} of {len(merged):,} infra segment(s) changed "
          f"(max +{merged['delta'].max():,.1f} / {merged['delta'].min():,.1f}) "
          f"-> {out}")

    if make_plots:
        _plot_diff_map(merged, base_network, dev_network, method)


# ===============================================================================
# LOADERS
# ===============================================================================

def _replicate_base_onto_split_pieces(base: gpd.GeoDataFrame, base_infra: str,
                                      dev_infra: str) -> gpd.GeoDataFrame:
    """Re-key baseline host-segment flows onto the dev network's split pieces.

    The dev flow table is unrolled on the composed infra, whose CC/CAP splits
    replace a base host segment with pieces under new Segment_IDs; an outer
    join would then show a spurious full decrease on the host plus full
    increases on the pieces. Baseline host trips replicate losslessly onto
    every piece (base trains traverse the whole host), so the diff lands at
    piece granularity (decision 2026-06-10). Hosts without a pass-through
    piece chain stay unchanged.
    """
    seg_base, _names_b, _pts_b, _adj_b, _jn_b = _infra_lookup(base_infra)
    seg_dev, _names_d, _pts_d, adj_dev, jn_dev = _infra_lookup(dev_infra)
    parent_pieces: dict = {}
    for (a, b), sid in sorted(seg_base.items()):
        if (a, b) in seg_dev or (b, a) in seg_dev:
            continue
        walked = ints_core.walk_segment_chain(adj_dev, a, b,
                                              pass_nodes=jn_dev)
        if walked is None:
            continue
        parent_pieces[sid] = [seg_dev.get((u, v)) or seg_dev.get((v, u))
                              for u, v in zip(walked[:-1], walked[1:])]
    hosts = base['Segment_ID'].isin(parent_pieces)
    if not hosts.any():
        return base
    piece_geom = gpd.read_file(
        os.path.join(paths.resolve_infra_dir(dev_infra), 'segments.gpkg')
    ).set_index('Segment_ID')['geometry']
    rows = []
    for _, r in base[hosts].iterrows():
        for psid in parent_pieces[r['Segment_ID']]:
            nr = r.copy()
            nr['Segment_ID'] = psid
            if psid in piece_geom.index:
                nr['geometry'] = piece_geom[psid]
            rows.append(nr)
    out = pd.concat([base[~hosts], pd.DataFrame(rows)], ignore_index=True)
    print(f"  [flow diff] split mapping: {int(hosts.sum())} base host "
          f"segment(s) re-keyed onto {len(rows)} split piece row(s)")
    return gpd.GeoDataFrame(out, geometry='geometry', crs=base.crs)


def _load_projected_links(svc_network: str, infra_version: str) -> pd.DataFrame:
    """All projected service links of a network as one frame with string join
    keys (_rid/_did/_vr). Per-svc-int networks use the 5C merged projection
    (rail_segments_merged.gpkg = base + delta); the baseline uses
    rail_segments.gpkg."""
    base_path = os.path.join(paths.MAIN, paths.RAIL_LINES_DIR, svc_network,
                             infra_version, 'rail_segments.gpkg')
    merged_path = os.path.join(os.path.dirname(base_path),
                               'rail_segments_merged.gpkg')
    path = merged_path if os.path.exists(merged_path) else base_path
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Projected service links missing at {path}. Run the service "
            f"projection (Phase 3B) / Phase 5C merged services first.")
    frames = []
    for layer, _ in pyogrio.list_layers(path):
        g = gpd.read_file(path, layer=layer)
        frames.append(pd.DataFrame(g.drop(columns='geometry')))
    df = pd.concat(frames, ignore_index=True)
    df['_rid'] = df['GTFS_ID'].astype(str)
    df['_did'] = df['direction_id'].astype(str).str.replace(
        r'\.0$', '', regex=True)
    df['_vr'] = df['variant_rank'].astype(str).str.replace(
        r'\.0$', '', regex=True)
    keep = ['_rid', '_did', '_vr', 'from_stop_name', 'to_stop_name',
            'Via_Segment', 'from_stop_E', 'from_stop_N', 'to_stop_E',
            'to_stop_N']
    df = df[[c for c in keep if c in df.columns]]
    # merged files can carry a link twice (base survivors + delta); keep one
    df = df.drop_duplicates(subset=['_rid', '_did', '_vr', 'from_stop_name',
                                    'to_stop_name'])
    print(f"  projected links: {len(df):,} from {os.path.basename(path)}")
    return df


def _stop_lookup(svc_network: str) -> tuple:
    """(nr -> name, nr -> Point) from the network's unprojected stops; per-svc-
    int networks read Merged/rail_stops.gpkg (base + delta-only stops)."""
    merged = os.path.join(paths.MAIN, paths.RAIL_LINES_DIR, svc_network,
                          'Merged', 'rail_stops.gpkg')
    unproj = os.path.join(paths.MAIN, paths.RAIL_LINES_DIR, svc_network,
                          paths.SERVICES_UNPROJECTED_SUBDIR, 'rail_stops.gpkg')
    st = gpd.read_file(merged if os.path.exists(merged) else unproj)
    st = st.to_crs(CODEBASE_CRS) if st.crs else st
    nrs = st['Number'].astype(str).str.replace(r'\.0$', '', regex=True)
    names = dict(zip(nrs, st['stop_name'].astype(str)))
    pts = dict(zip(nrs, st.geometry))
    return names, pts


def _infra_lookup(infra_version: str) -> tuple:
    """(seg_by_pair, node_names, node_pts, adjacency, junctions) from an infra
    version (real or composed — Developments/Derived resolved transparently).

    seg_by_pair keys each segment's BAV node pair (the 'Number' column
    'from_to'; lookups try both orientations). CC/CAP split pieces carry no
    'Number', so segments without a parsable pair fall back to
    From_Name/To_Name -> node Number (the capacity loader's resolution).
    adjacency is the undirected node graph of all keyed pairs and junctions
    the junction-class node set — together they drive the split-chain walk
    (a CC wye junction is degree-3 here, so degree alone cannot identify it
    as pass-through).
    """
    d = paths.resolve_infra_dir(infra_version)
    nodes = gpd.read_file(os.path.join(d, 'nodes.gpkg'))
    node_names = {int(n): str(nm) for n, nm in zip(nodes['Number'],
                                                   nodes['Name'])}
    node_pts = {int(n): g for n, g in zip(nodes['Number'], nodes.geometry)}
    if 'Node_Class' in nodes.columns:
        junctions = {int(n) for n, c in zip(nodes['Number'],
                                            nodes['Node_Class'])
                     if str(c) == 'junction'}
    else:
        junctions = set()
    name_to_nr: dict = {}
    for n, nm in node_names.items():
        name_to_nr.setdefault(nm, n)

    seg = gpd.read_file(os.path.join(d, 'segments.gpkg'))
    seg_by_pair: dict = {}
    n_name_fallback = 0
    for r in seg.itertuples(index=False):
        num = str(getattr(r, 'Number', ''))
        a = b = None
        if '_' in num:
            sa, sb = num.split('_', 1)
            try:
                a, b = int(sa), int(sb)
            except ValueError:
                a = b = None
        if a is None:
            a = name_to_nr.get(str(getattr(r, 'From_Name', '')))
            b = name_to_nr.get(str(getattr(r, 'To_Name', '')))
            if a is None or b is None or a == b:
                continue
            n_name_fallback += 1
        seg_by_pair[(a, b)] = r.Segment_ID
    adjacency: dict = {}
    for (a, b) in seg_by_pair:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    print(f"  infra '{infra_version}': {len(seg_by_pair):,} segment pair keys "
          f"({n_name_fallback} via name fallback), {len(node_names):,} nodes "
          f"({len(junctions)} junction-class)")
    return seg_by_pair, node_names, node_pts, adjacency, junctions


# ===============================================================================
# TABLE BUILDERS
# ===============================================================================

def _build_segments_gdf(seg_total: dict, infra_version: str, synthetic_rows,
                        stop_pts) -> gpd.GeoDataFrame:
    """Infra-segment flow table: real segments with totals + synthetic straight
    legs ('SYN_<from>_<to>', dashed in maps)."""
    d = paths.resolve_infra_dir(infra_version)
    seg = gpd.read_file(os.path.join(d, 'segments.gpkg'))
    seg = seg[['Segment_ID', 'From_Name', 'To_Name', 'geometry']].copy()
    seg['trips'] = seg['Segment_ID'].map(seg_total)
    seg = seg[seg['trips'].notna()].copy()
    seg['synthetic'] = False

    syn_agg: dict = {}
    for fr, to, fr_name, to_name, _vk, trips in synthetic_rows:
        k = (str(fr), str(to))
        if k in syn_agg:
            syn_agg[k]['trips'] += trips
        else:
            syn_agg[k] = {'fr_name': fr_name, 'to_name': to_name,
                          'trips': trips}
    syn_records = []
    for (fr, to), v in syn_agg.items():
        p1, p2 = stop_pts.get(fr), stop_pts.get(to)
        if p1 is None or p2 is None:
            continue
        syn_records.append({'Segment_ID': f'SYN_{fr}_{to}',
                            'From_Name': v['fr_name'], 'To_Name': v['to_name'],
                            'trips': v['trips'], 'synthetic': True,
                            'geometry': LineString([p1, p2])})
    if syn_records:
        seg = pd.concat([seg, gpd.GeoDataFrame(syn_records, crs=seg.crs)],
                        ignore_index=True)
    return gpd.GeoDataFrame(seg, geometry='geometry', crs=CODEBASE_CRS)


def _build_nodes_gdf(hops, prim_ev, passing_interior, stop_names, stop_pts,
                     node_names, node_pts) -> gpd.GeoDataFrame:
    """Node flow table with the local/passing split.

    local   = board / alight / transfer events at the station.
    passing = riders arriving at a scheduled stop without alighting or
              transferring (arrivals − alight − transfer) + traversals of
              interior Via_Segment nodes (junctions, skipped stations).
    Station rows bridge to BAV nodes by name; interior-only nodes get rows with
    zero local columns.
    """
    ev = prim_ev.copy()
    ev['nr'] = ev['station_id'].astype(str).str.lstrip('x')
    local = (ev.pivot_table(index='nr', columns='event', values='trips',
                            aggfunc='sum', fill_value=0.0)
             .reindex(columns=['board', 'alight', 'transfer'], fill_value=0.0))

    arrivals = hops.groupby('to_nr')['trips'].sum()
    tab = local.join(arrivals.rename('arrivals'), how='outer').fillna(0.0)
    tab['through'] = (tab['arrivals'] - tab['alight'] - tab['transfer']).clip(
        lower=0.0)
    tab = tab.reset_index().rename(columns={'index': 'nr'})
    if 'nr' not in tab.columns:
        tab = tab.rename(columns={tab.columns[0]: 'nr'})
    tab['name'] = tab['nr'].map(stop_names)

    name_to_bav = {}
    for n, nm in node_names.items():
        name_to_bav.setdefault(nm, n)
    tab['bav_nr'] = tab['name'].map(name_to_bav)

    interior = pd.Series(passing_interior, name='interior', dtype=float)
    tab = tab.merge(interior.rename_axis('bav_nr').reset_index(),
                    on='bav_nr', how='outer')
    tab['interior'] = tab['interior'].fillna(0.0)
    for c in ('board', 'alight', 'transfer', 'through'):
        tab[c] = tab[c].fillna(0.0)
    # interior-only rows (junctions etc.): fill identity from the BAV node
    no_name = tab['name'].isna() & tab['bav_nr'].notna()
    tab.loc[no_name, 'name'] = tab.loc[no_name, 'bav_nr'].map(node_names)
    tab['passing'] = tab['through'] + tab['interior']
    tab['total'] = tab[['board', 'alight', 'transfer', 'passing']].sum(axis=1)
    tab = tab[tab['total'] > 1e-9].copy()

    def _geom(row):
        if pd.notna(row['bav_nr']) and int(row['bav_nr']) in node_pts:
            return node_pts[int(row['bav_nr'])]
        return stop_pts.get(str(row['nr']))

    tab['geometry'] = tab.apply(_geom, axis=1)
    tab = tab[tab['geometry'].notna()].copy()
    cols = ['nr', 'bav_nr', 'name', 'board', 'alight', 'transfer', 'through',
            'interior', 'passing', 'total', 'geometry']
    return gpd.GeoDataFrame(tab[cols], geometry='geometry', crs=CODEBASE_CRS)


# ===============================================================================
# MAPS
# ===============================================================================

def _plot_geom(ax, geom, **kw) -> None:
    """Plot a LineString or MultiLineString."""
    if geom is None:
        return
    parts = geom.geoms if geom.geom_type == 'MultiLineString' else [geom]
    for part in parts:
        ax.plot(*part.xy, **kw)


def _plot_flow_map(segments: gpd.GeoDataFrame, svc_network: str,
                   method: str) -> None:
    """Per-network flow map: width ∝ load on real geometry, synthetic dashed."""
    fig, ax = plt.subplots(figsize=(14, 12))
    vmax = max(float(segments['trips'].max()), 1.0)
    for _, r in segments.iterrows():
        lw = 0.3 + 5.5 * (r['trips'] / vmax)
        _plot_geom(ax, r.geometry, color='#1f6fb4',
                   linewidth=lw, alpha=0.85,
                   linestyle='--' if r['synthetic'] else '-')
    ax.set_title(f'Passenger flows — {svc_network} ({method}, full day)',
                 fontsize=14)
    ax.set_aspect('equal')
    ax.set_axis_off()
    handles = [Line2D([0], [0], color='#1f6fb4', lw=3, label=f'{vmax:,.0f} trips'),
               Line2D([0], [0], color='#1f6fb4', lw=3, linestyle='--',
                      label='synthetic leg')]
    ax.legend(handles=handles, loc='lower right', fontsize=9)
    out_dir = paths.get_flow_plot_dir(svc_network, method)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'flow_map.png')
    fig.savefig(out, bbox_inches='tight', dpi=180)
    plt.close(fig)
    print(f"  flow map -> {out}")


def _plot_diff_map(merged: gpd.GeoDataFrame, base_network: str,
                   dev_network: str, method: str) -> None:
    """Red/green diff map: green = increase, red = decrease, width ∝ |Δ|."""
    fig, ax = plt.subplots(figsize=(14, 12))
    unchanged = merged[merged['delta'].abs() <= 1e-6]
    for _, r in unchanged.iterrows():
        _plot_geom(ax, r.geometry, color='#cccccc', linewidth=0.4, alpha=0.6)
    changed = merged[merged['delta'].abs() > 1e-6]
    dmax = max(float(changed['delta'].abs().max()), 1.0) if len(changed) else 1.0
    for _, r in changed.iterrows():
        lw = 0.6 + 5.5 * (abs(r['delta']) / dmax)
        _plot_geom(ax, r.geometry,
                   color='#2ca02c' if r['delta'] > 0 else '#d62728',
                   linewidth=lw, alpha=0.9,
                   linestyle='--' if bool(r.get('synthetic')) else '-')
    ax.set_title(f'Flow diff — {dev_network} vs {base_network} '
                 f'({method}, full day)', fontsize=14)
    ax.set_aspect('equal')
    ax.set_axis_off()
    handles = [
        Line2D([0], [0], color='#2ca02c', lw=3, label=f'+{dmax:,.0f} trips'),
        Line2D([0], [0], color='#d62728', lw=3, label=f'-{dmax:,.0f} trips'),
        Line2D([0], [0], color='#999999', lw=2, linestyle='--',
               label='synthetic leg'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=9)
    out_dir = paths.get_flow_plot_dir(dev_network, method)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'flow_diff_vs_{base_network}.png')
    fig.savefig(out, bbox_inches='tight', dpi=180)
    plt.close(fig)
    print(f"  diff map -> {out}")


if __name__ == '__main__':
    _svc = settings.SVC_VERSION
    if _svc == 'Build_New':
        _svc = settings.SVC_BUILD_NEW_NAME
    _net = input(f"Network (with _network suffix) [{_svc}_network]: ").strip() \
        or f'{_svc}_network'
    _infra = input(f"Infra version [{settings.INFRA_VERSION}]: ").strip() \
        or settings.INFRA_VERSION
    _m_def = settings.ROUTING_ASSIGNMENT_METHOD
    _m_def = 'logit' if _m_def == 'both' else _m_def
    _method = input(f"Assignment method [{_m_def}]: ").strip() or _m_def
    _p_def = 'y' if settings.PLOT_FLOWS else 'n'
    _plots = (input(f"Generate plots? [{_p_def}]: ").strip() or _p_def) == 'y'
    build_passenger_flows(_net, _infra, method=_method, make_plots=_plots,
                          use_cache=False)
    _diff = input("Diff against a base network (blank = skip): ").strip()
    if _diff:
        build_flow_diff(_diff, _net, _infra, method=_method, make_plots=_plots)
