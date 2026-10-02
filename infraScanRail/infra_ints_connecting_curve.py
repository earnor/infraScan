"""
infra_ints_connecting_curve — Auto-discover and place connecting curves (CC).
Last modified: 2026-06-20

Automates the predecessor's connecting-curve logic on the real base graph
(no hardcoded corridors / AK2035 reads). CC is a **rail-only** intervention: the
routing graph and candidate stations are restricted to the train layer
(Transport_Mode containing 'train'), so reversals never form via tram/other track.
For each pair of neighbouring rail stations with **no direct service** (the NDC
candidates):

  1. trace the infra shortest path between them;
  2. detect the **backtrack node** — the node where travel direction reverses
     (the two path segments leave it in nearly the same direction);
  3. treat the two reversal **arms** (path from the backtrack node back toward each
     station) as polylines and fit the **nearest 300 m tangent circle** between them
     (R = 11.8·v²/(u+u_f), settings A3). The tangent points are the two junctions —
     they land on whichever arm segment fits (so the curve may "walk back" past
     intermediate nodes, e.g. Dübendorf–Dietlikon originates at Chriesbach);
  4. split those two base segments at the junctions, add a 1-track curve between them
     with physics travel time (auto_tt @ design speed, junctions are pass-through),
     and composition from costs_connection_curves.xlsx (tunnel/bridge kept, the
     normal/free-track piece absorbs any length mismatch — decision A).

A curve is registered only when its two branch stations pass the study-area scope
gate (both within the buffer, ≥1 within the boundary). Pairs whose path does not
reverse, or whose curve fails the scope gate, are handed to 5B as pure NDCs.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import fiona
from shapely.geometry import Point, LineString
from shapely.ops import linemerge, substring

import paths
import settings
import infrabuild_network_builder as ic
import ints_core as core   # registry + compose engine + shared plot primitives

_ARC_POINTS = 16
_DETECT_LOOKAHEAD_M = 80.0      # heading look-ahead for backtrack detection
_MAX_PATH_M = 25000.0           # ignore station pairs further apart than this
_TANGENCY_TOL_FRAC = 0.05       # |O−foot| must be within this fraction of R


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def min_curve_radius() -> float:
    """Minimum connecting-curve radius [m] from the cant/speed design constants."""
    v = float(getattr(settings, 'CC_DESIGN_SPEED_KMH', 80.0))
    u = float(getattr(settings, 'CC_CANT_MM', 150.0))
    uf = float(getattr(settings, 'CC_CANT_DEFICIENCY_MM', 100.0))
    return 11.8 * v * v / (u + uf)


def discover_and_register(
    base_version: str,
    svc_version: str,
    sa_polygon=None,
    buffer_polygon=None,
    interactive: bool = False,
    registry_path: Optional[str] = None,
) -> Dict:
    """Discover connecting curves for no-direct-service station pairs and register them.

    Candidate stations are rail stations within the study-area buffer. A curve is
    registered only when its two **branch stations** (the first station on each
    reversal arm) satisfy: both within the buffer, and at least one within the
    study-area boundary. Two-in-buffer (neither in the boundary) → no curve.

    Args:
        sa_polygon: study-area boundary (≥1 branch must be inside).
        buffer_polygon: study-area buffer (both branches must be inside).
        interactive: if True, prompt for missing CC composition via CLI.

    Returns dict(cc_ids=[...], ndc_candidates=[...]); each ndc_candidate carries
    requires_infra (the cc_id when a curve was placed, else empty) for 5B.
    """
    nodes, segs = _load_infra(base_version)
    # CC is a rail-only intervention: restrict the routing graph and candidate
    # stations to the train layer, so reversals never form via tram/other track.
    segs = _rail_segments(segs)
    graph = ic.build_networkx_graph(nodes, segs)
    seg_lookup = _segment_lookup(segs)
    direct = _direct_service_pairs(svc_version, base_version, nodes)
    stations, sa_set = _station_names(nodes, sa_polygon, buffer_polygon)
    buffer_set = set(stations)   # rail stations within the SA buffer
    R = min_curve_radius()
    print(f"  [cc] {len(stations)} candidate stations ({len(sa_set)} in study area), R={R:.0f} m")

    cc_ids: List[str] = []
    ndc_candidates: List[Dict] = []
    # one curve per junction+branch-leg pair. Seed from the EXISTING registry so a
    # re-run reuses the same int_id for an already-registered curve instead of
    # appending a duplicate (idempotent discovery — keyed on the superseded leg pair).
    combo = f"{base_version}__{svc_version}"   # registry partition key (decision H)
    legs_to_ccid: Dict[frozenset, str] = {}
    for iid in core.list_intervention_ids('cc', network=combo):
        meta = core.read_record_meta('cc', iid, network=combo)
        removes = (meta or {}).get('removes_base_rows') or []
        if len(removes) >= 2:
            legs_to_ccid[frozenset(str(r) for r in removes)] = iid

    for i in range(len(stations)):
        for j in range(i + 1, len(stations)):
            s1, s2 = stations[i], stations[j]
            if frozenset((s1, s2)) in direct:
                continue
            if s1 not in graph or s2 not in graph:
                continue
            try:
                path = nx.shortest_path(graph, s1, s2, weight='length_m')
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if len(path) < 3 or _path_length(path, seg_lookup) > _MAX_PATH_M:
                continue
            b_idx = _detect_backtrack(path, seg_lookup)
            if b_idx is None:
                continue

            ndc = {'station_a': s1, 'station_b': s2, 'backtrack_node': path[b_idx],
                   'branch_a': None, 'branch_b': None, 'requires_infra': []}
            resolved = _resolve_curve(path, b_idx, nodes, seg_lookup, R)
            # Scope gate on the curve's BRANCH stations (not the pair endpoints):
            # both branches must lie within the buffer, and at least one within the
            # study-area boundary (two-in-buffer → no curve).
            if resolved is not None and _curve_in_scope(
                    resolved['branch_in'], resolved['branch_out'], sa_set, buffer_set):
                ndc['branch_a'], ndc['branch_b'] = resolved['branch_in'], resolved['branch_out']
                legkey = frozenset([resolved['leg_in_id'], resolved['leg_out_id']])
                if legkey in legs_to_ccid:
                    ndc['requires_infra'] = [legs_to_ccid[legkey]]  # shares an existing curve
                else:
                    int_id = core.next_int_id('cc', 'connecting_curve', registry_path,
                                              network=combo)
                    r_nodes, r_segs, r_comp = _materialize_record(
                        resolved, int_id, base_version, interactive)
                    core.append_records('cc', nodes=r_nodes, segments=r_segs,
                                        composition=r_comp, registry_path=registry_path,
                                        network=combo)
                    legs_to_ccid[legkey] = int_id
                    cc_ids.append(int_id)
                    ndc['requires_infra'] = [int_id]
            ndc_candidates.append(ndc)

    print(f"  [cc] registered {len(cc_ids)} connecting curve(s) for "
          f"{sum(1 for n in ndc_candidates if n['requires_infra'])} reversal NDC(s); "
          f"{sum(1 for n in ndc_candidates if not n['requires_infra'])} clean NDC candidate(s)")
    return {'cc_ids': sorted(cc_ids), 'ndc_candidates': ndc_candidates}


# Connecting-curve diff colours (distinct from the standard green/red infrabuild diff).
_CC_COLOR = '#7e3ff2'        # strong purple: the new curve + junctions (added)
_CC_EDGE = '#3d0a8c'         # darker purple node edge
_CC_SUPERSEDED = '#c9a8f5'   # light purple: existing host segments that got split


def plot_cc_changes(base_version: str, extents=('CA', 'SA')) -> List[str]:
    """Base-vs-(base+all CC) diff figures, with CC additions drawn in purple.

    Reuses the infrabuild diff renderer via the orchestrator's shared glue, so the
    style/content matches the rest of the pipeline; only the added category is
    recoloured. Returns the written file paths.
    """
    ids = core.list_intervention_ids('cc', network=core._combo(base_version))
    if not ids:
        print("  [plot] no CC interventions — skipping CC diff")
        return []
    out_dir = core.plot_out_dir(core._combo(base_version), 'cc')
    return core.render_int_diff(base_version, ids, out_tag='CC',
                                added_label='Connecting curve', added_color=_CC_COLOR,
                                added_edge=_CC_EDGE, out_dir=out_dir, extents=extents,
                                superseded_color=_CC_SUPERSEDED,
                                title='Generated connecting curves')


# ─────────────────────────────────────────────────────────────────────────────
# Candidate inputs
# ─────────────────────────────────────────────────────────────────────────────

_RAIL_MODE_TOKEN = 'train'   # rail = Transport_Mode containing 'train' (excludes tram, cog_railway)


def _rail_segments(segs: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Subset to rail track only (Transport_Mode containing 'train')."""
    if 'Transport_Mode' not in segs.columns:
        return segs
    mask = segs['Transport_Mode'].astype(str).str.contains(_RAIL_MODE_TOKEN, case=False, na=False)
    rail = segs[mask].reset_index(drop=True)
    print(f"  [cc] rail filter: {len(rail)}/{len(segs)} segments are rail ('{_RAIL_MODE_TOKEN}')")
    return rail


def _station_names(nodes: gpd.GeoDataFrame, sa_polygon, buffer_polygon):
    """(candidate station names within the buffer, set of names within the SA boundary).

    Candidates are rail stations only (Transport_Mode containing 'train').
    """
    sel = nodes[nodes['Node_Class'].astype(str) == 'station']
    if 'Transport_Mode' in sel.columns:
        sel = sel[sel['Transport_Mode'].astype(str).str.contains(
            _RAIL_MODE_TOKEN, case=False, na=False)]
    in_buffer = sel[sel.geometry.within(buffer_polygon)] if buffer_polygon is not None else sel
    stations = sorted(in_buffer['Name'].astype(str).unique().tolist())
    if sa_polygon is not None:
        in_sa = sel[sel.geometry.within(sa_polygon)]
        sa_set = set(in_sa['Name'].astype(str).tolist())
    else:
        sa_set = set(stations)
    return stations, sa_set


def _curve_in_scope(branch_a, branch_b, sa_set, buffer_set) -> bool:
    """True if the curve's branch stations qualify under the SA boundary/buffer rule.

    Both branches must be within the buffer; at least one within the boundary.
    (Both-in-boundary and one-boundary+one-buffer pass; two-in-buffer fails.)
    """
    if branch_a not in buffer_set or branch_b not in buffer_set:
        return False
    return (branch_a in sa_set) or (branch_b in sa_set)


def _direct_service_pairs(svc_version, infra_version, nodes) -> set:
    """frozenset{nameA,nameB} for every consecutive-stop pair served directly."""
    nr_to_name = _number_to_name(nodes)
    seg_path = paths.get_projected_services_path(svc_version, infra_version)
    pairs: set = set()
    try:
        layers = fiona.listlayers(seg_path)
    except Exception:
        return pairs
    for layer in layers:
        df = gpd.read_file(seg_path, layer=layer)
        for _, r in df.iterrows():
            a = nr_to_name.get(_as_int(r.get('from_stop_nr')))
            b = nr_to_name.get(_as_int(r.get('to_stop_nr')))
            if a and b and a != b:
                pairs.add(frozenset((a, b)))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Backtrack detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_backtrack(path: List[str], seg_lookup) -> Optional[int]:
    """Index of the path node where travel reverses (smallest leg angle), or None.

    At an interior node the two path segments' 'leaving' directions are compared:
    a small angle between them means the train arrives and departs along nearly the
    same bearing → it must reverse. Returns the index with the clearest reversal.
    """
    threshold = float(getattr(settings, 'CC_BACKTRACK_ANGLE_DEG', 120.0))
    best_idx, best_theta = None, threshold
    for i in range(1, len(path) - 1):
        s_prev = seg_lookup.get(frozenset((path[i - 1], path[i])))
        s_next = seg_lookup.get(frozenset((path[i], path[i + 1])))
        if s_prev is None or s_next is None:
            continue
        a = _leaving_dir(s_prev, path[i])
        b = _leaving_dir(s_next, path[i])
        if a is None or b is None:
            continue
        theta = math.degrees(_angle_between(a, b))
        if theta < best_theta:
            best_theta, best_idx = theta, i
    return best_idx


def _leaving_dir(seg_row, node_name, lookahead=_DETECT_LOOKAHEAD_M):
    """Unit direction the segment heads as it leaves ``node_name`` (look-ahead)."""
    g = _merged_line(seg_row)
    L = g.length
    if L <= 0:
        return None
    coords = list(g.coords)
    if str(seg_row['From_Name']) == str(node_name):
        p0 = np.array(coords[0]); p1 = np.array(g.interpolate(min(lookahead, L)).coords[0])
    elif str(seg_row['To_Name']) == str(node_name):
        p0 = np.array(coords[-1]); p1 = np.array(g.interpolate(max(0.0, L - lookahead)).coords[0])
    else:
        return None
    return _unit(p1 - p0)


# ─────────────────────────────────────────────────────────────────────────────
# Curve record (placement + TT + composition)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_curve(path, b_idx, nodes, seg_lookup, R) -> Optional[Dict]:
    """Resolve placement (arms → tangent circle → legs/fracs/branches). No record yet.

    Cheap enough to run for every NDC pair; the full record (composition, TT, rows)
    is only materialised once per unique curve via _materialize_record.
    """
    inc_names = path[:b_idx + 1][::-1]   # [B, ..., origin]
    out_names = path[b_idx:]             # [B, ..., destination]
    inc_line, inc_segs = _build_arm(inc_names, seg_lookup)
    out_line, out_segs = _build_arm(out_names, seg_lookup)
    if inc_line is None or out_line is None:
        return None
    Bpt = _node_point(nodes, path[b_idx])

    placed = _place_tangent_circle(inc_line, out_line, Bpt, R)
    if placed is None:
        return None
    O, P1, P2, arc, arc_len = placed

    leg_in, frac_in = _segment_for_point(inc_segs, P1)
    leg_out, frac_out = _segment_for_point(out_segs, P2)
    if leg_in is None or leg_out is None or str(leg_in['Segment_ID']) == str(leg_out['Segment_ID']):
        return None

    # the curve connects two BRANCHES; identity + composition key are the first
    # station on each arm (skipping intermediate junctions like '... (Abzw)')
    branch_in = _first_station_on_arm(inc_names[1:], nodes) or inc_names[-1]
    branch_out = _first_station_on_arm(out_names[1:], nodes) or out_names[-1]

    return {
        'P1': P1, 'P2': P2, 'arc': arc, 'arc_len': arc_len, 'nodes': nodes,
        'leg_in_id': str(leg_in['Segment_ID']), 'leg_out_id': str(leg_out['Segment_ID']),
        'frac_in': frac_in, 'frac_out': frac_out,
        'gauge': leg_in.get('Gauge'), 'electrification': leg_in.get('Electrification_Class'),
        'branch_in': branch_in, 'branch_out': branch_out,
    }


def _materialize_record(rv: Dict, int_id: str, base_version: str, interactive: bool = False):
    """Build the (nodes, segments, composition) registry rows for a resolved curve."""
    j1, j2 = f"{int_id}_J1", f"{int_id}_J2"
    arc, arc_len = rv['arc'], rv['arc_len']
    v = float(getattr(settings, 'CC_DESIGN_SPEED_KMH', 80.0))
    tt_stop, tt_pass = ic.auto_tt(arc_len, v, 0)  # junctions pass-through (n_sta=0)

    r_comp = _curve_composition(int_id, j1, j2, arc, arc_len,
                                rv['branch_in'], rv['branch_out'], base_version, interactive)

    import cost_parameters
    cost = _composition_cost(r_comp)
    maint = cost * float(getattr(cost_parameters, 'yearly_maintenance_to_construction_cost_factor', 0.03))

    meta = {
        'int_id': int_id, 'int_type': 'cc', 'int_subtype': 'connecting_curve',
        'base_authored': '', 'removes_base_rows': [rv['leg_in_id'], rv['leg_out_id']],
        'requires': [], 'conflicts_with': [],
        'construction_cost_chf': cost, 'maintenance_cost_annual_chf': maint,
        'length_m': arc_len, 'design_speed_kmh': v,
    }

    tmpl = _junction_template(rv['nodes'])
    # Unique synthetic Number per junction. The projection/routing graph keys on Number,
    # so the two junctions MUST NOT share one (nor collide with the host's base junction):
    # otherwise the curve segment J1->J2 collapses to a self-loop and is dropped, so the
    # arc never gets routed or drawn. Deterministic block per int_id, clear of the BAV
    # range (~8.5e6): cc_0004 -> 9_000_041 / 9_000_042.
    _num_base = 9_000_000 + int(''.join(c for c in int_id if c.isdigit()) or '0') * 10

    def _jn(name, pt, host, frac, number):
        row = tmpl.copy()
        row['Node_ID'] = name; row['Name'] = name; row['Code'] = name[-8:]
        row['Number'] = number
        row['E'] = pt.x; row['N'] = pt.y; row['Node_Class'] = 'junction'
        row['Track_Count'] = pd.NA; row['Platform_Count'] = pd.NA; row['Parent_Node'] = pd.NA
        row['on_segment'] = host; row['split_position_frac'] = frac; row['geometry'] = pt
        return row

    r_nodes = gpd.GeoDataFrame(pd.concat([
        _jn(j1, rv['P1'], rv['leg_in_id'], rv['frac_in'], _num_base + 1),
        _jn(j2, rv['P2'], rv['leg_out_id'], rv['frac_out'], _num_base + 2),
    ], ignore_index=True), crs=core.SWISS_CRS)

    r_segs = gpd.GeoDataFrame({
        'Segment_ID': [f"{int_id}_curve"], 'From_Name': [j1], 'To_Name': [j2],
        'Num_Tracks': [1.0], 'Length': [arc_len],
        'Gauge': [rv['gauge']], 'Electrification_Class': [rv['electrification']],
        # Rail by construction (it joins two rail branches): tag 'train' so the capacity
        # loader's train-segment filter keeps the curve (a missing mode is dropped).
        'Transport_Mode': ['train'],
        'Average_Speed': [v], 'Predominant_Speed': [v], 'Speed_Coverage_Pct': [100.0],
        'TT_Stopping': [tt_stop], 'TT_Passing': [tt_pass], 'speed_source': ['design'],
        'geometry': [arc],
    }, crs=core.SWISS_CRS)

    return (core.attach_metadata(r_nodes, meta),
            core.attach_metadata(r_segs, meta),
            core.attach_metadata(r_comp, meta))


# tunnel/bridge priced at their own rates (F4); unknown structures fall back to track.
_STRUCT_RATE_ATTR = {'tunnel': 'tunnel_cost_per_meter', 'bridge': 'bridge_cost_per_meter'}


def _composition_cost(comp) -> float:
    """Construction cost from the composition rows: Σ piece length × structure rate."""
    import cost_parameters
    base_rate = float(getattr(cost_parameters, 'track_cost_per_meter', 33250.0))
    total = 0.0
    for _, r in comp.iterrows():
        attr = _STRUCT_RATE_ATTR.get(str(r['Engineering_Structure']).strip().lower())
        rate = float(getattr(cost_parameters, attr, base_rate)) if attr else base_rate
        total += float(r['Piece_Length']) * rate
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Geometry: arms + tangent circle
# ─────────────────────────────────────────────────────────────────────────────

def _build_arm(node_names, seg_lookup):
    """Polyline + ordered base-segment rows for a consecutive node sequence."""
    seg_rows, pts = [], []
    for k in range(len(node_names) - 1):
        s = seg_lookup.get(frozenset((node_names[k], node_names[k + 1])))
        if s is None:
            return None, None
        seg_rows.append(s)
        g = _merged_line(s)
        coords = list(g.coords)
        # orient so this piece flows node_names[k] -> node_names[k+1]
        if str(s['From_Name']) != str(node_names[k]):
            coords = coords[::-1]
        if pts and pts[-1] == coords[0]:
            coords = coords[1:]
        pts.extend(coords)
    if len(pts) < 2:
        return None, None
    return LineString(pts), seg_rows


def _place_tangent_circle(arm1, arm2, Bpt, R):
    """Nearest-to-B circle of radius R tangent to both arms; return O,P1,P2,arc,len."""
    best = None
    for o1 in _offset_variants(arm1, R):
        for o2 in _offset_variants(arm2, R):
            it = o1.intersection(o2)
            if it.is_empty:
                continue
            cands = [it] if it.geom_type == 'Point' else \
                [g for g in getattr(it, 'geoms', []) if g.geom_type == 'Point']
            for O in cands:
                P1 = arm1.interpolate(arm1.project(O))
                P2 = arm2.interpolate(arm2.project(O))
                if abs(O.distance(P1) - R) > R * _TANGENCY_TOL_FRAC:
                    continue
                if abs(O.distance(P2) - R) > R * _TANGENCY_TOL_FRAC:
                    continue
                key = O.distance(Bpt) if Bpt is not None else O.distance(P1)
                if best is None or key < best[0]:
                    best = (key, O, P1, P2)
    if best is None:
        return None
    _, O, P1, P2 = best
    Oa = np.array([O.x, O.y])
    arc = _arc_polyline(Oa, np.array([P1.x, P1.y]), np.array([P2.x, P2.y]), R)
    if arc is None:
        return None
    a1 = math.atan2(P1.y - O.y, P1.x - O.x)
    a2 = math.atan2(P2.y - O.y, P2.x - O.x)
    da = abs((a2 - a1 + math.pi) % (2 * math.pi) - math.pi)
    return O, P1, P2, arc, R * da


def _offset_variants(line, R):
    out = []
    for s in (R, -R):
        try:
            o = line.offset_curve(s)
        except Exception:
            continue
        if o.is_empty:
            continue
        out += [o] if o.geom_type == 'LineString' else list(getattr(o, 'geoms', []))
    return out


def _segment_for_point(seg_rows, P):
    """The base segment (of an arm) nearest P, and P's frac from its native start."""
    best = None
    for s in seg_rows:
        g = _merged_line(s)
        d = g.distance(P)
        if best is None or d < best[0]:
            best = (d, s, g)
    if best is None:
        return None, None
    _, s, g = best
    frac = (g.project(P) / g.length) if g.length > 0 else 0.5
    return s, min(max(frac, 0.0), 1.0)


def _arc_polyline(O, P1, P2, R):
    a1 = math.atan2(P1[1] - O[1], P1[0] - O[0])
    a2 = math.atan2(P2[1] - O[1], P2[0] - O[0])
    da = a2 - a1
    while da <= -math.pi:
        da += 2 * math.pi
    while da > math.pi:
        da -= 2 * math.pi
    pts = [(O[0] + R * math.cos(a1 + da * k / _ARC_POINTS),
            O[1] + R * math.sin(a1 + da * k / _ARC_POINTS)) for k in range(_ARC_POINTS + 1)]
    try:
        return LineString(pts)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Composition (manual Excel; tunnel/bridge kept, normal absorbs mismatch)
# ─────────────────────────────────────────────────────────────────────────────

def _curve_composition(int_id, j1, j2, arc, arc_len, branch_in, branch_out, base_version, interactive):
    """Composition rows for the curve from the CC composition cache (decision A).

    Cache hit → use that structure breakdown (tunnel/bridge kept, normal absorbs the
    geometry mismatch). Cache miss + interactive → prompt the user (CLI) and persist.
    Cache miss + non-interactive → single normal piece + warning. The cache is
    partitioned by infra network (``base_version``).
    """
    key = _branch_key(branch_in, branch_out)
    cache = _load_comp_cache(base_version)
    defn = cache.get(key)

    if defn is None and interactive:
        defn = _prompt_composition(branch_in, branch_out, arc_len)
        _save_comp_cache(base_version, key, defn)
    if not defn:
        if defn is None:
            print(f"  [cc]   {int_id}: no cached composition for '{branch_in} - "
                  f"{branch_out}' — using normal-only (run the CLI to enter it)")
        defn = [('normal', arc_len)]

    adj = _reconcile_pieces(defn, arc_len, int_id, branch_in, branch_out)
    rows, cum = [], 0.0
    for struct, plen in adj:
        if plen <= 0.5:
            continue
        sub = substring(arc, cum, min(cum + plen, arc_len))
        rows.append({'Segment_ID': f"{int_id}_curve", 'From_Name': j1, 'To_Name': j2,
                     'Engineering_Structure': struct, 'Edge_Level': 1, 'Under_Construction': 0,
                     'Piece_Length': plen, 'Num_Tracks': 1.0, 'geometry': sub})
        cum += plen
    if not rows:
        rows.append({'Segment_ID': f"{int_id}_curve", 'From_Name': j1, 'To_Name': j2,
                     'Engineering_Structure': 'normal', 'Edge_Level': 1, 'Under_Construction': 0,
                     'Piece_Length': arc_len, 'Num_Tracks': 1.0, 'geometry': arc})
    return gpd.GeoDataFrame(rows, crs=core.SWISS_CRS)


def _reconcile_pieces(defn, arc_len, int_id, a, b):
    """Fit cached pieces to the geometric arc length: keep tunnel/bridge, flex normal."""
    tunnel = sum(l for s, l in defn if s == 'tunnel')
    bridge = sum(l for s, l in defn if s == 'bridge')
    normal_in = sum(l for s, l in defn if s == 'normal')
    normal_target = arc_len - tunnel - bridge

    if normal_target < 0:  # structures exceed the geometric arc → scale them, warn
        sc = arc_len / (tunnel + bridge) if (tunnel + bridge) > 0 else 0.0
        print(f"  [cc]   {int_id} ({a}-{b}): tunnel+bridge ({tunnel + bridge:.0f}m) > arc "
              f"({arc_len:.0f}m); scaling structures by {sc:.2f}")
        return [(s, l * sc) for s, l in defn if s in ('tunnel', 'bridge')]

    if normal_in > 0:
        nsc = normal_target / normal_in
        return [(s, (l * nsc if s == 'normal' else l)) for s, l in defn]

    out = list(defn)
    if normal_target > 0.5:  # geometry longer than the structures → trailing normal
        out.append(('normal', normal_target))
    return out


# ── CC composition cache (CSV; per infra network; replaces costs_connection_curves.xlsx)
# utf-8-sig on read/write so Swiss names (Dübendorf, Zürich) round-trip even if the file
# is opened/re-saved in Excel.

_COMP_CACHE: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}


def _branch_key(a: str, b: str) -> str:
    return '|'.join(sorted([_norm(a), _norm(b)]))


def _load_comp_cache(base_version: str) -> Dict[str, List[Tuple[str, float]]]:
    if base_version in _COMP_CACHE:
        return _COMP_CACHE[base_version]
    cache: Dict[str, List[Tuple[str, float]]] = {}
    _COMP_CACHE[base_version] = cache
    p = Path(paths.get_cc_composition_cache(core._combo(base_version)))
    if not p.exists():
        return cache
    try:
        df = pd.read_csv(p, encoding='utf-8-sig')
    except Exception as exc:
        print(f"  [cc]   composition cache unreadable ({exc})")
        return cache
    for key, grp in df.sort_values('seq').groupby('branch_key'):
        cache[str(key)] = [(str(r['structure']), float(r['length_m']))
                           for _, r in grp.iterrows()]
    return cache


def _save_comp_cache(base_version: str, key: str, pieces: List[Tuple[str, float]]) -> None:
    p = Path(paths.get_cc_composition_cache(core._combo(base_version)))
    p.parent.mkdir(parents=True, exist_ok=True)
    df = (pd.read_csv(p, encoding='utf-8-sig') if p.exists()
          else pd.DataFrame(columns=['branch_key', 'seq', 'structure', 'length_m']))
    df = df[df['branch_key'].astype(str) != key]
    new = pd.DataFrame([{'branch_key': key, 'seq': i + 1, 'structure': s, 'length_m': l}
                        for i, (s, l) in enumerate(pieces)])
    pd.concat([df, new], ignore_index=True).to_csv(p, index=False, encoding='utf-8-sig')
    _COMP_CACHE.pop(base_version, None)  # invalidate this network's cached entry
    print(f"  [cc]   saved composition for '{key}' to {p.name}")


def _prompt_composition(a: str, b: str, arc_len: float) -> List[Tuple[str, float]]:
    """CLI: enter the curve composition in sequence (type + length) until length is filled."""
    print(f"\n[CC composition] No cached composition for '{a} - {b}' "
          f"(curve length {arc_len:.0f} m).")
    print(f"  Enter pieces walking FROM the {a} end TOWARD the {b} end "
          f"(piece 1 starts at {a}).")
    print("  Types: normal / tunnel / bridge.")
    pieces: List[Tuple[str, float]] = []
    remaining, seq = arc_len, 1
    while remaining > 0.5:
        t = (input(f"  Piece {seq} type (normal/tunnel/bridge) [normal]: ").strip().lower()
             or 'normal')
        if t not in ('normal', 'tunnel', 'bridge'):
            print("    invalid type"); continue
        raw = input(f"  Piece {seq} length m (blank = remaining {remaining:.0f}): ").strip()
        try:
            ln = float(raw) if raw else remaining
        except ValueError:
            print("    invalid number"); continue
        ln = min(max(ln, 0.0), remaining)
        if ln <= 0:
            continue
        pieces.append((t, ln)); remaining -= ln; seq += 1
        print(f"    + {t} {ln:.0f} m; remaining {remaining:.0f} m")
    return pieces


def _norm(name: str) -> str:
    s = str(name).lower().strip()
    for tok in ('(abzw)', 'zh', '  '):
        s = s.replace(tok, ' ')
    return ' '.join(s.split())


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

_STATION_CLASS: Dict[str, str] = {}


def _first_station_on_arm(arm_node_names, nodes) -> Optional[str]:
    """First node classed 'station' along an arm (skips junctions like '(Abzw)')."""
    if not _STATION_CLASS:
        for _, r in nodes.iterrows():
            _STATION_CLASS[str(r['Name'])] = str(r.get('Node_Class', ''))
    for name in arm_node_names:
        if _STATION_CLASS.get(str(name)) == 'station':
            return str(name)
    return None


def _segment_lookup(segs: gpd.GeoDataFrame) -> Dict:
    out = {}
    for _, s in segs.iterrows():
        out[frozenset((str(s['From_Name']), str(s['To_Name'])))] = s
    return out


def _path_length(path, seg_lookup) -> float:
    total = 0.0
    for k in range(len(path) - 1):
        s = seg_lookup.get(frozenset((path[k], path[k + 1])))
        if s is not None:
            total += _merged_line(s).length
    return total


def _merged_line(seg_row) -> LineString:
    g = seg_row.geometry
    return linemerge(g) if g.geom_type == 'MultiLineString' else g


def _angle_between(u, v) -> float:
    cu, cv = _unit(np.array(u)), _unit(np.array(v))
    return math.acos(float(np.clip(np.dot(cu, cv), -1.0, 1.0)))


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


_NODE_PT_CACHE: Dict[str, Point] = {}


def _node_point(nodes, name) -> Optional[Point]:
    if str(name) in _NODE_PT_CACHE:
        return _NODE_PT_CACHE[str(name)]
    m = nodes[nodes['Name'].astype(str) == str(name)]
    pt = m.geometry.iloc[0] if not m.empty else None
    if pt is not None:
        _NODE_PT_CACHE[str(name)] = pt
    return pt


def _number_to_name(nodes: gpd.GeoDataFrame) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for _, r in nodes.iterrows():
        n = _as_int(r.get('Number'))
        if n is not None:
            out[n] = str(r['Name'])
    return out


def _as_int(v) -> Optional[int]:
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _junction_template(nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    j = nodes[nodes['Node_Class'] == 'junction']
    src = j if not j.empty else nodes
    return src.iloc[[0]].copy()


def _load_infra(base_version: str) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    nodes, segs = ic.load_version(base_version)
    for _, r in nodes.iterrows():
        _NODE_PT_CACHE[str(r['Name'])] = r.geometry
    return nodes, segs


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI — discover + register CC against the chosen base/service version
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.chdir(paths.MAIN)
    core.cli_header("infraScanRail — Connecting Curves (Phase 5A · CC discovery)")

    core.cli_step(1, "Base infrastructure network?")
    base = core.cli_pick("Infra version:", core.cli_infra_versions(), core._resolve_base_version())

    core.cli_step(2, "Service network (defines existing direct-service pairs)?")
    svc = core.cli_pick("Service version:", core.cli_svc_versions(), core._resolve_svc_version())

    combo = f"{base}__{svc}"
    if core.list_intervention_ids('cc', network=combo):
        core.cli_step(3, f"Existing CC registry found for '{combo}' — clear it first?")
        if core.cli_pick_yesno("Clear?", True):
            core.delete_records('cc', core.list_intervention_ids('cc', network=combo), network=combo)
            print("  cleared cc registry")

    res = discover_and_register(
        base, svc,
        sa_polygon=core._load_polygon(), buffer_polygon=core._load_buffer(),
        interactive=True,   # prompt for any missing CC composition
    )

    print(f"\n=== {len(res['cc_ids'])} connecting curve(s) registered ===")
    for ndc in res['ndc_candidates']:
        if ndc['requires_infra']:
            print(f"  {ndc['requires_infra'][0]}: {ndc['branch_a']} <-> {ndc['branch_b']}")

    core.cli_step(4, "Generate CC change plots?")
    if core.cli_pick_yesno("Plots?", getattr(settings, 'PLOT_INFRA_INTS', False)):
        plot_cc_changes(base)
