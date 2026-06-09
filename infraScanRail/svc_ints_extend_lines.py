"""
svc_ints_extend_lines — Phase 5B: discover line-extension (EXT) service interventions.
Last modified: 2026-06-07

Auto-discovers rail line extensions on the resolved base infra + service network, the
modern replacement for the legacy ``generate_infrastructure.generate_rail_edges``
endpoint-buffer prototype — with all hardcoded corridor knowledge dropped (no
``[112,113,720,2200]`` node exclusions, no station-name map, no AK2035 reads) and no
defaulted travel time (``apply_svc_int`` derives real infra TT from the projection).

Scope (binding, 2026-06-07 note): an EXT is generated only for a **rail** line
(``mode_class == 'rail'``) that **serves ≥2 stations inside the study-area boundary**
(measured on the line's stop coordinates). Each line endpoint (origin / destination) is
buffered by ``settings.EXT_BUFFER_RADIUS_M``; the nearest ``settings.EXT_MAX_CANDIDATES``
**rail-infra stations** (``Node_Class=='station'`` & ``Transport_Mode`` containing
'train') not already on the line, within the SA buffer, become extension candidates.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import networkx as nx
from shapely.geometry import Point

import paths
import settings
import infrabuild_network_builder as ic
import svc_ints_orchestrator as si

_RAIL_MODE_TOKEN = 'train'


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def discover_ext_candidates(
    base_infra: str,
    base_svc: str,
    sa_polygon=None,
    buffer_polygon=None,
) -> List[Dict]:
    """Discover endpoint-extension candidates for SA-serving rail lines.

    Returns a list of candidate dicts (route_id, variant_rank, from_end, endpoint,
    target, dist_m, line_short_name, line_type, total_dep, layer).
    """
    lines, segs, _stops = si._load_base_unprojected(base_svc)
    nodes, infra_segs = ic.load_version(base_infra)
    rail_stations = _rail_stations(nodes, buffer_polygon)
    # an extension target must be an actually-served rail station (appears as a stop in
    # the base rail service network) — an unserved infra-only station is not a candidate.
    served = set(si._build_stop_index(segs).keys())
    # per-route served-station set (across ALL variants/directions) for the redundancy
    # check — the legacy filter_unnecessary_links rule: don't extend a line to a station
    # its own route already serves.
    served_by_route = _served_by_route(segs)
    # infra rail graph (by station name) for the routing-overlap check
    graph = _rail_graph(nodes, infra_segs)
    radius = float(getattr(settings, 'EXT_BUFFER_RADIUS_M', 10000))
    nmax = int(getattr(settings, 'EXT_MAX_CANDIDATES', 5))
    # whole-day frequency gate: a route is extended only if its departures summed across
    # ALL dir-0 variants reach EXT_MIN_FREQ_DEP_PER_H over the GK window (full-day logic).
    min_freq = float(getattr(settings, 'EXT_MIN_FREQ_DEP_PER_H', 2))
    window_h = float(getattr(settings, 'GK_WINDOW_MIN', 840)) / 60.0
    route_total_dep = _route_total_dep(lines)

    candidates: List[Dict] = []
    seen: set = set()                       # (route_id, from_end, endpoint, target) — dedup variants
    kept_count: Dict = defaultdict(int)     # (route_id, from_end, endpoint) — cap across variants
    termini: set = set()
    n_lines = n_eligible = 0
    low_freq_routes: set = set()
    for layer, ldf in lines.items():
        dir0 = ldf[ldf['direction_id'].astype(str) == '0']
        for _, line in dir0.iterrows():
            if str(line.get('mode_class', 'rail')) != 'rail':
                continue
            n_lines += 1
            rid, vr = str(line['route_id']), int(line['variant_rank'])
            # per-route whole-day frequency gate (counts every variant once via the sum)
            if route_total_dep.get(rid, 0) / window_h < min_freq:
                low_freq_routes.add(rid)
                continue
            _, _, lseg = si._find_target(lines, segs, rid, '0', vr)
            if lseg is None or lseg.empty:
                continue
            seq = si._reconstruct_sequence(lseg)
            if len(seq) < 2:
                continue
            if sa_polygon is not None:
                in_sa = sum(1 for s in seq if sa_polygon.contains(Point(s['E'], s['N'])))
                if in_sa < 2:
                    continue
            n_eligible += 1
            on_line = {s['name'] for s in seq}
            # the original service's OWN served stations (across all its variants/dirs);
            # criterion 3 forbids the extension routing back through any of these.
            route_served = served_by_route.get(rid, on_line)
            for end, ep in (('origin', seq[0]), ('destination', seq[-1])):
                # (1) only extend a terminus that lies INSIDE the study-area boundary
                if sa_polygon is not None and not sa_polygon.contains(Point(ep['E'], ep['N'])):
                    continue
                tkey = (rid, end, ep['name'])
                termini.add(tkey)
                for tname, dist in _stations_in_radius(ep, rail_stations, on_line, served, radius):
                    if tname in route_served:                      # redundancy
                        continue
                    # (3) reject only if the route to the target backtracks through a
                    #     station the ORIGINAL service serves (passing stations served by
                    #     other services is fine — the extension may skip them).
                    if not _route_clear_of_served(graph, ep['name'], tname, route_served):
                        continue
                    ckey = (rid, end, ep['name'], tname)
                    if ckey in seen:                               # (4) dedup across variants
                        continue
                    if kept_count[tkey] >= nmax:                   # cap per route+terminus
                        break
                    seen.add(ckey)
                    kept_count[tkey] += 1
                    candidates.append({
                        'route_id': rid, 'variant_rank': vr, 'from_end': end,
                        'endpoint': ep['name'], 'target': tname, 'dist_m': round(dist, 1),
                        'line_short_name': line.get('line_short_name'),
                        'line_type': int(line.get('line_type', 106)),
                        # whole-day departures summed over the route's variants (the freq
                        # basis); apply_svc_int still derives per-variant TT from the base.
                        'total_dep': int(route_total_dep.get(rid, 0)), 'layer': layer,
                    })
    print(f"  [ext] {n_eligible}/{n_lines} rail lines serve ≥2 SA stations; "
          f"{len(low_freq_routes)} route(s) dropped below {min_freq:.0f} dep/h whole-day; "
          f"{len(termini)} terminus(es) in SA; {len(candidates)} extension candidate(s)")
    return candidates


def discover_and_register(
    base_infra: str,
    base_svc: str,
    sa_polygon=None,
    buffer_polygon=None,
    registry_path: Optional[str] = None,
    network: Optional[str] = None,
) -> Dict:
    """Discover EXT candidates and append them to the ext svc-int registry."""
    cands = discover_ext_candidates(base_infra, base_svc, sa_polygon, buffer_polygon)

    # allocate a contiguous id block once (next_svc_int_id is registry-state based)
    start = si.next_svc_int_id('ext', registry_path, network)
    n0 = int(start.split('_')[1])

    records: List[Dict] = []
    seen: set = set()
    for c in cands:
        # one EXT per (route, end, endpoint, target) — NOT per variant: the extension
        # applies to every variant of the route terminating at endpoint (apply_svc_int).
        key = (c['route_id'], c['from_end'], c['endpoint'], c['target'])
        if key in seen:
            continue
        seen.add(key)
        iid = f"ext_{n0 + len(records)}"
        records.append(_ext_to_record(c, iid, base_infra, base_svc))

    si.append_records('ext', records, registry_path=registry_path, network=network)
    ext_ids = [r['int_id'] for r in records]
    print(f"  [ext] registered {len(ext_ids)} EXT svc-int(s)")
    return {'ext_ids': ext_ids, 'candidates': cands}


# ─────────────────────────────────────────────────────────────────────────────
# Record construction
# ─────────────────────────────────────────────────────────────────────────────

def _ext_to_record(c: Dict, int_id: str, base_infra: str, base_svc: str) -> Dict:
    """Build one EXT svc-int record (an `extend` op; TT derived later at apply time).

    The op carries `endpoint` (the terminus station) so apply_svc_int extends EVERY
    variant of the route that terminates there — keyed by route_id, not variant_rank.
    """
    return {
        'int_id': int_id, 'int_type': 'ext', 'base_authored': base_infra,
        'svc_version': base_svc, 'route_id': c['route_id'], 'direction_id': '0',
        'variant_rank': c['variant_rank'],
        'operations': [{'op': 'extend',
                        'params': {'from_end': c['from_end'], 'endpoint': c['endpoint'],
                                   'stops': [c['target']]}}],
        'total_dep': c['total_dep'], 'line_type': c['line_type'], 'mode_class': 'rail',
        'requires_infra': [],
        'affected_stations': [c['endpoint'], c['target']],
        'affected_services': [c['route_id']],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Spatial helpers
# ─────────────────────────────────────────────────────────────────────────────

def _route_total_dep(base_lines) -> Dict[str, int]:
    """route_id → whole-day departures summed across ALL its dir-0 rail variants.

    Full-day logic: total_dep is additive across a route's variants, so the sum is the
    route's whole-day directional departure count; dividing by GK_WINDOW_MIN/60 gives its
    dep/h. Only dir-0 rows are summed (one direction) to match the directional freq basis.
    """
    out: Dict[str, int] = defaultdict(int)
    for ldf in base_lines.values():
        d0 = ldf[ldf['direction_id'].astype(str) == '0']
        for _, r in d0.iterrows():
            if str(r.get('mode_class', 'rail')) != 'rail':
                continue
            out[str(r['route_id'])] += int(r.get('total_dep', 0) or 0)
    return out


def _served_by_route(base_segs) -> Dict[str, set]:
    """route_id → set of every station name it serves across all variants/directions."""
    out: Dict[str, set] = {}
    for segs in base_segs.values():
        for _, r in segs.iterrows():
            rid = str(r['GTFS_ID'])
            s = out.setdefault(rid, set())
            s.add(str(r['from_stop_name']))
            s.add(str(r['to_stop_name']))
    return out


def _rail_stations(nodes: gpd.GeoDataFrame, buffer_polygon=None) -> gpd.GeoDataFrame:
    """Rail stations (Node_Class station + Transport_Mode train), deduped by Name."""
    sel = nodes[nodes['Node_Class'].astype(str) == 'station']
    if 'Transport_Mode' in sel.columns:
        sel = sel[sel['Transport_Mode'].astype(str).str.contains(
            _RAIL_MODE_TOKEN, case=False, na=False)]
    if buffer_polygon is not None:
        sel = sel[sel.geometry.within(buffer_polygon)]
    sel = sel.drop_duplicates(subset='Name').reset_index(drop=True)
    return sel[['Name', 'geometry']].copy()


def _stations_in_radius(ep, rail_stations, on_line, served, radius) -> List[Tuple[str, float]]:
    """All *served* rail stations within `radius` of `ep` (excl. on-line), nearest first.

    `served` is the set of station names that appear as stops in the base rail service
    network; an unserved infra-only station is never a candidate. Capping to
    EXT_MAX_CANDIDATES happens in the caller, after the routing-overlap filter.
    """
    if rail_stations.empty:
        return []
    p = Point(ep['E'], ep['N'])
    d = rail_stations.geometry.distance(p)
    cand = rail_stations.assign(_d=d)
    cand = cand[(cand['_d'] <= radius)
                & (~cand['Name'].astype(str).isin(on_line))
                & (cand['Name'].astype(str).isin(served))]
    cand = cand.sort_values('_d')
    return [(str(r['Name']), float(r['_d'])) for _, r in cand.iterrows()]


def _rail_graph(nodes, infra_segs):
    """Infra rail graph keyed by station name (Transport_Mode 'train'), for routing."""
    if 'Transport_Mode' in infra_segs.columns:
        rail = infra_segs[infra_segs['Transport_Mode'].astype(str).str.contains(
            _RAIL_MODE_TOKEN, case=False, na=False)]
    else:
        rail = infra_segs
    return ic.build_networkx_graph(nodes, rail)


def _route_clear_of_served(graph, a, b, route_served) -> bool:
    """True iff the infra shortest path a→b never re-enters the original service's stops.

    Endpoints a (terminus) and b (target) are excluded; any intermediate node that the
    ORIGINAL service serves (`route_served`) means the extension backtracks through its
    own route → reject. Stations served only by *other* services are allowed (the
    extension may skip them). Unroutable pairs are rejected (False).
    """
    if a not in graph or b not in graph:
        return False
    try:
        path = nx.shortest_path(graph, a, b, weight='length_m')
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return False
    return not any(n in route_served for n in path[1:-1])


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.chdir(paths.MAIN)
    import ints_core as core

    core.cli_header("infraScanRail — Extended Lines (Phase 5B · EXT discovery)")

    core.cli_step(1, "Base infrastructure network?")
    base = core.cli_pick("Infra version:", core.cli_infra_versions(), core._resolve_base_version())

    core.cli_step(2, "Service network to extend?")
    svc = core.cli_pick("Service version:", core.cli_svc_versions(), core._resolve_svc_version())

    combo = f"{base}__{svc}"
    if si.list_svc_int_ids('ext', network=combo):
        core.cli_step(3, f"Existing EXT registry found for '{combo}' — clear it first?")
        if core.cli_pick_yesno("Clear?", True):
            si.delete_records('ext', si.list_svc_int_ids('ext', network=combo), network=combo)
            print("  cleared ext registry")

    res = discover_and_register(
        base, svc, sa_polygon=core._load_polygon(), buffer_polygon=core._load_buffer())
    print(f"\n=== {len(res['ext_ids'])} EXT svc-int(s) registered ===")
    for c in res['candidates'][:20]:
        print(f"  {c['line_short_name']} ({c['route_id']}) {c['from_end']}: "
              f"{c['endpoint']} → {c['target']} ({c['dist_m']:.0f} m)")

    core.cli_step(4, "Materialise + plot the EXT deltas?")
    if core.cli_pick_yesno("Plots?", getattr(settings, 'PLOT_SVC_INTS', False)):
        sa = core._load_polygon()
        si.plot_ext_candidates(base, svc, res['candidates'], sa_polygon=sa)
        si.materialise_and_plot(base, svc, ext_ids=res['ext_ids'], sa_polygon=sa)
