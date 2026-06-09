"""
svc_ints_new_direct_connections — Phase 5B: build new-direct-connection (NDC) svc-ints.
Last modified: 2026-06-07

Turns the connecting-curve candidates handed over by Phase 5A
(``infra_ints_connecting_curve.discover_and_register`` → ``ndc_candidates``) into NDC
service interventions: a new through-service that rides one constituent line to its
terminus, crosses the curve, and continues on the other constituent line to its
terminus. The modern replacement for the legacy
``generate_infrastructure.generate_new_railway_lines`` path-to-termini prototype, run on
the **composed** (base + CC) rail graph with no hardcoded corridors / AK2035 reads.

Frequency: the through-service can only run as often as its scarcer leg, so its
``total_dep`` is a rule (``settings.NDC_FREQ_RULE``: min/max/mean) over the two
constituent services. Real travel time comes later from ``apply_svc_int``'s projection.

Scope (decision 9): rail-only constituents; the NDC must keep ≥2 stops inside the
study-area boundary (5A's curve scope gate already rail-restricts the candidates).
"""

from typing import Dict, List, Optional

import networkx as nx
import pandas as pd
from shapely.geometry import Point

import paths
import settings
import infrabuild_network_builder as ic
import ints_core as core
import infra_ints_connecting_curve as cc
import svc_ints_orchestrator as si

_RAIL_MODE_TOKEN = 'train'


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_ndc_records(
    ndc_candidates: List[Dict],
    base_infra: str,
    base_svc: str,
    sa_polygon=None,
    network: Optional[str] = None,
) -> List[Dict]:
    """Build NDC svc-int records from 5A connecting-curve candidates.

    Args:
        ndc_candidates: list from infra_ints_connecting_curve (branch_a/b, requires_infra).
        base_infra/base_svc: resolved base versions.
        sa_polygon: study-area boundary (≥2 NDC stops must fall inside it).
        network: registry combo for id allocation; must match where the records are
            written so ids stay contiguous within the target combo (not the settings default).

    Returns:
        a list of NDC record dicts (deduped by stop set; off-scope dropped).
    """
    lines, segs, _ = si._load_base_unprojected(base_svc)
    line_seqs = _line_sequences(lines, segs)
    stop_index = si._build_stop_index(segs)
    # end stations = the termini of existing rail lines (an NDC runs terminus→terminus).
    end_stations = set()
    for s in line_seqs:
        if s['names']:
            end_stations.add(s['names'][0])
            end_stations.add(s['names'][-1])
    cap = int(getattr(settings, 'NDC_MAX_TERMINI_PER_END', 2))

    # Infra-only (Q4 2026-06-08): only curve-requiring candidates produce NDCs; the
    # pseudo-clean ones (reversal pairs without a placed curve) are dropped. Genuine
    # no-reversal missing connections are a separate, deferred discovery.
    # Build PER CURVE from its two branch stations (not per OD pair): from each branch
    # enumerate up to `cap` termini outward (away from the curve) and combine — this is
    # the legacy generate_new_railway_lines combinatorial set, bounded.
    curves: Dict[str, Tuple[str, str]] = {}
    for c in ndc_candidates:
        ba, bb = c.get('branch_a'), c.get('branch_b')
        for cid in (c.get('requires_infra') or []):
            if cid not in curves and ba and bb:
                curves[cid] = (ba, bb)

    graph_cache: Dict = {}
    n0 = int(si.next_svc_int_id('ndc', network=network).split('_')[1])
    records: List[Dict] = []
    seen: set = set()
    for cid in sorted(curves):
        ba, bb = curves[cid]
        G, seg_lookup = _composed_rail_graph(base_infra, [cid], graph_cache)
        if ba not in G or bb not in G:
            continue
        try:
            core = nx.shortest_path(G, ba, bb, weight='length_m')   # crosses the curve
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        fcore = set(core)
        arms_a = _termini_options(G, ba, fcore, end_stations, cap)
        arms_b = _termini_options(G, bb, fcore, end_stations, cap)
        svc_a, svc_b = _service_through(ba, line_seqs), _service_through(bb, line_seqs)
        line_type = int((svc_a or svc_b or {}).get('line_type', 109))
        constituents = [s for s in ((svc_a or {}).get('route_id'),
                                    (svc_b or {}).get('route_id')) if s]
        n_curve = n_reject = 0
        for pa in arms_a:
            for pb in arms_b:
                if set(pa) & set(pb):                       # arms must not overlap
                    continue
                nodes_seq = list(reversed(pa))[:-1] + core + pb[1:]
                # reject if the full routed path backtracks anywhere OTHER than the
                # placed curve — that reversal would itself require an un-placed CC
                # (e.g. Dietlikon→Bassersdorf, Kemptthal→Winterthur Töss).
                if cc._detect_backtrack(nodes_seq, seg_lookup) is not None:
                    n_reject += 1
                    continue
                seq = _served_seq(nodes_seq, stop_index)
                if len(seq) < 2:
                    continue
                # (relaxed 2026-06-08) the NDC line need only touch the SA once
                if sa_polygon is not None and \
                        sum(1 for n in seq if _in_sa(n, stop_index, sa_polygon)) < 1:
                    continue
                key = frozenset(seq)
                if key in seen:
                    continue
                seen.add(key)
                iid = f"ndc_{n0 + len(records)}"
                records.append(_ndc_to_record(
                    iid, seq, _ndc_frequency(), line_type, [cid],
                    base_infra, base_svc, constituents))
                n_curve += 1
        print(f"  [ndc]   {cid} ({ba}–{bb}): {len(arms_a)}×{len(arms_b)} termini → "
              f"{n_curve} line(s) ({n_reject} rejected: route needs another curve)")

    print(f"  [ndc] {len(curves)} curve(s) → {len(records)} NDC svc-int(s) "
          f"(infra-only, ≤{cap} termini/end, no extra-curve reversals, ≥1 SA stop)")
    return records


def discover_and_register(
    base_infra: str,
    base_svc: str,
    ndc_candidates: List[Dict],
    sa_polygon=None,
    registry_path: Optional[str] = None,
    network: Optional[str] = None,
) -> Dict:
    """Build NDC records from 5A candidates and append them to the ndc registry."""
    records = build_ndc_records(ndc_candidates, base_infra, base_svc, sa_polygon, network=network)
    si.append_records('ndc', records, registry_path=registry_path, network=network)
    ndc_ids = [r['int_id'] for r in records]
    print(f"  [ndc] registered {len(ndc_ids)} NDC svc-int(s)")
    return {'ndc_ids': ndc_ids, 'records': records}


# ─────────────────────────────────────────────────────────────────────────────
# Frequency rule + record
# ─────────────────────────────────────────────────────────────────────────────

def _ndc_frequency(svc_a: Optional[Dict] = None, svc_b: Optional[Dict] = None) -> int:
    """Standardised NDC total_dep from a fixed dep/h over the whole-day window.

    Every NDC runs at settings.NDC_FREQ_DEP_PER_H departures/hour; total_dep =
    dep/h × (GK_WINDOW_MIN / 60). The constituent services no longer set the frequency.
    """
    dep_h = float(getattr(settings, 'NDC_FREQ_DEP_PER_H', 2))
    window_h = float(getattr(settings, 'GK_WINDOW_MIN', 840)) / 60.0
    return int(round(dep_h * window_h))


def _ndc_to_record(int_id, seq_names, total_dep, line_type, req, base_infra,
                   base_svc, constituents) -> Dict:
    return {
        'int_id': int_id, 'int_type': 'ndc', 'base_authored': base_infra,
        'svc_version': base_svc, 'route_id': int_id, 'direction_id': '0', 'variant_rank': 1,
        'operations': [{'op': 'new_line',
                        'params': {'stops': seq_names, 'total_dep': total_dep,
                                   'line_type': line_type, 'requires_infra': req}}],
        'total_dep': total_dep, 'line_type': line_type, 'mode_class': 'rail',
        'requires_infra': req, 'affected_stations': seq_names,
        'affected_services': constituents,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Through-sequence on the composed rail graph (path-to-termini)
# ─────────────────────────────────────────────────────────────────────────────

def _termini_options(G, start, forbidden, end_stations, cap) -> List[List[str]]:
    """Up to `cap` outward paths [start, …, terminus] to the nearest distinct end stations.

    Reproduces the legacy generate_new_railway_lines enumeration: instead of a single
    nearest terminus, return the `cap` nearest end stations reachable from `start`
    without entering `forbidden` (the curve/core region). Returns [[start]] when start
    is itself a terminus or none are reachable.
    """
    if start in end_stations:
        return [[start]]
    allowed = (set(G.nodes()) - forbidden) | {start}
    H = G.subgraph(allowed)
    try:
        dist = nx.single_source_dijkstra_path_length(H, start, weight='length_m')
    except nx.NodeNotFound:
        return [[start]]
    reachable = sorted((d, n) for n, d in dist.items()
                       if n in end_stations and n != start and n not in forbidden)
    paths: List[List[str]] = []
    for _, term in reachable[:cap]:
        try:
            paths.append(nx.shortest_path(H, start, term, weight='length_m'))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    return paths or [[start]]


def _served_seq(nodes_seq, stop_index) -> List[str]:
    """Keep only served stations along a node sequence, de-duplicating consecutive repeats."""
    out: List[str] = []
    for n in nodes_seq:
        if n in stop_index and (not out or out[-1] != n):
            out.append(n)
    return out


def _composed_rail_graph(base_infra: str, req: List[str], cache: Dict):
    """Composed (base + req) rail graph + geometry segment-lookup; cached per req-set.

    Keeps rail track (Transport_Mode 'train') plus every intervention-tagged row
    (the curve segment carries no Transport_Mode but must be in the graph). The
    segment-lookup (frozenset(from,to) → row) drives the backtrack check.
    Returns (G, seg_lookup).
    """
    key = tuple(sorted(str(r) for r in req))
    if key in cache:
        return cache[key]
    nodes, segs, _comp, _warn = core.compose_frames(base_infra, list(req))
    if 'Transport_Mode' in segs.columns:
        railmask = segs['Transport_Mode'].astype(str).str.contains(
            _RAIL_MODE_TOKEN, case=False, na=False)
    else:
        railmask = pd.Series(True, index=segs.index)
    if 'int_type' in segs.columns:
        railmask = railmask | segs['int_type'].notna()
    rail = segs[railmask].reset_index(drop=True)
    G = ic.build_networkx_graph(nodes, rail)
    seg_lookup = cc._segment_lookup(rail)
    cache[key] = (G, seg_lookup)
    return G, seg_lookup


# ─────────────────────────────────────────────────────────────────────────────
# Constituent services
# ─────────────────────────────────────────────────────────────────────────────

def _line_sequences(lines, segs) -> List[Dict]:
    """dir-0 rail line stop sequences: [{route_id, variant_rank, names, total_dep, line_type, route_id}]."""
    out: List[Dict] = []
    for layer, ldf in lines.items():
        for _, line in ldf[ldf['direction_id'].astype(str) == '0'].iterrows():
            if str(line.get('mode_class', 'rail')) != 'rail':
                continue
            rid, vr = str(line['route_id']), int(line['variant_rank'])
            _, _, lseg = si._find_target(lines, segs, rid, '0', vr)
            if lseg is None or lseg.empty:
                continue
            seq = si._reconstruct_sequence(lseg)
            if len(seq) < 2:
                continue
            out.append({'route_id': rid, 'variant_rank': vr,
                        'names': [s['name'] for s in seq],
                        'total_dep': int(line.get('total_dep', 0)),
                        'line_type': int(line.get('line_type', 109))})
    return out


def _service_through(station: str, line_seqs: List[Dict]) -> Optional[Dict]:
    """The longest dir-0 rail line whose stop sequence contains `station` (representative leg)."""
    best = None
    for s in line_seqs:
        if station in s['names'] and (best is None or len(s['names']) > len(best['names'])):
            best = s
    return best


def _in_sa(name: str, stop_index, sa_polygon) -> bool:
    rec = stop_index.get(name)
    if rec is None:
        return False
    return sa_polygon.contains(Point(rec[1], rec[2]))


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.chdir(paths.MAIN)
    import infra_ints_connecting_curve as cc

    core.cli_header("infraScanRail — New Direct Connections (Phase 5B · NDC discovery)")

    core.cli_step(1, "Base infrastructure network?")
    base = core.cli_pick("Infra version:", core.cli_infra_versions(), core._resolve_base_version())

    core.cli_step(2, "Service network (defines existing direct-service pairs)?")
    svc = core.cli_pick("Service version:", core.cli_svc_versions(), core._resolve_svc_version())

    combo = f"{base}__{svc}"
    if si.list_svc_int_ids('ndc', network=combo):
        core.cli_step(3, f"Existing NDC registry found for '{combo}' — clear it first?")
        if core.cli_pick_yesno("Clear?", True):
            si.delete_records('ndc', si.list_svc_int_ids('ndc', network=combo), network=combo)
            print("  cleared ndc registry")

    sa, buf = core._load_polygon(), core._load_buffer()
    disc = cc.discover_and_register(base, svc, sa_polygon=sa, buffer_polygon=buf)
    res = discover_and_register(base, svc, disc['ndc_candidates'], sa_polygon=sa)
    print(f"\n=== {len(res['ndc_ids'])} NDC svc-int(s) registered ===")
    for r in res['records'][:20]:
        stops = r['operations'][0]['params']['stops']
        print(f"  {r['int_id']} req={r['requires_infra']} dep={r['total_dep']} "
              f"{stops[0]} … {stops[-1]} ({len(stops)} stops)")

    core.cli_step(4, "Materialise + plot the NDC deltas?")
    if core.cli_pick_yesno("Plots?", getattr(settings, 'PLOT_SVC_INTS', False)):
        si.plot_ndc_candidates(base, svc, disc['ndc_candidates'], sa_polygon=sa)
        si.materialise_and_plot(base, svc, ndc_ids=res['ndc_ids'], sa_polygon=sa)
