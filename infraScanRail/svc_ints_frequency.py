"""
svc_ints_frequency — FRQ svc-int discovery (Phase 5B, expansion part 1).
Last modified: 2026-06-11

Frequency-change interventions ('frq', id block DEV_ID_START_FRQ), two modes that
always generate together:

  (a) Corridor homogenisation — constant-frequency segment runs inside the study
      area whose whole-day dep/h is below a neighbouring run's are raised by
      extending a service that terminates at a corridor end station across the
      corridor with ALL-STOP service (an `extend` op under int_type='frq').
      Stricter than EXT: any reversal (mid-route OR at the old terminus) rejects
      the extender — frq records never carry `reversal_at_endpoint`.
  (b) Frequency doubling — one int per rail line serving ≥2 SA stations, doubling
      every variant's total_dep via `set_frequency{factor: 2}` (the applier scales
      each variant's own departures, so period-split variants double within their
      periods). Generated only while the DOUBLED whole-day frequency stays within
      FRQ_DOUBLE_MAX_FREQ_PER_H (ladder 1→2→4).

Detection runs on whole-day frequency only (total_dep over the GK window) — the
homogenisation invariant; per-period columns stay diagnostic. An (a)-candidate that
is op-identical to a registered EXT is kept (user decision 2026-06-11) but carries
`twin_of=<ext id>` so Phase 6 reuses the twin's outputs instead of recomputing.

Plan: docs/infraClaude/plans/2026-06-11-svc-int-frq-part1.md.
"""

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import fiona
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

import paths
import settings
import infrabuild_network_builder as ic
import svc_ints_extend_lines as ext
import svc_ints_orchestrator as si


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def discover_frq_candidates(
    base_infra: str,
    base_svc: str,
    sa_polygon=None,
    buffer_polygon=None,
    network: Optional[str] = None,
) -> Dict:
    """Derive FRQ candidates of both modes on the base networks (no registration).

    Args:
        base_infra: base infra version the services are projected on.
        base_svc: base service version WITHOUT the '_network' suffix.
        sa_polygon: study-area boundary (corridor scope + the ≥2-SA-stations test).
        buffer_polygon: unused (kept for 5B call-signature symmetry).
        network: registry partition for the EXT twin lookup (default: settings combo).

    Returns:
        dict(corridors, corridor_candidates, doubling_candidates) — corridors are
        the accepted mode-(a) corridors; candidate dicts feed the record builders
        and the candidate plot.
    """
    lines, segs, _stops = si._load_base_unprojected(base_svc)
    nodes, infra_segs = ic.load_version(base_infra)
    window_h = float(getattr(settings, 'GK_WINDOW_MIN', 840)) / 60.0
    route_total_dep = ext._route_total_dep(lines)
    served_by_route = ext._served_by_route(segs)
    seqs = _variant_seqs(lines, segs)

    # mode (a) — corridor homogenisation -------------------------------------
    corridors: List[Dict] = []
    corridor_candidates: List[Dict] = []
    if sa_polygon is None:
        print("  [frq] no SA polygon — corridor homogenisation skipped")
    else:
        ninfo = _node_info(nodes, sa_polygon)
        seg_freq, seg_svcs = _seg_freq_window(base_svc, base_infra, lines, window_h)
        runs = _freq_runs(seg_freq, seg_svcs)
        corridors = _corridor_candidates(runs, ninfo)
        graph, seg_lookup = ext._rail_graph(nodes, infra_segs)
        approach, _passed = ext._base_projection_info(base_svc, base_infra, nodes,
                                                      served_by_route)
        ext_twins = _ext_twin_index(network)
        corridor_candidates = _corridor_extenders(
            corridors, seqs, served_by_route, route_total_dep, window_h,
            graph, seg_lookup, approach, ninfo, ext_twins)

    # mode (b) — frequency doubling -------------------------------------------
    doubling_candidates = _doubling_candidates(
        seqs, sa_polygon, route_total_dep, window_h)

    print(f"  [frq] {len(corridors)} corridor(s) -> "
          f"{len(corridor_candidates)} extension int(s); "
          f"{len(doubling_candidates)} doubling int(s)")
    return {'corridors': corridors,
            'corridor_candidates': corridor_candidates,
            'doubling_candidates': doubling_candidates}


def discover_and_register(
    base_infra: str,
    base_svc: str,
    sa_polygon=None,
    buffer_polygon=None,
    registry_path: Optional[str] = None,
    network: Optional[str] = None,
) -> Dict:
    """Discover FRQ candidates (both modes) and append them to the frq registry.

    Deterministic id order: mode (a) first (corridors by normalised station tuple,
    extenders alphabetical), then mode (b) by route_id.
    """
    disc = discover_frq_candidates(base_infra, base_svc, sa_polygon,
                                   buffer_polygon, network=network)

    start = si.next_svc_int_id('frq', registry_path, network)
    n0 = int(start.split('_')[1])

    records: List[Dict] = []
    for c in disc['corridor_candidates']:
        iid = f"frq_{n0 + len(records)}"
        records.append(_corridor_to_record(c, iid, base_infra, base_svc))
    for c in disc['doubling_candidates']:
        iid = f"frq_{n0 + len(records)}"
        records.append(_doubling_to_record(c, iid, base_infra, base_svc))

    si.append_records('frq', records, registry_path=registry_path, network=network)
    frq_ids = [r['int_id'] for r in records]
    n_twin = sum(1 for r in records if r.get('twin_of'))
    print(f"  [frq] registered {len(frq_ids)} FRQ svc-int(s) "
          f"({len(disc['corridor_candidates'])} corridor extension(s), "
          f"{len(disc['doubling_candidates'])} doubling(s); {n_twin} EXT twin(s))")
    return {'frq_ids': frq_ids, 'records': records, **disc}


# ─────────────────────────────────────────────────────────────────────────────
# Whole-day segment-frequency substrate (shared with the frequency plots)
# ─────────────────────────────────────────────────────────────────────────────

def _seg_freq_window(base_svc, base_infra, lines, window_h
                     ) -> Tuple[Dict[Tuple[int, int], float],
                                Dict[Tuple[int, int], set]]:
    """Whole-day dep/h + service set per undirected BAV infra edge.

    Same aggregation as the (moved) _compute_seg_freq — direction-deduplicated
    (prefer dir 0, add dir-1-only variants) over the projected hop rows'
    path_nodes — but on freq_per_h_window = total_dep / window (joined from the
    unprojected line rows), never on the diagnostic per-period columns.
    """
    proj_path = paths.get_projected_services_path(base_svc, base_infra)
    hops = pd.concat([gpd.read_file(proj_path, layer=lay)
                      for lay in fiona.listlayers(proj_path)], ignore_index=True)
    deps: Dict[Tuple[str, str, int], float] = {}
    for ldf in lines.values():
        for _, r in ldf.iterrows():
            deps[(str(r['route_id']), str(r['direction_id']),
                  int(r['variant_rank']))] = float(r.get('total_dep', 0) or 0)

    dir0 = hops[hops['direction_id'].astype(str) == '0']
    d0_keys = set(map(tuple, dir0[['GTFS_ID', 'variant_rank']]
                      .drop_duplicates().values.tolist()))
    d1only = hops[(hops['direction_id'].astype(str) != '0') &
                  (~hops[['GTFS_ID', 'variant_rank']].apply(tuple, axis=1)
                   .isin(d0_keys))]
    src = pd.concat([dir0, d1only], ignore_index=True)

    seg_freq: Dict[Tuple[int, int], float] = defaultdict(float)
    seg_svcs: Dict[Tuple[int, int], set] = defaultdict(set)
    for _, row in src.iterrows():
        pn = str(row.get('path_nodes', '') or '')
        if not pn:
            continue
        dep = deps.get((str(row['GTFS_ID']), str(row['direction_id']),
                        int(row['variant_rank'])))
        if not dep:
            continue
        try:
            nids = [int(n) for n in pn.split(';') if n.strip()]
        except ValueError:
            continue
        for a, b in zip(nids[:-1], nids[1:]):
            e = (min(a, b), max(a, b))
            seg_freq[e] += dep / window_h
            seg_svcs[e].add(str(row['GTFS_ID']))
    return seg_freq, seg_svcs


def _freq_runs(seg_freq, seg_svcs) -> List[Dict]:
    """Maximal constant-frequency chains of the loaded edge network.

    A run extends through nodes of degree 2 whose two edges carry the same
    accumulated frequency; it breaks at junctions (degree ≠ 2) and at frequency
    changes. Deterministic: edges are seeded in sorted order.
    """
    adj: Dict[int, set] = defaultdict(set)
    for (a, b) in seg_freq:
        adj[a].add(b)
        adj[b].add(a)

    def is_break(n):
        if len(adj[n]) != 2:
            return True
        x, y = adj[n]
        return abs(seg_freq[(min(n, x), max(n, x))]
                   - seg_freq[(min(n, y), max(n, y))]) > 1e-6

    visited: set = set()
    runs: List[Dict] = []
    for e0 in sorted(seg_freq):
        if e0 in visited:
            continue
        chain = [e0[0], e0[1]]
        visited.add(e0)
        for pos, prev_i in ((0, 1), (-1, -2)):
            while True:
                n, prev = chain[pos], chain[prev_i]
                if is_break(n):
                    break
                nxt = next(iter(adj[n] - {prev}))
                e = (min(n, nxt), max(n, nxt))
                if e in visited:
                    break
                visited.add(e)
                chain.insert(0, nxt) if pos == 0 else chain.append(nxt)
        freq = seg_freq[(min(chain[0], chain[1]), max(chain[0], chain[1]))]
        svcs: set = set()
        for a, b in zip(chain[:-1], chain[1:]):
            svcs |= seg_svcs[(min(a, b), max(a, b))]
        runs.append({'chain': chain, 'freq': round(freq, 3), 'services': svcs})
    return runs


def _node_info(nodes, sa_polygon) -> Dict[int, Dict]:
    """BAV node Number → {name, is_station, in_sa, xy} (rail infra nodes)."""
    out: Dict[int, Dict] = {}
    for _, r in nodes.iterrows():
        try:
            num = int(float(r['Number']))
        except (TypeError, ValueError):
            continue
        geom = r.geometry
        out[num] = {'name': str(r.get('Name', '')),
                    'is_station': str(r.get('Node_Class', '')) == 'station',
                    'in_sa': bool(geom is not None and sa_polygon.contains(geom)),
                    'xy': (geom.x, geom.y) if geom is not None else None}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Mode (a) — corridor detection + extender gates
# ─────────────────────────────────────────────────────────────────────────────

def _corridor_candidates(runs, ninfo) -> List[Dict]:
    """Apply the corridor rule to the frequency runs (decision F1, 2026-06-11).

    Candidate iff: ≥2 named stations on the run; ALL its stations inside the SA
    boundary; max neighbouring-run frequency > FRQ_CORRIDOR_NEIGHBOUR_RATIO × its
    own. Runs with ≥1 SA station that fail print their reason (the connectors and
    out-of-SA runs stay silent).
    """
    ratio = float(getattr(settings, 'FRQ_CORRIDOR_NEIGHBOUR_RATIO', 1.0))
    end_runs: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(runs):
        end_runs[r['chain'][0]].append(i)
        end_runs[r['chain'][-1]].append(i)

    out: List[Dict] = []
    for i, run in enumerate(runs):
        info = [ninfo.get(n) for n in run['chain']]
        sta = [(n, d) for n, d in zip(run['chain'], info)
               if d and d['is_station']]
        names = [d['name'] for _, d in sta]
        n_in_sa = sum(1 for _, d in sta if d['in_sa'])
        if n_in_sa == 0:
            continue
        label = ' - '.join(names) if names else '(junction connector)'
        if len(sta) < 2:
            print(f"  [frq]   run {label}: <2 stations — not a corridor")
            continue
        if n_in_sa < len(sta):
            outside = [d['name'] for _, d in sta if not d['in_sa']]
            print(f"  [frq]   run {label}: station(s) outside SA "
                  f"({', '.join(outside)}) — not a corridor")
            continue
        nbrs: set = set()
        for endn in (run['chain'][0], run['chain'][-1]):
            nbrs |= {j for j in end_runs[endn] if j != i}
        nbr_freqs = [runs[j]['freq'] for j in nbrs]
        max_nbr = max(nbr_freqs) if nbr_freqs else 0.0
        if max_nbr <= ratio * run['freq'] + 1e-9:
            print(f"  [frq]   run {label} ({run['freq']:.1f} dep/h): no busier "
                  f"neighbour (max {max_nbr:.1f}) — not a corridor")
            continue
        out.append({'chain': list(run['chain']), 'stations': names,
                    'station_nodes': [n for n, _ in sta],
                    'chain_names': [d['name'] for d in info if d],
                    'freq': run['freq'], 'max_nbr_freq': max_nbr,
                    'services': set(run['services'])})
        print(f"  [frq]   CORRIDOR {label}: {run['freq']:.1f} dep/h vs max "
              f"neighbour {max_nbr:.1f} [{','.join(sorted(run['services']))}]")
    # deterministic order: orientation-normalised station tuple
    out.sort(key=lambda c: min(tuple(c['stations']),
                               tuple(reversed(c['stations']))))
    return out


def _corridor_extenders(corridors, seqs, served_by_route, route_total_dep,
                        window_h, graph, seg_lookup, approach, ninfo,
                        ext_twins) -> List[Dict]:
    """One candidate per (corridor × admissible extender) — decision F2.

    Extender = service terminating exactly at a corridor end station; excluded:
    the corridor's own services, routes already serving a corridor station to be
    added, routes below the EXT_MIN_FREQ_DEP_PER_H whole-day gate; ANY reversal
    on [base approach] + corridor chain rejects (no terminus flag-and-keep).
    """
    min_freq = float(getattr(settings, 'EXT_MIN_FREQ_DEP_PER_H', 2))
    max_detour = float(getattr(settings, 'EXT_MAX_DETOUR_FACTOR', 3.0))

    # terminus station name → [(rid, vr, end, layer, line_row)]
    termini: Dict[str, List[Tuple]] = defaultdict(list)
    for (rid, vr), (layer, line_row, seq) in seqs.items():
        if len(seq) >= 2:
            termini[seq[0]['name']].append((rid, vr, 'origin', layer, line_row))
            termini[seq[-1]['name']].append((rid, vr, 'destination', layer, line_row))

    name2node = {d['name']: n for n, d in ninfo.items() if d['is_station']}
    out: List[Dict] = []
    seen: set = set()
    for cor in corridors:
        sta_names = cor['stations']
        chain_names = cor['chain_names']
        for T, far in ((sta_names[0], sta_names[-1]),
                       (sta_names[-1], sta_names[0])):
            for rid, vr, end, layer, line_row in sorted(termini.get(T, [])):
                key = (tuple(sta_names), rid, T)
                if key in seen:
                    continue
                if rid in cor['services']:
                    print(f"  [frq]   {rid} at {T}: corridor's own service — "
                          f"not an extender")
                    continue
                if route_total_dep.get(rid, 0) / window_h < min_freq:
                    continue
                # corridor walk from T to the far end station (names incl. junctions)
                walk = _chain_between(chain_names, T, far)
                if walk is None:
                    continue
                added = [s for s in sta_names if s != T]
                if T != sta_names[0]:
                    added = list(reversed(added))
                overlap = [s for s in added
                           if s in served_by_route.get(rid, set())]
                if overlap:
                    print(f"  [frq]   {rid} at {T}: already serves "
                          f"{', '.join(overlap)} — rejected")
                    continue
                apr = approach.get((rid, vr, T, end))
                rev_chain = ([apr] + walk) if apr else walk
                revs = ext._path_reversals(rev_chain, seg_lookup)
                if revs:
                    j, theta = revs[0]
                    kind = ('terminus reversal' if apr and j == 1
                            else 'mid-route reversal')
                    print(f"  [frq]   {rid} at {T}: {kind} at {rev_chain[j]} "
                          f"({theta:.0f} deg) — rejected (frq allows none)")
                    continue
                routed_m = ext._path_length(graph, walk)
                xy_t, xy_f = (ninfo[name2node[T]]['xy'],
                              ninfo[name2node[far]]['xy'])
                beeline = math.hypot(xy_f[0] - xy_t[0], xy_f[1] - xy_t[1])
                detour = routed_m / beeline if beeline > 0 else 1.0
                if detour > max_detour:
                    print(f"  [frq]   {rid} at {T}: detour {detour:.2f} > "
                          f"{max_detour:.1f} — rejected")
                    continue
                # op stops in APPLY order: from_end='origin' prepends the list,
                # so the outermost (far) station must come first there.
                stops_op = list(reversed(added)) if end == 'origin' else added
                twin = ext_twins.get((rid, end, T, tuple(stops_op)), '')
                seen.add(key)
                out.append({
                    'route_id': rid, 'variant_rank': vr, 'from_end': end,
                    'endpoint': T, 'stops': stops_op, 'stops_walk': added,
                    'corridor_stations': list(sta_names),
                    'corridor_freq': cor['freq'],
                    'max_nbr_freq': cor['max_nbr_freq'],
                    'routed_m': round(routed_m, 1), 'detour': round(detour, 2),
                    'line_short_name': line_row.get('line_short_name'),
                    'line_type': int(line_row.get('line_type', 109)),
                    'total_dep': int(route_total_dep.get(rid, 0)),
                    'layer': layer, 'twin_of': twin,
                })
                print(f"  [frq]   EXTENDER {rid} at {T}: all-stop "
                      f"+[{', '.join(added)}] raises {cor['freq']:.1f} -> "
                      f"{cor['freq'] + route_total_dep.get(rid, 0) / window_h:.1f}"
                      f" dep/h" + (f" (twin of {twin})" if twin else ""))
        if not any(c['corridor_stations'] == list(sta_names) for c in out):
            print(f"  [frq]   corridor {' - '.join(sta_names)}: no admissible "
                  f"extender")
    out.sort(key=lambda c: (min(tuple(c['corridor_stations']),
                                tuple(reversed(c['corridor_stations']))),
                            c['route_id']))
    return out


def _chain_between(chain_names: List[str], a: str, b: str) -> Optional[List[str]]:
    """Sub-chain of node names from a to b (inclusive), oriented a→b."""
    try:
        ia, ib = chain_names.index(a), chain_names.index(b)
    except ValueError:
        return None
    return (chain_names[ia:ib + 1] if ia <= ib
            else list(reversed(chain_names[ib:ia + 1])))


def _ext_twin_index(network: Optional[str]) -> Dict[Tuple, str]:
    """(route_id, from_end, endpoint, stops) → ext int_id of registered EXTs."""
    out: Dict[Tuple, str] = {}
    for rec in si.read_records('ext', network=network):
        op = next((o for o in (rec.get('operations') or [])
                   if o.get('op') == 'extend'), None)
        if not op:
            continue
        p = op.get('params', {})
        out[(str(rec.get('route_id')), str(p.get('from_end', 'destination')),
             str(p.get('endpoint')), tuple(p.get('stops') or []))] = \
            str(rec['int_id'])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Mode (b) — frequency doubling
# ─────────────────────────────────────────────────────────────────────────────

def _doubling_candidates(seqs, sa_polygon, route_total_dep,
                         window_h) -> List[Dict]:
    """One doubling per rail route with ≥2 SA stations (decision F3).

    Eligibility: any dir-0 variant serves ≥2 stations inside the SA (the EXT
    criterion); route whole-day freq ≥ EXT_MIN_FREQ_DEP_PER_H; the DOUBLED
    whole-day freq ≤ FRQ_DOUBLE_MAX_FREQ_PER_H (1→2→4 ladder; a base-4 line is
    not pushed to 8).
    """
    min_freq = float(getattr(settings, 'EXT_MIN_FREQ_DEP_PER_H', 2))
    max_freq = float(getattr(settings, 'FRQ_DOUBLE_MAX_FREQ_PER_H', 4.0))

    by_route: Dict[str, List[Tuple]] = defaultdict(list)
    for (rid, vr), (layer, line_row, seq) in seqs.items():
        by_route[rid].append((vr, layer, line_row, seq))

    out: List[Dict] = []
    for rid in sorted(by_route):
        variants = sorted(by_route[rid])
        total = route_total_dep.get(rid, 0)
        if total <= 0 or total / window_h < min_freq:
            continue
        sa_best = 0
        sa_names: set = set()
        if sa_polygon is not None:
            for _vr, _lay, _row, seq in variants:
                in_sa = [s['name'] for s in seq
                         if sa_polygon.contains(Point(s['E'], s['N']))]
                sa_best = max(sa_best, len(in_sa))
                sa_names |= set(in_sa)
            if sa_best < 2:
                continue
        doubled_freq = 2.0 * total / window_h
        if doubled_freq > max_freq + 1e-9:
            print(f"  [frq]   {rid}: doubled freq {doubled_freq:.1f} dep/h > "
                  f"cap {max_freq:.1f} — not doubled")
            continue
        vr0, layer0, row0, _ = variants[0]
        longest = max(variants, key=lambda v: len(v[3]))[3]
        order = [s['name'] for s in longest]
        all_names = {s['name'] for _, _, _, seq in variants for s in seq}
        affected = order + sorted(all_names - set(order))
        out.append({
            'route_id': rid, 'variant_rank': vr0, 'layer': layer0,
            'line_short_name': row0.get('line_short_name'),
            'line_type': int(row0.get('line_type', 109)),
            'total_dep': int(total), 'doubled_dep': int(total) * 2,
            'freq_window': round(total / window_h, 2),
            'n_sa': sa_best, 'sa_stations': sorted(sa_names),
            'affected_stations': affected,
            'variants': [{'variant_rank': vr,
                          'service_period': row.get('service_period'),
                          'total_dep': int(row.get('total_dep', 0) or 0),
                          'origin': row.get('origin'),
                          'destination': row.get('destination')}
                         for vr, _l, row, _s in variants],
        })
        print(f"  [frq]   DOUBLE {rid}: {total} -> {2 * total} dep "
              f"({total / window_h:.1f} -> {doubled_freq:.1f} dep/h), "
              f"{sa_best} SA station(s), {len(variants)} variant(s)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Record construction
# ─────────────────────────────────────────────────────────────────────────────

def _corridor_to_record(c: Dict, int_id: str, base_infra: str,
                        base_svc: str) -> Dict:
    """Mode-(a) record: an `extend` op (all corridor stations as stops).

    Like EXT, `endpoint` keys the op so apply_svc_int extends every variant of
    the route terminating there; unlike EXT the stop list spans the corridor
    (all-stop) and `reversal_at_endpoint` is never set (frq rejects reversals).
    """
    return {
        'int_id': int_id, 'int_type': 'frq', 'base_authored': base_infra,
        'svc_version': base_svc, 'route_id': c['route_id'], 'direction_id': '0',
        'variant_rank': c['variant_rank'],
        'operations': [{'op': 'extend',
                        'params': {'from_end': c['from_end'],
                                   'endpoint': c['endpoint'],
                                   'stops': list(c['stops'])}}],
        'total_dep': c['total_dep'], 'line_type': c['line_type'],
        'mode_class': 'rail',
        'line_short_name': si.svc_int_line_name(
            {'int_id': int_id, 'int_type': 'frq'},
            base_short=c.get('line_short_name')),
        'requires_infra': [],
        'affected_stations': list(c['corridor_stations']),
        'affected_services': [c['route_id']],
        'twin_of': c.get('twin_of', ''),
    }


def _doubling_to_record(c: Dict, int_id: str, base_infra: str,
                        base_svc: str) -> Dict:
    """Mode-(b) record: one `set_frequency{factor}` op for the whole route.

    The applier multiplies EACH variant's own total_dep by the factor (O: the
    offpeak_only and peak_only variants double within their periods); the
    record's total_dep states the doubled route total.
    """
    return {
        'int_id': int_id, 'int_type': 'frq', 'base_authored': base_infra,
        'svc_version': base_svc, 'route_id': c['route_id'], 'direction_id': '0',
        'variant_rank': c['variant_rank'],
        'operations': [{'op': 'set_frequency', 'params': {'factor': 2}}],
        'total_dep': c['doubled_dep'], 'line_type': c['line_type'],
        'mode_class': 'rail',
        'line_short_name': si.svc_int_line_name(
            {'int_id': int_id, 'int_type': 'frq'},
            base_short=c.get('line_short_name')),
        'requires_infra': [],
        'affected_stations': list(c['affected_stations']),
        'affected_services': [c['route_id']],
        'twin_of': '',
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _variant_seqs(lines, segs) -> Dict[Tuple[str, int], Tuple]:
    """(route_id, variant_rank) → (layer, line_row, stop sequence) — dir-0 rail."""
    out: Dict[Tuple[str, int], Tuple] = {}
    for layer, ldf in lines.items():
        dir0 = ldf[ldf['direction_id'].astype(str) == '0']
        for _, line in dir0.iterrows():
            if str(line.get('mode_class', 'rail')) != 'rail':
                continue
            rid, vr = str(line['route_id']), int(line['variant_rank'])
            _, _, lseg = si._find_target(lines, segs, rid, '0', vr)
            if lseg is None or lseg.empty:
                continue
            seq = si._reconstruct_sequence(lseg)
            if len(seq) >= 2:
                out[(rid, vr)] = (layer, line, seq)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Frequency plots (moved from services_service_projection 2026-06-11 —
# functional port, current look; the C2 cleanup backlog restyles them)
# ─────────────────────────────────────────────────────────────────────────────

# Rail frequency bins: (lo_services, hi_services, hex_colour, linewidth, legend_label)
_FREQ_BINS: List[Tuple[int, int, str, float, str]] = [
    (1, 2, "#91bdd9", 1.0, "1–2"),
    (3, 4, "#4a8bbf", 2.0, "3–4"),
    (5, 8, "#1e5fa3", 3.5, "5–8"),
    (9, 9999, "#0c2d6b", 5.5, "9+"),
]

# Diff-plot bins: (lo_delta, hi_delta, hex_colour, linewidth, legend_label)
# Negative = loss in peak vs off-peak (red); positive = gain (green).
# Delta == 0 is drawn as unchanged (thin grey) and not listed here.
_DIFF_BINS: List[Tuple[int, int, str, float, str]] = [
    (-9999, -5, "#7b241c", 5.5, "≤ −5"),
    (   -4, -3, "#c0392b", 3.5, "−3 to −4"),
    (   -2, -1, "#e74c3c", 2.0, "−1 to −2"),
    (    1,  2, "#1a9850", 2.0, "+1 to +2"),
    (    3,  4, "#006837", 3.5, "+3 to +4"),
    (    5, 9999, "#003d1c", 5.5, "≥ +5"),
]

# Corridor-candidate palette (one colour per corridor, cycled).
_CORRIDOR_COLORS = ['#7b2d8e', '#c2185b', '#00838f', '#e65100', '#33691e']


def compute_seg_freq(
    rail_enriched: gpd.GeoDataFrame,
    freq_type: str,
) -> Dict[Tuple[int, int], float]:
    """Accumulate frequency (dep/hr) per undirected BAV segment edge.

    Direction-deduplicates rail_enriched: prefers direction_id == "0"; includes
    direction "1" only for variants absent from direction 0, so bidirectional
    services are never double-counted. freq_type 'peak' uses
    mean(freq_am_peak_dep_hr, freq_pm_peak_dep_hr), 'offpeak' uses
    freq_offpeak_dep_hr — diagnostic per-period views for the plots; detection
    uses the whole-day _seg_freq_window instead.

    Returns {(min_id, max_id): dep_hr}. Only edges with freq > 0 are included.
    """
    pn_col = "path_nodes" if "path_nodes" in rail_enriched.columns else None
    if freq_type == "peak":
        am_col   = "freq_am_peak_dep_hr"
        pm_col   = "freq_pm_peak_dep_hr"
        has_freq = am_col in rail_enriched.columns or pm_col in rail_enriched.columns
    else:
        op_col   = "freq_offpeak_dep_hr"
        has_freq = op_col in rail_enriched.columns

    seg_freq: Dict[Tuple[int, int], float] = {}
    if not pn_col or not has_freq:
        return seg_freq

    freq_source = rail_enriched
    if "direction_id" in rail_enriched.columns:
        key_cols = [c for c in ["Service", "variant_rank"] if c in rail_enriched.columns]
        if key_cols:
            dir0      = rail_enriched[rail_enriched["direction_id"].astype(str) == "0"]
            dir0_keys = set(map(tuple, dir0[key_cols].drop_duplicates().values.tolist()))
            dir1_only = rail_enriched[
                (rail_enriched["direction_id"].astype(str) != "0") &
                (~rail_enriched[key_cols].apply(tuple, axis=1).isin(dir0_keys))
            ]
            freq_source = pd.concat([dir0, dir1_only], ignore_index=True)

    for _, row in freq_source.iterrows():
        pn = str(row.get(pn_col, "") or "")
        if not pn:
            continue
        if freq_type == "peak":
            fam  = row.get(am_col)
            fpm  = row.get(pm_col)
            vals = [v for v in [fam, fpm] if v is not None and pd.notna(v) and float(v) > 0]
            fval = float(sum(vals) / len(vals)) if vals else 0.0
        else:
            fv   = row.get(op_col)
            fval = float(fv) if (fv is not None and pd.notna(fv) and float(fv) > 0) else 0.0
        if fval <= 0:
            continue
        try:
            nids = [int(n) for n in pn.split(";") if n.strip()]
        except ValueError:
            continue
        for i in range(len(nids) - 1):
            ekey = (min(nids[i], nids[i + 1]), max(nids[i], nids[i + 1]))
            seg_freq[ekey] = seg_freq.get(ekey, 0.0) + fval

    return seg_freq


def _seg_geom_lookup(infra_dir, raw_infra_dir):
    """((min_id, max_id) → segment geometry, name_to_id) for an infra version.

    Raw nodes extend the lookup so nodes absent from the working version
    (e.g. operational yards healed during projection) still resolve.
    """
    import services_service_projection as ssp
    from pathlib import Path

    nodes_gdf = gpd.read_file(Path(infra_dir) / "nodes.gpkg")
    segs_gdf  = gpd.read_file(Path(infra_dir) / "segments.gpkg")
    name_to_id = ssp._build_name_to_id(nodes_gdf)
    raw_nodes_path = Path(raw_infra_dir) / "nodes.gpkg"
    if raw_nodes_path.exists():
        raw_nodes = gpd.read_file(raw_nodes_path)
        for _, rn in raw_nodes.iterrows():
            rname = rn.get("Name", "")
            if rname and rname not in name_to_id and pd.notna(rn.get("Number")):
                name_to_id[rname] = int(rn["Number"])
    lookup: Dict[Tuple[int, int], object] = {}
    for _, seg in segs_gdf.iterrows():
        fn = name_to_id.get(seg.get("From_Name"))
        tn = name_to_id.get(seg.get("To_Name"))
        if fn is not None and tn is not None and seg.geometry is not None:
            key = (min(fn, tn), max(fn, tn))
            if key not in lookup:
                lookup[key] = seg.geometry
    return lookup, segs_gdf


def plot_frequency_map(
    infra_dir,
    raw_infra_dir,
    svc_version: str,
    infra_version: str,
    rail_enriched: gpd.GeoDataFrame,
    boundary_gpkg,
    boundary_name: str,
    freq_type: str = "offpeak",
) -> None:
    """Rail service frequency map — segment width scaled by departures/hr.

    freq_type: 'offpeak' uses freq_offpeak_dep_hr;
               'peak'    uses mean(freq_am_peak_dep_hr, freq_pm_peak_dep_hr).
    Segment frequencies are summed across all services routing through each edge.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from pathlib import Path
    from shapely.geometry import box as _sbox
    import services_service_projection as ssp

    boundary_gpkg = Path(boundary_gpkg)
    if not boundary_gpkg.exists():
        print(f"  Skipping frequency map ({boundary_name}, {freq_type}) — boundary not found.")
        return

    main_path    = Path(paths.MAIN)
    boundary_gdf = gpd.read_file(boundary_gpkg)
    is_sa        = boundary_name == "study_area"
    extent       = ssp._extent_from_gdf(boundary_gdf, margin_m=2000)

    seg_geom_lookup, segs_gdf = _seg_geom_lookup(infra_dir, raw_infra_dir)
    seg_freq = compute_seg_freq(rail_enriched, freq_type)

    def _bin(val: float) -> Tuple[str, float, str]:
        iv = int(val)
        for lo, hi, bc, blw, blbl in _FREQ_BINS:
            if lo <= iv <= hi:
                return bc, blw, blbl
        return _FREQ_BINS[-1][2], _FREQ_BINS[-1][3], _FREQ_BINS[-1][4]

    bin_geoms: Dict[str, List] = defaultdict(list)
    bin_props: Dict[str, Tuple[str, float]] = {}
    for ekey, fval in seg_freq.items():
        geom = seg_geom_lookup.get(ekey)
        if geom is None or geom.is_empty:
            continue
        bc, blw, blbl = _bin(fval)
        bin_geoms[blbl].append(geom)
        bin_props[blbl] = (bc, blw)

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_aspect("equal")
    ax.set_xlabel("E [m]", fontsize=10)
    ax.set_ylabel("N [m]", fontsize=10)
    ax.grid(True, alpha=0.3)
    boundary_gdf.plot(ax=ax, facecolor="none", edgecolor="black",
                      linewidth=1.5, linestyle="--", alpha=0.6)

    lakes_path = main_path / paths.LAKES_SHP
    if lakes_path.exists():
        try:
            lakes = gpd.read_file(lakes_path)
            if is_sa and extent is not None:
                clip_box = gpd.GeoDataFrame(
                    geometry=[_sbox(extent[0], extent[2], extent[1], extent[3])],
                    crs=si.SWISS_CRS)
                lakes_clip = gpd.clip(lakes, clip_box)
            else:
                lakes_clip = gpd.clip(lakes, boundary_gdf)
            if not lakes_clip.empty:
                lakes_clip.plot(ax=ax, color="#c8e8f5", linewidth=0.3, edgecolor="#99c4d8")
        except Exception:
            pass

    try:
        if is_sa and extent is not None:
            bg_box = gpd.GeoDataFrame(
                geometry=[_sbox(extent[0], extent[2], extent[1], extent[3])],
                crs=segs_gdf.crs if segs_gdf.crs else si.SWISS_CRS)
            bg = gpd.clip(segs_gdf, bg_box)
        else:
            bg = gpd.clip(segs_gdf, boundary_gdf)
        if not bg.empty:
            bg.plot(ax=ax, color="#d4d4d4", linewidth=0.4, alpha=0.5, zorder=1)
    except Exception:
        segs_gdf.plot(ax=ax, color="#d4d4d4", linewidth=0.4, alpha=0.5, zorder=1)

    for blbl, geoms in bin_geoms.items():
        bc, blw = bin_props[blbl]
        gpd.GeoDataFrame({"geometry": geoms}, crs=si.SWISS_CRS).plot(
            ax=ax, color=bc, linewidth=blw, zorder=3,
        )

    period_label = "Off-peak" if freq_type == "offpeak" else "Peak"
    legend_handles = [
        Line2D([0], [0], color="#d4d4d4", linewidth=1.5, label="Infrastructure (no service)"),
    ]
    for _, _, bc, blw, blbl in _FREQ_BINS:
        legend_handles.append(
            Line2D([0], [0], color=bc, linewidth=blw * 0.7,
                   label=f"{blbl} dep / hr")
        )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              title=f"Rail frequency\n{period_label}", title_fontsize=8)

    ax.set_title(
        f"Rail Service Frequency ({period_label}) — "
        f"{svc_version} on {infra_version}"
        f"\nBoundary: {boundary_name}",
        fontsize=14, fontweight="bold",
    )
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    ssp._add_north_arrow(ax, location="upper left", scale=0.5)
    ssp._add_scale_bar(ax, location=(0.755, 0.012))
    plt.tight_layout()

    out_dir = (main_path / paths.NETWORK_PLOTS_DIR / "Rail_Lines"
               / svc_version / infra_version)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname    = (f"frequency_{freq_type}_{svc_version}_"
                f"{infra_version}_{boundary_name}.pdf")
    out_path = out_dir / fname
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Frequency plot saved → {out_path}")


def plot_frequency_diff(
    infra_dir,
    raw_infra_dir,
    svc_version: str,
    infra_version: str,
    rail_enriched: gpd.GeoDataFrame,
    boundary_gpkg,
    boundary_name: str,
) -> None:
    """Peak vs off-peak frequency difference map.

    Computes (peak − off-peak) dep/hr per segment. Base is off-peak.
    Green segments gained frequency in peak; red segments lost frequency.
    Unchanged segments (delta = 0) are shown as thin grey for context.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from pathlib import Path
    from shapely.geometry import box as _sbox
    import services_service_projection as ssp

    boundary_gpkg = Path(boundary_gpkg)
    if not boundary_gpkg.exists():
        print(f"  Skipping frequency diff ({boundary_name}) — boundary not found.")
        return

    main_path    = Path(paths.MAIN)
    boundary_gdf = gpd.read_file(boundary_gpkg)
    is_sa        = boundary_name == "study_area"
    extent       = ssp._extent_from_gdf(boundary_gdf, margin_m=2000)

    seg_geom_lookup, _segs_gdf = _seg_geom_lookup(infra_dir, raw_infra_dir)
    seg_offpeak = compute_seg_freq(rail_enriched, "offpeak")
    seg_peak    = compute_seg_freq(rail_enriched, "peak")

    all_keys = set(seg_offpeak) | set(seg_peak)
    seg_delta: Dict[Tuple[int, int], float] = {
        k: seg_peak.get(k, 0.0) - seg_offpeak.get(k, 0.0)
        for k in all_keys
    }

    def _diff_bin(val: float):
        iv = int(val)
        for lo, hi, col, lw, lbl in _DIFF_BINS:
            if lo <= iv <= hi:
                return col, lw, lbl
        return None, None, None

    unchanged_geoms = []
    bin_geoms: Dict[str, list] = defaultdict(list)
    bin_props: Dict[str, Tuple[str, float]] = {}
    for ekey, delta in seg_delta.items():
        geom = seg_geom_lookup.get(ekey)
        if geom is None or geom.is_empty:
            continue
        if abs(delta) < 0.5:
            unchanged_geoms.append(geom)
        else:
            col, lw, lbl = _diff_bin(delta)
            if lbl is not None:
                bin_geoms[lbl].append(geom)
                bin_props[lbl] = (col, lw)

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_aspect("equal")
    ax.set_xlabel("E [m]", fontsize=10)
    ax.set_ylabel("N [m]", fontsize=10)
    ax.grid(True, alpha=0.3)
    boundary_gdf.plot(ax=ax, facecolor="none", edgecolor="black",
                      linewidth=1.5, linestyle="--", alpha=0.6)

    lakes_path = main_path / paths.LAKES_SHP
    if lakes_path.exists():
        try:
            lakes = gpd.read_file(lakes_path)
            if is_sa and extent is not None:
                clip_box = gpd.GeoDataFrame(
                    geometry=[_sbox(extent[0], extent[2], extent[1], extent[3])],
                    crs=si.SWISS_CRS)
                lakes_clipped = gpd.clip(lakes, clip_box)
            else:
                lakes_clipped = gpd.clip(lakes, boundary_gdf)
            if not lakes_clipped.empty:
                lakes_clipped.plot(ax=ax, color="#c8e8f5", linewidth=0.3, edgecolor="#99c4d8")
        except Exception:
            pass

    if unchanged_geoms:
        gpd.GeoDataFrame({"geometry": unchanged_geoms}, crs=si.SWISS_CRS).plot(
            ax=ax, color="#cccccc", linewidth=0.6, alpha=0.7, zorder=2)

    legend_handles = []
    loss_labels = [lbl for lo, _, _, _, lbl in _DIFF_BINS if lo < 0]
    gain_labels = [lbl for lo, _, _, _, lbl in _DIFF_BINS if lo > 0]
    for lbl in loss_labels + gain_labels:
        if lbl not in bin_geoms:
            continue
        col, lw = bin_props[lbl]
        gpd.GeoDataFrame({"geometry": bin_geoms[lbl]}, crs=si.SWISS_CRS).plot(
            ax=ax, color=col, linewidth=lw, zorder=3)
        legend_handles.append(
            Line2D([0], [0], color=col, linewidth=lw, label=f"{lbl} dep/hr"))

    legend_handles = (
        [Line2D([0], [0], color="#cccccc", linewidth=0.6, label="0 (unchanged)")]
        + legend_handles
    )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              title="Δ dep/hr (peak − off-peak)", title_fontsize=8)

    ax.set_title(
        f"Rail Frequency Change: Peak vs Off-Peak — "
        f"{svc_version} on {infra_version}"
        f"\nBoundary: {boundary_name}",
        fontsize=14, fontweight="bold",
    )
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    ssp._add_north_arrow(ax, location="upper left", scale=0.5)
    ssp._add_scale_bar(ax, location=(0.755, 0.012))
    plt.tight_layout()

    out_dir = (main_path / paths.NETWORK_PLOTS_DIR / "Rail_Lines"
               / svc_version / infra_version)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname    = (f"frequency_diff_{svc_version}_"
                f"{infra_version}_{boundary_name}.pdf")
    out_path = out_dir / fname
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Frequency diff plot saved → {out_path}")


def plot_frq_candidates(base_infra: str, base_svc: str, corridors: List[Dict],
                        candidates: List[Dict], sa_polygon=None) -> Optional[str]:
    """Candidate overview: low-frequency corridors + their admissible extenders.

    Each corridor is drawn in its own colour over the grey infra backdrop,
    labelled with its whole-day dep/h vs the max neighbouring run; every
    extender candidate is annotated at its terminus with the service short
    name, the added stops and the resulting corridor frequency (twins marked).
    """
    if not corridors:
        print("  [plot] no FRQ corridors — candidate plot skipped")
        return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from pathlib import Path
    from shapely.geometry import LineString
    import ints_core as core
    import infrabuild_network_builder as ic

    window_h = float(getattr(settings, 'GK_WINDOW_MIN', 840)) / 60.0
    combo = f"{base_infra}__{base_svc}"
    ctx = si._svc_plot_context(base_infra, base_svc, sa_polygon)

    try:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_aspect('equal')
        ax.set_axis_off()
        if ctx['lakes'] is not None and not ctx['lakes'].empty:
            ctx['lakes'].plot(ax=ax, facecolor='#c8e8f5', edgecolor='#99c4d8',
                              linewidth=0.4, zorder=0)
        if ctx['backdrop'] is not None and not ctx['backdrop'].empty:
            ctx['backdrop'].plot(ax=ax, color='#d4d4d4', linewidth=1.0, zorder=1)
        if ctx['sa_gdf'] is not None:
            ctx['sa_gdf'].plot(ax=ax, facecolor='none', edgecolor='black',
                               linewidth=1.2, linestyle='--', alpha=0.7, zorder=2)

        handles: List = []
        node_xy = ctx['node_xy']     # name → (code, x, y)
        for k, cor in enumerate(corridors):
            color = _CORRIDOR_COLORS[k % len(_CORRIDOR_COLORS)]
            pts = [(node_xy[n][1], node_xy[n][2])
                   for n in cor.get('chain_names', []) if n in node_xy]
            if len(pts) >= 2:
                gpd.GeoSeries([LineString(pts)], crs=si.SWISS_CRS).plot(
                    ax=ax, color=color, linewidth=3.2, zorder=3)
            label = ' – '.join(cor['stations'])
            handles.append(Line2D([0], [0], color=color, lw=3,
                                  label=f"{label} ({cor['freq']:.0f} vs "
                                        f"{cor['max_nbr_freq']:.0f} dep/h)"))
            # station markers
            for n in cor['stations']:
                if n in node_xy:
                    ax.plot(node_xy[n][1], node_xy[n][2], 'o', mfc='white',
                            mec=color, ms=5, zorder=4)
            # extender annotations at their terminus
            cands = [c for c in candidates
                     if c['corridor_stations'] == list(cor['stations'])]
            for j, c in enumerate(cands):
                if c['endpoint'] not in node_xy:
                    continue
                x, y = node_xy[c['endpoint']][1], node_xy[c['endpoint']][2]
                new_f = cor['freq'] + c['total_dep'] / window_h
                txt = (f"{c['line_short_name']}: +{', '.join(c['stops_walk'])} "
                       f"(→ {new_f:.0f} dep/h)"
                       + (" [= EXT twin]" if c.get('twin_of') else ""))
                ax.annotate(txt, xy=(x, y), xytext=(8, 8 + 11 * j),
                            textcoords='offset points', fontsize=7,
                            color=color, fontweight='bold', zorder=6,
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                      edgecolor=color, alpha=0.85))
                ax.plot(x, y, 'o', mfc=color, mec='black', ms=6, zorder=5)

        if ctx['extent'] is not None:
            ax.set_xlim(ctx['extent'][0], ctx['extent'][1])
            ax.set_ylim(ctx['extent'][2], ctx['extent'][3])
        ax.legend(handles=handles, loc='upper right', fontsize=7,
                  title='Low-frequency corridors', title_fontsize=8)
        ax.set_title(f"FRQ corridor candidates — {combo}\n"
                     f"{len(corridors)} corridor(s), {len(candidates)} "
                     f"extension int(s)", fontsize=12, fontweight='bold')
        ic._add_north_arrow(ax, location='upper left', scale=0.5)
        ic._add_scale_bar(ax, location=(0.755, 0.012))
        plt.tight_layout()
        out_dir = core.plot_out_dir(combo, 'frq')
        out_path = Path(out_dir) / f"frq_candidates_{base_infra}.pdf"
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"  [plot] wrote {out_path.name}")
        return str(out_path)
    except Exception as exc:
        print(f"  [plot]   WARNING frq_candidates: {exc}")
        try:
            plt.close('all')
        except Exception:
            pass
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.chdir(paths.MAIN)
    import ints_core as core

    core.cli_header("infraScanRail — Frequency Changes (Phase 5B · FRQ discovery)")

    core.cli_step(1, "Base infrastructure network?")
    base = core.cli_pick("Infra version:", core.cli_infra_versions(),
                         core._resolve_base_version())

    core.cli_step(2, "Service network to modify?")
    svc = core.cli_pick("Service version:", core.cli_svc_versions(),
                        core._resolve_svc_version())

    combo = f"{base}__{svc}"
    core.cli_step(3, "Dry run (list only) or register into the catalogue?")
    register = not core.cli_pick_yesno("Dry run?", True)

    sa = core._load_polygon()
    if register:
        if si.list_svc_int_ids('frq', network=combo):
            core.cli_step(4, f"Existing FRQ registry found for '{combo}' — clear it first?")
            if core.cli_pick_yesno("Clear?", True):
                si.delete_records('frq', si.list_svc_int_ids('frq', network=combo),
                                  network=combo)
                print("  cleared frq registry")
        res = discover_and_register(base, svc, sa_polygon=sa, network=combo)
        print(f"\n=== {len(res['frq_ids'])} FRQ svc-int(s) registered ===")
    else:
        res = discover_frq_candidates(base, svc, sa_polygon=sa, network=combo)
        print(f"\n=== dry run: {len(res['corridor_candidates'])} corridor "
              f"extension(s) + {len(res['doubling_candidates'])} doubling(s) ===")

    core.cli_step(5, "Generate the candidate plot"
                     + (" + materialise/plot the FRQ deltas?" if register else "?"))
    if core.cli_pick_yesno("Plots?", getattr(settings, 'PLOT_SVC_INTS', False)):
        plot_frq_candidates(base, svc, res['corridors'],
                            res['corridor_candidates'], sa_polygon=sa)
        if register:
            si.materialise_and_plot(base, svc, frq_ids=res['frq_ids'], sa_polygon=sa)
