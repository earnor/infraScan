"""catchment_OD_rail_network.py
Last modified: 2026-06-21

Passenger routing for infraScanRail (Phase 4C). Takes the W3 station-pair OD
matrices from catchment_OD_preparation and assigns demand onto rail services
over a frequency-aware directed graph (entry/exit/sub nodes; boarding-wait,
in-vehicle, transfer and alighting edges).

Two assignment methods:
  shortest_path — deterministic all-or-nothing on generalised cost (one Dijkstra
                  per origin, multi-target)
  logit         — Logit route choice over a candidate set; engine per
                  settings.ROUTING_LOGIT_ENGINE: 'table' (connection-table
                  itinerary proposal, fast) or 'yen' (k-shortest, exact baseline)

Transfers are unlimited up to settings.ROUTING_MAX_TRANSFERS. The cost model
mirrors catchment_allocate (Phase 4A): boarding wait = W_WAIT·t_wait_min(h);
transfer penalty per settings.TRANSFER_COST_MODEL; weights per
settings.TRAVEL_COST_METHOD. A single assignment runs on the full_day network
(the single source of truth) and trip-bearing outputs are τ-scaled to
peak / off_peak / full_day.

Public entry points:
  passenger_routing(svc_version, use_cache, od_method, assignment_method) -> None
  route_svc_int(svc_int_id, base_svc_network, ...) -> dict   (Phase 6C: closure
      re-route on a developed network, per-pair merge into the baseline
      primitive, full Phase-4C output re-derivation per svc-int)

Outputs under data/Traffic_Flow/Assignment/<svc_network>/<method>/ (workbooks have
one sheet per service period: peak / off_peak / full_day, unless noted):
  path_assignment.xlsx    segment_loads.xlsx    station_events.xlsx
  service_loads.xlsx      service_boardings.xlsx
  station_flows.xlsx      (enter / exit / transfer / total)
  skims.xlsx              (one sheet per skim: travel time, generalised cost,
                           frequency, transfers — station x station matrices)
  matrix_sa_top20.xlsx    unresolved_pairs.xlsx
Plots under plots/Traffic_Flow/Assignment/<svc_network>/<method>/ (each also _sa = SA x SA):
  matrix_travel.png / matrix_generalised_cost.png            (total actual / weighted)
  matrix_actual_cost_components.png /                        (W/V/T per cell)
    matrix_generalised_cost_components.png
  matrix_frequency.png                                       (direct D / transfer T rows)
  Sankey/sankey_service_<corridor>.{pdf,html}
  ServiceLoads/service_load_<line>.png   (per-service station-sequence loads)
"""

import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyogrio

import cache_manifest
import paths
import settings
import cost_parameters as cp
import catchment_base
import catchment_allocate

_CODEBASE_CRS = catchment_base.CODEBASE_CRS

# Curated same-complex station equivalences. Some physically-single interchange
# complexes carry distinct DiDok numbers in the GTFS-level services data, so the
# routing graph would otherwise treat them as separate (even disconnected)
# stations. Each (a, b) pair gets bidirectional transfer edges between the two
# stations' sub-nodes with the normal transfer penalty (a real cross-platform
# change). The build-time split-station diagnostic flags further candidates.
#   (8503059, 8503003): Zürich Stadelhofen, Bahnhof (Forchbahn / S18) <->
#       Zürich Stadelhofen (S-Bahn) — connects the otherwise isolated Forchbahn.

STATION_TRANSFER_LINKS = [
    (8503059, 8503003)
]

# Warn about co-located distinct served stations within this distance that are
# not in STATION_TRANSFER_LINKS (possible unmodelled split complex).
SPLIT_STATION_WARN_M = 150.0

# SBB station-frequency methodology (first sheet of the passenger-numbers workbook),
# recorded with the station-flow totals so the column can validate our methodology:
# total = enter + exit + 2*transfer.
_SBB_STATION_FLOW_NOTE = (
    "total = enter + exit + 2*transfer. Mirrors the SBB station-frequency "
    "methodology: the figures count passengers boarding and alighting the railway "
    "services; passengers boarding/alighting other public transport and passers-by "
    "are not taken into account; passengers changing trains are counted twice "
    "(once alighting, once boarding)."
)

# ===============================================================================
# COST MODEL  (mirrors catchment_allocate / cost_parameters, in minutes)
# ===============================================================================

def _active_weights() -> dict:
    """Active GC weights. 'absolute' -> all 1.0; 'calibrated' -> cost_parameters."""
    if settings.TRAVEL_COST_METHOD == 'absolute':
        return {'ivt': 1.0, 'wait': 1.0, 'transfer': 1.0}
    return {'ivt': float(cp.W_IVT), 'wait': float(cp.W_WAIT),
            'transfer': float(cp.W_TRANSFER)}


def _wait_min(headway_min: float) -> float:
    """Unweighted boarding wait (clock minutes): h/2 below 12, else 6+0.25(h-12)."""
    return cp.t_wait_min(headway_min)


def _transfer_penalty_min(headway_next_min: float) -> float:
    """Weighted GC transfer penalty (min), keyed on the connecting line's headway.

    explicit    -> w_transfer * max(TRANSFER_WALK_MIN, t_wait(h_next))
    fixed_value -> PI_TRANSFER_MIN (calibrated) | average_train_change_time (absolute)
    """
    is_abs = (settings.TRAVEL_COST_METHOD == 'absolute')
    if settings.TRANSFER_COST_MODEL == 'explicit':
        w = 1.0 if is_abs else float(cp.W_TRANSFER)
        return w * max(cp.TRANSFER_WALK_MIN, cp.t_wait_min(headway_next_min))
    if is_abs:
        return float(cp.average_train_change_time)
    return float(cp.PI_TRANSFER_MIN)


def _transfer_time_min(headway_next_min: float) -> float:
    """Unweighted clock time of a transfer (min)."""
    if settings.TRANSFER_COST_MODEL == 'explicit':
        return max(cp.TRANSFER_WALK_MIN, cp.t_wait_min(headway_next_min))
    return float(cp.average_train_change_time)


# ===============================================================================
# GRAPH
# ===============================================================================

def _build_rail_graph(rail_segments: pd.DataFrame,
                      rail_lines: pd.DataFrame,
                      rail_stations: gpd.GeoDataFrame) -> tuple:
    """Build the frequency-aware directed routing graph.

    Nodes (attributes kind / station_key / variant_key):
        entry_{skey}, exit_{skey}, sub_{skey}_{variant_key}
    Edges (attributes gc=weighted-min, time=clock-min):
        entry->sub (board wait), sub->sub same variant (in-vehicle),
        sub->sub same station diff variant (transfer), sub->exit (alight, 0).
        Board and transfer waits use the COMBINED headway of all lines sharing the
        segment the rider takes out of the stop (common lines: 60 / Σ freq), not
        each line's individual headway.

    skey is the station id_point (int) when the stop resolves, else the stop_id
    string (out-of-catchment through stops — never an OD endpoint, but routable).

    Returns:
        (G, vfreq, direct_freq, variant_seq, report_ctx)
        G:           nx.DiGraph
        vfreq:       dict[variant_key -> freq_per_h_window]  (routing headways)
        direct_freq: dict[(skey_a, skey_c) -> summed freq of variants serving it directly]
        variant_seq: dict[variant_key -> ordered list of skey (the stop sequence)]
        report_ctx:  dict with reporting-only structures:
            'vfreq_report'        variant_key -> lowest positive period dep/h
            'direct_freq_report'  (skey_a, skey_c) -> summed vfreq_report
            'variant_period_freq' variant_key -> {'peak', 'off_peak'} dep/h
            'route_name'          route_id (str) -> line_short_name
    """
    stn = rail_stations.copy()
    stn['stop_id_str'] = stn['stop_id'].astype(str)
    stop_to_idp = dict(zip(stn['stop_id_str'], stn['id_point'].astype(int)))

    seg = rail_segments.copy()
    seg['from_stop_id'] = seg['from_stop_id'].astype(str)
    seg['to_stop_id']   = seg['to_stop_id'].astype(str)
    seg['route_id']     = seg['route_id'].astype(str)
    seg = seg.dropna(subset=['from_stop_id', 'to_stop_id', 'route_id',
                             'direction_id', 'variant_rank', 'travel_time_min'])

    ln = rail_lines.copy()
    ln['route_id'] = ln['route_id'].astype(str)
    ln = ln.dropna(subset=['route_id', 'direction_id', 'variant_rank',
                           'freq_per_h_window'])
    ln_freq = ln.set_index(
        ['route_id', 'direction_id', 'variant_rank'])['freq_per_h_window'].to_dict()

    # Per-variant period departures/hour (precomputed columns; 0.0 if absent).
    # The reporting frequency (skim/heatmap) uses the lowest positive period
    # frequency; the per-period service frequency (load plots) uses
    # peak=max(am,pm) — the capacity_calculator convention — and off_peak=offpeak.
    def _period_lookup(col):
        if col not in ln.columns:
            return {}
        s = pd.to_numeric(ln[col], errors='coerce').fillna(0.0)
        return dict(zip(zip(ln['route_id'], ln['direction_id'], ln['variant_rank']), s))

    ln_am = _period_lookup('freq_am_peak_dep_hr')
    ln_pm = _period_lookup('freq_pm_peak_dep_hr')
    ln_op = _period_lookup('freq_offpeak_dep_hr')

    # route_id -> line_short_name (first non-null), fallback handled by _line_name().
    route_name_lookup = {}
    if 'line_short_name' in ln.columns:
        for r in ln[['route_id', 'line_short_name']].itertuples(index=False):
            nm = r.line_short_name
            if pd.notna(nm) and str(nm).strip():
                route_name_lookup.setdefault(str(r.route_id), str(nm))

    w = _active_weights()
    G = nx.DiGraph()
    vfreq = {}
    vfreq_report = {}          # variant_key -> lowest positive period dep/h (reporting)
    variant_period_freq = {}   # variant_key -> {'peak': max(am,pm), 'off_peak': offpeak}
    direct_freq = {}
    direct_freq_report = {}    # (skey_a, skey_c) -> summed vfreq_report (reporting)
    variant_seq = {}           # variant_key -> ordered list of skey (stop sequence)
    station_subs = {}          # skey -> list of (sub_node, pooled_headway_min)
    seg_window_freq = {}       # (skey_a, skey_b) -> Σ vfreq over variants on the
                               # consecutive segment (combined common-lines freq)
    n_variants_used = n_variants_skip = 0

    def _skey(sid):
        return stop_to_idp.get(sid, f"x{sid}")

    def _ensure_portal(skey):
        en, ex = f"entry_{skey}", f"exit_{skey}"
        if en not in G:
            G.add_node(en, kind='entry', station_key=skey)
            G.add_node(ex, kind='exit',  station_key=skey)
        return en, ex

    # Pass 1: per-variant sequence, frequencies and in-vehicle times; accumulate the
    # combined per-segment frequency used for common-lines boarding-wait pooling.
    variants = []   # (variant_key, sequence_raw, seq_skeys, tt_map, freq)
    for (rid, did, vrnk), var_segs in seg.groupby(
            ['route_id', 'direction_id', 'variant_rank'], sort=False):
        freq = ln_freq.get((rid, did, vrnk))
        if freq is None or freq <= 0:
            n_variants_skip += 1
            continue
        sequence = catchment_allocate._reconstruct_stop_sequence(var_segs)
        if len(sequence) < 2:
            n_variants_skip += 1
            continue

        # In-vehicle time per hop = TT + IVWT (dwell at intermediate timetable
        # holds is elapsed in-vehicle time; also carries the EXT terminus-reversal
        # penalty). Decided 2026-06-11.
        tt_map = {}
        for _, row in var_segs.iterrows():
            ivwt = float(row['ivwt_min']) if 'ivwt_min' in var_segs.columns else 0.0
            tt_map[(str(row['from_stop_id']), str(row['to_stop_id']))] = \
                float(row['travel_time_min']) + ivwt

        variant_key = f"{rid}_{did}_{vrnk}"
        vfreq[variant_key] = float(freq)
        am = float(ln_am.get((rid, did, vrnk), 0.0))
        pm = float(ln_pm.get((rid, did, vrnk), 0.0))
        op = float(ln_op.get((rid, did, vrnk), 0.0))
        positive = [x for x in (am, pm, op) if x > 0]
        vfreq_report[variant_key] = min(positive) if positive else float(freq)
        variant_period_freq[variant_key] = {'peak': max(am, pm), 'off_peak': op}
        n_variants_used += 1

        seq_skeys = [_skey(sid) for sid in sequence]
        variant_seq[variant_key] = seq_skeys
        variants.append((variant_key, sequence, seq_skeys, tt_map, float(freq)))

        # combined per-segment frequency (common-lines pooling) + direct frequency
        for i in range(len(seq_skeys) - 1):
            key = (seq_skeys[i], seq_skeys[i + 1])
            seg_window_freq[key] = seg_window_freq.get(key, 0.0) + float(freq)
        freq_rep = vfreq_report[variant_key]
        for i in range(len(seq_skeys)):
            a = seq_skeys[i]
            for j in range(i + 1, len(seq_skeys)):
                c = seq_skeys[j]
                direct_freq[(a, c)] = direct_freq.get((a, c), 0.0) + float(freq)
                direct_freq_report[(a, c)] = \
                    direct_freq_report.get((a, c), 0.0) + freq_rep

    # Pass 2: nodes + edges. The boarding wait uses the COMBINED frequency of every
    # line sharing the segment the rider takes out of the stop (common lines /
    # Chriqui-Robillard): h_pool = 60 / Σ freq, so parallel trunk services give the
    # pooled headway, not each line's individual headway. The same pooled headway
    # keys the transfer wait onto that line. Terminus stops (no outgoing segment)
    # fall back to the line's own headway — no in-vehicle edge follows there.
    def _board_headway(seq_skeys, i, own_freq):
        if i < len(seq_skeys) - 1:
            f = seg_window_freq.get((seq_skeys[i], seq_skeys[i + 1]), 0.0)
            if f > 0:
                return 60.0 / f
        return 60.0 / own_freq

    for variant_key, sequence, seq_skeys, tt_map, freq in variants:
        sub_nodes = []
        for i, skey in enumerate(seq_skeys):
            en, ex = _ensure_portal(skey)
            sub = f"sub_{skey}_{variant_key}"
            G.add_node(sub, kind='sub', station_key=skey, variant_key=variant_key)
            h_pool = _board_headway(seq_skeys, i, freq)
            G.add_edge(en, sub, gc=w['wait'] * _wait_min(h_pool),
                       time=_wait_min(h_pool), board_freq=60.0 / h_pool)
            G.add_edge(sub, ex, gc=0.0, time=0.0)
            station_subs.setdefault(skey, []).append((sub, h_pool))
            sub_nodes.append((skey, sub))

        # in-vehicle edges between consecutive stops
        for i in range(len(sequence) - 1):
            ivt = tt_map.get((sequence[i], sequence[i + 1]), 0.0)
            su = sub_nodes[i][1]
            sv = sub_nodes[i + 1][1]
            G.add_edge(su, sv, gc=w['ivt'] * ivt, time=ivt)

    # transfer edges: between every pair of sub-nodes at a station (cost by the
    # target line's pooled boarding headway, so transfers pool like boardings)
    n_transfer_edges = 0
    for skey, subs in station_subs.items():
        for su, _h_u in subs:
            for sv, h_v in subs:
                if su == sv:
                    continue
                G.add_edge(su, sv, gc=_transfer_penalty_min(h_v),
                           time=_transfer_time_min(h_v), board_freq=60.0 / h_v)
                n_transfer_edges += 1

    # Curated same-complex links: connect sub-nodes across physically-single
    # interchanges that carry distinct numbers (see STATION_TRANSFER_LINKS).
    n_link_edges = 0
    for s1, s2 in STATION_TRANSFER_LINKS:
        subs1, subs2 = station_subs.get(s1), station_subs.get(s2)
        if not subs1 or not subs2:
            continue
        for src_subs, dst_subs in ((subs1, subs2), (subs2, subs1)):
            for su, _h in src_subs:
                for sv, h_v in dst_subs:
                    G.add_edge(su, sv, gc=_transfer_penalty_min(h_v),
                               time=_transfer_time_min(h_v), board_freq=60.0 / h_v)
                    n_link_edges += 1

    # Diagnostic: flag co-located distinct served stations not already linked.
    _linked = {frozenset((int(a), int(b))) for a, b in STATION_TRANSFER_LINKS}
    rs_idp = rail_stations.copy()
    rs_idp['idp'] = pd.to_numeric(rs_idp['id_point'], errors='coerce')
    served_geo = [(int(r['idp']), r.geometry.x, r.geometry.y)
                  for _, r in rs_idp.dropna(subset=['idp']).iterrows()
                  if int(r['idp']) in station_subs and r.geometry is not None]
    split_warn = []
    for i in range(len(served_geo)):
        a, ax, ay = served_geo[i]
        for j in range(i + 1, len(served_geo)):
            c, cx, cy = served_geo[j]
            if ((ax - cx) ** 2 + (ay - cy) ** 2) ** 0.5 <= SPLIT_STATION_WARN_M \
                    and frozenset((a, c)) not in _linked:
                split_warn.append((a, c))

    print(f"  Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges "
          f"({n_variants_used} variants, {n_variants_skip} skipped, "
          f"{n_transfer_edges:,} transfer + {n_link_edges} link edges).")
    if split_warn:
        print(f"  WARNING: {len(split_warn)} co-located served station pair(s) "
              f"< {SPLIT_STATION_WARN_M:.0f} m not in STATION_TRANSFER_LINKS "
              f"(possible split complex): {split_warn[:8]}")
    report_ctx = {'vfreq_report': vfreq_report,
                  'direct_freq_report': direct_freq_report,
                  'variant_period_freq': variant_period_freq,
                  'route_name': route_name_lookup}
    return G, vfreq, direct_freq, variant_seq, report_ctx


# ===============================================================================
# PATH DECOMPOSITION
# ===============================================================================

def _walk_path(G: nx.DiGraph, path: list) -> dict:
    """Classify a node path into segments + board/alight/transfer events.

    Returns dict with:
        gc, journey_time       floats (summed weighted / clock minutes)
        n_transfers            int
        lines_used             list[variant_key] in boarding order
        segments               list[(from_skey, to_skey, variant_key)]
        events                 list[(station_key, event, variant_key)]
        wait_t/ivt_t/xfer_t    unweighted component minutes (boarding wait /
                               in-vehicle incl. IVWT / transfer)
        wait_g/ivt_g/xfer_g    the same components with GC weights applied
        frequency              bottleneck of the per-leg pooled boarding frequency
                               (min over board + transfer edges); nan if no in-graph
                               boarding (e.g. a forced gateway-only leg)
    """
    gc = jt = 0.0
    wait_t = ivt_t = xfer_t = wait_g = ivt_g = xfer_g = 0.0
    n_transfers = 0
    lines, segments, events, board_freqs = [], [], [], []

    for u, v in zip(path[:-1], path[1:]):
        ed = G.edges[u, v]
        gc += ed['gc']
        jt += ed['time']
        du, dv = G.nodes[u], G.nodes[v]
        ku, kv = du['kind'], dv['kind']

        if ku == 'entry' and kv == 'sub':                      # board
            events.append((dv['station_key'], 'board', dv['variant_key']))
            lines.append(dv['variant_key'])
            wait_t += ed['time']; wait_g += ed['gc']
            if 'board_freq' in ed:
                board_freqs.append(ed['board_freq'])
        elif ku == 'sub' and kv == 'exit':                     # alight
            events.append((du['station_key'], 'alight', du['variant_key']))
        elif ku == 'sub' and kv == 'sub':
            if du['variant_key'] == dv['variant_key']:          # in-vehicle (same line)
                segments.append((du['station_key'], dv['station_key'],
                                 du['variant_key']))
                ivt_t += ed['time']; ivt_g += ed['gc']
            else:                                # transfer (same station or curated link)
                n_transfers += 1
                events.append((dv['station_key'], 'transfer', dv['variant_key']))
                lines.append(dv['variant_key'])
                xfer_t += ed['time']; xfer_g += ed['gc']
                if 'board_freq' in ed:
                    board_freqs.append(ed['board_freq'])

    return {'gc': gc, 'journey_time': jt, 'n_transfers': n_transfers,
            'lines_used': lines, 'segments': segments, 'events': events,
            'wait_t': wait_t, 'ivt_t': ivt_t, 'xfer_t': xfer_t,
            'wait_g': wait_g, 'ivt_g': ivt_g, 'xfer_g': xfer_g,
            'frequency': min(board_freqs) if board_freqs else float('nan')}


# ===============================================================================
# ASSIGNMENT
# ===============================================================================

def _empty_primitives() -> dict:
    return {'paths': [], 'segments': [], 'events': [], 'unresolved': []}


def _prim_empty(rows) -> bool:
    """True when a primitive table has no rows. Primitive tables are lists of
    dicts on the assignment path and DataFrames after a Phase-6 merge (parquet
    reload); plain truthiness raises on a DataFrame, so every consumer guards
    through this helper."""
    return rows is None or len(rows) == 0


def _record_path(prim: dict, A: int, C: int, path_id: int, share: float,
                 od_trips: float, info: dict) -> None:
    """Append one chosen path's rows (paths/segments/events) at pre-τ trips."""
    trips = od_trips * share
    prim['paths'].append({
        'origin_id': A, 'dest_id': C, 'path_id': path_id,
        'n_transfers': info['n_transfers'],
        'lines_used': '>'.join(info['lines_used']),
        'journey_time_min': round(info['journey_time'], 3),
        'gc_min': round(info['gc'], 3),
        'wait_t': round(info.get('wait_t', 0.0), 3),
        'ivt_t': round(info.get('ivt_t', 0.0), 3),
        'xfer_t': round(info.get('xfer_t', 0.0), 3),
        'wait_g': round(info.get('wait_g', 0.0), 3),
        'ivt_g': round(info.get('ivt_g', 0.0), 3),
        'xfer_g': round(info.get('xfer_g', 0.0), 3),
        'frequency': round(info.get('frequency', float('nan')), 3),
        'freq_direct': round(info.get('freq_direct', float('nan')), 3),
        'freq_transfer': round(info.get('freq_transfer', float('nan')), 3),
        'share': share, 'trips': trips,
    })
    # Gateways are routing-only terminals: drop the synthetic terminal hop and the
    # gateway alight so downstream segment/station-flow consumers (service loads,
    # capacity) see exactly what they did before (gateways contribute no segments).
    gw = _GATEWAY_IDS
    for (fr, to, vk) in info['segments']:
        if fr in gw or to in gw:
            continue
        prim['segments'].append({
            'origin_id': A, 'dest_id': C, 'path_id': path_id,
            'from_id': fr, 'to_id': to, 'variant_key': vk, 'trips': trips})
    for (skey, ev, vk) in info['events']:
        if skey in gw:
            continue
        prim['events'].append({
            'origin_id': A, 'dest_id': C, 'path_id': path_id,
            'station_id': skey, 'event': ev, 'variant_key': vk, 'trips': trips})


def _assign_shortest_path(G: nx.DiGraph, od_long: pd.DataFrame) -> dict:
    """All-or-nothing assignment: 100% of each pair on its least-GC path.

    One multi-target Dijkstra per origin (single_source_dijkstra) replaces the
    per-pair call: ~N-origin Dijkstras instead of one per OD pair. Results are
    identical in GC/journey-time/transfers (path identity may differ only on exact
    GC ties, which carry the same cost). Returns a `primitives` dict.
    """
    prim = _empty_primitives()
    n_ok = n_unres = 0
    for A, grp in od_long.groupby('origin_id', sort=False):
        A = int(A)
        src = f"entry_{A}"
        if src not in G:
            for r in grp.itertuples(index=False):
                prim['unresolved'].append({'origin_id': A, 'dest_id': int(r.dest_id),
                                           'trips': float(r.trips)})
                n_unres += 1
            continue
        _dist, sp_paths = nx.single_source_dijkstra(G, src, weight='gc')
        for r in grp.itertuples(index=False):
            C, trips = int(r.dest_id), float(r.trips)
            path = sp_paths.get(f"exit_{C}")
            if path is None:
                prim['unresolved'].append({'origin_id': A, 'dest_id': C, 'trips': trips})
                n_unres += 1
                continue
            _record_path(prim, A, C, 0, 1.0, trips, _walk_path(G, path))
            n_ok += 1
    print(f"  [shortest_path] routed {n_ok:,}, unresolved {n_unres:,} "
          f"(per-origin Dijkstra).")
    return prim


# --- Logit: unified scoring over engine-generated candidates -------------------

def _choose_pair(prim: dict, A: int, C: int, trips: float, infos: list,
                 vfreq: dict, theta: float, max_transfers: int, k: int,
                 window_min: float, window_pct: float) -> bool:
    """Attractive-set choice model for one OD pair. Returns True if it resolved.

    Keeps EVERY direct service (no speed/frequency drop); admits a transfer only
    when no direct exists or its clock travel time <= the fastest direct's, then
    caps the transfer alternatives to the k lowest-GC (directs are never capped).
    The boarding wait and each transfer-hub wait are pooled over the distinct
    services of the kept set (combined headway, common-lines). Demand splits by a
    frequency-weighted logit (share ~ f_first * exp(-theta*gc)) over the kept set,
    which renormalises to 1 so the capped tail's demand is conserved. The reported
    components are re-scored with the pooled waits. Each chosen path is recorded.
    """
    if not infos:
        prim['unresolved'].append({'origin_id': A, 'dest_id': C, 'trips': trips})
        return False

    # Dedupe by line sequence, keep the cheapest enumerated GC variant.
    seen, uniq = set(), []
    for i in sorted(infos, key=lambda x: x['gc']):
        key = '>'.join(i['lines_used'])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(i)

    direct = [i for i in uniq if i['n_transfers'] == 0]
    transfer = [i for i in uniq if 1 <= i['n_transfers'] <= max_transfers]

    # Admission (decision 5): all direct; transfers only if competitive on clock
    # travel time, else (no direct) bounded by the GC cost window.
    if direct:
        min_direct = min(i['journey_time'] for i in direct)
        transfer = [i for i in transfer if i['journey_time'] <= min_direct + 1e-9]
    elif transfer:
        best = min(i['gc'] for i in transfer)
        cut = best + max(window_min, best * window_pct)
        transfer = [i for i in transfer if i['gc'] <= cut + 1e-9]
    viable_full = direct + transfer
    if not viable_full:
        prim['unresolved'].append({'origin_id': A, 'dest_id': C, 'trips': trips})
        return False

    # Boarding legs: ordered (station_key, variant_key) per board + transfer event;
    # leg[0] is the origin boarding, the rest are transfer hubs.
    def _legs(info):
        return [(st, vk) for (st, ev, vk) in info['events']
                if ev in ('board', 'transfer')]

    # Pool the boarding + transfer waits over the FULL viable set (decision c): the
    # wait reflects every service a passenger could take, including those the
    # storage cap below drops. Origin pooled wait = combined headway of the distinct
    # first-leg services; per-hub pooled wait = combined headway of the distinct
    # services boarded at that hub.
    first_svcs, hub_svcs = {}, {}
    for i in viable_full:
        lg = _legs(i)
        if lg:
            first_svcs[lg[0][1]] = float(vfreq.get(lg[0][1], 0.0))
        for (st, vk) in lg[1:]:
            hub_svcs.setdefault(st, {})[vk] = float(vfreq.get(vk, 0.0))
    f_origin = sum(first_svcs.values())
    h_origin = 60.0 / f_origin if f_origin > 0 else float('inf')
    hub_h = {st: (60.0 / s if (s := sum(d.values())) > 0 else float('inf'))
             for st, d in hub_svcs.items()}

    # Cap transfer alternatives to the k lowest-GC for storage (decision A). The
    # pooled waits above already account for the dropped tail; the logit over the
    # kept set renormalises so demand is conserved. Directs are never capped.
    if len(transfer) > k:
        transfer = sorted(transfer, key=lambda i: i['gc'])[:k]
    viable = direct + transfer

    w_wait = _active_weights()['wait']
    wait_t = _wait_min(h_origin)
    wait_g = w_wait * wait_t

    scored = []
    for i in viable:
        hubs = [st for (st, _vk) in _legs(i)[1:]]
        xfer_t = sum(_transfer_time_min(hub_h[st]) for st in hubs)
        xfer_g = sum(_transfer_penalty_min(hub_h[st]) for st in hubs)
        gc = wait_g + i['ivt_g'] + xfer_g
        jt = wait_t + i['ivt_t'] + xfer_t
        lg = _legs(i)
        f_first = float(vfreq.get(lg[0][1], 0.0)) if lg else 0.0
        # Per-path connection frequency = bottleneck of THIS path's pooled legs
        # (origin pooled, then each of its own transfer hubs). A minor alternative
        # via a thin hub no longer drags the whole OD's reported frequency down.
        freq_path = min([f_origin] + [sum(hub_svcs[st].values()) for st in hubs])
        scored.append((i, gc, jt, xfer_t, xfer_g, f_first, freq_path))

    # Direct vs transfer connection frequency for the frequency plot: D = combined
    # supply of the direct services; T = the best (least-GC) transfer route's
    # bottleneck. Either is nan when the OD has no option of that kind.
    direct_first = {}
    for i in direct:
        lg = _legs(i)
        if lg:
            direct_first[lg[0][1]] = float(vfreq.get(lg[0][1], 0.0))
    freq_direct = sum(direct_first.values()) if direct_first else float('nan')
    _xfer_scored = [s for s in scored if s[0]['n_transfers'] > 0]
    freq_transfer = (min(_xfer_scored, key=lambda s: s[1])[6]
                     if _xfer_scored else float('nan'))

    gcs = np.array([s[1] for s in scored], dtype=float)
    fw = np.array([s[5] for s in scored], dtype=float)
    u = -theta * gcs
    u -= u.max()
    wts = fw * np.exp(u)
    tot = wts.sum()
    shares = (wts / tot) if tot > 0 else np.full(len(scored), 1.0 / len(scored))

    for pid, idx in enumerate(np.argsort(gcs).tolist()):
        info, gc, jt, xfer_t, xfer_g, _f, freq_path = scored[idx]
        rec = dict(info)
        rec.update(gc=gc, journey_time=jt, wait_t=wait_t, wait_g=wait_g,
                   xfer_t=xfer_t, xfer_g=xfer_g, frequency=freq_path,
                   freq_direct=freq_direct, freq_transfer=freq_transfer)
        _record_path(prim, A, C, pid, float(shares[idx]), trips, rec)
    return True


def _yen_infos(G: nx.DiGraph, src: str, tgt: str, window_min: float,
               window_pct: float, k: int, max_examine: int) -> list:
    """Baseline engine: Yen k-shortest paths, walked, cost-ordered within window."""
    out, best_gc, examined = [], None, 0
    try:
        for path in nx.shortest_simple_paths(G, src, tgt, weight='gc'):
            examined += 1
            info = _walk_path(G, path)
            if best_gc is None:
                best_gc = info['gc']
            if info['gc'] > best_gc + max(window_min, best_gc * window_pct):
                break
            out.append(info)
            if len(out) >= k or examined >= max_examine:
                break
    except nx.NetworkXNoPath:
        pass
    return out


def _build_table_context(variant_seq: dict) -> dict:
    """Precompute the connection table once: per-variant stop index, and for every
    station its downstream / upstream reachability with the serving variants.

    Returns dict with:
        seq:  variant_key -> [skey,...]
        idx:  variant_key -> {skey: first_pos}
        down: skey -> {reachable_downstream_skey: set(variant_key)}  (board here, ride to)
        up:   skey -> {upstream_skey: set(variant_key)}              (board there, reach here)
    """
    vidx = {}
    sv = {}
    for vk, seq in variant_seq.items():
        idx = {}
        for pos, sk in enumerate(seq):
            if sk not in idx:
                idx[sk] = pos
            sv.setdefault(sk, set()).add(vk)
        vidx[vk] = idx

    down, up = {}, {}
    for sk, vks in sv.items():
        dmap, umap = {}, {}
        # sorted: vks is a set of variant-key strings whose iteration order
        # varies with the process hash seed; dmap/umap insertion order must not.
        for vk in sorted(vks):
            seq, pos = variant_seq[vk], vidx[vk][sk]
            for s2 in seq[pos + 1:]:
                dmap.setdefault(s2, set()).add(vk)
            for s2 in seq[:pos]:
                umap.setdefault(s2, set()).add(vk)
        down[sk], up[sk] = dmap, umap
    return {'seq': variant_seq, 'idx': vidx, 'sv': sv, 'down': down, 'up': up}


def _realise_path(ctx: dict, A, C, legs: list):
    """Realise an itinerary [(variant_key, from_skey, to_skey), ...] as a node-path
    in G (entry -> sub-run -> [transfer] -> sub-run -> exit). None if any leg invalid."""
    nodes = [f"entry_{A}"]
    for (vk, a, b) in legs:
        seq, idx = ctx['seq'][vk], ctx['idx'][vk]
        ia, ib = idx.get(a), idx.get(b)
        if ia is None or ib is None or ia >= ib:
            return None
        nodes += [f"sub_{seq[p]}_{vk}" for p in range(ia, ib + 1)]
    nodes.append(f"exit_{C}")
    return nodes


def _table_infos(G: nx.DiGraph, A: int, C: int, sp_path: list, ctx: dict,
                 dist_f: dict, window_min: float, window_pct: float,
                 cap: int = 500) -> list:
    """Connection-table engine: propose 0-, 1- and 2-transfer itineraries from the
    precomputed reachability table, realise each as a real node-path in G and score
    it with _walk_path (one cost model). Interchanges and variants are pruned by the
    shortest-path GC oracle (dist_f) to the cost window. Always includes the SP path,
    so every pair resolves and the optimum is present; 3-transfer alternatives (≈0%
    of demand) are represented only by that SP path (documented approximation)."""
    best_gc = dist_f.get(f"exit_{C}")
    cutoff = (best_gc + max(window_min, best_gc * window_pct)
              if best_gc is not None else None)

    def _in(node):
        return cutoff is None or dist_f.get(node, float('inf')) <= cutoff

    sv, idx, down, up = ctx['sv'], ctx['idx'], ctx['down'], ctx['up']
    downA, upC = down.get(A, {}), up.get(C, {})
    proposals = []

    # All set iterations below are sorted: variant keys are strings (and station
    # keys mix int and 'x…' str), so raw set order varies with the per-process
    # hash seed — which would change proposal order, the cap'd proposal SUBSET
    # and equal-GC tie-breaks in _finalize_pair between processes.
    for vk in sorted(sv.get(A, ())):                              # 0-transfer
        ia, ic = idx[vk].get(A), idx[vk].get(C)
        if ia is not None and ic is not None and ia < ic:
            proposals.append([(vk, A, C)])

    # Boundary gateways are pure terminals (origin/dest), never transfer hubs:
    # transferring at one means riding out to the boundary and back. Exclude them
    # from interchange candidates (the synthetic gateway sub also has no transfer
    # edge, so proposing one would dead-end).
    interch1 = [T for T in sorted(set(downA) & set(upC), key=str)
                if T not in (A, C) and T not in _GATEWAY_IDS and _in(f"exit_{T}")]
    for T in interch1:                                            # 1-transfer
        for v1 in sorted(downA[T]):
            if not _in(f"sub_{T}_{v1}"):
                continue
            for v2 in sorted(upC[T]):
                if v1 != v2:
                    proposals.append([(v1, A, T), (v2, T, C)])
        if len(proposals) >= cap:
            break

    if len(proposals) < cap:                                      # 2-transfer
        set_upC = set(upC)
        for T1 in sorted((t for t in downA if t not in (A, C)
                          and t not in _GATEWAY_IDS and _in(f"exit_{t}")), key=str):
            downT1 = down.get(T1, {})
            for T2 in sorted(set(downT1) & set_upC, key=str):
                if T2 in (A, C, T1) or T2 in _GATEWAY_IDS or not _in(f"exit_{T2}"):
                    continue
                for v1 in sorted(downA[T1]):
                    if not _in(f"sub_{T1}_{v1}"):
                        continue
                    for v2 in sorted(downT1[T2]):
                        if v2 == v1 or not _in(f"sub_{T2}_{v2}"):
                            continue
                        for v3 in sorted(upC[T2]):
                            if v3 != v2:
                                proposals.append([(v1, A, T1), (v2, T1, T2), (v3, T2, C)])
                if len(proposals) >= cap:
                    break
            if len(proposals) >= cap:
                break

    infos = []
    for legs in proposals:
        p = _realise_path(ctx, A, C, legs)
        if p is not None:
            infos.append(_walk_path(G, p))
    infos.append(_walk_path(G, sp_path))              # guarantee the optimum is present
    return infos


def _assign_logit(G: nx.DiGraph, od_long: pd.DataFrame, engine: str,
                  variant_seq: dict, vfreq: dict, k: int, window_min: float,
                  window_pct: float, max_transfers: int, max_examine: int,
                  theta: float) -> dict:
    """Logit route-choice assignment over an engine-generated candidate set.

    engine: 'table' — connection-table itinerary proposal (0/1/2-transfer + SP path),
                      pruned to the cost window by the per-origin SP oracle (fast);
            'yen'   — Yen k-shortest on the full graph (exact baseline, slow).
    Both engines score candidates with the same _walk_path (one cost model) and split
    demand by the same attractive-set choice model in _choose_pair, so they are
    directly comparable.
    """
    engine = (engine or 'table').strip().lower()
    prim = _empty_primitives()
    n_ok = n_unres = 0

    tablectx = _build_table_context(variant_seq or {}) if engine == 'table' else None

    for A, grp in od_long.groupby('origin_id', sort=False):
        A = int(A)
        src = f"entry_{A}"
        if src not in G:
            for r in grp.itertuples(index=False):
                prim['unresolved'].append({'origin_id': A, 'dest_id': int(r.dest_id),
                                           'trips': float(r.trips)})
                n_unres += 1
            continue
        dist_f = sp_paths = None
        if engine == 'table':
            dist_f, sp_paths = nx.single_source_dijkstra(G, src, weight='gc')

        for r in grp.itertuples(index=False):
            C, trips = int(r.dest_id), float(r.trips)
            tgt = f"exit_{C}"
            if tgt not in G:
                prim['unresolved'].append({'origin_id': A, 'dest_id': C, 'trips': trips})
                n_unres += 1
                continue
            if engine == 'yen':
                infos = _yen_infos(G, src, tgt, window_min, window_pct, k, max_examine)
            else:
                sp = sp_paths.get(tgt)
                if sp is None:
                    prim['unresolved'].append({'origin_id': A, 'dest_id': C, 'trips': trips})
                    n_unres += 1
                    continue
                infos = _table_infos(G, A, C, sp, tablectx, dist_f,
                                     window_min, window_pct)
            if _choose_pair(prim, A, C, trips, infos, vfreq, theta,
                            max_transfers, k, window_min, window_pct):
                n_ok += 1
            else:
                n_unres += 1
    print(f"  [logit/{engine}] routed {n_ok:,}, unresolved {n_unres:,} "
          f"(K={k}, window={window_min}min/{window_pct:.0%}, "
          f"max_transfers={max_transfers}, theta={theta}).")
    return prim


# ===============================================================================
# GATEWAY INJECTION  (inject external demand at boundary gateways onto the
# crossing services — stopping AND passing — split by the connection table)
# ===============================================================================

def _merge_primitives(dst: dict, src: dict) -> None:
    """Extend dst's primitive lists in place with src's."""
    for k in ('paths', 'segments', 'events', 'unresolved'):
        dst[k].extend(src[k])


def _norm_vk_parts(rid, did, vr) -> tuple:
    """Normalise (route_id, direction_id, variant_rank) so a connection-table
    triple matches a graph variant_key regardless of int/float/str formatting."""
    def _i(x):
        try:
            return str(int(float(x)))
        except (TypeError, ValueError):
            return str(x)
    return (str(rid), _i(did), _i(vr))


def _build_gateway_conn_lookup(conn_df: pd.DataFrame, variant_seq: dict) -> tuple:
    """Resolve connection-table rows to graph variants and renormalise weights.

    Returns (lookup, gateway_ids):
        lookup: dict[(gateway_id:int, role:str) -> list[(variant_key, weight,
                attach_skey)]], weights renormalised to sum 1.0 over the variants
                actually present in the graph for that (gateway, role).
        gateway_ids: set of boundary gateway ids with >=1 resolved connection.

    attach_skey: the gateway's own id when the variant stops there, else the
    variant's first (inbound) / last (outbound) in-network stop.
    """
    if conn_df is None or conn_df.empty:
        return {}, set()
    graph_vk = {}
    for vk in variant_seq:
        rid, did, vr = vk.rsplit('_', 2)
        graph_vk[_norm_vk_parts(rid, did, vr)] = vk

    raw = {}
    for r in conn_df.itertuples(index=False):
        vk = graph_vk.get(_norm_vk_parts(r.route_id, r.direction_id, r.variant_rank))
        if vk is None:
            continue
        seq = variant_seq.get(vk)
        if not seq:
            continue
        gid, role = int(r.gateway_station_id), str(r.direction_role)
        if gid in seq:
            attach = gid
        else:
            attach = seq[0] if role == 'inbound' else seq[-1]
        raw.setdefault((gid, role), []).append(
            (vk, float(r.boarding_weight), attach))

    lookup, gateway_ids = {}, set()
    for (gid, role), items in raw.items():
        tot = sum(w for _vk, w, _a in items)
        if tot <= 0:
            continue
        lookup[(gid, role)] = [(vk, w / tot, a) for (vk, w, a) in items]
        gateway_ids.add(gid)
    return lookup, gateway_ids


def _attach_gateways(G: nx.DiGraph, lookup: dict, vfreq: dict,
                     variant_seq: dict) -> dict:
    """Make each boundary gateway an ordinary terminal stop of its crossing
    services, so gateway ODs route through the normal choice model and crossing
    services pool at the transfer hub (replaces the forced gwsrc/gwsnk mechanism).

    For every (gateway gid, role, crossing service vk, attach stop): ensure the
    `entry_{gid}` / `exit_{gid}` portals and a `sub_{gid}_{vk}` node. A passing
    service (gid not in its sequence) gets a 0-cost terminal hop
    `sub_{attach}_{vk} -> sub_{gid}_{vk}` that delivers demand to the boundary
    (mirroring the old 0-cost gwsnk); a stopping service already has `gid` in its
    sequence, so its `sub_{gid}_{vk}` and board/alight edges exist. A board edge
    `entry_{gid} -> sub_{gid}_{vk}` lets gateway-origin demand board the crossing
    service (the choice model re-pools this wait over the combined crossing supply).

    Returns `routing_seq` = variant_seq with `gid` appended to each passing crossing
    service — used only for the table-engine reachability so it proposes
    A -> hub -> crossing -> gateway itineraries. Reported segments/events at gateway
    ids are dropped in _record_path, so downstream (service loads, capacity) is
    unchanged.
    """
    w_wait = _active_weights()['wait']
    routing_seq = {vk: list(s) for vk, s in variant_seq.items()}
    gw_vks = {}
    for (gid, _role), items in lookup.items():
        gw_vks.setdefault(gid, set()).update(vk for vk, _w, _a in items)
    gw_h = {gid: (60.0 / s if (s := sum(vfreq.get(v, 0.0) for v in vks)) > 0
                  else 60.0)
            for gid, vks in gw_vks.items()}

    for (gid, role), items in lookup.items():
        en, ex = f"entry_{gid}", f"exit_{gid}"
        if en not in G:
            G.add_node(en, kind='entry', station_key=gid)
            G.add_node(ex, kind='exit', station_key=gid)
        h_gw = gw_h.get(gid, 60.0)
        for vk, _w, attach in items:
            attach_sub = f"sub_{attach}_{vk}"
            if attach_sub not in G:
                continue
            gsub = f"sub_{gid}_{vk}"
            if gsub in G:                # stopping service: gid already a real stop
                continue
            # Passing service: synthesise the gateway terminal on this service. The
            # boundary sits BEFORE the first in-network stop for an inbound (origin)
            # service and AFTER the last for an outbound (destination) one.
            G.add_node(gsub, kind='sub', station_key=gid, variant_key=vk)
            seq = routing_seq[vk]
            if role == 'inbound':
                G.add_edge(gsub, attach_sub, gc=0.0, time=0.0)   # ride in
                G.add_edge(en, gsub, gc=w_wait * _wait_min(h_gw),
                           time=_wait_min(h_gw), board_freq=60.0 / h_gw,
                           var_freq=float(vfreq.get(vk, 0.0)))
                if gid not in seq:
                    seq.insert(0, gid)
            else:                                                # outbound: ride out
                G.add_edge(attach_sub, gsub, gc=0.0, time=0.0)
                G.add_edge(gsub, ex, gc=0.0, time=0.0)
                if gid not in seq:
                    seq.append(gid)
    return routing_seq


def _expand_gateway_od(od_long: pd.DataFrame, lookup: dict,
                       gateway_ids: set) -> tuple:
    """Gateway demand now routes through the normal choice model: boundary gateways
    are ordinary terminal stations (see _attach_gateways), so every OD pair —
    including gateway origins/destinations — is a normal row keyed by its
    entry_/exit_ portal. Returns (od_normal, empty od_gw); the empty second frame
    keeps the legacy `if not od_gw.empty` call sites as no-ops."""
    od_normal = od_long[['origin_id', 'dest_id', 'trips']].copy()
    od_gw = pd.DataFrame(
        columns=['origin_id', 'dest_id', 'trips', 'src_node', 'tgt_node'])
    return od_normal, od_gw


# ===============================================================================
# SKIMS
# ===============================================================================

def _build_skims(prim: dict, vfreq_report: dict, direct_freq_report: dict) -> dict:
    """Build journey-time (unweighted clock), generalised-cost (weighted),
    transfers, component and frequency long-format skims.
    Returns dict[name -> DataFrame(origin_id, dest_id, value)].

    journey_time / gc / transfers and the components (wait_t/ivt_t/xfer_t +
    weighted wait_g/ivt_g/xfer_g) are the **demand-weighted blend** across the
    chosen paths of each OD pair (Σ share·value / Σ share), so the reported value
    reflects how passengers actually split (decision 4), not the single best path.
    frequency:    the OD connection frequency carried on the paths (the pooled
                  attractive-set bottleneck from the choice model, equal across a
                  pair's paths); gateway ends are still min'd against the gateway's
                  combined crossing supply until the gateway rework lands. Falls
                  back to the legacy direct/single-variant bottleneck when the
                  primitive predates the per-path frequency column.
    """
    keys = ('journey_time', 'gc', 'frequency', 'freq_direct', 'freq_transfer',
            'transfers', 'wait_t', 'ivt_t', 'xfer_t', 'wait_g', 'ivt_g', 'xfer_g')
    empty = pd.DataFrame()
    if _prim_empty(prim['paths']):
        return {k: empty for k in keys}
    pdf = pd.DataFrame(prim['paths']).copy()
    pdf['share'] = pdf['share'].astype(float)

    # Demand-weighted blend per (origin, dest). den ~ 1.0 per pair (shares sum to
    # 1); dividing by it keeps the blend exact even if they don't.
    den = pdf.groupby(['origin_id', 'dest_id'])['share'].sum().rename('den')
    skims = {}
    for src, name in (('journey_time_min', 'journey_time'), ('gc_min', 'gc'),
                      ('n_transfers', 'transfers'),
                      ('wait_t', 'wait_t'), ('ivt_t', 'ivt_t'), ('xfer_t', 'xfer_t'),
                      ('wait_g', 'wait_g'), ('ivt_g', 'ivt_g'), ('xfer_g', 'xfer_g')):
        if src not in pdf.columns:
            continue
        num = (pdf[src].astype(float) * pdf['share']).groupby(
            [pdf['origin_id'], pdf['dest_id']]).sum().rename('num')
        m = pd.concat([num, den], axis=1).reset_index()
        m['value'] = m['num'] / m['den']
        skims[name] = m[['origin_id', 'dest_id', 'value']]

    # Frequency is an OD property (equal across a pair's paths): take the least-GC
    # representative's per-path pooled bottleneck. Gateways are ordinary stations
    # now, so their crossing supply is already in the per-path frequency — no floor.
    best = pdf[pdf['path_id'] == 0].copy()
    if 'gc_min' in best.columns:
        best = best.sort_values('gc_min').drop_duplicates(
            ['origin_id', 'dest_id'], keep='first')
    if 'frequency' in best.columns:
        skims['frequency'] = best[['origin_id', 'dest_id', 'frequency']].rename(
            columns={'frequency': 'value'})
        for col in ('freq_direct', 'freq_transfer'):
            if col in best.columns:
                skims[col] = best[['origin_id', 'dest_id', col]].rename(
                    columns={col: 'value'})
    else:
        def _freq(row):
            a, c, nt = int(row['origin_id']), int(row['dest_id']), int(row['n_transfers'])
            if nt == 0:
                f = direct_freq_report.get((a, c))
                if f is not None:
                    return f
            variants = [v for v in str(row['lines_used']).split('>') if v]
            fr = [vfreq_report[v] for v in variants if v in vfreq_report]
            return min(fr) if fr else np.nan
        fq = best[['origin_id', 'dest_id', 'lines_used', 'n_transfers']].copy()
        fq['value'] = fq.apply(_freq, axis=1)
        skims['frequency'] = fq[['origin_id', 'dest_id', 'value']]
    return skims


# ===============================================================================
# OUTPUT
# ===============================================================================

def _name(name_lookup: dict, skey) -> str:
    """Readable station name for an int id_point key; pass through other keys."""
    return name_lookup.get(str(skey), str(skey))


def _lines_used_names(lines_used: str) -> str:
    """'>'-joined variant keys -> '>'-joined line_short_names (route_id fallback)."""
    return '>'.join(_line_name(v) for v in str(lines_used).split('>') if v)


def _prepare_trip_df(rows: list, name_lookup: dict, tau: float,
                     id_cols: list) -> pd.DataFrame:
    """τ-scale a trip-bearing primitive and enrich with station names, route_id and
    line_short_name. Returns an empty DataFrame (no columns) when there is no data."""
    if _prim_empty(rows):
        return pd.DataFrame()
    df = pd.DataFrame(rows).copy()
    df['trips'] = df['trips'] * tau
    df = df[df['trips'] > 0].copy()
    for col in id_cols:
        df[col.replace('_id', '_name')] = df[col].map(lambda k: _name(name_lookup, k))
    if 'variant_key' in df.columns:
        df['route_id'] = df['variant_key'].map(_route_of)
        df['line_short_name'] = df['variant_key'].map(_line_name)
    if 'lines_used' in df.columns:
        df['lines_used_names'] = df['lines_used'].map(_lines_used_names)
    return df


# Excel's hard per-sheet cap is 1,048,576 rows including the header.
_EXCEL_MAX_DATA_ROWS = 1_048_575


def _write_trip_workbook(rows: list, name_lookup: dict, windows: list,
                         id_cols: list, out_path: str, label: str) -> None:
    """Write one trip-bearing primitive to a workbook with a sheet per service
    period (peak / off_peak / full_day), each τ-scaled and enriched.

    Falls back to one CSV per period (beside out_path) when any period exceeds
    Excel's per-sheet row limit — large primitives (logit segments/events) can
    run to millions of rows, which openpyxl cannot hold."""
    if _prim_empty(rows):
        print(f"    ({label}): no data — skipping {out_path}")
        return
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    period_dfs = [(window, _prepare_trip_df(rows, name_lookup, tau, id_cols))
                  for tau, window in windows]
    if max((len(df) for _, df in period_dfs), default=0) > _EXCEL_MAX_DATA_ROWS:
        stem = str(Path(out_path).with_suffix(''))
        for window, df in period_dfs:
            df.to_csv(f"{stem}_{window}.csv", index=False)
        print(f"    {label}: exceeds Excel's {_EXCEL_MAX_DATA_ROWS:,}-row limit "
              f"— wrote {len(period_dfs)} period CSVs beside "
              f"{Path(out_path).name} instead")
        return
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for window, df in period_dfs:
            df.to_excel(writer, sheet_name=window, index=False)
    print(f"    Saved -> {out_path}  ({label}, {len(windows)} period sheets)")


def _skim_to_matrix(df: pd.DataFrame, name_lookup: dict) -> pd.DataFrame:
    """Pivot a long skim (origin_id, dest_id, value) to a wide named station×station
    matrix. Returns an empty DataFrame when the skim is empty."""
    if df is None or df.empty:
        return pd.DataFrame()
    mat = df.pivot_table(index='origin_id', columns='dest_id', values='value',
                         aggfunc='mean')
    mat.index = [_name(name_lookup, s) for s in mat.index]
    mat.columns = [_name(name_lookup, s) for s in mat.columns]
    mat.index.name, mat.columns.name = 'origin', 'destination'
    return mat


def _write_skims_workbook(skims: dict, name_lookup: dict, out_path: str) -> None:
    """Write the station×station skim matrices (travel time, generalised cost,
    frequency, transfers) as one workbook with a sheet per skim."""
    sheets = [('journey_time', 'Travel_Time_min'),
              ('gc',           'Generalised_Cost_min'),
              ('frequency',    'Frequency_per_h'),
              ('transfers',    'Transfers')]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    wrote = 0
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for key, sheet in sheets:
            mat = _skim_to_matrix(skims.get(key), name_lookup)
            if mat.empty:
                pd.DataFrame({'(no data)': []}).to_excel(writer, sheet_name=sheet)
            else:
                mat.to_excel(writer, sheet_name=sheet)
                wrote += 1
    print(f"    Saved skims -> {out_path}  ({wrote}/{len(sheets)} matrix sheets)")


def _write_unresolved(rows: list, name_lookup: dict, out_path: str) -> None:
    if _prim_empty(rows):
        return
    df = pd.DataFrame(rows)
    df['origin_name'] = df['origin_id'].map(lambda k: _name(name_lookup, k))
    df['dest_name']   = df['dest_id'].map(lambda k: _name(name_lookup, k))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='unresolved', index=False)
    print(f"    Saved unresolved -> {out_path}  ({len(df):,} pairs, "
          f"trips={df['trips'].sum():,.1f})")


def _write_method_outputs(prim: dict, skims: dict, name_lookup: dict,
                          svc_network: str, method: str, windows: list,
                          use_cache: bool, write_workbooks: bool = True) -> None:
    """Write all per-method workbooks: one trip-table workbook per primitive (a
    sheet per service period), one skims workbook, and the unresolved-pairs file.
    With write_workbooks=False the three large trip workbooks are skipped (machine
    consumers read the parquet primitives); skims + unresolved always write."""
    out_dir = paths.get_assignment_method_dir(svc_network, method)
    os.makedirs(out_dir, exist_ok=True)

    if not write_workbooks:
        print("    WRITE_INT_WORKBOOKS = False — skipping path_assignment/"
              "segment_loads/station_events.xlsx (parquet primitives carry the data)")
    else:
        specs = [
            (prim['paths'],    ['origin_id', 'dest_id'],
             'path_assignment.xlsx', f'{method} paths'),
            (prim['segments'], ['origin_id', 'dest_id', 'from_id', 'to_id'],
             'segment_loads.xlsx',   f'{method} segments'),
            (prim['events'],   ['origin_id', 'dest_id', 'station_id'],
             'station_events.xlsx',  f'{method} events'),
        ]
        for rows, id_cols, fname, label in specs:
            fpath = os.path.join(out_dir, fname)
            if use_cache and Path(fpath).exists():
                print(f"    cached: {fpath}")
            else:
                _write_trip_workbook(rows, name_lookup, windows, id_cols, fpath, label)

    skims_path = os.path.join(out_dir, 'skims.xlsx')
    if use_cache and Path(skims_path).exists():
        print(f"    cached: {skims_path}")
    else:
        _write_skims_workbook(skims, name_lookup, skims_path)

    _write_unresolved(prim['unresolved'], name_lookup,
                      os.path.join(out_dir, 'unresolved_pairs.xlsx'))


# ===============================================================================
# REPORTING  (workbooks; heatmaps and Sankeys in the PLOTS section below)
# ===============================================================================

def _sa_station_ids() -> list:
    """Ordered, deduped study-area station id_points (the 12 corridor stations).

    Sourced from catchment_OD_preparation.SANKEY_CORRIDORS so the SA universe is
    defined in one place; corridors are listed in geographic order and may share a
    junction station (deduped here, first occurrence wins).
    """
    import catchment_OD_preparation as cod
    seen, out = set(), []
    for ids in cod.SANKEY_CORRIDORS.values():
        for s in ids:
            si = int(s)
            if si not in seen:
                seen.add(si)
                out.append(si)
    return out


# route_id (str) -> line_short_name for the active run; populated in
# passenger_routing from the graph builder's report_ctx. Empty -> id pass-through.
_ROUTE_NAME = {}

# Boundary gateway station ids for the active run; set in _build_routing_graph.
# Segments/events at these terminal stops are dropped from the recorded primitive
# (gateways are routing-only terminals, so downstream loads/capacity are unchanged).
_GATEWAY_IDS = set()


def _route_of(variant_key) -> str:
    """route_id portion of a 'rid_did_vrnk' variant key."""
    return str(variant_key).split('_')[0]


def _direction_of(variant_key) -> str:
    """direction_id portion of a 'rid_did_vrnk' variant key ('' if malformed)."""
    s = str(variant_key)
    return s.rsplit('_', 2)[1] if s.count('_') >= 2 else ''


def _route_id_of(variant_key) -> str:
    """route_id of a 'rid_did_vrnk' variant key — multi-token-safe.

    _route_of splits on the first '_', so it collapses multi-token route ids
    (IC_C/IC_D -> 'IC', 'ndc_103001_0_1' -> 'ndc') and mismatches the route_id in
    rail_segs_tt / _variant_keys_for, silently dropping those services from the
    service-load plots. rsplit keeps the full route id (consistent with
    _direction_of and the f"{route}_{direction}_" prefix in _variant_keys_for)."""
    return str(variant_key).rsplit('_', 2)[0]


def _wholeday_period_freq(vfreq: dict) -> dict:
    """Per-period frequency lookup that returns the WHOLE-DAY frequency (vfreq) for
    both peak and off-peak, so per-period service loads = full_day_trips x tau /
    whole_day_freq. New lines (e.g. NDC) carry no peak/off-peak split, so the real
    variant_period_freq is 0 for them; the whole-day freq is always present."""
    return {vk: {'peak': float(f), 'off_peak': float(f)} for vk, f in vfreq.items()}


def _line_name(variant_key_or_route) -> str:
    """Readable line_short_name for a variant key or route_id; route_id fallback."""
    rid = _route_of(variant_key_or_route)
    return _ROUTE_NAME.get(rid, rid)


def _routes_serving_sa(rail_segs_tt: pd.DataFrame, rail_stations: gpd.GeoDataFrame,
                       sa_ids) -> set:
    """route_ids whose physical stop pattern visits at least two SA stations.

    Args:
        rail_segs_tt:  full-day segment table (from_stop_id, to_stop_id, route_id).
        rail_stations: GDF with stop_id and id_point columns.
        sa_ids:        study-area station id_points.

    Returns:
        set[str] of qualifying route_ids (>= 2 distinct SA stops anywhere in pattern).
    """
    if rail_segs_tt is None or rail_segs_tt.empty:
        return set()
    stn = rail_stations.copy()
    stop_to_idp = dict(zip(stn['stop_id'].astype(str), stn['id_point'].astype(int)))
    sa = set(int(s) for s in sa_ids)
    seg = rail_segs_tt.copy()
    seg['route_id'] = seg['route_id'].astype(str)
    served = {}
    for r in seg.itertuples(index=False):
        rid = r.route_id
        for sid in (str(r.from_stop_id), str(r.to_stop_id)):
            idp = stop_to_idp.get(sid)
            if idp is not None:
                served.setdefault(rid, set()).add(idp)
    return {rid for rid, idps in served.items() if len(idps & sa) >= 2}


def _sanitize_sheet(name, used: set) -> str:
    """Excel-safe (<=31 char, no []:*?/\\) and unique-within-workbook sheet name."""
    s = ''.join('_' if c in '[]:*?/\\' else c for c in str(name))[:31]
    base = s or 'sheet'
    s = base
    i = 1
    while s.lower() in used:
        suffix = f"_{i}"
        s = base[:31 - len(suffix)] + suffix
        i += 1
    used.add(s.lower())
    return s


def _report_sa_top20(prim: dict, skims: dict, name_lookup: dict, svc_network: str,
                     method: str, sa_ids, tau_full_day: float, top_n: int = 20) -> None:
    """Per SA origin station: top-N destinations by full-day trips, with travel
    time, frequency, transfers and a direct (Y/N) flag. One sheet per station."""
    if _prim_empty(prim['paths']):
        print("    sa_top20: no paths — skipping SA top-N workbook")
        return
    pdf = pd.DataFrame(prim['paths'])
    od = pdf.groupby(['origin_id', 'dest_id'], as_index=False)['trips'].sum()
    od['trips'] = od['trips'] * tau_full_day

    def _skim_col(key, col):
        df = skims.get(key)
        if df is None or df.empty:
            return pd.DataFrame(columns=['origin_id', 'dest_id', col])
        return df.rename(columns={'value': col})

    od = od.merge(_skim_col('journey_time', 'travel_time_min'),
                  on=['origin_id', 'dest_id'], how='left')
    od = od.merge(_skim_col('frequency', 'frequency_per_h'),
                  on=['origin_id', 'dest_id'], how='left')
    od = od.merge(_skim_col('transfers', 'n_transfers'),
                  on=['origin_id', 'dest_id'], how='left')
    od['dest_name'] = od['dest_id'].map(lambda k: _name(name_lookup, k))
    od['direct'] = np.where(od['n_transfers'] == 0, 'Y', 'N')

    cols = ['dest_name', 'trips', 'travel_time_min', 'frequency_per_h',
            'n_transfers', 'direct']
    out = paths.get_assignment_report_xlsx(svc_network, method, 'matrix_sa_top20')
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    used, n_sheets = set(), 0
    with pd.ExcelWriter(out, engine='openpyxl') as writer:
        for o in sa_ids:
            sub = od[od['origin_id'] == int(o)]
            if sub.empty:
                continue
            top = sub.nlargest(top_n, 'trips')[cols]
            sheet = _sanitize_sheet(_name(name_lookup, o), used)
            top.to_excel(writer, sheet_name=sheet, index=False)
            n_sheets += 1
        if n_sheets == 0:
            pd.DataFrame({'(no SA origins routed)': []}).to_excel(
                writer, sheet_name='empty', index=False)
    print(f"    Saved sa_top20 -> {out}  ({n_sheets} SA station sheets)")


def _service_leg_endpoints(seg_df: pd.DataFrame) -> list:
    """Boarding / alighting station per service leg, from segment chains.

    Each (origin_id, dest_id, path_id, variant_key) group is one contiguous,
    loopless leg on a single variant. The boarding station is the from_id that is
    never a to_id within the leg; the alighting station is the to_id never a from_id.
    This recovers gross board/alight per service per station even at transfers (the
    flat events primitive only records the final alight).

    Returns:
        list[(variant_key, station_id, 'board'|'alight', trips)].
    """
    rows = []
    for (_o, _d, _p, vk), g in seg_df.groupby(
            ['origin_id', 'dest_id', 'path_id', 'variant_key'], sort=False):
        froms, tos = set(g['from_id']), set(g['to_id'])
        trips = float(g['trips'].iloc[0])
        for b in (froms - tos):
            rows.append((vk, b, 'board', trips))
        for a in (tos - froms):
            rows.append((vk, a, 'alight', trips))
    return rows


def _service_loads_dfs(prim: dict, qualifying_routes: set, name_lookup: dict,
                       tau: float) -> tuple:
    """Per-window (segment loads, station boardings/alightings) DataFrames for the
    services visiting >= 2 SA stations, τ-scaled. Returns (loads_df, boardings_df);
    either may be empty."""
    if _prim_empty(prim['segments']):
        return pd.DataFrame(), pd.DataFrame()
    seg = pd.DataFrame(prim['segments']).copy()
    seg['route_id'] = seg['variant_key'].map(_route_of)
    seg = seg[seg['route_id'].isin(qualifying_routes)].copy()
    if seg.empty:
        return pd.DataFrame(), pd.DataFrame()
    seg['trips'] = seg['trips'] * tau

    loads = (seg.groupby(['route_id', 'from_id', 'to_id'], as_index=False)['trips']
             .sum().rename(columns={'trips': 'pax_load'}))
    loads = loads[loads['pax_load'] > 0].copy()
    loads['line_short_name'] = loads['route_id'].map(_line_name)
    loads['from_name'] = loads['from_id'].map(lambda k: _name(name_lookup, k))
    loads['to_name'] = loads['to_id'].map(lambda k: _name(name_lookup, k))
    loads = loads[['route_id', 'line_short_name', 'from_id', 'to_id',
                   'from_name', 'to_name', 'pax_load']]

    legs = _service_leg_endpoints(seg)
    if not legs:
        return loads, pd.DataFrame()
    bdf = pd.DataFrame(legs, columns=['variant_key', 'station_id', 'event', 'trips'])
    bdf['route_id'] = bdf['variant_key'].map(_route_of)
    piv = bdf.pivot_table(index=['route_id', 'station_id'], columns='event',
                          values='trips', aggfunc='sum', fill_value=0.0).reset_index()
    for c in ('board', 'alight'):
        if c not in piv.columns:
            piv[c] = 0.0
    piv = piv.rename(columns={'board': 'boardings', 'alight': 'alightings'})
    piv['line_short_name'] = piv['route_id'].map(_line_name)
    piv['station_name'] = piv['station_id'].map(lambda k: _name(name_lookup, k))
    piv = piv[['route_id', 'line_short_name', 'station_id', 'station_name',
               'boardings', 'alightings']]
    return loads, piv


def _station_flows_df(prim: dict, name_lookup: dict, tau: float) -> pd.DataFrame:
    """Per-window per-station enter / exit / transfer / total flow DataFrame.

    total = enter + exit + 2*transfer mirrors the SBB station-frequency
    methodology: passengers changing trains are counted twice (once alighting,
    once boarding); pass-through and non-rail passengers are excluded.
    """
    if _prim_empty(prim['events']):
        return pd.DataFrame()
    ev = pd.DataFrame(prim['events']).copy()
    ev['trips'] = ev['trips'] * tau
    piv = ev.pivot_table(index='station_id', columns='event', values='trips',
                         aggfunc='sum', fill_value=0.0).reset_index()
    ren = {'board': 'enter', 'alight': 'exit', 'transfer': 'transfer'}
    for src in ren:
        if src not in piv.columns:
            piv[src] = 0.0
    piv = piv.rename(columns=ren)
    piv['total'] = piv['enter'] + piv['exit'] + 2.0 * piv['transfer']
    piv['station_name'] = piv['station_id'].map(lambda k: _name(name_lookup, k))
    piv = piv[['station_id', 'station_name', 'enter', 'exit', 'transfer', 'total']]
    return piv.sort_values('total', ascending=False)


def _write_period_workbook(per_window: dict, out_path: str, label: str,
                           note: str = '') -> None:
    """Write a {window -> DataFrame} mapping to a workbook with one sheet per
    service period. An optional note is added as a leading 'Methodology' sheet."""
    if all(df is None or df.empty for df in per_window.values()):
        print(f"    ({label}): no data — skipping {out_path}")
        return
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        if note:
            pd.DataFrame({'Methodology': [note]}).to_excel(
                writer, sheet_name='Methodology', index=False)
        for window, df in per_window.items():
            (df if df is not None else pd.DataFrame()).to_excel(
                writer, sheet_name=window, index=False)
    print(f"    Saved {label} -> {out_path}  ({len(per_window)} period sheets)")


# ===============================================================================
# PLOTS
# ===============================================================================

def _annotation_colour(rgba) -> str:
    """Black or white text for legibility on a cell of the given RGBA colour
    (WCAG relative-luminance threshold; also keeps greyscale prints readable)."""
    r, g, b = rgba[0], rgba[1], rgba[2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if lum < 0.55 else 'black'


def _draw_matrix_heatmap(data: np.ndarray, row_labels: list, col_labels: list,
                         title: str, cmap: str, out_png: str) -> None:
    """Annotated origin × destination heatmap for one numeric metric. Cell values
    are bold with per-cell black/white contrast; colour maps are perceptually
    uniform (colour-blind-safe, greyscale-readable)."""
    fig, ax = plt.subplots(figsize=(max(8.0, len(col_labels) * 0.65 + 3.0),
                                    max(5.0, len(row_labels) * 0.5 + 2.0)))
    im = ax.imshow(data, aspect='auto', cmap=cmap)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(title, fontsize=10)
    cmap_obj, norm = im.cmap, im.norm
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                tc = _annotation_colour(cmap_obj(norm(v)))
                ax.text(j, i, f"{v:.0f}" if abs(v) >= 1 else f"{v:.1f}",
                        ha='center', va='center', fontsize=9, fontweight='bold',
                        color=tc)
    ax.set_title(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved heatmap -> {out_png}")


# Connection-type categories for the interchange plot: code -> (colour, label).
# Colour-blind-safe blue/orange pairing; grey for same-station / no service.
_CONN_CATEGORIES = [
    (0,  '#2c7fb8', 'Direct (no interchange)'),
    (1,  '#e6550d', 'Interchange (>= 1 transfer)'),
    (-1, '#d9d9d9', 'Same station / no service'),
]


def _draw_interchange_heatmap(coded: np.ndarray, row_labels: list, col_labels: list,
                              title: str, out_png: str) -> None:
    """Categorical origin × destination plot: direct vs interchange vs
    same-station, colour-coded with a legend (no numbers, no colourbar)."""
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches

    order = [-1, 0, 1]
    colour_of = {code: col for code, col, _lab in _CONN_CATEGORIES}
    idx = np.full(coded.shape, np.nan)
    for k, code in enumerate(order):
        idx[coded == code] = k
    cmap = ListedColormap([colour_of[c] for c in order])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(max(8.0, len(col_labels) * 0.65 + 3.0),
                                    max(5.0, len(row_labels) * 0.5 + 2.0)))
    ax.imshow(idx, aspect='auto', cmap=cmap, norm=norm)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    handles = [mpatches.Patch(color=col, label=lab)
               for _code, col, lab in _CONN_CATEGORIES]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1.0), loc='upper left',
              frameon=False, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved interchange plot -> {out_png}")


def _draw_component_heatmap(total: np.ndarray, w: np.ndarray, v: np.ndarray,
                            t: np.ndarray, row_labels: list, col_labels: list,
                            title: str, out_png: str) -> None:
    """Origin × destination heatmap coloured by the total travel time, each cell
    annotated with its three components — W: wait, V: in-vehicle (IVT+IVWT),
    T: transfer. Colour scheme matches the total travel-time map (viridis_r)."""
    fig, ax = plt.subplots(figsize=(max(9.0, len(col_labels) * 0.9 + 3.0),
                                    max(6.0, len(row_labels) * 0.75 + 2.0)))
    im = ax.imshow(total, aspect='auto', cmap='viridis_r')
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('total travel time (min)', fontsize=10)
    cmap_obj, norm = im.cmap, im.norm
    for i in range(total.shape[0]):
        for j in range(total.shape[1]):
            tv = total[i, j]
            if np.isfinite(tv):
                tc = _annotation_colour(cmap_obj(norm(tv)))
                ax.text(j, i, f"W:{w[i, j]:.0f}\nV:{v[i, j]:.0f}\nT:{t[i, j]:.0f}",
                        ha='center', va='center', fontsize=8, fontweight='bold',
                        color=tc, linespacing=1.05)
    ax.set_title(f"{title}\nW: wait   V: in-vehicle (IVT+IVWT)   T: transfer",
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved component heatmap -> {out_png}")


def _draw_freq_dt_heatmap(fd: np.ndarray, ft: np.ndarray, row_labels: list,
                          col_labels: list, title: str, out_png: str) -> None:
    """Origin × destination service-frequency heatmap. Each cell lists the direct
    combined frequency (D) and/or the best transfer route's bottleneck (T) on
    separate rows; both appear only when the OD has both options. Coloured by the
    better (higher) of the two."""
    best = fd if ft is None else np.fmax(fd, ft)     # nan-aware max
    fig, ax = plt.subplots(figsize=(max(9.0, len(col_labels) * 0.8 + 3.0),
                                    max(6.0, len(row_labels) * 0.7 + 2.0)))
    im = ax.imshow(best, aspect='auto', cmap='viridis')
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('service frequency (trains/h)', fontsize=10)
    cmap_obj, norm = im.cmap, im.norm
    for i in range(best.shape[0]):
        for j in range(best.shape[1]):
            d = fd[i, j]
            t = ft[i, j] if ft is not None else float('nan')
            rows = []
            if np.isfinite(d):
                rows.append(f"D:{d:.0f}")
            if np.isfinite(t):
                rows.append(f"T:{t:.0f}")
            if not rows:
                continue
            bv = best[i, j]
            tc = _annotation_colour(cmap_obj(norm(bv))) if np.isfinite(bv) else 'black'
            ax.text(j, i, "\n".join(rows), ha='center', va='center', fontsize=9,
                    fontweight='bold', color=tc, linespacing=1.05)
    ax.set_title(f"{title}\nD: direct (combined)   T: transfer (bottleneck)",
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved frequency heatmap -> {out_png}")


def _plot_matrix_set(skims: dict, name_lookup: dict, rows: list, cols: list,
                     out_dir: str, suffix: str, scope: str) -> None:
    """Draw the heatmap set for the given origin rows × destination cols:
    total actual travel time (`matrix_travel`), total weighted GC
    (`matrix_generalised_cost`), their W/V/T component versions
    (`matrix_actual_cost_components`, `matrix_generalised_cost_components`), and the
    service-frequency plot (`matrix_frequency`, direct/transfer rows). Component
    plots are skipped when the component skims are absent (legacy primitive)."""
    if not rows or not cols:
        print(f"    heatmaps{suffix}: no origins/destinations — skipped")
        return

    def _mat(key):
        df = skims.get(key)
        if df is None or df.empty:
            return None
        piv = df.pivot_table(index='origin_id', columns='dest_id', values='value',
                             aggfunc='mean')
        return piv.reindex(index=rows, columns=cols)

    row_labels = [_name(name_lookup, s) for s in rows]
    col_labels = [_name(name_lookup, d) for d in cols]
    jt, gc = _mat('journey_time'), _mat('gc')
    wt, it, xt = _mat('wait_t'), _mat('ivt_t'), _mat('xfer_t')
    wg, ig, xg = _mat('wait_g'), _mat('ivt_g'), _mat('xfer_g')
    fd, ft = _mat('freq_direct'), _mat('freq_transfer')

    # Totals (one value per cell).
    if jt is not None:
        _draw_matrix_heatmap(jt.values, row_labels, col_labels,
                             f'Travel time — actual total (min){scope}', 'viridis_r',
                             os.path.join(out_dir, f'matrix_travel{suffix}.png'))
    if gc is not None:
        _draw_matrix_heatmap(gc.values, row_labels, col_labels,
                             f'Generalised cost — weighted total (min){scope}',
                             'viridis_r',
                             os.path.join(out_dir, f'matrix_generalised_cost{suffix}.png'))

    # Component versions (W/V/T per cell).
    if jt is not None and all(m is not None for m in (wt, it, xt)):
        _draw_component_heatmap(
            jt.values, wt.values, it.values, xt.values, row_labels, col_labels,
            f'Travel time — actual cost components (min){scope}',
            os.path.join(out_dir, f'matrix_actual_cost_components{suffix}.png'))
    if gc is not None and all(m is not None for m in (wg, ig, xg)):
        _draw_component_heatmap(
            gc.values, wg.values, ig.values, xg.values, row_labels, col_labels,
            f'Generalised cost — weighted cost components (min){scope}',
            os.path.join(out_dir, f'matrix_generalised_cost_components{suffix}.png'))

    if fd is not None or ft is not None:
        _draw_freq_dt_heatmap(
            fd.values if fd is not None else np.full((len(rows), len(cols)), np.nan),
            ft.values if ft is not None else None, row_labels, col_labels,
            f'Service frequency (trains/h){scope}',
            os.path.join(out_dir, f'matrix_frequency{suffix}.png'))


def _plot_sa_destination_matrices(prim: dict, skims: dict, name_lookup: dict,
                                  svc_network: str, method: str, sa_ids,
                                  n_dest: int = 15) -> None:
    """Heatmap sets over (i) SA origins × global top-N destinations and (ii) SA
    origins × SA destinations (the corridor where interventions are studied)."""
    if _prim_empty(prim['paths']):
        print("    heatmaps: no paths — skipping matrices")
        return
    pdf = pd.DataFrame(prim['paths'])
    dest_tot = pdf.groupby('dest_id')['trips'].sum().sort_values(ascending=False)
    top_dests = [int(d) for d in dest_tot.index[:n_dest]]
    sa_rows = [int(s) for s in sa_ids]

    out_dir = paths.get_assignment_plot_dir(svc_network, method)
    os.makedirs(out_dir, exist_ok=True)
    _plot_matrix_set(skims, name_lookup, sa_rows, top_dests, out_dir, '', '')
    _plot_matrix_set(skims, name_lookup, sa_rows, sa_rows, out_dir, '_sa',
                     ' — study area')


def _plot_corridor_service_sankeys(prim: dict, name_lookup: dict, svc_network: str,
                                   method: str, tau_full_day: float,
                                   top_n_dest: int = 5) -> None:
    """Per corridor: 3-column Sankey origin SA station -> line boarded at origin
    -> final destination (top-N + 'Other'), full-day trips."""
    import catchment_OD_preparation as cod
    if _prim_empty(prim['paths']) or _prim_empty(prim['events']):
        print("    sankeys: no paths/events — skipping")
        return
    pdf = pd.DataFrame(prim['paths'])
    ev = pd.DataFrame(prim['events'])
    board = ev[ev['event'] == 'board'].copy()
    board['board_route'] = board['variant_key'].map(_route_of)
    first_line = (board.groupby(['origin_id', 'dest_id', 'path_id'])['board_route']
                  .first().reset_index())
    flows = pdf.merge(first_line, on=['origin_id', 'dest_id', 'path_id'], how='left')
    flows['trips'] = flows['trips'] * tau_full_day

    out_dir = os.path.join(paths.get_assignment_plot_dir(svc_network, method), 'Sankey')
    os.makedirs(out_dir, exist_ok=True)
    for cname, ids in cod.SANKEY_CORRIDORS.items():
        _build_service_sankey(flows, ids, name_lookup, cname, method, out_dir,
                              top_n_dest)


def _build_service_sankey(flows: pd.DataFrame, station_ids, name_lookup: dict,
                          cname: str, method: str, out_dir: str,
                          top_n_dest: int) -> None:
    """One corridor's 3-column station -> service -> destination Sankey."""
    import catchment_OD_preparation as cod
    sset = set(int(s) for s in station_ids)
    sub = flows[flows['origin_id'].isin(sset) & flows['board_route'].notna()].copy()
    sub = sub[sub['origin_id'] != sub['dest_id']]
    sub = sub[sub['trips'] > 0]
    if sub.empty:
        print(f"    {cname}: no flow — skipped")
        return

    present = set(sub['origin_id'].astype(int))
    left_ids = [int(s) for s in station_ids if int(s) in present]
    routes = sorted(sub['board_route'].unique(),
                    key=lambda r: -sub[sub['board_route'] == r]['trips'].sum())
    dest_tot = sub.groupby('dest_id')['trips'].sum().sort_values(ascending=False)
    named = [int(d) for d in dest_tot.index[:top_n_dest]]
    named_set = set(named)

    left_idx = {s: i for i, s in enumerate(left_ids)}
    route_idx = {r: i for i, r in enumerate(routes)}
    right_idx = {d: i for i, d in enumerate(named)}
    other_i = len(named)

    l0, l1 = {}, {}
    for r in sub.itertuples(index=False):
        li = left_idx[int(r.origin_id)]
        ri = route_idx[r.board_route]
        di = right_idx[int(r.dest_id)] if int(r.dest_id) in named_set else other_i
        v = float(r.trips)
        l0[(li, ri)] = l0.get((li, ri), 0.0) + v
        l1[(ri, di)] = l1.get((ri, di), 0.0) + v

    links = ([(0, li, ri, v) for (li, ri), v in l0.items()]
             + [(1, ri, di, v) for (ri, di), v in l1.items()])
    columns = [[_name(name_lookup, s) for s in left_ids],
               [_line_name(r) for r in routes],
               [_name(name_lookup, d) for d in named] + ['Other']]
    title = (f"{cname.replace('_', ' ')} — station → service → destination "
             f"({method}, full-day)")
    out_pdf = os.path.join(out_dir, f"sankey_service_{cname}.pdf")
    out_html = os.path.join(out_dir, f"sankey_service_{cname}.html")
    cod._draw_sankey_multi_mpl(columns, links, title, out_pdf)
    cod._write_sankey_multi_html(columns, links, title, out_html)


def _variant_keys_for(route: str, direction: str, variant_seq: dict) -> list:
    """variant_keys of the given route_id + direction_id present in variant_seq."""
    pref = f"{route}_{direction}_"
    return [vk for vk in variant_seq if str(vk).startswith(pref)]


def _segment_period_freq(route: str, direction: str, period: str,
                         variant_seq: dict, variant_period_freq: dict) -> dict:
    """Per consecutive (from_skey, to_skey) departures/hour for a route+direction in
    a period, summed over the variants that traverse it (the segment's service)."""
    out = {}
    for vk in _variant_keys_for(route, direction, variant_seq):
        f = float(variant_period_freq.get(vk, {}).get(period, 0.0))
        if f <= 0:
            continue
        seq = variant_seq[vk]
        for a, b in zip(seq[:-1], seq[1:]):
            out[(a, b)] = out.get((a, b), 0.0) + f
    return out


def _service_dir_period(rseg_route: pd.DataFrame, route: str, direction: str,
                        period: str, tau: float, variant_seq: dict,
                        variant_period_freq: dict) -> dict:
    """Assemble one (direction, period) column: station sequence, per-station
    per-train board/alight counts, the per-train segment load (running integral of
    net boardings) and the line (trunk) frequency for the period."""
    seq = max((variant_seq[vk]
               for vk in _variant_keys_for(route, direction, variant_seq)),
              key=len, default=[])
    sub = rseg_route[rseg_route['direction_id'] == direction].copy()
    if sub.empty or len(seq) < 2:
        return {}
    sub['trips'] = sub['trips'] * tau
    seg_freq = _segment_period_freq(route, direction, period, variant_seq,
                                    variant_period_freq)

    board = {s: 0.0 for s in seq}
    alight = {s: 0.0 for s in seq}
    for vk, st, ev, tr in _service_leg_endpoints(sub):
        if ev == 'board':
            board[st] = board.get(st, 0.0) + tr
        else:
            alight[st] = alight.get(st, 0.0) + tr

    # Per-train so board/alight reconcile with the per-train segment loads:
    # boardings leave on downstream trains (f of the next segment), alightings
    # arrived on upstream trains (f of the previous segment).
    idx = {s: i for i, s in enumerate(seq)}
    board_pt, alight_pt = {}, {}
    for s in seq:
        i = idx[s]
        f_down = seg_freq.get((seq[i], seq[i + 1]), 0.0) if i < len(seq) - 1 else 0.0
        f_up = seg_freq.get((seq[i - 1], seq[i]), 0.0) if i > 0 else 0.0
        board_pt[s] = (board[s] / f_down) if f_down > 0 else 0.0
        alight_pt[s] = (alight[s] / f_up) if f_up > 0 else 0.0

    # Segment load = running integral of net per-train boardings, so the drawn
    # loads reconcile exactly with the board/alight figures. A direct per-segment
    # pax/dep_h count diverges mid-line where rare OD legs loop and traverse a
    # segment twice (the load counts both, board/alight mark only start+end); the
    # integral form keeps the profile internally consistent.
    load_factor = {}
    cum = 0.0
    for i in range(len(seq) - 1):
        cum += board_pt[seq[i]] - alight_pt[seq[i]]
        load_factor[(seq[i], seq[i + 1])] = cum

    line_freq = max(seg_freq.values()) if seg_freq else 0.0
    return {'seq': seq, 'load_factor': load_factor, 'board': board_pt,
            'alight': alight_pt, 'line_freq': line_freq}


def _draw_service_sequence(route: str, columns: list, name_lookup: dict,
                           out_png: str) -> None:
    """Draw one service's station-sequence load plot: each (period, direction) is a
    vertical station column; segment line width ∝ load factor (pax/train), with
    per-train board (↑) and alight (↓) counts per station. columns: list of
    (header, data-dict)."""
    cols = [(h, d) for h, d in columns if d]
    if not cols:
        return
    max_n = max(len(d['seq']) for _h, d in cols)
    all_lf = [v for _h, d in cols for v in d['load_factor'].values()
              if np.isfinite(v)]
    lf_max = max(all_lf) if all_lf else 1.0

    fig, ax = plt.subplots(figsize=(max(6.0, 3.2 * len(cols)),
                                    max(5.0, max_n * 0.55 + 2.0)))
    for ci, (header, d) in enumerate(cols):
        x = ci * 3.0
        seq = d['seq']
        y = {s: -i for i, s in enumerate(seq)}
        for (a, b), lf in d['load_factor'].items():
            lw = (max(0.8, 0.8 + 5.0 * (lf / lf_max))
                  if np.isfinite(lf) and lf_max > 0 else 0.8)
            ax.plot([x, x], [y[a], y[b]], color='#3182bd', lw=lw,
                    solid_capstyle='round', zorder=1)
            ym = (y[a] + y[b]) / 2.0
            if np.isfinite(lf) and lf >= 0.5:
                ax.text(x + 0.18, ym, f"{lf:.0f}",
                        fontsize=8, fontweight='bold', va='center', ha='left',
                        color='#08519c', zorder=3)
        for s in seq:
            ax.scatter([x], [y[s]], s=42, color='white', edgecolor='black',
                       zorder=2)
            nm = _name(name_lookup, s)
            ax.text(x - 0.18, y[s], nm[:22], fontsize=8, va='center', ha='right',
                    zorder=3)
            bd, al = d['board'].get(s, 0.0), d['alight'].get(s, 0.0)
            if bd >= 0.5:
                ax.text(x + 0.18, y[s] + 0.18, f"↑{bd:.0f}", fontsize=7,
                        color='#238b45', va='bottom', ha='left', zorder=3)
            if al >= 0.5:
                ax.text(x + 0.18, y[s] - 0.18, f"↓{al:.0f}", fontsize=7,
                        color='#cb181d', va='top', ha='left', zorder=3)
        ax.text(x, 0.8, header, fontsize=10, fontweight='bold', va='bottom',
                ha='center')

    ax.set_xlim(-1.4, (len(cols) - 1) * 3.0 + 1.6)
    ax.set_ylim(-(max_n - 1) - 1.2, 2.0)
    ax.axis('off')
    line_label = _line_name(route)
    ax.set_title(f"Service {line_label} — loads, boardings (↑) & alightings (↓) "
                 f"[all pax/train]", fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved service load plot -> {out_png}")


def _plot_service_load_sequences(prim: dict, variant_seq: dict,
                                 variant_period_freq: dict, qualifying_routes: set,
                                 name_lookup: dict, svc_network: str,
                                 method: str, windows: list) -> None:
    """One plot per service (route visiting >= 2 SA stations): the full station
    sequence per direction (dir 0 left, dir 1 right) with peak and off-peak shown as
    separate sequences. Load factor = period segment pax / period dep_h; a service
    running in only one period gets only that period's columns."""
    if _prim_empty(prim['segments']):
        print("    service load plots: no segments — skipped")
        return
    seg = pd.DataFrame(prim['segments']).copy()
    seg['route_id'] = seg['variant_key'].map(_route_id_of)
    seg['direction_id'] = seg['variant_key'].map(_direction_of)
    seg = seg[seg['route_id'].isin(qualifying_routes)].copy()
    if seg.empty:
        print("    service load plots: no qualifying-route segments — skipped")
        return

    tau_map = {w: tau for tau, w in windows}
    period_specs = [('peak', 'Peak'), ('off_peak', 'Off-peak')]
    out_dir = os.path.join(paths.get_assignment_plot_dir(svc_network, method),
                           'ServiceLoads')
    os.makedirs(out_dir, exist_ok=True)

    used = set()
    for route in sorted(qualifying_routes):
        rseg = seg[seg['route_id'] == route]
        if rseg.empty:
            continue
        directions = sorted(rseg['direction_id'].unique())
        columns = []
        for period, plabel in period_specs:
            tau = tau_map.get(period, 0.0)
            for direction in directions:
                data = _service_dir_period(rseg, route, direction, period, tau,
                                           variant_seq, variant_period_freq)
                if data:
                    try:
                        dlabel = str(int(float(direction)))
                    except (TypeError, ValueError):
                        dlabel = str(direction)
                    header = (f"{plabel}\nDir {dlabel} — "
                              f"{data.get('line_freq', 0.0):.0f} dep/h")
                    columns.append((header, data))
        if not columns:
            continue
        stem = _sanitize_sheet(f"service_load_{_line_name(route)}", used)
        _draw_service_sequence(route, columns, name_lookup,
                               os.path.join(out_dir, f"{stem}.png"))


# ---------------------------------------------------------------------------
# Phase-6C service-load DELTA plots (developed-vs-baseline, one per drawn route).
# ---------------------------------------------------------------------------
_SLOAD_GREEN = '#238b45'   # load added (or boarding)
_SLOAD_RED   = '#cb181d'   # load reduced (or alighting)
_SLOAD_BLUE  = '#3182bd'   # unchanged


def _sload_delta_str(delta: float, tol: float = 0.5) -> str:
    """Signed compact delta label ('+12' / '-3' / '0')."""
    return (f"{'+' if delta > 0 else ''}{delta:.0f}") if abs(delta) >= tol else '0'


def _service_load_seg(prim: dict) -> pd.DataFrame:
    """Segment table with multi-token-safe route_id/direction_id and numeric
    from_id/to_id. _replace_pairs / _persist_primitive str-cast object columns, so
    re-routed segments carry from_id/to_id as strings while variant_seq stations
    are ints — without coercion board/alight never match and the new-line loads
    integrate to 0."""
    seg = pd.DataFrame(prim['segments']).copy()
    if seg.empty:
        seg['route_id'] = []
        seg['direction_id'] = []
        return seg
    seg['route_id'] = seg['variant_key'].map(_route_id_of)
    seg['direction_id'] = seg['variant_key'].map(_direction_of)
    for c in ('from_id', 'to_id'):
        if c in seg.columns:
            seg[c] = pd.to_numeric(seg[c], errors='coerce')
    return seg


def _service_load_column(seg: pd.DataFrame, route: str, direction: str, period: str,
                         tau: float, variant_seq: dict, variant_period_freq: dict) -> dict:
    """One (period, direction) service-load column for a route, or {} if absent."""
    rseg = seg[seg['route_id'] == route]
    if rseg.empty:
        return {}
    return _service_dir_period(rseg, route, direction, period, tau,
                               variant_seq, variant_period_freq)


def _draw_service_sequence_delta(route: str, columns: list, name_lookup: dict,
                                 out_png: str, dev_label: str, tol: float = 0.5) -> bool:
    """Developed-vs-baseline station-sequence load plot: each (period, direction) is
    a vertical station column; a segment is coloured green (load added) / red (less)
    / blue (unchanged) and labelled `dev_total (delta)`. Per-station boardings (↑)
    and alightings (↓) carry their own (delta). columns: (header, dev, base)."""
    cols = [(h, dd, bd) for h, dd, bd in columns if dd and dd.get('load_factor')]
    if not cols:
        return False
    max_n = max(len(dd['seq']) for _h, dd, _b in cols)
    all_lf = [v for _h, dd, _b in cols for v in dd['load_factor'].values()
              if np.isfinite(v)]
    lf_max = max(all_lf) if all_lf else 1.0

    fig, ax = plt.subplots(figsize=(max(6.0, 3.4 * len(cols)),
                                    max(5.0, max_n * 0.55 + 2.0)))
    for ci, (header, dd, bd) in enumerate(cols):
        x = ci * 3.2
        seq = dd['seq']
        y = {s: -i for i, s in enumerate(seq)}
        blf = (bd or {}).get('load_factor', {})
        bboard = (bd or {}).get('board', {})
        balight = (bd or {}).get('alight', {})
        for (a, b), lf in dd['load_factor'].items():
            d = lf - blf.get((a, b), 0.0)
            col = (_SLOAD_GREEN if d > tol else
                   (_SLOAD_RED if d < -tol else _SLOAD_BLUE))
            lw = (max(0.8, 0.8 + 5.0 * (lf / lf_max))
                  if np.isfinite(lf) and lf_max > 0 else 0.8)
            ax.plot([x, x], [y[a], y[b]], color=col, lw=lw,
                    solid_capstyle='round', zorder=1)
            ym = (y[a] + y[b]) / 2.0
            if np.isfinite(lf) and (lf >= tol or abs(d) >= tol):
                ax.text(x + 0.2, ym, f"{lf:.0f} ({_sload_delta_str(d, tol)})",
                        fontsize=7.5, fontweight='bold', va='center', ha='left',
                        color=col, zorder=3)
        for s in seq:
            ax.scatter([x], [y[s]], s=42, color='white', edgecolor='black', zorder=2)
            ax.text(x - 0.2, y[s], _name(name_lookup, s)[:22], fontsize=8,
                    va='center', ha='right', zorder=3)
            bdv, alv = dd['board'].get(s, 0.0), dd['alight'].get(s, 0.0)
            bdd, ald = bdv - bboard.get(s, 0.0), alv - balight.get(s, 0.0)
            if bdv >= tol:
                ax.text(x + 0.2, y[s] + 0.16, f"↑{bdv:.0f} ({_sload_delta_str(bdd, tol)})",
                        fontsize=6.5, color=_SLOAD_GREEN, va='bottom', ha='left', zorder=3)
            if alv >= tol:
                ax.text(x + 0.2, y[s] - 0.16, f"↓{alv:.0f} ({_sload_delta_str(ald, tol)})",
                        fontsize=6.5, color=_SLOAD_RED, va='top', ha='left', zorder=3)
        ax.text(x, 0.8, header, fontsize=10, fontweight='bold', va='bottom', ha='center')

    ax.set_xlim(-1.6, (len(cols) - 1) * 3.2 + 2.0)
    ax.set_ylim(-(max_n - 1) - 1.2, 2.2)
    ax.axis('off')
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color=_SLOAD_GREEN, label='Load added'),
                       mpatches.Patch(color=_SLOAD_RED, label='Load reduced'),
                       mpatches.Patch(color=_SLOAD_BLUE, label='Unchanged')],
              loc='lower right', frameon=False, fontsize=8)
    ax.set_title(f"Service {route} — {dev_label}", fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved service load delta -> {out_png}")
    return True


def plot_service_load_delta(base_prim: dict, dev_prim: dict, base_variant_seq: dict,
                            dev_variant_seq: dict, base_vfreq: dict, dev_vfreq: dict,
                            name_lookup: dict, rail_segs_tt: pd.DataFrame,
                            rail_stations: gpd.GeoDataFrame, sa_ids,
                            affected_route_ids, svc_network: str, dev_label: str,
                            method: str, windows: list) -> None:
    """Developed-vs-baseline service-load delta plots for one svc-int.

    One png per drawn route. The drawn set is the study-area scope only: the
    svc-int's own line(s) (always), plus any service that visits >= 2 SA stations
    AND whose per-train loads actually change. Per-period loads use the whole-day
    frequency (full_day_trips x tau / whole_day_freq) so new lines without a
    peak/off-peak split still resolve."""
    if _prim_empty(dev_prim['segments']):
        print("    service load delta: no developed segments — skipped")
        return
    tol = 0.5
    base_seg = _service_load_seg(base_prim)
    dev_seg = _service_load_seg(dev_prim)
    base_vpf = _wholeday_period_freq(base_vfreq)
    dev_vpf = _wholeday_period_freq(dev_vfreq)
    qualifying = _routes_serving_sa(rail_segs_tt, rail_stations, sa_ids)
    aff = {str(r) for r in (affected_route_ids or [])}
    tau_map = {w: tau for tau, w in windows}
    period_specs = [('peak', 'Peak'), ('off_peak', 'Off-peak')]
    out_dir = os.path.join(paths.get_assignment_plot_dir(svc_network, method),
                           'ServiceLoads')
    os.makedirs(out_dir, exist_ok=True)

    routes = sorted(set(dev_seg['route_id'].dropna()) | set(base_seg['route_id'].dropna()))
    used, n = set(), 0
    for route in routes:
        is_aff_line = str(route) in aff
        if not is_aff_line and route not in qualifying:
            continue
        directions = sorted(dev_seg[dev_seg['route_id'] == route]['direction_id'].unique())
        if not directions:
            directions = sorted(base_seg[base_seg['route_id'] == route]['direction_id'].unique())
        columns, maxd = [], 0.0
        for period, plabel in period_specs:
            tau = tau_map.get(period, 0.0)
            for direction in directions:
                dd = _service_load_column(dev_seg, route, direction, period, tau,
                                          dev_variant_seq, dev_vpf)
                bd = _service_load_column(base_seg, route, direction, period, tau,
                                          base_variant_seq, base_vpf)
                if not dd:
                    continue
                try:
                    dlabel = str(int(float(direction)))
                except (TypeError, ValueError):
                    dlabel = str(direction)
                columns.append((f"{plabel}\nDir {dlabel} — "
                                f"{dd.get('line_freq', 0.0):.0f} dep/h", dd, bd))
                blf = (bd or {}).get('load_factor', {})
                for k, v in dd['load_factor'].items():
                    maxd = max(maxd, abs(v - blf.get(k, 0.0)))
                for k, v in blf.items():
                    if k not in dd['load_factor']:
                        maxd = max(maxd, abs(v))
        if not columns:
            continue
        if not is_aff_line and maxd < 1.0:
            continue   # qualifying but unaffected by this intervention
        stem = _sanitize_sheet(f"service_load_{route}", used)
        if _draw_service_sequence_delta(route, columns, name_lookup,
                                        os.path.join(out_dir, f"{stem}_delta.png"),
                                        dev_label, tol):
            n += 1
    print(f"    service load delta: {n} route(s) drawn -> {out_dir}")


# ---------------------------------------------------------------------------
# Phase-6C SA-matrix DELTA heatmaps (developed vs baseline): travel time and
# frequency (direct / transfer), for SA x SA and SA x baseline-top-15.
# ---------------------------------------------------------------------------
_MAT_GREEN = (0.80, 0.93, 0.80)   # improved
_MAT_RED   = (0.96, 0.80, 0.80)   # declined
_MAT_GREY  = (0.92, 0.92, 0.92)   # unchanged
_MAT_TOL   = 1e-6


def _matrix_pivot(skims: dict, key: str, rows: list, cols: list) -> np.ndarray:
    """Skim long-table -> dense [rows x cols] array (NaN where the OD is absent)."""
    df = skims.get(key)
    if df is None or df.empty:
        return np.full((len(rows), len(cols)), np.nan)
    piv = df.pivot_table(index='origin_id', columns='dest_id', values='value',
                         aggfunc='mean')
    return piv.reindex(index=rows, columns=cols).values.astype(float)


def _matrix_single_delta_str(dev: float, base: float) -> str:
    """Bracket text for a single-valued dev vs base ('new' / 'lost' / signed)."""
    if not np.isfinite(base):
        return 'new'
    if not np.isfinite(dev):
        return 'lost'
    d = dev - base
    return (f"{'+' if d > 0 else ''}{d:.0f}") if abs(d) >= _MAT_TOL else '0'


def _matrix_freq_row(label: str, dev: float, base: float):
    """'D:6 (+2)' style row; a missing service counts as 0 so a gained/lost
    connection shows its signed amount (+2 / -2) rather than new/lost."""
    dv = dev if np.isfinite(dev) else 0.0
    bv = base if np.isfinite(base) else 0.0
    if dv == 0 and bv == 0:
        return None
    d = dv - bv
    ds = (f"{'+' if d > 0 else ''}{d:.0f}") if abs(d) >= _MAT_TOL else '0'
    return f"{label}:{dv:.0f} ({ds})"


def _draw_matrix_axes(n_r: int, n_c: int, rlab: list, clab: list, title: str):
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(max(8.0, n_c * 0.75 + 3.0),
                                    max(5.0, n_r * 0.55 + 2.0)))
    ax.set_xticks(range(n_c)); ax.set_xticklabels(clab, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_r)); ax.set_yticklabels(rlab, fontsize=8)
    ax.legend(handles=[mpatches.Patch(color=_MAT_GREEN, label='Improved'),
                       mpatches.Patch(color=_MAT_RED, label='Declined'),
                       mpatches.Patch(color=_MAT_GREY, label='Unchanged')],
              bbox_to_anchor=(1.01, 1.0), loc='upper left', frameon=False, fontsize=8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    return fig, ax


def _draw_matrix_save(fig, ax, bg: np.ndarray, txt: list, out: str) -> None:
    ax.imshow(bg, aspect='auto')
    for i in range(bg.shape[0]):
        for j in range(bg.shape[1]):
            if txt[i][j]:
                ax.text(j, i, txt[i][j], ha='center', va='center', fontsize=7.2,
                        fontweight='bold', color='black', linespacing=1.0)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f"    Saved matrix delta -> {out}")


def _draw_matrix_travel_delta(dev: np.ndarray, base: np.ndarray, rlab: list,
                              clab: list, title: str, out: str) -> None:
    """Single-value (lower-is-better) delta heatmap: improved green / declined red
    / unchanged grey; a newly-served OD is green ('new'), a lost OD red ('lost')."""
    n_r, n_c = dev.shape
    bg = np.ones((n_r, n_c, 3)); txt = [['' for _ in range(n_c)] for _ in range(n_r)]
    for i in range(n_r):
        for j in range(n_c):
            dv, bv = dev[i, j], base[i, j]
            if not np.isfinite(dv) and not np.isfinite(bv):
                continue
            if np.isfinite(dv) and not np.isfinite(bv):
                bg[i, j] = _MAT_GREEN; txt[i][j] = f"{dv:.0f}\n(new)"; continue
            if np.isfinite(bv) and not np.isfinite(dv):
                bg[i, j] = _MAT_RED; txt[i][j] = "-\n(lost)"; continue
            d = dv - bv
            bg[i, j] = (_MAT_GREEN if d < -_MAT_TOL else
                        (_MAT_RED if d > _MAT_TOL else _MAT_GREY))
            txt[i][j] = f"{dv:.0f}\n({_matrix_single_delta_str(dv, bv)})"
    fig, ax = _draw_matrix_axes(n_r, n_c, rlab, clab, title)
    _draw_matrix_save(fig, ax, bg, txt, out)


def _draw_matrix_freq_delta(dD, dT, bD, bT, rlab, clab, title, out) -> None:
    """Frequency D/T delta heatmap with direct-priority colouring: more direct
    service is an improvement (green) even if transfer frequency drops; only when
    direct is unchanged does transfer decide. A T->D switch is therefore green."""
    n_r, n_c = dD.shape
    bg = np.ones((n_r, n_c, 3)); txt = [['' for _ in range(n_c)] for _ in range(n_r)]
    for i in range(n_r):
        for j in range(n_c):
            dD0 = dD[i, j] if np.isfinite(dD[i, j]) else 0.0
            bD0 = bD[i, j] if np.isfinite(bD[i, j]) else 0.0
            dT0 = dT[i, j] if np.isfinite(dT[i, j]) else 0.0
            bT0 = bT[i, j] if np.isfinite(bT[i, j]) else 0.0
            if not (dD0 or bD0 or dT0 or bT0):
                continue
            ddel, tdel = dD0 - bD0, dT0 - bT0
            if ddel > _MAT_TOL:
                cat = 1
            elif ddel < -_MAT_TOL:
                cat = -1
            elif tdel > _MAT_TOL:
                cat = 1
            elif tdel < -_MAT_TOL:
                cat = -1
            else:
                cat = 0
            bg[i, j] = _MAT_GREEN if cat == 1 else (_MAT_RED if cat == -1 else _MAT_GREY)
            rows = [r for r in (_matrix_freq_row('D', dD[i, j], bD[i, j]),
                                _matrix_freq_row('T', dT[i, j], bT[i, j])) if r]
            txt[i][j] = "\n".join(rows)
    fig, ax = _draw_matrix_axes(n_r, n_c, rlab, clab, title)
    _draw_matrix_save(fig, ax, bg, txt, out)


def plot_service_matrix_delta(base_skims: dict, dev_skims: dict, base_paths,
                              name_lookup: dict, sa_ids, svc_network: str,
                              dev_label: str, method: str) -> None:
    """Developed-vs-baseline SA-matrix heatmaps (travel time + frequency D/T) for
    SA x SA and SA x baseline-top-15 destinations. The top-15 destinations are
    fixed from the baseline so the base and dev columns align."""
    sa = sorted(int(s) for s in sa_ids)
    if base_paths is None or len(base_paths) == 0:
        print("    matrix delta: no baseline paths — skipped")
        return
    dest_tot = (pd.DataFrame(base_paths).groupby('dest_id')['trips'].sum()
                .sort_values(ascending=False))
    top15 = [int(d) for d in dest_tot.index[:15]]
    out_dir = paths.get_assignment_plot_dir(svc_network, method)
    os.makedirs(out_dir, exist_ok=True)
    for sname, rows, cols in (('sa', sa, sa), ('top15', sa, top15)):
        rlab = [_name(name_lookup, s) for s in rows]
        clab = [_name(name_lookup, c) for c in cols]
        _draw_matrix_travel_delta(
            _matrix_pivot(dev_skims, 'journey_time', rows, cols),
            _matrix_pivot(base_skims, 'journey_time', rows, cols), rlab, clab,
            f"Travel time (min) — {dev_label}",
            os.path.join(out_dir, f"matrix_travel_{sname}_delta.png"))
        _draw_matrix_freq_delta(
            _matrix_pivot(dev_skims, 'freq_direct', rows, cols),
            _matrix_pivot(dev_skims, 'freq_transfer', rows, cols),
            _matrix_pivot(base_skims, 'freq_direct', rows, cols),
            _matrix_pivot(base_skims, 'freq_transfer', rows, cols), rlab, clab,
            f"Frequency D/T (trains/h) — {dev_label}",
            os.path.join(out_dir, f"matrix_frequency_{sname}_delta.png"))


# ---------------------------------------------------------------------------
# Phase-6C per-cell accessibility improve/decline map (PT-Feeder). Needs the
# routed skims, so it lives here rather than in catchment_allocate (6A).
# ---------------------------------------------------------------------------
_CELL_PLOT_BASE_CACHE: dict = {}   # (base_svc_network, infra) -> base alloc + map base layers


def _cell_pt_feeder_allocation(network_name: str) -> pd.DataFrame:
    """Load a network's PT-Feeder cell allocation (RELI, cell coords, chosen
    station, access time); None when absent."""
    p = os.path.join(catchment_base.CATCHMENT_DATA_DIR, network_name,
                     'PT_Feeder', 'allocation_pt_feeder.parquet')
    if not os.path.exists(p):
        return None
    a = pd.read_parquet(p)
    a['id_point'] = a['id_point'].astype(int)
    return a[['RELI', 'E_KOORD', 'N_KOORD', 'id_point', 'access_time_sec']]


def _cell_skim_col() -> str:
    return 'gc_min' if settings.TRAVEL_COST_METHOD == 'calibrated' else 'journey_time_min'


def _cell_best_path_od(prim: dict) -> pd.DataFrame:
    """Per-(origin, dest) BEST-path skim (min over the choice set) + OD trips."""
    pdf = prim['paths']
    if _prim_empty(pdf):
        return pd.DataFrame(columns=['origin_id', 'dest_id', 'skim', 'trips'])
    col = _cell_skim_col()
    g = pd.DataFrame(pdf).groupby(['origin_id', 'dest_id'])
    return pd.DataFrame({'skim': g[col].min(), 'trips': g['trips'].sum()}).reset_index()


def _cell_onward_mean(od: pd.DataFrame) -> pd.Series:
    """T_onward(s): trip-weighted mean onward skim per origin station."""
    if od.empty:
        return pd.Series(dtype=float)
    return od.groupby('origin_id').apply(
        lambda d: np.average(d['skim'], weights=d['trips'])
        if d['trips'].sum() > 0 else np.nan)


def _cell_onward_dev(base_od: pd.DataFrame, dev_od: pd.DataFrame) -> pd.Series:
    """Onward skim with BASE demand weights and the DEV skim (no floor), so the
    cell-GC delta isolates the intervention's travel-cost change at fixed demand.
    Pairs absent from the dev skim fall back to the base skim."""
    if base_od.empty:
        return pd.Series(dtype=float)
    m = base_od.merge(dev_od[['origin_id', 'dest_id', 'skim']].rename(
        columns={'skim': 'skim_dev'}), on=['origin_id', 'dest_id'], how='left')
    m['skim_dev'] = m['skim_dev'].fillna(m['skim'])
    return m.groupby('origin_id').apply(
        lambda d: np.average(d['skim_dev'], weights=d['trips'])
        if d['trips'].sum() > 0 else np.nan)


def _service_station_set(rail_dir: str, route_ids: set) -> set:
    """Station Numbers served by any of `route_ids` in a rail network dir
    (rail_segments from_stop_nr / to_stop_nr where GTFS_ID == route_id)."""
    if not route_ids:
        return set()
    import fiona
    sp = os.path.join(rail_dir, 'rail_segments.gpkg')
    if not os.path.exists(sp):
        return set()
    sg = pd.concat([gpd.read_file(sp, layer=L) for L in fiona.listlayers(sp)],
                   ignore_index=True)
    sel = sg[sg['GTFS_ID'].astype(str).isin({str(r) for r in route_ids})]
    return (set(pd.to_numeric(sel['from_stop_nr'], errors='coerce').dropna().astype(int))
            | set(pd.to_numeric(sel['to_stop_nr'], errors='coerce').dropna().astype(int)))


_CELL_DEP_CACHE: dict = {}


def _station_total_dep(rail_dir: str) -> dict:
    """Whole-day stopping departures per station Number for a rail network dir.

    station_total_dep(S) = Σ rail_lines.total_dep over every
    (route_id, direction_id, variant_rank) whose rail_segments call at S
    (from_stop_nr / to_stop_nr). Mirrors the station_total_dep measure in
    catchment_allocate._compute_station_freq_penalties — the segment key
    `GTFS_ID` equals rail_lines `route_id`. Returns {} when files are absent."""
    import fiona
    lp = os.path.join(rail_dir, 'rail_lines.gpkg')
    sp = os.path.join(rail_dir, 'rail_segments.gpkg')
    if not (os.path.exists(lp) and os.path.exists(sp)):
        return {}
    ln = pd.concat([gpd.read_file(lp, layer=L) for L in fiona.listlayers(lp)],
                   ignore_index=True)
    sg = pd.concat([gpd.read_file(sp, layer=L) for L in fiona.listlayers(sp)],
                   ignore_index=True)
    ln['route_id'] = ln['route_id'].astype(str)
    ln['total_dep'] = pd.to_numeric(ln['total_dep'], errors='coerce').fillna(0.0)
    sg['GTFS_ID'] = sg['GTFS_ID'].astype(str)
    for df in (ln, sg):
        df['direction_id'] = pd.to_numeric(df['direction_id'], errors='coerce')
        df['variant_rank'] = pd.to_numeric(df['variant_rank'], errors='coerce')
    key = ['direction_id', 'variant_rank']
    fr = sg[['from_stop_nr', 'GTFS_ID'] + key].rename(columns={'from_stop_nr': 'nr'})
    to = sg[['to_stop_nr', 'GTFS_ID'] + key].rename(columns={'to_stop_nr': 'nr'})
    pairs = pd.concat([fr, to], ignore_index=True).drop_duplicates()
    pairs = pairs.merge(ln[['route_id'] + key + ['total_dep']],
                        left_on=['GTFS_ID'] + key, right_on=['route_id'] + key,
                        how='left').dropna(subset=['total_dep'])
    dep = pairs.groupby('nr')['total_dep'].sum()
    return {int(k): float(v) for k, v in dep.items() if pd.notna(k)}


def _station_total_dep_cached(rail_dir: str) -> dict:
    if rail_dir not in _CELL_DEP_CACHE:
        _CELL_DEP_CACHE[rail_dir] = _station_total_dep(rail_dir)
    return _CELL_DEP_CACHE[rail_dir]


def plot_cell_accessibility_delta(base_prim: dict, dev_prim: dict,
                                  base_svc_network: str, svc_int_network: str,
                                  svc_int_id: str, infra_version: str,
                                  combo: str, method: str) -> None:
    """Phase-6C per-cell accessibility improve/decline map (PT-Feeder).

    Service-scoped generalised-cost colouring (decision 2026-06-23): only cells
    that route to a station served by the AFFECTED SERVICE(S) — every stop up- and
    down-line of the intervention, base ∪ dev, not just the touched stop — are
    coloured. Such a cell is GREEN when its generalised cost (access + demand-
    weighted onward skim, GC-min) FALLS dev vs base, RED when it RISES (±0.5-min
    tolerance), so frequency, in-vehicle time and transfers are all captured (e.g.
    a new direct connection's removed transfer, an extension's shorter ride, an
    express drop's penalty at the dropped stop and benefit to through-riders).
    Cells whose chosen station CHANGED are a distinct 're-assigned' category;
    cells gaining / losing all rail access are green / red; out-of-scope cells are
    unchanged. The directly touched stops are drawn as coloured circles by their
    whole-day departure change. Needs the routing primitives (base_prim, dev_prim)
    and the dev 6A allocation; skips when the dev allocation is absent."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from shapely.geometry import box
    import svc_ints_orchestrator as so

    if not infra_version:
        print("    cell accessibility delta: no infra version — skipped")
        return
    dev_alloc = _cell_pt_feeder_allocation(svc_int_network)
    if dev_alloc is None:
        print("    cell accessibility delta: no developed allocation — skipped")
        return

    bkey = (base_svc_network, infra_version)
    if bkey not in _CELL_PLOT_BASE_CACHE:
        nodes = gpd.read_file(os.path.join(
            paths.resolve_infra_dir(infra_version), 'nodes.gpkg'))
        boundary = catchment_allocate._load_catchment_boundary()
        _CELL_PLOT_BASE_CACHE[bkey] = (
            _cell_pt_feeder_allocation(base_svc_network), boundary,
            catchment_allocate._load_lakes_for_extent(boundary, scope='ca'),
            {int(n): g for n, g in zip(nodes['Number'], nodes.geometry)},
            {str(nm): int(n) for nm, n in zip(nodes['Name'], nodes['Number'])
             if pd.notna(nm)})
    base_alloc, boundary, lakes, node_pts, name_to_nr = _CELL_PLOT_BASE_CACHE[bkey]
    if base_alloc is None:
        print("    cell accessibility delta: no baseline allocation — skipped")
        return

    no_pt = catchment_allocate.NO_PT_ID
    int_type = svc_int_id.split('_')[0]
    try:
        _rec = so.read_record(int_type, svc_int_id, network=combo)
    except Exception:
        _rec = None

    # Directly touched stops (for the coloured-circle markers).
    _aff = (_rec or {}).get('affected_stations') or []
    if isinstance(_aff, str):
        _aff = _aff.split(',')
    aff_names = [str(s).strip() for s in _aff if str(s).strip()]
    aff_nodes = {name_to_nr[n] for n in aff_names if n in name_to_nr}

    base_rail = os.path.join(paths.RAIL_LINES_DIR, base_svc_network, infra_version)
    dev_rail = os.path.join(str(paths.get_svc_int_network_dir(svc_int_id, combo)), 'Merged')

    # Colouring scope = every station served by the affected service(s), base ∪ dev,
    # so up-/down-line stations (not just the touched stop) are included.
    _svc = (_rec or {}).get('affected_services') or []
    if isinstance(_svc, str):
        _svc = _svc.split(',')
    routes = {str(s).strip() for s in _svc if str(s).strip()}
    if (_rec or {}).get('route_id'):
        routes.add(str(_rec['route_id']))
    scope = (_service_station_set(base_rail, routes)
             | _service_station_set(dev_rail, routes))

    # Departure change at the directly-touched stops (for the circle colour).
    base_dep = _station_total_dep_cached(base_rail)
    dev_dep = _station_total_dep(dev_rail)
    dep_sign = {nr: dev_dep.get(nr, 0.0) - base_dep.get(nr, 0.0) for nr in aff_nodes}

    # Per-cell generalised-cost change (access + onward), base vs dev, at fixed demand.
    base_od = _cell_best_path_od(base_prim)
    base_onward = _cell_onward_mean(base_od)
    dev_onward = _cell_onward_dev(base_od, _cell_best_path_od(dev_prim))
    tol = 0.5

    def _metric(alloc, onward):
        m = alloc.copy()
        m['onward'] = m['id_point'].map(onward)
        m['metric'] = np.where(m['id_point'] == no_pt, np.nan,
                               m['access_time_sec'] / 60.0 + m['onward'])
        return m

    b = _metric(base_alloc, base_onward).rename(columns={'id_point': 'sb', 'metric': 'mb'})
    d = _metric(dev_alloc, dev_onward).rename(columns={'id_point': 'sd', 'metric': 'md'})
    c = b[['RELI', 'E_KOORD', 'N_KOORD', 'sb', 'mb']].merge(
        d[['RELI', 'sd', 'md']], on='RELI', how='inner')
    both = (c['sb'] != no_pt) & (c['sd'] != no_pt)
    c['switched'] = both & (c['sb'] != c['sd'])
    c['delta'] = c['md'] - c['mb']
    # Only cells routing to a station of the affected service(s) are coloured.
    in_scope = c['sb'].isin(scope) | c['sd'].isin(scope)
    # 1 improved (GC down) / -1 declined (GC up) / 2 re-assigned (blue) / 0 unchanged
    cat = np.zeros(len(c), dtype=int)
    gc_cell = in_scope & both & (~c['switched'])
    cat = np.where(gc_cell & (c['delta'] < -tol), 1, cat)
    cat = np.where(gc_cell & (c['delta'] > tol), -1, cat)
    cat = np.where(in_scope & (c['sb'] == no_pt) & (c['sd'] != no_pt), 1, cat)
    cat = np.where(in_scope & (c['sb'] != no_pt) & (c['sd'] == no_pt), -1, cat)
    cat = np.where(in_scope & c['switched'], 2, cat)
    c['cat'] = cat

    gdf = gpd.GeoDataFrame(
        c, geometry=[box(e, n, e + 100, n + 100)
                     for e, n in zip(c['E_KOORD'], c['N_KOORD'])], crs=_CODEBASE_CRS)
    green, red, grey, blue = ((0.62, 0.85, 0.62), (0.95, 0.70, 0.70),
                              (0.86, 0.86, 0.86), (0.40, 0.55, 0.80))
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.set_facecolor('#E8E8E8')
    bnd = gpd.GeoDataFrame(geometry=[boundary], crs=_CODEBASE_CRS)
    bnd.plot(ax=ax, color='white', edgecolor='none', zorder=0)
    for val, col in ((0, grey), (1, green), (-1, red), (2, blue)):
        sub = gdf[gdf['cat'] == val]
        if not sub.empty:
            sub.plot(ax=ax, color=col, edgecolor='none', zorder=2)
    if lakes is not None and not lakes.empty:
        lakes.plot(ax=ax, color='#A8D8EA', edgecolor='none', zorder=3)
    bnd.boundary.plot(ax=ax, color='black', linewidth=1.8, linestyle='--', zorder=5)

    icol = so._TYPE_COLOR.get(int_type, '#000000')
    iseg = so._load_delta_segments(
        Path(paths.get_svc_int_network_dir(svc_int_id, combo)) / infra_version
        / 'rail_segments.gpkg')
    if iseg is not None and not iseg.empty:
        iseg = iseg.to_crs(_CODEBASE_CRS) if iseg.crs else iseg
        iseg.plot(ax=ax, color=icol, linewidth=2.8, zorder=5, alpha=0.95)

    chosen = set(c.loc[c['sd'] != no_pt, 'sd']) | set(c.loc[c['sb'] != no_pt, 'sb'])
    other = [s for s in chosen if s not in aff_nodes and s in node_pts]
    ax.scatter([node_pts[s].x for s in other], [node_pts[s].y for s in other],
               s=22, c='white', edgecolors='black', linewidths=0.8,
               marker='o', zorder=6)
    # Affected station(s): same circle as every other station, just filled by its
    # whole-day departure change (green +dep / red -dep), drawn on top.
    for s in sorted(aff_nodes):
        if s not in node_pts:
            continue
        ds = dep_sign.get(s, 0.0)
        scol = ('#1a8a1a' if ds > 0 else '#c0392b' if ds < 0 else '#555555')
        ax.scatter([node_pts[s].x], [node_pts[s].y], s=22, c=scol,
                   edgecolors='black', linewidths=0.8, marker='o', zorder=7)
    bx0, by0, bx1, by1 = boundary.bounds
    ax.set_xlim(bx0 - 200, bx1 + 200)
    ax.set_ylim(by0 - 200, by1 + 200)
    ax.set_aspect('equal')
    ax.set_xlabel('E [m]')
    ax.set_ylabel('N [m]')
    catchment_allocate._add_map_elements(ax)
    ax.set_title(f'Accessibility change — {svc_int_id}', fontsize=14)
    ax.legend(handles=[
        Patch(facecolor=green, label='Improved (lower generalised cost)'),
        Patch(facecolor=red, label='Declined (higher generalised cost)'),
        Patch(facecolor=blue, label='Re-assigned station'),
        Patch(facecolor=grey, label='Unchanged / out of scope'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=8, label='Rail station'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=grey,
               markeredgecolor='black', markersize=8,
               label='Affected stop (green +dep / red −dep)'),
        Line2D([0], [0], color=icol, lw=2.8,
               label=f'Service intervention ({int_type})'),
    ], loc='upper right', fontsize=9, framealpha=0.9)

    out_dir = paths.get_assignment_plot_dir(svc_int_network, method)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'cell_accessibility_change.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"    cell accessibility delta: {int((cat == 1).sum()):,} improved / "
          f"{int((cat == -1).sum()):,} declined / {int((cat == 2).sum()):,} "
          f"re-assigned -> {out}")


def _write_reports(prim: dict, skims: dict, rail_segs_tt: pd.DataFrame,
                   rail_stations: gpd.GeoDataFrame, name_lookup: dict,
                   svc_network: str, method: str, windows: list, sa_ids,
                   variant_seq: dict, variant_period_freq: dict,
                   make_plots: bool = True,
                   plot_service_loads: bool = True) -> None:
    """Reporting workbooks + plots for one assignment method.

    Skim matrices and heatmaps are τ-independent and written once; trip-bearing
    service loads and station flows become one workbook each (a sheet per service
    period); corridor Sankeys use full-day trips; per-service load plots split peak
    vs off-peak. Built from the in-memory primitives.
    """
    print(f"\n[Step 5] Reporting{' & plots' if make_plots else ''} — {method} ...")
    out_dir = paths.get_assignment_method_dir(svc_network, method)
    os.makedirs(out_dir, exist_ok=True)
    tau_full_day = next((tau for tau, w in windows if w == 'full_day'), 1.0)
    qualifying = _routes_serving_sa(rail_segs_tt, rail_stations, sa_ids)

    _report_sa_top20(prim, skims, name_lookup, svc_network, method,
                     sa_ids, tau_full_day)
    if make_plots:
        _plot_sa_destination_matrices(prim, skims, name_lookup, svc_network, method,
                                      sa_ids)

    # Per-period trip-bearing workbooks: service loads, service boardings, flows.
    loads_sheets, board_sheets, flow_sheets = {}, {}, {}
    for tau, window in windows:
        loads_df, board_df = _service_loads_dfs(prim, qualifying, name_lookup, tau)
        loads_sheets[window] = loads_df
        board_sheets[window] = board_df
        flow_sheets[window] = _station_flows_df(prim, name_lookup, tau)
    _write_period_workbook(loads_sheets, os.path.join(out_dir, 'service_loads.xlsx'),
                           'service loads')
    _write_period_workbook(board_sheets,
                           os.path.join(out_dir, 'service_boardings.xlsx'),
                           'service boardings')
    _write_period_workbook(flow_sheets, os.path.join(out_dir, 'station_flows.xlsx'),
                           'station flows', note=_SBB_STATION_FLOW_NOTE)

    if make_plots:
        _plot_corridor_service_sankeys(prim, name_lookup, svc_network, method,
                                       tau_full_day)
        # Phase 6C draws the developed-vs-baseline delta plots instead (study-area
        # scope only); the absolute per-service plots are the Phase-4C baseline.
        if plot_service_loads:
            _plot_service_load_sequences(prim, variant_seq, variant_period_freq,
                                         qualifying, name_lookup, svc_network,
                                         method, windows)


# ===============================================================================
# DATA LOADING
# ===============================================================================

def _load_rail_segments_with_tt() -> pd.DataFrame:
    """Load full-day rail segments (top-level file), keeping per-segment travel time.

    Reads the complete service set from <_RAIL_BASE>/rail_segments.gpkg (full_day),
    which includes peak-only services (e.g. route O on the Effretikon–Wetzikon
    corridor serving Kempten) that the All_Day subfolder omits. The TT column is
    renamed to travel_time_min.

    Returns:
        DataFrame[from_stop_id, to_stop_id, route_id, direction_id,
                  variant_rank, travel_time_min, ivwt_min], deduped on the
        5-column key. ivwt_min (from IVWT, in-vehicle dwell at intermediate
        holds) is 0.0 when the column is absent (older gpkgs).
    """
    path = os.path.join(catchment_allocate._RAIL_BASE, 'rail_segments.gpkg')
    if not os.path.exists(path):
        return pd.DataFrame()
    frames = []
    for layer_name, _ in pyogrio.list_layers(path):
        gdf = gpd.read_file(path, layer=layer_name)
        gdf = gdf.rename(columns={
            'from_stop_nr': 'from_stop_id',
            'to_stop_nr':   'to_stop_id',
            'GTFS_ID':      'route_id',
            'TT':           'travel_time_min',
            'IVWT':         'ivwt_min',
        })
        if 'ivwt_min' not in gdf.columns:
            gdf['ivwt_min'] = 0.0
        keep = ['from_stop_id', 'to_stop_id', 'route_id',
                'direction_id', 'variant_rank', 'travel_time_min', 'ivwt_min']
        keep = [c for c in keep if c in gdf.columns]
        frames.append(gdf[keep].copy())

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined['travel_time_min'] = pd.to_numeric(
        combined['travel_time_min'], errors='coerce').fillna(0.0)
    combined['ivwt_min'] = pd.to_numeric(
        combined['ivwt_min'], errors='coerce').fillna(0.0)
    return combined.drop_duplicates(
        subset=['from_stop_id', 'to_stop_id',
                'route_id', 'direction_id', 'variant_rank'])


def _through_stop_names() -> dict:
    """{str(stop_nr) -> stop_name} from the full-day rail_segments.gpkg.

    Covers every stop in the service network (including out-of-catchment through-
    stops absent from the infra nodes.gpkg), so the load-sequence plots can resolve
    the 'x<stop_id>' skeys to station names.
    """
    path = os.path.join(catchment_allocate._RAIL_BASE, 'rail_segments.gpkg')
    if not os.path.exists(path):
        return {}
    out = {}
    for layer_name, _ in pyogrio.list_layers(path):
        g = gpd.read_file(path, layer=layer_name)
        for col_id, col_nm in (('from_stop_nr', 'from_stop_name'),
                               ('to_stop_nr', 'to_stop_name')):
            if col_id not in g.columns or col_nm not in g.columns:
                continue
            for sid, nm in zip(g[col_id], g[col_nm]):
                if pd.notna(sid) and pd.notna(nm) and str(nm).strip():
                    out.setdefault(str(int(sid)), str(nm).strip())
    return out


def _load_rail_line_freqs_full_day() -> pd.DataFrame:
    """Full-day per-line frequency table from the top-level rail_lines.gpkg.

    Unlike catchment_allocate._load_rail_line_freqs (which unions the temporal
    subfolders and drops peak-only variants lacking an all-day total_dep), this
    reads the complete full-day line set so peak-only services (e.g. route O
    serving Kempten) carry a frequency and therefore appear in the graph.

    Returns:
        DataFrame with one row per (route_id, direction_id, variant_rank) and a
        freq_per_h_window = total_dep / (GK_WINDOW_MIN / 60) column.
    """
    path = os.path.join(catchment_allocate._RAIL_BASE, 'rail_lines.gpkg')
    if not os.path.exists(path):
        return pd.DataFrame()
    frames = [gpd.read_file(path, layer=layer_name)
              for layer_name, _ in pyogrio.list_layers(path)]
    df = pd.concat(frames, ignore_index=True)
    df['total_dep'] = pd.to_numeric(df.get('total_dep'), errors='coerce')
    df = df.dropna(subset=['total_dep'])
    df['freq_per_h_window'] = df['total_dep'] / (catchment_allocate.GK_WINDOW_MIN / 60.0)
    return df


def _load_gateway_connections(svc_version: str) -> pd.DataFrame:
    """Load the gateway service-connection table written by
    catchment_OD_preparation (Gateway/gateway_service_connections.xlsx). Returns an
    empty DataFrame when absent (gateway injection then disabled)."""
    path = paths.get_gateway_connections_xlsx(svc_version)
    if not os.path.exists(path):
        print(f"  No gateway connection table at {path} — gateway injection "
              f"disabled (gateway demand routes via plain portals only).")
        return pd.DataFrame()
    df = pd.read_excel(path, sheet_name='Connections')
    print(f"  Loaded gateway connection table: {len(df)} variant-rows, "
          f"{df['gateway_station_id'].nunique() if not df.empty else 0} gateways.")
    return df


def _load_routing_od(svc_network: str, od_method: str) -> pd.DataFrame:
    """Whole-day routing OD from the Hook-3 long CSV (TAU_FULL_DAY_SHARE = 1.0:
    the whole-day long table IS the full-day OD). Id-keyed, so no name->id
    round-trip; the per-window xlsx remains a human export.

    Args:
        svc_network: Service network folder name (with '_network').
        od_method:   'pt_feeder' (attribution from settings.OD_ATTRIBUTION_MODE)
                     | 'municipal'.

    Returns:
        DataFrame[origin_id (int), dest_id (int), trips (float)].
        Intrazonal and zero-trip rows excluded.
    """
    attribution = (settings.OD_ATTRIBUTION_MODE.strip().lower()
                   if od_method == 'pt_feeder' else 'municipal')
    path = paths.get_station_od_long_csv(svc_network, od_method, attribution)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Long-format OD CSV not found: {path}. "
            f"Run catchment_OD_preparation.prepare_all_od_matrices() first.")
    od = pd.read_csv(path).rename(columns={'origin_station_id': 'origin_id',
                                           'dest_station_id': 'dest_id'})
    od['origin_id'] = od['origin_id'].astype(int)
    od['dest_id']   = od['dest_id'].astype(int)
    od = od[od['origin_id'] != od['dest_id']]
    od = od[od['trips'] > 0][['origin_id', 'dest_id', 'trips']].copy()
    print(f"    routing OD loaded ({od_method}/{attribution}): {len(od):,} "
          f"non-zero inter-station pairs")
    return od


# ===============================================================================
# PUBLIC ENTRY POINT
# ===============================================================================

def _resolve_methods(assignment_method: str) -> list:
    m = (assignment_method or settings.ROUTING_ASSIGNMENT_METHOD).strip().lower()
    if m == 'both':
        return ['shortest_path', 'logit']
    if m in ('shortest_path', 'logit'):
        return [m]
    raise ValueError(f"Unknown assignment_method '{m}' "
                     f"(expected shortest_path | logit | both).")


def _build_routing_graph(svc_version: str, od_method: str = 'pt_feeder',
                         infra_version: str = '', rail_base: str = None) -> dict:
    """Build the full-day rail routing graph + gateway wiring for a network.

    Encapsulates the shared-data / graph-build / gateway-wiring sequence so both
    passenger_routing (full run) and route_subset (Phase 6 subset) construct an
    identical graph. `rail_base` overrides catchment_allocate._RAIL_BASE so a
    developed (svc-int) network can be routed; default reuses the version's
    Unprojected services dir, giving byte-identical behaviour for the full run.

    Returns:
        dict with G, name_lookup, rail_stations (in-catchment only), rail_segs_tt,
        variant_seq, report_ctx, gw_lookup, gw_ids, infra_version.
    """
    catchment_base.setup_versioned_dirs(svc_version)
    catchment_allocate._RAIL_BASE = rail_base or os.path.join(
        paths.RAIL_LINES_DIR, svc_version, paths.SERVICES_UNPROJECTED_SUBDIR)
    os.chdir(paths.MAIN)

    # --- Shared data ---
    boundary      = catchment_base._load_catchment_boundary()
    rail_stations = catchment_allocate._load_rail_stations(boundary, 'full_day', buffer=0)
    import catchment_OD_preparation as cod
    name_lookup = cod._build_station_name_lookup(rail_stations)
    cod._extend_name_lookup_from_breakdown(name_lookup, od_method)

    # Gateway (boundary) stations sit outside the catchment, so the in-catchment
    # rail_stations load above omits them. Without this the graph has no
    # integer-id portal for the gateway-keyed OD rows and the outputs lack
    # readable gateway names. Recover both: add gateway names to name_lookup
    # (plain, mirroring catchment_OD_preparation) and seed gateway rows into the
    # graph's station set.
    infra_version    = cod._resolve_infra_version(svc_version, infra_version)
    conv_map         = cod._load_convergence_map(
        paths.get_gateway_dir(svc_version),
        cod._served_station_index(svc_version, infra_version))
    conv_ids         = sorted(set(conv_map.values()))
    gateway_rows     = cod._build_gateway_station_rows(svc_version, infra_version,
                                                       extra_ids=conv_ids)
    for _sid, _nm in zip(gateway_rows.get('id_point', []),
                         gateway_rows.get('stop_name', [])):
        _k = str(int(_sid))
        if _k not in name_lookup:
            name_lookup[_k] = str(_nm)
    rail_stations_g = (pd.concat([rail_stations, gateway_rows], ignore_index=True)
                       if not gateway_rows.empty else rail_stations)

    print("\n[Step 2] Building rail graph (full-day services) ...")
    rail_segs_tt = _load_rail_segments_with_tt()
    rail_lines   = _load_rail_line_freqs_full_day()
    G, vfreq, direct_freq, variant_seq, report_ctx = _build_rail_graph(
        rail_segs_tt, rail_lines, rail_stations_g)

    # Out-of-catchment through-stops carry the skey 'x<stop_id>' (never an OD
    # endpoint, but they appear in the per-service load sequences). Resolve their
    # names from the infra nodes (same source as gateway names) so the plots show
    # station names instead of 'x<number>'.
    _ooc_skeys = sorted({sk for seq in variant_seq.values() for sk in seq
                         if isinstance(sk, str) and sk.startswith('x') and sk[1:].isdigit()})
    if _ooc_skeys:
        # Primary source: segment stop names (covers every network stop, including
        # those beyond the infra nodes.gpkg). Fallback: infra nodes (rare gaps).
        _seg_names = _through_stop_names()
        _ids = [int(sk[1:]) for sk in _ooc_skeys]
        _node_names = cod._build_boundary_station_index(_ids, infra_version)
        for sk in _ooc_skeys:
            sid = sk[1:]
            nm = _seg_names.get(sid)
            if not nm:
                _n = _node_names.get(int(sid))
                nm = str(_n[0]) if _n else None
            if nm:
                name_lookup.setdefault(sk, nm)

    print("\n[Step 2b] Gateway service-connection wiring ...")
    conn_df = _load_gateway_connections(svc_version)
    gw_lookup, gw_ids = _build_gateway_conn_lookup(conn_df, variant_seq)
    routing_seq = _attach_gateways(G, gw_lookup, vfreq, variant_seq)
    global _GATEWAY_IDS
    _GATEWAY_IDS = set(int(g) for g in gw_ids)

    return {'G': G, 'name_lookup': name_lookup, 'rail_stations': rail_stations,
            'rail_segs_tt': rail_segs_tt, 'variant_seq': variant_seq,
            'routing_seq': routing_seq, 'report_ctx': report_ctx,
            'gw_lookup': gw_lookup, 'gw_ids': gw_ids,
            'vfreq': vfreq, 'infra_version': infra_version}


def passenger_routing(svc_version: str = '',
                      use_cache: bool = False,
                      od_method: str = 'pt_feeder',
                      assignment_method: str = '',
                      make_plots: bool = True,
                      infra_version: str = '') -> None:
    """Assign W3 station-pair OD onto the rail network (shortest-path and/or Logit).

    Args:
        svc_version:       Service network folder name (with '_network').
        use_cache:         Skip writing CSVs that already exist.
        od_method:         W3 OD source — 'pt_feeder' | 'municipal'.
        assignment_method: '' -> settings.ROUTING_ASSIGNMENT_METHOD;
                           'shortest_path' | 'logit' | 'both'.
        make_plots:        When True (default; standalone), render the Phase-4C
                           assignment plots (skim heatmaps, corridor Sankeys,
                           per-service load sequences). main_new passes
                           settings.PLOT_ASSIGNMENT so the plots honour the
                           Phase-4C visualisation toggle. Excel/CSV outputs are
                           always written regardless.
        infra_version:     Infrastructure version holding boundary_stations.json /
                           nodes.gpkg for gateway recovery. main_new passes the
                           propagated (enhanced) version so no prompt fires; empty
                           (standalone) resolves/prompts via _resolve_infra_version.
    """
    if not svc_version:
        raise ValueError("passenger_routing requires svc_version to locate the "
                         "versioned W3 OD matrix under data/Traffic_Flow/OD/.")

    methods = _resolve_methods(assignment_method)
    print("=" * 70)
    print("PASSENGER ROUTING (Phase 4C)")
    print(f"  Service version : {svc_version}")
    print(f"  W3 OD method    : {od_method}")
    print(f"  Methods         : {', '.join(methods)}")
    print(f"  Cost model      : {settings.TRAVEL_COST_METHOD} / "
          f"{settings.TRANSFER_COST_MODEL}  |  temporal=full_day")
    print("=" * 70)

    ctx = _build_routing_graph(svc_version, od_method, infra_version)
    G            = ctx['G']
    name_lookup  = ctx['name_lookup']
    rail_stations = ctx['rail_stations']
    rail_segs_tt = ctx['rail_segs_tt']
    variant_seq  = ctx['variant_seq']
    report_ctx   = ctx['report_ctx']
    gw_lookup    = ctx['gw_lookup']
    gw_ids       = ctx['gw_ids']
    infra_version = ctx['infra_version']

    global _ROUTE_NAME
    _ROUTE_NAME = report_ctx['route_name']

    print(f"\n[Step 1] Loading W3 full-day OD ({od_method}) ...")
    od_long = _load_routing_od(svc_version, od_method)

    od_normal, od_gw = _expand_gateway_od(od_long, gw_lookup, gw_ids)
    print(f"  Gateways: {len(gw_ids)} boundary gateways as terminal stations "
          f"({len(gw_lookup)} (gateway,role) entries); {len(od_normal):,} OD pairs "
          f"routed through the normal choice model.")

    windows = [(cp.TAU_PEAK_SHARE, 'peak'),
               (cp.TAU_OFFPEAK_SHARE, 'off_peak'),
               (cp.TAU_FULL_DAY_SHARE, 'full_day')]
    svc_network = svc_version
    sa_ids = _sa_station_ids()

    for method in methods:
        print(f"\n[Step 3] Assigning — {method} ...")
        if method == 'shortest_path':
            prim = _assign_shortest_path(G, od_normal)
        else:
            prim = _assign_logit(
                G, od_normal,
                engine=getattr(settings, 'ROUTING_LOGIT_ENGINE', 'table'),
                variant_seq=ctx['routing_seq'], vfreq=ctx['vfreq'],
                k=settings.ROUTING_K_PATHS,
                window_min=settings.ROUTING_COST_WINDOW_MIN,
                window_pct=settings.ROUTING_COST_WINDOW_PCT,
                max_transfers=settings.ROUTING_MAX_TRANSFERS,
                max_examine=settings.ROUTING_MAX_EXAMINE,
                theta=cp.LOGIT_ROUTE_THETA)
        skims = _build_skims(prim, report_ctx['vfreq_report'],
                             report_ctx['direct_freq_report'])

        routed = sum(p['trips'] for p in prim['paths'])
        unres  = sum(u['trips'] for u in prim['unresolved'])
        print(f"    Conservation [{method}]: routed {routed:,.1f} + unresolved "
              f"{unres:,.1f} = {routed + unres:,.1f}  (W3 total "
              f"{od_long['trips'].sum():,.1f})")

        print(f"\n[Step 4] Writing outputs — {method} ...")
        _write_method_outputs(prim, skims, name_lookup, svc_network, method,
                              windows, use_cache)
        _persist_primitive(prim, svc_network, method, use_cache)

        _write_reports(prim, skims, rail_segs_tt, rail_stations, name_lookup,
                       svc_network, method, windows, sa_ids, variant_seq,
                       report_ctx['variant_period_freq'], make_plots=make_plots)

        cache_manifest.write_manifest(
            paths.get_assignment_method_dir(svc_network, method),
            'assignment_4c',
            {'svc_network': svc_network, 'infra_version': infra_version})

    print("\n=== Phase 4C passenger routing done ===")


def _persist_primitive(prim: dict, svc_network: str, method: str,
                       use_cache: bool) -> None:
    """Persist the pre-τ per-pair primitive (paths/segments/events/unresolved) as
    parquet — the reloadable baseline a Phase 6 subset recompute overwrites per
    (origin_id, dest_id) and re-aggregates. Additive: the τ-scaled workbooks above
    are unchanged."""
    for table in ('paths', 'segments', 'events', 'unresolved'):
        fpath = paths.get_routing_primitive_path(svc_network, method, table)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        if use_cache and Path(fpath).exists():
            print(f"    cached: {fpath}")
            continue
        df = pd.DataFrame(prim[table])
        # from_id/to_id/station_id mix int and 'x<stop_id>' (out-of-catchment)
        # ids; cast object columns to str so each parquet column has one type.
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str)
        df.to_parquet(fpath, index=False)
    print(f"    primitive parquet -> {paths.get_assignment_method_dir(svc_network, method)}")


def route_subset(svc_version: str, od_subset: pd.DataFrame, method: str = '',
                 rail_base: str = None, infra_version: str = '',
                 od_method: str = 'pt_feeder') -> dict:
    """Route a subset of OD pairs on a given network; return the pre-τ primitive.

    The Phase 6 subset-routing entry: builds the routing graph (optionally on a
    developed `rail_base`), expands gateways on `od_subset`, and runs the same
    assignment the full path uses. Writes no files — Phase 6 owns the merge into
    the persisted baseline primitive and the re-aggregation.

    Args:
        svc_version: service network folder name (with '_network').
        od_subset:   DataFrame[origin_id, dest_id, trips] — the pairs to route.
        method:      '' -> settings.ROUTING_ASSIGNMENT_METHOD; 'both' -> 'logit';
                     else 'shortest_path' | 'logit'.
        rail_base:   override for catchment_allocate._RAIL_BASE (developed network);
                     None reuses the version's Unprojected services dir.

    Returns:
        prim dict {paths, segments, events, unresolved} at pre-τ trips.
    """
    m = (method or settings.ROUTING_ASSIGNMENT_METHOD).strip().lower()
    if m == 'both':
        m = 'logit'
    if m not in ('shortest_path', 'logit'):
        raise ValueError(f"Unknown method '{m}' (expected shortest_path | logit).")

    ctx = _build_routing_graph(svc_version, od_method, infra_version, rail_base=rail_base)
    G = ctx['G']
    global _ROUTE_NAME
    _ROUTE_NAME = ctx['report_ctx']['route_name']

    od_normal, od_gw = _expand_gateway_od(od_subset, ctx['gw_lookup'], ctx['gw_ids'])
    if m == 'shortest_path':
        prim = _assign_shortest_path(G, od_normal)
    else:
        prim = _assign_logit(
            G, od_normal,
            engine=getattr(settings, 'ROUTING_LOGIT_ENGINE', 'table'),
            variant_seq=ctx['routing_seq'], vfreq=ctx['vfreq'],
            k=settings.ROUTING_K_PATHS,
            window_min=settings.ROUTING_COST_WINDOW_MIN,
            window_pct=settings.ROUTING_COST_WINDOW_PCT,
            max_transfers=settings.ROUTING_MAX_TRANSFERS,
            max_examine=settings.ROUTING_MAX_EXAMINE,
            theta=cp.LOGIT_ROUTE_THETA)
    return prim


# ===============================================================================
# PHASE 6C — SELECTIVE PER-SVC-INT ROUTING  (closure -> re-route -> merge -> re-aggregate)
# ===============================================================================

# Process-lifetime cache: (svc_network, method) -> baseline primitive dict. Every
# svc-int in a Phase-6 run re-routes against the SAME baseline, so the ~1.18M-row
# segments parquet is read once, not once per intervention. Consumers
# (_closure_pairs, _replace_pairs) only READ it — the cached DataFrames are
# shared and MUST NOT be mutated in place; each caller gets a shallow-copied dict
# so reassigning a table stays local.
_BASELINE_PRIM_CACHE: dict = {}


def clear_baseline_primitive_cache() -> None:
    """Drop the in-process baseline-primitive cache (call if Phase 4C re-runs in
    the same process and the baseline changes underneath Phase 6)."""
    _BASELINE_PRIM_CACHE.clear()
    _BASELINE_SLOAD_CACHE.clear()


# Process-lifetime cache: (base_svc_network, od_method, infra_version) -> the
# baseline graph bits the Phase-6C delta plots diff against (variant_seq +
# whole-day vfreq for service loads; reporting frequencies for the base skims of
# the matrix heatmaps). Built once per Phase-6 run (plotting only).
_BASELINE_SLOAD_CACHE: dict = {}


def _load_baseline_plot_bits(base_svc_network: str, od_method: str,
                             infra_version: str) -> dict:
    """Baseline graph bits for the Phase-6C delta plots: variant_seq + whole-day
    vfreq (service-load base columns) and the reporting frequencies (matrix base
    skims). Builds the baseline routing graph once (cached per process);
    _build_routing_graph mutates the _ROUTE_NAME / _GATEWAY_IDS globals, so they
    are saved and restored around the build."""
    key = (base_svc_network, od_method, infra_version)
    if key in _BASELINE_SLOAD_CACHE:
        return _BASELINE_SLOAD_CACHE[key]
    global _ROUTE_NAME, _GATEWAY_IDS
    saved_rn, saved_gw = _ROUTE_NAME, _GATEWAY_IDS
    try:
        bctx = _build_routing_graph(base_svc_network, od_method, infra_version)
    finally:
        _ROUTE_NAME, _GATEWAY_IDS = saved_rn, saved_gw
    bits = {'variant_seq': bctx['variant_seq'], 'vfreq': bctx['vfreq'],
            'vfreq_report': bctx['report_ctx']['vfreq_report'],
            'direct_freq_report': bctx['report_ctx']['direct_freq_report']}
    _BASELINE_SLOAD_CACHE[key] = bits
    return bits


def _load_baseline_primitive(svc_network: str, method: str,
                             use_cache: bool = True) -> dict:
    """Reload the persisted pre-τ baseline primitive as DataFrames.

    Cached per (svc_network, method) for the process lifetime: a 30-svc-int run
    reads the baseline once instead of 30×. The returned dict is a fresh shallow
    copy (reassigning a table is local) but the DataFrames are shared and must
    not be mutated in place. Pass use_cache=False to force a fresh read.

    Raises FileNotFoundError when any table is missing — Phase 4C
    (passenger_routing) must have produced the baseline for this method first.
    """
    key = (svc_network, method)
    if use_cache and key in _BASELINE_PRIM_CACHE:
        cached = _BASELINE_PRIM_CACHE[key]
        print("    baseline primitive: reusing cached copy ("
              + ", ".join(f"{t}={len(cached[t]):,}" for t in cached) + ")")
        return dict(cached)
    prim = {}
    for table in ('paths', 'segments', 'events', 'unresolved'):
        fpath = paths.get_routing_primitive_path(svc_network, method, table)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Baseline primitive missing: {fpath}. Run Phase 4C "
                f"(passenger_routing) for '{svc_network}'/{method} first.")
        prim[table] = pd.read_parquet(fpath)
    print(f"    baseline primitive loaded: "
          + ", ".join(f"{t}={len(prim[t]):,}" for t in prim))
    if use_cache:
        _BASELINE_PRIM_CACHE[key] = prim
    return dict(prim)


def _closure_pairs(base_prim: dict, od_long: pd.DataFrame, affected_stations,
                   affected_services, od_changed_pairs=None) -> set:
    """Service-anchored Phase-6C closure (architecture decision C).

    A pair re-routes iff (a) its baseline path used any service that stops or
    passes at an affected station — the affected-service set is the hook's
    variant_keys expanded by every variant_key with a board/alight/pass event at
    an affected station, catching substitution among co-located services — or
    (b) either endpoint is an affected station, or (c) its OD value changed in
    6B. The documented residual (a pair newly transferring through a brand-new
    corridor without touching an affected station in baseline) escapes this set
    and is caught only by the use_full_recompute_ints oracle.

    Returns:
        set[(origin_id, dest_id)] of pairs to re-route.
    """
    aff_int = {int(s) for s in (affected_stations or [])}
    aff_str = {str(s) for s in aff_int}
    vks = {str(v) for v in (affected_services or [])}

    ev = base_prim['events']
    if len(ev) and aff_str:
        sid = ev['station_id'].astype(str).str.lstrip('x')
        vks |= set(ev.loc[sid.isin(aff_str), 'variant_key'].astype(str).unique())

    pairs: set = set()
    seg = base_prim['segments']
    if len(seg) and vks:
        m = seg['variant_key'].astype(str).isin(vks)
        pairs |= set(zip(seg.loc[m, 'origin_id'].astype(int),
                         seg.loc[m, 'dest_id'].astype(int)))
    n_service = len(pairs)
    if len(od_long) and aff_int:
        m2 = od_long['origin_id'].isin(aff_int) | od_long['dest_id'].isin(aff_int)
        pairs |= set(zip(od_long.loc[m2, 'origin_id'].astype(int),
                         od_long.loc[m2, 'dest_id'].astype(int)))
    n_endpoint = len(pairs) - n_service
    n_od = 0
    if od_changed_pairs:
        before = len(pairs)
        pairs |= {(int(a), int(b)) for a, b in od_changed_pairs}
        n_od = len(pairs) - before
    print(f"    closure: {len(vks)} affected service(s) -> {len(pairs):,} pairs "
          f"({n_service:,} via services, +{n_endpoint:,} endpoint, "
          f"+{n_od:,} OD-changed)")
    return pairs


def _replace_pairs(base_prim: dict, prim_new: dict, pairs: set) -> dict:
    """Per-pair replacement merge: drop every baseline row of the closure pairs
    from all four tables and append the re-routed rows (routing is whole-pair).
    Object columns of the new rows are cast to str to match the parquet dtypes."""
    merged = {}
    for table in ('paths', 'segments', 'events', 'unresolved'):
        b = base_prim[table]
        if len(b):
            key = pd.Series(list(zip(b['origin_id'].astype(int),
                                     b['dest_id'].astype(int))), index=b.index)
            b = b[~key.isin(pairs).values]
        n = pd.DataFrame(prim_new[table])
        if len(n):
            for c in n.columns:
                if n[c].dtype == object:
                    n[c] = n[c].astype(str)
            merged[table] = pd.concat([b, n], ignore_index=True)
        else:
            merged[table] = b.reset_index(drop=True)
    return merged


def route_svc_int(svc_int_id: str, base_svc_network: str, affected_stations,
                  affected_services, rail_base: str, infra_version: str = '',
                  od_changed_pairs=None, od_long_dev: pd.DataFrame = None,
                  method: str = '', od_method: str = 'pt_feeder',
                  make_plots: bool = False,
                  full_recompute: bool = False,
                  write_workbooks: bool = None) -> dict:
    """Phase-6C selective routing for one service intervention.

    Computes the closure on the baseline primitive, re-routes exactly those OD
    pairs on the developed (merged) network, replaces them in the baseline
    primitive and re-derives every Phase-4C output — workbooks, skims, reports,
    primitive parquet — under Assignment/<svc_int_id>_network/<method>/ (the
    Phase-4 schema, keyed by svc-int). Unaffected pairs keep their baseline rows
    byte-identically; the additive globals (service loads/boardings, station
    flows) are re-aggregated from the merged primitive.

    Args:
        svc_int_id:        svc-int id (e.g. 'ext_100001').
        base_svc_network:  baseline service network WITH the '_network' suffix.
        affected_stations: iterable[int] affected id_points (Hook-1 CSV).
        affected_services: iterable[str] affected variant_keys (Hook-1 CSV).
        rail_base:         merged developed-network dir
                           (svc_ints_orchestrator.build_merged_unprojected).
        infra_version:     BASE infra version — gateway recovery (boundary
                           stations, connections) reads the baseline network.
        od_changed_pairs:  optional iterable[(origin_id, dest_id)] from 6B.
        od_long_dev:       optional per-svc-int OD (6B merge); None routes the
                           baseline OD (Municipal: OD is intervention-invariant).
        method:            '' -> settings.ROUTING_ASSIGNMENT_METHOD; 'both' -> 'logit'.
        od_method:         'pt_feeder' | 'municipal'.
        make_plots:        render the Phase-4C report plots for the svc-int
                           (workbooks/CSVs are always written).
        full_recompute:    oracle mode (architecture decision C): route EVERY
                           OD pair on the developed network — no baseline load,
                           closure or merge — through the same loaders, so a
                           parity check against the selective result isolates
                           the closure, not the loaders.
        write_workbooks:   write the three large trip workbooks (path_assignment/
                           segment_loads/station_events.xlsx) for this svc-int;
                           None -> settings.WRITE_INT_WORKBOOKS. The parquet
                           primitives + skims are the machine contract and always
                           write.

    Returns:
        dict(n_pairs_total, n_pairs_closure, routed_trips, unresolved_trips).
    """
    m = (method or settings.ROUTING_ASSIGNMENT_METHOD).strip().lower()
    if m == 'both':
        m = 'logit'
    if m not in ('shortest_path', 'logit'):
        raise ValueError(f"Unknown method '{m}' (expected shortest_path | logit).")
    if write_workbooks is None:
        write_workbooks = getattr(settings, 'WRITE_INT_WORKBOOKS', False)

    if infra_version:
        combo = f"{infra_version}__{base_svc_network.removesuffix('_network')}"
    else:
        import ints_core as _core
        combo = _core.default_combo(
            svc_version=base_svc_network.removesuffix('_network'))
    svc_int_network = paths.svc_int_network_name(svc_int_id, combo)
    print("=" * 70)
    print(f"PHASE 6C {'FULL-RECOMPUTE ORACLE' if full_recompute else 'SELECTIVE ROUTING'} "
          f"— {svc_int_id}")
    print(f"  Baseline network : {base_svc_network}")
    print(f"  Developed network: {rail_base}")
    print(f"  Method           : {m}  |  OD: {od_method}")
    print("=" * 70)

    base_prim = None if full_recompute else _load_baseline_primitive(
        base_svc_network, m)

    prev_rail_base = catchment_allocate._RAIL_BASE
    try:
        ctx = _build_routing_graph(base_svc_network, od_method, infra_version,
                                   rail_base=rail_base)
        global _ROUTE_NAME
        _ROUTE_NAME = ctx['report_ctx']['route_name']

        if od_long_dev is not None:
            od_long = od_long_dev[['origin_id', 'dest_id', 'trips']].copy()
            print(f"    using 6B per-svc-int OD: {len(od_long):,} pairs")
        else:
            od_long = _load_routing_od(base_svc_network, od_method)

        if full_recompute:
            pairs = set(zip(od_long['origin_id'].astype(int),
                            od_long['dest_id'].astype(int)))
        else:
            pairs = _closure_pairs(base_prim, od_long, affected_stations,
                                   affected_services, od_changed_pairs)
        if not pairs:
            print("    empty closure — svc-int touches no routed pair; "
                  "baseline outputs would be identical. Skipping.")
            return {'n_pairs_total': len(od_long), 'n_pairs_closure': 0,
                    'routed_trips': 0.0, 'unresolved_trips': 0.0}

        # Closure keys absent from the OD: with a 6B per-svc-int OD these are
        # legitimate removals (the pair's demand went to zero in the delta —
        # baseline rows are deleted, nothing re-routed). Routing the BASELINE
        # OD they signal a structural od_method mismatch between primitive and
        # OD, which would silently lose trips — refuse.
        od_keys = set(zip(od_long['origin_id'].astype(int),
                          od_long['dest_id'].astype(int)))
        missing = len(pairs - od_keys)
        if full_recompute:
            pass                               # pairs == od_keys by construction
        elif od_long_dev is not None:
            if missing:
                print(f"    {missing:,} closure pair(s) have no OD row in the "
                      f"6B OD (demand removed by the delta) — baseline rows "
                      f"deleted, not re-routed.")
        elif missing > max(10, 0.01 * len(pairs)):
            raise ValueError(
                f"{missing:,} of {len(pairs):,} closure pairs are absent from "
                f"the OD — the baseline primitive under '{base_svc_network}'/"
                f"{m} was built from a different od_method than '{od_method}'. "
                f"Re-run Phase 4C with the matching method first.")

        keys = np.fromiter(
            (k in pairs for k in zip(od_long['origin_id'].astype(int),
                                     od_long['dest_id'].astype(int))),
            dtype=bool, count=len(od_long))
        od_sub = od_long[keys].copy()
        print(f"    re-routing {len(od_sub):,} of {len(od_long):,} OD pairs "
              f"({od_sub['trips'].sum():,.1f} of {od_long['trips'].sum():,.1f} trips)")

        od_normal, od_gw = _expand_gateway_od(od_sub, ctx['gw_lookup'],
                                              ctx['gw_ids'])
        if m == 'shortest_path':
            prim_new = _assign_shortest_path(ctx['G'], od_normal)
        else:
            prim_new = _assign_logit(
                ctx['G'], od_normal,
                engine=getattr(settings, 'ROUTING_LOGIT_ENGINE', 'table'),
                variant_seq=ctx['routing_seq'], vfreq=ctx['vfreq'],
                k=settings.ROUTING_K_PATHS,
                window_min=settings.ROUTING_COST_WINDOW_MIN,
                window_pct=settings.ROUTING_COST_WINDOW_PCT,
                max_transfers=settings.ROUTING_MAX_TRANSFERS,
                max_examine=settings.ROUTING_MAX_EXAMINE,
                theta=cp.LOGIT_ROUTE_THETA)

        if full_recompute:
            cols = {'paths': ['origin_id', 'dest_id', 'path_id', 'n_transfers',
                              'lines_used', 'journey_time_min', 'gc_min',
                              'share', 'trips'],
                    'segments': ['origin_id', 'dest_id', 'path_id', 'from_id',
                                 'to_id', 'variant_key', 'trips'],
                    'events': ['origin_id', 'dest_id', 'path_id', 'station_id',
                               'event', 'variant_key', 'trips'],
                    'unresolved': ['origin_id', 'dest_id', 'trips']}
            merged = {k: (pd.DataFrame(v) if v else pd.DataFrame(columns=cols[k]))
                      for k, v in prim_new.items()}
        else:
            merged = _replace_pairs(base_prim, prim_new, pairs)
        skims = _build_skims(merged, ctx['report_ctx']['vfreq_report'],
                             ctx['report_ctx']['direct_freq_report'])

        windows = [(cp.TAU_PEAK_SHARE, 'peak'),
                   (cp.TAU_OFFPEAK_SHARE, 'off_peak'),
                   (cp.TAU_FULL_DAY_SHARE, 'full_day')]
        print(f"\n    writing per-svc-int outputs -> "
              f"{paths.get_assignment_method_dir(svc_int_network, m)}")
        _write_method_outputs(merged, skims, ctx['name_lookup'],
                              svc_int_network, m, windows, use_cache=False,
                              write_workbooks=write_workbooks)
        _persist_primitive(merged, svc_int_network, m, use_cache=False)
        _write_reports(merged, skims, ctx['rail_segs_tt'], ctx['rail_stations'],
                       ctx['name_lookup'], svc_int_network, m, windows,
                       _sa_station_ids(), ctx['variant_seq'],
                       ctx['report_ctx']['variant_period_freq'],
                       make_plots=make_plots, plot_service_loads=full_recompute)

        # Phase-6C developed-vs-baseline DELTA plots (service loads + SA matrices)
        # replace the absolute per-service plots: study-area scope only (the
        # svc-int's own line plus SA services that actually change). The oracle has
        # no baseline to diff, so it keeps the absolute plots (plot_service_loads).
        if make_plots and not full_recompute:
            base_bits = _load_baseline_plot_bits(base_svc_network, od_method,
                                                 infra_version)
            aff_route_ids = {_route_id_of(v) for v in (affected_services or [])}
            plot_service_load_delta(
                base_prim, merged, base_bits['variant_seq'], ctx['variant_seq'],
                base_bits['vfreq'], ctx['vfreq'], ctx['name_lookup'],
                ctx['rail_segs_tt'], ctx['rail_stations'], _sa_station_ids(),
                aff_route_ids, svc_int_network, svc_int_id, m, windows)
            base_skims = _build_skims(base_prim, base_bits['vfreq_report'],
                                      base_bits['direct_freq_report'])
            plot_service_matrix_delta(
                base_skims, skims, base_prim['paths'], ctx['name_lookup'],
                _sa_station_ids(), svc_int_network, svc_int_id, m)
            # Per-cell accessibility map (PT-Feeder only; needs the dev 6A allocation)
            if od_method == 'pt_feeder':
                plot_cell_accessibility_delta(
                    base_prim, merged, base_svc_network, svc_int_network,
                    svc_int_id, infra_version, combo, m)

        cache_manifest.write_manifest(
            paths.get_assignment_method_dir(svc_int_network, m),
            'assignment_4c',
            {'svc_network': base_svc_network, 'infra_version': infra_version,
             'svc_int_id': svc_int_id})

        routed = float(merged['paths']['trips'].sum()) if len(merged['paths']) else 0.0
        unres = float(merged['unresolved']['trips'].sum()) if len(merged['unresolved']) else 0.0
        print(f"    Conservation [{svc_int_id}/{m}]: routed {routed:,.1f} + "
              f"unresolved {unres:,.1f} = {routed + unres:,.1f}  "
              f"(OD total {od_long['trips'].sum():,.1f})")
        return {'n_pairs_total': len(od_long), 'n_pairs_closure': len(pairs),
                'routed_trips': routed, 'unresolved_trips': unres}
    finally:
        catchment_allocate._RAIL_BASE = prev_rail_base


# ===============================================================================
# STANDALONE ENTRY POINT
# ===============================================================================

if __name__ == '__main__':
    os.chdir(paths.MAIN)

    _feeder_root = os.path.join(paths.MAIN, paths.FEEDER_LINES_DIR)
    _svc_versions = sorted([
        d for d in os.listdir(_feeder_root)
        if os.path.isdir(os.path.join(_feeder_root, d))
        and os.path.exists(os.path.join(
            _feeder_root, d, paths.SERVICES_UNPROJECTED_SUBDIR,
            'pt_feeder_stops.gpkg'))
    ]) if os.path.isdir(_feeder_root) else []

    _svc = ''
    if not _svc_versions:
        print("WARNING: no service versions found.")
    elif len(_svc_versions) == 1:
        _svc = _svc_versions[0]
        print(f"Service version: {_svc}")
    else:
        print("Available service versions:")
        for _i, _sv in enumerate(_svc_versions, 1):
            print(f"  {_i}) {_sv}")
        while True:
            _raw = input("Select service version [1]: ").strip() or '1'
            if _raw.isdigit() and 1 <= int(_raw) <= len(_svc_versions):
                _svc = _svc_versions[int(_raw) - 1]
                break
            print(f"  Invalid — enter 1-{len(_svc_versions)}.")

    print("\nOD method:")
    print("  1) pt_feeder")
    print("  2) municipal")
    print("  3) both")
    _od_map = {'1': ['pt_feeder'], 'pt_feeder': ['pt_feeder'],
               '2': ['municipal'], 'municipal': ['municipal'],
               '3': ['pt_feeder', 'municipal'], 'both': ['pt_feeder', 'municipal']}
    while True:
        _m = input("Select OD method [1]: ").strip().lower() or '1'
        if _m in _od_map:
            _od_methods = _od_map[_m]; break
        print("  Enter 1, 2 or 3.")

    print("\nAssignment method:")
    print("  1) logit")
    print("  2) shortest_path")
    print("  3) both")
    _map = {'1': 'logit', '2': 'shortest_path', '3': 'both'}
    while True:
        _a = input("Select assignment method [1]: ").strip() or '1'
        if _a in _map:
            _assignment = _map[_a]; break
        print("  Enter 1, 2 or 3.")

    print(f"\nCost model (from settings): TRAVEL_COST_METHOD={settings.TRAVEL_COST_METHOD}, "
          f"TRANSFER_COST_MODEL={settings.TRANSFER_COST_MODEL}")
    print(f"Logit params: engine={getattr(settings, 'ROUTING_LOGIT_ENGINE', 'table')}, "
          f"K={settings.ROUTING_K_PATHS}, "
          f"max_transfers={settings.ROUTING_MAX_TRANSFERS}, "
          f"theta={cp.LOGIT_ROUTE_THETA}")

    # Plot generation — standalone asks; default follows settings.PLOT_ASSIGNMENT.
    # (In the main pipeline the toggle decides without prompting.)
    _default_plot = bool(getattr(settings, 'PLOT_ASSIGNMENT', True))
    _default_str  = 'y' if _default_plot else 'n'
    print("\nPlots (assignment heatmaps + corridor Sankeys + service loads):")
    print(f"  Y/N — default '{_default_str}' from settings.PLOT_ASSIGNMENT")
    while True:
        _p = input(f"Generate plots? [{_default_str}]: ").strip().lower() or _default_str
        if _p in ('y', 'n', 'yes', 'no'):
            _make_plots = _p.startswith('y')
            break
        print("  Invalid — enter y or n.")

    print("\nRun mode:")
    print("  1) full Phase-4C routing (baseline)")
    print("  2) Phase-6C svc-int selective re-routing")
    while True:
        _r = input("Select run mode [1]: ").strip() or '1'
        if _r in ('1', '2'):
            break
        print("  Enter 1 or 2.")

    _svc_arg = (_svc + '_network'
                if _svc and not _svc.endswith('_network') else _svc)

    if _r == '1':
        for _od_method in _od_methods:
            passenger_routing(
                svc_version=_svc_arg,
                use_cache=getattr(settings, 'use_cache_railRouting', False),
                od_method=_od_method,
                assignment_method=_assignment,
                make_plots=_make_plots)
    else:
        import svc_ints_orchestrator as _so
        _base_svc = _svc_arg.replace('_network', '')
        _iids = [i for t in _so.SUPPORTED_SVC_INT_TYPES
                 for i in _so.list_svc_int_ids(t)]
        if not _iids:
            raise SystemExit("No svc-ints registered — run Phase 5B first.")
        print("\nRegistered svc-ints: " + ", ".join(_iids))
        while True:
            _iid = input(f"Svc-int id [{_iids[0]}]: ").strip() or _iids[0]
            if _iid in _iids:
                break
            print("  Unknown id.")
        _infra = input(f"Base infra version [{settings.INFRA_VERSION}]: ").strip() \
            or settings.INFRA_VERSION
        _itype = _iid.split('_')[0]
        _rec = _so.read_record(_itype, _iid)
        _aff_path = os.path.join(
            paths.get_svc_int_catalogue_dir(_so._svc_network(None)),
            f'svc_int_affected_set_{_infra}.csv')
        _stations, _services = [], []
        if os.path.exists(_aff_path):
            _adf = pd.read_csv(_aff_path, encoding='utf-8-sig')
            _row = _adf[_adf['int_id'].astype(str) == _iid]
            if not _row.empty:
                _raw_st = _row.iloc[0].get('affected_stations')
                _raw_sv = _row.iloc[0].get('affected_services')
                _stations = [int(float(t)) for t in str(_raw_st or '').split(',')
                             if t.strip() and str(_raw_st) != 'nan']
                _services = [t.strip() for t in str(_raw_sv or '').split(',')
                             if t.strip() and str(_raw_sv) != 'nan']
        else:
            print(f"WARNING: no affected-set CSV at {_aff_path} — closure will "
                  f"use endpoint pairs only.")
        _so.apply_svc_int(_rec, _base_svc, _infra, use_cache=True)
        _merged = _so.build_merged_unprojected(_rec, _base_svc, _infra,
                                               use_cache=True)
        if not _merged:
            raise SystemExit(f"{_iid}: empty delta — nothing to route.")
        for _od_method in _od_methods:
            route_svc_int(_iid, _svc_arg, _stations, _services,
                          rail_base=_merged, infra_version=_infra,
                          method=_assignment, od_method=_od_method,
                          make_plots=_make_plots)
