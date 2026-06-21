"""
svc_ints_stop_patterns — STP svc-int discovery (Phase 5B, expansion part 2).
Last modified: 2026-06-20

Stopping-pattern-change interventions ('stp', id block DEV_ID_START_STP), two
modes that always generate together. STP edits the CALLING PATTERN (which
traversed stations a service stops at vs passes) while keeping the ROUTED PATH
identical — a station moves between the stop sequence (board/alight/transfer
portal) and Via_Nodes/path_nodes only (passing).

  (a) Corridor homogenisation — make a service that PASSES a study-area station
      another service already stops at call there (no-orphan trivially met). One
      `add_stop` int per passing service, plus one COMBINED multi-route int that
      makes all passing services on the corridor stop (full homogenisation).
  (b) Express conversion — drop intermediate calls a service makes, gated by the
      protection rules (co-stopper exists, not a cross-service terminus, not
      sole-served). Per service: one drop-all `drop_stop` int + one per dropped
      stop. Among services identical in the SA, only the LONGEST line generates
      (the shorter is suppressed — K over L).

Detection is structural on the whole-day projected stop pattern; passed stations
are read from path_nodes (the routed path) vs the stop sequence. Materialisation
(split/merge of the affected hops) lives in svc_ints_orchestrator._build_stp_delta.

STP carries NO rolling-stock operating-cost delta (permanent limitation): adding /
dropping calls changes dwell + run time but not train-km in a way the 8B train-metre
model captures, and the cycle-time rolling-stock effect is out of scope.

Plan: docs/infraClaude/plans/2026-06-13-svc-int-stp-part2.md.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

import paths
import settings
import infrabuild_network_builder as ic
import svc_ints_orchestrator as si


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def discover_stp_candidates(
    base_infra: str,
    base_svc: str,
    sa_polygon=None,
    buffer_polygon=None,
    network: Optional[str] = None,
) -> Dict:
    """Derive STP candidates of both modes on the base networks (no registration).

    Returns dict(mode_a, mode_a_combined, mode_b, patterns) — mode_a is the list
    of individual passing-service candidates, mode_a_combined is the single
    multi-route homogenisation candidate (or None), mode_b the express drops.
    """
    if sa_polygon is None:
        print("  [stp] no SA polygon — STP discovery skipped")
        return {'mode_a': [], 'mode_a_combined': None, 'mode_b': []}

    pat = _service_patterns(base_infra, base_svc, sa_polygon)

    mode_a, mode_a_combined = _discover_homogenisation(pat)
    mode_b = _discover_express(pat)

    print(f"  [stp] mode (a): {len(mode_a)} passing service(s)"
          + (" + 1 combined" if mode_a_combined else "")
          + f"; mode (b): {len(mode_b)} express int(s)")
    return {'mode_a': mode_a, 'mode_a_combined': mode_a_combined,
            'mode_b': mode_b, 'patterns': pat}


def discover_and_register(
    base_infra: str,
    base_svc: str,
    sa_polygon=None,
    buffer_polygon=None,
    registry_path: Optional[str] = None,
    network: Optional[str] = None,
) -> Dict:
    """Discover STP candidates (both modes) and append them to the stp registry.

    Deterministic id order: mode (a) individuals (alphabetical by route_id), then
    the combined int, then mode (b) (services alphabetical; per service the
    drop-all int, then the per-stop ints in stop-sequence order).
    """
    disc = discover_stp_candidates(base_infra, base_svc, sa_polygon,
                                   buffer_polygon, network=network)

    start = si.next_svc_int_id('stp', registry_path, network)
    n0 = int(start.split('_')[1])
    records: List[Dict] = []

    def _next_id():
        return f"stp_{n0 + len(records)}"

    for c in disc['mode_a']:
        records.append(_homogenisation_record(c, _next_id(), base_infra, base_svc))
    if disc['mode_a_combined']:
        records.append(_combined_record(disc['mode_a_combined'], _next_id(),
                                        base_infra, base_svc))
    for c in disc['mode_b']:
        records.append(_express_record(c, _next_id(), base_infra, base_svc))

    si.append_records('stp', records, registry_path=registry_path, network=network)
    stp_ids = [r['int_id'] for r in records]
    print(f"  [stp] registered {len(stp_ids)} STP svc-int(s) "
          f"({len(disc['mode_a'])} homogenisation"
          + (" + 1 combined" if disc['mode_a_combined'] else "")
          + f", {len(disc['mode_b'])} express)")
    return {'stp_ids': stp_ids, 'records': records, **disc}


# ─────────────────────────────────────────────────────────────────────────────
# Service-pattern substrate
# ─────────────────────────────────────────────────────────────────────────────

def _service_patterns(base_infra: str, base_svc: str, sa_polygon) -> Dict:
    """Per-route stop sequence, passed SA stations, and network station roles.

    Built from the base PROJECTED chains (the routed path with path_nodes) joined
    to the infra node classification. Returns a dict with:
      seqs            route_id -> ordered [stop_name]  (representative dir-0 variant)
      passed_sa       route_id -> [passed SA station names, in path order]
      stop_services   station_name -> {route_id stopping there}
      terminus        {station_name that is a terminus of any service}
      name_in_sa      station_name -> bool
      route_len       route_id -> stop count (longest-line tiebreak)
      route_meta      route_id -> {variant_rank, line_short_name, line_type, total_dep, layer}
    """
    nodes, _segs = ic.load_version(base_infra)
    ninfo: Dict[int, Dict] = {}
    for _, r in nodes.iterrows():
        try:
            num = int(float(r['Number']))
        except (TypeError, ValueError):
            continue
        g = r.geometry
        ninfo[num] = {
            'name': str(r.get('Name', '')),
            'is_station': str(r.get('Node_Class', '')) == 'station',
            'in_sa': bool(g is not None and sa_polygon.contains(g)),
        }
    name_in_sa: Dict[str, bool] = {}
    for inf in ninfo.values():
        if inf['is_station'] and inf['name']:
            name_in_sa[inf['name']] = name_in_sa.get(inf['name'], False) or inf['in_sa']

    hops_by_variant, _borrow = si._stp_base_hops(base_svc, base_infra)
    base_lines, base_segs, _ = si._load_base_unprojected(base_svc)

    # representative dir-0 variant per route = the projected chain with most stops
    rep: Dict[str, Tuple] = {}
    for (rid, did, var), chain in hops_by_variant.items():
        if did != '0' or not chain:
            continue
        if rid not in rep or len(chain) > len(rep[rid][1]):
            rep[rid] = ((rid, did, var), chain)

    seqs: Dict[str, List[str]] = {}
    passed_sa: Dict[str, List[str]] = {}
    stop_services: Dict[str, set] = defaultdict(set)
    terminus: set = set()
    route_len: Dict[str, int] = {}

    # termini across ALL projected chains (both directions, all variants)
    for chain in hops_by_variant.values():
        if chain:
            terminus.add(chain[0]['from_name'])
            terminus.add(chain[-1]['to_name'])

    for rid, (_key, chain) in rep.items():
        names = [chain[0]['from_name']] + [h['to_name'] for h in chain]
        seqs[rid] = names
        route_len[rid] = len(names)
        stopset = set()
        for nm in names:
            stop_services[nm].add(rid)
            stopset.add(nm)
        # passed SA stations: station-class path nodes (interior) not stopped at
        passed: List[str] = []
        seen_pass: set = set()
        for h in chain:
            for nid in h['path_nodes'][1:-1]:
                inf = ninfo.get(nid)
                if (inf and inf['is_station'] and inf['in_sa']
                        and inf['name'] not in stopset
                        and inf['name'] not in seen_pass):
                    passed.append(inf['name'])
                    seen_pass.add(inf['name'])
        passed_sa[rid] = passed

    route_meta = _route_meta(base_lines, rep)
    return {'seqs': seqs, 'passed_sa': passed_sa, 'stop_services': stop_services,
            'terminus': terminus, 'name_in_sa': name_in_sa, 'route_len': route_len,
            'route_meta': route_meta}


def _route_meta(base_lines, rep) -> Dict[str, Dict]:
    """route_id -> line metadata from the representative dir-0 base line row."""
    meta: Dict[str, Dict] = {}
    for rid, (key, _chain) in rep.items():
        var = key[2]
        for layer, ldf in base_lines.items():
            m = ((ldf['route_id'].astype(str) == rid) &
                 (ldf['direction_id'].astype(str) == '0') &
                 (ldf['variant_rank'].astype(int) == int(var)))
            if m.any():
                row = ldf[m].iloc[0]
                meta[rid] = {'variant_rank': int(var), 'layer': layer,
                             'line_short_name': row.get('line_short_name'),
                             'line_type': int(row.get('line_type', 109)),
                             'total_dep': int(row.get('total_dep', 0) or 0)}
                break
        meta.setdefault(rid, {'variant_rank': int(var), 'layer': 'sbahn',
                              'line_short_name': rid, 'line_type': 109,
                              'total_dep': 0})
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Mode (a) — corridor homogenisation
# ─────────────────────────────────────────────────────────────────────────────

def _discover_homogenisation(pat) -> Tuple[List[Dict], Optional[Dict]]:
    """One add candidate per passing service + one combined int.

    A passing service's added stops = the SA stations it passes that another
    service stops at (no-orphan trivially met). No filters (decision F2).
    """
    stop_services = pat['stop_services']
    out: List[Dict] = []
    for rid in sorted(pat['passed_sa']):
        passed = pat['passed_sa'][rid]
        adds = [s for s in passed if (stop_services.get(s, set()) - {rid})]
        if not adds:
            continue
        out.append({'route_id': rid, 'stops': adds, **pat['route_meta'][rid]})
        print(f"  [stp]   HOMOGENISE {rid}: stop at +[{', '.join(adds)}]")

    combined = None
    if len(out) >= 2:
        routes = [c['route_id'] for c in out]
        combined = {'routes': routes,
                    'stops_per_route': {c['route_id']: list(c['stops']) for c in out},
                    'all_stations': sorted({s for c in out for s in c['stops']})}
        print(f"  [stp]   COMBINED homogenisation: {', '.join(routes)} all stop at "
              f"[{', '.join(combined['all_stations'])}]")
    return out, combined


# ─────────────────────────────────────────────────────────────────────────────
# Mode (b) — express conversion
# ─────────────────────────────────────────────────────────────────────────────

def _discover_express(pat) -> List[Dict]:
    """Drop-all + per-stop express candidates with the protection + L-suppression rules."""
    seqs = pat['seqs']
    stop_services = pat['stop_services']
    name_in_sa = pat['name_in_sa']
    protect_term = bool(getattr(settings, 'STP_PROTECT_CROSS_SERVICE_TERMINI', True))
    terminus = pat['terminus'] if protect_term else set()

    droppable: Dict[str, List[str]] = {}
    for rid, seq in seqs.items():
        inter = seq[1:-1]
        d = [s for s in inter
             if name_in_sa.get(s)
             and (stop_services.get(s, set()) - {rid})        # co-stopper exists
             and s not in terminus]                            # not a protected terminus
        if d:
            droppable[rid] = d

    # L-suppression: among services with an identical SA stop sequence, keep the
    # longest line only (the others would duplicate it on the corridor).
    sa_signature: Dict[str, List[str]] = {}
    for rid in droppable:
        sa_stops = tuple(s for s in seqs[rid] if name_in_sa.get(s))
        sa_signature.setdefault(sa_stops, []).append(rid)
    suppressed: set = set()
    for sig, rids in sa_signature.items():
        if len(rids) < 2:
            continue
        keep = max(rids, key=lambda r: (pat['route_len'][r], r))
        for r in rids:
            if r != keep:
                suppressed.add(r)
                print(f"  [stp]   {r}: identical SA pattern to {keep} (longer) — "
                      f"express ints suppressed")

    out: List[Dict] = []
    for rid in sorted(droppable):
        if rid in suppressed:
            continue
        drops = droppable[rid]                  # stop-sequence order
        meta = pat['route_meta'][rid]
        if len(drops) >= 2:
            out.append({'route_id': rid, 'drop': list(drops), 'kind': 'all', **meta})
        for s in drops:
            out.append({'route_id': rid, 'drop': [s], 'kind': 'one', **meta})
        print(f"  [stp]   EXPRESS {rid}: droppable [{', '.join(drops)}] -> "
              f"{1 + len(drops) if len(drops) >= 2 else 1} int(s)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Record construction
# ─────────────────────────────────────────────────────────────────────────────

def _homogenisation_record(c, int_id, base_infra, base_svc) -> Dict:
    return {
        'int_id': int_id, 'int_type': 'stp', 'base_authored': base_infra,
        'svc_version': base_svc, 'route_id': c['route_id'], 'direction_id': '0',
        'variant_rank': c['variant_rank'],
        'operations': [{'op': 'add_stop', 'params': {'stops': list(c['stops'])}}],
        'total_dep': c['total_dep'], 'line_type': c['line_type'], 'mode_class': 'rail',
        'line_short_name': si.svc_int_line_name(
            {'int_id': int_id, 'int_type': 'stp'}, base_short=c.get('line_short_name')),
        'requires_infra': [], 'affected_stations': list(c['stops']),
        'affected_services': [c['route_id']], 'twin_of': '',
    }


def _combined_record(c, int_id, base_infra, base_svc) -> Dict:
    """Multi-route homogenisation int: route_id='', per-route add_stop ops."""
    return {
        'int_id': int_id, 'int_type': 'stp', 'base_authored': base_infra,
        'svc_version': base_svc, 'route_id': '', 'direction_id': '0', 'variant_rank': 1,
        'operations': [{'op': 'add_stop',
                        'params': {'route_id': rid, 'stops': list(stops)}}
                       for rid, stops in c['stops_per_route'].items()],
        'total_dep': 0, 'line_type': 109, 'mode_class': 'rail',
        'line_short_name': si.svc_int_line_name({'int_id': int_id, 'int_type': 'stp'}),
        'requires_infra': [], 'affected_stations': list(c['all_stations']),
        'affected_services': list(c['routes']), 'twin_of': '',
    }


def _express_record(c, int_id, base_infra, base_svc) -> Dict:
    return {
        'int_id': int_id, 'int_type': 'stp', 'base_authored': base_infra,
        'svc_version': base_svc, 'route_id': c['route_id'], 'direction_id': '0',
        'variant_rank': c['variant_rank'],
        'operations': [{'op': 'drop_stop', 'params': {'stops': list(c['drop'])}}],
        'total_dep': c['total_dep'], 'line_type': c['line_type'], 'mode_class': 'rail',
        'line_short_name': si.svc_int_line_name(
            {'int_id': int_id, 'int_type': 'stp'}, base_short=c.get('line_short_name')),
        'requires_infra': [], 'affected_stations': list(c['drop']),
        'affected_services': [c['route_id']], 'twin_of': '',
    }


# ─────────────────────────────────────────────────────────────────────────────
# Candidate plot
# ─────────────────────────────────────────────────────────────────────────────

_STP_ADD_COLOR = '#0f7b6c'    # teal: added (homogenisation) stops
_STP_DROP_COLOR = '#b8541a'   # orange: dropped (express) stops


def plot_stp_candidates(base_infra: str, base_svc: str, disc: Dict,
                        sa_polygon=None) -> Optional[str]:
    """Candidate overview: added (filled teal) vs dropped (hollow orange) stops.

    Added stops are drawn on the passing service's corridor; dropped stops marked
    hollow. A note flags the Aathal/Wetzikon effect-overlap with the FRQ-(a) ints.
    """
    if not disc.get('mode_a') and not disc.get('mode_b'):
        print("  [plot] no STP candidates — candidate plot skipped")
        return None
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from pathlib import Path
    import ints_core as core

    combo = f"{base_infra}__{base_svc}"
    ctx = si._svc_plot_context(base_infra, base_svc, sa_polygon)
    node_xy = ctx['node_xy']           # name -> (code, x, y)

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

        added = {s for c in disc.get('mode_a', []) for s in c['stops']}
        dropped = {s for c in disc.get('mode_b', []) for s in c['drop']}
        for nm in sorted(added):
            if nm in node_xy:
                ax.plot(node_xy[nm][1], node_xy[nm][2], 'o', mfc=_STP_ADD_COLOR,
                        mec='black', ms=8, zorder=5)
                ax.annotate(nm, xy=(node_xy[nm][1], node_xy[nm][2]), xytext=(6, 6),
                            textcoords='offset points', fontsize=7,
                            color=_STP_ADD_COLOR, fontweight='bold', zorder=6)
        for nm in sorted(dropped):
            if nm in node_xy:
                ax.plot(node_xy[nm][1], node_xy[nm][2], 'o', mfc='white',
                        mec=_STP_DROP_COLOR, mew=2.0, ms=9, zorder=4)

        handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=_STP_ADD_COLOR,
                   markeredgecolor='black', markersize=9,
                   label='added stop (homogenisation)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor=_STP_DROP_COLOR, markeredgewidth=2, markersize=9,
                   label='dropped stop (express)'),
        ]
        n_a = len(disc.get('mode_a', []))
        n_comb = 1 if disc.get('mode_a_combined') else 0
        n_b = len(disc.get('mode_b', []))
        if ctx['extent'] is not None:
            ax.set_xlim(ctx['extent'][0], ctx['extent'][1])
            ax.set_ylim(ctx['extent'][2], ctx['extent'][3])
        ax.legend(handles=handles, loc='upper right', fontsize=8,
                  title='STP candidates', title_fontsize=9)
        ax.set_title(f"STP candidates — {combo}\n"
                     f"{n_a} homogenisation + {n_comb} combined + {n_b} express int(s)\n"
                     f"(added stops overlap the FRQ-(a) corridor remedies)",
                     fontsize=11, fontweight='bold')
        ic._add_north_arrow(ax, location='upper left', scale=0.5)
        ic._add_scale_bar(ax, location=(0.755, 0.012))
        plt.tight_layout()
        out_dir = core.plot_out_dir(combo, 'stp')
        out_path = Path(out_dir) / f"stp_candidates_{base_infra}.pdf"
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"  [plot] wrote {out_path.name}")
        return str(out_path)
    except Exception as exc:
        print(f"  [plot]   WARNING stp_candidates: {exc}")
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

    core.cli_header("infraScanRail — Stopping-Pattern Changes (Phase 5B · STP discovery)")

    core.cli_step(1, "Base infrastructure network?")
    base = core.cli_pick("Infra version:", core.cli_infra_versions(),
                         core._resolve_base_version_propagated())

    core.cli_step(2, "Service network to modify?")
    svc = core.cli_pick("Service version:", core.cli_svc_versions(),
                        core._resolve_svc_version())

    combo = f"{base}__{svc}"
    core.cli_step(3, "Dry run (list only) or register into the catalogue?")
    register = not core.cli_pick_yesno("Dry run?", True)

    sa = core._load_polygon()
    if register:
        if si.list_svc_int_ids('stp', network=combo):
            core.cli_step(4, f"Existing STP registry found for '{combo}' — clear it first?")
            if core.cli_pick_yesno("Clear?", True):
                si.delete_records('stp', si.list_svc_int_ids('stp', network=combo),
                                  network=combo)
                print("  cleared stp registry")
        res = discover_and_register(base, svc, sa_polygon=sa, network=combo)
        print(f"\n=== {len(res['stp_ids'])} STP svc-int(s) registered ===")
    else:
        res = discover_stp_candidates(base, svc, sa_polygon=sa, network=combo)
        n_b = len(res['mode_b'])
        n_a = len(res['mode_a']) + (1 if res['mode_a_combined'] else 0)
        print(f"\n=== dry run: {n_a} homogenisation + {n_b} express STP candidate(s) ===")

    core.cli_step(5, "Generate the candidate plot"
                     + (" + materialise/plot the STP deltas?" if register else "?"))
    if core.cli_pick_yesno("Plots?", getattr(settings, 'PLOT_SVC_INTS', False)):
        plot_stp_candidates(base, svc, res, sa_polygon=sa)
        if register:
            si.materialise_and_plot(base, svc, stp_ids=res['stp_ids'], sa_polygon=sa)
