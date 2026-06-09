"""
mixed_ints_orchestrator — Phase 5C: capacity (CAP) on the matched composed network.
Last modified: 2026-06-08

Capacity is a **mixed** intervention: it needs both the infra (base + CC) and the service
(base + svc-int delta) data to function. This module orchestrates Phase 5C — per svc-int,
design CAP on its own matched network, register/compose/plot it, and write the svc-int→CAP
attribution table for the CBA. The CAP record-builders live in ``mixed_ints_capacity``; the
registry + compose engine + shared plot primitives in ``ints_core``.

CAP outputs land under data/Developments/<infra>__<svc>/cap/ and plots under
plots/Developments/<infra>__<svc>/cap/ (decision H — one combined workspace per combo).
"""

from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd

import paths
import settings
import ints_core as core
import mixed_ints_capacity as cap


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 5C — capacity on the matched (composed base+CC+merged-services) network
# ═════════════════════════════════════════════════════════════════════════════

def phase_5c_capacity_on_matched(
    base_infra: str,
    base_svc: str,
    svc_int_ids: Optional[List[str]] = None,
    make_plots: Optional[bool] = None,
    use_cache: Optional[bool] = None,
) -> dict:
    """Design, register and compose CAP per svc-int on its matched network (one-shot).

    Per-svc-int (fully isolated — one svc-int per network):
      1. compose the svc-int's infra (base + its requires_infra CC);
      2. build its **merged** train supply (base services, minus the EXT parent route,
         plus the svc-int delta) projected on the composed infra;
      3. ``capacity_on_composed`` over the **modified** segments only → resolving CAP;
      4. register the CAP (host = composed net, requires = the CC, id namespaced by the
         svc-int) and compose the svc-int's full network (base + CC + CAP);
      5. record the per-CAP **attributable** cost = candidate − baseline (3C do-nothing).

    Args:
        base_infra: base infra version (e.g. 'AS_2026_ZH').
        base_svc: base service version WITHOUT '_network' (e.g. 'AK_2026_S18').
        svc_int_ids: subset of svc-int ids to process; default = all registered EXT+NDC.
        make_plots: render the per-svc-int capacity/service maps + CAP diff; default
            settings.PLOT_MIXED_INTS (independent of the Phase-3C PLOT_CAPACITY toggle).
        use_cache: keep existing CAP registry + merged-services workbooks; default
            settings.use_cache_svc_int_cap.

    Returns:
        dict(cap_ids, attribution, plots).
    """
    import svc_ints_orchestrator as so
    import capacity_workflow_wrapper as cww
    from capacity_interventions import _target_key

    if make_plots is None:
        make_plots = getattr(settings, 'PLOT_MIXED_INTS', False)
    if use_cache is None:
        use_cache = getattr(settings, 'use_cache_svc_int_cap', False)

    # CAP registry partition key — the infra+svc combination this run targets.
    cap_network = f"{base_infra}__{base_svc}"

    # Resolve the svc-int set as (int_type, int_id) pairs.
    if svc_int_ids is None:
        pairs = ([('ext', i) for i in so.list_svc_int_ids('ext', network=cap_network)] +
                 [('ndc', i) for i in so.list_svc_int_ids('ndc', network=cap_network)])
    else:
        pairs = [(_svc_int_type(i), str(i)) for i in svc_int_ids]

    print(f"\n=== Phase 5C — Capacity on the Matched Network ({len(pairs)} svc-int(s)) ===")
    print(f"  base infra: {base_infra} | services: {base_svc} | cap registry: {cap_network}")
    if not pairs:
        print("  no svc-ints registered — nothing to do")
        return {'cap_ids': [], 'attribution': [], 'plots': []}

    # 5C is the authoritative CAP generator — start from a clean cap registry.
    if not use_cache:
        existing = core.list_intervention_ids('cap', network=cap_network)
        if existing:
            core.delete_records('cap', existing, network=cap_network)
            print(f"  [5C] cleared {len(existing)} existing CAP record(s)")

    # Step 1 — per-svc-int prep (compose infra, merged supply, modified segments).
    prepped: List[dict] = []
    mod_union: set = set()
    for int_type, iid in pairs:
        rec = so.read_record(int_type, iid, network=cap_network)
        if rec is None:
            continue
        req = rec.get('requires_infra') or []
        res = so.apply_svc_int(rec, base_svc, base_infra, use_cache=use_cache)
        composed = res.get('composed_infra', base_infra)
        delta_path = res.get('projected_path')
        if not delta_path or not Path(delta_path).exists():
            print(f"  [5C] {iid}: no materialised delta — skipped")
            continue
        merged_path = _build_merged_services(rec, base_infra, base_svc, delta_path, use_cache)
        mod = _changed_load_segments(base_svc, base_infra, composed, merged_path)
        mod_union |= mod
        prepped.append({'int_type': int_type, 'id': iid, 'rec': rec, 'req': req,
                        'composed': composed, 'merged': merged_path, 'mod': mod})

    if not prepped:
        print("  [5C] no svc-int deltas available — nothing to do")
        return {'cap_ids': [], 'attribution': [], 'plots': []}

    # Step 2 — baseline (3C do-nothing) CAP cost per target, over the union of
    # modified segments, for attribution (candidate − baseline).
    baseline_cost: dict = {}
    base_services = paths.get_projected_services_path(base_svc, base_infra)
    if Path(base_services).exists():
        baseline = cww.capacity_on_composed(
            base_infra, base_infra, base_services,
            network_label=f"{base_svc}_baseline", modified_segment_ids=mod_union)
        for c in baseline:
            baseline_cost[_target_key(c)] = float(c.construction_cost_chf or 0.0)
        print(f"  [5C] baseline reference: {len(baseline)} do-nothing CAP over modified segments")

    # Step 3 — per-svc-int CAP on the merged composed network, register, compose, attribute.
    attribution: List[dict] = []
    all_cap_ids: List[str] = []
    per_svc_plots: List[str] = []
    for p in prepped:
        caps = cww.capacity_on_composed(
            base_infra, p['composed'], p['merged'],
            network_label=p['id'], modified_segment_ids=p['mod'])
        # Default (no CAP): the svc-int's "full" network is just base + its CC.
        cap_ids: set = set()
        composed_full = p['composed']
        if not caps:
            print(f"  [5C] {p['id']}: no over-capacity on modified segments — no CAP")
        else:
            cap_ids = set(cap.register_cap_interventions(
                caps, base_infra, host_version=p['composed'],
                requires=p['req'], id_namespace=p['id'], network=cap_network))
            # compose the svc-int's full network (base + CC + its CAP)
            composed_full = core.compose_infra(base_infra, list(p['req']) + sorted(cap_ids),
                                               svc_version=base_svc)
            svc_rows: List[dict] = []
            for c in caps:
                cid = cap._cap_int_id(c, p['id'])
                if cid not in cap_ids:
                    continue
                all_cap_ids.append(cid)
                cand = float(c.construction_cost_chf or 0.0)
                base_c = baseline_cost.get(_target_key(c), 0.0)
                attrib = {
                    'svc_int_id': p['id'], 'int_type': p['int_type'], 'cap_id': cid,
                    'cap_type': c.type, 'strategy': c.strategy or '',
                    'segment_id': c.segment_id or '',
                    'node_id': c.node_id if c.node_id is not None else '',
                    'candidate_cost_chf': cand, 'baseline_cost_chf': base_c,
                    'attributable_cost_chf': max(0.0, cand - base_c),
                    'composed_version': composed_full,
                }
                attribution.append(attrib)
                svc_rows.append({**attrib,
                                 'current_tracks': c.current_tracks, 'tracks_added': c.tracks_added,
                                 'length_m': c.length_m, 'siding_length_m': c.siding_length_m,
                                 'design_speed_kmh': c.design_speed_kmh,
                                 'maintenance_cost_annual_chf': c.maintenance_cost_annual_chf})
            print(f"  [5C] {p['id']}: {len(cap_ids)} CAP → {composed_full}")

            # Per-svc-int CAP list (data) — alongside the per-svc-int capacity workbooks.
            if svc_rows:
                cap_data_dir = Path(paths.get_svc_int_cap_dir(cap_network)) / p['id']
                cap_data_dir.mkdir(parents=True, exist_ok=True)
                cap_list_path = cap_data_dir / f"cap_list_{p['id']}.xlsx"
                pd.DataFrame(svc_rows).to_excel(cap_list_path, index=False)
                print(f"  [xlsx] wrote {cap_list_path.name} ({len(svc_rows)} CAP)")

        # Per-svc-int capacity + service artifacts for EVERY svc-int (pre-CAP always;
        # post-CAP map + infra-diff only when a CAP was built). Workbooks (data) always
        # write; the PDFs are gated by make_plots.
        try:
            per_svc_plots += _svc_int_capacity_artifacts(
                base_infra, base_svc, cap_network, p, composed_full, sorted(cap_ids),
                make_plots=make_plots)
        except Exception as exc:
            print(f"  [5C] {p['id']} per-svc-int artifacts failed: {exc}")

    # Step 4 — attribution table (the 6C→8B / CBA hand-off input).
    if attribution:
        out = Path(paths.get_svc_int_cap_attribution_path(cap_network))
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(attribution).to_csv(out, index=False)
        print(f"  [csv] wrote {out.name} ({len(attribution)} CAP attribution row(s))")

    # Step 5 — master refresh + CAP plots (per-svc-int maps from the loop + the aggregate diff).
    plots: List[str] = list(per_svc_plots)
    if all_cap_ids:
        try:
            core.build_master_network(base_infra, svc_version=base_svc)
        except Exception as exc:
            print(f"  [5C] master refresh failed: {exc}")
        if make_plots:
            try:
                plots += plot_cap_changes(base_infra, base_svc)
            except Exception as exc:
                print(f"  [plot] CAP plots failed: {exc}")
            # Regenerate the combined all-infra-ints diff from the CC+CAP master so it
            # shows CC (purple) and CAP (orange) together, superseding the 5A CC-only one.
            try:
                import infra_ints_orchestrator as _io
                plots += _io._plot_all_changes(base_infra, svc_version=base_svc)
            except Exception as exc:
                print(f"  [plot] combined all-ints diff failed: {exc}")

    print(f"=== Phase 5C done: {len(set(all_cap_ids))} CAP across "
          f"{len({a['svc_int_id'] for a in attribution})} svc-int(s) ===\n")
    return {'cap_ids': sorted(set(all_cap_ids)), 'attribution': attribution, 'plots': plots}


def _svc_int_type(int_id: str) -> str:
    """Infer the svc-int registry type from its id prefix."""
    return 'ndc' if str(int_id).startswith('ndc') else 'ext'


def _build_merged_services(rec, base_infra, base_svc, delta_path, use_cache) -> str:
    """Write base+delta merged projected services for one svc-int (its own network).

    Base projected rail_segments + the svc-int delta, per layer; for an EXT the base
    rows of the parent route_id are dropped (the delta carries the extended line). NDC
    is additive. Result drives the 5C capacity supply (total load on modified segments).
    """
    import fiona

    out_path = Path(delta_path).with_name('rail_segments_merged.gpkg')
    if use_cache and out_path.exists():
        return str(out_path)

    base_path = paths.get_projected_services_path(base_svc, base_infra)
    remove = {str(rec.get('route_id'))} if rec.get('int_type') == 'ext' else set()

    base_layers = set(fiona.listlayers(base_path)) if Path(base_path).exists() else set()
    delta_layers = set(fiona.listlayers(str(delta_path)))
    if out_path.exists():
        out_path.unlink()

    for layer in sorted(base_layers | delta_layers):
        frames = []
        if layer in base_layers:
            b = gpd.read_file(base_path, layer=layer)
            if remove and 'GTFS_ID' in b.columns:
                b = b[~b['GTFS_ID'].astype(str).isin(remove)]
            frames.append(b)
        if layer in delta_layers:
            frames.append(gpd.read_file(str(delta_path), layer=layer))
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            continue
        merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=core.SWISS_CRS)
        _check_period_freq_consistency(merged, f"{rec.get('svc_int_id') or rec.get('int_id')}/{layer}")
        merged.to_file(out_path, layer=layer, driver='GPKG')
    return str(out_path)


def _check_period_freq_consistency(gdf, label: str) -> None:
    """Warn-only safeguard: peak-only and off-peak-only services never co-operate.

    The peak capacity load sums ``freq_am_peak``/``freq_pm_peak`` (0 for off-peak-only) and
    the off-peak load sums ``freq_offpeak`` (0 for peak-only), so non-coincident services
    can never co-count in a single window. This flags any merged delta row that violates the
    invariant (e.g. a malformed EXT/NDC delta), which would otherwise silently inflate the
    peak utilisation. Best-effort: skips silently if the period/frequency columns are absent.
    """
    need = {'service_period', 'freq_am_peak_dep_hr', 'freq_pm_peak_dep_hr', 'freq_offpeak_dep_hr'}
    if not need.issubset(gdf.columns):
        return
    sp = gdf['service_period'].astype(str).str.lower()
    am = pd.to_numeric(gdf['freq_am_peak_dep_hr'], errors='coerce').fillna(0.0)
    pm = pd.to_numeric(gdf['freq_pm_peak_dep_hr'], errors='coerce').fillna(0.0)
    off = pd.to_numeric(gdf['freq_offpeak_dep_hr'], errors='coerce').fillna(0.0)
    op_bad = int(((sp == 'offpeak_only') & ((am > 0) | (pm > 0))).sum())
    pk_bad = int(((sp == 'peak_only') & (off > 0)).sum())
    if op_bad or pk_bad:
        print(f"  [5C][WARN] {label}: service-period/frequency mismatch — "
              f"{op_bad} offpeak_only row(s) carry peak freq, {pk_bad} peak_only row(s) carry "
              f"off-peak freq. Peak capacity may double-count non-coincident services.")


def _changed_load_segments(base_svc, base_infra, composed_infra, merged_path) -> set:
    """'from-to' BAV segments where the svc-int RAISED peak load vs the base supply.

    Precise scoping (uniform for EXT and NDC): a segment is in scope for CAP only when
    the merged supply (base + delta, EXT parent dropped) carries MORE peak load there
    than the base supply — not merely because the delta traverses it. This sheds an
    EXT's unchanged original portion (same trains, farther) while keeping every segment
    whose load actually increased — including tail segments already served by another
    line (caught by load increase, which a segment-set difference would miss) and the
    CC-re-sectioned segments (new ids absent from base → counted as increased).
    """
    from capacity_calculator import load_projected_services

    def _peak_by_seg(links) -> dict:
        if links is None or links.empty:
            return {}
        g = (links.groupby([links['seg_from_node'].astype(int),
                            links['seg_to_node'].astype(int)])['freq_peak'].sum())
        return {f"{a}-{b}": float(v) for (a, b), v in g.items()}

    base = _peak_by_seg(load_projected_services(base_svc, base_infra))
    merged = _peak_by_seg(load_projected_services(
        f"{base_svc}_merged", composed_infra, gpkg_path=str(merged_path)))
    eps = 1e-6
    return {seg for seg, load in merged.items() if load > base.get(seg, 0.0) + eps}


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_cap_changes(base_version: str, svc_version: Optional[str] = None,
                     extents=('CA', 'SA')) -> List[str]:
    """Base-vs-(base+all CAP) diff (CAP additions in orange) + capacity-relief map.

    The diff reuses the shared ints_core renderer; the relief map (post-CAP section
    utilisation) is best-effort and skips cleanly when the capacity workbooks are absent.
    The CAP registry + plot folder are keyed by the '<infra>__<svc>' combination.
    Returns the written file paths.
    """
    network = f"{base_version}__{svc_version}" if svc_version else core._default_network('cap')
    ids = core.list_intervention_ids('cap', network=network)
    if not ids:
        print("  [plot] no CAP interventions — skipping CAP diff")
        return []
    out_dir = core.plot_out_dir(network, 'cap')
    written = core.render_int_diff(base_version, ids, out_tag='CAP',
                                   added_label='Capacity intervention', added_color=cap._CAP_COLOR,
                                   added_edge=cap._CAP_EDGE, out_dir=out_dir, extents=extents,
                                   svc_version=svc_version)
    written += _plot_capacity_relief(base_version, network)
    return written


def _plot_capacity_relief(base_version: str, network: Optional[str] = None) -> List[str]:
    """Best-effort capacity/utilisation map of the post-CAP network (Phase 3C artifact).

    Renders capacity_network_plots.plot_capacity_network on the latest capacity prep +
    sections workbooks. Skips cleanly (with a note) when those workbooks are absent.
    """
    try:
        import capacity_network_plots as cnp
    except Exception as exc:
        print(f"  [plot] capacity-relief skipped (import: {exc})")
        return []

    out_dir = core.plot_out_dir(network or core._combo(base_version), 'cap')
    out_path = out_dir / f"infra_ints_CAP_relief_SA_{base_version}.pdf"
    boundary = str(Path(paths.MAIN) / paths.STUDY_AREA_BOUNDARY_GPKG)
    try:
        _, cap_path = cnp.plot_capacity_network(
            output_path=str(out_path), generate_network=False,
            infra_version=base_version, boundary_path=boundary)
        print(f"  [plot] wrote {Path(cap_path).name}")
        return [str(cap_path)]
    except (FileNotFoundError, ValueError) as exc:
        print(f"  [plot] capacity-relief skipped — no capacity workbook "
              f"({exc}). Run the Phase 3C capacity workflow first.")
        return []
    except Exception as exc:
        print(f"  [plot] capacity-relief failed: {exc}")
        return []


def _svc_int_capacity_artifacts(base_infra, base_svc, combo, p, composed_full, cap_ids,
                                make_plots: bool = True) -> List[str]:
    """Per-svc-int capacity + service artifacts — produced for EVERY svc-int (data always):

      1. capacity utilisation PRE-CAP  (matched net base+CC + the svc-int's merged services)
      2. service-frequency map         (Phase-3C style; services are CAP-invariant → one map)
      3. capacity utilisation POST-CAP (only when a CAP was built: base+CC+CAP + same services)
      4. infra additions diff          (base → base+CC+CAP: the CC + CAP for this svc-int)

    Maps 1–2 are rendered for all svc-ints so a no-CAP decision can be verified; maps 3–4
    only when ``cap_ids`` is non-empty. The pre/post capacity workbooks (data) are ALWAYS
    written under data/Developments/<combo>/cap/<svc_int_id>/, regardless of make_plots — only
    the figures (plots/…/cap/<svc_int_id>/) are gated. Maps 1–3 are **study-area only**, built
    + rendered exactly the way the Phase-3C SA workflow makes the Network/Capacity plots (SA
    node-set scoping + real composed geometry); map 4 renders both CA and SA extents.
    Best-effort: a failed map is logged and skipped, never aborting 5C.
    """
    import capacity_workflow_wrapper as cww

    svc_id = p['id']
    data_dir = Path(paths.get_svc_int_cap_dir(combo)) / svc_id
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = core.plot_out_dir(combo, 'cap') / svc_id
    if make_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)
    sa_boundary = str(Path(paths.MAIN) / paths.STUDY_AREA_BOUNDARY_GPKG)
    lakes_sa = paths.LAKES_SA_GPKG
    written: List[str] = []

    # Maps 1 & 2 — pre/post-CAP utilisation, STUDY AREA only. The workbook (data) always
    # writes; the figure is built the same way the Phase-3C SA workflow does, gated by plots.
    # preCAP is produced for every svc-int; postCAP only when a CAP changed the network.
    tags = [('preCAP', p['composed'])]
    if cap_ids and composed_full != p['composed']:
        tags.append(('postCAP', composed_full))
    for tag, composed_ver in tags:
        infra_dir = (None if composed_ver == base_infra
                     else str(Path(paths.get_derived_infra_version_dir(composed_ver)).parent))
        wb = cww.write_sa_composed_capacity_workbook(
            base_infra, composed_ver, p['merged'],
            svc_version=base_svc, network_label=f"{svc_id}_{tag}",
            out_path=data_dir / f"capacity_{svc_id}_{tag}.xlsx")
        if wb is None:
            print(f"  [5C] {svc_id} {tag}: no SA sections — skipped")
            continue
        print(f"  [xlsx] wrote capacity_{svc_id}_{tag}.xlsx")
        if not make_plots:
            continue
        try:
            import capacity_network_plots as cnp
            _, cap_png = cnp.plot_capacity_network(
                workbook_path=wb, sections_workbook_path=wb,
                output_path=str(plot_dir / f"capacity_util_{svc_id}_{tag}.pdf"),
                generate_network=False, network_label=f"{svc_id}_{tag}",
                infra_version=composed_ver, infra_dir=infra_dir,
                lakes_path=lakes_sa, boundary_path=sa_boundary)
            written.append(str(cap_png))
            print(f"  [plot] {svc_id}: wrote {Path(cap_png).name}")
        except Exception as exc:
            print(f"  [plot] {svc_id} {tag} util map failed: {exc}")

        # Service-frequency map (SA) — same plot Phase-3C makes. Services are identical
        # pre/post CAP (CAP adds track, not service), so render it once on the preCAP net.
        if tag == 'preCAP':
            try:
                # Mirror the Phase-3C service map exactly: schematic workbook geometry
                # (no infra_version → no BAV curve-following) with the info tables and
                # line short-name labels (include_labels=True is the 3C default).
                svc_png = cnp.plot_service_network(
                    workbook_path=wb,
                    output_path=str(plot_dir / f"service_{svc_id}.pdf"),
                    network_label=f"{svc_id}", lakes_path=lakes_sa)
                written.append(str(svc_png))
                print(f"  [plot] {svc_id}: wrote {Path(svc_png).name}")
            except Exception as exc:
                print(f"  [plot] {svc_id} service map failed: {exc}")

    # Map 4 — infra additions (this svc-int's CC and/or CAP), both CA + SA extents (plot-gated).
    add_ids = list(p['req']) + list(cap_ids)
    if make_plots and add_ids:
        _diff_label = 'CC + CAP' if cap_ids else 'Connecting curve'
        try:
            written += core.render_int_diff(
                base_infra, add_ids, out_tag=f"{svc_id}_DELTA",
                added_label=_diff_label, added_color=cap._CAP_COLOR,
                added_edge=cap._CAP_EDGE, out_dir=plot_dir, extents=('CA', 'SA'),
                svc_version=base_svc)
        except Exception as exc:
            print(f"  [plot] {svc_id} infra-diff failed: {exc}")
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.chdir(paths.MAIN)
    core.cli_header("infraScanRail — Mixed Interventions (Phase 5C · CAP on matched network)")

    core.cli_step(1, "Base infrastructure network?")
    base = core.cli_pick("Infra version:", core.cli_infra_versions(), core._resolve_base_version_propagated())

    core.cli_step(2, "Service network (its registered svc-ints get capacity)?")
    svc = core.cli_pick("Service version:", core.cli_svc_versions(), core._resolve_svc_version())

    core.cli_step(3, "Generate CAP change plots?")
    make_plots = core.cli_pick_yesno("Plots?", getattr(settings, 'PLOT_MIXED_INTS', False))

    res = phase_5c_capacity_on_matched(base, svc, make_plots=make_plots)
    print(f"\n=== Phase 5C: {len(res['cap_ids'])} CAP, "
          f"{len(res['attribution'])} attribution row(s) ===")
