"""
infra_ints_orchestrator — Phase 5A: connecting-curve (CC) infra interventions.
Last modified: 2026-06-08

Infra interventions are **CC only**. This module orchestrates Phase 5A: discover and
register connecting curves (via infra_ints_connecting_curve), build the master tagged
network, and render the change plots. The registry I/O + compose engine + shared plot
primitives live in ``ints_core``; capacity (CAP) moved to ``mixed_ints_orchestrator``
(Phase 5C) on 2026-06-08.

CC outputs land under data/Developments/<infra>__<svc>/cc/ and plots under
plots/Developments/<infra>__<svc>/cc/ (decision H — one combined workspace per combo).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import pandas as pd

import cache_manifest
import paths
import settings
import infrabuild_network_builder as ic
import ints_core as core
# Shared config resolvers live in ints_core; re-exported here so the main_new phase
# wiring (5A/5B/5C) and downstream CLIs can reach them via this module unchanged.
from ints_core import _resolve_base_version, _resolve_svc_version, compose_infra

# Infra interventions are CC only (CAP is a mixed int — see mixed_ints_orchestrator).
SUPPORTED_INFRA_INT_TYPES = ('cc',)
# svc-int types whose activation is driven by INFRA_INT_MODE (legacy resolver helpers).
SVC_INT_TYPES = ('ext', 'cc')

_MASTER_PLOT_SUBTYPE = 'Dev_Full'


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 5A — entry point
# ═════════════════════════════════════════════════════════════════════════════

def phase_5a_infra_interventions(
    base_version: str,
    svc_version: str,
    mode: Optional[str] = None,
    make_plots: Optional[bool] = None,
    use_cache: Optional[bool] = None,
    polygon=None,
    buffer_polygon=None,
    interactive: bool = False,
) -> Dict:
    """Discover/register connecting curves + master tagged network for a base version.

    Args:
        base_version: base infra version (e.g. 'AS_2026_ZH') — NOT a derived version.
        svc_version: service network name (e.g. 'AK_2026_S18').
        mode: INFRA_INT_MODE override ('NONE'|'ALL'|'CC'); default from settings.
        make_plots: render base-vs-developed change plots; default settings.PLOT_INFRA_INTS.
        use_cache: skip CC re-discovery if the cc registry is already populated;
            default from settings.use_cache_infra_ints.
        polygon: study-area boundary (≥1 CC endpoint must be inside).
        buffer_polygon: study-area buffer (CC candidate-station extent).
        interactive: prompt for missing CC composition via CLI.

    Returns:
        dict(cc_ids, ndc_candidates, master) — ndc_candidates feed Phase 5B.
    """
    mode = (mode or getattr(settings, 'INFRA_INT_MODE', 'NONE'))
    if make_plots is None:
        make_plots = getattr(settings, 'PLOT_INFRA_INTS', False)
    if use_cache is None:
        use_cache = getattr(settings, 'use_cache_infra_ints', False)
    if buffer_polygon is None:
        buffer_polygon = core._load_buffer()

    active = _active_infra_types(mode)
    print(f"\n=== Phase 5A — Infrastructure Interventions (mode={mode}) ===")
    print(f"  base infra: {base_version} | services: {svc_version} | active: {active or 'none'}")

    result: Dict = {'cc_ids': [], 'ndc_candidates': [], 'master': None}
    combo = f"{base_version}__{svc_version}"   # registry partition key (decision H)

    # Auto-clear inactive types: the master network composes every registered
    # infra int, so a type kept from a previous run (e.g. CC) would compose into
    # the master even when deselected now. Drop the registry rows of any
    # supported infra type NOT selected this run.
    for _t in SUPPORTED_INFRA_INT_TYPES:
        if _t in active:
            continue
        _stale = core.list_intervention_ids(_t, network=combo)
        if _stale:
            print(f"  [{_t}] clearing {len(_stale)} stale infra-int record(s) — "
                  f"type not selected this run")
            core.delete_records(_t, _stale, network=combo)

    # CC — auto-discover and register (unless cached) -------------------------
    if 'cc' in active:
        cc_dir = os.path.dirname(paths.get_infra_int_registry('cc', combo))
        have_cc = bool(core.list_intervention_ids('cc', network=combo))
        if use_cache and have_cc and cache_manifest.check_manifest(
                cc_dir, 'infra_ints_5a',
                {'infra_version': base_version, 'svc_version': svc_version}):
            print("  [cc] use_cache: keeping existing cc registry; skipping discovery")
            result['cc_ids'] = core.list_intervention_ids('cc', network=combo)
        else:
            import infra_ints_connecting_curve as cc   # lazy — avoids import cycle
            disc = cc.discover_and_register(
                base_version, svc_version,
                sa_polygon=polygon, buffer_polygon=buffer_polygon, interactive=interactive)
            result['cc_ids'] = disc['cc_ids']
            result['ndc_candidates'] = disc['ndc_candidates']
        cache_manifest.write_manifest(
            cc_dir, 'infra_ints_5a',
            {'infra_version': base_version, 'svc_version': svc_version})

    # Master tagged network (CC-only for 5A; 5C rebuilds it with CAP) ---------
    result['master'] = core.build_master_network(base_version)
    print(f"  [master] {result['master']}")

    # Change plots (base vs developed) ---------------------------------------
    if make_plots:
        try:
            result['plots'] = plot_infra_interventions(base_version, polygon)
        except Exception as exc:
            print(f"  [plot] WARNING: change plots failed: {exc}")

    print(f"=== Phase 5A done: {len(result['cc_ids'])} CC ===\n")
    return result


def plot_infra_interventions(base_version: str, sa_polygon=None) -> List[str]:
    """Render the CC + combined change plots for the registered infra ints.

    Delegates the CC diff figures (purple) to infra_ints_connecting_curve and renders the
    combined base-vs-Dev_Full diff + the developed-network maps here. CAP plots are owned
    by Phase 5C (mixed_ints_orchestrator).
    """
    written: List[str] = []
    cc_ids = core.list_intervention_ids('cc', network=core._combo(base_version))

    if cc_ids:
        import infra_ints_connecting_curve as cc
        written += cc.plot_cc_changes(base_version)

    written += _plot_all_changes(base_version)
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Active-type resolution
# ─────────────────────────────────────────────────────────────────────────────

def _active_infra_types(mode) -> List[str]:
    """Map INFRA_INT_MODE to the infra-int registry types to generate/collect (CC only).

    Accepts the legacy string forms ('NONE' | 'ALL' | 'CC' | 'CAP') or an
    explicit list/tuple of type codes (e.g. ['CC']).
    """
    if isinstance(mode, (list, tuple, set)):
        return core.normalise_int_types(mode, SUPPORTED_INFRA_INT_TYPES, 'INFRA_INT_MODE')
    m = str(mode).upper()
    if m == 'NONE':
        return []
    if m in ('ALL', 'CC'):
        return ['cc']
    if m == 'CAP':
        print("  [intervention] INFRA_INT_MODE='CAP' — capacity is a Phase 5C mixed int "
              "(mixed_ints_orchestrator); nothing for 5A to do")
        return []
    print(f"  [intervention] unknown INFRA_INT_MODE='{mode}' — treating as 'NONE'")
    return []


def infra_int_active(mode=None) -> bool:
    """True when INFRA_INT_MODE resolves to at least one infra-int type.

    Replaces the ``str(INFRA_INT_MODE).upper() == 'NONE'`` gate, which misreads a
    list value (and an empty list) as active.
    """
    if mode is None:
        mode = getattr(settings, 'INFRA_INT_MODE', 'NONE')
    return bool(_active_infra_types(mode))


def resolve_active_svc_int_types() -> List[str]:
    """Resolve which svc-int types are active under settings.INFRA_INT_MODE."""
    raw = getattr(settings, 'INFRA_INT_MODE', 'NONE')
    if isinstance(raw, (list, tuple, set)):
        return core.normalise_int_types(raw, SVC_INT_TYPES, 'INFRA_INT_MODE')
    mode = str(raw).upper()
    if mode == 'NONE':
        return []
    if mode == 'ALL':
        return list(SVC_INT_TYPES)
    if mode.lower() in SVC_INT_TYPES:
        return [mode.lower()]
    return []


def enumerate_active_infra_ints() -> List[str]:
    """Collect all infra-int IDs implied by the active types."""
    active = resolve_active_svc_int_types()
    if not active:
        return []
    ids: set = set()
    infra_types = set(SUPPORTED_INFRA_INT_TYPES)
    combo = core._combo()
    for t in active:
        if t in infra_types:
            ids.update(core.list_intervention_ids(t))
        else:
            svc_path = Path(paths.get_svc_int_registry(t, combo))
            if not svc_path.exists():
                continue
            try:
                df = pd.read_excel(svc_path, sheet_name='extensions')
            except Exception as exc:
                print(f"  [registry] cannot read {svc_path.name}: {exc}")
                continue
            if 'requires_infra' in df.columns:
                for raw in df['requires_infra'].dropna().astype(str):
                    for token in raw.split(','):
                        s = token.strip()
                        if s:
                            ids.add(s)
    return sorted(ids)


# ═════════════════════════════════════════════════════════════════════════════
# PLOTS — combined (base vs Dev_Full) + developed-network maps
# ═════════════════════════════════════════════════════════════════════════════
#
# The per-type CC figures (purple) live in infra_ints_connecting_curve.plot_cc_changes and
# call ints_core.render_int_diff; the combined base-vs-Dev_Full diff and the developed-network
# maps live below.

def _plot_all_changes(base_version: str, svc_version: Optional[str] = None) -> List[str]:
    """Combined base-vs-Dev_Full diff + developed-network maps.

    With ``svc_version`` the svc-versioned master '<base>__<svc>_full' (CC + CAP) is
    plotted; without it the CC-only master '<base>_full'. The diff colours added /
    track-gained items BY int_type — CC purple, CAP orange — so one figure shows every
    infra intervention distinctly.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import infra_ints_connecting_curve as _cc
    import mixed_ints_capacity as _cap

    full_name = (f"{base_version}__{svc_version}_full" if svc_version
                 else f"{base_version}_full")
    full_dir = str(Path(paths.MAIN) / paths.DEVELOPMENTS_DEV_FULL_DIR)
    full_seg_path = Path(full_dir) / full_name / 'segments.gpkg'
    if not full_seg_path.exists():
        print(f"  [plot] Dev_Full '{full_name}' not found — skipping combined plots")
        return []

    base_nodes, base_segs = ic.load_version(base_version)
    full_nodes, full_segs = ic.load_version(full_name, infra_dir=full_dir)

    # int_type → colour: CC additions purple, CAP additions orange (one combined plot).
    type_colors = {
        'cc':  {'color': _cc._CC_COLOR,  'edge': _cc._CC_EDGE,  'label': 'Connecting curve'},
        'cap': {'color': _cap._CAP_COLOR, 'edge': _cap._CAP_EDGE, 'label': 'Capacity intervention'},
    }

    out_dir = core.plot_out_dir(core._combo(base_version, svc_version), _MASTER_PLOT_SUBTYPE)
    boundaries = core._plot_boundaries()
    written: List[str] = []

    # combined diff — added/track-gained coloured by int_type (CC vs CAP)
    for ext_key in ('CA', 'SA'):
        bdry, ext, is_ca = boundaries[ext_key]
        net_a = core._net_from_frames(base_nodes, base_segs, base_version, bdry)
        net_b = core._net_from_frames(full_nodes, full_segs, full_name, bdry)
        path = out_dir / f"infra_ints_ALL_{ext_key}_{base_version}.pdf"
        fig = ic.plot_infrastructure_diff(
            net_a=net_a, net_b=net_b, extent=ext, output_path=path,
            is_catchment=is_ca, show_outside=(not is_ca), rail_only=is_ca,
            added_type_colors=type_colors,
            title="All infrastructure changes necessary for each generated service intervention")
        plt.close(fig)
        written.append(str(path))
        print(f"  [plot] wrote {path.name}")

    # developed-network maps: canonical (CA+SA) + engineering structures (SA)
    written += _plot_developed_maps(base_version, full_name, full_dir,
                                    full_nodes, full_segs, out_dir, boundaries)
    return written


def _plot_developed_maps(base_version, full_name, full_dir, full_nodes, full_segs,
                         out_dir, boundaries) -> List[str]:
    """Standard canonical + engineering-structures cartography of the Dev_Full network."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    written: List[str] = []
    for ext_key in ('CA', 'SA'):
        bdry, ext, is_ca = boundaries[ext_key]
        net = core._net_from_frames(full_nodes, full_segs, full_name, bdry)
        path = out_dir / f"infra_ints_DEV_infrastructure_{ext_key}_{base_version}.pdf"
        try:
            kwargs = {'is_catchment': True, 'show_labels': False, 'rail_only': True} if is_ca \
                else {'show_outside': True}
            fig = ic.plot_infrastructure_canonical(net, extent=ext, output_path=path, **kwargs)
            plt.close(fig)
            written.append(str(path))
            print(f"  [plot] wrote {path.name}")
        except Exception as exc:
            print(f"  [plot] WARNING: developed canonical {ext_key} failed: {exc}")

    comp_path = Path(full_dir) / full_name / 'segments_composition.gpkg'
    if comp_path.exists():
        comp = gpd.read_file(comp_path)
        if not comp.empty:
            bdry, ext, _ = boundaries['SA']
            net = core._net_from_frames(full_nodes, full_segs, full_name, bdry)
            path = out_dir / f"infra_ints_DEV_engineering_structures_SA_{base_version}.pdf"
            try:
                fig = ic.plot_engineering_structures(
                    net, comp, extent=ext, output_path=path,
                    show_outside=True, nodes=full_nodes)
                plt.close(fig)
                written.append(str(path))
                print(f"  [plot] wrote {path.name}")
            except Exception as exc:
                print(f"  [plot] WARNING: developed engineering-structures failed: {exc}")
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.chdir(paths.MAIN)
    core.cli_header("infraScanRail — Infrastructure Interventions (Phase 5A · CC)")

    core.cli_step(1, "Base infrastructure network?")
    base = core.cli_pick("Infra version:", core.cli_infra_versions(), core._resolve_base_version_propagated())

    core.cli_step(2, "Service network (defines existing direct-service pairs)?")
    svc = core.cli_pick("Service version:", core.cli_svc_versions(), _resolve_svc_version())

    core.cli_step(3, "Intervention mode?")
    mode = core.cli_pick("INFRA_INT_MODE:", ['NONE', 'CC'],
                         getattr(settings, 'INFRA_INT_MODE', 'NONE'))

    core.cli_step(4, "Generate base-vs-developed change plots?")
    make_plots = core.cli_pick_yesno("Plots?", getattr(settings, 'PLOT_INFRA_INTS', False))

    phase_5a_infra_interventions(
        base_version=base, svc_version=svc, mode=mode,
        make_plots=make_plots, polygon=core._load_polygon(), buffer_polygon=core._load_buffer(),
        interactive=True,   # standalone CLI may prompt for missing CC composition
    )
