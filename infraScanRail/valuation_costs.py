"""
valuation_costs — Phase 8B: construction / maintenance / operating cost per svc-int.
Last modified: 2026-06-10

NEW cost track only (the legacy OLD 12-trains/track methodology was dropped,
decision 2026-06-10). Sources:
  - CC construction/maintenance: 5A registry cost fields (read_record_meta) —
    full cost to EACH svc-int that requires the CC (standalone alternatives,
    no splitting). Replaces the costs_connection_curves.xlsx lookup.
  - CAP construction/maintenance: the 5C attribution table
    (svc_int_cap_attribution.csv, candidate - baseline deltas).
  - Uncovered operating cost: signed per-direction delta route-metres of the
    changed/added line variants (projected delta vs base projection, real
    routed geometry — no detour factor; truncations come out negative)
    x operating_cost_s_bahn_per_meter x (1 - general_KDG).

Output keeps the legacy construction_cost.csv column schema
(Dev_/CapInt_/Total*/Yearly*) keyed by svc-int id, so the Phase 9
create_cost_and_benefit_df adaptation stays mechanical. Replaces the legacy
scoring.construction_costs chain (no Rail-Service_Link_construction_cost.csv,
no per-dev gpkg, no dev_id mapping).
"""

import os
import time

import fiona
import geopandas as gpd
import pandas as pd

import cache_manifest
import cost_parameters as cp
import ints_core
import paths
import settings
import svc_ints_orchestrator as so


def compute_construction_costs(svc_int_ids=None, *, combo: str,
                               base_infra: str = None, use_cache: bool = True,
                               make_plots: bool = False) -> pd.DataFrame:
    """Phase 8B: construction + maintenance + operating cost per svc-int.

    Args:
        svc_int_ids: svc-int ids to cost (None = all registered ext + ndc).
        combo: the '<infra>__<svc>' workspace key.
        base_infra: base infra version the deltas were projected on
            (None = the combo's infra half).
        use_cache: skip when construction_cost.csv already covers all ids and
            the costs_8b manifest matches.
        make_plots: cost-composition plot (data outputs always write).

    Returns:
        DataFrame(Development, Dev_ConstructionCost, Dev_MaintenanceCost,
        CapInt_ConstructionCost, CapInt_MaintenanceCost, TotalConstructionCost,
        TotalMaintenanceCost, YearlyMaintenanceCost, uncoveredOperatingCost) —
        also written to data/costs/Developments/<combo>/construction_cost.csv.
    """
    t0 = time.time()
    base_infra = base_infra or combo.split('__')[0]
    svc_version = combo.split('__', 1)[1]
    if svc_int_ids is None:
        svc_int_ids = (so.list_svc_int_ids('ext', network=combo)
                       + so.list_svc_int_ids('ndc', network=combo))
    svc_int_ids = [str(i) for i in svc_int_ids]
    out_dir = paths.get_costs_combo_dir(combo)
    csv_path = paths.get_construction_cost_csv(combo)
    versions = {'combo': combo, 'base_infra': base_infra}

    print(f"\n=== Phase 8B — Construction/Maintenance/Operating Costs "
          f"({len(svc_int_ids)} svc-int(s)) ===")
    print(f"  combo: {combo} | duration: {cp.duration}y | op rate: "
          f"{cp.operating_cost_s_bahn_per_meter} CHF/m/a x "
          f"(1 - KDG {cp.general_KDG})")
    if not svc_int_ids:
        print("  no svc-ints registered — nothing to do")
        return pd.DataFrame(columns=_OUT_COLS)

    if use_cache and os.path.exists(csv_path) and cache_manifest.check_manifest(
            out_dir, 'costs_8b', versions):
        cached = pd.read_csv(csv_path)
        if set(svc_int_ids) <= set(cached['Development'].astype(str)):
            print(f"  [8B] use_cache_costs: {csv_path} covers all "
                  f"{len(svc_int_ids)} svc-int(s) — skipping")
            return cached
        print("  [8B] cache present but missing svc-int(s) — recomputing")

    attribution = _load_attribution(combo)
    base_lengths = _route_lengths(
        paths.get_projected_services_path(svc_version, base_infra))

    rows = []
    for iid in svc_int_ids:
        rec = so.read_record(_svc_int_type(iid), iid, network=combo)
        if rec is None:
            print(f"  [8B] {iid}: not in the registry — skipped")
            continue
        rows.append(_costs_for_svc_int(iid, rec, combo, base_infra,
                                       attribution, base_lengths))
    result = pd.DataFrame(rows, columns=_OUT_COLS)

    os.makedirs(out_dir, exist_ok=True)
    result.to_csv(csv_path, index=False)
    cache_manifest.write_manifest(out_dir, 'costs_8b', versions)
    print(f"  [csv] wrote {csv_path} ({len(result)} svc-int row(s))")

    if make_plots:
        try:
            _plot_costs(result, combo)
        except Exception as exc:
            print(f"  [plot] 8B cost plot failed: {exc}")
    print(f"  Phase 8B done in {time.time() - t0:.1f}s")
    return result


_OUT_COLS = ['Development', 'Dev_ConstructionCost', 'Dev_MaintenanceCost',
             'CapInt_ConstructionCost', 'CapInt_MaintenanceCost',
             'TotalConstructionCost', 'TotalMaintenanceCost',
             'YearlyMaintenanceCost', 'uncoveredOperatingCost']

_VARIANT_KEY = ['GTFS_ID', 'direction_id', 'variant_rank']


def _svc_int_type(svc_int_id: str) -> str:
    return str(svc_int_id).split('_')[0]


def _load_attribution(combo: str) -> pd.DataFrame:
    """5C CAP attribution table; empty frame when 5C produced no CAP."""
    p = paths.get_svc_int_cap_attribution_path(combo)
    if not os.path.exists(p):
        print(f"  [8B] WARNING: no 5C attribution CSV at {p} — "
              f"CapInt_* costs read as 0 (run Phase 5C first).")
        return pd.DataFrame(columns=['svc_int_id', 'attributable_cost_chf',
                                     'attributable_maintenance_annual_chf'])
    df = pd.read_csv(p)
    if 'attributable_maintenance_annual_chf' not in df.columns:
        print("  [8B] WARNING: attribution CSV predates the maintenance "
              "extension — CAP maintenance read as 0 (rerun Phase 5C).")
        df['attributable_maintenance_annual_chf'] = 0.0
    return df


def _route_lengths(gpkg_path: str) -> pd.Series:
    """Projected route-metres per line variant, summed over the hop rows of
    every mode layer. Index (GTFS_ID, direction_id, variant_rank); from
    path_length_m (geometry length where unset)."""
    if not os.path.exists(gpkg_path):
        raise FileNotFoundError(f"projected network missing: {gpkg_path}")
    parts = []
    for layer in fiona.listlayers(gpkg_path):
        gdf = gpd.read_file(gpkg_path, layer=layer)
        if gdf.empty or not set(_VARIANT_KEY) <= set(gdf.columns):
            continue
        length = pd.to_numeric(gdf.get('path_length_m'), errors='coerce')
        length = length.fillna(gdf.geometry.length)
        parts.append(pd.DataFrame({
            'GTFS_ID': gdf['GTFS_ID'].astype(str),
            'direction_id': pd.to_numeric(gdf['direction_id'],
                                          errors='coerce').fillna(0).astype(int),
            'variant_rank': pd.to_numeric(gdf['variant_rank'],
                                          errors='coerce').fillna(0).astype(int),
            'length_m': length.to_numpy(dtype=float)}))
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts).groupby(_VARIANT_KEY)['length_m'].sum()


def _delta_route_m(svc_int_id: str, combo: str, base_infra: str,
                   base_lengths: pd.Series) -> float:
    """Signed mean-over-directions delta route-metres of the svc-int's
    changed/added variants (dev minus base; absent in base = new line).
    set_frequency-only deltas come out 0 (geometry unchanged, frequency-blind
    operating cost — train-km scaling is the deferred refinement)."""
    delta_path = paths.get_svc_int_projected_path(svc_int_id, base_infra, combo)
    dev = _route_lengths(delta_path)
    if dev.empty:
        return 0.0
    per_direction: dict = {}
    for (gtfs_id, direction, rank), dev_len in dev.items():
        base_len = float(base_lengths.get((gtfs_id, direction, rank), 0.0))
        per_direction[direction] = (per_direction.get(direction, 0.0)
                                    + float(dev_len) - base_len)
    return sum(per_direction.values()) / len(per_direction)


def _costs_for_svc_int(svc_int_id: str, rec: dict, combo: str, base_infra: str,
                       attribution: pd.DataFrame,
                       base_lengths: pd.Series) -> dict:
    """One construction_cost.csv row (worker-safe: no settings reads)."""
    # CC (Dev_*): full registry cost of every required CC (8B decision 4).
    dev_constr = dev_maint_annual = 0.0
    for cc_id in (rec.get('requires_infra') or []):
        meta = ints_core.read_record_meta('cc', str(cc_id), network=combo)
        if meta is None:
            print(f"  [8B] {svc_int_id}: required CC '{cc_id}' not in the "
                  f"registry — its cost reads as 0")
            continue
        dev_constr += float(meta.get('construction_cost_chf') or 0.0)
        dev_maint_annual += float(meta.get('maintenance_cost_annual_chf') or 0.0)

    # CAP (CapInt_*): attributable candidate-baseline deltas from 5C.
    mine = attribution[attribution['svc_int_id'].astype(str) == svc_int_id]
    cap_constr = float(mine['attributable_cost_chf'].sum())
    cap_maint_annual = float(mine['attributable_maintenance_annual_chf'].sum())

    # Operating: signed delta route-metres x rate x uncovered share.
    try:
        delta_m = _delta_route_m(svc_int_id, combo, base_infra, base_lengths)
    except FileNotFoundError as exc:
        print(f"  [8B] {svc_int_id}: {exc} — operating cost reads as 0")
        delta_m = 0.0
    op_cost = (delta_m * float(cp.operating_cost_s_bahn_per_meter)
               * (1.0 - float(cp.general_KDG)))

    dev_maint = dev_maint_annual * float(cp.duration)
    cap_maint = cap_maint_annual * float(cp.duration)
    total_constr = dev_constr + cap_constr
    total_maint = dev_maint + cap_maint
    print(f"  [8B] {svc_int_id}: CC {dev_constr / 1e6:,.1f}M + CAP "
          f"{cap_constr / 1e6:,.1f}M constr | maint {total_maint / 1e6:,.1f}M/"
          f"{cp.duration}y | delta {delta_m / 1000.0:+,.2f} km/dir -> op "
          f"{op_cost / 1e6:,.2f}M/a")
    return {'Development': svc_int_id,
            'Dev_ConstructionCost': dev_constr,
            'Dev_MaintenanceCost': dev_maint,
            'CapInt_ConstructionCost': cap_constr,
            'CapInt_MaintenanceCost': cap_maint,
            'TotalConstructionCost': total_constr,
            'TotalMaintenanceCost': total_maint,
            'YearlyMaintenanceCost': total_maint / float(cp.duration),
            'uncoveredOperatingCost': op_cost}


def _plot_costs(result: pd.DataFrame, combo: str) -> None:
    """Stacked construction bars + annual cost bars per svc-int."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(result))
    labels = result['Development'].astype(str)

    ax1.bar(x, result['Dev_ConstructionCost'] / 1e6, label='CC (Dev)')
    ax1.bar(x, result['CapInt_ConstructionCost'] / 1e6,
            bottom=result['Dev_ConstructionCost'] / 1e6, label='CAP (CapInt)')
    ax1.set_title('Construction cost [Mio. CHF]')
    ax1.set_xticks(x, labels, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')

    ax2.bar(x - 0.2, result['YearlyMaintenanceCost'] / 1e6, width=0.4,
            label='Maintenance /a')
    ax2.bar(x + 0.2, result['uncoveredOperatingCost'] / 1e6, width=0.4,
            label='Uncovered operating /a')
    ax2.set_title('Annual costs [Mio. CHF/a]')
    ax2.set_xticks(x, labels, rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')

    fig.suptitle(f'Phase 8B cost composition ({combo})')
    out_dir = paths.get_developments_plot_dir(combo, 'valuation')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'construction_costs.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  [plot] saved {out}")


if __name__ == "__main__":
    import ints_core as _core

    print("=== Phase 8B — Construction/Maintenance/Operating Costs "
          "(standalone) ===")
    _svc = input(f"Service version [{settings.SVC_VERSION}]: ").strip() \
        or settings.SVC_VERSION
    if _svc == 'Build_New':
        _svc = settings.SVC_BUILD_NEW_NAME
    _default_combo = _core.default_combo(svc_version=_svc)
    _combo = input(f"Combo [{_default_combo}]: ").strip() or _default_combo
    _cache_def = 'y' if settings.use_cache_costs else 'n'
    _cache = (input(f"Use cost cache? [y/n] [{_cache_def}]: ").strip()
              or _cache_def).lower() == 'y'
    _plot_def = 'y' if settings.PLOT_COSTS else 'n'
    _plots = (input(f"Generate plots? [y/n] [{_plot_def}]: ").strip()
              or _plot_def).lower() == 'y'
    compute_construction_costs(combo=_combo, use_cache=_cache,
                               make_plots=_plots)
