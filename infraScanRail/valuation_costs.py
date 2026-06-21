"""
valuation_costs — Phase 8B: construction / maintenance / operating cost per svc-int.
Last modified: 2026-06-21

NEW cost track only (the legacy OLD 12-trains/track methodology was dropped,
decision 2026-06-10). Sources:
  - CC construction/maintenance: 5A registry cost fields (read_record_meta) —
    full cost to EACH svc-int that requires the CC (standalone alternatives,
    no splitting). Replaces the costs_connection_curves.xlsx lookup.
  - CAP construction/maintenance: the 5C attribution table
    (svc_int_cap_attribution.csv, candidate - baseline deltas).
  - Uncovered operating cost: signed per-direction delta TRAIN-metres of the
    changed/added line variants (Option A, decision 2026-06-11): per variant
    dev_route_m x dev_total_dep - base_route_m x base_total_dep, normalised by
    operating_cost_ref_daily_dep (28 = the S14 service level behind the
    879 CHF/m/a rate), x operating_cost_s_bahn_per_meter x (1 - general_KDG).
    At total_dep 28 this reproduces the previous delta-route-m rule exactly;
    a pure frequency doubling now pays L x base_dep/28 x rate instead of 0.

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
                               base_infra: str = None,
                               use_cache: bool = True) -> pd.DataFrame:
    """Phase 8B: construction + maintenance + operating cost per svc-int.

    Args:
        svc_int_ids: svc-int ids to cost (None = all registered ext + ndc).
        combo: the '<infra>__<svc>' workspace key.
        base_infra: base infra version the deltas were projected on
            (None = the combo's infra half).
        use_cache: skip when construction_cost.csv already covers all ids and
            the costs_8b manifest matches.

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
        svc_int_ids = [i for t in so.SUPPORTED_SVC_INT_TYPES
                       for i in so.list_svc_int_ids(t, network=combo)]
    svc_int_ids = [str(i) for i in svc_int_ids]
    out_dir = paths.get_costs_combo_dir(combo)
    csv_path = paths.get_construction_cost_csv(combo)
    versions = {'combo': combo, 'base_infra': base_infra}

    print(f"\n=== Phase 8B — Construction/Maintenance/Operating Costs "
          f"({len(svc_int_ids)} svc-int(s)) ===")
    print(f"  combo: {combo} | duration: {cp.duration}y | op rate: "
          f"{cp.operating_cost_s_bahn_per_meter} CHF/m/a x "
          f"(1 - KDG {cp.general_KDG}) x dep/"
          f"{cp.operating_cost_ref_daily_dep} (delta train-m)")
    if not svc_int_ids:
        print("  no svc-ints registered — nothing to do")
        return pd.DataFrame(columns=_OUT_COLS)

    if use_cache and os.path.exists(csv_path) and cache_manifest.check_manifest(
            out_dir, 'costs_8b', versions, name='_settings_manifest_costs_8b.json'):
        cached = pd.read_csv(csv_path)
        if set(svc_int_ids) <= set(cached['Development'].astype(str)):
            print(f"  [8B] use_cache_costs: {csv_path} covers all "
                  f"{len(svc_int_ids)} svc-int(s) — skipping")
            return cached
        print("  [8B] cache present but missing svc-int(s) — recomputing")

    attribution = _load_attribution(combo)
    base_lengths = _route_lengths(
        paths.get_projected_services_path(svc_version, base_infra))
    base_deps = _route_deps(os.path.join(
        paths.MAIN, paths.RAIL_LINES_DIR, svc_version + '_network',
        paths.SERVICES_UNPROJECTED_SUBDIR, 'rail_lines.gpkg'))

    # Serial by design: 8B is ~50 ms/svc-int (registry + attribution reads + one
    # gpkg read), so loky spawn/pickle overhead makes parallelism slower
    # (measured 0.74x). 8A is the parallel valuation subphase; 8B stays serial.
    rows = []
    for iid in svc_int_ids:
        rec = so.read_record(_svc_int_type(iid), iid, network=combo)
        if rec is None:
            print(f"  [8B] {iid}: not in the registry — skipped")
            continue
        rows.append(_costs_for_svc_int(iid, rec, combo, base_infra,
                                       attribution, base_lengths, base_deps))
    result = pd.DataFrame(rows, columns=_OUT_COLS)

    os.makedirs(out_dir, exist_ok=True)
    result.to_csv(csv_path, index=False)
    cache_manifest.write_manifest(out_dir, 'costs_8b', versions,
                                  name='_settings_manifest_costs_8b.json')
    print(f"  [csv] wrote {csv_path} ({len(result)} svc-int row(s))")
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


def _route_deps(lines_gpkg_path: str) -> dict:
    """total_dep per line variant from a rail_lines.gpkg, summed over layers.
    Keys (route_id, direction_id, variant_rank) typed to match _route_lengths
    (str, int, int). Missing file → {} (caller falls back, warned)."""
    if not os.path.exists(lines_gpkg_path):
        print(f"  [8B] WARNING: no rail_lines.gpkg at {lines_gpkg_path} — "
              f"train-m scaling falls back to route-m for these variants")
        return {}
    out: dict = {}
    for layer in fiona.listlayers(lines_gpkg_path):
        gdf = gpd.read_file(lines_gpkg_path, layer=layer)
        if gdf.empty or 'total_dep' not in gdf.columns:
            continue
        for _, r in gdf.iterrows():
            key = (str(r['route_id']), int(float(r['direction_id'])),
                   int(float(r['variant_rank'])))
            out[key] = out.get(key, 0.0) + float(r.get('total_dep', 0) or 0)
    return out


def _delta_train_m(svc_int_id: str, combo: str, base_infra: str,
                   base_lengths: pd.Series, base_deps: dict) -> float:
    """Signed mean-over-directions delta train-metres at the calibration
    reference (Option A, 2026-06-11): per changed/added variant
    dev_len x dev_dep - base_len x base_dep, divided by
    operating_cost_ref_daily_dep. Variants absent from the delta are unchanged
    and contribute 0; at total_dep 28 (every EXT/NDC) this equals the previous
    delta route-metres exactly; a pure doubling yields L x base_dep/ref.

    STP (stopping-pattern change) keeps both the route length and the departures,
    so this reads ~0 for it. The fleet step a longer cycle time (added stops) or a
    doubled headway implies is NOT modelled here — a vehicle-count / rolling-stock
    operating cost is a deferred, permanent limitation (decision F4, 2026-06-13:
    needs cycle-time/turnaround data and a consistent global re-cost of EXT/NDC/FRQ).
    STP is NOT benefit-only regardless: the user-time effects of re-stopping (a
    longer ride from an added stop; a longer wait/access from a dropped call) are
    valued in Phase 8A. As of 2026-06-21 (supervisor decision) those slowdowns are
    no longer netted against the travel-time savings but split off as a separate
    travel-time-LOSS cost (monetized_tt_loss_yearly in traveltime_savings.csv) that
    Phase 9 routes to the CBA cost side — so they appear as a cost here in spirit,
    computed in 8A where the demand and skims live, not recomputed in 8B."""
    delta_path = paths.get_svc_int_projected_path(svc_int_id, base_infra, combo)
    dev = _route_lengths(delta_path)
    if dev.empty:
        return 0.0
    dev_deps = _route_deps(os.path.join(
        paths.get_svc_int_network_dir(svc_int_id, combo),
        paths.SERVICES_UNPROJECTED_SUBDIR, 'rail_lines.gpkg'))
    ref = float(cp.operating_cost_ref_daily_dep)
    per_direction: dict = {}
    for (gtfs_id, direction, rank), dev_len in dev.items():
        key = (gtfs_id, direction, rank)
        base_len = float(base_lengths.get(key, 0.0))
        base_dep = float(base_deps.get(key, 0.0))
        dev_dep = dev_deps.get(key)
        if dev_dep is None:
            # delta lines row missing — degrade to the old length-only rule
            dev_dep = base_dep if base_dep > 0 else ref
        per_direction[direction] = (per_direction.get(direction, 0.0)
                                    + float(dev_len) * float(dev_dep)
                                    - base_len * base_dep)
    return sum(per_direction.values()) / len(per_direction) / ref


def _costs_for_svc_int(svc_int_id: str, rec: dict, combo: str, base_infra: str,
                       attribution: pd.DataFrame,
                       base_lengths: pd.Series, base_deps: dict) -> dict:
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

    # Operating: signed delta train-metres (at the 28-dep reference) x rate
    # x uncovered share.
    try:
        delta_m = _delta_train_m(svc_int_id, combo, base_infra,
                                 base_lengths, base_deps)
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
          f"{cp.duration}y | delta train-km {delta_m / 1000.0:+,.2f} km/dir "
          f"@{cp.operating_cost_ref_daily_dep}dep -> op "
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
    compute_construction_costs(combo=_combo, use_cache=_cache)
