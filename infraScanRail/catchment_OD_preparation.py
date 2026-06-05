"""catchment_OD_preparation.py
Last modified: 2026-05-27

Station-pair OD matrix preparation for two methods (PT-feeder, Municipal)
across three time windows (AM peak, off-peak, all-day). Both methods consume
the same scaled communal OD; differ only in cell-to-station attribution.

Public entry point: prepare_all_od_matrices(use_cache: bool) -> None

Pipeline (W3):
  1. Load catchment boundary, rail stations (within boundary), name lookup.
  2. Load daily communal OD via scoring.GetOevDemandPerCommune(tau=1).
  3. Scale each OD pair forward to POPULATION_BASE_YEAR using the geometric
     mean of its endpoints' per-commune growth factors.
  4. PT-feeder branch: apply per-(commune, station) Pop/FTE shares read from
     catchment_allocate's station_commune_breakdown.csv as origin/dest weights.
  5. Municipal branch (Phase 3): commune→station 1:1 mapping (Phase 3).
  6. Emit per (method, time-window) name-keyed station OD CSV.
  7. Print conservation diagnostics for each method.
"""

import json
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np
import pandas as pd

import paths
import settings
import scoring
import cost_parameters as cp
import catchment_base
import catchment_allocate

_CODEBASE_CRS = 'EPSG:2056'

# Interactive flag — standalone __main__ leaves it True (prompts for gateway
# assignment); main_new.py sets it False so saved assignments load silently.
_INTERACTIVE_MODE = True

# Diagnostic plots — updated to the versioned path by prepare_all_od_matrices()
# when catchment_base.setup_versioned_dirs() has been called beforehand.
OD_COMPARISON_PLOT_DIR = catchment_base.OD_COMPARISON_PLOT_DIR


# ===============================================================================
# PUBLIC ENTRY POINT
# ===============================================================================

def prepare_all_od_matrices(use_cache: bool = False, svc_version: str = '',
                            infra_version: str = '', method: str = '',
                            attribution_mode: str = '') -> None:
    """Generate station-pair OD matrices for the selected method(s) × three windows.

    Args:
        use_cache:   If True, skip writing any CSV that already exists at its
                     target path. The long-format build is still done (cheap).
        svc_version: Service version folder name, e.g. 'SVC2026_ZH_S18_network'.
                     Required when running standalone; inferred automatically when
                     called after catchment_allocate.get_catchment().
        infra_version: Infrastructure version subfolder holding boundary_stations.json
                     (e.g. 'AS_2026_ZH'). Resolved automatically when empty.
        method:      'pt_feeder' | 'municipal' | 'both' | '' (empty ->
                     settings.CATCHMENT_METHOD). main_new passes the active method.
        attribution_mode: 'specific' | 'blended' | '' (empty ->
                     settings.OD_ATTRIBUTION_MODE). PT-Feeder only.

    Writes (under paths.MAIN):
        data/Traffic_Flow/OD/<svc_network>/<PT_Feeder|Municipal>/od_matrix_stations_{peak,off_peak,full_day}.csv
        data/Traffic_Flow/OD/<svc_network>/<PT_Feeder|Municipal>/od_matrix_stations.xlsx  (full station×station matrix, one sheet per window)
        data/Traffic_Flow/OD/<svc_network>/<PT_Feeder|Municipal>/od_top5.xlsx
        data/Traffic_Flow/OD/<svc_network>/Gateway/{gateway_zone_assignment.json,gateway_out_of_catchment.csv,gateway_splits.csv}
    """
    global OD_COMPARISON_PLOT_DIR

    if svc_version:
        catchment_base.setup_versioned_dirs(svc_version)
        catchment_allocate._RAIL_BASE = os.path.join(
            paths.RAIL_LINES_DIR, svc_version, paths.SERVICES_UNPROJECTED_SUBDIR)

    OD_COMPARISON_PLOT_DIR = catchment_base.OD_COMPARISON_PLOT_DIR

    methods   = _resolve_methods(method)
    attr_mode = (attribution_mode or settings.OD_ATTRIBUTION_MODE).strip().lower()

    print("\n=== Preparing station-pair OD matrices (W3) ===")
    print(f"  Method(s): {', '.join(methods)}   Attribution: {attr_mode}")

    boundary      = catchment_base._load_catchment_boundary()
    rail_stations = catchment_allocate._load_rail_stations(boundary, 'all', buffer=0)
    name_lookup   = _build_station_name_lookup(rail_stations)

    scale_factors, agg_fallback = _compute_commune_scale_factors()

    communal_od   = _load_communal_od()
    total_before  = float(communal_od['wert'].sum())
    communal_od   = _apply_scale_factors(communal_od, scale_factors, agg_fallback)
    total_after   = float(communal_od['wert'].sum())
    growth_pct    = 100.0 * (total_after / total_before - 1.0) if total_before else 0.0
    print(f"  Communal OD scaled 2018→{settings.POPULATION_BASE_YEAR} "
          f"(per-commune geometric mean): "
          f"total {total_before:,.0f} → {total_after:,.0f} ({growth_pct:+.1f}%)")

    # --- Gateway routing: classify in/out of catchment, assign gateways ---
    commune_gdf, bfs_col = _load_commune_boundaries()
    in_bnd_bfs           = _get_in_catchment_bfs(boundary, commune_gdf, bfs_col)
    internal_od, external_od = _classify_od(communal_od, in_bnd_bfs)
    gateways = _prepare_gateways(external_od, commune_gdf, bfs_col,
                                 in_bnd_bfs, svc_version, infra_version)

    # Extend name_lookup with gateway stations (they are outside the catchment
    # boundary so _build_station_name_lookup never sees them).
    for _gid, (_gname, _) in gateways['bs_index'].items():
        _sid_str = str(int(_gid))
        if _sid_str not in name_lookup:
            name_lookup[_sid_str] = _gname

    # Expand external-leg OD to gateway stations (split across gateways by
    # service volume), then route through the branch machinery via identity
    # weights/lookup (Option A). branch_od excludes both-ends-external pairs.
    gateway_od = _build_gateway_od(external_od, gateways['gw_weights'], in_bnd_bfs)
    branch_od  = pd.concat([internal_od, gateway_od], ignore_index=True)
    gw_ids     = gateways['gateway_station_ids']

    # --- Branch dispatch (only the selected method(s)) ---
    pt_long = muni_long = None
    if 'pt_feeder' in methods:
        pt_long, pt_orig_weights = _run_pt_feeder_branch(branch_od, gw_ids, attr_mode)
        _diagnose_conservation(branch_od, pt_long, pt_orig_weights, 'PT-Feeder')
    if 'municipal' in methods:
        muni_long = _run_municipal_branch(branch_od, gw_ids)
        _diagnose_conservation(branch_od, muni_long, None, 'Municipal')

    # --- Emit CSVs per (method, window) into the versioned OD tree ---
    windows = [(cp.TAU_PEAK_SHARE,     'peak'),
               (cp.TAU_OFFPEAK_SHARE,  'off_peak'),
               (cp.TAU_FULL_DAY_SHARE, 'full_day')]
    branch_longs = {'pt_feeder': pt_long, 'municipal': muni_long}
    print("\n  Writing CSVs ...")
    for branch in methods:
        long_df = branch_longs[branch]
        if long_df is None:
            continue
        for tau, suffix in windows:
            out_path = paths.get_station_od_csv(svc_version, branch, suffix)
            if use_cache and Path(out_path).exists():
                print(f"    cached: {out_path}")
            else:
                _apply_window_scaling(long_df, tau, name_lookup, out_path,
                                      label=f'{branch} {suffix}')

    # --- Top-5 origins/destinations Excel export + per-method OD pie map ---
    sa_stations_gdf = _load_sa_stations(rail_stations)
    sa_ids = set(pd.to_numeric(sa_stations_gdf['id_point'], errors='coerce')
                 .dropna().astype(int).tolist())
    for branch in methods:
        long_df = branch_longs[branch]
        if long_df is not None:
            _export_od_matrix_excel(long_df, windows, name_lookup,
                                    svc_version, branch)
            _export_top5_excel(long_df, sa_ids, name_lookup, branch, svc_version)
            _plot_sa_stations_od_map(long_df, sa_stations_gdf, name_lookup,
                                     branch, svc_version)

    # --- Diagnostic plots (comparison; only when both methods are present) ---
    if pt_long is not None and muni_long is not None:
        _plot_od_diagnostics(pt_long, muni_long, rail_stations, boundary,
                              name_lookup)

    print("\n=== W3 OD matrices done ===")


def _resolve_methods(method: str) -> list:
    """Resolve the method argument to a list of branch names.

    Empty -> [settings.CATCHMENT_METHOD]; 'both' -> both branches.
    """
    m = (method or '').strip().lower().replace('-', '_')
    if m in ('pt_feeder', 'ptfeeder'):
        return ['pt_feeder']
    if m in ('municipal', 'muni'):
        return ['municipal']
    if m == 'both':
        return ['pt_feeder', 'municipal']
    cm = settings.CATCHMENT_METHOD.strip().lower().replace('-', '_')
    return ['pt_feeder'] if cm == 'pt_feeder' else ['municipal']


# Backwards-compat alias — DEPRECATED. New callers should use
# prepare_all_od_matrices, which produces six matrices instead of one.
prepare_od_pt_feeder_a = prepare_all_od_matrices


# ===============================================================================
# NEW SHARED HELPERS (W3)
# ===============================================================================

def _compute_commune_scale_factors() -> tuple:
    """Per-commune forward growth factors pop_{POPULATION_BASE_YEAR} / pop_2018.

    Both years are loaded via catchment_base.load_commune_pop(year) (cantonal
    xlsx actuals 1962-2025; bezirk projection 2026-2050; Eurostat-extended
    2051-2100). A factor is produced for every commune with positive 2018
    population. Communes absent from either series fall back to the catchment-
    wide aggregate factor.

    Returns:
        (factors, agg_fallback)
        factors:      dict[int BFS_NR -> float growth factor]
        agg_fallback: float — aggregate pop_base / pop_2018 over all communes.
    """
    base_year = settings.POPULATION_BASE_YEAR
    print(f"  Computing per-commune 2018->{base_year} growth factors ...")

    pop_2018 = catchment_base.load_commune_pop(2018)
    pop_base = catchment_base.load_commune_pop(base_year)

    factors = {}
    for bfs, p0 in pop_2018.items():
        p0 = float(p0)
        p1 = float(pop_base.get(bfs, 0.0))
        if p0 > 0 and p1 > 0:
            factors[int(bfs)] = p1 / p0

    tot0 = float(pop_2018[pop_2018 > 0].sum())
    tot1 = float(pop_base[pop_base.index.isin(pop_2018.index)].sum())
    agg_fallback = (tot1 / tot0) if tot0 > 0 else 1.0

    if factors:
        vals = np.array(list(factors.values()), dtype=float)
        print(f"    {len(factors)} commune factors  "
              f"min={vals.min():.3f}  median={np.median(vals):.3f}  "
              f"max={vals.max():.3f}")
    print(f"    Aggregate fallback factor 2018->{base_year}: {agg_fallback:.4f}")
    return factors, agg_fallback


def _apply_scale_factors(communal_od: pd.DataFrame, factors: dict,
                         agg_fallback: float) -> pd.DataFrame:
    """Scale each OD pair by the geometric mean of its endpoint growth factors.

    trips_scaled(i,j) = wert(i,j) * sqrt(factor_i * factor_j)

    The geometric mean avoids the over-statement of the multiplicative
    factor_i * factor_j form (e.g. two communes each growing 20% would inflate a
    flow by 44% rather than 20%). Communes without a factor use agg_fallback.

    Args:
        communal_od:  DataFrame[quelle_code, ziel_code, wert].
        factors:      dict[int BFS_NR -> float] from _compute_commune_scale_factors.
        agg_fallback: float fallback factor for communes missing from `factors`.

    Returns:
        Copy of communal_od with `wert` scaled in place.
    """
    od = communal_od.copy()
    f_o = od['quelle_code'].map(factors).fillna(agg_fallback)
    f_d = od['ziel_code'].map(factors).fillna(agg_fallback)
    od['wert'] = od['wert'] * np.sqrt(f_o * f_d)
    return od


# ===============================================================================
# GATEWAY ROUTING (out-of-catchment handling)
# ===============================================================================

def _get_in_catchment_bfs(boundary, commune_gdf, bfs_col) -> set:
    """Return the set of BFS codes whose commune intersects the catchment boundary."""
    bnd_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=_CODEBASE_CRS)
    inb = gpd.sjoin(commune_gdf, bnd_gdf, predicate='intersects', how='inner')
    return set(pd.to_numeric(inb[bfs_col], errors='coerce')
               .dropna().astype(int).tolist())


def _classify_od(communal_od: pd.DataFrame, in_bnd_bfs: set) -> tuple:
    """Split communal OD into internal (both ends in catchment) and external
    (at least one end outside: code > 9999, or a BFS not in the catchment).

    External includes both single-leg pairs (one end outside) AND both-ends-
    external pairs — every external end is routed to its gateway(s) downstream.

    Returns:
        (internal_df, external_df) — same columns as the input.
    """
    o_in = communal_od['quelle_code'].isin(in_bnd_bfs)
    d_in = communal_od['ziel_code'].isin(in_bnd_bfs)
    internal = communal_od[o_in & d_in].copy()
    external = communal_od[~(o_in & d_in)].copy()
    n_both   = int((~o_in & ~d_in).sum())

    tot = float(communal_od['wert'].sum())
    print(f"\n  OD classification (vs catchment boundary):")
    print(f"    Internal pairs:     {len(internal):>7,}  "
          f"({100*internal['wert'].sum()/max(tot,1e-9):5.1f}% demand)")
    print(f"    External pairs:     {len(external):>7,}  "
          f"({100*external['wert'].sum()/max(tot,1e-9):5.1f}% demand)  "
          f"[{n_both:,} both-ends external]")
    return internal, external


def _load_zone_names() -> dict:
    """Read external-zone names (code > 9999) from the KTZH OD xlsx.

    Returns dict[int code -> str name]; empty if the name columns are absent.
    """
    try:
        raw = pd.read_excel(
            paths.OD_KT_ZH_PATH,
            usecols=lambda c: c in ('quelle_code', 'quelle_name',
                                    'ziel_code', 'ziel_name'))
    except Exception as exc:
        print(f"    Could not read zone names from OD xlsx: {exc}")
        return {}
    names = {}
    for code_col, name_col in (('quelle_code', 'quelle_name'),
                               ('ziel_code', 'ziel_name')):
        if code_col in raw.columns and name_col in raw.columns:
            sub = raw[[code_col, name_col]].dropna()
            for code, name in zip(sub[code_col], sub[name_col]):
                try:
                    ci = int(code)
                except (ValueError, TypeError):
                    continue
                if ci > 9999:
                    names.setdefault(ci, str(name).strip())
    return names


def _resolve_infra_version(svc_network: str, infra_version: str) -> str:
    """Resolve the infra-version subfolder holding boundary_stations.json.

    Uses infra_version when given; otherwise scans
    data/Network/Rail_Lines/{svc_network}/ for subfolders that contain the JSON,
    returning the single match or prompting when several exist (interactive only).
    """
    if infra_version:
        return infra_version
    base = os.path.join(paths.MAIN, paths.RAIL_LINES_DIR, svc_network)
    found = []
    if os.path.isdir(base):
        for d in sorted(os.listdir(base)):
            if os.path.exists(os.path.join(base, d, 'boundary_stations.json')):
                found.append(d)
    if not found:
        return ''
    if len(found) == 1:
        return found[0]
    if _INTERACTIVE_MODE:
        print("  Multiple infrastructure versions with boundary stations:")
        for i, d in enumerate(found, 1):
            print(f"    {i}) {d}")
        while True:
            raw = input("  Select infra version [1]: ").strip() or '1'
            if raw.isdigit() and 1 <= int(raw) <= len(found):
                return found[int(raw) - 1]
            print(f"    Invalid — enter 1–{len(found)}.")
    return found[0]


def _load_boundary_stations(svc_network: str, infra_version: str) -> list:
    """Load confirmed boundary (gateway) station node IDs from the Phase-3B JSON.

    Returns list[int]; empty list if the file is absent (gateway routing then
    disabled with a warning).
    """
    infra = _resolve_infra_version(svc_network, infra_version)
    if not infra:
        print("  WARNING: no boundary_stations.json found — gateway routing disabled.")
        return []
    path = paths.get_boundary_stations_json(svc_network, infra)
    if not os.path.exists(path):
        print(f"  WARNING: boundary stations file missing at {path} — "
              f"gateway routing disabled.")
        return []
    with open(path, encoding='utf-8') as f:
        ids = json.load(f)
    out = [int(x) for x in ids]
    print(f"  Loaded {len(out)} boundary (gateway) stations from infra '{infra}'")
    return out


def _build_boundary_station_index(boundary_ids: list, infra_version: str) -> dict:
    """Map each boundary station id -> (name, shapely point).

    Reads the infrastructure nodes GeoPackage
    (data/Infrastructure/<infra_version>/nodes.gpkg) — the same source the
    services projection uses for boundary-station names — so every boundary
    station resolves to its name and coordinates, including the ones outside the
    catchment buffer (rail stops only cover in-catchment stations).
    """
    if not boundary_ids:
        return {}
    nodes_path = os.path.join(paths.get_infra_version_dir(infra_version),
                              'nodes.gpkg')
    if not os.path.exists(nodes_path):
        print(f"    WARNING: infra nodes not found at {nodes_path}; "
              f"boundary station names/geometry unavailable.")
        return {}
    nodes = gpd.read_file(nodes_path).to_crs(_CODEBASE_CRS)
    nodes['num_int'] = pd.to_numeric(nodes['Number'], errors='coerce')
    nodes = nodes.dropna(subset=['num_int'])
    nodes['num_int'] = nodes['num_int'].astype(int)

    want = set(int(b) for b in boundary_ids)
    idx = {}
    for _, r in nodes[nodes['num_int'].isin(want)].iterrows():
        name = str(r.get('Name') or r['num_int'])
        idx[r['num_int']] = (name, r.geometry)
    missing = want - set(idx.keys())
    if missing:
        print(f"    Note: {len(missing)} boundary station(s) absent from infra "
              f"nodes: {sorted(missing)}")
    return idx


def _normalise_assignment(raw: dict) -> dict:
    """Normalise a loaded zone-assignment dict to dict[int code -> list[int]].

    Accepts both the legacy single-station format ({code: id}) and the
    multi-station format ({code: [id, ...]}).
    """
    out = {}
    for k, v in raw.items():
        code = int(k)
        if isinstance(v, (list, tuple)):
            out[code] = [int(x) for x in v]
        else:
            out[code] = [int(v)]
    return out


def _assign_external_zones(zone_codes, boundary_ids, bs_index, zone_names,
                           json_path) -> dict:
    """Assign each external GVM zone (code > 9999) to one or more gateway stations.

    Multiple stations per zone are allowed; the zone's demand is later split
    across them by service volume. Persisted to json_path as {code: [ids]}. On
    re-run the saved assignment is offered for reuse (interactive) or loaded
    silently (automated). Mirrors the municipal station-assignment override.

    Returns dict[int zone_code -> list[int gateway_station_id]] (skipped absent).
    """
    zone_codes = sorted(int(z) for z in zone_codes)
    saved = None
    if os.path.exists(json_path):
        with open(json_path, encoding='utf-8') as f:
            saved = _normalise_assignment(json.load(f))

    if saved is not None:
        complete = all(z in saved for z in zone_codes)
        if not _INTERACTIVE_MODE:
            if not complete:
                missing = [z for z in zone_codes if z not in saved]
                raise FileNotFoundError(
                    f"Gateway zone assignment at {json_path} is missing zones "
                    f"{missing}. Run catchment_OD_preparation standalone to "
                    f"complete the assignment.")
            print(f"  Loaded gateway zone assignment ({len(saved)} zones)")
            return {z: saved[z] for z in zone_codes}
        print(f"  Existing gateway zone assignment found: {json_path}")
        print("  1) Use existing")
        print("  2) Recreate")
        while True:
            choice = input("  Select [1]: ").strip() or '1'
            if choice == '1':
                if complete:
                    return {z: saved[z] for z in zone_codes}
                print("    Existing assignment incomplete; prompting for "
                      "missing zones only.")
                break
            if choice == '2':
                saved = None
                break
            print("    Enter 1 or 2.")

    if not _INTERACTIVE_MODE:
        raise FileNotFoundError(
            f"No gateway zone assignment at {json_path} and not in interactive "
            f"mode. Run catchment_OD_preparation standalone first.")

    mapping = dict(saved) if saved else {}
    ordered_ids = sorted(boundary_ids)
    print("\n  Assign each external zone to one or more gateway (boundary) stations.")
    print("  Enter station numbers separated by commas to split a zone across "
          "several gateways.")
    print("  Boundary stations:")
    for i, sid in enumerate(ordered_ids, 1):
        nm = bs_index.get(sid, (str(sid), None))[0]
        print(f"    {i:>2}) {sid}  {nm}")
    for z in zone_codes:
        if z in mapping:
            continue
        zlabel = zone_names.get(z, '')
        while True:
            raw = input(f"    Zone {z} {zlabel} -> station number(s), "
                        f"comma-separated (or 's' to skip): ").strip()
            if raw.lower() == 's':
                print(f"      zone {z} skipped (demand dropped)")
                break
            picks = [p for p in raw.replace(',', ' ').split() if p]
            if picks and all(p.isdigit() and 1 <= int(p) <= len(ordered_ids)
                             for p in picks):
                seen, chosen = set(), []
                for p in picks:
                    sid = ordered_ids[int(p) - 1]
                    if sid not in seen:
                        seen.add(sid)
                        chosen.append(sid)
                mapping[z] = chosen
                names = ', '.join(bs_index.get(s, (str(s), None))[0] for s in chosen)
                print(f"      zone {z} -> {names}")
                break
            print(f"      Invalid — enter one or more of 1–{len(ordered_ids)} "
                  f"(comma-separated) or 's'.")

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({str(k): [int(s) for s in v] for k, v in mapping.items()},
                  f, indent=2)
    print(f"  Saved gateway zone assignment → {json_path}")
    return mapping


def _assign_out_of_catchment_communes(bfs_codes, commune_gdf, bfs_col,
                                      boundary_ids, bs_index, csv_path) -> dict:
    """Auto-assign each out-of-catchment commune (real BFS, not in catchment) to
    its nearest gateway boundary station (Euclidean from commune centroid).

    Writes csv_path with the full assignment table.
    Returns dict[int BFS -> int gateway_station_id].
    """
    bfs_codes = sorted(int(b) for b in bfs_codes)
    if not bfs_codes or not boundary_ids:
        return {}
    pts = {sid: geom for sid, (nm, geom) in bs_index.items() if geom is not None}
    if not pts:
        print("    No boundary station geometries — cannot auto-assign communes.")
        return {}

    cg = commune_gdf.copy()
    cg['bfs_int'] = pd.to_numeric(cg[bfs_col], errors='coerce')
    cg = cg.dropna(subset=['bfs_int'])
    cg['bfs_int'] = cg['bfs_int'].astype(int)
    cg = cg[cg['bfs_int'].isin(bfs_codes)]

    sid_list  = list(pts.keys())
    geom_list = [pts[s] for s in sid_list]
    mapping, rows = {}, []
    for _, r in cg.iterrows():
        c = r.geometry.centroid
        dists = [c.distance(g) for g in geom_list]
        j = int(np.argmin(dists))
        sid = sid_list[j]
        mapping[r['bfs_int']] = sid
        rows.append({'BFS_NR': r['bfs_int'], 'gateway_station_id': sid,
                     'gateway_name': bs_index.get(sid, (str(sid), None))[0],
                     'distance_m': round(float(dists[j]), 1)})
    if rows:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        pd.DataFrame(rows).sort_values('BFS_NR').to_csv(
            csv_path, index=False, encoding='utf-8-sig')
        print(f"    {len(rows)} out-of-catchment communes auto-assigned → {csv_path}")
    return mapping


# --- Service-supply split (S-Bahn vs long-distance / inter-regional) ---------

# rail_segments.gpkg layer names by service bucket for the gateway split.
# Suburban/regional feeder side groups S-Bahn (109) with regional rail (RE/RB);
# long-distance side groups long-distance (102) with inter-regional (103).
_LOCAL_LAYERS = ('sbahn', 'regional_rail')
_LDIRT_LAYERS = ('long_distance_rail', 'inter_regional_rail')


def _gateway_segments_path(svc_network: str, infra_version: str) -> str:
    return os.path.join(paths.MAIN, paths.RAIL_LINES_DIR, svc_network,
                        infra_version, 'rail_segments.gpkg')


def _window_freq_series(df: pd.DataFrame, window: str) -> pd.Series:
    """Per-segment frequency for the chosen temporal window (dep/hr).

    full_day / all_day -> am + pm + offpeak ; peak -> am + pm ; offpeak -> offpeak.
    """
    for c in ('freq_am_peak_dep_hr', 'freq_pm_peak_dep_hr', 'freq_offpeak_dep_hr'):
        if c not in df.columns:
            df[c] = 0.0
    am = pd.to_numeric(df['freq_am_peak_dep_hr'],  errors='coerce').fillna(0.0)
    pm = pd.to_numeric(df['freq_pm_peak_dep_hr'],  errors='coerce').fillna(0.0)
    op = pd.to_numeric(df['freq_offpeak_dep_hr'],  errors='coerce').fillna(0.0)
    w = (window or 'full_day').lower()
    if w == 'peak':
        return am + pm
    if w in ('offpeak', 'off_peak'):
        return op
    return am + pm + op


def _freq_by_route_at_gateways(seg_path: str, layer: str, gateway_ids,
                               window: str) -> dict:
    """Summed window frequency of services crossing the canton boundary at each
    gateway station, for one route_type layer.

    Matching uses boundary_entry_node / boundary_exit_node — the node where a
    service enters/leaves the catchment — NOT the scheduled stop columns:
    long-distance / inter-regional trains traverse boundary stations without
    stopping, so they never appear as from/to stops. boundary_entry/exit_node is
    set consistently across all route types, giving a comparable cross-border
    supply measure. Each service variant is counted once per direction.
    """
    from pyogrio import list_layers
    try:
        available = list_layers(seg_path)[:, 0].tolist()
    except Exception:
        available = []
    if layer not in available:
        return {}

    seg = gpd.read_file(seg_path, layer=layer)
    if 'geometry' in seg.columns:
        seg = pd.DataFrame(seg.drop(columns='geometry'))
    for c in ('boundary_entry_node', 'boundary_exit_node'):
        seg[c] = pd.to_numeric(seg.get(c), errors='coerce')
    seg['freq'] = _window_freq_series(seg, window)

    want = {int(g) for g in gateway_ids}
    touch = seg[seg['boundary_entry_node'].isin(want)
                | seg['boundary_exit_node'].isin(want)]
    keys = [k for k in ('Service', 'direction_id', 'variant_rank')
            if k in touch.columns]
    out = {}
    for gid in want:
        sub = touch[(touch['boundary_entry_node'] == gid)
                    | (touch['boundary_exit_node'] == gid)]
        if sub.empty:
            continue
        uniq = sub.drop_duplicates(keys) if keys else sub
        out[gid] = float(uniq['freq'].sum())
    return out


def _compute_gateway_splits(gateway_ids, svc_network, infra_version, window,
                            bs_index, csv_path) -> pd.DataFrame:
    """Local (S-Bahn + RE) vs long-distance (LD + IR) service-supply split per
    gateway, written to csv_path.

    local_share = f(sbahn + regional) / (f(sbahn + regional) + f(long_distance +
    inter_regional)). Gateways with no crossing service in the window default to
    local_share=1.0. The split is metadata for downstream routing — the gateway
    carries full demand in the OD matrix itself.
    """
    gateway_ids = sorted({int(g) for g in gateway_ids})
    if not gateway_ids:
        return pd.DataFrame()
    seg_path = _gateway_segments_path(svc_network, infra_version)
    if not os.path.exists(seg_path):
        print(f"    Gateway split: rail_segments.gpkg missing at {seg_path}")
        return pd.DataFrame()

    local = {}
    for lyr in _LOCAL_LAYERS:
        for gid, f in _freq_by_route_at_gateways(
                seg_path, lyr, gateway_ids, window).items():
            local[gid] = local.get(gid, 0.0) + f
    ldirt = {}
    for lyr in _LDIRT_LAYERS:
        for gid, f in _freq_by_route_at_gateways(
                seg_path, lyr, gateway_ids, window).items():
            ldirt[gid] = ldirt.get(gid, 0.0) + f

    rows = []
    for gid in gateway_ids:
        lf_local, lf_ld = local.get(gid, 0.0), ldirt.get(gid, 0.0)
        tot = lf_local + lf_ld
        if tot > 0:
            ls = lf_local / tot
        else:
            ls = 1.0
            gid_name = bs_index.get(gid, (str(gid), None))[0]
            print(f"    Gateway {gid_name} ({gid}): no crossing service in window "
                  f"'{window}' — defaulting local share=1.0")
        rows.append({
            'gateway_station_id': gid,
            'station_name': bs_index.get(gid, (str(gid), None))[0],
            'local_freq': round(lf_local, 2), 'longdist_freq': round(lf_ld, 2),
            'local_share': round(ls, 4), 'longdist_share': round(1.0 - ls, 4),
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"    Gateway service-supply splits ({len(df)} gateways, "
          f"window='{window}') → {csv_path}")
    return df


def _build_gateway_weights(assignment: dict, volumes: dict) -> dict:
    """Turn a code -> [station, ...] assignment into code -> [(station, share)].

    Shares are proportional to each gateway's crossing service volume (sum of all
    route-type frequencies at the gateway). When no volume is available the demand
    is split equally across the assigned gateways.
    """
    gw_weights = {}
    for code, stations in assignment.items():
        if not stations:
            continue
        vols = [max(float(volumes.get(s, 0.0)), 0.0) for s in stations]
        tot = sum(vols)
        if tot > 0:
            shares = [v / tot for v in vols]
        else:
            shares = [1.0 / len(stations)] * len(stations)
        gw_weights[code] = list(zip([int(s) for s in stations], shares))
    return gw_weights


def _build_gateway_od(external_od: pd.DataFrame, gw_weights: dict,
                      in_bnd_bfs: set) -> pd.DataFrame:
    """Expand external OD into station-pair rows by replacing each external end
    with its assigned gateway station(s), splitting demand by gateway share.

    - one external end  -> the in-catchment end keeps its commune code (the
      branch disaggregates it); the external end becomes a gateway station id.
    - both ends external -> cross-product over both ends' gateways, demand
      multiplied by both shares.
    Pairs whose external code has no assignment (skipped zones) are dropped.
    """
    rows = []
    for r in external_od.itertuples(index=False):
        q, z, w = int(r.quelle_code), int(r.ziel_code), float(r.wert)
        q_ext = q not in in_bnd_bfs
        z_ext = z not in in_bnd_bfs
        if q_ext and z_ext:                    # both ends external
            for so, wo in gw_weights.get(q, []):
                for sd, wd in gw_weights.get(z, []):
                    rows.append((so, sd, w * wo * wd))
        elif q_ext:                            # origin external
            for stn, share in gw_weights.get(q, []):
                rows.append((stn, z, w * share))
        else:                                  # destination external
            for stn, share in gw_weights.get(z, []):
                rows.append((q, stn, w * share))
    return pd.DataFrame(rows, columns=['quelle_code', 'ziel_code', 'wert'])


def _prepare_gateways(external_od, commune_gdf, bfs_col, in_bnd_bfs,
                      svc_network, infra_version) -> dict:
    """Load gateway stations, assign external zones (multi) + out-of-catchment
    communes, compute the service-supply split, and derive per-code gateway
    weights (split by crossing service volume).

    Returns dict with keys: gw_weights (code -> [(station, share)]),
    gateway_station_ids (set), bs_index, splits.
    """
    empty = {'gw_weights': {}, 'gateway_station_ids': set(),
             'bs_index': {}, 'splits': pd.DataFrame()}
    infra = _resolve_infra_version(svc_network, infra_version)
    boundary_ids = _load_boundary_stations(svc_network, infra)
    if not boundary_ids:
        return empty

    bs_index   = _build_boundary_station_index(boundary_ids, infra)
    zone_names = _load_zone_names()

    ext_codes = set(external_od['quelle_code']).union(set(external_od['ziel_code']))
    ext_codes = {int(c) for c in ext_codes if int(c) not in in_bnd_bfs}
    zone_codes    = {c for c in ext_codes if c > 9999}
    commune_codes = {c for c in ext_codes if c <= 9999}

    gateway_dir = paths.get_gateway_dir(svc_network)
    json_path   = os.path.join(gateway_dir, 'gateway_zone_assignment.json')
    csv_path    = os.path.join(gateway_dir, 'gateway_out_of_catchment.csv')
    splits_path = os.path.join(gateway_dir, 'gateway_splits.csv')

    zone_map = (_assign_external_zones(zone_codes, boundary_ids, bs_index,
                                       zone_names, json_path)
                if zone_codes else {})
    commune_map = (_assign_out_of_catchment_communes(
                       commune_codes, commune_gdf, bfs_col, boundary_ids,
                       bs_index, csv_path)
                   if commune_codes else {})

    # Unified assignment: code -> list of gateway stations (communes are single).
    assignment = dict(zone_map)
    for code, stn in commune_map.items():
        assignment[code] = [stn]

    used_gateways = {s for stations in assignment.values() for s in stations}
    splits = _compute_gateway_splits(used_gateways, svc_network, infra,
                                     settings.TEMPORAL, bs_index, splits_path)

    volumes = {}
    if not splits.empty:
        volumes = {int(r.gateway_station_id): float(r.local_freq) + float(r.longdist_freq)
                   for r in splits.itertuples(index=False)}
    gw_weights = _build_gateway_weights(assignment, volumes)

    multi = sum(1 for v in gw_weights.values() if len(v) > 1)
    print(f"  Gateways ready: {len(zone_map)} external zones "
          f"({multi} split across multiple gateways), "
          f"{len(commune_map)} out-of-catchment communes, "
          f"{len(used_gateways)} gateways used.")
    return {'gw_weights': gw_weights, 'gateway_station_ids': used_gateways,
            'bs_index': bs_index, 'splits': splits}


def _build_station_name_lookup(rail_stations) -> dict:
    """Return dict[id_point_str → readable_name].

    Disambiguates collisions: if a stop_name appears more than once, append
    `(id_point)` so the CSV index/columns remain unique.
    """
    df = rail_stations[['id_point', 'stop_name']].copy()
    df['id_point']  = df['id_point'].astype(str)
    df['stop_name'] = df['stop_name'].fillna('').astype(str).str.strip()
    df = df.drop_duplicates('id_point')

    name_counts = df['stop_name'].value_counts()
    duplicated = set(name_counts[name_counts > 1].index)

    if duplicated:
        print(f"  Station-name lookup: {len(duplicated)} name collision(s); "
              f"appending (id_point) to disambiguate.")

    lookup = {}
    for _, row in df.iterrows():
        sid  = row['id_point']
        name = row['stop_name'] or sid
        if name in duplicated:
            name = f"{name} ({sid})"
        lookup[sid] = name
    return lookup


def _build_window_matrix(long_df: pd.DataFrame, tau_window: float,
                         name_lookup: dict) -> pd.DataFrame:
    """Scale a long-format station-pair OD by tau and pivot to a wide, name-keyed
    station×station matrix (origin rows × destination columns).

    Returns None when there is no positive-trip data after scaling.
    """
    if long_df is None or long_df.empty:
        return None
    scaled = long_df.copy()
    scaled['trips'] = scaled['trips'] * tau_window
    scaled = scaled[scaled['trips'] > 0].copy()
    if scaled.empty:
        return None

    matrix = scaled.pivot_table(
        index='origin_station_id',
        columns='dest_station_id',
        values='trips',
        aggfunc='sum',
        fill_value=0.0,
    )
    matrix.index   = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    matrix = matrix.rename(index=name_lookup, columns=name_lookup)
    matrix.index.name   = 'origin'
    matrix.columns.name = 'destination'
    return matrix


def _apply_window_scaling(long_df: pd.DataFrame, tau_window: float,
                          name_lookup: dict, output_path: str,
                          label: str = '') -> None:
    """Build the window station×station matrix and write it to CSV at
    `output_path` (absolute).
    """
    matrix = _build_window_matrix(long_df, tau_window, name_lookup)
    if matrix is None:
        print(f"    ({label}): no data — skipping {output_path}")
        return

    out_full = Path(output_path)
    out_full.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out_full, encoding='utf-8-sig')
    print(f"    Saved → {out_full}")
    print(f"      {label}: {matrix.shape[0]}×{matrix.shape[1]} stations, "
          f"total trips = {matrix.values.sum():,.1f}  (τ={tau_window})")


def _export_od_matrix_excel(long_df: pd.DataFrame, windows: list,
                            name_lookup: dict, svc_network: str,
                            method: str) -> None:
    """Write the full station×station OD matrix to a per-method workbook with one
    sheet per time window (peak / off_peak / full_day).

    Each sheet lists every rail station on both axes with the trips between them
    — the Excel counterpart of the per-window od_matrix_stations_*.csv files.
    """
    if long_df is None or long_df.empty:
        print(f"    OD matrix Excel ({method}): no data — skipped")
        return
    out_path = paths.get_station_od_matrix_xlsx(svc_network, method)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    wrote = False
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for tau, suffix in windows:
            matrix = _build_window_matrix(long_df, tau, name_lookup)
            if matrix is None:
                continue
            matrix.to_excel(writer, sheet_name=suffix)   # suffixes are ≤31 chars
            wrote = True
    if wrote:
        print(f"    Full OD matrix workbook → {out_path}")


def _diagnose_conservation(communal_od: pd.DataFrame,
                           station_long_df: pd.DataFrame,
                           orig_weights,
                           method_label: str) -> None:
    """Print conservation diagnostic for a method.

    Always reports total communal vs total station-pair trips. When
    `orig_weights` is provided (PT-feeder), additionally lists the top-5
    origin communes by share-loss (cells with no PT assignment).
    """
    total_communal = float(communal_od['wert'].sum())
    total_station  = float(station_long_df['trips'].sum())
    delta          = total_communal - total_station
    pct            = 100.0 * delta / max(total_communal, 1e-9)

    print(f"\n  Conservation diagnostic [{method_label}]:")
    print(f"    Communal OD total: {total_communal:>14,.1f}")
    print(f"    Station OD total:  {total_station:>14,.1f}")
    print(f"    Residual:          {delta:>+14,.1f}  ({pct:+.2f}%)")

    if orig_weights is None or len(orig_weights) == 0:
        if method_label.lower().startswith('munici'):
            print(f"    Municipal: 1:1 commune→station mapping; residual = "
                  f"trips for communes outside the assignment lookup.")
        return

    # Per-commune origin-side weight loss (PT-feeder)
    bfs_key = 'BFS' if 'BFS' in orig_weights.columns else 'quelle_code'
    sum_w = orig_weights.groupby(bfs_key)['orig_weight'].sum()
    losses = (1.0 - sum_w).clip(lower=0.0)
    losses = losses[losses > 1e-6].sort_values(ascending=False)
    if losses.empty:
        print(f"    PT-feeder: every origin commune fully covered "
              f"(no cells with NoPT assignment).")
        return

    row_total_per_bfs = communal_od.groupby('quelle_code')['wert'].sum()
    print(f"    Top-5 origin communes by share loss "
          f"(cells with no PT assignment):")
    for bfs, loss in losses.head(5).items():
        bfs_int   = int(bfs)
        row_total = float(row_total_per_bfs.get(bfs_int, 0.0))
        lost      = row_total * float(loss)
        print(f"      BFS {bfs_int:>5}  loss={float(loss)*100:5.1f}%  "
              f"row_total={row_total:>10,.0f}  lost≈{lost:>9,.1f}")


def _load_municipal_assignment() -> pd.DataFrame:
    """Read the municipal commune→station lookup written by
    catchment_allocate._run_municipal_method.

    Returns:
        DataFrame[BFS_NR (int), station_id (int), station_name (str)].
        Communes without a valid station assignment are dropped.
    """
    path = os.path.join(paths.MAIN, catchment_base.MUNICIPAL_DATA_DIR, 'station_assignment.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Municipal commune→station lookup not found at {path}. "
            f"Run catchment_allocate.get_catchment() with the municipal "
            f"method first."
        )
    print(f"  Loading municipal commune→station assignment ...")
    df = pd.read_csv(path, encoding='utf-8-sig')
    keep = ['BFS_NR', 'station_id', 'station_name']
    for c in keep:
        if c not in df.columns:
            raise ValueError(
                f"station_assignment.csv missing column '{c}'. "
                f"Available: {list(df.columns)}"
            )
    df = df[keep].copy()
    df['BFS_NR']     = pd.to_numeric(df['BFS_NR'],     errors='coerce')
    df['station_id'] = pd.to_numeric(df['station_id'], errors='coerce')
    df = df.dropna(subset=['BFS_NR', 'station_id'])
    df = df[df['station_id'] > 0]
    df['BFS_NR']     = df['BFS_NR'].astype(int)
    df['station_id'] = df['station_id'].astype(int)
    print(f"    {len(df):,} communes with a valid station assignment")
    return df


def _load_station_breakdown(method: str) -> pd.DataFrame:
    """Read the per-(commune, station) catchment breakdown written by
    catchment_allocate (station_commune_breakdown.csv).

    pop_share_pct / empl_share_pct are the population / FTE shares of each
    commune attributed to a station — already computed by the allocation
    (year-scaled, boundary-filtered, no-PT cells excluded), so they are used
    directly as the OD origin / destination weights. No-PT rows (station id
    NO_PT_ID = -1) are dropped; the remaining shares sum to ≤ 1 per commune.

    Args:
        method: 'pt_feeder' | 'municipal' — selects the versioned data dir.

    Returns:
        DataFrame[BFS (int), station_id (int), pop_share (float), empl_share (float)].
    """
    data_dir = (catchment_base.PT_FEEDER_DATA_DIR if method == 'pt_feeder'
                else catchment_base.MUNICIPAL_DATA_DIR)
    path = os.path.join(paths.MAIN, data_dir, 'station_commune_breakdown.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Station-commune breakdown not found at {path}. Run "
            f"catchment_allocate.get_catchment() (method '{method}') first."
        )
    print(f"  Loading station-commune catchment breakdown ({method}) ...")
    df = pd.read_csv(path, encoding='utf-8-sig')
    df = df[['BFS_NR', 'station_number', 'pop_share_pct', 'empl_share_pct']].copy()
    df['BFS']        = pd.to_numeric(df['BFS_NR'],         errors='coerce')
    df['station_id'] = pd.to_numeric(df['station_number'], errors='coerce')
    df = df.dropna(subset=['BFS', 'station_id'])
    df['BFS']        = df['BFS'].astype(int)
    df['station_id'] = df['station_id'].astype(int)
    df = df[df['station_id'] > 0]   # drop No-PT sentinel (NO_PT_ID = -1)
    df['pop_share']  = pd.to_numeric(df['pop_share_pct'],  errors='coerce').fillna(0.0) / 100.0
    df['empl_share'] = pd.to_numeric(df['empl_share_pct'], errors='coerce').fillna(0.0) / 100.0
    out = df[['BFS', 'station_id', 'pop_share', 'empl_share']].reset_index(drop=True)
    print(f"    {out['BFS'].nunique()} communes, {out['station_id'].nunique()} stations, "
          f"{len(out):,} (commune, station) pairs")
    return out


def _run_municipal_branch(communal_od: pd.DataFrame,
                          gateway_station_ids=None) -> pd.DataFrame:
    """Map communal OD to station-pair OD via the municipal commune→station
    1:1 lookup. Origin commune → origin station, destination commune →
    destination station; trips are summed by (origin_station, dest_station).

    Gateway ends in the OD are already station ids (the external leg was expanded
    upstream), so they are added to the lookup as identity entries (id -> id).

    Returns:
        Long-format DataFrame: origin_station_id (int), dest_station_id (int),
        trips (float).
    """
    print("\n  --- Municipal branch ---")
    assign = _load_municipal_assignment()
    bfs_to_stn = dict(zip(assign['BFS_NR'].astype(int),
                          assign['station_id'].astype(int)))
    if gateway_station_ids:
        bfs_to_stn.update({int(g): int(g) for g in gateway_station_ids})
        print(f"    + {len(gateway_station_ids)} gateway station(s) as identity entries")

    od = communal_od.copy()
    od['origin_station_id'] = od['quelle_code'].map(bfs_to_stn)
    od['dest_station_id']   = od['ziel_code']  .map(bfs_to_stn)

    n_in  = len(od)
    od    = od.dropna(subset=['origin_station_id', 'dest_station_id'])
    n_out = len(od)
    if n_out < n_in:
        print(f"    {n_in - n_out:,} of {n_in:,} OD pairs dropped "
              f"(commune outside the assignment lookup).")

    od['origin_station_id'] = od['origin_station_id'].astype(int)
    od['dest_station_id']   = od['dest_station_id'].astype(int)

    n_before_intra = len(od)
    od = od[od['origin_station_id'] != od['dest_station_id']]
    n_intra = n_before_intra - len(od)
    if n_intra:
        print(f"    {n_intra:,} intra-station pairs dropped (origin == dest station).")

    station_od = (od.groupby(['origin_station_id', 'dest_station_id'],
                             as_index=False)['wert'].sum()
                    .rename(columns={'wert': 'trips'}))
    station_od = station_od[station_od['trips'] > 0].copy()
    print(f"    {len(station_od):,} non-zero station OD pairs")
    return station_od


def _run_pt_feeder_branch(communal_od: pd.DataFrame, gateway_station_ids=None,
                          attribution_mode: str = 'specific') -> tuple:
    """Transplant communal OD to (rail-station, rail-station) pairs using the
    PT-Feeder catchment shares from catchment_allocate
    (station_commune_breakdown.csv).

    Origin / destination weights are the commune's population / FTE share
    attributed to each station — read directly from the allocation output, not
    recomputed, so the OD attribution matches the station catchments exactly.

    Attribution mode:
        'specific' — origin weights = population share, dest weights = FTE share.
        'blended'  — both sides = OD_SCALING_POP_WEIGHT×pop_share +
                     OD_SCALING_EMPL_WEIGHT×empl_share (symmetric).

    Gateway ends in the OD are already station ids (the external leg was expanded
    upstream), so each gateway station is injected as an identity (weight 1.0) row
    into both weight tables and resolves to itself in the reaggregation.

    Returns:
        (long_df, orig_weights)
        long_df:      DataFrame[origin_station_id, dest_station_id, trips]
        orig_weights: DataFrame[BFS, station_id, orig_weight] — for diagnostic.
    """
    print("\n  --- PT-feeder branch ---")
    breakdown = _load_station_breakdown('pt_feeder')

    if attribution_mode == 'blended':
        blended      = _compute_blended_weights(breakdown)
        orig_weights = blended.rename(columns={'weight': 'orig_weight'})
        dest_weights = blended.rename(columns={'weight': 'dest_weight'})
    else:
        orig_weights = (breakdown[['BFS', 'station_id', 'pop_share']]
                        .rename(columns={'pop_share': 'orig_weight'}))
        dest_weights = (breakdown[['BFS', 'station_id', 'empl_share']]
                        .rename(columns={'empl_share': 'dest_weight'}))

    orig_weights = _inject_identity_weights(orig_weights, gateway_station_ids,
                                            'orig_weight')
    dest_weights = _inject_identity_weights(dest_weights, gateway_station_ids,
                                            'dest_weight')

    print(f"    Origin weights: {orig_weights['BFS'].nunique()} communes, "
          f"{orig_weights['station_id'].nunique()} stations")
    print(f"    Dest  weights: {dest_weights['BFS'].nunique()} communes, "
          f"{dest_weights['station_id'].nunique()} stations")

    long_df = _reaggregate_to_stations(communal_od, orig_weights, dest_weights)
    return long_df, orig_weights


def _inject_identity_weights(weights: pd.DataFrame, gateway_station_ids,
                             weight_col: str) -> pd.DataFrame:
    """Append identity (weight 1.0) rows mapping each gateway station to itself,
    so gateway ends already present as station ids resolve through reaggregation."""
    if not gateway_station_ids:
        return weights
    rows = pd.DataFrame([
        {'BFS': int(g), 'station_id': int(g), weight_col: 1.0}
        for g in gateway_station_ids
    ])
    return pd.concat([weights, rows], ignore_index=True)


def _compute_blended_weights(breakdown: pd.DataFrame) -> pd.DataFrame:
    """Symmetric blended weight per (commune, station):
    (POP_WEIGHT×pop_share + EMPL_WEIGHT×empl_share) / (POP_WEIGHT + EMPL_WEIGHT).

    With OD_SCALING_EMPL_WEIGHT = 0 this reduces to the population share, applied
    to both origin and destination sides. pop_share / empl_share come from the
    catchment breakdown (each (commune, station) row carries both).

    Args:
        breakdown: DataFrame[BFS, station_id, pop_share, empl_share] from
                   _load_station_breakdown.

    Returns:
        DataFrame[BFS, station_id, weight].
    """
    pw = float(settings.OD_SCALING_POP_WEIGHT)
    ew = float(settings.OD_SCALING_EMPL_WEIGHT)
    denom = (pw + ew) if (pw + ew) > 0 else 1.0
    m = breakdown.copy()
    m['weight'] = (pw * m['pop_share'] + ew * m['empl_share']) / denom
    return m[['BFS', 'station_id', 'weight']]


# ===============================================================================
# SHARED HELPERS (commune boundaries, communal OD, station reaggregation)
# ===============================================================================

def _load_commune_boundaries() -> tuple:
    """Load municipal boundary GeoPackage, detect BFS column, project to LV95.

    Returns:
        (muni_gdf, bfs_col_name)
    """
    print("  Loading commune boundaries ...")
    muni = gpd.read_file(paths.MUNICIPAL_BOUNDARIES_GPKG).to_crs(_CODEBASE_CRS)
    if 'objektart' in muni.columns:
        muni = muni[muni['objektart'] == 'Gemeindegebiet'].copy()

    bfs_col = None
    for candidate in ['BFS_NR', 'bfs_nr', 'BFS_NUMMER', 'bfs_nummer', 'GMDNR', 'gmdnr']:
        if candidate in muni.columns:
            bfs_col = candidate
            break
    if bfs_col is None:
        raise ValueError(
            f"No BFS column found in commune boundaries. "
            f"Available columns: {list(muni.columns)}"
        )

    muni[bfs_col] = pd.to_numeric(muni[bfs_col], errors='coerce')
    muni = muni[[bfs_col, 'geometry']].dropna(subset=[bfs_col])
    print(f"    {len(muni)} communes loaded (BFS column: '{bfs_col}')")
    return muni, bfs_col


def _load_communal_od() -> pd.DataFrame:
    """Load canton-level PT OD (tau=1 = full daily demand), drop intrazonal pairs.

    Returns:
        DataFrame with columns: quelle_code (int), ziel_code (int), wert (float).
    """
    print("  Loading communal OD ...")
    od = scoring.GetOevDemandPerCommune(tau=1)
    od = od[od['quelle_code'] != od['ziel_code']].copy()
    od = od[od['wert'] > 0][['quelle_code', 'ziel_code', 'wert']].copy()
    od['quelle_code'] = od['quelle_code'].astype(int)
    od['ziel_code']   = od['ziel_code'].astype(int)
    print(f"    {len(od)} OD pairs with positive demand (intrazonal removed)")
    return od


def _reaggregate_to_stations(
    communal_od: pd.DataFrame,
    orig_weights: pd.DataFrame,
    dest_weights: pd.DataFrame
) -> pd.DataFrame:
    """Merge communal OD with (commune, station) weights; compute station-pair flows.

    Formula:
        trips(A, C) = Σ_{i,j} T_ij × orig_weight(i, A) × dest_weight(j, C)

    Returns:
        Long-format DataFrame: origin_station_id (int), dest_station_id (int), trips (float).
    """
    print("  Reaggregating communal OD to station pairs ...")

    merged = communal_od.merge(
        orig_weights.rename(columns={
            'BFS': 'quelle_code',
            'station_id': 'origin_station_id',
            'orig_weight': 'ow'
        }),
        on='quelle_code',
        how='inner'
    )

    merged = merged.merge(
        dest_weights.rename(columns={
            'BFS': 'ziel_code',
            'station_id': 'dest_station_id',
            'dest_weight': 'dw'
        }),
        on='ziel_code',
        how='inner'
    )

    merged['trips'] = merged['wert'] * merged['ow'] * merged['dw']

    n_before_intra = len(merged)
    merged = merged[merged['origin_station_id'] != merged['dest_station_id']]
    n_intra = n_before_intra - len(merged)
    if n_intra:
        print(f"    {n_intra:,} intra-station rows dropped (origin == dest station).")

    station_od = (
        merged.groupby(['origin_station_id', 'dest_station_id'])['trips']
        .sum()
        .reset_index()
    )
    station_od = station_od[station_od['trips'] > 0].copy()
    print(f"    {len(station_od):,} non-zero station OD pairs")
    return station_od


# ===============================================================================
# TOP-5 EXCEL EXPORT (study-area stations)
# ===============================================================================

def _load_sa_stations(rail_stations) -> gpd.GeoDataFrame:
    """Return SA rail stations using the same method catchment_allocate uses for
    the station_catchments breakdown.

    Filters the already-loaded `rail_stations` by the study-area boundary via
    catchment_allocate._filter_stations_to_sa — identical to the SA filter
    behind the Communes_by_station sheet, so the top-5 Excel and OD pie map
    cover exactly that station set.

    Returns:
        GeoDataFrame[id_point, stop_name, geometry] for SA-scoped stations.
    """
    sa_boundary = catchment_allocate._load_sa_boundary()
    sa_gdf      = catchment_allocate._filter_stations_to_sa(rail_stations, sa_boundary)
    if sa_gdf is None or sa_gdf.empty:
        print("  Study-area boundary unavailable — using all catchment stations.")
        return rail_stations.copy()
    print(f"  Study-area stations: {len(sa_gdf)}")
    return sa_gdf


def _top5_table(long_df: pd.DataFrame, sa_ids: set, name_lookup: dict,
                group_col: str, partner_col: str, partner_label: str,
                top_n: int = 5) -> pd.DataFrame:
    """Build a wide top-N table for SA stations on the given grouping side.

    group_col   = 'origin_station_id' (top destinations) or 'dest_station_id'
                  (top origins). partner_col is the other end.
    """
    def _nm(sid):
        return name_lookup.get(str(int(sid)), str(int(sid)))

    sub = long_df[long_df[group_col].isin(sa_ids)]
    totals = sub.groupby(group_col)['trips'].sum().sort_values(ascending=False)

    rows = []
    for sid in totals.index:
        grp = sub[sub[group_col] == sid].nlargest(top_n, 'trips')
        row = {'station': _nm(sid), 'total_trips': round(float(totals[sid]), 1)}
        for i, (_, r) in enumerate(grp.iterrows(), 1):
            row[f'{partner_label}_{i}'] = _nm(r[partner_col])
            row[f'trips_{i}']           = round(float(r['trips']), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def _export_top5_excel(long_df, sa_ids, name_lookup, method, svc_network,
                       top_n: int = 5) -> None:
    """Write 'Top_Destinations' and 'Top_Origins' sheets (SA stations only) to a
    standalone od_top5.xlsx in the versioned per-method OD directory.
    """
    if long_df is None or long_df.empty or not sa_ids:
        print(f"    Top-5 export ({method}): no data — skipped")
        return

    dest_tbl = _top5_table(long_df, sa_ids, name_lookup,
                           'origin_station_id', 'dest_station_id', 'dest', top_n)
    orig_tbl = _top5_table(long_df, sa_ids, name_lookup,
                           'dest_station_id', 'origin_station_id', 'origin', top_n)

    out_path = paths.get_od_top5_xlsx(svc_network, method)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        dest_tbl.to_excel(writer, sheet_name='Top_Destinations', index=False)
        orig_tbl.to_excel(writer, sheet_name='Top_Origins', index=False)
    print(f"    Top-5 origins/destinations → {out_path}")


# ===============================================================================
# DIAGNOSTIC PLOTS (W3 Phase 4)
# ===============================================================================

def _build_station_summary(rail_stations, pt_long, muni_long, name_lookup):
    """Combine per-station attracted trips for both methods + geometry + name.

    Attracted trips = sum over destinations (i.e., each value is the total
    number of incoming PT trips per station, all-day τ=1).
    """
    pt_attr = (pt_long.groupby('dest_station_id')['trips'].sum()
               if pt_long is not None and not pt_long.empty else pd.Series(dtype=float))
    mu_attr = (muni_long.groupby('dest_station_id')['trips'].sum()
               if muni_long is not None and not muni_long.empty else pd.Series(dtype=float))

    df = rail_stations[['id_point', 'stop_name', 'geometry']].copy()
    df['id_point']      = df['id_point'].astype(int)
    df['readable_name'] = (df['id_point'].astype(str).map(name_lookup)
                                         .fillna(df['stop_name']))
    df['pt_feeder_attracted'] = df['id_point'].map(pt_attr).fillna(0).astype(float)
    df['municipal_attracted'] = df['id_point'].map(mu_attr).fillna(0).astype(float)
    df['diff_pt_minus_muni']  = df['pt_feeder_attracted'] - df['municipal_attracted']
    return gpd.GeoDataFrame(df, geometry='geometry',
                            crs=catchment_base.CODEBASE_CRS)


def _plot_bar_attracted(summary: gpd.GeoDataFrame, top_n: int = 20) -> None:
    """Top-N stations bar chart, two bars per station (PT-Feeder vs Municipal)."""
    df = summary.copy()
    df['_max_attr'] = df[['pt_feeder_attracted', 'municipal_attracted']].max(axis=1)
    df = df.sort_values('_max_attr', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(df))
    w = 0.4
    ax.bar(x - w/2, df['pt_feeder_attracted'], width=w,
           color='#1565C0', label='PT-Feeder')
    ax.bar(x + w/2, df['municipal_attracted'], width=w,
           color='#E65100', label='Municipal')
    ax.set_xticks(x)
    ax.set_xticklabels(df['readable_name'].tolist(), rotation=60, ha='right')
    ax.set_ylabel('Attracted trips per day (all-day, τ=1)')
    ax.set_title(f'Top {top_n} stations by attracted PT trips — '
                 f'PT-Feeder vs Municipal')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(OD_COMPARISON_PLOT_DIR,
                            'od_compare_attracted_trips_bar.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved → {out_path}")


def _plot_spatial_attracted(summary: gpd.GeoDataFrame, boundary) -> None:
    """Two-panel spatial map: PT-feeder vs Municipal, stations as circles
    sized by attracted trips. Shared scale across panels."""
    max_val = float(max(summary['pt_feeder_attracted'].max(),
                        summary['municipal_attracted'].max(), 1.0))

    lakes = None
    if os.path.exists(paths.LAKES_SHP):
        lakes = gpd.read_file(paths.LAKES_SHP).to_crs(catchment_base.CODEBASE_CRS)
        lakes = lakes[lakes.geometry.intersects(boundary)].copy()
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary],
                                    crs=catchment_base.CODEBASE_CRS)

    fig, axes = plt.subplots(1, 2, figsize=(20, 11))
    panels = [
        (axes[0], 'pt_feeder_attracted', 'PT-Feeder', '#1565C0'),
        (axes[1], 'municipal_attracted', 'Municipal', '#E65100'),
    ]
    for ax, col, title, color in panels:
        ax.set_facecolor('#E8E8E8')
        boundary_gdf.plot(ax=ax, color='white', edgecolor='none', zorder=0)
        if lakes is not None and not lakes.empty:
            lakes.plot(ax=ax, color='#A8D8EA', edgecolor='none', zorder=2)
        boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8,
                                    linestyle='--', zorder=4)
        sizes = (summary[col] / max_val) * 800 + 5
        ax.scatter(summary.geometry.x, summary.geometry.y,
                   s=sizes, c=color, alpha=0.65,
                   edgecolors='black', linewidths=0.4, zorder=5)
        bx_min, by_min, bx_max, by_max = boundary.bounds
        pad = 200
        ax.set_xlim(bx_min - pad, bx_max + pad)
        ax.set_ylim(by_min - pad, by_max + pad)
        ax.set_aspect('equal')
        ax.set_title(f'{title}: attracted trips per day (all-day)', fontsize=13)
        ax.set_xlabel('E [m]')
        ax.set_ylabel('N [m]')
        catchment_base._add_map_elements(ax)

    fig.suptitle('Per-station attracted PT trips — PT-Feeder vs Municipal',
                 fontsize=15, y=0.98)
    out_path = os.path.join(OD_COMPARISON_PLOT_DIR,
                            'od_compare_attracted_trips_map.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved → {out_path}")


def _plot_diff_map(summary: gpd.GeoDataFrame, boundary,
                    top_n_label: int = 5) -> None:
    """Signed difference map: PT-feeder − Municipal. Circle size by |diff|,
    colour by signed diff (RdBu_r). Top-N |diff| stations annotated."""
    diffs   = summary['diff_pt_minus_muni']
    abs_max = float(max(abs(diffs.min()), abs(diffs.max()), 1.0))

    lakes = None
    if os.path.exists(paths.LAKES_SHP):
        lakes = gpd.read_file(paths.LAKES_SHP).to_crs(catchment_base.CODEBASE_CRS)
        lakes = lakes[lakes.geometry.intersects(boundary)].copy()
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary],
                                    crs=catchment_base.CODEBASE_CRS)

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_facecolor('#E8E8E8')
    boundary_gdf.plot(ax=ax, color='white', edgecolor='none', zorder=0)
    if lakes is not None and not lakes.empty:
        lakes.plot(ax=ax, color='#A8D8EA', edgecolor='none', zorder=2)
    boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8,
                                linestyle='--', zorder=4)

    sizes = (diffs.abs() / abs_max) * 800 + 5
    sc = ax.scatter(summary.geometry.x, summary.geometry.y,
                    s=sizes, c=diffs, cmap='RdBu_r',
                    vmin=-abs_max, vmax=abs_max, alpha=0.85,
                    edgecolors='black', linewidths=0.4, zorder=5)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label('PT-Feeder − Municipal (trips/day)', fontsize=10)

    # Label top-N |diff| outliers
    top_outliers = summary.reindex(
        diffs.abs().sort_values(ascending=False).index).head(top_n_label)
    for _, row in top_outliers.iterrows():
        ax.annotate(row['readable_name'],
                    xy=(row.geometry.x, row.geometry.y),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='black', alpha=0.85),
                    zorder=6)

    bx_min, by_min, bx_max, by_max = boundary.bounds
    pad = 200
    ax.set_xlim(bx_min - pad, bx_max + pad)
    ax.set_ylim(by_min - pad, by_max + pad)
    ax.set_aspect('equal')
    ax.set_title('Difference: PT-Feeder − Municipal '
                 '(per-station attracted trips, all-day)', fontsize=14)
    ax.set_xlabel('E [m]')
    ax.set_ylabel('N [m]')
    catchment_base._add_map_elements(ax)

    out_path = os.path.join(OD_COMPARISON_PLOT_DIR,
                            'od_compare_attracted_trips_diff.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved → {out_path}")


def _plot_sa_stations_od_map(long_df: pd.DataFrame,
                              sa_stations: gpd.GeoDataFrame,
                              name_lookup: dict,
                              method: str,
                              svc_version: str) -> None:
    """OD pie-chart map for SA stations — mirrors _plot_sa_stations_overview_map.

    For each SA station two pies are drawn above the marker:
      Left  ('Dest'): top-3 stations this station sends trips to   (as origin).
      Right ('Orig'): top-3 stations that send trips to this station (as dest).
    Slice labels show partner station name + share %. Colours are keyed
    deterministically by partner station id. Slices below the visible threshold
    are merged into 'Other'.

    Output: plots/Traffic_Flow/OD/<svc_version>/<Method>/od_sa_stations_od_map.pdf
    """
    if long_df is None or long_df.empty or sa_stations.empty:
        print(f"    OD pie map ({method}): no data — skipped")
        return
    print(f"  Building SA-stations OD pie map ({method}) ...")

    sa_boundary = catchment_allocate._load_sa_boundary()

    # ── Map extent ──────────────────────────────────────────────────────────
    if sa_boundary is not None:
        sa_xmin, sa_ymin, sa_xmax, sa_ymax = sa_boundary.bounds
    else:
        sa_xmin, sa_ymin, sa_xmax, sa_ymax = sa_stations.total_bounds
    sa_w = sa_xmax - sa_xmin
    sa_h = sa_ymax - sa_ymin
    margin = 3000.0
    xmin, xmax = sa_xmin - margin, sa_xmax + margin
    ymin, ymax = sa_ymin - margin, sa_ymax + margin

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_facecolor('#f5f5f5')

    # ── Background: SA area (white fill), lakes, SA boundary outline ─────────
    if sa_boundary is not None:
        sa_gdf = gpd.GeoDataFrame(geometry=[sa_boundary],
                                  crs=catchment_base.CODEBASE_CRS)
        sa_gdf.plot(ax=ax, color='white', edgecolor='none', zorder=1)
        sa_gdf.boundary.plot(ax=ax, color='black', linewidth=1.3,
                             linestyle='--', zorder=7)
    if os.path.exists(paths.LAKES_SHP):
        try:
            from shapely.geometry import box as _box
            lakes = gpd.read_file(paths.LAKES_SHP).to_crs(catchment_base.CODEBASE_CRS)
            ext   = _box(xmin, ymin, xmax, ymax)
            lakes = lakes[lakes.geometry.intersects(ext)]
            if not lakes.empty:
                lakes.plot(ax=ax, facecolor='#A8D8EA', edgecolor='none',
                           alpha=0.8, zorder=3)
        except Exception as _exc:
            print(f"    WARNING: could not load lakes: {_exc}")

    # ── Station markers ──────────────────────────────────────────────────────
    ax.scatter(sa_stations.geometry.x, sa_stations.geometry.y,
               s=22, c='white', edgecolors='black', linewidths=0.9, zorder=8)

    # ── Deterministic colour palette (tab20, keyed by partner station id) ───
    palette = plt.colormaps['tab20'].resampled(20)
    all_ids = sorted({int(v) for v in pd.concat([
        long_df['origin_station_id'], long_df['dest_station_id'],
    ]).dropna().unique()})
    colour_map = {sid: palette(i % 20) for i, sid in enumerate(all_ids)}
    colour_map[-1] = '#888888'

    def _nm(sid):
        return name_lookup.get(str(int(sid)), str(int(sid)))

    def _top_slices(anchor_id, group_col, partner_col, top_n=3):
        sub = long_df[long_df[group_col] == anchor_id]
        if sub.empty:
            return []
        total = float(sub['trips'].sum())
        if total <= 0:
            return []
        top   = sub.nlargest(top_n, 'trips')
        other = total - float(top['trips'].sum())
        slices = []
        for _, r in top.iterrows():
            pid = int(r[partner_col])
            slices.append((_nm(pid), float(r['trips']) / total,
                           colour_map.get(pid, '#888888')))
        if other > 1e-6:
            slices.append(('Other', other / total, '#888888'))
        return slices

    # ── Pie geometry constants (match _plot_sa_stations_overview_map) ────────
    pie_radius = min(sa_w, sa_h) * 0.022
    pie_dx     = pie_radius * 1.35
    pie_dy     = pie_radius * 1.25
    label_dx   = pie_radius * 1.9

    sa_row_by_id = {int(r['id_point']): r for _, r in sa_stations.iterrows()}

    def _closest_angle_in_arc(theta1: float, theta2: float, target: float) -> float:
        a = theta1 % 360
        b = theta2 % 360
        t = target % 360
        arc_size = (theta2 - theta1) % 360
        if arc_size == 0:
            arc_size = 360
        d_at = (t - a) % 360
        if d_at <= arc_size:
            return t
        d_a = min(abs(t - a), 360 - abs(t - a))
        d_b = min(abs(t - b), 360 - abs(t - b))
        return a if d_a < d_b else b

    def _draw_pie(cx, cy, slices, title, label_side, clockwise):
        if not slices:
            return
        target  = 180.0 if label_side == 'left' else 0.0
        lx_text = cx - label_dx if label_side == 'left' else cx + label_dx
        ha      = 'right' if label_side == 'left' else 'left'

        items = []
        start = 90.0
        for name, share, colour in slices:
            delta = share * 360.0
            if clockwise:
                theta1, theta2 = start - delta, start
                start -= delta
            else:
                theta1, theta2 = start, start + delta
                start += delta
            ax.add_patch(Wedge(
                center=(cx, cy), r=pie_radius, theta1=theta1, theta2=theta2,
                facecolor=colour, edgecolor='white', linewidth=0.5, zorder=9,
            ))
            anchor_deg = _closest_angle_in_arc(theta1, theta2, target)
            rad = np.deg2rad(anchor_deg)
            items.append({
                'name':  name,
                'share': share,
                'sx':    cx + pie_radius * np.cos(rad),
                'sy':    cy + pie_radius * np.sin(rad),
                'ly':    cy + pie_radius * 1.25 * np.sin(rad),
            })

        min_gap     = pie_radius * 0.45
        items_sorted = sorted(items, key=lambda d: -d['ly'])
        for i in range(1, len(items_sorted)):
            ceiling = items_sorted[i - 1]['ly'] - min_gap
            if items_sorted[i]['ly'] > ceiling:
                items_sorted[i]['ly'] = ceiling

        for d in items_sorted:
            ax.plot([d['sx'], lx_text], [d['sy'], d['ly']],
                    color='black', linewidth=0.4, zorder=10)
            ax.text(lx_text, d['ly'], f"{d['name']} ({d['share'] * 100:.1f}%)",
                    fontsize=5, ha=ha, va='center', zorder=11,
                    bbox=dict(boxstyle='square,pad=0.15',
                              facecolor='white', edgecolor='none'))

        ax.text(cx, cy - pie_radius * 1.25, title,
                fontsize=5, ha='center', va='top', fontweight='bold', zorder=11,
                bbox=dict(boxstyle='square,pad=0.15',
                          facecolor='white', edgecolor='none'))

    # ── Per-station rendering ────────────────────────────────────────────────
    for sid, row in sa_row_by_id.items():
        x, y = row.geometry.x, row.geometry.y

        dest_slices = _top_slices(sid, 'origin_station_id', 'dest_station_id')
        orig_slices = _top_slices(sid, 'dest_station_id',   'origin_station_id')

        _draw_pie(x - pie_dx, y + pie_dy, dest_slices,
                  title='Dest', label_side='left',  clockwise=True)
        _draw_pie(x + pie_dx, y + pie_dy, orig_slices,
                  title='Orig', label_side='right', clockwise=False)

        total_out = float(long_df[long_df['origin_station_id'] == sid]['trips'].sum())
        total_in  = float(long_df[long_df['dest_station_id']   == sid]['trips'].sum())
        stn_name  = name_lookup.get(str(sid), row.get('stop_name', str(sid)))
        label     = f"{stn_name}\nOut {total_out:,.0f} · In {total_in:,.0f}"
        label_y   = (y + pie_dy + pie_radius * 1.25
                     if (dest_slices or orig_slices) else y + pie_radius * 0.4)
        ax.text(x, label_y, label,
                fontsize=5, fontweight='bold', va='bottom', ha='center', zorder=11,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none'))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel('E [m]')
    ax.set_ylabel('N [m]')
    method_label = 'PT-Feeder' if method == 'pt_feeder' else 'Municipal'
    ax.set_title(f'SA stations — top-3 OD partners ({method_label}): '
                 f'Dest (left pie) / Orig (right pie)')
    catchment_base._add_map_elements(ax)

    out_dir  = paths.get_od_method_plot_dir(svc_version, method)
    out_path = os.path.join(out_dir, 'od_sa_stations_od_map.pdf')
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved → {out_path}")


def _plot_od_diagnostics(pt_long, muni_long, rail_stations, boundary,
                          name_lookup) -> None:
    """Three comparison figures from the all-day station-pair OD matrices.

    Outputs (under plots/Catchment_Area/OD_Comparison/):
      - od_compare_attracted_trips_bar.pdf   — top-20 station bar chart
      - od_compare_attracted_trips_map.pdf   — two-panel spatial map
      - od_compare_attracted_trips_diff.pdf  — diff map with top-5 labels
    Plots are built from the unscaled (τ=1) long-format OD so the comparison
    is window-independent.
    """
    print("\n  Building OD comparison plots ...")
    os.makedirs(OD_COMPARISON_PLOT_DIR, exist_ok=True)

    summary = _build_station_summary(rail_stations, pt_long, muni_long,
                                      name_lookup)
    if summary.empty:
        print("    No station data — skipping plots.")
        return
    if (summary['pt_feeder_attracted'].sum() == 0
        and summary['municipal_attracted'].sum() == 0):
        print("    Both methods have zero attracted trips — skipping plots.")
        return

    _plot_bar_attracted(summary, top_n=20)
    _plot_spatial_attracted(summary, boundary)
    _plot_diff_map(summary, boundary, top_n_label=5)


# ===============================================================================
# STANDALONE ENTRY POINT
# ===============================================================================

if __name__ == '__main__':
    os.chdir(paths.MAIN)
    import settings

    # Discover available service versions (same logic as catchment_allocate)
    _feeder_root = os.path.join(paths.MAIN, paths.FEEDER_LINES_DIR)
    _svc_versions = sorted([
        d for d in os.listdir(_feeder_root)
        if os.path.isdir(os.path.join(_feeder_root, d))
        and os.path.exists(os.path.join(_feeder_root, d,
                                        paths.SERVICES_UNPROJECTED_SUBDIR,
                                        'pt_feeder_stops.gpkg'))
    ]) if os.path.isdir(_feeder_root) else []

    _svc = ''
    if not _svc_versions:
        print("WARNING: no service versions found — _RAIL_BASE will be empty.")
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
            print(f"  Invalid — enter 1–{len(_svc_versions)}.")

    # Method choice — list all three; default to the active settings method.
    _active = settings.CATCHMENT_METHOD.strip().lower().replace('-', '_')
    _active = 'pt_feeder' if _active == 'pt_feeder' else 'municipal'
    _opts = {'1': 'pt_feeder', '2': 'municipal', '3': 'both'}
    _default = '1' if _active == 'pt_feeder' else '2'
    print(f"\nMethod  [active: {_active}]:")
    print("  1) pt_feeder")
    print("  2) municipal")
    print("  3) both")
    _m = input(f"Select method [{_default}]: ").strip() or _default
    _method = _opts.get(_m, _active)

    # Attribution mode (PT-Feeder only)
    _attr = settings.OD_ATTRIBUTION_MODE
    if _method in ('pt_feeder', 'both'):
        _default = '1' if _attr == 'specific' else '2'
        print("\nAttribution:")
        print("  1) specific [origin=pop, dest=FTE]")
        print("  2) symmetric blended")
        _a = input(f"Select attribution [{_default}]: ").strip() or _default
        _attr = 'specific' if _a == '1' else 'blended'

    prepare_all_od_matrices(use_cache=settings.use_cache_stationsOD,
                            svc_version=_svc, method=_method,
                            attribution_mode=_attr)
