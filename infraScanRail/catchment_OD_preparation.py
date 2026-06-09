"""catchment_OD_preparation.py
Last modified: 2026-05-27

Station-pair OD matrix preparation for two methods (PT-feeder, Municipal)
across three time windows (AM peak, off-peak, all-day). Both methods consume
the same scaled communal OD; differ only in cell-to-station attribution.

Public entry point: prepare_all_od_matrices(use_cache: bool) -> None

Pipeline (W3):
  1. Load catchment boundary, rail stations (within boundary), name lookup.
  2. Load the GVM-anchored commune OD for the target year via od_communal (2018
     actual blended toward the symmetrised 2040 forecast; exact at 2018 / 2040,
     population-only beyond 2040).
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
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
from matplotlib.path import Path as _MplPath
import numpy as np
import pandas as pd
from pyogrio import list_layers

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
                            attribution_mode: str = '',
                            both_attributions: bool = False,
                            make_plots: bool = True) -> None:
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
        both_attributions: When True (standalone runs only), PT-Feeder runs BOTH
                     attribution modes so each per-window workbook carries a
                     'Specific' and a 'Blended' sheet. When False (the main_new
                     pipeline default) only `attribution_mode` is run, producing a
                     single-sheet workbook.
        make_plots:  When True (default; standalone), render the SA-stations OD
                     pie map, the corridor Sankeys and the method-comparison
                     diagnostics. main_new passes settings.PLOT_STATION_OD so the
                     plots honour the Phase-4B visualisation toggle. Excel/CSV
                     data outputs are always written regardless.

    Writes (under paths.MAIN):
        data/Traffic_Flow/OD/<svc_network>/<PT_Feeder|Municipal>/od_matrix_stations_{peak,off_peak,full_day}.xlsx  (a sheet per attribution mode)
        data/Traffic_Flow/OD/<svc_network>/<PT_Feeder|Municipal>/od_matrix_stations.xlsx  (chosen-attribution station×station matrix, one sheet per window)
        data/Traffic_Flow/OD/<svc_network>/<PT_Feeder|Municipal>/od_station_top_relations.xlsx
        data/Traffic_Flow/OD/<svc_network>/Gateway/{gateway_zone_assignment.json,gateway_out_of_catchment.csv,gateway_splits.xlsx}
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

    # Resolve the infra version once — used for the authoritative SA station set
    # (rail_stops_sa.gpkg) and for gateway routing.
    infra_version = _resolve_infra_version(svc_version, infra_version)

    # Demand layer (network-agnostic): GVM-anchored commune OD at the target year.
    # od_communal scales the 2018 actual toward the symmetrised 2040 forecast (exact
    # at 2018 / 2040), replacing the former pop-only geometric-mean scaling. The
    # spatial station attribution (count-blend) below is unchanged. The target year
    # matches the catchment pop/empl grid (settings.start_year_scenario) so demand
    # and attribution stay on the same year.
    target_year = settings.start_year_scenario
    communal_od = od_communal(year=target_year)

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
    # `attr_mode` selects the attribution feeding the combined matrix, top-relations,
    # pie map and comparison plots. With both_attributions=True (standalone) PT-Feeder
    # additionally runs the other mode so each per-window workbook carries a
    # 'Specific' and a 'Blended' sheet; from main_new only `attr_mode` is run.
    chosen_label = 'Blended' if attr_mode == 'blended' else 'Specific'
    pt_long = muni_long = None
    branch_attr_longs = {}   # branch -> {sheet_label -> long_df}
    if 'pt_feeder' in methods:
        if both_attributions:
            pt_specific, ow_specific = _run_pt_feeder_branch(branch_od, gw_ids, 'specific')
            pt_blended,  ow_blended  = _run_pt_feeder_branch(branch_od, gw_ids, 'blended')
            branch_attr_longs['pt_feeder'] = {'Specific': pt_specific, 'Blended': pt_blended}
            if attr_mode == 'blended':
                pt_long, pt_orig_weights = pt_blended, ow_blended
            else:
                pt_long, pt_orig_weights = pt_specific, ow_specific
        else:
            pt_long, pt_orig_weights = _run_pt_feeder_branch(branch_od, gw_ids, attr_mode)
            branch_attr_longs['pt_feeder'] = {chosen_label: pt_long}
        _diagnose_conservation(branch_od, pt_long, pt_orig_weights, 'PT-Feeder')
    if 'municipal' in methods:
        muni_long = _run_municipal_branch(branch_od, gw_ids)
        branch_attr_longs['municipal'] = {'Municipal': muni_long}
        _diagnose_conservation(branch_od, muni_long, None, 'Municipal')

    # Fill any name_lookup gaps from the allocation breakdown (peak-only stations
    # absent from the all-day rail-stops load), then verify full coverage.
    for _m in methods:
        _extend_name_lookup_from_breakdown(name_lookup, _m)
    _check_name_coverage(pt_long,   name_lookup, 'PT-Feeder')
    _check_name_coverage(muni_long, name_lookup, 'Municipal')

    # --- Emit per-window workbooks (a sheet per attribution mode) ---
    windows = [(cp.TAU_PEAK_SHARE,     'peak'),
               (cp.TAU_OFFPEAK_SHARE,  'off_peak'),
               (cp.TAU_FULL_DAY_SHARE, 'full_day')]
    branch_longs = {'pt_feeder': pt_long, 'municipal': muni_long}  # chosen attribution
    print("\n  Writing per-window OD workbooks ...")
    for branch in methods:
        attr_longs = branch_attr_longs.get(branch)
        if not attr_longs:
            continue
        for tau, suffix in windows:
            out_path = paths.get_station_od_window_xlsx(svc_version, branch, suffix)
            if use_cache and Path(out_path).exists():
                print(f"    cached: {out_path}")
            else:
                _write_window_xlsx(attr_longs, tau, name_lookup, out_path,
                                   label=f'{branch} {suffix}')

    # --- Persist reloadable baseline artifacts (Phase 6B subset reaggregation) ---
    # Additive only: the whole-day long-format OD, the PT-feeder attribution weights
    # and the gateway-expanded communal OD are written so reaggregate_subset can be
    # driven from disk without re-running this function. The wide xlsx outputs above
    # are unchanged. Weights are reproduced via the attribution_weights() seam (same
    # producer as the live branch) so disk and live stay in lock-step.
    print("\n  Persisting long-format OD + attribution weights ...")
    for branch in methods:
        attr_longs = branch_attr_longs.get(branch)
        if not attr_longs:
            continue
        for label, long_df in attr_longs.items():
            attribution = label.lower()
            long_path = paths.get_station_od_long_csv(svc_version, branch, attribution)
            Path(long_path).parent.mkdir(parents=True, exist_ok=True)
            if use_cache and Path(long_path).exists():
                print(f"    cached: {long_path}")
            else:
                long_df.to_csv(long_path, index=False, encoding='utf-8-sig')
            if branch == 'pt_feeder':
                ow, dw = attribution_weights('pt_feeder', attribution, gw_ids)
                ow.to_csv(paths.get_attribution_weights_csv(
                    svc_version, branch, attribution, 'orig'),
                    index=False, encoding='utf-8-sig')
                dw.to_csv(paths.get_attribution_weights_csv(
                    svc_version, branch, attribution, 'dest'),
                    index=False, encoding='utf-8-sig')
    communal_path = paths.get_communal_od_csv(svc_version)
    Path(communal_path).parent.mkdir(parents=True, exist_ok=True)
    if use_cache and Path(communal_path).exists():
        print(f"    cached: {communal_path}")
    else:
        branch_od.to_csv(communal_path, index=False, encoding='utf-8-sig')
    print(f"    long OD + weights + communal OD under "
          f"{paths.get_od_version_dir(svc_version)}")

    # --- Top-5 origins/destinations Excel export + per-method OD pie map ---
    sa_stations_gdf = _load_sa_stations(rail_stations, svc_version, infra_version)
    sa_ids = set(pd.to_numeric(sa_stations_gdf['id_point'], errors='coerce')
                 .dropna().astype(int).tolist())
    for branch in methods:
        long_df = branch_longs[branch]
        if long_df is not None:
            _export_od_matrix_excel(long_df, windows, name_lookup,
                                    svc_version, branch)
            _export_top_relations_excel(long_df, sa_ids, name_lookup, branch,
                                        svc_version, windows)
            if make_plots:
                _plot_sa_stations_od_map(long_df, sa_stations_gdf, name_lookup,
                                         branch, svc_version, attr_mode)
                build_corridor_sankeys(long_df, name_lookup, svc_version, branch,
                                       attr_mode)
                _plot_sa_relation_heatmaps(long_df, sa_ids, name_lookup,
                                           svc_version, branch)

    # --- Method comparison + diagnostic plots (only when both methods present) ---
    if pt_long is not None and muni_long is not None:
        _export_method_comparison_excel(pt_long, muni_long, sa_ids,
                                        name_lookup, svc_version)
        if make_plots:
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


# ===============================================================================
# NEW SHARED HELPERS (W3)
# ===============================================================================

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


def _build_boundary_station_index(boundary_ids: list, infra_version: str,
                                  verbose: bool = True) -> dict:
    """Map each boundary station id -> (name, shapely point).

    Reads the infrastructure nodes GeoPackage
    (data/Infrastructure/<infra_version>/nodes.gpkg) — the same source the
    services projection uses for boundary-station names — so every boundary
    station resolves to its name and coordinates, including the ones outside the
    catchment buffer (rail stops only cover in-catchment stations).

    verbose=False suppresses the "absent from infra nodes" note (used when
    indexing the full served-station set, where many stops are legitimately not
    infra nodes).
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
    if missing and verbose:
        print(f"    Note: {len(missing)} boundary station(s) absent from infra "
              f"nodes: {sorted(missing)}")
    return idx


def _build_gateway_station_rows(svc_network: str, infra_version: str = '',
                                extra_ids=None) -> gpd.GeoDataFrame:
    """rail_stations-schema rows for the gateway (boundary) stations plus any
    extra_ids (e.g. convergence stations), so the routing graph can host
    boardable gateway portals keyed by integer id.

    Gateways lie outside the catchment, so catchment_allocate._load_rail_stations
    omits them; this fills the gap from the infrastructure nodes.gpkg. Columns
    match _load_rail_stations: stop_id, stop_name, diva_nr, id_point, mode,
    geometry (stop_id == id_point == str(node number)).

    Args:
        svc_network:   Service network folder name.
        infra_version: Infra subfolder holding nodes.gpkg / boundary_stations.json
                       (resolved when empty).
        extra_ids:     Optional iterable of additional node ids to include
                       (convergence stations are served nodes outside the
                       catchment, otherwise absent from the routing graph).

    Returns:
        GeoDataFrame (EPSG:2056); empty (with the schema columns) when no
        gateways resolve.
    """
    cols = ['stop_id', 'stop_name', 'diva_nr', 'id_point', 'mode', 'geometry']
    infra = _resolve_infra_version(svc_network, infra_version)
    ids = [int(x) for x in _load_boundary_stations(svc_network, infra)]
    if extra_ids:
        ids += [int(x) for x in extra_ids]
    if not ids:
        return gpd.GeoDataFrame(columns=cols, geometry='geometry',
                                crs=_CODEBASE_CRS)
    bs_index = _build_boundary_station_index(sorted(set(ids)), infra)
    rows = []
    for sid, (nm, geom) in bs_index.items():
        if geom is None:
            continue
        rows.append({'stop_id': str(int(sid)), 'stop_name': str(nm),
                     'diva_nr': None, 'id_point': str(int(sid)),
                     'mode': 'gateway', 'geometry': geom})
    if not rows:
        return gpd.GeoDataFrame(columns=cols, geometry='geometry',
                                crs=_CODEBASE_CRS)
    return gpd.GeoDataFrame(rows, geometry='geometry', crs=_CODEBASE_CRS)


def _normalise_assignment(raw: dict) -> dict:
    """Normalise a loaded zone-assignment dict to dict[int code -> list[int]].

    Accepts the legacy single-station ({code: id}) and multi-station
    ({code: [id, ...]}) formats, plus the readable format where each station is a
    {"id": int, "name": str} object ({code: [{"id": .., "name": ..}, ...]}).
    """
    def _sid(x):
        return int(x['id']) if isinstance(x, dict) else int(x)
    out = {}
    for k, v in raw.items():
        code = int(k)
        if isinstance(v, (list, tuple)):
            out[code] = [_sid(x) for x in v]
        else:
            out[code] = [_sid(v)]
    return out


def _save_zone_assignment(mapping: dict, bs_index: dict, json_path: str) -> None:
    """Write the zone→gateway assignment in the readable format:
    {code: [{"id": station_number, "name": station_name}, ...]}.

    Carries both the station number and name for readability; round-trips through
    _normalise_assignment (which tolerates this and the legacy id-only formats).
    """
    def _entry(sid):
        name = bs_index.get(int(sid), (str(int(sid)), None))[0]
        return {'id': int(sid), 'name': name}
    payload = {str(k): [_entry(s) for s in v] for k, v in mapping.items()}
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _assign_external_zones(zone_codes, boundary_ids, bs_index, zone_names,
                           json_path) -> dict:
    """Assign each external GVM zone (code > 9999) to one or more gateway stations.

    Multiple stations per zone are allowed; the zone's demand is later split
    across them by service volume. Persisted to json_path in the readable format
    {code: [{"id": station_number, "name": station_name}, ...]} (legacy id-only
    files are auto-upgraded on load). A complete saved assignment is loaded
    silently (interactive or not); an incomplete one prompts for the missing zones
    (interactive) or raises (automated). Delete the json to force a full rebuild.

    Returns dict[int zone_code -> list[int gateway_station_id]] (skipped absent).
    """
    zone_codes = sorted(int(z) for z in zone_codes)
    saved = None
    legacy_format = False
    if os.path.exists(json_path):
        with open(json_path, encoding='utf-8') as f:
            raw = json.load(f)
        saved = _normalise_assignment(raw)
        legacy_format = bool(raw) and not all(
            isinstance(v, list) and all(isinstance(e, dict) for e in v)
            for v in raw.values())

    if saved is not None:
        complete = all(z in saved for z in zone_codes)
        if complete:
            # Upgrade a legacy id-only file to the readable name+number format
            # in place (no re-prompting), then use it silently. Delete the json
            # to force a full rebuild.
            if legacy_format:
                _save_zone_assignment(saved, bs_index, json_path)
                print(f"  Upgraded gateway zone assignment to readable format")
            print(f"  Loaded gateway zone assignment ({len(saved)} zones)")
            return {z: saved[z] for z in zone_codes}
        if not _INTERACTIVE_MODE:
            missing = [z for z in zone_codes if z not in saved]
            raise FileNotFoundError(
                f"Gateway zone assignment at {json_path} is missing zones "
                f"{missing}. Run catchment_OD_preparation standalone to "
                f"complete the assignment.")
        print(f"  Existing gateway zone assignment incomplete — prompting for "
              f"missing zones only.")

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

    _save_zone_assignment(mapping, bs_index, json_path)
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


def _load_line_freq_per_h_window(svc_network: str, infra_version: str) -> dict:
    """Whole-day per-variant freq_per_h_window from the full-day rail_lines.gpkg.

    freq_per_h_window = total_dep / (GK_WINDOW_MIN/60). Keyed as strings on
    (route_id, direction_id, variant_rank) to match the segment GTFS_ID /
    direction_id / variant_rank columns used at the gateways.
    """
    lines_path = os.path.join(paths.MAIN, paths.RAIL_LINES_DIR, svc_network,
                              infra_version, 'rail_lines.gpkg')
    if not os.path.exists(lines_path):
        return {}
    frames = []
    for lyr in list_layers(lines_path)[:, 0].tolist():
        g = gpd.read_file(lines_path, layer=lyr)
        if 'geometry' in g.columns:
            g = pd.DataFrame(g.drop(columns='geometry'))
        frames.append(g)
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    df['total_dep'] = pd.to_numeric(df.get('total_dep'), errors='coerce')
    df = df.dropna(subset=['total_dep'])
    fpw = df['total_dep'] / (catchment_allocate.GK_WINDOW_MIN / 60.0)
    return {
        (str(rid), str(did), str(vr)): float(f)
        for rid, did, vr, f in zip(
            df['route_id'], df['direction_id'], df['variant_rank'], fpw)
    }


def _freq_by_route_at_gateways(seg_path: str, layer: str, gateway_ids,
                               freq_lookup: dict) -> dict:
    """Summed whole-day frequency (freq_per_h_window) of services crossing the
    canton boundary at each gateway station, for one route_type layer.

    Matching uses boundary_entry_node / boundary_exit_node — the node where a
    service enters/leaves the catchment — NOT the scheduled stop columns:
    long-distance / inter-regional trains traverse boundary stations without
    stopping, so they never appear as from/to stops. boundary_entry/exit_node is
    set consistently across all route types, giving a comparable cross-border
    supply measure. Each service variant is counted once per direction; its
    frequency is the whole-day freq_per_h_window joined from the lines table on
    (GTFS_ID, direction_id, variant_rank).
    """
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
    seg['freq'] = [
        freq_lookup.get((str(r), str(d), str(v)), 0.0)
        for r, d, v in zip(seg.get('GTFS_ID'), seg.get('direction_id'),
                           seg.get('variant_rank'))
    ]

    want = {int(g) for g in gateway_ids}
    touch = seg[seg['boundary_entry_node'].isin(want)
                | seg['boundary_exit_node'].isin(want)]
    keys = [k for k in ('GTFS_ID', 'direction_id', 'variant_rank')
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


def _compute_gateway_splits(gateway_ids, svc_network, infra_version,
                            bs_index) -> pd.DataFrame:
    """Local (S-Bahn + RE) vs long-distance (LD + IR) service-supply split per
    gateway.

    local_share = f(sbahn + regional) / (f(sbahn + regional) + f(long_distance +
    inter_regional)), where f is the whole-day freq_per_h_window summed over the
    variants crossing the boundary at the gateway. Gateways with no crossing
    service default to local_share=1.0. The split is metadata for downstream
    routing — the gateway carries full demand in the OD matrix itself. Returned
    for the caller to write as the 'Gateway_Split' sheet of the gateway routing
    workbook.
    """
    gateway_ids = sorted({int(g) for g in gateway_ids})
    if not gateway_ids:
        return pd.DataFrame()
    seg_path = _gateway_segments_path(svc_network, infra_version)
    if not os.path.exists(seg_path):
        print(f"    Gateway split: rail_segments.gpkg missing at {seg_path}")
        return pd.DataFrame()
    freq_lookup = _load_line_freq_per_h_window(svc_network, infra_version)

    local = {}
    for lyr in _LOCAL_LAYERS:
        for gid, f in _freq_by_route_at_gateways(
                seg_path, lyr, gateway_ids, freq_lookup).items():
            local[gid] = local.get(gid, 0.0) + f
    ldirt = {}
    for lyr in _LDIRT_LAYERS:
        for gid, f in _freq_by_route_at_gateways(
                seg_path, lyr, gateway_ids, freq_lookup).items():
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
            print(f"    Gateway {gid_name} ({gid}): no crossing service "
                  f"— defaulting local share=1.0")
        rows.append({
            'gateway_station_id': gid,
            'station_name': bs_index.get(gid, (str(gid), None))[0],
            'local_freq': round(lf_local, 2), 'longdist_freq': round(lf_ld, 2),
            'local_share': round(ls, 4), 'longdist_share': round(1.0 - ls, 4),
        })
    df = pd.DataFrame(rows)
    print(f"    Gateway service-supply splits computed ({len(df)} gateways, "
          f"whole-day freq_per_h_window)")
    return df


def _load_line_name_lookup(svc_network: str, infra_version: str) -> dict:
    """route_id (str) -> line_short_name from the full-day rail_lines.gpkg.

    Readability only for the connection table; falls back to the route_id when a
    name is missing.
    """
    lines_path = os.path.join(paths.MAIN, paths.RAIL_LINES_DIR, svc_network,
                              infra_version, 'rail_lines.gpkg')
    if not os.path.exists(lines_path):
        return {}
    out = {}
    for lyr in list_layers(lines_path)[:, 0].tolist():
        g = gpd.read_file(lines_path, layer=lyr)
        if 'route_id' not in g.columns or 'line_short_name' not in g.columns:
            continue
        for rid, nm in zip(g['route_id'], g['line_short_name']):
            if pd.notna(nm) and str(nm).strip():
                out.setdefault(str(rid), str(nm).strip())
    return out


def _build_gateway_connections(gateway_ids, svc_network: str, infra_version: str,
                               splits_df: pd.DataFrame, freq_lookup: dict,
                               bs_index: dict) -> pd.DataFrame:
    """Per-gateway service-connection table consumed by passenger routing.

    One row per (gateway_station_id, direction_role, route_id, direction_id,
    variant_rank): the crossing service's type (local/longdist from the segment
    layer), whether it stops at the gateway, its whole-day freq_per_h_window, and
    the nested boarding weight. Weights sum to 1.0 per (gateway, direction_role).

    The boarding weight is the **nested key**: the service-type total comes from
    the gateway's supply share (`local_share`/`longdist_share`), then *within* a
    type demand is distributed by per-service `freq_per_h_window`. Type shares are
    renormalised over the types actually present in that (gateway, role) so the
    weights always sum to 1.0. (Numerically equal to a flat per-service frequency
    split while both levels are frequency-derived; the structure is kept so the
    type totals can be re-sourced/calibrated later.)

    direction_role: 'inbound' (boundary_entry_node == gateway; used when the
    gateway is an OD origin) / 'outbound' (boundary_exit_node == gateway; gateway
    as an OD destination).

    Only boundary gateways appear here. Convergence stations (served hubs used as
    substitutes for dead boundary stations) route via normal graph portals, not
    this table — their crossing services are not reliably flagged at the
    convergence node (e.g. Zug carries no boundary crossing), so a connection-
    table split would misroute them.

    Returns an empty DataFrame when no projected segments / gateways resolve.
    """
    gateway_ids = sorted({int(g) for g in gateway_ids})
    seg_path = _gateway_segments_path(svc_network, infra_version)
    if not gateway_ids or not os.path.exists(seg_path):
        return pd.DataFrame()

    type_share = {int(r.gateway_station_id):
                  {'local': float(r.local_share), 'longdist': float(r.longdist_share)}
                  for r in splits_df.itertuples(index=False)} if not splits_df.empty else {}
    layer_type = {**{l: 'local' for l in _LOCAL_LAYERS},
                  **{l: 'longdist' for l in _LDIRT_LAYERS}}
    name_lookup = _load_line_name_lookup(svc_network, infra_version)

    try:
        available = list_layers(seg_path)[:, 0].tolist()
    except Exception:
        available = []

    want = set(gateway_ids)
    recs = []
    for layer, stype in layer_type.items():
        if layer not in available:
            continue
        seg = gpd.read_file(seg_path, layer=layer)
        if 'geometry' in seg.columns:
            seg = pd.DataFrame(seg.drop(columns='geometry'))
        for c in ('boundary_entry_node', 'boundary_exit_node',
                  'from_stop_nr', 'to_stop_nr'):
            seg[c] = pd.to_numeric(seg.get(c), errors='coerce')
        for gid in want:
            for role, col in (('inbound', 'boundary_entry_node'),
                              ('outbound', 'boundary_exit_node')):
                sub = seg[seg[col] == gid]
                if sub.empty:
                    continue
                for (rid, did, vr), _g in sub.groupby(
                        ['GTFS_ID', 'direction_id', 'variant_rank'], sort=False):
                    freq = float(freq_lookup.get((str(rid), str(did), str(vr)), 0.0))
                    vrows = seg[(seg['GTFS_ID'] == rid)
                                & (seg['direction_id'] == did)
                                & (seg['variant_rank'] == vr)]
                    stops = bool((vrows['from_stop_nr'] == gid).any()
                                 or (vrows['to_stop_nr'] == gid).any())
                    recs.append({
                        'gateway_station_id': gid,
                        'station_name': bs_index.get(gid, (str(gid), None))[0],
                        'direction_role': role,
                        'route_id': str(rid), 'direction_id': str(did),
                        'variant_rank': vr,
                        'line_short_name': name_lookup.get(str(rid), str(rid)),
                        'service_type': stype,
                        'stops_at_gateway': stops,
                        'freq_per_h_window': round(freq, 4),
                    })

    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df = df.drop_duplicates(['gateway_station_id', 'direction_role',
                             'route_id', 'direction_id', 'variant_rank'])
    df = _apply_nested_boarding_weights(df, type_share)
    return df.sort_values(['gateway_station_id', 'direction_role',
                           'service_type', 'freq_per_h_window'],
                          ascending=[True, True, True, False]).reset_index(drop=True)


def _apply_nested_boarding_weights(df: pd.DataFrame, type_share: dict) -> pd.DataFrame:
    """Add boarding_weight per (gateway, direction_role): renormalised type share
    × within-type frequency share. Weights sum to 1.0 per group (or 0 when the
    group has no positive frequency)."""
    out = []
    for (gid, _role), grp in df.groupby(['gateway_station_id', 'direction_role'],
                                        sort=False):
        grp = grp.copy()
        ts = type_share.get(int(gid), {})
        tfreq = grp.groupby('service_type')['freq_per_h_window'].sum()
        present = [t for t in tfreq.index if tfreq[t] > 0]
        denom = sum(ts.get(t, 0.0) for t in present)
        if present and denom > 0:
            eff = {t: ts.get(t, 0.0) / denom for t in present}
        elif present:                       # no supply share available -> equal types
            eff = {t: 1.0 / len(present) for t in present}
        else:
            eff = {}
        w = []
        for _, row in grp.iterrows():
            t, f = row['service_type'], row['freq_per_h_window']
            intra = (f / tfreq[t]) if tfreq.get(t, 0.0) > 0 else 0.0
            w.append(round(eff.get(t, 0.0) * intra, 6))
        grp['boarding_weight'] = w
        out.append(grp)
    return pd.concat(out, ignore_index=True)


def _build_region_gateways_table(gw_weights: dict, zone_names: dict,
                                 bs_index: dict) -> pd.DataFrame:
    """Wide region→gateway share table for the workbook's 'Region_Gateways' sheet.

    One row per routed region (gw_weights key): the region label followed by
    `Gateway i` / `Share gateway i` columns for each assigned gateway, ordered by
    descending share. Single-gateway regions list one gateway at 100 %. Region
    labelled by external-zone name where known, else 'BFS <code>' for out-of-
    catchment communes (code ≤ 9999) or the bare zone code. Multi-gateway regions
    are listed first.
    """
    if not gw_weights:
        return pd.DataFrame()

    def _region_label(code):
        if code in zone_names:
            return zone_names[code]
        return f"BFS {code}" if code <= 9999 else str(code)

    def _gw_name(sid):
        return bs_index.get(int(sid), (str(int(sid)), None))[0]

    rows, max_gw = [], 0
    for code, pairs in gw_weights.items():
        ordered = sorted(pairs, key=lambda sv: -sv[1])
        row = {'Region': _region_label(int(code))}
        for i, (sid, share) in enumerate(ordered, 1):
            row[f'Gateway {i}']       = _gw_name(sid)
            row[f'Share gateway {i}'] = round(float(share), 4)
        rows.append((int(code), len(ordered), row))
        max_gw = max(max_gw, len(ordered))

    rows.sort(key=lambda t: (-t[1], t[0]))   # multi-gateway first, then by code
    cols = ['Region']
    for i in range(1, max_gw + 1):
        cols += [f'Gateway {i}', f'Share gateway {i}']
    return pd.DataFrame([r for _, _, r in rows]).reindex(columns=cols)


def _write_gateway_splits_xlsx(splits: pd.DataFrame, gw_weights: dict,
                               zone_names: dict, bs_index: dict,
                               out_path: str) -> None:
    """Write the gateway routing workbook (replaces the old gateway_splits.csv).

    Sheet 'Gateway_Split'   — per-gateway local vs long-distance service split.
    Sheet 'Region_Gateways' — every routed region with its gateway station(s) and
        demand shares (see _build_region_gateways_table).
    """
    region_tbl = _build_region_gateways_table(gw_weights, zone_names, bs_index)
    has_splits = splits is not None and not splits.empty
    if not has_splits and region_tbl.empty:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        (splits if has_splits else pd.DataFrame()).to_excel(
            writer, sheet_name='Gateway_Split', index=False)
        region_tbl.to_excel(writer, sheet_name='Region_Gateways', index=False)
    print(f"    Gateway routing workbook "
          f"({len(splits) if has_splits else 0} gateways, "
          f"{len(region_tbl)} regions) → {out_path}")


def _write_gateway_connections_xlsx(connections: pd.DataFrame,
                                    out_path: str) -> None:
    """Write the gateway service-connection table (single 'Connections' sheet)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = connections if connections is not None and not connections.empty \
        else pd.DataFrame()
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Connections', index=False)
    n_gw = df['gateway_station_id'].nunique() if not df.empty else 0
    print(f"    Gateway service-connection table ({len(df)} variant-rows, "
          f"{n_gw} gateways) → {out_path}")


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


def _served_station_ids(svc_network: str, infra_version: str) -> set:
    """Station numbers that appear as a from/to stop in the projected segments
    (i.e. boardable in the service network) — used to validate convergence
    targets."""
    seg_path = _gateway_segments_path(svc_network, infra_version)
    if not os.path.exists(seg_path):
        return set()
    served = set()
    for lyr in list_layers(seg_path)[:, 0].tolist():
        g = gpd.read_file(seg_path, layer=lyr)
        for col in ('from_stop_nr', 'to_stop_nr'):
            served |= set(pd.to_numeric(g.get(col), errors='coerce')
                          .dropna().astype(int).tolist())
    return served


def _served_station_index(svc_network: str, infra_version: str) -> dict:
    """dict[int station id -> name] for served stations (boardable in the service
    network), so convergence targets can be entered and stored by name as well as
    by number."""
    served = _served_station_ids(svc_network, infra_version)
    if not served:
        return {}
    return {i: nm for i, (nm, _g)
            in _build_boundary_station_index(sorted(served), infra_version,
                                             verbose=False).items()}


def _resolve_station_token(text, served_ids: set, id_to_name: dict,
                           allow_substring: bool = True) -> tuple:
    """Resolve a typed token (station number or name) to a served station id.

    Returns (id, None) on success, else (None, message). Matching: a numeric token
    is taken as the id; otherwise a case-insensitive exact name match, then
    (optionally) a unique case-insensitive substring match.
    """
    t = str(text).strip()
    if not t:
        return None, "empty"
    if t.isdigit():
        return (int(t), None) if int(t) in served_ids \
            else (None, f"id {t} is not a served station")
    tl = t.lower()
    exact = [i for i, n in id_to_name.items()
             if i in served_ids and str(n).strip().lower() == tl]
    if len(exact) == 1:
        return exact[0], None
    if len(exact) > 1:
        return None, "ambiguous name: " + ", ".join(
            f"{id_to_name[i]} ({i})" for i in exact)
    if allow_substring:
        subs = [i for i, n in id_to_name.items()
                if i in served_ids and tl in str(n).strip().lower()]
        if len(subs) == 1:
            return subs[0], None
        if len(subs) > 1:
            return None, "matches several: " + ", ".join(
                f"{id_to_name[i]} ({i})" for i in subs[:10])
    return None, "no served station matches"


def _load_convergence_map(gateway_dir: str, id_to_name: dict = None) -> dict:
    """Load the optional convergence-map override
    (gateway_convergence_map.json): {"<zone_or_gateway_id>": <served_station>}.

    A key matching an assignment zone redirects that whole zone to the convergence
    station; any other key is treated as a gateway station id and replaced wherever
    it appears. The value may be a bare id, a readable {"id":.., "name":..} object,
    or a station NAME (a bare string or {"name":..}); names are resolved against
    served stations when `id_to_name` is supplied (exact case-insensitive match).
    Absent file -> {}.
    """
    path = os.path.join(gateway_dir, 'gateway_convergence_map.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    served_ids = set(id_to_name) if id_to_name else set()
    out = {}
    for k, v in raw.items():
        val  = v.get('id') if isinstance(v, dict) else v
        name = v.get('name') if isinstance(v, dict) else None
        sid = None
        if val is not None:
            try:
                sid = int(val)
            except (ValueError, TypeError):
                name = name or val          # value was a name string, not an id
        if sid is None and name and id_to_name:
            sid, msg = _resolve_station_token(name, served_ids, id_to_name,
                                              allow_substring=False)
            if sid is None:
                print(f"  Convergence map: cannot resolve '{name}' for key {k} "
                      f"({msg}); skipped.")
                continue
        if sid is None:
            continue
        try:
            out[int(k)] = sid
        except (ValueError, TypeError):
            continue
    return out


def _save_convergence_map(conv_map: dict, bs_index: dict, gateway_dir: str) -> None:
    """Persist the convergence map in the readable {key: {"id":.., "name":..}} form."""
    path = os.path.join(gateway_dir, 'gateway_convergence_map.json')
    payload = {str(k): {'id': int(v),
                        'name': bs_index.get(int(v), (str(int(v)), None))[0]}
               for k, v in conv_map.items()}
    os.makedirs(gateway_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _apply_convergence_redirect(assignment: dict, conv_map: dict) -> dict:
    """Redirect `assignment` per `conv_map`. Keys matching an assignment zone
    redirect that whole zone to [conv_id]; other keys are gateway station ids and
    are replaced wherever they occur (order-preserving dedupe). Mutates and returns
    `assignment`."""
    if not conv_map:
        return assignment
    for key, conv in conv_map.items():
        if key in assignment:
            assignment[key] = [int(conv)]
    gw_redirect = {k: int(v) for k, v in conv_map.items() if k not in assignment}
    if gw_redirect:
        for z, stns in assignment.items():
            seen, out = set(), []
            for s in (gw_redirect.get(int(s), int(s)) for s in stns):
                if s not in seen:
                    seen.add(s)
                    out.append(s)
            assignment[z] = out
    return assignment


def _guard_dead_gateways(assignment: dict, dead_set: set, served_ids: set,
                         id_to_name: dict, bs_index: dict, conv_map: dict,
                         gateway_dir: str) -> dict:
    """Ensure no zone routes only through dead (no-service) gateways. Zones whose
    entire assignment is dead get a convergence station: prompt interactively, else
    raise. The prompt accepts a station NAME or number; the resolved name+number is
    persisted. Mutates+returns assignment.
    """
    problem = {z: stns for z, stns in assignment.items()
               if stns and all(int(s) in dead_set for s in stns)}
    if not problem:
        return assignment
    conv_json = os.path.join(gateway_dir, 'gateway_convergence_map.json')
    if not _INTERACTIVE_MODE:
        raise FileNotFoundError(
            f"Gateway zones {sorted(problem)} route only through dead (no-service) "
            f"boundary stations: "
            f"{ {z: [bs_index.get(int(s),(str(s),None))[0] for s in stns] for z,stns in problem.items()} }. "
            f"Add a convergence (served upstream) station for each to {conv_json}, "
            f"or run catchment_OD_preparation standalone to assign one.")
    print(f"\n  {len(problem)} zone(s) route only through dead gateways — assign a "
          f"convergence (served upstream) station for each (enter a station name "
          f"or number):")
    changed = False
    for z, stns in problem.items():
        dead_names = ', '.join(bs_index.get(int(s), (str(s), None))[0] for s in stns)
        while True:
            raw = input(f"    Zone {z} (dead: {dead_names}) -> served station "
                        f"name or number (or 's' to skip/drop demand): ").strip()
            if raw.lower() == 's':
                print(f"      zone {z} left dead (demand will drop)")
                break
            sid, msg = _resolve_station_token(raw, served_ids, id_to_name)
            if sid is None:
                print(f"      {msg} — enter a served station name or number, or 's'.")
                continue
            nm = id_to_name.get(sid, str(sid))
            assignment[z] = [sid]
            conv_map[z] = sid
            bs_index[sid] = (nm, bs_index.get(sid, (None, None))[1])
            changed = True
            print(f"      zone {z} -> {nm} ({sid})")
            break
    if changed:
        _save_convergence_map(conv_map, bs_index, gateway_dir)
        print(f"  Saved convergence map → {conv_json}")
    return assignment


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
    splits_xlsx = os.path.join(gateway_dir, 'gateway_splits.xlsx')

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

    # Convergence overrides: redirect specific zones / dead gateways to an upstream
    # served station (e.g. Schaffhausen for Thayngen+Neunkirch, Zug for Zug Casino).
    # Loaded from a separate map so the boundary assignment stays untouched.
    boundary_set = {int(b) for b in boundary_ids}
    served_names = _served_station_index(svc_network, infra)   # id -> name
    served_ids   = set(served_names)
    conv_map     = _load_convergence_map(gateway_dir, served_names)
    if conv_map:
        bad = sorted({int(v) for v in conv_map.values()
                      if served_ids and int(v) not in served_ids})
        if bad:
            raise ValueError(
                f"Convergence target(s) {bad} are not served stations (no stop in "
                f"the service network); pick an upstream served station.")
        assignment = _apply_convergence_redirect(assignment, conv_map)

    # Dead-gateway guard: boundary gateways with no crossing service can't board
    # demand. Detect them, then ensure no zone routes only through dead gateways
    # (prompt for a convergence station interactively, else raise).
    boundary_used = {int(s) for stns in assignment.values() for s in stns} & boundary_set
    splits = _compute_gateway_splits(boundary_used, svc_network, infra, bs_index)
    dead_set = {int(r.gateway_station_id) for r in splits.itertuples(index=False)
                if (float(r.local_freq) + float(r.longdist_freq)) == 0.0}
    assignment = _guard_dead_gateways(assignment, dead_set, served_ids,
                                      served_names, bs_index, conv_map, gateway_dir)

    # Final gateway set (post redirect + guard). Convergence stations are served
    # nodes outside the catchment; pull their names/geometry into bs_index so the
    # name lookup and the OD matrix label them, and split off the boundary subset
    # (only boundary gateways get a connection-table entry; convergence stations
    # route via normal graph portals).
    used_gateways = {int(s) for stns in assignment.values() for s in stns}
    conv_ids      = sorted(used_gateways - boundary_set)
    if conv_ids:
        bs_index.update(_build_boundary_station_index(conv_ids, infra))
    boundary_used = used_gateways & boundary_set
    splits = _compute_gateway_splits(boundary_used, svc_network, infra, bs_index)

    volumes = {}
    if not splits.empty:
        volumes = {int(r.gateway_station_id): float(r.local_freq) + float(r.longdist_freq)
                   for r in splits.itertuples(index=False)}
    gw_weights = _build_gateway_weights(assignment, volumes)

    _write_gateway_splits_xlsx(splits, gw_weights, zone_names, bs_index, splits_xlsx)

    # Per-service connection table: which crossing services (stopping + passing)
    # each boundary gateway feeds, with the nested type-then-frequency boarding
    # weights the router uses to inject + split demand. Written for routing.
    freq_lookup = _load_line_freq_per_h_window(svc_network, infra)
    connections = _build_gateway_connections(
        boundary_used, svc_network, infra, splits, freq_lookup, bs_index)
    conn_xlsx = paths.get_gateway_connections_xlsx(svc_network)
    _write_gateway_connections_xlsx(connections, conn_xlsx)

    multi = sum(1 for v in gw_weights.values() if len(v) > 1)
    print(f"  Gateways ready: {len(zone_map)} external zones "
          f"({multi} split across multiple gateways), "
          f"{len(commune_map)} out-of-catchment communes, "
          f"{len(used_gateways)} gateways used"
          f"{f' ({len(conv_ids)} convergence)' if conv_ids else ''}.")
    return {'gw_weights': gw_weights, 'gateway_station_ids': used_gateways,
            'bs_index': bs_index, 'splits': splits, 'connections': connections,
            'convergence_ids': conv_ids}


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


def _extend_name_lookup_from_breakdown(name_lookup: dict, method: str) -> int:
    """Fill missing station ids in `name_lookup` with (station_number ->
    station_name) pairs from station_commune_breakdown.csv.

    The all-day rail-stops load behind `name_lookup` omits peak-only stations
    (e.g. the Effretikon–Wetzikon corridor), which would otherwise render as bare
    numeric ids. The breakdown carries every attributed station with its name, so
    it is the authoritative gap-filler. Only absent ids are added (existing,
    collision-disambiguated names are left untouched). Returns the count added.
    """
    data_dir = (catchment_base.PT_FEEDER_DATA_DIR if method == 'pt_feeder'
                else catchment_base.MUNICIPAL_DATA_DIR)
    path = os.path.join(paths.MAIN, data_dir, 'station_commune_breakdown.csv')
    if not os.path.exists(path):
        return 0
    df = pd.read_csv(path, encoding='utf-8-sig')
    if 'station_number' not in df.columns or 'station_name' not in df.columns:
        return 0
    added = 0
    for num, nm in zip(df['station_number'], df['station_name']):
        n = pd.to_numeric(num, errors='coerce')
        if pd.isna(n):
            continue
        key  = str(int(n))
        name = str(nm).strip()
        if key not in name_lookup and name and name.lower() != 'nan':
            name_lookup[key] = name
            added += 1
    if added:
        print(f"  Name lookup extended with {added} station name(s) from the "
              f"{method} breakdown (peak-only / out-of-all-day stations).")
    return added


def _check_name_coverage(long_df: pd.DataFrame, name_lookup: dict,
                         label: str) -> None:
    """Warn if any station id in `long_df` has no entry in `name_lookup` (which
    would surface as a bare number in the matrices/plots)."""
    if long_df is None or long_df.empty:
        return
    ids = pd.to_numeric(
        pd.concat([long_df['origin_station_id'], long_df['dest_station_id']]),
        errors='coerce').dropna().astype(int).unique()
    missing = sorted(int(i) for i in ids if str(int(i)) not in name_lookup)
    if missing:
        print(f"  WARNING [{label}]: {len(missing)} station id(s) have no name and "
              f"will render as bare numbers: {missing[:15]}"
              f"{' ...' if len(missing) > 15 else ''}")
    else:
        print(f"  Name coverage [{label}]: all {len(ids)} station ids resolve to names.")


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


def _write_window_xlsx(attr_longs: dict, tau_window: float, name_lookup: dict,
                       output_path: str, label: str = '') -> None:
    """Write one per-window station×station OD workbook with a sheet per attribution
    mode (PT-Feeder: 'Specific' + 'Blended'; Municipal: 'Municipal').

    Args:
        attr_longs: dict[sheet_label -> long-format OD DataFrame].
    """
    matrices = {}
    for sheet, long_df in attr_longs.items():
        matrix = _build_window_matrix(long_df, tau_window, name_lookup)
        if matrix is not None:
            matrices[sheet] = matrix
    if not matrices:
        print(f"    ({label}): no data — skipping {output_path}")
        return

    out_full = Path(output_path)
    out_full.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_full, engine='openpyxl') as writer:
        for sheet, matrix in matrices.items():
            matrix.to_excel(writer, sheet_name=sheet)   # labels are ≤31 chars
    sizes = '; '.join(f"{s}: {m.values.sum():,.0f}" for s, m in matrices.items())
    print(f"    Saved → {out_full}")
    print(f"      {label}: {', '.join(matrices)}  ({sizes}; τ={tau_window})")


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
    (year-scaled, boundary-filtered, no-PT cells excluded). The 'specific' mode
    uses the shares directly as origin / destination weights; the 'blended' mode
    uses the absolute counts (pop_count / empl_count) with the commune totals
    (pop_total / empl_total) for a count-based trip-end blend. No-PT rows (station
    id NO_PT_ID = -1) are dropped; the remaining shares sum to ≤ 1 per commune.

    Args:
        method: 'pt_feeder' | 'municipal' — selects the versioned data dir.

    Returns:
        DataFrame[BFS (int), station_id (int), pop_share, empl_share, pop_count,
        empl_count, pop_total, empl_total (floats)]. pop_total / empl_total are
        the commune-wide totals, constant per commune (repeated on every row).
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
    df = df[['BFS_NR', 'station_number', 'pop_share_pct', 'empl_share_pct',
             'pop_in_station', 'empl_in_station',
             'pop_total_commune', 'empl_total_commune']].copy()
    df['BFS']        = pd.to_numeric(df['BFS_NR'],         errors='coerce')
    df['station_id'] = pd.to_numeric(df['station_number'], errors='coerce')
    df = df.dropna(subset=['BFS', 'station_id'])
    df['BFS']        = df['BFS'].astype(int)
    df['station_id'] = df['station_id'].astype(int)
    df = df[df['station_id'] > 0]   # drop No-PT sentinel (NO_PT_ID = -1)
    df['pop_share']  = pd.to_numeric(df['pop_share_pct'],  errors='coerce').fillna(0.0) / 100.0
    df['empl_share'] = pd.to_numeric(df['empl_share_pct'], errors='coerce').fillna(0.0) / 100.0
    df['pop_count']  = pd.to_numeric(df['pop_in_station'],     errors='coerce').fillna(0.0)
    df['empl_count'] = pd.to_numeric(df['empl_in_station'],    errors='coerce').fillna(0.0)
    df['pop_total']  = pd.to_numeric(df['pop_total_commune'],  errors='coerce').fillna(0.0)
    df['empl_total'] = pd.to_numeric(df['empl_total_commune'], errors='coerce').fillna(0.0)
    out = df[['BFS', 'station_id', 'pop_share', 'empl_share',
              'pop_count', 'empl_count', 'pop_total', 'empl_total']].reset_index(drop=True)
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
        'blended'  — both sides = count-based trip-end blend
                     (OD_BLEND_POP_RATE·pop_in_station + OD_BLEND_EMPL_RATE·empl_in_station,
                     normalised by commune total activity; symmetric).

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
    orig_weights, dest_weights = _attribution_weight_tables(
        breakdown, attribution_mode, gateway_station_ids)

    print(f"    Origin weights: {orig_weights['BFS'].nunique()} communes, "
          f"{orig_weights['station_id'].nunique()} stations")
    print(f"    Dest  weights: {dest_weights['BFS'].nunique()} communes, "
          f"{dest_weights['station_id'].nunique()} stations")

    long_df = _reaggregate_to_stations(communal_od, orig_weights, dest_weights)
    return long_df, orig_weights


def _attribution_weight_tables(breakdown: pd.DataFrame,
                               attribution_mode: str = 'blended',
                               gateway_station_ids=None) -> tuple:
    """Build the (orig_weights, dest_weights) spatial attribution tables from a
    station-commune breakdown.

    'specific' — orig = population share, dest = FTE share (directional).
    'blended'  — both = count-based trip-end blend (symmetric).
    Gateway stations are injected as identity (weight 1.0) rows so an already-
    expanded gateway end resolves to itself. Shared by _run_pt_feeder_branch and
    the standalone attribution_weights() seam so both stay in lock-step.

    Returns:
        (orig_weights[BFS, station_id, orig_weight],
         dest_weights[BFS, station_id, dest_weight]).
    """
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
    return orig_weights, dest_weights


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
    """Symmetric count-based trip-end weight per (commune, station):

        weight = (α·pop_in_station + β·empl_in_station)
                 / (α·pop_total_commune + β·empl_total_commune)

    α = OD_BLEND_POP_RATE, β = OD_BLEND_EMPL_RATE (both 1.0 by default: one daily
    trip-end per resident and per job). Normalising by the commune total — not the
    per-station sum — keeps Σ weight ≤ 1 per commune, so the no-PT share is dropped
    exactly as in the share-based attribution. Applied to both origin and
    destination ends, which (with a symmetric communal OD) yields a symmetric
    station OD. Employment's influence is proportional to the actual job count, so
    job-dense stations are credited without a global ratio overriding the local
    pop:jobs mix. For single-station communes this resolves to ≈ 1.0 (minus no-PT),
    matching the previous behaviour; multi-station communes shift toward their
    job-dense stations.

    Args:
        breakdown: DataFrame[BFS, station_id, pop_count, empl_count, pop_total,
                   empl_total] from _load_station_breakdown.

    Returns:
        DataFrame[BFS, station_id, weight].
    """
    a = float(settings.OD_BLEND_POP_RATE)
    b = float(settings.OD_BLEND_EMPL_RATE)
    m = breakdown.copy()
    numer = a * m['pop_count'] + b * m['empl_count']
    denom = a * m['pop_total'] + b * m['empl_total']
    m['weight'] = np.where(denom > 0, numer / denom, 0.0)
    return m[['BFS', 'station_id', 'weight']]


# ===============================================================================
# COMMUNE OD DEMAND LAYER (GVM-anchored, network-agnostic)
# ===============================================================================
# Year-parameterised commune OD: the 2018 actual scaled by population for the
# trajectory shape and converging exactly to the canton's symmetrised 2040 forecast
# at OD_ANCHOR_YEAR (both years are in the KTZH GVM Excel). No base-year switch —
# the blend is a continuous function of year. Decoupled from the spatial station
# attribution (the count-blend) below; the future scenario loop multiplies a
# per-commune population deviation onto od_communal (the `scenario` hook).
#
#     T(i,j;Y) = T18·pf(i,j;Y) + w(Y)·[Gsym(i,j) − T18·pf(i,j;OD_ANCHOR_YEAR)]
#     pf(i,j;Y) = sqrt(g_i(Y)·g_j(Y)),  g = pop(Y)/pop(OD_BASE_YEAR)
#     Gsym(i,j) = mean(T2040(i,j), T2040(j,i))   # symmetrised → whole-day invariant
#     w(Y)      = clip((Y−OD_BASE_YEAR)/(OD_ANCHOR_YEAR−OD_BASE_YEAR), 0, 1)
#     Y>OD_ANCHOR_YEAR: T = Gsym · pf(Y)/pf(OD_ANCHOR_YEAR)   # population beyond anchor

# Immutable data anchors (years present in the KTZH GVM Excel) — the single source
# of truth for the OD trajectory; not run parameters.
OD_BASE_YEAR = 2018
OD_ANCHOR_YEAR = 2040


def od_communal(year: int, scenario=None) -> pd.DataFrame:
    """GVM-anchored commune OD at `year` (symmetric, whole-day, full daily demand).

    Additive blend of the population-scaled 2018 OD and the symmetrised 2040
    forecast, phased so the result equals the 2018 OD at OD_BASE_YEAR and the 2040
    forecast at OD_ANCHOR_YEAR exactly; beyond OD_ANCHOR_YEAR the 2040 level is
    carried forward by population only. The additive form (vs multiplicative)
    handles the ~1,210 pairs that are new in 2040 (zero in 2018) and never divides
    by a 2018 flow.

    Args:
        year:     Target year for the OD.
        scenario: Reserved scenario hook for the future scenario loop (dormant).
                  When wired, it resolves to a per-commune population deviation
                  delta_i = pop_i(scenario, year) / pop_i(mean, year) (ratio-to-mean,
                  ~1 centrally), applied per pair as wert * sqrt(delta_i * delta_j),
                  so the scenario fan is centred on this GVM-anchored mean without
                  double-counting the structural uplift already in the 2040 anchor.
                  None -> deterministic mean (the central scenario at `year`).

    Returns:
        DataFrame[quelle_code (int), ziel_code (int), wert (float)] with positive
        demand; symmetric. Used by both the PT-Feeder and Municipal branches.
    """
    if scenario is not None:
        # Layer-2 hook (see scenario/delta seams): the scenario loop will scale the
        # mean by per-commune population deviation. Not yet wired — return the mean.
        pass

    t18 = _od_year_symmetric(OD_BASE_YEAR).rename(columns={'wert': 't18'})
    g   = _od_year_symmetric(OD_ANCHOR_YEAR).rename(columns={'wert': 'g'})
    df  = t18.merge(g, on=['quelle_code', 'ziel_code'], how='outer')
    df[['t18', 'g']] = df[['t18', 'g']].fillna(0.0)

    f_y, agg_y = _commune_pop_factors(year)
    f_a, agg_a = _commune_pop_factors(OD_ANCHOR_YEAR)

    def _pf(factors, agg):
        fq = df['quelle_code'].map(factors).fillna(agg)
        fz = df['ziel_code'].map(factors).fillna(agg)
        return np.sqrt(fq.to_numpy(dtype=float) * fz.to_numpy(dtype=float))

    pf_y = _pf(f_y, agg_y)
    pf_a = _pf(f_a, agg_a)
    span = float(OD_ANCHOR_YEAR - OD_BASE_YEAR)
    w = min(max((year - OD_BASE_YEAR) / span, 0.0), 1.0)

    t18v, gv = df['t18'].to_numpy(dtype=float), df['g'].to_numpy(dtype=float)
    blended = t18v * pf_y + w * (gv - t18v * pf_a)
    beyond = np.where(pf_a > 0, gv * (pf_y / pf_a), gv)
    wert = np.where(year > OD_ANCHOR_YEAR, beyond, blended)
    df['wert'] = np.clip(wert, 0.0, None)

    out = df[df['wert'] > 0][['quelle_code', 'ziel_code', 'wert']].reset_index(drop=True)
    print(f"  od_communal({year}): {len(out):,} pairs, total {out['wert'].sum():,.0f} "
          f"(w={w:.2f}; 2018={t18v.sum():,.0f}, Gsym={gv.sum():,.0f})")
    return out


def _od_year_symmetric(year: int) -> pd.DataFrame:
    """Load one GVM year (tau=1 full daily demand), drop intrazonal, symmetrise.

    The raw 2040 forecast carries ~1,148 directionally-asymmetric pairs; averaging
    (i,j)/(j,i) preserves the whole-day-symmetric invariant and conserves total
    demand. The already-symmetric 2018 actual round-trips unchanged.
    """
    od = scoring.GetOevDemandPerCommune(tau=1, year=year)
    od = od[od['quelle_code'] != od['ziel_code']].copy()
    od = od[od['wert'] > 0][['quelle_code', 'ziel_code', 'wert']].copy()
    od['quelle_code'] = od['quelle_code'].astype(int)
    od['ziel_code']   = od['ziel_code'].astype(int)
    if od.empty:
        return od
    od['lo'] = np.minimum(od['quelle_code'], od['ziel_code'])
    od['hi'] = np.maximum(od['quelle_code'], od['ziel_code'])
    pair_mean = od.groupby(['lo', 'hi'], as_index=False)['wert'].mean()
    fwd = pair_mean.rename(columns={'lo': 'quelle_code', 'hi': 'ziel_code'})
    rev = pair_mean.rename(columns={'hi': 'quelle_code', 'lo': 'ziel_code'})
    out = pd.concat([fwd, rev], ignore_index=True)
    out = out[out['quelle_code'] != out['ziel_code']]
    return out[['quelle_code', 'ziel_code', 'wert']].reset_index(drop=True)


def _commune_pop_factors(year: int) -> tuple:
    """Per-commune population factor pop(year)/pop(OD_BASE_YEAR).

    Returns:
        (factors, agg_fallback)
        factors:      dict[int BFS_NR -> float] for communes with positive base pop.
        agg_fallback: float aggregate pop(year)/pop(OD_BASE_YEAR) for missing communes.
    """
    pop_base = catchment_base.load_commune_pop(OD_BASE_YEAR)
    pop_year = catchment_base.load_commune_pop(year)
    factors = {}
    for bfs, p0 in pop_base.items():
        p0 = float(p0)
        p1 = float(pop_year.get(bfs, 0.0))
        if p0 > 0 and p1 > 0:
            factors[int(bfs)] = p1 / p0
    tot0 = float(pop_base[pop_base > 0].sum())
    tot1 = float(pop_year[pop_year.index.isin(pop_base.index)].sum())
    agg_fallback = (tot1 / tot0) if tot0 > 0 else 1.0
    return factors, agg_fallback


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
# SCENARIO / DELTA SEAMS  (dormant — NOT on the deterministic W3 path)
# ===============================================================================
# Interfaces the future scenario loop and per-intervention delta engine will call.
# They reuse the live attribution / reaggregation internals so behaviour stays in
# lock-step, but nothing in prepare_all_od_matrices invokes them today — they lie
# dormant until the scenario tool is wired in.
#
# Composition contract (temporal demand ⟂ spatial attribution):
#     station_OD(network, year, scenario)
#         = _reaggregate_to_stations( od_communal(year, scenario),
#                                     *attribution_weights(network...) )
# W_network is network-specific and — under Option A's frozen pop:job ratio —
# time-invariant, so it is computed once per network and a per-intervention delta
# is merged for the affected communes only (see reaggregate_subset).

def attribution_weights(method: str = 'pt_feeder',
                        attribution_mode: str = 'blended',
                        gateway_station_ids=None) -> tuple:
    """Network-tagged spatial weights W_network as standalone (orig, dest) tables.

    Wraps the same producer the live PT-Feeder branch uses
    (_attribution_weight_tables) so the scenario loop / delta engine can cache base
    weights and merge per-intervention deltas without re-running the branch.
    Municipal uses a 1:1 commune→station map (see _load_municipal_assignment); this
    seam covers the PT-Feeder count-blend / specific weights.

    Returns:
        (orig_weights, dest_weights) — DataFrame[BFS, station_id, *_weight].
    """
    breakdown = _load_station_breakdown(method)
    return _attribution_weight_tables(breakdown, attribution_mode,
                                      gateway_station_ids)


def commune_candidates(method: str = 'pt_feeder') -> dict:
    """dict[int BFS -> sorted list[int station_id]] of each commune's candidate
    stations (those it is attributed to in the breakdown).

    Lets the delta engine identify the communes an intervention touches:
        affected_communes = { c : candidates(c) ∩ affected_stations ≠ ∅ }.
    """
    breakdown = _load_station_breakdown(method)
    return {int(bfs): sorted(int(s) for s in grp['station_id'].unique())
            for bfs, grp in breakdown.groupby('BFS')}


def reaggregate_subset(communal_od: pd.DataFrame, orig_weights: pd.DataFrame,
                       dest_weights: pd.DataFrame, communes) -> pd.DataFrame:
    """Delta entry point: station OD for only the pairs whose origin OR destination
    commune is in `communes` (the affected set).

    The caller merges the returned long_df into the cached base station OD,
    replacing the affected pairs. Reuses _reaggregate_to_stations on the filtered
    communal OD, so the maths matches a full run exactly.
    """
    cset = {int(c) for c in communes}
    sub = communal_od[communal_od['quelle_code'].isin(cset)
                      | communal_od['ziel_code'].isin(cset)]
    return _reaggregate_to_stations(sub, orig_weights, dest_weights)


# ===============================================================================
# TOP-5 EXCEL EXPORT (study-area stations)
# ===============================================================================

def _load_sa_stations(rail_stations, svc_version: str = '',
                      infra_version: str = '') -> gpd.GeoDataFrame:
    """Return the authoritative study-area rail-station set.

    Sourced from rail_stops_sa.gpkg (per svc/infra version), which is filtered from
    the FULL rail_stops layer and therefore includes peak-only stations (e.g. the
    Effretikon–Wetzikon corridor) that the all-day rail-stops load behind
    `rail_stations` omits. Falls back to the legacy SA-boundary filter on
    `rail_stations` when the file is absent.

    Returns:
        GeoDataFrame[id_point (int), stop_name, geometry] for SA-scoped stations.
    """
    sa_path = (paths.get_rail_stops_sa(svc_version, infra_version)
               if (svc_version and infra_version) else '')
    if sa_path and os.path.exists(sa_path):
        from pyogrio import list_layers
        frames = [gpd.read_file(sa_path, layer=lyr)
                  for lyr in list_layers(sa_path)[:, 0]]
        sa = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True)).to_crs(_CODEBASE_CRS)
        # Tolerate both the nodes schema ('Name') and the legacy GTFS schema.
        name_col = ('stop_name' if 'stop_name' in sa.columns
                    else ('Name' if 'Name' in sa.columns else None))
        sa['id_point'] = pd.to_numeric(sa['Number'], errors='coerce')
        sa = sa.dropna(subset=['id_point'])
        sa['id_point']  = sa['id_point'].astype(int)
        sa['stop_name'] = (sa[name_col].astype(str).str.strip()
                           if name_col else sa['id_point'].astype(str))
        out = sa[['id_point', 'stop_name', 'geometry']].reset_index(drop=True)
        print(f"  Study-area stations (rail_stops_sa): {len(out)}")
        return out

    print("  rail_stops_sa.gpkg unavailable — falling back to SA-boundary filter.")
    sa_boundary = catchment_allocate._load_sa_boundary()
    sa_gdf      = catchment_allocate._filter_stations_to_sa(rail_stations, sa_boundary)
    if sa_gdf is None or sa_gdf.empty:
        print("  Study-area boundary unavailable — using all catchment stations.")
        return rail_stations.copy()
    print(f"  Study-area stations: {len(sa_gdf)}")
    return sa_gdf


def _top_relations_table(long_df: pd.DataFrame, sa_ids: set, name_lookup: dict,
                         group_col: str, partner_col: str, partner_label: str,
                         top_n: int = 10) -> pd.DataFrame:
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


def _export_top_relations_excel(long_df, sa_ids, name_lookup, method, svc_network,
                                windows, top_n: int = 10) -> None:
    """Write 'Top_Destinations' and 'Top_Origins' sheets (SA stations only) to a
    standalone od_station_top_relations.xlsx in the versioned per-method OD directory.

    Each sheet stacks one top-N table per time window — full-day first, then peak,
    then off-peak — separated by a blank row and a window header. τ is a uniform
    scalar, so the partner ranking is identical across windows; only the trip
    magnitudes scale.
    """
    if long_df is None or long_df.empty or not sa_ids:
        print(f"    Top-relations export ({method}): no data — skipped")
        return

    tau_by_suffix = {suffix: tau for tau, suffix in windows}
    ordered = [('Full day', tau_by_suffix.get('full_day', 1.0)),
               ('Peak',     tau_by_suffix.get('peak',     1.0)),
               ('Off-peak', tau_by_suffix.get('off_peak', 1.0))]
    sides = [('Top_Destinations', 'origin_station_id', 'dest_station_id', 'dest'),
             ('Top_Origins',      'dest_station_id',   'origin_station_id', 'origin')]

    out_path = paths.get_od_top_relations_xlsx(svc_network, method)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        for sheet, group_col, partner_col, partner_label in sides:
            start = 0
            for win_label, tau in ordered:
                scaled = long_df.copy()
                scaled['trips'] = scaled['trips'] * tau
                tbl = _top_relations_table(scaled, sa_ids, name_lookup,
                                           group_col, partner_col, partner_label,
                                           top_n)
                tbl.to_excel(writer, sheet_name=sheet, startrow=start + 1,
                             index=False)
                writer.sheets[sheet].cell(row=start + 1, column=1,
                                          value=f'{win_label} (τ={tau:g})')
                start += len(tbl) + 3
    print(f"    Top origins/destinations (full-day + peak + off-peak) → {out_path}")


def _export_method_comparison_excel(pt_long: pd.DataFrame, muni_long: pd.DataFrame,
                                    sa_ids: set, name_lookup: dict,
                                    svc_network: str) -> None:
    """Per-SA-station OD-flow comparison between PT-Feeder and Municipal (τ=1).

    For each study-area station: outgoing (sum as origin) and incoming (sum as
    destination) trips under each method, their differences, and the percentage
    change in total throughput Municipal → PT-Feeder. Written to a single-sheet
    od_method_comparison.xlsx. Only meaningful when both methods are present.
    """
    if pt_long is None or muni_long is None or not sa_ids:
        print("    Method comparison: needs both methods — skipped")
        return

    def _flows(long_df):
        return (long_df.groupby('origin_station_id')['trips'].sum(),
                long_df.groupby('dest_station_id')['trips'].sum())

    pt_out, pt_in = _flows(pt_long)
    mu_out, mu_in = _flows(muni_long)

    rows = []
    for sid in sorted(sa_ids):
        po, pi = float(pt_out.get(sid, 0.0)), float(pt_in.get(sid, 0.0))
        mo, mi = float(mu_out.get(sid, 0.0)), float(mu_in.get(sid, 0.0))
        base   = mo + mi
        rows.append({
            'station':    name_lookup.get(str(sid), str(sid)),
            'muni_out':   round(mo, 1), 'pt_out': round(po, 1),
            'muni_in':    round(mi, 1), 'pt_in':  round(pi, 1),
            'delta_out':  round(po - mo, 1), 'delta_in': round(pi - mi, 1),
            'pct_delta':  (round(100.0 * ((po + pi) - base) / base, 1)
                           if base > 0 else None),
        })
    df = pd.DataFrame(rows).sort_values('station').reset_index(drop=True)

    out_path = paths.get_od_method_comparison_xlsx(svc_network)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Method_Comparison', index=False)
    print(f"    Method comparison ({len(df)} SA stations) → {out_path}")


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
                              svc_version: str,
                              attribution_mode: str = '') -> None:
    """OD pie-chart map for SA stations — mirrors _plot_sa_stations_overview_map.

    Each station's top-3 OD partners are shown as pie wedges with one compact
    callout box per pie (single leader) listing 'partner (share %)'. Colours are
    keyed deterministically by partner station id; sub-threshold partners merge
    into 'Other'.

    When the station OD is symmetric (Municipal, or PT-Feeder 'blended') Dest and
    Orig are identical, so a single pie is drawn per station with its callout
    fanned outward (away from the map centre). PT-Feeder 'specific' is directional,
    so two pies are drawn — Dest (left, box left) and Orig (right, box right).

    Output: plots/Traffic_Flow/OD/<svc_version>/<Method>/od_sa_stations_od_map.pdf
    """
    if long_df is None or long_df.empty or sa_stations.empty:
        print(f"    OD pie map ({method}): no data — skipped")
        return
    symmetric = (method == 'municipal'
                 or (attribution_mode or '').strip().lower() == 'blended')
    print(f"  Building SA-stations OD pie map ({method}; "
          f"{'1 pie' if symmetric else '2 pies'}) ...")

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

    # ── Background: rail + surroundings (ghost outside SA, solid inside) ──────
    #     Mirrors catchment_allocate._plot_sa_stations_overview_map so the OD map
    #     carries the same infrastructure context (no white SA mask — the
    #     surrounding catchment area stays visible at reduced opacity).
    from shapely.geometry import box as _box
    extent_box = _box(xmin, ymin, xmax, ymax)

    try:
        muni = gpd.read_file(paths.MUNICIPAL_BOUNDARIES_GPKG).to_crs(
            catchment_base.CODEBASE_CRS)
        if 'objektart' in muni.columns:
            muni = muni[muni['objektart'] == 'Gemeindegebiet']
        muni_full = muni[muni.geometry.intersects(extent_box)].copy()
        muni_full.boundary.plot(ax=ax, color='#B0B0B0', linewidth=0.3,
                                linestyle='--', alpha=0.35, zorder=1)
        if sa_boundary is not None and not muni_full.empty:
            muni_in = gpd.clip(muni_full, sa_boundary)
            if not muni_in.empty:
                muni_in.boundary.plot(ax=ax, color='#808080', linewidth=0.45,
                                      linestyle='--', alpha=0.9, zorder=2)
    except Exception as exc:
        print(f"    WARNING: failed to load municipal boundaries: {exc}")

    try:
        lakes_full = catchment_allocate._load_lakes_for_extent(extent_box, scope='ca')
        if not lakes_full.empty:
            lakes_full.plot(ax=ax, facecolor='#D6E9F2', edgecolor='none',
                            alpha=0.4, zorder=3)
            if sa_boundary is not None:
                lakes_in = gpd.clip(lakes_full, sa_boundary)
                if not lakes_in.empty:
                    lakes_in.plot(ax=ax, facecolor='#A8D8EA', edgecolor='none',
                                  alpha=1.0, zorder=4)
    except Exception as exc:
        print(f"    WARNING: failed to load lakes: {exc}")

    try:
        rail_full = catchment_allocate._load_rail_lines_for_plot(
            extent_box, temporal='all')
        if not rail_full.empty:
            rail_full.plot(ax=ax, color='#FF7F00', linewidth=0.8,
                           alpha=0.35, zorder=5)
            if sa_boundary is not None:
                rail_in = gpd.clip(rail_full, sa_boundary)
                if not rail_in.empty:
                    rail_in.plot(ax=ax, color='#FF7F00', linewidth=1.2,
                                 alpha=1.0, zorder=6)
    except Exception as exc:
        print(f"    WARNING: failed to load rail lines: {exc}")

    if sa_boundary is not None:
        sa_gdf = gpd.GeoDataFrame(geometry=[sa_boundary],
                                  crs=catchment_base.CODEBASE_CRS)
        sa_gdf.boundary.plot(ax=ax, color='black', linewidth=1.3,
                             linestyle='--', zorder=7)

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
        # Show only the top-N partners that each hold >= 5%; everything else
        # (smaller top-N entries plus the long tail) collapses into 'Other'.
        shown = top[top['trips'] / total >= 0.05]
        other = total - float(shown['trips'].sum())
        slices = []
        for _, r in shown.iterrows():
            pid = int(r[partner_col])
            slices.append((_nm(pid), float(r['trips']) / total,
                           colour_map.get(pid, '#888888')))
        if other > 1e-6:
            slices.append(('Other', other / total, '#888888'))
        return slices

    # ── Pie + callout geometry ───────────────────────────────────────────────
    #     One compact callout box per pie (single leader) listing every slice,
    #     placed with collision avoidance (see below).
    pie_radius = min(sa_w, sa_h) * 0.022
    pie_dx     = pie_radius * 1.45   # half-separation of the Dest/Orig pair (specific)
    pie_dy     = pie_radius * 1.30   # vertical lift of pie centre above the marker
    label_dx   = pie_radius * 1.55   # pie centre → callout box anchor
    centre_x   = 0.5 * (xmin + xmax)

    sa_row_by_id = {int(r['id_point']): r for _, r in sa_stations.iterrows()}

    # Axes limits must be final before text extents are measured for de-collision.
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

    _CALLOUT_BBOX = dict(boxstyle='square,pad=0.3', facecolor='white',
                         edgecolor='black', linewidth=0.4)
    _TOTALS_BBOX  = dict(boxstyle='round,pad=0.15', facecolor='white',
                         edgecolor='black', linewidth=0.4)
    _TITLE_BBOX   = dict(boxstyle='square,pad=0.15', facecolor='white',
                         edgecolor='black', linewidth=0.4)

    def _draw_pie(cx, cy, slices):
        """Draw the wedges of one pie (clockwise from 12 o'clock); no labels."""
        start = 90.0
        for _name, share, colour in slices:
            delta = share * 360.0
            ax.add_patch(Wedge(center=(cx, cy), r=pie_radius,
                               theta1=start - delta, theta2=start,
                               facecolor=colour, edgecolor='white',
                               linewidth=0.3, zorder=9))
            start -= delta

    # --- Collision-aware label placement -------------------------------------
    # Each text box is placed at the first candidate position whose rendered
    # extent clears every previously placed box and pie; callouts try their
    # preferred side then the other, each at increasing vertical nudges, and the
    # leader is drawn from the rim to the chosen anchor. If extent measurement is
    # unavailable the preferred slot is used directly (no regression).
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.transforms import Bbox
        renderer = FigureCanvasAgg(fig).get_renderer()
    except Exception:
        renderer = None
    placed = []

    def _clash(bb, pad=4.0):
        for o in placed:
            if (bb.x0 < o.x1 + pad and bb.x1 > o.x0 - pad and
                    bb.y0 < o.y1 + pad and bb.y1 > o.y0 - pad):
                return True
        return False

    def _pie_obstacle(cx, cy):
        if renderer is None:
            return
        (x0, y0) = ax.transData.transform((cx - pie_radius, cy - pie_radius))
        (x1, y1) = ax.transData.transform((cx + pie_radius, cy + pie_radius))
        placed.append(Bbox([[min(x0, x1), min(y0, y1)],
                            [max(x0, x1), max(y0, y1)]]))

    def _try_text(tx, ty, text, ha, va, bbox_kw, bold):
        """Add the text; return it if it clears all obstacles, else remove it."""
        t = ax.text(tx, ty, text, fontsize=5, ha=ha, va=va, zorder=11,
                    fontweight=('bold' if bold else 'normal'), bbox=bbox_kw)
        if renderer is None:
            return t                       # cannot measure → accept first slot
        if not _clash(t.get_window_extent(renderer)):
            placed.append(t.get_window_extent(renderer))
            return t
        t.remove()
        return None

    def _place_callout(cx, cy, slices, pref_side):
        if not slices:
            return
        text   = '\n'.join(f"{nm} ({sh * 100:.1f}%)" for nm, sh, _c in slices)
        sides  = [pref_side, 'right' if pref_side == 'left' else 'left']
        nudges = [0.0, 1.9 * pie_radius, -1.9 * pie_radius,
                  3.8 * pie_radius, -3.8 * pie_radius]
        for s, dy in [(s, dy) for s in sides for dy in nudges]:
            left = (s == 'left')
            bx   = cx - label_dx if left else cx + label_dx
            ha   = 'right' if left else 'left'
            if _try_text(bx, cy + dy, text, ha, 'center', _CALLOUT_BBOX, False):
                rim = cx - pie_radius if left else cx + pie_radius
                ax.plot([rim, bx], [cy, cy + dy], color='black',
                        linewidth=0.4, zorder=10)
                return
        # fallback: preferred side, no nudge (drawn even if it clashes)
        left = (pref_side == 'left')
        bx   = cx - label_dx if left else cx + label_dx
        ha   = 'right' if left else 'left'
        ax.text(bx, cy, text, fontsize=5, ha=ha, va='center', zorder=11,
                bbox=_CALLOUT_BBOX)
        rim = cx - pie_radius if left else cx + pie_radius
        ax.plot([rim, bx], [cy, cy], color='black', linewidth=0.4, zorder=10)

    def _place_totals(cx, cy, stn_name, total_out, total_in):
        text = f"{stn_name}\nOut {total_out:,.0f} · In {total_in:,.0f}"
        for dy in (pie_radius * 1.30, pie_radius * 2.60,
                   -pie_radius * 1.45, pie_radius * 3.90):
            va = 'bottom' if dy >= 0 else 'top'
            if _try_text(cx, cy + dy, text, 'center', va, _TOTALS_BBOX, True):
                return
        ax.text(cx, cy + pie_radius * 1.30, text, fontsize=5, fontweight='bold',
                va='bottom', ha='center', zorder=11, bbox=_TOTALS_BBOX)

    def _place_pie_title(cx, cy, txt):
        t = ax.text(cx, cy - pie_radius * 1.25, txt, fontsize=5, ha='center',
                    va='top', fontweight='bold', zorder=11, bbox=_TITLE_BBOX)
        if renderer is not None:
            placed.append(t.get_window_extent(renderer))

    # ── Per-station rendering (two passes) ───────────────────────────────────
    # Pass 1 draws every pie so callouts can avoid all of them; pass 2 places the
    # text boxes busiest-station-first so the largest callouts win the prime slots.
    stations = []
    for sid, row in sa_row_by_id.items():
        x, y = row.geometry.x, row.geometry.y
        dest_slices = _top_slices(sid, 'origin_station_id', 'dest_station_id')
        orig_slices = _top_slices(sid, 'dest_station_id',   'origin_station_id')
        total_out = float(long_df[long_df['origin_station_id'] == sid]['trips'].sum())
        total_in  = float(long_df[long_df['dest_station_id']   == sid]['trips'].sum())
        stn_name  = name_lookup.get(str(sid), row.get('stop_name', str(sid)))
        cy = y + pie_dy
        if symmetric:
            _draw_pie(x, cy, dest_slices)
            _pie_obstacle(x, cy)
        else:
            _draw_pie(x - pie_dx, cy, dest_slices)
            _draw_pie(x + pie_dx, cy, orig_slices)
            _pie_obstacle(x - pie_dx, cy)
            _pie_obstacle(x + pie_dx, cy)
        stations.append((x, y, cy, dest_slices, orig_slices,
                         total_out, total_in, stn_name))

    stations.sort(key=lambda s: -(s[5] + s[6]))
    for x, y, cy, dest_slices, orig_slices, total_out, total_in, stn_name in stations:
        if symmetric:
            _place_totals(x, cy, stn_name, total_out, total_in)
            _place_callout(x, cy, dest_slices, 'left' if x < centre_x else 'right')
        else:
            _place_pie_title(x - pie_dx, cy, 'Dest')
            _place_pie_title(x + pie_dx, cy, 'Orig')
            _place_totals(x, cy, stn_name, total_out, total_in)
            _place_callout(x - pie_dx, cy, dest_slices, 'left')
            _place_callout(x + pie_dx, cy, orig_slices, 'right')

    ax.set_xlabel('E [m]')
    ax.set_ylabel('N [m]')
    method_label = 'PT-Feeder' if method == 'pt_feeder' else 'Municipal'
    subtitle = ('in = out (symmetric)' if symmetric
                else 'Dest (left pie) / Orig (right pie)')
    ax.set_title(f'SA stations — top-3 OD partners ({method_label}): {subtitle}')
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
# SA RELATION HEATMAPS  (study-area stations x partners / x study area)
# ===============================================================================
# Annotated trip heatmaps mirroring the routing skim heatmaps
# (catchment_OD_rail_network._draw_matrix_heatmap), but for OD trip volumes.
# Rows are the SA corridor stations (SANKEY_CORRIDORS order). Two views per method:
#   (a) SA x top-N partner stations  — the N stations with the most aggregate
#       SA-origin trips (gateways included; they dominate by design).
#   (b) SA x SA — the corridor stations among themselves.
# Values are read from the FULL-DAY station matrix built by _build_window_matrix
# (the same construction as od_matrix_stations_full_day.xlsx), so the heatmaps
# match the produced full-day Excel exactly. One set of plots per method, written
# under that method's own OD plot dir.

def _ordered_sa_ids(present: set) -> list:
    """SA station ids in SANKEY_CORRIDORS (geographic) order, deduped, filtered to
    those present in the OD data."""
    seen, out = set(), []
    for ids in SANKEY_CORRIDORS.values():
        for s in ids:
            si = int(s)
            if si not in seen and si in present:
                seen.add(si)
                out.append(si)
    return out


def _heat_text_colour(rgba) -> str:
    """Black/white annotation colour for legibility on a cell (WCAG luminance)."""
    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
    return 'white' if lum < 0.55 else 'black'


def _fmt_trips(v: float) -> str:
    """Compact trip-count cell label (k-suffixed above 1000)."""
    if v >= 10000:
        return f"{v / 1000:.0f}k"
    if v >= 1000:
        return f"{v / 1000:.1f}k"
    if v >= 1:
        return f"{v:.0f}"
    return f"{v:.1f}" if v > 0 else ''


def _draw_relation_heatmap(mat: np.ndarray, row_labels: list, col_labels: list,
                           title: str, out_pdf: str, cmap: str = 'YlOrRd') -> None:
    """One annotated trip heatmap (rows x cols, trips/day) with a colourbar."""
    vmax = float(np.nanmax(mat)) if mat.size else 0.0
    vmax = vmax if vmax > 0 else 1.0
    fig, ax = plt.subplots(figsize=(max(0.6 * len(col_labels) + 3.0, 6.0),
                                    max(0.5 * len(row_labels) + 2.0, 5.0)))
    im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Trips/day (full-day)', fontsize=9)
    cmap_obj = im.cmap
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if v > 0:
                ax.text(j, i, _fmt_trips(v), ha='center', va='center', fontsize=7,
                        fontweight='bold', color=_heat_text_colour(cmap_obj(v / vmax)))
    ax.set_title(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved relation heatmap → {out_pdf}")


def _plot_sa_relation_heatmaps(long_df: pd.DataFrame, sa_ids: set,
                               name_lookup: dict, svc_version: str, method: str,
                               top_n: int = 10) -> None:
    """Two full-day trip heatmaps for one method's SA corridor stations:
    (a) SA x top-`top_n` partner stations/gateways, (b) SA x SA. Built from the
    full-day station matrix (matches od_matrix_stations_full_day.xlsx) and written
    under the method's own OD plot dir."""
    if long_df is None or long_df.empty or not sa_ids:
        print(f"    SA relation heatmaps ({method}): no data — skipped")
        return
    full_mat = _build_window_matrix(long_df, cp.TAU_FULL_DAY_SHARE, name_lookup)
    if full_mat is None or full_mat.empty:
        print(f"    SA relation heatmaps ({method}): no full-day matrix — skipped")
        return

    sa_set = set(int(s) for s in sa_ids)
    sa_names = []
    for sid in _ordered_sa_ids(sa_set):
        nm = name_lookup.get(str(int(sid)))
        if nm in full_mat.index and nm not in sa_names:
            sa_names.append(nm)
    if not sa_names:
        print(f"    SA relation heatmaps ({method}): no SA origins in matrix — skipped")
        return

    method_label = 'PT-Feeder' if method == 'pt_feeder' else 'Municipal'
    out_dir = paths.get_od_method_plot_dir(svc_version, method)
    print(f"\n  Building SA relation heatmaps ({method_label}, full-day) ...")

    # (a) SA x top-N partners / gateways (cols ranked by total SA-origin trips,
    #     SA stations themselves excluded — those are the (b) view).
    sa_rows = full_mat.loc[sa_names]
    partner_cols = sa_rows.drop(columns=[c for c in sa_names if c in sa_rows.columns],
                                errors='ignore')
    partners = list(partner_cols.sum(axis=0).sort_values(ascending=False).index[:top_n])
    if partners:
        mat_a = full_mat.reindex(index=sa_names, columns=partners,
                                 fill_value=0.0).to_numpy(dtype=float)
        _draw_relation_heatmap(
            mat_a, sa_names, partners,
            f'{method_label}: SA → top-{top_n} partner stations / gateways (full-day trips)',
            os.path.join(out_dir, 'od_heatmap_sa_top_partners.pdf'))

    # (b) SA x SA
    mat_b = full_mat.reindex(index=sa_names, columns=sa_names,
                             fill_value=0.0).to_numpy(dtype=float)
    _draw_relation_heatmap(
        mat_b, sa_names, sa_names,
        f'{method_label}: SA → SA (study area, full-day trips)',
        os.path.join(out_dir, 'od_heatmap_sa_x_sa.pdf'))


# ===============================================================================
# CORRIDOR SANKEY DIAGRAMS
# ===============================================================================
# Per corridor and direction the named partner-node set is the union of each
# corridor station's top-N partners (SANKEY_TOP_N_NAMED); every corridor station's
# flow to a named partner is drawn explicitly (even if it is that station's 5th/6th
# choice), and only flow to non-named partners collapses into a single 'Other' node.
# Static PDF via matplotlib (always); interactive HTML via plotly when importable.

# Study-area corridor membership (id_point), in geographic order.
SANKEY_CORRIDORS = {
    'Corridor_A_Uster_Oberland': [8503128, 8503127, 8503126, 8503125,
                                  8503124, 8503123, 8503130],
    'Corridor_B_Effretikon_Wetzikon': [8503305, 8503303, 8503302, 8503301,
                                       8503300, 8503123],
}
SANKEY_TOP_N_NAMED = 4   # named partners per station (4 named + 'Other' -> "top-5")


def build_corridor_sankeys(long_df: pd.DataFrame, name_lookup: dict,
                           svc_network: str, method: str,
                           attribution_mode: str = '') -> None:
    """Render corridor Sankeys (PDF + optional HTML) from the all-day (tau=1)
    long-format OD, written directly in the per-method plot dir.

    Origins and destinations are identical whenever the station OD is symmetric —
    Municipal, or PT-Feeder 'blended' (both apply a single symmetric attribution
    weight to a symmetric communal OD) — so only the Destination Sankey is drawn
    for those. PT-Feeder 'specific' (origin=Pop, dest=FTE) is directional, so both
    Origin and Destination Sankeys are drawn.
    """
    if long_df is None or long_df.empty:
        print("    Corridor Sankeys: no OD data — skipped")
        return
    symmetric  = (method == 'municipal'
                  or (attribution_mode or '').strip().lower() == 'blended')
    directions = ('dest',) if symmetric else ('dest', 'orig')
    out_dir = paths.get_od_method_plot_dir(svc_network, method)
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Building corridor Sankeys ({method}; "
          f"{'dest only' if symmetric else 'dest + orig'}) ...")
    for cname, ids in SANKEY_CORRIDORS.items():
        for direction in directions:
            _build_corridor_sankey(long_df, ids, direction, name_lookup,
                                   cname, method, out_dir)


def _sk_nm(name_lookup: dict, sid) -> str:
    return name_lookup.get(str(int(sid)), str(int(sid)))


def _corridor_partner_flows(long_df: pd.DataFrame, station_ids, direction) -> dict:
    """dict[corridor_station_id -> dict[partner_id -> trips]].

    'dest': corridor station is the ORIGIN, partner the destination.
    'orig': corridor station is the DESTINATION, partner the origin. Self pairs dropped.
    """
    sset = set(int(s) for s in station_ids)
    s_col, p_col = (('origin_station_id', 'dest_station_id') if direction == 'dest'
                    else ('dest_station_id', 'origin_station_id'))
    sub = long_df[long_df[s_col].isin(sset)]
    flows = {}
    for r in sub.itertuples(index=False):
        S = int(getattr(r, s_col)); P = int(getattr(r, p_col))
        if P == S:
            continue
        flows.setdefault(S, {})
        flows[S][P] = flows[S].get(P, 0.0) + float(r.trips)
    return flows


def _sankey_named_set(flows: dict, top_n: int = SANKEY_TOP_N_NAMED) -> list:
    """Union of each station's top-`top_n` partners, ordered by total flow desc."""
    seen, totals = set(), {}
    for pv in flows.values():
        for P, v in pv.items():
            totals[P] = totals.get(P, 0.0) + v
    for pv in flows.values():
        for P, _v in sorted(pv.items(), key=lambda kv: -kv[1])[:top_n]:
            seen.add(P)
    return sorted(seen, key=lambda P: -totals.get(P, 0.0))


def _build_corridor_sankey(long_df, station_ids, direction, name_lookup, cname,
                           method, out_dir) -> None:
    flows = _corridor_partner_flows(long_df, station_ids, direction)
    if not flows:
        print(f"    {cname} {direction}: no flow — skipped")
        return
    named = _sankey_named_set(flows)
    named_index = {P: i for i, P in enumerate(named)}
    other_idx = len(named)

    corridor_labels = [_sk_nm(name_lookup, s) for s in station_ids]
    partner_labels  = [_sk_nm(name_lookup, p) for p in named] + ['Other']

    links = []
    for si, S in enumerate(station_ids):
        pv = flows.get(int(S), {})
        per_named, other = {}, 0.0
        for P, val in pv.items():
            if P in named_index:
                pi = named_index[P]
                per_named[pi] = per_named.get(pi, 0.0) + val
            else:
                other += val
        for pi, val in per_named.items():
            if val > 0:
                links.append((si, pi, val) if direction == 'dest' else (pi, si, val))
        if other > 1e-9:
            links.append((si, other_idx, other) if direction == 'dest'
                         else (other_idx, si, other))

    if direction == 'dest':
        left_labels, right_labels = corridor_labels, partner_labels
    else:
        left_labels, right_labels = partner_labels, corridor_labels

    method_label = 'PT-Feeder' if method == 'pt_feeder' else 'Municipal'
    pretty = cname.replace('_', ' ')
    kind   = 'Destinations' if direction == 'dest' else 'Origins'
    title  = f"{pretty} — top {kind} ({method_label}, all-day)"

    out_pdf  = os.path.join(out_dir, f"sankey_{cname}_{direction}.pdf")
    out_html = os.path.join(out_dir, f"sankey_{cname}_{direction}.html")
    _draw_sankey_mpl(left_labels, right_labels, links, title, out_pdf)
    _write_sankey_html(left_labels, right_labels, links, title, out_html)


def _sankey_ribbon(ax, x0, x1, sy0, sy1, ty0, ty1, color) -> None:
    """Filled S-curve band from a source segment [sy0,sy1] to a target [ty0,ty1]."""
    xm = (x0 + x1) / 2.0
    verts = [(x0, sy1), (xm, sy1), (xm, ty1), (x1, ty1),
             (x1, ty0), (xm, ty0), (xm, sy0), (x0, sy0), (x0, sy1)]
    codes = [_MplPath.MOVETO, _MplPath.CURVE4, _MplPath.CURVE4, _MplPath.CURVE4,
             _MplPath.LINETO,  _MplPath.CURVE4, _MplPath.CURVE4, _MplPath.CURVE4,
             _MplPath.CLOSEPOLY]
    ax.add_patch(mpatches.PathPatch(_MplPath(verts, codes), facecolor=color,
                                    edgecolor='none', alpha=0.45, zorder=2))


def _sankey_stack(tots, gap, scale):
    """Vertically-centred stacked spans [(y0, y1), ...] for one Sankey column."""
    side_h = sum(t * scale for t in tots) + max(len(tots) - 1, 0) * gap
    y = 0.5 + side_h / 2.0
    spans = []
    for t in tots:
        h = t * scale
        spans.append((y - h, y))
        y -= h + gap
    return spans


def _draw_sankey_multi_mpl(columns, links, title, out_pdf) -> None:
    """Draw an N-column static Sankey (PDF).

    Args:
        columns: ordered list of label-lists, one per column (left -> right).
        links:   list of (col_idx, src_local, dst_local, value) — a flow from node
                 src_local in column col_idx to node dst_local in column col_idx+1.
        title, out_pdf: figure title and output path.

    Node heights take max(inflow, outflow); for conserving middle columns the two
    coincide. The 2-column callers go through _draw_sankey_mpl (a thin wrapper).
    """
    ncols = len(columns)
    sizes = [len(c) for c in columns]
    in_tot  = [[0.0] * n for n in sizes]
    out_tot = [[0.0] * n for n in sizes]
    for (c, s, d, v) in links:
        out_tot[c][s] += v
        in_tot[c + 1][d] += v
    node_tot = [[max(in_tot[c][i], out_tot[c][i]) for i in range(sizes[c])]
                for c in range(ncols)]
    total = max((sum(col) for col in node_tot if col), default=0.0)
    if total <= 0:
        print(f"    Sankey: no flow — skipped {os.path.basename(out_pdf)}")
        return

    gap = 0.02
    scale = (1.0 - (max(sizes) - 1) * gap) / total
    spans = [_sankey_stack(node_tot[c], gap, scale) for c in range(ncols)]
    xs = [0.10] if ncols == 1 else list(np.linspace(0.10, 0.90, ncols))
    nw = 0.02
    palette = plt.colormaps['tab20'].resampled(20)

    fig, ax = plt.subplots(figsize=(13, 8))
    col_colors = []
    for c in range(ncols):
        cols = (['#555555'] * sizes[c] if c == ncols - 1
                else [palette((i + 3 * c) % 20) for i in range(sizes[c])])
        col_colors.append(cols)
        for i, (y0, y1) in enumerate(spans[c]):
            ax.add_patch(mpatches.Rectangle((xs[c] - nw / 2, y0), nw, y1 - y0,
                                            facecolor=cols[i], edgecolor='white',
                                            linewidth=0.5, zorder=3))
            if c == 0:
                ax.text(xs[c] - nw / 2 - 0.012, (y0 + y1) / 2, columns[c][i],
                        ha='right', va='center', fontsize=7)
            elif c == ncols - 1:
                ax.text(xs[c] + nw / 2 + 0.012, (y0 + y1) / 2, columns[c][i],
                        ha='left', va='center', fontsize=7)
            else:
                ax.text(xs[c], y1 + 0.004, columns[c][i], ha='center', va='bottom',
                        fontsize=7)

    right_cur = [[y1 for (_y0, y1) in spans[c]] for c in range(ncols)]
    left_cur  = [[y1 for (_y0, y1) in spans[c]] for c in range(ncols)]
    for (c, s, d, v) in sorted(links, key=lambda t: (t[0], t[1], t[2])):
        h = v * scale
        sy_hi = right_cur[c][s]; sy_lo = sy_hi - h; right_cur[c][s] = sy_lo
        ty_hi = left_cur[c + 1][d]; ty_lo = ty_hi - h; left_cur[c + 1][d] = ty_lo
        _sankey_ribbon(ax, xs[c] + nw / 2, xs[c + 1] - nw / 2,
                       sy_lo, sy_hi, ty_lo, ty_hi, col_colors[c][s])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"    Sankey → {out_pdf}")


def _write_sankey_multi_html(columns, links, title, out_html) -> None:
    """Write an interactive N-column Plotly Sankey; skips silently if plotly absent.

    columns/links use the same convention as _draw_sankey_multi_mpl.
    """
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        print(f"    (plotly unavailable — interactive Sankey skipped: {exc})")
        return
    offs, labels, tot = [], [], 0
    for col in columns:
        offs.append(tot)
        labels += list(col)
        tot += len(col)
    src = [offs[c] + s for (c, s, d, v) in links]
    tgt = [offs[c + 1] + d for (c, s, d, v) in links]
    val = [v for (c, s, d, v) in links]
    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=15, thickness=16,
                  line=dict(color='white', width=0.5)),
        link=dict(source=src, target=tgt, value=val),
    ))
    fig.update_layout(title_text=title, font_size=11)
    fig.write_html(out_html)
    print(f"    Sankey (interactive) → {out_html}")


def _draw_sankey_mpl(left_labels, right_labels, links, title, out_pdf) -> None:
    """2-column static Sankey (thin wrapper over _draw_sankey_multi_mpl)."""
    _draw_sankey_multi_mpl([left_labels, right_labels],
                           [(0, li, ri, v) for (li, ri, v) in links], title, out_pdf)


def _write_sankey_html(left_labels, right_labels, links, title, out_html) -> None:
    """2-column interactive Sankey (thin wrapper over _write_sankey_multi_html)."""
    _write_sankey_multi_html([left_labels, right_labels],
                             [(0, li, ri, v) for (li, ri, v) in links], title, out_html)


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

    # Plot generation — standalone asks; default follows settings.PLOT_STATION_OD.
    # (In the main pipeline the toggle decides without prompting.)
    _default_plot = bool(getattr(settings, 'PLOT_STATION_OD', True))
    _default_str  = 'y' if _default_plot else 'n'
    print("\nPlots (SA stations OD pie map + corridor Sankeys):")
    print(f"  Y/N — default '{_default_str}' from settings.PLOT_STATION_OD")
    while True:
        _p = input(f"Generate plots? [{_default_str}]: ").strip().lower() or _default_str
        if _p in ('y', 'n', 'yes', 'no'):
            _make_plots = _p.startswith('y')
            break
        print("  Invalid — enter y or n.")

    prepare_all_od_matrices(use_cache=settings.use_cache_stationsOD,
                            svc_version=_svc, method=_method,
                            attribution_mode=_attr,
                            both_attributions=True,
                            make_plots=_make_plots)
