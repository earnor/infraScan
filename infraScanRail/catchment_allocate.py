# catchment_allocate.py
#
# Station catchment area generation for infraScanRail.
# Houses both the municipal (centroid-based) and PT-Feeder (GTFS multimodal)
# methods. Both methods are always run so their results can be compared.
#
# Public entry point: get_catchment(use_cache: bool) -> None
#
# Directory layout:
#   Data outputs  -> data/Catchment_Area/            (shared Step 1)
#                    data/Catchment_Area/Municipal/   (municipal method)
#                    data/Catchment_Area/PT_Feeder/   (PT-feeder method)
#   Plot outputs  -> plots/Catchment_Area/            (shared Step 1 plots)
#                    plots/Catchment_Area/Municipal/  (municipal method)
#                    plots/Catchment_Area/PT_Feeder/  (PT-feeder method)

import os
import time

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Wedge
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
import fiona
import re
import pyogrio
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep

import paths
import settings
import cost_parameters as cp

# Shared foundation: output dirs, cartographic primitives, Step 1 (boundary +
# population/employment loading + per-municipality plots). See catchment_base.py.
import catchment_base
from catchment_base import (
    CODEBASE_CRS,
    CELL_SIZE_M,
    MUNICIPAL_DATA_DIR,
    PT_FEEDER_DATA_DIR,
    MUNICIPAL_PLOT_DIR,
    PT_FEEDER_PLOT_DIR,
    GUETEKLASSEN_PLOT_DIR,
    _ensure_dirs,
    _add_north_arrow,
    _add_scale_bar,
    _add_map_elements,
    _load_catchment_boundary,
    load_population_grid_cached,
    load_employment_grid_cached,
)

# ===============================================================================
# CONSTANTS
# ===============================================================================

# ARE-aligned buffer radii (walking access to PT stops)
BUFFER_RAIL_M  = 1000    # m - walking to rail station
BUFFER_TRAM_M  =  750    # m - walking to tram stop
BUFFER_BUS_M   =  500    # m - walking to bus/feeder stop

# Access speeds
WALK_SPEED_MS  = 1.389   # m/s  (5 km/h)
CYCLE_SPEED_MS = 4.167   # m/s  (15 km/h)
CYCLE_RADIUS_M = 2500    # m - cycling search radius to rail

# noPT sentinel
NO_PT_ID       = -1

# Access mode codes (used in allocation logic and visualisation)
MODE_NO_PT  = 0
MODE_WALK   = 1
MODE_BUS    = 2
MODE_TRAM   = 3
MODE_CYCLE  = 4

# Station search constraint (PT-Feeder method)
MAX_CANDIDATE_STATIONS = 5


# ===============================================================================
# TRAVEL-COST METHOD HELPERS
# ===============================================================================
# `settings.TRAVEL_COST_METHOD` controls how access-time components are combined:
#   'calibrated' — literature weights from cost_parameters.py
#   'absolute'   — all weights = 1.0 (raw minutes), unweighted transfer penalty.
# Walk/cycle detour factors are mode-independent (always the cost_parameters
# values) — only the GC weights and transfer penalty switch by method.


def _get_active_weights() -> dict:
    """Return the active travel-cost weights based on settings.TRAVEL_COST_METHOD.

    'calibrated' → values from cost_parameters.py
    'absolute'   → all 1.0
    """
    if settings.TRAVEL_COST_METHOD == 'absolute':
        return {'ivt': 1.0, 'wait': 1.0, 'walk': 1.0, 'bike': 1.0, 'transfer': 1.0}
    return {
        'ivt':      float(cp.W_IVT),
        'wait':     float(cp.W_WAIT),
        'walk':     float(cp.W_WALK),
        'bike':     float(cp.W_BIKE),
        'transfer': float(cp.W_TRANSFER),
    }


def _get_active_detours() -> dict:
    """Return walk and cycle detour factors (Luftlinie → network distance).

    Mode-independent: both 'calibrated' and 'absolute' use the cost_parameters
    values. The travel-cost method only switches the GC weights and transfer
    penalty, not the geometric detour correction.
    """
    return {'walk': float(cp.WALK_DETOUR), 'cycle': float(cp.CYCLE_DETOUR)}


def _get_active_transfer_penalty_sec(headway_min: float = None) -> float:
    """Return the active transfer penalty (seconds) for a single transfer edge.

    'calibrated' + 'fixed_value' → cp.PI_TRANSFER_MIN (12.1 min eq. IVT, pre-weighted)
    'calibrated' + 'explicit'    → W_TRANSFER * (TRANSFER_WALK_MIN + t_wait(h)) minutes
    'absolute'   + 'fixed_value' → cp.average_train_change_time (7.1 min, raw)
    'absolute'   + 'explicit'    → 1.0 * (TRANSFER_WALK_MIN + t_wait(h)) minutes

    Args:
        headway_min: Connecting-service headway (minutes). Only used in 'explicit' model.
    """
    is_abs = (settings.TRAVEL_COST_METHOD == 'absolute')
    if settings.TRANSFER_COST_MODEL == 'explicit':
        h = headway_min if headway_min is not None else float('inf')
        w_transfer = 1.0 if is_abs else float(cp.W_TRANSFER)
        return w_transfer * (cp.TRANSFER_WALK_MIN + cp.t_wait_min(h)) * 60.0
    if is_abs:
        return float(cp.average_train_change_time) * 60.0
    return float(cp.PI_TRANSFER_MIN) * 60.0


# Resolved base paths — set at runtime by get_catchment() via _interactive_config().
# _FEEDER_BASE : absolute path to the selected feeder network directory
#                (FEEDER_LINES_DIR/<svc>/Unprojected or .../Versions/<name>)
# _RAIL_BASE   : absolute path to the rail network Unprojected directory
#                (RAIL_LINES_DIR/<svc>/Unprojected — always Unprojected; named
#                 versions only affect the feeder network)
_FEEDER_BASE: str = ''
_RAIL_BASE:   str = ''
# Active infrastructure projection for plot helpers — when set, the line-loading
# functions look up sibling projected `*_lines.gpkg` under <svc>/<projection>/
# next to Unprojected/. None means fall back to Unprojected.
_INFRA_PROJECTION: str = None
# Interactive mode — True only when this module is invoked as __main__ (so
# the standalone CLI prompts for ambiguous choices). False when called from
# another orchestrator (e.g. main_new.py), in which case cached defaults win.
_INTERACTIVE_MODE: bool = False

# Temporal variant → subfolder + filename-suffix mapping for the unprojected
# rail/feeder GeoPackages. Keys:
#   'full_day' — top-level files (e.g. rail_stops.gpkg) containing ALL services
#   'all_day'  — All_Day subfolder, services operating throughout the day
#   'peak'     — Peak subfolder (AM+PM peak)
#   'offpeak'  — Off_Peak subfolder
# Legacy alias 'all' is normalised to 'all_day' for backward compatibility.
_TEMPORAL_FOLDER_MAP = {
    'full_day': '',
    'all_day':  'All_Day',
    'peak':     'Peak',
    'offpeak':  'Off_Peak',
}
_TEMPORAL_SUFFIX_MAP = {
    'full_day': '',
    'all_day':  '_allday',
    'peak':     '_peak',
    'offpeak':  '_offpeak',
}
_TEMPORAL_LABELS = {
    'full_day': 'Full day',
    'all_day':  'All-day',
    'peak':     'Peak only',
    'offpeak':  'Off-peak only',
}


def _normalise_temporal(t: str) -> str:
    """Map legacy 'all' → 'all_day'; pass through known temporal keys.

    Raises ValueError on unknown keys so callers fail fast rather than
    silently picking the wrong file.
    """
    if t == 'all':
        return 'all_day'
    if t in _TEMPORAL_FOLDER_MAP:
        return t
    raise ValueError(
        f"Unknown temporal '{t}'; expected one of "
        f"{list(_TEMPORAL_FOLDER_MAP.keys())} (or legacy 'all')."
    )


def _temporal_paths(base_dir: str, stem: str, temporal: str) -> str:
    """Resolve `<base_dir>[/<subfolder>]/<stem><suffix>.gpkg` for the temporal
    variant. `stem` is the file stem before any suffix (e.g. 'rail_stops')."""
    t = _normalise_temporal(temporal)
    subfolder = _TEMPORAL_FOLDER_MAP[t]
    suffix    = _TEMPORAL_SUFFIX_MAP[t]
    if subfolder:
        return os.path.join(base_dir, subfolder, f'{stem}{suffix}.gpkg')
    return os.path.join(base_dir, f'{stem}.gpkg')

# ÖV-Güteklassen classification constants (ARE 2022)
# Operational window for headway calculation (ARE standard: 06:00–20:00 = 840 min)
GK_WINDOW_MIN  = 840

# Maximum walk-access buffer radius (m) per Haltestellenkategorie (1=I … 5=V)
GK_MAX_RADIUS  = {1: 1000, 2: 1000, 3: 750, 4: 500, 5: 300}

# Distance bands for Güteklasse ring assignment
GK_DIST_BANDS  = [(0, 300), (300, 500), (500, 750), (750, 1000)]
GK_DIST_LABELS = ['0-300', '300-500', '500-750', '750-1000']

# Güteklasse lookup: (Kat_int, dist_band_0based) → 'A'/'B'/'C'/'D'
# Kat: 1=I, 2=II, 3=III, 4=IV, 5=V  |  band: 0=(0-300m), 1=(300-500m), …
_GK_LOOKUP = {
    (1, 0): 'A', (1, 1): 'A', (1, 2): 'B', (1, 3): 'C',
    (2, 0): 'A', (2, 1): 'B', (2, 2): 'C', (2, 3): 'D',
    (3, 0): 'B', (3, 1): 'C', (3, 2): 'D',
    (4, 0): 'C', (4, 1): 'D',
    (5, 0): 'D',
}

GK_PLOT_COLOURS = {'A': '#2D6A2D', 'B': '#1565C0', 'C': '#E65100', 'D': '#B71C1C'}

_GUETEKLASSEN_ARE_GPKG = os.path.join(
    'data', 'Spatial_Data', 'Transit_Network',
    'Gueteklassen_oev_2026_2056.gpkg', 'OeV_Gueteklassen_ARE.gpkg')


# NOTE: shared output directory layout, cartographic primitives (north arrow,
# scale bar, _add_map_elements) and Step 1 (boundary, population/employment
# loading + per-municipality plots) live in catchment_base.py and are imported
# at the top of this file.

# ===============================================================================
# (Cartographic helpers and Step 1 moved to catchment_base.py)
# ===============================================================================


# ===============================================================================
# STEP 2-MUN: MUNICIPAL METHOD
# ===============================================================================

def _assign_stations_to_municipalities(muni_gdf, rail_stations, bfs_col, name_col):
    """Assign each municipality to one rail station based on spatial proximity.

    Logic per municipality:
      - 0 stations inside polygon -> nearest station (Euclidean from centroid)
      - 1 station inside polygon  -> auto-assign
      - 2+ stations inside polygon -> interactive user prompt

    Returns
    -------
    pd.DataFrame
        Columns: [BFS_NR, NAME, station_id, station_name]
    """
    assignments = []

    for _, row in muni_gdf.iterrows():
        muni_name = row[name_col]
        bfs_nr = row[bfs_col]
        centroid = row.geometry.centroid

        # Find stations whose point geometry falls within this municipality
        mask = rail_stations.geometry.within(row.geometry)
        stations_within = rail_stations[mask]

        if len(stations_within) == 0:
            # Nearest station by Euclidean distance from municipality centroid
            dists = rail_stations.geometry.distance(centroid)
            nearest_idx = dists.idxmin()
            station = rail_stations.loc[nearest_idx]
            sname = (str(station['stop_name'])
                     if pd.notna(station.get('stop_name'))
                     else f"Station {station['id_point']}")
            assignments.append({
                'BFS_NR': bfs_nr, 'NAME': muni_name,
                'station_id': station['id_point'], 'station_name': sname,
            })
            print(f"    {muni_name}: no station inside "
                  f"-> nearest = {sname} ({dists[nearest_idx]:.0f}m)")

        elif len(stations_within) == 1:
            station = stations_within.iloc[0]
            sname = (str(station['stop_name'])
                     if pd.notna(station.get('stop_name'))
                     else f"Station {station['id_point']}")
            assignments.append({
                'BFS_NR': bfs_nr, 'NAME': muni_name,
                'station_id': station['id_point'], 'station_name': sname,
            })
            print(f"    {muni_name}: 1 station -> {sname}")

        else:
            # Multiple stations - interactive prompt
            sorted_st = stations_within.copy()
            sorted_st['_dist'] = sorted_st.geometry.distance(centroid)
            sorted_st = sorted_st.sort_values('_dist')

            print(f"\n  Municipality '{muni_name}' has "
                  f"{len(sorted_st)} stations:")
            for i, (_, st) in enumerate(sorted_st.iterrows()):
                sn = (str(st['stop_name'])
                      if pd.notna(st.get('stop_name'))
                      else f"Station {st['id_point']}")
                print(f"    [{i + 1}] {sn} "
                      f"({st['_dist']:.0f}m from centroid)")

            while True:
                choice = input(f"  Select station for '{muni_name}' "
                               f"[1-{len(sorted_st)}]: ")
                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(sorted_st):
                        break
                except ValueError:
                    pass
                print(f"  Invalid choice. Please enter a number "
                      f"between 1 and {len(sorted_st)}.")

            station = sorted_st.iloc[choice_idx]
            sname = (str(station['stop_name'])
                     if pd.notna(station.get('stop_name'))
                     else f"Station {station['id_point']}")
            assignments.append({
                'BFS_NR': bfs_nr, 'NAME': muni_name,
                'station_id': station['id_point'], 'station_name': sname,
            })
            print(f"    {muni_name}: user selected -> {sname}")

    return pd.DataFrame(assignments)


def _run_municipal_method(boundary, rail_stations, pop_grid=None, empl_grid=None,
                          visualize: bool = True):
    """Automatic municipality-to-station assignment based on spatial proximity.

    For each municipality in the study area:
      - 0 stations inside -> assign nearest station (Euclidean from centroid)
      - 1 station inside  -> auto-assign
      - 2+ stations inside -> interactive prompt

    Builds catchment geometry and exports station summary CSV.

    Returns
    -------
    gpd.GeoDataFrame or None
        Municipal catchment polygons (for diff plot), or None on failure.
    """
    print("\n--- Municipal Method ---")

    # Load municipalities — only those whose centroid falls within the
    # study-area boundary (drops peripheral municipalities that merely
    # touch the edge and would otherwise be assigned to an internal station)
    muni = gpd.read_file(paths.MUNICIPAL_BOUNDARIES_GPKG)
    muni = muni.to_crs(CODEBASE_CRS)
    if 'objektart' in muni.columns:
        muni = muni[muni['objektart'] == 'Gemeindegebiet']
    muni = muni[muni.geometry.centroid.within(boundary)].copy()
    print(f"    {len(muni)} municipalities in study area (centroid within boundary)")

    # Identify BFS column
    bfs_col = None
    for candidate in ['BFS_NR', 'bfs_nr', 'BFS_NUMMER', 'bfs_nummer',
                       'GMDNR', 'gmdnr']:
        if candidate in muni.columns:
            bfs_col = candidate
            break
    if bfs_col is None:
        for c in muni.columns:
            if muni[c].dtype in ['int64', 'int32', 'float64'] \
                    and c != 'geometry':
                bfs_col = c
                break
    if bfs_col is None:
        print("    WARNING: Cannot identify BFS column "
              "- skipping municipal method")
        return None

    # Identify name column
    name_col = None
    for candidate in ['NAME', 'name', 'GMDNAME', 'gmdname',
                       'GEMEINDENAME']:
        if candidate in muni.columns:
            name_col = candidate
            break
    if name_col is None:
        name_col = bfs_col

    # Clip rail stations to the study-area boundary (same filter applied to muni)
    rail_stations = rail_stations[
        rail_stations.geometry.within(boundary)].copy()
    print(f"    {len(rail_stations)} rail stations within study area boundary")

    # Assign stations to municipalities (load cached assignment if available).
    # Standalone run (catchment_allocate.py __main__) prompts the user when a
    # cached CSV is found; main_new.py / non-interactive callers default to
    # reusing the cache (`_INTERACTIVE_MODE` flag).
    assignment_csv = os.path.join(MUNICIPAL_DATA_DIR, 'station_assignment.csv')
    assignment_df = None
    if os.path.exists(assignment_csv):
        print(f"  Existing station assignment found: {assignment_csv}")
        if _INTERACTIVE_MODE:
            while True:
                choice = input("  Use existing assignment? [1] Yes  [2] Reassign: ").strip()
                if choice == '1':
                    assignment_df = pd.read_csv(assignment_csv)
                    assignment_df['BFS_NR'] = pd.to_numeric(
                        assignment_df['BFS_NR'], errors='coerce')
                    # station_id must stay as str to match rail_stations['id_point']
                    assignment_df['station_id'] = (
                        assignment_df['station_id'].astype(str))
                    print(f"    Loaded {len(assignment_df)} assignments from cache")
                    break
                elif choice == '2':
                    break
                else:
                    print("  Please enter 1 or 2.")
        else:
            assignment_df = pd.read_csv(assignment_csv)
            assignment_df['BFS_NR'] = pd.to_numeric(
                assignment_df['BFS_NR'], errors='coerce')
            assignment_df['station_id'] = assignment_df['station_id'].astype(str)
            print(f"    Non-interactive run — reusing cache "
                  f"({len(assignment_df)} assignments)")
    if assignment_df is None:
        print("  Assigning stations to municipalities ...")
        assignment_df = _assign_stations_to_municipalities(
            muni, rail_stations, bfs_col, name_col)
        assignment_df.to_csv(assignment_csv, index=False, encoding='utf-8-sig')
        print(f"    Assignment saved -> {assignment_csv}")
    n_stations = assignment_df['station_id'].nunique()
    print(f"    {len(assignment_df)} municipalities assigned to "
          f"{n_stations} stations")

    # Build catchment geometry (geometry only)
    print("  Building municipal catchment polygons ...")
    muni_catchment = _build_municipal_catchment_geometry(
        assignment_df, muni, bfs_col)

    # Enriched per-(commune, station) breakdown (single source of truth for
    # per-station Pop/FTE, commune shares, and the Stations_Summary roll-up).
    # Hamilton is degenerate for Municipal (1:1) so reconciliation is automatic.
    breakdown = _compute_station_commune_breakdown_municipal(
        assignment_df, rail_stations)

    # Write station_catchments.xlsx (2 sheets) — replaces the previous
    # station_catchment_summary.csv + station_commune_shares.csv pair.
    _, stations_summary = _write_station_catchments_xlsx(
        MUNICIPAL_DATA_DIR, breakdown, rail_stations, method_label='Municipal')

    # Write enriched catchment GPKG: geometry + Pop/FTE/municipalities
    out_path = os.path.join(MUNICIPAL_DATA_DIR, 'catchment.gpkg')
    save_df = muni_catchment.rename(columns={'id_point': 'id'}).copy()
    save_df['train_station'] = save_df['id']
    save_df['id'] = pd.to_numeric(save_df['id'], errors='coerce').astype('Int64')
    summary_idx = stations_summary.set_index('station_number') if not stations_summary.empty \
                  else pd.DataFrame()
    if not summary_idx.empty:
        save_df['station_name']   = save_df['id'].map(summary_idx['station_name']).fillna('—')
        save_df['pop']            = save_df['id'].map(summary_idx['pop']).fillna(0).astype(int)
        save_df['empl']           = save_df['id'].map(summary_idx['empl']).fillna(0).astype(int)
        save_df['municipalities'] = save_df['id'].map(summary_idx['municipalities']).fillna('—')
    else:
        save_df['station_name']   = '—'
        save_df['pop']            = 0
        save_df['empl']           = 0
        save_df['municipalities'] = '—'
    save_df['id'] = save_df['id'].astype('Int64').astype(int)
    save_df[['train_station', 'id', 'station_name', 'pop', 'empl',
             'municipalities', 'geometry']].to_file(out_path, driver='GPKG')
    print(f"    Municipal catchment GPKG saved -> {out_path}  ({len(save_df)} stations)")

    # Phase 4A new plot suite (added 2026-05-25) — Municipal branch produces
    # Excel (already written above) and overview map. PT-Feeder-only plots
    # (pyramids, modes bar, scatters, Güteklassen) are skipped because the
    # municipal method does not provide per-cell mode/access-time data.
    if visualize and pop_grid is not None and empl_grid is not None:
        make_phase_4a_plots(
            method='municipal',
            sa_boundary=None,
            ca_boundary=boundary,
            allocation=pd.DataFrame(),    # no per-cell allocation for municipal
            rail_stations=rail_stations,
            pop_grid=pop_grid,
            empl_grid=empl_grid,
            data_dir=MUNICIPAL_DATA_DIR,
            plot_dir=MUNICIPAL_PLOT_DIR,
            breakdown=breakdown,
        )

    return muni_catchment, assignment_df, muni, bfs_col


def _build_municipal_catchment_geometry(assignment_df, muni_gdf, bfs_col):
    """Dissolve municipality polygons by their assigned station to create
    catchment areas for the municipal method.

    Returns
    -------
    gpd.GeoDataFrame
        Columns: [id_point, geometry] dissolved per station.
    """
    muni_with_station = muni_gdf[[bfs_col, 'geometry']].copy()
    muni_with_station[bfs_col] = pd.to_numeric(
        muni_with_station[bfs_col], errors='coerce')

    lookup = assignment_df[['BFS_NR', 'station_id']].copy()
    lookup['BFS_NR'] = pd.to_numeric(lookup['BFS_NR'], errors='coerce')
    lookup = lookup.rename(columns={'BFS_NR': bfs_col})

    merged = muni_with_station.merge(lookup, on=bfs_col, how='left')
    merged = merged.dropna(subset=['station_id'])

    # Dissolve by station
    dissolved = merged.dissolve(by='station_id', as_index=False)
    dissolved = dissolved[['station_id', 'geometry']].copy()
    dissolved = dissolved.rename(columns={'station_id': 'id_point'})

    return dissolved


def _assign_cells_to_municipal_catchment(pop_grid, empl_grid, muni_catchment):
    """Spatial-join population/employment cells to dissolved municipal catchment
    polygons to obtain a cell-level station assignment for the municipal method.

    Returns
    -------
    pd.DataFrame
        Columns: [RELI, id_point] — one row per unique cell that has pop or empl,
        with the municipal-method station assignment.
    """
    if muni_catchment is None or muni_catchment.empty:
        return pd.DataFrame(columns=['RELI', 'id_point'])

    # Combine pop and empl RELIs (union — a cell counts if it has either)
    pop_relis = set(pop_grid['RELI'].values)
    empl_relis = set(empl_grid['RELI'].values)
    all_relis = pop_relis | empl_relis

    # Build a point GeoDataFrame from both grids (deduplicated)
    combined = pd.concat([
        pop_grid[['RELI', 'geometry']],
        empl_grid[['RELI', 'geometry']],
    ]).drop_duplicates(subset='RELI')
    combined = combined[combined['RELI'].isin(all_relis)].copy()
    combined = gpd.GeoDataFrame(combined, geometry='geometry', crs=CODEBASE_CRS)

    # Spatial join: which municipal catchment polygon contains each cell centroid
    catchment = muni_catchment[['id_point', 'geometry']].copy()
    joined = gpd.sjoin(combined, catchment, how='left', predicate='within')

    # Nearest-catchment fallback for edge cells not strictly within any polygon
    unassigned_mask = joined['id_point'].isna()
    if unassigned_mask.any():
        unassigned = combined.loc[joined.index[unassigned_mask]]
        catchment_reset = catchment.reset_index(drop=True)
        for idx, row in unassigned.iterrows():
            dists = catchment_reset.geometry.boundary.distance(row.geometry)
            nearest = catchment_reset.loc[dists.idxmin(), 'id_point']
            joined.loc[idx, 'id_point'] = nearest

    return joined[['RELI', 'id_point']].copy()


def _plot_municipal_catchments(muni_catchment, pop_grid, empl_grid,
                               rail_stations, boundary,
                               assignment_df=None, muni_gdf=None,
                               bfs_col=None):
    """Plot municipal catchment areas with graph-colouring.  Each coloured
    area is a dissolved municipal catchment (one per assigned station).
    Only cells with population or employment are filled."""
    if muni_catchment is None or muni_catchment.empty:
        print("  Skipping municipal catchment plot - no geometry available")
        return
    print("  Building municipal catchment plot ...")

    # Clip dissolved catchment geometries to the study area boundary
    plot_catchments = gpd.clip(muni_catchment, boundary).reset_index(drop=True)

    # --- Adjacency graph of catchment areas ----------------------------------
    G = nx.Graph()
    for i in range(len(plot_catchments)):
        G.add_node(plot_catchments.loc[i, 'id_point'])

    for i in range(len(plot_catchments)):
        for j in range(i + 1, len(plot_catchments)):
            geom_i = plot_catchments.loc[i, 'geometry']
            geom_j = plot_catchments.loc[j, 'geometry']
            if geom_i.intersects(geom_j):
                inter = geom_i.intersection(geom_j)
                if hasattr(inter, 'length') and inter.length > 0:
                    G.add_edge(plot_catchments.loc[i, 'id_point'],
                               plot_catchments.loc[j, 'id_point'])

    # --- Graph colouring -----------------------------------------------------
    coloring = nx.coloring.greedy_color(G, strategy='largest_first')
    n_colors = max(coloring.values()) + 1 if coloring else 1

    base_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#aec7e8',
        '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
        '#f7b6d2', '#dbdb8d', '#9edae5', '#393b79', '#637939',
    ]
    palette = base_palette[:max(n_colors, 1)]
    color_map = {sid: base_palette[cidx % len(base_palette)]
                 for sid, cidx in coloring.items()}
    max_deg = max(dict(G.degree()).values()) if G.degree() else 0
    print(f"    Graph colouring: {n_colors} colours for "
          f"{len(plot_catchments)} catchment areas "
          f"(max adjacency {max_deg})")

    # --- Build cell geometries -----------------------------------------------
    pop_relis = set(pop_grid['RELI'].values)
    empl_relis = set(empl_grid['RELI'].values)
    all_relis = pop_relis | empl_relis
    combined = pd.concat([
        pop_grid[['RELI', 'E_KOORD', 'N_KOORD', 'geometry']],
        empl_grid[['RELI', 'E_KOORD', 'N_KOORD', 'geometry']],
    ]).drop_duplicates(subset='RELI')
    combined = combined[combined['RELI'].isin(all_relis)].copy()
    combined = gpd.GeoDataFrame(combined, geometry='geometry',
                                crs=CODEBASE_CRS)

    # Spatial join cells -> dissolved catchment polygons
    joined = gpd.sjoin(
        combined, plot_catchments[['id_point', 'geometry']],
        how='left', predicate='within')
    joined['color'] = joined['id_point'].map(color_map).fillna('#D3D3D3')

    # Build square cell polygons
    cell_geoms = [
        Polygon([
            (e, n), (e + CELL_SIZE_M, n),
            (e + CELL_SIZE_M, n + CELL_SIZE_M), (e, n + CELL_SIZE_M),
        ])
        for e, n in zip(joined['E_KOORD'], joined['N_KOORD'])
    ]
    cells_gdf = gpd.GeoDataFrame(
        joined, geometry=cell_geoms, crs=CODEBASE_CRS)

    # --- Plot ----------------------------------------------------------------
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    # Study area background
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=CODEBASE_CRS)
    boundary_gdf.plot(ax=ax, color='#F0F0F0', edgecolor='none')

    # Coloured cells
    for color, group in cells_gdf.groupby('color'):
        group.plot(ax=ax, color=color, edgecolor='none', zorder=2)

    # Catchment area borders
    plot_catchments.boundary.plot(ax=ax, color='#2C3E50', linewidth=0.8,
                                 zorder=4)

    # Municipal borders — thin, light, dashed so individual municipalities
    # within dissolved catchments remain traceable
    muni_plot = gpd.read_file(paths.MUNICIPAL_BOUNDARIES_GPKG).to_crs(CODEBASE_CRS)
    if 'objektart' in muni_plot.columns:
        muni_plot = muni_plot[muni_plot['objektart'] == 'Gemeindegebiet']
    muni_plot = muni_plot[muni_plot.geometry.intersects(boundary)].copy()
    muni_plot = gpd.clip(muni_plot, boundary)
    muni_lines = muni_plot.boundary.explode(index_parts=False)
    muni_lines = muni_lines[~muni_lines.geom_type.isin(['Point', 'MultiPoint'])]
    muni_lines.plot(ax=ax, color='#B0B0B0', linewidth=0.25,
                    linestyle='--', zorder=3)

    # Connecting lines: drawn when a municipality is not contiguous with the
    # primary component of its dissolved catchment (islands) or when the
    # assigned station falls outside the dissolved catchment entirely (cutoff).

    # Pre-compute the primary component for each station:
    # the sub-polygon of the dissolved catchment that contains the station point.
    station_primary = {}  # station_id -> primary component geometry or None
    station_pt_lookup = dict(zip(rail_stations['id_point'], rail_stations.geometry))
    for _, crow in muni_catchment.iterrows():
        sid = crow['id_point']
        diss_geom = crow['geometry']
        st_pt = station_pt_lookup.get(sid)
        if st_pt is None or diss_geom is None:
            station_primary[sid] = None
            continue
        components = (list(diss_geom.geoms)
                      if diss_geom.geom_type == 'MultiPolygon'
                      else [diss_geom])
        station_primary[sid] = next(
            (c for c in components if c.covers(st_pt)), None)

    has_connect_lines = False
    if assignment_df is not None and muni_gdf is not None and bfs_col is not None:
        # Build lookup dict once to avoid per-row GDF boolean-index filtering
        _bfs_num = pd.to_numeric(muni_gdf[bfs_col], errors='coerce')
        muni_geom_lookup = dict(zip(_bfs_num, muni_gdf.geometry))

        line_geoms = []
        for _, arow in assignment_df.iterrows():
            sid = arow['station_id']
            bfs_numeric = pd.to_numeric(arow['BFS_NR'], errors='coerce')
            muni_geom = muni_geom_lookup.get(bfs_numeric)
            st_pt = station_pt_lookup.get(sid)
            if muni_geom is None or st_pt is None:
                continue
            primary = station_primary.get(sid)
            needs_line = (primary is None) or (not muni_geom.intersects(primary))
            if needs_line:
                line_geoms.append(LineString([
                    (st_pt.x, st_pt.y),
                    (muni_geom.centroid.x, muni_geom.centroid.y),
                ]))

        print(f"    Connecting lines: {len(line_geoms)} detached municipality "
              f"assignments found")
        if line_geoms:
            gpd.GeoDataFrame(geometry=line_geoms, crs=CODEBASE_CRS).plot(
                ax=ax, color='black', linewidth=0.4, linestyle='-', zorder=8)
            has_connect_lines = True

    # Study area boundary
    boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8,
                               linestyle='--', zorder=5)

    # Clip view to study area bounds
    bx_min, by_min, bx_max, by_max = boundary.bounds
    pad = 200
    ax.set_xlim(bx_min - pad, bx_max + pad)
    ax.set_ylim(by_min - pad, by_max + pad)

    # Lakes — above catchment fills, below stations; clipped to boundary
    if os.path.exists(paths.LAKES_SHP):
        lakes = gpd.read_file(paths.LAKES_SHP).to_crs(CODEBASE_CRS)
        lakes = lakes[lakes.geometry.intersects(boundary)].copy()
        if not lakes.empty:
            gpd.clip(lakes, boundary).plot(
                ax=ax, color='#A8D4F0', edgecolor='none', zorder=6)

    # Station markers (circles) — only stations within boundary
    prep_bnd = prep(boundary)
    stations_in_boundary = rail_stations[
        rail_stations.geometry.apply(lambda p: prep_bnd.contains(p))]
    assigned_ids = set(plot_catchments['id_point'].astype(str).values)
    has_catchment = stations_in_boundary[
        stations_in_boundary['id_point'].astype(str).isin(assigned_ids)]
    no_catchment = stations_in_boundary[
        ~stations_in_boundary['id_point'].astype(str).isin(assigned_ids)]

    if len(has_catchment) > 0:
        ax.scatter(has_catchment.geometry.x, has_catchment.geometry.y,
                   s=25, c='white', edgecolors='black', linewidths=0.8,
                   marker='o', zorder=7)
    if len(no_catchment) > 0:
        ax.scatter(no_catchment.geometry.x, no_catchment.geometry.y,
                   s=25, c='red', edgecolors='black', linewidths=0.8,
                   marker='o', zorder=7)

    # Legend
    legend_handles = [
        Patch(facecolor=palette[0], edgecolor='none',
              label='Catchment area (coloured)'),
        Line2D([0], [0], color='#2C3E50', linewidth=0.8,
               label='Catchment boundary'),
        Line2D([0], [0], color='#B0B0B0', linewidth=0.25, linestyle='--',
               label='Municipal boundary'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markeredgewidth=0.8,
               markersize=8, label='Station (with catchment)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markeredgecolor='black', markeredgewidth=0.8,
               markersize=8, label='Station (no catchment)'),
        Line2D([0], [0], color='black', linewidth=1.8, linestyle='--',
               label='Study area boundary'),
    ]
    if has_connect_lines:
        legend_handles.insert(-1, Line2D(
            [0], [0], color='black', linewidth=0.4,
            label='External station assignment'))
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8,
              framealpha=0.9)

    # Add cartographic elements
    _add_map_elements(ax)

    ax.set_title('Municipal Catchment Areas', fontsize=14)
    ax.set_xlabel('E [m]')
    ax.set_ylabel('N [m]')
    ax.set_aspect('equal')

    out_path = os.path.join(MUNICIPAL_PLOT_DIR,
                            'catchment_municipal_areas.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path}")


def _resolve_projected_lines_path(base_dir: str, lines_filename: str) -> str:
    """Return the projected `<lines_filename>` path under <svc>/<projection>/
    when a projection is active (or auto-detectable). Empty string if none.

    Precedence:
      1. Explicit `_INFRA_PROJECTION` selected by the user / main_new.
      2. Auto-detect: any sibling of base_dir's parent containing the file.
      3. None — caller falls back to Unprojected.
    """
    if not base_dir:
        return ''
    parent = os.path.dirname(base_dir)
    if _INFRA_PROJECTION:
        cand = os.path.join(parent, _INFRA_PROJECTION, lines_filename)
        if os.path.isfile(cand):
            return cand
    if not os.path.isdir(parent):
        return ''
    for entry in sorted(os.listdir(parent)):
        full = os.path.join(parent, entry)
        if (entry == paths.SERVICES_UNPROJECTED_SUBDIR
                or not os.path.isdir(full)):
            continue
        cand = os.path.join(full, lines_filename)
        if os.path.isfile(cand):
            return cand
    return ''


def _load_lines_gpkg(lines_path: str, boundary, mode_from_layer: bool = False):
    """Read all layers of a lines GeoPackage, reproject to LV95, and clip to
    `boundary.buffer(500)`. When `mode_from_layer` is True the layer name is
    stored in a `mode` column (used for PT-feeder mode-coloured plotting).
    """
    if not lines_path or not os.path.exists(lines_path):
        return gpd.GeoDataFrame(geometry=[], crs=CODEBASE_CRS)
    try:
        layers = fiona.listlayers(lines_path)
    except Exception:
        layers = [None]
    frames = []
    for layer in layers:
        try:
            gdf = (gpd.read_file(lines_path, layer=layer)
                   if layer else gpd.read_file(lines_path))
            gdf = gdf.to_crs(CODEBASE_CRS)
            if mode_from_layer and layer is not None:
                gdf['mode'] = layer
            frames.append(gdf)
        except Exception:
            continue
    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=CODEBASE_CRS)
    out = pd.concat(frames, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry='geometry', crs=CODEBASE_CRS)
    expanded = boundary.buffer(500)
    return out[out.geometry.intersects(expanded)].copy()


def _load_feeder_lines(boundary, temporal='full_day'):
    """Load PT-feeder line geometries for visualisation.

    Precedence: selected projection's pt_feeder_lines.gpkg (no temporal split
    in projections — they aggregate all-day), then auto-detected projection,
    then Unprojected `pt_feeder_lines.gpkg` resolved via the temporal map
    (full_day = top-level; all_day/peak/offpeak = subfolder).

    Returns gpd.GeoDataFrame with columns [geometry, mode].
    """
    # Projected lookup (mode comes from layer name)
    proj_path = _resolve_projected_lines_path(_FEEDER_BASE, 'pt_feeder_lines.gpkg')
    if proj_path:
        print(f"    Using projected feeder lines: {proj_path}")
        return _load_lines_gpkg(proj_path, boundary, mode_from_layer=True)

    # Unprojected fallback (temporal-aware via the shared mapping)
    lines_path = _temporal_paths(_FEEDER_BASE, 'pt_feeder_lines', temporal)
    if not os.path.exists(lines_path):
        print(f"    WARNING: {lines_path} not found — skipping feeder lines")
        return gpd.GeoDataFrame(columns=['geometry', 'mode'], crs=CODEBASE_CRS)
    print(f"    Using Unprojected feeder lines: {lines_path}")
    return _load_lines_gpkg(lines_path, boundary, mode_from_layer=True)


def _load_lakes_for_extent(boundary, scope='ca'):
    """Return lakes (clipped to `boundary`) read from the appropriate
    pre-clipped GPKG.

    Args:
        boundary: Shapely polygon used to filter and clip the lakes.
        scope:    'ca' uses LAKES_CA_GPKG; 'sa' uses LAKES_SA_GPKG. Falls back
                  to the other scope's GPKG or to LAKES_SHP if neither exists.
    """
    path_map = {
        'ca': os.path.join(paths.MAIN, paths.LAKES_CA_GPKG),
        'sa': os.path.join(paths.MAIN, paths.LAKES_SA_GPKG),
    }
    lakes_path = path_map.get(scope, path_map['ca'])
    if not os.path.exists(lakes_path):
        # Fall back to the other scope, then the original shapefile
        alt_scope = 'ca' if scope == 'sa' else 'sa'
        if os.path.exists(path_map[alt_scope]):
            lakes_path = path_map[alt_scope]
        elif os.path.exists(os.path.join(paths.MAIN, paths.LAKES_SHP)):
            lakes_path = os.path.join(paths.MAIN, paths.LAKES_SHP)
        else:
            return gpd.GeoDataFrame(geometry=[], crs=CODEBASE_CRS)
    lakes = gpd.read_file(lakes_path).to_crs(CODEBASE_CRS)
    lakes = lakes[lakes.geometry.intersects(boundary)].copy()
    if lakes.empty:
        return lakes
    return gpd.clip(lakes, boundary)


def _load_rail_lines_for_plot(boundary, temporal='full_day'):
    """Return rail line geometries for plotting.

    Precedence:
      1. Selected projection (_INFRA_PROJECTION) under <svc>/<projection>/rail_lines.gpkg.
      2. Auto-detected: any sibling of Unprojected containing rail_lines.gpkg.
      3. Unprojected `rail_lines.gpkg` resolved via the temporal map
         (full_day = top-level; all_day/peak/offpeak = subfolder).
    """
    # Projected lookup
    proj_path = _resolve_projected_lines_path(_RAIL_BASE, 'rail_lines.gpkg')
    if proj_path:
        print(f"    Using projected rail lines: {proj_path}")
        return _load_lines_gpkg(proj_path, boundary, mode_from_layer=False)

    # Unprojected fallback (temporal-aware via the shared mapping)
    rail_path = _temporal_paths(_RAIL_BASE, 'rail_lines', temporal)
    if not os.path.exists(rail_path):
        print(f"    WARNING: rail lines not found at {rail_path} — no rail layer plotted")
        return gpd.GeoDataFrame(geometry=[], crs=CODEBASE_CRS)
    print(f"    Using Unprojected rail lines: {rail_path}")
    return _load_lines_gpkg(rail_path, boundary, mode_from_layer=False)


def _plot_catchments_with_network(catchment_gdf, rail_stations, boundary,
                                  output_dir, method_label, temporal='all'):
    """Plot dissolved catchment boundaries with the PT-feeder + rail network
    overlaid. Rail lines are drawn orange, funiculars dark green. Lines sit
    above the lakes and below the station markers.

    Args:
        catchment_gdf:  Dissolved catchment polygons (one per station) with
                        an `id_point` column.
        rail_stations:  Full rail stations GeoDataFrame.
        boundary:       Shapely polygon — the study/catchment area used for
                        clipping and view extent.
        output_dir:     Directory to write the PDF.
        method_label:   Either 'Municipal' or 'PT-Feeder' — used in the title
                        and the output filename.
        temporal:       'all' | 'peak' | 'offpeak' for the unprojected loader
                        fallback (ignored when a projection is active).
    """
    if catchment_gdf is None or catchment_gdf.empty:
        print(f"  Skipping {method_label} catchment + network plot — no geometry available")
        return
    print(f"  Building {method_label} catchment + network plot ...")

    feeder_lines = _load_feeder_lines(boundary, temporal)
    rail_lines   = _load_rail_lines_for_plot(boundary, temporal)
    # The line loaders intersect with boundary.buffer(500); clip strictly to
    # the boundary so the network ends exactly at the SA edge.
    if not feeder_lines.empty:
        feeder_lines = gpd.clip(feeder_lines, boundary)
    if rail_lines is not None and not rail_lines.empty:
        rail_lines = gpd.clip(rail_lines, boundary)

    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    # Light grey background for study area
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=CODEBASE_CRS)
    boundary_gdf.plot(ax=ax, color='#F0F0F0', edgecolor='none', zorder=1)

    # Clip catchment geometries to the study area boundary
    clipped_catchment = gpd.clip(catchment_gdf, boundary)
    clipped_catchment.boundary.plot(ax=ax, color='#2C3E50', linewidth=1.0, zorder=2)

    # Lakes — drawn BEFORE the network so lines sit on top (user requirement)
    lakes = _load_lakes_for_extent(boundary, scope='ca')
    if not lakes.empty:
        lakes.plot(ax=ax, color='#A8D4F0', edgecolor='none', zorder=3)

    # PT-feeder network lines coloured by mode
    mode_colours = {
        'bus':          '#0000FF',
        'express_bus':  '#0000FF',
        'on_demand_bus':'#0000FF',
        'tram':         '#FF66CC',
        'metro':        '#00246B',
        'ship':         '#004B8D',
        'funicular':    '#1B5E20',   # dark green
    }
    plotted_modes = []
    if not feeder_lines.empty:
        for mode_name, colour in mode_colours.items():
            subset = feeder_lines[
                feeder_lines['mode'].fillna('').str.lower() == mode_name]
            if len(subset) > 0:
                subset.plot(ax=ax, color=colour, linewidth=0.6, alpha=0.85, zorder=4)
                plotted_modes.append((mode_name, colour))

    # Rail lines — orange, on top of feeder lines
    rail_plotted = rail_lines is not None and not rail_lines.empty
    if rail_plotted:
        rail_lines.plot(ax=ax, color='#FF7F00', linewidth=1.1, alpha=1.0, zorder=5)

    # Study area boundary (between network and stations)
    boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8,
                               linestyle='--', zorder=6)

    # Station markers (circles) — only stations within boundary, on top of everything
    prep_bnd_net = prep(boundary)
    stations_in_bnd = rail_stations[
        rail_stations.geometry.apply(lambda p: prep_bnd_net.contains(p))]
    # Municipal catchment carries 'id_point'; PT-Feeder rebuilds it as 'id'.
    id_col = 'id_point' if 'id_point' in clipped_catchment.columns else 'id'
    assigned_ids = set(clipped_catchment[id_col].astype(str).values)
    has_catchment = stations_in_bnd[stations_in_bnd['id_point'].astype(str).isin(assigned_ids)]
    no_catchment = stations_in_bnd[~stations_in_bnd['id_point'].astype(str).isin(assigned_ids)]

    if len(has_catchment) > 0:
        ax.scatter(has_catchment.geometry.x, has_catchment.geometry.y,
                   s=25, c='white', edgecolors='black', linewidths=0.8,
                   marker='o', zorder=7)
    if len(no_catchment) > 0:
        ax.scatter(no_catchment.geometry.x, no_catchment.geometry.y,
                   s=25, c='red', edgecolors='black', linewidths=0.8,
                   marker='o', zorder=7)

    # Clip view to study area bounds
    bx_min, by_min, bx_max, by_max = boundary.bounds
    pad = 200
    ax.set_xlim(bx_min - pad, bx_max + pad)
    ax.set_ylim(by_min - pad, by_max + pad)

    # Legend
    legend_handles = [
        Line2D([0], [0], color='#2C3E50', linewidth=1.0,
               label=f'{method_label} catchment boundary'),
    ]
    mode_labels = {
        'bus': 'Bus', 'express_bus': 'Express bus', 'on_demand_bus': 'On-demand bus',
        'tram': 'Tram', 'metro': 'Metro', 'ship': 'Ship', 'funicular': 'Funicular',
    }
    for mode_name, colour in plotted_modes:
        legend_handles.append(
            Line2D([0], [0], color=colour, linewidth=1.0, alpha=0.85,
                   label=mode_labels.get(mode_name, mode_name)))
    if rail_plotted:
        legend_handles.append(
            Line2D([0], [0], color='#FF7F00', linewidth=1.1, label='Rail line'))
    legend_handles += [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markeredgewidth=0.8,
               markersize=8, label='Station (with catchment)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markeredgecolor='black', markeredgewidth=0.8,
               markersize=8, label='Station (no catchment)'),
        Line2D([0], [0], color='black', linewidth=1.8, linestyle='--',
               label='Study area boundary'),
    ]
    _leg = ax.legend(handles=legend_handles, loc='upper right', fontsize=8,
                     facecolor='white', edgecolor='black', framealpha=1.0)
    _leg.set_zorder(20)   # above the dashed SA boundary (zorder=6)

    _add_map_elements(ax)
    ax.set_title(f'{method_label} Catchment Areas with PT-Feeder Network', fontsize=14)
    ax.set_xlabel('E [m]')
    ax.set_ylabel('N [m]')
    ax.set_aspect('equal')

    slug = method_label.lower().replace(' ', '_').replace('-', '_')
    out_path = os.path.join(output_dir, f'catchment_{slug}_areas_network.pdf')
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path}")


# Backwards-compatible alias retained — Municipal caller and any external
# imports continue to resolve.
def _plot_municipal_catchments_network(muni_catchment, rail_stations,
                                       boundary, temporal='all'):
    return _plot_catchments_with_network(
        muni_catchment, rail_stations, boundary,
        output_dir=MUNICIPAL_PLOT_DIR, method_label='Municipal',
        temporal=temporal,
    )


# ===============================================================================
# STEP 2-PT: PT-FEEDER BUFFERS
# ===============================================================================

def _load_feeder_stops(boundary, temporal='all'):
    """Load PT-feeder stops from the multi-layer GPKG, assign buffer radii.

    Parameters
    ----------
    temporal : 'full_day' | 'all_day' | 'peak' | 'offpeak'  (legacy 'all' → 'all_day')
        Which temporal variant of the network file to load.
    """
    print("  Loading feeder stops ...")
    stops_path = _temporal_paths(_FEEDER_BASE, 'pt_feeder_stops', temporal)

    layers = fiona.listlayers(stops_path)
    frames = []
    for layer in layers:
        gdf = gpd.read_file(stops_path, layer=layer)
        gdf['mode'] = layer
        frames.append(gdf)
    all_stops = pd.concat(frames, ignore_index=True)
    all_stops = all_stops.rename(columns={'Number': 'stop_id'})
    all_stops = gpd.GeoDataFrame(all_stops, geometry='geometry', crs=CODEBASE_CRS)

    # buffer_radius_m is assigned later by _compute_stop_gueteklassen
    max_buffer = max(GK_MAX_RADIUS.values())  # 1000 m — max possible Güteklassen radius
    expanded = boundary.buffer(max_buffer)
    all_stops = all_stops[all_stops.geometry.within(expanded)].copy()

    keep_cols = ['stop_id', 'stop_name', 'mode', 'geometry']
    for c in keep_cols:
        if c not in all_stops.columns and c != 'geometry':
            all_stops[c] = None
    all_stops = all_stops[keep_cols].copy()

    print(f"    {len(all_stops):,} feeder stops loaded")
    return all_stops


def _load_rail_stations(boundary, temporal='full_day', buffer=BUFFER_RAIL_M):
    """Load rail stations from the rail_stops GPKG produced by services_network_builder.py.
    Maps stop_id -> ID_point by spatial join (100 m buffer) against the same
    rail_stops file — no external ZVV file required.

    Parameters
    ----------
    temporal : 'full_day' | 'all_day' | 'peak' | 'offpeak'  (legacy 'all' → 'all_day')
        Which temporal variant of the network file to load.
    """
    print("  Loading rail stations ...")
    rail_path = _temporal_paths(_RAIL_BASE, 'rail_stops', temporal)

    layers = fiona.listlayers(rail_path)
    frames = []
    for layer in layers:
        gdf = gpd.read_file(rail_path, layer=layer)
        gdf['mode'] = layer
        frames.append(gdf)
    rail = pd.concat(frames, ignore_index=True)
    rail = rail.rename(columns={'Number': 'stop_id'})
    rail = gpd.GeoDataFrame(rail, geometry='geometry', crs=CODEBASE_CRS)

    expanded = boundary.buffer(buffer) if buffer > 0 else boundary
    rail = rail[rail.geometry.within(expanded)].copy()

    rail['stop_id'] = rail['stop_id'].astype(str)
    rail = rail.drop_duplicates(subset='stop_id').copy()

    # The rail_stops.gpkg produced by services_network_builder.py uses the
    # GTFS parent-station numeric ID as stop_id (e.g. '8502224'). This is
    # the same identifier used as the network node key downstream, so we
    # use it directly as id_point without any external crosswalk file.
    rail['diva_nr'] = None
    rail['id_point'] = rail['stop_id']

    keep_cols = ['stop_id', 'stop_name', 'diva_nr', 'id_point', 'mode', 'geometry']
    for c in keep_cols:
        if c not in rail.columns and c != 'geometry':
            rail[c] = None
    rail = rail[keep_cols].copy()

    n_mapped = rail['id_point'].notna().sum()
    print(f"    {len(rail)} rail stations loaded, {n_mapped} mapped to ID_point")
    return rail


def _load_feeder_segments(temporal='full_day'):
    """Load PT-feeder segment edges from the multi-layer GPKG.

    Parameters
    ----------
    temporal : 'full_day' | 'all_day' | 'peak' | 'offpeak'  (legacy 'all' → 'all_day')
        Which temporal variant of the network file to load.
    """
    print("  Loading feeder segments ...")
    seg_path = _temporal_paths(_FEEDER_BASE, 'pt_feeder_segments', temporal)

    layers = fiona.listlayers(seg_path)
    frames = []
    for layer in layers:
        gdf = gpd.read_file(seg_path, layer=layer)
        frames.append(gdf)
    segments = pd.concat(frames, ignore_index=True)

    # Normalise column names: services_network_builder uses abbreviated names
    segments = segments.rename(columns={
        'from_stop_nr': 'from_stop_id',
        'to_stop_nr':   'to_stop_id',
        'TT':           'travel_time_min',
        'Service':      'line_short_name',
    })

    # Also surface route_id (from GTFS_ID) for joining with line-frequency tables
    if 'GTFS_ID' in segments.columns and 'route_id' not in segments.columns:
        segments = segments.rename(columns={'GTFS_ID': 'route_id'})

    keep = ['from_stop_id', 'to_stop_id', 'travel_time_min', 'mode_label',
            'line_short_name', 'route_id', 'direction_id', 'variant_rank',
            'service_period']
    for c in keep:
        if c not in segments.columns:
            segments[c] = None
    segments = segments[keep].copy()

    segments['travel_time_min'] = pd.to_numeric(segments['travel_time_min'], errors='coerce')
    segments = segments.dropna(subset=['travel_time_min'])
    segments = segments[segments['travel_time_min'] > 0].copy()

    print(f"    {len(segments):,} feeder segments loaded")
    return segments


def _build_pt_buffers(feeder_stops, rail_stations, boundary, pop_grid, empl_grid):
    """Generate buffer geometries for visualisation and validation.

    Produces three dissolved, boundary-clipped buffer polygons:
      - walk_to_rail  : Güteklassen walk radius around each rail station
      - pt_feeder     : walk-to-stop buffer for all bus + tram feeder stops
      - cycle_to_rail : fixed CYCLE_RADIUS_M cycling buffer around rail stations

    Attributes per feature: buffer_type, area_m2, population, employment.
    Saved as three separate named layers in buffers_visualisation.gpkg so that
    QGIS loads them with walk_to_rail on top (last written = top of panel).
    """
    print("  Building PT buffers for visualisation ...")
    records = []

    bus_mask  = ~feeder_stops['mode'].str.lower().str.contains('tram')
    tram_mask =  feeder_stops['mode'].str.lower().str.contains('tram')

    for _, mask, buf_type in [
        ('bus',  bus_mask,  'walk_to_feeder_bus'),
        ('tram', tram_mask, 'walk_to_feeder_tram'),
    ]:
        subset = feeder_stops[mask]
        if subset.empty:
            continue
        buf_geoms = [geom.buffer(float(r))
                     for geom, r in zip(subset.geometry, subset['buffer_radius_m'])]
        records.extend([
            {'stop_id': sid, 'buffer_type': buf_type, 'geometry': g}
            for sid, g in zip(subset['stop_id'], buf_geoms)
        ])

    # Rail: walk buffer (Güteklassen radius) + cycling buffer (fixed)
    rail_walk_geoms  = [g.buffer(float(r))
                        for g, r in zip(rail_stations.geometry,
                                        rail_stations['buffer_radius_m'])]
    rail_cycle_geoms = [g.buffer(CYCLE_RADIUS_M) for g in rail_stations.geometry]

    records.extend([
        {'stop_id': sid, 'buffer_type': 'walk_to_rail',  'geometry': g}
        for sid, g in zip(rail_stations['stop_id'], rail_walk_geoms)
    ])
    records.extend([
        {'stop_id': sid, 'buffer_type': 'cycle_to_rail', 'geometry': g}
        for sid, g in zip(rail_stations['stop_id'], rail_cycle_geoms)
    ])

    buffers_gdf = gpd.GeoDataFrame(records, geometry='geometry', crs=CODEBASE_CRS)

    # Dissolve into 3 categories, clip to boundary, compute stats
    pop_pts  = pop_grid[['geometry', 'NUMMER']].copy()
    empl_pts = empl_grid[['geometry', 'NUMMER']].copy()

    def _buffer_stats(geom):
        """Return (area_m2, population, employment) for a dissolved buffer geom."""
        clipped = geom.intersection(boundary)
        if clipped.is_empty:
            return 0.0, 0, 0
        buf_gdf = gpd.GeoDataFrame([{'geometry': clipped}],
                                   geometry='geometry', crs=CODEBASE_CRS)
        pop_in  = int(gpd.sjoin(pop_pts,  buf_gdf, how='inner',
                                predicate='within')['NUMMER'].sum())
        empl_in = int(gpd.sjoin(empl_pts, buf_gdf, how='inner',
                                predicate='within')['NUMMER'].sum())
        return round(clipped.area, 0), pop_in, empl_in

    # Layer order: cycle_to_rail first (bottom in QGIS), pt_feeder second,
    # walk_to_rail last (top in QGIS panel when loaded from file)
    layer_order = [
        (['walk_to_feeder_bus', 'walk_to_feeder_tram'], 'pt_feeder'),
        (['walk_to_rail'],                              'walk_to_rail'),
        (['cycle_to_rail'],                             'cycle_to_rail'),
    ]

    dissolved_by_label = {}
    for buf_types, label in layer_order:
        sub = buffers_gdf[buffers_gdf['buffer_type'].isin(buf_types)]
        if not sub.empty:
            merged_geom = unary_union(sub.geometry.values)
            area_m2, pop_in, empl_in = _buffer_stats(merged_geom)
            clipped_geom = merged_geom.intersection(boundary)
            dissolved_by_label[label] = {
                'buffer_type': label,
                'area_m2':     area_m2,
                'population':  pop_in,
                'employment':  empl_in,
                'geometry':    clipped_geom,
            }

    out_path = os.path.join(PT_FEEDER_DATA_DIR, 'buffers_visualisation.gpkg')
    if os.path.exists(out_path):
        os.remove(out_path)

    # Write in render order: cycle_to_rail → pt_feeder → walk_to_rail
    for label in ['cycle_to_rail', 'pt_feeder', 'walk_to_rail']:
        if label in dissolved_by_label:
            row = dissolved_by_label[label]
            layer_gdf = gpd.GeoDataFrame([row], geometry='geometry', crs=CODEBASE_CRS)
            layer_gdf.to_file(out_path, layer=label, driver='GPKG')
            print(f"    Layer '{label}': area={row['area_m2']:,.0f} m2, "
                  f"pop={row['population']:,}, empl={row['employment']:,}")

    print(f"    Saved -> {out_path}  ({len(dissolved_by_label)} layers)")

    return buffers_gdf


# ===============================================================================
# STEP 3: FEEDER NETWORK GRAPH (PT-Feeder only)
# ===============================================================================

def _build_feeder_graph(feeder_stops, feeder_segments, rail_stations):
    """Build a directed, node-split feeder network and pre-compute shortest
    paths from each rail station to all reachable feeder stops.

    Node-splitting approach
    -----------------------
    Each physical feeder stop is split into one node per **service variant**:
        node id = (stop_id, route_id, direction_id, variant_rank)

    A variant is a unique (route_id, direction_id, variant_rank) tuple.
    Segment edges follow each variant's stop sequence in physical travel
    direction (as listed in GTFS: from_stop_id → to_stop_id), so the graph
    preserves stop ordering within a direction. The two directions of the
    same (route_id, variant_rank) are independent paths in the graph and
    cannot be interchanged without transferring.

    Edge types
    ----------
    • Segment (IVT): directed arc `(from_stop, route, dir, var) →
      (to_stop, route, dir, var)`, weight = travel_time_min × 60 × W_IVT.
      Duplicate rows for the same arc keep the **median** travel time.

    • Transfer: bidirectional arc pair between two variants at the same
      physical stop. ALLOWED between any pair UNLESS they share both
      route_id and variant_rank but differ in direction_id (those two arcs
      are the same service in opposite directions — a passenger turning
      around would need to wait for the return service, modelled by routing
      via another line). Weight from `_get_active_transfer_penalty_sec()`
      using the stop's aggregate headway.

    • Walk-entry (alighting walk): directed arc `(near-rail-feeder, route,
      dir, var) → rail_station`, weight = dist / WALK_SPEED_M × W_WALK.
      Added for every variant at every feeder ≤ 350 m from the rail station,
      with hierarchical preference for stops whose name contains
      'Bahnhof' / 'HB'.

    Dijkstra runs from each rail station on `G.reverse(copy=False)`, which
    traverses edges in the reverse of physical travel direction, yielding
    minimum total time from the rail station outward to every reachable
    feeder split node. The result for a physical stop is the minimum across
    all its split nodes. Each edge carries a `component` attribute
    ('walk', 'ivt', or 'transfer'); the chosen path is decomposed along
    those attributes for diagnostic reporting.

    Returns
    -------
    tuple
        (feeder_stop_to_rail_times, feeder_stop_to_rail_components, G)
        - feeder_stop_to_rail_times[feeder_stop_id] = {rail_stop_id: time_sec}
        - feeder_stop_to_rail_components[feeder_stop_id][rail_stop_id] =
            {'walk': sec, 'ivt': sec, 'transfer': sec}
    """
    print("  Building feeder network graph (directed, per-variant split) ...")
    G = nx.DiGraph()

    _fs_headway = dict(zip(
        feeder_stops['stop_id'].astype(str),
        pd.to_numeric(feeder_stops['headway_min'], errors='coerce').fillna(np.inf)
    ))

    # --- Rail station nodes (unsplit) ---
    rail_ids = set()
    for _, row in rail_stations.iterrows():
        sid = str(row['stop_id'])
        rail_ids.add(sid)
        G.add_node(sid, type='rail')

    feeder_ids = set(feeder_stops['stop_id'].astype(str))

    # --- Build stop -> set of variants serving it ---
    # variant key: (route_id, direction_id, variant_rank)
    def _variant_key(row):
        rid = str(row['route_id']) if pd.notna(row['route_id']) else '__unknown_route__'
        did = int(row['direction_id']) if pd.notna(row['direction_id']) else -1
        vrk = int(row['variant_rank']) if pd.notna(row['variant_rank']) else -1
        return (rid, did, vrk)

    stop_variants = {}   # stop_id (str) -> set of (route_id, direction_id, variant_rank)
    for _, row in feeder_segments.iterrows():
        variant = _variant_key(row)
        for sid in (str(row['from_stop_id']), str(row['to_stop_id'])):
            stop_variants.setdefault(sid, set()).add(variant)

    def _split(stop_id, variant):
        rid, did, vrk = variant
        return (stop_id, rid, did, vrk)

    for sid, variants in stop_variants.items():
        for v in variants:
            G.add_node(_split(sid, v), type='feeder', stop_id=sid)

    # --- Segment edges (directed, physical direction; median across duplicates) ---
    weights = _get_active_weights()
    w_ivt = weights['ivt']
    seg_weight_candidates = {}   # (u, v) -> list of seconds
    for _, row in feeder_segments.iterrows():
        from_id = str(row['from_stop_id'])
        to_id   = str(row['to_stop_id'])
        variant = _variant_key(row)
        weight  = float(row['travel_time_min']) * 60 * w_ivt
        u = _split(from_id, variant)
        v = _split(to_id,   variant)
        if G.has_node(u) and G.has_node(v):
            seg_weight_candidates.setdefault((u, v), []).append(weight)

    n_seg_edges = 0
    for (u, v), cands in seg_weight_candidates.items():
        w = float(np.median(cands))
        G.add_edge(u, v, weight=w, component='ivt')
        n_seg_edges += 1

    # --- Transfer edges (bidirectional pair at same physical stop) ---
    # Allowed: any two variants UNLESS same (route_id, variant_rank) with
    # different direction_id — that's the same service reversing direction,
    # not a real transfer opportunity.
    n_transfer_arcs = 0
    for sid, variants in stop_variants.items():
        h_stop = _fs_headway.get(sid, np.inf)
        transfer_weight = _get_active_transfer_penalty_sec(headway_min=h_stop)
        var_list = list(variants)
        for a in range(len(var_list)):
            v_a = var_list[a]
            for b in range(a + 1, len(var_list)):
                v_b = var_list[b]
                # Skip same variant, opposite direction
                if v_a[0] == v_b[0] and v_a[2] == v_b[2] and v_a[1] != v_b[1]:
                    continue
                u = _split(sid, v_a)
                w = _split(sid, v_b)
                if not G.has_edge(u, w):
                    G.add_edge(u, w, weight=transfer_weight, component='transfer')
                    n_transfer_arcs += 1
                if not G.has_edge(w, u):
                    G.add_edge(w, u, weight=transfer_weight, component='transfer')
                    n_transfer_arcs += 1

    # --- Walk-entry edges: feeder split node -> rail station (alighting walk) ---
    WALK_RADIUS_M = 350
    _BHF_RE = re.compile(r'\bBahnhof\b|\bHB\b', re.IGNORECASE)

    f_coords = np.column_stack([feeder_stops.geometry.x, feeder_stops.geometry.y])
    r_coords = np.column_stack([rail_stations.geometry.x, rail_stations.geometry.y])
    f_ids_arr = feeder_stops['stop_id'].astype(str).values
    r_ids_arr = rail_stations['stop_id'].astype(str).values
    f_names   = feeder_stops['stop_name'].fillna('').astype(str).values
    f_is_named = np.array([bool(_BHF_RE.search(n)) for n in f_names])

    feeder_tree = cKDTree(f_coords)

    n_named_edges    = 0
    n_fallback_edges = 0
    n_rails_named_hit    = 0
    n_rails_fallback_hit = 0
    w_walk = weights['walk']

    for ri in range(len(r_ids_arr)):
        rsid = r_ids_arr[ri]
        near_idxs = feeder_tree.query_ball_point(r_coords[ri], r=WALK_RADIUS_M)
        if not near_idxs:
            continue
        served = [j for j in near_idxs
                  if stop_variants.get(f_ids_arr[j], set())]
        if not served:
            continue
        named = [j for j in served if f_is_named[j]]
        if named:
            chosen = named
            is_named_tier = True
            n_rails_named_hit += 1
        else:
            chosen = served
            is_named_tier = False
            n_rails_fallback_hit += 1

        for j in chosen:
            fsid = f_ids_arr[j]
            dist = np.sqrt(((f_coords[j] - r_coords[ri]) ** 2).sum())
            walk_sec = (dist / WALK_SPEED_MS) * w_walk
            for variant in stop_variants.get(fsid, set()):
                u = _split(fsid, variant)
                if not G.has_node(u):
                    continue
                # Physical direction: alighting at feeder, walking to rail
                if G.has_edge(u, rsid):
                    if G[u][rsid]['weight'] > walk_sec:
                        G[u][rsid]['weight'] = walk_sec
                        G[u][rsid]['component'] = 'walk'
                else:
                    G.add_edge(u, rsid, weight=walk_sec, component='walk')
                    if is_named_tier:
                        n_named_edges += 1
                    else:
                        n_fallback_edges += 1

    _penalty_label = (
        f"calibrated (PI={cp.PI_TRANSFER_MIN}min)"
        if settings.TRAVEL_COST_METHOD == 'calibrated'
        else f"absolute (raw {cp.average_train_change_time}min)"
    )
    print(f"    Graph: {G.number_of_nodes()} nodes, "
          f"{n_seg_edges} segment arcs, "
          f"{n_transfer_arcs} transfer arcs ({_penalty_label}), "
          f"{n_named_edges} named-feeder walk arcs ({n_rails_named_hit} rail stns hit), "
          f"{n_fallback_edges} fallback walk arcs ({n_rails_fallback_hit} rail stns w/o named, "
          f"≤{WALK_RADIUS_M}m)")
    print(f"    Rail stations: {len(rail_ids)}, Physical feeder stops: {len(feeder_ids)}")

    # --- Shortest paths: Dijkstra on the reversed view from each rail station ---
    # G arcs point in physical travel direction (toward rail). Reversing the
    # view lets single_source_dijkstra walk outward from the rail node.
    print("    Computing shortest paths from each rail station ...")
    feeder_stop_to_rail_times = {sid: {} for sid in feeder_ids}
    feeder_stop_to_rail_components = {sid: {} for sid in feeder_ids}

    G_rev = G.reverse(copy=False)

    def _decompose_path(path):
        """Sum edge weights along `path` by their 'component' attribute."""
        comps = {'walk': 0.0, 'ivt': 0.0, 'transfer': 0.0}
        for u, v in zip(path, path[1:]):
            edata = G_rev[u][v]
            c = edata.get('component', 'ivt')
            comps[c] += float(edata.get('weight', 0.0))
        return comps

    for rail_id in rail_ids:
        if rail_id not in G.nodes:
            continue
        try:
            lengths, paths = nx.single_source_dijkstra(G_rev, rail_id, weight='weight')
        except nx.NetworkXError:
            continue
        for node_id, dist in lengths.items():
            if not isinstance(node_id, tuple):
                continue
            physical_sid = node_id[0]
            if physical_sid not in feeder_stop_to_rail_times:
                continue
            existing = feeder_stop_to_rail_times[physical_sid].get(rail_id, np.inf)
            if dist < existing:
                feeder_stop_to_rail_times[physical_sid][rail_id] = dist
                feeder_stop_to_rail_components[physical_sid][rail_id] = (
                    _decompose_path(paths[node_id])
                )

    n_reachable = sum(1 for v in feeder_stop_to_rail_times.values() if v)
    print(f"    {n_reachable}/{len(feeder_ids)} feeder stops reachable from at least one rail station")

    return feeder_stop_to_rail_times, feeder_stop_to_rail_components, G


# ===============================================================================
# STEP 4: CATCHMENT ALLOCATION (PT-Feeder only)
# ===============================================================================

def _compute_walk_to_rail_times(grid, rail_stations):
    """For each cell centroid, find rail stations within their Güteklassen walk
    buffer radius and compute walk time.

    Returns per-cell-station rows with:
      total_time_sec  — generalised-cost weighted access time (= weighted walk time)
      walk_min / bike_min / wait_min / ivt_min / transfer_min — weighted minutes
      access_mode     — 'Walk'
    Per-component minutes sum to total_time_sec / 60 (only walk_min is non-zero).
    """
    coords_grid = np.column_stack([grid.geometry.x, grid.geometry.y])
    coords_rail = np.column_stack([rail_stations.geometry.x, rail_stations.geometry.y])
    rail_radii  = rail_stations['buffer_radius_m'].values.astype(float)

    # Query with the largest radius present; filter per-station below
    max_radius = rail_radii.max() if len(rail_radii) > 0 else float(BUFFER_RAIL_M)

    tree = cKDTree(coords_rail)
    results = []
    neighbors = tree.query_ball_point(coords_grid, r=max_radius)

    reli_values   = grid['RELI'].values
    rail_stop_ids = rail_stations['stop_id'].values

    weights = _get_active_weights()
    detours = _get_active_detours()
    w_walk = weights['walk']
    d_walk = detours['walk']

    for i, near_idxs in enumerate(neighbors):
        for j in near_idxs:
            dist = np.sqrt((coords_grid[i, 0] - coords_rail[j, 0])**2 +
                           (coords_grid[i, 1] - coords_rail[j, 1])**2)
            if dist > rail_radii[j]:
                continue
            walk_sec_weighted = (dist * d_walk / WALK_SPEED_MS) * w_walk
            results.append({
                'RELI': reli_values[i],
                'rail_stop_id': str(rail_stop_ids[j]),
                'total_time_sec': walk_sec_weighted,
                'walk_min':     walk_sec_weighted / 60.0,
                'bike_min':     0.0,
                'wait_min':     0.0,
                'ivt_min':      0.0,
                'transfer_min': 0.0,
                'access_mode': 'Walk'
            })

    return pd.DataFrame(results)


def _compute_cycle_to_rail_times(grid, rail_stations):
    """For each cell centroid, find rail stations within CYCLE_RADIUS_M and
    compute cycling time.

    Returns per-cell-station rows with:
      total_time_sec  — generalised-cost weighted access time (= weighted bike time)
      walk_min / bike_min / wait_min / ivt_min / transfer_min — weighted minutes
      access_mode     — 'Cycle'
    Per-component minutes sum to total_time_sec / 60 (only bike_min is non-zero).
    """
    coords_grid = np.column_stack([grid.geometry.x, grid.geometry.y])
    coords_rail = np.column_stack([rail_stations.geometry.x, rail_stations.geometry.y])

    tree = cKDTree(coords_rail)
    results = []
    neighbors = tree.query_ball_point(coords_grid, r=CYCLE_RADIUS_M)

    reli_values = grid['RELI'].values
    rail_stop_ids = rail_stations['stop_id'].values

    weights = _get_active_weights()
    detours = _get_active_detours()
    w_bike = weights['bike']
    d_cycle = detours['cycle']

    for i, near_idxs in enumerate(neighbors):
        for j in near_idxs:
            dist = np.sqrt((coords_grid[i, 0] - coords_rail[j, 0])**2 +
                           (coords_grid[i, 1] - coords_rail[j, 1])**2)
            bike_sec_weighted = (dist * d_cycle / CYCLE_SPEED_MS) * w_bike
            results.append({
                'RELI': reli_values[i],
                'rail_stop_id': str(rail_stop_ids[j]),
                'total_time_sec': bike_sec_weighted,
                'walk_min':     0.0,
                'bike_min':     bike_sec_weighted / 60.0,
                'wait_min':     0.0,
                'ivt_min':      0.0,
                'transfer_min': 0.0,
                'access_mode': 'Cycle'
            })

    return pd.DataFrame(results)


def _compute_feeder_to_rail_times(grid, feeder_stops, feeder_stop_to_rail_times,
                                   feeder_stop_to_rail_components=None,
                                   transfer_free_headway=None):
    """For each cell centroid, find reachable feeder stops within their
    Güteklassen walk buffer radii, then compute walk-to-stop + graph-derived
    time to each rail station.

    Transfer penalties are already embedded in feeder_stop_to_rail_times by
    _build_feeder_graph: the active transfer penalty (see
    `_get_active_transfer_penalty_sec`) is added only when a path changes
    service line at an intermediate stop.

    When `transfer_free_headway` is provided (W2 feature (a)), the boarding
    wait at the feeder stop is computed per (feeder_stop, rail_station) pair
    using the union frequency of services that go transfer-free between them.
    Pairs without a transfer-free service fall back to the aggregate stop
    headway from `feeder_stops['headway_min']` — strict superset of the
    pre-W2 behaviour.

    When `feeder_stop_to_rail_components` is provided, per-row component
    minutes are populated by decomposing the graph path's edge weights along
    'walk', 'ivt', 'transfer' attributes. Otherwise components default to
    zero except for the cell-to-stop access walk (always known).
    """
    coords_grid = np.column_stack([grid.geometry.x, grid.geometry.y])
    results = []
    reli_values = grid['RELI'].values

    if feeder_stops.empty:
        return pd.DataFrame(results)

    def _feeder_mode_label(mode_str):
        m = mode_str.lower()
        if 'funicular' in m: return 'Funicular'
        if 'tram'      in m: return 'Tram'
        if 'ship'      in m: return 'Ship'
        return 'Bus'

    coords_fs   = np.column_stack([feeder_stops.geometry.x, feeder_stops.geometry.y])
    fs_radii    = feeder_stops['buffer_radius_m'].values.astype(float)
    fs_stop_ids = feeder_stops['stop_id'].astype(str).values
    fs_modes    = np.array([_feeder_mode_label(m)
                            for m in feeder_stops['mode'].astype(str)])

    # Headway lookup for boarding-wait calculation
    _fs_headway_lookup = dict(zip(
        feeder_stops['stop_id'].astype(str),
        pd.to_numeric(feeder_stops['headway_min'], errors='coerce').fillna(np.inf)
    ))

    weights = _get_active_weights()
    w_walk = weights['walk']
    w_wait = weights['wait']

    # Diagnostics for (a): destination-conditional vs aggregate-fallback usage
    n_tf_used = 0
    n_agg_fallback = 0

    # Query with the largest per-stop radius present; filter individually below
    max_radius = fs_radii.max() if len(fs_radii) > 0 else 1000.0
    tree = cKDTree(coords_fs)
    neighbors = tree.query_ball_point(coords_grid, r=max_radius)

    for i, near_idxs in enumerate(neighbors):
        for j in near_idxs:
            dist = np.sqrt((coords_grid[i, 0] - coords_fs[j, 0])**2 +
                           (coords_grid[i, 1] - coords_fs[j, 1])**2)
            if dist > fs_radii[j]:
                continue

            feeder_sid = fs_stop_ids[j]
            rail_times = feeder_stop_to_rail_times.get(feeder_sid, {})
            if not rail_times:
                continue

            # Cell -> feeder stop access walk (weighted)
            access_walk_sec = (dist / WALK_SPEED_MS) * w_walk
            aggregate_h_boarding = _fs_headway_lookup.get(feeder_sid, np.inf)

            for rail_id, graph_time_sec in rail_times.items():
                if transfer_free_headway is not None:
                    key = (feeder_sid, str(rail_id))
                    h_tf = transfer_free_headway.get(key)
                    if h_tf is not None:
                        h_boarding = h_tf
                        n_tf_used += 1
                    else:
                        h_boarding = aggregate_h_boarding
                        n_agg_fallback += 1
                else:
                    h_boarding = aggregate_h_boarding
                    n_agg_fallback += 1
                boarding_wait = w_wait * cp.t_wait_min(h_boarding) * 60.0

                total = access_walk_sec + boarding_wait + graph_time_sec

                # Decompose graph_time_sec into walk / ivt / transfer using the
                # path's edge components (when available).
                if feeder_stop_to_rail_components is not None:
                    comps = feeder_stop_to_rail_components.get(
                        feeder_sid, {}).get(rail_id, {})
                    graph_walk_sec     = comps.get('walk', 0.0)
                    graph_ivt_sec      = comps.get('ivt',  graph_time_sec)
                    graph_transfer_sec = comps.get('transfer', 0.0)
                else:
                    graph_walk_sec     = 0.0
                    graph_ivt_sec      = graph_time_sec
                    graph_transfer_sec = 0.0

                results.append({
                    'RELI': reli_values[i],
                    'rail_stop_id': rail_id,
                    'feeder_stop_id': feeder_sid,
                    'total_time_sec': total,
                    'walk_min':     (access_walk_sec + graph_walk_sec) / 60.0,
                    'bike_min':     0.0,
                    'wait_min':     boarding_wait / 60.0,
                    'ivt_min':      graph_ivt_sec / 60.0,
                    'transfer_min': graph_transfer_sec / 60.0,
                    'access_mode': fs_modes[j]
                })

    # Diagnostic
    total_pairs = n_tf_used + n_agg_fallback
    if transfer_free_headway is not None and total_pairs > 0:
        pct = 100.0 * n_tf_used / total_pairs
        print(f"    Boarding wait: {n_tf_used:,} (cell, feeder, rail) tuples used "
              f"destination-conditional ({pct:.1f}%); "
              f"{n_agg_fallback:,} used aggregate fallback ({100-pct:.1f}%).")

    return pd.DataFrame(results)


def _allocate_cells(walk_df, cycle_df, feeder_df, grid, rail_stations,
                    station_freq_penalty=None):
    """Hierarchical allocation: walk and PT-feeder compete on minimum time;
    cycling only fills cells that neither walk nor feeder can reach.

    When `station_freq_penalty` is provided (W2 feature (b)), an "invisible"
    penalty `cp.t_wait_min(60/dep_per_h_station) * 60` (raw seconds, no
    weight) is added to each row's `total_time_sec` to form `score_time_sec`,
    which is used for the argmin / hierarchical choice. The original
    `total_time_sec` (without penalty) survives as `access_time_sec` in the
    output, so plots and summary tables remain comparable to ARE Güteklasse
    buffer outputs. Effective in both cost methods — the penalty is raw
    seconds, independent of the active weights.

    Returns
    -------
    pd.DataFrame
        Allocation with columns: RELI, E_KOORD, N_KOORD, station_id, id_point,
        access_time_sec (display, without penalty), access_mode,
        walk_min, bike_min, wait_min, ivt_min, transfer_min (weighted minutes).
    """
    print("    Running hierarchical allocation (walk+PT primary, cycle fallback) ...")

    rail_id_map = dict(zip(rail_stations['stop_id'].astype(str),
                           rail_stations['id_point']))

    use_penalty = (station_freq_penalty is not None
                   and len(station_freq_penalty) > 0)

    def _augment(df):
        """Add freq_penalty_sec and score_time_sec; return augmented copy."""
        if df is None or len(df) == 0:
            return df
        out = df.copy()
        if use_penalty:
            out['freq_penalty_sec'] = (out['rail_stop_id'].astype(str)
                                       .map(station_freq_penalty)
                                       .fillna(0.0))
        else:
            out['freq_penalty_sec'] = 0.0
        out['score_time_sec'] = out['total_time_sec'] + out['freq_penalty_sec']
        return out

    walk_df_aug   = _augment(walk_df)
    feeder_df_aug = _augment(feeder_df)
    cycle_df_aug  = _augment(cycle_df)

    # --- Primary competition: walk vs PT-feeder ---
    primary_frames = [df for df in [walk_df_aug, feeder_df_aug]
                      if df is not None and len(df) > 0]
    if primary_frames:
        primary = pd.concat(primary_frames, ignore_index=True)
        allocation = primary.loc[primary.groupby('RELI')['score_time_sec'].idxmin()].copy()
    else:
        allocation = pd.DataFrame(columns=['RELI', 'rail_stop_id', 'total_time_sec',
                                            'score_time_sec', 'access_mode'])

    # --- Cycling fallback: only for cells unreached by walk or PT ---
    if cycle_df_aug is not None and len(cycle_df_aug) > 0:
        covered = set(allocation['RELI'])
        cycle_uncovered = cycle_df_aug[~cycle_df_aug['RELI'].isin(covered)]
        if len(cycle_uncovered) > 0:
            cycle_best = cycle_uncovered.loc[
                cycle_uncovered.groupby('RELI')['score_time_sec'].idxmin()].copy()
            allocation = pd.concat([allocation, cycle_best], ignore_index=True)

    if len(allocation) == 0:
        print("    WARNING: No access routes found - all cells get NO_PT")

    # Drop the score column from the public output but keep access_time_sec
    # equal to the displayed (penalty-free) total_time_sec.
    if 'score_time_sec' in allocation.columns:
        allocation = allocation.drop(columns=['score_time_sec'])
    if 'freq_penalty_sec' in allocation.columns:
        allocation = allocation.drop(columns=['freq_penalty_sec'])

    allocation = allocation.rename(columns={'rail_stop_id': 'station_id',
                                             'total_time_sec': 'access_time_sec'})
    allocation['id_point'] = allocation['station_id'].map(rail_id_map)

    grid_coords = grid[['RELI', 'E_KOORD', 'N_KOORD']].drop_duplicates(subset='RELI')
    allocation = allocation.merge(grid_coords, on='RELI', how='right')

    allocation['station_id'] = allocation['station_id'].fillna(str(NO_PT_ID))
    allocation['id_point'] = allocation['id_point'].fillna(NO_PT_ID)
    allocation['access_time_sec'] = allocation['access_time_sec'].fillna(99999)
    allocation['access_mode'] = allocation['access_mode'].fillna('No PT')

    # Fill component columns to 0 for NO_PT cells (right-join survivors with no
    # matching row in walk/cycle/feeder DataFrames).
    for _comp_col in ('walk_min', 'bike_min', 'wait_min', 'ivt_min', 'transfer_min'):
        if _comp_col in allocation.columns:
            allocation[_comp_col] = allocation[_comp_col].fillna(0.0)
        else:
            allocation[_comp_col] = 0.0

    allocation['id_point'] = allocation['id_point'].astype(int)

    n_assigned = (allocation['id_point'] != NO_PT_ID).sum()
    print(f"    {n_assigned}/{len(allocation)} cells assigned to a rail station")

    return allocation


def _build_candidates_csv(walk_df, cycle_df, feeder_df,
                           pop_grid, empl_grid, rail_stations, output_dir):
    """Build wide-format cell-station candidates CSV combining population and
    employment for each cell.

    Columns: RELI, Pop, Emp,
             Station_1_ID, Access_Time_1_min, Access_Mode_1,
             Walk_1_min, Bike_1_min, Wait_1_min, IVT_1_min, Transfer_1_min,
             … (up to MAX_CANDIDATE_STATIONS slots).

    Per-slot component minutes sum to Access_Time_{i}_min within rounding
    tolerance. They are weighted by the active travel-cost weights
    (`_get_active_weights()`).

    Slot ordering: nearest station first (by best access time across modes),
    then all available modes to that station in priority order
    (Walk → PT-Feeder → Cycle), then next nearest station, and so on.
    Cells with no access to any station are excluded.

    In the CSV output all PT feeder modes (Bus, Tram, Ship, Funicular) are
    collapsed to 'PT-Feeder'.  The plots use the original mode names.
    """
    _PT_FEEDER_MODES = {'Bus', 'Tram', 'Ship', 'Funicular'}
    _MODE_ORDER = {'Walk': 0, 'PT-Feeder': 1, 'Cycle': 2}

    rail_id_map = dict(zip(rail_stations['stop_id'].astype(str),
                           rail_stations['id_point']))

    pop_dict  = pop_grid.groupby('RELI')['NUMMER'].sum().to_dict()
    empl_dict = empl_grid.groupby('RELI')['NUMMER'].sum().to_dict()

    all_frames = [df for df in [walk_df, feeder_df, cycle_df] if len(df) > 0]
    if not all_frames:
        print("    WARNING: No access data — candidates CSV not written")
        return

    _comp_cols = ['walk_min', 'bike_min', 'wait_min', 'ivt_min', 'transfer_min']

    # Defensive: ensure all per-component columns exist in each source frame
    aligned_frames = []
    for _df in all_frames:
        _df = _df.copy()
        for _c in _comp_cols:
            if _c not in _df.columns:
                _df[_c] = 0.0
        aligned_frames.append(_df)

    combined = pd.concat(aligned_frames, ignore_index=True)
    combined['id_point'] = combined['rail_stop_id'].astype(str).map(rail_id_map)

    # One entry per (cell, station, mode): keep the row with the minimum
    # total_time_sec (so per-component minutes correspond to the chosen path).
    combined = combined.sort_values('total_time_sec', kind='mergesort')
    combined = combined.drop_duplicates(
        subset=['RELI', 'rail_stop_id', 'access_mode'], keep='first')

    # Collapse all PT feeder mode names to 'PT-Feeder' for CSV output, then
    # keep the fastest per (cell, station, collapsed-mode) so Bus+Tram don't
    # consume two slots.
    combined['access_mode'] = combined['access_mode'].apply(
        lambda m: 'PT-Feeder' if m in _PT_FEEDER_MODES else m)
    combined = combined.sort_values('total_time_sec', kind='mergesort')
    combined = combined.drop_duplicates(
        subset=['RELI', 'rail_stop_id', 'access_mode'], keep='first')

    # Station rank within cell: minimum access time to that station across all modes
    station_min = (combined.groupby(['RELI', 'rail_stop_id'])['total_time_sec']
                   .min().rename('station_min_sec').reset_index())
    combined = combined.merge(station_min, on=['RELI', 'rail_stop_id'])

    # Sort: nearest station first, then Walk → PT-Feeder → Cycle within station
    combined['mode_ord'] = combined['access_mode'].map(_MODE_ORDER).fillna(9).astype(int)
    combined = combined.sort_values(['RELI', 'station_min_sec', 'mode_ord'],
                                    kind='mergesort')

    # Assign slot numbers (1-based) and keep top-N per cell
    combined['slot'] = combined.groupby('RELI').cumcount() + 1
    top_n = combined[combined['slot'] <= MAX_CANDIDATE_STATIONS].copy()

    top_n['time_min']   = (top_n['total_time_sec'] / 60).round(1)
    top_n['station_id'] = top_n['id_point'].fillna(NO_PT_ID).astype(int)
    for _c in _comp_cols:
        top_n[_c] = top_n[_c].round(2)

    # Pivot to wide format (fully vectorised)
    piv_id   = top_n.pivot(index='RELI', columns='slot', values='station_id')
    piv_time = top_n.pivot(index='RELI', columns='slot', values='time_min')
    piv_mode = top_n.pivot(index='RELI', columns='slot', values='access_mode')

    piv_id.columns   = [f'Station_{c}_ID'      for c in piv_id.columns]
    piv_time.columns = [f'Access_Time_{c}_min' for c in piv_time.columns]
    piv_mode.columns = [f'Access_Mode_{c}'     for c in piv_mode.columns]

    # Component pivots — one per (component, slot)
    _comp_label = {'walk_min': 'Walk', 'bike_min': 'Bike', 'wait_min': 'Wait',
                   'ivt_min':  'IVT',  'transfer_min': 'Transfer'}
    comp_pivots = {}
    for _src, _lbl in _comp_label.items():
        p = top_n.pivot(index='RELI', columns='slot', values=_src)
        p.columns = [f'{_lbl}_{c}_min' for c in p.columns]
        comp_pivots[_lbl] = p

    wide = pd.concat([piv_id, piv_time, piv_mode] + list(comp_pivots.values()),
                     axis=1)

    # Interleave per slot: Station_i_ID, Access_Time_i_min, Access_Mode_i,
    # Walk_i_min, Bike_i_min, Wait_i_min, IVT_i_min, Transfer_i_min
    slot_cols = []
    for i in range(1, MAX_CANDIDATE_STATIONS + 1):
        for col in (f'Station_{i}_ID', f'Access_Time_{i}_min', f'Access_Mode_{i}',
                    f'Walk_{i}_min', f'Bike_{i}_min', f'Wait_{i}_min',
                    f'IVT_{i}_min', f'Transfer_{i}_min'):
            if col in wide.columns:
                slot_cols.append(col)

    wide = wide[slot_cols].reset_index()
    # Cells stay as float (Phase 2 keeps them float; integers only at aggregate
    # boundaries — commune totals and per-(commune, station) splits).
    wide['Pop'] = wide['RELI'].map(pop_dict).fillna(0.0).round(2)
    wide['Emp'] = wide['RELI'].map(empl_dict).fillna(0.0).round(2)
    wide = wide[['RELI', 'Pop', 'Emp'] + slot_cols]

    out_path = os.path.join(output_dir, 'cell_station_candidates.csv')
    wide.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"    Cell candidates ({len(wide):,} cells, ≤{MAX_CANDIDATE_STATIONS} slots each)"
          f" saved -> {out_path}")


# ===============================================================================
# OUTPUTS
# ===============================================================================

def _commune_targets_from_summary():
    """Read per-commune Pop/FTE integer targets from the municipal_pop_empl_summary CSV.

    Returns (commune_pop_dict, commune_empl_dict) keyed by integer BFS_NR.
    Returns ({}, {}) when the summary file is unavailable; the caller falls
    back to in-data sums as targets, which still yields integer totals but
    without cross-checking against Phase 2's commune totals.
    """
    summary_path = os.path.join(paths.MAIN, catchment_base.POP_EMPL_DATA_DIR,
                                'municipal_pop_empl_summary.csv')
    if not os.path.exists(summary_path):
        print(f"    WARNING: {summary_path} not found — Hamilton targets fall back to in-data sums")
        return {}, {}
    df = pd.read_csv(summary_path, encoding='utf-8-sig')
    df['BFS_NR'] = pd.to_numeric(df['BFS_NR'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['BFS_NR'])
    df['BFS_NR'] = df['BFS_NR'].astype(int)
    pop_d  = dict(zip(df['BFS_NR'], df['total_population'].round().astype(int)))
    empl_d = dict(zip(df['BFS_NR'], df['total_employment'].round().astype(int)))
    return pop_d, empl_d


def _apply_hamilton_per_commune(per_pair: pd.DataFrame,
                                  commune_pop: dict, commune_empl: dict) -> tuple:
    """Apply Hamilton's largest-remainder rounding per commune so the sum of
    station shares for that commune equals the commune integer target.

    Args:
        per_pair:     DataFrame with columns [BFS_NR, station_id, pop, empl].
                       pop and empl may be float; values inside each commune
                       are reconciled to that commune's integer total.
        commune_pop:  dict[BFS_NR -> int] of commune Pop targets.
        commune_empl: dict[BFS_NR -> int] of commune FTE targets.
        Empty dicts → fall back to in-data sums per commune.

    Returns:
        (per_pair_int, drift_pop, drift_empl) where per_pair_int has integer
        pop and empl columns; drift_* is the total absolute drift across all
        communes (must be 0 for exact reconciliation).
    """
    out = per_pair.copy()
    n = len(out)
    new_pop  = np.zeros(n, dtype=int)
    new_empl = np.zeros(n, dtype=int)
    drift_pop = drift_empl = 0
    for bfs_id, group in out.groupby('BFS_NR', sort=False):
        positions = out.index.get_indexer(group.index)
        vals_pop  = group['pop'].values.astype(float)
        vals_empl = group['empl'].values.astype(float)
        bfs_int = int(bfs_id) if pd.notna(bfs_id) else None

        if bfs_int is not None and bfs_int in commune_pop:
            target_pop = int(commune_pop[bfs_int])
        else:
            target_pop = int(round(float(vals_pop.sum())))

        if bfs_int is not None and bfs_int in commune_empl:
            target_empl = int(commune_empl[bfs_int])
        else:
            target_empl = int(round(float(vals_empl.sum())))

        new_pop[positions]  = catchment_base.hamilton_round(vals_pop,  target_pop)
        new_empl[positions] = catchment_base.hamilton_round(vals_empl, target_empl)
        drift_pop  += abs(int(new_pop[positions].sum())  - target_pop)
        drift_empl += abs(int(new_empl[positions].sum()) - target_empl)

    out['pop']  = new_pop
    out['empl'] = new_empl
    return out, drift_pop, drift_empl


def _load_commune_name_lookup() -> dict:
    """Return dict[int BFS_NR → str commune name] from MUNICIPAL_BOUNDARIES_GPKG."""
    try:
        muni = gpd.read_file(paths.MUNICIPAL_BOUNDARIES_GPKG).to_crs(CODEBASE_CRS)
    except Exception as exc:
        print(f"    WARNING: failed to load commune-name lookup: {exc}")
        return {}
    if 'objektart' in muni.columns:
        muni = muni[muni['objektart'] == 'Gemeindegebiet']
    bfs_col  = next((c for c in ['BFS_NR', 'bfs_nr', 'BFS_NUMMER', 'bfs_nummer',
                                   'GMDNR', 'gmdnr'] if c in muni.columns), None)
    name_col = next((c for c in ['NAME', 'name', 'GMDNAME', 'gmdname',
                                   'GEMEINDENAME'] if c in muni.columns), None)
    if not (bfs_col and name_col):
        return {}
    muni[bfs_col] = pd.to_numeric(muni[bfs_col], errors='coerce').astype('Int64')
    muni = muni.dropna(subset=[bfs_col])
    return dict(zip(muni[bfs_col].astype(int), muni[name_col].astype(str)))


def _enrich_breakdown(per_pair: pd.DataFrame, rail_stations,
                        commune_pop: dict, commune_empl: dict,
                        muni_name_lookup: dict) -> pd.DataFrame:
    """Enrich the raw [BFS_NR, station_id, pop, empl] breakdown with commune
    and station names, commune totals, and share percentages.

    Column renames:
      station_id → station_number  (rail_stops "Number" field, also id_point)
      pop        → pop_in_station
      empl       → empl_in_station

    Returns DataFrame with columns:
      BFS_NR, commune_name, station_number, station_name,
      pop_in_station, empl_in_station,
      pop_total_commune, empl_total_commune,
      pop_share_pct, empl_share_pct
    """
    df = per_pair.copy()
    df['BFS_NR']     = df['BFS_NR'].astype(int)
    df['station_id'] = df['station_id'].astype(int)

    # Station name from rail_stops.gpkg's `stop_name` (Number → id_point match).
    # rail_stations['id_point'] is stored as str (from GTFS parent-station ID)
    # but df['station_id'] is cast to int — coerce the lookup keys to match.
    _name_series = rail_stations.set_index(
        rail_stations['id_point'].astype(int)
    )['stop_name'].astype(str)
    name_map = _name_series.to_dict()
    df['station_name'] = df['station_id'].map(name_map).fillna(
        df['station_id'].apply(lambda s: 'No PT' if s == NO_PT_ID else f'Station {s}')
    )

    # Commune name lookup
    df['commune_name'] = df['BFS_NR'].map(muni_name_lookup).fillna('—')

    # Commune totals (fallback to in-data per-commune sums when missing from summary)
    df['pop_total_commune']  = df['BFS_NR'].map(commune_pop).astype('Int64')
    df['empl_total_commune'] = df['BFS_NR'].map(commune_empl).astype('Int64')
    fallback_pop  = df.groupby('BFS_NR')['pop'].transform('sum').round().astype(int)
    fallback_empl = df.groupby('BFS_NR')['empl'].transform('sum').round().astype(int)
    df['pop_total_commune']  = df['pop_total_commune'].fillna(fallback_pop).astype(int)
    df['empl_total_commune'] = df['empl_total_commune'].fillna(fallback_empl).astype(int)

    df['pop_share_pct'] = np.where(
        df['pop_total_commune'] > 0,
        (df['pop'] / df['pop_total_commune'] * 100.0).round(2),
        0.0,
    )
    df['empl_share_pct'] = np.where(
        df['empl_total_commune'] > 0,
        (df['empl'] / df['empl_total_commune'] * 100.0).round(2),
        0.0,
    )

    df = df.rename(columns={
        'station_id': 'station_number',
        'pop':        'pop_in_station',
        'empl':       'empl_in_station',
    })

    cols = ['BFS_NR', 'commune_name', 'station_number', 'station_name',
            'pop_in_station', 'empl_in_station',
            'pop_total_commune', 'empl_total_commune',
            'pop_share_pct', 'empl_share_pct']
    return df[cols].sort_values(['BFS_NR', 'station_number']).reset_index(drop=True)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted median of `values` using `weights`."""
    sorted_idx = np.argsort(values)
    sv = values[sorted_idx]
    sw = weights[sorted_idx].astype(float)
    cum = np.cumsum(sw)
    if cum[-1] == 0:
        return float('nan')
    cum /= cum[-1]
    return float(np.interp(0.5, cum, sv))


def _build_mode_by_station_sheet(alloc_combined, pop_grid, empl_grid,
                                   sa_boundary, sa_stations, sa_ids) -> pd.DataFrame:
    """Build the Mode_access_by_station sheet: Pop/FTE by mode per SA station.

    Returns a DataFrame with columns: station_name, station_number, access_mode,
    pop, pop_pct, fte, fte_pct, median_access_min_pop, median_access_min_fte.
    Sorted by (station_name, access_mode). Returns None when no data.
    """
    sa_cells = _build_sa_cells(alloc_combined, pop_grid, empl_grid, sa_boundary)
    cells = sa_cells[sa_cells['station_id'].isin(sa_ids)].copy()
    if cells.empty:
        return None

    name_map = dict(zip(sa_stations['id_point'].astype(int),
                        sa_stations['stop_name'].astype(str)))

    records = []
    for (sid, mode), grp in cells.groupby(['station_id', 'access_mode']):
        pop_vals  = grp['pop'].values.astype(int)
        fte_vals  = grp['fte'].values.astype(int)
        times_min = grp['access_time_min'].values.astype(float)
        pop_total = int(pop_vals.sum())
        fte_total = int(fte_vals.sum())
        med_pop = _weighted_median(times_min, pop_vals) if pop_total > 0 else float('nan')
        med_fte = _weighted_median(times_min, fte_vals) if fte_total > 0 else float('nan')
        records.append({
            'station_name':          name_map.get(int(sid), str(sid)),
            'station_number':        int(sid),
            'access_mode':           mode,
            'pop':                   pop_total,
            'fte':                   fte_total,
            'median_access_min_pop': round(med_pop, 2) if not np.isnan(med_pop) else None,
            'median_access_min_fte': round(med_fte, 2) if not np.isnan(med_fte) else None,
        })

    df = pd.DataFrame(records)
    sta_pop  = df.groupby('station_number')['pop'].transform('sum')
    sta_fte  = df.groupby('station_number')['fte'].transform('sum')
    df['pop_pct'] = np.where(sta_pop  > 0, (df['pop'] / sta_pop  * 100).round(2), 0.0)
    df['fte_pct'] = np.where(sta_fte  > 0, (df['fte'] / sta_fte  * 100).round(2), 0.0)
    df = df[['station_name', 'station_number', 'access_mode',
             'pop', 'pop_pct', 'fte', 'fte_pct',
             'median_access_min_pop', 'median_access_min_fte']]
    return df.sort_values(['station_name', 'access_mode']).reset_index(drop=True)


def _compute_variant_freq_lookup() -> dict:
    """Aggregate per-variant frequency + service_period across all temporal
    subfolders (cross-period view).

    Uses `_load_lines_all_periods` to read All_Day + Peak + Off_Peak files
    with their canonical service_period filters (All_Day no filter, Peak →
    peak_only, Off_Peak → offpeak_only). Cross-subfolder duplicates are then
    removed on (route_id, variant_rank, direction_id) so each variant-
    direction row contributes once. The CLI `settings.TEMPORAL` does NOT
    scope this helper — until a concrete use case justifies per-temporal
    variant data here, every sheet that consumes line frequencies sees the
    full cross-period view.

    Per (route_id, variant_rank) aggregation:
      - freq_am / freq_pm / freq_op : mean of freq_{am,pm,op}_peak_dep_hr /
        freq_offpeak_dep_hr across direction rows.
      - total_dep : sum across direction rows over the ARE 06:00-20:00 window.
      - service_period : most common non-placeholder tag across the rows;
        falls back to 'unknown' only when no row carried a recognised tag.
      - freq_per_h : per-variant period-specific dep/h:
            peak_only            -> max(freq_am, freq_pm)
            offpeak_only / all_day -> freq_op

    Returns dict[(route_id_str, variant_rank_int)] -> dict of the columns
    above. Empty dict when no lines files are available.
    """
    f = _load_lines_all_periods(_FEEDER_BASE, 'pt_feeder_lines')
    if f is None or f.empty:
        return {}
    f['route_id_str']   = f['route_id'].astype(str)
    f['variant_rank_i'] = pd.to_numeric(f['variant_rank'], errors='coerce')
    f = f.dropna(subset=['variant_rank_i'])
    f['variant_rank_i'] = f['variant_rank_i'].astype(int)
    for col in ('freq_am_peak_dep_hr', 'freq_pm_peak_dep_hr',
                'freq_offpeak_dep_hr', 'total_dep'):
        if col in f.columns:
            f[col] = pd.to_numeric(f[col], errors='coerce').fillna(0.0)
        else:
            f[col] = 0.0
    if 'service_period' not in f.columns:
        f['service_period'] = ''
    f['service_period'] = f['service_period'].fillna('').astype(str)
    # Cross-subfolder dedupe: identical (route_id, variant_rank, direction_id)
    # rows from All_Day + Peak / Off_Peak would otherwise inflate `total_dep`
    if 'direction_id' in f.columns:
        f['direction_id_i'] = pd.to_numeric(f['direction_id'], errors='coerce')
        f = f.dropna(subset=['direction_id_i'])
        f['direction_id_i'] = f['direction_id_i'].astype(int)
        f = f.drop_duplicates(subset=['route_id_str', 'variant_rank_i', 'direction_id_i'],
                                keep='first')

    _PLACEHOLDERS = {'', 'nan', 'none', 'unknown'}
    def _sp_mode(s):
        vals = [v for v in s.tolist() if v and v.lower() not in _PLACEHOLDERS]
        return max(set(vals), key=vals.count) if vals else 'unknown'

    agg = (f.groupby(['route_id_str', 'variant_rank_i'])
            .agg(freq_am=('freq_am_peak_dep_hr', 'mean'),
                 freq_pm=('freq_pm_peak_dep_hr', 'mean'),
                 freq_op=('freq_offpeak_dep_hr', 'mean'),
                 total_dep=('total_dep', 'sum'),
                 service_period=('service_period', _sp_mode))
            .reset_index())

    def _per_period_freq(row):
        if row['service_period'] == 'peak_only':
            return max(float(row['freq_am'] or 0.0), float(row['freq_pm'] or 0.0))
        return float(row['freq_op'] or 0.0)
    agg['freq_per_h'] = agg.apply(_per_period_freq, axis=1)

    return agg.set_index(['route_id_str', 'variant_rank_i']).to_dict('index')


def _build_lines_per_station_sheet(feeder_segments, feeder_stops, feeder_graph,
                                     sa_stations, pop_grid, empl_grid) -> pd.DataFrame:
    """Build the Lines_per_station sheet: PT-feeder lines/variants that fulfil
    rail-station walk-access for SA rail stations.

    Each row corresponds to a unique (line_short_name, variant_rank). A line
    with two variant_ranks therefore produces two rows. When the variant
    serves more than one SA rail station the names are comma-separated in
    `rail_stations_served`.

    Columns
    -------
    line_short_name      : str
    variant_rank         : int
    rail_stations_served : str (comma-separated SA rail station names)
    travel_time_ivt_min  : float — sum of segment travel_time_min along the
                           variant, averaged across the two directions when
                           both exist (single value when circular / one-way)
    n_stops              : int — unique stops on the variant (union of dirs)
    pop_accessible       : int — Pop in pop_grid cells within any of the
                           variant's stops' Güteklassen buffer_radius_m,
                           de-duplicated by RELI
    fte_accessible       : int — same for empl_grid
    service_period       : str — 'all_day' / 'peak_only' / 'offpeak_only';
                           per-variant tag from services_network_builder
    freq_per_h           : float — dep/h of the window relevant to the
                           variant: peak_only → max(AM peak, PM peak);
                           offpeak_only & all_day → off-peak. Read directly
                           from freq_am_peak_dep_hr / freq_pm_peak_dep_hr /
                           freq_offpeak_dep_hr in the All_Day lines file;
                           mean across directions
    total_dep_06_20      : int — sum across directions in the ARE 06:00–20:00
                           window from the All_Day lines file

    "Serves" is defined by the walk-entry edges in `feeder_graph`: a variant
    serves rail station R if any of its split nodes has a walk arc to R
    (i.e., one of its stops falls within the 350 m / named-Bahnhof rule).
    Returns None when no variant serves any SA rail station.
    """
    if sa_stations.empty or feeder_graph is None:
        return None
    sa_rail_ids = set(sa_stations['stop_id'].astype(str))
    rail_name_map = dict(zip(
        sa_stations['stop_id'].astype(str),
        sa_stations['stop_name'].astype(str),
    ))

    # 1. From the graph: which variants walk-access which SA rail stations
    variant_rails = {}   # (route_id, variant_rank) -> set of SA rail_station_ids
    for u, v, data in feeder_graph.edges(data=True):
        if data.get('component') != 'walk':
            continue
        if not (isinstance(u, tuple) and isinstance(v, str)):
            continue
        if v not in sa_rail_ids:
            continue
        _, route_id, _, var_rank = u
        variant_rails.setdefault((route_id, var_rank), set()).add(v)
    if not variant_rails:
        return None

    # 2. (route_id, variant_rank) -> line_short_name
    seg = feeder_segments.copy()
    seg['route_id_str']   = seg['route_id'].astype(str)
    seg['variant_rank_i'] = pd.to_numeric(seg['variant_rank'], errors='coerce').astype('Int64')
    seg['direction_id_i'] = pd.to_numeric(seg['direction_id'], errors='coerce').astype('Int64')
    line_name_lookup = (seg.dropna(subset=['variant_rank_i'])
                          .drop_duplicates(subset=['route_id_str', 'variant_rank_i'])
                          .set_index(['route_id_str', 'variant_rank_i'])['line_short_name']
                          .to_dict())

    # 3. Per-variant: unique stops; per-(variant, direction) IVT sum
    variant_stops   = {}   # (route_id, var) -> set of stop_ids
    variant_dir_ivt = {}   # (route_id, var, dir) -> minutes
    for _, r in seg.iterrows():
        vr = r['variant_rank_i']
        di = r['direction_id_i']
        if pd.isna(vr):
            continue
        key_v = (r['route_id_str'], int(vr))
        variant_stops.setdefault(key_v, set()).add(str(r['from_stop_id']))
        variant_stops[key_v].add(str(r['to_stop_id']))
        if pd.notna(di) and pd.notna(r['travel_time_min']):
            key_d = (r['route_id_str'], int(vr), int(di))
            variant_dir_ivt[key_d] = (variant_dir_ivt.get(key_d, 0.0)
                                       + float(r['travel_time_min']))

    # 4. Frequencies per variant via the shared helper
    freq_per_v = _compute_variant_freq_lookup()

    # 5. Stop-level Pop/FTE catchment via Güteklassen buffer_radius_m
    needed_stops = set().union(*(variant_stops.get(k, set()) for k in variant_rails))
    fs = feeder_stops.drop_duplicates(subset='stop_id', keep='first').copy()
    fs['stop_id'] = fs['stop_id'].astype(str)
    fs_idx = fs.set_index('stop_id')

    def _build_lookup(grid):
        coords = np.column_stack([grid.geometry.x, grid.geometry.y])
        relis  = grid['RELI'].astype(int).values
        nums   = grid['NUMMER'].astype(int).values
        tree   = cKDTree(coords) if len(coords) else None
        reli_to_val = dict(zip(relis.tolist(), nums.tolist()))
        return tree, relis, reli_to_val

    pop_tree, pop_relis, pop_lookup = _build_lookup(pop_grid)
    fte_tree, fte_relis, fte_lookup = _build_lookup(empl_grid)

    stop_pop_relis = {}
    stop_fte_relis = {}
    for sid in needed_stops:
        if sid not in fs_idx.index:
            continue
        row = fs_idx.loc[sid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        radius = float(row['buffer_radius_m']) if pd.notna(row['buffer_radius_m']) else 0.0
        if radius <= 0:
            continue
        pt = (row.geometry.x, row.geometry.y)
        if pop_tree is not None:
            idxs = pop_tree.query_ball_point(pt, r=radius)
            stop_pop_relis[sid] = {int(pop_relis[j]) for j in idxs}
        if fte_tree is not None:
            idxs = fte_tree.query_ball_point(pt, r=radius)
            stop_fte_relis[sid] = {int(fte_relis[j]) for j in idxs}

    # 6. Compose rows
    rows = []
    for (route_id, var_rank), rail_set in variant_rails.items():
        line_name = line_name_lookup.get((route_id, var_rank))
        if pd.isna(line_name) or line_name is None:
            line_name = str(route_id)
        stops = variant_stops.get((route_id, var_rank), set())

        ivt_vals = [variant_dir_ivt[(route_id, var_rank, d)]
                    for d in (0, 1)
                    if (route_id, var_rank, d) in variant_dir_ivt]
        ivt_min = float(np.mean(ivt_vals)) if ivt_vals else float('nan')

        pop_union = set().union(*(stop_pop_relis.get(s, set()) for s in stops)) if stops else set()
        fte_union = set().union(*(stop_fte_relis.get(s, set()) for s in stops)) if stops else set()
        pop_total = int(sum(pop_lookup.get(r, 0) for r in pop_union))
        fte_total = int(sum(fte_lookup.get(r, 0) for r in fte_union))

        fk = freq_per_v.get((route_id, var_rank))
        if fk:
            sp     = str(fk.get('service_period', 'unknown'))
            freq_h = float(fk.get('freq_per_h', 0.0) or 0.0)
            dep_w  = int(fk.get('total_dep', 0) or 0)
        else:
            sp     = 'unknown'
            freq_h = float('nan')
            dep_w  = 0

        rail_names = sorted({rail_name_map.get(r, r) for r in rail_set})

        rows.append({
            'line_short_name':       str(line_name),
            'variant_rank':          int(var_rank),
            'rail_stations_served':  ', '.join(rail_names),
            'travel_time_ivt_min':   round(ivt_min, 2) if not np.isnan(ivt_min) else None,
            'n_stops':               len(stops),
            'pop_accessible':        pop_total,
            'fte_accessible':        fte_total,
            'service_period':        sp,
            'freq_per_h':            round(freq_h, 2) if not np.isnan(freq_h) else None,
            'total_dep_06_20':       dep_w,
        })

    df = pd.DataFrame(rows)
    return df.sort_values(['line_short_name', 'variant_rank']).reset_index(drop=True)


def _build_pt_stops_sa_sheet(feeder_stops, feeder_segments, feeder_graph,
                               sa_boundary, sa_stations) -> pd.DataFrame:
    """Build the PT-stops_SA sheet: PT-feeder stops within the SA boundary.

    Columns
    -------
    stop_id              : str — BAV / GTFS numeric identifier
    stop_name            : str
    municipality         : str — commune name via spatial join with
                           paths.MUNICIPAL_BOUNDARIES_GPKG (predicate=within)
    gueteklasse          : str — ARE ÖV-Güteklasse class at the stop centre
                           (A / B / C / D), derived from `hst_kat` via the
                           (Kategorie, distance-band-0) lookup. '-' if the
                           stop has no class (hst_kat == 0).
    lines                : str — comma-separated unique line_short_names
                           found in feeder_segments where this stop appears
                           as from_stop or to_stop
    direct_to_station    : bool — True iff this stop has a no-transfer path
                           (segments + final alighting walk) to at least one
                           SA rail station. Computed via reverse BFS from
                           every SA rail node on the graph, excluding all
                           transfer arcs.
    dep_per_h            : float — sum of the period-specific `freq_per_h` of
                           every (route_id, variant_rank) visiting the stop,
                           using the same Lines_per_station rule per variant
                           (peak_only → max(AM, PM); offpeak_only / all_day
                           → off-peak)
    wait_min             : float — cp.t_wait_min(60 / dep_per_h); piecewise
                           random-arrival (h/2) below 12 min, schedule-
                           anchored (6 + 0.25*(h-12)) above

    Returns None when no SA stops or no SA boundary.
    """
    if sa_boundary is None or feeder_stops is None or feeder_stops.empty:
        return None

    # 1. SA-clip feeder stops
    prep_sa = prep(sa_boundary)
    fs = feeder_stops.copy()
    fs['stop_id'] = fs['stop_id'].astype(str)
    fs = fs.drop_duplicates(subset='stop_id', keep='first').copy()
    fs['_in_sa'] = fs.geometry.apply(prep_sa.contains)
    fs = fs[fs['_in_sa']].drop(columns='_in_sa').copy()
    if fs.empty:
        return None

    # 2. Municipality via point-in-polygon sjoin
    stop_muni = {}
    try:
        muni = gpd.read_file(paths.MUNICIPAL_BOUNDARIES_GPKG).to_crs(CODEBASE_CRS)
        if 'objektart' in muni.columns:
            muni = muni[muni['objektart'] == 'Gemeindegebiet']
        name_col = next((c for c in ['NAME', 'name', 'GMDNAME', 'gmdname',
                                       'GEMEINDENAME'] if c in muni.columns), None)
        if name_col:
            joined = gpd.sjoin(fs[['stop_id', 'geometry']],
                                muni[[name_col, 'geometry']],
                                predicate='within', how='left')
            joined = joined.drop_duplicates(subset='stop_id', keep='first')
            stop_muni = dict(zip(joined['stop_id'].astype(str),
                                  joined[name_col].fillna('').astype(str)))
    except Exception as exc:
        print(f"    WARNING: PT-stops_SA municipality join failed: {exc}")

    # 3. Lines per stop, and the set of (route_id, variant_rank) visiting it
    stop_lines_map = {}        # stop_id -> set of line_short_name strings
    stop_variants_map = {}     # stop_id -> set of (route_id_str, variant_rank_int)
    if feeder_segments is not None and not feeder_segments.empty:
        for _, r in feeder_segments.iterrows():
            line = r.get('line_short_name')
            if pd.notna(line):
                line_s = str(line)
                stop_lines_map.setdefault(str(r['from_stop_id']), set()).add(line_s)
                stop_lines_map.setdefault(str(r['to_stop_id']),   set()).add(line_s)
            rid = r.get('route_id')
            vrk = r.get('variant_rank')
            if pd.notna(rid) and pd.notna(vrk):
                vk = (str(rid), int(vrk))
                stop_variants_map.setdefault(str(r['from_stop_id']), set()).add(vk)
                stop_variants_map.setdefault(str(r['to_stop_id']),   set()).add(vk)

    # 4. Direct-to-station: multi-source reverse BFS from all SA rail nodes,
    # excluding transfer arcs. Visits every variant-split node that can reach
    # a rail station with no line change.
    direct_stops = set()
    sa_rail_ids = (set(sa_stations['stop_id'].astype(str))
                   if sa_stations is not None and not sa_stations.empty else set())
    if feeder_graph is not None and sa_rail_ids:
        seeds = sa_rail_ids & set(feeder_graph.nodes)
        visited = set(seeds)
        stack = list(seeds)
        while stack:
            u = stack.pop()
            for pred in feeder_graph.predecessors(u):
                if feeder_graph[pred][u].get('component') == 'transfer':
                    continue
                if pred not in visited:
                    visited.add(pred)
                    stack.append(pred)
        for n in visited:
            if isinstance(n, tuple) and len(n) == 4:
                direct_stops.add(n[0])

    # 5. Per-variant freq lookup (period-specific) + Kategorie lookup
    freq_per_v = _compute_variant_freq_lookup()
    kat_lookup = dict(zip(
        feeder_stops['stop_id'].astype(str),
        pd.to_numeric(feeder_stops.get('hst_kat'), errors='coerce').fillna(0).astype(int),
    )) if 'hst_kat' in feeder_stops.columns else {}

    # 6. Compose rows
    rows = []
    for _, r in fs.iterrows():
        sid = r['stop_id']
        lines = sorted(stop_lines_map.get(sid, set()))
        # dep_per_h = Σ freq_per_h across every (route_id, variant_rank) visiting
        # this stop. Each variant's freq_per_h is the period-specific value
        # produced by `_compute_variant_freq_lookup` (peak_only → max(AM, PM);
        # offpeak_only / all_day → off-peak).
        dep_h = sum(float(freq_per_v.get(vk, {}).get('freq_per_h', 0.0) or 0.0)
                    for vk in stop_variants_map.get(sid, set()))
        # Implicit headway from the per-line sum; wait via the same piecewise
        # t_wait_min so stop-level wait is consistent with the dep_per_h figure
        h_implicit = (60.0 / dep_h) if dep_h > 0 else float('inf')
        wait  = cp.t_wait_min(h_implicit) if np.isfinite(h_implicit) else 0.0
        kat   = kat_lookup.get(sid, 0)
        gk    = _GK_LOOKUP.get((kat, 0), '-') if kat > 0 else '-'
        rows.append({
            'stop_id':           sid,
            'stop_name':         str(r.get('stop_name') or sid),
            'municipality':      stop_muni.get(sid, '') or '',
            'gueteklasse':       gk,
            'lines':             ', '.join(lines),
            'direct_to_station': bool(sid in direct_stops),
            'dep_per_h':         round(float(dep_h), 2),
            'wait_min':          round(float(wait),  2),
        })

    df = pd.DataFrame(rows)
    return df.sort_values(['municipality', 'stop_name']).reset_index(drop=True)


def _write_station_catchments_xlsx(output_dir: str, breakdown: pd.DataFrame,
                                     rail_stations,
                                     method_label: str = '',
                                     alloc_combined=None,
                                     pop_grid=None,
                                     empl_grid=None,
                                     feeder_segments=None,
                                     feeder_stops=None,
                                     feeder_graph=None) -> tuple:
    """Write station_catchments.xlsx with up to five sheets:
      Stations_Summary       — one row per station (pop, empl, n_communes, municipalities)
                               — covers ALL stations in the breakdown.
      Communes_by_station    — long-format breakdown filtered to SA stations, sorted by commune.
                               Includes shares as % of commune total AND % of station total.
      Mode_access_by_station — PT-Feeder only (when alloc_combined supplied): absolute Pop/FTE
                               per access mode per SA station, with % and pop/FTE-weighted
                               median access time.
      Lines_per_station      — PT-Feeder only (when feeder_segments/stops/graph supplied):
                               one row per (line_short_name, variant_rank) that walk-accesses
                               at least one SA rail station, with travel-time, n_stops,
                               Pop/FTE catchment, and frequency / total_dep.
      PT-stops_SA            — PT-Feeder only (when feeder_segments/stops/graph supplied):
                               one row per PT-feeder stop inside the SA boundary, with
                               municipality, lines serving it, direct-to-station flag,
                               dep/h and expected wait minutes.

    Args:
        output_dir:      Output directory for the workbook.
        breakdown:       Enriched per-(commune, station) DataFrame.
        rail_stations:   Full rail stations GeoDataFrame.
        method_label:    Optional label for the console message.
        alloc_combined:  Per-cell allocation DataFrame (PT-Feeder only) — enables Sheet 3.
        pop_grid:        Population grid (PT-Feeder only).
        empl_grid:       Employment grid (PT-Feeder only).
        feeder_segments: PT-feeder segments DataFrame (PT-Feeder only) — enables Sheets 4 & 5.
        feeder_stops:    PT-feeder stops GeoDataFrame (PT-Feeder only) — enables Sheets 4 & 5.
        feeder_graph:    Directed feeder NetworkX graph (PT-Feeder only) — enables Sheets 4 & 5.

    Returns (output_path, stations_summary_df).
    """
    if breakdown is None or breakdown.empty:
        print("    Breakdown is empty — skipping station_catchments.xlsx")
        return None, pd.DataFrame()

    # Stations_Summary: aggregate per station (all stations in breakdown)
    summary = breakdown.groupby(
        ['station_number', 'station_name'], as_index=False, sort=False
    ).agg(
        pop=('pop_in_station',  'sum'),
        empl=('empl_in_station', 'sum'),
        n_communes=('BFS_NR', 'nunique'),
        municipalities=('commune_name',
                        lambda x: ', '.join(sorted(set(str(c) for c in x if pd.notna(c) and c != '—')))),
    )
    summary['pop']  = summary['pop'].astype(int)
    summary['empl'] = summary['empl'].astype(int)
    summary = summary.sort_values('pop', ascending=False).reset_index(drop=True)

    # Communes_by_station: add station-total share columns, filter to SA, sort
    enriched = breakdown.copy()
    station_pop_total  = enriched.groupby('station_number')['pop_in_station'].transform('sum')
    station_empl_total = enriched.groupby('station_number')['empl_in_station'].transform('sum')
    enriched['pop_share_station_pct'] = np.where(
        station_pop_total > 0,
        (enriched['pop_in_station'] / station_pop_total * 100.0).round(2),
        0.0,
    )
    enriched['empl_share_station_pct'] = np.where(
        station_empl_total > 0,
        (enriched['empl_in_station'] / station_empl_total * 100.0).round(2),
        0.0,
    )

    # SA filter via paths.STUDY_AREA_BOUNDARY_GPKG
    sa_boundary  = _load_sa_boundary()
    sa_stations  = _filter_stations_to_sa(rail_stations, sa_boundary)
    sa_ids       = set(sa_stations['id_point'].astype(int)) if not sa_stations.empty else set()
    if sa_ids:
        communes_sheet = enriched[enriched['station_number'].isin(sa_ids)].copy()
    else:
        print("    WARNING: SA filter empty — Communes_by_station will be empty.")
        communes_sheet = enriched.iloc[0:0].copy()

    communes_sheet = communes_sheet[[
        'commune_name', 'BFS_NR', 'station_number', 'station_name',
        'pop_in_station', 'pop_total_commune',
        'pop_share_pct', 'pop_share_station_pct',
        'empl_in_station', 'empl_total_commune',
        'empl_share_pct', 'empl_share_station_pct',
    ]].sort_values(['commune_name', 'station_number']).reset_index(drop=True)

    # Mode_access_by_station (PT-Feeder only, when allocation data provided)
    mode_sheet = None
    if alloc_combined is not None and pop_grid is not None and empl_grid is not None and sa_ids:
        mode_sheet = _build_mode_by_station_sheet(
            alloc_combined, pop_grid, empl_grid, sa_boundary, sa_stations, sa_ids)

    # Lines_per_station (PT-Feeder only, when graph + segments + stops supplied)
    lines_sheet = None
    if (feeder_segments is not None and feeder_stops is not None
            and feeder_graph is not None and pop_grid is not None
            and empl_grid is not None and not sa_stations.empty):
        lines_sheet = _build_lines_per_station_sheet(
            feeder_segments, feeder_stops, feeder_graph,
            sa_stations, pop_grid, empl_grid)

    # PT-stops_SA (PT-Feeder only)
    pt_stops_sheet = None
    if (feeder_segments is not None and feeder_stops is not None
            and feeder_graph is not None and sa_boundary is not None):
        pt_stops_sheet = _build_pt_stops_sa_sheet(
            feeder_stops, feeder_segments, feeder_graph, sa_boundary, sa_stations)

    out_path = os.path.join(output_dir, 'station_catchments.xlsx')
    os.makedirs(output_dir, exist_ok=True)
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        summary.to_excel(writer, sheet_name='Stations_Summary', index=False)
        communes_sheet.to_excel(writer, sheet_name='Communes_by_station', index=False)
        if mode_sheet is not None:
            mode_sheet.to_excel(writer, sheet_name='Mode_access_by_station', index=False)
        if lines_sheet is not None:
            lines_sheet.to_excel(writer, sheet_name='Lines_per_station', index=False)
        if pt_stops_sheet is not None:
            pt_stops_sheet.to_excel(writer, sheet_name='PT-stops_SA', index=False)

    # Full per-(commune, station) breakdown for ALL stations (not SA-filtered).
    # Consumed by catchment_OD_preparation.py: pop_share_pct / empl_share_pct
    # are the origin / destination weights for transplanting communal OD onto
    # rail stations, so the OD attribution matches these catchment shares exactly.
    breakdown_csv = os.path.join(output_dir, 'station_commune_breakdown.csv')
    breakdown.to_csv(breakdown_csv, index=False, encoding='utf-8-sig')
    print(f"    Station-commune breakdown ({len(breakdown):,} rows, all stations)"
          f" saved -> {breakdown_csv}")

    suffix = f" ({method_label})" if method_label else ""
    mode_info     = f", Mode_access_by_station: {len(mode_sheet)} rows"     if mode_sheet     is not None else ""
    lines_info    = f", Lines_per_station: {len(lines_sheet)} rows"         if lines_sheet    is not None else ""
    pt_stops_info = f", PT-stops_SA: {len(pt_stops_sheet)} rows"            if pt_stops_sheet is not None else ""
    print(f"    Station catchments Excel{suffix} saved -> {out_path}  "
          f"(Stations_Summary: {len(summary)} rows, "
          f"Communes_by_station: {len(communes_sheet)} rows, SA stations: {len(sa_ids)}"
          f"{mode_info}{lines_info}{pt_stops_info})")
    return out_path, summary


def _compute_station_commune_breakdown_pt_feeder(
        allocation, pop_grid, empl_grid, rail_stations) -> pd.DataFrame:
    """Build the enriched per-(commune, station) Pop/FTE breakdown for PT-Feeder.

    Spatially joins cells to communes, sums Pop and FTE per (BFS_NR, id_point),
    applies Hamilton's largest-remainder rounding against per-commune integer
    targets from `municipal_pop_empl_summary.csv` so per-commune sums reconcile
    exactly, then enriches with names + share columns via `_enrich_breakdown`.

    NO_PT cells (id_point = NO_PT_ID) are kept as a pseudo-station so the
    per-commune Pop/FTE sums reconcile to the commune integer total even when
    parts of the commune are outside the PT-feeder reach.
    """
    print("  Computing per-(commune, station) Pop/FTE breakdown (PT-Feeder) ...")

    muni = gpd.read_file(paths.MUNICIPAL_BOUNDARIES_GPKG).to_crs(CODEBASE_CRS)
    if 'objektart' in muni.columns:
        muni = muni[muni['objektart'] == 'Gemeindegebiet'].copy()
    bfs_col = next((c for c in ['BFS_NR', 'bfs_nr', 'BFS_NUMMER', 'bfs_nummer',
                                 'GMDNR', 'gmdnr'] if c in muni.columns), None)
    if bfs_col is None:
        print("    WARNING: no BFS column found — skipping breakdown")
        return pd.DataFrame()
    muni[bfs_col] = pd.to_numeric(muni[bfs_col], errors='coerce').astype('Int64')
    muni = muni.dropna(subset=[bfs_col])[[bfs_col, 'geometry']].copy()

    pop_renamed  = pop_grid[['RELI', 'NUMMER', 'geometry']].rename(columns={'NUMMER': 'pop'})
    empl_renamed = empl_grid[['RELI', 'NUMMER']].rename(columns={'NUMMER': 'empl'})
    cells = pop_renamed.merge(empl_renamed, on='RELI', how='outer')
    cells['pop']  = cells['pop'].fillna(0.0)
    cells['empl'] = cells['empl'].fillna(0.0)
    missing_geom_mask = cells['geometry'].isna()
    if missing_geom_mask.any() and 'geometry' in empl_grid.columns:
        empl_geom = empl_grid.set_index('RELI')['geometry']
        cells.loc[missing_geom_mask, 'geometry'] = (
            cells.loc[missing_geom_mask, 'RELI'].map(empl_geom))
    cells = cells.dropna(subset=['geometry'])
    cells_gdf = gpd.GeoDataFrame(cells, geometry='geometry', crs=CODEBASE_CRS)

    joined = gpd.sjoin(cells_gdf, muni, how='left', predicate='within')
    joined = joined.rename(columns={bfs_col: 'BFS_NR'})
    joined = joined[['RELI', 'BFS_NR', 'pop', 'empl']].dropna(subset=['BFS_NR']).copy()
    joined['BFS_NR'] = joined['BFS_NR'].astype(int)

    alloc = allocation[['RELI', 'id_point']].drop_duplicates('RELI').copy()
    joined = joined.merge(alloc, on='RELI', how='left')
    joined['id_point'] = joined['id_point'].fillna(NO_PT_ID).astype(int)

    per_pair = joined.groupby(['BFS_NR', 'id_point'], as_index=False).agg(
        pop=('pop', 'sum'), empl=('empl', 'sum')
    ).rename(columns={'id_point': 'station_id'})

    commune_pop, commune_empl = _commune_targets_from_summary()
    per_pair, drift_pop, drift_empl = _apply_hamilton_per_commune(
        per_pair, commune_pop, commune_empl)
    print(f"    Hamilton reconciliation: pop drift = {drift_pop}, empl drift = {drift_empl} "
          f"(must be 0 for exact commune reconciliation)")

    muni_name_lookup = _load_commune_name_lookup()
    return _enrich_breakdown(per_pair, rail_stations, commune_pop, commune_empl,
                              muni_name_lookup)


def _compute_station_commune_breakdown_municipal(
        assignment_df, rail_stations) -> pd.DataFrame:
    """Build the enriched per-(commune, station) breakdown for the Municipal method.

    Each commune is assigned wholly to one station, so the breakdown has one
    row per commune with pop_share_pct = empl_share_pct = 100% by construction.
    """
    print("  Computing per-(commune, station) Pop/FTE breakdown (Municipal) ...")

    commune_pop, commune_empl = _commune_targets_from_summary()

    df = assignment_df[['BFS_NR', 'station_id']].copy()
    df['BFS_NR']     = pd.to_numeric(df['BFS_NR'], errors='coerce').astype('Int64')
    df['station_id'] = pd.to_numeric(df['station_id'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['BFS_NR', 'station_id'])
    df['BFS_NR']     = df['BFS_NR'].astype(int)
    df['station_id'] = df['station_id'].astype(int)
    df['pop']  = df['BFS_NR'].map(commune_pop).fillna(0).astype(int)
    df['empl'] = df['BFS_NR'].map(commune_empl).fillna(0).astype(int)
    print(f"    Municipal breakdown (1:1 commune→station): "
          f"{len(df)} pairs, {df['BFS_NR'].nunique()} communes")

    muni_name_lookup = _load_commune_name_lookup()
    return _enrich_breakdown(df, rail_stations, commune_pop, commune_empl,
                              muni_name_lookup)


def _build_catchment_gpkg(allocation, pop_grid, empl_grid, rail_stations,
                            output_dir, breakdown: pd.DataFrame = None):
    """Dissolve cells per station into catchment polygons.

    Schema: [train_station, id, station_name, pop, empl, geometry].

    Per-station Pop and FTE come from the Hamilton-reconciled per-(commune,
    station) breakdown. If `breakdown` is None, the PT-Feeder breakdown is
    computed inline.

    Returns:
        (dissolved_gpkg_gdf, breakdown_df)
    """
    print("  Building catchment GeoPackage ...")

    if breakdown is None:
        breakdown = _compute_station_commune_breakdown_pt_feeder(
            allocation, pop_grid, empl_grid, rail_stations)

    merged = allocation.merge(pop_grid[['RELI', 'geometry']], on='RELI', how='left')
    merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=CODEBASE_CRS)
    merged['geometry'] = merged.geometry.buffer(CELL_SIZE_M / 2, cap_style=3)

    dissolved = merged.dissolve(by='id_point', as_index=False)
    dissolved = dissolved.rename(columns={'id_point': 'id'})
    dissolved['train_station'] = dissolved['id'].copy()
    dissolved['id'] = dissolved['id'].astype(int)
    dissolved['train_station'] = dissolved['train_station'].astype(int)

    # Per-station totals from the reconciled breakdown
    if not breakdown.empty:
        stats = breakdown.groupby('station_number', as_index=False).agg(
            pop=('pop_in_station',  'sum'),
            empl=('empl_in_station', 'sum'),
        )
        stats['pop']  = stats['pop'].astype(int)
        stats['empl'] = stats['empl'].astype(int)
        stats = stats.rename(columns={'station_number': 'id'})
    else:
        stats = pd.DataFrame(columns=['id', 'pop', 'empl'])

    name_map = rail_stations.set_index(
        rail_stations['id_point'].astype(int)
    )['stop_name'].astype(str).to_dict()
    dissolved['station_name'] = dissolved['id'].map(name_map).fillna('—')

    dissolved = dissolved.merge(stats[['id', 'pop', 'empl']], on='id', how='left')
    dissolved['pop']  = dissolved['pop'].fillna(0).astype(int)
    dissolved['empl'] = dissolved['empl'].fillna(0).astype(int)

    dissolved = dissolved[['train_station', 'id', 'station_name',
                            'pop', 'empl', 'geometry']].copy()

    out_path = os.path.join(output_dir, 'catchment.gpkg')
    dissolved.to_file(out_path, driver='GPKG')
    print(f"    Saved -> {out_path}  ({len(dissolved)} features)")

    return dissolved, breakdown




def _build_visualisation(allocation, grid, rail_stations, boundary, method_label,
                         empl_grid=None,
                         walk_df=None, cycle_df=None, feeder_df=None,
                         catchment_gdf=None):
    """Produce a thesis-quality catchment visualisation as two side-by-side panels.

    Left panel:  cells coloured by access mode (Walk / Bus / Tram / Cycle / No PT).
    Right panel: cells coloured by station allocation (graph-coloured catchment fills).

    A 3-column summary table (Modal Access | Hierarchical | Non-Hierarchical) is
    rendered below the figure.

    Saved to plots/Catchment_Area/PT_Feeder/.
    """
    print(f"  Building {method_label} catchment visualisation ...")

    # --- Build 100×100 m square cells from allocation coordinates ---
    alloc = allocation.copy()
    alloc['geometry'] = [
        box(e, n, e + CELL_SIZE_M, n + CELL_SIZE_M)
        for e, n in zip(alloc['E_KOORD'], alloc['N_KOORD'])
    ]
    alloc = gpd.GeoDataFrame(alloc, geometry='geometry', crs=CODEBASE_CRS)
    alloc = gpd.clip(alloc, boundary)

    mode_colors = {
        'No PT':     '#d9d9d9',
        'Walk':      '#2ca02c',
        'Bus':       '#1f77b4',
        'Tram':      '#ff7f0e',
        'Ship':      '#004B8D',
        'Funicular': '#7B3F00',
        'Cycle':     '#9467bd',
    }
    mode_labels = {
        'No PT':     'No access',
        'Walk':      'Walk to rail',
        'Bus':       'Bus feeder',
        'Tram':      'Tram feeder',
        'Ship':      'Ship feeder',
        'Funicular': 'Funicular feeder',
        'Cycle':     'Cycle to rail',
    }

    # Graph-colouring palette (shared between both panels)
    boundary_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#393b79',
        '#637939', '#8c6d31', '#843c39', '#7b4173', '#5254a3',
    ]

    # --- Build graph colouring for station catchments (right panel) ---
    color_map = {}   # id_point (int) -> hex colour
    clipped_catchment = None
    if catchment_gdf is not None and not catchment_gdf.empty:
        clipped_catchment = gpd.clip(catchment_gdf, boundary).reset_index(drop=True)
        id_col = 'id' if 'id' in clipped_catchment.columns else 'train_station'

        G = nx.Graph()
        for i in range(len(clipped_catchment)):
            G.add_node(clipped_catchment.loc[i, id_col])
        for i in range(len(clipped_catchment)):
            for j in range(i + 1, len(clipped_catchment)):
                geom_i = clipped_catchment.loc[i, 'geometry']
                geom_j = clipped_catchment.loc[j, 'geometry']
                if geom_i.intersects(geom_j):
                    inter = geom_i.intersection(geom_j)
                    if hasattr(inter, 'length') and inter.length > 0:
                        G.add_edge(clipped_catchment.loc[i, id_col],
                                   clipped_catchment.loc[j, id_col])

        coloring = nx.coloring.greedy_color(G, strategy='largest_first')
        n_colors = max(coloring.values()) + 1 if coloring else 1
        color_map = {sid: boundary_palette[cidx % len(boundary_palette)]
                     for sid, cidx in coloring.items()}
        max_deg = max(dict(G.degree()).values()) if G.degree() else 0
        print(f"    Graph colouring: {n_colors} colours for "
              f"{len(clipped_catchment)} station catchments (max adjacency {max_deg})")

    # --- Common base layers (loaded once) ---
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=CODEBASE_CRS)
    prep_bnd_vis = prep(boundary)
    stations_in_bnd = rail_stations[
        rail_stations.geometry.apply(lambda p: prep_bnd_vis.contains(p))]

    lakes_gdf = None
    if os.path.exists(paths.LAKES_SHP):
        lakes_raw = gpd.read_file(paths.LAKES_SHP).to_crs(CODEBASE_CRS)
        lakes_raw = lakes_raw[lakes_raw.geometry.intersects(boundary)]
        if not lakes_raw.empty:
            lakes_gdf = gpd.clip(lakes_raw, boundary)

    bx_min, by_min, bx_max, by_max = boundary.bounds
    pad = 200

    def _base_setup(ax, ylabel='N [m]'):
        """Apply grey background, boundary, lakes, stations, limits, map elements."""
        ax.set_facecolor('#E8E8E8')
        boundary_gdf.plot(ax=ax, color='white', edgecolor='none', zorder=0)
        if lakes_gdf is not None and not lakes_gdf.empty:
            lakes_gdf.plot(ax=ax, color='#A8D8EA', edgecolor='none', zorder=3)
        boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8,
                                   linestyle='--', zorder=5)
        if not stations_in_bnd.empty:
            ax.scatter(stations_in_bnd.geometry.x, stations_in_bnd.geometry.y,
                       s=25, c='white', edgecolors='black', linewidths=0.8,
                       marker='o', zorder=6)
        ax.set_xlim(bx_min - pad, bx_max + pad)
        ax.set_ylim(by_min - pad, by_max + pad)
        ax.set_aspect('equal')
        ax.set_xlabel('E [m]')
        ax.set_ylabel(ylabel)
        _add_map_elements(ax)

    # --- Figure: 1×2 panels ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 11), sharey=True)
    fig.suptitle(f'Catchment Area Allocation — {method_label}', fontsize=14)

    # ---- Left panel: Access mode ----
    legend_left = []
    for mode, color in mode_colors.items():
        subset = alloc[alloc['access_mode'] == mode]
        if len(subset) > 0:
            subset.plot(ax=ax1, color=color, edgecolor='none', alpha=0.85, zorder=2)
            legend_left.append(
                Patch(facecolor=color, edgecolor='none', label=mode_labels[mode]))
    _base_setup(ax1, ylabel='N [m]')
    legend_left.append(
        Line2D([0], [0], color='black', linewidth=1.8, linestyle='--',
               label='Study area boundary'))
    legend_left.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=8, label='Rail station'))
    ax1.set_title('Access Mode', fontsize=13)
    ax1.legend(handles=legend_left, loc='upper center',
               bbox_to_anchor=(0.5, -0.06), bbox_transform=ax1.transAxes,
               ncol=3, fontsize=8, framealpha=0.9, borderaxespad=0)

    # ---- Right panel: Station allocation ----
    legend_right = []
    if color_map:
        for sid, color in color_map.items():
            subset = alloc[alloc['id_point'] == sid]
            if len(subset) > 0:
                subset.plot(ax=ax2, color=color, edgecolor='none', alpha=0.85, zorder=2)
        legend_right.append(
            Patch(facecolor=boundary_palette[0], edgecolor='none',
                  label='Station catchment'))
    no_pt_cells = alloc[alloc['id_point'] == NO_PT_ID]
    if len(no_pt_cells) > 0:
        no_pt_cells.plot(ax=ax2, color='#d9d9d9', edgecolor='none', alpha=0.85, zorder=2)
    legend_right.append(
        Patch(facecolor='#d9d9d9', edgecolor='none', label='No access'))
    legend_right.append(
        Line2D([0], [0], color='black', linewidth=1.8, linestyle='--',
               label='Study area boundary'))
    legend_right.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=8, label='Rail station'))
    _base_setup(ax2, ylabel='')
    ax2.set_title('Station Allocation', fontsize=13)
    ax2.legend(handles=legend_right, loc='upper center',
               bbox_to_anchor=(0.5, -0.06), bbox_transform=ax2.transAxes,
               ncol=3, fontsize=8, framealpha=0.9, borderaxespad=0)

    # --- Summary table: Modal Access | Hierarchical | Non-Hierarchical ---
    pop_map_d  = grid.set_index('RELI')['NUMMER'].to_dict()
    empl_map_d = empl_grid.set_index('RELI')['NUMMER'].to_dict() if empl_grid is not None else {}

    alloc['_pop']  = alloc['RELI'].map(pop_map_d).fillna(0)
    alloc['_empl'] = alloc['RELI'].map(empl_map_d).fillna(0)
    total_pop  = alloc['_pop'].sum()
    total_empl = alloc['_empl'].sum()

    # No access: same across all three columns (cells no mode can reach at all)
    no_acc_relis = set(alloc.loc[alloc['id_point'] == NO_PT_ID, 'RELI'].values)

    reli_pop_d  = alloc.set_index('RELI')['_pop'].to_dict()
    reli_empl_d = alloc.set_index('RELI')['_empl'].to_dict()

    def _reli_stats(relis):
        p  = sum(reli_pop_d.get(r,  0) for r in relis)
        e  = sum(reli_empl_d.get(r, 0) for r in relis)
        pp = p / total_pop  * 100 if total_pop  > 0 else 0.0
        ep = e / total_empl * 100 if total_empl > 0 else 0.0
        return int(p), pp, int(e), ep

    # Modal Access: cells counted in ALL reachable modes simultaneously
    walk_relis   = set(walk_df['RELI'].values)   if walk_df   is not None and len(walk_df)   > 0 else set()
    feeder_relis = set(feeder_df['RELI'].values) if feeder_df is not None and len(feeder_df) > 0 else set()
    cycle_relis  = set(cycle_df['RELI'].values)  if cycle_df  is not None and len(cycle_df)  > 0 else set()
    modal_access_stats = {
        'Walk':      _reli_stats(walk_relis),
        'PT Feeder': _reli_stats(feeder_relis),
        'Cycling':   _reli_stats(cycle_relis),
        'No access': _reli_stats(no_acc_relis),
    }

    # Hierarchical: winner-takes-all (Walk/PT primary; Cycle only fills uncovered cells)
    pt_modes = {'Bus', 'Tram', 'Ship', 'Funicular'}

    def _group_mode(mode):
        if mode == 'Walk':    return 'Walk'
        if mode in pt_modes:  return 'PT Feeder'
        if mode == 'Cycle':   return 'Cycling'
        return 'No access'

    alloc['_group'] = alloc['access_mode'].apply(_group_mode)
    hier_stats = {}
    for grp in ('Walk', 'PT Feeder', 'Cycling', 'No access'):
        sub = alloc[alloc['_group'] == grp]
        p  = int(sub['_pop'].sum())
        e  = int(sub['_empl'].sum())
        pp = p / total_pop  * 100 if total_pop  > 0 else 0.0
        ep = e / total_empl * 100 if total_empl > 0 else 0.0
        hier_stats[grp] = (p, pp, e, ep)

    # Non-Hierarchical: all three modes compete equally on raw travel time
    nh_frames = [df for df in [walk_df, feeder_df, cycle_df]
                 if df is not None and len(df) > 0]
    if nh_frames:
        nh_combined = pd.concat(nh_frames, ignore_index=True)
        nh_best = nh_combined.loc[
            nh_combined.groupby('RELI')['total_time_sec'].idxmin()].copy()
        nh_best['_group'] = nh_best['access_mode'].apply(_group_mode)
    else:
        nh_best = pd.DataFrame(columns=['RELI', '_group'])
    nh_reli_sets = {grp: set(nh_best.loc[nh_best['_group'] == grp, 'RELI'].values)
                    for grp in ('Walk', 'PT Feeder', 'Cycling')}
    nonhier_stats = {
        'Walk':      _reli_stats(nh_reli_sets['Walk']),
        'PT Feeder': _reli_stats(nh_reli_sets['PT Feeder']),
        'Cycling':   _reli_stats(nh_reli_sets['Cycling']),
        'No access': _reli_stats(no_acc_relis),
    }

    # Build 3-column table
    col_w = 9
    hdr = f"{'Pop':>{col_w}}{'Pop%':>{col_w}}{'FTE':>{col_w}}{'FTE%':>{col_w}}"
    tbl  = (f"{'':14}"
            f"{'── Modal Access ──':>{4*col_w}}  "
            f"{'── Hierarchical ──':>{4*col_w}}  "
            f"{'── Non-Hierarchical ──':>{4*col_w}}\n")
    tbl += f"{'Mode':<14}{hdr}  {hdr}  {hdr}\n"
    tbl += "─" * (14 + 3 * (4*col_w) + 4) + "\n"

    def _fmt(p, pp, e, ep):
        return f"{p:>{col_w},}{pp:>{col_w-1}.1f}%{e:>{col_w},}{ep:>{col_w-1}.1f}%"

    for grp in ('Walk', 'PT Feeder', 'Cycling', 'No access'):
        tbl += f"{grp:<14}{_fmt(*modal_access_stats[grp])}  {_fmt(*hier_stats[grp])}  {_fmt(*nonhier_stats[grp])}\n"
    tbl += "─" * (14 + 3 * (4*col_w) + 4) + "\n"
    tbl += ("Modal Access: all reachable modes counted per cell  |  "
            "Hierarchical: walk/PT primary + cycle fallback  |  "
            "Non-Hierarchical: fastest mode wins (cycle competes equally)")

    fig.subplots_adjust(top=0.93, bottom=0.24, wspace=0.02)
    fig.text(0.5, 0.02, tbl, ha='center', va='bottom',
             fontsize=7.5, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

    out_path = os.path.join(PT_FEEDER_PLOT_DIR,
                            f'catchment_visualisation_{method_label.lower().replace(" ", "_")}.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path}")


# ===============================================================================
# DIFF COMPARISON PLOT
# ===============================================================================

def _build_diff_plot(muni_catchment, pt_catchment, pt_allocation,
                     pop_grid, empl_grid, rail_stations, boundary):
    """Cell-level comparison plot: Municipal vs PT-Feeder station assignment.

    Only cells with population and/or employment are coloured.  Colours:
      - Same station in both methods       → light grey
      - Different station in both methods   → orange (reassigned)
      - Municipal only (no PT-Feeder)       → red
      - PT-Feeder only (no Municipal)       → blue

    Combined station catchment boundaries from both methods are overlaid.

    Parameters
    ----------
    muni_catchment : gpd.GeoDataFrame
        Municipal catchment with columns [id_point, geometry], dissolved per station.
    pt_catchment : gpd.GeoDataFrame
        PT-Feeder catchment GPKG with columns [train_station, id, geometry].
    pt_allocation : pd.DataFrame
        PT-Feeder cell allocation with columns [RELI, id_point, E_KOORD, N_KOORD].
    pop_grid, empl_grid : gpd.GeoDataFrame
        Population / employment grids with RELI column.
    rail_stations : gpd.GeoDataFrame
    boundary : shapely.Polygon
    """
    if muni_catchment is None or pt_catchment is None:
        print("  Skipping diff plot - one method did not produce catchment geometry")
        return

    print("  Building Municipal vs PT-Feeder diff plot (cell-level) ...")

    # --- Cell-level municipal assignment via spatial join ---
    muni_cells = _assign_cells_to_municipal_catchment(
        pop_grid, empl_grid, muni_catchment)
    muni_cells = muni_cells.rename(columns={'id_point': 'muni_station'})
    muni_cells['muni_station'] = pd.to_numeric(
        muni_cells['muni_station'], errors='coerce')

    # --- PT-Feeder cell assignment ---
    pt_cells = pt_allocation[['RELI', 'id_point', 'E_KOORD', 'N_KOORD']].copy()
    pt_cells = pt_cells.rename(columns={'id_point': 'pt_station'})

    # Only cells with pop or empl
    pop_relis = set(pop_grid['RELI'].values)
    empl_relis = set(empl_grid['RELI'].values)
    active_relis = pop_relis | empl_relis
    pt_cells = pt_cells[pt_cells['RELI'].isin(active_relis)].copy()

    # Merge both assignments
    merged = pt_cells.merge(muni_cells, on='RELI', how='outer')

    # Fill coordinates for cells only in municipal (need grid coords)
    coord_lookup = pd.concat([
        pop_grid[['RELI', 'E_KOORD', 'N_KOORD']],
        empl_grid[['RELI', 'E_KOORD', 'N_KOORD']],
    ]).drop_duplicates(subset='RELI')
    missing_coords = merged['E_KOORD'].isna()
    if missing_coords.any():
        fill = merged.loc[missing_coords, ['RELI']].merge(
            coord_lookup, on='RELI', how='left', suffixes=('_old', ''))
        merged.loc[missing_coords, 'E_KOORD'] = fill['E_KOORD'].values
        merged.loc[missing_coords, 'N_KOORD'] = fill['N_KOORD'].values

    merged = merged.dropna(subset=['E_KOORD', 'N_KOORD'])

    # Replace NO_PT sentinel with NaN for cleaner logic
    merged.loc[merged['pt_station'] == NO_PT_ID, 'pt_station'] = np.nan

    # Classify each cell
    has_muni = merged['muni_station'].notna()
    has_pt   = merged['pt_station'].notna()
    same     = has_muni & has_pt & (merged['muni_station'] == merged['pt_station'])
    diff     = has_muni & has_pt & (merged['muni_station'] != merged['pt_station'])
    muni_only = has_muni & ~has_pt
    pt_only   = ~has_muni & has_pt

    CLR_SAME      = '#D3D3D3'
    CLR_DIFFERENT = '#E67E22'
    CLR_MUNI_ONLY = '#B2182B'
    CLR_PT_ONLY   = '#2166AC'

    merged['category'] = ''
    merged.loc[same, 'category']      = 'same'
    merged.loc[diff, 'category']      = 'different'
    merged.loc[muni_only, 'category'] = 'muni_only'
    merged.loc[pt_only, 'category']   = 'pt_only'
    # Cells with neither assignment are ignored
    merged = merged[merged['category'] != ''].copy()

    # Build cell square geometries
    e_arr = merged['E_KOORD'].values
    n_arr = merged['N_KOORD'].values
    merged['geometry'] = [box(e, n, e + CELL_SIZE_M, n + CELL_SIZE_M)
                          for e, n in zip(e_arr, n_arr)]
    merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=CODEBASE_CRS)

    # Summary stats
    n_same = same.sum()
    n_diff = diff.sum()
    n_muni = muni_only.sum()
    n_pt   = pt_only.sum()
    print(f"    Cells — same station: {n_same:,}, different: {n_diff:,}, "
          f"municipal only: {n_muni:,}, PT-feeder only: {n_pt:,}")

    # --- Pop and empl per category (for summary table) ---
    pop_map  = pop_grid.set_index('RELI')['NUMMER'].to_dict()
    empl_map = empl_grid.set_index('RELI')['NUMMER'].to_dict()
    merged['_pop']  = merged['RELI'].map(pop_map).fillna(0)
    merged['_empl'] = merged['RELI'].map(empl_map).fillna(0)
    total_pop  = merged['_pop'].sum()
    total_empl = merged['_empl'].sum()

    def _cat_stats(cat):
        sub = merged[merged['category'] == cat]
        p = sub['_pop'].sum()
        e = sub['_empl'].sum()
        return (p / total_pop  * 100 if total_pop  > 0 else 0.0,
                e / total_empl * 100 if total_empl > 0 else 0.0)

    pct_same  = _cat_stats('same')
    pct_diff  = _cat_stats('different')
    pct_muni  = _cat_stats('muni_only')
    pct_pt    = _cat_stats('pt_only')

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))

    # Light grey background
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=CODEBASE_CRS)
    boundary_gdf.plot(ax=ax, color='#F0F0F0', edgecolor='none')

    # Plot cells by category
    cat_cfg = [
        ('same',      CLR_SAME,      'Same station'),
        ('different', CLR_DIFFERENT, 'Different station (reassigned)'),
        ('muni_only', CLR_MUNI_ONLY, 'Municipal only'),
        ('pt_only',   CLR_PT_ONLY,   'PT-Feeder only'),
    ]
    legend_handles = []
    for cat, colour, label in cat_cfg:
        subset = merged[merged['category'] == cat]
        if len(subset) > 0:
            subset.plot(ax=ax, color=colour, edgecolor='none', zorder=2)
            legend_handles.append(
                Patch(facecolor=colour, edgecolor='none', label=label))

    # Normalise catchment column names
    muni_c = muni_catchment.copy()
    if 'id_point' not in muni_c.columns and 'id' in muni_c.columns:
        muni_c = muni_c.rename(columns={'id': 'id_point'})
    muni_c = muni_c[muni_c['id_point'] != NO_PT_ID]

    pt_c = pt_catchment.copy()
    if 'id_point' not in pt_c.columns and 'id' in pt_c.columns:
        pt_c = pt_c.rename(columns={'id': 'id_point'})
    pt_c = pt_c[pt_c['id_point'] != NO_PT_ID]

    # Clip catchment boundaries to study area
    muni_c = gpd.clip(muni_c, boundary)
    pt_c = gpd.clip(pt_c, boundary)

    # Combined station catchment boundaries
    muni_c.boundary.plot(ax=ax, color='#d6604d', linewidth=0.5, linestyle='--',
                         alpha=0.7, zorder=3)
    pt_c.boundary.plot(ax=ax, color='#4393c3', linewidth=0.5, linestyle='-',
                       alpha=0.7, zorder=3)

    legend_handles += [
        Line2D([0], [0], color='#d6604d', linewidth=1, linestyle='--',
               label='Municipal catchment boundary'),
        Line2D([0], [0], color='#4393c3', linewidth=1, linestyle='-',
               label='PT-Feeder catchment boundary'),
    ]

    # --- Station markers: white fill + black outline for all; green fill for
    #     stations that have a PT-feeder catchment but no municipalities assigned
    #     in the municipal method (n_communes == 0 in the Stations_Summary
    #     sheet of station_catchments.xlsx). ---
    prep_bnd_diff = prep(boundary)
    stations_in_bnd = rail_stations[
        rail_stations.geometry.apply(lambda p: prep_bnd_diff.contains(p))].copy()

    # Load no-municipality stations from the municipal Stations_Summary sheet.
    # All ID sets use strings to match rail_stations['id_point'] (str from stop_id).
    summary_xlsx = os.path.join(MUNICIPAL_DATA_DIR, 'station_catchments.xlsx')
    no_muni_ids: set = set()
    if os.path.exists(summary_xlsx):
        try:
            _sum = pd.read_excel(summary_xlsx, sheet_name='Stations_Summary',
                                  engine='openpyxl')
            no_muni_mask = _sum['n_communes'].fillna(0).astype(int) == 0
            no_muni_ids = set(
                _sum.loc[no_muni_mask, 'station_number'].astype(str).values
            )
        except Exception as exc:
            print(f"    WARNING: failed to read Stations_Summary from {summary_xlsx}: {exc}")

    # A station is "PT-feeder new" if it has no municipal assignment AND exists
    # in the PT-feeder catchment
    pt_ids = {str(x) for x in pt_c['id_point'].values
              if str(x) != str(NO_PT_ID)}
    pt_new_ids = no_muni_ids & pt_ids

    sta_all    = stations_in_bnd
    sta_pt_new = stations_in_bnd[
        stations_in_bnd['id_point'].astype(str).isin(pt_new_ids)]

    # Draw all stations white+black first, then overlay green for new PT stations
    if not sta_all.empty:
        ax.scatter(sta_all.geometry.x, sta_all.geometry.y,
                   s=25, c='white', edgecolors='black', linewidths=0.8,
                   marker='o', zorder=6)
    if not sta_pt_new.empty:
        ax.scatter(sta_pt_new.geometry.x, sta_pt_new.geometry.y,
                   s=25, c='#2ca02c', edgecolors='black', linewidths=0.8,
                   marker='o', zorder=7)

    legend_handles += [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='black', markersize=8,
               label='Rail station (both methods)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
               markeredgecolor='black', markersize=8,
               label='Rail station (PT-Feeder only)'),
    ]

    # Study area boundary
    boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8, linestyle='--', zorder=4)
    legend_handles.append(
        Line2D([0], [0], color='black', linewidth=1.8, linestyle='--',
               label='Study area boundary'))

    # Clip view to study area bounds
    bx_min, by_min, bx_max, by_max = boundary.bounds
    pad = 200
    ax.set_xlim(bx_min - pad, bx_max + pad)
    ax.set_ylim(by_min - pad, by_max + pad)

    ax.set_title('Catchment Comparison: Municipal vs PT-Feeder', fontsize=14)
    ax.set_xlabel('E [m]')
    ax.set_ylabel('N [m]')
    ax.set_aspect('equal')

    # Add cartographic elements
    _add_map_elements(ax)

    # --- Summary table ---
    col_w = 9
    tbl  = "Population & employment shift: Municipal → PT-Feeder\n"
    tbl += f"{'Category':<28}{'Pop (%)':>{col_w}}{'FTE (%)':>{col_w}}\n"
    tbl += "─" * (28 + 2 * col_w) + "\n"
    rows = [
        ('Same station',              pct_same),
        ('Changed station',           pct_diff),
        ('Lost access (Mun. only)',   pct_muni),
        ('Gained access (PT only)',   pct_pt),
    ]
    for label, (pp, ep) in rows:
        tbl += f"{label:<28}{pp:>{col_w-1}.1f}%{ep:>{col_w-1}.1f}%\n"

    # Legend and summary stacked vertically, centered below the plot
    fig.subplots_adjust(bottom=0.30)
    ax.legend(handles=legend_handles, loc='upper center',
              bbox_to_anchor=(0.5, -0.08),
              bbox_transform=ax.transAxes,
              ncol=2, fontsize=8, framealpha=0.9, borderaxespad=0)
    fig.text(0.5, 0.18, tbl, ha='center', va='top',
             fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

    out_path = os.path.join(os.path.dirname(PT_FEEDER_PLOT_DIR),
                            'catchment_diff_municipal_vs_pt_feeder.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path}")


# ===============================================================================
# ACCESS TIME PLOTS
# ===============================================================================

# Shared bin edges and labels for all access-time plots (minutes)
_ACCESS_BINS   = [0, 5, 10, 15, 20, 25, 30, np.inf]
_ACCESS_LABELS = ['0–5', '5–10', '10–15', '15–20', '20–25', '25–30', '≥ 30']
_ACCESS_GREY   = '#BDBDBD'


def _plot_access_times(walk_df, cycle_df, feeder_df, alloc_pop, pop_grid,
                       rail_stations, feeder_stops, boundary, empl_grid=None):
    """Produce four access-time maps (plasma_r palette, discrete 3/5-min bins).

    Plots
    -----
    1. Walk        — cells within BUFFER_RAIL_M of a rail station
    2. Cycle       — cells within CYCLE_RADIUS_M of a rail station
    3. Feeder      — cells within bus/tram buffer whose feeder stop is graph-reachable
    4. Best choice — winner-takes-all result (alloc_pop), all in-buffer cells

    Cells outside their respective buffer are shown in grey.
    Colour scale: plasma_r (yellow = fast, dark purple/blue = slow).
    Bins: 0–3, 3–6, 6–9, 9–12, 12–15 min (3-min steps) then
          15–20, 20–25, 25–30, ≥30 min (5-min steps).
    """
    print("  Plotting access time maps ...")

    n_bins  = len(_ACCESS_LABELS)
    cmap    = plt.get_cmap('plasma_r', n_bins)
    colours = [cmap(k / (n_bins - 1)) for k in range(n_bins)]

    # Load lakes once for all access-time plots
    lakes = None
    if os.path.exists(paths.LAKES_SHP):
        lakes = gpd.read_file(paths.LAKES_SHP).to_crs(CODEBASE_CRS)
        lakes = lakes[lakes.geometry.intersects(boundary)].copy()

    # Pre-build square polygon geometry for the grid (100 m × 100 m)
    _e = pop_grid['E_KOORD'].values
    _n = pop_grid['N_KOORD'].values
    squares = [box(e, n, e + CELL_SIZE_M, n + CELL_SIZE_M)
               for e, n in zip(_e, _n)]
    grid_plot = gpd.GeoDataFrame(
        pop_grid[['RELI', 'E_KOORD', 'N_KOORD']].copy(),
        geometry=squares, crs=CODEBASE_CRS
    )

    def _class(time_sec):
        """Convert array of seconds to 0-based class index into _ACCESS_BINS."""
        mins = np.array(time_sec, dtype=float) / 60.0
        idx  = np.digitize(mins, _ACCESS_BINS[1:], right=False)
        return np.clip(idx, 0, n_bins - 1)

    pop_map  = pop_grid.set_index('RELI')['NUMMER'].to_dict()
    empl_map = empl_grid.set_index('RELI')['NUMMER'].to_dict() if empl_grid is not None else {}

    def _render(title, fname, reli_series, time_series):
        """Plot one access-time map given aligned RELI and time_sec arrays."""
        # Merge onto grid_plot
        df = pd.DataFrame({'RELI': reli_series, 'time_sec': time_series})
        # Keep best time per cell (minimum across candidate stations)
        df = df.groupby('RELI', as_index=False)['time_sec'].min()
        gdf = grid_plot.merge(df, on='RELI', how='left')

        in_buffer = gdf['time_sec'].notna()
        gdf['class_idx'] = -1  # -1 = outside buffer (grey)
        gdf.loc[in_buffer, 'class_idx'] = _class(gdf.loc[in_buffer, 'time_sec'].values)

        # Attach pop/empl values for the summary
        gdf['_pop']  = gdf['RELI'].map(pop_map).fillna(0)
        gdf['_empl'] = gdf['RELI'].map(empl_map).fillna(0)
        total_pop_grid  = gdf['_pop'].sum()
        total_empl_grid = gdf['_empl'].sum()
        total_pop  = gdf.loc[in_buffer, '_pop'].sum()
        total_empl = gdf.loc[in_buffer, '_empl'].sum()

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Grey for out-of-buffer cells
        if (~in_buffer).any():
            gdf[~in_buffer].plot(ax=ax, color=_ACCESS_GREY, edgecolor='none', linewidth=0)

        # Coloured cells per class
        for ci, colour in enumerate(colours):
            mask = gdf['class_idx'] == ci
            if mask.any():
                gdf[mask].plot(ax=ax, color=colour, edgecolor='none', linewidth=0)

        # Lakes — above coloured cells, below boundary
        if lakes is not None and not lakes.empty:
            gpd.clip(lakes, boundary).plot(ax=ax, color='#A8D8EA', edgecolor='none', zorder=3)

        # Boundary
        boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=CODEBASE_CRS)
        boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8,
                                   linestyle='--', zorder=5)

        # Clip view to study area bounds
        bx_min, by_min, bx_max, by_max = boundary.bounds
        pad = 200
        ax.set_xlim(bx_min - pad, bx_max + pad)
        ax.set_ylim(by_min - pad, by_max + pad)

        # Legend — only include colours that actually appear in the plot
        used_classes = set(gdf.loc[in_buffer, 'class_idx'].astype(int).unique())
        legend_handles = [
            Patch(facecolor=colours[ci], edgecolor='none', label=_ACCESS_LABELS[ci])
            for ci in sorted(used_classes)
        ]
        legend_handles.append(Patch(facecolor=_ACCESS_GREY, edgecolor='none',
                                    label='Outside buffer'))
        if lakes is not None and not lakes.empty:
            legend_handles.append(Patch(facecolor='#A8D8EA', edgecolor='none', label='Lake'))
        legend_handles.append(Line2D([0], [0], color='black', linewidth=1.8,
                                     linestyle='--', label='Catchment area boundary'))

        ax.set_title(title, fontsize=13)
        ax.set_xlabel('E [m]')
        ax.set_ylabel('N [m]')
        ax.set_aspect('equal')

        # Add cartographic elements
        _add_map_elements(ax)

        # --- Summary table: pop and FTE by access-time bin ---
        # Percentages are relative to total grid (in-buffer + outside buffer)
        unit_lbl = 'min'
        bin_hdr  = f'Bin ({unit_lbl})'
        col_w    = 9
        hdr_w    = max(10, len(bin_hdr) + 1)
        tbl  = f"{bin_hdr:<{hdr_w}}{'Pop':>{col_w}}{'Pop%':>{col_w}}{'FTE':>{col_w}}{'FTE%':>{col_w}}\n"
        tbl += "─" * (hdr_w + 4 * col_w) + "\n"
        for ci in sorted(used_classes):
            mask_ci = gdf['class_idx'] == ci
            p  = int(gdf.loc[mask_ci, '_pop'].sum())
            e  = int(gdf.loc[mask_ci, '_empl'].sum())
            pp = p / total_pop_grid  * 100 if total_pop_grid  > 0 else 0.0
            ep = e / total_empl_grid * 100 if total_empl_grid > 0 else 0.0
            tbl += (f"{_ACCESS_LABELS[ci]:<{hdr_w}}"
                    f"{p:>{col_w},}{pp:>{col_w-1}.1f}%"
                    f"{e:>{col_w},}{ep:>{col_w-1}.1f}%\n")
        p_out = int(total_pop_grid  - total_pop)
        e_out = int(total_empl_grid - total_empl)
        pp_out = p_out / total_pop_grid  * 100 if total_pop_grid  > 0 else 0.0
        ep_out = e_out / total_empl_grid * 100 if total_empl_grid > 0 else 0.0
        tbl += (f"{'Outside':<{hdr_w}}"
                f"{p_out:>{col_w},}{pp_out:>{col_w-1}.1f}%"
                f"{e_out:>{col_w},}{ep_out:>{col_w-1}.1f}%\n")
        p_tot  = int(total_pop_grid)
        e_tot  = int(total_empl_grid)
        tbl += "─" * (hdr_w + 4 * col_w) + "\n"
        tbl += f"{'Total':<{hdr_w}}{p_tot:>{col_w},}{'100.0':>{col_w}}%{e_tot:>{col_w},}{'100.0':>{col_w-1}}%\n"

        # Legend and summary stacked vertically, centered below the plot
        fig.subplots_adjust(bottom=0.30)
        ax.legend(handles=legend_handles, title=f'Access time ({unit_lbl})',
                  title_fontsize=8, fontsize=7,
                  loc='upper center',
                  bbox_to_anchor=(0.5, -0.08),
                  bbox_transform=ax.transAxes,
                  ncol=3, framealpha=0.9, borderaxespad=0)
        fig.text(0.5, 0.16, tbl, ha='center', va='top',
                 fontsize=8, fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

        out_path = os.path.join(PT_FEEDER_PLOT_DIR, fname)
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"    Saved -> {out_path}")

    # --- 1. Walk ---
    if len(walk_df) > 0:
        _render('Access Time — Walk to Rail Station',
                'plot_access_time_walk.pdf',
                walk_df['RELI'].values,
                walk_df['total_time_sec'].values)

    # --- 2. Cycle ---
    if len(cycle_df) > 0:
        _render('Access Time — Cycle to Rail Station',
                'plot_access_time_cycle.pdf',
                cycle_df['RELI'].values,
                cycle_df['total_time_sec'].values)

    # --- 3. Feeder (bus/tram) ---
    if len(feeder_df) > 0:
        _render('Access Time — PT Feeder to Rail Station',
                'plot_access_time_feeder.pdf',
                feeder_df['RELI'].values,
                feeder_df['total_time_sec'].values)

    # --- 4. Best choice (winner-takes-all from alloc_pop) ---
    # alloc_pop already has the minimum-time station per cell
    best_mask = alloc_pop['id_point'] != NO_PT_ID
    if best_mask.any():
        _render('Access Time — Best Mode (Walk / Cycle / PT)',
                'plot_access_time_best.pdf',
                alloc_pop.loc[best_mask, 'RELI'].values,
                alloc_pop.loc[best_mask, 'access_time_sec'].values)


# ===============================================================================
# STEP 2-PT: ÖV-GÜTEKLASSEN CLASSIFICATION (ARE 2022)
# ===============================================================================

def _headway_to_kat(headway_min, mode_group, is_bahnknoten=False):
    """Convert headway (min) to Haltestellenkategorie integer (1=I … 5=V).

    Returns 0 if headway > 60 min (no category) or mode_group is unknown.
    """
    if not np.isfinite(headway_min) or headway_min <= 0 or headway_min > 60:
        return 0

    if mode_group == 'C':       # Funicular: always Kat. V
        return 5

    # Headway bucket: 0 = <5, 1 = 5–9, 2 = 10–19, 3 = 20–39, 4 = 40–60
    if   headway_min <  5: bucket = 0
    elif headway_min < 10: bucket = 1
    elif headway_min < 20: bucket = 2
    elif headway_min < 40: bucket = 3
    else:                  bucket = 4

    if mode_group == 'A':
        return [1, 1, 2, 3, 4][bucket] if is_bahnknoten else [1, 2, 3, 4, 5][bucket]
    else:                               # mode_group == 'B'
        return [2, 3, 4, 5, 5][bucket]


def _kat_to_ring_records(stop_row, boundary_geom=None):
    """Return a list of ring-polygon records for one stop.

    Each record is a dict with keys:
        stop_id, stop_name, hst_kat, gk_class, dist_band,
        headway_min, n_lines, is_bahnknoten, geometry
    """
    kat = int(stop_row['hst_kat'])
    if kat == 0:
        return []

    max_radius = GK_MAX_RADIUS[kat]
    records = []

    for band_idx, (r_inner, r_outer) in enumerate(GK_DIST_BANDS):
        if r_inner >= max_radius:
            break
        r_outer_clamp = min(r_outer, max_radius)
        gk = _GK_LOOKUP.get((kat, band_idx))
        if gk is None:
            continue

        outer = stop_row.geometry.buffer(r_outer_clamp)
        geom  = outer.difference(stop_row.geometry.buffer(r_inner)) if r_inner > 0 else outer
        if boundary_geom is not None:
            geom = geom.intersection(boundary_geom)
        if geom.is_empty:
            continue

        records.append({
            'stop_id':       str(stop_row['stop_id']),
            'stop_name':     stop_row.get('stop_name', ''),
            'hst_kat':       kat,
            'gk_class':      gk,
            'dist_band':     GK_DIST_LABELS[band_idx],
            'headway_min':   round(float(stop_row.get('headway_min', np.inf)), 1),
            'n_lines':       int(stop_row.get('n_lines', 0)),
            'is_bahnknoten': bool(stop_row.get('is_bahnknoten', False)),
            'geometry':      geom,
        })

    return records


# ===============================================================================
# SHARED LOADERS — per-line frequency and per-segment tables across all periods
# ===============================================================================

# Each (subfolder, suffix, keep_period) tuple defines one of the three
# service-period GeoPackages produced by services_network_builder. Loading all
# three with these filters yields exactly one row per (route_id, direction_id,
# variant_rank) regardless of the periods the line operates in.
_PERIOD_SPECS = [
    ('All_Day',  '_allday',  None),
    ('Peak',     '_peak',    'peak_only'),
    ('Off_Peak', '_offpeak', 'offpeak_only'),
]


def _load_lines_all_periods(base, name, mode_grp_map=None):
    """Load per-line frequency table from all three temporal subfolders.

    Args:
        base: Absolute path to the network root (_FEEDER_BASE or _RAIL_BASE).
        name: File basename without suffix or extension, e.g. 'pt_feeder_lines'.
        mode_grp_map: Optional dict mapping layer name → mode_group code. When
            given, adds a 'mode_group' column to each layer's frame; falls
            back to 'B' for layers not in the map. None to skip the column.

    Returns:
        Concatenation of all rows with all original columns retained, plus
        'mode_group' if mode_grp_map was provided.
    """
    frames = []
    for subfolder, suffix, keep_period in _PERIOD_SPECS:
        path = os.path.join(base, subfolder, f'{name}{suffix}.gpkg')
        for layer_name, _schema in pyogrio.list_layers(path):
            gdf = gpd.read_file(path, layer=layer_name)
            if keep_period is not None and 'service_period' in gdf.columns:
                gdf = gdf[gdf['service_period'] == keep_period]
            if mode_grp_map is not None:
                gdf['mode_group'] = mode_grp_map.get(layer_name, 'B')
            frames.append(gdf)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_segs_all_periods(base, name):
    """Load per-segment edges from all three temporal subfolders, deduped.

    Args:
        base: Absolute path to the network root (_FEEDER_BASE or _RAIL_BASE).
        name: File basename without suffix or extension, e.g. 'pt_feeder_segments'.

    Returns:
        DataFrame with columns ['from_stop_id', 'to_stop_id', 'route_id',
        'direction_id', 'variant_rank']. Bidirectional rows preserved so
        terminal stops appearing only in direction_id=1 are included.
    """
    frames = []
    for subfolder, suffix, _period in _PERIOD_SPECS:
        path = os.path.join(base, subfolder, f'{name}{suffix}.gpkg')
        for layer_name, _schema in pyogrio.list_layers(path):
            gdf = gpd.read_file(path, layer=layer_name)
            gdf = gdf.rename(columns={
                'from_stop_nr': 'from_stop_id',
                'to_stop_nr':   'to_stop_id',
                'GTFS_ID':      'route_id',
            })
            frames.append(
                gdf[['from_stop_id', 'to_stop_id',
                     'route_id', 'direction_id', 'variant_rank']].copy())
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(
        subset=['from_stop_id', 'to_stop_id', 'route_id', 'direction_id', 'variant_rank'])


def _load_feeder_line_freqs():
    """Per-line frequency table for PT-feeder services across the ARE window.

    Returns one row per (route_id, direction_id, variant_rank) with
    `total_dep` (departures over 06:00–20:00) and a derived
    `freq_per_h_window = total_dep / (GK_WINDOW_MIN / 60)` column. Rows
    with missing total_dep are dropped.

    Initial home: catchment_allocate.py. Marked as a candidate for extraction
    to a sibling services_frequency_unions.py once (c)/(d) consume it.
    """
    df = _load_lines_all_periods(_FEEDER_BASE, 'pt_feeder_lines')
    if df.empty:
        return df
    df['total_dep'] = pd.to_numeric(df.get('total_dep'), errors='coerce')
    df = df.dropna(subset=['total_dep'])
    df['freq_per_h_window'] = df['total_dep'] / (GK_WINDOW_MIN / 60.0)
    return df


def _load_rail_line_freqs():
    """Per-line frequency table for rail services. Same shape as feeders."""
    df = _load_lines_all_periods(_RAIL_BASE, 'rail_lines')
    if df.empty:
        return df
    df['total_dep'] = pd.to_numeric(df.get('total_dep'), errors='coerce')
    df = df.dropna(subset=['total_dep'])
    df['freq_per_h_window'] = df['total_dep'] / (GK_WINDOW_MIN / 60.0)
    return df


def _load_rail_segments_table():
    """Per-segment table for rail services (for station-membership lookup)."""
    return _load_segs_all_periods(_RAIL_BASE, 'rail_segments')


# ===============================================================================
# TRANSFER-FREE OD EFFECTIVE HEADWAY UTILITY
# ===============================================================================

def _reconstruct_stop_sequence(variant_segments):
    """Reconstruct the ordered stop sequence for a single variant by chaining
    its segments from→to.

    Args:
        variant_segments: Rows of `segments` belonging to a single
            (route_id, direction_id, variant_rank) with `from_stop_id` /
            `to_stop_id` columns.

    Returns:
        List of stop_ids (str) in order. Empty list if the segments do not
        form a single chain (disconnected or branching variant — best-effort
        fallback picks the lexicographically smallest start; cyclic variants
        terminate at the first repeated stop).
    """
    if variant_segments.empty:
        return []
    pairs = [(str(r['from_stop_id']), str(r['to_stop_id']))
             for _, r in variant_segments.iterrows()]
    pairs = list(set(pairs))   # dedupe in case the same segment appears twice
    succ = {a: b for a, b in pairs}
    targets = set(succ.values())
    starts = [s for s in succ.keys() if s not in targets]
    if not starts:
        return []
    start = min(starts)        # deterministic fallback for branching variants
    seq = [start]
    seen = {start}
    while seq[-1] in succ:
        nxt = succ[seq[-1]]
        if nxt in seen:        # cycle guard
            break
        seq.append(nxt)
        seen.add(nxt)
    return seq


def _compute_transfer_free_headways(
    segments,
    lines,
    stops,
    destinations,
    walk_entry_radius_m=350,
    name_pattern=r'\bBahnhof\b|\bHB\b',
):
    """Effective boarding-wait headway per (origin_stop, destination_stop) for
    services connecting them transfer-free (Spiess & Florian 1989 common-lines,
    applied at the boarding stage).

    For each variant V = (route_id, direction_id, variant_rank) in `lines`:
      1. Reconstruct V's ordered stop sequence by chaining segments from→to.
      2. For each origin O at index i in the sequence, walk forward through
         downstream stops S = seq[j > i]. For each S, look up destinations
         that can be reached on foot (hierarchy: see below).
      3. Add V's `freq_per_h_window` to the (O, D) accumulator for each such D.
    Final headway = 60.0 / Σ freq per (O, D). Pairs with no transfer-free
    service receive no entry; the caller falls back to aggregate stop headway.

    Hierarchical landing-stop rule (mirrors `_build_feeder_graph`):
        For each destination D, search source stops within `walk_entry_radius_m`.
        If any source stop's name matches `name_pattern` (e.g. 'Bahnhof' / 'HB'),
        only those named stops are valid "landing" stops for D — unnamed stops
        in range are excluded. If no named stop is within range, fall back to
        all unnamed stops in range. Single radius; preference replaces the
        pre-W2 separate radii.

    Args:
        segments: Per-segment table with route_id, direction_id, variant_rank,
            from_stop_id, to_stop_id (from `_load_feeder_segments` after the
            Phase 1 expansion, or `_load_segs_all_periods` for rail).
        lines: Per-line frequency table with route_id, direction_id,
            variant_rank, freq_per_h_window (from `_load_feeder_line_freqs` or
            `_load_rail_line_freqs`).
        stops: Source stops GDF with stop_id, stop_name, geometry.
        destinations: Destination stops GDF with stop_id, geometry.
        walk_entry_radius_m: Single proximity radius for the hierarchical
            search (mirrors WALK_RADIUS_M in `_build_feeder_graph`).
        name_pattern: Regex applied to stop_name to identify named stops.

    Returns:
        dict[(origin_stop_id_str, destination_stop_id_str)] ->
            effective_headway_min (float).

    Initial home: catchment_allocate.py. Marked as a candidate for extraction
    to a sibling services_frequency_unions.py once (c)/(d) at the rail-rail
    level are implemented.
    """
    if segments is None or segments.empty or lines is None or lines.empty:
        return {}
    if destinations is None or destinations.empty:
        return {}

    # --- Normalise types and dedupe segments ---
    seg = segments.copy()
    seg['from_stop_id'] = seg['from_stop_id'].astype(str)
    seg['to_stop_id']   = seg['to_stop_id'].astype(str)
    seg['route_id']     = seg['route_id'].astype(str)
    seg = seg.dropna(subset=['from_stop_id', 'to_stop_id', 'route_id',
                             'direction_id', 'variant_rank'])
    seg = seg.drop_duplicates(
        subset=['from_stop_id', 'to_stop_id',
                'route_id', 'direction_id', 'variant_rank'])

    ln = lines.copy()
    ln['route_id'] = ln['route_id'].astype(str)
    ln = ln.dropna(subset=['route_id', 'direction_id', 'variant_rank',
                           'freq_per_h_window'])
    ln_freq = ln.set_index(
        ['route_id', 'direction_id', 'variant_rank'])['freq_per_h_window'].to_dict()

    # --- Build source-stop name and coordinate lookup ---
    name_re = re.compile(name_pattern, re.IGNORECASE)
    stops_df = stops.copy()
    stops_df['stop_id'] = stops_df['stop_id'].astype(str)
    stops_df = stops_df.drop_duplicates(subset='stop_id')
    name_map = dict(zip(stops_df['stop_id'], stops_df['stop_name'].fillna('')))
    coords_map = {sid: (geom.x, geom.y)
                  for sid, geom in zip(stops_df['stop_id'], stops_df.geometry)
                  if geom is not None and not geom.is_empty}

    # --- Build destination spatial index ---
    dests_df = destinations.copy()
    dests_df['stop_id'] = dests_df['stop_id'].astype(str)
    dests_df = dests_df.drop_duplicates(subset='stop_id')
    dests_df = dests_df[dests_df.geometry.notna()]
    if dests_df.empty:
        return {}
    d_coords = np.column_stack([dests_df.geometry.x.values,
                                dests_df.geometry.y.values])
    d_ids = dests_df['stop_id'].values

    # --- Hierarchical landing-stop precompute (per destination) ---
    # For each destination, prefer name-matched source stops within radius;
    # fall back to unnamed source stops only when no named is in range.
    # Output `nearby_per_stop`: feeder_id -> set of destinations it can serve
    # as a "landing" stop (this preserves the variant-walk interface below).
    src_ids = list(coords_map.keys())
    if not src_ids:
        nearby_per_stop = {}
        n_dests_named_tier = 0
        n_dests_fallback_tier = 0
    else:
        src_coords = np.array([coords_map[s] for s in src_ids])
        src_named  = np.array([bool(name_re.search(name_map.get(s, '')))
                               for s in src_ids])
        src_tree = cKDTree(src_coords)
        nearby_per_stop = {}
        n_dests_named_tier = 0
        n_dests_fallback_tier = 0
        for d_idx in range(len(d_ids)):
            d_id = d_ids[d_idx]
            d_pt = d_coords[d_idx]
            near = src_tree.query_ball_point(d_pt, r=walk_entry_radius_m)
            if not near:
                continue
            named_idxs = [k for k in near if src_named[k]]
            if named_idxs:
                chosen = named_idxs
                n_dests_named_tier += 1
            else:
                chosen = near
                n_dests_fallback_tier += 1
            for k in chosen:
                nearby_per_stop.setdefault(src_ids[k], set()).add(d_id)

    # --- Walk each variant's sequence, accumulate (origin, dest) -> Σ freq ---
    acc = {}
    n_used = 0
    n_skipped = 0
    for (rid, did, vrnk), var_segs in seg.groupby(
            ['route_id', 'direction_id', 'variant_rank'], sort=False):
        freq = ln_freq.get((rid, did, vrnk))
        if freq is None or freq <= 0:
            n_skipped += 1
            continue
        seq = _reconstruct_stop_sequence(var_segs)
        if len(seq) < 2:
            n_skipped += 1
            continue
        n_used += 1

        # Walk backward: downstream = ⋃ nearby(seq[i+1..end]) — strictly
        # forward of the origin, so j > i (origin must travel along the line).
        downstream = set()
        for i in range(len(seq) - 1, -1, -1):
            if i < len(seq) - 1:
                downstream |= nearby_per_stop.get(seq[i + 1], set())
            if not downstream:
                continue
            origin = seq[i]
            for d in downstream:
                key = (origin, d)
                acc[key] = acc.get(key, 0.0) + freq

    result = {od: (60.0 / f) for od, f in acc.items() if f > 0}

    # --- Diagnostic ---
    print(f"  Transfer-free headways: {len(result):,} (origin, dest) pairs "
          f"from {n_used} variants used ({n_skipped} skipped: no freq or seq).")
    print(f"    Landing-stop tiers: {n_dests_named_tier} destinations served by "
          f"named-matched stops, {n_dests_fallback_tier} by unnamed fallback.")
    if result:
        vals = np.fromiter(result.values(), dtype=float)
        unique_origins = len({o for o, _ in result})
        unique_dests   = len({d for _, d in result})
        print(f"    Headway min/median/max: "
              f"{vals.min():.2f} / {np.median(vals):.2f} / {vals.max():.2f} min")
        print(f"    Coverage: {unique_origins} unique origins, "
              f"{unique_dests} unique destinations.")
    return result


def _compute_station_freq_penalties(
    rail_lines,
    rail_segments,
    rail_stations,
    op_window_hours=14.0,
):
    """Per-station "invisible" frequency penalty for the catchment-stage station
    choice (W2 feature (b)).

    For each rail station S:
        station_total_dep = Σ rail_lines.total_dep over all
                            (route_id, direction_id, variant_rank) whose
                            segments touch S as from_stop_id or to_stop_id
        headway_min       = op_window_hours * 60.0 / station_total_dep
        freq_penalty_sec  = cp.t_wait_min(headway_min) * 60.0   # raw min, no weight

    The penalty is added to `score_time_sec` (used in argmin) but stripped from
    the displayed `total_time_sec`, so high-frequency stations win close-call
    cell→station decisions without distorting the access-time maps.

    Args:
        rail_lines: Per-line frequency table (`_load_rail_line_freqs()` output).
        rail_segments: Per-segment table (`_load_rail_segments_table()` output).
        rail_stations: GDF of rail stations to score.
        op_window_hours: ARE operational window in hours (default 14 = 06:00–20:00).

    Returns:
        dict[rail_station_id_str] -> freq_penalty_sec (float). Stations with
        total_dep <= 0 are absent from the dict; the caller treats those
        stations as non-candidates.
    """
    if rail_lines is None or rail_lines.empty:
        return {}
    if rail_segments is None or rail_segments.empty:
        return {}

    ln = rail_lines.copy()
    ln['route_id']  = ln['route_id'].astype(str)
    ln['total_dep'] = pd.to_numeric(ln['total_dep'], errors='coerce').fillna(0)

    seg = rail_segments.copy()
    seg['route_id']     = seg['route_id'].astype(str)
    seg['from_stop_id'] = seg['from_stop_id'].astype(str)
    seg['to_stop_id']   = seg['to_stop_id'].astype(str)

    # Variant → set of stations it touches (from + to)
    from_pairs = seg[['from_stop_id', 'route_id', 'direction_id',
                      'variant_rank']].rename(columns={'from_stop_id': 'stop_id'})
    to_pairs   = seg[['to_stop_id',   'route_id', 'direction_id',
                      'variant_rank']].rename(columns={'to_stop_id':   'stop_id'})
    pairs = pd.concat([from_pairs, to_pairs], ignore_index=True).drop_duplicates()

    # Join total_dep per variant
    pairs = pairs.merge(
        ln[['route_id', 'direction_id', 'variant_rank', 'total_dep']],
        on=['route_id', 'direction_id', 'variant_rank'], how='left'
    )
    pairs = pairs.dropna(subset=['total_dep'])

    # Sum departures per station
    station_dep = pairs.groupby('stop_id', as_index=False)['total_dep'].sum()
    station_dep = station_dep[station_dep['total_dep'] > 0].copy()

    # Filter to provided rail_stations set
    valid_ids = set(rail_stations['stop_id'].astype(str))
    station_dep = station_dep[station_dep['stop_id'].isin(valid_ids)].copy()

    # Headway → t_wait → penalty seconds (raw minutes, no weight)
    op_window_min = op_window_hours * 60.0
    station_dep['headway_min']      = op_window_min / station_dep['total_dep']
    station_dep['penalty_sec']      = station_dep['headway_min'].apply(
        lambda h: cp.t_wait_min(h) * 60.0)

    result = dict(zip(station_dep['stop_id'], station_dep['penalty_sec']))

    # --- Diagnostic ---
    print(f"  Station freq penalties: {len(result):,} rail stations covered "
          f"(of {len(rail_stations)} provided).")
    if result:
        vals = np.fromiter(result.values(), dtype=float)
        print(f"    Penalty min/median/max: "
              f"{vals.min():.1f} / {np.median(vals):.1f} / {vals.max():.1f} sec")
        # Map id → name for the top/bottom listing
        name_map = dict(zip(
            rail_stations['stop_id'].astype(str),
            rail_stations.get('stop_name', pd.Series(dtype=str)).fillna('')
        ))
        sorted_items = sorted(result.items(), key=lambda kv: kv[1])
        lo3 = sorted_items[:3]
        hi3 = sorted_items[-3:][::-1]
        print(f"    Lowest penalty (highest frequency):")
        for sid, p in lo3:
            print(f"      {sid:>10}  {p:6.1f} sec  {name_map.get(sid, '')}")
        print(f"    Highest penalty (lowest frequency):")
        for sid, p in hi3:
            print(f"      {sid:>10}  {p:6.1f} sec  {name_map.get(sid, '')}")
    return result


def _compute_stop_gueteklassen(feeder_stops, rail_stops):
    """Classify every feeder and rail stop according to the ARE ÖV-Güteklassen
    methodology (ARE 2022) and add the following columns to both GDFs:

        hst_kat         int   0 (none) or 1–5  (I–V)
        buffer_radius_m int   maximum walk-access radius derived from hst_kat
        headway_min     float effective combined headway across all lines at stop
        n_lines         int   number of distinct routes serving the stop
        is_bahnknoten   bool  True if ≥ 2 distinct rail routes serve the stop

    Loads all three service-period GeoPackages and keeps each
    (route_id, direction_id, variant_rank) exactly once:
        All_Day  → all rows          (service_period = 'all_day')
        Peak     → peak_only rows    (service_period = 'peak_only')
        Off_Peak → offpeak_only rows (service_period = 'offpeak_only')
    dep_value = total_dep (actual departure count over the 06:00–20:00 window).
    headway   = GK_WINDOW_MIN / sum(total_dep per stop per mode group).

    For stops served by multiple mode groups the best (lowest) Kat wins.
    Bahnknoten threshold: ≥ 2 distinct rail route_ids.
    """
    print("  Computing ÖV-Güteklassen stop categories (all service periods) ...")

    # --- Load lines (frequency data) using shared module-level loader ---
    feeder_mode_grp = {'tram': 'B', 'bus': 'B', 'ship': 'B', 'funicular': 'C'}
    rail_mode_grp   = {k: 'A' for k in
                       ['sbahn', 'long_distance_rail',
                        'inter_regional_rail', 'regional_rail']}

    feeder_lines = _load_lines_all_periods(_FEEDER_BASE, 'pt_feeder_lines', feeder_mode_grp)
    rail_lines   = _load_lines_all_periods(_RAIL_BASE,   'rail_lines',      rail_mode_grp)
    all_lines = pd.concat([feeder_lines, rail_lines], ignore_index=True)

    # total_dep is the actual departure count over the full 06:00–20:00 window,
    # correctly computed for every service_period in services_network_builder.
    all_lines['dep_value'] = pd.to_numeric(all_lines['total_dep'], errors='coerce').fillna(0)

    # One dep_value per (route_id, variant_rank): mean across direction_ids
    # gives the bidirectional average naturally (both dirs present → (dir0+dir1)/2).
    route_dep = (all_lines.groupby(['route_id', 'variant_rank'], as_index=False)
                 .agg(dep_value=('dep_value', 'mean'),
                      mode_group=('mode_group', 'first')))

    # --- Load segments to map stops → routes (shared module-level loader) ---
    feeder_segs = _load_segs_all_periods(_FEEDER_BASE, 'pt_feeder_segments')
    rail_segs   = _load_segs_all_periods(_RAIL_BASE,   'rail_segments')
    all_segs = pd.concat([feeder_segs, rail_segs], ignore_index=True)
    # Both directions kept so terminal stops appearing only in direction_id=1 are included.

    # Build unique (stop_id, route_id, variant_rank) pairs
    from_p = all_segs[['from_stop_id', 'route_id', 'variant_rank']].rename(
        columns={'from_stop_id': 'stop_id'})
    to_p   = all_segs[['to_stop_id',   'route_id', 'variant_rank']].rename(
        columns={'to_stop_id':   'stop_id'})
    pairs = (pd.concat([from_p, to_p])
               .drop_duplicates()
               .astype({'stop_id': str}))

    # Join departure counts
    pairs = pairs.merge(route_dep, on=['route_id', 'variant_rank'], how='left')
    pairs = pairs.dropna(subset=['dep_value'])
    pairs = pairs[pairs['dep_value'] > 0]

    # --- Aggregate per stop per mode group ---
    stop_grp = (pairs.groupby(['stop_id', 'mode_group'], as_index=False)
                .agg(dep_sum  =('dep_value', 'sum'),
                     n_routes =('route_id',  'nunique')))

    stop_grp['headway_min'] = GK_WINDOW_MIN / stop_grp['dep_sum'].clip(lower=1e-6)

    # Bahnknoten: rail stops with ≥ 2 distinct route_ids
    rail_route_cnt = (pairs[pairs['mode_group'] == 'A']
                      .groupby('stop_id')['route_id']
                      .nunique()
                      .rename('rail_n_routes')
                      .reset_index())

    # Pivot headways to wide format: one row per stop
    pivot = stop_grp.pivot(index='stop_id', columns='mode_group',
                           values='headway_min').reset_index()
    pivot.columns.name = None
    for col in ('A', 'B', 'C'):
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot = pivot.rename(columns={'A': 'hw_A', 'B': 'hw_B', 'C': 'hw_C'})
    pivot = pivot.merge(rail_route_cnt, on='stop_id', how='left')
    pivot['is_bahnknoten'] = pivot['rail_n_routes'].fillna(0) >= 2

    # Per-stop n_lines (all groups combined)
    n_lines = (pairs.groupby('stop_id')['route_id']
               .nunique().rename('n_lines').reset_index())
    pivot = pivot.merge(n_lines, on='stop_id', how='left')
    pivot['n_lines'] = pivot['n_lines'].fillna(0).astype(int)

    # Best headway across groups (for display)
    pivot['headway_min'] = pivot[['hw_A', 'hw_B', 'hw_C']].min(axis=1)

    # Classify: best (lowest) Kat across all mode groups at the stop
    def _best_kat(row):
        kats = [
            _headway_to_kat(row['hw_A'], 'A', row['is_bahnknoten']),
            _headway_to_kat(row['hw_B'], 'B'),
            _headway_to_kat(row['hw_C'], 'C'),
        ]
        valid = [k for k in kats if k > 0]
        return min(valid) if valid else 0

    pivot['hst_kat'] = pivot.apply(_best_kat, axis=1)
    pivot['buffer_radius_m'] = pivot['hst_kat'].map(GK_MAX_RADIUS).fillna(0).astype(int)

    gk_cols = ['stop_id', 'hst_kat', 'buffer_radius_m',
               'headway_min', 'n_lines', 'is_bahnknoten']
    gk = pivot[gk_cols].copy()

    def _apply_gk(stops_gdf):
        out = stops_gdf.copy()
        out['_sid'] = out['stop_id'].astype(str)
        out = out.merge(gk, left_on='_sid', right_on='stop_id',
                        how='left', suffixes=('', '_gk'))
        out = out.drop(columns=['_sid', 'stop_id_gk'], errors='ignore')
        out['hst_kat']       = out['hst_kat'].fillna(0).astype(int)
        out['buffer_radius_m'] = out['buffer_radius_m'].fillna(0).astype(int)
        out['headway_min']   = out['headway_min'].fillna(np.inf)
        out['n_lines']       = out['n_lines'].fillna(0).astype(int)
        out['is_bahnknoten'] = out['is_bahnknoten'].fillna(False)
        return gpd.GeoDataFrame(out, geometry='geometry', crs=CODEBASE_CRS)

    feeder_stops = _apply_gk(feeder_stops)
    rail_stops   = _apply_gk(rail_stops)

    n_cls = ((feeder_stops['hst_kat'] > 0).sum()
             + (rail_stops['hst_kat'] > 0).sum())
    n_tot = len(feeder_stops) + len(rail_stops)
    print(f"    Classified {n_cls}/{n_tot} stops  "
          f"(window={GK_WINDOW_MIN} min, method=total_dep, all service periods)")

    kat_names = {0: 'none', 1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V'}
    for k in range(6):
        n = ((feeder_stops['hst_kat'] == k).sum()
             + (rail_stops['hst_kat'] == k).sum())
        if n > 0:
            print(f"      Kat {kat_names[k]}: {n:,} stops")

    return feeder_stops, rail_stops


def _dissolve_hierarchical(rings_gdf):
    """Return a GeoDataFrame with one row per Güteklasse (A–D), where each
    class geometry has been clipped to remove overlap with better classes.

    A is best: unchanged.  B = B_raw − A_raw.  C = C_raw − (A∪B)_raw.  Etc.
    """
    class_order = ['A', 'B', 'C', 'D']
    raw_union = {}
    for cls in class_order:
        sub = rings_gdf[rings_gdf['gk_class'] == cls]
        raw_union[cls] = unary_union(sub.geometry.values) if not sub.empty else None

    rows = []
    cumulative_mask = None
    for cls in class_order:
        geom = raw_union[cls]
        if geom is not None and not geom.is_empty:
            if cumulative_mask is not None:
                geom = geom.difference(cumulative_mask)
            if not geom.is_empty:
                rows.append({'gk_class': cls, 'geometry': geom})
        if raw_union[cls] is not None:
            cumulative_mask = (raw_union[cls] if cumulative_mask is None
                               else cumulative_mask.union(raw_union[cls]))
    return gpd.GeoDataFrame(rows, geometry='geometry', crs=CODEBASE_CRS)


def _build_gueteklassen_gpkgs(feeder_stops, rail_stops, boundary, pop_grid, empl_grid):
    """Save two GeoPackages to PT_FEEDER_DATA_DIR:

    gueteklassen_by_stop.gpkg   — ring polygons per stop (one layer per mode)
                                  with cross-stop hierarchy applied: lower-class
                                  rings are clipped where higher-class rings overlap.
    gueteklassen_by_class.gpkg  — single layer, 4 rows (one per Güteklasse A–D)
                                  after hierarchical masking; columns:
                                  fid, Class, Class_Area_m2, Pop, Empl.
    """
    print("  Building Güteklassen GeoPackages ...")

    # Tag each stop with its transport-mode layer name
    feeder_s = feeder_stops.copy()
    feeder_s['_layer'] = feeder_s['mode'].str.lower().apply(
        lambda m: ('funicular' if 'funicular' in m else
                   'tram'      if 'tram'      in m else
                   'ship'      if 'ship'      in m else 'bus'))
    rail_s = rail_stops.copy()
    rail_s['_layer'] = 'rail'

    all_stops = gpd.GeoDataFrame(
        pd.concat([feeder_s, rail_s], ignore_index=True),
        geometry='geometry', crs=CODEBASE_CRS)

    all_records = []
    for _, row in all_stops.iterrows():
        rings = _kat_to_ring_records(row, boundary_geom=boundary)
        for r in rings:
            r['transport_mode'] = row['_layer']
        all_records.extend(rings)

    if not all_records:
        print("    WARNING: no classified stops — skipping GPKG output")
        return

    rings_gdf = gpd.GeoDataFrame(all_records, geometry='geometry', crs=CODEBASE_CRS)

    # Pre-compute raw class unions for hierarchy masking
    class_order = ['A', 'B', 'C', 'D']
    raw_union = {}
    for cls in class_order:
        sub = rings_gdf[rings_gdf['gk_class'] == cls]
        raw_union[cls] = unary_union(sub.geometry.values) if not sub.empty else None

    # --- GPKG 1: by stop — cross-stop hierarchy applied to individual rings ---
    gpkg1 = os.path.join(PT_FEEDER_DATA_DIR, 'gueteklassen_by_stop.gpkg')
    if os.path.exists(gpkg1):
        os.remove(gpkg1)

    clipped_frames = []
    cumulative_mask = None
    for cls in class_order:
        cls_rings = rings_gdf[rings_gdf['gk_class'] == cls].copy()
        if not cls_rings.empty and cumulative_mask is not None:
            cls_rings['geometry'] = cls_rings.geometry.apply(
                lambda g: g.difference(cumulative_mask))
            cls_rings = cls_rings[~cls_rings.geometry.is_empty].copy()
        clipped_frames.append(cls_rings)
        if raw_union[cls] is not None:
            cumulative_mask = (raw_union[cls] if cumulative_mask is None
                               else cumulative_mask.union(raw_union[cls]))

    clipped_rings = gpd.GeoDataFrame(
        pd.concat(clipped_frames, ignore_index=True),
        geometry='geometry', crs=CODEBASE_CRS)

    for layer in ['rail', 'bus', 'tram', 'ship', 'funicular']:
        subset = clipped_rings[clipped_rings['transport_mode'] == layer].drop(
            columns=['transport_mode'], errors='ignore')
        if not subset.empty:
            subset.to_file(gpkg1, layer=layer, driver='GPKG')
    print(f"    Saved -> {gpkg1}  ({len(clipped_rings):,} features)")

    # --- GPKG 2: by class — single layer, 4 rows, hierarchical, with stats ---
    gpkg2 = os.path.join(PT_FEEDER_DATA_DIR, 'gueteklassen_by_class.gpkg')
    if os.path.exists(gpkg2):
        os.remove(gpkg2)

    # Build exclusive geometries (A intact, each lower class minus better classes)
    exclusive_union = {}
    cumulative_mask = None
    for cls in class_order:
        geom = raw_union[cls]
        if geom is not None and not geom.is_empty:
            if cumulative_mask is not None:
                geom = geom.difference(cumulative_mask)
            exclusive_union[cls] = geom if not geom.is_empty else None
        else:
            exclusive_union[cls] = None
        if raw_union[cls] is not None:
            cumulative_mask = (raw_union[cls] if cumulative_mask is None
                               else cumulative_mask.union(raw_union[cls]))

    pop_pts  = pop_grid[['geometry', 'NUMMER']].copy()
    empl_pts = empl_grid[['geometry', 'NUMMER']].copy()

    class_rows = []
    for i, cls in enumerate(class_order, 1):
        geom = exclusive_union.get(cls)
        if geom is None or geom.is_empty:
            continue
        cls_poly = gpd.GeoDataFrame([{'geometry': geom}],
                                    geometry='geometry', crs=CODEBASE_CRS)
        pop_in  = int(gpd.sjoin(pop_pts,  cls_poly, how='inner',
                                predicate='within')['NUMMER'].sum())
        empl_in = int(gpd.sjoin(empl_pts, cls_poly, how='inner',
                                predicate='within')['NUMMER'].sum())
        class_rows.append({
            'fid':           i,
            'Class':         cls,
            'Class_Area_m2': round(geom.area, 0),
            'Pop':           pop_in,
            'Empl':          empl_in,
            'geometry':      geom,
        })

    if class_rows:
        by_class = gpd.GeoDataFrame(class_rows, geometry='geometry', crs=CODEBASE_CRS)
        by_class.to_file(gpkg2, layer='gueteklassen', driver='GPKG')
        print(f"    Saved -> {gpkg2}  ({len(by_class)} classes)")
    else:
        print(f"    WARNING: no classified zones — skipping {gpkg2}")


def _plot_gueteklassen_comparison(feeder_stops, rail_stops, boundary, pop_grid, empl_grid):
    """Side-by-side map: our computed Güteklassen vs. official ARE 2026 data,
    plus a diff panel showing improvement/worsening per cell.
    Includes a summary table in the figure with coverage of inhabited area,
    pop%, and FTE% per class, plus diff statistics.
    """
    out_path = os.path.join(GUETEKLASSEN_PLOT_DIR, 'gueteklassen_comparison.pdf')
    if os.path.exists(out_path):
        print(f"  Güteklassen comparison plot already exists — skipping "
              f"(delete to regenerate: {out_path})")
        return
    print("  Plotting Güteklassen comparison ...")

    are_path = os.path.join(paths.MAIN, _GUETEKLASSEN_ARE_GPKG)
    if not os.path.exists(are_path):
        print(f"    WARNING: official ARE data not found at {are_path} — skipping")
        return

    official = gpd.read_file(are_path, layer='OeV_Gueteklassen_ARE').to_crs(CODEBASE_CRS)
    official = gpd.clip(official, boundary)

    # Build computed rings from classified stops
    all_stops = gpd.GeoDataFrame(
        pd.concat([feeder_stops, rail_stops], ignore_index=True),
        geometry='geometry', crs=CODEBASE_CRS)

    all_records = []
    for _, row in all_stops.iterrows():
        all_records.extend(_kat_to_ring_records(row, boundary_geom=boundary))

    if not all_records:
        print("    WARNING: no classified stops — skipping comparison plot")
        return

    rings = gpd.GeoDataFrame(all_records, geometry='geometry', crs=CODEBASE_CRS)
    # Apply hierarchical dissolving so lower classes don't overlap higher ones
    computed = _dissolve_hierarchical(rings)

    # --- Inhabited area: union of 100 m cell squares that contain pop or empl ---
    inhab_cells = pd.concat([
        pop_grid[pop_grid['NUMMER'] > 0][['E_KOORD', 'N_KOORD', 'RELI']],
        empl_grid[empl_grid['NUMMER'] > 0][['E_KOORD', 'N_KOORD', 'RELI']],
    ]).drop_duplicates(subset='RELI')
    inhabited_geom = unary_union([
        box(e, n, e + CELL_SIZE_M, n + CELL_SIZE_M)
        for e, n in zip(inhab_cells['E_KOORD'], inhab_cells['N_KOORD'])
    ])
    inhabited_area = inhabited_geom.area

    # --- Coverage statistics: % of inhabited area ---
    def _area_pct(gdf, col):
        """% of inhabited area covered by each class."""
        result = {}
        for gk in ('A', 'B', 'C', 'D'):
            if (gdf[col] == gk).any():
                class_geom = unary_union(gdf.loc[gdf[col] == gk, 'geometry'])
                result[gk] = class_geom.intersection(inhabited_geom).area / inhabited_area * 100
            else:
                result[gk] = 0.0
        # No-class: inhabited area not covered by any class
        all_covered = unary_union(gdf['geometry'])
        result['No class'] = (
            inhabited_geom.difference(all_covered).area / inhabited_area * 100
        )
        return result

    comp_pct = _area_pct(computed, 'gk_class')
    off_pct  = _area_pct(official, 'KLASSE')

    # --- Pop % and FTE % per class ---
    total_pop  = pop_grid['NUMMER'].sum()
    total_empl = empl_grid['NUMMER'].sum()
    pop_pts  = pop_grid[['geometry', 'NUMMER', 'RELI']].copy()
    empl_pts = empl_grid[['geometry', 'NUMMER', 'RELI']].copy()

    def _pop_empl_pct(gdf, col):
        """% of total pop and empl in each class (no-class = remainder)."""
        pop_j  = gpd.sjoin(pop_pts,  gdf[['geometry', col]],
                           how='left', predicate='within')
        pop_j  = pop_j[~pop_j.index.duplicated(keep='first')]
        empl_j = gpd.sjoin(empl_pts, gdf[['geometry', col]],
                           how='left', predicate='within')
        empl_j = empl_j[~empl_j.index.duplicated(keep='first')]

        pop_res  = {}
        empl_res = {}
        for gk in ('A', 'B', 'C', 'D'):
            p  = pop_j.loc[pop_j[col]  == gk, 'NUMMER'].sum()
            e  = empl_j.loc[empl_j[col] == gk, 'NUMMER'].sum()
            pop_res[gk]  = p  / total_pop  * 100 if total_pop  > 0 else 0.0
            empl_res[gk] = e  / total_empl * 100 if total_empl > 0 else 0.0
        # No class = not matched by any class
        pop_res['No class']  = pop_j[pop_j[col].isna()]['NUMMER'].sum() / total_pop  * 100 \
            if total_pop  > 0 else 0.0
        empl_res['No class'] = empl_j[empl_j[col].isna()]['NUMMER'].sum() / total_empl * 100 \
            if total_empl > 0 else 0.0
        return pop_res, empl_res

    comp_pop_pct, comp_empl_pct = _pop_empl_pct(computed, 'gk_class')
    off_pop_pct,  off_empl_pct  = _pop_empl_pct(official, 'KLASSE')

    # --- Build diff cells for third panel ---
    # Class hierarchy: A > B > C > D > None (A is best = 0, None = 4)
    class_rank = {'A': 0, 'B': 1, 'C': 2, 'D': 3, None: 4}

    # Build cell geometries for inhabited cells
    cell_geoms = [
        box(e, n, e + CELL_SIZE_M, n + CELL_SIZE_M)
        for e, n in zip(inhab_cells['E_KOORD'], inhab_cells['N_KOORD'])
    ]
    cells_gdf = gpd.GeoDataFrame(
        inhab_cells[['RELI', 'E_KOORD', 'N_KOORD']].copy(),
        geometry=cell_geoms, crs=CODEBASE_CRS
    )

    # Area-based majority classification: assign each cell to the class covering
    # the majority of its area (eliminates false "neither" for boundary cells)
    def _classify_by_area_majority(cells, class_gdf, class_col):
        """For each cell, compute intersection area with each class polygon
        and return the class that covers the majority of the cell's area."""
        results = [None] * len(cells)
        class_rank = {'A': 0, 'B': 1, 'C': 2, 'D': 3}  # lower rank = better class

        for idx, cell_row in enumerate(cells.itertuples()):
            cell_geom = cell_row.geometry
            cell_area = cell_geom.area
            best_class = None
            best_area = 0.0

            # Find all intersecting class polygons via spatial index
            candidates = class_gdf.iloc[
                list(class_gdf.sindex.query(cell_geom, predicate='intersects'))]
            for _, poly_row in candidates.iterrows():
                try:
                    inter_area = cell_geom.intersection(poly_row.geometry).area
                except Exception:
                    inter_area = 0.0
                gk = poly_row[class_col]
                # If this class covers more area, or equal area but better rank, use it
                if inter_area > best_area or (
                    inter_area == best_area and
                    class_rank.get(gk, 99) < class_rank.get(best_class, 99)
                ):
                    best_area = inter_area
                    best_class = gk

            # Only assign if at least some area is covered
            if best_area > 0:
                results[idx] = best_class
        return results

    cells_gdf['comp_class'] = _classify_by_area_majority(
        cells_gdf, computed[['geometry', 'gk_class']], 'gk_class')
    cells_gdf['off_class'] = _classify_by_area_majority(
        cells_gdf, official[['geometry', 'KLASSE']], 'KLASSE')

    # Classify diff: improved (green), worsened (red), same (light grey), neither (dark grey)
    def _classify_diff(comp, off):
        comp_rank = class_rank.get(comp, 4)
        off_rank  = class_rank.get(off, 4)
        if comp_rank == 4 and off_rank == 4:
            return 'neither'
        elif comp_rank < off_rank:
            return 'improved'
        elif comp_rank > off_rank:
            return 'worsened'
        else:
            return 'same'

    cells_gdf['diff_class'] = [
        _classify_diff(c, o)
        for c, o in zip(cells_gdf['comp_class'], cells_gdf['off_class'])
    ]

    # Diff colour scheme
    diff_colours = {
        'improved':  '#2ca02c',  # green
        'worsened':  '#d62728',  # red
        'same':      '#d9d9d9',  # light grey
        'neither':   '#808080',  # dark grey
    }

    # Compute diff statistics (pop and FTE)
    pop_map  = pop_grid.set_index('RELI')['NUMMER'].to_dict()
    empl_map = empl_grid.set_index('RELI')['NUMMER'].to_dict()
    cells_gdf['_pop']  = cells_gdf['RELI'].map(pop_map).fillna(0)
    cells_gdf['_empl'] = cells_gdf['RELI'].map(empl_map).fillna(0)
    total_diff_pop  = cells_gdf['_pop'].sum()
    total_diff_empl = cells_gdf['_empl'].sum()

    diff_stats = {}
    for dc in ('improved', 'worsened', 'same', 'neither'):
        sub = cells_gdf[cells_gdf['diff_class'] == dc]
        p  = int(sub['_pop'].sum())
        e  = int(sub['_empl'].sum())
        pp = p / total_diff_pop  * 100 if total_diff_pop  > 0 else 0.0
        ep = e / total_diff_empl * 100 if total_diff_empl > 0 else 0.0
        diff_stats[dc] = (p, pp, e, ep)

    # --- Create 3-panel figure ---
    fig, axes = plt.subplots(1, 3, figsize=(28, 12), sharey=True)
    fig.subplots_adjust(top=0.93, bottom=0.24, wspace=0.08)
    bx0, by0, bx1, by1 = boundary.bounds
    pad = 500
    boundary_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=CODEBASE_CRS)

    # Panel 1 & 2: Computed and Official
    panel_data = [
        (axes[0], computed, 'gk_class', 'Computed (infraScanRail)'),
        (axes[1], official, 'KLASSE',   'Official ARE (2026)'),
    ]
    for ax, gdf, col, title in panel_data:
        boundary_gdf.plot(ax=ax, color='#F5F5F5', edgecolor='none', zorder=0)
        for gk in ('D', 'C', 'B', 'A'):   # D first → A on top
            sub = gdf[gdf[col] == gk]
            if not sub.empty:
                sub.plot(ax=ax, color=GK_PLOT_COLOURS[gk],
                         edgecolor='none', alpha=0.80, zorder=1)
        if os.path.exists(paths.LAKES_SHP):
            lakes = gpd.read_file(paths.LAKES_SHP).to_crs(CODEBASE_CRS)
            lakes = lakes[lakes.geometry.intersects(boundary)]
            if not lakes.empty:
                gpd.clip(lakes, boundary).plot(
                    ax=ax, color='#A8D4F0', edgecolor='none', zorder=2)
        boundary_gdf.boundary.plot(
            ax=ax, color='black', linewidth=1.8, linestyle='--', zorder=3)
        ax.set_xlim(bx0 - pad, bx1 + pad)
        ax.set_ylim(by0 - pad, by1 + pad)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('E [m]')
        ax.set_ylabel('N [m]')
        # Add cartographic elements
        _add_map_elements(ax)

    # Panel 3: Diff (improvement / worsening)
    ax3 = axes[2]
    boundary_gdf.plot(ax=ax3, color='#F5F5F5', edgecolor='none', zorder=0)

    # Plot diff cells by category
    for dc, colour in diff_colours.items():
        sub = cells_gdf[cells_gdf['diff_class'] == dc]
        if not sub.empty:
            sub.plot(ax=ax3, color=colour, edgecolor='none', alpha=0.85, zorder=1)

    if os.path.exists(paths.LAKES_SHP):
        lakes = gpd.read_file(paths.LAKES_SHP).to_crs(CODEBASE_CRS)
        lakes = lakes[lakes.geometry.intersects(boundary)]
        if not lakes.empty:
            gpd.clip(lakes, boundary).plot(
                ax=ax3, color='#A8D4F0', edgecolor='none', zorder=2)
    boundary_gdf.boundary.plot(
        ax=ax3, color='black', linewidth=1.8, linestyle='--', zorder=3)
    ax3.set_xlim(bx0 - pad, bx1 + pad)
    ax3.set_ylim(by0 - pad, by1 + pad)
    ax3.set_aspect('equal')
    ax3.set_title('Difference (Computed vs. Official)', fontsize=13)
    ax3.set_xlabel('E [m]')

    # Add cartographic elements to panel 3
    _add_map_elements(ax3)

    # Legend for panels 1 & 2
    legend_handles_gk = [
        Patch(facecolor=GK_PLOT_COLOURS[gk], edgecolor='none',
              label=f'Class {gk}')
        for gk in ('A', 'B', 'C', 'D')
    ]
    legend_handles_gk.append(
        Line2D([0], [0], color='black', linewidth=1.8,
               linestyle='--', label='Study area boundary'))
    axes[1].legend(handles=legend_handles_gk, loc='upper right',
                   fontsize=8, framealpha=0.9)

    # Legend for panel 3 (diff)
    legend_handles_diff = [
        Patch(facecolor=diff_colours['improved'], edgecolor='none', label='Improved'),
        Patch(facecolor=diff_colours['worsened'], edgecolor='none', label='Worsened'),
        Patch(facecolor=diff_colours['same'], edgecolor='none', label='Same class'),
        Patch(facecolor=diff_colours['neither'], edgecolor='none', label='Neither classified'),
        Line2D([0], [0], color='black', linewidth=1.8,
               linestyle='--', label='Study area boundary'),
    ]
    ax3.legend(handles=legend_handles_diff, loc='upper right',
               fontsize=8, framealpha=0.9)

    # --- Summary table (% of inhabited area + pop% + FTE% + diff stats) ---
    col_w = 8
    hdr = (f"{'':9}"
           f"{'— Computed —':>{3*col_w}}"
           f"  {'— Official —':>{3*col_w}}\n")
    hdr += (f"{'Class':<9}"
            f"{'Area%':>{col_w}}{'Pop%':>{col_w}}{'FTE%':>{col_w}}"
            f"  {'Area%':>{col_w}}{'Pop%':>{col_w}}{'FTE%':>{col_w}}\n")
    hdr += "─" * (9 + 3*col_w + 2 + 3*col_w) + "\n"

    tbl = f"Coverage (% of inhabited area)\n{hdr}"
    for gk in ('A', 'B', 'C', 'D', 'No class'):
        label = f'Class {gk}' if gk != 'No class' else 'No class'
        tbl += (f"{label:<9}"
                f"{comp_pct[gk]:>{col_w-1}.1f}%"
                f"{comp_pop_pct[gk]:>{col_w-1}.1f}%"
                f"{comp_empl_pct[gk]:>{col_w-1}.1f}%"
                f"  "
                f"{off_pct[gk]:>{col_w-1}.1f}%"
                f"{off_pop_pct[gk]:>{col_w-1}.1f}%"
                f"{off_empl_pct[gk]:>{col_w-1}.1f}%\n")

    # Add diff statistics
    tbl += "\n"
    tbl += f"{'Diff':9}{'Pop':>{col_w}}{'Pop%':>{col_w}}{'FTE':>{col_w}}{'FTE%':>{col_w}}\n"
    tbl += "─" * (9 + 4*col_w) + "\n"
    for dc, label in [('improved', 'Improved'), ('worsened', 'Worsened'),
                      ('same', 'Same'), ('neither', 'Neither')]:
        p, pp, e, ep = diff_stats[dc]
        tbl += f"{label:<9}{p:>{col_w},}{pp:>{col_w-1}.1f}%{e:>{col_w},}{ep:>{col_w-1}.1f}%\n"

    fig.text(0.5, 0.02, tbl, ha='center', fontsize=7.5, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
    fig.suptitle('ÖV-Güteklassen: Computed vs. Official ARE (2026)',
                 fontsize=15, y=0.97)

    out_path = os.path.join(GUETEKLASSEN_PLOT_DIR, 'gueteklassen_comparison.pdf')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path}")


# ===============================================================================
# ORCHESTRATORS
# ===============================================================================

def _run_pt_feeder_method(boundary, pop_grid, empl_grid, temporal='all',
                          visualize: bool = True):
    """Full PT-Feeder allocation pipeline.

    Parameters
    ----------
    temporal : 'all' | 'peak' | 'offpeak'
        Which temporal variant of the network files to load.

    Returns
    -------
    tuple(gpd.GeoDataFrame, pd.DataFrame)
        PT-Feeder catchment polygons and the population cell allocation.
    """
    print("\n--- PT-Feeder Method ---")
    st = time.time()

    # Step 2-PT: Load network data
    feeder_stops    = _load_feeder_stops(boundary, temporal)
    # Restrict to rail stations strictly inside the catchment boundary —
    # out-of-boundary stations (e.g. Killwangen-Spreitenbach) are not
    # candidates for allocation or downstream ODs.
    rail_stations   = _load_rail_stations(boundary, temporal, buffer=0)
    feeder_segments = _load_feeder_segments(temporal)

    # Güteklassen classification → assigns buffer_radius_m per stop
    feeder_stops, rail_stations = _compute_stop_gueteklassen(
        feeder_stops, rail_stations)
    _build_gueteklassen_gpkgs(feeder_stops, rail_stations, boundary, pop_grid, empl_grid)
    if visualize:
        _plot_gueteklassen_comparison(feeder_stops, rail_stations, boundary, pop_grid, empl_grid)

    _build_pt_buffers(feeder_stops, rail_stations, boundary, pop_grid, empl_grid)

    print(f"  [Step 2-PT complete: {time.time() - st:.1f}s]")
    st = time.time()

    # Step 3: Build feeder graph
    feeder_stop_to_rail_times, feeder_stop_to_rail_components, feeder_graph = _build_feeder_graph(
        feeder_stops, feeder_segments, rail_stations)

    # Step 3b: W2 frequency-aware lookups. Run in both cost methods — the only
    # difference between 'calibrated' and 'absolute' is the weights applied
    # downstream (w_wait scales the boarding-wait term; the station penalty is
    # raw seconds either way). The frequency logic itself is mode-independent.
    print("  Computing W2 frequency-aware lookups ...")
    feeder_lines  = _load_feeder_line_freqs()
    rail_lines    = _load_rail_line_freqs()
    rail_segments = _load_rail_segments_table()
    # (a) destination-conditional boarding wait
    transfer_free_headway = _compute_transfer_free_headways(
        feeder_segments, feeder_lines, feeder_stops, rail_stations,
    )
    # (b) invisible station-attractiveness penalty
    station_freq_penalty = _compute_station_freq_penalties(
        rail_lines, rail_segments, rail_stations,
    )

    print(f"  [Step 3 complete: {time.time() - st:.1f}s]")
    st = time.time()

    # Step 4: Allocate cells — population grid first
    print("\n  Allocating population cells ...")
    walk_pop  = _compute_walk_to_rail_times(pop_grid, rail_stations)
    cycle_pop = _compute_cycle_to_rail_times(pop_grid, rail_stations)
    feeder_pop = _compute_feeder_to_rail_times(
        pop_grid, feeder_stops, feeder_stop_to_rail_times,
        feeder_stop_to_rail_components=feeder_stop_to_rail_components,
        transfer_free_headway=transfer_free_headway,
    )
    alloc_pop = _allocate_cells(walk_pop, cycle_pop, feeder_pop,
                                pop_grid, rail_stations,
                                station_freq_penalty=station_freq_penalty)

    print(f"  [Pop allocation complete: {time.time() - st:.1f}s]")
    st = time.time()

    # Step 4b: Allocate employment-only cells (RELIs in empl_grid but not in pop_grid).
    # Necessary so the diff plot can compare all inhabited cells, not just pop cells.
    pop_relis_set = set(pop_grid['RELI'].values)
    empl_only_grid = empl_grid[~empl_grid['RELI'].isin(pop_relis_set)].copy()
    if not empl_only_grid.empty:
        print(f"\n  Allocating {len(empl_only_grid):,} employment-only cells ...")
        walk_empl   = _compute_walk_to_rail_times(empl_only_grid, rail_stations)
        cycle_empl  = _compute_cycle_to_rail_times(empl_only_grid, rail_stations)
        feeder_empl = _compute_feeder_to_rail_times(
            empl_only_grid, feeder_stops, feeder_stop_to_rail_times,
            feeder_stop_to_rail_components=feeder_stop_to_rail_components,
            transfer_free_headway=transfer_free_headway,
        )
        alloc_empl = _allocate_cells(walk_empl, cycle_empl, feeder_empl,
                                     empl_only_grid, rail_stations,
                                     station_freq_penalty=station_freq_penalty)
        alloc_combined = pd.concat([alloc_pop, alloc_empl], ignore_index=True)
        walk_all   = pd.concat([walk_pop,   walk_empl  ], ignore_index=True)
        cycle_all  = pd.concat([cycle_pop,  cycle_empl ], ignore_index=True)
        feeder_all = pd.concat([feeder_pop, feeder_empl], ignore_index=True)
        print(f"  [Empl-only allocation complete: {time.time() - st:.1f}s]")
    else:
        alloc_combined = alloc_pop.copy()
        walk_all, cycle_all, feeder_all = walk_pop, cycle_pop, feeder_pop
        print("  No employment-only cells found — combined allocation equals pop allocation")

    # Single cell-station candidates CSV covering ALL cells with Pop>0 OR Emp>0
    # (replaces the previous split between cell_station_candidates.csv and
    # cell_station_candidates_empl.csv; downstream consumers in
    # catchment_OD_preparation.py read the unified file).
    _build_candidates_csv(walk_all, cycle_all, feeder_all,
                          pop_grid, empl_grid, rail_stations, PT_FEEDER_DATA_DIR)

    st = time.time()

    # Compute the per-(commune, station) breakdown once (used by the GPKG
    # enrichment, the Stations_Summary Excel sheet, and the Phase 4A plots).
    breakdown = _compute_station_commune_breakdown_pt_feeder(
        alloc_combined, pop_grid, empl_grid, rail_stations)

    # Write station_catchments.xlsx (Stations_Summary + Communes_by_station).
    _write_station_catchments_xlsx(PT_FEEDER_DATA_DIR, breakdown, rail_stations,
                                     method_label='PT-Feeder',
                                     alloc_combined=alloc_combined,
                                     pop_grid=pop_grid, empl_grid=empl_grid,
                                     feeder_segments=feeder_segments,
                                     feeder_stops=feeder_stops,
                                     feeder_graph=feeder_graph)

    # Outputs — produced from population allocation (primary) but using the
    # Hamilton-reconciled breakdown for per-station Pop/FTE.
    pt_catchment, _ = _build_catchment_gpkg(
        alloc_pop, pop_grid, empl_grid, rail_stations, PT_FEEDER_DATA_DIR,
        breakdown=breakdown)
    if visualize:
        plot_rs = _get_plot_rail_stations(rail_stations, scope='ca')
        _build_visualisation(alloc_pop, pop_grid, plot_rs, boundary, 'PT-Feeder',
                             empl_grid,
                             walk_df=walk_pop, cycle_df=cycle_pop, feeder_df=feeder_pop,
                             catchment_gdf=pt_catchment)
        _plot_access_times(walk_pop, cycle_pop, feeder_pop, alloc_pop, pop_grid,
                           plot_rs, feeder_stops, boundary, empl_grid=empl_grid)
        # Catchment + network overlay (analogue of the Municipal plot)
        _plot_catchments_with_network(
            pt_catchment, plot_rs, boundary,
            output_dir=PT_FEEDER_PLOT_DIR, method_label='PT-Feeder',
            temporal=temporal,
        )

    # Phase 4A plot suite (added 2026-05-25) — gated by the `visualize` flag.
    # alloc_combined is the union of population and employment-only cells, with
    # access_mode + per-component minutes attached by the compute functions.
    if visualize:
        make_phase_4a_plots(
            method='pt_feeder',
            sa_boundary=None,    # _load_sa_boundary() called inside the orchestrator
            ca_boundary=boundary,
            allocation=alloc_combined,
            rail_stations=rail_stations,
            pop_grid=pop_grid,
            empl_grid=empl_grid,
            feeder_stops=feeder_stops,
            rail_stops=rail_stations,   # rail_stations carries Güteklassen cols after _compute_stop_gueteklassen
            data_dir=PT_FEEDER_DATA_DIR,
            plot_dir=PT_FEEDER_PLOT_DIR,
            breakdown=breakdown,
        )

    print(f"  [Outputs complete: {time.time() - st:.1f}s]")

    return pt_catchment, alloc_combined


# ===============================================================================
# PHASE 4A — NEW EXPORTS & PLOTS (added 2026-05-25)
# ===============================================================================
# Outputs added in Phase 4A of the new main pipeline:
#   - Access-time pyramid plots (Plot a): 6 PDFs (3 modes × Pop/FTE)
#   - Güteklassen 100%-stacked-bar (Plot c): 1 PDF (PT-Feeder only)
#   - SA-station overview Excel + map + per-station modes/access-time figures
# All Phase 4A plots share the bin scheme of _plot_access_times via _ACCESS_BINS
# and _ACCESS_LABELS so the visual story is consistent across plots.


# --- Pyramid configuration -----------------------------------------------------
PYRAMID_COMPONENT_COLOURS = {
    'walk_min':     '#4CAF50',
    'bike_min':     '#FF9800',
    'wait_min':     '#9C27B0',
    'ivt_min':      '#1976D2',
    'transfer_min': '#D32F2F',
}
PYRAMID_COMPONENT_LABELS = {
    'walk_min':     'Walk',
    'bike_min':     'Bike',
    'wait_min':     'Wait',
    'ivt_min':      'IVT',
    'transfer_min': 'Transfer',
}

# Mode groupings — must match the access_mode strings produced by the three
# _compute_*_to_rail_times functions. Walk and Cycle are combined into a
# single plot (4 bars per bin: Walk-Pop, Walk-FTE, Cycle-Pop, Cycle-FTE);
# PT keeps its own plot with component-stacked bars.
PYRAMID_MODE_GROUPS = {
    'walk_cycle': {'Walk', 'Cycle'},
    'pt':         {'Bus', 'Tram', 'Ship', 'Funicular'},
}
PYRAMID_MODE_TITLES = {
    'walk_cycle': 'Walking + Cycling',
    'pt':         'Public Transport (feeder)',
}


def _load_sa_boundary():
    """Read paths.STUDY_AREA_BOUNDARY_GPKG and return a Shapely polygon in LV95."""
    sa_path = os.path.join(paths.MAIN, paths.STUDY_AREA_BOUNDARY_GPKG)
    if not os.path.isfile(sa_path):
        print(f"  WARNING: SA boundary not found at {sa_path}; SA-side comparisons will be empty")
        return None
    gdf = gpd.read_file(sa_path).to_crs(CODEBASE_CRS)
    return gdf.geometry.unary_union


def _assign_access_bin(access_time_min_values: np.ndarray) -> np.ndarray:
    """Assign each cell to a bin index using the existing _ACCESS_BINS scheme.

    Returns int array with values in [0, len(_ACCESS_LABELS) - 1].
    """
    idx = np.digitize(access_time_min_values, _ACCESS_BINS[1:], right=False)
    return np.clip(idx, 0, len(_ACCESS_LABELS) - 1)


def _build_pyramid_cells(allocation, pop_grid, empl_grid, sa_boundary):
    """Build the per-cell DataFrame used by the pyramid plots.

    Columns: RELI, E_KOORD, N_KOORD, access_time_min, access_mode,
             walk_min, bike_min, wait_min, ivt_min, transfer_min,
             pop (int), fte (int), in_sa (bool), bin (int)

    Excludes cells flagged as 'No PT' in the allocation (their access_time_sec
    is the 99999 sentinel from _allocate_cells).
    """
    pop_lookup  = pop_grid.set_index('RELI')['NUMMER']
    empl_lookup = empl_grid.set_index('RELI')['NUMMER']

    cells = allocation[['RELI', 'E_KOORD', 'N_KOORD', 'access_time_sec',
                        'access_mode', 'walk_min', 'bike_min', 'wait_min',
                        'ivt_min', 'transfer_min']].copy()
    cells = cells[cells['access_mode'] != 'No PT'].copy()
    cells['pop'] = cells['RELI'].map(pop_lookup).fillna(0).astype(int)
    cells['fte'] = cells['RELI'].map(empl_lookup).fillna(0).astype(int)
    cells['access_time_min'] = cells['access_time_sec'] / 60.0
    cells['bin'] = _assign_access_bin(cells['access_time_min'].values)

    # SA membership via prepared geometry — fast for many cells against one polygon
    if sa_boundary is not None:
        prep_sa = prep(sa_boundary)
        centroids = [Point(e + CELL_SIZE_M / 2, n + CELL_SIZE_M / 2)
                     for e, n in zip(cells['E_KOORD'], cells['N_KOORD'])]
        cells['in_sa'] = [prep_sa.contains(p) for p in centroids]
    else:
        cells['in_sa'] = False

    return cells


def _plot_one_pyramid(cells_df, mode_key, output_path):
    """Produce one mirrored access-time pyramid for a mode group.

    Two layouts depending on `mode_key`:
      - 'walk_cycle': four bars per bin (Walk-Pop, Walk-FTE, Cycle-Pop,
        Cycle-FTE), each in its mode colour from `_PH4A_MODE_COLOURS`,
        FTE hatched.
      - 'pt': two bars per bin (Pop above, FTE below), each stacked by
        access-time component (`PYRAMID_COMPONENT_COLOURS`), FTE hatched.

    Only populated bins are rendered (bins with at least one nonzero
    SA-or-CA value across any series). Y-axis labels stay aligned with the
    surviving bins. Shortest access-time bin at the bottom. Annotations:
    absolute Pop/FTE count inside each bar, % outside.
    """
    mode_set  = PYRAMID_MODE_GROUPS[mode_key]
    n_bins    = len(_ACCESS_LABELS)
    comp_cols = list(PYRAMID_COMPONENT_COLOURS.keys())
    is_pt     = (mode_key == 'pt')
    is_combo  = (mode_key == 'walk_cycle')

    sub = cells_df[cells_df['access_mode'].isin(mode_set)].copy()
    if sub.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5,
                f"No cells routed via {PYRAMID_MODE_TITLES[mode_key]} mode",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return

    # --- Series definitions -------------------------------------------------
    # Each series = one (mode_filter, weight column) combo drawn as a bar at
    # a vertical offset within a bin.
    if is_combo:
        walk_c = _PH4A_MODE_COLOURS['Walk']
        cyc_c  = _PH4A_MODE_COLOURS['Cycle']
        series = [
            dict(key='walk_pop',  modes={'Walk'},  wc='pop', colour=walk_c, hatch='',   offset=+0.30),
            dict(key='walk_fte',  modes={'Walk'},  wc='fte', colour=walk_c, hatch='//', offset=+0.10),
            dict(key='cycle_pop', modes={'Cycle'}, wc='pop', colour=cyc_c,  hatch='',   offset=-0.10),
            dict(key='cycle_fte', modes={'Cycle'}, wc='fte', colour=cyc_c,  hatch='//', offset=-0.30),
        ]
        BAR_H = 0.16
    else:
        series = [
            dict(key='pop', modes=mode_set, wc='pop', colour=None, hatch='',   offset=+0.195),
            dict(key='fte', modes=mode_set, wc='fte', colour=None, hatch='//', offset=-0.195),
        ]
        BAR_H = 0.32

    def _comp_shares(df_, wc):
        result = {ci: {c: 0.0 for c in comp_cols} for ci in range(n_bins)}
        for ci, group in df_.groupby('bin'):
            w = group[wc].values.astype(float)
            if w.sum() <= 0:
                continue
            sums = {c: float((group[c].values * w).sum()) for c in comp_cols}
            total = sum(sums.values())
            if total <= 0:
                continue
            for c in comp_cols:
                result[ci][c] = sums[c] / total
        return result

    sa_all = sub[sub['in_sa']]
    series_data = []
    for s in series:
        sub_s = sub[sub['access_mode'].isin(s['modes'])]
        sa_s  = sa_all[sa_all['access_mode'].isin(s['modes'])]
        wc = s['wc']
        ca_pb = sub_s.groupby('bin')[wc].sum().reindex(range(n_bins), fill_value=0).astype(int)
        sa_pb = sa_s.groupby('bin')[wc].sum().reindex(range(n_bins), fill_value=0).astype(int)
        ca_tot = float(ca_pb.sum()); sa_tot = float(sa_pb.sum())
        ca_pct = (ca_pb / ca_tot * 100.0) if ca_tot > 0 else ca_pb * 0.0
        sa_pct = (sa_pb / sa_tot * 100.0) if sa_tot > 0 else sa_pb * 0.0
        sd = dict(spec=s, sa_pb=sa_pb, ca_pb=ca_pb, sa_pct=sa_pct, ca_pct=ca_pct,
                  sa_tot=sa_tot, ca_tot=ca_tot)
        if is_pt:
            sd['sa_comp'] = _comp_shares(sa_s,  wc)
            sd['ca_comp'] = _comp_shares(sub_s, wc)
        series_data.append(sd)

    # --- Populated-bins filter ----------------------------------------------
    populated_mask = np.zeros(n_bins, dtype=bool)
    for sd in series_data:
        populated_mask |= (sd['sa_pb'].values > 0)
        populated_mask |= (sd['ca_pb'].values > 0)
    populated_bins = [bi for bi in range(n_bins) if populated_mask[bi]]
    if not populated_bins:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f"No populated bins for {PYRAMID_MODE_TITLES[mode_key]}",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        return

    n_show = len(populated_bins)
    fig, ax = plt.subplots(figsize=(13, max(5.5, 0.7 * n_show + 2.5)))
    y_positions = np.arange(n_show)

    all_pct_vals = []
    for sd in series_data:
        all_pct_vals.extend([sd['sa_pct'].max(), sd['ca_pct'].max()])
    max_pct = max(float(p) for p in all_pct_vals if not np.isnan(float(p))) if all_pct_vals else 1.0
    max_pct = max(max_pct, 1.0)
    label_gap = max_pct * 0.04
    MIN_INNER = max_pct * 0.04

    # --- Draw bars + annotations -------------------------------------------
    for yi, bi in enumerate(populated_bins):
        for sd in series_data:
            spec = sd['spec']
            y = yi + spec['offset']
            for side, pb, pct in [('sa', sd['sa_pb'], sd['sa_pct']),
                                  ('ca', sd['ca_pb'], sd['ca_pct'])]:
                count = int(pb.iloc[bi])
                p     = float(pct.iloc[bi])
                if p <= 0 or count == 0:
                    continue
                sign = 1 if side == 'ca' else -1
                if is_pt:
                    comp = sd['sa_comp'][bi] if side == 'sa' else sd['ca_comp'][bi]
                    cum = 0.0
                    for c in comp_cols:
                        seg = p * comp.get(c, 0.0)
                        if seg > 0:
                            ax.barh(y, sign * seg, left=sign * cum, height=BAR_H,
                                    color=PYRAMID_COMPONENT_COLOURS[c],
                                    edgecolor='white', linewidth=0.3, hatch=spec['hatch'])
                            cum += seg
                else:
                    ax.barh(y, sign * p, left=0, height=BAR_H,
                            color=spec['colour'], edgecolor='white', linewidth=0.3,
                            hatch=spec['hatch'])
                if p >= MIN_INNER:
                    ax.text(sign * p / 2, y, f"{count:,}",
                            ha='center', va='center', fontsize=6,
                            color='white', fontweight='bold')
                ax.text(sign * (p + label_gap), y, f"{p:.1f}%",
                        ha='left' if side == 'ca' else 'right',
                        va='center', fontsize=7)

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([_ACCESS_LABELS[bi] for bi in populated_bins])

    x_pad = max_pct * 0.38
    ax.set_xlim(-max_pct - x_pad, max_pct + x_pad)
    xticks = ax.get_xticks()
    ax.set_xticklabels([f'{abs(t):.0f}%' for t in xticks])
    ax.set_xlabel('% of total  (Study area ← | → Catchment area)')
    ax.set_title(f'Access-time pyramid — {PYRAMID_MODE_TITLES[mode_key]}\n'
                 f'SA (left) vs CA (right)'
                 + (', bars stacked by access-time component' if is_pt else ''))
    ax.grid(axis='x', alpha=0.3)

    # --- Legend -------------------------------------------------------------
    handles = []
    if is_pt:
        used_comps = [c for c in comp_cols if any(
            sd['sa_comp'][bi][c] > 0 or sd['ca_comp'][bi][c] > 0
            for sd in series_data for bi in range(n_bins))]
        handles += [Patch(color=PYRAMID_COMPONENT_COLOURS[c],
                          label=PYRAMID_COMPONENT_LABELS[c]) for c in used_comps]
        handles += [
            Patch(facecolor='white', edgecolor='#444', linewidth=0.8,
                  label='Solid = Population'),
            Patch(facecolor='white', edgecolor='#444', linewidth=0.8,
                  hatch='//', label='Hatched = FTE'),
        ]
        legend_title = 'Access-time component'
    elif is_combo:
        handles = [
            Patch(facecolor=_PH4A_MODE_COLOURS['Walk'],  edgecolor='#444', linewidth=0.5,
                  label='Walk · Pop'),
            Patch(facecolor=_PH4A_MODE_COLOURS['Walk'],  edgecolor='#444', linewidth=0.5,
                  hatch='//', label='Walk · FTE'),
            Patch(facecolor=_PH4A_MODE_COLOURS['Cycle'], edgecolor='#444', linewidth=0.5,
                  label='Cycle · Pop'),
            Patch(facecolor=_PH4A_MODE_COLOURS['Cycle'], edgecolor='#444', linewidth=0.5,
                  hatch='//', label='Cycle · FTE'),
        ]
        legend_title = 'Mode · Weight'
    else:
        legend_title = ''
    if handles:
        ax.legend(handles=handles, loc='lower right', fontsize=9,
                  framealpha=0.9, title=legend_title)

    # --- Footer totals ------------------------------------------------------
    if is_combo:
        tot = {sd['spec']['key']: sd for sd in series_data}
        fig.text(0.5, 0.005,
                 f"Walk — SA Pop: {int(tot['walk_pop']['sa_tot']):,}  "
                 f"CA Pop: {int(tot['walk_pop']['ca_tot']):,}  |  "
                 f"SA FTE: {int(tot['walk_fte']['sa_tot']):,}  "
                 f"CA FTE: {int(tot['walk_fte']['ca_tot']):,}     "
                 f"Cycle — SA Pop: {int(tot['cycle_pop']['sa_tot']):,}  "
                 f"CA Pop: {int(tot['cycle_pop']['ca_tot']):,}  |  "
                 f"SA FTE: {int(tot['cycle_fte']['sa_tot']):,}  "
                 f"CA FTE: {int(tot['cycle_fte']['ca_tot']):,}  "
                 f"(weighted by actual Pop / FTE values)",
                 ha='center', fontsize=7.5, color='#444444')
    else:
        s_pop = next(sd for sd in series_data if sd['spec']['wc'] == 'pop')
        s_fte = next(sd for sd in series_data if sd['spec']['wc'] == 'fte')
        fig.text(0.5, 0.005,
                 f"SA Pop: {int(s_pop['sa_tot']):,}  CA Pop: {int(s_pop['ca_tot']):,}  |  "
                 f"SA FTE: {int(s_fte['sa_tot']):,}  CA FTE: {int(s_fte['ca_tot']):,}  "
                 f"(weighted by actual Pop / FTE values)",
                 ha='center', fontsize=7.5, color='#444444')

    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def _plot_access_time_pyramids(allocation, pop_grid, empl_grid,
                                 sa_boundary, output_dir):
    """Produce two access-time pyramid PDFs in `output_dir`.

    Output filenames:
      pyramid_walk_cycle.pdf — Walk + Cycle combined (4 bars per bin)
      pyramid_pt.pdf         — PT feeder, component-stacked

    Only populated bins are rendered. Shortest bin at the bottom.
    Totals weighted by actual Pop / FTE values (not cell counts).
    """
    print("  Building access-time pyramid plots ...")

    if sa_boundary is None:
        sa_boundary = _load_sa_boundary()

    cells = _build_pyramid_cells(allocation, pop_grid, empl_grid, sa_boundary)
    n_in_sa = int(cells['in_sa'].sum()) if 'in_sa' in cells.columns else 0
    print(f"    {len(cells):,} allocated cells; {n_in_sa:,} inside SA")

    os.makedirs(output_dir, exist_ok=True)
    for mode_key in PYRAMID_MODE_GROUPS:
        out_path = os.path.join(output_dir, f'pyramid_{mode_key}.pdf')
        _plot_one_pyramid(cells, mode_key, out_path)
        print(f"    Saved -> {out_path}")


# --- Phase 6 common helpers ----------------------------------------------------

# Access-mode palette (matches mode_colors used in _build_visualisation)
_PH4A_MODE_COLOURS = {
    'Walk':      '#2ca02c',
    'Cycle':     '#9467bd',
    'Bus':       '#1f77b4',
    'Tram':      '#ff7f0e',
    'Ship':      '#004B8D',
    'Funicular': '#7B3F00',
    'Municipal': '#777777',  # placeholder mode for Municipal method
    'No PT':     '#d9d9d9',
}

# Mode display order (for stacked bars + legends)
_PH4A_MODE_ORDER = ['Walk', 'Cycle', 'Bus', 'Tram', 'Ship', 'Funicular', 'Municipal']

# Güteklassen "none" colour (cells outside any A–D ring)
_GK_NONE_COLOUR = '#cccccc'


def _filter_stations_to_sa(rail_stations, sa_boundary):
    """Return rail stations whose geometry lies within the SA boundary."""
    if sa_boundary is None or rail_stations.empty:
        return rail_stations.iloc[0:0].copy()
    prep_sa = prep(sa_boundary)
    in_sa_mask = rail_stations.geometry.apply(prep_sa.contains)
    return rail_stations[in_sa_mask].copy()


def _read_rail_stops_gpkg(path: str):
    """Read a multi-layer rail_stops GeoPackage and normalise its columns to
    match `_load_rail_stations` output: `stop_id` (str), `id_point` (str alias),
    `stop_name`, `geometry` (LV95). Returns None on failure or empty file.
    """
    try:
        layers = fiona.listlayers(path)
    except Exception:
        layers = [None]
    frames = []
    for layer in layers:
        try:
            gdf = (gpd.read_file(path, layer=layer)
                   if layer else gpd.read_file(path))
            frames.append(gdf.to_crs(CODEBASE_CRS))
        except Exception:
            continue
    if not frames:
        return None
    stops = pd.concat(frames, ignore_index=True)
    stops = gpd.GeoDataFrame(stops, geometry='geometry', crs=CODEBASE_CRS)
    if 'Number' in stops.columns and 'stop_id' not in stops.columns:
        stops = stops.rename(columns={'Number': 'stop_id'})
    if 'stop_id' in stops.columns:
        stops['stop_id'] = stops['stop_id'].astype(str)
        stops = stops.drop_duplicates(subset='stop_id').copy()
        stops['id_point'] = stops['stop_id']
    elif 'id_point' in stops.columns:
        stops['id_point'] = stops['id_point'].astype(str)
    else:
        return None
    if 'stop_name' not in stops.columns:
        stops['stop_name'] = stops['stop_id']
    return stops


def _load_rail_stops_for_plot(scope: str = 'ca'):
    """Return rail stops for plotting from the canonical top-level file.

    Precedence:
      1. When an infra projection is active: <rail_root>/<projection>/rail_stops[_sa].gpkg.
      2. Top-level Unprojected: <_RAIL_BASE>/rail_stops.gpkg (no temporal subfolder).

    The 'sa' scope returns rail_stops_sa.gpkg when available (projection only).
    When only rail_stops.gpkg is available it is loaded in full and then
    SA-filtered via `_filter_stations_to_sa` so the caller still receives an
    SA-scoped set. Returns None if neither projected nor top-level Unprojected
    files exist (caller should fall back to the temporal-subfolder loader).
    """
    if not _RAIL_BASE:
        return None
    rail_root = os.path.dirname(_RAIL_BASE)
    sa_filename = 'rail_stops_sa.gpkg'
    ca_filename = 'rail_stops.gpkg'

    # Build the (path, label, needs_sa_filter) candidate list in priority order.
    candidates = []
    if scope == 'sa':
        if _INFRA_PROJECTION and rail_root:
            candidates.append(
                (os.path.join(rail_root, _INFRA_PROJECTION, sa_filename),
                 f"projected/{_INFRA_PROJECTION}/{sa_filename}", False))
            candidates.append(
                (os.path.join(rail_root, _INFRA_PROJECTION, ca_filename),
                 f"projected/{_INFRA_PROJECTION}/{ca_filename}", True))
        candidates.append(
            (os.path.join(_RAIL_BASE, ca_filename),
             f"Unprojected/{ca_filename}", True))
    else:  # 'ca'
        if _INFRA_PROJECTION and rail_root:
            candidates.append(
                (os.path.join(rail_root, _INFRA_PROJECTION, ca_filename),
                 f"projected/{_INFRA_PROJECTION}/{ca_filename}", False))
        candidates.append(
            (os.path.join(_RAIL_BASE, ca_filename),
             f"Unprojected/{ca_filename}", False))

    for path, label, needs_sa_filter in candidates:
        if not os.path.isfile(path):
            continue
        stops = _read_rail_stops_gpkg(path)
        if stops is None or stops.empty:
            continue
        if needs_sa_filter:
            sa_bdy = _load_sa_boundary()
            stops = _filter_stations_to_sa(stops, sa_bdy)
        print(f"    Plot rail stops ({scope}): {label}  ({len(stops)} stops)")
        return stops
    return None


def _get_plot_rail_stations(fallback_rail_stations, scope: str = 'ca'):
    """Return the rail stations to use for plotting.

    Prefers the canonical top-level files via `_load_rail_stops_for_plot`.
    Falls back to `fallback_rail_stations` (typically what the allocation
    pipeline already loaded) when neither projection nor top-level Unprojected
    files exist. For SA scope, the fallback is SA-filtered via
    `_filter_stations_to_sa`; for CA scope it is returned as-is.
    """
    loaded = _load_rail_stops_for_plot(scope=scope)
    if loaded is not None and not loaded.empty:
        return loaded
    if scope == 'sa':
        return _filter_stations_to_sa(fallback_rail_stations, _load_sa_boundary())
    return fallback_rail_stations


def _build_sa_cells(allocation, pop_grid, empl_grid, sa_boundary):
    """Per-cell DataFrame with Pop, FTE, SA flag, station_id, mode, components.

    Excludes 'No PT' cells (their access_time is the 99999 sentinel).
    """
    pop_lookup  = pop_grid.set_index('RELI')['NUMMER']
    empl_lookup = empl_grid.set_index('RELI')['NUMMER']

    cells = allocation[['RELI', 'E_KOORD', 'N_KOORD', 'id_point',
                        'access_time_sec', 'access_mode',
                        'walk_min', 'bike_min', 'wait_min',
                        'ivt_min', 'transfer_min']].copy()
    cells = cells[cells['access_mode'] != 'No PT'].copy()
    cells['station_id'] = cells['id_point'].astype(int)
    cells['pop'] = cells['RELI'].map(pop_lookup).fillna(0).astype(int)
    cells['fte'] = cells['RELI'].map(empl_lookup).fillna(0).astype(int)
    cells['access_time_min'] = cells['access_time_sec'] / 60.0

    if sa_boundary is not None:
        prep_sa = prep(sa_boundary)
        cells['in_sa'] = [
            prep_sa.contains(Point(e + CELL_SIZE_M / 2, n + CELL_SIZE_M / 2))
            for e, n in zip(cells['E_KOORD'], cells['N_KOORD'])
        ]
    else:
        cells['in_sa'] = False

    return cells


# --- SA-stations overview map (with commune-share pies) -----------------------

def _plot_sa_stations_overview_map(sa_stations, sa_boundary, shares_df,
                                     output_dir):
    """SA-extent overview map.

    Background follows the infrabuild ghost+solid pattern: rail lines and
    municipal boundaries are drawn faded across the whole map extent and again
    solid inside the SA boundary. No white mask is applied — the surrounding
    catchment area remains visible at reduced opacity.

    Each SA station gets two pies above the marker (Pop left, FTE right). The
    Pop pie's wedges are drawn clockwise from 12 o'clock; the FTE pie is
    mirrored (counterclockwise) so labels can stack on opposite sides without
    overlap. Slice labels carry the commune name + share percentage with
    leader lines to the corresponding wedge arc.
    """
    print("  Building SA-stations overview map ...")
    if sa_stations.empty:
        print("    No SA stations — skipping overview map.")
        return None

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_facecolor('#f5f5f5')

    # --- Extent (bigger than SA so the faded surroundings are visible) -------
    if sa_boundary is not None:
        sa_xmin, sa_ymin, sa_xmax, sa_ymax = sa_boundary.bounds
    else:
        sa_xmin, sa_ymin, sa_xmax, sa_ymax = sa_stations.total_bounds
    sa_w = sa_xmax - sa_xmin
    sa_h = sa_ymax - sa_ymin
    margin = 3000.0   # match the SA buffer convention (3 km beyond SA boundary)
    xmin, xmax = sa_xmin - margin, sa_xmax + margin
    ymin, ymax = sa_ymin - margin, sa_ymax + margin
    extent_box = box(xmin, ymin, xmax, ymax)

    # --- Municipal boundaries (ghost outside SA, solid inside) ----------------
    try:
        muni = gpd.read_file(paths.MUNICIPAL_BOUNDARIES_GPKG).to_crs(CODEBASE_CRS)
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

    # --- Lakes (ghost outside SA, solid inside) ------------------------------
    try:
        lakes_full = _load_lakes_for_extent(extent_box, scope='ca')
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

    # --- Rail lines (orange, ghost outside SA, solid inside) -----------------
    try:
        rail_full = _load_rail_lines_for_plot(extent_box, temporal='all')
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

    # --- SA boundary (dashed black, no fill) ---------------------------------
    if sa_boundary is not None:
        sa_gdf = gpd.GeoDataFrame(geometry=[sa_boundary], crs=CODEBASE_CRS)
        sa_gdf.boundary.plot(ax=ax, color='black', linewidth=1.3,
                             linestyle='--', zorder=7)

    # --- Station markers ------------------------------------------------------
    ax.scatter(sa_stations.geometry.x, sa_stations.geometry.y,
               s=22, c='white', edgecolors='black', linewidths=0.9,
               zorder=8)

    # --- Pie sizing ----------------------------------------------------------
    pie_radius     = min(sa_w, sa_h) * 0.022
    pie_dx         = pie_radius * 1.35   # horizontal offset of each pie centre from station
    # Lower the pies so the "Pop" / "FTE" titles (drawn at cy - 1.25*r) sit
    # exactly at the station marker's y.
    pie_dy         = pie_radius * 1.25   # vertical lift of pie centre above marker
    label_dx       = pie_radius * 1.9    # horizontal distance from pie centre to label
    palette = plt.colormaps['tab20'].resampled(20)
    sa_row_by_id = {int(r['id_point']): r for _, r in sa_stations.iterrows()}
    sa_ids = sorted(sa_row_by_id.keys())

    def _slices_for(sub, value_col):
        """Return (label, share, colour) per slice, sorted desc; <3% merged into Other.
        Communes share colours across Pop/FTE via a BFS-keyed mapping."""
        if sub.empty:
            return []
        total = float(sub[value_col].sum())
        if total <= 0:
            return []
        # Deterministic colour key — ordered by descending Pop
        order = sub.sort_values(
            ['pop_in_station', 'BFS_NR'], ascending=[False, True]
        ).reset_index(drop=True)
        colour_for_bfs = {int(r['BFS_NR']): palette(i % 20) for i, r in order.iterrows()}

        ranked = sub.sort_values(value_col, ascending=False).copy()
        ranked['share'] = ranked[value_col] / total
        main         = ranked[ranked['share'] >= 0.03]
        other_share  = float(ranked[ranked['share'] < 0.03]['share'].sum())
        out = []
        for _, r in main.iterrows():
            name = (str(r['commune_name']) if pd.notna(r.get('commune_name'))
                    else f"BFS {r['BFS_NR']}")
            out.append((name, float(r['share']), colour_for_bfs[int(r['BFS_NR'])]))
        if other_share > 0:
            out.append(('Other', other_share, '#888888'))
        return out

    def _closest_angle_in_arc(theta1: float, theta2: float, target: float) -> float:
        """Angle in the counterclockwise arc [theta1, theta2] closest to `target`.

        Used to anchor the leader line on the slice's perimeter at the point
        nearest the labelled side, so the leader never crosses the pie.
        """
        a = theta1 % 360
        b = theta2 % 360
        t = target % 360
        arc_size = (theta2 - theta1) % 360
        if arc_size == 0:
            arc_size = 360
        # Distance from `a` going counterclockwise to `t`
        d_at = (t - a) % 360
        if d_at <= arc_size:
            return t
        # `t` is outside the arc — return the nearer endpoint
        d_a = min(abs(t - a), 360 - abs(t - a))
        d_b = min(abs(t - b), 360 - abs(t - b))
        return a if d_a < d_b else b

    def _draw_pie(cx, cy, slices, title, label_side, clockwise):
        """Draw one pie at (cx, cy) with leader-lined labels on `label_side`.

        Two-pass label placement:
          1. Wedges drawn + each slice's natural label y computed from the
             closest-arc anchor (target = 180° for left, 0° for right).
          2. Labels sorted top-to-bottom; if any pair is closer than
             ~0.45·pie_radius the lower one is pushed down to enforce
             the gap, so nameplates never stack on top of each other.
        Leaders are then drawn from the slice arc to the resolved label y.
        """
        if not slices:
            return
        target = 180.0 if label_side == 'left' else 0.0
        lx_text = cx - label_dx if label_side == 'left' else cx + label_dx
        ha      = 'right' if label_side == 'left' else 'left'

        # ── Pass 1: draw wedges, collect anchor info ──────────────────────
        items = []   # one dict per slice: name, share, sx, sy, ly
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

        # ── Pass 2: enforce minimum vertical gap between stacked labels ────
        min_gap = pie_radius * 0.45
        items_sorted = sorted(items, key=lambda d: -d['ly'])
        for i in range(1, len(items_sorted)):
            ceiling = items_sorted[i - 1]['ly'] - min_gap
            if items_sorted[i]['ly'] > ceiling:
                items_sorted[i]['ly'] = ceiling

        # ── Pass 3: draw leaders + labels at the resolved y positions ─────
        for d in items_sorted:
            ax.plot([d['sx'], lx_text], [d['sy'], d['ly']],
                    color='black', linewidth=0.4, zorder=10)
            ax.text(lx_text, d['ly'], f"{d['name']} ({d['share'] * 100:.1f}%)",
                    fontsize=5, ha=ha, va='center', zorder=11,
                    bbox=dict(boxstyle='square,pad=0.15',
                              facecolor='white', edgecolor='none'))

        # Pie title at the bottom of the pie (Pop/FTE — sits at marker y)
        ax.text(cx, cy - pie_radius * 1.25, title,
                fontsize=5, ha='center', va='top', fontweight='bold', zorder=11,
                bbox=dict(boxstyle='square,pad=0.15',
                          facecolor='white', edgecolor='none'))

    for sid in sa_ids:
        row = sa_row_by_id[sid]
        x, y = row.geometry.x, row.geometry.y

        sub = (shares_df[shares_df['station_number'] == sid]
               if not shares_df.empty else pd.DataFrame())

        pop_slices = _slices_for(sub, 'pop_in_station')
        fte_slices = _slices_for(sub, 'empl_in_station')

        # Pies sit ABOVE the marker — Pop on the left (labels to the left),
        # FTE on the right (labels to the right). FTE is mirrored so slice
        # geometry doesn't crowd the centre of the pair.
        _draw_pie(x - pie_dx, y + pie_dy, pop_slices,
                  title='Pop', label_side='left',  clockwise=True)
        _draw_pie(x + pie_dx, y + pie_dy, fte_slices,
                  title='FTE', label_side='right', clockwise=False)

        # Station name + totals below the marker
        if not sub.empty:
            tp = int(sub['pop_in_station'].sum())
            tf = int(sub['empl_in_station'].sum())
        else:
            tp = tf = 0
        label = f"{row['stop_name']}\nPop {tp:,} · FTE {tf:,}"
        # When the station has pies above, mirror the Pop/FTE-to-pie spacing
        # (anchor at pie_centre + 1.25*r → label bottom sits 0.25*r above the
        # pie top). When neither pie was drawn (no contributing communes),
        # park the label just above the marker instead — no need to leave
        # room for absent pies.
        if pop_slices or fte_slices:
            label_y = y + pie_dy + pie_radius * 1.25
        else:
            label_y = y + pie_radius * 0.4
        ax.text(x, label_y, label,
                fontsize=5, fontweight='bold', va='bottom', ha='center', zorder=11,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='none'))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')
    ax.set_xlabel('E [m]')
    ax.set_ylabel('N [m]')
    ax.set_title('SA stations — catchment commune shares (Pop / FTE)')
    _add_map_elements(ax)

    out_path = os.path.join(output_dir, 'sa_stations_overview_map.pdf')
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path}")
    return out_path


# --- Output b3: stacked bar of access modes per SA station ---------------------

def _plot_sa_stations_access_modes(sa_cells, sa_stations, output_dir):
    """Stacked bar: x = SA station, paired Pop + FTE bars per station,
    segments = access mode (Walk / Cycle / Bus / Tram / etc.).
    Y-axis = % share within each bar.
    """
    print("  Building per-SA-station access-modes stacked bar ...")
    if sa_stations.empty:
        print("    No SA stations — skipping modes bar plot.")
        return None
    sa_ids = sorted(set(sa_stations['id_point'].astype(int)))
    cells = sa_cells[sa_cells['station_id'].isin(sa_ids)].copy()
    if cells.empty:
        print("    No SA cells — skipping modes bar plot.")
        return None

    # Use int keys for consistent id_point lookup regardless of stored dtype
    name_map = dict(zip(sa_stations['id_point'].astype(int),
                        sa_stations['stop_name'].astype(str)))
    modes_present = [m for m in _PH4A_MODE_ORDER if m in cells['access_mode'].unique()]

    # Pivot pop and fte by station x mode
    pop_tbl = cells.pivot_table(index='station_id', columns='access_mode',
                                 values='pop', aggfunc='sum', fill_value=0)
    fte_tbl = cells.pivot_table(index='station_id', columns='access_mode',
                                 values='fte', aggfunc='sum', fill_value=0)
    pop_tbl = pop_tbl.reindex(index=sa_ids, columns=modes_present, fill_value=0)
    fte_tbl = fte_tbl.reindex(index=sa_ids, columns=modes_present, fill_value=0)

    # Normalise to percentage per station
    pop_pct = pop_tbl.div(pop_tbl.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
    fte_pct = fte_tbl.div(fte_tbl.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100

    fig_w = max(10, 0.8 * len(sa_ids) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    x = np.arange(len(sa_ids))
    bar_w = 0.4

    pop_cum = np.zeros(len(sa_ids))
    fte_cum = np.zeros(len(sa_ids))
    for mode in modes_present:
        col = _PH4A_MODE_COLOURS.get(mode, '#888888')
        ax.bar(x - bar_w / 2, pop_pct[mode].values, width=bar_w,
               bottom=pop_cum, color=col, edgecolor='white', linewidth=0.3)
        pop_cum += pop_pct[mode].values
        ax.bar(x + bar_w / 2, fte_pct[mode].values, width=bar_w,
               bottom=fte_cum, color=col, edgecolor='white', linewidth=0.3,
               hatch='//')
        fte_cum += fte_pct[mode].values

    ax.set_xticks(x)
    ax.set_xticklabels([name_map.get(sid, str(sid)) for sid in sa_ids],
                        rotation=60, ha='right')
    ax.set_ylabel('% of station catchment')
    ax.set_ylim(0, 110)
    ax.set_title('Access-mode shares per SA station')
    ax.grid(axis='y', alpha=0.3)

    # Legend: mode colours + Pop/FTE distinguisher
    mode_handles = [Patch(color=_PH4A_MODE_COLOURS.get(m, '#888888'), label=m)
                    for m in modes_present]
    type_handles = [
        Patch(facecolor='#888888', edgecolor='black', linewidth=0.5, label='Population (solid)'),
        Patch(facecolor='#888888', edgecolor='black', linewidth=0.5, hatch='//',
              label='FTE (hatched)'),
    ]
    ax.legend(handles=mode_handles + type_handles, loc='upper right',
              framealpha=0.9, fontsize=8, ncol=2)
    fig.tight_layout()

    out_path = os.path.join(output_dir, 'access_modes_by_station.pdf')
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path}")
    return out_path


# --- Output b4: access-time scatter per SA station (Pop and FTE) ---------------

def _plot_sa_stations_access_time_violin(sa_cells, sa_stations, output_dir):
    """Two violin PDFs (Pop-weighted, FTE-weighted) of per-SA-station access times.

    For each station, three side-by-side weighted KDE violins are drawn:
    Walk (green), Cycle (purple), PT feeder (blue). Violin shape is a symmetric
    KDE weighted by Pop (or FTE). Weighted median marked with a white/black line.
    Distribution is weighted by actual Pop / FTE values (not cell counts).
    """
    print("  Building per-SA-station access-time violin plots ...")
    if sa_stations.empty:
        print("    No SA stations — skipping violin plots.")
        return []
    sa_ids = sorted(set(sa_stations['id_point'].astype(int)))
    cells = sa_cells[sa_cells['station_id'].isin(sa_ids)].copy()
    if cells.empty:
        print("    No SA cells — skipping violin plots.")
        return []

    name_map = dict(zip(sa_stations['id_point'].astype(int),
                        sa_stations['stop_name'].astype(str)))

    # 3 mode groups with x-offset and colour (matching scatter-plot palette)
    VIOLIN_MODES = [
        ('Walk',  {'Walk'},                             _PH4A_MODE_COLOURS['Walk'],   -0.3),
        ('Cycle', {'Cycle'},                            _PH4A_MODE_COLOURS['Cycle'],   0.0),
        ('PT',    {'Bus', 'Tram', 'Ship', 'Funicular'}, _PH4A_MODE_COLOURS['Bus'],     0.3),
    ]
    VIOLIN_HALF_W = 0.12  # half-width of each violin in x-axis units

    fig_w = max(12, 0.9 * len(sa_ids) + 3)
    saved = []

    for weight_col in ('pop', 'fte'):
        fig, ax = plt.subplots(figsize=(fig_w, 8))
        any_drawn = False

        for xi, sid in enumerate(sa_ids):
            sc = cells[cells['station_id'] == sid]
            for mode_label, mode_set, colour, x_off in VIOLIN_MODES:
                mc = sc[sc['access_mode'].isin(mode_set)].copy()
                weights = mc[weight_col].values.astype(float)
                times   = mc['access_time_min'].values.astype(float)
                total_w = weights.sum()
                if total_w <= 0 or len(mc) < 2:
                    continue

                x_center = xi + x_off
                try:
                    kde = gaussian_kde(times, weights=weights)
                    y_lo = max(0.0, float(times.min()) - 0.5)
                    y_hi = float(times.max()) + 0.5
                    y_grid = np.linspace(y_lo, y_hi, 200)
                    k = kde(y_grid)
                    k_max = k.max()
                    if k_max > 0:
                        k = k / k_max * VIOLIN_HALF_W
                    ax.fill_betweenx(y_grid, x_center - k, x_center + k,
                                     color=colour, alpha=0.75, linewidth=0)
                    ax.plot(x_center + k, y_grid, color='black', linewidth=0.4, alpha=0.5)
                    ax.plot(x_center - k, y_grid, color='black', linewidth=0.4, alpha=0.5)
                    # Weighted median
                    si = np.argsort(times)
                    sv = times[si]; sw = weights[si]
                    cum = np.cumsum(sw); cum /= cum[-1]
                    med = float(np.interp(0.5, cum, sv))
                    hw = VIOLIN_HALF_W
                    ax.plot([x_center - hw, x_center + hw], [med, med],
                            color='white', linewidth=2.2, solid_capstyle='round', zorder=5)
                    ax.plot([x_center - hw, x_center + hw], [med, med],
                            color='black', linewidth=0.9, solid_capstyle='round', zorder=6)
                    any_drawn = True
                except Exception:
                    pass

        if not any_drawn:
            print(f"    No data for violin plot ({weight_col}) — skipping.")
            plt.close(fig)
            continue

        ax.set_xticks(np.arange(len(sa_ids)))
        ax.set_xticklabels([name_map.get(sid, str(sid)) for sid in sa_ids],
                            rotation=60, ha='right')
        ax.set_ylabel('Access time (min)')
        ax.set_xlabel('SA station')
        ax.set_title(f'Per-cell access time to SA station — {weight_col.upper()}-weighted '
                     f'violins, bar = weighted median')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        mode_handles = []
        for mode_label, _, colour, _ in VIOLIN_MODES:
            lbl = 'PT feeder' if mode_label == 'PT' else mode_label
            mode_handles.append(Patch(facecolor=colour, edgecolor='black',
                                      linewidth=0.5, label=lbl))
        mode_handles.append(Line2D([0], [0], color='black', linewidth=1.0,
                                    label='Weighted median'))
        ax.legend(handles=mode_handles, loc='upper right', fontsize=8,
                   framealpha=0.9, title='Access mode')

        fig.tight_layout()
        out_path = os.path.join(output_dir, f'access_time_by_station_{weight_col}.pdf')
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"    Saved -> {out_path}")
        saved.append(out_path)

    return saved


# --- Output c: Güteklassen 100%-stacked bar (PT-Feeder only) -------------------

def _compute_cell_gueteklassen(feeder_stops, rail_stops, pop_grid, empl_grid):
    """For each cell with Pop or FTE > 0, return its Güteklasse ('A'/'B'/'C'/'D'
    or 'None').

    Uses the hierarchical exclusive-union scheme: each cell falls in the best
    (lowest-letter) class whose raw ring contains it.

    Returns DataFrame [RELI, Pop, Empl, gk_class] with gk_class ∈ {A, B, C, D, None}.
    """
    print("    Classifying cells by Güteklasse (hierarchical) ...")
    # Build per-class ring polygons via _kat_to_ring_records, then exclusive union
    all_stops = gpd.GeoDataFrame(
        pd.concat([feeder_stops, rail_stops], ignore_index=True),
        geometry='geometry', crs=CODEBASE_CRS)

    records = []
    for _, row in all_stops.iterrows():
        records.extend(_kat_to_ring_records(row, boundary_geom=None))
    if not records:
        # No classified stops — every cell is "None"
        cells = pop_grid[['RELI', 'NUMMER', 'geometry']].rename(columns={'NUMMER': 'pop'})
        empl_r = empl_grid[['RELI', 'NUMMER']].rename(columns={'NUMMER': 'empl'})
        cells = cells.merge(empl_r, on='RELI', how='outer')
        cells['pop']  = cells['pop'].fillna(0)
        cells['empl'] = cells['empl'].fillna(0)
        cells['gk_class'] = 'None'
        return cells[['RELI', 'pop', 'empl', 'gk_class']]

    rings = gpd.GeoDataFrame(records, geometry='geometry', crs=CODEBASE_CRS)
    class_order = ['A', 'B', 'C', 'D']
    exclusive = {}
    cumulative = None
    for cls in class_order:
        sub = rings[rings['gk_class'] == cls]
        geom = unary_union(sub.geometry.values) if not sub.empty else None
        if geom is not None and not geom.is_empty:
            if cumulative is not None:
                geom = geom.difference(cumulative)
            if not geom.is_empty:
                exclusive[cls] = geom
                cumulative = (geom if cumulative is None else cumulative.union(geom))
        elif cumulative is None:
            cumulative = None

    # Cells with pop or fte
    pop_renamed  = pop_grid[['RELI', 'NUMMER', 'geometry']].rename(columns={'NUMMER': 'pop'})
    empl_renamed = empl_grid[['RELI', 'NUMMER']].rename(columns={'NUMMER': 'empl'})
    cells = pop_renamed.merge(empl_renamed, on='RELI', how='outer')
    cells['pop']  = cells['pop'].fillna(0)
    cells['empl'] = cells['empl'].fillna(0)
    cells = cells.dropna(subset=['geometry'])
    # Backfill geometry for empl-only cells
    missing_geom_mask = cells['geometry'].isna()
    if missing_geom_mask.any() and 'geometry' in empl_grid.columns:
        empl_geom = empl_grid.set_index('RELI')['geometry']
        cells.loc[missing_geom_mask, 'geometry'] = (
            cells.loc[missing_geom_mask, 'RELI'].map(empl_geom))
    cells = cells.dropna(subset=['geometry'])
    cells_gdf = gpd.GeoDataFrame(cells, geometry='geometry', crs=CODEBASE_CRS)

    cells_gdf['gk_class'] = 'None'
    for cls in class_order:
        if cls not in exclusive:
            continue
        cls_poly = gpd.GeoDataFrame([{'geometry': exclusive[cls]}],
                                     geometry='geometry', crs=CODEBASE_CRS)
        in_class = gpd.sjoin(cells_gdf, cls_poly, how='inner',
                              predicate='within').index
        cells_gdf.loc[in_class, 'gk_class'] = cls

    return cells_gdf[['RELI', 'pop', 'empl', 'gk_class']]


def _plot_gueteklassen_stacked_bar(feeder_stops, rail_stops,
                                     sa_boundary, ca_boundary,
                                     pop_grid, empl_grid, output_dir):
    """One PDF with four 100%-stacked bars: SA-Pop, SA-FTE, CA-Pop, CA-FTE.

    Segments are Güteklassen A, B, C, D, plus 'None' (cells outside any class).
    """
    out_path = os.path.join(output_dir, 'gueteklassen_stacked_bar.pdf')
    if os.path.exists(out_path):
        print(f"  Güteklassen stacked bar already exists — skipping "
              f"(delete to regenerate: {out_path})")
        return None
    print("  Building Güteklassen 100%-stacked bar ...")

    cells_class = _compute_cell_gueteklassen(feeder_stops, rail_stops,
                                              pop_grid, empl_grid)
    if cells_class.empty:
        print("    No classified cells — skipping Güteklassen bar.")
        return None

    # Attach geometry + SA membership
    cells_geom = pop_grid.set_index('RELI')['geometry']
    cells_class = cells_class.copy()
    cells_class['geometry'] = cells_class['RELI'].map(cells_geom)
    # Empl-only cells fallback
    miss = cells_class['geometry'].isna()
    if miss.any() and 'geometry' in empl_grid.columns:
        empl_geom = empl_grid.set_index('RELI')['geometry']
        cells_class.loc[miss, 'geometry'] = cells_class.loc[miss, 'RELI'].map(empl_geom)
    cells_class = cells_class.dropna(subset=['geometry'])

    if sa_boundary is not None:
        prep_sa = prep(sa_boundary)
        cells_class['in_sa'] = cells_class['geometry'].apply(prep_sa.contains)
    else:
        cells_class['in_sa'] = False

    class_order = ['A', 'B', 'C', 'D', 'None']
    colours = {**GK_PLOT_COLOURS, 'None': _GK_NONE_COLOUR}

    def _pct_per_class(df, weight_col):
        total = float(df[weight_col].sum())
        if total <= 0:
            return {c: 0.0 for c in class_order}
        return {c: float(df[df['gk_class'] == c][weight_col].sum()) / total * 100.0
                for c in class_order}

    sa = cells_class[cells_class['in_sa']]
    ca = cells_class
    sa_pop  = _pct_per_class(sa, 'pop')
    sa_fte  = _pct_per_class(sa, 'empl')
    ca_pop  = _pct_per_class(ca, 'pop')
    ca_fte  = _pct_per_class(ca, 'empl')

    bar_labels = ['SA — Pop', 'SA — FTE', 'CA — Pop', 'CA — FTE']
    bar_data   = [sa_pop, sa_fte, ca_pop, ca_fte]
    bar_totals = [
        int(sa['pop'].sum()), int(sa['empl'].sum()),
        int(ca['pop'].sum()), int(ca['empl'].sum()),
    ]

    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(len(bar_labels))
    bar_w = 0.6
    cum = np.zeros(len(bar_labels))
    for cls in class_order:
        vals = np.array([d[cls] for d in bar_data])
        ax.bar(x, vals, width=bar_w, bottom=cum,
               color=colours[cls], edgecolor='white', linewidth=0.5,
               label=cls)
        # Show class percentage label inside each segment if large enough
        for xi, vi, ci in zip(x, vals, cum):
            if vi >= 4.0:
                ax.text(xi, ci + vi / 2, f'{vi:.1f}%',
                        ha='center', va='center', fontsize=9,
                        color='white' if cls != 'None' else 'black',
                        fontweight='bold')
        cum += vals

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.set_ylabel('% of total')
    ax.set_ylim(0, 108)
    ax.set_title('ÖV-Güteklassen distribution — SA vs CA (Pop and FTE)')
    ax.legend(loc='upper right', fontsize=9, title='Güteklasse', framealpha=0.95)
    for xi, total in zip(x, bar_totals):
        ax.text(xi, 102, f"{total:,}", ha='center', va='bottom',
                fontsize=8, color='#444444')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(output_dir, 'gueteklassen_stacked_bar.pdf')
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"    Saved -> {out_path}")
    return out_path


# --- Orchestrator --------------------------------------------------------------

def make_phase_4a_plots(method, sa_boundary, ca_boundary,
                          allocation, rail_stations,
                          pop_grid, empl_grid,
                          feeder_stops=None, rail_stops=None,
                          data_dir=None, plot_dir=None,
                          breakdown=None):
    """Generate all Phase 4A plots/exports for the active method.

    Args:
        method:       'pt_feeder' or 'municipal'
        sa_boundary:  Shapely polygon (or None — see _load_sa_boundary fallback)
        ca_boundary:  Shapely polygon
        allocation:   per-cell DataFrame (PT-Feeder method only). For Municipal
                      pass an empty DataFrame; cell-level plots are skipped.
        rail_stations: full rail station GeoDataFrame
        pop_grid:     2023+ scaled population grid
        empl_grid:    2023+ scaled employment grid
        feeder_stops: PT-Feeder feeder stops with Güteklassen columns (PT-Feeder only)
        rail_stops:   PT-Feeder rail stops with Güteklassen columns (PT-Feeder only)
        data_dir:     <METHOD>_DATA_DIR — used as the SA Excel output directory
        plot_dir:     <METHOD>_PLOT_DIR — output dir for PDFs
        breakdown:    enriched per-(commune, station) DataFrame from
                      `_compute_station_commune_breakdown_*`. Preferred over
                      reading from disk. If None, the Communes_by_station sheet
                      of station_catchments.xlsx is loaded as a fallback.

    Method-conditional outputs:
      PT-Feeder: pyramids + SA Excel + map + modes bar + access-time scatter + Güteklassen bar
      Municipal: SA Excel (no Modes sheet) + map only
    """
    print(f"\n  === Phase 4A plot suite ({method}) ===")
    if data_dir is None:
        data_dir = PT_FEEDER_DATA_DIR if method == 'pt_feeder' else MUNICIPAL_DATA_DIR
    if plot_dir is None:
        plot_dir = PT_FEEDER_PLOT_DIR if method == 'pt_feeder' else MUNICIPAL_PLOT_DIR
    os.makedirs(plot_dir, exist_ok=True)

    if sa_boundary is None:
        sa_boundary = _load_sa_boundary()

    # Caller must pass the in-memory breakdown — the on-disk fallback was
    # removed because station_catchments.xlsx no longer carries the unfiltered
    # per-(commune, station) breakdown (Sheet 2 is SA-filtered).
    if breakdown is None:
        raise RuntimeError(
            "make_phase_4a_plots requires `breakdown` to be passed by the caller. "
            "The on-disk fallback was removed.")
    shares_df = breakdown

    # SA stations sourced via the projection-aware loader: projected
    # rail_stops_sa.gpkg → projected rail_stops.gpkg + SA filter → Unprojected
    # top-level rail_stops.gpkg + SA filter. Falls back to the analysis
    # rail_stations + SA filter if none of those files are present.
    sa_stations = _get_plot_rail_stations(rail_stations, scope='sa')
    print(f"    SA stations: {len(sa_stations)}")

    is_pt_feeder = (method == 'pt_feeder')

    if is_pt_feeder and allocation is not None and not allocation.empty:
        # Per-cell DataFrame used by mode bar + scatter
        sa_cells = _build_sa_cells(allocation, pop_grid, empl_grid, sa_boundary)
    else:
        sa_cells = pd.DataFrame()

    # Plot (a): pyramids — PT_Feeder only
    if is_pt_feeder and allocation is not None and not allocation.empty:
        _plot_access_time_pyramids(allocation, pop_grid, empl_grid,
                                    sa_boundary, plot_dir)

    # Output: SA-stations overview map (both methods) — SA extent, two pies per station
    _plot_sa_stations_overview_map(sa_stations, sa_boundary, shares_df,
                                     plot_dir)

    # PT_Feeder-only outputs (require per-cell mode/access data)
    if is_pt_feeder and not sa_cells.empty:
        _plot_sa_stations_access_modes(sa_cells, sa_stations, plot_dir)
        _plot_sa_stations_access_time_violin(sa_cells, sa_stations, plot_dir)

    # Plot (c): Güteklassen stacked bar — PT_Feeder only (needs classified stops)
    if is_pt_feeder and feeder_stops is not None and rail_stops is not None:
        _plot_gueteklassen_stacked_bar(feeder_stops, rail_stops,
                                         sa_boundary, ca_boundary,
                                         pop_grid, empl_grid, plot_dir)

    print(f"  === Phase 4A plot suite complete ({method}) ===\n")


# ===============================================================================
# PUBLIC ENTRY POINT
# ===============================================================================

def _interactive_config():
    """Prompt the user to choose method, network folder, and temporal variant.

    Returns
    -------
    dict with keys:
        method        : 'municipal' | 'pt_feeder' | 'both'
        feeder_base: str  – resolved path to the selected feeder network directory
        rail_base:   str  – resolved path to the rail Unprojected directory
        temporal      : 'full_day' | 'all_day' | 'peak' | 'offpeak'
    """
    print("=" * 70)
    print("catchment_allocate.py — PIPELINE CONFIGURATION")
    print("=" * 70)

    # --- A. Method selection ---
    print("\nA. CATCHMENT METHOD")
    print("   1) Both        — run both and produce comparison diff plot")
    print("   2) Municipal   — commune centroid-based assignment")
    print("   3) PT-Feeder   — GTFS multimodal network routing")

    while True:
        choice = input("\n   Select method (1-3) [1]: ").strip() or "1"
        if choice in ('1', '2', '3'):
            break
        print("   Invalid selection. Please enter 1, 2, or 3.")

    method_map = {'1': 'both', '2': 'municipal', '3': 'pt_feeder'}
    method = method_map[choice]

    # --- B. Service version (needed for all methods — Municipal still needs
    #        rail_stops.gpkg to map municipalities to stations) ---
    need_feeder = method in ('pt_feeder', 'both')
    unprojected = paths.SERVICES_UNPROJECTED_SUBDIR

    if need_feeder:
        disc_root  = os.path.join(paths.MAIN, paths.FEEDER_LINES_DIR)
        disc_label = paths.FEEDER_LINES_DIR
        disc_probe = os.path.join(unprojected, 'pt_feeder_stops.gpkg')
    else:
        disc_root  = os.path.join(paths.MAIN, paths.RAIL_LINES_DIR)
        disc_label = paths.RAIL_LINES_DIR
        disc_probe = os.path.join(unprojected, 'All_Day', 'rail_stops_allday.gpkg')

    svc_versions = []
    if os.path.isdir(disc_root):
        svc_versions = sorted([
            d for d in os.listdir(disc_root)
            if os.path.isdir(os.path.join(disc_root, d))
            and os.path.exists(os.path.join(disc_root, d, disc_probe))
        ])

    if not svc_versions:
        raise FileNotFoundError(
            f"No service versions found under {disc_label} containing "
            f"'{disc_probe}'. Run services_network_builder.py first.")

    print("\nB. SERVICE VERSION")
    print(f"   Available versions under {disc_label}:")
    for i, sv in enumerate(svc_versions, 1):
        print(f"     {i}) {sv}")

    while True:
        raw = input(f"\n   Select service version [1]: ").strip() or '1'
        if raw.isdigit() and 1 <= int(raw) <= len(svc_versions):
            svc_version = svc_versions[int(raw) - 1]
            break
        if raw in svc_versions:
            svc_version = raw
            break
        print(f"   Invalid — enter a number 1–{len(svc_versions)} or the folder name.")

    # Resolve base paths (always Unprojected)
    feeder_base = os.path.join(paths.FEEDER_LINES_DIR, svc_version, unprojected)
    rail_base   = os.path.join(paths.RAIL_LINES_DIR,   svc_version, unprojected)

    # --- C. TEMPORAL NETWORK VARIANT ---
    print("\nC. TEMPORAL NETWORK VARIANT")
    print("   1) Full day — all services (top-level rail_stops.gpkg / pt_feeder_stops.gpkg)")
    print("   2) All-day  — services operating throughout the day (All_Day/*_allday.gpkg)")
    print("   3) Peak     — services operating during peak hours (Peak/*_peak.gpkg)")
    print("   4) Off-peak — services operating during off-peak hours (Off_Peak/*_offpeak.gpkg)")

    # Probe availability per option against the side we already discovered against
    if need_feeder:
        abs_probe_base = os.path.join(paths.MAIN, feeder_base)
        stem = 'pt_feeder_stops'
    else:
        abs_probe_base = os.path.join(paths.MAIN, rail_base)
        stem = 'rail_stops'

    # Map option number → temporal key
    tchoice_map = {'1': 'full_day', '2': 'all_day', '3': 'peak', '4': 'offpeak'}
    availability = {}
    for opt, key in tchoice_map.items():
        path = _temporal_paths(abs_probe_base, stem, key)
        availability[opt] = os.path.isfile(path)
    for opt, key in tchoice_map.items():
        if not availability[opt]:
            print(f"   ({_TEMPORAL_LABELS[key].lower()} files not found — option {opt} unavailable)")

    valid = {opt for opt, ok in availability.items() if ok}
    if not valid:
        raise FileNotFoundError(
            f"No temporal variant of {stem}*.gpkg found under {abs_probe_base}. "
            "Run services_network_builder.py first.")

    # Default from settings.TEMPORAL; fall back to '1' (full_day) when the
    # settings value is unavailable for the resolved base directory.
    _settings_temporal = _normalise_temporal(getattr(settings, 'TEMPORAL', 'all_day'))
    _settings_opt = next((opt for opt, key in tchoice_map.items()
                          if key == _settings_temporal), None)
    if _settings_opt and _settings_opt in valid:
        default_opt = _settings_opt
    else:
        default_opt = '1' if '1' in valid else sorted(valid)[0]
    while True:
        tchoice = input(f"\n   Select temporal variant (1-4) "
                         f"[default from settings: {_settings_temporal}] [{default_opt}]: "
                         ).strip() or default_opt
        if tchoice in valid:
            break
        print(f"   Invalid or unavailable selection. Choose from: {sorted(valid)}")

    temporal = tchoice_map[tchoice]

    # --- D. Plot generation ---
    print("\nD. PLOTS")
    default_plot = bool(getattr(settings, 'PLOT_CATCHMENT', False))
    default_str  = 'y' if default_plot else 'n'
    print(f"   Generate all catchment plots (municipal/PT-Feeder maps, access-time "
          f"plots, Güteklassen, Phase 4A suite)?")
    print(f"   Y/N — default '{default_str}' from settings.PLOT_CATCHMENT")
    while True:
        raw = input(f"\n   Generate plots? [{default_str}]: ").strip().lower() or default_str
        if raw in ('y', 'n', 'yes', 'no'):
            plot_catchment = raw.startswith('y')
            break
        print("   Invalid — enter y or n.")

    # --- E. INFRA PROJECTION (for plotted rail/feeder lines) ---
    # Sibling folders of Unprojected under <svc>/ that contain rail_lines.gpkg
    # are candidate projections (e.g. AS_2026_ZH_enhanced). The selected
    # projection is used by _load_rail_lines_for_plot and _load_feeder_lines.
    infra_projection = None
    proj_parent = os.path.dirname(os.path.join(paths.MAIN, rail_base))
    if os.path.isdir(proj_parent):
        projections = sorted([
            d for d in os.listdir(proj_parent)
            if os.path.isdir(os.path.join(proj_parent, d))
            and d != paths.SERVICES_UNPROJECTED_SUBDIR
            and os.path.isfile(os.path.join(proj_parent, d, 'rail_lines.gpkg'))
        ])
    else:
        projections = []

    if projections:
        print("\nE. INFRA PROJECTION (for plotted rail/feeder lines)")
        print("   0) Unprojected only (no projection)")
        for i, p in enumerate(projections, 1):
            print(f"   {i}) {p}")
        while True:
            raw = input(f"\n   Select projection [1]: ").strip() or '1'
            if raw == '0':
                infra_projection = None
                break
            if raw.isdigit() and 1 <= int(raw) <= len(projections):
                infra_projection = projections[int(raw) - 1]
                break
            if raw in projections:
                infra_projection = raw
                break
            print(f"   Invalid — enter 0–{len(projections)} or the folder name.")
    else:
        print("\nE. INFRA PROJECTION: no projected service versions found; "
              "Unprojected will be used for plotted lines.")

    # --- F. TRAVEL COST WEIGHTS ---
    print("\nF. TRAVEL COST WEIGHTS")
    print("   Controls how access-time components are weighted in the generalised-cost calculation.")
    print("   1) Calibrated — literature weights from cost_parameters.py "
          "(W_IVT, W_WAIT, W_WALK, W_BIKE, transfer penalty)")
    print("   2) Absolute   — raw minutes, all weights = 1.0")
    _settings_tcm = getattr(settings, 'TRAVEL_COST_METHOD', 'calibrated')
    _default_tcm  = '1' if _settings_tcm == 'calibrated' else '2'
    while True:
        raw = input(f"\n   Select (1-2) [default from settings: {_settings_tcm}] [{_default_tcm}]: "
                    ).strip() or _default_tcm
        if raw in ('1', '2'):
            break
        print("   Invalid — enter 1 or 2.")
    travel_cost_method = 'calibrated' if raw == '1' else 'absolute'

    # --- G. TRANSFER COST MODEL ---
    print("\nG. TRANSFER COST MODEL")
    print("   Controls how the per-transfer penalty is computed on feeder-graph transfer edges.")
    print("   1) Fixed value — single PI_TRANSFER_MIN (calibrated) or "
          "average_train_change_time (absolute), independent of connecting headway")
    print("   2) Explicit    — TRANSFER_WALK_MIN + t_wait_min(connecting_headway), "
          "weighted by W_TRANSFER (calibrated) or 1.0 (absolute)")
    _settings_tcm2 = getattr(settings, 'TRANSFER_COST_MODEL', 'fixed_value')
    _default_tcm2  = '1' if _settings_tcm2 == 'fixed_value' else '2'
    while True:
        raw = input(f"\n   Select (1-2) [default from settings: {_settings_tcm2}] [{_default_tcm2}]: "
                    ).strip() or _default_tcm2
        if raw in ('1', '2'):
            break
        print("   Invalid — enter 1 or 2.")
    transfer_cost_model = 'fixed_value' if raw == '1' else 'explicit'

    # --- Summary ---
    method_labels = {
        'municipal': 'Municipal (commune centroid)',
        'pt_feeder': 'PT-Feeder (GTFS multimodal)',
        'both':      'Both + diff comparison plot',
    }
    print("\n" + "-" * 70)
    print("  CONFIGURATION SUMMARY")
    print("-" * 70)
    print(f"  Method         : {method_labels[method]}")
    print(f"  Service version: {svc_version}")
    if need_feeder:
        print(f"  Feeder base    : {feeder_base}")
    print(f"  Rail base      : {rail_base}")
    print(f"  Temporal       : {_TEMPORAL_LABELS.get(_normalise_temporal(temporal), temporal)}")
    print(f"  Plot suite     : {'enabled' if plot_catchment else 'disabled'}")
    print(f"  Infra projection: {infra_projection if infra_projection else 'Unprojected'}")
    print(f"  Travel cost    : {travel_cost_method}")
    print(f"  Transfer model : {transfer_cost_model}")
    print("-" * 70)

    return {
        'method':              method,
        'feeder_base':         feeder_base,
        'rail_base':           rail_base,
        'temporal':            temporal,
        'plot_catchment':      plot_catchment,
        'infra_projection':    infra_projection,
        'travel_cost_method':  travel_cost_method,
        'transfer_cost_model': transfer_cost_model,
    }


def get_catchment(use_cache: bool, method: str = 'both',
                  feeder_base: str = '',
                  rail_base:   str = '',
                  temporal: str = 'all',
                  visualize: bool = True,
                  infra_projection: str = None) -> None:
    """Run the selected catchment method(s) and produce outputs.

    Parameters
    ----------
    use_cache    : If True, skip regeneration when all output files exist.
    method       : 'municipal' | 'pt_feeder' | 'both'
    feeder_base  : Path (relative to MAIN) to the feeder network directory —
                   either FEEDER_LINES_DIR/<svc>/Unprojected or
                   FEEDER_LINES_DIR/<svc>/Versions/<name>.
    rail_base    : Path (relative to MAIN) to the rail Unprojected directory —
                   RAIL_LINES_DIR/<svc>/Unprojected.
    temporal     : 'all' | 'peak' | 'offpeak' — which temporal variant
                   of the network files (stops + segments) to load.
    visualize    : If True, generate all plots produced by this module
                   (municipal/PT-Feeder maps, access-time plots, Güteklassen
                   comparison, diff plot, and the Phase 4A plot suite). Data
                   outputs (CSV/GPKG/Excel) are always written. When called
                   from main_new.py, pass settings.PLOT_CATCHMENT.
    infra_projection : Name of a projected infra version (sibling folder of
                   Unprojected under <svc>/) whose rail_lines.gpkg /
                   pt_feeder_lines.gpkg should be used by the line-plot
                   helpers. None falls back to Unprojected/*_lines.gpkg.
    """
    global _FEEDER_BASE, _RAIL_BASE, _INFRA_PROJECTION
    global MUNICIPAL_DATA_DIR, PT_FEEDER_DATA_DIR
    global MUNICIPAL_PLOT_DIR, PT_FEEDER_PLOT_DIR, GUETEKLASSEN_PLOT_DIR
    _FEEDER_BASE      = feeder_base
    _RAIL_BASE        = rail_base
    _INFRA_PROJECTION = infra_projection

    # Derive svc version from the network path and point output dirs under it.
    # feeder_base = 'data/Network/Feeder_Lines/{svc}/Unprojected'  → parent.name = svc
    _ref = feeder_base or rail_base
    if _ref:
        svc_version = os.path.basename(os.path.dirname(_ref))
        catchment_base.setup_versioned_dirs(svc_version)
        MUNICIPAL_DATA_DIR    = catchment_base.MUNICIPAL_DATA_DIR
        PT_FEEDER_DATA_DIR    = catchment_base.PT_FEEDER_DATA_DIR
        MUNICIPAL_PLOT_DIR    = catchment_base.MUNICIPAL_PLOT_DIR
        PT_FEEDER_PLOT_DIR    = catchment_base.PT_FEEDER_PLOT_DIR
        GUETEKLASSEN_PLOT_DIR = catchment_base.GUETEKLASSEN_PLOT_DIR

    os.chdir(paths.MAIN)
    _ensure_dirs()

    print("=" * 70)
    print("CATCHMENT ALLOCATION")
    print(f"  Method : {method}")
    if method in ('pt_feeder', 'both'):
        print(f"  Feeder   : {feeder_base}")
        print(f"  Rail     : {rail_base}")
        print(f"  Temporal : {_TEMPORAL_LABELS.get(_normalise_temporal(temporal), temporal)}")
        _xp_sec = _get_active_transfer_penalty_sec()
        print(f"  Travel cost method : {settings.TRAVEL_COST_METHOD}")
        print(f"  Transfer cost model: {settings.TRANSFER_COST_MODEL}")
        print(f"  Transfer penalty   : {_xp_sec:.0f}s "
              f"({_xp_sec / 60:.2f} min, "
              f"{'comfort-weighted PI_TRANSFER' if settings.TRAVEL_COST_METHOD == 'calibrated' else 'raw average_train_change_time'})")
    print("=" * 70)

    total_start = time.time()

    # Cache check (only meaningful when running both methods)
    if use_cache and method == 'both':
        expected_files = [
            os.path.join(catchment_base.POP_EMPL_DATA_DIR,
                          'municipal_pop_empl_summary.csv'),
            os.path.join(PT_FEEDER_DATA_DIR, 'catchment.gpkg'),
            os.path.join(MUNICIPAL_DATA_DIR, 'catchment.gpkg'),
            os.path.join(os.path.dirname(PT_FEEDER_PLOT_DIR),
                         'catchment_diff_municipal_vs_pt_feeder.pdf'),
        ]
        all_exist = all(os.path.exists(f) for f in expected_files)
        if all_exist:
            print("Using cached catchment files - skipping generation.")
            return
        else:
            missing = [f for f in expected_files if not os.path.exists(f)]
            print(f"Cache enabled but {len(missing)} file(s) missing. Regenerating ...")

    # --- Step 1: Read shared data from catchment_base cache ---
    # Step 1 (boundary, filtered grids, per-municipality summary, raster +
    # choropleth plots) is produced by `python catchment_base.py`. This module
    # only consumes the cached outputs — if any are missing, the cache readers
    # raise FileNotFoundError with an actionable message.
    print("\n[Step 1] Reading shared data from catchment_base cache ...")
    boundary  = _load_catchment_boundary()
    pop_grid  = load_population_grid_cached()
    empl_grid = load_employment_grid_cached()

    muni_catchment = None
    pt_catchment   = None
    pt_allocation  = None

    if method in ('municipal', 'both'):
        rail_stations = _load_rail_stations(boundary, temporal, buffer=0)
        result = _run_municipal_method(boundary, rail_stations,
                                        pop_grid=pop_grid, empl_grid=empl_grid,
                                        visualize=visualize)
        if result is not None:
            muni_catchment, assignment_df, muni_gdf, bfs_col = result
        else:
            muni_catchment = None

        # Municipal catchment visualisation plots — use projection-aware stops
        if muni_catchment is not None and visualize:
            plot_rs = _get_plot_rail_stations(rail_stations, scope='ca')
            _plot_municipal_catchments(muni_catchment, pop_grid, empl_grid,
                                       plot_rs, boundary,
                                       assignment_df, muni_gdf, bfs_col)
            _plot_municipal_catchments_network(muni_catchment, plot_rs,
                                               boundary, temporal)

    if method in ('pt_feeder', 'both'):
        pt_catchment, pt_allocation = _run_pt_feeder_method(
            boundary, pop_grid, empl_grid, temporal, visualize=visualize)

    if method == 'both' and visualize:
        print("\n[Comparison] Building diff plot ...")
        rail_stations = _load_rail_stations(boundary, temporal, buffer=0)
        plot_rs = _get_plot_rail_stations(rail_stations, scope='ca')
        _build_diff_plot(muni_catchment, pt_catchment, pt_allocation,
                         pop_grid, empl_grid, plot_rs, boundary)

    elapsed = time.time() - total_start
    print(f"\nCatchment allocation complete - {elapsed:.1f}s total")


if __name__ == '__main__':
    _INTERACTIVE_MODE = True   # enables CLI prompts (e.g. "use existing assignment?")
    cfg = _interactive_config()
    settings.TRAVEL_COST_METHOD  = cfg['travel_cost_method']
    settings.TRANSFER_COST_MODEL = cfg['transfer_cost_model']
    settings.TEMPORAL            = cfg['temporal']
    get_catchment(use_cache=False,
                  method=cfg['method'],
                  feeder_base=cfg['feeder_base'],
                  rail_base=cfg['rail_base'],
                  temporal=cfg['temporal'],
                  visualize=bool(cfg['plot_catchment']),
                  infra_projection=cfg['infra_projection'])
