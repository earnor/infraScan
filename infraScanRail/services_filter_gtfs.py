# services_filter_gtfs.py
#
# GTFS stop coordinates are always WGS84 (EPSG:4326) and are reprojected to EPSG:2056
# before spatial filtering.

import json
import os
import shutil
import time
from datetime import datetime, timedelta

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import paths
import settings

_start_time = time.time()

# ---------------------------------------------------------------------------
# Input folder default — may be overridden by _configure_pipeline() at runtime
# ---------------------------------------------------------------------------
GTFS_INPUT_FOLDER  = 'GTFS_SVC2026_CH_raw'
GTFS_OUTPUT_FOLDER = 'GTFS_SVC2026_ZH'

# ---------------------------------------------------------------------------
# Route-type lookups — kept separate so Task 2 can differentiate the two groups
# ---------------------------------------------------------------------------
PT_FEEDER_ROUTE_TYPES = {
    900:  'Tram',
    401:  'Metro',
    700:  'Bus',
    702:  'Express Bus',
    715:  'On-demand Bus',
    1000: 'Ship',
    1400: 'Funicular',
}

RAIL_ROUTE_TYPES = {
    102: 'Long-Distance Rail',
    103: 'Inter-Regional Rail',
    106: 'Regional Rail',
    109: 'S-Bahn / Suburban Rail',
}

# ---------------------------------------------------------------------------
# Route-type overrides — reclassify specific routes for testing
# Key: route_short_name (str), Value: new route_type (int)
# Set to empty dict {} to disable all overrides
# ---------------------------------------------------------------------------
ROUTE_TYPE_OVERRIDES = {
    '18': 109,   # Uncomment to treat Tram line 18 as S-Bahn (109)
}

# ---------------------------------------------------------------------------
# Retention policies — how each route type's trips are clipped spatially
#   'full'      : keep all stops of any trip touching the canton (106, 109)
#   'border+1'  : envelope around canton stops + 1 stop padding each side (102, 103)
#   'border-1'  : only stops within the canton (PT feeders)
# ---------------------------------------------------------------------------
FULL_ROUTE_TYPES      = {106, 109}
BORDER_PLUS1_ROUTE_TYPES = {102, 103}

# Combined set used for the route-type filter pass
ALL_RETAINED_ROUTE_TYPES = {**PT_FEEDER_ROUTE_TYPES, **RAIL_ROUTE_TYPES}

# Required GTFS files — raise if any are absent
REQUIRED_FILES = [
    'stops.txt',
    'routes.txt',
    'trips.txt',
    'stop_times.txt',
    'calendar.txt',
    'calendar_dates.txt',
    'agency.txt',
]

# Optional GTFS files handled explicitly
OPTIONAL_FILES = ['transfers.txt', 'feed_info.txt', 'frequencies.txt']

CODEBASE_CRS = 'EPSG:2056'
GTFS_CRS     = 'EPSG:4326'

# Tier 1 service duration filter: drop service_ids active on fewer than this
# many weekday days across the full timetable year. Removes one-off events,
# single-weekend specials, and very short construction-period services.
MIN_WEEKDAY_ACTIVE_DAYS = 20

# ---------------------------------------------------------------------------
# Boundary GeoPackage paths
# ---------------------------------------------------------------------------
CANTONS_GPKG       = paths.CANTON_BOUNDARIES_GPKG
BEZIRKE_GPKG       = paths.BEZIRKE_BOUNDARIES_GPKG
MUNICIPALITIES_GPKG = paths.MUNICIPAL_BOUNDARIES_GPKG

# Timetable year — extracted from the input folder name for the default
# output folder suggestion. Falls back to 'GTFS' if not parseable.
_TIMETABLE_YEAR_TAG = 'SVC2026'


# ===========================================================================
# Pipeline Configuration TUI — GTFS folders only
# ===========================================================================

def _configure_gtfs_folders():
    """Prompt the user for GTFS input and output folder names.

    The catchment area boundary is read from the GeoPackage exported by
    initialisation.py. Run initialisation.py first if that file does not exist.

    Returns a dict with keys:
        gtfs_input_folder  : str – subfolder name under GTFS_TRANSIT_DIR
        gtfs_output_folder : str – subfolder name under GTFS_TRANSIT_DIR
    """
    print("=" * 70)
    print("services_filter_gtfs.py — Pipeline configuration")
    print("=" * 70)
    print(f"\n   Catchment boundary : {paths.CATCHMENT_AREA_BOUNDARY_GPKG}")
    print("   Run initialisation.py first if that file does not exist.\n")

    # --- A. GTFS input folder ------------------------------------------------
    print("A. GTFS input folder")
    print(f"   Base path: {paths.GTFS_TRANSIT_DIR}")
    gtfs_base = os.path.join(paths.MAIN, paths.GTFS_TRANSIT_DIR)
    if os.path.isdir(gtfs_base):
        subfolders = sorted([
            d for d in os.listdir(gtfs_base)
            if os.path.isdir(os.path.join(gtfs_base, d)) and '_raw' in d
        ])
        if subfolders:
            print(f"   Available subfolders:")
            for i, sf in enumerate(subfolders, 1):
                print(f"     {i}) {sf}")
    else:
        subfolders = []

    while True:
        gtfs_input = input(
            f"\n   Enter GTFS input folder name [{GTFS_INPUT_FOLDER}]: "
        ).strip() or GTFS_INPUT_FOLDER
        full_input = os.path.join(gtfs_base, gtfs_input)
        if os.path.isdir(full_input):
            break
        if gtfs_input.isdigit() and 1 <= int(gtfs_input) <= len(subfolders):
            gtfs_input = subfolders[int(gtfs_input) - 1]
            break
        print(f"   Folder not found: {full_input}")
        print(f"   Please enter an existing subfolder name.")

    # --- B. GTFS output folder -----------------------------------------------
    # Derive default output name from catchment gpkg primary attribute
    _area_tag = 'filtered'
    _ca_gpkg_path = os.path.join(paths.MAIN, paths.CATCHMENT_AREA_BOUNDARY_GPKG)
    if os.path.isfile(_ca_gpkg_path):
        try:
            _ca_preview = gpd.read_file(_ca_gpkg_path)
            _primary = str(_ca_preview.iloc[0].get('primary', '')).strip()
            if _primary and _primary not in ('coordinates', 'nan', ''):
                _area_tag = _primary.replace(', ', '_').replace(' ', '')[:30]
        except Exception:
            pass
    default_output = f"GTFS_{_TIMETABLE_YEAR_TAG}_{_area_tag}"

    print("\nB. GTFS output folder")
    print(f"   Base path: {paths.GTFS_TRANSIT_DIR}")
    gtfs_output = input(
        f"\n   Enter output folder name [{default_output}]: "
    ).strip() or default_output

    return {
        'gtfs_input_folder':  gtfs_input,
        'gtfs_output_folder': gtfs_output,
    }


import argparse as _ap
_cli = _ap.ArgumentParser(add_help=False)
_cli.add_argument('--input-folder',  default=None)
_cli.add_argument('--output-folder', default=None)
_cli_args, _ = _cli.parse_known_args()

if _cli_args.input_folder and _cli_args.output_folder:
    GTFS_INPUT_FOLDER  = _cli_args.input_folder
    GTFS_OUTPUT_FOLDER = _cli_args.output_folder
    print(f"  Non-interactive: input='{GTFS_INPUT_FOLDER}', output='{GTFS_OUTPUT_FOLDER}'")
else:
    _gtfs_cfg = _configure_gtfs_folders()
    GTFS_INPUT_FOLDER  = _gtfs_cfg['gtfs_input_folder']
    GTFS_OUTPUT_FOLDER = _gtfs_cfg['gtfs_output_folder']


# ===========================================================================
# Helpers
# ===========================================================================

def _gtfs_path(filename):
    return os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_INPUT_FOLDER, filename)


def _out_path(filename):
    return os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_OUTPUT_FOLDER, filename)


def _load_required(filename):
    fpath = _gtfs_path(filename)
    if not os.path.isfile(fpath):
        raise FileNotFoundError(
            f"Required GTFS file not found: {fpath}\n"
            f"Please place all required GTFS files in: {os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_INPUT_FOLDER)}"
        )
    print(f"  Loading {filename} ...", end=' ', flush=True)
    df = pd.read_csv(fpath, dtype=str, low_memory=False)
    print(f"{len(df):,} rows")
    return df


def _load_optional(filename):
    """Returns (DataFrame, True) if present, (None, False) if absent."""
    fpath = _gtfs_path(filename)
    if not os.path.isfile(fpath):
        return None, False
    print(f"  Loading {filename} ...", end=' ', flush=True)
    df = pd.read_csv(fpath, dtype=str, low_memory=False)
    print(f"{len(df):,} rows")
    return df, True


def _write(df, filename):
    out = _out_path(filename)
    df.to_csv(out, index=False, encoding='utf-8')
    print(f"  Wrote {filename}: {len(df):,} rows")


# ===========================================================================
# Step 0 — prepare output directory
# ===========================================================================

os.chdir(paths.MAIN)  # All path constants in paths.py are relative to MAIN

print("=" * 70)
print("services_filter_gtfs.py")
print(f"  CATCHMENT_METHOD      : {settings.CATCHMENT_METHOD}")
print(f"  Catchment boundary    : {paths.CATCHMENT_AREA_BOUNDARY_GPKG}")
print(f"  GTFS source           : {os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_INPUT_FOLDER)}")
print(f"  GTFS output           : {os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_OUTPUT_FOLDER)}")
print(f"  Spatial CRS           : {CODEBASE_CRS}")
print("=" * 70)

_gtfs_output_dir = os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_OUTPUT_FOLDER)
if os.path.isdir(_gtfs_output_dir):
    print(f"\n[0] Output directory exists — removing for clean overwrite: {_gtfs_output_dir}")
    shutil.rmtree(_gtfs_output_dir)
os.makedirs(_gtfs_output_dir)
print(f"[0] Created output directory: {_gtfs_output_dir}")


# ===========================================================================
# Step 1 — load catchment boundary from initialisation.py output
# ===========================================================================

print(f"\n[1] Loading catchment boundary ...")
_ca_gpkg_path = os.path.join(paths.MAIN, paths.CATCHMENT_AREA_BOUNDARY_GPKG)
if not os.path.isfile(_ca_gpkg_path):
    raise FileNotFoundError(
        f"Catchment area buffer not found: {_ca_gpkg_path}\n"
        f"Run initialisation.py first to define and export the catchment area."
    )
_ca_gdf = gpd.read_file(_ca_gpkg_path).to_crs(CODEBASE_CRS)
canton_polygon  = _ca_gdf.geometry.iloc[0]
_ca_row         = _ca_gdf.iloc[0]
_ca_admin_level = str(_ca_row.get('admin_level', 'unknown'))
_ca_primary     = str(_ca_row.get('primary', ''))
_ca_buffer_m    = float(_ca_row.get('buffer_m', 0))
print(f"  Loaded: {_ca_gpkg_path}")
print(f"  Admin level : {_ca_admin_level}")
print(f"  Primary     : {_ca_primary}")
print(f"  Buffer      : {_ca_buffer_m:.0f} m")
print(f"  Bounds      : {canton_polygon.bounds}")


# ===========================================================================
# Step 2 — load required GTFS files
# ===========================================================================

print("\n[2] Loading required GTFS files ...")
for f in REQUIRED_FILES:
    if not os.path.isfile(_gtfs_path(f)):
        raise FileNotFoundError(
            f"Required GTFS file not found: {_gtfs_path(f)}\n"
            f"Place all required GTFS files in: {os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_INPUT_FOLDER)}"
        )

stops          = _load_required('stops.txt')
routes_raw     = _load_required('routes.txt')
trips_raw      = _load_required('trips.txt')
stop_times_raw = _load_required('stop_times.txt')
calendar_raw   = _load_required('calendar.txt')
cal_dates_raw  = _load_required('calendar_dates.txt')
agency_raw     = _load_required('agency.txt')

orig_counts = {
    'stops':    len(stops),
    'routes':   len(routes_raw),
    'trips':    len(trips_raw),
    'agencies': len(agency_raw),
}

# Derive timetable validity period from calendar.txt date ranges
_cal_starts = pd.to_datetime(calendar_raw['start_date'], format='%Y%m%d', errors='coerce')
_cal_ends   = pd.to_datetime(calendar_raw['end_date'],   format='%Y%m%d', errors='coerce')
TIMETABLE_START = _cal_starts.min().strftime('%Y-%m-%d')
TIMETABLE_END   = _cal_ends.max().strftime('%Y-%m-%d')
print(f"\n  Timetable period (from calendar.txt): {TIMETABLE_START} to {TIMETABLE_END}")


# ===========================================================================
# Step 3 — service duration filter (Tier 1)
#
# Compute effective weekday active days per service_id using calendar.txt +
# calendar_dates.txt, and drop service_ids below MIN_WEEKDAY_ACTIVE_DAYS.
# This removes one-off events, single-weekend specials, and very short
# construction-period services before any downstream processing.
# ===========================================================================

print(f"\n[3] Service duration filter (Tier 1): min {MIN_WEEKDAY_ACTIVE_DAYS} weekday active days ...")

def _compute_weekday_active_days(cal_df, cal_dates_df, tt_start, tt_end):
    """Return a Series mapping service_id → number of effective weekday (Mon–Fri)
    active days, accounting for calendar_dates exceptions."""
    start = datetime.strptime(tt_start, '%Y%m%d' if len(tt_start) == 8 else '%Y-%m-%d')
    end   = datetime.strptime(tt_end,   '%Y%m%d' if len(tt_end)   == 8 else '%Y-%m-%d')

    # Build full date range
    all_dates = []
    d = start
    while d <= end:
        all_dates.append(d)
        d += timedelta(days=1)

    # Day-of-week column names in calendar.txt (0=Mon ... 6=Sun)
    dow_cols = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

    # Parse calendar base schedule
    cal = cal_df.copy()
    for col in dow_cols:
        cal[col] = pd.to_numeric(cal[col], errors='coerce').fillna(0).astype(int)
    cal['_start'] = pd.to_datetime(cal['start_date'], format='%Y%m%d', errors='coerce')
    cal['_end']   = pd.to_datetime(cal['end_date'],   format='%Y%m%d', errors='coerce')

    # Parse calendar_dates exceptions
    cde = cal_dates_df.copy()
    cde['_date'] = pd.to_datetime(cde['date'], format='%Y%m%d', errors='coerce')
    cde['_type'] = pd.to_numeric(cde['exception_type'], errors='coerce').fillna(0).astype(int)

    # Build removal and addition sets per service_id
    removals = cde.loc[cde['_type'] == 2].groupby('service_id')['_date'].apply(set).to_dict()
    additions = cde.loc[cde['_type'] == 1].groupby('service_id')['_date'].apply(set).to_dict()

    results = {}
    for _, row in cal.iterrows():
        sid = row['service_id']
        s_start = row['_start']
        s_end   = row['_end']
        if pd.isna(s_start) or pd.isna(s_end):
            results[sid] = 0
            continue

        svc_removals  = removals.get(sid, set())
        svc_additions = additions.get(sid, set())

        count = 0
        for dt in all_dates:
            if dt.weekday() >= 5:  # Skip Sat/Sun — we only count weekdays
                continue
            # Base schedule: is this date within range and matching day-of-week?
            in_base = (s_start <= dt <= s_end) and (row[dow_cols[dt.weekday()]] == 1)
            # Apply exceptions
            if dt in svc_removals:
                active = False
            elif dt in svc_additions:
                active = True
            else:
                active = in_base
            if active:
                count += 1
        results[sid] = count

    return pd.Series(results, name='weekday_active_days')


_tt_start = TIMETABLE_START.replace('-', '')
_tt_end   = TIMETABLE_END.replace('-', '')

# Use raw calendar/dates (before any filtering)
_weekday_active = _compute_weekday_active_days(calendar_raw, cal_dates_raw, _tt_start, _tt_end)
_dropped_sids = set(_weekday_active[_weekday_active < MIN_WEEKDAY_ACTIVE_DAYS].index)
_kept_sids = set(_weekday_active[_weekday_active >= MIN_WEEKDAY_ACTIVE_DAYS].index)

# Also keep service_ids that appear only in calendar_dates (no calendar.txt row)
# — these are already edge cases; Tier 1 only filters what it can measure.
_cal_only_sids = set(cal_dates_raw['service_id']) - set(calendar_raw['service_id'])

print(f"  Service IDs in calendar.txt: {len(calendar_raw):,}")
print(f"  Service IDs below threshold: {len(_dropped_sids):,}")
print(f"  Service IDs retained       : {len(_kept_sids):,}")
if _cal_only_sids:
    print(f"  Service IDs only in calendar_dates (unmeasured, kept): {len(_cal_only_sids):,}")

# Filter trips to retained service_ids
_retained_service_ids_t1 = _kept_sids | _cal_only_sids
_trips_before_t1 = len(trips_raw)
trips_raw = trips_raw[trips_raw['service_id'].isin(_retained_service_ids_t1)].copy()
print(f"  Trips after Tier 1 filter  : {len(trips_raw):,} / {_trips_before_t1:,}")

# Store stats for the report
_tier1_stats = {
    'total_sids':   len(_weekday_active),
    'dropped_sids': len(_dropped_sids),
    'kept_sids':    len(_kept_sids),
    'min_threshold': MIN_WEEKDAY_ACTIVE_DAYS,
    'trips_before': _trips_before_t1,
    'trips_after':  len(trips_raw),
}


# ===========================================================================
# Step 4 — route type filtering (always before spatial filtering)
# ===========================================================================

print("\n[4] Route-type filtering ...")
routes_raw['route_type_int'] = pd.to_numeric(routes_raw['route_type'], errors='coerce')

# Apply route-type overrides if configured
if ROUTE_TYPE_OVERRIDES:
    for idx, row in routes_raw.iterrows():
        short_name = str(row.get('route_short_name', '')).strip()
        if short_name in ROUTE_TYPE_OVERRIDES:
            original_type = int(row['route_type_int']) if pd.notna(row['route_type_int']) else None
            new_type = ROUTE_TYPE_OVERRIDES[short_name]
            routes_raw.at[idx, 'route_type_int'] = new_type
            routes_raw.at[idx, 'route_type'] = str(new_type)
            print(f"    Override: route '{short_name}' reclassified from {original_type} to {new_type}")

routes = routes_raw[routes_raw['route_type_int'].isin(ALL_RETAINED_ROUTE_TYPES)].copy()

# Tag each route with its mode class — carried through for Task 2, not written to GTFS
routes['mode_class'] = routes['route_type_int'].apply(
    lambda rt: 'rail' if int(rt) in RAIL_ROUTE_TYPES else 'pt_feeder'
)
routes = routes.drop(columns=['route_type_int'])

n_pt  = (routes['mode_class'] == 'pt_feeder').sum()
n_rl  = (routes['mode_class'] == 'rail').sum()
print(f"  Routes after route-type filter: {len(routes):,} / {len(routes_raw):,}"
      f"  ({n_pt:,} pt_feeder, {n_rl:,} rail)")

retained_route_ids = set(routes['route_id'])

# Cascade into trips so subsequent steps work on retained routes only
trips_typed = trips_raw[trips_raw['route_id'].isin(retained_route_ids)].copy()
print(f"  Trips after route-type cascade: {len(trips_typed):,} / {len(trips_raw):,}")


# ===========================================================================
# Step 5 — spatial filtering: stops within study area boundary
# ===========================================================================

_area_desc = _ca_primary if _ca_primary and _ca_primary not in ('coordinates', 'nan', '') else 'Switzerland'
print(f"\n[5] Spatial filtering of stops to {_area_desc} ...")

stops['stop_lon_f'] = pd.to_numeric(stops['stop_lon'], errors='coerce')
stops['stop_lat_f'] = pd.to_numeric(stops['stop_lat'], errors='coerce')
stops_geo = gpd.GeoDataFrame(
    stops,
    geometry=[Point(lon, lat) for lon, lat in zip(stops['stop_lon_f'], stops['stop_lat_f'])],
    crs=GTFS_CRS  # GTFS coordinates are always WGS84
)
stops_geo = stops_geo.to_crs(CODEBASE_CRS)

stops_in_canton = stops_geo[stops_geo.geometry.within(canton_polygon)].copy()

# Add Swiss coordinate columns (EPSG:2056 convention: N = northing/y, E = easting/x)
stops_in_canton['stop_N'] = stops_in_canton.geometry.y
stops_in_canton['stop_E'] = stops_in_canton.geometry.x

# Drop helper and geometry columns; retain original stop_lon / stop_lat (WGS84)
stops_filtered = stops_in_canton.drop(
    columns=['stop_lon_f', 'stop_lat_f', 'geometry'], errors='ignore'
)

retained_stop_ids = set(stops_in_canton['stop_id'])
print(f"  Stops within canton: {len(stops_filtered):,} / {len(stops):,}")


# ===========================================================================
# Step 6 — assign retention policy to each trip
# ===========================================================================

print("\n[6] Assigning retention policy per trip ...")

# Build route_type_int → retention policy mapping
def _retention_policy(rt):
    if rt in FULL_ROUTE_TYPES:
        return 'full'
    if rt in BORDER_PLUS1_ROUTE_TYPES:
        return 'border+1'
    return 'border-1'

_route_policy = routes[['route_id']].copy()
_route_policy['route_type_int'] = pd.to_numeric(
    routes['route_type'], errors='coerce'
).astype('Int64')
_route_policy['retention_policy'] = _route_policy['route_type_int'].apply(_retention_policy)

trips_typed = trips_typed.merge(
    _route_policy[['route_id', 'retention_policy']], on='route_id', how='left'
)
trips_typed['retention_policy'] = trips_typed['retention_policy'].fillna('border-1')

for pol in ['full', 'border+1', 'border-1']:
    n = (trips_typed['retention_policy'] == pol).sum()
    print(f"  {pol:<10}: {n:,} trips")


# ===========================================================================
# Step 7 — filter stop_times & trips per retention policy
# ===========================================================================

print("\n[7] Filtering stop_times per retention policy ...")

# Pre-filter stop_times to only typed trips (route-type + Tier 1 filtered)
typed_trip_ids = set(trips_typed['trip_id'])
st_typed = stop_times_raw[stop_times_raw['trip_id'].isin(typed_trip_ids)].copy()

# Ensure stop_sequence is numeric for ordering
st_typed['stop_sequence_int'] = pd.to_numeric(st_typed['stop_sequence'], errors='coerce')

# Tag each stop_time row with whether the stop is inside the canton
st_typed['_in_canton'] = st_typed['stop_id'].isin(retained_stop_ids)

# Build trip → policy lookup
_trip_policy = trips_typed.set_index('trip_id')['retention_policy']
st_typed['_policy'] = st_typed['trip_id'].map(_trip_policy)

# --- border-1: only canton stops ------------------------------------------
mask_bm1 = st_typed['_policy'] == 'border-1'
st_bm1 = st_typed.loc[mask_bm1 & st_typed['_in_canton']].copy()
print(f"  border-1  : {len(st_bm1):,} stop_time rows")

# --- full: all stops of trips that touch the canton -----------------------
mask_full = st_typed['_policy'] == 'full'
_full_trips_touching = set(
    st_typed.loc[mask_full & st_typed['_in_canton'], 'trip_id']
)
st_full = st_typed.loc[mask_full & st_typed['trip_id'].isin(_full_trips_touching)].copy()
print(f"  full      : {len(st_full):,} stop_time rows  "
      f"({len(_full_trips_touching):,} trips touch canton)")

# --- border+1: envelope [min_canton_idx-1 .. max_canton_idx+1] ------------
mask_bp1 = st_typed['_policy'] == 'border+1'
st_bp1_all = st_typed.loc[mask_bp1].copy()

# For each trip, find the min and max stop_sequence of in-canton stops
_canton_seqs = (
    st_bp1_all.loc[st_bp1_all['_in_canton']]
    .groupby('trip_id')['stop_sequence_int']
    .agg(['min', 'max'])
    .rename(columns={'min': '_seq_min', 'max': '_seq_max'})
)
# Only trips that have at least one canton stop
st_bp1_all = st_bp1_all.merge(_canton_seqs, on='trip_id', how='inner')

# Find per-trip the actual sequence values one step before min and one after max
# We need the largest sequence < _seq_min and the smallest sequence > _seq_max
def _envelope_bounds(group):
    """Return (lower_bound, upper_bound) sequence values for the envelope."""
    seq_min = group['_seq_min'].iloc[0]
    seq_max = group['_seq_max'].iloc[0]
    seqs = group['stop_sequence_int'].sort_values()
    # One stop before the first canton stop
    before = seqs[seqs < seq_min]
    lower = before.iloc[-1] if len(before) > 0 else seq_min
    # One stop after the last canton stop
    after = seqs[seqs > seq_max]
    upper = after.iloc[0] if len(after) > 0 else seq_max
    return pd.Series({'_env_lo': lower, '_env_hi': upper})

_env_bounds = st_bp1_all.groupby('trip_id').apply(_envelope_bounds, include_groups=False)
st_bp1_all = st_bp1_all.merge(_env_bounds, on='trip_id', how='left')

st_bp1 = st_bp1_all.loc[
    (st_bp1_all['stop_sequence_int'] >= st_bp1_all['_env_lo']) &
    (st_bp1_all['stop_sequence_int'] <= st_bp1_all['_env_hi'])
].copy()
print(f"  border+1  : {len(st_bp1):,} stop_time rows  "
      f"({len(_canton_seqs):,} trips touch canton)")

# --- combine all three groups ---------------------------------------------
_helper_cols = ['stop_sequence_int', '_in_canton', '_policy',
                '_seq_min', '_seq_max', '_env_lo', '_env_hi']
for _df in (st_bm1, st_full, st_bp1):
    _df.drop(columns=[c for c in _helper_cols if c in _df.columns],
             inplace=True, errors='ignore')

stop_times = pd.concat([st_bm1, st_full, st_bp1], ignore_index=True)
print(f"  Total stop_times retained : {len(stop_times):,} / {len(stop_times_raw):,}")

# Derive the set of retained trips (any trip present in stop_times)
trips_with_canton_stops = set(stop_times['trip_id'])
trips = trips_typed[trips_typed['trip_id'].isin(trips_with_canton_stops)].copy()
print(f"  Trips retained            : {len(trips):,} / {len(trips_typed):,}")

# Expand retained_stop_ids to include out-of-canton stops now referenced
extra_stop_ids = set(stop_times['stop_id']) - retained_stop_ids
if extra_stop_ids:
    print(f"  Extra out-of-canton stops added: {len(extra_stop_ids):,}")


# ===========================================================================
# Step 8 — filter routes to those with at least one retained trip
# ===========================================================================

print("\n[8] Filtering routes to those with retained trips ...")
active_route_ids = set(trips['route_id'])
routes = routes[routes['route_id'].isin(active_route_ids)].copy()
print(f"  Routes retained: {len(routes):,}")


# ===========================================================================
# Step 9 — filter agencies
# ===========================================================================

print("\n[9] Filtering agencies to those operating retained routes ...")
active_agency_ids = set(routes['agency_id'].dropna())
agency = agency_raw[agency_raw['agency_id'].isin(active_agency_ids)].copy()
print(f"  Agencies retained: {len(agency):,} / {len(agency_raw):,}")


# ===========================================================================
# Step 10 — filter calendar and calendar_dates
# ===========================================================================

print("\n[10] Filtering calendar and calendar_dates to retained service IDs ...")
retained_service_ids = set(trips['service_id'])

calendar = calendar_raw[calendar_raw['service_id'].isin(retained_service_ids)].copy()
cal_dates = cal_dates_raw[cal_dates_raw['service_id'].isin(retained_service_ids)].copy()
print(f"  calendar rows retained: {len(calendar):,} / {len(calendar_raw):,}")
print(f"  calendar_dates rows retained: {len(cal_dates):,} / {len(cal_dates_raw):,}")


# ===========================================================================
# Step 11 — optional files
# ===========================================================================

print("\n[11] Processing optional GTFS files ...")
skipped_optional = []

# Use the expanded stop set (canton + out-of-canton stops from full/border+1 trips)
all_retained_stop_ids = retained_stop_ids | extra_stop_ids

transfers, transfers_present = _load_optional('transfers.txt')
if transfers_present:
    transfers = transfers[
        transfers['from_stop_id'].isin(all_retained_stop_ids) &
        transfers['to_stop_id'].isin(all_retained_stop_ids)
    ].copy()
    print(f"    transfers.txt retained: {len(transfers):,} rows")
else:
    skipped_optional.append('transfers.txt')
    print("    transfers.txt: not present — skipped")

feed_info, feed_info_present = _load_optional('feed_info.txt')
if feed_info_present:
    print(f"    feed_info.txt: copied unchanged ({len(feed_info):,} rows)")
else:
    skipped_optional.append('feed_info.txt')
    print("    feed_info.txt: not present — skipped")

frequencies, frequencies_present = _load_optional('frequencies.txt')
if frequencies_present:
    _freq_before = len(frequencies)
    retained_trip_ids = set(trips['trip_id'])
    frequencies = frequencies[frequencies['trip_id'].isin(retained_trip_ids)].copy()
    print(f"    frequencies.txt retained: {len(frequencies):,} / {_freq_before:,} rows")
else:
    skipped_optional.append('frequencies.txt')
    print("    frequencies.txt: not present — skipped")


# ===========================================================================
# Step 12 — write output
# ===========================================================================

print(f"\n[12] Writing filtered GTFS to: {_gtfs_output_dir}")

# Expand stops_filtered with out-of-canton stops referenced by full/border+1 trips.
# Also include their parent stations (location_type=1) so that
# services_network_builder.py can resolve parent-station coordinates.
if extra_stop_ids:
    extra_stops = stops[stops['stop_id'].isin(extra_stop_ids)].copy()
    # Find parent stations of these extra stops (if not already retained)
    existing_stop_ids = retained_stop_ids | extra_stop_ids
    if 'parent_station' in extra_stops.columns:
        parent_ids = set(extra_stops['parent_station'].dropna()) - existing_stop_ids
        if parent_ids:
            extra_parents = stops[stops['stop_id'].isin(parent_ids)].copy()
            extra_stops = pd.concat([extra_stops, extra_parents], ignore_index=True)
    # Add Swiss coordinate columns for the extra stops
    extra_geo = gpd.GeoDataFrame(
        extra_stops,
        geometry=[Point(lon, lat) for lon, lat in
                  zip(pd.to_numeric(extra_stops['stop_lon'], errors='coerce'),
                      pd.to_numeric(extra_stops['stop_lat'], errors='coerce'))],
        crs=GTFS_CRS,
    ).to_crs(CODEBASE_CRS)
    extra_stops['stop_N'] = extra_geo.geometry.y
    extra_stops['stop_E'] = extra_geo.geometry.x
    extra_stops = extra_stops.drop(columns=['stop_lon_f', 'stop_lat_f'], errors='ignore')
    stops_filtered = pd.concat([stops_filtered, extra_stops], ignore_index=True)
    print(f"  stops_filtered expanded with {len(extra_stops):,} out-of-canton stops (incl. parent stations)")

# Write mode_class sidecar before stripping the column from routes
mode_class = routes[['route_id', 'mode_class']].copy()
_write(mode_class, 'mode_class.txt')

# Strip mode_class and retention_policy before writing standard GTFS files
routes_gtfs = routes.drop(columns=['mode_class'])
trips_gtfs = trips.drop(columns=['retention_policy'], errors='ignore')
_write(stops_filtered,  'stops.txt')
_write(routes_gtfs,     'routes.txt')
_write(trips_gtfs,      'trips.txt')
_write(stop_times,      'stop_times.txt')
_write(calendar,        'calendar.txt')
_write(cal_dates,       'calendar_dates.txt')
_write(agency,          'agency.txt')

if transfers_present:
    _write(transfers, 'transfers.txt')
if feed_info_present:
    _write(feed_info, 'feed_info.txt')
if frequencies_present:
    _write(frequencies, 'frequencies.txt')

# Write pipeline configuration sidecar so services_network_builder.py can
# auto-detect the GTFS folder and catchment boundary.
_sidecar = {
    'gtfs_output_folder':    GTFS_OUTPUT_FOLDER,
    'catchment_buffer_path': paths.CATCHMENT_AREA_BOUNDARY_GPKG,
}
_sidecar_path = _out_path('filter_config.json')
with open(_sidecar_path, 'w', encoding='utf-8') as _f:
    json.dump(_sidecar, _f, indent=2, ensure_ascii=False)
print(f"  Wrote filter_config.json")


# ===========================================================================
# Step 13 — validation check
# ===========================================================================

print("\n[13] Running self-consistency validation ...")

# Re-load written outputs for a clean check
stops_check      = pd.read_csv(_out_path('stops.txt'),      dtype=str)
routes_check     = pd.read_csv(_out_path('routes.txt'),     dtype=str)
trips_check      = pd.read_csv(_out_path('trips.txt'),      dtype=str)
stop_times_check = pd.read_csv(_out_path('stop_times.txt'), dtype=str)
calendar_check   = pd.read_csv(_out_path('calendar.txt'),   dtype=str)
cal_dates_check  = pd.read_csv(_out_path('calendar_dates.txt'), dtype=str)

valid_stop_ids    = set(stops_check['stop_id'])
valid_route_ids   = set(routes_check['route_id'])
valid_service_ids = set(calendar_check['service_id']) | set(cal_dates_check['service_id'])

# Check 1: stop_times → stops
bad_stop_times = stop_times_check[~stop_times_check['stop_id'].isin(valid_stop_ids)]
n1 = len(bad_stop_times)
tag1 = "PASS" if n1 == 0 else f"FAIL ({n1} violation(s))"
print(f"  stop_times → stops       : {tag1}")

# Check 2: trips → routes
bad_trips_route = trips_check[~trips_check['route_id'].isin(valid_route_ids)]
n2 = len(bad_trips_route)
tag2 = "PASS" if n2 == 0 else f"FAIL ({n2} violation(s))"
print(f"  trips → routes           : {tag2}")

# Check 3: trips → service IDs
bad_trips_service = trips_check[~trips_check['service_id'].isin(valid_service_ids)]
n3 = len(bad_trips_service)
tag3 = "PASS" if n3 == 0 else f"FAIL ({n3} violation(s))"
print(f"  trips → service IDs      : {tag3}")


# ===========================================================================
# Step 14 — filter report
# ===========================================================================

print(f"\n[14] Writing filter_report.txt ...")

# Route-type breakdown of retained routes
routes_check['route_type_int'] = pd.to_numeric(routes_check['route_type'], errors='coerce')
route_type_counts = routes_check['route_type_int'].value_counts().sort_index()

report_lines = [
    "=" * 70,
    "GTFS FILTER REPORT — services_filter_gtfs.py",
    "=" * 70,
    "",
    "Configuration",
    f"  CATCHMENT_METHOD      : {settings.CATCHMENT_METHOD}",
    f"  Catchment buffer used : {paths.CATCHMENT_AREA_BOUNDARY_GPKG}",
    f"  Admin level           : {_ca_admin_level}",
    f"  Primary selection     : {_ca_primary or '(all)'}",
    f"  Buffer applied        : {_ca_buffer_m:.0f} m",
    "",
    "Coordinate reference systems",
    f"  CRS of catchment boundary (GeoPackage)  : {CODEBASE_CRS}",
    f"  CRS used for all spatial operations     : {CODEBASE_CRS}",
    f"  (EPSG:2056 confirmed as codebase standard — see catchment_pt.py,",
    f"   data_import.py, generate_infrastructure.py, and others)",
    "",
    "RECORD COUNTS",
    f"  {'File':<20}  {'Original':>10}  {'Retained':>10}",
    f"  {'-'*20}  {'-'*10}  {'-'*10}",
    f"  {'stops.txt':<20}  {orig_counts['stops']:>10,}  {len(stops_filtered):>10,}  ({len(extra_stop_ids):,} out-of-canton)",
    f"  {'trips.txt':<20}  {orig_counts['trips']:>10,}  {len(trips):>10,}",
    f"  {'routes.txt':<20}  {orig_counts['routes']:>10,}  {len(routes_gtfs):>10,}",
    f"  {'agency.txt':<20}  {orig_counts['agencies']:>10,}  {len(agency):>10,}",
    "",
    "ROUTE-TYPE BREAKDOWN (retained routes)",
    f"  {'route_type':<12}  {'Mode':<20}  {'Class':<12}  {'Policy':<10}  {'Count':>6}",
    f"  {'-'*12}  {'-'*20}  {'-'*12}  {'-'*10}  {'-'*6}",
]
for rt, count in route_type_counts.items():
    rt_int = int(rt) if pd.notna(rt) else None
    label  = ALL_RETAINED_ROUTE_TYPES.get(rt_int, 'Unknown') if rt_int is not None else 'Unknown'
    cls    = 'rail' if rt_int in RAIL_ROUTE_TYPES else 'pt_feeder'
    pol    = _retention_policy(rt_int) if rt_int is not None else 'border-1'
    report_lines.append(f"  {rt_int:<12}  {label:<20}  {cls:<12}  {pol:<10}  {count:>6,}")

report_lines += [
    "",
    "SERVICE DURATION FILTER (Tier 1)",
    f"  Timetable period           : {TIMETABLE_START} to {TIMETABLE_END}",
    f"  Min weekday active days    : {_tier1_stats['min_threshold']}",
    f"  Service IDs evaluated      : {_tier1_stats['total_sids']:,}",
    f"  Service IDs dropped        : {_tier1_stats['dropped_sids']:,}",
    f"  Service IDs retained       : {_tier1_stats['kept_sids']:,}",
    f"  Trips before Tier 1 filter : {_tier1_stats['trips_before']:,}",
    f"  Trips after Tier 1 filter  : {_tier1_stats['trips_after']:,}",
    "",
    "OPTIONAL FILES",
]
for f in OPTIONAL_FILES:
    status = "skipped (not present)" if f in skipped_optional else "included"
    report_lines.append(f"  {f:<25}  {status}")

report_lines += [
    "",
    "SPATIAL RETENTION POLICIES",
    f"  full      (RT 106, 109) : keep all stops of trips touching canton",
    f"  border+1  (RT 102, 103) : canton stops + 1 stop padding each side",
    f"  border-1  (PT feeders)  : only stops within the canton",
    "",
    "TRIP FILTERING NOTE",
    "  Trips are retained if they have at least one stop within the canton.",
    "  Spatial clipping depends on the retention policy assigned to each",
    "  trip's route type (see above). Trips with no canton stops are dropped.",
    "",
    "CROSS-EVALUATION NOTE",
    "  OD_type = 'pt_catchment_perimeter' and perimeter_demand are retained",
    "  in settings.py for cross-evaluation. Once both the PT-Feeder method",
    "  (this filtered output) and the existing perimeter-based method have",
    "  been run, their results can be compared directly.",
    "",
    "VALIDATION",
    f"  stop_times → stops       : {tag1}",
    f"  trips → routes           : {tag2}",
    f"  trips → service IDs      : {tag3}",
    "",
    "RUNTIME",
    f"  Total elapsed            : {time.time() - _start_time:.1f} seconds",
    "",
    "=" * 70,
]

report_text = "\n".join(report_lines)
with open(_out_path('filter_report.txt'), 'w', encoding='utf-8') as f:
    f.write(report_text)

print(report_text)
print("\nDone.")
