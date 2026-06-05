# services_network_builder.py
#
# CRS used throughout this script: EPSG:2056 (LV95 — Swiss National Grid).
# Stop coordinates are read directly from the stop_E / stop_N columns written
# by services_filter_gtfs.py — no reprojection is performed here.
#
# Outputs (written to subfolders defined by the folder constants below)
# -------
#   PT-Feeder (bus / tram / funicular / ship / metro):
#     paths.FEEDER_LINES_DIR / PT_FEEDER_OUTPUT_FOLDER / PT_FEEDER_STOPS_FILE
#     paths.FEEDER_LINES_DIR / PT_FEEDER_OUTPUT_FOLDER / PT_FEEDER_LINES_FILE
#
#   Rail (S-Bahn / regional / inter-regional / long-distance):
#     paths.RAIL_LINES_DIR / RAIL_OUTPUT_FOLDER / RAIL_STOPS_FILE
#     paths.RAIL_LINES_DIR / RAIL_OUTPUT_FOLDER / RAIL_LINES_FILE
#
# Each lines GeoPackage contains one layer per route_type, each with one
# feature per (route_id, direction_id, variant_rank) using structural
# stop-sequence variants classified by trip share, hour spread, and headway CV.
# Each stops GeoPackage contains one layer per dominant mode type.
# Circular lines (first stop == last stop) are stored as a single feature.

import os
import statistics
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import substring, linemerge
import numpy as np
from scipy.spatial import cKDTree

import paths
import settings

_start_time = time.time()

# ---------------------------------------------------------------------------
# Folder / file name constants — adjust here to target a different GTFS source
# ---------------------------------------------------------------------------

GTFS_INPUT_FOLDER      = 'GTFS_SVC2026_ZH'          # subfolder of paths.GTFS_TRANSIT_DIR

PT_FEEDER_OUTPUT_FOLDER = 'SVC2026_ZH_network'       # subfolder of paths.FEEDER_LINES_DIR
PT_FEEDER_STOPS_FILE    = 'pt_feeder_stops.gpkg'
PT_FEEDER_LINES_FILE    = 'pt_feeder_lines.gpkg'

RAIL_OUTPUT_FOLDER      = 'SVC2026_ZH_network'       # subfolder of paths.RAIL_LINES_DIR
RAIL_STOPS_FILE         = 'rail_stops.gpkg'
RAIL_LINES_FILE         = 'rail_lines.gpkg'

PT_FEEDER_SEGMENTS_FILE = 'pt_feeder_segments.gpkg'
RAIL_SEGMENTS_FILE      = 'rail_segments.gpkg'

PT_FEEDER_PROJECT_FILE          = 'pt_feeder_lines.qgz'
RAIL_PROJECT_FILE               = 'rail_lines.qgz'
PT_FEEDER_SEGMENTS_PROJECT_FILE = 'pt_feeder_segments.qgz'
RAIL_SEGMENTS_PROJECT_FILE      = 'rail_segments.qgz'

# All-day only filtered output files
PT_FEEDER_LINES_ALLDAY_FILE       = 'pt_feeder_lines_allday.gpkg'
PT_FEEDER_SEGMENTS_ALLDAY_FILE    = 'pt_feeder_segments_allday.gpkg'
PT_FEEDER_STOPS_ALLDAY_FILE       = 'pt_feeder_stops_allday.gpkg'
RAIL_LINES_ALLDAY_FILE            = 'rail_lines_allday.gpkg'
RAIL_SEGMENTS_ALLDAY_FILE         = 'rail_segments_allday.gpkg'
RAIL_STOPS_ALLDAY_FILE            = 'rail_stops_allday.gpkg'

# Peak / off-peak filtered output files
PT_FEEDER_LINES_PEAK_FILE       = 'pt_feeder_lines_peak.gpkg'
PT_FEEDER_LINES_OFFPEAK_FILE    = 'pt_feeder_lines_offpeak.gpkg'
PT_FEEDER_SEGMENTS_PEAK_FILE    = 'pt_feeder_segments_peak.gpkg'
PT_FEEDER_SEGMENTS_OFFPEAK_FILE = 'pt_feeder_segments_offpeak.gpkg'
PT_FEEDER_STOPS_PEAK_FILE       = 'pt_feeder_stops_peak.gpkg'
PT_FEEDER_STOPS_OFFPEAK_FILE    = 'pt_feeder_stops_offpeak.gpkg'
RAIL_LINES_PEAK_FILE            = 'rail_lines_peak.gpkg'
RAIL_LINES_OFFPEAK_FILE         = 'rail_lines_offpeak.gpkg'
RAIL_SEGMENTS_PEAK_FILE         = 'rail_segments_peak.gpkg'
RAIL_SEGMENTS_OFFPEAK_FILE      = 'rail_segments_offpeak.gpkg'
RAIL_STOPS_PEAK_FILE            = 'rail_stops_peak.gpkg'
RAIL_STOPS_OFFPEAK_FILE         = 'rail_stops_offpeak.gpkg'

PT_FEEDER_LINES_ALLDAY_PROJECT_FILE       = 'pt_feeder_lines_allday.qgz'
PT_FEEDER_SEGMENTS_ALLDAY_PROJECT_FILE    = 'pt_feeder_segments_allday.qgz'
RAIL_LINES_ALLDAY_PROJECT_FILE            = 'rail_lines_allday.qgz'
RAIL_SEGMENTS_ALLDAY_PROJECT_FILE         = 'rail_segments_allday.qgz'

PT_FEEDER_LINES_PEAK_PROJECT_FILE       = 'pt_feeder_lines_peak.qgz'
PT_FEEDER_LINES_OFFPEAK_PROJECT_FILE    = 'pt_feeder_lines_offpeak.qgz'
PT_FEEDER_SEGMENTS_PEAK_PROJECT_FILE    = 'pt_feeder_segments_peak.qgz'
PT_FEEDER_SEGMENTS_OFFPEAK_PROJECT_FILE = 'pt_feeder_segments_offpeak.qgz'
RAIL_LINES_PEAK_PROJECT_FILE            = 'rail_lines_peak.qgz'
RAIL_LINES_OFFPEAK_PROJECT_FILE         = 'rail_lines_offpeak.qgz'
RAIL_SEGMENTS_PEAK_PROJECT_FILE         = 'rail_segments_peak.qgz'
RAIL_SEGMENTS_OFFPEAK_PROJECT_FILE      = 'rail_segments_offpeak.qgz'

BUILD_REPORT_FILE       = 'build_report.txt'

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CODEBASE_CRS = 'EPSG:2056'

# Frequency windows (used as validation / fallback for adaptive peak detection)
AM_PEAK_START  = '06:00:00'
AM_PEAK_END    = '09:00:00'
PM_PEAK_START  = '16:00:00'
PM_PEAK_END    = '19:00:00'

# Off-peak windows — midday and evening (Güteklasse alignment: operational day 06:00–20:00)
# Used as a list so frequency computation can iterate over multiple windows
OFFPEAK_WINDOWS = [
    ('09:00:00', '16:00:00'),  # midday off-peak
    ('19:00:00', '20:00:00'),  # evening off-peak
]
# Legacy single-window constants (kept for backward compatibility with adaptive peak detection)
OFFPEAK_START  = '09:00:00'
OFFPEAK_END    = '16:00:00'

# Operational day boundaries (for trip filtering)
OPERATIONAL_DAY_START = '06:00:00'  # = AM_PEAK_START
OPERATIONAL_DAY_END   = '20:00:00'  # extended for Güteklasse alignment

# Adaptive peak detection — per (route_id, direction_id) window detection
HALFDAY_MIDPOINT              = '12:00:00'  # splits operating day into AM / PM halves
PEAK_BIN_WIDTH_MIN            = 30          # bin width in minutes for departure histogram
PEAK_MIN_OVERLAP              = 0.50        # min overlap fraction with validation window
PEAK_MIN_CLUSTER_BINS         = 2           # cluster must span ≥ this many bins (avoids single-bin artifacts)
PEAK_CLASSIFICATION_THRESHOLD = 0.80        # fraction of deps to classify peak_only / offpeak_only

# Line-level acceptance gate — applied per route_id before variant classification.
# Each direction individually must have ≥ LINE_MIN_DEPARTURES_PER_DIR departures
# in at least one time window (AM peak, off-peak, PM peak), AND each direction's
# departures must span ≥ LINE_MIN_HOUR_SPREAD distinct hours.
LINE_MIN_DEPARTURES_PER_DIR = 2     # ≈ min deps per direction in any window
LINE_MIN_HOUR_SPREAD        = 2     # min distinct hours per direction

# Variant classification thresholds
VARIANT_MIN_TRIP_SHARE           = 0.10  # min fraction of direction's trips
VARIANT_MIN_HOUR_SPREAD          = 2     # min distinct hours variant appears in
VARIANT_HEADWAY_CV_MAX           = 0.42  # max coefficient of variation of inter-departure gaps
VARIANT_MIN_DEPARTURES_IN_WINDOW = 3     # min departures in a window to compute frequency
VARIANT_FREQ_MIN_DEPARTURES      = 2     # min departures in a window for frequency computation

# Short-working suppression: a passing variant is suppressed if its stop sequence
# is a contiguous prefix of a longer passing variant AND at least this fraction of
# its first-stop departures (within 1 minute) overlap with the longer variant.
SHORTWORKING_OVERLAP_MIN = 0.80

# Standard headway snapping: dep/hr values are snapped to the nearest value in
# this list. The list corresponds to headways of 1,2,3,4,5,6,7.5,10,12,15,20,30,60,120,240 min.
STANDARD_FREQUENCIES = [60, 30, 20, 15, 12, 10, 8, 6, 5, 4, 3, 2, 1, 0.5, 0.25]

# Minimum acceptable frequency (dep/hr). Variants whose highest non-NULL
# window frequency is below this threshold are dropped from the network.
MIN_FREQUENCY_DEP_HR = 1

# Minimum number of weekdays (Mon–Fri) a service_id must run on to be
# considered a "typical weekday" service for frequency counting.
# ≥3/5 keeps Mon–Thu (4/5) and full-week (5/5) services while excluding
# MO-only supplements (1/5) and other partial-week exception IDs.
WEEKDAY_MIN_DAYS = 3

# Maximum gap in minutes between two departures on the same route+direction
# that are treated as a single "service slot" (two-terminal branch lines).
TERMINAL_MERGE_MINUTES = 2

# Tier 2 service duration filter: a service_id must be active on at least this
# fraction of available weekdays in the timetable year. This excludes
# construction-period replacements and seasonal-only services that do not
# represent the standard network.
MIN_WEEKDAY_ACTIVE_FRACTION = 0.50

# Mode selection — controls which mode groups are built.
#   'all'         : build both PT-Feeder and Rail (production default)
#   'pt_feeder'   : only PT-Feeder modes (tram, bus, metro, ship, funicular)
#   'rail'        : only Rail modes (S-Bahn, regional, inter-regional, long-distance)
#   list of ints  : only specific route_types, e.g. [900] for tram, [109, 106] for S-Bahn + regional
# NOTE: default value — overridden by _configure_pipeline() at runtime.
BUILD_MODES = 'all'


PT_FEEDER_LINE_TYPES = {
    900:  'Tram',
    401:  'Metro',
    700:  'Bus',
    702:  'Express Bus',
    715:  'On-demand Bus',
    1000: 'Ship',
    1400: 'Funicular',
}

RAIL_LINE_TYPES = {
    102: 'Long-Distance Rail',
    103: 'Inter-Regional Rail',
    106: 'Regional Rail',
    109: 'S-Bahn / Suburban Rail',
}

# ---------------------------------------------------------------------------
# Colour scheme (OpenRailwayMap-aligned) for line layers
# ---------------------------------------------------------------------------

# Line colours per route_type
LINE_COLOURS = {
    102:  '#FF0000',  # Long-Distance Rail  — red
    103:  '#FF0000',  # Inter-Regional Rail — red
    106:  '#000000',  # Regional Rail       — black
    109:  '#000000',  # S-Bahn              — black
    401:  '#00246B',  # Metro               — dark blue (ORM)
    900:  '#FF66CC',  # Tram                — pink (ORM)
    700:  '#0000FF',  # Bus                 — blue
    702:  '#0000FF',  # Express Bus         — blue
    715:  '#0000FF',  # On-demand Bus       — blue
    1000: "#0099FF",  # Ship                — blue dashed
    1400: '#000000',  # Funicular           — black dashed
}

LINE_STYLE = {
    1000: 'dashed',
    1400: 'dashed',
}

# PT-feeder stop dominant-mode hierarchy (lower index = higher priority)
PT_FEEDER_STOP_HIERARCHY = [401, 900, 700, 702, 715, 1400, 1000]

# Rail stop style — always white fill, black outline
RAIL_STOP_FILL    = '#FFFFFF'
RAIL_STOP_OUTLINE = '#000000'

# Layer name mapping: route_type → layer name used inside the GeoPackage
LAYER_NAMES = {
    102:  'long_distance_rail',
    103:  'inter_regional_rail',
    106:  'regional_rail',
    109:  'sbahn',
    401:  'metro',
    900:  'tram',
    700:  'bus',
    702:  'express_bus',
    715:  'on_demand_bus',
    1000: 'ship',
    1400: 'funicular',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fpath(filename):
    return os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_INPUT_FOLDER, filename)


def _load(filename):
    p = _fpath(filename)
    if not os.path.isfile(p):
        return None
    print(f"  Loading {filename} ...", end=' ', flush=True)
    df = pd.read_csv(p, dtype=str, low_memory=False)
    print(f"{len(df):,} rows")
    return df


def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def _build_linestring(stop_id_seq, coord_dict):
    """Build a LineString from an ordered sequence of stop_ids using LV95 coords."""
    coords = []
    for sid in stop_id_seq:
        pt = coord_dict.get(sid)
        if pt is not None:
            coords.append((pt.x, pt.y))
    if len(coords) < 2:
        return None
    return LineString(coords)
def _detect_dominant_period(route_trip_ids, sid_active_sets):
    """Identify the dominant calendar period for a route and return its trip_ids.

    Swiss GTFS encodes seasonal / construction timetable changes as separate
    service_ids with non-overlapping active-weekday date sets.  Pooling all
    trips across the year creates phantom variant interleaving (S7's CV issue)
    and spurious construction-only variants (S16's skipped-stop variant).

    Algorithm:
      1. Collect (service_id → active_weekday_set) for all trips in the route.
      2. Cluster service_ids into calendar periods: two service_ids belong to
         the same period if their active date sets overlap (union-find).
      3. For each period, compute the union of active weekdays.
      4. The dominant period is the one covering the most weekdays.
      5. Return the set of trip_ids whose service_id belongs to the dominant
         period.

    If all service_ids share dates (single period), returns all trip_ids unchanged.
    """
    # Map trip_ids → service_ids
    trip_sids = trips_all.loc[
        trips_all['trip_id'].isin(route_trip_ids),
        ['trip_id', 'service_id']
    ]
    if trip_sids.empty:
        return route_trip_ids

    # Unique service_ids with their active date sets
    sids = trip_sids['service_id'].unique()
    sid_dates = {s: sid_active_sets.get(s, set()) for s in sids}

    # Union-find clustering on overlapping date sets
    parent = {s: s for s in sids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    sid_list = list(sids)
    for i in range(len(sid_list)):
        for j in range(i + 1, len(sid_list)):
            if sid_dates[sid_list[i]] & sid_dates[sid_list[j]]:
                union(sid_list[i], sid_list[j])

    # Group service_ids by cluster
    clusters = defaultdict(set)
    for s in sids:
        clusters[find(s)].add(s)

    if len(clusters) <= 1:
        return route_trip_ids  # single period — use everything

    # Find dominant cluster (most union-weekdays)
    best_root, best_days = None, 0
    for root, members in clusters.items():
        union_days = set()
        for s in members:
            union_days |= sid_dates[s]
        if len(union_days) > best_days:
            best_root, best_days = root, len(union_days)

    dominant_sids = clusters[best_root]
    dominant_trips = set(
        trip_sids.loc[trip_sids['service_id'].isin(dominant_sids), 'trip_id']
    )
    return dominant_trips


def _is_circular(stop_id_seq, single_direction=False):
    """Detect circular (loop) routes.

    Primary check: first stop equals last stop.
    Secondary check (only when *single_direction* is True): the last stop
    equals one of the first two intermediate stops.  This catches GTFS loop
    routes whose terminus is encoded as a nearby platform/quay with a
    different parent-station ID (e.g. "Bahnhof" → "Bhf West").
    """
    if len(stop_id_seq) < 3:
        return False
    if stop_id_seq[0] == stop_id_seq[-1]:
        return True
    # Secondary: last stop appears near the start of the sequence
    if single_direction and len(stop_id_seq) >= 4:
        early_stops = set(stop_id_seq[1:3])  # indices 1 and 2
        if stop_id_seq[-1] in early_stops:
            return True
    return False


def _to_minutes_str(t):
    """Convert 'HH:MM:SS' string to fractional minutes past midnight."""
    try:
        h, m, s = str(t).split(':')
        return int(h) * 60 + int(m) + int(s) / 60
    except Exception:
        return None


def _minutes_to_time_str(m):
    """Convert fractional minutes past midnight to 'HH:MM:SS' string."""
    h = int(m // 60)
    mins = int(m % 60)
    secs = int((m % 1) * 60)
    return f"{h:02d}:{mins:02d}:{secs:02d}"


def _get_variant_dep_minutes(trip_ids, time_col):
    """Return a set of weekday first-stop departure times (fractional minutes)
    for the given trip_ids across the full operational day.

    Uses the same dual-path logic (schedule + frequency-based trips) and
    deduplication as _get_window_departures, but spans the whole day so
    the mutual-exclusivity check in _detect_peak_windows sees every departure.
    """
    weekday_trips = set(
        trips_all.loc[
            trips_all['trip_id'].isin(trip_ids) &
            trips_all['service_id'].isin(_weekday_service_ids),
            'trip_id'
        ]
    )
    if not weekday_trips:
        return set()

    sched_trips = weekday_trips - _freq_trip_ids
    freq_trips  = weekday_trips & _freq_trip_ids

    parts = []
    if sched_trips:
        _valid = list(sched_trips & _scheduled_trip_ids)
        if _valid:
            parts.append(_first_stop_per_trip.loc[_valid].reset_index())
    if freq_trips:
        freq_deps = _expand_freq_departures(
            freq_trips, time_col, OPERATIONAL_DAY_START, OPERATIONAL_DAY_END)
        if not freq_deps.empty:
            parts.append(freq_deps)

    if not parts:
        return set()

    combined = pd.concat(parts, ignore_index=True)
    combined = _dedup_first_stops(combined, time_col)

    result = set()
    for t in combined[time_col].dropna():
        m = _to_minutes_str(t)
        if m is not None:
            result.add(m)
    return result


def _detect_peak_windows(route_trip_ids, variants=None):
    """Detect adaptive AM and PM peak windows for a (route_id, direction_id).

    Two detection paths are tried in order:

    1. Mutual-exclusivity path (when *variants* is provided and has ≥ 2 entries):
       Checks whether the accepted variants never share the same operational
       hour.  If so, they replace each other across the day (e.g. S3: the
       Bülach peak service runs instead of the Hardbrücke off-peak service).
       The variant with the longest stop sequence is identified as the peak
       service; its first AM/PM departures and the first subsequent off-peak
       departure define the window boundaries directly from the timetable,
       giving direction-specific adaptive windows without density clustering.

    2. Density-clustering path (pooled departures, existing logic):
       Pools all weekday departures, bins into PEAK_BIN_WIDTH_MIN slots, finds
       the densest contiguous cluster above the half-day median, then validates
       against the fixed AM/PM windows.  Falls back to the longest stop-sequence
       variant when the pooled density is flat.

    Returns dict with keys:
        am_start, am_end, pm_start, pm_end, op_start, op_end (str 'HH:MM:SS')
        am_adaptive, pm_adaptive (bool — True if adaptively detected)
    """
    fallback = {
        'am_start': AM_PEAK_START, 'am_end': AM_PEAK_END,
        'pm_start': PM_PEAK_START, 'pm_end': PM_PEAK_END,
        'op_start': OFFPEAK_START, 'op_end': OFFPEAK_END,
        'am_adaptive': False, 'pm_adaptive': False,
    }

    # Get weekday trips in this direction
    time_col = 'departure_time' if 'departure_time' in stop_times.columns else 'arrival_time'

    # -------------------------------------------------------------------
    # Path 1: mutual-exclusivity detection
    # -------------------------------------------------------------------
    if variants and len(variants) >= 2:
        variant_dep_sets = [
            _get_variant_dep_minutes(tids, time_col) for _, tids in variants
        ]

        # Check: no two variants have departures in the same operational hour.
        # Hour-level granularity correctly identifies co-active services (S10:
        # both variants depart HB every hour) vs. substituting services (S3:
        # only one variant active per hour).
        exclusive = True
        for h in range(6, 20):
            h_start, h_end = h * 60, (h + 1) * 60
            n_active = sum(
                1 for deps in variant_dep_sets
                if any(h_start <= t < h_end for t in deps)
            )
            if n_active >= 2:
                exclusive = False
                break

        if exclusive:
            # Peak variant = most stops (extended service); tiebreak: fewest departures
            peak_idx = max(
                range(len(variants)),
                key=lambda i: (len(variants[i][0]), -len(variant_dep_sets[i]))
            )
            peak_deps = variant_dep_sets[peak_idx]
            offpeak_deps: set = set()
            for i, deps in enumerate(variant_dep_sets):
                if i != peak_idx:
                    offpeak_deps |= deps

            midpoint  = _to_minutes_str(HALFDAY_MIDPOINT)
            day_start = _to_minutes_str(AM_PEAK_START)
            day_end   = _to_minutes_str(PM_PEAK_END)

            peak_am = sorted(t for t in peak_deps if day_start <= t < midpoint)
            peak_pm = sorted(t for t in peak_deps if midpoint   <= t < day_end)

            # AM_END: first off-peak departure strictly after the last peak AM dep
            am_start = _to_minutes_str(AM_PEAK_START)
            if peak_am:
                later_offpeak = sorted(
                    t for t in offpeak_deps if t > peak_am[-1] and t < midpoint
                )
                am_end = later_offpeak[0] if later_offpeak else _to_minutes_str(AM_PEAK_END)
            else:
                am_end = _to_minutes_str(AM_PEAK_END)

            # PM_START: first peak departure in the PM half
            pm_end = _to_minutes_str(PM_PEAK_END)
            if peak_pm:
                pm_start = peak_pm[0]
            else:
                pm_start = _to_minutes_str(PM_PEAK_START)

            # Only use these windows if the off-peak gap has positive duration
            if am_end < pm_start:
                return {
                    'am_start':    _minutes_to_time_str(am_start),
                    'am_end':      _minutes_to_time_str(am_end),
                    'pm_start':    _minutes_to_time_str(pm_start),
                    'pm_end':      _minutes_to_time_str(pm_end),
                    'op_start':    _minutes_to_time_str(am_end),
                    'op_end':      _minutes_to_time_str(pm_start),
                    'am_adaptive': True,
                    'pm_adaptive': True,
                }
            # Guard failed — fall through to pairwise scan below

        # -------------------------------------------------------------------
        # Path 1b: pairwise mutual exclusivity
        # -------------------------------------------------------------------
        # When a co-active all-day variant (e.g. Seuzach on S11) prevents the
        # all-variant check from succeeding, scan every pair of variants for
        # one that is pairwise hour-exclusive AND whose stop sequences have a
        # prefix or suffix containment relationship (the hallmark of a
        # short-working / terminal-extension substitution).  That pair alone
        # drives window derivation; co-active variants are left to the
        # density-clustering fallback that follows.
        _mid = _to_minutes_str(HALFDAY_MIDPOINT)
        _ds  = _to_minutes_str(AM_PEAK_START)
        _de  = _to_minutes_str(PM_PEAK_END)

        for _i in range(len(variants)):
            for _j in range(_i + 1, len(variants)):
                _seq_i, _  = variants[_i]
                _seq_j, _  = variants[_j]
                _deps_i    = variant_dep_sets[_i]
                _deps_j    = variant_dep_sets[_j]

                # Pairwise hour-exclusivity
                _pair_excl = True
                for _h in range(6, 20):
                    _hs, _he = _h * 60, (_h + 1) * 60
                    if (any(_hs <= t < _he for t in _deps_i) and
                            any(_hs <= t < _he for t in _deps_j)):
                        _pair_excl = False
                        break
                if not _pair_excl:
                    continue

                # Stop-sequence containment: shorter must be a prefix or suffix
                # of longer (short-working relationship)
                _li, _lj = len(_seq_i), len(_seq_j)
                if _li < _lj:
                    _shorter, _longer  = _seq_i, _seq_j
                    _off_deps, _pk_deps = _deps_i, _deps_j
                else:
                    _shorter, _longer  = _seq_j, _seq_i
                    _off_deps, _pk_deps = _deps_j, _deps_i
                _n = len(_shorter)
                _is_prefix = _longer[:_n] == _shorter
                _is_suffix = _longer[len(_longer) - _n:] == _shorter
                if not (_is_prefix or _is_suffix):
                    continue

                # Derive windows from this exclusive pair (same logic as Path 1)
                _pk_am = sorted(t for t in _pk_deps if _ds  <= t < _mid)
                _pk_pm = sorted(t for t in _pk_deps if _mid <= t < _de)

                _am_start_pw = _to_minutes_str(AM_PEAK_START)
                if _pk_am:
                    _later_pw  = sorted(t for t in _off_deps if t > _pk_am[-1] and t < _mid)
                    _am_end_pw = _later_pw[0] if _later_pw else _to_minutes_str(AM_PEAK_END)
                else:
                    _am_end_pw = _to_minutes_str(AM_PEAK_END)

                _pm_end_pw   = _to_minutes_str(PM_PEAK_END)
                _pm_start_pw = _pk_pm[0] if _pk_pm else _to_minutes_str(PM_PEAK_START)

                if _am_end_pw >= _pm_start_pw:
                    continue  # degenerate window — try next pair

                return {
                    'am_start':    _minutes_to_time_str(_am_start_pw),
                    'am_end':      _minutes_to_time_str(_am_end_pw),
                    'pm_start':    _minutes_to_time_str(_pm_start_pw),
                    'pm_end':      _minutes_to_time_str(_pm_end_pw),
                    'op_start':    _minutes_to_time_str(_am_end_pw),
                    'op_end':      _minutes_to_time_str(_pm_start_pw),
                    'am_adaptive': True,
                    'pm_adaptive': True,
                }
        # No exclusive pair found — fall through to density clustering

    weekday_trips = set(
        trips_all.loc[
            trips_all['trip_id'].isin(route_trip_ids) &
            trips_all['service_id'].isin(_weekday_service_ids),
            'trip_id'
        ]
    )
    if not weekday_trips:
        return fallback

    # Dual-path: schedule-based trips from stop_times, frequency-based from headway bands
    sched_trips = weekday_trips - _freq_trip_ids
    freq_trips  = weekday_trips & _freq_trip_ids

    parts = []
    if sched_trips:
        _valid = list(sched_trips & _scheduled_trip_ids)
        if _valid:
            parts.append(_first_stop_per_trip.loc[_valid].reset_index())
    if freq_trips:
        freq_deps = _expand_freq_departures(freq_trips, time_col, AM_PEAK_START, PM_PEAK_END)
        if not freq_deps.empty:
            parts.append(freq_deps)

    if not parts:
        return fallback

    # Deduplicate first-stop departures (same logic as _get_window_departures)
    # so that peak detection and frequency computation see the same counts.
    _first_all = pd.concat(parts, ignore_index=True)
    _first_all = _dedup_first_stops(_first_all, time_col)

    # Keep sub for the fallback longest-variant detection (schedule-based only)
    sub = stop_times[stop_times['trip_id'].isin(weekday_trips)].copy()

    midpoint = _to_minutes_str(HALFDAY_MIDPOINT)
    day_start = _to_minutes_str(AM_PEAK_START)   # 360 (06:00)
    day_end = _to_minutes_str(PM_PEAK_END)       # 1140 (19:00)

    def _first_stop_minutes(first_stops_df):
        """Get departure minutes from a pre-deduped first-stops DataFrame."""
        mins = [_to_minutes_str(t) for t in first_stops_df[time_col].dropna()]
        return [m for m in mins if m is not None]

    def _find_peak_cluster(deps, half_start, half_end):
        """Find the densest contiguous cluster of bins above the half-day median."""
        if len(deps) < VARIANT_MIN_DEPARTURES_IN_WINDOW:
            return None, None

        bin_starts = []
        b = half_start
        while b < half_end:
            bin_starts.append(b)
            b += PEAK_BIN_WIDTH_MIN

        if not bin_starts:
            return None, None

        bin_counts = [sum(1 for d in deps if bs <= d < bs + PEAK_BIN_WIDTH_MIN)
                      for bs in bin_starts]

        median_count = sorted(bin_counts)[len(bin_counts) // 2]

        # Mark bins as dense (strictly above median)
        dense = [c > median_count for c in bin_counts]

        if not any(dense):
            return None, None  # flat density — no peak detectable

        # Find contiguous runs of dense bins
        runs = []
        i = 0
        while i < len(dense):
            if dense[i]:
                j = i
                total = 0
                while j < len(dense) and dense[j]:
                    total += bin_counts[j]
                    j += 1
                runs.append((i, j - i, total))
                i = j
            else:
                i += 1

        if not runs:
            return None, None

        best = max(runs, key=lambda r: (r[1], r[2]))
        if best[1] < PEAK_MIN_CLUSTER_BINS:
            return None, None  # cluster too narrow — likely a scheduling artifact
        cluster_start = bin_starts[best[0]]
        cluster_end = bin_starts[best[0] + best[1] - 1] + PEAK_BIN_WIDTH_MIN
        return cluster_start, cluster_end

    # --- Primary: pooled density detection ---
    dep_minutes = _first_stop_minutes(_first_all)
    if len(dep_minutes) < VARIANT_MIN_DEPARTURES_IN_WINDOW:
        return fallback

    am_deps = [m for m in dep_minutes if day_start <= m < midpoint]
    pm_deps = [m for m in dep_minutes if midpoint <= m < day_end]

    am_cluster_start, am_cluster_end = _find_peak_cluster(am_deps, day_start, midpoint)
    pm_cluster_start, pm_cluster_end = _find_peak_cluster(pm_deps, midpoint, day_end)

    # --- Fallback: longest-variant detection ---
    # When pooled density is flat (interleaving variants at same headway),
    # detect peaks from the longest stop-sequence variant instead.
    if am_cluster_start is None or pm_cluster_start is None:
        # Build stop sequences per trip to find the longest variant
        seq_per_trip = (
            sub.sort_values('stop_sequence_int')
            .groupby('trip_id')['stop_id']
            .apply(tuple)
        )
        seq_to_trips = defaultdict(set)
        for tid, seq in seq_per_trip.items():
            seq_to_trips[seq].add(tid)

        if len(seq_to_trips) >= 2:
            # Find the longest variant by stop count
            longest_seq = max(seq_to_trips.keys(), key=len)
            longest_trips = seq_to_trips[longest_seq]

            if len(longest_trips) >= VARIANT_MIN_DEPARTURES_IN_WINDOW:
                longest_first = _first_all[_first_all['trip_id'].isin(longest_trips)]
                longest_deps = _first_stop_minutes(longest_first)

                longest_am = [m for m in longest_deps if day_start <= m < midpoint]
                longest_pm = [m for m in longest_deps if midpoint <= m < day_end]

                if am_cluster_start is None:
                    am_cluster_start, am_cluster_end = _find_peak_cluster(
                        longest_am, day_start, midpoint)
                if pm_cluster_start is None:
                    pm_cluster_start, pm_cluster_end = _find_peak_cluster(
                        longest_pm, midpoint, day_end)

    # --- Validate against fixed windows ---
    am_val_start = _to_minutes_str(AM_PEAK_START)
    am_val_end = _to_minutes_str(AM_PEAK_END)
    am_adaptive = False
    if am_cluster_start is not None:
        overlap = max(0, min(am_cluster_end, am_val_end) - max(am_cluster_start, am_val_start))
        duration = am_cluster_end - am_cluster_start
        if duration > 0 and overlap / duration >= PEAK_MIN_OVERLAP:
            am_start = am_cluster_start
            am_end = am_cluster_end
            am_adaptive = True
        else:
            am_start = am_val_start
            am_end = am_val_end
    else:
        am_start = am_val_start
        am_end = am_val_end

    pm_val_start = _to_minutes_str(PM_PEAK_START)
    pm_val_end = _to_minutes_str(PM_PEAK_END)
    pm_adaptive = False
    if pm_cluster_start is not None:
        overlap = max(0, min(pm_cluster_end, pm_val_end) - max(pm_cluster_start, pm_val_start))
        duration = pm_cluster_end - pm_cluster_start
        if duration > 0 and overlap / duration >= PEAK_MIN_OVERLAP:
            pm_start = pm_cluster_start
            pm_end = pm_cluster_end
            pm_adaptive = True
        else:
            pm_start = pm_val_start
            pm_end = pm_val_end
    else:
        pm_start = pm_val_start
        pm_end = pm_val_end

    # Guard: off-peak must have positive duration
    if am_end >= pm_start:
        return fallback

    return {
        'am_start': _minutes_to_time_str(am_start),
        'am_end':   _minutes_to_time_str(am_end),
        'pm_start': _minutes_to_time_str(pm_start),
        'pm_end':   _minutes_to_time_str(pm_end),
        'op_start': _minutes_to_time_str(am_end),
        'op_end':   _minutes_to_time_str(pm_start),
        'am_adaptive': am_adaptive,
        'pm_adaptive': pm_adaptive,
    }


def _dedup_first_stops(first_stops, time_col):
    """Remove overcounted departures from a first-stops DataFrame.

    Two passes, applied in order:

    1. Deduplicate on (departure_time, stop_id) — collapses trips from
       MO-only supplement IDs and seasonal-split FULL-WEEK IDs that encode the
       same physical departure multiple times.  stop_ids are already at
       parent-station level (platform suffixes resolved earlier).

    2. Time-window dedup within the same (route_id, direction_id): any two
       remaining rows that depart within TERMINAL_MERGE_MINUTES of each other
       are collapsed into one.  This handles two-terminal branch lines where
       direction 0 departs from terminus A at :03 and terminus B at :04 —
       both fill the same service slot and a passenger at any intermediate stop
       sees only one tram.  The merge is scoped to the same route+direction so
       two different lines that happen to share a nearby departure time are
       never collapsed.
    """
    if first_stops.empty:
        return first_stops

    df = first_stops.copy()

    # Pass 1 — dedup on (time, stop_id) — already parent-station level
    df = df.drop_duplicates(subset=[time_col, 'stop_id'])

    # Pass 2 — time-window dedup within (route_id, direction_id)
    if 'route_id' not in df.columns or 'direction_id' not in df.columns:
        return df

    def _merge_close(group):
        group = group.sort_values(time_col).reset_index(drop=True)
        keep = [True] * len(group)
        for i in range(1, len(group)):
            try:
                if _to_minutes_str(group.loc[i, time_col]) - _to_minutes_str(group.loc[i - 1, time_col]) <= TERMINAL_MERGE_MINUTES:
                    keep[i] = False
            except Exception:
                pass
        return group[keep]

    df = (
        df.groupby(['route_id', 'direction_id'], group_keys=False)
          .apply(_merge_close)
          .reset_index(drop=True)
    )

    return df


def _expand_freq_departures(trip_ids, time_col, window_start, window_end):
    """Generate synthetic first-stop departures for frequency-based trips.

    For each trip_id that has entries in frequencies.txt, expands the headway
    bands into individual departure times.  Only departures within the
    half-open window [window_start, window_end) are returned.

    Returns a DataFrame with the same schema as first_stops (columns include
    time_col, 'stop_id', 'route_id', 'direction_id') or an empty DataFrame.
    """
    freq_trips = trip_ids & _freq_trip_ids
    if not freq_trips:
        return pd.DataFrame()

    w_start = _to_minutes_str(window_start) if isinstance(window_start, str) else window_start
    w_end   = _to_minutes_str(window_end)   if isinstance(window_end,   str) else window_end
    if w_start is None or w_end is None:
        return pd.DataFrame()

    # Get the first stop_id for each frequency-based trip (template stop)
    sub = stop_times[stop_times['trip_id'].isin(freq_trips)].copy()
    if sub.empty:
        return pd.DataFrame()
    first_stops = sub.loc[sub.groupby('trip_id')['stop_sequence_int'].idxmin()].copy()
    first_stops = first_stops.merge(
        trips_all[['trip_id', 'route_id', 'direction_id']],
        on='trip_id', how='left'
    )

    rows = []
    for _, row in first_stops.iterrows():
        tid = row['trip_id']
        bands = _freq_lookup.get(tid, [])
        for band_idx, (band_start, band_end, headway_min) in enumerate(bands):
            if headway_min <= 0:
                continue
            dep_idx = 0
            t = band_start
            while t < band_end:
                if w_start <= t < w_end:
                    rows.append({
                        'trip_id':      f"{tid}__freq_{band_idx}_{dep_idx}",
                        time_col:       _minutes_to_time_str(t),
                        'stop_id':      row['stop_id'],
                        'route_id':     row.get('route_id', ''),
                        'direction_id': row.get('direction_id', ''),
                    })
                dep_idx += 1
                t += headway_min
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _get_window_departures(trip_ids, time_col, window_start, window_end):
    """Get deduplicated first-stop departures within a half-open time window [window_start, window_end)
    for a set of trip_ids. Only includes weekday trips. Returns a DataFrame.
    Returns an empty DataFrame if no weekday trips exist (no weekend fallback).

    Dual-path: schedule-based trips are handled via stop_times as before;
    frequency-based trips (those in _freq_trip_ids) are expanded from
    frequencies.txt headway bands via _expand_freq_departures().
    """
    weekday_trips = set(
        trips_all.loc[
            trips_all['trip_id'].isin(trip_ids) &
            trips_all['service_id'].isin(_weekday_service_ids),
            'trip_id'
        ]
    )
    if not weekday_trips:
        return pd.DataFrame()

    # Split into schedule-based and frequency-based trips
    schedule_trips = weekday_trips - _freq_trip_ids
    freq_trips     = weekday_trips & _freq_trip_ids

    parts = []

    # Path A: schedule-based trips — existing logic
    if schedule_trips:
        _valid = list(schedule_trips & _scheduled_trip_ids)
        if _valid:
            first_stops = _first_stop_per_trip.loc[_valid].reset_index()
            in_window = first_stops[
                (first_stops[time_col] >= window_start) & (first_stops[time_col] < window_end)
            ].copy()
            if not in_window.empty:
                parts.append(in_window)

    # Path B: frequency-based trips — expand from headway bands
    if freq_trips:
        freq_deps = _expand_freq_departures(freq_trips, time_col, window_start, window_end)
        if not freq_deps.empty:
            parts.append(freq_deps)

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    combined = _dedup_first_stops(combined, time_col)
    return combined


def _snap_to_standard_freq(raw_freq):
    """Snap a raw dep/hr value to the nearest entry in STANDARD_FREQUENCIES."""
    if raw_freq is None:
        return None
    return min(STANDARD_FREQUENCIES, key=lambda f: abs(f - raw_freq))


def _median_freq(departures_df, time_col, min_departures=None, use_mean=False):
    """Compute dep/hr from interior inter-departure gaps, snapped to the nearest
    standard frequency. Returns None if fewer than *min_departures* departures
    (defaults to VARIANT_MIN_DEPARTURES_IN_WINDOW).

    use_mean=True  : use the arithmetic mean of interior gaps (appropriate for
                     single-window queries — AM peak, PM peak — where two
                     interlaced sub-series produce alternating short/long gaps
                     and the median would be biased toward whichever gap type
                     comes first in the window).
    use_mean=False : use the lower-median (default, backward-compatible),
                     which is robust to the large inter-window gap that appears
                     when the combined off-peak pool (midday + evening) is passed.
    """
    if min_departures is None:
        min_departures = VARIANT_MIN_DEPARTURES_IN_WINDOW
    if len(departures_df) < min_departures:
        return None

    times = departures_df[time_col].apply(_to_minutes_str).dropna().sort_values().tolist()
    if len(times) < min_departures:
        return None

    gaps = [times[i+1] - times[i] for i in range(len(times)-1)]
    # Drop boundary gaps (first and last) if we have enough interior gaps
    if len(gaps) >= 3:
        gaps = gaps[1:-1]
    if not gaps:
        return None

    if use_mean:
        representative_gap = sum(gaps) / len(gaps)
    else:
        representative_gap = sorted(gaps)[len(gaps) // 2]

    if representative_gap <= 0:
        return None
    return _snap_to_standard_freq(60 / representative_gap)


def _compute_frequencies(trip_ids, windows=None, min_departures=None):
    """Compute am_peak, pm_peak, offpeak frequencies for a set of trip_ids.

    Parameters
    ----------
    trip_ids : set
        Trip IDs (single direction).
    windows : dict or None
        Detected peak windows from _detect_peak_windows.
        If None, uses the fixed constants.
    min_departures : int or None
        Minimum departures required per window to compute frequency.
        Defaults to VARIANT_MIN_DEPARTURES_IN_WINDOW.

    Returns dict with keys: freq_am_peak_dep_hr, freq_pm_peak_dep_hr, freq_offpeak_dep_hr.

    Off-peak frequency uses the midday window only (op_start–op_end, typically 09:00–16:00
    or an adaptive equivalent).  The evening off-peak window (19:00–20:00) is excluded from
    frequency computation because its short duration (1 h) and the large gap separating it
    from the midday pool would bias the mean upward.  Evening departures still count toward
    total_dep, which is computed separately in build_outputs using the fixed OFFPEAK_WINDOWS.
    All three windows use the arithmetic mean of interior gaps (use_mean=True) for consistency.
    """
    time_col = 'departure_time' if 'departure_time' in stop_times.columns else 'arrival_time'

    if windows is not None:
        am_s, am_e = windows['am_start'], windows['am_end']
        pm_s, pm_e = windows['pm_start'], windows['pm_end']
        op_s, op_e = windows['op_start'], windows['op_end']
    else:
        am_s, am_e = AM_PEAK_START, AM_PEAK_END
        pm_s, pm_e = PM_PEAK_START, PM_PEAK_END
        op_s, op_e = OFFPEAK_START, OFFPEAK_END  # midday only: 09:00–16:00

    am_deps = _get_window_departures(trip_ids, time_col, am_s, am_e)
    pm_deps = _get_window_departures(trip_ids, time_col, pm_s, pm_e)
    op_deps = _get_window_departures(trip_ids, time_col, op_s, op_e)

    return {
        'freq_am_peak_dep_hr':  _median_freq(am_deps, time_col, min_departures=min_departures, use_mean=True),
        'freq_pm_peak_dep_hr':  _median_freq(pm_deps, time_col, min_departures=min_departures, use_mean=True),
        'freq_offpeak_dep_hr':  _median_freq(op_deps, time_col, min_departures=min_departures, use_mean=True),
    }



def _hex_to_rgba(hex_colour, alpha=255):
    """Convert '#RRGGBB' to QGIS RGBA string 'R,G,B,A'."""
    h = hex_colour.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b},{alpha}"


# --- QGIS project (.qgz) generation ---

_QGIS_VERSION = "3.44.9-Solothurn"

_SRS_BLOCK = """<spatialrefsys nativeFormat="wkt">
      <proj4>+proj=somerc +lat_0=46.9524055555556 +lon_0=7.43958333333333 +k_0=1 +x_0=2600000 +y_0=1200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs</proj4>
      <srsid>47</srsid>
      <srid>2056</srid>
      <authid>EPSG:2056</authid>
      <description>CH1903+ / LV95</description>
      <projectionacronym>somerc</projectionacronym>
      <ellipsoidacronym>bessel</ellipsoidacronym>
    </spatialrefsys>"""


def _line_maplayer(layer_id, gpkg_relpath, layer_name, display_name, rgba, line_style, width='0.5'):
    """Return a <maplayer> XML block for a line layer."""
    pen = 'dash' if line_style == 'dashed' else 'solid'
    return f"""  <maplayer geometry="Line" type="vector" hasScaleBasedVisibilityFlag="0">
    <id>{layer_id}</id>
    <datasource>{gpkg_relpath}|layername={layer_name}</datasource>
    <layername>{display_name}</layername>
    <provider encoding="UTF-8">ogr</provider>
    <srs>{_SRS_BLOCK}</srs>
    <renderer-v2 forceraster="0" symbollevels="0" type="singleSymbol" enableorderby="0">
      <symbols>
        <symbol alpha="1" clip_to_extent="1" type="line" name="0" force_rhr="0">
          <layer pass="0" class="SimpleLine" locked="0" enabled="1">
            <prop k="capstyle" v="square"/>
            <prop k="customdash" v="5;2"/>
            <prop k="customdash_map_unit_scale" v="3x:0,0,0,0,0,0"/>
            <prop k="customdash_unit" v="MM"/>
            <prop k="draw_inside_polygon" v="0"/>
            <prop k="joinstyle" v="bevel"/>
            <prop k="line_color" v="{rgba}"/>
            <prop k="line_style" v="{pen}"/>
            <prop k="line_width" v="{width}"/>
            <prop k="line_width_unit" v="MM"/>
            <prop k="offset" v="0"/>
            <prop k="offset_map_unit_scale" v="3x:0,0,0,0,0,0"/>
            <prop k="offset_unit" v="MM"/>
            <prop k="use_custom_dash" v="0"/>
            <prop k="width_map_unit_scale" v="3x:0,0,0,0,0,0"/>
          </layer>
        </symbol>
      </symbols>
      <rotation/>
      <sizescale/>
    </renderer-v2>
  </maplayer>"""


def _marker_maplayer(layer_id, gpkg_relpath, layer_name, display_name, fill_rgba, outline_rgba, size='2', outline_width='0.2'):
    """Return a <maplayer> XML block for a point (stops) layer."""
    return f"""  <maplayer geometry="Point" type="vector" hasScaleBasedVisibilityFlag="0">
    <id>{layer_id}</id>
    <datasource>{gpkg_relpath}|layername={layer_name}</datasource>
    <layername>{display_name}</layername>
    <provider encoding="UTF-8">ogr</provider>
    <srs>{_SRS_BLOCK}</srs>
    <renderer-v2 forceraster="0" symbollevels="0" type="singleSymbol" enableorderby="0">
      <symbols>
        <symbol alpha="1" clip_to_extent="1" type="marker" name="0" force_rhr="0">
          <layer pass="0" class="SimpleMarker" locked="0" enabled="1">
            <prop k="angle" v="0"/>
            <prop k="color" v="{fill_rgba}"/>
            <prop k="horizontal_anchor_point" v="1"/>
            <prop k="joinstyle" v="bevel"/>
            <prop k="name" v="circle"/>
            <prop k="offset" v="0,0"/>
            <prop k="offset_map_unit_scale" v="3x:0,0,0,0,0,0"/>
            <prop k="offset_unit" v="MM"/>
            <prop k="outline_color" v="{outline_rgba}"/>
            <prop k="outline_style" v="solid"/>
            <prop k="outline_width" v="{outline_width}"/>
            <prop k="outline_width_map_unit_scale" v="3x:0,0,0,0,0,0"/>
            <prop k="outline_width_unit" v="MM"/>
            <prop k="scale_method" v="diameter"/>
            <prop k="size" v="{size}"/>
            <prop k="size_map_unit_scale" v="3x:0,0,0,0,0,0"/>
            <prop k="size_unit" v="MM"/>
            <prop k="vertical_anchor_point" v="1"/>
          </layer>
        </symbol>
      </symbols>
      <rotation/>
      <sizescale/>
    </renderer-v2>
  </maplayer>"""


_WMS_LAYER_ID   = 'Swisstopo_National_Map__grey__e16b0296_87b7_4e32_b8e8_b46b5990275e'
_WMS_LAYER_NAME = 'Swisstopo National Map (grey)'
_WMS_SOURCE     = ('contextualWMSLegend=0&amp;crs=EPSG:2056&amp;dpiMode=7'
                   '&amp;featureCount=10&amp;format=image/png'
                   '&amp;layers=ch.swisstopo.pixelkarte-grau'
                   '&amp;styles=&amp;url=https://wms.geo.admin.ch/')


def _wms_maplayer():
    """Return a <maplayer> XML block for the Swisstopo WMS basemap."""
    return f"""  <maplayer type="raster" hasScaleBasedVisibilityFlag="0">
    <id>{_WMS_LAYER_ID}</id>
    <datasource>{_WMS_SOURCE}</datasource>
    <layername>{_WMS_LAYER_NAME}</layername>
    <provider encoding="">wms</provider>
    <srs>{_SRS_BLOCK}</srs>
  </maplayer>"""


def _build_qgz(qgz_path, layers):
    """Write a QGIS .qgz project file containing the given layers.

    Parameters
    ----------
    qgz_path : str
        Output path for the .qgz file.
    layers : list of dict
        Each dict has keys:
            layer_id, gpkg_relpath, layer_name, display_name, geom_type,
            colour (hex), line_style ('solid'/'dashed'), fill_colour (hex),
            outline_colour (hex)
        geom_type is 'line' or 'point'.
    """
    import zipfile

    tree_entries = []
    maplayer_blocks = []

    for lyr in layers:
        lid   = lyr['layer_id']
        src   = f"{lyr['gpkg_relpath']}|layername={lyr['layer_name']}"
        dname = lyr['display_name']

        tree_entries.append(
            f'    <layer-tree-layer id="{lid}" name="{dname}" '
            f'checked="Qt::Checked" expanded="1" source="{src}" providerKey="ogr"/>'
        )

        if lyr['geom_type'] == 'line':
            rgba = _hex_to_rgba(lyr['colour'])
            maplayer_blocks.append(
                _line_maplayer(lid, lyr['gpkg_relpath'], lyr['layer_name'],
                               dname, rgba, lyr.get('line_style', 'solid'))
            )
        else:
            fill_rgba    = _hex_to_rgba(lyr['fill_colour'])
            outline_rgba = _hex_to_rgba(lyr['outline_colour'])
            maplayer_blocks.append(
                _marker_maplayer(lid, lyr['gpkg_relpath'], lyr['layer_name'],
                                 dname, fill_rgba, outline_rgba)
            )

    # Swisstopo WMS basemap — bottom of layer tree (rendered first = background)
    tree_entries.append(
        f'    <layer-tree-layer id="{_WMS_LAYER_ID}" name="{_WMS_LAYER_NAME}" '
        f'checked="Qt::Checked" expanded="0" source="{_WMS_SOURCE}" providerKey="wms"/>'
    )
    maplayer_blocks.append(_wms_maplayer())

    tree_xml     = "\n".join(tree_entries)
    layers_xml   = "\n".join(maplayer_blocks)

    qgs = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis projectname="Network Build" version="{_QGIS_VERSION}">
  <homePath path=""/>
  <title>Network Build</title>
  <autotransaction active="0"/>
  <evaluateDefaultValues active="0"/>
  <trust active="0"/>
  <projectCrs>
    {_SRS_BLOCK}
  </projectCrs>
  <layer-tree-group>
    <customproperties/>
{tree_xml}
    <custom-order enabled="0"/>
  </layer-tree-group>
  <projectlayers>
{layers_xml}
  </projectlayers>
  <mapcanvas name="theMapCanvas">
    <units>meters</units>
    <rotation>0</rotation>
    <destinationsrs>
      {_SRS_BLOCK}
    </destinationsrs>
  </mapcanvas>
</qgis>
"""
    with zipfile.ZipFile(qgz_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('project.qgs', qgs)


# ---------------------------------------------------------------------------
# Pipeline configuration — interactive choice interface
# ---------------------------------------------------------------------------

def _configure_pipeline():
    """Prompt the user for pipeline configuration choices, or read from CLI args.

    When called with --non-interactive (e.g. from main_new.py), all prompts are
    skipped and values are taken from CLI args instead:
        --gtfs-folder  : GTFS input subfolder name
        --output-name  : output folder name (both Rail and Feeder)
        --modes        : all | pt_feeder | rail  (default: all)
        --all-periods  : write all time-period outputs (full-day + all-day + peak + off-peak)

    Returns a dict with keys:
        gtfs_input_folder  : str   – subfolder name under GTFS_TRANSIT_DIR
        output_folder      : str   – subfolder name used for both PT-Feeder and Rail output
        build_modes        : 'all' | 'pt_feeder' | 'rail'
        write_fullday      : bool
        write_allday       : bool
        write_peak         : bool
        write_offpeak      : bool
    """
    import argparse as _ap
    _parser = _ap.ArgumentParser(add_help=False)
    _parser.add_argument('--gtfs-folder',      default=None)
    _parser.add_argument('--output-name',      default=None)
    _parser.add_argument('--modes',            default=None,
                         choices=['all', 'pt_feeder', 'rail'])
    _parser.add_argument('--all-periods',      action='store_true')
    _parser.add_argument('--non-interactive',  action='store_true')
    _args, _ = _parser.parse_known_args()

    gtfs_base = os.path.join(paths.MAIN, paths.GTFS_TRANSIT_DIR)

    if _args.non_interactive:
        gtfs_input    = _args.gtfs_folder or GTFS_INPUT_FOLDER
        build_modes   = _args.modes or 'all'
        output_folder = _args.output_name or (
            gtfs_input.replace('GTFS_', '', 1) + '_network'
            if gtfs_input.startswith('GTFS_') else gtfs_input + '_network'
        )
        full_input = os.path.join(gtfs_base, gtfs_input)
        if not os.path.isdir(full_input):
            raise SystemExit(f"  ERROR: GTFS folder not found: {full_input}")

        write_fullday = write_allday = write_peak = write_offpeak = _args.all_periods
        if not _args.all_periods:
            write_fullday = True

        mode_labels = {'all': 'ALL', 'pt_feeder': 'PT-FEEDER ONLY', 'rail': 'RAIL ONLY'}
        print("=" * 70)
        print("services_network_builder.py — NON-INTERACTIVE CONFIGURATION")
        print("=" * 70)
        print(f"  GTFS input folder  : {gtfs_input}")
        print(f"  Mode selection     : {mode_labels[build_modes]}")
        print(f"  Time-period output : {'ALL' if _args.all_periods else 'FULL-DAY ONLY'}")
        print(f"  Output folder      : {output_folder}")
        print("=" * 70)

        return {
            'gtfs_input_folder': gtfs_input,
            'output_folder':     output_folder,
            'build_modes':       build_modes,
            'write_fullday':     write_fullday,
            'write_allday':      write_allday,
            'write_peak':        write_peak,
            'write_offpeak':     write_offpeak,
        }

    # --- Interactive path ----------------------------------------------------
    print("=" * 70)
    print("services_network_builder.py — PIPELINE CONFIGURATION")
    print("=" * 70)

    # --- A. GTFS Input Folder ------------------------------------------------
    print("\nA. GTFS INPUT FOLDER")
    print(f"   Base path: {paths.GTFS_TRANSIT_DIR}")
    if os.path.isdir(gtfs_base):
        subfolders = sorted([
            d for d in os.listdir(gtfs_base)
            if os.path.isdir(os.path.join(gtfs_base, d)) and '_raw' not in d
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

    # --- B. Mode Selection ---------------------------------------------------
    print("\nB. MODE SELECTION")
    print("   Which mode groups should be built?")
    print("   1) All         - Build both PT-Feeder and Rail (default)")
    print("   2) PT-Feeder   - Only PT-Feeder modes (tram, bus, metro, ship, funicular)")
    print("   3) Rail        - Only Rail modes (S-Bahn, regional, inter-regional, long-distance)")

    while True:
        mode_choice = input("\n   Select mode (1-3) [1]: ").strip() or "1"
        if mode_choice in ['1', '2', '3']:
            break
        print("   Invalid selection. Please enter 1, 2, or 3.")

    mode_map = {'1': 'all', '2': 'pt_feeder', '3': 'rail'}
    build_modes = mode_map[mode_choice]

    # --- C. Time-Period Outputs ----------------------------------------------
    print("\nC. TIME-PERIOD OUTPUTS")
    print("   Which time-period GeoPackages & QGZ files should be written?")
    print("   1) All            - Full-day + all-day + peak + off-peak (default)")
    print("   2) All day only   - Services that function constantly throughout the day")
    print("   3) Peak only      - Network during peak hours")
    print("   4) Off-peak only  - Network during off-peak hours")

    while True:
        period_choice = input("\n   Select time-period output (1-4) [1]: ").strip() or "1"
        if period_choice in ['1', '2', '3', '4']:
            break
        print("   Invalid selection. Please enter 1, 2, 3, or 4.")

    #                  (fullday, allday, peak, offpeak)
    period_map = {
        '1': (True,  True,  True,  True),
        '2': (False, True,  False, False),
        '3': (False, False, True,  False),
        '4': (False, False, False, True),
    }
    write_fullday, write_allday, write_peak, write_offpeak = period_map[period_choice]

    # --- D. Output Folder ----------------------------------------------------
    default_output = gtfs_input.replace('GTFS_', '', 1) + '_network' if gtfs_input.startswith('GTFS_') else gtfs_input + '_network'
    print("\nD. OUTPUT FOLDER")
    print(f"   PT-Feeder base: {paths.FEEDER_LINES_DIR}")
    print(f"   Rail base:      {paths.RAIL_LINES_DIR}")
    output_folder = input(
        f"\n   Enter output folder name [{default_output}]: "
    ).strip() or default_output

    # --- Summary -------------------------------------------------------------
    mode_labels = {'all': 'ALL', 'pt_feeder': 'PT-FEEDER ONLY', 'rail': 'RAIL ONLY'}
    period_labels = {
        '1': 'ALL (FULL-DAY + ALL-DAY + PEAK + OFF-PEAK)',
        '2': 'ALL-DAY ONLY',
        '3': 'PEAK ONLY',
        '4': 'OFF-PEAK ONLY',
    }
    print("\n" + "-" * 70)
    print("  CONFIGURATION SUMMARY")
    print("-" * 70)
    print(f"  GTFS input folder  : {gtfs_input}")
    print(f"  Mode selection     : {mode_labels[build_modes]}")
    print(f"  Time-period output : {period_labels[period_choice]}")
    print(f"  Output folder      : {output_folder}")
    print("-" * 70)

    return {
        'gtfs_input_folder': gtfs_input,
        'output_folder':     output_folder,
        'build_modes':       build_modes,
        'write_fullday':     write_fullday,
        'write_allday':      write_allday,
        'write_peak':        write_peak,
        'write_offpeak':     write_offpeak,
    }


# Run configuration and apply to module-level constants
_pipeline_cfg        = _configure_pipeline()
GTFS_INPUT_FOLDER    = _pipeline_cfg['gtfs_input_folder']
PT_FEEDER_OUTPUT_FOLDER = _pipeline_cfg['output_folder']
RAIL_OUTPUT_FOLDER   = _pipeline_cfg['output_folder']
BUILD_MODES          = _pipeline_cfg['build_modes']
WRITE_FULLDAY        = _pipeline_cfg['write_fullday']
WRITE_ALLDAY         = _pipeline_cfg['write_allday']
WRITE_PEAK           = _pipeline_cfg['write_peak']
WRITE_OFFPEAK        = _pipeline_cfg['write_offpeak']


# ---------------------------------------------------------------------------
# Step 0 — setup
# ---------------------------------------------------------------------------

os.chdir(paths.MAIN)

_pt_out_dir       = os.path.join(paths.FEEDER_LINES_DIR,   PT_FEEDER_OUTPUT_FOLDER, paths.SERVICES_UNPROJECTED_SUBDIR)
_rail_out_dir     = os.path.join(paths.RAIL_LINES_DIR,   RAIL_OUTPUT_FOLDER,       paths.SERVICES_UNPROJECTED_SUBDIR)
_pt_allday_dir    = os.path.join(_pt_out_dir,   'All_Day')
_pt_peak_dir      = os.path.join(_pt_out_dir,   'Peak')
_pt_offpeak_dir   = os.path.join(_pt_out_dir,   'Off_Peak')
_rail_allday_dir  = os.path.join(_rail_out_dir,  'All_Day')
_rail_peak_dir    = os.path.join(_rail_out_dir,  'Peak')
_rail_offpeak_dir = os.path.join(_rail_out_dir,  'Off_Peak')

pt_stops_path     = os.path.join(_pt_out_dir,   PT_FEEDER_STOPS_FILE)
pt_lines_path     = os.path.join(_pt_out_dir,   PT_FEEDER_LINES_FILE)
pt_segments_path  = os.path.join(_pt_out_dir,   PT_FEEDER_SEGMENTS_FILE)
rl_stops_path     = os.path.join(_rail_out_dir,  RAIL_STOPS_FILE)
rl_lines_path     = os.path.join(_rail_out_dir,  RAIL_LINES_FILE)
rl_segments_path  = os.path.join(_rail_out_dir,  RAIL_SEGMENTS_FILE)

# All-day paths — written into All_Day subdirectories
pt_lines_allday_path       = os.path.join(_pt_allday_dir,    PT_FEEDER_LINES_ALLDAY_FILE)
pt_segments_allday_path    = os.path.join(_pt_allday_dir,    PT_FEEDER_SEGMENTS_ALLDAY_FILE)
pt_stops_allday_path       = os.path.join(_pt_allday_dir,    PT_FEEDER_STOPS_ALLDAY_FILE)
rl_lines_allday_path       = os.path.join(_rail_allday_dir,  RAIL_LINES_ALLDAY_FILE)
rl_segments_allday_path    = os.path.join(_rail_allday_dir,  RAIL_SEGMENTS_ALLDAY_FILE)
rl_stops_allday_path       = os.path.join(_rail_allday_dir,  RAIL_STOPS_ALLDAY_FILE)

# Peak / off-peak paths — written into Peak / Off_Peak subdirectories
pt_lines_peak_path       = os.path.join(_pt_peak_dir,      PT_FEEDER_LINES_PEAK_FILE)
pt_lines_offpeak_path    = os.path.join(_pt_offpeak_dir,   PT_FEEDER_LINES_OFFPEAK_FILE)
pt_segments_peak_path    = os.path.join(_pt_peak_dir,      PT_FEEDER_SEGMENTS_PEAK_FILE)
pt_segments_offpeak_path = os.path.join(_pt_offpeak_dir,   PT_FEEDER_SEGMENTS_OFFPEAK_FILE)
pt_stops_peak_path       = os.path.join(_pt_peak_dir,      PT_FEEDER_STOPS_PEAK_FILE)
pt_stops_offpeak_path    = os.path.join(_pt_offpeak_dir,   PT_FEEDER_STOPS_OFFPEAK_FILE)
rl_lines_peak_path       = os.path.join(_rail_peak_dir,    RAIL_LINES_PEAK_FILE)
rl_lines_offpeak_path    = os.path.join(_rail_offpeak_dir, RAIL_LINES_OFFPEAK_FILE)
rl_segments_peak_path    = os.path.join(_rail_peak_dir,    RAIL_SEGMENTS_PEAK_FILE)
rl_segments_offpeak_path = os.path.join(_rail_offpeak_dir, RAIL_SEGMENTS_OFFPEAK_FILE)
rl_stops_peak_path       = os.path.join(_rail_peak_dir,    RAIL_STOPS_PEAK_FILE)
rl_stops_offpeak_path    = os.path.join(_rail_offpeak_dir, RAIL_STOPS_OFFPEAK_FILE)

print("=" * 70)
print("services_network_builder.py")
print(f"  CATCHMENT_METHOD : {settings.CATCHMENT_METHOD}")
print(f"  CATCHMENT_CANTON : {settings.CATCHMENT_CANTON_ABBREV}")
print(f"  GTFS source      : {os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_INPUT_FOLDER)}")
print(f"  PT-Feeder output : {_pt_out_dir}")
print(f"  Rail output      : {_rail_out_dir}")
print(f"  Spatial CRS      : {CODEBASE_CRS}")
print("=" * 70)

_ensure_dir(_pt_out_dir)
_ensure_dir(_rail_out_dir)
if WRITE_ALLDAY:
    _ensure_dir(_pt_allday_dir)
    _ensure_dir(_rail_allday_dir)
if WRITE_PEAK:
    _ensure_dir(_pt_peak_dir)
    _ensure_dir(_rail_peak_dir)
if WRITE_OFFPEAK:
    _ensure_dir(_pt_offpeak_dir)
    _ensure_dir(_rail_offpeak_dir)


# ---------------------------------------------------------------------------
# Step 1 — load filtered GTFS
# ---------------------------------------------------------------------------

print("\n[1] Loading filtered GTFS files ...")

stops          = _load('stops.txt')
routes_all     = _load('routes.txt')
trips_all      = _load('trips.txt')
stop_times     = _load('stop_times.txt')
mode_class     = _load('mode_class.txt')
calendar       = _load('calendar.txt')
calendar_dates = _load('calendar_dates.txt')
frequencies_df = _load('frequencies.txt')      # optional — frequency-based GTFS

if any(x is None for x in [stops, routes_all, trips_all, stop_times, mode_class, calendar]):
    raise FileNotFoundError(
        "One or more required filtered GTFS files are missing from "
        f"{os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_INPUT_FOLDER)}. "
        "Run services_filter_gtfs.py first."
    )

# Build frequency lookup: trip_id → list of (start_time, end_time, headway_secs)
# Used by the dual-path logic to expand frequency-based departures on-the-fly.
_freq_trip_ids = set()            # trip_ids that have frequency entries
_freq_lookup   = defaultdict(list)  # trip_id → [(start_minutes, end_minutes, headway_minutes)]
if frequencies_df is not None and len(frequencies_df) > 0:
    for _, row in frequencies_df.iterrows():
        tid = row['trip_id']
        start_min = _to_minutes_str(row['start_time'])
        end_min   = _to_minutes_str(row['end_time'])
        headway_s = float(row['headway_secs'])
        if start_min is None or end_min is None or headway_s <= 0:
            continue
        _freq_trip_ids.add(tid)
        _freq_lookup[tid].append((start_min, end_min, headway_s / 60.0))
    print(f"  frequencies.txt: {len(frequencies_df):,} entries for "
          f"{len(_freq_trip_ids):,} frequency-based trips")
else:
    print("  frequencies.txt: not present — all trips are schedule-based")

# Service IDs that run on at least WEEKDAY_MIN_DAYS out of Mon–Fri
# Derive timetable validity period from calendar.txt date ranges
_cal_starts = pd.to_datetime(calendar['start_date'], format='%Y%m%d', errors='coerce')
_cal_ends   = pd.to_datetime(calendar['end_date'],   format='%Y%m%d', errors='coerce')
TIMETABLE_START = _cal_starts.min().strftime('%Y-%m-%d')
TIMETABLE_END   = _cal_ends.max().strftime('%Y-%m-%d')
print(f"  Timetable period (from calendar.txt): {TIMETABLE_START} to {TIMETABLE_END}")

_weekday_cols = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
_wd_counts = calendar[_weekday_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
_weekday_service_ids = set(calendar.loc[_wd_counts >= WEEKDAY_MIN_DAYS, 'service_id'])
print(f"  Weekday service IDs (≥{WEEKDAY_MIN_DAYS}/5 days): {len(_weekday_service_ids):,}")


# ---------------------------------------------------------------------------
# Step 1.5 — Tier 2 service duration filter (route-level)
#
# Swiss GTFS often encodes a year-round route across many short-lived
# service_ids (seasonal/construction splits).  Evaluating each service_id in
# isolation would incorrectly drop routes like S7 or IC1.
#
# Instead we:
#   1. Compute the set of active weekdays per service_id.
#   2. Map service_ids → route_ids via trips_all.
#   3. Union the active-weekday sets per route_id.
#   4. Retain all service_ids belonging to routes whose union coverage
#      ≥ MIN_WEEKDAY_ACTIVE_FRACTION of available weekdays.
# ---------------------------------------------------------------------------

print(f"\n[1.5] Service duration filter (Tier 2, route-level): "
      f"min {MIN_WEEKDAY_ACTIVE_FRACTION:.0%} of weekdays ...")

def _compute_weekday_active_sets_t2(cal_df, cal_dates_df, tt_start, tt_end):
    """Return (dict: service_id → set of active weekday dates, int: total_available_weekdays)."""
    start = datetime.strptime(tt_start, '%Y-%m-%d')
    end   = datetime.strptime(tt_end,   '%Y-%m-%d')

    # All weekdays in the timetable period
    all_weekdays = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon–Fri
            all_weekdays.append(d)
        d += timedelta(days=1)
    total_weekdays = len(all_weekdays)

    dow_cols = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

    cal = cal_df.copy()
    for col in dow_cols:
        cal[col] = pd.to_numeric(cal[col], errors='coerce').fillna(0).astype(int)
    cal['_start'] = pd.to_datetime(cal['start_date'], format='%Y%m%d', errors='coerce')
    cal['_end']   = pd.to_datetime(cal['end_date'],   format='%Y%m%d', errors='coerce')

    # Parse calendar_dates exceptions
    if cal_dates_df is not None and not cal_dates_df.empty:
        cde = cal_dates_df.copy()
        cde['_date'] = pd.to_datetime(cde['date'], format='%Y%m%d', errors='coerce')
        cde['_type'] = pd.to_numeric(cde['exception_type'], errors='coerce').fillna(0).astype(int)
        removals  = cde.loc[cde['_type'] == 2].groupby('service_id')['_date'].apply(set).to_dict()
        additions = cde.loc[cde['_type'] == 1].groupby('service_id')['_date'].apply(set).to_dict()
    else:
        removals, additions = {}, {}

    results = {}
    for _, row in cal.iterrows():
        sid = row['service_id']
        s_start, s_end = row['_start'], row['_end']
        if pd.isna(s_start) or pd.isna(s_end):
            results[sid] = set()
            continue
        svc_removals  = removals.get(sid, set())
        svc_additions = additions.get(sid, set())
        active_dates = set()
        for dt in all_weekdays:
            in_base = (s_start <= dt <= s_end) and (row[dow_cols[dt.weekday()]] == 1)
            if dt in svc_removals:
                active = False
            elif dt in svc_additions:
                active = True
            else:
                active = in_base
            if active:
                active_dates.add(dt)
        results[sid] = active_dates

    return results, total_weekdays

_sid_active_sets, _total_weekdays = _compute_weekday_active_sets_t2(
    calendar, calendar_dates, TIMETABLE_START, TIMETABLE_END
)
_min_days = int(_total_weekdays * MIN_WEEKDAY_ACTIVE_FRACTION)

print(f"  Timetable period       : {TIMETABLE_START} to {TIMETABLE_END}")
print(f"  Available weekdays     : {_total_weekdays}")
print(f"  Threshold              : {MIN_WEEKDAY_ACTIVE_FRACTION:.0%} = {_min_days} days")

# Map service_ids → route_ids via trips
_sid_to_routes = trips_all.groupby('service_id')['route_id'].apply(set).to_dict()

# Union active-weekday sets per route_id
_route_active_days = {}  # route_id → set of active weekday dates
for sid, active_dates in _sid_active_sets.items():
    for rid in _sid_to_routes.get(sid, set()):
        if rid not in _route_active_days:
            _route_active_days[rid] = set()
        _route_active_days[rid] |= active_dates

# Determine which routes pass the threshold
_routes_passing = {rid for rid, dates in _route_active_days.items() if len(dates) >= _min_days}
_routes_failing = set(_route_active_days.keys()) - _routes_passing

print(f"  Lines evaluated        : {len(_route_active_days):,}")
print(f"  Lines passing (≥{_min_days}d)  : {len(_routes_passing):,}")
print(f"  Lines failing          : {len(_routes_failing):,}")

# Retain all service_ids belonging to passing routes
_sids_of_passing_routes = set()
for sid, routes in _sid_to_routes.items():
    if routes & _routes_passing:
        _sids_of_passing_routes.add(sid)

# Also retain service_ids not linked to any trip (edge case — keep them)
_all_trip_sids = set(_sid_to_routes.keys())
_orphan_sids = set(_sid_active_sets.keys()) - _all_trip_sids
_t2_kept_sids = _sids_of_passing_routes | _orphan_sids
_t2_dropped_sids = set(_sid_active_sets.keys()) - _t2_kept_sids

print(f"  Service IDs retained   : {len(_t2_kept_sids):,}")
print(f"  Service IDs dropped    : {len(_t2_dropped_sids):,}")

# Restrict _weekday_service_ids to only Tier 2 survivors
_weekday_service_ids = _weekday_service_ids & _t2_kept_sids
print(f"  Weekday service IDs after Tier 2: {len(_weekday_service_ids):,}")

# Filter trips_all and cascade into stop_times
_trips_before = len(trips_all)
trips_all = trips_all[trips_all['service_id'].isin(_t2_kept_sids)].copy()
_retained_trip_ids = set(trips_all['trip_id'])
stop_times = stop_times[stop_times['trip_id'].isin(_retained_trip_ids)].copy()
print(f"  Trips after Tier 2     : {len(trips_all):,} / {_trips_before:,}")
print(f"  stop_times after Tier 2: {len(stop_times):,}")


# ---------------------------------------------------------------------------
# Step 2 — build stop point geometry from LV95 columns
# ---------------------------------------------------------------------------

print("\n[2] Building stop geometry from LV95 columns (stop_E, stop_N) ...")

stops['stop_E_f'] = pd.to_numeric(stops['stop_E'], errors='coerce')
stops['stop_N_f'] = pd.to_numeric(stops['stop_N'], errors='coerce')

stops_geo = gpd.GeoDataFrame(
    stops,
    geometry=gpd.points_from_xy(stops['stop_E_f'], stops['stop_N_f']),
    crs=CODEBASE_CRS
)

# --- Parent-station normalisation ---
# GTFS parent stations (location_type=1) have stop_id = "Parent<numeric_id>".
# Child stops (platforms) reference them via parent_station.
# We remap all stop_times stop_ids to the parent's numeric ID so that each
# physical station is a single node in the network.

_parents = stops_geo[stops_geo['location_type'] == '1'].copy()
_parents['parent_numeric_id'] = _parents['stop_id'].str.replace('Parent', '', n=1)

_children = stops_geo[stops_geo['location_type'] != '1'].copy()
_child_to_parent = dict(zip(_children['stop_id'], _children['parent_station'].str.replace('Parent', '', n=1)))

# Build stop_coord and stop_name keyed by parent numeric ID
stop_coord = {
    row['parent_numeric_id']: row['geometry']
    for _, row in _parents.iterrows()
    if row['geometry'] is not None and not row['geometry'].is_empty
}

stop_name = {
    row['parent_numeric_id']: row['stop_name']
    for _, row in _parents.iterrows()
    if 'stop_name' in _parents.columns
}

print(f"  Parent stations : {len(_parents):,}")
print(f"  Child stops     : {len(_children):,}")
print(f"  stop_coord keys : {len(stop_coord):,}")
print(f"  stop_name keys  : {len(stop_name):,}")

# Remap stop_times.stop_id to parent numeric ID
stop_times['stop_id'] = stop_times['stop_id'].map(_child_to_parent).fillna(stop_times['stop_id'])
_unmapped = (~stop_times['stop_id'].isin(stop_coord)).sum()
if _unmapped > 0:
    print(f"  WARNING: {_unmapped:,} stop_times rows with unmapped stop_id")
else:
    print(f"  All stop_times stop_ids mapped to parent stations")




# ---------------------------------------------------------------------------
# Step 3 — split by mode class
# ---------------------------------------------------------------------------

print("\n[3] Splitting lines by mode class ...")

routes_all['route_type_int'] = pd.to_numeric(routes_all['route_type'], errors='coerce')
routes_all = routes_all.merge(mode_class[['route_id', 'mode_class']], on='route_id', how='left')

if isinstance(BUILD_MODES, list):
    # Explicit route_type list — pick only matching route_types from either group
    _selected = set(BUILD_MODES)
    pt_route_ids   = set(routes_all.loc[
        (routes_all['mode_class'] == 'pt_feeder') & (routes_all['route_type_int'].isin(_selected)), 'route_id'
    ])
    rail_route_ids = set(routes_all.loc[
        (routes_all['mode_class'] == 'rail') & (routes_all['route_type_int'].isin(_selected)), 'route_id'
    ])
    print(f"  BUILD_MODES filter: route_types {sorted(_selected)}")
elif BUILD_MODES == 'pt_feeder':
    pt_route_ids   = set(routes_all.loc[routes_all['mode_class'] == 'pt_feeder', 'route_id'])
    rail_route_ids = set()
    print("  BUILD_MODES filter: pt_feeder only")
elif BUILD_MODES == 'rail':
    pt_route_ids   = set()
    rail_route_ids = set(routes_all.loc[routes_all['mode_class'] == 'rail', 'route_id'])
    print("  BUILD_MODES filter: rail only")
else:  # 'all'
    pt_route_ids   = set(routes_all.loc[routes_all['mode_class'] == 'pt_feeder', 'route_id'])
    rail_route_ids = set(routes_all.loc[routes_all['mode_class'] == 'rail',      'route_id'])

print(f"  PT-Feeder lines  : {len(pt_route_ids):,}")
print(f"  Rail lines       : {len(rail_route_ids):,}")


# ---------------------------------------------------------------------------
# Step 4 — prepare stop_times
# ---------------------------------------------------------------------------

print("\n[4] Preparing stop_times ...")

stop_times['stop_sequence_int'] = pd.to_numeric(stop_times['stop_sequence'], errors='coerce')

# Filter trips to operational window (OPERATIONAL_DAY_START–OPERATIONAL_DAY_END, i.e. 06:00–20:00).
# Only trips whose first-stop departure falls within this window are retained.
# This affects variant classification, frequency computation, and trip-share calculations.
_time_col = 'departure_time' if 'departure_time' in stop_times.columns else 'arrival_time'
_st_seq = stop_times.copy()
_st_seq['_seq_int'] = pd.to_numeric(_st_seq['stop_sequence'], errors='coerce')
_first_dep = (
    _st_seq.loc[_st_seq.groupby('trip_id')['_seq_int'].idxmin(), ['trip_id', _time_col]]
    .set_index('trip_id')[_time_col]
)
_in_window = _first_dep[
    (_first_dep >= OPERATIONAL_DAY_START) & (_first_dep < OPERATIONAL_DAY_END)
].index
stop_times = stop_times[stop_times['trip_id'].isin(_in_window)].copy()
print(f"  Trips in operational window ({OPERATIONAL_DAY_START}–{OPERATIONAL_DAY_END}): {len(_in_window):,}")

trips_slim = trips_all[['trip_id', 'route_id', 'direction_id', 'service_id']].copy()
stop_times_enriched = stop_times.merge(trips_slim, on='trip_id', how='left')
print(f"  stop_times enriched: {len(stop_times_enriched):,} rows")

# Pre-build first-stop lookup: one row per trip_id with the earliest stop record.
# Eliminates repeated full-scan of stop_times inside _get_window_departures,
# _detect_peak_windows, and _classify_variants (the dominant runtime bottleneck).
_fst_idx = stop_times.groupby('trip_id')['stop_sequence_int'].idxmin()
_fst_cols = ['trip_id'] + [
    c for c in ['stop_id', 'departure_time', 'arrival_time'] if c in stop_times.columns
]
_first_stop_per_trip = (
    stop_times.loc[_fst_idx, _fst_cols]
    .merge(trips_all[['trip_id', 'route_id', 'direction_id']], on='trip_id', how='left')
    .set_index('trip_id')
)
_scheduled_trip_ids = set(_first_stop_per_trip.index)
print(f"  First-stop lookup  : {len(_first_stop_per_trip):,} trips indexed")


# ---------------------------------------------------------------------------
# Step 5 — frequency computation (uses _compute_frequencies, called from build_outputs)
# ---------------------------------------------------------------------------

print("\n[5] Frequency computation ready (median inter-departure gap method) ...")


# ---------------------------------------------------------------------------
# Step 6 — variant classification and core builder
# ---------------------------------------------------------------------------

def _classify_variants(dir_group, allowed_trip_ids=None, relax_cv=False):
    """Classify stop-sequence variants for a (route_id, direction_id) group.

    Applies three metrics:
      A. Trip share >= VARIANT_MIN_TRIP_SHARE
      B. Hour spread >= VARIANT_MIN_HOUR_SPREAD
      C. Headway CV <= VARIANT_HEADWAY_CV_MAX (using weekday trips, all-day departures)
         — skipped when relax_cv=True (bidirectional enforcement: opposite
         direction already passed all three criteria).

    Parameters
    ----------
    dir_group : DataFrame
        stop_times_enriched rows for one (route_id, direction_id).
    allowed_trip_ids : set or None
        If given, only these trip_ids are considered (dominant-period filter).
        If None, all weekday trips in dir_group are used.
    relax_cv : bool
        If True, skip metric C (headway CV). Used for bidirectional
        enforcement when the opposite direction already passed.

    Returns a list of (stop_id_tuple, trip_id_set) for structural variants only,
    sorted by trip count descending (variant_rank=1 is most common).
    """
    dir_group = dir_group.copy()

    # Get weekday trip_ids for this direction
    all_trip_ids = set(dir_group['trip_id'].unique())
    weekday_trip_ids = set(
        trips_all.loc[
            trips_all['trip_id'].isin(all_trip_ids) &
            trips_all['service_id'].isin(_weekday_service_ids),
            'trip_id'
        ]
    )
    # Restrict to dominant-period trips if provided
    if allowed_trip_ids is not None:
        weekday_trip_ids = weekday_trip_ids & allowed_trip_ids

    if not weekday_trip_ids:
        return []  # No weekday trips — skip (no weekend fallback)

    total_trips = len(weekday_trip_ids)
    if total_trips == 0:
        return []

    # Build stop sequence per trip (stop_ids are already parent-station level)
    seq_per_trip = (
        dir_group[dir_group['trip_id'].isin(weekday_trip_ids)]
        .sort_values('stop_sequence_int')
        .groupby('trip_id')['stop_id']
        .apply(tuple)
    )

    # Group trips by sequence
    seq_to_trips = defaultdict(set)
    for trip_id, seq in seq_per_trip.items():
        seq_to_trips[seq].add(trip_id)

    time_col_global = 'departure_time' if 'departure_time' in stop_times.columns else 'arrival_time'

    structural = []
    for seq, trip_ids in seq_to_trips.items():
        # Metric A: trip share
        share = len(trip_ids) / total_trips
        if share < VARIANT_MIN_TRIP_SHARE:
            continue

        # Get first-stop departure times for these trips.
        # Dual-path: schedule-based trips use stop_times directly;
        # frequency-based trips are expanded from headway bands.
        sched_ids = trip_ids - _freq_trip_ids
        freq_ids  = trip_ids & _freq_trip_ids

        parts = []
        if sched_ids:
            _valid = list(sched_ids & _scheduled_trip_ids)
            if _valid:
                parts.append(_first_stop_per_trip.loc[_valid].reset_index())
        if freq_ids:
            # Expand frequency-based trips across the full operational window
            freq_deps = _expand_freq_departures(
                freq_ids, time_col_global, AM_PEAK_START, PM_PEAK_END)
            if not freq_deps.empty:
                parts.append(freq_deps)

        if not parts:
            continue
        first_stops = pd.concat(parts, ignore_index=True)

        # Full dedup (pass 1: exact time+stop_id, pass 2: terminal merge
        # within TERMINAL_MERGE_MINUTES per route+direction) — matches the
        # dedup that _compute_frequencies sees via _get_window_departures.
        first_stops = _dedup_first_stops(first_stops, time_col_global)

        dep_times = first_stops[time_col_global].apply(_to_minutes_str).dropna().sort_values().tolist()

        # Metric B: hour spread
        hours = set(int(t // 60) % 24 for t in dep_times)
        if len(hours) < VARIANT_MIN_HOUR_SPREAD:
            continue

        # Metric C: headway CV computed per window (AM peak, off-peak, PM peak).
        # A variant passes if its intra-window CV is <= VARIANT_HEADWAY_CV_MAX in
        # at least one window that has enough departures to measure. This prevents
        # penalising lines that legitimately run different headways at different
        # times of day (e.g. 7.5 min peak / 15 min off-peak).
        def _window_cv(w_start, w_end):
            w_times = [t for t in dep_times
                       if _to_minutes_str(w_start) <= t < _to_minutes_str(w_end)]
            if len(w_times) < VARIANT_MIN_DEPARTURES_IN_WINDOW + 1:
                return None  # not enough departures to assess
            gaps = [w_times[i+1] - w_times[i] for i in range(len(w_times)-1)]
            # Drop boundary gaps (first and last) if enough interior gaps remain
            if len(gaps) >= 3:
                gaps = gaps[1:-1]
            if not gaps:
                return None
            mean_gap = sum(gaps) / len(gaps)
            if mean_gap <= 0:
                return None
            try:
                std_gap = statistics.stdev(gaps)
            except Exception:
                std_gap = 0.0
            return std_gap / mean_gap

        cv_am = _window_cv(AM_PEAK_START, AM_PEAK_END)
        cv_op = _window_cv(OFFPEAK_START,  OFFPEAK_END)
        cv_pm = _window_cv(PM_PEAK_START,  PM_PEAK_END)

        # At least one window must have a measurable, regular headway
        # (skipped when relax_cv=True — bidirectional enforcement)
        if not relax_cv:
            measurable = [cv for cv in (cv_am, cv_op, cv_pm) if cv is not None]
            if measurable and min(measurable) > VARIANT_HEADWAY_CV_MAX:
                continue

        structural.append((seq, trip_ids, share))

    # Sort by trip count descending
    structural.sort(key=lambda x: len(x[1]), reverse=True)
    return [(seq, trip_ids) for seq, trip_ids, _ in structural]


def _service_period_tag(trip_ids, windows=None):
    """Return 'peak_only', 'offpeak_only', or 'all_day' based on where departures fall.

    Uses threshold-based classification: if ≥ PEAK_CLASSIFICATION_THRESHOLD of
    departures fall in peak (or off-peak) windows, classify accordingly.
    """
    time_col = 'departure_time' if 'departure_time' in stop_times.columns else 'arrival_time'

    if windows is not None:
        am_s, am_e = windows['am_start'], windows['am_end']
        pm_s, pm_e = windows['pm_start'], windows['pm_end']
        op_s, op_e = windows['op_start'], windows['op_end']
    else:
        am_s, am_e = AM_PEAK_START, AM_PEAK_END
        pm_s, pm_e = PM_PEAK_START, PM_PEAK_END
        op_s, op_e = OFFPEAK_START, OFFPEAK_END

    n_am = len(_get_window_departures(trip_ids, time_col, am_s, am_e))
    n_pm = len(_get_window_departures(trip_ids, time_col, pm_s, pm_e))
    n_op = len(_get_window_departures(trip_ids, time_col, op_s, op_e))

    n_peak = n_am + n_pm
    n_total = n_peak + n_op

    if n_total == 0:
        return 'unknown'

    peak_frac = n_peak / n_total
    op_frac   = n_op / n_total

    if peak_frac >= PEAK_CLASSIFICATION_THRESHOLD:
        return 'peak_only'
    elif op_frac >= PEAK_CLASSIFICATION_THRESHOLD:
        return 'offpeak_only'
    else:
        return 'all_day'


def build_outputs(route_ids, mode_label_map, mode_class_tag):
    """
    Build stops and lines dicts of GeoDataFrames, keyed by route_type.

    For each (route_id, direction_id), variants are classified using three metrics.
    Non-structural variants (depot runs, incidental short-workings) are dropped entirely.
    Each structural variant becomes one feature row.
    Directionality is computed across direction 0 and direction 1 for each line.
    """

    ste        = stop_times_enriched[stop_times_enriched['route_id'].isin(route_ids)].copy()
    routes_sub = routes_all[routes_all['route_id'].isin(route_ids)].copy()

    line_records   = {}   # route_type_int → list of dicts
    stop_ids_by_type = {}  # route_type_int → set of stop_id

    for route_id, rt_group in ste.groupby('route_id'):
        route_row = routes_sub[routes_sub['route_id'] == route_id]
        if route_row.empty:
            continue
        route_row = route_row.iloc[0]

        route_type_int = int(pd.to_numeric(route_row.get('route_type', None), errors='coerce')
                             ) if pd.notna(route_row.get('route_type', None)) else None
        mode_label = mode_label_map.get(route_type_int, 'Unknown') if route_type_int is not None else 'Unknown'
        agency_id  = route_row.get('agency_id', None)
        short_name = route_row.get('route_short_name', None)

        # --- Phase 0: detect dominant calendar period for this route ---
        # Computed once per route_id so both directions share the same period.
        # Moved BEFORE the line gate so departure counts are not inflated by
        # non-overlapping seasonal calendar splits.
        all_route_trip_ids = set(rt_group['trip_id'].unique())
        dominant_trip_ids = _detect_dominant_period(all_route_trip_ids, _sid_active_sets)

        directions = sorted(rt_group['direction_id'].dropna().unique().tolist())
        if not directions:
            directions = ['0']

        # --- Line-level acceptance gate (per direction) ---
        # Each direction individually must have ≥ LINE_MIN_DEPARTURES_PER_DIR
        # departures in at least one window AND ≥ LINE_MIN_HOUR_SPREAD distinct
        # hours.  Uses static windows (adaptive detection runs later).
        _time_col_gate = 'departure_time' if 'departure_time' in stop_times.columns else 'arrival_time'
        _gate_pass = True
        for _dir in directions:
            _dir_trips = set(
                rt_group.loc[rt_group['direction_id'] == _dir, 'trip_id']
            ) & dominant_trip_ids
            _weekday_dir_trips = set(
                trips_all.loc[
                    trips_all['trip_id'].isin(_dir_trips) &
                    trips_all['service_id'].isin(_weekday_service_ids),
                    'trip_id'
                ]
            )
            if not _weekday_dir_trips:
                _gate_pass = False
                break

            # Check ≥ LINE_MIN_DEPARTURES_PER_DIR in at least one window
            _n_am = len(_get_window_departures(_weekday_dir_trips, _time_col_gate, AM_PEAK_START, AM_PEAK_END))
            _n_pm = len(_get_window_departures(_weekday_dir_trips, _time_col_gate, PM_PEAK_START, PM_PEAK_END))
            _n_op = len(_get_window_departures(_weekday_dir_trips, _time_col_gate, OFFPEAK_START, OFFPEAK_END))
            if max(_n_am, _n_pm, _n_op) < LINE_MIN_DEPARTURES_PER_DIR:
                _gate_pass = False
                break

            # Check hour spread per direction
            _gate_sub = stop_times[stop_times['trip_id'].isin(_weekday_dir_trips)]
            _gate_first = _gate_sub.loc[
                _gate_sub.groupby('trip_id')['stop_sequence_int'].idxmin(), _time_col_gate
            ]
            _gate_hours = set()
            for _t in _gate_first.dropna():
                try:
                    _h, _, _ = str(_t).split(':')
                    _gate_hours.add(int(_h) % 24)
                except Exception:
                    pass
            if len(_gate_hours) < LINE_MIN_HOUR_SPREAD:
                _gate_pass = False
                break
        if not _gate_pass:
            continue

        # --- Phase 1: classify variants and apply short-working suppression ---
        def _departure_set(trip_ids_set):
            """Return set of first-stop departure times (minutes) for a trip set."""
            _tc = 'departure_time' if 'departure_time' in stop_times.columns else 'arrival_time'
            _sub = stop_times[stop_times['trip_id'].isin(trip_ids_set)]
            if _sub.empty:
                return set()
            _first = _sub.loc[_sub.groupby('trip_id')['stop_sequence_int'].idxmin(), _tc]
            result = set()
            for t in _first.dropna():
                try:
                    h, m, _ = t.split(':')
                    result.add(int(h) * 60 + int(m))
                except Exception:
                    pass
            return result

        def _overlap_ratio(deps_short, deps_long):
            """Fraction of deps_short that match a departure in deps_long within 1 min."""
            if not deps_short:
                return 0.0
            matched = sum(
                1 for d in deps_short
                if any(abs(d - dl) <= 1 for dl in deps_long)
            )
            return matched / len(deps_short)

        # Collect variants per direction (using dominant-period trips only)
        dir_variants = {}  # direction_id → list of (seq, trip_ids)
        for direction_id in directions:
            dir_group = rt_group[rt_group['direction_id'] == direction_id].copy()
            variants = _classify_variants(dir_group, allowed_trip_ids=dominant_trip_ids)
            if not variants:
                continue
            dir_variants[direction_id] = variants

        # Short-working suppression within each direction (prefix OR suffix match)
        # plus cross-direction propagation: if a short-working is suppressed in one
        # direction, the corresponding short variant in the opposite direction is
        # also suppressed.
        suppressed_stop_counts = set()  # n_stops values suppressed in any direction
        for direction_id, variants in dir_variants.items():
            suppress = set()
            for i, (seq_i, trips_i) in enumerate(variants):
                for j, (seq_j, trips_j) in enumerate(variants):
                    if i == j or len(seq_i) >= len(seq_j):
                        continue
                    # Check if seq_i is a contiguous prefix or suffix of seq_j
                    is_prefix = seq_j[:len(seq_i)] == seq_i
                    is_suffix = seq_j[len(seq_j) - len(seq_i):] == seq_i
                    if is_prefix or is_suffix:
                        deps_i = _departure_set(trips_i)
                        deps_j = _departure_set(trips_j)
                        if _overlap_ratio(deps_i, deps_j) >= SHORTWORKING_OVERLAP_MIN:
                            suppress.add(i)
            for idx in suppress:
                suppressed_stop_counts.add(len(variants[idx][0]))
            dir_variants[direction_id] = [v for k, v in enumerate(variants) if k not in suppress]

        # Cross-direction propagation: if a short-working was suppressed in any
        # direction, suppress variants with the same stop count in other directions
        # when they overlap with a longer variant (relaxed: uses departure overlap
        # against any longer variant, regardless of prefix/suffix match).
        if suppressed_stop_counts:
            for direction_id, variants in dir_variants.items():
                suppress = set()
                for i, (seq_i, trips_i) in enumerate(variants):
                    if len(seq_i) not in suppressed_stop_counts:
                        continue
                    # Check departure overlap against any longer variant
                    for j, (seq_j, trips_j) in enumerate(variants):
                        if i == j or len(seq_i) >= len(seq_j):
                            continue
                        deps_i = _departure_set(trips_i)
                        deps_j = _departure_set(trips_j)
                        if _overlap_ratio(deps_i, deps_j) >= SHORTWORKING_OVERLAP_MIN:
                            suppress.add(i)
                            break
                if suppress:
                    dir_variants[direction_id] = [v for k, v in enumerate(variants) if k not in suppress]

        # --- Variant symmetry enforcement (rail only) ---
        # In Swiss rail the same service always runs the same stopping pattern
        # in both directions (reversed).  After independent per-direction
        # classification we cross-match variants by stop *set* so that both
        # directions use the same structural variants.
        # Bus/tram/ship/funicular routes commonly have minor directional
        # asymmetries (one-way streets, loop terminals, directional-only
        # stops), so symmetry enforcement is skipped for PT-feeder modes.
        #
        # Algorithm:
        #   1. Build the union of accepted variant stop-sets across both
        #      directions.
        #   2. For each canonical stop-set, locate the matching variant in
        #      each direction.  If a direction has no match, force-accept the
        #      trip group whose stop-set matches (relax_cv=True).
        #   3. Variant ranks are synchronised: rank 1 in dir 0 corresponds to
        #      rank 1 in dir 1 (same stop-set).

        if mode_class_tag == 'rail' and len(directions) >= 2:
            d0, d1 = directions[0], directions[1]
            vars_0 = dir_variants.get(d0, [])
            vars_1 = dir_variants.get(d1, [])

            # Index by frozenset of stop_ids
            _set_to_var_0 = {frozenset(seq): (seq, tids) for seq, tids in vars_0}
            _set_to_var_1 = {frozenset(seq): (seq, tids) for seq, tids in vars_1}

            # Canonical stop-sets: union of both directions
            _canonical_sets = list(dict.fromkeys(
                list(_set_to_var_0.keys()) + list(_set_to_var_1.keys())
            ))

            # For each canonical set, ensure both directions have a variant.
            # If one is missing, force-find it from the raw trips.
            for _cset in _canonical_sets:
                for _miss_dir, _, _miss_idx, _ in [
                    (d0, d1, _set_to_var_0, _set_to_var_1),
                    (d1, d0, _set_to_var_1, _set_to_var_0),
                ]:
                    if _cset in _miss_idx:
                        continue  # already present
                    # Try to force-accept from the missing direction's trips
                    _dg_miss = rt_group[rt_group['direction_id'] == _miss_dir].copy()
                    _all_miss = _classify_variants(_dg_miss,
                                                   allowed_trip_ids=dominant_trip_ids,
                                                   relax_cv=True)
                    for _rseq, _rtids in _all_miss:
                        if frozenset(_rseq) == _cset:
                            _miss_idx[_cset] = (_rseq, _rtids)
                            break

            # Rebuild dir_variants with synchronised ranks
            _synced_0, _synced_1 = [], []
            for _cset in _canonical_sets:
                _v0 = _set_to_var_0.get(_cset)
                _v1 = _set_to_var_1.get(_cset)
                if _v0 and _v1:
                    _synced_0.append(_v0)
                    _synced_1.append(_v1)
                # If only one direction has the variant, drop it (no symmetric match)

            dir_variants[d0] = _synced_0 if _synced_0 else []
            dir_variants[d1] = _synced_1 if _synced_1 else []

        # --- Phase 2: detect adaptive peak windows per direction ---
        # Uses only the final accepted variants' trip_ids per direction.
        dir_windows = {}  # direction_id → windows dict or None
        for direction_id in directions:
            variants = dir_variants.get(direction_id)
            if variants:
                accepted_trips = set()
                for _seq, _tids in variants:
                    accepted_trips |= _tids
                dir_windows[direction_id] = _detect_peak_windows(accepted_trips, variants=variants)
            else:
                dir_windows[direction_id] = None
            w = dir_windows[direction_id]
            if w and (w['am_adaptive'] or w['pm_adaptive']):
                print(f"    {route_id} dir={direction_id}: "
                      f"AM=[{w['am_start']}–{w['am_end']}) "
                      f"PM=[{w['pm_start']}–{w['pm_end']}) "
                      f"OP=[{w['op_start']}–{w['op_end']})")

        # --- Emit line features per direction / variant ---
        for direction_id in directions:
            variants = dir_variants.get(direction_id)
            if not variants:
                continue
            dir_group = rt_group[rt_group['direction_id'] == direction_id].copy()

            for variant_rank, (seq, variant_trip_ids) in enumerate(variants, start=1):
                # stop_ids are parent-station level — direct lookup in stop_coord
                geom = _build_linestring(seq, stop_coord)
                if geom is None:
                    continue

                circular        = _is_circular(seq, single_direction=(len(directions) == 1))
                _variant_windows = dir_windows.get(direction_id)
                freqs           = _compute_frequencies(variant_trip_ids, windows=_variant_windows,
                                                       min_departures=VARIANT_FREQ_MIN_DEPARTURES)

                # Drop variants with no computable frequency in any window
                freq_vals = [freqs['freq_am_peak_dep_hr'], freqs['freq_pm_peak_dep_hr'], freqs['freq_offpeak_dep_hr']]
                if all(f is None for f in freq_vals):
                    continue

                # Drop variants whose best frequency is below the minimum threshold
                non_null = [f for f in freq_vals if f is not None]
                if max(non_null) < MIN_FREQUENCY_DEP_HR:
                    continue

                service_period  = _service_period_tag(variant_trip_ids, windows=_variant_windows)

                # Align peak frequencies with service_period classification.
                # offpeak_only: AM/PM values are artefacts of the off-peak service
                #   running across a window boundary, not genuine peak service.
                # peak_only: off-peak value is an artefact of the peak service
                #   spilling into the off-peak window (rare, but makes it explicit).
                freq_am = freqs['freq_am_peak_dep_hr']
                freq_pm = freqs['freq_pm_peak_dep_hr']
                freq_op = freqs['freq_offpeak_dep_hr']
                if service_period == 'offpeak_only':
                    freq_am = None
                    freq_pm = None
                elif service_period == 'peak_only':
                    freq_op = None

                # Total weekday departures across the full operational window (including all off-peak windows)
                _time_col_td = 'departure_time' if 'departure_time' in stop_times.columns else 'arrival_time'
                total_dep = (
                    len(_get_window_departures(variant_trip_ids, _time_col_td, AM_PEAK_START, AM_PEAK_END))
                    + len(_get_window_departures(variant_trip_ids, _time_col_td, PM_PEAK_START, PM_PEAK_END))
                )
                # Add departures from all off-peak windows
                for op_s, op_e in OFFPEAK_WINDOWS:
                    total_dep += len(_get_window_departures(variant_trip_ids, _time_col_td, op_s, op_e))

                # Trip share: weekday trips only (stop_times already filtered to operational window)
                all_dir_trip_ids = set(dir_group['trip_id'].unique())
                weekday_dir_trips = set(
                    trips_all.loc[
                        trips_all['trip_id'].isin(all_dir_trip_ids) &
                        trips_all['service_id'].isin(_weekday_service_ids),
                        'trip_id'
                    ]
                )
                if not weekday_dir_trips:
                    continue  # No weekday trips — skip (no weekend fallback)
                weekday_variant_trips = variant_trip_ids & weekday_dir_trips
                variant_share = len(weekday_variant_trips) / max(len(weekday_dir_trips), 1)

                # Origin and destination from stop names
                origin      = stop_name.get(seq[0],  '') if seq else ''
                destination = stop_name.get(seq[-1], '') if seq else ''
                long_name   = f"{short_name}: {origin} - {destination}" if short_name else f"{origin} - {destination}"

                # Track stop_ids for this route_type (parent-station level)
                stop_ids_by_type.setdefault(route_type_int, set()).update(seq)

                line_records.setdefault(route_type_int, []).append({
                    'route_id':              route_id,
                    'direction_id':          direction_id,
                    'variant_rank':          variant_rank,
                    'variant_trip_share':    round(variant_share, 3),
                    'line_short_name':       short_name,
                    'origin':                origin,
                    'destination':           destination,
                    'line_long_name':        long_name,
                    'line_type':             route_type_int,
                    'mode_label':            mode_label,
                    'mode_class':            mode_class_tag,
                    'agency_id':             agency_id,
                    'is_circular':           circular,
                    'n_stops':               len(seq),
                    'service_period':        service_period,
                    'freq_am_peak_dep_hr':   freq_am,
                    'freq_pm_peak_dep_hr':   freq_pm,
                    'freq_offpeak_dep_hr':   freq_op,
                    'total_dep':             total_dep,
                    'freq_directional':      False,  # placeholder — computed below
                    'geometry':              geom,
                    '_stop_sequence':        seq,
                    '_variant_trip_ids':     variant_trip_ids,
                })

                if circular:
                    break

    # --- Fix 4: Recompute directionality from accepted variant trips ---
    # For each route_id, collect the emitted variant trip_ids per direction
    # and compute directionality using per-direction adaptive windows.
    _route_records = defaultdict(list)  # route_id → list of record dicts
    for rt, records in line_records.items():
        for rec in records:
            _route_records[rec['route_id']].append(rec)

    for route_id, recs in _route_records.items():
        # Collect accepted trip_ids and windows per direction
        _dir_trip_pools = defaultdict(set)  # direction_id → set of trip_ids
        for rec in recs:
            _dir_trip_pools[rec['direction_id']] |= rec['_variant_trip_ids']

        _dirs = sorted(_dir_trip_pools.keys())
        if len(_dirs) >= 2:
            # Use per-direction windows: detect from each direction's own trips
            _w0 = _detect_peak_windows(_dir_trip_pools[_dirs[0]])
            _w1 = _detect_peak_windows(_dir_trip_pools[_dirs[1]])
            # Directionality: check both peak windows using each direction's
            # own adaptive windows for its own frequency computation.
            _time_col_dir = 'departure_time' if 'departure_time' in stop_times.columns else 'arrival_time'
            _directional = False
            for _w in (_w0, _w1):
                if _w is None:
                    continue
                _peak_pairs = [(_w['am_start'], _w['am_end']),
                               (_w['pm_start'], _w['pm_end'])]
                for _ws, _we in _peak_pairs:
                    _f0 = _median_freq(
                        _get_window_departures(_dir_trip_pools[_dirs[0]], _time_col_dir, _ws, _we),
                        _time_col_dir)
                    _f1 = _median_freq(
                        _get_window_departures(_dir_trip_pools[_dirs[1]], _time_col_dir, _ws, _we),
                        _time_col_dir)
                    _f0_ok = _f0 is not None and _f0 > 0
                    _f1_ok = _f1 is not None and _f1 > 0
                    if _f0_ok and _f1_ok:
                        if max(_f0, _f1) / min(_f0, _f1) >= 1.5:
                            _directional = True
                            break
                    elif _f0_ok or _f1_ok:
                        _directional = True
                        break
                if _directional:
                    break
            for rec in recs:
                rec['freq_directional'] = _directional
        else:
            # Single direction — not directional (will be dropped by Fix 1)
            for rec in recs:
                rec['freq_directional'] = False

    # --- Fix 1: Drop routes present in only one direction ---
    # Swiss PT services always operate bidirectionally.  Routes with only
    # one direction surviving indicate geographic filter artefacts or
    # incomplete GTFS data; they are removed entirely.
    # Exception: circular routes (first stop == last stop) legitimately have
    # only one direction_id because the outbound and return are the same trip.
    _single_dir_dropped = 0
    for rt in list(line_records.keys()):
        _route_dirs = defaultdict(set)       # route_id → set of direction_ids
        _route_circular = defaultdict(bool)   # route_id → True if any variant is circular
        for rec in line_records[rt]:
            _route_dirs[rec['route_id']].add(rec['direction_id'])
            if rec.get('is_circular', False):
                _route_circular[rec['route_id']] = True
        _drop_routes = {
            rid for rid, dirs in _route_dirs.items()
            if len(dirs) < 2 and not _route_circular[rid]
        }
        if _drop_routes:
            _before = len(line_records[rt])
            line_records[rt] = [
                r for r in line_records[rt] if r['route_id'] not in _drop_routes
            ]
            _n_dropped = _before - len(line_records[rt])
            _single_dir_dropped += _n_dropped
            # Also remove stop_ids contributed exclusively by dropped routes
            if rt in stop_ids_by_type:
                _remaining_stops = set()
                for r in line_records[rt]:
                    _remaining_stops.update(r['_stop_sequence'])
                stop_ids_by_type[rt] = _remaining_stops
    if _single_dir_dropped:
        print(f"    Dropped {_single_dir_dropped} single-direction features")

    # Remove route_types that became empty after filtering
    line_records = {rt: recs for rt, recs in line_records.items() if recs}

    # Build lines GeoDataFrames per route_type
    lines_by_type = {
        rt: gpd.GeoDataFrame(records, crs=CODEBASE_CRS)
        for rt, records in line_records.items()
    }

    # Build stops GeoDataFrames — only from retained structural variant stops
    stops_by_type = _build_stops_by_type(
        stop_ids_by_type, mode_class_tag, mode_label_map
    )

    return stops_by_type, lines_by_type


def _round_half_min(minutes, floor=0.1):
    """Round a value in minutes to the nearest 0.1. floor=0.1 for TT, floor=0 for IVWT."""
    if minutes is None:
        return None
    return max(floor, round(minutes * 10) / 10)


def _compute_segment_travel_times(variant_trip_ids, seq, time_col='departure_time'):
    """Compute median travel time and dwell time per consecutive stop pair.

    Returns a list of dicts (one per segment) with keys:
        travel_time_min : float  (median, rounded to .0 / .5)
        InVehWait_min   : float  (dwell at from_stop, rounded to .0 / .5)
    """
    # Get stop_times for this variant's trips, sorted by trip then sequence
    sub = stop_times[stop_times['trip_id'].isin(variant_trip_ids)].copy()
    sub = sub[sub['stop_id'].isin(seq)]
    sub = sub.sort_values(['trip_id', 'stop_sequence_int'])

    # Parse times to minutes
    arr_col = 'arrival_time' if 'arrival_time' in sub.columns else time_col
    dep_col = 'departure_time' if 'departure_time' in sub.columns else time_col
    sub['_arr_min'] = sub[arr_col].apply(_to_minutes_str)
    sub['_dep_min'] = sub[dep_col].apply(_to_minutes_str)

    # Build per-trip ordered records: stop_id → (arrival_min, departure_min)
    trip_stop_times = {}  # trip_id → {stop_id → (arr, dep)}
    for _, row in sub.iterrows():
        tid = row['trip_id']
        sid = row['stop_id']
        trip_stop_times.setdefault(tid, {})[sid] = (row['_arr_min'], row['_dep_min'])

    results = []
    for i in range(len(seq) - 1):
        from_sid = seq[i]
        to_sid   = seq[i + 1]

        travel_times = []
        dwell_times  = []
        for tid, st_map in trip_stop_times.items():
            from_rec = st_map.get(from_sid)
            to_rec   = st_map.get(to_sid)
            if from_rec is None or to_rec is None:
                continue
            from_dep = from_rec[1]  # departure at from_stop
            to_arr   = to_rec[0]    # arrival at to_stop
            from_arr = from_rec[0]  # arrival at from_stop
            from_dep_val = from_rec[1]

            if from_dep is not None and to_arr is not None:
                tt = to_arr - from_dep
                if tt >= 0:
                    travel_times.append(tt)
            if from_arr is not None and from_dep_val is not None:
                dw = from_dep_val - from_arr
                if dw >= 0:
                    dwell_times.append(dw)

        if travel_times:
            travel_times.sort()
            median_tt = travel_times[len(travel_times) // 2]
        else:
            median_tt = None

        if dwell_times:
            dwell_times.sort()
            median_dw = dwell_times[len(dwell_times) // 2]
        else:
            median_dw = None

        results.append({
            'travel_time_min': _round_half_min(median_tt),
            'InVehWait_min':   _round_half_min(median_dw, floor=0),
        })

    return results


def build_segments(lines_by_type, mode_class_tag):
    """Build segment GeoDataFrames from accepted line variants.

    For each line feature (variant), generates one segment per consecutive
    stop pair. Each segment carries the parent variant's metadata plus
    travel time and dwell time.

    Returns dict: route_type_int → GeoDataFrame of segments.
    """
    time_col = 'departure_time' if 'departure_time' in stop_times.columns else 'arrival_time'
    segment_records = {}  # route_type_int → list of dicts

    for rt, gdf in lines_by_type.items():
        if gdf.empty:
            continue
        for _, row in gdf.iterrows():
            seq       = row['_stop_sequence']
            trip_ids  = row['_variant_trip_ids']
            if seq is None or len(seq) < 2:
                continue

            seg_times = _compute_segment_travel_times(trip_ids, seq, time_col)

            for i in range(len(seq) - 1):
                from_sid = seq[i]
                to_sid   = seq[i + 1]

                from_pt = stop_coord.get(from_sid)
                to_pt   = stop_coord.get(to_sid)
                if from_pt is None or to_pt is None:
                    continue

                geom = LineString([(from_pt.x, from_pt.y), (to_pt.x, to_pt.y)])

                seg_rec = {
                    'GTFS_ID':            row['route_id'],
                    'Service':            row['line_short_name'],
                    'direction_id':       row['direction_id'],
                    'variant_rank':       row['variant_rank'],
                    'mode_label':         row['mode_label'],
                    'mode_class':         mode_class_tag,
                    'from_stop_nr':       int(from_sid),
                    'to_stop_nr':         int(to_sid),
                    'from_stop_name':     stop_name.get(from_sid, ''),
                    'to_stop_name':       stop_name.get(to_sid, ''),
                    'from_stop_E':        from_pt.x,
                    'from_stop_N':        from_pt.y,
                    'to_stop_E':          to_pt.x,
                    'to_stop_N':          to_pt.y,
                    'TT':                 seg_times[i]['travel_time_min'],
                    'tt_source':          'gtfs',
                    'IVWT':              seg_times[i]['InVehWait_min'],
                    'service_period':     row['service_period'],
                    'freq_am_peak_dep_hr': row.get('freq_am_peak_dep_hr'),
                    'freq_pm_peak_dep_hr': row.get('freq_pm_peak_dep_hr'),
                    'freq_offpeak_dep_hr': row.get('freq_offpeak_dep_hr'),
                    'geometry':           geom,
                }
                segment_records.setdefault(rt, []).append(seg_rec)

    segments_by_type = {
        rt: gpd.GeoDataFrame(records, crs=CODEBASE_CRS)
        for rt, records in segment_records.items()
    }

    return segments_by_type


def _build_stops_by_type(stop_ids_by_type, mode_class_tag, mode_label_map):
    """
    Build per-mode-type stop GeoDataFrames with colour attributes.

    stop_ids are parent-station numeric IDs (e.g. '8502224').  We look them
    up in the parent-station rows of stops_geo (location_type == '1').

    For PT-feeder: each stop is assigned to exactly one layer based on the
    dominant mode (highest-priority route_type serving it per PT_FEEDER_STOP_HIERARCHY).
    For rail: all stops go into a single layer with white fill / black outline.
    """
    # Collect all stop_ids across all types and which types serve each stop
    stop_to_types = {}  # stop_id → set of route_type_int
    for rt, sids in stop_ids_by_type.items():
        for sid in sids:
            stop_to_types.setdefault(sid, set()).add(rt)

    all_stop_ids = set(stop_to_types.keys())

    # Match against parent stations using their numeric ID
    parents = stops_geo[stops_geo['location_type'] == '1'].copy()
    parents['parent_numeric_id'] = parents['stop_id'].str.replace('Parent', '', n=1)
    stops_sub = parents[parents['parent_numeric_id'].isin(all_stop_ids)].copy()

    keep_cols = [c for c in ['parent_numeric_id', 'stop_name', 'stop_lat', 'stop_lon',
                              'geometry'] if c in stops_sub.columns]
    stops_sub = stops_sub[keep_cols].copy()
    stops_sub = stops_sub.rename(columns={'parent_numeric_id': 'Number'})

    if mode_class_tag == 'rail':
        # Single layer: use the first (and typically only) route_type key
        rt = next(iter(stop_ids_by_type)) if stop_ids_by_type else 109
        return {rt: stops_sub.reset_index(drop=True)}

    # PT-feeder: assign dominant mode per stop
    def _dominant(sid):
        served = stop_to_types.get(sid, set())
        for rt in PT_FEEDER_STOP_HIERARCHY:
            if rt in served:
                return rt
        return next(iter(served)) if served else None

    stops_sub['dominant_rt'] = stops_sub['Number'].map(_dominant)

    stops_by_type = {}
    for rt in stops_sub['dominant_rt'].dropna().unique():
        rt_int = int(rt)
        layer  = stops_sub[stops_sub['dominant_rt'] == rt].drop(
            columns=['dominant_rt']
        ).reset_index(drop=True)
        stops_by_type[rt_int] = layer

    return stops_by_type


# ---------------------------------------------------------------------------
# Step 7 — build PT-Feeder outputs
# ---------------------------------------------------------------------------

print("\n[7] Building PT-Feeder stops, lines, and segments ...")
pt_stops_by_type, pt_lines_by_type = build_outputs(pt_route_ids, PT_FEEDER_LINE_TYPES, 'pt_feeder')
pt_segments_by_type = build_segments(pt_lines_by_type, 'pt_feeder')

n_pt_stops    = sum(len(v) for v in pt_stops_by_type.values())
n_pt_lines    = sum(len(v) for v in pt_lines_by_type.values())
n_pt_segments = sum(len(v) for v in pt_segments_by_type.values())
print(f"  PT-Feeder stops    : {n_pt_stops:,} across {len(pt_stops_by_type)} mode layer(s)")
print(f"  PT-Feeder lines    : {n_pt_lines:,} features across {len(pt_lines_by_type)} mode layer(s)")
print(f"  PT-Feeder segments : {n_pt_segments:,} features across {len(pt_segments_by_type)} mode layer(s)")


# ---------------------------------------------------------------------------
# Step 8 — build Rail outputs
# ---------------------------------------------------------------------------

print("\n[8] Building Rail stops, lines, and segments ...")
rail_stops_by_type, rail_lines_by_type = build_outputs(rail_route_ids, RAIL_LINE_TYPES, 'rail')
rail_segments_by_type = build_segments(rail_lines_by_type, 'rail')

n_rail_stops    = sum(len(v) for v in rail_stops_by_type.values())
n_rail_lines    = sum(len(v) for v in rail_lines_by_type.values())
n_rail_segments = sum(len(v) for v in rail_segments_by_type.values())
print(f"  Rail stops    : {n_rail_stops:,} across {len(rail_stops_by_type)} mode layer(s)")
print(f"  Rail lines    : {n_rail_lines:,} features across {len(rail_lines_by_type)} mode layer(s)")
print(f"  Rail segments : {n_rail_segments:,} features across {len(rail_segments_by_type)} mode layer(s)")


# ---------------------------------------------------------------------------
# Step 9 — write GeoPackages
# ---------------------------------------------------------------------------

print("\n[9] Writing GeoPackages ...")


def _write_layers(by_type, gpkg_path):
    """Write each route_type as a named layer into a single GeoPackage.
    Internal columns (prefixed with '_') are stripped before writing."""
    written = []
    for rt, gdf in by_type.items():
        if gdf.empty:
            continue
        layer_name = LAYER_NAMES.get(rt, f'type_{rt}')
        # Strip internal columns before writing
        internal_cols = [c for c in gdf.columns if c.startswith('_')]
        gdf_out = gdf.drop(columns=internal_cols, errors='ignore')
        gdf_out.to_file(gpkg_path, layer=layer_name, driver='GPKG')
        written.append(layer_name)
        print(f"    Layer '{layer_name}': {len(gdf_out):,} features")

    if written:
        print(f"  Wrote {gpkg_path} ({len(written)} layer(s))")
    else:
        print(f"  Skipped {gpkg_path} (no data)")
    return written


if WRITE_FULLDAY:
    print(f"  PT-Feeder lines    → {pt_lines_path}")
    _write_layers(pt_lines_by_type, pt_lines_path)

    print(f"  PT-Feeder stops    → {pt_stops_path}")
    _write_layers(pt_stops_by_type,  pt_stops_path)

    print(f"  PT-Feeder segments → {pt_segments_path}")
    _write_layers(pt_segments_by_type, pt_segments_path)

    print(f"  Rail lines    → {rl_lines_path}")
    _write_layers(rail_lines_by_type, rl_lines_path)

    print(f"  Rail stops    → {rl_stops_path}")
    _write_layers(rail_stops_by_type,  rl_stops_path)

    print(f"  Rail segments → {rl_segments_path}")
    _write_layers(rail_segments_by_type, rl_segments_path)
else:
    print("  Skipping full-day GeoPackage writes (not selected)")


# ---------------------------------------------------------------------------
# Step 9a — peak / off-peak filtering and writing
# ---------------------------------------------------------------------------

print("\n[9a] Building peak / off-peak filtered networks ...")


def _filter_lines_by_period(lines_by_type, periods):
    """Filter line GeoDataFrames to variants matching given service_period values."""
    result = {}
    for rt, gdf in lines_by_type.items():
        if gdf.empty:
            continue
        filtered = gdf[gdf['service_period'].isin(periods)].copy()
        if not filtered.empty:
            result[rt] = filtered
    return result


def _filter_segments_by_period(segments_by_type, periods):
    """Filter segment GeoDataFrames to segments matching given service_period values."""
    result = {}
    for rt, gdf in segments_by_type.items():
        if gdf.empty:
            continue
        filtered = gdf[gdf['service_period'].isin(periods)].copy()
        if not filtered.empty:
            result[rt] = filtered
    return result


def _filter_stops_for_lines(lines_by_type, stops_by_type):
    """Filter stops to only those referenced by the given line variants."""
    # Collect all stop_ids from retained line features
    all_stop_ids = set()
    for rt, gdf in lines_by_type.items():
        if gdf.empty or '_stop_sequence' not in gdf.columns:
            continue
        for seq in gdf['_stop_sequence']:
            if seq is not None:
                all_stop_ids.update(seq)

    result = {}
    for rt, gdf in stops_by_type.items():
        if gdf.empty:
            continue
        filtered = gdf[gdf['Number'].isin(all_stop_ids)].copy()
        if not filtered.empty:
            result[rt] = filtered
    return result


ALLDAY_PERIODS  = {'all_day'}
PEAK_PERIODS    = {'peak_only', 'all_day'}
OFFPEAK_PERIODS = {'offpeak_only', 'all_day'}

# Initialise filtered dicts to empty — populated below only when requested.
pt_lines_allday = pt_lines_peak = pt_lines_offpeak = {}
pt_segs_allday = pt_segs_peak = pt_segs_offpeak = {}
pt_stops_allday = pt_stops_peak = pt_stops_offpeak = {}
rl_lines_allday = rl_lines_peak = rl_lines_offpeak = {}
rl_segs_allday = rl_segs_peak = rl_segs_offpeak = {}
rl_stops_allday = rl_stops_peak = rl_stops_offpeak = {}

if WRITE_ALLDAY:
    pt_lines_allday   = _filter_lines_by_period(pt_lines_by_type, ALLDAY_PERIODS)
    pt_segs_allday    = _filter_segments_by_period(pt_segments_by_type, ALLDAY_PERIODS)
    pt_stops_allday   = _filter_stops_for_lines(pt_lines_allday,  pt_stops_by_type)
    rl_lines_allday   = _filter_lines_by_period(rail_lines_by_type, ALLDAY_PERIODS)
    rl_segs_allday    = _filter_segments_by_period(rail_segments_by_type, ALLDAY_PERIODS)
    rl_stops_allday   = _filter_stops_for_lines(rl_lines_allday,  rail_stops_by_type)

if WRITE_PEAK:
    pt_lines_peak     = _filter_lines_by_period(pt_lines_by_type, PEAK_PERIODS)
    pt_segs_peak      = _filter_segments_by_period(pt_segments_by_type, PEAK_PERIODS)
    pt_stops_peak     = _filter_stops_for_lines(pt_lines_peak,    pt_stops_by_type)
    rl_lines_peak     = _filter_lines_by_period(rail_lines_by_type, PEAK_PERIODS)
    rl_segs_peak      = _filter_segments_by_period(rail_segments_by_type, PEAK_PERIODS)
    rl_stops_peak     = _filter_stops_for_lines(rl_lines_peak,    rail_stops_by_type)

if WRITE_OFFPEAK:
    pt_lines_offpeak  = _filter_lines_by_period(pt_lines_by_type, OFFPEAK_PERIODS)
    pt_segs_offpeak   = _filter_segments_by_period(pt_segments_by_type, OFFPEAK_PERIODS)
    pt_stops_offpeak  = _filter_stops_for_lines(pt_lines_offpeak, pt_stops_by_type)
    rl_lines_offpeak  = _filter_lines_by_period(rail_lines_by_type, OFFPEAK_PERIODS)
    rl_segs_offpeak   = _filter_segments_by_period(rail_segments_by_type, OFFPEAK_PERIODS)
    rl_stops_offpeak  = _filter_stops_for_lines(rl_lines_offpeak, rail_stops_by_type)

# Write all-day / peak / off-peak GeoPackages
_period_write_list = []
if WRITE_ALLDAY:
    _period_write_list += [
        ('PT-Feeder lines allday',     pt_lines_allday,   pt_lines_allday_path),
        ('PT-Feeder segments allday',  pt_segs_allday,    pt_segments_allday_path),
        ('PT-Feeder stops allday',     pt_stops_allday,   pt_stops_allday_path),
        ('Rail lines allday',          rl_lines_allday,   rl_lines_allday_path),
        ('Rail segments allday',       rl_segs_allday,    rl_segments_allday_path),
        ('Rail stops allday',          rl_stops_allday,   rl_stops_allday_path),
    ]
if WRITE_PEAK:
    _period_write_list += [
        ('PT-Feeder lines peak',       pt_lines_peak,     pt_lines_peak_path),
        ('PT-Feeder segments peak',    pt_segs_peak,      pt_segments_peak_path),
        ('PT-Feeder stops peak',       pt_stops_peak,     pt_stops_peak_path),
        ('Rail lines peak',            rl_lines_peak,     rl_lines_peak_path),
        ('Rail segments peak',         rl_segs_peak,      rl_segments_peak_path),
        ('Rail stops peak',            rl_stops_peak,     rl_stops_peak_path),
    ]
if WRITE_OFFPEAK:
    _period_write_list += [
        ('PT-Feeder lines offpeak',    pt_lines_offpeak,  pt_lines_offpeak_path),
        ('PT-Feeder segments offpeak', pt_segs_offpeak,   pt_segments_offpeak_path),
        ('PT-Feeder stops offpeak',    pt_stops_offpeak,  pt_stops_offpeak_path),
        ('Rail lines offpeak',         rl_lines_offpeak,  rl_lines_offpeak_path),
        ('Rail segments offpeak',      rl_segs_offpeak,   rl_segments_offpeak_path),
        ('Rail stops offpeak',         rl_stops_offpeak,  rl_stops_offpeak_path),
    ]

if _period_write_list:
    for label, data, path in _period_write_list:
        print(f"  {label} → {path}")
        _write_layers(data, path)
else:
    print("  Skipping peak/off-peak GeoPackage writes (not selected)")


# ---------------------------------------------------------------------------
# Step 9b — write QGIS project files (.qgz) with styled layers
# ---------------------------------------------------------------------------

print("\n[9b] Writing QGIS project files ...")

def _collect_line_layers(by_type, gpkg_filename, label_map, suffix='Lines'):
    """Collect line layer descriptors for a QGZ project."""
    layers = []
    counter = 0
    for rt, gdf in sorted(by_type.items()):
        if gdf.empty:
            continue
        layer_name = LAYER_NAMES.get(rt, f'type_{rt}')
        counter += 1
        layers.append({
            'layer_id':      f'{layer_name}_lines_{counter:04d}',
            'gpkg_relpath':  f'./{gpkg_filename}',
            'layer_name':    layer_name,
            'display_name':  f'{label_map.get(rt, layer_name)} {suffix}',
            'geom_type':     'line',
            'colour':        LINE_COLOURS.get(rt, '#888888'),
            'line_style':    LINE_STYLE.get(rt, 'solid'),
        })
    return layers

def _collect_stop_layers(by_type, gpkg_filename, label_map, mode_class_tag):
    """Collect stop layer descriptors for a QGZ project."""
    layers = []
    counter = 0
    for rt, gdf in sorted(by_type.items()):
        if gdf.empty:
            continue
        layer_name = LAYER_NAMES.get(rt, f'type_{rt}')
        if mode_class_tag == 'rail':
            fill = RAIL_STOP_FILL
            outline = RAIL_STOP_OUTLINE
        else:
            fill = LINE_COLOURS.get(rt, '#888888')
            outline = '#000000'
        counter += 1
        layers.append({
            'layer_id':        f'{layer_name}_stops_{counter:04d}',
            'gpkg_relpath':    f'./{gpkg_filename}',
            'layer_name':      layer_name,
            'display_name':    f'{label_map.get(rt, layer_name)} Stops',
            'geom_type':       'point',
            'fill_colour':     fill,
            'outline_colour':  outline,
        })
    return layers

def _write_qgz(out_dir, qgz_filename, layer_list):
    """Build and write a QGZ file from a list of layer descriptors."""
    qgz_path = os.path.join(out_dir, qgz_filename)
    _build_qgz(qgz_path, layer_list)
    print(f"  Wrote {qgz_path} ({len(layer_list)} layer(s))")
    return qgz_path

def _build_lines_qgz(out_dir, qgz_filename, lines_by_type, lines_gpkg,
                      stops_by_type, stops_gpkg, label_map, mode_class_tag):
    """Build a QGZ with stops below and lines on top."""
    layers  = _collect_stop_layers(stops_by_type, stops_gpkg, label_map, mode_class_tag)
    layers += _collect_line_layers(lines_by_type, lines_gpkg, label_map, suffix='Lines')
    return _write_qgz(out_dir, qgz_filename, layers)

def _build_segments_qgz(out_dir, qgz_filename, segs_by_type, segs_gpkg,
                         stops_by_type, stops_gpkg, label_map, mode_class_tag):
    """Build a QGZ with stops below and segments on top."""
    layers  = _collect_stop_layers(stops_by_type, stops_gpkg, label_map, mode_class_tag)
    layers += _collect_line_layers(segs_by_type, segs_gpkg, label_map, suffix='Segments')
    return _write_qgz(out_dir, qgz_filename, layers)

# --- PT-Feeder QGZ files ---
if WRITE_FULLDAY:
    _build_lines_qgz(_pt_out_dir, PT_FEEDER_PROJECT_FILE,
                      pt_lines_by_type, PT_FEEDER_LINES_FILE,
                      pt_stops_by_type, PT_FEEDER_STOPS_FILE,
                      PT_FEEDER_LINE_TYPES, 'pt_feeder')

    _build_segments_qgz(_pt_out_dir, PT_FEEDER_SEGMENTS_PROJECT_FILE,
                         pt_segments_by_type, PT_FEEDER_SEGMENTS_FILE,
                         pt_stops_by_type, PT_FEEDER_STOPS_FILE,
                         PT_FEEDER_LINE_TYPES, 'pt_feeder')

# PT-Feeder all-day QGZ
if WRITE_ALLDAY:
    _build_lines_qgz(_pt_allday_dir, PT_FEEDER_LINES_ALLDAY_PROJECT_FILE,
                      pt_lines_allday, PT_FEEDER_LINES_ALLDAY_FILE,
                      pt_stops_allday, PT_FEEDER_STOPS_ALLDAY_FILE,
                      PT_FEEDER_LINE_TYPES, 'pt_feeder')
    _build_segments_qgz(_pt_allday_dir, PT_FEEDER_SEGMENTS_ALLDAY_PROJECT_FILE,
                         pt_segs_allday, PT_FEEDER_SEGMENTS_ALLDAY_FILE,
                         pt_stops_allday, PT_FEEDER_STOPS_ALLDAY_FILE,
                         PT_FEEDER_LINE_TYPES, 'pt_feeder')

# PT-Feeder peak / off-peak QGZ
if WRITE_PEAK:
    _build_lines_qgz(_pt_peak_dir, PT_FEEDER_LINES_PEAK_PROJECT_FILE,
                      pt_lines_peak, PT_FEEDER_LINES_PEAK_FILE,
                      pt_stops_peak, PT_FEEDER_STOPS_PEAK_FILE,
                      PT_FEEDER_LINE_TYPES, 'pt_feeder')
    _build_segments_qgz(_pt_peak_dir, PT_FEEDER_SEGMENTS_PEAK_PROJECT_FILE,
                         pt_segs_peak, PT_FEEDER_SEGMENTS_PEAK_FILE,
                         pt_stops_peak, PT_FEEDER_STOPS_PEAK_FILE,
                         PT_FEEDER_LINE_TYPES, 'pt_feeder')
if WRITE_OFFPEAK:
    _build_lines_qgz(_pt_offpeak_dir, PT_FEEDER_LINES_OFFPEAK_PROJECT_FILE,
                      pt_lines_offpeak, PT_FEEDER_LINES_OFFPEAK_FILE,
                      pt_stops_offpeak, PT_FEEDER_STOPS_OFFPEAK_FILE,
                      PT_FEEDER_LINE_TYPES, 'pt_feeder')
    _build_segments_qgz(_pt_offpeak_dir, PT_FEEDER_SEGMENTS_OFFPEAK_PROJECT_FILE,
                         pt_segs_offpeak, PT_FEEDER_SEGMENTS_OFFPEAK_FILE,
                         pt_stops_offpeak, PT_FEEDER_STOPS_OFFPEAK_FILE,
                         PT_FEEDER_LINE_TYPES, 'pt_feeder')

# --- Rail QGZ files ---
if WRITE_FULLDAY:
    _build_lines_qgz(_rail_out_dir, RAIL_PROJECT_FILE,
                      rail_lines_by_type, RAIL_LINES_FILE,
                      rail_stops_by_type, RAIL_STOPS_FILE,
                      RAIL_LINE_TYPES, 'rail')
    _build_segments_qgz(_rail_out_dir, RAIL_SEGMENTS_PROJECT_FILE,
                        rail_segments_by_type, RAIL_SEGMENTS_FILE,
                        rail_stops_by_type, RAIL_STOPS_FILE,
                        RAIL_LINE_TYPES, 'rail')

# Rail all-day QGZ
if WRITE_ALLDAY:
    _build_lines_qgz(_rail_allday_dir, RAIL_LINES_ALLDAY_PROJECT_FILE,
                      rl_lines_allday, RAIL_LINES_ALLDAY_FILE,
                      rl_stops_allday, RAIL_STOPS_ALLDAY_FILE,
                      RAIL_LINE_TYPES, 'rail')
    _build_segments_qgz(_rail_allday_dir, RAIL_SEGMENTS_ALLDAY_PROJECT_FILE,
                         rl_segs_allday, RAIL_SEGMENTS_ALLDAY_FILE,
                         rl_stops_allday, RAIL_STOPS_ALLDAY_FILE,
                         RAIL_LINE_TYPES, 'rail')

# Rail peak / off-peak QGZ
if WRITE_PEAK:
    _build_lines_qgz(_rail_peak_dir, RAIL_LINES_PEAK_PROJECT_FILE,
                      rl_lines_peak, RAIL_LINES_PEAK_FILE,
                      rl_stops_peak, RAIL_STOPS_PEAK_FILE,
                      RAIL_LINE_TYPES, 'rail')
    _build_segments_qgz(_rail_peak_dir, RAIL_SEGMENTS_PEAK_PROJECT_FILE,
                         rl_segs_peak, RAIL_SEGMENTS_PEAK_FILE,
                         rl_stops_peak, RAIL_STOPS_PEAK_FILE,
                         RAIL_LINE_TYPES, 'rail')
if WRITE_OFFPEAK:
    _build_lines_qgz(_rail_offpeak_dir, RAIL_LINES_OFFPEAK_PROJECT_FILE,
                      rl_lines_offpeak, RAIL_LINES_OFFPEAK_FILE,
                      rl_stops_offpeak, RAIL_STOPS_OFFPEAK_FILE,
                      RAIL_LINE_TYPES, 'rail')
    _build_segments_qgz(_rail_offpeak_dir, RAIL_SEGMENTS_OFFPEAK_PROJECT_FILE,
                         rl_segs_offpeak, RAIL_SEGMENTS_OFFPEAK_FILE,
                         rl_stops_offpeak, RAIL_STOPS_OFFPEAK_FILE,
                         RAIL_LINE_TYPES, 'rail')


# ---------------------------------------------------------------------------
# Step 10 — validation
# ---------------------------------------------------------------------------

print("\n[10] Validation ...")

tag_pt_stops    = "PASS" if n_pt_stops    > 0 else "FAIL (empty)"
tag_pt_lines    = "PASS" if n_pt_lines    > 0 else "FAIL (empty)"
tag_pt_segments = "PASS" if n_pt_segments > 0 else "FAIL (empty)"
tag_rl_stops    = "PASS" if n_rail_stops    > 0 else "FAIL (empty)"
tag_rl_lines    = "PASS" if n_rail_lines    > 0 else "FAIL (empty)"
tag_rl_segments = "PASS" if n_rail_segments > 0 else "FAIL (empty)"

# Null geometry check across all layers
def _null_geom_count(by_type):
    return sum(gdf['geometry'].isna().sum() for gdf in by_type.values() if not gdf.empty)

bad_pt_geom     = _null_geom_count(pt_lines_by_type)
bad_rl_geom     = _null_geom_count(rail_lines_by_type)
bad_pt_seg_geom = _null_geom_count(pt_segments_by_type)
bad_rl_seg_geom = _null_geom_count(rail_segments_by_type)
tag_pt_geom     = "PASS" if bad_pt_geom == 0 else f"FAIL ({bad_pt_geom} null geometries)"
tag_rl_geom     = "PASS" if bad_rl_geom == 0 else f"FAIL ({bad_rl_geom} null geometries)"
tag_pt_seg_geom = "PASS" if bad_pt_seg_geom == 0 else f"FAIL ({bad_pt_seg_geom} null geometries)"
tag_rl_seg_geom = "PASS" if bad_rl_seg_geom == 0 else f"FAIL ({bad_rl_seg_geom} null geometries)"

# Direction_id check
def _bad_dir_count(by_type):
    count = 0
    for gdf in by_type.values():
        if not gdf.empty and 'direction_id' in gdf.columns:
            count += (~gdf['direction_id'].isin(['0', '1'])).sum()
    return count

bad_dir_pt = _bad_dir_count(pt_lines_by_type)
bad_dir_rl = _bad_dir_count(rail_lines_by_type)
tag_dir_pt = "PASS" if bad_dir_pt == 0 else f"WARN ({bad_dir_pt} unexpected direction_id values)"
tag_dir_rl = "PASS" if bad_dir_rl == 0 else f"WARN ({bad_dir_rl} unexpected direction_id values)"

print(f"  PT-Feeder stops non-empty      : {tag_pt_stops}")
print(f"  PT-Feeder lines non-empty      : {tag_pt_lines}")
print(f"  PT-Feeder segments non-empty   : {tag_pt_segments}")
print(f"  PT-Feeder line geometries      : {tag_pt_geom}")
print(f"  PT-Feeder segment geometries   : {tag_pt_seg_geom}")
print(f"  PT-Feeder direction_id valid   : {tag_dir_pt}")
print(f"  Rail stops non-empty           : {tag_rl_stops}")
print(f"  Rail lines non-empty           : {tag_rl_lines}")
print(f"  Rail segments non-empty        : {tag_rl_segments}")
print(f"  Rail line geometries           : {tag_rl_geom}")
print(f"  Rail segment geometries        : {tag_rl_seg_geom}")
print(f"  Rail direction_id valid        : {tag_dir_rl}")


# ---------------------------------------------------------------------------
# Step 11 — build report
# ---------------------------------------------------------------------------

elapsed = time.time() - _start_time
print(f"\n[11] Writing {BUILD_REPORT_FILE} ...")


def _rt_breakdown(by_type, label_map):
    if not by_type:
        return ["  (no data)"]
    lines = []
    for rt, gdf in sorted(by_type.items()):
        if gdf.empty:
            continue
        label      = label_map.get(rt, 'Unknown')
        layer_name = LAYER_NAMES.get(rt, f'type_{rt}')
        n_lines    = gdf['route_id'].nunique() if 'route_id' in gdf.columns else len(gdf)
        lines.append(f"    {rt:<6}  {label:<25}  {layer_name:<22}  {n_lines:>5,} lines")
    return lines or ["  (no data)"]


_mode_label_report = {'all': 'All (PT-Feeder + Rail)', 'pt_feeder': 'PT-Feeder only', 'rail': 'Rail only'}
_period_parts = []
if WRITE_FULLDAY:  _period_parts.append('full-day')
if WRITE_ALLDAY:   _period_parts.append('all-day')
if WRITE_PEAK:     _period_parts.append('peak')
if WRITE_OFFPEAK:  _period_parts.append('off-peak')
_period_label_report = ' + '.join(_period_parts) if _period_parts else '(none)'

_output_files_report = []
if WRITE_FULLDAY:
    _output_files_report += [
        f"  {pt_stops_path}", f"  {pt_lines_path}", f"  {pt_segments_path}",
        f"  {rl_stops_path}", f"  {rl_lines_path}", f"  {rl_segments_path}",
    ]
if WRITE_ALLDAY:
    _output_files_report += [
        f"  {pt_lines_allday_path}", f"  {pt_segments_allday_path}", f"  {pt_stops_allday_path}",
        f"  {rl_lines_allday_path}", f"  {rl_segments_allday_path}", f"  {rl_stops_allday_path}",
    ]
if WRITE_PEAK:
    _output_files_report += [
        f"  {pt_lines_peak_path}", f"  {pt_segments_peak_path}", f"  {pt_stops_peak_path}",
        f"  {rl_lines_peak_path}", f"  {rl_segments_peak_path}", f"  {rl_stops_peak_path}",
    ]
if WRITE_OFFPEAK:
    _output_files_report += [
        f"  {pt_lines_offpeak_path}", f"  {pt_segments_offpeak_path}", f"  {pt_stops_offpeak_path}",
        f"  {rl_lines_offpeak_path}", f"  {rl_segments_offpeak_path}", f"  {rl_stops_offpeak_path}",
    ]
_output_files_report.append("  + QGIS .qgz project files for each written GeoPackage")

report_lines = [
    "=" * 70,
    "NETWORK BUILD REPORT — services_network_builder.py",
    "=" * 70,
    "",
    "PIPELINE CONFIGURATION",
    f"  GTFS input folder  : {GTFS_INPUT_FOLDER}",
    f"  Output folder      : {PT_FEEDER_OUTPUT_FOLDER}",
    f"  Mode selection     : {_mode_label_report.get(BUILD_MODES, BUILD_MODES)}",
    f"  Time-period output : {_period_label_report}",
    f"  Geometry source    : Straight-line (projection applied separately by services_service_projection.py)",
    "",
    "CONFIGURATION",
    f"  CATCHMENT_METHOD : {settings.CATCHMENT_METHOD}",
    f"  CATCHMENT_CANTON : {settings.CATCHMENT_CANTON_ABBREV}",
    f"  GTFS source      : {os.path.join(paths.GTFS_TRANSIT_DIR, GTFS_INPUT_FOLDER)}",
    f"  Output folder    : {PT_FEEDER_OUTPUT_FOLDER}",
    f"  Spatial CRS      : {CODEBASE_CRS}",
    f"  Weekday filter   : ≥{WEEKDAY_MIN_DAYS}/5 weekdays",
    f"  Freq-based trips : {len(_freq_trip_ids):,} trips from frequencies.txt",
    "",
    "OUTPUT FILES",
] + _output_files_report + [
    "",
    "RECORD COUNTS",
    f"  {'Layer':<40}  {'Features':>8}",
    f"  {'-'*40}  {'-'*8}",
    f"  {'pt_feeder_stops (all layers)':<40}  {n_pt_stops:>8,}",
    f"  {'pt_feeder_lines (all layers)':<40}  {n_pt_lines:>8,}",
    f"  {'pt_feeder_segments (all layers)':<40}  {n_pt_segments:>8,}",
    f"  {'rail_stops (all layers)':<40}  {n_rail_stops:>8,}",
    f"  {'rail_lines (all layers)':<40}  {n_rail_lines:>8,}",
    f"  {'rail_segments (all layers)':<40}  {n_rail_segments:>8,}",
    "",
    "PT-FEEDER LINE TYPE BREAKDOWN",
    f"  {'rt':<6}  {'Mode':<25}  {'Layer':<22}  {'Lines':>6}",
    f"  {'-'*6}  {'-'*25}  {'-'*22}  {'-'*6}",
] + _rt_breakdown(pt_lines_by_type, PT_FEEDER_LINE_TYPES) + [
    "",
    "RAIL LINE TYPE BREAKDOWN",
    f"  {'rt':<6}  {'Mode':<25}  {'Layer':<22}  {'Lines':>6}",
    f"  {'-'*6}  {'-'*25}  {'-'*22}  {'-'*6}",
] + _rt_breakdown(rail_lines_by_type, RAIL_LINE_TYPES) + [
    "",
    "REPRESENTATIVE TRIP SELECTION",
    "  Structural variants only — classified by trip share, hour spread,",
    "  and headway CV. Non-structural variants (depot runs, incidental",
    "  short-workings) are dropped. Variants with NULL frequency in all",
    "  windows or max frequency < {0} dep/hr are also dropped.".format(MIN_FREQUENCY_DEP_HR),
    "  Each structural variant becomes one feature row. Circular lines",
    "  produce one feature only.",
    "",
    "FREQUENCY",
    f"  AM peak validation window : {AM_PEAK_START} – {AM_PEAK_END}",
    f"  PM peak validation window : {PM_PEAK_START} – {PM_PEAK_END}",
    f"  Off-peak fallback window  : {OFFPEAK_START} – {OFFPEAK_END}",
    "  Peak detection             : adaptive per (route_id, direction_id)",
    f"  Detection bin width        : {PEAK_BIN_WIDTH_MIN} min",
    f"  Min cluster width          : {PEAK_MIN_CLUSTER_BINS} bins ({PEAK_MIN_CLUSTER_BINS * PEAK_BIN_WIDTH_MIN} min)",
    f"  Validation min overlap     : {PEAK_MIN_OVERLAP:.0%}",
    f"  Classification threshold   : {PEAK_CLASSIFICATION_THRESHOLD:.0%}",
    f"  Min deps per window (CV)   : {VARIANT_MIN_DEPARTURES_IN_WINDOW} (variant classification headway CV)",
    f"  Min deps per window (freq) : {VARIANT_FREQ_MIN_DEPARTURES} (frequency computation)",
    f"  Weekday filter             : service_ids running on ≥{WEEKDAY_MIN_DAYS} of Mon–Fri",
    "  Method                     : median inter-departure gap (interior gaps, boundary gaps dropped)",
    "  Columns                    : freq_am_peak_dep_hr, freq_pm_peak_dep_hr, freq_offpeak_dep_hr",
    "                               freq_directional (True if peak freq differs ≥1.5× between directions)",
    "  Variants                   : structural only (trip share, hour spread, headway CV filters applied)",
    "  Column                     : service_period (all_day / peak_only / offpeak_only / unknown)",
    "",
    "GEOMETRY",
    "  Source : straight-line stop-to-stop (unprojected)",
    "  Projection is applied separately by services_service_projection.py",
    "",
    "SEGMENTS",
    "  Each segment = one stop-to-stop edge from a variant's stop sequence",
    "  travel_time_min : median scheduled travel time (rounded to .0 / .5 min)",
    "  InVehWait_min   : median dwell at from_stop (rounded to .0 / .5 min)",
    "  Coordinates     : from_stop_E/N, to_stop_E/N (EPSG:2056)",
    "",
    "ALL-DAY / PEAK / OFF-PEAK NETWORKS",
    f"  All-day outputs written : {'Yes' if WRITE_ALLDAY else 'No (skipped)'}",
    f"  Peak outputs written    : {'Yes' if WRITE_PEAK else 'No (skipped)'}",
    f"  Off-peak outputs written: {'Yes' if WRITE_OFFPEAK else 'No (skipped)'}",
    "  All-day : service_period in {all_day}",
    "  Peak    : service_period in {peak_only, all_day}",
    "  Off-peak: service_period in {offpeak_only, all_day}",
    "",
    "STYLING",
    "  Styling applied via QGZ project files (no colour columns in GeoPackage)",
    "  QGIS .qgz project files with pre-configured symbology per output dir",
    "",
    "VALIDATION",
    f"  PT-Feeder stops non-empty      : {tag_pt_stops}",
    f"  PT-Feeder lines non-empty      : {tag_pt_lines}",
    f"  PT-Feeder segments non-empty   : {tag_pt_segments}",
    f"  PT-Feeder line geometries      : {tag_pt_geom}",
    f"  PT-Feeder segment geometries   : {tag_pt_seg_geom}",
    f"  PT-Feeder direction_id valid   : {tag_dir_pt}",
    f"  Rail stops non-empty           : {tag_rl_stops}",
    f"  Rail lines non-empty           : {tag_rl_lines}",
    f"  Rail segments non-empty        : {tag_rl_segments}",
    f"  Rail line geometries           : {tag_rl_geom}",
    f"  Rail segment geometries        : {tag_rl_seg_geom}",
    f"  Rail direction_id valid        : {tag_dir_rl}",
    "",
    "RUNTIME",
    f"  Total elapsed                : {elapsed:.1f} seconds",
    "",
    "=" * 70,
]

report_text = "\n".join(report_lines)
report_path = os.path.join(_pt_out_dir, BUILD_REPORT_FILE)
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

print(report_text)
print("\nDone.")
