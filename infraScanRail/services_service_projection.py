"""
Service Projection Module

Maps rail, tram, and funicular services onto the BAV infrastructure network.
Routes GTFS-derived service links along shortest paths on the BAV graph,
enriching existing service files with real segment geometry and via-node annotations.

Workflow (interactive):
  Phase 0 — CLI setup: choose infra version + service version
  Phase 1 — Projection: match stops → route paths → enrich geopackages
  Phase 2 — Corrections: inspect combined lines, reroute services interactively
  Phase 3 — Plotting: overview plots for study area and catchment area

Usage:
    python services_service_projection.py
"""

import sys
import os
import json
import shutil
from pathlib import Path
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, unary_union, substring
from scipy.spatial import cKDTree

from matplotlib.patches import Rectangle
from matplotlib_map_utils.core.north_arrow import NorthArrow, north_arrow

sys.path.insert(0, str(Path(__file__).parent))
import paths
import settings


# =============================================================================
# Constants
# =============================================================================

SWISS_CRS = "EPSG:2056"

NAME_MATCH_THRESHOLD = 0.85
SPATIAL_MATCH_THRESHOLD = 200  # metres

RAIL_ROUTE_TYPES = {
    0: "tram",
    1: "metro",
    2: "rail",                   # Standard GTFS fallback
    100: "rail",                 # Extended GTFS: Generic Railway Service
    101: "long_distance_rail",    # High Speed Rail Service
    102: "long_distance_rail",    # Long Distance Trains
    103: "inter_regional_rail",   # Inter Regional Rail Service
    106: "regional_rail",         # Regional Rail Service
    109: "sbahn",                # Suburban Railway (S-Bahn)
    400: "metro",
    900: "tram",
    1000: "ferry",
    1300: "cable",
    1400: "funicular",
}

# Mode colours for plotting
MODE_COLOURS = {
    "rail": "#1f3a6e",       # dark blue
    "tram": "#8b0000",       # dark red
    "funicular": "#1a5c2e",  # dark green
    "fallback": "#e07b00",   # orange (straight-line / unmatched)
}

# Feeder background colours for the gazette-with-feeders plot
_BUS_COLOUR  = "#0274F7"   # blue
_SHIP_COLOUR = "#0f0202"   # black
_FEEDER_BG_COLOURS: Dict[str, str] = {
    "bus": _BUS_COLOUR, "express_bus": _BUS_COLOUR, "ondemand_bus": _BUS_COLOUR,
    "ship": _SHIP_COLOUR,
}
_FEEDER_BG_LW = 0.225  # one quarter of tram solid linewidth (0.9 / 4)

# Rail frequency bins: (lo_services, hi_services, hex_colour, linewidth, legend_label)
_FREQ_BINS: List[Tuple[int, int, str, float, str]] = [
    (1, 2, "#91bdd9", 1.0, "1–2"),
    (3, 4, "#4a8bbf", 2.0, "3–4"),
    (5, 8, "#1e5fa3", 3.5, "5–8"),
    (9, 9999, "#0c2d6b", 5.5, "9+"),
]

# Diff-plot bins: (lo_delta, hi_delta, hex_colour, linewidth, legend_label)
# Negative = loss in peak vs off-peak (red); positive = gain (green).
# Delta == 0 is drawn as unchanged (thin grey) and not listed here.
_DIFF_BINS: List[Tuple[int, int, str, float, str]] = [
    (-9999, -5, "#7b241c", 5.5, "≤ −5"),
    (   -4, -3, "#c0392b", 3.5, "−3 to −4"),
    (   -2, -1, "#e74c3c", 2.0, "−1 to −2"),
    (    1,  2, "#1a9850", 2.0, "+1 to +2"),
    (    3,  4, "#006837", 3.5, "+3 to +4"),
    (    5, 9999, "#003d1c", 5.5, "≥ +5"),
]

# Maximum extra travel time (seconds) to still prefer a dead-end terminal hub node
# over a cheaper through-running child.  If the cheapest terminal node costs
# less than this much more than the cheapest candidate overall, the terminal is
# preferred for terminal (first/last) stops at hub stations.
# (~1500 m at 45 km/h ≈ 120 s; set to 300 s for a wider tolerance margin.)
TERMINAL_PREFERENCE_S: int = 300

# Default routing speeds (km/h) used when OSM maxspeed data is absent.
# Mixed-mode segments use the minimum (most conservative) of the applicable values.
RAIL_DEFAULT_SPEED_KMH: int        = 50   # train (all gauges)
TRAM_DEFAULT_SPEED_KMH: int        = 30   # urban tram / light rail
FUNICULAR_DEFAULT_SPEED_KMH: int   = 10   # funicular / Standseilbahn
COG_RAILWAY_DEFAULT_SPEED_KMH: int = 15   # cog railway
BUS_DEFAULT_SPEED_KMH: int         = 30   # urban PT bus feeder

# Canonical mode-speed lookup used as routing graph weight fallback when TT_Stopping
# is absent from segments.gpkg. Conservative values — intentionally below typical operating speeds
# so that segments with missing OSM data are not preferred in routing.
_MODE_DEFAULT_SPEEDS: Dict[str, int] = {
    "train":       RAIL_DEFAULT_SPEED_KMH,
    "tram":        TRAM_DEFAULT_SPEED_KMH,
    "funicular":   FUNICULAR_DEFAULT_SPEED_KMH,
    "cog_railway": COG_RAILWAY_DEFAULT_SPEED_KMH,
    "bus":         BUS_DEFAULT_SPEED_KMH,
}

# Junction-aware physics constants — must match
# infrabuild_network_builder._compute_approx_travel_times and
# infrabuild_infrastructure_enhancement.{_DECEL_A, _BUFFER, _STATION_CLASSES}.
# Update all three together if recalibrating. Used by the per-(service, edge)
# routing weight callable produced by _make_weight_fn.
_DECEL_A:         float        = settings.SERVICE_BRAKE_DECEL_MS2
_BUFFER:          float        = settings.TT_OPERATIONAL_BUFFER
_STATION_CLASSES: frozenset    = frozenset({'station'})

# Layers subject to boundary gateway rerouting (Phase 1.5).
# Services in these layers that start/end outside the buffer are re-routed
# via the nearest confirmed boundary station.
_GATEWAY_LAYERS: frozenset = frozenset({
    "long_distance_rail",
    "inter_regional_rail",
    "regional_rail",
    "sbahn",
})

# =============================================================================
# QGIS project styling constants (mirrored from services_network_builder.py)
# =============================================================================
_RAIL_LINE_TYPES: Dict[int, str] = {
    102: "Long-Distance Rail",
    103: "Inter-Regional Rail",
    106: "Regional Rail",
    109: "S-Bahn / Suburban Rail",
}
_PT_FEEDER_LINE_TYPES: Dict[int, str] = {
    900:  "Tram",
    401:  "Metro",
    700:  "Bus",
    702:  "Express Bus",
    715:  "On-demand Bus",
    1000: "Ship",
    1400: "Funicular",
}
_QGZ_LINE_COLOURS: Dict[int, str] = {
    102:  "#FF0000",   # Long-Distance Rail  — red
    103:  "#FF0000",   # Inter-Regional Rail — red
    106:  "#000000",   # Regional Rail       — black
    109:  "#000000",   # S-Bahn              — black
    401:  "#00246B",   # Metro               — dark blue
    900:  "#FF66CC",   # Tram                — pink
    700:  "#0000FF",   # Bus                 — blue
    702:  "#0000FF",   # Express Bus         — blue
    715:  "#0000FF",   # On-demand Bus       — blue
    1000: "#0099FF",   # Ship                — blue dashed
    1400: "#000000",   # Funicular           — black dashed
}
_QGZ_LINE_STYLE: Dict[int, str] = {
    1000: "dashed",
    1400: "dashed",
}
_QGZ_LAYER_NAMES: Dict[int, str] = {
    102:  "long_distance_rail",
    103:  "inter_regional_rail",
    106:  "regional_rail",
    109:  "sbahn",
    900:  "tram",
    700:  "bus",
    702:  "express_bus",
    715:  "ondemand_bus",
    1000: "ship",
    1400: "funicular",
    401:  "metro",
}
# Reverse: layer name in gpkg → route_type int (for re-splitting concatenated frames)
_LAYER_NAME_TO_RT: Dict[str, int] = {v: k for k, v in _QGZ_LAYER_NAMES.items()}
_RAIL_STOP_FILL    = "#FFFFFF"
_RAIL_STOP_OUTLINE = "#000000"


# =============================================================================
# ZVV Geometry — used as primary source for non-track modes (bus/ship) and
# as post-projection check/fallback for track-based modes
# =============================================================================

ZVV_LINES_GPKG        = os.path.join('data', 'Spatial_Data', 'Transit_Network',
                                     'ZVV_Lines_2026.gpkg')
ZVV_SEGMENT_LAYER     = 'ZVV_LINIEN_L'
ZVV_SBAHN_LAYER       = 'ZVV_S_BAHN_LINIEN_L'
ZVV_MATCH_TOLERANCE_M = 150    # max distance (m) for spatial stop crosswalk
ZVV_DIVERGENCE_THRESHOLD: float = 0.30  # flag when |proj/zvv − 1| exceeds this

SBAHN_SNAP_TOLERANCE_M = 2000
SBAHN_MAX_SINUOSITY    = 3.5
SBAHN_MIN_SINUOSITY    = 0.85
SBAHN_MIN_PART_LENGTH  = 50

# Track-based feeder modes — use BAV routing + ZVV post-pass check/replace
TRACK_BASED_FEEDER_LAYERS: frozenset = frozenset({"tram", "funicular", "metro"})
# Non-track feeder modes — use ZVV directly as geometry source
NON_TRACK_FEEDER_LAYERS: frozenset = frozenset({"bus", "express_bus", "ondemand_bus", "ship"})

# Module-level ZVV geometry state (populated in _run_phase1 before projection)
_zvv_seg_index:    Dict = {}  # (line_name, from_sid, to_sid) → LineString
_zvv_chain_index:  Dict = {}  # (line_name, from_sid) → [(to_sid, geom, seq), ...]
_zvv_sbahn_index:  Dict = {}  # line_name → [LineString, ...]
_zvv_sbahn_jgraph: Dict = {}  # line_name → {junction_id → [(part_idx, other_junction, geom)]}
_zvv_match_counts: Dict = defaultdict(int)

# Module-level stop coordinate dict — built from unprojected feeder segments
stop_coord: Dict[str, Point] = {}  # stop_id → Point(E, N) in EPSG:2056


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MatchResult:
    """Result of matching a service stop to a BAV network node."""
    node_id: Optional[int]         # BAV Betriebspunkt_Nummer; None if unmatched
    confidence: float
    method: str  # 'id' | 'code' | 'name' | 'spatial' | 'raw_bav' | 'unmatched'
    distance_m: Optional[float] = None
    candidates: List[int] = field(default_factory=list)


@dataclass
class ProjectionConfig:
    """All path information needed to run the projection."""
    infra_version: str
    svc_version: str
    infra_dir: Path          # data/Infrastructure/<infra_version>/
    svc_dir: Path            # data/Network/Feeder_Lines/<svc_version>/
    rail_input: Path         # data/Network/Rail_Lines/<svc_version>/rail_segments.gpkg
    rail_output_dir: Path    # data/Network/Rail_Lines/<svc>/<infra>/
    feeder_output_dir: Path  # data/Network/Feeder_Lines/<svc>/<infra>/
    raw_infra_dir: Path      # data/Infrastructure/Raw/
    auto_mode: bool = False             # when True, skip all interactive prompts
    include_feeder_plots: bool = True   # when False, skip _with_feeders plot variant
    include_plots: bool = True          # when False, skip Phase 3 entirely (PLOT_SERVICES=False)


# =============================================================================
# Infrastructure Graph
# =============================================================================

def _default_speed(mode, gauge=None) -> float:
    """Return the default routing speed (km/h) for a segment.

    Prefers transport_mode; falls back to gauge when mode is absent.
    Mixed-mode segments use the minimum (most conservative) value.
    """
    if mode is not None and not (isinstance(mode, float) and np.isnan(mode)):
        mode_str = str(mode).strip()
        if mode_str:
            speeds = [
                _MODE_DEFAULT_SPEEDS.get(m.strip(), RAIL_DEFAULT_SPEED_KMH)
                for m in mode_str.split('/')
                if m.strip()
            ]
            if speeds:
                return float(min(speeds))
    # Gauge fallback for NaN-mode segments
    if gauge is None or (isinstance(gauge, float) and np.isnan(gauge)):
        return float(RAIL_DEFAULT_SPEED_KMH)
    g = int(gauge)
    if g <= 900:
        return float(FUNICULAR_DEFAULT_SPEED_KMH)
    if g == 1000:
        return float(TRAM_DEFAULT_SPEED_KMH)
    return float(RAIL_DEFAULT_SPEED_KMH)


def _build_name_to_id(nodes: gpd.GeoDataFrame) -> Dict[str, int]:
    """
    Build a stable name → Number lookup.

    Nodes that have a valid Number use it directly.  Nodes that
    were manually inserted without one (e.g. synthetic junction nodes added via
    the version manager) receive a synthetic integer >= 9_000_000, assigned in
    DataFrame order so the result is identical across all callers within the same
    run.
    """
    existing_ids: set = {
        int(r["Number"])
        for _, r in nodes.iterrows()
        if pd.notna(r.get("Number"))
    }
    synth_counter = max(existing_ids, default=0)
    synth_counter = max(synth_counter, 9_000_000 - 1)

    name_to_id: Dict[str, int] = {}
    for _, row in nodes.iterrows():
        name = row.get("Name", "")
        if not name or not pd.notna(name):
            continue
        if pd.notna(row.get("Number")):
            name_to_id[name] = int(row["Number"])
        else:
            synth_counter += 1
            while synth_counter in existing_ids:
                synth_counter += 1
            existing_ids.add(synth_counter)
            name_to_id[name] = synth_counter
    return name_to_id



# =============================================================================
# ZVV Geometry helpers — stop_coord builder, geometry loaders, post-pass
# =============================================================================

def _build_stop_coord_from_segments(svc_dir) -> dict:
    """Build stop_id -> Point(E, N) from unprojected feeder segment columns."""
    import fiona as _fiona
    gpkg = svc_dir / "pt_feeder_segments.gpkg"
    coord = {}
    if not gpkg.exists():
        return coord
    for layer in _fiona.listlayers(str(gpkg)):
        gdf = gpd.read_file(gpkg, layer=layer)
        for _, row in gdf.iterrows():
            for id_col, e_col, n_col in [
                ("from_stop_nr", "from_stop_E", "from_stop_N"),
                ("to_stop_nr",   "to_stop_E",   "to_stop_N"),
            ]:
                sid = str(row.get(id_col, "")).strip()
                E = row.get(e_col)
                N = row.get(n_col)
                if sid and pd.notna(E) and pd.notna(N):
                    coord[sid] = Point(float(E), float(N))
    return coord


def _load_zvv_geometry() -> bool:
    """Load ZVV GeoPackage and build geometry lookup indices.

    Uses module-level stop_coord to build a spatial crosswalk between ZVV
    internal stop numbering and GTFS DiDok/UIC stop IDs.
    Populates _zvv_seg_index, _zvv_chain_index, _zvv_sbahn_index.
    Returns True on success, False on failure.
    """
    global _zvv_seg_index, _zvv_chain_index, _zvv_sbahn_index

    zvv_path = os.path.join(paths.MAIN, ZVV_LINES_GPKG)
    if not os.path.isfile(zvv_path):
        print(f"  WARNING: ZVV GeoPackage not found at {zvv_path}")
        print("           Falling back to straight-line geometry.")
        return False

    print(f"  Loading {ZVV_SEGMENT_LAYER} ...", end=" ", flush=True)
    zvv_seg = gpd.read_file(zvv_path, layer=ZVV_SEGMENT_LAYER)
    print(f"{len(zvv_seg):,} segments")

    print(f"  Loading {ZVV_SBAHN_LAYER} ...", end=" ", flush=True)
    zvv_sbahn = gpd.read_file(zvv_path, layer=ZVV_SBAHN_LAYER)
    print(f"{len(zvv_sbahn):,} lines")

    print("  Building spatial stop crosswalk (ZVV -> GTFS) ...")

    zvv_stop_obs = defaultdict(list)
    for _, row in zvv_seg.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        coords = list(row.geometry.coords)
        zvv_stop_obs[int(row["VONHALTESTELLENNR"])].append(coords[0])
        zvv_stop_obs[int(row["BISHALTESTELLENNR"])].append(coords[-1])

    zvv_stops = {}
    for vid, obs in zvv_stop_obs.items():
        xs = [c[0] for c in obs]
        ys = [c[1] for c in obs]
        zvv_stops[vid] = (float(np.median(xs)), float(np.median(ys)))

    gtfs_ids = list(stop_coord.keys())
    if not gtfs_ids:
        print("  WARNING: stop_coord is empty — ZVV crosswalk cannot be built.")
        return False
    gtfs_pts = np.array([(stop_coord[sid].x, stop_coord[sid].y) for sid in gtfs_ids])
    tree = cKDTree(gtfs_pts)

    zvv_ids = list(zvv_stops.keys())
    zvv_pts = np.array([zvv_stops[vid] for vid in zvv_ids])
    dists, idxs = tree.query(zvv_pts)

    crosswalk = {}
    matched_dists = []
    for i, zvv_id in enumerate(zvv_ids):
        if dists[i] <= ZVV_MATCH_TOLERANCE_M:
            crosswalk[zvv_id] = gtfs_ids[idxs[i]]
            matched_dists.append(dists[i])

    med_dist = float(np.median(matched_dists)) if matched_dists else float("nan")
    print(
        f"    {len(zvv_stops):,} ZVV stops -> "
        f"{len(crosswalk):,} matched within {ZVV_MATCH_TOLERANCE_M}m "
        f"(median {med_dist:.0f}m)"
    )

    if not crosswalk:
        print("  WARNING: No stops matched -- ZVV geometry unavailable.")
        return False

    print("  Building ZVV segment index ...")
    seg_index   = {}
    chain_index = defaultdict(list)

    for _, row in zvv_seg.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        line_name = str(row["LINIENNUMMER"]).strip()
        von = int(row["VONHALTESTELLENNR"])
        bis = int(row["BISHALTESTELLENNR"])
        seq_nr = int(row["SEQUENZNR"]) if pd.notna(row.get("SEQUENZNR")) else 0
        gtfs_from = crosswalk.get(von)
        gtfs_to   = crosswalk.get(bis)
        if gtfs_from is None or gtfs_to is None:
            continue
        geom = row.geometry
        key  = (line_name, gtfs_from, gtfs_to)
        if key not in seg_index:
            seg_index[key] = geom
        chain_index[(line_name, gtfs_from)].append((gtfs_to, geom, seq_nr))

    for k in chain_index:
        chain_index[k].sort(key=lambda x: x[2])

    print("  Validating segment geometry directions ...")
    n_correct = n_reversed = n_unknown = 0
    for key, geom in list(seg_index.items()):
        _, from_sid, to_sid = key
        from_pt = stop_coord.get(from_sid)
        to_pt   = stop_coord.get(to_sid)
        if from_pt is None or to_pt is None:
            n_unknown += 1
            continue
        geom_start = Point(geom.coords[0])
        geom_end   = Point(geom.coords[-1])
        dist_ok  = geom_start.distance(from_pt) + geom_end.distance(to_pt)
        dist_rev = geom_start.distance(to_pt)   + geom_end.distance(from_pt)
        if dist_rev < dist_ok:
            seg_index[key] = LineString(list(geom.coords)[::-1])
            n_reversed += 1
        else:
            n_correct += 1
    print(
        f"    Direction: {n_correct:,} correct, "
        f"{n_reversed:,} reversed, {n_unknown:,} unknown"
    )

    for k, entries in chain_index.items():
        new_entries = []
        for to_sid, geom, seq_nr in entries:
            seg_key = (k[0], k[1], to_sid)
            if seg_key in seg_index:
                new_entries.append((to_sid, seg_index[seg_key], seq_nr))
            else:
                new_entries.append((to_sid, geom, seq_nr))
        chain_index[k] = new_entries

    print("  Building S-Bahn line index ...")
    sbahn_index = defaultdict(list)
    for _, row in zvv_sbahn.iterrows():
        name = str(row["LINIESBAHN"]).strip()
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, MultiLineString):
            merged = linemerge(geom)
            if isinstance(merged, MultiLineString):
                for part in merged.geoms:
                    if (part and not part.is_empty
                            and len(part.coords) >= 2
                            and part.length >= SBAHN_MIN_PART_LENGTH):
                        sbahn_index[name].append(part)
            else:
                if merged.length >= SBAHN_MIN_PART_LENGTH:
                    sbahn_index[name].append(merged)
        else:
            if geom.length >= SBAHN_MIN_PART_LENGTH:
                sbahn_index[name].append(geom)

    n_sbahn = len(sbahn_index)
    print(f"    {n_sbahn} S-Bahn lines indexed")

    JUNCTION_TOLERANCE = 100
    jgraph = {}
    n_jgraph = 0
    for name, parts in sbahn_index.items():
        if len(parts) < 2:
            continue
        endpoints = []
        for idx, p in enumerate(parts):
            c = list(p.coords)
            endpoints.append((idx, 0, c[0][0],  c[0][1]))
            endpoints.append((idx, 1, c[-1][0], c[-1][1]))
        ep_pts = np.array([(e[2], e[3]) for e in endpoints])
        junc_ids = list(range(len(endpoints)))
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                if np.hypot(
                    ep_pts[i, 0] - ep_pts[j, 0],
                    ep_pts[i, 1] - ep_pts[j, 1],
                ) < JUNCTION_TOLERANCE:
                    old_id = max(junc_ids[i], junc_ids[j])
                    new_id = min(junc_ids[i], junc_ids[j])
                    for k_i in range(len(junc_ids)):
                        if junc_ids[k_i] == old_id:
                            junc_ids[k_i] = new_id
        graph = defaultdict(list)
        for idx, p in enumerate(parts):
            start_j = junc_ids[idx * 2]
            end_j   = junc_ids[idx * 2 + 1]
            if start_j == end_j:
                continue
            graph[start_j].append((idx, end_j, p))
            graph[end_j].append((idx, start_j, p))
        if graph:
            jgraph[name] = dict(graph)
            n_jgraph += 1
    print(f"    {n_jgraph} S-Bahn junction graphs built")

    _zvv_seg_index.update(seg_index)
    _zvv_chain_index.update(chain_index)
    _zvv_sbahn_index.update(sbahn_index)
    _zvv_sbahn_jgraph.update(jgraph)

    n_lines_matched = len({k[0] for k in seg_index})
    print(
        f"  ZVV geometry ready: {len(seg_index):,} segment mappings "
        f"across {n_lines_matched} lines, {n_sbahn} S-Bahn lines"
    )
    return True


def _chain_zvv_segments(line_name, from_sid, to_sid, max_hops=30):
    """Follow ZVV segment chain from from_sid to to_sid."""
    if not _zvv_chain_index:
        return None
    visited = set()
    current = from_sid
    geoms   = []
    for _ in range(max_hops):
        if current == to_sid:
            break
        next_segs = _zvv_chain_index.get((line_name, current), [])
        if not next_segs:
            return None
        found = False
        for nxt, geom, _ in next_segs:
            if nxt not in visited:
                geoms.append(geom)
                visited.add(current)
                current = nxt
                found = True
                break
        if not found:
            return None
    if current != to_sid or not geoms:
        return None
    coords = []
    for g in geoms:
        c = list(g.coords)
        if coords:
            coords.extend(c[1:])
        else:
            coords.extend(c)
    return LineString(coords) if len(coords) >= 2 else None


def _snap_to_sbahn(from_pt, to_pt, line_name):
    """Project from_pt/to_pt onto an S-Bahn whole-line geometry."""
    geom_list = _zvv_sbahn_index.get(line_name)
    if not geom_list:
        return None
    from_point = Point(from_pt.x, from_pt.y)
    to_point   = Point(to_pt.x, to_pt.y)
    straight_dist = from_point.distance(to_point)
    if straight_dist < 1.0:
        return None
    best_sub   = None
    best_score = float("inf")
    for geom in geom_list:
        if geom is None or geom.is_empty or len(geom.coords) < 2:
            continue
        d_from = geom.distance(from_point)
        d_to   = geom.distance(to_point)
        if d_from > SBAHN_SNAP_TOLERANCE_M or d_to > SBAHN_SNAP_TOLERANCE_M:
            continue
        orig_from_proj = geom.project(from_point)
        orig_to_proj   = geom.project(to_point)
        geom_len       = geom.length
        orig_sub_len   = abs(orig_from_proj - orig_to_proj)
        orig_sinuosity = orig_sub_len / straight_dist
        if orig_sinuosity < SBAHN_MIN_SINUOSITY:
            close_d = min(d_from, d_to)
            both_near_start = orig_from_proj < 1000 and orig_to_proj < 1000
            both_near_end   = (
                orig_from_proj > geom_len - 1000
                and orig_to_proj > geom_len - 1000
            )
            if not (close_d < 100 and (both_near_start or both_near_end)):
                continue
        work_geom   = geom
        work_coords = list(geom.coords)
        for pt, d_pt in [(from_point, d_from), (to_point, d_to)]:
            if d_pt <= 50:
                continue
            proj = geom.project(pt)
            if proj < 50:
                work_coords = [(pt.x, pt.y)] + work_coords
            elif proj > geom_len - 50:
                work_coords = work_coords + [(pt.x, pt.y)]
        if len(work_coords) > len(geom.coords):
            work_geom = LineString(work_coords)
        from_proj = work_geom.project(from_point)
        to_proj   = work_geom.project(to_point)
        if abs(from_proj - to_proj) < 10.0:
            continue
        if from_proj <= to_proj:
            sub = substring(work_geom, from_proj, to_proj)
        else:
            sub = substring(work_geom, to_proj, from_proj)
            sub = LineString(list(sub.coords)[::-1])
        if sub is None or sub.is_empty or len(sub.coords) < 2:
            continue
        sinuosity = sub.length / straight_dist
        if sinuosity > SBAHN_MAX_SINUOSITY:
            continue
        if sub.length < best_score:
            best_score = sub.length
            best_sub   = sub
    return best_sub


def _snap_to_sbahn_chained(from_pt, to_pt, line_name, max_hops=3):
    """Chain multiple S-Bahn geometry parts via junction graph."""
    graph = _zvv_sbahn_jgraph.get(line_name)
    geom_list = _zvv_sbahn_index.get(line_name)
    if not graph or not geom_list or len(geom_list) < 2:
        return None
    from_point = Point(from_pt.x, from_pt.y)
    to_point   = Point(to_pt.x, to_pt.y)
    straight_dist = from_point.distance(to_point)
    if straight_dist < 1.0:
        return None
    from_parts = []
    to_parts   = []
    for idx, geom in enumerate(geom_list):
        d_from = geom.distance(from_point)
        d_to   = geom.distance(to_point)
        if d_from <= SBAHN_SNAP_TOLERANCE_M:
            from_parts.append((idx, geom.project(from_point), d_from))
        if d_to <= SBAHN_SNAP_TOLERANCE_M:
            to_parts.append((idx, geom.project(to_point), d_to))
    if not from_parts or not to_parts:
        return None
    part_juncs = {}
    for junc, neighbours in graph.items():
        for pidx, other_junc, _ in neighbours:
            if pidx not in part_juncs:
                part_juncs[pidx] = (junc, other_junc)
    best_result = None
    best_score  = float("inf")
    for from_idx, from_proj, _ in from_parts:
        for to_idx, to_proj, _ in to_parts:
            if from_idx == to_idx:
                continue
            fj = part_juncs.get(from_idx)
            tj = part_juncs.get(to_idx)
            if fj is None or tj is None:
                continue
            target_juncs = set(tj)
            for start_junc in fj:
                queue = [(start_junc, [])]
                visited = {start_junc}
                found_path = None
                while queue:
                    cur, path = queue.pop(0)
                    if cur in target_juncs and len(path) > 0:
                        found_path = (path, cur)
                        break
                    if len(path) >= max_hops:
                        continue
                    for pidx, nxt_junc, _ in graph.get(cur, []):
                        if nxt_junc not in visited:
                            visited.add(nxt_junc)
                            queue.append((nxt_junc, path + [pidx]))
                if found_path is None:
                    continue
                path_parts, end_junc = found_path
                from_geom = geom_list[from_idx]
                to_geom   = geom_list[to_idx]
                _, from_end_j = part_juncs[from_idx]
                from_fwd = (from_end_j == start_junc)
                if from_fwd:
                    sub_from = substring(from_geom, from_proj, from_geom.length)
                else:
                    sub_from = substring(from_geom, 0, from_proj)
                if (
                    sub_from is None or sub_from.is_empty
                    or sub_from.geom_type != "LineString"
                    or len(sub_from.coords) < 2
                ):
                    continue
                if not from_fwd:
                    sub_from = LineString(list(sub_from.coords)[::-1])
                chain_geoms = [sub_from]
                prev_junc = start_junc
                valid = True
                for pidx in path_parts:
                    p_start_j, p_end_j = part_juncs.get(pidx, (None, None))
                    p_geom = geom_list[pidx]
                    if pidx == from_idx or pidx == to_idx:
                        if pidx == to_idx:
                            break
                        continue
                    if p_start_j == prev_junc:
                        chain_geoms.append(p_geom)
                        prev_junc = p_end_j
                    elif p_end_j == prev_junc:
                        chain_geoms.append(LineString(list(p_geom.coords)[::-1]))
                        prev_junc = p_start_j
                    else:
                        valid = False
                        break
                if not valid:
                    continue
                to_start_j, _ = part_juncs[to_idx]
                to_fwd = (to_start_j == end_junc)
                if to_fwd:
                    sub_to = substring(to_geom, 0, to_proj)
                else:
                    sub_to = substring(to_geom, to_proj, to_geom.length)
                if (
                    sub_to is None or sub_to.is_empty
                    or sub_to.geom_type != "LineString"
                    or len(sub_to.coords) < 2
                ):
                    continue
                if not to_fwd:
                    sub_to = LineString(list(sub_to.coords)[::-1])
                chain_geoms.append(sub_to)
                all_coords = []
                for g in chain_geoms:
                    c = list(g.coords)
                    if all_coords:
                        all_coords.extend(c[1:])
                    else:
                        all_coords.extend(c)
                if len(all_coords) < 2:
                    continue
                result = LineString(all_coords)
                if result.length / straight_dist > SBAHN_MAX_SINUOSITY:
                    continue
                if result.length < best_score:
                    best_score  = result.length
                    best_result = result
    return best_result


def _get_zvv_segment_geom(from_sid: str, to_sid: str, line_name: str,
                          allow_crossline: bool = False):
    """Look up ZVV geometry for a single stop-to-stop segment.

    allow_crossline: try snapping to a *different* S-Bahn line when the
    segment's own line lookup fails.  Restricted to sbahn segments that
    need correction and lie within the catchment buffer — callers set it.
    """
    geom = _zvv_seg_index.get((line_name, from_sid, to_sid))
    if geom is not None:
        _zvv_match_counts["direct"] += 1
        return geom
    chained = _chain_zvv_segments(line_name, from_sid, to_sid)
    if chained is not None:
        _zvv_match_counts["chain"] += 1
        return chained
    if line_name in _zvv_sbahn_index:
        from_pt = stop_coord.get(from_sid)
        to_pt   = stop_coord.get(to_sid)
        if from_pt is not None and to_pt is not None:
            snapped = _snap_to_sbahn(from_pt, to_pt, line_name)
            if snapped is not None:
                _zvv_match_counts["sbahn"] += 1
                return snapped
    if line_name in _zvv_sbahn_jgraph:
        from_pt = stop_coord.get(from_sid)
        to_pt   = stop_coord.get(to_sid)
        if from_pt is not None and to_pt is not None:
            chained_sbahn = _snap_to_sbahn_chained(from_pt, to_pt, line_name)
            if chained_sbahn is not None:
                _zvv_match_counts["sbahn_chain"] += 1
                return chained_sbahn
    if allow_crossline:
        from_pt = stop_coord.get(from_sid)
        to_pt   = stop_coord.get(to_sid)
        if from_pt is not None and to_pt is not None:
            for other_line in _zvv_sbahn_index:
                if other_line == line_name:
                    continue
                snapped = _snap_to_sbahn(from_pt, to_pt, other_line)
                if snapped is not None:
                    _zvv_match_counts["sbahn_crossline"] += 1
                    return snapped
    _zvv_match_counts["fallback"] += 1
    return None


def _print_zvv_match_summary(label: str) -> None:
    """Print and reset ZVV geometry match statistics."""
    total = sum(_zvv_match_counts.values())
    if total == 0:
        return
    parts = []
    for src in ["direct", "chain", "sbahn", "sbahn_chain", "sbahn_crossline", "fallback"]:
        n = _zvv_match_counts.get(src, 0)
        if n > 0:
            parts.append(f"{src} {n:,} ({100 * n / total:.0f}%)")
    print(f"  ZVV [{label}]: {' | '.join(parts)}")
    _zvv_match_counts.clear()


def _apply_zvv_postpass(
    enriched_gdf: "gpd.GeoDataFrame",
    layer_name: str,
    is_track_based: bool,
    buffer_geom=None,
) -> "gpd.GeoDataFrame":
    """Apply ZVV geometry as post-projection check/replacement.

    Track-based (is_track_based=True):
      needs_correction=True rows -> replace with ZVV if found
      good rows -> flag zvv_divergence if |proj/zvv - 1| > ZVV_DIVERGENCE_THRESHOLD

    Non-track (is_track_based=False):
      Apply ZVV directly; keep straight-line where ZVV unavailable.

    sbahn_crossline is enabled only for sbahn segments that need correction
    and whose geometry lies within buffer_geom (or everywhere when buffer_geom
    is None).  All other modes go straight to fallback.
    """
    rows = enriched_gdf.to_dict("records")
    updated = 0
    flagged = 0

    for rec in rows:
        from_sid  = str(rec.get("from_stop_nr", "")).strip()
        to_sid    = str(rec.get("to_stop_nr",   "")).strip()
        line_name = str(rec.get("Service", "")).strip()
        if not from_sid or not to_sid:
            continue

        # Determine the specific mode layer for this record.
        # Rail records carry _source_layer (e.g. "sbahn", "regional_rail");
        # feeder records are identified by layer_name directly.
        seg_layer  = rec.get("_source_layer", layer_name)
        needs_corr = bool(rec.get("needs_correction", False))

        # sbahn_crossline: only sbahn segments that need correction and are
        # within the catchment area buffer.
        if seg_layer == "sbahn" and needs_corr:
            seg_geom = rec.get("geometry")
            in_buf = (
                buffer_geom is None
                or (seg_geom is not None
                    and not seg_geom.is_empty
                    and seg_geom.intersects(buffer_geom))
            )
            allow_crossline = in_buf
        else:
            allow_crossline = False

        zvv_geom = _get_zvv_segment_geom(from_sid, to_sid, line_name,
                                         allow_crossline=allow_crossline)

        if is_track_based:
            if needs_corr and zvv_geom is not None:
                rec["geometry"] = zvv_geom
                rec["needs_correction"] = False
                rec["zvv_source"] = True
                updated += 1
            elif zvv_geom is not None and not needs_corr:
                proj_geom = rec.get("geometry")
                proj_len  = proj_geom.length if proj_geom else 0
                zvv_len   = zvv_geom.length
                if zvv_len > 0 and abs(proj_len / zvv_len - 1) > ZVV_DIVERGENCE_THRESHOLD:
                    rec["zvv_divergence"] = True
                    flagged += 1
        else:
            if zvv_geom is not None:
                rec["geometry"] = zvv_geom
                rec["zvv_source"] = True
                updated += 1

    print(f"  ZVV post-pass [{layer_name}]: {updated} replaced, {flagged} divergence flags")
    _print_zvv_match_summary(layer_name)
    return gpd.GeoDataFrame(rows, crs=SWISS_CRS)



def build_infra_graph(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    raw_nodes: Optional[gpd.GeoDataFrame] = None,
) -> nx.Graph:
    """
    Build an undirected NetworkX graph from BAV nodes and segments.

    Nodes are identified by Betriebspunkt_Nummer (int). Segments connect via from_name/to_name,
    which are resolved to Betriebspunkt_Nummer through a name lookup. Segments whose endpoints
    are not present in nodes are silently skipped, UNLESS `raw_nodes` is provided,
    in which case missing nodes are added dynamically with `node_class='removed'`
    to heal broken graphs and enable continuous routing.

    Edge attributes (no fixed `weight`): the per-(service, edge) routing weight is
    computed at routing time by `_make_weight_fn`. Each edge stores:
      length_m            — physical distance in metres (also used for path_length_m output)
      _cruise_speed_ms    — m/s, derived once via the speed cascade
                            Predominant_Speed → Average_Speed → mode default → 50 km/h
      segment_id, geometry, gauge, electrification — passthrough attributes

    Graph-level attribute `node_classes` (dict: bpnr → class string) lets the
    weight function resolve endpoint classes without indexing edge orientation.

    Node attributes: name, code, E, N, node_class.
    """
    name_to_id = _build_name_to_id(nodes)

    raw_name_lookup = {}
    if raw_nodes is not None:
        for _, row in raw_nodes.iterrows():
            if pd.notna(row.get("Name")) and row["Name"] and pd.notna(row.get("Number")):
                raw_name_lookup[row["Name"]] = row

    G = nx.Graph()

    # Add all active nodes (synthetic-ID nodes included via name_to_id)
    node_id_to_class: Dict[int, str] = {}
    for _, row in nodes.iterrows():
        name = row.get("Name", "")
        nid = name_to_id.get(name) if name else None
        if nid is None:
            continue
        nc = str(row.get("Node_Class", "") or "")
        G.add_node(
            nid,
            name=row.get("Name", ""),
            code=row.get("Code", ""),
            E=float(row.get("E", 0)),
            N=float(row.get("N", 0)),
            node_class=nc,
        )
        node_id_to_class[nid] = nc

    # Add edges from segments, auto-healing missing nodes from raw_nodes
    skipped = 0
    healed = 0
    for _, seg in segments.iterrows():
        fn_name = seg.get("From_Name")
        tn_name = seg.get("To_Name")

        fn = name_to_id.get(fn_name)
        tn = name_to_id.get(tn_name)

        if (fn is None or tn is None) and raw_nodes is not None:
            # Try to heal missing nodes using raw_nodes
            for name, missing_id_var in [(fn_name, 'fn'), (tn_name, 'tn')]:
                if locals()[missing_id_var] is None and name in raw_name_lookup:
                    raw_row = raw_name_lookup[name]
                    raw_id = int(raw_row["Number"])
                    if raw_id not in G:
                        G.add_node(
                            raw_id,
                            name=raw_row.get("Name", ""),
                            code=raw_row.get("Code", ""),
                            E=float(raw_row.get("E", 0)),
                            N=float(raw_row.get("N", 0)),
                            node_class="removed",  # Flag as a healed virtual node
                        )
                    if missing_id_var == 'fn':
                        fn = raw_id
                    else:
                        tn = raw_id
                    healed += 1

        if fn is None or tn is None:
            skipped += 1
            continue

        seg_length_m = float(seg.get("Length", 0))

        # Cruise speed in m/s. Cascade: Average_Speed (length-weighted OSM mean,
        # the harmonically-correct measure for traversal time) → Predominant_Speed
        # (most-common bin, less accurate for time) → mode default → 50 km/h.
        def _to_float(v):
            try:
                f = float(v)
                return f if not pd.isna(f) else None
            except (TypeError, ValueError):
                return None

        spd = _to_float(seg.get("Average_Speed"))
        if not spd or spd <= 0:
            spd = _to_float(seg.get("Predominant_Speed"))
        if not spd or spd <= 0:
            spd = _to_float(_default_speed(seg.get("Transport_Mode"), seg.get("Gauge")))
        if not spd or spd <= 0:
            spd = RAIL_DEFAULT_SPEED_KMH
        cruise_speed_ms = float(spd) / 3.6

        # _cruise_time_s = service-agnostic cruise traversal time in seconds.
        # Used as the static fallback weight in auxiliary lookups (gateway
        # forced flag, BFS approach, candidate stop-node ranking) where we
        # don't yet have a service_stops set. Time-based ranking properly
        # weights short-slow vs long-fast segments — distance ranking would
        # not.
        cruise_time_s = seg_length_m / cruise_speed_ms if cruise_speed_ms > 0 else float('inf')

        G.add_edge(
            fn, tn,
            length_m=seg_length_m,             # physical distance for path_length_m output
            _cruise_speed_ms=cruise_speed_ms,  # m/s; consumed by _make_weight_fn
            _cruise_time_s=cruise_time_s,      # service-agnostic time fallback for aux lookups
            segment_id=seg.get("Segment_ID", ""),
            geometry=seg.geometry,
            gauge=seg.get("Gauge"),
            electrification=seg.get("Electrification_Class"),
        )

    # Graph-level node-class lookup: lets _make_weight_fn classify edge
    # endpoints without indexing orientation in the undirected graph.
    G.graph['node_classes'] = dict(node_id_to_class)

    if healed:
        print(f"  [graph] {healed} missing node endpoints dynamically healed from raw_nodes.")
    if skipped:
        print(f"  [graph] {skipped} segments skipped (from_name or to_name unresolvable).")
    return G


def _is_sentinel_tt(tt_source, travel_time_min) -> bool:
    """True iff this service link's TT must be filled from the path-formula sum.

    Forward-compat with the future-state plan: detect a 'formula' tt_source or
    an explicitly null/zero/sentinel travel_time_min. Today's GTFS pipeline
    never trips this — there is no tt_source column and travel_time_min is
    always a positive number — so behaviour for current data is unchanged.
    Robust to GPKG round-trip (str '<NA>', 'nan', '' all treated as sentinel).
    """
    src = '' if tt_source is None else str(tt_source).strip().lower()
    if src == 'formula':
        return True
    if src in ('', 'nan', 'none', '<na>'):
        # No explicit source — fall back to value-based detection below
        pass
    if travel_time_min is None:
        return True
    if isinstance(travel_time_min, float) and pd.isna(travel_time_min):
        return True
    s = str(travel_time_min).strip().lower()
    if s in ('', 'nan', 'none', '<na>'):
        return True
    try:
        return float(s) <= 0
    except (ValueError, TypeError):
        return True


def _round_half_min(x: float) -> float:
    """Round to nearest 0.1 minutes, floor 0.1 min."""
    return max(0.1, round(x * 10) / 10)


def _make_weight_fn(
    service_stops: set,
    G: nx.Graph,
    a: float = _DECEL_A,
    buffer: float = _BUFFER,
):
    """Return a per-(service, edge) weight callable for nx.shortest_path.

    For each edge (u, v) along a candidate path, returns travel time in seconds:

        weight = (L / v + n_decel · 0.5 · v / a) · buffer

    where n_decel counts the edge's endpoints that are BOTH stations (Node_Class
    in _STATION_CLASSES) AND members of service_stops. Junctions and stations
    the service skips contribute 0 — so junction-junction segments collapse to
    pure cruise, and station endpoints cost half the kinematic decel only when
    the service actually stops there.

    The factory pattern binds service_stops by closure so each routing call
    gets an isolated weight function — avoids late-binding bugs across iterations.

    Args:
        service_stops: BAV Betriebspunkt_Nummer of the service link's two GTFS
                       stops (= the only nodes this service stops at within the link).
        G:             routing graph built by build_infra_graph; needs
                       G.graph['node_classes'] populated.
        a:             deceleration (m/s²).
        buffer:        operational overhead multiplier.

    Returns:
        Callable (u, v, edge_data) -> float (seconds). Returns +inf if the edge
        has no usable cruise speed (defensive against malformed segments).
    """
    node_classes = G.graph.get('node_classes', {})
    stops = set(service_stops or ())

    def _weight(u, v, d):
        v_ms = float(d.get('_cruise_speed_ms', 0) or 0)
        if v_ms <= 0:
            return float('inf')
        L = float(d.get('length_m', 0) or 0)
        u_decel = 1 if (node_classes.get(u, '') in _STATION_CLASSES and u in stops) else 0
        v_decel = 1 if (node_classes.get(v, '') in _STATION_CLASSES and v in stops) else 0
        n_decel = u_decel + v_decel
        return (L / v_ms + n_decel * 0.5 * v_ms / a) * buffer

    return _weight


def build_segment_lookup(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    raw_nodes: Optional[gpd.GeoDataFrame] = None,
) -> Dict[Tuple[int, int], pd.Series]:
    """
    Build {(from_id, to_id): segment_row} dict for fast geometry retrieval.
    Both directions are stored (undirected). Includes healed virtual nodes if `raw_nodes` provided.
    """
    name_to_id = _build_name_to_id(nodes)

    raw_name_lookup = {}
    if raw_nodes is not None:
        raw_name_lookup = {
            row["Name"]: int(row["Number"])
            for _, row in raw_nodes.iterrows()
            if pd.notna(row.get("Name")) and row["Name"] and pd.notna(row.get("Number"))
        }

    lookup: Dict[Tuple[int, int], pd.Series] = {}
    for _, seg in segments.iterrows():
        fn_name = seg.get("From_Name")
        tn_name = seg.get("To_Name")

        fn = name_to_id.get(fn_name)
        tn = name_to_id.get(tn_name)

        if fn is None and raw_nodes is not None:
            fn = raw_name_lookup.get(fn_name)
        if tn is None and raw_nodes is not None:
            tn = raw_name_lookup.get(tn_name)

        if fn is None or tn is None:
            continue
        lookup[(fn, tn)] = seg
        lookup[(tn, fn)] = seg
    return lookup


def build_node_attrs(nodes: gpd.GeoDataFrame) -> Dict[int, Dict]:
    """
    Build {Number: {name, code, E, N, node_class}} dict for fast attribute lookup.
    """
    return {
        int(row["Number"]): {
            "name": row.get("Name", ""),
            "code": row.get("Code", ""),
            "E": float(row.get("E", 0)),
            "N": float(row.get("N", 0)),
            "node_class": row.get("Node_Class", ""),
        }
        for _, row in nodes.iterrows() if pd.notna(row.get("Number"))
    }


# =============================================================================
# Gauge / electrification routing helpers
# =============================================================================

def _node_gauge(node_id: int, G: nx.Graph) -> Optional[int]:
    """Return the single gauge (mm) shared by all edges incident to node_id.
    Returns None when values are mixed, all-null, or node absent."""
    gauges: set = set()
    for _, _, edata in G.edges(node_id, data=True):
        g = edata.get("gauge")
        if g is not None and not (isinstance(g, float) and np.isnan(g)):
            try:
                gauges.add(int(float(g)))
            except (ValueError, TypeError):
                pass
    return next(iter(gauges)) if len(gauges) == 1 else None


def _node_electrification(node_id: int, G: nx.Graph) -> Optional[str]:
    """Return the single electrification shared by all edges incident to node_id.
    Returns None when values are mixed, all-null, or node absent."""
    elecs: set = set()
    for _, _, edata in G.edges(node_id, data=True):
        e = edata.get("electrification")
        if e is not None and not (isinstance(e, float) and np.isnan(e)):
            s = str(e).strip()
            if s:
                elecs.add(s)
    return next(iter(elecs)) if len(elecs) == 1 else None


def _build_gauge_graphs(G: nx.Graph) -> Dict[int, nx.Graph]:
    """Pre-build one gauge-filtered subgraph view per distinct gauge value.

    Each view includes edges whose gauge matches the key plus all edges with
    null/missing gauge (permissive — treated as compatible with any gauge).
    Built once per Phase 1 run; O(1) per routing call thereafter.
    """
    gauge_values: set = set()
    for _, _, edata in G.edges(data=True):
        g = edata.get("gauge")
        if g is not None and not (isinstance(g, float) and np.isnan(g)):
            try:
                gauge_values.add(int(float(g)))
            except (ValueError, TypeError):
                pass

    def _make_filter(target: int):
        def _f(u, v):
            edata = G.edges[u, v]
            g = edata.get("gauge")
            if g is None or (isinstance(g, float) and np.isnan(g)):
                return True  # null gauge: permissive
            try:
                return int(float(g)) == target
            except (ValueError, TypeError):
                return True  # unparseable: permissive
        return _f

    gauge_graphs: Dict[int, nx.Graph] = {
        gv: nx.subgraph_view(G, filter_edge=_make_filter(gv))
        for gv in gauge_values
    }
    if gauge_graphs:
        print(f"  Gauge-filtered graphs: {sorted(gauge_graphs.keys())} mm")
    return gauge_graphs


# =============================================================================
# Hub Topology
# =============================================================================

def _bfs_outlying_stations(
    G: nx.Graph,
    gateway: int,
    cluster: frozenset,
) -> frozenset:
    """
    Branch-exhausting BFS from `gateway` outward through G, excluding all
    cluster nodes.  Returns the frozenset of station-class nodes that are the
    first station encountered on each outward branch.

    Stopping rule per branch: once a station node is found that branch
    terminates — we do not traverse past it.

    Args:
        G:       infrastructure graph
        gateway: starting node (first node outside the hub cluster)
        cluster: frozenset of all hub cluster node IDs (hub + children)

    Returns:
        frozenset of outlying station node IDs (may be empty if no station
        is reachable outward from this gateway).
    """
    if gateway not in G:
        return frozenset()

    node_classes = nx.get_node_attributes(G, "node_class")

    if node_classes.get(gateway) == "station":
        return frozenset({gateway})

    found: set = set()
    visited: set = set(cluster) | {gateway}
    queue: deque = deque([gateway])

    while queue:
        current = queue.popleft()
        for nbr in G.neighbors(current):
            if nbr in visited:
                continue
            visited.add(nbr)
            if node_classes.get(nbr) == "station":
                found.add(nbr)
                # Do not explore past this station on this branch
            else:
                queue.append(nbr)

    return frozenset(found)


def build_hub_topology(
    nodes: gpd.GeoDataFrame,
    G: nx.Graph,
) -> Dict[int, Dict]:
    """
    Precompute hub topology for all hub stations.

    A hub is a station node (node_class='station') that has at least one child
    that is also a station.  For each hub, computes:

      hub_node_type  — 'terminal' or 'through' classification of the hub parent.

      children       — per station child: node_type ('terminal'/'through') and,
                       per gateway, forced-routing flag + set of outlying stations
                       found by branch-exhausting BFS (_bfs_outlying_stations).

      crossing_table — maps frozenset({outlying_a, outlying_b}) to the through
                       child that connects them without backtracking.

      all_outlying   — union of all outlying station sets across the hub cluster
                       (children + hub parent perimeter).

    Returns
    -------
    {
      hub_id: {
        "hub_node_type": "terminal" | "through",
        "children": {
          child_id: {
            "node_type": "terminal" | "through",
            "gateways": {
              gateway_id: {
                "forced": bool,
                "outlying_stations": frozenset[int]
              }
            }
          }
        },
        "crossing_table": {frozenset({a, b}): child_id, ...},
        "all_outlying":   frozenset[int]
      }
    }
    """
    # ── Build parent → [children] mapping from nodes table ───────────────────
    node_uuid_to_bpnr: Dict[str, int] = {}
    for _, row in nodes.iterrows():
        if pd.notna(row.get("Number")) and pd.notna(row.get("Node_ID")):
            node_uuid_to_bpnr[str(row["Node_ID"]).strip()] = int(row["Number"])

    children_by_parent: Dict[int, List[int]] = {}
    node_class_map: Dict[int, str] = {}
    for _, row in nodes.iterrows():
        if pd.isna(row.get("Number")):
            continue
        nid = int(row["Number"])
        node_class_map[nid] = str(row.get("Node_Class", ""))
        pn = row.get("Parent_Node")
        if pd.isna(pn):
            continue
        raw = str(pn).strip()
        if not raw or raw.lower() in ("none", "nan"):
            continue
        try:
            pid = int(float(raw))
        except ValueError:
            pid = node_uuid_to_bpnr.get(raw)
        if pid is not None:
            children_by_parent.setdefault(pid, []).append(nid)

    hub_topology: Dict[int, Dict] = {}

    for hub_id, children in children_by_parent.items():
        if node_class_map.get(hub_id) != "station":
            continue
        station_children = [c for c in children if node_class_map.get(c) == "station"]
        if not station_children:
            continue

        cluster = frozenset([hub_id] + children)

        # ── Hub parent node type (step a) ─────────────────────────────────────
        hub_perimeter = (
            [n for n in G.neighbors(hub_id) if n not in cluster]
            if hub_id in G else []
        )
        hub_node_type = "through" if len(hub_perimeter) >= 2 else "terminal"

        # ── Station children: node type + gateway data (steps a + b) ─────────
        child_data: Dict[int, Dict] = {}
        for c in station_children:
            if c not in G:
                continue
            gateways = [n for n in G.neighbors(c) if n not in cluster]
            if not gateways:
                continue

            node_type = "through" if len(gateways) >= 2 else "terminal"

            gateway_data: Dict[int, Dict] = {}
            for gw in gateways:
                # Forced-routing flag: is there a surface path faster (in
                # service-agnostic cruise time) than the direct edge? If so
                # Dijkstra would bypass the segment under typical service
                # weights, requiring forced_via routing. Uses _cruise_time_s
                # (= length_m / cruise_speed_ms) — the service-agnostic
                # proxy for what Dijkstra prefers, accounting for both
                # distance and speed. The per-(service, edge) callable adds
                # decel terms on top, which rarely flip routing decisions
                # at corridor scale.
                direct_w = G[gw][c].get("_cruise_time_s", 0.0)
                G_tmp = G.copy()
                G_tmp.remove_edge(gw, c)
                try:
                    surface_len = nx.shortest_path_length(G_tmp, gw, c, weight="_cruise_time_s")
                    forced = surface_len < direct_w
                except nx.NetworkXNoPath:
                    forced = False

                gateway_data[gw] = {
                    "forced": forced,
                    "outlying_stations": _bfs_outlying_stations(G, gw, cluster),
                }

            child_data[c] = {"node_type": node_type, "gateways": gateway_data}

        if not child_data:
            continue

        # ── Crossing table + all_outlying (step c) ───────────────────────────
        all_outlying: set = set()
        crossing_table: Dict = {}

        # Collect all gateway node IDs across ALL station children so we can
        # exclude them from through-child outlying when building the crossing
        # table cross-products.  This prevents a through-child from stealing
        # crossing-table pairs that belong to another child's exclusive gateway
        # corridor (e.g. ZLOE claiming Stadelhofen pairs that belong to ZMUS).
        all_child_gateway_nodes: set = set()
        for cdata in child_data.values():
            all_child_gateway_nodes |= set(cdata["gateways"].keys())

        for child_id, cdata in child_data.items():
            # Collect outlying stations from every child (terminal + through)
            for gdata in cdata["gateways"].values():
                all_outlying |= gdata["outlying_stations"]

            if cdata["node_type"] != "through":
                continue

            # Gateway nodes that belong to OTHER children — exclude from this
            # child's cross-product so each child owns its exclusive corridor.
            own_gateways    = set(cdata["gateways"].keys())
            other_gateways  = all_child_gateway_nodes - own_gateways

            gws = list(cdata["gateways"].keys())
            for i in range(len(gws)):
                for j in range(i + 1, len(gws)):
                    os_i = cdata["gateways"][gws[i]]["outlying_stations"] - other_gateways
                    os_j = cdata["gateways"][gws[j]]["outlying_stations"] - other_gateways
                    for o_a in os_i:
                        for o_b in os_j:
                            key = frozenset({o_a, o_b})
                            if key not in crossing_table:  # first match wins
                                crossing_table[key] = child_id

        # Also collect outlying stations reachable via the hub parent's own
        # perimeter edges (hub parent may have direct outside connections)
        for gw in hub_perimeter:
            all_outlying |= _bfs_outlying_stations(G, gw, cluster)

        hub_topology[hub_id] = {
            "hub_node_type": hub_node_type,
            "children": child_data,
            "crossing_table": crossing_table,
            "all_outlying": frozenset(all_outlying),
        }

    # ── Print summary ─────────────────────────────────────────────────────────
    def _hub_name(nid: int) -> str:
        return next(
            (str(r.get("Name", nid)) for _, r in nodes.iterrows()
             if pd.notna(r.get("Number"))
             and int(r["Number"]) == nid),
            str(nid),
        )

    for hub_id, hub_data in hub_topology.items():
        children = hub_data["children"]
        print(
            f"  [hub] {_hub_name(hub_id)} ({hub_id}) [{hub_data['hub_node_type']}]: "
            f"{len(children)} station child(ren), "
            f"{len(hub_data['all_outlying'])} outlying station(s), "
            f"{len(hub_data['crossing_table'])} crossing path(s)"
        )
        for child_id, cdata in children.items():
            print(
                f"       {_hub_name(child_id)} [{cdata['node_type']}]: "
                f"{len(cdata['gateways'])} gateway(s)"
            )
            for gw, gdata in cdata["gateways"].items():
                print(
                    f"         gateway {gw}: forced={gdata['forced']}, "
                    f"outlying={len(gdata['outlying_stations'])} station(s)"
                )

    return hub_topology


# =============================================================================
# Boundary Station Detection
# =============================================================================

def detect_boundary_station_candidates(
    G: nx.Graph,
    nodes: gpd.GeoDataFrame,
    buffer_geom,
    threshold_m: float = 5000,
) -> List[int]:
    """
    Detect candidate boundary stations: leaf station nodes (degree=1 in the
    working graph) whose geometry lies within `threshold_m` metres of the
    catchment area buffer boundary edge.

    These stations mark where rail lines enter the catchment area from outside —
    the BAV network is spatially clipped at the buffer, so through-stations at
    the boundary appear as degree-1 leaf nodes.

    Returns a sorted list of Number (int).
    """
    if buffer_geom is None:
        return []
    buf_boundary = buffer_geom.boundary
    candidates: List[int] = []
    for _, row in nodes.iterrows():
        if pd.isna(row.get("Number")):
            continue
        nid = int(row["Number"])
        if G.nodes.get(nid, {}).get("node_class") != "station":
            continue
        if nid not in G or G.degree(nid) != 1:
            continue
        pt = row.geometry
        if pt is None or pt.is_empty:
            continue
        if pt.distance(buf_boundary) <= threshold_m:
            candidates.append(nid)
    return sorted(candidates)


# =============================================================================
# Boundary Station — Persistence & Confirmation CLI
# =============================================================================

def _save_boundary_stations(boundary_ids: List[int], path: Path) -> None:
    """Persist confirmed boundary station node IDs to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(boundary_ids, f, indent=2)


def _load_boundary_stations(path: Path) -> Optional[List[int]]:
    """Load persisted boundary station node IDs from JSON. Returns None if absent."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_boundary_mapping(mapping: Dict[str, int], path: Path) -> None:
    """Persist stop_id → boundary_node_id mapping to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)


def _load_boundary_mapping(path: Path) -> Optional[Dict[str, int]]:
    """Load persisted stop_id → boundary_node_id mapping. Returns None if absent."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_boundary_station_confirmation_cli(
    candidates: List[int],
    node_attrs: Dict[int, Dict],
) -> List[int]:
    """
    Show the auto-detected boundary station candidates and let the user
    remove false positives by entering comma-separated list indices.

    Returns the confirmed list of node IDs.
    """
    print("\n  Candidate boundary stations (leaf stations ≤5 km from buffer edge):")
    for i, nid in enumerate(candidates, 1):
        name = node_attrs.get(nid, {}).get("name", str(nid))
        print(f"    {i:3}) {name}  (node {nid})")

    print(
        "\n  Enter comma-separated indices to REMOVE (false positives), "
        "or Enter to accept all:"
    )
    raw = input("  Remove: ").strip()
    if not raw:
        print(f"  All {len(candidates)} candidate(s) accepted.")
        return list(candidates)

    to_remove: set = set()
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit() and 1 <= int(part) <= len(candidates):
            to_remove.add(int(part) - 1)

    confirmed = [nid for i, nid in enumerate(candidates) if i not in to_remove]
    print(f"  Confirmed {len(confirmed)} boundary station(s).")
    return confirmed


# =============================================================================
# Stop Matching
# =============================================================================

_STOP_NAME_SUFFIXES = [
    " bahnhof/hb", " bahnhof", " bhf", ", bahnhof",
    " hb", " station", " gare", " stazione",
]


def normalize_stop_name(name: str) -> str:
    """Lowercase and strip common Swiss station name suffixes for fuzzy matching."""
    s = str(name).lower().strip()
    for suffix in _STOP_NAME_SUFFIXES:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def build_stop_lookups(nodes: gpd.GeoDataFrame) -> Dict[str, dict]:
    """
    Build lookup tables for stop-to-node matching.

    Returns:
        {
          'by_id':   {Number (int): Number},
          'by_name': {normalized_name (str): Number},
          'by_code': {Code (str): Number},
          'children':{parent_node_id (int): [child_node_id...]}
        }

    Parent-child relationships are built from the `Parent_Node` column.  The BAV
    geopackage stores parent references as the UUID value found in the `ID`
    column of the parent node — not as a numeric Number.  A first
    pass therefore builds a ID-UUID → Number reverse-lookup so
    that UUID-style Parent_Node values can be resolved to their numeric parent ID.
    """
    # First pass: build ID (UUID) → Number reverse-lookup
    node_uuid_to_bpnr: Dict[str, int] = {}
    for _, row in nodes.iterrows():
        if pd.isna(row.get("Number")):
            continue
        raw_uuid = str(row.get("Node_ID", "")).strip() if pd.notna(row.get("Node_ID")) else ""
        if raw_uuid:
            node_uuid_to_bpnr[raw_uuid] = int(row["Number"])

    # Second pass: build all lookup tables
    by_id: Dict[int, int] = {}
    by_name: Dict[str, int] = {}
    by_code: Dict[str, int] = {}
    children_by_parent: Dict[int, List[int]] = {}

    for _, row in nodes.iterrows():
        if pd.isna(row.get("Number")):
            continue
        nid = int(row["Number"])
        by_id[nid] = nid

        # Build parent-child relationships
        if "Parent_Node" in row and pd.notna(row["Parent_Node"]):
            raw_pid = str(row["Parent_Node"]).strip()
            if raw_pid and raw_pid.lower() not in ["none", "nan"]:
                try:
                    # Numeric Number stored directly
                    pid = int(float(raw_pid))
                    children_by_parent.setdefault(pid, []).append(nid)
                except ValueError:
                    # UUID — resolve via ID reverse-lookup
                    pid = node_uuid_to_bpnr.get(raw_pid)
                    if pid is not None:
                        children_by_parent.setdefault(pid, []).append(nid)

        if pd.notna(row.get("Name")) and row["Name"]:
            by_name[normalize_stop_name(str(row["Name"]))] = nid
        if pd.notna(row.get("Code")) and row["Code"]:
            by_code[str(row["Code"]).strip()] = nid

    return {"by_id": by_id, "by_name": by_name, "by_code": by_code, "children": children_by_parent}


def match_stop_to_node(
    stop_id: str,
    stop_name: str,
    E: float,
    N: float,
    nodes: gpd.GeoDataFrame,
    lookups: Dict[str, dict],
) -> MatchResult:
    """
    Match a service stop to a BAV infrastructure node using a 3-tier strategy.

    Tier 1a — Numeric ID:  strip GTFS prefix, compare to Betriebspunkt_Nummer.
    Tier 1b — CODE match:  stop_id treated as a station code (e.g. 'ZUE').
    Tier 2  — Name match:  SequenceMatcher on normalized names (threshold 0.85).
    Tier 3  — Spatial:     nearest node within 200 m.

    Returns MatchResult with node_id=None and method='unmatched' if all tiers fail.
    """
    sid = str(stop_id).strip()

    def _make_result(nid: int, conf: float, meth: str, dist: Optional[float] = None) -> MatchResult:
        candidates = [nid]
        if nid in lookups.get("children", {}):
            candidates.extend(lookups["children"][nid])
        return MatchResult(node_id=nid, confidence=conf, method=meth, distance_m=dist, candidates=candidates)

    # --- Tier 1a: numeric ID match ---
    try:
        numeric_id = int(sid.split(":")[-1]) if ":" in sid else int(sid)
        if numeric_id in lookups["by_id"]:
            return _make_result(numeric_id, 1.0, "id")
    except ValueError:
        pass

    # --- Tier 1b: exact CODE match (for rail station codes like 'ZUE') ---
    if sid in lookups["by_code"]:
        return _make_result(lookups["by_code"][sid], 1.0, "code")

    # --- Tier 2: fuzzy name match ---
    norm_query = normalize_stop_name(stop_name)
    best_ratio = 0.0
    best_id: Optional[int] = None
    for norm_name, nid in lookups["by_name"].items():
        ratio = SequenceMatcher(None, norm_query, norm_name).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_id = nid
    if best_ratio >= NAME_MATCH_THRESHOLD and best_id is not None:
        return _make_result(best_id, best_ratio, "name")

    # --- Tier 3: spatial proximity ---
    if pd.notna(E) and pd.notna(N):
        stop_pt = Point(float(E), float(N))
        distances = nodes.geometry.distance(stop_pt)
        min_idx = distances.idxmin()
        min_dist = float(distances[min_idx])
        if min_dist <= SPATIAL_MATCH_THRESHOLD:
            return _make_result(
                int(nodes.loc[min_idx, "Number"]),
                0.7 * (1 - min_dist / SPATIAL_MATCH_THRESHOLD),
                "spatial",
                min_dist,
            )

    return MatchResult(node_id=None, confidence=0.0, method="unmatched")


# =============================================================================
# Tier 4 — Raw BAV Fallback (interactive)
# =============================================================================

def _tier4_raw_bav_fallback(
    stop_id: str,
    stop_name: str,
    E: float,
    N: float,
    raw_nodes: gpd.GeoDataFrame,
    working_nodes: gpd.GeoDataFrame,
    working_segments: gpd.GeoDataFrame,
    infra_version_dir: Path,
) -> Tuple[Optional[MatchResult], gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Tier 4 matching: search the raw (pre-filter) BAV nodes for this stop.

    If a match is found in raw data, ask the user whether to add it (and its
    connecting raw segments) to the working infrastructure version.

    Returns:
        (MatchResult or None, updated_working_nodes, updated_working_segments)

    The caller must rebuild the graph + lookups if working_nodes was extended.
    """
    raw_lookups = build_stop_lookups(raw_nodes)
    raw_result = match_stop_to_node(stop_id, stop_name, E, N, raw_nodes, raw_lookups)

    if raw_result.node_id is None:
        return None, working_nodes, working_segments

    # Found in raw — explain and ask user
    raw_row = raw_nodes[raw_nodes["Number"] == raw_result.node_id].iloc[0]
    print(f"\n  [Tier 4] Stop '{stop_name}' not in current version but found in Raw:")
    print(f"    Name:   {raw_row.get('Name', '?')}")
    print(f"    Code:   {raw_row.get('Code', '?')}")
    print(f"    Match:  {raw_result.method}  (confidence {raw_result.confidence:.2f})")

    ans = input(
        f"  Add this node and its connecting segments to the working version? (y/n) [n]: "
    ).strip().lower() or "n"

    if ans != "y":
        print(f"  Skipped — '{stop_name}' will use straight-line geometry.")
        return None, working_nodes, working_segments

    # Load raw segments and find those connecting this node
    raw_segments_path = infra_version_dir.parent / "Raw" / "segments.gpkg"
    if not raw_segments_path.exists():
        print(f"  WARNING: Raw segments not found at {raw_segments_path}. Cannot add.")
        return None, working_nodes, working_segments

    raw_segs = gpd.read_file(raw_segments_path)
    node_name = raw_row["Name"]
    conn_mask = (
        (raw_segs["From_Name"] == node_name) | (raw_segs["To_Name"] == node_name)
    )
    conn_segs = raw_segs[conn_mask]

    # Append node to working_nodes
    new_node_row = raw_row.to_frame().T.reset_index(drop=True)
    new_node_row = gpd.GeoDataFrame(new_node_row, geometry="geometry", crs=SWISS_CRS)
    working_nodes = pd.concat([working_nodes, new_node_row], ignore_index=True)

    # Append connecting segments to working_segments (only those connecting to
    # existing working nodes — avoids dangling chain additions)
    existing_names = set(working_nodes["Name"].dropna().tolist())
    for _, seg in conn_segs.iterrows():
        other_name = (
            seg["To_Name"] if seg["From_Name"] == node_name else seg["From_Name"]
        )
        if other_name in existing_names:
            new_seg = seg.to_frame().T.reset_index(drop=True)
            new_seg_gdf = gpd.GeoDataFrame(new_seg, geometry="geometry", crs=SWISS_CRS)
            working_segments = pd.concat(
                [working_segments, new_seg_gdf], ignore_index=True
            )

    # Persist to disk
    working_nodes.to_file(infra_version_dir / "nodes.gpkg", driver="GPKG")
    working_segments.to_file(infra_version_dir / "segments.gpkg", driver="GPKG")
    print(f"  Added node '{node_name}' to working version. Infrastructure files updated.")

    final_result = MatchResult(
        node_id=int(raw_result.node_id),
        confidence=raw_result.confidence,
        method="raw_bav",
        distance_m=raw_result.distance_m,
    )
    return final_result, working_nodes, working_segments


# =============================================================================
# Path Routing
# =============================================================================

def route_between_nodes(
    G: nx.Graph,
    nodes_a: List[int],
    nodes_b: List[int],
    seg_lookup: Dict[Tuple[int, int], pd.Series],
    node_attrs: Dict[int, Dict],
    forced_via: Optional[List[int]] = None,
    service_stops: Optional[set] = None,
) -> Tuple[Optional[object], str, str, float, Optional[int], Optional[int], str, float]:
    """
    Find the shortest path between two BAV nodes and return routing metadata.

    Returns:
        (geometry, via_stations_str, via_junctions_str, path_length_m, chosen_a, chosen_b,
         path_nodes_str, path_weight_s)

    path_nodes_str: ';'-joined Betriebspunkt_Nummer values for every node on the path,
        including endpoints. Empty string when no path is found.
    path_weight_s: total weight (= per-(service, edge) travel time in seconds) summed
        along the chosen path. Used by sentinel TT computation in callers.

    'via_stations_str' — ';'-joined NAMEs of intermediate nodes with node_class='station'.
    'via_junctions_str' — ';'-joined NAMEs of all other intermediate nodes.
    Returns (None, '', '', 0.0, None, None, '', 0.0) when no path exists.

    forced_via — optional ordered list of node IDs that must appear on the path.
        The path is stitched as shortest(a, via[0]) + shortest(via[0], via[1]) + …
        + shortest(via[-1], b).  This overrides pure Dijkstra and is used for
        DML through-service routing where the physically longer tunnel must be
        preferred over the shorter surface path.  Falls back to normal Dijkstra
        if any via-node is absent from G or no path exists through the waypoints.

    service_stops — set of two BAV Betriebspunkt_Nummer values for the service link
        being routed (= nodes_a/nodes_b's chosen endpoints). Drives the per-(service,
        edge) weight via _make_weight_fn. When None, falls back to weight='_cruise_time_s'
        (service-agnostic cruise time) — defensive default for callers that haven't
        been updated; never triggers in the current pipeline. Forced-via stitching
        reuses the same weight_fn across all sub-calls.
    """
    best_path = None
    best_length = float('inf')

    # Validate forced_via — all waypoints must be graph nodes
    use_forced_via = bool(forced_via) and all(v in G for v in forced_via)

    # Per-service weight function (one closure for this whole routing call).
    # Defensive fallback when service_stops absent — never triggered today.
    if service_stops:
        weight_fn = _make_weight_fn(service_stops, G)
    else:
        weight_fn = "_cruise_time_s"

    def _path_total_weight(p: List[int]) -> float:
        """Sum the weight callable (or attribute) over consecutive edges."""
        total = 0.0
        for i in range(len(p) - 1):
            d = G[p[i]][p[i + 1]]
            if callable(weight_fn):
                total += float(weight_fn(p[i], p[i + 1], d))
            else:
                total += float(d.get(weight_fn, 0))
        return total

    for a in nodes_a:
        for b in nodes_b:
            if a == b:
                continue
            try:
                if use_forced_via:
                    checkpoints = [a] + list(forced_via) + [b]
                    stitched: List[int] = []
                    for k in range(len(checkpoints) - 1):
                        c_from, c_to = checkpoints[k], checkpoints[k + 1]
                        # Use the direct edge when consecutive waypoints share one —
                        # prevents Dijkstra from substituting a cheaper surface detour
                        # for a physically longer but operationally required segment
                        # (e.g. DML tunnel ZOES↔ZLOE = 5 060 m vs surface ~4 500 m).
                        if G.has_edge(c_from, c_to):
                            sub_path = [c_from, c_to]
                        else:
                            sub_path = nx.shortest_path(G, c_from, c_to, weight=weight_fn)
                        if k == 0:
                            stitched.extend(sub_path)
                        else:
                            stitched.extend(sub_path[1:])  # drop duplicate junction node
                    path: List[int] = stitched
                else:
                    path = nx.shortest_path(G, a, b, weight=weight_fn)
                path_length = _path_total_weight(path)
                if path_length < best_length:
                    best_length = path_length
                    best_path = path

            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

    if not best_path:
        return None, "", "", 0.0, None, None, "", 0.0

    path = best_path
    chosen_a = path[0]
    chosen_b = path[-1]
    # path_weight_s is in seconds when weight_fn is the callable; when falling
    # back to length_m it's metres — callers should ignore the value in that case.
    path_weight_s = best_length if callable(weight_fn) else 0.0

    # Collect segment geometries along path
    geoms = []
    total_length = 0.0
    for i in range(len(path) - 1):
        seg = seg_lookup.get((path[i], path[i + 1]))
        if seg is not None and seg.geometry is not None:
            raw_geom = seg.geometry
            if raw_geom.geom_type == "LineString":
                geoms.append(raw_geom)
            elif hasattr(raw_geom, 'geoms'):
                geoms.extend([g for g in raw_geom.geoms if g.geom_type == "LineString"])
            total_length += float(G[path[i]][path[i + 1]].get("length_m", 0))

    final_geom = linemerge(geoms) if geoms else None

    # Classify intermediate nodes
    via_stations: List[str] = []
    via_junctions: List[str] = []
    for nid in path[1:-1]:
        attrs = node_attrs.get(nid, {})
        name = attrs.get("name", str(nid))
        if attrs.get("node_class") == "station":
            via_stations.append(name)
        else:
            via_junctions.append(name)

    path_nodes_str = ";".join(str(n) for n in path)

    return (
        final_geom,
        ";".join(via_stations),
        ";".join(via_junctions),
        total_length,
        chosen_a,
        chosen_b,
        path_nodes_str,
        path_weight_s,
    )


def _derive_forced_via(
    node_from: Optional[int],
    node_to: Optional[int],
    hub_topology: Dict,
    G: nx.Graph,
) -> Optional[List[int]]:
    """Derive a forced_via waypoint list for a routing call between two nodes.

    For hub children whose direct approach edge is physically longer than the
    surface detour (e.g. the DML tunnel), find the natural gateway the train
    passes through on its surface approach and return it as a forced waypoint
    when that gateway is flagged forced=True.

    Mirrors the inline logic previously in _apply_enrichment so the post-pass
    can reproduce the same forced_via that the original routing call used.

    Returns None when no forced gateway applies.
    """
    if not hub_topology or node_from is None or node_to is None:
        return None

    for child_id, other_id in [(node_from, node_to), (node_to, node_from)]:
        for hub_id, hub_data in hub_topology.items():
            children = hub_data.get("children", {})
            if child_id not in children:
                continue
            gw_entries = children[child_id]["gateways"]

            G_surface = G.copy()
            for gw, gdata in gw_entries.items():
                if gdata.get("forced") and G_surface.has_edge(gw, child_id):
                    G_surface.remove_edge(gw, child_id)

            try:
                surface_path = nx.shortest_path(
                    G_surface, other_id, child_id, weight="length_m"
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                surface_path = []

            natural_gw: Optional[int] = None
            for node in surface_path:
                if node in gw_entries:
                    natural_gw = node
                    break

            if (
                natural_gw is not None
                and gw_entries[natural_gw].get("forced")
                and natural_gw in G
            ):
                return [natural_gw]
            break  # only the first hub containing child_id is considered
    return None


# =============================================================================
# Stop Sequence Builders
# =============================================================================

def build_stop_sequence_rail(
    edges: gpd.GeoDataFrame,
) -> Dict[Tuple[str, str], List[Dict]]:
    """
    Build ordered stop sequences from edges_in_corridor.gpkg.

    Groups edges by (Service, Direction), sorts by 'Link NR', then builds a
    de-duplicated ordered list of stops.

    Each stop dict: {stop_id, stop_name, E, N}
    stop_id = FromCode (e.g. 'WS'), used for matching by code.

    Returns: {(service, direction): [stop_dict, ...]}
    """
    sequences: Dict[Tuple[str, str], List[Dict]] = {}

    for (svc, direction), group in edges.groupby(["Service", "Direction"]):
        group_sorted = group.sort_values("Link NR")
        stops: List[Dict] = []
        seen_ids: set = set()

        for _, row in group_sorted.iterrows():
            for prefix in ("From", "To"):
                sid = str(row.get(f"{prefix}Code", "")).strip()
                sname = str(row.get(f"{prefix}Station", "")).strip()
                E_col = "x_origin" if prefix == "From" else "x_dest"
                N_col = "y_origin" if prefix == "From" else "y_dest"
                E = float(row.get(E_col, 0))
                N = float(row.get(N_col, 0))

                if sid and sid not in seen_ids:
                    seen_ids.add(sid)
                    stops.append(
                        {"stop_id": sid, "stop_name": sname, "E": E, "N": N}
                    )

        if len(stops) >= 2:
            sequences[(str(svc), str(direction))] = stops

    return sequences


def build_stop_sequence_feeder(
    segments: gpd.GeoDataFrame,
) -> Dict[Tuple[str, int, int], List[Dict]]:
    """
    Build ordered stop sequences from pt_feeder_segments.gpkg.

    Groups by (route_id, direction_id, variant_rank) and chains consecutive
    from_stop_id → to_stop_id pairs into a de-duplicated ordered list.

    Each stop dict: {stop_id, stop_name, E, N}

    Returns: {(route_id, direction_id, variant_rank): [stop_dict, ...]}
    """
    sequences: Dict[Tuple[str, int, int], List[Dict]] = {}

    group_cols = ["GTFS_ID", "direction_id", "variant_rank"]
    for key_vals, group in segments.groupby(group_cols):
        key = (str(key_vals[0]), int(key_vals[1]), int(key_vals[2]))
        stops: List[Dict] = []
        seen_ids: set = set()

        for _, row in group.iterrows():
            for prefix in ("from", "to"):
                sid = str(row.get(f"{prefix}_stop_nr", "")).strip()
                sname = str(row.get(f"{prefix}_stop_name", "")).strip()
                E = float(row.get(f"{prefix}_stop_E", 0))
                N = float(row.get(f"{prefix}_stop_N", 0))

                if sid and sid not in seen_ids:
                    seen_ids.add(sid)
                    stops.append(
                        {"stop_id": sid, "stop_name": sname, "E": E, "N": N}
                    )

        if len(stops) >= 2:
            sequences[key] = stops

    return sequences


# =============================================================================
# Service Enrichment
# =============================================================================

_NEW_COLS = [
    "node_id_from", "node_id_to",   # retained in output — needed by infrabuild_infrastructure_enhancement
    "from_code", "to_code",
    "match_method_from", "match_method_to",
    "Via_Nodes", "Via_Segment",
    "Via_Station", "Via_Junction",  # internal; dropped at file write via _OUTPUT_DROP
    "path_length_m", "needs_correction",
    "path_nodes",                   # retained in output — needed by infrabuild_infrastructure_enhancement
    "elec_mismatch",
]

# Rename internal processing column names → final output column names.
# Applied at every write to edges_in_corridor.gpkg so the file on disk
# uses the agreed schema while rail_enriched keeps internal names for
# downstream Phase 1.5 / correction processing.
_OUTPUT_RENAME: Dict[str, str] = {
    "Service":     "GTFS_ID",
    "TrainType":   "Service",
    "Direction":   "direction_id",
    "FromCode":    "from_stop_nr",
    "ToCode":      "to_stop_nr",
    "FromStation": "from_stop_name",
    "ToStation":   "to_stop_name",
    "x_origin":    "from_stop_E",
    "y_origin":    "from_stop_N",
    "x_dest":      "to_stop_E",
    "y_dest":      "to_stop_N",
    "TravelTime":  "TT",
    "InVehWait":   "IVWT",
}

# Columns dropped before writing rail output files.
# path_nodes, node_id_from, node_id_to are retained so infrabuild_infrastructure_enhancement
# can distribute GTFS rail TT across BAV segments.
_OUTPUT_DROP: List[str] = [
    "line_short_name",
    "Via_Station", "Via_Junction",
    "_path_tt_min",  # internal: per-(service, edge) path TT in minutes for sentinel computation
    # Legacy columns from older rail_segments.gpkg input versions — no longer produced by
    # services_network_builder.py but may persist in existing projected files.
    "Peak", "OffPeak", "Capacity", "Speed", "FromGde", "ToGde",
    "NR_x", "NR_y", "Link NR", "FromNode", "ToNode", "Via",
    "FromEnd", "ToEnd", "TotalPeakCapacity", "Frequency", "PeakTrainLength",
]


def _make_via_cols(path_nodes_str: str) -> tuple:
    """Derive Via_Nodes and Via_Segment from a semicolon-separated path string.

    Args:
        path_nodes_str: ';'-joined BAV Betriebspunkt_Nummer integers for every
            node on the routed path including endpoints (as returned by
            route_between_nodes). Empty string when no path is available.

    Returns:
        (via_nodes, via_segment) where:
          via_nodes   — pipe-separated integers for intermediate nodes only
          via_segment — pipe-separated 'FROM-TO' pairs for all consecutive node pairs
    """
    if not path_nodes_str:
        return "", ""
    path = [int(n) for n in path_nodes_str.split(";") if n.strip()]
    via_nodes   = "|".join(str(n) for n in path[1:-1])
    via_segment = "|".join(f"{a}-{b}" for a, b in zip(path[:-1], path[1:]))
    return via_nodes, via_segment


def _parse_path_nodes(s) -> List[int]:
    """Parse a ';'-joined node_id string (from the path_nodes column) into a
    list of ints. Empty / non-string input returns []. Unparseable tokens
    are silently skipped — defensive against GPKG round-trip quirks.
    """
    if not isinstance(s, str) or not s:
        return []
    out: List[int] = []
    for tok in s.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(float(tok)))
        except ValueError:
            pass
    return out


def _to_int_or_none(x) -> Optional[int]:
    """Coerce a value to int, returning None for None/NaN/pd.NA/unparseable.
    Tolerates the float64 round-trip GPKG performs on integer columns.
    """
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _lookup_node_code(node_id, nodes: "gpd.GeoDataFrame") -> str:
    """Return the BAV Code abbreviation for a node Number, or '' if not found."""
    if node_id is None:
        return ""
    match = nodes[nodes["Number"] == node_id]
    if match.empty:
        return ""
    code = match.iloc[0].get("Code", "")
    return str(code) if pd.notna(code) else ""


def _apply_enrichment(
    row_idx,
    from_id: str,
    from_name: str,
    from_E: float,
    from_N: float,
    to_id: str,
    to_name: str,
    to_E: float,
    to_N: float,
    nodes: gpd.GeoDataFrame,
    bav_segments: gpd.GeoDataFrame,
    G: nx.Graph,
    seg_lookup: Dict,
    node_attrs: Dict,
    lookups: Dict,
    buffer_geom,
    match_cache: Dict[str, MatchResult],
    raw_nodes: Optional[gpd.GeoDataFrame],
    raw_segments: Optional[gpd.GeoDataFrame],
    infra_version_dir: Optional[Path],
    stop_overrides: Optional[Dict[str, int]] = None,
    hub_topology: Optional[Dict] = None,
    gauge_graphs: Optional[Dict] = None,
) -> Tuple[Dict, gpd.GeoDataFrame, gpd.GeoDataFrame, nx.Graph, Dict, Dict]:
    """
    Core enrichment logic for a single service link (from-stop → to-stop).

    Returns:
        (enrichment_dict, possibly-updated nodes, segments, G, seg_lookup, match_cache)
    """
    _gauge_graphs = gauge_graphs  # local copy — may be rebuilt after Tier 4
    # Match FROM stop (use cache)
    if from_id not in match_cache:
        r = match_stop_to_node(from_id, from_name, from_E, from_N, nodes, lookups)
        # Tier 4 if unmatched + inside buffer
        if r.method == "unmatched" and raw_nodes is not None:
            from_pt = Point(from_E, from_N)
            if buffer_geom is not None and from_pt.within(buffer_geom):
                r4, nodes, bav_segments = _tier4_raw_bav_fallback(
                    from_id, from_name, from_E, from_N,
                    raw_nodes, nodes, bav_segments, infra_version_dir
                )
                if r4 is not None:
                    r = r4
                    # Rebuild graph + lookups after infra change
                    G = build_infra_graph(nodes, bav_segments, raw_nodes)
                    seg_lookup = build_segment_lookup(nodes, bav_segments, raw_nodes)
                    node_attrs.update(build_node_attrs(nodes))
                    lookups = build_stop_lookups(nodes)
                    if _gauge_graphs is not None:
                        _gauge_graphs = _build_gauge_graphs(G)
        match_cache[from_id] = r
    match_from = match_cache[from_id]

    # Match TO stop (use cache)
    if to_id not in match_cache:
        r = match_stop_to_node(to_id, to_name, to_E, to_N, nodes, lookups)
        if r.method == "unmatched" and raw_nodes is not None:
            to_pt = Point(to_E, to_N)
            if buffer_geom is not None and to_pt.within(buffer_geom):
                r4, nodes, bav_segments = _tier4_raw_bav_fallback(
                    to_id, to_name, to_E, to_N,
                    raw_nodes, nodes, bav_segments, infra_version_dir
                )
                if r4 is not None:
                    r = r4
                    G = build_infra_graph(nodes, bav_segments, raw_nodes)
                    seg_lookup = build_segment_lookup(nodes, bav_segments, raw_nodes)
                    node_attrs.update(build_node_attrs(nodes))
                    lookups = build_stop_lookups(nodes)
        match_cache[to_id] = r
    match_to = match_cache[to_id]

    # Apply per-service pre-selected node overrides: replace candidates with the
    # single pre-chosen child node so routing is consistent across directions.
    if stop_overrides:
        if from_id in stop_overrides:
            forced = stop_overrides[from_id]
            match_from = MatchResult(
                node_id=forced, confidence=match_from.confidence,
                method=match_from.method, candidates=[forced],
            )
        if to_id in stop_overrides:
            forced = stop_overrides[to_id]
            match_to = MatchResult(
                node_id=forced, confidence=match_to.confidence,
                method=match_to.method, candidates=[forced],
            )

    # Decide geometry: route if both matched + both inside buffer; else straight-line
    from_pt = Point(from_E, from_N)
    to_pt = Point(to_E, to_N)
    inside_from = buffer_geom is None or from_pt.within(buffer_geom)
    inside_to = buffer_geom is None or to_pt.within(buffer_geom)

    via_st, via_jn, path_len = "", "", 0.0
    path_nodes_str = ""
    needs_correction = False
    elec_mismatch = False
    path_weight_s = 0.0  # per-(service, edge) total along path; 0 when no routing happened

    node_id_from = match_from.node_id
    node_id_to = match_to.node_id

    if match_from.candidates and match_to.candidates and inside_from and inside_to:
        forced_via = _derive_forced_via(
            match_from.node_id, match_to.node_id, hub_topology or {}, G,
        )

        # Hard constraint: route on gauge-filtered graph when gauge is determinable.
        G_route = G
        expected_gauge = None
        if _gauge_graphs and match_from.node_id is not None:
            expected_gauge = _node_gauge(match_from.node_id, G)
            if expected_gauge is not None and expected_gauge in _gauge_graphs:
                G_route = _gauge_graphs[expected_gauge]

        # Relaxed constraint: expected electrification from from-node incident edges.
        expected_elec = None
        if match_from.node_id is not None:
            expected_elec = _node_electrification(match_from.node_id, G)

        # service_stops drives the per-(service, edge) weight in route_between_nodes:
        # Dijkstra picks chosen_a/chosen_b from these candidates, so we pass the
        # candidate lists as the stop set. Once chosen, only chosen_a and chosen_b
        # will contribute decel under the formula (others are intermediate skips).
        service_stops = set(match_from.candidates) | set(match_to.candidates)
        geom, via_st, via_jn, path_len, chosen_from, chosen_to, path_nodes_str, path_weight_s = route_between_nodes(
            G_route, match_from.candidates, match_to.candidates, seg_lookup, node_attrs,
            forced_via=forced_via,
            service_stops=service_stops,
        )
        if geom is not None:
            node_id_from = chosen_from
            node_id_to = chosen_to
        if geom is None:
            # No path found despite matched nodes
            geom = LineString([(from_E, from_N), (to_E, to_N)])
            path_nodes_str = ""
            needs_correction = True
            path_weight_s = 0.0

        # Electrification post-check along routed path (relaxed — flag only).
        if path_nodes_str and expected_elec is not None:
            node_ids = [int(n) for n in path_nodes_str.split(";") if n.strip()]
            for i in range(len(node_ids) - 1):
                u, v = node_ids[i], node_ids[i + 1]
                try:
                    edata = G.edges[u, v]
                except KeyError:
                    continue
                e = edata.get("electrification")
                if e is not None and not (isinstance(e, float) and np.isnan(e)):
                    if str(e).strip() != expected_elec:
                        elec_mismatch = True
                        break
    elif not inside_from or not inside_to:
        # Outside buffer — straight-line is expected
        geom = LineString([(from_E, from_N), (to_E, to_N)])
    else:
        # Unmatched stop inside buffer — flag for correction
        geom = LineString([(from_E, from_N), (to_E, to_N)])
        needs_correction = match_from.node_id is None or match_to.node_id is None
        if needs_correction:
            print(
                f"  WARNING: Stop unmatched — straight-line fallback "
                f"({from_name or from_id} → {to_name or to_id})"
            )

    _via_nodes, _via_segment = _make_via_cols(path_nodes_str)
    # Path TT in minutes from per-(service, edge) physics — used by callers for
    # sentinel-sourced services to set travel_time_min from the formula sum
    # instead of GTFS. Internal column; dropped at file write.
    path_tt_min = path_weight_s / 60.0 if path_weight_s > 0 else 0.0
    enrichment = {
        # Internal columns (kept for Phase 1.5 rerouting; dropped at file write)
        "node_id_from": node_id_from,
        "node_id_to":   node_id_to,
        "match_method_from": match_from.method,
        "match_method_to":   match_to.method,
        # New derived columns
        "from_code": _lookup_node_code(node_id_from, nodes),
        "to_code":   _lookup_node_code(node_id_to,   nodes),
        "Via_Nodes":    _via_nodes,
        "Via_Segment":  _via_segment,
        # Internal via/path columns kept for Phase 1.5; dropped at file write
        "Via_Station":      via_st,
        "Via_Junction":     via_jn,
        "path_nodes":       path_nodes_str,
        "path_length_m":    path_len,
        "_path_tt_min":     path_tt_min,
        "needs_correction": needs_correction,
        "elec_mismatch":    elec_mismatch,
        "geometry":         geom,
    }
    # Also overwrite FromCode/ToCode with matched BAV Number so the final
    # output (after _OUTPUT_RENAME: FromCode → from_stop_nr) carries the
    # authoritative BAV Number rather than the original GTFS stop integer.
    if node_id_from is not None:
        enrichment["FromCode"] = node_id_from
    if node_id_to is not None:
        enrichment["ToCode"] = node_id_to

    return enrichment, nodes, bav_segments, G, seg_lookup, match_cache


def _get_hub_node_type(c: int, hub_id: int, hub: dict) -> str:
    """
    Return the node_type ('terminal' or 'through') for candidate c within hub.

    If c is the hub parent itself, returns hub['hub_node_type'].
    If c is a known child, returns its recorded node_type.
    Defaults to 'terminal' for any node that is neither the hub parent nor a known child (safe fallback).
    """
    if c == hub_id:
        return hub.get("hub_node_type", "terminal")
    return hub.get("children", {}).get(c, {}).get("node_type", "terminal")


def _is_boundary_terminal(node_id: Optional[int], hub_topology: Dict) -> bool:
    """Return True iff node_id is a terminal hub parent or terminal hub child.

    Used by the inter-leg backtracking post-pass to exempt legitimate
    Kopfbahnhof reversals (e.g. HB parent, Zug) and connector turnarounds
    (e.g. Rehalp / Anschluss FB) from the same-edge veto. Stops not present
    in hub_topology are treated as non-terminal — the veto applies.

    Args:
        node_id: BAV Betriebspunkt_Nummer of the candidate boundary stop.
        hub_topology: Output of build_hub_topology.
    """
    if node_id is None:
        return False
    for hub_id, hub in hub_topology.items():
        if node_id == hub_id or node_id in hub.get("children", {}):
            return _get_hub_node_type(node_id, hub_id, hub) == "terminal"
    return False


def _path_traverses_sibling(
    cheapest_term: int,
    approaching: int,
    hub_id: int,
    hub: Dict,
    G: nx.Graph,
) -> bool:
    """Return True iff the shortest path from cheapest_term to approaching
    traverses any other valid candidate of the same hub (parent or any child
    other than cheapest_term itself).

    Used by _preselect_rail_stop_nodes to override the terminal-preference
    boost when picking the terminal would force the leg to back-haul through
    a through-child anyway (the S-Bahn U at HB → Stadelhofen case).

    Weight: _cruise_time_s (same service-agnostic weight as the surrounding
    pre-selection logic). Returns False on unreachable / missing nodes — the
    caller falls back to the existing TERMINAL_PREFERENCE_S logic.
    """
    if cheapest_term is None or approaching is None:
        return False
    if cheapest_term not in G or approaching not in G:
        return False
    try:
        path = nx.shortest_path(
            G, cheapest_term, approaching, weight="_cruise_time_s"
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return False
    siblings = {hub_id} | set(hub.get("children", {}).keys())
    siblings.discard(cheapest_term)
    for node in path[1:-1]:
        if node in siblings:
            return True
    return False


def _nearest_outlying(
    ref: Optional[int],
    all_outlying: frozenset,
    G: nx.Graph,
) -> Optional[int]:
    """
    Return the node in all_outlying with the shortest graph distance to ref.

    If ref is itself in all_outlying it is returned directly (distance = 0).
    Returns None if ref is None or all_outlying is empty.
    Returns None also if no outlying station is graph-reachable from ref.
    """
    if ref is None or not all_outlying:
        return None
    if ref in all_outlying:
        return ref
    best: Optional[int] = None
    best_d = float("inf")
    # Ranking auxiliary lookup — use _cruise_time_s (service-agnostic time)
    # so short-slow vs long-fast segments are ranked by traversal time, not
    # raw distance.
    for o in all_outlying:
        try:
            d = nx.shortest_path_length(G, ref, o, weight="_cruise_time_s")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            d = float("inf")
        if d < best_d:
            best_d, best = d, o
    return best


def _is_same_gateway(
    child_id: int,
    prev_node: Optional[int],
    next_node: Optional[int],
    hub_children: dict,
    G: nx.Graph,
) -> bool:
    """
    Return True if prev_node and next_node both approach child_id from the
    same physical gateway — i.e. the service would enter and exit via the
    same segment (backtracking).

    Uses a surface-path approach: forced gateway→child edges are removed from
    the graph so that the natural (surface-level) path to child_id is found.
    The first gateway node encountered on that path is the natural gateway for
    that approach direction.  If both prev_node and next_node resolve to the
    same natural gateway, the service is backtracking.

    Falls back to True (safe / conservative) when either node is None, when
    child_id has fewer than two gateways, or when no surface path exists.
    """
    gw_entries = hub_children.get(child_id, {}).get("gateways", {})
    gateways = list(gw_entries.keys())
    if len(gateways) < 2:
        return True  # terminal child — no valid through route

    # Build surface graph: remove direct forced-gateway→child edges so that
    # Dijkstra must use the real track approach rather than shortcuts.
    G_surface = G.copy()
    for gw, gdata in gw_entries.items():
        if gdata.get("forced") and G_surface.has_edge(gw, child_id):
            G_surface.remove_edge(gw, child_id)

    def _natural_gw(ref: Optional[int]) -> Optional[int]:
        if ref is None:
            return None
        # If the stop itself is a gateway (e.g. Stadelhofen as next stop for
        # ZMUS), return it directly.
        if ref in gw_entries:
            return ref
        # Check exclusive BFS outlying membership: if ref appears in exactly one
        # gateway's outlying set, that gateway is the unambiguous approach side
        # (e.g. Oerlikon → ZOES only, Wiedikon → Langstrasse only).
        # This is robust to forced gateways that have no surface path alternative
        # (where removing the forced edge would disconnect them from child_id).
        containing = [gw for gw in gateways
                      if ref in gw_entries[gw].get("outlying_stations", frozenset())]
        if len(containing) == 1:
            return containing[0]
        if len(containing) > 1:
            # Ambiguous (in multiple outlying sets, e.g. Wipkingen reachable from
            # both ZOES and Langstrasse): use surface-path to determine the
            # natural approach direction.
            try:
                path = nx.shortest_path(G_surface, ref, child_id, weight="_cruise_time_s")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None
            for node in path[1:]:
                if node in gw_entries:
                    return node
            return None
        # In zero outlying sets (e.g. Altstetten not reached by any gateway BFS,
        # or Oerlikon not reached when ZOES's BFS is blocked by an intermediate
        # station): fall back to nearest gateway by graph distance on the full G.
        # This was the original working approach and is correct for unambiguous
        # nodes that simply fall outside all BFS-reachable outlying sets.
        best_gw, best_d = None, float("inf")
        for gw in gateways:
            try:
                d = nx.shortest_path_length(G, ref, gw, weight="_cruise_time_s")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if d < best_d:
                best_d, best_gw = d, gw
        return best_gw

    gw_prev = _natural_gw(prev_node)
    gw_next = _natural_gw(next_node)

    if gw_prev is None or gw_next is None:
        return True  # cannot determine direction → assume backtracking (safe)
    return gw_prev == gw_next


def _preselect_rail_stop_nodes(
    edges: gpd.GeoDataFrame,
    G: nx.Graph,
    lookups: Dict,
    nodes: gpd.GeoDataFrame,
    hub_topology: Dict,
) -> Dict[Tuple[str, str], Dict[str, int]]:
    """
    For each (Service, Direction) group, pre-select the best BAV node for every
    stop that has multiple routing candidates (i.e. parent nodes with children).

    Uses hub topology (build_hub_topology) to distinguish:
      - Valid through-running candidates: prev and next stops lie in different
        exclusive approach zones of the candidate child node.
      - Reversal candidates: prev and next both approach from the same side —
        service legitimately reverses at this hub.  In this case terminal nodes
        (dead-end platforms) are preferred over through-running children.

    For terminal stops (first or last in the sequence) the candidate with the
    shortest approach distance is chosen, with a preference boost of
    TERMINAL_PREFERENCE_S for terminal (single-approach) nodes.

    For stops not associated with any hub in hub_topology the original bilateral
    cost minimisation is used unchanged.

    Returns: {(service, direction): {stop_id: forced_node_id}}
    Only stops where the selected node differs from the default match are included.
    """

    def _bilateral(c: int, prev_node: Optional[int], next_node: Optional[int]) -> float:
        # Pre-selection ranks candidate stop nodes by service-agnostic cruise
        # time (_cruise_time_s = length_m / cruise_speed_ms). Time-based
        # ranking avoids favouring short-slow segments over long-fast ones,
        # which raw distance would do.
        cost = 0.0
        for ref in (prev_node, next_node):
            if ref is None or c not in G:
                continue
            try:
                cost += nx.shortest_path_length(G, ref, c, weight="_cruise_time_s")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                cost += 1e9
        return cost

    def _approach_cost(c: int, ref: Optional[int]) -> float:
        if ref is None or c not in G:
            return 1e9
        try:
            return nx.shortest_path_length(G, ref, c, weight="_cruise_time_s")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 1e9

    overrides: Dict[Tuple, Dict[str, int]] = {}

    # Group by variant as well when available so that multi-variant services
    # (e.g. S21 with variant_rank 1 and 2) are not merged into one stop sequence.
    # Merging would cause a variant's branch stops to appear after the terminal,
    # incorrectly promoting the terminal stop to a through-service position.
    has_variant = "variant_rank" in edges.columns
    group_cols = ["Service", "Direction", "variant_rank"] if has_variant else ["Service", "Direction"]

    for group_key, group in edges.groupby(group_cols):
        svc       = group_key[0]
        direction = group_key[1]
        group_sorted = group.sort_values("Link NR") if "Link NR" in group.columns else group

        # Build ordered stop list with coordinates
        stop_list: List[Dict] = []
        seen: set = set()
        for _, row in group_sorted.iterrows():
            for code_col, name_col, e_col, n_col in [
                ("FromCode", "FromStation", "x_origin", "y_origin"),
                ("ToCode",   "ToStation",   "x_dest",   "y_dest"),
            ]:
                sid = str(row.get(code_col, "")).strip()
                if sid and sid not in seen:
                    seen.add(sid)
                    stop_list.append({
                        "stop_id":   sid,
                        "stop_name": str(row.get(name_col, "")),
                        "E":         float(row.get(e_col, 0)),
                        "N":         float(row.get(n_col, 0)),
                    })

        if len(stop_list) < 2:
            continue

        # Quick-match each stop (no Tier 4 — pre-pass only)
        stop_matches: Dict[str, MatchResult] = {}
        for s in stop_list:
            stop_matches[s["stop_id"]] = match_stop_to_node(
                s["stop_id"], s["stop_name"], s["E"], s["N"], nodes, lookups
            )

        # Skip service if no stop has multiple candidates
        if not any(len(r.candidates) > 1 for r in stop_matches.values()):
            continue

        # Greedy left-to-right selection
        stop_node_choice: Dict[str, int] = {}
        for i, s in enumerate(stop_list):
            sid = s["stop_id"]
            r = stop_matches[sid]

            if not r.candidates or r.node_id is None:
                stop_node_choice[sid] = r.node_id
                continue

            if len(r.candidates) == 1:
                stop_node_choice[sid] = r.candidates[0]
                continue

            prev_node = stop_node_choice.get(stop_list[i - 1]["stop_id"]) if i > 0 else None
            next_r    = stop_matches.get(stop_list[i + 1]["stop_id"]) if i < len(stop_list) - 1 else None
            next_node = next_r.node_id if next_r else None

            if prev_node is None and next_node is None:
                stop_node_choice[sid] = r.node_id
                continue

            hub_id = r.node_id  # default match is the parent node
            # i==0 (first stop) is treated as terminating: no prev_node available,
            # so the crossing table lookup would degrade to bilateral anyway, and
            # terminal-node preference gives better results for origin stops.
            is_through = (0 < i < len(stop_list) - 1)

            if hub_id in hub_topology:
                # ── Hub-topology-aware selection (DML logic) ──────────────────
                hub = hub_topology[hub_id]
                valid_cands = [c for c in r.candidates if c is not None and c in G]
                if not valid_cands:
                    stop_node_choice[sid] = r.node_id
                    continue

                if is_through:
                    # Step (d): through service — use crossing table
                    all_outlying    = hub.get("all_outlying", frozenset())
                    crossing_table  = hub.get("crossing_table", {})
                    from_outlying   = _nearest_outlying(prev_node, all_outlying, G)
                    to_outlying     = _nearest_outlying(next_node,  all_outlying, G)

                    # Build crossing key only when the two outlying stations differ
                    key = (
                        frozenset({from_outlying, to_outlying})
                        if from_outlying and to_outlying and from_outlying != to_outlying
                        else None
                    )
                    through_child = crossing_table.get(key) if key else None

                    # One-sided fallback: exactly one of from_outlying / to_outlying
                    # is None because one adjacent stop is outside the study buffer
                    # (e.g. IC1/IC5 approaching from Bern/Aarau or departing to Bern/
                    # Aarau).  Use the resolved side to scan the crossing table.
                    # all_outlying includes hub-parent perimeter stations that never
                    # appear in crossing table keys, so re-resolve against ct_outlying
                    # (the set of stations that actually appear in crossing table keys).
                    # If the scan is ambiguous, prefer the child with a forced gateway —
                    # forced gateways mark DML-type tunnels used exclusively by
                    # long-distance through services.
                    # _skip_backtrack bypasses _is_same_gateway which cannot evaluate
                    # direction when one adjacent node is None.
                    _skip_backtrack = False
                    _one_sided = (
                        through_child is None
                        and (from_outlying is None) != (to_outlying is None)
                    )
                    if _one_sided:
                        ct_children = hub.get("children", {})
                        ct_outlying = frozenset(
                            o for key_ct in crossing_table.keys() for o in key_ct
                        )
                        # Use whichever side IS resolved; re-resolve against ct_outlying
                        ref_node = next_node if from_outlying is None else prev_node
                        resolved_ct = (
                            _nearest_outlying(ref_node, ct_outlying, G)
                            if ct_outlying else None
                        )
                        if resolved_ct is not None:
                            matching = {
                                child for key_ct, child in crossing_table.items()
                                if resolved_ct in key_ct and child in valid_cands
                            }
                            if len(matching) == 1:
                                through_child = next(iter(matching))
                                _skip_backtrack = True
                            elif len(matching) > 1:
                                forced_side = {
                                    c for c in matching
                                    if any(
                                        gdata.get("forced")
                                        for gdata in ct_children.get(c, {}).get("gateways", {}).values()
                                    )
                                }
                                if len(forced_side) == 1:
                                    through_child = next(iter(forced_side))
                                    _skip_backtrack = True

                    if through_child is not None and through_child in valid_cands:
                        # Backtracking check: do prev and next approach from the
                        # same physical gateway of through_child?  Skip when the
                        # child was resolved via forced-gateway fallback — the forced
                        # gateway is physical proof of through-running and
                        # _is_same_gateway would give a false positive with
                        # prev_node=None.
                        children = hub.get("children", {})
                        if not _skip_backtrack and _is_same_gateway(
                            through_child, prev_node, next_node, children, G
                        ):
                            is_through = False  # falls through to terminating branch below
                        else:
                            best = through_child
                    else:
                        # No crossing table match — bilateral cost fallback with
                        # terminal preference.  Pairs with no crossing entry are
                        # typically same-side (both approaching from Langstrasse,
                        # etc.) and would backtrack through any through-running
                        # child.  Apply TERMINAL_PREFERENCE_S tolerance so a
                        # terminal node wins over a through-running child by a
                        # thin cost margin.
                        _blt = lambda c: _bilateral(c, prev_node, next_node)
                        _cheapest_all_b  = min(valid_cands, key=_blt)
                        _term_cands_b    = [
                            c for c in valid_cands
                            if _get_hub_node_type(c, hub_id, hub) == "terminal"
                        ]
                        if _term_cands_b:
                            _cheapest_term_b = min(_term_cands_b, key=_blt)
                            _extra_b = _blt(_cheapest_term_b) - _blt(_cheapest_all_b)
                            best = _cheapest_term_b if _extra_b <= TERMINAL_PREFERENCE_S else _cheapest_all_b
                        else:
                            best = _cheapest_all_b

                # Note: 'if not is_through' (not elif) — is_through may have been
                # mutated to False inside the through-branch (backtracking case).
                if not is_through:
                    # Step (e): terminating service (or backtracking through service)
                    # Prefer terminal child nodes; apply TERMINAL_PREFERENCE_S tolerance.
                    # Override: if picking the terminal would force a back-haul
                    # through a sibling through-child (e.g. S-Bahn U HB --> Stadelhofen),
                    # prefer the through-child directly.
                    approaching    = prev_node if i > 0 else next_node
                    cheapest_all   = min(valid_cands, key=lambda c: _approach_cost(c, approaching))
                    terminal_cands = [
                        c for c in valid_cands
                        if _get_hub_node_type(c, hub_id, hub) == "terminal"
                    ]
                    if terminal_cands:
                        cheapest_term = min(terminal_cands, key=lambda c: _approach_cost(c, approaching))
                        if (
                            cheapest_term != cheapest_all
                            and _path_traverses_sibling(cheapest_term, approaching, hub_id, hub, G)
                        ):
                            best = cheapest_all
                        else:
                            extra = (
                                _approach_cost(cheapest_term, approaching)
                                - _approach_cost(cheapest_all, approaching)
                            )
                            best = cheapest_term if extra <= TERMINAL_PREFERENCE_S else cheapest_all
                    else:
                        best = cheapest_all

            else:
                # ── Standard bilateral cost minimisation (non-hub stops) ──────
                best = r.node_id
                best_cost = float("inf")
                for candidate in r.candidates:
                    if candidate is None or candidate not in G:
                        continue
                    cost = _bilateral(candidate, prev_node, next_node)
                    if cost < best_cost:
                        best_cost = cost
                        best = candidate

            stop_node_choice[sid] = best

        # Pin ALL multi-candidate stops to their pre-selected node, even when
        # the pre-selected node happens to equal the default match node_id.
        # Without this, _apply_enrichment passes the full candidate list to
        # route_between_nodes, which picks the cheapest graph distance —
        # e.g. a through-running child 61 m closer than the correct terminal
        # platform — overriding the hub-topology decision made here.
        svc_overrides = {
            sid: chosen
            for sid, chosen in stop_node_choice.items()
            if chosen is not None
            and len(stop_matches[sid].candidates) > 1
        }
        if svc_overrides:
            key = (str(svc), str(direction), str(group_key[2])) if has_variant else (str(svc), str(direction))
            overrides[key] = svc_overrides

    return overrides


def _reroute_leg_with_edge_forbidden(
    row: object,
    forbidden_edge: Tuple[int, int],
    G: nx.Graph,
    seg_lookup: Dict,
    node_attrs: Dict,
    hub_topology: Dict,
    gauge_graphs: Optional[Dict],
) -> Optional[Dict]:
    """Try to reroute one leg between its existing endpoints with one edge removed.

    Returns a dict of new column values on success, or None if no path exists or
    inputs are invalid. The caller decides whether to apply the result (e.g. by
    comparing the resulting total path-length against an alternative reroute).
    """
    node_from = _to_int_or_none(getattr(row, "node_id_from", None))
    node_to   = _to_int_or_none(getattr(row, "node_id_to",   None))
    if node_from is None or node_to is None:
        return None

    # Gauge-filtered graph (mirror _apply_enrichment behaviour)
    G_route = G
    if gauge_graphs:
        eg = _node_gauge(node_from, G)
        if eg is not None and eg in gauge_graphs:
            G_route = gauge_graphs[eg]

    u, v = forbidden_edge
    G_restricted = G_route.copy()
    if G_restricted.has_edge(u, v):
        G_restricted.remove_edge(u, v)

    forced_via = _derive_forced_via(node_from, node_to, hub_topology, G)

    geom, via_st, via_jn, path_len, _, _, pns, pw = route_between_nodes(
        G_restricted, [node_from], [node_to],
        seg_lookup, node_attrs,
        forced_via=forced_via,
        service_stops={node_from, node_to},
    )
    if geom is None or not pns:
        return None

    via_nodes, via_segment = _make_via_cols(pns)
    return {
        "geometry":      geom,
        "Via_Station":   via_st,
        "Via_Junction":  via_jn,
        "path_nodes":    pns,
        "Via_Nodes":     via_nodes,
        "Via_Segment":   via_segment,
        "path_length_m": path_len,
        "_path_tt_min":  pw / 60.0 if pw > 0 else 0.0,
    }


def _apply_reroute_to_row(
    enriched: gpd.GeoDataFrame, idx, result: Dict,
) -> None:
    """Apply a reroute result dict to a single row of the enriched GeoDataFrame.

    Also refreshes TravelTime / tt_source when the row was filled by the
    sentinel-TT branch (forward-compat; inert today).
    """
    for col, val in result.items():
        enriched.at[idx, col] = val
    if "tt_source" in enriched.columns and "TravelTime" in enriched.columns:
        if _is_sentinel_tt(
            enriched.at[idx, "tt_source"],
            enriched.at[idx, "TravelTime"],
        ):
            tt_min_new = result.get("_path_tt_min", 0.0)
            if tt_min_new > 0:
                enriched.at[idx, "TravelTime"] = _round_half_min(tt_min_new)
                enriched.at[idx, "tt_source"]  = "formula"


def _postprocess_inter_leg_backtracking(
    enriched: gpd.GeoDataFrame,
    G: nx.Graph,
    seg_lookup: Dict,
    node_attrs: Dict,
    hub_topology: Dict,
    gauge_graphs: Optional[Dict] = None,
) -> gpd.GeoDataFrame:
    """Fix 1 post-pass: veto inter-leg same-edge re-use (symmetric).

    Iterates each (Service, Direction, variant_rank) group, chains legs by
    matching FromCode/ToCode, and for every transition where leg N+1's first
    edge equals leg N's last edge (undirected), tries BOTH reroute options —
    re-routing leg N+1 with the conflict edge removed, OR re-routing leg N
    with the conflict edge removed — and applies whichever yields the shorter
    sum of the two legs' path-lengths. Tie → reroute leg N+1 (preserves
    backward-compat with the original asymmetric behaviour).

    Terminal-type boundary stops (per build_hub_topology) are exempt — the
    veto skips them so legitimate Kopfbahnhof reversals (HB, Zug, Rehalp
    connector) are preserved.

    When neither reroute yields a path, the original rows are kept unchanged
    (per spec: "allow when necessary").

    Updates the rerouted row's geometry, Via_Station, Via_Junction, path_nodes,
    Via_Nodes, Via_Segment, path_length_m, _path_tt_min. When the row was
    filled by the sentinel-TT branch (tt_source='formula' or sentinel
    TravelTime), TravelTime and tt_source are also refreshed.

    path_nodes is re-read from the GeoDataFrame on each iteration so that
    cascading conflicts (a reroute in transition N→N+1 affecting transition
    N+1→N+2) are detected correctly.
    """
    has_variant = "variant_rank" in enriched.columns
    group_cols = (
        ["Service", "Direction", "variant_rank"]
        if has_variant else ["Service", "Direction"]
    )

    rewrites_next = 0   # leg N+1 rerouted (original asymmetric behaviour)
    rewrites_prev = 0   # leg N rerouted (new symmetric option)
    fallbacks = 0
    skipped_exempt = 0

    for _, grp in enriched.groupby(group_cols, dropna=False):
        # Chain legs by FromCode -> ToCode (BAV Numbers, set by _apply_enrichment)
        rows = list(grp.itertuples())
        by_from: Dict[int, object] = {}
        for r in rows:
            f = _to_int_or_none(getattr(r, "FromCode", None))
            if f is not None and f not in by_from:
                by_from[f] = r
        tos = {_to_int_or_none(getattr(r, "ToCode", None)) for r in rows}
        froms = set(by_from.keys())
        starts = (froms - tos) or ({next(iter(by_from))} if by_from else set())

        for start in starts:
            ordered: List = []
            cur, seen = start, set()
            while cur in by_from and cur not in seen:
                seen.add(cur)
                r = by_from[cur]
                ordered.append(r)
                cur = _to_int_or_none(getattr(r, "ToCode", None))

            for i in range(len(ordered) - 1):
                row_a, row_b = ordered[i], ordered[i + 1]
                # Re-read paths from enriched so that a prior iteration's
                # rewrite is visible to this iteration's conflict check.
                p1 = _parse_path_nodes(enriched.at[row_a.Index, "path_nodes"])
                p2 = _parse_path_nodes(enriched.at[row_b.Index, "path_nodes"])
                if len(p1) < 2 or len(p2) < 2:
                    continue
                # Undirected same-edge check: leg_{N+1}'s first edge = leg_N's last edge
                if not (p2[0] == p1[-1] and p2[1] == p1[-2]):
                    continue

                boundary = p1[-1]
                if _is_boundary_terminal(boundary, hub_topology):
                    skipped_exempt += 1
                    continue  # legit reversal — keep original

                forbidden = (p1[-2], p1[-1])
                len_a_orig = float(enriched.at[row_a.Index, "path_length_m"] or 0.0)
                len_b_orig = float(enriched.at[row_b.Index, "path_length_m"] or 0.0)

                # Symmetric reroute: try both sides, pick the smaller total length.
                result_next = _reroute_leg_with_edge_forbidden(
                    row_b, forbidden, G, seg_lookup, node_attrs,
                    hub_topology, gauge_graphs,
                )
                result_prev = _reroute_leg_with_edge_forbidden(
                    row_a, forbidden, G, seg_lookup, node_attrs,
                    hub_topology, gauge_graphs,
                )

                total_next = (
                    len_a_orig + result_next["path_length_m"]
                    if result_next else float("inf")
                )
                total_prev = (
                    result_prev["path_length_m"] + len_b_orig
                    if result_prev else float("inf")
                )

                if total_next == float("inf") and total_prev == float("inf"):
                    fallbacks += 1
                    continue

                # Tie → reroute leg N+1 (preserves original asymmetric default).
                if total_next <= total_prev:
                    _apply_reroute_to_row(enriched, row_b.Index, result_next)
                    rewrites_next += 1
                else:
                    _apply_reroute_to_row(enriched, row_a.Index, result_prev)
                    rewrites_prev += 1

    rewrites_total = rewrites_next + rewrites_prev
    if rewrites_total or fallbacks or skipped_exempt:
        print(
            f"  Inter-leg backtracking pass: {rewrites_total} rerouted "
            f"({rewrites_prev} leg N, {rewrites_next} leg N+1), "
            f"{fallbacks} fallback (no alt path), "
            f"{skipped_exempt} exempt (terminal boundary)."
        )
    return enriched


def enrich_rail_links(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    bav_segments: gpd.GeoDataFrame,
    G: nx.Graph,
    seg_lookup: Dict,
    node_attrs: Dict,
    lookups: Dict,
    buffer_geom,
    raw_nodes: Optional[gpd.GeoDataFrame],
    raw_segments: Optional[gpd.GeoDataFrame],
    infra_version_dir: Optional[Path],
    hub_topology: Optional[Dict] = None,
    gauge_graphs: Optional[Dict] = None,
) -> gpd.GeoDataFrame:
    """
    Enrich edges_in_corridor.gpkg with real infrastructure geometry.
    Returns the input GeoDataFrame with new columns appended and geometry replaced.
    """
    stop_overrides_by_service = _preselect_rail_stop_nodes(
        edges, G, lookups, nodes, hub_topology or {}
    )

    enriched_rows = []
    match_cache: Dict[str, MatchResult] = {}
    _has_variant = "variant_rank" in edges.columns

    for idx, row in edges.iterrows():
        if _has_variant:
            svc_key = (
                str(row.get("Service", "")),
                str(row.get("Direction", "")),
                str(row.get("variant_rank", "")),
            )
        else:
            svc_key = (str(row.get("Service", "")), str(row.get("Direction", "")))
        stop_overrides = stop_overrides_by_service.get(svc_key, {})

        enrichment, nodes, bav_segments, G, seg_lookup, match_cache = _apply_enrichment(
            idx,
            str(row.get("FromCode", "")), str(row.get("FromStation", "")),
            float(row.get("x_origin", 0)), float(row.get("y_origin", 0)),
            str(row.get("ToCode", "")), str(row.get("ToStation", "")),
            float(row.get("x_dest", 0)), float(row.get("y_dest", 0)),
            nodes, bav_segments, G, seg_lookup, node_attrs, lookups,
            buffer_geom, match_cache, raw_nodes, raw_segments, infra_version_dir,
            stop_overrides=stop_overrides,
            hub_topology=hub_topology,
            gauge_graphs=gauge_graphs,
        )
        new_row = row.to_dict()
        new_row.update(enrichment)
        # Sentinel TT: future-state services with tt_source='formula' (or
        # explicit null travel_time) get their TravelTime filled from the
        # per-(service, edge) path-formula sum. Inert today (no such rows).
        if _is_sentinel_tt(new_row.get('tt_source'), new_row.get('TravelTime')):
            tt_min = enrichment.get('_path_tt_min', 0.0)
            if tt_min > 0:
                new_row['TravelTime'] = _round_half_min(tt_min)
                new_row['tt_source']  = 'formula'
        enriched_rows.append(new_row)

    result = gpd.GeoDataFrame(enriched_rows, crs=SWISS_CRS)
    result = _postprocess_inter_leg_backtracking(
        result, G, seg_lookup, node_attrs, hub_topology or {}, gauge_graphs,
    )
    return result


def enrich_feeder_segments(
    segments: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    bav_segments: gpd.GeoDataFrame,
    G: nx.Graph,
    seg_lookup: Dict,
    node_attrs: Dict,
    lookups: Dict,
    buffer_geom,
    raw_nodes: Optional[gpd.GeoDataFrame],
    raw_segs: Optional[gpd.GeoDataFrame],
    infra_version_dir: Optional[Path],
    hub_topology: Optional[Dict] = None,
    gauge_graphs: Optional[Dict] = None,
) -> gpd.GeoDataFrame:
    """
    Enrich pt_feeder_segments.gpkg with real infrastructure geometry.
    Returns the input GeoDataFrame with new columns appended and geometry replaced.
    """
    enriched_rows = []
    match_cache: Dict[str, MatchResult] = {}

    for idx, row in segments.iterrows():
        enrichment, nodes, bav_segments, G, seg_lookup, match_cache = _apply_enrichment(
            idx,
            str(row.get("from_stop_nr", "")), str(row.get("from_stop_name", "")),
            float(row.get("from_stop_E", 0)), float(row.get("from_stop_N", 0)),
            str(row.get("to_stop_nr", "")), str(row.get("to_stop_name", "")),
            float(row.get("to_stop_E", 0)), float(row.get("to_stop_N", 0)),
            nodes, bav_segments, G, seg_lookup, node_attrs, lookups,
            buffer_geom, match_cache, raw_nodes, raw_segs, infra_version_dir,
            hub_topology=hub_topology,
            gauge_graphs=gauge_graphs,
        )
        new_row = row.to_dict()
        new_row.update(enrichment)
        # Sentinel TT: see enrich_rail_links for rationale. Feeder uses
        # 'travel_time_min' as its column name (no _OUTPUT_RENAME for feeders).
        if _is_sentinel_tt(new_row.get('tt_source'), new_row.get('travel_time_min')):
            tt_min = enrichment.get('_path_tt_min', 0.0)
            if tt_min > 0:
                new_row['travel_time_min'] = _round_half_min(tt_min)
                new_row['tt_source']       = 'formula'
        enriched_rows.append(new_row)

    result = gpd.GeoDataFrame(enriched_rows, crs=SWISS_CRS)
    return result

# =============================================================================
# QGIS Project (.qgz) Helpers
# =============================================================================
# Functions and templates below are the minimal subset needed to produce styled
# QGIS project files.  They mirror the implementation in services_network_builder.py
# so both scripts produce visually identical projects.

import zipfile as _zipfile

_QGIS_VERSION = "3.44.9-Solothurn"

_QGZ_SRS_BLOCK = """<spatialrefsys nativeFormat="wkt">
      <proj4>+proj=somerc +lat_0=46.9524055555556 +lon_0=7.43958333333333 +k_0=1 +x_0=2600000 +y_0=1200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs</proj4>
      <srsid>47</srsid>
      <srid>2056</srid>
      <authid>EPSG:2056</authid>
      <description>CH1903+ / LV95</description>
      <projectionacronym>somerc</projectionacronym>
      <ellipsoidacronym>bessel</ellipsoidacronym>
    </spatialrefsys>"""

_QGZ_WMS_LAYER_ID   = "Swisstopo_National_Map__grey__e16b0296_87b7_4e32_b8e8_b46b5990275e"
_QGZ_WMS_LAYER_NAME = "Swisstopo National Map (grey)"
_QGZ_WMS_SOURCE     = (
    "contextualWMSLegend=0&amp;crs=EPSG:2056&amp;dpiMode=7"
    "&amp;featureCount=10&amp;format=image/png"
    "&amp;layers=ch.swisstopo.pixelkarte-grau"
    "&amp;styles=&amp;url=http://wms.geo.admin.ch/"
)


def _qgz_hex_to_rgba(hex_colour: str, alpha: int = 255) -> str:
    """Convert '#RRGGBB' to QGIS RGBA string 'R,G,B,A'."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{r},{g},{b},{alpha}"


def _qgz_line_maplayer(layer_id, gpkg_relpath, layer_name, display_name, rgba, line_style, width="0.5"):
    pen = "dash" if line_style == "dashed" else "solid"
    return f"""  <maplayer geometry="Line" type="vector" hasScaleBasedVisibilityFlag="0">
    <id>{layer_id}</id>
    <datasource>{gpkg_relpath}|layername={layer_name}</datasource>
    <layername>{display_name}</layername>
    <provider encoding="UTF-8">ogr</provider>
    <srs>{_QGZ_SRS_BLOCK}</srs>
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


def _qgz_marker_maplayer(layer_id, gpkg_relpath, layer_name, display_name, fill_rgba, outline_rgba, size="2", outline_width="0.2"):
    return f"""  <maplayer geometry="Point" type="vector" hasScaleBasedVisibilityFlag="0">
    <id>{layer_id}</id>
    <datasource>{gpkg_relpath}|layername={layer_name}</datasource>
    <layername>{display_name}</layername>
    <provider encoding="UTF-8">ogr</provider>
    <srs>{_QGZ_SRS_BLOCK}</srs>
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


def _qgz_wms_maplayer():
    return f"""  <maplayer type="raster" hasScaleBasedVisibilityFlag="0">
    <id>{_QGZ_WMS_LAYER_ID}</id>
    <datasource>{_QGZ_WMS_SOURCE}</datasource>
    <layername>{_QGZ_WMS_LAYER_NAME}</layername>
    <provider encoding="">wms</provider>
    <srs>{_QGZ_SRS_BLOCK}</srs>
  </maplayer>"""


def _build_qgz(qgz_path: str, layers: List[dict]) -> None:
    """Write a QGIS .qgz project file.

    Parameters
    ----------
    qgz_path : str
        Output path for the .qgz file.
    layers : list of dict
        Each dict: layer_id, gpkg_relpath, layer_name, display_name,
        geom_type ('line'|'point'), colour (hex), line_style, fill_colour,
        outline_colour.  Layers are listed top-to-bottom in the legend.
    """
    tree_entries: List[str] = []
    maplayer_blocks: List[str] = []

    for lyr in layers:
        lid  = lyr["layer_id"]
        src  = f"{lyr['gpkg_relpath']}|layername={lyr['layer_name']}"
        name = lyr["display_name"]
        tree_entries.append(
            f'    <layer-tree-layer id="{lid}" name="{name}" '
            f'checked="Qt::Checked" expanded="1" source="{src}" providerKey="ogr"/>'
        )
        if lyr["geom_type"] == "line":
            rgba = _qgz_hex_to_rgba(lyr["colour"])
            maplayer_blocks.append(
                _qgz_line_maplayer(lid, lyr["gpkg_relpath"], lyr["layer_name"],
                                   name, rgba, lyr.get("line_style", "solid"))
            )
        else:
            fill_rgba    = _qgz_hex_to_rgba(lyr["fill_colour"])
            outline_rgba = _qgz_hex_to_rgba(lyr["outline_colour"])
            maplayer_blocks.append(
                _qgz_marker_maplayer(lid, lyr["gpkg_relpath"], lyr["layer_name"],
                                     name, fill_rgba, outline_rgba)
            )

    # Swisstopo WMS — always at the bottom of the layer tree
    tree_entries.append(
        f'    <layer-tree-layer id="{_QGZ_WMS_LAYER_ID}" name="{_QGZ_WMS_LAYER_NAME}" '
        f'checked="Qt::Checked" expanded="0" source="{_QGZ_WMS_SOURCE}" providerKey="wms"/>'
    )
    maplayer_blocks.append(_qgz_wms_maplayer())

    tree_xml   = "\n".join(tree_entries)
    layers_xml = "\n".join(maplayer_blocks)

    qgs = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis projectname="Network Build" version="{_QGIS_VERSION}">
  <homePath path=""/>
  <title>Network Build</title>
  <autotransaction active="0"/>
  <evaluateDefaultValues active="0"/>
  <trust active="0"/>
  <projectCrs>
    {_QGZ_SRS_BLOCK}
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
      {_QGZ_SRS_BLOCK}
    </destinationsrs>
  </mapcanvas>
</qgis>
"""
    with _zipfile.ZipFile(qgz_path, "w", _zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.qgs", qgs)


def _collect_qgz_line_layers(
    by_type: Dict[int, "gpd.GeoDataFrame"],
    gpkg_relpath: str,
    label_map: Dict[int, str],
    suffix: str = "Segments",
) -> List[dict]:
    """Build line layer descriptor list for _build_qgz from a route_type→GDF dict."""
    layers = []
    counter = 0
    for rt, gdf in sorted(by_type.items()):
        if gdf is None or gdf.empty:
            continue
        layer_name = _QGZ_LAYER_NAMES.get(rt, f"type_{rt}")
        counter += 1
        layers.append({
            "layer_id":     f"{layer_name}_seg_{counter:04d}",
            "gpkg_relpath": gpkg_relpath,
            "layer_name":   layer_name,
            "display_name": f"{label_map.get(rt, layer_name)} {suffix}",
            "geom_type":    "line",
            "colour":       _QGZ_LINE_COLOURS.get(rt, "#888888"),
            "line_style":   _QGZ_LINE_STYLE.get(rt, "solid"),
        })
    return layers


def _collect_qgz_stop_layers(
    by_type: Dict[int, "gpd.GeoDataFrame"],
    gpkg_relpath: str,
    label_map: Dict[int, str],
    is_rail: bool = False,
) -> List[dict]:
    """Build point layer descriptor list for _build_qgz from a route_type→GDF dict."""
    layers = []
    counter = 0
    for rt, gdf in sorted(by_type.items()):
        if gdf is None or gdf.empty:
            continue
        layer_name = _QGZ_LAYER_NAMES.get(rt, f"type_{rt}")
        counter += 1
        if is_rail:
            fill, outline = _RAIL_STOP_FILL, _RAIL_STOP_OUTLINE
        else:
            fill, outline = _QGZ_LINE_COLOURS.get(rt, "#888888"), "#000000"
        layers.append({
            "layer_id":       f"{layer_name}_stops_{counter:04d}",
            "gpkg_relpath":   gpkg_relpath,
            "layer_name":     layer_name,
            "display_name":   f"{label_map.get(rt, layer_name)} Stops",
            "geom_type":      "point",
            "fill_colour":    fill,
            "outline_colour": outline,
        })
    return layers


# =============================================================================
# Phase 0 — CLI Setup
# =============================================================================

def _check_prerequisites() -> bool:
    """
    Verify that both infrastructure and service network outputs exist.
    Prints an error and returns False if any check fails.
    """
    main = Path(paths.MAIN)
    infra_root = main / paths.NETWORK_INFRASTRUCTURE_DIR
    feeder_root = main / paths.FEEDER_LINES_DIR

    ok = True
    # Check at least one non-Raw infra version exists
    infra_versions = [
        d for d in infra_root.iterdir()
        if d.is_dir() and d.name != "Raw"
        and (d / "nodes.gpkg").exists() and (d / "segments.gpkg").exists()
    ] if infra_root.exists() else []

    if not infra_versions:
        print(
            "\n  ERROR: No infrastructure version found under "
            f"{infra_root}\n"
            "  Run infrabuild_network_builder.py first."
        )
        ok = False

    # Check at least one svc_version with Unprojected/ base exists
    feeder_versions = [
        d for d in feeder_root.iterdir()
        if d.is_dir()
        and (d / paths.SERVICES_UNPROJECTED_SUBDIR / "pt_feeder_segments.gpkg").exists()
    ] if feeder_root.exists() else []

    if not feeder_versions:
        print(
            "\n  ERROR: No service version found under "
            f"{feeder_root}\n"
            "  Run services_network_builder.py first."
        )
        ok = False

    return ok


def _list_infra_versions() -> List[str]:
    """Return sorted list of infrastructure version names (excludes any Raw*)."""
    root = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR
    versions = [
        d.name for d in sorted(root.iterdir())
        if d.is_dir() and not d.name.startswith("Raw")
        and (d / "nodes.gpkg").exists() and (d / "segments.gpkg").exists()
    ]
    # Put Base first if present
    if "Base" in versions:
        versions = ["Base"] + [v for v in versions if v != "Base"]
    return versions


def _list_raw_dirs() -> List[str]:
    """Return Raw* folder names containing nodes.gpkg and segments.gpkg.

    infrabuild_filter_network.py writes to a study-area-suffixed name
    (e.g. 'Raw_ZH') or plain 'Raw'. Both are valid raw sources for projection.
    """
    root = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR
    if not root.exists():
        return []
    dirs = sorted([
        d.name for d in root.iterdir()
        if d.is_dir()
        and d.name.startswith("Raw")
        and (d / "nodes.gpkg").exists()
        and (d / "segments.gpkg").exists()
    ])
    if "Raw" in dirs:
        dirs = ["Raw"] + [d for d in dirs if d != "Raw"]
    return dirs


def _list_svc_versions() -> List[str]:
    """Return svc_version names that have an Unprojected/ base."""
    root = Path(paths.MAIN) / paths.FEEDER_LINES_DIR
    if not root.exists():
        return []
    return [
        d.name for d in sorted(root.iterdir())
        if d.is_dir()
        and (d / paths.SERVICES_UNPROJECTED_SUBDIR / "pt_feeder_segments.gpkg").exists()
    ]


def _list_source_options(svc_version: str) -> List[Tuple[str, Path]]:
    """Return (label, path) pairs for Unprojected/ source."""
    root = Path(paths.MAIN) / paths.FEEDER_LINES_DIR / svc_version
    return [("Unprojected (canonical base)", root / paths.SERVICES_UNPROJECTED_SUBDIR)]


def _list_projection_outputs() -> List[Tuple[str, str]]:
    """Return (svc_version, infra_version) tuples for already-projected outputs.

    Scans Feeder_Lines/<svc>/<infra>/ for pt_feeder_segments.gpkg,
    skipping the reserved subdirectory names.
    """
    _SKIP = {
        paths.SERVICES_UNPROJECTED_SUBDIR,
        paths.SERVICES_PROJECTED_SUBDIR,
    }
    results = []
    root = Path(paths.MAIN) / paths.FEEDER_LINES_DIR
    if not root.exists():
        return results
    for svc_dir in sorted(root.iterdir()):
        if not svc_dir.is_dir():
            continue
        for infra_dir in sorted(svc_dir.iterdir()):
            if (
                infra_dir.is_dir()
                and infra_dir.name not in _SKIP
                and (infra_dir / "pt_feeder_segments.gpkg").exists()
            ):
                results.append((svc_dir.name, infra_dir.name))
    return results


def _pick_one(labels: List[str], prompt: str = "Select") -> Optional[int]:
    """Display numbered list; return 0-based index or None on empty Enter."""
    for i, lbl in enumerate(labels, 1):
        print(f"     {i}) {lbl}")
    while True:
        raw = input(f"   {prompt} (number): ").strip()
        if not raw:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(labels):
            return int(raw) - 1
        print(f"   Invalid — enter 1–{len(labels)} or press Enter to cancel.")


def _run_phase0() -> Optional[Tuple[ProjectionConfig, str]]:
    """
    Interactive CLI setup.

    Returns:
        (ProjectionConfig, mode)  where mode is 'map' or 'correct'
        None if user cancels.
    """
    main = Path(paths.MAIN)
    print("\n" + "─" * 60)
    print("  Service Projection")
    print("─" * 60)

    # Raw folder — used by Phase 1.5 boundary rerouting to fill in junction
    # nodes that were dropped during macro-simplification. Mirrors the
    # selection pattern in infrabuild_network_builder.py.
    raw_dirs = _list_raw_dirs()
    if not raw_dirs:
        print("\n  ERROR: No Raw folder found in data/Infrastructure/.")
        print("  Run infrabuild_filter_network.py first.")
        return None
    if len(raw_dirs) == 1:
        chosen_raw = raw_dirs[0]
        print(f"\n  Raw folder: {chosen_raw}")
    else:
        print("\n  Available Raw folders:")
        for i, name in enumerate(raw_dirs, 1):
            print(f"    {i}) {name}")
        while True:
            sel = input("\n  Select Raw folder [1]: ").strip() or "1"
            if sel.isdigit() and 1 <= int(sel) <= len(raw_dirs):
                chosen_raw = raw_dirs[int(sel) - 1]
                break
            if sel in raw_dirs:
                chosen_raw = sel
                break
            print(f"  Invalid — enter 1–{len(raw_dirs)} or an exact name.")
    raw_infra_dir = main / paths.NETWORK_INFRASTRUCTURE_DIR / chosen_raw

    # Q1 — operation mode
    print("\n  What do you want to do?")
    print("    1) Map services (full pipeline: Phase 1 → 2 → 3)")
    print("    2) Correct an existing projection (Phase 2 onwards)")
    print("    3) Re-plot only (load existing projection, Phase 3 only)")
    while True:
        choice = input("  Select (1/2/3): ").strip()
        if choice in ("1", "2", "3"):
            break
        print("  Enter 1, 2, or 3.")

    if choice in ("2", "3"):
        existing = _list_projection_outputs()
        if not existing:
            print("  No existing projections found. Run mapping first.")
            return None
        _mode_label = "correct" if choice == "2" else "re-plot"
        print(f"\n  Choose an existing projection to {_mode_label}:")
        labels = [f"{svc}  /  {infra}" for svc, infra in existing]
        idx = _pick_one(labels, "Projection")
        if idx is None:
            return None
        svc_version, infra_version = existing[idx]
        source_feeder_path = main / paths.FEEDER_LINES_DIR / svc_version / infra_version
        source_rail_path   = main / paths.RAIL_LINES_DIR   / svc_version / infra_version
        mode = "correct" if choice == "2" else "plot"
    else:
        # Q2 — svc_version
        svc_versions = _list_svc_versions()
        if not svc_versions:
            print("  No service versions found. Run services_network_builder.py first.")
            return None
        print("\n  Choose service version:")
        idx = _pick_one(svc_versions, "Service version")
        if idx is None:
            return None
        svc_version = svc_versions[idx]

        # Q3 — source (Unprojected or named Version)
        source_options = _list_source_options(svc_version)
        print("\n  Choose source:")
        idx = _pick_one([label for label, _ in source_options], "Source")
        if idx is None:
            return None
        source_label, source_feeder_path = source_options[idx]

        # Derive parallel rail source path from feeder source path
        # e.g. Feeder_Lines/<svc>/Unprojected -> Rail_Lines/<svc>/Unprojected
        feeder_base = main / paths.FEEDER_LINES_DIR / svc_version
        try:
            rel = source_feeder_path.relative_to(feeder_base)
            source_rail_path = main / paths.RAIL_LINES_DIR / svc_version / rel
        except ValueError:
            source_rail_path = main / paths.RAIL_LINES_DIR / svc_version / paths.SERVICES_UNPROJECTED_SUBDIR

        # Q4 — infrastructure version
        infra_versions = _list_infra_versions()
        print("\n  Choose infrastructure version:")
        idx = _pick_one(infra_versions, "Infrastructure version")
        if idx is None:
            return None
        infra_version = infra_versions[idx]
        mode = "map"

    rail_output_dir   = main / paths.RAIL_LINES_DIR   / svc_version / infra_version
    feeder_output_dir = main / paths.FEEDER_LINES_DIR / svc_version / infra_version
    rail_output_dir.mkdir(parents=True, exist_ok=True)
    feeder_output_dir.mkdir(parents=True, exist_ok=True)

    config = ProjectionConfig(
        infra_version=infra_version,
        svc_version=svc_version,
        infra_dir=main / paths.NETWORK_INFRASTRUCTURE_DIR / infra_version,
        svc_dir=source_feeder_path,
        rail_input=source_rail_path / "rail_segments.gpkg",
        rail_output_dir=rail_output_dir,
        feeder_output_dir=feeder_output_dir,
        raw_infra_dir=raw_infra_dir,
    )

    print(f"\n  Service version: {svc_version}")
    print(f"  Source         : {source_feeder_path.name}")
    print(f"  Infrastructure : {infra_version}")
    print(f"  Rail output    : {rail_output_dir}")
    print(f"  Feeder output  : {feeder_output_dir}")
    return config, mode

# =============================================================================
# Spatial Output Helpers
# =============================================================================

def _load_polygon(path: Path):
    """Load a geopackage boundary and return its unioned geometry, or None."""
    try:
        if path.exists():
            gdf = gpd.read_file(path)
            return gdf.geometry.union_all()
    except Exception:
        pass
    return None


def _classify_edges_spatial(
    gdf: gpd.GeoDataFrame,
    sa_poly,
    ca_poly,
) -> gpd.GeoDataFrame:
    """Add _sa_from/_sa_to and _ca_from/_ca_to boolean columns.

    Uses internal column names x_origin/y_origin/x_dest/y_dest (before _OUTPUT_RENAME).
    When a boundary polygon is None (file absent), all edges are treated as inside.
    """
    from shapely.geometry import Point

    def _flags(row, poly):
        if poly is None:
            return True, True
        fp = Point(float(row.get("x_origin", 0) or 0), float(row.get("y_origin", 0) or 0))
        tp = Point(float(row.get("x_dest",   0) or 0), float(row.get("y_dest",   0) or 0))
        return fp.within(poly), tp.within(poly)

    result = gdf.copy()
    if result.empty:
        for col in ("_sa_from", "_sa_to", "_ca_from", "_ca_to"):
            result[col] = pd.Series(dtype=bool)
        return result

    sa_flags = [_flags(row, sa_poly) for _, row in result.iterrows()]
    ca_flags = [_flags(row, ca_poly) for _, row in result.iterrows()]
    result["_sa_from"], result["_sa_to"] = zip(*sa_flags)
    result["_ca_from"], result["_ca_to"] = zip(*ca_flags)
    return result


# =============================================================================
# Output Writers — shared helpers, rail writer, feeder writer
# =============================================================================

_PERIOD_SPECS: List[Tuple[str, str, frozenset]] = [
    ("All_Day",  "allday",  frozenset({"all_day"})),
    ("Peak",     "peak",    frozenset({"all_day", "peak_only"})),
    ("Off_Peak", "offpeak", frozenset({"all_day", "offpeak_only"})),
]


def _write_multilayer_gpkg(
    gdf: gpd.GeoDataFrame,
    path: Path,
    drop_cols: List[str],
    rename_map: Dict[str, str],
) -> None:
    # Remove existing file so layers are never appended to a stale version
    if Path(path).exists():
        Path(path).unlink()
    """Write a GDF as multi-layer GPKG; uses _source_layer for layer names."""
    if path.exists():
        path.unlink()
    if "_source_layer" in gdf.columns:
        for layer_name, layer_gdf in gdf.groupby("_source_layer"):
            (layer_gdf
             .drop(columns=["_source_layer"] + drop_cols, errors="ignore")
             .rename(columns=rename_map)
             .to_file(path, driver="GPKG", layer=layer_name))
    else:
        (gdf.drop(columns=drop_cols, errors="ignore")
            .rename(columns=rename_map)
            .to_file(path, driver="GPKG"))


def _copy_multilayer_gpkg(src: Path, dst: Path) -> None:
    """Copy a multi-layer GPKG layer-by-layer (preserves all layers)."""
    if not src.exists():
        print(f"  WARNING: {src.name} not found — skipping copy")
        return
    if dst.exists():
        dst.unlink()
    import pyogrio as _pyogrio
    for layer_name, _ in _pyogrio.list_layers(str(src)):
        gpd.read_file(src, layer=layer_name).to_file(
            dst, driver="GPKG", layer=layer_name)


def _write_stops_sa(src: Path, dst: Path, sa_poly) -> None:
    """Filter multi-layer stops GPKG to stops whose geometry is within sa_poly."""
    if dst.exists():
        dst.unlink()
    import pyogrio as _pyogrio
    if sa_poly is None:
        _copy_multilayer_gpkg(src, dst)
        return
    for layer_name, _ in _pyogrio.list_layers(str(src)):
        stops = gpd.read_file(src, layer=layer_name)
        within = stops[stops.geometry.within(sa_poly)]
        if not within.empty:
            within.to_file(dst, driver="GPKG", layer=layer_name)
    print(f"  Written: {dst.name}")


def _write_stops_filtered(src: Path, dst: Path, keep_numbers: set) -> None:
    """Write multi-layer stops GPKG filtered to a set of Number values."""
    import pyogrio as _pyogrio
    if not src.exists():
        return
    if dst.exists():
        dst.unlink()
    keep_str = {str(x) for x in keep_numbers}
    for layer_name, _ in _pyogrio.list_layers(str(src)):
        stops = gpd.read_file(src, layer=layer_name)
        if "Number" not in stops.columns:
            continue
        keep = stops[stops["Number"].astype(str).isin(keep_str)]
        if not keep.empty:
            keep.to_file(dst, driver="GPKG", layer=layer_name)


def _write_segments_qgz(
    subset_gdf: gpd.GeoDataFrame,
    qgz_path: Path,
    gpkg_name: str,
) -> None:
    """Build QGZ project for a rail segments GPKG."""
    by_type: Dict[int, gpd.GeoDataFrame] = {}
    if "_source_layer" in subset_gdf.columns:
        for layer_name, layer_gdf in subset_gdf.groupby("_source_layer"):
            rt = _LAYER_NAME_TO_RT.get(str(layer_name))
            if rt is not None:
                by_type[rt] = layer_gdf.drop(columns=["_source_layer"], errors="ignore")
    if not by_type:
        by_type[100] = subset_gdf
    layers_list = _collect_qgz_line_layers(
        by_type, gpkg_name, _RAIL_LINE_TYPES, suffix="Segments"
    )
    _build_qgz(str(qgz_path), layers_list)
    print(f"  Written: {qgz_path.name} ({len(layers_list)} layer(s))")


def _write_lines_qgz(
    lines_by_layer: Dict[str, gpd.GeoDataFrame],
    qgz_path: Path,
    gpkg_name: str,
    line_type_map: Optional[Dict[int, str]] = None,
) -> None:
    """Build QGZ project for a lines GPKG (rail or feeder)."""
    if not lines_by_layer:
        return
    lm = line_type_map if line_type_map is not None else _RAIL_LINE_TYPES
    by_type: Dict[int, gpd.GeoDataFrame] = {}
    for layer_name, gdf_l in lines_by_layer.items():
        rt = _LAYER_NAME_TO_RT.get(str(layer_name))
        if rt is not None:
            by_type[rt] = gdf_l
    if not by_type:
        return
    layers_list = _collect_qgz_line_layers(by_type, gpkg_name, lm, suffix="Lines")
    _build_qgz(str(qgz_path), layers_list)
    print(f"  Written: {qgz_path.name} ({len(layers_list)} layer(s))")


def _write_feeder_qgz(
    feeder_gdf: gpd.GeoDataFrame,
    qgz_path: Path,
    gpkg_name: str,
    suffix: str = "Segments",
) -> None:
    """Build QGZ project for a feeder segments GPKG."""
    by_type: Dict[int, gpd.GeoDataFrame] = {}
    if "_source_layer" in feeder_gdf.columns:
        for layer_name, layer_gdf in feeder_gdf.groupby("_source_layer"):
            rt = _LAYER_NAME_TO_RT.get(str(layer_name))
            if rt is not None:
                by_type[rt] = layer_gdf.drop(columns=["_source_layer"], errors="ignore")
    if not by_type:
        return
    layers_list = _collect_qgz_line_layers(
        by_type, gpkg_name, _PT_FEEDER_LINE_TYPES, suffix=suffix
    )
    _build_qgz(str(qgz_path), layers_list)
    print(f"  Written: {qgz_path.name} ({len(layers_list)} layer(s))")


def _filter_segments_by_sa(
    gdf: gpd.GeoDataFrame,
    sa_poly,
) -> gpd.GeoDataFrame:
    """Keep segments whose both endpoint coordinates lie within sa_poly."""
    if sa_poly is None:
        return gdf

    def _both_inside(row):
        f = Point(float(row.get("from_stop_E", 0) or 0),
                  float(row.get("from_stop_N", 0) or 0))
        t = Point(float(row.get("to_stop_E",   0) or 0),
                  float(row.get("to_stop_N",   0) or 0))
        return f.within(sa_poly) and t.within(sa_poly)

    mask = gdf.apply(_both_inside, axis=1)
    return gdf[mask]


def _build_projected_lines(
    projected_segs: gpd.GeoDataFrame,
    unprojected_lines_path: Path,
) -> Dict[str, gpd.GeoDataFrame]:
    """Aggregate projected segments into per-line merged geometries.

    Groups by (_source_layer, GTFS_ID, direction_id, variant_rank), linemerges
    the geometry per group, then left-joins metadata from the Unprojected lines
    GPKG (route_id → GTFS_ID key). Drops tt_source — projected lines don't carry it.

    Returns: {layer_name: GeoDataFrame}
    """
    import pyogrio as _pyogrio
    if not unprojected_lines_path.exists():
        print(f"  WARNING: {unprojected_lines_path.name} not found — skipping lines build")
        return {}

    out: Dict[str, gpd.GeoDataFrame] = {}
    layers = [l[0] for l in _pyogrio.list_layers(str(unprojected_lines_path))]

    for layer_name in layers:
        unproj = gpd.read_file(unprojected_lines_path, layer=layer_name)
        unproj = unproj.drop(columns=["tt_source", "geometry"], errors="ignore")

        segs = (projected_segs[projected_segs["_source_layer"] == layer_name]
                if "_source_layer" in projected_segs.columns
                else projected_segs)
        if segs.empty:
            continue

        gtfs_col = "GTFS_ID" if "GTFS_ID" in segs.columns else "Service"
        dir_col  = "direction_id" if "direction_id" in segs.columns else "Direction"

        line_rows = []
        for (gtfs_id, dir_id, var_rank), grp in segs.groupby(
                [gtfs_col, dir_col, "variant_rank"]):
            geoms = [g for g in grp.geometry
                     if g is not None and not g.is_empty and g.length > 0]
            if not geoms:
                continue
            union = unary_union(geoms)
            merged = linemerge(union) if union.geom_type != "LineString" else union
            line_rows.append({
                "route_id":     gtfs_id,
                "direction_id": dir_id,
                "variant_rank": var_rank,
                "geometry":     merged,
            })

        if not line_rows:
            continue

        proj_lines = gpd.GeoDataFrame(line_rows, crs=SWISS_CRS)
        merged_df  = proj_lines.merge(
            unproj, on=["route_id", "direction_id", "variant_rank"], how="left"
        )
        out[layer_name] = gpd.GeoDataFrame(merged_df, geometry="geometry", crs=SWISS_CRS)

    return out


def _write_period_subfolders_rail(
    segments_gdf: gpd.GeoDataFrame,
    lines_by_layer: Dict[str, gpd.GeoDataFrame],
    stops_path: Path,
    out_dir: Path,
    drop_cols: List[str],
    rename_map: Dict[str, str],
) -> None:
    """Write All_Day/, Peak/, Off_Peak/ subfolders for the rail projected output.

    Each subfolder contains rail_segments_<suffix>.gpkg/.qgz,
    rail_lines_<suffix>.gpkg/.qgz, and rail_stops_<suffix>.gpkg.
    """
    if "service_period" not in segments_gdf.columns:
        print("  [period] service_period column absent — skipping rail subfolders")
        return

    for subfolder_name, suffix, period_set in _PERIOD_SPECS:
        sub_dir = out_dir / subfolder_name
        sub_dir.mkdir(parents=True, exist_ok=True)

        seg_mask   = segments_gdf["service_period"].isin(period_set)
        seg_subset = segments_gdf[seg_mask]
        _write_multilayer_gpkg(
            seg_subset, sub_dir / f"rail_segments_{suffix}.gpkg", drop_cols, rename_map
        )

        lines_gpkg = sub_dir / f"rail_lines_{suffix}.gpkg"
        if lines_gpkg.exists():
            lines_gpkg.unlink()
        period_lines: Dict[str, gpd.GeoDataFrame] = {}
        for lname, lines_gdf in lines_by_layer.items():
            if "service_period" not in lines_gdf.columns:
                continue
            sub_lines = lines_gdf[lines_gdf["service_period"].isin(period_set)]
            if not sub_lines.empty:
                sub_lines.to_file(lines_gpkg, driver="GPKG", layer=lname)
                period_lines[lname] = sub_lines

        stop_nrs: set = set()
        for col in ("from_stop_nr", "to_stop_nr", "FromCode", "ToCode"):
            if col in seg_subset.columns:
                stop_nrs |= {str(x) for x in seg_subset[col].dropna().tolist()}
        _write_stops_filtered(
            src=stops_path,
            dst=sub_dir / f"rail_stops_{suffix}.gpkg",
            keep_numbers=stop_nrs,
        )

        _write_segments_qgz(seg_subset, sub_dir / f"rail_segments_{suffix}.qgz",
                            gpkg_name=f"rail_segments_{suffix}.gpkg")
        _write_lines_qgz(period_lines, sub_dir / f"rail_lines_{suffix}.qgz",
                         gpkg_name=f"rail_lines_{suffix}.gpkg")
        print(f"  Written: {subfolder_name}/ ({len(seg_subset)} rail segments)")


def _write_period_subfolders_feeder(
    segments_gdf: gpd.GeoDataFrame,
    lines_by_layer: Dict[str, gpd.GeoDataFrame],
    stops_path: Path,
    out_dir: Path,
) -> None:
    """Write All_Day/, Peak/, Off_Peak/ subfolders for the feeder projected output."""
    if "service_period" not in segments_gdf.columns:
        print("  [period] service_period column absent — skipping feeder subfolders")
        return

    for subfolder_name, suffix, period_set in _PERIOD_SPECS:
        sub_dir = out_dir / subfolder_name
        sub_dir.mkdir(parents=True, exist_ok=True)

        seg_mask   = segments_gdf["service_period"].isin(period_set)
        seg_subset = segments_gdf[seg_mask]
        _write_multilayer_gpkg(
            seg_subset, sub_dir / f"pt_feeder_segments_{suffix}.gpkg",
            drop_cols=[], rename_map={},
        )

        lines_gpkg = sub_dir / f"pt_feeder_lines_{suffix}.gpkg"
        if lines_gpkg.exists():
            lines_gpkg.unlink()
        period_lines: Dict[str, gpd.GeoDataFrame] = {}
        for lname, lines_gdf in lines_by_layer.items():
            if "service_period" not in lines_gdf.columns:
                continue
            sub_lines = lines_gdf[lines_gdf["service_period"].isin(period_set)]
            if not sub_lines.empty:
                sub_lines.to_file(lines_gpkg, driver="GPKG", layer=lname)
                period_lines[lname] = sub_lines

        stop_nrs: set = set()
        for col in ("from_stop_nr", "to_stop_nr"):
            if col in seg_subset.columns:
                stop_nrs |= {str(x) for x in seg_subset[col].dropna().tolist()}
        _write_stops_filtered(
            src=stops_path,
            dst=sub_dir / f"pt_feeder_stops_{suffix}.gpkg",
            keep_numbers=stop_nrs,
        )

        _write_feeder_qgz(seg_subset, sub_dir / f"pt_feeder_segments_{suffix}.qgz",
                          gpkg_name=f"pt_feeder_segments_{suffix}.gpkg")
        _write_lines_qgz(period_lines, sub_dir / f"pt_feeder_lines_{suffix}.qgz",
                         gpkg_name=f"pt_feeder_lines_{suffix}.gpkg",
                         line_type_map=_PT_FEEDER_LINE_TYPES)
        print(f"  Written: {subfolder_name}/ ({len(seg_subset)} feeder segments)")


def _write_feeder_outputs(
    config: "ProjectionConfig",
    track_feeder_enriched: Dict[str, gpd.GeoDataFrame],
    non_track_feeder_processed: Dict[str, gpd.GeoDataFrame],
) -> None:
    """Write projected feeder outputs mirroring the Unprojected folder schema.

    Produces:
      pt_feeder_segments.gpkg       — all projected feeder segments, multi-layer
      pt_feeder_segments_sa.gpkg    — SA-filtered
      pt_feeder_lines.gpkg          — aggregated per-line geometry
      pt_feeder_stops.gpkg          — copied from Unprojected
      pt_feeder_stops_sa.gpkg       — stops within SA boundary
      pt_feeder_segments.qgz, pt_feeder_lines.qgz
      All_Day/, Peak/, Off_Peak/ subfolders
    """
    out_dir  = config.feeder_output_dir
    main_dir = Path(paths.MAIN)
    sa_poly  = _load_polygon(main_dir / paths.STUDY_AREA_BOUNDARY_GPKG)

    all_layers: List[gpd.GeoDataFrame] = []
    for lname, lgdf in {**track_feeder_enriched, **non_track_feeder_processed}.items():
        if lgdf.empty:
            continue
        tagged = lgdf.copy()
        tagged["_source_layer"] = lname
        all_layers.append(tagged)
    if not all_layers:
        print("  [feeder] no projected feeder data — skipping outputs")
        return
    feeder_all = gpd.GeoDataFrame(pd.concat(all_layers, ignore_index=True), crs=SWISS_CRS)

    seg_path = out_dir / "pt_feeder_segments.gpkg"
    if seg_path.exists():
        seg_path.unlink()
    _write_multilayer_gpkg(feeder_all, seg_path, drop_cols=[], rename_map={})
    print(f"  Written: pt_feeder_segments.gpkg ({len(feeder_all)} segments)")

    feeder_sa = _filter_segments_by_sa(feeder_all, sa_poly)
    sa_path = out_dir / "pt_feeder_segments_sa.gpkg"
    if sa_path.exists():
        sa_path.unlink()
    _write_multilayer_gpkg(feeder_sa, sa_path, drop_cols=[], rename_map={})
    print(f"  Written: pt_feeder_segments_sa.gpkg ({len(feeder_sa)} segments)")

    lines_by_layer = _build_projected_lines(
        projected_segs=feeder_all,
        unprojected_lines_path=config.svc_dir / "pt_feeder_lines.gpkg",
    )
    lines_path = out_dir / "pt_feeder_lines.gpkg"
    if lines_path.exists():
        lines_path.unlink()
    for lname, gdf_l in lines_by_layer.items():
        gdf_l.to_file(lines_path, driver="GPKG", layer=lname)
    n_lines = sum(len(v) for v in lines_by_layer.values())
    print(f"  Written: pt_feeder_lines.gpkg ({n_lines} lines, {len(lines_by_layer)} layer(s))")

    _copy_multilayer_gpkg(
        src=config.svc_dir / "pt_feeder_stops.gpkg",
        dst=out_dir / "pt_feeder_stops.gpkg",
    )
    print("  Written: pt_feeder_stops.gpkg")
    _write_stops_sa(
        src=out_dir / "pt_feeder_stops.gpkg",
        dst=out_dir / "pt_feeder_stops_sa.gpkg",
        sa_poly=sa_poly,
    )

    _write_feeder_qgz(feeder_all, out_dir / "pt_feeder_segments.qgz",
                      gpkg_name="pt_feeder_segments.gpkg", suffix="Segments")
    _write_lines_qgz(lines_by_layer, out_dir / "pt_feeder_lines.qgz",
                     gpkg_name="pt_feeder_lines.gpkg",
                     line_type_map=_PT_FEEDER_LINE_TYPES)

    _write_period_subfolders_feeder(
        segments_gdf=feeder_all,
        lines_by_layer=lines_by_layer,
        stops_path=out_dir / "pt_feeder_stops.gpkg",
        out_dir=out_dir,
    )


def _write_rail_outputs(
    enriched: gpd.GeoDataFrame,
    config: "ProjectionConfig",
) -> None:
    """Write projected rail outputs mirroring the Unprojected folder schema.

    Produces:
      rail_segments.gpkg       — all projected segments, multi-layer by route type
      rail_segments_sa.gpkg    — both endpoints within SA, multi-layer
      rail_lines.gpkg          — one row per (GTFS_ID, direction_id, variant_rank)
      rail_stops.gpkg          — copied from Unprojected (unchanged by projection)
      rail_stops_sa.gpkg       — stops within SA boundary
      rail_segments.qgz, rail_lines.qgz
      All_Day/, Peak/, Off_Peak/ subfolders with the above set at each time slice
    """
    out_dir  = config.rail_output_dir
    main_dir = Path(paths.MAIN)
    sa_poly  = _load_polygon(main_dir / paths.STUDY_AREA_BOUNDARY_GPKG)
    if sa_poly is None:
        print("  [spatial] SA boundary absent — SA outputs will include all edges.")

    gdf = _classify_edges_spatial(enriched, sa_poly, None)
    _TEMP_COLS  = ["_sa_from", "_sa_to", "_ca_from", "_ca_to"]
    drop_all    = _OUTPUT_DROP + _TEMP_COLS
    already_out = "GTFS_ID" in enriched.columns
    rename_map  = {} if already_out else _OUTPUT_RENAME

    _write_multilayer_gpkg(gdf, out_dir / "rail_segments.gpkg", drop_all, rename_map)
    print(f"  Written: rail_segments.gpkg ({len(gdf)} edges)")

    sa_gdf = gdf[gdf["_sa_from"] & gdf["_sa_to"]]
    _write_multilayer_gpkg(sa_gdf, out_dir / "rail_segments_sa.gpkg", drop_all, rename_map)
    print(f"  Written: rail_segments_sa.gpkg ({len(sa_gdf)} edges)")

    _copy_multilayer_gpkg(
        src=config.rail_input.parent / "rail_stops.gpkg",
        dst=out_dir / "rail_stops.gpkg",
    )
    print("  Written: rail_stops.gpkg")
    _write_stops_sa(
        src=out_dir / "rail_stops.gpkg",
        dst=out_dir / "rail_stops_sa.gpkg",
        sa_poly=sa_poly,
    )

    _write_segments_qgz(gdf, out_dir / "rail_segments.qgz",
                        gpkg_name="rail_segments.gpkg")

    lines_by_layer = _build_projected_lines(
        projected_segs=gdf,
        unprojected_lines_path=config.rail_input.parent / "rail_lines.gpkg",
    )
    lines_path = out_dir / "rail_lines.gpkg"
    if lines_path.exists():
        lines_path.unlink()
    for lname, lines_gdf in lines_by_layer.items():
        lines_gdf.to_file(lines_path, driver="GPKG", layer=lname)
    n_lines = sum(len(v) for v in lines_by_layer.values())
    print(f"  Written: rail_lines.gpkg ({n_lines} lines, {len(lines_by_layer)} layer(s))")

    _write_lines_qgz(lines_by_layer, out_dir / "rail_lines.qgz",
                     gpkg_name="rail_lines.gpkg")

    _write_period_subfolders_rail(
        segments_gdf=gdf,
        lines_by_layer=lines_by_layer,
        stops_path=out_dir / "rail_stops.gpkg",
        out_dir=out_dir,
        drop_cols=drop_all,
        rename_map=rename_map,
    )


# =============================================================================
# Phase 1 — Projection Orchestrator
# =============================================================================

def _run_phase1(
    config: ProjectionConfig,
) -> Tuple[gpd.GeoDataFrame, Dict[str, gpd.GeoDataFrame], Dict[str, gpd.GeoDataFrame]]:
    """
    Phase 1: load data, match stops, route paths.

    Returns:
        (rail_enriched, track_feeder_enriched, non_track_feeder_processed)
    """
    main = Path(paths.MAIN)
    print("\n" + "─" * 60)
    print("  Phase 1 — Service Projection")
    print("─" * 60)

    # 1a. Load infrastructure
    print("\n  Loading infrastructure...")
    nodes = gpd.read_file(config.infra_dir / "nodes.gpkg").reset_index(drop=True)
    bav_segments = gpd.read_file(config.infra_dir / "segments.gpkg").reset_index(drop=True)
    print(f"  {len(nodes)} nodes, {len(bav_segments)} segments loaded.")

    # Load raw infrastructure for Tier 4 fallback
    raw_nodes_path = config.raw_infra_dir / "nodes.gpkg"
    raw_segs_path = config.raw_infra_dir / "segments.gpkg"
    raw_nodes = gpd.read_file(raw_nodes_path) if raw_nodes_path.exists() else None
    raw_segs = gpd.read_file(raw_segs_path) if raw_segs_path.exists() else None
    if raw_nodes is None:
        print("  NOTE: Raw nodes not found — Tier 4 fallback disabled.")

    # 1b. Build graph, lookups, node attributes
    print("  Building infrastructure graph (with missing node healing from raw_nodes)...")
    G = build_infra_graph(nodes, bav_segments, raw_nodes)
    seg_lookup = build_segment_lookup(nodes, bav_segments, raw_nodes)
    node_attrs = build_node_attrs(nodes)
    lookups = build_stop_lookups(nodes)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    gauge_graphs = _build_gauge_graphs(G)

    # 1b-ii. Precompute hub topology (forced-routing flags for hub stations)
    print("  Building hub topology...")
    hub_topology = build_hub_topology(nodes, G)
    print(f"  Hub topology: {len(hub_topology)} hub(s) identified.")

    # 1c. Load buffer geometry for inside/outside decision
    buffer_geom = None
    buf_path = main / paths.CATCHMENT_AREA_BUFFER_GPKG
    if buf_path.exists():
        buf_gdf = gpd.read_file(buf_path)
        buffer_geom = buf_gdf.geometry.union_all()
        print(f"  Buffer loaded: {buf_path.name}")
    else:
        print("  WARNING: Catchment area buffer not found — all stops treated as inside.")

    # 1d. Build stop_coord from feeder segments + load ZVV geometry
    print("\n  Building stop coordinate index from feeder segments...")
    global stop_coord
    stop_coord = _build_stop_coord_from_segments(config.svc_dir)
    print(f"  {len(stop_coord):,} stop coordinates indexed.")

    print("\n  Loading ZVV geometry...")
    _zvv_seg_index.clear()
    _zvv_chain_index.clear()
    _zvv_sbahn_index.clear()
    _zvv_sbahn_jgraph.clear()
    _zvv_geometry_available = _load_zvv_geometry()
    if not _zvv_geometry_available:
        print("  ZVV geometry unavailable — straight-line fallback for non-track modes.")

    # 1e. Load service data (all feeder layers dynamically)
    print("\n  Loading service data...")

    import fiona as _fiona
    rail_layers = _fiona.listlayers(str(config.rail_input))
    rail_gdfs = []
    for layer in rail_layers:
        gdf = gpd.read_file(config.rail_input, layer=layer)
        gdf["_source_layer"] = layer
        rail_gdfs.append(gdf)
    rail_segments = pd.concat(rail_gdfs, ignore_index=True) if rail_gdfs else gpd.GeoDataFrame()

    col_mapping = {
        # New pre-projection names → internal processing names (unchanged)
        'GTFS_ID':        'Service',      # route_id (GTFS identifier)
        'Service':        'TrainType',    # line_short_name (human-readable)
        'direction_id':   'Direction',
        'from_stop_nr':   'FromCode',     # BAV parent station integer
        'to_stop_nr':     'ToCode',
        'from_stop_name': 'FromStation',
        'to_stop_name':   'ToStation',
        'from_stop_E':    'x_origin',
        'from_stop_N':    'y_origin',
        'to_stop_E':      'x_dest',
        'to_stop_N':      'y_dest',
        'TT':             'TravelTime',
        'IVWT':           'InVehWait',
    }
    rail_edges = rail_segments.rename(columns=col_mapping)
    if 'TrainType' in rail_edges.columns:
        rail_edges['line_short_name'] = rail_edges['TrainType']

    feeder_gpkg = config.svc_dir / "pt_feeder_segments.gpkg"
    all_feeder_layers = _fiona.listlayers(str(feeder_gpkg))
    track_feeder_gdfs: Dict[str, gpd.GeoDataFrame] = {}
    non_track_feeder_gdfs: Dict[str, gpd.GeoDataFrame] = {}
    for layer in all_feeder_layers:
        gdf = gpd.read_file(feeder_gpkg, layer=layer)
        if layer in TRACK_BASED_FEEDER_LAYERS:
            track_feeder_gdfs[layer] = gdf
        else:
            non_track_feeder_gdfs[layer] = gdf

    print(
        f"  Rail: {len(rail_edges)} links | "
        f"Track feeders: {sum(len(v) for v in track_feeder_gdfs.values())} segs "
        f"({', '.join(track_feeder_gdfs.keys())}) | "
        f"Non-track: {sum(len(v) for v in non_track_feeder_gdfs.values())} segs "
        f"({', '.join(non_track_feeder_gdfs.keys())})"
    )

    # 1f. Enrich rail (BAV projection)
    print("\n  Enriching rail links...")
    rail_enriched = enrich_rail_links(
        rail_edges, nodes, bav_segments, G, seg_lookup, node_attrs,
        lookups, buffer_geom, raw_nodes, raw_segs, config.infra_dir,
        hub_topology=hub_topology, gauge_graphs=gauge_graphs,
    )
    n_corrected = rail_enriched["needs_correction"].sum()
    print(f"  Rail done. {n_corrected} links need correction.")

    # 1g. Enrich track-based feeders (BAV projection)
    track_feeder_enriched: Dict[str, gpd.GeoDataFrame] = {}
    for layer_name, segs in track_feeder_gdfs.items():
        print(f"\n  Enriching {layer_name} segments...")
        enriched = enrich_feeder_segments(
            segs, nodes, bav_segments, G, seg_lookup, node_attrs,
            lookups, buffer_geom, raw_nodes, raw_segs, config.infra_dir,
            hub_topology=hub_topology, gauge_graphs=gauge_graphs,
        )
        print(f"  {layer_name} done. {enriched['needs_correction'].sum()} need correction.")
        track_feeder_enriched[layer_name] = enriched

    # 1h. ZVV post-pass: replace straight-line results + check divergence
    if _zvv_geometry_available:
        print("\n  Applying ZVV post-pass to rail...")
        rail_enriched = _apply_zvv_postpass(
            rail_enriched, "rail", is_track_based=True, buffer_geom=buffer_geom
        )

        for layer_name in list(track_feeder_enriched.keys()):
            print(f"  Applying ZVV post-pass to {layer_name}...")
            track_feeder_enriched[layer_name] = _apply_zvv_postpass(
                track_feeder_enriched[layer_name], layer_name,
                is_track_based=True, buffer_geom=buffer_geom
            )

    # 1i. ZVV geometry for non-track modes (bus, ship, etc.)
    non_track_feeder_processed: Dict[str, gpd.GeoDataFrame] = {}
    if non_track_feeder_gdfs:
        print("\n  Applying ZVV geometry to non-track feeder modes...")
        for layer_name, segs in non_track_feeder_gdfs.items():
            processed = _apply_zvv_postpass(segs, layer_name, is_track_based=False)
            non_track_feeder_processed[layer_name] = processed

    # 1j. Rail and feeder outputs are written after Phase 1.5 completes (in main())
    print(f"\n  Phase 1 complete.")
    return rail_enriched, track_feeder_enriched, non_track_feeder_processed

# =============================================================================
# Phase 1.5 — Boundary Station Routing
# =============================================================================

def _collect_outside_stops_gateway(
    rail_enriched: gpd.GeoDataFrame,
    buffer_geom,
) -> Dict[str, Dict]:
    """
    Collect unique stops from gateway-layer services (LD, IR, RE, S-Bahn) whose
    coordinates lie outside buffer_geom.  Only rows whose _source_layer is in
    _GATEWAY_LAYERS are examined.

    Returns {stop_id: {name, E, N, services: set()}}
    """
    outside: Dict[str, Dict] = {}
    if "_source_layer" not in rail_enriched.columns:
        return outside

    mask = rail_enriched["_source_layer"].isin(_GATEWAY_LAYERS)
    for _, row in rail_enriched[mask].iterrows():
        svc = str(row.get("Service", "?"))

        from_E = float(row.get("x_origin", 0) or 0)
        from_N = float(row.get("y_origin", 0) or 0)
        to_E   = float(row.get("x_dest",   0) or 0)
        to_N   = float(row.get("y_dest",   0) or 0)

        from_inside = buffer_geom is None or Point(from_E, from_N).within(buffer_geom)
        to_inside   = buffer_geom is None or Point(to_E,   to_N  ).within(buffer_geom)

        # Only collect a stop if it is outside AND its partner in this link is
        # inside — this restricts the mapping to boundary-crossing stops only.
        # Stops where both endpoints are outside are skipped; they keep their
        # Phase-1 straight-line geometry unchanged.
        for sid_col, sname_col, e_col, n_col, this_inside, partner_inside in [
            ("FromCode", "FromStation", "x_origin", "y_origin", from_inside, to_inside),
            ("ToCode",   "ToStation",   "x_dest",   "y_dest",   to_inside,   from_inside),
        ]:
            if this_inside or not partner_inside:
                continue  # stop is inside, or partner is also outside
            sid   = str(row.get(sid_col,   "")).strip()
            sname = str(row.get(sname_col, "")).strip()
            E     = float(row.get(e_col, 0) or 0)
            N     = float(row.get(n_col, 0) or 0)
            if not sid:
                continue
            if sid not in outside:
                outside[sid] = {"name": sname, "E": E, "N": N, "services": set()}
            outside[sid]["services"].add(svc)

    return outside


def _run_destination_mapping_cli(
    outside_stops: Dict[str, Dict],
    confirmed_boundary_ids: List[int],
    node_attrs: Dict[int, Dict],
    existing_mapping: Optional[Dict[str, int]] = None,
) -> Dict[str, int]:
    """
    Interactive CLI: assign each unique outside stop to a boundary station.

    Displays existing assignments when editing a saved mapping so the user can
    skip unchanged entries with Enter.  Returns {stop_id: boundary_node_id}.
    """
    if not outside_stops:
        print("  No outside destinations found — nothing to map.")
        return {}
    if not confirmed_boundary_ids:
        print("  No boundary stations confirmed — cannot map.")
        return {}

    mapping: Dict[str, int] = dict(existing_mapping or {})

    boundary_id_set = set(confirmed_boundary_ids)

    # Auto-assign outside stops that are themselves boundary stations.
    # stop_id is the Betriebspunkt_Nummer as a string (e.g. "8506137").
    auto_assigned_stops: set = set()
    for stop_id, info in outside_stops.items():
        try:
            numeric_id = int(stop_id.split(":")[-1]) if ":" in stop_id else int(stop_id)
        except (ValueError, AttributeError):
            continue
        if numeric_id in boundary_id_set:
            mapping[stop_id] = numeric_id
            auto_assigned_stops.add(stop_id)
    if auto_assigned_stops:
        print(f"\n  {len(auto_assigned_stops)} outside stop(s) are boundary stations — auto-assigned to themselves.")

    print("\n  Available boundary stations:")
    for i, nid in enumerate(confirmed_boundary_ids, 1):
        name = node_attrs.get(nid, {}).get("name", str(nid))
        print(f"    {i:3}) {name}  (node {nid})")

    print(f"\n  Outside stops on gateway-layer services ({len(outside_stops)} unique):")
    for stop_id, info in outside_stops.items():
        if stop_id in auto_assigned_stops:
            continue

        svc_sample = ", ".join(sorted(info["services"])[:5])
        n_svc      = len(info["services"])
        existing   = mapping.get(stop_id)
        cur_str    = ""
        if existing is not None:
            cur_name = node_attrs.get(existing, {}).get("name", str(existing))
            cur_str  = f"  [currently → {cur_name}]"

        raw = input(
            f"\n  '{info['name']}' ({n_svc} service(s): {svc_sample}){cur_str}\n"
            f"    → boundary station number (or Enter to skip): "
        ).strip()

        if not raw:
            continue
        if raw.isdigit() and 1 <= int(raw) <= len(confirmed_boundary_ids):
            chosen_nid  = confirmed_boundary_ids[int(raw) - 1]
            mapping[stop_id] = chosen_nid
            chosen_name = node_attrs.get(chosen_nid, {}).get("name", str(chosen_nid))
            print(f"    → Assigned to: {chosen_name}")
        else:
            print(f"    Invalid input — skipped.")

    return mapping


def _apply_boundary_rerouting(
    rail_enriched: gpd.GeoDataFrame,
    boundary_mapping: Dict[str, int],
    node_attrs: Dict[int, Dict],
    G: nx.Graph,
    seg_lookup: Dict,
    buffer_geom,
) -> gpd.GeoDataFrame:
    """
    Post-process rail_enriched: for gateway-layer links (LD, IR, RE, S-Bahn)
    where from_stop or to_stop has a boundary mapping, replace straight-line
    geometry with:

        straight_line(outside_coords → boundary_node)
        + routed_path(boundary_node → inside_node)

    Both entry and exit cases are handled symmetrically.  When both stops are
    outside (service passing through), both segments are stitched together with
    the graph-routed middle portion.

    New columns added: boundary_entry_node, boundary_exit_node (pd.NA when unused).
    Clears needs_correction on updated rows.
    """
    for col in ("boundary_entry_node", "boundary_exit_node"):
        if col not in rail_enriched.columns:
            rail_enriched[col] = pd.NA

    if "_source_layer" not in rail_enriched.columns:
        return rail_enriched

    mask    = rail_enriched["_source_layer"].isin(_GATEWAY_LAYERS)
    updated = 0

    for idx, row in rail_enriched[mask].iterrows():
        from_id = str(row.get("FromCode", "")).strip()
        to_id   = str(row.get("ToCode",   "")).strip()
        from_E  = float(row.get("x_origin", 0) or 0)
        from_N  = float(row.get("y_origin", 0) or 0)
        to_E    = float(row.get("x_dest",   0) or 0)
        to_N    = float(row.get("y_dest",   0) or 0)

        from_bnode = boundary_mapping.get(from_id)
        to_bnode   = boundary_mapping.get(to_id)
        if from_bnode is None and to_bnode is None:
            continue  # no mapping for either stop — nothing to do

        from_pt      = Point(from_E, from_N)
        to_pt        = Point(to_E,   to_N)
        from_outside = buffer_geom is None or not from_pt.within(buffer_geom)
        to_outside   = buffer_geom is None or not to_pt.within(buffer_geom)

        if from_outside and to_outside:
            continue  # both stops outside — not a boundary-crossing link, keep Phase-1 geometry

        # Determine effective routing endpoints on the BAV graph
        if from_outside and from_bnode is not None:
            route_from: Optional[int] = from_bnode
        else:
            nf = row.get("node_id_from")
            route_from = int(nf) if pd.notna(nf) else None

        if to_outside and to_bnode is not None:
            route_to: Optional[int] = to_bnode
        else:
            nt = row.get("node_id_to")
            route_to = int(nt) if pd.notna(nt) else None

        if route_from is None or route_to is None:
            continue

        # Degenerate: boundary node equals the other matched endpoint
        if route_from == route_to:
            bE = node_attrs.get(route_from, {}).get("E", from_E)
            bN = node_attrs.get(route_from, {}).get("N", from_N)
            rail_enriched.at[idx, "geometry"] = LineString(
                [(from_E, from_N), (bE, bN)]
            )
            if from_outside and from_bnode is not None:
                rail_enriched.at[idx, "boundary_entry_node"] = str(route_from)
            rail_enriched.at[idx, "needs_correction"] = False
            updated += 1
            continue

        # Route between the two effective endpoints on the BAV graph (Phase 1.5
        # boundary rerouting). The reroute is purely geometric — we don't need
        # the per-service weight here, so service_stops carries just the two
        # effective endpoints. Both will be in service_stops, so n_decel
        # behaves the same as for any direct service link.
        routed_geom, via_st, via_jn, path_len, _, _, _pns, _pw = route_between_nodes(
            G, [route_from], [route_to], seg_lookup, node_attrs,
            service_stops={route_from, route_to},
        )
        if routed_geom is None:
            continue  # no graph path found — keep existing geometry

        # Build combined geometry parts
        geom_parts: List = []

        if from_outside and from_bnode is not None:
            bE = node_attrs.get(from_bnode, {}).get("E", from_E)
            bN = node_attrs.get(from_bnode, {}).get("N", from_N)
            geom_parts.append(LineString([(from_E, from_N), (bE, bN)]))
            rail_enriched.at[idx, "boundary_entry_node"] = str(from_bnode)

        if routed_geom.geom_type == "LineString":
            geom_parts.append(routed_geom)
        elif hasattr(routed_geom, "geoms"):
            geom_parts.extend(
                g for g in routed_geom.geoms if g.geom_type == "LineString"
            )

        if to_outside and to_bnode is not None:
            bE = node_attrs.get(to_bnode, {}).get("E", to_E)
            bN = node_attrs.get(to_bnode, {}).get("N", to_N)
            geom_parts.append(LineString([(bE, bN), (to_E, to_N)]))
            rail_enriched.at[idx, "boundary_exit_node"] = str(to_bnode)

        combined = (
            linemerge(geom_parts) if len(geom_parts) > 1
            else (geom_parts[0] if geom_parts else routed_geom)
        )

        _via_nodes, _via_segment = _make_via_cols(_pns)
        rail_enriched.at[idx, "geometry"]         = combined
        rail_enriched.at[idx, "Via_Station"]      = via_st
        rail_enriched.at[idx, "Via_Junction"]     = via_jn
        rail_enriched.at[idx, "path_nodes"]       = _pns
        rail_enriched.at[idx, "Via_Nodes"]        = _via_nodes
        rail_enriched.at[idx, "Via_Segment"]      = _via_segment
        rail_enriched.at[idx, "path_length_m"]    = path_len
        rail_enriched.at[idx, "needs_correction"] = False
        updated += 1

    print(f"  Boundary rerouting: {updated} link(s) updated.")
    return rail_enriched


def _run_phase1_5(
    config: ProjectionConfig,
    rail_enriched: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Phase 1.5: Boundary station detection and outside-destination rerouting.

    Workflow
    --------
    a) Auto-detect boundary station candidates (leaf stations ≤5 km from buffer
       edge); user confirms the list once — saved to boundary_stations.json.
    b) Collect unique outside stops from LD/IR service links in rail_enriched.
    c) User maps each outside stop to a boundary station — saved to
       boundary_mapping.json.  On re-run, existing mapping is loaded and the
       user can skip unchanged assignments.
    d) Re-route affected links: straight_line(outside→boundary) + graph route
       (boundary→inside stop), replacing the Phase-1 straight-line geometry.
    e) Updated rail_enriched is saved to disk, overwriting the Phase-1 output
       so Phase 2 and QGIS inspection see the improved geometry.

    Returns the updated rail_enriched GeoDataFrame.
    """
    main_dir  = Path(paths.MAIN)
    bs_path   = config.rail_output_dir / "boundary_stations.json"
    bm_path   = config.rail_output_dir / "boundary_mapping.json"

    print("\n" + "─" * 60)
    print("  Phase 1.5 — Boundary Station Mapping")
    print("─" * 60)

    # Re-load infrastructure (may have been updated by Tier 4 during Phase 1).
    # raw_nodes must be passed so missing junction nodes (e.g. Winterthur Nord)
    # are healed into the graph — without them boundary rerouting takes long
    # detours on severed corridors.
    raw_nodes_path = config.raw_infra_dir / "nodes.gpkg"
    raw_nodes_1_5  = gpd.read_file(raw_nodes_path) if raw_nodes_path.exists() else None
    nodes      = gpd.read_file(config.infra_dir / "nodes.gpkg")
    bav_segs   = gpd.read_file(config.infra_dir / "segments.gpkg")
    G          = build_infra_graph(nodes, bav_segs, raw_nodes_1_5)
    seg_lookup = build_segment_lookup(nodes, bav_segs, raw_nodes_1_5)
    node_attrs = build_node_attrs(nodes)

    # Load buffer geometry (same source as Phase 1)
    buffer_geom = None
    buf_path    = main_dir / paths.CATCHMENT_AREA_BUFFER_GPKG
    if buf_path.exists():
        buf_gdf     = gpd.read_file(buf_path)
        buffer_geom = buf_gdf.geometry.union_all()

    # ── a. Boundary station list ──────────────────────────────────────────────
    existing_bs = _load_boundary_stations(bs_path)
    if existing_bs is not None:
        print(f"\n  Loaded {len(existing_bs)} boundary station(s) from {bs_path.name}.")
        ans = input("  Re-detect and re-confirm? (y/n) [n]: ").strip().lower() or "n"
        confirmed_bs: Optional[List[int]] = existing_bs if ans != "y" else None
    else:
        confirmed_bs = None

    if confirmed_bs is None:
        candidates = detect_boundary_station_candidates(G, nodes, buffer_geom)
        print(f"\n  {len(candidates)} candidate boundary station(s) detected.")
        if not candidates:
            print("  No candidates found — Phase 1.5 skipped.")
            return rail_enriched
        confirmed_bs = _run_boundary_station_confirmation_cli(candidates, node_attrs)
        _save_boundary_stations(confirmed_bs, bs_path)
        print(f"  Boundary stations saved to {bs_path.name}.")

    if not confirmed_bs:
        print("  No boundary stations confirmed — Phase 1.5 skipped.")
        return rail_enriched

    # ── b. Collect outside stops from gateway-layer services ─────────────────
    outside_stops = _collect_outside_stops_gateway(rail_enriched, buffer_geom)
    print(f"\n  {len(outside_stops)} unique outside stop(s) found on gateway-layer services.")
    if not outside_stops:
        print("  Nothing to map — Phase 1.5 skipped.")
        return rail_enriched

    # ── c. Destination → boundary station mapping ─────────────────────────────
    existing_bm = _load_boundary_mapping(bm_path)
    if existing_bm is not None:
        print(f"  Loaded {len(existing_bm)} mapping(s) from {bm_path.name}.")
        ans = input("  Edit existing mapping? (y/n) [n]: ").strip().lower() or "n"
        if ans == "y":
            boundary_mapping = _run_destination_mapping_cli(
                outside_stops, confirmed_bs, node_attrs,
                existing_mapping=existing_bm,
            )
        else:
            boundary_mapping = existing_bm
    else:
        boundary_mapping = _run_destination_mapping_cli(
            outside_stops, confirmed_bs, node_attrs,
        )

    if not boundary_mapping:
        print("  No mappings provided — Phase 1.5 skipped.")
        return rail_enriched

    _save_boundary_mapping(boundary_mapping, bm_path)
    print(f"  Mapping saved to {bm_path.name}  ({len(boundary_mapping)} entry/ies).")

    # ── d. Re-route affected gateway-layer links ─────────────────────────────
    print("\n  Applying boundary rerouting to gateway-layer links...")
    rail_enriched = _apply_boundary_rerouting(
        rail_enriched, boundary_mapping, node_attrs, G, seg_lookup, buffer_geom,
    )

    print("\n  Phase 1.5 complete.")
    return rail_enriched


# =============================================================================
# Phase 2a — QGIS Projects and Clipped Segment Geopackages
# =============================================================================

def _save_phase2_outputs(
    config: ProjectionConfig,
    rail_enriched: gpd.GeoDataFrame,
    tram_enriched: gpd.GeoDataFrame,
    func_enriched: gpd.GeoDataFrame,
) -> None:
    """
    Rebuild the PT-Feeder QGIS project file (.qgz) from disk after Phase 2 corrections.

    Reads the pt_feeder_segments.gpkg and pt_feeder_stops.gpkg that were written
    by _write_feeder_outputs() (or updated in-place by tram/funicular corrections
    in _run_phase2()) and writes a fresh pt_feeder_segments.qgz.
    """
    import fiona as _fiona
    main = Path(paths.MAIN)
    print("\n  Building QGIS projects and clipped segment geopackages...")

    # ── PT-Feeder QGZ ──────────────────────────────────────────────────────────
    orig_seg_path   = config.feeder_output_dir / "pt_feeder_segments.gpkg"
    orig_stops_path = config.feeder_output_dir / "pt_feeder_stops.gpkg"

    # Enriched modes written by Phase 1
    enriched_feeder_by_type: Dict[int, gpd.GeoDataFrame] = {
        900:  tram_enriched,
        1400: func_enriched,
    }
    # Pass-through modes (bus, ship, …) — original geometry from svc_version output
    passthrough_by_type: Dict[int, gpd.GeoDataFrame] = {}
    if orig_seg_path.exists():
        for lname in _fiona.listlayers(str(orig_seg_path)):
            rt = _LAYER_NAME_TO_RT.get(lname)
            if rt is not None and rt not in enriched_feeder_by_type:
                passthrough_by_type[rt] = gpd.read_file(str(orig_seg_path), layer=lname)

    # Stops — all from original (stop locations are unchanged by projection)
    feeder_stops_by_type: Dict[int, gpd.GeoDataFrame] = {}
    if orig_stops_path.exists():
        for lname in _fiona.listlayers(str(orig_stops_path)):
            rt = _LAYER_NAME_TO_RT.get(lname)
            if rt is not None:
                feeder_stops_by_type[rt] = gpd.read_file(str(orig_stops_path), layer=lname)

    feeder_layers_list: List[dict] = []
    if feeder_stops_by_type:
        feeder_layers_list += _collect_qgz_stop_layers(
            feeder_stops_by_type, "./pt_feeder_stops.gpkg", _PT_FEEDER_LINE_TYPES, is_rail=False
        )
    feeder_layers_list += _collect_qgz_line_layers(
        enriched_feeder_by_type, "./pt_feeder_segments.gpkg",
        _PT_FEEDER_LINE_TYPES, suffix="Segments (projected)"
    )
    if passthrough_by_type:
        feeder_layers_list += _collect_qgz_line_layers(
            passthrough_by_type, "./pt_feeder_segments.gpkg",
            _PT_FEEDER_LINE_TYPES, suffix="Segments"
        )

    feeder_qgz = config.feeder_output_dir / "pt_feeder_segments.qgz"
    _build_qgz(str(feeder_qgz), feeder_layers_list)
    print(f"  PT-Feeder QGZ → {feeder_qgz}  ({len(feeder_layers_list)} layer(s))")


# =============================================================================
# Phase 2b — Corrections TUI
# =============================================================================

def _show_service_stops(
    enriched: gpd.GeoDataFrame,
    service_code: str,
    route_col: str,
    from_name_col: str,
    to_name_col: str,
    from_method_col: str,
    path_len_col: str,
) -> List[int]:
    """
    Print stop sequence for a service. Returns list of row indices in sequence order.
    """
    subset = enriched[enriched[route_col] == service_code]
    if subset.empty:
        print(f"  No links found for service '{service_code}'.")
        return []

    print(f"\n  Stop sequence for '{service_code}':")
    indices = list(subset.index)
    stop_num = 1

    for i, idx in enumerate(indices):
        row = enriched.loc[idx]
        from_name = row.get(from_name_col, "?")
        method = row.get(from_method_col, "?")
        node_id = row.get("node_id_from", None)
        km = row.get(path_len_col, 0) / 1000.0
        flag = " ← UNMATCHED" if row.get("needs_correction", False) else ""
        node_str = f"node {node_id}" if node_id else "UNMATCHED"
        print(
            f"    {stop_num:2}. {from_name:<30}  [{method:8}, {node_str}]  "
            f"→ {km:.1f} km{flag}"
        )
        stop_num += 1

        # Print the to-stop of the last link
        if i == len(indices) - 1:
            to_name = row.get(to_name_col, "?")
            to_method = row.get("match_method_to", "?")
            to_node = row.get("node_id_to", None)
            to_node_str = f"node {to_node}" if to_node else "UNMATCHED"
            print(f"    {stop_num:2}. {to_name:<30}  [{to_method:8}, {to_node_str}]")

    return indices


def _reroute_link(
    enriched: gpd.GeoDataFrame,
    link_idx: int,
    from_node_id: int,
    to_node_id: int,
    G: nx.Graph,
    seg_lookup: Dict,
    node_attrs: Dict,
) -> gpd.GeoDataFrame:
    """
    Interactively build a new path from from_node_id towards to_node_id by
    letting the user pick segments step by step.

    Updates the row at link_idx in enriched and returns the modified GeoDataFrame.
    """
    current_node = from_node_id
    path_nodes: List[int] = [current_node]

    print(f"\n  Building new path from node {from_node_id} → target node {to_node_id}")
    print("  At each step, pick the next segment. Type DONE to confirm when ready.\n")

    while True:
        neighbours = list(G.neighbors(current_node))
        if not neighbours:
            print("  Dead end — no reachable neighbours. Path confirmed as-is.")
            break

        labels = []
        for nb in neighbours:
            nb_name = node_attrs.get(nb, {}).get("name", str(nb))
            seg = seg_lookup.get((current_node, nb))
            seg_id = seg["segment_id"] if seg is not None else "?"
            km = G[current_node][nb].get("length_m", 0) / 1000.0
            labels.append(f"{nb_name}  [{seg_id}, {km:.2f} km]")

        current_name = node_attrs.get(current_node, {}).get("name", str(current_node))
        print(f"  From: {current_name}")
        for i, lbl in enumerate(labels, 1):
            print(f"    {i}) {lbl}")
        print("    d) DONE — confirm path up to here")

        raw = input("  Pick next segment (number or d): ").strip().lower()
        if raw == "d":
            if current_node != to_node_id:
                print(
                    f"  WARNING: Path ends at node {current_node} "
                    f"(target was {to_node_id})."
                )
                confirm = input("  Confirm anyway? (y/n) [n]: ").strip().lower() or "n"
                if confirm != "y":
                    continue
            break
        if raw.isdigit() and 1 <= int(raw) <= len(neighbours):
            current_node = neighbours[int(raw) - 1]
            path_nodes.append(current_node)
            if current_node == to_node_id:
                print(f"  Reached target node {to_node_id}.")
                break
        else:
            print(f"  Invalid — enter 1–{len(neighbours)} or d.")

    if len(path_nodes) < 2:
        print("  No path built — no changes made.")
        return enriched

    # Reconstruct geometry and via columns from path_nodes
    geoms = []
    path_length = 0.0
    via_st: List[str] = []
    via_jn: List[str] = []

    for i in range(len(path_nodes) - 1):
        seg = seg_lookup.get((path_nodes[i], path_nodes[i + 1]))
        if seg is not None and seg.geometry is not None:
            g = seg.geometry
            if g.geom_type == "LineString":
                geoms.append(g)
            elif hasattr(g, 'geoms'):
                geoms.extend([sub_g for sub_g in g.geoms if sub_g.geom_type == "LineString"])
        path_length += float(G[path_nodes[i]][path_nodes[i + 1]].get("length_m", 0))

    for nid in path_nodes[1:-1]:
        attrs = node_attrs.get(nid, {})
        name = attrs.get("name", str(nid))
        if attrs.get("node_class") == "station":
            via_st.append(name)
        else:
            via_jn.append(name)

    new_geom = linemerge(geoms) if geoms else enriched.at[link_idx, "geometry"]

    _pns_manual = ";".join(str(n) for n in path_nodes)
    _via_nodes, _via_segment = _make_via_cols(_pns_manual)
    enriched.at[link_idx, "geometry"]         = new_geom
    enriched.at[link_idx, "Via_Station"]      = ";".join(via_st)
    enriched.at[link_idx, "Via_Junction"]     = ";".join(via_jn)
    enriched.at[link_idx, "path_nodes"]       = _pns_manual
    enriched.at[link_idx, "Via_Nodes"]        = _via_nodes
    enriched.at[link_idx, "Via_Segment"]      = _via_segment
    enriched.at[link_idx, "path_length_m"]    = path_length
    enriched.at[link_idx, "needs_correction"] = False

    print(
        f"  Rerouted: {len(path_nodes)-1} segments, {path_length/1000:.2f} km. "
        f"Row {link_idx} updated."
    )
    return enriched


def _run_phase2(
    config: ProjectionConfig,
    rail_enriched: gpd.GeoDataFrame,
    tram_enriched: gpd.GeoDataFrame,
    func_enriched: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Phase 2: build combined-line outputs for QGIS inspection, then run rerouting TUI.
    Returns updated (rail_enriched, tram_enriched, func_enriched).
    """
    print("\n" + "─" * 60)
    print("  Phase 2 — Corrections")
    print("─" * 60)

    _save_phase2_outputs(config, rail_enriched, tram_enriched, func_enriched)

    print("\n  Open the .qgz files in QGIS to inspect routing and identify any errors before making corrections.")
    if config.auto_mode:
        print("  Auto mode — skipping reroute step.")
        return rail_enriched, tram_enriched, func_enriched
    ans = input("\n  Do you want to reroute any service? (y/n) [n]: ").strip().lower() or "n"
    if ans != "y":
        print("  No corrections made.")
        return rail_enriched, tram_enriched, func_enriched

    # Load graph for corrections
    nodes = gpd.read_file(config.infra_dir / "nodes.gpkg")
    bav_segs = gpd.read_file(config.infra_dir / "segments.gpkg")
    G = build_infra_graph(nodes, bav_segs)
    seg_lookup = build_segment_lookup(nodes, bav_segs)
    node_attrs = build_node_attrs(nodes)

    mode_map = {
        "rail": ("rail", rail_enriched, "Service", "FromStation", "ToStation",
                 "match_method_from", "path_length_m"),
        "tram": ("tram", tram_enriched, "GTFS_ID", "from_stop_name", "to_stop_name",
                 "match_method_from", "path_length_m"),
        "funicular": ("funicular", func_enriched, "GTFS_ID", "from_stop_name",
                      "to_stop_name", "match_method_from", "path_length_m"),
    }

    while True:
        print("\n  Which mode? (rail / tram / funicular / done)")
        mode_input = input("  Mode: ").strip().lower()
        if mode_input in ("done", "d", ""):
            break
        if mode_input not in mode_map:
            print("  Enter rail, tram, funicular, or done.")
            continue

        mode_label, enriched_df, route_col, fn_col, tn_col, meth_col, len_col = (
            mode_map[mode_input]
        )

        svc = input(f"  Service code (e.g. S14, 10, Polybahn): ").strip()
        link_indices = _show_service_stops(
            enriched_df, svc, route_col, fn_col, tn_col, meth_col, len_col
        )
        if not link_indices:
            continue

        raw_from = input(
            "\n  FROM stop number to begin rerouting (or Enter to cancel): "
        ).strip()
        if not raw_from:
            continue
        if not raw_from.isdigit() or not (1 <= int(raw_from) <= len(link_indices)):
            print("  Invalid stop number.")
            continue

        link_pos = int(raw_from) - 1
        link_idx = link_indices[link_pos]
        row = enriched_df.loc[link_idx]
        from_node = row.get("node_id_from")
        to_node = row.get("node_id_to")

        if pd.isna(from_node) or pd.isna(to_node):
            print(
                "  Cannot reroute — from or to node is unmatched. "
                "Fix the matching first."
            )
            continue

        enriched_df = _reroute_link(
            enriched_df, link_idx, int(from_node), int(to_node),
            G, seg_lookup, node_attrs
        )

        # Update the mode map reference
        if mode_input == "rail":
            rail_enriched = enriched_df
            _write_rail_outputs(rail_enriched, config)
        elif mode_input == "tram":
            tram_enriched = enriched_df
            tram_enriched.to_file(
                config.feeder_output_dir / "pt_feeder_segments.gpkg",
                driver="GPKG", layer="tram"
            )
        else:
            func_enriched = enriched_df
            func_enriched.to_file(
                config.feeder_output_dir / "pt_feeder_segments.gpkg",
                driver="GPKG", layer="funicular"
            )
        print("  Changes saved.")
        mode_map[mode_input] = (
            mode_label, enriched_df, route_col, fn_col, tn_col, meth_col, len_col
        )

        ans = input("\n  Reroute another service? (y/n) [n]: ").strip().lower() or "n"
        if ans != "y":
            break

    return rail_enriched, tram_enriched, func_enriched

# =============================================================================
# Phase 3 — Plotting
# =============================================================================

_SCALE_BAR_NICE_KM = [1, 2, 5, 10, 20, 50, 100, 200, 500]

def _extent_from_gdf(gdf, margin_m: int = 2000):
    if gdf is None or gdf.empty:
        return None
    b = gdf.total_bounds
    return (b[0] - margin_m, b[2] + margin_m, b[1] - margin_m, b[3] + margin_m)

def _add_north_arrow(ax, location='upper left', scale=0.5):
    north_arrow(ax, location=location, scale=scale, rotation={"degrees": 0})

def _add_scale_bar(ax, location=(0.72, 0.04)):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    map_w, map_h = xlim[1] - xlim[0], ylim[1] - ylim[0]
    target_km = (map_w / 4.0) / 1000.0
    total_km  = min(_SCALE_BAR_NICE_KM, key=lambda v: abs(v - target_km))
    n_cells   = 4 if total_km >= 4 else 2
    cell_m    = (total_km * 1000.0) / n_cells
    x0, y0 = xlim[0] + map_w * location[0], ylim[0] + map_h * location[1]
    bar_h = map_h * 0.008
    for i in range(n_cells):
        color = 'black' if i % 2 == 0 else 'white'
        ax.add_patch(Rectangle((x0 + i * cell_m, y0), cell_m, bar_h, facecolor=color, edgecolor='black', linewidth=0.6, zorder=7))
    for i in range(n_cells + 1):
        val_km = (i * cell_m) / 1000.0
        label = f'{val_km:.0f} km' if val_km == int(val_km) else f'{val_km:.1f} km'
        ax.text(x0 + i * cell_m, y0 + bar_h * 1.6, label, ha='center', va='bottom', fontsize=7, zorder=7)


def _plot_gazette_style(
    config: ProjectionConfig,
    rail_enriched: gpd.GeoDataFrame,
    tram_enriched: gpd.GeoDataFrame,
    func_enriched: gpd.GeoDataFrame,
    boundary_gpkg: Path,
    boundary_name: str,
    feeder_bg_gdfs: Optional[Dict[str, gpd.GeoDataFrame]] = None,
) -> None:
    """Railway Gazette style plot.

    For each service/direction within the boundary:
    - Draws inside links in mode colour.
    - Exiting links are clipped to the boundary; a stub continues beyond it
      labelled "{Service} → {next stop outside}".
    - Terminated services get a filled circle at the terminus plus the service
      label placed alongside the last inside segment.
    """
    if not boundary_gpkg.exists():
        print(f"  Skipping gazette plot ({boundary_name}) — boundary file not found.")
        return

    main = Path(paths.MAIN)
    boundary_gdf  = gpd.read_file(boundary_gpkg)
    boundary_poly = boundary_gdf.geometry.union_all()
    from shapely.prepared import prep as _prep_boundary
    _bp = _prep_boundary(boundary_poly)
    is_sa         = boundary_name == "study_area"
    extent        = _extent_from_gdf(boundary_gdf, margin_m=2000)

    # ── Infrastructure ────────────────────────────────────────────────────────
    bav_segs = gpd.read_file(config.infra_dir / "segments.gpkg")
    node_gdf = gpd.read_file(config.infra_dir / "nodes.gpkg")

    nc_col = 'Node_Class' if 'Node_Class' in node_gdf.columns else \
             'node_class'  if 'node_class'  in node_gdf.columns else None
    tm_col = 'Transport_Mode' if 'Transport_Mode' in node_gdf.columns else \
             'transport_mode'  if 'transport_mode'  in node_gdf.columns else None

    if nc_col:
        train_stations = node_gdf[node_gdf[nc_col] == 'station']
    else:
        train_stations = node_gdf

    # Rail-only stations used for code labelling (excludes tram/funicular stops)
    if nc_col and tm_col:
        rail_stations = train_stations[
            train_stations[tm_col].astype(str).str.contains('train', case=False, na=False)
        ]
    else:
        rail_stations = train_stations

    train_names = set(train_stations['Name'].tolist())
    bav_segs_filtered = (
        bav_segs[bav_segs['From_Name'].isin(train_names) | bav_segs['To_Name'].isin(train_names)]
        .dropna(subset=['From_Name', 'To_Name'])
        if train_names else bav_segs
    )

    lakes_path = main / paths.LAKES_SHP
    lakes_gdf  = gpd.read_file(lakes_path) if lakes_path.exists() else None

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_aspect("equal")
    ax.set_xlabel('E [m]', fontsize=10)
    ax.set_ylabel('N [m]', fontsize=10)
    ax.grid(True, alpha=0.3)
    boundary_gdf.plot(ax=ax, facecolor='none', edgecolor='black',
                      linewidth=1.5, linestyle='--', alpha=0.6)

    # ── Lakes ─────────────────────────────────────────────────────────────────
    if lakes_gdf is not None:
        try:
            if is_sa and extent is not None:
                from shapely.geometry import box as _sbox_lk
                clip_box = gpd.GeoDataFrame(
                    geometry=[_sbox_lk(extent[0], extent[2], extent[1], extent[3])],
                    crs=SWISS_CRS)
                lakes_clipped = gpd.clip(lakes_gdf, clip_box)
            else:
                lakes_clipped = gpd.clip(lakes_gdf, boundary_gdf)
            if not lakes_clipped.empty:
                lakes_clipped.plot(ax=ax, color="#c8e8f5", linewidth=0.3, edgecolor="#99c4d8")
        except Exception:
            pass

    # ── BAV infrastructure background ─────────────────────────────────────────
    if is_sa and extent is not None:
        from shapely.geometry import box as _sbox
        bbox_gdf = gpd.GeoDataFrame(
            geometry=[_sbox(extent[0], extent[2], extent[1], extent[3])],
            crs=bav_segs.crs if bav_segs.crs else SWISS_CRS)
        try:
            segs_extent = gpd.clip(bav_segs, bbox_gdf)
            if not segs_extent.empty:
                segs_extent.plot(ax=ax, color="#d0d0d0", linewidth=0.4, alpha=0.40, zorder=1)
        except Exception:
            pass
        try:
            segs_inside = gpd.clip(bav_segs_filtered, boundary_gdf)
            if not segs_inside.empty:
                segs_inside.plot(ax=ax, color="#d0d0d0", linewidth=0.5, alpha=1.0, zorder=2)
        except Exception:
            pass
    else:
        try:
            segs_inside = gpd.clip(bav_segs_filtered, boundary_gdf)
            if not segs_inside.empty:
                segs_inside.plot(ax=ax, color="#d0d0d0", linewidth=0.4, zorder=1)
        except Exception:
            pass

    # ── Non-track feeder background (bus / ship) ─────────────────────────────
    _labelled_feeders: set = set()  # (layer, line_short_name) already labelled

    if feeder_bg_gdfs:
        _feeder_batches: Dict[str, List] = defaultdict(list)

        for _flayer, _fgdf in feeder_bg_gdfs.items():
            if _fgdf is None or _fgdf.empty:
                continue
            _fcol = _FEEDER_BG_COLOURS.get(_flayer, _BUS_COLOUR)

            for _, _frow in _fgdf.iterrows():
                _fg = _frow.geometry
                if _fg is None or _fg.is_empty:
                    continue
                try:
                    _fclip = _fg.intersection(boundary_poly)
                except Exception:
                    _fclip = _fg
                if _fclip.is_empty:
                    continue
                _feeder_batches[_fcol].append(_fclip)

            # SA: label bus/ship numbers once per line (direction 0 only)
            if is_sa:
                _svc_col = 'line_short_name' if 'line_short_name' in _fgdf.columns else None
                _dir_col = 'direction_id'    if 'direction_id'    in _fgdf.columns else None
                if _svc_col:
                    for _, _frow in _fgdf.iterrows():
                        if _dir_col and str(_frow.get(_dir_col, '0')) != '0':
                            continue
                        _fsvc = str(_frow.get(_svc_col, '')).strip()
                        _fkey = (_flayer, _fsvc)
                        if not _fsvc or _fkey in _labelled_feeders:
                            continue
                        _labelled_feeders.add(_fkey)
                        _g = _frow.geometry
                        if _g is None or _g.is_empty:
                            continue
                        try:
                            _fc = _g.intersection(boundary_poly)
                        except Exception:
                            _fc = _g
                        if _fc.is_empty:
                            continue
                        _fline = (
                            _fc if _fc.geom_type == 'LineString'
                            else max(_fc.geoms, key=lambda _x: _x.length)
                            if _fc.geom_type == 'MultiLineString' and list(_fc.geoms)
                            else None
                        )
                        if _fline is None or _fline.length < 1:
                            continue
                        _fmid = _fline.interpolate(0.5, normalized=True)
                        try:
                            _fp1  = _fline.interpolate(0.45, normalized=True)
                            _fp2  = _fline.interpolate(0.55, normalized=True)
                            _fang = float(np.degrees(
                                np.arctan2(_fp2.y - _fp1.y, _fp2.x - _fp1.x)))
                            if _fang < -90: _fang += 180
                            if _fang >  90: _fang -= 180
                        except Exception:
                            _fang = 0
                        ax.text(
                            _fmid.x, _fmid.y, str(_fsvc),
                            fontsize=5, color=_fcol, ha="center", va="bottom",
                            rotation=_fang, zorder=6,
                            bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                                      alpha=0.55, edgecolor="none"),
                        )

        for _fbcol, _fbgeoms in _feeder_batches.items():
            gpd.GeoDataFrame({"geometry": _fbgeoms}, crs=SWISS_CRS).plot(
                ax=ax, color=_fbcol, linewidth=_FEEDER_BG_LW,
                alpha=0.75, zorder=2,
            )

    # Normalize rail column names to output schema.
    # Phase 1 ("map" mode) delivers internal names (TrainType, Direction, x_origin, …);
    # "plot" / correction mode loads from the already-written file where _OUTPUT_RENAME
    # has been applied (Service, direction_id, from_stop_E, …).
    if rail_enriched is not None and not rail_enriched.empty and "TrainType" in rail_enriched.columns:
        rail_enriched = rail_enriched.rename(columns=_OUTPUT_RENAME)

    # ── Column schema per mode ────────────────────────────────────────────────
    _MC = {
        "rail": dict(
            gdf=rail_enriched,
            group_cols=["Service", "direction_id"],
            label_col="Service",
            from_name="from_stop_name", to_name="to_stop_name",
            from_e="from_stop_E",       from_n="from_stop_N",
            to_e="to_stop_E",           to_n="to_stop_N",
            from_id="from_stop_nr",     to_id="to_stop_nr",
        ),
        "tram": dict(
            gdf=tram_enriched,
            group_cols=["Service", "direction_id"],
            label_col="Service",
            from_name="from_stop_name", to_name="to_stop_name",
            from_e="from_stop_E",       from_n="from_stop_N",
            to_e="to_stop_E",           to_n="to_stop_N",
            from_id="from_stop_nr",     to_id="to_stop_nr",
        ),
        "funicular": dict(
            gdf=func_enriched,
            group_cols=["Service", "direction_id"],
            label_col="Service",
            from_name="from_stop_name", to_name="to_stop_name",
            from_e="from_stop_E",       from_n="from_stop_N",
            to_e="to_stop_E",           to_n="to_stop_N",
            from_id="from_stop_nr",     to_id="to_stop_nr",
        ),
    }

    # ── Stub length ──────────────────────────────────────────────────────────
    if extent is not None:
        _map_min = min(extent[1] - extent[0], extent[3] - extent[2])
    else:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        _map_min = min(xlim[1] - xlim[0], ylim[1] - ylim[0])
    _preferred_stub = max(375, _map_min * 0.03)

    # ── Per-service rendering ─────────────────────────────────────────────────
    exit_labels:  Dict = {}
    terminus_pts: list = []
    terminus_bpnr: set = set()
    _geom_batches: Dict[Tuple[str, float], List] = defaultdict(list)

    for mode, mc in _MC.items():
        gdf = mc["gdf"]
        if gdf is None or gdf.empty:
            continue

        colour   = MODE_COLOURS[mode]
        lw_solid = 1.8 if mode == "rail" else 0.9
        lw_fall  = 1.2 if mode == "rail" else 0.6

        gc = mc["group_cols"]
        valid_gc = [c for c in gc if c in gdf.columns]
        if not valid_gc:
            continue

        for _, grp in gdf.groupby(valid_gc):
            svc_label = str(grp.iloc[0].get(mc["label_col"], ""))
            if not svc_label:
                continue

            # Terminus: to_id values that never appear as from_id
            from_ids     = set(grp[mc["from_id"]].dropna().astype(str))
            to_ids       = set(grp[mc["to_id"]].dropna().astype(str))
            terminus_ids = to_ids - from_ids
            last_inside_geom = None

            for _, row in grp.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue

                _nc = row.get("needs_correction", False)
                is_fallback = str(_nc).strip().lower() == 'true'
                line_colour = MODE_COLOURS["fallback"] if is_fallback else colour
                lw          = lw_fall if is_fallback else lw_solid

                try:
                    fe  = float(row.get(mc["from_e"], 0) or 0)
                    fn_ = float(row.get(mc["from_n"], 0) or 0)
                    te  = float(row.get(mc["to_e"],   0) or 0)
                    tn_ = float(row.get(mc["to_n"],   0) or 0)
                except (TypeError, ValueError):
                    continue
                from_in = _bp.contains(Point(fe, fn_))
                to_in   = _bp.contains(Point(te, tn_))

                if not from_in and not to_in:
                    continue

                try:
                    clipped = geom.intersection(boundary_poly)
                except Exception:
                    clipped = geom
                if clipped.is_empty:
                    continue

                _geom_batches[(line_colour, lw)].append(clipped)

                to_id_str = str(row.get(mc["to_id"], ""))
                if to_id_str in terminus_ids and from_in:
                    last_inside_geom = clipped

                if from_in and not to_in:
                    to_stop_name = str(row.get(mc["to_name"], "") or "")
                    if not to_stop_name:
                        continue
                    label_text = f"{svc_label} → {to_stop_name}"

                    try:
                        cross = boundary_poly.boundary.intersection(geom)
                    except Exception:
                        continue
                    cp = None
                    if cross.is_empty:
                        continue
                    if cross.geom_type == "Point":
                        cp = cross
                    elif hasattr(cross, "geoms"):
                        pts = [g for g in cross.geoms if g.geom_type == "Point"]
                        if pts:
                            cp = pts[0]
                    if cp is None:
                        continue

                    dx, dy = te - cp.x, tn_ - cp.y
                    d = (dx**2 + dy**2) ** 0.5
                    if d < 1:
                        continue
                    udx, udy = dx / d, dy / d

                    grid_key = (round(cp.x / 100) * 100, round(cp.y / 100) * 100)
                    if grid_key not in exit_labels:
                        exit_labels[grid_key] = {
                            "pt": cp, "texts": [], "colour": line_colour,
                            "udx": udx, "udy": udy, "mode": mode,
                        }
                    exit_labels[grid_key]["texts"].append(label_text)

            for _, row in grp.iterrows():
                to_id_str = str(row.get(mc["to_id"], ""))
                if to_id_str not in terminus_ids:
                    continue
                try:
                    te  = float(row.get(mc["to_e"], 0) or 0)
                    tn_ = float(row.get(mc["to_n"], 0) or 0)
                except (TypeError, ValueError):
                    continue
                if not _bp.contains(Point(te, tn_)):
                    continue
                terminus_pts.append((te, tn_, colour, svc_label, mode, last_inside_geom))
                try:
                    terminus_bpnr.add(int(float(to_id_str)))
                except (ValueError, TypeError):
                    pass

    # Batch-render all collected service geometries (one .plot() per colour/lw bucket)
    for (_col, _lw), _geoms in _geom_batches.items():
        gpd.GeoDataFrame({"geometry": _geoms}, crs=SWISS_CRS).plot(
            ax=ax, color=_col, linewidth=_lw, zorder=3,
        )

    # ── Exit stubs + labels ───────────────────────────────────────────────────
    if extent is not None:
        _xmin, _xmax, _ymin, _ymax = extent[0], extent[1], extent[2], extent[3]
    else:
        _xmin, _xmax = ax.get_xlim(); _ymin, _ymax = ax.get_ylim()

    for cp_info in exit_labels.values():
        if not is_sa and cp_info.get("mode", "rail") != "rail":
            continue  # suppress tram/funicular exit stubs in CA
        cp  = cp_info["pt"]
        ux, uy = cp_info["udx"], cp_info["udy"]
        _margins = []
        if ux > 0  and _xmax > cp.x: _margins.append((_xmax - cp.x) / ux)
        elif ux < 0 and _xmin < cp.x: _margins.append((_xmin - cp.x) / ux)
        if uy > 0  and _ymax > cp.y: _margins.append((_ymax - cp.y) / uy)
        elif uy < 0 and _ymin < cp.y: _margins.append((_ymin - cp.y) / uy)
        avail    = min(_margins) * 0.90 if _margins else _preferred_stub
        stub_len = min(_preferred_stub, max(100, avail))
        stub_end = Point(cp.x + ux * stub_len, cp.y + uy * stub_len)

        stub_gdf = gpd.GeoDataFrame(
            {"geometry": [LineString([(cp.x, cp.y), (stub_end.x, stub_end.y)])]},
            crs=SWISS_CRS)
        stub_gdf.plot(ax=ax, color=cp_info["colour"], linewidth=1.0, linestyle="--", zorder=4)

        ha = "right" if ux < 0 else ("center" if abs(ux) < abs(uy) else "left")
        va = "top"   if uy < 0 else ("center" if abs(uy) < abs(ux) else "bottom")
        ox = (-4 if ux < 0 else (0 if abs(ux) < abs(uy) else 4))
        oy = (-4 if uy < 0 else (0 if abs(uy) < abs(ux) else 4))

        sorted_texts = sorted(set(cp_info["texts"]))
        ax.annotate(
            "\n".join(sorted_texts),
            xy=(stub_end.x, stub_end.y),
            xytext=(ox, oy), textcoords="offset points",
            fontsize=5, color="#333333", va=va, ha=ha,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.75),
            zorder=6,
        )

    # ── Terminus circles + service labels ─────────────────────────────────────
    seen_termini: set = set()
    for (te, tn_, colour, svc_label, t_mode, seg_geom) in terminus_pts:
        coord_key = (round(te / 50) * 50, round(tn_ / 50) * 50)
        if coord_key in seen_termini:
            continue
        seen_termini.add(coord_key)

        if not is_sa and t_mode != "rail":
            continue  # suppress tram/funicular terminus circles and labels in CA

        # Filled circle — feeder termini at 1/3 the size of rail
        _t_ms = 6 if t_mode == "rail" else 2
        ax.plot(te, tn_, marker='o', markersize=_t_ms, color=colour,
                markeredgecolor='white', markeredgewidth=0.8, zorder=8)

        # Service label alongside last inside segment — suppressed for funiculars
        if t_mode == "funicular":
            continue
        if seg_geom is not None and not seg_geom.is_empty:
            try:
                coords = list(seg_geom.coords) if seg_geom.geom_type == "LineString" \
                    else list(seg_geom.geoms[0].coords)
                if len(coords) >= 2:
                    frac  = 0.60
                    idx_f = int(frac * (len(coords) - 1))
                    lx = coords[idx_f][0]
                    ly = coords[idx_f][1]
                    p1 = coords[max(0, idx_f - 1)]
                    p2 = coords[min(len(coords) - 1, idx_f + 1)]
                    sdx, sdy = p2[0] - p1[0], p2[1] - p1[1]
                    slen = (sdx**2 + sdy**2) ** 0.5
                    if slen > 0:
                        px, py = -sdy / slen, sdx / slen
                        lx += px * 350
                        ly += py * 350
                    ax.annotate(
                        svc_label,
                        xy=(lx, ly),
                        fontsize=5, fontweight='bold', color=colour,
                        ha='center', va='center', zorder=8,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                  ec='none', alpha=0.75),
                    )
            except Exception:
                pass

    # ── Train stations ────────────────────────────────────────────────────────
    train_gdf_inside = gpd.clip(train_stations, boundary_gdf)
    inside_idx       = set(train_gdf_inside.index) if not train_gdf_inside.empty else set()
    rail_gdf_inside  = gpd.clip(rail_stations, boundary_gdf)
    rail_inside_idx  = set(rail_gdf_inside.index) if not rail_gdf_inside.empty else set()

    _ms_rail = 30 if is_sa else 20

    if is_sa and extent is not None:
        from shapely.geometry import box as _sbox2
        ext_box = gpd.GeoDataFrame(
            geometry=[_sbox2(extent[0], extent[2], extent[1], extent[3])],
            crs=train_stations.crs if train_stations.crs else SWISS_CRS)
        train_clip = gpd.clip(train_stations, ext_box)
        # Ghost pass (all stations at same size) then solid pass (inside only)
        train_clip.plot(ax=ax, facecolor='white', edgecolor='black',
                        markersize=_ms_rail, marker='o', linewidth=0.8, alpha=0.40, zorder=5)
        train_gdf_inside.plot(ax=ax, facecolor='white', edgecolor='black',
                              markersize=_ms_rail, marker='o', linewidth=0.8, zorder=6)
    else:
        train_clip = train_gdf_inside
        # Rail stations
        rail_gdf_inside.plot(ax=ax, facecolor='white', edgecolor='black',
                             markersize=_ms_rail, marker='o', linewidth=0.8, zorder=5)
        # Feeder stations — omitted in CA (tram/funicular stops not shown at catchment scale)

    # Station code labels: rail stations only.
    # SA → all inside rail stations; CA → only where a service terminates.
    for idx, row in train_clip.iterrows():
        if idx not in inside_idx or idx not in rail_inside_idx:
            continue
        code = str(row.get("Code", "")).strip()
        if not code:
            continue
        if not is_sa:
            # CA: only label where a service terminates
            try:
                bpnr = int(float(row.get("Number", 0)))
            except (TypeError, ValueError):
                continue
            if bpnr not in terminus_bpnr:
                continue
        ax.annotate(
            code,
            xy=(row.geometry.x, row.geometry.y),
            xytext=(5, 5), textcoords="offset points",
            fontsize=7, fontweight='bold', color="#333333", zorder=7,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='none', alpha=0.7),
        )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color=MODE_COLOURS["rail"],     linewidth=2, label="Rail"),
        Line2D([0], [0], color=MODE_COLOURS["tram"],     linewidth=2, label="Tram"),
        Line2D([0], [0], color=MODE_COLOURS["funicular"],linewidth=2, label="Funicular"),
        Line2D([0], [0], color=MODE_COLOURS["fallback"],  linewidth=1.5, linestyle="--",
               label="Straight-line"),
        Line2D([0], [0], color="#d0d0d0", linewidth=1.5, label="Unused infrastructure"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f3a6e',
               markersize=6, label="Terminus"),
    ]
    if feeder_bg_gdfs:
        if any(k in feeder_bg_gdfs for k in ("bus", "express_bus", "ondemand_bus")):
            legend_handles.append(
                Line2D([0], [0], color=_BUS_COLOUR, linewidth=1, label="Bus")
            )
        if "ship" in feeder_bg_gdfs:
            legend_handles.append(
                Line2D([0], [0], color=_SHIP_COLOUR, linewidth=1, label="Ship")
            )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7)

    ax.set_title(
        f"Service Projection — {config.svc_version} on {config.infra_version}"
        f"\nBoundary: {boundary_name}",
        fontsize=14, fontweight='bold',
    )
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    _add_north_arrow(ax, location='upper left', scale=0.5)
    _add_scale_bar(ax, location=(0.755, 0.012))
    plt.tight_layout()

    _lines_subdir = "Feeder_Lines" if feeder_bg_gdfs else "Rail_Lines"
    out_dir = (main / paths.NETWORK_PLOTS_DIR / _lines_subdir
               / config.svc_version / config.infra_version)
    out_dir.mkdir(parents=True, exist_ok=True)
    _fsuffix = "_with_feeders" if feeder_bg_gdfs else ""
    fname    = f"{config.svc_version}_{config.infra_version}_{boundary_name}{_fsuffix}.pdf"
    out_path = out_dir / fname
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


def _compute_seg_freq(
    rail_enriched: gpd.GeoDataFrame,
    freq_type: str,
) -> Dict[Tuple[int, int], float]:
    """Accumulate frequency (dep/hr) per undirected BAV segment edge.

    Direction-deduplicates rail_enriched: prefers direction_id == "0"; includes
    direction "1" only for variants absent from direction 0, so bidirectional
    services are never double-counted.

    Returns {(min_id, max_id): dep_hr}. Only edges with freq > 0 are included.
    """
    pn_col = "path_nodes" if "path_nodes" in rail_enriched.columns else None
    if freq_type == "peak":
        am_col   = "freq_am_peak_dep_hr"
        pm_col   = "freq_pm_peak_dep_hr"
        has_freq = am_col in rail_enriched.columns or pm_col in rail_enriched.columns
    else:
        op_col   = "freq_offpeak_dep_hr"
        has_freq = op_col in rail_enriched.columns

    seg_freq: Dict[Tuple[int, int], float] = {}
    if not pn_col or not has_freq:
        return seg_freq

    freq_source = rail_enriched
    if "direction_id" in rail_enriched.columns:
        key_cols = [c for c in ["Service", "variant_rank"] if c in rail_enriched.columns]
        if key_cols:
            dir0      = rail_enriched[rail_enriched["direction_id"].astype(str) == "0"]
            dir0_keys = set(map(tuple, dir0[key_cols].drop_duplicates().values.tolist()))
            dir1_only = rail_enriched[
                (rail_enriched["direction_id"].astype(str) != "0") &
                (~rail_enriched[key_cols].apply(tuple, axis=1).isin(dir0_keys))
            ]
            freq_source = pd.concat([dir0, dir1_only], ignore_index=True)

    for _, row in freq_source.iterrows():
        pn = str(row.get(pn_col, "") or "")
        if not pn:
            continue
        if freq_type == "peak":
            fam  = row.get(am_col)
            fpm  = row.get(pm_col)
            vals = [v for v in [fam, fpm] if v is not None and pd.notna(v) and float(v) > 0]
            fval = float(sum(vals) / len(vals)) if vals else 0.0
        else:
            fv   = row.get(op_col)
            fval = float(fv) if (fv is not None and pd.notna(fv) and float(fv) > 0) else 0.0
        if fval <= 0:
            continue
        try:
            nids = [int(n) for n in pn.split(";") if n.strip()]
        except ValueError:
            continue
        for i in range(len(nids) - 1):
            ekey = (min(nids[i], nids[i + 1]), max(nids[i], nids[i + 1]))
            seg_freq[ekey] = seg_freq.get(ekey, 0.0) + fval

    return seg_freq


def _plot_frequency_map(
    config: ProjectionConfig,
    rail_enriched: gpd.GeoDataFrame,
    boundary_gpkg: Path,
    boundary_name: str,
    freq_type: str = "offpeak",
) -> None:
    """Rail service frequency map — segment width scaled by departures/hr.

    freq_type: 'offpeak' uses freq_offpeak_dep_hr;
               'peak'    uses mean(freq_am_peak_dep_hr, freq_pm_peak_dep_hr).
    Segment frequencies are summed across all services routing through each edge.
    Raw infrastructure nodes are used as a fallback so that nodes absent from the
    working version (e.g. operational yards healed during projection) are resolved.
    """
    if not boundary_gpkg.exists():
        print(f"  Skipping frequency map ({boundary_name}, {freq_type}) — boundary not found.")
        return

    main_path    = Path(paths.MAIN)
    boundary_gdf = gpd.read_file(boundary_gpkg)
    is_sa        = boundary_name == "study_area"
    extent       = _extent_from_gdf(boundary_gdf, margin_m=2000)

    # ── Infrastructure — extend lookup with raw nodes so healed nodes resolve ──
    nodes_gdf = gpd.read_file(config.infra_dir / "nodes.gpkg")
    segs_gdf  = gpd.read_file(config.infra_dir / "segments.gpkg")

    name_to_id = _build_name_to_id(nodes_gdf)

    # Fallback: raw nodes carry real BAV Numbers for nodes not in the working version
    _raw_nodes_path = config.raw_infra_dir / "nodes.gpkg"
    if _raw_nodes_path.exists():
        _raw_nodes = gpd.read_file(_raw_nodes_path)
        for _, _rn in _raw_nodes.iterrows():
            _rname = _rn.get("Name", "")
            if _rname and _rname not in name_to_id and pd.notna(_rn.get("Number")):
                name_to_id[_rname] = int(_rn["Number"])

    # Build (min_id, max_id) → segment geometry lookup
    seg_geom_lookup: Dict[Tuple[int, int], object] = {}
    for _, _seg in segs_gdf.iterrows():
        _fn = name_to_id.get(_seg.get("From_Name"))
        _tn = name_to_id.get(_seg.get("To_Name"))
        if _fn is not None and _tn is not None and _seg.geometry is not None:
            _key = (min(_fn, _tn), max(_fn, _tn))
            if _key not in seg_geom_lookup:
                seg_geom_lookup[_key] = _seg.geometry

    # ── Sum frequency per segment edge ─────────────────────────────────────────
    seg_freq = _compute_seg_freq(rail_enriched, freq_type)

    # ── Assign frequency bins ──────────────────────────────────────────────────
    def _bin(val: float) -> Tuple[str, float, str]:
        _iv = int(val)
        for _lo, _hi, _bc, _blw, _blbl in _FREQ_BINS:
            if _lo <= _iv <= _hi:
                return _bc, _blw, _blbl
        return _FREQ_BINS[-1][2], _FREQ_BINS[-1][3], _FREQ_BINS[-1][4]

    bin_geoms: Dict[str, List] = defaultdict(list)
    bin_props: Dict[str, Tuple[str, float]] = {}

    for _ekey, _fval in seg_freq.items():
        _geom = seg_geom_lookup.get(_ekey)
        if _geom is None or _geom.is_empty:
            continue
        _bc, _blw, _blbl = _bin(_fval)
        bin_geoms[_blbl].append(_geom)
        bin_props[_blbl] = (_bc, _blw)

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_aspect("equal")
    ax.set_xlabel("E [m]", fontsize=10)
    ax.set_ylabel("N [m]", fontsize=10)
    ax.grid(True, alpha=0.3)
    boundary_gdf.plot(ax=ax, facecolor="none", edgecolor="black",
                      linewidth=1.5, linestyle="--", alpha=0.6)

    _lakes_path = main_path / paths.LAKES_SHP
    if _lakes_path.exists():
        try:
            _lakes = gpd.read_file(_lakes_path)
            if is_sa and extent is not None:
                from shapely.geometry import box as _sbox_fm
                _clip_box = gpd.GeoDataFrame(
                    geometry=[_sbox_fm(extent[0], extent[2], extent[1], extent[3])],
                    crs=SWISS_CRS)
                _lakes_clip = gpd.clip(_lakes, _clip_box)
            else:
                _lakes_clip = gpd.clip(_lakes, boundary_gdf)
            if not _lakes_clip.empty:
                _lakes_clip.plot(ax=ax, color="#c8e8f5", linewidth=0.3, edgecolor="#99c4d8")
        except Exception:
            pass

    try:
        if is_sa and extent is not None:
            from shapely.geometry import box as _sbox_fm2
            _bg_box = gpd.GeoDataFrame(
                geometry=[_sbox_fm2(extent[0], extent[2], extent[1], extent[3])],
                crs=segs_gdf.crs if segs_gdf.crs else SWISS_CRS)
            _bg = gpd.clip(segs_gdf, _bg_box)
        else:
            _bg = gpd.clip(segs_gdf, boundary_gdf)
        if not _bg.empty:
            _bg.plot(ax=ax, color="#d4d4d4", linewidth=0.4, alpha=0.5, zorder=1)
    except Exception:
        segs_gdf.plot(ax=ax, color="#d4d4d4", linewidth=0.4, alpha=0.5, zorder=1)

    for _blbl, _geoms in bin_geoms.items():
        _bc, _blw = bin_props[_blbl]
        gpd.GeoDataFrame({"geometry": _geoms}, crs=SWISS_CRS).plot(
            ax=ax, color=_bc, linewidth=_blw, zorder=3,
        )

    # ── Legend ─────────────────────────────────────────────────────────────────
    _period_label = "Off-peak" if freq_type == "offpeak" else "Peak"
    _legend_handles = [
        Line2D([0], [0], color="#d4d4d4", linewidth=1.5, label="Infrastructure (no service)"),
    ]
    for _, _, _bc, _blw, _blbl in _FREQ_BINS:
        _legend_handles.append(
            Line2D([0], [0], color=_bc, linewidth=_blw * 0.7,
                   label=f"{_blbl} dep / hr")
        )
    ax.legend(handles=_legend_handles, loc="upper right", fontsize=8,
              title=f"Rail frequency\n{_period_label}", title_fontsize=8)

    ax.set_title(
        f"Rail Service Frequency ({_period_label}) — "
        f"{config.svc_version} on {config.infra_version}"
        f"\nBoundary: {boundary_name}",
        fontsize=14, fontweight="bold",
    )
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    _add_north_arrow(ax, location="upper left", scale=0.5)
    _add_scale_bar(ax, location=(0.755, 0.012))
    plt.tight_layout()

    _out_dir = (main_path / paths.NETWORK_PLOTS_DIR / "Rail_Lines"
                / config.svc_version / config.infra_version)
    _out_dir.mkdir(parents=True, exist_ok=True)
    _fname    = (f"frequency_{freq_type}_{config.svc_version}_"
                 f"{config.infra_version}_{boundary_name}.pdf")
    _out_path = _out_dir / _fname
    fig.savefig(_out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Frequency plot saved → {_out_path}")


def _plot_frequency_diff(
    config: ProjectionConfig,
    rail_enriched: gpd.GeoDataFrame,
    boundary_gpkg: Path,
    boundary_name: str,
) -> None:
    """Peak vs off-peak frequency difference map.

    Computes (peak − off-peak) dep/hr per segment. Base is off-peak.
    Green segments gained frequency in peak; red segments lost frequency.
    Unchanged segments (delta = 0) are shown as thin grey for context.
    """
    if not boundary_gpkg.exists():
        print(f"  Skipping frequency diff ({boundary_name}) — boundary not found.")
        return

    main_path    = Path(paths.MAIN)
    boundary_gdf = gpd.read_file(boundary_gpkg)
    is_sa        = boundary_name == "study_area"
    extent       = _extent_from_gdf(boundary_gdf, margin_m=2000)

    # ── Infrastructure ────────────────────────────────────────────────────────
    nodes_gdf = gpd.read_file(config.infra_dir / "nodes.gpkg")
    segs_gdf  = gpd.read_file(config.infra_dir / "segments.gpkg")

    name_to_id = _build_name_to_id(nodes_gdf)
    _raw_nodes_path = config.raw_infra_dir / "nodes.gpkg"
    if _raw_nodes_path.exists():
        _raw_nodes = gpd.read_file(_raw_nodes_path)
        for _, _rn in _raw_nodes.iterrows():
            _rname = _rn.get("Name", "")
            if _rname and _rname not in name_to_id and pd.notna(_rn.get("Number")):
                name_to_id[_rname] = int(_rn["Number"])

    seg_geom_lookup: Dict[Tuple[int, int], object] = {}
    for _, _seg in segs_gdf.iterrows():
        _fn = name_to_id.get(_seg.get("From_Name"))
        _tn = name_to_id.get(_seg.get("To_Name"))
        if _fn is not None and _tn is not None and _seg.geometry is not None:
            _key = (min(_fn, _tn), max(_fn, _tn))
            if _key not in seg_geom_lookup:
                seg_geom_lookup[_key] = _seg.geometry

    # ── Compute per-segment frequencies ──────────────────────────────────────
    seg_offpeak = _compute_seg_freq(rail_enriched, "offpeak")
    seg_peak    = _compute_seg_freq(rail_enriched, "peak")

    all_keys = set(seg_offpeak) | set(seg_peak)
    seg_delta: Dict[Tuple[int, int], float] = {
        k: seg_peak.get(k, 0.0) - seg_offpeak.get(k, 0.0)
        for k in all_keys
    }

    # ── Classify into bins ────────────────────────────────────────────────────
    def _diff_bin(val: float):
        iv = int(val)
        for lo, hi, col, lw, lbl in _DIFF_BINS:
            if lo <= iv <= hi:
                return col, lw, lbl
        return None, None, None

    unchanged_geoms = []
    bin_geoms: Dict[str, list] = defaultdict(list)
    bin_props: Dict[str, Tuple[str, float]] = {}

    for ekey, delta in seg_delta.items():
        geom = seg_geom_lookup.get(ekey)
        if geom is None or geom.is_empty:
            continue
        if abs(delta) < 0.5:
            unchanged_geoms.append(geom)
        else:
            col, lw, lbl = _diff_bin(delta)
            if lbl is not None:
                bin_geoms[lbl].append(geom)
                bin_props[lbl] = (col, lw)

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_aspect("equal")
    ax.set_xlabel("E [m]", fontsize=10)
    ax.set_ylabel("N [m]", fontsize=10)
    ax.grid(True, alpha=0.3)
    boundary_gdf.plot(ax=ax, facecolor="none", edgecolor="black",
                      linewidth=1.5, linestyle="--", alpha=0.6)

    _lakes_path = main_path / paths.LAKES_SHP
    if _lakes_path.exists():
        try:
            _lakes = gpd.read_file(_lakes_path)
            if is_sa and extent is not None:
                from shapely.geometry import box as _sbox_fd
                _clip_box = gpd.GeoDataFrame(
                    geometry=[_sbox_fd(extent[0], extent[2], extent[1], extent[3])],
                    crs=SWISS_CRS)
                _lakes_clipped = gpd.clip(_lakes, _clip_box)
            else:
                _lakes_clipped = gpd.clip(_lakes, boundary_gdf)
            if not _lakes_clipped.empty:
                _lakes_clipped.plot(ax=ax, color="#c8e8f5", linewidth=0.3, edgecolor="#99c4d8")
        except Exception:
            pass

    # Unchanged (grey background context)
    if unchanged_geoms:
        gpd.GeoDataFrame({"geometry": unchanged_geoms}, crs=SWISS_CRS).plot(
            ax=ax, color="#cccccc", linewidth=0.6, alpha=0.7, zorder=2)

    # Changed segments by bin (losses drawn before gains so gains sit on top)
    _legend_handles = []
    loss_labels = [lbl for lo, _, _, _, lbl in _DIFF_BINS if lo < 0]
    gain_labels = [lbl for lo, _, _, _, lbl in _DIFF_BINS if lo > 0]

    for lbl in loss_labels + gain_labels:
        if lbl not in bin_geoms:
            continue
        col, lw = bin_props[lbl]
        gpd.GeoDataFrame({"geometry": bin_geoms[lbl]}, crs=SWISS_CRS).plot(
            ax=ax, color=col, linewidth=lw, zorder=3)
        _legend_handles.append(
            Line2D([0], [0], color=col, linewidth=lw, label=f"{lbl} dep/hr"))

    # Legend: losses first, then gains
    _legend_handles = (
        [Line2D([0], [0], color="#cccccc", linewidth=0.6, label="0 (unchanged)")]
        + _legend_handles
    )
    ax.legend(handles=_legend_handles, loc="upper right", fontsize=8,
              title="Δ dep/hr (peak − off-peak)", title_fontsize=8)

    ax.set_title(
        f"Rail Frequency Change: Peak vs Off-Peak — "
        f"{config.svc_version} on {config.infra_version}"
        f"\nBoundary: {boundary_name}",
        fontsize=14, fontweight="bold",
    )
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

    _add_north_arrow(ax, location="upper left", scale=0.5)
    _add_scale_bar(ax, location=(0.755, 0.012))
    plt.tight_layout()

    _out_dir = (main_path / paths.NETWORK_PLOTS_DIR / "Rail_Lines"
                / config.svc_version / config.infra_version)
    _out_dir.mkdir(parents=True, exist_ok=True)
    _fname    = (f"frequency_diff_{config.svc_version}_"
                 f"{config.infra_version}_{boundary_name}.pdf")
    _out_path = _out_dir / _fname
    fig.savefig(_out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Frequency diff plot saved → {_out_path}")


def _run_phase3(
    config: ProjectionConfig,
    rail_enriched: gpd.GeoDataFrame,
    tram_enriched: gpd.GeoDataFrame,
    func_enriched: gpd.GeoDataFrame,
) -> None:
    """Phase 3: produce overview plots for study area and catchment area."""
    print("\n" + "─" * 60)
    print("  Phase 3 — Plotting")
    print("─" * 60)

    main = Path(paths.MAIN)

    # Check Catchment_Area folder first for study_area_boundary (per user paths)
    study_area_boundary = main / paths.CATCHMENT_AREA_DIR / "study_area_boundary.gpkg"
    if not study_area_boundary.exists():
        study_area_boundary = main / paths.STUDY_AREA_BOUNDARY_GPKG

    # Load bus / ship feeder backgrounds for the overlay variant (line-level)
    _feeder_bg: Dict[str, gpd.GeoDataFrame] = {}
    _feeder_lines_path = config.feeder_output_dir / "pt_feeder_lines.gpkg"
    if _feeder_lines_path.exists():
        for _layer in ("bus", "express_bus", "ondemand_bus", "ship"):
            try:
                _gdf = gpd.read_file(_feeder_lines_path, layer=_layer)
                if not _gdf.empty:
                    _feeder_bg[_layer] = _gdf
            except Exception:
                pass  # layer absent in this service version — skip

    for boundary_path, label in [
        (study_area_boundary, "study_area"),
        (main / paths.CATCHMENT_AREA_BOUNDARY_GPKG, "catchment_area"),
    ]:
        print(f"\n  Plotting {label}...")
        _plot_gazette_style(
            config, rail_enriched, tram_enriched, func_enriched,
            boundary_path, label,
        )
        if _feeder_bg and config.include_feeder_plots:
            _plot_gazette_style(
                config, rail_enriched, tram_enriched, func_enriched,
                boundary_path, label,
                feeder_bg_gdfs=_feeder_bg,
            )
        _plot_frequency_map(config, rail_enriched, boundary_path, label, freq_type="offpeak")
        _plot_frequency_map(config, rail_enriched, boundary_path, label, freq_type="peak")
        _plot_frequency_diff(config, rail_enriched, boundary_path, label)

    print("\n  Phase 3 complete.")

# =============================================================================
# Main
# =============================================================================

def _run_phase0_auto(svc_version: str, infra_version: str, include_feeder_plots: bool = True, include_plots: bool = True, plot_only: bool = False) -> Optional[Tuple[ProjectionConfig, str]]:
    """Non-interactive Phase 0: mode=map, source=Unprojected.

    Args:
        svc_version:           Service version name WITHOUT '_network' suffix.
        infra_version:         Infrastructure version name (e.g. 'AS_2026_ZH').
        include_feeder_plots:  When False, skip the _with_feeders plot variant.
        include_plots:         When False, skip Phase 3 entirely (driven by PLOT_SERVICES).
        plot_only:             When True, load existing projected data and run Phase 3 only.
    """
    main = Path(paths.MAIN)
    svc_folder = svc_version + '_network'

    raw_dirs = _list_raw_dirs()
    if not raw_dirs:
        print("  ERROR: No Raw folder found in data/Infrastructure/.")
        return None
    chosen_raw = raw_dirs[0]
    raw_infra_dir = main / paths.NETWORK_INFRASTRUCTURE_DIR / chosen_raw

    available_svc = _list_svc_versions()
    if svc_folder not in available_svc:
        print(f"  ERROR: Service version '{svc_folder}' not found in {paths.FEEDER_LINES_DIR}.")
        print(f"  Available: {available_svc}")
        return None

    available_infra = _list_infra_versions()
    if infra_version not in available_infra:
        print(f"  ERROR: Infrastructure version '{infra_version}' not found.")
        print(f"  Available: {available_infra}")
        return None

    source_feeder_path = main / paths.FEEDER_LINES_DIR / svc_folder / paths.SERVICES_UNPROJECTED_SUBDIR
    source_rail_path   = main / paths.RAIL_LINES_DIR   / svc_folder / paths.SERVICES_UNPROJECTED_SUBDIR

    rail_output_dir   = main / paths.RAIL_LINES_DIR   / svc_folder / infra_version
    feeder_output_dir = main / paths.FEEDER_LINES_DIR / svc_folder / infra_version
    rail_output_dir.mkdir(parents=True, exist_ok=True)
    feeder_output_dir.mkdir(parents=True, exist_ok=True)

    _include_feeder_plots = include_feeder_plots

    config = ProjectionConfig(
        infra_version=infra_version,
        svc_version=svc_folder,
        infra_dir=main / paths.NETWORK_INFRASTRUCTURE_DIR / infra_version,
        svc_dir=source_feeder_path,
        rail_input=source_rail_path / "rail_segments.gpkg",
        rail_output_dir=rail_output_dir,
        feeder_output_dir=feeder_output_dir,
        raw_infra_dir=raw_infra_dir,
        auto_mode=True,
        include_feeder_plots=_include_feeder_plots,
        include_plots=include_plots,
    )

    mode = "plot" if plot_only else "map"
    print(f"\n  Auto-config (non-interactive):")
    print(f"  Service version : {svc_folder}")
    print(f"  Source          : {'Projected (plot only)' if plot_only else 'Unprojected'}")
    print(f"  Infrastructure  : {infra_version}")
    print(f"  Rail output     : {rail_output_dir}")
    print(f"  Feeder output   : {feeder_output_dir}")
    return config, mode


def main() -> None:
    """Entry point for the service projection pipeline."""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--svc-version',      default=None,
                        help='Service version name (without _network suffix)')
    parser.add_argument('--infra-version',    default=None,
                        help='Infrastructure version name')
    parser.add_argument('--no-feeder-plots',  action='store_true', default=False,
                        help='Skip the _with_feeders plot variant')
    parser.add_argument('--no-plots',         action='store_true', default=False,
                        help='Skip Phase 3 plots entirely (set by main_new when PLOT_SERVICES=False)')
    parser.add_argument('--plot-only',        action='store_true', default=False,
                        help='Load existing projected data and run Phase 3 only; skip projection')
    args, _ = parser.parse_known_args()

    if not _check_prerequisites():
        raise SystemExit(1)

    if args.svc_version and args.infra_version:
        result = _run_phase0_auto(
            args.svc_version, args.infra_version,
            include_feeder_plots=not args.no_feeder_plots,
            include_plots=not args.no_plots,
            plot_only=args.plot_only,
        )
        if result is None:
            raise SystemExit(1)
    else:
        result = _run_phase0()
        if result is None:
            raise SystemExit(0)

    config, mode = result

    if mode == "map":
        rail_enriched, track_feeder_enriched, non_track_feeder_processed = _run_phase1(config)
        tram_enriched = track_feeder_enriched.get("tram",      gpd.GeoDataFrame())
        func_enriched = track_feeder_enriched.get("funicular", gpd.GeoDataFrame())
        rail_enriched = _run_phase1_5(config, rail_enriched)
        print("\n  Writing rail outputs...")
        _write_rail_outputs(rail_enriched, config)
        print("\n  Writing feeder outputs...")
        _write_feeder_outputs(config, track_feeder_enriched, non_track_feeder_processed)
    else:
        # Load existing projection for correction
        rail_out = config.rail_output_dir / "rail_segments.gpkg"
        feeder_out = config.feeder_output_dir / "pt_feeder_segments.gpkg"
        if not rail_out.exists() or not feeder_out.exists():
            print(f"\n  ERROR: Projected files not found in {config.rail_output_dir}.")
            raise SystemExit(1)
        print(f"\n  Loading existing projection...")
        import fiona as _fiona
        _rail_layers = _fiona.listlayers(rail_out)
        _rail_gdfs   = []
        for _lyr in _rail_layers:
            _gdf = gpd.read_file(rail_out, layer=_lyr)
            _gdf["_source_layer"] = _lyr
            _rail_gdfs.append(_gdf)
        rail_enriched = (
            gpd.GeoDataFrame(pd.concat(_rail_gdfs, ignore_index=True), crs=SWISS_CRS)
            if _rail_gdfs else gpd.GeoDataFrame()
        )
        tram_enriched = gpd.read_file(feeder_out, layer="tram")
        func_enriched = gpd.read_file(feeder_out, layer="funicular")
        print(
            f"  Loaded: {len(rail_enriched)} rail links, "
            f"{len(tram_enriched)} tram segments, "
            f"{len(func_enriched)} funicular segments."
        )
        if mode != "plot":
            rail_enriched = _run_phase1_5(config, rail_enriched)
            print("\n  Writing rail outputs...")
            _write_rail_outputs(rail_enriched, config)

    if mode != "plot":
        rail_enriched, tram_enriched, func_enriched = _run_phase2(
            config, rail_enriched, tram_enriched, func_enriched
        )
    if config.include_plots:
        _run_phase3(config, rail_enriched, tram_enriched, func_enriched)
    else:
        print("\n  Phase 3 — Plotting skipped (PLOT_SERVICES = False).")

    print("\n" + "─" * 60)
    print("  Service projection complete.")
    print(f"  Rail output    : {config.rail_output_dir}")
    print(f"  Feeder output  : {config.feeder_output_dir}")
    print("─" * 60)


if __name__ == "__main__":
    main()
