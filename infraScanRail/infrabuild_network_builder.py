"""
Network Builder Module

Two-phase interactive pipeline:

  Phase 1 (Build Base) — optional
    Loads the raw filtered network from data/Infrastructure/Raw/, applies macroscopic simplification, and writes the selectable
    Base version to data/Infrastructure/Base/. Raw/ is produced by infrabuild_filter_network.py.

  Phase 2 (Load & Analyse)
    Selects a prepared version from data/Infrastructure/<version>/, builds a NetworkX graph, and optionally generates infrastructure plots.
    Named versions (beyond Base) are produced by infrabuild_version_manager.py.

"""

import os
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter, deque
from shapely.ops import linemerge, unary_union, substring as shp_substring
from shapely.geometry import MultiLineString, LineString
import math
import sys
import uuid
import zipfile
from matplotlib.patches import Rectangle
from matplotlib_map_utils.core.north_arrow import NorthArrow, north_arrow

sys.path.insert(0, str(Path(__file__).parent))
import paths
import settings


# =============================================================================
# Color Schemes
# =============================================================================

GAUGE_COLORS = {
    1435: '#000000',  # black           – Standard gauge
    1668: '#B8860B',  # dark goldenrod  – Iberian gauge
    1520: '#8B0000',  # dark red        – Russian/broad gauge
    1000: '#7B00D4',  # violet          – Metre gauge
    1067: "#0011FF",  # dark blue       - Cape gauge
    1676: "#D6F814",  # orange-red
    1600: '#FF6347',  # tomato
    600:  '#FF69B4',  # hot pink
}
GAUGE_DEFAULT = '#808080'

# Category-based electrification colours (keyed by canonical class).
ELECTRIFICATION_COLORS = {
    'no_electrification': '#000000',  # black       – nicht elektrifiziert
    'dc':                 "#046DAA",  # light blue  – Gleichstrom
    'ac_16_7hz':          '#2ca02c',  # green       – Wechselstrom 15 kV / 16.7 Hz
    'ac_25kv':            '#d62728',  # red         – Wechselstrom 25 kV
    'unknown':            '#7f7f7f',  # grey
}
ELECTRIFICATION_LABELS = {
    'no_electrification': 'No electrification',
    'dc':                 'DC',
    'ac_16_7hz':          'AC 15 kV / 16.7 Hz',
    'ac_25kv':            'AC 25 kV / 50 Hz',
    'unknown':            'Unknown',
}

# Maps harmonised Electrification_Class values → ELECTRIFICATION_COLORS keys.
# AC_16.7Hz_DC is plotted as ac_16_7hz (primary system on mixed segments).
_ELEC_CLASS_PLOT = {
    'AC_16.7Hz':     'ac_16_7hz',
    'DC':            'dc',
    'AC_16.7Hz_DC':  'ac_16_7hz',
    'non_electrified': 'no_electrification',
    'unknown':       'unknown',
}



# Speed colour bins (upper bound km/h → colour).  Ordered from slow to fast.
# Segments with no speed data (NaN) use SPEED_NO_DATA_COLOR.
SPEED_BINS: list = [
    (30,  '#0000AA', '≤ 30 km/h'),
    (60,  '#0066FF', '31–60 km/h'),
    (80,  '#00AA44', '61–80 km/h'),
    (100, '#AACC00', '81–100 km/h'),
    (120, '#FFAA00', '101–120 km/h'),
    (160, '#FF4400', '121–160 km/h'),
    (999, '#CC0000', '161+ km/h'),
]
SPEED_NO_DATA_COLOR = '#888888'
SPEED_NO_DATA_LABEL = 'No data'


TRACK_COLORS = {
    1: '#FF0000',
    2: '#0000FF',
    3: '#00FF00',
    4: '#800080',
}
TRACK_DEFAULT = '#FFA500'

CONSTRUCT_COLORS = {
    'tunnel':  '#8B4513',
    'bridge':  '#4169E1',
    'gallery': '#708090',
    'normal':  '#2E8B57',
}
CONSTRUCT_DEFAULT = '#808080'

# High-contrast palette for owner map — 30 perceptually distinct colours.
# SBB always gets '#E3000F'; remaining owners cycle through this list.
_OWNER_PALETTE = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b',
    '#e377c2', '#17becf', '#bcbd22', '#393b79', '#637939',
    '#8c6d31', '#843c39', '#7b4173', '#5254a3', '#6b6ecf',
    '#b5cf6b', '#cedb9c', '#e7969c', '#de9ed6', '#3182bd',
    '#31a354', '#756bb1', '#636363', '#fd8d3c', '#fdae6b',
    '#c7e9c0', '#dadaeb', '#fee6ce', '#deebf7', '#d9d9d9',
]


# =============================================================================
# OSM Maxspeed Enrichment Constants
# =============================================================================

OSM_BUFFER_M         = 10     # buffer radius (m) around each BAV segment
OSM_MAX_BEARING_DIFF = 45     # degrees — max angular divergence to accept
OSM_NODE_SNAP_M      = 15     # BAV node snap radius for conflict resolution
OSM_COVERAGE_MIN     = 0.10   # min coverage fraction to record speed

# BAV gauge (mm) → compatible OSM railway types.
# Segments with gauge not in this dict (or gauge = None) accept all types.
GAUGE_TO_OSM_TYPES: Dict[int, set] = {
    1435: {"rail", "light_rail", "subway"},
    1000: {"tram", "narrow_gauge", "light_rail"},
    900:  {"narrow_gauge", "funicular"},
    800:  {"narrow_gauge", "funicular"},
    750:  {"narrow_gauge", "funicular"},
    600:  {"narrow_gauge", "funicular"},
    1520: {"rail"},
    1668: {"rail"},
}
_ALL_OSM_TYPES = {t for s in GAUGE_TO_OSM_TYPES.values() for t in s}

# Mode-to-OSM-type mapping used after transport_mode propagation.
# Tighter than GAUGE_TO_OSM_TYPES: distinguishes 1000 mm trains from trams.
# Gauge is kept as fallback only for NaN-mode segments.
MODE_TO_OSM_TYPES: Dict[str, set] = {
    "train":       {"rail"},
    "tram":        {"tram", "light_rail"},
    "funicular":   {"funicular"},
    "cog_railway": {"narrow_gauge"},
}
_OSM_COLS = [
    "osm_id", "railway_type", "maxspeed",
    "maxspeed_forward", "maxspeed_backward", "osm_name",
]


NODE_COLORS = {
    'station':           '#FF0000',
    'abandoned_station': '#888888',
    'junction':          '#0000FF',
}
NODE_DEFAULT = '#AAAAAA'


# =============================================================================
# Data Class
# =============================================================================

@dataclass
class NetworkData:
    """Container for a built infrastructure version."""
    nodes:    gpd.GeoDataFrame
    segments: gpd.GeoDataFrame
    graph:    nx.Graph
    version:  str
    boundary: Optional[gpd.GeoDataFrame] = field(default=None)

    def __repr__(self):
        return (
            f"NetworkData(version='{self.version}': "
            f"{len(self.nodes)} nodes, {len(self.segments)} segments, "
            f"{self.graph.number_of_edges()} edges)"
        )


# =============================================================================
# Version Discovery
# =============================================================================

def list_versions(infra_dir: Optional[str] = None) -> List[str]:
    """
    Scan data/Infrastructure/ for selectable network versions.

    A subfolder qualifies when it contains both nodes.gpkg and segments.gpkg.
    Raw/ is excluded (intermediate artifact, not a selectable network state).

    Returns version names sorted alphabetically, with 'Base' first when present.
    """
    root = Path(infra_dir or (Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR))
    if not root.exists():
        return []

    versions = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.startswith('Raw'):
            continue
        if (sub / 'nodes.gpkg').exists() and (sub / 'segments.gpkg').exists():
            versions.append(sub.name)

    # Base-type folders sort first, with 'Base' before 'Base_*'
    base_versions  = ['Base'] if 'Base' in versions else []
    base_versions += sorted([v for v in versions if v.startswith('Base') and v != 'Base'])
    other_versions = [v for v in versions if not v.startswith('Base')]
    return base_versions + other_versions


def list_raw_dirs(infra_dir: Optional[str] = None) -> List[str]:
    """
    Scan data/Infrastructure/ for Raw-type folders.

    A subfolder qualifies when its name starts with 'Raw' and contains
    both nodes.gpkg and segments.gpkg (written by infrabuild_filter_network.py).

    Returns folder names sorted alphabetically, with 'Raw' first when present.
    """
    root = Path(infra_dir or (Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR))
    if not root.exists():
        return []

    dirs = sorted([
        sub.name for sub in root.iterdir()
        if sub.is_dir()
        and sub.name.startswith('Raw')
        and (sub / 'nodes.gpkg').exists()
        and (sub / 'segments.gpkg').exists()
    ])
    if 'Raw' in dirs:
        dirs = ['Raw'] + [d for d in dirs if d != 'Raw']
    return dirs


# =============================================================================
# Loading
# =============================================================================

def load_version(
    version: str,
    infra_dir: Optional[str] = None,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load nodes.gpkg and segments.gpkg for a named version."""
    root = Path(infra_dir or (Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR))
    folder = root / version

    nodes_path = folder / "nodes.gpkg"
    segs_path  = folder / "segments.gpkg"

    if not nodes_path.exists():
        raise FileNotFoundError(f"nodes.gpkg not found in {folder}")
    if not segs_path.exists():
        raise FileNotFoundError(f"segments.gpkg not found in {folder}")

    nodes    = gpd.read_file(nodes_path)
    segments = gpd.read_file(segs_path)

    # GeoPackage can serialise numeric columns as string on round-trip.
    # Coerce here so every downstream consumer sees proper numeric dtypes.
    _SEG_NUM = ['Gauge', 'Length', 'Average_Speed', 'Predominant_Speed',
                'Speed_Coverage_Pct', 'Track_Count']
    for _col in _SEG_NUM:
        if _col in segments.columns:
            segments[_col] = pd.to_numeric(segments[_col], errors='coerce')

    _NODE_NUM = ['Number', 'E', 'N']
    for _col in _NODE_NUM:
        if _col in nodes.columns:
            nodes[_col] = pd.to_numeric(nodes[_col], errors='coerce')

    print(f"Loaded '{version}': {len(nodes)} nodes, {len(segments)} segments")
    return nodes, segments


# =============================================================================
# Macroscopic Simplification
# =============================================================================



def _sub_lines(geom):
    """Return list of LineString parts from a LineString or MultiLineString."""
    if geom.geom_type == 'MultiLineString':
        return list(geom.geoms)
    return [geom]


def _reverse_line(geom):
    """Reverse coordinate order of a LineString."""
    return LineString(list(geom.coords)[::-1])


def _build_merged_segment(s1: pd.Series, s2: pd.Series, node_name: str,
                           nodes: Optional[gpd.GeoDataFrame] = None) -> Optional[dict]:
    """
    Merge two segments sharing node_name into a single segment A→B.
    Returns None when the merge would create a self-loop (A == B).
    """
    if s1['From_Name'] == node_name:
        A = s1['To_Name']
        A_N, A_E = s1['To_N'], s1['To_E']
        lines1 = [_reverse_line(g) for g in reversed(_sub_lines(s1.geometry))]
    else:
        A = s1['From_Name']
        A_N, A_E = s1['From_N'], s1['From_E']
        lines1 = _sub_lines(s1.geometry)

    if s2['To_Name'] == node_name:
        B = s2['From_Name']
        B_N, B_E = s2['From_N'], s2['From_E']
        lines2 = [_reverse_line(g) for g in reversed(_sub_lines(s2.geometry))]
    else:
        B = s2['To_Name']
        B_N, B_E = s2['To_N'], s2['To_E']
        lines2 = _sub_lines(s2.geometry)

    if A == B:
        return None

    merged_geom = linemerge(MultiLineString(lines1 + lines2))

    # Derive Number and Code from surviving endpoint names
    number = pd.NA
    code   = pd.NA
    if nodes is not None and not nodes.empty:
        name_to_num  = nodes.dropna(subset=['Name']).set_index('Name')['Number'].to_dict()
        name_to_code = nodes.dropna(subset=['Name']).set_index('Name')['Code'].to_dict()
        a_num  = name_to_num.get(A)
        b_num  = name_to_num.get(B)
        a_code = name_to_code.get(A)
        b_code = name_to_code.get(B)
        if pd.notna(a_num) and pd.notna(b_num):
            number = f"{int(a_num)}_{int(b_num)}"
        if pd.notna(a_code) and pd.notna(b_code):
            code = f"{a_code}_{b_code}"

    return {
        'Segment_ID':          f"{s1['Segment_ID']}+{s2['Segment_ID']}",
        'Number':              number,
        'Code':                code,
        'From_Name':           A,
        'To_Name':             B,
        'From_N':              A_N,
        'From_E':              A_E,
        'To_N':                B_N,
        'To_E':                B_E,
        'Length':              s1['Length'] + s2['Length'],
        'Num_Tracks':          s1['Num_Tracks'],
        'Gauge':               s1['Gauge'],
        'Electrification_Class': s1['Electrification_Class'],
        'Km_Start':            s1.get('Km_Start', pd.NA),
        'Km_End':              s2.get('Km_End', pd.NA),
        'Route_Number':        s1.get('Route_Number', pd.NA),
        'Route_Name':          s1.get('Route_Name', pd.NA),
        'Route_Owner':         s1.get('Route_Owner', pd.NA),
        'Tunnel_Length':       s1.get('Tunnel_Length', 0.0) + s2.get('Tunnel_Length', 0.0),
        'Bridge_Length':       s1.get('Bridge_Length', 0.0) + s2.get('Bridge_Length', 0.0),
        'Conventional_Length': s1.get('Conventional_Length', 0.0) + s2.get('Conventional_Length', 0.0),
        'geometry':            merged_geom,
    }


def filter_macroscopic_nodes(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Simplify raw filtered network to macroscopic representation.

    Removal candidates
    ------------------
    freight_yard, operational_yard, depot, km_change  – always removed
    switch                                             – removed if degree < 3

    Degree rules
    ------------
    degree 0–1 : node + dangling segment dropped.
    degree 2   : flanking segments merged when num_tracks + gauge +
                 electrification all match; otherwise kept as pass-through.
    degree >= 3: kept (topological junction).
    """
    print("Applying macroscopic simplification...")

    segs_df = segments.copy().reset_index(drop=True)

    degree: Counter = Counter()
    for fn, tn in zip(segs_df['From_Name'], segs_df['To_Name']):
        degree[fn] += 1
        degree[tn] += 1

    _MACRO_KEEP = {'station', 'junction', 'abandoned_station'}

    def _is_candidate(row) -> bool:
        nc   = str(row.get('Node_Class', ''))
        name = row.get('Name')
        if pd.isna(name):
            return False
        if nc not in _MACRO_KEEP:
            return True
        return False

    candidate_names: set = set(
        nodes.loc[nodes.apply(_is_candidate, axis=1), 'Name'].dropna()
    )
    print(f"  Removal candidates: {len(candidate_names)}")

    drop_names:   set  = set()
    segs_to_add:  list = []
    segs_to_drop: set  = set()
    n_merged = 0
    n_passthrough = 0

    for name in candidate_names:
        deg = degree[name]
        
        # Drop nodes by default if they are candidates
        drop_names.add(name)

        if deg <= 1:
            dangling = segs_df[
                (segs_df['From_Name'] == name) | (segs_df['To_Name'] == name)
            ]
            segs_to_drop.update(dangling['Segment_ID'])
            continue

        if deg >= 3:
            n_passthrough += 1
            continue

        connected = segs_df[
            (segs_df['From_Name'] == name) | (segs_df['To_Name'] == name)
        ]
        available = connected[~connected['Segment_ID'].isin(segs_to_drop)]
        if len(available) != 2:
            n_passthrough += 1
            continue

        s1, s2 = available.iloc[0], available.iloc[1]
        attrs_match = (
            s1['Num_Tracks']          == s2['Num_Tracks'] and
            s1['Gauge']               == s2['Gauge'] and
            s1['Electrification_Class'] == s2['Electrification_Class']
        )

        if attrs_match:
            merged = _build_merged_segment(s1, s2, name, nodes)
            if merged is None:
                n_passthrough += 1
                continue
            segs_to_add.append(merged)
            segs_to_drop.update([s1['Segment_ID'], s2['Segment_ID']])
            n_merged += 1
        else:
            n_passthrough += 1

    print(f"  Merged: {n_merged}  |  Pass-through: {n_passthrough}  "
          f"|  Dead-end dropped: {len(drop_names) - n_merged}")

    macro_nodes = (
        nodes[~nodes['Name'].isin(drop_names)].copy().reset_index(drop=True)
    )
    remaining = segs_df[~segs_df['Segment_ID'].isin(segs_to_drop)].copy()
    if segs_to_add:
        added = gpd.GeoDataFrame(segs_to_add, crs=segments.crs)
        macro_segs = pd.concat([remaining, added], ignore_index=True)
    else:
        macro_segs = remaining

    # Build segment ID remap: raw_ID → final_macro_ID.
    # Used by run_build_base to update segment references in segments_composition.
    seg_id_remap: dict = {}
    for sid in remaining['Segment_ID'].dropna():
        seg_id_remap[str(sid)] = str(sid)
    for merged_row in segs_to_add:
        merged_id = str(merged_row['Segment_ID'])
        for part in merged_id.split('+'):
            seg_id_remap[part] = merged_id

    print(f"  → {len(macro_nodes)} nodes, {len(macro_segs)} segments")
    return macro_nodes, macro_segs.reset_index(drop=True), seg_id_remap


# =============================================================================
# OSM Maxspeed Enrichment
# =============================================================================

def _parse_speed(val) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    try:
        return float(str(val).replace("km/h", "").replace("mph", "").strip())
    except ValueError:
        return np.nan


def _best_speed(row) -> float:
    for col in ("maxspeed", "maxspeed_forward", "maxspeed_backward"):
        v = _parse_speed(row.get(col))
        if not np.isnan(v):
            return v
    return np.nan


def _compute_osm_bearing(geom) -> float:
    """Overall orientation [0, 180) using start-to-end direction."""
    if geom is None or geom.is_empty:
        return np.nan
    if geom.geom_type in ("MultiLineString", "GeometryCollection"):
        lines = [g for g in geom.geoms if g.geom_type == "LineString" and not g.is_empty]
        if not lines:
            return np.nan
        merged = linemerge(lines)
        coords = (
            [c for g in merged.geoms for c in g.coords]
            if merged.geom_type == "MultiLineString"
            else list(merged.coords)
        )
    elif geom.geom_type == "LineString":
        coords = list(geom.coords)
    else:
        return np.nan
    if len(coords) < 2:
        return np.nan
    dx = coords[-1][0] - coords[0][0]
    dy = coords[-1][1] - coords[0][1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return np.nan
    return np.degrees(np.arctan2(dy, dx)) % 180


def _angular_diff_osm(b1: float, b2: float) -> float:
    if np.isnan(b1) or np.isnan(b2):
        return np.nan
    diff = abs(b1 - b2) % 180
    return min(diff, 180 - diff)


def _resolve_osm_conflicts_at_nodes(
    pairs: gpd.GeoDataFrame,
    bav_nodes: gpd.GeoDataFrame,
    buf_lookup: dict,
) -> gpd.GeoDataFrame:
    """
    Split OSM ways that appear in multiple BAV buffers at BAV node positions,
    then reassign each piece to the buffer whose midpoint it falls in.
    Unresolvable conflicts (no nearby nodes) are kept as-is.
    """
    conflict_ids = set(pairs[pairs.duplicated("osm_id", keep=False)]["osm_id"].unique())
    if not conflict_ids:
        return pairs

    print(f"  Resolving {len(conflict_ids)} OSM ways spanning multiple buffers...")

    non_conflict  = pairs[~pairs["osm_id"].isin(conflict_ids)].copy()
    conflict_rows = pairs[pairs["osm_id"].isin(conflict_ids)].copy()

    osm_unique = (
        conflict_rows[["osm_id", "geometry"]]
        .drop_duplicates("osm_id")
        .set_index("osm_id")
    )

    resolved: List[dict] = []
    kept_as_is: List[pd.DataFrame] = []

    for osm_id, osm_row in osm_unique.iterrows():
        osm_geom = osm_row["geometry"]
        if osm_geom.geom_type != "LineString":
            kept_as_is.append(conflict_rows[conflict_rows["osm_id"] == osm_id])
            continue

        nearby = bav_nodes[bav_nodes.distance(osm_geom) <= OSM_NODE_SNAP_M]
        total_len = osm_geom.length
        split_dists = sorted({
            d for pt in nearby.geometry
            if 1.0 < (d := osm_geom.project(pt)) < total_len - 1.0
        })

        if not split_dists:
            kept_as_is.append(conflict_rows[conflict_rows["osm_id"] == osm_id])
            continue

        template = conflict_rows[conflict_rows["osm_id"] == osm_id].iloc[0]
        osm_attrs = {col: template[col] for col in _OSM_COLS if col in template.index}
        matched_segs = conflict_rows[conflict_rows["osm_id"] == osm_id]["Segment_ID"].tolist()

        for start, end in zip([0.0] + split_dists, split_dists + [total_len]):
            piece = shp_substring(osm_geom, start, end)
            if piece.is_empty or piece.length < 0.5:
                continue
            midpoint = piece.interpolate(0.5, normalized=True)
            assigned = next(
                (sid for sid in matched_segs if buf_lookup[sid].contains(midpoint)),
                None,
            )
            if assigned is None:
                continue
            resolved.append({**osm_attrs, "Segment_ID": assigned, "geometry": piece})

    parts = [non_conflict]
    if resolved:
        parts.append(gpd.GeoDataFrame(resolved, crs=pairs.crs))
    parts.extend(kept_as_is)

    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=pairs.crs)


def _empty_osm_result(bav: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = bav.copy()
    out["Average_Speed"]       = np.nan
    out["Predominant_Speed"]   = np.nan
    out["Speed_Coverage_Pct"]  = 0.0
    return out


def _join_osm_speeds(
    bav: gpd.GeoDataFrame,
    bav_nodes: gpd.GeoDataFrame,
    osm: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Overlay OSM railway ways onto BAV segments and aggregate speed data.

    Returns bav with three new columns:
      Average_Speed       — length-weighted mean speed (km/h) or NaN
      Predominant_Speed   — modal speed by length (km/h) or NaN
      Speed_Coverage_Pct  — fraction of segment length covered by the predominant speed
    """
    buf_series   = bav.geometry.buffer(OSM_BUFFER_M)
    bav_buf      = gpd.GeoDataFrame(bav[["Segment_ID"]].copy(), geometry=buf_series, crs=bav.crs)
    buf_lookup   = bav_buf.set_index("Segment_ID")["geometry"]
    bav_lengths  = bav.set_index("Segment_ID")["Length"]
    bav_bearings = bav.set_index("Segment_ID")["geometry"].apply(_compute_osm_bearing)
    seg_gauge    = bav.set_index("Segment_ID")["Gauge"].to_dict()

    osm_join = osm[_OSM_COLS + ["geometry"]].copy()
    pairs = gpd.sjoin(osm_join, bav_buf, how="inner", predicate="intersects")
    pairs = pairs.drop(columns=["index_right"], errors="ignore").reset_index(drop=True)
    print(f"  {len(pairs)} candidate (buffer, OSM way) pairs")

    if pairs.empty:
        return _empty_osm_result(bav)

    pairs = _resolve_osm_conflicts_at_nodes(pairs, bav_nodes, buf_lookup)

    pairs["clipped_geom"] = pairs.apply(
        lambda r: r.geometry.intersection(buf_lookup[r["Segment_ID"]]), axis=1
    )
    is_line = pairs["clipped_geom"].geom_type.isin(["LineString", "MultiLineString"])
    pairs = pairs[is_line].copy()
    pairs["clipped_length_m"] = pairs["clipped_geom"].length
    pairs = pairs[pairs["clipped_length_m"] > 0].copy()

    pairs["osm_bearing"] = pairs["clipped_geom"].apply(_compute_osm_bearing)
    pairs["bav_bearing"] = pairs["Segment_ID"].map(bav_bearings)
    pairs["bearing_diff"] = pairs.apply(
        lambda r: _angular_diff_osm(r["osm_bearing"], r["bav_bearing"]), axis=1
    )
    n_before = len(pairs)
    pairs = pairs[
        pairs["bearing_diff"].isna() | (pairs["bearing_diff"] <= OSM_MAX_BEARING_DIFF)
    ].copy()
    print(f"  {n_before - len(pairs)} pairs removed by bearing filter")

    seg_mode = (
        bav.set_index("Segment_ID")["Transport_Mode"].to_dict()
        if "Transport_Mode" in bav.columns else {}
    )
    pairs["bav_mode"]  = pairs["Segment_ID"].map(seg_mode)
    pairs["bav_gauge"] = pairs["Segment_ID"].map(seg_gauge)

    def _allowed_osm_types(mode, gauge) -> set:
        """Return the set of OSM railway types compatible with this segment."""
        if pd.notna(mode) and str(mode).strip():
            types: set = set()
            for m in str(mode).split('/'):
                types |= MODE_TO_OSM_TYPES.get(m.strip(), set())
            if types:
                return types
        return GAUGE_TO_OSM_TYPES.get(gauge, _ALL_OSM_TYPES)

    compatible_mask = pairs.apply(
        lambda r: r["railway_type"] in _allowed_osm_types(r["bav_mode"], r["bav_gauge"]),
        axis=1,
    )
    n_before = len(pairs)
    pairs = pairs[compatible_mask].drop(columns=["bav_mode", "bav_gauge"]).copy()
    print(f"  {n_before - len(pairs)} pairs removed by railway-type filter ({len(pairs)} remain)")

    pairs["speed_kmh"] = pairs.apply(_best_speed, axis=1)

    def _aggregate(g):
        seg_len = bav_lengths.get(g.name, np.nan)
        valid   = g[g["speed_kmh"].notna()]
        if valid.empty:
            return pd.Series({
                "Average_Speed":      np.nan,
                "Predominant_Speed":  np.nan,
                "Speed_Coverage_Pct": 0.0,
            })

        avg = (
            (valid["speed_kmh"] * valid["clipped_length_m"]).sum()
            / valid["clipped_length_m"].sum()
        )
        spd_lengths = valid.groupby("speed_kmh")["clipped_length_m"].sum()
        predominant = spd_lengths.idxmax()
        pct = (
            float(spd_lengths[predominant] / seg_len)
            if seg_len and seg_len > 0
            else 0.0
        )
        return pd.Series({
            "Average_Speed":      round(avg, 1),
            "Predominant_Speed":  predominant,
            "Speed_Coverage_Pct": round(min(pct, 1.0), 3),
        })

    if pairs.empty:
        return _empty_osm_result(bav)

    agg = pairs.groupby("Segment_ID").apply(_aggregate, include_groups=False)

    _SPEED_COLS = ("Average_Speed", "Predominant_Speed", "Speed_Coverage_Pct")
    bav_base = bav.drop(columns=[c for c in _SPEED_COLS if c in bav.columns])
    result = bav_base.merge(agg.reset_index(), on="Segment_ID", how="left")
    result["Speed_Coverage_Pct"] = (
        result["Speed_Coverage_Pct"].fillna(0.0)
    )
    return result


def enrich_segments_with_osm_speed(
    segments: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    raw_dir: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Load OSM maxspeed data fetched by infrabuild_filter_network.py and attach
    speed columns to segments.

    Reads osm_maxspeed_segments.gpkg from data/Infrastructure/Raw/ (written by
    infrabuild_filter_network.run_filter_network).  If the file is absent the
    function returns segments unchanged with NaN speed columns and prints a
    warning — re-run infrabuild_filter_network.py to populate it.

    Adds three columns to the returned GeoDataFrame:
      Average_Speed       — length-weighted mean OSM speed (km/h) or NaN
      Predominant_Speed   — modal speed by segment length (km/h) or NaN
      Speed_Coverage_Pct  — fraction of segment under predominant speed

    Args:
        segments: BAV segment GeoDataFrame (must have ID, Length, Gauge, geometry)
        nodes:    BAV node GeoDataFrame (used for conflict resolution at junctions)
        raw_dir:  Override for Raw/ directory path (defaults to paths.NETWORK_INFRASTRUCTURE_RAW)

    Returns:
        segments GeoDataFrame with speed columns added.
    """
    osm_path = (
        Path(raw_dir) if raw_dir
        else Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_RAW
    ) / "osm_maxspeed_segments.gpkg"

    if not osm_path.exists():
        print(
            f"  WARNING: {osm_path.name} not found in Raw/.\n"
            "  Run infrabuild_filter_network.py to fetch OSM data.\n"
            "  Speed columns will be NaN for this build."
        )
        return _empty_osm_result(segments)

    osm = gpd.read_file(osm_path)
    if osm.crs is None or osm.crs.to_epsg() != 2056:
        osm = osm.to_crs("EPSG:2056")
    print(f"  Loaded {len(osm)} OSM ways from {osm_path.name}")

    if osm.empty:
        print("  WARNING: OSM file is empty — speed columns will be NaN.")
        return _empty_osm_result(segments)

    joined = _join_osm_speeds(segments, nodes, osm)

    # When 100% of the segment runs at the predominant speed, the weighted
    # average is identical — use the exact value to avoid float rounding drift.
    full_cov = joined["Speed_Coverage_Pct"] == 1.0
    joined.loc[full_cov, "Average_Speed"] = joined.loc[full_cov, "Predominant_Speed"]

    total     = len(joined)
    has_speed = joined["Average_Speed"].notna().sum()
    no_speed  = (joined["Average_Speed"].isna() & (joined["Speed_Coverage_Pct"] > 0)).sum()
    no_osm    = (joined["Speed_Coverage_Pct"] == 0).sum()
    print(f"\n  OSM speed coverage:")
    print(f"    Speed assigned     : {has_speed:>4}  ({100*has_speed/total:.1f}%)")
    print(f"    OSM found, no speed: {no_speed:>4}  ({100*no_speed/total:.1f}%)")
    print(f"    No OSM match       : {no_osm:>4}  ({100*no_osm/total:.1f}%)")

    return joined


def derive_segment_transport_mode(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Propagate transport_mode from station nodes to segments and junctions via BFS.

    Single-mode stations seed the traversal.  Composite-mode stations
    (e.g. "train / tram") are left unseeded so propagation fills them from
    their neighbours.  Each segment and junction node receives the union of
    all modes reaching it.  Segments or junctions unreachable from any seed
    retain NaN; OSM matching falls back to gauge for those.

    Args:
        nodes:    macro-level nodes GeoDataFrame (must have NAME, transport_mode,
                  node_class columns)
        segments: macro-level segments GeoDataFrame (must have From_Name, To_Name)

    Returns:
        (nodes, segments) — nodes has derived Transport_Mode filled on previously-NaN
        junctions; segments has a new Transport_Mode column.
    """
    # Only modes that correspond to tracked rail infrastructure are propagated.
    # Bus, ship, etc. are excluded — they don't run on the BAV rail graph.
    _RAIL_MODES = frozenset(MODE_TO_OSM_TYPES.keys())

    def _parse_all(val) -> set:
        if pd.isna(val) or not str(val).strip():
            return set()
        return {m.strip() for m in str(val).split('/') if m.strip()}

    def _parse_rail(val) -> set:
        return _parse_all(val) & _RAIL_MODES

    def _fmt(modes: set):
        return ' / '.join(sorted(modes)) if modes else pd.NA

    nodes    = nodes.copy()
    segments = segments.copy()

    # Adjacency: node_name → [(seg_index, other_node_name), ...]
    adj: Dict[str, list] = {}
    for idx, row in segments.iterrows():
        fn, tn = row.get('From_Name'), row.get('To_Name')
        if fn and tn:
            adj.setdefault(fn, []).append((idx, tn))
            adj.setdefault(tn, []).append((idx, fn))

    # Collect OeV-sourced rail modes (non-rail modes stripped at source)
    node_modes: Dict[str, set] = {}
    for _, row in nodes.iterrows():
        name = row.get('Name')
        if not name:
            continue
        m = _parse_rail(row.get('Transport_Mode'))
        if m:
            node_modes[name] = m

    # Nodes with OeV-sourced rail data act as hard propagation boundaries —
    # the BFS never overwrites them, preventing modes bleeding across network types.
    original_rail_seeds: frozenset = frozenset(node_modes)

    seg_modes: Dict[int, set] = {}

    # Seed: only single-mode stations (composite stations are unseeded)
    queue: deque = deque(n for n, m in node_modes.items() if len(m) == 1)

    while queue:
        name = queue.popleft()
        n_modes = node_modes.get(name, set())
        if not n_modes:
            continue
        for seg_idx, other in adj.get(name, []):
            old_s = seg_modes.get(seg_idx, set())
            new_s = old_s | n_modes
            if new_s != old_s:
                seg_modes[seg_idx] = new_s
                # Hard stop: never cross into a node that already has OeV rail data
                if other not in original_rail_seeds:
                    old_o = node_modes.get(other, set())
                    new_o = old_o | new_s
                    if new_o != old_o:
                        node_modes[other] = new_o
                        queue.append(other)

    # Write segment modes
    segments['Transport_Mode'] = [_fmt(seg_modes.get(i, set())) for i in segments.index]

    # Update junctions: only fill nodes that had no OeV rail data
    name_to_idx: Dict[str, int] = {
        row.get('Name'): idx
        for idx, row in nodes.iterrows()
        if row.get('Name')
    }
    for name, modes in node_modes.items():
        if name in original_rail_seeds:
            continue  # station already has OeV data — do not overwrite
        idx = name_to_idx.get(name)
        if idx is None:
            continue
        existing = _parse_all(
            nodes.at[idx, 'Transport_Mode'] if 'Transport_Mode' in nodes.columns else pd.NA
        )
        if not existing and modes:
            nodes.at[idx, 'Transport_Mode'] = _fmt(modes)

    n_segs = segments['Transport_Mode'].notna().sum()
    n_junc = (
        nodes[nodes.get('Node_Class', pd.Series(dtype=str)) == 'junction']['Transport_Mode'].notna().sum()
        if 'Node_Class' in nodes.columns else 0
    )
    print(f"  Mode assigned: {n_segs}/{len(segments)} segments, {n_junc} junctions derived")
    return nodes, segments


# =============================================================================
# ASP Station Import + Parent-Child Assignment
# =============================================================================

ASP_IMPORT_RADIUS_M    = 200  # max distance from prefix-matching parent station
PARENT_CHILD_RADIUS_M  = 200  # max distance for prefix-based parent detection


def _import_asp_stations(
    macro_nodes: gpd.GeoDataFrame,
    macro_segs:  gpd.GeoDataFrame,
    raw_nodes:   gpd.GeoDataFrame,
    raw_segs:    gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Import assigned_service_point nodes from raw_nodes into the macroscopic network.

    A node qualifies if:
      - its node_class is 'assigned_service_point'
      - it lies within ASP_IMPORT_RADIUS_M of a station node in macro_nodes
      - the station's NAME is a word-for-word prefix of the ASP's NAME

    Qualifying nodes are reclassified as 'station'. Their raw connecting segments
    are added where the other endpoint is already in the (updated) macro_nodes.
    """
    asp_raw = raw_nodes[raw_nodes['Node_Class'] == 'assigned_service_point'].copy()
    if asp_raw.empty:
        return macro_nodes, macro_segs

    macro_station_mask = macro_nodes['Node_Class'].isin({'station', 'abandoned_station'})
    macro_stations = macro_nodes[macro_station_mask].copy()
    macro_names = set(macro_nodes['Name'].dropna())

    # ── Pass 1: identify which ASP nodes to import ────────────────────────────
    to_import = []
    for _, asp in asp_raw.iterrows():
        asp_name = str(asp['Name'])
        if asp_name in macro_names:
            continue  # already present
        asp_words = asp_name.lower().split()
        aE = float(asp['E'])
        aN = float(asp['N'])

        best_parent_name = None
        best_len = 0
        for _, stn in macro_stations.iterrows():
            stn_name = str(stn['Name'])
            stn_words = stn_name.lower().split()
            if len(stn_words) >= len(asp_words):
                continue  # parent must have fewer words
            if asp_words[:len(stn_words)] != stn_words:
                continue
            dist = np.hypot(float(stn['E']) - aE, float(stn['N']) - aN)
            if dist <= ASP_IMPORT_RADIUS_M and len(stn_words) > best_len:
                best_parent_name = stn_name
                best_len = len(stn_words)

        if best_parent_name is not None:
            new_node = asp.copy()
            new_node['Node_Class'] = 'station'
            to_import.append((new_node, best_parent_name))

    if not to_import:
        return macro_nodes, macro_segs

    # ── Pass 2: add all qualifying nodes then add their segments ─────────────
    new_node_rows = [n for n, _ in to_import]
    new_nodes_gdf = gpd.GeoDataFrame(new_node_rows, crs=macro_nodes.crs)
    macro_nodes = pd.concat([macro_nodes, new_nodes_gdf], ignore_index=True)
    macro_names = set(macro_nodes['Name'].dropna())

    new_segs = []
    seen_seg_ids = set(macro_segs['Segment_ID'].dropna())
    for new_node, parent_name in to_import:
        asp_name = str(new_node['Name'])
        mask = (raw_segs['From_Name'] == asp_name) | (raw_segs['To_Name'] == asp_name)
        for _, seg in raw_segs[mask].iterrows():
            other = seg['To_Name'] if seg['From_Name'] == asp_name else seg['From_Name']
            if other in macro_names and seg['Segment_ID'] not in seen_seg_ids:
                new_segs.append(seg)
                seen_seg_ids.add(seg['Segment_ID'])
        print(
            f"  [ASP] imported '{asp_name}' → station"
            f"  (prefix-parent: '{parent_name}',"
            f" dist: {np.hypot(float(new_node['E']) - float(macro_nodes[macro_nodes['Name'] == parent_name]['E'].iloc[0]), float(new_node['N']) - float(macro_nodes[macro_nodes['Name'] == parent_name]['N'].iloc[0])):.0f}m)"
        )

    if new_segs:
        new_segs_gdf = gpd.GeoDataFrame(new_segs, crs=macro_segs.crs)
        macro_segs = pd.concat([macro_segs, new_segs_gdf], ignore_index=True)

    # Remove merged bypass segments whose constituent IDs were just rescued.
    # filter_macroscopic_nodes() merges degree-2 non-KEEP nodes by joining their
    # segment IDs with '+'. When an ASP node is rescued, those original segments
    # are restored, making the bypass a duplicate. Drop it.
    rescued_ids = {str(s['Segment_ID']) for s in new_segs}
    if rescued_ids:
        def _is_bypass(sid):
            sid_str = str(sid)
            if '+' not in sid_str:
                return False
            return bool(rescued_ids.intersection(sid_str.split('+')))

        bypass_mask = macro_segs['Segment_ID'].apply(_is_bypass)
        n_bypass = int(bypass_mask.sum())
        if n_bypass:
            macro_segs = macro_segs[~bypass_mask].reset_index(drop=True)
            print(f"  [ASP] removed {n_bypass} merged bypass segment(s) superseded by rescued originals")

    print(f"  [ASP] {len(to_import)} node(s) imported, {len(new_segs)} segment(s) added")
    return macro_nodes.reset_index(drop=True), macro_segs.reset_index(drop=True)


def _assign_parent_child(macro_nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Detect parent-child station relationships via word-prefix name matching.

    For each station with a NULL Parent_Node: if another station's Name is a
    strict word-for-word prefix of this station's Name and lies within
    PARENT_CHILD_RADIUS_M, the longer-named station becomes the child and the
    shorter-named station's Number is written into Parent_Node.

    Only fills NULL Parent_Node values; never overwrites existing relationships.
    """
    station_mask = macro_nodes['Node_Class'].isin({'station', 'abandoned_station'})
    stations = macro_nodes[station_mask].reset_index()  # keep original index

    updated = 0
    for idx, row in macro_nodes.iterrows():
        if not station_mask.get(idx, False):
            continue
        existing_parent = row.get('Parent_Node')
        if pd.notna(existing_parent) and str(existing_parent).strip().lower() not in ('', 'none', 'nan'):
            continue  # already has a parent

        child_name = str(row.get('Name', ''))
        if not child_name:
            continue
        child_words = child_name.lower().split()
        cE = float(row['E'])
        cN = float(row['N'])

        best_bpnr = None
        best_len = 0
        for _, stn in stations.iterrows():
            stn_name = str(stn['Name'])
            if stn_name == child_name:
                continue
            stn_words = stn_name.lower().split()
            if len(stn_words) >= len(child_words):
                continue
            if child_words[:len(stn_words)] != stn_words:
                continue
            dist = np.hypot(float(stn['E']) - cE, float(stn['N']) - cN)
            if dist > PARENT_CHILD_RADIUS_M:
                continue
            bpnr = stn.get('Number')
            if pd.notna(bpnr) and len(stn_words) > best_len:
                best_bpnr = int(float(bpnr))
                best_len = len(stn_words)

        if best_bpnr is not None:
            macro_nodes.at[idx, 'Parent_Node'] = best_bpnr
            updated += 1

    print(f"  [parent-child] {updated} relationship(s) assigned")
    return macro_nodes


# Mode-specific speed defaults (km/h) — mirrors _MODE_DEFAULT_SPEEDS in
# services_service_projection.py. Both dicts must be kept in sync.
# Conservative values so that null-speed segments are not favoured in routing.
_MODE_DEFAULT_SPEEDS: Dict[str, float] = {
    'train':       50.0,
    'tram':        30.0,
    'funicular':   10.0,
    'cog_railway': 15.0,
    'bus':         30.0,
}
_MODE_DEFAULT_FALLBACK: float = 50.0  # matches RAIL_DEFAULT_SPEED_KMH


def _compute_approx_travel_times(
    segments: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Add TT_Passing and TT_Stopping columns (approximate travel times in minutes).

    Uses Average_Speed (OSM-derived) where available; falls back to
    _MODE_DEFAULT_SPEEDS when null.

    Formula (junction-aware, parameterised by n_sta = number of station
    endpoints of the segment, ∈ {0, 1, 2})
    -----------------------------------------------------------------------
    v_ms = speed_kmh / 3.6

    TT_Passing  = max( (Length / v_ms)                          * 1.30 / 60, 0.1 )
    TT_Stopping = max( (Length / v_ms + n_sta · 0.5 · v_ms / a) * 1.30 / 60, 0.1 )

    n_sta is derived per-segment from endpoint Node_Class:
      - resolved as 'station'                  → contributes 1
      - resolved as anything else              → contributes 0
      - unresolved name (not present in nodes) → contributes 0
        (macro-consolidation correctly drops junctions/yards; their names
         linger on segments but their absence from nodes implies non-station)

    Junction–junction segments collapse to TT_Stopping = TT_Passing = pure
    cruise · buffer / 60, satisfying the invariant that at jct-jct the two
    columns must be identical.

    Deceleration a = 0.7 m/s²: typical operational value within the European
    service-brake range (0.5–1.3 m/s², UIC 544-1 / ERTMS). Initially set to
    the conservative lower bound (0.5 m/s²); raised to 0.7 after calibration
    against GTFS-derived TT_Stopping in
    infrabuild_infrastructure_enhancement.py.

    Buffer 1.30 (shared): operational overhead for signal delays, junction
    speed restrictions, and schedule padding. Single shared value for
    stopping and passing — the n_sta term carries the regime difference.
    Calibrated against GTFS in the n_sta=0 (jct-jct) bucket of
    infrabuild_infrastructure_enhancement.py, where the formula reduces to
    pure cruise · buffer and the GTFS/predicted ratio is most diagnostic
    of buffer / cruise-speed mismatch. Bumped from 1.20 → 1.30 after the
    junction-aware refactor: ratio centred from ~1.08 to ~1.00 on jct-jct.
    """
    a = settings.SERVICE_BRAKE_DECEL_MS2
    buffer = settings.TT_OPERATIONAL_BUFFER

    # name → Node_Class lookup. Anything not in this map (or class falsy)
    # is treated as a junction per the calibration policy.
    name_to_class: Dict[str, str] = {}
    if nodes is not None and 'Name' in nodes.columns:
        for _, nrow in nodes.iterrows():
            name = nrow.get('Name')
            if name is None or (isinstance(name, float) and pd.isna(name)):
                continue
            cls = nrow.get('Node_Class') if 'Node_Class' in nodes.columns else None
            name_to_class[str(name)] = str(cls).strip() if cls else ''

    def _is_station(name) -> int:
        if name is None or (isinstance(name, float) and pd.isna(name)):
            return 0
        return 1 if name_to_class.get(str(name), '') == 'station' else 0

    def _speed(row) -> float:
        spd = row.get('Average_Speed')
        if pd.notna(spd) and float(spd) > 0:
            return float(spd)
        mode = str(row.get('Transport_Mode', '')).strip()
        for m in mode.split('/'):
            s = _MODE_DEFAULT_SPEEDS.get(m.strip())
            if s:
                return s
        return _MODE_DEFAULT_FALLBACK

    rows = segments.copy()
    rows['_v_ms']   = rows.apply(_speed, axis=1) / 3.6
    rows['_cruise'] = rows['Length'] / rows['_v_ms']
    rows['_n_sta']  = rows.apply(
        lambda r: _is_station(r.get('From_Name')) + _is_station(r.get('To_Name')),
        axis=1,
    )
    rows['TT_Passing']  = (rows['_cruise'] * buffer / 60).clip(lower=0.1)
    rows['TT_Stopping'] = (
        (rows['_cruise'] + rows['_n_sta'] * 0.5 * rows['_v_ms'] / a) * buffer / 60
    ).clip(lower=0.1)
    return rows.drop(columns=['_v_ms', '_cruise', '_n_sta'])


def run_build_base(
    raw_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    seed_year: Optional[str] = None,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load Raw/ network, apply macroscopic simplification, export to Base/.

    Reads from:  data/Infrastructure/Raw/   (written by infrabuild_filter_network.py)
    Writes to:   data/Infrastructure/Base/  (nodes.gpkg, segments.gpkg,
                                             segments_composition.gpkg, Base.qgz)

    Args:
        seed_year: 4-digit year string to apply seeds for (e.g. '2026').
                   If None, year extraction from the output folder name is attempted;
                   if that also yields nothing, seeds are skipped.

    Returns (macro_nodes, macro_segs, composition).
    """
    print("=" * 60)
    print("Build Macroscopic Base Network")
    print("=" * 60)

    raw_path = Path(raw_dir  or (Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_RAW))
    out_path = Path(output_dir or (Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_BASE))

    nodes_path = raw_path / "nodes.gpkg"
    segs_path  = raw_path / "segments.gpkg"

    if not nodes_path.exists() or not segs_path.exists():
        raise FileNotFoundError(
            f"Raw network not found in {raw_path}.\n"
            "Run infrabuild_filter_network.py first."
        )

    print("\n--- Loading raw network ---")
    nodes    = gpd.read_file(nodes_path)
    segments = gpd.read_file(segs_path)
    print(f"  {len(nodes)} nodes, {len(segments)} segments")

    comp_raw_path = raw_path / "segments_composition.gpkg"
    composition_raw = (
        gpd.read_file(comp_raw_path) if comp_raw_path.exists() else gpd.GeoDataFrame()
    )
    if not composition_raw.empty:
        print(f"  {len(composition_raw)} composition pieces (raw)")

    print("\n--- Macroscopic simplification ---")
    macro_nodes, macro_segs, seg_id_remap = filter_macroscopic_nodes(nodes, segments)

    composition = (
        _filter_composition_for_macro(composition_raw, seg_id_remap)
        if not composition_raw.empty else gpd.GeoDataFrame()
    )

    print("\n--- ASP station import ---")
    macro_nodes, macro_segs = _import_asp_stations(macro_nodes, macro_segs, nodes, segments)

    print("\n--- Parent-child assignment ---")
    macro_nodes = _assign_parent_child(macro_nodes)

    print("\n--- Transport mode propagation ---")
    macro_nodes, macro_segs = derive_segment_transport_mode(macro_nodes, macro_segs)

    print("\n--- OSM speed enrichment ---")
    macro_segs = enrich_segments_with_osm_speed(macro_segs, macro_nodes, raw_dir=str(raw_path))

    print("\n--- Approximate travel times ---")
    macro_segs = _compute_approx_travel_times(macro_segs, macro_nodes)

    print("\n--- Seed augmentation ---")
    _seed_key = seed_year if seed_year else out_path.name
    macro_nodes, macro_segs, composition = load_and_apply_seed(
        _seed_key, macro_nodes, macro_segs, composition,
    )

    out_path.mkdir(parents=True, exist_ok=True)
    print("\n--- Exporting ---")

    # ---- Round numeric columns -----------------------------------------------
    _SEG_ROUND_COLS_3 = [
        "Length", "Km_Start", "Km_End",
        "Tunnel_Length", "Bridge_Length", "Conventional_Length",
        "From_N", "From_E", "To_N", "To_E",
    ]
    for col in _SEG_ROUND_COLS_3:
        if col in macro_segs.columns:
            macro_segs[col] = pd.to_numeric(macro_segs[col], errors='coerce').round(3)
    for col in ("TT_Passing", "TT_Stopping"):
        if col in macro_segs.columns:
            macro_segs[col] = pd.to_numeric(macro_segs[col], errors='coerce').round(1)

    _NODE_ROUND_COLS = ["E", "N"]
    for col in _NODE_ROUND_COLS:
        if col in macro_nodes.columns:
            macro_nodes[col] = pd.to_numeric(macro_nodes[col], errors='coerce').round(3)

    # ---- Enforce canonical column order ------------------------------------
    _has_osm_speed = (
        (macro_segs['Average_Speed'].notna() & (macro_segs['Average_Speed'] > 0)) |
        (macro_segs['Predominant_Speed'].notna() & (macro_segs['Predominant_Speed'] > 0))
    )
    macro_segs['speed_source'] = _has_osm_speed.map({True: 'formula', False: 'estimate'})

    _SEG_COL_ORDER = [
        "Segment_ID", "From_Name", "To_Name", "Number", "Code",
        "From_N", "From_E", "To_N", "To_E",
        "Length", "Num_Tracks", "Gauge", "Electrification_Class",
        "Km_Start", "Km_End",
        "Route_Number", "Route_Name", "Route_Owner",
        "Tunnel_Length", "Bridge_Length", "Conventional_Length",
        "Transport_Mode", "Average_Speed", "Predominant_Speed", "Speed_Coverage_Pct",
        "TT_Passing", "TT_Stopping", "speed_source",
    ]
    macro_segs = macro_segs[
        [c for c in _SEG_COL_ORDER if c in macro_segs.columns]
        + [c for c in macro_segs.columns if c not in _SEG_COL_ORDER and c != macro_segs.geometry.name]
        + [macro_segs.geometry.name]
    ]

    _NODE_COL_ORDER = [
        "Node_ID", "Number", "Name", "Code",
        "E", "N",
        "Node_Class", "Transport_Mode",
        "Track_Count", "Platform_Count",
        "Parent_Node",
    ]
    macro_nodes = macro_nodes[
        [c for c in _NODE_COL_ORDER if c in macro_nodes.columns]
        + [c for c in macro_nodes.columns if c not in _NODE_COL_ORDER and c != macro_nodes.geometry.name]
        + [macro_nodes.geometry.name]
    ]

    macro_nodes.to_file(out_path / "nodes.gpkg", driver="GPKG")
    macro_segs.to_file(out_path  / "segments.gpkg", driver="GPKG")
    print(f"  nodes.gpkg    → {out_path / 'nodes.gpkg'}")
    print(f"  segments.gpkg → {out_path / 'segments.gpkg'}")
    if not composition.empty:
        composition.to_file(out_path / "segments_composition.gpkg", driver="GPKG")
        print(f"  segments_composition.gpkg → {out_path / 'segments_composition.gpkg'}"
              f"  ({len(composition)} pieces)")

    print("\n--- QGIS project ---")
    _qgz_name = out_path.name + ".qgz"
    _build_infra_qgz(str(out_path / _qgz_name), out_path)
    print(f"  {_qgz_name} → {out_path / _qgz_name}")

    print("\n" + "=" * 60)
    print(f"Done  |  {len(macro_nodes)} nodes  |  {len(macro_segs)} segments")
    print("=" * 60)

    return macro_nodes, macro_segs, composition


# =============================================================================
# Composition Handling
# =============================================================================

def _filter_composition_for_macro(
    composition: gpd.GeoDataFrame,
    seg_id_remap: dict,
) -> gpd.GeoDataFrame:
    """Filter and remap segments_composition after macroscopic simplification.

    Drops pieces for segments that were removed (dead-ends); remaps segment_id
    to the merged ID for segments that were joined through an intermediate node.
    """
    if composition.empty or 'Segment_ID' not in composition.columns:
        return composition
    mask = composition['Segment_ID'].astype(str).isin(seg_id_remap)
    result = composition[mask].copy()
    result['Segment_ID'] = result['Segment_ID'].astype(str).map(seg_id_remap)
    return result.reset_index(drop=True)


def _filter_composition_for_version(
    composition_base: gpd.GeoDataFrame,
    version_segment_ids: set,
) -> gpd.GeoDataFrame:
    """Filter Base composition to segments present in a named version.

    Segments removed by infrabuild_version_manager are absent from
    version_segment_ids and are simply dropped.  Manually added segments
    (new_From_To IDs) have no composition entry, so they are absent already.
    """
    if composition_base.empty or 'ID' not in composition_base.columns:
        return composition_base
    mask = composition_base['Segment_ID'].isin(version_segment_ids)
    return composition_base[mask].reset_index(drop=True)


# =============================================================================
# QGIS Project Generation
# =============================================================================

_QGIS_VERSION = "3.44.9-Solothurn"

_INFRA_SRS_BLOCK = """<spatialrefsys nativeFormat="wkt">
      <proj4>+proj=somerc +lat_0=46.9524055555556 +lon_0=7.43958333333333 +k_0=1 +x_0=2600000 +y_0=1200000 +ellps=bessel +towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs</proj4>
      <srsid>47</srsid>
      <srid>2056</srid>
      <authid>EPSG:2056</authid>
      <description>CH1903+ / LV95</description>
      <projectionacronym>somerc</projectionacronym>
      <ellipsoidacronym>bessel</ellipsoidacronym>
    </spatialrefsys>"""

_INFRA_WMS_ID   = 'Swisstopo_NationalMap_grey_infrabuild'
_INFRA_WMS_NAME = 'Swisstopo National Map (grey)'
_INFRA_WMS_SRC  = (
    'contextualWMSLegend=0&amp;crs=EPSG:2056&amp;dpiMode=7'
    '&amp;featureCount=10&amp;format=image/png'
    '&amp;layers=ch.swisstopo.pixelkarte-grau'
    '&amp;styles=&amp;url=http://wms.geo.admin.ch/'
)


def _multi_track_symbol_xml(sym_name: str, num_tracks: int) -> str:
    """XML for a line symbol with num_tracks parallel SimpleLine sub-layers."""
    width = '0.40'
    if num_tracks == 1:
        offsets = [0.0]
    elif num_tracks == 2:
        offsets = [0.400, -0.400]
    elif num_tracks == 3:
        offsets = [0.800, 0.0, -0.800]
    else:
        offsets = [1.200, 0.400, -0.400, -1.200]

    layers_xml = ''
    for off in offsets:
        layers_xml += f"""
          <layer pass="0" class="SimpleLine" locked="0" enabled="1">
            <prop k="line_color" v="0,0,0,255"/>
            <prop k="line_width" v="{width}"/>
            <prop k="line_width_unit" v="MM"/>
            <prop k="line_style" v="solid"/>
            <prop k="offset" v="{off:.3f}"/>
            <prop k="offset_unit" v="MM"/>
            <prop k="capstyle" v="square"/>
            <prop k="joinstyle" v="bevel"/>
          </layer>"""
    return (
        f'    <symbol alpha="1" clip_to_extent="1" type="line"'
        f' name="{sym_name}" force_rhr="0">\n'
        f'{layers_xml}\n'
        f'    </symbol>'
    )


def _segments_maplayer_xml(layer_id: str, gpkg_relpath: str,
                            display_name: str) -> str:
    """<maplayer> XML for segments with a rule-based multi-track renderer."""
    _RULES = [
        (1,  '&quot;Num_Tracks&quot; = 1',    '1 track',  '0'),
        (2,  '&quot;Num_Tracks&quot; = 2',    '2 tracks', '1'),
        (3,  '&quot;Num_Tracks&quot; = 3',    '3 tracks', '2'),
        (99, '&quot;Num_Tracks&quot; &gt;= 4','4+ tracks','3'),
    ]
    root_key = uuid.uuid4().hex
    rules_parts, symbols_parts = [], []
    for n_tracks, flt, label, sym_name in _RULES:
        rkey = uuid.uuid4().hex
        rules_parts.append(
            f'      <rule key="{rkey}" filter="{flt}" label="{label}" symbol="{sym_name}"/>'
        )
        symbols_parts.append(_multi_track_symbol_xml(sym_name, n_tracks))

    renderer_xml = (
        f'    <renderer-v2 type="RuleRenderer" symbollevels="0"'
        f' forceraster="0" enableorderby="0">\n'
        f'      <rules key="{root_key}">\n'
        + '\n'.join(rules_parts) + '\n'
        f'      </rules>\n'
        f'      <symbols>\n'
        + '\n'.join(symbols_parts) + '\n'
        f'      </symbols>\n'
        f'      <rotation/>\n'
        f'      <sizescale/>\n'
        f'    </renderer-v2>'
    )
    return (
        f'  <maplayer geometry="Line" type="vector" hasScaleBasedVisibilityFlag="0">\n'
        f'    <id>{layer_id}</id>\n'
        f'    <datasource>{gpkg_relpath}|layername=segments</datasource>\n'
        f'    <layername>{display_name}</layername>\n'
        f'    <provider encoding="UTF-8">ogr</provider>\n'
        f'    <srs>{_INFRA_SRS_BLOCK}</srs>\n'
        f'{renderer_xml}\n'
        f'  </maplayer>'
    )


def _nodes_maplayer_xml(layer_id: str, gpkg_relpath: str, display_name: str) -> str:
    """<maplayer> XML for all nodes using a RuleRenderer."""
    root_key = uuid.uuid4().hex
    rules_xml = f"""\
      <rules key="{root_key}">\
        <rule key="{uuid.uuid4().hex}" filter="&quot;Node_Class&quot; = 'station' AND &quot;Transport_Mode&quot; LIKE '%train%'" label="Train Stations" symbol="0"/>\
        <rule key="{uuid.uuid4().hex}" filter="&quot;Node_Class&quot; = 'station' AND (&quot;Transport_Mode&quot; LIKE '%tram%' OR &quot;Transport_Mode&quot; LIKE '%funicular%' OR &quot;Transport_Mode&quot; LIKE '%cog_railway%')" label="Tram / Funicular" symbol="1"/>\
        <rule key="{uuid.uuid4().hex}" filter="&quot;Node_Class&quot; = 'junction'" label="Junctions" symbol="2"/>\
      </rules>"""

    marker_xml = f"""\
    <renderer-v2 forceraster="0" symbollevels="0" type="RuleRenderer" enableorderby="0">
{rules_xml}
      <symbols>
        <symbol alpha="1" clip_to_extent="1" type="marker" name="0" force_rhr="0">
          <layer pass="0" class="SimpleMarker" locked="0" enabled="1">
            <prop k="angle" v="0"/>
            <prop k="color" v="255,255,255,255"/>
            <prop k="joinstyle" v="bevel"/>
            <prop k="name" v="circle"/>
            <prop k="offset" v="0,0"/>
            <prop k="offset_unit" v="MM"/>
            <prop k="outline_color" v="0,0,0,255"/>
            <prop k="outline_style" v="solid"/>
            <prop k="outline_width" v="0.25"/>
            <prop k="outline_width_unit" v="MM"/>
            <prop k="size" v="2.5"/>
            <prop k="size_unit" v="MM"/>
          </layer>
        </symbol>
        <symbol alpha="1" clip_to_extent="1" type="marker" name="1" force_rhr="0">
          <layer pass="0" class="SimpleMarker" locked="0" enabled="1">
            <prop k="angle" v="0"/>
            <prop k="color" v="0,0,0,255"/>
            <prop k="joinstyle" v="bevel"/>
            <prop k="name" v="circle"/>
            <prop k="offset" v="0,0"/>
            <prop k="offset_unit" v="MM"/>
            <prop k="outline_color" v="0,0,0,255"/>
            <prop k="outline_style" v="solid"/>
            <prop k="outline_width" v="0.25"/>
            <prop k="outline_width_unit" v="MM"/>
            <prop k="size" v="1.75"/>
            <prop k="size_unit" v="MM"/>
          </layer>
        </symbol>
        <symbol alpha="1" clip_to_extent="1" type="marker" name="2" force_rhr="0">
          <layer pass="0" class="SimpleMarker" locked="0" enabled="1">
            <prop k="angle" v="0"/>
            <prop k="color" v="150,150,150,255"/>
            <prop k="joinstyle" v="bevel"/>
            <prop k="name" v="circle"/>
            <prop k="offset" v="0,0"/>
            <prop k="offset_unit" v="MM"/>
            <prop k="outline_color" v="120,120,120,255"/>
            <prop k="outline_style" v="solid"/>
            <prop k="outline_width" v="0.20"/>
            <prop k="outline_width_unit" v="MM"/>
            <prop k="size" v="1.25"/>
            <prop k="size_unit" v="MM"/>
          </layer>
        </symbol>
      </symbols>
      <rotation/>
      <sizescale/>
    </renderer-v2>"""

    labeling_xml = """\
    <labeling type="rule-based">
      <rules>
        <rule filter="&quot;Node_Class&quot; = 'station' AND &quot;Transport_Mode&quot; LIKE '%train%'">
          <settings calloutType="simple">
            <text-style fieldName="Code" fontFamily="Arial" fontSize="8" fontWeight="0" fontItalic="0" fontUnderline="0" fontStrikeout="0" textColor="35,35,35,255" textOpacity="1" blendMode="0" namedStyle="Regular" isExpression="0" useSubstitutions="0" multilineHeight="1" fontCapitals="0" fontLetterSpacing="0" fontWordSpacing="0" fontSizeUnit="Point" textOrientation="horizontal" previewBkgrdColor="255,255,255,255"/>
            <text-buffer bufferDraw="1" bufferSize="1" bufferColor="255,255,255,255" bufferOpacity="1" bufferJoinStyle="128" bufferNoFill="1" bufferSizeUnits="MM" bufferSizeMapUnitScale="3x:0,0,0,0,0,0" bufferBlendMode="0"/>
            <background shapeDraw="0"/>
            <shadow shadowDraw="0"/>
            <placement placement="1" centroidWhole="0" placementFlags="10" priority="5" offsetType="0" quadOffset="4" xOffset="2" yOffset="2" offsetUnits="MM" dist="1" distInMapUnits="0" distMapUnitScale="3x:0,0,0,0,0,0" rotationAngle="0" geometryGenerator="" geometryGeneratorEnabled="0" geometryGeneratorType="PointGeometry" isExpression="0" labelPerPart="0"/>
            <rendering drawLabels="1" obstacle="1" obstacleFactor="1" obstacleType="1" limitNumLabels="0" maxNumLabels="2000" minFeatureSize="0" fontMinPixelSize="3" fontMaxPixelSize="10000" displayAll="0" upsidedownLabels="0" mergeLines="0" zIndex="0" scaleVisibility="0" scaleMin="1" scaleMax="10000000" labelPerPart="0"/>
            <dd_properties/>
          </settings>
        </rule>
      </rules>
    </labeling>"""

    return (
        f'  <maplayer geometry="Point" type="vector" hasScaleBasedVisibilityFlag="0">\n'
        f'    <id>{layer_id}</id>\n'
        f'    <datasource>{gpkg_relpath}|layername=nodes</datasource>\n'
        f'    <layername>{display_name}</layername>\n'
        f'    <provider encoding="UTF-8">ogr</provider>\n'
        f'    <srs>{_INFRA_SRS_BLOCK}</srs>\n'
        f'{marker_xml}\n'
        f'{labeling_xml}\n'
        f'  </maplayer>'
    )


def _wms_maplayer_xml() -> str:
    return (
        f'  <maplayer type="raster" hasScaleBasedVisibilityFlag="0">\n'
        f'    <id>{_INFRA_WMS_ID}</id>\n'
        f'    <datasource>{_INFRA_WMS_SRC}</datasource>\n'
        f'    <layername>{_INFRA_WMS_NAME}</layername>\n'
        f'    <provider encoding="">wms</provider>\n'
        f'    <srs>{_INFRA_SRS_BLOCK}</srs>\n'
        f'  </maplayer>'
    )


def _boundary_maplayer_xml(layer_id: str, gpkg_relpath: str,
                            display_name: str, color: str = '0,0,0,255') -> str:
    """<maplayer> XML for a polygon boundary: dashed outline, no fill."""
    return (
        f'  <maplayer geometry="Polygon" type="vector" hasScaleBasedVisibilityFlag="0">\n'
        f'    <id>{layer_id}</id>\n'
        f'    <datasource>{gpkg_relpath}</datasource>\n'
        f'    <layername>{display_name}</layername>\n'
        f'    <provider encoding="UTF-8">ogr</provider>\n'
        f'    <srs>{_INFRA_SRS_BLOCK}</srs>\n'
        f'    <renderer-v2 type="singleSymbol" symbollevels="0" forceraster="0" enableorderby="0">\n'
        f'      <symbols>\n'
        f'        <symbol alpha="1" clip_to_extent="1" type="fill" name="0" force_rhr="0">\n'
        f'          <layer pass="0" class="SimpleFill" locked="0" enabled="1">\n'
        f'            <prop k="color" v="0,0,0,0"/>\n'
        f'            <prop k="style" v="no"/>\n'
        f'            <prop k="outline_color" v="{color}"/>\n'
        f'            <prop k="outline_style" v="dash"/>\n'
        f'            <prop k="outline_width" v="0.6"/>\n'
        f'            <prop k="outline_width_unit" v="MM"/>\n'
        f'            <prop k="joinstyle" v="miter"/>\n'
        f'            <prop k="offset" v="0,0"/>\n'
        f'            <prop k="offset_unit" v="MM"/>\n'
        f'          </layer>\n'
        f'        </symbol>\n'
        f'      </symbols>\n'
        f'      <rotation/>\n'
        f'      <sizescale/>\n'
        f'    </renderer-v2>\n'
        f'  </maplayer>'
    )


def _build_infra_qgz(
    qgz_path: str,
    version_dir: Path,
    name: str = None,
    nodes_file: str = 'nodes.gpkg',
    segments_file: str = 'segments.gpkg',
) -> None:
    """Write a QGIS .qgz infrastructure project alongside the network geopackages.

    Layer order (top = rendered last = on top):
      1. Stations/halts — white circles, black outline, CODE label
      2. Catchment area boundary — dashed black outline, no fill
      3. Study area boundary    — dashed dark-red outline, no fill
      4. Segments — black parallel lines scaled by num_tracks
      5. Swisstopo WMS basemap

    Args:
        qgz_path: absolute path to write the .qgz file.
        version_dir: directory containing the gpkg files (used to derive the
            project title when name is not provided).
        name: optional label override for the project title and layer names.
            Defaults to version_dir.name.
        nodes_file: relative filename of the nodes geopackage (layer name
            inside the file must be 'nodes').
        segments_file: relative filename of the segments geopackage (layer
            name inside the file must be 'segments').
    """
    version = name if name is not None else version_dir.name
    seg_id  = f'seg_{uuid.uuid4().hex[:8]}'
    nd_id   = f'nd_{uuid.uuid4().hex[:8]}'
    ca_id   = f'ca_{uuid.uuid4().hex[:8]}'
    sa_id   = f'sa_{uuid.uuid4().hex[:8]}'

    # Relative paths from the version directory to boundary files
    # data/Infrastructure/<version>/ → ../../Catchment_Area/Boundaries/
    ca_relpath = '../../Catchment_Area/Boundaries/catchment_area_boundary.gpkg'
    sa_relpath = '../../Catchment_Area/Boundaries/study_area_boundary.gpkg'

    seg_block = _segments_maplayer_xml(seg_id, segments_file,
                                        f'Segments — {version}')
    nd_block  = _nodes_maplayer_xml(nd_id, nodes_file,
                                    f'Nodes — {version}')
    ca_block  = _boundary_maplayer_xml(ca_id, ca_relpath,
                                        'Catchment Area Boundary', '0,0,0,255')
    sa_block  = _boundary_maplayer_xml(sa_id, sa_relpath,
                                        'Study Area Boundary', '139,0,0,255')
    wms_block = _wms_maplayer_xml()

    tree_xml = (
        f'    <layer-tree-layer id="{nd_id}" name="Nodes — {version}" '
        f'checked="Qt::Checked" expanded="1" '
        f'source="{nodes_file}|layername=nodes" providerKey="ogr"/>\n'
        f'    <layer-tree-layer id="{ca_id}" name="Catchment Area Boundary" '
        f'checked="Qt::Checked" expanded="0" '
        f'source="{ca_relpath}" providerKey="ogr"/>\n'
        f'    <layer-tree-layer id="{sa_id}" name="Study Area Boundary" '
        f'checked="Qt::Checked" expanded="0" '
        f'source="{sa_relpath}" providerKey="ogr"/>\n'
        f'    <layer-tree-layer id="{seg_id}" name="Segments — {version}" '
        f'checked="Qt::Checked" expanded="1" '
        f'source="{segments_file}|layername=segments" providerKey="ogr"/>\n'
        f'    <layer-tree-layer id="{_INFRA_WMS_ID}" name="{_INFRA_WMS_NAME}" '
        f'checked="Qt::Checked" expanded="0" '
        f'source="{_INFRA_WMS_SRC}" providerKey="wms"/>'
    )

    qgs = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis projectname="{version} Infrastructure" version="{_QGIS_VERSION}">
  <homePath path=""/>
  <title>{version} Infrastructure</title>
  <autotransaction active="0"/>
  <evaluateDefaultValues active="0"/>
  <trust active="0"/>
  <projectCrs>
    {_INFRA_SRS_BLOCK}
  </projectCrs>
  <layer-tree-group>
    <customproperties/>
{tree_xml}
    <custom-order enabled="0"/>
  </layer-tree-group>
  <projectlayers>
{nd_block}
{ca_block}
{sa_block}
{seg_block}
{wms_block}
  </projectlayers>
  <mapcanvas name="theMapCanvas">
    <units>meters</units>
    <rotation>0</rotation>
    <destinationsrs>
      {_INFRA_SRS_BLOCK}
    </destinationsrs>
  </mapcanvas>
</qgis>
"""
    with zipfile.ZipFile(qgz_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('project.qgs', qgs)


# =============================================================================
# Graph Building
# =============================================================================

def build_networkx_graph(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
) -> nx.Graph:
    """
    Build a NetworkX graph from macroscopic nodes and segments.

    Node key  : Name (station/junction name)
    Node attrs: Code, N, E, Node_Class, geometry
    Edge attrs: ID, Length, Num_Tracks, Gauge, Electrification_Class, geometry
    """
    G = nx.Graph()

    for _, row in nodes.iterrows():
        name = row.get('Name')
        if pd.isna(name):
            continue
        G.add_node(
            name,
            Code=row.get('Code', ''),
            N=row.get('N', 0),
            E=row.get('E', 0),
            Node_Class=row.get('Node_Class', 'unknown'),
            geometry=row.geometry,
        )

    skipped = 0
    for _, row in segments.iterrows():
        fn = row.get('From_Name')
        tn = row.get('To_Name')
        if pd.isna(fn) or pd.isna(tn):
            skipped += 1
            continue
        if fn not in G or tn not in G:
            skipped += 1
            continue
        G.add_edge(
            fn, tn,
            segment_id=row.get('Segment_ID'),
            length_m=row.get('Length', 0),
            num_tracks=row.get('Num_Tracks', 1),
            gauge=row.get('Gauge', 1435),
            electrification=row.get('Electrification_Class', 'unknown'),
            geometry=row.geometry,
        )

    if skipped:
        print(f"  Skipped {skipped} segments (missing node reference)")
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


# =============================================================================
# Node Classification
# =============================================================================

def _classify_nodes(nodes: gpd.GeoDataFrame):
    """Split nodes GDF into (train_stations, tram_funicular, junctions).

    train_stations : node_class == 'station' AND transport_mode contains 'train'
                     OR node_class == 'abandoned_station'
    tram_funicular : Node_Class == 'station' AND Transport_Mode contains
                     'tram', 'funicular', or 'cog_railway' (and NOT 'train')
    junctions      : Node_Class == 'junction'
    """
    if nodes is None or nodes.empty:
        empty = gpd.GeoDataFrame()
        return empty, empty, empty

    nc = (nodes['Node_Class']     if 'Node_Class'     in nodes.columns
          else pd.Series([''] * len(nodes), index=nodes.index))
    tm = (nodes['Transport_Mode'] if 'Transport_Mode' in nodes.columns
          else pd.Series([''] * len(nodes), index=nodes.index))

    is_station = nc.astype(str) == 'station'
    is_abandoned = nc.astype(str) == 'abandoned_station'
    has_train  = tm.astype(str).str.contains('train', na=False)
    has_tram_f = tm.astype(str).str.contains(
        'tram|funicular|cog_railway', na=False, regex=True)

    train_stations = nodes[is_station & has_train | is_abandoned]
    tram_funicular = nodes[is_station & has_tram_f & ~has_train]
    junctions      = nodes[nc.astype(str) == 'junction']
    return train_stations, tram_funicular, junctions


# =============================================================================
# Plotting Utilities
# =============================================================================

def _get_line_width(num_tracks: int) -> float:
    return 1.0 + (num_tracks - 1) * 0.5


_SCALE_BAR_NICE_KM = [1, 2, 5, 10, 20, 50, 100, 200, 500]


def _add_north_arrow(ax, location='upper left', scale=0.5):
    """Draw a north arrow using matplotlib_map_utils."""
    north_arrow(ax, location=location, scale=scale, rotation={"degrees": 0})


def _add_scale_bar(ax, location=(0.97, 0.020)):
    """Adaptive scale bar with 2-4 alternating black/white cells.

    All coordinates are in axes fraction so the bar stays stable regardless
    of aspect ratio or tight_layout timing.  location is (right_edge, bottom).
    Bar length is derived from ax.get_xlim() and expressed as a fraction of
    the axes width so the labelled distance remains accurate.
    """
    xlim  = ax.get_xlim()
    map_w = xlim[1] - xlim[0]

    target_km = (map_w / 4.0) / 1000.0
    total_km  = min(_SCALE_BAR_NICE_KM, key=lambda v: abs(v - target_km))
    n_cells   = 4 if total_km >= 4 else 2
    cell_m    = (total_km * 1000.0) / n_cells
    cell_frac = cell_m / map_w  # axes-fraction width per cell

    x0    = location[0] - n_cells * cell_frac  # left edge, anchored from right
    y0    = location[1]
    bar_h = 0.012  # axes fraction

    for i in range(n_cells):
        color = 'black' if i % 2 == 0 else 'white'
        rect = Rectangle(
            (x0 + i * cell_frac, y0), cell_frac, bar_h,
            facecolor=color, edgecolor='black', linewidth=0.6,
            transform=ax.transAxes, zorder=7,
        )
        ax.add_patch(rect)

    for i in range(n_cells + 1):
        val_km = (i * cell_m) / 1000.0
        label  = (f'{val_km:.0f} km' if val_km == int(val_km)
                  else f'{val_km:.1f} km')
        ax.text(
            x0 + i * cell_frac, y0 + bar_h * 1.6,
            label, ha='center', va='bottom', fontsize=7,
            transform=ax.transAxes, zorder=7,
        )


def _clip_to_boundary(gdf: gpd.GeoDataFrame,
                      boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Clip gdf geometries to boundary; returns original if clip fails."""
    if gdf is None or gdf.empty or boundary is None or boundary.empty:
        return gdf
    try:
        clipped = gpd.clip(gdf, boundary)
        return clipped if not clipped.empty else gdf
    except Exception:
        return gdf


def _clip_to_extent(gdf: gpd.GeoDataFrame, extent) -> gpd.GeoDataFrame:
    """Clip gdf geometries precisely to the extent bounding box to fix mask bounds."""
    if gdf is None or gdf.empty or extent is None:
        return gdf
    try:
        from shapely.geometry import box as shapely_box
        bbox = gpd.GeoDataFrame(
            geometry=[shapely_box(extent[0], extent[2], extent[1], extent[3])],
            crs=gdf.crs
        )
        clipped = gpd.clip(gdf, bbox)
        return clipped if not clipped.empty else gpd.GeoDataFrame(columns=gdf.columns).set_crs(gdf.crs)
    except Exception:
        return gdf


def _plot_lakes(ax, boundary: Optional[gpd.GeoDataFrame] = None,
                extent=None) -> None:
    """Overlay lakes from swissTLMRegio.

    Clipping priority:
      1. extent (xmin, xmax, ymin, ymax) — used for SA plots, clips to plot frame
      2. boundary GeoDataFrame            — used for CA plots, clips to study polygon
      3. Neither provided                 — no clipping (full coverage)
    """
    from shapely.geometry import box as shapely_box
    lakes_path = Path(paths.MAIN) / paths.LAKES_SHP
    if not lakes_path.exists():
        return
    try:
        lakes = gpd.read_file(lakes_path)
        if lakes.crs is None:
            lakes = lakes.set_crs('EPSG:2056')
        elif lakes.crs.to_epsg() != 2056:
            lakes = lakes.to_crs('EPSG:2056')

        if extent is not None:
            clip_geom = gpd.GeoDataFrame(
                geometry=[shapely_box(extent[0], extent[2], extent[1], extent[3])],
                crs='EPSG:2056',
            )
            lakes = gpd.clip(lakes, clip_geom)
        elif boundary is not None and not boundary.empty:
            lakes = _clip_to_boundary(lakes, boundary)

        if not lakes.empty:
            lakes.plot(ax=ax, facecolor='#a8cfe0', edgecolor='#5a9ab5',
                       linewidth=0.4, zorder=1)
    except Exception as exc:
        print(f"  Warning: could not load lakes — {exc}")


def _base_plot(network: NetworkData, title: str, figsize: Tuple[int, int],
               extent=None):
    """Create figure, draw boundary, return (fig, ax).

    extent : optional (xmin, xmax, ymin, ymax) in map coordinates to set the
             axes limits.  When None the view auto-fits the plotted data.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('E [m]', fontsize=10)
    ax.set_ylabel('N [m]', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if network.boundary is not None:
        network.boundary.plot(
            ax=ax, facecolor='none', edgecolor='black',
            linewidth=1.5, linestyle='--', alpha=0.6,
        )
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    return fig, ax


# =============================================================================
# Plot Functions
# =============================================================================

def plot_infrastructure(
    network: NetworkData,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show_labels: bool = True,
    is_catchment: bool = False,
) -> plt.Figure:
    """Infrastructure overview: segments coloured by track count, nodes by class."""
    if title is None:
        title = f"Infrastructure — {network.version}"
    fig, ax = _base_plot(network, title, figsize)
    _plot_lakes(ax, network.boundary)

    segs  = _clip_to_boundary(network.segments, network.boundary)
    nodes = _clip_to_boundary(network.nodes,    network.boundary)

    tracks_present = set()
    has_other_tracks = False

    # Segments by track count
    if len(segs) > 0:
        for n_tracks, color in TRACK_COLORS.items():
            mask = segs['Num_Tracks'] == n_tracks
            if mask.any():
                tracks_present.add(n_tracks)
                for _, row in segs[mask].iterrows():
                    _draw_parallel_tracks(
                        ax,
                        geom=row.geometry,
                        num_tracks=n_tracks,
                        color='black',
                        linewidth=1.5,
                        alpha=0.75,
                        zorder=2,
                        track_spacing_m=_TRACK_SPACING_M_CA if is_catchment else _TRACK_SPACING_M_SA
                    )
        other = ~segs['Num_Tracks'].isin(TRACK_COLORS)
        if other.any():
            has_other_tracks = True
            for _, row in segs[other].iterrows():
                _draw_parallel_tracks(
                    ax,
                    geom=row.geometry,
                    num_tracks=1,
                    color=TRACK_DEFAULT,
                    linewidth=1.5,
                    alpha=0.7,
                    zorder=2,
                    track_spacing_m=_TRACK_SPACING_M_CA if is_catchment else _TRACK_SPACING_M_SA
                )

    nodes_present = set()

    # Nodes by class
    if len(nodes) > 0:
        for cls, color in NODE_COLORS.items():
            mask = nodes['Node_Class'] == cls
            if mask.any():
                nodes_present.add(cls)
                size   = 80 if cls == 'station' else 30
                marker = 'o' if cls == 'station' else 's'
                nodes[mask].plot(
                    ax=ax, color=color, markersize=size, marker=marker,
                    alpha=0.85, zorder=4,
                    edgecolor='black' if cls == 'station' else None,
                    linewidth=1 if cls == 'station' else 0,
                )

        if show_labels:
            stations = nodes[nodes['Node_Class'] == 'station']
            for _, row in stations.iterrows():
                name = row.get('Name') or row.get('Code', '')
                if pd.notna(name):
                    ax.annotate(
                        str(name)[:25],
                        xy=(row.geometry.x, row.geometry.y),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=7, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  edgecolor='gray', alpha=0.8),
                        zorder=6,
                    )

    # Legend
    legend = []
    handler_map = {}
    
    if tracks_present or has_other_tracks:
        legend.append(Line2D([0], [0], color='none', label='─ Tracks ─'))
        for n in sorted(tracks_present):
            agg_linewidth = 1.5 if n == 1 else n * 1.35 
            handle = Line2D([0], [0], color='black', linewidth=agg_linewidth, label=f'{n} track(s)')
            legend.append(handle)
            handler_map[handle] = _MultiTrackLegendHandler()
            handle.track_count = n
            handle.total_width = agg_linewidth
            handle.color = 'black'
            
        if has_other_tracks:
            legend.append(Line2D([0], [0], color=TRACK_DEFAULT, linewidth=1, label='Other'))
    
    if nodes_present:
        if tracks_present or has_other_tracks:
            legend.append(Line2D([0], [0], color='none', label=''))
        legend.append(Line2D([0], [0], color='none', label='─ Nodes ─'))
        for cls, color in NODE_COLORS.items():
            if cls in nodes_present:
                marker = 'o' if cls == 'station' else 's'
                size   = 10 if cls == 'station' else 7
                legend.append(Line2D([0], [0], marker=marker, color='w',
                                     markerfacecolor=color, markersize=size,
                                     markeredgecolor='black' if cls == 'station' else None,
                                     label=cls.replace('_', ' ').title()))

    if legend:
        _lgnd = ax.legend(handles=legend, handler_map=handler_map,
                          loc='upper right', fontsize=8, title='Infrastructure')
        _lgnd.get_title().set_fontweight('bold')

    _add_north_arrow(ax)
    _add_scale_bar(ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"  Saved → {output_path}")
    return fig


def plot_gauge_map(
    network: NetworkData,
    extent=None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show_outside: bool = False,
    is_catchment: bool = False,
) -> plt.Figure:
    """OpenRailwayMap-style gauge map."""
    if title is None:
        title = f"Gauge Map — {network.version}"
    fig, ax = _base_plot(network, title, figsize, extent=extent)
    ax.set_facecolor('#f5f5f5')
    if show_outside:
        _plot_lakes(ax, extent=extent)
    else:
        _plot_lakes(ax, boundary=network.boundary)

    GAUGE_LABELS = {
        1435: '1435 mm (Standard)',
        1668: '1668 mm (Iberian)',
        1520: '1520 mm (Russian)',
        1000: '1000 mm (Metre)',
        900:  '900 mm',
        800:  '800 mm',
        750:  '750 mm',
        600:  '600 mm',
    }

    segs_all = _clip_to_extent(network.segments, extent)
    segs_in  = (_clip_to_boundary(network.segments, network.boundary)
                if (show_outside and network.boundary is not None
                    and not network.boundary.empty)
                else None)
    segs     = _clip_to_boundary(network.segments, network.boundary)

    gauges_present = set()
    if show_outside and segs_in is not None:
        # Ghost pass
        if len(segs_all) > 0:
            for gauge, color in GAUGE_COLORS.items():
                mask = segs_all['Gauge'] == gauge
                if mask.any():
                    gauges_present.add(gauge)
                    segs_all[mask].plot(ax=ax, color=color, linewidth=2.5, alpha=0.40)
            unknown = ~segs_all['Gauge'].isin(GAUGE_COLORS)
            if unknown.any():
                segs_all[unknown].plot(ax=ax, color=GAUGE_DEFAULT, linewidth=2, alpha=0.40)
                gauges_present.add('unknown')
        # Solid pass
        if len(segs_in) > 0:
            for gauge, color in GAUGE_COLORS.items():
                mask = segs_in['Gauge'] == gauge
                if mask.any():
                    segs_in[mask].plot(ax=ax, color=color, linewidth=2.5, alpha=1.0)
            unknown = ~segs_in['Gauge'].isin(GAUGE_COLORS)
            if unknown.any():
                segs_in[unknown].plot(ax=ax, color=GAUGE_DEFAULT, linewidth=2, alpha=1.0)
    else:
        if len(segs) > 0:
            for gauge, color in GAUGE_COLORS.items():
                mask = segs['Gauge'] == gauge
                if mask.any():
                    gauges_present.add(gauge)
                    segs[mask].plot(ax=ax, color=color, linewidth=2.5, alpha=0.85)
            unknown = ~segs['Gauge'].isin(GAUGE_COLORS)
            if unknown.any():
                segs[unknown].plot(ax=ax, color=GAUGE_DEFAULT, linewidth=2, alpha=0.7)
                gauges_present.add('unknown')

    legend = []
    for gauge, color in GAUGE_COLORS.items():
        if gauge in gauges_present:
            legend.append(Line2D([0], [0], color=color, linewidth=3,
                                 label=GAUGE_LABELS.get(gauge, f'{gauge} mm')))
    if 'unknown' in gauges_present:
        legend.append(Line2D([0], [0], color=GAUGE_DEFAULT, linewidth=2, label='Unknown'))
    _lgnd = ax.legend(handles=legend, loc='upper right', fontsize=10,
                      title='Track Gauge', title_fontsize=11)
    _lgnd.get_title().set_fontweight('bold')

    _add_north_arrow(ax)
    _add_scale_bar(ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"  Saved → {output_path}")
    return fig


def plot_electrification_map(
    network: NetworkData,
    extent=None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show_outside: bool = False,
    is_catchment: bool = False,
) -> plt.Figure:
    """OpenRailwayMap-style electrification map (English legend, German-value-aware)."""
    if title is None:
        title = f"Electrification — {network.version}"
    fig, ax = _base_plot(network, title, figsize, extent=extent)
    ax.set_facecolor('#f5f5f5')

    if show_outside:
        _plot_lakes(ax, extent=extent)
    else:
        _plot_lakes(ax, boundary=network.boundary)

    segs_all = _clip_to_extent(network.segments, extent)
    segs_in  = (_clip_to_boundary(network.segments, network.boundary)
                if (show_outside and network.boundary is not None and not network.boundary.empty)
                else None)
    segs     = _clip_to_boundary(network.segments, network.boundary)

    cats_present = set()
    
    if show_outside and segs_in is not None:
        if len(segs_all) > 0 and 'Electrification_Class' in segs_all.columns:
            elec_cat_all = segs_all['Electrification_Class'].map(_ELEC_CLASS_PLOT).fillna('unknown')
            for cat, color in ELECTRIFICATION_COLORS.items():
                mask = elec_cat_all == cat
                if mask.any():
                    cats_present.add(cat)
                    lw = 2 if cat == 'no_electrification' else 3
                    segs_all[mask].plot(ax=ax, color=color, linewidth=lw, alpha=0.40, zorder=2)
                    
        if len(segs_in) > 0 and 'Electrification_Class' in segs_in.columns:
            elec_cat_in = segs_in['Electrification_Class'].map(_ELEC_CLASS_PLOT).fillna('unknown')
            for cat, color in ELECTRIFICATION_COLORS.items():
                mask = elec_cat_in == cat
                if mask.any():
                    cats_present.add(cat)
                    lw = 2 if cat == 'no_electrification' else 3
                    segs_in[mask].plot(ax=ax, color=color, linewidth=lw, alpha=0.85, zorder=3)
    else:
        if len(segs) > 0 and 'Electrification_Class' in segs.columns:
            elec_cat = segs['Electrification_Class'].map(_ELEC_CLASS_PLOT).fillna('unknown')
            for cat, color in ELECTRIFICATION_COLORS.items():
                mask = elec_cat == cat
                if mask.any():
                    cats_present.add(cat)
                    lw = 2 if cat == 'no_electrification' else 3
                    segs[mask].plot(ax=ax, color=color, linewidth=lw, alpha=0.85, zorder=2)


    legend = [
        Line2D([0], [0], color=ELECTRIFICATION_COLORS[cat], linewidth=3,
               label=ELECTRIFICATION_LABELS[cat])
        for cat in ELECTRIFICATION_COLORS
        if cat in cats_present
    ]
    _lgnd = ax.legend(handles=legend, loc='upper right', fontsize=9,
                      title='Electrification', title_fontsize=10)
    _lgnd.get_title().set_fontweight('bold')

    _add_north_arrow(ax)
    _add_scale_bar(ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"  Saved → {output_path}")
    return fig


def plot_construct_type_map(
    network: NetworkData,
    extent=None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show_outside: bool = False,
    is_catchment: bool = False,
) -> plt.Figure:
    """Tunnel / bridge / gallery construct type map."""
    if title is None:
        title = f"Construct Type — {network.version}"
    fig, ax = _base_plot(network, title, figsize, extent=extent)
    ax.set_facecolor('#f5f5f5')
    
    if show_outside:
        _plot_lakes(ax, extent=extent)
    else:
        _plot_lakes(ax, boundary=network.boundary)

    segs     = _clip_to_boundary(network.segments, network.boundary)

    construct_present = set()
    if len(segs) > 0 and 'construct_type' in segs.columns:
        for ctype, color in CONSTRUCT_COLORS.items():
            mask = segs['construct_type'] == ctype
            if mask.any():
                construct_present.add(ctype)
                lw = 3 if ctype in ('tunnel', 'bridge') else 2
                segs[mask].plot(ax=ax, color=color, linewidth=lw, alpha=0.85)
        unknown = ~segs['construct_type'].isin(CONSTRUCT_COLORS)
        if unknown.any():
            segs[unknown].plot(ax=ax, color=CONSTRUCT_DEFAULT, linewidth=2, alpha=0.7)
            construct_present.add('unknown')
    else:
        segs.plot(ax=ax, color='gray', linewidth=2, alpha=0.7)

    if construct_present:
        legend = []
        for ctype, color in CONSTRUCT_COLORS.items():
            if ctype in construct_present:
                legend.append(Line2D([0], [0], color=color, linewidth=3,
                                     label=ctype.title()))
        if 'unknown' in construct_present:
            legend.append(Line2D([0], [0], color=CONSTRUCT_DEFAULT, linewidth=2,
                                 label='Unknown'))
        _lgnd = ax.legend(handles=legend, loc='upper right', fontsize=10,
                          title='Construct Type', title_fontsize=11)
        _lgnd.get_title().set_fontweight('bold')

    _add_north_arrow(ax)
    _add_scale_bar(ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"  Saved → {output_path}")
    return fig


# =============================================================================
# Canonical Infrastructure Plot
# =============================================================================

class _MultiTrackLegendHandle:
    """Placeholder object for rendering multi-track legend entries."""

    def __init__(self, total_width: float, track_count: int, color: str = "black"):
        self.total_width = total_width
        self.track_count = track_count
        self.color = color


class _MultiTrackLegendHandler(HandlerBase):
    """Custom legend handler that renders multiple parallel lines for tracks."""

    def create_artists(
        self,
        legend,
        orig_handle: "_MultiTrackLegendHandle",
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        x0 = xdescent
        x1 = xdescent + width
        y = ydescent + height / 2.0

        track_count = orig_handle.track_count
        gap_factor = 0.4
        
        if track_count == 1:
            individual_line_width = orig_handle.total_width
            line_spacing = 0
        else:
            individual_line_width = orig_handle.total_width / (track_count + (track_count - 1) * gap_factor)
            line_spacing = individual_line_width * (1 + gap_factor)

        artists = []
        for track_idx in range(track_count):
            offset_y = y + (track_idx - (track_count - 1) / 2.0) * line_spacing
            line = Line2D([x0, x1], [offset_y, offset_y], color=orig_handle.color, linewidth=individual_line_width, solid_capstyle="round")
            line.set_transform(trans)
            artists.append(line)

        return artists


_TRACK_SPACING_M_CA = 140   # CA track spacing
_TRACK_SPACING_M_SA = 60    # SA track spacing


def _draw_parallel_tracks(ax, geom, num_tracks: int,
                           color: str = 'black', linewidth: float = 1.05, alpha: float = 1.0, zorder: int = 3, track_spacing_m: float = 30, linestyle: str = 'solid') -> None:
    """Draw num_tracks parallel offset lines for one segment geometry."""
    if num_tracks <= 0 or geom is None or geom.is_empty:
        return
    sub_lines = list(geom.geoms) if geom.geom_type == 'MultiLineString' else [geom]
    offsets = [(i - (num_tracks - 1) / 2) * track_spacing_m
               for i in range(num_tracks)]

    for line in sub_lines:
        if line.length < 1:
            continue
        for offset_m in offsets:
            if abs(offset_m) < 0.01:
                draw_line = line
            else:
                side = 'left' if offset_m > 0 else 'right'
                try:
                    draw_line = line.parallel_offset(
                        abs(offset_m), side, resolution=8,
                        join_style=2, mitre_limit=5,
                    )
                    if draw_line is None or draw_line.is_empty:
                        draw_line = line
                except Exception:
                    draw_line = line

            parts = (list(draw_line.geoms)
                     if draw_line.geom_type == 'MultiLineString' else [draw_line])
            for part in parts:
                if not part.is_empty and len(part.coords) >= 2:
                    xs, ys = zip(*[(c[0], c[1]) for c in part.coords])
                    ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle,
                            solid_capstyle='round', alpha=alpha, zorder=zorder)


def _draw_seg_category(ax, segs_gdf, color, linewidth, alpha, zorder, ts):
    """Draw all segments of one diff category, merged per Num_Tracks group.

    Groups segments by Num_Tracks, merges each group's geometries into a
    single MultiLineString via unary_union + linemerge, then calls
    _draw_parallel_tracks once per group. This eliminates the per-segment
    alpha-compositing that produces darker blobs at junction nodes where
    many segments share an endpoint — with alpha=1.0 and round caps the
    result is a uniform solid-colour network.
    """
    if segs_gdf is None or segs_gdf.empty:
        return
    if 'Num_Tracks' in segs_gdf.columns:
        nt_series = segs_gdf['Num_Tracks'].fillna(1).astype(int).clip(lower=1)
    else:
        nt_series = pd.Series(1, index=segs_gdf.index)
    for n_tracks in sorted(nt_series.unique()):
        group = segs_gdf[nt_series == n_tracks]
        valid = [g for g in group.geometry if g is not None and not g.is_empty]
        if not valid:
            continue
        union = unary_union(valid)
        # linemerge only accepts collections; if unary_union produced a single
        # LineString (all segments connected into one), pass it directly.
        merged = union if union.geom_type in ('LineString', 'LinearRing') else linemerge(union)
        _draw_parallel_tracks(ax, merged, int(n_tracks),
                              color=color, linewidth=linewidth,
                              alpha=alpha, zorder=zorder, track_spacing_m=ts)


def _draw_parallel_tracks_mixed(
    ax, geom, colors: list,
    linewidth: float = 1.05, alpha: float = 1.0,
    zorder: int = 3, track_spacing_m: float = 30,
) -> None:
    """Like _draw_parallel_tracks but each track gets its own colour.

    len(colors) determines how many parallel lines are drawn.
    The outermost track (last offset) gets the last colour — use this for the
    diff track (green = gained, red = lost).
    """
    num_tracks = len(colors)
    if num_tracks == 0 or geom is None or geom.is_empty:
        return
    sub_lines = list(geom.geoms) if geom.geom_type == 'MultiLineString' else [geom]
    offsets = [(i - (num_tracks - 1) / 2) * track_spacing_m for i in range(num_tracks)]

    for line in sub_lines:
        if line.length < 1:
            continue
        for offset_m, color in zip(offsets, colors):
            if abs(offset_m) < 0.01:
                draw_line = line
            else:
                side = 'left' if offset_m > 0 else 'right'
                try:
                    draw_line = line.parallel_offset(
                        abs(offset_m), side, resolution=8,
                        join_style=2, mitre_limit=5,
                    )
                    if draw_line is None or draw_line.is_empty:
                        draw_line = line
                except Exception:
                    draw_line = line

            parts = (list(draw_line.geoms)
                     if draw_line.geom_type == 'MultiLineString' else [draw_line])
            for part in parts:
                if not part.is_empty and len(part.coords) >= 2:
                    xs, ys = zip(*[(c[0], c[1]) for c in part.coords])
                    ax.plot(xs, ys, color=color, linewidth=linewidth,
                            solid_capstyle='butt', alpha=alpha, zorder=zorder)


def plot_infrastructure_canonical(
    network: NetworkData,
    extent=None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show_labels: bool = True,
    is_catchment: bool = False,
    show_outside: bool = False,
) -> plt.Figure:
    """Infrastructure overview: black parallel tracks by count,
    white station circles with CODE labels."""
    if title is None:
        title = f"Infrastructure — {network.version}"
    fig, ax = _base_plot(network, title, figsize, extent=extent)
    if show_outside:
        _plot_lakes(ax, extent=extent)
    else:
        _plot_lakes(ax, boundary=network.boundary)

    segs_all = _clip_to_extent(network.segments, extent)
    segs_in  = (_clip_to_boundary(network.segments, network.boundary)
                if (show_outside and network.boundary is not None and not network.boundary.empty)
                else None)
    segs  = _clip_to_boundary(network.segments, network.boundary)

    tracks_present = set()

    # Segments as back parallel lines (one line per track)
    if show_outside and segs_in is not None:
        if len(segs_all) > 0:
            for _, row in segs_all.iterrows():
                n = int(row.get('Num_Tracks', 1) or 1)
                n = max(1, min(n, 4))
                tracks_present.add(n)
                _draw_parallel_tracks(ax, row.geometry, n, color='black', alpha=0.40, zorder=2, track_spacing_m=_TRACK_SPACING_M_CA if is_catchment else _TRACK_SPACING_M_SA)
        if len(segs_in) > 0:
            for _, row in segs_in.iterrows():
                n = int(row.get('Num_Tracks', 1) or 1)
                n = max(1, min(n, 4))
                tracks_present.add(n)
                _draw_parallel_tracks(ax, row.geometry, n, color='black', alpha=1.0, zorder=3, track_spacing_m=_TRACK_SPACING_M_CA if is_catchment else _TRACK_SPACING_M_SA)
    else:
        if len(segs) > 0:
            for _, row in segs.iterrows():
                n = int(row.get('Num_Tracks', 1) or 1)
                n = max(1, min(n, 4))
                tracks_present.add(n)
                _draw_parallel_tracks(ax, row.geometry, n, color='black', alpha=1.0, zorder=3, track_spacing_m=_TRACK_SPACING_M_CA if is_catchment else _TRACK_SPACING_M_SA)

    # Nodes: three layers via _classify_nodes
    ts_all, tf_all, jn_all = _classify_nodes(network.nodes)

    if show_outside and network.boundary is not None and not network.boundary.empty:
        nodes_in    = _clip_to_boundary(network.nodes, network.boundary)
        ts_in, tf_in, jn_in = _classify_nodes(nodes_in)
        # Ghost pass — full network at 40%
        _ms_ts = 20  if is_catchment else 60
        _ms_tf = 5   if is_catchment else 25
        _ms_jn = 5  if is_catchment else 5
        
        if len(ts_all) > 0:
            ts_all.plot(ax=ax, facecolor='white', edgecolor='black',
                        markersize=_ms_ts, marker='o', linewidth=1.2,
                        alpha=0.40, zorder=4)
        if len(tf_all) > 0:
            tf_all.plot(ax=ax, facecolor='black', edgecolor='black',
                        markersize=_ms_tf, marker='o', linewidth=0.8,
                        alpha=0.40, zorder=4)
        if len(jn_all) > 0:
            jn_all.plot(ax=ax, color='#888888', markersize=_ms_jn, marker='o',
                        alpha=0.40, zorder=4)
        # Solid pass — clipped network at full opacity
        ts_draw, tf_draw, jn_draw = ts_in, tf_in, jn_in
    else:
        nodes_disp = _clip_to_boundary(network.nodes, network.boundary)
        ts_draw, tf_draw, jn_draw = _classify_nodes(nodes_disp)

    _ms_ts = 20  if is_catchment else 60
    _ms_tf = 5   if is_catchment else 25
    _ms_jn = 5  if is_catchment else 5

    if len(ts_draw) > 0:
        ts_draw.plot(ax=ax, facecolor='white', edgecolor='black',
                     markersize=_ms_ts, marker='o', linewidth=1.2,
                     alpha=1.0, zorder=5)
        if show_labels and not is_catchment:
            for _, row in ts_draw.iterrows():
                code = row.get('Code', '')
                if pd.notna(code) and str(code).strip():
                    ax.annotate(
                        str(code),
                        xy=(row.geometry.x, row.geometry.y),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=7, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                  edgecolor='none', alpha=0.7),
                        zorder=6,
                    )

    if len(tf_draw) > 0:
        tf_draw.plot(ax=ax, facecolor='black', edgecolor='black',
                     markersize=_ms_tf, marker='o', linewidth=0.8,
                     alpha=1.0, zorder=5)

    if len(jn_draw) > 0:
        jn_draw.plot(ax=ax, color='#888888', markersize=_ms_jn, marker='o',
                     alpha=1.0, zorder=4)

    legend = []
    handler_map = {}
    
    # Add track styles based on what was plotted
    for n in sorted(tracks_present):
        label_text = f'{n} track{"s" if n > 1 else ""}'
        if n == 4:
            label_text = '4+ tracks'
        
        # Increase total visual width for handler map sizing
        agg_linewidth = 1.5 if n == 1 else n * 1.35 
        handle = Line2D([0], [0], color='black', linewidth=agg_linewidth, label=label_text)
        legend.append(handle)
        handler_map[handle] = _MultiTrackLegendHandler()
        handle.track_count = n
        handle.total_width = agg_linewidth
        handle.color = 'black'
        
    has_ts = len(ts_all) > 0 if show_outside else len(ts_draw) > 0
    if has_ts:
        legend.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                             markeredgecolor='black', markersize=8, label='Train station'))
    
    has_tf = len(tf_all) > 0 if show_outside else len(tf_draw) > 0
    if has_tf:
        legend.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                             markeredgecolor='black', markersize=6, label='Tram / Funicular'))

    has_jn = len(jn_all) > 0 if show_outside else len(jn_draw) > 0
    if has_jn:
        legend.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888',
                   markersize=5, label='Junction / turning loop')
        )
    
    if legend:
        ax.legend(handles=legend, handler_map=handler_map, loc='upper right', fontsize=8)
    _add_north_arrow(ax)
    _add_scale_bar(ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"  Saved → {output_path}")
    return fig


# =============================================================================
# Diff Plot
# =============================================================================

_DIFF_RED    = '#d62728'
_DIFF_GREEN  = '#2ca02c'
_DIFF_BLACK  = "#A7A1A1D5"
_DIFF_YELLOW = '#B8860B'  


def export_infrastructure_diff(
    net_a: NetworkData,
    net_b: NetworkData,
    comp_a: Optional[gpd.GeoDataFrame] = None,
    comp_b: Optional[gpd.GeoDataFrame] = None,
    output_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Export the infrastructure diff between two versions to a two-sheet Excel file.

    Args:
        net_a: Reference network.
        net_b: Comparison network (what changed relative to net_a).
        comp_a: Composition GeoDataFrame for net_a (optional).
        comp_b: Composition GeoDataFrame for net_b (optional).
        output_path: .xlsx path. DataFrames are returned regardless.

    Returns:
        (nodes_df, segments_df) — one row per changed node / segment only
        (unchanged items are omitted).
    """
    def _norm(v):
        try:
            return '' if pd.isna(v) else str(v)
        except (TypeError, ValueError):
            return str(v)

    def _tc_int(v) -> int:
        x = pd.to_numeric(v, errors='coerce')
        return 0 if pd.isna(x) else int(x)

    _SEG_MOD_ATTRS = (
        'Gauge', 'Electrification_Class', 'Average_Speed',
        'Km_Start', 'Km_End', 'Route_Number', 'Route_Name', 'Route_Owner',
    )
    _NODE_MOD_ATTRS = ('Node_Class', 'Platform_Count', 'Transport_Mode')

    comp_has = (
        comp_a is not None and not comp_a.empty
        and comp_b is not None and not comp_b.empty
        and 'Segment_ID' in getattr(comp_a, 'columns', [])
        and 'Segment_ID' in getattr(comp_b, 'columns', [])
    )

    def _comp_diff(sid: str) -> str:
        if not comp_has:
            return ''
        pcs_a = comp_a[comp_a['Segment_ID'] == sid]
        pcs_b = comp_b[comp_b['Segment_ID'] == sid]
        if pcs_a.empty and pcs_b.empty:
            return ''

        def _dist(pcs):
            if pcs.empty or 'Engineering_Structure' not in pcs.columns or 'Piece_Length' not in pcs.columns:
                return {}
            return pcs.groupby('Engineering_Structure')['Piece_Length'].sum().round(0).to_dict()

        da, db = _dist(pcs_a), _dist(pcs_b)
        if da == db:
            return ''
        parts = []
        n_a, n_b = len(pcs_a), len(pcs_b)
        if n_a != n_b:
            parts.append(f'Pieces: {n_a}→{n_b}')
        for t in sorted(set(da) | set(db)):
            va, vb = da.get(t, 0), db.get(t, 0)
            if va != vb:
                parts.append(f'{t}: {va:.0f}→{vb:.0f} m')
        return ' | '.join(parts)

    # --- Segment diff ---
    segs_a = (net_a.segments.drop_duplicates('Segment_ID').set_index('Segment_ID')
              if not net_a.segments.empty else pd.DataFrame())
    segs_b = (net_b.segments.drop_duplicates('Segment_ID').set_index('Segment_ID')
              if not net_b.segments.empty else pd.DataFrame())

    ids_a = set(segs_a.index.dropna()) if not segs_a.empty else set()
    ids_b = set(segs_b.index.dropna()) if not segs_b.empty else set()

    seg_rows: list = []

    for sid in sorted(ids_a - ids_b):
        r = segs_a.loc[sid]
        seg_rows.append({
            'Segment_ID': sid,
            'From_Name': _norm(r.get('From_Name')),
            'To_Name': _norm(r.get('To_Name')),
            'Status': 'Removed',
            'Attribute_Changes': '',
            'Composition_Changes': '',
        })

    for sid in sorted(ids_b - ids_a):
        r = segs_b.loc[sid]
        seg_rows.append({
            'Segment_ID': sid,
            'From_Name': _norm(r.get('From_Name')),
            'To_Name': _norm(r.get('To_Name')),
            'Status': 'Added',
            'Attribute_Changes': '',
            'Composition_Changes': _comp_diff(sid),
        })

    for sid in sorted(ids_a & ids_b):
        ra, rb = segs_a.loc[sid], segs_b.loc[sid]
        n_a = int(ra.get('Num_Tracks', 1) or 1)
        n_b = int(rb.get('Num_Tracks', 1) or 1)
        comp_chg = _comp_diff(sid)

        changed = []
        if n_b != n_a:
            changed.append(f'Num_Tracks: {n_a}→{n_b}')
        changed += [a for a in _SEG_MOD_ATTRS
                    if _norm(ra.get(a)) != _norm(rb.get(a))]
        if not ra['geometry'].equals(rb['geometry']):
            changed.append('geometry')
        if changed or comp_chg:
            status, attr_chg = 'Modified', ', '.join(changed)
        else:
            continue  # unchanged — omit

        seg_rows.append({
            'Segment_ID': sid,
            'From_Name': _norm(rb.get('From_Name')),
            'To_Name': _norm(rb.get('To_Name')),
            'Status': status,
            'Attribute_Changes': attr_chg,
            'Composition_Changes': comp_chg,
        })

    _SEG_ORDER = {'Added': 0, 'Removed': 1, 'Modified': 2}
    seg_rows.sort(key=lambda r: _SEG_ORDER.get(r['Status'], 5))
    segments_df = pd.DataFrame(seg_rows,
                               columns=['Segment_ID', 'From_Name', 'To_Name',
                                        'Status', 'Attribute_Changes', 'Composition_Changes'])

    # --- Node diff ---
    nodes_a_idx = (net_a.nodes.drop_duplicates('Name').set_index('Name')
                   if not net_a.nodes.empty else pd.DataFrame())
    nodes_b_idx = (net_b.nodes.drop_duplicates('Name').set_index('Name')
                   if not net_b.nodes.empty else pd.DataFrame())

    names_a = set(nodes_a_idx.index.dropna()) if not nodes_a_idx.empty else set()
    names_b = set(nodes_b_idx.index.dropna()) if not nodes_b_idx.empty else set()

    node_rows: list = []

    for name in sorted(names_a - names_b):
        r = nodes_a_idx.loc[name]
        node_rows.append({
            'Name': name,
            'Code': _norm(r.get('Code')),
            'Node_Class': _norm(r.get('Node_Class')),
            'Status': 'Removed',
            'Changes': '',
        })

    for name in sorted(names_b - names_a):
        r = nodes_b_idx.loc[name]
        node_rows.append({
            'Name': name,
            'Code': _norm(r.get('Code')),
            'Node_Class': _norm(r.get('Node_Class')),
            'Status': 'Added',
            'Changes': '',
        })

    for name in sorted(names_a & names_b):
        ra, rb = nodes_a_idx.loc[name], nodes_b_idx.loc[name]
        tc_a = _tc_int(ra.get('Track_Count', 0))
        tc_b = _tc_int(rb.get('Track_Count', 0))
        changed = []
        if tc_b != tc_a:
            changed.append(f'Track_Count: {tc_a}→{tc_b}')
        changed += [a for a in _NODE_MOD_ATTRS
                    if _norm(ra.get(a)) != _norm(rb.get(a))]
        if not ra['geometry'].equals(rb['geometry']):
            changed.append('geometry')
        if changed:
            node_rows.append({
                'Name': name,
                'Code': _norm(rb.get('Code')),
                'Node_Class': _norm(rb.get('Node_Class')),
                'Status': 'Modified',
                'Changes': ', '.join(changed),
            })

    _NODE_ORDER = {'Added': 0, 'Removed': 1, 'Modified': 2}
    node_rows.sort(key=lambda r: _NODE_ORDER.get(r['Status'], 3))
    nodes_df = pd.DataFrame(node_rows,
                            columns=['Name', 'Code', 'Node_Class', 'Status', 'Changes'])

    if output_path is not None:
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            raise ImportError(
                "openpyxl is required for diff export: pip install openpyxl"
            ) from None
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            nodes_df.to_excel(writer, sheet_name='Nodes', index=False)
            segments_df.to_excel(writer, sheet_name='Segments', index=False)
        print(f"  Diff report → {output_path}")

    return nodes_df, segments_df


def plot_infrastructure_diff(
    net_a: NetworkData,
    net_b: NetworkData,
    extent=None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show_labels: bool = True,
    is_catchment: bool = False,
    show_outside: bool = False,
) -> plt.Figure:
    """Diff plot: net_a is the reference, net_b is the comparison.

    Segments
    --------
    - Removed (in A, not in B)          : red tracks
    - Added   (in B, not in A)          : green tracks
    - Track gained (n_b > n_a)          : n_b tracks drawn; outermost = green
    - Track lost   (n_b < n_a)          : max(n_a,n_b)+1 tracks drawn; outermost = red
    - Unchanged                         : black tracks, low opacity

    Nodes
    -----
    - Removed : red markers
    - Added   : green markers
    - Unchanged: grey markers, low opacity

    When show_outside=True the diff is computed on all (extent-clipped) data;
    items outside net_a's boundary are rendered at ghost alpha (0.25) and
    in-boundary items are redrawn at full alpha on top, matching the
    ghost-pass pattern used by all other plot functions.
    """
    if title is None:
        title = f"Infrastructure diff — {net_b.version} vs {net_a.version}"

    fig, ax = _base_plot(net_b, title, figsize, extent=extent)
    if is_catchment:
        _plot_lakes(ax, boundary=net_b.boundary)
    else:
        _plot_lakes(ax, extent=extent)

    ts       = _TRACK_SPACING_M_CA if is_catchment else _TRACK_SPACING_M_SA
    boundary = net_a.boundary  # authoritative for show_outside

    # ── Data preparation ──────────────────────────────────────────────────────
    # CA diff: always clip to boundary — no ghost pass outside the CA.
    # SA diff with show_outside=True: ghost pass first, then clip to boundary.
    # SA diff with show_outside=False: clip to boundary directly.
    if is_catchment:
        segs_a  = _clip_to_boundary(net_a.segments, boundary)
        segs_b  = _clip_to_boundary(net_b.segments, boundary)
        nodes_a = _clip_to_boundary(net_a.nodes, boundary)
        nodes_b = _clip_to_boundary(net_b.nodes, boundary)
        show_outside = False  # suppress ghost pass for CA
    elif extent is not None:
        segs_a  = _clip_to_extent(net_a.segments, extent)
        segs_b  = _clip_to_extent(net_b.segments, extent)
        nodes_a = _clip_to_extent(net_a.nodes, extent)
        nodes_b = _clip_to_extent(net_b.nodes, extent)
    elif not show_outside:
        segs_a  = _clip_to_boundary(net_a.segments, boundary)
        segs_b  = _clip_to_boundary(net_b.segments, boundary)
        nodes_a = _clip_to_boundary(net_a.nodes, boundary)
        nodes_b = _clip_to_boundary(net_b.nodes, boundary)
    else:
        segs_a  = net_a.segments.copy()
        segs_b  = net_b.segments.copy()
        nodes_a = net_a.nodes.copy()
        nodes_b = net_b.nodes.copy()

    # ── Segment diff ──────────────────────────────────────────────────────────
    ids_a = set(segs_a['Segment_ID'].dropna())
    ids_b = set(segs_b['Segment_ID'].dropna())

    removed_segs = segs_a[segs_a['Segment_ID'].isin(ids_a - ids_b)]
    added_segs   = segs_b[segs_b['Segment_ID'].isin(ids_b - ids_a)]
    common_ids   = ids_a & ids_b

    common_a = segs_a[segs_a['Segment_ID'].isin(common_ids)].set_index('Segment_ID')
    common_b = segs_b[segs_b['Segment_ID'].isin(common_ids)].set_index('Segment_ID')

    _SEG_MOD_ATTRS = (
        'Gauge', 'Electrification_Class', 'Average_Speed',
        'Km_Start', 'Km_End', 'Route_Number', 'Route_Name', 'Route_Owner',
    )

    def _norm(v):
        try:
            return '' if pd.isna(v) else str(v)
        except (TypeError, ValueError):
            return str(v)

    def _tc_int(v) -> int:
        x = pd.to_numeric(v, errors='coerce')
        return 0 if pd.isna(x) else int(x)

    unchanged_ids = []
    modified_ids  = []
    track_gained  = []  # list of (geom, n_a, n_b)
    track_lost    = []  # list of (geom, n_a, n_b)

    for sid in common_ids:
        if sid not in common_a.index or sid not in common_b.index:
            continue
        n_a  = int(common_a.loc[sid].get('Num_Tracks', 1) or 1)
        n_b  = int(common_b.loc[sid].get('Num_Tracks', 1) or 1)
        geom = common_b.loc[sid].geometry
        if n_b > n_a:
            track_gained.append((geom, n_a, n_b))
        elif n_b < n_a:
            track_lost.append((geom, n_a, n_b))
        else:
            geom_changed = not common_a.loc[sid].geometry.equals(geom)
            attr_changed = any(
                _norm(common_a.loc[sid].get(a)) != _norm(common_b.loc[sid].get(a))
                for a in _SEG_MOD_ATTRS
            )
            if geom_changed or attr_changed:
                modified_ids.append(sid)
            else:
                unchanged_ids.append(sid)

    unchanged_segs = segs_b[segs_b['Segment_ID'].isin(unchanged_ids)]
    modified_segs  = segs_b[segs_b['Segment_ID'].isin(modified_ids)]

    # ── Node diff ─────────────────────────────────────────────────────────────
    names_a = set(nodes_a['Name'].dropna()) if not nodes_a.empty else set()
    names_b = set(nodes_b['Name'].dropna()) if not nodes_b.empty else set()

    removed_nodes = nodes_a[nodes_a['Name'].isin(names_a - names_b)]
    added_nodes   = nodes_b[nodes_b['Name'].isin(names_b - names_a)]

    _NODE_MOD_ATTRS = ('Node_Class', 'Platform_Count', 'Transport_Mode')
    modified_node_names  = []
    unchanged_node_names = []
    if not nodes_a.empty and not nodes_b.empty:
        for _name in names_a & names_b:
            _ra = nodes_a[nodes_a['Name'] == _name]
            _rb = nodes_b[nodes_b['Name'] == _name]
            if _ra.empty or _rb.empty:
                unchanged_node_names.append(_name)
                continue
            _ra, _rb = _ra.iloc[0], _rb.iloc[0]
            _geom_changed = not _ra.geometry.equals(_rb.geometry)
            _tc_changed   = (_tc_int(_ra.get('Track_Count', 0))
                             != _tc_int(_rb.get('Track_Count', 0)))
            _attr_changed = _tc_changed or any(
                _norm(_ra.get(a)) != _norm(_rb.get(a)) for a in _NODE_MOD_ATTRS
            )
            if _geom_changed or _attr_changed:
                modified_node_names.append(_name)
            else:
                unchanged_node_names.append(_name)
    else:
        unchanged_node_names = list(names_a & names_b)

    modified_nodes  = nodes_b[nodes_b['Name'].isin(modified_node_names)]
    unchanged_nodes = nodes_b[nodes_b['Name'].isin(unchanged_node_names)]

    ms_ts = 20 if is_catchment else 55
    ms_jn = 5  if is_catchment else 8

    def _plot_node_set(nodes_gdf, facecolor, edgecolor, alpha, zorder):
        if nodes_gdf.empty:
            return
        ts_n, tf_n, jn_n = _classify_nodes(nodes_gdf)
        if not ts_n.empty:
            ts_n.plot(ax=ax, facecolor=facecolor, edgecolor=edgecolor,
                      markersize=ms_ts, marker='o', linewidth=1.2,
                      alpha=alpha, zorder=zorder)
        if not tf_n.empty:
            tf_n.plot(ax=ax, facecolor=facecolor, edgecolor=edgecolor,
                      markersize=ms_ts // 2, marker='o', linewidth=1.0,
                      alpha=alpha, zorder=zorder)
        if not jn_n.empty:
            jn_n.plot(ax=ax, color=facecolor, markersize=ms_jn, marker='o',
                      alpha=alpha, zorder=zorder - 1)

    # ── Ghost pass (show_outside=True only) ───────────────────────────────────
    # All diff categories drawn at ghost alpha across the full extent, then
    # boundary-clipped subsets are redrawn at full alpha (solid pass below).
    if show_outside:
        _ga = 0.25
        # z-order hierarchy (bottom → top): Unchanged, Modified, Removed, Added.
        # track_lost stacks with Removed; track_gained stacks with Added.
        for _, row in unchanged_segs.iterrows():
            _draw_parallel_tracks(ax, row.geometry, int(row.get('Num_Tracks', 1) or 1),
                                  color='black', linewidth=1.05, alpha=_ga, zorder=2,
                                  track_spacing_m=ts)
        for _, row in modified_segs.iterrows():
            _draw_parallel_tracks(ax, row.geometry, int(row.get('Num_Tracks', 1) or 1),
                                  color=_DIFF_YELLOW, linewidth=1.4, alpha=_ga, zorder=3,
                                  track_spacing_m=ts)
        for _, row in removed_segs.iterrows():
            _draw_parallel_tracks(ax, row.geometry, int(row.get('Num_Tracks', 1) or 1),
                                  color=_DIFF_RED, linewidth=1.4, alpha=_ga, zorder=4,
                                  track_spacing_m=ts)
        for _, row in added_segs.iterrows():
            _draw_parallel_tracks(ax, row.geometry, int(row.get('Num_Tracks', 1) or 1),
                                  color=_DIFF_GREEN, linewidth=1.4, alpha=_ga, zorder=5,
                                  track_spacing_m=ts)
        for geom, n_a, n_b in track_lost:
            _draw_parallel_tracks_mixed(ax, geom, ['black'] * n_b + [_DIFF_RED] * (n_a - n_b),
                                        linewidth=1.2, alpha=_ga, zorder=4, track_spacing_m=ts)
        for geom, n_a, n_b in track_gained:
            _draw_parallel_tracks_mixed(ax, geom, ['black'] * n_a + [_DIFF_GREEN] * (n_b - n_a),
                                        linewidth=1.2, alpha=_ga, zorder=5, track_spacing_m=ts)
        _plot_node_set(unchanged_nodes, _DIFF_BLACK,   _DIFF_BLACK,   _ga, 5)
        _plot_node_set(modified_nodes,  _DIFF_YELLOW, '#7a6000',    _ga, 6)
        _plot_node_set(removed_nodes,   _DIFF_RED,    '#7f0000',    _ga, 7)
        _plot_node_set(added_nodes,     _DIFF_GREEN,  '#005a00',    _ga, 8)

        # Clip all categories to boundary for the solid pass
        unchanged_segs  = _clip_to_boundary(unchanged_segs,  boundary)
        modified_segs   = _clip_to_boundary(modified_segs,   boundary)
        removed_segs    = _clip_to_boundary(removed_segs,    boundary)
        added_segs      = _clip_to_boundary(added_segs,      boundary)
        unchanged_nodes = _clip_to_boundary(unchanged_nodes, boundary)
        modified_nodes  = _clip_to_boundary(modified_nodes,  boundary)
        removed_nodes   = _clip_to_boundary(removed_nodes,   boundary)
        added_nodes     = _clip_to_boundary(added_nodes,     boundary)
        if boundary is not None and not boundary.empty:
            _bgeom      = boundary.geometry.union_all()
            track_gained = [(g, na, nb) for g, na, nb in track_gained
                            if g is not None and not g.is_empty
                            and g.centroid.within(_bgeom)]
            track_lost   = [(g, na, nb) for g, na, nb in track_lost
                            if g is not None and not g.is_empty
                            and g.centroid.within(_bgeom)]

    # ── Draw segments (solid pass) ────────────────────────────────────────────
    # z-order hierarchy (bottom → top): Unchanged, Modified, Removed, Added.
    # _draw_seg_category merges all geometries per Num_Tracks group into a
    # single MultiLineString before drawing — eliminates per-segment alpha
    # compositing that creates darker blobs at junctions. alpha=1.0 throughout.
    _draw_seg_category(ax, unchanged_segs, _DIFF_BLACK,  linewidth=1.05, alpha=1.0, zorder=2, ts=ts)
    _draw_seg_category(ax, modified_segs,  _DIFF_YELLOW, linewidth=1.4,  alpha=1.0, zorder=3, ts=ts)
    _draw_seg_category(ax, removed_segs,   _DIFF_RED,    linewidth=1.4,  alpha=1.0, zorder=4, ts=ts)
    _draw_seg_category(ax, added_segs,     _DIFF_GREEN,  linewidth=1.4,  alpha=1.0, zorder=5, ts=ts)
    for geom, n_a, n_b in track_lost:
        _draw_parallel_tracks_mixed(ax, geom, ['black'] * n_b + [_DIFF_RED] * (n_a - n_b),
                                    linewidth=1.2, alpha=1.0, zorder=4, track_spacing_m=ts)
    for geom, n_a, n_b in track_gained:
        _draw_parallel_tracks_mixed(ax, geom, ['black'] * n_a + [_DIFF_GREEN] * (n_b - n_a),
                                    linewidth=1.2, alpha=1.0, zorder=5, track_spacing_m=ts)

    # ── Draw nodes (solid pass) ───────────────────────────────────────────────
    _plot_node_set(unchanged_nodes, _DIFF_BLACK,   _DIFF_BLACK,   1.0, 5)
    _plot_node_set(modified_nodes,  _DIFF_YELLOW, '#7a6000',    1.0, 6)
    _plot_node_set(removed_nodes,   _DIFF_RED,    '#7f0000',    1.0, 7)
    _plot_node_set(added_nodes,     _DIFF_GREEN,  '#005a00',    1.0, 8)

    # ── Labels (added/removed/modified stations only) ─────────────────────────
    if show_labels and not is_catchment:
        for nodes_gdf, color in ((added_nodes,    _DIFF_GREEN),
                                 (removed_nodes,  _DIFF_RED),
                                 (modified_nodes, _DIFF_YELLOW)):
            ts_n, tf_n, _ = _classify_nodes(nodes_gdf)
            for _, row in pd.concat([ts_n, tf_n]).iterrows():
                code = row.get('Code', '')
                if pd.notna(code) and str(code).strip():
                    ax.annotate(
                        str(code),
                        xy=(row.geometry.x, row.geometry.y),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=7, fontweight='bold', color=color,
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                                  edgecolor='none', alpha=0.7),
                        zorder=8,
                    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend = [Line2D([0], [0], color='none', label=r'$\bf{Segments}$')]
    legend.append(Line2D([0], [0], color=_DIFF_BLACK,   linewidth=1.0,
                         label='Unchanged'))
    legend.append(Line2D([0], [0], color=_DIFF_YELLOW, linewidth=1.4,
                         label='Modified'))
    legend.append(Line2D([0], [0], color=_DIFF_GREEN,  linewidth=1.4,
                         label='Added'))
    legend.append(Line2D([0], [0], color=_DIFF_RED,    linewidth=1.4,
                         label='Removed'))
    legend.append(Line2D([0], [0], color='none', label=r'$\bf{Nodes}$'))
    legend.append(Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=_DIFF_BLACK, markersize=6,
                         label='Unchanged'))
    legend.append(Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=_DIFF_YELLOW,
                         markeredgecolor='#7a6000', markersize=8,
                         label='Modified'))
    legend.append(Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=_DIFF_GREEN,
                         markeredgecolor='#005a00', markersize=8,
                         label='Added'))
    legend.append(Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=_DIFF_RED,
                         markeredgecolor='#7f0000', markersize=8,
                         label='Removed'))

    ax.legend(handles=legend, loc='upper right', fontsize=8)
    _add_north_arrow(ax)
    _add_scale_bar(ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"  Saved → {output_path}")
    return fig


# =============================================================================
# Engineering Structures Plot
# =============================================================================

_CONSTRUCT_TRACK_COLOR  = '#FF6600'   # orange — OpenRailwayMap track colour (rail)
_TRAM_TRACK_COLOR       = '#FF69B4'   # pink — tram tracks
_FUNICULAR_TRACK_COLOR  = '#2D6A2D'   # dark green — funicular / cog-railway tracks
_CONSTRUCT_BRIDGE_RAIL  = '#000000'   # black border lines for bridges
_CONSTRUCT_PORTAL_COLOR = '#000000'   # black portal bars for tunnels
_BRIDGE_OFFSET_M = 30                 # offset of bridge border lines from centre (m)
_PORTAL_BAR_M  = 4 * _BRIDGE_OFFSET_M # = 120 m — portal spans 4× bridge half-width


class _BridgeLegendHandler(HandlerBase):
    """Draws a border | track | border cross-section in the legend key for bridges."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        cy     = ydescent + height / 2
        x0, x1 = xdescent, xdescent + width
        offset = height * 0.38
        track   = plt.Line2D([x0, x1], [cy,          cy         ],
                             color=_CONSTRUCT_TRACK_COLOR, linewidth=2.0, transform=trans)
        border1 = plt.Line2D([x0, x1], [cy + offset, cy + offset],
                             color=_CONSTRUCT_BRIDGE_RAIL, linewidth=1.5, transform=trans)
        border2 = plt.Line2D([x0, x1], [cy - offset, cy - offset],
                             color=_CONSTRUCT_BRIDGE_RAIL, linewidth=1.5, transform=trans)
        return [border1, track, border2]


def _draw_geom(ax, geom, color, linewidth, linestyle='solid', zorder=3,
               alpha: float = 1.0) -> None:
    """Draw a LineString or MultiLineString on ax."""
    if geom is None or geom.is_empty:
        return
    parts = list(geom.geoms) if geom.geom_type == 'MultiLineString' else [geom]
    for part in parts:
        if not part.is_empty and len(part.coords) >= 2:
            xs, ys = zip(*part.coords)
            ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle,
                    solid_capstyle='butt', zorder=zorder, alpha=alpha)


def _draw_tunnel_portals(ax, geom, num_tracks: int = 1, track_spacing_m: float = 60, alpha: float = 1.0, zorder: int = 6) -> None:
    """Draw perpendicular portal bars at both ends of a tunnel piece."""
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == 'MultiLineString':
        all_coords = [c for sub in geom.geoms for c in sub.coords]
    else:
        all_coords = list(geom.coords)
    if len(all_coords) < 2:
        return

    for _, (pt, ref) in enumerate(((all_coords[0], all_coords[1]),
                                   (all_coords[-1], all_coords[-2]))):
        dx, dy = ref[0] - pt[0], ref[1] - pt[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            continue
        ux, uy = dx / length, dy / length          # Inward pointing unit vector
        px, py = -uy, ux                           # Perpendicular
        
        width = (num_tracks - 1) * track_spacing_m + 2 * _BRIDGE_OFFSET_M
        h = width / 2.0
        depth = width / 3.0
        
        p1 = (pt[0] - px * h - ux * depth, pt[1] - py * h - uy * depth)
        p2 = (pt[0] - px * h, pt[1] - py * h)
        p3 = (pt[0] + px * h, pt[1] + py * h)
        p4 = (pt[0] + px * h - ux * depth, pt[1] + py * h - uy * depth)

        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            color=_CONSTRUCT_PORTAL_COLOR, linewidth=2.5,
            solid_capstyle='butt', zorder=zorder, alpha=alpha,
        )
        ax.plot(
            [p3[0], p4[0]], [p3[1], p4[1]],
            color=_CONSTRUCT_PORTAL_COLOR, linewidth=2.5,
            solid_capstyle='butt', zorder=zorder, alpha=alpha,
        )


def plot_engineering_structures(
    network: NetworkData,
    composition: gpd.GeoDataFrame,
    extent=None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show_outside: bool = False,
    nodes: Optional[gpd.GeoDataFrame] = None,
) -> plt.Figure:
    """Study-area Engineering Structures map (OpenRailwayMap-inspired).

    Normal  — solid track (orange=rail, pink=tram, dark-green=funicular)
    Bridge  — solid track + parallel black border lines on each side
    Tunnel  — dashed track + perpendicular portal bar at each end
    Gallery — same treatment as tunnel

    Args:
        nodes: Optional node GeoDataFrame used to infer Transport_Mode per
               segment (rail / tram / funicular) for track colour differentiation.
    """
    composition_all = gpd.GeoDataFrame()
    composition_in = None
    if not composition.empty:
        if 'Num_Tracks' not in composition.columns and 'ID' in composition.columns and not network.segments.empty:
            composition = composition.merge(network.segments[['ID', 'Num_Tracks']], on='ID', how='left')

        composition_all = composition.copy()
        composition_in = (_clip_to_boundary(composition, network.boundary)
                          if show_outside and network.boundary is not None and not network.boundary.empty
                          else None)
        composition = _clip_to_boundary(composition, network.boundary)

    if title is None:
        title = f"Engineering Structures — {network.version}"
    fig, ax = _base_plot(network, title, figsize, extent=extent)

    if show_outside:
        _plot_lakes(ax, extent=extent)
    else:
        _plot_lakes(ax, boundary=network.boundary)

    # ── Per-segment track colour lookup (rail / tram / funicular) ─────────────
    _nodes_src = nodes if nodes is not None else network.nodes
    seg_id_to_track_color: dict = {}
    if not _nodes_src.empty and 'Transport_Mode' in _nodes_src.columns and not composition_all.empty:
        _name_to_mode = (
            _nodes_src.drop_duplicates('Name').set_index('Name')['Transport_Mode'].to_dict()
            if 'Name' in _nodes_src.columns else {}
        )
        _comp_src = composition_all if not composition_all.empty else composition
        for _sid in _comp_src['Segment_ID'].unique():
            _pcs = _comp_src[_comp_src['Segment_ID'] == _sid]
            if _pcs.empty:
                continue
            _fn = str(_pcs.iloc[0].get('From_Name') or '')
            _tn = str(_pcs.iloc[0].get('To_Name')   or '')
            _mode_str = ' '.join(
                str(m).lower()
                for nm in (_fn, _tn)
                for m in [_name_to_mode.get(nm)]
                if m is not None and pd.notna(m)
            )
            if 'funicular' in _mode_str or 'cog_railway' in _mode_str:
                seg_id_to_track_color[_sid] = _FUNICULAR_TRACK_COLOR
            elif 'tram' in _mode_str:
                seg_id_to_track_color[_sid] = _TRAM_TRACK_COLOR

    legend_handles: list = []
    legend_seen:    set  = set()
    _bridge_proxy = None  # set on first bridge encounter; used for HandlerTuple legend

    def _add_legend(label, color, lw, ls='solid'):
        if label not in legend_seen:
            legend_seen.add(label)
            legend_handles.append(
                Line2D([0], [0], color=color, linewidth=lw,
                       linestyle=ls, label=label)
            )

    def _plot_comp_pass(comp_df, alpha, z_base):
        nonlocal _bridge_proxy
        if comp_df.empty or 'Engineering_Structure' not in comp_df.columns:
            return
        for _, piece in comp_df.iterrows():
            geom  = piece.geometry
            if geom is None or geom.is_empty:
                continue
            ctype = str(piece.get('Engineering_Structure', 'normal')).lower()
            try:
                nt = piece.get('Num_Tracks', 1)
                num_tracks = int(float(nt)) if pd.notna(nt) else 1
            except (ValueError, TypeError):
                num_tracks = 1
            num_tracks = max(1, min(num_tracks, 4))

            track_dist  = _TRACK_SPACING_M_SA
            track_color = seg_id_to_track_color.get(
                str(piece.get('Segment_ID', '')), _CONSTRUCT_TRACK_COLOR
            )

            if ctype == 'normal':
                _draw_parallel_tracks(ax, geom, num_tracks, color=track_color,
                                      linewidth=1.1, zorder=z_base, alpha=alpha,
                                      track_spacing_m=track_dist)
                if alpha == 1.0:
                    if track_color == _TRAM_TRACK_COLOR:
                        _add_legend('Tram track', _TRAM_TRACK_COLOR, 2)
                    elif track_color == _FUNICULAR_TRACK_COLOR:
                        _add_legend('Funicular track', _FUNICULAR_TRACK_COLOR, 2)
                    else:
                        _add_legend('Rail track', _CONSTRUCT_TRACK_COLOR, 2)

            elif ctype == 'bridge':
                _draw_parallel_tracks(ax, geom, num_tracks, color=track_color,
                                      linewidth=1.1, zorder=z_base+1, alpha=alpha,
                                      track_spacing_m=track_dist)
                outer_offset = ((num_tracks - 1) / 2.0) * track_dist + _BRIDGE_OFFSET_M
                for side in ('left', 'right'):
                    try:
                        sub_lines = list(geom.geoms) if geom.geom_type == 'MultiLineString' else [geom]
                        for sub in sub_lines:
                            border = sub.parallel_offset(outer_offset, side, resolution=8,
                                                         join_style=2, mitre_limit=5)
                            if border and not border.is_empty:
                                _draw_geom(ax, border, _CONSTRUCT_BRIDGE_RAIL, 1.0,
                                           zorder=z_base+2, alpha=alpha)
                    except Exception:
                        pass
                if alpha == 1.0 and 'Bridge' not in legend_seen:
                    legend_seen.add('Bridge')
                    _bridge_proxy = Line2D([0], [0], color='none', label='Bridge')
                    legend_handles.append(_bridge_proxy)

            elif ctype in ('tunnel', 'gallery'):
                _draw_parallel_tracks(ax, geom, num_tracks, color=track_color,
                                      linewidth=1.1, linestyle='dashed',
                                      zorder=z_base, alpha=alpha,
                                      track_spacing_m=track_dist)
                _draw_tunnel_portals(ax, geom, num_tracks=num_tracks,
                                     track_spacing_m=track_dist, alpha=alpha,
                                     zorder=z_base+3)
                if alpha == 1.0:
                    lbl = 'Tunnel' if ctype == 'tunnel' else 'Gallery'
                    _add_legend(lbl, _CONSTRUCT_TRACK_COLOR, 2, ls='dashed')

            else:
                _draw_parallel_tracks(ax, geom, num_tracks, color='#888888',
                                      linewidth=1.2, zorder=max(1, z_base-1),
                                      alpha=alpha, track_spacing_m=track_dist)

    if show_outside and composition_in is not None:
        _plot_comp_pass(composition_all, alpha=0.40, z_base=2)
        _plot_comp_pass(composition_in, alpha=1.0, z_base=10)
    elif not composition.empty and 'Engineering_Structure' in composition.columns:
        _plot_comp_pass(composition, alpha=1.0, z_base=3)
    else:
        if len(network.segments) > 0:
            network.segments.plot(ax=ax, color='#888888', linewidth=1.2, zorder=2)
        print("  Warning: no composition data — falling back to plain segment plot.")

    # Stations overlay
    _ms_ts, _ms_tf, _ms_jn = 60, 20, 10
    
    ts_all, tf_all, jn_all = _classify_nodes(network.nodes)
    if show_outside and network.boundary is not None and not network.boundary.empty:
        ts_draw = _clip_to_boundary(ts_all, network.boundary)
        tf_draw = _clip_to_boundary(tf_all, network.boundary)
        jn_draw = _clip_to_boundary(jn_all, network.boundary)
        
        # Ghost pass — full network at 40%
        if len(ts_all) > 0:
            ts_all.plot(ax=ax, facecolor='white', edgecolor='black',
                        markersize=_ms_ts, marker='o', linewidth=1.2,
                        alpha=0.40, zorder=14)
        if len(tf_all) > 0:
            tf_all.plot(ax=ax, facecolor='black', edgecolor='black',
                        markersize=_ms_tf, marker='o', linewidth=0.8,
                        alpha=0.40, zorder=14)
        if len(jn_all) > 0:
            jn_all.plot(ax=ax, color='#888888', markersize=_ms_jn, marker='o',
                        alpha=0.40, zorder=13)
    else:
        nodes_disp = _clip_to_boundary(network.nodes, network.boundary)
        ts_draw, tf_draw, jn_draw = _classify_nodes(nodes_disp)
        
    if len(ts_draw) > 0:
        ts_draw.plot(ax=ax, facecolor='white', edgecolor='black',
                     markersize=_ms_ts, marker='o', linewidth=1.2,
                     alpha=1.0, zorder=16)
        for _, row in ts_draw.iterrows():
            code = row.get('Code', '')
            if pd.notna(code) and str(code).strip():
                ax.annotate(
                    str(code),
                    xy=(row.geometry.x, row.geometry.y),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor='none', alpha=0.7),
                    zorder=17,
                )

    if len(tf_draw) > 0:
        tf_draw.plot(ax=ax, facecolor='black', edgecolor='black',
                     markersize=_ms_tf, marker='o', linewidth=0.8,
                     alpha=1.0, zorder=16)

    if len(jn_draw) > 0:
        jn_draw.plot(ax=ax, color='#888888', markersize=_ms_jn, marker='o',
                     alpha=1.0, zorder=15)

    # Adding legends for nodes
    has_ts = len(ts_all) > 0 if show_outside else len(ts_draw) > 0
    if has_ts:
        legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                             markeredgecolor='black', markersize=8, label='Train station'))
    
    has_tf = len(tf_all) > 0 if show_outside else len(tf_draw) > 0
    if has_tf:
        legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                             markeredgecolor='black', markersize=6, label='Tram / Funicular'))

    has_jn = len(jn_all) > 0 if show_outside else len(jn_draw) > 0
    if has_jn:
        legend_handles.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888',
                   markersize=5, label='Junction / turning loop')
        )
    _handler_map = ({_bridge_proxy: _BridgeLegendHandler()}
                   if _bridge_proxy is not None else {})
    _lgnd = ax.legend(handles=legend_handles, loc='upper right', fontsize=8,
                      title='Engineering Structures',
                      handler_map=_handler_map if _handler_map else None)
    _lgnd.get_title().set_fontweight('bold')
    _add_north_arrow(ax)
    _add_scale_bar(ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"  Saved → {output_path}")
    return fig


# =============================================================================

# =============================================================================
# Owner Map Plot
# =============================================================================

def plot_owner_map(
    network: NetworkData,
    extent=None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show_outside: bool = False,
    is_catchment: bool = False,
) -> plt.Figure:
    import matplotlib.cm as cm
    
    if title is None:
        title = f"Owner Map - {network.version}"
    fig, ax = _base_plot(network, title, figsize, extent=extent)
    
    if show_outside:
        _plot_lakes(ax, extent=extent)
    else:
        _plot_lakes(ax, boundary=network.boundary)

    segs_all = _clip_to_extent(network.segments, extent)
    segs_in  = (_clip_to_boundary(network.segments, network.boundary)
                if (show_outside and network.boundary is not None and not network.boundary.empty)
                else None)
    segs = _clip_to_boundary(network.segments, network.boundary)

    # Owners are drawn from catchment-clipped `segs`; `segs_all` used only for
    # background rendering when show_outside=True.
    catchment_segs = segs if len(segs) > 0 else segs_all
    if len(catchment_segs) > 0 and 'Route_Owner' in catchment_segs.columns:
        # Determine owner set from catchment boundary only
        unique_owners = catchment_segs['Route_Owner'].dropna().unique()

        colors = {}
        palette_idx = 0
        for o in sorted(unique_owners, key=str):
            if 'SBB' in str(o).upper():
                colors[o] = '#E3000F'
            else:
                colors[o] = _OWNER_PALETTE[palette_idx % len(_OWNER_PALETTE)]
                palette_idx += 1

        plotted_owners = set()

        if show_outside and segs_in is not None:
            for owner, color in colors.items():
                mask_all = segs_all['Route_Owner'] == owner
                if mask_all.any():
                    plotted_owners.add(owner)
                    segs_all[mask_all].plot(ax=ax, color=color, linewidth=2, alpha=0.40, zorder=2)

            if len(segs_in) > 0:
                for owner, color in colors.items():
                    mask_in = segs_in['Route_Owner'] == owner
                    if mask_in.any():
                        plotted_owners.add(owner)
                        segs_in[mask_in].plot(ax=ax, color=color, linewidth=2, alpha=0.8, zorder=3)
        else:
            if len(segs) > 0:
                for owner, color in colors.items():
                    mask = segs['Route_Owner'] == owner
                    if mask.any():
                        plotted_owners.add(owner)
                        segs[mask].plot(ax=ax, color=color, linewidth=2, alpha=0.8, zorder=2)

        legend = [
            Line2D([0], [0], color=colors[own], linewidth=2, label=str(own))
            for own in sorted(plotted_owners, key=str)
        ]
        if len(legend) <= 40:
            _lgnd = ax.legend(handles=legend, loc='upper right', fontsize=8,
                              ncol=2, title='Route Owner')
            _lgnd.get_title().set_fontweight('bold')
            
    _add_north_arrow(ax)
    _add_scale_bar(ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"  Saved -> {output_path}")
    return fig

def plot_speed_map(
    network: NetworkData,
    extent=None,
    output_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show_outside: bool = False,
    is_catchment: bool = False,
) -> plt.Figure:
    """Speed map coloured by predominant OSM maxspeed per segment."""
    if title is None:
        title = f"Speed Map — {network.version}"
    fig, ax = _base_plot(network, title, figsize, extent=extent)
    ax.set_facecolor('#f5f5f5')

    if show_outside:
        _plot_lakes(ax, extent=extent)
    else:
        _plot_lakes(ax, boundary=network.boundary)

    segs_all = _clip_to_extent(network.segments, extent)
    segs_in  = (
        _clip_to_boundary(network.segments, network.boundary)
        if (show_outside and network.boundary is not None and not network.boundary.empty)
        else None
    )
    segs = _clip_to_boundary(network.segments, network.boundary)

    if 'Predominant_Speed' not in network.segments.columns:
        ax.text(0.5, 0.5, 'No speed data\n(run network builder with OSM enrichment)',
                transform=ax.transAxes, ha='center', va='center', fontsize=12, color='grey')
        _add_north_arrow(ax)
        _add_scale_bar(ax)
        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, bbox_inches='tight')
            print(f"  Saved → {output_path}  (no speed data)")
        return fig

    bins_present: set = set()

    def _plot_segs(gdf, alpha):
        if gdf is None or gdf.empty:
            return
        gdf = gdf.copy()
        gdf['Predominant_Speed'] = pd.to_numeric(gdf['Predominant_Speed'], errors='coerce')
        no_data = gdf['Predominant_Speed'].isna()
        if no_data.any():
            gdf[no_data].plot(ax=ax, color=SPEED_NO_DATA_COLOR, linewidth=2, alpha=alpha)
            bins_present.add('no_data')
        prev_upper = 0
        for upper, color, _ in SPEED_BINS:
            mask = (
                ~gdf['Predominant_Speed'].isna()
                & (gdf['Predominant_Speed'] > prev_upper)
                & (gdf['Predominant_Speed'] <= upper)
            )
            if mask.any():
                gdf[mask].plot(ax=ax, color=color, linewidth=2.5, alpha=alpha)
                bins_present.add(upper)
            prev_upper = upper

    if show_outside and segs_in is not None:
        _plot_segs(segs_all, alpha=0.35)
        _plot_segs(segs_in,  alpha=1.0)
    else:
        _plot_segs(segs, alpha=0.85)

    legend = []
    if 'no_data' in bins_present:
        legend.append(Line2D([0], [0], color=SPEED_NO_DATA_COLOR, linewidth=3,
                             label=SPEED_NO_DATA_LABEL))
    for upper, color, label in SPEED_BINS:
        if upper in bins_present:
            legend.append(Line2D([0], [0], color=color, linewidth=3, label=label))
    _lgnd = ax.legend(handles=legend, loc='upper right', fontsize=10,
                      title='Predominant Speed', title_fontsize=11)
    _lgnd.get_title().set_fontweight('bold')

    _add_north_arrow(ax)
    _add_scale_bar(ax)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"  Saved → {output_path}")
    return fig


# =============================================================================
# Import core (helpers shared with infrabuild_version_manager.py)
# Pure data-frame primitives used by the version manager TUI, the seed
# augmentation hook above, and the derived-version builder. No I/O policy here.
# =============================================================================

SWISS_CRS = "EPSG:2056"
NODE_SNAP_TOLERANCE_M = 25.0

_MODE_DEFAULT_SPEEDS_VM: dict = {
    'train': 50.0, 'tram': 30.0, 'funicular': 10.0,
    'cog_railway': 15.0, 'bus': 30.0,
}
_MODE_DEFAULT_FALLBACK_VM: float = 50.0


# ─────────────────────────────────────────────────────────────────────────────
# Pure physics helpers
# ─────────────────────────────────────────────────────────────────────────────

def auto_tt(length_m: float, speed_kmh: float, n_sta: int) -> Tuple[float, float]:
    """Return (TT_Stopping, TT_Passing) in minutes via the standard physics formula.

    Uses settings.SERVICE_BRAKE_DECEL_MS2 and settings.TT_OPERATIONAL_BUFFER as
    single source of truth (mirrors infrabuild_network_builder._compute_approx_travel_times).

    Args:
        length_m: segment length in metres.
        speed_kmh: cruise speed in km/h.
        n_sta: number of segment endpoints classified as 'station' (0, 1, or 2).

    Returns:
        (TT_Stopping_min, TT_Passing_min) both rounded to 0.1.
    """
    a = settings.SERVICE_BRAKE_DECEL_MS2
    b = settings.TT_OPERATIONAL_BUFFER
    v = speed_kmh / 3.6
    cruise = length_m / v
    tt_pass = round(max(0.1, (cruise * b) / 60), 1)
    tt_stop = round(max(0.1, ((cruise + n_sta * 0.5 * v / a) * b) / 60), 1)
    return tt_stop, tt_pass


def count_stations_at_endpoints(
    from_name: Optional[str],
    to_name: Optional[str],
    nodes: gpd.GeoDataFrame,
) -> int:
    """Return 0, 1, or 2 — how many of the named endpoints are 'station' nodes."""
    n = 0
    for name in (from_name, to_name):
        if name is None or (isinstance(name, float) and pd.isna(name)):
            continue
        match = nodes[nodes['Name'] == name]
        if not match.empty and str(match.iloc[0].get('Node_Class', '')).strip() == 'station':
            n += 1
    return n


def default_speed_for_segment(segment_row, gauge_fallback: Optional[float] = None) -> float:
    """Pick a mode-default cruise speed (km/h) for a segment based on its gauge.

    Args:
        segment_row: a row-like with a 'Gauge' field (1435, 1000, ≤900).
        gauge_fallback: optional override used when gauge is missing.
    """
    g = segment_row.get('Gauge')
    if g is not None and not (isinstance(g, float) and pd.isna(g)):
        try:
            gi = int(float(g))
            if gi <= 900:
                return _MODE_DEFAULT_SPEEDS_VM.get('funicular', 10.0)
            if gi == 1000:
                return _MODE_DEFAULT_SPEEDS_VM.get('tram', 30.0)
        except (ValueError, TypeError):
            pass
    return gauge_fallback if gauge_fallback is not None else _MODE_DEFAULT_FALLBACK_VM


# ─────────────────────────────────────────────────────────────────────────────
# Geometry / topology
# ─────────────────────────────────────────────────────────────────────────────

def split_segment_at(
    segments: gpd.GeoDataFrame,
    composition: gpd.GeoDataFrame,
    seg_idx: int,
    split_dist: float,
    new_node_name: str,
    new_node_code: str,
    nodes: gpd.GeoDataFrame,
    new_node_class: Optional[str] = None,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Split segments.loc[seg_idx] at split_dist metres along its geometry.

    Produces S_A (before split) and S_B (after split). Redistributes composition
    pieces between A and B by cumulative length. Recomputes TT_Stopping and
    TT_Passing via auto_tt (speed_source='formula').

    Args:
        segments: current segments GeoDataFrame.
        composition: current segments_composition GeoDataFrame.
        seg_idx: index of the segment to split.
        split_dist: split position in metres from the segment start.
        new_node_name: name of the new node at the split point.
        new_node_code: short code of the new node.
        nodes: current nodes GeoDataFrame (for Code lookup and station counts).
        new_node_class: 'station' (counts as station endpoint) or anything else.

    Returns:
        Updated (segments, composition).
    """
    S = segments.loc[seg_idx]

    geom = S.geometry
    if geom.geom_type == 'MultiLineString':
        geom = linemerge(geom)

    seg_len = geom.length
    split_dist = min(max(split_dist, 0.0), seg_len)

    geom_A = shp_substring(geom, 0, split_dist)
    geom_B = shp_substring(geom, split_dist, seg_len)

    t = split_dist / seg_len if seg_len > 0 else 0.0
    km_start = (float(S['Km_Start'])
                if S['Km_Start'] is not None
                   and not (isinstance(S['Km_Start'], float) and pd.isna(S['Km_Start']))
                else 0.0)
    km_end = (float(S['Km_End'])
              if S['Km_End'] is not None
                 and not (isinstance(S['Km_End'], float) and pd.isna(S['Km_End']))
              else 0.0)
    km_mid = km_start + t * (km_end - km_start)

    split_pt = geom.interpolate(split_dist)

    avg_spd = S.get('Average_Speed')
    speed_kmh = (
        float(avg_spd)
        if pd.notna(avg_spd) and float(avg_spd) > 0
        else default_speed_for_segment(S)
    )

    n_from = count_stations_at_endpoints(S['From_Name'], None, nodes)
    n_to   = count_stations_at_endpoints(None, S['To_Name'], nodes)
    n_new  = 1 if (new_node_class or '').strip() == 'station' else 0

    tt_stop_A, tt_pass_A = auto_tt(geom_A.length, speed_kmh, n_from + n_new)
    tt_stop_B, tt_pass_B = auto_tt(geom_B.length, speed_kmh, n_new + n_to)

    from_code_rows = nodes[nodes['Name'] == S['From_Name']]
    to_code_rows   = nodes[nodes['Name'] == S['To_Name']]
    from_code = from_code_rows.iloc[0]['Code'] if not from_code_rows.empty else S['From_Name']
    to_code   = to_code_rows.iloc[0]['Code']   if not to_code_rows.empty   else S['To_Name']

    id_A = f"c{from_code}_{new_node_code}"
    id_B = f"c{new_node_code}_{to_code}"

    row_A = _split_row_template(S, geom_A, S['From_Name'], new_node_name,
                                S.get('From_E', pd.NA), S.get('From_N', pd.NA),
                                split_pt.x, split_pt.y,
                                km_start, km_mid, id_A, tt_stop_A, tt_pass_A)
    row_B = _split_row_template(S, geom_B, new_node_name, S['To_Name'],
                                split_pt.x, split_pt.y,
                                S.get('To_E', pd.NA), S.get('To_N', pd.NA),
                                km_mid, km_end, id_B, tt_stop_B, tt_pass_B)

    segments = segments[segments.index != seg_idx].reset_index(drop=True)
    segments = pd.concat(
        [segments, gpd.GeoDataFrame([row_A, row_B], crs=SWISS_CRS)],
        ignore_index=True,
    )

    composition = _redistribute_composition(
        composition, S['Segment_ID'], split_dist,
        id_A, id_B, S['From_Name'], new_node_name, S['To_Name'],
        geom_A, geom_B,
    )
    return segments, composition


def _split_row_template(S, geom, from_name, to_name,
                        from_e, from_n, to_e, to_n,
                        km_start, km_end, seg_id, tt_stop, tt_pass) -> dict:
    return {
        'Segment_ID':            seg_id,
        'From_Name':             from_name,
        'To_Name':               to_name,
        'From_N':                from_n,
        'From_E':                from_e,
        'To_N':                  to_n,
        'To_E':                  to_e,
        'Length':                geom.length,
        'Num_Tracks':            S['Num_Tracks'],
        'Gauge':                 S['Gauge'],
        'Electrification_Class': S['Electrification_Class'],
        'Km_Start':              km_start,
        'Km_End':                km_end,
        'Route_Number':          S['Route_Number'],
        'Route_Name':            S['Route_Name'],
        'Route_Owner':           S['Route_Owner'],
        'Average_Speed':         S.get('Average_Speed', pd.NA),
        'Predominant_Speed':     S.get('Predominant_Speed', pd.NA),
        'Speed_Coverage_Pct':    S.get('Speed_Coverage_Pct', 0.0),
        'TT_Stopping':           tt_stop,
        'TT_Passing':            tt_pass,
        'speed_source':          'formula',
        'geometry':              geom,
    }


def _redistribute_composition(composition, old_seg_id, split_dist,
                              id_A, id_B, from_name, mid_name, to_name,
                              geom_A, geom_B) -> gpd.GeoDataFrame:
    old_comp = composition[composition['Segment_ID'] == old_seg_id].copy()
    composition = composition[composition['Segment_ID'] != old_seg_id].reset_index(drop=True)

    new_rows = []
    cumulative = 0.0
    for _, piece in old_comp.iterrows():
        piece_len   = float(piece['Piece_Length'])
        piece_start = cumulative
        piece_end   = cumulative + piece_len
        base = piece.to_dict()
        base.pop('geometry', None)

        if piece_end <= split_dist:
            new_rows.append({**base, 'Segment_ID': id_A,
                             'From_Name': from_name, 'To_Name': mid_name,
                             '_geom': geom_A})
        elif piece_start >= split_dist:
            new_rows.append({**base, 'Segment_ID': id_B,
                             'From_Name': mid_name, 'To_Name': to_name,
                             '_geom': geom_B})
        else:
            len_in_A = split_dist - piece_start
            len_in_B = piece_end  - split_dist
            new_rows.append({**base, 'Segment_ID': id_A,
                             'From_Name': from_name, 'To_Name': mid_name,
                             'Piece_Length': len_in_A, '_geom': geom_A})
            new_rows.append({**base, 'Segment_ID': id_B,
                             'From_Name': mid_name, 'To_Name': to_name,
                             'Piece_Length': len_in_B, '_geom': geom_B})
        cumulative = piece_end

    if new_rows:
        geoms = [r.pop('_geom') for r in new_rows]
        new_gdf = gpd.GeoDataFrame(new_rows, geometry=geoms, crs=SWISS_CRS)
        composition = pd.concat([composition, new_gdf], ignore_index=True)
    return composition


# ─────────────────────────────────────────────────────────────────────────────
# Headless merge functions (no TUI, all-rows merge for programmatic callers)
# ─────────────────────────────────────────────────────────────────────────────

def merge_nodes(
    target_nodes: gpd.GeoDataFrame,
    new_nodes: gpd.GeoDataFrame,
    target_segs: Optional[gpd.GeoDataFrame] = None,
    target_comp: Optional[gpd.GeoDataFrame] = None,
    snap_tolerance_m: float = NODE_SNAP_TOLERANCE_M,
    snap_and_split: bool = True,
    replace_existing: bool = False,
) -> Tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]:
    """Append every row of `new_nodes` to `target_nodes` and optionally snap each
    new node onto the nearest segment within tolerance (splitting that segment).

    Used by:
      - Topic 3 seed augmentation (`infrabuild_network_builder._apply_seed_if_present`)
      - Topic 2 derived-version builder
      - Topic 1 Tier-1 cap-intervention insertion

    Args:
        target_nodes: existing nodes GeoDataFrame to merge into.
        new_nodes: rows to append (schema must match target).
        target_segs / target_comp: required when snap_and_split=True.
        snap_tolerance_m: max distance (m) from new node to its host segment.
        snap_and_split: if True and segs/comp provided, snap each new node onto
            the nearest segment within tolerance and split that segment at the
            node's projected position. Nodes further than tolerance are still
            appended (graph nodes that are not on any segment, e.g. greenfield).
        replace_existing: if True, any target row whose Number matches a Number
            in new_nodes is removed before appending (upsert by BAV Number).
            Safe for cap-junction nodes whose Numbers start at 9,100,001 — they
            will never collide with existing BAV nodes.

    Returns:
        (merged_nodes, updated_segs_or_None, updated_comp_or_None)
    """
    if new_nodes is None or new_nodes.empty:
        return target_nodes, target_segs, target_comp

    if replace_existing and 'Number' in new_nodes.columns and 'Number' in target_nodes.columns:
        new_nrs = set(new_nodes['Number'].dropna().astype(str).tolist())
        if new_nrs:
            mask = target_nodes['Number'].astype(str).isin(new_nrs)
            if mask.any():
                target_nodes = target_nodes[~mask].reset_index(drop=True)

    merged_nodes = pd.concat(
        [target_nodes, gpd.GeoDataFrame(new_nodes, crs=SWISS_CRS)],
        ignore_index=True,
    )

    if not snap_and_split or target_segs is None or target_comp is None:
        return merged_nodes, target_segs, target_comp

    segs = target_segs
    comp = target_comp

    for _, n_row in new_nodes.iterrows():
        node_pt = n_row.geometry
        if node_pt is None or node_pt.is_empty:
            continue

        dists = segs.geometry.distance(node_pt)
        nearest_idx  = dists.idxmin()
        nearest_dist = dists[nearest_idx]
        if nearest_dist > snap_tolerance_m:
            continue

        seg_row = segs.loc[nearest_idx]
        split_dist = seg_row.geometry.project(node_pt)
        node_name  = str(n_row.get('Name', ''))
        node_code  = str(n_row.get('Code', node_name[:4].upper()))
        node_class = str(n_row.get('Node_Class', ''))

        segs, comp = split_segment_at(
            segs, comp, nearest_idx, split_dist,
            node_name, node_code, merged_nodes, node_class,
        )
        print(f"  [merge_nodes] snapped '{node_name}' onto segment "
              f"'{seg_row.get('Segment_ID', '?')}' (dist={nearest_dist:.1f} m, "
              f"split at {split_dist:.0f} m)")

    return merged_nodes, segs, comp


def merge_segments(
    target_segs: gpd.GeoDataFrame,
    target_comp: gpd.GeoDataFrame,
    new_segs: gpd.GeoDataFrame,
    new_comp: Optional[gpd.GeoDataFrame] = None,
    replace_existing: bool = True,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Append every row of `new_segs` (and corresponding `new_comp`) to the targets.

    If a Segment_ID in `new_segs` already exists in `target_segs` and
    replace_existing=True, the existing row (and its composition) is removed
    before append.

    Used by:
      - Topic 3 seed augmentation
      - Topic 2 derived-version builder
      - Topic 1 Tier-1 cap-intervention adjusted-segment insertion
    """
    if new_segs is None or new_segs.empty:
        return target_segs, target_comp

    if replace_existing and 'Segment_ID' in new_segs.columns:
        new_ids = set(new_segs['Segment_ID'].dropna().astype(str).tolist())
        if new_ids and 'Segment_ID' in target_segs.columns:
            mask = target_segs['Segment_ID'].astype(str).isin(new_ids)
            if mask.any():
                target_segs = target_segs[~mask].reset_index(drop=True)
            if 'Segment_ID' in target_comp.columns:
                cmask = target_comp['Segment_ID'].astype(str).isin(new_ids)
                if cmask.any():
                    target_comp = target_comp[~cmask].reset_index(drop=True)

    target_segs = pd.concat(
        [target_segs, gpd.GeoDataFrame(new_segs, crs=SWISS_CRS)],
        ignore_index=True,
    )

    if new_comp is not None and not new_comp.empty:
        target_comp = pd.concat(
            [target_comp, gpd.GeoDataFrame(new_comp, crs=SWISS_CRS)],
            ignore_index=True,
        )

    return target_segs, target_comp


# ─────────────────────────────────────────────────────────────────────────────
# Validation and auto-fill
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_autofill(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    composition: gpd.GeoDataFrame,
    interactive: bool = True,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, bool]:
    """Auto-fill derivable nulls and report remaining issues before save.

    Steps:
      1. Length from geometry.
      2. From_E/N, To_E/N from nodes table.
      3. Average_Speed from Predominant_Speed.
      4. Speed_Coverage_Pct.
      5. speed_source inference.
      6. TT_Stopping / TT_Passing from formula.
      7. Number / Code from endpoint nodes.
      8. Missing composition rows (default to 'normal' / Edge_Level=1).
      9. Tunnel_Length / Bridge_Length / Conventional_Length from composition.
     10. Orphan composition row cleanup.

    Args:
        interactive: when True, prompts the user before saving if blocking
            issues remain; when False, returns ok_to_save=False on any blocker.

    Returns:
        (nodes, segments, composition, ok_to_save).
    """
    print("\n" + "─" * 60)
    print("  Pre-save validation")
    print("─" * 60)

    counts: dict = {}
    warnings: List[str] = []
    blockers: List[str] = []

    node_lookup = {
        str(r['Name']): r
        for _, r in nodes.iterrows()
        if pd.notna(r.get('Name'))
    }

    # Step 1 — Length from geometry
    bad_len_mask = segments['Length'].isnull() | (segments['Length'] <= 0)
    for i in segments.index[bad_len_mask]:
        g = segments.at[i, 'geometry']
        if g is not None and not g.is_empty:
            segments.at[i, 'Length'] = float(g.length)
            counts['Length'] = counts.get('Length', 0) + 1

    # Step 2 — From_E/N, To_E/N from nodes table
    for i, row in segments.iterrows():
        for prefix, name_col in (('From', 'From_Name'), ('To', 'To_Name')):
            e_col, n_col = f'{prefix}_E', f'{prefix}_N'
            if e_col not in segments.columns or n_col not in segments.columns:
                continue
            if pd.notna(segments.at[i, e_col]) and pd.notna(segments.at[i, n_col]):
                continue
            name = row.get(name_col)
            if name in node_lookup:
                segments.at[i, e_col] = float(node_lookup[name].get('E', 0) or 0)
                segments.at[i, n_col] = float(node_lookup[name].get('N', 0) or 0)
                counts['Endpoint_coords'] = counts.get('Endpoint_coords', 0) + 1

    # Step 3 — Average_Speed from Predominant_Speed
    if 'Predominant_Speed' in segments.columns:
        mask = segments['Average_Speed'].isnull() & segments['Predominant_Speed'].notna()
        n_fill = int(mask.sum())
        if n_fill:
            segments.loc[mask, 'Average_Speed'] = segments.loc[mask, 'Predominant_Speed']
            counts['Average_Speed'] = n_fill

    # Step 4 — Speed_Coverage_Pct
    if 'Speed_Coverage_Pct' in segments.columns:
        mask_null = segments['Speed_Coverage_Pct'].isnull()
        if mask_null.any():
            segments.loc[mask_null & segments['Average_Speed'].notna(), 'Speed_Coverage_Pct'] = 1.0
            segments.loc[mask_null & segments['Average_Speed'].isnull(),  'Speed_Coverage_Pct'] = 0.0
            counts['Speed_Coverage_Pct'] = int(mask_null.sum())

    # Step 5 — speed_source inference
    if 'speed_source' not in segments.columns:
        segments['speed_source'] = pd.NA
    mask_null_src = (
        segments['speed_source'].isnull()
        | (segments['speed_source'].astype(str).str.strip() == '')
    )
    if mask_null_src.any():
        has_speed = segments['Average_Speed'].notna()
        segments.loc[mask_null_src &  has_speed, 'speed_source'] = 'formula'
        segments.loc[mask_null_src & ~has_speed, 'speed_source'] = 'estimate'
        counts['speed_source'] = int(mask_null_src.sum())

    # Step 6 — TT_Stopping / TT_Passing from formula
    for col in ('TT_Stopping', 'TT_Passing'):
        if col not in segments.columns:
            segments[col] = pd.NA
    null_tt_mask = segments['TT_Stopping'].isnull() | segments['TT_Passing'].isnull()
    for i in segments.index[null_tt_mask]:
        row = segments.loc[i]
        length_m = float(row.get('Length') or 0)
        if length_m <= 0:
            continue
        spd = row.get('Average_Speed')
        if pd.notna(spd) and float(spd) > 0:
            speed_kmh = float(spd)
            new_src = 'formula'
        else:
            speed_kmh = default_speed_for_segment(row)
            new_src = 'estimate'
        n_sta = count_stations_at_endpoints(
            row.get('From_Name'), row.get('To_Name'), nodes,
        )
        tt_s, tt_p = auto_tt(length_m, speed_kmh, n_sta)
        segments.at[i, 'TT_Stopping'] = tt_s
        segments.at[i, 'TT_Passing']  = tt_p
        if new_src == 'formula' and str(segments.at[i, 'speed_source']) == 'estimate':
            segments.at[i, 'speed_source'] = 'formula'
        counts['TT_filled'] = counts.get('TT_filled', 0) + 1

    # Step 7 — Number / Code from endpoint nodes
    for i, row in segments.iterrows():
        if 'Number' in segments.columns and pd.isna(row.get('Number')):
            fn = node_lookup.get(str(row.get('From_Name', '') or ''))
            tn = node_lookup.get(str(row.get('To_Name',   '') or ''))
            if (fn is not None and tn is not None
                    and pd.notna(fn.get('Number')) and pd.notna(tn.get('Number'))):
                segments.at[i, 'Number'] = f"{int(fn['Number'])}_{int(tn['Number'])}"
                counts['Number'] = counts.get('Number', 0) + 1
        if 'Code' in segments.columns and pd.isna(row.get('Code')):
            fn = node_lookup.get(str(row.get('From_Name', '') or ''))
            tn = node_lookup.get(str(row.get('To_Name',   '') or ''))
            if (fn is not None and tn is not None
                    and pd.notna(fn.get('Code')) and pd.notna(tn.get('Code'))):
                segments.at[i, 'Code'] = f"{fn['Code']}_{tn['Code']}"
                counts['Code'] = counts.get('Code', 0) + 1

    # Step 8 — Missing composition entries
    seg_ids_with_comp = (
        set(composition['Segment_ID'].dropna().astype(str).unique())
        if not composition.empty else set()
    )
    missing_comp_rows = []
    for _, row in segments.iterrows():
        sid = str(row.get('Segment_ID', '') or '')
        if not sid or sid in seg_ids_with_comp:
            continue
        missing_comp_rows.append({
            'Segment_ID':            sid,
            'From_Name':             row.get('From_Name'),
            'To_Name':               row.get('To_Name'),
            'Engineering_Structure': 'normal',
            'Edge_Level':            1,
            'Under_Construction':    0,
            'Piece_Length':          float(row.get('Length') or 0),
            'geometry':              row.get('geometry'),
        })
    if missing_comp_rows:
        composition = pd.concat(
            [composition, gpd.GeoDataFrame(missing_comp_rows, crs=SWISS_CRS)],
            ignore_index=True,
        )
        counts['Composition_added'] = len(missing_comp_rows)

    # Step 9 — Tunnel/Bridge/Conventional length from composition
    derived_cols = [
        ('Tunnel_Length',       'tunnel'),
        ('Bridge_Length',       'bridge'),
        ('Conventional_Length', 'normal'),
    ]
    if all(c in segments.columns for c, _ in derived_cols):
        for i, row in segments.iterrows():
            if not any(pd.isna(segments.at[i, col]) for col, _ in derived_cols):
                continue
            sid = row.get('Segment_ID')
            pieces = composition[composition['Segment_ID'] == sid]
            if pieces.empty:
                continue
            by_type = pieces.groupby('Engineering_Structure')['Piece_Length'].sum()
            for col, struct in derived_cols:
                if pd.isna(segments.at[i, col]):
                    segments.at[i, col] = float(by_type.get(struct, 0.0))
                    counts['Length_breakdown'] = counts.get('Length_breakdown', 0) + 1

    # Step 10 — Orphan composition rows
    seg_ids_existing = set(segments['Segment_ID'].dropna().astype(str).unique())
    orphan_mask = ~composition['Segment_ID'].astype(str).isin(seg_ids_existing)
    n_orphan = int(orphan_mask.sum())
    if n_orphan > 0:
        composition = composition[~orphan_mask].reset_index(drop=True)
        counts['Composition_orphans_removed'] = n_orphan

    # Warnings
    null_class = nodes[
        nodes['Node_Class'].isnull()
        | (nodes['Node_Class'].astype(str).str.strip() == '')
    ]
    if not null_class.empty:
        names = null_class['Name'].dropna().astype(str).tolist()
        warnings.append(
            f"{len(null_class)} node(s) have null Node_Class: "
            f"{', '.join(names[:5])}{'…' if len(names) > 5 else ''}"
        )

    null_num = nodes[nodes['Number'].isnull()]
    if not null_num.empty:
        names = null_num['Name'].dropna().astype(str).tolist()
        warnings.append(
            f"{len(null_num)} node(s) have null Number: "
            f"{', '.join(names[:5])}{'…' if len(names) > 5 else ''}"
        )

    # Blockers
    null_endpoint = segments[segments['From_Name'].isnull() | segments['To_Name'].isnull()]
    if not null_endpoint.empty:
        ids = null_endpoint['Segment_ID'].dropna().astype(str).tolist()
        blockers.append(
            f"{len(null_endpoint)} segment(s) have null From_Name or To_Name: "
            f"{', '.join(ids[:5])}{'…' if len(ids) > 5 else ''}"
        )

    bad_length = segments[segments['Length'].isnull() | (segments['Length'] <= 0)]
    if not bad_length.empty:
        ids = bad_length['Segment_ID'].dropna().astype(str).tolist()
        blockers.append(
            f"{len(bad_length)} segment(s) have Length <= 0 after auto-fill: "
            f"{', '.join(ids[:5])}{'…' if len(ids) > 5 else ''}"
        )

    if counts:
        print("\n  Auto-filled:")
        for key, n in counts.items():
            print(f"    {key:32s} {n} row(s)")
    else:
        print("\n  Nothing to auto-fill — all derivable values already populated.")

    if warnings:
        print("\n  Warnings (save will proceed):")
        for w in warnings:
            print(f"    • {w}")

    ok_to_save = True
    if blockers:
        print("\n  BLOCKING ISSUES:")
        for b in blockers:
            print(f"    • {b}")
        if interactive:
            ans = input("\n  Save anyway? (y/n) [n]: ").strip().lower() or 'n'
            if ans != 'y':
                print("  Save aborted. Fix the issues above and try again.")
                ok_to_save = False
            else:
                print("  Saving despite blocking issues — downstream pipeline may fail.")
        else:
            ok_to_save = False

    return nodes, segments, composition, ok_to_save


# ─────────────────────────────────────────────────────────────────────────────
# Seed loading (Topic 3)
# ─────────────────────────────────────────────────────────────────────────────

def load_and_apply_seed(
    version_name: str,
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    composition: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Apply the year-matched seed files if they exist, otherwise return inputs unchanged.

    The seed is keyed by the 4-digit year extracted from version_name (so
    'AS_2026_ZH', 'AS_2026_ZH_enhanced', 'Base_2026' all wire to year 2026).

    Reads three files from data/Infrastructure/Seeds/:
      nodes_<year>.gpkg             — must have a 'Status' column ('Added'|'Removed'|'Modified')
      segments_<year>.gpkg          — must have a 'Status' column ('Added'|'Removed'|'Modified')
      segments_composition_<year>.gpkg — additions only; no Status column

    Processing order (removals before modifications before additions):
      1. Remove nodes by Code
      2. Remove segments by Segment_ID and cascade their composition rows
      3. Modify nodes — overwrite attribute columns (not geometry) for matching Code
      4. Modify segments — overwrite attribute columns (not geometry) for matching Segment_ID
      5. Merge added nodes via merge_nodes (snap_and_split=True)
      6. Merge added segments via merge_segments

    Returns (nodes, segments, composition) with the seed applied. Idempotent
    when seed elements are already present / already absent.
    """
    nodes_path = Path(paths.get_seed_nodes_gpkg(version_name))
    segs_path  = Path(paths.get_seed_segments_gpkg(version_name))
    comp_path  = Path(paths.get_seed_composition_gpkg(version_name))

    if not nodes_path.exists() and not segs_path.exists():
        print(f"  [seed] no seed files for year extracted from '{version_name}' — proceeding as-is")
        return nodes, segments, composition

    print(f"  [seed] applying seed for '{version_name}'")

    # --- load seed layers ------------------------------------------------
    seed_nodes = gpd.read_file(nodes_path, layer='nodes') if nodes_path.exists() else None
    seed_segs  = gpd.read_file(segs_path,  layer='segments') if segs_path.exists() else None
    seed_comp  = gpd.read_file(comp_path,  layer='segments_composition') if comp_path.exists() else None

    # --- 1. remove nodes -------------------------------------------------
    if seed_nodes is not None and not seed_nodes.empty and 'Status' in seed_nodes.columns:
        remove_codes = seed_nodes.loc[seed_nodes['Status'] == 'Removed', 'Code'].dropna().tolist()
        if remove_codes:
            before = len(nodes)
            nodes = nodes[~nodes['Code'].isin(remove_codes)].copy()
            print(f"  [seed]   removed {before - len(nodes)} node(s): {remove_codes}")

    # --- 2. remove segments (and cascade composition) --------------------
    if seed_segs is not None and not seed_segs.empty and 'Status' in seed_segs.columns:
        remove_ids = seed_segs.loc[seed_segs['Status'] == 'Removed', 'Segment_ID'].dropna().tolist()
        if remove_ids:
            before_s = len(segments)
            before_c = len(composition)
            segments    = segments[~segments['Segment_ID'].isin(remove_ids)].copy()
            composition = composition[~composition['Segment_ID'].isin(remove_ids)].copy()
            print(f"  [seed]   removed {before_s - len(segments)} segment(s) "
                  f"and {before_c - len(composition)} composition row(s)")

    # --- 3. modify nodes (update attribute columns, preserve geometry) ------
    if seed_nodes is not None and not seed_nodes.empty and 'Status' in seed_nodes.columns:
        mod_nodes = seed_nodes[seed_nodes['Status'] == 'Modified'].drop(columns='Status', errors='ignore')
        if not mod_nodes.empty:
            attr_cols = [c for c in mod_nodes.columns if c not in ('geometry', 'Code')]
            n_updated = 0
            for _, seed_row in mod_nodes.iterrows():
                mask = nodes['Code'] == seed_row['Code']
                if mask.any():
                    for col in attr_cols:
                        if col in nodes.columns:
                            val = seed_row[col]
                            if pd.api.types.is_numeric_dtype(nodes[col].dtype):
                                val = pd.to_numeric(val, errors='coerce')
                            nodes.loc[mask, col] = val
                    n_updated += 1
            print(f"  [seed]   modified attributes of {n_updated} node(s)")

    # --- 4. modify segments (update attribute columns, preserve geometry) ---
    if seed_segs is not None and not seed_segs.empty and 'Status' in seed_segs.columns:
        mod_segs = seed_segs[seed_segs['Status'] == 'Modified'].drop(columns='Status', errors='ignore')
        if not mod_segs.empty:
            attr_cols = [c for c in mod_segs.columns if c not in ('geometry', 'Segment_ID')]
            n_updated = 0
            for _, seed_row in mod_segs.iterrows():
                mask = segments['Segment_ID'] == seed_row['Segment_ID']
                if mask.any():
                    for col in attr_cols:
                        if col in segments.columns:
                            val = seed_row[col]
                            if pd.api.types.is_numeric_dtype(segments[col].dtype):
                                val = pd.to_numeric(val, errors='coerce')
                            segments.loc[mask, col] = val
                    n_updated += 1
            print(f"  [seed]   modified attributes of {n_updated} segment(s)")

    # --- 5. add nodes ----------------------------------------------------
    if seed_nodes is not None and not seed_nodes.empty and 'Status' in seed_nodes.columns:
        add_nodes = seed_nodes[seed_nodes['Status'] == 'Added'].drop(columns='Status', errors='ignore')
        if not add_nodes.empty:
            print(f"  [seed]   merging {len(add_nodes)} added node(s)")
            nodes, segments, composition = merge_nodes(
                nodes, add_nodes, segments, composition, snap_and_split=True,
            )

    # --- 6. add segments -------------------------------------------------
    if seed_segs is not None and not seed_segs.empty and 'Status' in seed_segs.columns:
        add_segs = seed_segs[seed_segs['Status'] == 'Added'].drop(columns='Status', errors='ignore')
        if not add_segs.empty:
            print(f"  [seed]   merging {len(add_segs)} added segment(s)")
            segments, composition = merge_segments(
                segments, composition, add_segs, seed_comp,
            )

    return nodes, segments, composition

if __name__ == "__main__":
    os.chdir(paths.MAIN)

    print("=" * 60)
    print("infraScanRail — Network Builder")
    print("=" * 60)

    _infra_root = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR
    _base_path  = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_BASE

    _raw_dirs   = list_raw_dirs()
    _base_ready = (_base_path / "nodes.gpkg").exists() and (_base_path / "segments.gpkg").exists()
    _any_base_ready = _base_ready or (
        _infra_root.exists() and any(
            sub.is_dir() and sub.name.startswith('Base')
            and (sub / 'nodes.gpkg').exists() and (sub / 'segments.gpkg').exists()
            for sub in _infra_root.iterdir()
        )
    )

    # ── Prerequisite check ────────────────────────────────────────────────────
    if not _raw_dirs:
        print("\n  No filtered network found in data/Infrastructure/Raw*/.")
        print("  Run infrabuild_filter_network.py first, then return here.")
        raise SystemExit(1)

    # ── Step 0 / Q1: Which network version to build? ─────────────────────────
    print("\n" + "─" * 60)
    print("[Step 0 / Q1]  Which network version to build?")
    print("─" * 60)

    _NEW_BASE = '[New Base]'

    if not _any_base_ready:
        print("\n  Note: No Base network exists yet.")
        print("  Build Base first before creating or analysing named versions.")
        _versions_avail: List[str] = [_NEW_BASE]
    else:
        _versions_avail = list_versions()
        _versions_avail.append(_NEW_BASE)

    print("\n  Available:")
    for _i, _v in enumerate(_versions_avail, 1):
        if _v == _NEW_BASE:
            _tag = "  [build from Raw]"
        elif _v == 'Base':
            _tag = "  [base]"
        else:
            _tag = ""
        print(f"    {_i}) {_v}{_tag}")

    if len(_versions_avail) == 1:
        _chosen = _versions_avail[0]
        print(f"\n  Only one option available — auto-selecting: {_chosen}")
    else:
        while True:
            _sel = input("\n  Select version (number or name) [1]: ").strip() or "1"
            if _sel.isdigit() and 1 <= int(_sel) <= len(_versions_avail):
                _chosen = _versions_avail[int(_sel) - 1]
                break
            elif _sel in _versions_avail:
                _chosen = _sel
                break
            print(f"  Invalid — enter 1–{len(_versions_avail)} or an exact name.")
    print(f"  → {_chosen}")

    # ── Step 0 / Q2: Which plots to generate? ────────────────────────────────
    print("\n" + "─" * 60)
    print("[Step 0 / Q2]  Which plots to generate?")
    print("─" * 60)
    print("\n  Catchment area (extent = catchment_area_boundary):")
    print("    Infrastructure · Gauge · Electrification · Speed · Track Owner")
    print("  Study area (extent = study_area_boundary):")
    print("    Infrastructure · Gauge · Electrification · Speed · Track Owner · Engineering Structures")
    print("\n  Options:")
    print("    1) All")
    print("    2) Catchment area only")
    print("    3) Study area only")
    print("    4) Manual choice")
    print("    5) None")

    while True:
        _scope = input("\n  Select (1–5) [1]: ").strip() or "1"
        if _scope in ('1', '2', '3', '4', '5'):
            break
        print("  Invalid — enter 1–5.")

    _ALL_PLOTS = [
        ('ca_infra',     'Catchment — Infrastructure'),
        ('ca_gauge',     'Catchment — Gauge'),
        ('ca_elec',      'Catchment — Electrification'),
        ('ca_speed',     'Catchment — Speed'),
        ('ca_owner',     'Catchment — Track Owner'),
        ('sa_infra',     'Study area — Infrastructure'),
        ('sa_gauge',     'Study area — Gauge'),
        ('sa_elec',      'Study area — Electrification'),
        ('sa_speed',     'Study area — Speed'),
        ('sa_owner',     'Study area — Track Owner'),
        ('sa_construct', 'Study area — Engineering Structures'),
    ]
    _CA_KEYS = [k for k, _ in _ALL_PLOTS if k.startswith('ca_')]
    _SA_KEYS = [k for k, _ in _ALL_PLOTS if k.startswith('sa_')]

    if _scope == '1':
        _plot_set = [k for k, _ in _ALL_PLOTS]
    elif _scope == '2':
        _plot_set = _CA_KEYS[:]
    elif _scope == '3':
        _plot_set = _SA_KEYS[:]
    elif _scope == '5':
        _plot_set = []
    else:  # manual
        print("\n  Select plots:")
        for _i, (_k, _lbl) in enumerate(_ALL_PLOTS, 1):
            print(f"    {_i}) {_lbl}")
        while True:
            _raw_sel = input("\n  Numbers (comma-sep) or 'all' [all]: ").strip() or "all"
            if _raw_sel.lower() == 'all':
                _plot_set = [k for k, _ in _ALL_PLOTS]
                break
            _parts = [p.strip() for p in _raw_sel.split(',') if p.strip()]
            if _parts and all(p.isdigit() and 1 <= int(p) <= len(_ALL_PLOTS) for p in _parts):
                _plot_set = [_ALL_PLOTS[int(p) - 1][0] for p in _parts]
                break
            if not _raw_sel:
                _plot_set = []
                break
            print(f"  Invalid — enter numbers 1–{len(_ALL_PLOTS)}, 'all', or leave empty.")

    _plot_label_map = dict(_ALL_PLOTS)

    # ── Step 0 / Q3: Diff plot? ───────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("[Step 0 / Q3]  Diff plot")
    print("─" * 60)

    _do_diff = input("\n  Generate a diff plot? (y/n) [n]: ").strip().lower() or "n"
    _ref_version  = None
    _diff_scope   = None

    if _do_diff == "y":
        _diff_candidates = [
            v for v in sorted(
                d.name for d in _infra_root.iterdir()
                if d.is_dir() and not d.name.startswith("Raw")
                and (d / "nodes.gpkg").exists()
                and (d / "segments.gpkg").exists()
            )
            if v != _chosen
        ]
        if not _diff_candidates:
            print("  No other versions available — skipping diff.")
            _do_diff = "n"
        else:
            print("\n  Compare against (reference version):")
            for _i, _v in enumerate(_diff_candidates, 1):
                _marker = "  [Base]" if _v == "Base" else ""
                print(f"    {_i}) {_v}{_marker}")
            while True:
                _ref_raw = input("  Select (number): ").strip()
                if _ref_raw.isdigit() and 1 <= int(_ref_raw) <= len(_diff_candidates):
                    _ref_version = _diff_candidates[int(_ref_raw) - 1]
                    break
                print(f"  Invalid — enter 1–{len(_diff_candidates)}.")

            print("\n  Extent:")
            print("    1) Catchment area")
            print("    2) Study area")
            print("    3) Both")
            while True:
                _diff_scope = input("  Select (1–3) [3]: ").strip() or "3"
                if _diff_scope in ("1", "2", "3"):
                    break
                print("  Enter 1, 2, or 3.")

    # ── Step 1: Build ─────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("[Step 1]  Network building")
    print("─" * 60)

    if _chosen == _NEW_BASE:
        if len(_raw_dirs) == 1:
            _chosen_raw = _raw_dirs[0]
            print(f"\n  Using Raw folder: {_chosen_raw}")
        else:
            print("\n  Available Raw folders:")
            for _ri, _rn in enumerate(_raw_dirs, 1):
                print(f"    {_ri}) {_rn}")
            while True:
                _rsel = input("\n  Select Raw folder [1]: ").strip() or "1"
                if _rsel.isdigit() and 1 <= int(_rsel) <= len(_raw_dirs):
                    _chosen_raw = _raw_dirs[int(_rsel) - 1]
                    break
                elif _rsel in _raw_dirs:
                    _chosen_raw = _rsel
                    break
                print(f"  Invalid — enter 1–{len(_raw_dirs)} or an exact name.")

        _raw_suffix       = _chosen_raw[3:]
        _out_folder       = 'Base' + _raw_suffix
        _chosen_base_path = _infra_root / _out_folder
        _chosen           = _out_folder

        if (_chosen_base_path / 'nodes.gpkg').exists():
            _overwrite = input(
                f"\n  {_out_folder} already exists. Overwrite? (y/n) [n]: "
            ).strip().lower() or "n"
            if _overwrite != 'y':
                print("  Aborted.")
                raise SystemExit(0)

        # ── Seed selection ────────────────────────────────────────────────────
        _seeds_dir  = Path(paths.MAIN) / paths.INFRASTRUCTURE_SEEDS_DIR
        _seed_years = sorted(
            p.name.removeprefix('nodes_').removesuffix('.gpkg')
            for p in _seeds_dir.glob('nodes_*.gpkg')
        ) if _seeds_dir.is_dir() else []

        _chosen_seed_year = None
        if _seed_years:
            print("\n" + "─" * 60)
            print("[Step 1b]  Apply a seed to this build?")
            print("─" * 60)
            print("\n  Available seed years:")
            for _si, _sy in enumerate(_seed_years, 1):
                print(f"    {_si}) {_sy}")
            print(f"    {len(_seed_years) + 1}) None — skip seed")
            if len(_seed_years) == 1:
                _chosen_seed_year = _seed_years[0]
                print(f"\n  Only one seed available — auto-selecting: {_chosen_seed_year}")
            else:
                while True:
                    _ssel = input(
                        f"\n  Select (1–{len(_seed_years) + 1}) [1]: "
                    ).strip() or "1"
                    if _ssel.isdigit():
                        _sidx = int(_ssel)
                        if 1 <= _sidx <= len(_seed_years):
                            _chosen_seed_year = _seed_years[_sidx - 1]
                            break
                        elif _sidx == len(_seed_years) + 1:
                            break
                    print(f"  Invalid — enter 1–{len(_seed_years) + 1}.")
            if _chosen_seed_year:
                print(f"  → Seed year: {_chosen_seed_year}")
            else:
                print("  → No seed applied.")

        _nodes, _segments, _composition = run_build_base(
            raw_dir=str(_infra_root / _chosen_raw),
            output_dir=str(_chosen_base_path),
            seed_year=_chosen_seed_year,
        )
        _version_dir = _chosen_base_path
        _base_path   = _chosen_base_path

    elif _chosen.startswith('Base'):
        _chosen_base_path = _infra_root / _chosen
        _chosen_base_ready = (
            (_chosen_base_path / 'nodes.gpkg').exists() and
            (_chosen_base_path / 'segments.gpkg').exists()
        )

        if _chosen_base_ready:
            _rebuild = input(
                f"\n  {_chosen} already exists. Rebuild from Raw? (y/n) [n]: "
            ).strip().lower() or "n"
            _do_build = (_rebuild == 'y')
        else:
            _do_build = True

        if _do_build:
            if len(_raw_dirs) == 1:
                _chosen_raw = _raw_dirs[0]
                print(f"\n  Using Raw folder: {_chosen_raw}")
            else:
                print("\n  Available Raw folders:")
                for _ri, _rn in enumerate(_raw_dirs, 1):
                    print(f"    {_ri}) {_rn}")
                while True:
                    _rsel = input("\n  Select Raw folder [1]: ").strip() or "1"
                    if _rsel.isdigit() and 1 <= int(_rsel) <= len(_raw_dirs):
                        _chosen_raw = _raw_dirs[int(_rsel) - 1]
                        break
                    elif _rsel in _raw_dirs:
                        _chosen_raw = _rsel
                        break
                    print(f"  Invalid — enter 1–{len(_raw_dirs)} or an exact name.")

            _raw_suffix       = _chosen_raw[3:]          # '' or '_ZH'
            _out_folder       = 'Base' + _raw_suffix
            _chosen_base_path = _infra_root / _out_folder
            _chosen           = _out_folder

            _nodes, _segments, _composition = run_build_base(
                raw_dir=str(_infra_root / _chosen_raw),
                output_dir=str(_chosen_base_path),
            )
        else:
            print(f"\n--- Loading existing {_chosen} ---")
            _nodes, _segments = load_version(_chosen)
            _comp_path = _chosen_base_path / "segments_composition.gpkg"
            _composition = (gpd.read_file(_comp_path)
                            if _comp_path.exists() else gpd.GeoDataFrame())
            print(f"  {len(_nodes)} nodes, {len(_segments)} segments")
            print("\n--- QGIS project ---")
            _qgz = str(_chosen_base_path / f"{_chosen}.qgz")
            _build_infra_qgz(_qgz, _chosen_base_path)
            print(f"  {_chosen}.qgz → {_qgz}")

        _version_dir = _chosen_base_path
        _base_path   = _chosen_base_path  # update for downstream composition lookup

    else:
        _version_dir = _infra_root / _chosen
        if not (_version_dir / "nodes.gpkg").exists():
            print(f"\n  Version '{_chosen}' not found in data/Infrastructure/.")
            print("  Use infrabuild_version_manager.py to create it first.")
            raise SystemExit(1)

        print(f"\n--- Loading version '{_chosen}' ---")
        _nodes, _segments = load_version(_chosen)
        print(f"  {len(_nodes)} nodes, {len(_segments)} segments")

        # Enrich any segments that were added via version manager and have no OSM speed
        _speed_col_missing = 'Average_Speed' not in _segments.columns
        if _speed_col_missing:
            _unmatched_mask = pd.Series([True] * len(_segments), index=_segments.index)
        else:
            _unmatched_mask = _segments['Average_Speed'].isna()
        if _unmatched_mask.any():
            _n_new = int(_unmatched_mask.sum())
            print(f"\n--- OSM speed enrichment for {_n_new} segment(s) missing speed ---")
            _new_segs = _segments[_unmatched_mask].copy()
            _enriched = enrich_segments_with_osm_speed(_new_segs, _nodes)
            for _col in ('Average_Speed', 'Predominant_Speed', 'Speed_Coverage_Pct'):
                if _col in _enriched.columns:
                    _segments.loc[_unmatched_mask, _col] = _enriched[_col].values
            _segments.to_file(_version_dir / 'segments.gpkg', driver='GPKG')
            print(f"  segments.gpkg updated → {_version_dir / 'segments.gpkg'}")

        # Derive and export composition for this version
        _base_comp_path = _base_path / "segments_composition.gpkg"
        if _base_comp_path.exists():
            _base_comp    = gpd.read_file(_base_comp_path)
            _ver_sids     = set(_segments['Segment_ID'].dropna())
            _composition  = _filter_composition_for_version(_base_comp, _ver_sids)
            _composition.to_file(_version_dir / "segments_composition.gpkg", driver="GPKG")
            print(f"  segments_composition.gpkg → {_version_dir / 'segments_composition.gpkg'}"
                  f"  ({len(_composition)} pieces)")
        else:
            _composition = gpd.GeoDataFrame()
            print("  Warning: Base composition not found — skipping composition export.")

        # QGIS project
        print("\n--- QGIS project ---")
        _qgz = _version_dir / f"{_chosen}.qgz"
        _build_infra_qgz(str(_qgz), _version_dir)
        print(f"  {_chosen}.qgz → {_qgz}")

    # Build NetworkX graph
    print("\n--- Building NetworkX graph ---")
    G = build_networkx_graph(_nodes, _segments)

    # Load boundaries and extents — needed by both Step 2 (plots) and Step 3
    # (diff), so resolved unconditionally here rather than inside either block.
    _ca_boundary, _sa_boundary = None, None
    _ca_bdry_path = Path(paths.MAIN) / paths.CATCHMENT_AREA_BOUNDARY_GPKG
    _sa_bdry_path = Path(paths.MAIN) / paths.STUDY_AREA_BOUNDARY_GPKG
    if _ca_bdry_path.exists():
        _ca_boundary = gpd.read_file(_ca_bdry_path)
    if _sa_bdry_path.exists():
        _sa_boundary = gpd.read_file(_sa_bdry_path)

    def _extent_from_gdf(gdf, margin_m: int = 2000):
        if gdf is None:
            return None
        b = gdf.total_bounds          # [minx, miny, maxx, maxy]
        return (b[0] - margin_m, b[2] + margin_m,
                b[1] - margin_m, b[3] + margin_m)

    _ca_ext = _extent_from_gdf(_ca_boundary)
    _sa_ext = _extent_from_gdf(_sa_boundary)

    # ── Step 2: Plots ─────────────────────────────────────────────────────────
    if not _plot_set:
        print("\n  No plots requested.")
    else:
        print("\n" + "─" * 60)
        print("[Step 2]  Generating plots")
        print("─" * 60)

        _plot_dir = Path(paths.MAIN) / paths.INFRASTRUCTURE_PLOTS_DIR / _chosen
        _plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Output: {_plot_dir}")

        _net_ca = NetworkData(nodes=_nodes, segments=_segments, graph=G,
                              version=_chosen, boundary=_ca_boundary)
        _net_sa = NetworkData(nodes=_nodes, segments=_segments, graph=G,
                              version=_chosen, boundary=_sa_boundary)

        _plot_dispatch = {
            'ca_infra':  (plot_infrastructure_canonical, _net_ca, _ca_ext,
                          'ca_infrastructure.pdf'),
            'ca_gauge':  (plot_gauge_map,                _net_ca, _ca_ext,
                          'ca_gauge.pdf'),
            'ca_elec':   (plot_electrification_map,      _net_ca, _ca_ext,
                          'ca_electrification.pdf'),
            'ca_speed':  (plot_speed_map,                _net_ca, _ca_ext,
                          'ca_speed.pdf'),
            'ca_owner':  (plot_owner_map,                _net_ca, _ca_ext,
                          'ca_owner.pdf'),
            'sa_infra':  (plot_infrastructure_canonical, _net_sa, _sa_ext,
                          'sa_infrastructure.pdf'),
            'sa_gauge':  (plot_gauge_map,                _net_sa, _sa_ext,
                          'sa_gauge.pdf'),
            'sa_elec':   (plot_electrification_map,      _net_sa, _sa_ext,
                          'sa_electrification.pdf'),
            'sa_speed':  (plot_speed_map,                _net_sa, _sa_ext,
                          'sa_speed.pdf'),
            'sa_owner':  (plot_owner_map,                _net_sa, _sa_ext,
                          'sa_owner.pdf'),
        }

        for _pk in _plot_set:
            if _pk == 'sa_construct':
                if _composition.empty:
                    print("  Skipping Engineering Structures — no composition data.")
                    continue
                print(f"  {_plot_label_map[_pk]} ...")
                _fig = plot_engineering_structures(
                    _net_sa, _composition, extent=_sa_ext,
                    output_path=_plot_dir / "sa_engineering_structures.pdf",
                    show_outside=True,
                    nodes=_nodes,
                )
                plt.close(_fig)
            else:
                _fn, _net, _ext, _fname = _plot_dispatch[_pk]
                print(f"  {_plot_label_map[_pk]} ...")
                
                kwargs = {}
                if _pk.startswith('sa_'):
                    kwargs['show_outside'] = True
                if _pk.startswith('ca_'):
                    kwargs['is_catchment'] = True
                if _pk == 'ca_infra':
                    kwargs['show_labels'] = False
                
                _fig = _fn(_net, extent=_ext, output_path=_plot_dir / _fname, **kwargs)
                plt.close(_fig)

    # ── Step 3: Diff plot ─────────────────────────────────────────────────────
    if _do_diff == "y" and _ref_version is not None:
        print("\n" + "─" * 60)
        print("[Step 3]  Diff plot")
        print("─" * 60)

        print(f"\n  Loading reference '{_ref_version}'...")
        _ref_nodes, _ref_segs = load_version(_ref_version)
        _ref_G = build_networkx_graph(_ref_nodes, _ref_segs)

        _ref_comp_path = _infra_root / _ref_version / 'segments_composition.gpkg'
        _ref_comp = (gpd.read_file(_ref_comp_path)
                     if _ref_comp_path.exists() else gpd.GeoDataFrame())

        _diff_dir = Path(paths.MAIN) / paths.INFRASTRUCTURE_PLOTS_DIR / _chosen
        _diff_dir.mkdir(parents=True, exist_ok=True)

        # Excel diff report — saved alongside the version's geopackages
        _diff_xlsx = _infra_root / _chosen / f"diff_{_chosen}_vs_{_ref_version}.xlsx"
        print(f"  Diff report ...")
        _ref_net_full  = NetworkData(nodes=_ref_nodes, segments=_ref_segs,
                                     graph=_ref_G, version=_ref_version)
        _comp_net_full = NetworkData(nodes=_nodes, segments=_segments,
                                     graph=G, version=_chosen)
        export_infrastructure_diff(
            net_a=_ref_net_full,
            net_b=_comp_net_full,
            comp_a=_ref_comp,
            comp_b=_composition,
            output_path=_diff_xlsx,
        )

        _diff_pairs = []
        if _diff_scope in ("1", "3"):
            _diff_pairs.append(("ca", _ca_boundary, _ca_ext, True))
        if _diff_scope in ("2", "3"):
            _diff_pairs.append(("sa", _sa_boundary, _sa_ext, False))

        for _scope_key, _bdry, _ext, _is_ca in _diff_pairs:
            _ref_net  = NetworkData(nodes=_ref_nodes, segments=_ref_segs,
                                    graph=_ref_G, version=_ref_version,
                                    boundary=_bdry)
            _comp_net = NetworkData(nodes=_nodes, segments=_segments,
                                    graph=G, version=_chosen,
                                    boundary=_bdry)
            _diff_fname = f"diff_{_chosen}_vs_{_ref_version}_{_scope_key}.pdf"
            print(f"  Diff ({_scope_key.upper()}) ...")
            _fig = plot_infrastructure_diff(
                net_a=_ref_net,
                net_b=_comp_net,
                extent=_ext,
                output_path=_diff_dir / _diff_fname,
                is_catchment=_is_ca,
                show_outside=True,
            )
            plt.close(_fig)
        print(f"  Diff output → {_diff_dir}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Version : {_chosen}")
    print(f"Nodes   : {G.number_of_nodes()}")
    print(f"Edges   : {G.number_of_edges()}")
    if _plot_set:
        print(f"Plots   : {_plot_dir}")
    print("=" * 60)


