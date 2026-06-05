"""
Services Version Manager

Interactive TUI for creating and editing named service network versions derived
from the Unprojected/ base output of services_network_builder.py.  Named versions
live in Versions/<name>/ alongside Unprojected/.

Workflow
--------
  Phase 0: Select or create a version; pick infra version for station catalog.
  Phase 1: Rail segment/line editing loop — remove / adjust / add / import.
  Phase 2: Track-based feeder editing loop (optional).
  Phase 3: Save all four geopackages.

Usage (interactive):
    python services_version_manager.py
Last modified: 2026-05-11
"""

import shutil
import geopandas as gpd
import pandas as pd
import numpy as np
import fiona
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from shapely.geometry import LineString
from shapely.ops import linemerge
import sys

sys.path.insert(0, str(Path(__file__).parent))
import paths


# =============================================================================
# Constants
# =============================================================================

SWISS_CRS = "EPSG:2056"

# GeoPackage filenames
RAIL_SEG_GPKG     = "rail_segments.gpkg"
FEEDER_SEG_GPKG   = "pt_feeder_segments.gpkg"
RAIL_LINES_GPKG   = "rail_lines.gpkg"
FEEDER_LINES_GPKG = "pt_feeder_lines.gpkg"

# Legacy aliases — kept until Phase 0 / main are replaced in Phases 2 and 7
RAIL_GPKG   = RAIL_SEG_GPKG
FEEDER_GPKG = FEEDER_SEG_GPKG

# Column names — segments
SEG_ID_COL   = 'GTFS_ID'        # route identifier
SEG_NAME_COL = 'Service'         # line short name (e.g. 'S7')
SEG_TT_COL   = 'TT'             # travel time minutes
SEG_IVWT_COL = 'IVWT'           # in-vehicle wait minutes
SEG_FROM_NR  = 'from_stop_nr'   # BAV parent station number (from)
SEG_TO_NR    = 'to_stop_nr'     # BAV parent station number (to)

# Column names — lines
LINE_ID_COL   = 'route_id'       # matches SEG_ID_COL value
LINE_NAME_COL = 'line_short_name'

ADJUSTABLE_SEG_COLS = [
    'TT', 'tt_source', 'IVWT', 'service_period', 'Service', 'mode_label',
]
ADJUSTABLE_LINE_COLS = [
    'freq_am_peak_dep_hr', 'freq_pm_peak_dep_hr', 'freq_offpeak_dep_hr',
    'service_period', 'freq_directional',
]

# Track-based feeder layers eligible for editing (bus/ship excluded)
TRACK_BASED_FEEDER_LAYERS: Set[str] = {'tram', 'funicular', 'metro'}

_TT_SOURCE_VALUES = {'gtfs', 'formula'}

# Search radius (m) for matching a stop coordinate to the nearest BAV infra node
GAUGE_CHECK_RADIUS_M = 300


# =============================================================================
# Gauge validation helpers
# =============================================================================

def _load_base_infra() -> Tuple[Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]:
    """Load Base/ nodes and segments for gauge validation. Returns (None, None) on failure."""
    base = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_BASE
    nodes_path = base / "nodes.gpkg"
    segs_path  = base / "segments.gpkg"
    try:
        nodes = gpd.read_file(nodes_path) if nodes_path.exists() else None
        segs  = gpd.read_file(segs_path)  if segs_path.exists()  else None
        return nodes, segs
    except Exception:
        return None, None


def _gauge_at_stop(E: float, N: float,
                   infra_nodes: gpd.GeoDataFrame,
                   infra_segs:  gpd.GeoDataFrame) -> Optional[Set[int]]:
    """
    Return the set of gauge values (mm) found on segments connected to the
    nearest BAV node within GAUGE_CHECK_RADIUS_M.  Returns None when no node
    is found in range.
    """
    from shapely.geometry import Point as _Pt
    pt = _Pt(E, N)
    dists = infra_nodes.geometry.distance(pt)
    nearest_dist = float(dists.min())
    if nearest_dist > GAUGE_CHECK_RADIUS_M:
        return None
    nearest_name = str(infra_nodes.loc[dists.idxmin(), 'NAME'])
    conn = infra_segs[
        (infra_segs['from_name'] == nearest_name) |
        (infra_segs['to_name']   == nearest_name)
    ]
    gauges: Set[int] = set()
    for _, seg in conn.iterrows():
        g = seg.get('gauge')
        if g is not None and not (isinstance(g, float) and np.isnan(g)):
            try:
                gauges.add(int(float(g)))
            except (ValueError, TypeError):
                pass
    return gauges if gauges else None


def _check_gauge_compat(
    from_E: float, from_N: float,
    to_E:   float, to_N:   float,
    existing_layers: Dict[str, gpd.GeoDataFrame],
    layer_name: str,
) -> bool:
    """
    Validate gauge compatibility before adding a new service segment.

    Checks:
    1. From-stop and to-stop gauge (via Base infra) must not conflict.
    2. If the route already has segments, the new endpoints must match the
       route's established gauge.

    Prints a warning and prompts for confirmation when a mismatch is found.
    Returns True to proceed, False to cancel.
    """
    infra_nodes, infra_segs = _load_base_infra()
    if infra_nodes is None or infra_segs is None:
        return True  # cannot validate — proceed without check

    from_gauges = _gauge_at_stop(from_E, from_N, infra_nodes, infra_segs)
    to_gauges   = _gauge_at_stop(to_E,   to_N,   infra_nodes, infra_segs)

    # Both endpoints resolved and different gauge → hard conflict
    if from_gauges and to_gauges and not from_gauges.intersection(to_gauges):
        print(
            f"\n  ⚠  GAUGE CONFLICT: from-stop is on {sorted(from_gauges)}mm infrastructure"
            f" but to-stop is on {sorted(to_gauges)}mm infrastructure."
        )
        print("  A single service segment cannot span incompatible gauge infrastructure.")
        ans = input("  Add anyway? (y/n) [n]: ").strip().lower() or 'n'
        return ans == 'y'

    # Check against route's established gauge from existing segments
    gdf = existing_layers.get(layer_name)
    if gdf is not None and not gdf.empty:
        existing_gauges: Set[int] = set()
        for stop_e, stop_n in (
            list(zip(gdf.get('from_stop_E', pd.Series(dtype=float)),
                     gdf.get('from_stop_N', pd.Series(dtype=float)))) +
            list(zip(gdf.get('to_stop_E',   pd.Series(dtype=float)),
                     gdf.get('to_stop_N',   pd.Series(dtype=float))))
        ):
            try:
                g = _gauge_at_stop(float(stop_e), float(stop_n), infra_nodes, infra_segs)
                if g:
                    existing_gauges |= g
            except (TypeError, ValueError):
                pass

        endpoint_gauges = (from_gauges or set()) | (to_gauges or set())
        if existing_gauges and endpoint_gauges and not existing_gauges.intersection(endpoint_gauges):
            print(
                f"\n  ⚠  GAUGE MISMATCH: existing route segments use {sorted(existing_gauges)}mm"
                f" but new endpoints are on {sorted(endpoint_gauges)}mm infrastructure."
            )
            ans = input("  Add anyway? (y/n) [n]: ").strip().lower() or 'n'
            return ans == 'y'

    return True


# =============================================================================
# Version Discovery
# =============================================================================

def list_svc_versions() -> List[str]:
    """Return svc_version names that have an Unprojected/ rail base."""
    root = Path(paths.MAIN) / paths.RAIL_LINES_DIR
    if not root.exists():
        return []
    return [
        d.name for d in sorted(root.iterdir())
        if d.is_dir()
        and (d / paths.SERVICES_UNPROJECTED_SUBDIR / RAIL_SEG_GPKG).exists()
    ]


def list_all_networks() -> List[str]:
    """Return all *_network folder names in Rail_Lines/ that have a complete Unprojected/ base."""
    return list_svc_versions()


# =============================================================================
# Search / pick helpers
# =============================================================================

def _search_svc_segments(gdf: gpd.GeoDataFrame, term: str) -> gpd.GeoDataFrame:
    """Return rows whose GTFS_ID, Service, from_stop_name, or to_stop_name
    contain term (case-insensitive)."""
    t = term.lower()
    mask = (
        gdf.get(SEG_ID_COL,       pd.Series(dtype=str)).fillna('').str.lower().str.contains(t, regex=False) |
        gdf.get(SEG_NAME_COL,     pd.Series(dtype=str)).fillna('').str.lower().str.contains(t, regex=False) |
        gdf.get('from_stop_name', pd.Series(dtype=str)).fillna('').str.lower().str.contains(t, regex=False) |
        gdf.get('to_stop_name',   pd.Series(dtype=str)).fillna('').str.lower().str.contains(t, regex=False)
    )
    return gdf[mask]


def _pick_one(labels: List[str], prompt: str = "Select") -> Optional[int]:
    """
    Display a numbered list; return 0-based index of chosen item.
    Returns None if user presses Enter without a number.
    """
    for i, lbl in enumerate(labels, 1):
        print(f"     {i}) {lbl}")
    while True:
        raw = input(f"   {prompt} (number): ").strip()
        if not raw:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(labels):
            return int(raw) - 1
        print(f"   Invalid — enter 1–{len(labels)} or press Enter to cancel.")


# =============================================================================
# GPKG load/save helpers
# =============================================================================

def _load_all_layers(gpkg_path: Path) -> Dict[str, gpd.GeoDataFrame]:
    """Load all layers from a GeoPackage into a dict keyed by layer name."""
    if not gpkg_path.exists():
        return {}
    layer_names = fiona.listlayers(str(gpkg_path))
    result = {}
    for name in layer_names:
        gdf = gpd.read_file(gpkg_path, layer=name).reset_index(drop=True)
        if 'tt_source' not in gdf.columns:
            gdf['tt_source'] = 'gtfs'
        result[name] = gdf
    return result


def _save_all_layers(layers: Dict[str, gpd.GeoDataFrame], gpkg_path: Path):
    """Write all layers to a GeoPackage (overwrites if exists)."""
    if gpkg_path.exists():
        gpkg_path.unlink()
    for layer_name, gdf in layers.items():
        if gdf.empty:
            continue
        gdf.to_file(str(gpkg_path), layer=layer_name, driver='GPKG')


def _load_svc_data(
    rail_dir: Path,
    feeder_dir: Path,
) -> Tuple[
    Dict[str, gpd.GeoDataFrame], Dict[str, gpd.GeoDataFrame],
    Dict[str, gpd.GeoDataFrame], Dict[str, gpd.GeoDataFrame],
]:
    """Load rail and feeder segment and line dicts from the given directories.

    Returns (rail_seg, rail_line, feed_seg, feed_line).
    Missing gpkgs return empty dicts.
    """
    rail_seg   = _load_all_layers(rail_dir   / RAIL_SEG_GPKG)
    rail_line  = _load_all_layers(rail_dir   / RAIL_LINES_GPKG)
    feed_seg   = _load_all_layers(feeder_dir / FEEDER_SEG_GPKG)
    feed_line  = _load_all_layers(feeder_dir / FEEDER_LINES_GPKG)
    return rail_seg, rail_line, feed_seg, feed_line


def _save_svc_data(
    rail_seg:       Dict[str, gpd.GeoDataFrame],
    rail_line:      Dict[str, gpd.GeoDataFrame],
    feed_seg:       Dict[str, gpd.GeoDataFrame],
    feed_line:      Dict[str, gpd.GeoDataFrame],
    out_rail_dir:   Path,
    out_feeder_dir: Path,
) -> None:
    """Write all four gpkgs to the given output directories."""
    out_rail_dir.mkdir(parents=True, exist_ok=True)
    out_feeder_dir.mkdir(parents=True, exist_ok=True)

    _save_all_layers(rail_seg,  out_rail_dir   / RAIL_SEG_GPKG)
    _save_all_layers(rail_line, out_rail_dir   / RAIL_LINES_GPKG)
    _save_all_layers(feed_seg,  out_feeder_dir / FEEDER_SEG_GPKG)
    _save_all_layers(feed_line, out_feeder_dir / FEEDER_LINES_GPKG)

    print(f"  → {out_rail_dir   / RAIL_SEG_GPKG}")
    print(f"  → {out_rail_dir   / RAIL_LINES_GPKG}")
    print(f"  → {out_feeder_dir / FEEDER_SEG_GPKG}")
    print(f"  → {out_feeder_dir / FEEDER_LINES_GPKG}")


def _parse_selection(ans: str, max_count: int) -> List[int]:
    """Parse a diff-selection string into sorted 0-based indices.

    Accepts 'all', individual numbers, ranges (e.g. '21-25'), and any
    comma-separated combination. Out-of-range entries are silently ignored.
    """
    if ans.strip().lower() == 'all':
        return list(range(max_count))
    indices: List[int] = []
    for part in ans.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            bounds = part.split('-', 1)
            if bounds[0].isdigit() and bounds[1].isdigit():
                lo = int(bounds[0]) - 1
                hi = int(bounds[1]) - 1
                for i in range(min(lo, hi), max(lo, hi) + 1):
                    if 0 <= i < max_count:
                        indices.append(i)
        elif part.isdigit():
            val = int(part) - 1
            if 0 <= val < max_count:
                indices.append(val)
    return indices


# =============================================================================
# Infrastructure version helpers (for station catalog)
# =============================================================================

def list_infra_versions() -> List[str]:
    """Return infra version names that have nodes.gpkg (excludes Raw*)."""
    root = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR
    if not root.exists():
        return []
    versions = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if sub.name.startswith('Raw'):
            continue
        if (sub / 'nodes.gpkg').exists():
            versions.append(sub.name)
    base_versions  = ['Base'] if 'Base' in versions else []
    base_versions += sorted([v for v in versions if v.startswith('Base') and v != 'Base'])
    other_versions = [v for v in versions if not v.startswith('Base')]
    return base_versions + other_versions


def _load_station_catalog(infra_version: str) -> gpd.GeoDataFrame:
    """Load BAV nodes filtered to station class from the chosen infra version."""
    nodes_path = (
        Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR
        / infra_version / 'nodes.gpkg'
    )
    if not nodes_path.exists():
        print(f"  WARNING: Infra nodes not found at {nodes_path}. Station search disabled.")
        return gpd.GeoDataFrame()
    nodes = gpd.read_file(nodes_path)
    # Handle both capitalisation conventions
    cls_col = 'Node_Class' if 'Node_Class' in nodes.columns else 'node_class'
    if cls_col in nodes.columns:
        stations = nodes[nodes[cls_col] == 'station'].reset_index(drop=True)
    else:
        stations = nodes.reset_index(drop=True)
    print(f"  Station catalog: {len(stations)} stations from '{infra_version}'.")
    return stations


# =============================================================================
# Phase 0 — version selection
# =============================================================================

def _run_phase0(
    auto_infra_version: Optional[str] = None,
    auto_network: Optional[str] = None,
) -> Tuple[
    Dict[str, gpd.GeoDataFrame], Dict[str, gpd.GeoDataFrame],
    Dict[str, gpd.GeoDataFrame], Dict[str, gpd.GeoDataFrame],
    gpd.GeoDataFrame, str, str, str,
]:
    """
    Phase 0 — version selection and setup.

    Args:
        auto_infra_version: When provided, auto-selects this infra version for the
            station catalog without prompting.
        auto_network: When provided, skips the mode menu and directly enters
            'adjust' mode for this network name (with or without '_network' suffix).

    Returns
    -------
    rail_seg, rail_line, feed_seg, feed_line,
    orig_rail_seg, orig_rail_line,
    station_catalog, version_name, out_rail_dir, out_feeder_dir, mode
    mode is 'new' or 'adjust'.
    orig_rail_seg / orig_rail_line are deep copies taken before editing for
    propagation diffing in Phase 3.
    """
    import copy as _copy

    all_networks = list_all_networks()
    if not all_networks:
        print("\n  ERROR: No network found in Rail_Lines/.")
        print("  Run services_network_builder.py first.")
        raise SystemExit(1)

    rail_base   = Path(paths.MAIN) / paths.RAIL_LINES_DIR
    feeder_base = Path(paths.MAIN) / paths.FEEDER_LINES_DIR

    if auto_network is not None:
        # Non-interactive: directly enter adjust mode for the given network
        version_name = (auto_network if auto_network.endswith('_network')
                        else auto_network + '_network')
        out_rail_dir   = rail_base   / version_name / paths.SERVICES_UNPROJECTED_SUBDIR
        out_feeder_dir = feeder_base / version_name / paths.SERVICES_UNPROJECTED_SUBDIR
        if not out_rail_dir.exists():
            print(f"\n  ERROR: Network '{version_name}' not found at {out_rail_dir}")
            raise SystemExit(1)
        mode = 'adjust'
        future_template = False
        print(f"\n  Auto-selected network '{version_name}' for adjustment.")
    else:
        print("\n" + "-" * 60)
        print("  What do you want to do?")
        print("    1) Create a new network version (copy from an existing one)")
        print("    2) Adjust an existing network")
        while True:
            choice = input("  Select (1/2): ").strip()
            if choice in ('1', '2'):
                break
            print("  Enter 1 or 2.")

        if choice == '1':
            mode = 'new'

            print("\n  Choose source network to copy from:")
            src_idx = _pick_one(all_networks, "Source network")
            if src_idx is None:
                raise SystemExit(0)
            src_name = all_networks[src_idx]

            while True:
                raw_name = input("\n  Name for the new network (without '_network' suffix): ").strip()
                if not raw_name:
                    print("  Name cannot be empty.")
                    continue
                version_name   = raw_name if raw_name.endswith('_network') else raw_name + '_network'
                out_rail_dir   = rail_base   / version_name / paths.SERVICES_UNPROJECTED_SUBDIR
                out_feeder_dir = feeder_base / version_name / paths.SERVICES_UNPROJECTED_SUBDIR
                if out_rail_dir.exists() and (out_rail_dir / RAIL_SEG_GPKG).exists():
                    print(f"  '{version_name}' already exists.")
                    overwrite = input("  Overwrite? (y/n) [n]: ").strip().lower() or 'n'
                    if overwrite != 'y':
                        continue
                    shutil.rmtree(out_rail_dir,   ignore_errors=True)
                    shutil.rmtree(out_feeder_dir, ignore_errors=True)
                break

            # Copy entire Unprojected/ tree (gpkgs, subfolders, QGIS projects)
            src_rail_unproj   = rail_base   / src_name / paths.SERVICES_UNPROJECTED_SUBDIR
            src_feeder_unproj = feeder_base / src_name / paths.SERVICES_UNPROJECTED_SUBDIR
            print(f"\n  Copying Unprojected/ from '{src_name}' → '{version_name}' ...")
            if src_rail_unproj.exists():
                shutil.copytree(str(src_rail_unproj), str(out_rail_dir))
                print(f"  Rail Unprojected/ copied.")
            else:
                print(f"  WARNING: Source rail Unprojected/ not found: {src_rail_unproj}")
                out_rail_dir.mkdir(parents=True, exist_ok=True)

            if src_feeder_unproj.exists():
                shutil.copytree(str(src_feeder_unproj), str(out_feeder_dir))
                print(f"  Feeder Unprojected/ copied.")
            else:
                out_feeder_dir.mkdir(parents=True, exist_ok=True)

            future_template = input(
                "\n  Apply future-scenario template (set all TTs to formula)? (y/n) [n]: "
            ).strip().lower() == 'y'

        else:
            mode = 'adjust'
            future_template = False
            print("\n  Choose network to adjust:")
            idx2 = _pick_one(all_networks, "Network")
            if idx2 is None:
                raise SystemExit(0)
            version_name   = all_networks[idx2]
            out_rail_dir   = rail_base   / version_name / paths.SERVICES_UNPROJECTED_SUBDIR
            out_feeder_dir = feeder_base / version_name / paths.SERVICES_UNPROJECTED_SUBDIR

    # -- Infra version for station catalog ------------------------------------
    infra_versions = list_infra_versions()
    station_catalog: gpd.GeoDataFrame = gpd.GeoDataFrame()
    if auto_infra_version and auto_infra_version in infra_versions:
        print(f"\n  Station catalog: auto-selected infra version '{auto_infra_version}'.")
        station_catalog = _load_station_catalog(auto_infra_version)
    elif infra_versions:
        print("\n" + "-" * 60)
        print("  Choose infra version for station catalog:")
        iv_idx = _pick_one(infra_versions, "Infra version")
        if iv_idx is not None:
            station_catalog = _load_station_catalog(infra_versions[iv_idx])
    else:
        print("  WARNING: No infra versions found — station search disabled.")

    # -- Load all four dicts --------------------------------------------------
    print(f"\n  Loading '{version_name}/Unprojected/' ...")
    rail_seg, rail_line, feed_seg, feed_line = _load_svc_data(out_rail_dir, out_feeder_dir)

    if future_template:
        for layers in (rail_seg, feed_seg):
            for gdf in layers.values():
                gdf['tt_source'] = 'formula'
                if SEG_TT_COL in gdf.columns:
                    gdf[SEG_TT_COL] = pd.NA
        print("  Future-scenario template: all TTs set to formula/sentinel.")

    orig_rail_seg  = {k: v.copy() for k, v in rail_seg.items()}
    orig_rail_line = {k: v.copy() for k, v in rail_line.items()}

    n_rail_seg  = sum(len(g) for g in rail_seg.values())
    n_rail_line = sum(len(g) for g in rail_line.values())
    n_feed_seg  = sum(len(g) for g in feed_seg.values())
    n_feed_line = sum(len(g) for g in feed_line.values())
    print(f"  Rail  : {len(rail_seg)} seg layer(s), {n_rail_seg} segments"
          f" / {len(rail_line)} line layer(s), {n_rail_line} lines")
    print(f"  Feeder: {len(feed_seg)} seg layer(s), {n_feed_seg} segments"
          f" / {len(feed_line)} line layer(s), {n_feed_line} lines")

    return (rail_seg, rail_line, feed_seg, feed_line,
            orig_rail_seg, orig_rail_line,
            station_catalog, version_name, out_rail_dir, out_feeder_dir, mode)


# =============================================================================
# Core editing helpers — stop chain, route display, line sync, station search
# =============================================================================

def _build_stop_chain(
    gdf: gpd.GeoDataFrame,
    gtfs_id: str,
    direction_id: int,
    variant_rank: int,
) -> List[pd.Series]:
    """Return segment rows in stop-sequence order for the given route variant.

    Follows the from_stop_nr → to_stop_nr chain. Returns an empty list when
    no rows match or the chain cannot be resolved.
    """
    mask = (
        (gdf[SEG_ID_COL].astype(str)    == str(gtfs_id)) &
        (gdf['direction_id'].astype(str) == str(direction_id)) &
        (gdf['variant_rank'].astype(str) == str(variant_rank))
    )
    rows = gdf[mask]
    if rows.empty:
        return []

    chain: dict   = {}
    seg_for: dict = {}
    for _, row in rows.iterrows():
        f_nr = row[SEG_FROM_NR]
        t_nr = row[SEG_TO_NR]
        chain[f_nr]       = t_nr
        seg_for[f_nr] = row

    all_to = set(chain.values())
    starts = [k for k in chain if k not in all_to]
    cur = starts[0] if starts else next(iter(chain))

    ordered: List[pd.Series] = []
    visited: set = set()
    while cur in chain and cur not in visited:
        ordered.append(seg_for[cur])
        visited.add(cur)
        cur = chain[cur]
    return ordered


def _display_stop_sequence(chain: List[pd.Series], title: str = '') -> None:
    """Print a numbered stop list with TT values for each segment."""
    if title:
        print(f"\n  {title}")
    print("  " + "-" * 56)
    if not chain:
        print("  (no stops)")
        return

    first = chain[0]
    print(f"    1) {first.get('from_stop_name', str(first.get(SEG_FROM_NR, '?')))}  (start)")
    for i, row in enumerate(chain, start=2):
        name  = row.get('to_stop_name', str(row.get(SEG_TO_NR, '?')))
        tt    = row.get(SEG_TT_COL)
        src   = row.get('tt_source', 'gtfs')
        if tt is None or (isinstance(tt, float) and pd.isna(tt)):
            tt_str = 'TT=formula'
        else:
            tt_str = f"TT={float(tt):.1f} min"
        suffix = f"  [formula]" if src == 'formula' else ''
        print(f"    {i}) {name}  {tt_str}{suffix}")
    print("  " + "-" * 56)
    print(f"  {len(chain) + 1} stops, {len(chain)} segment(s)")


def _pick_route_variant(
    layers: Dict[str, gpd.GeoDataFrame],
    prompt: str = "Select route variant",
) -> Optional[Tuple[str, str, int, int]]:
    """Search across all layers and return (layer_name, gtfs_id, direction_id, variant_rank).

    Prompts for a search term; Enter without input searches across all routes
    (capped at 80 displayed results). Returns None if the user cancels.
    """
    term = input(f"\n  {prompt} — search (Enter to list all): ").strip()

    items: List[Tuple[str, str, int, int, str, int]] = []  # (layer, gtfs_id, dir, var, label, n_segs)
    for layer_name, gdf in layers.items():
        if gdf.empty:
            continue
        sub = _search_svc_segments(gdf, term) if term else gdf
        if sub.empty:
            continue
        for keys, grp in sub.groupby(
            [SEG_ID_COL, 'direction_id', 'variant_rank'] if all(
                c in sub.columns for c in [SEG_ID_COL, 'direction_id', 'variant_rank']
            ) else [SEG_ID_COL],
            sort=True,
        ):
            if not isinstance(keys, tuple):
                keys = (keys, None, None)
            gtfs_id  = str(keys[0])
            dir_id   = str(keys[1]) if keys[1] is not None else '0'
            var_rank = str(keys[2]) if keys[2] is not None else '1'
            svc_name   = str(grp[SEG_NAME_COL].iloc[0]) if SEG_NAME_COL in grp.columns else gtfs_id
            n_segs     = len(grp)
            label = (f"{svc_name}  [{gtfs_id}]  dir={dir_id}  var={var_rank}"
                     f"  [{layer_name}]  {n_segs} seg(s)")
            items.append((layer_name, gtfs_id, dir_id, var_rank, label, n_segs))

    if not items:
        print(f"  No routes found{' matching ' + repr(term) if term else ''}.")
        return None

    if len(items) > 80:
        print(f"  {len(items)} variants found — showing first 80. Refine your search.")
        items = items[:80]

    labels = [it[4] for it in items]
    idx = _pick_one(labels, "Route variant")
    if idx is None:
        return None
    layer_name, gtfs_id, dir_id, var_rank, _, _ = items[idx]
    return layer_name, gtfs_id, dir_id, var_rank


def _sync_line_from_segments(
    seg_layers:  Dict[str, gpd.GeoDataFrame],
    line_layers: Dict[str, gpd.GeoDataFrame],
    layer_name:  str,
    gtfs_id:     str,
    direction_id: int,
    variant_rank: int,
) -> Dict[str, gpd.GeoDataFrame]:
    """Update (or remove) the line feature for a route variant after segment changes.

    Recalculates geometry and n_stops from the current segment rows.
    Preserves all frequency and other line attributes.
    Removes the line record when no segments remain.
    Returns the updated line_layers dict.
    """
    seg_gdf  = seg_layers.get(layer_name, gpd.GeoDataFrame())
    line_gdf = line_layers.get(layer_name, gpd.GeoDataFrame())

    # Locate the matching line record (route_id == gtfs_id)
    has_line_cols = (not line_gdf.empty
                     and LINE_ID_COL    in line_gdf.columns
                     and 'direction_id' in line_gdf.columns
                     and 'variant_rank' in line_gdf.columns)
    if has_line_cols:
        line_mask = (
            (line_gdf[LINE_ID_COL].astype(str)    == str(gtfs_id)) &
            (line_gdf['direction_id'].astype(str)  == str(direction_id)) &
            (line_gdf['variant_rank'].astype(str)  == str(variant_rank))
        )
    else:
        line_mask = pd.Series(False, index=line_gdf.index) if not line_gdf.empty \
                    else pd.Series(dtype=bool)

    # Build the current stop chain
    chain = _build_stop_chain(seg_gdf, gtfs_id, direction_id, variant_rank)

    if not chain:
        # No segments left — remove line record
        if has_line_cols and line_mask.any():
            line_layers[layer_name] = line_gdf[~line_mask].reset_index(drop=True)
        return line_layers

    # Rebuild geometry from segment chain
    geoms = [r.geometry for r in chain if r.geometry is not None]
    try:
        merged_geom = linemerge(geoms) if geoms else None
    except Exception:
        merged_geom = geoms[0] if geoms else None

    n_stops     = len(chain) + 1
    first_seg   = chain[0]
    last_seg    = chain[-1]
    origin      = first_seg.get('from_stop_name', str(first_seg.get(SEG_FROM_NR, '')))
    destination = last_seg.get('to_stop_name',   str(last_seg.get(SEG_TO_NR,   '')))

    if has_line_cols and line_mask.any():
        # Update existing record in place
        idx = line_gdf[line_mask].index[0]
        line_gdf.at[idx, 'geometry']    = merged_geom
        line_gdf.at[idx, 'n_stops']     = n_stops
        line_gdf.at[idx, 'origin']      = origin
        line_gdf.at[idx, 'destination'] = destination
        line_layers[layer_name] = line_gdf
    else:
        # No existing record — create a minimal one from segment attributes
        seg_row = first_seg
        new_rec = {
            LINE_ID_COL:      gtfs_id,
            LINE_NAME_COL:    seg_row.get(SEG_NAME_COL, ''),
            'direction_id':   direction_id,
            'variant_rank':   variant_rank,
            'origin':         origin,
            'destination':    destination,
            'n_stops':        n_stops,
            'mode_label':     seg_row.get('mode_label', ''),
            'mode_class':     seg_row.get('mode_class', ''),
            'service_period': seg_row.get('service_period', ''),
            'geometry':       merged_geom,
        }
        new_line_gdf = gpd.GeoDataFrame([new_rec], crs=SWISS_CRS)
        existing = line_layers.get(layer_name, gpd.GeoDataFrame())
        line_layers[layer_name] = pd.concat(
            [existing, new_line_gdf], ignore_index=True
        ) if not existing.empty else new_line_gdf.reset_index(drop=True)

    return line_layers


def _station_search(
    catalog: gpd.GeoDataFrame,
    term: str,
) -> gpd.GeoDataFrame:
    """Return stations whose Name or Code contain term (case-insensitive)."""
    if catalog.empty:
        return gpd.GeoDataFrame()
    t = term.lower()
    name_col = 'Name' if 'Name' in catalog.columns else (
               'name' if 'name' in catalog.columns else None)
    code_col = 'Code' if 'Code' in catalog.columns else (
               'code' if 'code' in catalog.columns else None)
    masks = []
    if name_col:
        masks.append(catalog[name_col].fillna('').str.lower().str.contains(t, regex=False))
    if code_col:
        masks.append(catalog[code_col].fillna('').str.lower().str.contains(t, regex=False))
    if not masks:
        return gpd.GeoDataFrame()
    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m
    return catalog[combined].reset_index(drop=True)


def _pick_station(
    catalog: gpd.GeoDataFrame,
    prompt: str = "Search station",
) -> Optional[pd.Series]:
    """Interactive station picker using the BAV infra catalog.

    Returns the chosen station row (with Name, Code, Number, E, N) or None.
    """
    if catalog.empty:
        print("  Station catalog not available — enter coordinates manually.")
        return None
    while True:
        term = input(f"\n  {prompt} (partial name or code, Enter to cancel): ").strip()
        if not term:
            return None
        hits = _station_search(catalog, term)
        if hits.empty:
            print(f"  No stations matching '{term}'. Try again.")
            continue
        name_col = 'Name' if 'Name' in hits.columns else 'name'
        code_col = 'Code' if 'Code' in hits.columns else 'code'
        labels = [
            f"{r.get(name_col, '?')}  [{r.get(code_col, '?')}]"
            for _, r in hits.iterrows()
        ]
        idx = _pick_one(labels, "Station")
        if idx is None:
            continue
        return hits.iloc[idx]


# =============================================================================
# Segment operations
# =============================================================================

def _remove_route(
    seg_layers:  Dict[str, gpd.GeoDataFrame],
    line_layers: Dict[str, gpd.GeoDataFrame],
    layer_name:  str,
) -> Tuple[Dict[str, gpd.GeoDataFrame], Dict[str, gpd.GeoDataFrame]]:
    """Remove an entire route (one variant or all variants) from seg and line layers."""
    if not seg_layers.get(layer_name, gpd.GeoDataFrame()).shape[0]:
        print("  No segments in this layer.")
        return seg_layers, line_layers

    result = _pick_route_variant({layer_name: seg_layers[layer_name]})
    if result is None:
        return seg_layers, line_layers
    _, gtfs_id, dir_id, var_rank = result

    gdf   = seg_layers[layer_name]
    chain = _build_stop_chain(gdf, gtfs_id, dir_id, var_rank)
    _display_stop_sequence(chain, f"{gtfs_id}  dir={dir_id}  var={var_rank}")

    print("\n  Remove:")
    print("    1) This variant only  (dir=%s  var=%s)" % (dir_id, var_rank))
    print("    2) All variants of this GTFS_ID  [%s]" % gtfs_id)
    while True:
        c = input("  Select (1/2): ").strip()
        if c in ('1', '2'):
            break
        print("  Enter 1 or 2.")

    if c == '1':
        keep_mask = ~(
            (gdf[SEG_ID_COL].astype(str)    == str(gtfs_id)) &
            (gdf['direction_id'].astype(str) == str(dir_id)) &
            (gdf['variant_rank'].astype(str) == str(var_rank))
        )
        n_removed = int((~keep_mask).sum())
        seg_layers[layer_name] = gdf[keep_mask].reset_index(drop=True)
        line_layers = _sync_line_from_segments(
            seg_layers, line_layers, layer_name, gtfs_id, dir_id, var_rank
        )
        print("  Removed %d segment(s) for variant %s dir=%s var=%s." % (
            n_removed, gtfs_id, dir_id, var_rank))
    else:
        id_mask = gdf[SEG_ID_COL].astype(str) == str(gtfs_id)
        variants_df = gdf[id_mask][['direction_id', 'variant_rank']].drop_duplicates()
        n_removed = int(id_mask.sum())
        seg_layers[layer_name] = gdf[~id_mask].reset_index(drop=True)
        for _, vrow in variants_df.iterrows():
            line_layers = _sync_line_from_segments(
                seg_layers, line_layers, layer_name,
                gtfs_id, str(vrow['direction_id']), str(vrow['variant_rank'])
            )
        print("  Removed %d segment(s) for all variants of '%s'." % (n_removed, gtfs_id))

    return seg_layers, line_layers


def _trim_route(
    seg_layers:  Dict[str, gpd.GeoDataFrame],
    line_layers: Dict[str, gpd.GeoDataFrame],
    layer_name:  str,
) -> Tuple[Dict[str, gpd.GeoDataFrame], Dict[str, gpd.GeoDataFrame]]:
    """Trim a route by dropping stops from the head or tail."""
    if not seg_layers.get(layer_name, gpd.GeoDataFrame()).shape[0]:
        print("  No segments in this layer.")
        return seg_layers, line_layers

    result = _pick_route_variant({layer_name: seg_layers[layer_name]})
    if result is None:
        return seg_layers, line_layers
    _, gtfs_id, dir_id, var_rank = result

    gdf   = seg_layers[layer_name]
    chain = _build_stop_chain(gdf, gtfs_id, dir_id, var_rank)
    if not chain:
        print("  No segments found for this variant.")
        return seg_layers, line_layers
    if len(chain) < 2:
        print("  Route has only one segment — nothing to trim.")
        return seg_layers, line_layers

    _display_stop_sequence(chain, f"{gtfs_id}  dir={dir_id}  var={var_rank}")
    n_stops = len(chain) + 1

    print("\n  Trim from:")
    print("    1) Head (remove stops from the start)")
    print("    2) Tail (remove stops from the end)")
    while True:
        c = input("  Select (1/2): ").strip()
        if c in ('1', '2'):
            break
        print("  Enter 1 or 2.")
    trim_head = (c == '1')

    if trim_head:
        max_stop = n_stops - 1
        print("\n  New first stop — pick a stop number (2 to %d):" % max_stop)
        while True:
            raw = input("  Stop number: ").strip()
            if raw.isdigit() and 2 <= int(raw) <= max_stop:
                pivot = int(raw)
                break
            print("  Enter a number between 2 and %d." % max_stop)
        # Remove segments chain[0 .. pivot-2] (stops before new first stop)
        remove_idx  = {chain[i].name for i in range(pivot - 1)}
        new_terminus = chain[pivot - 1].get('from_stop_name',
                        str(chain[pivot - 1].get(SEG_FROM_NR, '?')))
    else:
        min_stop = 2
        max_stop = n_stops - 1
        print("\n  New last stop — pick a stop number (2 to %d):" % max_stop)
        while True:
            raw = input("  Stop number: ").strip()
            if raw.isdigit() and min_stop <= int(raw) <= max_stop:
                pivot = int(raw)
                break
            print("  Enter a number between 2 and %d." % max_stop)
        # Remove segments chain[pivot-1 ..] (stops after new last stop)
        remove_idx  = {chain[i].name for i in range(pivot - 1, len(chain))}
        new_terminus = chain[pivot - 2].get('to_stop_name',
                        str(chain[pivot - 2].get(SEG_TO_NR, '?')))

    n_before = len(gdf)
    seg_layers[layer_name] = gdf[~gdf.index.isin(remove_idx)].reset_index(drop=True)
    n_removed = n_before - len(seg_layers[layer_name])
    line_layers = _sync_line_from_segments(
        seg_layers, line_layers, layer_name, gtfs_id, dir_id, var_rank
    )
    action = "head" if trim_head else "tail"
    print("  Trimmed %s: removed %d segment(s). New terminus: %s." % (
        action, n_removed, new_terminus))
    return seg_layers, line_layers


_SERVICE_PERIOD_OPTIONS = ['all_day', 'peak_only', 'offpeak_only', 'unknown']


def _adjust_segment(
    seg_layers:  Dict[str, gpd.GeoDataFrame],
    line_layers: Dict[str, gpd.GeoDataFrame],
    layer_name:  str,
) -> Tuple[Dict[str, gpd.GeoDataFrame], Dict[str, gpd.GeoDataFrame]]:
    """Field-by-field editing of a single segment's adjustable attributes."""
    if not seg_layers.get(layer_name, gpd.GeoDataFrame()).shape[0]:
        print("  No segments in this layer.")
        return seg_layers, line_layers

    result = _pick_route_variant({layer_name: seg_layers[layer_name]})
    if result is None:
        return seg_layers, line_layers
    _, gtfs_id, dir_id, var_rank = result

    gdf   = seg_layers[layer_name]
    chain = _build_stop_chain(gdf, gtfs_id, dir_id, var_rank)
    if not chain:
        print("  No segments found.")
        return seg_layers, line_layers

    _display_stop_sequence(chain, "%s  dir=%s  var=%s" % (gtfs_id, dir_id, var_rank))
    seg_labels = [
        "Seg %d:  %s  ->  %s   TT=%s  [%s]" % (
            i + 1,
            s.get('from_stop_name', '?'),
            s.get('to_stop_name',   '?'),
            s.get(SEG_TT_COL,       'formula') if not (
                s.get(SEG_TT_COL) is None or
                (isinstance(s.get(SEG_TT_COL), float) and pd.isna(s.get(SEG_TT_COL)))
            ) else 'formula',
            s.get('tt_source', 'gtfs'),
        )
        for i, s in enumerate(chain)
    ]
    print("\n  Which segment to edit?")
    seg_idx = _pick_one(seg_labels, "Segment")
    if seg_idx is None:
        return seg_layers, line_layers

    target    = chain[seg_idx]
    row_idx   = target.name
    row       = gdf.loc[row_idx]

    print("\n  Editing  %s -> %s  (Enter to keep current value)" % (
        row.get('from_stop_name', '?'), row.get('to_stop_name', '?')))

    changed = False

    # TT + tt_source (linked field)
    cur_tt  = row.get(SEG_TT_COL)
    cur_src = str(row.get('tt_source', 'gtfs'))
    tt_display = 'formula' if (cur_tt is None or (isinstance(cur_tt, float) and pd.isna(cur_tt))) \
                 else str(cur_tt)
    raw = input("  TT (current: %s, tt_source: %s): " % (tt_display, cur_src)).strip()
    if raw:
        if raw.lower() in ('f', 'formula', 'auto'):
            gdf.at[row_idx, SEG_TT_COL]   = pd.NA
            gdf.at[row_idx, 'tt_source']   = 'formula'
            changed = True
        else:
            try:
                gdf.at[row_idx, SEG_TT_COL]   = max(0.1, float(raw))
                gdf.at[row_idx, 'tt_source']   = 'gtfs'
                changed = True
            except ValueError:
                print("  Invalid TT — skipped.")

    # IVWT
    cur_ivwt = row.get(SEG_IVWT_COL, '')
    raw = input("  IVWT (current: %s): " % cur_ivwt).strip()
    if raw:
        try:
            gdf.at[row_idx, SEG_IVWT_COL] = max(0.1, float(raw))
            changed = True
        except ValueError:
            print("  Invalid IVWT — skipped.")

    # service_period
    cur_sp = str(row.get('service_period', ''))
    print("  service_period (current: %s):" % cur_sp)
    for i, opt in enumerate(_SERVICE_PERIOD_OPTIONS, 1):
        print("    %d) %s" % (i, opt))
    raw = input("  Select (1-%d, Enter to keep): " % len(_SERVICE_PERIOD_OPTIONS)).strip()
    if raw.isdigit() and 1 <= int(raw) <= len(_SERVICE_PERIOD_OPTIONS):
        gdf.at[row_idx, 'service_period'] = _SERVICE_PERIOD_OPTIONS[int(raw) - 1]
        changed = True

    # Service name
    cur_svc = str(row.get(SEG_NAME_COL, ''))
    raw = input("  Service name (current: %s): " % cur_svc).strip()
    if raw:
        gdf.at[row_idx, SEG_NAME_COL] = raw
        changed = True

    # mode_label
    cur_ml = str(row.get('mode_label', ''))
    raw = input("  mode_label (current: %s): " % cur_ml).strip()
    if raw:
        gdf.at[row_idx, 'mode_label'] = raw
        changed = True

    if changed:
        seg_layers[layer_name] = gdf
        print("  Segment updated.")
    else:
        print("  No changes made.")
    return seg_layers, line_layers


def _prompt_tt(prompt: str = "TT (min, or 'formula' / 'f')") -> Tuple[Optional[float], str]:
    """Prompt for a travel time. Returns (float_or_None, tt_source_str)."""
    while True:
        raw = input("  %s: " % prompt).strip()
        if not raw:
            print("  Travel time required.")
            continue
        if raw.lower() in ('f', 'formula', 'auto'):
            return None, 'formula'
        try:
            val = float(raw)
            if val <= 0:
                print("  Must be > 0.")
                continue
            return max(0.1, val), 'gtfs'
        except ValueError:
            print("  Enter a number or 'formula' / 'f'.")


def _mode_class_for_layer(layer_name: str) -> str:
    """Return 'rail' for rail layers, 'pt_feeder' for track-based feeder layers."""
    return 'rail' if layer_name in {
        'sbahn', 'long_distance_rail', 'inter_regional_rail', 'regional_rail'
    } else 'pt_feeder'


def _new_seg_dict(
    gtfs_id:        str,
    service_name:   str,
    dir_id:         str,
    var_rank:       str,
    mode_label:     str,
    mode_class:     str,
    service_period: str,
    from_nr,
    from_name:      str,
    from_e:         float,
    from_n:         float,
    to_nr,
    to_name:        str,
    to_e:           float,
    to_n:           float,
    tt,
    tt_source:      str,
    ivwt:           float = 0.5,
) -> dict:
    """Build a new segment row dict with the canonical column schema."""
    geom = LineString([(from_e, from_n), (to_e, to_n)])
    return {
        SEG_ID_COL:       gtfs_id,
        SEG_NAME_COL:     service_name,
        'direction_id':   dir_id,
        'variant_rank':   var_rank,
        'mode_label':     mode_label,
        'mode_class':     mode_class,
        'service_period': service_period,
        SEG_FROM_NR:      from_nr,
        SEG_TO_NR:        to_nr,
        'from_stop_name': from_name,
        'to_stop_name':   to_name,
        'from_stop_E':    from_e,
        'from_stop_N':    from_n,
        'to_stop_E':      to_e,
        'to_stop_N':      to_n,
        SEG_TT_COL:       tt,
        SEG_IVWT_COL:     ivwt,
        'tt_source':      tt_source,
        'geometry':       geom,
    }


def _append_seg_to_layer(
    layers:     Dict[str, gpd.GeoDataFrame],
    layer_name: str,
    row_dict:   dict,
) -> Dict[str, gpd.GeoDataFrame]:
    """Append a new segment row to a layer, aligning columns with the existing data."""
    new_gdf = gpd.GeoDataFrame([row_dict], crs=SWISS_CRS)
    existing = layers.get(layer_name, gpd.GeoDataFrame())
    if existing.empty:
        layers[layer_name] = new_gdf.reset_index(drop=True)
    else:
        for col in existing.columns:
            if col not in new_gdf.columns:
                new_gdf[col] = pd.NA
        layers[layer_name] = pd.concat(
            [existing, new_gdf[existing.columns]], ignore_index=True
        )
    return layers


def _resolve_station(station_row: pd.Series) -> Tuple[int, str, float, float]:
    """Extract (nr, name, E, N) from a BAV catalog station row."""
    name_col = 'Name' if 'Name' in station_row.index else 'name'
    nr_col   = 'Number' if 'Number' in station_row.index else 'number'
    e_col    = 'E' if 'E' in station_row.index else 'e'
    n_col    = 'N' if 'N' in station_row.index else 'n'
    return (
        int(station_row.get(nr_col, 0)),
        str(station_row.get(name_col, '')),
        float(station_row.get(e_col, 0)),
        float(station_row.get(n_col, 0)),
    )


def _pick_station_or_manual(
    catalog: gpd.GeoDataFrame,
    prompt:  str,
) -> Optional[Tuple[int, str, float, float]]:
    """Pick a station from catalog or enter manually. Returns (nr, name, E, N) or None."""
    if not catalog.empty:
        srow = _pick_station(catalog, prompt)
        if srow is not None:
            return _resolve_station(srow)
    print("  Enter stop details manually (or press Enter to cancel):")
    name = input("  Stop name: ").strip()
    if not name:
        return None
    try:
        nr  = int(input("  Stop number (BAV): ").strip())
        e   = float(input("  E (m, EPSG:2056): ").strip())
        n   = float(input("  N (m, EPSG:2056): ").strip())
        return nr, name, e, n
    except ValueError:
        print("  Invalid input. Cancelled.")
        return None


# =============================================================================
# Stop sequence operations
# =============================================================================

def _drop_stop(
    seg_layers:  Dict[str, gpd.GeoDataFrame],
    line_layers: Dict[str, gpd.GeoDataFrame],
    layer_name:  str,
) -> Tuple[Dict[str, gpd.GeoDataFrame], Dict[str, gpd.GeoDataFrame]]:
    """Drop a middle stop from a route, merging the two surrounding segments into one."""
    if not seg_layers.get(layer_name, gpd.GeoDataFrame()).shape[0]:
        print("  No segments in this layer.")
        return seg_layers, line_layers

    result = _pick_route_variant({layer_name: seg_layers[layer_name]})
    if result is None:
        return seg_layers, line_layers
    _, gtfs_id, dir_id, var_rank = result

    gdf   = seg_layers[layer_name]
    chain = _build_stop_chain(gdf, gtfs_id, dir_id, var_rank)
    if len(chain) < 2:
        print("  Route needs at least 3 stops to drop a middle stop.")
        return seg_layers, line_layers

    _display_stop_sequence(chain, "%s  dir=%s  var=%s" % (gtfs_id, dir_id, var_rank))
    n_stops = len(chain) + 1

    print("\n  Pick stop to drop (2 to %d):" % (n_stops - 1))
    while True:
        raw = input("  Stop number: ").strip()
        if raw.isdigit() and 2 <= int(raw) <= n_stops - 1:
            drop_n = int(raw)
            break
        print("  Enter a number between 2 and %d." % (n_stops - 1))

    seg_before = chain[drop_n - 2]  # ends at drop stop
    seg_after  = chain[drop_n - 1]  # starts at drop stop
    drop_name  = seg_before.get('to_stop_name', '?')

    print("\n  Merging %s -> %s -> %s into one segment." % (
        seg_before.get('from_stop_name', '?'), drop_name, seg_after.get('to_stop_name', '?')))
    tt, tt_src = _prompt_tt("TT for merged segment")
    try:
        ivwt = max(0.1, float(input("  IVWT [0.5]: ").strip() or '0.5'))
    except ValueError:
        ivwt = 0.5

    new_row = _new_seg_dict(
        gtfs_id, seg_before.get(SEG_NAME_COL, ''), dir_id, var_rank,
        seg_before.get('mode_label', ''), seg_before.get('mode_class', ''),
        seg_before.get('service_period', ''),
        seg_before.get(SEG_FROM_NR), seg_before.get('from_stop_name', ''),
        float(seg_before.get('from_stop_E', 0)), float(seg_before.get('from_stop_N', 0)),
        seg_after.get(SEG_TO_NR), seg_after.get('to_stop_name', ''),
        float(seg_after.get('to_stop_E', 0)), float(seg_after.get('to_stop_N', 0)),
        tt, tt_src, ivwt,
    )

    remove_idx = {seg_before.name, seg_after.name}
    seg_layers[layer_name] = gdf[~gdf.index.isin(remove_idx)].reset_index(drop=True)
    seg_layers = _append_seg_to_layer(seg_layers, layer_name, new_row)
    line_layers = _sync_line_from_segments(
        seg_layers, line_layers, layer_name, gtfs_id, dir_id, var_rank
    )
    print("  Dropped stop '%s'. Route now has %d stops." % (drop_name, n_stops - 1))
    return seg_layers, line_layers


def _insert_stop(
    seg_layers:     Dict[str, gpd.GeoDataFrame],
    line_layers:    Dict[str, gpd.GeoDataFrame],
    layer_name:     str,
    station_catalog: gpd.GeoDataFrame,
) -> Tuple[Dict[str, gpd.GeoDataFrame], Dict[str, gpd.GeoDataFrame]]:
    """Split an existing segment by inserting a new intermediate stop."""
    if not seg_layers.get(layer_name, gpd.GeoDataFrame()).shape[0]:
        print("  No segments in this layer.")
        return seg_layers, line_layers

    result = _pick_route_variant({layer_name: seg_layers[layer_name]})
    if result is None:
        return seg_layers, line_layers
    _, gtfs_id, dir_id, var_rank = result

    gdf   = seg_layers[layer_name]
    chain = _build_stop_chain(gdf, gtfs_id, dir_id, var_rank)
    if not chain:
        print("  No segments found.")
        return seg_layers, line_layers

    _display_stop_sequence(chain, "%s  dir=%s  var=%s" % (gtfs_id, dir_id, var_rank))
    seg_labels = [
        "Seg %d:  %s  ->  %s" % (i + 1, s.get('from_stop_name', '?'), s.get('to_stop_name', '?'))
        for i, s in enumerate(chain)
    ]
    print("\n  Which segment to split?")
    seg_idx = _pick_one(seg_labels, "Segment to split")
    if seg_idx is None:
        return seg_layers, line_layers
    target = chain[seg_idx]

    info = _pick_station_or_manual(station_catalog, "Search new intermediate stop")
    if info is None:
        return seg_layers, line_layers
    new_nr, new_name, new_e, new_n = info

    ref = target
    print("\n  TT for  %s -> %s:" % (ref.get('from_stop_name', '?'), new_name))
    tt1, src1 = _prompt_tt()
    print("  TT for  %s -> %s:" % (new_name, ref.get('to_stop_name', '?')))
    tt2, src2 = _prompt_tt()
    try:
        ivwt = max(0.1, float(input("  IVWT per segment [0.5]: ").strip() or '0.5'))
    except ValueError:
        ivwt = 0.5

    row1 = _new_seg_dict(
        gtfs_id, ref.get(SEG_NAME_COL, ''), dir_id, var_rank,
        ref.get('mode_label', ''), ref.get('mode_class', ''), ref.get('service_period', ''),
        ref.get(SEG_FROM_NR), ref.get('from_stop_name', ''),
        float(ref.get('from_stop_E', 0)), float(ref.get('from_stop_N', 0)),
        new_nr, new_name, new_e, new_n,
        tt1, src1, ivwt,
    )
    row2 = _new_seg_dict(
        gtfs_id, ref.get(SEG_NAME_COL, ''), dir_id, var_rank,
        ref.get('mode_label', ''), ref.get('mode_class', ''), ref.get('service_period', ''),
        new_nr, new_name, new_e, new_n,
        ref.get(SEG_TO_NR), ref.get('to_stop_name', ''),
        float(ref.get('to_stop_E', 0)), float(ref.get('to_stop_N', 0)),
        tt2, src2, ivwt,
    )

    seg_layers[layer_name] = gdf[gdf.index != target.name].reset_index(drop=True)
    seg_layers = _append_seg_to_layer(seg_layers, layer_name, row1)
    seg_layers = _append_seg_to_layer(seg_layers, layer_name, row2)
    line_layers = _sync_line_from_segments(
        seg_layers, line_layers, layer_name, gtfs_id, dir_id, var_rank
    )
    print("  Inserted stop '%s'. Route now has %d stops." % (new_name, len(chain) + 2))
    return seg_layers, line_layers


def _extend_route(
    seg_layers:     Dict[str, gpd.GeoDataFrame],
    line_layers:    Dict[str, gpd.GeoDataFrame],
    layer_name:     str,
    station_catalog: gpd.GeoDataFrame,
) -> Tuple[Dict[str, gpd.GeoDataFrame], Dict[str, gpd.GeoDataFrame]]:
    """Extend a route by adding stops at the head or tail."""
    if not seg_layers.get(layer_name, gpd.GeoDataFrame()).shape[0]:
        print("  No segments in this layer.")
        return seg_layers, line_layers

    result = _pick_route_variant({layer_name: seg_layers[layer_name]})
    if result is None:
        return seg_layers, line_layers
    _, gtfs_id, dir_id, var_rank = result

    chain = _build_stop_chain(seg_layers[layer_name], gtfs_id, dir_id, var_rank)
    if not chain:
        print("  No segments found.")
        return seg_layers, line_layers

    _display_stop_sequence(chain, "%s  dir=%s  var=%s" % (gtfs_id, dir_id, var_rank))
    print("\n  Extend at:")
    print("    1) Tail (add stops after the last stop)")
    print("    2) Head (add stops before the first stop)")
    while True:
        c = input("  Select (1/2): ").strip()
        if c in ('1', '2'):
            break
        print("  Enter 1 or 2.")
    extend_tail = (c == '1')

    ref = chain[-1] if extend_tail else chain[0]
    mode_label     = ref.get('mode_label', '')
    mode_class     = ref.get('mode_class', '')
    service_period = ref.get('service_period', '')
    svc_name       = ref.get(SEG_NAME_COL, '')

    if extend_tail:
        anchor_nr   = ref.get(SEG_TO_NR)
        anchor_name = ref.get('to_stop_name', '?')
        anchor_e    = float(ref.get('to_stop_E', 0))
        anchor_n    = float(ref.get('to_stop_N', 0))
    else:
        anchor_nr   = ref.get(SEG_FROM_NR)
        anchor_name = ref.get('from_stop_name', '?')
        anchor_e    = float(ref.get('from_stop_E', 0))
        anchor_n    = float(ref.get('from_stop_N', 0))

    n_added = 0
    while True:
        direction_str = "after '%s'" % anchor_name if extend_tail else "before '%s'" % anchor_name
        info = _pick_station_or_manual(station_catalog, "Next stop %s (Enter to finish)" % direction_str)
        if info is None:
            break
        new_nr, new_name, new_e, new_n = info

        prompt = "TT for %s -> %s" % (anchor_name, new_name) if extend_tail \
                 else "TT for %s -> %s" % (new_name, anchor_name)
        tt, tt_src = _prompt_tt(prompt)
        try:
            ivwt = max(0.1, float(input("  IVWT [0.5]: ").strip() or '0.5'))
        except ValueError:
            ivwt = 0.5

        if extend_tail:
            row = _new_seg_dict(
                gtfs_id, svc_name, dir_id, var_rank, mode_label, mode_class, service_period,
                anchor_nr, anchor_name, anchor_e, anchor_n,
                new_nr, new_name, new_e, new_n, tt, tt_src, ivwt,
            )
        else:
            row = _new_seg_dict(
                gtfs_id, svc_name, dir_id, var_rank, mode_label, mode_class, service_period,
                new_nr, new_name, new_e, new_n,
                anchor_nr, anchor_name, anchor_e, anchor_n,
                tt, tt_src, ivwt,
            )

        seg_layers = _append_seg_to_layer(seg_layers, layer_name, row)
        anchor_nr, anchor_name, anchor_e, anchor_n = new_nr, new_name, new_e, new_n
        n_added += 1
        print("  Added stop '%s'. Continue? (Enter to add another, 'q' to finish)" % new_name)
        if input("  ").strip().lower() == 'q':
            break

    if n_added:
        line_layers = _sync_line_from_segments(
            seg_layers, line_layers, layer_name, gtfs_id, dir_id, var_rank
        )
        print("  Added %d stop(s). Route now has %d stops." % (
            n_added, len(_build_stop_chain(seg_layers[layer_name], gtfs_id, dir_id, var_rank)) + 1
        ))
    else:
        print("  No stops added.")
    return seg_layers, line_layers


def _add_new_route(
    seg_layers:     Dict[str, gpd.GeoDataFrame],
    line_layers:    Dict[str, gpd.GeoDataFrame],
    layer_name:     str,
    station_catalog: gpd.GeoDataFrame,
) -> Tuple[Dict[str, gpd.GeoDataFrame], Dict[str, gpd.GeoDataFrame]]:
    """Build a new service route from scratch using an interactive stop sequence builder."""
    print("\n  --- New route ---")
    gtfs_id = input("  GTFS_ID / route identifier (e.g. SYNTH_001): ").strip()
    if not gtfs_id:
        print("  Cancelled.")
        return seg_layers, line_layers

    existing = seg_layers.get(layer_name, gpd.GeoDataFrame())
    if not existing.empty and (existing[SEG_ID_COL].astype(str) == str(gtfs_id)).any():
        print("  GTFS_ID '%s' already exists in this layer." % gtfs_id)
        if input("  Continue anyway? (y/n) [n]: ").strip().lower() != 'y':
            return seg_layers, line_layers

    svc_name = input("  Service name (e.g. S99): ").strip() or gtfs_id
    dir_id   = input("  direction_id [0]: ").strip() or '0'
    var_rank = input("  variant_rank [1]: ").strip() or '1'
    mode_label = input("  mode_label [%s]: " % layer_name).strip() or layer_name
    mode_class = _mode_class_for_layer(layer_name)

    period_opts = ['all_day', 'peak_only', 'offpeak_only', 'unknown']
    print("  service_period:")
    sp_idx = _pick_one(period_opts, "service_period")
    service_period = period_opts[sp_idx] if sp_idx is not None else 'all_day'

    # Build stop sequence
    print("\n  Build stop sequence (pick stops; press Enter without a stop to finish).")
    print("  You need at least 2 stops (1 segment).")
    stops: List[Tuple[int, str, float, float]] = []  # (nr, name, E, N)

    while True:
        prompt = "First stop" if not stops else "Next stop after '%s' (Enter to finish)" % stops[-1][1]
        info = _pick_station_or_manual(station_catalog, prompt)
        if info is None:
            if len(stops) < 2:
                print("  Need at least 2 stops.")
                continue
            break
        stops.append(info)
        print("  Stop %d: %s" % (len(stops), info[1]))

    if len(stops) < 2:
        print("  Cancelled — not enough stops.")
        return seg_layers, line_layers

    # Collect TTs for each segment
    new_rows = []
    for i in range(len(stops) - 1):
        f_nr, f_name, f_e, f_n = stops[i]
        t_nr, t_name, t_e, t_n = stops[i + 1]
        print("  TT for  %s -> %s:" % (f_name, t_name))
        tt, tt_src = _prompt_tt()
        try:
            ivwt = max(0.1, float(input("  IVWT [0.5]: ").strip() or '0.5'))
        except ValueError:
            ivwt = 0.5
        new_rows.append(_new_seg_dict(
            gtfs_id, svc_name, dir_id, var_rank, mode_label, mode_class, service_period,
            f_nr, f_name, f_e, f_n, t_nr, t_name, t_e, t_n, tt, tt_src, ivwt,
        ))

    for row in new_rows:
        seg_layers = _append_seg_to_layer(seg_layers, layer_name, row)

    # Frequency values for the line feature (optional)
    print("\n  Frequency values (dep/hr) for this route — press Enter to skip:")
    def _opt_float(p):
        raw = input("  %s: " % p).strip()
        try:
            return float(raw) if raw else pd.NA
        except ValueError:
            return pd.NA

    freq_am  = _opt_float("freq_am_peak_dep_hr")
    freq_pm  = _opt_float("freq_pm_peak_dep_hr")
    freq_off = _opt_float("freq_offpeak_dep_hr")

    # Build line geometry and create line feature
    chain_new = _build_stop_chain(seg_layers[layer_name], gtfs_id, dir_id, var_rank)
    geoms = [r.geometry for r in chain_new if r.geometry is not None]
    try:
        line_geom = linemerge(geoms) if geoms else None
    except Exception:
        line_geom = geoms[0] if geoms else None

    new_line = {
        LINE_ID_COL:              gtfs_id,
        LINE_NAME_COL:            svc_name,
        'direction_id':           dir_id,
        'variant_rank':           var_rank,
        'origin':                 stops[0][1],
        'destination':            stops[-1][1],
        'line_long_name':         '%s: %s - %s' % (svc_name, stops[0][1], stops[-1][1]),
        'n_stops':                len(stops),
        'mode_label':             mode_label,
        'mode_class':             mode_class,
        'service_period':         service_period,
        'freq_am_peak_dep_hr':    freq_am,
        'freq_pm_peak_dep_hr':    freq_pm,
        'freq_offpeak_dep_hr':    freq_off,
        'freq_directional':       False,
        'total_dep':              pd.NA,
        'variant_trip_share':     pd.NA,
        'geometry':               line_geom,
    }

    existing_lines = line_layers.get(layer_name, gpd.GeoDataFrame())
    new_line_gdf   = gpd.GeoDataFrame([new_line], crs=SWISS_CRS)
    if existing_lines.empty:
        line_layers[layer_name] = new_line_gdf.reset_index(drop=True)
    else:
        for col in existing_lines.columns:
            if col not in new_line_gdf.columns:
                new_line_gdf[col] = pd.NA
        line_layers[layer_name] = pd.concat(
            [existing_lines, new_line_gdf[existing_lines.columns]], ignore_index=True
        )

    print("  Created route '%s' (%s) with %d stops, %d segment(s)." % (
        svc_name, gtfs_id, len(stops), len(new_rows)))
    return seg_layers, line_layers


# =============================================================================
# Attribute editing — lines
# =============================================================================

def _pick_line_variant(
    line_layers: Dict[str, gpd.GeoDataFrame],
    prompt: str = "Select route variant",
) -> Optional[Tuple[str, str, str, str]]:
    """Search line layers and return (layer_name, route_id, direction_id, variant_rank)."""
    term = input("\n  %s — search (Enter to list all): " % prompt).strip()
    items: List[Tuple[str, str, str, str, str]] = []

    for layer_name, gdf in line_layers.items():
        if gdf.empty:
            continue
        if term:
            t = term.lower()
            mask = (
                gdf.get(LINE_ID_COL,   pd.Series(dtype=str)).fillna('').str.lower().str.contains(t, regex=False) |
                gdf.get(LINE_NAME_COL, pd.Series(dtype=str)).fillna('').str.lower().str.contains(t, regex=False)
            )
            sub = gdf[mask]
        else:
            sub = gdf
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            rid      = str(row.get(LINE_ID_COL, ''))
            lname    = str(row.get(LINE_NAME_COL, ''))
            dir_id   = str(row.get('direction_id', '0'))
            var_rank = str(row.get('variant_rank', '1'))
            origin   = str(row.get('origin', ''))
            dest     = str(row.get('destination', ''))
            label = "%s  [%s]  dir=%s  var=%s  [%s]  %s -> %s" % (
                lname, rid, dir_id, var_rank, layer_name, origin, dest)
            items.append((layer_name, rid, dir_id, var_rank, label))

    if not items:
        print("  No routes found%s." % (' matching ' + repr(term) if term else ''))
        return None
    if len(items) > 80:
        print("  %d variants found — showing first 80." % len(items))
        items = items[:80]

    idx = _pick_one([it[4] for it in items], "Route variant")
    if idx is None:
        return None
    return items[idx][0], items[idx][1], items[idx][2], items[idx][3]


def _edit_line_attributes(
    line_layers: Dict[str, gpd.GeoDataFrame],
    seg_layers:  Dict[str, gpd.GeoDataFrame],
    layer_name:  str,
) -> Tuple[Dict[str, gpd.GeoDataFrame], Dict[str, gpd.GeoDataFrame]]:
    """Field-by-field editing of line-level attributes (frequencies, service_period).

    service_period changes are synced back to all matching segment rows.
    """
    if not line_layers.get(layer_name, gpd.GeoDataFrame()).shape[0]:
        print("  No line features in this layer.")
        return line_layers, seg_layers

    result = _pick_line_variant({layer_name: line_layers[layer_name]})
    if result is None:
        return line_layers, seg_layers
    _, route_id, dir_id, var_rank = result

    line_gdf  = line_layers[layer_name]
    line_mask = (
        (line_gdf[LINE_ID_COL].astype(str)    == str(route_id)) &
        (line_gdf['direction_id'].astype(str)  == str(dir_id)) &
        (line_gdf['variant_rank'].astype(str)  == str(var_rank))
    )
    if not line_mask.any():
        print("  Line record not found.")
        return line_layers, seg_layers

    row_idx = line_gdf[line_mask].index[0]
    row     = line_gdf.loc[row_idx]

    print("\n  Editing line  %s [%s]  dir=%s  var=%s  (Enter to keep)" % (
        row.get(LINE_NAME_COL, ''), route_id, dir_id, var_rank))

    changed     = False
    sp_changed  = False
    old_sp      = str(row.get('service_period', ''))

    # Frequency fields
    for col, label in [
        ('freq_am_peak_dep_hr',  'AM peak freq (dep/hr)'),
        ('freq_pm_peak_dep_hr',  'PM peak freq (dep/hr)'),
        ('freq_offpeak_dep_hr',  'off-peak freq (dep/hr)'),
    ]:
        if col not in line_gdf.columns:
            continue
        cur = row.get(col)
        cur_str = str(cur) if (cur is not None and not (isinstance(cur, float) and pd.isna(cur))) else 'None'
        raw = input("  %s (current: %s): " % (label, cur_str)).strip()
        if raw:
            if raw.lower() in ('none', 'null', '-'):
                line_gdf.at[row_idx, col] = pd.NA
                changed = True
            else:
                try:
                    line_gdf.at[row_idx, col] = float(raw)
                    changed = True
                except ValueError:
                    print("  Invalid value — skipped.")

    # service_period
    print("  service_period (current: %s):" % old_sp)
    for i, opt in enumerate(_SERVICE_PERIOD_OPTIONS, 1):
        print("    %d) %s" % (i, opt))
    raw = input("  Select (1-%d, Enter to keep): " % len(_SERVICE_PERIOD_OPTIONS)).strip()
    if raw.isdigit() and 1 <= int(raw) <= len(_SERVICE_PERIOD_OPTIONS):
        new_sp = _SERVICE_PERIOD_OPTIONS[int(raw) - 1]
        if new_sp != old_sp:
            line_gdf.at[row_idx, 'service_period'] = new_sp
            changed   = True
            sp_changed = True

    # freq_directional
    if 'freq_directional' in line_gdf.columns:
        cur_fd = bool(row.get('freq_directional', False))
        raw = input("  freq_directional (current: %s, y/n, Enter to keep): " % cur_fd).strip().lower()
        if raw in ('y', 'yes', 'true', '1'):
            line_gdf.at[row_idx, 'freq_directional'] = True
            changed = True
        elif raw in ('n', 'no', 'false', '0'):
            line_gdf.at[row_idx, 'freq_directional'] = False
            changed = True

    line_layers[layer_name] = line_gdf

    # Sync service_period back to segment rows
    if sp_changed and layer_name in seg_layers and not seg_layers[layer_name].empty:
        seg_gdf = seg_layers[layer_name]
        seg_mask = (
            (seg_gdf[SEG_ID_COL].astype(str)    == str(route_id)) &
            (seg_gdf['direction_id'].astype(str)  == str(dir_id)) &
            (seg_gdf['variant_rank'].astype(str)  == str(var_rank))
        )
        if seg_mask.any():
            seg_gdf.loc[seg_mask, 'service_period'] = new_sp
            seg_layers[layer_name] = seg_gdf
            print("  service_period synced to %d segment row(s)." % int(seg_mask.sum()))

    if changed:
        print("  Line attributes updated.")
    else:
        print("  No changes made.")
    return line_layers, seg_layers


# =============================================================================
# Import operations
# =============================================================================

def _mark_tt_source_bulk(
    layers: Dict[str, gpd.GeoDataFrame],
) -> Dict[str, gpd.GeoDataFrame]:
    """Bulk-set tt_source (and TT for formula) for matched routes."""
    print("\n  Mark TT source — choose target:")
    print("    1) gtfs    (measured/trusted — keeps existing TT value)")
    print("    2) formula (sentinel — sets TT to NA for all matched segments)")
    src_choice = input("  Select (1/2): ").strip()
    if src_choice not in ('1', '2'):
        print("  Cancelled.")
        return layers
    target_source = 'gtfs' if src_choice == '1' else 'formula'

    term = input("  Search routes (GTFS_ID / line name / 'all'): ").strip()
    if not term:
        print("  Cancelled.")
        return layers

    total = 0
    preview: List[str] = []
    for layer_name, gdf in layers.items():
        if gdf.empty:
            continue
        if term.lower() == 'all':
            mask = pd.Series(True, index=gdf.index)
        else:
            mask = _search_svc_segments(gdf, term).index
            mask = gdf.index.isin(mask)
        n = int(mask.sum())
        if n:
            total += n
            sample_ids = gdf.loc[mask, SEG_ID_COL].dropna().unique()[:3]
            preview.append(f"  {layer_name}: {n} segment(s)  [{', '.join(str(r) for r in sample_ids)}]")

    if total == 0:
        print(f"  No segments match '{term}'.")
        return layers

    print(f"\n  Will set tt_source='{target_source}' on {total} segment(s):")
    for p in preview:
        print(p)
    if target_source == 'formula':
        print(f"  {SEG_TT_COL} will be set to NA (sentinel — projection computes it).")
    ans = input(f"  Apply? (y/n) [n]: ").strip().lower() or 'n'
    if ans != 'y':
        print("  Cancelled.")
        return layers

    for layer_name, gdf in layers.items():
        if gdf.empty:
            continue
        if term.lower() == 'all':
            mask = pd.Series(True, index=gdf.index)
        else:
            matched = _search_svc_segments(gdf, term).index
            mask = gdf.index.isin(matched)
        if not mask.any():
            continue
        if 'tt_source' not in gdf.columns:
            gdf['tt_source'] = 'gtfs'
        gdf.loc[mask, 'tt_source'] = target_source
        if target_source == 'formula' and SEG_TT_COL in gdf.columns:
            gdf.loc[mask, SEG_TT_COL] = pd.NA
        layers[layer_name] = gdf

    print(f"  Done — {total} segment(s) updated.")
    return layers


def _import_from_version(
    seg_layers:   Dict[str, gpd.GeoDataFrame],
    line_layers:  Dict[str, gpd.GeoDataFrame],
    is_rail:      bool,
    version_name: str,
) -> Tuple[Dict[str, gpd.GeoDataFrame], Dict[str, gpd.GeoDataFrame]]:
    """Import routes from another network's Unprojected/, diffing at (GTFS_ID, direction_id, variant_rank).

    Lists all available *_network folders as import sources (excluding the current one).
    """
    base_dir  = Path(paths.MAIN) / (paths.RAIL_LINES_DIR if is_rail else paths.FEEDER_LINES_DIR)
    seg_gpkg  = RAIL_SEG_GPKG   if is_rail else FEEDER_SEG_GPKG
    line_gpkg = RAIL_LINES_GPKG if is_rail else FEEDER_LINES_GPKG

    all_nets = [n for n in list_all_networks() if n != version_name]
    if not all_nets:
        print("  No other networks available to import from.")
        return seg_layers, line_layers

    print("\n  Import from network:")
    src_idx = _pick_one(all_nets, "Source network")
    if src_idx is None:
        return seg_layers, line_layers

    src_dir = base_dir / all_nets[src_idx] / paths.SERVICES_UNPROJECTED_SUBDIR

    if not (src_dir / seg_gpkg).exists():
        print("  Source segments not found: %s" % (src_dir / seg_gpkg))
        return seg_layers, line_layers

    src_seg_layers  = _load_all_layers(src_dir / seg_gpkg)
    src_line_layers = _load_all_layers(src_dir / line_gpkg) if (src_dir / line_gpkg).exists() else {}

    def _variants_in(lyrs):
        vs = set()
        for gdf in lyrs.values():
            if SEG_ID_COL in gdf.columns and 'direction_id' in gdf.columns and 'variant_rank' in gdf.columns:
                for _, r in gdf[[SEG_ID_COL, 'direction_id', 'variant_rank']].drop_duplicates().iterrows():
                    vs.add((str(r[SEG_ID_COL]), str(r['direction_id']), str(r['variant_rank'])))
        return vs

    current_v = _variants_in(seg_layers)
    source_v  = _variants_in(src_seg_layers)

    def _seg_count(lyrs, gid, did, vr):
        return sum(
            int(((g[SEG_ID_COL].astype(str) == gid) &
                 (g['direction_id'].astype(str) == did) &
                 (g['variant_rank'].astype(str) == vr)).sum())
            for g in lyrs.values() if SEG_ID_COL in g.columns
        )

    new_v     = source_v - current_v
    changed_v = {v for v in source_v & current_v
                 if _seg_count(src_seg_layers, *v) != _seg_count(seg_layers, *v)}
    removed_v = current_v - source_v

    diff_items = (
        [(v, 'New')     for v in sorted(new_v)] +
        [(v, 'Changed') for v in sorted(changed_v)] +
        [(v, 'Removed') for v in sorted(removed_v)]
    )
    if not diff_items:
        print("  No differences found between current version and source.")
        return seg_layers, line_layers

    print("\n  Available imports:")
    for i, ((gid, did, vr), status) in enumerate(diff_items, 1):
        print("    %d) [%s]  %s  dir=%s  var=%s" % (i, status, gid, did, vr))

    ans = input("\n  Enter numbers to apply (e.g. 1,3,5-10), 'all', or Enter to cancel: ").strip()
    if not ans:
        return seg_layers, line_layers
    selected = _parse_selection(ans, len(diff_items))
    if not selected:
        print("  No valid choices.")
        return seg_layers, line_layers

    for i in selected:
        (gid, did, vr), status = diff_items[i]

        # Find which source layer holds this variant
        src_layer = next((ln for ln, g in src_seg_layers.items()
                          if SEG_ID_COL in g.columns and
                          ((g[SEG_ID_COL].astype(str) == gid) &
                           (g['direction_id'].astype(str) == did) &
                           (g['variant_rank'].astype(str) == vr)).any()), None)

        # Remove from current (Changed or Removed)
        if status in ('Changed', 'Removed'):
            for ln in list(seg_layers):
                if SEG_ID_COL in seg_layers[ln].columns:
                    keep = ~((seg_layers[ln][SEG_ID_COL].astype(str) == gid) &
                             (seg_layers[ln]['direction_id'].astype(str) == did) &
                             (seg_layers[ln]['variant_rank'].astype(str) == vr))
                    seg_layers[ln] = seg_layers[ln][keep].reset_index(drop=True)
            tgt_ln = src_layer or next(iter(line_layers), None)
            if tgt_ln and tgt_ln in line_layers and LINE_ID_COL in line_layers[tgt_ln].columns:
                ldf = line_layers[tgt_ln]
                keep = ~((ldf[LINE_ID_COL].astype(str) == gid) &
                         (ldf['direction_id'].astype(str) == did) &
                         (ldf['variant_rank'].astype(str) == vr))
                line_layers[tgt_ln] = ldf[keep].reset_index(drop=True)

        # Add from source (New or Changed)
        if status in ('New', 'Changed') and src_layer:
            src_gdf = src_seg_layers[src_layer]
            mask = ((src_gdf[SEG_ID_COL].astype(str) == gid) &
                    (src_gdf['direction_id'].astype(str) == did) &
                    (src_gdf['variant_rank'].astype(str) == vr))
            to_add = src_gdf[mask]
            if not to_add.empty:
                existing = seg_layers.get(src_layer, gpd.GeoDataFrame())
                seg_layers[src_layer] = pd.concat(
                    [existing, to_add], ignore_index=True
                ) if not existing.empty else to_add.reset_index(drop=True)

            if src_layer in src_line_layers:
                src_ldf = src_line_layers[src_layer]
                if LINE_ID_COL in src_ldf.columns:
                    lmask = ((src_ldf[LINE_ID_COL].astype(str) == gid) &
                             (src_ldf['direction_id'].astype(str) == did) &
                             (src_ldf['variant_rank'].astype(str) == vr))
                    line_to_add = src_ldf[lmask]
                    if not line_to_add.empty:
                        ex_lines = line_layers.get(src_layer, gpd.GeoDataFrame())
                        if ex_lines.empty:
                            line_layers[src_layer] = line_to_add.reset_index(drop=True)
                        else:
                            for col in ex_lines.columns:
                                if col not in line_to_add.columns:
                                    line_to_add = line_to_add.copy()
                                    line_to_add[col] = pd.NA
                            line_layers[src_layer] = pd.concat(
                                [ex_lines, line_to_add[ex_lines.columns]], ignore_index=True
                            )

    print("  Applied %d change(s)." % len(selected))
    return seg_layers, line_layers


# =============================================================================
# Line derivation
# =============================================================================

def _derive_lines_from_segments(
    segments_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Build one line feature per (GTFS_ID, direction_id, variant_rank) by
    stitching segment geometries in stop-sequence order and merging via linemerge.
    Carries route-level attributes from the first segment of each group.
    """
    if segments_gdf.empty or SEG_ID_COL not in segments_gdf.columns:
        return gpd.GeoDataFrame(geometry=[], crs=SWISS_CRS)

    group_cols = [c for c in [SEG_ID_COL, 'direction_id', 'variant_rank']
                  if c in segments_gdf.columns]
    if not group_cols:
        return gpd.GeoDataFrame(geometry=[], crs=SWISS_CRS)

    lines = []
    for keys, grp in segments_gdf.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        if SEG_FROM_NR in grp.columns and SEG_TO_NR in grp.columns:
            chain = {}     # from_stop_nr → to_stop_nr
            seg_map = {}   # (from_nr, to_nr) → geometry
            for _, row in grp.iterrows():
                f_nr = row[SEG_FROM_NR]
                t_nr = row[SEG_TO_NR]
                chain[f_nr] = t_nr
                seg_map[(f_nr, t_nr)] = row.geometry

            to_stops = set(chain.values())
            starts = [f for f in chain if f not in to_stops]
            start = starts[0] if starts else next(iter(chain))

            geoms = []
            cur = start
            visited = set()
            while cur in chain and cur not in visited:
                nxt = chain[cur]
                g = seg_map.get((cur, nxt))
                if g is not None:
                    geoms.append(g)
                visited.add(cur)
                cur = nxt
        else:
            geoms = [row.geometry for _, row in grp.iterrows() if row.geometry is not None]

        if not geoms:
            continue

        try:
            merged = linemerge(geoms)
        except Exception:
            merged = geoms[0]

        first = grp.iloc[0]
        carry_cols = [
            SEG_ID_COL, 'direction_id', 'variant_rank', SEG_NAME_COL,
            'mode_label', 'mode_class', 'service_period',
            'freq_am_peak_dep_hr', 'freq_pm_peak_dep_hr', 'freq_offpeak_dep_hr',
        ]
        use_nr_chain = SEG_FROM_NR in grp.columns
        rec = {
            'n_stops': len(visited) + 1 if use_nr_chain else len(grp) + 1,
            'geometry': merged,
        }
        for col in carry_cols:
            if col in grp.columns:
                rec[col] = first.get(col, pd.NA)

        lines.append(rec)

    if not lines:
        return gpd.GeoDataFrame(geometry=[], crs=SWISS_CRS)

    return gpd.GeoDataFrame(lines, crs=SWISS_CRS)


# =============================================================================
# Main TUI helpers
# =============================================================================

def _pick_layer(
    layers:     Dict[str, gpd.GeoDataFrame],
    prompt:     str = "Choose layer",
    filter_set: Optional[Set[str]] = None,
) -> Optional[str]:
    """Pick a non-empty layer from the dict, optionally restricted to filter_set."""
    names = [ln for ln, gdf in layers.items()
             if not gdf.empty and (filter_set is None or ln in filter_set)]
    if not names:
        print("  No data available.")
        return None
    if len(names) == 1:
        return names[0]
    print("\n  %s:" % prompt)
    labels = ["%s  (%d row(s))" % (ln, len(layers[ln])) for ln in names]
    idx = _pick_one(labels, "Layer")
    return names[idx] if idx is not None else None


def _editing_loop(
    seg_layers:      Dict[str, gpd.GeoDataFrame],
    line_layers:     Dict[str, gpd.GeoDataFrame],
    station_catalog: gpd.GeoDataFrame,
    svc_version:     str,
    mode_label:      str,
    is_rail:         bool,
    layer_filter:    Optional[Set[str]] = None,
) -> Tuple[Dict[str, gpd.GeoDataFrame], Dict[str, gpd.GeoDataFrame]]:
    """Shared interactive editing loop for rail and track-based feeder services."""
    while True:
        n_segs  = sum(len(g) for g in seg_layers.values())
        n_lines = sum(len(g) for g in line_layers.values())
        print("\n" + "-" * 60)
        print("  %s  |  %d segment(s)  |  %d line feature(s)" % (mode_label, n_segs, n_lines))
        print("    1) View / search routes")
        print("    2) Remove route  (full or trim head/tail)")
        print("    3) Edit segment attributes  (TT, IVWT, service_period, ...)")
        print("    4) Edit stop sequence  (drop / insert / extend)")
        print("    5) Add new route")
        print("    6) Edit line attributes  (frequencies, service_period)")
        print("    7) Import routes from another version")
        print("    8) Bulk TT-source marking  (all layers)")
        print("    s) Done  ->")
        c = input("  Select: ").strip().lower()

        if c == 's':
            break

        elif c == '1':
            layer_name = _pick_layer(seg_layers, "View routes in which layer", layer_filter)
            if layer_name:
                result = _pick_route_variant({layer_name: seg_layers[layer_name]})
                if result:
                    _, gid, did, vr = result
                    chain = _build_stop_chain(seg_layers[layer_name], gid, did, vr)
                    _display_stop_sequence(chain, "%s  dir=%s  var=%s" % (gid, did, vr))

        elif c == '2':
            layer_name = _pick_layer(seg_layers, "Remove from which layer", layer_filter)
            if layer_name:
                print("\n  Remove options:")
                print("    1) Remove route entirely  (all or specific variant)")
                print("    2) Trim from head or tail")
                sub = input("  Select (1/2): ").strip()
                if sub == '1':
                    seg_layers, line_layers = _remove_route(seg_layers, line_layers, layer_name)
                elif sub == '2':
                    seg_layers, line_layers = _trim_route(seg_layers, line_layers, layer_name)

        elif c == '3':
            layer_name = _pick_layer(seg_layers, "Edit segments in which layer", layer_filter)
            if layer_name:
                seg_layers, line_layers = _adjust_segment(seg_layers, line_layers, layer_name)

        elif c == '4':
            layer_name = _pick_layer(seg_layers, "Edit stop sequence in which layer", layer_filter)
            if layer_name:
                print("\n  Stop sequence edit:")
                print("    1) Drop a stop  (merge two surrounding segments)")
                print("    2) Insert a stop  (split a segment)")
                print("    3) Extend  (add stops at head or tail)")
                sub = input("  Select (1/2/3): ").strip()
                if sub == '1':
                    seg_layers, line_layers = _drop_stop(seg_layers, line_layers, layer_name)
                elif sub == '2':
                    seg_layers, line_layers = _insert_stop(
                        seg_layers, line_layers, layer_name, station_catalog)
                elif sub == '3':
                    seg_layers, line_layers = _extend_route(
                        seg_layers, line_layers, layer_name, station_catalog)

        elif c == '5':
            layer_name = _pick_layer(seg_layers, "Add route to which layer", layer_filter)
            if layer_name:
                seg_layers, line_layers = _add_new_route(
                    seg_layers, line_layers, layer_name, station_catalog)

        elif c == '6':
            layer_name = _pick_layer(line_layers, "Edit line attributes in which layer", layer_filter)
            if layer_name:
                line_layers, seg_layers = _edit_line_attributes(
                    line_layers, seg_layers, layer_name)

        elif c == '7':
            seg_layers, line_layers = _import_from_version(
                seg_layers, line_layers, is_rail, svc_version)

        elif c == '8':
            seg_layers = _mark_tt_source_bulk(seg_layers)

        else:
            print("  Invalid choice.")

    return seg_layers, line_layers


# =============================================================================
# Period-subfolder propagation
# =============================================================================

_ALLDAY_PERIODS  = {'all_day'}
_PEAK_PERIODS    = {'all_day', 'peak_only'}
_OFFPEAK_PERIODS = {'all_day', 'offpeak_only'}

_PERIOD_SUBDIR_MAP = {
    'All_Day':  _ALLDAY_PERIODS,
    'Peak':     _PEAK_PERIODS,
    'Off_Peak': _OFFPEAK_PERIODS,
}

_SEG_KEY_COLS  = ('GTFS_ID', 'from_stop_nr', 'to_stop_nr')
_LINE_KEY_COL  = 'route_id'

_PERIOD_SEG_FILES = {
    'All_Day':  'rail_segments_allday.gpkg',
    'Peak':     'rail_segments_peak.gpkg',
    'Off_Peak': 'rail_segments_offpeak.gpkg',
}
_PERIOD_LINE_FILES = {
    'All_Day':  'rail_lines_allday.gpkg',
    'Peak':     'rail_lines_peak.gpkg',
    'Off_Peak': 'rail_lines_offpeak.gpkg',
}


def _propagate_to_period_folders(
    orig_rail_seg:  Dict[str, gpd.GeoDataFrame],
    new_rail_seg:   Dict[str, gpd.GeoDataFrame],
    orig_rail_line: Dict[str, gpd.GeoDataFrame],
    new_rail_line:  Dict[str, gpd.GeoDataFrame],
    unprojected_dir: Path,
) -> None:
    """Sync rail_segments and rail_lines edits into All_Day / Peak / Off_Peak subfolders.

    For each route_id that was removed or modified, rows are deleted from the
    period files that included that service period, then re-inserted from the
    updated data for modified routes.
    """
    network_dir = unprojected_dir.parent

    def _all_rows(layers: Dict[str, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
        parts = [g for g in layers.values() if not g.empty]
        return pd.concat(parts, ignore_index=True) if parts else gpd.GeoDataFrame()

    orig_segs = _all_rows(orig_rail_seg)
    new_segs  = _all_rows(new_rail_seg)
    orig_lines = _all_rows(orig_rail_line)
    new_lines  = _all_rows(new_rail_line)

    if orig_segs.empty and new_segs.empty:
        return

    id_col = SEG_ID_COL  # 'GTFS_ID'

    orig_ids = set(orig_segs[id_col].dropna().astype(str)) if id_col in orig_segs.columns else set()
    new_ids  = set(new_segs[id_col].dropna().astype(str))  if id_col in new_segs.columns  else set()

    removed_ids  = orig_ids - new_ids
    modified_ids = set()
    for rid in orig_ids & new_ids:
        o = orig_segs[orig_segs[id_col].astype(str) == rid]
        n = new_segs[new_segs[id_col].astype(str) == rid]
        if len(o) != len(n) or not o[list(_SEG_KEY_COLS)].reset_index(drop=True).equals(
                n[list(_SEG_KEY_COLS)].reset_index(drop=True)):
            modified_ids.add(rid)
        elif not o.drop(columns=['geometry'], errors='ignore').reset_index(drop=True).equals(
                 n.drop(columns=['geometry'], errors='ignore').reset_index(drop=True)):
            modified_ids.add(rid)
    added_ids = new_ids - orig_ids

    changed_ids = removed_ids | modified_ids

    if not changed_ids and not added_ids:
        print("  Period folders: no changes to propagate.")
        return

    print(f"  Propagating to period folders: "
          f"{len(removed_ids)} removed, {len(modified_ids)} modified, {len(added_ids)} added.")

    for subdir_name, period_set in _PERIOD_SUBDIR_MAP.items():
        period_dir = network_dir / subdir_name
        if not period_dir.exists():
            continue

        # -- Segments ----------------------------------------------------------
        seg_file = period_dir / _PERIOD_SEG_FILES[subdir_name]
        if seg_file.exists():
            try:
                p_seg = gpd.read_file(seg_file)
                if id_col in p_seg.columns:
                    p_seg_ids = p_seg[id_col].astype(str)
                    # Remove changed or removed routes
                    keep = ~p_seg_ids.isin(changed_ids)
                    p_seg = p_seg[keep].copy()
                    # Re-insert modified routes that belong to this period
                    if modified_ids and not new_segs.empty:
                        rows_to_add = new_segs[
                            new_segs[id_col].astype(str).isin(modified_ids) &
                            new_segs.get('service_period', pd.Series('all_day', index=new_segs.index))
                            .isin(period_set)
                        ]
                        if not rows_to_add.empty:
                            p_seg = pd.concat([p_seg, rows_to_add], ignore_index=True)
                    # Insert genuinely new routes that belong to this period
                    if added_ids and not new_segs.empty:
                        rows_to_add = new_segs[
                            new_segs[id_col].astype(str).isin(added_ids) &
                            new_segs.get('service_period', pd.Series('all_day', index=new_segs.index))
                            .isin(period_set)
                        ]
                        if not rows_to_add.empty:
                            p_seg = pd.concat([p_seg, rows_to_add], ignore_index=True)
                    if isinstance(p_seg, gpd.GeoDataFrame):
                        p_seg.to_file(str(seg_file), driver='GPKG')
                    else:
                        gpd.GeoDataFrame(p_seg).to_file(str(seg_file), driver='GPKG')
            except Exception as exc:
                print(f"  WARNING: Could not update {seg_file.name}: {exc}")

        # -- Lines -------------------------------------------------------------
        line_file = period_dir / _PERIOD_LINE_FILES[subdir_name]
        if line_file.exists() and not orig_lines.empty:
            try:
                p_line = gpd.read_file(line_file)
                lcol = _LINE_KEY_COL
                if lcol in p_line.columns:
                    orig_line_ids = set(orig_lines[lcol].dropna().astype(str)) \
                                    if lcol in orig_lines.columns else set()
                    new_line_ids  = set(new_lines[lcol].dropna().astype(str)) \
                                    if lcol in new_lines.columns else set()
                    changed_line_ids = (orig_line_ids - new_line_ids) | \
                                       {r for r in orig_line_ids & new_line_ids
                                        if not orig_lines[orig_lines[lcol].astype(str) == r]
                                        .reset_index(drop=True).equals(
                                            new_lines[new_lines[lcol].astype(str) == r]
                                            .reset_index(drop=True))}
                    added_line_ids = new_line_ids - orig_line_ids

                    keep = ~p_line[lcol].astype(str).isin(changed_line_ids)
                    p_line = p_line[keep].copy()

                    reinsert_ids = {r for r in changed_line_ids if r in new_line_ids} | added_line_ids
                    if reinsert_ids and not new_lines.empty:
                        rows_to_add = new_lines[new_lines[lcol].astype(str).isin(reinsert_ids)]
                        if not rows_to_add.empty:
                            p_line = pd.concat([p_line, rows_to_add], ignore_index=True)

                    if isinstance(p_line, gpd.GeoDataFrame):
                        p_line.to_file(str(line_file), driver='GPKG')
                    else:
                        gpd.GeoDataFrame(p_line).to_file(str(line_file), driver='GPKG')
            except Exception as exc:
                print(f"  WARNING: Could not update {line_file.name}: {exc}")

    print("  Period folders updated.")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--infra-version', default=None,
                        help='Infra version for station catalog (auto-select, no prompt)')
    parser.add_argument('--network', default=None,
                        help='Auto-select this network in adjust mode, skipping the Phase 0 menu')
    args, _ = parser.parse_known_args()

    try:
        (rail_seg, rail_line, feed_seg, feed_line,
         orig_rail_seg, orig_rail_line,
         station_catalog, version_name,
         out_rail_dir, out_feeder_dir, _) = _run_phase0(
            auto_infra_version=args.infra_version,
            auto_network=args.network,
        )
    except SystemExit:
        return

    # Phase 1 — Rail services editing
    print("\n" + "=" * 60)
    print("  RAIL SERVICES  —  network: %s" % version_name)
    rail_seg, rail_line = _editing_loop(
        rail_seg, rail_line, station_catalog, version_name,
        "Rail services", is_rail=True,
    )

    # Phase 2 — Track-based feeder (optional)
    n_feed = sum(len(g) for g in feed_seg.values() if not g.empty)
    if n_feed > 0:
        print("\n" + "=" * 60)
        ans = input("  Edit track-based feeder services (tram / metro / funicular)? (y/n) [n]: "
                    ).strip().lower() or 'n'
        if ans == 'y':
            feed_seg_tb  = {k: v for k, v in feed_seg.items()  if k in TRACK_BASED_FEEDER_LAYERS}
            feed_line_tb = {k: v for k, v in feed_line.items() if k in TRACK_BASED_FEEDER_LAYERS}
            feed_seg_tb, feed_line_tb = _editing_loop(
                feed_seg_tb, feed_line_tb, station_catalog, version_name,
                "Track-based feeder", is_rail=False,
                layer_filter=TRACK_BASED_FEEDER_LAYERS,
            )
            for ln, gdf in feed_seg_tb.items():
                feed_seg[ln] = gdf
            for ln, gdf in feed_line_tb.items():
                feed_line[ln] = gdf

    # Phase 3 — Save
    print("\n" + "=" * 60)
    print("  Saving '%s/Unprojected/' ..." % version_name)
    _save_svc_data(rail_seg, rail_line, feed_seg, feed_line, out_rail_dir, out_feeder_dir)

    # Propagate rail edits to All_Day / Peak / Off_Peak subfolders
    _propagate_to_period_folders(
        orig_rail_seg, rail_seg,
        orig_rail_line, rail_line,
        out_rail_dir,
    )
    print("  Done.")


if __name__ == '__main__':
    main()
