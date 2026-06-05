"""
Infrastructure Enhancement Module

Produces a named "enhanced" infrastructure version by combining:
  1. Travel time enrichment — distribute GTFS-observed travel times (TT_Stopping,
     TT_Passing) across BAV segments via projected service path_nodes, covering
     both rail (rail_segments.gpkg) and track-based feeder services (tram, metro,
     funicular via pt_feeder_segments.gpkg).
  2. Gauge / electrification fill — infer missing attribute values from service
     traversal paths via path-based consensus.
  3. Feeder-derived segment import — promote track-based feeder segments with ZVV
     geometry and no BAV routing path into first-class infrastructure segments,
     with matching segments_composition.gpkg entries.

The source infrastructure version is never modified. All enrichment is written to
a new named version in the same infrastructure directory.

Output column names TT_Stopping / TT_Passing match the column read by
build_infra_graph in services_service_projection.py for routing weight calibration.

Rail TT enrichment note: requires rail_segments.gpkg to contain path_nodes (retained
after the _OUTPUT_DROP fix in services_service_projection.py). If path_nodes is
absent from an existing rail_segments.gpkg, rail services are skipped and only feeder
TT is distributed. Re-run services_service_projection.py after that fix to generate
a path_nodes-aware rail output.

Usage (interactive):
    python infrabuild_infrastructure_enhancement.py
Last modified: 2026-05-07
"""

import difflib
import sys
import fiona
import geopandas as gpd
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from shapely.geometry import Point
from shapely.ops import linemerge

sys.path.insert(0, str(Path(__file__).parent))
import paths
import settings
from infrabuild_version_manager import list_versions, _pick_one


# =============================================================================
# Constants
# =============================================================================

SWISS_CRS = "EPSG:2056"
TRACK_BASED_FEEDER_LAYERS = {"tram", "funicular", "metro"}

# Physics parameters — sourced from settings.py §11 (single source of truth).
_DECEL_A = settings.SERVICE_BRAKE_DECEL_MS2
_BUFFER  = settings.TT_OPERATIONAL_BUFFER
_STATION_CLASSES = {'station'}

_MODE_DEFAULT_SPEEDS: Dict[str, float] = {
    'train':       50.0,
    'tram':        30.0,
    'funicular':   10.0,
    'cog_railway': 15.0,
    'bus':         30.0,
}
_MODE_DEFAULT_FALLBACK: float = 50.0

_FEEDER_LAYER_GAUGE: Dict[str, int] = {
    "tram":      1000,
    "funicular": 1000,
    "metro":     1000,
}
_FEEDER_LAYER_ELEC: Dict[str, str] = {
    "tram":      "Gleichstrom",
    "funicular": "Gleichstrom",
    "metro":     "Gleichstrom",
}
_FEEDER_LAYER_MODE: Dict[str, str] = {
    "tram":      "tram",
    "funicular": "funicular",
    "metro":     "tram",
}


# Tier rank for segment speed_source values.
# 'design' (3) is a locked tier — never updated by service contributions.
_SPEED_SOURCE_TIER: Dict[str, int] = {
    'estimate': 0,
    'formula':  1,
    'gtfs':     2,
    'design':   3,
}

# Tier rank for service link tt_source values.
_TT_SOURCE_TIER: Dict[str, int] = {
    'estimate': 0,
    'formula':  1,
    'gtfs':     2,
}


# =============================================================================
# Helpers — TT arithmetic
# =============================================================================

def _round_half_min(x: float) -> float:
    """Round to nearest 0.1 minutes, floor 0.1 min."""
    return max(0.1, round(x * 10) / 10)


def _seg_expected_tt(seg_row: pd.Series) -> float:
    """Expected traversal time in minutes based on segment length and speed.

    Speed cascade: Average_Speed (length-weighted OSM mean — harmonically
    correct for traversal time) → Predominant_Speed (most-common bin) → mode
    default → fallback. Must match the cascade in step [4] of compute_segment_stats
    and the routing graph build in services_service_projection.py.
    """
    length_m = float(seg_row.get('Length', 0) or 0)
    speed = seg_row.get('Average_Speed')
    if pd.isna(speed) or float(speed) <= 0:
        speed = seg_row.get('Predominant_Speed')
    if pd.isna(speed) or float(speed) <= 0:
        mode = str(seg_row.get('Transport_Mode', '')).strip()
        for m in mode.split('/'):
            s = _MODE_DEFAULT_SPEEDS.get(m.strip())
            if s:
                speed = s
                break
        else:
            speed = _MODE_DEFAULT_FALLBACK
    speed = float(speed)
    if speed <= 0:
        return 0.0
    return (length_m / 1000.0) / speed * 60.0


def _is_boundary_bpnr(v) -> bool:
    """True iff v is a parseable BPNR set by Phase 1.5 boundary rerouting.

    Robust to GPKG round-trip artefacts: pd.NA written to a mixed-type object
    column may come back as None, NaN, '', 'nan', or '<NA>' — pd.notna() treats
    the string forms as truthy and would incorrectly mark every rail link as
    boundary-rerouted. Requiring the value to parse as an integer (a real
    Betriebspunkt-Nummer) eliminates the false positives.
    """
    if v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    s = str(v).strip()
    if not s or s.lower() in ('nan', 'none', '<na>'):
        return False
    try:
        int(float(s))
        return True
    except (ValueError, TypeError):
        return False


# =============================================================================
# Helpers — version discovery
# =============================================================================

def list_svc_versions() -> List[str]:
    """Return svc_version names that have an Unprojected/ feeder base."""
    root = Path(paths.MAIN) / paths.FEEDER_LINES_DIR
    if not root.exists():
        return []
    return [
        d.name for d in sorted(root.iterdir())
        if d.is_dir()
        and (d / paths.SERVICES_UNPROJECTED_SUBDIR / "pt_feeder_segments.gpkg").exists()
    ]


def _list_projection_sources(svc_version: str, infra_version: str) -> List[Tuple[str, Path, Path]]:
    """
    Return available projected source directories for (svc_version, infra_version).
    Each entry: (label, feeder_dir, rail_dir)
    """
    _SKIP = {
        paths.SERVICES_UNPROJECTED_SUBDIR,
        paths.SERVICES_PROJECTED_SUBDIR,
    }
    feeder_root = Path(paths.MAIN) / paths.FEEDER_LINES_DIR / svc_version
    rail_root   = Path(paths.MAIN) / paths.RAIL_LINES_DIR   / svc_version

    options = []
    proj_feeder = feeder_root / infra_version
    proj_rail   = rail_root   / infra_version
    if proj_feeder.exists() or proj_rail.exists():
        options.append((infra_version, proj_feeder, proj_rail))

    if feeder_root.exists():
        for d in sorted(feeder_root.iterdir()):
            if d.is_dir() and d.name not in _SKIP and d.name != infra_version:
                options.append((d.name, d, rail_root / d.name))

    return options


# =============================================================================
# Helpers — enhanced version name validation and directory management
# =============================================================================

def _validate_enhanced_name(name: str) -> Optional[str]:
    """
    Return an error message if name is invalid, None if valid.

    Rules: non-empty, no Raw prefix, no path separators.
    Base prefix is explicitly allowed.
    """
    if not name:
        return "Name cannot be empty."
    if name.startswith('Raw'):
        return f"'{name}' starts with reserved prefix 'Raw'. Choose another name."
    if '/' in name or '\\' in name:
        return "Name must not contain path separators."
    return None


def _check_prior_run(enhanced_dir: Path) -> int:
    """
    Return count of segments with non-null TT_Stopping in an existing enhanced version.
    Returns 0 if the file or column does not exist.
    """
    seg_path = enhanced_dir / 'segments.gpkg'
    if not seg_path.exists():
        return 0
    try:
        layers = fiona.listlayers(str(seg_path))
        if not layers:
            return 0
        gdf = gpd.read_file(seg_path, layer=layers[0])
        if 'TT_Stopping' not in gdf.columns:
            return 0
        return int(gdf['TT_Stopping'].notna().sum())
    except Exception:
        return 0


def _create_enhanced_dir(enhanced_dir: Path) -> None:
    """Create the enhanced directory.

    nodes.gpkg is written later by compute_segment_stats after the catchment
    filter and any junction reimport, so no source copy is needed here.
    """
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Ready: {enhanced_dir}")


def _load_catchment_buffer():
    """Load the catchment-area buffer geometry, or None if absent."""
    buf_path = Path(paths.MAIN) / paths.CATCHMENT_AREA_BUFFER_GPKG
    if not buf_path.exists():
        return None
    buf_gdf = gpd.read_file(buf_path)
    if buf_gdf.empty:
        return None
    return buf_gdf.geometry.union_all()


def _find_raw_nodes_path(infra_version: str) -> Optional[Path]:
    """Locate the raw infrastructure's nodes.gpkg.

    infrabuild_filter_network.py writes to a study-area-suffixed folder
    (e.g. 'Raw_ZH'), or plain 'Raw' when no suffix is given. Try in order:

      1. Raw_<last suffix of infra_version>  (e.g. AS_2026_ZH → Raw_ZH)
      2. Plain Raw/
      3. First Raw*/ folder alphabetically that contains nodes.gpkg

    Returns None if nothing matches.
    """
    infra_root = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR
    if not infra_root.exists():
        return None

    parts = infra_version.split('_')
    if len(parts) > 1:
        candidate = infra_root / f"Raw_{parts[-1]}" / 'nodes.gpkg'
        if candidate.exists():
            return candidate

    plain = infra_root / "Raw" / 'nodes.gpkg'
    if plain.exists():
        return plain

    for d in sorted(infra_root.iterdir()):
        if d.is_dir() and d.name.startswith('Raw'):
            cand = d / 'nodes.gpkg'
            if cand.exists():
                return cand
    return None


def _filter_to_catchment(
    segments: gpd.GeoDataFrame,
    nodes_gdf: gpd.GeoDataFrame,
    buffer_geom,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Filter segments and nodes using the same rules as infrabuild_filter_network.py
    (filter_to_catchment), so the enhanced version has the same spatial extent as
    its source infrastructure version.

    Segments: kept if at least one endpoint (From_E/N, To_E/N) lies within the
              buffer. Cross-boundary segments are kept with full unclipped geometry.
    Nodes:    nodes strictly inside the buffer, plus any outside-boundary node that
              is an endpoint of a kept cross-boundary segment.
    """
    # --- Nodes strictly inside ---------------------------------------------------
    inside_mask = nodes_gdf.geometry.apply(
        lambda g: g is not None and not g.is_empty and g.within(buffer_geom)
    )
    print(f"  Nodes inside buffer: {inside_mask.sum()} / {len(nodes_gdf)}")

    # --- Segments: at least one endpoint inside ----------------------------------
    def _at_least_one_in(row) -> bool:
        try:
            fE, fN = float(row['From_E']), float(row['From_N'])
            tE, tN = float(row['To_E']),   float(row['To_N'])
        except (KeyError, ValueError, TypeError):
            return False
        if any(pd.isna(v) for v in (fE, fN, tE, tN)):
            return False
        return Point(fE, fN).within(buffer_geom) or Point(tE, tN).within(buffer_geom)

    seg_mask = segments.apply(_at_least_one_in, axis=1)
    filtered_segs = segments[seg_mask].reset_index(drop=True)
    print(f"  Segments kept: {seg_mask.sum()} / {len(segments)}")

    # --- Include outside endpoint nodes from cross-boundary segments -------------
    # Enhancement segments carry From_Name/To_Name; use name-based lookup to find
    # which outside nodes need to be re-added (mirrors filter_network's
    # outside_endpoint_ids logic, adapted from Node_ID to Name).
    inside_names: set = set(nodes_gdf.loc[inside_mask, 'Name'].dropna().astype(str))
    name_set: set = set(nodes_gdf['Name'].dropna().astype(str))
    outside_endpoint_names: set = set()
    for _, seg in filtered_segs.iterrows():
        for col in ('From_Name', 'To_Name'):
            name = str(seg.get(col, '') or '')
            if name and name not in inside_names and name in name_set:
                outside_endpoint_names.add(name)

    outside_nodes = nodes_gdf[nodes_gdf['Name'].isin(outside_endpoint_names)]
    print(f"  Outside endpoint nodes included: {len(outside_nodes)}")

    filtered_nodes = pd.concat(
        [nodes_gdf[inside_mask], outside_nodes]
    ).drop_duplicates(subset=['Number']).reset_index(drop=True)

    return filtered_segs, filtered_nodes


def _import_bpnr_referenced_junctions(
    segments: gpd.GeoDataFrame,
    nodes_source_gdf: gpd.GeoDataFrame,
    enhanced_dir: Path,
    infra_version: str,
    buffer_geom=None,
    min_segments: int = 2,
) -> None:
    """Import BAV nodes referenced in segment Numbers but absent from the source nodes.

    A BPNR is a candidate when it appears in the Number column of >= min_segments
    distinct segments AND is absent from nodes_source_gdf. The min_segments >= 2
    guard ensures only true intermediate junctions are imported — a BPNR that
    appears in a single segment Number is a leaf endpoint, already handled by
    the catchment filter's outside-endpoint logic or present in nodes_source_gdf
    as a station.

    Imported nodes receive Node_Class='junction'. Raw geometry is filtered by
    buffer_geom when provided.
    """
    # Count distinct segments per BPNR across all parseable Number values
    bpnr_seg_refs: Dict[int, set] = {}
    for _, row in segments.iterrows():
        sid = row.get('Segment_ID')
        if not sid:
            continue
        number = str(row.get('Number', '') or '')
        if '_' not in number:
            continue
        parts = number.split('_', 1)
        try:
            fi, ti = int(parts[0]), int(parts[1])
        except (ValueError, TypeError):
            continue
        bpnr_seg_refs.setdefault(fi, set()).add(sid)
        bpnr_seg_refs.setdefault(ti, set()).add(sid)

    candidate_bpnrs = {
        bpnr for bpnr, sids in bpnr_seg_refs.items()
        if len(sids) >= min_segments
    }

    source_bpnrs: set = set()
    for _, nrow in nodes_source_gdf.iterrows():
        b = nrow.get('Number')
        if pd.notna(b):
            try:
                source_bpnrs.add(int(float(b)))
            except (ValueError, TypeError):
                pass

    missing_bpnrs = candidate_bpnrs - source_bpnrs
    if not missing_bpnrs:
        print(f"  BPNR scan: no missing junction nodes detected.")
        return

    raw_path = _find_raw_nodes_path(infra_version)
    if raw_path is None:
        print(f"  BPNR scan: Raw nodes not found — "
              f"{len(missing_bpnrs)} missing BPNR(s) cannot be resolved.")
        return

    raw_nodes_gdf = gpd.read_file(raw_path)
    raw_bpnr_to_row: Dict[int, pd.Series] = {}
    skipped_outside = 0
    for _, nrow in raw_nodes_gdf.iterrows():
        bpnr = nrow.get('Number')
        if not pd.notna(bpnr):
            continue
        if buffer_geom is not None:
            geom = nrow.geometry
            if geom is None or geom.is_empty or not geom.within(buffer_geom):
                skipped_outside += 1
                continue
        try:
            raw_bpnr_to_row[int(float(bpnr))] = nrow
        except (ValueError, TypeError):
            pass

    to_add = sorted(missing_bpnrs & set(raw_bpnr_to_row.keys()))
    if not to_add:
        outside = len(missing_bpnrs) - len(missing_bpnrs & set(raw_bpnr_to_row.keys()))
        print(f"  BPNR scan: {len(missing_bpnrs)} missing BPNR(s) detected, "
              f"none resolvable from Raw ({skipped_outside} outside catchment).")
        return

    enhanced_nodes_path = enhanced_dir / 'nodes.gpkg'
    enhanced_nodes_gdf = gpd.read_file(enhanced_nodes_path)
    if 'Node_Class' not in enhanced_nodes_gdf.columns:
        enhanced_nodes_gdf['Node_Class'] = None

    existing_bpnrs: set = set()
    for _, nrow in enhanced_nodes_gdf.iterrows():
        b = nrow.get('Number')
        if pd.notna(b):
            try:
                existing_bpnrs.add(int(float(b)))
            except (ValueError, TypeError):
                pass

    new_bpnrs = [bp for bp in to_add if bp not in existing_bpnrs]
    if not new_bpnrs:
        print(f"  BPNR scan: all {len(to_add)} candidate(s) already present in enhanced nodes.")
        return

    target_cols = list(enhanced_nodes_gdf.columns)
    new_rows = []
    for bp_int in new_bpnrs:
        raw_row = raw_bpnr_to_row[bp_int]
        row_dict = {
            col: (raw_row[col] if col in raw_row.index else None)
            for col in target_cols
        }
        row_dict['Node_Class'] = 'junction'
        new_rows.append(row_dict)

    additions = gpd.GeoDataFrame(new_rows, crs=enhanced_nodes_gdf.crs)
    combined = pd.concat([enhanced_nodes_gdf, additions], ignore_index=True)
    combined.to_file(enhanced_nodes_path, driver='GPKG')
    if skipped_outside:
        print(f"  BPNR scan: {skipped_outside} raw node(s) outside catchment — excluded.")
    print(f"  BPNR scan: imported {len(new_bpnrs)} missing junction(s) "
          f"(referenced by ≥{min_segments} segments, absent from source nodes).")


def _import_missing_junction_nodes(
    still_unresolved: List[Dict],
    name_to_bpnr: Dict[str, int],
    seg_by_nodes: Dict[Tuple[int, int], str],
    enhanced_dir: Path,
    infra_version: str,
    buffer_geom=None,
) -> List[Dict]:
    """
    Resolve still-unresolved segment endpoints via Raw/nodes.gpkg and persist
    the newly-matched junction nodes into enhanced_dir/nodes.gpkg with
    Node_Class='junction'.

    The Base/AS_* infrastructure versions correctly drop nodes that aren't
    macro-consolidatable (operational yards, degree-3+ junctions like Winterthur
    Nord). When segments retain those names, they fail the kept-nodes lookup.
    Pulling them back from Raw/ as classified junctions makes the enhanced
    version's nodes.gpkg self-contained — segments resolvable, future edits
    can reference these nodes directly.

    The raw folder is discovered dynamically (Raw_<suffix> matching the chosen
    infra_version, e.g. AS_2026_ZH → Raw_ZH). When buffer_geom is provided,
    only raw nodes whose Point lies within the catchment buffer are eligible
    for import.

    Mutates seg_by_nodes in place for resolved segments. Returns the list of
    segments that remain unresolved after this pass.
    """
    raw_path = _find_raw_nodes_path(infra_version)
    if raw_path is None:
        print(f"  Raw nodes not available — {len(still_unresolved)} segment(s) "
              f"remain unresolved.")
        return still_unresolved
    print(f"  Raw nodes source: {raw_path}")

    raw_nodes_gdf = gpd.read_file(raw_path)
    raw_name_to_bpnr: Dict[str, int] = {}
    raw_bpnr_to_row: Dict[int, pd.Series] = {}
    skipped_outside = 0
    for _, nrow in raw_nodes_gdf.iterrows():
        bpnr = nrow.get('Number')
        name = nrow.get('Name')
        if not (pd.notna(bpnr) and pd.notna(name)):
            continue
        # Apply catchment filter to raw candidates if a buffer is provided
        if buffer_geom is not None:
            geom = nrow.geometry
            if geom is None or geom.is_empty or not geom.within(buffer_geom):
                skipped_outside += 1
                continue
        try:
            bp_int = int(float(bpnr))
            raw_name_to_bpnr[str(name)] = bp_int
            raw_bpnr_to_row[bp_int] = nrow
        except (ValueError, TypeError):
            pass
    if skipped_outside:
        print(f"  Raw fallback: {skipped_outside} node(s) outside catchment ignored.")

    bpnrs_to_import: set = set()
    new_still: List[Dict] = []
    resolved_via_raw = 0

    for u in still_unresolved:
        fn_name = u.get('From_Name')
        tn_name = u.get('To_Name')

        fi = name_to_bpnr.get(str(fn_name)) if fn_name else None
        if fi is None and fn_name:
            fi = raw_name_to_bpnr.get(str(fn_name))
            if fi is not None:
                bpnrs_to_import.add(fi)

        ti = name_to_bpnr.get(str(tn_name)) if tn_name else None
        if ti is None and tn_name:
            ti = raw_name_to_bpnr.get(str(tn_name))
            if ti is not None:
                bpnrs_to_import.add(ti)

        if fi is not None and ti is not None:
            sid = u['Segment_ID']
            seg_by_nodes[(fi, ti)] = sid
            seg_by_nodes[(ti, fi)] = sid
            resolved_via_raw += 1
        else:
            new_still.append({
                **u,
                'From_in': 'OK' if fi is not None else 'MISSING',
                'To_in':   'OK' if ti is not None else 'MISSING',
            })

    if resolved_via_raw:
        print(f"  Resolved {resolved_via_raw} additional segment(s) via raw-node fallback.")

    # Persist imported junction nodes into the enhanced version's nodes.gpkg
    if bpnrs_to_import:
        enhanced_nodes_path = enhanced_dir / 'nodes.gpkg'
        if enhanced_nodes_path.exists():
            enhanced_nodes_gdf = gpd.read_file(enhanced_nodes_path)
            if 'Node_Class' not in enhanced_nodes_gdf.columns:
                enhanced_nodes_gdf['Node_Class'] = None

            existing_bpnrs: set = set()
            for _, nrow in enhanced_nodes_gdf.iterrows():
                b = nrow.get('Number')
                if pd.notna(b):
                    try:
                        existing_bpnrs.add(int(float(b)))
                    except (ValueError, TypeError):
                        pass

            new_to_add = sorted(bpnrs_to_import - existing_bpnrs)
            if new_to_add:
                target_cols = list(enhanced_nodes_gdf.columns)
                new_rows = []
                for bp_int in new_to_add:
                    raw_row = raw_bpnr_to_row[bp_int]
                    row_dict = {
                        col: (raw_row[col] if col in raw_row.index else None)
                        for col in target_cols
                    }
                    row_dict['Node_Class'] = 'junction'
                    new_rows.append(row_dict)

                additions = gpd.GeoDataFrame(new_rows, crs=enhanced_nodes_gdf.crs)
                combined = pd.concat([enhanced_nodes_gdf, additions], ignore_index=True)
                combined.to_file(enhanced_nodes_path, driver='GPKG')
                print(f"  Imported {len(new_to_add)} junction node(s) "
                      f"(Node_Class='junction') into enhanced nodes.gpkg.")

    return new_still


# =============================================================================
# Phase 0 — CLI
# =============================================================================

def _run_phase0():
    """
    Interactive CLI to select infra version, svc_version, projection source,
    and enhanced version name.

    Returns
    -------
    infra_version, svc_version, feeder_source_dir, rail_source_dir,
    infra_dir, enhanced_dir, extend_mode
    """
    infra_root = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR

    # Q1: infrastructure version. All versions are selectable; _enhanced versions
    # trigger extend mode (soft prompt + in-place update) instead of being blocked.
    infra_versions = list_versions()
    if not infra_versions:
        print("\n  ERROR: No infrastructure versions found.")
        print("  Run infrabuild_network_builder.py first.")
        raise SystemExit(1)

    print("\n" + "─" * 60)
    print("  Q1: Choose infrastructure version:")
    idx = _pick_one(infra_versions, "Infra version")
    if idx is None:
        raise SystemExit(0)
    infra_version = infra_versions[idx]
    infra_dir = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR / infra_version

    # Detect extend mode from the _enhanced suffix.
    extend_mode = infra_version.lower().endswith('_enhanced')
    if extend_mode:
        enhanced_dir = infra_dir

    # Q2: svc_version
    svc_versions = list_svc_versions()
    if not svc_versions:
        print("\n  ERROR: No svc_versions found.")
        raise SystemExit(1)

    print("\n" + "─" * 60)
    print("  Q2: Choose service version (svc_version):")
    idx2 = _pick_one(svc_versions, "svc_version")
    if idx2 is None:
        raise SystemExit(0)
    svc_version = svc_versions[idx2]

    # Projected source: must be the projection produced against the chosen
    # infra_version. No selection prompt — the pairing is fixed.
    sources = _list_projection_sources(svc_version, infra_version)
    matching_idx = next(
        (i for i, (label, _, _) in enumerate(sources) if label == infra_version),
        None,
    )
    if matching_idx is None:
        print(f"\n  ERROR: No projected source found for "
              f"infra='{infra_version}', svc='{svc_version}'.")
        print(f"  Run services_service_projection.py first against this "
              f"infrastructure version.")
        raise SystemExit(1)

    print("\n" + "─" * 60)
    print(f"  Using projected source: {sources[matching_idx][0]}")
    _, feeder_source_dir, rail_source_dir = sources[matching_idx]

    if not extend_mode:
        # Q3: enhanced version name (skipped in extend mode — output is in-place)
        base_name = infra_version
        while base_name.lower().endswith('_enhanced'):
            base_name = base_name[:-len('_enhanced')]
        default_name = f"{base_name}_enhanced"
        print("\n" + "─" * 60)
        print("  Q3: Name for the enhanced infrastructure version")
        print(f"      (Base prefix allowed; Raw prefix reserved)")

        while True:
            raw = input(f"  Name [{default_name}]: ").strip()
            name = raw or default_name

            err = _validate_enhanced_name(name)
            if err:
                print(f"  {err}")
                continue

            enhanced_dir = infra_root / name

            if enhanced_dir.exists() and (enhanced_dir / 'segments.gpkg').exists():
                prior_count = _check_prior_run(enhanced_dir)
                if prior_count > 0:
                    print(f"\n  This enhanced version already has TT_Stopping populated "
                          f"for {prior_count} segment(s) from a previous run.")
                    ans = input("  Re-run and overwrite? (y/n) [n]: ").strip().lower() or 'n'
                    if ans != 'y':
                        raise SystemExit(0)
                else:
                    ans = input(f"  Version '{name}' already exists. Overwrite? (y/n) [n]: "
                                ).strip().lower() or 'n'
                    if ans != 'y':
                        continue
            break

        _create_enhanced_dir(enhanced_dir)

    return (infra_version, svc_version,
            feeder_source_dir, rail_source_dir,
            infra_dir, enhanced_dir, extend_mode)


# =============================================================================
# Attribute fill and feeder-derived segment creation
# =============================================================================

def _fill_missing_infra_attrs(
    segments: gpd.GeoDataFrame,
    service_links_gdf: gpd.GeoDataFrame,
    seg_by_nodes: Dict,
) -> gpd.GeoDataFrame:
    """
    Fill NULL gauge and electrification in infra segments using the actual gauge
    and electrification values of other segments traversed in the same service
    link path (path-based inference from real BAV data).

    For each service link: collect known gauge/elec values from segments in the
    path that already have those values populated, then attribute the consensus
    to the NULL segments in the same path. Skips when values conflict across the
    path. Only fills cells that are currently NULL/NaN; never overwrites.
    """
    seg_lookup: Dict[str, pd.Series] = {
        str(row['Segment_ID']): row
        for _, row in segments.iterrows()
        if row.get('Segment_ID')
    }

    gauge_inferences: Dict[str, List[int]] = defaultdict(list)
    elec_inferences:  Dict[str, List[str]] = defaultdict(list)

    for _, row in service_links_gdf.iterrows():
        # Skip boundary-rerouted links — same rationale as in step [4].
        if _is_boundary_bpnr(row.get('boundary_entry_node')) or \
                _is_boundary_bpnr(row.get('boundary_exit_node')):
            continue

        path_str = str(row.get('path_nodes', '')).strip()
        if not path_str:
            continue
        try:
            node_ids = [int(n.strip()) for n in path_str.split(';') if n.strip()]
        except ValueError:
            continue
        if len(node_ids) < 2:
            continue

        path_sids = [
            seg_by_nodes.get((u, v)) or seg_by_nodes.get((v, u))
            for u, v in zip(node_ids[:-1], node_ids[1:])
        ]

        known_gauges: List[int] = []
        known_elecs:  List[str] = []
        null_sids: List[str] = []

        for sid in path_sids:
            if sid is None:
                continue
            seg_row = seg_lookup.get(str(sid))
            if seg_row is None:
                continue
            g = seg_row.get('Gauge')
            e = seg_row.get('Electrification_Class')
            g_null = g is None or (isinstance(g, float) and pd.isna(g))
            e_null = not e or (isinstance(e, float) and pd.isna(e))
            if not g_null:
                known_gauges.append(int(g))
            if not e_null:
                known_elecs.append(str(e).strip())
            if g_null or e_null:
                null_sids.append(str(sid))

        if null_sids:
            unique_g = set(known_gauges)
            unique_e = set(known_elecs)
            for sid in null_sids:
                if len(unique_g) == 1:
                    gauge_inferences[sid].append(next(iter(unique_g)))
                if len(unique_e) == 1:
                    elec_inferences[sid].append(next(iter(unique_e)))

    n_gauge = 0
    n_elec  = 0
    for idx, row in segments.iterrows():
        sid = str(row.get('Segment_ID', ''))
        if not sid:
            continue

        cur_gauge = row.get('Gauge')
        if (cur_gauge is None or (isinstance(cur_gauge, float) and pd.isna(cur_gauge))) \
                and sid in gauge_inferences:
            vals = gauge_inferences[sid]
            if vals and len(set(vals)) == 1:
                segments.at[idx, 'Gauge'] = vals[0]
                n_gauge += 1

        cur_elec = row.get('Electrification_Class')
        if (not cur_elec or (isinstance(cur_elec, float) and pd.isna(cur_elec))) \
                and sid in elec_inferences:
            vals = elec_inferences[sid]
            if vals and len(set(vals)) == 1:
                segments.at[idx, 'Electrification_Class'] = vals[0]
                n_elec += 1

    print(f"  Filled: {n_gauge} gauge, {n_elec} electrification values (path-based inference)")
    return segments


def _build_feeder_derived_segments(
    feeder_all_gdf: gpd.GeoDataFrame,
    infra_dir: Path,
    seg_by_nodes: Dict,
    segments: gpd.GeoDataFrame,
) -> Optional[gpd.GeoDataFrame]:
    """
    Build new infra segment records from projected feeder segments that:
      - are in a track-based layer (tram / funicular / metro)
      - have ZVV geometry (zvv_source=True) but no BAV routing (path_nodes empty)
      - have both node_id_from and node_id_to matched to valid BAV nodes
      - do not already exist as an infra segment in seg_by_nodes

    Returns a GeoDataFrame of new segments ready to concat with segments.gpkg,
    or None if nothing qualifies.
    """
    if feeder_all_gdf is None or feeder_all_gdf.empty:
        return None
    if 'zvv_source' not in feeder_all_gdf.columns:
        print("  No zvv_source column in feeder data — no derived segments added.")
        return None

    nodes_path = infra_dir / "nodes.gpkg"
    if not nodes_path.exists():
        print("  WARNING: nodes.gpkg not found — cannot build feeder-derived segments.")
        return None
    nodes_gdf = gpd.read_file(nodes_path)
    bpnr_to_info: Dict[int, Dict] = {}
    for _, nrow in nodes_gdf.iterrows():
        bpnr = nrow.get('Number')
        if pd.notna(bpnr):
            try:
                bpnr_to_info[int(float(bpnr))] = {
                    'name': str(nrow.get('Name', '')),
                    'E': float(nrow.get('E', 0)),
                    'N': float(nrow.get('N', 0)),
                }
            except (ValueError, TypeError):
                pass

    existing_ids = set(segments['Segment_ID'].dropna().astype(str))

    mask_zvv = feeder_all_gdf['zvv_source'].fillna(False).astype(bool)
    mask_no_path = (
        feeder_all_gdf['path_nodes'].isna() |
        (feeder_all_gdf['path_nodes'].astype(str).str.strip() == '')
    ) if 'path_nodes' in feeder_all_gdf.columns else pd.Series(True, index=feeder_all_gdf.index)
    candidates = feeder_all_gdf[mask_zvv & mask_no_path].copy()

    if candidates.empty:
        print("  No feeder segments with ZVV geometry and empty path_nodes found.")
        return None

    # Pre-build mean TT_Stopping per node-pair from candidate rows.
    # These segments are stop-to-stop with ZVV geometry, so both endpoints are
    # stops — TT goes to TT_Stopping. Normalise to (min, max) to handle both
    # directions. travel_time_min is already renamed from 'TT' by the caller.
    tt_accumulator: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    if 'travel_time_min' in candidates.columns:
        for _, crow in candidates.iterrows():
            try:
                nf_ = int(float(crow['node_id_from']))
                nt_ = int(float(crow['node_id_to']))
            except (ValueError, TypeError):
                continue
            tt = crow.get('travel_time_min')
            if tt is not None and not pd.isna(tt):
                tt_accumulator[(min(nf_, nt_), max(nf_, nt_))].append(float(tt))
    tt_by_pair: Dict[Tuple[int, int], float] = {
        k: _round_half_min(sum(v) / len(v))
        for k, v in tt_accumulator.items()
    }

    new_rows = []
    seen_pairs: set = set()

    for _, row in candidates.iterrows():
        try:
            nf = int(float(row['node_id_from']))
            nt = int(float(row['node_id_to']))
        except (ValueError, TypeError, KeyError):
            continue

        pair = (min(nf, nt), max(nf, nt))
        if pair in seen_pairs:
            continue
        if seg_by_nodes.get((nf, nt)) or seg_by_nodes.get((nt, nf)):
            continue

        from_info = bpnr_to_info.get(nf)
        to_info   = bpnr_to_info.get(nt)
        if from_info is None or to_info is None:
            continue

        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        seen_pairs.add(pair)
        tt_stopping = tt_by_pair.get(pair, float('nan'))
        layer = str(row.get('_layer', 'tram'))
        seg_id = f"feeder_{nf}_{nt}"
        if seg_id in existing_ids:
            seg_id = f"feeder_{nf}_{nt}_b"

        merged = linemerge(geom) if geom.geom_type == 'MultiLineString' else geom
        if merged.geom_type == 'MultiLineString':
            first_coord = list(merged.geoms[0].coords)[0]
            last_coord  = list(merged.geoms[-1].coords)[-1]
        else:
            coords_list = list(merged.coords)
            first_coord = coords_list[0]
            last_coord  = coords_list[-1]
        new_rows.append({
            'Segment_ID':            seg_id,
            'Number':                f"{nf}_{nt}",
            'Code':                  f"feeder_{nf}_{nt}",
            'From_Name':             from_info['name'],
            'To_Name':               to_info['name'],
            'From_E':                first_coord[0],
            'From_N':                first_coord[1],
            'To_E':                  last_coord[0],
            'To_N':                  last_coord[1],
            'Length':                geom.length,
            'Num_Tracks':            2,
            'Gauge':                 _FEEDER_LAYER_GAUGE.get(layer),
            'Electrification_Class': _FEEDER_LAYER_ELEC.get(layer),
            'Transport_Mode':        _FEEDER_LAYER_MODE.get(layer, layer),
            'Km_Start':              float('nan'),
            'Km_End':                float('nan'),
            'Route_Number':          None,
            'Route_Name':            None,
            'Route_Owner':           None,
            'Average_Speed':         float('nan'),
            'Predominant_Speed':     float('nan'),
            'Speed_Coverage_Pct':    float('nan'),
            'TT_Stopping':           tt_stopping,
            'TT_Passing':            float('nan'),
            'geometry':              geom,
        })

    if not new_rows:
        print("  No qualifying feeder-derived segments to add.")
        return None

    derived = gpd.GeoDataFrame(new_rows, crs=SWISS_CRS)
    print(f"  {len(derived)} feeder-derived segment(s) built "
          f"({', '.join(sorted(set(r['Transport_Mode'] for r in new_rows)))})")
    return derived


def _build_feeder_composition(derived: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Build one normal composition piece per feeder-derived segment."""
    rows = []
    for _, row in derived.iterrows():
        rows.append({
            'Segment_ID':            row['Segment_ID'],
            'From_Name':             row['From_Name'],
            'To_Name':               row['To_Name'],
            'Engineering_Structure': 'normal',
            'Edge_Level':            1,
            'Under_Construction':    0,
            'Piece_Length':          row.geometry.length,
            'geometry':              row.geometry,
        })
    return gpd.GeoDataFrame(rows, crs=SWISS_CRS)


# =============================================================================
# Helpers — TT validation
# =============================================================================

def _analyse_tt_approximation(
    segments: gpd.GeoDataFrame,
    n_sta_endpoints: Dict[str, int],
    stop_counts: Dict[str, int],
    pass_counts: Dict[str, int],
    a: float = _DECEL_A,
    buffer: float = _BUFFER,
    ineligible_sids: Optional[set] = None,
) -> None:
    """
    Print two validation analyses for TT quality, junction-aware.

    (a) Internal consistency: effective-speed comparison for segments with both
        TT_Stopping and TT_Passing. Filtered to n_sta >= 1 — jct-jct segments
        have stop = pass by construction so they carry no signal here.

    (b) Formula calibration: GTFS / formula-predicted ratios bucketed by n_sta.
          TT_Stopping predicted = (L/v + n_sta · 0.5 · v/a) · buffer / 60
          TT_Passing  predicted = (L/v)                   · buffer / 60

        Three weighted means alongside the unweighted median/mean:
          wL = length-weighted    — aggregate routing-impact bias
          wC = count-weighted     — sample-quality (services per segment)
          wT = TT-weighted        — operational service-time at stake

        Segments listed in ineligible_sids are excluded from (b) — these were
        contributed-to by 'user' or other non-GTFS sources, which would
        contaminate the GTFS-vs-formula calibration signal.

        Must run before Average_Speed is overwritten by the back-calc in [5d].
    """
    skip = ineligible_sids or set()
    # ── (a) Internal consistency — effective speed comparison ────────────────
    # Restrict to segments where BOTH columns came from GTFS contributions.
    # Mixed-source comparisons (e.g., GTFS stopping vs formula passing) are
    # apples-to-oranges and produce spurious "violations" — particularly on
    # tram segments where the rail-calibrated formula underpredicts vs real
    # GTFS times.
    v_stops: List[float] = []
    v_passes: List[float] = []
    for _, row in segments.iterrows():
        sid = row.get('Segment_ID')
        if not sid or n_sta_endpoints.get(sid, 0) < 1:
            continue
        if stop_counts.get(sid, 0) <= 0 or pass_counts.get(sid, 0) <= 0:
            continue
        tt_stop = row.get('TT_Stopping')
        tt_pass = row.get('TT_Passing')
        length_m = float(row.get('Length', 0) or 0)
        if pd.isna(tt_stop) or pd.isna(tt_pass) or length_m <= 0:
            continue
        if float(tt_stop) <= 0 or float(tt_pass) <= 0:
            continue
        v_stops.append(length_m / (float(tt_stop) * 60.0) * 3.6)
        v_passes.append(length_m / (float(tt_pass) * 60.0) * 3.6)

    n_both = len(v_stops)
    print(f"\n  (a) Internal consistency — n_sta>=1 GTFS-on-both segments: {n_both}")
    if n_both:
        vs_s = pd.Series(v_stops)
        vp_s = pd.Series(v_passes)
        n_higher = int((vp_s >= vs_s).sum())
        ratios = vp_s / vs_s
        print(f"      Effective speed (TT_Stopping): median={vs_s.median():.1f} km/h, "
              f"mean={vs_s.mean():.1f} km/h")
        print(f"      Effective speed (TT_Passing) : median={vp_s.median():.1f} km/h, "
              f"mean={vp_s.mean():.1f} km/h")
        print(f"      v_pass >= v_stop: {n_higher}/{n_both} ({100 * n_higher / n_both:.1f}%)")
        print(f"      Ratio v_pass/v_stop: median={ratios.median():.2f}, "
              f"mean={ratios.mean():.2f}")

    # ── (b) Formula calibration bucketed by n_sta ────────────────────────────
    # Each bucket entry: dict with ratio, length, count, tt_total. tt_total =
    # ratio's underlying TT × count (≈ total service-link-minutes carried),
    # used for TT-weighting. count=0 for mirrored jct-jct values gives them
    # zero weight in count- and TT-weighted means but keeps them in the
    # length-weighted and unweighted summaries.
    # Only segments with at least one GTFS contribution carry calibration signal.
    # Segments with their TT_Stopping/TT_Passing preserved from the builder formula
    # (no GTFS distributed onto them) compare formula-vs-formula trivially and add
    # floor-clip noise on short segments — exclude them.
    stop_buckets: Dict[int, List[Dict]] = {0: [], 1: [], 2: []}
    pass_buckets: Dict[int, List[Dict]] = {0: [], 1: [], 2: []}
    n_skipped = 0
    for _, row in segments.iterrows():
        length_m = float(row.get('Length', 0) or 0)
        if length_m <= 0:
            continue
        spd = row.get('Average_Speed')
        if pd.isna(spd) or float(spd) <= 0:
            continue
        v_ms = float(spd) / 3.6
        sid = row.get('Segment_ID')
        if sid in skip:
            n_skipped += 1
            continue
        n_sta = n_sta_endpoints.get(sid, 0) if sid else 0

        cnt_stop = stop_counts.get(sid, 0) if sid else 0
        cnt_pass = pass_counts.get(sid, 0) if sid else 0

        tt_stop = row.get('TT_Stopping')
        if not pd.isna(tt_stop) and cnt_stop > 0:
            pred = (length_m / v_ms + n_sta * 0.5 * v_ms / a) * buffer / 60.0
            if pred > 0:
                stop_buckets[n_sta].append({
                    'ratio':    float(tt_stop) / pred,
                    'length':   length_m,
                    'count':    cnt_stop,
                    'tt_total': float(tt_stop) * cnt_stop,
                })

        tt_pass = row.get('TT_Passing')
        if not pd.isna(tt_pass) and cnt_pass > 0:
            pred = (length_m / v_ms) * buffer / 60.0
            if pred > 0:
                pass_buckets[n_sta].append({
                    'ratio':    float(tt_pass) / pred,
                    'length':   length_m,
                    'count':    cnt_pass,
                    'tt_total': float(tt_pass) * cnt_pass,
                })

    def _wmean(vals: List[Dict], wkey: str) -> float:
        w_total = sum(d[wkey] for d in vals)
        if w_total <= 0:
            return float('nan')
        return sum(d['ratio'] * d[wkey] for d in vals) / w_total

    def _print_bucket(label: str, n_sta: int, data: List[Dict]) -> None:
        if not data:
            print(f"      {label} n_sta={n_sta}: no segments.")
            return
        s = pd.Series([d['ratio'] for d in data])
        wL = _wmean(data, 'length')
        wC = _wmean(data, 'count')
        wT = _wmean(data, 'tt_total')
        gt1 = 100 * (s > 1).sum() / len(s)
        print(f"      {label} n_sta={n_sta} (N={len(s)}): "
              f"med={s.median():.2f}, mean={s.mean():.2f}, "
              f"wL={wL:.2f}, wC={wC:.2f}, wT={wT:.2f}, "
              f"std={s.std():.2f}, GTFS>formula: {gt1:.1f}%")

    print(f"\n  (b) Formula calibration (GTFS / formula-predicted, OSM Average_Speed)")
    print(f"      Weights: wL=length, wC=GTFS-count, wT=service-time-carried")
    if n_skipped:
        print(f"      Excluded {n_skipped} segment(s) with non-GTFS contributions.")
    for n_sta in (0, 1, 2):
        _print_bucket("TT_Stopping", n_sta, stop_buckets[n_sta])
    for n_sta in (0, 1, 2):
        _print_bucket("TT_Passing ", n_sta, pass_buckets[n_sta])


# =============================================================================
# Core computation
# =============================================================================

def compute_segment_stats(
    infra_version: str,
    svc_version: str,
    feeder_source_dir: Path,
    rail_source_dir: Path,
    infra_dir: Path,
    enhanced_dir: Path,
    extend_mode: bool = False,
) -> Tuple[gpd.GeoDataFrame, Optional[gpd.GeoDataFrame]]:
    """
    Returns (enriched_base_segments, feeder_derived_segments).

    enriched_base_segments: source segments with TT_Stopping, TT_Passing,
        and inferred gauge/electrification columns populated.
        TT_Stopping matches the column name read by build_infra_graph in
        services_service_projection.py for routing weight calibration.
    feeder_derived_segments: new infra segments built from track-based feeder
        data with ZVV geometry and no BAV path. None if none qualify.

    Rail TT is read from rail_segments.gpkg (column 'TT', normalised to
    'travel_time_min'). Feeder TT is read from pt_feeder_segments.gpkg
    (column 'travel_time_min'). Both use path_nodes for decomposition.
    Rail TT requires rail_segments.gpkg to have been generated after the
    _OUTPUT_DROP fix in services_service_projection.py.

    Args:
        infra_version: Name of the infrastructure version.
        svc_version: Name of the service version.
        feeder_source_dir: Directory containing pt_feeder_segments.gpkg.
        rail_source_dir: Directory containing rail_segments.gpkg.
        infra_dir: Directory of the source infrastructure version.
        enhanced_dir: Output directory for the enhanced version (already created
            by Phase 0 with a copy of nodes.gpkg). Used to persist any junction
            nodes imported from Raw/ to resolve unresolved segment endpoints.
        extend_mode: When True, existing non-null TT_Stopping / TT_Passing /
            Average_Speed values are preserved; only null slots are filled.
            Triggered automatically when the source version name ends in '_enhanced'.

    Returns:
        (enriched_base_segments, feeder_derived_segments)
    """
    # ── 1. Load infrastructure ────────────────────────────────────────────────
    print(f"\n[1] Loading infrastructure '{infra_version}'...")
    segments = gpd.read_file(infra_dir / "segments.gpkg").reset_index(drop=True)
    nodes_source_gdf = gpd.read_file(infra_dir / "nodes.gpkg")
    if 'speed_source' not in segments.columns:
        segments['speed_source'] = 'formula'
    print(f"  {len(segments)} segments, {len(nodes_source_gdf)} nodes loaded.")

    # Catchment-area filter — same rules as infrabuild_filter_network.py:
    # segments kept if at least one endpoint is inside; cross-boundary endpoint
    # nodes re-added so the spatial extent matches the source infrastructure version.
    buffer_geom = _load_catchment_buffer()
    if buffer_geom is not None:
        segments, nodes_source_gdf = _filter_to_catchment(
            segments, nodes_source_gdf, buffer_geom
        )
        print(f"  → {len(segments)} segments, {len(nodes_source_gdf)} nodes after filter.")
    else:
        print(f"  Catchment buffer not available — using full infrastructure.")

    # Persist nodes (filtered or unfiltered) to the enhanced version's nodes.gpkg.
    # _import_bpnr_referenced_junctions and _import_missing_junction_nodes will
    # append additional junctions to this file.
    nodes_source_gdf.to_file(enhanced_dir / "nodes.gpkg", driver="GPKG")

    # ── 1b. BPNR scan — import missing intermediate junctions ────────────────
    # Any BPNR referenced in ≥2 distinct segment Numbers but absent from the
    # source nodes is a macro-consolidated-away junction that the routing graph
    # needs. Import it from Raw as Node_Class='junction' so the enhanced version
    # is self-consistent without relying on build_infra_graph's healer.
    print("\n[1b] Scanning for missing junction nodes via BPNR references...")
    _import_bpnr_referenced_junctions(
        segments, nodes_source_gdf, enhanced_dir, infra_version,
        buffer_geom=buffer_geom,
    )

    seg_by_nodes: Dict[Tuple[int, int], str] = {}
    for _, row in segments.iterrows():
        sid = row.get('Segment_ID')
        if not sid:
            continue
        # Primary: parse Number column ("bpnr_from_bpnr_to"). Present on all
        # segments produced by infrabuild_filter_network and infrabuild_network_builder,
        # and on feeder-derived segments (Number = f"{nf}_{nt}").
        number = str(row.get('Number', '') or '')
        if '_' in number:
            parts = number.split('_', 1)
            try:
                fi, ti = int(parts[0]), int(parts[1])
                seg_by_nodes[(fi, ti)] = sid
                seg_by_nodes[(ti, fi)] = sid
                continue
            except (ValueError, TypeError):
                pass
        # Number absent or unparseable — resolved by name-based lookup below.

    # Supplement with name-based lookup for any segments not yet resolved.
    # Fires only when Number is absent or invalid (edge cases).
    resolved_sids = set(seg_by_nodes.values())
    unresolved = segments[~segments['Segment_ID'].isin(resolved_sids)]
    if not unresolved.empty and (infra_dir / "nodes.gpkg").exists():
        nodes_gdf = gpd.read_file(infra_dir / "nodes.gpkg")
        name_to_bpnr: Dict[str, int] = {}
        for _, nrow in nodes_gdf.iterrows():
            bpnr = nrow.get('Number')
            name = nrow.get('Name')
            if bpnr is not None and name is not None and pd.notna(bpnr) and pd.notna(name):
                try:
                    name_to_bpnr[str(name)] = int(float(bpnr))
                except (ValueError, TypeError):
                    pass
        still_unresolved: List[Dict] = []
        for _, row in unresolved.iterrows():
            sid = row.get('Segment_ID')
            if not sid:
                continue
            fn_name = row.get('From_Name')
            tn_name = row.get('To_Name')
            fi = name_to_bpnr.get(str(fn_name)) if fn_name else None
            ti = name_to_bpnr.get(str(tn_name)) if tn_name else None
            if fi is not None and ti is not None:
                seg_by_nodes[(fi, ti)] = sid
                seg_by_nodes[(ti, fi)] = sid
            else:
                still_unresolved.append({
                    'Segment_ID': sid,
                    'From_Name':  fn_name,
                    'To_Name':    tn_name,
                    'From_in':    'OK' if fi is not None else 'MISSING',
                    'To_in':      'OK' if ti is not None else 'MISSING',
                })

        # Raw-node fallback: resolve via Raw_*/nodes.gpkg and persist matches
        # as junctions in the enhanced version's nodes.gpkg. Imports are filtered
        # by the catchment buffer to keep the enhanced network in-catchment.
        if still_unresolved:
            still_unresolved = _import_missing_junction_nodes(
                still_unresolved, name_to_bpnr, seg_by_nodes, enhanced_dir,
                infra_version=infra_version,
                buffer_geom=buffer_geom,
            )

        if still_unresolved:
            print(f"  {len(still_unresolved)} segment(s) unresolved (name not found in nodes.gpkg):")
            all_names = list(name_to_bpnr.keys())
            print(f"    {'Segment_ID':<22} {'From_Name':<28} {'To_Name':<28} "
                  f"{'From':<8} {'To':<8} Suggestions (cutoff 0.7)")
            for u in still_unresolved:
                suggestions: List[str] = []
                if u['From_in'] == 'MISSING' and u['From_Name']:
                    close = difflib.get_close_matches(
                        str(u['From_Name']), all_names, n=1, cutoff=0.7
                    )
                    if close:
                        suggestions.append(f"From → '{close[0]}'")
                if u['To_in'] == 'MISSING' and u['To_Name']:
                    close = difflib.get_close_matches(
                        str(u['To_Name']), all_names, n=1, cutoff=0.7
                    )
                    if close:
                        suggestions.append(f"To → '{close[0]}'")
                print(f"    {str(u['Segment_ID'])[:22]:<22} "
                      f"{str(u['From_Name'])[:28]:<28} "
                      f"{str(u['To_Name'])[:28]:<28} "
                      f"{u['From_in']:<8} {u['To_in']:<8} "
                      f"{'; '.join(suggestions)}")

    print(f"  {len(seg_by_nodes) // 2} segment node-pairs indexed.")

    # ── Node-class lookup (post junction-import) ─────────────────────────────
    # Re-read enhanced_dir/nodes.gpkg so any junctions appended by the Raw
    # fallback above are included with Node_Class='junction'. Anything not
    # present in this file (or not classified as a station) is treated as a
    # junction by the calibration policy — macro consolidation drops exactly
    # the non-station endpoints (junctions, yards, technical points).
    final_nodes_gdf = gpd.read_file(enhanced_dir / "nodes.gpkg")
    bpnr_to_class: Dict[int, str] = {}
    if 'Number' in final_nodes_gdf.columns:
        for _, nrow in final_nodes_gdf.iterrows():
            b = nrow.get('Number')
            if pd.notna(b):
                try:
                    cls = nrow.get('Node_Class') if 'Node_Class' in final_nodes_gdf.columns else None
                    bpnr_to_class[int(float(b))] = (
                        str(cls).strip() if cls and not (isinstance(cls, float) and pd.isna(cls)) else ''
                    )
                except (ValueError, TypeError):
                    pass

    def _is_sta(bpnr) -> int:
        return 1 if bpnr_to_class.get(bpnr, '') in _STATION_CLASSES else 0

    sid_to_endpoints: Dict[str, Tuple[int, int]] = {}
    for (u, v), sid in seg_by_nodes.items():
        if sid not in sid_to_endpoints:
            sid_to_endpoints[sid] = (u, v)

    n_sta_endpoints: Dict[str, int] = {
        sid: _is_sta(u) + _is_sta(v)
        for sid, (u, v) in sid_to_endpoints.items()
    }
    n_jj = sum(1 for v in n_sta_endpoints.values() if v == 0)
    n_sj = sum(1 for v in n_sta_endpoints.values() if v == 1)
    n_ss = sum(1 for v in n_sta_endpoints.values() if v == 2)
    print(f"  Endpoint geometry: jct-jct={n_jj}, sta-jct={n_sj}, sta-sta={n_ss}")

    # ── Per-segment cruise time and per-decel-event additive time ────────────
    # expected_tt(service, seg) [min] = cruise[seg] + n_decel(svc, seg) * decel_unit[seg]
    # n_decel counts segment endpoints that are BOTH stations AND stops of the
    # service. Speed cascade: Average_Speed (length-weighted OSM mean,
    # harmonically correct for traversal time) → Predominant_Speed (most-common
    # bin) → mode default → fallback. Must match _seg_expected_tt and the
    # routing graph build in services_service_projection.py to preserve the
    # self-consistency invariant (sentinel TT computed at routing ≡ this
    # distribution).
    seg_cruise_tt:     Dict[str, float] = {}
    seg_decel_unit_tt: Dict[str, float] = {}
    for _, row in segments.iterrows():
        sid = row.get('Segment_ID')
        if not sid:
            continue
        length_m = float(row.get('Length', 0) or 0)
        speed = row.get('Average_Speed')
        if pd.isna(speed) or float(speed) <= 0:
            speed = row.get('Predominant_Speed')
        if pd.isna(speed) or float(speed) <= 0:
            mode = str(row.get('Transport_Mode', '')).strip()
            for m in mode.split('/'):
                s = _MODE_DEFAULT_SPEEDS.get(m.strip())
                if s:
                    speed = s
                    break
            else:
                speed = _MODE_DEFAULT_FALLBACK
        speed = float(speed)
        if speed <= 0:
            seg_cruise_tt[sid]     = 0.0
            seg_decel_unit_tt[sid] = 0.0
            continue
        v_ms = speed / 3.6
        seg_cruise_tt[sid]     = (length_m / v_ms) / 60.0
        seg_decel_unit_tt[sid] = (0.5 * v_ms / _DECEL_A) / 60.0

    # ── 2. Load projected service data ───────────────────────────────────────
    print("\n[2] Loading projected service data...")

    service_link_frames = []
    feeder_all_frames   = []

    # Rail: rail_segments.gpkg
    rail_gpkg = rail_source_dir / "rail_segments.gpkg"
    if rail_gpkg.exists():
        try:
            rail_layers = fiona.listlayers(str(rail_gpkg))
            for lname in rail_layers:
                gdf = gpd.read_file(rail_gpkg, layer=lname)
                gdf['_layer'] = lname
                # Normalise TT column: rail output uses 'TT', feeder uses 'travel_time_min'
                if 'TT' in gdf.columns and 'travel_time_min' not in gdf.columns:
                    gdf = gdf.rename(columns={'TT': 'travel_time_min'})
                if 'path_nodes' in gdf.columns:
                    service_link_frames.append(gdf)
                    print(f"  Rail layer '{lname}': {len(gdf)} rows")
                else:
                    print(f"  Rail layer '{lname}': skipped (no path_nodes — "
                          f"re-run services_service_projection.py to populate)")
        except Exception as e:
            print(f"  WARNING: Could not read {rail_gpkg}: {e}")
    else:
        print(f"  WARNING: Rail gpkg not found: {rail_gpkg}")

    # Feeder: pt_feeder_segments.gpkg — track-based layers only
    feeder_gpkg = feeder_source_dir / "pt_feeder_segments.gpkg"
    if feeder_gpkg.exists():
        try:
            feeder_layers = fiona.listlayers(str(feeder_gpkg))
            for lname in feeder_layers:
                if lname in TRACK_BASED_FEEDER_LAYERS:
                    gdf = gpd.read_file(feeder_gpkg, layer=lname)
                    gdf['_layer'] = lname
                    # Normalise TT column: network builder writes 'TT', not 'travel_time_min'
                    if 'TT' in gdf.columns and 'travel_time_min' not in gdf.columns:
                        gdf = gdf.rename(columns={'TT': 'travel_time_min'})
                    feeder_all_frames.append(gdf)
                    if 'path_nodes' in gdf.columns:
                        service_link_frames.append(gdf)
                    print(f"  Feeder layer '{lname}': {len(gdf)} rows")
        except Exception as e:
            print(f"  WARNING: Could not read {feeder_gpkg}: {e}")
    else:
        print(f"  WARNING: Feeder gpkg not found: {feeder_gpkg}")

    if not service_link_frames:
        print("  No service links with path_nodes found. Exiting.")
        raise SystemExit(1)

    feeder_all_gdf = (
        pd.concat(feeder_all_frames, ignore_index=True)
        if feeder_all_frames else None
    )

    service_links_gdf = pd.concat(service_link_frames, ignore_index=True)
    service_links_gdf = service_links_gdf[
        service_links_gdf['path_nodes'].notna() &
        (service_links_gdf['path_nodes'].astype(str).str.strip() != '')
    ].reset_index(drop=True)
    print(f"  {len(service_links_gdf)} service links with path_nodes after filtering.")

    # ── 3. Build stop set ────────────────────────────────────────────────────
    # The first and last node in path_nodes are the matched BAV nodes for the
    # FROM and TO stops of each service link — same information as node_id_from/to
    # without requiring those columns to be stored on service links.
    print("\n[3] Building stop set from path_nodes endpoints...")
    stop_ids: set = set()
    for _, svc_row in service_links_gdf.iterrows():
        pstr = str(svc_row.get('path_nodes', '')).strip()
        if not pstr:
            continue
        try:
            pnodes = [int(n.strip()) for n in pstr.split(';') if n.strip()]
            if pnodes:
                stop_ids.add(pnodes[0])
                stop_ids.add(pnodes[-1])
        except (ValueError, TypeError):
            pass
    print(f"  {len(stop_ids)} unique stop node IDs.")

    # ── 4. Decompose path_nodes and collect contributions ────────────────────
    # Four-tier source system (see _SPEED_SOURCE_TIER / _TT_SOURCE_TIER constants):
    #   estimate(0) < formula(1) < gtfs(2) < design(locked)
    #
    # Acceptance rules:
    #   'estimate' / 'formula' service links → skipped at outer loop
    #   'design' infra segment → locked, never updated
    #   'gtfs' service, 'gtfs' infra → accept (mean of GTFS observations)
    #   'gtfs' service, 'formula' infra → accept (GTFS calibrates formula-derived)
    #   'gtfs' service, 'estimate' infra → accept
    print(f"\n[4] Distributing travel times across infrastructure segments...")
    stopping_contributions: Dict[str, List[float]] = defaultdict(list)
    passing_contributions:  Dict[str, List[float]] = defaultdict(list)

    # Per-segment speed_source lookup (built once for O(1) access in the loop).
    # GPKG round-trip may produce NULL/NaN/'' — fall back to 'formula' (safe mid-tier).
    def _parse_speed_source(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return 'formula'
        s = str(v).strip()
        return s if s in _SPEED_SOURCE_TIER else 'formula'

    sid_to_speed_source: Dict[str, str] = {
        str(row['Segment_ID']): _parse_speed_source(row.get('speed_source'))
        for _, row in segments.iterrows()
        if row.get('Segment_ID')
    }

    skipped_no_path  = 0
    skipped_no_tt    = 0
    skipped_no_dist  = 0
    skipped_boundary = 0
    skipped_estimate = 0
    source_counts    = {'gtfs': 0, 'formula': 0, 'estimate': 0, 'other': 0}
    used_links = 0

    for link_row_idx, row in service_links_gdf.iterrows():
        # Classify service link tt_source; skip estimate and formula sources.
        raw_src = row.get('tt_source')
        if raw_src is None or (isinstance(raw_src, float) and pd.isna(raw_src)):
            tt_source = 'gtfs'
        else:
            tt_source = str(raw_src).strip().lower() or 'gtfs'
        if tt_source in source_counts:
            source_counts[tt_source] += 1
        else:
            source_counts['other'] += 1
        # 'estimate' and 'formula' service links never contribute to infra calibration.
        if tt_source in ('estimate', 'formula'):
            skipped_estimate += 1
            continue

        # Skip boundary-rerouted service links: their path_nodes covers only
        # the in-catchment portion, but travel_time_min retains the full GTFS
        # time including the outside portion (e.g., IC1 Zürich→Bern keeps the
        # ~60 min total even after path is truncated to Zürich→Mellingen
        # Heitersberg). Including these inflates passing TT on boundary segments.
        # _is_boundary_bpnr is robust to the GPKG NA-roundtrip artefact —
        # plain pd.notna() incorrectly flags '' / '<NA>' / 'nan' string forms.
        if _is_boundary_bpnr(row.get('boundary_entry_node')) or \
                _is_boundary_bpnr(row.get('boundary_exit_node')):
            skipped_boundary += 1
            continue

        path_str = str(row.get('path_nodes', '')).strip()
        if not path_str:
            skipped_no_path += 1
            continue

        try:
            node_ids = [int(n.strip()) for n in path_str.split(';') if n.strip()]
        except ValueError:
            skipped_no_path += 1
            continue

        if len(node_ids) < 2:
            skipped_no_path += 1
            continue

        tt_raw = row.get('travel_time_min')
        if tt_raw is None or (isinstance(tt_raw, float) and pd.isna(tt_raw)):
            skipped_no_tt += 1
            continue
        travel_time = float(tt_raw)

        # Service stops: first and last node in path_nodes are the matched BAV
        # nodes for the FROM and TO stops of this service link.
        service_stops = {node_ids[0], node_ids[-1]}

        pairs = list(zip(node_ids[:-1], node_ids[1:]))

        # Per-(service, segment) physics-aware expected_tt:
        #   cruise[seg] + n_decel(service, seg) * decel_unit[seg]
        # n_decel counts segment endpoints that are BOTH stations AND stops of
        # this service. Junctions and stations the service skips contribute 0.
        seg_ids: List[Optional[str]] = []
        expected_times: List[float] = []
        for u, v in pairs:
            sid = seg_by_nodes.get((u, v)) or seg_by_nodes.get((v, u))
            seg_ids.append(sid)
            if sid is None:
                expected_times.append(0.0)
                continue
            n_decel = (
                (1 if (_is_sta(u) and u in service_stops) else 0)
                + (1 if (_is_sta(v) and v in service_stops) else 0)
            )
            expected_times.append(
                seg_cruise_tt.get(sid, 0.0)
                + n_decel * seg_decel_unit_tt.get(sid, 0.0)
            )

        total_expected = sum(expected_times)

        if total_expected <= 0:
            skipped_no_dist += 1
            continue

        # Column attribution: "passing" iff any intermediate node in path_nodes
        # is a scheduled stop in some service (= service skips at least one
        # station between its consecutive stops). Per-segment physics is
        # already handled by n_decel above; this label only decides which
        # column receives each segment's contribution.
        intermediate = node_ids[1:-1]
        is_passing_link = any(n in stop_ids for n in intermediate)

        used_links += 1
        for sid, exp_tt in zip(seg_ids, expected_times):
            if sid is None or exp_tt <= 0:
                continue
            prop_tt = _round_half_min(travel_time * exp_tt / total_expected)

            seg_ss   = sid_to_speed_source.get(sid, 'formula')
            seg_tier = _SPEED_SOURCE_TIER.get(seg_ss, 1)
            svc_tier = _TT_SOURCE_TIER.get(tt_source, 1)

            # Rule 1: 'design' segment is locked — never updated by service data.
            if seg_tier == _SPEED_SOURCE_TIER['design']:
                continue

            # Rule 2: service tier must be >= infra tier.
            # Exception: gtfs == gtfs → accept (mean of observations).
            if svc_tier < seg_tier:
                continue

            # Accept contribution.
            if is_passing_link:
                passing_contributions[sid].append(prop_tt)
            else:
                stopping_contributions[sid].append(prop_tt)

    print(f"  Used {used_links} service links.")
    print(f"  Sources: gtfs={source_counts['gtfs']}, formula={source_counts['formula']}, "
          f"estimate={source_counts['estimate']}, other={source_counts['other']}")
    print(f"  Skipped: {skipped_no_path} (no path), {skipped_no_tt} (no travel time), "
          f"{skipped_no_dist} (zero expected distance), "
          f"{skipped_boundary} (boundary-rerouted), "
          f"{skipped_estimate} (estimate/formula-sourced).")

    # ── 5. Compute means ─────────────────────────────────────────────────────
    # Ensure TT columns exist — builder-produced Base versions carry them, but
    # defensive creation avoids KeyError if a version predates that convention.
    for _col in ('TT_Stopping', 'TT_Passing'):
        if _col not in segments.columns:
            segments[_col] = pd.NA

    print("\n[5] Computing per-segment means...")

    def _mean_or_nan(vals: List[float]) -> float:
        if not vals:
            return float('nan')
        return _round_half_min(sum(vals) / len(vals))

    new_stop = segments['Segment_ID'].map(
        lambda sid: _mean_or_nan(stopping_contributions.get(sid, []))
    )
    new_pass = segments['Segment_ID'].map(
        lambda sid: _mean_or_nan(passing_contributions.get(sid, []))
    )

    # Capture null state BEFORE assignment so [5c] can identify newly-calibrated
    # segments (NULL → value this run) for its diagnostic subset.
    null_before_stop = segments['TT_Stopping'].isna()
    null_before_pass = segments['TT_Passing'].isna()

    if extend_mode:
        # Extend only: fillna leaves every non-null cell untouched.
        segments['TT_Stopping'] = segments['TT_Stopping'].fillna(new_stop)
        segments['TT_Passing']  = segments['TT_Passing'].fillna(new_pass)
    else:
        segments['TT_Stopping'] = new_stop.where(new_stop.notna(), segments.get('TT_Stopping'))
        segments['TT_Passing']  = new_pass.where(new_pass.notna(),  segments.get('TT_Passing'))

    # Mask of segments that transitioned NULL → value this run (used by [5c]).
    newly_calibrated_mask = null_before_stop & segments['TT_Stopping'].notna()

    n_stopping_tot = int(segments['TT_Stopping'].notna().sum())
    n_passing_tot  = int(segments['TT_Passing'].notna().sum())

    if extend_mode:
        n_stop_locked   = int((~null_before_stop).sum())
        n_stop_extended = int(newly_calibrated_mask.sum())
        n_pass_extended = int((null_before_pass & segments['TT_Passing'].notna()).sum())
        print(f"  TT_Stopping: {n_stop_locked} locked (preserved), "
              f"{n_stop_extended} newly filled, "
              f"{n_stopping_tot}/{len(segments)} total")
        print(f"  TT_Passing : {n_pass_extended} newly filled, "
              f"{n_passing_tot}/{len(segments)} total")
    else:
        n_stopping_new = int(new_stop.notna().sum())
        n_passing_new  = int(new_pass.notna().sum())
        print(f"  TT_Stopping: {n_stopping_new}/{len(segments)} from GTFS+user, "
              f"{n_stopping_tot}/{len(segments)} populated total (rest = builder formula)")
        print(f"  TT_Passing : {n_passing_new}/{len(segments)} from GTFS+user, "
              f"{n_passing_tot}/{len(segments)} populated total (rest = builder formula)")

    # ── 5*. Junction-junction mirror fill ────────────────────────────────────
    # On jct-jct segments (n_sta=0) no service stops or accelerates anywhere on
    # the segment — TT_Stopping and TT_Passing are physically identical. If
    # only one population traverses the segment, mirror its value into the
    # empty column to satisfy the "both populated" invariant.
    n_mirrored = 0
    for idx, row in segments.iterrows():
        sid = row.get('Segment_ID')
        if not sid or n_sta_endpoints.get(sid, 0) != 0:
            continue
        ts = row.get('TT_Stopping')
        tp = row.get('TT_Passing')
        ts_null = ts is None or (isinstance(ts, float) and pd.isna(ts))
        tp_null = tp is None or (isinstance(tp, float) and pd.isna(tp))
        if not ts_null and tp_null:
            segments.at[idx, 'TT_Passing'] = ts
            n_mirrored += 1
        elif ts_null and not tp_null:
            segments.at[idx, 'TT_Stopping'] = tp
            n_mirrored += 1
    if n_mirrored:
        print(f"  Junction-junction mirror fill: {n_mirrored} segment(s).")

    # ── 5a. Fill NULL gauge / electrification ────────────────────────────────
    print("\n[5a] Filling missing gauge and electrification...")
    segments = _fill_missing_infra_attrs(segments, service_links_gdf, seg_by_nodes)

    # ── 5b. Build feeder-derived infrastructure segments ─────────────────────
    # Built before validation/correction so they also receive the speed
    # correction in [5d]. Their TT_Stopping is pre-computed from feeder rows
    # inside _build_feeder_derived_segments.
    print("\n[5b] Building feeder-derived infrastructure segments...")
    derived = _build_feeder_derived_segments(feeder_all_gdf, infra_dir, seg_by_nodes, segments)
    if derived is not None and not derived.empty:
        print(f"  {len(derived)} feeder-derived segment(s) ready.")

    # ── 5c. TT approximation validation (before speed correction) ────────────
    # Per-segment GTFS contribution counts — used as weights in [5c].
    stop_counts: Dict[str, int] = {sid: len(vals) for sid, vals in stopping_contributions.items()}
    pass_counts: Dict[str, int] = {sid: len(vals) for sid, vals in passing_contributions.items()}

    print("\n[5c] TT approximation validation...")
    if extend_mode:
        n_locked   = int((~null_before_stop).sum())
        n_extended = int(newly_calibrated_mask.sum())
        print(f"  Extend mode — formula calibration skipped for locked segments.")
        print(f"  Locked  : {n_locked} segment(s) (values preserved)")
        print(f"  Extended: {n_extended} segment(s) (newly calibrated this run)")
        if n_extended == 0:
            print("  No newly-calibrated segments — nothing to validate.")
        else:
            _analyse_tt_approximation(
                segments[newly_calibrated_mask], n_sta_endpoints, stop_counts, pass_counts,
            )
    else:
        _analyse_tt_approximation(
            segments, n_sta_endpoints, stop_counts, pass_counts,
        )

    # ── 5d. Correct Average_Speed from TT_Stopping ───────────────────────────
    # Direct effective speed v = L / (TT * 60) * 3.6, consistent with the
    # proportional cruise-time distribution used in step [4]. Applies to base
    # segments and feeder-derived segments alike.
    # In extend mode: skip segments whose Average_Speed is already populated —
    # their calibrated speed is the source of truth and must not be recalculated.
    print("\n[5d] Correcting Average_Speed from TT_Stopping...")
    n_corrected = 0
    targets = [segments] + ([derived] if derived is not None and not derived.empty else [])
    for target in targets:
        for idx, row in target.iterrows():
            tt_stop = row.get('TT_Stopping')
            if tt_stop is None or (isinstance(tt_stop, float) and pd.isna(tt_stop)):
                continue
            length_m = float(row.get('Length', 0) or 0)
            if length_m <= 0 or float(tt_stop) <= 0:
                continue
            if extend_mode:
                cur_spd = row.get('Average_Speed')
                already_set = (
                    cur_spd is not None
                    and not (isinstance(cur_spd, float) and pd.isna(cur_spd))
                    and float(cur_spd) > 0
                )
                if already_set:
                    continue
            target.at[idx, 'Average_Speed'] = round(
                length_m / (float(tt_stop) * 60.0) * 3.6, 1
            )
            # Mark as gtfs-calibrated — speed is now derived from timetable TT.
            # All segments reaching this point had service win the tier decision;
            # infra-win segments are skip-gated in step 4 and never reach here.
            sid = row.get('Segment_ID')
            if sid and target is segments:
                target.at[idx, 'speed_source'] = 'gtfs'
                sid_to_speed_source[str(sid)] = 'gtfs'
            n_corrected += 1

    # Feeder-derived segments carry GTFS TT from birth — mark them accordingly.
    if derived is not None and not derived.empty:
        if 'speed_source' not in derived.columns:
            derived['speed_source'] = 'gtfs'
        else:
            derived['speed_source'] = derived['speed_source'].fillna('gtfs')

    print(f"  Average_Speed corrected: {n_corrected} segment(s).")

    # ── 4.5 Infra → Service back-calculation (Direction 2) ───────────────────
    # For each service link whose tt_source tier is strictly lower than the
    # infra data along its path, recompute service TT from the enriched segment
    # TT values and write it back to the service files.
    #
    # Fire condition (weakest-link rule):
    #   - Any 'estimate' on path → skip (contaminated path)
    #   - Compute path_output_tier: 'gtfs' when all segs are {gtfs/design},
    #     'formula' when any seg is 'formula'
    #   - Fire when svc_tier < path_output_tier_rank
    #   - Also fire when service is 'gtfs' and path has at least one 'design'
    #     segment (design override)
    #
    # Output tt_source: 'gtfs' when path_output_tier is 'gtfs', 'formula' otherwise.
    print("\n[4.5] Infra → Service back-calculation (Direction 2)...")

    # Build post-step5d segment TT lookup (uses newly enriched values).
    seg_tt_stop_d2: Dict[str, float] = {}
    seg_tt_pass_d2: Dict[str, float] = {}
    for _, sr in segments.iterrows():
        _sid = str(sr.get('Segment_ID', ''))
        if not _sid:
            continue
        _ts = sr.get('TT_Stopping')
        _tp = sr.get('TT_Passing')
        if _ts is not None and not (isinstance(_ts, float) and pd.isna(_ts)):
            seg_tt_stop_d2[_sid] = float(_ts)
        if _tp is not None and not (isinstance(_tp, float) and pd.isna(_tp)):
            seg_tt_pass_d2[_sid] = float(_tp)

    # path_nodes_str → (new_tt, output_label)
    pn_to_d2: Dict[str, tuple] = {}

    d2_skipped_estimate = 0
    d2_skipped_same_tier = 0
    d2_skipped_no_tt = 0

    for _, svc_row in service_links_gdf.iterrows():
        raw_src = svc_row.get('tt_source')
        if raw_src is None or (isinstance(raw_src, float) and pd.isna(raw_src)):
            svc_tt_source = 'gtfs'
        else:
            svc_tt_source = str(raw_src).strip().lower() or 'gtfs'

        path_str = str(svc_row.get('path_nodes', '')).strip()
        if not path_str or path_str in pn_to_d2:
            continue

        try:
            node_ids = [int(n.strip()) for n in path_str.split(';') if n.strip()]
        except (ValueError, TypeError):
            continue
        if len(node_ids) < 2:
            continue

        pairs = list(zip(node_ids[:-1], node_ids[1:]))

        # Collect path segment tiers
        path_tiers: List[int] = []
        for u, v in pairs:
            _sid = seg_by_nodes.get((u, v)) or seg_by_nodes.get((v, u))
            if _sid is None:
                continue
            path_tiers.append(_SPEED_SOURCE_TIER.get(
                sid_to_speed_source.get(_sid, 'formula'), 1
            ))

        if not path_tiers:
            continue

        # Any 'estimate' (tier 0) on path → contaminated, skip
        if any(t == 0 for t in path_tiers):
            d2_skipped_estimate += 1
            continue

        # Determine path output tier label
        path_has_design = any(t == _SPEED_SOURCE_TIER['design'] for t in path_tiers)
        all_gtfs_or_design = all(t >= _SPEED_SOURCE_TIER['gtfs'] for t in path_tiers)
        path_output_tier = 'gtfs' if all_gtfs_or_design else 'formula'
        path_output_rank = _TT_SOURCE_TIER[path_output_tier]
        svc_tier_rank    = _TT_SOURCE_TIER.get(svc_tt_source, 1)

        # Fire condition
        fire = False
        if svc_tier_rank < path_output_rank:
            fire = True
        elif svc_tt_source == 'gtfs' and path_output_tier == 'gtfs' and path_has_design:
            fire = True  # design override: design can update gtfs service

        if not fire:
            d2_skipped_same_tier += 1
            continue

        # Determine stopping/passing label for this link
        intermediate = node_ids[1:-1]
        is_pass = any(n in stop_ids for n in intermediate)

        # Sum TT from enriched segment values
        new_tt = 0.0
        for u, v in pairs:
            _sid = seg_by_nodes.get((u, v)) or seg_by_nodes.get((v, u))
            if _sid is None:
                continue
            _tt = seg_tt_pass_d2.get(_sid) if is_pass else seg_tt_stop_d2.get(_sid)
            if _tt is not None:
                new_tt += _tt

        if new_tt <= 0:
            d2_skipped_no_tt += 1
            continue

        pn_to_d2[path_str] = (_round_half_min(new_tt), path_output_tier)

    print(f"  Direction 2: {len(pn_to_d2)} path(s) queued for service update.")
    print(f"  Skipped: {d2_skipped_estimate} (estimate on path), "
          f"{d2_skipped_same_tier} (same/lower tier), "
          f"{d2_skipped_no_tt} (no TT on segments).")

    if pn_to_d2:
        def _write_d2_back(
            gpkg_path: Path,
            track_layers: Optional[set] = None,
        ) -> None:
            """Re-read gpkg, apply Direction 2 TT updates, write all layers back."""
            if not gpkg_path.exists():
                return
            all_layers = fiona.listlayers(str(gpkg_path))
            first = True
            for lname in all_layers:
                gdf = gpd.read_file(str(gpkg_path), layer=lname)
                if track_layers is not None and lname not in track_layers:
                    gdf.to_file(str(gpkg_path), layer=lname, driver='GPKG',
                                mode='w' if first else 'a')
                    first = False
                    continue
                if 'path_nodes' not in gdf.columns:
                    gdf.to_file(str(gpkg_path), layer=lname, driver='GPKG',
                                mode='w' if first else 'a')
                    first = False
                    continue
                tt_col = 'TT' if 'TT' in gdf.columns else 'travel_time_min'
                if 'tt_source' not in gdf.columns:
                    gdf['tt_source'] = 'gtfs'
                n_upd_gtfs = 0
                n_upd_formula = 0
                for idx, row in gdf.iterrows():
                    pn = str(row.get('path_nodes', ''))
                    if pn not in pn_to_d2:
                        continue
                    new_tt_val, out_label = pn_to_d2[pn]
                    gdf.at[idx, tt_col]      = new_tt_val
                    gdf.at[idx, 'tt_source'] = out_label
                    if out_label == 'gtfs':
                        n_upd_gtfs += 1
                    else:
                        n_upd_formula += 1
                if n_upd_gtfs or n_upd_formula:
                    print(f"    {gpkg_path.name} / '{lname}': "
                          f"{n_upd_gtfs} → tt_source='gtfs', "
                          f"{n_upd_formula} → tt_source='formula'")
                gdf.to_file(str(gpkg_path), layer=lname, driver='GPKG',
                            mode='w' if first else 'a')
                first = False

        _write_d2_back(rail_source_dir / "rail_segments.gpkg")
        _write_d2_back(
            feeder_source_dir / "pt_feeder_segments.gpkg",
            track_layers=TRACK_BASED_FEEDER_LAYERS,
        )
        print("  Service files updated in-place (Direction 2).")

    return segments, derived


# =============================================================================
# Save
# =============================================================================

def _save_enhanced_version(
    enhanced_dir: Path,
    base_segments: gpd.GeoDataFrame,
    derived_segments: Optional[gpd.GeoDataFrame],
    source_composition_path: Path,
) -> None:
    """
    Write segments.gpkg and segments_composition.gpkg to enhanced_dir.

    segments.gpkg   = base_segments (TT-enriched) + derived_segments (if any).
    segments_composition.gpkg = source composition + one normal piece per
    feeder-derived segment.

    nodes.gpkg was already copied by _create_enhanced_dir in Phase 0.

    Args:
        enhanced_dir: Output directory (already created with nodes.gpkg).
        base_segments: TT-enriched source segments GeoDataFrame.
        derived_segments: Feeder-derived new segments, or None.
        source_composition_path: Path to source segments_composition.gpkg.
    """
    source_comp = gpd.read_file(source_composition_path)

    n_base     = len(base_segments)
    n_derived  = 0
    n_stopping = int(base_segments['TT_Stopping'].notna().sum())

    if derived_segments is not None and not derived_segments.empty:
        n_derived       = len(derived_segments)
        derived_comp    = _build_feeder_composition(derived_segments)
        final_segments  = pd.concat([base_segments, derived_segments], ignore_index=True)
        final_comp      = pd.concat([source_comp, derived_comp], ignore_index=True)
    else:
        final_segments = base_segments
        final_comp     = source_comp

    final_segments.to_file(str(enhanced_dir / 'segments.gpkg'), driver='GPKG')
    final_comp.to_file(str(enhanced_dir / 'segments_composition.gpkg'), driver='GPKG')

    print(f"\n  Summary:")
    print(f"    Base segments  : {n_base}  ({n_stopping} with TT_Stopping populated)")
    print(f"    Derived segments added: {n_derived}")
    print(f"    Total in enhanced version: {len(final_segments)}")
    print(f"  Saved → {enhanced_dir}")


# =============================================================================
# Main
# =============================================================================

def _run_phase0_auto(
    infra_version: str,
    svc_version:   str,
    enhanced_name: str,
):
    """Non-interactive Phase 0 for pipeline invocation.

    Args:
        infra_version: e.g. 'AS_2026_ZH' (base, not yet enhanced).
        svc_version:   e.g. 'SVC2026_ZH_S18_network'.
        enhanced_name: e.g. 'AS_2026_ZH_enhanced'.
    """
    infra_root = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR
    infra_dir  = infra_root / infra_version

    if not infra_dir.exists():
        print(f"  ERROR: Infrastructure version '{infra_version}' not found at {infra_dir}.")
        raise SystemExit(1)

    # Validate svc_version
    all_svc = list_svc_versions()
    if svc_version not in all_svc:
        print(f"  ERROR: Service version '{svc_version}' not found. Available: {all_svc}")
        raise SystemExit(1)

    # Find projection source
    sources = _list_projection_sources(svc_version, infra_version)
    matching_idx = next(
        (i for i, (label, _, _) in enumerate(sources) if label == infra_version),
        None,
    )
    if matching_idx is None:
        print(f"  ERROR: No projected source for infra='{infra_version}', svc='{svc_version}'.")
        print(f"  Run services_service_projection.py first.")
        raise SystemExit(1)

    _, feeder_source_dir, rail_source_dir = sources[matching_idx]

    extend_mode = infra_version.lower().endswith('_enhanced')
    enhanced_dir = infra_dir if extend_mode else infra_root / enhanced_name
    if not extend_mode:
        _create_enhanced_dir(enhanced_dir)

    print(f"  Auto-config (non-interactive):")
    print(f"  Infrastructure : {infra_version}")
    print(f"  Service version: {svc_version}")
    print(f"  Enhanced output: {enhanced_dir}")
    return (infra_version, svc_version,
            feeder_source_dir, rail_source_dir,
            infra_dir, enhanced_dir, extend_mode)


def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--infra-version',  default=None,
                        help='Infrastructure version to enhance (without _enhanced suffix)')
    parser.add_argument('--svc-version',    default=None,
                        help='Service version name (with _network suffix)')
    parser.add_argument('--enhanced-name',  default=None,
                        help='Name for the enhanced output version')
    args, _ = parser.parse_known_args()

    try:
        if args.infra_version and args.svc_version and args.enhanced_name:
            result = _run_phase0_auto(
                args.infra_version,
                args.svc_version,
                args.enhanced_name,
            )
        else:
            result = _run_phase0()
    except SystemExit:
        return

    (infra_version, svc_version,
     feeder_source_dir, rail_source_dir,
     infra_dir, enhanced_dir, extend_mode) = result

    print(f"\n  Infrastructure : {infra_version}")
    print(f"  Service version: {svc_version}")

    print(f"  Feeder source  : {feeder_source_dir}")
    print(f"  Rail source    : {rail_source_dir}")
    mode_str = "extend (in-place)" if extend_mode else "initial"
    print(f"  Enhanced output: {enhanced_dir}  [{mode_str}]")

    base_segments, derived = compute_segment_stats(
        infra_version, svc_version,
        feeder_source_dir, rail_source_dir,
        infra_dir, enhanced_dir,
        extend_mode=extend_mode,
    )

    print("\n[6] Saving enhanced version...")
    _save_enhanced_version(
        enhanced_dir,
        base_segments,
        derived,
        infra_dir / 'segments_composition.gpkg',
    )
    print("  Done.")


if __name__ == '__main__':
    main()
