"""
Version Manager Module

Interactive TUI for creating named infrastructure versions derived from the
macroscopic Base network, or adjusting existing versions.

Workflow
--------
  Phase 0: Select or create a version (folder creation + copy for new versions).
  Phase 1: Node editing loop — remove / adjust / add.
  Phase 2: Segment editing loop — remove / adjust / add.
  Phase 3: Save all three geopackages to disk.

Usage (interactive):
    python infrabuild_version_manager.py
"""

import os
import shutil
import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import fiona
from shapely.geometry import LineString, Point
from shapely.ops import substring, linemerge
import sys

sys.path.insert(0, str(Path(__file__).parent))
import paths
import settings
import infrabuild_network_builder as ic
from infrabuild_network_builder import (
    auto_tt as _auto_tt_core,
    count_stations_at_endpoints as _count_stations_at_endpoints,
    default_speed_for_segment as _default_speed_for_segment,
    split_segment_at as _split_segment_at_core,
    validate_and_autofill as _validate_and_autofill_core,
)


# =============================================================================
# Constants
# =============================================================================

SWISS_CRS = ic.SWISS_CRS
MERGE_ATTRS = ('Num_Tracks', 'Gauge', 'Electrification_Class')
_NODE_SNAP_TOLERANCE_M = ic.NODE_SNAP_TOLERANCE_M

GAUGE_OPTIONS           = ['1435', '1000']
ELECTRIFICATION_OPTIONS = ['AC_16.7Hz', 'DC', 'non_electrified']
NODE_CLASS_OPTIONS      = ['station', 'junction', 'abandoned_station']
CONSTRUCT_TYPE_OPTIONS  = ['bridge', 'normal', 'tunnel']
SPEED_SOURCE_OPTIONS    = ['estimate', 'formula', 'gtfs', 'design']
TRACK_MODE_OPTIONS      = ['train', 'tram', 'cog_railway', 'train / tram']
_EDGE_LEVEL_BY_CT       = {'bridge': ['2', '3'], 'normal': ['1'], 'tunnel': ['-1', '-2']}


# Helpers _auto_tt, _count_stations_at_endpoints, _default_speed_for_segment are
# imported from infrabuild_network_builder above. Re-exported below for callers that
# still reference the local underscored name.
_auto_tt = _auto_tt_core
_MODE_DEFAULT_SPEEDS_VM = ic._MODE_DEFAULT_SPEEDS_VM
_MODE_DEFAULT_FALLBACK_VM = ic._MODE_DEFAULT_FALLBACK_VM


# =============================================================================
# Prompt helpers
# =============================================================================

def _prompt_enum(
    label: str,
    options: List[str],
    current: Optional[str] = None,
    required: bool = False,
    allow_other: bool = True,
) -> Optional[str]:
    """Single-question enum prompt. Enter keeps current or skips (add mode)."""
    cur_str = f'  (current: {current})' if current is not None else ''
    print(f'\n  {label}{cur_str}:')
    for i, opt in enumerate(options, 1):
        print(f'    {i}) {opt}')
    other_n = len(options) + 1
    if allow_other:
        print(f'    {other_n}) other')
    max_n = other_n if allow_other else len(options)
    enter_hint = 'keep' if current is not None else ('required' if required else 'skip')
    prompt = f'  Select (1–{max_n}' + (f', Enter to {enter_hint}' if not required else '') + '): '

    while True:
        raw = input(prompt).strip()
        if not raw:
            if required:
                print('  This field is required.')
                continue
            return current
        if raw.isdigit():
            n = int(raw)
            if 1 <= n <= len(options):
                return options[n - 1]
            if allow_other and n == other_n:
                val = input('  Enter value: ').strip()
                return val if val else current
        print(f'  Enter 1–{max_n}.')


def _prompt_text(
    label: str,
    example: str = '',
    current: Optional[str] = None,
    required: bool = False,
    cast=None,
) -> Optional[object]:
    """Single-question free-text / numeric prompt. Enter keeps current or skips."""
    if current is not None:
        hint = f' (current: {current}, Enter to keep)'
    elif example:
        hint = f' (e.g. {example}, Enter to skip)'
    else:
        hint = ' (Enter to skip)'
    prompt = f'\n  {label}{hint}: '

    while True:
        raw = input(prompt).strip()
        if not raw:
            if required:
                print('  This field is required.')
                continue
            return current
        if cast is not None:
            try:
                return cast(raw)
            except (ValueError, TypeError):
                print(f'  Expected {cast.__name__}. Try again.')
                continue
        return raw


def _owner_options(segments: gpd.GeoDataFrame) -> List[str]:
    """Return sorted unique Route_Owner values present in the loaded segments."""
    return sorted(segments['Route_Owner'].dropna().astype(str).unique().tolist())


def _parse_selection(ans: str, max_count: int) -> List[int]:
    """Parse a diff-selection string into a sorted list of 0-based indices.

    Accepts 'all', individual numbers, ranges (e.g. '21-25'), and any
    comma-separated combination (e.g. '1,3,21-25,30').
    Out-of-range entries are silently ignored.
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


def _prompt_composition_pieces(
    seg_id: str,
    from_name: str,
    to_name: str,
    seg_geom,
    seg_len: float,
) -> gpd.GeoDataFrame:
    """
    Interactively collect one or more composition pieces for a segment and
    return a GeoDataFrame of those pieces (NOT appended to composition yet).
    """
    print(f"\n  Segment length: {seg_len:.0f} m — all pieces must sum to this.")
    while True:
        multi_raw = input("  Multiple composition pieces? (y/n) [n]: ").strip().lower() or 'n'
        if multi_raw in ('y', 'n'):
            break
        print("  Enter y or n.")
    multi = multi_raw == 'y'

    pieces = []
    remaining = seg_len

    while True:
        ct = _prompt_enum(
            f'Construct type (remaining: {remaining:.0f} m)',
            CONSTRUCT_TYPE_OPTIONS,
            required=True,
            allow_other=False,
        )

        if multi:
            try:
                pl = float(input(f"  Piece length (m) [remaining={remaining:.0f}]: ").strip())
            except ValueError:
                print("  Invalid length. Skipping.")
                continue
        else:
            pl = seg_len

        el_opts = _EDGE_LEVEL_BY_CT.get(ct, ['1'])
        el_raw = _prompt_enum('Edge level', el_opts, current=el_opts[0], allow_other=False)
        el = int(el_raw) if el_raw is not None else int(el_opts[0])

        uc_raw = _prompt_enum('Under construction', ['0 — no', '1 — yes'],
                              current='0 — no', allow_other=False)
        uc = 1 if uc_raw and uc_raw.startswith('1') else 0

        pieces.append({
            'Segment_ID':            seg_id,
            'From_Name':             from_name,
            'To_Name':               to_name,
            'Engineering_Structure': ct,
            'Edge_Level':            el,
            'Under_Construction':    uc,
            'Piece_Length':          pl,
            'geometry':              seg_geom,
        })
        remaining -= pl
        print(f"  Piece added: {ct}  {pl:.0f} m.")

        if not multi:
            break
        if remaining <= 0:
            print("  Total length reached.")
            break
        add_more = input("  Add another piece? (y/n) [y]: ").strip().lower() or 'y'
        if add_more != 'y':
            break

    # Budget guard
    total = sum(p['Piece_Length'] for p in pieces)
    if pieces and abs(total - seg_len) > 0.5:
        print(f"  WARNING: pieces sum to {total:.0f} m but segment length is {seg_len:.0f} m "
              f"(delta = {total - seg_len:+.0f} m).")
        adj = input("  Auto-adjust last piece to close the gap? (y/n) [y]: ").strip().lower() or 'y'
        if adj == 'y':
            corrected = seg_len - (total - pieces[-1]['Piece_Length'])
            if corrected > 0:
                pieces[-1]['Piece_Length'] = corrected
                print(f"  Last piece adjusted to {corrected:.0f} m.")
            else:
                print("  Gap correction would produce a non-positive piece. Keeping as-is.")

    return gpd.GeoDataFrame(pieces, crs=SWISS_CRS)


# =============================================================================
# Version Discovery
# =============================================================================

def list_versions(infra_dir: Optional[str] = None) -> List[str]:
    """Return selectable version names (excludes Raw* folders)."""
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
    base_versions  = ['Base'] if 'Base' in versions else []
    base_versions += sorted([v for v in versions if v.startswith('Base') and v != 'Base'])
    other_versions = [v for v in versions if not v.startswith('Base')]
    return base_versions + other_versions


# =============================================================================
# Search / pick helpers
# =============================================================================

def _search_nodes(nodes: gpd.GeoDataFrame, term: str) -> gpd.GeoDataFrame:
    """Return rows whose Name or Code contain term (case-insensitive)."""
    term_l = term.lower()
    mask = (
        nodes['Name'].fillna('').str.lower().str.contains(term_l, regex=False) |
        nodes['Code'].fillna('').str.lower().str.contains(term_l, regex=False)
    )
    return nodes[mask]


def _search_segments(segments: gpd.GeoDataFrame, term: str) -> gpd.GeoDataFrame:
    """Return rows whose From_Name, To_Name, or Segment_ID contain term (case-insensitive)."""
    term_l = term.lower()
    mask = (
        segments['From_Name'].fillna('').str.lower().str.contains(term_l, regex=False) |
        segments['To_Name'].fillna('').str.lower().str.contains(term_l, regex=False) |
        segments['Segment_ID'].fillna('').astype(str).str.lower().str.contains(term_l, regex=False)
    )
    return segments[mask]


def _pick_one(labels: List[str], prompt: str = "Select") -> Optional[int]:
    """
    Display a numbered list and return the 0-based index of the chosen item.
    Returns None if the user presses Enter without a number.
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
# Phase 0 — Version selection
# =============================================================================

def _run_phase0(
    create_from: Optional[str] = None,
    preset_name: Optional[str] = None,
    auto_overwrite: bool = False,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame,
           str, str, Path, str]:
    """
    Phase 0 — version selection.

    Parameters
    ----------
    create_from : str, optional
        If set, skip Q1/Q2 and create a new version from this base version name.
    preset_name : str, optional
        If set, skip Q3 and use this as the new version name.
    auto_overwrite : bool
        If True, automatically answer 'y' to the overwrite prompt.

    Returns
    -------
    nodes, segments, composition, version_name, source_version, out_dir, mode
    mode is 'new' or 'adjust'.
    """
    infra_root = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR

    # Prerequisite: at least one Base* folder with all three geopackages must exist
    required = ['nodes.gpkg', 'segments.gpkg', 'segments_composition.gpkg']
    base_dirs = sorted([
        d for d in infra_root.iterdir()
        if d.is_dir() and d.name.startswith('Base')
        and all((d / f).exists() for f in required)
    ]) if infra_root.exists() else []
    if not base_dirs:
        print("\n  ERROR: No complete Base version found.")
        print(f"  Expected a folder starting with 'Base' under: {infra_root}")
        print(f"  Each must contain: {', '.join(required)}")
        print("  Run infrabuild_network_builder.py to build Base first.")
        raise SystemExit(1)

    versions = list_versions()
    if not versions:
        print(f"\n  No versions found in {infra_root}.")
        print("  Run infrabuild_network_builder.py to build Base first.")
        raise SystemExit(1)

    if create_from is not None:
        # Non-interactive path: create a new version from the specified base
        mode = 'new'
        if create_from not in versions:
            print(f"\n  ERROR: Base version '{create_from}' not found.")
            print(f"  Available: {', '.join(versions)}")
            raise SystemExit(1)
        source_version = create_from
        print(f"\n  [auto] Creating new version from '{source_version}'.")

        if preset_name is None:
            name = input("\n  Name for the new version (e.g. AS_2035): ").strip()
        else:
            name = preset_name
            print(f"  [auto] Version name: '{name}'.")

        if not name:
            print("  Name cannot be empty.")
            raise SystemExit(1)
        if name.startswith('Raw') or name.startswith('Base'):
            print(f"  '{name}' starts with a reserved prefix (Raw/Base).")
            raise SystemExit(1)

        out_dir = infra_root / name
        if out_dir.exists() and (out_dir / 'nodes.gpkg').exists():
            if auto_overwrite:
                print(f"  Version '{name}' already exists — overwriting (auto).")
            else:
                print(f"  Version '{name}' already exists.")
                overwrite = input("  Overwrite? (y/n) [n]: ").strip().lower() or 'n'
                if overwrite != 'y':
                    raise SystemExit(0)

        source_dir = infra_root / source_version
        out_dir.mkdir(parents=True, exist_ok=True)
        for gpkg in ('nodes.gpkg', 'segments.gpkg', 'segments_composition.gpkg'):
            shutil.copy2(source_dir / gpkg, out_dir / gpkg)
        print(f"  Copied {source_version}/ → {name}/")

    else:
        # Q1 — what do you want to do?
        print("\n" + "─" * 60)
        print("  What do you want to do?")
        print("    1) Create a new version")
        print("    2) Adjust an existing version")
        while True:
            choice = input("  Select (1/2): ").strip()
            if choice in ('1', '2'):
                break
            print("  Enter 1 or 2.")

        if choice == '1':
            mode = 'new'

            # Q2 — choose base version
            print("\n  Choose base version:")
            idx = _pick_one(versions, "Base version")
            if idx is None:
                raise SystemExit(0)
            source_version = versions[idx]

            # Q3 — name for the new version
            while True:
                name = input("\n  Name for the new version (e.g. AS_2035): ").strip()
                if not name:
                    print("  Name cannot be empty.")
                    continue
                if name.startswith('Raw') or name.startswith('Base'):
                    print(f"  '{name}' starts with a reserved prefix (Raw/Base). Choose another.")
                    continue
                out_dir = infra_root / name
                if out_dir.exists() and (out_dir / 'nodes.gpkg').exists():
                    print(f"  Version '{name}' already exists.")
                    overwrite = input("  Overwrite? (y/n) [n]: ").strip().lower() or 'n'
                    if overwrite != 'y':
                        continue
                break

            # Create folder and copy geopackages from source version
            source_dir = infra_root / source_version
            out_dir.mkdir(parents=True, exist_ok=True)
            for gpkg in ('nodes.gpkg', 'segments.gpkg', 'segments_composition.gpkg'):
                shutil.copy2(source_dir / gpkg, out_dir / gpkg)
            print(f"  Copied {source_version}/ → {name}/")

        else:  # choice == '2'
            mode = 'adjust'
            # list_versions already excludes Raw
            print("\n  Choose version to adjust:")
            idx = _pick_one(versions, "Version")
            if idx is None:
                raise SystemExit(0)
            name = versions[idx]
            source_version = name
            out_dir = infra_root / name

    # Load the three geopackages
    print(f"\n  Loading '{name}'...")
    nodes       = gpd.read_file(out_dir / 'nodes.gpkg').reset_index(drop=True)
    segments    = gpd.read_file(out_dir / 'segments.gpkg').reset_index(drop=True)
    composition = gpd.read_file(out_dir / 'segments_composition.gpkg').reset_index(drop=True)
    if 'speed_source' not in segments.columns:
        segments['speed_source'] = 'formula'
    print(f"  {len(nodes)} nodes, {len(segments)} segments, "
          f"{len(composition)} composition pieces loaded.")

    return nodes, segments, composition, name, source_version, out_dir, mode


# =============================================================================
# Node operations
# =============================================================================

def _adjust_node(nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Inline editing of a single node's mutable attributes.

    Read-only (identity): Name, Code, E, N.
    Editable: Node_Class, Transport_Mode, Platform_Count, Track_Count, Parent_Node.
    """
    term = input("  Search node (CODE or partial NAME): ").strip()
    if not term:
        return nodes
    hits = _search_nodes(nodes, term)
    if hits.empty:
        print(f"  No nodes matching '{term}'.")
        return nodes

    labels = [f"{r['Name']}  [{r['Code']}]" for _, r in hits.iterrows()]
    idx = _pick_one(labels, "Node to adjust")
    if idx is None:
        return nodes

    row_idx = hits.index[idx]
    row = nodes.loc[row_idx]

    print(f"\n  Node: {row.get('Name', '')}  [{row.get('Code', '')}]"
          f"  E={row.get('E', '')}  N={row.get('N', '')}  [read-only]")

    def _cur(col):
        v = row.get(col)
        return None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)

    nc = _prompt_enum('Node class', NODE_CLASS_OPTIONS, current=_cur('Node_Class'),
                      allow_other=False)
    if nc is not None:
        nodes.at[row_idx, 'Node_Class'] = nc

    tm = _prompt_enum('Transport mode', TRACK_MODE_OPTIONS, current=_cur('Transport_Mode'),
                      allow_other=False)
    if tm is not None:
        nodes.at[row_idx, 'Transport_Mode'] = tm

    effective_class = nc if nc is not None else _cur('Node_Class')
    if effective_class != 'junction':
        pc = _prompt_text('Platform count', example='2', current=_cur('Platform_Count'), cast=int)
        if pc is not None:
            nodes.at[row_idx, 'Platform_Count'] = int(pc) if isinstance(pc, str) else pc

    tc = _prompt_text('Track count', example='4', current=_cur('Track_Count'), cast=int)
    if tc is not None:
        nodes.at[row_idx, 'Track_Count'] = int(tc) if isinstance(tc, str) else tc

    pn = _prompt_text('Parent node', example='Zürich HB', current=_cur('Parent_Node'))
    if pn is not None:
        nodes.at[row_idx, 'Parent_Node'] = pn

    print(f"  Node '{nodes.at[row_idx, 'Name']}' updated.")
    return nodes


def _remove_node(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    composition: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Remove a node with optional segment merge."""
    term = input("  Search node (Code or partial Name): ").strip()
    if not term:
        return nodes, segments, composition
    hits = _search_nodes(nodes, term)
    if hits.empty:
        print(f"  No nodes matching '{term}'.")
        return nodes, segments, composition

    labels = [f"{r['Name']}  [{r['Code']}]" for _, r in hits.iterrows()]
    idx = _pick_one(labels, "Node to remove")
    if idx is None:
        return nodes, segments, composition

    row_idx   = hits.index[idx]
    node_row  = nodes.loc[row_idx]
    node_name = node_row['Name']

    # Print node attributes
    print(f"\n  Node: {node_name}")
    for col in ['Code', 'E', 'N', 'Node_Class', 'Transport_Mode', 'Platform_Count']:
        print(f"    {col}: {node_row.get(col, '')}")

    # Connected segments
    conn_mask = (segments['From_Name'] == node_name) | (segments['To_Name'] == node_name)
    conn_segs = segments[conn_mask]
    print(f"\n  Connected segments ({len(conn_segs)}):")
    for _, s in conn_segs.iterrows():
        print(f"    {s['From_Name']} → {s['To_Name']}  [{s['Segment_ID']}]")

    # Merge check: exactly 2 connected segs with identical MERGE_ATTRS
    merged = False
    if len(conn_segs) == 2:
        s0, s1 = conn_segs.iloc[0], conn_segs.iloc[1]
        if all(s0[a] == s1[a] for a in MERGE_ATTRS):
            ans = input(
                "\n  Merge the two adjacent segments into one? (y/n) [n]: "
            ).strip().lower() or 'n'
            if ans == 'y':
                # Surviving outer endpoints
                all_ends = {s0['From_Name'], s0['To_Name'],
                            s1['From_Name'], s1['To_Name']}
                surviving = [e for e in all_ends if e != node_name]
                from_end, to_end = surviving[0], surviving[1]

                fe_code_rows = nodes[nodes['Name'] == from_end]
                te_code_rows = nodes[nodes['Name'] == to_end]
                fe_code = fe_code_rows.iloc[0]['Code'] if not fe_code_rows.empty else from_end
                te_code = te_code_rows.iloc[0]['Code'] if not te_code_rows.empty else to_end
                new_seg_id  = f"c{fe_code}_{te_code}"
                new_geom    = linemerge([s0.geometry, s1.geometry])
                new_length  = new_geom.length
                new_km_start = min(s0['Km_Start'], s1['Km_Start'])
                new_km_end   = max(s0['Km_End'],   s1['Km_End'])

                fn_row = nodes[nodes['Name'] == from_end]
                tn_row = nodes[nodes['Name'] == to_end]

                fe_num = fn_row.iloc[0]['Number'] if not fn_row.empty else pd.NA
                te_num = tn_row.iloc[0]['Number'] if not tn_row.empty else pd.NA
                new_number = f"{int(fe_num)}_{int(te_num)}" if pd.notna(fe_num) and pd.notna(te_num) else pd.NA

                new_row = {
                    'Segment_ID':                  new_seg_id,
                    'Number':              new_number,
                    'Code':                f"{fe_code}_{te_code}",
                    'From_Name':           from_end,
                    'To_Name':             to_end,
                    'From_N': fn_row.iloc[0]['N'] if not fn_row.empty else pd.NA,
                    'From_E': fn_row.iloc[0]['E'] if not fn_row.empty else pd.NA,
                    'To_N':   tn_row.iloc[0]['N'] if not tn_row.empty else pd.NA,
                    'To_E':   tn_row.iloc[0]['E'] if not tn_row.empty else pd.NA,
                    'Length':              new_length,
                    'Num_Tracks':          s0['Num_Tracks'],
                    'Gauge':               s0['Gauge'],
                    'Electrification_Class': s0['Electrification_Class'],
                    'Km_Start':            new_km_start,
                    'Km_End':              new_km_end,
                    'Route_Number':        s0['Route_Number'],
                    'Route_Name':          s0['Route_Name'],
                    'Route_Owner':         s0['Route_Owner'],
                    'speed_source':        s0.get('speed_source', 'formula'),
                    'geometry':            new_geom,
                }

                # Remove the two old segments
                old_ids = conn_segs['Segment_ID'].tolist()
                segments = segments[
                    ~segments['Segment_ID'].isin(old_ids)
                ].reset_index(drop=True)
                segments = pd.concat(
                    [segments, gpd.GeoDataFrame([new_row], crs=SWISS_CRS)],
                    ignore_index=True
                )

                # Replace composition: remove 2 old rows, add 1 normal row
                composition = composition[
                    ~composition['Segment_ID'].isin(old_ids)
                ].reset_index(drop=True)
                comp_row = gpd.GeoDataFrame([{
                    'Segment_ID':                   new_seg_id,
                    'From_Name':            from_end,
                    'To_Name':              to_end,
                    'Engineering_Structure': 'normal',
                    'Edge_Level':           1,
                    'Under_Construction':   0,
                    'Piece_Length':         new_length,
                    'geometry':             new_geom,
                }], crs=SWISS_CRS)
                composition = pd.concat([composition, comp_row], ignore_index=True)

                # Remove node
                nodes = nodes[nodes.index != row_idx].reset_index(drop=True)
                print(f"  Merged → '{new_seg_id}' ({new_length / 1000:.2f} km).")
                merged = True

    if not merged:
        n_conn = len(conn_segs)
        print(f"\n  How should connected segments be handled?")
        print(f"    1) Remove node only — leave {n_conn} connected segment(s) as-is")
        print(f"    2) Remove node and all {n_conn} connected segment(s)")
        while True:
            c = input("  Select (1/2): ").strip()
            if c in ('1', '2'):
                break
            print("  Enter 1 or 2.")

        # Remove node
        nodes = nodes[nodes.index != row_idx].reset_index(drop=True)

        if c == '2':
            remove_ids = conn_segs['Segment_ID'].tolist()
            segments = segments[
                ~segments['Segment_ID'].isin(remove_ids)
            ].reset_index(drop=True)
            composition = composition[
                ~composition['Segment_ID'].isin(remove_ids)
            ].reset_index(drop=True)
            print(f"  Removed node '{node_name}' and {len(remove_ids)} segment(s).")
        else:
            print(f"  Removed node '{node_name}'. Connected segments left as-is.")

    return nodes, segments, composition


# =============================================================================
# Segment split helper (used by Add Node)
# =============================================================================

def _split_segment_at(
    segments: gpd.GeoDataFrame,
    composition: gpd.GeoDataFrame,
    seg_idx: int,
    split_dist: float,
    new_node_name: str,
    new_node_code: str,
    nodes: gpd.GeoDataFrame,
    new_node_class: Optional[str] = None,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Delegates to infrabuild_network_builder.split_segment_at (pure version)."""
    return _split_segment_at_core(
        segments, composition, seg_idx, split_dist,
        new_node_name, new_node_code, nodes, new_node_class,
    )


def _add_node(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    composition: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Add a node by route+km or by coordinates, splitting the nearest segment."""
    print("\n  How do you want to locate the new node?")
    print("    1) By route number and kilometre position")
    print("    2) By exact coordinates (E, N)")
    while True:
        c = input("  Select (1/2): ").strip()
        if c in ('1', '2'):
            break
        print("  Enter 1 or 2.")

    seg_idx    = None
    split_dist = None
    node_E = node_N = None

    if c == '1':
        route = input("  Route number: ").strip()
        route_segs = segments[
            segments['Route_Number'].fillna('').astype(str) == route
        ]
        if route_segs.empty:
            print(f"  No segments found for route '{route}'.")
            return nodes, segments, composition

        while True:
            try:
                km = float(input("  Kilometre position: ").strip())
            except ValueError:
                print("  Enter a numeric km value.")
                continue
            match = route_segs[
                (route_segs['Km_Start'].fillna(-1) <= km) &
                (route_segs['Km_End'].fillna(-1)   >= km)
            ]
            if match.empty:
                print(f"  No segment covers km {km} on route '{route}'. Try again.")
                continue
            seg_idx = match.index[0]
            S = segments.loc[seg_idx]
            t          = (km - S['Km_Start']) / (S['Km_End'] - S['Km_Start'])
            split_dist = t * S.geometry.length
            split_pt   = S.geometry.interpolate(split_dist)
            node_E, node_N = split_pt.x, split_pt.y
            break

    else:  # by coordinates
        try:
            node_E = float(input("  Easting  E (m, EPSG:2056): ").strip())
            node_N = float(input("  Northing N (m, EPSG:2056): ").strip())
        except ValueError:
            print("  Invalid coordinates.")
            return nodes, segments, composition
        snap_raw = input("  Snap to nearest segment and split it? (y/n) [n]: ").strip().lower() or 'n'
        if snap_raw == 'y':
            pt         = Point(node_E, node_N)
            seg_idx    = segments.geometry.distance(pt).idxmin()
            S          = segments.loc[seg_idx]
            split_dist = S.geometry.project(pt)
            split_pt   = S.geometry.interpolate(split_dist)
            node_E, node_N = split_pt.x, split_pt.y

    # Prompt node attributes
    print(f"\n  New node will be at E={node_E:.1f}, N={node_N:.1f}")
    name = input("  Name: ").strip()
    if not name:
        print("  Name cannot be empty. Cancelled.")
        return nodes, segments, composition
    if name in nodes['Name'].values:
        print(f"  A node named '{name}' already exists. Cancelled.")
        return nodes, segments, composition

    code = input(f"  Code [{name[:4].upper()}]: ").strip() or name[:4].upper()
    existing_codes = set(nodes['Code'].dropna().astype(str).str.strip())
    if code in existing_codes:
        print(f"  Code '{code}' is already used by another node. Cancelled.")
        return nodes, segments, composition

    node_class = _prompt_enum('Node class', NODE_CLASS_OPTIONS, required=True, allow_other=False)

    # Generate a synthetic Number above the BAV range (max + 1,
    # floored at 9_000_000 so synthetic nodes are clearly distinguishable).
    existing_ids = nodes['Number'].dropna()
    try:
        max_existing = int(existing_ids.astype(float).max())
    except (ValueError, TypeError):
        max_existing = 0
    synthetic_bpnr = max(max_existing + 1, 9_000_000)
    # Ensure uniqueness in the unlikely case of multiple additions in one session
    while synthetic_bpnr in existing_ids.astype(float).values:
        synthetic_bpnr += 1

    synthetic_node_id = f"synth_{synthetic_bpnr}"

    new_node = {
        'Node_ID':                  synthetic_node_id,
        'Number':           synthetic_bpnr,
        'Name':             name,
        'Code':             code,
        'E':                node_E,
        'N':                node_N,
        'Node_Class':       node_class,
        'Transport_Mode':   None,
        'Platform_Count':   None,
        'Track_Count':      None,
        'Parent_Node':      None,
        'geometry':         Point(node_E, node_N),
    }
    print(f"  Assigned synthetic Number: {synthetic_bpnr}")

    if seg_idx is not None:
        segments, composition = _split_segment_at(
            segments, composition, seg_idx, split_dist, name, code, nodes, node_class
        )

    # Append new node
    nodes = pd.concat(
        [nodes, gpd.GeoDataFrame([new_node], crs=SWISS_CRS)],
        ignore_index=True
    )
    if seg_idx is not None:
        print(f"  Added node '{name}' ({node_class}) — segment split at {split_dist:.0f} m.")
    else:
        print(f"  Added node '{name}' ({node_class}).")
    return nodes, segments, composition

# =============================================================================
# Segment operations
# =============================================================================

def _remove_segment(
    segments: gpd.GeoDataFrame,
    composition: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Phase 2 — Remove a segment, with optional orphan-node cleanup."""
    term = input("  Search segment (partial from-name, to-name, or segment_name): ").strip()
    if not term:
        return segments, composition, nodes
    hits = _search_segments(segments, term)
    if hits.empty:
        print(f"  No segments matching '{term}'.")
        return segments, composition, nodes

    labels = [
        f"{r['From_Name']} → {r['To_Name']}  [{r['Segment_ID']}]"
        for _, r in hits.iterrows()
    ]
    idx = _pick_one(labels, "Segment to remove")
    if idx is None:
        return segments, composition, nodes

    row_idx = hits.index[idx]
    S = segments.loc[row_idx]

    ans = input(f"  Remove segment '{S['From_Name']} → {S['To_Name']}'? (y/n) [n]: ").strip().lower() or 'n'
    if ans == 'y':
        endpoints = (S['From_Name'], S['To_Name'])
        segments    = segments[segments.index != row_idx].reset_index(drop=True)
        composition = composition[composition['Segment_ID'] != S['Segment_ID']].reset_index(drop=True)
        print(f"  Removed segment '{S['Segment_ID']}'.")

        # Offer to remove any endpoint node now left with zero connected segments
        for ep_name in endpoints:
            still_connected = (
                (segments['From_Name'] == ep_name) | (segments['To_Name'] == ep_name)
            ).any()
            if not still_connected and (nodes['Name'] == ep_name).any():
                orphan_ans = input(
                    f"  Node '{ep_name}' is now isolated (no connected segments). "
                    f"Remove it? (y/n) [n]: "
                ).strip().lower() or 'n'
                if orphan_ans == 'y':
                    nodes = nodes[nodes['Name'] != ep_name].reset_index(drop=True)
                    print(f"  Removed orphan node '{ep_name}'.")

    return segments, composition, nodes


def _adjust_segment(
    segments: gpd.GeoDataFrame,
    composition: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Phase 2 — Adjust a segment's attributes and optionally redefine its composition."""
    term = input("  Search segment (partial from-name, to-name, or segment_name): ").strip()
    if not term:
        return segments, composition
    hits = _search_segments(segments, term)
    if hits.empty:
        print(f"  No segments matching '{term}'.")
        return segments, composition

    labels = [
        f"{r['From_Name']} → {r['To_Name']}  [{r['Segment_ID']}]"
        for _, r in hits.iterrows()
    ]
    idx = _pick_one(labels, "Segment to adjust")
    if idx is None:
        return segments, composition

    row_idx = hits.index[idx]
    row = segments.loc[row_idx]

    # Read-only context
    print(f"\n  Segment: {row.get('From_Name', '')} → {row.get('To_Name', '')}  [{row.get('Segment_ID', '')}]")
    print(f"  Route: {row.get('Route_Number', 'N/A')} — {row.get('Route_Name', 'N/A')}  [read-only]")
    print(
        f"  Formula-derived (read-only): Predominant_Speed={row.get('Predominant_Speed', 'N/A')}  "
        f"Speed_Coverage_Pct={row.get('Speed_Coverage_Pct', 'N/A')}"
    )

    pieces = composition[composition['Segment_ID'] == row['Segment_ID']]
    comp_strs = [f"{p['Engineering_Structure']} {float(p['Piece_Length']):.0f}m" for _, p in pieces.iterrows()]
    print(f"  Composition: {len(pieces)} pieces — {' / '.join(comp_strs)}")

    # Editable fields (Length, Km_Start, Km_End are geometry-derived — not editable here;
    # correct via remove + add-segment with accurate geometry)
    EDITABLE_COLS = [
        'Num_Tracks', 'Gauge', 'Electrification_Class',
        'Route_Owner', 'Average_Speed', 'TT_Stopping', 'TT_Passing', 'speed_source',
    ]
    _readonly_ctx = (f"  Length={row.get('Length', 'N/A'):.0f} m  "
                     f"Km_Start={row.get('Km_Start', 'N/A')}  "
                     f"Km_End={row.get('Km_End', 'N/A')}  [read-only]"
                     if pd.notna(row.get('Length')) else "")
    if _readonly_ctx:
        print(_readonly_ctx)

    def _cur(col):
        v = row.get(col)
        return None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)

    owner_opts = _owner_options(segments)
    changed = False

    for col in EDITABLE_COLS:
        if col not in segments.columns:
            continue
        cur = _cur(col)
        if col == 'Num_Tracks':
            val = _prompt_text('Num tracks', example='2', current=cur, cast=int)
        elif col == 'Gauge':
            val = _prompt_enum('Gauge (mm)', GAUGE_OPTIONS, current=cur)
        elif col == 'Electrification_Class':
            val = _prompt_enum('Electrification class', ELECTRIFICATION_OPTIONS, current=cur)
        elif col == 'Route_Owner':
            val = _prompt_enum('Route owner', owner_opts, current=cur)
        elif col == 'Average_Speed':
            val = _prompt_text('Average speed (km/h)', example='120', current=cur, cast=float)
        elif col in ('TT_Stopping', 'TT_Passing'):
            val = _prompt_text(col.replace('_', ' ') + ' (min)', example='2.5',
                               current=cur, cast=float)
        elif col == 'speed_source':
            val = _prompt_enum('Speed source', SPEED_SOURCE_OPTIONS,
                               current=cur or 'formula', allow_other=False)
        else:
            val = _prompt_text(col, current=cur)

        if val is not None and val != cur:
            if col == 'Num_Tracks':
                segments.at[row_idx, col] = int(val) if isinstance(val, str) else val
            elif col in ('Average_Speed', 'TT_Stopping', 'TT_Passing'):
                segments.at[row_idx, col] = float(val) if isinstance(val, str) else val
            else:
                segments.at[row_idx, col] = val
            changed = True

    if changed:
        print(f"  Segment '{row['Segment_ID']}' attributes updated.")

    # When Average_Speed changed and source is not 'design', offer to recompute TT
    new_spd = segments.at[row_idx, 'Average_Speed']
    new_src = segments.at[row_idx, 'speed_source'] if 'speed_source' in segments.columns else 'formula'
    spd_changed = changed and new_spd is not None and not (isinstance(new_spd, float) and pd.isna(new_spd))
    if spd_changed and str(new_src) in ('formula', 'estimate'):
        recompute = input(
            '  Average_Speed changed — recompute TT_Stopping/TT_Passing from new speed? (y/n) [y]: '
        ).strip().lower() or 'y'
        if recompute == 'y':
            n_sta = sum(
                1 for node_name in (row.get('From_Name'), row.get('To_Name'))
                for _, n in nodes.iterrows()
                if n.get('Name') == node_name and str(n.get('Node_Class', '')).strip() == 'station'
            )
            tt_s, tt_p = _auto_tt(float(row.get('Length', 0)), float(new_spd), n_sta)
            segments.at[row_idx, 'TT_Stopping'] = tt_s
            segments.at[row_idx, 'TT_Passing']  = tt_p
            segments.at[row_idx, 'speed_source'] = 'formula'
            print(f"  TT recomputed: TT_Stopping={tt_s} min, TT_Passing={tt_p} min.")

    # Composition update
    recomp = input("\n  Redefine composition for this segment? (y/n) [n]: ").strip().lower() or 'n'
    if recomp == 'y':
        seg_geom = segments.loc[row_idx].geometry
        seg_len  = float(segments.loc[row_idx].get('Length', seg_geom.length))
        # Drop existing composition pieces for this segment
        composition = composition[
            composition['Segment_ID'] != row['Segment_ID']
        ].reset_index(drop=True)
        new_comp_gdf = _prompt_composition_pieces(
            row['Segment_ID'],
            segments.loc[row_idx, 'From_Name'],
            segments.loc[row_idx, 'To_Name'],
            seg_geom, seg_len,
        )
        composition = pd.concat([composition, new_comp_gdf], ignore_index=True)
        print(f"  Composition updated: {len(new_comp_gdf)} piece(s).")

    return segments, composition


def _add_segment(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    composition: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Phase 2 — Add a segment."""
    term_from = input("  FROM node (Code or partial Name): ").strip()
    hits_f = _search_nodes(nodes, term_from) if term_from else gpd.GeoDataFrame()
    if hits_f.empty:
        print("  Cancel.")
        return segments, composition
    from_labels = [f"{r['Name']}  [{r['Code']}]" for _, r in hits_f.iterrows()]
    idx_f = _pick_one(from_labels, "FROM node")
    if idx_f is None: return segments, composition
    f_node = hits_f.iloc[idx_f]

    term_to = input("  TO node (Code or partial Name): ").strip()
    hits_t = _search_nodes(nodes, term_to) if term_to else gpd.GeoDataFrame()
    if hits_t.empty:
        print("  Cancel.")
        return segments, composition
    to_labels = [f"{r['Name']}  [{r['Code']}]" for _, r in hits_t.iterrows()]
    idx_t = _pick_one(to_labels, "TO node")
    if idx_t is None: return segments, composition
    t_node = hits_t.iloc[idx_t]

    if f_node['Name'] == t_node['Name']:
        print("  FROM and TO node must be different.")
        return segments, composition

    geom = LineString([(f_node['E'], f_node['N']), (t_node['E'], t_node['N'])])

    nt_raw  = _prompt_text('Num tracks', example='2', cast=int)
    gauge   = _prompt_enum('Gauge (mm)', GAUGE_OPTIONS)
    elec    = _prompt_enum('Electrification class', ELECTRIFICATION_OPTIONS)
    rn      = _prompt_text('Route number', example='750')
    rname   = _prompt_text('Route name', example='Zürich–Bern')
    owner_opts = _owner_options(segments)
    owner   = _prompt_enum('Route owner', owner_opts)
    km_s    = _prompt_text('Km start', example='0.0', cast=float)
    km_e    = _prompt_text('Km end', example='12.5', cast=float)
    avg_spd = _prompt_text('Average speed (km/h) — leave blank for mode default', cast=float)

    # Prompt for explicit TT only when user has known measured values
    tt_known_raw = input(
        '\n  Do you have known TT_Stopping and TT_Passing (min) for this segment? (y/n) [n]: '
    ).strip().lower() or 'n'
    tt_stop_manual = tt_pass_manual = None
    if tt_known_raw == 'y':
        tt_stop_manual = _prompt_text('TT_Stopping (min)', example='2.5', cast=float)
        tt_pass_manual = _prompt_text('TT_Passing (min)', example='1.8', cast=float)

    def _na(v):
        return pd.NA if v is None else v

    vals = {
        'Num_Tracks':            int(nt_raw) if isinstance(nt_raw, str) else _na(nt_raw),
        'Gauge':                 _na(gauge),
        'Electrification_Class': _na(elec),
        'Route_Number':          _na(rn),
        'Route_Name':            _na(rname),
        'Route_Owner':           _na(owner),
        'Km_Start':              float(km_s) if isinstance(km_s, str) else _na(km_s),
        'Km_End':                float(km_e) if isinstance(km_e, str) else _na(km_e),
    }

    seg_id  = f"c{f_node['Code']}_{t_node['Code']}"
    existing_seg_ids = set(segments['Segment_ID'].dropna().astype(str))
    if seg_id in existing_seg_ids:
        suffix = 2
        candidate = f"{seg_id}_{suffix}"
        while candidate in existing_seg_ids:
            suffix += 1
            candidate = f"{seg_id}_{suffix}"
        print(f"  Segment ID '{seg_id}' already exists. Using '{candidate}'.")
        seg_id = candidate
    seg_len = geom.length

    # Determine speed_source and compute TT from what the user provided
    n_sta = sum(
        1 for n in (f_node, t_node)
        if str(n.get('Node_Class', '')).strip() == 'station'
    )
    if tt_stop_manual is not None and avg_spd is not None:
        speed_source = 'design'
        avg_speed    = float(avg_spd)
        tt_stop      = float(tt_stop_manual)
        tt_pass      = float(tt_pass_manual) if tt_pass_manual is not None else tt_stop
    elif avg_spd is not None:
        speed_source     = 'formula'
        avg_speed        = float(avg_spd)
        tt_stop, tt_pass = _auto_tt(seg_len, avg_speed, n_sta)
    else:
        speed_source = 'estimate'
        avg_speed    = pd.NA  # no measured speed — keep null
        # infer mode default from gauge when available
        default_spd  = _MODE_DEFAULT_FALLBACK_VM
        if gauge is not None:
            try:
                g = int(float(gauge))
                if g <= 900:
                    default_spd = _MODE_DEFAULT_SPEEDS_VM.get('funicular', 10.0)
                elif g == 1000:
                    default_spd = _MODE_DEFAULT_SPEEDS_VM.get('tram', 30.0)
            except (ValueError, TypeError):
                pass
        tt_stop, tt_pass = _auto_tt(seg_len, default_spd, n_sta)

    fn_num = f_node.get('Number')
    tn_num = t_node.get('Number')
    seg_number = f"{int(fn_num)}_{int(tn_num)}" if pd.notna(fn_num) and pd.notna(tn_num) else pd.NA

    new_seg = {
        'Segment_ID':                    seg_id,
        'Number':                seg_number,
        'Code':                  f"{f_node['Code']}_{t_node['Code']}",
        'From_Name':             f_node['Name'],
        'To_Name':               t_node['Name'],
        'From_N':                f_node['N'],
        'From_E':                f_node['E'],
        'To_N':                  t_node['N'],
        'To_E':                  t_node['E'],
        'Length':                seg_len,
        'Num_Tracks':            vals.get('Num_Tracks', pd.NA),
        'Gauge':                 vals.get('Gauge', pd.NA),
        'Electrification_Class': vals.get('Electrification_Class', pd.NA),
        'Km_Start':              vals.get('Km_Start', pd.NA),
        'Km_End':                vals.get('Km_End', pd.NA),
        'Route_Number':          vals.get('Route_Number', pd.NA),
        'Route_Name':            vals.get('Route_Name', pd.NA),
        'Route_Owner':           vals.get('Route_Owner', pd.NA),
        'Average_Speed':         avg_speed,
        'Predominant_Speed':     avg_speed,
        'Speed_Coverage_Pct':    1.0 if pd.notna(avg_speed) else 0.0,
        'TT_Stopping':           tt_stop,
        'TT_Passing':            tt_pass,
        'speed_source':          speed_source,
        'geometry':              geom,
    }

    print("\n  Now define the segment composition:")
    new_comp_gdf = _prompt_composition_pieces(
        seg_id, f_node['Name'], t_node['Name'], geom, seg_len,
    )

    segments    = pd.concat([segments,    gpd.GeoDataFrame([new_seg], crs=SWISS_CRS)], ignore_index=True)
    composition = pd.concat([composition, new_comp_gdf], ignore_index=True)

    print(f"  Added segment '{seg_id}' with {len(new_comp_gdf)} composition piece(s).")
    return segments, composition


# =============================================================================
# Composition operations
# =============================================================================

def _edit_composition(
    segments: gpd.GeoDataFrame,
    composition: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Phase 3 — Interactively add, remove, or edit composition pieces for a segment."""
    term = input("  Search segment (partial From_Name, To_Name, or ID): ").strip()
    if not term:
        return composition
    hits = _search_segments(segments, term)
    if hits.empty:
        print(f"  No segments matching '{term}'.")
        return composition

    labels = [
        f"{r['From_Name']} → {r['To_Name']}  [{r['Segment_ID']}]"
        for _, r in hits.iterrows()
    ]
    idx = _pick_one(labels, "Segment to edit composition")
    if idx is None:
        return composition

    seg_row = hits.iloc[idx]
    seg_id  = seg_row['Segment_ID']
    seg_geom = seg_row.geometry
    seg_len  = float(seg_row.get('Length', seg_geom.length if seg_geom else 0))

    COMP_COLS = ['Engineering_Structure', 'Piece_Length', 'Edge_Level', 'Under_Construction']

    while True:
        pieces = composition[composition['Segment_ID'] == seg_id].copy()
        total_comp = float(pieces['Piece_Length'].sum()) if not pieces.empty else 0.0

        print(f"\n  Segment: {seg_row['From_Name']} → {seg_row['To_Name']}  "
              f"(segment length: {seg_len:.0f} m)")
        if pieces.empty:
            print("  Composition: (no pieces)")
        else:
            print(f"  Composition ({len(pieces)} pieces, {total_comp:.0f} m total):")
            for i, (_, p) in enumerate(pieces.iterrows(), 1):
                vals = "  ".join(f"{c}={p.get(c, '')}" for c in COMP_COLS)
                print(f"    {i})  {vals}")

        print("\n    1) Add a piece")
        print("    2) Edit a piece")
        print("    r) Replace all pieces (redefine from scratch)")
        print("    d) Done")
        c = input("  Select (1/2/r/d): ").strip().lower()

        if c == 'd':
            break

        elif c == '1':
            ct = _prompt_enum('Construct type', CONSTRUCT_TYPE_OPTIONS, required=True,
                              allow_other=False)
            pl_raw = _prompt_text('Piece length (m)', example='500', cast=float)
            if pl_raw is None:
                print("  Piece length required. Cancelled.")
                continue
            pl = float(pl_raw) if isinstance(pl_raw, str) else pl_raw
            el_opts = _EDGE_LEVEL_BY_CT.get(ct, ['1'])
            el_raw = _prompt_enum('Edge level', el_opts, current=el_opts[0], allow_other=False)
            el = int(el_raw) if el_raw is not None else int(el_opts[0])
            uc_raw = _prompt_enum('Under construction', ['0 — no', '1 — yes'],
                                  current='0 — no', allow_other=False)
            uc = 1 if uc_raw and uc_raw.startswith('1') else 0

            new_piece = gpd.GeoDataFrame([{
                'Segment_ID':            seg_id,
                'From_Name':             seg_row['From_Name'],
                'To_Name':               seg_row['To_Name'],
                'Engineering_Structure': ct,
                'Edge_Level':            el,
                'Under_Construction':    uc,
                'Piece_Length':          pl,
                'geometry':              seg_geom,
            }], crs=SWISS_CRS)
            composition = pd.concat([composition, new_piece], ignore_index=True)
            print(f"  Added piece: {ct}  {pl:.0f} m.")
            new_total = float(composition[composition['Segment_ID'] == seg_id]['Piece_Length'].sum())
            if abs(new_total - seg_len) > 0.5:
                print(f"  Budget: {new_total:.0f} m / {seg_len:.0f} m  "
                      f"(delta = {new_total - seg_len:+.0f} m).")

        elif c == '2':
            if pieces.empty:
                print("  No pieces to edit.")
                continue
            piece_labels = [
                f"{p['Engineering_Structure']}  {float(p['Piece_Length']):.0f} m"
                for _, p in pieces.iterrows()
            ]
            pidx = _pick_one(piece_labels, "Piece to edit")
            if pidx is None:
                continue
            edit_loc = pieces.index[pidx]
            p = composition.loc[edit_loc]

            def _pcur(col):
                v = p.get(col)
                return None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)

            es = _prompt_enum('Construct type', CONSTRUCT_TYPE_OPTIONS,
                              current=_pcur('Engineering_Structure'), allow_other=False)
            if es is not None and es != _pcur('Engineering_Structure'):
                composition.at[edit_loc, 'Engineering_Structure'] = es

            pl_val = _prompt_text('Piece length (m)', example='500',
                                  current=_pcur('Piece_Length'), cast=float)
            if pl_val is not None:
                composition.at[edit_loc, 'Piece_Length'] = (
                    float(pl_val) if isinstance(pl_val, str) else pl_val
                )

            effective_ct = es if es is not None else _pcur('Engineering_Structure')
            el_opts = _EDGE_LEVEL_BY_CT.get(effective_ct or '', ['1'])
            el_val = _prompt_enum('Edge level', el_opts,
                                  current=_pcur('Edge_Level') or el_opts[0], allow_other=False)
            if el_val is not None:
                composition.at[edit_loc, 'Edge_Level'] = int(el_val)

            uc_cur = '1 — yes' if str(p.get('Under_Construction', 0)) == '1' else '0 — no'
            uc_val = _prompt_enum('Under construction', ['0 — no', '1 — yes'],
                                  current=uc_cur, allow_other=False)
            if uc_val is not None and uc_val != uc_cur:
                composition.at[edit_loc, 'Under_Construction'] = (
                    1 if uc_val.startswith('1') else 0
                )
            print("  Piece updated.")
            new_total = float(composition[composition['Segment_ID'] == seg_id]['Piece_Length'].sum())
            if abs(new_total - seg_len) > 0.5:
                delta = new_total - seg_len
                print(f"  Budget: {new_total:.0f} m / {seg_len:.0f} m  (delta = {delta:+.0f} m).")
                other_pieces = composition[
                    (composition['Segment_ID'] == seg_id) & (composition.index != edit_loc)
                ]
                if not other_pieces.empty:
                    print(f"  Select a piece to absorb {-delta:+.0f} m (Enter to skip):")
                    absorb_labels = [
                        f"{p['Engineering_Structure']}  "
                        f"{float(p['Piece_Length']):.0f} m → {float(p['Piece_Length']) - delta:.0f} m"
                        for _, p in other_pieces.iterrows()
                    ]
                    absorb_idx = _pick_one(absorb_labels, "Absorb in")
                    if absorb_idx is not None:
                        absorb_loc  = other_pieces.index[absorb_idx]
                        new_abs_len = float(composition.at[absorb_loc, 'Piece_Length']) - delta
                        if new_abs_len > 0:
                            composition.at[absorb_loc, 'Piece_Length'] = new_abs_len
                            print(f"  Piece adjusted to {new_abs_len:.0f} m — budget closed.")
                        else:
                            print(f"  Cannot absorb: would result in {new_abs_len:.0f} m. Keeping as-is.")

        elif c == 'r':
            composition = composition[
                composition['Segment_ID'] != seg_id
            ].reset_index(drop=True)
            new_comp_gdf = _prompt_composition_pieces(
                seg_id, seg_row['From_Name'], seg_row['To_Name'],
                seg_geom, seg_len,
            )
            composition = pd.concat([composition, new_comp_gdf], ignore_index=True)
            print(f"  Composition replaced: {len(new_comp_gdf)} piece(s).")

        else:
            print("  Invalid choice.")

    return composition


# =============================================================================
# Import operations
# =============================================================================

def _import_nodes(
    current_nodes: gpd.GeoDataFrame,
    current_segs: gpd.GeoDataFrame,
    current_comp: gpd.GeoDataFrame,
    current_version: str,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Import and update nodes from another version.

    Shows three diff categories:
      New     — in source, not in current → offer to add; if the node lands
                within _NODE_SNAP_TOLERANCE_M of an existing segment, offers
                to split that segment via _split_segment_at.
      Changed — in both but attributes differ → offer to replace
      Removed — in current, not in source → offer to delete from current
    """
    infra_root = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR
    versions = [v for v in list_versions(infra_root) if v != current_version]
    if not versions:
        print("  No other versions available to import from.")
        return current_nodes, current_segs, current_comp
    idx = _pick_one(versions, "Version to import nodes from")
    if idx is None:
        return current_nodes, current_segs, current_comp
    source_version = versions[idx]

    print(f"  Loading nodes from '{source_version}'...")
    source_nodes = gpd.read_file(infra_root / source_version / 'nodes.gpkg').reset_index(drop=True)

    # diff_items: (source_idx_or_None, status, current_idx_or_None, description)
    diff_items = []

    def get_node_key(row):
        bpn = str(row.get('Number', ''))
        name = str(row.get('Name', ''))
        if bpn not in ('None', 'nan', '<NA>', ''):
            return f"Number_{bpn}"
        return f"Name_{name}"

    curr_keys = current_nodes.apply(get_node_key, axis=1)
    source_key_set = set(source_nodes.apply(get_node_key, axis=1))

    # --- New / Changed (source → current) ------------------------------------
    for i, s_row in source_nodes.iterrows():
        s_key = get_node_key(s_row)
        match_idx = current_nodes.index[curr_keys == s_key].tolist()

        if not match_idx:
            diff_items.append((i, "New", None,
                               f"New node '{s_row.get('Name', 'Unknown')}'"))
        else:
            c_idx = match_idx[0]
            c_row = current_nodes.loc[c_idx]

            changes = []
            for col in ['E', 'N', 'Node_Class', 'Transport_Mode', 'Platform_Count', 'Name']:
                s_val = s_row.get(col)
                c_val = c_row.get(col)
                if pd.isna(s_val) and pd.isna(c_val):
                    continue
                if str(s_val) != str(c_val):
                    changes.append(f"{col}: {c_val} -> {s_val}")

            if not s_row.geometry.equals(c_row.geometry):
                changes.append("geometry changed")

            if changes:
                diff_items.append((i, "Changed", c_idx,
                                   f"Update node '{s_row.get('Name', 'Unknown')}' ({', '.join(changes)})"))

    # --- Removed (in current, absent from source) ----------------------------
    for i, c_row in current_nodes.iterrows():
        if get_node_key(c_row) not in source_key_set:
            diff_items.append((None, "Removed", i,
                               f"Remove node '{c_row.get('Name', 'Unknown')}' (not in source)"))

    if not diff_items:
        print("  No differences found between current version and source.")
        return current_nodes, current_segs, current_comp

    print("\n  Available imports:")
    for j, (_, status, _, desc) in enumerate(diff_items, 1):
        print(f"    {j}) [{status}] {desc}")

    ans = input("\n  Enter numbers to apply (e.g. 1,3,21-25), 'all', or Enter to cancel: ").strip()
    if not ans:
        return current_nodes, current_segs, current_comp

    selected_indices = _parse_selection(ans, len(diff_items))

    if not selected_indices:
        print("  No valid choices selected.")
        return current_nodes, current_segs, current_comp

    new_rows = []
    new_node_source_rows = []  # (s_row) for "New" nodes only — checked for segment snap
    drop_indices = []

    for idx_in_diff in selected_indices:
        s_idx, status, c_idx, desc = diff_items[idx_in_diff]
        if status == "Removed":
            drop_indices.append(c_idx)
        else:
            s_row = source_nodes.loc[s_idx]
            new_rows.append(s_row)
            if status == "New":
                new_node_source_rows.append(s_row)
            if status == "Changed" and c_idx is not None:
                drop_indices.append(c_idx)

    if drop_indices:
        current_nodes = current_nodes.drop(index=drop_indices)

    if new_rows:
        current_nodes = pd.concat(
            [current_nodes, gpd.GeoDataFrame(new_rows, crs=SWISS_CRS)],
            ignore_index=True
        )

    # --- Snap new nodes onto segments ----------------------------------------
    for s_row in new_node_source_rows:
        node_name  = str(s_row.get('Name', ''))
        node_code  = str(s_row.get('Code', node_name[:4].upper()))
        node_class = str(s_row.get('Node_Class', ''))
        node_pt    = s_row.geometry

        dists      = current_segs.geometry.distance(node_pt)
        nearest_idx  = dists.idxmin()
        nearest_dist = dists[nearest_idx]

        if nearest_dist > _NODE_SNAP_TOLERANCE_M:
            continue

        seg_row = current_segs.loc[nearest_idx]
        print(f"\n  Node '{node_name}' is {nearest_dist:.1f} m from segment "
              f"'{seg_row['From_Name']} → {seg_row['To_Name']}' [{seg_row['Segment_ID']}].")
        ans2 = input("  Split this segment at the node? (y/n) [y]: ").strip().lower() or 'y'
        if ans2 != 'y':
            continue

        split_dist = seg_row.geometry.project(node_pt)
        current_segs, current_comp = _split_segment_at(
            current_segs, current_comp, nearest_idx, split_dist,
            node_name, node_code, current_nodes, node_class,
        )
        print(f"  Segment split at {split_dist:.0f} m.")

    print(f"  Applied {len(selected_indices)} change(s).")
    return current_nodes, current_segs, current_comp


def _import_segments(
    current_segs: gpd.GeoDataFrame,
    current_comp: gpd.GeoDataFrame,
    current_version: str,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Import and update segments (and their composition) from another version.

    Shows three diff categories:
      New     — in source, not in current → offer to add
      Changed — in both but attributes differ → offer to replace
      Removed — in current, not in source → offer to delete from current
    """
    infra_root = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR
    versions = [v for v in list_versions(infra_root) if v != current_version]
    if not versions:
        print("  No other versions available to import from.")
        return current_segs, current_comp
    idx = _pick_one(versions, "Version to import segments from")
    if idx is None:
        return current_segs, current_comp
    source_version = versions[idx]

    print(f"  Loading segments and composition from '{source_version}'...")
    source_dir = infra_root / source_version
    source_segs = gpd.read_file(source_dir / 'segments.gpkg').reset_index(drop=True)
    if (source_dir / 'segments_composition.gpkg').exists():
        source_comp = gpd.read_file(source_dir / 'segments_composition.gpkg').reset_index(drop=True)
    else:
        source_comp = gpd.GeoDataFrame(columns=current_comp.columns)

    # diff_items: (source_idx_or_None, status, current_idx_or_None, description)
    diff_items = []

    def get_seg_key(row):
        return str(row.get('Segment_ID', ''))

    curr_keys = current_segs.apply(get_seg_key, axis=1)
    source_key_set = set(source_segs.apply(get_seg_key, axis=1))

    # --- New / Changed (source → current) ------------------------------------
    for i, s_row in source_segs.iterrows():
        s_key = get_seg_key(s_row)
        match_idx = current_segs.index[curr_keys == s_key].tolist()

        if not match_idx:
            desc = (f"New segment '{s_row.get('Segment_ID', '')}' "
                    f"({s_row.get('From_Name', '')} -> {s_row.get('To_Name', '')})")
            diff_items.append((i, "New", None, desc))
        else:
            c_idx = match_idx[0]
            c_row = current_segs.loc[c_idx]

            changes = []
            for col in [
                'Num_Tracks', 'Gauge', 'Electrification_Class',
                'Length', 'Km_Start', 'Km_End', 'From_Name', 'To_Name',
                'Average_Speed', 'TT_Stopping', 'TT_Passing',
                'speed_source', 'Transport_Mode', 'Route_Owner',
            ]:
                s_val = s_row.get(col)
                c_val = c_row.get(col)
                s_na = s_val is None or (isinstance(s_val, float) and pd.isna(s_val))
                c_na = c_val is None or (isinstance(c_val, float) and pd.isna(c_val))
                if s_na and c_na:
                    continue
                if str(s_val) != str(c_val):
                    changes.append(col)

            if not s_row.geometry.equals(c_row.geometry):
                changes.append("geometry")

            s_c = source_comp[source_comp['Segment_ID'] == s_row.get('Segment_ID', '')]
            c_c = current_comp[current_comp['Segment_ID'] == c_row.get('Segment_ID', '')]
            if len(s_c) != len(c_c):
                changes.append("composition count")

            if changes:
                desc = (f"Update segment '{s_row.get('Segment_ID', '')}' "
                        f"({s_row.get('From_Name', '')} -> {s_row.get('To_Name', '')}) "
                        f"[{', '.join(changes)}]")
                diff_items.append((i, "Changed", c_idx, desc))

    # --- Removed (in current, absent from source) ----------------------------
    for i, c_row in current_segs.iterrows():
        if get_seg_key(c_row) not in source_key_set:
            desc = (f"Remove segment '{c_row.get('Segment_ID', '')}' "
                    f"({c_row.get('From_Name', '')} -> {c_row.get('To_Name', '')})")
            diff_items.append((None, "Removed", i, desc))

    if not diff_items:
        print("  No differences found between current version and source.")
        return current_segs, current_comp

    print("\n  Available imports:")
    for j, (_, status, _, desc) in enumerate(diff_items, 1):
        print(f"    {j}) [{status}] {desc}")

    ans = input("\n  Enter numbers to apply (e.g. 1,3,21-25), 'all', or Enter to cancel: ").strip()
    if not ans:
        return current_segs, current_comp

    selected_indices = _parse_selection(ans, len(diff_items))

    if not selected_indices:
        print("  No valid choices selected.")
        return current_segs, current_comp

    new_seg_rows = []
    new_comp_rows = []
    drop_seg_indices = []
    drop_comp_seg_ids = []

    for idx_in_diff in selected_indices:
        s_idx, status, c_idx, desc = diff_items[idx_in_diff]

        if status == "Removed":
            drop_seg_indices.append(c_idx)
            drop_comp_seg_ids.append(current_segs.at[c_idx, 'Segment_ID'])
        else:
            s_row = source_segs.loc[s_idx]
            s_id = s_row.get('Segment_ID')
            new_seg_rows.append(s_row)
            if s_id:
                s_comp_pieces = source_comp[source_comp['Segment_ID'] == s_id]
                if not s_comp_pieces.empty:
                    new_comp_rows.extend(s_comp_pieces.to_dict('records'))
            if status == "Changed" and c_idx is not None:
                drop_seg_indices.append(c_idx)
                drop_comp_seg_ids.append(current_segs.at[c_idx, 'Segment_ID'])

    if drop_seg_indices:
        current_segs = current_segs.drop(index=drop_seg_indices)
    if drop_comp_seg_ids:
        current_comp = current_comp[~current_comp['Segment_ID'].isin(drop_comp_seg_ids)]

    if new_seg_rows:
        current_segs = pd.concat(
            [current_segs, gpd.GeoDataFrame(new_seg_rows, crs=SWISS_CRS)],
            ignore_index=True
        )
    if new_comp_rows:
        geoms = [r.pop('geometry', None) for r in new_comp_rows]
        new_comp_gdf = gpd.GeoDataFrame(new_comp_rows, geometry=geoms, crs=SWISS_CRS)
        current_comp = pd.concat([current_comp, new_comp_gdf], ignore_index=True)

    print(f"  Applied {len(selected_indices)} change(s).")
    return current_segs, current_comp


def _adjust_network_speed_source(
    segments: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Set speed_source for all segments in the network (or a filtered subset)."""
    if 'speed_source' not in segments.columns:
        segments['speed_source'] = 'formula'

    counts = segments['speed_source'].value_counts().to_dict()
    print(f"\n  Current speed_source distribution:")
    for val in ('OSM', 'gtfs', 'infra'):
        print(f"    {val}: {counts.get(val, 0)} segment(s)")

    print("\n  Set speed_source to:")
    print("    1) formula — OSM/formula-derived; open to GTFS calibration")
    print("    2) gtfs    — timetable-calibrated; protected from formula re-calibration")
    print("    3) design  — designer-set; locked unconditionally (no GTFS override)")
    src_choice = input("  Select (1/2/3): ").strip()
    target_map = {'1': 'formula', '2': 'gtfs', '3': 'design'}
    if src_choice not in target_map:
        print("  Cancelled.")
        return segments
    target_source = target_map[src_choice]

    print(f"\n  Apply to:")
    print(f"    1) All {len(segments)} segments")
    print(f"    2) By Transport_Mode")
    print(f"    3) By Segment_ID list")
    scope = input("  Select (1/2/3): ").strip()

    if scope == '1':
        target_mask = pd.Series(True, index=segments.index)
    elif scope == '2':
        mode_term = input("  Transport_Mode filter (e.g. 'train', 'tram'): ").strip()
        target_mask = (
            segments.get('Transport_Mode', pd.Series('', index=segments.index))
            .fillna('').str.lower().str.contains(mode_term.lower(), regex=False)
        )
    elif scope == '3':
        raw_ids = input("  Segment_IDs (comma-separated): ").strip()
        id_set = {s.strip() for s in raw_ids.split(',') if s.strip()}
        target_mask = segments['Segment_ID'].isin(id_set)
    else:
        print("  Cancelled.")
        return segments

    n_target = int(target_mask.sum())
    if n_target == 0:
        print("  No matching segments found.")
        return segments

    ans = input(f"  Set speed_source='{target_source}' for {n_target} segment(s)? (y/n) [n]: "
                ).strip().lower() or 'n'
    if ans != 'y':
        print("  Cancelled.")
        return segments

    segments.loc[target_mask, 'speed_source'] = target_source
    print(f"  Updated {n_target} segment(s) to speed_source='{target_source}'.")
    return segments


# =============================================================================
# Pre-save validation and auto-fill
# =============================================================================

def _validate_and_autofill(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    composition: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, bool]:
    """Delegates to infrabuild_network_builder.validate_and_autofill (interactive)."""
    return _validate_and_autofill_core(nodes, segments, composition, interactive=True)


def _apply_seed_interactive(
    nodes: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    composition: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Pick a seed year from data/Infrastructure/Seeds/ and import its elements.

    Seeds are read-only — the version manager only imports FROM them; it never
    writes back. Files are authored by hand (QGIS or scripted).

    Available years are discovered from nodes_<year>.gpkg files in the Seeds
    directory.
    """
    seeds_dir = Path(paths.MAIN) / paths.INFRASTRUCTURE_SEEDS_DIR
    if not seeds_dir.is_dir():
        print(f"\n  Seeds directory does not exist: {seeds_dir}")
        return nodes, segments, composition

    years = sorted(
        p.name.removeprefix('nodes_').removesuffix('.gpkg')
        for p in seeds_dir.glob('nodes_*.gpkg')
    )
    if not years:
        print(f"\n  No nodes_<year>.gpkg seed files found in {seeds_dir}")
        return nodes, segments, composition

    idx = _pick_one(years, "Seed year to apply")
    if idx is None:
        return nodes, segments, composition

    chosen_year = years[idx]
    print(f"\n  Importing seed for year '{chosen_year}' (read-only)…")
    return ic.load_and_apply_seed(chosen_year, nodes, segments, composition)



# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Infrastructure Version Manager')
    parser.add_argument('--create-from', default=None,
                        help='Base version to copy from (skips interactive Q1/Q2)')
    parser.add_argument('--name', default=None,
                        help='Name for the new version (skips interactive Q3)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Auto-answer yes to the overwrite prompt')
    args, _ = parser.parse_known_args()

    try:
        nodes, segments, composition, name, source_version, out_dir, mode = _run_phase0(
            create_from=args.create_from,
            preset_name=args.name,
            auto_overwrite=args.overwrite,
        )
    except SystemExit:
        return

    phase = 1  # 1 = nodes, 2 = segments, 3 = composition, 4 = save

    while phase <= 4:

        # ── Phase 1: node editing ────────────────────────────────────────────
        if phase == 1:
            while True:
                print("\n" + "─" * 60)
                print("  Phase 1 — Node editing")
                print("    1) Remove a node")
                print("    2) Adjust a node")
                print("    3) Add a node")
                print("    4) Import nodes from another version")
                print("    5) Apply a seed file (import-only, read-only on seed)")
                print("    6) Proceed to segment editing  →")
                c = input("  Select (1-6): ").strip()

                if c == '1':
                    nodes, segments, composition = _remove_node(nodes, segments, composition)
                elif c == '2':
                    nodes = _adjust_node(nodes)
                elif c == '3':
                    nodes, segments, composition = _add_node(nodes, segments, composition)
                elif c == '4':
                    nodes, segments, composition = _import_nodes(nodes, segments, composition, name)
                elif c == '5':
                    nodes, segments, composition = _apply_seed_interactive(
                        nodes, segments, composition,
                    )
                elif c == '6':
                    phase = 2
                    break
                else:
                    print("  Invalid choice.")

        # ── Phase 2: segment editing ─────────────────────────────────────────
        elif phase == 2:
            while True:
                print("\n" + "─" * 60)
                print("  Phase 2 — Segment editing")
                print("    1) Remove a segment")
                print("    2) Adjust a segment")
                print("    3) Add a segment")
                print("    4) Import segments from another version")
                print("    5) Adjust speed source (bulk re-tier segments)")
                print("    6) Proceed to composition editing  →")
                print("    7) ← Back to node editing")
                c = input("  Select (1-7): ").strip()

                if c == '1':
                    segments, composition, nodes = _remove_segment(segments, composition, nodes)
                elif c == '2':
                    segments, composition = _adjust_segment(segments, composition, nodes)
                elif c == '3':
                    segments, composition = _add_segment(nodes, segments, composition)
                elif c == '4':
                    segments, composition = _import_segments(segments, composition, name)
                elif c == '5':
                    segments = _adjust_network_speed_source(segments)
                elif c == '6':
                    phase = 3
                    break
                elif c == '7':
                    phase = 1
                    break
                else:
                    print("  Invalid choice.")

        # ── Phase 3: composition editing ─────────────────────────────────────
        elif phase == 3:
            while True:
                print("\n" + "─" * 60)
                print("  Phase 3 — Composition editing")
                print("    1) Edit composition for a segment")
                print("    2) Save and close")
                print("    3) ← Back to segment editing")
                c = input("  Select (1-3): ").strip()

                if c == '1':
                    composition = _edit_composition(segments, composition)
                elif c == '2':
                    nodes, segments, composition, ok = _validate_and_autofill(
                        nodes, segments, composition
                    )
                    if not ok:
                        continue
                    print("\n  Saving GeoPackages...")
                    nodes.to_file(out_dir / 'nodes.gpkg', driver='GPKG')
                    segments.to_file(out_dir / 'segments.gpkg', driver='GPKG')
                    composition.to_file(out_dir / 'segments_composition.gpkg', driver='GPKG')
                    print(f"  Done. Version '{name}' saved.")
                    phase = 5  # exit outer loop
                    break
                elif c == '3':
                    phase = 2
                    break
                else:
                    print("  Invalid choice.")

        # ── Phase 4: save ────────────────────────────────────────────────────
        elif phase == 4:
            print("\n" + "─" * 60)
            print(f"  Version: {name}")
            print(f"  Based on: {source_version}")
            print(f"  Nodes:    {len(nodes)}")
            print(f"  Segments: {len(segments)}")
            print(f"  Composition pieces: {len(composition)}")
            print(f"  Output: {out_dir}")

            print("\n  1) Save")
            print("  2) Discard changes")
            print("  3) ← Back to composition editing")
            while True:
                ans = input("  Select (1-3): ").strip()
                if ans in ('1', '2', '3'):
                    break
                print("  Enter 1, 2, or 3.")
            if ans == '3':
                phase = 3
                continue
            if ans == '1':
                nodes, segments, composition, ok = _validate_and_autofill(
                    nodes, segments, composition
                )
                if not ok:
                    continue
                print("  Saving GeoPackages...")
                nodes.to_file(out_dir / 'nodes.gpkg', driver='GPKG')
                segments.to_file(out_dir / 'segments.gpkg', driver='GPKG')
                composition.to_file(out_dir / 'segments_composition.gpkg', driver='GPKG')
                print(f"  Done. Version '{name}' saved.")
            else:
                print("  Discarded changes.")
            break

if __name__ == '__main__':
    main()

# =============================================================================
# Derived infrastructure versions (Topic 2)
# Builds and caches infra versions derived from a named base by applying a
# sorted set of infra-intervention IDs. Naming: <BASE>+<id1>+<id2>+...
# =============================================================================

SUPPORTED_INFRA_INT_TYPES: Tuple[str, ...] = ('cc', 'cap')


def list_intervention_ids(int_type: str) -> List[str]:
    """Return all distinct int_ids found in <int_type>_interventions.gpkg.

    Args:
        int_type: registry short code (e.g. 'cc', 'cap').

    Returns:
        Sorted list of int_id strings. Empty list if the registry file or its
        layers are missing.
    """
    gpkg_path = Path(paths.get_infra_int_registry(int_type))
    if not gpkg_path.exists():
        return []
    try:
        layers = set(fiona.listlayers(str(gpkg_path)))
    except Exception as exc:
        print(f"  [registry] {gpkg_path.name}: cannot list layers ({exc})")
        return []

    ids: set = set()
    for layer in ('nodes', 'segments'):
        if layer not in layers:
            continue
        gdf = gpd.read_file(gpkg_path, layer=layer)
        if 'int_id' in gdf.columns:
            ids.update(gdf['int_id'].dropna().astype(str).tolist())
    return sorted(ids)


def read_intervention_rows(
    int_type: str,
    int_ids: List[str],
) -> Tuple[Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]:
    """Return (nodes, segments, composition) GeoDataFrames filtered to int_ids.

    Each returned frame may be None if the corresponding layer does not exist
    or has no rows for the requested IDs.
    """
    gpkg_path = Path(paths.get_infra_int_registry(int_type))
    if not gpkg_path.exists() or not int_ids:
        return None, None, None

    layers = set(fiona.listlayers(str(gpkg_path)))
    wanted = set(str(i) for i in int_ids)

    def _filter(layer: str) -> Optional[gpd.GeoDataFrame]:
        if layer not in layers:
            return None
        gdf = gpd.read_file(gpkg_path, layer=layer)
        if 'int_id' not in gdf.columns:
            return None
        sel = gdf[gdf['int_id'].astype(str).isin(wanted)].reset_index(drop=True)
        return sel if not sel.empty else None

    return _filter('nodes'), _filter('segments'), _filter('segments_composition')


def derived_version_name(base_version_name: str, infra_int_ids: List[str]) -> str:
    """Compute the verbose derived-version folder name.

    Empty int list → returns base name unchanged (no derived version needed).
    """
    if not infra_int_ids:
        return base_version_name
    suffix = '+'.join(sorted(str(i) for i in infra_int_ids))
    return f"{base_version_name}+{suffix}"


def resolve_or_build(
    base_version_name: str,
    infra_int_ids: List[str],
) -> str:
    """Return the name of a derived infra version with the given ints applied.

    Reuses an existing folder if one already exists; otherwise builds the
    derived version by copying the base and applying each int in turn.

    Args:
        base_version_name: existing infra version (e.g. 'AS_2026_ZH').
        infra_int_ids: list of registry int IDs to apply (any combination of
            'cc_*', 'cap_*', ...). Sorted alphabetically for naming.

    Returns:
        Name of the derived infra version on disk.
    """
    name = derived_version_name(base_version_name, infra_int_ids)
    target_dir = Path(paths.get_derived_infra_version_dir(name))

    if target_dir.is_dir() and paths.derived_version_exists(name):
        print(f"  [derived] reusing existing '{name}'")
        return name

    base_dir = Path(paths.get_infra_version_dir(base_version_name))
    if not paths.infra_version_exists(base_version_name):
        raise FileNotFoundError(
            f"Base infra version '{base_version_name}' not found at {base_dir}"
        )

    print(f"  [derived] building '{name}' from '{base_version_name}'")
    nodes = gpd.read_file(base_dir / 'nodes.gpkg')
    segs  = gpd.read_file(base_dir / 'segments.gpkg')
    comp_path = base_dir / 'segments_composition.gpkg'
    comp = (gpd.read_file(comp_path)
            if comp_path.exists()
            else gpd.GeoDataFrame(columns=['Segment_ID'], crs=ic.SWISS_CRS))

    by_type: Dict[str, List[str]] = {}
    for iid in infra_int_ids:
        prefix = str(iid).split('_', 1)[0].lower()
        by_type.setdefault(prefix, []).append(str(iid))

    for int_type, ids in by_type.items():
        print(f"  [derived]   applying {len(ids)} {int_type}-int(s): {ids}")
        new_nodes, new_segs, new_comp = read_intervention_rows(int_type, ids)
        if new_nodes is not None and not new_nodes.empty:
            nodes, segs, comp = ic.merge_nodes(
                nodes, new_nodes, segs, comp,
                snap_and_split=True,
                replace_existing=(int_type == 'cap'),
            )
        if new_segs is not None and not new_segs.empty:
            segs, comp = ic.merge_segments(segs, comp, new_segs, new_comp)

    if '_delete' in segs.columns:
        segs = segs[~segs['_delete'].fillna(False)].drop(columns='_delete').reset_index(drop=True)

    nodes, segs, comp, ok = ic.validate_and_autofill(
        nodes, segs, comp, interactive=False,
    )
    if not ok:
        print(f"  [derived] validation reported blockers; '{name}' NOT saved.")
        raise RuntimeError(f"Validation failed building derived version '{name}'")

    target_dir.mkdir(parents=True, exist_ok=True)
    nodes.to_file(target_dir / 'nodes.gpkg', driver='GPKG')
    segs.to_file(target_dir / 'segments.gpkg', driver='GPKG')
    if not comp.empty:
        comp.to_file(target_dir / 'segments_composition.gpkg', driver='GPKG')
    print(f"  [derived] saved to {target_dir}")
    return name


def build_full_reference(base_version_name: str) -> str:
    """Build a 'full' reference version with ALL ints from ALL registries applied.

    Stored under data/Infrastructure/Developments/Dev_Full/<base>_full/.
    For visualisation and sanity checking only — not used by the main pipeline.
    """
    all_ids: List[str] = []
    for int_type in SUPPORTED_INFRA_INT_TYPES:
        all_ids.extend(list_intervention_ids(int_type))

    full_name = f"{base_version_name}_full"
    target_dir = Path(paths.MAIN) / paths.INFRASTRUCTURE_DEV_FULL_DIR / full_name
    target_dir.mkdir(parents=True, exist_ok=True)

    if not all_ids:
        print(f"  [full] no infra ints registered; '{full_name}' will mirror base.")
    else:
        print(f"  [full] applying {len(all_ids)} infra int(s): {all_ids}")

    derived_name = resolve_or_build(base_version_name, all_ids)
    derived_dir = Path(paths.get_derived_infra_version_dir(derived_name))

    for fname in ('nodes.gpkg', 'segments.gpkg', 'segments_composition.gpkg'):
        src = derived_dir / fname
        if src.exists():
            (target_dir / fname).write_bytes(src.read_bytes())

    qgz_path = target_dir / f"{full_name}.qgz"
    ic._build_infra_qgz(
        qgz_path=str(qgz_path),
        version_dir=target_dir,
        name=full_name,
    )
    print(f"  [full] copied to {target_dir}")
    return full_name


# ─────────────────────────────────────────────────────────────────────────────
# Cap-intervention registry write helper (called by the capacity workflow)
# ─────────────────────────────────────────────────────────────────────────────

def append_cap_intervention_rows(
    new_nodes: Optional[gpd.GeoDataFrame],
    new_segments: Optional[gpd.GeoDataFrame],
    new_composition: Optional[gpd.GeoDataFrame] = None,
) -> None:
    """Append auto-generated Tier-1 cap-intervention rows to cap_interventions.gpkg.

    Each row must carry an 'int_id' column (e.g. 'cap_ps_017'). The gpkg is
    created if missing; existing rows with the same int_id are replaced.
    """
    gpkg_path = Path(paths.get_infra_int_registry('cap'))
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)

    existing_layers = set()
    if gpkg_path.exists():
        try:
            existing_layers = set(fiona.listlayers(str(gpkg_path)))
        except Exception:
            pass

    def _append(layer_name: str, new_gdf: Optional[gpd.GeoDataFrame]) -> None:
        if new_gdf is None or new_gdf.empty:
            return
        if layer_name in existing_layers:
            existing = gpd.read_file(gpkg_path, layer=layer_name)
            if 'int_id' in existing.columns and 'int_id' in new_gdf.columns:
                new_ids = set(new_gdf['int_id'].dropna().astype(str).tolist())
                existing = existing[~existing['int_id'].astype(str).isin(new_ids)]
            combined = pd.concat([existing, new_gdf], ignore_index=True)
            combined = gpd.GeoDataFrame(combined, crs=ic.SWISS_CRS)
        else:
            combined = new_gdf
        combined.to_file(gpkg_path, layer=layer_name, driver='GPKG')

    _append('nodes', new_nodes)
    _append('segments', new_segments)
    _append('segments_composition', new_composition)
    print(f"  [cap-registry] appended rows to {gpkg_path.name}")


SVC_INT_TYPES: Tuple[str, ...] = ('ext', 'cc')


def resolve_active_svc_int_types() -> List[str]:
    """Resolve which svc-int types are active under settings.INFRA_INT_MODE.

    Mode values (mirrors old main_cap.infra_generation_modification_type):
      'NONE' → []
      'ALL'  → every type in SVC_INT_TYPES
      'EXT'  → ['ext']
      'CC'   → ['cc']

    Returns sorted lower-case short codes. Unknown modes log a warning and
    return [] (baseline).
    """
    mode = str(getattr(settings, 'INFRA_INT_MODE', 'NONE')).upper()
    if mode == 'NONE':
        return []
    if mode == 'ALL':
        return list(SVC_INT_TYPES)
    if mode.lower() in SVC_INT_TYPES:
        return [mode.lower()]
    print(f"  [intervention] unknown INFRA_INT_MODE='{mode}' — treating as 'NONE'")
    return []


def enumerate_active_infra_ints() -> List[str]:
    """Collect all infra-int IDs implied by the active svc-int types.

    Resolution:
      1. SVC_INT_MODE / SVC_INT_SELECTED → list of active svc-int types.
      2. For each active type that has its own infra registry (e.g. 'cc'),
         add every int_id from that registry.
      3. For each active type that has an svc-only registry (e.g. 'ext'),
         read the registry's 'requires_infra' column and add the listed IDs.

    Cap ints are NOT enumerated here — they are auto-generated during the
    capacity workflow and persisted to cap_interventions.gpkg separately.

    Returns:
        Sorted, deduplicated list of infra-int IDs.
    """
    active_types = resolve_active_svc_int_types()
    if not active_types:
        return []

    active_ids: set = set()
    infra_registry_types = set(SUPPORTED_INFRA_INT_TYPES)

    for t in active_types:
        if t in infra_registry_types:
            active_ids.update(list_intervention_ids(t))
        else:
            svc_path = Path(paths.get_svc_int_registry(t))
            if not svc_path.exists():
                continue
            try:
                df = pd.read_excel(svc_path, sheet_name='extensions')
            except Exception as exc:
                print(f"  [registry] cannot read {svc_path.name}: {exc}")
                continue
            if 'requires_infra' in df.columns:
                for raw in df['requires_infra'].dropna().astype(str):
                    for token in raw.split(','):
                        s = token.strip()
                        if s:
                            active_ids.add(s)

    return sorted(active_ids)
