"""
svc_ints_orchestrator — Phase 5B engine + entry point + standalone CLI.
Last modified: 2026-06-07

Single home for the service-intervention stack (mirrors infra_ints_orchestrator):

  • Registry I/O + schema       (the svc-int catalogue, xlsx)
  • apply_svc_int               (delta materialisation — added in Phase 2)
  • Phase 5B orchestration + standalone CLI   (added in Phase 5)

EXT discovery lives in ``svc_ints_extension`` and NDC building in ``svc_ints_ndc``;
both import THIS module for registry helpers and are imported lazily (inside the phase
function) to avoid an import cycle.

Registry (declarative source of truth)
--------------------------------------
One logical record per service intervention (extended line 'ext' / new direct
connection 'ndc'), identified by ``int_id`` and stored as a row of the ``extensions``
sheet of a per-type xlsx (``ext_interventions.xlsx`` / ``ndc_interventions.xlsx``).
A record stores the *declarative operation list* (the service delta), keyed by the
in-run ``(route_id, direction_id, variant_rank)`` of the line it targets — valid
because svc-ints are generated against the active version each run.

  operations     : [{op, params}, …] — extend / truncate / reroute / set_frequency
                   on an existing line (applied to both direction_id rows), or
                   new_line for an NDC.
  requires_infra : the cc_id(s) the intervention activates (empty for pure EXT).
  affected_*     : affected-set primaries (stations / services) — Phase 6 expands
                   these to the exact closure.

The materialised form (a complete projected rail_lines/rail_segments delta per
svc-int) is produced on demand by ``apply_svc_int`` (Phase 2), at a deterministic
per-svc-int path the existing catchment/OD/routing readers consume unchanged.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
from shapely.geometry import LineString, Point

import cache_manifest
import paths
import settings

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# All svc-int registries store their rows in one sheet, named for back-compat with
# infra_ints_orchestrator.enumerate_active_infra_ints (reads sheet 'extensions').
SHEET = 'extensions'

SUPPORTED_SVC_INT_TYPES = ('ext', 'ndc')

# Column order of the catalogue row (the declarative svc-int record).
RECORD_COLS: tuple = (
    'int_id', 'int_type', 'base_authored', 'svc_version',
    'route_id', 'direction_id', 'variant_rank',
    'operations', 'total_dep', 'line_type', 'mode_class',
    'requires_infra', 'affected_stations', 'affected_services',
)

# Comma-joined list fields (stored as strings in the xlsx cell).
_LIST_FIELDS: tuple = ('requires_infra', 'affected_stations', 'affected_services')

# int_id prefix per type (e.g. ext_100001, ndc_103001).
_TYPE_CODE: Dict[str, str] = {'ext': 'ext', 'ndc': 'ndc'}

# Per-type id start block (mirrors the infra-int DEV_ID_START_* convention).
_TYPE_START: Dict[str, str] = {
    'ext': 'DEV_ID_START_EXT',
    'ndc': 'DEV_ID_START_NDC',
}


# ═════════════════════════════════════════════════════════════════════════════
# (DE)SERIALISATION
# ═════════════════════════════════════════════════════════════════════════════

def serialize_list(values: Optional[List]) -> str:
    """Comma-join a list of ids into the stored string form ('' for empty/None)."""
    if values is None:
        return ''
    return ','.join(str(v).strip() for v in values if str(v).strip())


def deserialize_list(raw) -> List[str]:
    """Split a stored comma-joined string back into a list ([] for empty)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    s = str(raw).strip()
    if not s:
        return []
    return [tok.strip() for tok in s.split(',') if tok.strip()]


def serialize_ops(ops: Optional[List[Dict]]) -> str:
    """JSON-encode the operation list for a single xlsx cell ('[]' for empty/None)."""
    return json.dumps(ops or [], ensure_ascii=False)


def deserialize_ops(raw) -> List[Dict]:
    """Decode the operation-list JSON from a cell ([] for empty/unparseable)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    s = str(raw).strip()
    if not s:
        return []
    try:
        val = json.loads(s)
    except (ValueError, TypeError):
        return []
    return val if isinstance(val, list) else []


# ═════════════════════════════════════════════════════════════════════════════
# REGISTRY — read
# ═════════════════════════════════════════════════════════════════════════════

def _svc_network(network: Optional[str] = None) -> str:
    """Resolve the registry partition key for a svc-int registry — the '<infra>__<svc>' combo.

    Defaults from the configured run versions when not given explicitly, so standalone /
    incidental reads land in the right Developments/<combo>/<type>/ workspace (decision H).
    """
    if network:
        return network
    iv = getattr(settings, 'INFRA_VERSION', 'Build_New')
    iv = getattr(settings, 'INFRA_BUILD_NEW_NAME', iv) if iv == 'Build_New' else iv
    sv = getattr(settings, 'SVC_VERSION', 'Build_New')
    sv = getattr(settings, 'SVC_BUILD_NEW_NAME', sv) if sv == 'Build_New' else sv
    return f"{iv}__{sv}"


def read_registry(
    int_type: str,
    registry_path: Optional[str] = None,
    network: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Return the raw catalogue DataFrame for a svc-int type, or None if absent."""
    xlsx = Path(registry_path or paths.get_svc_int_registry(int_type, _svc_network(network)))
    if not xlsx.exists():
        return None
    try:
        df = pd.read_excel(xlsx, sheet_name=SHEET)
    except Exception as exc:
        print(f"  [svc-registry] cannot read {xlsx.name}: {exc}")
        return None
    return df if not df.empty else None


def read_record(
    int_type: str,
    int_id: str,
    registry_path: Optional[str] = None,
    network: Optional[str] = None,
) -> Optional[Dict]:
    """Return one svc-int record as a dict (ops + list fields deserialised)."""
    df = read_registry(int_type, registry_path, network)
    if df is None or 'int_id' not in df.columns:
        return None
    sel = df[df['int_id'].astype(str) == str(int_id)]
    if sel.empty:
        return None
    return _row_to_record(sel.iloc[0])


def read_records(
    int_type: str,
    registry_path: Optional[str] = None,
    network: Optional[str] = None,
) -> List[Dict]:
    """Return every svc-int record of a type as dicts (deserialised)."""
    df = read_registry(int_type, registry_path, network)
    if df is None:
        return []
    return [_row_to_record(row) for _, row in df.iterrows()]


def list_svc_int_ids(
    int_type: str,
    registry_path: Optional[str] = None,
    network: Optional[str] = None,
) -> List[str]:
    """Return all distinct int_ids in a svc-int registry (sorted; [] if absent)."""
    df = read_registry(int_type, registry_path, network)
    if df is None or 'int_id' not in df.columns:
        return []
    return sorted(df['int_id'].dropna().astype(str).unique().tolist())


# ═════════════════════════════════════════════════════════════════════════════
# REGISTRY — write
# ═════════════════════════════════════════════════════════════════════════════

def append_records(
    int_type: str,
    records: List[Dict],
    registry_path: Optional[str] = None,
    network: Optional[str] = None,
) -> None:
    """Append svc-int records to a registry, replacing rows with the same int_id."""
    if not records:
        return
    xlsx = Path(registry_path or paths.get_svc_int_registry(int_type, _svc_network(network)))
    xlsx.parent.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame([_record_to_row(r) for r in records], columns=RECORD_COLS)

    old_df = read_registry(int_type, registry_path, network)
    if old_df is not None and 'int_id' in old_df.columns:
        new_ids = set(new_df['int_id'].astype(str))
        old_df = old_df[~old_df['int_id'].astype(str).isin(new_ids)]
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_excel(xlsx, sheet_name=SHEET, index=False)
    print(f"  [svc-registry] wrote {len(new_df)} record(s) to {xlsx.name}")


def delete_records(
    int_type: str,
    int_ids: List[str],
    registry_path: Optional[str] = None,
    network: Optional[str] = None,
) -> None:
    """Remove the given int_ids from a svc-int registry."""
    if not int_ids:
        return
    xlsx = Path(registry_path or paths.get_svc_int_registry(int_type, _svc_network(network)))
    df = read_registry(int_type, registry_path, network)
    if df is None or 'int_id' not in df.columns:
        return
    drop = {str(i) for i in int_ids}
    kept = df[~df['int_id'].astype(str).isin(drop)].reset_index(drop=True)
    kept.to_excel(xlsx, sheet_name=SHEET, index=False)


# ═════════════════════════════════════════════════════════════════════════════
# REGISTRY — id allocation
# ═════════════════════════════════════════════════════════════════════════════

def next_svc_int_id(
    int_type: str,
    registry_path: Optional[str] = None,
    network: Optional[str] = None,
) -> str:
    """Allocate the next sequential int_id for a svc-int type (e.g. 'ext_100001')."""
    code = _TYPE_CODE.get(int_type)
    if code is None:
        raise ValueError(f"unknown svc int_type '{int_type}'")
    start_attr = _TYPE_START[int_type]
    start = int(getattr(settings, start_attr))
    prefix = f"{code}_"
    max_n = start
    for iid in list_svc_int_ids(int_type, registry_path, network):
        if str(iid).startswith(prefix):
            tail = str(iid)[len(prefix):]
            if tail.isdigit():
                max_n = max(max_n, int(tail))
    return f"{prefix}{max_n + 1}"


# ─────────────────────────────────────────────────────────────────────────────
# Row <-> record conversion
# ─────────────────────────────────────────────────────────────────────────────

def _record_to_row(record: Dict) -> Dict:
    """Serialise a record dict into the stored (string-cell) row form."""
    row: Dict = {}
    for col in RECORD_COLS:
        val = record.get(col)
        if col == 'operations':
            val = serialize_ops(val if isinstance(val, list) else deserialize_ops(val))
        elif col in _LIST_FIELDS and isinstance(val, (list, tuple)):
            val = serialize_list(val)
        row[col] = val
    return row


def _row_to_record(row) -> Dict:
    """Deserialise a stored row into a record dict (ops + list fields as lists)."""
    rec: Dict = {}
    for col in RECORD_COLS:
        val = row[col] if col in row.index else None
        if col == 'operations':
            val = deserialize_ops(val)
        elif col in _LIST_FIELDS:
            val = deserialize_list(val)
        rec[col] = val
    return rec


# ═════════════════════════════════════════════════════════════════════════════
# MATERIALISATION — apply_svc_int (the per-svc-int delta, Phase 2)
# ═════════════════════════════════════════════════════════════════════════════

SWISS_CRS = 'EPSG:2056'

# GTFS route_type → rail_lines/rail_segments layer name.
_LAYER_FOR_LINE_TYPE: Dict[int, str] = {
    102: 'long_distance_rail', 103: 'inter_regional_rail',
    106: 'regional_rail', 109: 'sbahn',
}


def apply_svc_int(
    svc_int: Dict,
    base_svc_version: str,
    base_infra_version: str,
    *,
    use_cache: Optional[bool] = None,
    run_zvv: bool = True,
) -> Dict:
    """Materialise a svc-int's delta network (changed/added lines), re-projected.

    Applies the declarative operation list to the base service network, builds the
    changed/added Unprojected rail_lines/rail_segments/rail_stops, then re-projects
    only those lines onto the composed infra (base + requires_infra CC) with REAL
    infra travel time (services_service_projection.project_lines) and writes a
    complete projected delta at the per-svc-int path the catchment/OD/routing/
    capacity readers consume unchanged.

    Args:
        svc_int: a svc-int record dict (as from read_record).
        base_svc_version: base service version WITHOUT the '_network' suffix.
        base_infra_version: base infra version (e.g. 'AS_2026_ZH').
        use_cache: skip if the projected delta already exists (default settings.use_cache_svc_ints).
        run_zvv: apply the ZVV geometry post-pass (matches the full pipeline).

    Returns:
        dict(svc_int_id, unprojected_dir, projected_path, composed_infra,
             changed_route_ids, affected_stations).
    """
    import ints_core as core
    import services_service_projection as ssp

    if use_cache is None:
        use_cache = getattr(settings, 'use_cache_svc_ints', False)

    svc_int_id = str(svc_int['int_id'])
    out_dir = Path(paths.get_svc_int_network_dir(svc_int_id))
    unproj_dir = out_dir / paths.SERVICES_UNPROJECTED_SUBDIR
    proj_dir = out_dir / base_infra_version
    projected_path = proj_dir / 'rail_segments.gpkg'

    if (use_cache and projected_path.exists()
            and cache_manifest.check_manifest(
                paths.get_svc_int_catalogue_dir(
                    f'{base_infra_version}__{base_svc_version}'),
                'svc_ints_5b',
                {'infra_version': base_infra_version,
                 'svc_version': base_svc_version})):
        print(f"  [apply] {svc_int_id}: cached delta at {projected_path} — reuse")
        return _apply_result(svc_int_id, unproj_dir, projected_path, base_infra_version, svc_int)

    # 1. Compose the infra the svc-int routes on (base + its requires_infra CC).
    requires = svc_int.get('requires_infra') or []
    composed_infra = core.compose_infra(base_infra_version, requires, svc_version=base_svc_version)
    composed_dir = (Path(paths.get_infra_version_dir(composed_infra))
                    if not requires else
                    Path(paths.get_derived_infra_version_dir(composed_infra)))

    # 2. Load the base service network + build the delta lines/segments.
    base_lines, base_segs, base_stops = _load_base_unprojected(base_svc_version)
    stop_index = _build_stop_index(base_segs)
    infra_nodes = gpd.read_file(composed_dir / 'nodes.gpkg')
    resolve = _make_stop_resolver(stop_index, infra_nodes)

    delta_lines, delta_segs, delta_stop_nrs, changed_routes, affected = _build_delta(
        svc_int, base_lines, base_segs, resolve)
    if not delta_segs:
        print(f"  [apply] {svc_int_id}: produced no segments — skipped")
        return _apply_result(svc_int_id, unproj_dir, projected_path, base_infra_version, svc_int)

    # 3. Write the delta Unprojected folder (rail_lines / rail_segments / rail_stops).
    _write_delta_unprojected(unproj_dir, delta_lines, delta_segs, base_stops,
                             delta_stop_nrs, infra_nodes)

    # 4. Re-project the delta onto the composed infra with real TT.
    seg_gdf = gpd.GeoDataFrame(
        pd.concat([gpd.GeoDataFrame(v, crs=SWISS_CRS) for v in delta_segs.values()],
                  ignore_index=True), crs=SWISS_CRS)
    config = ssp.ProjectionConfig(
        infra_version=composed_infra, svc_version=svc_int_id + '_network',
        infra_dir=composed_dir,
        svc_dir=Path(paths.MAIN) / paths.FEEDER_LINES_DIR /
                (base_svc_version + '_network') / paths.SERVICES_UNPROJECTED_SUBDIR,
        rail_input=unproj_dir / 'rail_segments.gpkg',
        rail_output_dir=proj_dir,
        feeder_output_dir=proj_dir,
        raw_infra_dir=Path(paths.get_infra_version_dir(ssp._list_raw_dirs()[0])),
        auto_mode=True, include_plots=False,
    )
    proj_dir.mkdir(parents=True, exist_ok=True)
    enriched = ssp.project_lines(seg_gdf, config, run_zvv=run_zvv)
    ssp._write_rail_outputs(enriched, config)
    print(f"  [apply] {svc_int_id}: projected {len(enriched)} segment(s) → {projected_path}")

    return _apply_result(svc_int_id, unproj_dir, projected_path, base_infra_version,
                         svc_int, changed_routes, affected, composed_infra)


def build_merged_unprojected(
    svc_int: Dict,
    base_svc_version: str,
    base_infra_version: str,
    *,
    use_cache: Optional[bool] = None,
) -> str:
    """Write the base+delta merged routing network for one svc-int (Phase 6C input).

    apply_svc_int materialises only the delta (changed/added lines), so routing a
    svc-int needs base and delta merged back into one complete network folder:
    <svc_int_id>_network/Merged/{rail_lines,rail_segments,rail_stops}.gpkg, the
    same files the routing graph loads from a rail_base dir. Replacement is at
    variant-key grain: base rows whose (route_id, direction_id, variant_rank)
    appears in the delta are dropped and the delta rows added — an EXT replaces
    exactly its extended variants (non-extended variants survive), an NDC is
    purely additive. Segment rows come from the PROJECTED delta because the
    Unprojected delta carries TT=NaN for new stop-pairs; the projected delta has
    the real infra TT.

    Args:
        svc_int: a svc-int record dict (as from read_record).
        base_svc_version: base service version WITHOUT the '_network' suffix.
        base_infra_version: base infra version (e.g. 'AS_2026_ZH_enhanced').
        use_cache: reuse an existing Merged/ folder (default settings.use_cache_svc_ints).

    Returns:
        Absolute path to the Merged/ directory, or '' when the delta is empty
        (nothing to route — the svc-int network equals the base).

    Both sides merge in the UNPROJECTED id-space: the projected delta re-keys
    interchange stations to platform-group Betriebspunkte (e.g. Zürich HB →
    Zürich HB Löwenstrasse 8516144), which would break station membership
    against the base ids. Only the genuinely-new stop-pairs carry TT=NaN in the
    unprojected delta (base-known pairs keep their GTFS TT); their real infra
    TT is patched in from the projected delta, whose endpoint ids match for new
    legs.
    """
    if use_cache is None:
        use_cache = getattr(settings, 'use_cache_svc_ints', False)
    svc_int_id = str(svc_int['int_id'])
    out_dir = Path(paths.get_svc_int_network_dir(svc_int_id)) / 'Merged'
    targets = {n: out_dir / f'{n}.gpkg'
               for n in ('rail_lines', 'rail_segments', 'rail_stops')}
    if (use_cache and all(p.exists() for p in targets.values())
            and cache_manifest.check_manifest(
                paths.get_svc_int_catalogue_dir(
                    f'{base_infra_version}__{base_svc_version}'),
                'svc_ints_5b',
                {'infra_version': base_infra_version,
                 'svc_version': base_svc_version})):
        print(f"  [merge] {svc_int_id}: cached merged network at {out_dir} — reuse")
        return str(out_dir)

    base_dir = (Path(paths.MAIN) / paths.RAIL_LINES_DIR /
                f"{base_svc_version}_network" / paths.SERVICES_UNPROJECTED_SUBDIR)
    delta_dir = (Path(paths.get_svc_int_network_dir(svc_int_id)) /
                 paths.SERVICES_UNPROJECTED_SUBDIR)
    proj_delta = Path(paths.get_svc_int_projected_path(svc_int_id, base_infra_version))

    def _norm(x) -> str:
        try:
            return str(int(float(x)))
        except (TypeError, ValueError):
            return str(x)

    def _keys_of(df: pd.DataFrame, rid_col: str) -> pd.Series:
        return pd.Series(
            list(zip(df[rid_col].astype(str),
                     df['direction_id'].map(_norm),
                     df['variant_rank'].map(_norm))),
            index=df.index)

    # Variant keys present in the delta (from its line layers) — the replace set.
    delta_lines_path = delta_dir / 'rail_lines.gpkg'
    delta_keys: set = set()
    if delta_lines_path.exists():
        for layer in fiona.listlayers(str(delta_lines_path)):
            g = gpd.read_file(delta_lines_path, layer=layer)
            delta_keys |= set(_keys_of(g, 'route_id'))
    if not delta_keys:
        print(f"  [merge] {svc_int_id}: empty delta — nothing to merge")
        return ''

    out_dir.mkdir(parents=True, exist_ok=True)
    for p in targets.values():
        if p.exists():
            p.unlink()

    def _merge_layered(base_path: Path, delta_path: Path, out_path: Path,
                       rid_col: str, label: str) -> None:
        base_layers = set(fiona.listlayers(str(base_path))) if base_path.exists() else set()
        delta_layers = set(fiona.listlayers(str(delta_path))) if delta_path.exists() else set()
        n_dropped = n_added = 0
        for layer in sorted(base_layers | delta_layers):
            frames = []
            if layer in base_layers:
                b = gpd.read_file(base_path, layer=layer)
                if {rid_col, 'direction_id', 'variant_rank'}.issubset(b.columns):
                    drop = _keys_of(b, rid_col).isin(delta_keys)
                    n_dropped += int(drop.sum())
                    b = b[~drop.values]
                frames.append(b)
            if layer in delta_layers:
                d = gpd.read_file(delta_path, layer=layer)
                n_added += len(d)
                frames.append(d)
            frames = [f for f in frames if f is not None and not f.empty]
            if not frames:
                continue
            merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                                      crs=frames[0].crs)
            merged.to_file(out_path, layer=layer, driver='GPKG')
        print(f"  [merge] {svc_int_id}/{label}: -{n_dropped} base row(s) "
              f"(replaced variants), +{n_added} delta row(s)")

    _merge_layered(base_dir / 'rail_lines.gpkg', delta_lines_path,
                   targets['rail_lines'], 'route_id', 'lines')

    # TT patch lookup from the projected delta (real infra TT for new legs)
    tt_lookup: Dict[tuple, float] = {}
    if proj_delta.exists():
        for layer in fiona.listlayers(str(proj_delta)):
            g = gpd.read_file(proj_delta, layer=layer)
            for r in g.itertuples(index=False):
                tt = getattr(r, 'TT', None)
                if tt is not None and pd.notna(tt):
                    tt_lookup[(str(r.GTFS_ID), _norm(r.direction_id),
                               _norm(r.variant_rank), _norm(r.from_stop_nr),
                               _norm(r.to_stop_nr))] = float(tt)
    else:
        print(f"  [merge] WARNING {svc_int_id}: projected delta missing at "
              f"{proj_delta} — new stop-pairs keep TT=NaN (routed as 0 min). "
              f"Run apply_svc_int first.")

    base_seg_path = base_dir / 'rail_segments.gpkg'
    delta_seg_path = delta_dir / 'rail_segments.gpkg'
    base_layers = set(fiona.listlayers(str(base_seg_path))) if base_seg_path.exists() else set()
    delta_seg_layers = set(fiona.listlayers(str(delta_seg_path))) if delta_seg_path.exists() else set()
    n_drop = n_add = n_patch = n_nan = 0
    for layer in sorted(base_layers | delta_seg_layers):
        frames = []
        if layer in base_layers:
            b = gpd.read_file(base_seg_path, layer=layer)
            if {'GTFS_ID', 'direction_id', 'variant_rank'}.issubset(b.columns):
                drop = _keys_of(b, 'GTFS_ID').isin(delta_keys)
                n_drop += int(drop.sum())
                b = b[~drop.values]
            frames.append(b)
        if layer in delta_seg_layers:
            d = gpd.read_file(delta_seg_path, layer=layer)
            if 'TT' in d.columns:
                tt = pd.to_numeric(d['TT'], errors='coerce')
                for i in d.index[tt.isna()]:
                    r = d.loc[i]
                    k = (str(r['GTFS_ID']), _norm(r['direction_id']),
                         _norm(r['variant_rank']), _norm(r['from_stop_nr']),
                         _norm(r['to_stop_nr']))
                    if k in tt_lookup:
                        d.at[i, 'TT'] = tt_lookup[k]
                        n_patch += 1
                    else:
                        n_nan += 1
            n_add += len(d)
            frames.append(d)
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            continue
        merged = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True),
                                  crs=frames[0].crs)
        merged.to_file(targets['rail_segments'], layer=layer, driver='GPKG')
    print(f"  [merge] {svc_int_id}/segments: -{n_drop} base row(s) "
          f"(replaced variants), +{n_add} delta row(s); TT patched from the "
          f"projected delta for {n_patch} new leg(s)"
          + (f"; WARNING {n_nan} leg(s) keep TT=NaN" if n_nan else ""))

    # Stops: base set + delta-only stop numbers (per layer; dedupe on Number).
    base_stops_path = base_dir / 'rail_stops.gpkg'
    delta_stops_path = delta_dir / 'rail_stops.gpkg'
    out_frames: Dict[str, gpd.GeoDataFrame] = {}
    base_nums: set = set()
    if base_stops_path.exists():
        for layer in fiona.listlayers(str(base_stops_path)):
            g = gpd.read_file(base_stops_path, layer=layer)
            out_frames[layer] = g
            if 'Number' in g.columns:
                base_nums |= set(g['Number'].astype(str))
    n_new_stops = 0
    if delta_stops_path.exists():
        for layer in fiona.listlayers(str(delta_stops_path)):
            g = gpd.read_file(delta_stops_path, layer=layer)
            if 'Number' in g.columns:
                g = g[~g['Number'].astype(str).isin(base_nums)]
            if g.empty:
                continue
            n_new_stops += len(g)
            if layer in out_frames:
                out_frames[layer] = gpd.GeoDataFrame(
                    pd.concat([out_frames[layer], g], ignore_index=True),
                    crs=out_frames[layer].crs)
            else:
                out_frames[layer] = g
    for layer, g in out_frames.items():
        g.to_file(targets['rail_stops'], layer=layer, driver='GPKG')
    print(f"  [merge] {svc_int_id}/stops: {n_new_stops} delta-only stop(s) added; "
          f"merged network -> {out_dir}")
    return str(out_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Delta construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_delta(svc_int, base_lines, base_segs, resolve):
    """Apply the op list → per-layer delta lines/segments + touched stops."""
    ops = svc_int.get('operations') or []
    route_id = str(svc_int.get('route_id', ''))
    variant_rank = svc_int.get('variant_rank', 1)
    is_new = (svc_int.get('int_type') == 'ndc') or any(o.get('op') == 'new_line' for o in ops)

    delta_lines: Dict[str, List[Dict]] = {}
    delta_segs: Dict[str, List[Dict]] = {}
    stop_nrs: set = set()
    changed_routes: set = set()
    affected: set = set()

    if is_new:
        nl = next(o for o in ops if o.get('op') == 'new_line')['params']
        line_type = int(nl.get('line_type', svc_int.get('line_type', 109)))
        total_dep = int(nl.get('total_dep', svc_int.get('total_dep', 0)))
        layer = _layer_for_line_type(line_type)
        short = svc_int.get('line_short_name') or svc_int_id_short(svc_int)
        for dir_id in ('0', '1'):
            names = nl['stops'] if dir_id == '0' else list(reversed(nl['stops']))
            seq = [resolve(n) for n in names]
            seq = [s for s in seq if s is not None]
            if len(seq) < 2:
                continue
            meta = {'route_id': route_id or svc_int['int_id'], 'short': short,
                    'line_type': line_type, 'mode_label': _MODE_LABEL.get(layer, 'rail'),
                    'variant_rank': variant_rank}
            _emit_line(delta_lines, delta_segs, layer, meta, dir_id, seq, total_dep, {})
            stop_nrs.update(s['nr'] for s in seq)
            affected.update(s['name'] for s in seq)
        changed_routes.add(meta['route_id'])
    else:
        # EXT: extend EVERY variant of the route that terminates at the op's `endpoint`
        # (variants sharing a terminus all get the extension), both directions.
        ext_op = next((o for o in ops if o.get('op') == 'extend'), None)
        params = (ext_op or {}).get('params', {})
        from_end = params.get('from_end', 'destination')
        endpoint = params.get('endpoint')
        _flip = {'origin': 'destination', 'destination': 'origin'}
        for var in _route_variants(base_lines, route_id):
            # qualify the variant on its dir-0 terminus at the op's end
            _, lr0, ls0 = _find_target(base_lines, base_segs, route_id, '0', var)
            if lr0 is None or ls0 is None or ls0.empty:
                continue
            seq0 = _reconstruct_sequence(ls0)
            term = seq0[0]['name'] if from_end == 'origin' else seq0[-1]['name']
            if endpoint and term != endpoint:
                continue
            for dir_id in ('0', '1'):
                layer, line_row, line_seg_df = _find_target(
                    base_lines, base_segs, route_id, dir_id, var)
                if line_row is None or line_seg_df is None or line_seg_df.empty:
                    continue
                base_seq = _reconstruct_sequence(line_seg_df)
                # dir-1 sequence is reversed, so extend the opposite logical end
                fe = from_end if dir_id == '0' else _flip[from_end]
                dir_ops = [{'op': 'extend',
                            'params': {'from_end': fe, 'stops': params.get('stops', [])}}
                           if o.get('op') == 'extend' else o for o in ops]
                seq, total_dep_override = _apply_ops_to_seq(base_seq, dir_ops, resolve)
                seq = [s for s in seq if s is not None]
                if len(seq) < 2:
                    continue
                total_dep = int(total_dep_override if total_dep_override is not None
                                else line_row.get('total_dep', 0))
                base_tt = {(int(r['from_stop_nr']), int(r['to_stop_nr'])): r
                           for _, r in line_seg_df.iterrows()}
                meta = {'route_id': route_id, 'short': line_row.get('line_short_name'),
                        'line_type': int(line_row.get('line_type', 106)),
                        'mode_label': line_row.get('mode_label', 'rail'),
                        'variant_rank': var}
                _emit_line(delta_lines, delta_segs, layer, meta, dir_id, seq, total_dep, base_tt)
                stop_nrs.update(s['nr'] for s in seq)
                affected.update(s['name'] for s in seq)
                changed_routes.add(route_id)

    return delta_lines, delta_segs, stop_nrs, changed_routes, affected


def _route_variants(base_lines, route_id) -> List[int]:
    """All dir-0 variant_ranks of a route_id across the base line layers."""
    out: set = set()
    for ldf in base_lines.values():
        m = ((ldf['route_id'].astype(str) == str(route_id)) &
             (ldf['direction_id'].astype(str) == '0'))
        out.update(int(v) for v in ldf.loc[m, 'variant_rank'].tolist())
    return sorted(out)


def _emit_line(delta_lines, delta_segs, layer, meta, dir_id, seq, total_dep, base_tt):
    """Append one direction's line row + its stop-pair segment rows to the delta."""
    freq_hr = round(total_dep / (getattr(settings, 'GK_WINDOW_MIN', 840) / 60.0), 3)
    line_geom = LineString([(s['E'], s['N']) for s in seq])
    delta_lines.setdefault(layer, []).append({
        'route_id': meta['route_id'], 'direction_id': dir_id,
        'variant_rank': meta['variant_rank'], 'variant_trip_share': 1.0,
        'line_short_name': meta['short'], 'origin': seq[0]['name'],
        'destination': seq[-1]['name'],
        'line_long_name': f"{meta['short']}: {seq[0]['name']} - {seq[-1]['name']}",
        'line_type': meta['line_type'], 'mode_label': meta['mode_label'],
        'mode_class': 'rail', 'agency_id': '', 'is_circular': seq[0]['nr'] == seq[-1]['nr'],
        'n_stops': len(seq), 'service_period': 'all_day',
        'freq_am_peak_dep_hr': freq_hr, 'freq_pm_peak_dep_hr': freq_hr,
        'freq_offpeak_dep_hr': freq_hr, 'total_dep': total_dep,
        'freq_directional': False, 'tt_source': 'projected', 'geometry': line_geom,
    })
    rows = delta_segs.setdefault(layer, [])
    for a, b in zip(seq[:-1], seq[1:]):
        prev = base_tt.get((a['nr'], b['nr']))
        keep_tt = prev is not None and pd.notna(prev.get('TT'))
        rows.append({
            'GTFS_ID': meta['route_id'], 'Service': meta['short'],
            'direction_id': dir_id, 'variant_rank': meta['variant_rank'],
            'mode_label': meta['mode_label'], 'mode_class': 'rail',
            'from_stop_nr': a['nr'], 'to_stop_nr': b['nr'],
            'from_stop_name': a['name'], 'to_stop_name': b['name'],
            'from_stop_E': a['E'], 'from_stop_N': a['N'],
            'to_stop_E': b['E'], 'to_stop_N': b['N'],
            'TT': float(prev['TT']) if keep_tt else np.nan,
            'tt_source': 'gtfs' if keep_tt else 'formula',
            'IVWT': float(prev['IVWT']) if keep_tt and pd.notna(prev.get('IVWT')) else 0.0,
            'service_period': 'all_day',
            'freq_am_peak_dep_hr': freq_hr, 'freq_pm_peak_dep_hr': freq_hr,
            'freq_offpeak_dep_hr': freq_hr,
            '_source_layer': layer,
            'geometry': LineString([(a['E'], a['N']), (b['E'], b['N'])]),
        })


def _apply_ops_to_seq(seq, ops, resolve):
    """Apply extend/truncate/reroute/set_frequency to a stop sequence."""
    total_dep_override = None
    for op in ops:
        kind = op.get('op'); p = op.get('params', {})
        if kind == 'extend':
            added = [resolve(n) for n in p.get('stops', [])]
            added = [s for s in added if s is not None]
            seq = (seq + added) if p.get('from_end', 'destination') == 'destination' \
                else (added + seq)
        elif kind == 'truncate':
            end = p.get('from_end', 'destination')
            if 'n' in p:
                n = int(p['n'])
                seq = seq[:-n] if end == 'destination' else seq[n:]
            elif 'to_stop' in p:
                names = [s['name'] for s in seq]
                if p['to_stop'] in names:
                    i = names.index(p['to_stop'])
                    seq = seq[:i + 1] if end == 'destination' else seq[i:]
        elif kind == 'reroute':
            via = [resolve(n) for n in p.get('via', [])]
            via = [s for s in via if s is not None]
            repl = list(p.get('replace', []))
            seq = _splice_reroute(seq, repl, via)
        elif kind == 'set_frequency':
            total_dep_override = int(p['total_dep'])
        elif kind == 'new_line':
            seq = [resolve(n) for n in p.get('stops', [])]
            seq = [s for s in seq if s is not None]
            total_dep_override = int(p.get('total_dep', total_dep_override or 0))
    return seq, total_dep_override


def _splice_reroute(seq, repl_names, via):
    """Replace the first contiguous run matching repl_names with via."""
    if not repl_names:
        return seq
    names = [s['name'] for s in seq]
    for i in range(len(names) - len(repl_names) + 1):
        if names[i:i + len(repl_names)] == repl_names:
            return seq[:i] + via + seq[i + len(repl_names):]
    return seq


def _reconstruct_sequence(seg_df):
    """Chain a line's stop-pair rows into an ordered [{nr,name,E,N}] sequence."""
    by_from: Dict[int, object] = {}
    coord: Dict[int, Tuple] = {}
    froms: set = set(); tos: set = set()
    for _, r in seg_df.iterrows():
        f, t = int(r['from_stop_nr']), int(r['to_stop_nr'])
        by_from[f] = r
        froms.add(f); tos.add(t)
        coord[f] = (r['from_stop_name'], r['from_stop_E'], r['from_stop_N'])
        coord[t] = (r['to_stop_name'], r['to_stop_E'], r['to_stop_N'])
    starts = froms - tos
    start = (sorted(starts)[0] if starts else int(seg_df.iloc[0]['from_stop_nr']))
    seq: List[Dict] = []; cur = start; seen: set = set()
    while cur is not None and cur not in seen:
        seen.add(cur)
        nm, e, n = coord[cur]
        seq.append({'nr': cur, 'name': nm, 'E': e, 'N': n})
        nxt = by_from.get(cur)
        cur = int(nxt['to_stop_nr']) if nxt is not None else None
    return seq


def _find_target(base_lines, base_segs, route_id, dir_id, variant_rank):
    """Locate (layer, line_row, line_segments) for a target (route_id, dir, variant)."""
    for layer, lines in base_lines.items():
        m = ((lines['route_id'].astype(str) == route_id) &
             (lines['direction_id'].astype(str) == str(dir_id)) &
             (lines['variant_rank'].astype(int) == int(variant_rank)))
        if m.any():
            line_row = lines[m].iloc[0].to_dict()
            segs = base_segs.get(layer)
            sm = ((segs['GTFS_ID'].astype(str) == route_id) &
                  (segs['direction_id'].astype(str) == str(dir_id)) &
                  (segs['variant_rank'].astype(int) == int(variant_rank)))
            return layer, line_row, segs[sm]
    return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# Base loading + stop resolution
# ─────────────────────────────────────────────────────────────────────────────

def _load_base_unprojected(base_svc_version):
    """Load base Unprojected (lines_by_layer, segments_by_layer, stops)."""
    base = (Path(paths.MAIN) / paths.RAIL_LINES_DIR /
            (base_svc_version + '_network') / paths.SERVICES_UNPROJECTED_SUBDIR)
    lines = {L: gpd.read_file(base / 'rail_lines.gpkg', layer=L)
             for L in fiona.listlayers(str(base / 'rail_lines.gpkg'))}
    segs = {L: gpd.read_file(base / 'rail_segments.gpkg', layer=L)
            for L in fiona.listlayers(str(base / 'rail_segments.gpkg'))}
    stops_path = base / 'rail_stops.gpkg'
    stops = gpd.read_file(stops_path) if stops_path.exists() else None
    return lines, segs, stops


def _build_stop_index(base_segs):
    """name → (stop_nr, E, N) from every base segment endpoint."""
    idx: Dict[str, Tuple] = {}
    for segs in base_segs.values():
        for _, r in segs.iterrows():
            idx.setdefault(str(r['from_stop_name']),
                           (int(r['from_stop_nr']), r['from_stop_E'], r['from_stop_N']))
            idx.setdefault(str(r['to_stop_name']),
                           (int(r['to_stop_nr']), r['to_stop_E'], r['to_stop_N']))
    return idx


def _make_stop_resolver(stop_index, infra_nodes):
    """Return resolve(name) → {nr,name,E,N}; base stops first, infra nodes fallback."""
    node_idx: Dict[str, Tuple] = {}
    if infra_nodes is not None and 'Name' in infra_nodes.columns:
        for _, r in infra_nodes.iterrows():
            g = r.geometry
            if g is not None:
                node_idx.setdefault(str(r['Name']), (r.get('Number'), g.x, g.y))

    def resolve(name):
        name = str(name)
        if name in stop_index:
            nr, e, n = stop_index[name]
            return {'nr': int(nr), 'name': name, 'E': float(e), 'N': float(n)}
        if name in node_idx:
            nr, e, n = node_idx[name]
            return {'nr': int(nr) if pd.notna(nr) else abs(hash(name)) % 9_000_000,
                    'name': name, 'E': float(e), 'N': float(n)}
        print(f"  [apply]   WARN stop '{name}' not found in base stops or infra nodes")
        return None
    return resolve


# ─────────────────────────────────────────────────────────────────────────────
# Delta Unprojected writing
# ─────────────────────────────────────────────────────────────────────────────

def _write_delta_unprojected(unproj_dir, delta_lines, delta_segs, base_stops,
                             delta_stop_nrs, infra_nodes):
    """Write the delta rail_lines/rail_segments/rail_stops Unprojected folder."""
    unproj_dir.mkdir(parents=True, exist_ok=True)
    lines_path = unproj_dir / 'rail_lines.gpkg'
    segs_path = unproj_dir / 'rail_segments.gpkg'
    stops_path = unproj_dir / 'rail_stops.gpkg'
    for p in (lines_path, segs_path, stops_path):
        if p.exists():
            p.unlink()

    for layer, rows in delta_lines.items():
        gpd.GeoDataFrame(rows, crs=SWISS_CRS).to_file(lines_path, layer=layer, driver='GPKG')
    for layer, rows in delta_segs.items():
        gpd.GeoDataFrame(rows, crs=SWISS_CRS).to_file(segs_path, layer=layer, driver='GPKG')

    _write_delta_stops(stops_path, base_stops, delta_stop_nrs, infra_nodes)


def _write_delta_stops(stops_path, base_stops, delta_stop_nrs, infra_nodes):
    """Stops the delta touches: base rows where present, infra-node fallback otherwise."""
    want = {str(n) for n in delta_stop_nrs}
    rows = []
    found: set = set()
    if base_stops is not None and 'Number' in base_stops.columns:
        sel = base_stops[base_stops['Number'].astype(str).isin(want)]
        for _, r in sel.iterrows():
            rows.append(r.to_dict())
            found.add(str(r['Number']))
    missing = want - found
    if missing and infra_nodes is not None:
        by_nr = {str(r.get('Number')): r for _, r in infra_nodes.iterrows()}
        for nr in missing:
            r = by_nr.get(nr)
            if r is None or r.geometry is None:
                continue
            rows.append({'Number': nr, 'stop_name': str(r.get('Name', '')),
                         'stop_lat': '', 'stop_lon': '', 'geometry': Point(r.geometry.x, r.geometry.y)})
    if rows:
        gpd.GeoDataFrame(rows, crs=SWISS_CRS).to_file(stops_path, layer='sbahn', driver='GPKG')


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────

_MODE_LABEL: Dict[str, str] = {
    'sbahn': 'S-Bahn / Suburban Rail', 'long_distance_rail': 'Long Distance Rail',
    'inter_regional_rail': 'Inter-Regional Rail', 'regional_rail': 'Regional Rail',
}


def _layer_for_line_type(line_type) -> str:
    try:
        return _LAYER_FOR_LINE_TYPE.get(int(line_type), 'regional_rail')
    except (ValueError, TypeError):
        return 'regional_rail'


def svc_int_id_short(svc_int) -> str:
    """A short service label for an NDC line lacking an explicit line_short_name."""
    return str(svc_int.get('int_id', 'NDC')).replace('ndc_', 'NDC')


def _apply_result(svc_int_id, unproj_dir, projected_path, base_infra_version, svc_int,
                  changed_routes=None, affected=None, composed_infra=None) -> Dict:
    return {
        'svc_int_id': svc_int_id,
        'int_type': svc_int.get('int_type'),
        'unprojected_dir': str(unproj_dir),
        'projected_path': str(projected_path),
        'composed_infra': composed_infra or base_infra_version,
        'changed_route_ids': sorted(changed_routes or []),
        'affected_stations': sorted(affected or svc_int.get('affected_stations', [])),
    }


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 5B — orchestration (discover → register → materialise → plot)
# ═════════════════════════════════════════════════════════════════════════════

def phase_5b_service_interventions(
    base_infra: str,
    base_svc: str,
    mode: Optional[str] = None,
    make_plots: Optional[bool] = None,
    use_cache: Optional[bool] = None,
    sa_polygon=None,
    buffer_polygon=None,
    ndc_candidates: Optional[List[Dict]] = None,
    interactive: bool = False,
) -> Dict:
    """Generate, register and materialise the service-intervention catalogue.

    Discovers EXT (line extensions) and/or NDC (new direct connections) per
    settings.SVC_INT_MODE, registers them in the per-type xlsx catalogues, then
    eagerly materialises each one's delta network (apply_svc_int) for the downstream
    5C capacity pass and the delta plots. Always writes the catalogue + affected-set
    CSVs; renders delta / per-type / all-produced overlays when make_plots.

    Args:
        base_infra: base infra version (e.g. 'AS_2026_ZH') — NOT a derived version.
        base_svc: base service version WITHOUT the '_network' suffix.
        mode: SVC_INT_MODE override ('NONE'|'ALL'|'EXT'|'NDC'); default from settings.
        make_plots: render delta/overlay plots; default settings.PLOT_SVC_INTS.
        use_cache: keep existing catalogue + materialised deltas; default
            settings.use_cache_svc_ints.
        sa_polygon: study-area boundary (EXT terminus / NDC scope gate).
        buffer_polygon: study-area buffer (EXT/NDC candidate-station extent).
        ndc_candidates: 5A connecting-curve candidates (branch_a/b, requires_infra);
            if None and NDC is active, discovered via infra_ints_connecting_curve
            (which also registers the CC).
        interactive: standalone-CLI flag (re-discovery may prompt for CC composition).

    Returns:
        dict(ext_ids, ndc_ids, materialised, plots).
    """
    mode = (mode or getattr(settings, 'SVC_INT_MODE', 'NONE'))
    if make_plots is None:
        make_plots = getattr(settings, 'PLOT_SVC_INTS', False)
    if use_cache is None:
        use_cache = getattr(settings, 'use_cache_svc_ints', False)

    active = _active_svc_int_types(mode)
    # Registry partition key — the '<infra>__<svc>' combo this run targets. base_infra is
    # the propagated (enhanced) version, so EXT/NDC land in the SAME combo 5A's CC and 5C's
    # CAP use; passed explicitly because the _svc_network() default reads settings (base).
    combo = f"{base_infra}__{base_svc}"
    print(f"\n=== Phase 5B — Service Interventions (mode={mode}) ===")
    print(f"  base infra: {base_infra} | services: {base_svc} | combo: {combo} | active: {active or 'none'}")

    result: Dict = {'ext_ids': [], 'ndc_ids': [], 'materialised': [], 'plots': []}
    if not active:
        return result

    cat_dir = paths.get_svc_int_catalogue_dir(combo)
    manifest_ok = use_cache and cache_manifest.check_manifest(
        cat_dir, 'svc_ints_5b',
        {'infra_version': base_infra, 'svc_version': base_svc})

    ext_candidates: Optional[List[Dict]] = None

    # EXT — discover + register (lazy import breaks the cycle) -----------------
    if 'ext' in active:
        if manifest_ok and list_svc_int_ids('ext', network=combo):
            result['ext_ids'] = list_svc_int_ids('ext', network=combo)
            print(f"  [ext] use_cache: keeping {len(result['ext_ids'])} existing EXT record(s)")
        else:
            delete_records('ext', list_svc_int_ids('ext', network=combo), network=combo)
            import svc_ints_extend_lines as ext
            disc = ext.discover_and_register(base_infra, base_svc, sa_polygon, buffer_polygon,
                                             network=combo)
            result['ext_ids'] = disc['ext_ids']
            ext_candidates = disc.get('candidates')

    # NDC — build from 5A candidates + register -------------------------------
    if 'ndc' in active:
        if manifest_ok and list_svc_int_ids('ndc', network=combo):
            result['ndc_ids'] = list_svc_int_ids('ndc', network=combo)
            print(f"  [ndc] use_cache: keeping {len(result['ndc_ids'])} existing NDC record(s)")
        else:
            if ndc_candidates is None:
                import infra_ints_connecting_curve as cc   # lazy — also registers CC
                ndc_candidates = cc.discover_and_register(
                    base_infra, base_svc, sa_polygon=sa_polygon,
                    buffer_polygon=buffer_polygon, interactive=interactive
                ).get('ndc_candidates', [])
            delete_records('ndc', list_svc_int_ids('ndc', network=combo), network=combo)
            import svc_ints_new_direct_connections as ndc
            disc = ndc.discover_and_register(
                base_infra, base_svc, ndc_candidates, sa_polygon=sa_polygon, network=combo)
            result['ndc_ids'] = disc['ndc_ids']

    # Materialise each registered svc-int (delta network, real infra TT) -------
    todo = ([('ext', i) for i in result['ext_ids']] +
            [('ndc', i) for i in result['ndc_ids']])
    print(f"  [apply] materialising {len(todo)} svc-int delta network(s)…")
    for int_type, iid in todo:
        rec = read_record(int_type, iid, network=combo)
        if rec is None:
            continue
        try:
            result['materialised'].append(
                apply_svc_int(rec, base_svc, base_infra, use_cache=use_cache))
        except Exception as exc:
            print(f"  [apply]   WARNING {iid} failed: {exc}")

    # Data outputs — always written (independent of make_plots) ---------------
    _write_svc_int_csvs(base_infra, result, network=combo)
    _write_affected_set_csv(base_infra, base_svc, network=combo)
    cache_manifest.write_manifest(cat_dir, 'svc_ints_5b',
                                  {'infra_version': base_infra,
                                   'svc_version': base_svc})

    # Candidate-overview + delta / overlay plots ------------------------------
    if make_plots:
        try:
            if ext_candidates:
                p = plot_ext_candidates(base_infra, base_svc, ext_candidates, sa_polygon)
                if p:
                    result['plots'].append(p)
            if ndc_candidates:
                p = plot_ndc_candidates(base_infra, base_svc, ndc_candidates, sa_polygon)
                if p:
                    result['plots'].append(p)
            result['plots'] += plot_svc_interventions(base_infra, base_svc, result, sa_polygon)
        except Exception as exc:
            print(f"  [plot] WARNING: svc-int plots failed: {exc}")

    print(f"=== Phase 5B done: {len(result['ext_ids'])} EXT, "
          f"{len(result['ndc_ids'])} NDC, {len(result['materialised'])} materialised ===\n")
    return result


def _active_svc_int_types(mode: str) -> List[str]:
    """Map SVC_INT_MODE to the svc-int registry types to generate."""
    m = str(mode).upper()
    if m == 'NONE':
        return []
    if m == 'ALL':
        return list(SUPPORTED_SVC_INT_TYPES)
    if m in ('EXT', 'NDC'):
        return [m.lower()]
    print(f"  [svc-int] unknown SVC_INT_MODE='{mode}' — treating as 'NONE'")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Data outputs (catalogue + affected sets)
# ─────────────────────────────────────────────────────────────────────────────

def _write_svc_int_csvs(base_infra: str, result: Dict,
                        network: Optional[str] = None) -> None:
    """Write the svc-int catalogue dump + per-svc-int affected-set CSVs."""
    out_dir = Path(paths.get_svc_int_catalogue_dir(_svc_network(network)))
    out_dir.mkdir(parents=True, exist_ok=True)
    materialised = {m['svc_int_id']: m for m in result.get('materialised', [])}

    cat_rows: List[Dict] = []
    aff_rows: List[Dict] = []
    for int_type in ('ext', 'ndc'):
        for rec in read_records(int_type, network=network):
            iid = str(rec['int_id'])
            # affected_stations summarises the change span: [endpoint, target] for an
            # EXT, the full new stop sequence for an NDC.
            span = rec.get('affected_stations') or []
            cat_rows.append({
                'int_id': iid, 'int_type': int_type, 'route_id': rec.get('route_id'),
                'total_dep': rec.get('total_dep'), 'line_type': rec.get('line_type'),
                'n_stops': len(span), 'origin': span[0] if span else '',
                'destination': span[-1] if span else '',
                'requires_infra': serialize_list(rec.get('requires_infra')),
                'materialised': iid in materialised,
                'projected_path': materialised.get(iid, {}).get('projected_path', ''),
            })
            aff_rows.append({
                'int_id': iid, 'int_type': int_type,
                'affected_stations': serialize_list(rec.get('affected_stations')),
                'affected_services': serialize_list(rec.get('affected_services')),
            })

    if cat_rows:
        # utf-8-sig so Swiss names (Dübendorf, Pfäffikon, Zürich) survive an Excel round-trip
        cat_path = out_dir / f"svc_int_catalogue_{base_infra}.csv"
        pd.DataFrame(cat_rows).to_csv(cat_path, index=False, encoding='utf-8-sig')
        print(f"  [csv] wrote {cat_path.name} ({len(cat_rows)} svc-int(s))")
        aff_path = out_dir / f"svc_int_affected_sets_{base_infra}.csv"
        pd.DataFrame(aff_rows).to_csv(aff_path, index=False, encoding='utf-8-sig')
        print(f"  [csv] wrote {aff_path.name}")


def _resolve_affected_sets(records: List[Dict], base_lines: Dict,
                           base_segs: Dict) -> List[Dict]:
    """Resolve each svc-int's affected_set to the keys Phase 6's closure consumes.

    services → concrete variant_keys (``route_id_direction_rank``, the 4C routing
    key): for an EXT, every (direction_id, variant_rank) of the route in the base
    network; for an NDC, the new line's synthetic keys (int_id, both directions,
    its variant_rank). stations → id_point (== from_stop_nr), resolved from the
    stored station names. Names that don't resolve are dropped (no crash)."""
    # route_id -> {(direction_id, variant_rank)} from the base line layers
    route_variants: Dict[str, set] = {}
    for ldf in base_lines.values():
        for _, r in ldf.iterrows():
            route_variants.setdefault(str(r['route_id']), set()).add(
                (str(r['direction_id']), int(r['variant_rank'])))
    name_to_nr = {n: nr for n, (nr, _e, _n) in _build_stop_index(base_segs).items()}

    rows: List[Dict] = []
    for rec in records:
        iid, itype = str(rec['int_id']), rec['int_type']
        rid = str(rec.get('route_id', ''))
        if itype == 'ndc':
            rank = int(rec.get('variant_rank', 1) or 1)
            vks = [f"{rid}_{d}_{rank}" for d in ('0', '1')]
        else:
            vks = sorted(f"{rid}_{d}_{v}"
                         for (d, v) in route_variants.get(rid, set()))
        stations = sorted({name_to_nr[str(n)]
                           for n in (rec.get('affected_stations') or [])
                           if str(n) in name_to_nr})
        rows.append({'int_id': iid, 'int_type': itype,
                     'affected_stations': stations, 'affected_services': vks})
    return rows


def _write_affected_set_csv(base_infra: str, base_svc: str,
                            network: Optional[str] = None) -> None:
    """Write the canonical per-svc-int affected_set (id_point stations + concrete
    variant_keys) — additive Phase-6 hook. Leaves the legacy
    svc_int_affected_sets_<base_infra>.csv untouched."""
    recs = [r for it in ('ext', 'ndc') for r in read_records(it, network=network)]
    if not recs:
        return
    base_lines, base_segs, _ = _load_base_unprojected(base_svc)
    rows = _resolve_affected_sets(recs, base_lines, base_segs)
    for row in rows:
        row['affected_stations'] = serialize_list(row['affected_stations'])
        row['affected_services'] = serialize_list(row['affected_services'])
    out_dir = Path(paths.get_svc_int_catalogue_dir(_svc_network(network)))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"svc_int_affected_set_{base_infra}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"  [csv] wrote {out_path.name} ({len(rows)} svc-int(s), variant_key + id_point)")


# ─────────────────────────────────────────────────────────────────────────────
# Plots (service-line delta over the SA infra: lakes + stations + termini labels)
# ─────────────────────────────────────────────────────────────────────────────

_EXT_COLOR     = '#1f9e4f'   # dark green: the new (extended) segment
_EXT_COLOR_OLD = '#a6dcb8'   # light green: the pre-existing run of the line
_NDC_COLOR     = '#e8730c'   # orange: NDC (an entirely new line)
_NDC_COLOR_OLD = '#f4c9a0'   # light orange (NDC has no existing part — unused in practice)
_BACKDROP      = '#d4d4d4'   # unused infrastructure
_LAKE_FC       = '#c8e8f5'
_LAKE_EC       = '#99c4d8'
_ENDNODE_FC    = '#f6a21e'   # orange terminus / branch marker (candidate plots)
_OFFSET_M      = 160.0       # perpendicular spacing between interventions sharing a track


def plot_svc_interventions(base_infra: str, base_svc: str, result: Dict,
                           sa_polygon=None) -> List[str]:
    """Render per-svc-int delta plots + per-type and all-produced overlays (study area).

    Each svc-int's materialised delta is drawn on the SA infra (grey) with lakes and the
    service's stops as white circles (Code-annotated) and its short name at the termini.
    For EXT the pre-existing run is light green and only the new segment dark green; an
    NDC is a whole new line (orange). In the all-interventions overlays, interventions
    sharing a track are offset side-by-side. Returns the written file paths.
    """
    import matplotlib
    matplotlib.use('Agg')
    import ints_core as core

    materialised = [m for m in result.get('materialised', []) if m.get('projected_path')]
    if not materialised:
        return []

    ctx = _svc_plot_context(base_infra, base_svc, sa_polygon)
    written: List[str] = []
    combo = f"{base_infra}__{base_svc}"   # plots tree partition (decision H)

    by_type: Dict[str, List] = {'ext': [], 'ndc': []}
    for m in materialised:
        seg = _load_delta_segments(m.get('projected_path'))
        if seg is None or seg.empty:
            continue
        t = str(m.get('int_type') or ('ndc' if str(m['svc_int_id']).startswith('ndc') else 'ext'))
        info = _svc_int_plot_info(m, seg, t, ctx)
        by_type.setdefault(t, []).append(info)
        out_t = core.plot_out_dir(combo, t)
        p = out_t / f"svc_int_{m['svc_int_id']}_{base_infra}.pdf"
        if _render_svc_fig([info], ctx, p, f"{m['svc_int_id']} — {info['short']}", offset=False):
            written.append(str(p))

    for t, items in by_type.items():
        if not items:
            continue
        out_t = core.plot_out_dir(combo, t)
        p = out_t / f"svc_int_ALL_{t.upper()}_{base_infra}.pdf"
        if _render_svc_fig(items, ctx, p, f"All {t.upper()} svc-ints ({len(items)})", offset=True):
            written.append(str(p))

    allp = by_type.get('ext', []) + by_type.get('ndc', [])
    if allp:
        out_all = core.plot_out_dir(combo, None)
        p = out_all / f"svc_int_ALL_PRODUCED_{base_infra}.pdf"
        if _render_svc_fig(allp, ctx, p, "All svc-ints (EXT green, NDC orange)", offset=True):
            written.append(str(p))
    return written


def _svc_plot_context(base_infra: str, base_svc: str, sa_polygon) -> Dict:
    """Shared backdrop for every svc-int figure: SA extent, grey infra, lakes, base pairs."""
    import infrabuild_network_builder as ic

    sa_gdf = extent = None
    if sa_polygon is not None:
        sa_gdf = gpd.GeoSeries([sa_polygon], crs=SWISS_CRS)
        bx = sa_polygon.bounds
        m = 2000.0
        extent = (bx[0] - m, bx[2] + m, bx[1] - m, bx[3] + m)   # xmin, xmax, ymin, ymax

    backdrop = None
    node_xy: Dict[str, Tuple] = {}
    try:
        nodes, segs = ic.load_version(base_infra)
        backdrop = _clip_to_extent(segs, extent)
        for _, r in nodes.iterrows():
            nm = str(r.get('Name', '') or '')
            if nm and nm not in node_xy and r.geometry is not None:
                node_xy[nm] = (str(r.get('Code', '') or ''), r.geometry.x, r.geometry.y)
    except Exception as exc:
        print(f"  [plot]   (backdrop unavailable: {exc})")

    return {'sa_gdf': sa_gdf, 'extent': extent, 'backdrop': backdrop, 'node_xy': node_xy,
            'lakes': _load_lakes(extent), 'base_pairs': _base_route_pairs(base_svc)}


# ─────────────────────────────────────────────────────────────────────────────
# Candidate-overview plots (potential connections from the original termini)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ext_candidates(base_infra: str, base_svc: str, candidates: List[Dict],
                        sa_polygon=None):
    """Overview of every discovered EXT candidate as a numbered terminus→target line."""
    conns = _ext_candidate_conns(candidates)
    return _plot_candidates('ext', base_infra, base_svc, conns, sa_polygon, highlight='a',
                            endlabel='Terminus',
                            title=f"Extended-line candidates — {base_infra} ({len(conns)})")


def plot_ndc_candidates(base_infra: str, base_svc: str, ndc_candidates: List[Dict],
                        sa_polygon=None):
    """Overview of every NDC connecting-curve candidate as a numbered branch–branch line."""
    conns = _ndc_candidate_conns(ndc_candidates)
    return _plot_candidates('ndc', base_infra, base_svc, conns, sa_polygon, highlight='both',
                            endlabel='Branch station',
                            title=f"New-direct-connection candidates — {base_infra} ({len(conns)})")


def _ext_candidate_conns(candidates: Optional[List[Dict]]) -> List[Dict]:
    """EXT candidate dicts → deduped [{a, b, label}] (terminus → target)."""
    out: List[Dict] = []
    seen: set = set()
    for c in (candidates or []):
        ep, tgt = str(c.get('endpoint', '') or ''), str(c.get('target', '') or '')
        if not ep or not tgt:
            continue
        key = (ep, tgt)
        if key in seen:
            continue
        seen.add(key)
        sn = str(c.get('line_short_name', '') or '').strip()
        out.append({'a': ep, 'b': tgt,
                    'label': f"{len(out) + 1}: {sn + ' ' if sn else ''}{ep} → {tgt}"})
    return out


def _ndc_candidate_conns(ndc_candidates: Optional[List[Dict]]) -> List[Dict]:
    """NDC connecting-curve candidate dicts → deduped [{a, b, label}] (branch – branch)."""
    out: List[Dict] = []
    seen: set = set()
    for c in (ndc_candidates or []):
        a, b = c.get('branch_a'), c.get('branch_b')
        if not a or not b:
            continue
        key = frozenset((str(a), str(b)))
        if key in seen:
            continue
        seen.add(key)
        req = ','.join(str(r) for r in (c.get('requires_infra') or []))
        out.append({'a': str(a), 'b': str(b),
                    'label': f"{len(out) + 1}: {a} – {b}" + (f" ({req})" if req else "")})
    return out


def _plot_candidates(kind: str, base_infra: str, base_svc: str, conns: List[Dict],
                     sa_polygon, *, highlight: str, endlabel: str, title: str):
    """Draw candidate connections as numbered straight lines over the SA infra backdrop.

    Mirrors the legacy 'developments' / 'missing connections' overview logic with our
    styling: grey infra + lakes, each candidate a distinct colour with a numbered legend
    entry, the original termini (EXT) / branch stations (NDC) marked as orange end-nodes.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.lines import Line2D
    import ints_core as core
    import infrabuild_network_builder as ic

    if not conns:
        print(f"  [plot] no {kind.upper()} candidates — skipping candidate plot")
        return None

    ctx = _svc_plot_context(base_infra, base_svc, sa_polygon)
    nxy = ctx['node_xy']
    try:
        fig, ax = plt.subplots(figsize=(11, 13))
        ax.set_aspect('equal')
        ax.set_xlabel('E [m]', fontsize=10)
        ax.set_ylabel('N [m]', fontsize=10)
        ax.grid(True, alpha=0.3)

        if ctx['lakes'] is not None and not ctx['lakes'].empty:
            ctx['lakes'].plot(ax=ax, color=_LAKE_FC, edgecolor=_LAKE_EC, linewidth=0.3, zorder=0)
        if ctx['backdrop'] is not None and not ctx['backdrop'].empty:
            ctx['backdrop'].plot(ax=ax, color=_BACKDROP, linewidth=0.5, zorder=1)
        if ctx['sa_gdf'] is not None:
            ctx['sa_gdf'].boundary.plot(ax=ax, color='black', linewidth=0.8,
                                        linestyle='--', alpha=0.6, zorder=2)

        cmap = cm.get_cmap('tab20', max(len(conns), 2))
        endnodes: set = set()
        involved: set = set()
        handles: List = []
        for i, c in enumerate(conns):
            a, b = nxy.get(c['a']), nxy.get(c['b'])
            if not a or not b:
                continue
            col = cmap(i % cmap.N)
            ax.plot([a[1], b[1]], [a[2], b[2]], color=col, linewidth=3.0,
                    solid_capstyle='round', alpha=0.9, zorder=4)
            involved.update((c['a'], c['b']))
            endnodes.update((c['a'], c['b']) if highlight == 'both' else (c['a'],))
            handles.append(Line2D([0], [0], color=col, lw=3, label=c['label']))

        _plot_stations(ax, {nm: nxy[nm] for nm in involved if nm in nxy}, set())
        for nm in endnodes:
            p = nxy.get(nm)
            if p:
                ax.plot(p[1], p[2], marker='o', markersize=9, markerfacecolor=_ENDNODE_FC,
                        markeredgecolor='black', markeredgewidth=0.9, zorder=7)

        if ctx['extent'] is not None:
            ax.set_xlim(ctx['extent'][0], ctx['extent'][1])
            ax.set_ylim(ctx['extent'][2], ctx['extent'][3])

        base_handles = [
            Line2D([0], [0], color=_BACKDROP, lw=1.5, label='Unused infrastructure'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='black', markersize=6, label='Service stop'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=_ENDNODE_FC,
                   markeredgecolor='black', markersize=8, label=endlabel),
        ]
        ax.legend(handles=base_handles + handles, loc='upper right', fontsize=6)

        ax.set_title(title, fontsize=13, fontweight='bold')
        ic._add_north_arrow(ax, location='upper left', scale=0.5)
        ic._add_scale_bar(ax, location=(0.755, 0.012))
        plt.tight_layout()

        out_dir = core.plot_out_dir(core._combo(base_infra), kind)
        out_path = out_dir / f"{kind}_candidates_{base_infra}.pdf"
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"  [plot] wrote {out_path.name} ({len(conns)} candidate connection(s))")
        return str(out_path)
    except Exception as exc:
        print(f"  [plot]   WARNING {kind} candidates: {exc}")
        try:
            plt.close('all')
        except Exception:
            pass
        return None


def _svc_int_plot_info(m: Dict, seg, t: str, ctx: Dict) -> Dict:
    """Classify a delta's segments (old vs new), collect stops + termini + short name."""
    base_pairs: set = set()
    for r in (m.get('changed_route_ids') or []):
        base_pairs |= ctx['base_pairs'].get(str(r), set())

    has_names = 'from_stop_name' in seg.columns and 'to_stop_name' in seg.columns
    if has_names and base_pairs:
        is_old = seg.apply(
            lambda r: frozenset((str(r['from_stop_name']), str(r['to_stop_name']))) in base_pairs,
            axis=1)
    else:                                   # NDC (no base) or missing names → all new
        is_old = pd.Series(False, index=seg.index)

    stations = _collect_stations(seg)
    short = _short_name(seg, m)
    return {'type': t, 'id': m['svc_int_id'], 'short': short,
            'old': seg[is_old], 'new': seg[~is_old], 'stations': stations,
            'termini': _termini(seg, stations)}


def _render_svc_fig(infos: List[Dict], ctx: Dict, out_path, title: str,
                    offset: bool = False) -> bool:
    """Draw lakes + SA infra + the svc-int line(s) (old light / new dark) + stops → PDF."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import infrabuild_network_builder as ic
    try:
        fig, ax = plt.subplots(figsize=(11, 13))
        ax.set_aspect('equal')
        ax.set_xlabel('E [m]', fontsize=10)
        ax.set_ylabel('N [m]', fontsize=10)
        ax.grid(True, alpha=0.3)

        if ctx['lakes'] is not None and not ctx['lakes'].empty:
            ctx['lakes'].plot(ax=ax, color=_LAKE_FC, edgecolor=_LAKE_EC, linewidth=0.3, zorder=0)
        if ctx['backdrop'] is not None and not ctx['backdrop'].empty:
            ctx['backdrop'].plot(ax=ax, color=_BACKDROP, linewidth=0.5, zorder=1)
        if ctx['sa_gdf'] is not None:
            ctx['sa_gdf'].boundary.plot(ax=ax, color='black', linewidth=0.8,
                                        linestyle='--', alpha=0.6, zorder=2)

        n = len(infos)
        drawn_st: set = set()
        for i, info in enumerate(infos):
            dist = ((i - (n - 1) / 2.0) * _OFFSET_M) if (offset and n > 1) else 0.0
            dark = _NDC_COLOR if info['type'] == 'ndc' else _EXT_COLOR
            light = _NDC_COLOR_OLD if info['type'] == 'ndc' else _EXT_COLOR_OLD
            _plot_segs(ax, info['old'], light, 1.8, dist)
            _plot_segs(ax, info['new'], dark, 2.8, dist)
        for info in infos:
            _plot_stations(ax, info['stations'], drawn_st)
        for info in infos:
            color = _NDC_COLOR if info['type'] == 'ndc' else _EXT_COLOR
            _plot_termini_labels(ax, info, color)

        if ctx['extent'] is not None:
            ax.set_xlim(ctx['extent'][0], ctx['extent'][1])
            ax.set_ylim(ctx['extent'][2], ctx['extent'][3])

        types = {info['type'] for info in infos}
        handles: List = []
        if 'ext' in types:
            handles += [Line2D([0], [0], color=_EXT_COLOR_OLD, lw=2, label='Existing line (EXT)'),
                        Line2D([0], [0], color=_EXT_COLOR, lw=2.5, label='Extension (new)')]
        if 'ndc' in types:
            handles += [Line2D([0], [0], color=_NDC_COLOR, lw=2.5, label='New direct connection')]
        handles += [Line2D([0], [0], color=_BACKDROP, lw=1.5, label='Unused infrastructure'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                           markeredgecolor='black', markersize=6, label='Service stop')]
        ax.legend(handles=handles, loc='upper right', fontsize=7)

        ax.set_title(title, fontsize=13, fontweight='bold')
        ic._add_north_arrow(ax, location='upper left', scale=0.5)
        ic._add_scale_bar(ax, location=(0.755, 0.012))
        plt.tight_layout()
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        print(f"  [plot] wrote {Path(out_path).name}")
        return True
    except Exception as exc:
        print(f"  [plot]   WARNING {Path(out_path).name}: {exc}")
        try:
            plt.close('all')
        except Exception:
            pass
        return False


def _plot_segs(ax, gdf, color: str, lw: float, dist: float) -> None:
    """Plot a segment GeoDataFrame, optionally shifted perpendicular by ``dist`` metres."""
    if gdf is None or gdf.empty:
        return
    geoms = gdf.geometry
    if dist:
        geoms = geoms.apply(lambda g: _safe_offset(g, dist))
    gpd.GeoSeries(list(geoms.values), crs=SWISS_CRS).plot(
        ax=ax, color=color, linewidth=lw, zorder=3)


def _safe_offset(geom, dist: float):
    """Parallel-offset a (Multi)LineString by ``dist`` m; fall back to original on failure.

    The projected delta geometries are MultiLineStrings (which have no ``offset_curve``),
    so merge to a LineString first and offset each resulting part — that is why the earlier
    naive ``geom.offset_curve`` silently no-op'd and the overlays overlapped.
    """
    if not dist or geom is None or geom.is_empty:
        return geom
    from shapely.ops import linemerge
    from shapely.geometry import MultiLineString

    def _off_ls(ls):
        try:
            o = ls.offset_curve(dist)
            return o if (o is not None and not o.is_empty) else ls
        except Exception:
            return ls

    try:
        g = linemerge(geom) if geom.geom_type == 'MultiLineString' else geom
    except Exception:
        g = geom
    if g.geom_type == 'LineString':
        return _off_ls(g)
    if g.geom_type == 'MultiLineString':
        parts: List = []
        for sub in g.geoms:
            r = _off_ls(sub)
            parts.extend(list(r.geoms) if r.geom_type == 'MultiLineString' else [r])
        return MultiLineString(parts) if parts else geom
    return geom


def _plot_stations(ax, stations: Dict, drawn: set) -> None:
    """White circle + Code label per stop (deduplicated across the whole figure)."""
    for name, (code, x, y) in stations.items():
        if name in drawn or x is None or y is None:
            continue
        drawn.add(name)
        ax.plot(x, y, marker='o', markersize=5, markerfacecolor='white',
                markeredgecolor='black', markeredgewidth=0.8, zorder=5)
        if code:
            ax.annotate(code, xy=(x, y), xytext=(4, 4), textcoords='offset points',
                        fontsize=6, fontweight='bold', color='#333333', zorder=6,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))


def _plot_termini_labels(ax, info: Dict, color: str) -> None:
    """Annotate the service short name (S3, IR13 …) at each terminus of the line.

    Placed to the bottom-left of the stop so it never overlaps the Code label, which
    sits to the top-right (see _plot_stations).
    """
    for name, x, y in info['termini']:
        if x is None or y is None:
            continue
        ax.annotate(info['short'], xy=(x, y), xytext=(-6, -8), textcoords='offset points',
                    ha='right', va='top', fontsize=6, fontweight='bold', color=color, zorder=7,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=color, alpha=0.85))


def _collect_stations(seg) -> Dict:
    """name → (Code, E, N) for every stop touched by the delta segments."""
    out: Dict = {}
    for _, r in seg.iterrows():
        for nk, ck, ek, nk2 in (('from_stop_name', 'from_code', 'from_stop_E', 'from_stop_N'),
                                ('to_stop_name', 'to_code', 'to_stop_E', 'to_stop_N')):
            name = str(r.get(nk, '') or '')
            if name and name not in out:
                ex, ny = r.get(ek), r.get(nk2)
                out[name] = (str(r.get(ck, '') or ''),
                             float(ex) if pd.notna(ex) else None,
                             float(ny) if pd.notna(ny) else None)
    return out


def _termini(seg, stations: Dict) -> List:
    """Degree-1 stops of the (undirected) delta line → [(name, E, N)] termini."""
    from collections import Counter
    edges: set = set()
    for _, r in seg.iterrows():
        a, b = str(r.get('from_stop_name', '') or ''), str(r.get('to_stop_name', '') or '')
        if a and b and a != b:
            edges.add(frozenset((a, b)))
    deg: Counter = Counter()
    for e in edges:
        for nm in e:
            deg[nm] += 1
    out: List = []
    for nm, d in deg.items():
        if d == 1 and nm in stations:
            _code, x, y = stations[nm]
            out.append((nm, x, y))
    return out


def _short_name(seg, m: Dict) -> str:
    """The service short name for labels (the Service column; fall back to the id)."""
    if 'Service' in seg.columns:
        vals = [str(v) for v in seg['Service'].dropna().tolist() if str(v).strip()]
        if vals:
            return max(set(vals), key=vals.count)
    return str(m.get('svc_int_id', ''))


def _base_route_pairs(base_svc: str) -> Dict[str, set]:
    """route_id → set of frozenset(from_name, to_name) over the BASE service network.

    Lets the delta plot mark only the genuinely-new stop-pairs (the extension) dark.
    """
    pairs: Dict[str, set] = {}
    try:
        _lines, segs, _stops = _load_base_unprojected(base_svc)
    except Exception as exc:
        print(f"  [plot]   (base pairs unavailable: {exc})")
        return pairs
    for segdf in segs.values():
        for _, r in segdf.iterrows():
            rid = str(r.get('GTFS_ID', '') or '')
            a, b = str(r.get('from_stop_name', '') or ''), str(r.get('to_stop_name', '') or '')
            if rid and a and b:
                pairs.setdefault(rid, set()).add(frozenset((a, b)))
    return pairs


def _clip_to_extent(gdf, extent):
    """Clip a GeoDataFrame to the SA extent box (xmin, xmax, ymin, ymax); pass through on error."""
    if gdf is None or extent is None or gdf.empty:
        return gdf
    from shapely.geometry import box as _box
    try:
        bx = gpd.GeoDataFrame(
            geometry=[_box(extent[0], extent[2], extent[1], extent[3])],
            crs=gdf.crs or SWISS_CRS)
        return gpd.clip(gdf, bx)
    except Exception:
        return gdf


def _load_lakes(extent):
    """Lakes shapefile clipped to the SA extent (or None)."""
    p = Path(paths.MAIN) / paths.LAKES_SHP
    if not p.exists():
        return None
    try:
        return _clip_to_extent(gpd.read_file(p), extent)
    except Exception:
        return None


def _load_delta_segments(projected_path):
    """Load all layers of a svc-int's projected rail_segments.gpkg into one GeoDataFrame."""
    if not projected_path:
        return None
    p = Path(projected_path)
    if not p.exists():
        return None
    try:
        frames = [gpd.read_file(p, layer=L) for L in fiona.listlayers(str(p))]
    except Exception:
        return None
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return None
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=SWISS_CRS)


def materialise_and_plot(
    base_infra: str,
    base_svc: str,
    *,
    ext_ids: Optional[List[str]] = None,
    ndc_ids: Optional[List[str]] = None,
    make_plots: bool = True,
    use_cache: bool = False,
    sa_polygon=None,
) -> Dict:
    """Materialise the given svc-int ids (delta networks) and optionally plot them.

    Standalone-CLI convenience giving the per-type EXT/NDC files the same discover →
    materialise → plot flow as the cc CLI, without re-running discovery. Plots need the
    materialised projected deltas, so they are produced together behind one toggle.
    """
    result: Dict = {'ext_ids': list(ext_ids or []), 'ndc_ids': list(ndc_ids or []),
                    'materialised': [], 'plots': []}
    todo = ([('ext', i) for i in result['ext_ids']] +
            [('ndc', i) for i in result['ndc_ids']])
    print(f"  [apply] materialising {len(todo)} svc-int delta network(s)…")
    for int_type, iid in todo:
        rec = read_record(int_type, iid)
        if rec is None:
            continue
        try:
            result['materialised'].append(
                apply_svc_int(rec, base_svc, base_infra, use_cache=use_cache))
        except Exception as exc:
            print(f"  [apply]   WARNING {iid} failed: {exc}")
    if make_plots:
        try:
            result['plots'] = plot_svc_interventions(base_infra, base_svc, result, sa_polygon)
        except Exception as exc:
            print(f"  [plot] WARNING: svc-int plots failed: {exc}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Standalone CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.chdir(paths.MAIN)
    import ints_core as core

    core.cli_header("infraScanRail — Service Interventions (Phase 5B · EXT/NDC)")

    core.cli_step(1, "Base infrastructure network?")
    base = core.cli_pick("Infra version:", core.cli_infra_versions(), core._resolve_base_version_propagated())

    core.cli_step(2, "Service network to extend / connect?")
    svc = core.cli_pick("Service version:", core.cli_svc_versions(), core._resolve_svc_version())

    core.cli_step(3, "Intervention mode?")
    mode = core.cli_pick("SVC_INT_MODE:", ['NONE', 'EXT', 'NDC', 'ALL'],
                         getattr(settings, 'SVC_INT_MODE', 'NONE'))

    core.cli_step(4, "Generate svc-int delta plots?")
    make_plots = core.cli_pick_yesno("Plots?", getattr(settings, 'PLOT_SVC_INTS', False))

    res = phase_5b_service_interventions(
        base_infra=base, base_svc=svc, mode=mode, make_plots=make_plots,
        sa_polygon=core._load_polygon(), buffer_polygon=core._load_buffer(),
        interactive=True)
    print(f"\n=== Phase 5B: {len(res['ext_ids'])} EXT + {len(res['ndc_ids'])} NDC, "
          f"{len(res['materialised'])} materialised, {len(res['plots'])} plot(s) ===")
