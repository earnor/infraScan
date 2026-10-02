"""
svc_ints_orchestrator — Phase 5B engine + entry point + standalone CLI.
Last modified: 2026-06-20

Single home for the service-intervention stack (mirrors infra_ints_orchestrator):

  • Registry I/O + schema       (the svc-int catalogue, xlsx)
  • apply_svc_int               (delta materialisation — added in Phase 2)
  • Phase 5B orchestration + standalone CLI   (added in Phase 5)

Per-type discovery lives in the sibling modules — ``svc_ints_extend_lines`` (EXT),
``svc_ints_new_direct_connections`` (NDC), ``svc_ints_frequency`` (FRQ) and
``svc_ints_stop_patterns`` (STP); all import THIS module for registry helpers and are
imported lazily (inside the phase function) to avoid an import cycle.

Registry (declarative source of truth)
--------------------------------------
One logical record per service intervention (extended line 'ext' / new direct
connection 'ndc' / frequency change 'frq' / stopping-pattern change 'stp'), identified
by ``int_id`` and stored as a row of the ``extensions`` sheet of a per-type xlsx
(``ext_interventions.xlsx`` / ``ndc_interventions.xlsx`` / ``frq_interventions.xlsx`` /
``stp_interventions.xlsx``).
A record stores the *declarative operation list* (the service delta), keyed by the
in-run ``(route_id, direction_id, variant_rank)`` of the line it targets — valid
because svc-ints are generated against the active version each run.

  operations     : [{op, params}, …] — extend / truncate / reroute / set_frequency /
                   add_stop / drop_stop on an existing line (applied to both
                   direction_id rows), or new_line for an NDC.
  requires_infra : the cc_id(s) the intervention activates (empty for pure EXT).
  affected_*     : affected-set primaries (stations / services) — Phase 6 expands
                   these to the exact closure.

The materialised form (a complete projected rail_lines/rail_segments delta per
svc-int) is produced on demand by ``apply_svc_int`` (Phase 2), at a deterministic
per-svc-int path the existing catchment/OD/routing readers consume unchanged.
"""

import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import fiona
from shapely.geometry import LineString, Point
from shapely.ops import linemerge, substring

import cache_manifest
import paths
import settings

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# All svc-int registries store their rows in one sheet (historically read by name).
SHEET = 'extensions'

SUPPORTED_SVC_INT_TYPES = ('ext', 'ndc', 'frq', 'stp')

# Column order of the catalogue row (the declarative svc-int record).
# twin_of: id of an op-identical svc-int of another type ('' when unset) — set by FRQ
# discovery when a corridor extension duplicates a registered EXT; Phase 6 reuses the
# twin's outputs instead of recomputing.
RECORD_COLS: tuple = (
    'int_id', 'int_type', 'base_authored', 'svc_version',
    'route_id', 'direction_id', 'variant_rank',
    'operations', 'total_dep', 'line_type', 'mode_class', 'line_short_name',
    'requires_infra', 'affected_stations', 'affected_services', 'twin_of',
)

# Comma-joined list fields (stored as strings in the xlsx cell).
_LIST_FIELDS: tuple = ('requires_infra', 'affected_stations', 'affected_services')

# int_id prefix per type (e.g. ext_100001, ndc_103001, frq_104001, stp_105001).
_TYPE_CODE: Dict[str, str] = {'ext': 'ext', 'ndc': 'ndc', 'frq': 'frq', 'stp': 'stp'}

# Per-type id start block (mirrors the infra-int DEV_ID_START_* convention).
_TYPE_START: Dict[str, str] = {
    'ext': 'DEV_ID_START_EXT',
    'ndc': 'DEV_ID_START_NDC',
    'frq': 'DEV_ID_START_FRQ',
    'stp': 'DEV_ID_START_STP',
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


def _purge_svc_int_outputs(int_ids: List[str], combo: str) -> None:
    """Delete all per-svc-int output folders of de-registered ids (active combo).

    Svc-int ids restart from the DEV_ID_START_* blocks each run, so once 5B
    clears the registry the old ids' artifacts are stale and a regenerated id
    of the same name would silently mix with them — disk must mirror the live
    catalogue. Covers the five data trees + their plots mirrors, the 5C
    cap/<id>/ workbooks and the per-id Developments PDFs; transitional: also
    sweeps the legacy flat '<id>_network' layout (pre-2026-06-10).
    """
    if not int_ids:
        return
    data_roots = (paths.RAIL_LINES_DIR, paths.CATCHMENT_AREA_DIR,
                  paths.TRAFFIC_FLOW_OD_DIR, paths.TRAFFIC_FLOW_ASSIGNMENT_DIR,
                  paths.SCENARIO_DIR)
    plot_roots = (paths.CATCHMENT_PLOTS_DIR, paths.TRAFFIC_FLOW_OD_PLOTS_DIR,
                  paths.TRAFFIC_FLOW_ASSIGNMENT_PLOTS_DIR)
    n_dirs = n_pdfs = 0
    for iid in (str(i) for i in int_ids):
        names = (paths.svc_int_network_name(iid, combo), f'{iid}_network')
        dirs = [Path(paths.MAIN) / root / name
                for root in data_roots + plot_roots for name in names]
        dirs.append(Path(paths.get_svc_int_cap_dir(combo)) / iid)
        for d in dirs:
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                n_dirs += 1
        for subtype in SUPPORTED_SVC_INT_TYPES:
            pdir = Path(paths.get_developments_plot_dir(combo, subtype))
            if pdir.is_dir():
                for p in pdir.glob(f'svc_int_{iid}_*.pdf'):
                    p.unlink(missing_ok=True)
                    n_pdfs += 1
    print(f"  [5B] purged {n_dirs} output folder(s) + {n_pdfs} plot PDF(s) of "
          f"{len(int_ids)} de-registered svc-int(s)")


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
        elif col in ('twin_of', 'route_id'):
            # empty xlsx cells read as NaN (truthy float) — normalise to ''
            # (route_id is '' for the multi-route STP combined int, which keys
            # its routes off affected_services instead).
            val = '' if val is None or (isinstance(val, float) and pd.isna(val)) \
                else str(val).strip()
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
    combo = f'{base_infra_version}__{base_svc_version}'
    out_dir = Path(paths.get_svc_int_network_dir(svc_int_id, combo))
    unproj_dir = out_dir / paths.SERVICES_UNPROJECTED_SUBDIR
    proj_dir = out_dir / base_infra_version
    projected_path = proj_dir / 'rail_segments.gpkg'

    if (use_cache and projected_path.exists()
            and cache_manifest.check_manifest(
                paths.get_svc_int_catalogue_dir(combo),
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

    delta_lines, delta_segs, delta_stop_nrs, changed_routes, affected, stp_proj_cache = \
        _build_delta(svc_int, base_lines, base_segs, resolve,
                     base_svc_version, base_infra_version)
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
    # svc_version here is a display label only (output dirs are explicit below);
    # keep it flat — the combo-keyed name contains path separators that would
    # break the projection's plot filenames.
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
    # Flagged terminus reversals (EXT, reversal_at_endpoint) are allowed by design
    # (penalised via IVWT) — exempt them from the inter-leg backtracking reroute.
    exempt = {str(o.get('params', {}).get('endpoint', '')).strip()
              for o in (svc_int.get('operations') or [])
              if o.get('op') == 'extend' and o.get('params', {}).get('reversal_at_endpoint')}
    exempt.discard('')
    # STP injects split/merge sub-hops into the reuse cache so their geometry is a
    # verbatim slice of the base path (never re-routed); they override/extend the
    # base unchanged-pair cache.
    reuse_lookup = dict(_build_base_projection_cache(
        base_svc_version, base_infra_version))
    if stp_proj_cache:
        reuse_lookup.update(stp_proj_cache)
    enriched = ssp.project_lines(
        seg_gdf, config, run_zvv=run_zvv,
        base_projection_lookup=reuse_lookup,
        reversal_exempt_stops=exempt or None)
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
    combo = f'{base_infra_version}__{base_svc_version}'
    out_dir = Path(paths.get_svc_int_network_dir(svc_int_id, combo)) / 'Merged'
    targets = {n: out_dir / f'{n}.gpkg'
               for n in ('rail_lines', 'rail_segments', 'rail_stops')}
    if (use_cache and all(p.exists() for p in targets.values())
            and cache_manifest.check_manifest(
                paths.get_svc_int_catalogue_dir(combo),
                'svc_ints_5b',
                {'infra_version': base_infra_version,
                 'svc_version': base_svc_version})):
        print(f"  [merge] {svc_int_id}: cached merged network at {out_dir} — reuse")
        return str(out_dir)

    base_dir = (Path(paths.MAIN) / paths.RAIL_LINES_DIR /
                f"{base_svc_version}_network" / paths.SERVICES_UNPROJECTED_SUBDIR)
    delta_dir = (Path(paths.get_svc_int_network_dir(svc_int_id, combo)) /
                 paths.SERVICES_UNPROJECTED_SUBDIR)
    proj_delta = Path(paths.get_svc_int_projected_path(svc_int_id, base_infra_version,
                                                       combo))

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

def _build_delta(svc_int, base_lines, base_segs, resolve,
                 base_svc=None, base_infra=None):
    """Apply the op list → per-layer delta lines/segments + touched stops.

    Returns (delta_lines, delta_segs, stop_nrs, changed_routes, affected,
    stp_proj_cache). stp_proj_cache is the supplementary base-projection-reuse
    cache for STP's split/merge sub-hops (empty {} for ext/ndc/frq).
    """
    ops = svc_int.get('operations') or []
    route_id = str(svc_int.get('route_id', ''))
    variant_rank = svc_int.get('variant_rank', 1)
    is_new = (svc_int.get('int_type') == 'ndc') or any(o.get('op') == 'new_line' for o in ops)

    if any(o.get('op') in ('add_stop', 'drop_stop') for o in ops):
        return _build_stp_delta(svc_int, base_lines, base_segs, resolve,
                                base_svc, base_infra)

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
        short = svc_int.get('line_short_name') or svc_int_line_name(svc_int)
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
                # AND reverse the stop list (stored order is the dir-0 apply
                # order — single-stop EXT is order-invariant, multi-stop FRQ not)
                fe = from_end if dir_id == '0' else _flip[from_end]
                stops_d = (list(params.get('stops', [])) if dir_id == '0'
                           else list(reversed(params.get('stops', []))))
                dir_ops = [{'op': 'extend',
                            'params': {'from_end': fe, 'stops': stops_d}}
                           if o.get('op') == 'extend' else o for o in ops]
                seq, total_dep_override = _apply_ops_to_seq(base_seq, dir_ops, resolve)
                seq = [s for s in seq if s is not None]
                if len(seq) < 2:
                    continue
                base_dep = int(line_row.get('total_dep', 0) or 0)
                if isinstance(total_dep_override, tuple):       # ('factor', f)
                    factor = float(total_dep_override[1])
                    total_dep = int(round(base_dep * factor))
                else:
                    factor = None
                    total_dep = int(total_dep_override if total_dep_override is not None
                                    else base_dep)
                base_tt = {(int(r['from_stop_nr']), int(r['to_stop_nr'])): r
                           for _, r in line_seg_df.iterrows()}
                # modified line keeps its base name + suffix ('S14_EXT5')
                short = (svc_int.get('line_short_name')
                         or svc_int_line_name(svc_int,
                                              base_short=line_row.get('line_short_name')))
                meta = {'route_id': route_id, 'short': short,
                        'line_type': int(line_row.get('line_type', 109)),
                        'mode_label': line_row.get('mode_label', 'rail'),
                        'variant_rank': var}
                if svc_int.get('int_type') == 'frq':
                    # FRQ contract (CAP-round F8): deltas carry REAL per-period
                    # rates — base rates scaled by the frequency factor, period
                    # label preserved (peak_only doubles to 4/4/0, not flat).
                    f = factor if factor is not None else 1.0
                    meta['service_period'] = str(line_row.get('service_period')
                                                 or 'all_day')
                    meta['period_rates'] = tuple(
                        round(float(line_row.get(col, 0) or 0) * f, 3)
                        for col in ('freq_am_peak_dep_hr', 'freq_pm_peak_dep_hr',
                                    'freq_offpeak_dep_hr'))
                _emit_line(delta_lines, delta_segs, layer, meta, dir_id, seq, total_dep, base_tt)
                stop_nrs.update(s['nr'] for s in seq)
                affected.update(s['name'] for s in seq)
                changed_routes.add(route_id)

        # Flagged terminus reversal: the extension hop carries the direction-change
        # dwell as IVWT (both direction rows; GC-effective once routing reads IVWT).
        pen = float(getattr(settings, 'EXT_TERMINUS_REVERSAL_PENALTY_MIN', 2.0))
        first = (params.get('stops') or [None])[0]
        if pen > 0 and first and params.get('reversal_at_endpoint'):
            hop = frozenset((str(endpoint), str(first)))
            n_pen = 0
            for rows in delta_segs.values():
                for row in rows:
                    if frozenset((str(row['from_stop_name']),
                                  str(row['to_stop_name']))) == hop:
                        row['IVWT'] = float(row.get('IVWT', 0.0) or 0.0) + pen
                        n_pen += 1
            if n_pen:
                print(f"  [apply] terminus reversal at {endpoint}: +{pen:.0f} min IVWT "
                      f"on {n_pen} extension-hop row(s)")

    return delta_lines, delta_segs, stop_nrs, changed_routes, affected, {}


# ─────────────────────────────────────────────────────────────────────────────
# STP — stopping-pattern delta (add_stop / drop_stop)
#
# Edits the CALLING PATTERN (which traversed stations a service stops at) while
# keeping the ROUTED PATH identical. Affected hops are SPLIT (add) or MERGED
# (drop) slices of the base projected path, so geometry / Via_Segment / path_nodes
# stay a verbatim part of the base corridor; they are injected into the
# projection reuse cache (keyed like _build_base_projection_cache) so project_lines
# reuses them instead of re-routing. Sub-hop TT is borrowed from a co-stopper /
# co-passer real GTFS time where one exists on the corridor, else modelled with the
# kinematic stop penalty (settings.STP_STOP_RUNTIME_PENALTY_MIN). Decision F1.
# ─────────────────────────────────────────────────────────────────────────────

_STP_BASE_HOPS: Dict[Tuple[str, str], Tuple[Dict, Dict]] = {}


def _stp_base_hops(base_svc, base_infra):
    """Memoised base projected hops for STP.

    Returns (hops_by_variant, borrow_index):
      hops_by_variant : (GTFS_ID, direction_id, variant_rank) -> ordered chain of
                        hop dicts (path_nodes list, geometry, Via_*, length, TT,
                        IVWT, node/code cols).
      borrow_index    : (from_name, to_name) -> hop dict, over GTFS-timed hops of
                        ANY service (the real stopping/passing time source).
    """
    mkey = (str(base_svc), str(base_infra))
    if mkey in _STP_BASE_HOPS:
        return _STP_BASE_HOPS[mkey]
    hops_by_variant: Dict[Tuple, List[Dict]] = {}
    borrow: Dict[Tuple[str, str], Dict] = {}
    proj = paths.get_projected_services_path(base_svc, base_infra) if base_svc else None
    if not proj or not Path(proj).exists():
        print(f"  [apply] STP: base projected services missing at {proj}")
        _STP_BASE_HOPS[mkey] = (hops_by_variant, borrow)
        return hops_by_variant, borrow

    def _pn(s):
        out = []
        for tok in str(s or '').split(';'):
            tok = tok.strip()
            if tok:
                try:
                    out.append(int(float(tok)))
                except ValueError:
                    pass
        return out

    grouped: Dict[Tuple, List[Dict]] = {}
    for layer in fiona.listlayers(proj):
        g = gpd.read_file(proj, layer=layer)
        for _, r in g.iterrows():
            if pd.isna(r.get('from_stop_nr')) or pd.isna(r.get('to_stop_nr')):
                continue
            hop = {
                'from_nr': int(float(r['from_stop_nr'])),
                'to_nr': int(float(r['to_stop_nr'])),
                'from_name': str(r.get('from_stop_name', '')),
                'to_name': str(r.get('to_stop_name', '')),
                'path_nodes': _pn(r.get('path_nodes')),
                'geometry': r.geometry,
                'path_length_m': r.get('path_length_m'),
                'TT': float(r['TT']) if pd.notna(r.get('TT')) else np.nan,
                'IVWT': float(r['IVWT']) if pd.notna(r.get('IVWT')) else 0.0,
                'tt_source': str(r.get('tt_source', '')),
                'from_code': r.get('from_code', ''), 'to_code': r.get('to_code', ''),
            }
            key = (str(r.get('GTFS_ID', '')), str(r.get('direction_id', '')),
                   int(float(r.get('variant_rank', 1) or 1)))
            grouped.setdefault(key, []).append(hop)
            if hop['tt_source'].lower() == 'gtfs':
                borrow.setdefault((hop['from_name'], hop['to_name']), hop)

    for key, hl in grouped.items():
        by_from = {h['from_nr']: h for h in hl}
        starts = {h['from_nr'] for h in hl} - {h['to_nr'] for h in hl}
        cur = sorted(starts)[0] if starts else hl[0]['from_nr']
        chain, seen = [], set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            h = by_from.get(cur)
            if h is None:
                break
            chain.append(h)
            cur = h['to_nr']
        hops_by_variant[key] = chain if chain else hl
    _STP_BASE_HOPS[mkey] = (hops_by_variant, borrow)
    return hops_by_variant, borrow


def _stp_route_ops(svc_int):
    """(route_id, op) pairs — single-route uses the record route_id; the combined
    multi-route int carries route_id per add_stop/drop_stop op."""
    rec_route = str(svc_int.get('route_id', '') or '')
    out = []
    for o in svc_int.get('operations') or []:
        if o.get('op') not in ('add_stop', 'drop_stop'):
            continue
        rid = str(o.get('params', {}).get('route_id', '') or rec_route)
        out.append((rid, o))
    return out


def _stp_cache_payload(geom, path_nodes_list, length, from_nr, to_nr,
                       from_code, to_code, method):
    """Supplementary projection-reuse payload (shape of _build_base_projection_cache)."""
    import services_service_projection as ssp
    pn_str = ";".join(str(n) for n in path_nodes_list)
    via_nodes, via_seg = ssp._make_via_cols(pn_str)
    return {
        'node_id_from': from_nr, 'node_id_to': to_nr,
        'match_method_from': method, 'match_method_to': method,
        'from_code': from_code or '', 'to_code': to_code or '',
        'Via_Nodes': via_nodes, 'Via_Segment': via_seg,
        'Via_Station': '', 'Via_Junction': '',
        'path_nodes': pn_str, 'path_length_m': length,
        '_path_tt_min': 0.0, 'needs_correction': False, 'elec_mismatch': False,
        'geometry': geom, 'FromCode': from_nr, 'ToCode': to_nr,
    }


def _as_linestring(geom):
    """Coerce a (possibly Multi)LineString to a single LineString for substring."""
    if geom is None:
        return None
    if geom.geom_type == 'LineString':
        return geom
    merged = linemerge(geom)
    return merged if merged.geom_type == 'LineString' else None


def _force_linestring(geom):
    """Coerce a (Multi)LineString to ONE LineString, bridging the small (~metre)
    seams that defeat linemerge.

    A long passing host hop (e.g. Oerlikon->Uster, a stop never called by the
    base service) projects to a MultiLineString whose parts meet within a few
    metres but not exactly, so linemerge keeps them apart and _as_linestring
    returns None — which left every added-stop sub-hop with null geometry. Chain
    the parts greedily (flipping as needed) so substring() has a continuous line.
    """
    if geom is None:
        return None
    if geom.geom_type == 'LineString':
        return geom
    merged = linemerge(geom)
    if merged.geom_type == 'LineString':
        return merged
    parts = (list(merged.geoms) if merged.geom_type == 'MultiLineString'
             else list(getattr(geom, 'geoms', [])))
    parts = [p for p in parts if p is not None and not p.is_empty]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    coords = list(parts[0].coords)
    remaining = parts[1:]
    while remaining:
        tx, ty = coords[-1]
        best_i, best_flip, best_d = 0, False, None
        for i, p in enumerate(remaining):
            pc = p.coords
            d0 = (pc[0][0] - tx) ** 2 + (pc[0][1] - ty) ** 2
            d1 = (pc[-1][0] - tx) ** 2 + (pc[-1][1] - ty) ** 2
            if best_d is None or min(d0, d1) < best_d:
                best_d, best_i, best_flip = min(d0, d1), i, d1 < d0
        pc = list(remaining.pop(best_i).coords)
        if best_flip:
            pc = pc[::-1]
        coords += pc[1:] if pc and pc[0] == coords[-1] else pc
    return LineString(coords)


def _build_stp_delta(svc_int, base_lines, base_segs, resolve, base_svc, base_infra):
    """STP delta: split (add_stop) / merge (drop_stop) the calling pattern.

    Returns the 6-tuple (delta_lines, delta_segs, stop_nrs, changed_routes,
    affected, stp_proj_cache). Geometry of the affected sub-hops comes verbatim
    from the base projected path; nothing on the route is re-routed.
    """
    import services_service_projection as ssp

    penalty = float(getattr(settings, 'STP_STOP_RUNTIME_PENALTY_MIN', 1.0))
    default_ivwt = float(getattr(settings, 'SVC_INT_DEFAULT_IVWT_MIN', 0.5))
    hops_by_variant, borrow = _stp_base_hops(base_svc, base_infra)

    delta_lines: Dict[str, List[Dict]] = {}
    delta_segs: Dict[str, List[Dict]] = {}
    stop_nrs: set = set()
    changed_routes: set = set()
    affected: set = set()
    stp_proj_cache: Dict[Tuple, Dict] = {}

    for rid, op in _stp_route_ops(svc_int):
        kind = op.get('op')
        names = list(op.get('params', {}).get('stops', []))
        # iterate the route's PROJECTED chains (the variants that actually project —
        # base_lines may carry degenerate variants with no projected service)
        chain_keys = sorted(k for k in hops_by_variant if k[0] == rid)
        if not chain_keys:
            print(f"  [apply]   STP {rid}: no projected chain — skipped")
            continue
        for (_rid, dir_id, var) in chain_keys:
            layer, line_row, _ = _find_target(base_lines, base_segs, rid, dir_id, var)
            if line_row is None:
                continue
            chain = hops_by_variant[(rid, dir_id, var)]
            short = (svc_int.get('line_short_name')
                     or svc_int_line_name(svc_int,
                                          base_short=line_row.get('line_short_name')))
            meta = {'route_id': rid, 'short': short,
                    'line_type': int(line_row.get('line_type', 109)),
                    'mode_label': line_row.get('mode_label', 'rail'),
                    'variant_rank': var}
            total_dep = int(line_row.get('total_dep', 0) or 0)

            if kind == 'drop_stop':
                new_seq, synth_tt, cache = _stp_apply_drop(
                    chain, set(names), rid, dir_id, var, resolve, borrow,
                    penalty, ssp)
            else:
                new_seq, synth_tt, cache = _stp_apply_add(
                    chain, names, rid, dir_id, var, resolve, borrow,
                    penalty, default_ivwt, ssp)
            if len(new_seq) < 2:
                continue
            stp_proj_cache.update(cache)
            _emit_line(delta_lines, delta_segs, layer, meta, dir_id,
                       new_seq, total_dep, synth_tt)
            stop_nrs.update(s['nr'] for s in new_seq)
            affected.update(s['name'] for s in new_seq)
            changed_routes.add(rid)

    return delta_lines, delta_segs, stop_nrs, changed_routes, affected, stp_proj_cache


def _stp_stop_dict(name, nr, resolve):
    """{nr,name,E,N} for a chain endpoint name (resolve gives base coords)."""
    s = resolve(name)
    if s is not None:
        return s
    return {'nr': int(nr), 'name': str(name), 'E': 0.0, 'N': 0.0}


def _stp_apply_drop(chain, dropped, rid, dir_id, var, resolve, borrow, penalty, ssp):
    """Merge base hops around each dropped intermediate stop (run-time saving)."""
    base_names = [chain[0]['from_name']] + [h['to_name'] for h in chain]
    base_nrs = [chain[0]['from_nr']] + [h['to_nr'] for h in chain]
    keep = [i for i, nm in enumerate(base_names)
            if not (nm in dropped and 0 < i < len(base_names) - 1)]

    new_seq = [_stp_stop_dict(base_names[i], base_nrs[i], resolve) for i in keep]
    synth_tt: Dict[Tuple[int, int], Dict] = {}
    cache: Dict[Tuple, Dict] = {}

    for j in range(len(keep) - 1):
        i0, i1 = keep[j], keep[j + 1]
        consumed = chain[i0:i1]
        # Key the preserved/merged TT by the SAME stop numbers _emit_line uses
        # (the resolve()-based new_seq nrs), not the base projected-chain nrs.
        # Major hubs (e.g. Zürich HB: 8516144 in the projected chain vs 8503000
        # from the resolver) carry two ids; keying by base_nrs made _emit_line's
        # lookup miss for those hops, dropping the GTFS time to the kinematic
        # formula on segments the drop never touched.
        a_nr, b_nr = new_seq[j]['nr'], new_seq[j + 1]['nr']
        a_nm, b_nm = base_names[i0], base_names[i1]
        ivwt = float(consumed[0]['IVWT'] or 0.0)
        if len(consumed) == 1:                      # unchanged hop — base reuse
            synth_tt[(a_nr, b_nr)] = {'TT': consumed[0]['TT'], 'IVWT': ivwt}
            continue
        # merge: concat path_nodes (drop the seam duplicates), linemerge geometry
        pn: List[int] = list(consumed[0]['path_nodes'])
        for h in consumed[1:]:
            pn += h['path_nodes'][1:]
        geom = _as_linestring(_safe_union([h['geometry'] for h in consumed]))
        length = sum(float(h.get('path_length_m') or 0) for h in consumed)
        b = borrow.get((a_nm, b_nm))
        if b is not None and pd.notna(b['TT']):     # a real co-passer time exists
            # Borrow the co-passer's run TIME only — the dropped-stop line still runs the
            # verbatim concatenated path (a co-passer's A->C express hop is a shorter
            # geometry; overriding here mis-measured the line's length -> negative train-km).
            tt = float(b['TT'])
        else:
            n_dropped = len(consumed) - 1
            tt = sum(float(h['TT'] or 0) for h in consumed) - penalty * n_dropped
            tt = max(tt, 0.1)
        synth_tt[(a_nr, b_nr)] = {'TT': tt, 'IVWT': ivwt}
        key = ssp.base_reuse_key(rid, dir_id, var, a_nm, b_nm)
        cache[key] = _stp_cache_payload(geom, pn, length, a_nr, b_nr,
                                        consumed[0]['from_code'],
                                        consumed[-1]['to_code'], 'stp_merge')
    return new_seq, synth_tt, cache


def _stp_apply_add(chain, names, rid, dir_id, var, resolve, borrow,
                   penalty, default_ivwt, ssp):
    """Split each host base hop at the added stops it traverses (run-time penalty)."""
    # map each added station to its host hop + path index
    added_by_hop: Dict[int, List[Tuple]] = {}
    for nm in names:
        s = resolve(nm)
        if s is None:
            continue
        node = s['nr']
        placed = False
        for k, h in enumerate(chain):
            if node in h['path_nodes'][1:-1]:
                added_by_hop.setdefault(k, []).append(
                    (h['path_nodes'].index(node), nm, s))
                placed = True
                break
        if not placed:
            print(f"  [apply]   STP add {rid}: '{nm}' not traversed by dir {dir_id} "
                  f"var {var} — not a stop-pattern change, skipped")

    new_seq: List[Dict] = []
    synth_tt: Dict[Tuple[int, int], Dict] = {}
    cache: Dict[Tuple, Dict] = {}
    for k, h in enumerate(chain):
        a = _stp_stop_dict(h['from_name'], h['from_nr'], resolve)
        b = _stp_stop_dict(h['to_name'], h['to_nr'], resolve)
        if not new_seq:
            new_seq.append(a)
        adds = sorted(added_by_hop.get(k, []))
        if not adds:                                # unchanged hop — base reuse
            synth_tt[(a['nr'], b['nr'])] = {'TT': h['TT'],
                                            'IVWT': float(h['IVWT'] or 0.0)}
            new_seq.append(b)
            continue
        # ordered stops across this host hop: from + added(by path order) + to
        mids = [s for (_idx, _nm, s) in adds]
        seq_pts = [a] + mids + [b]
        line = _force_linestring(h['geometry'])
        total_len = float(h.get('path_length_m') or (line.length if line else 0.0))
        pn = h['path_nodes']
        node_seq = [a['nr']] + [s['nr'] for s in mids] + [b['nr']]
        node_idx = [pn.index(n) if n in pn else None for n in node_seq]
        for si in range(len(seq_pts) - 1):
            sa, sb = seq_pts[si], seq_pts[si + 1]
            # IVWT = dwell at the sub-hop's from-stop: the first sub-hop keeps the
            # host hop's from-stop dwell; each sub-hop leaving an ADDED stop carries
            # the default dwell.
            ivwt = float(h['IVWT'] or 0.0) if si == 0 else default_ivwt
            i0, i1 = node_idx[si], node_idx[si + 1]
            sub_pn = (pn[i0:i1 + 1] if i0 is not None and i1 is not None and i0 <= i1
                      else [sa['nr'], sb['nr']])
            bor = borrow.get((sa['name'], sb['name']))
            borrowed = bor is not None and pd.notna(bor['TT'])
            # Geometry comes from the host-hop substring (and sub_pn from the host path
            # slice above). The LENGTH is a proportional slice of the host's routed
            # path_length_m (total_len), NOT the raw substring geometry length: adding a
            # stop does not change the physical route, so the split pieces must sum back to
            # the host length (mirrors the drop merge, which sums path_length_m). Using
            # geom.length undercounted train-km -> a spurious negative operating cost in 8B,
            # and broke TT conservation across the split.
            geom = _stp_substring(line, seq_pts, si)
            frac = (geom.length / line.length
                    if geom is not None and line is not None and line.length > 0
                    else 1.0 / (len(seq_pts) - 1))
            length = total_len * frac
            if borrowed:                                    # real stopping time — TT only
                tt = float(bor['TT'])
            else:
                base_tt = float(h['TT'] or 0.0)
                tt = base_tt * (length / total_len) if total_len > 0 else base_tt
                # each added stop adds the modelled run-time (decel into it)
                if si < len(mids):
                    tt += penalty
            synth_tt[(sa['nr'], sb['nr'])] = {'TT': round(tt, 3), 'IVWT': ivwt}
            key = ssp.base_reuse_key(rid, dir_id, var, sa['name'], sb['name'])
            method = 'stp_borrow' if borrowed else 'stp_split'
            cache[key] = _stp_cache_payload(
                geom, sub_pn, length, sa['nr'], sb['nr'], '', '', method)
            new_seq.append(sb)
    return new_seq, synth_tt, cache


def _stp_substring(line, seq_pts, si):
    """Substring of the host hop geometry between consecutive seq points."""
    if line is None:
        return None
    da = line.project(Point(seq_pts[si]['E'], seq_pts[si]['N']))
    db = line.project(Point(seq_pts[si + 1]['E'], seq_pts[si + 1]['N']))
    if db < da:
        da, db = db, da
    try:
        return substring(line, da, db)
    except Exception:
        return line


def _safe_union(geoms):
    """linemerge a list of (Multi)LineStrings into a single geometry."""
    parts = []
    for g in geoms:
        if g is None:
            continue
        if g.geom_type == 'LineString':
            parts.append(g)
        elif hasattr(g, 'geoms'):
            parts.extend([p for p in g.geoms if p.geom_type == 'LineString'])
    if not parts:
        return None
    return linemerge(parts) if len(parts) > 1 else parts[0]


def _route_variants(base_lines, route_id) -> List[int]:
    """All dir-0 variant_ranks of a route_id across the base line layers."""
    out: set = set()
    for ldf in base_lines.values():
        m = ((ldf['route_id'].astype(str) == str(route_id)) &
             (ldf['direction_id'].astype(str) == '0'))
        out.update(int(v) for v in ldf.loc[m, 'variant_rank'].tolist())
    return sorted(out)


def _emit_line(delta_lines, delta_segs, layer, meta, dir_id, seq, total_dep, base_tt):
    """Append one direction's line row + its stop-pair segment rows to the delta.

    Default (EXT/NDC): per-period columns carry the flat whole-day rate and
    service_period 'all_day' (homogeneous all-day services — F8 closure). FRQ
    passes meta['period_rates'] / meta['service_period'] so period-split variants
    keep their real scaled rates and canonical period label.
    """
    freq_hr = round(total_dep / (getattr(settings, 'GK_WINDOW_MIN', 840) / 60.0), 3)
    rate_am, rate_pm, rate_off = meta.get('period_rates') or (freq_hr, freq_hr, freq_hr)
    period = meta.get('service_period') or 'all_day'
    line_geom = LineString([(s['E'], s['N']) for s in seq])
    delta_lines.setdefault(layer, []).append({
        'route_id': meta['route_id'], 'direction_id': dir_id,
        'variant_rank': meta['variant_rank'], 'variant_trip_share': 1.0,
        'line_short_name': meta['short'], 'origin': seq[0]['name'],
        'destination': seq[-1]['name'],
        'line_long_name': f"{meta['short']}: {seq[0]['name']} - {seq[-1]['name']}",
        'line_type': meta['line_type'], 'mode_label': meta['mode_label'],
        'mode_class': 'rail', 'agency_id': meta.get('agency_id', ''),
        'is_circular': seq[0]['nr'] == seq[-1]['nr'],
        'n_stops': len(seq), 'service_period': period,
        'freq_am_peak_dep_hr': rate_am, 'freq_pm_peak_dep_hr': rate_pm,
        'freq_offpeak_dep_hr': rate_off, 'total_dep': total_dep,
        'freq_directional': False, 'tt_source': 'projected', 'geometry': line_geom,
    })
    rows = delta_segs.setdefault(layer, [])
    default_ivwt = float(getattr(settings, 'SVC_INT_DEFAULT_IVWT_MIN', 0.5))
    for idx, (a, b) in enumerate(zip(seq[:-1], seq[1:])):
        prev = base_tt.get((a['nr'], b['nr']))
        keep_tt = prev is not None and pd.notna(prev.get('TT'))
        # IVWT = dwell at the from-stop (base GTFS convention): base-reused hops keep
        # their base dwell; added hops get the standard default, except a direction's
        # first hop (the train originates there, dwell 0).
        if keep_tt:
            ivwt = float(prev['IVWT']) if pd.notna(prev.get('IVWT')) else 0.0
        else:
            ivwt = 0.0 if idx == 0 else default_ivwt
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
            'IVWT': ivwt,
            'service_period': period,
            'freq_am_peak_dep_hr': rate_am, 'freq_pm_peak_dep_hr': rate_pm,
            'freq_offpeak_dep_hr': rate_off,
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
            # absolute total_dep, or ('factor', f) — the caller scales each
            # variant's OWN base total_dep, so period-split variants stay split
            if 'total_dep' in p:
                total_dep_override = int(p['total_dep'])
            else:
                total_dep_override = ('factor', float(p['factor']))
        elif kind == 'add_stop':
            # STP: insert each named intermediate stop at its geographic position
            # (the existing consecutive pair it lies between — min added detour).
            # Direction-safe: operates on the current sequence's coordinates, so
            # dir-0/dir-1 each place the stop in their own order (no stored offset
            # → the FRQ multi-stop reversal gotcha cannot occur). The STP delta
            # branch re-derives placement from path_nodes for the geometry/TT; this
            # is the sequence-level primitive.
            for name in p.get('stops', []):
                s = resolve(name)
                if s is None or any(x['nr'] == s['nr'] for x in seq):
                    continue
                best_i, best_cost = None, None
                for i in range(len(seq) - 1):
                    a, b = seq[i], seq[i + 1]
                    detour = (math.hypot(a['E'] - s['E'], a['N'] - s['N'])
                              + math.hypot(s['E'] - b['E'], s['N'] - b['N'])
                              - math.hypot(a['E'] - b['E'], a['N'] - b['N']))
                    if best_cost is None or detour < best_cost:
                        best_cost, best_i = detour, i + 1
                if best_i is not None:
                    seq = seq[:best_i] + [s] + seq[best_i:]
        elif kind == 'drop_stop':
            # STP: remove each named intermediate call (never a terminus).
            drop = set(p.get('stops', []))
            seq = [s for i, s in enumerate(seq)
                   if not (s['name'] in drop and 0 < i < len(seq) - 1)]
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

_BASE_PROJ_REUSE: Dict[Tuple[str, str], Dict] = {}


def _build_base_projection_cache(base_svc_version, base_infra_version) -> Dict:
    """base_reuse_key → base-projected enrichment payload, for unchanged stop-pairs.

    Passed to services_service_projection.project_lines so a delta's unchanged
    stop-pairs (tt_source='gtfs') keep the base-projected path instead of being
    re-routed — equal-weight Dijkstra ties must never flip an unchanged hop between
    per-svc-int networks (determinism fix, 2026-06-11). Memoised per (svc, infra).
    """
    import services_service_projection as ssp

    mkey = (str(base_svc_version), str(base_infra_version))
    if mkey in _BASE_PROJ_REUSE:
        return _BASE_PROJ_REUSE[mkey]
    cache: Dict = {}
    _BASE_PROJ_REUSE[mkey] = cache
    proj = paths.get_projected_services_path(base_svc_version, base_infra_version)
    if not Path(proj).exists():
        print(f"  [apply] base projected services missing at {proj} — "
              f"unchanged-pair reuse disabled")
        return cache

    def _flag(v) -> bool:
        return bool(v) if pd.notna(v) else False

    for layer in fiona.listlayers(proj):
        g = gpd.read_file(proj, layer=layer)
        if g.empty:
            continue
        for _, r in g.iterrows():
            key = ssp.base_reuse_key(
                r.get('GTFS_ID', ''), r.get('direction_id', ''),
                r.get('variant_rank', ''),
                r.get('from_stop_name', ''), r.get('to_stop_name', ''))
            if key in cache:
                continue
            cache[key] = {
                'node_id_from': r.get('node_id_from'), 'node_id_to': r.get('node_id_to'),
                'match_method_from': r.get('match_method_from', 'base_reuse'),
                'match_method_to': r.get('match_method_to', 'base_reuse'),
                'from_code': r.get('from_code', ''), 'to_code': r.get('to_code', ''),
                'Via_Nodes': r.get('Via_Nodes', ''), 'Via_Segment': r.get('Via_Segment', ''),
                'Via_Station': '', 'Via_Junction': '',
                'path_nodes': r.get('path_nodes', ''),
                'path_length_m': r.get('path_length_m'),
                '_path_tt_min': 0.0,
                'needs_correction': _flag(r.get('needs_correction')),
                'elec_mismatch': _flag(r.get('elec_mismatch')),
                'geometry': r.geometry,
                'FromCode': r.get('from_stop_nr'), 'ToCode': r.get('to_stop_nr'),
            }
    print(f"  [apply] base-projection reuse cache: {len(cache)} stop-pair row(s)")
    return cache


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


def svc_int_line_name(svc_int, base_short: Optional[str] = None) -> str:
    """Readable line short name for a svc-int (binding naming scheme 2026-06-10).

    Suffix = type abbreviation + the int's global per-type sequence number
    (1:1 with the registry id: n = id number − DEV_ID_START_<TYPE>). New lines
    → 'NDC2'; modified lines keep their base short name → 'S14_EXT5'.

    Args:
        svc_int: record dict (int_id, int_type).
        base_short: the modified line's existing short name (None for new lines).
    """
    int_id = str(svc_int.get('int_id', ''))
    int_type = str(svc_int.get('int_type', '') or int_id.split('_')[0]).lower()
    try:
        num = int(int_id.split('_')[1])
    except (IndexError, ValueError):
        return base_short or int_id
    start = int(getattr(settings, _TYPE_START.get(int_type, ''), num))
    code = f"{int_type.upper()}{num - start}"
    return f"{base_short}_{code}" if base_short else code


def svc_int_id_short(svc_int) -> str:
    """Legacy fallback label for records predating line_short_name in the registry."""
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
    print(f"\n=== [5B] Service Interventions (mode={mode}) ===")
    print(f"  base infra: {base_infra} | services: {base_svc} | combo: {combo} | active: {active or 'none'}")

    result: Dict = {'ext_ids': [], 'ndc_ids': [], 'frq_ids': [], 'stp_ids': [],
                    'materialised': [], 'plots': []}

    # Auto-clear inactive types: downstream phases read ALL registry types, so a
    # type generated by a previous run (e.g. NDC under SVC_INT_MODE='ALL') would
    # leak into a later subset run. Drop the registry rows + per-id outputs of
    # every supported type NOT selected this run so disk mirrors the live set.
    for _t in SUPPORTED_SVC_INT_TYPES:
        if _t in active:
            continue
        _stale = list_svc_int_ids(_t, network=combo)
        if _stale:
            print(f"  [{_t}] clearing {len(_stale)} stale record(s) — type not "
                  f"selected this run")
            delete_records(_t, _stale, network=combo)
            _purge_svc_int_outputs(_stale, combo)

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
            _old_ext = list_svc_int_ids('ext', network=combo)
            delete_records('ext', _old_ext, network=combo)
            _purge_svc_int_outputs(_old_ext, combo)
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
            _old_ndc = list_svc_int_ids('ndc', network=combo)
            delete_records('ndc', _old_ndc, network=combo)
            _purge_svc_int_outputs(_old_ndc, combo)
            import svc_ints_new_direct_connections as ndc
            disc = ndc.discover_and_register(
                base_infra, base_svc, ndc_candidates, sa_polygon=sa_polygon, network=combo)
            result['ndc_ids'] = disc['ndc_ids']

    # FRQ — corridor homogenisation + doubling (after EXT: twin detection reads
    # the freshly registered ext catalogue) ------------------------------------
    frq_disc: Optional[Dict] = None
    if 'frq' in active:
        if manifest_ok and list_svc_int_ids('frq', network=combo):
            result['frq_ids'] = list_svc_int_ids('frq', network=combo)
            print(f"  [frq] use_cache: keeping {len(result['frq_ids'])} existing FRQ record(s)")
        else:
            _old_frq = list_svc_int_ids('frq', network=combo)
            delete_records('frq', _old_frq, network=combo)
            _purge_svc_int_outputs(_old_frq, combo)
            import svc_ints_frequency as frqmod
            frq_disc = frqmod.discover_and_register(
                base_infra, base_svc, sa_polygon, buffer_polygon, network=combo)
            result['frq_ids'] = frq_disc['frq_ids']

    # STP — stopping-pattern changes (passing-service homogenisation + express) --
    stp_disc: Optional[Dict] = None
    if 'stp' in active:
        if manifest_ok and list_svc_int_ids('stp', network=combo):
            result['stp_ids'] = list_svc_int_ids('stp', network=combo)
            print(f"  [stp] use_cache: keeping {len(result['stp_ids'])} existing STP record(s)")
        else:
            _old_stp = list_svc_int_ids('stp', network=combo)
            delete_records('stp', _old_stp, network=combo)
            _purge_svc_int_outputs(_old_stp, combo)
            import svc_ints_stop_patterns as stpmod
            stp_disc = stpmod.discover_and_register(
                base_infra, base_svc, sa_polygon, buffer_polygon, network=combo)
            result['stp_ids'] = stp_disc['stp_ids']

    # Materialise each registered svc-int (delta network, real infra TT) -------
    todo = ([('ext', i) for i in result['ext_ids']] +
            [('ndc', i) for i in result['ndc_ids']] +
            [('frq', i) for i in result['frq_ids']] +
            [('stp', i) for i in result['stp_ids']])
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
            if frq_disc and frq_disc.get('corridors'):
                import svc_ints_frequency as frqmod
                p = frqmod.plot_frq_candidates(
                    base_infra, base_svc, frq_disc['corridors'],
                    frq_disc['corridor_candidates'], sa_polygon)
                if p:
                    result['plots'].append(p)
            if stp_disc and (stp_disc.get('mode_a') or stp_disc.get('mode_b')):
                import svc_ints_stop_patterns as stpmod
                p = stpmod.plot_stp_candidates(base_infra, base_svc, stp_disc, sa_polygon)
                if p:
                    result['plots'].append(p)
            result['plots'] += plot_svc_interventions(base_infra, base_svc, result, sa_polygon)
        except Exception as exc:
            print(f"  [plot] WARNING: svc-int plots failed: {exc}")

    print(f"=== [5B] done: {len(result['ext_ids'])} EXT, "
          f"{len(result['ndc_ids'])} NDC, {len(result['frq_ids'])} FRQ, "
          f"{len(result['stp_ids'])} STP, "
          f"{len(result['materialised'])} materialised ===\n")
    return result


def _active_svc_int_types(mode) -> List[str]:
    """Map SVC_INT_MODE to the svc-int registry types to generate.

    Accepts the legacy string forms ('NONE' | 'ALL' | a single type) or an
    explicit list/tuple of type codes (e.g. ['EXT', 'NDC']) — list entries are
    lowercased, validated and returned in canonical SUPPORTED_SVC_INT_TYPES order.
    """
    if isinstance(mode, (list, tuple, set)):
        import ints_core as core
        return core.normalise_int_types(mode, SUPPORTED_SVC_INT_TYPES, 'SVC_INT_MODE')
    m = str(mode).upper()
    if m == 'NONE':
        return []
    if m == 'ALL':
        return list(SUPPORTED_SVC_INT_TYPES)
    if m in ('EXT', 'NDC', 'FRQ', 'STP'):
        return [m.lower()]
    print(f"  [svc-int] unknown SVC_INT_MODE='{mode}' — treating as 'NONE'")
    return []


def svc_int_active(mode=None) -> bool:
    """True when SVC_INT_MODE resolves to at least one svc-int type.

    Replaces the scattered ``str(SVC_INT_MODE).upper() == 'NONE'`` gates, which
    misread a list value (and an empty list) as active.
    """
    if mode is None:
        mode = getattr(settings, 'SVC_INT_MODE', 'NONE')
    return bool(_active_svc_int_types(mode))


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
    for int_type in SUPPORTED_SVC_INT_TYPES:
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
    key): for an EXT/FRQ, every (direction_id, variant_rank) of the route in the
    base network; for an NDC, the new line's synthetic keys (int_id, both
    directions, its variant_rank); for an STP, the union over its modified route(s)
    — affected_services, which is the route list (the combined homogenisation int
    modifies three routes). stations → id_point (== from_stop_nr), resolved from
    the stored station names. Names that don't resolve are dropped (no crash)."""
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
        elif itype == 'stp':
            routes = [str(r) for r in (rec.get('affected_services') or [rid]) if r]
            vks = sorted(f"{r}_{d}_{v}"
                         for r in routes
                         for (d, v) in route_variants.get(r, set()))
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
    recs = [r for it in SUPPORTED_SVC_INT_TYPES for r in read_records(it, network=network)]
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
_FRQ_COLOR     = '#7b2d8e'   # violet: FRQ (corridor extension hop / doubled line)
_FRQ_COLOR_OLD = '#cfa8dc'   # light violet: the pre-existing run of the line
_STP_COLOR     = '#0f7b6c'   # teal: STP (the re-stopped / re-routed-pattern line)
_STP_COLOR_OLD = '#7fc8bd'   # light teal: the pre-existing run of the line
_TYPE_COLOR     = {'ext': _EXT_COLOR, 'ndc': _NDC_COLOR, 'frq': _FRQ_COLOR, 'stp': _STP_COLOR}
_TYPE_COLOR_OLD = {'ext': _EXT_COLOR_OLD, 'ndc': _NDC_COLOR_OLD, 'frq': _FRQ_COLOR_OLD,
                   'stp': _STP_COLOR_OLD}
_BACKDROP      = '#d4d4d4'   # unused infrastructure
_LAKE_FC       = '#c8e8f5'
_LAKE_EC       = '#99c4d8'
_ENDNODE_FC    = '#f6a21e'   # orange terminus / branch marker (candidate plots)
_OFFSET_M      = 160.0       # perpendicular spacing between interventions sharing a track (single-int legacy path)

# Bundle / candidate / overview rendering (the per-int colour + dynamic-offset family).
_SVC_SP    = 110.0   # perpendicular spacing between lines sharing one infra hop
_SVC_LW    = 2.2     # uniform service-line width
_ST_ALONG  = 70.0    # stadium half-thickness along the track (the short axis / cap radius)
_ST_MARGIN = 45.0    # extra stadium half-width beyond the bundle offset (the long axis)


def _svc_overview_colors(ids: List[str]) -> Dict[str, tuple]:
    """Stable per-int colour map (tab20) shared by the candidate + overview plots."""
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap('tab20', max(len(ids), 2))
    return {i: cmap(k % cmap.N) for k, i in enumerate(ids)}


def _fade(color, f: float = 0.55):
    """Lighten a colour toward white by fraction f (the faded 'existing line' shade)."""
    import matplotlib.colors as mcolors
    r, g, b, _ = mcolors.to_rgba(color)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f, 1.0)


def _alt_mult(rank: int) -> float:
    """Alternating-outward slot multiplier: 0→0 (centre), 1→+1, 2→-1, 3→+2, 4→-2 …"""
    if rank == 0:
        return 0.0
    k = (rank + 1) // 2
    return float(k if rank % 2 == 1 else -k)


def _hop_membership(infos: List[Dict], new_only: bool = False) -> Dict[frozenset, List[str]]:
    """hop (frozenset of the two stop names) → ordered list of int ids traversing it."""
    mem: Dict[frozenset, List[str]] = {}
    for info in infos:
        seg = info['seg']
        seg = seg[~seg['_old']] if new_only else seg
        for h in seg['_hop'].dropna().unique():
            mem.setdefault(h, [])
            if info['id'] not in mem[h]:
                mem[h].append(info['id'])
    return mem


def _draw_stadiums(ax, infos: List[Dict], hopmem: Dict[frozenset, List[str]], ctx: Dict) -> None:
    """Capacity-plot-style station stadiums: oriented perpendicular to the local track
    (doubled-angle mean) and sized to span only the bundle offset, so they connect the
    parallel lines without painting over hops that traverse a junction."""
    import math
    from shapely.geometry import LineString
    sxy: Dict[str, tuple] = {}
    dir2: Dict[str, List[float]] = {}
    maxoff: Dict[str, float] = {}
    for info in infos:
        for _, r in info['seg'].iterrows():
            for nk, ek, nn in (('from_stop_name', 'from_stop_E', 'from_stop_N'),
                               ('to_stop_name', 'to_stop_E', 'to_stop_N')):
                nm = str(r.get(nk, '') or '')
                if nm and nm not in sxy and pd.notna(r.get(ek)) and pd.notna(r.get(nn)):
                    sxy[nm] = (float(r[ek]), float(r[nn])); dir2[nm] = [0.0, 0.0]; maxoff[nm] = 0.0
    def _dir_at(geom, x, y):
        """Tangent angle of ``geom`` at the geometry endpoint nearest (x, y).

        Handles MultiLineStrings (the part actually touching the station) so the
        stadium orientation follows the SERVICE direction at the stop — not the first
        MLS part, which mis-oriented termini like Kempten / Kemptthal.
        """
        parts = list(geom.geoms) if geom.geom_type == 'MultiLineString' else [geom]
        best = None
        for ls in parts:
            cs = list(ls.coords)
            if len(cs) < 2:
                continue
            for idx, nb in ((0, 1), (-1, -2)):
                px, py = cs[idx]; qx, qy = cs[nb]
                d = (px - x) ** 2 + (py - y) ** 2
                if best is None or d < best[0]:
                    best = (d, math.atan2(qy - py, qx - px))
        return best[1] if best else None

    for info in infos:
        for _, r in info['seg'].iterrows():
            g = r.geometry
            if g is None or g.is_empty:
                continue
            off = (len(hopmem.get(r['_hop'], [1])) - 1) / 2.0 * _SVC_SP
            for nm in (str(r.get('from_stop_name', '') or ''), str(r.get('to_stop_name', '') or '')):
                if nm not in dir2:
                    continue
                th = _dir_at(g, sxy[nm][0], sxy[nm][1])
                if th is None:
                    continue
                dir2[nm][0] += math.sin(2 * th); dir2[nm][1] += math.cos(2 * th)
                maxoff[nm] = max(maxoff[nm], off)
    caps = []
    for nm, (x, y) in sxy.items():
        s2, c2 = dir2[nm]
        orient = 0.5 * math.atan2(s2, c2) if (abs(s2) > 1e-9 or abs(c2) > 1e-9) else 0.0
        vx, vy = -math.sin(orient), math.cos(orient)
        half = max(maxoff[nm] + _ST_MARGIN - _ST_ALONG, 0.0)
        caps.append(LineString([(x - vx * half, y - vy * half),
                                (x + vx * half, y + vy * half)]).buffer(_ST_ALONG, cap_style=1))
    if caps:
        gpd.GeoSeries(caps, crs=SWISS_CRS).plot(ax=ax, facecolor='white', edgecolor='black',
                                                linewidth=0.8, zorder=5)
        for nm, (x, y) in sxy.items():
            code = (ctx.get('node_xy', {}).get(nm, ('',)) or ('',))[0]
            if code:
                ax.annotate(code, xy=(x, y), xytext=(6, 6), textcoords='offset points', fontsize=5.5,
                            fontweight='bold', color='#333333', zorder=7,
                            bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.6))



def _svc_backdrop(ax, ctx) -> None:
    """Lakes + grey infra + dashed SA boundary + extent (shared map base)."""
    if ctx['lakes'] is not None and not ctx['lakes'].empty:
        ctx['lakes'].plot(ax=ax, color=_LAKE_FC, edgecolor=_LAKE_EC, linewidth=0.3, zorder=0)
    if ctx['backdrop'] is not None and not ctx['backdrop'].empty:
        ctx['backdrop'].plot(ax=ax, color=_BACKDROP, linewidth=0.5, zorder=1)
    if ctx['sa_gdf'] is not None:
        ctx['sa_gdf'].boundary.plot(ax=ax, color='black', linewidth=0.8, linestyle='--',
                                    alpha=0.6, zorder=2)
    if ctx['extent'] is not None:
        ax.set_xlim(ctx['extent'][0], ctx['extent'][1]); ax.set_ylim(ctx['extent'][2], ctx['extent'][3])


def _classify_seg(seg, ctx):
    """Tag delta segments with their hop (frozenset of stop names) + old/new flag.

    A hop is 'old' (existing line) when it already exists on the BASE network for the
    SAME GTFS_ID; everything else is 'new' (the delta). NDC = all-new (no base GTFS_ID).
    """
    seg = seg.copy()
    if {'from_stop_name', 'to_stop_name'} <= set(seg.columns):
        hops = seg.apply(lambda r: frozenset((str(r['from_stop_name']), str(r['to_stop_name']))), axis=1)
        seg['_hop'] = hops
        bp: set = set()
        if 'GTFS_ID' in seg.columns:
            for g in seg['GTFS_ID'].dropna().astype(str).unique():
                bp |= ctx['base_pairs'].get(g, set())
        seg['_old'] = hops.isin(bp) if bp else False
    else:
        seg['_hop'] = None; seg['_old'] = False
    return seg


def _load_svc_int_infos(int_type: str, base_infra: str, base_svc: str, ctx: Dict) -> List[Dict]:
    """Load every registered svc-int of a type as a render-ready info (classified delta)."""
    combo = f"{base_infra}__{base_svc}"
    infos: List[Dict] = []
    for rec in read_records(int_type, network=combo):
        iid = rec['int_id']
        pp = Path(paths.get_svc_int_network_dir(iid, combo)) / base_infra / 'rail_segments.gpkg'
        seg = _load_delta_segments(pp)
        if seg is None or seg.empty:
            continue
        aff = rec.get('affected_stations') or []
        ops = rec.get('operations')
        ops = ops if isinstance(ops, list) else deserialize_ops(ops)
        seg = _classify_seg(seg, ctx)
        # FRQ doubling (a set_frequency factor op) changes the whole line, not its geometry —
        # paint the entire line as changed rather than faded-existing.
        if any(o.get('op') == 'set_frequency' for o in ops):
            seg['_old'] = False
        added: set = set()
        dropped: set = set()
        for o in ops:
            if o.get('op') == 'add_stop':
                added |= {str(s) for s in (o.get('params', {}).get('stops') or [])}
            elif o.get('op') == 'drop_stop':
                dropped |= {str(s) for s in (o.get('params', {}).get('stops') or [])}
        infos.append({'id': iid, 'type': int_type,
                      'short': str(rec.get('line_short_name') or iid),
                      'endpoint': str(aff[0]) if aff else '', 'target': str(aff[-1]) if aff else '',
                      'added_stops': added, 'dropped_stops': dropped,
                      'seg': seg})
    return infos


def _render_overview(infos: List[Dict], ctx: Dict, out_path, title: str,
                     new_label: str) -> Optional[str]:
    """All-of-a-type overview: per-int colour, faded existing, per-hop bundle offset
    (centred — tightens as lines branch off), uniform width, perpendicular stadiums."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import infrabuild_network_builder as ic
    if not infos:
        return None
    try:
        fig, ax = plt.subplots(figsize=(11, 13))
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.set_xlabel('E [m]', fontsize=10); ax.set_ylabel('N [m]', fontsize=10)
        _svc_backdrop(ax, ctx)
        colors = _svc_overview_colors([i['id'] for i in infos])
        hopmem = _hop_membership(infos)
        for info in infos:
            col = colors[info['id']]
            for _, row in info['seg'].iterrows():
                members = hopmem.get(row['_hop'], [info['id']]); cnt = len(members)
                dist = (members.index(info['id']) - (cnt - 1) / 2.0) * _SVC_SP \
                    if info['id'] in members else 0.0
                g = _safe_offset(row.geometry, dist)
                c = _fade(col) if row['_old'] else col
                gpd.GeoSeries([g], crs=SWISS_CRS).plot(ax=ax, color=c, linewidth=_SVC_LW,
                                                       zorder=3 if row['_old'] else 4)
        is_stp = any(i['type'] == 'stp' for i in infos)
        if is_stp:
            _draw_stp_markers(ax, infos, ctx)
        else:
            _draw_stadiums(ax, infos, hopmem, ctx)
        if ctx['extent'] is not None:
            ax.set_xlim(ctx['extent'][0], ctx['extent'][1]); ax.set_ylim(ctx['extent'][2], ctx['extent'][3])
        handles = [Line2D([0], [0], color='#888', lw=_SVC_LW, label=new_label),
                   Line2D([0], [0], color='#cfcfcf', lw=_SVC_LW, label='Existing line (faded)'),
                   Line2D([0], [0], color=_BACKDROP, lw=1.5, label='Unused infrastructure')]
        if is_stp:
            handles += [Line2D([0], [0], marker='o', color='w', markerfacecolor=_STP_ADD_FC,
                               markeredgecolor='black', markersize=7, label='Added call'),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor=_STP_DROP_FC,
                               markeredgecolor='black', markersize=7, label='Dropped call')]
        ax.legend(handles=handles, loc='upper right', fontsize=7)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ic._add_north_arrow(ax, location='upper left', scale=0.5)
        ic._add_scale_bar(ax, location=(0.755, 0.012))
        plt.tight_layout(); fig.savefig(out_path, bbox_inches='tight'); plt.close(fig)
        print(f"  [plot] wrote {Path(out_path).name}")
        return str(out_path)
    except Exception as exc:
        print(f"  [plot]   WARNING {Path(out_path).name}: {exc}")
        try:
            plt.close('all')
        except Exception:
            pass
        return None


def _render_candidates(infos: List[Dict], ctx: Dict, out_path, title: str,
                       mark_terminus: bool) -> Optional[str]:
    """Candidate overview: each candidate's NEW (delta) geometry along the real infra,
    per-hop offset (longest on the centre line, alternating outward; alone → on geometry).
    End stations as white circles; the original terminus kept orange (EXT)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import infrabuild_network_builder as ic
    if not infos:
        return None
    try:
        fig, ax = plt.subplots(figsize=(11, 13))
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.set_xlabel('E [m]', fontsize=10); ax.set_ylabel('N [m]', fontsize=10)
        _svc_backdrop(ax, ctx)
        colors = _svc_overview_colors([i['id'] for i in infos])
        order = sorted(infos, key=lambda info: -float(info['seg'][~info['seg']['_old']].geometry.length.sum()))
        rank = {info['id']: r for r, info in enumerate(order)}
        hopmem = _hop_membership(infos, new_only=True)
        nxy = ctx['node_xy']
        handles: List = []
        target_st: Dict = {}
        termini: set = set()
        for k, info in enumerate(infos):
            col = colors[info['id']]
            for _, row in info['seg'][~info['seg']['_old']].iterrows():
                members = sorted(hopmem.get(row['_hop'], [info['id']]), key=lambda e: rank[e])
                dist = _alt_mult(members.index(info['id'])) * _SVC_SP if info['id'] in members else 0.0
                g = _safe_offset(row.geometry, dist)
                gpd.GeoSeries([g], crs=SWISS_CRS).plot(ax=ax, color=col, linewidth=_SVC_LW, alpha=0.95, zorder=4)
            new_ends = (info['target'],) if mark_terminus else (info['endpoint'], info['target'])
            for nm in new_ends:
                if nm in nxy:
                    target_st[nm] = nxy[nm]
            if mark_terminus and info['endpoint'] in nxy:
                termini.add(info['endpoint'])
            handles.append(Line2D([0], [0], color=col, lw=_SVC_LW,
                           label=f"{k+1}: {info['short']}  {info['endpoint']} → {info['target']}"))
        _plot_stations(ax, target_st, set())     # white circle + code at new end stations
        for nm in termini:                        # original terminus kept orange
            p = nxy.get(nm)
            if p:
                ax.plot(p[1], p[2], marker='o', markersize=9, markerfacecolor=_ENDNODE_FC,
                        markeredgecolor='black', markeredgewidth=0.9, zorder=7)
                if p[0]:
                    ax.annotate(p[0], xy=(p[1], p[2]), xytext=(4, 4), textcoords='offset points',
                                fontsize=6, fontweight='bold', color='#333333', zorder=8,
                                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.7))
        if ctx['extent'] is not None:
            ax.set_xlim(ctx['extent'][0], ctx['extent'][1]); ax.set_ylim(ctx['extent'][2], ctx['extent'][3])
        base = [Line2D([0], [0], color=_BACKDROP, lw=1.5, label='Unused infrastructure'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                       markeredgecolor='black', markersize=6, label='End station')]
        if mark_terminus:
            base.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=_ENDNODE_FC,
                               markeredgecolor='black', markersize=8, label='Original terminus'))
        ax.legend(handles=base + handles, loc='upper right', fontsize=6)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ic._add_north_arrow(ax, location='upper left', scale=0.5)
        ic._add_scale_bar(ax, location=(0.755, 0.012))
        plt.tight_layout(); fig.savefig(out_path, bbox_inches='tight'); plt.close(fig)
        print(f"  [plot] wrote {Path(out_path).name}")
        return str(out_path)
    except Exception as exc:
        print(f"  [plot]   WARNING {Path(out_path).name}: {exc}")
        try:
            plt.close('all')
        except Exception:
            pass
        return None


_OVERVIEW_NEW_LABEL = {'ext': 'Extension (new)', 'ndc': 'New direct connection',
                       'frq': 'Frequency change', 'stp': 'Stopping-pattern change'}
_STP_ADD_FC = '#8bd3a0'   # light green: a stop added to a line
_STP_DROP_FC = '#f3a6a6'  # light red: a stop dropped from a line


def _draw_stp_markers(ax, infos: List[Dict], ctx: Dict) -> None:
    """STP marks the changed CALLS, not stations: a small light-green circle per added
    stop and light-red per dropped stop (per line). Drawn instead of the stadiums."""
    xy: Dict[str, tuple] = {}
    for info in infos:
        for _, r in info['seg'].iterrows():
            for nk, ek, nn in (('from_stop_name', 'from_stop_E', 'from_stop_N'),
                               ('to_stop_name', 'to_stop_E', 'to_stop_N')):
                nm = str(r.get(nk, '') or '')
                if nm and nm not in xy and pd.notna(r.get(ek)) and pd.notna(r.get(nn)):
                    xy[nm] = (float(r[ek]), float(r[nn]))
    for info in infos:
        for nm in info.get('added_stops', set()):
            if nm in xy:
                ax.plot(xy[nm][0], xy[nm][1], marker='o', markersize=6, markerfacecolor=_STP_ADD_FC,
                        markeredgecolor='black', markeredgewidth=0.7, zorder=6)
        for nm in info.get('dropped_stops', set()):
            if nm in xy:
                ax.plot(xy[nm][0], xy[nm][1], marker='o', markersize=6, markerfacecolor=_STP_DROP_FC,
                        markeredgecolor='black', markeredgewidth=0.7, zorder=6)


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
    # affected_stations in REGISTRY order ([endpoint, target] for EXT; the changed span's
    # ends for the other types) — m['affected_stations'] is alphabetically sorted, so read
    # the records for the single-int title '<short>: <A> - <B>'.
    recs_by_id = {r['int_id']: r for it in SUPPORTED_SVC_INT_TYPES
                  for r in read_records(it, network=combo)}

    by_type: Dict[str, List] = {t: [] for t in SUPPORTED_SVC_INT_TYPES}
    for m in materialised:
        seg = _load_delta_segments(m.get('projected_path'))
        if seg is None or seg.empty:
            continue
        t = str(m.get('int_type') or '')
        if t not in SUPPORTED_SVC_INT_TYPES:
            prefix = str(m['svc_int_id']).split('_')[0]
            t = prefix if prefix in SUPPORTED_SVC_INT_TYPES else 'ext'
        info = _svc_int_plot_info(m, seg, t, ctx)
        by_type.setdefault(t, []).append(info)
        out_t = core.plot_out_dir(combo, t)
        p = out_t / f"svc_int_{m['svc_int_id']}_{base_infra}.pdf"
        _rec = recs_by_id.get(m['svc_int_id'], {}) or {}
        _aff = _rec.get('affected_stations') or []
        if t == 'stp':
            _ops = _rec.get('operations')
            _ops = _ops if isinstance(_ops, list) else deserialize_ops(_ops)
            _dropped, _added = [], []
            for o in (_ops or []):
                _st = [str(s) for s in (o.get('params', {}).get('stops') or [])]
                if o.get('op') == 'drop_stop':
                    _dropped += _st
                elif o.get('op') == 'add_stop':
                    _added += _st
            info['dropped_stops'] = list(dict.fromkeys(_dropped))
            info['added_stops'] = list(dict.fromkeys(_added))
            _what = []
            if info['dropped_stops']:
                _what.append("Dropped " + ", ".join(info['dropped_stops']))
            if info['added_stops']:
                _what.append("Added " + ", ".join(info['added_stops']))
            _ttl = (f"{info['short']}: " + "; ".join(_what)) if _what else \
                   (f"{info['short']}: {_aff[0]} - {_aff[-1]}" if len(_aff) >= 2
                    else info['short'])
        else:
            _ttl = f"{info['short']}: {_aff[0]} - {_aff[-1]}" if len(_aff) >= 2 \
                else info['short']
        if _render_svc_fig([info], ctx, p, _ttl, offset=False):
            written.append(str(p))

    for t, items in by_type.items():
        if not items:
            continue
        out_t = core.plot_out_dir(combo, t)
        p = out_t / f"svc_int_ALL_{t.upper()}_{base_infra}.pdf"
        t_infos = _load_svc_int_infos(t, base_infra, base_svc, ctx)
        wp = _render_overview(t_infos, ctx, p,
                              f"Overview of all generated {t.upper()}-interventions ({len(t_infos)})",
                              _OVERVIEW_NEW_LABEL.get(t, 'New (delta)'))
        if wp:
            written.append(wp)

    out_all = core.plot_out_dir(combo, None)
    p = out_all / f"svc_int_ALL_PRODUCED_{base_infra}.pdf"
    wp = _render_all_produced(base_infra, base_svc, ctx, p)
    if wp:
        written.append(wp)
    return written


def _render_type_panel(ax, infos: List[Dict], ctx: Dict, color, title: str) -> None:
    """One ALL_PRODUCED panel: a type's lines in its single colour (faded existing),
    bundle offset + stadiums (STP → add/drop call circles), legend listing the lines."""
    from matplotlib.lines import Line2D
    _svc_backdrop(ax, ctx)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    handles: List = []
    if infos:
        hopmem = _hop_membership(infos)
        for k, info in enumerate(infos):
            for _, row in info['seg'].iterrows():
                members = hopmem.get(row['_hop'], [info['id']]); cnt = len(members)
                dist = (members.index(info['id']) - (cnt - 1) / 2.0) * _SVC_SP \
                    if info['id'] in members else 0.0
                g = _safe_offset(row.geometry, dist)
                c = _fade(color) if row['_old'] else color
                gpd.GeoSeries([g], crs=SWISS_CRS).plot(ax=ax, color=c, linewidth=_SVC_LW,
                                                       zorder=3 if row['_old'] else 4)
            handles.append(Line2D([0], [0], color=color, lw=_SVC_LW,
                           label=f"{k+1}: {info['short']}  {info['endpoint']} → {info['target']}"))
        if any(i['type'] == 'stp' for i in infos):
            _draw_stp_markers(ax, infos, ctx)
        else:
            _draw_stadiums(ax, infos, hopmem, ctx)
    if ctx['extent'] is not None:
        ax.set_xlim(ctx['extent'][0], ctx['extent'][1]); ax.set_ylim(ctx['extent'][2], ctx['extent'][3])
    ax.set_title(title, fontsize=11, fontweight='bold', color=color)
    if handles:
        ax.legend(handles=handles, loc='upper right', fontsize=5, framealpha=0.85)


_TYPE_PANEL_NAME = {'ext': 'EXT (line extensions)', 'ndc': 'NDC (new direct connections)',
                    'frq': 'FRQ (frequency changes)', 'stp': 'STP (stopping-pattern changes)'}


def _render_all_produced(base_infra: str, base_svc: str, ctx: Dict, out_path) -> Optional[str]:
    """ALL_PRODUCED as a 2x2 panel — one type each in its distinct colour, each panel
    legending its generated lines; no shared bottom legend."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        fig, axes = plt.subplots(2, 2, figsize=(17, 19))
        for ax, t in zip(axes.ravel(), ('ext', 'ndc', 'frq', 'stp')):
            infos = _load_svc_int_infos(t, base_infra, base_svc, ctx)
            _render_type_panel(ax, infos, ctx, _TYPE_COLOR[t],
                               f"{_TYPE_PANEL_NAME[t]} ({len(infos)})")
        fig.suptitle("All generated service interventions", fontsize=16, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(out_path, bbox_inches='tight'); plt.close(fig)
        print(f"  [plot] wrote {Path(out_path).name}")
        return str(out_path)
    except Exception as exc:
        print(f"  [plot]   WARNING {Path(out_path).name}: {exc}")
        try:
            plt.close('all')
        except Exception:
            pass
        return None


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

def plot_ext_candidates(base_infra: str, base_svc: str, candidates: List[Dict] = None,
                        sa_polygon=None):
    """EXT candidate overview — each candidate's extension along the real infra geometry.

    Reads the materialised deltas (not the discovery dicts) so candidates follow the
    routed track with the per-hop bundle offset; the original terminus is marked orange
    and the new end station as a white circle. ``candidates`` is accepted for call-site
    compatibility but no longer used.
    """
    ctx = _svc_plot_context(base_infra, base_svc, sa_polygon)
    infos = _load_svc_int_infos('ext', base_infra, base_svc, ctx)
    out = Path(paths.get_developments_plot_dir(f"{base_infra}__{base_svc}", 'ext'))
    out.mkdir(parents=True, exist_ok=True)
    return _render_candidates(infos, ctx, out / f"ext_candidates_{base_infra}.pdf",
                              f"Extend-line candidates ({len(infos)})", mark_terminus=True)


def plot_ndc_candidates(base_infra: str, base_svc: str, ndc_candidates: List[Dict] = None,
                        sa_polygon=None):
    """NDC candidate overview — each new through-service along the real infra geometry.

    Reads the materialised deltas; both ends are new stations (white circles). The
    ``ndc_candidates`` arg is accepted for call-site compatibility but no longer used.
    """
    ctx = _svc_plot_context(base_infra, base_svc, sa_polygon)
    infos = _load_svc_int_infos('ndc', base_infra, base_svc, ctx)
    out = Path(paths.get_developments_plot_dir(f"{base_infra}__{base_svc}", 'ndc'))
    out.mkdir(parents=True, exist_ok=True)
    return _render_candidates(infos, ctx, out / f"ndc_candidates_{base_infra}.pdf",
                              f"New-direct-connection candidates ({len(infos)})", mark_terminus=False)


def _svc_int_plot_info(m: Dict, seg, t: str, ctx: Dict) -> Dict:
    """Classify a delta's segments (old vs new), collect stops + termini + short name."""
    # Old (existing) vs new (delta) split: a delta stop-pair is "old" when it already
    # exists on the BASE network for the SAME line. ctx['base_pairs'] is keyed by GTFS_ID,
    # so look it up by the delta's GTFS_ID — NOT changed_route_ids (route_id), which never
    # matches a GTFS_ID and silently left every segment classed "new" (all-dark plots).
    base_pairs: set = set()
    if 'GTFS_ID' in seg.columns:
        for g in seg['GTFS_ID'].dropna().astype(str).unique():
            base_pairs |= ctx['base_pairs'].get(g, set())

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


_STP_DROP_RED = '#d62728'    # per-int plot: a call dropped from the line
_STP_ADD_GREEN = '#2ca02c'   # per-int plot: a call added to the line


def _stp_change_xy(name, info: Dict, ctx: Dict):
    """(x, y) of a changed-call station: an added stop sits in the delta segments,
    a dropped stop is gone from them — fall back to the base node coordinates."""
    st = (info.get('stations') or {}).get(name)
    if st and st[1] is not None and st[2] is not None:
        return (st[1], st[2])
    nx = (ctx.get('node_xy') or {}).get(name)
    if nx and nx[1] is not None and nx[2] is not None:
        return (nx[1], nx[2])
    return None


def _draw_stp_change(ax, name: str, xy, color: str) -> None:
    """Filled colour marker + name label for one changed call (red drop / green add)."""
    ax.plot(xy[0], xy[1], marker='o', markersize=8, markerfacecolor=color,
            markeredgecolor='black', markeredgewidth=1.0, zorder=8)
    ax.annotate(name, xy=xy, xytext=(5, -9), textcoords='offset points',
                fontsize=6.5, fontweight='bold', color=color, zorder=9,
                bbox=dict(boxstyle='round,pad=0.18', fc='white', ec=color, alpha=0.9))


def _focus_extent(infos: List[Dict], focus_xy: List, margin: float = 2500.0,
                  min_span: float = 9000.0):
    """Axis extent (xmin, xmax, ymin, ymax) framing the int geometry + changed calls.

    The shared SA extent crops STP plots whose changed stops sit outside the study
    area; this frames the line's own bounds plus every red/green marker instead.
    """
    xs: List = []
    ys: List = []
    for info in infos:
        for key in ('old', 'new'):
            g = info.get(key)
            if g is not None and not g.empty:
                b = g.total_bounds   # minx, miny, maxx, maxy
                if np.all(np.isfinite(b)):
                    xs += [b[0], b[2]]
                    ys += [b[1], b[3]]
    for (x, y) in focus_xy:
        xs.append(x)
        ys.append(y)
    if not xs:
        return None
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    spanx = max(max(xs) - min(xs), min_span)
    spany = max(max(ys) - min(ys), min_span)
    return (cx - spanx / 2 - margin, cx + spanx / 2 + margin,
            cy - spany / 2 - margin, cy + spany / 2 + margin)


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
            dark = _TYPE_COLOR.get(info['type'], _EXT_COLOR)
            light = _TYPE_COLOR_OLD.get(info['type'], _EXT_COLOR_OLD)
            _plot_segs(ax, info['old'], light, 1.8, dist)
            _plot_segs(ax, info['new'], dark, 2.8, dist)
        for info in infos:
            _plot_stations(ax, info['stations'], drawn_st)
        for info in infos:
            color = _TYPE_COLOR.get(info['type'], _EXT_COLOR)
            _plot_termini_labels(ax, info, color)

        # STP: mark the changed calls — dropped red, added green, with name labels.
        focus_xy: List = []
        for info in infos:
            for nm in info.get('dropped_stops') or []:
                xy = _stp_change_xy(nm, info, ctx)
                if xy:
                    _draw_stp_change(ax, nm, xy, _STP_DROP_RED)
                    focus_xy.append(xy)
            for nm in info.get('added_stops') or []:
                xy = _stp_change_xy(nm, info, ctx)
                if xy:
                    _draw_stp_change(ax, nm, xy, _STP_ADD_GREEN)
                    focus_xy.append(xy)

        # Per-int STP plots frame the int's own geometry + the changed calls (which
        # can sit outside the SA frame); other types keep the shared SA extent.
        is_stp = any(info.get('type') == 'stp' for info in infos)
        focus = _focus_extent(infos, focus_xy) if is_stp else None
        if focus is not None:
            ax.set_xlim(focus[0], focus[1])
            ax.set_ylim(focus[2], focus[3])
        elif ctx['extent'] is not None:
            ax.set_xlim(ctx['extent'][0], ctx['extent'][1])
            ax.set_ylim(ctx['extent'][2], ctx['extent'][3])

        types = {info['type'] for info in infos}
        handles: List = []
        if 'ext' in types:
            handles += [Line2D([0], [0], color=_EXT_COLOR_OLD, lw=2, label='Existing line (EXT)'),
                        Line2D([0], [0], color=_EXT_COLOR, lw=2.5, label='Extension (new)')]
        if 'ndc' in types:
            handles += [Line2D([0], [0], color=_NDC_COLOR, lw=2.5, label='New direct connection')]
        if 'frq' in types:
            handles += [Line2D([0], [0], color=_FRQ_COLOR_OLD, lw=2, label='Existing line (FRQ)'),
                        Line2D([0], [0], color=_FRQ_COLOR, lw=2.5, label='Frequency change')]
        if 'stp' in types:
            handles += [Line2D([0], [0], color=_STP_COLOR_OLD, lw=2, label='Existing line (STP)'),
                        Line2D([0], [0], color=_STP_COLOR, lw=2.5, label='Stopping-pattern change'),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor=_STP_DROP_RED,
                               markeredgecolor='black', markersize=8, label='Dropped call'),
                        Line2D([0], [0], marker='o', color='w', markerfacecolor=_STP_ADD_GREEN,
                               markeredgecolor='black', markersize=8, label='Added call')]
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
    frq_ids: Optional[List[str]] = None,
    stp_ids: Optional[List[str]] = None,
    make_plots: bool = True,
    use_cache: bool = False,
    sa_polygon=None,
) -> Dict:
    """Materialise the given svc-int ids (delta networks) and optionally plot them.

    Standalone-CLI convenience giving the per-type EXT/NDC/FRQ/STP files the same discover →
    materialise → plot flow as the cc CLI, without re-running discovery. Plots need the
    materialised projected deltas, so they are produced together behind one toggle.
    """
    result: Dict = {'ext_ids': list(ext_ids or []), 'ndc_ids': list(ndc_ids or []),
                    'frq_ids': list(frq_ids or []), 'stp_ids': list(stp_ids or []),
                    'materialised': [], 'plots': []}
    todo = ([('ext', i) for i in result['ext_ids']] +
            [('ndc', i) for i in result['ndc_ids']] +
            [('frq', i) for i in result['frq_ids']] +
            [('stp', i) for i in result['stp_ids']])
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
