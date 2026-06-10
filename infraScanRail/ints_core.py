"""
ints_core — Shared registry I/O + compose/supersede engine for all intervention families.
Last modified: 2026-06-08

Type-agnostic substrate imported by every intervention orchestrator (extracted from
infra_ints_orchestrator on 2026-06-08):

  • Registry I/O + schema       (connecting-curve 'cc' / capacity 'cap' gpkg registries)
  • Compose / supersede engine  (base − superseded + activated → derived/master networks)
  • Shared config + plot primitives (version/polygon resolvers, per-int diff renderer)

The three orchestrators — infra_ints_orchestrator (CC, 5A), svc_ints_orchestrator
(EXT/NDC, 5B) and mixed_ints_orchestrator (CAP, 5C) — are siblings over THIS module;
none imports another for plumbing.

Registry (base-agnostic source of truth)
----------------------------------------
One logical record per intervention ('cc' / 'cap'), identified by ``int_id`` and stored
across three GeoPackage layers (``nodes`` / ``segments`` / ``segments_composition``). A
record stores *intent*, keyed on stable identities in the chosen base — not pre-baked
geometry — so the same intervention re-materialises across base versions. The supersede
triple is:

  removes_base_rows : base Segment_IDs this intervention supersedes
  added_nodes       : new/replacement nodes; junctions carry ``on_segment`` +
                      ``split_position_frac`` (fraction along the host segment)
  adds_rows         : new/replacement segments

Compose (base − superseded + activated)
---------------------------------------
``compose_infra(base, int_ids)`` materialises a derived version, cached under the shared
``Developments/Derived/<base>__<hash8>`` with a ``manifest.json`` recording the
exact int-id set. ``build_master_network(base)`` composes *every* registered int into one
tagged ``Dev_Full`` reference (visualisation / sanity only). Composition length rule:
a new segment's manually-authored pieces are reconciled to the segment length on
``normal`` pieces only — ``bridge``/``tunnel`` keep their measured length.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
import fiona

import paths
import settings
import infrabuild_network_builder as ic

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SWISS_CRS = ic.SWISS_CRS

LAYER_NODES = 'nodes'
LAYER_SEGMENTS = 'segments'
LAYER_COMPOSITION = 'segments_composition'
LAYERS = (LAYER_NODES, LAYER_SEGMENTS, LAYER_COMPOSITION)

# Every registry type the compose engine knows how to union for the master network.
SUPPORTED_INT_TYPES: Tuple[str, ...] = ('cc', 'cap')

# Per-intervention metadata, denormalised onto every spatial row of the int.
META_COLS: Tuple[str, ...] = (
    'int_id', 'int_type', 'int_subtype', 'base_authored',
    'removes_base_rows', 'requires', 'conflicts_with',
    'construction_cost_chf', 'maintenance_cost_annual_chf',
    'length_m', 'design_speed_kmh',
)

# Extra columns on the nodes layer for junctions that split a host segment.
# on_segment empty + split_position_frac NaN  → upsert-by-identity (modify in place).
NODE_SPLIT_COLS: Tuple[str, ...] = ('on_segment', 'split_position_frac')

# List-valued metadata fields stored as comma-joined strings.
_LIST_FIELDS: Tuple[str, ...] = ('removes_base_rows', 'requires', 'conflicts_with')

# Subtype short codes embedded in the int_id (cc_, cap_st_, cap_et_, cap_ps_).
_SUBTYPE_CODE: Dict[str, str] = {
    'connecting_curve': 'cc',
    'station_track': 'cap_st',
    'segment_track': 'cap_et',
    'siding_track': 'cap_ps',
}

_TAG_COLS = ('int_id', 'int_type')
_COMP_LEN_TOL_M = 1.0   # reconcile composition only when the gap exceeds this


# ═════════════════════════════════════════════════════════════════════════════
# REGISTRY — list (de)serialisation
# ═════════════════════════════════════════════════════════════════════════════

def serialize_list(values: Optional[List]) -> str:
    """Comma-join a list of ids into the stored string form ('' for empty/None)."""
    if values is None:
        return ''
    return ','.join(str(v).strip() for v in values if str(v).strip())


def deserialize_list(raw) -> List[str]:
    """Split a stored comma-joined string back into a list of ids ([] for empty)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    s = str(raw).strip()
    if not s:
        return []
    return [tok.strip() for tok in s.split(',') if tok.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY — read
# ─────────────────────────────────────────────────────────────────────────────

def _combo(base_infra: Optional[str] = None, svc_version: Optional[str] = None) -> str:
    """The single '<infra>__<svc>' registry partition key (decision H).

    Each half defaults from the configured run versions, so standalone / incidental reads
    land in the right Developments/<combo>/ workspace without threading both versions.
    """
    bi = base_infra or _resolve_base_version()
    sv = svc_version or _resolve_svc_version()
    return f"{bi}__{sv}"


def _default_network(int_type: str) -> str:
    """Resolve the registry partition key from the configured run versions.

    Used when a caller does not pass an explicit ``network`` (standalone / incidental
    reads). All registries now partition on the '<infra>__<svc>' combo (decision H).
    """
    return _combo()


def default_combo(base_infra: Optional[str] = None, svc_version: Optional[str] = None) -> str:
    """Default '<infra>__<svc>' combo on the PROPAGATED (enhanced) base.

    The pipeline runs Phase 5+ on the enhanced network, so standalone / incidental
    per-svc-int reads must default to the same combo (decision H) — unlike
    ``_combo``, whose plain-base default suits registry creation before
    propagation. Either half can be pinned explicitly.
    """
    return _combo(base_infra or _resolve_base_version_propagated(), svc_version)


def read_registry(
    int_type: str,
    registry_path: Optional[str] = None,
    network: Optional[str] = None,
) -> Tuple[Optional[gpd.GeoDataFrame],
           Optional[gpd.GeoDataFrame],
           Optional[gpd.GeoDataFrame]]:
    """Return (nodes, segments, composition) for a registry, or Nones if absent."""
    gpkg = Path(registry_path or paths.get_infra_int_registry(int_type, network or _default_network(int_type)))
    if not gpkg.exists():
        return None, None, None
    try:
        present = set(fiona.listlayers(str(gpkg)))
    except Exception as exc:
        print(f"  [registry] {gpkg.name}: cannot list layers ({exc})")
        return None, None, None

    def _read(layer: str) -> Optional[gpd.GeoDataFrame]:
        if layer not in present:
            return None
        gdf = gpd.read_file(gpkg, layer=layer)
        return gdf if not gdf.empty else None

    return _read(LAYER_NODES), _read(LAYER_SEGMENTS), _read(LAYER_COMPOSITION)


def read_records(
    int_type: str,
    int_ids: List[str],
    registry_path: Optional[str] = None,
    network: Optional[str] = None,
) -> Tuple[Optional[gpd.GeoDataFrame],
           Optional[gpd.GeoDataFrame],
           Optional[gpd.GeoDataFrame]]:
    """Return (nodes, segments, composition) filtered to the requested int_ids."""
    if not int_ids:
        return None, None, None
    nodes, segs, comp = read_registry(int_type, registry_path, network)
    wanted = {str(i) for i in int_ids}

    def _filter(gdf: Optional[gpd.GeoDataFrame]) -> Optional[gpd.GeoDataFrame]:
        if gdf is None or 'int_id' not in gdf.columns:
            return None
        sel = gdf[gdf['int_id'].astype(str).isin(wanted)].reset_index(drop=True)
        return sel if not sel.empty else None

    return _filter(nodes), _filter(segs), _filter(comp)


def list_intervention_ids(
    int_type: str,
    registry_path: Optional[str] = None,
    network: Optional[str] = None,
) -> List[str]:
    """Return all distinct int_ids in a registry (sorted; [] if absent)."""
    nodes, segs, comp = read_registry(int_type, registry_path, network)
    ids: set = set()
    for gdf in (nodes, segs, comp):
        if gdf is not None and 'int_id' in gdf.columns:
            ids.update(gdf['int_id'].dropna().astype(str).tolist())
    return sorted(ids)


def read_record_meta(
    int_type: str,
    int_id: str,
    registry_path: Optional[str] = None,
    network: Optional[str] = None,
) -> Optional[Dict]:
    """Return the metadata dict for one intervention (supersede + cost fields)."""
    nodes, segs, comp = read_records(int_type, [int_id], registry_path, network)
    for gdf in (segs, nodes, comp):
        if gdf is not None and not gdf.empty:
            row = gdf.iloc[0]
            meta = {c: (row[c] if c in gdf.columns else None) for c in META_COLS}
            for f in _LIST_FIELDS:
                meta[f] = deserialize_list(meta.get(f))
            return meta
    return None


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY — write
# ─────────────────────────────────────────────────────────────────────────────

def attach_metadata(
    gdf: Optional[gpd.GeoDataFrame],
    meta: Dict,
) -> Optional[gpd.GeoDataFrame]:
    """Stamp per-intervention metadata columns onto every row of a layer."""
    if gdf is None or gdf.empty:
        return gdf
    out = gdf.copy()
    for col in META_COLS:
        if col not in meta:
            continue
        val = meta[col]
        if col in _LIST_FIELDS and isinstance(val, (list, tuple)):
            val = serialize_list(val)
        out[col] = val
    return out


def append_records(
    int_type: str,
    nodes: Optional[gpd.GeoDataFrame] = None,
    segments: Optional[gpd.GeoDataFrame] = None,
    composition: Optional[gpd.GeoDataFrame] = None,
    registry_path: Optional[str] = None,
    network: Optional[str] = None,
) -> None:
    """Append intervention rows to a registry, replacing rows with the same int_id."""
    gpkg = Path(registry_path or paths.get_infra_int_registry(int_type, network or _default_network(int_type)))
    gpkg.parent.mkdir(parents=True, exist_ok=True)

    existing_layers: set = set()
    if gpkg.exists():
        try:
            existing_layers = set(fiona.listlayers(str(gpkg)))
        except Exception:
            pass

    def _append(layer: str, new_gdf: Optional[gpd.GeoDataFrame]) -> None:
        if new_gdf is None or new_gdf.empty:
            return
        new_gdf = gpd.GeoDataFrame(new_gdf, crs=SWISS_CRS)
        if layer in existing_layers:
            old = gpd.read_file(gpkg, layer=layer)
            if 'int_id' in old.columns and 'int_id' in new_gdf.columns:
                new_ids = set(new_gdf['int_id'].dropna().astype(str).tolist())
                old = old[~old['int_id'].astype(str).isin(new_ids)]
            combined = pd.concat([old, new_gdf], ignore_index=True)
            combined = gpd.GeoDataFrame(combined, crs=SWISS_CRS)
        else:
            combined = new_gdf
        combined.to_file(gpkg, layer=layer, driver='GPKG')

    _append(LAYER_NODES, nodes)
    _append(LAYER_SEGMENTS, segments)
    _append(LAYER_COMPOSITION, composition)
    print(f"  [registry] appended rows to {gpkg.name}")


def delete_records(
    int_type: str,
    int_ids: List[str],
    registry_path: Optional[str] = None,
    network: Optional[str] = None,
) -> None:
    """Remove all rows for the given int_ids from every layer of a registry."""
    gpkg = Path(registry_path or paths.get_infra_int_registry(int_type, network or _default_network(int_type)))
    if not gpkg.exists() or not int_ids:
        return
    try:
        present = set(fiona.listlayers(str(gpkg)))
    except Exception:
        return
    drop = {str(i) for i in int_ids}
    for layer in LAYERS:
        if layer not in present:
            continue
        gdf = gpd.read_file(gpkg, layer=layer)
        if 'int_id' not in gdf.columns:
            continue
        kept = gdf[~gdf['int_id'].astype(str).isin(drop)].reset_index(drop=True)
        kept = gpd.GeoDataFrame(kept, crs=SWISS_CRS)
        kept.to_file(gpkg, layer=layer, driver='GPKG')


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY — validation + id allocation
# ─────────────────────────────────────────────────────────────────────────────

def validate_record(
    meta: Dict,
    nodes_df: Optional[gpd.GeoDataFrame],
    base_segments: gpd.GeoDataFrame,
) -> Tuple[bool, List[str]]:
    """Check that an intervention's stable identities exist in a given base."""
    warnings: List[str] = []
    base_ids = set(base_segments['Segment_ID'].astype(str)) \
        if 'Segment_ID' in base_segments.columns else set()
    int_id = meta.get('int_id', '<unknown>')

    removes = meta.get('removes_base_rows')
    removes = removes if isinstance(removes, list) else deserialize_list(removes)
    for seg_id in removes:
        if str(seg_id) not in base_ids:
            warnings.append(f"{int_id}: removes_base_rows '{seg_id}' absent from base")

    if nodes_df is not None and 'on_segment' in nodes_df.columns:
        for host in nodes_df['on_segment'].dropna().astype(str):
            if host.strip() and host not in base_ids:
                warnings.append(
                    f"{int_id}: added_nodes.on_segment '{host}' absent from base")

    return (len(warnings) == 0), warnings


def next_int_id(
    int_type: str,
    int_subtype: str,
    registry_path: Optional[str] = None,
    network: Optional[str] = None,
) -> str:
    """Allocate the next sequential int_id for a subtype (e.g. 'cc_0001')."""
    code = _SUBTYPE_CODE.get(int_subtype)
    if code is None:
        raise ValueError(f"unknown int_subtype '{int_subtype}'")
    prefix = f"{code}_"
    existing = [i for i in list_intervention_ids(int_type, registry_path, network)
                if str(i).startswith(prefix)]
    max_n = 0
    for i in existing:
        tail = str(i)[len(prefix):]
        if tail.isdigit():
            max_n = max(max_n, int(tail))
    return f"{prefix}{max_n + 1:04d}"


# ═════════════════════════════════════════════════════════════════════════════
# COMPOSE — public API
# ═════════════════════════════════════════════════════════════════════════════

# Readable derived names up to this length; longer int-sets fall back to a bounded form
# so the path stays clear of the Windows MAX_PATH (~260) limit.
_DERIVED_NAME_MAXLEN = 80


def derived_version_name(base_version: str, int_ids: List[str]) -> str:
    """Readable derived-version name ``<base>__<id>+<id>…`` (empty list → base).

    Small int-sets — the common case, e.g. an NDC pulling in a single CC — get a
    self-documenting name (``AS_2026_ZH__cc_0004``). Large sets (a full CAP set) fall
    back to ``<base>__<N>ints_<hash8>`` to stay within the Windows path limit. Content
    staleness is handled by the manifest fingerprint, not the name, so reusing the same
    name across content changes is safe (see ``compose_infra``).
    """
    if not int_ids:
        return base_version
    ids = sorted(str(i) for i in int_ids)
    joined = '+'.join(ids)
    if len(joined) <= _DERIVED_NAME_MAXLEN:
        return f"{base_version}__{joined}"
    h = hashlib.sha1(','.join(ids).encode('utf-8')).hexdigest()[:8]
    return f"{base_version}__{len(ids)}ints_{h}"


def _registry_network(int_type: str, base_version: str, svc_version: Optional[str]) -> str:
    """Partition key for reading a registry during compose — the '<infra>__<svc>' combo."""
    return _combo(base_version, svc_version)


def compose_frames(
    base_version: str,
    int_ids: List[str],
    svc_version: Optional[str] = None,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, List[str]]:
    """Compose in memory; return (nodes, segments, composition, warnings).

    Does not write to disk. Invalid records (stable ids absent from the base) are
    skipped with a warning. The result carries ``int_id`` / ``int_type`` tags on
    every added or changed row (null on untouched base rows). ``svc_version`` is
    required when CAP ids are present (their registry is keyed by '<infra>__<svc>').
    """
    base_dir = Path(paths.get_infra_version_dir(base_version))
    if not paths.infra_version_exists(base_version):
        raise FileNotFoundError(f"Base infra version '{base_version}' not found at {base_dir}")

    # load_version coerces numeric columns (Length, Num_Tracks, Gauge, …) — gpkg
    # round-trips can otherwise leave them as object/str and break comparisons.
    nodes, segs = ic.load_version(base_version)
    comp_path = base_dir / 'segments_composition.gpkg'
    comp = (gpd.read_file(comp_path) if comp_path.exists()
            else gpd.GeoDataFrame(columns=['Segment_ID'], geometry=[], crs=SWISS_CRS))

    for col in _TAG_COLS:
        for frame in (nodes, segs, comp):
            if col not in frame.columns:
                frame[col] = pd.NA

    warnings: List[str] = []
    if not int_ids:
        return nodes, segs, comp, warnings

    # group requested ids by registry type (prefix before first '_')
    by_type: Dict[str, List[str]] = {}
    for iid in int_ids:
        t = _registry_type_of(str(iid))
        by_type.setdefault(t, []).append(str(iid))

    for int_type in sorted(by_type):
        net = _registry_network(int_type, base_version, svc_version)
        r_nodes, r_segs, r_comp = read_registry(int_type, network=net)
        for iid in sorted(by_type[int_type]):
            i_nodes = _rows_for(r_nodes, iid)
            i_segs = _rows_for(r_segs, iid)
            i_comp = _rows_for(r_comp, iid)
            meta = read_record_meta(int_type, iid, network=net)
            if meta is None:
                warnings.append(f"{iid}: no record found in '{int_type}' registry — skipped")
                continue
            ok, w = validate_record(meta, i_nodes, segs)
            if not ok:
                warnings.extend(w)
                warnings.append(f"{iid}: validation failed — skipped")
                continue
            nodes, segs, comp = _apply_intervention(
                nodes, segs, comp, iid, int_type, meta, i_nodes, i_segs, i_comp,
            )

    return nodes, segs, comp, warnings


def _intervention_fingerprint(base_version: str, int_ids: List[str],
                              svc_version: Optional[str] = None) -> str:
    """Content signature of the registries feeding a derived network.

    Keyed by each contributing registry's file mtime+size, so a derived folder is
    rebuilt when the curve/cap rows behind it change even though the int-id set — and
    thus the folder name — stays the same. This is the guard that prevents the stale-
    cache trap (a fixed CC re-using the pre-fix composed geometry).
    """
    by_type: Dict[str, List[str]] = {}
    for iid in int_ids:
        t = 'cap' if str(iid).startswith('cap') else 'cc'
        by_type.setdefault(t, []).append(str(iid))
    parts = [base_version, f"base={_base_network_signature(base_version)}"]
    for t in sorted(by_type):
        net = _registry_network(t, base_version, svc_version)
        p = Path(paths.get_infra_int_registry(t, net))
        sig = f"{int(p.stat().st_mtime)}:{p.stat().st_size}" if p.exists() else "missing"
        parts.append(f"{t}[{','.join(sorted(by_type[t]))}]={sig}")
    return hashlib.sha1('|'.join(parts).encode('utf-8')).hexdigest()[:12]


def _base_network_signature(base_version: str) -> str:
    """Content signature (mtime+size) of the base network's geopackages.

    Folded into the derived-network fingerprint so a rebuilt base (e.g. re-enhanced
    AS_2026_ZH_enhanced) invalidates every derived folder composed from it — the
    int-registry signature alone would miss a base change that leaves the int rows
    untouched.
    """
    base_dir = Path(paths.get_derived_infra_version_dir(base_version)) \
        if paths.derived_version_exists(base_version) \
        else Path(paths.get_infra_version_dir(base_version))
    sigs = []
    for fname in ('nodes.gpkg', 'segments.gpkg', 'segments_composition.gpkg'):
        f = base_dir / fname
        sigs.append(f"{int(f.stat().st_mtime)}:{f.stat().st_size}" if f.exists() else "x")
    return ','.join(sigs)


def compose_infra(base_version: str, int_ids: List[str], svc_version: Optional[str] = None) -> str:
    """Return the name of a cached derived version with the given ints applied.

    Reuses an existing derived folder only when its manifest fingerprint still matches
    the current registry content; otherwise (re)composes and writes
    nodes/segments/segments_composition.gpkg + manifest.json under the shared
    Developments/Derived/<name>. ``svc_version`` is required when CAP ids are in
    the set (their registry is keyed by '<infra>__<svc>').
    """
    name = derived_version_name(base_version, int_ids)
    if not int_ids:
        return base_version

    target = Path(paths.get_derived_infra_version_dir(name))
    fp = _intervention_fingerprint(base_version, int_ids, svc_version)
    if (target.is_dir() and paths.derived_version_exists(name)
            and _manifest_fingerprint(target) == fp):
        print(f"  [compose] reusing cached '{name}'")
        return name
    if target.is_dir():
        print(f"  [compose] '{name}' is stale (registry changed) — rebuilding")

    print(f"  [compose] building '{name}' from '{base_version}' "
          f"({len(int_ids)} int(s))")
    nodes, segs, comp, warnings = compose_frames(base_version, int_ids, svc_version)
    for w in warnings:
        print(f"  [compose]   WARN {w}")

    _validate_and_write(nodes, segs, comp, target, name)
    _write_manifest(target, base_version, int_ids, fingerprint=fp)
    return name


def build_master_network(base_version: str, svc_version: Optional[str] = None) -> str:
    """Compose every registered int into one tagged 'Dev_Full' reference.

    With no ``svc_version`` it unions only the CC registry (the 5A CC-only master,
    ``<base>_full``); with one it also unions the CAP registry (the 5C full master,
    ``<base>__<svc>_full``). Both read from the same ``<infra>__<svc>`` combo partition
    (decision H). Stored under the shared Developments/Dev_Full/<name>/. For visualisation /
    sanity checking; the per-svc-int path uses compose_infra.
    """
    combo = _combo(base_version, svc_version)
    all_ids: List[str] = list_intervention_ids('cc', network=combo)
    if svc_version:
        all_ids += list_intervention_ids('cap', network=combo)
    all_ids = sorted(set(all_ids))

    full_name = f"{base_version}__{svc_version}_full" if svc_version else f"{base_version}_full"
    target = Path(paths.MAIN) / paths.DEVELOPMENTS_DEV_FULL_DIR / full_name
    target.mkdir(parents=True, exist_ok=True)

    if not all_ids:
        print(f"  [master] no infra ints registered; '{full_name}' mirrors base.")
    else:
        print(f"  [master] applying {len(all_ids)} infra int(s): {all_ids}")

    nodes, segs, comp, warnings = compose_frames(base_version, all_ids, svc_version)
    for w in warnings:
        print(f"  [master]   WARN {w}")
    _validate_and_write(nodes, segs, comp, target, full_name, build_qgz=True)
    _write_manifest(target, base_version, all_ids)
    return full_name


def walk_segment_chain(adjacency: Dict[int, set], a: int, b: int,
                       max_hops: int = 50,
                       pass_nodes: Optional[set] = None) -> Optional[List[int]]:
    """Shortest pass-through node path a→…→b over a split host segment.

    A CC/CAP split turns host a–b into a chain a→j1→…→b. Intermediate nodes
    must be pass-through: degree-2 in the segment graph, or members of
    ``pass_nodes`` (junction-class — a CC wye junction carries the curve, so
    it is degree-3 in the full graph; the capacity graph mode-filters the
    curve away, hence degree-2 there). BFS shortest-hop with sorted neighbor
    expansion: the direct piece chain always beats a detour through the curve
    and ties resolve deterministically. Used to re-expand service hops
    projected on the pre-split infra onto the composed sections (5C capacity
    supply, 6D flow unroll — decision 2026-06-10).

    Args:
        adjacency:  node -> set of neighbor nodes (undirected segment graph).
        a, b:       hop endpoints (BAV node numbers).
        max_hops:   safety bound on path length.
        pass_nodes: node numbers traversable regardless of degree
                    (junction-class nodes).

    Returns:
        Full node path [a, j1, …, b], or None when b is only reachable
        through a non-pass-through node (e.g. a real station).
    """
    from collections import deque

    allowed = pass_nodes or set()
    prev: Dict[int, Optional[int]] = {a: None}
    queue = deque([(a, 0)])
    while queue:
        cur, dist = queue.popleft()
        if dist >= max_hops:
            continue
        for nxt in sorted(adjacency.get(cur, ())):
            if nxt in prev:
                continue
            prev[nxt] = cur
            if nxt == b:
                path = [b]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1]
            if len(adjacency.get(nxt, ())) == 2 or nxt in allowed:
                queue.append((nxt, dist + 1))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSE — per-intervention application
# ─────────────────────────────────────────────────────────────────────────────

def _apply_intervention(
    nodes, segs, comp, int_id, int_type, meta, i_nodes, i_segs, i_comp,
):
    """Apply one intervention's supersede triple to the working frames."""
    seg_ids_before = set(segs['Segment_ID'].astype(str)) if 'Segment_ID' in segs.columns else set()

    # 1. junction nodes + host splits ----------------------------------------
    if i_nodes is not None and not i_nodes.empty:
        split_nodes = i_nodes[i_nodes['on_segment'].astype(str).str.strip().replace('nan', '') != ''] \
            if 'on_segment' in i_nodes.columns else i_nodes.iloc[0:0]
        modify_nodes = i_nodes[~i_nodes.index.isin(split_nodes.index)]

        # 1a. nodes that modify an existing node in place (e.g. station-track)
        if not modify_nodes.empty:
            nodes, _, _ = ic.merge_nodes(
                nodes, _strip_int_cols(modify_nodes), None, None,
                snap_and_split=False, replace_existing=True,
            )

        # 1b. junctions that split a host segment, grouped by host
        if not split_nodes.empty:
            for host_id, grp in split_nodes.groupby(split_nodes['on_segment'].astype(str)):
                nodes, segs, comp = _split_host_for_junctions(
                    nodes, segs, comp, host_id, grp,
                )

    # 2. explicit removes (idempotent — splits already dropped their hosts) ---
    removes = meta.get('removes_base_rows') or []
    if removes and 'Segment_ID' in segs.columns:
        still = segs['Segment_ID'].astype(str).isin({str(r) for r in removes})
        if still.any():
            dropped = set(segs.loc[still, 'Segment_ID'].astype(str))
            segs = segs[~still].reset_index(drop=True)
            if 'Segment_ID' in comp.columns:
                comp = comp[~comp['Segment_ID'].astype(str).isin(dropped)].reset_index(drop=True)

    # 3. adds_rows: modify-in-place or append-new ----------------------------
    if i_segs is not None and not i_segs.empty:
        for _, add in i_segs.iterrows():
            segs, comp = _apply_add_segment(segs, comp, add, i_comp, int_id, int_type)

    # 4. tag split-derived (new) segments with this int --------------------
    if 'Segment_ID' in segs.columns:
        new_ids = set(segs['Segment_ID'].astype(str)) - seg_ids_before
        if new_ids:
            mask = segs['Segment_ID'].astype(str).isin(new_ids)
            untagged = mask & segs['int_id'].isna()
            segs.loc[untagged, 'int_id'] = int_id
            segs.loc[untagged, 'int_type'] = int_type

    return nodes, segs, comp


def _split_host_for_junctions(nodes, segs, comp, host_id, junctions):
    """Split host segment ``host_id`` at every junction in ``junctions``.

    Multiple junctions are applied descending by ``split_position_frac`` so each
    cut is measured from the original segment start and lands in the first
    remaining piece.
    """
    if 'Segment_ID' not in segs.columns:
        return nodes, segs, comp
    host_mask = segs['Segment_ID'].astype(str) == str(host_id)
    if not host_mask.any():
        print(f"  [compose]   WARN host segment '{host_id}' absent — junctions skipped")
        return nodes, segs, comp

    host_len = float(segs.loc[host_mask, 'Length'].iloc[0]) \
        if 'Length' in segs.columns else float(segs.loc[host_mask].geometry.length.iloc[0])
    from_code = _from_code_of(segs.loc[host_mask].iloc[0], nodes)

    grp = junctions.copy()
    grp['_frac'] = pd.to_numeric(grp['split_position_frac'], errors='coerce').fillna(0.5)
    grp = grp.sort_values('_frac', ascending=False)

    # add the junction nodes (no snap/split here — split is explicit below)
    nodes, _, _ = ic.merge_nodes(
        nodes, _strip_int_cols(grp.drop(columns=['_frac'])), None, None,
        snap_and_split=False, replace_existing=False,
    )

    current_id = str(host_id)
    for _, jn in grp.iterrows():
        idx = segs.index[segs['Segment_ID'].astype(str) == current_id]
        if len(idx) == 0:
            print(f"  [compose]   WARN piece '{current_id}' vanished mid-split")
            break
        seg_idx = idx[0]
        split_dist = float(jn['_frac']) * host_len
        node_name = str(jn.get('Name', ''))
        node_code = str(jn.get('Code', node_name[:4].upper()))
        node_class = str(jn.get('Node_Class', 'junction'))
        segs, comp = ic.split_segment_at(
            segs, comp, seg_idx, split_dist, node_name, node_code, nodes, node_class,
        )
        # the first piece retains the original From; its id is deterministic
        current_id = f"c{from_code}_{node_code}"

    return nodes, segs, comp


def _apply_add_segment(segs, comp, add, i_comp, int_id, int_type):
    """Modify a matching existing segment in place, else append ``add`` as new."""
    add_sid = str(add.get('Segment_ID', '')).strip()
    from_n = str(add.get('From_Name', '')).strip()
    to_n = str(add.get('To_Name', '')).strip()

    match = None
    if add_sid and 'Segment_ID' in segs.columns:
        m = segs['Segment_ID'].astype(str) == add_sid
        if m.any():
            match = m
    if match is None and from_n and to_n and {'From_Name', 'To_Name'} <= set(segs.columns):
        m = ((segs['From_Name'].astype(str) == from_n) &
             (segs['To_Name'].astype(str) == to_n))
        if m.any():
            match = m

    if match is not None:
        # modify-in-place: copy attributes that the add row specifies
        for col in ('Num_Tracks', 'Electrification_Class', 'Gauge'):
            if col in add.index and pd.notna(add[col]):
                segs.loc[match, col] = add[col]
        segs.loc[match, 'int_id'] = int_id
        segs.loc[match, 'int_type'] = int_type
        if 'Num_Tracks' in add.index and pd.notna(add['Num_Tracks']) and 'Segment_ID' in segs.columns:
            sids = set(segs.loc[match, 'Segment_ID'].astype(str))
            if 'Num_Tracks' in comp.columns and 'Segment_ID' in comp.columns:
                comp.loc[comp['Segment_ID'].astype(str).isin(sids), 'Num_Tracks'] = add['Num_Tracks']
        return segs, comp

    # append as new segment (e.g. connecting curve)
    new_seg = gpd.GeoDataFrame([add], crs=SWISS_CRS)
    new_seg['int_id'] = int_id
    new_seg['int_type'] = int_type
    new_comp = _composition_for_new_segment(add, i_comp, int_id, int_type)
    segs, comp = ic.merge_segments(segs, comp, new_seg, new_comp, replace_existing=True)
    return segs, comp


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSE — composition handling for appended (new) segments
# ─────────────────────────────────────────────────────────────────────────────

def _composition_for_new_segment(add, i_comp, int_id, int_type):
    """Build composition rows for an appended segment.

    Uses the registry's manually-authored composition pieces for this segment when
    present (reconciling their total length to the segment length on ``normal``
    pieces only); otherwise synthesises a single ``normal`` full-length piece.
    """
    add_sid = str(add.get('Segment_ID', '')).strip()
    seg_len = (float(add['Length']) if 'Length' in add.index and pd.notna(add['Length'])
               else float(add.geometry.length))

    pieces = None
    if i_comp is not None and not i_comp.empty and 'Segment_ID' in i_comp.columns:
        sel = i_comp[i_comp['Segment_ID'].astype(str) == add_sid].copy()
        if not sel.empty:
            pieces = sel

    if pieces is None:
        pieces = gpd.GeoDataFrame([{
            'Segment_ID': add.get('Segment_ID'),
            'From_Name': add.get('From_Name'), 'To_Name': add.get('To_Name'),
            'Engineering_Structure': 'normal', 'Edge_Level': 1, 'Under_Construction': 0,
            'Piece_Length': seg_len,
            'Num_Tracks': add.get('Num_Tracks'), 'Gauge': add.get('Gauge'),
            'geometry': add.geometry,
        }], crs=SWISS_CRS)
    else:
        pieces = _reconcile_piece_lengths(pieces, seg_len)

    pieces = pieces.copy()
    pieces['int_id'] = int_id
    pieces['int_type'] = int_type
    return pieces


def _reconcile_piece_lengths(pieces: gpd.GeoDataFrame, seg_len: float) -> gpd.GeoDataFrame:
    """Absorb a length gap on ``normal`` pieces only; never bridge/tunnel."""
    pieces = pieces.copy()
    total = float(pd.to_numeric(pieces['Piece_Length'], errors='coerce').fillna(0).sum())
    gap = seg_len - total
    if abs(gap) <= _COMP_LEN_TOL_M:
        return pieces

    is_normal = pieces['Engineering_Structure'].astype(str).str.lower() == 'normal'
    norm_len = float(pd.to_numeric(pieces.loc[is_normal, 'Piece_Length'],
                                   errors='coerce').fillna(0).sum())
    if not is_normal.any() or norm_len <= 0:
        print(f"  [compose]   WARN composition gap {gap:.1f} m but no 'normal' "
              f"piece to absorb it; leaving as-is")
        return pieces

    for idx in pieces.index[is_normal]:
        share = float(pieces.at[idx, 'Piece_Length']) / norm_len
        pieces.at[idx, 'Piece_Length'] = max(0.0, float(pieces.at[idx, 'Piece_Length']) + gap * share)
    print(f"  [compose]   reconciled composition by {gap:+.1f} m on normal pieces")
    return pieces


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSE — helpers
# ─────────────────────────────────────────────────────────────────────────────

def _registry_type_of(int_id: str) -> str:
    """Map an int_id to its registry short code ('cc_*' → 'cc', 'cap_*' → 'cap')."""
    return int_id.split('_', 1)[0].lower()


def _rows_for(gdf: Optional[gpd.GeoDataFrame], int_id: str) -> Optional[gpd.GeoDataFrame]:
    if gdf is None or 'int_id' not in gdf.columns:
        return None
    sel = gdf[gdf['int_id'].astype(str) == str(int_id)].reset_index(drop=True)
    return sel if not sel.empty else None


def _strip_int_cols(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop registry-only columns so a row matches the base node/segment schema."""
    drop = [c for c in (META_COLS + NODE_SPLIT_COLS) if c in gdf.columns]
    return gdf.drop(columns=drop) if drop else gdf


def _from_code_of(seg_row, nodes: gpd.GeoDataFrame) -> str:
    """The Code of a segment's From node (mirrors split_segment_at's id scheme)."""
    fn = seg_row.get('From_Name')
    rows = nodes[nodes['Name'] == fn] if 'Name' in nodes.columns else nodes.iloc[0:0]
    return rows.iloc[0]['Code'] if not rows.empty else str(fn)


def _write_manifest(target: Path, base_version: str, int_ids: List[str],
                    fingerprint: Optional[str] = None) -> None:
    """Record the int-id set + registry fingerprint that produced a derived/master folder."""
    manifest = {'base': base_version, 'ints': sorted(str(i) for i in int_ids),
                'fingerprint': fingerprint}
    try:
        (target / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    except Exception as exc:
        print(f"  [compose]   (manifest skipped: {exc})")


def _manifest_fingerprint(target: Path) -> Optional[str]:
    """Read the stored registry fingerprint from a derived folder's manifest (None if absent)."""
    try:
        data = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
        return data.get('fingerprint')
    except Exception:
        return None


def _validate_and_write(nodes, segs, comp, target: Path, name: str, build_qgz: bool = False):
    # appended registry rows (curve, siding) can carry object-dtype numerics — coerce
    for col in ('Length', 'Num_Tracks', 'Gauge', 'Average_Speed', 'Predominant_Speed',
                'Speed_Coverage_Pct', 'TT_Stopping', 'TT_Passing'):
        if col in segs.columns:
            segs[col] = pd.to_numeric(segs[col], errors='coerce')
    nodes, segs, comp, ok = ic.validate_and_autofill(nodes, segs, comp, interactive=False)
    if not ok:
        raise RuntimeError(f"Validation failed composing '{name}'")
    target.mkdir(parents=True, exist_ok=True)
    nodes.to_file(target / 'nodes.gpkg', driver='GPKG')
    segs.to_file(target / 'segments.gpkg', driver='GPKG')
    if comp is not None and not comp.empty:
        comp.to_file(target / 'segments_composition.gpkg', driver='GPKG')
    if build_qgz:
        try:
            ic._build_infra_qgz(qgz_path=str(target / f"{name}.qgz"),
                                version_dir=target, name=name)
        except Exception as exc:
            print(f"  [compose]   (qgz skipped: {exc})")
    print(f"  [compose] wrote {target}")
    return nodes, segs, comp


# ═════════════════════════════════════════════════════════════════════════════
# SHARED — config + polygon resolvers
# ═════════════════════════════════════════════════════════════════════════════

def _resolve_base_version() -> str:
    """The base infra version name (Build_New → the configured build name)."""
    v = getattr(settings, 'INFRA_VERSION', 'Build_New')
    return getattr(settings, 'INFRA_BUILD_NEW_NAME', v) if v == 'Build_New' else v


def _resolve_base_version_propagated() -> str:
    """Standalone-CLI default base — the enhanced version when it exists.

    The pipeline runs Phase 5 on the propagated enhanced network (Phase 3B output), so a
    standalone run should default to the same network to land in the matching
    '<infra>__<svc>' combo. Prefers '<base>_enhanced' on disk; else the plain base.
    """
    base = _resolve_base_version()
    if base.endswith('_enhanced'):
        return base
    enhanced = f"{base}_enhanced"
    return enhanced if paths.infra_version_exists(enhanced) else base


def _resolve_svc_version() -> str:
    v = getattr(settings, 'SVC_VERSION', 'Build_New')
    return getattr(settings, 'SVC_BUILD_NEW_NAME', v) if v == 'Build_New' else v


def _load_polygon() -> Optional[object]:
    """Load the study-area boundary polygon for CC-centre restriction, if present."""
    return _read_polygon(paths.SA_BOUNDARY_PATH)


def _load_buffer() -> Optional[object]:
    """Load the study-area buffer polygon (candidate-station extent for CC discovery)."""
    return _read_polygon(paths.STUDY_AREA_BUFFER_GPKG)


def _read_polygon(rel_path) -> Optional[object]:
    p = Path(paths.MAIN) / rel_path
    if p.exists():
        try:
            gdf = gpd.read_file(p)
            if not gdf.empty:
                return gdf.geometry.union_all() if hasattr(gdf.geometry, 'union_all') \
                    else gdf.geometry.unary_union
        except Exception:
            pass
    return None


# ═════════════════════════════════════════════════════════════════════════════
# SHARED — per-int diff renderer (reuses infrabuild_network_builder.plot_infrastructure_diff)
# ═════════════════════════════════════════════════════════════════════════════
#
# Used by BOTH the CC plots (infra_ints_connecting_curve.plot_cc_changes, purple) and the
# CAP plots (mixed_ints_orchestrator.plot_cap_changes, orange); the combined base-vs-Dev_Full
# diff + developed-network maps live in the infra orchestrator.

def plot_out_dir(combo: str, subtype: Optional[str]) -> Path:
    """Return (and create) the mirrored plots dir plots/Developments/<combo>/<subtype>/."""
    out = Path(paths.get_developments_plot_dir(combo, subtype))
    out.mkdir(parents=True, exist_ok=True)
    return out


def render_int_diff(
    base_version: str,
    int_ids: List[str],
    out_tag: str,
    added_label: str,
    added_color: str,
    added_edge: Optional[str] = None,
    out_dir=None,
    extents=('CA', 'SA'),
    svc_version: Optional[str] = None,
    superseded_color: Optional[str] = None,
) -> List[str]:
    """Base-vs-(base+int_ids) diff at the requested extents, recoloured by int type.

    Reuses the infrabuild diff renderer (parallel-track style, track-gained/lost,
    lakes, labels); the developed side is composed in memory (no derived folder).
    ``out_dir`` routes the PDFs to the caller's mirrored plots folder; ``svc_version``
    is required when the int set includes CAP ids (cap registry keyed by '<infra>__<svc>').
    """
    if not int_ids:
        return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    base_nodes, base_segs = ic.load_version(base_version)
    dev_nodes, dev_segs, _, warns = compose_frames(base_version, int_ids, svc_version)
    for w in warns:
        print(f"  [plot]   WARN {w}")

    # Host segments superseded (split) by these ints — drawn in the lighter shade so
    # the strong added_color is reserved for the genuinely-new curve + junctions.
    superseded = _superseded_seg_ids(int_ids, base_version, svc_version) if superseded_color else None

    out_dir = Path(out_dir) if out_dir is not None \
        else plot_out_dir(_combo(base_version, svc_version), 'Dev_Full')
    out_dir.mkdir(parents=True, exist_ok=True)
    boundaries = _plot_boundaries()
    written: List[str] = []
    for ext_key in extents:
        bdry, ext, is_ca = boundaries[ext_key]
        net_a = _net_from_frames(base_nodes, base_segs, base_version, bdry)
        net_b = _net_from_frames(dev_nodes, dev_segs, f"{base_version}+{out_tag}", bdry)
        path = out_dir / f"infra_ints_{out_tag}_{ext_key}_{base_version}.pdf"
        fig = ic.plot_infrastructure_diff(
            net_a=net_a, net_b=net_b, extent=ext, output_path=path,
            is_catchment=is_ca, show_outside=(not is_ca),
            added_color=added_color, added_edge=added_edge, added_label=added_label,
            superseded_seg_ids=superseded, superseded_color=superseded_color,
        )
        plt.close(fig)
        written.append(str(path))
        print(f"  [plot] wrote {path.name}")
    return written


def _superseded_seg_ids(int_ids: List[str], base_version: str,
                        svc_version: Optional[str]) -> set:
    """Union of removes_base_rows across the given ints (the split host segments)."""
    out: set = set()
    for iid in int_ids:
        it = _registry_type_of(str(iid))
        meta = read_record_meta(it, str(iid),
                                network=_registry_network(it, base_version, svc_version))
        if meta:
            out.update(str(r) for r in (meta.get('removes_base_rows') or []))
    return out


def _net_from_frames(nodes, segs, version, boundary):
    """Build a NetworkData from in-memory frames (graph rebuilt each call)."""
    G = ic.build_networkx_graph(nodes, segs)
    return ic.NetworkData(nodes=nodes, segments=segs, graph=G,
                          version=version, boundary=boundary)


def _plot_boundaries() -> Dict[str, Tuple]:
    """Return {'CA': (boundary_gdf, extent, is_ca), 'SA': (...)} for diff/maps."""
    def _load(rel):
        p = Path(paths.MAIN) / rel
        try:
            return gpd.read_file(p) if p.exists() else None
        except Exception:
            return None

    def _extent(gdf, margin_m=2000):
        if gdf is None:
            return None
        b = gdf.total_bounds
        return (b[0] - margin_m, b[2] + margin_m, b[1] - margin_m, b[3] + margin_m)

    ca = _load(paths.CATCHMENT_AREA_BOUNDARY_GPKG)
    sa = _load(paths.STUDY_AREA_BOUNDARY_GPKG)
    return {'CA': (ca, _extent(ca), True), 'SA': (sa, _extent(sa), False)}


# ═════════════════════════════════════════════════════════════════════════════
# STANDALONE CLI helpers (enumerated pick, settings-default; shared by every int CLI)
# ═════════════════════════════════════════════════════════════════════════════
#
# Every int orchestrator's __main__ presents numbered choices (user enters a number,
# Enter = the settings default). Version pickers exclude Base* (not a selectable
# intervention base). When the module is wired into main_new the public function takes
# every decision as a parameter, so these prompts never run on the pipeline path.

def cli_header(title: str) -> None:
    """Print a boxed CLI title banner."""
    print("=" * 60)
    print(title)
    print("=" * 60)


def cli_step(n: int, title: str) -> None:
    """Print a ruled '[Step n / Qn]' section header."""
    print("\n" + "─" * 60)
    print(f"[Step {n} / Q{n}]  {title}")
    print("─" * 60)


def cli_pick(label: str, options, default=None) -> str:
    """Print a numbered menu and return the chosen option (Enter = default).

    The ``default`` is always kept selectable (prepended if not already listed), so the
    settings value is honoured even when it is not among the auto-discovered options.
    """
    opts = [str(o) for o in options]
    if default is not None and str(default) not in opts:
        opts = [str(default)] + opts
    if not opts:
        raw = input(f"  {label} [{default or ''}]: ").strip()
        return raw or (default or '')
    default = str(default) if (default is not None and str(default) in opts) else opts[0]
    print(f"  {label}")
    for i, o in enumerate(opts, 1):
        print(f"    {i}) {o}{'   (default)' if o == default else ''}")
    di = opts.index(default) + 1
    while True:
        raw = input(f"  Select [{di}]: ").strip()
        if not raw:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(opts):
            return opts[int(raw) - 1]
        print(f"    invalid — enter 1–{len(opts)}")


def cli_pick_yesno(label: str, default_bool: bool) -> bool:
    """Numbered yes/no pick; returns a bool (Enter = the settings default)."""
    return cli_pick(label, ['yes', 'no'], 'yes' if default_bool else 'no') == 'yes'


def cli_infra_versions() -> List[str]:
    """Selectable infra versions for a CLI picker (excludes Base* and Developments)."""
    return [v for v in ic.list_versions() if not v.startswith('Base')]


def cli_svc_versions() -> List[str]:
    """Selectable service versions from Rail_Lines/<svc>_network (excludes Base*, svc-ints)."""
    root = Path(paths.MAIN) / paths.RAIL_LINES_DIR
    out: List[str] = []
    if root.exists():
        for d in sorted(root.iterdir()):
            if not (d.is_dir() and d.name.endswith('_network')):
                continue
            name = d.name[:-len('_network')]
            if name.startswith(('Base', 'ext_', 'ndc_')):
                continue
            out.append(name)
    return out
