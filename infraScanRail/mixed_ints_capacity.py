"""
mixed_ints_capacity — Bridge capacity-engine interventions into the cap registry.
Last modified: 2026-06-08

Converts the capacity engine's ``CapacityIntervention`` objects
(from capacity_workflow_wrapper.capacity_on_composed, the Phase 5C reactive CAP pass —
the sole cost-bearing CAP generator) into base-agnostic explicit-intent registry
records and appends them to ``cap_interventions.gpkg``
(under data/Developments/<infra>__<svc>/cap/). The engine itself is untouched;
this module is the only place capacity output reaches the registry.

Renamed from infra_ints_capacity on 2026-06-08: capacity is a **mixed** intervention
(it needs both infra and service data), so it lives in the mixed-ints family. The Phase 5C
orchestration is in ``mixed_ints_orchestrator``; this module holds only the record-builders.

All three CAP subtypes map onto the unified supersede schema (see ints_core), so they
compose exactly like a connecting curve:

  station_track  → node upsert (Track_Count +1); on_segment empty → modify in place.
  segment_track  → segment upsert (Num_Tracks +1) by Segment_ID → modify in place.
  siding_track   → 2 junction nodes (on_segment = host Segment_ID, split_position_frac
                   = kp/length) + a middle segment (Num_Tracks +1); compose splits the
                   host and bumps the middle piece. Junctions are pass-through
                   (Node_Class='junction'), so routing is unaffected.
"""

from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd
from shapely.ops import linemerge

import paths
import settings
import ints_core as core   # registry + compose engine (shared substrate)
from capacity_interventions import (
    _plan_siding_km_positions,
    _lookup_segment,
    _next_junction_nr,
    _CAP_JUNCTION_NR_START,
)

_INT_TYPE = 'cap'

# Capacity-intervention diff colours (distinct from the standard green/red diff).
_CAP_COLOR = '#ff7f0e'   # orange fill for added tracks/sidings
_CAP_EDGE = '#a65000'    # darker orange node edge


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def register_cap_interventions(
    interventions: List,
    infra_version: str,
    registry_path: Optional[str] = None,
    *,
    host_version: Optional[str] = None,
    requires: Optional[List[str]] = None,
    id_namespace: Optional[str] = None,
    network: Optional[str] = None,
) -> List[str]:
    """Convert and append capacity interventions to the cap registry.

    Args:
        interventions: list of CapacityIntervention from capacity_on_composed() (Phase 5C).
        infra_version: base infra version the interventions were designed against.
        registry_path: explicit registry gpkg (defaults to paths.get_infra_int_registry).
        network: cap registry partition key ('<infra>__<svc>'); routes the records to
            Developments/<combo>/cap/.
        host_version: version whose nodes/segments host the interventions — pass a
            composed (Derived base+CC) name for Phase 5C so a CAP on CC-altered topology
            resolves its host. When None (or == infra_version) the base version is read.
        requires: cc_id(s) each CAP requires (Phase 5C: the svc-int's CC), stamped into
            the record so compose orders the CAP after the curve. Defaults to [].
        id_namespace: token folded into each cap int_id (e.g. the svc-int id) so CAP
            from different svc-ints stay globally unique in the shared registry.

    Returns:
        Sorted list of int_ids registered.
    """
    if host_version and host_version != infra_version and paths.derived_version_exists(host_version):
        host_dir = Path(paths.get_derived_infra_version_dir(host_version))
    else:
        host_dir = Path(paths.get_infra_version_dir(infra_version))
    nodes_base = gpd.read_file(host_dir / 'nodes.gpkg')
    segs_base = gpd.read_file(host_dir / 'segments.gpkg')

    junction_nr = _starting_junction_nr(nodes_base, registry_path, network)

    out_nodes: List[gpd.GeoDataFrame] = []
    out_segs: List[gpd.GeoDataFrame] = []
    registered: List[str] = []

    for itv in interventions:
        int_id = _cap_int_id(itv, id_namespace)
        try:
            if itv.type == 'station_track':
                n, s = _record_station_track(itv, int_id, nodes_base, requires)
            elif itv.strategy == 'extra_track':
                n, s = _record_segment_track(itv, int_id, segs_base, requires)
            else:
                n, s = _record_siding(itv, int_id, segs_base, nodes_base, junction_nr, requires)
                junction_nr += 2
        except Exception as exc:
            print(f"  [cap-adapter] skipping {int_id}: {exc}")
            continue
        if n is not None and not n.empty:
            out_nodes.append(n)
        if s is not None and not s.empty:
            out_segs.append(s)
        registered.append(int_id)

    nodes_gdf = (gpd.GeoDataFrame(pd.concat(out_nodes, ignore_index=True), crs=core.SWISS_CRS)
                 if out_nodes else None)
    segs_gdf = (gpd.GeoDataFrame(pd.concat(out_segs, ignore_index=True), crs=core.SWISS_CRS)
                if out_segs else None)

    core.append_records(_INT_TYPE, nodes=nodes_gdf, segments=segs_gdf,
                        registry_path=registry_path, network=network)
    return sorted(set(registered))


# Note (2026-06-08): CAP coverage for CC-altered topology is intentionally NOT done
# from base-network capacity. A CC junction is a degree-3 section boundary, so it
# re-sections the host (each shorter sub-section + the new curve has its own recomputed
# UIC capacity); the base section's available capacity is invalid across that split.
# Correct CAP-on-CC therefore requires recomputing sections+capacity on the COMPOSED
# network — handled by the per-svc-int recompute in mixed_ints_orchestrator (Phase 5C).


# ─────────────────────────────────────────────────────────────────────────────
# Per-subtype record builders
# ─────────────────────────────────────────────────────────────────────────────

def _record_station_track(itv, int_id, nodes_base, requires=None):
    """Node upsert: existing station node with Track_Count (+ optional Platform_Count) raised."""
    mask = nodes_base['Number'] == itv.node_id
    if not mask.any():
        mask = pd.to_numeric(nodes_base['Number'], errors='coerce') == float(itv.node_id)
    if not mask.any():
        raise KeyError(f"station node {itv.node_id} not found in base")
    node = nodes_base[mask].copy()
    node['Track_Count'] = (pd.to_numeric(node['Track_Count'], errors='coerce').fillna(0)
                           + max(1.0, float(getattr(itv, 'tracks_added', 1.0) or 1.0)))
    if getattr(itv, 'platforms_added', None):
        node['Platform_Count'] = (pd.to_numeric(node['Platform_Count'], errors='coerce').fillna(0)
                                  + itv.platforms_added)
    node['on_segment'] = ''
    node['split_position_frac'] = pd.NA
    node = core.attach_metadata(node, _meta(itv, int_id, 'station_track', removes=[], requires=requires))
    return node, None


def _record_segment_track(itv, int_id, segs_base, requires=None):
    """Segment upsert by Segment_ID: whole segment Num_Tracks +1 (modify in place)."""
    from_nr, to_nr = itv.segment_id.split('-')
    seg = _lookup_segment(segs_base, from_nr, to_nr).copy()
    seg['Num_Tracks'] = seg['Num_Tracks'] + 1
    seg = core.attach_metadata(seg, _meta(itv, int_id, 'segment_track', removes=[], requires=requires))
    return None, seg


def _record_siding(itv, int_id, segs_base, nodes_base, junc_a_nr, requires=None):
    """2 junctions splitting the host + a middle segment with Num_Tracks +1."""
    from_nr, to_nr = itv.segment_id.split('-')
    seg = _lookup_segment(segs_base, from_nr, to_nr)
    geom = seg.geometry.iloc[0]
    line = linemerge(geom) if geom.geom_type == 'MultiLineString' else geom
    seg_len = line.length
    host_id = str(seg['Segment_ID'].iloc[0])
    base_tracks = float(seg['Num_Tracks'].iloc[0])

    kp_A, kp_B = _plan_siding_km_positions(itv.section_length_m, itv.siding_length_m)
    frac_A = min(max(kp_A / seg_len, 0.0), 1.0) if seg_len > 0 else 0.4
    frac_B = min(max(kp_B / seg_len, 0.0), 1.0) if seg_len > 0 else 0.6
    pt_A, pt_B = line.interpolate(kp_A), line.interpolate(kp_B)

    junc_b_nr = junc_a_nr + 1
    template = _junction_template(nodes_base)
    name_a, name_b = f"{int_id}_A", f"{int_id}_B"

    def _junction(nr, name, point, frac):
        row = template.copy()
        row['Number'] = nr
        row['Node_ID'] = f"cap_junc_{nr}"
        row['Name'] = name
        row['Code'] = f"CJ{nr % 10000:04d}"
        row['E'] = point.x
        row['N'] = point.y
        row['Node_Class'] = 'junction'
        row['Track_Count'] = base_tracks
        row['Platform_Count'] = pd.NA
        row['Parent_Node'] = pd.NA
        row['on_segment'] = host_id
        row['split_position_frac'] = frac
        row['geometry'] = point
        return row

    nodes = gpd.GeoDataFrame(
        pd.concat([_junction(junc_a_nr, name_a, pt_A, frac_A),
                   _junction(junc_b_nr, name_b, pt_B, frac_B)], ignore_index=True),
        crs=core.SWISS_CRS,
    )

    # middle segment: matched in compose by (From_Name, To_Name) after the split
    from shapely.geometry import LineString
    mid = gpd.GeoDataFrame({
        'Segment_ID': [f"{int_id}_mid"],
        'From_Name': [name_a], 'To_Name': [name_b],
        'Num_Tracks': [base_tracks + 1.0],
        'Gauge': [seg['Gauge'].iloc[0] if 'Gauge' in seg.columns else pd.NA],
        'Electrification_Class': [seg['Electrification_Class'].iloc[0]
                                  if 'Electrification_Class' in seg.columns else pd.NA],
        'geometry': [LineString([pt_A, pt_B])],
    }, crs=core.SWISS_CRS)

    meta = _meta(itv, int_id, 'siding_track', removes=[host_id], requires=requires)
    nodes = core.attach_metadata(nodes, meta)
    mid = core.attach_metadata(mid, meta)
    return nodes, mid


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cap_int_id(itv, namespace: Optional[str] = None) -> str:
    """Map an intervention to its registry int_id (preserves the engine's counter).

    A namespace (e.g. the svc-int id) is folded in so CAP from different svc-ints stay
    globally unique in the shared registry: ``cap_<namespace>_st_0001``.
    """
    counter = int(str(itv.intervention_id).split('_')[-1])
    if itv.type == 'station_track':
        sub = 'st'
    elif itv.strategy == 'extra_track':
        sub = 'et'
    else:
        sub = 'ps'
    return f"cap_{namespace}_{sub}_{counter:04d}" if namespace else f"cap_{sub}_{counter:04d}"


def _meta(itv, int_id, subtype, removes, requires=None):
    return {
        'int_id': int_id, 'int_type': _INT_TYPE, 'int_subtype': subtype,
        'base_authored': '', 'removes_base_rows': removes,
        'requires': list(requires or []), 'conflicts_with': [],
        'construction_cost_chf': float(getattr(itv, 'construction_cost_chf', 0.0) or 0.0),
        'maintenance_cost_annual_chf': float(getattr(itv, 'maintenance_cost_annual_chf', 0.0) or 0.0),
        'length_m': float(getattr(itv, 'length_m', 0.0) or 0.0),
        'design_speed_kmh': float(getattr(itv, 'design_speed_kmh', 0.0) or 0.0),
    }


def _junction_template(nodes_base: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    junctions = nodes_base[nodes_base['Node_Class'] == 'junction']
    src = junctions if not junctions.empty else nodes_base
    return src.iloc[[0]].copy()


def _starting_junction_nr(nodes_base, registry_path, network=None) -> int:
    """Next free cap-junction Number across both the base and the existing registry."""
    start = _next_junction_nr(nodes_base)
    reg_nodes, _, _ = core.read_registry(_INT_TYPE, registry_path, network=network)
    if reg_nodes is not None and 'Number' in reg_nodes.columns:
        nums = pd.to_numeric(reg_nodes['Number'], errors='coerce').dropna()
        cap_nums = nums[nums >= _CAP_JUNCTION_NR_START]
        if not cap_nums.empty:
            start = max(start, int(cap_nums.max()) + 1)
    return start
