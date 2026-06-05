"""Capacity calculator for the rail network.

Reads infrastructure from infrabuild outputs (nodes.gpkg, segments.gpkg) and
projected service data (rail_segments.gpkg) produced by services_service_projection.
Builds peak and off-peak station/segment frequency tables, runs the sectioning
graph-walk, and computes capacity and utilization per section.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import re
from typing import Dict, List, Optional, Set, Tuple

import geopandas as gpd
import pandas as pd

import paths
import settings

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

CAPACITY_ROOT = Path(paths.MAIN) / "data" / "Network" / "Capacity"

def capacity_output_path(network_label: str = None, output_dir: Path = None) -> Path:
    """Return the capacity workbook path for the active rail network.

    Directory structure:
      - CAPACITY_ROOT / Baseline / {network} / capacity_{network}_network.xlsx
      - CAPACITY_ROOT / Enhanced / {network}_enhanced / capacity_{network}_enhanced_network.xlsx
      - CAPACITY_ROOT / Developments / {dev_id} / capacity_{network}_dev_{dev_id}_network.xlsx

    Args:
        network_label: Optional custom network label (e.g., "AK_2035_dev_100023" or "AK_2035_enhanced").
                      If None, uses settings.rail_network.
        output_dir: Optional custom output directory.
                   - If None: auto-detects category (Baseline/Enhanced/Developments)
                   - If provided: uses directory directly (for Developments workflow)

    Returns:
        Path to the capacity workbook.
    """
    if network_label is not None:
        network_tag = network_label
    else:
        network_tag = getattr(settings, "rail_network", "current")  # Use the configured scenario name.

    safe_network_tag = re.sub(r"[^\w-]+", "_", str(network_tag)).strip("_") or "current"
    filename = f"capacity_{safe_network_tag}_network.xlsx"

    if output_dir is not None:
        # DEVELOPMENT MODE: Use provided directory directly (already in Developments subdirectory)
        return output_dir / filename
    else:
        # AUTO-DETECT MODE: Determine category based on network_label
        # Check if this is a development network (_dev_XXXXX pattern)
        dev_match = re.search(r"_dev_(\d+)", safe_network_tag)
        # Check if this is an enhanced network (_enhanced suffix)
        is_enhanced = "_enhanced" in safe_network_tag

        if dev_match:
            # DEVELOPMENT: CAPACITY_ROOT / Developments / {dev_id} / ...
            dev_id = dev_match.group(1)
            network_subdir = CAPACITY_ROOT / "Developments" / dev_id
        elif is_enhanced:
            # ENHANCED: CAPACITY_ROOT / Enhanced / {network}_enhanced / ...
            network_subdir = CAPACITY_ROOT / "Enhanced" / safe_network_tag
        else:
            # BASELINE: CAPACITY_ROOT / Baseline / {network} / ...
            network_subdir = CAPACITY_ROOT / "Baseline" / safe_network_tag

        network_subdir.mkdir(parents=True, exist_ok=True)
        return network_subdir / filename

DECIMAL_COMMA = ","

LV95_E_OFFSET = 2_000_000
LV95_N_OFFSET = 1_000_000

DEFAULT_HEADWAY_MIN = 3.0  # minutes

try:
    import openpyxl  # noqa: F401

    EXCEL_ENGINE = "openpyxl"
    APPEND_ENGINE = "openpyxl"
except ImportError as exc:  # pragma: no cover - fail fast if dependency missing
    raise ImportError(
        "The 'openpyxl' package is required to export Excel files. "
        "Install it and rerun the script."
    ) from exc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_output_directory() -> None:
    """Create the capacity output directory if it does not exist."""
    CAPACITY_ROOT.mkdir(parents=True, exist_ok=True)


def parse_int(value: str) -> int:
    """Convert a value to integer, returning zero when conversion fails."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def parse_float(value: str | float | int) -> float:
    """Convert numeric strings that may use comma decimals into floats."""
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return 0.0
    normalized = str(value).replace(DECIMAL_COMMA, ".")
    try:
        return float(normalized)
    except ValueError:
        return 0.0


def parse_bool_flag(value: str) -> bool:
    """Interpret various truthy strings used in the data extracts."""
    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized in {"true", "1", "yes", "wahr"}


_VIA_SENTINELS = {"", "nan", "-99", "-1", "[]", "[ ]"}


def extract_via_nodes(value: str) -> List[int]:
    """Return a list of node IDs that a service passes (from the Via column)."""
    if value is None:
        return []
    token = str(value).strip()
    if token.lower() in _VIA_SENTINELS:
        return []

    # The Via field mixes formats such as "[6, 2122]" or "1,8,8,5,,,3"
    matches = re.findall(r"-?\d+", token)
    nodes: List[int] = []
    for match in matches:
        try:
            as_int = int(match)
        except ValueError:
            continue
        # Negative codes are sentinels (e.g. -99) and should be ignored.
        if as_int >= 0:
            nodes.append(as_int)
    return nodes


def _format_frequency_value(value: float) -> str:
    """Format frequency values as floored integers (no fractional tphpd allowed)."""
    if value is None:
        return ""
    numeric = float(value)
    if math.isnan(numeric):
        return ""
    return str(int(math.floor(numeric)))


def _format_service_frequency_map(frequencies: Dict[str, float]) -> str:
    """Return formatted bidirectional service frequency tokens."""
    tokens: List[str] = []
    for service, freq in sorted(frequencies.items(), key=lambda item: item[0]):
        formatted = _format_frequency_value(freq)
        if formatted:
            tokens.append(f"{service}.{formatted}")
    return ", ".join(tokens)


def _format_service_direction_frequency_map(
    frequencies: Dict[Tuple[str, str], float]
) -> str:
    """Return formatted per-direction service frequency tokens."""
    tokens: List[str] = []
    for (service, direction), freq in sorted(
        frequencies.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        formatted = _format_frequency_value(freq)
        if not formatted:
            continue
        direction_token = str(direction).strip()
        if not direction_token:
            direction_token = "?"
        tokens.append(f"{service}.{direction_token}.{formatted}")
    return ", ".join(tokens)


def _parse_service_frequency_string(cell: str) -> Dict[str, float]:
    """Convert a string of 'Service.Frequency' tokens into a mapping."""
    result: Dict[str, float] = {}
    if not cell:
        return result
    tokens = [token.strip() for token in re.split(r"[;,]", str(cell)) if token.strip()]
    for token in tokens:
        parts = token.split(".", 1)
        if len(parts) != 2:
            continue
        service = parts[0].strip()
        freq = parse_float(parts[1])
        if service:
            result[service] = max(result.get(service, 0.0), freq)
    return result


def _parse_service_direction_frequency_string(cell: str) -> Dict[Tuple[str, str], float]:
    """Convert 'Service.Direction.Frequency' tokens into a mapping."""
    result: Dict[Tuple[str, str], float] = {}
    if not cell:
        return result
    tokens = [token.strip() for token in re.split(r"[;,]", str(cell)) if token.strip()]
    for token in tokens:
        first = token.split(".", 1)
        if len(first) != 2:
            continue
        service = first[0].strip()
        remainder = first[1]
        second = remainder.split(".", 1)
        if len(second) != 2:
            continue
        direction = second[0].strip()
        freq = parse_float(second[1])
        if service:
            key = (service, direction)
            result[key] = max(result.get(key, 0.0), freq)
    return result


# ---------------------------------------------------------------------------
# New infrabuild / projected-services loaders (replace legacy processed/ loaders)
# ---------------------------------------------------------------------------

def load_infra_data(infra_version: str) -> tuple:
    """Load nodes and segments from infrabuild outputs.

    Args:
        infra_version: Named infra version (e.g. 'AS_2026_ZH_enhanced').

    Returns:
        (nodes_df, segments_df, name_to_number, number_to_name)
        nodes_df columns: NR (int), Name, Code, Node_Class, Track_Count,
            Platform_Count, E_LV95, N_LV95
        segments_df columns: from_node (int), to_node (int), from_name, to_name,
            Num_Tracks, Length, TT_Stopping, TT_Passing, Average_Speed, geometry
    """
    from infrabuild_network_builder import load_version

    nodes_gdf, segments_gdf = load_version(infra_version)

    # Retain only rail (train) nodes and segments — excludes tram, funicular, etc.
    # Nodes with no Transport_Mode that are referenced by at least one train segment are
    # treated as train nodes (e.g. junction nodes like Winterthur Nord whose mode is unset).
    if "Transport_Mode" in segments_gdf.columns:
        train_seg_mask = segments_gdf["Transport_Mode"].str.contains("train", case=False, na=False)
        segments_gdf = segments_gdf[train_seg_mask].copy()

    if "Transport_Mode" in nodes_gdf.columns:
        explicit_train = nodes_gdf["Transport_Mode"].str.contains("train", case=False, na=False)
        unknown_mode = nodes_gdf["Transport_Mode"].isna() | (
            nodes_gdf["Transport_Mode"].astype(str).str.strip() == ""
        )
        adjacent_train = pd.Series(False, index=nodes_gdf.index)
        if "Name" in nodes_gdf.columns and not segments_gdf.empty:
            _train_names = (
                set(segments_gdf["From_Name"].dropna().astype(str))
                | set(segments_gdf["To_Name"].dropna().astype(str))
            )
            adjacent_train = unknown_mode & nodes_gdf["Name"].astype(str).isin(_train_names)
        train_node_mask = explicit_train | adjacent_train
        n_before = len(nodes_gdf)
        nodes_gdf = nodes_gdf[train_node_mask].copy()
        print(f"  load_infra_data: Transport_Mode filter kept {len(nodes_gdf)}/{n_before} nodes "
              f"({int(explicit_train.sum())} explicit train, "
              f"{int(adjacent_train.sum())} rescued via adjacent segments)")

    # Build name ↔ number lookups
    name_to_number: Dict[str, int] = {}
    number_to_name: Dict[int, str] = {}
    for _, row in nodes_gdf.iterrows():
        nr = row.get("Number")
        name = row.get("Name", "")
        if pd.notna(nr) and name:
            nr_int = int(float(nr))
            name_to_number[str(name)] = nr_int
            number_to_name[nr_int] = str(name)

    # Normalise nodes DataFrame
    nodes_df = pd.DataFrame({
        "NR":             nodes_gdf["Number"].apply(lambda v: int(float(v)) if pd.notna(v) else pd.NA),
        "Name":           nodes_gdf["Name"].astype(str),
        "Code":           nodes_gdf["Code"].astype(str) if "Code" in nodes_gdf.columns else "",
        "Node_Class":     nodes_gdf["Node_Class"].astype(str) if "Node_Class" in nodes_gdf.columns else "station",
        "Track_Count":    pd.to_numeric(nodes_gdf.get("Track_Count"), errors="coerce"),
        "Platform_Count": pd.to_numeric(nodes_gdf.get("Platform_Count"), errors="coerce"),
        "E_LV95":         pd.to_numeric(nodes_gdf["E"], errors="coerce"),
        "N_LV95":         pd.to_numeric(nodes_gdf["N"], errors="coerce"),
    })
    nodes_df = nodes_df.dropna(subset=["NR"]).reset_index(drop=True)
    nodes_df["NR"] = nodes_df["NR"].astype(int)

    # Normalise segments DataFrame: resolve integer node IDs from Name lookup
    seg_rows = []
    for _, row in segments_gdf.iterrows():
        from_name = str(row.get("From_Name", "") or "")
        to_name   = str(row.get("To_Name", "") or "")
        from_node = name_to_number.get(from_name)
        to_node   = name_to_number.get(to_name)
        if from_node is None or to_node is None:
            continue
        seg_rows.append({
            "from_node":    from_node,
            "to_node":      to_node,
            "from_name":    from_name,
            "to_name":      to_name,
            "Num_Tracks":   parse_float(row.get("Num_Tracks") or row.get("Track_Count")),
            "Length":       parse_float(row.get("Length")),
            "TT_Stopping":  parse_float(row.get("TT_Stopping")),
            "TT_Passing":   parse_float(row.get("TT_Passing")),
            "Average_Speed": parse_float(row.get("Average_Speed") or row.get("Predominant_Speed")),
            "geometry":     row.geometry,
        })
    segments_df = pd.DataFrame(seg_rows)
    if segments_df.empty:
        segments_df = pd.DataFrame(columns=[
            "from_node", "to_node", "from_name", "to_name",
            "Num_Tracks", "Length", "TT_Stopping", "TT_Passing", "Average_Speed", "geometry",
        ])

    print(f"  load_infra_data: {len(nodes_df)} nodes, {len(segments_df)} segments "
          f"from '{infra_version}'")
    return nodes_df, segments_df, name_to_number, number_to_name


def load_projected_services(svc_version: str, infra_version: str) -> pd.DataFrame:
    """Load projected service links expanded to per-BAV-segment contributions.

    Reads all route-type layers from the projected rail_segments.gpkg, then
    expands each GTFS link (from_stop → to_stop via path_nodes) into one row
    per consecutive BAV-segment pair.

    Args:
        svc_version:   Service version name (e.g. 'AK_2026').
        infra_version: Infra version used for projection (e.g. 'AS_2026_ZH_enhanced').

    Returns:
        DataFrame with one row per service × direction × BAV-segment:
          service, direction_id, from_stop_nr (int), to_stop_nr (int),
          from_stop_name, to_stop_name,
          seg_from_node (int), seg_to_node (int),
          freq_am_peak (float), freq_pm_peak (float),
          freq_peak (float), freq_offpeak (float),
          is_origin (bool), is_destination (bool)
    """
    import fiona

    gpkg_path = paths.get_projected_services_path(svc_version, infra_version)
    if not Path(gpkg_path).exists():
        raise FileNotFoundError(
            f"Projected services not found: {gpkg_path}\n"
            f"Run services_service_projection.py for '{svc_version}' on '{infra_version}' first."
        )

    # Read all route-type layers and concatenate
    layers = fiona.listlayers(gpkg_path)
    gdfs = []
    for layer in layers:
        try:
            gdf = gpd.read_file(gpkg_path, layer=layer)
            if not gdf.empty:
                gdf["_layer"] = layer
                gdfs.append(gdf)
        except Exception:
            continue

    if not gdfs:
        raise ValueError(f"No data found in projected services file: {gpkg_path}")

    raw = pd.concat(gdfs, ignore_index=True)
    print(f"  load_projected_services: {len(raw)} service links from '{svc_version}'"
          f" projected on '{infra_version}' ({len(layers)} layers)")

    # _freq_am_peak / _freq_pm_peak feed temporal-aware aggregation that avoids
    # double-counting when AM and PM peaks reverse direction; _freq_peak stays
    # as max(AM, PM) for the per-service display columns (services_tph(pd)).
    am   = pd.to_numeric(raw["freq_am_peak_dep_hr"],  errors="coerce").fillna(0.0)
    pm   = pd.to_numeric(raw["freq_pm_peak_dep_hr"],  errors="coerce").fillna(0.0)
    offp = pd.to_numeric(raw["freq_offpeak_dep_hr"],  errors="coerce").fillna(0.0)
    raw["_freq_am_peak"] = am
    raw["_freq_pm_peak"] = pm
    raw["_freq_peak"]    = am.combine(pm, max)
    raw["_freq_offpeak"] = offp

    # Drop rows where service never runs
    raw = raw[(raw["_freq_peak"] > 0) | (raw["_freq_offpeak"] > 0)].reset_index(drop=True)

    # Expand each service link into per-BAV-segment rows using path_nodes
    expanded_rows = []
    for _, row in raw.iterrows():
        # Parse path_nodes (semicolon-separated BAV Number integers)
        raw_path = str(row.get("path_nodes") or "")
        node_strs = [s.strip() for s in raw_path.split(";") if s.strip()]
        try:
            path = [int(float(s)) for s in node_strs]
        except (ValueError, TypeError):
            path = []

        # Fallback to from/to stop node IDs
        if len(path) < 2:
            nid_from = row.get("node_id_from")
            nid_to   = row.get("node_id_to")
            try:
                path = [int(float(nid_from)), int(float(nid_to))]
            except (TypeError, ValueError):
                continue

        n_pairs = len(path) - 1
        svc       = str(row.get("Service") or row.get("GTFS_ID") or "")
        direction = str(row.get("direction_id") or "")
        try:
            from_stop_nr = int(float(row.get("from_stop_nr") or row.get("node_id_from") or 0))
            to_stop_nr   = int(float(row.get("to_stop_nr")   or row.get("node_id_to")   or 0))
        except (TypeError, ValueError):
            from_stop_nr = to_stop_nr = 0

        from_stop_name = str(row.get("from_stop_name") or "")
        to_stop_name   = str(row.get("to_stop_name")   or "")
        freq_am_peak   = float(row["_freq_am_peak"])
        freq_pm_peak   = float(row["_freq_pm_peak"])
        freq_peak      = float(row["_freq_peak"])
        freq_offpeak   = float(row["_freq_offpeak"])

        for i in range(n_pairs):
            expanded_rows.append({
                "service":        svc,
                "direction_id":   direction,
                "from_stop_nr":   from_stop_nr,
                "to_stop_nr":     to_stop_nr,
                "from_stop_name": from_stop_name,
                "to_stop_name":   to_stop_name,
                "seg_from_node":  path[i],
                "seg_to_node":    path[i + 1],
                "freq_am_peak":   freq_am_peak,
                "freq_pm_peak":   freq_pm_peak,
                "freq_peak":      freq_peak,
                "freq_offpeak":   freq_offpeak,
                "is_origin":      (i == 0),
                "is_destination": (i == n_pairs - 1),
            })

    result = pd.DataFrame(expanded_rows)
    if result.empty:
        result = pd.DataFrame(columns=[
            "service", "direction_id", "from_stop_nr", "to_stop_nr",
            "from_stop_name", "to_stop_name", "seg_from_node", "seg_to_node",
            "freq_am_peak", "freq_pm_peak", "freq_peak", "freq_offpeak",
            "is_origin", "is_destination",
        ])
    print(f"  load_projected_services: expanded to {len(result)} per-segment rows")
    return result


# ---------------------------------------------------------------------------
# Spatially filtered loaders — Study Area and Catchment Area variants
# ---------------------------------------------------------------------------

def _extract_sa_node_set(infra_version: str) -> Set[int]:
    """Return all BAV node Numbers whose point geometry falls within the SA boundary.

    Spatially filters nodes.gpkg against study_area_boundary.gpkg (EPSG:2056).

    Args:
        infra_version: Infra version name (e.g. 'AS_2026_ZH_enhanced').

    Returns:
        Set of integer BAV node Numbers within the study area.
    """
    from infrabuild_network_builder import load_version

    sa_boundary_path = Path(paths.MAIN) / paths.STUDY_AREA_BOUNDARY_GPKG
    if not sa_boundary_path.exists():
        raise FileNotFoundError(f"SA boundary not found: {sa_boundary_path}")

    sa_boundary = gpd.read_file(str(sa_boundary_path)).to_crs(epsg=2056)
    sa_polygon  = sa_boundary.unary_union

    nodes_gdf, _ = load_version(infra_version)
    if nodes_gdf.crs is None:
        nodes_gdf = nodes_gdf.set_crs(epsg=2056)
    else:
        nodes_gdf = nodes_gdf.to_crs(epsg=2056)

    within_mask = nodes_gdf.geometry.within(sa_polygon)
    sa_nodes: Set[int] = set()
    for _, row in nodes_gdf[within_mask].iterrows():
        nr = row.get("Number")
        if pd.notna(nr):
            sa_nodes.add(int(float(nr)))

    print(f"  _extract_sa_node_set: {len(sa_nodes)} SA nodes for '{infra_version}'")
    return sa_nodes


def _get_ca_node_set(infra_version: str) -> Set[int]:
    """Return all BAV node Numbers whose point geometry falls within the CA boundary.

    Spatially filters nodes.gpkg against catchment_area_boundary.gpkg (EPSG:2056).

    Args:
        infra_version: Infra version name (e.g. 'AS_2026_ZH_enhanced').

    Returns:
        Set of integer BAV node Numbers within the catchment area.
    """
    from infrabuild_network_builder import load_version

    ca_boundary_path = Path(paths.MAIN) / paths.CA_BOUNDARY_PATH
    if not ca_boundary_path.exists():
        raise FileNotFoundError(f"CA boundary not found: {ca_boundary_path}")

    ca_boundary = gpd.read_file(str(ca_boundary_path)).to_crs(epsg=2056)
    ca_polygon = ca_boundary.unary_union

    nodes_gdf, _ = load_version(infra_version)
    if nodes_gdf.crs is None:
        nodes_gdf = nodes_gdf.set_crs(epsg=2056)
    else:
        nodes_gdf = nodes_gdf.to_crs(epsg=2056)

    within_mask = nodes_gdf.geometry.within(ca_polygon)
    ca_nodes: Set[int] = set()
    for _, row in nodes_gdf[within_mask].iterrows():
        nr = row.get("Number")
        if pd.notna(nr):
            ca_nodes.add(int(float(nr)))

    print(f"  _get_ca_node_set: {len(ca_nodes)} CA nodes for '{infra_version}'")
    return ca_nodes


def load_infra_data_filtered(infra_version: str, node_set: Set[int]) -> tuple:
    """Load nodes and segments from infrabuild outputs, filtered to a node set.

    Shared by the Study Area and Catchment Area workflows. Only segments where
    both endpoints are in node_set are returned.

    Args:
        infra_version: Named infra version (e.g. 'AS_2026_ZH_enhanced').
        node_set:      Set of integer BAV node Numbers to retain.

    Returns:
        Same (nodes_df, segments_df, name_to_number, number_to_name) tuple as
        load_infra_data(), but filtered to node_set.
    """
    nodes_df, segments_df, name_to_number, number_to_name = load_infra_data(infra_version)

    nodes_df = nodes_df[nodes_df["NR"].isin(node_set)].reset_index(drop=True)

    seg_mask = (
        segments_df["from_node"].isin(node_set) &
        segments_df["to_node"].isin(node_set)
    )
    segments_df = segments_df[seg_mask].reset_index(drop=True)

    # Rebuild name lookups restricted to filtered nodes
    name_to_number = {name: nr for name, nr in name_to_number.items() if nr in node_set}
    number_to_name = {nr: name for nr, name in number_to_name.items() if nr in node_set}

    print(f"  load_infra_data_filtered: {len(nodes_df)} nodes, {len(segments_df)} segments "
          f"({len(node_set)} node filter applied)")
    return nodes_df, segments_df, name_to_number, number_to_name


def load_projected_services_sa(
    svc_version: str,
    infra_version: str,
    sa_node_set: Set[int],
) -> pd.DataFrame:
    """Load SA-clipped projected service links from the full rail_segments.gpkg.

    Loads all service rows, then clips each row's path_nodes to the contiguous
    sub-sequence of SA nodes (strips leading/trailing non-SA nodes). Services
    that pass through the SA but have an origin or destination outside it are
    included for the segments they actually traverse within the SA.

    is_origin / is_destination are set True only when the clipped endpoint
    coincides with the service's true origin / destination — i.e. when no
    nodes were stripped from that end.

    Args:
        svc_version:   Service version name (e.g. 'AK_2026').
        infra_version: Infra version name (e.g. 'AS_2026_ZH_enhanced').
        sa_node_set:   Set of BAV node Numbers within the study area.

    Returns:
        DataFrame with the same schema as load_projected_services().
    """
    import fiona

    gpkg_path = Path(paths.get_projected_services_path(svc_version, infra_version))
    if not gpkg_path.exists():
        raise FileNotFoundError(
            f"Projected services not found: {gpkg_path}\n"
            f"Run services_service_projection.py for '{svc_version}' on '{infra_version}' first."
        )

    layers = fiona.listlayers(str(gpkg_path))
    gdfs = []
    for layer in layers:
        try:
            gdf = gpd.read_file(str(gpkg_path), layer=layer)
            if not gdf.empty:
                gdf["_layer"] = layer
                gdfs.append(gdf)
        except Exception:
            continue

    if not gdfs:
        raise ValueError(f"No data found in projected services file: {gpkg_path}")

    raw = pd.concat(gdfs, ignore_index=True)
    print(f"  load_projected_services_sa: {len(raw)} total service links from '{svc_version}' "
          f"({len(layers)} layers)")

    am   = pd.to_numeric(raw["freq_am_peak_dep_hr"], errors="coerce").fillna(0.0)
    pm   = pd.to_numeric(raw["freq_pm_peak_dep_hr"], errors="coerce").fillna(0.0)
    offp = pd.to_numeric(raw["freq_offpeak_dep_hr"], errors="coerce").fillna(0.0)
    raw["_freq_am_peak"] = am
    raw["_freq_pm_peak"] = pm
    raw["_freq_peak"]    = am.combine(pm, max)
    raw["_freq_offpeak"] = offp
    raw = raw[(raw["_freq_peak"] > 0) | (raw["_freq_offpeak"] > 0)].reset_index(drop=True)

    expanded_rows = []
    n_clipped = 0
    for _, row in raw.iterrows():
        raw_path = str(row.get("path_nodes") or "")
        node_strs = [s.strip() for s in raw_path.split(";") if s.strip()]
        try:
            path = [int(float(s)) for s in node_strs]
        except (ValueError, TypeError):
            path = []

        if len(path) < 2:
            nid_from = row.get("node_id_from")
            nid_to   = row.get("node_id_to")
            try:
                path = [int(float(nid_from)), int(float(nid_to))]
            except (TypeError, ValueError):
                continue

        # Clip path to contiguous SA sub-sequence (strip leading/trailing non-SA nodes)
        start_idx = next((i for i, n in enumerate(path) if n in sa_node_set), None)
        if start_idx is None:
            continue  # no SA nodes at all
        end_idx = len(path) - 1 - next(
            (i for i, n in enumerate(reversed(path)) if n in sa_node_set), 0
        )
        clipped = path[start_idx : end_idx + 1]
        if len(clipped) < 2:
            continue  # single SA node — no segment to contribute

        if start_idx > 0 or end_idx < len(path) - 1:
            n_clipped += 1

        n_pairs = len(clipped) - 1
        svc       = str(row.get("Service") or row.get("GTFS_ID") or "")
        direction = str(row.get("direction_id") or "")
        try:
            from_stop_nr = int(float(row.get("from_stop_nr") or row.get("node_id_from") or 0))
            to_stop_nr   = int(float(row.get("to_stop_nr")   or row.get("node_id_to")   or 0))
        except (TypeError, ValueError):
            from_stop_nr = to_stop_nr = 0

        from_stop_name  = str(row.get("from_stop_name") or "")
        to_stop_name    = str(row.get("to_stop_name")   or "")
        freq_am_peak    = float(row["_freq_am_peak"])
        freq_pm_peak    = float(row["_freq_pm_peak"])
        freq_peak       = float(row["_freq_peak"])
        freq_offpeak    = float(row["_freq_offpeak"])
        true_origin     = (start_idx == 0)
        true_destination = (end_idx == len(path) - 1)

        for i in range(n_pairs):
            expanded_rows.append({
                "service":        svc,
                "direction_id":   direction,
                "from_stop_nr":   from_stop_nr,
                "to_stop_nr":     to_stop_nr,
                "from_stop_name": from_stop_name,
                "to_stop_name":   to_stop_name,
                "seg_from_node":  clipped[i],
                "seg_to_node":    clipped[i + 1],
                "freq_am_peak":   freq_am_peak,
                "freq_pm_peak":   freq_pm_peak,
                "freq_peak":      freq_peak,
                "freq_offpeak":   freq_offpeak,
                "is_origin":      (i == 0 and true_origin),
                "is_destination": (i == n_pairs - 1 and true_destination),
            })

    result = pd.DataFrame(expanded_rows)
    if result.empty:
        result = pd.DataFrame(columns=[
            "service", "direction_id", "from_stop_nr", "to_stop_nr",
            "from_stop_name", "to_stop_name", "seg_from_node", "seg_to_node",
            "freq_am_peak", "freq_pm_peak", "freq_peak", "freq_offpeak",
            "is_origin", "is_destination",
        ])
    print(f"  load_projected_services_sa: {n_clipped} rows path-clipped, "
          f"expanded to {len(result)} per-segment rows")
    return result


def load_projected_services_filtered(
    svc_version: str,
    infra_version: str,
    node_set: Set[int],
) -> pd.DataFrame:
    """Load projected services filtered to segments within a node set.

    Reads the full rail_segments.gpkg, expands path_nodes, then retains only
    rows where both seg_from_node and seg_to_node are in node_set.

    Args:
        svc_version:   Service version name.
        infra_version: Infra version name.
        node_set:      Set of integer BAV node Numbers to retain.

    Returns:
        DataFrame with the same schema as load_projected_services(), filtered.
    """
    result = load_projected_services(svc_version, infra_version)
    mask = (
        result["seg_from_node"].isin(node_set) &
        result["seg_to_node"].isin(node_set)
    )
    filtered = result[mask].reset_index(drop=True)
    print(f"  load_projected_services_filtered: {len(filtered)} rows after node-set filter "
          f"({len(result)} total)")
    return filtered


# ---------------------------------------------------------------------------
# Scope-aware capacity table builders
# ---------------------------------------------------------------------------

def build_capacity_tables_sa(
    infra_version: str,
    svc_version: str,
    network_label: str,
    output_dir: Optional[Path] = None,
) -> tuple:
    """Build capacity tables for the Study Area scope.

    Uses SA-filtered infra (nodes from path_nodes in rail_segments_sa.gpkg)
    and SA-filtered services (rail_segments_sa.gpkg).

    Returns:
        (stations_peak, segments_peak, stations_offpeak, segments_offpeak,
         junction_numbers, sa_node_set)
    """
    print("  [SA] Extracting study area node set from infra geometry ...")
    sa_node_set = _extract_sa_node_set(infra_version)

    print("  [SA] Loading filtered infrastructure ...")
    nodes_df, segments_df, _n2nr, _nr2n = load_infra_data_filtered(infra_version, sa_node_set)

    print("  [SA] Loading SA-clipped projected services ...")
    service_links_df = load_projected_services_sa(svc_version, infra_version, sa_node_set)

    junction_numbers: Set[int] = set(
        nodes_df.loc[nodes_df["Node_Class"] == "junction", "NR"].astype(int)
    )
    print(f"  [SA] Junction nodes: {len(junction_numbers)}")

    results = {}
    for period in ("peak", "offpeak"):
        print(f"  [SA] Building {period} tables ...")
        sl = build_stop_lookup(service_links_df, period)
        st = aggregate_station_metrics(nodes_df, service_links_df, junction_numbers, period)
        sg = aggregate_segment_metrics(segments_df, service_links_df, sl, junction_numbers, period)
        results[period] = (st, sg)

    stations_peak,    segments_peak    = results["peak"]
    stations_offpeak, segments_offpeak = results["offpeak"]

    safe_label = re.sub(r"[^\w-]+", "_", network_label).strip("_") or "capacity_sa"
    out_dir = output_dir if output_dir is not None else CAPACITY_ROOT / infra_version / svc_version
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"capacity_{safe_label}.xlsx"
    with pd.ExcelWriter(out_path, engine=EXCEL_ENGINE) as writer:
        stations_peak.to_excel(writer,    sheet_name="Stations_Peak",    index=False)
        segments_peak.to_excel(writer,    sheet_name="Segments_Peak",    index=False)
        stations_offpeak.to_excel(writer, sheet_name="Stations_Offpeak", index=False)
        segments_offpeak.to_excel(writer, sheet_name="Segments_Offpeak", index=False)
    print(f"  [SA] Capacity workbook -> {out_path}")

    return stations_peak, segments_peak, stations_offpeak, segments_offpeak, junction_numbers, sa_node_set


def build_capacity_tables_ca(
    infra_version: str,
    svc_version: str,
    network_label: str,
    output_dir: Optional[Path] = None,
) -> tuple:
    """Build capacity tables for the Catchment Area scope.

    Uses CA-filtered infra (nodes within catchment_area_boundary.gpkg) and
    CA-filtered services (full rail_segments.gpkg, clipped to CA nodes).

    Returns:
        (stations_peak, segments_peak, stations_offpeak, segments_offpeak,
         junction_numbers, ca_node_set, sa_node_set)
    """
    print("  [CA] Extracting catchment area node set from boundary ...")
    ca_node_set = _get_ca_node_set(infra_version)

    print("  [CA] Extracting study area node set from infra geometry ...")
    sa_node_set = _extract_sa_node_set(infra_version)

    print("  [CA] Loading filtered infrastructure ...")
    nodes_df, segments_df, _n2nr, _nr2n = load_infra_data_filtered(infra_version, ca_node_set)

    print("  [CA] Loading filtered projected services ...")
    service_links_df = load_projected_services_filtered(svc_version, infra_version, ca_node_set)

    junction_numbers: Set[int] = set(
        nodes_df.loc[nodes_df["Node_Class"] == "junction", "NR"].astype(int)
    )
    print(f"  [CA] Junction nodes: {len(junction_numbers)}")

    results = {}
    for period in ("peak", "offpeak"):
        print(f"  [CA] Building {period} tables ...")
        sl = build_stop_lookup(service_links_df, period)
        st = aggregate_station_metrics(nodes_df, service_links_df, junction_numbers, period)
        sg = aggregate_segment_metrics(segments_df, service_links_df, sl, junction_numbers, period)
        results[period] = (st, sg)

    stations_peak,    segments_peak    = results["peak"]
    stations_offpeak, segments_offpeak = results["offpeak"]

    safe_label = re.sub(r"[^\w-]+", "_", network_label).strip("_") or "capacity_ca"
    out_dir = output_dir if output_dir is not None else CAPACITY_ROOT / infra_version / svc_version
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"capacity_{safe_label}.xlsx"
    with pd.ExcelWriter(out_path, engine=EXCEL_ENGINE) as writer:
        stations_peak.to_excel(writer,    sheet_name="Stations_Peak",    index=False)
        segments_peak.to_excel(writer,    sheet_name="Segments_Peak",    index=False)
        stations_offpeak.to_excel(writer, sheet_name="Stations_Offpeak", index=False)
        segments_offpeak.to_excel(writer, sheet_name="Segments_Offpeak", index=False)
    print(f"  [CA] Capacity workbook -> {out_path}")

    return (stations_peak, segments_peak, stations_offpeak, segments_offpeak,
            junction_numbers, ca_node_set, sa_node_set)


# ---------------------------------------------------------------------------
# Retired loaders (replaced by load_infra_data / load_projected_services)
# ---------------------------------------------------------------------------

def extract_stations_from_edges(*args, **kwargs):
    raise RuntimeError("extract_stations_from_edges() is retired. Use load_infra_data() instead.")


def load_service_links(*args, **kwargs):
    raise RuntimeError("load_service_links() is retired. Use load_projected_services() instead.")


def apply_enrichment(*args, **kwargs):
    raise RuntimeError("apply_enrichment() is retired. Infrastructure data comes from load_infra_data() instead.")


def load_corridor_nodes_from_master(*args, **kwargs):
    raise RuntimeError("load_corridor_nodes_from_master() is retired. Use load_infra_data() instead.")


def load_corridor_nodes(*args, **kwargs):
    raise RuntimeError("load_corridor_nodes() is retired. Use load_infra_data() instead.")

def build_stop_records(
    service_links: pd.DataFrame,
    corridor_node_ids: set[int],
) -> pd.DataFrame:
    """Derive stop frequencies per node directly from corridor service links."""
    stop_freq: Dict[Tuple[int, str, str], float] = {}

    for _, row in service_links.iterrows():
        frequency = row["Frequency"]
        if frequency <= 0:
            continue
        service = row["Service"]
        direction = row["Direction"]
        for node in (row["FromNode"], row["ToNode"]):
            if node not in corridor_node_ids:
                continue
            key = (node, service, direction)
            # Frequency per service-direction is constant along the corridor; keep the max to avoid duplicates.
            stop_freq[key] = max(stop_freq.get(key, 0.0), frequency)

    records = [
        {"Node": node, "Service": svc, "Direction": direction, "Frequency": freq}
        for (node, svc, direction), freq in stop_freq.items()
    ]
    if not records:
        return pd.DataFrame(columns=["Node", "Service", "Direction", "Frequency"])
    return pd.DataFrame(records)


def build_segment_contributions(
    service_links: pd.DataFrame,
    stop_lookup: set[Tuple[str, str, int]],
    corridor_node_ids: set[int],
) -> Dict[Tuple[int, int], Dict[str, object]]:
    """Aggregate stopping/passing frequencies per unordered segment pair."""
    contributions: Dict[Tuple[int, int], Dict[str, object]] = {}

    for _, row in service_links.iterrows():
        frequency = row["Frequency"]
        if frequency <= 0:
            continue
        service = row["Service"]
        direction = row["Direction"]
        path_nodes: List[int] = [row["FromNode"], *row["ViaNodes"], row["ToNode"]]
        for start, end in zip(path_nodes, path_nodes[1:]):
            if start not in corridor_node_ids or end not in corridor_node_ids:
                continue
            pair = tuple(sorted((start, end)))
            segment = contributions.setdefault(pair, {"stop_freq": 0.0, "pass_freq": 0.0})
            stop_start = (service, direction, start) in stop_lookup
            stop_end = (service, direction, end) in stop_lookup
            if stop_start and stop_end:
                segment["stop_freq"] += frequency
            else:
                segment["pass_freq"] += frequency

    return contributions


# ---------------------------------------------------------------------------
# Aggregation logic
# ---------------------------------------------------------------------------

def _binding_metrics_from_grouped(
    am_stop: Dict[str, float],
    am_pass: Dict[str, float],
    pm_stop: Dict[str, float],
    pm_pass: Dict[str, float],
    has_pm: bool,
) -> Tuple[float, float, float, float, float, float]:
    """Derive temporal-aware binding metrics from per-direction frequency sums.

    Each *_stop / *_pass argument maps direction_id → summed frequency at one
    segment or node within one peak window. For off-peak only AM is supplied
    (has_pm=False); the PM dicts should be empty.

    Returns (in order):
        stopping_tph    bidirectional binding throughput for stopping trains
                        — max(am_stop_total, pm_stop_total).
        passing_tph     same, for passing trains.
        total_tph       bidirectional binding throughput counting stop+pass
                        jointly — max over peak windows of (stop+pass) summed
                        across directions.
        stopping_tphpd  binding per-direction load on stopping infrastructure
                        — max over (peak window × direction).
        passing_tphpd   same, for passing.
        total_tphpd     binding per-direction load on shared infrastructure
                        — max over (peak window × direction) of (stop+pass)
                        at that bucket.

    Symmetric inputs reproduce the pre-Level-2 numbers exactly; asymmetric
    inputs replace the old `/2` halving with the binding peak × direction.
    """
    am_stop_total = sum(am_stop.values())
    am_pass_total = sum(am_pass.values())
    pm_stop_total = sum(pm_stop.values()) if has_pm else 0.0
    pm_pass_total = sum(pm_pass.values()) if has_pm else 0.0

    if has_pm:
        stop_tph  = max(am_stop_total, pm_stop_total)
        pass_tph  = max(am_pass_total, pm_pass_total)
        total_tph = max(am_stop_total + am_pass_total, pm_stop_total + pm_pass_total)
    else:
        stop_tph  = am_stop_total
        pass_tph  = am_pass_total
        total_tph = am_stop_total + am_pass_total

    directions = set(am_stop) | set(am_pass) | set(pm_stop) | set(pm_pass)
    stop_dir_values: List[float] = []
    pass_dir_values: List[float] = []
    combined_dir_values: List[float] = []
    for d in directions:
        am_s = am_stop.get(d, 0.0)
        am_p = am_pass.get(d, 0.0)
        stop_dir_values.append(am_s)
        pass_dir_values.append(am_p)
        combined_dir_values.append(am_s + am_p)
        if has_pm:
            pm_s = pm_stop.get(d, 0.0)
            pm_p = pm_pass.get(d, 0.0)
            stop_dir_values.append(pm_s)
            pass_dir_values.append(pm_p)
            combined_dir_values.append(pm_s + pm_p)

    stop_tphpd  = max(stop_dir_values) if stop_dir_values else 0.0
    pass_tphpd  = max(pass_dir_values) if pass_dir_values else 0.0
    total_tphpd = max(combined_dir_values) if combined_dir_values else 0.0

    return stop_tph, pass_tph, total_tph, stop_tphpd, pass_tphpd, total_tphpd


def aggregate_station_metrics(
    nodes_df: pd.DataFrame,
    service_links_df: pd.DataFrame,
    junction_numbers: set,
    period: str,
) -> pd.DataFrame:
    """Compute station-level capacity inputs for the given service period.

    Args:
        nodes_df: Output of load_infra_data() nodes table.
        service_links_df: Output of load_projected_services().
        junction_numbers: Set of BAV node Numbers classified as junctions.
        period: 'peak' or 'offpeak'.

    Returns:
        DataFrame with NR, Name, Code, Node_Class, E_LV95, N_LV95,
        Track_Count, Platform_Count, stopping_tph/tphpd, passing_tph/tphpd,
        stopping_services, passing_services.

    Notes:
        For 'peak' the AM and PM windows are aggregated independently before
        being collapsed: tph reports the binding window's bidirectional total
        and tphpd reports the binding direction in the binding window. This
        avoids double-counting commuter services that reverse direction
        between AM and PM. Off-peak is collapsed across its single window.
    """
    if period == "peak":
        am_col = "freq_am_peak"
        pm_col = "freq_pm_peak"
        has_pm = True
        active = service_links_df[
            (service_links_df[am_col] > 0) | (service_links_df[pm_col] > 0)
        ].copy()
    else:
        am_col = "freq_offpeak"
        pm_col = None
        has_pm = False
        active = service_links_df[service_links_df[am_col] > 0].copy()

    junction_set: Set[int] = set(int(j) for j in junction_numbers)

    stop_from_mask = active["is_origin"] & ~active["seg_from_node"].isin(junction_set)
    stop_to_mask   = active["is_destination"] & ~active["seg_to_node"].isin(junction_set)
    pass_from_mask = ~stop_from_mask
    pass_to_mask   = ~stop_to_mask

    def _node_contributions(mask_from, mask_to, col: str) -> Dict[int, Dict[str, float]]:
        sub_from = active.loc[mask_from, ["seg_from_node", "direction_id", col]].rename(
            columns={"seg_from_node": "node"}
        )
        sub_to = active.loc[mask_to, ["seg_to_node", "direction_id", col]].rename(
            columns={"seg_to_node": "node"}
        )
        combined = pd.concat([sub_from, sub_to], ignore_index=True)
        if combined.empty:
            return {}
        combined["node"] = combined["node"].astype(int)
        combined["direction_id"] = combined["direction_id"].astype(str)
        grouped = combined.groupby(["node", "direction_id"])[col].sum()
        result: Dict[int, Dict[str, float]] = defaultdict(dict)
        for (node, direction), val in grouped.items():
            result[int(node)][str(direction)] = float(val)
        return result

    am_stop_by_node = _node_contributions(stop_from_mask, stop_to_mask, am_col)
    am_pass_by_node = _node_contributions(pass_from_mask, pass_to_mask, am_col)
    if has_pm:
        pm_stop_by_node = _node_contributions(stop_from_mask, stop_to_mask, pm_col)
        pm_pass_by_node = _node_contributions(pass_from_mask, pass_to_mask, pm_col)
    else:
        pm_stop_by_node = {}
        pm_pass_by_node = {}

    # Service-name maps per node (display, unchanged behaviour).
    stop_svc_map: Dict[int, set] = defaultdict(set)
    for _, row in active[stop_from_mask].iterrows():
        stop_svc_map[int(row["seg_from_node"])].add(str(row["service"]))
    for _, row in active[stop_to_mask].iterrows():
        stop_svc_map[int(row["seg_to_node"])].add(str(row["service"]))

    pass_svc_map: Dict[int, set] = defaultdict(set)
    for _, row in active[pass_from_mask].iterrows():
        pass_svc_map[int(row["seg_from_node"])].add(str(row["service"]))
    for _, row in active[pass_to_mask].iterrows():
        pass_svc_map[int(row["seg_to_node"])].add(str(row["service"]))

    def _node_metrics(node_nr: int) -> Tuple[float, float, float, float]:
        stop_tph, pass_tph, _t_tph, stop_tphpd, pass_tphpd, _t_tphpd = (
            _binding_metrics_from_grouped(
                am_stop_by_node.get(node_nr, {}),
                am_pass_by_node.get(node_nr, {}),
                pm_stop_by_node.get(node_nr, {}),
                pm_pass_by_node.get(node_nr, {}),
                has_pm,
            )
        )
        return stop_tph, pass_tph, stop_tphpd, pass_tphpd

    result = nodes_df.copy()
    metrics = {int(nr): _node_metrics(int(nr)) for nr in result["NR"]}
    result["stopping_tph"]   = result["NR"].map(lambda n: metrics[int(n)][0]).fillna(0.0)
    result["passing_tph"]    = result["NR"].map(lambda n: metrics[int(n)][1]).fillna(0.0)
    result["stopping_tphpd"] = result["NR"].map(lambda n: metrics[int(n)][2]).fillna(0.0)
    result["passing_tphpd"]  = result["NR"].map(lambda n: metrics[int(n)][3]).fillna(0.0)

    result["stopping_services"] = result["NR"].map(
        {k: ", ".join(sorted(v)) for k, v in stop_svc_map.items()}
    ).fillna("")
    result["passing_services"]  = result["NR"].map(
        {k: ", ".join(sorted(v)) for k, v in pass_svc_map.items()}
    ).fillna("")

    out_cols = [
        "NR", "Name", "Code", "Node_Class", "E_LV95", "N_LV95",
        "Track_Count", "Platform_Count",
        "stopping_tph", "stopping_tphpd", "passing_tph", "passing_tphpd",
        "stopping_services", "passing_services",
    ]
    available = [c for c in out_cols if c in result.columns]
    return result[available].sort_values("NR").reset_index(drop=True)


def build_stop_lookup(service_links_df: pd.DataFrame, period: str) -> set:
    """Return (service, direction_id, node_id) tuples where a service stops.

    A node is a stop if it is the origin (is_origin=True) or destination
    (is_destination=True) of a link with non-zero frequency in the given period.

    Args:
        service_links_df: Output of load_projected_services().
        period: 'peak' uses freq_peak; 'offpeak' uses freq_offpeak.
    """
    freq_col = "freq_peak" if period == "peak" else "freq_offpeak"
    active = service_links_df[service_links_df[freq_col] > 0]
    stops: set = set()
    for _, row in active[active["is_origin"]].iterrows():
        stops.add((str(row["service"]), str(row["direction_id"]), int(row["seg_from_node"])))
    for _, row in active[active["is_destination"]].iterrows():
        stops.add((str(row["service"]), str(row["direction_id"]), int(row["seg_to_node"])))
    return stops


def aggregate_segment_metrics(
    segments_df: pd.DataFrame,
    service_links_df: pd.DataFrame,
    stop_lookup: set,
    junction_numbers: set,
    period: str,
) -> pd.DataFrame:
    """Compute segment-level capacity inputs for the given service period.

    Args:
        segments_df: Output of load_infra_data() segments table.
        service_links_df: Output of load_projected_services().
        stop_lookup: Output of build_stop_lookup() for the given period.
        junction_numbers: Set of BAV node Numbers classified as junctions.
        period: 'peak' or 'offpeak'.

    Returns:
        DataFrame with infra attributes merged with service frequencies.

    Notes:
        For 'peak' the AM and PM windows are aggregated independently before
        being collapsed. tph reports the binding window's bidirectional total
        and tphpd reports the binding direction in the binding window (max
        over peak window × direction). total_* considers stop+pass jointly
        within each (window, direction) bucket so segments are not
        double-counted when stopping and passing services peak in different
        windows.
    """
    if period == "peak":
        am_col = "freq_am_peak"
        pm_col = "freq_pm_peak"
        peak_col = "freq_peak"
        has_pm = True
        active = service_links_df[
            (service_links_df[am_col] > 0) | (service_links_df[pm_col] > 0)
        ].copy()
    else:
        am_col = "freq_offpeak"
        pm_col = None
        peak_col = "freq_offpeak"
        has_pm = False
        active = service_links_df[service_links_df[am_col] > 0].copy()

    def _is_stopping(row) -> bool:
        # Junction endpoints are transparent infrastructure — only station
        # endpoints can disqualify a service from "stopping". A service counts
        # as stopping on the segment if it has a real stop at every station
        # endpoint; junctions are treated as if they weren't there. This means
        # services are only classified as passing/express when they actually
        # pass a station without stopping (not merely when they route through
        # a junction).
        fn = int(row["seg_from_node"])
        tn = int(row["seg_to_node"])
        svc = str(row["service"])
        did = str(row["direction_id"])
        fn_ok = fn in junction_numbers or (svc, did, fn) in stop_lookup
        tn_ok = tn in junction_numbers or (svc, did, tn) in stop_lookup
        return fn_ok and tn_ok

    active["_stopping"] = active.apply(_is_stopping, axis=1)
    active["_key"] = active.apply(
        lambda r: tuple(sorted((int(r["seg_from_node"]), int(r["seg_to_node"])))), axis=1
    )
    active["_direction"] = active["direction_id"].astype(str)

    def _segment_contributions(is_stop: bool, col: str) -> Dict[tuple, Dict[str, float]]:
        subset = active[active["_stopping"] == is_stop]
        if subset.empty:
            return {}
        grouped = subset.groupby(["_key", "_direction"])[col].sum()
        result: Dict[tuple, Dict[str, float]] = defaultdict(dict)
        for (key, direction), val in grouped.items():
            result[key][str(direction)] = float(val)
        return result

    am_stop_by_seg = _segment_contributions(True, am_col)
    am_pass_by_seg = _segment_contributions(False, am_col)
    if has_pm:
        pm_stop_by_seg = _segment_contributions(True, pm_col)
        pm_pass_by_seg = _segment_contributions(False, pm_col)
    else:
        pm_stop_by_seg = {}
        pm_pass_by_seg = {}

    # Per-service display maps preserve current behaviour using max-of-AM/PM
    # per row (collapsed into _freq_peak / _freq_offpeak).
    svc_totals:     Dict[tuple, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    svc_dir_totals: Dict[tuple, Dict[tuple, float]] = defaultdict(lambda: defaultdict(float))
    stop_svc_seg:   Dict[tuple, set] = defaultdict(set)
    pass_svc_seg:   Dict[tuple, set] = defaultdict(set)

    for _, row in active.iterrows():
        key = row["_key"]
        svc = str(row["service"])
        did = str(row["direction_id"])
        freq = float(row[peak_col])
        svc_totals[key][svc] += freq
        svc_dir_totals[key][(svc, did)] += freq
        if row["_stopping"]:
            stop_svc_seg[key].add(svc)
        else:
            pass_svc_seg[key].add(svc)

    records = []
    for _, seg in segments_df.iterrows():
        fn  = int(seg["from_node"])
        tn  = int(seg["to_node"])
        key = tuple(sorted((fn, tn)))

        stop_tph, pass_tph, total_tph, stop_tphpd, pass_tphpd, total_tphpd = (
            _binding_metrics_from_grouped(
                am_stop_by_seg.get(key, {}),
                am_pass_by_seg.get(key, {}),
                pm_stop_by_seg.get(key, {}),
                pm_pass_by_seg.get(key, {}),
                has_pm,
            )
        )

        records.append({
            "from_node":        fn,
            "to_node":          tn,
            "from_name":        seg.get("from_name", ""),
            "to_name":          seg.get("to_name", ""),
            "Num_Tracks":       seg.get("Num_Tracks"),
            "Length":           seg.get("Length"),
            "TT_Stopping":      seg.get("TT_Stopping"),
            "TT_Passing":       seg.get("TT_Passing"),
            "Average_Speed":    seg.get("Average_Speed"),
            "stopping_tph":     stop_tph,
            "passing_tph":      pass_tph,
            "total_tph":        total_tph,
            "stopping_tphpd":   stop_tphpd,
            "passing_tphpd":    pass_tphpd,
            "total_tphpd":      total_tphpd,
            "services_tph":     _format_service_frequency_map(dict(svc_totals.get(key, {}))),
            "services_tphpd":   _format_service_direction_frequency_map(dict(svc_dir_totals.get(key, {}))),
            "stopping_services": ", ".join(sorted(stop_svc_seg.get(key, set()))),
            "passing_services":  ", ".join(sorted(pass_svc_seg.get(key, set()))),
        })

    result = pd.DataFrame(records)
    return result.sort_values(["from_node", "to_node"]).reset_index(drop=True)


def _derive_baseline_prep_path(*args, **kwargs):
    raise RuntimeError("_derive_baseline_prep_path() is retired. Use load_infra_data() instead.")

def build_capacity_tables(
    infra_version: str,
    svc_version: str,
    network_label: str,
    output_dir: Optional[Path] = None,
) -> tuple:
    """Build station and segment capacity tables for peak and off-peak periods.

    Args:
        infra_version: Named infra version (e.g. 'AS_2026_ZH_enhanced').
        svc_version:   Service version (e.g. 'AK_2026').
        network_label: Label for the output workbook filename.
        output_dir:    Directory to write the capacity workbook. When None, defaults
                       to CAPACITY_ROOT / 'Baseline' / safe_label.

    Returns:
        (stations_peak, segments_peak, stations_offpeak, segments_offpeak, junction_numbers)
    """
    print("  Loading infrastructure data ...")
    nodes_df, segments_df, _n2nr, _nr2n = load_infra_data(infra_version)

    print("  Loading projected services ...")
    service_links_df = load_projected_services(svc_version, infra_version)

    junction_numbers: set = set(
        nodes_df.loc[nodes_df["Node_Class"] == "junction", "NR"].astype(int)
    )
    print(f"  Junction nodes: {len(junction_numbers)}")

    results = {}
    for period in ("peak", "offpeak"):
        print(f"  Building {period} tables ...")
        sl = build_stop_lookup(service_links_df, period)
        st = aggregate_station_metrics(nodes_df, service_links_df, junction_numbers, period)
        sg = aggregate_segment_metrics(segments_df, service_links_df, sl, junction_numbers, period)
        results[period] = (st, sg)

    stations_peak,    segments_peak    = results["peak"]
    stations_offpeak, segments_offpeak = results["offpeak"]

    safe_label = re.sub(r"[^\w-]+", "_", network_label).strip("_") or "capacity"
    out_dir = output_dir if output_dir is not None else CAPACITY_ROOT / "Baseline" / safe_label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"capacity_{safe_label}.xlsx"
    with pd.ExcelWriter(out_path, engine=EXCEL_ENGINE) as writer:
        stations_peak.to_excel(writer,    sheet_name="Stations_Peak",    index=False)
        segments_peak.to_excel(writer,    sheet_name="Segments_Peak",    index=False)
        stations_offpeak.to_excel(writer, sheet_name="Stations_Offpeak", index=False)
        segments_offpeak.to_excel(writer, sheet_name="Segments_Offpeak", index=False)
    print(f"  Capacity workbook -> {out_path}")

    return stations_peak, segments_peak, stations_offpeak, segments_offpeak, junction_numbers


def _derive_prep_path(output_path: Path) -> Path:
    """Return the expected path of the manually enriched workbook."""
    return output_path.with_name(f"{output_path.stem}_prep{output_path.suffix}")


def _derive_sections_path(output_path: Path) -> Path:
    """Return the path for the exported sections workbook."""
    return output_path.with_name(f"{output_path.stem}_sections{output_path.suffix}")


def _post_export_capacity_processing(output_path: Path) -> None:
    """Prompt for manual enrichment and, if ready, export the Sections workbook."""
    print(
        "\nPlease add the remaining station/segment inputs (tracks, platforms, length, "
        "speed, passing time) to the exported capacity workbook before continuing."
    )
    response = input("Have you added the missing data (y/n)? ").strip().lower()
    if response not in {"y", "yes"}:
        print("Skipping section aggregation. Re-run after updating the workbook.")
        return

    prep_path = _derive_prep_path(output_path)
    if not prep_path.exists():
        print(f"Expected manual workbook at {prep_path}. Please save your edits there and rerun.")
        return

    if APPEND_ENGINE is None:
        print(
            "The 'openpyxl' package is required to read the manual workbook and export sections. "
            "Install it and rerun the script to generate the Sections workbook."
        )
        return

    try:
        stations_df = pd.read_excel(prep_path, sheet_name="Stations")
        segments_df = pd.read_excel(prep_path, sheet_name="Segments")
    except ValueError as exc:
        print(f"Failed to read required sheets from {prep_path}: {exc}")
        return
    except FileNotFoundError as exc:
        print(f"Unable to open {prep_path}: {exc}")
        return

    sections_df = _build_sections_dataframe(stations_df, segments_df)
    if sections_df.empty:
        print("No sections were identified with the current data. Update the workbook and rerun.")
        return

    float_columns = sections_df.select_dtypes(include=["float"]).columns
    if len(float_columns) > 0:
        sections_df[float_columns] = sections_df[float_columns].round(3)

    sections_path = _derive_sections_path(output_path)
    sections_engine = APPEND_ENGINE or EXCEL_ENGINE
    with pd.ExcelWriter(sections_path, engine=sections_engine) as writer:
        stations_df.to_excel(writer, sheet_name="Stations", index=False)
        segments_df.to_excel(writer, sheet_name="Segments", index=False)
        sections_df.to_excel(writer, sheet_name="Sections", index=False)

    print(f"Sections workbook written to {sections_path}.")


def _build_sections_dataframe(
    stations_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    junction_numbers: Optional[set] = None,
    segments_offpeak_df: Optional[pd.DataFrame] = None,
    compute_capacity: bool = True,
    sa_node_set: Optional[Set[int]] = None,
    grouping_strategy: str = "manual",
) -> pd.DataFrame:
    """Assemble continuous sections that share the same track count.

    Args:
        stations_df: Station table from aggregate_station_metrics().
        segments_df: Segment table from aggregate_segment_metrics() (peak period).
        junction_numbers: Set of BAV node Numbers classified as junctions.
            Degree>=3 junctions always act as boundaries via the degree condition.
            Degree-2 junctions are transparent when both adjacent segments share
            the same Num_Tracks (handled automatically by track-group separation).
        segments_offpeak_df: Optional off-peak segment table. When provided the
            returned DataFrame carries paired _peak / _offpeak columns.
            Capacity_offpeak = Capacity_peak (infrastructure does not change);
            only Utilization_offpeak differs.
        compute_capacity: When False, skips all UIC formula calls for every
            section. Use for pure Set_Value mode to avoid wasted computation.
        sa_node_set: When provided, UIC formulas run only for sections whose
            entire node sequence is within this set; all other sections skip
            the formula (Set_Value is applied by the caller). Overrides
            compute_capacity for per-section decisions.
    """
    _junctions: set = junction_numbers if junction_numbers is not None else set()

    # ── Column aliases: accept both new infrabuild names and old names ────
    stations_df = stations_df.copy()
    segments_df = segments_df.copy()

    if 'tracks' not in stations_df.columns and 'Track_Count' in stations_df.columns:
        stations_df['tracks'] = stations_df['Track_Count']
    if 'NAME' not in stations_df.columns and 'Name' in stations_df.columns:
        stations_df['NAME'] = stations_df['Name']

    _seg_aliases = [
        ('Num_Tracks',    'tracks'),
        ('Length',        'length_m'),
        ('TT_Passing',    'travel_time_passing'),
        ('TT_Stopping',   'travel_time_stopping'),
        ('Average_Speed', 'speed'),
    ]
    for _new_col, _old_col in _seg_aliases:
        if _old_col not in segments_df.columns and _new_col in segments_df.columns:
            segments_df[_old_col] = segments_df[_new_col]

    required_station_cols = {"NR", "tracks"}
    required_segment_cols = {"from_node", "to_node", "tracks"}
    service_columns = {
        "stopping_services",
        "passing_services",
    }

    missing_station = required_station_cols - set(stations_df.columns)
    if missing_station:
        print(
            "Stations sheet is missing required columns: "
            + ", ".join(sorted(missing_station))
        )
        return pd.DataFrame()

    missing_segment = required_segment_cols - set(segments_df.columns)
    if missing_segment:
        print(
            "Segments sheet is missing required columns: "
            + ", ".join(sorted(missing_segment))
        )
        return pd.DataFrame()

    stations_df = stations_df.copy()
    segments_df = segments_df.copy()

    stations_df["NR"] = pd.to_numeric(stations_df["NR"], errors="coerce")
    stations_df = stations_df.dropna(subset=["NR"]).reset_index(drop=True)
    stations_df["NR"] = stations_df["NR"].astype(int)

    node_tracks_series = pd.to_numeric(stations_df.get("tracks"), errors="coerce")
    node_tracks: Dict[int, float] = {}
    for node_id, track_value in zip(stations_df["NR"], node_tracks_series):
        if pd.notna(track_value):
            node_tracks[int(node_id)] = float(track_value)

    node_names = {
        int(row_NR): str(name) if pd.notna(name) else ""
        for row_NR, name in zip(stations_df["NR"], stations_df.get("NAME", ""))
    }

    def _parse_services(cell: str) -> tuple[str, tuple[str, ...]]:
        raw_tokens = [token.strip() for token in re.split(r"[;,]", cell) if token.strip()]
        tokens: List[str] = []
        for token in raw_tokens:
            base = token.split(".")[0].strip()
            if base:
                tokens.append(base)
        if not tokens:
            return "", tuple()
        unique_tokens = sorted(dict.fromkeys(tokens))
        canonical = "; ".join(unique_tokens)
        return canonical, tuple(unique_tokens)

    if "stopping_services" in stations_df.columns:
        station_stop_tokens = (
            stations_df["stopping_services"]
            .fillna("")
            .astype(str)
            .map(lambda cell: _parse_services(cell)[1])
        )
        node_stop_services = {
            int(node_id): set(tokens)
            for node_id, tokens in zip(stations_df["NR"], station_stop_tokens)
        }
    else:
        node_stop_services = {int(node_id): set() for node_id in stations_df["NR"]}

    if "passing_services" in stations_df.columns:
        station_pass_tokens = (
            stations_df["passing_services"]
            .fillna("")
            .astype(str)
            .map(lambda cell: _parse_services(cell)[1])
        )
        node_pass_services = {
            int(node_id): set(tokens)
            for node_id, tokens in zip(stations_df["NR"], station_pass_tokens)
        }
    else:
        node_pass_services = {int(node_id): set() for node_id in stations_df["NR"]}

    segments_df["from_node"] = pd.to_numeric(segments_df["from_node"], errors="coerce")
    segments_df["to_node"] = pd.to_numeric(segments_df["to_node"], errors="coerce")
    segments_df["track_key"] = pd.to_numeric(segments_df["tracks"], errors="coerce")

    segments_df = segments_df.dropna(subset=["from_node", "to_node", "track_key"])
    if segments_df.empty:
        return pd.DataFrame()

    segments_df["from_node"] = segments_df["from_node"].astype(int)
    segments_df["to_node"] = segments_df["to_node"].astype(int)
    segments_df["track_key"] = segments_df["track_key"].astype(float)

    segments_df["length_value"] = pd.to_numeric(segments_df.get("length_m"), errors="coerce").fillna(0.0)
    segments_df["passing_value"] = pd.to_numeric(
        segments_df.get("travel_time_passing"), errors="coerce"
    ).fillna(0.0)
    segments_df["speed_value"] = pd.to_numeric(segments_df.get("speed"), errors="coerce")
    def _coerce_frequency(column: str) -> pd.Series:
        if column in segments_df.columns:
            series = pd.to_numeric(segments_df[column], errors="coerce")
        else:
            series = pd.Series([float("nan")] * len(segments_df), index=segments_df.index, dtype="float64")
        return series.fillna(0.0)

    segments_df["stopping_bidirectional_value"] = _coerce_frequency("stopping_tph")
    segments_df["passing_bidirectional_value"] = _coerce_frequency("passing_tph")
    segments_df["total_bidirectional_value"] = _coerce_frequency("total_tph")
    segments_df["stopping_per_direction_value"] = _coerce_frequency("stopping_tphpd")
    segments_df["passing_per_direction_value"] = _coerce_frequency("passing_tphpd")
    segments_df["total_per_direction_value"] = _coerce_frequency("total_tphpd")
    segments_df["stop_time_value"] = pd.to_numeric(segments_df.get("travel_time_stopping"), errors="coerce").fillna(0.0)

    for column in service_columns:
        if column not in segments_df.columns:
            segments_df[column] = ""
        segments_df[column] = segments_df[column].fillna("").astype(str)
        parsed = segments_df[column].map(_parse_services)
        segments_df[column] = parsed.map(lambda pair: pair[0])
        segments_df[f"{column}_tokens"] = parsed.map(lambda pair: pair[1])

    edges_by_track: Dict[float, Dict[frozenset, Dict[str, float]]] = {}
    adjacency_by_track: Dict[float, defaultdict[int, set[int]]] = {}

    for row in segments_df.itertuples(index=False):
        track = float(row.track_key)
        u = int(row.from_node)
        v = int(row.to_node)
        key = frozenset({u, v})

        stopping_tokens = row.stopping_services_tokens
        if isinstance(stopping_tokens, str):
            stopping_tokens = tuple(token.strip() for token in stopping_tokens.split(";") if token.strip())
        elif isinstance(stopping_tokens, (list, tuple)):
            stopping_tokens = tuple(stopping_tokens)
        else:
            stopping_tokens = tuple()

        passing_tokens = row.passing_services_tokens
        if isinstance(passing_tokens, str):
            passing_tokens = tuple(token.strip() for token in passing_tokens.split(";") if token.strip())
        elif isinstance(passing_tokens, (list, tuple)):
            passing_tokens = tuple(passing_tokens)
        else:
            passing_tokens = tuple()

        via_tokens: tuple[int, ...]
        via_value = getattr(row, "Via", getattr(row, "ViaNodes", []))
        if isinstance(via_value, str):
            via_tokens = tuple(int(token) for token in re.findall(r"\d+", via_value))
        elif isinstance(via_value, (list, tuple)):
            via_tokens = tuple(int(token) for token in via_value)
        else:
            via_tokens = tuple()
        services_tph_cell = str(getattr(row, "services_tph", "") or "")
        services_tphpd_cell = str(getattr(row, "services_tphpd", "") or "")

        edge_info = {
            "from_node": u,
            "to_node": v,
            "length": float(row.length_value),
            "passing_time": float(row.passing_value),
            "stopping_time": float(row.stop_time_value),
            "speed": None if pd.isna(row.speed_value) else float(row.speed_value),
            "stopping_tph": float(row.stopping_bidirectional_value),
            "passing_tph": float(row.passing_bidirectional_value),
            "total_tph": float(row.total_bidirectional_value),
            "stopping_tphpd": float(row.stopping_per_direction_value),
            "passing_tphpd": float(row.passing_per_direction_value),
            "total_tphpd": float(row.total_per_direction_value),
            "stopping_services": row.stopping_services,
            "stopping_service_tokens": stopping_tokens,
            "passing_services": row.passing_services,
            "passing_service_tokens": passing_tokens,
            "services_tph": services_tph_cell,
            "services_tphpd": services_tphpd_cell,
            "services_tph_map": _parse_service_frequency_string(services_tph_cell),
            "services_tphpd_map": _parse_service_direction_frequency_string(services_tphpd_cell),
            "track_count": track,
            "via_nodes": via_tokens,
        }

        track_edges = edges_by_track.setdefault(track, {})
        track_edges[key] = edge_info

        adjacency = adjacency_by_track.setdefault(track, defaultdict(set))
        adjacency[u].add(v)
        adjacency[v].add(u)

    sections: List[Dict[str, object]] = []
    section_counter = 1

    for track, edges_dict in edges_by_track.items():
        adjacency = adjacency_by_track[track]
        visited_edges: set[frozenset] = set()

        def node_valid(node_id: int) -> bool:
            # Junctions transparent to track-count check:
            # degree>=3 already forces boundary via len(adjacency)!=2;
            # degree-2 junctions only split when Num_Tracks differs across them,
            # handled automatically by track-group separation.
            if node_id in _junctions:
                return True
            track_value = node_tracks.get(node_id)
            return track_value == track

        nodes = list(adjacency.keys())
        start_nodes = [node for node in nodes if len(adjacency[node]) != 2 or not node_valid(node)]

        # print(f"\n{'='*80}")
        # print(f"=== PROCESSING TRACK COUNT GROUP: {track} ===")
        # print(f"- Total edges in this group: {len(edges_dict)}")
        # print(f"- Total nodes: {len(adjacency)}")
        # print(f"- Start nodes (branch/terminal points): {start_nodes}")
        # print(f"{'='*80}\n")

        for start in start_nodes:
            start_name = node_names.get(start, f"Node_{start}")
            # print(f"--- Starting path search from node {start} ({start_name}, track={track}) ---")
            # print(f"  Neighbors to explore: {list(adjacency[start])}")

            for neighbor in list(adjacency[start]):
                # ...existing code...

                neighbor_name = node_names.get(neighbor, f"Node_{neighbor}")
                # print(f"  Traversing edge: {start} ({start_name}) -> {neighbor} ({neighbor_name})")

                path_nodes, edge_records, path_edge_keys = _traverse_path(
                    start,
                    neighbor,
                    adjacency,
                    edges_dict,
                    visited_edges,
                    node_valid,
                )
                if edge_records:
                    refined_sections = _split_section_by_service_patterns(
                        path_nodes,
                        edge_records,
                        node_stop_services,
                        node_pass_services,
                    )
                    for refined_nodes, refined_edges in refined_sections:
                        _sec_compute_cap = (
                            all(n in sa_node_set for n in refined_nodes)
                            if sa_node_set is not None else compute_capacity
                        )
                        sections.append(
                            _summarise_section(
                                section_counter,
                                track,
                                refined_nodes,
                                refined_edges,
                                node_names,
                                node_stop_services,
                                node_pass_services,
                                compute_capacity=_sec_compute_cap,
                                grouping_strategy=grouping_strategy,
                            )
                        )
                        section_counter += 1
                    visited_edges.update(path_edge_keys)

        # print(f"\n--- Processing remaining unvisited edges (track={track}) ---")
        for edge_key, edge_info in edges_dict.items():
            if edge_key in visited_edges:
                continue
            u, v = tuple(edge_key)
            u_name = node_names.get(u, f"Node_{u}")
            v_name = node_names.get(v, f"Node_{v}")
            # print(f"  Unvisited edge: {u} ({u_name}) <-> {v} ({v_name})")
            path_nodes, edge_records, path_edge_keys = _traverse_path(
                u,
                v,
                adjacency,
                edges_dict,
                visited_edges,
                node_valid,
            )
            if edge_records:
                refined_sections = _split_section_by_service_patterns(
                    path_nodes,
                    edge_records,
                    node_stop_services,
                    node_pass_services,
                )
                for refined_nodes, refined_edges in refined_sections:
                    _sec_compute_cap = (
                        all(n in sa_node_set for n in refined_nodes)
                        if sa_node_set is not None else compute_capacity
                    )
                    sections.append(
                        _summarise_section(
                            section_counter,
                            track,
                            refined_nodes,
                            refined_edges,
                            node_names,
                            node_stop_services,
                            node_pass_services,
                            compute_capacity=_sec_compute_cap,
                            grouping_strategy=grouping_strategy,
                        )
                    )
                    section_counter += 1
                visited_edges.update(path_edge_keys)

    if not sections:
        return pd.DataFrame()

    df = pd.DataFrame(sections)

    # ── Rename single-period columns to _peak variants ────────────────────────
    _peak_rename = {
        "total_tph":      "total_tph_peak",
        "stopping_tph":   "stopping_tph_peak",
        "passing_tph":    "passing_tph_peak",
        "total_tphpd":    "total_tphpd_peak",
        "stopping_tphpd": "stopping_tphpd_peak",
        "passing_tphpd":  "passing_tphpd_peak",
        "Capacity":       "Capacity_peak",
        "Utilization":    "Utilization_peak",
    }
    df.rename(columns={k: v for k, v in _peak_rename.items() if k in df.columns}, inplace=True)

    # ── Add offpeak columns when offpeak segment data is supplied ─────────────
    if segments_offpeak_df is not None and not segments_offpeak_df.empty:
        _op = segments_offpeak_df.copy()
        for _nc, _oc in _seg_aliases:
            if _oc not in _op.columns and _nc in _op.columns:
                _op[_oc] = _op[_nc]

        _op_lookup: Dict[tuple, Dict] = {}
        for _, _r in _op.iterrows():
            _k = tuple(sorted((int(_r["from_node"]), int(_r["to_node"]))))
            _op_lookup[_k] = {
                "total_tphpd":    float(_r.get("total_tphpd",    0.0) or 0.0),
                "stopping_tphpd": float(_r.get("stopping_tphpd", 0.0) or 0.0),
                "passing_tphpd":  float(_r.get("passing_tphpd",  0.0) or 0.0),
            }

        def _offpeak_for_section(seg_seq: str) -> Dict:
            vals: Dict[str, list] = {"total": [], "stop": [], "pass_": []}
            for seg_str in str(seg_seq).split("|"):
                parts = [p.strip() for p in seg_str.split("-") if p.strip()]
                if len(parts) == 2:
                    try:
                        k = tuple(sorted((int(parts[0]), int(parts[1]))))
                        d = _op_lookup.get(k, {})
                        vals["total"].append(d.get("total_tphpd",    0.0))
                        vals["stop"].append( d.get("stopping_tphpd", 0.0))
                        vals["pass_"].append(d.get("passing_tphpd",  0.0))
                    except ValueError:
                        pass
            return {
                "total_tphpd_offpeak":    max(vals["total"],  default=0.0),
                "stopping_tphpd_offpeak": max(vals["stop"],   default=0.0),
                "passing_tphpd_offpeak":  max(vals["pass_"],  default=0.0),
            }

        _op_rows = df["segment_sequence"].map(_offpeak_for_section)
        _op_frame = pd.DataFrame(list(_op_rows))
        df["total_tphpd_offpeak"]    = _op_frame["total_tphpd_offpeak"].values
        df["stopping_tphpd_offpeak"] = _op_frame["stopping_tphpd_offpeak"].values
        df["passing_tphpd_offpeak"]  = _op_frame["passing_tphpd_offpeak"].values
        # Capacity is infrastructure-limited — same for both periods
        df["Capacity_offpeak"]    = df.get("Capacity_peak", float("nan"))
        df["Utilization_offpeak"] = df.apply(
            lambda r: (
                r["total_tphpd_offpeak"] / r["Capacity_offpeak"]
                if pd.notna(r.get("Capacity_offpeak")) and r.get("Capacity_offpeak", 0) > 0
                else float("nan")
            ),
            axis=1,
        )
    else:
        # No offpeak data: mirror peak as offpeak so callers always get both columns
        for _src, _dst in [
            ("total_tphpd_peak",    "total_tphpd_offpeak"),
            ("stopping_tphpd_peak", "stopping_tphpd_offpeak"),
            ("passing_tphpd_peak",  "passing_tphpd_offpeak"),
            ("Capacity_peak",       "Capacity_offpeak"),
            ("Utilization_peak",    "Utilization_offpeak"),
        ]:
            if _src in df.columns:
                df[_dst] = df[_src]

    return df


def _traverse_path(
    start: int,
    neighbor: int,
    adjacency: Dict[int, set[int]],
    edges_dict: Dict[frozenset, Dict[str, float]],
    visited_edges: set[frozenset],
    node_valid,
) -> Tuple[List[int], List[Tuple[int, int, Dict[str, float]]], List[frozenset]]:
    """Walk a path while track conditions remain satisfied."""
    path_nodes: List[int] = [start]
    edge_records: List[Tuple[int, int, Dict[str, float]]] = []
    path_edge_keys: List[frozenset] = []
    current = start
    next_node = neighbor
    local_edges: set[frozenset] = set()

    while True:
        # print(f"    Edge step: {current} -> {next_node}")
        edge_key = frozenset({current, next_node})
        if edge_key in visited_edges or edge_key in local_edges:
            # print(f"    STOP: Edge already visited/processed")
            break
        edge_info = edges_dict.get(edge_key)
        if edge_info is None:
            # print(f"    STOP: Edge not found in edges_dict")
            break

        local_edges.add(edge_key)
        path_edge_keys.append(edge_key)
        edge_records.append((current, next_node, edge_info))
        path_nodes.append(next_node)

        if not node_valid(next_node) or len(adjacency[next_node]) != 2:
            node_track = edges_dict.get(edge_key, {}).get("track_count", "?")
            # print(f"    STOP: Terminal/branch point - node {next_node} (valid={node_valid(next_node)}, neighbors={len(adjacency[next_node])}, track={node_track})")
            break

        candidates = adjacency[next_node] - {current}
        if not candidates:
            # print(f"    STOP: No forward candidates from node {next_node}")
            break
        candidate = next(iter(candidates))
        candidate_edge = frozenset({next_node, candidate})
        if candidate_edge in visited_edges or candidate_edge in local_edges:
            # print(f"    STOP: Next edge {next_node} -> {candidate} already visited")
            break

        # Stop section if the next edge changes track or service patterns.
        current_edge_info = edges_dict.get(edge_key)
        candidate_edge_info = edges_dict.get(candidate_edge)
        if candidate_edge_info is None:
            # print(f"    STOP: Next edge {next_node} -> {candidate} not found in edges_dict")
            break
        if current_edge_info["track_count"] != candidate_edge_info["track_count"]:
            # print(f"    STOP: Track count change - current={current_edge_info['track_count']}, next={candidate_edge_info['track_count']}")
            break
        if current_edge_info["stopping_service_tokens"] != candidate_edge_info["stopping_service_tokens"]:
            # print(f"    STOP: Stopping services differ - current={current_edge_info['stopping_service_tokens']}, next={candidate_edge_info['stopping_service_tokens']}")
            break
        if current_edge_info["passing_service_tokens"] != candidate_edge_info["passing_service_tokens"]:
            # print(f"    STOP: Passing services differ - current={current_edge_info['passing_service_tokens']}, next={candidate_edge_info['passing_service_tokens']}")
            break

        current, next_node = next_node, candidate

    # print(f"    Path complete: {len(path_nodes)} nodes, {len(edge_records)} edges")
    # print(f"    Node sequence: {path_nodes}")
    return path_nodes, edge_records, path_edge_keys


def _classify_service_pattern(
    service: str,
    path_nodes: List[int],
    node_stop_services: Dict[int, set[str]]
) -> str:
    """
    Classify a service's stopping pattern for a given path.

    Args:
        service: Service identifier (e.g., "S14", "IC1")
        path_nodes: List of node IDs representing the path
        node_stop_services: Dict mapping node ID to set of services that stop there

    Returns:
        Pattern type: "ALL-STOP" | "ENDS-ONLY" | "PARTIAL" | "ABSENT"

    Pattern Definitions:
        - ALL-STOP: Service stops at all nodes in the path
        - ENDS-ONLY: Service stops only at first and last nodes (requires 3+ nodes)
        - PARTIAL: Service stops at some nodes (not all, not just ends)
        - ABSENT: Service does not stop at any node in the path
    """
    if len(path_nodes) < 2:
        return "ABSENT"

    # Identify which nodes the service stops at
    stops_at = [node for node in path_nodes if service in node_stop_services.get(node, set())]

    stops_count = len(stops_at)
    total_nodes = len(path_nodes)

    # Classify pattern
    if stops_count == 0:
        return "ABSENT"
    elif stops_count == total_nodes:
        return "ALL-STOP"
    elif stops_count == 2 and total_nodes >= 3:
        # Check if stops only at first and last
        if stops_at == [path_nodes[0], path_nodes[-1]]:
            return "ENDS-ONLY"
        else:
            return "PARTIAL"
    else:
        # Stops at some but not all nodes
        return "PARTIAL"


def _classify_all_service_patterns(
    path_nodes: List[int],
    all_services: set[str],
    node_stop_services: Dict[int, set[str]]
) -> Dict[str, str]:
    """
    Classify patterns for all services operating on a path.

    Args:
        path_nodes: List of node IDs representing the path
        all_services: Set of all service identifiers to classify
        node_stop_services: Dict mapping node ID to set of services that stop there

    Returns:
        Dictionary mapping service name to pattern type
        Example: {"S14": "ALL-STOP", "G": "ENDS-ONLY", "S15": "ENDS-ONLY"}
    """
    patterns = {}
    for service in all_services:
        patterns[service] = _classify_service_pattern(service, path_nodes, node_stop_services)
    return patterns


def _patterns_are_compatible(
    patterns_current: Dict[str, str],
    patterns_extended: Dict[str, str]
) -> Tuple[bool, List[str]]:
    """
    Check if two pattern dictionaries are compatible.

    Patterns are compatible if all services maintain their pattern type
    when the path is extended.

    Args:
        patterns_current: Pattern classifications for current path
        patterns_extended: Pattern classifications for extended path

    Returns:
        Tuple of (compatible: bool, changed_services: List[str])

    Examples:
        - ALL-STOP -> ALL-STOP: Compatible
        - ENDS-ONLY -> ENDS-ONLY: Compatible
        - ALL-STOP -> ENDS-ONLY: Incompatible (service started skipping nodes)
        - ENDS-ONLY -> ALL-STOP: Incompatible (service started stopping at middle nodes)
    """
    changed_services = []

    # Get union of all services (current + extended)
    all_services = set(patterns_current.keys()) | set(patterns_extended.keys())

    for service in all_services:
        current_pattern = patterns_current.get(service, "ABSENT")
        extended_pattern = patterns_extended.get(service, "ABSENT")

        # Check if pattern changed
        if current_pattern != extended_pattern:
            changed_services.append(service)

    # Compatible if no services changed pattern
    compatible = len(changed_services) == 0

    return compatible, changed_services


def _split_section_by_service_patterns(
    path_nodes: List[int],
    edge_records: List[Tuple[int, int, Dict[str, float]]],
    node_stop_services: Dict[int, set[str]],
    node_pass_services: Dict[int, set[str]],
) -> List[Tuple[List[int], List[Tuple[int, int, Dict[str, float]]]]]:
    """
    Split an infrastructure section where service patterns become inconsistent.

    This function implements pattern-based section splitting. Sections are split
    when a service's stopping pattern would change if the path were extended.

    Pattern Types:
        - ALL-STOP: Service stops at all stations
        - ENDS-ONLY: Service stops only at section endpoints
        - PARTIAL: Service stops at some intermediate stations
        - ABSENT: Service does not operate on this section

    Algorithm:
        1. Build path incrementally, adding one node at a time
        2. For paths with 2 nodes: Cannot classify patterns yet, continue
        3. For paths with 3+ nodes:
           - Classify all service patterns for current path
           - Try extending by one node
           - Re-classify all service patterns for extended path
           - If any service pattern changes: SPLIT at current last node
           - Otherwise: Continue extending

    Args:
        path_nodes: List of node IDs representing the infrastructure path
        edge_records: List of edge tuples (from_node, to_node, edge_info)
        node_stop_services: Dict mapping node ID to set of services that stop there
        node_pass_services: Dict mapping node ID to set of services that pass there

    Returns:
        List of refined sections as (nodes, edges) tuples

    Example:
        Path: [Uster, Aathal, Wetzikon]

        At [Uster, Aathal]:
            Too few nodes, continue

        At [Uster, Aathal, Wetzikon]:
            S14: stops at all 3 -> ALL-STOP
            G: stops at [Uster, Wetzikon] -> ENDS-ONLY
            All patterns consistent -> ONE section
    """
    # Handle trivial cases
    if len(path_nodes) <= 2 or not edge_records:
        return [(path_nodes, edge_records)]

    # Collect all services operating on this path
    all_services: set[str] = set()
    for node in path_nodes:
        all_services.update(node_stop_services.get(node, set()))
        all_services.update(node_pass_services.get(node, set()))

    if not all_services:
        return [(path_nodes, edge_records)]

    # Build sections incrementally with pattern checking
    sections: List[Tuple[List[int], List[Tuple[int, int, Dict[str, float]]]]] = []
    current_start_idx = 0
    current_patterns: Optional[Dict[str, str]] = None

    # Start from index 2 (3rd node) because we need 3+ nodes to classify patterns
    for idx in range(2, len(path_nodes)):
        # Current path up to and including idx
        current_path = path_nodes[0:idx+1]

        # Classify patterns for current path
        patterns = _classify_all_service_patterns(current_path, all_services, node_stop_services)

        if current_patterns is None:
            # First time classifying patterns (have 3+ nodes now)
            current_patterns = patterns
            continue

        # Check if patterns are compatible with previous
        compatible, changed_services = _patterns_are_compatible(current_patterns, patterns)

        if not compatible:
            # Pattern break detected - SPLIT before current node
            split_idx = idx - 1  # Index of last node before split

            # Create section from start to split point
            section_nodes = path_nodes[current_start_idx : split_idx + 1]
            section_edge_count = split_idx - current_start_idx
            section_edges = edge_records[current_start_idx : current_start_idx + section_edge_count]

            if section_edges:
                sections.append((section_nodes, section_edges))

            # Start new section at split node
            current_start_idx = split_idx
            current_patterns = None  # Reset for new section

            # Re-evaluate from this point (don't advance idx yet)
            # On next iteration, we'll re-classify starting from split_idx

    # Add final section (from current_start_idx to end)
    final_nodes = path_nodes[current_start_idx:]
    final_edge_count = len(path_nodes) - 1 - current_start_idx
    final_edges = edge_records[current_start_idx : current_start_idx + final_edge_count]

    if final_edges or len(final_nodes) > 1:
        sections.append((final_nodes, final_edges))

    # Return sections or original if no splits
    return sections if sections else [(path_nodes, edge_records)]


def _calculate_single_passing_track_capacity(
    total_passing_time: float,
    headway: float
) -> float:
    """Calculate capacity for single passing track.

    Args:
        total_passing_time: Total travel time for passing trains (minutes)
        headway: Minimum headway between trains (minutes)

    Returns:
        Capacity in tphpd (trains per hour per direction)
    """
    if total_passing_time <= 0:
        return float("nan")
    raw_capacity = 60.0 / total_passing_time
    return raw_capacity / 2.0  # Convert to per-direction


def _calculate_double_track_good_capacity(
    headway: float,
    travel_time_penalty: float,
    service_count: int,
    n_stop: int,
    n_pass: int
) -> float:
    """Calculate double-track capacity using 'good' strategy.

    Args:
        headway: Base headway (minutes)
        travel_time_penalty: Additional time penalty for mixed traffic
        service_count: Total number of distinct services
        n_stop: Number of stopping services
        n_pass: Number of passing services

    Returns:
        Capacity in tphpd
    """
    if service_count <= 0:
        return float("nan")

    # Calculate pattern changes for "good" strategy
    def _strategy_pattern_changes(strategy: str, stops: int, passes: int) -> int:
        if stops <= 0 or passes <= 0:
            return 0
        if strategy == "good":
            return 1
        return 0

    pattern_changes = _strategy_pattern_changes("good", n_stop, n_pass)
    feasible_changes = min(pattern_changes, max(service_count - 1, 0))
    denominator = headway + (feasible_changes / service_count) * travel_time_penalty

    if denominator <= 0:
        return float("nan")

    return 60.0 / denominator


def _calculate_uniform_double_track_capacity(headway: float) -> float:
    """Calculate double-track capacity for uniform traffic.

    Args:
        headway: Base headway (minutes)

    Returns:
        Capacity in tphpd
    """
    if headway <= 0:
        return float("nan")
    return 60.0 / headway


def _calculate_3_track_capacity(
    headway: float,
    travel_time_penalty: float,
    service_count: int,
    n_stop: int,
    n_pass: int,
    total_passing_time: float,
    has_stopping: bool,
    has_passing: bool
) -> float:
    """Calculate 3-track capacity: 2-track (good) + 1 passing track.

    Args:
        headway: Base headway (minutes)
        travel_time_penalty: Additional time for mixed traffic
        service_count: Total number of services
        n_stop: Number of stopping services
        n_pass: Number of passing services
        total_passing_time: Total passing travel time
        has_stopping: Whether stopping services exist
        has_passing: Whether passing services exist

    Returns:
        Capacity in tphpd
    """
    if has_stopping and has_passing:
        # Mixed traffic: 2-track good + 1 passing
        capacity_double = _calculate_double_track_good_capacity(
            headway, travel_time_penalty, service_count, n_stop, n_pass
        )
        capacity_single = _calculate_single_passing_track_capacity(
            total_passing_time, headway
        )
        return capacity_double + capacity_single
    else:
        # Uniform traffic: 1.5x double-track
        return 1.5 * _calculate_uniform_double_track_capacity(headway)


def _calculate_4_track_capacity(
    section_id: int,
    start_node: int,
    end_node: int,
    headway: float,
    travel_time_penalty: float,
    service_count: int,
    n_stop: int,
    n_pass: int,
    stopping_tphpd: float,
    passing_tphpd: float,
    has_stopping: bool,
    has_passing: bool
) -> float:
    """Calculate 4-track capacity: Separated pairs with overflow handling.

    Logic:
    - Try to separate stopping and passing onto dedicated pairs (2 tracks each)
    - If one service overflows, prompt user for strategy selection
    - If both overflow, keep homogeneous (indicates over-capacity)

    Args:
        section_id: Section identifier for user prompt
        start_node: Starting node ID
        end_node: Ending node ID
        headway: Base headway (minutes)
        travel_time_penalty: Additional time for mixed traffic
        service_count: Total number of services
        n_stop: Number of stopping services
        n_pass: Number of passing services
        stopping_tphpd: Current stopping train demand
        passing_tphpd: Current passing train demand
        has_stopping: Whether stopping services exist
        has_passing: Whether passing services exist

    Returns:
        Capacity in tphpd
    """
    if has_stopping and has_passing:
        # Calculate dedicated pair capacities (homogeneous)
        capacity_per_pair = _calculate_uniform_double_track_capacity(headway)

        stopping_overflow = stopping_tphpd > capacity_per_pair
        passing_overflow = passing_tphpd > capacity_per_pair

        if stopping_overflow and passing_overflow:
            # Both overflow: Keep homogeneous (2 pairs, will show over-capacity)
            return 2 * capacity_per_pair

        elif stopping_overflow:
            # Stopping overflows: Prompt user for strategy
            print(f"\n[4-TRACK OVERFLOW] Section {section_id} ({start_node}->{end_node})")
            print(f"  Stopping services: {stopping_tphpd:.1f} tphpd (exceeds {capacity_per_pair:.1f} tphpd capacity)")
            print(f"  Passing services: {passing_tphpd:.1f} tphpd (within capacity)")
            print("\nSelect track allocation strategy:")
            print("  1) Stopping-Stopping + Mixed (overflow stopping + all passing on 2nd pair)")
            print("  2) Keep homogeneous (2 stopping pairs, will show over-capacity)")

            while True:
                response = input("Enter choice (1-2): ").strip()
                if response == "1":
                    # Pair 1: Stopping at capacity, Pair 2: Mixed
                    pair1_capacity = capacity_per_pair
                    pair2_capacity = _calculate_double_track_good_capacity(
                        headway, travel_time_penalty, service_count, n_stop, n_pass
                    )
                    return pair1_capacity + pair2_capacity
                elif response == "2":
                    # Keep homogeneous
                    return 2 * capacity_per_pair
                print("Invalid selection. Please enter 1 or 2.")

        elif passing_overflow:
            # Passing overflows: Prompt user for strategy
            print(f"\n[4-TRACK OVERFLOW] Section {section_id} ({start_node}->{end_node})")
            print(f"  Stopping services: {stopping_tphpd:.1f} tphpd (within capacity)")
            print(f"  Passing services: {passing_tphpd:.1f} tphpd (exceeds {capacity_per_pair:.1f} tphpd capacity)")
            print("\nSelect track allocation strategy:")
            print("  1) Passing-Passing + Mixed (overflow passing + all stopping on 2nd pair)")
            print("  2) Keep homogeneous (2 passing pairs, will show over-capacity)")

            while True:
                response = input("Enter choice (1-2): ").strip()
                if response == "1":
                    # Pair 1: Passing at capacity, Pair 2: Mixed
                    pair1_capacity = capacity_per_pair
                    pair2_capacity = _calculate_double_track_good_capacity(
                        headway, travel_time_penalty, service_count, n_stop, n_pass
                    )
                    return pair1_capacity + pair2_capacity
                elif response == "2":
                    # Keep homogeneous
                    return 2 * capacity_per_pair
                print("Invalid selection. Please enter 1 or 2.")

        else:
            # No overflow: Perfect separation (2 homogeneous pairs)
            return 2 * capacity_per_pair
    else:
        # Uniform traffic: 2x double-track
        return 2.0 * _calculate_uniform_double_track_capacity(headway)


def _calculate_multi_track_capacity(
    section_id: int,
    start_node: int,
    end_node: int,
    formula_track: int,
    headway: float,
    travel_time_penalty: float,
    service_count: int,
    n_stop: int,
    n_pass: int,
    total_passing_time: float,
    stopping_tphpd: float,
    passing_tphpd: float,
    has_stopping: bool,
    has_passing: bool
) -> float:
    """Calculate capacity for 5+ tracks using recursive building blocks.

    Logic:
    - Base: 4-track capacity
    - Remaining tracks: Allocate based on count
      - +1: Single passing track
      - +2: Pair allocated to service with higher demand
      - +3: Allocated pair + passing track
      - +4+: Recursively add another 4-track block

    Args:
        section_id: Section identifier for user prompts
        start_node: Starting node ID
        end_node: Ending node ID
        formula_track: Number of tracks (5+)
        [other args same as helper functions]

    Returns:
        Capacity in tphpd
    """
    # Base: 4-track capacity
    base_capacity = _calculate_4_track_capacity(
        section_id, start_node, end_node,
        headway, travel_time_penalty, service_count, n_stop, n_pass,
        stopping_tphpd, passing_tphpd, has_stopping, has_passing
    )

    remaining_tracks = formula_track - 4

    if remaining_tracks <= 0:
        additional = 0.0

    elif remaining_tracks == 1:
        # +1 passing track
        additional = _calculate_single_passing_track_capacity(
            total_passing_time, headway
        )

    elif remaining_tracks == 2:
        # +1 pair allocated to service with higher demand
        if has_stopping and has_passing:
            # Allocate to whichever has more demand
            additional = _calculate_uniform_double_track_capacity(headway)
        else:
            # Uniform traffic
            additional = _calculate_uniform_double_track_capacity(headway)

    elif remaining_tracks == 3:
        # +1 allocated pair + 1 passing track
        pair_capacity = _calculate_uniform_double_track_capacity(headway)
        passing_capacity = _calculate_single_passing_track_capacity(
            total_passing_time, headway
        )
        additional = pair_capacity + passing_capacity

    else:  # remaining_tracks >= 4
        # Recursively add another 4-track block
        additional = _calculate_multi_track_capacity(
            section_id, start_node, end_node,
            remaining_tracks,
            headway, travel_time_penalty, service_count, n_stop, n_pass,
            total_passing_time, stopping_tphpd, passing_tphpd,
            has_stopping, has_passing
        )

    return base_capacity + additional


def _summarise_section(
    section_id: int,
    track: float,
    path_nodes: List[int],
    edge_records: List[Tuple[int, int, Dict[str, float]]],
    node_names: Dict[int, str],
    node_stop_services: Dict[int, set[str]],
    node_pass_services: Dict[int, set[str]],
    compute_capacity: bool = True,
    grouping_strategy: str = "manual",
) -> Dict[str, object]:
    """Combine edge metrics into a section summary.

    Args:
        compute_capacity: When False, skips all UIC capacity formula calls and
            returns NaN for Capacity and Utilization. Use for Set_Value mode to
            avoid unnecessary computation.
        grouping_strategy: How to resolve ambiguous 2-track groupings when
            multiple options exist: 'manual' prompts interactively,
            'conservative' picks the lowest capacity option, 'baseline' picks
            the middle option, 'optimal' picks the highest capacity option.
    """
    start_node = path_nodes[0]
    end_node = path_nodes[-1]
    start_name = node_names.get(start_node, f"Node_{start_node}")
    end_name = node_names.get(end_node, f"Node_{end_node}")

    # print(f"\n  === SUMMARIZING SECTION {section_id} ===")
    # print(f"  Track: {track}")
    # print(f"  Nodes: {path_nodes}")
    # print(f"  From: {start_node} ({start_name}) -> To: {end_node} ({end_name})")
    # print(f"  Edges: {len(edge_records)}")

    def _collect_unique_numeric(field: str) -> List[float]:
        unique: set[float] = set()
        for _, _, edge_info in edge_records:
            value = edge_info.get(field)
            if value is None:
                continue
            numeric = float(value)
            if math.isnan(numeric):
                continue
            unique.add(numeric)
        return sorted(unique)

    def _collect_frequency_map(field: str) -> Dict[object, float]:
        aggregated: Dict[object, float] = {}
        for _, _, edge_info in edge_records:
            freq_map = edge_info.get(field)
            if not isinstance(freq_map, dict):
                continue
            for key, value in freq_map.items():
                numeric = float(value)
                if math.isnan(numeric):
                    continue
                aggregated[key] = max(aggregated.get(key, 0.0), numeric)
        return aggregated

    def _floor_capacity(value: float) -> float:
        if value is None:
            return float("nan")
        if math.isnan(value) or value <= 0:
            return float("nan")
        return float(math.floor(value))

    def _strategy_pattern_changes(strategy: str, stops: int, passes: int) -> int:
        if stops <= 0 or passes <= 0:
            return 0
        if strategy == "bad":
            return min(stops, passes)
        if strategy == "base":
            return max(min(stops, passes) - 1, 0)
        if strategy == "good":
            return 1
        return 0

    total_length = 0.0
    passing_time_values: List[float] = []
    total_stopping_time = 0.0
    for _, _, edge_info in edge_records:
        length = float(edge_info["length"])
        total_length += length

        raw_passing_time = edge_info.get("passing_time")
        passing_time = None
        if raw_passing_time is not None:
            passing_time = float(raw_passing_time)
            if math.isnan(passing_time):
                passing_time = None
        if (passing_time is None or passing_time <= 0) and edge_info.get("speed") not in (None, 0, float("nan")):
            passing_time = (length / 1000.0) / float(edge_info["speed"]) * 60.0
        if passing_time is not None and not math.isnan(passing_time):
            passing_time_values.append(passing_time)

        total_stopping_time += float(edge_info["stopping_time"])

    total_passing_time = sum(passing_time_values)

    stopping_tph_values = _collect_unique_numeric("stopping_tph")
    passing_tph_values = _collect_unique_numeric("passing_tph")
    total_tph_values = _collect_unique_numeric("total_tph")
    stopping_tphpd_values = _collect_unique_numeric("stopping_tphpd")
    passing_tphpd_values = _collect_unique_numeric("passing_tphpd")
    total_tphpd_values = _collect_unique_numeric("total_tphpd")
    services_tph_map = _collect_frequency_map("services_tph_map")
    services_tphpd_map = _collect_frequency_map("services_tphpd_map")

    start_node = path_nodes[0]
    end_node = path_nodes[-1]

    node_stop_seq = [node_stop_services.get(node, set()) for node in path_nodes]
    node_pass_seq = [node_pass_services.get(node, set()) for node in path_nodes]
    candidate_services = sorted(set().union(*node_stop_seq, *node_pass_seq))

    stopping_services: List[str] = []
    passing_services_list: List[str] = []
    for service in candidate_services:
        present_all = all(
            (service in stop_set) or (service in pass_set)
            for stop_set, pass_set in zip(node_stop_seq, node_pass_seq)
        )
        if not present_all:
            continue
        stops_all = all(service in stop_set for stop_set in node_stop_seq)
        stops_some = any(service in stop_set for stop_set in node_stop_seq)
        passes_some = any(service in pass_set for pass_set in node_pass_seq)
        if stops_all:
            stopping_services.append(service)
        elif passes_some or not stops_some:
            passing_services_list.append(service)

    n_stop = len(stopping_services)
    n_pass = len(passing_services_list)
    all_services = sorted(set(stopping_services) | set(passing_services_list))
    service_count = len(all_services)

    stopping_tph_value = (
        stopping_tph_values[0] if len(stopping_tph_values) == 1 else float("nan")
    )
    passing_tph_value = (
        passing_tph_values[0] if len(passing_tph_values) == 1 else float("nan")
    )
    total_tph_value = (
        total_tph_values[0] if len(total_tph_values) == 1 else float("nan")
    )
    stopping_tphpd_value = (
        stopping_tphpd_values[0] if len(stopping_tphpd_values) == 1 else float("nan")
    )
    passing_tphpd_value = (
        passing_tphpd_values[0] if len(passing_tphpd_values) == 1 else float("nan")
    )
    total_tphpd_value = (
        total_tphpd_values[0] if len(total_tphpd_values) == 1 else float("nan")
    )
    services_tph_value = _format_service_frequency_map(services_tph_map)
    services_tphpd_value = _format_service_direction_frequency_map(services_tphpd_map)

    stopping_tph_estimate = (
        stopping_tph_values[0] if stopping_tph_values else float("nan")
    )
    passing_tph_estimate = (
        passing_tph_values[0] if passing_tph_values else float("nan")
    )
    stopping_tphpd_estimate = (
        stopping_tphpd_values[0] if stopping_tphpd_values else float("nan")
    )
    passing_tphpd_estimate = (
        passing_tphpd_values[0] if passing_tphpd_values else float("nan")
    )

    bidirectional_estimates = [
        estimate for estimate in (stopping_tph_estimate, passing_tph_estimate) if not math.isnan(estimate)
    ]
    per_direction_estimates = [
        estimate for estimate in (stopping_tphpd_estimate, passing_tphpd_estimate) if not math.isnan(estimate)
    ]

    total_tph = float(sum(bidirectional_estimates)) if bidirectional_estimates else float("nan")
    total_tphpd = float(sum(per_direction_estimates)) if per_direction_estimates else float("nan")
    has_bidirectional_data = len(bidirectional_estimates) > 0
    has_per_direction_data = len(per_direction_estimates) > 0

    def _utilization(capacity_value: float, demand: float) -> float:
        if capacity_value is None or math.isnan(capacity_value) or capacity_value <= 0:
            return float("nan")
        if demand is None or math.isnan(demand):
            return float("nan")
        return demand / capacity_value

    # Default NaN capacity columns — always present in return dict regardless of compute_capacity.
    capacity_columns = {
        "capacity_single_track_tphpd": float("nan"),
        "capacity_uniform_pattern_tphpd": float("nan"),
        "capacity_bad_tphpd": float("nan"),
        "capacity_base_tphpd": float("nan"),
        "capacity_good_tphpd": float("nan"),
        "pattern_changes_bad": float("nan"),
        "pattern_changes_base": float("nan"),
        "pattern_changes_good": float("nan"),
        "utilization_single_track": float("nan"),
        "utilization_uniform_pattern": float("nan"),
        "utilization_bad": float("nan"),
        "utilization_base": float("nan"),
        "utilization_good": float("nan"),
    }
    selected_capacity    = float("nan")
    selected_utilization = float("nan")

    if compute_capacity:
        headway = DEFAULT_HEADWAY_MIN
        travel_time_penalty = max(0.0, float(total_stopping_time) - float(total_passing_time) - headway)
        strategy_metrics: List[Tuple[str, float, float]] = []

        # print(f"  Capacity calculation inputs:")
        # print(f"    Total length: {total_length}m")
        # print(f"    Stopping time: {total_stopping_time}min")
        # print(f"    Passing time: {total_passing_time}min")
        # print(f"    Headway: {headway}min")
        # print(f"    Service count: {service_count}")
        # print(f"    All services: {all_services}")
        # print(f"    Stopping services: {stopping_services}")
        # print(f"    Passing services: {passing_services_list}")

        # Fractional track support: .5 increments halve section travel times
        is_fractional = (track % 1 == 0.5)  # True for 1.5, 2.5, 3.5, 4.5, 5.5, etc.
        base_track = math.floor(track)  # 1.5→1, 2.5→2, 3.5→3, 4.5→4, etc.

        if is_fractional:
            # Halve travel times to simulate section_length_m / 2
            total_stopping_time = total_stopping_time / 2.0
            total_passing_time = total_passing_time / 2.0
            travel_time_penalty = max(0.0, total_stopping_time - total_passing_time - headway)
            formula_track = base_track
            # print(f"    Fractional track ({track}): Using formula_track={formula_track}, times halved")
        else:
            formula_track = int(track)
            # print(f"    Track formula: formula_track={formula_track} (is_fractional={is_fractional})")

        if formula_track == 1:
            single_capacity = float("nan")
            if total_stopping_time > 0:
                # Floor the per-direction value (not the bidirectional), so odd
                # bidirectional capacities round down conservatively rather than
                # leaving a .5 train slot.
                bidirectional_capacity = 60.0 / float(total_stopping_time)
                single_capacity = _floor_capacity(bidirectional_capacity / 2.0)
            capacity_columns["capacity_single_track_tphpd"] = single_capacity
            demand_single_track = (
                total_tphpd if not math.isnan(total_tphpd) else (total_tph / 2.0 if not math.isnan(total_tph) else float("nan"))
            )
            capacity_columns["utilization_single_track"] = _utilization(single_capacity, demand_single_track)
        elif formula_track == 2:
            if not stopping_services or not passing_services_list:
                uniform_capacity = _floor_capacity(60.0 / headway)
                capacity_columns["capacity_uniform_pattern_tphpd"] = uniform_capacity
                capacity_columns["utilization_uniform_pattern"] = _utilization(
                    uniform_capacity, total_tphpd
                )
            elif service_count > 0:
                strategy_definitions: List[str] = []
                if service_count >= 6:
                    strategy_definitions = ["bad", "base", "good"]
                elif service_count >= 4:
                    strategy_definitions = ["bad", "good"]
                elif service_count >= 2:
                    strategy_definitions = ["bad"]
                for strategy_key in strategy_definitions:
                    pattern_changes = _strategy_pattern_changes(strategy_key, n_stop, n_pass)
                    feasible_changes = min(pattern_changes, max(service_count - 1, 0))
                    denominator = headway + (feasible_changes / service_count) * travel_time_penalty
                    capacity_value = _floor_capacity(60.0 / denominator) if denominator > 0 else float("nan")
                    capacity_columns[f"capacity_{strategy_key}_tphpd"] = capacity_value
                    capacity_columns[f"pattern_changes_{strategy_key}"] = float(feasible_changes)
                    capacity_columns[f"utilization_{strategy_key}"] = _utilization(
                        capacity_value, total_tphpd
                    )
                    strategy_metrics.append(
                        (
                            strategy_key,
                            capacity_columns[f"capacity_{strategy_key}_tphpd"],
                            capacity_columns[f"utilization_{strategy_key}"],
                        )
                    )
        selected_capacity = float("nan")
        selected_utilization = float("nan")

        # Helper variables for 3+ track calculations
        has_stopping = bool(stopping_services)
        has_passing = bool(passing_services_list)

        if formula_track == 1:
            selected_capacity = capacity_columns["capacity_single_track_tphpd"]
            selected_utilization = capacity_columns["utilization_single_track"]
        elif formula_track == 2:
            if not stopping_services or not passing_services_list:
                selected_capacity = capacity_columns["capacity_uniform_pattern_tphpd"]
                selected_utilization = capacity_columns["utilization_uniform_pattern"]
            elif strategy_metrics:
                if len(strategy_metrics) == 1:
                    _, cap_value, util_value = strategy_metrics[0]
                    selected_capacity = cap_value
                    selected_utilization = util_value
                else:
                    all_caps = [m[1] for m in strategy_metrics]
                    if len(set(all_caps)) == 1:
                        _, cap_value, util_value = strategy_metrics[0]
                        selected_capacity = cap_value
                        selected_utilization = util_value
                    elif grouping_strategy == "conservative":
                        _, cap_value, util_value = min(strategy_metrics, key=lambda m: m[1])
                        selected_capacity = cap_value
                        selected_utilization = util_value
                    elif grouping_strategy == "optimal":
                        _, cap_value, util_value = max(strategy_metrics, key=lambda m: m[1])
                        selected_capacity = cap_value
                        selected_utilization = util_value
                    elif grouping_strategy == "baseline":
                        _, cap_value, util_value = strategy_metrics[len(strategy_metrics) // 2]
                        selected_capacity = cap_value
                        selected_utilization = util_value
                    else:  # manual
                        print(
                            f"\nSection {section_id} ({start_node}->{end_node}) offers multiple capacity groupings."
                        )
                        print("Available options:")
                        for idx, (strategy_key, cap_value, util_value) in enumerate(strategy_metrics, start=1):
                            label = strategy_key.capitalize()
                            cap_display = "n/a" if cap_value is None or math.isnan(cap_value) else str(cap_value)
                            util_display = "n/a" if util_value is None or math.isnan(util_value) else f"{util_value:.3f}"
                            print(f"  {idx}) {label} (capacity={cap_display}, utilization={util_display})")
                        while True:
                            response = input("Select the strategy number to apply (press Enter to skip): ").strip()
                            if response == "":
                                print("No strategy selected; leaving Capacity/Utilization empty for this section.")
                                break
                            if response.isdigit():
                                choice = int(response)
                                if 1 <= choice <= len(strategy_metrics):
                                    _, cap_value, util_value = strategy_metrics[choice - 1]
                                    selected_capacity = cap_value
                                    selected_utilization = util_value
                                    break
                            print("Invalid selection. Please enter a listed number or press Enter to skip.")
        elif formula_track == 3:
            # THREE TRACKS: 2-track (good) + 1 passing track
            capacity_value = _calculate_3_track_capacity(
                headway, travel_time_penalty, service_count, n_stop, n_pass,
                total_passing_time, has_stopping, has_passing
            )
            selected_capacity = _floor_capacity(capacity_value)
            capacity_columns["capacity_good_tphpd"] = selected_capacity
            selected_utilization = _utilization(selected_capacity, total_tphpd)
            capacity_columns["utilization_good"] = selected_utilization
        elif formula_track == 4:
            # FOUR TRACKS: Separated pairs with overflow handling
            capacity_value = _calculate_4_track_capacity(
                section_id, start_node, end_node,
                headway, travel_time_penalty, service_count, n_stop, n_pass,
                stopping_tphpd_estimate, passing_tphpd_estimate,
                has_stopping, has_passing
            )
            selected_capacity = _floor_capacity(capacity_value)
            capacity_columns["capacity_good_tphpd"] = selected_capacity
            selected_utilization = _utilization(selected_capacity, total_tphpd)
            capacity_columns["utilization_good"] = selected_utilization
        else:  # formula_track >= 5
            # MULTI-TRACK: Recursive building blocks (4-track base + additions)
            capacity_value = _calculate_multi_track_capacity(
                section_id, start_node, end_node,
                formula_track, headway, travel_time_penalty, service_count,
                n_stop, n_pass, total_passing_time,
                stopping_tphpd_estimate, passing_tphpd_estimate,
                has_stopping, has_passing
            )
            selected_capacity = _floor_capacity(capacity_value)
            capacity_columns["capacity_good_tphpd"] = selected_capacity
            selected_utilization = _utilization(selected_capacity, total_tphpd)
            capacity_columns["utilization_good"] = selected_utilization

    # print(f"  Section {section_id} complete:")
    # print(f"    Route: {start_node} ({start_name}) -> {end_node} ({end_name})")
    # print(f"    Selected Capacity: {selected_capacity}")
    # print(f"    Selected Utilization: {selected_utilization}")

    return {
        "section_id": section_id,
        "track_count": track,
        "start_node": start_node,
        "start_station": node_names.get(start_node, ""),
        "end_node": end_node,
        "end_station": node_names.get(end_node, ""),
        "node_sequence": " -> ".join(str(node) for node in path_nodes),
        "segment_sequence": " | ".join(f"{u}-{v}" for u, v, _ in edge_records),
        "segment_count": len(edge_records),
        "total_length_m": total_length,
        "total_travel_time_passing_min": total_passing_time,
        "total_travel_time_stopping_min": total_stopping_time,
        "stopping_tph": stopping_tph_value,
        "passing_tph": passing_tph_value,
        "total_tph": total_tph_value,
        "stopping_tphpd": stopping_tphpd_value,
        "passing_tphpd": passing_tphpd_value,
        "total_tphpd": total_tphpd_value,
        "distinct_service_count": service_count,
        "stopping_services": ", ".join(stopping_services),
        "passing_services": ", ".join(passing_services_list),
        "all_services": ", ".join(all_services),
        "services_tph": services_tph_value,
        "services_tphpd": services_tphpd_value,
        "Capacity": selected_capacity,
        "Utilization": selected_utilization,
        **capacity_columns,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def export_capacity_workbook(
    edges_path: Path = None,
    network_label: str = None,
    enrichment_source: Path = None,
    output_dir: Path = None,
    skip_manual_checkpoint: bool = False,
) -> Path:
    """Build the capacity workbook and return the output path.

    Args:
        edges_path: Optional custom path to edges file.
                   - If None: BASELINE workflow (uses edges_in_corridor.gpkg + master points)
                   - If provided: DEVELOPMENT workflow (uses custom edges + edge-derived stations)
        network_label: Optional custom network label for output naming.
                      For developments, should include dev ID (e.g., "AK_2035_dev_100023")
        enrichment_source: Optional path to baseline prep workbook for auto-enrichment.
                          - If None: generates empty workbook for manual enrichment (baseline workflow)
                          - If provided: inherits baseline data and applies defaults (development workflow)
        output_dir: Optional custom output directory.
                   - If None: auto-detected (baseline or development based on network_label)
                   - If provided: uses directory as-is
        skip_manual_checkpoint: If True, skips manual enrichment prompt and sections export.
                               Used for development workflow with auto-enrichment.

    Returns:
        Path to the exported capacity workbook.
    """
    is_baseline = edges_path is None

    # Auto-detect output directory for developments
    if output_dir is None and not is_baseline and network_label is not None:
        # Extract dev ID from network_label (e.g., "AK_2035_dev_100023" → "100023")
        import re
        dev_match = re.search(r'_dev_(\d+)', network_label)
        if dev_match:
            dev_id = dev_match.group(1)
            output_dir = CAPACITY_ROOT / "Developments" / dev_id
            print(f"[INFO] Auto-detected development output directory: {output_dir}")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        ensure_output_directory()

    station_metrics, segment_metrics = build_capacity_tables(
        edges_path=edges_path,
        network_label=network_label,
        enrichment_source=enrichment_source,
    )

    output_path = capacity_output_path(network_label=network_label, output_dir=output_dir)
    prep_path = _derive_prep_path(output_path)

    # Save initial workbook
    with pd.ExcelWriter(output_path, engine=EXCEL_ENGINE) as writer:
        station_metrics.to_excel(writer, sheet_name="Stations", index=False)
        segment_metrics.to_excel(writer, sheet_name="Segments", index=False)

    print(f"[INFO] Capacity workbook written to {output_path}")

    # Check if manual enrichment is needed
    has_na_tracks_stations = station_metrics["tracks"].isna().any()
    has_na_platforms = station_metrics["platforms"].isna().any()
    has_na_tracks_segments = segment_metrics["tracks"].isna().any()
    has_na_speed = segment_metrics["speed"].isna().any()

    needs_manual_enrichment = has_na_tracks_stations or has_na_platforms or has_na_tracks_segments or has_na_speed

    if needs_manual_enrichment and not skip_manual_checkpoint:
        # Prompt user to fill missing values
        print("\n" + "="*80)
        print("MANUAL ENRICHMENT REQUIRED")
        print("="*80)
        if has_na_tracks_stations or has_na_platforms:
            print(f"  - {station_metrics['tracks'].isna().sum()} stations missing 'tracks'")
            print(f"  - {station_metrics['platforms'].isna().sum()} stations missing 'platforms'")
        if has_na_tracks_segments or has_na_speed:
            print(f"  - {segment_metrics['tracks'].isna().sum()} segments missing 'tracks'")
            print(f"  - {segment_metrics['speed'].isna().sum()} segments missing 'speed'")
        print(f"\nPlease do the following:")
        print(f"  1. Open the raw workbook in Excel: {output_path}")
        print(f"  2. Fill all NA values for tracks, platforms, speed, length_m")
        print(f"  3. Save the file as: {prep_path}")
        print(f"  4. Return here and confirm completion")
        print("="*80)

        response = input("\nHave you filled the missing data and saved as *_prep.xlsx (y/n)? ").strip().lower()
        if response not in {"y", "yes"}:
            print("Skipping section calculation. Re-run after updating the workbook.")
            return output_path

        # Check if prep workbook exists
        if not prep_path.exists():
            print(f"\n[ERROR] Prep workbook not found at: {prep_path}")
            print(f"Please save your enriched workbook as {prep_path.name} and re-run.")
            return output_path

        # Reload enriched data from prep workbook
        print(f"[INFO] Reloading enriched data from {prep_path}...")
        station_metrics = pd.read_excel(prep_path, sheet_name="Stations")
        segment_metrics = pd.read_excel(prep_path, sheet_name="Segments")

    elif needs_manual_enrichment and skip_manual_checkpoint:
        # Skip manual enrichment but still save prep workbook (only if it doesn't exist)
        import shutil
        if not prep_path.exists():
            shutil.copy2(output_path, prep_path)
            print(f"[INFO] Prep workbook saved to {prep_path} (manual enrichment skipped)")
        else:
            print(f"[INFO] Prep workbook already exists, not overwriting: {prep_path}")
        return output_path
    else:
        # No manual enrichment needed: save prep workbook for sections calculation (only if doesn't exist)
        import shutil
        if not prep_path.exists():
            shutil.copy2(output_path, prep_path)
            print(f"[INFO] Prep workbook saved to {prep_path}")
        else:
            print(f"[INFO] Using existing prep workbook: {prep_path}")

    # Calculate and export sections
    try:
        print(f"[INFO] Calculating sections...")
        sections_df = _build_sections_dataframe(station_metrics, segment_metrics)
        if not sections_df.empty:
            # Round float columns
            float_columns = sections_df.select_dtypes(include=["float"]).columns
            if len(float_columns) > 0:
                sections_df[float_columns] = sections_df[float_columns].round(3)

            # Export sections workbook
            sections_path = _derive_sections_path(output_path)
            sections_engine = APPEND_ENGINE or EXCEL_ENGINE
            with pd.ExcelWriter(sections_path, engine=sections_engine) as writer:
                station_metrics.to_excel(writer, sheet_name="Stations", index=False)
                segment_metrics.to_excel(writer, sheet_name="Segments", index=False)
                sections_df.to_excel(writer, sheet_name="Sections", index=False)

            print(f"[INFO] Sections workbook written to {sections_path}")
        else:
            print("[WARNING] No sections could be identified from the enriched data")
    except Exception as e:
        print(f"[WARNING] Could not calculate sections: {e}")

    return output_path
