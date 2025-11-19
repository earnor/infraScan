"""Capacity calculator for the processed baseline rail network.

This module prepares two worksheets:
* Stations: node metadata and aggregated stopping / passing service frequencies.
* Segments: bidirectional rail link statistics with combined frequencies.

The script assumes that the baseline (status-quo) network has already been
processed and stored in ``data/Network/processed``. Run this module after the
main infrastructure generation pipeline to ensure the inputs exist.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math
import re
from typing import Dict, List, Set, Tuple

import geopandas as gpd
import pandas as pd

import paths
import settings

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

DATA_ROOT = Path(paths.MAIN) / "data" / "Network"
PROCESSED_ROOT = DATA_ROOT / "processed"
CAPACITY_ROOT = DATA_ROOT / "capacity"

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

EDGES_IN_CORRIDOR_PATH = PROCESSED_ROOT / "edges_in_corridor.gpkg"
EDGES_ON_BORDER_PATH = PROCESSED_ROOT / "edges_on_corridor_border.gpkg"
MASTER_POINTS_PATH = PROCESSED_ROOT / "points.gpkg"
CORRIDOR_POINTS_PATH = PROCESSED_ROOT / "points_corridor.gpkg"

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
    """Format frequency values without trailing zeroes."""
    if value is None:
        return ""
    numeric = float(value)
    if math.isnan(numeric):
        return ""
    rounded = round(numeric)
    if math.isclose(numeric, rounded, abs_tol=1e-6):
        return str(int(rounded))
    return f"{numeric:.3f}".rstrip("0").rstrip(".")


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


def extract_stations_from_edges(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Derive station points from edge endpoints when no separate points file exists.

    Used for development networks where only edges are available.
    Returns a GeoDataFrame with columns: ID_point, NAME, CODE, XKOORD, YKOORD, geometry.
    Extracts station names and codes from edge attributes.
    """
    from shapely.geometry import Point

    # DEBUG: Print available columns in edges
    print(f"\n[DEBUG] Edges columns available: {list(edges_gdf.columns)}")

    # Collect unique endpoints with their names and codes
    node_data: Dict[int, Dict[str, any]] = {}

    # Try different possible column name variations
    name_columns = ["FromStation", "FromName", "From_Station", "From_Name", "from_station", "from_name"]
    code_columns = ["FromCode", "From_Code", "from_code", "FromStationCode", "from_station_code"]

    # DEBUG: Check which name/code columns exist
    existing_name_cols = [col for col in name_columns if col in edges_gdf.columns]
    existing_code_cols = [col for col in code_columns if col in edges_gdf.columns]
    print(f"[DEBUG] Found name columns: {existing_name_cols}")
    print(f"[DEBUG] Found code columns: {existing_code_cols}")

    for _, row in edges_gdf.iterrows():
        from_node = parse_int(row.get("FromNode", 0))
        to_node = parse_int(row.get("ToNode", 0))

        # Extract FromNode data
        if from_node and from_node not in node_data and hasattr(row.geometry, 'coords'):
            coords = list(row.geometry.coords)
            if coords:
                # Try to find name in various column names
                from_name = None
                for col in name_columns:
                    if col in row.index and row.get(col):
                        from_name = row.get(col)
                        break
                if not from_name:
                    from_name = f"Node_{from_node}"

                # Try to find code in various column names
                from_code = None
                for col in code_columns:
                    if col in row.index and row.get(col):
                        from_code = row.get(col)
                        break
                if not from_code:
                    from_code = str(from_node)

                node_data[from_node] = {
                    "coords": coords[0],
                    "name": from_name,
                    "code": from_code
                }

        # Extract ToNode data
        if to_node and to_node not in node_data and hasattr(row.geometry, 'coords'):
            coords = list(row.geometry.coords)
            if coords:
                # Try to find name (replace "From" with "To" in column names)
                to_name = None
                to_name_columns = [col.replace("From", "To") for col in name_columns]
                for col in to_name_columns:
                    if col in row.index and row.get(col):
                        to_name = row.get(col)
                        break
                if not to_name:
                    to_name = f"Node_{to_node}"

                # Try to find code
                to_code = None
                to_code_columns = [col.replace("From", "To") for col in code_columns]
                for col in to_code_columns:
                    if col in row.index and row.get(col):
                        to_code = row.get(col)
                        break
                if not to_code:
                    to_code = str(to_node)

                node_data[to_node] = {
                    "coords": coords[-1],
                    "name": to_name,
                    "code": to_code
                }

    # Build points GeoDataFrame
    records = []
    sample_count = 0
    for node_id, data in node_data.items():
        x, y = data["coords"]
        # Convert from LV95 to offset coordinates
        x_offset = x - LV95_E_OFFSET
        y_offset = y - LV95_N_OFFSET

        # DEBUG: Print first 3 stations to verify extraction
        if sample_count < 3:
            print(f"[DEBUG] Station {node_id}: name='{data['name']}', code='{data['code']}'")
            sample_count += 1

        records.append({
            "ID_point": node_id,
            "NAME": str(data["name"]) if data["name"] else f"Node_{node_id}",
            "CODE": str(data["code"]) if data["code"] else str(node_id),
            "XKOORD": x_offset,
            "YKOORD": y_offset,
            "geometry": Point(x, y),
        })

    points_gdf = gpd.GeoDataFrame(records, crs=edges_gdf.crs)
    print(f"[DEBUG] Extracted {len(points_gdf)} unique stations from edges\n")
    return points_gdf


def load_service_links(edges_path: Path = None) -> pd.DataFrame:
    """Load service link records from the processed corridor edges GeoPackage.

    Args:
        edges_path: Optional custom path to edges file. If None, uses default baseline path(s).
                   Both standard and extended baseline modes use edges_in_corridor.gpkg.

    Returns:
        DataFrame with service link records.
    """
    # BASELINE MODE: Load from default path(s)
    if edges_path is None:
        # Check if extended mode based on settings
        is_extended = str(getattr(settings, "rail_network", "")).endswith("_extended")

        # Load main corridor edges (used for both standard and extended modes)
        if not EDGES_IN_CORRIDOR_PATH.exists():
            raise FileNotFoundError(
                f"Processed corridor edges not found at {EDGES_IN_CORRIDOR_PATH}."
            )

        gdf = gpd.read_file(EDGES_IN_CORRIDOR_PATH)

        if is_extended:
            print(f"[INFO] Baseline extended mode: loaded {len(gdf)} edges from edges_in_corridor.gpkg")
        else:
            print(f"[INFO] Baseline standard mode: loaded {len(gdf)} edges from edges_in_corridor.gpkg")

        print(f"[INFO] Total edges for baseline: {len(gdf)}")

    # DEVELOPMENT MODE: Load from custom path
    else:
        if not edges_path.exists():
            raise FileNotFoundError(
                f"Development edges not found at {edges_path}."
            )

        gdf = gpd.read_file(edges_path)
        print(f"[INFO] Loaded {len(gdf)} edges from {edges_path.name}")

    geometry_columns = [col for col in ("geom", "geometry") if col in gdf.columns]
    df = pd.DataFrame(gdf.drop(columns=geometry_columns, errors="ignore"))

    df["FromNode"] = df["FromNode"].apply(parse_int)
    df["ToNode"] = df["ToNode"].apply(parse_int)
    df["Frequency"] = df["Frequency"].apply(parse_float)
    df["TravelTime"] = df["TravelTime"].apply(parse_float)
    df["ViaNodes"] = df["Via"].apply(extract_via_nodes)
    df["Service"] = df["Service"].astype(str)
    df["Direction"] = df["Direction"].astype(str)
    df["FromEndFlag"] = df["FromEnd"].apply(parse_bool_flag)
    df["ToEndFlag"] = df["ToEnd"].apply(parse_bool_flag)
    return df

def apply_enrichment(
    stations_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    baseline_prep: Path,
    edges_gdf: gpd.GeoDataFrame,
    new_station_ids: Set[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply enrichment from baseline prep workbook to development network data.

    Inherits infrastructure attributes from baseline for existing nodes/segments.
    For NEW stations: sets tracks/platforms to NA (requires manual enrichment).
    For NEW segments: calculates length from geometry, sets tracks/speed to NA.

    Args:
        stations_df: Raw station metrics (tracks/platforms = NA)
        segments_df: Raw segment metrics (tracks/speed/length = NA)
        baseline_prep: Path to manually enriched baseline workbook
        edges_gdf: Original edges GeoDataFrame for geometry lookups
        new_station_ids: Set of station IDs that are NEW (not in points.gpkg)

    Returns:
        Enriched (stations_df, segments_df) with infrastructure attributes filled where possible.
    """
    if new_station_ids is None:
        new_station_ids = set()

    if not baseline_prep.exists():
        raise FileNotFoundError(f"Baseline prep workbook not found: {baseline_prep}")

    # Load baseline enrichment data
    baseline_stations = pd.read_excel(baseline_prep, sheet_name="Stations")
    baseline_segments = pd.read_excel(baseline_prep, sheet_name="Segments")

    # Create lookup maps for baseline data
    station_lookup = {}
    for _, row in baseline_stations.iterrows():
        node_id = int(row["NR"])
        station_lookup[node_id] = {
            "tracks": row.get("tracks"),
            "platforms": row.get("platforms"),
        }

    segment_lookup = {}
    for _, row in baseline_segments.iterrows():
        from_node = int(row["from_node"])
        to_node = int(row["to_node"])
        pair = tuple(sorted((from_node, to_node)))
        segment_lookup[pair] = {
            "tracks": row.get("tracks"),
            "speed": row.get("speed"),
            "length_m": row.get("length_m"),
        }

    # Create geometry lookup for segments (need to calculate length)
    geometry_lookup = {}
    for _, row in edges_gdf.iterrows():
        from_node = parse_int(row.get("FromNode", 0))
        to_node = parse_int(row.get("ToNode", 0))
        if from_node and to_node:
            pair = tuple(sorted((from_node, to_node)))
            if hasattr(row.geometry, 'length'):
                geometry_lookup[pair] = row.geometry.length

    # Enrich stations
    enriched_stations = stations_df.copy()
    new_station_count = 0

    for idx, row in enriched_stations.iterrows():
        node_id = int(row["NR"])
        if node_id in station_lookup:
            # Existing station: inherit from baseline
            baseline_data = station_lookup[node_id]
            enriched_stations.at[idx, "tracks"] = baseline_data["tracks"]
            enriched_stations.at[idx, "platforms"] = baseline_data["platforms"]
        elif node_id in new_station_ids:
            # NEW station: set to NA (requires manual enrichment)
            enriched_stations.at[idx, "tracks"] = pd.NA
            enriched_stations.at[idx, "platforms"] = pd.NA
            new_station_count += 1
        else:
            # Should not happen if logic is correct
            enriched_stations.at[idx, "tracks"] = pd.NA
            enriched_stations.at[idx, "platforms"] = pd.NA

    if new_station_count > 0:
        print(f"[INFO] {new_station_count} NEW stations require manual enrichment (tracks/platforms = NA)")

    # Enrich segments
    enriched_segments = segments_df.copy()
    new_segment_count = 0

    for idx, row in enriched_segments.iterrows():
        from_node = int(row["from_node"])
        to_node = int(row["to_node"])
        pair = tuple(sorted((from_node, to_node)))

        # Check if segment has any new endpoints
        has_new_endpoint = (from_node in new_station_ids) or (to_node in new_station_ids)

        if pair in segment_lookup and not has_new_endpoint:
            # Existing segment with no new endpoints: inherit from baseline
            baseline_data = segment_lookup[pair]
            enriched_segments.at[idx, "tracks"] = baseline_data["tracks"]
            enriched_segments.at[idx, "speed"] = baseline_data["speed"]
            enriched_segments.at[idx, "length_m"] = baseline_data["length_m"]
        else:
            # NEW segment or segment with new endpoint: set to NA, calculate length
            enriched_segments.at[idx, "tracks"] = pd.NA
            enriched_segments.at[idx, "speed"] = pd.NA
            new_segment_count += 1

            # Calculate length from geometry
            if pair in geometry_lookup:
                enriched_segments.at[idx, "length_m"] = geometry_lookup[pair]
            else:
                # Fallback: try reverse pair
                reverse_pair = (pair[1], pair[0])
                if reverse_pair in geometry_lookup:
                    enriched_segments.at[idx, "length_m"] = geometry_lookup[reverse_pair]
                else:
                    enriched_segments.at[idx, "length_m"] = pd.NA

    if new_segment_count > 0:
        print(f"[INFO] {new_segment_count} NEW or modified segments require manual enrichment (tracks/speed = NA)")

    return enriched_stations, enriched_segments


def load_corridor_nodes_from_master(
    edges_gdf: gpd.GeoDataFrame,
    master_points_path: Path = None,
    baseline_prep_path: Path = None,
    is_development: bool = False,
) -> Tuple[gpd.GeoDataFrame, Set[int]]:
    """Load stations from points file, filtered to nodes present in edges and baseline.

    This function is used for BOTH baseline and development workflows.
    For baseline standard: Uses points_corridor.gpkg (stations within corridor boundary).
    For baseline extended: Uses points.gpkg (all stations in edges, no corridor filtering).
    For development: Uses points.gpkg, filters to baseline prep NR column + flags new stations.

    Args:
        edges_gdf: Edges GeoDataFrame containing FromNode and ToNode columns.
        master_points_path: Path to points file. If None, auto-detects based on mode:
                           - Baseline standard: points_corridor.gpkg
                           - Baseline extended: points.gpkg
                           - Development: points.gpkg
        baseline_prep_path: Path to baseline prep workbook (required for development workflow).
        is_development: If True, applies development filtering logic.

    Returns:
        Tuple of:
        - GeoDataFrame with station points (columns: ID_point, NAME, CODE, XKOORD, YKOORD, geometry).
        - Set of new station IDs (empty for baseline, contains IDs not in points.gpkg for development)

    Raises:
        FileNotFoundError: If points file doesn't exist.
        ValueError: If baseline edge references corridor node not found in points_corridor.gpkg.
    """
    if master_points_path is None:
        # Check if extended mode based on settings
        is_extended = str(getattr(settings, "rail_network", "")).endswith("_extended")

        if is_development:
            # DEVELOPMENT: Use full master points
            master_points_path = MASTER_POINTS_PATH
        elif is_extended:
            # BASELINE EXTENDED: Use full master points (no corridor filtering)
            master_points_path = MASTER_POINTS_PATH
        else:
            # BASELINE STANDARD: Use corridor points only
            master_points_path = CORRIDOR_POINTS_PATH

    if not master_points_path.exists():
        raise FileNotFoundError(
            f"Points file not found at {master_points_path}. "
            f"Please ensure the file exists."
        )

    # Extract unique node IDs from edges (FromNode, ToNode, Via)
    node_ids = set()
    for _, row in edges_gdf.iterrows():
        from_node = parse_int(row.get("FromNode", 0))
        to_node = parse_int(row.get("ToNode", 0))

        # Parse Via column directly here (edges_gdf doesn't have ViaNodes yet)
        via_value = row.get("Via", "")
        via_nodes = extract_via_nodes(via_value)

        if from_node:
            node_ids.add(from_node)
        if to_node:
            node_ids.add(to_node)

        # Add Via nodes
        if via_nodes:
            for via_node in via_nodes:
                if via_node and via_node != -99:
                    node_ids.add(via_node)

    print(f"[INFO] Extracted {len(node_ids)} unique node IDs from edges (FromNode, ToNode, Via)")

    # Load points file
    master_points = gpd.read_file(master_points_path)

    # Check if extended mode based on settings
    is_extended = str(getattr(settings, "rail_network", "")).endswith("_extended")

    # Determine points file name for logging
    if is_development:
        points_file_name = "points.gpkg"
    elif is_extended:
        points_file_name = "points.gpkg"
    else:
        points_file_name = "points_corridor.gpkg"

    print(f"[INFO] Loaded {len(master_points)} stations from {points_file_name}")

    # Ensure ID_point is integer for matching
    master_points["ID_point"] = master_points["ID_point"].apply(parse_int)
    points_station_ids = set(master_points["ID_point"])

    # Classify stations: existing (in points.gpkg) vs new (not in points.gpkg)
    new_station_ids = node_ids - points_station_ids
    existing_station_ids = node_ids & points_station_ids

    if new_station_ids:
        print(f"[INFO] Found {len(new_station_ids)} NEW stations not in points.gpkg: {sorted(new_station_ids)}")

    # DEVELOPMENT WORKFLOW: Filter to baseline prep NR column
    if is_development:
        if baseline_prep_path is None:
            raise ValueError("baseline_prep_path is required for development workflow")

        if not baseline_prep_path.exists():
            raise FileNotFoundError(f"Baseline prep workbook not found: {baseline_prep_path}")

        # Load baseline prep stations
        baseline_prep_stations = pd.read_excel(baseline_prep_path, sheet_name="Stations")
        baseline_station_ids = set(baseline_prep_stations["NR"].apply(parse_int))

        print(f"[INFO] Loaded {len(baseline_station_ids)} stations from baseline prep workbook")

        # Filter: Keep only stations that are EITHER in baseline prep OR are new
        # This filters the development to relevant stations based on baseline scope
        relevant_existing_stations = existing_station_ids & baseline_station_ids
        stations_to_keep = relevant_existing_stations | new_station_ids

        # Report filtering results
        filtered_out = existing_station_ids - baseline_station_ids
        if filtered_out:
            print(f"[INFO] Filtered out {len(filtered_out)} stations not in baseline prep scope")

        print(f"[INFO] Keeping {len(relevant_existing_stations)} baseline + {len(new_station_ids)} new = {len(stations_to_keep)} total stations")

    elif is_extended:
        # BASELINE EXTENDED WORKFLOW: Keep ALL stations from edges (no corridor filtering)
        # This captures all stations including those outside corridor boundary
        # Via nodes are included if they appear in edges
        stations_to_keep = existing_station_ids  # All nodes that exist in points.gpkg
        new_station_ids = set()  # No new stations in baseline

        # Report stations not found in points.gpkg (should extract from edges if any)
        outside_points = node_ids - existing_station_ids
        if outside_points:
            print(f"[INFO] {len(outside_points)} edge nodes not found in points.gpkg (will extract from edges)")

        print(f"[INFO] Keeping {len(stations_to_keep)} stations from edges (no corridor filtering)")

    else:
        # BASELINE STANDARD WORKFLOW: Keep only corridor stations (intersection of edge nodes and corridor points)
        # This filters out stations outside the corridor (e.g., endpoints of through-services)
        # but still captures services passing through corridor stations via the Via list
        stations_to_keep = existing_station_ids  # Only nodes that exist in points_corridor.gpkg
        new_station_ids = set()  # No new stations in baseline

        # Report stations filtered out (appear in edges but not in corridor)
        outside_corridor = node_ids - existing_station_ids
        if outside_corridor:
            print(f"[INFO] {len(outside_corridor)} edge nodes are outside corridor boundary (filtered out)")
            print(f"       These are typically endpoints of through-services")

        print(f"[INFO] Keeping {len(stations_to_keep)} stations within corridor boundary")

    # Filter master points to stations we're keeping
    points_gdf = master_points[master_points["ID_point"].isin(stations_to_keep)].copy()

    # For extended mode, also need to extract stations not found in points.gpkg
    if is_extended and not is_development:
        outside_points = node_ids - existing_station_ids
        if outside_points:
            new_station_ids = outside_points

    # For NEW stations (not in points.gpkg), we need to extract from edges
    if new_station_ids:
        print(f"[INFO] Extracting {len(new_station_ids)} new stations from edge geometry...")
        new_stations_gdf = extract_stations_from_edges(edges_gdf)
        new_stations_gdf = new_stations_gdf[new_stations_gdf["ID_point"].isin(new_station_ids)].copy()

        # Flag new stations with [NEW] suffix in NAME (only for development mode)
        if is_development:
            for idx, row in new_stations_gdf.iterrows():
                original_name = row["NAME"]
                new_stations_gdf.at[idx, "NAME"] = f"{original_name} [NEW]"

        # Combine with points from master
        points_gdf = gpd.GeoDataFrame(
            pd.concat([points_gdf, new_stations_gdf], ignore_index=True),
            crs=points_gdf.crs
        )

    print(f"[INFO] Final station count: {len(points_gdf)} stations")

    # Convert offset coordinates to LV95
    points_gdf["E_LV95"] = points_gdf["XKOORD"] + LV95_E_OFFSET
    points_gdf["N_LV95"] = points_gdf["YKOORD"] + LV95_N_OFFSET

    # Remove duplicates based on ID_point (shouldn't happen but be safe)
    initial_count = len(points_gdf)
    points_gdf = points_gdf.drop_duplicates(subset=["ID_point"])
    if len(points_gdf) < initial_count:
        print(f"[WARNING] Removed {initial_count - len(points_gdf)} duplicate stations")

    # Return with required columns
    return points_gdf[["ID_point", "NAME", "CODE", "XKOORD", "YKOORD", "geometry"]].copy(), new_station_ids


def load_corridor_nodes(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Extract station points from edges (used for DEVELOPMENT workflow).

    This function derives station points from edge geometry endpoints.
    For BASELINE workflow, use load_corridor_nodes_from_master() instead.

    Args:
        edges_gdf: Edges GeoDataFrame to extract stations from.

    Returns:
        GeoDataFrame with station points.
    """
    if edges_gdf is None or len(edges_gdf) == 0:
        raise ValueError("edges_gdf must be provided and non-empty")

    # Extract stations from edges
    points_gdf = extract_stations_from_edges(edges_gdf)
    print(f"[INFO] Extracted {len(points_gdf)} stations from development edges")

    return points_gdf


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

def aggregate_station_metrics(
    rail_nodes: pd.DataFrame,
    stop_records: pd.DataFrame,
    service_links: pd.DataFrame,
    corridor_node_ids: set[int],
    stop_lookup: set[Tuple[str, str, int]],
) -> pd.DataFrame:
    """Compute station-level capacity inputs."""

    # Sum per-direction frequencies for services that stop at the node. Each
    # record in ``stop_records`` represents an actual station stop.
    stopping_per_node = (
        stop_records.groupby("Node")["Frequency"].sum().rename("stopping_tph").fillna(0.0)
    )  # Total trains per hour (both directions) that stop at each corridor node.
    stop_services: Dict[int, set[str]] = defaultdict(set)
    for service, direction, node_id in stop_lookup:
        if node_id in corridor_node_ids:
            stop_services[node_id].add(service)
    stop_services_map = {node: ", ".join(sorted(services)) for node, services in stop_services.items()}

    # Count services that pass through the node according to the Via list.
    passing_counter: Dict[int, float] = defaultdict(float)
    passing_services: Dict[int, set[str]] = defaultdict(set)
    for _, row in service_links.iterrows():
        frequency = row["Frequency"]
        if not frequency:
            continue
        service_name = row["Service"]
        for node_id in row["ViaNodes"]:
            if node_id in corridor_node_ids:
                passing_counter[node_id] += frequency  # Aggregate trains per hour that only pass through the node.
                passing_services[node_id].add(service_name)
    passing_per_node = pd.Series(passing_counter, name="passing_tph")
    passing_services_map = {
        node: ", ".join(sorted(services))
        for node, services in passing_services.items()
    }

    # Merge stopping / passing totals back onto the node attributes.
    merged = rail_nodes.merge(
        stopping_per_node,
        how="left",
        left_on="NR",
        right_index=True,
    ).merge(
        passing_per_node,
        how="left",
        left_on="NR",
        right_index=True,
    )

    merged["stopping_tph"] = merged["stopping_tph"].fillna(0.0)
    merged["passing_tph"] = merged["passing_tph"].fillna(0.0)
    merged["stopping_tphpd"] = merged["stopping_tph"] / 2.0
    merged["passing_tphpd"] = merged["passing_tph"] / 2.0
    merged["stopping_services"] = merged["NR"].map(stop_services_map).fillna("")
    merged["passing_services"] = merged["NR"].map(passing_services_map).fillna("")
    merged["tracks"] = pd.NA  # Placeholder to be filled manually.
    merged["platforms"] = pd.NA  # Placeholder to be filled manually.

    output_columns = [
        "NR",
        "NAME",
        "CODE",
        "E_LV95",
        "N_LV95",
        "stopping_tph",
        "passing_tph",
        "stopping_tphpd",
        "passing_tphpd",
        "stopping_services",
        "passing_services",
        "tracks",
        "platforms",
    ]
    return merged[output_columns].sort_values("NR").reset_index(drop=True)


def build_stop_lookup(stop_records: pd.DataFrame) -> set[Tuple[str, str, int]]:
    """Create a lookup set of (service, direction, node) tuples where the service stops."""
    return {
        (row["Service"], row["Direction"], int(row["Node"]))
        for _, row in stop_records.iterrows()
    }


def aggregate_segment_metrics(
    service_links: pd.DataFrame,
    stop_lookup: set[Tuple[str, str, int]],
    corridor_node_ids: set[int],
) -> pd.DataFrame:
    """Compute segment-level statistics directly from processed service links."""
    segment_contribs = build_segment_contributions(service_links, stop_lookup, corridor_node_ids)

    pair_meta: Dict[Tuple[int, int], Dict[str, object]] = {}
    for _, row in service_links.iterrows():
        frequency = row["Frequency"]
        if frequency <= 0:
            continue
        service = row["Service"]
        direction = row["Direction"]
        via_nodes: List[int] = row["ViaNodes"]
        path_nodes: List[int] = [row["FromNode"], *via_nodes, row["ToNode"]]
        segment_count = len(path_nodes) - 1

        for start, end in zip(path_nodes, path_nodes[1:]):
            if start not in corridor_node_ids or end not in corridor_node_ids:
                continue
            pair = tuple(sorted((start, end)))
            meta = pair_meta.setdefault(
                pair,
                {
                    "stop_tts": [],
                    "pass_tts": [],
                    "service_totals": defaultdict(float),
                    "service_direction_totals": defaultdict(float),
                },
            )
            if "service_totals" not in meta:
                meta["service_totals"] = defaultdict(float)
            if "service_direction_totals" not in meta:
                meta["service_direction_totals"] = defaultdict(float)
            service_totals: defaultdict[str, float] = meta["service_totals"]  # type: ignore[assignment]
            direction_totals: defaultdict[Tuple[str, str], float] = meta["service_direction_totals"]  # type: ignore[assignment]
            service_totals[service] += frequency
            direction_key = str(direction).strip()
            direction_totals[(service, direction_key)] += frequency
            stop_from = (service, direction, start) in stop_lookup
            stop_to = (service, direction, end) in stop_lookup
            if pd.notna(row["TravelTime"]) and segment_count == 1:
                if stop_from and stop_to:
                    meta["stop_tts"].append(row["TravelTime"])
                else:
                    meta["pass_tts"].append(row["TravelTime"])

    records: List[Dict[str, object]] = []
    for from_node, to_node in sorted(segment_contribs.keys()):
        meta = pair_meta.get(
            (from_node, to_node),
            {
                "stop_tts": [],
                "pass_tts": [],
                "service_totals": {},
                "service_direction_totals": {},
            },
        )
        contrib = segment_contribs.get((from_node, to_node), {"stop_freq": 0.0, "pass_freq": 0.0})

        stop_tts = meta.get("stop_tts", [])
        pass_tts = meta.get("pass_tts", [])
        service_totals_map = dict(meta.get("service_totals", {}))
        service_direction_totals_map = dict(meta.get("service_direction_totals", {}))
        services_tph = _format_service_frequency_map(service_totals_map)
        services_tphpd = _format_service_direction_frequency_map(service_direction_totals_map)

        travel_time_stopping = max(stop_tts) if stop_tts else pd.NA
        travel_time_passing = max(pass_tts) if pass_tts else pd.NA
        stopping_tph = contrib.get("stop_freq", 0.0)
        passing_tph = contrib.get("pass_freq", 0.0)
        total_tph = stopping_tph + passing_tph
        stopping_tphpd = stopping_tph / 2.0
        passing_tphpd = passing_tph / 2.0
        total_tphpd = stopping_tphpd + passing_tphpd
        records.append(
            {
                "from_node": from_node,
                "to_node": to_node,
                "length_m": pd.NA,
                "tracks": pd.NA,
                "speed": pd.NA,
                "travel_time_stopping": travel_time_stopping,
                "travel_time_passing": travel_time_passing,
                "stopping_tph": stopping_tph,
                "passing_tph": passing_tph,
                "total_tph": total_tph,
                "stopping_tphpd": stopping_tphpd,
                "passing_tphpd": passing_tphpd,
                "total_tphpd": total_tphpd,
                "services_tph": services_tph,
                "services_tphpd": services_tphpd,
            }
        )

    segments_df = pd.DataFrame(records)
    return segments_df.sort_values(["from_node", "to_node"]).reset_index(drop=True)


def _derive_baseline_prep_path() -> Path:
    """Auto-detect baseline prep workbook path from settings.

    Returns:
        Path to the baseline prep workbook.

    Raises:
        FileNotFoundError: If baseline prep workbook doesn't exist.
    """
    network_tag = getattr(settings, "rail_network", "current")
    safe_network_tag = re.sub(r"[^\w-]+", "_", str(network_tag)).strip("_") or "current"

    # Try new structure first: Baseline subdirectory
    prep_path = CAPACITY_ROOT / "Baseline" / safe_network_tag / f"capacity_{safe_network_tag}_network_prep.xlsx"

    if prep_path.exists():
        return prep_path

    # Fallback 1: Old structure with subdirectory (for backwards compatibility)
    prep_path_old_subdir = CAPACITY_ROOT / safe_network_tag / f"capacity_{safe_network_tag}_network_prep.xlsx"
    if prep_path_old_subdir.exists():
        return prep_path_old_subdir

    # Fallback 2: Old structure without subdirectory
    prep_path_old_flat = CAPACITY_ROOT / f"capacity_{safe_network_tag}_network_prep.xlsx"
    if prep_path_old_flat.exists():
        return prep_path_old_flat

    raise FileNotFoundError(
        f"Baseline prep workbook not found. Tried:\n"
        f"  - {prep_path}\n"
        f"  - {prep_path_old_subdir}\n"
        f"  - {prep_path_old_flat}\n\n"
        f"Please run the baseline workflow and manually enrich the workbook first."
    )


def build_capacity_tables(
    edges_path: Path = None,
    network_label: str = None,
    enrichment_source: Path = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return the station and segment capacity tables for the active network.

    Args:
        edges_path: Optional custom path to edges file.
                   - If None: BASELINE workflow (uses edges_in_corridor.gpkg + master points)
                   - If provided: DEVELOPMENT workflow (uses custom edges + edge-derived stations)
        network_label: Optional custom network label for output naming.
        enrichment_source: Optional path to baseline prep workbook for auto-enrichment.
                          - If None: generates empty workbook for manual enrichment (baseline workflow)
                          - If provided: inherits baseline data and applies defaults (development workflow)

    Returns:
        Tuple of (stations_df, segments_df) with capacity metrics.
    """
    is_baseline = edges_path is None

    print(f"\n{'='*80}")
    print(f"[INFO] build_capacity_tables - {'BASELINE' if is_baseline else 'DEVELOPMENT'} workflow")
    print(f"  - edges_path: {edges_path if edges_path else 'Default (baseline)'}")
    print(f"  - network_label: {network_label}")
    print(f"  - enrichment_source: {enrichment_source}")
    print(f"{'='*80}\n")

    # Load service links (handles baseline extended mode internally)
    service_links_full = load_service_links(edges_path=edges_path)

    # Get edges GeoDataFrame for later use (needed for enrichment geometry lookups)
    if is_baseline:
        # For baseline (both standard and extended), load edges from edges_in_corridor.gpkg
        edges_gdf = gpd.read_file(EDGES_IN_CORRIDOR_PATH)
    else:
        # For development, load from custom path
        edges_gdf = gpd.read_file(edges_path)

    # Extract stations based on workflow
    new_station_ids = set()  # Track new stations for enrichment

    if is_baseline:
        # BASELINE: Load stations from master points file
        corridor_points, new_station_ids = load_corridor_nodes_from_master(
            edges_gdf,
            is_development=False
        )
    else:
        # DEVELOPMENT: Load and filter stations
        # Auto-detect baseline prep path
        baseline_prep_path = _derive_baseline_prep_path()

        corridor_points, new_station_ids = load_corridor_nodes_from_master(
            edges_gdf,
            baseline_prep_path=baseline_prep_path,
            is_development=True
        )

    corridor_nodes = set(corridor_points["ID_point"].astype(int))

    # Filter service links to those related to corridor nodes
    def _link_has_corridor_relation(row: pd.Series) -> bool:
        return (
            row["FromNode"] in corridor_nodes
            or row["ToNode"] in corridor_nodes
            or any(node in corridor_nodes for node in row["ViaNodes"])
        )

    service_links = (
        service_links_full[service_links_full.apply(_link_has_corridor_relation, axis=1)]
        .reset_index(drop=True)
    )

    stop_records = build_stop_records(service_links, corridor_nodes)
    stop_lookup = build_stop_lookup(stop_records)

    # Prepare corridor table with proper coordinate handling
    corridor_table = corridor_points.drop(columns=[corridor_points.geometry.name], errors="ignore").copy()
    corridor_table.rename(columns={"ID_point": "NR"}, inplace=True)
    corridor_table["NR"] = corridor_table["NR"].apply(parse_int)
    corridor_table["NAME"] = corridor_table["NAME"].astype(str)
    if "CODE" in corridor_table.columns:
        corridor_table["CODE"] = corridor_table["CODE"].astype(str)
    else:
        corridor_table["CODE"] = ""
    corridor_table["XKOORD"] = corridor_table["XKOORD"].apply(parse_float)
    corridor_table["YKOORD"] = corridor_table["YKOORD"].apply(parse_float)

    # E_LV95/N_LV95 already added by load_corridor_nodes_from_master for baseline
    # For development, need to add them
    if "E_LV95" not in corridor_table.columns:
        corridor_table["E_LV95"] = corridor_table["XKOORD"] + LV95_E_OFFSET
    if "N_LV95" not in corridor_table.columns:
        corridor_table["N_LV95"] = corridor_table["YKOORD"] + LV95_N_OFFSET

    station_metrics = aggregate_station_metrics(
        corridor_table,
        stop_records,
        service_links,
        corridor_nodes,
        stop_lookup,
    )

    node_name_lookup = dict(zip(corridor_table["NR"], corridor_table["NAME"]))

    segment_metrics = aggregate_segment_metrics(service_links, stop_lookup, corridor_nodes)
    segment_metrics["from_station"] = segment_metrics["from_node"].map(node_name_lookup)
    segment_metrics["to_station"] = segment_metrics["to_node"].map(node_name_lookup)

    output_columns = [
        "from_node",
        "from_station",
        "to_node",
        "to_station",
        "length_m",
        "speed",
        "tracks",
        "travel_time_stopping",
        "travel_time_passing",
        "stopping_tph",
        "passing_tph",
        "total_tph",
        "stopping_tphpd",
        "passing_tphpd",
        "total_tphpd",
        "services_tph",
        "services_tphpd",
    ]
    segment_metrics = segment_metrics[output_columns]

    # Apply enrichment if source provided (development workflow)
    if enrichment_source is not None:
        station_metrics, segment_metrics = apply_enrichment(
            station_metrics,
            segment_metrics,
            enrichment_source,
            edges_gdf,
            new_station_ids
        )
    elif not is_baseline:
        # Development workflow without explicit enrichment_source: use auto-detected baseline prep
        baseline_prep_path = _derive_baseline_prep_path()
        station_metrics, segment_metrics = apply_enrichment(
            station_metrics,
            segment_metrics,
            baseline_prep_path,
            edges_gdf,
            new_station_ids
        )

    return station_metrics, segment_metrics


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


def _build_sections_dataframe(stations_df: pd.DataFrame, segments_df: pd.DataFrame) -> pd.DataFrame:
    """Assemble continuous sections that share the same track count."""
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
    node_tracks: Dict[int, int] = {}
    for node_id, track_value in zip(stations_df["NR"], node_tracks_series):
        if pd.notna(track_value):
            node_tracks[int(node_id)] = int(track_value)

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
    segments_df["track_key"] = segments_df["track_key"].astype(int)

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

    edges_by_track: Dict[int, Dict[frozenset, Dict[str, float]]] = {}
    adjacency_by_track: Dict[int, defaultdict[int, set[int]]] = {}

    for row in segments_df.itertuples(index=False):
        track = int(row.track_key)
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
            track_value = node_tracks.get(node_id)
            return track_value == track

        nodes = list(adjacency.keys())
        start_nodes = [node for node in nodes if len(adjacency[node]) != 2 or not node_valid(node)]

        for start in start_nodes:
            for neighbor in list(adjacency[start]):
                edge_key = frozenset({start, neighbor})
                if edge_key in visited_edges:
                    continue
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
                        sections.append(
                            _summarise_section(
                                section_counter,
                                track,
                                refined_nodes,
                                refined_edges,
                                node_names,
                                node_stop_services,
                                node_pass_services,
                            )
                        )
                        section_counter += 1
                    visited_edges.update(path_edge_keys)

        for edge_key, edge_info in edges_dict.items():
            if edge_key in visited_edges:
                continue
            u, v = tuple(edge_key)
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
                    sections.append(
                        _summarise_section(
                            section_counter,
                            track,
                            refined_nodes,
                            refined_edges,
                            node_names,
                            node_stop_services,
                            node_pass_services,
                        )
                    )
                    section_counter += 1
                visited_edges.update(path_edge_keys)

    return pd.DataFrame(sections)


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
        edge_key = frozenset({current, next_node})
        if edge_key in visited_edges or edge_key in local_edges:
            break
        edge_info = edges_dict.get(edge_key)
        if edge_info is None:
            break

        local_edges.add(edge_key)
        path_edge_keys.append(edge_key)
        edge_records.append((current, next_node, edge_info))
        path_nodes.append(next_node)

        if not node_valid(next_node) or len(adjacency[next_node]) != 2:
            break

        candidates = adjacency[next_node] - {current}
        if not candidates:
            break
        candidate = next(iter(candidates))
        candidate_edge = frozenset({next_node, candidate})
        if candidate_edge in visited_edges or candidate_edge in local_edges:
            break

        # Stop section if the next edge changes track or service patterns.
        current_edge_info = edges_dict.get(edge_key)
        candidate_edge_info = edges_dict.get(candidate_edge)
        if candidate_edge_info is None:
            break
        if current_edge_info["track_count"] != candidate_edge_info["track_count"]:
            break
        if current_edge_info["stopping_service_tokens"] != candidate_edge_info["stopping_service_tokens"]:
            break
        if current_edge_info["passing_service_tokens"] != candidate_edge_info["passing_service_tokens"]:
            break

        current, next_node = next_node, candidate

    return path_nodes, edge_records, path_edge_keys


def _split_section_by_service_patterns(
    path_nodes: List[int],
    edge_records: List[Tuple[int, int, Dict[str, float]]],
    node_stop_services: Dict[int, set[str]],
    node_pass_services: Dict[int, set[str]],
) -> List[Tuple[List[int], List[Tuple[int, int, Dict[str, float]]]]]:
    """Split an infrastructure section where service stop/pass patterns change."""
    if len(path_nodes) <= 2 or not edge_records:
        return [(path_nodes, edge_records)]

    candidate_services: set[str] = set()
    for node in path_nodes:
        candidate_services.update(node_stop_services.get(node, set()))
        candidate_services.update(node_pass_services.get(node, set()))

    if not candidate_services:
        return [(path_nodes, edge_records)]

    service_order = sorted(candidate_services)
    service_node_states: Dict[str, List[str]] = {}
    for service in service_order:
        states: List[str] = []
        for node in path_nodes:
            if service in node_stop_services.get(node, set()):
                states.append("stop")
            elif service in node_pass_services.get(node, set()):
                states.append("pass")
            else:
                states.append("absent")

        service_node_states[service] = states

    def _should_split(pattern_a: Tuple[str, ...], pattern_b: Tuple[str, ...]) -> bool:
        """Return True when a service transitions from pass/absent to stop."""
        for state_a, state_b in zip(pattern_a, pattern_b):
            if state_a == state_b:
                continue
            if state_b == "stop" and state_a != "stop":
                return True
            if state_a == "stop" and state_b == "pass":
                continue
        return False

    node_patterns: List[Tuple[str, ...]] = []
    for idx in range(len(path_nodes)):
        pattern = tuple(service_node_states[service][idx] for service in service_order)
        node_patterns.append(pattern)

    refined_sections: List[Tuple[List[int], List[Tuple[int, int, Dict[str, float]]]]] = []
    start_index = 0
    current_pattern = node_patterns[0]

    idx = 1
    while idx < len(path_nodes):
        next_pattern = node_patterns[idx]
        if not _should_split(current_pattern, next_pattern):
            current_pattern = next_pattern
            idx += 1
            continue

        split_edge = idx - 1
        sub_edges = edge_records[start_index : split_edge + 1]
        sub_nodes = path_nodes[start_index : split_edge + 2]
        if sub_edges:
            refined_sections.append((sub_nodes, sub_edges))

        current_pattern = next_pattern
        start_index = split_edge + 1
        idx += 1

    if start_index < len(edge_records):
        refined_sections.append((path_nodes[start_index:], edge_records[start_index:]))

    if not refined_sections:
        return [(path_nodes, edge_records)]

    return refined_sections


def _summarise_section(
    section_id: int,
    track: int,
    path_nodes: List[int],
    edge_records: List[Tuple[int, int, Dict[str, float]]],
    node_names: Dict[int, str],
    node_stop_services: Dict[int, set[str]],
    node_pass_services: Dict[int, set[str]],
) -> Dict[str, object]:
    """Combine edge metrics into a section summary."""
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

    headway = DEFAULT_HEADWAY_MIN
    travel_time_penalty = max(0.0, float(total_stopping_time) - float(total_passing_time) - headway)
    service_count = len(all_services)
    strategy_metrics: List[Tuple[str, float, float]] = []

    if track == 1:
        single_capacity = float("nan")
        if total_stopping_time > 0:
            raw_capacity = _floor_capacity(60.0 / float(total_stopping_time))
            single_capacity = raw_capacity / 2.0 if not math.isnan(raw_capacity) else float("nan")
        capacity_columns["capacity_single_track_tphpd"] = single_capacity
        demand_single_track = (
            total_tphpd if not math.isnan(total_tphpd) else (total_tph / 2.0 if not math.isnan(total_tph) else float("nan"))
        )
        capacity_columns["utilization_single_track"] = _utilization(single_capacity, demand_single_track)
    elif track == 2:
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

    if track == 1:
        selected_capacity = capacity_columns["capacity_single_track_tphpd"]
        selected_utilization = capacity_columns["utilization_single_track"]
    elif track == 2:
        if not stopping_services or not passing_services_list:
            selected_capacity = capacity_columns["capacity_uniform_pattern_tphpd"]
            selected_utilization = capacity_columns["utilization_uniform_pattern"]
        elif strategy_metrics:
            if len(strategy_metrics) == 1:
                _, cap_value, util_value = strategy_metrics[0]
                selected_capacity = cap_value
                selected_utilization = util_value
            else:
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
