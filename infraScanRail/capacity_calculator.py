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
from typing import Dict, List, Tuple

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

def capacity_output_path() -> Path:
    """Return the capacity workbook path for the active rail network."""
    network_tag = getattr(settings, "rail_network", "current")  # Use the configured scenario name.
    safe_network_tag = re.sub(r"[^\w-]+", "_", str(network_tag)).strip("_") or "current"
    filename = f"capacity_{safe_network_tag}_network.xlsx"
    return CAPACITY_ROOT / filename

EDGES_IN_CORRIDOR_PATH = PROCESSED_ROOT / "edges_in_corridor.gpkg"
CORRIDOR_POINTS_PATH = PROCESSED_ROOT / "points_corridor.gpkg"

DECIMAL_COMMA = ","

LV95_E_OFFSET = 2_000_000
LV95_N_OFFSET = 1_000_000

try:
    import xlsxwriter  # noqa: F401

    EXCEL_ENGINE = "xlsxwriter"
except ImportError:
    try:
        import openpyxl  # noqa: F401

        EXCEL_ENGINE = "openpyxl"
    except ImportError as exc:  # pragma: no cover - fail fast if neither available
        raise ImportError(
            "Neither 'xlsxwriter' nor 'openpyxl' is installed. "
            "Please install one of them to export Excel files."
        ) from exc

try:
    import openpyxl  # noqa: F401

    APPEND_ENGINE = "openpyxl"
except ImportError:
    APPEND_ENGINE = None

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


def load_service_links() -> pd.DataFrame:
    """Load service link records from the processed corridor edges GeoPackage."""
    if not EDGES_IN_CORRIDOR_PATH.exists():
        raise FileNotFoundError(
            f"Processed corridor edges not found at {EDGES_IN_CORRIDOR_PATH}."
        )

    gdf = gpd.read_file(EDGES_IN_CORRIDOR_PATH)
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

def load_corridor_nodes() -> gpd.GeoDataFrame:
    """Load the corridor-only nodes GeoPackage."""
    return gpd.read_file(CORRIDOR_POINTS_PATH)


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
            segment = contributions.setdefault(
                pair,
                {"stop_freq": 0.0, "pass_freq": 0.0, "dir_freq": defaultdict(float)},
            )
            stop_start = (service, direction, start) in stop_lookup
            stop_end = (service, direction, end) in stop_lookup
            if stop_start and stop_end:
                segment["stop_freq"] += frequency
            else:
                segment["pass_freq"] += frequency
            segment["dir_freq"][(service, direction)] += frequency

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
    )  # Total trains per hour that stop at each corridor node.
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

    pair_meta: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
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
            meta = pair_meta.setdefault(pair, {"stop_tts": [], "pass_tts": []})
            stop_from = (service, direction, start) in stop_lookup
            stop_to = (service, direction, end) in stop_lookup
            if pd.notna(row["TravelTime"]) and segment_count == 1:
                if stop_from and stop_to:
                    meta["stop_tts"].append(row["TravelTime"])
                else:
                    meta["pass_tts"].append(row["TravelTime"])

    records: List[Dict[str, object]] = []
    for from_node, to_node in sorted(segment_contribs.keys()):
        meta = pair_meta.get((from_node, to_node), {"stop_tts": [], "pass_tts": []})
        contrib = segment_contribs.get((from_node, to_node), {"stop_freq": 0.0, "pass_freq": 0.0, "dir_freq": {}})

        stop_tts = meta.get("stop_tts", [])
        pass_tts = meta.get("pass_tts", [])

        travel_time_stopping = max(stop_tts) if stop_tts else pd.NA
        travel_time_passing = max(pass_tts) if pass_tts else pd.NA
        stopping_tph = contrib.get("stop_freq", 0.0)
        passing_tph = contrib.get("pass_freq", 0.0)
        dir_freq = contrib.get("dir_freq", {})
        freq_items = dir_freq.items() if dir_freq else []
        frequency_summary = "; ".join(
            f"{svc}.{direction}: {freq:g}"
            for (svc, direction), freq in sorted(freq_items)
        )

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
                "tphpd": stopping_tph + passing_tph,
                "directional_frequency": frequency_summary,
            }
        )

    segments_df = pd.DataFrame(records)
    return segments_df.sort_values(["from_node", "to_node"]).reset_index(drop=True)


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
    segments_df["stopping_value"] = pd.to_numeric(segments_df.get("stopping_tph"), errors="coerce").fillna(0.0)
    segments_df["passing_tph_value"] = pd.to_numeric(segments_df.get("passing_tph"), errors="coerce").fillna(0.0)
    segments_df["tphpd_value"] = pd.to_numeric(segments_df.get("tphpd"), errors="coerce").fillna(0.0)
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

        directional_raw = getattr(row, "directional_frequency", "")
        if pd.isna(directional_raw):
            directional_raw = ""
        directional_raw = str(directional_raw)
        directional_tokens = _parse_services(directional_raw)[1] if directional_raw else tuple()

        edge_info = {
            "from_node": u,
            "to_node": v,
            "length": float(row.length_value),
            "passing_time": float(row.passing_value),
            "stopping_time": float(row.stop_time_value),
            "speed": None if pd.isna(row.speed_value) else float(row.speed_value),
            "stopping_tph": float(row.stopping_value),
            "passing_tph": float(row.passing_tph_value),
            "tphpd": float(row.tphpd_value),
            "stopping_services": row.stopping_services,
            "stopping_service_tokens": stopping_tokens,
            "passing_services": row.passing_services,
            "passing_service_tokens": passing_tokens,
            "track_count": track,
            "via_nodes": via_tokens,
            "directional_frequency": directional_raw,
            "directional_service_tokens": directional_tokens,
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
                path_nodes, edge_records = _traverse_path(
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

        for edge_key, edge_info in edges_dict.items():
            if edge_key in visited_edges:
                continue
            u, v = tuple(edge_key)
            path_nodes, edge_records = _traverse_path(
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

    return pd.DataFrame(sections)


def _traverse_path(
    start: int,
    neighbor: int,
    adjacency: Dict[int, set[int]],
    edges_dict: Dict[frozenset, Dict[str, float]],
    visited_edges: set[frozenset],
    node_valid,
) -> Tuple[List[int], List[Tuple[int, int, Dict[str, float]]]]:
    """Walk a path while track conditions remain satisfied."""
    path_nodes: List[int] = [start]
    edge_records: List[Tuple[int, int, Dict[str, float]]] = []
    current = start
    next_node = neighbor

    while True:
        edge_key = frozenset({current, next_node})
        if edge_key in visited_edges:
            break
        edge_info = edges_dict.get(edge_key)
        if edge_info is None:
            break

        visited_edges.add(edge_key)
        edge_records.append((current, next_node, edge_info))
        path_nodes.append(next_node)

        if not node_valid(next_node) or len(adjacency[next_node]) != 2:
            break

        candidates = adjacency[next_node] - {current}
        if not candidates:
            break
        candidate = next(iter(candidates))
        candidate_edge = frozenset({next_node, candidate})
        if candidate_edge in visited_edges:
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

    return path_nodes, edge_records


def _split_section_by_service_patterns(
    path_nodes: List[int],
    edge_records: List[Tuple[int, int, Dict[str, float]]],
    node_stop_services: Dict[int, set[str]],
    node_pass_services: Dict[int, set[str]],
) -> List[Tuple[List[int], List[Tuple[int, int, Dict[str, float]]]]]:
    """Split an infrastructure section where service stop/pass patterns change."""
    if len(path_nodes) <= 2 or not edge_records:
        return [(path_nodes, edge_records)]

    node_stop_seq = [node_stop_services.get(node, set()) for node in path_nodes]
    node_pass_seq = [node_pass_services.get(node, set()) for node in path_nodes]
    candidate_services: set[str] = set().union(*node_stop_seq, *node_pass_seq)

    if not candidate_services:
        return [(path_nodes, edge_records)]

    service_order = sorted(candidate_services)
    node_patterns: List[Tuple[str, ...]] = []
    for stop_set, pass_set in zip(node_stop_seq, node_pass_seq):
        pattern = []
        for service in service_order:
            if service in stop_set:
                pattern.append("stop")
            elif service in pass_set:
                pattern.append("pass")
            else:
                pattern.append("absent")
        node_patterns.append(tuple(pattern))

    refined_sections: List[Tuple[List[int], List[Tuple[int, int, Dict[str, float]]]]] = []
    start_index = 0
    current_pattern = node_patterns[0]

    idx = 1
    while idx < len(path_nodes):
        next_pattern = node_patterns[idx]
        if next_pattern == current_pattern:
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

    stopping_tph_values = sorted({edge_info["stopping_tph"] for _, _, edge_info in edge_records})
    passing_tph_values = sorted({edge_info["passing_tph"] for _, _, edge_info in edge_records})
    tphpd_values = sorted({edge_info["tphpd"] for _, _, edge_info in edge_records})

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

    local_services = stopping_services

    stopping_tph_value = stopping_tph_values[0] if len(stopping_tph_values) == 1 else float("nan")
    passing_tph_value = passing_tph_values[0] if len(passing_tph_values) == 1 else float("nan")
    tphpd_value = tphpd_values[0] if len(tphpd_values) == 1 else float("nan")

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
        "tphpd": tphpd_value,
        "stopping_services": ", ".join(local_services),
        "passing_services": ", ".join(passing_services_list),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def export_capacity_workbook() -> Path:
    """Build the capacity workbook and return the output path."""
    ensure_output_directory()

    # Corridor definition is taken directly from the dedicated GeoPackage.
    corridor_points = load_corridor_nodes()
    corridor_nodes = set(corridor_points["ID_point"].astype(int))  # Corridor nodes drive scope of capacity aggregation.

    service_links_full = load_service_links()

    # Retain only links that touch corridor nodes or list a corridor node in
    # their Via column. This keeps relevant services while ignoring distant
    # parts of the network.
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
    stop_lookup = build_stop_lookup(stop_records)  # Precompute set lookups for fast segment aggregation.

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
    corridor_table["E_LV95"] = corridor_table["XKOORD"] + LV95_E_OFFSET
    corridor_table["N_LV95"] = corridor_table["YKOORD"] + LV95_N_OFFSET

    station_metrics = aggregate_station_metrics(corridor_table, stop_records, service_links, corridor_nodes, stop_lookup)

    # Look up human-readable station names for the aggregated segments.
    node_name_lookup = dict(zip(corridor_table["NR"], corridor_table["NAME"]))

    segment_metrics = aggregate_segment_metrics(service_links, stop_lookup, corridor_nodes)
    segment_metrics["from_station"] = segment_metrics["from_node"].map(node_name_lookup)  # Provide human-readable names for output.
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
        "tphpd",
        "directional_frequency",
    ]
    segment_metrics = segment_metrics[output_columns]

    output_path = capacity_output_path()  # Name workbook after the active rail network.

    with pd.ExcelWriter(output_path, engine=EXCEL_ENGINE) as writer:
        station_metrics.to_excel(writer, sheet_name="Stations", index=False)  # Station-level metrics tab.
        segment_metrics.to_excel(writer, sheet_name="Segments", index=False)

    _post_export_capacity_processing(output_path)

    return output_path


if __name__ == "__main__":
    output_file = export_capacity_workbook()
    print(f"Capacity workbook written to {output_file}")
