"""Georeferenced plotting for the rail capacity workbook.

This module visualises the latest capacity prep workbook by plotting
stations and segments in LV95 coordinates, applying styling rules for
track/platform availability, and annotating the network with the key
capacity inputs.
"""

# ---------------------------------------------------------------------------
# Imports & paths
# ---------------------------------------------------------------------------

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, TYPE_CHECKING, Union

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle, Circle
from matplotlib.ticker import PercentFormatter
import pandas as pd

import paths
import settings

try:
    import geopandas as gpd  # type: ignore
    from shapely import make_valid  # type: ignore
    from shapely.geometry import LineString  # type: ignore
except ImportError:  # pragma: no cover - optional geospatial support
    gpd = None  # type: ignore
    make_valid = None  # type: ignore
    LineString = None  # type: ignore

try:
    from matplotlib_map_utils.core.north_arrow import north_arrow as _north_arrow_fn
    _HAS_NORTH_ARROW = True
except ImportError:
    _north_arrow_fn = None
    _HAS_NORTH_ARROW = False

if TYPE_CHECKING:
    from geopandas import GeoDataFrame
else:
    GeoDataFrame = Any  # type: ignore[misc]

# Colour palette for service plot.
SERVICE_COLOUR_STOP = "#2e7d32"
SERVICE_COLOUR_PASS = "#0277bd"
SERVICE_OFFSET_SPACING = 60.0
SERVICE_FREQUENCY_SPACING = 60.0  # Spacing between parallel frequency lines within a service
SERVICE_OVERLAP_EXPANSION_FACTOR = 1.15  # Factor to expand spacing when collision detected
SERVICE_SPACING_MAX_RECURSION = 10  # Max recursion depth for spacing calculation
SERVICE_MIN_GAP = 60.0  # Minimum gap (in units) between adjacent service frequency bundles
SERVICE_RECT_MARGIN = 40.0
STATION_BASE_HALF_SIZE = 60.0
STATION_PER_SERVICE_INCREMENT = 14.0

# Colour palette for catchment-area (infrabuild-style) plots.
_CA_TRACK_COLOURS: Dict[int, str] = {1: "#e41a1c", 2: "#377eb8", 3: "#4daf4a", 4: "#984ea3"}
_CA_TRACK_DEFAULT = "#ff7f00"
_CA_NODE_COLOURS: Dict[str, str] = {
    "station":           "#e41a1c",
    "abandoned_station": "#888888",
    "junction":          "#377eb8",
}
_CA_NODE_DEFAULT = "#aaaaaa"


def _safe_make_valid(geometry):
    """Apply shapely.make_valid when available, otherwise return the geometry unchanged."""
    if make_valid is None or geometry is None:
        return geometry
    try:
        return make_valid(geometry)
    except Exception:
        return geometry


def _calculate_required_spacing(service_map: Dict[str, float], spacing: float) -> Tuple[bool, float]:
    """Check if services overlap with given spacing and return maximum offset needed.
    
    Enforces both collision-free overlaps and a minimum gap between services.
    
    Args:
        service_map: Dictionary of service_name -> frequency (tphpd)
        spacing: Current SERVICE_OFFSET_SPACING value to test
    
    Returns:
        Tuple of (has_insufficient_gap: bool, max_offset: float) where:
        - has_insufficient_gap: True if services violate overlap or minimum gap constraints
        - max_offset: Maximum perpendicular distance needed from center line to contain all services
    """
    if len(service_map) <= 1:
        # Single service - no inter-service collision possible
        if service_map:
            service_name, frequency = next(iter(service_map.items()))
            int_freq = max(int(round(frequency)), 1)
            max_freq_offset = (int_freq - 1) * SERVICE_FREQUENCY_SPACING / 2.0
            return False, max_freq_offset
        return False, 0.0
    
    service_count = len(service_map)
    
    # Calculate offset positions for each service (centered around 0)
    service_positions = []
    for idx in range(service_count):
        service_offset = (idx - (service_count - 1) / 2.0) * spacing
        service_positions.append(service_offset)
    
    # For each service, calculate its frequency bundle width
    service_widths = []
    for idx, (service_name, frequency) in enumerate(service_map.items()):
        int_freq = max(int(round(frequency)), 1)
        freq_bundle_half_width = (int_freq - 1) * SERVICE_FREQUENCY_SPACING / 2.0
        service_center = service_positions[idx]
        service_min = service_center - freq_bundle_half_width
        service_max = service_center + freq_bundle_half_width
        service_widths.append((service_min, service_max))
    
    # Check for overlaps and insufficient gaps between consecutive service frequency bundles
    has_insufficient_gap = False
    for i in range(len(service_widths) - 1):
        min_i, max_i = service_widths[i]
        min_j, max_j = service_widths[i + 1]
        # Calculate gap between end of service i and start of service j
        gap = min_j - max_i
        # Insufficient gap if overlap (gap < 0) or gap smaller than minimum required
        if gap < SERVICE_MIN_GAP:
            has_insufficient_gap = True
            break
    
    # Calculate maximum offset needed (half-width from center to outermost point)
    all_points = [pt for min_pt, max_pt in service_widths for pt in (min_pt, max_pt)]
    if all_points:
        max_offset = max(abs(min(all_points)), abs(max(all_points)))
    else:
        max_offset = 0.0
    
    return has_insufficient_gap, max_offset


def _find_optimal_spacing(
    service_map: Dict[str, float],
    initial_spacing: float = SERVICE_OFFSET_SPACING,
    recursion_depth: int = 0,
) -> float:
    """Recursively find optimal SERVICE_OFFSET_SPACING to avoid service collisions.
    
    Tests current spacing for overlap. If overlap detected, expands spacing by
    SERVICE_OVERLAP_EXPANSION_FACTOR and retries. Stops at max recursion depth.
    
    Args:
        service_map: Dictionary of service_name -> frequency (tphpd)
        initial_spacing: Starting spacing value to test
        recursion_depth: Current recursion depth (incremented on each call)
    
    Returns:
        Optimal spacing value that avoids collisions, or maximum reached spacing if limit hit.
    """
    has_overlap, _ = _calculate_required_spacing(service_map, initial_spacing)
    
    if not has_overlap:
        # No overlap - this spacing is safe
        return initial_spacing
    
    if recursion_depth >= SERVICE_SPACING_MAX_RECURSION:
        # Hit recursion limit - return current spacing anyway
        return initial_spacing
    
    # Expand spacing and retry
    expanded_spacing = initial_spacing * SERVICE_OVERLAP_EXPANSION_FACTOR
    return _find_optimal_spacing(service_map, expanded_spacing, recursion_depth + 1)


class _DoubleTrackLegendHandle:
    """Placeholder object for rendering double-track legend entries with dividers."""

    def __init__(self, line_width: float, divider_width: float):
        self.line_width = line_width
        self.divider_width = divider_width


class _DoubleTrackLegendHandler(HandlerBase):
    """Custom legend handler that renders two parallel lines for double track."""

    def create_artists(
        self,
        legend,
        orig_handle: "_DoubleTrackLegendHandle",
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        x0 = xdescent
        x1 = xdescent + width
        y = ydescent + height / 2.0

        # Use parallel line rendering for 2-track
        track_count = 2
        gap_factor = 0.4
        individual_line_width = orig_handle.line_width / (track_count + (track_count - 1) * gap_factor)
        line_spacing = individual_line_width * (1 + gap_factor)

        artists = []
        for track_idx in range(track_count):
            offset_y = y + (track_idx - 0.5) * line_spacing
            line = Line2D([x0, x1], [offset_y, offset_y], color="black", linewidth=individual_line_width, solid_capstyle="round")
            line.set_transform(trans)
            artists.append(line)

        return artists


class _MultiTrackLegendHandle:
    """Placeholder object for rendering multi-track (3+) legend entries."""

    def __init__(self, total_width: float, track_count: int):
        self.total_width = total_width
        self.track_count = track_count


class _MultiTrackLegendHandler(HandlerBase):
    """Custom legend handler that renders multiple parallel lines for 3+ tracks."""

    def create_artists(
        self,
        legend,
        orig_handle: "_MultiTrackLegendHandle",
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        x0 = xdescent
        x1 = xdescent + width
        y = ydescent + height / 2.0

        track_count = orig_handle.track_count
        gap_factor = 0.4
        individual_line_width = orig_handle.total_width / (track_count + (track_count - 1) * gap_factor)
        line_spacing = individual_line_width * (1 + gap_factor)

        artists = []
        for track_idx in range(track_count):
            offset_y = y + (track_idx - (track_count - 1) / 2.0) * line_spacing
            line = Line2D([x0, x1], [offset_y, offset_y], color="black", linewidth=individual_line_width, solid_capstyle="round")
            line.set_transform(trans)
            artists.append(line)

        return artists


_SCALE_BAR_NICE_KM = [1, 2, 5, 10, 20, 50, 100, 200, 500]


def _add_north_arrow(ax, location: str = "upper left", scale: float = 0.5) -> None:
    if not _HAS_NORTH_ARROW:
        return
    try:
        _north_arrow_fn(ax, location=location, scale=scale, rotation={"degrees": 0})
    except Exception:
        pass


def _add_scale_bar(ax, location: Tuple[float, float] = (0.97, 0.020)) -> None:
    """Adaptive scale bar anchored at axes-fraction coordinates."""
    xlim = ax.get_xlim()
    map_w = xlim[1] - xlim[0]
    if map_w <= 0:
        return
    target_km = (map_w / 4.0) / 1000.0
    total_km = min(_SCALE_BAR_NICE_KM, key=lambda v: abs(v - target_km))
    n_cells = 4 if total_km >= 4 else 2
    cell_m = (total_km * 1000.0) / n_cells
    cell_frac = cell_m / map_w
    x0 = location[0] - n_cells * cell_frac
    y0 = location[1]
    bar_h = 0.012
    for i in range(n_cells):
        color = "black" if i % 2 == 0 else "white"
        rect = Rectangle(
            (x0 + i * cell_frac, y0), cell_frac, bar_h,
            facecolor=color, edgecolor="black", linewidth=0.6,
            transform=ax.transAxes, zorder=7,
        )
        ax.add_patch(rect)
    for i in range(n_cells + 1):
        val_km = (i * cell_m) / 1000.0
        label = f"{val_km:.0f} km" if val_km == int(val_km) else f"{val_km:.1f} km"
        ax.text(
            x0 + i * cell_frac, y0 + bar_h * 1.6,
            label, ha="center", va="bottom", fontsize=7,
            transform=ax.transAxes, zorder=7,
        )


CAPACITY_DIR = Path(paths.MAIN) / "data" / "Network" / "Capacity"
DEFAULT_OUTPUT_DIR = Path(paths.MAIN) / "plots" / "Network" / "Capacity"


def _derive_plot_output_path(
    plot_type: str,
    network_label: str = None,
    output_path: str = None,
    output_dir: Path = None,
) -> Path:
    """Derive plot output path with network-based subdirectories.

    For baseline workflow:
      - plots/network/AK_2035/AK_2035_network_{plot_type}.png
      - plots/network/AK_2035_extended/AK_2035_extended_network_{plot_type}.png

    For development workflow:
      - plots/network/developments/{devID}/AK_2035_dev_{devID}_network_{plot_type}.png

    Args:
        plot_type: Type of plot (e.g., "infrastructure", "capacity", "speed", "service")
        network_label: Optional custom network label (e.g., "AK_2035_dev_100023")
        output_path: Optional explicit path for output file
        output_dir: Optional custom output directory

    Returns:
        Path to the plot output file.
    """
    import re

    # If explicit output_path provided, use it directly
    if output_path is not None:
        return Path(output_path)

    # Determine network name
    if network_label is not None:
        network_tag = network_label
    else:
        network_tag = getattr(settings, "rail_network", "current")

    safe_network_tag = re.sub(r"[^\w-]+", "_", str(network_tag)).strip("_") or "current"
    filename = f"{safe_network_tag}_{plot_type}.pdf"

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename
    else:
        # Fallback: DEFAULT_OUTPUT_DIR / safe_network_tag
        fallback = DEFAULT_OUTPUT_DIR / safe_network_tag
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback / filename


def _calculate_figure_size(stations: Dict[int, Station]) -> Tuple[float, float]:
    """Calculate dynamic figure size based on station bounding box (Option A: aspect ratio).

    Uses fixed aspect ratio scaling with min/max limits:
    - Minimum: (12, 10) - the existing default size
    - Maximum: (24, 20) - reasonable upper limit

    Args:
        stations: Dictionary of stations with coordinates

    Returns:
        Tuple of (width, height) in inches
    """
    if not stations:
        return (12.0, 10.0)

    # Calculate bounding box
    x_coords = [s.x for s in stations.values()]
    y_coords = [s.y for s in stations.values()]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    bbox_width = max_x - min_x
    bbox_height = max_y - min_y

    if bbox_width <= 0 or bbox_height <= 0:
        return (12.0, 10.0)

    # Calculate aspect ratio
    aspect_ratio = bbox_width / bbox_height

    # Scale figure size based on aspect ratio, starting from base width
    base_width = 12.0
    fig_height = base_width / aspect_ratio

    # Apply minimum constraints (existing size)
    if base_width < 12.0:
        base_width = 12.0
    if fig_height < 10.0:
        fig_height = 10.0

    # Apply maximum constraints
    if base_width > 24.0:
        base_width = 24.0
    if fig_height > 20.0:
        fig_height = 20.0

    return (base_width, fig_height)


@dataclass(frozen=True)
class Station:
    node_id: int
    code: str
    name: str
    x: float  # LV95 Easting
    y: float  # LV95 Northing
    tracks: float
    platforms: float
    stopping_services: frozenset[str] = field(default_factory=frozenset)
    passing_services: frozenset[str] = field(default_factory=frozenset)
    stopping_tphpd: float = math.nan
    is_junction: bool = False
    node_class: str = "station"


@dataclass(frozen=True)
class Segment:
    from_node: int
    to_node: int
    tracks: float
    speed: float
    total_tphpd: float = math.nan
    capacity_tphpd: float = math.nan
    utilization: float = math.nan
    capacity_base_tphpd: float = math.nan
    length_m: float = math.nan
    travel_time_stopping: float = math.nan
    travel_time_passing: float = math.nan
    services_tphpd: str = ""
    travel_time_passing: float = math.nan


@dataclass(frozen=True)
class StationShape:
    orientation: float
    axis_u: Tuple[float, float]
    axis_v: Tuple[float, float]
    along_half: float
    across_half: float
    polygon: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]


@dataclass(frozen=True)
class SectionSummary:
    section_id: int
    track_count: float
    start: Station
    end: Station
    total_tphpd: float
    capacity_tphpd: float
    utilization: float
    stopping_tphpd: float
    passing_tphpd: float
    intermediate_stations: Tuple[Station, ...] = ()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _latest_prep_workbook() -> Path:
    """Return the most recently modified capacity prep workbook.

    Searches both:
    - New structure: CAPACITY_DIR/<network>/*_prep.xlsx
    - Legacy flat structure: CAPACITY_DIR/*_prep.xlsx
    """
    # Search recursively for prep workbooks (includes subdirectories)
    prep_files = sorted(
        CAPACITY_DIR.glob("**/*_prep.xlsx"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not prep_files:
        raise FileNotFoundError(
            f"No capacity prep workbooks were found under {CAPACITY_DIR}."
        )
    return prep_files[0]


def _resolve_workbook_path(network_label: str = None, output_dir: Path = None) -> Path:
    """Resolve workbook path using same logic as capacity_calculator.

    For development networks, auto-detects output directory from network_label.

    Args:
        network_label: Optional custom network label (e.g., "AK_2035_dev_100023").
                      If None, uses settings.rail_network.
        output_dir: Optional custom output directory. If None, auto-detects based on label.

    Returns:
        Path to the capacity prep workbook.
    """
    # Auto-detect output directory from network_label (development or enhanced)
    if output_dir is None and network_label is not None:
        import re
        dev_match = re.search(r'_dev_(\d+)', network_label)
        is_enhanced = "_enhanced" in network_label

        if dev_match:
            # Development network
            dev_id = dev_match.group(1)
            from capacity_calculator import CAPACITY_ROOT
            output_dir = CAPACITY_ROOT / "Developments" / dev_id
        elif is_enhanced:
            # Enhanced network - capacity_output_path will handle this
            # Just pass through to capacity_output_path
            pass

    from capacity_calculator import capacity_output_path
    base_path = capacity_output_path(network_label=network_label, output_dir=output_dir)
    return base_path.with_name(f"{base_path.stem}_prep{base_path.suffix}")


def _load_workbook(
    workbook_path: Optional[Path] = None,
    network_label: str = None,
    output_dir: Path = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load station and segment tables from the prep workbook.

    Args:
        workbook_path: Optional explicit path to workbook. If provided, other args are ignored.
        network_label: Optional custom network label (e.g., "AK_2035_dev_100023").
        output_dir: Optional custom output directory.

    Returns:
        Tuple of (stations_df, segments_df).
    """
    if workbook_path is not None:
        workbook = Path(workbook_path)
    elif network_label is not None or output_dir is not None:
        workbook = _resolve_workbook_path(network_label=network_label, output_dir=output_dir)
    else:
        workbook = _latest_prep_workbook()

    if not workbook.exists():
        raise FileNotFoundError(f"Workbook not found: {workbook}")

    xl = pd.ExcelFile(workbook)
    stn_sheet = "Stations" if "Stations" in xl.sheet_names else "Stations_Peak"
    seg_sheet = "Segments" if "Segments" in xl.sheet_names else "Segments_Peak"
    stations = pd.read_excel(workbook, sheet_name=stn_sheet)
    segments = pd.read_excel(workbook, sheet_name=seg_sheet)
    return stations, segments


def _parse_station_services(cell) -> Set[str]:
    if cell is None:
        return set()
    tokens = []
    for chunk in str(cell).replace(",", ";").split(";"):
        token = chunk.strip()
        if token:
            tokens.append(token)
    return set(tokens)


def _filter_stations_by_class(
    stations: Dict[int, Station],
    allowed_node_classes: Optional[Set[str]],
) -> Dict[int, Station]:
    """Return stations filtered to allowed Node_Class values; None means no filter."""
    if allowed_node_classes is None:
        return stations
    allowed_lower = {c.lower() for c in allowed_node_classes}
    return {nid: s for nid, s in stations.items() if s.node_class.lower() in allowed_lower}


def _to_stations(stations_df: pd.DataFrame) -> Dict[int, Station]:
    """Convert the stations dataframe into Station records indexed by NR."""
    parsed: Dict[int, Station] = {}
    for row in stations_df.itertuples(index=False):
        try:
            _nr = getattr(row, "NR", None) or getattr(row, "Number", None)
            node_id = int(_nr)
            _e = getattr(row, "E_LV95", None) or getattr(row, "E", None)
            _n = getattr(row, "N_LV95", None) or getattr(row, "N", None)
            x = float(_e)
            y = float(_n)
        except (TypeError, ValueError):
            continue

        code = str(getattr(row, "CODE", None) or getattr(row, "Code", "") or "").strip()
        name = str(getattr(row, "NAME", None) or getattr(row, "Name", "") or "").strip()
        tracks = _coerce_number(
            getattr(row, "tracks", None) or getattr(row, "Track_Count", math.nan)
        )
        platforms = _coerce_number(
            getattr(row, "platforms", None) or getattr(row, "Platform_Count", math.nan)
        )
        stopping_services = frozenset(_parse_station_services(getattr(row, "stopping_services", "")))
        passing_services = frozenset(_parse_station_services(getattr(row, "passing_services", "")))
        stopping_tphpd = _coerce_number(
            getattr(row, "stopping_tphpd", None)
            or getattr(row, "stopping_tphpd_peak", math.nan)
        )
        node_class = str(getattr(row, "Node_Class", "") or "").strip()
        is_junction = (node_class.lower() == "junction")
        parsed[node_id] = Station(
            node_id=node_id,
            code=code or name or str(node_id),
            name=name or code or str(node_id),
            x=x,
            y=y,
            tracks=tracks,
            platforms=platforms,
            stopping_services=stopping_services,
            passing_services=passing_services,
            stopping_tphpd=stopping_tphpd,
            is_junction=is_junction,
            node_class=node_class or "station",
        )
    return parsed


def _to_segments(segments_df: pd.DataFrame, valid_nodes: Iterable[int]) -> List[Segment]:
    """Convert the segments dataframe into segment records."""
    node_set = set(valid_nodes)
    records: List[Segment] = []
    for row in segments_df.itertuples(index=False):
        try:
            from_node = int(getattr(row, "from_node"))
            to_node = int(getattr(row, "to_node"))
        except (TypeError, ValueError):
            continue

        if from_node not in node_set or to_node not in node_set:
            continue

        tracks = _coerce_number(
            getattr(row, "tracks", None) or getattr(row, "Num_Tracks", math.nan)
        )
        speed = _coerce_number(
            getattr(row, "speed", None) or getattr(row, "Average_Speed", math.nan)
        )
        total_tphpd = _coerce_number(
            getattr(row, "total_tphpd", None) or getattr(row, "total_tphpd_peak", math.nan)
        )
        selected_capacity = _coerce_number(
            getattr(row, "Capacity", None) or getattr(row, "Capacity_peak", math.nan)
        )
        base_capacity = _coerce_number(getattr(row, "capacity_base_tphpd", math.nan))
        utilization = _coerce_number(
            getattr(row, "Utilization", None) or getattr(row, "Utilization_peak", math.nan)
        )
        length_m = _coerce_number(
            getattr(row, "length_m", None) or getattr(row, "Length", math.nan)
        )
        travel_time_stopping = _coerce_number(
            getattr(row, "travel_time_stopping", None) or getattr(row, "TT_Stopping", math.nan)
        )
        travel_time_passing = _coerce_number(
            getattr(row, "travel_time_passing", None) or getattr(row, "TT_Passing", math.nan)
        )
        services_tphpd_cell = str(getattr(row, "services_tphpd", "") or "")
        records.append(
            Segment(
                from_node=from_node,
                to_node=to_node,
                tracks=tracks,
                speed=speed,
                total_tphpd=total_tphpd,
                capacity_tphpd=selected_capacity,
                utilization=utilization,
                capacity_base_tphpd=base_capacity,
                length_m=length_m,
                travel_time_stopping=travel_time_stopping,
                travel_time_passing=travel_time_passing,
                services_tphpd=services_tphpd_cell,
            )
        )
    return records


def _coerce_number(value) -> float:
    """Convert spreadsheet values into floats, returning NaN when unavailable."""
    try:
        numeric = float(value)
        if math.isnan(numeric):
            return math.nan
        return numeric
    except (TypeError, ValueError):
        return math.nan


def _segment_key(from_node: int, to_node: int) -> Tuple[int, int]:
    """Generate a stable key for undirected segment lookup."""
    return tuple(sorted((from_node, to_node)))


_ASYMMETRIC_SERVICES: Set[str] = set()


def _polyline_midpoint(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    """Return the point at half the cumulative arc length along the polyline.

    Used to anchor section labels to the visual centre of the line that runs
    through intermediate stations, rather than the straight-line midpoint
    between endpoints (which can be far off the actual route).
    """
    if not xs or not ys:
        return (0.0, 0.0)
    if len(xs) == 1:
        return (xs[0], ys[0])
    seg_lens = [
        math.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(len(xs) - 1)
    ]
    total = sum(seg_lens)
    if total <= 0.0:
        return (xs[0], ys[0])
    target = total / 2.0
    cum = 0.0
    for i, seg_len in enumerate(seg_lens):
        if cum + seg_len >= target:
            t = (target - cum) / seg_len if seg_len > 0 else 0.0
            return (xs[i] + t * (xs[i + 1] - xs[i]), ys[i] + t * (ys[i + 1] - ys[i]))
        cum += seg_len
    return (xs[-1], ys[-1])


def _flush_directional_asymmetry_summary() -> None:
    """Emit one summary line listing services with asymmetric directional frequency."""
    if _ASYMMETRIC_SERVICES:
        services = ", ".join(sorted(_ASYMMETRIC_SERVICES))
        print(f"Note: Directional frequency asymmetry detected for services: {services} (using max per segment).")
        _ASYMMETRIC_SERVICES.clear()


def _merge_bounds(
    *bounds_list: Optional[Tuple[float, float, float, float]]
) -> Optional[Tuple[float, float, float, float]]:
    """Combine multiple bounding boxes, ignoring missing entries."""
    min_x = math.inf
    max_x = -math.inf
    min_y = math.inf
    max_y = -math.inf

    has_value = False
    for bounds in bounds_list:
        if bounds is None:
            continue
        bx_min, bx_max, by_min, by_max = bounds
        min_x = min(min_x, bx_min)
        max_x = max(max_x, bx_max)
        min_y = min(min_y, by_min)
        max_y = max(max_y, by_max)
        has_value = True

    if not has_value:
        return None
    return min_x, max_x, min_y, max_y


def _parse_service_frequencies(cell: str, segment_label: str) -> Dict[str, float]:
    tokens = []
    for chunk in str(cell or "").replace(",", ";").split(";"):
        token = chunk.strip()
        if token:
            tokens.append(token)

    frequency_lists: Dict[str, List[float]] = {}
    for token in tokens:
        parts = token.split(".")
        if len(parts) < 2:
            continue
        freq_part = parts[-1]
        try:
            freq_value = float(freq_part)
        except ValueError:
            continue
        base_service = parts[0].strip()
        if not base_service:
            continue
        frequency_lists.setdefault(base_service, []).append(freq_value)

    result: Dict[str, float] = {}
    for service, freq_values in frequency_lists.items():
        if not freq_values:
            continue
        max_freq = max(freq_values)
        if any(abs(value - max_freq) > 1e-6 for value in freq_values):
            _ASYMMETRIC_SERVICES.add(service)
        result[service] = max_freq
    return result


def _station_polygon_points(
    center_x: float,
    center_y: float,
    axis_u: Tuple[float, float],
    axis_v: Tuple[float, float],
    along_half: float,
    across_half: float,
) -> List[Tuple[float, float]]:
    """Return the four vertex coordinates of an oriented station polygon."""
    corners = [
        (
            center_x + axis_u[0] * along_half + axis_v[0] * across_half,
            center_y + axis_u[1] * along_half + axis_v[1] * across_half,
        ),
        (
            center_x + axis_u[0] * along_half - axis_v[0] * across_half,
            center_y + axis_u[1] * along_half - axis_v[1] * across_half,
        ),
        (
            center_x - axis_u[0] * along_half - axis_v[0] * across_half,
            center_y - axis_u[1] * along_half - axis_v[1] * across_half,
        ),
        (
            center_x - axis_u[0] * along_half + axis_v[0] * across_half,
            center_y - axis_u[1] * along_half + axis_v[1] * across_half,
        ),
    ]
    return corners


def _compute_station_shapes(
    stations: Dict[int, Station],
    segments: Sequence[Segment],
) -> Dict[int, StationShape]:
    """Compute oriented station polygons sized to encompass service offsets.
    
    Accounts for dynamic SERVICE_OFFSET_SPACING that expands to prevent service collisions.
    """
    direction_angles: Dict[int, List[float]] = defaultdict(list)
    max_service_counts: Dict[int, int] = defaultdict(lambda: 1)
    max_frequencies: Dict[int, int] = defaultdict(lambda: 1)
    adaptive_spacings: Dict[int, float] = defaultdict(lambda: SERVICE_OFFSET_SPACING)

    for segment in segments:
        start = stations.get(segment.from_node)
        end = stations.get(segment.to_node)
        if start is None or end is None:
            continue
        dx = end.x - start.x
        dy = end.y - start.y
        if dx == 0.0 and dy == 0.0:
            angle = 0.0
        else:
            angle = math.atan2(dy, dx)
        direction_angles[segment.from_node].append(angle)
        direction_angles[segment.to_node].append(angle)

        service_map = _parse_service_frequencies(segment.services_tphpd, f"{segment.from_node}-{segment.to_node}")
        count = len(service_map)
        if count > 0:
            max_service_counts[segment.from_node] = max(max_service_counts[segment.from_node], count)
            max_service_counts[segment.to_node] = max(max_service_counts[segment.to_node], count)
            
            # Calculate adaptive spacing for this service set
            adaptive_spacing = _find_optimal_spacing(service_map)
            adaptive_spacings[segment.from_node] = max(adaptive_spacings[segment.from_node], adaptive_spacing)
            adaptive_spacings[segment.to_node] = max(adaptive_spacings[segment.to_node], adaptive_spacing)
            
            # Track maximum frequency at each station
            for freq in service_map.values():
                int_freq = max(int(round(freq)), 1)
                max_frequencies[segment.from_node] = max(max_frequencies[segment.from_node], int_freq)
                max_frequencies[segment.to_node] = max(max_frequencies[segment.to_node], int_freq)

    station_shapes: Dict[int, StationShape] = {}
    for node_id, station in stations.items():
        angles = direction_angles.get(node_id, [])
        if angles:
            sum_sin = sum(math.sin(2.0 * angle) for angle in angles)
            sum_cos = sum(math.cos(2.0 * angle) for angle in angles)
            if math.isclose(sum_sin, 0.0, abs_tol=1e-9) and math.isclose(sum_cos, 0.0, abs_tol=1e-9):
                orientation = 0.0
            else:
                orientation = 0.5 * math.atan2(sum_sin, sum_cos)
        else:
            orientation = 0.0

        axis_u = (math.cos(orientation), math.sin(orientation))
        axis_v = (-axis_u[1], axis_u[0])

        along_half = STATION_BASE_HALF_SIZE

        max_services = max_service_counts.get(node_id, 1)
        max_freq = max_frequencies.get(node_id, 1)
        adaptive_spacing = adaptive_spacings.get(node_id, SERVICE_OFFSET_SPACING)
        
        if max_services > 1:
            # Width from adaptive service spacing between different services
            service_spacing_width = adaptive_spacing * (max_services - 1) / 2.0
            # Additional width from frequency lines within the widest service
            frequency_width = SERVICE_FREQUENCY_SPACING * (max_freq - 1) / 2.0
            max_offset = service_spacing_width + frequency_width
        else:
            max_offset = 0.0
        across_half = max(along_half, STATION_BASE_HALF_SIZE + max_offset + SERVICE_RECT_MARGIN)

        polygon = _station_polygon_points(station.x, station.y, axis_u, axis_v, along_half, across_half)
        station_shapes[node_id] = StationShape(
            orientation=orientation,
            axis_u=axis_u,
            axis_v=axis_v,
            along_half=along_half,
            across_half=across_half,
            polygon=tuple(polygon),
        )

    return station_shapes


def _project_offset_to_segment_boundary(
    offset_point: Tuple[float, float],
    station: Station,
    segment_dir_unit: Tuple[float, float],
    boundary_distance: float = STATION_BASE_HALF_SIZE,
) -> Tuple[float, float]:
    """Project a service offset endpoint onto a virtual boundary perpendicular to
    the segment direction at boundary_distance from the station centre.

    Preserves the lateral spread (perpendicular to that specific segment) exactly,
    avoiding the collapse caused by polygon clamping when the station's averaged
    orientation differs from the individual segment direction.
    """
    ux, uy = segment_dir_unit
    length = math.hypot(ux, uy)
    if length <= 1e-9:
        return offset_point
    ux /= length
    uy /= length
    nx, ny = -uy, ux

    rel_x = offset_point[0] - station.x
    rel_y = offset_point[1] - station.y
    normal_component = rel_x * nx + rel_y * ny

    new_x = station.x + boundary_distance * ux + normal_component * nx
    new_y = station.y + boundary_distance * uy + normal_component * ny
    return (new_x, new_y)


def _service_line_width(frequency: float) -> float:
    if math.isnan(frequency) or frequency <= 0.0:
        return 1.0
    return max(1.0, 0.6 + 0.5 * frequency)


def _service_station_table(station: Station) -> Optional[str]:
    if not station.stopping_services:
        return None
    services = ", ".join(sorted(station.stopping_services))
    total_text = _format_freq(station.stopping_tphpd)
    return f"Stops: {services}\nTotal: {total_text} tphpd"


def _normalise_station_label(label: str) -> str:
    return " ".join(str(label or "").strip().lower().split())


def _build_name_lookup(stations: Dict[int, Station]) -> Dict[str, Station]:
    """Map normalised station names to station records."""
    lookup: Dict[str, Station] = {}
    for station in stations.values():
        key = _normalise_station_label(station.name)
        if key and key not in lookup:
            lookup[key] = station
    return lookup


def _load_capacity_sections(
    workbook_path: Optional[Path] = None,
    network_label: str = None,
    output_dir: Path = None
) -> List[SectionSummary]:
    """Load section summaries from the dedicated sections workbook.

    Supports both baseline and development networks:
    - Baseline: CAPACITY_DIR/<network>/*_sections.xlsx
    - Development: CAPACITY_DIR/developments/<dev_id>/*_sections.xlsx
    - Legacy: CAPACITY_DIR/*_sections.xlsx

    Args:
        workbook_path: Optional explicit path to sections workbook.
        network_label: Optional network label (e.g., "AK_2035_dev_101023").
        output_dir: Optional custom output directory.

    Returns:
        List of SectionSummary objects.
    """
    if workbook_path:
        workbook = Path(workbook_path)
    else:
        # Determine network tag
        import re
        if network_label is not None:
            network_tag = network_label
        else:
            network_tag = getattr(settings, "rail_network", "current")

        safe_network_tag = re.sub(r"[^\w-]+", "_", str(network_tag)).strip("_") or "current"
        expected_filename = f"capacity_{safe_network_tag}_network_sections.xlsx"

        # Auto-detect output directory for developments and enhanced
        if output_dir is None and network_label is not None:
            dev_match = re.search(r'_dev_(\d+)', network_label)
            is_enhanced = "_enhanced" in network_label

            if dev_match:
                # Development network
                dev_id = dev_match.group(1)
                from capacity_calculator import CAPACITY_ROOT
                output_dir = CAPACITY_ROOT / "Developments" / dev_id
            elif is_enhanced:
                # Enhanced network
                from capacity_calculator import CAPACITY_ROOT
                output_dir = CAPACITY_ROOT / "Enhanced" / safe_network_tag

        if output_dir is not None:
            # Development, Enhanced, or custom directory
            new_path = output_dir / expected_filename
        else:
            # Baseline: Try new structure first (Baseline subdirectory)
            network_subdir = CAPACITY_DIR / "Baseline" / safe_network_tag
            new_path = network_subdir / expected_filename

        # Fallback attempts
        tried_paths = [new_path]

        if new_path.exists():
            workbook = new_path
        else:
            # Fallback 1: Old structure with network subdirectory (no Baseline parent)
            fallback1 = CAPACITY_DIR / safe_network_tag / expected_filename
            tried_paths.append(fallback1)
            if fallback1.exists():
                workbook = fallback1
            else:
                # Fallback 2: Legacy flat structure (no subdirectory)
                fallback2 = CAPACITY_DIR / expected_filename
                tried_paths.append(fallback2)
                if fallback2.exists():
                    workbook = fallback2
                else:
                    raise FileNotFoundError(
                        f"Sections workbook not found. Tried:\n" +
                        "\n".join(f"  - {p}" for p in tried_paths)
                    )

    if not workbook.exists():
        raise FileNotFoundError(f"Sections workbook not found: {workbook}")

    xl = pd.ExcelFile(workbook)
    stn_sheet = "Stations" if "Stations" in xl.sheet_names else "Stations_Peak"
    sections_df = pd.read_excel(workbook, sheet_name="Sections")
    stations_df = pd.read_excel(workbook, sheet_name=stn_sheet)

    station_records = _to_stations(stations_df)
    name_lookup = _build_name_lookup(station_records)

    sections: List[SectionSummary] = []
    for row in sections_df.itertuples(index=False):
        start_name = _normalise_station_label(getattr(row, "start_station", ""))
        end_name = _normalise_station_label(getattr(row, "end_station", ""))
        if not start_name or not end_name:
            continue

        start_station = name_lookup.get(start_name)
        end_station = name_lookup.get(end_name)
        if start_station is None or end_station is None:
            continue

        section_id_value = getattr(row, "section_id", None)
        try:
            section_id = int(section_id_value) if section_id_value is not None else len(sections) + 1
        except (TypeError, ValueError):
            section_id = len(sections) + 1

        track_count = _coerce_number(getattr(row, "track_count", math.nan))
        total_tphpd = _coerce_number(
            getattr(row, "total_tphpd_peak", None) or getattr(row, "total_tphpd", math.nan)
        )
        capacity_tphpd = _coerce_number(
            getattr(row, "Capacity_peak", None) or getattr(row, "Capacity", math.nan)
        )
        utilization = _coerce_number(
            getattr(row, "Utilization_peak", None) or getattr(row, "Utilization", math.nan)
        )
        stopping_tphpd = _coerce_number(
            getattr(row, "stopping_tphpd_peak", None) or getattr(row, "stopping_tphpd", math.nan)
        )
        passing_tphpd = _coerce_number(
            getattr(row, "passing_tphpd_peak", None) or getattr(row, "passing_tphpd", math.nan)
        )

        # Intermediate stations along the section (between start and end), used
        # to route the line through real station geometry and to render faded
        # markers for stops that do not act as section boundaries.
        node_seq_raw = str(getattr(row, "node_sequence", "") or "")
        intermediate: List[Station] = []
        if node_seq_raw:
            tokens = [tok.strip() for tok in node_seq_raw.split("->") if tok.strip()]
            try:
                path_ids = [int(tok) for tok in tokens]
            except ValueError:
                path_ids = []
            for nid in path_ids[1:-1]:
                station = station_records.get(nid)
                if station is not None:
                    intermediate.append(station)

        sections.append(
            SectionSummary(
                section_id=section_id,
                track_count=track_count,
                start=start_station,
                end=end_station,
                total_tphpd=total_tphpd,
                capacity_tphpd=capacity_tphpd,
                utilization=utilization,
                stopping_tphpd=stopping_tphpd,
                passing_tphpd=passing_tphpd,
                intermediate_stations=tuple(intermediate),
            )
        )

    return sections


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _offset_line(coords: List[Tuple[float, float]], offset: float) -> Optional[List[Tuple[float, float]]]:
    """Return coordinates of a polyline offset by a perpendicular distance."""
    if LineString is None or not coords or len(coords) < 2:
        return None

    try:
        line = LineString(coords)
        if line.is_empty:
            return None
        side = "left" if offset >= 0 else "right"
        distance = abs(offset)
        if distance == 0.0:
            return coords
        offset_geom = line.parallel_offset(distance, side=side, resolution=2, join_style=2)
        if offset_geom.is_empty:
            return None
        if offset_geom.geom_type == "MultiLineString":
            offset_geom = max(offset_geom.geoms, key=lambda geom: geom.length)
        return [(float(x), float(y)) for x, y in offset_geom.coords]
    except Exception:
        return None


def _offset_polyline_uniform(coords: Sequence[Tuple[float, float]], offset: float) -> List[Tuple[float, float]]:
    """Return a consistently shifted copy of the polyline, keeping lines parallel."""
    if not coords:
        return []
    coords_list = [(float(x), float(y)) for x, y in coords]
    if len(coords_list) < 2 or math.isclose(offset, 0.0, abs_tol=1e-9):
        return coords_list

    offset_coords: List[Tuple[float, float]] = []
    total_points = len(coords_list)
    for index, (x, y) in enumerate(coords_list):
        if index == 0:
            nx, ny = coords_list[1]
            dir_x, dir_y = nx - x, ny - y
        elif index == total_points - 1:
            px, py = coords_list[-2]
            dir_x, dir_y = x - px, y - py
        else:
            px, py = coords_list[index - 1]
            nx, ny = coords_list[index + 1]
            dir_x = 0.0
            dir_y = 0.0
            seg1_x, seg1_y = x - px, y - py
            seg2_x, seg2_y = nx - x, ny - y
            length1 = math.hypot(seg1_x, seg1_y)
            length2 = math.hypot(seg2_x, seg2_y)
            if length1 > 0.0:
                dir_x += seg1_x / length1
                dir_y += seg1_y / length1
            if length2 > 0.0:
                dir_x += seg2_x / length2
                dir_y += seg2_y / length2
            if math.isclose(dir_x, 0.0, abs_tol=1e-9) and math.isclose(dir_y, 0.0, abs_tol=1e-9):
                dir_x, dir_y = seg2_x, seg2_y

        length = math.hypot(dir_x, dir_y)
        if length == 0.0:
            offset_coords.append((x, y))
            continue
        normal_x = -dir_y / length
        normal_y = dir_x / length
        offset_coords.append((x + normal_x * offset, y + normal_y * offset))

    return offset_coords


def _station_colour(station_tracks: float, connected_tracks: List[float]) -> str:
    """Return the fill colour for a station based on track availability."""
    if math.isnan(station_tracks):
        return "#bdbdbd"  # Grey when station track count is unknown.

    valid_segment_tracks = [value for value in connected_tracks if not math.isnan(value) and value > 0.0]
    if not valid_segment_tracks:
        return "#d2b48c"  # Light brown for isolated stations.

    if len(valid_segment_tracks) == 1:
        connected_equivalent = valid_segment_tracks[0]
    elif len(valid_segment_tracks) == 2:
        connected_equivalent = sum(valid_segment_tracks) / 2.0  # Through station
    else:
        connected_equivalent = sum(valid_segment_tracks)  # Junction (3+ segments)
    if station_tracks > connected_equivalent + 1e-6:
        return "#4caf50"  # Green - surplus capacity.
    if math.isclose(station_tracks, connected_equivalent, rel_tol=1e-6, abs_tol=1e-6):
        if math.isclose(station_tracks, 1.0, abs_tol=1e-6):
            return "#d73027"  # Red - single track, no crossing possible even when matched.
        return "#ffffff"  # White - matched capacity.
    return "#d73027"  # Red - constrained.


def _line_width(track_count: float) -> float:
    """Return the plotting linewidth for a segment."""
    baseline = max(0.75, 0.8 * 2.0)  # Preserve historic double-track width as the new single-track base.
    if math.isnan(track_count) or track_count <= 0.0:
        return baseline
    return baseline * max(track_count, 1.0)


def _segment_track_category(track_count: float) -> str:
    """Bucket track counts for legend labelling."""
    if math.isnan(track_count) or track_count <= 0.0:
        return "unknown"
    rounded = int(round(track_count))
    if rounded <= 1:
        return "single"
    if rounded == 2:
        return "double"
    return "multi"


def _format_track(track_value: float) -> str:
    if math.isnan(track_value):
        return "n/a"
    if math.isclose(track_value, round(track_value)):
        return str(int(round(track_value)))
    return f"{track_value:.1f}"


def _format_speed(speed_value: float) -> str:
    if math.isnan(speed_value) or speed_value <= 0:
        return "n/a"
    if math.isclose(speed_value, round(speed_value)):
        return f"{int(round(speed_value))} km/h"
    return f"{speed_value:.1f} km/h"


def _format_percentage(ratio: float) -> str:
    if math.isnan(ratio):
        return "n/a"
    return f"{ratio * 100:.1f}%"


def _format_freq(value: float) -> str:
    """Format a service frequency value as a floored integer; '0' when missing."""
    if math.isnan(value):
        return "0"
    return str(int(math.floor(value)))


def _segment_utilization(segment: Segment) -> float:
    """Return utilization ratio (demand/capacity) for a segment."""
    if not math.isnan(segment.utilization):
        return segment.utilization

    demand = segment.total_tphpd
    capacity = segment.capacity_tphpd
    if math.isnan(capacity) or capacity <= 0.0:
        capacity = segment.capacity_base_tphpd

    if math.isnan(demand) or math.isnan(capacity) or capacity <= 0.0:
        return math.nan

    return demand / capacity


def _format_minutes(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value:.1f} min"


def _format_length(value_m: float) -> str:
    if math.isnan(value_m):
        return "n/a"
    if value_m >= 1000.0:
        return f"{value_m / 1000.0:.2f} km"
    return f"{value_m:.0f} m"


def _estimate_text_extent(text: str, char_width: float = 55.0, line_height: float = 140.0) -> Tuple[float, float]:
    """Return an approximate width/height in data units for text annotations."""
    lines = text.splitlines() or [text]
    max_chars = max(len(line) for line in lines)
    width = max_chars * char_width
    height = len(lines) * line_height
    return width, height


def _bounds_from_anchor(
    x: float,
    y: float,
    width: float,
    height: float,
    anchor: str,
) -> Tuple[float, float, float, float]:
    """Convert an anchor position into bounding-box extents."""
    if anchor == "left_bottom":
        left, right = x, x + width
        bottom, top = y, y + height
    elif anchor == "left_center":
        left, right = x, x + width
        bottom, top = y - height / 2.0, y + height / 2.0
    elif anchor == "center":
        left, right = x - width / 2.0, x + width / 2.0
        bottom, top = y - height / 2.0, y + height / 2.0
    else:
        raise ValueError(f"Unknown anchor mode '{anchor}'")
    return left, right, bottom, top


def _overlaps(bounds: Tuple[float, float, float, float], existing: List[Tuple[float, float, float, float]]) -> bool:
    """Check whether a candidate bounding box intersects any existing boxes."""
    left, right, bottom, top = bounds
    for xmin, xmax, ymin, ymax in existing:
        if left <= xmax and right >= xmin and bottom <= ymax and top >= ymin:
            return True
    return False


def _find_label_position(
    existing: List[Tuple[float, float, float, float]],
    base_x: float,
    base_y: float,
    width: float,
    height: float,
    candidates: Iterable[Tuple[float, float]],
    anchor: str,
) -> Tuple[float, float]:
    """Return a collision-free label position given candidate offsets."""
    candidate_list = list(candidates)
    for dx, dy in candidate_list:
        candidate_x = base_x + dx
        candidate_y = base_y + dy
        bounds = _bounds_from_anchor(candidate_x, candidate_y, width, height, anchor)
        if not _overlaps(bounds, existing):
            return candidate_x, candidate_y
    # Fall back to a radial search that incrementally increases offset distance.
    step = max(width, height, 120.0)
    max_radius = step * 8.0
    radius = step
    directions = 16
    while radius <= max_radius:
        for idx in range(directions):
            angle = (2.0 * math.pi / directions) * idx
            candidate_x = base_x + radius * math.cos(angle)
            candidate_y = base_y + radius * math.sin(angle)
            bounds = _bounds_from_anchor(candidate_x, candidate_y, width, height, anchor)
            if not _overlaps(bounds, existing):
                return candidate_x, candidate_y
        radius += step
    # As a last resort, fall back to the final provided candidate even if overlapping.
    fallback_dx, fallback_dy = candidate_list[-1] if candidate_list else (0.0, 0.0)
    return base_x + fallback_dx, base_y + fallback_dy


def _find_section_geometry(
    start_id: int,
    end_id: int,
    segment_geometries: Dict[Tuple[int, int], List[Tuple[float, float]]],
    max_depth: int = 40,
) -> Optional[List[Tuple[float, float]]]:
    """BFS through the segment graph to stitch a polyline from start to end node.

    Returns None when no path is found or the segment_geometries dict is empty.
    """
    if not segment_geometries or start_id == end_id:
        return None

    adj: Dict[int, List[int]] = defaultdict(list)
    for a, b in segment_geometries:
        adj[a].append(b)
        adj[b].append(a)

    if start_id not in adj or end_id not in adj:
        return None

    from collections import deque
    queue: deque = deque([(start_id, [start_id])])
    visited: Set[int] = {start_id}

    while queue:
        node, path = queue.popleft()
        if len(path) > max_depth:
            continue
        for neighbor in adj[node]:
            if neighbor == end_id:
                full_path = path + [end_id]
                coords: List[Tuple[float, float]] = []
                for i in range(len(full_path) - 1):
                    a, b = full_path[i], full_path[i + 1]
                    key = _segment_key(a, b)
                    seg = segment_geometries.get(key, [])
                    if not seg:
                        return None
                    seg_coords = list(seg) if a <= b else list(reversed(seg))
                    coords.extend(seg_coords[1:] if coords else seg_coords)
                return coords if len(coords) >= 2 else None
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return None


def _load_boundary(boundary_path: str) -> Optional["GeoDataFrame"]:
    """Load any boundary polygon from a path relative to paths.MAIN."""
    if gpd is None:
        return None
    try:
        full_path = Path(paths.MAIN) / boundary_path
        if not full_path.exists():
            return None
        boundary = gpd.read_file(str(full_path))
        return boundary.to_crs(epsg=2056) if boundary.crs is not None else boundary.set_crs(epsg=2056)
    except Exception:
        return None


def _load_map_overlays(
    infra_version: Optional[str] = None,
    lakes_path: Optional[str] = None,
) -> Tuple[Optional["GeoDataFrame"], Dict[Tuple[int, int], List[Tuple[float, float]]]]:
    """Load optional GIS overlays used to enrich the network map.

    Loads pre-clipped lakes for background water features when *lakes_path* is
    provided (e.g. paths.LAKES_SA_GPKG or paths.LAKES_CA_GPKG). When
    *infra_version* is provided, also loads actual BAV LineString geometry from
    infrabuild segments.gpkg so segments are drawn as true curves rather than
    straight lines.

    Args:
        infra_version: Named infra version (e.g. 'AS_2026_ZH_enhanced'). When
            None, segment_geometries is empty and segments fall back to straight
            lines between station coordinate pairs.
        lakes_path: Relative path (from paths.MAIN) to a pre-clipped lakes
            GeoPackage (e.g. paths.LAKES_SA_GPKG). When None, no water layer
            is drawn.

    Returns:
        Tuple of (lakes GeoDataFrame or None, segment_geometries dict)
    """
    if gpd is None or make_valid is None:
        return None, {}

    # Load pre-clipped lakes when a path is provided
    lakes = None
    if lakes_path:
        _lakes_file = Path(paths.MAIN) / lakes_path
        if _lakes_file.exists():
            try:
                lakes = gpd.read_file(str(_lakes_file))
                if lakes.crs is None:
                    lakes = lakes.set_crs(epsg=2056)
                else:
                    lakes = lakes.to_crs(epsg=2056)
                if "geometry" in lakes.columns:
                    lakes["geometry"] = lakes["geometry"].apply(_safe_make_valid)
            except Exception:
                lakes = None

    segment_geometries: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}

    # Load actual BAV segment geometry when an infra version is available
    if infra_version and gpd is not None:
        try:
            import pandas as _pd
            from infrabuild_network_builder import load_version as _load_v
            _nodes_gdf, _segs_gdf = _load_v(infra_version)

            # Train-mode filter mirrors capacity_calculator.load_infra_data so the
            # BFS graph uses the same Numbers as the workbook. Without this, a
            # tram-mode duplicate of a shared station name (e.g. "Zürich HB")
            # would overwrite the train Number in the lookup and BFS would fail
            # to find routes between the workbook's train-side endpoints.
            if "Transport_Mode" in _segs_gdf.columns:
                _segs_gdf = _segs_gdf[
                    _segs_gdf["Transport_Mode"].astype(str).str.contains("train", case=False, na=False)
                ].copy()
            if "Transport_Mode" in _nodes_gdf.columns:
                _train_names_in_segs: Set[str] = set()
                if not _segs_gdf.empty:
                    _train_names_in_segs = (
                        set(_segs_gdf["From_Name"].dropna().astype(str))
                        | set(_segs_gdf["To_Name"].dropna().astype(str))
                    )
                _mode = _nodes_gdf["Transport_Mode"]
                _explicit = _mode.astype(str).str.contains("train", case=False, na=False)
                _unknown = _mode.isna() | (_mode.astype(str).str.strip() == "")
                _adjacent = pd.Series(False, index=_nodes_gdf.index)
                if "Name" in _nodes_gdf.columns and _train_names_in_segs:
                    _adjacent = _unknown & _nodes_gdf["Name"].astype(str).isin(_train_names_in_segs)
                _nodes_gdf = _nodes_gdf[_explicit | _adjacent].copy()

            # Build station name → Number lookup
            _name_to_nr: Dict[str, int] = {}
            for _, _nr in _nodes_gdf.iterrows():
                _num = _nr.get("Number")
                _nm  = _nr.get("Name", "")
                if _pd.notna(_num) and _nm:
                    _name_to_nr[str(_nm)] = int(float(_num))

            for _, _seg in _segs_gdf.iterrows():
                _fn = _name_to_nr.get(str(_seg.get("From_Name", "") or ""))
                _tn = _name_to_nr.get(str(_seg.get("To_Name",   "") or ""))
                if not _fn or not _tn:
                    continue
                _geom = _seg.geometry
                if _geom is None or _geom.is_empty:
                    continue
                try:
                    if _geom.geom_type == "MultiLineString":
                        _coords = list(_geom.geoms[0].coords)
                    else:
                        _coords = list(_geom.coords)
                    if _coords:
                        _key = tuple(sorted((_fn, _tn)))
                        segment_geometries[_key] = _coords
                except Exception:
                    continue
            print(f"  _load_map_overlays: {len(segment_geometries)} segment geometries "
                  f"loaded from '{infra_version}'")
        except Exception as _e:
            print(f"  [WARN] Could not load segment geometry from '{infra_version}': {_e}")

    return lakes, segment_geometries


def _draw_station_annotations(
    ax,
    stations: Dict[int, Station],
    segments: List[Segment],
    station_shapes: Optional[Dict[int, StationShape]] = None,
    marker_style: str = "square",
    include_tables: bool = True,
    include_labels: bool = True,
    marker_scale: float = 1.0,
    colour_mode: str = "status",
    uniform_colour: str = "#222222",
    table_text_func: Optional[Callable[[Station], Optional[str]]] = None,
) -> Tuple[Optional[Tuple[float, float, float, float]], Set[str]]:
    """Render station markers, codes, and optionally attribute tables."""
    connectivity: Dict[int, List[float]] = {}
    neighbor_map: Dict[int, List[int]] = defaultdict(list)
    for segment in segments:
        connectivity.setdefault(segment.from_node, []).append(segment.tracks)
        connectivity.setdefault(segment.to_node, []).append(segment.tracks)
        neighbor_map[segment.from_node].append(segment.to_node)
        neighbor_map[segment.to_node].append(segment.from_node)

    used_station_colours: Set[str] = set()
    annotation_boxes: List[Tuple[float, float, float, float]] = []
    extent_min_x, extent_max_x = math.inf, -math.inf
    extent_min_y, extent_max_y = math.inf, -math.inf
    fig = ax.figure
    marker_style = (marker_style or "square").lower()
    if stations:
        centroid_x = sum(station.x for station in stations.values()) / len(stations)
        centroid_y = sum(station.y for station in stations.values()) / len(stations)
    else:
        centroid_x = 0.0
        centroid_y = 0.0

    _JUNCTION_RADIUS = STATION_BASE_HALF_SIZE / 4.0 * marker_scale

    for node_id, station in stations.items():
        # Junctions: small dark-grey dot, no label, no table
        if station.is_junction:
            junc = Circle(
                (station.x, station.y),
                radius=_JUNCTION_RADIUS,
                facecolor="#555555",
                edgecolor="#333333",
                linewidth=0.5,
                zorder=3,
            )
            ax.add_patch(junc)
            extent_min_x = min(extent_min_x, station.x - _JUNCTION_RADIUS)
            extent_max_x = max(extent_max_x, station.x + _JUNCTION_RADIUS)
            extent_min_y = min(extent_min_y, station.y - _JUNCTION_RADIUS)
            extent_max_y = max(extent_max_y, station.y + _JUNCTION_RADIUS)
            continue

        if colour_mode == "status":
            colour = _station_colour(station.tracks, connectivity.get(node_id, []))
        elif colour_mode == "node_class":
            colour = _CA_NODE_COLOURS.get(station.node_class.lower(), _CA_NODE_DEFAULT)
        else:
            colour = uniform_colour
        used_station_colours.add(colour)

        neighbor_ids = neighbor_map.get(node_id, [])
        neighbor_vectors: List[Tuple[float, float]] = []
        for neighbor_id in neighbor_ids:
            neighbor = stations.get(neighbor_id)
            if neighbor is None:
                continue
            dx = neighbor.x - station.x
            dy = neighbor.y - station.y
            length = math.hypot(dx, dy)
            if length > 1e-6:
                neighbor_vectors.append((dx / length, dy / length))

        if neighbor_vectors:
            avg_dx = sum(vec[0] for vec in neighbor_vectors) / len(neighbor_vectors)
            avg_dy = sum(vec[1] for vec in neighbor_vectors) / len(neighbor_vectors)
            norm = math.hypot(avg_dx, avg_dy)
            if norm <= 1e-6:
                along_dir = neighbor_vectors[0]
            else:
                along_dir = (avg_dx / norm, avg_dy / norm)
        else:
            radial_dx = station.x - centroid_x
            radial_dy = station.y - centroid_y
            radial_norm = math.hypot(radial_dx, radial_dy)
            if radial_norm > 1e-6:
                along_dir = (radial_dx / radial_norm, radial_dy / radial_norm)
            else:
                along_dir = (1.0, 0.0)

        perp_dir = (-along_dir[1], along_dir[0])
        perp_norm = math.hypot(perp_dir[0], perp_dir[1])
        if perp_norm <= 1e-6:
            perp_dir = (0.0, 1.0)
        else:
            perp_dir = (perp_dir[0] / perp_norm, perp_dir[1] / perp_norm)

        centroid_vec_x = station.x - centroid_x
        centroid_vec_y = station.y - centroid_y
        if centroid_vec_x * perp_dir[0] + centroid_vec_y * perp_dir[1] < 0:
            perp_dir = (-perp_dir[0], -perp_dir[1])

        shape = station_shapes.get(node_id) if station_shapes else None
        if shape:
            patch = Polygon(shape.polygon, closed=True, facecolor=colour, edgecolor="black", linewidth=0.8, zorder=3)
            ax.add_patch(patch)
            poly_x = [pt[0] for pt in shape.polygon]
            poly_y = [pt[1] for pt in shape.polygon]
            extent_min_x = min(extent_min_x, min(poly_x))
            extent_max_x = max(extent_max_x, max(poly_x))
            extent_min_y = min(extent_min_y, min(poly_y))
            extent_max_y = max(extent_max_y, max(poly_y))
        else:
            half_size = STATION_BASE_HALF_SIZE * marker_scale
            if marker_style == "circle":
                marker = Circle(
                    (station.x, station.y),
                    radius=half_size,
                    facecolor=colour,
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=3,
                )
                ax.add_patch(marker)
                extent_min_x = min(extent_min_x, station.x - half_size)
                extent_max_x = max(extent_max_x, station.x + half_size)
                extent_min_y = min(extent_min_y, station.y - half_size)
                extent_max_y = max(extent_max_y, station.y + half_size)
            else:
                rect = Rectangle(
                    (station.x - half_size, station.y - half_size),
                    2 * half_size,
                    2 * half_size,
                    facecolor=colour,
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=3,
                )
                ax.add_patch(rect)
                extent_min_x = min(extent_min_x, station.x - half_size)
                extent_max_x = max(extent_max_x, station.x + half_size)
                extent_min_y = min(extent_min_y, station.y - half_size)
                extent_max_y = max(extent_max_y, station.y + half_size)

        if not include_labels:
            continue

        code_text = station.code or station.name or str(node_id)
        code_width, code_height = _estimate_text_extent(code_text, char_width=48.0, line_height=120.0)
        code_candidates = [
            (220.0, 240.0),
            (260.0, 320.0),
            (320.0, 200.0),
            (240.0, 140.0),
            (220.0, 60.0),
            (220.0, -120.0),
            (320.0, -200.0),
            (180.0, -240.0),
            (140.0, 260.0),
            (140.0, -260.0),
        ]
        code_x, code_y = _find_label_position(
            annotation_boxes,
            station.x,
            station.y,
            code_width,
            code_height,
            code_candidates,
            anchor="left_bottom",
        )
        code_artist = ax.text(
            code_x,
            code_y,
            code_text,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="bottom",
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85),
            zorder=4,
        )

        code_bounds = _bounds_from_anchor(code_x, code_y, code_width, code_height, "left_bottom")
        annotation_boxes.append(code_bounds)
        code_min_x, code_max_x, code_min_y, code_max_y = code_bounds

        extent_min_x = min(extent_min_x, code_min_x)
        extent_max_x = max(extent_max_x, code_max_x)
        extent_min_y = min(extent_min_y, code_min_y)
        extent_max_y = max(extent_max_y, code_max_y)

        if include_tables:
            if table_text_func:
                table_text = table_text_func(station)
            else:
                table_text = f"Platforms: {_format_track(station.platforms)}\nTracks: {_format_track(station.tracks)}"

            if table_text:
                table_width, table_height = _estimate_text_extent(table_text, char_width=52.0, line_height=120.0)
                # Primary position: right of the station name with a clear gap
                _TABLE_GAP = 700.0
                table_base_x = code_x + code_width + _TABLE_GAP
                table_base_y = code_y + code_height / 4.0
                table_candidates = [
                    (0.0, 0.0),                          # right of name — preferred
                    (0.0, code_height),                  # above
                    (0.0, -code_height),                 # below
                    (0.0, code_height * 1.5),
                    (0.0, -code_height * 1.5),
                    (table_width * 0.5, 0.0),
                    (table_width * 0.5, code_height),
                    (table_width * 0.5, -code_height),
                    (-table_width - 400.0, 0.0),         # left of name (fallback)
                    (0.0, code_height * 2.0),
                    (0.0, -code_height * 2.0),
                ]
                table_x, table_y = _find_label_position(
                    annotation_boxes,
                    table_base_x,
                    table_base_y,
                    table_width,
                    table_height,
                    table_candidates,
                    anchor="left_center",
                )
                table_artist = ax.text(
                    table_x,
                    table_y,
                    table_text,
                    fontsize=7,
                    fontfamily="monospace",
                    ha="left",
                    va="center",
                    color="#111111",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=0.6),
                    zorder=4,
                )

                table_bounds = _bounds_from_anchor(table_x, table_y, table_width, table_height, "left_center")
                annotation_boxes.append(table_bounds)
                table_min_x, table_max_x, table_min_y, table_max_y = table_bounds

                extent_min_x = min(extent_min_x, table_min_x)
                extent_max_x = max(extent_max_x, table_max_x)
                extent_min_y = min(extent_min_y, table_min_y)
                extent_max_y = max(extent_max_y, table_max_y)

    if extent_min_x == math.inf:
        return None, used_station_colours
    return (extent_min_x, extent_max_x, extent_min_y, extent_max_y), used_station_colours


def _draw_segments(
    ax,
    stations: Dict[int, Station],
    segments: List[Segment],
    segment_geometries: Optional[Dict[Tuple[int, int], List[Tuple[float, float]]]] = None,
    colour_by_track: bool = False,
    max_tracks: Optional[int] = None,
    track_spacing: float = 60.0,
) -> Tuple[Set[str], bool, Set[int]]:
    """Render network segments and return (track_categories, separators_used, track_counts_used).

    When colour_by_track is True segments are coloured by track count using
    _CA_TRACK_COLOURS, matching the infrabuild infrastructure visual style.
    max_tracks caps the number of parallel lines drawn (e.g. 4 means 5-track segments
    display as 4-track). track_spacing sets the lateral offset between parallel lines in
    LV95 metres.
    """
    track_categories: Set[str] = set()
    track_counts_used: Set[int] = set()
    separators_used = False
    for segment in segments:
        start = stations[segment.from_node]
        end = stations[segment.to_node]
        track_count = int(round(segment.tracks)) if not math.isnan(segment.tracks) else 0
        display_tracks = min(track_count, max_tracks) if (max_tracks is not None and track_count > 0) else track_count
        line_width = _line_width(display_tracks)
        track_categories.add(_segment_track_category(segment.tracks))
        track_counts_used.add(track_count)

        seg_colour = _CA_TRACK_COLOURS.get(track_count, _CA_TRACK_DEFAULT) if colour_by_track else "black"

        key = _segment_key(segment.from_node, segment.to_node)
        if segment_geometries and key in segment_geometries:
            coords = list(segment_geometries[key])
        elif segment_geometries:
            stitched = _find_section_geometry(segment.from_node, segment.to_node, segment_geometries)
            coords = stitched if stitched else [(start.x, start.y), (end.x, end.y)]
        else:
            coords = [(start.x, start.y), (end.x, end.y)]
        xs, ys = zip(*coords)

        # Check for fractional tracks (passing sidings)
        is_fractional = (segment.tracks % 1 == 0.5) if not math.isnan(segment.tracks) else False

        if is_fractional:
            base_tracks = math.floor(segment.tracks)

            if base_tracks == 1:
                # 1.5 tracks: Keep existing logic (looks good)
                # Draw full single-track line
                ax.plot(
                    xs,
                    ys,
                    color=seg_colour,
                    linewidth=_line_width(1),
                    zorder=2,
                )

                # Draw partial double-track section in middle 30% of segment
                line_geom = LineString(coords)
                total_length = line_geom.length

                if total_length > 0:
                    # Calculate middle 30% section (centered at 50%)
                    start_fraction = 0.35
                    end_fraction = 0.65

                    # Extract middle section coordinates
                    middle_start = line_geom.interpolate(start_fraction, normalized=True)
                    middle_end = line_geom.interpolate(end_fraction, normalized=True)

                    # Create LineString for middle section
                    middle_section = LineString([middle_start, middle_end])
                    middle_coords = list(middle_section.coords)
                    mx, my = zip(*middle_coords)

                    # Draw double-track width for passing siding section
                    double_line_width = _line_width(2)
                    ax.plot(
                        mx,
                        my,
                        color=seg_colour,
                        linewidth=double_line_width,
                        zorder=2.1,
                    )

                    # Draw separator for passing siding section
                    separator_width = max(0.6, double_line_width * 0.18)
                    ax.plot(
                        mx,
                        my,
                        color="white",
                        linewidth=separator_width,
                        zorder=2.5,
                    )
                    separators_used = True
            else:
                # 2.5, 3.5, 4.5+ tracks: Use new parallel line approach
                individual_line_width = line_width / (base_tracks * 1.3)

                # Step 1: Draw base tracks (full length)
                for track_idx in range(base_tracks):
                    offset_distance = (track_idx - (base_tracks - 1) / 2.0) * track_spacing
                    offset_coords = _offset_polyline_uniform(coords, offset_distance)

                    if offset_coords:
                        lx, ly = zip(*offset_coords)
                        ax.plot(
                            lx,
                            ly,
                            color=seg_colour,
                            linewidth=individual_line_width,
                            solid_capstyle="round",
                            zorder=2,
                        )

                # Step 2: Add two extra tracks in middle 30% (one on each edge)
                line_geom = LineString(coords)
                total_length = line_geom.length

                if total_length > 0:
                    # Calculate middle 30% section
                    start_fraction = 0.35
                    end_fraction = 0.65

                    middle_start = line_geom.interpolate(start_fraction, normalized=True)
                    middle_end = line_geom.interpolate(end_fraction, normalized=True)
                    middle_section = LineString([middle_start, middle_end])
                    middle_coords = list(middle_section.coords)

                    # Top edge extra track (skip one full spacing to avoid overlap)
                    top_offset = ((base_tracks / 2.0) + 0.5) * track_spacing
                    top_coords = _offset_polyline_uniform(middle_coords, top_offset)
                    if top_coords:
                        tx, ty = zip(*top_coords)
                        ax.plot(
                            tx,
                            ty,
                            color=seg_colour,
                            linewidth=individual_line_width,
                            solid_capstyle="round",
                            zorder=2,
                        )

                    # Bottom edge extra track (skip one full spacing to avoid overlap)
                    bottom_offset = -((base_tracks / 2.0) + 0.5) * track_spacing
                    bottom_coords = _offset_polyline_uniform(middle_coords, bottom_offset)
                    if bottom_coords:
                        bx, by = zip(*bottom_coords)
                        ax.plot(
                            bx,
                            by,
                            color=seg_colour,
                            linewidth=individual_line_width,
                            solid_capstyle="round",
                            zorder=2,
                        )
        else:
            # Integer tracks: Draw as separate parallel lines
            if track_count == 1:
                # Single track: draw one line at center
                ax.plot(
                    xs,
                    ys,
                    color=seg_colour,
                    linewidth=line_width,
                    zorder=2,
                )
            else:
                # Multiple tracks: draw N separate parallel lines with fixed spacing in data coordinates
                # Individual line width for visual appearance (thinner than total for visual separation)
                individual_line_width = line_width / (display_tracks * 1.3)

                for track_idx in range(display_tracks):
                    # Calculate offset in data coordinates (metres)
                    offset_distance = (track_idx - (display_tracks - 1) / 2.0) * track_spacing

                    # Use _offset_polyline_uniform (same as service plot)
                    offset_coords = _offset_polyline_uniform(coords, offset_distance)

                    if offset_coords:
                        lx, ly = zip(*offset_coords)
                        ax.plot(
                            lx,
                            ly,
                            color=seg_colour,
                            linewidth=individual_line_width,
                            solid_capstyle="round",
                            zorder=2,
                        )

    return track_categories, separators_used, track_counts_used


def _add_network_legends(
    ax,
    station_colours: Set[str],
    segment_categories: Set[str],
    separators_present: bool,
    is_catchment: bool = False,
    track_counts_used: Optional[Set[int]] = None,
) -> None:
    """Add station and segment legends to the plot.

    When is_catchment is True renders the infrabuild-style legend: segments
    coloured by track count, nodes coloured by class.
    """
    if is_catchment:
        _ca_node_labels = {
            _CA_NODE_COLOURS["station"]:           "Station",
            _CA_NODE_COLOURS["junction"]:          "Junction",
            _CA_NODE_COLOURS["abandoned_station"]: "Abandoned station",
            _CA_NODE_DEFAULT:                      "Other node",
        }
        node_handles = [
            Patch(facecolor=colour, edgecolor="black", linewidth=0.6, label=label)
            for colour, label in _ca_node_labels.items()
            if colour in station_colours
        ]
        node_legend = None
        if node_handles:
            node_legend = ax.legend(
                handles=node_handles,
                title="Node Class",
                loc="upper right",
                frameon=True,
                fontsize=8,
                title_fontsize=9,
            )
            node_legend.get_frame().set_facecolor("#f7f7f7")
            ax.add_artist(node_legend)

        _ca_track_labels: Dict[int, str] = {1: "Single track", 2: "Double track", 3: "3 tracks", 4: "4 tracks"}
        counts = sorted(track_counts_used or set())
        track_handles = []
        for n in counts:
            if n <= 0:
                continue
            colour = _CA_TRACK_COLOURS.get(n, _CA_TRACK_DEFAULT)
            label = _ca_track_labels.get(n, f"{n} tracks")
            track_handles.append(Line2D([0], [0], color=colour, linewidth=_line_width(n), label=label))
        if track_handles:
            track_legend = ax.legend(
                handles=track_handles,
                title="Track Count",
                loc="lower left",
                frameon=True,
                fontsize=8,
                title_fontsize=9,
            )
            track_legend.get_frame().set_facecolor("#f7f7f7")
            if node_legend is not None:
                ax.add_artist(track_legend)
        return

    station_definitions = {
        "#4caf50": "Crossing & overtaking possible",
        "#ffffff": "Crossing possible",
        "#d73027": "No crossing/overtaking possible",
        "#d2b48c": "Isolated station",
        "#bdbdbd": "Unknown tracks",
    }
    station_handles = [
        Patch(facecolor=colour, edgecolor="black", linewidth=0.6, label=label)
        for colour, label in station_definitions.items()
        if colour in station_colours
    ]

    station_legend = None
    if station_handles:
        station_legend = ax.legend(
            handles=station_handles,
            title="Station Status",
            loc="upper right",
            frameon=True,
            fontsize=8,
            title_fontsize=9,
        )
        station_legend.get_frame().set_facecolor("#f7f7f7")
        ax.add_artist(station_legend)

    segment_definitions = {
        "single": ("Single track", _line_width(1)),
        "double": ("Double track", _line_width(2)),
        "multi": ("≥3 tracks", _line_width(3)),
        "unknown": ("Unknown tracks", _line_width(1)),
    }
    segment_handles: List[object] = []
    segment_labels: List[str] = []
    handler_map: Dict[object, HandlerBase] = {}

    for key, (label, width) in segment_definitions.items():
        if key not in segment_categories:
            continue
        if key == "double":
            handle_obj = _DoubleTrackLegendHandle(width, 0)
            handler_map[_DoubleTrackLegendHandle] = _DoubleTrackLegendHandler()
        elif key == "multi":
            handle_obj = _MultiTrackLegendHandle(width, 3)
            handler_map[_MultiTrackLegendHandle] = _MultiTrackLegendHandler()
        else:
            handle_obj = Line2D([0], [0], color="black", linewidth=width)
        segment_handles.append(handle_obj)
        segment_labels.append(label)

    if segment_handles:
        segment_legend = ax.legend(
            handles=segment_handles,
            labels=segment_labels,
            title="Segment Tracks",
            loc="lower left",
            frameon=True,
            fontsize=8,
            title_fontsize=9,
            handler_map=handler_map,
        )
        segment_legend.get_frame().set_facecolor("#f7f7f7")
        if station_legend is not None:
            ax.add_artist(segment_legend)


def _draw_capacity_map(
    ax,
    sections: List[SectionSummary],
    include_annotations: bool = True,
) -> Tuple[Optional[Tuple[float, float, float, float]], Set[str]]:
    """Render a capacity utilization view of the network sections."""
    cmap = plt.get_cmap("RdYlGn_r")

    utilization_values = [section.utilization for section in sections if not math.isnan(section.utilization)]
    vmax_ratio = max([1.0] + utilization_values) if utilization_values else 1.0
    norm = Normalize(vmin=0.0, vmax=max(1.0, vmax_ratio))

    annotation_boxes: List[Tuple[float, float, float, float]] = []
    extent_min_x = math.inf
    extent_max_x = -math.inf
    extent_min_y = math.inf
    extent_max_y = -math.inf

    # Separate regular, junction, and intermediate scatter points for different
    # marker sizes. Intermediate stations are stops that fall inside a section
    # (not at its endpoints) — they get a faded marker.
    regular_points: Dict[int, Tuple[float, float]] = {}
    junction_points: Dict[int, Tuple[float, float]] = {}
    intermediate_points: Dict[int, Tuple[float, float]] = {}
    track_categories: Set[str] = set()

    for section in sections:
        start = section.start
        end = section.end

        # Route scatter points to regular or junction buckets.
        for station in (start, end):
            target = junction_points if station.is_junction else regular_points
            target[station.node_id] = (station.x, station.y)
        # Intermediate junctions still drive the polyline geometry, but are
        # not rendered as markers — only real stations get a faded dot.
        for station in section.intermediate_stations:
            if station.is_junction:
                continue
            intermediate_points[station.node_id] = (station.x, station.y)

        util_value = section.utilization
        track_categories.add(_segment_track_category(section.track_count))

        if math.isnan(util_value):
            colour = "#bdbdbd"
            zorder = 2
        else:
            clipped = min(util_value, norm.vmax)
            colour = cmap(norm(clipped))
            zorder = 3

        # Cap line width at 2-track maximum for capacity plot
        capped_track_count = min(section.track_count, 2.0) if not math.isnan(section.track_count) else section.track_count

        # Route the section polyline through intermediate stations when present
        # to approximate the real track geometry. Label anchor follows the
        # polyline midpoint (by arc length) so utilisation/info boxes stay
        # near the centre of the rendered line, not the straight start↔end line.
        path_stations = [start, *section.intermediate_stations, end]
        xs = [s.x for s in path_stations]
        ys = [s.y for s in path_stations]
        mid_x, mid_y = _polyline_midpoint(xs, ys)

        ax.plot(
            xs,
            ys,
            color=colour,
            linewidth=_line_width(capped_track_count),
            solid_capstyle="round",
            zorder=zorder,
        )

        # Skip text annotations when either endpoint is a junction or annotations are suppressed.
        has_junction_endpoint = start.is_junction or end.is_junction
        if math.isnan(util_value) or not include_annotations or has_junction_endpoint:
            continue

        percent_text = _format_percentage(util_value)
        percent_width, percent_height = _estimate_text_extent(percent_text, char_width=50.0, line_height=120.0)
        percent_candidates = [
            (0.0, 0.0),
            (0.0, 200.0),
            (0.0, -200.0),
            (200.0, 0.0),
            (-200.0, 0.0),
            (200.0, 200.0),
            (200.0, -200.0),
            (-200.0, 200.0),
            (-200.0, -200.0),
        ]
        percent_x, percent_y = _find_label_position(
            annotation_boxes,
            mid_x,
            mid_y,
            percent_width,
            percent_height,
            percent_candidates,
            anchor="center",
        )

        ax.text(
            percent_x,
            percent_y,
            percent_text,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=0.6),
            zorder=4,
        )

        p_bounds = _bounds_from_anchor(percent_x, percent_y, percent_width, percent_height, "center")
        annotation_boxes.append(p_bounds)
        p_min_x, p_max_x, p_min_y, p_max_y = p_bounds

        extent_min_x = min(extent_min_x, p_min_x)
        extent_max_x = max(extent_max_x, p_max_x)
        extent_min_y = min(extent_min_y, p_min_y)
        extent_max_y = max(extent_max_y, p_max_y)

        # Round fractional (.5) values down to the nearest integer for display.
        def _fmt_int(v: float) -> str:
            return _format_track(math.floor(v) if not math.isnan(v) else v)

        detail_text = (
            f"Total: {_fmt_int(section.total_tphpd)} / {_fmt_int(section.capacity_tphpd)} tphpd\n"
            f"Local: {_fmt_int(section.stopping_tphpd)}\n"
            f"Express: {_fmt_int(section.passing_tphpd)}"
        )
        detail_width, detail_height = _estimate_text_extent(detail_text, char_width=55.0, line_height=120.0)
        # Increased gap between % badge and detail table.
        _DETAIL_GAP = 600.0
        detail_base_x = p_max_x
        detail_base_y = (p_min_y + p_max_y) / 2.0
        detail_candidates = [
            (_DETAIL_GAP, 0.0),
            (_DETAIL_GAP, 200.0),
            (_DETAIL_GAP, -200.0),
            (_DETAIL_GAP * 2, 0.0),
            (_DETAIL_GAP * 2, 200.0),
            (_DETAIL_GAP * 2, -200.0),
            (-_DETAIL_GAP, 0.0),
            (-_DETAIL_GAP, 200.0),
            (-_DETAIL_GAP, -200.0),
        ]
        detail_x, detail_y = _find_label_position(
            annotation_boxes,
            detail_base_x,
            detail_base_y,
            detail_width,
            detail_height,
            detail_candidates,
            anchor="left_center",
        )

        ax.text(
            detail_x,
            detail_y,
            detail_text,
            fontsize=7,
            fontfamily="monospace",
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=0.6),
            zorder=4,
        )

        d_bounds = _bounds_from_anchor(detail_x, detail_y, detail_width, detail_height, "left_center")
        annotation_boxes.append(d_bounds)
        d_min_x, d_max_x, d_min_y, d_max_y = d_bounds

        extent_min_x = min(extent_min_x, d_min_x)
        extent_max_x = max(extent_max_x, d_max_x)
        extent_min_y = min(extent_min_y, d_min_y)
        extent_max_y = max(extent_max_y, d_max_y)

    # Intermediate-only stations are those that never act as a section endpoint;
    # render them faded so users can see which stops a section passes through
    # without offering passing opportunities.
    for nid in list(intermediate_points.keys()):
        if nid in regular_points or nid in junction_points:
            del intermediate_points[nid]
    if intermediate_points:
        ixs, iys = zip(*intermediate_points.values())
        ax.scatter(ixs, iys, s=45, c="#bbbbbb", edgecolors="white", linewidths=0.4, alpha=0.6, zorder=4)

    if regular_points:
        rxs, rys = zip(*regular_points.values())
        ax.scatter(rxs, rys, s=45, c="#222222", edgecolors="white", linewidths=0.4, zorder=5)
    if junction_points:
        jxs, jys = zip(*junction_points.values())
        ax.scatter(jxs, jys, s=20, c="#444444", edgecolors="white", linewidths=0.3, zorder=5)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Utilization (%)")
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    if extent_min_x == math.inf:
        return None, track_categories
    return (extent_min_x, extent_max_x, extent_min_y, extent_max_y), track_categories


def _draw_speed_profile(
    ax,
    stations: Dict[int, Station],
    segments: List[Segment],
    segment_geometries: Optional[Dict[Tuple[int, int], List[Tuple[float, float]]]] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """Render segment speeds with colour-coded lines and annotations."""
    cmap = plt.get_cmap("plasma")

    speed_values = [segment.speed for segment in segments if not math.isnan(segment.speed) and segment.speed > 0.0]
    if speed_values:
        min_speed = min(speed_values)
        max_speed = max(speed_values)
        if math.isclose(min_speed, max_speed):
            min_speed = max(0.0, min_speed - 5.0)
            max_speed = max_speed + 5.0
        norm = Normalize(vmin=min_speed, vmax=max_speed)
    else:
        norm = Normalize(vmin=0.0, vmax=1.0)

    annotation_boxes: List[Tuple[float, float, float, float]] = []
    extent_min_x = math.inf
    extent_max_x = -math.inf
    extent_min_y = math.inf
    extent_max_y = -math.inf

    for segment in segments:
        start = stations[segment.from_node]
        end = stations[segment.to_node]

        key = _segment_key(segment.from_node, segment.to_node)
        if segment_geometries and key in segment_geometries:
            coords = list(segment_geometries[key])
        elif segment_geometries:
            stitched = _find_section_geometry(segment.from_node, segment.to_node, segment_geometries)
            coords = stitched if stitched else [(start.x, start.y), (end.x, end.y)]
        else:
            coords = [(start.x, start.y), (end.x, end.y)]
        xs, ys = zip(*coords)

        speed_value = segment.speed
        if math.isnan(speed_value) or speed_value <= 0.0:
            colour = "#bdbdbd"
        else:
            colour = cmap(norm(speed_value))

        # Cap line width at 2-track maximum for speed plot
        capped_track_count = min(segment.tracks, 2.0) if not math.isnan(segment.tracks) else segment.tracks
        ax.plot(
            xs,
            ys,
            color=colour,
            linewidth=_line_width(capped_track_count),
            solid_capstyle="round",
            zorder=2,
        )

        if math.isnan(speed_value) or speed_value <= 0.0:
            continue

        mid_x = (start.x + end.x) / 2.0
        mid_y = (start.y + end.y) / 2.0

        speed_text = _format_speed(speed_value)
        speed_width, speed_height = _estimate_text_extent(speed_text, char_width=50.0, line_height=120.0)
        speed_candidates = [
            (0.0, 0.0),
            (0.0, 200.0),
            (0.0, -200.0),
            (200.0, 0.0),
            (-200.0, 0.0),
            (200.0, 200.0),
            (200.0, -200.0),
            (-200.0, 200.0),
            (-200.0, -200.0),
        ]
        speed_x, speed_y = _find_label_position(
            annotation_boxes,
            mid_x,
            mid_y,
            speed_width,
            speed_height,
            speed_candidates,
            anchor="center",
        )

        speed_artist = ax.text(
            speed_x,
            speed_y,
            speed_text,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=0.6),
            zorder=4,
        )

        s_bounds = _bounds_from_anchor(speed_x, speed_y, speed_width, speed_height, "center")
        annotation_boxes.append(s_bounds)
        s_min_x, s_max_x, s_min_y, s_max_y = s_bounds

        extent_min_x = min(extent_min_x, s_min_x)
        extent_max_x = max(extent_max_x, s_max_x)
        extent_min_y = min(extent_min_y, s_min_y)
        extent_max_y = max(extent_max_y, s_max_y)

        detail_text = (
            f"TT Local: {_format_minutes(segment.travel_time_stopping)}\n"
            f"TT Express: {_format_minutes(segment.travel_time_passing)}\n"
            f"Length: {_format_length(segment.length_m)}"
        )
        detail_width, detail_height = _estimate_text_extent(detail_text, char_width=55.0, line_height=120.0)
        detail_base_x = s_max_x
        detail_base_y = (s_min_y + s_max_y) / 2.0
        detail_candidates = [
            (180.0, 0.0),
            (180.0, 200.0),
            (180.0, -200.0),
            (360.0, 0.0),
            (360.0, 200.0),
            (360.0, -200.0),
            (-180.0, 0.0),
            (-180.0, 200.0),
            (-180.0, -200.0),
        ]
        detail_x, detail_y = _find_label_position(
            annotation_boxes,
            detail_base_x,
            detail_base_y,
            detail_width,
            detail_height,
            detail_candidates,
            anchor="left_center",
        )

        detail_artist = ax.text(
            detail_x,
            detail_y,
            detail_text,
            fontsize=7,
            fontfamily="monospace",
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=0.6),
            zorder=4,
        )

        d_bounds = _bounds_from_anchor(detail_x, detail_y, detail_width, detail_height, "left_center")
        annotation_boxes.append(d_bounds)
        d_min_x, d_max_x, d_min_y, d_max_y = d_bounds

        extent_min_x = min(extent_min_x, d_min_x)
        extent_max_x = max(extent_max_x, d_max_x)
        extent_min_y = min(extent_min_y, d_min_y)
        extent_max_y = max(extent_max_y, d_max_y)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Max speed (km/h)")

    if extent_min_x == math.inf:
        return None
    return extent_min_x, extent_max_x, extent_min_y, extent_max_y


def _draw_service_map(
    ax,
    stations: Dict[int, Station],
    segments: List[Segment],
    segment_geometries: Optional[Dict[Tuple[int, int], List[Tuple[float, float]]]] = None,
    station_shapes: Optional[Dict[int, StationShape]] = None,
    include_service_labels: bool = True,
) -> Optional[Tuple[float, float, float, float]]:
    """Render service frequencies with coloured lines and service labels."""
    annotation_boxes: List[Tuple[float, float, float, float]] = []
    extent_min_x = math.inf
    extent_max_x = -math.inf
    extent_min_y = math.inf
    extent_max_y = -math.inf

    service_order: Dict[str, int] = {}
    service_station_links: Dict[Tuple[str, int], List[Tuple[Tuple[float, float], str, float]]] = defaultdict(list)
    label_candidates: List[Dict[str, Any]] = []

    for segment in segments:
        service_map = _parse_service_frequencies(segment.services_tphpd, f"{segment.from_node}-{segment.to_node}")
        if not service_map:
            continue

        for service_name in service_map:
            if service_name not in service_order:
                service_order[service_name] = len(service_order)

        start = stations[segment.from_node]
        end = stations[segment.to_node]
        key = _segment_key(segment.from_node, segment.to_node)
        if segment_geometries and key in segment_geometries:
            base_coords = list(segment_geometries[key])
            canonical_from, _ = key
            if segment.from_node != canonical_from:
                coords = list(reversed(base_coords))
            else:
                coords = list(base_coords)
        elif segment_geometries:
            stitched = _find_section_geometry(segment.from_node, segment.to_node, segment_geometries)
            coords = stitched if stitched else [(start.x, start.y), (end.x, end.y)]
        else:
            coords = [(start.x, start.y), (end.x, end.y)]

        services_sorted = sorted(
            service_map.items(),
            key=lambda kv: service_order.get(kv[0], float("inf")),
        )
        service_count = len(services_sorted)

        # Calculate optimal spacing to avoid service collisions
        adaptive_offset_spacing = _find_optimal_spacing(service_map)

        start_shape = station_shapes.get(segment.from_node) if station_shapes else None

        base_sign = 1.0
        if coords:
            if len(coords) >= 2:
                first_point = coords[0]
                next_point = coords[1]
                dir_x = next_point[0] - first_point[0]
                dir_y = next_point[1] - first_point[1]
            else:
                dir_x = end.x - start.x
                dir_y = end.y - start.y
            normal_x = -dir_y
            normal_y = dir_x
            normal_length = math.hypot(normal_x, normal_y)
            if normal_length > 0.0:
                normal_x /= normal_length
                normal_y /= normal_length
            else:
                normal_x, normal_y = 0.0, 1.0

            if start_shape:
                dot = normal_x * start_shape.axis_v[0] + normal_y * start_shape.axis_v[1]
                if dot < 0.0:
                    base_sign = -1.0
            else:
                base_sign = 1.0 if normal_y >= 0.0 else -1.0

        for index, (service_name, frequency) in enumerate(services_sorted):
            start_stops = service_name in start.stopping_services
            end_stops = service_name in end.stopping_services
            start_colour = SERVICE_COLOUR_STOP if start_stops else SERVICE_COLOUR_PASS
            end_colour = SERVICE_COLOUR_STOP if end_stops else SERVICE_COLOUR_PASS

            if service_count <= 1:
                offset_distance = 0.0
            else:
                offset_distance = base_sign * (index - (service_count - 1) / 2.0) * adaptive_offset_spacing

            offset_coords = _offset_polyline_uniform(coords, offset_distance)

            if len(offset_coords) >= 2:
                start_dir = (
                    offset_coords[1][0] - offset_coords[0][0],
                    offset_coords[1][1] - offset_coords[0][1],
                )
                end_dir = (
                    offset_coords[-2][0] - offset_coords[-1][0],
                    offset_coords[-2][1] - offset_coords[-1][1],
                )
                offset_coords[0] = _project_offset_to_segment_boundary(offset_coords[0], start, start_dir)
                offset_coords[-1] = _project_offset_to_segment_boundary(offset_coords[-1], end, end_dir)

            start_offset = offset_coords[0]
            end_offset = offset_coords[-1]
            mid_x = (start_offset[0] + end_offset[0]) / 2.0
            mid_y = (start_offset[1] + end_offset[1]) / 2.0

            # Round frequency to integer (1 service = 1 line, 2 services = 2 lines, etc.)
            int_frequency = max(int(round(frequency)), 1)

            # Fixed uniform line width for all service frequency lines
            individual_line_width = 1.2

            # Draw parallel lines for each frequency unit
            for freq_idx in range(int_frequency):
                # Calculate offset for this frequency line relative to the service's base offset
                freq_offset = (freq_idx - (int_frequency - 1) / 2.0) * SERVICE_FREQUENCY_SPACING

                # Apply frequency offset to the service's base offset coordinates
                freq_coords = _offset_polyline_uniform(offset_coords, freq_offset)

                if len(freq_coords) >= 2:
                    freq_start_dir = (
                        freq_coords[1][0] - freq_coords[0][0],
                        freq_coords[1][1] - freq_coords[0][1],
                    )
                    freq_end_dir = (
                        freq_coords[-2][0] - freq_coords[-1][0],
                        freq_coords[-2][1] - freq_coords[-1][1],
                    )
                    freq_coords[0] = _project_offset_to_segment_boundary(freq_coords[0], start, freq_start_dir)
                    freq_coords[-1] = _project_offset_to_segment_boundary(freq_coords[-1], end, freq_end_dir)

                # Update extent tracking
                xs_all = [pt[0] for pt in freq_coords]
                ys_all = [pt[1] for pt in freq_coords]
                extent_min_x = min(extent_min_x, min(xs_all))
                extent_max_x = max(extent_max_x, max(xs_all))
                extent_min_y = min(extent_min_y, min(ys_all))
                extent_max_y = max(extent_max_y, max(ys_all))

                # Draw this frequency line with uniform color or split coloring
                freq_start = freq_coords[0]
                freq_end = freq_coords[-1]
                freq_mid_x = (freq_start[0] + freq_end[0]) / 2.0
                freq_mid_y = (freq_start[1] + freq_end[1]) / 2.0

                if start_colour == end_colour:
                    # Uniform color for entire line
                    fx, fy = zip(*freq_coords)
                    ax.plot(
                        fx, fy,
                        color=start_colour,
                        linewidth=individual_line_width,
                        solid_capstyle="round",
                        zorder=2
                    )
                else:
                    # Split coloring at geometric midpoint
                    ax.plot(
                        [freq_start[0], freq_mid_x],
                        [freq_start[1], freq_mid_y],
                        color=start_colour,
                        linewidth=individual_line_width,
                        solid_capstyle="round",
                        zorder=2
                    )
                    ax.plot(
                        [freq_mid_x, freq_end[0]],
                        [freq_mid_y, freq_end[1]],
                        color=end_colour,
                        linewidth=individual_line_width,
                        solid_capstyle="round",
                        zorder=2
                    )

            # Store service station links for connector lines (using center offset and fixed width)
            service_station_links[(service_name, segment.from_node)].append((start_offset, start_colour, 1.0))
            service_station_links[(service_name, segment.to_node)].append((end_offset, end_colour, 1.0))

            label_candidates.append(
                {
                    "service": service_name,
                    "mid_x": mid_x,
                    "mid_y": mid_y,
                    "start_node": segment.from_node,
                    "end_node": segment.to_node,
                }
            )

    for (service_name, station_id), endpoints in service_station_links.items():
        if len(endpoints) != 2:
            continue
        (point_a, colour_a, _), (point_b, colour_b, _) = endpoints
        station = stations.get(station_id)
        if station is None:
            continue
        if colour_a == colour_b:
            connector_colour = colour_a
        elif service_name in station.stopping_services:
            connector_colour = SERVICE_COLOUR_STOP
        else:
            connector_colour = SERVICE_COLOUR_PASS
        connector_width = 1.0  # Fixed thin width matching service lines
        ax.plot(
            [point_a[0], point_b[0]],
            [point_a[1], point_b[1]],
            color=connector_colour,
            linewidth=connector_width,
            zorder=3.2,
        )

    endpoint_stations: Set[Tuple[str, int]] = {
        key for key, touches in service_station_links.items() if len(touches) <= 1
    }

    if not include_service_labels:
        labels_to_draw: List[Dict[str, Any]] = []
    else:
        labels_to_draw = []
        labelled_services: Set[str] = set()
        # First pass: keep current behaviour — tag at endpoint segments.
        for candidate in label_candidates:
            service_name = candidate["service"]
            if (service_name, candidate["start_node"]) in endpoint_stations or (
                service_name, candidate["end_node"]
            ) in endpoint_stations:
                labels_to_draw.append(candidate)
                labelled_services.add(service_name)
        # Second pass: services that only run through interior segments get
        # one tag placed on the first interior segment we encountered.
        for candidate in label_candidates:
            service_name = candidate["service"]
            if service_name in labelled_services:
                continue
            labels_to_draw.append(candidate)
            labelled_services.add(service_name)

    for candidate in labels_to_draw:
        service_name = candidate["service"]
        service_text = service_name
        service_width, service_height = _estimate_text_extent(
            service_text, char_width=50.0, line_height=120.0
        )
        service_candidates = [
            (0.0, 350.0),
            (0.0, -350.0),
            (350.0, 0.0),
            (-350.0, 0.0),
            (350.0, 350.0),
            (350.0, -350.0),
            (-350.0, 350.0),
            (-350.0, -350.0),
            (0.0, 600.0),
            (0.0, -600.0),
        ]
        label_x, label_y = _find_label_position(
            annotation_boxes,
            candidate["mid_x"],
            candidate["mid_y"],
            service_width,
            service_height,
            service_candidates,
            anchor="center",
        )

        label_artist = ax.text(
            label_x,
            label_y,
            service_text,
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", linewidth=0.6),
            zorder=4,
        )

        l_bounds = _bounds_from_anchor(label_x, label_y, service_width, service_height, "center")
        annotation_boxes.append(l_bounds)
        l_min_x, l_max_x, l_min_y, l_max_y = l_bounds

        extent_min_x = min(extent_min_x, l_min_x)
        extent_max_x = max(extent_max_x, l_max_x)
        extent_min_y = min(extent_min_y, l_min_y)
        extent_max_y = max(extent_max_y, l_max_y)

    if extent_min_x == math.inf:
        return None
    return extent_min_x, extent_max_x, extent_min_y, extent_max_y


def _format_plot_title(base_title: str, network_label: str = None) -> str:
    """Format plot title to include development ID if applicable.

    Args:
        base_title: Base title (e.g., "Rail Network Infrastructure")
        network_label: Optional network label (e.g., "AK_2035_dev_100023")

    Returns:
        Formatted title with network information.
    """
    if network_label is None:
        network_tag = getattr(settings, "rail_network", "")
    else:
        network_tag = network_label

    if not network_tag:
        return base_title

    # Check if development network
    import re
    dev_match = re.search(r'_dev_(\d+)', network_tag)
    if dev_match:
        dev_id = dev_match.group(1)
        # Extract base network name (e.g., "AK_2035" from "AK_2035_dev_100023")
        base_network = network_tag.split('_dev_')[0]
        return f"{base_title} - {base_network} Development {dev_id}"
    else:
        # Baseline network
        return f"{base_title} - {network_tag}"


def _configure_axes(
    ax,
    stations: Dict[int, Station],
    title: str = "Infrastructure Overview",
    annotation_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    """Set axis styling and limits."""
    xs = [station.x for station in stations.values()]
    ys = [station.y for station in stations.values()]
    if not xs or not ys:
        return

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    if annotation_bounds:
        ann_min_x, ann_max_x, ann_min_y, ann_max_y = annotation_bounds
        min_x = min(min_x, ann_min_x)
        max_x = max(max_x, ann_max_x)
        min_y = min(min_y, ann_min_y)
        max_y = max(max_y, ann_max_y)

    padding_x = max(750.0, 0.08 * (max_x - min_x))
    padding_y = max(750.0, 0.08 * (max_y - min_y))
    ax.set_xlim(min_x - padding_x, max_x + padding_x)
    ax.set_ylim(min_y - padding_y, max_y + padding_y)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting (LV95)")
    ax.set_ylabel("Northing (LV95)")
    ax.grid(False)
    ax.set_title(title, fontsize=14, fontweight="bold")


# ---------------------------------------------------------------------------
# Execution entry points
# ---------------------------------------------------------------------------

def network_current_map(
    workbook_path: Optional[str] = None,
    output_path: Optional[str] = None,
    show: bool = False,
    *,
    stations_df: Optional[pd.DataFrame] = None,
    segments_df: Optional[pd.DataFrame] = None,
    stations: Optional[Dict[int, Station]] = None,
    segments_list: Optional[List[Segment]] = None,
    return_figure: bool = False,
    network_label: str = None,
    output_dir: Path = None,
    infra_version: Optional[str] = None,
    lakes_path: Optional[str] = None,
    include_labels: bool = True,
    allowed_node_classes: Optional[Set[str]] = None,
    is_catchment: bool = False,
    marker_scale: float = 1.5,
    boundary_path: Optional[str] = None,
) -> Union[Path, Tuple[Path, Figure]]:
    """Render the current network infrastructure map.

    Args:
        workbook_path: Optional explicit path to workbook.
        output_path: Optional path for output file.
        show: If True, display the plot.
        stations_df: Optional preloaded stations DataFrame.
        segments_df: Optional preloaded segments DataFrame.
        stations: Optional preloaded stations dict.
        segments_list: Optional preloaded segments list.
        return_figure: If True, return both path and figure.
        network_label: Optional custom network label (e.g., "AK_2035_dev_100023").
        output_dir: Optional custom output directory.
        is_catchment: When True renders with infrabuild-style colours (segments by
            track count, nodes by class) suited to wide-area catchment maps.

    Returns:
        Path to saved image, or (Path, Figure) if return_figure=True.
    """
    if stations is None or segments_list is None:
        if stations_df is None or segments_df is None:
            stations_df, segments_df = _load_workbook(
                workbook_path=Path(workbook_path) if workbook_path else None,
                network_label=network_label,
                output_dir=output_dir
            )
        stations = _filter_stations_by_class(_to_stations(stations_df), allowed_node_classes)
        segments_list = _to_segments(segments_df, stations.keys())

    if not stations:
        raise ValueError("No stations were found in the prep workbook.")
    if not segments_list:
        raise ValueError("No segments were found linking the stations.")

    water_layer, segment_geometries = _load_map_overlays(infra_version, lakes_path=lakes_path)

    figsize = _calculate_figure_size(stations)
    fig, ax = plt.subplots(figsize=figsize)
    if water_layer is not None and not getattr(water_layer, "empty", True):
        try:
            water_layer.plot(ax=ax, color="#b7d4f0", edgecolor="#6ea3d5", linewidth=0.5, zorder=1)
        except Exception:
            pass

    # Ghost pass: infra segments whose endpoints are not both in the workbook (CA only)
    if is_catchment and segment_geometries:
        workbook_node_set = set(stations.keys())
        for (a, b), coords in segment_geometries.items():
            if a in workbook_node_set and b in workbook_node_set:
                continue
            xs, ys = zip(*coords)
            ax.plot(xs, ys, color="#555555", linewidth=1.2, alpha=0.40, zorder=1.5,
                    solid_capstyle="round")

    seg_kwargs: Dict = {"segment_geometries": segment_geometries}
    if is_catchment:
        seg_kwargs["max_tracks"] = 4
        seg_kwargs["track_spacing"] = 140.0
    segment_categories, separators_used, _ = _draw_segments(
        ax, stations, segments_list, **seg_kwargs,
    )
    annotation_bounds, station_colours = _draw_station_annotations(
        ax,
        stations,
        segments_list,
        marker_style="circle",
        include_labels=include_labels,
        include_tables=(not is_catchment),
        marker_scale=marker_scale,
        colour_mode="status",
    )

    plot_title = _format_plot_title("Rail Network Infrastructure", network_label)
    _configure_axes(ax, stations, title=plot_title, annotation_bounds=annotation_bounds)

    if boundary_path:
        _boundary_gdf = _load_boundary(boundary_path)
        if _boundary_gdf is not None and not getattr(_boundary_gdf, "empty", True):
            try:
                _boundary_gdf.boundary.plot(
                    ax=ax, color="#333333", linewidth=1.0, linestyle="--", alpha=0.6, zorder=5
                )
            except Exception:
                pass

    _add_network_legends(
        ax, station_colours, segment_categories, separators_used,
    )
    _add_north_arrow(ax)
    _add_scale_bar(ax)

    base_output = _derive_plot_output_path(
        plot_type="infrastructure",
        network_label=network_label,
        output_path=output_path,
        output_dir=output_dir
    )
    base_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_output, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    elif not return_figure:
        plt.close(fig)

    if return_figure:
        return base_output, fig
    return base_output


def plot_capacity_network(
    workbook_path: Optional[str] = None,
    output_path: Optional[str] = None,
    sections_workbook_path: Optional[str] = None,
    generate_network: bool = True,
    show: bool = False,
    network_label: str = None,
    output_dir: Path = None,
    infra_version: Optional[str] = None,
    lakes_path: Optional[str] = None,
    include_labels: bool = True,
    marker_scale: float = 1.0,
    allowed_node_classes: Optional[Set[str]] = None,
    is_catchment: bool = False,
    network_marker_scale: float = 1.5,
    boundary_path: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Plot the capacity prep workbook and return the saved image paths (network, capacity).

    Args:
        workbook_path: Optional explicit path to prep workbook.
        output_path: Optional path for output file.
        sections_workbook_path: Optional path to sections workbook.
        generate_network: If True, also generate network map.
        show: If True, display the plots.
        network_label: Optional custom network label (e.g., "AK_2035_dev_100023").
        output_dir: Optional custom output directory.
        infra_version: Optional infra version for real segment geometry.

    Returns:
        Tuple of (network_output_path, capacity_output_path).
    """
    stations_df, segments_df = _load_workbook(
        workbook_path=Path(workbook_path) if workbook_path else None,
        network_label=network_label,
        output_dir=output_dir
    )
    stations = _filter_stations_by_class(_to_stations(stations_df), allowed_node_classes)
    segments = _to_segments(segments_df, stations.keys())
    sections = _load_capacity_sections(
        workbook_path=Path(sections_workbook_path) if sections_workbook_path else None,
        network_label=network_label,
        output_dir=output_dir
    )
    if not sections:
        raise ValueError("No sections were found in the sections workbook.")

    section_stations: Dict[int, Station] = {}
    annotation_segments: List[Segment] = []
    for section in sections:
        section_stations[section.start.node_id] = section.start
        section_stations[section.end.node_id] = section.end
        annotation_segments.append(
            Segment(
                from_node=section.start.node_id,
                to_node=section.end.node_id,
                tracks=section.track_count,
                speed=math.nan,
            )
        )

    base_fig: Optional[Figure] = None
    if generate_network:
        base_result = network_current_map(
            output_path=output_path,
            stations_df=stations_df,
            segments_df=segments_df,
            stations=stations,
            segments_list=segments,
            show=False,
            return_figure=True,
            network_label=network_label,
            output_dir=output_dir,
            infra_version=infra_version,
            lakes_path=lakes_path,
            include_labels=include_labels,
            allowed_node_classes=allowed_node_classes,
            is_catchment=is_catchment,
            marker_scale=network_marker_scale,
            boundary_path=boundary_path,
        )

        if isinstance(base_result, tuple):
            base_output, base_fig = base_result
        else:
            base_output = base_result
    else:
        base_output = _derive_plot_output_path(
            plot_type="infrastructure",
            network_label=network_label,
            output_path=output_path,
            output_dir=output_dir
        )

    # Derive capacity output path with network subdirectories
    capacity_output = _derive_plot_output_path(
        plot_type="capacity",
        network_label=network_label,
        output_path=output_path,
        output_dir=output_dir
    )
    capacity_output.parent.mkdir(parents=True, exist_ok=True)

    # Calculate dynamic figure size based on section station bounding box
    figsize = _calculate_figure_size(section_stations)
    capacity_fig, capacity_ax = plt.subplots(figsize=figsize)

    # Draw lakes on capacity figure; also load segment geometries for BFS routing
    _cap_water, _ = _load_map_overlays(infra_version=infra_version, lakes_path=lakes_path)
    if _cap_water is not None and not getattr(_cap_water, "empty", True):
        try:
            _cap_water.plot(ax=capacity_ax, color="#b7d4f0", edgecolor="#6ea3d5", linewidth=0.5, zorder=1)
        except Exception:
            pass

    capacity_annotation_bounds, _ = _draw_capacity_map(
        capacity_ax,
        sections,
        include_annotations=include_labels,
    )
    station_annotation_bounds, _ = _draw_station_annotations(
        capacity_ax,
        section_stations,
        annotation_segments,
        marker_style="circle",

        include_tables=False,
        include_labels=include_labels,
        marker_scale=marker_scale,
        colour_mode="uniform",
        uniform_colour="#000000",
    )
    combined_bounds = _merge_bounds(capacity_annotation_bounds, station_annotation_bounds)

    # Format title with network information
    capacity_title = _format_plot_title("Capacity Utilization", network_label)
    _configure_axes(
        capacity_ax,
        section_stations,
        title=capacity_title,
        annotation_bounds=combined_bounds,
    )
    _add_north_arrow(capacity_ax)
    _add_scale_bar(capacity_ax)

    capacity_fig.savefig(capacity_output, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        if base_fig is not None:
            plt.close(base_fig)
        plt.close(capacity_fig)

    _flush_directional_asymmetry_summary()
    return base_output, capacity_output


def plot_speed_profile_network(
    workbook_path: Optional[str] = None,
    output_path: Optional[str] = None,
    show: bool = False,
    network_label: str = None,
    output_dir: Path = None,
) -> Path:
    """Plot the network speed profile and return the saved image path.

    Args:
        workbook_path: Optional explicit path to prep workbook.
        output_path: Optional path for output file.
        show: If True, display the plot.
        network_label: Optional custom network label (e.g., "AK_2035_dev_100023").
        output_dir: Optional custom output directory.

    Returns:
        Path to saved speed profile image.
    """
    stations_df, segments_df = _load_workbook(
        workbook_path=Path(workbook_path) if workbook_path else None,
        network_label=network_label,
        output_dir=output_dir
    )
    stations = _to_stations(stations_df)
    segments = _to_segments(segments_df, stations.keys())

    if not stations:
        raise ValueError("No stations were found in the prep workbook.")
    if not segments:
        raise ValueError("No segments were found linking the stations.")

    water_layer, segment_geometries = _load_map_overlays()

    # Calculate dynamic figure size based on station bounding box
    figsize = _calculate_figure_size(stations)
    fig, ax = plt.subplots(figsize=figsize)
    if water_layer is not None and not getattr(water_layer, "empty", True):
        try:
            water_layer.plot(ax=ax, color="#b7d4f0", edgecolor="#6ea3d5", linewidth=0.5, zorder=1)
        except Exception:
            pass

    speed_annotation_bounds = _draw_speed_profile(ax, stations, segments, segment_geometries=segment_geometries)
    station_annotation_bounds, _ = _draw_station_annotations(
        ax,
        stations,
        segments,
        marker_style="circle",

        include_tables=False,
        colour_mode="uniform",
        uniform_colour="#ffffff",
    )
    combined_bounds = _merge_bounds(speed_annotation_bounds, station_annotation_bounds)

    # Format title with network information
    speed_title = _format_plot_title("Speed Profile", network_label)
    _configure_axes(ax, stations, title=speed_title, annotation_bounds=combined_bounds)

    # Use new path derivation logic with network subdirectories
    speed_output = _derive_plot_output_path(
        plot_type="speed",
        network_label=network_label,
        output_path=output_path,
        output_dir=output_dir
    )
    speed_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(speed_output, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return speed_output


def plot_service_network(
    workbook_path: Optional[str] = None,
    output_path: Optional[str] = None,
    show: bool = False,
    network_label: str = None,
    output_dir: Path = None,
    lakes_path: Optional[str] = None,
    include_labels: bool = True,
    allowed_node_classes: Optional[Set[str]] = None,
) -> Path:
    """Plot network services with frequency-based styling and return the saved image path.

    Args:
        workbook_path: Optional explicit path to prep workbook.
        output_path: Optional path for output file.
        show: If True, display the plot.
        network_label: Optional custom network label (e.g., "AK_2035_dev_100023").
        output_dir: Optional custom output directory.

    Returns:
        Path to saved service network image.
    """
    stations_df, segments_df = _load_workbook(
        workbook_path=Path(workbook_path) if workbook_path else None,
        network_label=network_label,
        output_dir=output_dir
    )
    stations = _filter_stations_by_class(_to_stations(stations_df), allowed_node_classes)
    segments = _to_segments(segments_df, stations.keys())

    if not stations:
        raise ValueError("No stations were found in the prep workbook.")
    if not segments:
        raise ValueError("No segments were found linking the stations.")

    segments_with_services = [segment for segment in segments if segment.services_tphpd.strip()]
    if not segments_with_services:
        raise ValueError("No service frequency data was found in the segments sheet.")

    water_layer, segment_geometries = _load_map_overlays(lakes_path=lakes_path)

    station_shapes = _compute_station_shapes(stations, segments_with_services)

    # Calculate dynamic figure size based on station bounding box
    figsize = _calculate_figure_size(stations)
    fig, ax = plt.subplots(figsize=figsize)
    if water_layer is not None and not getattr(water_layer, "empty", True):
        try:
            water_layer.plot(ax=ax, color="#b7d4f0", edgecolor="#6ea3d5", linewidth=0.5, zorder=1)
        except Exception:
            pass

    service_annotation_bounds = _draw_service_map(
        ax,
        stations,
        segments_with_services,
        segment_geometries,
        station_shapes=station_shapes,
        include_service_labels=include_labels,
    )
    stations_no_junctions = {
        node_id: station for node_id, station in stations.items() if not station.is_junction
    }
    station_annotation_bounds, _ = _draw_station_annotations(
        ax,
        stations_no_junctions,
        segments_with_services,
        station_shapes=station_shapes,
        include_tables=include_labels,
        include_labels=include_labels,
        colour_mode="uniform",
        uniform_colour="#ffffff",
        table_text_func=_service_station_table,
    )
    combined_bounds = _merge_bounds(service_annotation_bounds, station_annotation_bounds)

    # Format title with network information
    service_title = _format_plot_title("Service Frequencies", network_label)
    _configure_axes(ax, stations, title=service_title, annotation_bounds=combined_bounds)
    _add_north_arrow(ax)
    _add_scale_bar(ax)

    legend_handles = [
        Line2D([0], [0], color=SERVICE_COLOUR_STOP, linewidth=2.0, label="Stopping"),
        Line2D([0], [0], color=SERVICE_COLOUR_PASS, linewidth=2.0, label="Passing"),
    ]
    ax.legend(handles=legend_handles, title="Service type", loc="upper right", frameon=True, fontsize=8, title_fontsize=9)

    # Use new path derivation logic with network subdirectories
    service_output = _derive_plot_output_path(
        plot_type="service",
        network_label=network_label,
        output_path=output_path,
        output_dir=output_dir
    )
    service_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(service_output, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    _flush_directional_asymmetry_summary()
    return service_output


if __name__ == "__main__":
    # network_output = network_current_map(show=True)
    # print(f"Network plot saved to {network_output}")
    # speed_output = plot_speed_profile_network(show=True)
    # print(f"Speed profile plot saved to {speed_output}")
    service_output = plot_service_network(show=True)
    print(f"Service plot saved to {service_output}")
    # capacity_output = plot_capacity_network(show=True, generate_network=False)
    # print(f"Capacity plot saved to {capacity_output}")
