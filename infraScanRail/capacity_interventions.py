"""
Capacity Expansion Interventions
Last modified: 2026-05-23

Identifies capacity-constrained sections and designs Tier-1 infrastructure
interventions (real nodes + adjusted segments) to bring all sections to
≥ settings.capacity_threshold tphpd available capacity.

Intervention types
------------------
- station_track   — multi-segment section: add +1 track at the central station.
- segment_passing_siding — single-segment section: insert a physics-sized
  passing siding (Topic 1, 2026-05). Length L = 2·L_train + v²/a; if L exceeds
  settings.MAX_SIDING_LENGTH_RATIO · L_section the whole section is duplicated.

Cost model is pure per-meter (track_cost_per_meter from cost_parameters); the
legacy lump-sum segment_siding_costs / station_siding_costs are unused here.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from openpyxl import load_workbook
import logging

import geopandas as gpd
from shapely.geometry import Point, MultiLineString
from shapely.ops import linemerge, substring

# Import from existing modules
from capacity_calculator import _build_sections_dataframe, build_capacity_tables
from capacity_network_plots import plot_capacity_network
import cost_parameters
import paths
import settings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tier-1 passing-siding geometry (Topic 1, 2026-05)
# ─────────────────────────────────────────────────────────────────────────────

def _required_siding_length(speed_kmh: float) -> float:
    """Minimum passing-siding length [m] for crossing at line speed.

    L_siding = 2 · L_train + v² / a
    Uses settings.MAX_TRAIN_LENGTH_M and settings.SERVICE_BRAKE_DECEL_MS2.
    """
    v = speed_kmh / 3.6
    return 2.0 * settings.MAX_TRAIN_LENGTH_M + (v ** 2) / settings.SERVICE_BRAKE_DECEL_MS2


def _decide_strategy(section_length_m: float, siding_length_m: float) -> str:
    """Return 'extra_track' if siding_length_m ≥ ratio · section_length_m,
    else 'siding_with_junctions'. Threshold = settings.MAX_SIDING_LENGTH_RATIO.
    """
    if section_length_m <= 0:
        return 'extra_track'
    if siding_length_m / section_length_m >= settings.MAX_SIDING_LENGTH_RATIO:
        return 'extra_track'
    return 'siding_with_junctions'


def _plan_siding_km_positions(
    section_length_m: float,
    siding_length_m: float,
) -> Tuple[float, float]:
    """Return (kp_A, kp_B) — junction-node positions centred on the section midpoint."""
    midpoint = section_length_m / 2.0
    kp_A = max(0.0, midpoint - siding_length_m / 2.0)
    kp_B = min(section_length_m, midpoint + siding_length_m / 2.0)
    return kp_A, kp_B


@dataclass
class CapacityIntervention:
    """A single capacity enhancement intervention.

    Two intervention types:
      'station_track'         — add +1 platform-track at a central station node.
      'segment_passing_siding' — insert a Tier-1 passing siding into a single-
          segment section. The strategy field distinguishes:
            'extra_track'           → entire section becomes 2-track
            'siding_with_junctions' → two new junction nodes split the segment;
                                      the middle sub-segment gets +1 track.

    Attributes:
        intervention_id: Unique identifier (e.g. "INT_ST_001", "INT_PS_017").
        section_id: Section requiring intervention.
        type: 'station_track' or 'segment_passing_siding'.
        node_id: Station node ID (station interventions only).
        segment_id: Segment identifier "from_node-to_node" (siding only).
        tracks_added: Effective track delta on the prep workbook (+1.0 always
            for Tier-1; +0.5 retained only for the legacy capacity calculator
            when strategy == 'siding_with_junctions').
        affected_segments: Segment IDs impacted.
        construction_cost_chf, maintenance_cost_annual_chf: see calculate_intervention_cost.
        length_m: Section length (full duplication) or siding length (junctions).
        current_tracks: Track count before intervention.
        iteration: enhancement-iteration index this intervention was added in.
        current_platforms, platforms_added, platform_cost_chf: station only.
        strategy: 'extra_track' | 'siding_with_junctions' | None.
          'extra_track'          → whole section gets +1 track (siding ≥ ratio · section length)
          'siding_with_junctions'→ two junction nodes inserted; middle sub-segment gets +1 track
        siding_length_m: Tier-1 physics-based siding length (junctions only).
        design_speed_kmh: Speed used to size the siding.
        section_length_m: Total section length (junctions only — used for the
            full-duplication threshold check).
    """
    intervention_id: str
    section_id: str
    type: str
    node_id: Optional[int]
    segment_id: Optional[str]
    tracks_added: float
    affected_segments: List[str]
    construction_cost_chf: float
    maintenance_cost_annual_chf: float
    length_m: Optional[float]
    current_tracks: float
    iteration: int = 1
    current_platforms: Optional[float] = None
    platforms_added: Optional[float] = None
    platform_cost_chf: Optional[float] = None
    strategy: Optional[str] = None
    siding_length_m: Optional[float] = None
    design_speed_kmh: Optional[float] = None
    section_length_m: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame export."""
        return {
            'intervention_id': self.intervention_id,
            'section_id': self.section_id,
            'type': self.type,
            'node_id': self.node_id,
            'segment_id': self.segment_id,
            'tracks_added': self.tracks_added,
            'affected_segments': '|'.join(self.affected_segments),
            'construction_cost_chf': self.construction_cost_chf,
            'maintenance_cost_annual_chf': self.maintenance_cost_annual_chf,
            'length_m': self.length_m,
            'current_tracks': self.current_tracks,
            'iteration': self.iteration,
            'current_platforms': self.current_platforms,
            'platforms_added': self.platforms_added,
            'platform_cost_chf': self.platform_cost_chf,
            'strategy': self.strategy,
            'siding_length_m': self.siding_length_m,
            'design_speed_kmh': self.design_speed_kmh,
            'section_length_m': self.section_length_m,
        }


def identify_capacity_constrained_sections(
    sections_df: pd.DataFrame,
    threshold_tphpd: float = 2.0
) -> pd.DataFrame:
    """
    Identify sections with available capacity below threshold.

    Available capacity = Capacity - total_tphpd (remaining capacity)

    Args:
        sections_df: Sections DataFrame from Phase 3
        threshold_tphpd: Minimum required available capacity (default: 2.0)

    Returns:
        DataFrame of constrained sections
    """
    logger.info(f"Identifying sections with available capacity < {threshold_tphpd} tphpd")

    # Calculate available capacity (remaining capacity)
    sections_df = sections_df.copy()
    sections_df['available_capacity'] = (
        sections_df['Capacity'] - sections_df['total_tphpd']
    )

    # Filter constrained sections
    constrained = sections_df[
        sections_df['available_capacity'] < threshold_tphpd
    ].copy()

    logger.info(f"Found {len(constrained)} constrained sections")

    if len(constrained) > 0:
        logger.info(f"Available capacity range: "
                   f"{constrained['available_capacity'].min():.2f} to "
                   f"{constrained['available_capacity'].max():.2f} tphpd")

    return constrained


def _find_geometric_center_station(
    segment_sequence: str,
    segments_df: pd.DataFrame,
    stations_df: pd.DataFrame
) -> tuple[int, str]:
    """
    Find station closest to geometric center of section based on rail distance.

    Args:
        segment_sequence: Pipe-separated segment IDs (e.g., "8-10|10-12|12-15")
        segments_df: Segments DataFrame with length_m column
        stations_df: Stations DataFrame with CODE column

    Returns:
        Tuple of (station_id, selection_method):
            - station_id: Node ID of selected station
            - selection_method: "geometric_center" or "fallback_index"
    """
    import logging
    logger = logging.getLogger(__name__)

    segments = segment_sequence.split('|')

    # Try geometric center calculation
    try:
        # Build list of (station_id, cumulative_distance)
        stations_with_distances = []
        cumulative_dist = 0.0

        for seg in segments:
            from_node, to_node = map(int, seg.split('-'))

            # Add from_node at current cumulative distance (if not duplicate)
            if not stations_with_distances or stations_with_distances[-1][0] != from_node:
                stations_with_distances.append((from_node, cumulative_dist))

            # Look up segment length
            seg_row = segments_df[
                (segments_df['from_node'] == from_node) &
                (segments_df['to_node'] == to_node)
            ]

            if len(seg_row) == 0:
                raise ValueError(f"Segment {seg} not found in segments_df")

            length_m = seg_row.iloc[0]['length_m']

            # Check if length is available (not NA/NaN)
            if pd.isna(length_m):
                raise ValueError(f"Segment {seg} has missing length_m")

            cumulative_dist += float(length_m)

        # Add final to_node
        final_to = int(segments[-1].split('-')[1])
        stations_with_distances.append((final_to, cumulative_dist))

        # Find station closest to midpoint (tie-break: earlier station)
        total_length = cumulative_dist
        midpoint = total_length / 2.0

        closest_station = min(
            stations_with_distances,
            key=lambda x: (abs(x[1] - midpoint), x[1])  # Sort by distance, then position
        )

        station_id = closest_station[0]
        station_code = stations_df[stations_df['NR'] == station_id]['CODE'].values[0]

        logger.info(
            f"Geometric center: Section midpoint at {midpoint:.0f}m, "
            f"selected station {station_code} (ID {station_id}) at {closest_station[1]:.0f}m"
        )

        return station_id, "geometric_center"

    except (ValueError, KeyError, IndexError) as e:
        # Fallback to index-based method
        logger.warning(
            f"⚠ Cannot calculate geometric center for section '{segment_sequence}': {e}"
        )
        logger.warning(
            f"⚠ Falling back to middle segment index method. "
            f"Please ensure segment lengths are enriched for accurate intervention placement."
        )

        # Use current index-based method
        middle_index = len(segments) // 2
        middle_segment = segments[middle_index]
        station_id = int(middle_segment.split('-')[0])

        return station_id, "fallback_index"


def design_section_intervention(
    section: pd.Series,
    segments_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    intervention_counter: int,
    iteration: int = 1
) -> CapacityIntervention:
    """
    Design appropriate intervention for a capacity-constrained section.

    Logic:
    - Multi-segment section (>1 segment): Add station track at geometric center station
      (based on cumulative segment lengths; falls back to middle segment index if lengths unavailable)
    - Single-segment section (1 segment): Add passing siding to segment

    Args:
        section: Single section record
        segments_df: Segments DataFrame with length_m column
        stations_df: Stations DataFrame with CODE column
        intervention_counter: Counter for generating unique IDs
        iteration: Current iteration number

    Returns:
        CapacityIntervention object
    """
    section_id = section['section_id']
    segment_sequence = section['segment_sequence']  # e.g., "8-10|10-12|12-15"

    # Parse segment sequence
    segments = segment_sequence.split('|')

    logger.debug(f"Designing intervention for section {section_id} "
                f"({len(segments)} segments)")

    if len(segments) > 1:
        # Multi-segment section: Station track intervention at geometric center
        middle_station_id, selection_method = _find_geometric_center_station(
            segment_sequence=segment_sequence,
            segments_df=segments_df,
            stations_df=stations_df
        )

        # Extract current track count from station
        station_row = stations_df[stations_df['NR'] == middle_station_id]
        if len(station_row) == 0:
            logger.warning(f"Station {middle_station_id} not found in stations_df")
            current_tracks = 1.0  # Default fallback
            current_platforms = None
            platforms_added = None
        else:
            current_tracks = float(station_row.iloc[0]['tracks'])
            # Check platform count
            current_platforms = float(station_row.iloc[0]['platforms'])
            # Add platform if fewer than 2 platforms exist
            if current_platforms < 2:
                platforms_added = 1.0
            else:
                platforms_added = None

        intervention = CapacityIntervention(
            intervention_id=f"INT_ST_{intervention_counter:04d}",
            section_id=str(section_id),
            type='station_track',
            node_id=middle_station_id,
            segment_id=None,
            tracks_added=1.0,
            affected_segments=segments,
            construction_cost_chf=0.0,  # Filled by calculate_intervention_cost()
            maintenance_cost_annual_chf=0.0,
            length_m=None,
            current_tracks=current_tracks,
            iteration=iteration,
            current_platforms=current_platforms,
            platforms_added=platforms_added
        )

        if platforms_added:
            logger.debug(f"  → Station track at node {middle_station_id} (+ platform)")
        else:
            logger.debug(f"  → Station track at node {middle_station_id}")

    else:
        # Single-segment section: Tier-1 passing siding intervention
        segment_id = segments[0]

        from_node, to_node = segment_id.split('-')
        from_node, to_node = int(from_node), int(to_node)

        segment_row = segments_df[
            (segments_df['from_node'] == from_node) &
            (segments_df['to_node'] == to_node)
        ]

        if len(segment_row) == 0:
            logger.warning(f"Segment {segment_id} not found in segments_df")
            section_length_m = 0.0
            current_tracks = 1.0
            speed_kmh = float(settings.SERVICE_BRAKE_DECEL_MS2 * 0 + 50)
        else:
            section_length_m = float(segment_row.iloc[0]['length_m'])
            current_tracks   = float(segment_row.iloc[0]['tracks'])
            speed_raw = segment_row.iloc[0].get('speed', None)
            speed_kmh = (float(speed_raw)
                         if speed_raw is not None and not pd.isna(speed_raw) and float(speed_raw) > 0
                         else 50.0)

        siding_length_m = _required_siding_length(speed_kmh)
        strategy = _decide_strategy(section_length_m, siding_length_m)

        if strategy == 'extra_track':
            length_m = section_length_m
            tracks_added = 1.0
            logger.debug(
                f"  → Extra track on segment {segment_id} "
                f"(L_siding={siding_length_m:.0f}m ≥ {settings.MAX_SIDING_LENGTH_RATIO:.0%} × "
                f"{section_length_m:.0f}m)"
            )
        else:
            length_m = siding_length_m
            tracks_added = 0.5
            kp_A, kp_B = _plan_siding_km_positions(section_length_m, siding_length_m)
            logger.debug(
                f"  → Passing siding on segment {segment_id} "
                f"(L_siding={siding_length_m:.0f}m of {section_length_m:.0f}m at "
                f"{speed_kmh:.0f} km/h, junctions at kp={kp_A:.0f}m / {kp_B:.0f}m)"
            )

        intervention = CapacityIntervention(
            intervention_id=f"INT_PS_{intervention_counter:04d}",
            section_id=str(section_id),
            type='segment_passing_siding',
            node_id=None,
            segment_id=segment_id,
            tracks_added=tracks_added,
            affected_segments=[segment_id],
            construction_cost_chf=0.0,
            maintenance_cost_annual_chf=0.0,
            length_m=length_m,
            current_tracks=current_tracks,
            iteration=iteration,
            strategy=strategy,
            siding_length_m=siding_length_m,
            design_speed_kmh=speed_kmh,
            section_length_m=section_length_m,
        )

    return intervention


def calculate_intervention_cost(
    intervention: CapacityIntervention,
    maintenance_rate: float = None
) -> CapacityIntervention:
    """Calculate construction and maintenance costs (pure per-meter).

    Cost formulas
    -------------
    station_track:
        cost = cost_parameters.station_siding_costs · floor(current_tracks)
        (+ platform_cost_per_unit · platforms_added, if any).

    segment_passing_siding:
        strategy='extra_track'           → cost = L_section · track_cost_per_meter
        strategy='siding_with_junctions' → cost = L_siding  · track_cost_per_meter

    Maintenance: construction_cost · maintenance_rate.

    Args:
        intervention: Intervention object with current_tracks/length_m populated.
        maintenance_rate: annual fraction. Defaults to
            cost_parameters.yearly_maintenance_to_construction_cost_factor.
    """
    import math

    if maintenance_rate is None:
        maintenance_rate = cost_parameters.yearly_maintenance_to_construction_cost_factor

    base_tracks = math.floor(intervention.current_tracks)

    if intervention.type == 'station_track':
        construction_cost = cost_parameters.station_siding_costs * base_tracks
        if intervention.platforms_added and intervention.platforms_added > 0:
            platform_cost = (
                cost_parameters.platform_cost_per_unit *
                intervention.platforms_added
            )
            construction_cost += platform_cost
            intervention.platform_cost_chf = platform_cost

    elif intervention.type == 'segment_passing_siding':
        track_length_m = float(intervention.length_m or 0.0)
        construction_cost = track_length_m * cost_parameters.track_cost_per_meter

    else:
        raise ValueError(f"Unknown intervention type: {intervention.type}")

    intervention.construction_cost_chf = construction_cost
    intervention.maintenance_cost_annual_chf = construction_cost * maintenance_rate
    return intervention


def apply_interventions_to_workbook(
    prep_workbook_path: Path,
    interventions_list: List[CapacityIntervention],
    output_path: Path
) -> None:
    """
    Apply track adjustments to workbook by updating tracks attributes.

    Args:
        prep_workbook_path: Path to original prep workbook
        interventions_list: List of interventions to apply
        output_path: Path for enhanced baseline workbook
    """
    logger.info(f"Applying {len(interventions_list)} interventions to workbook")

    # Load workbook
    stations_df = pd.read_excel(prep_workbook_path, sheet_name='Stations')
    segments_df = pd.read_excel(prep_workbook_path, sheet_name='Segments')

    # Track changes for logging
    station_changes = {}
    segment_changes = {}

    # Apply interventions
    for intervention in interventions_list:
        if intervention.type == 'station_track':
            # Add +1 track to station
            mask = stations_df['NR'] == intervention.node_id
            if mask.sum() > 0:
                old_tracks = stations_df.loc[mask, 'tracks'].values[0]
                stations_df.loc[mask, 'tracks'] += 1.0
                new_tracks = stations_df.loc[mask, 'tracks'].values[0]
                station_changes[intervention.node_id] = (old_tracks, new_tracks)
                logger.debug(f"  Station {intervention.node_id}: "
                           f"{old_tracks} → {new_tracks} tracks")

                # Add platforms if specified
                if intervention.platforms_added and intervention.platforms_added > 0:
                    old_platforms = stations_df.loc[mask, 'platforms'].values[0]
                    stations_df.loc[mask, 'platforms'] += intervention.platforms_added
                    new_platforms = stations_df.loc[mask, 'platforms'].values[0]
                    logger.debug(f"  Station {intervention.node_id}: "
                               f"{old_platforms} → {new_platforms} platforms")
            else:
                logger.warning(f"  Station {intervention.node_id} not found")

        elif intervention.type == 'segment_passing_siding':
            # Track delta on the prep workbook depends on the Tier-1 strategy:
            #   extra_track           → +1.0 (whole section becomes 2-track)
            #   siding_with_junctions → +0.5 (downstream capacity model reads
            #                                 fractional as a passing siding)
            from_node, to_node = intervention.segment_id.split('-')
            from_node, to_node = int(from_node), int(to_node)

            delta = (1.0 if intervention.strategy == 'extra_track'
                     else intervention.tracks_added)

            mask = ((segments_df['from_node'] == from_node) &
                   (segments_df['to_node'] == to_node))

            if mask.sum() > 0:
                old_tracks = segments_df.loc[mask, 'tracks'].values[0]
                segments_df.loc[mask, 'tracks'] += delta
                new_tracks = segments_df.loc[mask, 'tracks'].values[0]
                segment_changes[intervention.segment_id] = (old_tracks, new_tracks)
                logger.debug(f"  Segment {intervention.segment_id}: "
                           f"{old_tracks} → {new_tracks} tracks ({intervention.strategy})")
            else:
                logger.warning(f"  Segment {intervention.segment_id} not found")

    # Save enhanced workbook
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        stations_df.to_excel(writer, sheet_name='Stations', index=False)
        segments_df.to_excel(writer, sheet_name='Segments', index=False)

    logger.info(f"Enhanced workbook saved to: {output_path}")
    logger.info(f"  Modified {len(station_changes)} stations, "
               f"{len(segment_changes)} segments")


def recalculate_enhanced_capacity(
    enhanced_prep_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recalculate sections and capacity after interventions.

    This reloads the enhanced prep workbook and re-runs _build_sections_dataframe()
    to get updated section definitions and capacity values.

    Args:
        enhanced_prep_path: Path to enhanced baseline prep workbook

    Returns:
        Tuple of (enhanced_sections_df, enhanced_segments_df)
    """
    logger.info("Recalculating capacity with enhanced network")

    # Load enhanced workbook
    stations_df = pd.read_excel(enhanced_prep_path, sheet_name='Stations')
    segments_df = pd.read_excel(enhanced_prep_path, sheet_name='Segments')

    # Rebuild sections with updated track counts
    sections_df = _build_sections_dataframe(stations_df, segments_df)

    logger.info(f"Recalculated {len(sections_df)} sections")

    return sections_df, segments_df


def visualize_enhanced_network(
    enhanced_prep_path: Path,
    enhanced_sections_path: Path,
    interventions_list: List[CapacityIntervention],
    network_label: str = "AK_2035_enhanced",
    output_dir: Path = None
) -> Tuple[Path, Path]:
    """
    Generate infrastructure and capacity plots for enhanced network.

    The infrastructure plot uses the existing plot_capacity_network() function
    but applies it to the enhanced network with updated track counts.

    Args:
        enhanced_prep_path: Path to enhanced prep workbook
        enhanced_sections_path: Path to enhanced sections workbook
        interventions_list: List of interventions applied
        network_label: Network label for plot paths (auto-detects plot directory)
        output_dir: (Deprecated) Not used - plot directory auto-detected from network_label

    Returns:
        Tuple of (infrastructure_plot_path, capacity_plot_path)
    """
    logger.info("Generating enhanced network visualizations")

    # Generate infrastructure and capacity plots using existing function
    # Note: output_dir is NOT passed to allow auto-detection based on network_label
    infrastructure_plot, capacity_plot = plot_capacity_network(
        workbook_path=str(enhanced_prep_path),
        sections_workbook_path=str(enhanced_sections_path),
        generate_network=True,
        show=False,
        network_label=network_label
    )

    logger.info(f"Infrastructure plot saved to: {infrastructure_plot}")
    logger.info(f"Capacity plot saved to: {capacity_plot}")

    # Note: Passing siding visualization as offset parallel lines would require
    # modifying the core plotting functions in capacity_network_plots.py
    # For now, the enhanced plots show the updated track counts
    # Future enhancement: Add custom overlay for passing sidings

    return infrastructure_plot, capacity_plot


# ─────────────────────────────────────────────────────────────────────────────
# Cap-int → spatial conversion (bridge to infra-int registry)
# ─────────────────────────────────────────────────────────────────────────────

_CAP_JUNCTION_NR_START = 9_100_001


def _make_cap_int_id(intervention: 'CapacityIntervention') -> str:
    """Map an intervention to its canonical registry int_id."""
    counter = int(intervention.intervention_id.split('_')[-1])
    if intervention.type == 'station_track':
        return f"cap_st_{counter:04d}"
    if intervention.strategy == 'extra_track':
        return f"cap_et_{counter:04d}"
    return f"cap_ps_{counter:04d}"


def _lookup_segment(
    segs_base: gpd.GeoDataFrame,
    from_nr: str,
    to_nr: str,
) -> gpd.GeoDataFrame:
    """Return the segment row matching from_nr/to_nr (tries both directions)."""
    mask = segs_base['Number'] == f"{from_nr}_{to_nr}"
    if mask.any():
        return segs_base[mask]
    mask = segs_base['Number'] == f"{to_nr}_{from_nr}"
    if mask.any():
        return segs_base[mask]
    raise KeyError(f"Segment {from_nr}-{to_nr} not found in base network")


def _next_junction_nr(nodes_base: gpd.GeoDataFrame) -> int:
    """Return the next available cap junction node Number (≥ 9,100,001)."""
    if 'Number' not in nodes_base.columns:
        return _CAP_JUNCTION_NR_START
    existing = nodes_base['Number'].dropna()
    cap_existing = existing[existing.apply(
        lambda x: isinstance(x, (int, float)) and x >= _CAP_JUNCTION_NR_START
    )]
    return int(cap_existing.max()) + 1 if not cap_existing.empty else _CAP_JUNCTION_NR_START


def _spatial_station_track(
    intervention: 'CapacityIntervention',
    int_id: str,
    nodes_base: gpd.GeoDataFrame,
) -> Tuple[Optional[gpd.GeoDataFrame], None, None]:
    """Return updated node row for a station_track intervention."""
    mask = nodes_base['Number'] == intervention.node_id
    if not mask.any():
        logger.warning(f"  [bridge] station node {intervention.node_id} not found; skipping")
        return None, None, None
    node_row = nodes_base[mask].copy()
    node_row['Track_Count'] = node_row['Track_Count'] + 1
    if intervention.platforms_added:
        node_row['Platform_Count'] = (
            node_row['Platform_Count'].fillna(0) + intervention.platforms_added
        )
    node_row['int_id'] = int_id
    return node_row, None, None


def _spatial_extra_track(
    intervention: 'CapacityIntervention',
    int_id: str,
    segs_base: gpd.GeoDataFrame,
    comp_base: gpd.GeoDataFrame,
) -> Tuple[None, gpd.GeoDataFrame, Optional[gpd.GeoDataFrame]]:
    """Return updated segment row (+1 track) for an extra_track intervention."""
    from_nr, to_nr = intervention.segment_id.split('-')
    seg_row = _lookup_segment(segs_base, from_nr, to_nr).copy()
    seg_row['Num_Tracks'] = seg_row['Num_Tracks'] + 1
    seg_row['int_id'] = int_id

    orig_seg_id = seg_row['Segment_ID'].iloc[0]
    comp_rows = comp_base[comp_base['Segment_ID'] == orig_seg_id].copy()
    comp_rows['int_id'] = int_id

    return None, seg_row, comp_rows if not comp_rows.empty else None


def _spatial_siding_with_junctions(
    intervention: 'CapacityIntervention',
    int_id: str,
    segs_base: gpd.GeoDataFrame,
    comp_base: gpd.GeoDataFrame,
    nodes_base: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Optional[gpd.GeoDataFrame]]:
    """Return 2 junction nodes + 4 segment rows (tombstone + 3 sub-segs) for a siding intervention."""
    from_nr, to_nr = intervention.segment_id.split('-')
    seg_row = _lookup_segment(segs_base, from_nr, to_nr)
    seg_geom = seg_row.geometry.iloc[0]
    line = linemerge(seg_geom) if seg_geom.geom_type == 'MultiLineString' else seg_geom

    kp_A, kp_B = _plan_siding_km_positions(
        intervention.section_length_m, intervention.siding_length_m
    )
    pt_A = line.interpolate(kp_A)
    pt_B = line.interpolate(kp_B)

    junc_a_nr = _next_junction_nr(nodes_base)
    junc_b_nr = junc_a_nr + 1

    # Build junction node rows using an existing junction as schema template
    junction_nodes = nodes_base[nodes_base['Node_Class'] == 'junction']
    if junction_nodes.empty:
        junction_nodes = nodes_base
    template = junction_nodes.iloc[[0]].copy()

    def _make_junction(nr, point):
        row = template.copy()
        row['Number'] = nr
        row['Node_ID'] = f"cap_junc_{int_id}_{nr}"
        row['Name'] = f"Cap junction {nr}"
        row['Code'] = f"CJ{nr % 10000:04d}"
        row['E'] = point.x
        row['N'] = point.y
        row['Node_Class'] = 'junction'
        row['Transport_Mode'] = seg_row['Transport_Mode'].iloc[0]
        row['Track_Count'] = seg_row['Num_Tracks'].iloc[0]
        row['Platform_Count'] = None
        row['Parent_Node'] = None
        row['int_id'] = int_id
        row['geometry'] = point
        return row

    junc_a_row = _make_junction(junc_a_nr, pt_A)
    junc_b_row = _make_junction(junc_b_nr, pt_B)

    # Build sub-segment rows
    base_tracks = seg_row['Num_Tracks'].iloc[0]
    seg_len = seg_row['Length'].iloc[0] if 'Length' in seg_row.columns else line.length
    orig_seg_id = seg_row['Segment_ID'].iloc[0]

    def _make_sub_segment(f_nr, t_nr, geom, tracks):
        row = seg_row.copy()
        row['Segment_ID'] = f"{orig_seg_id}_{f_nr}_{t_nr}"
        row['Number'] = f"{f_nr}_{t_nr}"
        row['Code'] = f"CJ{int(f_nr) % 10000:04d}_CJ{int(t_nr) % 10000:04d}"
        row['Num_Tracks'] = tracks
        row['Length'] = geom.length
        frac = geom.length / (seg_len or 1.0)
        row['Tunnel_Length'] = (seg_row['Tunnel_Length'].iloc[0] or 0.0) * frac
        row['Bridge_Length'] = (seg_row['Bridge_Length'].iloc[0] or 0.0) * frac
        row['Conventional_Length'] = (seg_row['Conventional_Length'].iloc[0] or 0.0) * frac
        row['geometry'] = MultiLineString([list(geom.coords)]) if geom.geom_type == 'LineString' else geom
        row['int_id'] = int_id
        return row

    geom_ab = substring(line, 0,          kp_A)
    geom_bc = substring(line, kp_A,       kp_B)
    geom_cd = substring(line, kp_B, line.length)

    seg_ab = _make_sub_segment(from_nr,   junc_a_nr, geom_ab, base_tracks)
    seg_bc = _make_sub_segment(junc_a_nr, junc_b_nr, geom_bc, base_tracks + 1)
    seg_cd = _make_sub_segment(junc_b_nr, to_nr,     geom_cd, base_tracks)

    # Tombstone: original segment row with _delete=True so resolve_or_build removes it
    tombstone = seg_row.copy()
    tombstone['_delete'] = True
    tombstone['int_id'] = int_id

    # Composition rows for the siding sub-segment (BC) via proportional allocation
    comp_orig = comp_base[comp_base['Segment_ID'] == orig_seg_id]
    siding_frac = (kp_B - kp_A) / (line.length or 1.0)
    comp_bc_rows = []
    bc_seg_id = seg_bc['Segment_ID'].iloc[0]
    for _, piece in comp_orig.iterrows():
        piece_row = piece.copy()
        piece_row['Segment_ID'] = bc_seg_id
        piece_row['Piece_Length'] = piece['Piece_Length'] * siding_frac
        piece_row['int_id'] = int_id
        piece_row['geometry'] = geom_bc
        comp_bc_rows.append(piece_row)

    new_nodes = gpd.GeoDataFrame(
        pd.concat([junc_a_row, junc_b_row], ignore_index=True),
        crs=nodes_base.crs,
    )
    new_segs = gpd.GeoDataFrame(
        pd.concat([tombstone, seg_ab, seg_bc, seg_cd], ignore_index=True),
        crs=segs_base.crs,
    )
    new_comp = (
        gpd.GeoDataFrame(comp_bc_rows, crs=comp_base.crs)
        if comp_bc_rows else None
    )
    return new_nodes, new_segs, new_comp


def cap_ints_to_spatial(
    interventions: List['CapacityIntervention'],
    infra_version: str,
) -> Tuple[Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame], Optional[gpd.GeoDataFrame]]:
    """Convert CapacityIntervention objects to spatial registry rows.

    Loads the base infra network from infra_version and produces GeoDataFrames
    suitable for append_cap_intervention_rows(). Called by run_enhanced_workflow()
    after run_phase_four() returns.

    Args:
        interventions: list from run_phase_four().
        infra_version: name of the base infra version (e.g. 'AS_2026_ZH').

    Returns:
        (nodes_gdf, segments_gdf, comp_gdf) — any may be None if no interventions
        of that category were generated.
    """
    base_dir = Path(paths.get_infra_version_dir(infra_version))
    nodes_base = gpd.read_file(base_dir / 'nodes.gpkg')
    segs_base  = gpd.read_file(base_dir / 'segments.gpkg')
    comp_path  = base_dir / 'segments_composition.gpkg'
    comp_base  = (gpd.read_file(comp_path)
                  if comp_path.exists()
                  else gpd.GeoDataFrame(columns=['Segment_ID'], crs=nodes_base.crs))

    out_nodes: List[gpd.GeoDataFrame] = []
    out_segs:  List[gpd.GeoDataFrame] = []
    out_comp:  List[gpd.GeoDataFrame] = []

    for intervention in interventions:
        int_id = _make_cap_int_id(intervention)
        try:
            if intervention.type == 'station_track':
                n, s, c = _spatial_station_track(intervention, int_id, nodes_base)
            elif intervention.strategy == 'extra_track':
                n, s, c = _spatial_extra_track(intervention, int_id, segs_base, comp_base)
            else:
                n, s, c = _spatial_siding_with_junctions(
                    intervention, int_id, segs_base, comp_base, nodes_base
                )
        except Exception as exc:
            logger.warning(f"  [bridge] skipping {int_id}: {exc}")
            continue

        if n is not None and not n.empty:
            out_nodes.append(n)
        if s is not None and not s.empty:
            out_segs.append(s)
        if c is not None and not c.empty:
            out_comp.append(c)

    nodes_gdf = gpd.GeoDataFrame(
        pd.concat(out_nodes, ignore_index=True), crs=nodes_base.crs
    ) if out_nodes else None
    segs_gdf = gpd.GeoDataFrame(
        pd.concat(out_segs, ignore_index=True), crs=segs_base.crs
    ) if out_segs else None
    comp_gdf = gpd.GeoDataFrame(
        pd.concat(out_comp, ignore_index=True), crs=comp_base.crs
    ) if out_comp else None

    return nodes_gdf, segs_gdf, comp_gdf


def run_phase_four(
    original_sections_df: pd.DataFrame,
    original_segments_df: pd.DataFrame,
    original_stations_df: pd.DataFrame,
    prep_workbook_path: Path,
    output_dir: Path,
    network_label: str,
    threshold_tphpd: float = 2.0,
    max_iterations: int = 10
) -> Tuple[List[CapacityIntervention], Path, pd.DataFrame]:
    """
    Execute Phase 4 capacity interventions with iteration until convergence.

    Args:
        original_sections_df: Sections DataFrame from Phase 3
        original_segments_df: Segments DataFrame
        original_stations_df: Stations DataFrame
        prep_workbook_path: Path to original prep workbook
        output_dir: Directory for enhanced baseline outputs
        threshold_tphpd: Minimum required available capacity (default: 2.0)
        max_iterations: Maximum number of intervention iterations

    Returns:
        Tuple of (interventions_catalog, enhanced_prep_path, final_sections_df)
    """
    logger.info("=" * 60)
    logger.info("Phase 4: Capacity Enhancement Interventions")
    logger.info("=" * 60)

    # Initialize
    all_interventions = []
    intervention_counter = 1

    # Working copies
    current_sections_df = original_sections_df.copy()
    current_prep_path = prep_workbook_path

    # Iteration loop
    for iteration in range(1, max_iterations + 1):
        logger.info(f"\n--- Iteration {iteration} ---")

        # Step 1: Identify constrained sections
        constrained_sections = identify_capacity_constrained_sections(
            current_sections_df,
            threshold_tphpd
        )

        if len(constrained_sections) == 0:
            logger.info(f"✓ All sections have ≥{threshold_tphpd} tphpd available capacity")
            break

        # Step 2: Design interventions for this iteration
        iteration_interventions = []
        for idx, section in constrained_sections.iterrows():
            intervention = design_section_intervention(
                section,
                original_segments_df,
                original_stations_df,
                intervention_counter,
                iteration
            )
            intervention_counter += 1
            iteration_interventions.append(intervention)

        logger.info(f"Designed {len(iteration_interventions)} interventions:")
        station_count = sum(1 for i in iteration_interventions if i.type == 'station_track')
        siding_count = sum(1 for i in iteration_interventions if i.type == 'segment_passing_siding')
        logger.info(f"  - {station_count} station tracks")
        logger.info(f"  - {siding_count} passing sidings")

        # Step 3: Calculate costs
        for intervention in iteration_interventions:
            calculate_intervention_cost(intervention)

        total_construction = sum(i.construction_cost_chf for i in iteration_interventions)
        total_maintenance = sum(i.maintenance_cost_annual_chf for i in iteration_interventions)
        logger.info(f"Iteration costs:")
        logger.info(f"  Construction: {total_construction:,.0f} CHF")
        logger.info(f"  Annual maintenance: {total_maintenance:,.0f} CHF")

        # Step 4: Apply interventions to workbook
        enhanced_prep_path = output_dir / f"capacity_{network_label}_enhanced_network_prep_iter{iteration}.xlsx"
        apply_interventions_to_workbook(
            current_prep_path,
            iteration_interventions,
            enhanced_prep_path
        )

        # Step 5: Recalculate capacity
        current_sections_df, current_segments_df = recalculate_enhanced_capacity(
            enhanced_prep_path
        )

        # Update for next iteration
        current_prep_path = enhanced_prep_path
        all_interventions.extend(iteration_interventions)

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("Phase 4 Complete!")
    logger.info("=" * 60)
    logger.info(f"Total iterations: {min(iteration, max_iterations)}")
    logger.info(f"Total interventions: {len(all_interventions)}")

    total_construction = sum(i.construction_cost_chf for i in all_interventions)
    total_maintenance = sum(i.maintenance_cost_annual_chf for i in all_interventions)
    logger.info(f"Total construction cost: {total_construction:,.0f} CHF")
    logger.info(f"Total annual maintenance: {total_maintenance:,.0f} CHF")

    # Save final enhanced prep (rename from last iteration)
    final_prep_path = output_dir / f"capacity_{network_label}_enhanced_network_prep.xlsx"
    if enhanced_prep_path.exists():
        import shutil
        shutil.copy(enhanced_prep_path, final_prep_path)
        logger.info(f"\nFinal enhanced prep saved to: {final_prep_path}")

    # Save interventions catalog
    interventions_df = pd.DataFrame([i.to_dict() for i in all_interventions])
    catalog_path = output_dir / "capacity_interventions.csv"
    interventions_df.to_csv(catalog_path, index=False)
    logger.info(f"Interventions catalog saved to: {catalog_path}")

    # Save final sections (with stations and segments for plotting)
    final_sections_path = output_dir / f"capacity_{network_label}_enhanced_network_sections.xlsx"
    with pd.ExcelWriter(final_sections_path, engine='openpyxl') as writer:
        original_stations_df.to_excel(writer, sheet_name='Stations', index=False)
        original_segments_df.to_excel(writer, sheet_name='Segments', index=False)
        current_sections_df.to_excel(writer, sheet_name='Sections', index=False)
    logger.info(f"Final sections saved to: {final_sections_path}")

    return all_interventions, final_prep_path, current_sections_df
