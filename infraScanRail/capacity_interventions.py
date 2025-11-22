"""
Phase 4: Capacity Enhancement Interventions

This module identifies capacity-constrained sections and designs infrastructure
interventions to bring all sections to ≥2 tphpd available capacity.

Intervention Types:
- Station Track: Add +1.0 track to middle station of multi-segment sections
- Passing Siding: Add +0.5 tracks to single-segment sections
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from openpyxl import load_workbook
import logging

# Import from existing modules
from capacity_calculator import _build_sections_dataframe, build_capacity_tables
from network_plot import plot_capacity_network
import cost_parameters

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CapacityIntervention:
    """
    Represents a single capacity enhancement intervention.

    Attributes:
        intervention_id: Unique identifier (e.g., "INT_ST_001")
        section_id: Section requiring intervention
        type: 'station_track' or 'segment_passing_siding'
        node_id: Station node ID (for station interventions)
        segment_id: Segment identifier (for passing siding interventions)
        tracks_added: 1.0 for station track, 0.5 for passing siding
        affected_segments: List of segment IDs impacted
        construction_cost_chf: Construction cost
        maintenance_cost_annual_chf: Annual maintenance cost
        length_m: Segment length (for passing sidings)
        current_tracks: Current track count before intervention (for cost scaling)
        iteration: Which iteration this intervention was applied in
    """
    intervention_id: str
    section_id: str
    type: str  # 'station_track' or 'segment_passing_siding'
    node_id: Optional[int]
    segment_id: Optional[str]
    tracks_added: float  # 1.0 or 0.5
    affected_segments: List[str]
    construction_cost_chf: float
    maintenance_cost_annual_chf: float
    length_m: Optional[float]
    current_tracks: float
    iteration: int = 1
    current_platforms: Optional[float] = None
    platforms_added: Optional[float] = None
    platform_cost_chf: Optional[float] = None

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
            'platform_cost_chf': self.platform_cost_chf
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
        # Single-segment section: Passing siding intervention
        segment_id = segments[0]

        # Find segment in segments_df
        from_node, to_node = segment_id.split('-')
        from_node, to_node = int(from_node), int(to_node)

        segment_row = segments_df[
            (segments_df['from_node'] == from_node) &
            (segments_df['to_node'] == to_node)
        ]

        if len(segment_row) == 0:
            logger.warning(f"Segment {segment_id} not found in segments_df")
            length_m = 0.0
            current_tracks = 1.0  # Default fallback
        else:
            length_m = float(segment_row.iloc[0]['length_m'])
            current_tracks = float(segment_row.iloc[0]['tracks'])

        # Determine intervention type and tracks to add
        # For short segments with fractional tracks, add full track to avoid negative costs
        is_fractional = (current_tracks % 1 == 0.5)
        is_short_segment = (length_m < 1000)

        if is_fractional and is_short_segment:
            # Short fractional segment: add +1.0 full track instead of +0.5 siding
            tracks_added = 1.0
            intervention_type = 'segment_passing_siding'
            logger.debug(f"  → Short segment ({length_m:.0f}m < 1000m) with fractional tracks: adding +1.0 full track")
        else:
            # Standard case: add +0.5 passing siding
            tracks_added = 0.5
            intervention_type = 'segment_passing_siding'

        intervention = CapacityIntervention(
            intervention_id=f"INT_PS_{intervention_counter:04d}",
            section_id=str(section_id),
            type=intervention_type,
            node_id=None,
            segment_id=segment_id,
            tracks_added=tracks_added,
            affected_segments=[segment_id],
            construction_cost_chf=0.0,
            maintenance_cost_annual_chf=0.0,
            length_m=length_m,
            current_tracks=current_tracks,
            iteration=iteration
        )

        if tracks_added == 1.0:
            logger.debug(f"  → Full track on segment {segment_id} ({length_m:.0f}m)")
        else:
            logger.debug(f"  → Passing siding on segment {segment_id} ({length_m:.0f}m)")

    return intervention


def calculate_intervention_cost(
    intervention: CapacityIntervention,
    maintenance_rate: float = None
) -> CapacityIntervention:
    """
    Calculate construction and maintenance costs for intervention.

    Cost formulas (based on track_cost_per_meter):

    Station track:
    - Standard: track_cost_per_meter × 500m × floor(current_tracks)

    Passing siding:
    - Fractional → Whole (e.g., 1.5 → 2.0, length ≥ 1000m):
      (segment_length_m × track_cost_per_meter) - (1000m × track_cost_per_meter × floor(current_tracks))
    - Full track addition (tracks_added = 1.0):
      segment_length_m × track_cost_per_meter × 1
    - Standard siding (tracks_added = 0.5, not fractional → whole):
      track_cost_per_meter × 1000m × floor(current_tracks)

    Args:
        intervention: Intervention object with current_tracks populated
        maintenance_rate: Annual maintenance as fraction of construction cost
                         (default: uses cost_parameters.yearly_maintenance_to_construction_cost_factor)

    Returns:
        Updated CapacityIntervention with costs filled
    """
    import math

    if maintenance_rate is None:
        maintenance_rate = cost_parameters.yearly_maintenance_to_construction_cost_factor

    # Calculate base track count (floor of current tracks)
    base_tracks = math.floor(intervention.current_tracks)

    if intervention.type == 'station_track':
        # Station track cost: track_cost_per_meter × 500m × base_tracks
        construction_cost = (
            cost_parameters.track_cost_per_meter *
            cost_parameters.station_siding_length_m *
            base_tracks
        )

        # Add platform costs if platforms need to be added
        if intervention.platforms_added and intervention.platforms_added > 0:
            platform_cost = (
                cost_parameters.platform_cost_per_unit *
                intervention.platforms_added
            )
            construction_cost += platform_cost
            intervention.platform_cost_chf = platform_cost

    elif intervention.type == 'segment_passing_siding':
        # Detect fractional → whole transition
        is_fractional = (intervention.current_tracks % 1 == 0.5)
        is_completing_track = (intervention.tracks_added == 0.5)  # Adding 0.5 to fractional

        if is_fractional and is_completing_track:
            # Fractional → Whole: Reuse existing siding infrastructure
            # Cost = (full segment × single track) - (siding already paid for)
            segment_length_m = intervention.length_m  # Already stored from design phase

            full_track_cost = segment_length_m * cost_parameters.track_cost_per_meter * 1
            siding_cost_paid = (
                cost_parameters.segment_siding_length_m *
                cost_parameters.track_cost_per_meter *
                base_tracks
            )
            construction_cost = full_track_cost - siding_cost_paid

        elif intervention.tracks_added == 1.0:
            # Full track addition (short segments with fractional tracks)
            # Cost = full segment length × single track cost
            segment_length_m = intervention.length_m
            construction_cost = segment_length_m * cost_parameters.track_cost_per_meter * 1

        else:
            # Standard passing siding: track_cost_per_meter × 1000m × base_tracks
            construction_cost = (
                cost_parameters.track_cost_per_meter *
                cost_parameters.segment_siding_length_m *
                base_tracks
            )
    else:
        raise ValueError(f"Unknown intervention type: {intervention.type}")

    maintenance_cost_annual = construction_cost * maintenance_rate

    # Update intervention object
    intervention.construction_cost_chf = construction_cost
    intervention.maintenance_cost_annual_chf = maintenance_cost_annual

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
            # Add +0.5 tracks to segment
            from_node, to_node = intervention.segment_id.split('-')
            from_node, to_node = int(from_node), int(to_node)

            mask = ((segments_df['from_node'] == from_node) &
                   (segments_df['to_node'] == to_node))

            if mask.sum() > 0:
                old_tracks = segments_df.loc[mask, 'tracks'].values[0]
                segments_df.loc[mask, 'tracks'] += 0.5
                new_tracks = segments_df.loc[mask, 'tracks'].values[0]
                segment_changes[intervention.segment_id] = (old_tracks, new_tracks)
                logger.debug(f"  Segment {intervention.segment_id}: "
                           f"{old_tracks} → {new_tracks} tracks")
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
    # modifying the core plotting functions in network_plot.py
    # For now, the enhanced plots show the updated track counts
    # Future enhancement: Add custom overlay for passing sidings

    return infrastructure_plot, capacity_plot


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
