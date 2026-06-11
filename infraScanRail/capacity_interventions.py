"""
Capacity Expansion Interventions
Last modified: 2026-06-09

Identifies capacity-constrained sections and designs Tier-1 infrastructure
interventions (real nodes + adjusted segments) to restore the reactive trigger
margin (≥ settings.SVC_INT_CAP_THRESHOLD_TPHPD tphpd available capacity) on the
constrained sections. Phase 5C is the sole, cost-bearing CAP generator.

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


def _pick_col(df: pd.DataFrame, *names: str) -> str:
    """Return the first of `names` present in df (period-schema compatibility)."""
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"none of {names} present in columns {list(df.columns)}")


def _alias_cols(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Add target columns aliased from source columns when missing (non-destructive).

    mapping: {target_name: source_name}. Used to bridge the period-suffixed sections
    schema (Track_Count/Length/…) to the names the capacity engine reads (tracks/length_m/…).
    """
    out = df.copy()
    for target, source in mapping.items():
        if target not in out.columns and source in out.columns:
            out[target] = out[source]
    return out


def identify_capacity_constrained_sections(
    sections_df: pd.DataFrame,
    threshold_tphpd: float = 1.0
) -> pd.DataFrame:
    """
    Identify sections with available capacity below threshold.

    Available capacity = Capacity - total_tphpd (remaining capacity). Accepts both the
    legacy unsuffixed columns and the current period-suffixed schema
    (Capacity_peak / total_tphpd_peak).

    Args:
        sections_df: Sections DataFrame from Phase 3
        threshold_tphpd: Minimum required available capacity (default: 1.0)

    Returns:
        DataFrame of constrained sections
    """
    logger.info(f"Identifying sections with available capacity < {threshold_tphpd} tphpd")

    # Calculate available capacity (remaining capacity)
    sections_df = sections_df.copy()
    cap_col = _pick_col(sections_df, 'Capacity', 'Capacity_peak')
    load_col = _pick_col(sections_df, 'total_tphpd', 'total_tphpd_peak')
    sections_df['available_capacity'] = (
        pd.to_numeric(sections_df[cap_col], errors='coerce')
        - pd.to_numeric(sections_df[load_col], errors='coerce')
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

            # Look up segment length (orientation-agnostic)
            seg_row = segments_df[_seg_mask(segments_df, from_node, to_node)]

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


def _seg_mask(df: pd.DataFrame, from_node: int, to_node: int) -> pd.Series:
    """Orientation-agnostic segment mask: match (from,to) OR (to,from).

    Section ``segment_sequence`` and the Segments sheet can order a segment's two
    nodes differently; a single-direction match silently misses the row (and made
    the enhancement loop a no-op). Mirror _lookup_segment's both-directions logic.
    """
    f = pd.to_numeric(df['from_node'], errors='coerce')
    t = pd.to_numeric(df['to_node'], errors='coerce')
    return ((f == from_node) & (t == to_node)) | ((f == to_node) & (t == from_node))


def _target_key(intervention: 'CapacityIntervention'):
    """Physical target identity (orientation-agnostic) for once-per-run dedup."""
    if intervention.type == 'station_track':
        return ('station', int(intervention.node_id))
    fn, tn = str(intervention.segment_id).split('-')
    return ('segment', frozenset((int(fn), int(tn))))


def _connected_equivalent(station_nr: int, segments_df: pd.DataFrame) -> float:
    """Track requirement a station's adjacent line topology implies.

    Mirrors the capacity infrastructure plot (`_station_colour`): one adjacent
    segment → its track count (terminus); two → their mean (through station);
    three+ → their sum (junction). No usable adjacency → 1.0.
    """
    tcol = 'tracks' if 'tracks' in segments_df.columns else 'Num_Tracks'
    fn = pd.to_numeric(segments_df['from_node'], errors='coerce')
    tn = pd.to_numeric(segments_df['to_node'], errors='coerce')
    adj = segments_df[(fn == float(station_nr)) | (tn == float(station_nr))]
    vals = pd.to_numeric(adj[tcol], errors='coerce').dropna()
    vals = [float(v) for v in vals if v > 0]
    if not vals:
        return 1.0
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 2:
        return sum(vals) / 2.0
    return sum(vals)


def _required_station_tracks(connected_equivalent: float, terminating: bool,
                             mixed: bool) -> int:
    """Station tracks needed for its role: line equivalent + turnback + overtaking.

    `terminating` adds the turnback track (a terminating train must clear the
    through track); `mixed` adds the overtaking track for sections carrying both
    stopping and passing services. Floor of 2: crossing is impossible on a single
    track even when it matches the line (the plot's matched-at-1 = red rule).
    """
    import math
    need = math.ceil(connected_equivalent - 1e-9)
    if terminating:
        need += 1
    if mixed:
        need += 1
    return max(2, need)


def design_section_intervention(
    section: pd.Series,
    segments_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    intervention_counter: int,
    iteration: int = 1,
    termini_nodes: Optional[set] = None,
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

        # F5: add what the crossing/overtaking logic requires, not a blanket +1.
        equiv = _connected_equivalent(middle_station_id, segments_df)

        def _svc_str(v):
            s = str(v if v is not None else '').strip()
            return '' if s.lower() == 'nan' else s

        mixed = bool(_svc_str(section.get('stopping_services'))) and \
            bool(_svc_str(section.get('passing_services')))
        terminating = bool(termini_nodes and int(middle_station_id) in termini_nodes)
        required = _required_station_tracks(equiv, terminating, mixed)
        tracks_added = float(max(1, required - int(current_tracks)))

        intervention = CapacityIntervention(
            intervention_id=f"INT_ST_{intervention_counter:04d}",
            section_id=str(section_id),
            type='station_track',
            node_id=middle_station_id,
            segment_id=None,
            tracks_added=tracks_added,
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

        segment_row = segments_df[_seg_mask(segments_df, from_node, to_node)]

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


def _piece_rate(structure: str) -> float:
    """Per-meter construction rate for a composition piece (shared with the CC pricing)."""
    from infra_ints_connecting_curve import _STRUCT_RATE_ATTR
    attr = _STRUCT_RATE_ATTR.get(str(structure).strip().lower())
    base = float(getattr(cost_parameters, 'track_cost_per_meter', 33250.0))
    return float(getattr(cost_parameters, attr, base)) if attr else base


def calculate_intervention_cost(
    intervention: CapacityIntervention,
    maintenance_rate: float = None,
    composition: Optional[Dict] = None,
) -> CapacityIntervention:
    """Calculate construction and maintenance costs.

    Cost formulas
    -------------
    station_track:
        cost = cost_parameters.station_siding_costs · tracks_added
        (+ platform_cost_per_unit · platforms_added, if any).

    segment_passing_siding (composition-aware when `composition` carries the host):
        strategy='extra_track'           → Σ composition pieces · per-structure rate
        strategy='siding_with_junctions' → pieces overlapped by the centered siding
                                           window · per-structure rate
        host missing from `composition`  → flat length · track_cost_per_meter + warning.

    Maintenance: construction_cost · maintenance_rate.

    Args:
        intervention: Intervention object with tracks_added/length_m populated.
        maintenance_rate: annual fraction. Defaults to
            cost_parameters.yearly_maintenance_to_construction_cost_factor.
        composition: {frozenset(from_nr, to_nr): [(structure, length, start_m, end_m),
            …]} chainage-ordered pieces of the host segments (tunnel/bridge priced at
            their own rates, mirroring the CC F4 fix).
    """
    if maintenance_rate is None:
        maintenance_rate = cost_parameters.yearly_maintenance_to_construction_cost_factor

    if intervention.type == 'station_track':
        tracks_added = max(1.0, float(intervention.tracks_added or 1.0))
        construction_cost = cost_parameters.station_siding_costs * tracks_added
        if intervention.platforms_added and intervention.platforms_added > 0:
            platform_cost = (
                cost_parameters.platform_cost_per_unit *
                intervention.platforms_added
            )
            construction_cost += platform_cost
            intervention.platform_cost_chf = platform_cost

    elif intervention.type == 'segment_passing_siding':
        track_length_m = float(intervention.length_m or 0.0)
        pieces = None
        if composition and intervention.segment_id:
            fn, tn = str(intervention.segment_id).split('-')
            pieces = composition.get(frozenset((int(fn), int(tn))))
        if pieces:
            if intervention.strategy == 'extra_track':
                construction_cost = sum(plen * _piece_rate(s)
                                        for s, plen, _a, _b in pieces)
            else:
                total = max(p[3] for p in pieces)
                l_sid = min(float(intervention.siding_length_m or track_length_m), total)
                kp_a, kp_b = (total - l_sid) / 2.0, (total + l_sid) / 2.0
                construction_cost = sum(
                    max(0.0, min(b, kp_b) - max(a, kp_a)) * _piece_rate(s)
                    for s, _plen, a, b in pieces)
        else:
            if composition is not None:
                print(f"  [cap-cost] {intervention.segment_id}: no composition for the "
                      f"host segment — flat per-meter rate used")
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

    # Track columns must be float — siding_with_junctions applies a +0.5 delta that
    # cannot be assigned into an int64 column.
    for _c in ('tracks', 'platforms'):
        if _c in stations_df.columns:
            stations_df[_c] = pd.to_numeric(stations_df[_c], errors='coerce').astype(float)
    if 'tracks' in segments_df.columns:
        segments_df['tracks'] = pd.to_numeric(segments_df['tracks'], errors='coerce').astype(float)

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

            mask = _seg_mask(segments_df, from_node, to_node)

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
# Cap-int geometry lookup helpers (siding planning; used by infra_ints_capacity)
# ─────────────────────────────────────────────────────────────────────────────

_CAP_JUNCTION_NR_START = 9_100_001


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


def run_phase_four(
    original_sections_df: pd.DataFrame,
    original_segments_df: pd.DataFrame,
    original_stations_df: pd.DataFrame,
    prep_workbook_path: Path,
    output_dir: Path,
    network_label: str,
    threshold_tphpd: float = 1.0,
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
        threshold_tphpd: Minimum required available capacity (default: 1.0)
        max_iterations: Maximum number of intervention iterations

    Returns:
        Tuple of (interventions_catalog, enhanced_prep_path, final_sections_df)
    """
    logger.info("=" * 60)
    logger.info("Phase 4: Capacity Enhancement Interventions")
    logger.info("=" * 60)

    # Bridge the current period-suffixed sheet schema to the names the engine reads
    # (Stations_Peak: Track_Count/Platform_Count; Segments_Peak: Length/Num_Tracks/Average_Speed).
    original_stations_df = _alias_cols(
        original_stations_df, {'tracks': 'Track_Count', 'platforms': 'Platform_Count'})
    original_segments_df = _alias_cols(
        original_segments_df, {'length_m': 'Length', 'tracks': 'Num_Tracks', 'speed': 'Average_Speed'})

    # Initialize
    all_interventions = []
    intervention_counter = 1
    treated_targets: set = set()   # physical targets treated this run (once-per-run dedup)

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

        # Step 2: Design interventions for this iteration. Skip any target already
        # treated this run (dedup safeguard): keeps the catalogue to one intervention
        # per physical segment/station even if a section needs escalation.
        iteration_interventions = []
        for idx, section in constrained_sections.iterrows():
            intervention = design_section_intervention(
                section,
                original_segments_df,
                original_stations_df,
                intervention_counter,
                iteration
            )
            key = _target_key(intervention)
            if key in treated_targets:
                logger.info(f"  skip section {section.get('section_id')}: "
                            f"target {key} already treated this run")
                continue
            treated_targets.add(key)
            intervention_counter += 1
            iteration_interventions.append(intervention)

        if not iteration_interventions:
            logger.info("No new (untreated) targets among constrained sections — stopping")
            break

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
