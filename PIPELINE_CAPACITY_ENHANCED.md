# Capacity-Enhanced Development Pipeline - Complete Specification

## Core Concept
Transform infrastructure capacity from a **constraint** to an **expandable resource**. Instead of rejecting developments that exceed capacity, proactively identify and cost infrastructure upgrades, then evaluate developments with full upgrade costs included.

---

## Phase-by-Phase Workflow

### Phase 0: Initialization
- Initialize `PipelineConfig` with paths, caching, runtime tracking
- Synchronize capacity calculator paths
- Create output directory structure:
  - `infraScanRail/plots/network/developments/{dev_id}/`
  - `data/capacity/interventions/`
  - `data/capacity/enhanced_baseline/`

### Phase 1: Base Network Generation
- **Function**: `run_phase_one()`
- **Calls**: `generate_base_network()` from `build_network.py`
- **Purpose**: Import and prepare the baseline railway network
- **Output**: Baseline network files
  - `points.gpkg`
  - `edges_in_corridor.gpkg`

### Phase 2: Capacity Workbook Generation
- **Function**: `run_phase_two()`
- **Process**:
  1. Call `capacity_calculator.build_capacity_tables()`
  2. Export to `capacity_AK_2035_network.xlsx`
  3. **MANUAL CHECKPOINT**: User adds tracks, platforms, speeds
  4. Load enriched `*_prep.xlsx`
  5. Generate baseline plots (infrastructure + speed + service frequency)
- **Output**:
  - `capacity_AK_2035_network.xlsx` (initial)
  - `capacity_AK_2035_network_prep.xlsx` (manually enriched)
  - Baseline visualization plots

### Phase 3: Sections Analysis
- **Function**: `run_phase_three()`
- **Process**:
  1. Call `_build_sections_dataframe()` to identify continuous track sections
  2. Export sections workbook (Stations, Segments, Sections sheets)
  3. Generate capacity plot showing track counts and capacity limits
  4. Apply section capacities to segments DataFrame
- **Output**:
  - `sections_workbook.xlsx`
  - Capacity visualization plot

### Phase 4: Capacity Enhancement Interventions ⭐ NEW CORE PHASE

**Goal**: Create catalog of infrastructure interventions to bring all sections to ≥2 tphpd available capacity

- **Function**: `run_phase_four()`
- **Process**:
  1. **Identify** sections with <2 tphpd available capacity
  2. **Design interventions**:
     - **Multi-segment section**: Add +1 track to middle station (splits section operationally)
     - **Single-segment section**: Add +0.5 tracks (passing siding at midpoint)
  3. **Cost interventions**:
     - User provides: station track cost (CHF), passing siding cost (CHF/km)
     - Calculate maintenance using main.py formulas
  4. **Update workbook**: Apply +0.5/+1 track adjustments to prep workbook
  5. **Recalculate capacity**:
     - Multi-segment sections treated as two half-length sections
     - Single-segment sections updated capacity formula with +0.5 tracks
  6. **Visualize**: Enhanced network infrastructure + capacity plots

- **Outputs**:
  - `capacity_interventions.csv` (intervention catalog: ID, type, location, cost)
  - `enhanced_baseline_prep.xlsx` (workbook with track adjustments)
  - Enhanced network plots (infrastructure + capacity)

- **Implementation Details**: See PHASE_4_IMPLEMENTATION.md

### Phase 5: Capacity-Aware Development Generation ⭐ MODIFIED

**Goal**: Generate developments using main.py logic, but filter for capacity feasibility

- **Function**: `run_phase_five()`
- **Process**:
  1. Generate developments using main.py logic:
     - Line extensions (connect endpoint stations to nearby stations)
     - New direct connections (missing connections in graph)
  2. **Capacity filter**:
     - Each development assumes 2 tphpd frequency
     - Check if path segments have ≥2 tphpd available in enhanced baseline
     - Reject developments that still exceed capacity after Phase 4 interventions
  3. Convert to `ServiceProposal` objects
  4. Store in `InfrastructurePlan`

- **Output**:
  - `feasible_developments.gpkg`
  - Proposal list

### Phase 6: Development Costing ⭐ MODIFIED

**Goal**: Calculate total costs including both development and infrastructure interventions

- **Function**: `run_phase_six()`
- **Process**: For each development:
  1. **Identify used interventions**: Which Phase 4 interventions lie on development path?
  2. **Calculate total construction cost**:
     - Base development cost (track/tunnel/bridge as in main.py:162-182)
     - **Add**: Infrastructure intervention costs (full cost, no sharing)
  3. **Calculate maintenance** (same formulas as main.py)
  4. Store cost breakdown per development

- **Cost Attribution Rule**: Each development independently pays full cost of ALL interventions it uses (no cost sharing between developments)

- **Output**:
  - `development_costs.csv`
    - Columns: dev_id, base_cost, infra_intervention_cost, total_construction, maintenance

### Phase 7: Economic Evaluation ⭐ MODIFIED

**Goal**: Calculate cost-benefit analysis on development network

- **Function**: `run_phase_seven()`
- **Process**:
  1. For each development, create **Development Network**:
     - Development Network = Enhanced Baseline + Development service
  2. Calculate travel times on Development Network (using main.py:497-580 logic)
  3. Generate demand scenarios (main.py:147-152)
  4. Compute travel time savings (main.py:360-430)
  5. Apply discounting (main.py:176-187)
  6. Calculate **net benefit** = monetized_savings - total_costs (from Phase 6)
  7. Calculate **CBA ratio** = benefits / costs

- **Output**:
  - `development_evaluations.csv`
    - Columns: dev_id, benefits, costs, net_benefit, cba_ratio

### Phase 8: Top 10 Selection

**Goal**: Rank and select best developments

- **Function**: `run_phase_eight()`
- **Process**:
  1. Sort developments by net_benefit (descending)
  2. Select top 10
  3. Export summary table

- **Output**:
  - `top_10_developments.csv`

### Phase 9: Development Visualization ⭐ NEW

**Goal**: Generate comprehensive artefacts for top 10 developments

- **Function**: `run_phase_nine()`
- **Process**: For each top-10 development:
  1. Create development directory: `infraScanRail/plots/network/developments/{dev_id}/`
  2. Generate plots:
     - **Infrastructure network** (showing upgraded sections + new development)
     - **Service frequency** (baseline vs. with development)
     - **Capacity visualization** (utilization before/after)
     - Additional plots from main.py (cost-benefit, cumulative distribution, etc.)
  3. Export development-specific data:
     - Travel time matrices
     - Cost breakdown
     - Used interventions list

- **Output**:
  - Per-development folders: `infraScanRail/plots/network/developments/{dev_id}/`
  - Plots: infrastructure, service, capacity, cost-benefit, etc.
  - CSVs: travel times, costs, interventions used

### Phase 10: Final Report

**Goal**: Comprehensive pipeline summary

- **Function**: `run_phase_ten()`
- **Output**: `capacity_enhanced_pipeline_report.txt` containing:
  - Phase 4 summary (intervention count, total infra cost pool)
  - Phase 5 summary (developments generated, capacity-filtered count)
  - Phase 8 summary (top 10 net benefits, CBA ratios)
  - Paths to all key artefacts

---

## Network State Management

Three distinct network states throughout pipeline:

1. **Baseline Network** (Phases 1-3):
   - Original infrastructure
   - No interventions
   - Used for: Initial capacity analysis

2. **Enhanced Baseline** (Phase 4 output):
   - Baseline + all capacity interventions
   - Infrastructure catalog ready for developments
   - Used for: Development generation filtering, cost attribution

3. **Development Network** (Phase 7):
   - Enhanced Baseline + specific development service
   - Used for: Travel time calculations, benefit analysis

---

## Key Design Decisions

### Configuration Parameters:
- **Capacity threshold**: 2 tphpd (trains per hour per direction)
- **Development frequency**: 2 tphpd (assumed for all proposals)
- **Selection criteria**: Net benefit ranking (top 10)

### Cost Attribution:
- **No cost sharing**: Each development independently pays full cost of all interventions it uses
- If Development A uses upgraded Section X and Development B also uses Section X, both pay full upgrade cost

### Capacity Filter Logic:
- **Generation**: Same as main.py (line extensions + new direct connections)
- **Filtering**: Reject developments that exceed capacity even after Phase 4 interventions
- **Check**: For each proposal path segment, verify available capacity ≥ 2 tphpd in enhanced baseline

### Workbook Update Strategy:
- **Attribute changes only**: No network topology changes (no new nodes/edges)
- **Station interventions**: Increase `tracks` attribute by +1
- **Segment interventions**: Increase `tracks` attribute by +0.5

---

## Data Flow

```
Phase 1: Baseline Network Files (points.gpkg, edges_in_corridor.gpkg)
    ↓
Phase 2: capacity_AK_2035_network.xlsx → [MANUAL ENRICHMENT] → capacity_AK_2035_network_prep.xlsx
    ↓
Phase 3: sections_workbook.xlsx + capacity_plot.png
    ↓
Phase 4: capacity_interventions.csv + enhanced_baseline_prep.xlsx + enhanced_plots/
    ↓
Phase 5: feasible_developments.gpkg (capacity-filtered)
    ↓
Phase 6: development_costs.csv (base + infra interventions)
    ↓
Phase 7: development_evaluations.csv (benefits, costs, net_benefit, CBA)
    ↓
Phase 8: top_10_developments.csv
    ↓
Phase 9: plots/network/developments/{dev_id}/*.png + dev_{dev_id}_data.csv
    ↓
Phase 10: capacity_enhanced_pipeline_report.txt
```

---

## User Inputs Required

### Phase 2: Infrastructure Attributes (Manual Excel Enrichment)
- Tracks (number of tracks per segment/station)
- Platforms (number of platforms per station)
- Speed (design speed in km/h per segment)

### Phase 4: Intervention Costs
- `station_track_cost_chf`: Cost per additional station track (CHF)
- `segment_passing_siding_cost_chf_per_km`: Cost per km for passing siding (CHF/km)

---

## Files to Create/Modify

### New Files:
1. **`main_capacity_enhanced.py`**: New main orchestration script (Phases 0-10)
2. **`capacity_interventions.py`**: Phase 4 intervention logic (see PHASE_4_IMPLEMENTATION.md)

### Modified Files:
1. **`capacity_calculator.py`**:
   - Add `load_enhanced_baseline()` function
   - Modify `_build_sections_dataframe()` to handle section splits

---

## Core Data Structures

### ServiceProposal (existing from main_test.py)
```python
@dataclass
class ServiceProposal:
    dev_id: Optional[int]
    service: str
    frequency: float  # trains per hour (2 tphpd for all proposals)
    from_node: int
    to_node: int
    via: Optional[str]
    travel_time_min: Optional[float]
    path_length_m: Optional[float]
```

### CapacityIntervention (new)
```python
@dataclass
class CapacityIntervention:
    intervention_id: str
    section_id: str
    type: str  # 'station_track' or 'segment_passing_siding'
    node_id: Optional[int]  # For station interventions
    segment_id: Optional[str]  # For segment interventions (from_node-to_node)
    tracks_added: float  # 1.0 or 0.5
    affected_segments: List[str]  # Segment IDs impacted
    construction_cost_chf: float
    maintenance_cost_annual_chf: float
    length_m: Optional[float]  # For segment interventions
```

---

## Differences from main.py (Original Pipeline)

| Aspect | main.py (Original) | Capacity-Enhanced Pipeline |
|--------|-------------------|---------------------------|
| **Capacity Awareness** | None | Full capacity evaluation with proactive interventions |
| **Infrastructure** | Static | Dynamic - can be upgraded |
| **Filtering** | None (all developments scored) | Capacity filter after generation |
| **Cost Components** | Development only | Development + infrastructure interventions |
| **Network States** | Single baseline | Three states: Baseline, Enhanced, Development |
| **Development Rejection** | No rejection mechanism | Reject if exceeds enhanced baseline capacity |
| **Infrastructure Upgrades** | Not considered | Cataloged, costed, and attributed to developments |

---

## Differences from main_test.py (First Capacity Attempt)

| Aspect | main_test.py | Capacity-Enhanced Pipeline |
|--------|--------------|---------------------------|
| **Capacity Philosophy** | Constraint (reject/modify proposals) | Expandable resource (upgrade infrastructure) |
| **Manual Checkpoints** | Two (workbook enrichment + conflict resolution) | One (workbook enrichment only) |
| **Conflict Resolution** | Interactive (drop/reduce/shorten) | Automatic (infrastructure upgrades) |
| **User Selection** | Manual selection of interventions | Automatic top-10 selection |
| **Infrastructure Interventions** | Not cataloged | Cataloged with full costing |
| **Cost Attribution** | Not applicable | Each development assumes intervention costs |
| **Output Structure** | Selected interventions only | Top 10 with comprehensive artefacts |

---

## Implementation Priority

### Phase 0-3:
- Reuse existing code from main_test.py with minimal modifications

### Phase 4:
- **NEW - Highest complexity**
- Requires new `capacity_interventions.py` module
- See PHASE_4_IMPLEMENTATION.md for detailed design

### Phase 5-7:
- Modify existing main.py logic
- Key changes: capacity filtering, cost attribution, network state management

### Phase 8-10:
- Straightforward implementation
- Mostly data aggregation and reporting
