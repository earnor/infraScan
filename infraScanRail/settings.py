"""
infraScanRail — Pipeline Settings
Last modified: 2026-06-05

Central configuration file. Sections are numbered by the main_new.py phase whose
choices they drive (Phase 1 → 4C). Sections not yet wired into main_new are
marked 'X.'. Cross-cutting toggles live in the appendix (A1 Plots, A2 Cache,
A3 Physical attributes); settings read only by the legacy mains are grouped last.
Edit values here; all downstream modules read from this file.
"""

from shapely.geometry import Polygon

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATA INITIALISATION  (study area + catchment area)
# ═══════════════════════════════════════════════════════════════════════════════

# --- Study area ---
# 'coordinates' — use perimeter_infra_generation polygon defined below
# 'admin'       — dissolve SwissBoundaries admin units (fill STUDY_AREA_ADMIN_* below)
STUDY_AREA_METHOD = 'admin'

# Polygon used when STUDY_AREA_METHOD = 'coordinates'  (EPSG:2056)
perimeter_infra_generation = Polygon([
    (2700989.862, 1235663.403),
    (2708491.515, 1239608.529),
    (2694972.602, 1255514.900),
    (2687415.817, 1251056.404),
])

# Used when STUDY_AREA_METHOD = 'admin'
# Admin level: 'national' | 'cantonal' | 'bezirke' | 'municipal'
STUDY_AREA_ADMIN_LEVEL = 'municipal'
STUDY_AREA_ADMIN_NAMES = [
    'Dübendorf', 'Fällanden', 'Fehraltorf', 'Gossau (ZH)', 'Greifensee', 'Grüningen',
    'Hinwil', 'Illnau-Effretikon', 'Mönchaltorf', 'Pfäffikon', 'Schwerzenbach', 'Seegräben',
    'Uster',  'Volketswil', 'Wangen-Brüttisellen', 'Wetzikon (ZH)',
]
STUDY_AREA_ADMIN_SUBDIVISIONS = {'bezirke': [], 'municipal': []}  # optional extra units in subdivisions of the choosen admin level
STUDY_AREA_BUFFER_M = 3000                    # margin (m) for feeder network edge handling

# --- Catchment area ---
# Catchment boundary is always admin-based (study area must be fully contained within)
# Admin level: 'national' | 'cantonal' | 'bezirke' | 'municipal'
CATCHMENT_AREA_ADMIN_LEVEL = 'cantonal'
CATCHMENT_AREA_ADMIN_NAMES = ['Zürich']       # list of admin entity names to dissolve
CATCHMENT_AREA_ADMIN_SUBDIVISIONS = {'bezirke': [], 'municipal': []}
CATCHMENT_AREA_BUFFER_M = 5000                # buffer (m) around boundary for GTFS spatial filter
CATCHMENT_CANTON_ABBREV = 'ZH'               # canton abbreviation — used in folder and file naming

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — DATA PREPARATION  (raw filter inputs for infra + services)
# ═══════════════════════════════════════════════════════════════════════════════

# Base year for the population/employment grids, OD matrices, and scenario scaling
POPULATION_BASE_YEAR = 2035

# Named output folder for infrabuild_filter_network (Phase 2 skip-guard)
INFRA_RAW_VERSION   = 'Raw_ZH'             # infrabuild_filter_network output: data/Infrastructure/<INFRA_RAW_VERSION>/

GTFS_RAW_VERSION    = 'GTFS_SVC2026_CH_raw'    # services_filter_gtfs input under data/Network/GTFS_Timetable/
GTFS_FILTER_VERSION = 'GTFS_SVC2026_ZH'        # services_filter_gtfs output: data/Network/GTFS_Timetable/<GTFS_FILTER_VERSION>/

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3A — INFRASTRUCTURE NETWORK BUILD
# ═══════════════════════════════════════════════════════════════════════════════

# 'Build_New'           — run full infrabuild pipeline to create a new named version
# 'AS_2026_ZH'          — BAV network as-is for 2026 (must already exist on disk)
# 'AS_2026_ZH_enhanced' — AS_2026_ZH enriched with projected svc travel times and corrections
# 'AS_2035_ZH'          — BAV network as-is for 2035 (must already exist on disk)
# 'AS_2035_ZH_enhanced' — AS_2035_ZH enriched with projected svc travel times and corrections
INFRA_VERSION = 'Build_New'

INFRA_BUILD_NEW_NAME = 'AS_2026_ZH'        # name for the new version — used only when INFRA_VERSION = 'Build_New'

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3B — SERVICES NETWORK BUILD
# ═══════════════════════════════════════════════════════════════════════════════

# 'Build_New'   — run full services pipeline to create a new named version
# 'AK_2026'     — scheduled services as of 2026 timetable
# 'AK_2026_S18' — AK_2026 with S18 line included
# 'AK_2035'     — scheduled services as of 2035 timetable
# 'AK_2035_S18' — AK_2035 with S18 line included
SVC_VERSION = 'Build_New'

SVC_BUILD_NEW_NAME = 'AK_2026_S18'                 # name for the new version — used only when SVC_VERSION = 'Build_New'

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3C — CAPACITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

# 'None'      — skip capacity phases entirely
# 'Set_Value' — apply a fixed trains/hour/direction threshold to all sections
# 'Dynamic'   — full iterative capacity calculator workflow (capacity_workflow_wrapper.py)
CAPACITY_MODE = 'Dynamic'

# Spatial scope for capacity analysis (used by main_new.py Phase 3C and capacity_workflow_wrapper.py)
# 'SA' — Study Area only: infra/services filtered to nodes within the study area
# 'CA' — Catchment Area: infra/services filtered to nodes within the catchment area boundary
CAPACITY_SCOPE = 'CA'

# Capacity method per spatial scope — used when CAPACITY_SCOPE = 'SA' or 'CA'
# SA method must be the same or more precise than CA method.
# Valid combinations: both Dynamic | SA Dynamic + CA Set_Value | both Set_Value
CAPACITY_MODE_SA = 'Dynamic'    # method applied to Study Area sections
CAPACITY_MODE_CA = 'Dynamic'  # method applied to Catchment Area sections outside the SA

# Capacity grouping strategy — controls automated decisions in the capacity workflow
# 'manual'       — prompt for each capacity grouping decision
# 'conservative' — always choose the lowest capacity option
# 'baseline'     — always choose the middle option
# 'optimal'      — always choose the highest capacity option
CAPACITY_GROUPING_STRATEGY = 'baseline'

CAPACITY_SET_VALUE = 6             # trains/hour/direction — used when CAPACITY_MODE = 'Set_Value'
capacity_threshold = 2.0           # minimum available capacity (tphpd) — Dynamic only
max_enhancement_iterations = 10    # max Phase 4 enhancement iterations — Dynamic only

# Internal — set dynamically in main (do not edit)
baseline_network_for_developments = None

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4A — CATCHMENT ALLOCATION
# ═══════════════════════════════════════════════════════════════════════════════

# 'Municipal' — each commune assigned wholly to one rail station (centroid-based)
# 'PT_Feeder' — access-time raster decides which cells belong to which station
# OD allocation method is derived automatically:
#   Municipal  ->  commune-to-station lookup table
#   PT_Feeder  ->  communal OD re-weighted by PT-feeder catchment shares
CATCHMENT_METHOD = 'PT_Feeder'

# Travel-cost method — controls how access-time components are combined into a generalised cost
# 'calibrated' — use literature-calibrated weights from cost_parameters.py (W_IVT/W_WAIT/W_WALK/W_BIKE/W_TRANSFER and the comfort-weighted transfer penalty)
# 'absolute'   — set all weights to 1.0 at runtime (raw minutes, no weighting)
TRAVEL_COST_METHOD = 'calibrated'

# Transfer cost model — requires TRAVEL_COST_METHOD = 'calibrated' for the full literature value; 'absolute' uses raw minutes regardless.
# 'fixed_value' — flat 12.1 min eq. IVT penalty (Axhausen 2014)
# 'explicit'    — W_TRANSFER x (transfer walk time + wait time based on connecting headway)
TRANSFER_COST_MODEL = 'explicit'

# OD attribution mode — how communal OD is split across a commune's stations in the PT-Feeder branch.
# 'specific' — origins weighted by population share, destinations by FTE share (production/attraction split)
# 'blended'  — both origins and destinations weighted by OD_SCALING_POP_WEIGHT×pop_share + OD_SCALING_EMPL_WEIGHT×empl_share (symmetric)
# Municipal branch is 1:1 commune→station and ignores this setting.
OD_ATTRIBUTION_MODE = 'specific'

# DEPRECATED — the main pipeline always runs on the full_day union (top-level
# files), which is the single source of truth. This setting no longer affects
# catchment OD, routing, Güteklassen, or the gateway split. It is retained only
# as the default for the standalone catchment_allocate.py CLI temporal-diagnostic
# menu (peak / offpeak / all_day inspection of the subfolder networks). Leave at
# 'full_day' for normal runs.
TEMPORAL = 'full_day'

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4B — STATION OD MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

# OD scaling weights — control how population and employment growth are blended when computing per-commune OD growth factors.
# OD_SCALING_EMPL_WEIGHT = 0.0 because employment is derived from population (empl scales with pop), so adding employment weight changes nothing.
# Set OD_SCALING_EMPL_WEIGHT > 0 only when an independent employment projection is available (e.g. BFS STATENT projections beyond 2023).
OD_SCALING_POP_WEIGHT  = 1.0
OD_SCALING_EMPL_WEIGHT = 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4C — NETWORK ASSIGNMENT  (passenger routing)
# ═══════════════════════════════════════════════════════════════════════════════

# Which assignment method to run in Phase 4C (single method only).
# 'shortest_path' — deterministic all-or-nothing on generalised cost (one path/pair)
# 'logit'         — k-shortest-paths Logit route choice (demand split across paths)
ROUTING_ASSIGNMENT_METHOD = 'logit'

# Logit choice-set bounds (ignored by 'shortest_path').
ROUTING_K_PATHS         = 5       # max accepted paths per OD pair (Yen k-shortest)
ROUTING_COST_WINDOW_MIN = 15.0    # accept paths within best_gc + this many minutes ...
ROUTING_COST_WINDOW_PCT = 0.5     # ... or best_gc * (1 + this), whichever is larger
ROUTING_MAX_TRANSFERS   = 3       # transfer cap for accepted paths
ROUTING_MAX_EXAMINE     = 200     # hard guard on Yen candidates inspected per pair
# Logit candidate-generation engine: 'table' (connection-table itinerary proposal,
# 0/1/2-transfer + shortest path, fast) or 'yen' (k-shortest baseline, exact, slow).
ROUTING_LOGIT_ENGINE    = 'table'

# ═══════════════════════════════════════════════════════════════════════════════
# X. INTERVENTIONS  (not yet wired into main_new)
# ═══════════════════════════════════════════════════════════════════════════════
# Service interventions modify the rail offer (timetable + routing); the ones that also require new physical infrastructure pull in the corresponding infra
# intervention via the 'requires_infra' column of their svc-int xlsx.
#
# Available intervention types:
#   'EXT'  — Extended lines: route extensions, truncations and reroutes of existing services (svc-only — operates over existing infra)
#   'CC'   — Connecting curves: new short infra links between existing lines that enable new through-routings (infra + svc impact)
#
# INFRA_INT_MODE — pick which type(s) to run this session:
#   'NONE' — baseline, no intervention applied
#   'ALL'  — apply every type listed above
#   'EXT'  — apply only extended lines
#   'CC'   — apply only connecting curves
INFRA_INT_MODE = 'NONE'

# Per-type intervention ID start counters. Newly generated interventions are numbered sequentially from these starts; existing hand-authored rows keep
# their assigned IDs. Each type gets its own 1000-block (mirrors the old main_cap convention: dev_id_start_extended_lines = 100000, etc.).
DEV_ID_START_EXT = 100000              # extended-line interventions
DEV_ID_START_CC  = 101000              # connecting-curve interventions
DEV_ID_START_CAP = 102000              # capacity passing sidings (auto-generated by the capacity workflow)

# Cap interventions are not listed in INFRA_INT_MODE — they are auto-generated
# whenever CAPACITY_MODE is set ('Set_Value' or 'Dynamic') and written to
# data/Infrastructure/Developments/cap_interventions.gpkg.

# ═══════════════════════════════════════════════════════════════════════════════
# X. SCENARIOS  (not yet wired into main_new)
# ═══════════════════════════════════════════════════════════════════════════════

# 'GENERATED' — Monte Carlo random scenarios from population growth models
# 'STATIC_9'  — Fixed set of 9 canonical scenarios
# 'dummy'     — Minimal placeholder scenarios for testing
scenario_type = 'GENERATED'

amount_of_scenarios = 100
start_year_scenario = 2018
end_year_scenario = 2100
start_valuation_year = 2050

# ═══════════════════════════════════════════════════════════════════════════════
# A1. PLOTS  (per-phase visualisation toggles — all phases)
# ═══════════════════════════════════════════════════════════════════════════════

# Per-pipeline plot toggles depending on which plots are desired for this run.
PLOT_DATA      = True   # catchment_base population/employment maps (Phase 2)
PLOT_INFRA     = True    # infrabuild network plots (Phase 3A)
PLOT_SERVICES  = True   # services pipeline plots (Phase 3B)
PLOT_CAPACITY  = True   # capacity analysis plots (Phase 3C)
PLOT_CATCHMENT = True   # catchment allocation plots (Phase 4A)
PLOT_STATION_OD = True   # station OD pie map + corridor Sankeys (Phase 4B)
PLOT_ASSIGNMENT = True   # rail assignment heatmaps + Sankeys + service loads (Phase 4C)
PLOT_RESULTS   = False   # final CBA/result visualisations

plot_passenger_flow = False
plot_railway_line_load = False

# ═══════════════════════════════════════════════════════════════════════════════
# A2. CACHE  (load pre-computed outputs instead of recomputing — all phases)
# ═══════════════════════════════════════════════════════════════════════════════

# Set to True to load pre-computed outputs from disk instead of recomputing
use_cache_network = False
use_cache_pt_catchment = False
use_cache_developments = False
use_cache_catchmentOD = False
use_cache_stationsOD = False
use_cache_railRouting = False        # Phase 4C — skip writing routing CSVs that already exist
use_cache_traveltime_graph = False
use_cache_scenarios = False
use_cache_tts_calc = False

# ═══════════════════════════════════════════════════════════════════════════════
# A3. PHYSICAL ATTRIBUTES  (physics & design constants)
# ═══════════════════════════════════════════════════════════════════════════════
# Centralised physical and engineering constants. Each constant lists its valid range and the standard / calibration source. Downstream modules import from here.

MAX_TRAIN_LENGTH_M = 400              # universal train-length cap [m]. Range 100–500 (regional 80–200, long-distance ≤400)
SERVICE_BRAKE_DECEL_MS2 = 0.7         # service-brake deceleration [m/s²]. Range 0.5–1.3 (UIC 544-1 / ERTMS); calibrated to 0.7 vs GTFS
TT_OPERATIONAL_BUFFER = 1.30          # operational buffer on physics TTs. Range 1.0–1.5; calibrated to 1.30 vs GTFS jct-jct
MAX_SIDING_LENGTH_RATIO = 0.75        # passing-siding full-duplication threshold. Range 0.5–1.0. If L_siding ≥ ratio·L_section → duplicate

# ═══════════════════════════════════════════════════════════════════════════════
# A4. LEGACY  (read only by the older mains — main.py / main_cap.py; ignored by main_new)
# ═══════════════════════════════════════════════════════════════════════════════

# 'canton_ZH'              — station OD derived from commune-level cantonal survey data
# 'pt_catchment_perimeter' — OD derived from PT catchment area
OD_TYPE = 'canton_ZH'

# When True, OD demand is filtered to only include trips from/to the study area perimeter
only_demand_from_to_perimeter = True
