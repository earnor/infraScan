"""
infraScanRail — Pipeline Settings
Last modified: 2026-06-09

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
# Defines the study-area and catchment-area boundaries from coordinates or admin units.

# --- Study area ---
# STUDY_AREA_METHOD — how the study-area boundary is defined.
# 'coordinates' — use the perimeter_infra_generation polygon below
# 'admin'       — dissolve SwissBoundaries admin units (fill STUDY_AREA_ADMIN_* below)
STUDY_AREA_METHOD = 'admin'

# Polygon used when STUDY_AREA_METHOD = 'coordinates'  (EPSG:2056)
perimeter_infra_generation = Polygon([ # northern Bern
    (2597018.962, 1210475.783), # Upper left
    (2607054.685, 1210487.159), # upper right
    (2607052.842, 1198603.910), # lower right
    (2597077.391, 1198516.994), #lower left
])
#perimeter_infra_generation = Polygon([
#    (2700989.862, 1235663.403), # Dübendorf - Hinwil
#    (2708491.515, 1239608.529),
#    (2694972.602, 1255514.900),
#    (2687415.817, 1251056.404),
#])

# Used when STUDY_AREA_METHOD = 'admin'. Level: 'national' | 'cantonal' | 'bezirke' | 'municipal'
STUDY_AREA_ADMIN_LEVEL = 'municipal'
STUDY_AREA_ADMIN_NAMES = [
    'Dübendorf', 'Fällanden', 'Fehraltorf', 'Gossau (ZH)', 'Greifensee', 'Grüningen',
    'Hinwil', 'Illnau-Effretikon', 'Mönchaltorf', 'Pfäffikon', 'Schwerzenbach', 'Seegräben',
    'Uster',  'Volketswil', 'Wangen-Brüttisellen', 'Wetzikon (ZH)',
]
STUDY_AREA_ADMIN_SUBDIVISIONS = {'bezirke': [], 'municipal': []}  # optional extra units within the chosen admin level
STUDY_AREA_BUFFER_M = 3000                    # margin [m] for feeder-network edge handling

# --- Catchment area ---
# Boundary is always admin-based and must fully contain the study area. Level: 'national' | 'cantonal' | 'bezirke' | 'municipal'
CATCHMENT_AREA_ADMIN_LEVEL = 'cantonal'
CATCHMENT_AREA_ADMIN_NAMES = ['Bern']#['Zürich']       # list of admin entity names to dissolve
CATCHMENT_AREA_ADMIN_SUBDIVISIONS = {'bezirke': [], 'municipal': []}
CATCHMENT_AREA_BUFFER_M = 5000                # buffer (m) around boundary for GTFS spatial filter
CATCHMENT_CANTON_ABBREV = 'BE'#'ZH'               # canton abbreviation — used in folder and file naming

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — DATA PREPARATION  (raw filter inputs for infra + services)
# ═══════════════════════════════════════════════════════════════════════════════

# Named output folders for the two filter scripts (Phase 2 skip-guards)
INFRA_RAW_VERSION   = 'Raw_BE'             # infrabuild_filter_network output: data/Infrastructure/<INFRA_RAW_VERSION>/

GTFS_RAW_VERSION    = 'GTFS_SVC2026_CH_raw'    # services_filter_gtfs input under data/Network/GTFS_Timetable/
GTFS_FILTER_VERSION = 'GTFS_SVC2026_ZH'        # services_filter_gtfs output: data/Network/GTFS_Timetable/<GTFS_FILTER_VERSION>/

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3A — INFRASTRUCTURE NETWORK BUILD
# ═══════════════════════════════════════════════════════════════════════════════
# Builds a new BAV infrastructure network version or loads an existing one.

# INFRA_VERSION — infrastructure network to use.
# 'Build_New'           — run the full infrabuild pipeline to create a new named version
# 'AS_2026_ZH'          — BAV network as-is for 2026 (must already exist on disk)
# 'AS_2026_ZH_enhanced' — AS_2026_ZH enriched with projected svc travel times and corrections
# 'AS_2035_ZH'          — BAV network as-is for 2035 (must already exist on disk)
# 'AS_2035_ZH_enhanced' — AS_2035_ZH enriched with projected svc travel times and corrections
# 'AS_2026_BE'          — BAV network as-is for 2026 (must already exist on disk)
INFRA_VERSION = 'AS_2026_BE'

INFRA_BUILD_NEW_NAME = 'AS_2026_BE'        # name for the new version — used only when INFRA_VERSION = 'Build_New'

# When INFRA_VERSION = 'Build_New', open the interactive infra version manager to edit nodes/segments. False auto-creates the version from Base and saves without prompts.
OPEN_INFRA_VERSION_MANAGER = False
# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3B — SERVICES NETWORK BUILD
# ═══════════════════════════════════════════════════════════════════════════════
# Builds a new scheduled-services version (lines/segments) or loads an existing one.

# SVC_VERSION — services network to use.
GTFS_RAW_VERSION    = 'GTFS_SVC2026_CH_raw'    # services_filter_gtfs input under data/Network/GTFS_Timetable/
GTFS_FILTER_VERSION = 'GTFS_SVC2026_ZH'        # services_filter_gtfs output: data/Network/GTFS_Timetable/<GTFS_FILTER_VERSION>/

# 'Build_New'   — run full services pipeline to create a new named version
# 'AK_2026'     — scheduled services as of 2026 timetable
# 'AK_2026_S18' — AK_2026 with S18 line included
# 'AK_2035'     — scheduled services as of 2035 timetable
# 'AK_2035_S18' — AK_2035 with S18 line included
SVC_VERSION = 'Build_New'# 'AK_2026_S18'

SVC_BUILD_NEW_NAME = 'BE_AK_2026'                 # name for the new version — used only when SVC_VERSION = 'Build_New'

# When SVC_VERSION = 'Build_New', open the interactive services version manager to edit lines/services. False finalises the freshly built network and saves without prompts.
OPEN_SVC_VERSION_MANAGER = False
# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3C — CAPACITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
# Evaluates section capacity on the infrastructure network and flags constrained sections.

# CAPACITY_MODE — capacity method.
# 'None'      — skip capacity phases entirely
# 'Set_Value' — apply a fixed trains/hour/direction threshold to all sections
# 'Dynamic'   — full iterative capacity calculator workflow (capacity_workflow_wrapper.py)
CAPACITY_MODE = 'Dynamic'

# CAPACITY_SCOPE — spatial scope (used by main_new Phase 3C and capacity_workflow_wrapper.py).
# 'SA' — Study Area only: infra/services filtered to nodes within the study area
# 'CA' — Catchment Area: infra/services filtered to nodes within the catchment area boundary
CAPACITY_SCOPE = 'CA'

# Capacity method per spatial scope (when CAPACITY_SCOPE = 'SA' or 'CA'). SA must be the same or
# more precise than CA. Valid combinations: both Dynamic | SA Dynamic + CA Set_Value | both Set_Value.
CAPACITY_MODE_SA = 'Dynamic'    # method applied to Study Area sections
CAPACITY_MODE_CA = 'Dynamic'    # method applied to Catchment Area sections outside the SA

# CAPACITY_GROUPING_STRATEGY — how the workflow resolves capacity-grouping decisions.
# 'manual'       — prompt for each decision
# 'conservative' — always choose the lowest capacity option
# 'baseline'     — always choose the middle option
# 'optimal'      — always choose the highest capacity option
CAPACITY_GROUPING_STRATEGY = 'conservative'

CAPACITY_SET_VALUE = 6             # trains/hour/direction — used when CAPACITY_MODE = 'Set_Value'
# capacity_threshold — reactive CAP trigger margin: a section is constrained when available = Capacity − total_tphpd < this. User's choice [tphpd, 0–2 reasonable]:
# 0.0 — strictly-over only (sections at exactly 0 headroom never fire)
# 1.0 — one-train recovery buffer: sections AT capacity fire (default)
# 2.0 — conservative two-path margin
capacity_threshold = 0.0           # also the legacy Dynamic-workflow threshold and the 3C constrained-sections diagnostic margin
max_enhancement_iterations = 10    # max Phase 4 enhancement iterations — Dynamic only

# Internal — set dynamically in main (do not edit)
baseline_network_for_developments = None

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4A — CATCHMENT ALLOCATION
# ═══════════════════════════════════════════════════════════════════════════════
# Assigns each commune's population, employment and OD demand to rail stations.

# CATCHMENT_METHOD — how communes are allocated to stations (OD method derived automatically).
# 'Municipal' — each commune assigned wholly to one station (centroid-based); OD via commune→station lookup
# 'PT_Feeder' — access-time raster decides cell→station; communal OD re-weighted by PT-feeder catchment shares
CATCHMENT_METHOD = 'PT_Feeder'

# TRAVEL_COST_METHOD — how access-time components combine into a generalised cost.
# 'calibrated' — literature weights from cost_parameters.py (W_IVT/W_WAIT/W_WALK/W_BIKE/W_TRANSFER + comfort-weighted transfer penalty)
# 'absolute'   — all weights 1.0 at runtime (raw minutes, no weighting)
TRAVEL_COST_METHOD = 'calibrated'

# TRANSFER_COST_MODEL — transfer penalty (full literature value needs TRAVEL_COST_METHOD = 'calibrated'; 'absolute' uses raw minutes).
# 'fixed_value' — flat 12.1 min eq. IVT penalty (Axhausen 2014)
# 'explicit'    — W_TRANSFER × (transfer walk time + wait time from connecting headway)
TRANSFER_COST_MODEL = 'explicit'

# OD_ATTRIBUTION_MODE — how communal OD splits across a commune's stations (PT-Feeder branch only; Municipal is 1:1).
# 'specific' — origins by population share, destinations by FTE share (production/attraction split)
# 'blended'  — both ends by a count-based trip-end blend (α·pop_in_station + β·empl_in_station, normalised by commune activity; symmetric)
OD_ATTRIBUTION_MODE = 'blended'

# Trip-end rates for the PT-Feeder 'blended' mode (ignored otherwise): weight(commune→station) ∝ α·pop_in_station + β·empl_in_station,
# normalised by commune activity (α·pop_total + β·empl_total) so the no-PT share is dropped (Σ weight ≤ 1). Decoupled from OD_SCALING_* below.
OD_BLEND_POP_RATE  = 1.0   # α — trip-ends per resident per day; α=β=1 = one trip-end per resident/job (symmetric whole-day balance, no independent calibration available)
OD_BLEND_EMPL_RATE = 1.0   # β — trip-ends per job per day

# DEPRECATED — main pipeline always runs on the full_day union (single source of truth); no longer affects OD, routing,
# Güteklassen or the gateway split. Retained only as the default for the standalone catchment_allocate.py temporal-diagnostic CLI. Leave at 'full_day'.
TEMPORAL = 'full_day'

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4B — STATION OD MATRIX
# ═══════════════════════════════════════════════════════════════════════════════
# Projects the station-to-station OD matrix to the scenario horizon.

# Population/employment GROWTH weights for per-commune OD growth factors (get_commune_growth_factors). Temporal projection only — not the spatial OD_BLEND_* attribution.
OD_SCALING_POP_WEIGHT  = 1.0
OD_SCALING_EMPL_WEIGHT = 0.0   # 0: empl is derived from pop, so weighting it changes nothing. Raise only with an independent empl projection (e.g. BFS STATENT >2023).

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4C — NETWORK ASSIGNMENT  (passenger routing)
# ═══════════════════════════════════════════════════════════════════════════════
# Assigns passenger demand to routes across the network (single method per run).

# ROUTING_ASSIGNMENT_METHOD — assignment method.
# 'shortest_path' — deterministic all-or-nothing on generalised cost (one path/pair)
# 'logit'         — k-shortest-paths Logit route choice (demand split across paths)
ROUTING_ASSIGNMENT_METHOD = 'logit'

# Logit choice-set bounds (ignored by 'shortest_path').
ROUTING_K_PATHS         = 5       # max accepted paths per OD pair (Yen k-shortest)
ROUTING_COST_WINDOW_MIN = 15.0    # accept paths within best_gc + this many minutes ...
ROUTING_COST_WINDOW_PCT = 0.5     # ... or best_gc × (1 + this), whichever is larger
ROUTING_MAX_TRANSFERS   = 3       # transfer cap for accepted paths
ROUTING_MAX_EXAMINE     = 200     # hard guard on Yen candidates inspected per pair

# ROUTING_LOGIT_ENGINE — Logit candidate-generation engine.
# 'table' — connection-table itinerary proposal (0/1/2-transfer + shortest path; fast)
# 'yen'   — k-shortest baseline (exact; slow)
ROUTING_LOGIT_ENGINE    = 'table'

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5A — INFRASTRUCTURE INTERVENTIONS  (CC registry + master network)
# ═══════════════════════════════════════════════════════════════════════════════
# Reusable, versioned physical changes stored in a base-agnostic registry and composed on demand (infra_ints_* module).

# INFRA_INT_MODE — infra-int type(s) generated this session.
# 'NONE' — baseline, no infra interventions
# 'CC'   — connecting curves: new links enabling new through-routings (auto-discovered; an NDC pulls in its CC via the 'requires_infra' column of its svc-int xlsx)
# A list selects a subset, e.g. ['CC'] (codes case/order-insensitive); [] = NONE.
INFRA_INT_MODE = 'CC'

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5B — SERVICE INTERVENTIONS  (svc-int catalogue: EXT extended lines, NDC)
# ═══════════════════════════════════════════════════════════════════════════════
# Service interventions are the CBA units: declarative op lists (extend/truncate/reroute/set_frequency, or new_line for an NDC),
# keyed by the in-run (route_id, direction_id, variant_rank) with freq = total_dep, materialised as a per-svc-int rail_lines/rail_segments delta (svc_ints_* module).

# SVC_INT_MODE — svc-int type(s) generated this session.
# 'NONE' — baseline, no service interventions
# 'ALL'  — extended lines + new direct connections + frequency changes + stopping-pattern changes
# 'EXT'  — extended lines only: route extensions over existing infra (no new infra)
# 'NDC'  — new direct connections only: through-services that may require a CC (Phase 5A), pulled in via 'requires_infra'
# 'FRQ'  — frequency changes only: corridor homogenisation (all-stop extensions) + whole-line frequency doubling
# 'STP'  — stopping-pattern changes only: make passing services stop (homogenisation) + drop intermediate calls (express)
# A list selects a subset, e.g. ['EXT', 'NDC'] (codes case/order-insensitive); [] = NONE.
SVC_INT_MODE = 'ALL'

# Per-type ID start blocks; svc-ints numbered sequentially from these (e.g. ext_100001, ndc_103001). Mirrors the old main_cap convention.
DEV_ID_START_EXT = 100000              # extended-line interventions
DEV_ID_START_NDC = 103000              # new-direct-connection interventions
DEV_ID_START_FRQ = 104000              # frequency-change interventions
DEV_ID_START_STP = 105000              # stopping-pattern-change interventions

# NDC through-services run at a standardised frequency; total_dep = dep/h × 14 over the whole-day window (GK_WINDOW_MIN = 840 min).
NDC_FREQ_DEP_PER_H = 2                 # standardised NDC frequency [departures/hour]

# NDCs built per CC: from each of its two branch stations, enumerate up to this many termini outward and combine (legacy combinatorial set).
NDC_MAX_TERMINI_PER_END = 2

# Refill on rejection: an enumerated terminus whose every combination is rejected (backtrack/overlap/scope) is replaced by the
# next-ranked end station, admitted only while the arm stays within this length [m; 15 km admits Dietlikon-Winterthur via the
# Brütten tunnel (12.1 km), blocks Zürich Hardbrücke (24.0 km)]. The nearest-cap termini themselves stay distance-uncapped.
NDC_ARM_REFILL_MAX_M = 15000

# EXT discovery (replaces the legacy hardcoded radius/count): only termini inside the study area are extended; targets must be within the SA buffer and
# within EXT_BUFFER_RADIUS_M of the terminus, and are rejected if their infra route from the terminus passes through an already-served station.
EXT_BUFFER_RADIUS_M = 10000           # endpoint-buffer search radius [m]
EXT_MAX_CANDIDATES  = 5               # candidate stations kept per line terminus

# Min whole-day frequency to extend a route: Σtotal_dep over all dir-0 variants / (GK_WINDOW_MIN/60) ≥ this. Per-route gate; the extension applies to every variant.
EXT_MIN_FREQ_DEP_PER_H = 2            # min directional whole-day frequency to extend [dep/h]

# Geometric gates on the routed extension path (pruned from the nearest-EXT_MAX_CANDIDATES slate, no refill): mid-route reversals
# (angle < CC_BACKTRACK_ANGLE_DEG at a passed node) reject the candidate; a reversal at the old terminus is allowed but penalised.
EXT_MAX_DETOUR_FACTOR = 3.0           # max routed-path length / beeline distance to the target [-]
EXT_TERMINUS_REVERSAL_PENALTY_MIN = 2.0  # IVWT added to the extension hop when the service reverses at its old terminus [min]

# FRQ corridor homogenisation: a constant-frequency run (≥2 stations, all inside the SA) fires when a neighbouring run is
# busier than ratio × its own whole-day dep/h; remedy = all-stop extension of a service terminating at a corridor end (no reversals).
FRQ_CORRIDOR_NEIGHBOUR_RATIO = 1.0    # corridor fires when max neighbouring-run freq > ratio × corridor freq [-, 1.0 = strictly lower]
FRQ_DOUBLE_MAX_FREQ_PER_H = 4.0       # doubling generated only when the DOUBLED whole-day freq ≤ this [dep/h; ladder 1→2→4]

# STP stopping-pattern changes: (a) make passing services stop at SA stations another service already serves (homogenisation);
# (b) drop intermediate calls that have a co-stopper. A call is protected from dropping when it is a terminus of any service
# (transfer protection) or sole-served. Detection is structural (co-stopper / terminus / sole-server reads) — no numeric gate.
STP_PROTECT_CROSS_SERVICE_TERMINI = True  # mode (b): never drop a call at a station that is a terminus of any service [policy]
# Stop run-time penalty (decel + accel, excluding the IVWT dwell) used by the STP applier when no real GTFS stopping/passing
# time can be borrowed from a co-stopper/co-passer: an added stop adds this to the split hop, a dropped stop subtracts it from
# the merged hop. Real GTFS times are preferred where they exist on the corridor (decision F1).
STP_STOP_RUNTIME_PENALTY_MIN = 1.0    # modelled stopping-vs-passing run-time delta per intermediate stop [min]

# Svc-int-added hops (NDC lines, EXT extension hops) carry a default station dwell as IVWT, matching the base GTFS convention
# (dwell at the hop's from-stop; a direction's first hop = 0) so generated lines hold no GC advantage over base services.
SVC_INT_DEFAULT_IVWT_MIN = 0.5        # default dwell at the from-stop of svc-int-added hops [min, base network mean 0.6]

# ═══════════════════════════════════════════════════════════════════════════════
# 5C. CAPACITY ON THE MATCHED NETWORK  (svc-int capacity interventions)
# ═══════════════════════════════════════════════════════════════════════════════
# CAP (capacity) interventions — station/segment/siding-track — the mixed-int family, auto-generated by the capacity workflow on the matched
# (base+CC+merged-services) network (mixed_ints_* module). Registry keyed by '<infra>__<svc>', written under data/Developments/<infra>__<svc>/cap/; IDs cap_et_0001 / cap_ps_0001 …

# SVC_INT_CAP_THRESHOLD_TPHPD — the reactive 5C CAP trigger margin (= capacity_threshold, user-decidable, see Phase 3C
# block). Attribution is usage-based and full: a CAP fires exactly where the svc-int adds load to a constrained
# section/terminus and the int carries the whole cost (pre-existing or not, like its CC); ints not using the section
# never see it. The do-nothing baseline columns are diagnostic. Phase 5C is the sole, cost-bearing CAP generator.
SVC_INT_CAP_THRESHOLD_TPHPD = capacity_threshold

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — INTERVENTION RECOMPUTE  (per-svc-int catchment / OD / routing / flows)
# ═══════════════════════════════════════════════════════════════════════════════
# Recomputes catchment (6A) / OD (6B) / routing (6C) only where each svc-int changes them and unrolls the flows onto the infrastructure (6D), per svc-int under <svc_int_id>_network paths; 6A/6B run for PT_Feeder only, 6C/6D always.

# use_full_recompute_ints — selective vs full recompute per svc-int (correctness oracle).
# False — exact selective recompute of the affected cells/communes/pairs only (default)
# True  — full Phase-4 recompute per svc-int + parity report against the selective outputs
use_full_recompute_ints = False

# WRITE_INT_WORKBOOKS — per-svc-int 6C workbooks (path_assignment/segment_loads/station_events.xlsx).
# False — skip the three large workbooks for 6C runs (machine consumers read the parquet primitives)
# True  — write them per svc-int (human-facing; slow at ~1-3M rows each)
WRITE_INT_WORKBOOKS = False

PHASE_PARALLEL_N_JOBS = 3   # shared parallel-worker count for Phases 6 / 8A (loky; ~1-2.5 GB each for Phase 6); 1 = serial; Phase 6 forced serial when use_full_recompute_ints. 8B stays serial (~50 ms/int — parallel overhead loses)

# ═══════════════════════════════════════════════════════════════════════════════
# 7. SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════
# Builds the demand-growth factor store (baseline per-station vectors + per-svc-int overrides + modal/distance scalars) and the on-demand scenario-OD composer for Phase 8A.

# scenario_type — scenario generation method.
# 'GENERATED' — Monte Carlo random scenarios from population growth models
# 'STATIC_9'  — fixed set of 9 canonical scenarios (deferred, not implemented in main_new — Phase 7 raises)
# 'dummy'     — minimal placeholder scenarios for testing (deferred, not implemented in main_new — Phase 7 raises)
scenario_type = 'GENERATED'

amount_of_scenarios = 100
# Base year for grids/OD/demand scaling (snapshot year) and origin of the scenario demand trajectory (formerly POPULATION_BASE_YEAR).
start_year_scenario = 2018
end_year_scenario = 2100
start_valuation_year = 2050

# ═══════════════════════════════════════════════════════════════════════════════
# 8. VALUATION INPUTS
# ═══════════════════════════════════════════════════════════════════════════════
# Produces the per-svc-int CBA inputs: monetised travel-time savings per scenario x year (8A, rule of half on the 6C gc skims) and construction/maintenance/operating costs (8B); valuation years = start_valuation_year..end_year_scenario, monetary parameters from cost_parameters.py.

# ═══════════════════════════════════════════════════════════════════════════════
# 9. CBA RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
# Produces the discounted CBA per svc-int: costs-and-benefits (scenario x year), aggregated totals, summary and the core result plots; discount rate from cost_parameters.py, PV base year = start_valuation_year (factor 1.0).

RESULTS_TOP_N = [5, 10]   # cross-type 'best interventions' overviews: rank ALL svc-ints by mean net benefit, plot the top-N for each N.

# FRQ/STP corridor grouping: each affected line joins the corridor it shares the most
# study-area spine-stops with; ties default to the FIRST listed corridor. Values = the
# corridor's distinctive BAV node Numbers (Wetzikon, the shared terminus, is omitted so it
# never discriminates). Keyed by node number (encoding-safe); names are the inline comment.
RESULTS_CORRIDOR_SPINES = {
    'Dübendorf–Wetzikon':  [8503128, 8503127, 8503126, 8503125, 8503124],  # Dübendorf, Schwerzenbach ZH, Nänikon-Greifensee, Uster, Aathal
    'Effretikon–Wetzikon': [8503305, 8503303, 8503302, 8503301],           # Effretikon, Illnau, Fehraltorf, Pfäffikon ZH
}

# ═══════════════════════════════════════════════════════════════════════════════
# V. VALIDATION TRACK  (standalone validation_* CLIs — not a main_new phase)
# ═══════════════════════════════════════════════════════════════════════════════
# Compares model outputs against observed SBB/NPVM data (V1 stations, V2 link flows) and tests GC-settings sensitivity (V3); outputs under validation/.

# Reference baseline of the V-track (binding 2026-06-12) — independent of the pipeline's INFRA_VERSION/SVC_VERSION above.
VALIDATION_BASELINE_INFRA = 'AS_2026_ZH_enhanced'
VALIDATION_BASELINE_SVC   = 'AK_2026_S18'

VALIDATION_SBB_YEAR  = 2024    # V1 observed reference year (latest SBB counts; AK_2026-service-vs-2024-counts mismatch much smaller than the former 2018 anchor, noted not corrected) [switched 2026-06-22]
VALIDATION_NPVM_YEAR = 2017    # V2 observed reference year (NPVM 2017 base state, ARE release Stand 30.06.2023; model-assigned loads, not counts)
V1_EXCLUDE_CLIPPED   = True    # exclude observed DWV==49 stations from fit metrics (SBB clips volumes <50 to a fixed 49); kept in tables either way

VALIDATION_CATCHMENT_ONLY     = True   # restrict all validation (V1 stations, V2 + single-service segments) to elements fully within the catchment boundary; boundary-crossing/gateway elements structurally undershoot [decided 2026-06-22]
VALIDATION_CATCHMENT_BUFFER_M = 50.0   # tolerance added to the catchment boundary before the within-test [m]

V2_MATCH_BUFFER_M     = 30.0   # belastung-link <-> model-segment matching buffer [m; geometries verified to align within ~1 m]
V2_MATCH_MIN_COVERAGE = 0.5    # min share of model-segment length covered to accept a belastung feature [-]

VALIDATION_SQV_F_DAILY = 10000.0   # SQV scaling factor f for daily volumes [SQV = 1/(1+sqrt((m-o)^2/(f*o))); f=10000 = daily-volume convention]

# V3 GC-settings combos — '<TRAVEL_COST_METHOD>__<TRANSFER_COST_MODEL>' archive labels (calibrated__explicit = current defaults).
V3_COMBOS = ['calibrated__explicit', 'calibrated__fixed',
             'absolute__explicit', 'absolute__fixed']
# V3 svc-int CBA subset — spans types + BCR range (strong/clear/marginal positive, negative); + 1 FRQ appended once the regenerated 2026 catalogue exists [decided 2026-06-12].
V3_SVC_INT_SUBSET = ['ext_100001', 'ext_100003', 'ext_100010',
                     'ndc_103007', 'ndc_103009']

PLOT_VALIDATION = True               # standalone validation track only — V-track scatters, rank/share charts, corridor profiles, tau check (V1-V3)

# ═══════════════════════════════════════════════════════════════════════════════
# A1. PLOTS  (per-phase visualisation toggles — all phases)
# ═══════════════════════════════════════════════════════════════════════════════
# One toggle per phase; data/CSV outputs are always written regardless of these.

PLOT_DATA      = True   # Phase 2 — catchment_base population/employment maps
PLOT_INFRA     = True    # Phase 3A — infrabuild network plots
PLOT_SERVICES  = True    # Phase 3B — services pipeline plots
PLOT_CAPACITY  = True    # Phase 3C — capacity analysis plots
PLOT_CATCHMENT = True   # Phase 4A — catchment allocation plots
PLOT_STATION_OD = True   # Phase 4B — station OD pie map + corridor Sankeys
PLOT_ASSIGNMENT = True   # Phase 4C — rail assignment heatmaps + Sankeys + service loads
PLOT_INFRA_INTS = True   # Phase 5A — infra-int master tagged network .qgz
PLOT_SVC_INTS  = True    # Phase 5B — svc-int delta + per-type/all-produced overlays
PLOT_MIXED_INTS = True   # Phase 5C — per-svc-int capacity/service maps + CAP diff
PLOT_INT_RECOMPUTE = True  # Phase 6A-6C — per-svc-int catchment/OD/routing delta plots
PLOT_FLOWS     = True    # Phase 6D — infra-level passenger-flow maps + base-vs-dev diff
PLOT_SCENARIOS = True    # Phase 7 — population/modal/distance scenario fans
PLOT_RESULTS   = True   # Phase 9 — core CBA result set: savings/net-benefit/BCR charts, waterfalls, network maps

# ═══════════════════════════════════════════════════════════════════════════════
# A2. CACHE  (load pre-computed outputs instead of recomputing — all phases)
# ═══════════════════════════════════════════════════════════════════════════════
# Set True to load pre-computed outputs from disk instead of recomputing.

use_cache_pt_catchment = False        # Phase 4A — skip catchment allocation when outputs exist & manifest fresh
use_cache_stationsOD = False          # Phase 4B — skip station-OD preparation when outputs exist & manifest fresh
use_cache_railRouting = False         # Phase 4C — skip writing routing CSVs that already exist
use_cache_infra_ints = False          # Phase 5A — keep existing cc registry, skip re-discovery
use_cache_svc_ints = False            # Phase 5B — keep svc-int catalogue + materialised deltas
use_cache_svc_int_cap = False        # Phase 5C — keep svc-int CAP + merged-services workbooks
use_cache_int_recompute = False      # Phase 6A-6C — skip svc-ints whose recompute outputs exist
use_cache_flows = False              # Phase 6D — keep existing flow tables + maps
use_cache_scenarios = False           # Phase 7 — skip factor-store builds whose outputs exist
use_cache_tts = False                 # Phase 8A — load per-svc-int tts.parquet caches when present
use_cache_costs = False               # Phase 8B — skip when construction_cost.csv covers all svc-ints

# ═══════════════════════════════════════════════════════════════════════════════
# A3. PHYSICAL ATTRIBUTES  (physics & design constants)
# ═══════════════════════════════════════════════════════════════════════════════
# Centralised physical/engineering constants; each lists its valid range and standard/calibration source. Downstream modules import from here.

MAX_TRAIN_LENGTH_M = 400              # universal train-length cap [m]. Range 100–500 (regional 80–200, long-distance ≤400)
SERVICE_BRAKE_DECEL_MS2 = 0.7         # service-brake deceleration [m/s²]. Range 0.5–1.3 (UIC 544-1 / ERTMS); calibrated to 0.7 vs GTFS
TT_OPERATIONAL_BUFFER = 1.30          # operational buffer on physics TTs. Range 1.0–1.5; calibrated to 1.30 vs GTFS jct-jct
MAX_SIDING_LENGTH_RATIO = 0.25        # passing-siding full-duplication threshold: if L_siding ≥ ratio·L_section → duplicate

# Connecting-curve (CC) geometry constants (Phase 5A). Min radius R = 11.8·v²/(u + u_f); v=80, u=150, u_f=100 → R ≈ 302 m. Adjust per situation.
CC_DESIGN_SPEED_KMH     = 80          # design speed v [km/h]. Range 40–120
CC_CANT_MM              = 150         # track cant (superelevation) u [mm]. Range 0–160
CC_CANT_DEFICIENCY_MM   = 100         # cant deficiency u_f [mm]. Range 0–130 (moderate ≤100)
CC_BACKTRACK_ANGLE_DEG  = 120         # CC needed when the through-move's leg angle < this [deg]

# ═══════════════════════════════════════════════════════════════════════════════
# A4. LEGACY  (read only by the older mains — main.py / main_cap.py; ignored by main_new)
# ═══════════════════════════════════════════════════════════════════════════════

# OD_TYPE — source of the station OD matrix.
# 'canton_ZH'              — commune-level cantonal survey data
# 'pt_catchment_perimeter' — derived from the PT catchment area
OD_TYPE = 'canton_ZH'

only_demand_from_to_perimeter = True   # when True, OD demand is filtered to trips from/to the study-area perimeter

# Plot / cache toggles not driven by main_new (superseded by the A1/A2 per-phase toggles).
plot_passenger_flow = False          # main.py / main_cap.py only — passenger-flow maps
use_cache_network = True             # main.py / main_cap.py only — network build cache
use_cache_developments = False       # main.py / main_cap.py only — developments cache
use_cache_catchmentOD = True         # main.py / main_cap.py only — catchment-OD cache (superseded by 4A/4B)
use_cache_traveltime_graph = True    # main.py / main_cap.py only — travel-time graph cache
use_cache_tts_calc = True            # main.py / main_cap.py only — travel-time-savings calc cache
