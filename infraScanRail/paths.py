import os
from pathlib import Path

# Use relative path from the script location
MAIN = str(Path(__file__).parent.resolve())
RAIL_SERVICES_AK2035_PATH= r'data\temp\railway_services_ak2035.gpkg'
RAIL_SERVICES_AK2035_EXTENDED_PATH = r'data\temp\railway_services_ak2035_extended.gpkg'
RAIL_SERVICES_2024_PATH= r'data/temp/network_railway-services.gpkg'
RAIL_SERVICES_AK2024_EXTENDED_PATH = r'data/temp/network2024_railway_services_extended.gpkg'
NEW_LINKS_UPDATED_PATH = r"data\Network\processed\updated_new_links.gpkg"
NEW_RAILWAY_LINES_PATH = r"data\Network\processed\new_railway_lines.gpkg"
NETWORK_WITH_ALL_MODIFICATIONS = r"data\Network\processed\combined_network_with_all_modifications.gpkg"
# DEPRECATED (Phase 5A) — legacy per-dev_id gpkg directory. Superseded by the
# infra-int registry + master tagged network (data/Infrastructure/Developments/).
# Retained only until the last legacy reader is removed in Phase 5 wiring.
DEVELOPMENT_DIRECTORY = r"data\Network\processed\developments"

RAIL_NODES_PATH = r"data\Network\Rail_Node.csv"
RAIL_POINTS_PATH = r"data\Network\processed\points.gpkg"
#OD_KT_ZH_PATH = r'data/Spatial_Data/Transit_Network/Cantonal_OD_Demand/KTZH_00001982_00003903.xlsx'
#OD_KT_ZH_PATH = r'data/Traffic_Flow/OD/Original/KTZH_00001982_00003903.xlsx'
#OD_KT_BE_PATH = r'data/Traffic_Flow/OD/Bern/BE_2019_Ist2019_DWV.xlsx'
OD_KT_PATH = r'data/Traffic_Flow/OD/Bern/BE_2019_Ist2019_DWV.csv'
# SBB official station-users statistics (2013-2025): daily passenger totals and the
# hourly load distribution per major station (sheet 'Tag_Jour_Giorno_Day').
SBB_STATION_USERS_XLSX = r'data/Spatial_Data/Transit_Network/SBB_Station_Flows/b01x-sbb-cff-ffs-bhfbenutzer-usagersgares-utentistazione-stationusers_2013-2025.xlsx'
OD_STATIONS_KT_ZH_PATH      = r'data/Traffic_Flow/OD/Rail/ktzh/od_matrix_stations_ktzh_20.csv'   # legacy only (main.py / main_cap.py)
OD_STATIONS_KT_ZH_2040_PATH = r'data/Traffic_Flow/OD/Rail/ktzh/od_matrix_stations_ktzh_2040.csv' # legacy only (main.py / main_cap.py)

OD_KT_ZH_PATH = r'data/Traffic_Flow/OD/Original/KTZH_00001982_00003903.xlsx'
OD_STATIONS_KT_ZH_PATH      = r'data/Traffic_Flow/OD/Rail/ktzh/od_matrix_stations_ktzh_20.csv'
OD_STATIONS_KT_ZH_2040_PATH = r'data/Traffic_Flow/OD/Rail/ktzh/od_matrix_stations_ktzh_2040.csv'

# Versioned OD outputs (W3 station-pair matrices, W4b routing, gateways) live under
# data/Traffic_Flow/OD/<svc_network>/<Method|Gateway|Routing>/ — built via the
# get_station_od_* / get_gateway_dir / get_od_routing_* helpers below (mirrors the
# Catchment_Area/<svc_network>/<Method> layout). The legacy flat files under
# data/Traffic_Flow/OD/ and data/Traffic_Flow/OD/Rail/ are retained as-is.
TRAFFIC_FLOW_OD_DIR       = r'data/Traffic_Flow/OD'
TRAFFIC_FLOW_OD_PLOTS_DIR = os.path.join('plots', 'Traffic_Flow', 'OD')
TRAFFIC_FLOW_ASSIGNMENT_DIR = r'data/Traffic_Flow/Assignment'
TRAFFIC_FLOW_ASSIGNMENT_PLOTS_DIR = os.path.join('plots', 'Traffic_Flow', 'Assignment')
_OD_METHOD_DIRS = {'pt_feeder': 'PT_Feeder', 'municipal': 'Municipal'}

COMMUNE_TO_STATION_PATH = r"data\Network\processed\Communes_to_railway_stations_ZH.xlsx"  # legacy only (main.py / main_cap.py)
GRAPH_POS_PATH = r"data\Network\processed\graph_data.pkl"

# --- BAV Geopackages (Official Swiss Railway Infrastructure) ---
BAV_RAIL_NODES_GPKG = r"data/Spatial_Data/Railway_Infrastructure/Rail_Nodes.gpkg"
BAV_RAIL_SEGMENTS_GPKG = r"data/Spatial_Data/Railway_Infrastructure/Rail_Edges_Segments.gpkg"
BAV_RAIL_ROUTES_GPKG = r"data/Spatial_Data/Railway_Infrastructure/Rail_Edges_Routes.gpkg"
HALTESTELLEN_OEV_GPKG = r"data/Spatial_Data/Railway_Infrastructure/HaltestellenOeV.gpkg"

# --- TLMRegio Railway (Supplementary data for tunnels/bridges) ---
TLMREGIO_RAILWAY_SHP = r"data/Spatial_Data/Land_Use/Transportation/swissTLMRegio_Railway.shp"

# --- Network Infrastructure ---
# Root directory — all version subfolders live here
NETWORK_INFRASTRUCTURE_DIR  = r"data/Infrastructure"
# Raw/  : spatial-filtered BAV output (infrabuild_filter_network.py stage 1)
NETWORK_INFRASTRUCTURE_RAW  = r"data/Infrastructure/Raw"
# Seeds/ : per-INFRA_VERSION seed nodes/segments auto-applied during base build (Topic 3)
INFRASTRUCTURE_SEEDS_DIR    = r"data/Infrastructure/Seeds"
# Developments/ : all intervention outputs in one combined workspace per <infra>__<svc>
# (decision H, Phases 5A-5C). Registries nest under the combo (cc/ext/ndc/cap); the
# composed-network trees are shared — Derived/ and Dev_Full/ names are globally unique
# (deterministic on base + sorted int_ids), so no per-combo nesting is needed.
# Note: per-svc-int output trees nest as Developments/<combo>/<id>_network/ INSIDE each
# domain tree (Rail_Lines, Catchment_Area, OD, Assignment, Scenario + plots mirrors) —
# see svc_int_network_name. DEVELOPMENTS_DIR below is the absolute registry tree.
DEVELOPMENTS_DIR          = r"data/Developments"
DEVELOPMENTS_DERIVED_DIR  = r"data/Developments/Derived"     # base+int composed nets (shared)
DEVELOPMENTS_DEV_FULL_DIR = r"data/Developments/Dev_Full"    # master "all-ints" net (shared)
# Mirrored plots tree (plots/Developments/<combo>/<subtype>/)
DEVELOPMENTS_PLOTS_DIR    = r"plots/Developments"
# Subdir name nesting per-svc-int outputs inside each domain tree (combo-keyed layout)
DEVELOPMENTS_SUBDIR_NAME  = "Developments"


def _extract_seed_year(year_or_version: str) -> str:
    import re
    m = re.search(r'(20\d{2})', str(year_or_version))
    return m.group(1) if m else str(year_or_version)


def get_seed_nodes_gpkg(year_or_version: str) -> str:
    """Return absolute path to nodes_<year>.gpkg in the Seeds directory.

    Args:
        year_or_version: bare year ('2026') or full version name ('AS_2026_ZH').
    """
    return os.path.join(MAIN, INFRASTRUCTURE_SEEDS_DIR, f"nodes_{_extract_seed_year(year_or_version)}.gpkg")


def get_seed_segments_gpkg(year_or_version: str) -> str:
    """Return absolute path to segments_<year>.gpkg in the Seeds directory."""
    return os.path.join(MAIN, INFRASTRUCTURE_SEEDS_DIR, f"segments_{_extract_seed_year(year_or_version)}.gpkg")


def get_seed_composition_gpkg(year_or_version: str) -> str:
    """Return absolute path to segments_composition_<year>.gpkg in the Seeds directory."""
    return os.path.join(MAIN, INFRASTRUCTURE_SEEDS_DIR, f"segments_composition_{_extract_seed_year(year_or_version)}.gpkg")


def get_seed_qgz(year_or_version: str) -> str:
    """Return absolute path to seed_<year>.qgz in the Seeds directory."""
    return os.path.join(MAIN, INFRASTRUCTURE_SEEDS_DIR, f"seed_{_extract_seed_year(year_or_version)}.qgz")


def get_infra_int_registry(int_type: str, combo: str) -> str:
    """Return absolute path to the intervention registry gpkg for a type + combo.

    All registries partition on the '<infra>__<svc>' combo (decision H), so each lands
    under Developments/<combo>/<int_type>/. CC is regenerated per combo even though it is
    infra-only (accepted simplicity trade-off).

    Args:
        int_type: short code — 'cc' (connecting curves, infra ints) or 'cap' (capacity,
            mixed ints).
        combo: the '<infra>__<svc>' workspace key.
    """
    return os.path.join(MAIN, DEVELOPMENTS_DIR, combo, int_type, f"{int_type}_interventions.gpkg")


def get_svc_int_registry(int_type: str, combo: str) -> str:
    """Return absolute path to the svc-int registry xlsx for a type + combo.

    Args:
        int_type: short code — 'ext' (extended lines), 'ndc' (new direct connections),
            'frq' (frequency changes) or 'stp' (stopping-pattern changes). Stored under
            Developments/<combo>/<int_type>/.
        combo: the '<infra>__<svc>' workspace key.
    """
    return os.path.join(MAIN, DEVELOPMENTS_DIR, combo, int_type,
                        f"{int_type}_interventions.xlsx")


def get_cc_composition_cache(combo: str) -> str:
    """Return absolute path to the per-combo CC composition cache CSV."""
    return os.path.join(MAIN, DEVELOPMENTS_DIR, combo, 'cc', 'cc_composition.csv')


def get_cc_interventions_gpkg(combo: str) -> str:
    """Return absolute path to the per-combo CC interventions geopackage
    (nodes/segments/segments_composition layers for the connecting curves)."""
    return os.path.join(MAIN, DEVELOPMENTS_DIR, combo, 'cc', 'cc_interventions.gpkg')


def get_svc_int_catalogue_dir(combo: str) -> str:
    """Return the combo-root directory for svc-int catalogue / affected-set CSVs.

    Catalogues span both EXT and NDC, so they sit at the combo root
    (Developments/<combo>/) alongside the cc/ext/ndc/cap subfolders.
    """
    return os.path.join(MAIN, DEVELOPMENTS_DIR, combo)


def get_svc_int_cap_attribution_path(combo: str) -> str:
    """Return absolute path to the svc-int→CAP attribution CSV (mixed ints, 5C).

    Args:
        combo: the '<infra>__<svc>' workspace key this CAP set was generated against.
    """
    return os.path.join(MAIN, DEVELOPMENTS_DIR, combo, 'cap', 'svc_int_cap_attribution.csv')


def get_svc_int_cap_dir(combo: str) -> str:
    """Return the data-side cap workspace dir (Developments/<combo>/cap/).

    Data artifacts (registry, attribution, per-svc-int capacity workbooks) live here;
    the mirrored plots live under plots/Developments/<combo>/cap/ (get_developments_plot_dir).
    """
    return os.path.join(MAIN, DEVELOPMENTS_DIR, combo, 'cap')


def get_parity_csv(combo: str, svc_int_id: str) -> str:
    """Return absolute path to a svc-int's Phase-6 parity report CSV
    (Developments/<combo>/parity/parity_<svc_int_id>.csv) — the selective-vs-
    full-recompute comparison written when settings.use_full_recompute_ints."""
    return os.path.join(MAIN, DEVELOPMENTS_DIR, combo, 'parity',
                        f'parity_{svc_int_id}.csv')


def get_developments_plot_dir(combo: str, subtype: str = None) -> str:
    """Return a plots dir under the mirrored plots/Developments/<combo>/<subtype>/ tree.

    Args:
        combo: the '<infra>__<svc>' workspace key.
        subtype: optional leaf (e.g. 'cc', 'cap', 'ext', 'ndc', 'frq', 'stp').
    """
    parts = [MAIN, DEVELOPMENTS_PLOTS_DIR, combo]
    if subtype:
        parts.append(subtype)
    return os.path.join(*parts)

def svc_int_network_name(svc_int_id: str, combo: str) -> str:
    """Relative per-svc-int network dirname: Developments/<combo>/<id>_network.

    The single place the per-svc-int network name is constructed (decision
    2026-06-10): svc-int ids restart per run, so the name must carry the
    '<infra>__<svc>' combo to keep combinations from overwriting each other.
    The value nests inside each domain tree (Rail_Lines, Catchment_Area, OD,
    Assignment, Scenario + plots mirrors), which all plain-join it; baseline
    '<svc_version>_network' names stay flat.

    Args:
        svc_int_id: svc-int id (e.g. 'ext_100001', 'ndc_103001').
        combo:      the '<infra>__<svc>' workspace key.
    """
    return os.path.join(DEVELOPMENTS_SUBDIR_NAME, combo, svc_int_id + '_network')


def get_svc_int_network_dir(svc_int_id: str, combo: str) -> str:
    """Return absolute path to a per-svc-int delta network folder
    (data/Network/Rail_Lines/Developments/<combo>/<id>_network/).

    Mirrors the ``<svc_version>_network`` layout so the existing catchment/OD/routing
    readers consume a svc-int's materialised delta unchanged (Phase 5B apply_svc_int).

    Args:
        svc_int_id: svc-int id (e.g. 'ext_100001', 'ndc_103001').
        combo:      the '<infra>__<svc>' workspace key.
    """
    return os.path.join(MAIN, RAIL_LINES_DIR, svc_int_network_name(svc_int_id, combo))


def get_svc_int_projected_path(svc_int_id: str, infra_version: str, combo: str) -> str:
    """Return absolute path to a svc-int's projected rail segments delta
    (…/Developments/<combo>/<id>_network/<infra_version>/rail_segments.gpkg).

    The <infra_version> subfolder is named after the BASE infra version even when
    the delta was projected on a composed (base + CC) infra — see apply_svc_int.
    """
    return os.path.join(get_svc_int_network_dir(svc_int_id, combo),
                        infra_version, 'rail_segments.gpkg')


def get_infra_version_dir(version: str) -> str:
    """Return absolute path to the named infrastructure version directory."""
    return os.path.join(MAIN, NETWORK_INFRASTRUCTURE_DIR, version)

def get_derived_infra_version_dir(version: str) -> str:
    """Return absolute path to a derived (composed) infra version under the shared
    Developments/Derived/ tree. Version names are globally unique (deterministic on
    base + sorted int_ids), so no per-combo nesting is needed.

    Args:
        version: derived version name (e.g. 'AS_2026_ZH+cap_ps_0001').
    """
    return os.path.join(MAIN, DEVELOPMENTS_DERIVED_DIR, version)

def derived_version_exists(version: str) -> bool:
    """True if nodes.gpkg, segments.gpkg and segments_composition.gpkg all exist
    in the derived version directory."""
    d = get_derived_infra_version_dir(version)
    return all(
        os.path.isfile(os.path.join(d, f))
        for f in ('nodes.gpkg', 'segments.gpkg', 'segments_composition.gpkg')
    )


def resolve_infra_dir(version: str) -> str:
    """Absolute dir of a real OR derived (composed) infra version.

    Real versions win; falls back to Developments/Derived/<version>. Lets
    consumers (e.g. Phase 6D) take any infra version string without knowing
    whether it is a base or a compose_infra output.

    Raises:
        FileNotFoundError: when the version exists in neither tree.
    """
    if infra_version_exists(version):
        return get_infra_version_dir(version)
    if derived_version_exists(version):
        return get_derived_infra_version_dir(version)
    raise FileNotFoundError(
        f"Infra version '{version}' found neither at "
        f"{get_infra_version_dir(version)} nor at "
        f"{get_derived_infra_version_dir(version)}.")

def get_infra_raw_dir(version: str) -> str:
    """Return absolute path to the named infrastructure raw directory.

    Args:
        version: Raw folder name from settings.INFRA_RAW_VERSION, e.g. 'Raw_ZH'.
    """
    return os.path.join(MAIN, NETWORK_INFRASTRUCTURE_DIR, version)

def infra_version_exists(version: str) -> bool:
    """True if nodes.gpkg, segments.gpkg and segments_composition.gpkg all exist."""
    d = get_infra_version_dir(version)
    return all(
        os.path.isfile(os.path.join(d, f))
        for f in ('nodes.gpkg', 'segments.gpkg', 'segments_composition.gpkg')
    )

def get_projected_services_path(svc_version: str, infra_version: str) -> str:
    """Return absolute path to projected rail edges for a svc/infra version pair."""
    return os.path.join(MAIN, RAIL_LINES_DIR, svc_version + '_network', infra_version, 'rail_segments.gpkg')

def get_rail_stops_sa(svc_network: str, infra_version: str) -> str:
    """Return absolute path to the study-area rail-stops GPKG for a svc/infra pair
    (data/Network/Rail_Lines/<svc_network>/<infra_version>/rail_stops_sa.gpkg).

    This is the authoritative study-area station set: it is filtered from the full
    rail_stops layer, so it includes peak-only stations that the all-day rail-stops
    load omits.

    Args:
        svc_network:   service version folder name WITH the '_network' suffix.
        infra_version: infrastructure version subfolder, e.g. 'AS_2026_ZH_enhanced'.
    """
    return os.path.join(MAIN, RAIL_LINES_DIR, svc_network, infra_version, 'rail_stops_sa.gpkg')

def get_rail_stops(svc_network: str, infra_version: str) -> str:
    """Return absolute path to the full rail-stops GPKG for a svc/infra pair
    (data/Network/Rail_Lines/<svc_network>/<infra_version>/rail_stops.gpkg).

    All network stops with 'Number' (UIC) and geometry (EPSG:2056); the
    catchment-wide superset of rail_stops_sa.gpkg.

    Args:
        svc_network:   service version folder name WITH the '_network' suffix.
        infra_version: infrastructure version subfolder, e.g. 'AS_2026_ZH_enhanced'.
    """
    return os.path.join(MAIN, RAIL_LINES_DIR, svc_network, infra_version, 'rail_stops.gpkg')

def get_rail_lines(svc_network: str, infra_version: str) -> str:
    """Return absolute path to the rail_lines GPKG for a svc/infra pair
    (data/Network/Rail_Lines/<svc_network>/<infra_version>/rail_lines.gpkg).

    Carries one row per route variant with route_id and line_short_name (the
    S-Bahn/IR/IC line label), used to collapse GTFS route variants to lines.

    Args:
        svc_network:   service version folder name WITH the '_network' suffix.
        infra_version: infrastructure version subfolder, e.g. 'AS_2026_ZH_enhanced'.
    """
    return os.path.join(MAIN, RAIL_LINES_DIR, svc_network, infra_version, 'rail_lines.gpkg')

def get_od_version_dir(svc_network: str) -> str:
    """Return absolute path to the versioned OD output dir for a svc version.

    Args:
        svc_network: service version folder name WITH the '_network' suffix.
    """
    return os.path.join(MAIN, TRAFFIC_FLOW_OD_DIR, svc_network)


def get_station_od_dir(svc_network: str, method: str) -> str:
    """Return absolute path to the per-method station OD dir
    (data/Traffic_Flow/OD/<svc_network>/<PT_Feeder|Municipal>/)."""
    return os.path.join(get_od_version_dir(svc_network),
                        _OD_METHOD_DIRS.get(method, method))


def get_station_od_window_xlsx(svc_network: str, method: str, window: str) -> str:
    """Return absolute path to a per-window station-pair OD workbook for a
    (method, window). For PT-Feeder the workbook carries a 'Specific' and a
    'Blended' sheet; for Municipal a single 'Municipal' sheet.

    Args:
        method: 'pt_feeder' | 'municipal'.
        window: 'peak' | 'off_peak' | 'full_day'.
    """
    return os.path.join(get_station_od_dir(svc_network, method),
                        f'od_matrix_stations_{window}.xlsx')


def get_od_top_relations_xlsx(svc_network: str, method: str) -> str:
    """Return absolute path to the per-method top origins/destinations workbook."""
    return os.path.join(get_station_od_dir(svc_network, method),
                        'od_station_top_relations.xlsx')


def get_od_method_comparison_xlsx(svc_network: str) -> str:
    """Return absolute path to the cross-method (PT-Feeder vs Municipal) OD-flow
    comparison workbook for the study-area stations (version-level, not per-method)."""
    return os.path.join(get_od_version_dir(svc_network), 'od_method_comparison.xlsx')


def get_station_od_matrix_xlsx(svc_network: str, method: str) -> str:
    """Return absolute path to the per-method full station×station OD matrix
    workbook (one sheet per time window)."""
    return os.path.join(get_station_od_dir(svc_network, method),
                        'od_matrix_stations.xlsx')


def get_station_od_long_csv(svc_network: str, method: str, attribution: str) -> str:
    """Return absolute path to the persisted long-format station OD
    (origin_station_id, dest_station_id, trips) for a (method, attribution).

    Reloadable baseline for the Phase 6B subset reaggregation; attribution is the
    lower-cased sheet label ('specific' | 'blended' | 'municipal')."""
    return os.path.join(get_station_od_dir(svc_network, method),
                        f'od_long_{attribution}.csv')


def get_attribution_weights_csv(svc_network: str, method: str, attribution: str,
                                side: str) -> str:
    """Return absolute path to a persisted attribution weight table
    (BFS, station_id, <side>_weight) for a (method, attribution).

    side: 'orig' | 'dest'. Consumed (with the communal OD) by reaggregate_subset."""
    return os.path.join(get_station_od_dir(svc_network, method),
                        f'weights_{side}_{attribution}.csv')


def get_communal_od_csv(svc_network: str) -> str:
    """Return absolute path to the persisted gateway-expanded communal OD
    (quelle_code, ziel_code, wert) that reaggregate_subset consumes."""
    return os.path.join(get_od_version_dir(svc_network), 'communal_od_branch.csv')


def get_od_method_plot_dir(svc_network: str, method: str) -> str:
    """Return absolute path to the per-method OD plot dir
    (plots/Traffic_Flow/OD/<svc_network>/<PT_Feeder|Municipal>/).

    Mirrors get_station_od_dir on the plots side so the repo tree is symmetric.
    """
    return os.path.join(MAIN, TRAFFIC_FLOW_OD_PLOTS_DIR, svc_network,
                        _OD_METHOD_DIRS.get(method, method))


def get_od_sankey_dir(svc_network: str, method: str) -> str:
    """Return absolute path to the per-method corridor-Sankey plot dir
    (plots/Traffic_Flow/OD/<svc_network>/<PT_Feeder|Municipal>/Sankey/)."""
    return os.path.join(get_od_method_plot_dir(svc_network, method), 'Sankey')


def get_gateway_dir(svc_network: str) -> str:
    """Return absolute path to the gateway-assignment dir
    (data/Traffic_Flow/OD/<svc_network>/Gateway/)."""
    return os.path.join(get_od_version_dir(svc_network), 'Gateway')


SCENARIO_DIR = r"data/Scenario"


def get_scenario_version_dir(svc_network: str) -> str:
    """Return absolute path to the versioned scenario factor-store dir
    (data/Scenario/<svc_network>/). Baseline stores are keyed by the svc
    network, per-svc-int overrides by the combo-keyed svc-int network name
    (same layout as the OD/catchment per-version dirs).

    Args:
        svc_network: service version folder name WITH the '_network' suffix.
    """
    return os.path.join(MAIN, SCENARIO_DIR, svc_network)


def get_scenario_factor_dir(svc_network: str, method: str) -> str:
    """Return absolute path to the per-method factor dir
    (data/Scenario/<svc_network>/<PT_Feeder|Municipal>/)."""
    return os.path.join(get_scenario_version_dir(svc_network),
                        _OD_METHOD_DIRS.get(method, method))


def get_growth_factors_parquet(svc_network: str, method: str) -> str:
    """Return absolute path to the baseline per-station growth-factor vectors
    (scenario, year, station_id, factor) for a (svc_network, method).

    Phase 7 output; composed with the per-svc-int overrides by
    compose_scenario_od (Phase 8A demand input)."""
    return os.path.join(get_scenario_factor_dir(svc_network, method),
                        'growth_factors.parquet')


def get_modal_distance_factors_csv(svc_network: str) -> str:
    """Return absolute path to the station-independent modal-split and
    distance-per-person factor table (scenario, year, modal_factor,
    distance_factor). Method-independent, so version-level."""
    return os.path.join(get_scenario_version_dir(svc_network),
                        'modal_distance_factors.csv')


def get_growth_factor_overrides_csv(svc_int_network: str, method: str) -> str:
    """Return absolute path to a per-svc-int growth-factor override table
    (scenario, year, station_id, factor — affected stations only).

    Args:
        svc_int_network: '<svc_int_id>_network' folder name.
    """
    return os.path.join(get_scenario_factor_dir(svc_int_network, method),
                        'growth_factor_overrides.csv')


def get_station_commune_breakdown_csv(svc_network: str, method: str) -> str:
    """Return absolute path to the 4A/6A station-commune allocation breakdown
    (data/Catchment_Area/<svc_network>/<PT_Feeder|Municipal>/station_commune_breakdown.csv).

    Written by catchment_allocate (full run + reallocate_for_svc_int); Phase 7
    reads pop_in_station from it as station-growth weights."""
    return os.path.join(MAIN, CATCHMENT_AREA_DIR, svc_network,
                        _OD_METHOD_DIRS.get(method, method),
                        'station_commune_breakdown.csv')


def get_gateway_connections_xlsx(svc_network: str) -> str:
    """Return absolute path to the gateway service-connection table
    (Gateway/gateway_service_connections.xlsx) consumed by passenger routing."""
    return os.path.join(get_gateway_dir(svc_network),
                        'gateway_service_connections.xlsx')


def get_gateway_convergence_map_json(svc_network: str) -> str:
    """Return absolute path to the optional gateway convergence-map override
    (Gateway/gateway_convergence_map.json)."""
    return os.path.join(get_gateway_dir(svc_network),
                        'gateway_convergence_map.json')


def get_od_routing_dir(svc_network: str) -> str:
    """Return absolute path to the W4b rail-routing output dir
    (data/Traffic_Flow/OD/<svc_network>/Routing/)."""
    return os.path.join(get_od_version_dir(svc_network), 'Routing')


def get_assignment_dir(svc_network: str) -> str:
    """Return absolute path to the Phase-4C assignment output dir
    (data/Traffic_Flow/Assignment/<svc_network>/).

    Args:
        svc_network: service version folder name WITH the '_network' suffix.
    """
    return os.path.join(MAIN, TRAFFIC_FLOW_ASSIGNMENT_DIR, svc_network)


def get_assignment_method_dir(svc_network: str, method: str) -> str:
    """Return absolute path to the per-method assignment dir
    (data/Traffic_Flow/Assignment/<svc_network>/<shortest_path|logit>/).

    Args:
        svc_network: service version folder name WITH the '_network' suffix.
        method:      'shortest_path' | 'logit'.
    """
    return os.path.join(get_assignment_dir(svc_network), method)


def get_routing_primitive_path(svc_network: str, method: str, table: str) -> str:
    """Return absolute path to a persisted pre-τ routing-primitive table
    (data/Traffic_Flow/Assignment/<svc_network>/<method>/primitive_<table>.parquet).

    table: 'paths' | 'segments' | 'events' | 'unresolved'. The reloadable baseline
    a Phase 6 subset recompute overwrites per (origin_id, dest_id) and re-aggregates."""
    return os.path.join(get_assignment_method_dir(svc_network, method),
                        f'primitive_{table}.parquet')


def get_flow_dir(svc_network: str, method: str) -> str:
    """Return absolute path to the Phase-6D passenger-flow output dir
    (data/Traffic_Flow/Assignment/<svc_network>/<method>/flows/)."""
    return os.path.join(get_assignment_method_dir(svc_network, method), 'flows')


def get_flow_table_path(svc_network: str, method: str, name: str) -> str:
    """Return absolute path to a Phase-6D flow table inside get_flow_dir
    (e.g. 'flow_segments.gpkg', 'flow_nodes.gpkg', 'flow_segments_by_service.csv')."""
    return os.path.join(get_flow_dir(svc_network, method), name)


def get_flow_plot_dir(svc_network: str, method: str) -> str:
    """Return absolute path to the Phase-6D flow plot dir
    (plots/Traffic_Flow/Assignment/<svc_network>/<method>/flows/)."""
    return os.path.join(get_assignment_plot_dir(svc_network, method), 'flows')


def get_passenger_flow_plot_path(svc_network: str, kind: str) -> str:
    """Combined Phase-6D passenger-flow plot file (PDF).

    Baseline networks save under their own svc-network folder; per-svc-int
    networks pool under Developments/<combo>/ with the id in the filename (no
    per-int subfolders):
      plots/Traffic_Flow/Passenger_Flows/<svc_network>/flows_map_<svc_network>.pdf
      plots/Traffic_Flow/Passenger_Flows/Developments/<combo>/flows_map_<id>.pdf
      plots/Traffic_Flow/Passenger_Flows/Developments/<combo>/flows_map_diff_<id>.pdf

    Args:
        svc_network: baseline svc network ('..._network') or a dev network
                     ('Developments/<combo>/<id>_network').
        kind:        'map' (absolute) | 'diff'.
    """
    root = os.path.join(MAIN, 'plots', 'Traffic_Flow', 'Passenger_Flows')
    parts = svc_network.replace('\\', '/').split('/')
    if parts[0] == 'Developments':
        out_dir = os.path.join(root, 'Developments', parts[1])
        label = parts[2].removesuffix('_network')
    else:
        out_dir = os.path.join(root, svc_network)
        label = svc_network
    stem = f'flows_map_{label}' if kind == 'map' else f'flows_map_diff_{label}'
    return os.path.join(out_dir, f'{stem}.pdf')


def get_assignment_report_xlsx(svc_network: str, method: str, name: str) -> str:
    """Per-method assignment report workbook
    (data/Traffic_Flow/Assignment/<svc_network>/<method>/<name>.xlsx).

    Args:
        svc_network: service version folder name WITH the '_network' suffix.
        method:      'shortest_path' | 'logit'.
        name:        workbook stem (no extension), e.g. 'matrix_all_stations'.
    """
    return os.path.join(get_assignment_method_dir(svc_network, method),
                        f'{name}.xlsx')


def get_assignment_plot_dir(svc_network: str, method: str) -> str:
    """Per-method assignment plot dir
    (plots/Traffic_Flow/Assignment/<svc_network>/<method>/). Mirrors
    get_assignment_method_dir on the plots side so the repo tree is symmetric.

    Args:
        svc_network: service version folder name WITH the '_network' suffix.
        method:      'shortest_path' | 'logit'.
    """
    return os.path.join(MAIN, TRAFFIC_FLOW_ASSIGNMENT_PLOTS_DIR, svc_network, method)


def get_boundary_stations_json(svc_network: str, infra_version: str) -> str:
    """Return absolute path to boundary_stations.json for a svc/infra version pair.

    Args:
        svc_network:   service version folder name WITH the '_network' suffix,
                       e.g. 'AK_2026_S18_network'.
        infra_version: infrastructure version subfolder, e.g. 'AS_2026_ZH'.
    """
    return os.path.join(MAIN, RAIL_LINES_DIR, svc_network, infra_version,
                        'boundary_stations.json')


def svc_version_exists(svc_version: str) -> bool:
    """True if the svc_version network folder has a complete Unprojected rail base."""
    d = os.path.join(MAIN, RAIL_LINES_DIR, svc_version + '_network', SERVICES_UNPROJECTED_SUBDIR)
    return all(
        os.path.isfile(os.path.join(d, f))
        for f in ('rail_lines.gpkg', 'rail_segments.gpkg', 'rail_stops.gpkg')
    )

def svc_feeder_exists(svc_version: str) -> bool:
    """True if the svc_version network folder has a complete Unprojected PT-feeder base."""
    d = os.path.join(MAIN, FEEDER_LINES_DIR, svc_version + '_network', SERVICES_UNPROJECTED_SUBDIR)
    return all(
        os.path.isfile(os.path.join(d, f))
        for f in ('pt_feeder_lines.gpkg', 'pt_feeder_segments.gpkg', 'pt_feeder_stops.gpkg')
    )

def any_svc_rail_exists() -> bool:
    """True if at least one complete Unprojected rail network exists in Rail_Lines."""
    base = os.path.join(MAIN, RAIL_LINES_DIR)
    if not os.path.isdir(base):
        return False
    for entry in os.scandir(base):
        if entry.is_dir() and entry.name.endswith('_network'):
            d = os.path.join(entry.path, SERVICES_UNPROJECTED_SUBDIR)
            if all(os.path.isfile(os.path.join(d, f))
                   for f in ('rail_lines.gpkg', 'rail_segments.gpkg', 'rail_stops.gpkg')):
                return True
    return False

def any_svc_feeder_exists() -> bool:
    """True if at least one complete Unprojected PT-feeder network exists in Feeder_Lines."""
    base = os.path.join(MAIN, FEEDER_LINES_DIR)
    if not os.path.isdir(base):
        return False
    for entry in os.scandir(base):
        if entry.is_dir() and entry.name.endswith('_network'):
            d = os.path.join(entry.path, SERVICES_UNPROJECTED_SUBDIR)
            if all(os.path.isfile(os.path.join(d, f))
                   for f in ('pt_feeder_lines.gpkg', 'pt_feeder_segments.gpkg', 'pt_feeder_stops.gpkg')):
                return True
    return False

def svc_projected_exists(svc_version: str, infra_version: str) -> bool:
    """True if the service version has been projected to the given infra version."""
    return os.path.isfile(get_projected_services_path(svc_version, infra_version))
NETWORK_INFRASTRUCTURE_RAW_NODES                = r"data/Infrastructure/Raw/nodes.gpkg"
NETWORK_INFRASTRUCTURE_RAW_SEGMENTS             = r"data/Infrastructure/Raw/segments.gpkg"
NETWORK_INFRASTRUCTURE_RAW_SEGMENTS_COMPOSITION = r"data/Infrastructure/Raw/segments_composition.gpkg"
# Base/ : macroscopic-simplified network (infrabuild_filter_network.py stage 2)
#         This is the selectable base version for network_builder and version_manager
NETWORK_INFRASTRUCTURE_BASE          = r"data/Infrastructure/Base"
NETWORK_INFRASTRUCTURE_BASE_NODES    = r"data/Infrastructure/Base/nodes.gpkg"
NETWORK_INFRASTRUCTURE_BASE_SEGMENTS = r"data/Infrastructure/Base/segments.gpkg"

# --- Infrastructure Plots ---
INFRASTRUCTURE_PLOTS_DIR = r"plots/Infrastructure"

# --- Network Plots ---
NETWORK_PLOTS_DIR = r"plots/Network"

POPULATION_RASTER = r"data\independent_variable\processed\replacement.pop20_ArcGisExport.tif"
EMPLOYMENT_RASTER = r"data\independent_variable\processed\replacement.empl20_ArcGisExport.tif"
POPULATION_SCENARIO_CANTON_ZH_2050 = r"data\Scenario\KTZH_00000705_00001741.csv"
POPULATION_SCENARIO_CH_BFS_2055 = r"data\Scenario\pop_scenario_switzerland_2055.csv"
POPULATION_SCENARIO_CH_EUROSTAT_2100 = r"data\Scenario\Eurostat_population_CH_2100.xlsx"
POPULATION_PER_COMMUNE_ZH_2018 = r"data\Scenario\population_by_gemeinde_2018.csv"
RANDOM_SCENARIO_CACHE_PATH = r"data\Scenario\cache"  # legacy only (main.py / main_cap.py full-OD pickles; Phase 7 uses the factor store)
DISTRICT_PATH     = r"data/Spatial_Data/Boundaries/SwissBoundaries_Bezirke_2026_CH.gpkg"
COMMUNE_RASTER_TIF = r"data/Spatial_Data/Land_Use/Boundaries/gemeinde_zh.tif"
CANTON_BOUNDARIES_GPKG = r"data/Spatial_Data/Boundaries/Swissboundaries_Cantons_2026_CH.gpkg"
BEZIRKE_BOUNDARIES_GPKG = r"data/Spatial_Data/Boundaries/SwissBoundaries_Bezirke_2026_CH.gpkg"
MUNICIPAL_BOUNDARIES_GPKG = r"data/Spatial_Data/Boundaries/SwissBoundaries_Municipalities_2026_CH.gpkg"
GTFS_TRANSIT_DIR = r"data/Network/GTFS_Timetable"
BUS_LINES_DIR = r"data/Network/Buslines"
FEEDER_LINES_DIR = r"data/Network/Feeder_Lines"
RAIL_LINES_DIR = r"data/Network/Rail_Lines"

# Services subfolder hierarchy (used by services_* scripts)
SERVICES_UNPROJECTED_SUBDIR = "Unprojected"
SERVICES_PROJECTED_SUBDIR   = "Projected"
RAIL_PROCESSED_DIR = r"data/Network/processed"
EDGES_IN_CORRIDOR_GPKG = r"data/Network/processed/edges_in_corridor.gpkg"
CAPACITY_DIR      = r"data/Network/Capacity"
CAPACITY_PLOTS_DIR = r"plots/Network/Capacity"

# --- Catchment / Study Area Boundaries ---
CATCHMENT_AREA_BOUNDARIES_DIR = r"data/Catchment_Area/Boundaries"
SA_BOUNDARY_PATH = r"data/Catchment_Area/Boundaries/study_area_boundary.gpkg"
CA_BOUNDARY_PATH = r"data/Catchment_Area/Boundaries/catchment_area_boundary.gpkg"


def get_projected_services_sa_path(svc_version: str, infra_version: str) -> str:
    """Return absolute path to study-area-filtered projected rail segments for a svc/infra pair."""
    return os.path.join(MAIN, RAIL_LINES_DIR, svc_version + '_network', infra_version, 'rail_segments_sa.gpkg')

STUDY_AREA_DIR           = r"data/Catchment_Area/Boundaries"
STUDY_AREA_BOUNDARY_GPKG = r"data/Catchment_Area/Boundaries/study_area_boundary.gpkg"
STUDY_AREA_BUFFER_GPKG   = r"data/Catchment_Area/Boundaries/study_area_buffer.gpkg"

CATCHMENT_AREA_DIR           = r"data/Catchment_Area"
CATCHMENT_PLOTS_DIR          = r"plots/Catchment_Area"
CATCHMENT_AREA_BOUNDARY_GPKG = r"data/Catchment_Area/Boundaries/catchment_area_boundary.gpkg"
CATCHMENT_AREA_BUFFER_GPKG   = r"data/Catchment_Area/Boundaries/catchment_area_buffer.gpkg"
# Catchment plots - mirrors the data folder structure (data/Catchment_Area/Pop_Empl_Data/...)
POP_EMPL_PLOT_DIR            = r"plots/Catchment_Area/Pop_Empl_Data"
POPULATION_CSV_2023 = r"data/Spatial_Data/Land_Use/Population/Inhabitants_2023_CH.csv"
EMPLOYMENT_CSV_2023 = r"data/Spatial_Data/Land_Use/Employment/Employment_FTE_2023_CH.csv"
# Canton Zurich commune-level actuals (population 1962-2025, employment 2011-2023)
POPULATION_CANTON_ZH_XLSX = r"data/Spatial_Data/Land_Use/Population/Canton_Zurich/KTZH_00000127_00001245.xlsx"
# POPULATION_CANTON_BE_XLSX = r
POPULATION_CANTON_XLSX = POPULATION_CANTON_ZH_XLSX
EMPLOYMENT_CANTON_ZH_CSV  = r"data/Spatial_Data/Land_Use/Employment/Canton_Zurich/ZGZ_Daten_Komplett_vzae_sektor_2026-05-15_144034.csv"
# EMPLOYMENT_CANTON_BE_XLSX = r
LAKES_SHP    = r"data/Spatial_Data/Land_Use/Hydrography/swissTLMRegio_Lake.shp"
LAKES_CA_GPKG = r"data/Spatial_Data/Land_Use/Hydrography/lakes_ca.gpkg"
LAKES_SA_GPKG = r"data/Spatial_Data/Land_Use/Hydrography/lakes_sa.gpkg"

# Connecting-curve composition cache (per-CC structure breakdown) is now partitioned by
# infra network — see get_cc_composition_cache(network). Filled on cache-miss by the
# infra_ints_connecting_curve standalone CLI.

CONSTRUCTION_COSTS =  r"data/costs/construction_cost.csv"
TOTAL_COST_WITH_GEOMETRY = r"data/costs/total_costs_with_geometry.csv"
TOTAL_COST_RAW = r"data/costs/total_costs_raw.csv"
COST_AND_BENEFITS_DISCOUNTED = r"data/costs/costs_and_benefits_dev_discounted.csv"
COSTS_CONNECTION_CURVES = r"data/costs/costs_connection_curves.xlsx"

TTS_CACHE = r"data/Network/travel_time/cache/compute_tts_cache.pkl"

# Phase 8 valuation outputs — combo-keyed (main_new). The flat data/costs/*.csv
# constants above remain the legacy main.py/main_cap.py chain.
COSTS_DIR = r"data/costs"


def get_costs_combo_dir(combo: str) -> str:
    """Return the Phase-8 valuation output dir (data/costs/Developments/<combo>/).

    Args:
        combo: the '<infra>__<svc>' workspace key.
    """
    return os.path.join(MAIN, COSTS_DIR, DEVELOPMENTS_SUBDIR_NAME, combo)


def get_tts_csv(combo: str) -> str:
    """Return absolute path to the combined 8A benefits table
    (traveltime_savings.csv — one row per svc-int x scenario x year)."""
    return os.path.join(get_costs_combo_dir(combo), 'traveltime_savings.csv')


def get_tts_cache_path(svc_int_id: str, combo: str) -> str:
    """Return absolute path to a svc-int's 8A TTS cache parquet
    (data/costs/Developments/<combo>/<id>_network/tts.parquet)."""
    return os.path.join(MAIN, COSTS_DIR, svc_int_network_name(svc_int_id, combo),
                        'tts.parquet')


def get_construction_cost_csv(combo: str) -> str:
    """Return absolute path to the 8B cost table (construction_cost.csv — one
    row per svc-int, legacy Dev_/CapInt_/Total*/Yearly* column schema)."""
    return os.path.join(get_costs_combo_dir(combo), 'construction_cost.csv')


def get_costs_and_benefits_discounted_csv(combo: str) -> str:
    """Return absolute path to the Phase-9 discounted cost-benefit table
    (costs_and_benefits_discounted.csv — one row per svc-int x scenario x year,
    columns const_cost/maint_cost/uncovered_op_cost/benefit)."""
    return os.path.join(get_costs_combo_dir(combo),
                        'costs_and_benefits_discounted.csv')


def get_total_costs_raw_csv(combo: str) -> str:
    """Return absolute path to the Phase-9 aggregated table (total_costs_raw.csv
    — one row per svc-int x scenario, discounted sums over the valuation years)."""
    return os.path.join(get_costs_combo_dir(combo), 'total_costs_raw.csv')


def get_total_costs_csv(combo: str) -> str:
    """Return absolute path to the Phase-9 wide table (total_costs.csv — one
    row per svc-int, per-scenario savings/net-benefit/BCR columns)."""
    return os.path.join(get_costs_combo_dir(combo), 'total_costs.csv')


def get_total_costs_summary_csv(combo: str) -> str:
    """Return absolute path to the Phase-9 summary (total_costs_summary.csv —
    one row per svc-int, scenario statistics + net benefit + BCR)."""
    return os.path.join(get_costs_combo_dir(combo), 'total_costs_summary.csv')


def get_total_costs_geometry_gpkg(combo: str) -> str:
    """Return absolute path to the Phase-9 result layer
    (total_costs_with_geometry.gpkg — summary attributes + the svc-int's
    projected service delta unioned with its required CC arcs, EPSG:2056)."""
    return os.path.join(get_costs_combo_dir(combo),
                        'total_costs_with_geometry.gpkg')


def get_factsheet_path(combo: str, svc_int_id: str) -> str:
    """Return absolute path to a svc-int's Phase-9 A4 factsheet PDF
    (plots/Developments/<combo>/results/factsheets/factsheet_<id>.pdf)."""
    return os.path.join(get_developments_plot_dir(combo, 'results'),
                        'factsheets', f'factsheet_{svc_int_id}.pdf')

PLOT_DIRECTORY = r"plots"
PLOT_SCENARIOS = r"plots/scenarios"

def get_rail_services_path(version: str) -> str:
    """Return the rail services path for legacy version names.

    Legacy names (used by main.py and main_cap.py) are mapped to their fixed
    file paths.  New versioned names (e.g. 'AK_2035' paired with an infra
    version) must use get_projected_services_path(svc_version, infra_version).
    """
    _legacy = {
        'AK_2035':          RAIL_SERVICES_AK2035_PATH,
        'AK_2035_extended': RAIL_SERVICES_AK2035_EXTENDED_PATH,
        'current':          RAIL_SERVICES_2024_PATH,
        '2024_extended':    RAIL_SERVICES_AK2024_EXTENDED_PATH,
    }
    if version in _legacy:
        return _legacy[version]
    raise ValueError(
        f"Rail services path for '{version}' must be resolved via "
        f"get_projected_services_path(svc_version, infra_version)."
    )


# --- Validation track (V-track) ---
# Standalone analysis track (not a main_new phase): validation_* modules read
# pipeline outputs and write under infraScanRail/validation/ (sibling of data/
# and plots/; plots nest INSIDE each check dir per the 2026-06-12 decision).
VALIDATION_DIR = r"validation"
_VALIDATION_CHECKS = ('v1_station_numbers', 'v2_link_flows', 'v3_sensitivity',
                      'single_service')

# Observed reference datasets (V1 SBB Ein-/Aussteigende; V2 NPVM link loads)
SBB_STATION_NUMBERS_XLSX = r"data/Spatial_Data/Transit_Network/SBB_Station_Flows/SBB_Station_Numbers.xlsx"
BELASTUNG_RAIL_GPKG      = r"data/Spatial_Data/Transit_Network/Belastung-Personenverkehr-Bahn/belastung-personenverkehr-bahn_2056.gpkg"


def get_validation_dir() -> str:
    """Return absolute path to the V-track root (infraScanRail/validation/)."""
    return os.path.join(MAIN, VALIDATION_DIR)


def get_validation_check_dir(check: str) -> str:
    """Return absolute path to a validation check's output dir.

    Args:
        check: 'v1_station_numbers' | 'v2_link_flows' | 'v3_sensitivity'.
    """
    if check not in _VALIDATION_CHECKS:
        raise ValueError(f"Unknown validation check '{check}' — expected one of {_VALIDATION_CHECKS}.")
    return os.path.join(get_validation_dir(), check)


def get_validation_plot_dir(check: str) -> str:
    """Return absolute path to a check's plot dir (validation/<check>/plots/)."""
    return os.path.join(get_validation_check_dir(check), 'plots')


def get_validation_config_path(name: str) -> str:
    """Return absolute path to a V-track config CSV (validation/config/<name>).

    Configs are manual inputs (UIC alias map, V2 match exceptions) — modules
    create header templates when missing and never overwrite them.
    """
    return os.path.join(get_validation_dir(), 'config', name)


def get_validation_baseline_dir(alloc: str, routing: str) -> str:
    """Return absolute path to a snapshotted baseline variant
    (validation/_baselines/<alloc>__<routing>/).

    Allocation variants OVERWRITE each other in the 4C/6D trees (the Assignment
    tree keys routing method only), so every variant run is snapshotted here
    before settings change.

    Args:
        alloc:   'Municipal' | 'PT_Feeder'.
        routing: 'shortest_path' | 'logit'.
    """
    return os.path.join(get_validation_dir(), '_baselines', f'{alloc}__{routing}')


def get_v3_combo_dir(combo_label: str) -> str:
    """Return absolute path to a V3 GC-settings combo archive
    (validation/v3_sensitivity/<combo-label>/).

    Args:
        combo_label: '<TRAVEL_COST_METHOD>__<TRANSFER_COST_MODEL>',
            e.g. 'calibrated__explicit'.
    """
    return os.path.join(get_validation_check_dir('v3_sensitivity'), combo_label)


def get_v3_comparison_dir() -> str:
    """Return absolute path to the V3 cross-combo comparison dir
    (validation/v3_sensitivity/_comparison/)."""
    return os.path.join(get_validation_check_dir('v3_sensitivity'), '_comparison')