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
DEVELOPMENT_DIRECTORY = r"data\Network\processed\developments"

RAIL_NODES_PATH = r"data\Network\Rail_Node.csv"
RAIL_POINTS_PATH = r"data\Network\processed\points.gpkg"
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
_OD_METHOD_DIRS = {'pt_feeder': 'PT_Feeder', 'municipal': 'Municipal'}

COMMUNE_TO_STATION_PATH = r"data\Network\processed\Communes_to_railway_stations_ZH.xlsx"
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
# Developments/ : infra-intervention registries (cc, cap, …) — Topic 2
INFRASTRUCTURE_DEVELOPMENTS_DIR = r"data/Infrastructure/Developments"
# Dev_Full/ : reference version with ALL infra ints applied (visualisation)
INFRASTRUCTURE_DEV_FULL_DIR     = r"data/Infrastructure/Developments/Dev_Full"
# Derived/ : auto-built derived versions for capacity runs (not main infra folder)
INFRASTRUCTURE_DERIVED_DIR      = r"data/Infrastructure/Developments/Derived"
# Network Developments/ : service-intervention registries (ext, …) — Topic 2
NETWORK_DEVELOPMENTS_DIR        = r"data/Network/Developments"


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


def get_infra_int_registry(int_type: str) -> str:
    """Return absolute path to the infra-int registry gpkg for a given type.

    Args:
        int_type: short code (e.g. 'cc' for connecting curves, 'cap' for passing sidings).
    """
    return os.path.join(MAIN, INFRASTRUCTURE_DEVELOPMENTS_DIR, f"{int_type}_interventions.gpkg")


def get_svc_int_registry(int_type: str) -> str:
    """Return absolute path to the svc-int registry xlsx for a given type.

    Args:
        int_type: short code (e.g. 'ext' for extended lines).
    """
    return os.path.join(MAIN, NETWORK_DEVELOPMENTS_DIR, f"{int_type}_interventions.xlsx")

def get_infra_version_dir(version: str) -> str:
    """Return absolute path to the named infrastructure version directory."""
    return os.path.join(MAIN, NETWORK_INFRASTRUCTURE_DIR, version)

def get_derived_infra_version_dir(version: str) -> str:
    """Return absolute path to a derived infra version under Developments/Derived/.

    Args:
        version: derived version name (e.g. 'AS_2026_ZH+cap_ps_0001').
    """
    return os.path.join(MAIN, INFRASTRUCTURE_DERIVED_DIR, version)

def derived_version_exists(version: str) -> bool:
    """True if nodes.gpkg, segments.gpkg and segments_composition.gpkg all exist
    in the derived version directory."""
    d = get_derived_infra_version_dir(version)
    return all(
        os.path.isfile(os.path.join(d, f))
        for f in ('nodes.gpkg', 'segments.gpkg', 'segments_composition.gpkg')
    )

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


def get_station_od_csv(svc_network: str, method: str, window: str) -> str:
    """Return absolute path to a station-pair OD matrix CSV for a (method, window).

    Args:
        method: 'pt_feeder' | 'municipal'.
        window: 'peak' | 'off_peak' | 'full_day'.
    """
    return os.path.join(get_station_od_dir(svc_network, method),
                        f'od_matrix_stations_{window}.csv')


def get_od_top5_xlsx(svc_network: str, method: str) -> str:
    """Return absolute path to the per-method top-5 origins/destinations workbook."""
    return os.path.join(get_station_od_dir(svc_network, method), 'od_top5.xlsx')


def get_station_od_matrix_xlsx(svc_network: str, method: str) -> str:
    """Return absolute path to the per-method full station×station OD matrix
    workbook (one sheet per time window)."""
    return os.path.join(get_station_od_dir(svc_network, method),
                        'od_matrix_stations.xlsx')


def get_od_method_plot_dir(svc_network: str, method: str) -> str:
    """Return absolute path to the per-method OD plot dir
    (plots/Traffic_Flow/OD/<svc_network>/<PT_Feeder|Municipal>/).

    Mirrors get_station_od_dir on the plots side so the repo tree is symmetric.
    """
    return os.path.join(MAIN, TRAFFIC_FLOW_OD_PLOTS_DIR, svc_network,
                        _OD_METHOD_DIRS.get(method, method))


def get_gateway_dir(svc_network: str) -> str:
    """Return absolute path to the gateway-assignment dir
    (data/Traffic_Flow/OD/<svc_network>/Gateway/)."""
    return os.path.join(get_od_version_dir(svc_network), 'Gateway')


def get_od_routing_dir(svc_network: str) -> str:
    """Return absolute path to the W4b rail-routing output dir
    (data/Traffic_Flow/OD/<svc_network>/Routing/)."""
    return os.path.join(get_od_version_dir(svc_network), 'Routing')


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
RANDOM_SCENARIO_CACHE_PATH = r"data\Scenario\cache"
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
CATCHMENT_AREA_BOUNDARY_GPKG = r"data/Catchment_Area/Boundaries/catchment_area_boundary.gpkg"
CATCHMENT_AREA_BUFFER_GPKG   = r"data/Catchment_Area/Boundaries/catchment_area_buffer.gpkg"
# Catchment plots - mirrors the data folder structure (data/Catchment_Area/Pop_Empl_Data/...)
POP_EMPL_PLOT_DIR            = r"plots/Catchment_Area/Pop_Empl_Data"
POPULATION_CSV_2023 = r"data/Spatial_Data/Land_Use/Population/Inhabitants_2023_CH.csv"
EMPLOYMENT_CSV_2023 = r"data/Spatial_Data/Land_Use/Employment/Employment_FTE_2023_CH.csv"
# Canton Zurich commune-level actuals (population 1962-2025, employment 2011-2023)
POPULATION_CANTON_ZH_XLSX = r"data/Spatial_Data/Land_Use/Population/Canton_Zurich/KTZH_00000127_00001245.xlsx"
EMPLOYMENT_CANTON_ZH_CSV  = r"data/Spatial_Data/Land_Use/Employment/Canton_Zurich/ZGZ_Daten_Komplett_vzae_sektor_2026-05-15_144034.csv"
LAKES_SHP    = r"data/Spatial_Data/Land_Use/Hydrography/swissTLMRegio_Lake.shp"
LAKES_CA_GPKG = r"data/Spatial_Data/Land_Use/Hydrography/lakes_ca.gpkg"
LAKES_SA_GPKG = r"data/Spatial_Data/Land_Use/Hydrography/lakes_sa.gpkg"

CONSTRUCTION_COSTS =  r"data/costs/construction_cost.csv"
TOTAL_COST_WITH_GEOMETRY = r"data/costs/total_costs_with_geometry.csv"
TOTAL_COST_RAW = r"data/costs/total_costs_raw.csv"
COST_AND_BENEFITS_DISCOUNTED = r"data/costs/costs_and_benefits_dev_discounted.csv"
COSTS_CONNECTION_CURVES = r"data/costs/costs_connection_curves.xlsx"

TTS_CACHE = r"data/Network/travel_time/cache/compute_tts_cache.pkl"

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