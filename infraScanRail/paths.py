
MAIN = r"D:\ETH_Msc\MA\infraScan\infraScanRail"
RAIL_SERVICES_AK2035_PATH= r'data\temp\railway_services_ak2035.gpkg'
RAIL_SERVICES_AK2035_EXTENDED_PATH = r'data\temp\railway_services_ak2035_extended.gpkg'
RAIL_SERVICES_2024_PATH= r'data/temp/network_railway-services.gpkg'
RAIL_SERVICES_AK2024_EXTENDED_PATH = r'data/temp/network2024_railway_services_extended.gpkg'
NEW_LINKS_UPDATED_PATH = r"data\Network\processed\updated_new_links.gpkg"
NEW_RAILWAY_LINES_PATH = r"data\Network\processed\new_railway_lines.gpkg"
NETWORK_WITH_ALL_MODIFICATIONS = r"data\Network\processed\combined_network_with_all_modifications.gpkg"
DEVELOPMENT_DIRECTORY = r"data\Network\processed\developments"

RAIL_POINTS_PATH = r"data\Network\Rail_Node.csv"
OD_KT_ZH_PATH = r'data/_basic_data/KTZH_00001982_00003903.xlsx'
OD_STATIONS_KT_ZH_PATH = r'data/traffic_flow/od/rail/ktzh/od_matrix_stations_ktzh_20.csv'
COMMUNE_TO_STATION_PATH = r"data\Network\processed\Communes_to_railway_stations_ZH.xlsx"

POPULATION_RASTER = r"data\independent_variable\processed\replacement.pop20_ArcGisExport.tif"
EMPLOYMENT_RASTER = r"data\independent_variable\processed\replacement.empl20_ArcGisExport.tif"
POPULATION_SCENARIO_CANTON_ZH_2050 = r"data\Scenario\KTZH_00000705_00001741.csv"
POPULATION_SCENARIO_CH_BFS_2055 = r"data\Scenario\pop_scenario_switzerland_2055.csv"
POPULATION_SCENARIO_CH_EUROSTAT_2100 = r"data\Scenario\Eurostat_population_CH_2100.xlsx"
POPULATION_PER_COMMUNE_ZH_2018 = r"data\Scenario\population_by_gemeinde_2018.csv"
RANDOM_SCENARIO_CACHE_PATH = r"data\Scenario\cache"

CONSTRUCTION_COSTS =  r"data/costs/construction_cost.csv"
TOTAL_COST_WITH_GEOMETRY = r"data/costs/total_costs_with_geometry.csv"
TOTAL_COST_RAW = r"data/costs/total_costs_raw.csv"
COST_AND_BENEFITS_DISCOUNTED = r"data/costs/costs_and_benefits_dev_discounted.csv"

TTS_CACHE = r"data/Network/travel_time/cache/compute_tts_cache.pkl"

PLOT_DIRECTORY = r"plots"
PLOT_SCENARIOS = r"plots/scenarios"

def get_rail_services_path(rail_network_settings):
    """
    Returns the path to the rail services file based on the rail network settings.
    """
    if rail_network_settings == 'AK_2035':
        return RAIL_SERVICES_AK2035_PATH
    elif rail_network_settings == 'AK_2035_extended':
        return RAIL_SERVICES_AK2035_EXTENDED_PATH
    elif rail_network_settings == 'current':
        return RAIL_SERVICES_2024_PATH
    elif rail_network_settings == '2024_extended':
        return RAIL_SERVICES_AK2024_EXTENDED_PATH
    else:
        raise ValueError("Invalid rail network settings provided.")