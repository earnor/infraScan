from shapely.geometry import Polygon
import paths

rail_network = 'AK_2035_extended' # either 'AK_2035','AK_2035_extended' or 'current' or '2024_extended'

#CACHE
use_cache_network = True
use_cache_pt_catchment = True
use_cache_developments = False
use_cache_catchmentOD = True
use_cache_stationsOD = False
use_cache_traveltime_graph = True
use_cache_scenarios = False
use_cache_tts_calc = False

infra_generation_modification_type = 'NEW_DIRECT_CONNECTIONS' #either 'EXTEND_LINES' or 'NEW_DIRECT_CONNECTIONS' or 'ALL'
#infra_generation_rail_network: either 'RAIL_SERVICES_AK2035_PATH' or 'RAIL_SERVICES_AK2035_EXTENDED_PATH' or 'RAIL_SERVICES_2024_PATH' or 'RAIL_SERVICES_2024_EXTENDED_PATH'
infra_generation_rail_network = paths.RAIL_SERVICES_AK2035_PATH

OD_type = 'canton_ZH' #either 'canton_ZH' or 'pt_catchment_perimeter'
only_demand_from_to_perimeter = True

scenario_type = 'GENERATED' #either 'GENERATED' or 'STATIC_9' or 'dummy'
amount_of_scenarios = 100
start_year_scenario = 2018
end_year_scenario = 2100
start_valuation_year = 2050
#choose which OD

plot_passenger_flow = False
plot_railway_line_load = True




perimeter_infra_generation = Polygon([  #No GeoJSON with this polygon type!
    (2700989.862, 1235663.403),
    (2708491.515, 1239608.529),
    (2694972.602, 1255514.900),
    (2687415.817, 1251056.404)  # closing the polygon
])
perimeter_demand = perimeter_infra_generation



raster_size = (170,210)

pop_scenarios = [
        "pop_urban_", "pop_equal_", "pop_rural_",
        "pop_urba_1", "pop_equa_1", "pop_rura_1",
        "pop_urba_2", "pop_equa_2", "pop_rura_2"]
empl_scenarios = ["empl_urban", "empl_equal", "empl_rural",
                   "empl_urb_1", "empl_equ_1", "empl_rur_1",
                   "empl_urb_2", "empl_equ_2", "empl_rur_2"]

dev_id_start_extended_lines = 100000
dev_id_start_new_direct_connections = 101000

