from shapely.geometry import Polygon
import paths

rail_network = 'AK_2035_extended' # either 'AK_2035','AK_2035_extended' or 'current'

#CACHE
use_cache_pt_catchment = True
use_cache_developments = True
use_cache_catchmentOD = True
use_cache_stationsOD = False
use_cache_traveltime_graph = True

infra_generation_modification_type = 'EXTEND_LINES' #either 'EXTEND_LINES' or 'NEW_DIRECT_CONNECTIONS' or 'ALL'
#infra_generation_rail_network: either 'RAIL_SERVICES_AK2035_PATH' or 'RAIL_SERVICES_AK2035_EXTENDED_PATH' or 'RAIL_SERVICES_2024_PATH'
infra_generation_rail_network = paths.RAIL_SERVICES_AK2035_PATH

OD_type = 'canton_ZH' #either 'canton_ZH' or 'pt_catchment_perimeter'

scenario_type = 'GENERATED' #either 'GENERATED' or 'STATIC_9'
amount_of_scenarios = 10
#choose which OD



perimeter_infra_generation = Polygon([  #No GeoJSON with this polygon type!
    (2700989.862, 1235663.403),
    (2708491.515, 1239608.529),
    (2694972.602, 1255514.900),
    (2687415.817, 1251056.404)  # closing the polygon
])

raster_size = (170,210)

pop_scenarios = [
        "pop_urban_", "pop_equal_", "pop_rural_",
        "pop_urba_1", "pop_equa_1", "pop_rura_1",
        "pop_urba_2", "pop_equa_2", "pop_rura_2"]
empl_scenarios = ["empl_urban", "empl_equal", "empl_rural",
                   "empl_urb_1", "empl_equ_1", "empl_rur_1",
                   "empl_urb_2", "empl_equ_2", "empl_rur_2"]