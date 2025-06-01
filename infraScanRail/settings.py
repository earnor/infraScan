rail_network = 'AK_2035_extended' # either 'AK_2035','AK_2035_extended' or 'current'

use_cache_pt_catchment = True
use_cache_developments = True
use_cache_catchmentOD = True
use_cache_stationsOD = False
use_cache_traveltime_graph = True

OD_type = 'canton_ZH' #either 'canton_ZH' or 'pt_catchment_perimeter'
#choose with infra generation
#scenario type
amount_of_scenarios = 10
#choose which OD


raster_size = (170,210)

pop_scenarios = [
        "pop_urban_", "pop_equal_", "pop_rural_",
        "pop_urba_1", "pop_equa_1", "pop_rura_1",
        "pop_urba_2", "pop_equa_2", "pop_rura_2"]
empl_scenarios = ["empl_urban", "empl_equal", "empl_rural",
                   "empl_urb_1", "empl_equ_1", "empl_rur_1",
                   "empl_urb_2", "empl_equ_2", "empl_rur_2"]