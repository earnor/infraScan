import numpy as np

VTTS = 14.8 # CHF/h
# Construction costs
yearly_maintenance_to_construction_cost_factor = 0.03
track_cost_per_meter = 33250  # CHF per meter
tunnel_cost_per_meter = 104000  # CHF per meter per track (From Schweizer())
bridge_cost_per_meter = 70000  # CHF per meter per track
track_maintenance_cost = track_cost_per_meter * yearly_maintenance_to_construction_cost_factor  #132  # CHF per meter per track per year
tunnel_maintenance_cost = tunnel_cost_per_meter * yearly_maintenance_to_construction_cost_factor #132  # CHF/m/a
bridge_maintenance_cost = bridge_cost_per_meter * yearly_maintenance_to_construction_cost_factor#368.8  # CHF/m/a

operating_cost_s_bahn_per_meter = 879   #Estimation from S14 HB - Hinwil 2024 from the Abgeltungen and KDG data of BAV, based on real line length
detour_factor_tracks = 1.1  # Factor to account for detours in track length in comparison to a straight line between stations
general_KDG = 0.623

duration = 50  # 50 years
tts_valuation_period = (2050,2100)
construction_start_year = 2050

tau = 0.13
discount_rate = 0.03  # 3% discount rate

average_train_change_time = 7.1 #Axhausen, 2014
change_time_comfort_factor = 1.7
comfort_weighted_change_time = int(np.round(average_train_change_time * change_time_comfort_factor))  # Comfort weighted change time in minutes

# Capacity Enhancement Interventions
# Siding lengths for cost calculations (based on track_cost_per_meter)
segment_siding_length_m = 1000  # Passing siding length (meters)
station_siding_length_m = 500   # Station track length (meters)
platform_cost_per_unit = 50000  # CHF per platform