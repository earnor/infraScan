import numpy as np

VTTS = 14.8 # CHF/h
# Construction costs
yearly_maintenance_to_construction_cost_factor = 0.03
track_cost_per_meter = 33250  # CHF per meter: SBB Kostentool "22200" / Old approach "33250"
tunnel_cost_per_meter = 104000  # CHF per meter per track: SBB Kostentool "70000" / Old approach "104000"
bridge_cost_per_meter = 70000  # CHF per meter per track: SBB Kostentool "47000" / Old approach "70000"
track_maintenance_cost = track_cost_per_meter * yearly_maintenance_to_construction_cost_factor # CHF per meter per track per year
tunnel_maintenance_cost = tunnel_cost_per_meter * yearly_maintenance_to_construction_cost_factor # CHF/m/a
bridge_maintenance_cost = bridge_cost_per_meter * yearly_maintenance_to_construction_cost_factor # CHF/m/a

operating_cost_s_bahn_per_meter = 879   #Estimation from S14 HB - Hinwil 2024 from the Abgeltungen and KDG data of BAV, based on real line length
detour_factor_tracks = 1.1  # Factor to account for detours in track length in comparison to a straight line between stations
general_KDG = 0.623

duration = 50  # 50 years
tts_valuation_period = (2050,2100)
construction_start_year = 2050

# --- Demand-window share factors (fraction of daily interzonal PT demand
# per representative hour). No Swiss authority publishes an explicit τ;
# values triangulated from Swiss and international precedent. ---
TAU_PEAK_SHARE     = 0.13   # Peak hour (busiest single hour). Range 0.12-0.15 for
                            # commute-dominated commuter rail. Source: FHWA K-factor
                            # (8-15% urban, 15% rural); NPVM 2023 MSP/ASP-implied
                            # 12-15% range; UK PDFH commuter convention (Worsley
                            # 2012). Central value defensible for ZH-region S-Bahn.
TAU_OFFPEAK_SHARE  = 0.06   # Off-peak representative hour. Conservation over the
                            # 06:00-20:00 operational day (14 h) with the two
                            # busiest peak hours (AM + PM) at τ_peak:
                            # (1 − 2·0.13) / 12 ≈ 0.062. Peak windows are
                            # 06:00-09:00 and 16:00-19:00; the inter-peak off-peak
                            # window is 09:00-16:00.
TAU_FULL_DAY_SHARE = 1.00   # Full daily total. No scaling.

# Logit route-choice scale parameter (1 / GC-minute). Controls concentration
# of demand on lower-GC services. θ=0.20: a 10-min GC disadvantage reduces
# a service's share by factor exp(-2) ≈ 0.14. Range 0.10–0.30 from Swiss
# transit calibration (Vrtic & Axhausen 2000; Ben-Akiva & Lerman 1985).
# Central value 0.20 defensible for ZH-region S-Bahn.
LOGIT_ROUTE_THETA   = 0.20

discount_rate = 0.03  # 3% discount rate

average_train_change_time = 7.1 # Axhausen, 2014
change_time_comfort_factor = 1.7
comfort_weighted_change_time = int(np.round(average_train_change_time * change_time_comfort_factor))  # Comfort weighted change time in minutes

# --- Generalised cost weights (eq. IVT, GC reference v3 Section 2 & 11) ---
# Sources: NPVM (BAV/ARE); Axhausen et al. (2008) Swiss Mikrozensus route-choice.
# Per-weight metadata sourced from markdowns/MT_Task_1_OD/infraScanRail_Weight_Overview_Table.md.
# All unitless weights set to 1.0 (neutralised baseline); restore recommended values for production runs.
W_IVT      = 1.0   # range 1.0 (definitional). Wardman 2004; Axhausen 2008. Swiss: NPVM/NIBA/NISTRA all 1.0. Recommended: 1.0
W_WAIT     = 1.5   # range 1.5–2.5. Wardman 2004; Axhausen 2008 (β_wait/β_IVT≈1.5–2.0); Ortelli 2025 ≈1.98. Swiss: NPVM implicit >1, value not published. Recommended: 1.5
W_WALK     = 2.0   # range 1.5–2.5. Wardman 2004 ≈2.0; Axhausen 2008 (β_walk/β_IVT≈2.0); Ortelli 2025 ≈1.78. Swiss: NPVM implicit, ARE 2022 uses buffers instead. Recommended: 2.0
W_BIKE     = 1.5   # range 1.0–2.0. No Swiss-specific source; PDFH (UK) ≈1.4–1.6. Swiss: not in NPVM/ARE; SBB B+R not monetised. Recommended: 1.5 (author choice)
# Transfer-specific weight (Axhausen 2014 SVI 2001/534): applies in the 'explicit' model
# to the combined walk+wait component of a transfer (see formula in PI_TRANSFER_MIN block).
W_TRANSFER = 2.0   # range 1.5–2.5. Wardman 2004; Axhausen 2008; Ortelli 2025 ≈1.98. Swiss: NPVM implicit >1, "gewichtet" vs "ungewichtet" Abb. 59 NPVM 2023. Recommended: 2.0

# --- Speed and detour factors ---
# ARE 2022 implicit walking speed; NPVM convention for detour factor
WALK_SPEED_KMH    = 5.0    # km/h  (mirrors WALK_SPEED_MS in catchment_allocate.py)
WALK_DETOUR       = 1.25   # Luftlinie → actual walking distance (ω_walk)
CYCLE_SPEED_KMH   = 15.0   # km/h  (mirrors CYCLE_SPEED_MS in catchment_allocate.py)
CYCLE_DETOUR      = 1.20   # ω_bike (author choice)
CYCLE_MAX_RADIUS_M = 2500  # network distance cap (author choice)

# --- Transfer penalty ---
# Axhausen (2014): raw lump-sum 7.1 min × W_TRANSFER 1.7 = 12.1 min eq. IVT.
# Used as-is in the 'fixed_value' model; in the 'explicit' model
#   π_transfer = W_TRANSFER × (TRANSFER_WALK_MIN + t_wait(h_connecting)).
PI_TRANSFER_MIN  = 12.1   # eq. IVT min, Axhausen model (already weighted at 1.7)
TRANSFER_WALK_MIN = 4.0   # raw platform-walk time at transfer (min), explicit model only

# --- Piecewise wait function parameters (Wardman 2004; Bates et al. 2001) ---
# Not codified in Swiss norm — document explicitly in thesis.
WAIT_THRESHOLD_MIN = 12.0  # below: random-arrival h/2; above: schedule-anchored
WAIT_SLOPE_ABOVE   = 0.25  # slope for h > threshold (continuity gives offset = 6 min)


def t_wait_min(headway_min: float) -> float:
    """Expected wait time in minutes given aggregated stop-level headway (ARE 2022 §5).

    Piecewise: random-arrival regime below 12 min (h/2), schedule-anchored above
    (6 + 0.25*(h-12)). Returns 0.0 for non-finite or non-positive headway.
    """
    import math
    if not math.isfinite(headway_min) or headway_min <= 0:
        return 0.0
    if headway_min <= WAIT_THRESHOLD_MIN:
        return headway_min / 2.0
    return 6.0 + WAIT_SLOPE_ABOVE * (headway_min - WAIT_THRESHOLD_MIN)


# Capacity Enhancement Interventions
# Legacy fixed-cost siding parameters — superseded by pure per-meter cost
# (Topic 1, 2026-05). Kept as fallback / backward compatibility only.
# New Tier-1 cap-intervention cost = L_siding × track_cost_per_meter.
segment_siding_costs = 33250000  # Track siding costs (1000m): SBB Kostentool "11500000" / Old approach "33250000"  [LEGACY]
station_siding_costs = 33250000   # Station siding costs (1000m): SBB Kostentool "9950000" / Old approach "18300000"  [LEGACY, bumped 550→1000m for consistency]
platform_cost_per_unit = 0  # Platform costs per unit: SBB Kostentool "6930000" / Old approach "0" station adjustments in the station siding costs