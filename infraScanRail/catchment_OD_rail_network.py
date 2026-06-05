"""catchment_OD_rail_network.py
Last modified: 2026-05-06

Rail network OD routing for infraScanRail. Takes W3 station-pair OD matrices
and distributes demand across specific rail services using two methods:
  A) Spiess & Florian (1989) optimal strategy (attractive-set allocation)
  B) Logit route choice (GC-based, scale parameter θ = cp.LOGIT_ROUTE_THETA)

Both methods support direct services (item c) and 1-transfer routing (item d).
Pairs with no valid in-catchment path are flagged and their W3 trip value is
preserved unmodified in Output 2 (they likely use an out-of-catchment transfer).

Public entry point: route_od_matrices(svc_version, use_cache, od_method) -> None

Outputs (under data/Traffic_Flow/OD/<svc_network>/Routing/):
  sf_service_assignment_{window}.csv
  logit_service_assignment_{window}.csv
  gc_matrix_sf.csv
  gc_matrix_logit.csv
"""

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio

import paths
import cost_parameters as cp
import catchment_base
import catchment_allocate

_CODEBASE_CRS        = catchment_base.CODEBASE_CRS
TRANSFER_PENALTY_MIN = cp.comfort_weighted_change_time   # 12 min


# ===============================================================================
# DATA LOADING
# ===============================================================================

def _load_rail_segments_with_tt() -> pd.DataFrame:
    """Load rail segments from all temporal subfolders, keeping travel time.

    Extends catchment_allocate._load_segs_all_periods by also retaining the
    TT column (per-segment travel time in minutes), renamed to travel_time_min.

    Returns:
        DataFrame[route_id, direction_id, variant_rank, from_stop_id,
                  to_stop_id, travel_time_min], deduped on the 5-column key.
    """
    _PERIOD_SPECS = [
        ('All_Day',  '_allday',  None),
        ('Peak',     '_peak',    'peak_only'),
        ('Off_Peak', '_offpeak', 'offpeak_only'),
    ]
    base = catchment_allocate._RAIL_BASE
    name = 'rail_segments'
    frames = []
    for subfolder, suffix, _period in _PERIOD_SPECS:
        path = os.path.join(base, subfolder, f'{name}{suffix}.gpkg')
        if not os.path.exists(path):
            continue
        for layer_name, _ in pyogrio.list_layers(path):
            gdf = gpd.read_file(path, layer=layer_name)
            gdf = gdf.rename(columns={
                'from_stop_nr': 'from_stop_id',
                'to_stop_nr':   'to_stop_id',
                'GTFS_ID':      'route_id',
                'TT':           'travel_time_min',
            })
            keep = ['from_stop_id', 'to_stop_id', 'route_id',
                    'direction_id', 'variant_rank', 'travel_time_min']
            keep = [c for c in keep if c in gdf.columns]
            frames.append(gdf[keep].copy())

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined['travel_time_min'] = pd.to_numeric(
        combined['travel_time_min'], errors='coerce').fillna(0.0)
    return combined.drop_duplicates(
        subset=['from_stop_id', 'to_stop_id',
                'route_id', 'direction_id', 'variant_rank'])


def _load_w3_od(csv_path: str, rail_stations: gpd.GeoDataFrame,
                name_lookup: dict) -> pd.DataFrame:
    """Load a W3 station-pair OD CSV; return long-format with id_point keys.

    The W3 CSVs use station names (possibly disambiguated) as both the row
    index ('origin') and column headers ('destination'). This function reverses
    the name_lookup to recover integer id_point values.

    Args:
        csv_path:     Absolute path to the W3 all-day OD CSV.
        rail_stations: GDF with id_point and stop_name columns.
        name_lookup:  dict[id_point_str -> readable_name] as built by
                      catchment_OD_preparation._build_station_name_lookup.

    Returns:
        DataFrame[origin_id (int), dest_id (int), trips (float)].
        Intrazonal and zero-trip rows excluded.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"W3 OD CSV not found: {csv_path}. "
            f"Run catchment_OD_preparation.prepare_all_od_matrices() first."
        )

    matrix = pd.read_csv(csv_path, index_col=0, encoding='utf-8-sig')

    # Reverse name_lookup: readable_name -> id_point (int)
    reverse = {v: int(k) for k, v in name_lookup.items()}

    long = (
        matrix
        .rename_axis(index='origin_name', columns='dest_name')
        .stack()
        .reset_index(name='trips')
        .rename(columns={'origin_name': 'on', 'dest_name': 'dn'})
    )
    long['origin_id'] = long['on'].map(reverse)
    long['dest_id']   = long['dn'].map(reverse)
    long = long.dropna(subset=['origin_id', 'dest_id'])
    long['origin_id'] = long['origin_id'].astype(int)
    long['dest_id']   = long['dest_id'].astype(int)
    long = long[long['origin_id'] != long['dest_id']]
    long = long[long['trips'] > 0][['origin_id', 'dest_id', 'trips']].copy()
    print(f"    W3 OD loaded: {len(long):,} non-zero inter-station pairs")
    return long


# ===============================================================================
# SERVICE TABLE
# ===============================================================================

def _build_service_table(rail_segments: pd.DataFrame,
                         rail_lines: pd.DataFrame,
                         rail_stations: gpd.GeoDataFrame) -> dict:
    """Build (id_point_A, id_point_C) -> list[(variant_key, IVT_min, freq_per_h)].

    For every rail variant, reconstructs the ordered stop sequence, computes
    cumulative IVT from each stop to every downstream stop, then filters to
    pairs where both stops are in-catchment rail stations.

    Args:
        rail_segments: From _load_rail_segments_with_tt(); includes travel_time_min.
        rail_lines:    From catchment_allocate._load_rail_line_freqs().
        rail_stations: In-catchment rail stations GDF with stop_id and id_point.

    Returns:
        dict[(int, int)] -> list[(str, float, float)]
            key:   (origin id_point, dest id_point)
            value: list of (variant_key, IVT_min, freq_per_h_window)
    """
    # Build stop_id (str) -> id_point (int) mapping
    stn = rail_stations.copy()
    stn['stop_id_str'] = stn['stop_id'].astype(str)
    stop_to_idp = dict(zip(stn['stop_id_str'], stn['id_point'].astype(int)))
    station_set = set(stop_to_idp.keys())

    # Normalise segments
    seg = rail_segments.copy()
    seg['from_stop_id'] = seg['from_stop_id'].astype(str)
    seg['to_stop_id']   = seg['to_stop_id'].astype(str)
    seg['route_id']     = seg['route_id'].astype(str)
    seg = seg.dropna(subset=['from_stop_id', 'to_stop_id', 'route_id',
                             'direction_id', 'variant_rank', 'travel_time_min'])

    # Normalise lines
    ln = rail_lines.copy()
    ln['route_id'] = ln['route_id'].astype(str)
    ln = ln.dropna(subset=['route_id', 'direction_id', 'variant_rank',
                           'freq_per_h_window'])
    ln_freq = ln.set_index(
        ['route_id', 'direction_id', 'variant_rank'])['freq_per_h_window'].to_dict()

    service_table = {}
    n_variants_used = 0
    n_variants_skip = 0

    for (rid, did, vrnk), var_segs in seg.groupby(
            ['route_id', 'direction_id', 'variant_rank'], sort=False):
        freq = ln_freq.get((rid, did, vrnk))
        if freq is None or freq <= 0:
            n_variants_skip += 1
            continue

        seq = catchment_allocate._reconstruct_stop_sequence(var_segs)
        if len(seq) < 2:
            n_variants_skip += 1
            continue

        # Build per-segment TT lookup for this variant
        tt_map = {}
        for _, row in var_segs.iterrows():
            tt_map[(str(row['from_stop_id']), str(row['to_stop_id']))] = \
                float(row['travel_time_min'])

        n_variants_used += 1
        variant_key = f"{rid}_{did}_{vrnk}"

        # For each origin at position i, accumulate IVT to downstream stops
        for i, origin_sid in enumerate(seq):
            if origin_sid not in station_set:
                continue
            origin_idp = stop_to_idp[origin_sid]
            cumulative_tt = 0.0
            for j in range(i + 1, len(seq)):
                prev_sid = seq[j - 1]
                curr_sid = seq[j]
                seg_tt = tt_map.get((prev_sid, curr_sid), 0.0)
                cumulative_tt += seg_tt
                if curr_sid not in station_set:
                    continue
                dest_idp = stop_to_idp[curr_sid]
                key = (origin_idp, dest_idp)
                entry = (variant_key, cumulative_tt, freq)
                service_table.setdefault(key, []).append(entry)

    print(f"  Service table: {len(service_table):,} direct (A,C) station pairs "
          f"from {n_variants_used} variants ({n_variants_skip} skipped).")
    if service_table:
        all_ivts = [e[1] for entries in service_table.values() for e in entries]
        arr = np.array(all_ivts)
        print(f"    IVT min/median/max: "
              f"{arr.min():.1f} / {np.median(arr):.1f} / {arr.max():.1f} min")
    return service_table


def _validate_service_table(service_table: dict) -> dict:
    """Drop entries with IVT ≤ 0 or freq ≤ 0; print distribution stats."""
    clean = {}
    n_bad = 0
    for key, entries in service_table.items():
        valid = [(vk, ivt, fr) for vk, ivt, fr in entries
                 if ivt > 0 and fr > 0]
        n_bad += len(entries) - len(valid)
        if valid:
            clean[key] = valid
    if n_bad:
        print(f"  WARNING: {n_bad} service table entries dropped (IVT≤0 or freq≤0).")
    return clean


# ===============================================================================
# SHARE COMPUTATION — OPTION A (SPIESS & FLORIAN)
# ===============================================================================

def _sf_shares(services: list) -> tuple:
    """Spiess & Florian (1989) attractive-set allocation.

    Sort services by IVT ascending. Iteratively add service k if
    IVT_k < current expected cost c_S (= 30/F_S + weighted_IVT_S).
    Stop when IVT_k >= c_S — all remaining services are dominated.

    Args:
        services: list of (variant_key, IVT_min, freq_per_h).

    Returns:
        (shares_dict, expected_cost_min)
        shares_dict: dict[variant_key -> share], sums to 1.0 over attractive set.
        expected_cost_min: expected travel time under the optimal strategy (min).
    """
    if not services:
        return {}, float('inf')

    sorted_svcs = sorted(services, key=lambda x: x[1])

    F_S       = 0.0
    IVT_S_wtd = 0.0
    c_S       = float('inf')
    attractive = []

    for variant_key, IVT_k, freq_k in sorted_svcs:
        if freq_k <= 0:
            continue
        if IVT_k < c_S:
            prev_F   = F_S
            F_S     += freq_k
            IVT_S_wtd = (IVT_S_wtd * prev_F + freq_k * IVT_k) / F_S
            c_S      = 30.0 / F_S + IVT_S_wtd
            attractive.append((variant_key, freq_k))
        else:
            break

    if not attractive or F_S <= 0:
        return {}, float('inf')

    shares = {vk: fk / F_S for vk, fk in attractive}
    return shares, c_S


# ===============================================================================
# SHARE COMPUTATION — OPTION B (LOGIT)
# ===============================================================================

def _logit_shares(services: list,
                  theta: float = cp.LOGIT_ROUTE_THETA) -> tuple:
    """Logit route-choice allocation over parallel rail services.

    Per-service GC:
        wait_v  = cp.t_wait_min(60.0 / freq_v)   [individual headway]
        GC_v    = cp.W_IVT * IVT_v + cp.W_WAIT * wait_v
    Uses log-sum-exp shift for numerical stability.

    Args:
        services: list of (variant_key, IVT_min, freq_per_h).
        theta:    logit scale parameter (1/GC-minute), default cp.LOGIT_ROUTE_THETA.

    Returns:
        (shares_dict, expected_gc_min)
        shares_dict: dict[variant_key -> share], sums to 1.0.
        expected_gc_min: Σ share_v × GC_v.
    """
    if not services:
        return {}, float('inf')

    keys, GCs = [], []
    for variant_key, IVT_v, freq_v in services:
        if freq_v <= 0:
            continue
        wait_v = cp.t_wait_min(60.0 / freq_v)
        gc_v   = cp.W_IVT * IVT_v + cp.W_WAIT * wait_v
        keys.append(variant_key)
        GCs.append(gc_v)

    if not keys:
        return {}, float('inf')

    GCs_arr    = np.array(GCs)
    U          = -theta * GCs_arr
    U         -= U.max()            # log-sum-exp shift
    exp_U      = np.exp(U)
    shares_arr = exp_U / exp_U.sum()
    exp_gc     = float((shares_arr * GCs_arr).sum())

    shares = dict(zip(keys, shares_arr.tolist()))
    return shares, exp_gc


# ===============================================================================
# 1-TRANSFER ROUTING
# ===============================================================================

def _find_transfer_stations(A: int, C: int, service_table: dict) -> list:
    """Return all T where (A,T) and (T,C) both have direct service."""
    a_dests = {dc for (ao, dc) in service_table if ao == A and dc != C}
    c_origs = {ao for (ao, dc) in service_table if dc == C and ao != A}
    return sorted(a_dests & c_origs)


def _path_cost_sf(A: int, T: int, C: int,
                  service_table: dict) -> tuple:
    """Expected cost and binding frequency for A→T→C under S&F.

    Returns:
        (path_cost_min, binding_freq_per_h, c1, c2)
        c1: expected cost of leg 1 (A→T)
        c2: expected cost of leg 2 wait + IVT (T→C)
        binding_freq: min of attractive-set frequencies of the two legs.
    """
    _, c1    = _sf_shares(service_table.get((A, T), []))
    sh2, c2  = _sf_shares(service_table.get((T, C), []))
    if c1 == float('inf') or c2 == float('inf'):
        return float('inf'), 0.0, c1, c2

    # Binding frequency = min of total attractive-set freq of each leg
    F1 = sum(service_table[(A, T)][0][2]   # rough proxy: use first entry freq
             for _ in [None]) if (A, T) in service_table else 0.0
    # More accurately: F_attractive for each leg from _sf_shares internals.
    # Since _sf_shares doesn't expose F_S, re-derive as 30/(c_S - IVT_S_wtd).
    # Simpler: sum frequencies of all entries in attractive set (share * total).
    F1 = sum(fr for vk, ivt, fr in service_table.get((A, T), [])
             if vk in _sf_shares(service_table.get((A, T), []))[0])
    F2 = sum(fr for vk, ivt, fr in service_table.get((T, C), [])
             if vk in sh2)
    binding_freq = min(F1, F2) if F1 > 0 and F2 > 0 else 0.0
    path_cost = c1 + TRANSFER_PENALTY_MIN + c2
    return path_cost, binding_freq, c1, c2


def _path_cost_logit(A: int, T: int, C: int, service_table: dict,
                     theta: float = cp.LOGIT_ROUTE_THETA) -> tuple:
    """Expected GC and total frequency for A→T→C under Logit.

    Returns:
        (path_gc_min, total_freq_per_h)
    """
    _, gc1 = _logit_shares(service_table.get((A, T), []), theta)
    _, gc2 = _logit_shares(service_table.get((T, C), []), theta)
    if gc1 == float('inf') or gc2 == float('inf'):
        return float('inf'), 0.0
    total_freq = sum(fr for _, _, fr in service_table.get((A, T), []))
    path_gc = gc1 + TRANSFER_PENALTY_MIN + gc2
    return path_gc, total_freq


def _resolve_transfer_pair(A: int, C: int, method: str,
                           service_table: dict,
                           theta: float = cp.LOGIT_ROUTE_THETA) -> tuple:
    """Distribute demand for A→C via 1-transfer paths.

    Returns:
        (T_shares_dict, expected_path_cost)
        T_shares_dict: dict[T_id (int) -> share_of_total_AC_demand]
                       Empty dict if no valid T exists.
        expected_path_cost: float, inf if unresolvable.
    """
    T_list = _find_transfer_stations(A, C, service_table)
    if not T_list:
        return {}, float('inf')

    if method == 'sf':
        # Build pseudo-service list: each T is treated as a "service" with
        # IVT = path_cost and freq = binding_freq, then apply S&F.
        pseudo = []
        path_costs = {}
        for T in T_list:
            pc, bf, _, _ = _path_cost_sf(A, T, C, service_table)
            if pc < float('inf') and bf > 0:
                pseudo.append((str(T), pc, bf))
                path_costs[T] = pc
        if not pseudo:
            return {}, float('inf')
        shares_raw, exp_cost = _sf_shares(pseudo)
        T_shares = {int(k): v for k, v in shares_raw.items()}
        return T_shares, exp_cost

    else:  # logit
        pseudo = []
        path_costs = {}
        for T in T_list:
            pc, tf = _path_cost_logit(A, T, C, service_table, theta)
            if pc < float('inf') and tf > 0:
                pseudo.append((str(T), pc, tf))
                path_costs[T] = pc
        if not pseudo:
            return {}, float('inf')
        shares_raw, exp_cost = _logit_shares(pseudo, theta)
        T_shares = {int(k): v for k, v in shares_raw.items()}
        return T_shares, exp_cost


# ===============================================================================
# MAIN ROUTING LOOP
# ===============================================================================

def _route_single_method(od_long: pd.DataFrame,
                         service_table: dict,
                         method: str,
                         theta: float = cp.LOGIT_ROUTE_THETA) -> tuple:
    """Route all (A, C) pairs using a single distribution method.

    Args:
        od_long:       DataFrame[origin_id, dest_id, trips] from _load_w3_od.
        service_table: dict from _build_service_table.
        method:        'sf' or 'logit'.
        theta:         Logit scale parameter (only used when method='logit').

    Returns:
        (service_df, gc_df, unresolved_df)
        service_df:   DataFrame[origin_id, dest_id, transfer_station_id,
                                variant_key, leg, trips, method]
        gc_df:        DataFrame[origin_id, dest_id, expected_gc_min]
        unresolved_df: DataFrame[origin_id, dest_id, trips] for unrouted pairs.
    """
    share_fn = _sf_shares if method == 'sf' else \
               lambda svcs: _logit_shares(svcs, theta)

    svc_rows       = []
    gc_rows        = []
    unresolved_rows = []

    n_direct    = 0
    n_transfer  = 0
    n_unresolved = 0

    for _, row in od_long.iterrows():
        A, C, trips = int(row['origin_id']), int(row['dest_id']), float(row['trips'])

        if (A, C) in service_table:
            # --- Direct service ---
            shares, exp_cost = share_fn(service_table[(A, C)])
            if shares:
                for vk, sh in shares.items():
                    svc_rows.append({
                        'origin_id':          A,
                        'dest_id':            C,
                        'transfer_station_id': None,
                        'variant_key':        vk,
                        'leg':                1,
                        'trips':              trips * sh,
                        'method':             method,
                    })
                gc_rows.append({'origin_id': A, 'dest_id': C,
                                'expected_gc_min': exp_cost})
                n_direct += 1
                continue

        # --- 1-transfer routing ---
        T_shares, exp_cost = _resolve_transfer_pair(A, C, method,
                                                     service_table, theta)
        if T_shares:
            for T, t_sh in T_shares.items():
                # Leg 1: A → T
                leg1_shares, _ = share_fn(service_table.get((A, T), []))
                for vk, l1_sh in leg1_shares.items():
                    svc_rows.append({
                        'origin_id':          A,
                        'dest_id':            C,
                        'transfer_station_id': T,
                        'variant_key':        vk,
                        'leg':                1,
                        'trips':              trips * t_sh * l1_sh,
                        'method':             method,
                    })
                # Leg 2: T → C
                leg2_shares, _ = share_fn(service_table.get((T, C), []))
                for vk, l2_sh in leg2_shares.items():
                    svc_rows.append({
                        'origin_id':          A,
                        'dest_id':            C,
                        'transfer_station_id': T,
                        'variant_key':        vk,
                        'leg':                2,
                        'trips':              trips * t_sh * l2_sh,
                        'method':             method,
                    })
            gc_rows.append({'origin_id': A, 'dest_id': C,
                            'expected_gc_min': exp_cost})
            n_transfer += 1
        else:
            unresolved_rows.append({'origin_id': A, 'dest_id': C,
                                    'trips': trips})
            n_unresolved += 1

    service_df    = pd.DataFrame(svc_rows)
    gc_df         = pd.DataFrame(gc_rows)
    unresolved_df = pd.DataFrame(unresolved_rows)

    total = n_direct + n_transfer + n_unresolved
    print(f"\n  [{method.upper()}] Routing summary ({total} pairs):")
    print(f"    Direct service:        {n_direct:>5}  ({100*n_direct/max(total,1):4.1f}%)")
    print(f"    1-transfer:            {n_transfer:>5}  ({100*n_transfer/max(total,1):4.1f}%)")
    print(f"    Unresolved (flagged):  {n_unresolved:>5}  ({100*n_unresolved/max(total,1):4.1f}%)")

    if not unresolved_df.empty:
        top5 = unresolved_df.nlargest(5, 'trips')
        print(f"    Top unresolved pairs by trips:")
        for _, ur in top5.iterrows():
            print(f"      ({int(ur['origin_id'])}, {int(ur['dest_id'])}) "
                  f"= {ur['trips']:,.1f} trips/day")

    return service_df, gc_df, unresolved_df


# ===============================================================================
# OUTPUT HELPERS
# ===============================================================================

def _apply_tau_to_service(service_df: pd.DataFrame, tau: float) -> pd.DataFrame:
    """Scale per-service trip counts by tau; drop zero rows."""
    if service_df is None or service_df.empty:
        return service_df
    out = service_df.copy()
    out['trips'] = out['trips'] * tau
    return out[out['trips'] > 0].copy()


def _write_service_csv(service_df: pd.DataFrame, name_lookup: dict,
                       output_path: str, label: str = '') -> None:
    """Write per-service assignment CSV with human-readable station names."""
    if service_df is None or service_df.empty:
        print(f"    ({label}): no data — skipping {output_path}")
        return
    out = service_df.copy()
    out['origin_name']   = out['origin_id'].astype(str).map(name_lookup)
    out['dest_name']     = out['dest_id'].astype(str).map(name_lookup)
    out['transfer_name'] = (out['transfer_station_id']
                            .dropna().astype(int).astype(str).map(name_lookup))
    cols = ['origin_id', 'origin_name', 'dest_id', 'dest_name',
            'transfer_station_id', 'transfer_name',
            'variant_key', 'leg', 'trips', 'method']
    cols = [c for c in cols if c in out.columns]
    out_full = Path(output_path)
    out_full.parent.mkdir(parents=True, exist_ok=True)
    out[cols].to_csv(out_full, index=False, encoding='utf-8-sig')
    print(f"    Saved -> {out_full}")
    print(f"      {label}: {len(out):,} service rows, "
          f"total trips = {out['trips'].sum():,.1f}")


def _write_gc_matrix(gc_df: pd.DataFrame, name_lookup: dict,
                     output_path: str) -> None:
    """Pivot GC long-format to wide station×station matrix; write CSV."""
    if gc_df is None or gc_df.empty:
        print(f"    No GC data — skipping {output_path}")
        return
    matrix = gc_df.pivot_table(
        index='origin_id', columns='dest_id',
        values='expected_gc_min', aggfunc='mean', fill_value=np.nan,
    )
    str_idx        = matrix.index.astype(str).tolist()
    str_cols       = matrix.columns.astype(str).tolist()
    matrix.index   = [name_lookup.get(s, s) for s in str_idx]
    matrix.columns = [name_lookup.get(s, s) for s in str_cols]
    matrix.index.name   = 'origin'
    matrix.columns.name = 'destination'
    out_full = Path(output_path)
    out_full.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out_full, encoding='utf-8-sig')
    print(f"    Saved GC matrix -> {out_full}  "
          f"({matrix.shape[0]}×{matrix.shape[1]} stations)")


# ===============================================================================
# PUBLIC ENTRY POINT
# ===============================================================================

def route_od_matrices(svc_version: str = '',
                      use_cache:   bool = False,
                      od_method:   str  = 'pt_feeder') -> None:
    """Route W3 station-pair OD through the rail network (S&F and Logit).

    Args:
        svc_version: Service version folder name, e.g. 'SVC2026_ZH_S18_network'.
                     Required when running standalone; inferred when called
                     after catchment_allocate.get_catchment().
        use_cache:   If True, skip writing CSVs that already exist.
        od_method:   Which W3 OD to consume — 'pt_feeder' or 'municipal'.

    Writes (under data/Traffic_Flow/OD/<svc_network>/Routing/):
        sf_service_assignment_{window}.csv
        logit_service_assignment_{window}.csv
        gc_matrix_sf.csv
        gc_matrix_logit.csv
    """
    if svc_version:
        catchment_base.setup_versioned_dirs(svc_version)
        catchment_allocate._RAIL_BASE = os.path.join(
            paths.RAIL_LINES_DIR, svc_version, paths.SERVICES_UNPROJECTED_SUBDIR)

    os.chdir(paths.MAIN)
    print("=" * 70)
    print("RAIL NETWORK OD ROUTING (W4b)")
    print(f"  Service version : {svc_version or '(inherited)'}")
    print(f"  W3 OD method    : {od_method}")
    print(f"  S&F + Logit (θ={cp.LOGIT_ROUTE_THETA}), ≤1 transfer")
    print("=" * 70)

    # --- Load shared data ---
    boundary      = catchment_base._load_catchment_boundary()
    rail_stations = catchment_allocate._load_rail_stations(boundary, 'all', buffer=0)

    import catchment_OD_preparation as cod
    name_lookup = cod._build_station_name_lookup(rail_stations)

    # W3 full-day OD path (versioned)
    if not svc_version:
        raise ValueError("route_od_matrices requires svc_version to locate the "
                         "versioned W3 OD matrix under data/Traffic_Flow/OD/.")
    od_allday_path = paths.get_station_od_csv(svc_version, od_method, 'full_day')

    print(f"\n[Step 1] Loading W3 full-day OD ({od_method}) ...")
    od_long = _load_w3_od(od_allday_path, rail_stations, name_lookup)

    # --- Build service table ---
    print("\n[Step 2] Building rail service table ...")
    rail_segs_tt = _load_rail_segments_with_tt()
    rail_lines   = catchment_allocate._load_rail_line_freqs()
    service_table = _build_service_table(rail_segs_tt, rail_lines, rail_stations)
    service_table = _validate_service_table(service_table)

    # --- Route: S&F ---
    print("\n[Step 3A] Routing — Spiess & Florian ...")
    sf_svc_df, sf_gc_df, sf_unresolved = _route_single_method(
        od_long, service_table, method='sf')

    # --- Route: Logit ---
    print("\n[Step 3B] Routing — Logit ...")
    logit_svc_df, logit_gc_df, logit_unresolved = _route_single_method(
        od_long, service_table, method='logit',
        theta=cp.LOGIT_ROUTE_THETA)

    # --- Write outputs ---
    print("\n[Step 4] Writing outputs ...")

    routing_dir = paths.get_od_routing_dir(svc_version)
    os.makedirs(routing_dir, exist_ok=True)

    windows = [
        (cp.TAU_PEAK_SHARE,     'peak'),
        (cp.TAU_OFFPEAK_SHARE,  'off_peak'),
        (cp.TAU_FULL_DAY_SHARE, 'full_day'),
    ]

    for tau, label in windows:
        sf_scaled    = _apply_tau_to_service(sf_svc_df,    tau)
        logit_scaled = _apply_tau_to_service(logit_svc_df, tau)
        sf_path    = os.path.join(routing_dir, f'sf_service_assignment_{label}.csv')
        logit_path = os.path.join(routing_dir, f'logit_service_assignment_{label}.csv')

        if use_cache and Path(sf_path).exists():
            print(f"    cached: {sf_path}")
        else:
            _write_service_csv(sf_scaled, name_lookup, sf_path,
                               label=f'S&F {label}')

        if use_cache and Path(logit_path).exists():
            print(f"    cached: {logit_path}")
        else:
            _write_service_csv(logit_scaled, name_lookup, logit_path,
                               label=f'Logit {label}')

    # GC matrices (time-independent)
    sf_gc_path    = os.path.join(routing_dir, 'gc_matrix_sf.csv')
    logit_gc_path = os.path.join(routing_dir, 'gc_matrix_logit.csv')
    if use_cache and Path(sf_gc_path).exists():
        print(f"    cached: {sf_gc_path}")
    else:
        _write_gc_matrix(sf_gc_df, name_lookup, sf_gc_path)

    if use_cache and Path(logit_gc_path).exists():
        print(f"    cached: {logit_gc_path}")
    else:
        _write_gc_matrix(logit_gc_df, name_lookup, logit_gc_path)

    print("\n=== W4b rail routing done ===")


# ===============================================================================
# STANDALONE ENTRY POINT
# ===============================================================================

if __name__ == '__main__':
    os.chdir(paths.MAIN)
    import settings

    _feeder_root = os.path.join(paths.MAIN, paths.FEEDER_LINES_DIR)
    _svc_versions = sorted([
        d for d in os.listdir(_feeder_root)
        if os.path.isdir(os.path.join(_feeder_root, d))
        and os.path.exists(os.path.join(
            _feeder_root, d, paths.SERVICES_UNPROJECTED_SUBDIR,
            'pt_feeder_stops.gpkg'))
    ]) if os.path.isdir(_feeder_root) else []

    _svc = ''
    if not _svc_versions:
        print("WARNING: no service versions found — _RAIL_BASE will be empty.")
    elif len(_svc_versions) == 1:
        _svc = _svc_versions[0]
        print(f"Service version: {_svc}")
    else:
        print("Available service versions:")
        for _i, _sv in enumerate(_svc_versions, 1):
            print(f"  {_i}) {_sv}")
        while True:
            _raw = input("Select service version [1]: ").strip() or '1'
            if _raw.isdigit() and 1 <= int(_raw) <= len(_svc_versions):
                _svc = _svc_versions[int(_raw) - 1]
                break
            print(f"  Invalid — enter 1-{len(_svc_versions)}.")

    print("\nAvailable OD methods:  1) pt_feeder   2) municipal")
    while True:
        _m = input("Select OD method [1]: ").strip() or '1'
        if _m in ('1', 'pt_feeder'):
            _od_method = 'pt_feeder'
            break
        if _m in ('2', 'municipal'):
            _od_method = 'municipal'
            break
        print("  Enter 1 or 2.")

    route_od_matrices(svc_version=_svc,
                      use_cache=getattr(settings, 'use_cache_railRouting', False),
                      od_method=_od_method)
