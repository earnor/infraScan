"""Demand-growth scenario engine: Phase 7 factor store + legacy full-OD pickles.
Last modified: 2026-06-10

Phase 7 (main_new) builds a factor store instead of materialised ODs: baseline
per-station growth-factor vectors (scenario x year) weighted by the allocated
pop_in_station from the 4A/6A breakdown, with gateway and weightless stations
on the Swiss national (CH) trajectory, plus station-independent modal-split /
distance-per-person scalars. Scenario ODs are composed on demand
(compose_scenario_od). The LHS draws are seeded (42/43), so baseline and every
svc-int see identical stochastic paths.

The legacy path (get_random_scenarios -> full-OD pickles under
RANDOM_SCENARIO_CACHE_PATH, OD_STATIONS_* / COMMUNE_TO_STATION_PATH inputs) is
kept untouched below for main.py / main_cap.py.
"""
import json
import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.ticker import EngFormatter
from scipy.stats import norm, qmc
from tqdm import tqdm

import cache_manifest
import catchment_base
import paths
import settings

_CH_DISTRICT = '__CH__'
_DISTRICT_FACTOR_MEMO: Dict[tuple, pd.DataFrame] = {}
_MODAL_DISTANCE_MEMO: Dict[tuple, pd.DataFrame] = {}


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7 factor store (main_new path)
# ═══════════════════════════════════════════════════════════════════════════

def build_scenario_factor_store(svc_version: str, base_infra: str, method: str,
                                attribution: str, n_scenarios: int,
                                start_year: int, end_year: int,
                                make_plots: bool = False,
                                use_cache: bool = False) -> dict:
    """Build + persist the baseline Phase-7 factor store for a svc network.

    Writes growth_factors.parquet (scenario, year, station_id, factor) and
    modal_distance_factors.csv (scenario, year, modal_factor, distance_factor)
    under data/Scenario/<svc_version>_network/. Station universe = the 4B long
    OD's origins + destinations; weights = allocated pop_in_station from the
    4A breakdown; gateway and weightless stations follow the CH trajectory.

    Args:
        svc_version: service version WITHOUT the '_network' suffix.
        base_infra:  infrastructure version (gateway JSON lookup).
        method:      'pt_feeder' | 'municipal'.
        attribution: long-OD file key ('specific' | 'blended' | 'municipal').
        n_scenarios: number of LHS scenarios (seeded 42/43 — deterministic).
        start_year:  base year (factor 1.0) and trajectory origin.
        end_year:    last scenario year.
        make_plots:  population/modal/distance fan plots into plots/scenarios.
        use_cache:   skip the build when both outputs already exist.

    Returns:
        Summary dict: output paths, station classification, 'cached' flag.
    """
    svc_network = f'{svc_version}_network'
    parquet_path = paths.get_growth_factors_parquet(svc_network, method)
    md_path = paths.get_modal_distance_factors_csv(svc_network)
    if (use_cache and os.path.exists(parquet_path) and os.path.exists(md_path)
            and cache_manifest.check_manifest(
                os.path.dirname(parquet_path), 'scenarios_7',
                {'svc_network': svc_network, 'infra_version': base_infra})):
        print(f"  use_cache_scenarios: baseline factor store present — "
              f"skipping build\n    {parquet_path}")
        return {'factors_path': parquet_path, 'modal_distance_path': md_path,
                'cached': True}

    print(f"\n=== Building baseline factor store ({svc_network}, {method}, "
          f"{n_scenarios} scenarios, {start_year}-{end_year}) ===")
    od_long = pd.read_csv(
        paths.get_station_od_long_csv(svc_network, method, attribution))
    station_ids = _od_station_universe(od_long)
    gateway_ids = _load_gateway_ids(svc_network, base_infra)

    district_factors = _district_factor_table(n_scenarios, start_year, end_year)
    bd = _read_breakdown_weights(
        paths.get_station_commune_breakdown_csv(svc_network, method))
    W, classification = _station_weight_matrix(
        bd, station_ids, gateway_ids,
        set(district_factors['district'].unique()))
    factors = _station_factor_frame(W, district_factors, start_year)
    modal_distance = _modal_distance_table(n_scenarios, start_year, end_year)

    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    factors.to_parquet(parquet_path, index=False)
    modal_distance.to_csv(md_path, index=False, encoding='utf-8-sig')
    cache_manifest.write_manifest(os.path.dirname(parquet_path), 'scenarios_7',
                                  {'svc_network': svc_network,
                                   'infra_version': base_infra})

    print(f"  Stations: {len(station_ids)} in OD — "
          f"{len(classification['weighted'])} pop-weighted, "
          f"{len(classification['gateway'])} gateway (CH trajectory), "
          f"{len(classification['fallback'])} CH-fallback")
    print(f"  Factors : {factors['scenario'].nunique()} scenarios x "
          f"{factors['year'].nunique()} years x {len(station_ids)} stations "
          f"-> {len(factors):,} rows")
    print(f"    {parquet_path}\n    {md_path}")

    if make_plots:
        _plot_factor_store_inputs(n_scenarios, start_year, end_year)

    return {'factors_path': parquet_path, 'modal_distance_path': md_path,
            'classification': classification, 'cached': False}


def build_svc_int_factor_overrides(svc_int_id: str, svc_version: str,
                                   base_infra: str, method: str,
                                   attribution: str, n_scenarios: int,
                                   start_year: int, end_year: int,
                                   use_cache: bool = False) -> dict:
    """Build the per-svc-int growth-factor override table (PT_Feeder path).

    Recomputes factors from the 6A developed breakdown for exactly the
    stations whose {commune: pop_in_station} rows changed vs the baseline
    breakdown, plus stations new to the 6B dev OD that the baseline store
    does not cover (CH fallback, warned). Gateways never enter — their growth
    is weight-free (CH trajectory) and 6B gateway-split changes alter trips,
    not growth factors. Writes growth_factor_overrides.csv (same columns as
    the baseline parquet); an empty changed set writes no file (compose then
    falls through to the baseline vectors).

    Args:
        svc_int_id:  svc-int id (e.g. 'ext_100001'), WITHOUT '_network'.
        svc_version: baseline service version WITHOUT '_network'.
        (remaining args as build_scenario_factor_store)

    Returns:
        Summary dict: 'overrides_path' (None if nothing changed),
        'changed_stations', 'new_stations', 'cached' flag.
    """
    int_network = f'{svc_int_id}_network'
    svc_network = f'{svc_version}_network'
    out_csv = paths.get_growth_factor_overrides_csv(int_network, method)
    if (use_cache and os.path.exists(out_csv)
            and cache_manifest.check_manifest(
                os.path.dirname(out_csv), 'scenarios_7',
                {'svc_network': int_network, 'infra_version': base_infra,
                 'svc_int_id': svc_int_id})):
        print(f"  [{svc_int_id}] use_cache_scenarios: override table present "
              f"— skipping\n    {out_csv}")
        return {'overrides_path': out_csv, 'cached': True}

    print(f"\n--- Factor overrides [{svc_int_id}] ({method}) ---")
    dev_bd_csv = paths.get_station_commune_breakdown_csv(int_network, method)
    if not os.path.exists(dev_bd_csv):
        print(f"  No 6A breakdown at {dev_bd_csv} — no overrides "
              f"(allocation unchanged or 6A not run).")
        return {'overrides_path': None, 'changed_stations': [],
                'new_stations': [], 'cached': False}
    base_bd = _read_breakdown_weights(
        paths.get_station_commune_breakdown_csv(svc_network, method))
    dev_bd = _read_breakdown_weights(dev_bd_csv)
    changed = _changed_stations(base_bd, dev_bd)

    baseline_parquet = paths.get_growth_factors_parquet(svc_network, method)
    if not os.path.exists(baseline_parquet):
        raise FileNotFoundError(
            f"Baseline factor store missing at {baseline_parquet} — run "
            f"build_scenario_factor_store first.")
    base_station_ids = set(pd.read_parquet(
        baseline_parquet, columns=['station_id'])['station_id'].unique())
    dev_od = pd.read_csv(
        paths.get_station_od_long_csv(int_network, method, attribution))
    new_stations = sorted(set(_od_station_universe(dev_od))
                          - base_station_ids - set(changed))
    if new_stations:
        print(f"  {len(new_stations)} station(s) new to the dev OD without "
              f"baseline factors: {new_stations}")

    override_ids = sorted(set(changed) | set(new_stations))
    print(f"  Changed allocation: {len(changed)} station(s): {changed}")
    if not override_ids:
        if os.path.exists(out_csv):
            os.remove(out_csv)
            print(f"  Removed stale override file {out_csv}")
        print(f"  [{svc_int_id}] breakdown identical to baseline — no "
              f"override file (compose uses baseline factors).")
        return {'overrides_path': None, 'changed_stations': [],
                'new_stations': [], 'cached': False}

    gateway_ids = _load_gateway_ids(svc_network, base_infra)
    district_factors = _district_factor_table(n_scenarios, start_year, end_year)
    W, _ = _station_weight_matrix(
        dev_bd, override_ids, gateway_ids,
        set(district_factors['district'].unique()))
    overrides = _station_factor_frame(W, district_factors, start_year)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    overrides.to_csv(out_csv, index=False, encoding='utf-8-sig')
    cache_manifest.write_manifest(os.path.dirname(out_csv), 'scenarios_7',
                                  {'svc_network': int_network,
                                   'infra_version': base_infra,
                                   'svc_int_id': svc_int_id})
    print(f"  Overrides: {len(override_ids)} station(s) x "
          f"{overrides['scenario'].nunique()} scenarios x "
          f"{overrides['year'].nunique()} years -> {len(overrides):,} rows\n"
          f"    {out_csv}")
    return {'overrides_path': out_csv, 'changed_stations': changed,
            'new_stations': new_stations, 'cached': False}


def load_factor_store(svc_version: str, method: str, attribution: str,
                      svc_int_id: str = None) -> dict:
    """Load the factor store + matching long OD for (baseline | one svc-int).

    Returns:
        {'factors': DataFrame(scenario, year, station_id, factor) — baseline
            vectors with the svc-int overrides applied (rows replaced/appended),
         'modal_distance': DataFrame(scenario, year, modal_factor,
            distance_factor),
         'od_long': DataFrame(origin_station_id, dest_station_id, trips) —
            the 6B dev OD for a svc-int (PT_Feeder), else the 4B baseline}.

    Under Municipal the dev OD and factors equal the baseline by construction
    (6A/6B do not run — allocation is frequency-blind), so svc_int_id is
    ignored there.
    """
    svc_network = f'{svc_version}_network'
    factors = pd.read_parquet(
        paths.get_growth_factors_parquet(svc_network, method))
    modal_distance = pd.read_csv(
        paths.get_modal_distance_factors_csv(svc_network))
    od_path = paths.get_station_od_long_csv(svc_network, method, attribution)

    if svc_int_id is not None and method != 'municipal':
        int_network = f'{svc_int_id}_network'
        override_csv = paths.get_growth_factor_overrides_csv(int_network,
                                                             method)
        if os.path.exists(override_csv):
            overrides = pd.read_csv(override_csv)
            replaced = set(overrides['station_id'].unique())
            factors = pd.concat(
                [factors[~factors['station_id'].isin(replaced)], overrides],
                ignore_index=True)
        dev_od_path = paths.get_station_od_long_csv(int_network, method,
                                                    attribution)
        if os.path.exists(dev_od_path):
            od_path = dev_od_path
        else:
            print(f"  WARNING: no 6B OD for {svc_int_id} at {dev_od_path} — "
                  f"composing on the baseline OD.")
    od_long = pd.read_csv(od_path)
    return {'factors': factors, 'modal_distance': modal_distance,
            'od_long': od_long}


def compose_scenario_od(svc_int_id, scenario: int, year: int, *,
                        svc_version: str, method: str, attribution: str,
                        store: dict = None) -> pd.DataFrame:
    """Compose the scenario OD for (svc_int_id, scenario, year) on demand.

    Long-format throughout: trips x sqrt(f_origin * f_dest) x modal x
    distance. svc_int_id=None composes the baseline. Pass a preloaded
    load_factor_store result as store for bulk loops (Phase 8A) — it is
    loaded per call otherwise.

    Returns:
        DataFrame(origin_station_id, dest_station_id, trips) — same schema
        as the input long OD.

    Raises:
        ValueError: scenario/year outside the store, or an OD station without
            a factor row (stale store — rebuild Phase 7).
    """
    if store is None:
        store = load_factor_store(svc_version, method, attribution, svc_int_id)
    f = store['factors']
    fs = f[(f['scenario'] == scenario) & (f['year'] == year)]
    if fs.empty:
        raise ValueError(f"(scenario={scenario}, year={year}) not in the "
                         f"factor store — rebuild Phase 7 with a wider range.")
    md = store['modal_distance']
    md_row = md[(md['scenario'] == scenario) & (md['year'] == year)]
    if md_row.empty:
        raise ValueError(f"(scenario={scenario}, year={year}) not in the "
                         f"modal/distance table.")
    m_factor = float(md_row['modal_factor'].iloc[0])
    d_factor = float(md_row['distance_factor'].iloc[0])

    sqrt_f = np.sqrt(fs.set_index('station_id')['factor'])
    od = store['od_long'].copy()
    sq_o = od['origin_station_id'].map(sqrt_f)
    sq_d = od['dest_station_id'].map(sqrt_f)
    missing = sorted(set(od.loc[sq_o.isna(), 'origin_station_id'])
                     | set(od.loc[sq_d.isna(), 'dest_station_id']))
    if missing:
        raise ValueError(
            f"{len(missing)} OD station(s) without factor rows (stale factor "
            f"store — rebuild Phase 7): {missing}")
    od['trips'] = od['trips'] * sq_o * sq_d * m_factor * d_factor
    return od


def validate_factor_store(svc_version: str, base_infra: str, method: str,
                          attribution: str, n_scenarios: int, start_year: int,
                          end_year: int, sample_scenarios=(1, 50, 100),
                          sample_years=(2050, 2100)) -> bool:
    """Three-check validation of the factor store against direct application.

    1. Base-year identity: compose(None, s, start_year) == input OD.
    2. Long == wide: the long-format composition equals the legacy wide-matrix
       row/col sqrt-multiplication with the same factor slice.
    3. Legacy-mechanics parity: the vectorised G @ W^T station factors equal
       compute_growth_od_matrix_optimized's per-station loop fed identical
       weights and the identical seeded district scenarios (isolates the
       reimplementation from the intentional weight/gateway re-sourcing).

    Standalone-CLI / verification use only — not called from the pipeline.
    Returns True when all checks pass; prints per-check results.
    """
    print(f"\n=== Validating factor store ({svc_version}, {method}) ===")
    store = load_factor_store(svc_version, method, attribution, None)
    ok = True
    sample_scenarios = [s for s in sample_scenarios if s <= n_scenarios]

    # -- 1. base-year identity ------------------------------------------------
    base = compose_scenario_od(None, sample_scenarios[0], start_year,
                               svc_version=svc_version, method=method,
                               attribution=attribution, store=store)
    if np.allclose(base['trips'].values, store['od_long']['trips'].values,
                   rtol=1e-12, atol=1e-12):
        print("  [1] base-year identity: PASS")
    else:
        diff = np.abs(base['trips'].values
                      - store['od_long']['trips'].values).max()
        print(f"  [1] base-year identity: FAIL (max abs diff {diff})")
        ok = False

    # -- 2. long == wide ------------------------------------------------------
    f = store['factors']
    md = store['modal_distance']
    worst = 0.0
    for s in sample_scenarios:
        for y in sample_years:
            composed = compose_scenario_od(None, s, y,
                                           svc_version=svc_version,
                                           method=method,
                                           attribution=attribution,
                                           store=store)
            wide = store['od_long'].pivot_table(index='origin_station_id',
                                                columns='dest_station_id',
                                                values='trips',
                                                aggfunc='sum', fill_value=0.0)
            fac = f[(f['scenario'] == s)
                    & (f['year'] == y)].set_index('station_id')['factor']
            sqrt_f = np.sqrt(fac)
            grown = wide.mul(sqrt_f.reindex(wide.index), axis=0) \
                        .mul(sqrt_f.reindex(wide.columns), axis=1)
            md_row = md[(md['scenario'] == s) & (md['year'] == y)]
            grown *= (float(md_row['modal_factor'].iloc[0])
                      * float(md_row['distance_factor'].iloc[0]))
            check = composed.set_index(
                ['origin_station_id', 'dest_station_id'])['trips']
            wide_vals = grown.stack().reindex(check.index)
            worst = max(worst,
                        float(np.abs(wide_vals.values
                                     - check.values).max()))
    if worst <= 1e-9 * max(1.0, float(store['od_long']['trips'].max())):
        print(f"  [2] long == wide composition: PASS (max abs diff {worst:.2e})")
    else:
        print(f"  [2] long == wide composition: FAIL (max abs diff {worst:.2e})")
        ok = False

    # -- 3. legacy-mechanics parity -------------------------------------------
    refs = get_bezirk_population_scenarios()
    population_scenarios = {
        district: generate_population_scenarios(df, start_year, end_year,
                                                n_scenarios)
        for district, df in refs.items()
    }
    communes = _build_communes_population_df(settings.start_year_scenario)
    known_bfs = communes[communes['bezirk'].isin(population_scenarios)]
    known_bfs = known_bfs[pd.to_numeric(known_bfs['anzahl'],
                                        errors='coerce') > 0]
    bfs_pop = dict(zip(known_bfs['gemeinde_bfs_nr'].astype(int),
                       known_bfs['anzahl']))
    svc_network = f'{svc_version}_network'
    pairs = _read_breakdown_weights(
        paths.get_station_commune_breakdown_csv(svc_network, method))
    pairs = pairs[pairs['BFS_NR'].isin(bfs_pop)]
    # identical weights on both sides: commune total population (legacy
    # semantics), restricted to communes with a district trajectory
    synthetic = pairs.copy()
    synthetic['pop_in_station'] = synthetic['BFS_NR'].map(bfs_pop)
    stations = sorted(synthetic['station_id'].unique())
    district_factors = _district_factor_table(n_scenarios, start_year,
                                              end_year)
    W, _ = _station_weight_matrix(synthetic, stations, set(),
                                  set(district_factors['district'].unique()))
    vec = _station_factor_frame(W, district_factors, start_year)
    station_communes = synthetic.groupby('station_id')['BFS_NR'] \
                                .apply(list).to_dict()
    dummy_od = pd.DataFrame(1.0, index=stations,
                            columns=[str(s) for s in stations])
    worst3 = 0.0
    for s in sample_scenarios:
        for y in sample_years:
            legacy = compute_growth_od_matrix_optimized(
                dummy_od, station_communes, communes, population_scenarios,
                s - 1, y, start_year)
            legacy_diag = pd.Series(
                {st: legacy.loc[st, str(st)] for st in stations})
            vec_slice = vec[(vec['scenario'] == s) & (vec['year'] == y)] \
                .set_index('station_id')['factor'].reindex(stations)
            worst3 = max(worst3, float(np.abs(vec_slice.values
                                              - legacy_diag.values).max()))
    if worst3 <= 1e-9:
        print(f"  [3] legacy-mechanics parity: PASS (max abs diff {worst3:.2e})")
    else:
        print(f"  [3] legacy-mechanics parity: FAIL (max abs diff {worst3:.2e})")
        ok = False

    print(f"  => {'ALL CHECKS PASSED' if ok else 'VALIDATION FAILED'}")
    return ok


def _changed_stations(base_bd: pd.DataFrame, dev_bd: pd.DataFrame) -> list:
    """Stations whose (BFS_NR, pop_in_station) weight rows differ between the
    baseline and developed breakdowns (incl. stations only in one of them)."""
    merged = base_bd.merge(dev_bd, on=['station_id', 'BFS_NR'], how='outer',
                           suffixes=('_base', '_dev'), indicator=True)
    diff = merged[(merged['_merge'] != 'both')
                  | (merged['pop_in_station_base']
                     != merged['pop_in_station_dev'])]
    return sorted(diff['station_id'].unique())


def _od_station_universe(od_long: pd.DataFrame) -> list:
    """Sorted unique station ids appearing as origin or destination."""
    orig = pd.to_numeric(od_long['origin_station_id'], errors='coerce')
    dest = pd.to_numeric(od_long['dest_station_id'], errors='coerce')
    ids = set(orig.dropna().astype(int)) | set(dest.dropna().astype(int))
    return sorted(ids)


def _load_gateway_ids(svc_network: str, infra_version: str) -> set:
    """Gateway (boundary) station node ids from the Phase-3B JSON.

    Empty set with a warning if the file is absent — gateways then land in the
    CH-fallback class (same trajectory, louder print)."""
    path = paths.get_boundary_stations_json(svc_network, infra_version)
    if not os.path.exists(path):
        print(f"  WARNING: boundary stations file missing at {path} — "
              f"no gateway classification.")
        return set()
    with open(path, encoding='utf-8') as f:
        ids = json.load(f)
    return {int(x) for x in ids}


def _get_ch_population_reference(start_year: int, end_year: int) -> pd.DataFrame:
    """CH national reference trajectory (jahr, total_population, growth_rate).

    BFS observations + Referenzszenario A-00-2025 up to 2050, then Eurostat
    national growth rates 2051-2100 compounded — *unscaled*, unlike the
    districts, which get the Eurostat rates scaled relative to CH."""
    df_ch = pd.read_csv(paths.POPULATION_SCENARIO_CH_BFS_2055, sep=",")
    pop = pd.to_numeric(df_ch['Beobachtungen'], errors='coerce').combine_first(
        pd.to_numeric(df_ch['Referenzszenario A-00-2025'], errors='coerce'))
    base = pd.DataFrame({
        'jahr': pd.to_numeric(df_ch['Jahr'], errors='coerce'),
        'total_population': pop,
    }).dropna()
    base['jahr'] = base['jahr'].astype(int)
    base = base[base['jahr'] <= 2050].sort_values('jahr').reset_index(drop=True)

    eurostat_df = pd.read_excel(paths.POPULATION_SCENARIO_CH_EUROSTAT_2100)
    eurostat_df.columns = eurostat_df.columns.map(str)
    rate_row = eurostat_df[eurostat_df['unit'] == 'GROWTH_RATE']
    rates = rate_row[[str(y) for y in range(2051, 2101)]].iloc[0].astype(float)

    current = base['total_population'].iloc[-1]
    rows = []
    for year in range(2051, 2101):
        current *= (1 + rates[str(year)])
        rows.append({'jahr': year, 'total_population': current})
    ref_df = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)
    ref_df['growth_rate'] = ref_df['total_population'].pct_change().fillna(0.0)
    return ref_df


def _district_factor_table(n_scenarios: int, start_year: int,
                           end_year: int) -> pd.DataFrame:
    """Long (district, scenario, year, factor) growth-factor table.

    Districts from get_bezirk_population_scenarios plus the '__CH__'
    pseudo-district (gateways/fallbacks); factor = pop(s, y) / pop(s,
    start_year); scenario ids 1-based (legacy 'scenario_<N>' convention).
    Memoised — the LHS draws are seeded, so the table is deterministic for
    given (n_scenarios, start_year, end_year)."""
    key = (n_scenarios, start_year, end_year)
    if key in _DISTRICT_FACTOR_MEMO:
        return _DISTRICT_FACTOR_MEMO[key]
    refs = get_bezirk_population_scenarios()
    refs[_CH_DISTRICT] = _get_ch_population_reference(start_year, end_year)
    frames = []
    for district, ref in refs.items():
        scen = generate_population_scenarios(ref, start_year, end_year,
                                             n_scenarios)
        wide = scen.pivot(index='scenario', columns='year', values='population')
        fac = wide.div(wide[start_year].replace(0, np.nan), axis=0).fillna(1.0)
        long = (fac.reset_index()
                .melt(id_vars='scenario', var_name='year', value_name='factor'))
        long['district'] = district
        frames.append(long)
    out = pd.concat(frames, ignore_index=True)
    out['scenario'] = out['scenario'].astype(int) + 1
    out['year'] = out['year'].astype(int)
    out = out[['district', 'scenario', 'year', 'factor']]
    _DISTRICT_FACTOR_MEMO[key] = out
    return out


def _read_breakdown_weights(breakdown_csv: str) -> pd.DataFrame:
    """Cleaned (station_id, BFS_NR, pop_in_station) weight rows from a 4A/6A
    station-commune breakdown: NO_PT sentinel (-1) and zero-pop rows dropped."""
    bd = pd.read_csv(breakdown_csv, encoding='utf-8-sig')
    bd['station_id'] = pd.to_numeric(bd['station_number'], errors='coerce')
    bd['BFS_NR'] = pd.to_numeric(bd['BFS_NR'], errors='coerce')
    bd['pop_in_station'] = pd.to_numeric(bd['pop_in_station'],
                                         errors='coerce').fillna(0.0)
    bd = bd.dropna(subset=['station_id', 'BFS_NR'])
    bd['station_id'] = bd['station_id'].astype(int)
    bd['BFS_NR'] = bd['BFS_NR'].astype(int)
    bd = bd[(bd['station_id'] > 0) & (bd['pop_in_station'] > 0)]
    return bd[['station_id', 'BFS_NR', 'pop_in_station']].reset_index(drop=True)


def _station_weight_matrix(bd: pd.DataFrame, station_ids: list,
                           gateway_ids: set, valid_districts: set) -> tuple:
    """Row-normalised station x district weight matrix from pop_in_station.

    Every station in station_ids gets a row: breakdown-weighted stations carry
    their allocated population summed per Bezirk; gateways (no ZH communes by
    design) and weightless stations carry full weight on the CH pseudo-district
    — gateways silently, others with a printed warning. Breakdown rows whose
    Bezirk has no trajectory are reassigned to CH with a printed count.

    Args:
        bd: cleaned weight rows from _read_breakdown_weights.

    Returns:
        (W, classification): W indexed by station_id, columns = districts;
        classification dict with 'weighted'/'gateway'/'fallback' id lists.
    """
    bd = bd[bd['station_id'].isin(set(station_ids))].copy()
    communes = _build_communes_population_df(settings.start_year_scenario)
    bfs_to_bezirk = dict(zip(communes['gemeinde_bfs_nr'].astype(int),
                             communes['bezirk']))
    bd['district'] = bd['BFS_NR'].map(bfs_to_bezirk)
    bad = bd['district'].isna() | ~bd['district'].isin(valid_districts)
    if bad.any():
        print(f"  {int(bad.sum())} breakdown row(s) with unknown/unmapped "
              f"Bezirk -> CH trajectory")
        bd.loc[bad, 'district'] = _CH_DISTRICT

    W = (bd.groupby(['station_id', 'district'])['pop_in_station'].sum()
         .unstack(fill_value=0.0))
    if _CH_DISTRICT not in W.columns:
        W[_CH_DISTRICT] = 0.0
    weighted = set(W.index)
    gateway = sorted(s for s in station_ids
                     if s not in weighted and s in gateway_ids)
    fallback = sorted(s for s in station_ids
                      if s not in weighted and s not in gateway_ids)
    if fallback:
        print(f"  WARNING: {len(fallback)} OD station(s) without "
              f"pop_in_station weights and not gateways -> CH trajectory: "
              f"{fallback}")
    extra = pd.DataFrame(0.0, columns=W.columns,
                         index=pd.Index(gateway + fallback, name='station_id'))
    extra[_CH_DISTRICT] = 1.0
    W = pd.concat([W, extra]).sort_index()
    W = W.div(W.sum(axis=1), axis=0)
    return W, {'weighted': sorted(weighted), 'gateway': gateway,
               'fallback': fallback}


def _station_factor_frame(W: pd.DataFrame, district_factors: pd.DataFrame,
                          start_year: int) -> pd.DataFrame:
    """Per-station factors F = G @ W^T as long (scenario, year, station_id,
    factor); asserts factor == 1.0 at start_year."""
    G = district_factors.pivot(index=['scenario', 'year'], columns='district',
                               values='factor')
    missing = [d for d in W.columns if d not in G.columns]
    if missing:
        raise ValueError(f"No growth trajectory for district(s): {missing}")
    F = pd.DataFrame(G[W.columns].values @ W.T.values,
                     index=G.index, columns=W.index)
    base = F.xs(start_year, level='year')
    if not np.allclose(base.values, 1.0, atol=1e-9):
        raise AssertionError(
            "Base-year station factors deviate from 1.0 — check trajectories")
    long = F.stack().rename('factor').reset_index()
    long['station_id'] = long['station_id'].astype(int)
    return long[['scenario', 'year', 'station_id', 'factor']]


def _modal_distance_table(n_scenarios: int, start_year: int,
                          end_year: int) -> pd.DataFrame:
    """Station-independent (scenario, year, modal_factor, distance_factor),
    relative to start_year; legacy parameters kept (the hardcoded values from
    generate_od_growth_scenarios). Memoised like the district table."""
    key = (n_scenarios, start_year, end_year)
    if key in _MODAL_DISTANCE_MEMO:
        return _MODAL_DISTANCE_MEMO[key]
    modal = generate_modal_split_scenarios(
        avg_growth_rate=0.0045, start_value=0.209, start_year=start_year,
        end_year=end_year, n_scenarios=n_scenarios, start_std_dev=0.015,
        end_std_dev=0.045, std_dev_shocks=0.02)
    dist = generate_distance_per_person_scenarios(
        avg_growth_rate=-0.0027, start_value=39.79, start_year=start_year,
        end_year=end_year, n_scenarios=n_scenarios, start_std_dev=0.005,
        end_std_dev=0.015, std_dev_shocks=0.015)

    def _factorise(df, value_col, name):
        wide = df.pivot(index='scenario', columns='year', values=value_col)
        fac = wide.div(wide[start_year].replace(0, np.nan), axis=0).fillna(1.0)
        long = (fac.reset_index()
                .melt(id_vars='scenario', var_name='year', value_name=name))
        long['scenario'] = long['scenario'].astype(int) + 1
        long['year'] = long['year'].astype(int)
        return long

    out = _factorise(modal, 'modal_split', 'modal_factor').merge(
        _factorise(dist, 'distance_per_person', 'distance_factor'),
        on=['scenario', 'year'])
    _MODAL_DISTANCE_MEMO[key] = out
    return out


def _plot_factor_store_inputs(n_scenarios: int, start_year: int,
                              end_year: int, max_districts: int = 3) -> None:
    """Fan plots (range/mean/90% band) for the first districts, the CH
    pseudo-district, modal split and distance per person."""
    plot_dir = os.path.join(paths.MAIN, paths.PLOT_SCENARIOS)
    os.makedirs(plot_dir, exist_ok=True)
    refs = get_bezirk_population_scenarios()
    targets = list(refs.items())[:max_districts]
    targets.append(('CH', _get_ch_population_reference(start_year, end_year)))
    for district, ref in targets:
        scen = generate_population_scenarios(ref, start_year, end_year,
                                             n_scenarios)
        label = f"population_{district.replace(' ', '_')}"
        plot_scenarios_with_range(scen.rename(columns={'population': label}),
                                  plot_dir, label)
    modal = generate_modal_split_scenarios(
        avg_growth_rate=0.0045, start_value=0.209, start_year=start_year,
        end_year=end_year, n_scenarios=n_scenarios, start_std_dev=0.015,
        end_std_dev=0.045, std_dev_shocks=0.02)
    plot_scenarios_with_range(modal, plot_dir, 'modal_split')
    dist = generate_distance_per_person_scenarios(
        avg_growth_rate=-0.0027, start_value=39.79, start_year=start_year,
        end_year=end_year, n_scenarios=n_scenarios, start_std_dev=0.005,
        end_std_dev=0.015, std_dev_shocks=0.015)
    plot_scenarios_with_range(dist, plot_dir, 'distance_per_person')
    print(f"  Scenario fan plots -> {plot_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# Legacy full-OD pickle path (main.py / main_cap.py only) — kept untouched
# ═══════════════════════════════════════════════════════════════════════════

def get_bezirk_population_scenarios():
    # Read the Swiss population scenario CSV with "," separator
    df_ch = pd.read_csv(paths.POPULATION_SCENARIO_CH_BFS_2055, sep=",")
    # Extract the relevant values for 2018 and 2050
    pop_2018 = df_ch.loc[df_ch['Jahr'] == 2018, 'Beobachtungen'].values
    pop_2050 = df_ch.loc[df_ch['Jahr'] == 2050, 'Referenzszenario A-00-2025'].values
    # Compute the growth factor: population_2050 / population_2018
    swiss_growth_factor_18_50 = pop_2050[0] / pop_2018[0]
    # Read the CSV file with ";" as separator
    df = pd.read_csv(paths.POPULATION_SCENARIO_CANTON_ZH_2050, sep=';')
    # Step 1: Aggregate total population per district and year
    population_summary = (
        df.groupby(['bezirk', 'jahr'])['anzahl']
        .sum()
        .reset_index()
        .rename(columns={'anzahl': 'total_population'})
    )
    # Step 2: Create full grid for all districts and all years from 2011 to 2050
    all_years = pd.Series(range(2011, 2051), name='jahr')
    all_districts = population_summary['bezirk'].unique()
    full_index = pd.MultiIndex.from_product([all_districts, all_years], names=['bezirk', 'jahr'])
    # Reindex to ensure each district has all years, fill missing population with 0
    population_complete = (
        population_summary.set_index(['bezirk', 'jahr'])
        .reindex(full_index)
        .fillna({'total_population': 0})
        .reset_index()
    )
    # Step 3: Calculate year-over-year growth rate per district
    population_complete['growth_rate'] = (
        population_complete
        .sort_values(['bezirk', 'jahr'])
        .groupby('bezirk')['total_population']
        .pct_change()
    )
    # Step 4: Split the complete dataset into a dictionary by district
    district_tables = {}
    for district, group in population_complete.groupby('bezirk'):
        group = group.reset_index(drop=True)

        # Extract population for 2018 and 2050
        pop_2018 = group.loc[group['jahr'] == 2018, 'total_population'].values
        pop_2050 = group.loc[group['jahr'] == 2050, 'total_population'].values

        growth_factor_18_50 = pop_2050[0] / pop_2018[0]

        # Compute yearly relative growth factor vs. CH
        relative_growth = (growth_factor_18_50 - 1) / (swiss_growth_factor_18_50 - 1)
        yearly_growth_factor = relative_growth ** (
                    1 / 32)  # this factor is only applicable to yearly growth RATES in the form of 0.015 for example, not FACTORS 1.015!!!

        # Store in DataFrame attributes
        group.attrs['growth_factor_18_50'] = growth_factor_18_50
        group.attrs['yearly_growth_factor_district_to_CH'] = yearly_growth_factor

        district_tables[district] = group
    # Step 5: Read Swiss growth rates from Eurostat Excel file
    eurostat_df = pd.read_excel(paths.POPULATION_SCENARIO_CH_EUROSTAT_2100)
    # Convert all column names to strings FIRST (important!)
    eurostat_df.columns = eurostat_df.columns.map(str)
    # Filter for the row where unit == 'GROWTH_RATE'
    growth_rate_row = eurostat_df[eurostat_df['unit'] == 'GROWTH_RATE']
    # Define year columns as strings
    year_columns = [str(year) for year in range(2051, 2101)]
    # Extract growth rates from that row
    ch_growth_rates = growth_rate_row[year_columns].iloc[0].astype(float)
    # Step 6: Extend each district with projected growth rates and populations
    for district, df_district in district_tables.items():
        # Get the last known population (for 2050)
        last_population = df_district.loc[df_district['jahr'] == 2050, 'total_population'].values[0]

        # Get the district-specific yearly growth factor to scale national growth rates
        scaling_factor = df_district.attrs['yearly_growth_factor_district_to_CH']

        # Prepare data for years 2051–2100
        new_rows = []
        current_population = last_population

        for year in range(2051, 2101):
            base_growth_rate = ch_growth_rates[str(year)]  # national growth rate (e.g., 0.012)
            adjusted_growth_rate = base_growth_rate * scaling_factor

            current_population *= (1 + adjusted_growth_rate)

            new_rows.append({
                'bezirk': district,
                'jahr': year,
                'total_population': current_population,
                'growth_rate': adjusted_growth_rate
            })

        # Convert new rows to DataFrame and append
        extension_df = pd.DataFrame(new_rows)
        df_extended = pd.concat([df_district, extension_df], ignore_index=True)
        district_tables[district] = df_extended.reset_index(drop=True)
    return district_tables



def generate_population_scenarios(ref_df: pd.DataFrame,
                                  start_year: int,
                                  end_year: int,
                                  n_scenarios: int = 1000,
                                  start_std_dev: float = 0.01,
                                  end_std_dev: float = 0.03,
                                  std_dev_shocks: float = 0.02) -> pd.DataFrame:
    """
    Generate stochastic population scenarios using Latin Hypercube Sampling and a random walk process.
    The main growth rates are perturbed using LHS and a time-varying std dev. Random shocks are added separately.

    Parameters:
    - ref_df: DataFrame with columns "jahr", "total_population", "growth_rate"
              - Only the "total_population" value at start_year is used as the initial population.
              - "growth_rate" is used as the base deterministic growth.
    - start_year: year to begin scenario generation
    - end_year: year to end scenario generation
    - n_scenarios: number of scenarios to generate
    - start_std_dev: starting std deviation applied to growth rate perturbation
    - end_std_dev: ending std deviation applied to growth rate perturbation
    - std_dev_shocks: std deviation of yearly additive shocks

    Returns:
    - DataFrame with columns: "scenario", "year", "population", "growth_rate"
    """
    # Filter and sort reference data
    ref_df = ref_df.sort_values("jahr")
    ref_df = ref_df[(ref_df["jahr"] >= start_year) & (ref_df["jahr"] <= end_year)].reset_index(drop=True)

    years = ref_df["jahr"].values
    ref_growth = ref_df["growth_rate"].values  # deterministic base growth per year
    n_years = len(years)
    initial_population = ref_df[ref_df["jahr"] == start_year]["total_population"].values[0]

    # Linearly interpolate std devs across years for growth rate variation
    growth_std_devs = np.linspace(start_std_dev, end_std_dev, n_years)

    # Latin Hypercube Sampling: growth rate perturbations
    sampler = qmc.LatinHypercube(d=n_years, seed = 42)
    lhs_samples = sampler.random(n=n_scenarios)  # shape: (n_scenarios, n_years)
    growth_perturbations = norm.ppf(lhs_samples) * growth_std_devs  # shape: (n_scenarios, n_years)

    # Perturbed growth rate: base + scenario-specific offset
    scenario_growth = ref_growth + growth_perturbations  # shape: (n_scenarios, n_years)
    # Setze Wachstumsrate für das erste Jahr auf 0 (kein Wachstum im ersten Jahr)
    scenario_growth[:, 0] = 0

    # Random shocks: et ~ N(0, std_dev_shocks)
    shock_sampler = qmc.LatinHypercube(d=n_years, seed = 43)
    lhs_shocks = shock_sampler.random(n=n_scenarios)
    et = norm.ppf(lhs_shocks) * std_dev_shocks
    # Setze Schocks für das erste Jahr auf 0
    et[:, 0] = 0

    # Cumulative shocks per scenario
    cumulative_shocks = np.cumsum(et, axis=1)  # shape: (n_scenarios, n_years)

    # Deterministic growth: cumulative product of (1 + growth_rate)
    deterministic_growth = np.cumprod(1 + scenario_growth, axis=1)  # shape: (n_scenarios, n_years)

    # Population index = deterministic path × stochastic shocks
    population_index = deterministic_growth + cumulative_shocks

    # Scale by initial population
    pop_scenarios = initial_population * population_index

    # Assemble output DataFrame
    scenario_data = []
    for i in range(n_scenarios):
        for t in range(n_years):
            # Berechne growth_index: 100 am Anfang und dann entsprechend der relativen Bevölkerungsentwicklung
            growth_index = 100 * (pop_scenarios[i, t] / initial_population)

            # Berechne die effektive Wachstumsrate inklusive Schocks
            if t == 0:
                # Für das erste Jahr ist die Wachstumsrate definitionsgemäß 0
                effective_growth_rate = 0.0
            else:
                # Berechne die prozentuale Änderung zur Bevölkerung des Vorjahres
                effective_growth_rate = (pop_scenarios[i, t] / pop_scenarios[i, t - 1]) - 1

            scenario_data.append({
                "scenario": i,
                "year": years[t],
                "population": pop_scenarios[i, t],
                "growth_rate": effective_growth_rate,
                "growth_index": growth_index
            })

    return pd.DataFrame(scenario_data)


def generate_modal_split_scenarios(avg_growth_rate: float,
                                   start_value: float,
                                   start_year: int,
                                   end_year: int,
                                   n_scenarios: int = 1000,
                                   start_std_dev: float = 0.01,
                                   end_std_dev: float = 0.03,
                                   std_dev_shocks: float = 0.02) -> pd.DataFrame:
    """
    Generate stochastic modal split scenarios using Latin Hypercube Sampling and a random walk process.

    Parameters:
    - avg_growth_rate: average annual growth rate to apply (can be positive or negative)
    - start_value: initial modal split value at start_year
    - start_year: year to begin scenario generation
    - end_year: year to end scenario generation
    - n_scenarios: number of scenarios to generate
    - start_std_dev: starting std deviation applied to growth rate perturbation
    - end_std_dev: ending std deviation applied to growth rate perturbation
    - std_dev_shocks: std deviation of yearly additive shocks

    Returns:
    - DataFrame with columns: "scenario", "year", "modal_split", "growth_rate", "growth_index"
    """
    # Erstelle temporären Referenzdatensatz mit konstanter Wachstumsrate
    years = np.arange(start_year, end_year + 1)
    n_years = len(years)

    # Berechne die Werte mit konstanter Wachstumsrate
    growth_factors = np.ones(n_years) * (1 + avg_growth_rate)
    growth_factors[0] = 1  # Erster Faktor ist 1, da es der Startwert ist

    # Kumulatives Wachstum berechnen
    cumulative_growth = np.cumprod(growth_factors)
    modal_split_values = start_value * cumulative_growth

    # Erstelle Array von Wachstumsraten (erster Wert ist 0, danach konstant)
    growth_rates = np.zeros(n_years)
    growth_rates[1:] = avg_growth_rate  # Konstante Wachstumsrate für alle Jahre außer dem ersten

    # Erstelle temporären DataFrame
    ref_df = pd.DataFrame({
        "jahr": years,
        "total_population": modal_split_values,
        "growth_rate": growth_rates
    })

    # Verwende die bestehende Funktion
    modal_split_scenarios_df = generate_population_scenarios(
        ref_df=ref_df,
        start_year=start_year,
        end_year=end_year,
        n_scenarios=n_scenarios,
        start_std_dev=start_std_dev,
        end_std_dev=end_std_dev,
        std_dev_shocks=std_dev_shocks
    )

    # Umbenennen der Spalte "population" in "modal_split"
    modal_split_scenarios_df = modal_split_scenarios_df.rename(columns={"population": "modal_split"})

    return modal_split_scenarios_df


def generate_distance_per_person_scenarios(avg_growth_rate: float,
                                           start_value: float,
                                           start_year: int,
                                           end_year: int,
                                           n_scenarios: int = 1000,
                                           start_std_dev: float = 0.01,
                                           end_std_dev: float = 0.03,
                                           std_dev_shocks: float = 0.02) -> pd.DataFrame:
    """
    Generate stochastic trips per person scenarios using Latin Hypercube Sampling and a random walk process.

    Parameters:
    - avg_growth_rate: average annual growth rate to apply (can be positive or negative)
    - start_value: initial trips per person value at start_year
    - start_year: year to begin scenario generation
    - end_year: year to end scenario generation
    - n_scenarios: number of scenarios to generate
    - start_std_dev: starting std deviation applied to growth rate perturbation
    - end_std_dev: ending std deviation applied to growth rate perturbation
    - std_dev_shocks: std deviation of yearly additive shocks

    Returns:
    - DataFrame with columns: "scenario", "year", "trips_per_person", "growth_rate", "growth_index"
    """
    # Nutze die bestehende Funktion für Modal-Split-Szenarien
    scenarios_df = generate_modal_split_scenarios(
        avg_growth_rate=avg_growth_rate,
        start_value=start_value,
        start_year=start_year,
        end_year=end_year,
        n_scenarios=n_scenarios,
        start_std_dev=start_std_dev,
        end_std_dev=end_std_dev,
        std_dev_shocks=std_dev_shocks
    )

    # Benenne die Spalte "modal_split" in "trips_per_person" um
    scenarios_df = scenarios_df.rename(columns={"modal_split": "distance_per_person"})

    return scenarios_df


def plot_population_scenarios(scenarios_df: pd.DataFrame, n_to_plot: int = 10):
    """
    Plot a sample of population scenarios.

    Parameters:
    - scenarios_df: DataFrame with columns "scenario", "year", "population"
    - n_to_plot: number of scenarios to randomly plot
    """
    plt.figure(figsize=(10, 6),dpi=300)

    sample_ids = scenarios_df["scenario"].drop_duplicates().sample(n=min(n_to_plot, scenarios_df["scenario"].nunique()))
    sample_df = scenarios_df[scenarios_df["scenario"].isin(sample_ids)]

    for scenario_id in sample_df["scenario"].unique():
        data = sample_df[sample_df["scenario"] == scenario_id]
        plt.plot(data["year"], data["population"] / 1e3, label=f"Scenario {scenario_id}")

    plt.xlabel("Year")
    plt.ylabel("Population (thousands)")
    plt.title(f"Sample of {n_to_plot} Population Scenarios")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_scenarios_with_range(
        scenarios_df: pd.DataFrame,
        save_path,
        value_col: str = "population"
):
    """
    Plot the range of all scenarios for a given value column as a shaded area
    and a single example scenario, with automatically scaled SI-prefix axis.
    Also plots lines for +/- 1.65 standard deviations from the mean (contains 90% of all values).

    Parameters:
    - scenarios_df: DataFrame with columns "scenario", "year", and the specified value column
    - save_path: path where the plot will be saved
    - value_col: name of the column in scenarios_df containing the values to plot
    """
    # compute per-year stats
    year_stats = (
        scenarios_df
        .groupby("year")[value_col]
        .agg(min="min", max="max", mean="mean", std="std")
        .reset_index()
    )

    # Berechne +/- 1.65 Standardabweichungen (90% Konfidenzintervall)
    year_stats["mean_plus_1_65std"] = year_stats["mean"] + 1.65 * year_stats["std"]
    year_stats["mean_minus_1_65std"] = year_stats["mean"] - 1.65 * year_stats["std"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # shaded range
    ax.fill_between(
        year_stats["year"],
        year_stats["min"],
        year_stats["max"],
        color='grey', alpha=0.3,
        label="Total Range"
    )

    # +/- 1.65 Std. Abw. Linien (90% Konfidenzintervall)
    ax.plot(
        year_stats["year"],
        year_stats["mean_plus_1_65std"],
        color='red', linestyle='-', alpha=0.7,
        label="+1.65σ (95th percentile)"
    )

    ax.plot(
        year_stats["year"],
        year_stats["mean_minus_1_65std"],
        color='red', linestyle='-', alpha=0.7,
        label="-1.65σ (5th percentile)"
    )

    # mean line
    ax.plot(
        year_stats["year"],
        year_stats["mean"],
        color='grey', linestyle='--', alpha=0.8,
        label="Mean"
    )

    # pick a random scenario to highlight
    sample_id = scenarios_df["scenario"].drop_duplicates().sample(n=1).iloc[0]
    sample_df = scenarios_df[scenarios_df["scenario"] == sample_id]
    ax.plot(
        sample_df["year"],
        sample_df[value_col],
        color='blue', linewidth=2,
        label=f"Sample Scenario {sample_id}"
    )

    # apply automatic SI‐prefix scaling on the Y axis
    ax.yaxis.set_major_formatter(EngFormatter(unit='', places=2))

    # labels & styling
    col_title = value_col.replace('_', ' ').title()
    ax.set_xlabel("Year")
    ax.set_title(f"{col_title} Scenarios: Range, Mean and 90% Confidence Interval")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    # Save the plot, creating a filename based on the value column
    filename = f"{value_col.lower().replace(' ', '_')}_scenarios.png"
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path)
    # plt.show()
    plt.close(fig)


def build_station_to_communes_mapping(
        communes_to_stations: pd.DataFrame
) -> Dict[str, List[int]]:
    """
    Baut ein Mapping: station_id -> List[Commune_BFS_code].
    """
    return communes_to_stations.groupby('ID_point')['Commune_BFS_code'] \
        .apply(list) \
        .to_dict()


def compute_growth_od_matrix_optimized(
        initial_od: pd.DataFrame,
        station_communes: Dict[str, List[int]],
        communes_population: pd.DataFrame,
        population_scenarios: Dict[str, pd.DataFrame],
        scenario: int,
        year: int,
        start_year: int,
        station_commune_lookup: Dict[str, pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Optimierte Version von compute_growth_od_matrix
    """
    # --- 1) Struktur aufsetzen
    stations = initial_od.columns.tolist()
    from_stations = initial_od.index.tolist()

    # --- 2) Wachstumsindex für alle Stationen berechnen
    station_growth = {}

    # Wenn lookup nicht existiert, erstellen wir einen
    if station_commune_lookup is None:
        station_commune_lookup = {}

    for station in stations:
        # Lookup für diese Station verwenden wenn vorhanden
        if station not in station_commune_lookup:
            communes = station_communes.get(int(station), [])
            if not communes:
                station_growth[station] = 1.0
                continue

            # Vorfiltern der relevanten Gemeinden für diese Station
            station_data = []
            for commune in communes:
                row = communes_population[communes_population['gemeinde_bfs_nr'] == commune]
                if not row.empty:
                    district = row['bezirk'].iat[0]
                    pop_start_commune = row['anzahl'].iat[0]
                    station_data.append((commune, district, pop_start_commune))

            station_commune_lookup[station] = station_data

        sum_start = 0.0
        sum_curr = 0.0

        for commune, district, pop_start_commune in station_commune_lookup[station]:
            # Population im Szenario für Start- und Ziel-Jahr
            scen = population_scenarios[district]

            # Effizienterer Zugriff mit vorgefilterten Daten
            scenario_data = scen[(scen['scenario'] == scenario)]
            pop_d_start = scenario_data[scenario_data['year'] == start_year]['population'].iloc[0]
            pop_d_curr = scenario_data[scenario_data['year'] == year]['population'].iloc[0]

            # Bezirksfaktor
            factor_d = (pop_d_curr / pop_d_start) if pop_d_start > 0 else 1.0

            sum_start += pop_start_commune
            sum_curr += pop_start_commune * factor_d

        station_growth[station] = (sum_curr / sum_start) if sum_start > 0 else 1.0

    # --- 3) OD-Matrix mit Wachstumsfaktoren erzeugen
    growth_od = pd.DataFrame(1.0, index=from_stations, columns=stations)

    # Vektorisierte Anwendung der Faktoren
    sqrt_factors = {station: np.sqrt(factor) for station, factor in station_growth.items()}

    # Zeilenweise Multiplikation
    for station in set(list(map(str, from_stations))).intersection(sqrt_factors.keys()):
        growth_od.loc[int(station), :] *= sqrt_factors[station]

    # Spaltenweise Multiplikation
    for station in set(stations).intersection(sqrt_factors.keys()):
        growth_od.loc[:, station] *= sqrt_factors[station]

    return growth_od


def apply_modal_trips_optimized(
        initial_od: pd.DataFrame,
        growth_od: pd.DataFrame,
        modal_factors: Dict[tuple, float],
        distance_factors: Dict[tuple, float],
        scenario: int,
        start_year: int,
        year: int
) -> pd.DataFrame:
    """
    Optimierte Version von apply_modal_trips mit Lookup-Table
    """
    # Faktoren aus Lookup-Tabelle holen
    scenario_year_key = (scenario, year)
    m_factor = modal_factors.get(scenario_year_key, 1.0)
    d_factor = distance_factors.get(scenario_year_key, 1.0)

    # Einen Schritt berechnen
    return (initial_od * growth_od * m_factor * d_factor).astype('float32')


def precompute_modal_distance_factors(
        modal_df: pd.DataFrame,
        distance_df: pd.DataFrame,
        start_year: int
) -> tuple:
    """
    Vorausberechnung der Modal- und Distance-Faktoren
    """
    modal_factors = {}
    distance_factors = {}

    scenarios = modal_df['scenario'].unique()
    years = modal_df['year'].unique()

    # Modal split factors
    for s in scenarios:
        m_start = modal_df.loc[(modal_df['scenario'] == s) &
                               (modal_df['year'] == start_year), 'modal_split'].iat[0]

        for y in years:
            if y == start_year:
                modal_factors[(s, y)] = 1.0
                continue

            m_curr = modal_df.loc[(modal_df['scenario'] == s) &
                                  (modal_df['year'] == y), 'modal_split'].iat[0]
            m_factor = (m_curr / m_start) if m_start > 0 else 1.0
            modal_factors[(s, y)] = m_factor

    # Distance per person factors
    for s in scenarios:
        d_start = distance_df.loc[(distance_df['scenario'] == s) &
                                  (distance_df['year'] == start_year), 'distance_per_person'].iat[0]

        for y in years:
            if y == start_year:
                distance_factors[(s, y)] = 1.0
                continue

            d_curr = distance_df.loc[(distance_df['scenario'] == s) &
                                     (distance_df['year'] == y), 'distance_per_person'].iat[0]
            d_factor = (d_curr / d_start) if d_start > 0 else 1.0
            distance_factors[(s, y)] = d_factor

    return modal_factors, distance_factors


def generate_od_growth_scenarios(
        initial_od_matrix: pd.DataFrame,
        communes_to_stations: pd.DataFrame,
        communes_population: pd.DataFrame,
        start_year: int,
        end_year: int,
        num_of_scenarios: int,
        do_plot: bool = False,
        n_jobs: int = -1
) -> Dict[str, Dict[int, pd.DataFrame]]:
    """
    Optimierte Version von generate_od_growth_scenarios mit Multiprocessing
    """

    initial_od_matrix = initial_od_matrix.set_index('from_station')
    # 1) Bezirkspopulationsszenarien
    bezirk_pop_scenarios = get_bezirk_population_scenarios()
    population_scenarios = {
        bezirk: generate_population_scenarios(df, start_year, end_year, num_of_scenarios)
        for bezirk, df in bezirk_pop_scenarios.items()
    }

    # 2) Modal-Split- & Distance-per-Person-Szenarien
    modal_split_scenarios = generate_modal_split_scenarios(
        avg_growth_rate=0.0045,
        start_value=0.209,
        start_year=start_year,
        end_year=end_year,
        n_scenarios=num_of_scenarios,
        start_std_dev=0.015,
        end_std_dev=0.045,
        std_dev_shocks=0.02
    )
    distance_per_person_scenarios = generate_distance_per_person_scenarios(
        avg_growth_rate=-0.0027,
        start_value=39.79,
        start_year=start_year,
        end_year=end_year,
        n_scenarios=num_of_scenarios,
        start_std_dev=0.005,
        end_std_dev=0.015,
        std_dev_shocks=0.015
    )
    if do_plot:
        os.chdir(paths.MAIN)
        first_three_bezirk = list(population_scenarios.keys())[:3]
        first_three_scenarios = {bezirk: population_scenarios[bezirk] for bezirk in first_three_bezirk}
        for pop_scenario in first_three_scenarios.values():
            plot_scenarios_with_range(pop_scenario, paths.PLOT_SCENARIOS, 'population')
        plot_scenarios_with_range(modal_split_scenarios, paths.PLOT_SCENARIOS,'modal_split')
        plot_scenarios_with_range(distance_per_person_scenarios, paths.PLOT_SCENARIOS,'distance_per_person')

    # components = {
    #     "population_scenarios": population_scenarios,
    #     "modal_split_scenarios": modal_split_scenarios,
    #     "distance_per_person_scenarios": distance_per_person_scenarios
    # }


    # Speichere alle Komponenten in einer Datei
    #with open("scenario_data_for_plots.pkl", 'wb') as f:
    #    pickle.dump(components, f)
    # Vorauswertung aller Modal/Distance Faktoren
    print("Berechne Modal und Distance Faktoren...")

    modal_factors, distance_factors = precompute_modal_distance_factors(
        modal_split_scenarios, distance_per_person_scenarios, start_year
    )

    # 3) Station→Commune-Mapping (einmalig)
    print("Erstelle Station-Commune Mapping...")
    station_communes = build_station_to_communes_mapping(communes_to_stations)
    station_commune_lookup = {}  # Cache für station-commune Beziehungen

    # 4) Funktion für parallele Verarbeitung
    def process_scenario(s):
        key = f"scenario_{s + 1}"
        results_s = {}

        for y in range(start_year, end_year + 1):
            pop_growth_od = compute_growth_od_matrix_optimized(
                initial_od_matrix,
                station_communes,
                communes_population,
                population_scenarios,
                s, y, start_year,
                station_commune_lookup
            )

            final_od = apply_modal_trips_optimized(
                initial_od_matrix,
                pop_growth_od,
                modal_factors,
                distance_factors,
                s, start_year, y
            )
            results_s[y] = final_od

        return key, results_s

    # 5) Parallele Verarbeitung mit Fortschrittsbalken
    print(f"Berechne {num_of_scenarios} Szenarien mit {n_jobs} Prozessen...")
    scenario_results = Parallel(n_jobs=n_jobs, verbose = 100 )( #backend="loky", max_nbytes=None
        delayed(process_scenario)(s) for s in range(num_of_scenarios)
    )
    #scenario_results = [process_scenario(s) for s in range(num_of_scenarios)]
    print("fertig")
    # 6) Ergebnisse zusammenführen
    results = dict(scenario_results)
    return results


# bezirk_pop_scen = get_bezirk_population_scenarios()
# affoltern_df = bezirk_pop_scen['Affoltern']
# pop_scenarios_df = generate_population_scenarios(affoltern_df, 2022, 2100,n_scenarios=100, start_std_dev=0.005, end_std_dev=0.01, std_dev_shocks=0.02)
#
# plot_population_scenarios(pop_scenarios_df, n_to_plot=100)
# ms_scenario_df = generate_modal_split_scenarios(0.0045, 0.209, 2022, 2100, n_scenarios=100, start_std_dev=0.002, end_std_dev=0.005, std_dev_shocks=0.01)#growth rate assumption from verkehrsperspektiven 2017 til 2060
# trips_per_person_scenario_df = generate_distance_per_person_scenarios(-0.0027, 39.79, 2022, 2100, n_scenarios=100, start_std_dev=0.002, end_std_dev=0.005, std_dev_shocks=0.01)#growth rate assumption from verkehrsperspektiven 2017 til 2050 via gesamte verkehrsleistung, computed with chatgpt
#
# plot_scenarios_with_range(pop_scenarios_df,'population')
# plot_scenarios_with_range(ms_scenario_df, 'modal_split')
# plot_scenarios_with_range(trips_per_person_scenario_df, 'distance_per_person')



def load_scenarios_from_cache(cache_dir):
    """
    Load scenarios from individual .pkl files in the cache directory.
    
    Parameters:
    - cache_dir: Directory containing scenario .pkl files
    
    Returns:
    - Dictionary of loaded scenarios
    """
    scenarios = {}
    if os.path.exists(cache_dir):
        for file in os.listdir(cache_dir):
            if file.endswith('.pkl'):
                scenario_name = file.replace('.pkl', '')
                with open(os.path.join(cache_dir, file), 'rb') as f:
                    scenarios[scenario_name] = pickle.load(f)
        print(f"Loaded {len(scenarios)} scenarios from {cache_dir}")
    return scenarios

def _build_communes_population_df(year: int) -> pd.DataFrame:
    """Build the communes_population DataFrame for a given year.

    Uses the commune->bezirk skeleton from POPULATION_PER_COMMUNE_ZH_2018
    (bezirk names match the scenario keys in get_bezirk_population_scenarios)
    and replaces the anzahl column with actual year-specific populations from
    catchment_base.load_commune_pop(year).

    Returns:
        DataFrame with columns: gemeinde_bfs_nr, gemeinde, bezirk_code,
        bezirk, anzahl.
    """
    skeleton = pd.read_csv(
        paths.POPULATION_PER_COMMUNE_ZH_2018,
        usecols=['gemeinde_bfs_nr', 'gemeinde', 'bezirk_code', 'bezirk']
    )
    pop_series = catchment_base.load_commune_pop(year)
    pop_df = pop_series.rename('anzahl').reset_index()
    pop_df.columns = ['gemeinde_bfs_nr', 'anzahl']
    pop_df['gemeinde_bfs_nr'] = pd.to_numeric(
        pop_df['gemeinde_bfs_nr'], errors='coerce'
    ).astype('Int64')
    skeleton['gemeinde_bfs_nr'] = pd.to_numeric(
        skeleton['gemeinde_bfs_nr'], errors='coerce'
    ).astype('Int64')
    result = skeleton.merge(pop_df, on='gemeinde_bfs_nr', how='left')
    result['anzahl'] = result['anzahl'].fillna(0.0)
    return result


def _get_od_base_matrix(start_year: int) -> pd.DataFrame:
    """Return the station-level OD base matrix for start_year.

    For start_year >= 2040 and when a 2040 station-level OD file exists at
    paths.OD_STATIONS_KT_ZH_2040_PATH, the 2040 matrix is returned.
    Falls back to the 2018 matrix for all other cases (the 2040 file is
    derived from commune-level 2040 OD via the catchment allocation pipeline
    and is not yet produced automatically).
    """
    if start_year >= 2040 and os.path.exists(paths.OD_STATIONS_KT_ZH_2040_PATH):
        print(f"  OD base: using 2040 station matrix ({paths.OD_STATIONS_KT_ZH_2040_PATH})")
        return pd.read_csv(paths.OD_STATIONS_KT_ZH_2040_PATH)
    if start_year >= 2040:
        print(f"  OD base: 2040 matrix not found at {paths.OD_STATIONS_KT_ZH_2040_PATH},"
              f" falling back to 2018 matrix.")
    return pd.read_csv(paths.OD_STATIONS_KT_ZH_PATH)


def get_random_scenarios(start_year=2018, end_year=2100, num_of_scenarios=100, use_cache=False, do_plot=False):
    """
    Retrieve or generate random OD growth scenarios.

    Parameters:
    - start_year: The starting year for the scenarios.
    - end_year: The ending year for the scenarios.
    - num_of_scenarios: The number of scenarios to generate.
    - use_cache: If True, load scenarios from cache instead of regenerating.
    - do_plot: If True, plot the scenarios after generation.

    Returns:
    - scenarios: The scenario DataFrame.
    """
    cache_dir = paths.RANDOM_SCENARIO_CACHE_PATH

    if use_cache:
        return

    # Generate new scenarios
    scenarios = generate_od_growth_scenarios(
        _get_od_base_matrix(start_year),
        pd.read_excel(paths.COMMUNE_TO_STATION_PATH),
        _build_communes_population_df(settings.start_year_scenario),
        start_year=start_year,
        end_year=end_year,
        num_of_scenarios=num_of_scenarios,
        do_plot=do_plot
    )

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)  # Ensure directory exists

    # Empty the directory first (only .pkl files)
    for filename in os.listdir(cache_dir):
        if filename.endswith(".pkl"):
            os.remove(os.path.join(cache_dir, filename))
    # Save each scenario to a separate .pkl file
    for scenario_name, scenario_data in scenarios.items():
        scen_path = os.path.join(cache_dir, f"{scenario_name}.pkl")
        with open(scen_path, 'wb') as f:
            pickle.dump(scenario_data, f)
    print(f"Saved {len(scenarios)} scenarios to {cache_dir}")
    
    return


if __name__ == '__main__':
    # Standalone CLI (module-CLI pattern): prompts with settings defaults.
    # Through main_new, phase_7_scenarios passes the settings values directly.
    os.chdir(paths.MAIN)
    print("=== Phase 7: scenario factor store (standalone) ===")

    _svc_default = settings.SVC_VERSION
    if _svc_default == 'Build_New':
        _svc_default = settings.SVC_BUILD_NEW_NAME
    _svc = input(f"Service version [{_svc_default}]: ").strip() or _svc_default

    import infra_ints_orchestrator as _io
    _infra_default = _io._resolve_base_version()
    _infra = (input(f"Infrastructure version [{_infra_default}]: ").strip()
              or _infra_default)

    _method_default = ('municipal' if settings.CATCHMENT_METHOD == 'Municipal'
                       else 'pt_feeder')
    _method = (input(f"OD method (pt_feeder/municipal) [{_method_default}]: ")
               .strip() or _method_default)
    _attr_default = (settings.OD_ATTRIBUTION_MODE if _method == 'pt_feeder'
                     else 'municipal')
    _attr = (input(f"Attribution [{_attr_default}]: ").strip() or _attr_default)

    _n = int(input(f"Number of scenarios [{settings.amount_of_scenarios}]: ")
             .strip() or settings.amount_of_scenarios)
    _y0 = int(input(f"Start year [{settings.start_year_scenario}]: ")
              .strip() or settings.start_year_scenario)
    _y1 = int(input(f"End year [{settings.end_year_scenario}]: ")
              .strip() or settings.end_year_scenario)

    _plots_default = 'y' if settings.PLOT_SCENARIOS else 'n'
    _plots = (input(f"Generate plots? (y/n) [{_plots_default}]: ").strip()
              or _plots_default).lower() == 'y'
    _cache_default = 'y' if settings.use_cache_scenarios else 'n'
    _cache = (input(f"Use cache? (y/n) [{_cache_default}]: ").strip()
              or _cache_default).lower() == 'y'

    build_scenario_factor_store(_svc, _infra, _method, _attr, _n, _y0, _y1,
                                make_plots=_plots, use_cache=_cache)

    if _method == 'pt_feeder':
        _ov = (input("Build per-svc-int overrides? (y/n) [y]: ").strip()
               or 'y').lower() == 'y'
        if _ov:
            import svc_ints_orchestrator as _so
            _combo = f'{_infra}__{_svc}'
            _records = [r for t in ('ext', 'ndc')
                        for r in _so.read_records(t, network=_combo)]
            if not _records:
                print(f"  No svc-ints registered for combo '{_combo}'.")
            for _rec in _records:
                build_svc_int_factor_overrides(
                    str(_rec['int_id']), _svc, _infra, _method, _attr,
                    _n, _y0, _y1, use_cache=_cache)

    _val = (input("Run store validation? (y/n) [n]: ").strip()
            or 'n').lower() == 'y'
    if _val:
        validate_factor_store(_svc, _infra, _method, _attr, _n, _y0, _y1)