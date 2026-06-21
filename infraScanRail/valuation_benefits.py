"""
valuation_benefits — Phase 8A: monetised travel-time gains and losses per svc-int.
Last modified: 2026-06-21

Benefit measure = the 6C generalised-cost skim (gc_min of the least-GC path per
station pair, read from the routing primitive parquet), composed under the
active settings. Demand = Phase 7 factor-store compositions
(random_scenarios.compose_scenario_od). Per pair the rule-of-half contribution is
dGC x (q_base + q_dev) / 2; pairs that get faster contribute a GAIN, pairs that
get slower a LOSS. Decision 2026-06-21 (supervisor): the loss side is no longer
netted against the gain — it is reported as a separate travel-time-LOSS cost so
it lands on the CBA cost side, not as a smaller benefit. Especially relevant for
STP (an added stop lengthens the ride for through-passengers), but applies to any
int that slows some OD pairs. Annualisation = whole-day hours x 365 x VTTS.

Only pairs with a time in BOTH networks are valued; dropped demand pairs are
counted and printed per svc-int (no silent truncation).

Replaces the legacy compute_tts chain (TT_Delay.py / main_cap.py Phase 9):
no per-dev Dijkstra graphs, no Rail_Node.csv read, no 'Development_1'
convention, no scenario-OD pickles, no tau. Note on the output schema:
status_quo_tt / development_tt are true totals (sum gc x q / 60, hours/day).
tt_savings_daily stays the SIGNED net rule-of-half value (= tt_gain_daily -
tt_loss_daily, kept for reconciliation), but monetized_savings_yearly is now the
GROSS gain only (tt_gain_daily x 365 x VTTS) and monetized_tt_loss_yearly the
gross loss (a positive cost). Phase 9 consumes monetized_savings_yearly as the
benefit and monetized_tt_loss_yearly as a cost; gain - loss reproduces the old
net benefit exactly (NPV unchanged, only the cost composition / BCR shift).
"""

import os
import time

import numpy as np
import pandas as pd

import cache_manifest
import cost_parameters as cp
import paths
import random_scenarios as rs
import settings
import svc_ints_orchestrator as so


def compute_travel_time_savings(svc_int_ids=None, *, svc_version: str,
                                combo: str, method: str, attribution: str,
                                assignment_method: str, scenarios=None,
                                years=None, use_cache: bool = True) -> pd.DataFrame:
    """Phase 8A: monetised TTS per svc-int x scenario x year (rule of half).

    Args:
        svc_int_ids: svc-int ids to value (None = all registered ext + ndc).
        svc_version: base service version WITHOUT the '_network' suffix.
        combo: the '<infra>__<svc>' workspace key.
        method: OD method — 'pt_feeder' | 'municipal'.
        attribution: long-OD key ('specific' | 'blended' | 'municipal').
        assignment_method: routing method — 'shortest_path' | 'logit'.
        scenarios: iterable of 1-based scenario numbers
            (default 1..settings.amount_of_scenarios).
        years: iterable of valuation years
            (default settings.start_valuation_year..end_year_scenario).
        use_cache: load per-svc-int tts.parquet caches when present and the
            costs_8a manifest matches.

    Returns:
        DataFrame(development, scenario, year, status_quo_tt, development_tt,
        tt_savings_daily, tt_gain_daily, tt_loss_daily, monetized_savings_yearly,
        monetized_tt_loss_yearly) over all svc-ints — also written to
        data/costs/Developments/<combo>/traveltime_savings.csv.
    """
    t0 = time.time()
    if svc_int_ids is None:
        svc_int_ids = [i for t in so.SUPPORTED_SVC_INT_TYPES
                       for i in so.list_svc_int_ids(t, network=combo)]
    svc_int_ids = [str(i) for i in svc_int_ids]
    scenarios = list(scenarios if scenarios is not None
                     else range(1, int(settings.amount_of_scenarios) + 1))
    years = list(years if years is not None
                 else range(int(settings.start_valuation_year),
                            int(settings.end_year_scenario) + 1))
    svc_network = f'{svc_version}_network'
    out_dir = paths.get_costs_combo_dir(combo)
    versions = {'combo': combo, 'svc_network': svc_network,
                'assignment_method': assignment_method, 'method': method}

    print(f"\n=== Phase 8A — Travel-Time Savings ({len(svc_int_ids)} svc-int(s), "
          f"{len(scenarios)} scenario(s) x {len(years)} year(s)) ===")
    print(f"  combo: {combo} | method: {method} | attribution: {attribution} "
          f"| routing: {assignment_method} | VTTS: {cp.VTTS} CHF/h")
    if not svc_int_ids:
        print("  no svc-ints registered — nothing to do")
        return pd.DataFrame(columns=_OUT_COLS)

    manifest_ok = bool(use_cache) and cache_manifest.check_manifest(
        out_dir, 'costs_8a', versions, name='_settings_manifest_costs_8a.json')

    gc_base = _load_gc_skim(svc_network, assignment_method)
    print(f"  baseline gc skim: {len(gc_base):,} pairs")
    store_base = rs.load_factor_store(svc_version, method, attribution)

    ctx = {'svc_version': svc_version, 'combo': combo, 'method': method,
           'attribution': attribution, 'assignment_method': assignment_method,
           'scenarios': scenarios, 'years': years, 'gc_base': gc_base,
           'store_base': store_base, 'use_cache': use_cache,
           'manifest_ok': manifest_ok}

    n_jobs = max(1, int(getattr(settings, 'PHASE_PARALLEL_N_JOBS', 1)))
    if n_jobs == 1 or len(svc_int_ids) <= 1:
        results = [_tts_safe(iid, ctx) for iid in svc_int_ids]
    else:
        os.environ.setdefault('MPLBACKEND', 'Agg')
        from joblib import Parallel, delayed
        print(f"  [8A] parallel: {len(svc_int_ids)} svc-int(s) on {n_jobs} "
              f"loky worker(s)")
        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_tts_safe)(iid, ctx) for iid in svc_int_ids)
    frames = []
    for iid, fr, err in results:
        if err is not None:
            print(f"  [8A] {iid}: SKIPPED — {err}")
        elif fr is not None:
            frames.append(fr)
    if not frames:
        print("  [8A] no svc-int produced TTS rows — nothing written")
        return pd.DataFrame(columns=_OUT_COLS)

    result = pd.concat(frames, ignore_index=True)[_OUT_COLS]
    os.makedirs(out_dir, exist_ok=True)
    csv_path = paths.get_tts_csv(combo)
    result.to_csv(csv_path, index=False)
    cache_manifest.write_manifest(out_dir, 'costs_8a', versions,
                                  name='_settings_manifest_costs_8a.json')
    print(f"  [csv] wrote {csv_path} ({len(result):,} rows)")
    print(f"  Phase 8A done in {time.time() - t0:.1f}s")
    return result


_OUT_COLS = ['development', 'scenario', 'year', 'status_quo_tt',
             'development_tt', 'tt_savings_daily', 'tt_gain_daily',
             'tt_loss_daily', 'monetized_savings_yearly',
             'monetized_tt_loss_yearly']


def _load_gc_skim(network_name: str, assignment_method: str) -> pd.DataFrame:
    """Long-format gc skim (origin_id, dest_id, gc_min) of the least-GC path
    per pair, from the persisted routing primitive (the 4C/6C machine
    contract — the skims.xlsx workbook stores wide matrices and is
    human-facing only)."""
    p = paths.get_routing_primitive_path(network_name, assignment_method,
                                         'paths')
    if not os.path.exists(p):
        raise FileNotFoundError(f"routing primitive missing: {p}")
    pdf = pd.read_parquet(p)
    # path_id==0 is not unique per pair (gateway-injected service legs also
    # record path_id 0) — keep the least-GC row per pair.
    best = (pdf[pdf['path_id'] == 0]
            .sort_values('gc_min')
            .drop_duplicates(['origin_id', 'dest_id']))
    return best[['origin_id', 'dest_id', 'gc_min']].copy()


def _key(s: pd.Series) -> pd.Series:
    """Normalise station ids to merge-safe string keys ('8502202.0' -> '8502202')."""
    v = pd.to_numeric(s, errors='coerce')
    out = s.astype(str)
    m = v.notna()
    out[m] = v[m].astype('int64').astype(str)
    return out


def _align_indexer(pair_keys: pd.Series, od: pd.DataFrame) -> np.ndarray:
    """Positional indexer of each valued pair in an od_long frame (-1 = absent).

    compose_scenario_od scales a copy of the SAME od_long each call (row order
    preserved), so trips can be gathered positionally per scenario x year
    instead of re-merging 5'000+ times.
    """
    od_keys = _key(od['origin_station_id']) + '|' + _key(od['dest_station_id'])
    pos = pd.Series(np.arange(len(od_keys)), index=od_keys)
    pos = pos[~pos.index.duplicated(keep='first')]
    return pair_keys.map(pos).fillna(-1).astype(int).to_numpy()


def _gather(trips: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.zeros(len(idx), dtype=float)
    hit = idx >= 0
    out[hit] = trips[idx[hit]]
    return out


def _tts_safe(svc_int_id: str, ctx: dict) -> tuple:
    """Loky-dispatchable wrapper: (iid, frame|None, error|None). Module-level so
    it pickles by reference; FileNotFoundError (missing skim) skips, not crashes."""
    try:
        return svc_int_id, _tts_for_svc_int(svc_int_id, ctx), None
    except FileNotFoundError as exc:
        return svc_int_id, None, str(exc)


def _tts_for_svc_int(svc_int_id: str, ctx: dict) -> pd.DataFrame:
    """TTS rows for one svc-int (worker-safe: every decision arrives via ctx)."""
    cache_path = paths.get_tts_cache_path(svc_int_id, ctx['combo'])
    if ctx['use_cache'] and ctx['manifest_ok'] and os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        print(f"  [8A] {svc_int_id}: cached ({len(cached):,} rows)")
        return cached

    int_network = paths.svc_int_network_name(svc_int_id, ctx['combo'])
    gc_dev = _load_gc_skim(int_network, ctx['assignment_method'])
    gc_base = ctx['gc_base']

    pairs = gc_base.merge(gc_dev, on=['origin_id', 'dest_id'],
                          suffixes=('_base', '_dev'))
    pair_keys = _key(pairs['origin_id']) + '|' + _key(pairs['dest_id'])

    store_dev = rs.load_factor_store(ctx['svc_version'], ctx['method'],
                                     ctx['attribution'],
                                     svc_int_id=svc_int_id,
                                     combo=ctx['combo'])
    od_base, od_dev = ctx['store_base']['od_long'], store_dev['od_long']

    # Missing-pair report (8A decision 5): demand pairs without a time in both
    # skims are dropped from the valuation — print count + base-year volume.
    valued = set(pair_keys)
    for label, od in (('base', od_base), ('dev', od_dev)):
        keys = _key(od['origin_station_id']) + '|' + _key(od['dest_station_id'])
        miss = ~keys.isin(valued) & (od['trips'] > 0)
        if miss.any():
            print(f"  [8A] {svc_int_id}: {int(miss.sum()):,} {label} OD pair(s) "
                  f"without a time in both networks — dropped "
                  f"({od.loc[miss, 'trips'].sum():,.0f} base-year trips/day)")

    idx_b = _align_indexer(pair_keys, od_base)
    idx_d = _align_indexer(pair_keys, od_dev)
    gcb = pairs['gc_min_base'].to_numpy(dtype=float)
    gcd = pairs['gc_min_dev'].to_numpy(dtype=float)
    dgc = gcb - gcd

    kw = dict(svc_version=ctx['svc_version'], method=ctx['method'],
              attribution=ctx['attribution'], combo=ctx['combo'])
    rows = []
    for s in ctx['scenarios']:
        for y in ctx['years']:
            qb_od = rs.compose_scenario_od(None, s, y,
                                           store=ctx['store_base'], **kw)
            qd_od = rs.compose_scenario_od(svc_int_id, s, y,
                                           store=store_dev, **kw)
            qb = _gather(qb_od['trips'].to_numpy(dtype=float), idx_b)
            qd = _gather(qd_od['trips'].to_numpy(dtype=float), idx_d)
            # Gross-disbenefit split (decision 2026-06-21): faster pairs (dgc>0)
            # are a gain, slower pairs (dgc<0) a separate loss-cost, not netted.
            # (qb+qd)/2 >= 0, so sign(contrib) == sign(dgc).
            contrib = dgc * (qb + qd) / 2.0
            gain_daily = float(contrib[contrib > 0].sum()) / 60.0
            loss_daily = float(-contrib[contrib < 0].sum()) / 60.0
            rows.append({
                'development': svc_int_id, 'scenario': int(s), 'year': int(y),
                'status_quo_tt': float((gcb * qb).sum()) / 60.0,
                'development_tt': float((gcd * qd).sum()) / 60.0,
                'tt_savings_daily': gain_daily - loss_daily,
                'tt_gain_daily': gain_daily,
                'tt_loss_daily': loss_daily,
                'monetized_savings_yearly': gain_daily * 365.0 * float(cp.VTTS),
                'monetized_tt_loss_yearly': loss_daily * 365.0 * float(cp.VTTS),
            })
    out = pd.DataFrame(rows, columns=_OUT_COLS)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    out.to_parquet(cache_path, index=False)
    print(f"  [8A] {svc_int_id}: {len(pairs):,} valued pairs -> "
          f"{len(out):,} (scenario, year) rows  "
          f"[mean yearly gain {out['monetized_savings_yearly'].mean():,.0f} CHF, "
          f"loss {out['monetized_tt_loss_yearly'].mean():,.0f} CHF]")
    return out


if __name__ == "__main__":
    import ints_core as _core

    print("=== Phase 8A — Travel-Time Savings (standalone) ===")
    _svc = input(f"Service version [{settings.SVC_VERSION}]: ").strip() \
        or settings.SVC_VERSION
    if _svc == 'Build_New':
        _svc = settings.SVC_BUILD_NEW_NAME
    _default_combo = _core.default_combo(svc_version=_svc)
    _combo = input(f"Combo [{_default_combo}]: ").strip() or _default_combo
    _method = ('pt_feeder' if str(settings.CATCHMENT_METHOD).lower()
               == 'pt_feeder' else 'municipal')
    _attr = (settings.OD_ATTRIBUTION_MODE if _method == 'pt_feeder'
             else 'municipal')
    _assign = settings.ROUTING_ASSIGNMENT_METHOD
    _cache_def = 'y' if settings.use_cache_tts else 'n'
    _cache = (input(f"Use TTS caches? [y/n] [{_cache_def}]: ").strip()
              or _cache_def).lower() == 'y'
    compute_travel_time_savings(
        svc_version=_svc, combo=_combo, method=_method, attribution=_attr,
        assignment_method=_assign, use_cache=_cache)
