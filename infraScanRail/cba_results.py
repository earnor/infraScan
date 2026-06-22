"""
cba_results — Phase 9: CBA integration, discounting, aggregation, results.
Last modified: 2026-06-21

Combines the Phase 8 valuation inputs (8A traveltime_savings.csv, 8B
construction_cost.csv) into the discounted cost-benefit result, single track,
delta-only (no do-nothing row — benefits are skim deltas, costs per-svc-int
absolutes). MultiIndex (development, scenario, year) with STRING svc-int ids
(ext_100001, ndc_103001) and the columns const_cost / maint_cost /
uncovered_op_cost / tt_loss_cost / benefit. tt_loss_cost is the 8A gross
travel-time-LOSS (monetized_tt_loss_yearly) split off the benefit on 2026-06-21
(supervisor decision): it is demand-dependent (scenario-varying, like benefit),
follows the same per-year timing, and is discounted with it — so the net benefit
(benefit - all costs) is unchanged from the old netted result; only the BCR / cost
composition shift.

Discounting (corrected 2026-06-10): factor = 1/(1+r)^(year - base_year) with
base_year = start_valuation_year — present value at the start of the valuation
period, factor(base_year) = 1.0 exactly. The legacy off-by-one
(scoring.discounting's `- 1`, which inflated base-year values by x1.03) is
intentionally NOT ported.

Outputs (data/costs/Developments/<combo>/):
  - costs_and_benefits_discounted.csv  (svc-int x scenario x year)
  - total_costs_raw.csv                (svc-int x scenario, lean — no year grid)
  - total_costs.csv                    (wide per svc-int, per-scenario columns)
  - total_costs_summary.csv            (per svc-int scenario statistics)
Tables always write; Phase 9 has no cache (seconds of table work). Plots are
gated by make_plots (settings.PLOT_RESULTS).

Labels: placeholder EXT{n}/NDC{n} (per-type sequence number, 1:1 with the
registry id) pending the Svc-Int Expansion part-3 readable short-name scheme —
replace svc_int_label() wholesale when that lands.

Replaces the legacy main_cap Phase 11-13 chain (scoring.create_cost_and_benefit_df
/ discounting / aggregate_costs / transform_and_reshape_cost_df /
create_cost_summary); legacy files stay untouched as main_cap reference.
"""

import os
import textwrap
import time

import numpy as np
import pandas as pd

import cost_parameters as cp
import paths
import settings
import svc_ints_orchestrator as so


def run_cba(svc_int_ids=None, *, combo: str, base_infra: str = None,
            svc_version: str = None, discount_rate: float = None,
            base_year: int = None, make_plots: bool = False) -> pd.DataFrame:
    """Phase 9: discounted CBA tables + geometry + core result plots for one combo.

    Args:
        svc_int_ids: svc-int ids to evaluate (None = all ids covered by the
            Phase 8 tables).
        combo: the '<infra>__<svc>' workspace key.
        base_infra: base infra version the deltas were projected on
            (None = the combo's infra half).
        svc_version: service version (None = the combo's svc half).
        discount_rate: annual discount rate (None = cost_parameters.discount_rate).
        base_year: PV base year, factor 1.0 (None = settings.start_valuation_year).
        make_plots: core result plot set (data outputs always write).

    Returns:
        The total_costs_summary DataFrame (one row per svc-int).
    """
    t0 = time.time()
    base_infra = base_infra or combo.split('__')[0]
    svc_version = svc_version or combo.split('__', 1)[1]
    discount_rate = float(discount_rate if discount_rate is not None
                          else cp.discount_rate)
    base_year = int(base_year if base_year is not None
                    else settings.start_valuation_year)
    out_dir = paths.get_costs_combo_dir(combo)

    print(f"\n=== Phase 9 — CBA ({combo}) ===")
    print(f"  discount rate: {discount_rate:.1%} | PV base year: {base_year} "
          f"(factor 1.0)")

    benefits, costs, ids = _load_inputs(combo, svc_int_ids)
    if not ids:
        print("  [9] no svc-ints covered by both Phase 8 tables — nothing to do")
        return pd.DataFrame()
    scenarios = sorted(benefits['scenario'].unique())
    years = sorted(benefits['year'].unique())
    print(f"  [9] {len(ids)} svc-int(s) x {len(scenarios)} scenario(s) x "
          f"{len(years)} year(s) ({years[0]}-{years[-1]})")

    cb = _build_cost_benefit_frame(benefits, costs, ids, scenarios, years,
                                   base_year)
    cb_disc = _discount(cb, discount_rate, base_year)
    os.makedirs(out_dir, exist_ok=True)
    cb_path = paths.get_costs_and_benefits_discounted_csv(combo)
    cb_disc.to_csv(cb_path)
    print(f"  [csv] wrote {cb_path} ({len(cb_disc)} rows)")

    raw = _aggregate(cb_disc)
    raw_path = paths.get_total_costs_raw_csv(combo)
    raw.to_csv(raw_path, index=False)
    print(f"  [csv] wrote {raw_path} ({len(raw)} rows)")

    records = _load_records(ids, combo)
    wide = _reshape_wide(raw, records)
    wide_path = paths.get_total_costs_csv(combo)
    wide.to_csv(wide_path, index=False)
    print(f"  [csv] wrote {wide_path} ({len(wide)} svc-int row(s), "
          f"{len(wide.columns)} columns)")

    summary = _summary(wide, records)
    summary_path = paths.get_total_costs_summary_csv(combo)
    summary.to_csv(summary_path, index=False)
    print(f"  [csv] wrote {summary_path}")

    geo = _build_geometry(ids, combo, base_infra, records, summary)
    gpkg_path = paths.get_total_costs_geometry_gpkg(combo)
    geo.to_file(gpkg_path, driver='GPKG')
    print(f"  [gpkg] wrote {gpkg_path} ({geo.geometry.notna().sum()}/{len(geo)} "
          f"with geometry)")

    if make_plots:
        try:
            _make_result_plots(raw, cb_disc, records, geo, combo, base_infra,
                               svc_version)
        except Exception as exc:
            print(f"  [plot] Phase 9 result plots failed: {exc}")

    print(f"  Phase 9 done in {time.time() - t0:.1f}s")
    return summary


def svc_int_label(svc_int_id: str) -> str:
    """Placeholder readable label: EXT{n}/NDC{n} with n = the per-type sequence
    number (id minus the type's DEV_ID_START block — 1:1 with the registry).
    Superseded by the Svc-Int Expansion part-3 short-name scheme (S14_EXT5)."""
    sid = str(svc_int_id)
    try:
        int_type, num = sid.split('_', 1)
        start = {'ext': settings.DEV_ID_START_EXT,
                 'ndc': settings.DEV_ID_START_NDC,
                 'frq': settings.DEV_ID_START_FRQ,
                 'stp': settings.DEV_ID_START_STP}[int_type]
        return f"{int_type.upper()}{int(num) - start}"
    except (KeyError, ValueError):
        return sid


_CB_COLS = ['const_cost', 'maint_cost', 'uncovered_op_cost', 'tt_loss_cost',
            'benefit']


def _svc_int_type(svc_int_id: str) -> str:
    return str(svc_int_id).split('_')[0]


def _load_inputs(combo: str, svc_int_ids=None):
    """8A benefits + 8B costs; proceed on the id intersection (printed, never
    silent)."""
    tts_path = paths.get_tts_csv(combo)
    cost_path = paths.get_construction_cost_csv(combo)
    for p, phase in ((tts_path, '8A'), (cost_path, '8B')):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Phase {phase} output missing: {p} — "
                                    f"run Phase 8 first")
    benefits = pd.read_csv(tts_path)
    costs = pd.read_csv(cost_path)
    benefits['development'] = benefits['development'].astype(str)
    costs['Development'] = costs['Development'].astype(str)

    ben_ids = set(benefits['development'])
    cost_ids = set(costs['Development'])
    ids = ben_ids & cost_ids
    for missing, src in ((ben_ids - cost_ids, '8B costs'),
                         (cost_ids - ben_ids, '8A benefits')):
        if missing:
            print(f"  [9] WARNING: {sorted(missing)} missing from {src} — "
                  f"dropped from the CBA")
    if svc_int_ids is not None:
        requested = {str(i) for i in svc_int_ids}
        if requested - ids:
            print(f"  [9] WARNING: requested ids not covered by Phase 8: "
                  f"{sorted(requested - ids)}")
        ids &= requested
    ids = sorted(ids)
    benefits = benefits[benefits['development'].isin(ids)]
    costs = costs[costs['Development'].isin(ids)]
    return benefits, costs, ids


def _build_cost_benefit_frame(benefits, costs, ids, scenarios, years,
                              base_year: int) -> pd.DataFrame:
    """Full MultiIndex (development, scenario, year) frame, legacy semantics:
    benefit per (id, scenario, year); TotalConstructionCost at base_year only;
    YearlyMaintenanceCost + uncoveredOperatingCost (negative kept) broadcast to
    the years after it. tt_loss_cost (8A gross travel-time loss) is per (id,
    scenario, year) and follows the SAME all-years timing as benefit, so net =
    benefit - tt_loss_cost reproduces the old netted value year by year."""
    idx = pd.MultiIndex.from_product([ids, scenarios, years],
                                     names=['development', 'scenario', 'year'])
    cb = pd.DataFrame(0.0, index=idx, columns=_CB_COLS)

    by_pair = benefits.set_index(['development', 'scenario', 'year'])
    cb['benefit'] = (by_pair['monetized_savings_yearly']
                     .reindex(idx).fillna(0.0).to_numpy(dtype=float))
    cb['tt_loss_cost'] = (by_pair['monetized_tt_loss_yearly']
                          .reindex(idx).fillna(0.0).to_numpy(dtype=float))

    by_id = costs.set_index('Development')
    dev = idx.get_level_values('development')
    year = idx.get_level_values('year').to_numpy()
    const = dev.map(by_id['TotalConstructionCost']).to_numpy(dtype=float)
    maint = dev.map(by_id['YearlyMaintenanceCost']).to_numpy(dtype=float)
    op = dev.map(by_id['uncoveredOperatingCost']).to_numpy(dtype=float)
    cb['const_cost'] = np.where(year == base_year, const, 0.0)
    annual = year > base_year
    cb['maint_cost'] = np.where(annual, maint, 0.0)
    cb['uncovered_op_cost'] = np.where(annual, op, 0.0)
    return cb


def _discount(cb: pd.DataFrame, discount_rate: float,
              base_year: int) -> pd.DataFrame:
    """PV at base_year: factor = 1/(1+r)^(year - base_year), factor(base) = 1.0
    (the legacy `- 1` off-by-one is not ported)."""
    year = cb.index.get_level_values('year').to_numpy()
    factors = 1.0 / (1.0 + discount_rate) ** (year - base_year)
    out = cb.copy()
    for col in _CB_COLS:
        out[col] = out[col].to_numpy(dtype=float) * factors
    return out


def _aggregate(cb_disc: pd.DataFrame) -> pd.DataFrame:
    """Lean total_costs_raw: one row per (development, scenario), discounted
    sums over the valuation years + the alias/derived columns the plot ports
    read. Drops the legacy broadcast onto the year-grid template."""
    raw = (cb_disc.groupby(level=['development', 'scenario']).sum()
           .reset_index()
           .rename(columns={'const_cost': 'construction_cost',
                            'maint_cost': 'maintenance_cost',
                            'benefit': 'monetized_savings_total'}))
    raw['TotalConstructionCost'] = raw['construction_cost']
    raw['TotalMaintenanceCost'] = raw['maintenance_cost']
    raw['TotalUncoveredOperatingCost'] = raw['uncovered_op_cost']
    raw['TotalTtLossCost'] = raw['tt_loss_cost']
    raw['total_costs'] = (raw['construction_cost'] + raw['maintenance_cost']
                          + raw['uncovered_op_cost'] + raw['tt_loss_cost'])
    raw['total_net_benefit'] = raw['monetized_savings_total'] - raw['total_costs']
    # total_costs includes the SIGNED uncovered operating cost, so it can be 0
    # (STP, no CAP) or negative (EXT truncations) — the ratio is then meaningless
    # (inf or a misleading negative BCR). Guard to NaN; net_benefit stays the ranker.
    raw['cba_ratio'] = (raw['monetized_savings_total']
                        / raw['total_costs'].where(raw['total_costs'] > 0))
    return raw


def _load_records(ids, combo: str) -> dict:
    """Registry record per svc-int (labels/grouping); None when de-registered."""
    records = {}
    for iid in ids:
        rec = so.read_record(_svc_int_type(iid), iid, network=combo)
        if rec is None:
            print(f"  [9] {iid}: not in the 5B registry — label/grouping "
                  f"fall back to the id")
        records[iid] = rec
    return records


def _connection_label(record) -> str:
    """NDC missing-connection label (endpoint pair); '' for EXT/unknown."""
    if not record or record.get('int_type') != 'ndc':
        return ''
    stations = record.get('affected_stations') or []
    if len(stations) < 2:
        return ''
    return f"{stations[0]} – {stations[-1]}"


def _endpoints_label(record) -> str:
    """Human-readable endpoints: EXT 'endpoint → target', NDC 'first – last'."""
    if not record:
        return ''
    stations = record.get('affected_stations') or []
    if len(stations) < 2:
        return ''
    if record.get('int_type') in ('ext', 'frq', 'stp'):
        return f"{stations[0]} → {stations[-1]}"
    return f"{stations[0]} – {stations[-1]}"


def _reshape_wide(raw: pd.DataFrame, records: dict) -> pd.DataFrame:
    """Wide per-svc-int table (legacy transform_and_reshape_cost_df role):
    per scenario s the columns monetized_savings_total_{s} / tt_loss_cost_{s} /
    total_costs_scenario_{s} / Net_Benefit_scenario_{s} / cba_ratio_scenario_{s}.
    const/maint/op are scenario-invariant (one value per svc-int); the travel-time
    loss is demand-dependent, so it is kept per scenario and its mean is the
    displayed total_costs/Mio column. development stays the string id — no
    'Development_' prefix, no dev_id round-trip."""
    wide = raw.pivot_table(index='development', columns='scenario',
                           values='monetized_savings_total', aggfunc='first')
    wide.columns = [f"monetized_savings_total_{s}" for s in wide.columns]
    wide = wide.reset_index()

    loss_wide = raw.pivot_table(index='development', columns='scenario',
                                values='tt_loss_cost', aggfunc='first')
    loss_wide.columns = [f"tt_loss_cost_{s}" for s in loss_wide.columns]
    wide = wide.merge(loss_wide.reset_index(), on='development')

    costs_df = raw.groupby('development', as_index=False).agg(
        construction_cost=('construction_cost', 'first'),
        maintenance_cost=('maintenance_cost', 'first'),
        uncovered_op_cost=('uncovered_op_cost', 'first'))
    wide = wide.merge(costs_df, on='development')
    wide['fixed_costs'] = (wide['construction_cost'] + wide['maintenance_cost']
                           + wide['uncovered_op_cost'])
    loss_cols = [c for c in wide.columns if c.startswith('tt_loss_cost_')]
    wide['tt_loss_cost_mean'] = wide[loss_cols].mean(axis=1)
    wide['total_costs'] = wide['fixed_costs'] + wide['tt_loss_cost_mean']

    savings_cols = [c for c in wide.columns
                    if c.startswith('monetized_savings_total_')]
    derived = {}
    for col in savings_cols:
        s = col.rsplit('_', 1)[-1]
        tc_s = wide['fixed_costs'] + wide[f"tt_loss_cost_{s}"]
        derived[f"total_costs_scenario_{s}"] = tc_s
        derived[f"Net_Benefit_scenario_{s}"] = wide[col] - tc_s
        # Guard zero/negative total_costs (signed operating cost) -> NaN, not inf.
        derived[f"cba_ratio_scenario_{s}"] = wide[col] / tc_s.where(tc_s > 0)
    derived['Construction Cost [in Mio. CHF]'] = wide['construction_cost'] / 1e6
    derived['Maintenance Costs [in Mio. CHF]'] = wide['maintenance_cost'] / 1e6
    derived['Uncovered Operating Costs [in Mio. CHF]'] = (
        wide['uncovered_op_cost'] / 1e6)
    derived['Travel Time Loss Cost [in Mio. CHF]'] = (
        wide['tt_loss_cost_mean'] / 1e6)
    wide = pd.concat([wide, pd.DataFrame(derived, index=wide.index)], axis=1)

    wide.insert(1, 'label', wide['development'].map(svc_int_label))
    wide.insert(2, 'int_type', wide['development'].map(_svc_int_type))

    front = ['development', 'label', 'int_type',
             'Construction Cost [in Mio. CHF]',
             'Maintenance Costs [in Mio. CHF]',
             'Uncovered Operating Costs [in Mio. CHF]',
             'Travel Time Loss Cost [in Mio. CHF]', 'total_costs']
    order = front + [c for c in wide.columns if c not in front]
    return wide[order]


def _summary(wide: pd.DataFrame, records: dict) -> pd.DataFrame:
    """Per-svc-int scenario statistics (legacy create_cost_summary role):
    mean/min/max/std savings across scenarios, net benefit + BCR on the mean."""
    savings = wide[[c for c in wide.columns
                    if c.startswith('monetized_savings_total_')]]
    out = pd.DataFrame({
        'development': wide['development'],
        'label': wide['label'],
        'int_type': wide['int_type'],
        'connection': wide['development'].map(
            lambda iid: _connection_label(records.get(iid))),
        'Construction Cost [in Mio. CHF]':
            wide['Construction Cost [in Mio. CHF]'],
        'Maintenance Costs [in Mio. CHF]':
            wide['Maintenance Costs [in Mio. CHF]'],
        'Uncovered Operating Costs [in Mio. CHF]':
            wide['Uncovered Operating Costs [in Mio. CHF]'],
        'Travel Time Loss Cost [in Mio. CHF]':
            wide['Travel Time Loss Cost [in Mio. CHF]'],
        'Total Costs [in Mio. CHF]': wide['total_costs'] / 1e6,
        'Monetized Savings Mean [in Mio. CHF]': savings.mean(axis=1) / 1e6,
        'Monetized Savings Min [in Mio. CHF]': savings.min(axis=1) / 1e6,
        'Monetized Savings Max [in Mio. CHF]': savings.max(axis=1) / 1e6,
        'Monetized Savings Std [in Mio. CHF]': savings.std(axis=1) / 1e6,
    })
    out['Net Benefit [in Mio. CHF]'] = (
        out['Monetized Savings Mean [in Mio. CHF]']
        - out['Total Costs [in Mio. CHF]'])
    # Guard zero/negative total_costs (signed operating cost) -> NaN, not inf.
    _tc_mio = out['Total Costs [in Mio. CHF]']
    out['CBA Ratio'] = (out['Monetized Savings Mean [in Mio. CHF]']
                        / _tc_mio.where(_tc_mio > 0))
    num_cols = [c for c in out.columns if '[in Mio. CHF]' in c] + ['CBA Ratio']
    out[num_cols] = out[num_cols].round(2)
    return out


def _merge_lines(geoms):
    """Union line geometries into one (Multi)LineString; None when empty."""
    from shapely.geometry import LineString
    from shapely.ops import linemerge, unary_union

    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return None
    u = unary_union(geoms)
    if isinstance(u, LineString):
        return u
    return linemerge(u)


def _build_geometry(ids, combo: str, base_infra: str, records: dict,
                    summary: pd.DataFrame):
    """Decision-2 result layer: per svc-int the projected service delta
    (changed/added legs) unioned with the CC arcs it requires (5A registry
    'segments' layer). CAP sidings excluded by construction. Rows without a
    delta on disk keep their attributes with empty geometry."""
    import fiona
    import geopandas as gpd

    cc_path = paths.get_infra_int_registry('cc', combo)
    if os.path.exists(cc_path):
        cc_segs = gpd.read_file(cc_path, layer='segments')
        cc_segs['int_id'] = cc_segs['int_id'].astype(str)
    else:
        print(f"  [9] no CC registry at {cc_path} — geometries are delta-only")
        cc_segs = gpd.GeoDataFrame(columns=['int_id', 'geometry'])

    geoms = []
    for iid in ids:
        parts = []
        delta_path = paths.get_svc_int_projected_path(iid, base_infra, combo)
        if os.path.exists(delta_path):
            for layer in fiona.listlayers(delta_path):
                parts.extend(gpd.read_file(delta_path, layer=layer).geometry)
        else:
            print(f"  [9] {iid}: no projected delta at {delta_path} — "
                  f"empty geometry")
        required = [str(r) for r in
                    ((records.get(iid) or {}).get('requires_infra') or [])]
        if required and not cc_segs.empty:
            parts.extend(cc_segs.loc[cc_segs['int_id'].isin(required),
                                     'geometry'])
        geoms.append(_merge_lines(parts))

    attrs = summary.copy()
    attrs.insert(4, 'endpoints', attrs['development'].map(
        lambda iid: _endpoints_label(records.get(iid))))
    return gpd.GeoDataFrame(attrs, geometry=geoms, crs='EPSG:2056')


# ─────────────────────────────────────────────────────────────────────────────
# Result plots (PLOT_RESULTS) — legacy looks ported as-is from plots.py
# (create_and_save_plots / plot_cumulative_cost_distribution /
# plot_costs_benefits); string-id type splits, placeholder labels. The legacy
# .abs() on monetized_savings_total is NOT ported (signed savings kept).
# ─────────────────────────────────────────────────────────────────────────────

_COST_COLORS = {'TotalConstructionCost': '#a6bddb',
                'TotalMaintenanceCost': '#3690c0',
                'TotalUncoveredOperatingCost': '#034e7b',
                'TotalTtLossCost': '#fc8d59'}
_CHART_SUFFIXES = ['boxplot_savings', 'violinplot_savings',
                   'boxplot_net_benefit', 'boxplot_cba', 'cost_savings',
                   'cumulative_cost_distribution']


def _seaborn_pandas_compat() -> None:
    """seaborn 0.12 wraps categorical plots in
    pd.option_context('mode.use_inf_as_na', ...), an option pandas 3.x removed
    — re-register it as an inert bool so those calls run. No behaviour change
    for finite data (the option only ever masked +-inf as NA)."""
    import pandas._config.config as pcfg
    try:
        pd.get_option('mode.use_inf_as_na')
    except Exception:
        pcfg.register_option('mode.use_inf_as_na', False,
                             'compat shim for seaborn 0.12 on pandas 3.x',
                             validator=pcfg.is_bool)


def _sanitize(name: str) -> str:
    import re
    return re.sub(r'[^\w\-]+', '_', str(name)).strip('_')


def _frq_flavour(record) -> str:
    """FRQ flavour: 'double' (a set_frequency op) vs 'homogenise' (corridor extend)."""
    ops = [o.get('op') for o in ((record or {}).get('operations') or [])]
    return 'double' if 'set_frequency' in ops else 'homogenise'


def _sa_stops_by_route(base_infra: str, svc_version: str) -> dict:
    """{route_id: frozenset(study-area calling-stop node Numbers)} from the base supply
    — the corridor signature for FRQ/STP grouping."""
    from capacity_calculator import _extract_sa_node_set, load_projected_services
    sa = _extract_sa_node_set(base_infra)
    links = load_projected_services(svc_version, base_infra)
    out = {}
    for svc, g in links.groupby(links['service'].astype(str)):
        stops = (set(g['from_stop_nr'].astype(int))
                 | set(g['to_stop_nr'].astype(int))) & sa
        out[str(svc)] = frozenset(stops)
    return out


def _assign_corridor(record, sa_stops_by_route: dict) -> str:
    """Corridor label for a FRQ/STP line: the settings.RESULTS_CORRIDOR_SPINES entry it
    shares the most SA stops with; ties resolve to the first-listed corridor (dict order).
    '' when no spine overlaps (caller buckets these as 'other')."""
    spines = getattr(settings, 'RESULTS_CORRIDOR_SPINES', {}) or {}
    if not record or not spines:
        return ''
    route = str(record.get('route_id')
                or (record.get('affected_services') or [''])[0])
    stops = sa_stops_by_route.get(route, frozenset())
    best, best_n = '', 0
    for label, spine in spines.items():
        n = len(stops & set(spine))
        if n > best_n:
            best, best_n = label, n
    return best


def _make_result_plots(raw: pd.DataFrame, cb_disc: pd.DataFrame, records: dict,
                       geo, combo: str, base_infra: str,
                       svc_version: str) -> None:
    """The PLOT_RESULTS core set, uniform across all four svc-int types: per-type
    semantic chart families (EXT by parent line + by candidate; NDC by connecting
    curve; FRQ by flavour x corridor; STP by corridor), per-type ranked families,
    a cross-type top-N 'best interventions' overview, overview + per-group network
    maps, combined chart+map images, overall cumulative distribution, and per-svc-int
    discounted waterfalls. Corridors/top-N from settings.RESULTS_CORRIDOR_SPINES /
    RESULTS_TOP_N."""
    import matplotlib
    matplotlib.use('Agg')
    _seaborn_pandas_compat()

    base_dir = paths.get_developments_plot_dir(combo, 'results')
    benefits_dir = os.path.join(base_dir, 'Benefits')
    combined_dir = os.path.join(base_dir, 'Benefits_Combined')
    ranked_dir = os.path.join(base_dir, 'Benefits_Ranked')
    ranked_combined_dir = os.path.join(ranked_dir, 'combined')
    maps_dir = os.path.join(base_dir, 'maps')
    waterfall_dir = os.path.join(base_dir, 'waterfall')
    for d in (benefits_dir, combined_dir, ranked_dir, ranked_combined_dir,
              maps_dir, waterfall_dir):
        os.makedirs(d, exist_ok=True)

    data = raw.copy()
    data['line_name'] = data['development'].map(svc_int_label)
    geom_by_label = {svc_int_label(iid): g
                     for iid, g in zip(geo['development'], geo.geometry)}
    aff_by_label = {svc_int_label(iid): (records.get(iid) or {}).get('affected_stations')
                    for iid in geo['development']}
    map_ctx = _load_map_context(base_infra, svc_version)

    def _charts_map_combined(selected, prefix, chart_dir, out_combined_dir):
        _plot_basic_charts(selected, prefix, chart_dir)
        labels = selected['line_name'].unique().tolist()
        map_path = os.path.join(maps_dir, f"railway_lines_{prefix}.png")
        _plot_network_map(labels, geom_by_label, _type_colors(labels),
                          aff_by_label, map_path, map_ctx,
                          prefix.replace('_', ' '), zoom_to_group=True)
        for suffix in _CHART_SUFFIXES:
            _combine_images(
                os.path.join(chart_dir, f"{prefix}_{suffix}.png"), map_path,
                os.path.join(out_combined_dir, f"{prefix}_{suffix}_combined.png"))

    ext = data[data['development'].map(_svc_int_type) == 'ext']
    ndc = data[data['development'].map(_svc_int_type) == 'ndc']
    frq = data[data['development'].map(_svc_int_type) == 'frq']
    stp = data[data['development'].map(_svc_int_type) == 'stp']

    sa_stops = _sa_stops_by_route(base_infra, svc_version)

    def _devs(sub):
        return list(dict.fromkeys(sub['development']))

    # Semantic grouped chart families (Benefits/ charts + Benefits_Combined/ chart+map),
    # one prefix per group. EXT carries two grouping dimensions (parent line + candidate).
    families: dict = {}
    for dev in _devs(ext):
        rec = records.get(dev) or {}
        line = str(rec.get('route_id') or '?')
        cand = _sanitize((rec.get('affected_stations') or ['?'])[-1])
        families.setdefault(f"EXT_line_{line}", []).append(dev)
        families.setdefault(f"EXT_to_{cand}", []).append(dev)
    for dev in _devs(ndc):
        cc = str(((records.get(dev) or {}).get('requires_infra') or ['none'])[0])
        families.setdefault(f"NDC_{cc}", []).append(dev)
    for dev in _devs(frq):
        rec = records.get(dev) or {}
        corr = _sanitize(_assign_corridor(rec, sa_stops) or 'other')
        families.setdefault(f"FRQ_{_frq_flavour(rec)}_{corr}", []).append(dev)
    for dev in _devs(stp):
        corr = _sanitize(_assign_corridor(records.get(dev), sa_stops) or 'other')
        families.setdefault(f"STP_{corr}", []).append(dev)

    print(f"  [plot] {len(families)} grouped chart families "
          f"(Benefits/ + Benefits_Combined/)...")
    for prefix, devs in families.items():
        sub = data[data['development'].isin(devs)]
        if not sub.empty:
            _charts_map_combined(sub, prefix, benefits_dir, combined_dir)

    # Uniform per-type ranked family — every type, same set (Benefits_Ranked/).
    print("  [plot] per-type ranked families (Benefits_Ranked/)...")
    for sub, tag in ((ext, 'ext'), (ndc, 'ndc'), (frq, 'frq'), (stp, 'stp')):
        if not sub.empty:
            _charts_map_combined(sub, f"ranked_group_{tag}", ranked_dir,
                                 ranked_combined_dir)

    # Cross-type 'best interventions' overview — top-N by mean net benefit, any type.
    ranked_all = (data.groupby('development')['total_net_benefit'].mean()
                  .sort_values(ascending=False).index.tolist())
    for n in getattr(settings, 'RESULTS_TOP_N', [5, 10]):
        top = data[data['development'].isin(ranked_all[:n])]
        if not top.empty:
            print(f"  [plot] cross-type top-{n} overview...")
            _charts_map_combined(top, f"top_{n}", ranked_dir, ranked_combined_dir)

    print("  [plot] overview network maps...")
    for sub, name in ((ext, 'developments_ext'), (ndc, 'developments_ndc'),
                      (frq, 'developments_frq'), (stp, 'developments_stp')):
        if sub.empty:
            continue
        labels = sub['line_name'].unique().tolist()
        _plot_network_map(labels, geom_by_label, _type_colors(labels),
                          aff_by_label, os.path.join(maps_dir, f'{name}.png'),
                          map_ctx, name.replace('_', ' '), zoom_to_group=False)

    _plot_cumulative(data,
                     os.path.join(base_dir, 'cumulative_cost_distribution.png'),
                     group_by='line_name')

    devs = sorted(set(cb_disc.index.get_level_values('development')))
    print(f"  [plot] {len(devs)} discounted waterfalls...")
    for iid in devs:
        _plot_waterfall(cb_disc, iid, waterfall_dir,
                        _dev_label(iid, records.get(iid), combo, base_infra))

    print(f"  [plot] {len(devs)} per-svc-int factsheets...")
    name_ctx = _load_name_context(combo, base_infra)
    for iid in devs:
        sub = geo[geo['development'] == iid]
        if sub.empty:
            continue
        try:
            _plot_factsheet(iid, sub.iloc[0], records.get(iid), combo,
                            base_infra, svc_version, waterfall_dir, name_ctx,
                            paths.get_factsheet_path(combo, iid))
        except Exception as exc:
            print(f"  [plot] factsheet {iid} failed: {exc}")
    print(f"  [plot] result plots saved under {base_dir}")


def _load_map_context(base_infra: str, svc_version: str) -> dict:
    """Map base layers (5B style), loaded once and clipped to the CATCHMENT area:
    lakes, the projected base service network (grey rail backdrop) and the
    version's nodes.gpkg, exposed as Name->Code and Name->(x, y) lookups for the
    affected-station code labels."""
    import fiona
    import geopandas as gpd
    import catchment_allocate

    boundary = gpd.GeoDataFrame(geometry=[catchment_allocate._load_catchment_boundary()],
                                crs='EPSG:2056')
    lakes = gpd.clip(gpd.read_file(paths.LAKES_SHP).to_crs('EPSG:2056'), boundary)

    rail_path = paths.get_projected_services_path(svc_version, base_infra)
    rail = pd.concat([gpd.read_file(rail_path, layer=layer)[['geometry']]
                      for layer in fiona.listlayers(rail_path)])
    rail = gpd.clip(gpd.GeoDataFrame(rail, crs='EPSG:2056'), boundary)

    nodes = gpd.read_file(os.path.join(paths.get_infra_version_dir(base_infra),
                                       'nodes.gpkg')).to_crs('EPSG:2056')
    return {'boundary': boundary, 'lakes': lakes, 'rail': rail,
            'name2code': dict(zip(nodes['Name'], nodes['Code'])),
            'name2xy': {n: (g.x, g.y) for n, g in zip(nodes['Name'], nodes.geometry)}}


def _type_colors(labels) -> dict:
    """Per-svc-int-type colour map matching Phase 5B (svc_ints_orchestrator
    _TYPE_COLOR): the label's 3-letter prefix (EXT1 -> ext) keys the colour."""
    return {label: so._TYPE_COLOR.get(str(label)[:3].lower(), '#000000')
            for label in labels}


_MAP_LAKE_FC, _MAP_LAKE_EC, _MAP_BACKDROP = '#c8e8f5', '#99c4d8', '#d4d4d4'


def _plot_network_map(labels, geom_by_label: dict, color_dict: dict,
                      aff_by_label: dict, output_path: str, map_ctx: dict,
                      title: str, zoom_to_group: bool = False,
                      figsize: tuple = (13, 11)) -> None:
    """Network map in the Phase-5B style over the whole catchment area: grey rail
    backdrop, lakes, dashed catchment boundary, intervention geometries coloured
    by svc-int type, affected-station CODES (no full station names), shared
    north-arrow + scale bar, bold title, upper-right legend."""
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import catchment_allocate
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=figsize)
    if not map_ctx['lakes'].empty:
        map_ctx['lakes'].plot(ax=ax, color=_MAP_LAKE_FC, edgecolor=_MAP_LAKE_EC,
                              linewidth=0.3, zorder=1)
    map_ctx['rail'].plot(ax=ax, color=_MAP_BACKDROP, linewidth=0.5, zorder=2)
    map_ctx['boundary'].boundary.plot(ax=ax, color='black', linewidth=1.0,
                                      linestyle='--', alpha=0.6, zorder=3)

    plotted, affected = [], []
    for label in labels:
        geom = geom_by_label.get(label)
        if geom is None or geom.is_empty:
            continue
        gpd.GeoSeries([geom], crs='EPSG:2056').plot(
            ax=ax, color=color_dict.get(label, 'black'), linewidth=3.0, zorder=5)
        plotted.append(label)
        affected += list(aff_by_label.get(label) or [])

    if zoom_to_group and plotted:
        bounds = gpd.GeoSeries([geom_by_label[label] for label in plotted],
                               crs='EPSG:2056').total_bounds
        pad = 2000
        ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
        ax.set_ylim(bounds[1] - pad, bounds[3] + pad)
    else:
        b = map_ctx['boundary'].total_bounds
        ax.set_xlim(b[0], b[2])
        ax.set_ylim(b[1], b[3])

    # Affected-station codes only (white/black marker + code, no full station names).
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    for nm in dict.fromkeys(affected):
        xy = map_ctx['name2xy'].get(nm)
        code = map_ctx['name2code'].get(nm)
        if not xy or not code or not (x0 <= xy[0] <= x1 and y0 <= xy[1] <= y1):
            continue
        ax.scatter([xy[0]], [xy[1]], s=22, c='white', edgecolors='black',
                   linewidths=0.8, marker='o', zorder=6)
        ax.annotate(code, xy=xy, ha='center', va='bottom', xytext=(0, 5),
                    textcoords='offset points', fontsize=8, fontweight='bold',
                    color='black', zorder=7)

    ax.set_xlabel('E [m]', fontsize=10)
    ax.set_ylabel('N [m]', fontsize=10)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=13, fontweight='bold')
    catchment_allocate._add_map_elements(ax)

    handles = [Patch(facecolor=_MAP_LAKE_FC, edgecolor=_MAP_LAKE_EC,
                     label='Water bodies'),
               Line2D([0], [0], color=_MAP_BACKDROP, lw=1.5, label='Rail network'),
               Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                      markeredgecolor='black', markersize=6, label='Affected station')]
    handles += [Line2D([0], [0], color=color_dict.get(label, 'black'), lw=3,
                       label=label) for label in plotted]
    ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _combine_images(chart_path: str, map_path: str, combined_path: str) -> None:
    """Horizontal chart+map pairing (legacy PIL flow, plots.py:2376)."""
    from PIL import Image

    if not os.path.exists(chart_path) or not os.path.exists(map_path):
        print(f"  [plot] combine skipped (missing input): {combined_path}")
        return
    chart_image = Image.open(chart_path)
    map_image = Image.open(map_path)
    target_height = max(map_image.height, chart_image.height)

    def resize_to_height(img, target_h):
        w, h = img.size
        return img.resize((int(w * (target_h / h)), target_h), Image.LANCZOS)

    map_resized = resize_to_height(map_image, target_height)
    chart_resized = resize_to_height(chart_image, target_height)
    combined = Image.new('RGB',
                         (map_resized.width + chart_resized.width,
                          target_height), (255, 255, 255))
    combined.paste(map_resized, (0, 0))
    combined.paste(chart_resized, (map_resized.width, 0))
    combined.save(combined_path)


def _plot_basic_charts(data: pd.DataFrame, filename_prefix: str,
                       plot_directory: str, line_colors: dict = None) -> dict:
    """Six charts per group (legacy plot_basic_charts, plots.py:2046): boxplot/
    violin savings, net-benefit boxplot, BCR boxplot, stacked costs-vs-benefits,
    cumulative distribution. Group key = line_name (the svc-int label)."""
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import seaborn as sns

    import plot_parameter as pp

    order = (data.groupby('line_name')['total_net_benefit'].mean()
             .sort_values(ascending=False).index.tolist())
    n_lines = len(order)
    if line_colors is None:
        line_colors = {name: pp.zvv_colors[i % len(pp.zvv_colors)]
                       for i, name in enumerate(order)}
    colors = [line_colors[line] for line in order]

    def _boxplot(y, ylabel, hline, fname):
        plt.figure(figsize=(7, 5), dpi=300)
        ax = sns.boxplot(data=data, x='line_name', y=y, order=order,
                         palette=colors, width=0.4, linewidth=0.8,
                         showmeans=True,
                         meanprops={"marker": "o", "markerfacecolor": "black",
                                    "markeredgecolor": "black", "markersize": 5},
                         fliersize=3, showfliers=True)
        ax.set_xlim(-0.5, n_lines - 0.5)
        plt.xlabel('Line', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        if hline is not None:
            plt.axhline(y=hline, color='red', linestyle='-', alpha=0.5)
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        handles = [mlines.Line2D([0], [0], marker='o', color='black',
                                 label='Mean', markersize=5),
                   mpatches.Patch(color=colors[0], label='Line Colour')]
        plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1),
                   frameon=False)
        plt.tight_layout(rect=[0, 0, 0.95, 1])
        plt.savefig(os.path.join(plot_directory,
                                 f"{filename_prefix}_{fname}.png"), dpi=600)
        plt.close()

    _boxplot(data['monetized_savings_total'] / 1e6,
             'Monetised travel time savings in million CHF', None,
             'boxplot_savings')
    _boxplot(data['total_net_benefit'] / 1e6, 'Net benefit in CHF million', 0,
             'boxplot_net_benefit')
    _boxplot(data['cba_ratio'], 'Cost-benefit ratio', 1, 'boxplot_cba')

    # Violin + scenario stripplot
    plt.figure(figsize=(7, 5), dpi=300)
    ax = sns.violinplot(data=data, x='line_name',
                        y=data['monetized_savings_total'] / 1e6, order=order,
                        palette=colors, width=0.7, inner=None, linewidth=0.8,
                        cut=0, scale='width')
    unique_data = data.drop_duplicates(
        subset=['line_name', 'scenario', 'monetized_savings_total'])
    sns.stripplot(data=unique_data, x='line_name',
                  y=unique_data['monetized_savings_total'] / 1e6, order=order,
                  color='black', alpha=0.4, jitter=True, size=2, dodge=False)
    ax.set_xlim(-0.5, n_lines - 0.5)
    plt.xlabel('Line', fontsize=12)
    plt.ylabel('Monetised travel time savings in million CHF', fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    handles = [mlines.Line2D([], [], marker='o', color='black', alpha=0.4,
                             linestyle='None', markersize=3,
                             label='Individual Values'),
               mpatches.Patch(color=colors[0], label='Line Colour')]
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1),
               frameon=False)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(os.path.join(plot_directory,
                             f"{filename_prefix}_violinplot_savings.png"),
                dpi=600)
    plt.close()

    # Stacked costs-vs-benefits (means per line)
    grouped = data.groupby('line_name').agg(
        {'TotalConstructionCost': 'mean', 'TotalMaintenanceCost': 'mean',
         'TotalUncoveredOperatingCost': 'mean', 'TotalTtLossCost': 'mean',
         'monetized_savings_total': 'mean'}).loc[order]
    x_pos = np.arange(n_lines)
    bar_width = 0.6
    plt.figure(figsize=(7, 5), dpi=300)
    plt.bar(x_pos, -grouped['TotalConstructionCost'] / 1e6, width=bar_width,
            color=_COST_COLORS['TotalConstructionCost'],
            label='Construction costs')
    plt.bar(x_pos, -grouped['TotalMaintenanceCost'] / 1e6, width=bar_width,
            bottom=-grouped['TotalConstructionCost'] / 1e6,
            color=_COST_COLORS['TotalMaintenanceCost'],
            label='Uncovered maintenance costs')
    plt.bar(x_pos, -grouped['TotalUncoveredOperatingCost'] / 1e6,
            width=bar_width,
            bottom=-(grouped['TotalConstructionCost']
                     + grouped['TotalMaintenanceCost']) / 1e6,
            color=_COST_COLORS['TotalUncoveredOperatingCost'],
            label='Uncovered operating costs')
    plt.bar(x_pos, -grouped['TotalTtLossCost'] / 1e6, width=bar_width,
            bottom=-(grouped['TotalConstructionCost']
                     + grouped['TotalMaintenanceCost']
                     + grouped['TotalUncoveredOperatingCost']) / 1e6,
            color=_COST_COLORS['TotalTtLossCost'], label='Travel time loss')
    for i, line_name in enumerate(order):
        plt.bar(x_pos[i], grouped.loc[line_name, 'monetized_savings_total'] / 1e6,
                width=bar_width, color=line_colors[line_name], hatch='////',
                edgecolor='black')
    plt.axhline(y=0, color='black', linestyle='-')
    plt.xticks(x_pos, order, rotation=90)
    plt.xlabel('Line', fontsize=12)
    plt.ylabel('Value in CHF million', fontsize=12)
    plt.title('Costs and benefits per modification', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    handles = [
        mpatches.Patch(color=_COST_COLORS['TotalConstructionCost'],
                       label='Construction costs'),
        mpatches.Patch(color=_COST_COLORS['TotalMaintenanceCost'],
                       label='Uncovered maintenance costs'),
        mpatches.Patch(color=_COST_COLORS['TotalUncoveredOperatingCost'],
                       label='Uncovered operating costs'),
        mpatches.Patch(color=_COST_COLORS['TotalTtLossCost'],
                       label='Travel time loss'),
        mpatches.Patch(facecolor="none", hatch='////', edgecolor='black',
                       label='Travel time savings')]
    plt.legend(handles=handles, bbox_to_anchor=(1.01, 1))
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(os.path.join(plot_directory,
                             f"{filename_prefix}_cost_savings.png"), dpi=600)
    plt.close()

    _plot_cumulative(
        data,
        os.path.join(plot_directory,
                     f"{filename_prefix}_cumulative_cost_distribution.png"),
        color_dict=line_colors, group_by='line_name')
    return line_colors


def _plot_cumulative(df: pd.DataFrame, output_path: str, color_dict: dict = None,
                     group_by: str = 'line_name') -> None:
    """Cumulative probability of monetised savings per group (legacy
    plot_cumulative_cost_distribution, plots.py:3620)."""
    import matplotlib.pyplot as plt

    mean_by_dev = (df.groupby(group_by)['monetized_savings_total'].mean()
                   .sort_values(ascending=False))
    dev_ids_sorted = mean_by_dev.index.tolist()

    plt.figure(figsize=(10, 6))
    if color_dict is None:
        cmap = plt.get_cmap('tab20')
        color_dict = {dev_id: cmap(i % 20)
                      for i, dev_id in enumerate(dev_ids_sorted)}

    for i, dev_id in enumerate(dev_ids_sorted):
        values = np.sort(df.loc[df[group_by] == dev_id,
                                'monetized_savings_total'].dropna().values / 1e6)
        if not len(values):
            continue
        y_values = np.arange(1, len(values) + 1) / len(values)
        plt.plot(values, y_values, '-',
                 color=color_dict.get(dev_id, f"C{i % 10}"), linewidth=2,
                 label=f"Line {dev_id}: {values.mean():.1f} Mio. CHF")

    plt.axvline(x=0, color='lightgray', linestyle='-', linewidth=1)
    plt.xlabel('Monetised travel time savings [CHF million]', fontsize=12)
    plt.ylabel('Cumulative probability', fontsize=12)
    plt.title('Cumulative probability distribution of the net benefit \n'
              'of all developments considered', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Lines with moderate utility', fontsize=9,
               title_fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0, 1.05)
    x_min = df['monetized_savings_total'].min() / 1e6
    x_max = df['monetized_savings_total'].max() / 1e6
    plt.xlim(x_min - 5, x_max + 5)
    for q in [0.25, 0.5, 0.75]:
        plt.axhline(y=q, color='darkgray', linestyle=':', alpha=0.7)
        plt.text(x_max + 3, q, f'{int(q * 100)}%', va='center', fontsize=9,
                 color='darkgray')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


_WF_FONT = 'serif'


def _line_terminals(svc_int_id: str, combo: str, base_infra: str):
    """(start, end) terminal stop names of the svc-int's line, read from its
    projected delta (direction 0: the only-source / only-sink stop). None when
    the line is not a simple path."""
    from pathlib import Path
    seg = so._load_delta_segments(
        Path(paths.get_svc_int_network_dir(svc_int_id, combo)) / base_infra
        / 'rail_segments.gpkg')
    if seg is None or seg.empty or 'direction_id' not in seg.columns:
        return None
    d0 = seg[seg['direction_id'].astype(str).isin(('0', '0.0'))]
    if d0.empty:
        d0 = seg
    froms, tos = set(d0['from_stop_name']), set(d0['to_stop_name'])
    start, end = froms - tos, tos - froms
    if len(start) == 1 and len(end) == 1:
        return next(iter(start)), next(iter(end))
    return None


def _dev_label(svc_int_id: str, record, combo: str, base_infra: str) -> str:
    """Readable dev label '<line_short_name>: <A> - <B>'. ext/stp use the affected
    stations (first/last); ndc/frq use the line's route terminals."""
    rec = record or {}
    short = rec.get('line_short_name') or svc_int_id
    stops = rec.get('affected_stations') or []
    if rec.get('int_type') in ('ndc', 'frq'):
        term = _line_terminals(svc_int_id, combo, base_infra)
        if term:
            return f"{short}: {term[0]} - {term[1]}"
    if len(stops) >= 2:
        return f"{short}: {stops[0]} - {stops[-1]}"
    return short


def _plot_waterfall(cb_disc: pd.DataFrame, svc_int_id: str, output_dir: str,
                    label: str) -> None:
    """Per-svc-int discounted cost/benefit bars over the valuation years,
    scenario means (legacy plot_costs_benefits, plots.py:2699)."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import FuncFormatter

    plot_data = (cb_disc.xs(svc_int_id, level='development')
                 .groupby('year').mean())
    plot_data['const_cost_neg'] = -plot_data['const_cost']
    plot_data['maint_cost_neg'] = -plot_data['maint_cost']
    plot_data['uncovered_op_neg'] = -plot_data['uncovered_op_cost']
    plot_data['tt_loss_neg'] = -plot_data['tt_loss_cost']

    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = _WF_FONT
    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.8

    ax.bar(plot_data.index, plot_data['const_cost_neg'], width,
           label='Construction costs',
           color=_COST_COLORS['TotalConstructionCost'], alpha=0.8)
    ax.bar(plot_data.index, plot_data['maint_cost_neg'], width,
           bottom=plot_data['const_cost_neg'], label='Maintenance costs',
           color=_COST_COLORS['TotalMaintenanceCost'], alpha=0.8)
    sum_prev = plot_data['const_cost_neg'] + plot_data['maint_cost_neg']
    ax.bar(plot_data.index, plot_data['uncovered_op_neg'], width,
           bottom=sum_prev, label='Operating costs',
           color=_COST_COLORS['TotalUncoveredOperatingCost'], alpha=0.8)
    ax.bar(plot_data.index, plot_data['tt_loss_neg'], width,
           bottom=sum_prev + plot_data['uncovered_op_neg'],
           label='Travel time loss',
           color=_COST_COLORS['TotalTtLossCost'], alpha=0.8)
    ax.bar(plot_data.index, plot_data['benefit'], width,
           label='Travel time savings', color='#2ca02c', alpha=0.8)

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Values in CHF million', fontsize=11)
    ax.set_title(f'Discounted costs and benefits over time\n{label}',
                 fontsize=13, pad=16)
    ax.grid(True, which="both", ls="-", alpha=0.2)

    # Legacy y-crop: bottom at 150% of the year-2 cost stack (construction year
    # dwarfs everything otherwise); the cut stack's total is annotated below.
    cost_stack = (plot_data['const_cost_neg'] + plot_data['maint_cost_neg']
                  + plot_data['uncovered_op_neg'] + plot_data['tt_loss_neg'])
    year_2_costs = cost_stack.iloc[1] if len(plot_data) > 1 else 0
    y_bot = year_2_costs * 1.5
    y_top = max(plot_data['benefit']) * 1.2
    if y_bot < y_top and (y_bot or y_top):
        ax.set_ylim(bottom=y_bot, top=y_top)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1e6:.1f}'))

    # Legend in the emptier right corner: benefits decline rightward (top-right
    # clear) for most ints, but a sustained-benefit int (NDC) crowds the top, so
    # compare the right-portion benefit vs cost extents and pick the freer corner.
    n = len(plot_data)
    cut = max(1, int(n * 0.55))
    b_right = float(plot_data['benefit'].iloc[cut:].max()) if n > cut else 0.0
    c_right = float(-cost_stack.iloc[cut:].min()) if n > cut else 0.0
    loc = ('upper right' if (y_top - b_right) >= (abs(y_bot) - c_right)
           else 'lower right')
    ax.legend(loc=loc, frameon=True, edgecolor='black', fontsize=8.5)

    # First-year total cost: a white label 3 years to the right of the worst year,
    # an L-connector (horizontal then perpendicular) touching the cost bar's end
    # when it is within the plot, else a horizontal line to the cropped bar's side.
    worst_x = int(cost_stack.idxmin())
    cost_val = float(cost_stack.min())
    if cost_val >= y_bot:                       # bar end visible
        xy, box_y, conn = (worst_x, cost_val), cost_val * 0.72, 'angle,angleA=0,angleB=90'
    else:                                       # bar cropped → horizontal to the side
        box_y = y_bot * 0.88
        xy, conn = (worst_x, box_y), 'arc3,rad=0'
    ax.annotate(f'{-cost_val / 1e6:.1f} Mio. CHF', xy=xy, xycoords='data',
                xytext=(worst_x + 3, box_y), textcoords='data', ha='left',
                va='center', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='black',
                          alpha=0.95, lw=0.8),
                arrowprops=dict(arrowstyle='-', color='black', lw=0.8,
                                connectionstyle=conn))

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'cost_benefit_{svc_int_id}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Per-svc-int A4 factsheet — assembles the existing 5B/6C/6D/waterfall plots on
# one page plus a cost-benefit summary table with named infra/cap interventions.
# ─────────────────────────────────────────────────────────────────────────────

def _load_name_context(combo: str, base_infra: str) -> dict:
    """Lookups for naming infra (CC) and capacity interventions on the factsheet:
    the clean rail-station set, BAV-number->name map, the base segments and the
    combo's CC segments (loaded once)."""
    import geopandas as gpd
    infra_dir = paths.get_infra_version_dir(base_infra)
    nodes = gpd.read_file(os.path.join(infra_dir, 'nodes.gpkg'))
    base_seg = gpd.read_file(os.path.join(infra_dir, 'segments.gpkg'))
    cc_path = paths.get_cc_interventions_gpkg(combo)
    cc_seg = (gpd.read_file(cc_path, layer='segments')
              if os.path.exists(cc_path) else None)
    clean = set(nodes[(nodes['Node_Class'] == 'station')
                      & (nodes['Transport_Mode'] == 'train')
                      & (~nodes['Name'].astype(str).str.contains(r'[(,]'))]['Name'])
    num2name = dict(zip(nodes['Number'].astype('Int64').astype(str), nodes['Name']))
    return {'clean': clean, 'num2name': num2name, 'base_seg': base_seg,
            'cc_seg': cc_seg}


def _resolve_station(name, clean) -> str:
    """A clean rail-station name, or the parent station of a junction label
    ('Dietlikon Süd (Abzw)' -> 'Dietlikon'); None if unresolvable."""
    name = str(name)
    if name in clean:
        return name
    bare = name.split('(')[0].strip()
    cands = [c for c in clean if bare.startswith(c)]
    return max(cands, key=len) if cands else None


def _cc_name(cc_id: str, ctx: dict) -> str:
    """Readable connecting-curve name 'CC A - B' from the base segments it bridges
    (their endpoint stations); falls back to 'CC <id>'."""
    cc_seg = ctx['cc_seg']
    if cc_seg is None:
        return f"CC {cc_id}"
    bridged = set()
    for rb in cc_seg.loc[cc_seg['int_id'] == cc_id, 'removes_base_rows'].dropna():
        bridged.update(s.strip() for s in str(rb).split(','))
    base_seg, clean, names = ctx['base_seg'], ctx['clean'], []
    for sid in bridged:
        rows = base_seg[base_seg['Segment_ID'].astype(str) == sid]
        for _, r in rows.iterrows():
            for end in (r['From_Name'], r['To_Name']):
                rv = _resolve_station(end, clean)
                if rv and rv not in names:
                    names.append(rv)
    return f"CC {' - '.join(names[:2])}" if names else f"CC {cc_id}"


def _cap_names(svc_int_id: str, combo: str, ctx: dict) -> list:
    """Named capacity interventions for a svc-int from the 5C attribution table:
    'Station track <stn>' / 'Segment siding <a>-<b>'."""
    num2name = ctx['num2name']

    def _stn(num):
        if num is None or (isinstance(num, float) and np.isnan(num)):
            return '?'
        return num2name.get(str(int(float(num))), str(num))

    p = paths.get_svc_int_cap_attribution_path(combo)
    if not os.path.exists(p):
        return []
    d = pd.read_csv(p)
    d = d[d['svc_int_id'].astype(str) == svc_int_id].drop_duplicates('cap_id')
    out = []
    for _, r in d.iterrows():
        if r['cap_type'] == 'station_track':
            out.append(f"Station track {_stn(r['node_id'])}")
        elif r['cap_type'] == 'segment_passing_siding':
            a, b = (str(r['segment_id']).split('-') + ['?', '?'])[:2]
            out.append(f"Segment siding {_stn(a)}-{_stn(b)}")
        else:
            out.append(str(r['cap_type']).replace('_', ' '))
    return out


def _int_description(rec: dict) -> str:
    """One-line plain description of what the svc-int does, from its operations."""
    rec = rec or {}
    itype = rec.get('int_type')
    route = rec.get('route_id', '?')
    ops = rec.get('operations') or [{}]
    op, params = ops[0].get('op'), ops[0].get('params', {})
    stations = rec.get('affected_stations') or []
    if itype == 'ext':
        start = params.get('endpoint') or (stations[0] if stations else '?')
        end = (params.get('stops') or stations[-1:] or ['?'])[-1]
        return f"Extended line {route} from {start} to {end}"
    if itype == 'ndc':
        return (f"New direct connection {stations[0]} - {stations[-1]}"
                if stations else f"New direct line {route}")
    if itype == 'frq':
        if op in ('factor', 'double', 'multiply', 'set_frequency'):
            return f"Increased frequency of line {route}"
        tgt = (params.get('stops') or stations[-1:] or ['?'])[-1]
        return f"Frequency homogenisation: line {route} extended to {tgt}"
    if itype == 'stp':
        return (f"Modified stop pattern on line {route} "
                f"({stations[0]} - {stations[-1]})" if stations
                else f"Modified stop pattern on line {route}")
    return rec.get('line_short_name', '?')


def _fs_img(path):
    """Read a PNG (matplotlib) or rasterise a PDF (PyMuPDF) to a uint8 RGB array."""
    if str(path).lower().endswith('.pdf'):
        import fitz
        doc = fitz.open(path)
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n)
        doc.close()
        return arr
    import matplotlib.pyplot as plt
    arr = plt.imread(path)
    return arr if arr.dtype == np.uint8 else (arr[..., :3] * 255).astype(np.uint8)


def _fs_trim(arr, thr=248, pad=6):
    """Crop near-white borders so the embedded panel fills its cell."""
    mask = (arr[..., :3] < thr).any(axis=2)
    if not mask.any():
        return arr
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    return arr[max(rows[0] - pad, 0):rows[-1] + pad + 1,
               max(cols[0] - pad, 0):cols[-1] + pad + 1]


def _fs_frame(ax, caption=None):
    for s in ax.spines.values():
        s.set_edgecolor('#dddddd')
        s.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    if caption:
        ax.set_title(caption, fontsize=9.5, fontweight='bold', color='black',
                     pad=3)


def _fs_panel(ax, path, caption=None):
    import matplotlib.pyplot as plt  # noqa: F401  (ensures Agg backend chosen)
    _fs_frame(ax, caption)
    if path and os.path.exists(path):
        ax.imshow(_fs_trim(_fs_img(path)))
    else:
        ax.text(0.5, 0.5, f"missing:\n{os.path.basename(str(path))}",
                ha='center', va='center', fontsize=7, color='red',
                transform=ax.transAxes)


def _fs_table(ax, row, rec, combo, base_infra, ctx):
    ax.axis('off')
    infra = [_cc_name(c, ctx) for c in ((rec or {}).get('requires_infra') or [])]
    caps = _cap_names(row['development'], combo, ctx)

    def _wrap_list(items, width):
        if not items:
            return 'None'
        return '\n'.join('\n'.join(textwrap.wrap(f"- {it}", width,
                                                 subsequent_indent='  '))
                         for it in items)

    cells = [
        ('Description', '\n'.join(textwrap.wrap(_int_description(rec), 30))),
        ('Infrastructure interventions', _wrap_list(infra, 26)),
        ('Capacity interventions', _wrap_list(caps, 26)),
        ('Travel-time savings',
         f"{row['Monetized Savings Mean [in Mio. CHF]']:.1f} Mio. CHF"),
        ('Total costs', f"{row['Total Costs [in Mio. CHF]']:.1f} Mio. CHF"),
        ('Net benefit (NPV)', f"{row['Net Benefit [in Mio. CHF]']:.1f} Mio. CHF"),
        ('Benefit-cost ratio', f"{row['CBA Ratio']:.2f}"),
    ]
    text = [['\n'.join(textwrap.wrap(k, 15)), v] for k, v in cells]
    nlines = [max(k.count('\n') + 1, v.count('\n') + 1) for k, v in text]
    total = sum(nlines)

    tbl = ax.table(cellText=text, colWidths=[0.42, 0.58], cellLoc='left',
                   loc='upper center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        cell.set_linewidth(0.5)
        cell.set_height(nlines[r] / total)
        cell.set_text_props(va='center')
        if c == 0:
            cell.set_text_props(fontweight='bold', va='center')
        if r == len(cells) - 1:
            cell.set_facecolor('#f3f3f3')
            cell.set_text_props(fontweight='bold')
    ax.set_title('Cost-benefit summary', fontsize=9.5, fontweight='bold',
                 color='black', pad=3)


def _plot_factsheet(svc_int_id: str, row, rec, combo: str, base_infra: str,
                    svc_version: str, waterfall_dir: str, ctx: dict,
                    output_path: str) -> None:
    """One A4-portrait factsheet per svc-int, embedding the existing plots:
    5B service-intervention map + cost-benefit table & discounted waterfall (top),
    6C travel-time / interchange delta matrices (middle), 6D passenger-flow change
    + 6C cell-accessibility change (bottom)."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    itype = _svc_int_type(svc_int_id)
    method = settings.ROUTING_ASSIGNMENT_METHOD
    net = paths.svc_int_network_name(svc_int_id, combo)
    assign = paths.get_assignment_plot_dir(net, method)
    p_5b = os.path.join(paths.get_developments_plot_dir(combo, itype),
                        f"svc_int_{svc_int_id}_{base_infra}.pdf")
    p_tt = os.path.join(assign, 'matrix_travel_top15_delta.png')
    p_fq = os.path.join(assign, 'matrix_frequency_top15_delta.png')
    p_cell = os.path.join(assign, 'cell_accessibility_change.pdf')
    p_flow = paths.get_passenger_flow_plot_path(net, 'diff')
    p_wf = os.path.join(waterfall_dir, f"cost_benefit_{svc_int_id}.png")

    label = _dev_label(svc_int_id, rec, combo, base_infra)
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.suptitle(f"Factsheet: {label}", fontsize=14, fontweight='bold', y=0.985)
    gs = GridSpec(3, 2, figure=fig, left=0.035, right=0.965, top=0.935,
                  bottom=0.015, hspace=0.16, wspace=0.06,
                  height_ratios=[1.08, 0.62, 1.45])

    _fs_panel(fig.add_subplot(gs[0, 0]), p_5b, 'Service intervention')
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1], hspace=0.22,
                                    height_ratios=[1.25, 1.0])
    _fs_table(fig.add_subplot(inner[0, 0]), row, rec, combo, base_infra, ctx)
    _fs_panel(fig.add_subplot(inner[1, 0]), p_wf, 'Discounted costs & benefits')
    _fs_panel(fig.add_subplot(gs[1, 0]), p_tt, 'Travel-time change [min]')
    _fs_panel(fig.add_subplot(gs[1, 1]), p_fq, 'Interchange change')
    _fs_panel(fig.add_subplot(gs[2, 0]), p_flow, 'Passenger-flow change')
    _fs_panel(fig.add_subplot(gs[2, 1]), p_cell, 'Accessibility change')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    import ints_core as _core

    print("=== Phase 9 — CBA (standalone) ===")
    _svc = input(f"Service version [{settings.SVC_VERSION}]: ").strip() \
        or settings.SVC_VERSION
    if _svc == 'Build_New':
        _svc = settings.SVC_BUILD_NEW_NAME
    _default_combo = _core.default_combo(svc_version=_svc)
    _combo = input(f"Combo [{_default_combo}]: ").strip() or _default_combo
    _plot_def = 'y' if settings.PLOT_RESULTS else 'n'
    _plots = (input(f"Generate plots? [y/n] [{_plot_def}]: ").strip()
              or _plot_def).lower() == 'y'
    run_cba(combo=_combo, make_plots=_plots)
