"""
cba_results — Phase 9: CBA integration, discounting, aggregation, results.
Last modified: 2026-06-10

Combines the Phase 8 valuation inputs (8A traveltime_savings.csv, 8B
construction_cost.csv) into the discounted cost-benefit result, single track,
delta-only (no do-nothing row — benefits are skim deltas, costs per-svc-int
absolutes). MultiIndex (development, scenario, year) with STRING svc-int ids
(ext_100001, ndc_103001) and the legacy columns const_cost / maint_cost /
uncovered_op_cost / benefit.

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
                 'ndc': settings.DEV_ID_START_NDC}[int_type]
        return f"{int_type.upper()}{int(num) - start}"
    except (KeyError, ValueError):
        return sid


_CB_COLS = ['const_cost', 'maint_cost', 'uncovered_op_cost', 'benefit']


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
    the years after it."""
    idx = pd.MultiIndex.from_product([ids, scenarios, years],
                                     names=['development', 'scenario', 'year'])
    cb = pd.DataFrame(0.0, index=idx, columns=_CB_COLS)

    benefit = benefits.set_index(['development', 'scenario', 'year'])[
        'monetized_savings_yearly']
    cb['benefit'] = benefit.reindex(idx).fillna(0.0).to_numpy(dtype=float)

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
    raw['total_costs'] = (raw['construction_cost'] + raw['maintenance_cost']
                          + raw['uncovered_op_cost'])
    raw['total_net_benefit'] = raw['monetized_savings_total'] - raw['total_costs']
    raw['cba_ratio'] = raw['monetized_savings_total'] / raw['total_costs']
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
    if record.get('int_type') == 'ext':
        return f"{stations[0]} → {stations[-1]}"
    return f"{stations[0]} – {stations[-1]}"


def _reshape_wide(raw: pd.DataFrame, records: dict) -> pd.DataFrame:
    """Wide per-svc-int table (legacy transform_and_reshape_cost_df role):
    per scenario s the columns monetized_savings_total_{s} /
    Net_Benefit_scenario_{s} / cba_ratio_scenario_{s}; costs once per svc-int,
    Mio-CHF columns under the legacy names. development stays the string id —
    no 'Development_' prefix, no dev_id round-trip."""
    wide = raw.pivot_table(index='development', columns='scenario',
                           values='monetized_savings_total', aggfunc='first')
    wide.columns = [f"monetized_savings_total_{s}" for s in wide.columns]
    wide = wide.reset_index()

    costs_df = raw.groupby('development', as_index=False).agg(
        construction_cost=('construction_cost', 'first'),
        maintenance_cost=('maintenance_cost', 'first'),
        uncovered_op_cost=('uncovered_op_cost', 'first'))
    wide = wide.merge(costs_df, on='development')
    wide['total_costs'] = (wide['construction_cost'] + wide['maintenance_cost']
                           + wide['uncovered_op_cost'])

    savings_cols = [c for c in wide.columns
                    if c.startswith('monetized_savings_total_')]
    derived = {}
    for col in savings_cols:
        s = col.rsplit('_', 1)[-1]
        derived[f"Net_Benefit_scenario_{s}"] = wide[col] - wide['total_costs']
        derived[f"cba_ratio_scenario_{s}"] = wide[col] / wide['total_costs']
    derived['Construction Cost [in Mio. CHF]'] = wide['construction_cost'] / 1e6
    derived['Maintenance Costs [in Mio. CHF]'] = wide['maintenance_cost'] / 1e6
    derived['Uncovered Operating Costs [in Mio. CHF]'] = (
        wide['uncovered_op_cost'] / 1e6)
    wide = pd.concat([wide, pd.DataFrame(derived, index=wide.index)], axis=1)

    wide.insert(1, 'label', wide['development'].map(svc_int_label))
    wide.insert(2, 'int_type', wide['development'].map(_svc_int_type))

    front = ['development', 'label', 'int_type',
             'Construction Cost [in Mio. CHF]',
             'Maintenance Costs [in Mio. CHF]',
             'Uncovered Operating Costs [in Mio. CHF]', 'total_costs']
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
        'Total Costs [in Mio. CHF]': wide['total_costs'] / 1e6,
        'Monetized Savings Mean [in Mio. CHF]': savings.mean(axis=1) / 1e6,
        'Monetized Savings Min [in Mio. CHF]': savings.min(axis=1) / 1e6,
        'Monetized Savings Max [in Mio. CHF]': savings.max(axis=1) / 1e6,
        'Monetized Savings Std [in Mio. CHF]': savings.std(axis=1) / 1e6,
    })
    out['Net Benefit [in Mio. CHF]'] = (
        out['Monetized Savings Mean [in Mio. CHF]']
        - out['Total Costs [in Mio. CHF]'])
    out['CBA Ratio'] = (out['Monetized Savings Mean [in Mio. CHF]']
                        / out['Total Costs [in Mio. CHF]'])
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
                'TotalUncoveredOperatingCost': '#034e7b'}
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


def _chunked_by_benefit(data: pd.DataFrame, size: int = 6):
    """Yield data subsets of <= size developments, ordered by mean net benefit
    descending (legacy 6-per-plot grouping)."""
    ranked = (data.groupby('development')['total_net_benefit'].mean()
              .sort_values(ascending=False).index.tolist())
    for i in range(0, len(ranked), size):
        yield i // size + 1, data[data['development'].isin(ranked[i:i + size])]


def _make_result_plots(raw: pd.DataFrame, cb_disc: pd.DataFrame, records: dict,
                       geo, combo: str, base_infra: str,
                       svc_version: str) -> None:
    """The PLOT_RESULTS core set: EXT/NDC chart families (plain, grouped by
    connection, ranked), overview + per-group network maps, combined
    chart+map images, overall cumulative distribution, per-svc-int
    discounted waterfalls."""
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
    data['connection'] = data['development'].map(
        lambda iid: _connection_label(records.get(iid)))
    geom_by_label = {svc_int_label(iid): g
                     for iid, g in zip(geo['development'], geo.geometry)}
    map_ctx = _load_map_context(base_infra, svc_version)

    def _charts_map_combined(selected, prefix, chart_dir, out_combined_dir):
        line_colors = _plot_basic_charts(selected, prefix, chart_dir)
        map_path = os.path.join(maps_dir, f"railway_lines_{prefix}.png")
        _plot_network_map(selected['line_name'].unique().tolist(),
                          geom_by_label, line_colors, map_path, map_ctx,
                          zoom_to_group=True, figsize=(8, 6))
        for suffix in _CHART_SUFFIXES:
            _combine_images(
                os.path.join(chart_dir, f"{prefix}_{suffix}.png"), map_path,
                os.path.join(out_combined_dir, f"{prefix}_{suffix}_combined.png"))

    ext = data[data['development'].map(_svc_int_type) == 'ext']
    ndc = data[data['development'].map(_svc_int_type) == 'ndc']

    if not ext.empty:
        print("  [plot] EXT chart family + ranked/combined...")
        _plot_basic_charts(ext, 'EXT', benefits_dir)
        _charts_map_combined(ext, 'ranked_group_ext', ranked_dir,
                             ranked_combined_dir)

    if not ndc.empty:
        print("  [plot] NDC chart families (grouped by connection + ranked)...")
        for conn in [c for c in ndc['connection'].unique() if c]:
            sub = ndc[ndc['connection'] == conn]
            for i, selected in _chunked_by_benefit(sub):
                _charts_map_combined(selected, f"{_sanitize(conn)}_group_{i}",
                                     benefits_dir, combined_dir)
        for i, selected in _chunked_by_benefit(ndc):
            _charts_map_combined(selected, f"ranked_group_{i}", ranked_dir,
                                 ranked_combined_dir)

    print("  [plot] overview network maps...")
    for sub, name in ((ext, 'developments_ext'), (ndc, 'developments_ndc')):
        if sub.empty:
            continue
        labels = sub['line_name'].unique().tolist()
        _plot_network_map(labels, geom_by_label, _overview_colors(labels),
                          os.path.join(maps_dir, f'{name}.png'), map_ctx,
                          zoom_to_group=False, figsize=(12, 10))

    _plot_cumulative(data,
                     os.path.join(base_dir, 'cumulative_cost_distribution.png'),
                     group_by='line_name')

    print(f"  [plot] {cb_disc.index.get_level_values('development').nunique()} "
          f"discounted waterfalls...")
    for iid in sorted(set(cb_disc.index.get_level_values('development'))):
        _plot_waterfall(cb_disc, iid, waterfall_dir)
    print(f"  [plot] result plots saved under {base_dir}")


def _load_map_context(base_infra: str, svc_version: str) -> dict:
    """Map base layers, loaded once: lakes + study-area boundary, the projected
    base service network (rail context) and the version's nodes.gpkg (stations
    — replaces legacy points.gpkg/Rail_Node.xlsx)."""
    import fiona
    import geopandas as gpd

    boundary = gpd.read_file(paths.STUDY_AREA_BUFFER_GPKG).to_crs('EPSG:2056')
    lakes = gpd.read_file(paths.LAKES_SHP).to_crs('EPSG:2056')
    lakes = gpd.clip(lakes, boundary)

    rail_path = paths.get_projected_services_path(svc_version, base_infra)
    rail = pd.concat([gpd.read_file(rail_path, layer=layer)[['geometry']]
                      for layer in fiona.listlayers(rail_path)])
    rail = gpd.clip(gpd.GeoDataFrame(rail, crs='EPSG:2056'), boundary)

    nodes_path = os.path.join(paths.get_infra_version_dir(base_infra),
                              'nodes.gpkg')
    stations = gpd.read_file(nodes_path).to_crs('EPSG:2056')
    stations = gpd.clip(stations, boundary)
    return {'boundary': boundary, 'lakes': lakes, 'rail': rail,
            'stations': stations}


def _overview_colors(labels) -> dict:
    """Legacy overview colour scheme (plots.py:1859): tab10/tab20/nipy_spectral
    by group size."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    n = len(labels)
    cmap = plt.cm.tab10 if n <= 10 else plt.cm.tab20 if n <= 20 \
        else plt.cm.nipy_spectral
    return {label: mcolors.rgb2hex(cmap(i / max(n, 1)))
            for i, label in enumerate(labels)}


def _plot_network_map(labels, geom_by_label: dict, color_dict: dict,
                      output_path: str, map_ctx: dict,
                      zoom_to_group: bool = False,
                      figsize: tuple = (12, 10)) -> None:
    """Network map (composition port of plots.py:1789, sources reworked):
    lakes, base rail context, stations, coloured intervention geometries
    (Phase 2 delta + CC arcs), selected-station labels, scalebar, legend."""
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    import plot_parameter as pp

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    map_ctx['lakes'].plot(ax=ax, color='lightblue', edgecolor='blue', zorder=1)
    map_ctx['rail'].plot(ax=ax, color='red', linewidth=1.0, alpha=0.5, zorder=2)
    map_ctx['stations'].plot(ax=ax, color='red', markersize=10, zorder=3)

    plotted = []
    for label in labels:
        geom = geom_by_label.get(label)
        if geom is None or geom.is_empty:
            continue
        gpd.GeoSeries([geom], crs='EPSG:2056').plot(
            ax=ax, color=color_dict.get(label, 'black'), linewidth=4, zorder=5)
        plotted.append(label)

    # Extent: study area for overviews, group bounds + 2 km for group maps.
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

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    labelled = map_ctx['stations'][
        map_ctx['stations']['Name'].isin(pp.selected_stations)]
    for _, row in labelled.iterrows():
        x, y = row.geometry.x, row.geometry.y
        if x0 <= x <= x1 and y0 <= y <= y1:
            ax.annotate(row['Name'], xy=(x, y), ha='center', va='top',
                        xytext=(0, -10), textcoords='offset points',
                        fontsize=10, color='black', zorder=7)

    try:
        from matplotlib_scalebar.scalebar import ScaleBar
        ax.add_artist(ScaleBar(dx=1, units="m", location="lower left",
                               scale_loc="bottom"))
    except ImportError:
        pass

    handles = [Patch(facecolor='lightblue', edgecolor='blue',
                     label='Water Bodies'),
               Line2D([0], [0], color='red', marker='o', markersize=6,
                      linestyle='None', label='Stations'),
               Line2D([0], [0], color='red', lw=1.0, label='Rail network')]
    handles += [Line2D([0], [0], color=color_dict.get(label, 'black'), lw=4,
                       label=label) for label in plotted]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05),
              ncol=3, frameon=False, fontsize=10)
    ax.set_axis_off()

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
         'TotalUncoveredOperatingCost': 'mean',
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


def _plot_waterfall(cb_disc: pd.DataFrame, svc_int_id: str,
                    output_dir: str) -> None:
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

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.set_style('whitegrid')
    plt.rcParams['font.family'] = 'serif'
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
    ax.bar(plot_data.index, plot_data['benefit'], width,
           label='Travel time savings', color='#2ca02c', alpha=0.8)

    label = svc_int_label(svc_int_id)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Values in CHF million', fontsize=12)
    ax.set_title(f'Discounted costs and benefits over time\n'
                 f'Development: {label} ({svc_int_id})', fontsize=14, pad=20)
    ax.legend(loc='lower right', frameon=True, edgecolor='black')
    ax.grid(True, which="both", ls="-", alpha=0.2)

    # Legacy y-crop: bottom at 150% of the year-2 cost stack (construction year
    # dwarfs everything otherwise), annotated with the cut stack's total.
    year_2_costs = (plot_data['const_cost_neg'] + plot_data['maint_cost_neg']
                    + plot_data['uncovered_op_neg']).iloc[1] \
        if len(plot_data) > 1 else 0
    y_limit_bottom = year_2_costs * 1.5
    y_limit_top = max(plot_data['benefit']) * 1.2
    if y_limit_bottom < y_limit_top and (y_limit_bottom or y_limit_top):
        ax.set_ylim(bottom=y_limit_bottom, top=y_limit_top)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: f'{x / 1e6:.1f}'))

    total_by_year = (plot_data['const_cost_neg'] + plot_data['maint_cost_neg']
                     + plot_data['uncovered_op_neg'])
    ax.annotate(f'{-total_by_year.min() / 1e6:.1f} Mio. CHF',
                xy=(total_by_year.idxmin(), y_limit_bottom * 0.95),
                xytext=(0, 10), textcoords='offset points', ha='center',
                va='bottom', fontsize=10)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'cost_benefit_{svc_int_id}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


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
