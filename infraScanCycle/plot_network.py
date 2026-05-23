def plot_node_accessibility(
        accessibility_results,
        corridor_polygon=None,
        scenarios=('s1', 's2', 's3'),
        save_path='data/Network/accessibility/accessibility_plot.png'):
    """
    Two-row figure summarising Voronoi-based node accessibility:
      Row 1 — access_score maps (combined pop+empl, 0-1) for s1 / s2 / s3
      Row 2 — left: pop vs empl scatter coloured by score (s2);
               centre: score distribution violin per scenario;
               right: top-15 nodes by access_score (s2) ranked bar
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import matplotlib.patches as mpatches
    import numpy as np
    import geopandas as gpd

    # s1 = reference extrapolation, s2 = s1 − Δ/3 (low), s3 = s1 + Δ/3 (high)
    LABELS   = {'s1': 'Low growth (s1)', 's2': 'Medium growth (s2)', 's3': 'High growth (s3)'}
    S_COLORS = {'s1': '#78909C', 's2': '#1565C0', 's3': '#E53935'}

    avail = [s for s in scenarios if s in accessibility_results]
    n_scen = len(avail)

    fig = plt.figure(figsize=(7 * max(n_scen, 3), 14), facecolor='white')
    gs  = fig.add_gridspec(2, max(n_scen, 3), hspace=0.35, wspace=0.25)

    # shared norm across all scenarios — use nanmax so NaN nodes don't break the scale
    all_scores = np.concatenate([accessibility_results[s]['access_score'].values for s in avail])
    _vmax = float(np.nanmax(all_scores)) if not np.all(np.isnan(all_scores)) else 1.0
    norm = mcolors.Normalize(vmin=0, vmax=_vmax)
    cmap = cm.get_cmap('YlOrRd')

    # data bounds for consistent zoom
    ref_gdf = accessibility_results[avail[0]]
    bounds  = ref_gdf.total_bounds
    pad_x   = (bounds[2] - bounds[0]) * 0.03
    pad_y   = (bounds[3] - bounds[1]) * 0.03
    xlim = (bounds[0] - pad_x, bounds[2] + pad_x)
    ylim = (bounds[1] - pad_y, bounds[3] + pad_y)

    # ── Row 1: access_score maps ──────────────────────────────────────────
    for col_idx, s in enumerate(avail):
        ax = fig.add_subplot(gs[0, col_idx])
        gdf = accessibility_results[s]

        if corridor_polygon is not None:
            gpd.GeoDataFrame(geometry=[corridor_polygon], crs="EPSG:2056").plot(
                ax=ax, facecolor='#F5F5F5', edgecolor='#455A64',
                linewidth=1.4, zorder=0, alpha=0.4)

        gdf.plot(column='access_score', ax=ax, cmap=cmap, norm=norm,
                 edgecolor='#BDBDBD', linewidth=0.3, alpha=0.92,
                 missing_kwds={'color': '#EEEEEE'})

        # label top quartile polygons
        thresh = gdf['access_score'].quantile(0.75)
        for _, r in gdf[gdf['access_score'] >= thresh].iterrows():
            if r.geometry is None or r.geometry.is_empty:
                continue
            cx, cy = r.geometry.centroid.x, r.geometry.centroid.y
            ax.annotate(f'{r["access_score"]:.2f}', xy=(cx, cy),
                        ha='center', va='center', fontsize=6,
                        fontweight='bold', color='#212121')

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal')
        ax.axis('off')
        total_empl = gdf[f'{s}_empl'].sum() if f'{s}_empl' in gdf.columns else 0
        ax.set_title(f'{LABELS.get(s, s)}\nempl={total_empl:,.0f}',
                     fontsize=10, fontweight='bold' if s == 's2' else 'normal')

    # shared colorbar for row 1
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=[fig.axes[i] for i in range(n_scen)],
                      shrink=0.55, pad=0.02, aspect=25, location='right')
    cb.set_label('Accessibility score (0–1)', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    # ── Row 2, Panel 0: empl vs access score scatter (s2) ───────────────
    ax_sc = fig.add_subplot(gs[1, 0])
    ref_s = 's2' if 's2' in avail else avail[0]
    gdf2  = accessibility_results[ref_s]
    ec    = f'{ref_s}_empl'
    sc_colors = [cmap(norm(v)) for v in gdf2['access_score']]
    x_vals = gdf2[ec] if ec in gdf2.columns else gdf2['access_score']
    ax_sc.scatter(x_vals, gdf2['access_score'],
                  c=sc_colors, s=30, alpha=0.8, edgecolors='white', linewidths=0.3)
    ax_sc.set_xlabel('Employment in catchment', fontsize=9)
    ax_sc.set_ylabel('Gravity accessibility score (0–1)', fontsize=9)
    ax_sc.set_title(f'Employment vs Accessibility\n(scenario {ref_s})', fontsize=10)
    ax_sc.spines[['top', 'right']].set_visible(False)

    # ── Row 2, Panel 1: score distribution violin ────────────────────────
    ax_vl = fig.add_subplot(gs[1, 1])
    # drop NaN before violin (NaN nodes = Voronoi polygons outside raster extent)
    data  = [accessibility_results[s]['access_score'].dropna().values for s in avail]
    vp    = ax_vl.violinplot(data, positions=range(len(avail)),
                             showmedians=True, showextrema=True)
    for i, (body, s) in enumerate(zip(vp['bodies'], avail)):
        body.set_facecolor(S_COLORS.get(s, '#90A4AE'))
        body.set_alpha(0.75)
    vp['cmedians'].set_color('#212121')
    ax_vl.axhline(0, color='#9E9E9E', linewidth=0.8, linestyle='--')
    ax_vl.set_xticks(range(len(avail)))
    ax_vl.set_xticklabels([LABELS.get(s, s) for s in avail], fontsize=8, rotation=15, ha='right')
    ax_vl.set_ylabel('Access score (0–1)', fontsize=9)
    ax_vl.set_title('Score distribution\nby scenario', fontsize=10)
    ax_vl.spines[['top', 'right']].set_visible(False)

    # ── Row 2, Panel 2: top-15 nodes ranked bar ───────────────────────────
    ax_rk = fig.add_subplot(gs[1, 2])
    top15 = (gdf2.nlargest(15, 'access_score')
               .sort_values('access_score')
               .reset_index(drop=True))
    bar_colors = [cmap(norm(v)) for v in top15['access_score']]
    ax_rk.barh(range(len(top15)), top15['access_score'],
               color=bar_colors, edgecolor='white', height=0.7)
    ax_rk.set_yticks(range(len(top15)))
    ax_rk.set_yticklabels([f"Node {int(r)}" for r in top15['ID_point']], fontsize=7)
    ax_rk.set_xlabel('Access score (0–1)', fontsize=9)
    ax_rk.set_title(f'Top 15 nodes by accessibility\n(scenario {ref_s})', fontsize=10)
    ax_rk.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Node Accessibility — Voronoi Catchment Scoring', fontsize=14,
                 fontweight='bold', y=1.01)
    import os; os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_od_results(
        od_scenarios,
        voronoi_vals,
        points_gdf,
        corridor_polygon=None,
        scenarios=('s1', 's2', 's3'),
        edges_gdf=None,
        lakes_gdf=None,
        save_path="data/OD/od_plot.png"):
    """
    Three-panel OD summary (no routing — Voronoi-weighted):

      Left   — Origin strength map: each node bubble sized by population (s2),
               coloured by employment share (destination attractiveness)
      Centre — Scenario comparison: grouped bars, one cluster per scenario,
               split by origin-pop vs total-trips for quick magnitude check
      Right  — Top-20 origin nodes ranked by daily cycling trips generated (s2)
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import numpy as np
    import geopandas as gpd

    LABELS   = {'s1': 'Low (s1)', 's2': 'Medium (s2)', 's3': 'High (s3)'}
    S_COLORS = {'s1': '#78909C', 's2': '#1565C0', 's3': '#E53935'}

    fig, axes = plt.subplots(1, 3, figsize=(22, 9),
                             gridspec_kw={'width_ratios': [2.5, 1.5, 1.5]})
    ax_map, ax_bar, ax_rank = axes

    # ── build node pop/empl lookup from voronoi_vals ─────────────────────
    voronoi_vals = voronoi_vals.copy()
    voronoi_vals['ID_point'] = voronoi_vals['ID_point'].astype(int)

    ref_s = next((s for s in ['s2', 's1', 's3'] if s in od_scenarios), None)
    pop_col  = f'{ref_s}_pop'  if ref_s else None
    empl_col = f'{ref_s}_empl' if ref_s else None

    node_df = points_gdf[['ID_point', 'geometry']].copy()
    node_df['ID_point'] = node_df['ID_point'].astype(int)

    if pop_col and pop_col in voronoi_vals.columns:
        node_df = node_df.merge(
            voronoi_vals[['ID_point', pop_col, empl_col]].rename(
                columns={pop_col: 'pop', empl_col: 'empl'}),
            on='ID_point', how='left')
        node_df[['pop', 'empl']] = node_df[['pop', 'empl']].fillna(0)
    else:
        node_df['pop'] = node_df['empl'] = 0

    # join total outflow (trips generated) from s2
    if ref_s and ref_s in od_scenarios:
        outflow = od_scenarios[ref_s].groupby('origin_id')['trips'].sum().reset_index()
        outflow.columns = ['ID_point', 'outflow']
        node_df = node_df.merge(outflow, on='ID_point', how='left')
        node_df['outflow'] = node_df['outflow'].fillna(0)
    else:
        node_df['outflow'] = 0

    # ── Panel 1: origin strength map ─────────────────────────────────────
    if corridor_polygon is not None:
        gpd.GeoDataFrame(geometry=[corridor_polygon], crs="EPSG:2056").plot(
            ax=ax_map, facecolor='#F5F5F5', edgecolor='#90A4AE',
            linewidth=1.2, zorder=0, alpha=0.5)

    # water bodies (lakes)
    if lakes_gdf is not None and len(lakes_gdf) > 0:
        lakes_gdf.plot(ax=ax_map, facecolor='#AED6F1', edgecolor='#5DADE2',
                       linewidth=0.5, zorder=1, alpha=0.8)

    # cycling network edges
    if edges_gdf is not None and len(edges_gdf) > 0:
        edges_gdf.plot(ax=ax_map, color='#B0BEC5', linewidth=0.4, zorder=2, alpha=0.6)

    # grey base nodes
    points_gdf.plot(ax=ax_map, color='#CFD8DC', markersize=3, zorder=3, alpha=0.5)

    # scaled bubbles coloured by employment (destination attractiveness)
    max_pop  = node_df['pop'].max() or 1
    max_empl = node_df['empl'].max() or 1
    sizes    = 20 + 300 * (node_df['pop'] / max_pop)
    norm     = mcolors.Normalize(vmin=0, vmax=max_empl)
    cmap     = cm.get_cmap('YlOrRd')
    colors   = [cmap(norm(v)) for v in node_df['empl']]

    sc = ax_map.scatter(node_df.geometry.x, node_df.geometry.y,
                        s=sizes, c=colors, zorder=5, alpha=0.85,
                        edgecolors='white', linewidths=0.4)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax_map, shrink=0.45, pad=0.02,
                 label='Employment in Voronoi (destination strength)')

    # size legend
    for pop_val, label in [(max_pop*0.25, '25%'), (max_pop*0.75, '75%')]:
        s = 20 + 300 * (pop_val / max_pop)
        ax_map.scatter([], [], s=s, color='#90A4AE', alpha=0.7,
                       edgecolors='white', label=f'Pop ≈ {pop_val:,.0f}')
    ax_map.legend(title='Bubble = population', loc='lower left', fontsize=7, framealpha=0.9)
    ax_map.set_title(f'Origin strength (population) & destination attractiveness (employment)\n'
                     f'Scenario {ref_s or "—"} — {len(node_df)} nodes', fontsize=10)
    ax_map.set_axis_off()

    # zoom to corridor perimeter
    if corridor_polygon is not None:
        minx, miny, maxx, maxy = corridor_polygon.bounds
        margin = 500  # metres
        ax_map.set_xlim(minx - margin, maxx + margin)
        ax_map.set_ylim(miny - margin, maxy + margin)

    # ── Panel 2: total cycling trips per scenario ─────────────────────────
    s_keys = [s for s in scenarios if s in od_scenarios]
    totals = [od_scenarios[s]['trips'].sum() for s in s_keys]
    pops   = [voronoi_vals[f'{s}_pop'].clip(lower=0).fillna(0).sum() for s in s_keys]

    x = np.arange(len(s_keys))
    w = 0.35
    b1 = ax_bar.bar(x - w/2, pops,   width=w, label='Total origin population',
                    color='#90CAF9', edgecolor='white')
    b2 = ax_bar.bar(x + w/2, totals, width=w, label='Daily cycling trips',
                    color='#78909C', edgecolor='white')
    for bar, val in zip(list(b1)+list(b2), pops+totals):
        ax_bar.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                    f'{val:,.0f}', ha='center', va='bottom', fontsize=7, rotation=45)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([LABELS.get(s, s) for s in s_keys], fontsize=9)
    ax_bar.set_ylabel('Count', fontsize=9)
    ax_bar.set_title('Population vs cycling trips\nby scenario', fontsize=10)
    ax_bar.legend(fontsize=8, framealpha=0.9)
    ax_bar.spines[['top', 'right']].set_visible(False)

    # ── Panel 3: top-20 origin nodes by trips generated (s2) ─────────────
    if ref_s and ref_s in od_scenarios:
        top20 = (od_scenarios[ref_s]
                 .groupby('origin_id')['trips'].sum()
                 .nlargest(20)
                 .reset_index()
                 .sort_values('trips'))
        ax_rank.barh(range(len(top20)), top20['trips'],
                     color='#1565C0', edgecolor='white', height=0.7)
        ax_rank.set_yticks(range(len(top20)))
        ax_rank.set_yticklabels([f"Node {int(r)}" for r in top20['origin_id']], fontsize=7)
        ax_rank.set_xlabel('Daily cycling trips generated', fontsize=9)
        ax_rank.set_title(f'Top 20 origin nodes\n(scenario {ref_s})', fontsize=10)
        ax_rank.spines[['top', 'right']].set_visible(False)
    else:
        ax_rank.set_axis_off()

    plt.suptitle('OD Matrix — Voronoi-weighted Cycling Flows', fontsize=13, fontweight='bold')
    plt.tight_layout()
    import os; os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_network_all_types(
        edges_gdf,
        conn_gdf=None,
        corridor_polygon=None,
        save_path="data/Network/processed/network_all_types.png"):
    """
    Five-layer connectivity plot produced at the end of step 2:

      grey         — clean existing edges (ROUTENTYP-based)
      orange solid — Schwachstellen (existing, below quality standard)
      red dashed   — Netzlücken (planned, tagged ROUTENTYP='Netzlücke')
      purple dotted— auto-generated connectors (ROUTENTYP='Connector')

    A sub-plot on the right shows a ROUTENTYP edge-count bar chart so the
    mix of quality levels is immediately visible.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import pandas as pd
    import geopandas as gpd

    fig, axes = plt.subplots(1, 2, figsize=(20, 10),
                             gridspec_kw={'width_ratios': [3, 1]})
    ax_map, ax_bar = axes

    # ── Layer masks ──────────────────────────────────────────────────────
    is_nl = (edges_gdf.get('is_development',   0) == 1)
    is_sw = (edges_gdf.get('is_schwachstelle', 0) == 1)
    base  = edges_gdf[~is_nl & ~is_sw]
    sw    = edges_gdf[is_sw]
    nl    = edges_gdf[is_nl]
    conn  = conn_gdf if (conn_gdf is not None and len(conn_gdf) > 0) else None

    # ── Map panel ────────────────────────────────────────────────────────
    if corridor_polygon is not None:
        gpd.GeoDataFrame(geometry=[corridor_polygon], crs="EPSG:2056").plot(
            ax=ax_map, facecolor='#FFFDE7', edgecolor='black',
            linewidth=1.5, zorder=0, alpha=0.4)

    base.plot(ax=ax_map, color='#9E9E9E', linewidth=0.9, alpha=0.7, zorder=1)
    sw.plot(  ax=ax_map, color='#FF9800', linewidth=2.2, alpha=0.9, zorder=2)
    nl.plot(  ax=ax_map, color='#F44336', linewidth=2.5,
              linestyle='--', alpha=0.9, zorder=3)
    if conn is not None:
        conn.plot(ax=ax_map, color='#9C27B0', linewidth=2.0,
                  linestyle=':', alpha=0.85, zorder=4)

    handles = [
        mpatches.Patch(color='#9E9E9E', label=f'Existing ({len(base)})'),
        mpatches.Patch(color='#FF9800', label=f'Schwachstellen ({len(sw)})'),
        mpatches.Patch(color='#F44336', label=f'Netzlücken ({len(nl)})'),
    ]
    if conn is not None:
        handles.append(mpatches.Patch(color='#9C27B0',
                                      label=f'Auto-connectors ({len(conn)})'))
    if corridor_polygon is not None:
        handles.append(mpatches.Patch(facecolor='#FFFDE7', edgecolor='black',
                                      linewidth=1.5, label='Corridor'))

    ax_map.legend(handles=handles, loc='lower right', fontsize=9, framealpha=0.9)
    total_conn = len(conn) if conn is not None else 0
    ax_map.set_title(
        f"Corridor network — all route types\n"
        f"{len(base)} existing  |  {len(sw)} Schwachstellen  |  "
        f"{len(nl)} Netzlücken  |  {total_conn} connectors",
        fontsize=12)
    ax_map.set_axis_off()

    # ── Bar chart panel: edge count by ROUTENTYP ─────────────────────────
    rt_col = next((c for c in edges_gdf.columns if c.upper().startswith('ROUTENTYP')), None)

    all_edges = [edges_gdf]
    if conn is not None:
        all_edges.append(conn)
    combined = gpd.GeoDataFrame(
        pd.concat(all_edges, ignore_index=True), crs="EPSG:2056"
    ) if conn is not None else edges_gdf

    color_map = {
        'Velobahn':                        '#1a6b3c',
        'Veloschnellroute':                '#2e7d32',
        'Hauptverbindung':                 '#2196F3',
        'Nebenverbindung':                 '#9E9E9E',
        'Zusätzliche Freizeitverbindung':  '#FF9800',
        'Netzlücke':                       '#F44336',
        'Connector':                       '#9C27B0',
    }
    if rt_col and rt_col in combined.columns and len(combined) > 0:
        counts = combined[rt_col].value_counts()
        colors = [color_map.get(str(k), '#BDBDBD') for k in counts.index]
        if len(colors) == 0:
            colors = ['#BDBDBD']
        counts.plot(kind='barh', ax=ax_bar, color=colors, edgecolor='white')
        ax_bar.set_xlabel('Number of edges')
        ax_bar.set_title('Edge count\nby ROUTENTYP', fontsize=10)
        ax_bar.tick_params(axis='y', labelsize=8)
        for i, v in enumerate(counts):
            ax_bar.text(v + 0.3, i, str(v), va='center', fontsize=8)
    else:
        ax_bar.text(0.5, 0.5, 'ROUTENTYP\nnot available',
                    ha='center', va='center', transform=ax_bar.transAxes)
        ax_bar.set_axis_off()

    plt.tight_layout()
    import os; os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_network_graph(
        edges_aug,
        points_corridor,
        corridor_polygon=None,
        save_path="data/Network/processed/network_graph.png"):
    """
    Full spatial graph of the corridor network — edges coloured by ROUTENTYP,
    nodes coloured and sized by type (intersection / through-point / dead-end).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    import geopandas as gpd

    fig, ax = plt.subplots(figsize=(16, 14))

    # ── corridor backdrop ────────────────────────────────────────────────
    if corridor_polygon is not None:
        gpd.GeoDataFrame(geometry=[corridor_polygon], crs="EPSG:2056").plot(
            ax=ax, facecolor='#F5F5F5', edgecolor='#455A64',
            linewidth=1.5, zorder=0, alpha=0.5)

    # ── edge colours by ROUTENTYP ────────────────────────────────────────
    EDGE_STYLE = {
        'Velobahn':                       dict(color='#1a6b3c', lw=2.8, ls='-',  zorder=3),
        'Veloschnellroute':               dict(color='#2e7d32', lw=2.5, ls='-',  zorder=3),
        'Hauptverbindung':                dict(color='#1565C0', lw=1.8, ls='-',  zorder=2),
        'Nebenverbindung':                dict(color='#78909C', lw=1.2, ls='-',  zorder=1),
        'Zusätzliche Freizeitverbindung': dict(color='#FF8F00', lw=1.5, ls='-',  zorder=2),
        'Netzlücke':                      dict(color='#E53935', lw=2.0, ls='--', zorder=4),
        'Connector':                      dict(color='#AB47BC', lw=1.0, ls=':',  zorder=1),
    }
    DEFAULT_EDGE = dict(color='#BDBDBD', lw=1.0, ls='-', zorder=1)

    edge_legend = []
    if 'ROUTENTYP' in edges_aug.columns:
        for rt, style in EDGE_STYLE.items():
            subset = edges_aug[edges_aug['ROUTENTYP'] == rt]
            if len(subset) == 0:
                continue
            subset.plot(ax=ax, color=style['color'], linewidth=style['lw'],
                        linestyle=style['ls'], zorder=style['zorder'], alpha=0.85)
            edge_legend.append(mlines.Line2D([], [], color=style['color'],
                                              linewidth=style['lw'], linestyle=style['ls'],
                                              label=f'{rt} ({len(subset)})'))
        # anything not in the palette
        others = edges_aug[~edges_aug['ROUTENTYP'].isin(EDGE_STYLE)]
        if len(others) > 0:
            others.plot(ax=ax, **{k: v for k, v in DEFAULT_EDGE.items() if k != 'zorder'},
                        zorder=DEFAULT_EDGE['zorder'], alpha=0.7)
    else:
        edges_aug.plot(ax=ax, color='#78909C', linewidth=1.2, zorder=1)

    # ── nodes coloured by type ────────────────────────────────────────────
    NODE_STYLE = {
        'is_intersection':  dict(color='#1565C0', size=18, marker='o', label='Intersection',  zorder=6),
        'is_through_point': dict(color='#2E7D32', size=10, marker='o', label='Through-point', zorder=5),
        'is_endpoint':      dict(color='#E53935', size=22, marker='^', label='Dead-end',      zorder=7),
    }

    node_legend = []
    for col, style in NODE_STYLE.items():
        if col not in points_corridor.columns:
            continue
        subset = points_corridor[points_corridor[col] == 1]
        if len(subset) == 0:
            continue
        subset.plot(ax=ax, color=style['color'], markersize=style['size'],
                    marker=style['marker'], zorder=style['zorder'], alpha=0.9)
        node_legend.append(mpatches.Patch(color=style['color'],
                                           label=f"{style['label']} ({len(subset)})"))

    # ── legend ────────────────────────────────────────────────────────────
    legend_edges = ax.legend(handles=edge_legend, title='Edge type',
                              loc='lower left', fontsize=8, title_fontsize=9,
                              framealpha=0.92, ncol=1)
    ax.add_artist(legend_edges)
    ax.legend(handles=node_legend, title='Node type',
              loc='lower right', fontsize=8, title_fontsize=9, framealpha=0.92)

    ax.set_title(
        f'Corridor Network Graph\n'
        f'{len(edges_aug)} edges  |  {len(points_corridor)} nodes',
        fontsize=13, fontweight='bold')
    ax.set_axis_off()

    plt.tight_layout()
    import os; os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_network_quality(
        edges_aug,
        edges_corridor,
        edges_final,
        points_corridor,
        nodes_gdf,
        save_path="data/Network/processed/network_quality.png"):
    """
    Three-panel network quality summary figure:
      Left  — edge count by ROUTENTYP (bar chart, coloured by type)
      Centre — node type breakdown in corridor (stacked bar: intersections /
               through-points / endpoints)
      Right  — key stats table (corridor coverage, missing ROUTENTYP, totals)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(18, 7),
                             gridspec_kw={'width_ratios': [2, 1.2, 1.4]})
    ax_rt, ax_nodes, ax_stats = axes

    # ── colour palette ────────────────────────────────────────────────────
    COLOR_MAP = {
        'Velobahn':                       '#1a6b3c',
        'Veloschnellroute':               '#2e7d32',
        'Hauptverbindung':                '#2196F3',
        'Nebenverbindung':                '#9E9E9E',
        'Zusätzliche Freizeitverbindung': '#FF9800',
        'Netzlücke':                      '#F44336',
        'Connector':                      '#9C27B0',
    }

    # ── Panel 1: ROUTENTYP edge counts ───────────────────────────────────
    if 'ROUTENTYP' in edges_aug.columns and len(edges_aug) > 0:
        counts = edges_aug['ROUTENTYP'].value_counts()
        colors = [COLOR_MAP.get(str(k), '#BDBDBD') for k in counts.index]
        bars = ax_rt.barh(range(len(counts)), counts.values,
                          color=colors, edgecolor='white', height=0.6)
        ax_rt.set_yticks(range(len(counts)))
        ax_rt.set_yticklabels(counts.index, fontsize=9)
        ax_rt.set_xlabel('Number of edges', fontsize=9)
        ax_rt.set_title('Edge count by ROUTENTYP\n(corridor, incl. Netzlücken)', fontsize=10)
        for i, v in enumerate(counts.values):
            ax_rt.text(v + 0.5, i, str(v), va='center', fontsize=8)
        missing = edges_aug['ROUTENTYP'].isna().sum()
        if missing > 0:
            ax_rt.set_xlabel(f'Number of edges  ({missing} missing ROUTENTYP)', fontsize=9, color='red')
    else:
        ax_rt.text(0.5, 0.5, 'ROUTENTYP\nnot available',
                   ha='center', va='center', transform=ax_rt.transAxes, fontsize=11)
        ax_rt.set_axis_off()

    # ── Panel 2: node type breakdown in corridor ──────────────────────────
    n_inter  = int(points_corridor['is_intersection'].sum())  if 'is_intersection'  in points_corridor.columns else 0
    n_thru   = int(points_corridor['is_through_point'].sum()) if 'is_through_point' in points_corridor.columns else 0
    n_end    = int(points_corridor['is_endpoint'].sum())      if 'is_endpoint'      in points_corridor.columns else 0
    n_total  = len(points_corridor)
    n_other  = max(n_total - n_inter - n_thru - n_end, 0)

    categories  = ['Intersections', 'Through-points', 'Dead-ends', 'Other']
    values      = [n_inter, n_thru, n_end, n_other]
    node_colors = ['#1565C0', '#43A047', '#E53935', '#9E9E9E']
    x = np.array([0])

    bottom = 0
    for cat, val, col in zip(categories, values, node_colors):
        if val > 0:
            ax_nodes.bar(x, val, bottom=bottom, color=col,
                         edgecolor='white', width=0.5, label=f'{cat} ({val})')
            ax_nodes.text(0, bottom + val / 2, str(val),
                          ha='center', va='center', fontsize=9,
                          color='white', fontweight='bold')
            bottom += val

    ax_nodes.set_xlim(-0.6, 0.6)
    ax_nodes.set_xticks([])
    ax_nodes.set_ylabel('Node count', fontsize=9)
    ax_nodes.set_title(f'Node types in corridor\n({n_total} total)', fontsize=10)
    ax_nodes.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # ── Panel 3: key stats table ──────────────────────────────────────────
    total_full   = max(len(edges_final), 1)
    coverage_pct = 100 * len(edges_corridor) / total_full
    missing_rt   = edges_aug['ROUTENTYP'].isna().sum() if 'ROUTENTYP' in edges_aug.columns else 'n/a'
    sw_count     = int((edges_aug.get('is_schwachstelle', 0) == 1).sum())
    nl_count     = int((edges_aug.get('is_development',   0) == 1).sum())
    conn_count   = int((edges_aug.get('is_connector',     0) == 1).sum()) if 'is_connector' in edges_aug.columns else 0

    rows = [
        ('Total nodes (full network)',    f'{len(nodes_gdf)}'),
        ('Nodes in corridor',             f'{len(points_corridor)}'),
        ('',                              ''),
        ('Total edges (full network)',    f'{len(edges_final)}'),
        ('Edges in corridor',             f'{len(edges_corridor)}'),
        ('Corridor coverage',             f'{coverage_pct:.0f}%'),
        ('',                              ''),
        ('Schwachstellen in corridor',    f'{sw_count}'),
        ('Netzlücken in corridor',        f'{nl_count}'),
        ('Connectors in corridor',        f'{conn_count}'),
        ('',                              ''),
        ('Missing ROUTENTYP',             f'{missing_rt}'),
    ]

    ax_stats.set_axis_off()
    ax_stats.set_title('Network quality stats', fontsize=10)
    y_pos = 0.97
    step  = 0.08
    for label, value in rows:
        if label == '':
            y_pos -= step * 0.4
            continue
        ax_stats.text(0.02, y_pos, label, transform=ax_stats.transAxes,
                      fontsize=9, va='top', color='#333333')
        ax_stats.text(0.98, y_pos, value, transform=ax_stats.transAxes,
                      fontsize=9, va='top', ha='right',
                      fontweight='bold', color='#1565C0')
        ax_stats.plot([0.02, 0.98], [y_pos - 0.005, y_pos - 0.005],
                      transform=ax_stats.transAxes,
                      linewidth=0.3, color='#BDBDBD', clip_on=False)
        y_pos -= step

    plt.suptitle('Network Quality Overview — Corridor', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    import os; os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_raw_network(edges_gdf, title="Raw imported network", save_path="data/Network/processed/plot_01_raw_import.png"):
    """Simple edge-only plot for the freshly imported network before node classification."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(14, 10))

    routentyp_col = next((c for c in edges_gdf.columns if c.upper().startswith('ROUTENTYP')), None)
    if routentyp_col:
        palette = {
            'Velobahn':                       '#1a6b3c',
            'Hauptverbindung':                '#2196F3',
            'Nebenverbindung':                '#9E9E9E',
            'Zusätzliche Freizeitverbindung': '#FF9800',
        }
        handles = []
        for rtype, color in palette.items():
            subset = edges_gdf[edges_gdf[routentyp_col] == rtype]
            if len(subset):
                subset.plot(ax=ax, color=color, linewidth=1.2, alpha=0.8, zorder=2)
                handles.append(mpatches.Patch(color=color, label=f'{rtype} ({len(subset)})'))
        other = edges_gdf[~edges_gdf[routentyp_col].isin(palette)]
        if len(other):
            other.plot(ax=ax, color='#BDBDBD', linewidth=0.8, alpha=0.5, zorder=1)
            handles.append(mpatches.Patch(color='#BDBDBD', label=f'Other ({len(other)})'))
        ax.legend(handles=handles, loc='lower right', fontsize=8, framealpha=0.9)
    else:
        edges_gdf.plot(ax=ax, color='steelblue', linewidth=1.0, alpha=0.7)

    ax.set_title(f'{title}  |  {len(edges_gdf)} edges', fontsize=13)
    ax.set_axis_off()
    plt.tight_layout()
    import os; os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_all_developments(edges_gdf, connectivity_gdf=None, corridor_polygon=None,
                          save_path="data/Network/processed/plot_06_all_developments.png"):
    """
    Four-layer network plot after all developments are assembled:
      grey   — clean existing edges
      orange — Schwachstellen (quality issues)
      red    — Netzlücken (planned gaps)
      purple — auto-generated connectivity bridges
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import geopandas as gpd

    fig, ax = plt.subplots(figsize=(14, 10))

    is_dev = edges_gdf['is_development']   if 'is_development'   in edges_gdf.columns else 0
    is_sw  = edges_gdf['is_schwachstelle'] if 'is_schwachstelle' in edges_gdf.columns else 0

    base = edges_gdf[(is_dev == 0) & (is_sw == 0)]
    sw   = edges_gdf[is_sw == 1]
    nl   = edges_gdf[is_dev == 1]

    base.plot(ax=ax, color='#BDBDBD', linewidth=0.8, alpha=0.7, zorder=1)
    sw.plot(  ax=ax, color='#FF9800', linewidth=2.0, alpha=0.9, zorder=2)
    nl.plot(  ax=ax, color='#F44336', linewidth=2.5, linestyle='--', alpha=0.9, zorder=3)

    handles = [
        mpatches.Patch(color='#BDBDBD', label=f'Existing ({len(base)})'),
        mpatches.Patch(color='#FF9800', label=f'Schwachstellen ({len(sw)})'),
        mpatches.Patch(color='#F44336', label=f'Netzlücken ({len(nl)})'),
    ]

    if connectivity_gdf is not None and len(connectivity_gdf) > 0:
        connectivity_gdf.plot(ax=ax, color='#9C27B0', linewidth=1.8,
                              linestyle=':', alpha=0.85, zorder=4)
        handles.append(mpatches.Patch(color='#9C27B0',
                                      label=f'Connectivity bridges ({len(connectivity_gdf)})'))

    if corridor_polygon is not None:
        gpd.GeoDataFrame(geometry=[corridor_polygon], crs="EPSG:2056").plot(
            ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=5)
        handles.append(mpatches.Patch(facecolor='none', edgecolor='black',
                                      linewidth=1.5, label='Corridor'))

    ax.legend(handles=handles, loc='lower right', fontsize=9)
    ax.set_title("Development candidates — all types", fontsize=13)
    ax.set_axis_off()
    plt.tight_layout()
    import os; os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_combined_vs_status_quo(nodes_gdf, edges_gdf, combined_nodes, combined_edges,
                                 generated_points=None, routed_links=None, corridor_polygon=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import geopandas as gpd

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    titles = ['Status Quo — Existing Network', 'Combined — With New Infrastructure']

    for ax, title, use_combined in zip(axes, titles, [False, True]):

        # Corridor boundary
        if corridor_polygon is not None:
            gpd.GeoDataFrame({'geometry': [corridor_polygon]}, crs="EPSG:2056").boundary.plot(
                ax=ax, color='black', linewidth=1.5, linestyle='--', zorder=1, label='Corridor')

        if use_combined:
            nodes_plot = combined_nodes
            edges_plot = combined_edges

            # Existing edges
            existing_e = edges_plot[edges_plot['source'] == 'existing']
            new_e      = edges_plot[edges_plot['source'] == 'new_link']
            existing_e.plot(ax=ax, color='steelblue', linewidth=1.2, alpha=0.7, zorder=2)
            if len(new_e):
                new_e.plot(ax=ax, color='crimson', linewidth=1.5, alpha=0.85, zorder=3)

            # Existing nodes
            existing_n = nodes_plot[nodes_plot['source'] == 'existing']
            gen_n      = nodes_plot[nodes_plot['source'] == 'generated']

            if 'is_intersection' in existing_n.columns:
                inter = existing_n[existing_n['is_intersection'] == 1]
                ends  = existing_n[existing_n['is_endpoint']     == 1]
                thru  = existing_n[(existing_n['is_intersection'] == 0) & (existing_n['is_endpoint'] == 0)]
                if len(thru):  ax.scatter(thru.geometry.x,  thru.geometry.y,  s=6,  color='gray',       alpha=0.5, zorder=4)
                if len(inter): ax.scatter(inter.geometry.x, inter.geometry.y, s=25, color='orange',     alpha=0.9, zorder=5)
                if len(ends):  ax.scatter(ends.geometry.x,  ends.geometry.y,  s=15, color='salmon',     alpha=0.8, zorder=5)
            else:
                existing_n.plot(ax=ax, color='steelblue', markersize=5, zorder=4)

            if len(gen_n):
                ax.scatter(gen_n.geometry.x, gen_n.geometry.y,
                           s=30, color='crimson', marker='*', alpha=0.9, zorder=6)

            legend_elements = [
                Line2D([0], [0], color='steelblue', linewidth=1.5,  label=f'Existing edges ({len(existing_e)})'),
                Line2D([0], [0], color='crimson',   linewidth=1.5,  label=f'New links ({len(new_e)})'),
                Line2D([0], [0], color='orange',    marker='o', linestyle='None', markersize=7, label=f'Intersections ({len(inter) if "is_intersection" in existing_n.columns else "?"})'),
                Line2D([0], [0], color='crimson',   marker='*', linestyle='None', markersize=10, label=f'Generated points ({len(gen_n)})'),
                Line2D([0], [0], color='black',     linewidth=1.2, linestyle='--', label='Corridor'),
            ]

        else:
            nodes_plot = nodes_gdf
            edges_plot = edges_gdf

            edges_plot.plot(ax=ax, color='steelblue', linewidth=1.2, alpha=0.7, zorder=2)

            if 'is_intersection' in nodes_plot.columns:
                inter = nodes_plot[nodes_plot['is_intersection'] == 1]
                ends  = nodes_plot[nodes_plot['is_endpoint']     == 1]
                thru  = nodes_plot[(nodes_plot['is_intersection'] == 0) & (nodes_plot['is_endpoint'] == 0)]
                if len(thru):  ax.scatter(thru.geometry.x,  thru.geometry.y,  s=6,  color='gray',   alpha=0.5, zorder=3)
                if len(inter): ax.scatter(inter.geometry.x, inter.geometry.y, s=25, color='orange', alpha=0.9, zorder=4)
                if len(ends):  ax.scatter(ends.geometry.x,  ends.geometry.y,  s=15, color='salmon', alpha=0.8, zorder=4)
            else:
                nodes_plot.plot(ax=ax, color='steelblue', markersize=5, zorder=3)

            legend_elements = [
                Line2D([0], [0], color='steelblue', linewidth=1.5, label=f'Edges ({len(edges_plot)})'),
                Line2D([0], [0], color='orange',    marker='o', linestyle='None', markersize=7,  label=f'Intersections'),
                Line2D([0], [0], color='salmon',    marker='o', linestyle='None', markersize=6,  label=f'Dead ends'),
                Line2D([0], [0], color='black',     linewidth=1.2, linestyle='--', label='Corridor'),
            ]

        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Easting (EPSG:2056)')
        ax.set_ylabel('Northing (EPSG:2056)')
        ax.set_aspect('equal')

    # Summary stats in figure text
    n_new_edges = len(combined_edges[combined_edges['source'] == 'new_link'])
    n_gen_nodes = len(combined_nodes[combined_nodes['source'] == 'generated'])
    fig.suptitle(
        f'Cycling Network Comparison  |  '
        f'+{n_gen_nodes} generated nodes  |  +{n_new_edges} new links',
        fontsize=13, fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig('data/Network/combined/network_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved → data/Network/combined/network_comparison.png")


def plot_developments(edges_gdf, corridor_polygon=None, save_path="data/Network/processed/developments_plot.png"):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(14, 10))

    # Base network — existing, no issue
    base = edges_gdf[
        (edges_gdf['is_development'] == 0) & (edges_gdf['is_schwachstelle'] == 0)
    ]
    base.plot(ax=ax, color='lightgrey', linewidth=0.8, zorder=1)

    # Schwachstellen — existing but weak
    sw = edges_gdf[edges_gdf['is_schwachstelle'] == 1]
    sw.plot(ax=ax, color='orange', linewidth=2.0, zorder=2)

    # Netzlücken — planned gaps (is_development == 1)
    nl = edges_gdf[edges_gdf['is_development'] == 1]
    nl.plot(ax=ax, color='red', linewidth=2.5, linestyle='--', zorder=3)

    # Corridor outline
    if corridor_polygon is not None:
        import geopandas as gpd
        gpd.GeoDataFrame(geometry=[corridor_polygon], crs="EPSG:2056").plot(
            ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=4
        )

    legend_handles = [
        mpatches.Patch(color='lightgrey', label=f'Existing network ({len(base)})'),
        mpatches.Patch(color='orange',    label=f'Schwachstellen ({len(sw)})'),
        mpatches.Patch(color='red',       label=f'Netzlücken — planned ({len(nl)})'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=9)
    ax.set_title("ALLTAG Network — Schwachstellen & Netzlücken", fontsize=13)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_network_netzluecken_ids(
        edges_corridor_path="data/Network/processed/edges_corridor.gpkg",
        points_corridor_path="data/Network/processed/points_corridor.gpkg",
        development_candidates_path="data/Network/processed/development_candidates.gpkg",
        save_path="data/Network/processed/network_netzluecken_ids.png"):
    """
    Corridor network with all nodes and edges. Netzlücken are drawn in red dashed
    and annotated with their ID_new at the midpoint of each edge.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    import geopandas as gpd
    import os

    edges = gpd.read_file(edges_corridor_path)
    points = gpd.read_file(points_corridor_path)
    nl = gpd.read_file(development_candidates_path)
    nl = nl[nl['dev_type'] == 'netzluecke'] if 'dev_type' in nl.columns else nl

    is_nl_mask = edges['is_development'] == 1 if 'is_development' in edges.columns else (edges.index < 0)
    base = edges[~is_nl_mask]

    fig, ax = plt.subplots(figsize=(18, 15))

    # base edges
    base.plot(ax=ax, color='#B0BEC5', linewidth=0.8, alpha=0.75, zorder=1)

    # Netzlücken edges (from development_candidates for correct ID mapping)
    nl.plot(ax=ax, color='#E53935', linewidth=2.5, linestyle='--', alpha=0.92, zorder=3)

    # annotate each Netzlücke with its ID at the midpoint
    for _, row in nl.iterrows():
        mid = row.geometry.interpolate(0.5, normalized=True)
        ax.annotate(
            str(int(row['ID_new'])),
            xy=(mid.x, mid.y),
            xytext=(4, 4),
            textcoords='offset points',
            fontsize=7,
            fontweight='bold',
            color='#B71C1C',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#E53935',
                      linewidth=0.8, alpha=0.85),
            zorder=5,
        )

    # nodes — colour by type
    NODE_STYLE = {
        'is_intersection':  dict(color='#1565C0', size=20, marker='o', label='Intersection',  zorder=6),
        'is_through_point': dict(color='#43A047', size=8,  marker='o', label='Through-point', zorder=5),
        'is_endpoint':      dict(color='#FF6F00', size=28, marker='^', label='Dead-end',      zorder=7),
    }
    node_handles = []
    for col, style in NODE_STYLE.items():
        if col not in points.columns:
            continue
        subset = points[points[col] == 1]
        if len(subset) == 0:
            continue
        ax.scatter(subset.geometry.x, subset.geometry.y,
                   s=style['size'], c=style['color'],
                   marker=style['marker'], zorder=style['zorder'], alpha=0.9)
        node_handles.append(mpatches.Patch(color=style['color'],
                                           label=f"{style['label']} ({len(subset)})"))

    # legend
    edge_handles = [
        mlines.Line2D([], [], color='#B0BEC5', linewidth=1.5, label=f'Existing edges ({len(base)})'),
        mlines.Line2D([], [], color='#E53935', linewidth=2.0, linestyle='--',
                      label=f'Netzlücken ({len(nl)}) — labelled by ID'),
    ]
    leg1 = ax.legend(handles=edge_handles, title='Edges', loc='lower left',
                     fontsize=9, title_fontsize=10, framealpha=0.93)
    ax.add_artist(leg1)
    ax.legend(handles=node_handles, title='Nodes', loc='lower right',
              fontsize=9, title_fontsize=10, framealpha=0.93)

    ax.set_title(
        f'Corridor Network — Netzlücken highlighted with ID numbers\n'
        f'{len(edges)} edges total  |  {len(points)} nodes  |  {len(nl)} Netzlücken',
        fontsize=13, fontweight='bold')
    ax.set_axis_off()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved → {save_path}")