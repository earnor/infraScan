





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
    Three-panel OD summary (Voronoi-weighted, presentation quality):

      Left   — Origin strength map: bubbles sized by population, coloured by
               employment (destination attractiveness), YlOrRd colormap
      Centre — Population vs cycling trips by scenario on twin Y-axes so both
               series are readable at their own scale
      Right  — Top-20 origin nodes ranked by daily cycling trips (s2),
               sorted descending, x-gridlines, value annotations
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker
    import numpy as np
    import geopandas as gpd
    import os

    LABELS = {'s1': 'Low (s1)', 's2': 'Medium (s2)', 's3': 'High (s3)'}

    # ── figure — constrained_layout keeps titles/labels from overlapping ──────
    fig, axes = plt.subplots(
        1, 3, figsize=(24, 10),
        gridspec_kw={'width_ratios': [2.5, 1.5, 2]},
        constrained_layout=True)
    ax_map, ax_bar, ax_rank = axes

    # ── build node pop / empl lookup ──────────────────────────────────────────
    voronoi_vals = voronoi_vals.copy()
    voronoi_vals['ID_point'] = voronoi_vals['ID_point'].astype(int)

    ref_s    = next((s for s in ['s2', 's1', 's3'] if s in od_scenarios), None)
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

    if ref_s and ref_s in od_scenarios:
        outflow = (od_scenarios[ref_s]
                   .groupby('origin_id')['trips'].sum()
                   .reset_index()
                   .rename(columns={'origin_id': 'ID_point', 'trips': 'outflow'}))
        node_df = node_df.merge(outflow, on='ID_point', how='left')
        node_df['outflow'] = node_df['outflow'].fillna(0)
    else:
        node_df['outflow'] = 0

    # ── Panel 1: origin strength map ──────────────────────────────────────────
    if corridor_polygon is not None:
        gpd.GeoDataFrame(geometry=[corridor_polygon], crs="EPSG:2056").plot(
            ax=ax_map, facecolor='#EEF3F9', edgecolor='#607D8B',
            linewidth=1.4, zorder=0, alpha=0.9)

    if lakes_gdf is not None and len(lakes_gdf) > 0:
        lakes_gdf.plot(ax=ax_map, facecolor='#AED6F1', edgecolor='#5DADE2',
                       linewidth=0.6, zorder=1, alpha=0.85)

    if edges_gdf is not None and len(edges_gdf) > 0:
        edges_gdf.plot(ax=ax_map, color='#9E9E9E', linewidth=0.5, zorder=2, alpha=0.55)

    # faint base dots so every node is visible even without a bubble
    points_gdf.plot(ax=ax_map, color='#B0BEC5', markersize=5, zorder=3, alpha=0.55)

    # scaled + coloured bubbles
    max_pop  = node_df['pop'].max()  or 1
    max_empl = node_df['empl'].max() or 1
    sizes    = 50 + 550 * (node_df['pop'] / max_pop)          # min 50, max 600
    norm_empl = mcolors.Normalize(vmin=0, vmax=max_empl)
    cmap_empl = cm.get_cmap('YlOrRd')
    colors    = [cmap_empl(norm_empl(v)) for v in node_df['empl']]

    ax_map.scatter(node_df.geometry.x, node_df.geometry.y,
                   s=sizes, c=colors, zorder=5, alpha=0.92,
                   edgecolors='white', linewidths=0.7)

    sm = cm.ScalarMappable(cmap=cmap_empl, norm=norm_empl)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_map, shrink=0.55, pad=0.04)
    cbar.set_label('Employment in Voronoi cell\n(destination attractiveness)',
                   fontsize=11, labelpad=12)
    cbar.ax.tick_params(labelsize=9)

    # population size legend — three representative sizes
    for frac, lbl in [(0.25, f'{max_pop*0.25:,.0f}'),
                      (0.60, f'{max_pop*0.60:,.0f}'),
                      (1.00, f'{max_pop:,.0f}')]:
        s = 50 + 550 * frac
        ax_map.scatter([], [], s=s, color='#78909C', alpha=0.85,
                       edgecolors='white', linewidths=0.7, label=f'Pop ≈ {lbl}')
    ax_map.legend(title='Bubble size = population', loc='lower left',
                  fontsize=10, title_fontsize=11, framealpha=0.93,
                  borderpad=0.9, labelspacing=0.5)

    ax_map.set_title(
        f'Origin strength (population) &\ndestination attractiveness (employment)'
        f'\nScenario {ref_s or "—"}  —  {len(node_df)} nodes',
        fontsize=12, pad=8)
    ax_map.set_axis_off()

    if corridor_polygon is not None:
        minx, miny, maxx, maxy = corridor_polygon.bounds
        ax_map.set_xlim(minx - 500, maxx + 500)
        ax_map.set_ylim(miny - 500, maxy + 500)

    # ── Panel 2: population vs cycling trips — twin Y-axes ────────────────────
    s_keys = [s for s in scenarios if s in od_scenarios]
    totals = [od_scenarios[s]['trips'].sum() for s in s_keys]
    pops   = [voronoi_vals[f'{s}_pop'].clip(lower=0).fillna(0).sum() for s in s_keys]

    x = np.arange(len(s_keys))
    w = 0.34

    COLOR_POP   = '#2471A3'   # blue
    COLOR_TRIPS = '#CA6F1E'   # orange — clearly distinct from blue

    # left axis: population bars
    b1 = ax_bar.bar(x - w / 2, pops, width=w,
                    color=COLOR_POP, edgecolor='white', alpha=0.88,
                    label='Total origin population', zorder=3)
    ax_bar.set_ylabel('Total origin population', fontsize=11, color=COLOR_POP, labelpad=8)
    ax_bar.tick_params(axis='y', labelcolor=COLOR_POP, labelsize=10)
    ax_bar.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))

    # right axis: cycling trips bars
    ax_bar2 = ax_bar.twinx()
    b2 = ax_bar2.bar(x + w / 2, totals, width=w,
                     color=COLOR_TRIPS, edgecolor='white', alpha=0.88,
                     label='Daily cycling trips', zorder=3)
    ax_bar2.set_ylabel('Daily cycling trips', fontsize=11, color=COLOR_TRIPS, labelpad=8)
    ax_bar2.tick_params(axis='y', labelcolor=COLOR_TRIPS, labelsize=10)
    ax_bar2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))

    # annotations on top of each bar
    for bar, val in zip(b1, pops):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.015,
                    f'{val:,.0f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=COLOR_POP)
    for bar, val in zip(b2, totals):
        ax_bar2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() * 1.015,
                     f'{val:,.0f}', ha='center', va='bottom',
                     fontsize=9, fontweight='bold', color=COLOR_TRIPS)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([LABELS.get(s, s) for s in s_keys], fontsize=10)
    ax_bar.set_title('Population vs cycling trips\nby scenario', fontsize=12, pad=8)
    ax_bar.spines[['top']].set_visible(False)
    ax_bar2.spines[['top']].set_visible(False)
    ax_bar.yaxis.grid(True, linestyle='--', alpha=0.35, zorder=0)
    ax_bar.set_axisbelow(True)

    # single combined legend
    ax_bar.legend(handles=[b1[0], b2[0]],
                  labels=['Total origin population', 'Daily cycling trips'],
                  fontsize=10, framealpha=0.92, loc='upper left')

    # ── Panel 3: top-20 origin nodes ──────────────────────────────────────────
    if ref_s and ref_s in od_scenarios:
        top20 = (od_scenarios[ref_s]
                 .groupby('origin_id')['trips'].sum()
                 .nlargest(20)
                 .reset_index()
                 .sort_values('trips', ascending=True))   # ascending → largest at top

        y_pos = np.arange(len(top20))
        bars  = ax_rank.barh(y_pos, top20['trips'],
                             color='#2471A3', edgecolor='white',
                             height=0.70, alpha=0.90)

        ax_rank.set_yticks(y_pos)
        ax_rank.set_yticklabels(
            [f"Node {int(r)}" for r in top20['origin_id']], fontsize=10)
        ax_rank.set_xlabel('Daily cycling trips generated', fontsize=11, labelpad=8)
        ax_rank.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
        ax_rank.tick_params(axis='x', labelsize=10)
        ax_rank.set_title(f'Top 20 origin nodes  (scenario {ref_s})',
                          fontsize=12, pad=8)

        # x gridlines behind the bars
        ax_rank.xaxis.grid(True, linestyle='--', alpha=0.45, zorder=0)
        ax_rank.set_axisbelow(True)
        ax_rank.spines[['top', 'right']].set_visible(False)

        # value annotations at bar end
        x_max = top20['trips'].max()
        for bar, val in zip(bars, top20['trips']):
            ax_rank.text(val + x_max * 0.012,
                         bar.get_y() + bar.get_height() / 2,
                         f'{val:,.0f}',
                         va='center', ha='left', fontsize=9, color='#1A5276')
        ax_rank.set_xlim(right=x_max * 1.20)
    else:
        ax_rank.set_axis_off()

    # ── overall title ──────────────────────────────────────────────────────────
    fig.suptitle('OD Matrix — Voronoi-weighted Cycling Flows',
                 fontsize=16, fontweight='bold')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_od_map(
        voronoi_vals,
        points_gdf,
        corridor_polygon=None,
        edges_gdf=None,
        lakes_gdf=None,
        ref_s='s2',
        save_path="data/OD/plot1_map.png"):
    """
    Plot 1 — Standalone origin-strength map.
    Bubble size = population in Voronoi cell, colour = employment (destination
    attractiveness), YlOrRd colormap.  Figure size (12, 10), dpi=150.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import geopandas as gpd
    import os

    voronoi_vals = voronoi_vals.copy()
    voronoi_vals['ID_point'] = voronoi_vals['ID_point'].astype(int)
    pop_col  = f'{ref_s}_pop'
    empl_col = f'{ref_s}_empl'

    node_df = points_gdf[['ID_point', 'geometry']].copy()
    node_df['ID_point'] = node_df['ID_point'].astype(int)

    if pop_col in voronoi_vals.columns:
        node_df = node_df.merge(
            voronoi_vals[['ID_point', pop_col, empl_col]].rename(
                columns={pop_col: 'pop', empl_col: 'empl'}),
            on='ID_point', how='left')
        node_df[['pop', 'empl']] = node_df[['pop', 'empl']].fillna(0)
    else:
        node_df['pop'] = node_df['empl'] = 0

    fig, ax = plt.subplots(figsize=(12, 10), constrained_layout=True,
                           facecolor='white')
    ax.set_facecolor('white')

    if corridor_polygon is not None:
        gpd.GeoDataFrame(geometry=[corridor_polygon], crs="EPSG:2056").plot(
            ax=ax, facecolor='#EEF3F9', edgecolor='#607D8B',
            linewidth=1.4, zorder=0, alpha=0.9)

    if lakes_gdf is not None and len(lakes_gdf) > 0:
        lakes_gdf.plot(ax=ax, facecolor='#AED6F1', edgecolor='#5DADE2',
                       linewidth=0.6, zorder=1, alpha=0.85)

    if edges_gdf is not None and len(edges_gdf) > 0:
        edges_gdf.plot(ax=ax, color='#9E9E9E', linewidth=0.55,
                       zorder=2, alpha=0.55)

    # faint base dot for every node so isolated nodes are visible
    points_gdf.plot(ax=ax, color='#B0BEC5', markersize=5, zorder=3, alpha=0.55)

    # ── scaled coloured bubbles ───────────────────────────────────────────────
    max_pop   = node_df['pop'].max()  or 1
    max_empl  = node_df['empl'].max() or 1
    sizes     = 50 + 550 * (node_df['pop'] / max_pop)
    norm_empl = mcolors.Normalize(vmin=0, vmax=max_empl)
    cmap_empl = cm.get_cmap('YlOrRd')
    colors    = [cmap_empl(norm_empl(v)) for v in node_df['empl']]

    ax.scatter(node_df.geometry.x, node_df.geometry.y,
               s=sizes, c=colors, zorder=5, alpha=0.92,
               edgecolors='white', linewidths=0.8)

    sm = cm.ScalarMappable(cmap=cmap_empl, norm=norm_empl)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.04)
    cbar.set_label('Employment in Voronoi cell\n(destination attractiveness)',
                   fontsize=11, labelpad=14)
    cbar.ax.tick_params(labelsize=9)

    # ── population size legend (3 representative sizes) ───────────────────────
    for frac, lbl in [(0.25, f'{max_pop * 0.25:,.0f}'),
                      (0.60, f'{max_pop * 0.60:,.0f}'),
                      (1.00, f'{max_pop:,.0f}')]:
        ax.scatter([], [], s=50 + 550 * frac,
                   color='#78909C', alpha=0.85,
                   edgecolors='white', linewidths=0.8,
                   label=f'Pop ≈ {lbl}')
    ax.legend(title='Bubble size = population', loc='lower left',
              fontsize=10, title_fontsize=11, framealpha=0.93,
              borderpad=0.9, labelspacing=0.55)

    ax.set_title(
        'Origin Strength (Population) &\nDestination Attractiveness (Employment)',
        fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel(f'Scenario {ref_s}  —  {len(node_df)} nodes',
                  fontsize=11, labelpad=6)
    ax.set_axis_off()
    # subtitle as plain text below the map (bbox_inches='tight' will capture it)
    fig.text(0.5, 0.01, f'Scenario {ref_s}  —  {len(node_df)} nodes',
             ha='center', va='bottom', fontsize=11, color='#444444')

    if corridor_polygon is not None:
        minx, miny, maxx, maxy = corridor_polygon.bounds
        ax.set_xlim(minx - 500, maxx + 500)
        ax.set_ylim(miny - 500, maxy + 500)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_od_scenario_bars(
        od_scenarios,
        voronoi_vals,
        scenarios=('s1', 's2', 's3'),
        save_path="data/OD/plot2_trips.png"):
    """
    Plot 2 — Small multiples: stacked subplots instead of dual-axis.
    Top panel:    total origin population per scenario (blue bars).
    Bottom panel: daily cycling trips per scenario (orange bars).
    Each series has its own y-axis so the ~14× scale difference is not hidden.
    Figure size (10, 8), dpi=150.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    import os

    LABELS      = {'s1': 'Low\n(s1)', 's2': 'Medium\n(s2)', 's3': 'High\n(s3)'}
    COLOR_POP   = '#2471A3'   # blue — consistent with Plot 1 & 3
    COLOR_TRIPS = '#CA6F1E'   # orange

    s_keys   = [s for s in scenarios if s in od_scenarios]
    totals   = [od_scenarios[s]['trips'].sum() for s in s_keys]
    pops     = [voronoi_vals[f'{s}_pop'].clip(lower=0).fillna(0).sum()
                for s in s_keys]
    x_labels = [LABELS.get(s, s) for s in s_keys]
    x        = np.arange(len(s_keys))
    w        = 0.50

    fig, (ax_pop, ax_trips) = plt.subplots(
        2, 1, figsize=(10, 8),
        constrained_layout=True,
        sharex=True)

    # ── top: population ───────────────────────────────────────────────────────
    bars1 = ax_pop.bar(x, pops, width=w,
                       color=COLOR_POP, edgecolor='white', alpha=0.88)
    ax_pop.set_ylabel('Total origin population', fontsize=11, labelpad=8)
    ax_pop.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax_pop.tick_params(axis='y', labelsize=10)
    ax_pop.set_title('Total Origin Population per Scenario', fontsize=12, pad=6)
    ax_pop.spines[['top', 'right']].set_visible(False)
    ax_pop.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax_pop.set_axisbelow(True)
    for bar, val in zip(bars1, pops):
        ax_pop.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.015,
                    f'{val:,.0f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=COLOR_POP)

    # ── bottom: cycling trips ─────────────────────────────────────────────────
    bars2 = ax_trips.bar(x, totals, width=w,
                         color=COLOR_TRIPS, edgecolor='white', alpha=0.88)
    ax_trips.set_ylabel('Daily cycling trips', fontsize=11, labelpad=8)
    ax_trips.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax_trips.tick_params(axis='y', labelsize=10)
    ax_trips.set_title('Daily Cycling Trips per Scenario', fontsize=12, pad=6)
    ax_trips.spines[['top', 'right']].set_visible(False)
    ax_trips.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax_trips.set_axisbelow(True)
    ax_trips.set_xticks(x)
    ax_trips.set_xticklabels(x_labels, fontsize=10)
    ax_trips.set_xlabel('Scenario', fontsize=11, labelpad=8)
    for bar, val in zip(bars2, totals):
        ax_trips.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() * 1.015,
                      f'{val:,.0f}', ha='center', va='bottom',
                      fontsize=9, fontweight='bold', color=COLOR_TRIPS)

    fig.suptitle('Population and Daily Cycling Trips by Scenario',
                 fontsize=14, fontweight='bold')



    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Plot saved → {save_path}")


def plot_od_top20_nodes(
        od_scenarios,
        points_gdf,
        edges_gdf=None,
        corridor_polygon=None,
        lakes_gdf=None,
        ref_s='s2',
        save_path="data/OD/plot3_nodes.png"):
    """
    Plot 3 — Side-by-side: top-20 horizontal bar chart (left) + network map
    with top-20 nodes overlaid and labelled (right).
    Figure size (20, 10), width ratios [1, 1.8], dpi=150.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker
    import numpy as np
    import geopandas as gpd
    import os

    COLOR_BAR = '#2471A3'

    if ref_s not in od_scenarios:
        print(f"  [WARN] scenario {ref_s} not in od_scenarios — skipping plot3")
        return

    # top-20 sorted ascending so largest bar ends up at the top of the chart
    top20 = (od_scenarios[ref_s]
             .groupby('origin_id')['trips'].sum()
             .nlargest(20)
             .reset_index()
             .sort_values('trips', ascending=True)
             .rename(columns={'origin_id': 'ID_point'}))
    top20['ID_point'] = top20['ID_point'].astype(int)

    fig, (ax_bar, ax_map) = plt.subplots(
        1, 2, figsize=(20, 10),
        gridspec_kw={'width_ratios': [1, 1.8]},
        constrained_layout=True,
        facecolor='white')

    # ── left: horizontal bar chart ────────────────────────────────────────────
    y_pos = np.arange(len(top20))
    bars  = ax_bar.barh(y_pos, top20['trips'],
                        color=COLOR_BAR, edgecolor='white',
                        height=0.68, alpha=0.90)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(
        [f"Node {r}" for r in top20['ID_point']], fontsize=10)
    ax_bar.set_xlabel('Daily cycling trips generated', fontsize=11, labelpad=8)
    ax_bar.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax_bar.tick_params(axis='x', labelsize=10)
    ax_bar.set_title(f'Top 20 Origin Nodes\n(scenario {ref_s})',
                     fontsize=12, pad=8)
    ax_bar.xaxis.grid(True, linestyle='--', alpha=0.45, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.spines[['top', 'right']].set_visible(False)

    x_max = top20['trips'].max()
    for bar, val in zip(bars, top20['trips']):
        ax_bar.text(val + x_max * 0.012,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:,.0f}',
                    va='center', ha='left', fontsize=9, color='#1A5276')
    ax_bar.set_xlim(right=x_max * 1.22)

    # ── right: network map ────────────────────────────────────────────────────
    ax_map.set_facecolor('white')

    if corridor_polygon is not None:
        gpd.GeoDataFrame(geometry=[corridor_polygon], crs="EPSG:2056").plot(
            ax=ax_map, facecolor='#EEF3F9', edgecolor='#607D8B',
            linewidth=1.4, zorder=0, alpha=0.9)

    if lakes_gdf is not None and len(lakes_gdf) > 0:
        lakes_gdf.plot(ax=ax_map, facecolor='#AED6F1', edgecolor='#5DADE2',
                       linewidth=0.6, zorder=1, alpha=0.85)

    if edges_gdf is not None and len(edges_gdf) > 0:
        edges_gdf.plot(ax=ax_map, color='#9E9E9E', linewidth=0.55,
                       zorder=2, alpha=0.55)

    # all corridor nodes as small grey dots
    points_gdf.plot(ax=ax_map, color='#B0BEC5', markersize=6,
                    zorder=3, alpha=0.60)

    # ── top-20 nodes overlaid with trip-count colormap ────────────────────────
    pts = points_gdf.copy()
    pts['ID_point'] = pts['ID_point'].astype(int)
    top20_geo = top20.merge(pts[['ID_point', 'geometry']], on='ID_point', how='left')
    top20_geo = gpd.GeoDataFrame(top20_geo, geometry='geometry', crs=points_gdf.crs)
    top20_geo = top20_geo.dropna(subset=['geometry'])

    if len(top20_geo) > 0:
        norm_t = mcolors.Normalize(vmin=top20_geo['trips'].min(),
                                   vmax=top20_geo['trips'].max())
        cmap_t = cm.get_cmap('YlOrRd')

        ax_map.scatter(top20_geo.geometry.x, top20_geo.geometry.y,
                       c=[cmap_t(norm_t(v)) for v in top20_geo['trips']],
                       s=220, zorder=6,
                       edgecolors='white', linewidths=1.2, alpha=0.97)

        # node ID labels — small white-backed box, slight NE offset
        for _, row in top20_geo.iterrows():
            ax_map.annotate(
                str(int(row['ID_point'])),
                xy=(row.geometry.x, row.geometry.y),
                xytext=(6, 6), textcoords='offset points',
                fontsize=8.5, fontweight='bold', color='#111111', zorder=9,
                bbox=dict(boxstyle='round,pad=0.22', facecolor='white',
                          edgecolor='#AAAAAA', linewidth=0.7, alpha=0.88))

        sm = cm.ScalarMappable(cmap=cmap_t, norm=norm_t)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_map, shrink=0.55, pad=0.04)
        cbar.set_label('Daily cycling trips', fontsize=11, labelpad=12)
        cbar.ax.tick_params(labelsize=9)
        cbar.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))

    ax_map.set_title(
        f'Top 20 Origin Nodes by Daily Cycling Trips\n(Scenario {ref_s})',
        fontsize=12, pad=8)
    ax_map.set_axis_off()

    if corridor_polygon is not None:
        minx, miny, maxx, maxy = corridor_polygon.bounds
        ax_map.set_xlim(minx - 500, maxx + 500)
        ax_map.set_ylim(miny - 500, maxy + 500)

    fig.suptitle('Top 20 Origin Nodes — Bar Chart & Network Location',
                 fontsize=14, fontweight='bold')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
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


def plot_corridor_netzluecken_highlighted(
        edges_aug,
        points_corridor,
        corridor_polygon=None,
        save_path="data/Network/processed/network_netzluecken_highlighted.pdf"):
    """
    Presentation-quality corridor network plot — Netzlücken highlighted and
    labelled by ID.  Saved as vector PDF so slides stay crisp at any zoom.

    Visual hierarchy (high-contrast)
    ---------------------------------
    • White background, pale-blue corridor fill → maximum contrast base
    • Medium-dark grey lines (lw 1.6)  → existing edges
    • Thick crimson dashes  (lw 5.0)   → Netzlücken, each labelled with its
                                          ID_edge in a white box with red border
    • Royal-blue circles    (s 120)    → intersections
    • Forest-green circles  (s  60)    → through-points
    • Orange triangles      (s 160)    → dead-ends
    All text ≥ 14 pt so it reads on a projected slide.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    import matplotlib as mpl
    import geopandas as gpd
    import os

    # ── global font sizes for presentation readability ────────────────────────
    mpl.rcParams.update({
        'font.size':        14,
        'axes.titlesize':   18,
        'legend.fontsize':  13,
        'legend.title_fontsize': 14,
    })

    # ── figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(20, 15), facecolor='white')
    ax.set_facecolor('white')

    # ── corridor backdrop ─────────────────────────────────────────────────────
    if corridor_polygon is not None:
        gpd.GeoDataFrame(geometry=[corridor_polygon], crs="EPSG:2056").plot(
            ax=ax, facecolor='#E8EEF5', edgecolor='#111111',
            linewidth=2.2, linestyle='--', zorder=0, alpha=1.0)

    # ── split edges ───────────────────────────────────────────────────────────
    if 'ROUTENTYP' in edges_aug.columns:
        mask_nl  = edges_aug['ROUTENTYP'] == 'Netzlücke'
        existing = edges_aug[~mask_nl]
        nl       = edges_aug[mask_nl]
    else:
        existing = edges_aug
        nl       = edges_aug.iloc[0:0]

    # ── existing edges: dark enough to read, thin enough not to dominate ──────
    existing.plot(ax=ax, color='#444444', linewidth=1.6, alpha=0.85, zorder=1)

    # ── Netzlücken: thick, vivid, unmissable ─────────────────────────────────
    nl.plot(ax=ax, color='#D10000', linewidth=5.0, linestyle='--',
            alpha=1.0, zorder=4)

    # ── ID labels on Netzlücken ───────────────────────────────────────────────
    id_col = next((c for c in ('ID_edge', 'ID_new', 'id')
                   if c in nl.columns), None)
    if id_col is not None:
        for _, row in nl.iterrows():
            mid = row.geometry.interpolate(0.5, normalized=True)
            ax.annotate(
                str(int(row[id_col])),
                xy=(mid.x, mid.y),
                fontsize=11, fontweight='bold',
                color='#D10000',
                ha='center', va='center', zorder=9,
                bbox=dict(boxstyle='round,pad=0.35',
                          facecolor='white', edgecolor='#D10000',
                          linewidth=1.6, alpha=0.97),
            )

    # ── nodes ─────────────────────────────────────────────────────────────────
    NODE_CFG = {
        'is_intersection':  dict(color='#1040A0', size=120, marker='o',
                                 label='Intersection',  zorder=6, lw=1.2),
        'is_through_point': dict(color='#1A7A1A', size=60,  marker='o',
                                 label='Through-point', zorder=5, lw=0.9),
        'is_endpoint':      dict(color='#E06000', size=160, marker='^',
                                 label='Dead-end',      zorder=7, lw=1.2),
    }
    node_handles = []
    for col, cfg in NODE_CFG.items():
        if col not in points_corridor.columns:
            continue
        sub = points_corridor[points_corridor[col] == 1]
        if len(sub) == 0:
            continue
        ax.scatter(sub.geometry.x, sub.geometry.y,
                   c=cfg['color'], s=cfg['size'],
                   marker=cfg['marker'], zorder=cfg['zorder'],
                   edgecolors='white', linewidths=cfg['lw'], alpha=1.0)
        node_handles.append(
            ax.scatter([], [], c=cfg['color'], s=cfg['size'] + 20,
                       marker=cfg['marker'],
                       edgecolors='white', linewidths=cfg['lw'],
                       label=f"{cfg['label']} ({len(sub)})"))

    # ── legends — placed outside the axes below the map ──────────────────────
    # Using fig.legend() with bbox_to_anchor in figure coordinates so the
    # boxes sit in the reserved bottom margin, never touching the map content.
    edge_handles = [
        mlines.Line2D([], [], color='#444444', linewidth=2.5,
                      label=f'Existing edges ({len(existing)})'),
        mlines.Line2D([], [], color='#D10000', linewidth=4.0,
                      linestyle='--',
                      label=f'Netzlücken ({len(nl)}) — labelled by ID'),
    ]
    fig.legend(
        handles=edge_handles, title='Edges',
        loc='lower left', bbox_to_anchor=(0.04, 0.01),
        framealpha=0.97, edgecolor='#999999',
        borderpad=1.0, labelspacing=0.65, handlelength=2.5,
        fontsize=13, title_fontsize=14)
    fig.legend(
        handles=node_handles, title='Nodes',
        loc='lower right', bbox_to_anchor=(0.96, 0.01),
        framealpha=0.97, edgecolor='#999999',
        borderpad=1.0, labelspacing=0.65, handlelength=2.0,
        fontsize=13, title_fontsize=14)

    # ── title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        f'Corridor Network — Netzlücken highlighted with ID numbers\n'
        f'{len(edges_aug)} edges total  |  {len(points_corridor)} nodes'
        f'  |  {len(nl)} Netzlücken',
        fontweight='bold', pad=16)
    ax.set_axis_off()

    # reserve space at the bottom for the two legend boxes
    plt.subplots_adjust(bottom=0.13)
    plt.tight_layout(rect=[0, 0.13, 1, 1])
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()

    # restore matplotlib defaults so other plots are unaffected
    mpl.rcdefaults()
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
