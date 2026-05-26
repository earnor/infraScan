


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
