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
    if rt_col and rt_col in combined.columns:
        counts = combined[rt_col].value_counts()
        colors = [color_map.get(str(k), '#BDBDBD') for k in counts.index]
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