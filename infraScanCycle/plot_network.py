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