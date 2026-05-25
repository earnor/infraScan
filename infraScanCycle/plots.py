import numpy as np
import rasterio
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib_scalebar.scalebar import ScaleBar
import rasterio.plot
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import math
import matplotlib.lines as mlines


def generate_report_figures():
    """
    Generates all figures referenced in 05_Results.tex and saves them to figures/.

    Figures produced
    ----------------
    figures/network_all_types.png   — copied from pipeline output
    figures/od_plot.png             — copied from pipeline output
    figures/detour_distribution.png — histogram of OD detour factors
    figures/accessibility_best.png  — node accessibility map, ID 800 (best NB)
    figures/accessibility_worst.png — node accessibility map, ID 385 (worst NB)
    figures/safety_index_map.png    — per-link crash-rate coloured map
    figures/elevation_map.png       — DEM hillshade + contours + network overlay
    figures/comfort_index_map.png   — per-link comfort index (alpha x epsilon) map
    """
    import shutil
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.cm as mcm
    from matplotlib.colors import LinearSegmentedColormap, LightSource
    from matplotlib.patches import FancyArrowPatch
    from matplotlib_scalebar.scalebar import ScaleBar
    import rasterio
    import rasterio.plot
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    os.makedirs('figures', exist_ok=True)

    # ── shared data layers ────────────────────────────────────────────────────
    network_edges = gpd.read_file('data/Network/processed/edges_with_attribute.gpkg')
    corridor_pts  = gpd.read_file('data/Network/processed/points_corridor.gpkg')
    lakes_path    = 'data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp'
    cities_path   = 'data/manually_gathered_data/Cities.shp'

    net_bounds = network_edges.total_bounds   # [minx, miny, maxx, maxy]
    x_pad, y_pad = 500, 500

    def _add_base(ax):
        """Add lakes, grey network, city labels, scale bar, north arrow."""
        if os.path.exists(lakes_path):
            gpd.read_file(lakes_path).plot(ax=ax, color='lightblue', zorder=1)
        network_edges.plot(ax=ax, color='#bbbbbb', lw=0.8, zorder=2, alpha=0.6)
        if os.path.exists(cities_path):
            cities = gpd.read_file(cities_path, crs='epsg:2056')
            cities.plot(ax=ax, color='black', markersize=50, zorder=8)
            for _, r in cities.iterrows():
                ax.annotate(r['location'], xy=r.geometry.coords[0],
                            ha='center', va='top', xytext=(0, -5),
                            textcoords='offset points', fontsize=10, zorder=8)
        ax.add_artist(ScaleBar(1, location='lower right'))
        ax.text(0.96, 0.93, 'N', fontsize=22, weight='bold',
                ha='center', va='center', transform=ax.transAxes, zorder=100)
        ax.add_patch(FancyArrowPatch(
            (0.96, 0.90), (0.96, 0.97), color='black', lw=1.5,
            arrowstyle='->', mutation_scale=20, transform=ax.transAxes, zorder=100))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor('black'); sp.set_linewidth(1)
        ax.set_xlim(net_bounds[0] - x_pad, net_bounds[2] + x_pad)
        ax.set_ylim(net_bounds[1] - y_pad, net_bounds[3] + y_pad)

    # ── 1. Copy pipeline figures ──────────────────────────────────────────────
    for src, dst in [
        ('data/Network/processed/network_all_types.png', 'figures/network_all_types.png'),
        ('data/OD/od_plot.png',                          'figures/od_plot.png'),
    ]:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f'  copied  {src} → {dst}')
        else:
            print(f'  [WARN] pipeline figure not found: {src}')

    # ── 2. Detour factor histogram ────────────────────────────────────────────
    od_path = 'data/OD/od_base_travel_times.csv'
    if os.path.exists(od_path):
        od        = pd.read_csv(od_path)
        df_clipped = od['detour_factor'].clip(upper=20)
        median_v  = od['detour_factor'].median()
        mean_v    = od['detour_factor'].mean()
        pct95_v   = od['detour_factor'].quantile(0.95)
        max_v     = od['detour_factor'].max()

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(df_clipped, bins=80, color='steelblue', edgecolor='white',
                linewidth=0.4, zorder=3)
        ax.axvline(median_v, color='#e74c3c', lw=1.8, linestyle='--',
                   label=f'Median = {median_v:.2f}')
        ax.axvline(mean_v,   color='#e67e22', lw=1.8, linestyle=':',
                   label=f'Mean = {mean_v:.2f}')
        ax.set_xlabel('Detour factor  (routed distance / Euclidean distance)', fontsize=12)
        ax.set_ylabel('Number of OD pairs', fontsize=12)
        ax.set_title(
            f'Detour factor distribution across {len(od):,} base-graph OD pairs\n'
            f'(x-axis capped at 20; true max = {max_v:.1f})',
            fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.text(0.97, 0.97,
                f'95th pct = {pct95_v:.2f}\nMax = {max_v:.1f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig('figures/detour_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('  saved   figures/detour_distribution.png')

    # ── 3. Accessibility maps — best (800) and worst (385) ────────────────────
    cands_path = 'data/Network/processed/development_candidates.gpkg'
    if os.path.exists(cands_path) and len(corridor_pts) > 0:
        cands = gpd.read_file(cands_path)
        cands['ID_new'] = cands['ID_new'].astype(int)

        for dev_id, fname, subtitle in [
            (800, 'figures/accessibility_best.png',
             'Best-performing development  (ID 800, 35 m Velobahn)'),
            (385, 'figures/accessibility_worst.png',
             'Worst-performing development  (ID 385, 6.8 km Nebenverbindung)'),
        ]:
            dev_row = cands[cands['ID_new'] == dev_id]
            if dev_row.empty:
                print(f'  [WARN] ID {dev_id} not in development_candidates — skipping')
                continue

            dev_geom = dev_row.iloc[0].geometry
            pts = corridor_pts.copy()

            # accessibility gain proxy: 1 / (1 + distance-to-dev [km])
            pts['dist_km']  = pts.geometry.distance(dev_geom) / 1000.0
            pts['acc_gain'] = 1.0 / (1.0 + pts['dist_km'])
            gain_min, gain_max = pts['acc_gain'].min(), pts['acc_gain'].max()
            pts['acc_norm'] = (pts['acc_gain'] - gain_min) / (gain_max - gain_min + 1e-9)

            fig, ax = plt.subplots(figsize=(13, 9))
            _add_base(ax)

            sc = ax.scatter(
                pts.geometry.x, pts.geometry.y,
                c=pts['acc_norm'], cmap='YlOrRd', s=45,
                vmin=0, vmax=1, zorder=6, edgecolors='none', alpha=0.9,
            )
            dev_row.plot(ax=ax, color='red', lw=4, zorder=10)
            # label the development
            mid = dev_geom.interpolate(0.5, normalized=True)
            ax.annotate(f'ID {dev_id}', xy=(mid.x, mid.y),
                        xytext=(6, 6), textcoords='offset points',
                        fontsize=10, fontweight='bold', color='red', zorder=11)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='2%', pad=0.4)
            cbar = plt.colorbar(sc, cax=cax)
            cbar.set_label('Normalised accessibility gain\n(1 / (1 + distance to development [km]))',
                           rotation=90, labelpad=12, fontsize=10)

            leg = [mpatches.Patch(color='red', label=f'Development ID {dev_id} (highlighted)')]
            ax.legend(handles=leg, loc='upper left', fontsize=10, framealpha=0.85)
            ax.set_title(f'Node-level accessibility — scenario S2\n{subtitle}',
                         fontsize=12, pad=8)
            plt.tight_layout()
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
            print(f'  saved   {fname}')

    # ── 4. Safety index map ───────────────────────────────────────────────────
    CRASH_RATE = {
        'Velobahn':                       0.104,
        'Veloschnellroute':               0.104,
        'Hauptverbindung':                0.409,
        'Nebenverbindung':                0.714,
        'Zusätzliche Freizeitverbindung': 0.409,
        'Netzlücke':                      1.020,
        'connector':                      1.020,
    }

    edges_safe = network_edges.copy()
    edges_safe['crash_rate'] = edges_safe['ROUTENTYP'].map(CRASH_RATE).fillna(1.020)

    if os.path.exists(cands_path):
        nl = gpd.read_file(cands_path)[['geometry']].copy()
        nl['crash_rate'] = 1.020
        edges_safe = gpd.GeoDataFrame(
            pd.concat([edges_safe[['crash_rate', 'geometry']], nl], ignore_index=True),
            geometry='geometry', crs='epsg:2056')

    vmin_s, vmax_s = 0.104, 1.020
    norm_s  = mcolors.Normalize(vmin=vmin_s, vmax=vmax_s)
    cmap_s  = plt.cm.RdYlGn_r

    fig, ax = plt.subplots(figsize=(14, 9))
    _add_base(ax)
    edges_safe.plot(ax=ax, column='crash_rate', cmap=cmap_s, norm=norm_s,
                    lw=1.8, zorder=5, legend=False)

    sm_s = mcm.ScalarMappable(cmap=cmap_s, norm=norm_s)
    sm_s.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.4)
    cbar = plt.colorbar(sm_s, cax=cax)
    cbar.set_label('Crash-cost rate [CHF/Pkm]\n(KNA Limmattal methodology)',
                   rotation=90, labelpad=14, fontsize=11)

    leg_handles = [
        mpatches.Patch(color=cmap_s(norm_s(r)), label=f'{rt}  ({r:.3f} CHF/Pkm)')
        for rt, r in [('Velobahn', 0.104), ('Hauptverbindung / Freizeit', 0.409),
                      ('Nebenverbindung', 0.714), ('Netzlücke (unbuilt)', 1.020)]
    ]
    ax.legend(handles=leg_handles, loc='upper left', fontsize=9, framealpha=0.85,
              title='ROUTENTYP', title_fontsize=10)
    ax.set_title('Per-link safety index — crash-cost rate (CHF/Pkm)\n'
                 'Red = high risk (Netzlücke / Nebenverbindung)  ·  Green = low risk (Velobahn)',
                 fontsize=12, pad=8)
    plt.tight_layout()
    plt.savefig('figures/safety_index_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  saved   figures/safety_index_map.png')

    # ── 5. Elevation map with hillshade + contours ────────────────────────────
    dem_path = 'data/elevation_model/elevation.tif'
    if os.path.exists(dem_path):
        with rasterio.open(dem_path) as src:
            dem     = src.read(1).astype(float)
            extent  = [src.bounds.left, src.bounds.right,
                       src.bounds.bottom, src.bounds.top]

        dem_min, dem_max = float(np.nanmin(dem)), float(np.nanmax(dem))
        norm_dem  = mcolors.Normalize(vmin=dem_min, vmax=dem_max)
        cmap_dem  = plt.cm.terrain

        ls = LightSource(azdeg=315, altdeg=45)
        hs = ls.hillshade(dem, vert_exag=2)

        ny, nx = dem.shape
        xs = np.linspace(extent[0], extent[1], nx)
        ys = np.linspace(extent[2], extent[3], ny)[::-1]

        fig, ax = plt.subplots(figsize=(14, 9))
        ax.imshow(cmap_dem(norm_dem(dem)), extent=extent, origin='upper',
                  zorder=1, alpha=0.75)
        ax.imshow(hs, extent=extent, origin='upper',
                  cmap='gray', alpha=0.35, zorder=2)

        contour_step = 20
        c_levels = np.arange(
            int(dem_min // contour_step) * contour_step,
            int(dem_max // contour_step) * contour_step + contour_step,
            contour_step)
        cs = ax.contour(xs, ys, dem, levels=c_levels,
                        colors='black', linewidths=0.3, alpha=0.4, zorder=3)
        ax.clabel(cs, inline=True, fontsize=6, fmt='%d m')

        network_edges.plot(ax=ax, color='#222222', lw=0.9, zorder=4, alpha=0.7)
        if os.path.exists(lakes_path):
            gpd.read_file(lakes_path).plot(ax=ax, color='lightblue', zorder=5, alpha=0.85)
        if os.path.exists(cities_path):
            cities = gpd.read_file(cities_path, crs='epsg:2056')
            cities.plot(ax=ax, color='black', markersize=50, zorder=7)
            for _, r in cities.iterrows():
                ax.annotate(r['location'], xy=r.geometry.coords[0],
                            ha='center', va='top', xytext=(0, -5),
                            textcoords='offset points', fontsize=10, zorder=7)

        sm_dem = mcm.ScalarMappable(cmap=cmap_dem, norm=norm_dem)
        sm_dem.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.4)
        cbar = plt.colorbar(sm_dem, cax=cax)
        cbar.set_label('Elevation [m a.s.l.]', rotation=90, labelpad=14, fontsize=11)

        ax.add_artist(ScaleBar(1, location='lower right'))
        ax.text(0.96, 0.93, 'N', fontsize=22, weight='bold',
                ha='center', va='center', transform=ax.transAxes, zorder=100)
        ax.add_patch(FancyArrowPatch(
            (0.96, 0.90), (0.96, 0.97), color='black', lw=1.5,
            arrowstyle='->', mutation_scale=20, transform=ax.transAxes, zorder=100))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor('black'); sp.set_linewidth(1)
        ax.set_title('Digital elevation model (2 m resolution) with 20 m contour lines\n'
                     'and cycling network overlay', fontsize=12, pad=8)
        ax.set_xlim(net_bounds[0] - x_pad, net_bounds[2] + x_pad)
        ax.set_ylim(net_bounds[1] - y_pad, net_bounds[3] + y_pad)
        plt.tight_layout()
        plt.savefig('figures/elevation_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('  saved   figures/elevation_map.png')

    # ── 6. Comfort index map ──────────────────────────────────────────────────
    comfort_net_path = 'data/costs/route_comfort_network.gpkg'
    if os.path.exists(comfort_net_path):
        EPSILON = {
            'Velobahn': 1.0, 'Veloschnellroute': 1.0,
            'Hauptverbindung': 1.3, 'Nebenverbindung': 1.6,
            'Zusätzliche Freizeitverbindung': 1.3,
            'Netzlücke': 2.0, 'connector': 2.0,
        }
        comfort_net = gpd.read_file(comfort_net_path)
        comfort_net['epsilon'] = comfort_net['ROUTENTYP'].map(EPSILON).fillna(2.0)
        # comfort index = (1 + alpha) * epsilon  where extra_factor = 1 + alpha
        comfort_net['comfort_index'] = comfort_net['extra_factor'] * comfort_net['epsilon']

        vmin_c = comfort_net['comfort_index'].min()
        vmax_c = comfort_net['comfort_index'].max()
        norm_c = mcolors.Normalize(vmin=vmin_c, vmax=vmax_c)
        cmap_c = plt.cm.RdYlGn_r

        fig, ax = plt.subplots(figsize=(14, 9))
        _add_base(ax)
        comfort_net.plot(ax=ax, column='comfort_index', cmap=cmap_c, norm=norm_c,
                         lw=2.0, zorder=5, legend=False)

        sm_c = mcm.ScalarMappable(cmap=cmap_c, norm=norm_c)
        sm_c.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.4)
        cbar = plt.colorbar(sm_c, cax=cax)
        cbar.set_label('Comfort index  (1 + α) × ε\nα = slope discomfort,  ε = ROUTENTYP multiplier',
                       rotation=90, labelpad=14, fontsize=10)

        ax.set_title('Per-link route comfort index — slope discomfort factor α × ROUTENTYP multiplier ε\n'
                     'Red = steep / low-quality link;  Green = flat Velobahn',
                     fontsize=12, pad=8)

        comfort_bounds = comfort_net.total_bounds
        ax.set_xlim(comfort_bounds[0] - x_pad, comfort_bounds[2] + x_pad)
        ax.set_ylim(comfort_bounds[1] - y_pad, comfort_bounds[3] + y_pad)
        plt.tight_layout()
        plt.savefig('figures/comfort_index_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('  saved   figures/comfort_index_map.png')

    print('\n[generate_report_figures] done — figures saved to figures/')

def _make_diverging_cmap(min_val, max_val):
    """
    Red→grey→blue diverging colormap anchored at zero.

    The proportions of red and blue are scaled to the magnitude of the
    negative and positive ranges respectively, so the grey midpoint always
    falls at NB=0 regardless of how asymmetric the range is.

    max(1, ...) guards against an empty colour array when one side of the
    range is extremely dominant (e.g. all-negative or near-zero positive).
    """
    n = 256
    gray = [0.83, 0.83, 0.83, 1.0]
    if min_val < 0 and max_val > 0:
        total = abs(min_val) + abs(max_val)
        neg_p, pos_p = abs(min_val) / total, abs(max_val) / total
        neg_c = plt.cm.Reds_r(np.linspace(0.15, 0.8,  max(1, int(n * neg_p))))
        pos_c = plt.cm.Blues( np.linspace(0.3,  0.95, max(1, int(n * pos_p))))
        tl = int(n * 0.2)
        all_c = np.vstack((neg_c[:-1],
                           np.linspace(neg_c[-1], gray, tl),
                           np.linspace(gray, pos_c[0], tl),
                           pos_c[1:]))
    elif min_val >= 0:
        pos_c = plt.cm.Blues(np.linspace(0.3, 0.9, n))
        all_c = np.vstack((np.linspace(gray, pos_c[0], int(n * 0.3)), pos_c[1:]))
    else:
        neg_c = plt.cm.Reds_r(np.linspace(0.2, 0.8, n))
        all_c = np.vstack((neg_c[:-1], np.linspace(neg_c[-1], gray, int(n * 0.3))))
    return LinearSegmentedColormap.from_list("div_cmap", all_c)


def _base_map(ax, network, access_points, boundary):
    """Draw lakes, network, access points, city labels, scale bar, north arrow."""
    lakes = gpd.read_file(r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp")
    lakes.plot(ax=ax, color="lightblue", zorder=9)
    ax.add_artist(ScaleBar(1, location="lower right"))

    if isinstance(network, gpd.GeoDataFrame):
        network.plot(ax=ax, color="#888888", lw=1.0, zorder=10, alpha=0.55)
    if isinstance(access_points, gpd.GeoDataFrame):
        access_points.plot(ax=ax, color="black", markersize=40, zorder=12)

    loc = gpd.read_file(r"data/manually_gathered_data/Cities.shp", crs="epsg:2056")
    loc.plot(ax=ax, color="black", markersize=60, zorder=13)
    for _, row in loc.iterrows():
        ax.annotate(row["location"], xy=row.geometry.coords[0],
                    ha="center", va="top", xytext=(0, -6),
                    textcoords="offset points", fontsize=13, zorder=13)

    ax.text(0.96, 0.92, "N", fontsize=26, weight="bold", ha="center",
            va="center", transform=ax.transAxes, zorder=1000)
    ax.add_patch(FancyArrowPatch((0.96, 0.89), (0.96, 0.97), color="black", lw=2,
                                 arrowstyle="->", mutation_scale=25,
                                 transform=ax.transAxes, zorder=1000))
    ax.set_xticks([]); ax.set_yticks([])
    if boundary is not None:
        xmin, ymin, xmax, ymax = boundary.bounds
        ax.set_xlim(xmin - 100, xmax + 100)
        ax.set_ylim(ymin - 100, ymax + 100)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor("black")
        sp.set_linewidth(1); sp.set_zorder(1000)







def plot_cost_result(df_costs, banned_area, title_bar, boundary=None, network=None,
                     access_points=None, plot_name=False, col="total_medium"):
    # Join NB values to actual LineString edge geometries (development candidates)
    dev_geom = gpd.read_file("data/Network/processed/development_candidates.gpkg")[["ID_new", "geometry"]]
    dev_geom["ID_new"] = dev_geom["ID_new"].astype(int)
    df_costs = df_costs.copy()
    df_costs["ID_new"] = df_costs["ID_new"].astype(int)
    df_plot = dev_geom.merge(df_costs.drop(columns=["geometry"], errors="ignore"),
                             on="ID_new", how="inner")
    df_plot = gpd.GeoDataFrame(df_plot, geometry="geometry", crs="EPSG:2056")
    df_plot = df_plot.dropna(subset=[col])

    min_val, max_val = df_plot[col].min(), df_plot[col].max()
    cmap = _make_diverging_cmap(min_val, max_val)
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

    fig, ax = plt.subplots(figsize=(15, 10))
    _base_map(ax, network, access_points, boundary)

    # Development lines coloured by net-benefit value
    df_plot.plot(ax=ax, column=col, cmap=cmap, norm=norm,
                 linewidth=6, zorder=11, legend=False, capstyle="round")

    # ID label at each line's midpoint
    for _, row in df_plot.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        mid = row.geometry.interpolate(0.5, normalized=True)
        ax.annotate(str(int(row["ID_new"])), xy=(mid.x, mid.y),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="black", zorder=16,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, ec="none"))

    # Colorbar
    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.5)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(f"{title_bar} [Mio. CHF]", rotation=90, labelpad=20, fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    raster = rasterio.open(banned_area)
    rasterio.plot.show(raster, ax=ax, cmap=ListedColormap(["white", "white"]), zorder=3)

    water_patch = mpatches.Patch(facecolor="lightblue", label="Water bodies",
                                 edgecolor="black", linewidth=1)
    ax.legend(handles=[water_patch], loc="upper center",
              bbox_to_anchor=(0.5, -0.02), fontsize=13, frameon=False)

    if plot_name:
        plt.tight_layout()
        plt.savefig(fr"plot/results/04_{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_single_cost_result(df_costs, banned_area, title_bar, boundary=None, network=None,
                            access_points=None, plot_name=False, col="total_medium"):
    # Join component values to actual LineString edge geometries
    dev_geom = gpd.read_file("data/Network/processed/development_candidates.gpkg")[["ID_new", "geometry"]]
    dev_geom["ID_new"] = dev_geom["ID_new"].astype(int)
    df_costs = df_costs.copy()
    df_costs["ID_new"] = df_costs["ID_new"].astype(int)
    df_costs[col] = df_costs[col] / 1e6   # scale to Mio. CHF
    df_plot = dev_geom.merge(df_costs.drop(columns=["geometry"], errors="ignore"),
                             on="ID_new", how="inner")
    df_plot = gpd.GeoDataFrame(df_plot, geometry="geometry", crs="EPSG:2056")
    df_plot = df_plot.dropna(subset=[col])

    min_val, max_val = df_plot[col].min(), df_plot[col].max()
    cmap = _make_diverging_cmap(min_val, max_val)
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

    fig, ax = plt.subplots(figsize=(15, 10))
    _base_map(ax, network, access_points, boundary)

    df_plot.plot(ax=ax, column=col, cmap=cmap, norm=norm,
                 linewidth=6, zorder=11, legend=False, capstyle="round")

    for _, row in df_plot.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        mid = row.geometry.interpolate(0.5, normalized=True)
        ax.annotate(str(int(row["ID_new"])), xy=(mid.x, mid.y),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="black", zorder=16,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, ec="none"))

    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.5)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(f"{title_bar} [Mio. CHF]", rotation=90, labelpad=20, fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    raster = rasterio.open(banned_area)
    rasterio.plot.show(raster, ax=ax, cmap=ListedColormap(["white", "white"]), zorder=3)

    water_patch = mpatches.Patch(facecolor="lightblue", label="Water bodies",
                                 edgecolor="black", linewidth=1)
    ax.legend(handles=[water_patch], loc="upper center",
              bbox_to_anchor=(0.5, -0.02), fontsize=13, frameon=False)

    if plot_name:
        plt.tight_layout()
        plt.savefig(fr"plot/results/04_{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()

def plot_cost_uncertainty(df_costs, banned_area, col, legend_title, boundary=None, network=None, access_points=None, plot_name=False):

    # Determine the range of your data
    min_val = df_costs["mean_costs"].min()
    max_val = df_costs["mean_costs"].max()

    # Number of color intervals
    n_intervals = 256
    # Define a gray color for the zero point
    gray_color = [0.83, 0.83, 0.83, 1]  # RGBA for gray

    if (min_val < 0) & (max_val > 0):
        total_range = abs(min_val) + abs(max_val)
        neg_proportion = abs(min_val) / total_range
        pos_proportion = abs(max_val) / total_range

        # Generate colors for negative (red) and positive (blue) ranges
        neg_colors = plt.cm.Reds_r(np.linspace(0.15, 0.8, int(n_intervals * neg_proportion)))
        pos_colors = plt.cm.Blues(np.linspace(0.3, 0.95, int(n_intervals * pos_proportion)))

        # Create a transition array from reds to gray and from gray to blues
        transition_length = int(n_intervals * 0.2)  # Length of the transition zone
        reds_to_gray = np.linspace(neg_colors[-1], gray_color, transition_length)
        gray_to_blues = np.linspace(gray_color, pos_colors[0], transition_length)

        # Create an array that combines the colors with a smooth transition
        all_colors = np.vstack((neg_colors[:-1], reds_to_gray, gray_to_blues, pos_colors[1:]))

    elif min_val >= 0:
        # Case with only positive values
        pos_colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_intervals))
        gray_to_blues = np.linspace(gray_color, pos_colors[0], int(n_intervals * 0.3))
        all_colors = np.vstack((gray_to_blues, pos_colors[1:]))

    elif max_val <= 0:
        # Case with only negative values
        neg_colors = plt.cm.Reds_r(np.linspace(0.2, 0.8, n_intervals))
        reds_to_gray = np.linspace(neg_colors[-1], gray_color, int(n_intervals * 0.3))
        all_colors = np.vstack((neg_colors[:-1], reds_to_gray))

    # Create the new colormap
    cmap = LinearSegmentedColormap.from_list("custom_colormap", all_colors)
    fig, ax = plt.subplots(figsize=(20, 10))
    # Plot lakes
    lakes = gpd.read_file(r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp")
    lakes.plot(ax=ax, color="lightblue", zorder=9)

    # Add scale bar
    ax.add_artist(ScaleBar(1, location="lower right"))

    if isinstance(network, gpd.GeoDataFrame):
        network.plot(ax=ax, color="black", lw=2, zorder=11)

    if isinstance(access_points, gpd.GeoDataFrame):
        access_points.plot(ax=ax, color="black", markersize=50, zorder=12)

    location = gpd.read_file(r'data/manually_gathered_data/Cities.shp', crs="epsg:2056")
    # Plot the location as points
    location.plot(ax=ax, color="black", markersize=75, zorder=13)
    # Add city names to the plot
    for idx, row in location.iterrows():
        ax.annotate(row['location'], xy=row["geometry"].coords[0], ha="center", va="top", xytext=(0, -6),
                         textcoords='offset points', fontsize=15, zorder=13)
    """
    # Comopute markersize based on cv value but they should range within 2 - 50
    # Assuming 'df' is your DataFrame and 'value_column' is the column you want to normalize
    min_val, max_val = df_costs['std'].min(), df_costs['std'].max()
    scale_min, scale_max = 10, 400
    # Normalize the column
    df_costs['markersize'] = scale_max - (((df_costs['std'] - min_val) / (max_val - min_val)) * (scale_max - scale_min))
    # Plot points
    """
    scale_min, scale_max = 30, 500
    # Apply a non-linear transformation (log) — clip to avoid log(0) = -inf
    df_costs[f'log_{col}'] = np.log(df_costs[col].clip(lower=1e-6))
    # Normalize the transformed column; guard against all-identical values
    min_val = df_costs[f'log_{col}'].min()
    max_val = df_costs[f'log_{col}'].max()
    log_range = max_val - min_val if max_val > min_val else 1.0
    df_costs['markersize'] = scale_max - (((df_costs[f'log_{col}'] - min_val) / log_range) * (scale_max - scale_min))

    df_costs_sorted = df_costs.sort_values(by='mean_costs')
    df_costs_sorted.plot(ax=ax, column="mean_costs", markersize="markersize", cmap=cmap, zorder=4, edgecolor='black', linewidth=1)

    # Get the position of the current plot
    pos = ax.get_position()

    # Create a new axes for the colorbar on the right of the plot
    y_start = 0.25
    cbar_ax = fig.add_axes([pos.x1 + 0.1, pos.y0 + y_start, 0.01, pos.y1 - y_start - 0.005])

    # Add the colorbar
    cbar_gdf = fig.colorbar(ax.collections[4], cax=cbar_ax)

    cbar_gdf.set_label(
        f'Mean Net benefits [Mio. CHF]\n(Construction, maintenance, highway travel\ntime, access time and external effects)',
        rotation=90, labelpad=30, fontsize=16)
    cbar_gdf.ax.tick_params(labelsize=14)


    raster = rasterio.open(banned_area)
    gray_brigth = (0.88, 0.88, 0.88)
    cmap_raster = ListedColormap([gray_brigth, gray_brigth])
    rasterio.plot.show(raster, ax=ax, cmap=cmap_raster, zorder=3)

    # Create custom legend elements
    water_body_patch = mpatches.Patch(facecolor="lightblue", label='Water bodies', edgecolor='black', linewidth=1)
    protected_area_patch = mpatches.Patch(facecolor='lightgray', label='Protected area',
                                          edgecolor='black', linewidth=1)

    # Create the legend below the plot
    # legend = ax.legend(handles=[water_body_patch, protected_area_patch], loc='lower center',bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=16, frameon=False)
    """
    # Create actual scatter points on the plot for the legend
    # Choose a range of std values for the legend
    original_std_values = np.linspace(min_val, max_val, 6)
    # Calculate corresponding marker sizes for these std values
    legend_sizes = scale_max - ((original_std_values - min_val) / (max_val - min_val)) * (scale_max - scale_min)
    # Create scatter plot handles for the legend
    legend_handles = [mlines.Line2D([], [], color='white', marker='o', linestyle='solid', linewidth=1, markerfacecolor='white', markeredgecolor='black',
                                    markersize=np.sqrt(size), label=f'{std_val:.1f}')
                      for size, std_val in zip(legend_sizes, original_std_values)]
    """
    # Choose a range of std values for the legend
    original_std_values = np.linspace(df_costs[col].min(), df_costs[col].max(), 6)  # Use original std values

    # Calculate corresponding marker sizes for these std values (reversed mapping)
    legend_sizes = scale_max - (
                ((np.log(np.maximum(original_std_values, 1e-6)) - min_val) / log_range) * (scale_max - scale_min))

    # Create scatter plot handles for the legend with labels as original std values
    legend_handles = [
        mlines.Line2D([], [], color='white', marker='o', linestyle='solid', linewidth=1, markerfacecolor='white',
                      markeredgecolor='black',
                      markersize=np.sqrt(size), label=f'{std_val:.0f}')
        for size, std_val in zip(legend_sizes, original_std_values)]

    # Create patch elements for the legend
    water_body_patch = mpatches.Patch(facecolor="lightblue", label='Water bodies', edgecolor='black', linewidth=1)
    protected_area_patch = mpatches.Patch(facecolor=gray_brigth, label='Protected area', edgecolor='black', linewidth=1)

    # Combine scatter handles and patch elements
    combined_legend_elements = legend_handles + [water_body_patch, protected_area_patch]

    # Create a single combined legend below the plot
    combined_legend = ax.legend(handles=combined_legend_elements, loc='lower left', bbox_to_anchor=(1.015, 0),
                                fontsize=14, frameon=False, title=f'{legend_title}\n',title_fontsize=16)

    # Add the combined legend to the plot
    ax.add_artist(combined_legend)

    # Add a north arrow
    # Add the letter "N"
    ax.text(0.96, 0.92, "N", fontsize=28, weight=1, ha='center', va='center', transform=ax.transAxes, zorder=1000)

    # Add a custom north arrow
    arrow = FancyArrowPatch((0.96, 0.89), (0.96, 0.97), color='black', lw=2, arrowstyle='->', mutation_scale=25, transform=ax.transAxes, zorder=1000)
    ax.add_patch(arrow)

    ax.set_xticks([])
    ax.set_yticks([])

    # Get plot limits
    min_x, min_y, max_x, max_y = boundary.bounds
    ax.set_xlim(min_x - 100, max_x + 100)
    ax.set_ylim(min_y - 100, max_y + 100)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1)  # Adjust linewidth for frame thickness
        spine.set_zorder(1000)

    if plot_name != False:
        plt.savefig(fr"plot/results/04_{plot_name}.png", dpi=300, bbox_inches='tight')

    plt.show()
    return








def plot_scenario_grouped_bar(plot_name="scenario_grouped_bar"):
    """
    Grouped horizontal bar chart: for each development (sorted by NB_s2) three
    bars are shown side by side — one per growth scenario (s1 / s2 / s3).
    Makes the scenario spread directly readable per development.

    Reads:  data/costs/net_benefits.csv
            data/Network/processed/development_candidates.gpkg
    Saves:  plot/results/{plot_name}.png
    """
    nb_path   = "data/costs/net_benefits.csv"
    devs_path = "data/Network/processed/development_candidates.gpkg"
    for p in [nb_path, devs_path]:
        if not os.path.exists(p):
            print(f"[plot_scenario_grouped_bar] Missing: {p} — skipping")
            return

    nb   = pd.read_csv(nb_path)
    devs = gpd.read_file(devs_path)[["ID_new", "dev_type"]]
    nb   = nb.merge(devs, on="ID_new", how="left")
    nb   = nb.sort_values("NB_s2", ascending=True).reset_index(drop=True)

    for c in ["NB_s1", "NB_s2", "NB_s3"]:
        nb[c] = nb[c] / 1e6

    n     = len(nb)
    y     = np.arange(n)
    bar_h = 0.25

    scenarios = [
        ("NB_s1", "#74b9ff", "Low growth (s1)"),
        ("NB_s2", "#0984e3", "Medium growth (s2)"),
        ("NB_s3", "#2d3436", "High growth (s3)"),
    ]

    fig, ax = plt.subplots(figsize=(11, max(6, n * 0.55)))

    for i, (col, color, label) in enumerate(scenarios):
        offset = (i - 1) * bar_h   # –1 / 0 / +1
        ax.barh(y + offset, nb[col], height=bar_h,
                color=color, alpha=0.85, label=label, zorder=3)

    ax.axvline(0, color="black", lw=1.0, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(nb["ID_new"].astype(int).astype(str), fontsize=9)
    ax.set_xlabel("Net Benefit [Mio. CHF]", fontsize=11)
    ax.set_title("Net Benefit per Development — all growth scenarios\n(sorted by medium scenario)",
                 fontsize=12, pad=8)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)

    plt.tight_layout()
    os.makedirs("plot/results", exist_ok=True)
    plt.savefig(f"plot/results/{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[plot_scenario_grouped_bar] saved → plot/results/{plot_name}.png")


def plot_cost_benefit_scatter(plot_name="cost_benefit_scatter"):
    """
    Two-panel scatter: X = total costs |C+M|, Y = total benefits T+R+S (S2).
    Left panel: full scale (all 21 developments including outliers).
    Right panel: zoomed to 0–32 Mio. CHF on both axes; out-of-range points
    (IDs 676, 79, 385) are shown as arrows at the top edge with their labels.

    Reads:  data/costs/net_benefits.gpkg
    Saves:  plot/results/{plot_name}.png
    """
    nb_path = "data/costs/net_benefits.gpkg"
    if not os.path.exists(nb_path):
        print(f"[plot_cost_benefit_scatter] Missing: {nb_path} — skipping")
        return

    nb = gpd.read_file(nb_path)
    nb["costs_m"]    = (nb["C"].abs() + nb["M"].abs()) / 1e6
    nb["benefits_m"] = (nb["T_s2"] + nb["R_s2"] + nb["S_s2"]) / 1e6
    nb["NB_m"]       = nb["NB_s2"] / 1e6
    nb["color"]      = nb["NB_m"].apply(lambda v: "#27ae60" if v >= 0 else "#e74c3c")

    ZOOM = 32.0   # upper limit of zoom panel (Mio. CHF)

    legend_handles = [
        mpatches.Patch(color="#27ae60", label="Positive NB  (benefits > costs)"),
        mpatches.Patch(color="#e74c3c", label="Negative NB  (costs > benefits)"),
        plt.Line2D([0], [0], color="black", lw=1.2, linestyle="--",
                   alpha=0.6, label="Break-even  (NB = 0)"),
    ]

    def _draw_panel(ax, xlim, ylim, title, show_legend=False):
        """Draw the scatter on ax with given axis limits."""
        ax.fill_between([0, xlim], [0, xlim], ylim,
                        color="#27ae60", alpha=0.05, zorder=0)
        ax.fill_between([0, xlim], 0, [0, min(xlim, ylim)],
                        color="#e74c3c", alpha=0.05, zorder=0)
        ax.plot([0, min(xlim, ylim)], [0, min(xlim, ylim)],
                color="black", lw=1.2, linestyle="--", alpha=0.45, zorder=1)

        # Points inside range
        inside = nb[(nb["costs_m"] <= xlim) & (nb["benefits_m"] <= ylim)]
        ax.scatter(inside["costs_m"], inside["benefits_m"],
                   c=inside["color"], s=90, alpha=0.88,
                   zorder=3, edgecolors="white", linewidths=0.6)
        for _, row in inside.iterrows():
            ax.annotate(str(int(row["ID_new"])),
                        (row["costs_m"], row["benefits_m"]),
                        fontsize=8.5, ha="left", va="bottom",
                        xytext=(4, 3), textcoords="offset points", zorder=4)

        # Out-of-range points: draw arrow at top of panel + label
        outside = nb[nb["benefits_m"] > ylim]
        for _, row in outside.iterrows():
            x = min(row["costs_m"], xlim * 0.97)
            ax.annotate(
                f"ID {int(row['ID_new'])}\n({row['benefits_m']:.0f})",
                xy=(x, ylim), xytext=(x, ylim * 0.88),
                fontsize=7.5, ha="center", va="top", color=row["color"],
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=row["color"],
                                lw=1.2, shrinkA=0, shrinkB=2),
                zorder=5,
            )

        ax.set_xlim(0, xlim)
        ax.set_ylim(0, ylim)
        ax.set_xlabel("Total Costs  |C + M|  [Mio. CHF]", fontsize=10)
        ax.set_ylabel("Total Benefits  T + R + S  [Mio. CHF]", fontsize=10)
        ax.set_title(title, fontsize=10, pad=6)
        ax.grid(linestyle="--", alpha=0.3, zorder=0)
        if show_legend:
            ax.legend(handles=legend_handles, fontsize=8.5, loc="upper left")

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(15, 6.5))

    max_x = nb["costs_m"].max() * 1.08
    max_y = nb["benefits_m"].max() * 1.06
    _draw_panel(ax_full, xlim=max_x, ylim=max_y,
                title="Full scale  (all 21 developments)",
                show_legend=True)

    _draw_panel(ax_zoom, xlim=ZOOM, ylim=ZOOM,
                title=f"Zoom: 0 – {ZOOM:.0f} Mio. CHF  (out-of-range shown as arrows)")

    fig.suptitle("Cost–Benefit Balance per Development  (medium growth scenario, S2)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    os.makedirs("plot/results", exist_ok=True)
    plt.savefig(f"plot/results/{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[plot_cost_benefit_scatter] saved → plot/results/{plot_name}.png")





def boxplot(df, nbr):
    df["mean"] = df[['NB_s1', 'NB_s2', 'NB_s3']].mean(axis=1)
    df = df.sort_values(by=['mean'], ascending=False)
    df_top = df.head(nbr)
    df_top = df_top[['ID_new', 'NB_s1', 'NB_s2', 'NB_s3']]
    df_top = df_top.set_index('ID_new').T

    # Plotting the boxplot
    plt.figure(figsize=(20, 8))
    df_top.boxplot()

    # Color area 0f y<0 with light grey
    # Get min y value
    ymin, ymax = plt.ylim()
    plt.axhspan(ymin, 0, color='lightgrey', alpha=0.5)
    # Set y limit
    plt.ylim(ymin, ymax)

    plt.xlabel("Development ID", fontsize=22)
    plt.ylabel("Net benefits over all scenarios \n [Mio. CHF]", fontsize=22)
    # Increse fontsize for all ticks
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(r"plot/results/04_boxplot.png", dpi=500)
    plt.show()




# ─────────────────────────────────────────────────────────────────────────────
# NEW RESULT VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_priority_ranking(plot_name="priority_ranking"):
    """
    Horizontal bar chart: one bar per development, sorted by NB_s2 (medium
    scenario), with error bars spanning NB_s1 → NB_s3 as scenario uncertainty.
    Bars are coloured by dev_type (red = Netzlücke, orange = Schwachstelle).
    A vertical line at 0 separates net-positive from net-negative candidates.

    Reads:  data/costs/net_benefits.csv
            data/Network/processed/developments_list.csv
    Saves:  plot/results/{plot_name}.png
    """
    nb_path   = "data/costs/net_benefits.csv"
    devs_path = "data/Network/processed/development_candidates.gpkg"
    for p in [nb_path, devs_path]:
        if not os.path.exists(p):
            print(f"[plot_priority_ranking] Missing: {p} — skipping")
            return

    nb   = pd.read_csv(nb_path)
    devs = gpd.read_file(devs_path)[["ID_new", "dev_type"]]
    nb   = nb.merge(devs, on="ID_new", how="left")
    nb   = nb.sort_values("NB_s2").reset_index(drop=True)

    # Convert to Mio. CHF
    for c in ["NB_s1", "NB_s2", "NB_s3"]:
        nb[c] = nb[c] / 1e6

    err_lo = (nb["NB_s2"] - nb["NB_s1"]).clip(lower=0)
    err_hi = (nb["NB_s3"] - nb["NB_s2"]).clip(lower=0)

    colors = nb["dev_type"].map(
        {"netzluecke": "#c0392b", "schwachstelle": "#e67e22", "connectivity": "#7f8c8d"}
    ).fillna("#95a5a6")

    n = len(nb)
    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.38)))
    y = np.arange(n)

    ax.barh(y, nb["NB_s2"], xerr=[err_lo, err_hi],
            color=colors, capsize=3, alpha=0.85, error_kw={"lw": 1.2, "ecolor": "#333333"})
    ax.axvline(0, color="black", lw=1.0, zorder=5)

    labels = nb["ID_new"].astype(int).astype(str)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Net Benefit — medium scenario [Mio. CHF]", fontsize=11)
    ax.set_title("Development Ranking by Net Benefit\n(NB_s2 ± scenario range s1–s3)",
                 fontsize=12, pad=8)
    ax.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)

    present_types = nb["dev_type"].dropna().unique().tolist()
    type_meta = {
        "netzluecke":    ("#c0392b", "Netzlücke"),
        "schwachstelle": ("#e67e22", "Schwachstelle"),
        "connectivity":  ("#7f8c8d", "Connectivity bridge"),
    }
    legend_handles = [
        mpatches.Patch(color=col, label=lbl)
        for key, (col, lbl) in type_meta.items()
        if key in present_types
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", fontsize=9)

    plt.tight_layout()
    os.makedirs("plot/results", exist_ok=True)
    plt.savefig(f"plot/results/{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[plot_priority_ranking] saved → plot/results/{plot_name}.png")


def plot_nb_components_waterfall(plot_name="nb_components_waterfall"):
    """
    Stacked horizontal bar chart: for each development (sorted by NB_s2)
    the negative side shows Construction (C) + Maintenance (M) and the
    positive side shows Travel-time savings (T_s2) + Comfort (R) + Safety (S_s2).
    Makes it immediately clear what component drives each result.

    Reads:  data/costs/net_benefits.gpkg
            data/Network/processed/developments_list.csv
    Saves:  plot/results/{plot_name}.png
    """
    nb_path   = "data/costs/net_benefits.gpkg"
    devs_path = "data/Network/processed/developments_list.csv"
    for p in [nb_path, devs_path]:
        if not os.path.exists(p):
            print(f"[plot_nb_components_waterfall] Missing: {p} — skipping")
            return

    nb   = gpd.read_file(nb_path)
    devs = pd.read_csv(devs_path)
    nb   = nb.merge(devs[["ID_new", "dev_type"]], on="ID_new", how="left")
    nb   = nb.sort_values("NB_s2").reset_index(drop=True)

    M   = 1e6
    ids = nb["ID_new"].astype(int).astype(str)
    n   = len(nb)
    y   = np.arange(n)

    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.38)))

    # ── Costs (extend left, stacked) ─────────────────────────────────────────
    ax.barh(y, nb["C"] / M,  color="#c0392b", alpha=0.85, label="Construction (C)")
    ax.barh(y, nb["M"] / M,  left=nb["C"] / M,
            color="#e74c3c", alpha=0.85, label="Maintenance (M)")

    # ── Benefits (extend right, stacked) ─────────────────────────────────────
    ax.barh(y, nb["T_s2"] / M, color="#27ae60", alpha=0.85, label="Travel-time savings (T)")
    ax.barh(y, nb["R_s2"] / M,
            left=nb["T_s2"] / M,
            color="#2ecc71", alpha=0.85, label="Route comfort (R)")
    ax.barh(y, nb["S_s2"] / M,
            left=(nb["T_s2"] + nb["R_s2"]) / M,
            color="#3498db", alpha=0.85, label="Safety (S)")

    ax.axvline(0, color="black", lw=1.0, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(ids, fontsize=9)
    ax.set_xlabel("CHF [Mio.]", fontsize=11)
    ax.set_title("NB Components per Development (medium scenario)",
                 fontsize=12, pad=8)
    ax.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)
    ax.legend(loc="lower right", fontsize=9, frameon=True)

    plt.tight_layout()
    os.makedirs("plot/results", exist_ok=True)
    plt.savefig(f"plot/results/{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[plot_nb_components_waterfall] saved → plot/results/{plot_name}.png")


def plot_tt_improvement_map(boundary=None, network=None, top_n=3,
                             plot_name="tt_improvement_map"):
    """
    Spatial raster-diff map: for the top_n developments by NB_s2, shows the
    per-pixel travel-time improvement (status_quo − dev) in minutes.
    Green = shorter travel time with the new link.  The corridor network is
    overlaid as a grey backdrop.

    Reads:  data/costs/net_benefits.csv
            data/Network/travel_time/travel_time_raster.tif
            data/Network/travel_time/developments/dev{id}_travel_time_raster.tif
            data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp
    Saves:  plot/results/{plot_name}.png
    """
    import rasterio as _rio

    nb_path = "data/costs/net_benefits.csv"
    sq_path = "data/Network/travel_time/travel_time_raster.tif"
    for p in [nb_path, sq_path]:
        if not os.path.exists(p):
            print(f"[plot_tt_improvement_map] Missing: {p} — skipping")
            return

    nb     = pd.read_csv(nb_path).sort_values("NB_s2", ascending=False)
    top_ids = nb["ID_new"].astype(int).head(top_n).tolist()

    # Filter to developments that actually have a raster file
    available = [
        i for i in top_ids
        if os.path.exists(
            f"data/Network/travel_time/developments/dev{i}_travel_time_raster.tif"
        )
    ]
    if not available:
        print("[plot_tt_improvement_map] No development rasters found — skipping")
        return

    with _rio.open(sq_path) as src:
        sq_arr = src.read(1).astype(float)  # seconds
        ext    = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]

    lakes_path = r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp"

    n_plots = len(available)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 8),
                              squeeze=False)

    for ax, dev_id in zip(axes[0], available):
        dev_path = (
            f"data/Network/travel_time/developments/dev{dev_id}_travel_time_raster.tif"
        )
        with _rio.open(dev_path) as src2:
            dev_arr = src2.read(1).astype(float)

        # Improvement in minutes (positive = better with new link)
        diff_min = np.where(
            np.isnan(sq_arr) | np.isnan(dev_arr),
            np.nan,
            (sq_arr - dev_arr) / 60.0
        )

        vmax = max(0.5, float(np.nanpercentile(np.abs(diff_min), 95)))
        im   = ax.imshow(diff_min, cmap="RdYlGn",
                         vmin=-vmax, vmax=0,
                         extent=ext, origin="upper", alpha=0.85, zorder=2)

        if os.path.exists(lakes_path):
            gpd.read_file(lakes_path).plot(
                ax=ax, color="lightblue", zorder=3, alpha=0.7)

        if isinstance(network, gpd.GeoDataFrame):
            network.plot(ax=ax, color="#555555", lw=0.6, alpha=0.5, zorder=4)

        nb_val = nb.loc[nb["ID_new"] == dev_id, "NB_s2"].values
        nb_str = f"NB = {nb_val[0]/1e6:+.1f} Mio. CHF" if len(nb_val) else ""
        ax.set_title(f"Dev {dev_id}\n{nb_str}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        cbar = plt.colorbar(im, ax=ax, shrink=0.55, pad=0.02)
        cbar.set_label("ΔTT [min]", fontsize=9)

        if boundary:
            xmin, ymin, xmax, ymax = boundary.bounds
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

    fig.suptitle(
        f"Travel-Time Improvement vs. Status Quo — Top {len(available)} Developments",
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    os.makedirs("plot/results", exist_ok=True)
    plt.savefig(f"plot/results/{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[plot_tt_improvement_map] saved → plot/results/{plot_name}.png")


def plot_nb_on_network(boundary=None, network=None, access_points=None,
                       plot_name="nb_network_map"):
    """
    Map with development candidate **edge geometries** coloured by NB_s2
    (medium scenario net benefit).  Line width encodes construction cost
    so cheap high-NB links stand out.  The status-quo network is shown
    as a grey backdrop.  Uses the same custom red–grey–blue colormap as
    plot_cost_result() so colours are consistent across all result maps.

    Unlike plot_cost_result() (which uses centroid points + interpolation),
    this function draws the actual LineString geometry of each candidate —
    more accurate for infrastructure that spans several hundred metres.

    Reads:  data/Network/processed/development_candidates.gpkg
            data/costs/net_benefits.gpkg
            data/Network/processed/developments_list.csv
            data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp
    Saves:  plot/results/{plot_name}.png
    """
    cands_path = "data/Network/processed/development_candidates.gpkg"
    nb_path    = "data/costs/net_benefits.gpkg"
    devs_path  = "data/Network/processed/developments_list.csv"
    lakes_path = r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp"

    for p in [cands_path, nb_path]:
        if not os.path.exists(p):
            print(f"[plot_nb_on_network] Missing: {p} — skipping")
            return

    cands = gpd.read_file(cands_path)
    nb    = gpd.read_file(nb_path)[["ID_new", "NB_s2", "C"]]
    nb["NB_s2"] = pd.to_numeric(nb["NB_s2"], errors="coerce")
    nb["C"]     = pd.to_numeric(nb["C"],     errors="coerce")

    # Attach NB scores to the actual edge geometry (not centroids)
    merged = cands.merge(nb, on="ID_new", how="inner")
    if merged.empty:
        print("[plot_nb_on_network] No matching rows after merge — skipping")
        return

    # Attach description for labels if available
    if os.path.exists(devs_path):
        devs   = pd.read_csv(devs_path)[["ID_new", "dev_type"]]
        merged = merged.merge(devs, on="ID_new", how="left")

    nb_vals = merged["NB_s2"] / 1e6
    vabs    = float(nb_vals.abs().quantile(0.95)) or 1.0
    min_val, max_val = nb_vals.min(), nb_vals.max()

    # Build same Red–Grey–Blue colormap used throughout plots.py
    n_intervals = 256
    gray_color  = [0.83, 0.83, 0.83, 1]
    if min_val < 0 and max_val > 0:
        total_range   = abs(min_val) + abs(max_val)
        neg_colors    = plt.cm.Reds_r(
            np.linspace(0.15, 0.8, int(n_intervals * abs(min_val) / total_range))
        )
        pos_colors    = plt.cm.Blues(
            np.linspace(0.3, 0.95, int(n_intervals * abs(max_val) / total_range))
        )
        tr = int(n_intervals * 0.2)
        all_colors = np.vstack((
            neg_colors[:-1],
            np.linspace(neg_colors[-1], gray_color, tr),
            np.linspace(gray_color, pos_colors[0], tr),
            pos_colors[1:],
        ))
    elif min_val >= 0:
        pos_colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_intervals))
        all_colors = np.vstack((
            np.linspace(gray_color, pos_colors[0], int(n_intervals * 0.3)),
            pos_colors[1:],
        ))
    else:
        neg_colors = plt.cm.Reds_r(np.linspace(0.2, 0.8, n_intervals))
        all_colors = np.vstack((
            neg_colors[:-1],
            np.linspace(neg_colors[-1], gray_color, int(n_intervals * 0.3)),
        ))

    cmap = LinearSegmentedColormap.from_list("nb_edge_cmap", all_colors)
    norm = plt.Normalize(vmin=-vabs, vmax=vabs)

    fig, ax = plt.subplots(figsize=(15, 10))

    # ── Base layers ───────────────────────────────────────────────────────────
    if os.path.exists(lakes_path):
        gpd.read_file(lakes_path).plot(ax=ax, color="lightblue", zorder=1)

    if isinstance(network, gpd.GeoDataFrame):
        network.plot(ax=ax, color="#bbbbbb", lw=1.0, zorder=2, label="Status-quo network")

    # ── Candidate edges coloured by NB_s2 ────────────────────────────────────
    c_min = merged["C"].abs().min() or 1.0
    c_max = merged["C"].abs().max() or 1.0

    for _, row in merged.iterrows():
        nb_m  = row["NB_s2"] / 1e6
        color = cmap(norm(nb_m))
        # Line width 1.5–5.5 proportional to construction cost
        c_abs = abs(row["C"]) if pd.notna(row["C"]) else c_min
        lw    = 1.5 + 4.0 * (c_abs - c_min) / max(c_max - c_min, 1)
        gpd.GeoDataFrame([row], crs=cands.crs).plot(
            ax=ax, color=[color], lw=lw, zorder=5)
        # ID label at midpoint
        try:
            mid = row.geometry.interpolate(0.5, normalized=True)
            ax.annotate(
                str(int(row["ID_new"])),
                xy=(mid.x, mid.y), xytext=(4, 4),
                textcoords="offset points",
                fontsize=8, fontweight="bold", color="black", zorder=10
            )
        except Exception:
            pass

    if isinstance(access_points, gpd.GeoDataFrame):
        access_points.plot(ax=ax, color="black", markersize=25, zorder=6)

    # ── City labels ───────────────────────────────────────────────────────────
    cities_path = r"data/manually_gathered_data/Cities.shp"
    if os.path.exists(cities_path):
        cities = gpd.read_file(cities_path, crs="epsg:2056")
        cities.plot(ax=ax, color="black", markersize=60, zorder=7)
        for _, loc_row in cities.iterrows():
            ax.annotate(
                loc_row["location"],
                xy=loc_row["geometry"].coords[0],
                ha="center", va="top", xytext=(0, -6),
                textcoords="offset points", fontsize=13, zorder=7
            )

    # ── Colorbar ──────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="2%", pad=0.5)
    cbar    = plt.colorbar(sm, cax=cax)
    cbar.set_label("Net Benefit NB_s2 [Mio. CHF]\n(line width ∝ construction cost)",
                   rotation=90, labelpad=16, fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    # ── North arrow ───────────────────────────────────────────────────────────
    ax.text(0.96, 0.92, "N", fontsize=28, weight="bold",
            ha="center", va="center", transform=ax.transAxes, zorder=1000)
    ax.add_patch(FancyArrowPatch(
        (0.96, 0.89), (0.96, 0.97), color="black", lw=2,
        arrowstyle="->", mutation_scale=25,
        transform=ax.transAxes, zorder=1000
    ))
    ax.add_artist(ScaleBar(1, location="lower right"))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "Development Candidates — Net Benefit on Network\n"
        "(colour = NB_s2, line width = construction cost)",
        fontsize=13, pad=8
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(1)
        spine.set_zorder(1000)

    if boundary:
        xmin, ymin, xmax, ymax = boundary.bounds
        ax.set_xlim(xmin - 100, xmax + 100)
        ax.set_ylim(ymin - 100, ymax + 100)

    plt.tight_layout()
    os.makedirs("plot/results", exist_ok=True)
    plt.savefig(f"plot/results/{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[plot_nb_on_network] saved → plot/results/{plot_name}.png")


def plot_duebendorf_zoom(df_costs, banned_area, title_bar, network=None,
                         access_points=None, plot_name="duebendorf_zoom",
                         col="NB_s2", scale_to_mio=False):
    """
    Stand-alone zoomed map of the Dübendorf sub-area
    (E 2 687 000–2 697 500 / N 1 247 000–1 254 000, LV95/EPSG:2056).
    Same colour scheme as plot_cost_result so the two maps can be read
    side by side.  Development IDs are labelled at each line's midpoint.

    scale_to_mio : if True, divide col by 1e6 before plotting (use when
                   passing raw CHF values from net_benefits.gpkg, e.g. for
                   component maps C, M, T_s2, R_s2, S_s2).

    Reads:  data/Network/processed/development_candidates.gpkg
            data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp
    Saves:  plot/results/{plot_name}.png
    """
    DUB_E_MIN, DUB_E_MAX = 2_687_000, 2_697_500
    DUB_N_MIN, DUB_N_MAX = 1_247_000, 1_254_000

    # ── Join NB values to LineString edge geometries ──────────────────────────
    dev_geom = gpd.read_file(
        "data/Network/processed/development_candidates.gpkg")[["ID_new", "geometry"]]
    dev_geom["ID_new"] = dev_geom["ID_new"].astype(int)
    df_costs = df_costs.copy()
    df_costs["ID_new"] = df_costs["ID_new"].astype(int)
    if scale_to_mio:
        df_costs[col] = df_costs[col] / 1e6
    df_plot = dev_geom.merge(
        df_costs.drop(columns=["geometry"], errors="ignore"),
        on="ID_new", how="inner")
    df_plot = gpd.GeoDataFrame(df_plot, geometry="geometry", crs="EPSG:2056")
    df_plot = df_plot.dropna(subset=[col])

    # ── Clip to Dübendorf extent ──────────────────────────────────────────────
    df_dub = df_plot.cx[DUB_E_MIN:DUB_E_MAX, DUB_N_MIN:DUB_N_MAX].copy()
    if df_dub.empty:
        print(f"[plot_duebendorf_zoom] No developments in Dübendorf extent — skipping")
        return

    # Same diverging colormap as the full corridor maps
    min_val, max_val = df_plot[col].min(), df_plot[col].max()
    cmap = _make_diverging_cmap(min_val, max_val)
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

    fig, ax = plt.subplots(figsize=(10, 8))

    # ── Base layers ───────────────────────────────────────────────────────────
    lakes_path = r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp"
    if os.path.exists(lakes_path):
        gpd.read_file(lakes_path).plot(ax=ax, color="lightblue", zorder=9)

    if isinstance(network, gpd.GeoDataFrame):
        network.plot(ax=ax, color="#888888", lw=1.0, zorder=10, alpha=0.55)

    if isinstance(access_points, gpd.GeoDataFrame):
        access_points.plot(ax=ax, color="black", markersize=40, zorder=12)

    # ── Coloured development lines ────────────────────────────────────────────
    df_dub.plot(ax=ax, column=col, cmap=cmap, norm=norm,
                linewidth=7, zorder=11, legend=False, capstyle="round")

    # ── ID labels ─────────────────────────────────────────────────────────────
    for _, row in df_dub.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        mid = row.geometry.interpolate(0.5, normalized=True)
        ax.annotate(str(int(row["ID_new"])), xy=(mid.x, mid.y),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=10, fontweight="bold", color="black", zorder=16,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              alpha=0.8, ec="none"))

    # ── Raster overlay ────────────────────────────────────────────────────────
    raster = rasterio.open(banned_area)
    rasterio.plot.show(raster, ax=ax,
                       cmap=ListedColormap(["white", "white"]), zorder=3)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.4)
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(f"{title_bar} [Mio. CHF]", rotation=90, labelpad=16, fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    # ── Map extent, scale bar, north arrow ───────────────────────────────────
    ax.set_xlim(DUB_E_MIN, DUB_E_MAX)
    ax.set_ylim(DUB_N_MIN, DUB_N_MAX)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_artist(ScaleBar(1, location="lower right"))
    ax.text(0.96, 0.92, "N", fontsize=22, weight="bold",
            ha="center", va="center", transform=ax.transAxes, zorder=1000)
    ax.add_patch(FancyArrowPatch((0.96, 0.89), (0.96, 0.97), color="black", lw=2,
                                 arrowstyle="->", mutation_scale=20,
                                 transform=ax.transAxes, zorder=1000))
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor("black")
        sp.set_linewidth(1); sp.set_zorder(1000)

    water_patch = mpatches.Patch(facecolor="lightblue", label="Water bodies",
                                 edgecolor="black", linewidth=1)
    ax.legend(handles=[water_patch], loc="upper center",
              bbox_to_anchor=(0.5, -0.02), fontsize=12, frameon=False)

    ax.set_title(f"Dübendorf detail — {title_bar}", fontsize=13, pad=8)

    plt.tight_layout()
    os.makedirs("plot/results", exist_ok=True)
    plt.savefig(f"plot/results/{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[plot_duebendorf_zoom] saved → plot/results/{plot_name}.png")




def plot_bcr_bar(plot_name="bcr_bar"):
    """
    Horizontal bar chart of Cost-Benefit Ratio (CBR) per development,
    sorted descending.  CBR = |C + M| / (T_s2 + R_s2 + S_s2).
    A vertical dashed line at CBR = 1 marks the break-even threshold.
    Bars are green (CBR ≤ 1) or red (CBR > 1).
    Developments with zero benefits (CBR = ∞) are capped and labelled.

    Reads:  data/costs/net_benefits.gpkg
    Saves:  plot/results/{plot_name}.png
    """
    nb_path = "data/costs/net_benefits.gpkg"
    if not os.path.exists(nb_path):
        print(f"[plot_bcr_bar] Missing: {nb_path} — skipping")
        return

    nb = gpd.read_file(nb_path)
    for c in ["C", "M", "T_s2", "R_s2", "S_s2"]:
        nb[c] = pd.to_numeric(nb[c], errors="coerce")

    nb["total_costs"]    = nb["C"].abs() + nb["M"].abs()
    nb["total_benefits"] = nb["T_s2"] + nb["R_s2"] + nb["S_s2"]
    nb = nb[nb["total_costs"] > 0].copy()

    # CBR = costs / benefits; NaN where benefits = 0
    nb["CBR"] = nb.apply(
        lambda r: r["total_costs"] / r["total_benefits"]
        if r["total_benefits"] > 0 else np.nan, axis=1)

    # sort descending (worst first at top); NaN always at top
    nb = nb.sort_values("CBR", ascending=False, na_position="first").reset_index(drop=True)
    nb["is_inf"] = nb["CBR"].isna()

    # cap: place infinite bars clearly beyond the largest finite bar
    finite_max = nb["CBR"].dropna().max()
    CAP = finite_max * 2.0          # 2× gap makes infinite bars visually distinct
    nb["CBR_plot"] = nb["CBR"].fillna(CAP)

    # colours: infinite → green (CBR > 1), finite ≤ 1 → red, finite > 1 → green
    colors = []
    for i in range(len(nb)):
        if nb["is_inf"].iloc[i]:
            colors.append("#e74c3c")
        elif nb["CBR"].iloc[i] <= 1:
            colors.append("#27ae60")
        else:
            colors.append("#e74c3c")

    fig, ax = plt.subplots(figsize=(9, max(5, len(nb) * 0.45)))
    y = np.arange(len(nb))
    ax.barh(y, nb["CBR_plot"], color=colors, alpha=0.85,
            edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(1.0, color="black", lw=1.5, linestyle="--", zorder=5)
    ax.set_xlim(0, CAP * 1.05)

    # label and arrow for infinite bars
    for pos in nb.index[nb["is_inf"]]:
        ax.annotate("∞  (no benefits)",
                    xy=(CAP, pos), xytext=(CAP * 0.75, pos),
                    va="center", ha="left", fontsize=8,
                    color="white", fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.2))

    ax.set_yticks(y)
    ax.set_yticklabels(nb["ID_new"].astype(int).astype(str), fontsize=9)
    ax.set_xlabel("Cost-Benefit Ratio  |C + M| / (T + R + S)", fontsize=11)
    ax.set_title("Cost-Benefit Ratio per Development\n"
                 "(medium growth scenario  ·  CBR < 1 = net positive)",
                 fontsize=12, pad=8)
    ax.grid(axis="x", linestyle="--", alpha=0.5, zorder=0)

    legend_handles = [
        mpatches.Patch(color="#27ae60", label="CBR ≤ 1  (benefits exceed costs)"),
        mpatches.Patch(color="#e74c3c", label="CBR > 1  (costs exceed benefits)"),
        plt.Line2D([0], [0], color="black", lw=1.5, linestyle="--",
                   label="Break-even  (CBR = 1)"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right")

    plt.tight_layout()
    os.makedirs("plot/results", exist_ok=True)
    plt.savefig(f"plot/results/{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print(f"[plot_bcr_bar] saved → plot/results/{plot_name}.png")




def plot_netzluecken_closeup(
        dev_ids=(676, 97, 566),
        buffer_m=500,
        save_path="figures/netzluecken_closeup.pdf",
):
    """
    1 × 3 close-up map for each specified Netzlücke ID.

    Each panel shows:
      • OSM basemap (contextily, EPSG:3857)
      • Surrounding network clipped to a buffer around the edge (grey)
      • The highlighted Netzlücke edge (red, thick)
      • Voronoi node centroids (blue dots)
      • Title: ID · ΔT [h/day] · TTS [MCHF] · NB [MCHF]

    Data read from:
      data/Network/processed/network_full_annotated_edges.gpkg
      data/Voronoi/voronoi_developments_euclidian_values.shp
      data/OD/traveltime_savings_od.csv
      data/costs/net_benefits.csv
    """
    try:
        import contextily as ctx
        HAS_CTX = True
    except ImportError:
        HAS_CTX = False
        print("[plot_netzluecken_closeup] contextily not installed — basemap skipped")

    # ── Load data ────────────────────────────────────────────────────────────
    # Netzlücken geometries keyed by ID_new (original IDs preserved here)
    nl_all  = gpd.read_file("data/Network/processed/development_candidates.gpkg")
    nl_all["ID_new"] = nl_all["ID_new"].astype(int)

    # Full network for background (all route types)
    edges   = gpd.read_file("data/Network/processed/network_full_annotated_edges.gpkg")
    voronoi = gpd.read_file("data/Voronoi/voronoi_developments_euclidian_values.shp")
    tts_df  = pd.read_csv("data/OD/traveltime_savings_od.csv")
    nb_df   = pd.read_csv("data/costs/net_benefits.csv")

    # Voronoi node centroids
    voronoi_pts = voronoi.copy()
    voronoi_pts["geometry"] = voronoi_pts.geometry.centroid

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), facecolor="white")

    for ax, dev_id in zip(axes, dev_ids):

        # ── Lookup this Netzlücke ─────────────────────────────────────────────
        nl_row = nl_all[nl_all["ID_new"] == dev_id]
        if nl_row.empty:
            ax.set_title(f"ID {dev_id} — not found", fontsize=11)
            ax.axis("off")
            continue

        # ── TTS / NB labels ───────────────────────────────────────────────────
        tts_row = tts_df[tts_df["ID_new"] == dev_id]
        nb_row  = nb_df[nb_df["ID_new"]  == dev_id]
        dt_hday = tts_row["T_s2"].values[0]   if len(tts_row) else 0.0
        nb_mchf = nb_row["NB_s2"].values[0] / 1e6 if len(nb_row) else 0.0
        tts_mchf = (nb_row["T_s2"].values[0] / 1e6
                    if "T_s2" in nb_row.columns and len(nb_row) else 0.0)

        # ── Reproject to EPSG:3857 for basemap ───────────────────────────────
        nl_3857     = nl_row.to_crs(epsg=3857)
        edges_3857  = edges.to_crs(epsg=3857)
        voronoi_3857 = voronoi_pts.to_crs(epsg=3857)

        # ── Buffer around the Netzlücke → plot extent ────────────────────────
        buf         = nl_3857.geometry.buffer(buffer_m).unary_union
        minx, miny, maxx, maxy = buf.bounds

        # ── Clip network and Voronoi to buffer ────────────────────────────────
        from shapely.geometry import box as shapely_box
        clip_geom       = shapely_box(minx, miny, maxx, maxy)
        edges_clip      = edges_3857[edges_3857.geometry.intersects(clip_geom)]
        voronoi_clip    = voronoi_3857[voronoi_3857.geometry.within(clip_geom)]

        # ── Draw layers ───────────────────────────────────────────────────────
        # Background network
        edges_clip.plot(ax=ax, color="#888888", linewidth=1.0,
                        alpha=0.6, zorder=2)

        # Netzlücke edge (highlighted)
        nl_3857.plot(ax=ax, color="#e74c3c", linewidth=4.0,
                     zorder=4, label=f"Netzlücke ID {dev_id}")

        # Voronoi centroids
        if len(voronoi_clip) > 0:
            ax.scatter(voronoi_clip.geometry.x, voronoi_clip.geometry.y,
                       s=20, color="#2980b9", zorder=5, alpha=0.8,
                       label="Voronoi nodes")

        # ── Basemap ───────────────────────────────────────────────────────────
        if HAS_CTX:
            try:
                ctx.add_basemap(ax, crs="EPSG:3857",
                                source=ctx.providers.OpenStreetMap.Mapnik,
                                zoom="auto", alpha=0.6, zorder=1)
            except Exception as e:
                print(f"  [basemap] ID {dev_id}: {e}")

        # ── Extent and labels ─────────────────────────────────────────────────
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # Midpoint label on the edge
        mid = nl_3857.geometry.iloc[0].interpolate(0.5, normalized=True)
        ax.annotate(f"ID {dev_id}", xy=(mid.x, mid.y),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=10, fontweight="bold", color="#c0392b",
                    zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              alpha=0.7, ec="none"))

        length_m = tts_row["length_m"].values[0] if len(tts_row) else 0.0
        note = "\n← zero benefit: not on any OD path" if dt_hday == 0 else ""
        ax.set_title(
            f"ID {dev_id}  ·  {length_m:.0f} m\n"
            f"ΔT = {dt_hday:.2f} h/day  ·  "
            f"NB = {nb_mchf:.2f} MCHF{note}",
            fontsize=9, pad=6,
        )

        ax.legend(fontsize=7, loc="lower left", framealpha=0.8)

    fig.suptitle(
        "Netzlücken close-up — IDs 676, 97, 566  "
        "(medium-growth scenario S2, 50-year appraisal)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[plot_netzluecken_closeup] saved → {save_path}")


def plot_tts_area_closeups(
        network=None,
        access_points=None,
        col="T_s2",
        save_path="plot/results/tts_area_closeups.png",
):
    """
    1 × 3 close-up panels of travel-time savings (same visual style as
    plot_duebendorf_zoom / plot_single_cost_result) for three sub-areas:

      Panel 1 — Dübendorf cluster  (IDs 77, 78, 79, 90, 415, 676, 703, 799, 800, 908)
      Panel 2 — Uster              (ID 566)
      Panel 3 — Greifensee / lake  (IDs 97, 385)

    Development lines are coloured by TTS [Mio. CHF] using the corridor-wide
    diverging colourmap so all three panels share one consistent scale.
    """
    lakes_path = r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp"

    # ── Load and prepare data ─────────────────────────────────────────────────
    dev_geom = gpd.read_file(
        "data/Network/processed/development_candidates.gpkg")[["ID_new", "geometry"]]
    dev_geom["ID_new"] = dev_geom["ID_new"].astype(int)

    nb_df = pd.read_csv("data/costs/net_benefits.csv")
    nb_df["ID_new"] = nb_df["ID_new"].astype(int)
    nb_df[col] = nb_df[col] / 1e6   # CHF → Mio. CHF

    df_plot = dev_geom.merge(
        nb_df[["ID_new", col]].drop_duplicates("ID_new"),
        on="ID_new", how="inner",
    )
    df_plot = gpd.GeoDataFrame(df_plot, geometry="geometry", crs="EPSG:2056")
    df_plot = df_plot.dropna(subset=[col])

    # Corridor-wide colour scale so all panels are comparable
    min_val, max_val = df_plot[col].min(), df_plot[col].max()
    cmap = _make_diverging_cmap(min_val, max_val)
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

    # ── Three sub-area extents (EPSG:2056, LV95) ─────────────────────────────
    areas = [
        ("Dübendorf",        2_685_790, 2_692_550, 1_248_135, 1_253_237),
        ("Uster",            2_694_960, 2_698_400, 1_243_775, 1_247_078),
        ("Greifensee / lake",2_685_825, 2_693_293, 1_238_425, 1_246_696),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 8), facecolor="white")

    for ax, (area_name, e_min, e_max, n_min, n_max) in zip(axes, areas):

        # ── Base layers ───────────────────────────────────────────────────────
        if os.path.exists(lakes_path):
            gpd.read_file(lakes_path).plot(ax=ax, color="lightblue", zorder=1)

        if isinstance(network, gpd.GeoDataFrame):
            network.plot(ax=ax, color="#888888", lw=1.0, zorder=2, alpha=0.55)

        if isinstance(access_points, gpd.GeoDataFrame):
            access_points.plot(ax=ax, color="black", markersize=30, zorder=4)

        # ── Development lines coloured by TTS ────────────────────────────────
        df_area = df_plot.cx[e_min:e_max, n_min:n_max].copy()
        if not df_area.empty:
            df_area.plot(ax=ax, column=col, cmap=cmap, norm=norm,
                         linewidth=7, zorder=3, legend=False, capstyle="round")

            for _, row in df_area.iterrows():
                if row.geometry is None or row.geometry.is_empty:
                    continue
                mid = row.geometry.interpolate(0.5, normalized=True)
                ax.annotate(
                    str(int(row["ID_new"])), xy=(mid.x, mid.y),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=10, fontweight="bold", color="black", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              alpha=0.85, ec="none"),
                )

        # ── Extent, ticks, scale, north arrow ────────────────────────────────
        ax.set_xlim(e_min, e_max)
        ax.set_ylim(n_min, n_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_artist(ScaleBar(1, location="lower right"))
        ax.text(0.96, 0.92, "N", fontsize=18, weight="bold",
                ha="center", va="center", transform=ax.transAxes, zorder=100)
        ax.add_patch(FancyArrowPatch(
            (0.96, 0.89), (0.96, 0.97), color="black", lw=1.5,
            arrowstyle="->", mutation_scale=16,
            transform=ax.transAxes, zorder=100))
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor("black")
            sp.set_linewidth(1)

        ax.set_title(area_name, fontsize=12, pad=6)

    # ── Shared colourbar ──────────────────────────────────────────────────────
    sm = mcm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.55, pad=0.01, aspect=30)
    cbar.set_label("Travel-time savings [Mio. CHF]  (medium growth S2, 50-year appraisal)",
                   rotation=90, labelpad=14, fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle("Travel-Time Savings — close-up by sub-area",
                 fontsize=14, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"[plot_tts_area_closeups] saved → {save_path}")
