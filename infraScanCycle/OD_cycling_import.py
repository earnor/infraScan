"""
graph_generation.py
All 7 infraScanCycle visualisation plots in one file.
Run from the project root: python graph_generation.py

Plots saved to plots/01_... through plots/07_...
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import contextily as ctx
import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box

os.makedirs("plots", exist_ok=True)

E_MIN, E_MAX = 2687000, 2708000
N_MIN, N_MAX = 1237000, 1254000
CRS = "EPSG:2056"
CORRIDOR_BOX = box(E_MIN, N_MIN, E_MAX, N_MAX)


def add_basemap(ax):
    try:
        ctx.add_basemap(ax, crs=CRS, source=ctx.providers.CartoDB.Positron,
                        zoom=12, alpha=0.6)
    except Exception as e:
        print(f"  Basemap skipped: {e}")
        ax.set_facecolor("#F0F0EC")


def add_corridor(ax, lw=1.5, alpha=0.5):
    gpd.GeoDataFrame(geometry=[CORRIDOR_BOX], crs=CRS).boundary.plot(
        ax=ax, color="#215CAF", linewidth=lw, linestyle="--", alpha=alpha, zorder=6)


def base_ax(figsize=(11, 9), pad=300):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(E_MIN - pad, E_MAX + pad)
    ax.set_ylim(N_MIN - pad, N_MAX + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (LV95)", fontsize=9)
    ax.set_ylabel("Northing (LV95)", fontsize=9)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# 01 — Study corridor
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 01 Study corridor ──")

corridor = gpd.GeoDataFrame(geometry=[CORRIDOR_BOX], crs=CRS)
fig, ax = base_ax(figsize=(10, 8), pad=500)
add_basemap(ax)

corridor.plot(ax=ax, facecolor="none", edgecolor="#215CAF", linewidth=2.5, linestyle="--")
corridor.plot(ax=ax, facecolor="#215CAF", alpha=0.07)

for e, n, ha, va in [
    (E_MIN, N_MIN, "right", "top"),
    (E_MAX, N_MIN, "left",  "top"),
    (E_MIN, N_MAX, "right", "bottom"),
    (E_MAX, N_MAX, "left",  "bottom"),
]:
    ax.annotate(f"E {e:,}\nN {n:,}", xy=(e, n), fontsize=7,
                ha=ha, va=va, color="#215CAF",
                xytext=(8 if ha == "left" else -8, 8 if va == "bottom" else -8),
                textcoords="offset points")

sb_x0, sb_x1 = E_MIN + 1000, E_MIN + 6000
sb_y = N_MIN + 700
ax.plot([sb_x0, sb_x1], [sb_y, sb_y], color="black", lw=2)
ax.plot([sb_x0] * 2, [sb_y - 150, sb_y + 150], color="black", lw=1.5)
ax.plot([sb_x1] * 2, [sb_y - 150, sb_y + 150], color="black", lw=1.5)
ax.text((sb_x0 + sb_x1) / 2, sb_y + 350, "5 km", ha="center", fontsize=8.5, fontweight="bold")
ax.text((E_MIN + E_MAX) / 2, (N_MIN + N_MAX) / 2, "21 × 17 km",
        ha="center", va="center", fontsize=13, color="#215CAF", alpha=0.3, fontweight="bold")

ax.set_xlabel("Easting (LV95 / EPSG:2056)", fontsize=9)
ax.set_title("01 — Study corridor\ninfraScanCycle  |  Zürich region",
             fontsize=12, fontweight="bold", color="#215CAF", loc="left", pad=10)
patch = mpatches.Patch(facecolor="#215CAF", alpha=0.15, edgecolor="#215CAF", linewidth=2,
                       label=f"Corridor  {(E_MAX - E_MIN) / 1000:.0f} × {(N_MAX - N_MIN) / 1000:.0f} km")
ax.legend(handles=[patch], loc="lower right", fontsize=9, framealpha=0.9,
          edgecolor="#CCC", fancybox=False)

plt.tight_layout()
plt.savefig("plots/01_study_area.png", dpi=180, bbox_inches="tight")
print("Saved → plots/01_study_area.png")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 02 — Raw Velonetz Alltag OGD
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 02 Raw Velonetz ──")

PATH_VELONETZ = r"data/raw/ALLTAG/OGD_VELO_ALLTAG_NETZ_L_M.shp"

network = gpd.read_file(PATH_VELONETZ)
if network.crs.to_epsg() != 2056:
    network = network.to_crs(CRS)
network_clip = network.clip(CORRIDOR_BOX)
print(f"  Edges in corridor: {len(network_clip)}")
print(network_clip["ROUTENTYP"].value_counts())
print(network_clip["PLANUNGSTY"].value_counts())

ROUTENTYP_COLORS = {
    "Hauptverbindung":                "#185FA5",
    "Nebenverbindung":                "#5DCAA5",
    "Zusätzliche Freizeitverbindung": "#EF9F27",
    "Velobahn":                       "#E24B4A",
}
PLANUNGSTY_STYLES = {
    "bestehend":              "-",
    "geplant":                "--",
    "Variante":               ":",
    "bei Ersatz aufzuhebend": (0, (3, 1, 1, 1)),
}

fig, ax = base_ax()
add_basemap(ax)

for rtype, color in ROUTENTYP_COLORS.items():
    for ptype, style in PLANUNGSTY_STYLES.items():
        sub = network_clip[
            (network_clip["ROUTENTYP"] == rtype) &
            (network_clip["PLANUNGSTY"] == ptype)
        ]
        if len(sub):
            lw = 2.4 if rtype == "Hauptverbindung" else 1.6 if rtype == "Velobahn" else 1.8
            sub.plot(ax=ax, color=color, linewidth=lw, linestyle=style, alpha=0.9, zorder=4)

others = network_clip[~network_clip["ROUTENTYP"].isin(ROUTENTYP_COLORS)]
if len(others):
    others.plot(ax=ax, color="#888780", linewidth=1.0, alpha=0.5, zorder=3)

add_corridor(ax)

counts  = network_clip["ROUTENTYP"].value_counts()
pcounts = network_clip["PLANUNGSTY"].value_counts()

type_handles = [
    mlines.Line2D([], [], color=c, linewidth=2.5, label=f"{l}  ({counts.get(l, 0)})")
    for l, c in ROUTENTYP_COLORS.items()
]
if len(others):
    type_handles.append(mlines.Line2D([], [], color="#888780", linewidth=1.5,
                                      label=f"Other ({len(others)})"))

plan_handles = [
    mlines.Line2D([], [], color="#444", linewidth=1.8, linestyle=s,
                  label=f"{p}  ({pcounts.get(p, 0)})")
    for p, s in PLANUNGSTY_STYLES.items() if pcounts.get(p, 0) > 0
]

leg1 = ax.legend(handles=type_handles, loc="lower right", fontsize=9,
                 title="Route type", title_fontsize=9,
                 framealpha=0.93, edgecolor="#CCC", fancybox=False)
ax.add_artist(leg1)
ax.legend(handles=plan_handles, loc="lower left", fontsize=9,
          title="Planning status", title_fontsize=9,
          framealpha=0.93, edgecolor="#CCC", fancybox=False)

ax.set_title(f"02 — Raw Velonetz Alltag OGD  ·  {len(network_clip)} edges in corridor",
             fontsize=12, fontweight="bold", color="#185FA5", loc="left", pad=10)

plt.tight_layout()
plt.savefig("plots/02_velonetz_raw.png", dpi=180, bbox_inches="tight")
print("Saved → plots/02_velonetz_raw.png")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 03 — Processed network: simplified edges & nodes
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 03 Processed network ──")

edges  = gpd.read_file(r"data/Network/processed/edges_corridor.gpkg")
points = gpd.read_file(r"data/Network/processed/points_corridor.gpkg")
for gdf in [edges, points]:
    if gdf.crs and gdf.crs.to_epsg() != 2056:
        gdf.to_crs(CRS, inplace=True)

print(f"  Edges:  {len(edges)}")
print(f"  Points: {len(points)}")

fig, ax = base_ax()
add_basemap(ax)
edges.plot(ax=ax, color="#185FA5", linewidth=1.4, alpha=0.75, zorder=3)

if "is_intersection" in points.columns:
    intersections = points[points["is_intersection"] == 1]
    endpoints     = points[points["is_intersection"] == 0]
    intersections.plot(ax=ax, color="#E24B4A", markersize=18, marker="o", alpha=0.85, zorder=5)
    endpoints.plot(ax=ax, color="#EF9F27", markersize=10, marker="s", alpha=0.75, zorder=4)
    node_handles = [
        mpatches.Patch(color="#E24B4A", label=f"Intersection ({len(intersections)})"),
        mpatches.Patch(color="#EF9F27", label=f"Endpoint ({len(endpoints)})"),
    ]
else:
    points.plot(ax=ax, color="#E24B4A", markersize=12, marker="o", alpha=0.8, zorder=5)
    node_handles = [mpatches.Patch(color="#E24B4A", label=f"Node ({len(points)})")]

add_corridor(ax)
handles = [mlines.Line2D([], [], color="#185FA5", linewidth=2,
                         label=f"Simplified edge ({len(edges)})")] + node_handles
ax.legend(handles=handles, loc="lower right", fontsize=9,
          title="Processed network", title_fontsize=9,
          framealpha=0.93, edgecolor="#CCC", fancybox=False)
ax.set_title("03 — Processed network: simplified edges & nodes",
             fontsize=12, fontweight="bold", color="#185FA5", loc="left", pad=10)

plt.tight_layout()
plt.savefig("plots/03_network_processed.png", dpi=180, bbox_inches="tight")
print("Saved → plots/03_network_processed.png")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 04 — Generated access points & routed candidate links
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 04 Generated candidates ──")

edges      = gpd.read_file(r"data/Network/processed/edges_corridor.gpkg")
gen_nodes  = gpd.read_file(r"data/Network/processed/generated_nodes.gpkg")
candidates = gpd.read_file(r"data/Network/processed/new_links_realistic.gpkg")
access_pts = gpd.read_file(r"data/Network/processed/access_points_corridor.gpkg")
for gdf in [edges, gen_nodes, candidates, access_pts]:
    if gdf.crs and gdf.crs.to_epsg() != 2056:
        gdf.to_crs(CRS, inplace=True)

print(f"  Existing edges:   {len(edges)}")
print(f"  Generated points: {len(gen_nodes)}")
print(f"  Candidate links:  {len(candidates)}")
print(f"  Access points:    {len(access_pts)}")

fig, ax = base_ax()
add_basemap(ax)
edges.plot(ax=ax, color="#B5D4F4", linewidth=1.2, alpha=0.8, zorder=2)
candidates.plot(ax=ax, color="#E24B4A", linewidth=1.5, alpha=0.75, zorder=3)
access_pts.plot(ax=ax, color="#185FA5", markersize=12, marker="o", alpha=0.7, zorder=4)
gen_nodes.plot(ax=ax, color="#D4537E", markersize=22, marker="^", alpha=0.85, zorder=5)
add_corridor(ax)

handles = [
    mlines.Line2D([], [], color="#B5D4F4", linewidth=2,
                  label=f"Existing network ({len(edges)} edges)"),
    mlines.Line2D([], [], color="#E24B4A", linewidth=2,
                  label=f"Candidate links ({len(candidates)})"),
    mlines.Line2D([], [], color="#185FA5", marker="o", linewidth=0,
                  markersize=7, label=f"Existing access points ({len(access_pts)})"),
    mlines.Line2D([], [], color="#D4537E", marker="^", linewidth=0,
                  markersize=9, label=f"Generated points ({len(gen_nodes)})"),
]
ax.legend(handles=handles, loc="lower right", fontsize=9,
          title="Candidate generation", title_fontsize=9,
          framealpha=0.93, edgecolor="#CCC", fancybox=False)
ax.set_title("04 — Generated access points & routed candidate links",
             fontsize=12, fontweight="bold", color="#E24B4A", loc="left", pad=10)

plt.tight_layout()
plt.savefig("plots/04_generated_candidates.png", dpi=180, bbox_inches="tight")
print("Saved → plots/04_generated_candidates.png")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 05 — Protected areas & routing constraints
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 05 Protected areas ──")

PATH_PROTECTED_RASTER = r"data/landuse_landcover/processed/zone_no_infra/protected_area_corridor.tif"
PATH_UNPRODUCTIVE     = r"data/landuse_landcover/processed/unproductive_area.gpkg"
PATH_LANDUSE          = r"data/landuse_landcover/processed/landuse.gpkg"

fig, ax = base_ax()
add_basemap(ax)
handles = []

if os.path.exists(PATH_PROTECTED_RASTER):
    with rasterio.open(PATH_PROTECTED_RASTER) as src:
        data   = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    masked = np.where(data > 0, 1.0, np.nan)
    ax.imshow(masked, extent=extent, origin="upper",
              cmap="Greens", alpha=0.55, vmin=0, vmax=1, aspect="auto", zorder=3)
    handles.append(mpatches.Patch(facecolor="#639922", alpha=0.6, edgecolor="#3B6D11",
                                  linewidth=0.8, label="Protected area (routing blocked)"))
    print("  Protected raster: loaded")
else:
    print(f"  Raster not found: {PATH_PROTECTED_RASTER}")

for path, color, label in [
    (PATH_UNPRODUCTIVE, "#97C459", "Unproductive land"),
    (PATH_LANDUSE,      "#5DCAA5", "Land use zones"),
]:
    if os.path.exists(path):
        gdf = gpd.read_file(path)
        if gdf.crs and gdf.crs.to_epsg() != 2056:
            gdf = gdf.to_crs(CRS)
        gdf_clip = gdf.clip(CORRIDOR_BOX)
        gdf_clip.plot(ax=ax, facecolor=color, alpha=0.3, edgecolor=color,
                      linewidth=0.5, zorder=4)
        handles.append(mpatches.Patch(facecolor=color, alpha=0.4,
                                      edgecolor=color, label=f"{label} ({len(gdf_clip)})"))
        print(f"  {label}: {len(gdf_clip)} features")

add_corridor(ax, lw=1.5, alpha=0.6)
handles.append(mpatches.Patch(facecolor="none", edgecolor="#215CAF",
                               linewidth=1.5, linestyle="--", label="Corridor boundary"))
ax.legend(handles=handles, loc="lower right", fontsize=9,
          title="Protected areas", title_fontsize=9,
          framealpha=0.93, edgecolor="#CCC", fancybox=False)
ax.set_title("05 — Protected areas & routing constraints",
             fontsize=12, fontweight="bold", color="#3B6D11", loc="left", pad=10)

plt.tight_layout()
plt.savefig("plots/05_protected_areas.png", dpi=180, bbox_inches="tight")
print("Saved → plots/05_protected_areas.png")
plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 06 — OSM feeder network
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 06 OSM feeder network ──")

PATH_OSM_DIR  = r"data/Network/OSM_road"
PATH_VEL_PROC = r"data/Network/processed/edges_corridor.gpkg"

osm_files = glob.glob(os.path.join(PATH_OSM_DIR, "sub_area_edges_*.gpkg"))
if not osm_files:
    osm_files = glob.glob(os.path.join(PATH_OSM_DIR, "*.gpkg"))

osm_gdfs = []
for fp in sorted(osm_files):
    try:
        gdf = gpd.read_file(fp)
        if gdf.crs and gdf.crs.to_epsg() != 2056:
            gdf = gdf.to_crs(CRS)
        osm_gdfs.append(gdf)
    except Exception as e:
        print(f"  Could not load {fp}: {e}")

if not osm_gdfs:
    print(f"  No OSM files found in {PATH_OSM_DIR} — skipping plot 06")
else:
    osm_all  = pd.concat(osm_gdfs, ignore_index=True)
    osm_clip = osm_all.clip(CORRIDOR_BOX)
    print(f"  OSM edges in corridor: {len(osm_clip)}")

    HIGHWAY_COLORS = {
        "cycleway":      "#185FA5",
        "path":          "#5DCAA5",
        "residential":   "#888780",
        "service":       "#B4B2A9",
        "footway":       "#D3D1C7",
        "living_street": "#D3D1C7",
        "unclassified":  "#B4B2A9",
        "secondary":     "#EF9F27",
        "tertiary":      "#EF9F27",
        "primary":       "#E24B4A",
    }

    fig, ax = base_ax()
    add_basemap(ax)
    handles = []

    if "highway" in osm_clip.columns:
        for hw, color in HIGHWAY_COLORS.items():
            sub = osm_clip[osm_clip["highway"] == hw]
            if len(sub):
                lw = 1.8 if hw == "cycleway" else 0.8
                sub.plot(ax=ax, color=color, linewidth=lw, alpha=0.7, zorder=3)
                handles.append(mlines.Line2D([], [], color=color, linewidth=max(lw, 1.2),
                                             label=f"{hw} ({len(sub)})"))
        others = osm_clip[~osm_clip["highway"].isin(HIGHWAY_COLORS)]
        if len(others):
            others.plot(ax=ax, color="#CCCCCC", linewidth=0.5, alpha=0.4, zorder=2)
            handles.append(mlines.Line2D([], [], color="#CCCCCC", linewidth=1,
                                         label=f"Other ({len(others)})"))
    else:
        osm_clip.plot(ax=ax, color="#D4537E", linewidth=0.7, alpha=0.6, zorder=3)
        handles.append(mlines.Line2D([], [], color="#D4537E", linewidth=1.2,
                                     label=f"OSM edges ({len(osm_clip)})"))

    if os.path.exists(PATH_VEL_PROC):
        vel = gpd.read_file(PATH_VEL_PROC)
        if vel.crs and vel.crs.to_epsg() != 2056:
            vel = vel.to_crs(CRS)
        vel.plot(ax=ax, color="#215CAF", linewidth=2.0, linestyle="--", alpha=0.9, zorder=5)
        handles.append(mlines.Line2D([], [], color="#215CAF", linewidth=2,
                                     linestyle="--", label="Velonetz (reference)"))

    add_corridor(ax, alpha=0.4)
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              title="OSM highway type", title_fontsize=9,
              framealpha=0.93, edgecolor="#CCC", fancybox=False,
              ncol=2 if len(handles) > 6 else 1)
    ax.set_title(f"06 — OSM feeder network  ({len(osm_clip)} edges)\n"
                 f"Local roads used to reach the main cycling network",
                 fontsize=12, fontweight="bold", color="#D4537E", loc="left", pad=10)

    plt.tight_layout()
    plt.savefig("plots/06_osm_feeder_network.png", dpi=180, bbox_inches="tight")
    print("Saved → plots/06_osm_feeder_network.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 07 — OD commuter flows by Gemeinde (BFS Pendlermatrix)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 07 OD communes (BFS Pendlermatrix) ──")


def import_pendler_matrix(
    path_csv: str = 'data/OD/pendler_matrix.csv',
    canton_filter: str = 'ZH',
    cycling_mode_share: float = 0.05,
    max_distance_km: float = 15.0,
    output_dir: str = 'data/OD',
):
    """
    Import and process the BFS Pendlermatrix (OD matrix at Gemeinde level).

    DATA SOURCE — download manually before running:
      https://www.bfs.admin.ch/asset/de/ts-x-11.04.04.05-2018
      → CSV file: "Erwerbstätige nach Wohn- und Arbeitsgemeinde"

    Key columns:
      WOHNKANTON / WOHNGEMEINDE    : residence canton + BFS Gemeinde number
      ARBEITSKANTON / ARBEITSGEMEINDE : workplace canton + BFS Gemeinde number
      ERWERBSTAETIGE               : number of commuters on that OD pair
    """
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(path_csv):
        raise FileNotFoundError(
            f"Missing: {path_csv}\n"
            "Download from: https://www.bfs.admin.ch/asset/de/ts-x-11.04.04.05-2018\n"
            "Save the CSV as data/OD/pendler_matrix.csv"
        )

    raw = pd.read_csv(path_csv, sep=';', encoding='utf-8-sig', dtype=str)
    print(f"  Raw columns: {raw.columns.tolist()}")
    raw.columns = raw.columns.str.strip().str.upper()

    # Support both old BFS format and newer GEO_* format
    col_map = {
        'WOHNKANTON':      next((c for c in raw.columns if c in
                                 ['WOHNKANTON', 'GEO_CANT_RESID']), None),
        'WOHNGEMEINDE':    next((c for c in raw.columns if c in
                                 ['WOHNGEMEINDE', 'GEO_COMM_RESID']), None),
        'ARBEITSKANTON':   next((c for c in raw.columns if c in
                                 ['ARBEITSKANTON', 'GEO_CANT_WORK']), None),
        'ARBEITSGEMEINDE': next((c for c in raw.columns if c in
                                 ['ARBEITSGEMEINDE', 'GEO_COMM_WORK']), None),
        'ERWERBSTAETIGE':  next((c for c in raw.columns if c in
                                 ['ERWERBSTAETIGE', 'VALUE']), None),
    }
    missing = [k for k, v in col_map.items() if v is None]
    if missing:
        raise KeyError(f"Could not find columns: {missing}. Available: {raw.columns.tolist()}")

    od = raw.rename(columns={v: k for k, v in col_map.items() if v})[list(col_map.keys())].copy()
    od['ERWERBSTAETIGE'] = pd.to_numeric(od['ERWERBSTAETIGE'], errors='coerce').fillna(0).astype(int)
    print(f"  Loaded {len(od)} OD pairs, {od['ERWERBSTAETIGE'].sum():,} total commuters")

    # BFS new format uses numeric canton IDs (Zürich = '1'), old format uses 'ZH'
    # Try both so the function works with either CSV version
    zh_ids = {canton_filter, '1', 1, 'ZH', 'zh'}
    mask  = (od['WOHNKANTON'].astype(str).isin([str(x) for x in zh_ids])) |             (od['ARBEITSKANTON'].astype(str).isin([str(x) for x in zh_ids]))
    od_zh = od[mask].copy().reset_index(drop=True)
    print(f"  After canton filter ({canton_filter}): {len(od_zh)} pairs, "
          f"{od_zh['ERWERBSTAETIGE'].sum():,} commuters")

    gem_path = 'data/raw/Gemeinden/gemeinden_centroid.gpkg'
    if os.path.exists(gem_path):
        # tlm_hoheitsgebiet is the Gemeinde layer in the swisstopo GeoPackage
        import fiona
        layers = fiona.listlayers(gem_path)
        print(f"  Layers in {gem_path}: {layers}")
        gem_layer = next(
            (l for l in layers if 'hoheitsgebiet' in l.lower() or 'gemeinde' in l.lower()),
            layers[0]
        )
        print(f"  Using layer: {gem_layer}")
        gemeinden = gpd.read_file(gem_path, layer=gem_layer)
        if gemeinden.crs and gemeinden.crs.to_epsg() != 2056:
            gemeinden = gemeinden.to_crs("EPSG:2056")

        print(f"  Gemeinde columns: {gemeinden.columns.tolist()}")

        # Find the BFS municipality number column — swisstopo uses OBJECTVAL or GEMNAME
        id_col = next(
            (c for c in gemeinden.columns
             if c.upper() in ['GMDNR', 'BFS_NR', 'GEMEINDENR', 'NR', 'OBJECTVAL',
                               'BFSNR', 'GEM_NR', 'NUMMER', 'GKZ',
                               'BFS_NUMMER', 'BFSNUMMER']),
            None
        )
        if id_col is None:
            # Last resort: any column whose values look like 4-digit numbers
            for c in gemeinden.columns:
                if gemeinden[c].dtype in ['int64', 'float64', 'object']:
                    sample = gemeinden[c].dropna().astype(str).str.strip()
                    if sample.str.match(r'^\d{1,4}$').mean() > 0.8:
                        id_col = c
                        break

        if id_col is None:
            raise KeyError(
                f"Cannot find BFS Gemeinde ID column. "
                f"Available columns: {gemeinden.columns.tolist()}"
            )

        print(f"  Using ID column: {id_col}")
        gemeinden['BFS_NR'] = gemeinden[id_col].astype(str).str.strip().str.zfill(4)

        # Filter to Canton Zürich only (kantonsnummer == 1) if column exists
        if 'kantonsnummer' in gemeinden.columns:
            # kantonsnummer may be int, float, or string — normalise before compare
            ktn = gemeinden['kantonsnummer']
            print(f"  kantonsnummer dtype: {ktn.dtype}, sample: {ktn.dropna().head(3).tolist()}")
            gemeinden = gemeinden[
                pd.to_numeric(ktn, errors='coerce').fillna(-1).astype(int) == 1
            ].copy()
            print(f"  Gemeinden in Zürich canton: {len(gemeinden)}")

        gemeinden['x'] = gemeinden.geometry.centroid.x
        gemeinden['y'] = gemeinden.geometry.centroid.y
        gem_lookup = gemeinden.set_index('BFS_NR')[['x', 'y']].to_dict(orient='index')
        print(f"  Gemeinde lookup built: {len(gem_lookup)} entries")

        od_zh['WOHNGEMEINDE']    = od_zh['WOHNGEMEINDE'].astype(str).str.zfill(4)
        od_zh['ARBEITSGEMEINDE'] = od_zh['ARBEITSGEMEINDE'].astype(str).str.zfill(4)
        od_zh['x_wohn']   = od_zh['WOHNGEMEINDE'].map(lambda g: gem_lookup.get(g, {}).get('x'))
        od_zh['y_wohn']   = od_zh['WOHNGEMEINDE'].map(lambda g: gem_lookup.get(g, {}).get('y'))
        od_zh['x_arbeit'] = od_zh['ARBEITSGEMEINDE'].map(lambda g: gem_lookup.get(g, {}).get('x'))
        od_zh['y_arbeit'] = od_zh['ARBEITSGEMEINDE'].map(lambda g: gem_lookup.get(g, {}).get('y'))
        matched = od_zh[['x_wohn','y_wohn','x_arbeit','y_arbeit']].notna().all(axis=1).sum()
        print(f"  OD pairs with both centroids matched: {matched} / {len(od_zh)}")

        if matched == 0:
            print("  WARNING: No centroids matched — check BFS ID format in CSV vs gpkg")
            print(f"  Sample WOHNGEMEINDE values: {od_zh['WOHNGEMEINDE'].head(5).tolist()}")
            print(f"  Sample BFS_NR in lookup:    {list(gem_lookup.keys())[:5]}")
            od_zh['dist_km'] = np.nan
        else:
            od_zh['dist_km'] = np.sqrt(
                (pd.to_numeric(od_zh['x_arbeit'], errors='coerce') -
                 pd.to_numeric(od_zh['x_wohn'],   errors='coerce'))**2 +
                (pd.to_numeric(od_zh['y_arbeit'], errors='coerce') -
                 pd.to_numeric(od_zh['y_wohn'],   errors='coerce'))**2
            ) / 1000
            print(f"  Gemeinde centroids joined — dist range: "
                  f"{od_zh['dist_km'].min():.1f}–{od_zh['dist_km'].max():.1f} km")
    else:
        print(f"  Warning: {gem_path} not found — skipping coord join & distance filter")
        print(f"  Download Gemeindegrenzen from: https://data.geo.admin.ch")
        od_zh['dist_km'] = np.nan

    if od_zh['dist_km'].notna().any():
        before  = len(od_zh)
        od_zh   = od_zh[od_zh['dist_km'].isna() | (od_zh['dist_km'] <= max_distance_km)
                        ].copy().reset_index(drop=True)
        print(f"  After distance filter (<= {max_distance_km} km): "
              f"{len(od_zh)} pairs (dropped {before - len(od_zh)})")

    od_zh['commuters_total']   = od_zh['ERWERBSTAETIGE']
    od_zh['commuters_cycling'] = (od_zh['commuters_total'] * cycling_mode_share).round().astype(int)
    print(f"  Cycling commuters ({cycling_mode_share*100:.0f}% mode share): "
          f"{od_zh['commuters_cycling'].sum():,}")

    od_zh.to_csv(f'{output_dir}/od_matrix_zh.csv', index=False)
    od_zh[od_zh['commuters_cycling'] > 0].to_csv(
        f'{output_dir}/od_matrix_zh_cycling.csv', index=False)
    print(f"  Saved → {output_dir}/od_matrix_zh.csv")
    print(f"  Saved → {output_dir}/od_matrix_zh_cycling.csv")

    return od_zh


# ── Load OD matrix via import_pendler_matrix ─────────────────────────────
PATH_OD_CSV = 'data/OD/pendler_matrix.csv'

try:
    od_zh = import_pendler_matrix(
        path_csv=PATH_OD_CSV,
        canton_filter='ZH',
        cycling_mode_share=0.05,
        max_distance_km=15.0,
        output_dir='data/OD',
    )
    od_has_coords = all(c in od_zh.columns for c in ['x_wohn', 'y_wohn', 'x_arbeit', 'y_arbeit'])
except FileNotFoundError as e:
    print(f"  {e}")
    print("  Skipping plot 07 — download the BFS CSV first.")
    od_zh = None
    od_has_coords = False

if od_zh is not None:
    PATH_ACCESS_PTS = r"data/Network/processed/access_points_corridor.gpkg"
    access_pts = None
    if os.path.exists(PATH_ACCESS_PTS):
        access_pts = gpd.read_file(PATH_ACCESS_PTS)
        if access_pts.crs and access_pts.crs.to_epsg() != 2056:
            access_pts = access_pts.to_crs(CRS)

    od_cmap = cm.Blues

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax in axes:
        ax.set_xlim(E_MIN - 1000, E_MAX + 1000)
        ax.set_ylim(N_MIN - 1000, N_MAX + 1000)
        ax.set_aspect("equal")
        add_basemap(ax)

    # ── LEFT: total commuters choropleth by residence Gemeinde ────────────
    ax_choro = axes[0]

    if od_has_coords:
        # Aggregate total cycling commuters per residence Gemeinde
        wohn_agg = (
            od_zh.dropna(subset=['x_wohn', 'y_wohn'])
            .groupby(['WOHNGEMEINDE', 'x_wohn', 'y_wohn'])['commuters_cycling']
            .sum().reset_index()
        )
        wohn_gdf = gpd.GeoDataFrame(
            wohn_agg,
            geometry=gpd.points_from_xy(wohn_agg['x_wohn'], wohn_agg['y_wohn']),
            crs=CRS
        )
        wohn_corr = wohn_gdf.clip(CORRIDOR_BOX)

        if len(wohn_corr) > 0:
            norm = mcolors.Normalize(
                vmin=wohn_corr['commuters_cycling'].quantile(0.05),
                vmax=wohn_corr['commuters_cycling'].quantile(0.95)
            )
            wohn_corr.plot(
                column='commuters_cycling', ax=ax_choro,
                cmap='YlOrRd', norm=norm,
                markersize=60, alpha=0.85, zorder=4, legend=False
            )
            sm = cm.ScalarMappable(cmap='YlOrRd', norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax_choro, fraction=0.035, pad=0.02)
            cbar.set_label("Estimated cycling commuters\n(residence Gemeinde)", fontsize=9)
            cbar.ax.tick_params(labelsize=8)
            print(f"  Residence Gemeinden in corridor: {len(wohn_corr)}")
        else:
            print("  No residence Gemeinden fall within corridor — check coordinates")
    else:
        ax_choro.text(
            (E_MIN + E_MAX) / 2, (N_MIN + N_MAX) / 2,
            "Gemeinde centroids not available\n(gemeinden_centroid.gpkg missing)",
            ha='center', va='center', fontsize=10, color='#888',
            transform=ax_choro.transData
        )

    if access_pts is not None:
        access_pts.plot(ax=ax_choro, color="#185FA5", markersize=10,
                        marker="o", alpha=0.8, zorder=5)
    add_corridor(ax_choro, lw=1.8, alpha=0.6)
    ax_choro.set_title(
        f"Cycling commuters by residence Gemeinde\n"
        f"({od_zh['commuters_cycling'].sum():,} total  ·  5% mode share  ·  ≤15 km)",
        fontsize=11, fontweight="bold", color="#215CAF", loc="left"
    )
    ax_choro.set_xlabel("Easting (LV95)", fontsize=9)
    ax_choro.set_ylabel("Northing (LV95)", fontsize=9)

    # ── RIGHT: desire lines — top OD flows ───────────────────────────────
    ax_od = axes[1]

    if od_has_coords:
        od_plot = (
            od_zh.dropna(subset=['x_wohn', 'y_wohn', 'x_arbeit', 'y_arbeit'])
            [od_zh['commuters_cycling'] > 0]
            .copy()
        )
        # Clip to flows with at least one endpoint in corridor
        in_corr_mask = (
            od_plot['x_wohn'].between(E_MIN, E_MAX) & od_plot['y_wohn'].between(N_MIN, N_MAX)
        ) | (
            od_plot['x_arbeit'].between(E_MIN, E_MAX) & od_plot['y_arbeit'].between(N_MIN, N_MAX)
        )
        od_plot = od_plot[in_corr_mask].nlargest(80, 'commuters_cycling').reset_index(drop=True)
        print(f"  Drawing {len(od_plot)} desire lines")

        max_w = od_plot['commuters_cycling'].max() if len(od_plot) else 1

        for _, row in od_plot.iterrows():
            xi, yi = row['x_wohn'],   row['y_wohn']
            xj, yj = row['x_arbeit'], row['y_arbeit']
            w      = row['commuters_cycling']
            mx = (xi + xj) / 2 + (yj - yi) * 0.08
            my = (yi + yj) / 2 - (xj - xi) * 0.08
            t  = np.linspace(0, 1, 30)
            ax_od.plot(
                (1 - t)**2 * xi + 2*(1-t)*t * mx + t**2 * xj,
                (1 - t)**2 * yi + 2*(1-t)*t * my + t**2 * yj,
                color=od_cmap(0.25 + 0.65 * w / max_w),
                linewidth=0.3 + 2.5 * w / max_w,
                alpha=0.15 + 0.60 * w / max_w,
                zorder=3, solid_capstyle='round'
            )

        # Gemeinde centroids as dots
        wohn_xy  = od_plot[['x_wohn',   'y_wohn']].rename(columns={'x_wohn': 'x',   'y_wohn': 'y'})
        arbeit_xy = od_plot[['x_arbeit', 'y_arbeit']].rename(columns={'x_arbeit': 'x', 'y_arbeit': 'y'})
        all_pts  = pd.concat([wohn_xy, arbeit_xy]).drop_duplicates()
        ax_od.scatter(all_pts['x'], all_pts['y'], s=14, color="#215CAF",
                      alpha=0.6, zorder=5, linewidths=0)
    else:
        ax_od.text(
            (E_MIN + E_MAX) / 2, (N_MIN + N_MAX) / 2,
            "Gemeinde centroids not available",
            ha='center', va='center', fontsize=10, color='#888'
        )

    if access_pts is not None:
        access_pts.plot(ax=ax_od, color="#185FA5", markersize=12,
                        marker="o", alpha=0.85, zorder=6)
    add_corridor(ax_od, lw=1.8, alpha=0.6)

    n_pts = len(access_pts) if access_pts is not None else 0
    ax_od.legend(handles=[
        mlines.Line2D([], [], color=od_cmap(0.75), linewidth=2.5,
                      label="Top 80 OD flows (width ∝ cycling demand)"),
        mlines.Line2D([], [], color="#215CAF", marker="o", linewidth=0,
                      markersize=6, label="Gemeinde centroid"),
        mlines.Line2D([], [], color="#185FA5", marker="o", linewidth=0,
                      markersize=7, label=f"Network access points ({n_pts})"),
    ], loc="lower right", fontsize=9, framealpha=0.93, edgecolor="#CCC", fancybox=False)
    ax_od.set_title(
        f"Cycling OD desire lines — BFS Pendlermatrix\n"
        f"Canton ZH  ·  ≤15 km  ·  top 80 flows shown",
        fontsize=11, fontweight="bold", color="#185FA5", loc="left"
    )
    ax_od.set_xlabel("Easting (LV95)", fontsize=9)

    fig.suptitle("07 — Cycling commuter flows by Gemeinde  |  infraScanCycle",
                 fontsize=13, fontweight="bold", color="#215CAF", y=1.01)
    plt.tight_layout()
    plt.savefig("plots/07_od_communes.png", dpi=180, bbox_inches="tight")
    print("Saved → plots/07_od_communes.png")
    plt.show()

print("\nAll plots done.")
