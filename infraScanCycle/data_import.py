import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import math
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point, MultiPoint, shape, box, Polygon
from shapely.ops import split, snap, linemerge, unary_union
from rasterio import crs

from rasterio.features import shapes, rasterize
from geopandas.tools import sjoin
from plots import *
import requests
import zipfile
import glob
import numpy as np
import rasterio
from rasterio.transform import from_origin

from shapely.validation import make_valid



from collections import Counter
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D



    ##################################################################################
    # TODO: REVISE RASTERIZATION FOR SCENARIOS (import_data)
    # 1. MAINTAIN (Lines 34–68): The STATENT (employment) and STATPOP (population)
    #    imports remain critical. They are the primary drivers of cycling demand
    #    and potential origin-destination (OD) flows.
    # 2. ADD: Integrate a "Safety" raster layer. If accident hotspots or historical
    #    cycling safety data are available, import them here to create a "penalty"
    #    layer.
    # 3. LOGIC: This allows the scoring module to prioritize routes that avoid
    #    dangerous intersections or roads with high accident frequencies.
    ##################################################################################
def import_data(limits):
    """
    Reads the required data for the analysis. This includes employment, population and land use data.
    The data are given in different format, but generally treated in tabular form using pandas or geopandas
    DataFrames.
    As the structure of the individual raw data differ from timestamp to timestamp, they data is manipulated such
    that get an overall structure. The resulting datas are then stored as shapefiles
    :param limits: spatial limits required for further analysis given by the extent of the voronoi polygons of the access points
    :return:
    """

    # Read the CSV file into a Pandas DataFrame #TODO: for scenarios
    # Arealstatistik - 1985, 1997, 2009, 2018
    #areal_stat = pd.read_csv(r'data/landuse_landcover/landcover/ag-b-00.03-37-area-csv.csv', sep=";")
    #areal_stat = areal_stat.drop(areal_stat.columns[[3,4,5,6,7,8,9,10,11,12,13,14,15,24,25,26,27,28,29,30,31,32,33,34,35]], axis=1)
    #areal_stat = areal_stat[["E", "N", "AS18_17", "AS18_4", "LU18_10", "LU18_4"]]
    #print(areal_stat.head(50).to_string())
    # AS85_17 17 Klassen gemäss Standardnomenklatur der Arealstatistik 1979/85
    # AS85_4  4 Hauptbereiche gemäss Standardnomenklatur der Arealstatistik 1979/85
    # LU85_10 10 Klassen der Bodennutzung der Arealstatistik 1979/85
    # LU85_4  4 Hauptbereiche der Bodennutzung der Arealstatistik 1979/85


    betriebzaehlung20 = pd.read_csv(r"data/independent_variable/statent/ag-b-00.03-22-STATENT2020/STATENT_2020.csv", sep=";")
    # BXXS2: Arbeitsstätte Sektor 2, BxxVZAS2: Vollzeitäquivalent Sektor 2
    betriebzaehlung20 = betriebzaehlung20[["B08VZAT", "E_KOORD", "N_KOORD"]].rename(
        {'B08VZAT': 'empl20'}, axis=1)
    empl20_ch = fill_raster_dataframe(betriebzaehlung20)
    csv_to_tiff(empl20_ch, attribute="empl20", path=r"data/independent_variable/processed/raw/empl20_ch.tif")

    empl20 = empl20_ch[(empl20_ch["E_COORD"] >= limits[0]) & (empl20_ch["E_COORD"] <= limits[2] - 100) & (empl20_ch["N_COORD"] >= limits[1]) & (empl20_ch["N_COORD"] <= limits[3] - 100)]
    csv_to_tiff(empl20, attribute="empl20", path=r"data/independent_variable/processed/raw/empl20.tif")

    population20 = pd.read_csv(r"data/independent_variable/statpop/ag-b-00.03-vz2020statpop/STATPOP2020.csv", sep=";")
    population20 = population20[["B20BTOT", "E_KOORD", "N_KOORD"]].rename({"B20BTOT": "pop20"}, axis=1)
    pop20_ch = fill_raster_dataframe(population20)
    csv_to_tiff(pop20_ch, attribute="pop20", path=r"data/independent_variable/processed/raw/pop20_ch.tif")
    pop20 = pop20_ch[
        (pop20_ch["E_COORD"] >= limits[0]) & (pop20_ch["E_COORD"] <= limits[2] - 100) & (pop20_ch["N_COORD"] >= limits[1]) & (
                    pop20_ch["N_COORD"] <= limits[3] - 100)]
    csv_to_tiff(pop20, attribute="pop20", path=r"data/independent_variable/processed/raw/pop20.tif")

    # Store the restructured dataset as csv file
    #population20.to_csv(r"data/temp/pop_filtered.csv")
    #betriebzaehlung20.to_csv(r"data/temp/empl_filtered.csv")
    return

##################################################################################
    # TODO: NEW FUNCTION REQUIRED - ELEVATION & SLOPE ANALYSIS
    # 1. LOGIC: While a 5% incline is negligible for cars, it is a "dealbreaker" for cyclists.
    # 2. ADD: Integrate a function to join network edges with a Digital Elevation Model (DEM).
    # 3. CALCULATION: Sample LineString geometries at start, end, and midpoints to find gradient:
    #    Slope (%) = (ΔElevation / Distance) * 100
    # 4. NOTE: A realistic cycle network scan must penalize uphill routes within the
    #    routing algorithm to reflect actual user behavior/effort.
    ##################################################################################


def fill_raster_dataframe(df, rastersize=100):
    # Determine the minX, maxX, minY, maxY for consistent coverage
    try:
        df = df.rename(columns={"E_KOORD" : "E_COORD"})
        df = df.rename(columns={"N_KOORD" : "N_COORD"})
    except:
        print("")
    minX, maxX = df['E_COORD'].min(), df['E_COORD'].max()
    minX, maxX = round(math.floor(minX), -2), round(math.ceil(maxX), -2)
    minY, maxY = df['N_COORD'].min(), df['N_COORD'].max()
    minY, maxY = round(math.floor(minY), -2), round(math.ceil(maxY), -2)

    # Create a regular grid within the specified bounds
    x_grid = np.arange(minX, maxX, rastersize)
    y_grid = np.arange(minY, maxY, rastersize)

    # Create a new DataFrame with all combinations of X and Y
    new_x, new_y = np.meshgrid(x_grid, y_grid)
    new_data = pd.DataFrame({'E_COORD': new_x.ravel(), 'N_COORD': new_y.ravel()})

    # Merge the new DataFrame with the existing data and fill missing values with NaN
    merged_data = pd.merge(new_data, df, on=['E_COORD', 'N_COORD'], how='left')

    return merged_data


def csv_to_tiff(data_table, attribute, path, rastersize = 100):
    # Define the geospatial attributes
    crs_value = "epsg:2056"  # Define your desired CRS
    width = len(data_table['E_COORD'].unique())  # Match width to the number of unique X coordinates
    height = len(data_table['N_COORD'].unique())  # Match height to the number of unique Y coordinates

    x_min = min(data_table["E_COORD"])
    x_min = round(math.floor(x_min), -2)
    x_max = max(data_table["E_COORD"])
    x_max = round(math.ceil(x_max), -2)
    y_min = min(data_table["N_COORD"])
    y_min = round(math.floor(y_min), -2)
    y_max = max(data_table["N_COORD"])
    y_max = round(math.ceil(y_max), -2)

    #width = int((x_max - x_min) / rastersize)
    #height = int((y_max - y_min) / rastersize)
    #width = int(width)
    #height = int(height)
    transform = from_origin(x_min, y_max+100, rastersize, rastersize)

    #print(data_table[attribute].values.shape)
    #print(width, "   -   ", height, "   -   ", width*height)
    #sorted_df = df.sort_values(by=['Age', 'Salary'], ascending=[True, False])
    data_table_sorted = data_table.sort_values(by=["N_COORD", "E_COORD"], ascending=[False, True])
    # Create the GeoTIFF file
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=1,
                       dtype=data_table_sorted[attribute].dtype, crs=crs.CRS.from_string(crs_value), transform=transform) as dst:
        dst.write(data_table_sorted[attribute].values.reshape(height, width), 1)

    return


def import_locations():
    """
    This functions converts a csv files of location with coordinate to a geopandas DataFrame
    :return: GeoPandas DataFrame containing the locations as points
    """
    # Read csv file into pandas DataFrame
    df_cities = pd.read_csv(r"data/manually_gathered_data/City_map.csv", sep=";")

    # Convert single values into coordinates of geopandas DataFrame and initialize the coordinate reference system
    gdf_cities = gpd.GeoDataFrame(df_cities, geometry=gpd.points_from_xy(df_cities["x"], df_cities["y"]),
                                  crs="epsg:2056")

    gdf_cities.crs = "epsg:2056"
    gdf_cities.to_file('data/manually_gathered_data/cities.shp')
    return

def get_lake_data():
    gdf = gpd.read_file(r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp")
    gdf = gdf[gdf["GEWAESSERN"].isin(["Zürichsee", "Greifensee", "Pfäffikersee"])]
    # Set the CRS on the object first, then save without the crs argument
    gdf.crs = "epsg:2056"
    out = 'data/landuse_landcover/processed/lake_data_zh.gpkg'
    if os.path.exists(out):
        os.remove(out)
    gdf.to_file(out)
    return


def polygon_from_points(bounds=None, e_min=None, e_max=None, n_min=None, n_max=None, margin=0):
    """
    This function returns a square as polygon
    :param bounds: all limits of a polygon given as one element
    :param e_min: single limit values for polygon (same for e_max, n_min, n_max)
    :param margin: define if polygon should be bigger than the limits feede in
    :return:
    """
    if isinstance(bounds, np.ndarray):
        e_min, n_min, e_max, n_max = bounds
    if e_min is not None and e_max is not None and n_min is not None and n_max is not None:
        print("")
    else:
        print("No suitable coords for polygon")

    return Polygon([(e_min - margin, n_min - margin), (e_max + margin, n_min - margin), (e_max + margin, n_max + margin),
                 (e_min - margin, n_max + margin)])






def reformat_network():

    print("\nreformat_network: start\n")

    # --- config -----------------------------------------------------------
    KEEP_ROUTENTYP          = {'Velobahn', 'Hauptverbindung', 'Nebenverbindung'}
    DEVELOPMENT_PLANUNGSTYP = {'geplant', 'Variante'}
    # OGD network shapefile — source of VERBINDUNG / RW_KEY_NR (lost during momepy conversion)
    RAW_NETZ_PATH           = 'data/raw/ALLTAG/OGD_VELO_ALLTAG_NETZ_L_M.shp'
    # Schwachstellen OGD shapefile (408.4) — has NUMMER per segment
    SCHWACHSTELLEN_PATH     = 'data/raw/SCHWACHSTELLEN/TBA_VNP_SCHWACHSTELLEN_L.shp'
    SCHWACHSTELLEN_BUF_M    = 5
    COORD_ROUND             = 2   # 0.01 mm — eliminates float noise without costly sjoin
    SNAP_TOL                = 1.0

    # ------------------------------------------------------------------
    # 1. LOAD + immediate column drop (saves memory before any heavy ops)
    # ------------------------------------------------------------------
    edges_path = 'data/Network/processed/edges.gpkg'

    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"Missing: {edges_path} — run import_network_GIS_ALLTAG() first")

    edges_gdf = gpd.read_file(edges_path)

    if edges_gdf.crs.to_epsg() != 2056:
        edges_gdf = edges_gdf.to_crs("EPSG:2056")

    routentyp_col   = next((c for c in edges_gdf.columns if c.upper().startswith('ROUTENTYP')),   None)
    planungstyp_col = next((c for c in edges_gdf.columns if c.upper().startswith('PLANUNGSTY')), None)

    # ------------------------------------------------------------------
    # 1b. RE-ATTACH route name and route number from raw OGD shapefile.
    #     import_network_GIS_ALLTAG passes through momepy which drops all
    #     columns except ROUTENTYP and PLANUNGSTY.  We recover VERBINDUNG
    #     (route name, e.g. "Zürich - Dübendorf") and RW_KEY_NR (route
    #     number, e.g. "04_046a") via a nearest-edge spatial join.
    # ------------------------------------------------------------------
    edges_gdf['verbindung'] = ''
    edges_gdf['rw_key_nr']  = ''
    if os.path.exists(RAW_NETZ_PATH):
        try:
            raw = gpd.read_file(RAW_NETZ_PATH)[['geometry', 'VERBINDUNG', 'RW_KEY_NR']].copy()
            if raw.crs is None:
                raw = raw.set_crs("EPSG:2056")
            elif raw.crs.to_epsg() != 2056:
                raw = raw.to_crs("EPSG:2056")
            raw = raw.rename(columns={'VERBINDUNG': 'verbindung', 'RW_KEY_NR': 'rw_key_nr'})
            joined_names = gpd.sjoin_nearest(
                edges_gdf[['geometry']].reset_index(),
                raw, how='left', max_distance=50
            ).drop_duplicates(subset='index').set_index('index')
            edges_gdf['verbindung'] = joined_names['verbindung'].reindex(edges_gdf.index).fillna('').astype(object).values
            edges_gdf['rw_key_nr']  = joined_names['rw_key_nr'].reindex(edges_gdf.index).fillna('').astype(object).values
            print(f"  Route names joined: {(edges_gdf['verbindung'] != '').sum()} edges have VERBINDUNG")
        except Exception as e:
            print(f"  Warning: could not join route names: {e}")
    else:
        print(f"  Warning: raw OGD network not found at {RAW_NETZ_PATH} — route names unavailable")

    """
    # ------------------------------------------------------------------
    # 2. FILTER: Velorouten / Hauptverbindungen / Nebenverbindungen only
    # ------------------------------------------------------------------
    if routentyp_col:
        n_before = len(edges_gdf)
        edges_gdf = edges_gdf[edges_gdf[routentyp_col].isin(KEEP_ROUTENTYP)].copy()
        print(f"  ROUTENTYP filter: {n_before} → {len(edges_gdf)} edges")
    else:
        print("  Warning: ROUTENTYP column not found — no filter applied")

    # Drop every column that isn't needed for topology + tagging + description
    essential = {'geometry', 'length_m', 'node_start', 'node_end', 'verbindung', 'rw_key_nr'}
    if routentyp_col:   essential.add(routentyp_col)
    if planungstyp_col: essential.add(planungstyp_col)
    edges_gdf = edges_gdf.drop(
        columns=[c for c in edges_gdf.columns if c not in essential],
        errors='ignore'
    )
    """
    # ------------------------------------------------------------------
    # 3. TAG Netzlücken (planned edges are developments)
    # ------------------------------------------------------------------
    if planungstyp_col:
        edges_gdf['is_development'] = (
            edges_gdf[planungstyp_col].isin(DEVELOPMENT_PLANUNGSTYP).astype(np.int8)
        )
        print(f"  Netzlücken: {edges_gdf['is_development'].sum()} edges tagged as developments")
    else:
        edges_gdf['is_development'] = np.int8(0)
        print("  Warning: PLANUNGSTYP column not found")

    # ------------------------------------------------------------------
    # 4. TAG Schwachstellen (spatial overlay) + attach NUMMER
    #    The OGD Schwachstellen shapefile (408.4) only carries NUMMER
    #    (e.g. "S04_079") — that is the official identifier for each
    #    weak segment.  We buffer each Schwachstelle by 5 m and mark
    #    every network edge that intersects as is_schwachstelle=1.
    #    The NUMMER of the nearest Schwachstelle is stored in sw_nummer.
    # ------------------------------------------------------------------
    edges_gdf['is_schwachstelle'] = np.int8(0)
    edges_gdf['sw_nummer']        = ''
    if os.path.exists(SCHWACHSTELLEN_PATH):
        try:
            sw = gpd.read_file(SCHWACHSTELLEN_PATH)[['geometry', 'NUMMER']].copy()
            if sw.crs is None:
                sw = sw.set_crs("EPSG:2056")
            elif sw.crs.to_epsg() != 2056:
                sw = sw.to_crs("EPSG:2056")
            sw_buf = sw.copy()
            sw_buf['geometry'] = sw_buf.geometry.buffer(SCHWACHSTELLEN_BUF_M)

            joined = gpd.sjoin(
                edges_gdf.reset_index()[['index', 'geometry']],
                sw_buf.reset_index(drop=True)[['geometry', 'NUMMER']],
                how='left', predicate='intersects'
            )
            # is_schwachstelle flag
            hit_idx = joined.loc[joined['index_right'].notna(), 'index'].unique()
            edges_gdf.loc[edges_gdf.index.isin(hit_idx), 'is_schwachstelle'] = np.int8(1)

            # sw_nummer: take first matched NUMMER per edge
            nummer_map = (
                joined[joined['NUMMER'].notna()]
                .drop_duplicates(subset='index')
                .set_index('index')['NUMMER']
            )
            edges_gdf['sw_nummer'] = nummer_map.reindex(edges_gdf.index).fillna('').astype(object).values

            print(f"  Schwachstellen: {edges_gdf['is_schwachstelle'].sum()} edges tagged "
                  f"({edges_gdf['sw_nummer'].ne('').sum()} with NUMMER)")
            del sw, sw_buf, joined
        except Exception as e:
            print(f"  Warning: could not load Schwachstellen: {e}")
    else:
        print(f"  Warning: Schwachstellen not found at {SCHWACHSTELLEN_PATH}")

    print(f"  Loaded {len(edges_gdf)} edges")

    # ------------------------------------------------------------------
    # 5. IDENTIFY JUNCTIONS via coordinate counter (no sjoin clustering)
    # ------------------------------------------------------------------
    def _xy(coord):
        return (round(coord[0], COORD_ROUND), round(coord[1], COORD_ROUND))

    # TODO: junction detection counts how often each rounded coordinate appears
    # as an edge endpoint.  A count >= 3 means ≥ 3 edges share that endpoint,
    # which is the correct heuristic for a topological intersection.
    # However, this counts frequencies, not actual geometric connectivity — two
    # unrelated edges that happen to round to the same coordinate will look like
    # a junction even if they belong to disconnected sub-networks.
    # After splitting, verify that nodes flagged as intersections actually have
    # >= 3 incident edges in edges_split.
    starts_xy = [_xy(g.coords[0])  for g in edges_gdf.geometry]
    ends_xy   = [_xy(g.coords[-1]) for g in edges_gdf.geometry]
    counts    = Counter(starts_xy + ends_xy)
    junc_pts    = [Point(x, y) for (x, y), n in counts.items() if n >= 3]
    through_pts = {(x, y) for (x, y), n in counts.items() if n == 2}
    endpoint_pts = {(x, y) for (x, y), n in counts.items() if n == 1}

    junctions = gpd.GeoDataFrame(geometry=junc_pts, crs="EPSG:2056")
    print(f"  {len(junctions)} junction nodes, "
          f"{len(through_pts)} through-nodes, "
          f"{len(endpoint_pts)} dead-ends")

    # ------------------------------------------------------------------
    # 6. SPLIT EDGES at hubs + junctions using STRtree (replaces MultiPoint
    #    over all edges — only queries nearby split points per edge)
    # ------------------------------------------------------------------
    all_split_pts = junctions.geometry.tolist()

    if not all_split_pts:
        edges_split = edges_gdf.copy()
    else:
        # TODO: SNAP_TOL = 1.0 m controls which junction points are considered
        # "on" an edge for splitting.  If two edges share a node but their
        # coordinates differ by > 1 m before rounding (e.g. from a misaligned
        # source shapefile), the split is skipped and they remain disconnected.
        # Inspect the gap histogram after reformat_network() to confirm 1 m
        # is appropriate for the ALLTAG dataset; increase if gaps persist.
        tree = STRtree(all_split_pts)
        new_edges = []
        for _, edge in edges_gdf.iterrows():
            nearby_idx = tree.query(edge.geometry.buffer(SNAP_TOL))
            if len(nearby_idx) == 0:
                new_edges.append(edge)
                continue
            try:
                pt_coll = MultiPoint([all_split_pts[i] for i in nearby_idx])
                snapped = snap(edge.geometry, pt_coll, SNAP_TOL)
                parts   = split(snapped, pt_coll)
                if len(parts.geoms) > 1:
                    for part in parts.geoms:
                        row = edge.copy()
                        row['geometry'] = part
                        row['length_m'] = part.length
                        new_edges.append(row)
                else:
                    new_edges.append(edge)
            except Exception as e:
                print(f"  Warning: could not split edge {edge.name}: {e}")
                new_edges.append(edge)

        edges_split = gpd.GeoDataFrame(new_edges, crs="EPSG:2056").reset_index(drop=True)

    edges_split = edges_split[edges_split.geometry.length > 0.1].copy()
    edges_split['ID_edge'] = range(len(edges_split))

    # ------------------------------------------------------------------
    # 7. BUILD NODE TABLE via coordinate dict (replaces two sjoin calls)
    # ------------------------------------------------------------------
    coord_to_id = {}
    node_rows   = []

    def _get_or_add(coord):
        key = _xy(coord)
        if key not in coord_to_id:
            nid = len(coord_to_id)
            coord_to_id[key] = nid
            node_rows.append({'ID_point': nid, 'geometry': Point(key)})
        return coord_to_id[key]

    start_ids = [_get_or_add(g.coords[0])  for g in edges_split.geometry]
    end_ids   = [_get_or_add(g.coords[-1]) for g in edges_split.geometry]

    edges_split['start'] = start_ids
    edges_split['end']   = end_ids

    nodes_gdf = gpd.GeoDataFrame(node_rows, crs="EPSG:2056")

    # ------------------------------------------------------------------
    # 8. FLAG intersections, through-nodes, dead-ends and hub destinations
    # ------------------------------------------------------------------
    junc_keys     = {(round(pt.x, COORD_ROUND), round(pt.y, COORD_ROUND)) for pt in junc_pts}
    nodes_gdf['is_intersection'] = nodes_gdf['geometry'].apply(
        lambda g: int(_xy(g.coords[0]) in junc_keys)
    ).astype(np.int8)

    nodes_gdf['is_through_point'] = nodes_gdf['geometry'].apply(
        lambda g: int(_xy(g.coords[0]) in through_pts)
    ).astype(np.int8)

    nodes_gdf['is_endpoint'] = nodes_gdf['geometry'].apply(
        lambda g: int(_xy(g.coords[0]) in endpoint_pts)
    ).astype(np.int8)

    nodes_gdf['is_destination'] = np.int8(0)

    # ------------------------------------------------------------------
    # 9. EXPORT
    # ------------------------------------------------------------------
    os.makedirs('data/Network/processed', exist_ok=True)

    # Drop uppercase columns that have a separate lowercase duplicate.
    # GPKG/SQLite is case-insensitive: VERBINDUNG + verbindung → "Error adding field".
    _lower_count = Counter(c.lower() for c in edges_split.columns)
    drop_upper = [c for c in edges_split.columns
                  if c != c.lower() and _lower_count[c.lower()] > 1]
    if drop_upper:
        edges_split = edges_split.drop(columns=drop_upper)

    # Also drop columns whose names would be laundered/truncated by GDAL
    # (e.g. 'visual_geom' carrying WKT strings that belong in a separate layer)
    for _drop in ('visual_geom',):
        if _drop in edges_split.columns:
            edges_split = edges_split.drop(columns=[_drop])

    nodes_gdf.to_file('data/Network/processed/points.gpkg', driver='GPKG')
    edges_split.to_file('data/Network/processed/edges.gpkg', driver='GPKG')

    print(f"  -> {len(edges_split)} edges, {len(nodes_gdf)} nodes")
    print(f"     {nodes_gdf['is_intersection'].sum()} intersections, "
          f"{nodes_gdf['is_through_point'].sum()} through-nodes, "
          f"{nodes_gdf['is_endpoint'].sum()} dead-ends, "
          f"{nodes_gdf['is_destination'].sum()} hub destinations")
    print(f"     {edges_split['is_development'].sum()} Netzlücken, "
          f"{edges_split['is_schwachstelle'].sum()} Schwachstellen")
    print("\nreformat_network: end\n")

    return nodes_gdf, edges_split



def plot_network_classified(nodes_gdf, edges_gdf, figsize=(14, 10)):
    fig, ax = plt.subplots(figsize=figsize)

    # --- Edges ---
    edges_gdf.plot(ax=ax, color='steelblue', linewidth=0.8, alpha=0.6, zorder=1)

    has_endpoint     = 'is_endpoint'     in nodes_gdf.columns
    has_intersection = 'is_intersection' in nodes_gdf.columns

    # --- Intersections (degree >= 3) ---
    if has_intersection:
        intersections = nodes_gdf[nodes_gdf['is_intersection'] == 1]
        non_inter     = nodes_gdf[nodes_gdf['is_intersection'] == 0]
    else:
        intersections = nodes_gdf.iloc[0:0]
        non_inter     = nodes_gdf

    # --- Dead ends (degree == 1) ---
    if has_endpoint:
        endpoints = non_inter[non_inter['is_endpoint'] == 1]
        regular   = non_inter[non_inter['is_endpoint'] == 0]
    else:
        endpoints = non_inter.iloc[0:0]
        regular   = non_inter

    if len(regular):
        ax.scatter(regular.geometry.x, regular.geometry.y,
                   s=8, color='gray', alpha=0.5, zorder=2, label=f'Through node ({len(regular)})')
    if len(intersections):
        ax.scatter(intersections.geometry.x, intersections.geometry.y,
                   s=30, color='orange', alpha=0.85, zorder=3, label=f'Intersection ({len(intersections)})')
    if len(endpoints):
        ax.scatter(endpoints.geometry.x, endpoints.geometry.y,
                   s=20, color='red', alpha=0.85, zorder=4, label=f'Dead end ({len(endpoints)})')

    ax.set_title('Cycling Network — Node Classification', fontsize=14)
    ax.set_xlabel('Easting (EPSG:2056)')
    ax.set_ylabel('Northing (EPSG:2056)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('data/Network/processed/network_plot.png', dpi=150)
    plt.show()
    print("Plot saved → data/Network/processed/network_plot.png")









def get_edge_attributes():
    """
    Assigns cycling-specific free-flow speed, comfort index, and safety risk
    to edges based on ROUTENTYP from the ALLTAG dataset.

    Empirical calibration (Bernardi et al. observed 12.5–26.5 km/h across sites):
      - Dedicated commuter cycling highways (Velobahn):  near observed max ~25 km/h
      - Main commuter connections (Hauptverbindung):     above avg, ~20 km/h
      - Secondary mixed-use paths (Nebenverbindung):     matches annual avg 15–16 km/h
      - Recreational / shared paths (Freizeitverbindung): below avg due to 5–30%
                                                           pedestrian share → 13 km/h
      - Default / unclassified:                          empirical avg 15 km/h

    ROUTENTYP → ffs (km/h) | comfort CLI | safety risk weight:
        Velobahn / Veloschnellroute     → 20 km/h | 1.00 | 1.0
        Hauptverbindung                 → 20 km/h | 0.75 | 2.0
        Nebenverbindung                 → 18 km/h | 0.50 | 3.0
        Zusätzliche Freizeitverbindung  → 18 km/h | 0.25 | 4.0
        default / unclassified          → 15 km/h | 0.40 | 5.0
    """
    ROUTENTYP_ATTRS = {
        'Velobahn':                      {'ffs': 20, 'cli': 1.00, 'risk_w': 1.0},
        'Veloschnellroute':              {'ffs': 20, 'cli': 1.00, 'risk_w': 1.0},
        'Hauptverbindung':               {'ffs': 20, 'cli': 0.75, 'risk_w': 2.0},
        'Nebenverbindung':               {'ffs': 18, 'cli': 0.50, 'risk_w': 3.0},
        'Zusätzliche Freizeitverbindung':{'ffs': 18, 'cli': 0.25, 'risk_w': 4.0},
    }
    DEFAULT_ATTRS = {'ffs': 15, 'cli': 0.40, 'risk_w': 5.0}

    edges = gpd.read_file('data/Network/processed/edges_with_attribute.gpkg')
    if edges.crs is None:
        edges = edges.set_crs("EPSG:2056")

    print(f"  Edge columns: {edges.columns.tolist()}")

    # ------------------------------------------------------------------
    # 1. ROUTENTYP → ffs, comfort index, safety risk weight
    #    Use startswith match to survive any column name truncation
    # ------------------------------------------------------------------
    routentyp_col = next((c for c in edges.columns if c.upper().startswith('ROUTENTYP')), None)

    def lookup(routentyp, key):
        return ROUTENTYP_ATTRS.get(str(routentyp).strip(), DEFAULT_ATTRS)[key]

    if routentyp_col:
        edges['ffs']    = edges[routentyp_col].apply(lambda r: lookup(r, 'ffs'))
        edges['cli']    = edges[routentyp_col].apply(lambda r: lookup(r, 'cli'))
        edges['risk_w'] = edges[routentyp_col].apply(lambda r: lookup(r, 'risk_w'))
    else:
        print("  Warning: ROUTENTYP column not found — applying defaults")
        edges['ffs']    = DEFAULT_ATTRS['ffs']
        edges['cli']    = DEFAULT_ATTRS['cli']
        edges['risk_w'] = DEFAULT_ATTRS['risk_w']

    # ------------------------------------------------------------------
    # 2. FAHRRICHTUNGSTYP → oneway flag
    #    startswith match handles truncation (FAHRRICHTU, FAHRRICHTUNG, etc.)
    # ------------------------------------------------------------------
    fahrricht_col = next((c for c in edges.columns if c.upper().startswith('FAHRRICHT')), None)

    if fahrricht_col:
        edges['oneway'] = edges[fahrricht_col].apply(
            lambda x: 1 if str(x).strip() == 'eine Richtung' else 0
        )
        print(f"  Using column '{fahrricht_col}' for oneway flag")
    else:
        print("  Warning: FAHRRICHTUNGSTYP column not found — defaulting to bidirectional")
        edges['oneway'] = 0

    # ------------------------------------------------------------------
    # 3. Travel time per edge [minutes] = length / ffs
    # ------------------------------------------------------------------
    if 'length_m' in edges.columns:
        edges['tt_min'] = (edges['length_m'] / 1000) / edges['ffs'] * 60
    else:
        edges['tt_min'] = (edges.geometry.length / 1000) / edges['ffs'] * 60

    # TODO: ID_edge is reassigned here as a sequential range over the entire
    # edges_with_attribute.gpkg file.  However, edges_corridor.gpkg is written
    # by network_in_corridor() BEFORE get_edge_attributes() runs, so the
    # corridor file does NOT yet have ID_edge.  _build_base_gdf() in
    # OSM_network.py tries to merge ffs from edges_with_attribute.gpkg by
    # ID_edge.  This only works if network_in_corridor() preserves the ID_edge
    # values from reformat_network() in its output.  Confirm the ID_edge values
    # are consistent across both files; if not, reorder steps in main.py so
    # get_edge_attributes() runs before network_in_corridor().
    edges['ID_edge'] = range(len(edges))

    edges.to_file('data/Network/processed/edges_with_attribute.gpkg', driver='GPKG')

    print(f"  -> {len(edges)} edges attributed")
    print(f"     ffs range:      {edges['ffs'].min()}–{edges['ffs'].max()} km/h")
    print(f"     cli range:      {edges['cli'].min():.2f}–{edges['cli'].max():.2f}")
    print(f"     risk_w range:   {edges['risk_w'].min():.1f}–{edges['risk_w'].max():.1f}")
    print(f"     one-way edges:  {edges['oneway'].sum()}")
    print(f"     tt_min range:   {edges['tt_min'].min():.1f}–{edges['tt_min'].max():.1f} min")

    return edges

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import geopandas as gpd

def plot_edge_attributes(edges=None):
    if edges is None:
        edges = gpd.read_file('data/Network/processed/edges_with_attribute.gpkg')

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # --- 1. Free-flow speed ---
    ax = axes[0]
    speed_colors = {25: '#2ecc71', 20: '#3498db', 18: '#9b59b6', 15: '#e74c3c'}
    for speed, color in speed_colors.items():
        subset = edges[edges['ffs'] == speed]
        if len(subset):
            subset.plot(ax=ax, color=color, linewidth=1.5, alpha=0.8,
                        label=f'{speed} km/h ({len(subset)})')
    ax.set_title('Free-flow Speed (km/h)', fontsize=12)
    ax.legend(fontsize=8)
    ax.set_aspect('equal')

    # --- 2. Comfort index (CLI) ---
    ax = axes[1]
    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = cm.RdYlGn
    for _, row in edges.iterrows():
        color = cmap(norm(row['cli']))
        gpd.GeoDataFrame([row], crs=edges.crs).plot(ax=ax, color=[color], linewidth=1.5)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Comfort index (CLI)', shrink=0.6)
    ax.set_title('Comfort Index (CLI)', fontsize=12)
    ax.set_aspect('equal')

    # --- 3. Travel time ---
    ax = axes[2]
    norm2 = mcolors.Normalize(vmin=edges['tt_min'].min(), vmax=edges['tt_min'].max())
    cmap2 = cm.Blues
    for _, row in edges.iterrows():
        color = cmap2(norm2(row['tt_min']))
        gpd.GeoDataFrame([row], crs=edges.crs).plot(ax=ax, color=[color], linewidth=1.5)
    sm2 = cm.ScalarMappable(cmap=cmap2, norm=norm2)
    sm2.set_array([])
    plt.colorbar(sm2, ax=ax, label='Travel time (min)', shrink=0.6)
    ax.set_title('Travel Time (min)', fontsize=12)
    ax.set_aspect('equal')

    plt.suptitle('Edge Attributes — Cycling Network', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('data/Network/processed/edge_attributes_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved → data/Network/processed/edge_attributes_plot.png")

def network_in_corridor(polygon, access_point_min_dist=0):
    print(f"network_in_corridor(): start")
    os.makedirs('data/Network/processed', exist_ok=True)

    # ------------------------------------------------------------------
    # 1. LOAD inputs from reformat_network() outputs
    # ------------------------------------------------------------------
    edges  = gpd.read_file('data/Network/processed/edges.gpkg')
    points = gpd.read_file('data/Network/processed/points.gpkg')

    if edges.crs is None:
        edges  = edges.set_crs("EPSG:2056")
    if points.crs is None:
        points = points.set_crs("EPSG:2056")

    if edges.crs.to_epsg() != 2056:
        edges = edges.to_crs("EPSG:2056")
    if points.crs.to_epsg() != 2056:
        points = points.to_crs("EPSG:2056")

    print(f"  Loaded {len(edges)} edges, {len(points)} nodes")
    print(f"  Node columns: {points.columns.tolist()}")

    poly_gdf  = gpd.GeoDataFrame({'geometry': [polygon]}, crs="EPSG:2056")
    poly_geom = polygon

    # ------------------------------------------------------------------
    # 2. NODES inside corridor
    # ------------------------------------------------------------------
    points_corridor = gpd.sjoin(points, poly_gdf, how='inner', predicate='within') \
                         .drop(columns=['index_right'], errors='ignore') \
                         .reset_index(drop=True)
    points_corridor.to_file('data/Network/processed/points_corridor.gpkg', driver='GPKG')
    points_corridor.drop(columns='geometry').to_csv(
        'data/Network/processed/points_corridor.csv', index=False
    )
    print(f"  Nodes in corridor → points_corridor.gpkg + .csv  ({len(points_corridor)} rows)")

    # ------------------------------------------------------------------
    # 3. ACCESS POINTS inside corridor — all corridor nodes are access points
    # ------------------------------------------------------------------
    access_points_corridor = points_corridor.copy().reset_index(drop=True)
    access_points_corridor['ID_access'] = range(len(access_points_corridor))
    access_points_corridor.to_file('data/Network/processed/access_points_corridor.gpkg', driver='GPKG')
    access_points_corridor.drop(columns='geometry').to_csv(
        'data/Network/processed/access_points_corridor.csv', index=False
    )
    print(f"  Access points in corridor → access_points_corridor.gpkg + .csv  ({len(access_points_corridor)} rows)")

    # ------------------------------------------------------------------
    # 4. EDGES strictly inside corridor
    # ------------------------------------------------------------------
    edges_corridor = gpd.sjoin(edges, poly_gdf, how='inner', predicate='within') \
                        .drop(columns=['index_right'], errors='ignore') \
                        .reset_index(drop=True)
    edges_corridor.to_file('data/Network/processed/edges_corridor.gpkg', driver='GPKG')
    print(f"  Edges in corridor → edges_corridor.gpkg  ({len(edges_corridor)} rows)")

    # ------------------------------------------------------------------
    # 5. EDGES crossing the corridor border (exactly one endpoint inside)
    # ------------------------------------------------------------------
    def one_endpoint_inside(geom, poly):
        return poly.contains(Point(geom.coords[0])) != poly.contains(Point(geom.coords[-1]))

    edges['on_border'] = edges.geometry.apply(lambda g: one_endpoint_inside(g, poly_geom))
    edges_border = edges[edges['on_border']].copy().reset_index(drop=True)
    edges_border.to_file('data/Network/processed/edges_corridor_border.gpkg', driver='GPKG')
    print(f"  Edges on border   → edges_corridor_border.gpkg  ({len(edges_border)} rows)")

    # ------------------------------------------------------------------
    # 6. FLAG nodes with corridor membership
    # ------------------------------------------------------------------
    points['within_corridor'] = points.geometry.apply(lambda g: poly_geom.contains(g))

    # on_corridor_border: node touches a border-crossing edge
    points_buf = points.copy()
    points_buf['geometry'] = points.buffer(1e-6)
    points_buf = points_buf.reset_index().rename(columns={'index': 'orig_idx'})

    border_join = gpd.sjoin(
        points_buf[['orig_idx', 'geometry']],
        edges_border[['geometry']].reset_index(drop=True),
        how='left',
        predicate='intersects'
    )
    on_border_flag = border_join.groupby('orig_idx')['index_right'].apply(
        lambda x: x.notnull().any()
    )
    points['on_corridor_border'] = on_border_flag.reindex(points.index, fill_value=False).values

    # ------------------------------------------------------------------
    # 7. FLAG edges with corridor membership
    # ------------------------------------------------------------------
    edges['within_corridor'] = edges.geometry.apply(lambda g: poly_geom.contains(g))

    # ------------------------------------------------------------------
    # 8. SAVE enriched full datasets
    # ------------------------------------------------------------------
    points.to_file('data/Network/processed/points_with_attribute.gpkg', driver='GPKG')
    points.drop(columns='geometry').to_csv(
        'data/Network/processed/points_with_attribute.csv', index=False
    )
    edges.to_file('data/Network/processed/edges_with_attribute.gpkg', driver='GPKG')
    edges.drop(columns='geometry').to_csv(
        'data/Network/processed/edges_with_attribute.csv', index=False
    )
    print(f"  Full attributed nodes/edges saved (with within_corridor + on_corridor_border flags)")

    # ------------------------------------------------------------------
    # 9. SAVE corridor points with attributes
    # ------------------------------------------------------------------
    points_corridor_attribute = points[points['within_corridor']].copy().reset_index(drop=True)
    points_corridor_attribute.to_file(
        'data/Network/processed/points_corridor_attribute.gpkg', driver='GPKG'
    )
    points_corridor_attribute.drop(columns='geometry').to_csv(
        'data/Network/processed/points_corridor_attribute.csv', index=False
    )
    print(f"  Corridor nodes with attributes → points_corridor_attribute.gpkg  ({len(points_corridor_attribute)} rows)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n  Summary:")
    print(f"    {len(points_corridor)} / {len(points)} nodes inside corridor (all are access points)")
    for flag in ('is_intersection', 'is_through_point', 'is_endpoint'):
        if flag in points_corridor.columns:
            print(f"    {points_corridor[flag].sum()} {flag.replace('is_', '')} in corridor")
    print(f"    {len(edges_corridor)} / {len(edges)} edges inside corridor")
    print(f"    {len(edges_border)} edges crossing corridor border")

    print(f"network_in_corridor(): end")
    return points_corridor, edges_corridor, edges_border




def plot_corridor_network(polygon, points_corridor, edges_corridor, edges_border, points_full=None, edges_full=None):
    fig, ax = plt.subplots(figsize=(14, 11))

    # --- Background: full network (optional, greyed out) ---
    if edges_full is not None:
        edges_full.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.5, zorder=1)
    if points_full is not None:
        points_full.plot(ax=ax, color='lightgray', markersize=2, alpha=0.4, zorder=2)

    # --- Corridor polygon ---
    import geopandas as gpd
    poly_gdf = gpd.GeoDataFrame({'geometry': [polygon]}, crs="EPSG:2056")
    poly_gdf.boundary.plot(ax=ax, color='black', linewidth=1.5, linestyle='--', zorder=3)
    poly_gdf.plot(ax=ax, color='lightyellow', alpha=0.3, zorder=2)

    # --- Edges inside corridor ---
    if len(edges_corridor):
        edges_corridor.plot(ax=ax, color='steelblue', linewidth=1.2, alpha=0.8, zorder=4)

    # --- Border-crossing edges ---
    if len(edges_border):
        edges_border.plot(ax=ax, color='darkorange', linewidth=1.5, linestyle=':', alpha=0.9, zorder=5)

    # --- Nodes inside corridor: intersections vs endpoints vs through ---
    if 'is_intersection' in points_corridor.columns and 'is_endpoint' in points_corridor.columns:
        intersections = points_corridor[points_corridor['is_intersection'] == 1]
        endpoints = points_corridor[points_corridor['is_endpoint'] == 1]
        through = points_corridor[
            (points_corridor['is_intersection'] == 0) &
            (points_corridor['is_endpoint'] == 0)
            ]
        if len(through):
            ax.scatter(through.geometry.x, through.geometry.y,
                       s=10, color='gray', alpha=0.6, zorder=6)
        if len(intersections):
            ax.scatter(intersections.geometry.x, intersections.geometry.y,
                       s=40, color='orange', alpha=0.9, zorder=7)
        if len(endpoints):
            ax.scatter(endpoints.geometry.x, endpoints.geometry.y,
                       s=25, color='red', alpha=0.9, zorder=7)
    else:
        points_corridor.plot(ax=ax, color='steelblue', markersize=10, zorder=6)

    # --- Legend ---
    legend_elements = [
        Line2D([0], [0], color='steelblue', linewidth=1.5, label=f'Edges in corridor ({len(edges_corridor)})'),
        Line2D([0], [0], color='darkorange', linewidth=1.5, linestyle=':', label=f'Border-crossing edges ({len(edges_border)})'),
        Line2D([0], [0], color='black', linewidth=1.5, linestyle='--', label='Corridor boundary'),
        mpatches.Patch(facecolor='orange', label='Intersections'),
        mpatches.Patch(facecolor='red',    label='Dead ends'),
        mpatches.Patch(facecolor='gray',   label='Through-nodes'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    ax.set_title('Cycling Network — Corridor Clip', fontsize=14)
    ax.set_xlabel('Easting (EPSG:2056)')
    ax.set_ylabel('Northing (EPSG:2056)')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('data/Network/processed/corridor_plot.png', dpi=150)
    plt.show()
    print("Plot saved → data/Network/processed/corridor_plot.png")





def only_links_to_corridor():
    # 1. Load data
    all_links = gpd.read_file(r"data/Network/processed/new_links.gpkg")
    all_access_points = gpd.read_file(r"data/Network/processed/points_corridor_attribute.gpkg")

    # 2. Determine available columns to prevent KeyError
    # We always need ID_point for the join
    cols_to_use = ["ID_point"]

    if "cor_1" in all_access_points.columns:
        # Filter for corridor points if attribute exists
        access_corridor = all_access_points[all_access_points["cor_1"] == "1"].copy()
        cols_to_use.append("cor_1")
    else:
        print("Warning: 'cor_1' column not found. Skipping attribute filter and using all points.")
        access_corridor = all_access_points.copy()

    # 3. Join links to corridor points
    # We join all_links.ID_current to access_corridor.ID_point
    all_links["ID_current"] = all_links["ID_current"].astype(str)
    access_corridor["ID_point"] = access_corridor["ID_point"].astype(str)

    links_corridor = all_links.merge(
        right=access_corridor[cols_to_use],
        left_on="ID_current",
        right_on="ID_point"
    )

    print(f"Links connected to points within the corridor: {links_corridor.shape[0]} of {all_links.shape[0]}")

    # 4. Save processed links
    if "ID_current" in links_corridor.columns:
        links_corridor = links_corridor.drop(columns=["ID_current"])

    links_corridor.to_file(r"data/Network/processed/developments_to_corridor_attribute.gpkg")

    # 5. Process generated access points
    generated_points = gpd.read_file(r"data/Network/processed/generated_nodes.gpkg")

    # Ensure ID_new is also treated as string for the merge
    generated_points["ID_new"] = generated_points["ID_new"].astype(str)
    links_corridor["ID_new"] = links_corridor["ID_new"].astype(str)

    # Merge to filter only generated points that successfully found a link
    temp = generated_points.merge(right=links_corridor, on="ID_new")

    # geometry_x is the location of the NEW generated point
    generated_points_corridor = temp[["ID_new", "geometry_x", "ID_point"]].copy()

    # Rename geometry column to standard 'geometry'
    generated_points_corridor = generated_points_corridor.rename(columns={"geometry_x": "geometry"})

    # Convert back to a proper GeoDataFrame
    gdf_nodes_out = gpd.GeoDataFrame(
        generated_points_corridor,
        geometry="geometry",
        crs=generated_points.crs
    )

    gdf_nodes_out.to_file(r"data/Network/processed/generated_nodes_connecting_corridor.gpkg")
    print("Infrastructure generation step complete.")

def get_protected_area(limits):
    bln = gpd.read_file(r"data/landuse_landcover/Schutzzonen/BLN/N2017_Revision_landschaftnaturdenkmal_20170727_20221110.shp")
    wildkorridore = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Wildtierkorridore/Wildtierkorridore.gpkg")
    trockenweiden = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Trockenwiesen/TWW_LV95/trockenwiesenweiden.shp")
    trockenlandschaften = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Moorlandschaft/Moorlandschaft_LV95/moorlandschaft.shp")
    flachmoore = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Flachmoore/Flachmoor_LV95/flachmoor_20210701.shp")
    hochmoore = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Hochmoor/Hochmoor_LV95/hochmoor.shp")
    bundesinventar_auen = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Bundesinventar_auen/N2017_Revision_Auengebiete_20171101_20221122.shp")
    ramsar = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Ramsar/Ramsar_LV95/ra.shp")
    naturschutz = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Inventar_der_Natur-_und_Landsch...uberkommunaler_Bedeutung_-OGD/INV80_NATURSCHUTZOBJEKTE_F.shp")
    wald = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Waldareal_-OGD/WALD_WALDAREAL_F.shp")
    fruchtfolgeflaeche = gpd.read_file(r"data/landuse_landcover/Schutzzonen/Fruchtfolgeflachen_-OGD/FFF_F.shp")

    # Cycling: BLN, Moores, Auen, Ramsar and Naturschutz remain fully protected (legally binding).
    # Wildtierkorridore removed from fully_protected: a narrow cycle path has a small footprint
    # and may be permissible — treat as partly protected instead.
    gdf_fully_protected = [
        bln,
        flachmoore,
        hochmoore,
        bundesinventar_auen,
        ramsar,
        naturschutz
    ]
    names_fully_protected = [
        'bln',
        'flachmoore',
        'hochmoore',
        'bundesinventar_auen',
        'ramsar',
        'naturschutz'
    ]

    # Cycling: Wald removed from partly_protected — cycle paths through forests are common and
    # legally permitted. Wildtierkorridore added here as a soft constraint.
    gdf_partly_protected = [
        wildkorridore,
        fruchtfolgeflaeche,
        trockenweiden,
        trockenlandschaften
    ]
    names_partly_protected = [
        "wildkorridore",
        "fruchtfolgeflaeche",
        "trockenweiden",
        "trockenlandschaften"
    ]

    multiple_shp_to_one(gdf_fully_protected, names_fully_protected, "fully_protected", limits)
    multiple_shp_to_one(gdf_partly_protected, names_partly_protected, "partly_protected", limits)

    return


def multiple_shp_to_one(gdf_list, names_list, path, limits):
    # Initialize an empty list to store the dissolved geometries
    dissolved_geometries = []

    for gdf in gdf_list:
        # Dissolve all features within the GeoDataFrame into a single geometry
        gdf = gdf.copy()
        gdf['geometry'] = gdf.geometry.apply(lambda g: make_valid(g) if g is not None else g)
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)
        dissolved = gdf.dissolve()
        # Append the dissolved geometry to the list

        valid_geoms = dissolved.geometry.apply(make_valid)
        dissolved_geometries.append(valid_geoms.unary_union)

    # Now create a new DataFrame with the dissolved geometries
    # Use the 'unary_union' attribute to ensure that the geometry is merged into one
    combined_gdf = gpd.GeoDataFrame({'geometry': dissolved_geometries})

    combined_gdf["name"] = names_list
    combined_gdf.crs = "epsg:2056"

    combined_gdf.to_file(fr"data/landuse_landcover/Schutzzonen/{path}.gpkg", driver="GPKG")

    # Create a bounding box as a shapely object
    frame_box = box(limits[0], limits[1], limits[2], limits[3])

    # Clip the GeoDataFrame using the bounding box
    combined_gdf_frame = gpd.clip(combined_gdf, frame_box)
    #combined_gdf_frame.to_file(fr"data/landuse_landcover/Schutzzonen/{path}_frame.gpkg")
    combined_gdf_frame.to_file(fr"data/landuse_landcover/processed/{path}_frame.gpkg")


def all_protected_area_to_raster(suffix=""):
    # Load your shapefile with geopandas
    #shp_file = r"data/landuse_landcover/processed/fully_protected.gpkg"
    shp_file = r"data/landuse_landcover/Schutzzonen/fully_protected.gpkg" #correction by Arnor
    shapes = gpd.read_file(shp_file)

    # Load your raster file with rasterio
    tif_file = r"data/landuse_landcover/processed/protected_area.tif"

    try:
        # Load the CSV file with the coordinates
        csv_file = r"data/manually_gathered_data/cell_to_remove.csv"
        coords_df = pd.read_csv(csv_file, sep=";")
        print(coords_df.head().to_string())
    except:
        pass

    # Open the existing TIFF file
    with rasterio.open(tif_file) as src:
        meta = src.meta.copy()
        # Read the existing data
        existing_data = src.read(1)
        nodata_value = src.nodata or -9999
        meta.update(nodata=nodata_value)

        # Rasterize the shapes to the same dimension as the raster data
        burned = rasterize(
            [(shape, 1) for shape in shapes.geometry],
            out_shape=src.shape,
            transform=src.transform,
            fill=0,  # the default fill value
            all_touched=True  # mark all cells touched by polygons
        )

        # Merge the burned raster with the existing data
        # Where the burned data is 1, we update the existing data
        updated_data = np.where(burned == 1, 39, existing_data)

        try:
            # Process coordinates from the CSV file
            for _, row in coords_df.iterrows():
                row_x, row_y = row['x'], row['y']
                row_col, row_row = src.index(row_x, row_y)
                updated_data[row_col, row_row] = -9999
        except:
            print("No cell to remove")

        # Write the updated data to a new raster file
        with rasterio.open(fr'data/landuse_landcover/processed/zone_no_infra/protected_area_{suffix}.tif', 'w', **meta) as dst:
            dst.write(updated_data, 1)


def landuse(limits):
    # Read the CSV file into a Pandas DataFrame
    # Arealstatistik - 1985, 1997, 2009, 2018
    areal_stat = pd.read_csv(r'data/landuse_landcover/landcover/ag-b-00.03-37-area-csv.csv', sep=";")
    if limits:
        areal_stat = areal_stat[(areal_stat["E_COORD"] >= limits[0]) &
                                (areal_stat["E_COORD"] <= limits[2]) &
                                (areal_stat["N_COORD"] >= limits[1]) &
                                (areal_stat["N_COORD"] <= limits[3])]

    areal_stat = areal_stat[["E_COORD", "N_COORD", "AS18_27"]]

    # Cycling: much smaller footprint than a highway — fewer land-use categories are true barriers.
    # Removed: 4 (Parks/Grünanlagen), 5 (Strassen/Verkehr), 8 (Sportanlagen), 10 (Camping),
    #          17 (Obstgärten), 18 (Reben), 23 (Wald), 27 (Gewässer).
    # 4 removed: cycle paths through parks are common and desirable.
    # 5 removed: cycle paths almost always run alongside existing roads — blocking cat. 5 would
    #            prevent nearly all realistic cycle route generation.
    # Wald (23): cycle paths through forests are standard practice.
    # Gewässer (27): cycle bridges are cheap and small; handle via elevation model.
    # Kept: residential/industrial buildings, special buildings, airports, cemeteries.
    protected_categories = [1, 2, 3, 7, 9]

    protected_area = areal_stat[areal_stat["AS18_27"].isin(protected_categories)]
    protected_area_full = fill_raster_dataframe(protected_area)
    # Correction of the reference of each raster cell from bottom left to top left
    protected_area_full["N_COORD"] = protected_area_full["N_COORD"] + 100
    csv_to_tiff(protected_area_full, attribute="AS18_27", path=r"data/landuse_landcover/processed/protected_area.tif")
    # print(areal_stat.head(50).to_string())


def get_unproductive_area(limits):
    areal_stat = pd.read_csv(r'data/landuse_landcover/landcover/ag-b-00.03-37-area-csv.csv', sep=";")
    if limits:
        areal_stat = areal_stat[(areal_stat["E_COORD"] >= limits[0]) &
                                (areal_stat["E_COORD"] <= limits[2]) &
                                (areal_stat["N_COORD"] >= limits[1]) &
                                (areal_stat["N_COORD"] <= limits[3])]

    areal_stat = areal_stat[["E_COORD", "N_COORD", "AS18_27"]]
    #print(areal_stat.shape)
    # Cycling: only true water bodies are impassable. Forest (23), shrub forest (24), and
    # woody vegetation (25) are removed — cycle paths through these are common and permitted.
    unproductive_zones = [26, 27]
    unproductive_area = areal_stat[areal_stat["AS18_27"].isin(unproductive_zones)]
    #print(unproductive_area.shape)
    unproductive_area_full = fill_raster_dataframe(unproductive_area)
    #print(unproductive_area_full.head(10).to_string())
    # Correction of the reference of each raster cell from bottom left to top left
    unproductive_area_full["N_COORD"] = unproductive_area_full["N_COORD"] + 100
    csv_to_tiff(unproductive_area_full, attribute="AS18_27", path=r"data/landuse_landcover/processed/unproductive_area.tif")

    # AS85_17 17 Klassen gemäss Standardnomenklatur der Arealstatistik 1979/85
    # AS85_4  4 Hauptbereiche gemäss Standardnomenklatur der Arealstatistik 1979/85
    # LU85_10 10 Klassen der Bodennutzung der Arealstatistik 1979/85
    # LU85_4  4 Hauptbereiche der Bodennutzung der Arealstatistik 1979/85





"""
def import_pendler_matrix(
    path_csv: str = 'data/OD/pendler_matrix.csv',
    canton_filter: str = 'ZH',
    cycling_mode_share: float = 0.05,
    max_distance_km: float = 15.0,
    output_dir: str = 'data/OD',
):
    
    Import and process the BFS Pendlermatrix (OD matrix at Gemeinde level).

    DATA SOURCE — download manually before running:
      https://www.bfs.admin.ch/asset/de/ts-x-11.04.04.05-2018
      → CSV file: "Erwerbstätige nach Wohn- und Arbeitsgemeinde"

    Key columns:
      WOHNKANTON / WOHNGEMEINDE    : residence canton + BFS Gemeinde number
      ARBEITSKANTON / ARBEITSGEMEINDE : workplace canton + BFS Gemeinde number
      ERWERBSTAETIGE               : number of commuters on that OD pair
    
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
                    if sample.str.match(r'^[0-9]{1,4}$').mean() > 0.8:
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
"""