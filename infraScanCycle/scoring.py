import math
import sys
import os
from joblib import Parallel, delayed
import zipfile
import timeit

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
import numpy as np
import scipy.io
from scipy.interpolate import griddata
from scipy.optimize import minimize, Bounds, least_squares
import rasterio
from rasterio.transform import from_origin
from rasterio.features import geometry_mask, shapes, rasterize
from shapely.geometry import Point, Polygon, box, shape, MultiPolygon, mapping
from shapely.ops import unary_union
from pyproj import Transformer
from rasterio.mask import mask
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import networkx as nx
from itertools import islice
import time


_AP_IDS_CACHE: "frozenset | None" = None

def _get_ap_ids() -> frozenset:
    global _AP_IDS_CACHE
    if _AP_IDS_CACHE is None:
        _ap_path = r"data/Network/processed/points_corridor.gpkg"
        if os.path.exists(_ap_path):
            _ap_gdf = gpd.read_file(_ap_path)
            if "ID_point" in _ap_gdf.columns:
                _AP_IDS_CACHE = frozenset(_ap_gdf["ID_point"].astype(int).tolist())
            else:
                _AP_IDS_CACHE = frozenset(_ap_gdf.index.astype(int).tolist())
        else:
            _AP_IDS_CACHE = frozenset()
    return _AP_IDS_CACHE


def construction_costs(cycle_path, upgrade):

    candidates = gpd.read_file(r"data/Network/processed/development_candidates.gpkg")
    # Connectivity bridges (dev_type='connectivity') are never scored — exclude them.
    candidates = candidates[
        (candidates["within_corridor"] | candidates["on_border"]) &
        (candidates["dev_type"] != "connectivity")
    ].fillna(0)

    # Each candidate is already one edge → no groupby needed
    candidates["path_len"] = candidates.geometry.length

    # Differentiate build cost: Netzlücke = full build, Schwachstelle = upgrade
    candidates["unit_cost"] = candidates["dev_type"].apply(
        lambda t: cycle_path if t == "netzluecke" else upgrade
    )
    candidates["building_costs"] = candidates["path_len"] * candidates["unit_cost"]

    candidates.to_file(r"data/Network/processed/links_with_geometry_attributes.gpkg", driver="GPKG")

    out = candidates[["ID_new", "geometry", "building_costs"]]
    out.to_file(r"data/costs/construction.gpkg", driver="GPKG")
    return


def maintenance_costs(duration, cycle_path, structural):
    generated_links_gdf = gpd.read_file(r"data/Network/processed/links_with_geometry_attributes.gpkg")

    generated_links_gdf["operational_maint"] = duration * generated_links_gdf["path_len"] * cycle_path

    costs_links = gpd.read_file(r"data/costs/construction.gpkg")
    costs_links["structural_maint"] = costs_links["building_costs"] * structural * duration

    # Merge column "structural_maint" to generated links using ID_new
    generated_links_gdf = generated_links_gdf.merge(costs_links[["ID_new", "structural_maint"]], on="ID_new",
                                                    how="left")
    generated_links_gdf["maintenance"] = generated_links_gdf["operational_maint"] + generated_links_gdf[
        "structural_maint"]

    # Only keep df with ID_new and maintenance costs
    generated_links_gdf = generated_links_gdf[["ID_new", "geometry", "maintenance"]]

    # Store the modified GeoDataFrame
    generated_links_gdf.to_file(r"data/costs/maintenance.gpkg", driver='GPKG')
    return







def land_tb_reallocated(links, buffer_distance):
    zones = gpd.read_file(r"data/landuse_landcover/processed/partly_protected.gpkg")
    print("Zones", zones.name.unique())

    buffer = links.copy()
    buffer = buffer[buffer.geometry.notna() & buffer.geometry.is_valid]
    # Create a buffer around each line

    # TODO: BUG — `dissolved` and `dissolved_geometries` are used here but
    # never defined in this function.  This will raise NameError at runtime.
    # The intent seems to be buffering the links, dissolving by zone type, and
    # accumulating the union.  Implement the missing geometry buffering step:
    #   buffer['geometry'] = buffer.geometry.buffer(buffer_distance)
    #   dissolved = buffer.dissolve()
    #   dissolved_geometries = []
    # and place those lines BEFORE this block.
    valid_geoms = dissolved.geometry.apply(make_valid)
    dissolved_geometries.append(valid_geoms.unary_union)

    # Initialize the columns for the areas of overlap
    for mp_id in zones['name'].unique():
        links[f'{mp_id}_area'] = 0.0

    # Calculate the overlapping area for each polygon with each multipolygon
    for idx, multipolygon in zones.iterrows():
        # Get the current multipolygon_id
        mp_id = multipolygon['name']
        if multipolygon['geometry'] is None or not multipolygon['geometry'].is_valid:
            continue
            # Calculate the intersection with each polygon in A
        # This returns a GeoSeries of the intersecting geometries
        intersections = links['buffer'].intersection(multipolygon['geometry'])

        # Calculate the area of each intersection
        links[f'{mp_id}_area'] = intersections.area

    links = links.drop(columns="buffer")

    return links


def externalities_costs(ce_cycling_path, realloc_forest, realloc_FFF, realloc_dry_meadow, realloc_period,
                        nat_fragmentation, fragm_period, nat_loss_habitat, habitat_period):
    # Import dataframe with links geometries
    generated_links_gdf = gpd.read_file(r"data/Network/processed/links_with_geometry_attributes.gpkg")
    generated_links_gdf = generated_links_gdf.fillna(0)

    # Climate cost: CO2 from cycle path construction only (no tunnel/bridge emissions)
    generated_links_gdf["climate_cost"] = generated_links_gdf["path_len"] * ce_cycling_path

    # Land reallocation
    # Cycle paths have a smaller footprint than roads — use a narrower buffer
    buffer_distance = 5  # metres (cycle path width)
    generated_links_gdf = land_tb_reallocated(generated_links_gdf, buffer_distance)

    generated_links_gdf["land_realloc"] = realloc_period * (
            generated_links_gdf["wald_area"] * realloc_forest +
            generated_links_gdf["fruchtfolgeflaeche_area"] * realloc_FFF +
            (generated_links_gdf["trockenweiden_area"] +
             generated_links_gdf["trockenlandschaften_area"]) * realloc_dry_meadow)

    # Nature and landscape: fragmentation and habitat loss along full path length
    generated_links_gdf["nature"] = generated_links_gdf["path_len"] * (
            nat_fragmentation * fragm_period + nat_loss_habitat * habitat_period)

    generated_links_gdf = generated_links_gdf[
        ["ID_new", "geometry", "climate_cost", "land_realloc", "nature"]]
    generated_links_gdf.to_file(r"data/costs/externalities.gpkg")

    return






def accessibility_developments(costs, VTT_h, duration):
    scenario_paths = ['s1_pop.tif', 's2_pop.tif', 's3_pop.tif']
    voronoi_path = r"data/Voronoi/voronoi_developments_tt_values.shp"

    trip_generation_day_cell = 1.14  # trip/p/d
    duration_d = duration * 365
    trip_generation = trip_generation_day_cell * duration_d
    VTT = VTT_h / 60 / 60  # CHF/sec
    print(f"VTT: {VTT}")

    voronoi_gdf = gpd.read_file(voronoi_path)

    # Precompute polygon list per row
    polygon_lists = {
        idx: list(row['geometry'].geoms) if isinstance(row['geometry'], MultiPolygon) else [row['geometry']]
        for idx, row in voronoi_gdf.iterrows()
    }

    # Load all scenario rasters once to avoid reopening them per polygon
    scenario_data = {}
    for path in scenario_paths:
        with rasterio.open(fr"data/independent_variable/processed/scenario/{path}") as src:
            scenario_data[path] = {
                'data': src.read(1) * trip_generation,
                'transform': src.transform,
                'shape': (src.height, src.width),
            }
    # Group by development so each TT raster is opened only once (not once per scenario)
    for id_development, dev_rows in voronoi_gdf.groupby('ID_develop'):
        tt_path = fr"data/Network/travel_time/developments/dev{id_development}_travel_time_raster.tif"
        with rasterio.open(tt_path) as tt_src:
            tt_tif = tt_src.read(1)
            tt_transform = tt_src.transform
            tt_shape = (tt_src.height, tt_src.width)

        for idx, _ in dev_rows.iterrows():
            polygons = polygon_lists[idx]
            tt_mask = geometry_mask(polygons, transform=tt_transform, invert=True, out_shape=tt_shape)
            tt_masked = tt_tif * tt_mask

            for path in scenario_paths:
                scen = scenario_data[path]
                trip_mask = geometry_mask(polygons, transform=scen['transform'], invert=True, out_shape=scen['shape'])
                total_tt = tt_masked * (scen['data'] * trip_mask)
                column_name = path.split('.')[0]
                voronoi_gdf.at[idx, column_name] = np.nansum(total_tt) * VTT

    voronoi_gdf.to_file(r"data/Voronoi/voronoi_developments_local_accessibility.gpkg", driver='GPKG')
    voronoi_gdf = voronoi_gdf.drop(columns=['geometry'])
    grouped_sum = voronoi_gdf.groupby('ID_develop').sum()
    grouped_sum = grouped_sum[["s1_pop", "s2_pop", "s3_pop"]]
    costs = costs[["s1_pop", "s2_pop", "s3_pop"]]

    grouped_sum["local_s1"] = costs["s1_pop"] - grouped_sum["s1_pop"]
    grouped_sum["local_s2"] = costs["s2_pop"] - grouped_sum["s2_pop"]
    grouped_sum["local_s3"] = costs["s3_pop"] - grouped_sum["s3_pop"]
    grouped_sum = grouped_sum.reset_index().rename(columns={'index': 'ID_development'})

    grouped_sum.to_csv('data/costs/local_accessibility.csv', index=False)
    return


def accessibility_status_quo(VTT_h, duration):
    travel_time_path = r"data/Network/travel_time/travel_time_raster.tif"
    scenario_paths = ['s1_pop.tif', 's2_pop.tif', 's3_pop.tif']
    voronoi_path = r"data/Network/travel_time/Voronoi_statusquo.gpkg"

    trip_generation_day_cell = 1.14  # trip/p/d
    duration_d = duration * 365
    trip_generation = trip_generation_day_cell * duration_d
    VTT = VTT_h / 60 / 60  # CHF/sec

    voronoi_gdf = gpd.read_file(voronoi_path)

    # Precompute polygon list per row
    polygon_lists = {
        idx: list(row['geometry'].geoms) if isinstance(row['geometry'], MultiPolygon) else [row['geometry']]
        for idx, row in voronoi_gdf.iterrows()
    }

    # Open travel time raster once; precompute tt_masked per polygon
    with rasterio.open(travel_time_path) as tt_src:
        tt_tif = tt_src.read(1)
        tt_masked = {
            idx: tt_tif * geometry_mask(polygons, transform=tt_src.transform, invert=True,
                                        out_shape=(tt_src.height, tt_src.width))
            for idx, polygons in polygon_lists.items()
        }

    # Process each scenario, reusing precomputed TT masks
    for path in scenario_paths:
        with rasterio.open(fr"data/independent_variable/processed/scenario/{path}") as scenario_tif:
            trip_tif = scenario_tif.read(1) * trip_generation
            column_name = path.split('.')[0]

            for idx, polygons in polygon_lists.items():
                trip_mask = geometry_mask(polygons, transform=scenario_tif.transform, invert=True,
                                          out_shape=(scenario_tif.height, scenario_tif.width))
                total_tt = tt_masked[idx] * (trip_tif * trip_mask)
                voronoi_gdf.at[idx, column_name] = np.nansum(total_tt) * VTT

    voronoi_gdf.to_file(r"data/Voronoi/voronoi_developments_local_accessibility.gpkg", driver='GPKG')
    voronoi_gdf = voronoi_gdf.drop(columns=['geometry'])
    return voronoi_gdf.sum()


def nw_from_osm(limits):
    os.makedirs("data/Network/OSM_road", exist_ok=True)
    # Split the area into smaller polygons
    num_splits = 10  # Adjust this to get 1/10th of the area (e.g., 3 for a 1/9th split)
    sub_polygons = split_area(limits, num_splits)

    # Initialize the transformer between LV95 and WGS 84
    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    for i, lv95_sub_polygon in enumerate(sub_polygons):

        # Convert the coordinates of the sub-polygon to lat/lon
        lat_lon_frame = Polygon([transformer.transform(*point) for point in lv95_sub_polygon.exterior.coords])

        try:
            # Attempt to process the OSM data for the sub-polygon
            print(f"Processing sub-polygon {i + 1}/{len(sub_polygons)}", end='/r')

            # Retry up to 3 times for connection errors
            for attempt in range(3):
                try:
                    G = ox.graph_from_polygon(lat_lon_frame, network_type="all", simplify=True, truncate_by_edge=True)
                    break
                except Exception as e:
                    if attempt < 2:
                        print(f"  Retry {attempt + 1}/3 for sub-polygon {i + 1}...")
                        time.sleep(5)
                    else:
                        raise e  # re-raise on final attempt so outer except catches it

            G = ox.add_edge_speeds(G, fallback=30)
            # ... rest of the function unchanged

        except Exception as e:
            print(f"Skipping graph in sub-polygon {i + 1} due to error: {e}")
            continue


def split_area(limits, num_splits):
    """
    Split the given area defined by 'limits' into 'num_splits' smaller polygons.

    :param limits: Tuple of (min_x, max_x, min_y, max_y) in LV95 coordinates.
    :param num_splits: The number of splits along each axis (total areas = num_splits^2).
    :return: List of shapely Polygon objects representing the smaller areas.
    """
    min_x, min_y, max_x, max_y = limits
    width = (max_x - min_x) / num_splits
    height = (max_y - min_y) / num_splits

    sub_polygons = []
    for i in range(num_splits):
        for j in range(num_splits):
            # Calculate the corners of the sub-polygon
            sub_min_x = min_x + i * width
            sub_max_x = sub_min_x + width
            sub_min_y = min_y + j * height
            sub_max_y = sub_min_y + height

            # Create the sub-polygon and add it to the list
            sub_polygon = box(sub_min_x, sub_min_y, sub_max_x, sub_max_y)
            sub_polygons.append(sub_polygon)

    return sub_polygons


def osm_nw_to_raster(limits):
    # Add comment

    # Folder containing all the geopackages
    gpkg_folder = "data/Network/OSM_road"

    # List all geopackage files in the folder
    gpkg_files = [os.path.join(gpkg_folder, f) for f in os.listdir(gpkg_folder) if f.endswith('.gpkg')]

    # Combine all geopackages into one GeoDataFrame
    gdf_combined = gpd.GeoDataFrame(pd.concat([gpd.read_file(f) for f in gpkg_files], ignore_index=True))
    # Assuming 'speed' is the column with speed limits
    # Convert speeds to numeric, handling non-numeric values
    gdf_combined['speed'] = pd.to_numeric(gdf_combined['speed_kph'], errors='coerce')

    # Drop NaN values or replace them with 0, depending on how you want to handle them
    # gdf_combined.dropna(subset=['speed_kph'], inplace=True)
    gdf_combined['speed_kph'].fillna(30, inplace=True)
    # print(gdf_combined.crs)
    # print(gdf_combined.head(10).to_string())
    os.makedirs('data/Network/OSM_tif', exist_ok=True)
    gdf_combined.to_file('data/Network/OSM_tif/nw_speed_limit.gpkg')
    print("file stored")

    gdf_combined = gpd.read_file('data/Network/OSM_tif/nw_speed_limit.gpkg')
    gdf_combined = gdf_combined[gdf_combined.geometry.notna() & gdf_combined.geometry.is_valid]

    # Define the resolution
    resolution = 100

    # Define the bounds of the raster (aligned with your initial limits)
    minx, miny, maxx, maxy = limits
    print(limits)

    # Compute the number of rows and columns
    num_cols = int((maxx - minx) / resolution)
    num_rows = int((maxy - miny) / resolution)

    # Initialize the raster with 4 = minimal travel speed (or np.nan for no-data value)
    # raster = np.zeros((num_rows, num_cols), dtype=np.float32)
    raster = np.full((num_rows, num_cols), 4, dtype=np.float32)

    # Define the transform
    transform = from_origin(west=minx, north=maxy, xsize=resolution, ysize=resolution)

    # lake = gpd.read_file(r"data/landuse_landcover/landcover/water_ch/Typisierung_LV95/typisierung.gpkg")
    ###############################################################################################################

    print("ready to fill")

    tot_num = num_cols * num_rows
    count = 0

    for row in range(num_rows):
        for col in range(num_cols):

            # print(row, " - ", col)
            # Find the bounds of the cell
            cell_bounds = box(minx + col * resolution,
                              maxy - row * resolution,
                              minx + (col + 1) * resolution,
                              maxy - (row + 1) * resolution)

            # Find the roads that intersect with this cell
            # print(gdf_combined.head(10).to_string())
            intersecting_roads = gdf_combined[gdf_combined.intersects(cell_bounds)]

            # Debugging print
            # print(f"Cell {row},{col} intersects with {len(intersecting_roads)} roads")

            # If there are any intersecting roads, find the maximum speed limit
            if not intersecting_roads.empty:
                max_speed = intersecting_roads['speed_kph'].max()
                raster[row, col] = max_speed

            # Print the progress
            count += 1
            progress_percentage = (count / tot_num) * 100
            sys.stdout.write(f"\rProgress: {progress_percentage:.2f}%")
            sys.stdout.flush()

    # Check for spatial overlap with the second raster and update values if necessary
    with rasterio.open(r"data/landuse_landcover/processed/unproductive_area.tif") as src2:
        unproductive_area = src2.read(1)
        if raster.shape == unproductive_area.shape:
            print("Network raster and unproductive area are overalpping")
            mask = np.logical_and(unproductive_area > 0, unproductive_area < 100)
            raster[mask] = 0
        else:
            print("Network raster and unproductive area are not overalpping!!!!!")

    with rasterio.open(
            'data/Network/OSM_tif/speed_limit_raster.tif',
            'w',
            driver='GTiff',
            height=raster.shape[0],
            width=raster.shape[1],
            count=1,
            dtype=str(raster.dtype),
            crs="EPSG:2056",
            transform=transform,
    ) as dst:
        dst.write(raster, 1)


def tif_to_vector(raster_path, vector_path):
    # Step 1: Read the raster data
    with rasterio.open(raster_path) as src:
        image = src.read(1)  # Read the first band

    # Step 2: Apply threshold or classification
    # This is an example where we create a mask for all values above a threshold
    mask = image >= 0  # Define your own threshold value

    # Step 3: Convert the masked raster to vector shapes
    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) in enumerate(
        shapes(image, mask=mask, transform=src.transform)))

    # Step 4: Create Shapely polygons and generate a GeoDataFrame
    geometries = [shape(result['geometry']) for result in results]
    gdf = gpd.GeoDataFrame.from_features([
        {"geometry": geom, "properties": {"value": val}}
        for geom, val in zip(geometries, mask)
    ])

    # Save to a new Shapefile, if desired
    # gdf.to_file(vector_path)
    return gdf


def map_coordinates_to_developments():
    df_temp = gpd.read_file(r"data/Network/processed/new_links_realistic_costs.gpkg")
    _dc = gpd.read_file(r"data/Network/processed/development_candidates.gpkg")[["ID_new", "geometry"]]
    _dc["geometry"] = _dc.geometry.centroid
    points = _dc
    # print(points.columns)
    # print(points.head(10).to_string())
    # print(points["ID_new"].unique())

    df_temp = df_temp.merge(points, how='left', left_on='ID_new', right_on='ID_new')
    # print(df_temp["ID_new"].unique())
    # print(df_temp.head(10).to_string())
    df_temp['geometry'] = df_temp['geometry_y']
    # todo buildin and externality not in index
    df_temp = df_temp[['ID_current', 'ID_new', 'building_costs', 'externality_costs', 'geometry']]
    df_temp["total_cost"] = df_temp["building_costs"] + df_temp["externality_costs"]
    # df_temp['geometry'] = df_temp['geometry_y'].replace('geometry_y', 'geometry')

    # df_temp = df_temp.rename({"geometry_y":"geometry"})
    # print(df_temp.head(10).to_string())
    df_temp = gpd.GeoDataFrame(df_temp, geometry="geometry")
    df_temp.to_file(r"data/costs/building_externalities.gpkg")
    return

def route_comfort(
        edges_aug,
        od_scenarios,
        points_corridor,
        VTTS=18.2,
        duration=50,
        trips_per_year=250,
        scenarios=('s1', 's2', 's3'),
        n_samples=20,
):
    """
    Route comfort benefit per Netzlücke, per scenario (before/after path comparison).

    Requires compute_dijkstra_tts_od() to have been run first — it saves the
    time-optimal paths to data/OD/paths/base.parquet and dev_<ID>.parquet.
    Those paths are loaded here so Dijkstra does not need to be re-run.

    Algorithm:

    PRE-COMPUTE per-edge comfort cost [h/trip]:
      comfort_h = length_m × slope_extra_f × ε[ROUTENTYP] / (ffs_edge × 1000)

    FOR each saved path in base.parquet:
      C_base[i,j] = Σ_{e ∈ path} comfort_h[e]

    FOR d IN Netzlücken:
      Override comfort_h for edge d using routentyp_built and ffs_built.
      Load dev_d.parquet (paths on the upgraded graph).
      C_d[i,j] = Σ_{e ∈ path} comfort_h_d[e]
      FOR s IN scenarios:
        benefit[d][s] = Σ_{(i,j)} trips[s][i,j] × (C_base[i,j] − C_d[i,j])
                        × VTTS × trips_per_year × duration

    Slope VoD multipliers (Meister et al. 2021):
      < 2 %: +0 %,  2–6 %: +41 %,  ≥ 6 %: +251 %

    Route-type discomfort multiplier ε (Table 3, methodology):
      Veloschnellroute 1.0,  Hauptverbindung/Freizeit 1.3,
      Nebenverbindung 1.6,  Netzlücke/connector 2.0

    Saves
    -----
    data/costs/route_comfort.csv  — comfort_s1, comfort_s2, comfort_s3 per Netzlücke
    """
    import time

    SLOPE_FLAT_MAX   = 2.0
    SLOPE_MEDIUM_MAX = 6.0
    EXTRA_FLAT       = 0.00
    EXTRA_MEDIUM     = 0.41
    EXTRA_STEEP      = 2.51

    EPSILON = {
        'Velobahn':                       1.0,
        'Veloschnellroute':               1.0,
        'Hauptverbindung':                1.3,
        'Nebenverbindung':                1.6,
        'Zusätzliche Freizeitverbindung': 1.3,
        'Netzlücke':                      2.0,
        'connector':                      2.0,
    }
    EPSILON_DEFAULT = 2.0

    os.makedirs('data/costs', exist_ok=True)
    print(f"\n--- ROUTE COMFORT (Dijkstra, per-edge built ffs) ---")

    # ── 1. Pre-compute slope extra factor per edge via DEM ────────────────────
    def _sample_elevations(geom, elev_data, elev_transform, elev_nodata):
        length = geom.length
        if length == 0:
            return np.array([])
        fracs = np.linspace(0, 1, n_samples)
        pts   = [geom.interpolate(f, normalized=True) for f in fracs]
        rows, cols = rasterio.transform.rowcol(
            elev_transform,
            [p.x for p in pts],
            [p.y for p in pts],
        )
        h, w = elev_data.shape
        elevs = []
        for r, c in zip(rows, cols):
            if 0 <= r < h and 0 <= c < w:
                v = float(elev_data[r, c])
                elevs.append(np.nan if (elev_nodata is not None and v == elev_nodata) else v)
            else:
                elevs.append(np.nan)
        return np.array(elevs, dtype=float)

    def _slope_extra_factor(geom, elev_data, elev_transform, elev_nodata):
        elevs   = _sample_elevations(geom, elev_data, elev_transform, elev_nodata)
        valid   = ~np.isnan(elevs)
        if valid.sum() < 2:
            return 0.0
        seg_len  = geom.length / (n_samples - 1)
        grad_pct = np.abs(np.diff(elevs[valid])) / seg_len * 100.0
        flat_len   = np.sum(grad_pct <  SLOPE_FLAT_MAX)   * seg_len
        medium_len = np.sum((grad_pct >= SLOPE_FLAT_MAX) & (grad_pct < SLOPE_MEDIUM_MAX)) * seg_len
        steep_len  = np.sum(grad_pct >= SLOPE_MEDIUM_MAX) * seg_len
        total_len  = flat_len + medium_len + steep_len
        if total_len == 0:
            return 0.0
        return (flat_len * EXTRA_FLAT + medium_len * EXTRA_MEDIUM + steep_len * EXTRA_STEEP) / total_len

    elev_path = r"data/elevation_model/elevation.tif"
    elev_data = elev_transform = elev_nodata = None
    if os.path.exists(elev_path):
        with rasterio.open(elev_path) as src:
            elev_data      = src.read(1)
            elev_transform = src.transform
            elev_nodata    = src.nodata
    else:
        print("  WARNING: DEM not found — slope discomfort set to 0 for all edges")

    t0 = time.time()
    edge_slope_f = {}
    for _, row in edges_aug.iterrows():
        coords = list(row.geometry.coords)
        u = (round(coords[0][0], 1), round(coords[0][1], 1))
        v = (round(coords[-1][0], 1), round(coords[-1][1], 1))
        sf = _slope_extra_factor(row.geometry, elev_data, elev_transform, elev_nodata) \
             if elev_data is not None else 0.0
        edge_slope_f[(u, v)] = sf
        edge_slope_f[(v, u)] = sf
    print(f"  Slope pre-computation: {len(edges_aug)} edges  [{time.time() - t0:.1f}s]")

    # ── 2. Build per-edge comfort attribute lookup ────────────────────────────
    # No full graph needed — just a dict keyed by (u, v) node tuples so we can
    # sum comfort_h along pre-computed paths without re-running Dijkstra.
    edge_comfort = {}  # (u, v) → comfort_h [h/trip]
    for _, row in edges_aug.iterrows():
        coords    = list(row.geometry.coords)
        u         = (round(coords[0][0],  1), round(coords[0][1],  1))
        v         = (round(coords[-1][0], 1), round(coords[-1][1], 1))
        length_m  = float(row['length_m']) if row.get('length_m', 0) > 0 else row.geometry.length
        routetype = str(row.get('ROUTENTYP', ''))
        ffs_edge  = float(row['ffs']) if ('ffs' in row.index
                                          and pd.notna(row.get('ffs'))
                                          and float(row.get('ffs', 0)) > 0) else 13.0
        eps       = EPSILON.get(routetype, EPSILON_DEFAULT)
        slope_f   = edge_slope_f.get((u, v), 0.0)
        comfort_h = length_m * slope_f * eps / (ffs_edge * 1000.0)
        edge_comfort[(u, v)] = comfort_h
        edge_comfort[(v, u)] = comfort_h

    print(f"  Edge comfort attributes: {len(edge_comfort) // 2} edges")

    # ── 3. Collect active OD pairs (filter to IDs present in saved base paths) ─
    import json
    base_path_file = 'data/OD/paths/base.parquet'
    if not os.path.exists(base_path_file):
        raise FileNotFoundError(
            f"{base_path_file} not found — run compute_dijkstra_tts_od() first "
            f"to generate saved paths before calling route_comfort()")

    base_gdf      = gpd.read_parquet(base_path_file)
    reachable_ids = set(base_gdf['origin_id']) | set(base_gdf['dest_id'])

    od_by_scenario = {}
    for s, od_df in od_scenarios.items():
        if s not in scenarios:
            continue
        tmp = od_df[['origin_id', 'dest_id', 'trips']].copy()
        tmp['origin_id'] = tmp['origin_id'].astype(int)
        tmp['dest_id']   = tmp['dest_id'].astype(int)
        tmp = tmp[tmp['trips'].fillna(0) > 0].loc[
            tmp['origin_id'].isin(reachable_ids) & tmp['dest_id'].isin(reachable_ids)]
        od_by_scenario[s] = dict(zip(zip(tmp['origin_id'], tmp['dest_id']), tmp['trips']))

    # ── 4. Compute base path comfort from saved paths ─────────────────────────
    t0     = time.time()
    C_base = {}
    for _, row in base_gdf.iterrows():
        nodes = [tuple(n) for n in json.loads(row['path_nodes'])]
        C_base[(int(row['origin_id']), int(row['dest_id']))] = sum(
            edge_comfort.get((nodes[k], nodes[k + 1]), 0.0)
            for k in range(len(nodes) - 1)
        )
    print(f"  Base comfort: {len(C_base):,} OD pairs  [{time.time() - t0:.1f}s]")

    # ── 5. Per-development loop ───────────────────────────────────────────────
    nl_gdf  = edges_aug[edges_aug['ROUTENTYP'] == 'Netzlücke']
    records = []

    print(f"\n  Scoring {len(nl_gdf)} Netzlücken …")
    for orig_idx, nl_row in nl_gdf.iterrows():
        id_edge  = (int(nl_row['ID_edge'])
                    if 'ID_edge' in nl_row.index and pd.notna(nl_row.get('ID_edge'))
                    else orig_idx)
        length_m = float(nl_row.get('length_m', nl_row.geometry.length))

        coords = list(nl_row.geometry.coords)
        u = (round(coords[0][0],  1), round(coords[0][1],  1))
        v = (round(coords[-1][0], 1), round(coords[-1][1], 1))
        slope_f       = edge_slope_f.get((u, v), 0.0)
        ffs_b         = float(nl_row.get('ffs_built', 13.0))
        built_rt      = str(nl_row.get('routentyp_built', 'Hauptverbindung'))
        built_eps     = EPSILON.get(built_rt, EPSILON_DEFAULT)
        new_comfort_h = length_m * (1.0 + slope_f) * built_eps / (ffs_b * 1000.0)

        # Override comfort for this edge; all others stay the same
        edge_comfort_d = {**edge_comfort,
                          (u, v): new_comfort_h,
                          (v, u): new_comfort_h}

        t0      = time.time()
        C_d     = {}
        dev_file = f'data/OD/paths/dev_{id_edge}.parquet'
        if os.path.exists(dev_file):
            dev_gdf = gpd.read_parquet(dev_file)
            for _, prow in dev_gdf.iterrows():
                nodes = [tuple(n) for n in json.loads(prow['path_nodes'])]
                C_d[(int(prow['origin_id']), int(prow['dest_id']))] = sum(
                    edge_comfort_d.get((nodes[k], nodes[k + 1]), 0.0)
                    for k in range(len(nodes) - 1)
                )
        else:
            print(f"    WARNING: {dev_file} not found — comfort benefit will be 0")

        rec      = {'ID_new': id_edge, 'length_m': round(length_m, 1)}
        s2_ben_h = 0.0
        for s in scenarios:
            benefit_h = sum(
                trips * (C_base.get((i, j), 0.0)
                            - C_d.get((i, j), C_base.get((i, j), 0.0)))
                for (i, j), trips in od_by_scenario.get(s, {}).items()
                if (i, j) in C_base
            )
            rec[f'comfort_{s}'] = round(benefit_h * VTTS * trips_per_year * duration, 2)
            if s == 's2':
                s2_ben_h = benefit_h
        records.append(rec)

        print(f"    [{id_edge:>4}] {int(length_m):>5}m | s2 comfort benefit={s2_ben_h:.4f} h/day"
              f"  [{time.time() - t0:.1f}s]")

    result = pd.DataFrame(records)
    result.to_csv(r"data/costs/route_comfort.csv", index=False)
    print(f"\n  Route comfort → data/costs/route_comfort.csv  ({len(result)} Netzlücken)")
    return result


# ── Per-ROUTENTYP monetised crash cost (CHF/Pkm) ──────────────────────────────
# Source: KNA Limmattal; Veloschnellroute and Netzlücke are bounds,
# intermediate values interpolated linearly across four quality levels.
CRASH_RATE_CHF_PKM = {
    'Velobahn':                       0.104,
    'Veloschnellroute':               0.104,
    'Hauptverbindung':                0.409,
    'Nebenverbindung':                0.714,
    'Zusätzliche Freizeitverbindung': 0.409,
    'Netzlücke':                      1.020,
    'connector':                      1.020,
}
CRASH_RATE_DEFAULT = 1.020   # fallback for unknown ROUTENTYP



def safety_benefits(
        edges_aug,
        od_scenarios,
        points_corridor,
        duration=50,
        trips_per_year=250,
        scenarios=('s1', 's2', 's3'),
):
    """
    Safety benefit per Netzlücke, per scenario (before/after path comparison).

    Requires compute_dijkstra_tts_od() to have been run first — it saves the
    time-optimal paths to data/OD/paths/base.parquet and dev_<ID>.parquet.
    Those paths are loaded here so Dijkstra does not need to be re-run.

    Algorithm:

    PRE-COMPUTE per-edge safety cost [CHF/trip]:
      safety_chf = CRASH_RATE_CHF_PKM[ROUTENTYP] × length_km

    FOR each saved path in base.parquet:
      S_base[i,j] = Σ_{e ∈ path} safety_chf[e]

    FOR d IN Netzlücken:
      Override safety_chf for edge d using routentyp_built.
      Load dev_d.parquet (paths on the upgraded graph).
      S_d[i,j] = Σ_{e ∈ path} safety_chf_d[e]
      FOR s IN scenarios:
        benefit[d][s] = Σ_{(i,j)} trips[s][i,j] × (S_base[i,j] − S_d[i,j])
                        × trips_per_year × duration

    Crash cost values (CHF/Pkm) from KNA Limmattal, Table 4 methodology.

    Saves
    -----
    data/costs/safety_benefits.csv  — safety_s1, safety_s2, safety_s3 per Netzlücke
    data/costs/safety_benefits.gpkg
    """
    import time
    import json

    os.makedirs('data/costs', exist_ok=True)
    print(f"\n--- SAFETY BENEFITS (saved paths, CHF/Pkm crash rates) ---")

    # ── 1. Build per-edge safety cost lookup ──────────────────────────────────
    # No full graph or Dijkstra needed — just a dict keyed by (u, v) so we can
    # sum safety_chf along pre-computed paths loaded from data/OD/paths/.
    edge_safety = {}  # (u, v) → safety_chf [CHF/trip]
    for _, row in edges_aug.iterrows():
        coords    = list(row.geometry.coords)
        u         = (round(coords[0][0],  1), round(coords[0][1],  1))
        v         = (round(coords[-1][0], 1), round(coords[-1][1], 1))
        length_m  = float(row['length_m']) if row.get('length_m', 0) > 0 else row.geometry.length
        routetype = str(row.get('ROUTENTYP', ''))
        safety_chf = CRASH_RATE_CHF_PKM.get(routetype, CRASH_RATE_DEFAULT) * (length_m / 1000.0)
        edge_safety[(u, v)] = safety_chf
        edge_safety[(v, u)] = safety_chf

    print(f"  Edge safety attributes: {len(edge_safety) // 2} edges")

    # ── 2. Collect active OD pairs (filter to IDs present in saved base paths) ─
    base_path_file = 'data/OD/paths/base.parquet'
    if not os.path.exists(base_path_file):
        raise FileNotFoundError(
            f"{base_path_file} not found — run compute_dijkstra_tts_od() first "
            f"to generate saved paths before calling safety_benefits()")

    base_gdf      = gpd.read_parquet(base_path_file)
    reachable_ids = set(base_gdf['origin_id']) | set(base_gdf['dest_id'])

    od_by_scenario = {}
    for s, od_df in od_scenarios.items():
        if s not in scenarios:
            continue
        tmp = od_df[['origin_id', 'dest_id', 'trips']].copy()
        tmp['origin_id'] = tmp['origin_id'].astype(int)
        tmp['dest_id']   = tmp['dest_id'].astype(int)
        tmp = tmp[tmp['trips'].fillna(0) > 0].loc[
            tmp['origin_id'].isin(reachable_ids) & tmp['dest_id'].isin(reachable_ids)]
        od_by_scenario[s] = dict(zip(zip(tmp['origin_id'], tmp['dest_id']), tmp['trips']))

    # ── 3. Compute base path safety from saved paths ──────────────────────────
    t0     = time.time()
    S_base = {}
    for _, row in base_gdf.iterrows():
        nodes = [tuple(n) for n in json.loads(row['path_nodes'])]
        S_base[(int(row['origin_id']), int(row['dest_id']))] = sum(
            edge_safety.get((nodes[k], nodes[k + 1]), 0.0)
            for k in range(len(nodes) - 1)
        )
    print(f"  Base safety: {len(S_base):,} OD pairs  [{time.time() - t0:.1f}s]")

    # ── 4. Per-development loop ───────────────────────────────────────────────
    nl_gdf  = edges_aug[edges_aug['ROUTENTYP'] == 'Netzlücke']
    records = []

    print(f"\n  Scoring {len(nl_gdf)} Netzlücken …")

    for orig_idx, nl_row in nl_gdf.iterrows():
        id_edge  = (int(nl_row['ID_edge'])
                    if 'ID_edge' in nl_row.index and pd.notna(nl_row.get('ID_edge'))
                    else orig_idx)
        length_m = float(nl_row.get('length_m', nl_row.geometry.length))

        coords = list(nl_row.geometry.coords)
        u = (round(coords[0][0],  1), round(coords[0][1],  1))
        v = (round(coords[-1][0], 1), round(coords[-1][1], 1))
        built_rt       = str(nl_row.get('routentyp_built', 'Hauptverbindung'))
        built_rate     = CRASH_RATE_CHF_PKM.get(built_rt, CRASH_RATE_DEFAULT)
        new_safety_chf = built_rate * (length_m / 1000.0)

        # Override safety cost for this edge; all others stay the same
        edge_safety_d = {**edge_safety,
                         (u, v): new_safety_chf,
                         (v, u): new_safety_chf}

        t0      = time.time()
        S_d     = {}
        dev_file = f'data/OD/paths/dev_{id_edge}.parquet'
        if os.path.exists(dev_file):
            dev_gdf = gpd.read_parquet(dev_file)
            for _, prow in dev_gdf.iterrows():
                nodes = [tuple(n) for n in json.loads(prow['path_nodes'])]
                S_d[(int(prow['origin_id']), int(prow['dest_id']))] = sum(
                    edge_safety_d.get((nodes[k], nodes[k + 1]), 0.0)
                    for k in range(len(nodes) - 1)
                )
        else:
            print(f"    WARNING: {dev_file} not found — safety benefit will be 0")

        rec      = {'ID_new': id_edge, 'length_m': round(length_m, 1)}
        s2_ben   = 0.0
        for s in scenarios:
            benefit_chf = sum(
                trips * (S_base.get((i, j), 0.0)
                            - S_d.get((i, j), S_base.get((i, j), 0.0)))
                for (i, j), trips in od_by_scenario.get(s, {}).items()
                if (i, j) in S_base
            ) * trips_per_year * duration
            rec[f'safety_{s}'] = round(benefit_chf, 2)
            if s == 's2':
                s2_ben = benefit_chf
        records.append(rec)

        print(f"    [{id_edge:>4}] {int(length_m):>5}m | s2 safety benefit={s2_ben:,.0f} CHF"
              f"  [{time.time() - t0:.1f}s]")

    out_df = pd.DataFrame(records)
    out_df.to_csv("data/costs/safety_benefits.csv", index=False)

    nl_mask   = edges_aug['ROUTENTYP'] == 'Netzlücke'
    cands_geo = edges_aug[nl_mask][['ID_edge', 'geometry']].copy()
    cands_geo = cands_geo.rename(columns={'ID_edge': 'ID_new'})
    cands_geo['geometry'] = cands_geo.geometry.centroid
    out_gdf = cands_geo.merge(out_df, on='ID_new', how='right')
    if out_gdf.geometry.notna().any():
        out_gdf = gpd.GeoDataFrame(out_gdf, geometry='geometry', crs=edges_aug.crs)
        out_gdf.to_file("data/costs/safety_benefits.gpkg", driver="GPKG")

    print(f"\n  Safety benefits → data/costs/safety_benefits.csv / .gpkg  ({len(out_df)} Netzlücken)")
    if len(out_df):
        print(f"  Benefit range (s2): {out_df['safety_s2'].min():,.0f}–"
              f"{out_df['safety_s2'].max():,.0f} CHF")
    return out_df


def od_fastest_paths(*args, **kwargs):
    """Removed — travel-time comparisons are handled by
    OSM_network.travel_cost_developments() (multi-source Dijkstra, ALLTAG network).
    """
    print("[od_fastest_paths] Skipped — use travel_cost_developments() instead.")
    return None


def import_pendler_matrix(
    path_csv: str = 'data/OD/pendler_matrix.csv',
    canton_filter: str = 'ZH',
    cycling_mode_share: float = 0.08,
    output_dir: str = 'data/OD',
):
    """
    Import and process the BFS Pendlermatrix (OD matrix at Gemeinde level).

    DATA SOURCE — download manually before running:
      https://www.bfs.admin.ch/asset/de/ts-x-11.04.04.05-2018
      -> CSV file: "Erwerbstätige nach Wohn- und Arbeitsgemeinde"

    Key columns:
      WOHNKANTON / WOHNGEMEINDE       : residence canton + BFS Gemeinde number
      ARBEITSKANTON / ARBEITSGEMEINDE : workplace canton + BFS Gemeinde number
      ERWERBSTAETIGE                  : number of commuters on that OD pair

    Saves:
      od_matrix_zh.csv         — all canton-filtered pairs with commuters_total & commuters_cycling
      od_matrix_zh_cycling.csv — same, filtered to commuters_cycling > 0
    """
    import numpy as np
    import fiona

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
    zh_ids = {canton_filter, '1', 1, 'ZH', 'zh'}
    mask  = (od['WOHNKANTON'].astype(str).isin([str(x) for x in zh_ids])) | \
            (od['ARBEITSKANTON'].astype(str).isin([str(x) for x in zh_ids]))
    od_zh = od[mask].copy().reset_index(drop=True)
    print(f"  After canton filter ({canton_filter}): {len(od_zh)} pairs, "
          f"{od_zh['ERWERBSTAETIGE'].sum():,} commuters")

    gem_path = 'data/raw/Gemeinden/gemeinden_centroid.gpkg'
    if os.path.exists(gem_path):
        layers = fiona.listlayers(gem_path)
        gem_layer = next(
            (l for l in layers if 'hoheitsgebiet' in l.lower() or 'gemeinde' in l.lower()),
            layers[0]
        )
        gemeinden = gpd.read_file(gem_path, layer=gem_layer)
        if gemeinden.crs and gemeinden.crs.to_epsg() != 2056:
            gemeinden = gemeinden.to_crs("EPSG:2056")

        id_col = next(
            (c for c in gemeinden.columns
             if c.upper() in ['GMDNR', 'BFS_NR', 'GEMEINDENR', 'NR', 'OBJECTVAL',
                               'BFSNR', 'GEM_NR', 'NUMMER', 'GKZ',
                               'BFS_NUMMER', 'BFSNUMMER']),
            None
        )
        if id_col is None:
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

        gemeinden['BFS_NR'] = gemeinden[id_col].astype(str).str.strip().str.zfill(4)
        if 'kantonsnummer' in gemeinden.columns:
            ktn = gemeinden['kantonsnummer']
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
        matched = od_zh[['x_wohn', 'y_wohn', 'x_arbeit', 'y_arbeit']].notna().all(axis=1).sum()
        print(f"  OD pairs with both centroids matched: {matched} / {len(od_zh)}")

        if matched == 0:
            print("  WARNING: No centroids matched — check BFS ID format in CSV vs gpkg")
            od_zh['dist_km'] = np.nan
        else:
            od_zh['dist_km'] = np.sqrt(
                (pd.to_numeric(od_zh['x_arbeit'], errors='coerce') -
                 pd.to_numeric(od_zh['x_wohn'],   errors='coerce'))**2 +
                (pd.to_numeric(od_zh['y_arbeit'], errors='coerce') -
                 pd.to_numeric(od_zh['y_wohn'],   errors='coerce'))**2
            ) / 1000
            print(f"  Distance range: "
                  f"{od_zh['dist_km'].min():.1f}–{od_zh['dist_km'].max():.1f} km")
    else:
        print(f"  Warning: {gem_path} not found — skipping coord join")
        od_zh['dist_km'] = np.nan

    od_zh['commuters_total']   = od_zh['ERWERBSTAETIGE']
    od_zh['commuters_cycling'] = (od_zh['commuters_total'] * cycling_mode_share).round().astype(int)
    print(f"  Cycling commuters ({cycling_mode_share * 100:.0f}% mode share): "
          f"{od_zh['commuters_cycling'].sum():,}")

    od_zh.to_csv(f'{output_dir}/od_matrix_zh.csv', index=False)
    od_zh[od_zh['commuters_cycling'] > 0].to_csv(
        f'{output_dir}/od_matrix_zh_cycling.csv', index=False)
    print(f"  Saved -> {output_dir}/od_matrix_zh.csv")
    print(f"  Saved -> {output_dir}/od_matrix_zh_cycling.csv")

    return od_zh


def od_cycling_weighted():
    SCENARIOS     = ['s1', 's2', 's3']
    VORONOI_PATH  = 'data/Voronoi/voronoi_developments_euclidian_values.shp'
    COMMUNE_OD    = 'data/OD/od_matrix_zh_cycling.csv'
    GEMEINDE_PATH = 'data/_basic_data/Gemeindegrenzen/UP_GEMEINDEN_F.shp'

    print(f"\n--- OD MATRIX (commune-disaggregated) ---")

    voronoi_vals = gpd.read_file(VORONOI_PATH)
    voronoi_vals['ID_point'] = voronoi_vals['ID_point'].astype(int)

    # ── Assign commune_id to each Voronoi node via centroid spatial join ─────
    gemeinde = gpd.read_file(GEMEINDE_PATH)[['BFS', 'geometry']]
    vor_centroids = voronoi_vals[['ID_point', 'geometry']].copy()
    vor_centroids['geometry'] = vor_centroids.geometry.centroid
    vor_centroids = gpd.sjoin(
        vor_centroids,
        gemeinde.rename(columns={'BFS': 'commune_id'}),
        how='left', predicate='within',
    ).drop(columns=['index_right'], errors='ignore')
    voronoi_vals = voronoi_vals.merge(
        vor_centroids[['ID_point', 'commune_id']], on='ID_point', how='left'
    )
    # Nodes outside all Gemeinde polygons fall back to commune_id = -1
    voronoi_vals['commune_id'] = voronoi_vals['commune_id'].fillna(-1).astype(int)
    n_matched = (voronoi_vals['commune_id'] >= 0).sum()
    print(f"  Commune join: {n_matched}/{len(voronoi_vals)} Voronoi nodes matched to a Gemeinde")

    # ── Load commune-level OD (cycling commuters — modal share already baked in) ──
    commune_od = pd.read_csv(COMMUNE_OD)[['WOHNGEMEINDE', 'ARBEITSGEMEINDE', 'commuters_cycling']].copy()

    os.makedirs('data/OD', exist_ok=True)
    od_scenarios = {}

    for s in SCENARIOS:
        pop_col  = f'{s}_pop'
        empl_col = f'{s}_empl'
        if pop_col not in voronoi_vals.columns or empl_col not in voronoi_vals.columns:
            print(f"  Warning: {pop_col}/{empl_col} missing — skipping {s}")
            continue

        nodes = voronoi_vals[['ID_point', 'commune_id', pop_col, empl_col]].copy()
        nodes[pop_col]  = nodes[pop_col].clip(lower=0).fillna(0)
        nodes[empl_col] = nodes[empl_col].clip(lower=0).fillna(0)

        # Within-commune shares — pop drives origins, empl drives destinations
        nodes['origin_share'] = nodes.groupby('commune_id')[pop_col].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 0.0)
        nodes['dest_share'] = nodes.groupby('commune_id')[empl_col].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 0.0)

        # Cartesian product of all node pairs (i ≠ j)
        origins = (nodes[['ID_point', 'commune_id', 'origin_share']]
                   .rename(columns={'ID_point': 'origin_id', 'commune_id': 'commune_id_o'})
                   .assign(_key=1))
        dests   = (nodes[['ID_point', 'commune_id', 'dest_share']]
                   .rename(columns={'ID_point': 'dest_id', 'commune_id': 'commune_id_d'})
                   .assign(_key=1))
        od = origins.merge(dests, on='_key').drop(columns='_key')
        od = od[od['origin_id'] != od['dest_id']].reset_index(drop=True)

        # Join commune-to-commune total commuters and disaggregate
        od = od.merge(commune_od,
                      left_on=['commune_id_o', 'commune_id_d'],
                      right_on=['WOHNGEMEINDE', 'ARBEITSGEMEINDE'],
                      how='left').drop(columns=['WOHNGEMEINDE', 'ARBEITSGEMEINDE'], errors='ignore')
        od['commuters_cycling'] = od['commuters_cycling'].fillna(0)
        od['trips']    = od['origin_share'] * od['dest_share'] * od['commuters_cycling']
        od['scenario'] = s

        od.to_csv(f'data/OD/od_{s}.csv', index=False)
        od_scenarios[s] = od
        print(f"  {s}: {len(od):,} pairs  |  "
              f"origin pop = {nodes[pop_col].sum():,.0f}  |  "
              f"daily cycling trips = {od['trips'].sum():,.0f}")

    if od_scenarios:
        pd.concat(od_scenarios.values(), ignore_index=True).to_csv(
            'data/OD/od_scenarios.csv', index=False)
        print("  Combined → data/OD/od_scenarios.csv")

    return od_scenarios, voronoi_vals


def _save_paths_parquet(P, T, L, path_out):
    """
    Serialise a set of OD shortest paths to GeoParquet.

    Each row is one OD pair with columns:
      origin_id  – int ID_point of the origin node
      dest_id    – int ID_point of the destination node
      tt_sec     – travel time [s]
      tt_min     – travel time [min]
      length_m   – routed distance [m]
      n_edges    – number of edges in the path
      path_nodes – JSON list of [x, y] node coordinates (rounded to 0.1 m),
                   used by route_comfort / safety_benefits to traverse the
                   path edge-by-edge without re-running Dijkstra
      geometry   – LineString (EPSG:2056) for visualisation in QGIS etc.

    Parameters
    ----------
    P : dict  (origin_id, dest_id) → list of (x, y) node tuples
    T : dict  (origin_id, dest_id) → travel time [s]
    L : dict  (origin_id, dest_id) → routed distance [m]
    path_out : str  output .parquet file path
    """
    import json
    from shapely.geometry import LineString

    rows = []
    for (id_i, id_j), nodes in P.items():
        if len(nodes) < 2:
            continue
        tt_sec = T.get((id_i, id_j), float('nan'))
        length = L.get((id_i, id_j), float('nan'))
        rows.append({
            'origin_id':  id_i,
            'dest_id':    id_j,
            'tt_sec':     tt_sec,
            'tt_min':     tt_sec / 60.0 if tt_sec == tt_sec else float('nan'),
            'length_m':   length,
            'n_edges':    len(nodes) - 1,
            'path_nodes': json.dumps([[n[0], n[1]] for n in nodes]),
            'geometry':   LineString(nodes),
        })

    gdf = gpd.GeoDataFrame(rows, crs='EPSG:2056')
    gdf.to_parquet(path_out)
    print(f"    Paths → {path_out}  ({len(gdf)} OD pairs)")


def compute_dijkstra_tts_od(
        edges_aug,
        od_scenarios,
        points_corridor,
        VTTS=18.2,
        duration=50,
        trips_per_year=250,
        scenarios=('s1', 's2', 's3'),
):
    """
    Travel time savings (TTS) from upgrading each Netzlücke.

    Algorithm
    ---------
    PRE-COMPUTE base shortest paths for every active OD pair (trips > 0):
      T_base[i,j], L_base[i,j] ← Dijkstra(G_base, i)  [shared across scenarios]

    FOR d IN developments:
      G_d ← G_base with edge d upgraded to ffs_built
      T_d[i,j] ← Dijkstra(G_d, i)   for each active origin i
      FOR s IN scenarios:
        savings[d][s] ← Σ_{(i,j) ∈ OD[s]}  trips[s][i,j] × (T_base[i,j] − T_d[i,j])

    Outputs
    -------
    data/OD/od_base_travel_times.csv   — base tt, routed dist, detour per OD pair
    data/OD/traveltime_savings_od.csv  — TTS per Netzlücke × scenario (long format)
    data/costs/traveltime_savings.csv  — wide format for net_benefits()
    """
    import time
    import networkx as nx
    from OSM_network import build_graph_direct

    os.makedirs('data/OD',    exist_ok=True)
    os.makedirs('data/costs', exist_ok=True)

    print(f"\n--- TRAVEL TIME SAVINGS (Dijkstra, per-edge built ffs) ---")

    # ── 1. Build base graph ──────────────────────────────────────────────────
    t0 = time.time()
    G_base = build_graph_direct(edges_aug)
    print(f"  Base graph: {G_base.number_of_nodes()} nodes, "
          f"{G_base.number_of_edges()} edges  [{time.time() - t0:.1f}s]")

    # ── 2. Node ↔ ID_point mapping ───────────────────────────────────────────
    id_to_node = {}  # int ID_point → (x, y) graph node key
    id_to_xy   = {}  # int ID_point → (float x, float y) for straight-line dist
    for _, row in points_corridor.iterrows():
        id_pt = int(row['ID_point'])
        node  = (round(row.geometry.x, 1), round(row.geometry.y, 1))
        id_to_xy[id_pt] = (row.geometry.x, row.geometry.y)
        if node in G_base.nodes:
            id_to_node[id_pt] = node

    n_unmapped = len(points_corridor) - len(id_to_node)
    print(f"  Node mapping: {len(id_to_node)} mapped, {n_unmapped} unmapped "
          f"(boundary nodes not in graph — expected)")

    # ── 3. Collect active OD pairs (trips > 0 in any scenario, both ends in graph) ──
    od_by_scenario = {}   # s → {(i, j): trips}
    active_pairs   = set()

    for s, od_df in od_scenarios.items():
        if s not in scenarios:
            continue
        tmp = od_df[['origin_id', 'dest_id', 'trips']].copy()
        tmp['origin_id'] = tmp['origin_id'].astype(int)
        tmp['dest_id']   = tmp['dest_id'].astype(int)
        tmp = tmp[
            tmp['trips'].fillna(0) > 0
        ].loc[
            tmp['origin_id'].isin(id_to_node) & tmp['dest_id'].isin(id_to_node)
        ]
        od_by_scenario[s] = dict(zip(zip(tmp['origin_id'], tmp['dest_id']), tmp['trips']))
        active_pairs |= set(od_by_scenario[s].keys())

    # Pre-group destinations by origin for O(1) Dijkstra dispatch
    dests_by_origin: dict = {}
    for id_i, id_j in active_pairs:
        dests_by_origin.setdefault(id_i, set()).add(id_j)

    unique_origins = sorted(dests_by_origin.keys())
    print(f"  Active OD pairs: {len(active_pairs):,}  "
          f"({len(unique_origins)} unique origins across {len(od_by_scenario)} scenarios)")

    # ── 4. PRE-COMPUTE base shortest paths for all active OD pairs ──────────
    t0 = time.time()
    T_base = {}  # (i, j) → travel time [seconds]
    L_base = {}  # (i, j) → routed distance [metres]
    P_base = {}  # (i, j) → node list [(x, y), ...]

    for id_i in unique_origins:
        node_i = id_to_node[id_i]
        lengths, paths = nx.single_source_dijkstra(G_base, node_i, weight='weight')
        for id_j in dests_by_origin[id_i]:
            node_j = id_to_node[id_j]
            if node_j not in lengths:
                continue
            path = paths[node_j]
            T_base[(id_i, id_j)] = lengths[node_j]
            L_base[(id_i, id_j)] = sum(
                G_base[path[k]][path[k + 1]].get('length', 0.0)
                for k in range(len(path) - 1)
            )
            P_base[(id_i, id_j)] = path

    print(f"  Base Dijkstra: {len(T_base):,} reachable pairs  [{time.time() - t0:.1f}s]")

    # Persist base paths as GeoParquet (one LineString per OD pair)
    os.makedirs('data/OD/paths', exist_ok=True)
    _save_paths_parquet(P_base, T_base, L_base, 'data/OD/paths/base.parquet')

    # ── 5. Save base OD travel times (all reachable pairs) ───────────────────
    base_rows = []
    for (id_i, id_j), tt_sec in T_base.items():
        xi, yi   = id_to_xy.get(id_i, (None, None))
        xj, yj   = id_to_xy.get(id_j, (None, None))
        routed_m = L_base.get((id_i, id_j))
        air_m    = ((xi - xj)**2 + (yi - yj)**2)**0.5 if (xi and xj) else None
        detour   = routed_m / air_m if (air_m and air_m > 0 and routed_m) else None
        # Clamp to ≥1.0 — sub-1 is floating-point noise on near-coincident nodes
        if detour is not None and detour < 1.0:
            detour = 1.0
        base_rows.append({
            'origin_id':     id_i,
            'dest_id':       id_j,
            'tt_sec':        tt_sec,
            'tt_min':        tt_sec / 60,
            'dist_routed_m': routed_m,
            'dist_air_m':    air_m,
            'detour_factor': detour,
        })

    pd.DataFrame(base_rows).to_csv('data/OD/od_base_travel_times.csv', index=False)
    print(f"  Base OD travel times → data/OD/od_base_travel_times.csv  "
          f"({len(base_rows)} pairs, all reachable pairs kept)")

    # ── 6. FOR EACH DEVELOPMENT: upgrade edge, re-run Dijkstra, compute savings ──
    nl_gdf   = edges_aug[edges_aug['ROUTENTYP'] == 'Netzlücke']
    tts_rows = []

    print(f"\n  Scoring {len(nl_gdf)} Netzlücken …")

    for orig_idx, nl_row in nl_gdf.iterrows():
        id_edge  = (int(nl_row['ID_edge'])
                    if 'ID_edge' in nl_row.index and pd.notna(nl_row.get('ID_edge'))
                    else orig_idx)
        length_m = float(nl_row.get('length_m', nl_row.geometry.length))

        coords = list(nl_row.geometry.coords)
        u = (round(coords[0][0],  1), round(coords[0][1],  1))
        v = (round(coords[-1][0], 1), round(coords[-1][1], 1))

        ffs_b = float(nl_row.get('ffs_built', 13.0))
        G_d   = G_base.copy()
        new_w = (length_m / 1000.0) / ffs_b * 3600.0  # seconds
        if G_d.has_edge(u, v): G_d[u][v]['weight'] = new_w
        if G_d.has_edge(v, u): G_d[v][u]['weight'] = new_w

        # Dijkstra from each active origin on the development graph
        t0 = time.time()
        T_d: dict = {}
        L_d: dict = {}
        P_d: dict = {}

        for id_i in unique_origins:
            node_i = id_to_node[id_i]
            lengths_d, paths_d = nx.single_source_dijkstra(G_d, node_i, weight='weight')
            for id_j in dests_by_origin[id_i]:
                node_j = id_to_node[id_j]
                if node_j in lengths_d:
                    path_d = paths_d[node_j]
                    T_d[(id_i, id_j)] = lengths_d[node_j]
                    L_d[(id_i, id_j)] = sum(
                        G_d[path_d[k]][path_d[k + 1]].get('length', 0.0)
                        for k in range(len(path_d) - 1)
                    )
                    P_d[(id_i, id_j)] = path_d

        _save_paths_parquet(P_d, T_d, L_d,
                            f'data/OD/paths/dev_{id_edge}.parquet')

        n_improved = sum(
            1 for k in T_base
            if T_d.get(k, T_base[k]) < T_base[k]
        )

        # savings[d][s] = Σ_{(i,j) ∈ OD[s]}  trips × (T_base − T_d)  [person·h/day]
        row = {
            'ID_new':    id_edge,
            'length_m':  length_m,
            'ffs_built': ffs_b,
        }
        s2_tts_h = 0.0
        for s in scenarios:
            tts_sec = sum(
                trips * (T_base[(i, j)] - T_d.get((i, j), T_base[(i, j)]))
                for (i, j), trips in od_by_scenario.get(s, {}).items()
                if (i, j) in T_base
            )
            tts_h       = tts_sec / 3600.0
            savings_chf = tts_h * VTTS * trips_per_year * duration
            if s == 's2':
                s2_tts_h = tts_h
            row[f'T_{s}']           = tts_h       # travel-time saving [person·h/day]
            row[f'savings_chf_{s}'] = savings_chf  # monetary saving [CHF/day]
        tts_rows.append(row)

        print(f"    [{id_edge:>4}] {int(length_m):>5}m | {n_improved:>4} pairs improved | "
              f"s2 TTS={s2_tts_h:.3f} h/day  [{time.time() - t0:.1f}s]")

    tts_df = pd.DataFrame(tts_rows)

    # Primary output: one row per Netzlücke, T_s* = travel-time savings [person·h/day]
    t_cols   = [f'T_{s}' for s in scenarios if f'T_{s}' in tts_df.columns]
    out_cols = ['ID_new', *t_cols, 'length_m', 'ffs_built']
    tts_df[out_cols].to_csv('data/OD/traveltime_savings_od.csv', index=False)
    print(f"\n  TTS (wide) → data/OD/traveltime_savings_od.csv  "
          f"({len(tts_df)} Netzlücken, columns: {out_cols})")

    # ── 7. Monetary savings for net_benefits() ───────────────────────────────
    if not tts_df.empty:
        chf_rename = {
            'savings_chf_s2': 'tt_low',
            'savings_chf_s1': 'tt_medium',
            'savings_chf_s3': 'tt_high',
        }
        chf_present = [c for c in chf_rename if c in tts_df.columns]
        wide = tts_df[['ID_new', *chf_present]].copy()
        wide = wide.rename(columns=chf_rename)
        for c in ['tt_low', 'tt_medium', 'tt_high']:
            if c in wide.columns:
                wide[c] = -wide[c].abs()  # cost convention: negative = saving
        wide.to_csv('data/costs/traveltime_savings.csv', index=False)
        print(f"  net_benefits() input → data/costs/traveltime_savings.csv  "
              f"({len(wide)} developments)")

    return tts_df


def node_accessibility(
        voronoi_path='data/Voronoi/voronoi_developments_euclidian_values.shp',
        raster_template='data/landuse_landcover/processed/zone_no_infra/protected_area_corridor.tif',
        beta=2.0,
        scenarios=('s1', 's2', 's3'),
        travel_times_path=None):
    """
    Gravity-based accessibility score for each corridor node i under each scenario s.

    A) For every node i, sum the employment at all other nodes j weighted by
       inverse impedance raised to beta:

         A_{i,s} = sum_{j != i}  empl_{j,s} / impedance_{ij}^beta

       Impedance is network travel time in minutes when travel_times_path is given
       (output of compute_dijkstra_tts_od), otherwise Euclidean distance in metres.
       Unreachable pairs contribute 0. Scores are normalised globally across all
       scenarios to [0, 1].

    B) Rasterize Voronoi polygons at the resolution of the protected-area raster
       (50 m, EPSG:2056). Saves one .tif per scenario to data/Network/accessibility/.

    C) Save node-level scores to node_access_scores.csv and return per-scenario
       GeoDataFrames (Voronoi polygons with score columns attached).
    """
    import rasterio
    import rasterio.features
    import numpy as np

    voronoi_gdf = gpd.read_file(voronoi_path)
    voronoi_gdf['ID_point'] = voronoi_gdf['ID_point'].astype(int)

    os.makedirs('data/Network/accessibility', exist_ok=True)
    results  = {}
    all_rows = []

    # read raster template once
    with rasterio.open(raster_template) as tpl:
        tpl_meta      = tpl.meta.copy()
        tpl_transform = tpl.transform
        tpl_shape     = (tpl.height, tpl.width)
        tpl_crs       = tpl.crs

    tpl_meta.update(dtype='float32', nodata=-1.0, count=1)

    # reproject Voronoi to raster CRS if needed
    if voronoi_gdf.crs is None:
        voronoi_gdf = voronoi_gdf.set_crs("EPSG:2056")
    if voronoi_gdf.crs.to_epsg() != tpl_crs.to_epsg():
        voronoi_gdf = voronoi_gdf.to_crs(tpl_crs)

    ids = voronoi_gdf['ID_point'].values
    n   = len(ids)

    if travel_times_path is not None and os.path.exists(travel_times_path):
        # ── Network travel-time impedance (minutes) from Dijkstra base paths ──
        id_to_idx = {int(id_pt): idx for idx, id_pt in enumerate(ids)}
        tt_df = pd.read_csv(travel_times_path, usecols=['origin_id', 'dest_id', 'tt_min'])
        dist = np.full((n, n), np.nan)
        o_idx = tt_df['origin_id'].map(id_to_idx)
        d_idx = tt_df['dest_id'].map(id_to_idx)
        valid = o_idx.notna() & d_idx.notna()
        dist[o_idx[valid].astype(int).values,
             d_idx[valid].astype(int).values] = tt_df.loc[valid, 'tt_min'].values
        np.fill_diagonal(dist, np.nan)
        n_routed = int(valid.sum())
        print(f"  Impedance: network travel time (min) — {n_routed:,} routed pairs "
              f"from {travel_times_path}")
    else:
        # ── Fallback: pairwise Euclidean distances between Voronoi centroids ──
        centroids = voronoi_gdf.geometry.centroid
        cx = centroids.x.values
        cy = centroids.y.values
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.nan)
        if travel_times_path is not None:
            print(f"  Warning: {travel_times_path} not found — falling back to Euclidean distance")
        else:
            print("  Impedance: Euclidean distance (m)")

    # ── A) compute raw gravity scores for all scenarios first (for global normalisation) ──
    raw_scores = {}
    for s in scenarios:
        empl_col = f'{s}_empl'
        if empl_col not in voronoi_gdf.columns:
            continue
        empl = voronoi_gdf[empl_col].clip(lower=0).fillna(0).values.astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            contrib = np.where(np.isnan(dist), 0.0, empl[None, :] / dist ** beta)
        raw_scores[s] = contrib.sum(axis=1)

    global_max = float(np.concatenate(list(raw_scores.values())).max()) if raw_scores else 1.0
    global_max = global_max if global_max > 0 else 1.0
    print(f"  Gravity accessibility — beta={beta}  global max raw score: {global_max:.4f}")

    for s in scenarios:
        empl_col = f'{s}_empl'
        if s not in raw_scores:
            print(f"  Warning: {empl_col} missing — skipping {s}")
            continue

        gdf = voronoi_gdf.copy()
        gdf['access_score'] = raw_scores[s] / global_max  # normalised to [0, 1]

        # ── B) rasterize ─────────────────────────────────────────────────
        shapes = [
            (geom, float(score))
            for geom, score in zip(gdf.geometry, gdf['access_score'])
            if geom is not None and not geom.is_empty
        ]
        burned = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=tpl_shape,
            transform=tpl_transform,
            fill=-1.0,
            dtype='float32',
        )

        out_path = f'data/Network/accessibility/access_score_{s}.tif'
        with rasterio.open(out_path, 'w', **tpl_meta) as dst:
            dst.write(burned, 1)

        n_valid = (burned >= 0).sum()
        print(f"  {s}: {len(gdf)} nodes  |  "
              f"score min={gdf['access_score'].min():.3f}  "
              f"max={gdf['access_score'].max():.3f}  "
              f"mean={gdf['access_score'].mean():.3f}  |  "
              f"raster {n_valid:,} valid pixels  → {out_path}")

        results[s] = gdf[['ID_point', 'geometry', empl_col, 'access_score']].copy()

        for _, row in gdf.iterrows():
            all_rows.append({
                'ID_point':     row['ID_point'],
                'scenario':     s,
                empl_col:       row[empl_col],
                'access_score': row['access_score'],
            })

    # ── C) save node scores ───────────────────────────────────────────────
    if all_rows:
        score_df = pd.DataFrame(all_rows)
        score_df.to_csv('data/Network/accessibility/node_access_scores.csv', index=False)
        print(f"  Saved → data/Network/accessibility/node_access_scores.csv  "
              f"({len(score_df)} rows, {len(scenarios)} scenarios)")

    return results


def compute_accessibility_benefits(
        edges_aug,
        points_corridor,
        voronoi_path='data/Voronoi/voronoi_developments_euclidian_values.shp',
        beta=2.0,
        scenarios=('s1', 's2', 's3'),
):
    """
    Population-weighted accessibility benefit per Netzlücke per scenario.

    Algorithm
    ---------
    PRE-COMPUTE baseline:
      A_base[s][i] = Σ_{j≠i}  empl[j,s] / T_base[i,j]^β
      where T_base[i,j] = travel time (min) on G_base

    FOR d IN Netzlücken:
      G_d = G_base with edge d upgraded to ffs_built
      T_d[i,j] = Dijkstra(G_d, i)  for each origin i (pop > 0)
      A_d[s][i] = Σ_{j≠i}  empl[j,s] / T_d[i,j]^β
      ΔA[s][i]  = A_d[s][i] - A_base[s][i]          (positive = accessibility gained)
      benefits[d][s] = Σ_i  pop[i,s] × ΔA[s][i]     (population-weighted aggregate)

    Saves
    -----
    data/costs/accessibility_benefits.csv  (ID_new, acc_s1, acc_s2, acc_s3)
    """
    import time
    import networkx as nx
    import numpy as np
    from OSM_network import build_graph_direct

    os.makedirs('data/costs', exist_ok=True)
    print(f"\n--- ACCESSIBILITY BENEFITS (Dijkstra, beta={beta}, per-edge built ffs) ---")

    # ── 1. Build base graph ──────────────────────────────────────────────────
    t0 = time.time()
    G_base = build_graph_direct(edges_aug)
    print(f"  Base graph: {G_base.number_of_nodes()} nodes, "
          f"{G_base.number_of_edges()} edges  [{time.time() - t0:.1f}s]")

    # ── 2. Node ↔ ID_point mapping ───────────────────────────────────────────
    id_to_node = {}
    for _, row in points_corridor.iterrows():
        id_pt = int(row['ID_point'])
        node  = (round(row.geometry.x, 1), round(row.geometry.y, 1))
        if node in G_base.nodes:
            id_to_node[id_pt] = node

    # ── 3. Load pop/empl per node per scenario ───────────────────────────────
    voronoi_gdf = gpd.read_file(voronoi_path)
    voronoi_gdf['ID_point'] = voronoi_gdf['ID_point'].astype(int)

    node_ids    = voronoi_gdf['ID_point'].values
    node_to_idx = {int(nid): k for k, nid in enumerate(node_ids)}

    pop_s  = {}   # s → float array aligned with voronoi_gdf rows
    empl_s = {}
    valid_scenarios = []
    for s in scenarios:
        pc, ec = f'{s}_pop', f'{s}_empl'
        if pc not in voronoi_gdf.columns or ec not in voronoi_gdf.columns:
            print(f"  Warning: {pc}/{ec} missing — skipping {s}")
            continue
        pop_s[s]  = voronoi_gdf[pc].clip(lower=0).fillna(0).values.astype(float)
        empl_s[s] = voronoi_gdf[ec].clip(lower=0).fillna(0).values.astype(float)
        valid_scenarios.append(s)

    # ── 4. Identify active origins (pop > 0) and destinations (empl > 0) ────
    any_pop  = sum(pop_s.values())
    any_empl = sum(empl_s.values())

    origin_ids = [int(node_ids[k]) for k in range(len(node_ids))
                  if any_pop[k] > 0 and int(node_ids[k]) in id_to_node]
    dest_ids   = [int(node_ids[k]) for k in range(len(node_ids))
                  if any_empl[k] > 0 and int(node_ids[k]) in id_to_node]

    print(f"  Active nodes: {len(origin_ids)} origins (pop>0), "
          f"{len(dest_ids)} destinations (empl>0)")

    # ── 5. PRE-COMPUTE base travel times (origin → all empl-dests) ──────────
    t0 = time.time()
    # T_base[id_i] = {id_j: tt_min}  — only pairs where tt > 0
    T_base: dict = {}
    for id_i in origin_ids:
        lengths, _ = nx.single_source_dijkstra(G_base, id_to_node[id_i], weight='weight')
        row = {}
        for id_j in dest_ids:
            if id_j == id_i:
                continue
            node_j = id_to_node[id_j]
            tt_sec = lengths.get(node_j)
            if tt_sec and tt_sec > 0:
                row[id_j] = tt_sec / 60.0
        T_base[id_i] = row

    n_reachable = sum(len(v) for v in T_base.values())
    print(f"  Base Dijkstra: {n_reachable:,} reachable origin→dest pairs  "
          f"[{time.time() - t0:.1f}s]")

    # ── 6. Baseline accessibility A_base[s][i] ───────────────────────────────
    # A_base[s][i] = Σ_j  empl[j,s] / tt_base[i,j]^β
    A_base: dict = {}
    for s in valid_scenarios:
        empl = empl_s[s]
        A_base[s] = {}
        for id_i in origin_ids:
            A_base[s][id_i] = sum(
                empl[node_to_idx[id_j]] / (tt ** beta)
                for id_j, tt in T_base[id_i].items()
            )

    # ── 7. FOR EACH DEVELOPMENT: upgrade, re-run Dijkstra, compute benefits ──
    nl_gdf   = edges_aug[edges_aug['ROUTENTYP'] == 'Netzlücke']
    acc_rows = []

    print(f"\n  Scoring {len(nl_gdf)} Netzlücken …")

    for orig_idx, nl_row in nl_gdf.iterrows():
        id_edge  = (int(nl_row['ID_edge'])
                    if 'ID_edge' in nl_row.index and pd.notna(nl_row.get('ID_edge'))
                    else orig_idx)
        length_m = float(nl_row.get('length_m', nl_row.geometry.length))

        coords = list(nl_row.geometry.coords)
        u = (round(coords[0][0],  1), round(coords[0][1],  1))
        v = (round(coords[-1][0], 1), round(coords[-1][1], 1))

        ffs_b = float(nl_row.get('ffs_built', 13.0))
        G_d   = G_base.copy()
        new_w = (length_m / 1000.0) / ffs_b * 3600.0
        if G_d.has_edge(u, v): G_d[u][v]['weight'] = new_w
        if G_d.has_edge(v, u): G_d[v][u]['weight'] = new_w

        # Dijkstra on G_d from each active origin
        t0 = time.time()
        T_d: dict = {}
        for id_i in origin_ids:
            lengths_d, _ = nx.single_source_dijkstra(G_d, id_to_node[id_i], weight='weight')
            row_d = {}
            for id_j in dest_ids:
                if id_j == id_i:
                    continue
                node_j = id_to_node[id_j]
                tt_sec = lengths_d.get(node_j)
                if tt_sec and tt_sec > 0:
                    row_d[id_j] = tt_sec / 60.0
            T_d[id_i] = row_d

        # benefits[d][s] = Σ_i  pop[i,s] × (A_d[s][i] − A_base[s][i])
        rec = {'ID_new': id_edge, 'length_m': length_m}
        s2_benefit = 0.0

        for s in valid_scenarios:
            empl = empl_s[s]
            pop  = pop_s[s]
            total = 0.0
            for id_i in origin_ids:
                pop_i = pop[node_to_idx[id_i]]
                if pop_i <= 0:
                    continue
                a_d = sum(
                    empl[node_to_idx[id_j]] / (tt ** beta)
                    for id_j, tt in T_d[id_i].items()
                )
                total += pop_i * (a_d - A_base[s][id_i])
            rec[f'acc_{s}'] = total
            if s == 's2':
                s2_benefit = total

        for s in scenarios:
            if f'acc_{s}' not in rec:
                rec[f'acc_{s}'] = 0.0

        acc_rows.append(rec)
        print(f"    [{id_edge:>4}] {int(length_m):>5}m | "
              f"s2 ΔA(pop-weighted)={s2_benefit:.2f}  [{time.time() - t0:.1f}s]")

    acc_df = pd.DataFrame(acc_rows)
    acc_df.to_csv('data/costs/accessibility_benefits.csv', index=False)
    print(f"\n  Accessibility benefits → data/costs/accessibility_benefits.csv  "
          f"({len(acc_df)} developments)")

    return acc_df


def net_benefits():
    """
    Compute net benefit per development:  NB = C + M + T + R + S [+ A]

    C  = construction cost           [CHF, negative]
    M  = maintenance cost            [CHF, negative]
    T  = travel time savings         [CHF, positive]
    R  = route comfort benefit       [CHF]
    S  = safety benefit              [CHF, positive]
    A  = accessibility benefit index [optional; included when
         data/costs/accessibility_benefits.csv exists]

    Handles two traveltime_savings.csv formats:
      • Preferred: columns tt_low / tt_medium / tt_high  (3-scenario)
      • Fallback:  columns tt_1 … tt_N                   (legacy N-scenario)
        → averages all tt_ columns → applies to all three scenarios equally

    Outputs
    -------
    data/costs/net_benefits.csv
    data/costs/net_benefits.gpkg
    """
    import os
    import pandas as pd
    import geopandas as gpd

    # ── Construction costs ────────────────────────────────────────────────────
    c_constr = gpd.read_file(r"data/costs/construction.gpkg")[["ID_new", "building_costs"]]
    c_constr["ID_new"] = c_constr["ID_new"].astype(int)

    # ── Maintenance costs ─────────────────────────────────────────────────────
    c_maint = gpd.read_file(r"data/costs/maintenance.gpkg")[["ID_new", "maintenance"]]
    c_maint["ID_new"] = c_maint["ID_new"].astype(int)

    # ── Travel time savings — detect column format ────────────────────────────
    _tt_path = r"data/costs/traveltime_savings.csv"
    if not os.path.exists(_tt_path):
        print(f"[net_benefits] WARNING: {_tt_path} not found — setting T = 0 for all developments")
        c_tt = pd.DataFrame({
            "ID_new":    pd.Series(dtype=int),
            "tt_low":    pd.Series(dtype=float),
            "tt_medium": pd.Series(dtype=float),
            "tt_high":   pd.Series(dtype=float),
        })
    else:
        c_tt = pd.read_csv(_tt_path)
        if "Unnamed: 0" in c_tt.columns:
            c_tt = c_tt.drop(columns=["Unnamed: 0"])
        if "development" in c_tt.columns:
            c_tt = c_tt.rename(columns={"development": "ID_new"})
        c_tt["ID_new"] = c_tt["ID_new"].astype(int)

    if {"tt_low", "tt_medium", "tt_high"}.issubset(c_tt.columns):
        # 3-scenario format (generated by tt_optimization_all_developments)
        c_tt = c_tt[["ID_new", "tt_low", "tt_medium", "tt_high"]]
    else:
        # Legacy N-scenario format (tt_1 … tt_N): average all tt_ columns
        tt_cols = sorted([c for c in c_tt.columns if c.startswith("tt_")])
        print(f"[net_benefits] WARNING: tt_low/medium/high not found.\n"
              f"  Found: {tt_cols}\n"
              f"  → averaging {len(tt_cols)} columns as scenario-independent T")
        c_tt["tt_avg"] = c_tt[tt_cols].mean(axis=1)
        c_tt["tt_low"]    = c_tt["tt_avg"]
        c_tt["tt_medium"] = c_tt["tt_avg"]
        c_tt["tt_high"]   = c_tt["tt_avg"]
        c_tt = c_tt[["ID_new", "tt_low", "tt_medium", "tt_high"]]

    # ── Route comfort benefit ─────────────────────────────────────────────────
    c_comfort = pd.read_csv(r"data/costs/route_comfort.csv")[["ID_new", "comfort_s1", "comfort_s2", "comfort_s3"]]
    c_comfort["ID_new"] = c_comfort["ID_new"].astype(int)

    # ── Safety benefits ───────────────────────────────────────────────────────
    c_safety = pd.read_csv(r"data/costs/safety_benefits.csv")[["ID_new", "safety_s1", "safety_s2", "safety_s3"]]
    c_safety["ID_new"] = c_safety["ID_new"].astype(int)

    """
    # ── Accessibility benefits (optional — included when file exists) ─────────
    _acc_path = r"data/costs/accessibility_benefits.csv"
    has_accessibility = os.path.exists(_acc_path)
    if has_accessibility:
        c_acc = pd.read_csv(_acc_path)[["ID_new", "acc_s1", "acc_s2", "acc_s3"]]
        c_acc["ID_new"] = c_acc["ID_new"].astype(int)
        print(f"[net_benefits] Accessibility benefits loaded: {len(c_acc)} developments")
    else:
        print(f"[net_benefits] No accessibility_benefits.csv found — A = 0 for all developments")
    """
    # ── Merge all components on ID_new ────────────────────────────────────────
    nb = c_constr.copy()
    merge_dfs = [c_maint, c_tt, c_comfort, c_safety]
    """
    if has_accessibility:
        merge_dfs.append(c_acc)
    """
    for df in merge_dfs:
        nb = nb.merge(df, on="ID_new", how="left")
    nb = nb.fillna(0)

    # ── Signed components ─────────────────────────────────────────────────────
    nb["C"] = -nb["building_costs"].abs()   # cost → always negative
    nb["M"] = -nb["maintenance"].abs()      # cost → always negative

    # T is already negative (forced by tt_optimization: -abs(savings))
    # In NB = C + M + T + R + S, a more-negative T means a higher-cost network.
    # Reverse the sign so T contributes positively to NB when savings exist.
    nb["T_s1"] = nb["tt_low"].abs()
    nb["T_s2"] = nb["tt_medium"].abs()
    nb["T_s3"] = nb["tt_high"].abs()

    nb["R_s1"] = nb["comfort_s1"]
    nb["R_s2"] = nb["comfort_s2"]
    nb["R_s3"] = nb["comfort_s3"]

    nb["S_s1"] = nb["safety_s1"]
    nb["S_s2"] = nb["safety_s2"]
    nb["S_s3"] = nb["safety_s3"]
    nb["A_s1"] = 0.0
    nb["A_s2"] = 0.0
    nb["A_s3"] = 0.0
    # ── Net benefit per scenario: NB = C + M + T + R + S + A ─────────────────
    nb["NB_s1"] = nb["C"] + nb["M"] + nb["T_s1"] + nb["R_s1"] + nb["S_s1"] + nb["A_s1"]
    nb["NB_s2"] = nb["C"] + nb["M"] + nb["T_s2"] + nb["R_s2"] + nb["S_s2"] + nb["A_s2"]
    nb["NB_s3"] = nb["C"] + nb["M"] + nb["T_s3"] + nb["R_s3"] + nb["S_s3"] + nb["A_s3"]

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_cols = [
        "ID_new",
        "C", "M",
        "T_s1", "T_s2", "T_s3",
        "R_s1", "R_s2", "R_s3",
        "S_s1", "S_s2", "S_s3",
        "NB_s1", "NB_s2", "NB_s3",
    ]
    out = nb[out_cols].copy()
    os.makedirs("data/costs", exist_ok=True)
    out.to_csv(r"data/costs/net_benefits.csv", index=False)

    # ── Save GPKG (attach representative point geometry from development edges) ─
    _dev_cands = gpd.read_file(r"data/Network/processed/development_candidates.gpkg")[["ID_new", "geometry"]]
    _dev_cands["ID_new"] = _dev_cands["ID_new"].astype(int)
    _dev_cands["geometry"] = _dev_cands.geometry.centroid
    out_gdf = _dev_cands.merge(out, on="ID_new", how="right")
    out_gdf = gpd.GeoDataFrame(out_gdf, geometry="geometry", crs="EPSG:2056")
    out_gdf.to_file(r"data/costs/net_benefits.gpkg", driver="GPKG")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[net_benefits] {len(out)} developments")
    for label, t_col, r_col, s_col, a_col, nb_col in [
        ("s1 (low)",    "T_s1", "R_s1", "S_s1", "A_s1", "NB_s1"),
        ("s2 (medium)", "T_s2", "R_s2", "S_s2", "A_s2", "NB_s2"),
        ("s3 (high)",   "T_s3", "R_s3", "S_s3", "A_s3", "NB_s3"),
    ]:
        print(f"\n  Scenario {label}:")
        print(f"    C+M  : {(out['C']+out['M']).mean():>15,.0f} CHF (mean)")
        print(f"    T    : {out[t_col].mean():>15,.0f} CHF (mean)")
        print(f"    R    : {out[r_col].mean():>15,.0f} CHF (mean)")
        print(f"    S    : {out[s_col].mean():>15,.0f} CHF (mean)")
        print(f"    NB   : {out[nb_col].mean():>15,.0f} CHF (mean)  "
              f"[{out[nb_col].min():,.0f} – {out[nb_col].max():,.0f}]")
    top5 = out.nlargest(5, "NB_s2")[["ID_new", "NB_s1", "NB_s2", "NB_s3"]]
    print(f"\n  Top 5 by NB_s2:\n{top5.to_string(index=False)}")

    return out


def aggregate_costs():
    # Construction costs
    c_construction = gpd.read_file(r"data/costs/construction.gpkg")
    # Maintenance costs
    c_maintenance = gpd.read_file(r"data/costs/maintenance.gpkg")
    # Access time costs
    c_acces_time = pd.read_csv(r"data/costs/local_accessibility.csv")
    c_acces_time = c_acces_time[["ID_develop", "local_s1", "local_s2", "local_s3"]]
    # Import travel time costs
    c_tt = pd.read_csv(r"data/costs/traveltime_savings.csv")
    # Import externalities
    c_externalities = gpd.read_file(r"data/costs/externalities.gpkg")


    # Rename columns to simplify further steps
    c_acces_time = c_acces_time.rename(columns={'ID_develop': 'ID_new'})
    c_tt = c_tt.rename(columns={'development': 'ID_new'}).drop(columns=["Unnamed: 0"])

    # Find common values
    common_values = set(c_construction["ID_new"]).intersection(c_acces_time["ID_new"]).intersection(
        c_tt["ID_new"]).intersection(c_externalities["ID_new"])
    print(f"Number of developments: {len(common_values)}")

    # Merge construction costs and maintenance costs
    # c_construction = c_construction.merge(c_maintenance, how='inner', on='ID_new')
    # Add access time costs
    # total_costs = c_construction.merge(c_acces_time, how='inner', on='ID_new')
    # Add travel time
    # total_costs = total_costs.merge(c_tt, how='inner', on='ID_new')
    # Add externalities costs
    # total_costs = total_costs.merge(c_externalities, how='inner', on='ID_new')


    # geom_id_map = c_maintenance.drop("maintenance",axis=1)
    total_costs = c_construction.drop("geometry", axis=1).merge(c_maintenance.drop(["geometry"], axis=1), how='inner',
                                                                on='ID_new')
    # Add acccess time costs
    total_costs = total_costs.merge(c_acces_time, how='inner', on='ID_new')
    # Add travel time
    total_costs = total_costs.merge(c_tt, how='inner', on='ID_new')
    # Add externalities costs
    total_costs = total_costs.merge(c_externalities.drop("geometry", axis=1), how='inner', on='ID_new')

    total_costs = total_costs[['ID_new', 'building_costs',
                               'local_s1', 'local_s2', 'local_s3', 'tt_low', 'tt_medium', 'tt_high', 'climate_cost',
                               'land_realloc', 'nature', "maintenance"]]
    cost_columns = ['building_costs', 'climate_cost', 'land_realloc',
                    'nature', "maintenance"]

    # Multiply the values in these columns by -1
    for column in cost_columns:
        total_costs[column] = total_costs[column] * -1

    # Compute costs of externalities
    total_costs["externalities_s1"] = total_costs["climate_cost"] + total_costs["land_realloc"] + total_costs[
        "nature"]
    total_costs["externalities_s2"] = total_costs["climate_cost"] + total_costs["land_realloc"] + total_costs[
        "nature"]
    total_costs["externalities_s3"] = total_costs["climate_cost"] + total_costs["land_realloc"] + total_costs[
        "nature"]
    total_costs["construction_maintenance"] = total_costs["building_costs"] + total_costs["maintenance"]

    # Sum externality costs
    # total_costs["externalities"] = total_costs['climate_cost'] + total_costs['land_realloc'] + total_costs['nature']
    # Compute net benefit for each development
    total_costs["total_low"] = total_costs[["construction_maintenance", "local_s2", "tt_low", "externalities_s2"]].sum(
        axis=1)
    total_costs["total_medium"] = total_costs[
        ["construction_maintenance", "local_s1", "tt_medium", "externalities_s1"]].sum(
        axis=1)
    total_costs["total_high"] = total_costs[
        ["construction_maintenance", "local_s3", "tt_high", "externalities_s3"]].sum(
        axis=1)

    # print(total_costs.sort_values(by="total_medium", ascending=False).head(7).to_string())

    # Filter dataframe columns to store the data as csv
    total_costs[["ID_new", "total_low", "total_medium", "total_high"]].to_csv(r"data/costs/total_costs.csv")

    # Save Results a geodata
    # Map point geometries from development candidate centroids
    _dc = gpd.read_file(r"data/Network/processed/development_candidates.gpkg")[["ID_new", "geometry"]]
    _dc["geometry"] = _dc.geometry.centroid
    total_costs = total_costs.merge(right=_dc, how="left", on="ID_new")
    total_costs = gpd.GeoDataFrame(total_costs, geometry="geometry")

    # Store as file
    gpd.GeoDataFrame(total_costs).to_file(r"data/costs/total_costs.gpkg")



def stack_tif_files(var):
    # List of your TIFF file paths
    tiff_files = [f"/s1_{var}.tif", f"/s2_{var}.tif", f"/s3_{var}.tif"]

    # Open the first file to retrieve the metadata
    with rasterio.open(r"data/independent_variable/processed/scenario" + tiff_files[0]) as src0:
        meta = src0.meta

    # Update metadata to reflect the number of layers
    meta.update(count=len(tiff_files))

    out_fp = fr"data/independent_variable/processed/scenario/scen_{var}.tif"
    # Read each layer and write it to stack
    with rasterio.open(out_fp, 'w', **meta) as dst:
        for id, layer in enumerate(tiff_files, start=1):
            with rasterio.open(r"data/independent_variable/processed/scenario" + layer) as src1:
                dst.write_band(id, src1.read(1))



def GetCommunePopulation(y0):  # We find population of each commune.
    rawpop = pd.read_excel('data/_basic_data/KTZH_00000127_00001245.xlsx', sheet_name='Gemeinden', header=None)
    rawpop.columns = rawpop.iloc[5]
    rawpop = rawpop.drop([0, 1, 2, 3, 4, 5, 6])
    pop = pd.DataFrame(data=rawpop, columns=['BFS-NR  ', 'TOTAL_' + str(y0) + '  ']).sort_values(by='BFS-NR  ')
    popvec = np.array(pop['TOTAL_' + str(y0) + '  '])
    return popvec


def GetCommuneEmployment(y0):  # we find employment in each commune.
    rawjob = pd.read_excel('data/_basic_data/KANTON_ZUERICH_596.xlsx')
    rawjob = rawjob.loc[(rawjob['INDIKATOR_JAHR'] == y0) & (rawjob['BFS_NR'] > 0) & (rawjob['BFS_NR'] != 291)]

    # rawjob=rawjob.loc[(rawjob['INDIKATOR_JAHR']==y0)&(rawjob['BFS_NR']>0)&(rawjob['BFS_NR']!=291)]
    job = pd.DataFrame(data=rawjob, columns=['BFS_NR', 'INDIKATOR_VALUE']).sort_values(by='BFS_NR')
    jobvec = np.array(job['INDIKATOR_VALUE'])
    return jobvec


def GetCyclingDemandPerCommune(tau=1.0):
    # No velo mode in KTZH_00001982_00003903.xlsx (only miv/oev).
    # Use miv as a structural proxy: same zone-pair propensities, scaled by tau.
    y0 = 2018
    rawod = pd.read_excel('data/_basic_data/KTZH_00001982_00003903.xlsx')
    communalOD = rawod.loc[
        (rawod['jahr'] == y0) &
        (rawod['kategorie'] == 'Verkehrsaufkommen') &
        (rawod['verkehrsmittel'] == 'miv')
    ].copy()
    # Keep intrazonal trips — short bike trips within a commune are significant
    communalOD['wert'] = communalOD['wert'] * tau
    return communalOD

def GetCyclingOD(voronoi_gdf):
    od = pd.read_csv('data/OD/od_matrix_zh_cycling.csv')

    origin_gdf = gpd.GeoDataFrame(od, geometry=gpd.points_from_xy(od.x_wohn,   od.y_wohn),   crs="EPSG:2056")
    dest_gdf   = gpd.GeoDataFrame(od, geometry=gpd.points_from_xy(od.x_arbeit, od.y_arbeit), crs="EPSG:2056")

    origin_gdf = gpd.sjoin(origin_gdf, voronoi_gdf[['ID_point', 'geometry']], how='left', predicate='within')
    dest_gdf   = gpd.sjoin(dest_gdf,   voronoi_gdf[['ID_point', 'geometry']], how='left', predicate='within')

    od['zone_origin'] = origin_gdf['ID_point'].values
    od['zone_dest']   = dest_gdf['ID_point'].values

    od_matrix = (od.dropna(subset=['zone_origin', 'zone_dest'])
                   .groupby(['zone_origin', 'zone_dest'])['commuters_cycling']
                   .sum().reset_index())

    od_matrix.to_csv('data/traffic_flow/od/od_matrix_statusquo.csv', index=False)
    return od_matrix



def GetODMatrix(od):
    od_ext = od.loc[(od['quelle_code'] > 9999) | (od[
                                                      'ziel_code'] > 9999)]  # here we separate the parts of the od matrix that are outside the canton. We can add them later.
    od_int = od.loc[(od['quelle_code'] < 9999) & (od['ziel_code'] < 9999)]
    odmat = od_int.pivot(index='quelle_code', columns='ziel_code', values='wert')
    return odmat


def GetCommuneShapes(raster_path):  # todo this might be unnecessary if you already have these shapes.
    communalraw = gpd.read_file(r"data/_basic_data/Gemeindegrenzen/UP_GEMEINDEN_F.shp")
    communalraw = communalraw.loc[(communalraw['ART_TEXT'] == 'Gemeinde')]
    communedf = gpd.GeoDataFrame(data=communalraw, geometry=communalraw['geometry'], columns=['BFS', 'GEMEINDENA'],
                                 crs="epsg:2056").sort_values(by='BFS')

    # Read the reference TIFF file
    with rasterio.open(raster_path) as src:
        profile = src.profile
        profile.update(count=1)
        crs = src.crs

    # Rasterize
    with rasterio.open('data/_basic_data/Gemeindegrenzen/gemeinde_zh.tif', 'w', **profile) as dst:
        rasterized_image = rasterize(
            [(shape, value) for shape, value in zip(communedf.geometry, communedf['BFS'])],
            out_shape=(src.height, src.width),
            transform=src.transform,
            fill=0,
            all_touched=False,
            dtype=rasterio.int32
        )
        dst.write(rasterized_image, 1)

    # Convert the rasterized image to a numpy array
    commune_raster = np.array(rasterized_image)

    return commune_raster, communedf


def GetVoronoiOD():
    # Import the required data or define the path to access it
    voronoi_tif_path = r"data/Network/travel_time/source_id_raster.tif"
    voronoidf = gpd.read_file(r"data/Network/travel_time/Voronoi_statusquo.gpkg")

    scen_empl_path = r"data/independent_variable/processed/scenario/scen_empl.tif"
    scen_pop_path = r"data/independent_variable/processed/scenario/scen_pop.tif"

    # define dev (=ID of the polygons of a development)
    dev = 0

    # Get voronoidf crs
    print(voronoidf.crs)

    #todo: check scenarios again
    popvec = GetCommunePopulation(y0="2021")
    jobvec = GetCommuneEmployment(y0=2021)

    # Cycling gravity: uniform unit flow (no observed OD data available)
    # cout_r is computed later as odmat / outer(pop, empl)
    # For cycling, set odmat = outer(pop, empl) so cout_r = 1 everywhere,
    # then the actual scaling by scenario pop/empl happens in Step 3
    cycling_mode_share = 0.03  # ~3% of commute trips by bike — adjust to your study area
    odmat = pd.DataFrame(
        cycling_mode_share * np.outer(popvec, jobvec),
        index=popvec.index,
        columns=jobvec.index
    )

    # This function returns a np array of raster data storing the bfs number of the commune in each cell
    commune_raster, commune_df = GetCommuneShapes(raster_path=voronoi_tif_path)

    if jobvec.shape[0] != odmat.shape[0]:
        print(
            "Error: The number of communes in the OD matrix and the number of communes in the employment data do not match.")
    # com_idx = np.unique(od['quelle_code']) # previously od_mat
    # 1. Define a new raster file that stores the Commune's BFS ID as cell value
    # Think if new band or new tif makes more sense
    # using communeShapes

    # I guess here iterate over all developments
    # voronoidf = voronoidf.loc[(voronoidf['ID_develop'] == dev)] # Work with temp gdf of voronoi
    # If possible simplify all the amount of developments

    # Open scenario (medium) raster data    (low = band 2, high = band 3)
    with rasterio.open(scen_pop_path) as src:
        # Read the raster into a NumPy array (assuming you want the first band)
        #todo check order
        scen_pop_medium_tif = src.read(1)
        scen_pop_low_tif = src.read(2)
        scen_pop_high_tif = src.read(3)

    with rasterio.open(scen_empl_path) as src:
        # Read the raster into a NumPy array (assuming you want the first band)
        scen_empl_medium_tif = src.read(1)
        scen_empl_low_tif = src.read(2)
        scen_empl_high_tif = src.read(3)

    # Open status quo
    with rasterio.open(r"data/independent_variable/processed/raw/empl20.tif") as src:
        scen_empl_20_tif = src.read(1)

    with rasterio.open(r"data/independent_variable/processed/raw/pop20.tif") as src:
        scen_pop_20_tif = src.read(1)

    # Open voronoi raster data
    with rasterio.open(voronoi_tif_path) as src:
        # Read the raster into a NumPy array (assuming you want the first band)
        voronoi_tif = src.read(1)
    unique_voronoi_id = np.sort(np.unique(voronoi_tif))
    # vor_idx = unique_voronoi_id.tolist()
    vor_idx = unique_voronoi_id.size
    # vor_idx = voronoidf['ID_point'].sort_by('ID_point')

    # Get voronoi tif boundaries and filter the commune_df that lay in it or touch it
    # Get the bounds of the voronoi tif
    bounds = src.bounds
    # Get the commune_df that are within the bounds
    commune_df_filtered = commune_df.cx[bounds.left:bounds.right, bounds.bottom:bounds.top]
    # Get "BFS" value of the commune_df_filtered that are within the bounds
    commune_df_filtered = commune_df_filtered["BFS"].to_numpy()

    # Do a copy of odmat and filter the rows and columns that are not in commune_df_filtered
    odmat_frame = odmat.loc[commune_df_filtered, commune_df_filtered]

    # od_mn = np.zeros([len(vor_idx),len(vor_idx)])
    od_mn = np.zeros([vor_idx, vor_idx])

    # Assume vectorized functions are defined for the below operations
    def compute_cont_r(odmat, popvec, jobvec):
        # Convert popvec and jobvec to 2D arrays for broadcasting
        pop_matrix = np.array(popvec)[:, np.newaxis]
        job_matrix = np.array(jobvec)[np.newaxis, :]

        # Ensure odmat is a NumPy array
        odmat = np.array(odmat)

        # Perform the vectorized operation
        cont_r = odmat / (pop_matrix * job_matrix)
        return cont_r

    def compute_cont_v(cont_r, pop_m, job_n):
        # Sum over the cont_r matrix, multiply by pop_m and job_n
        cont_v = np.sum(cont_r)
        return cont_v

    # Step 1: generate unit_flow matrix from each commune to each other commune
    cout_r = odmat / np.outer(popvec, jobvec)

    # Step 2: Get all pairs of combinations from communes to polygons
    unique_commune_id = np.sort(np.unique(commune_raster))
    pairs = pd.DataFrame(columns=['commune_id', 'voronoi_id'])
    pop_empl = pd.DataFrame(columns=['commune_id', 'voronoi_id', "empl", "pop"])

    for i in tqdm(unique_voronoi_id, desc='Processing Voronoi IDs'):
        # Get the voronoi raster
        mask_voronoi = voronoi_tif == i
        for j in unique_commune_id:
            if j > 0:
                # Get the commune raster
                mask_commune = commune_raster == j
                # Combined mask
                mask = mask_commune & mask_voronoi
                # Check if there are overlaying values
                if np.nansum(mask) > 0:
                    # pairs = pairs.append({'commune_id': j, 'voronoi_id': i}, ignore_index=True)
                    temp = pd.Series({'commune_id': j, 'voronoi_id': i})
                    pairs = gpd.GeoDataFrame(
                        pd.concat([pairs, pd.DataFrame(temp).T], ignore_index=True))

                    # Get the population and employment values for multiple scenarios
                    pop20 = scen_pop_20_tif[mask]
                    empl20 = scen_empl_20_tif[mask]
                    pop_low = scen_pop_low_tif[mask]
                    empl_low = scen_empl_low_tif[mask]
                    pop_medium = scen_pop_medium_tif[mask]
                    empl_medium = scen_empl_medium_tif[mask]
                    pop_high = scen_pop_high_tif[mask]
                    empl_high = scen_empl_high_tif[mask]

                    temp = pd.Series({'commune_id': j, 'voronoi_id': i,
                                      'pop_20': np.nansum(pop20), 'empl_20': np.nansum(empl20),
                                      'pop_low': np.nansum(pop_low), 'empl_low': np.nansum(empl_low),
                                      'pop_medium': np.nansum(pop_medium), 'empl_medium': np.nansum(empl_medium),
                                      'pop_high': np.nansum(pop_high), 'empl_high': np.nansum(empl_high)})
                    pop_empl = gpd.GeoDataFrame(
                        pd.concat([pop_empl, pd.DataFrame(temp).T], ignore_index=True))
                    # pop_empl = pop_empl.append({'commune_id': j, 'voronoi_id': i,
                    #                            'pop_20': np.nansum(pop20), 'empl_20': np.nansum(empl20),
                    #                            'pop_low': np.nansum(pop_low), 'empl_low': np.nansum(empl_low),
                    #                            'pop_medium': np.nansum(pop_medium), 'empl_medium': np.nansum(empl_medium),
                    #                            'pop_high': np.nansum(pop_high), 'empl_high': np.nansum(empl_high)},
                    #                            ignore_index=True)

            else:
                continue

    # Print array shapes to compare
    print(f"cout_r: {cout_r.shape}")
    print(f"pairs: {pairs.shape}")
    print(f"pop_empl: {pop_empl.shape}")

    # Step 3 complete exploded matrix
    # Initialize the OD matrix DataFrame with zeros or NaNs
    tuples = list(zip(pairs['voronoi_id'], pairs['commune_id']))
    multi_index = pd.MultiIndex.from_tuples(tuples, names=['voronoi_id', 'commune_id'])
    temp_df = pd.DataFrame(index=multi_index, columns=multi_index).fillna(0).to_numpy('float')
    od_matrix = pd.DataFrame(data=temp_df, index=multi_index, columns=multi_index)

    # Handle raster without values
    # Drop pairs with 0 pop or empl

    set_id_destination = [col[1] for col in od_matrix.columns]

    # Get unique values from the second level of the index
    unique_values_second_index = od_matrix.index.get_level_values(1).unique()

    # Iterate over each cell in the od_matrix to fill it with corresponding values from other_matrix
    for commune_id_origin in unique_values_second_index:
        # for (polygon_id_o, commune_id_o), _ in tqdm(od_matrix.index.to_series().iteritems(), desc='Allocating unit_values to OD matrix'):

        # Extract the row for commune_id_o
        row_values = cout_r.loc[commune_id_origin]

        # Use the valid columns to extract values
        extracted_values = row_values[set_id_destination].to_numpy('float')

        # Create a boolean mask for rows where the second element of the index matches commune_id_o
        mask = od_matrix.index.get_level_values(1) == commune_id_origin

        # Update the rows in od_matrix where the mask is True
        od_matrix.loc[mask] = extracted_values  # .to_numpy('float')

    ####################################################################################################3
    # todo Filling happens here

    # Check for scenario based on column names in pop_empl
    # Sceanrio are defined like pop_XX and empl_XX get a list of all these endings (only XX)
    # Get the column names of pop_empl
    pop_empl_columns = pop_empl.columns
    # Get the column names that end with XX
    pop_empl_scenarios = [col.split("_")[1] for col in pop_empl_columns if col.startswith("pop_")]
    print(pop_empl_scenarios)

    # SEt index of df to access its single components
    pop_empl = pop_empl.set_index(['voronoi_id', 'commune_id'])

    # for each of these scenarios make an own copy of od_matrix named od_matrix+scen
    for scen in pop_empl_scenarios:
        print(f"Processing scenario {scen}")
        od_matrix_temp = od_matrix.copy()

        for polygon_id, row in tqdm(pop_empl.iterrows(), desc='Allocating pop and empl to OD matrix'):
            # Multiply all values in the row/column
            od_matrix_temp.loc[polygon_id] *= row[f'pop_{scen}']
            od_matrix_temp.loc[:, polygon_id] *= row[f'empl_{scen}']

        # Step 4: Group the OD matrix by voronoi_id (rows then columns)
        od_matrix_reset = od_matrix_temp.reset_index().sort_values('voronoi_id')
        if isinstance(od_matrix_reset.columns, pd.MultiIndex):
            od_matrix_reset = od_matrix_reset.sort_index(axis=1)
        od_grouped = od_matrix_reset.groupby('voronoi_id', sort=False).sum()
        od_grouped = od_grouped.T.sort_index().groupby('voronoi_id', sort=False).sum().T

        # Drop column commune_id
        od_grouped = od_grouped.drop(columns='commune_id')

        # Set diagonal values to 0
        temp_sum = od_grouped.sum().sum()
        arr = od_grouped.to_numpy().copy()
        np.fill_diagonal(arr, 0)
        od_grouped = pd.DataFrame(arr, index=od_grouped.index, columns=od_grouped.columns)
        # Compute the sum after changing the diagonal
        temp_sum2 = od_grouped.sum().sum()
        # Print difference
        print(f"Sum of OD matrix before {temp_sum} and after {temp_sum2} removing diagonal values")

        # Save pd df to csv
        od_grouped.to_csv(fr"data/traffic_flow/od/od_matrix_{scen}.csv")
        # odmat.to_csv(r"data/traffic_flow/od/od_matrix_raw.csv")

        # Print sum of all values in od df
        # Sum over all values in pd df
        sum_com = odmat.sum().sum()
        sum_poly = od_grouped.sum().sum()
        sum_com_frame = odmat_frame.sum().sum()
        print(
            f"Total trips before {sum_com_frame} ({odmat_frame.shape} communes) and after {sum_poly} ({od_grouped.shape} polygons)")
        print(
            f"Total trips before {sum_com} ({odmat.shape} communes) and after {sum_poly} ({od_grouped.shape} polygons)")

        # Sum all columns of od_grouped
        origin = od_grouped.sum(axis=1).reset_index()
        origin.columns = ["voronoi_id", "origin"]
        # Sum all rows of od_grouped
        destination = od_grouped.sum(axis=0)
        destination = destination.reset_index()
        destination.columns = ["voronoi_id", "destination"]

        # merge origin and destination to voronoidf based on voronoi_id
        # Make a copy of voronoidf
        voronoidf_temp = voronoidf.copy()
        voronoidf_temp = voronoidf_temp.merge(origin, how='left', left_on='ID_point', right_on='voronoi_id')
        voronoidf_temp = voronoidf_temp.merge(destination, how='left', left_on='ID_point', right_on='voronoi_id')
        voronoidf_temp.to_file(fr"data/traffic_flow/od/OD_voronoidf_{scen}.gpkg", driver="GPKG")
        del od_grouped, od_matrix_reset, od_matrix_temp, voronoidf_temp

        # Same for odmat and commune_df
        if scen == "20":
            origin_commune = odmat_frame.sum(axis=1).reset_index()
            origin_commune.columns = ["commune_id", "origin"]
            destination_commune = odmat_frame.sum(axis=0).reset_index()
            destination_commune.columns = ["commune_id", "destination"]
            commune_df = commune_df.merge(origin_commune, how='left', left_on='BFS', right_on='commune_id')
            commune_df = commune_df.merge(destination_commune, how='left', left_on='BFS', right_on='commune_id')
            commune_df.to_file(r"data/traffic_flow/od/OD_commune_filtered.gpkg", driver="GPKG")

    return


def GetVoronoiOD_multi():
    voronoi_tif_path = r"data/Network/travel_time/source_id_raster.tif"
    scen_empl_path = r"data/independent_variable/processed/scenario/scen_empl.tif"
    scen_pop_path = r"data/independent_variable/processed/scenario/scen_pop.tif"

    popvec = GetCommunePopulation(y0="2021")
    jobvec = GetCommuneEmployment(y0=2021)
    od = GetCyclingDemandPerCommune(tau=1.0)  # TODO: calibrate tau
    odmat = GetODMatrix(od)

    # This function returns a np array of raster data storing the bfs number of the commune in each cell
    commune_raster, commune_df = GetCommuneShapes(raster_path=voronoi_tif_path)

    if jobvec.shape[0] != odmat.shape[0]:
        print(
            "Error: The number of communes in the OD matrix and the number of communes in the employment data do not match.")

    # Open scenario (medium) raster data    (low = band 2, high = band 3)
    with rasterio.open(scen_pop_path) as src:
        # Read the raster into a NumPy array (assuming you want the first band)
        scen_pop_medium_tif = src.read(1)
        scen_pop_low_tif = src.read(2)
        scen_pop_high_tif = src.read(3)

    with rasterio.open(scen_empl_path) as src:
        # Read the raster into a NumPy array (assuming you want the first band)
        scen_empl_medium_tif = src.read(1)
        scen_empl_low_tif = src.read(2)
        scen_empl_high_tif = src.read(3)

    # Step 1: generate unit_flow matrix from each commune to each other commune
    cout_r = odmat / np.outer(popvec, jobvec)

    # Directory path to developments
    directory_path = "data/Network/travel_time/developments/"

    # List to hold extracted values
    xx_values = []

    # Iterate through files in the directory
    for filename in os.listdir(directory_path):
        # Check if the filename matches the pattern 'devXX_source_id_raster.tif'
        match = re.match(r'dev(\d+)_source_id_raster\.tif', filename)
        if match:
            # Extract XX value and add to the list
            xx = match.group(1)
            xx_values.append(xx)

    # Convert values to integers if needed; restrict to corridor + border developments
    xx_values = [int(xx) for xx in xx_values]
    _cands = gpd.read_file("data/Network/processed/development_candidates.gpkg")
    _corridor_ids = set(_cands[_cands["within_corridor"] | _cands["on_border"]]["ID_new"].tolist())
    xx_values = [xx for xx in xx_values if xx in _corridor_ids]
    print(f"{len(xx_values)} corridor developments to process")

    os.makedirs('data/traffic_flow/od/developments', exist_ok=True)

    for xx in tqdm(xx_values, desc='Processing Voronoi IDs'):
        # Skip if all three scenario OD matrices already exist for this development
        _od_paths = [
            f"data/traffic_flow/od/developments/cycling_od_matrix_dev{xx}_low.csv",
            f"data/traffic_flow/od/developments/cycling_od_matrix_dev{xx}_medium.csv",
            f"data/traffic_flow/od/developments/cycling_od_matrix_dev{xx}_high.csv",
        ]
        if all(os.path.exists(p) for p in _od_paths):
            continue

        # Construct the file path
        file_path = f"{directory_path}dev{xx}_source_id_raster.tif"

        # Open the file with rasterio
        with rasterio.open(file_path) as src:
            # Read the raster data
            voronoi_tif = src.read(1)

        # Step 2 (vectorized): aggregate pop/empl per (voronoi_id, commune_id) pair.
        # Replaces the O(V×C) nested loop + pd.concat with a single groupby.
        flat_voronoi = voronoi_tif.ravel().astype(float)
        flat_commune = commune_raster.ravel().astype(float)
        valid = (flat_commune > 0) & np.isfinite(flat_voronoi) & (flat_voronoi != -1.0)

        if valid.sum() == 0:
            continue

        pop_empl = pd.DataFrame({
            'voronoi_id':  flat_voronoi[valid].astype(int),
            'commune_id':  flat_commune[valid].astype(int),
            'pop_low':     np.nan_to_num(scen_pop_low_tif.ravel()[valid].astype(float)),
            'empl_low':    np.nan_to_num(scen_empl_low_tif.ravel()[valid].astype(float)),
            'pop_medium':  np.nan_to_num(scen_pop_medium_tif.ravel()[valid].astype(float)),
            'empl_medium': np.nan_to_num(scen_empl_medium_tif.ravel()[valid].astype(float)),
            'pop_high':    np.nan_to_num(scen_pop_high_tif.ravel()[valid].astype(float)),
            'empl_high':   np.nan_to_num(scen_empl_high_tif.ravel()[valid].astype(float)),
        }).groupby(['voronoi_id', 'commune_id']).sum()

        # Step 3: build MultiIndex OD matrix from the (voronoi_id, commune_id) pairs
        multi_index = pd.MultiIndex.from_tuples(
            pop_empl.index.tolist(), names=['voronoi_id', 'commune_id']
        )
        n = len(multi_index)

        # Fill unit-flow values from cout_r vectorized: od[i,j] = cout_r[c_i, c_j]
        row_communes = multi_index.get_level_values(1)
        col_communes = multi_index.get_level_values(1)
        od_values = cout_r.reindex(
            index=row_communes, columns=col_communes, fill_value=0.0
        ).to_numpy(dtype=float)
        od_matrix = pd.DataFrame(od_values, index=multi_index, columns=multi_index)

        # Step 4: per-scenario scaling and groupby (vectorized outer product)
        for scen in ['low', 'medium', 'high']:
            pop_s  = pop_empl[f'pop_{scen}'].reindex(multi_index).fillna(0).to_numpy(float)
            empl_s = pop_empl[f'empl_{scen}'].reindex(multi_index).fillna(0).to_numpy(float)

            # od[i,j] *= pop[i] * empl[j]
            scaled = od_matrix.values * pop_s[:, np.newaxis] * empl_s[np.newaxis, :]
            od_matrix_temp = pd.DataFrame(scaled, index=multi_index, columns=multi_index)

            od_matrix_reset = od_matrix_temp.reset_index().sort_values('voronoi_id')
            if isinstance(od_matrix_reset.columns, pd.MultiIndex):
                od_matrix_reset = od_matrix_reset.sort_index(axis=1)
            od_grouped = od_matrix_reset.groupby('voronoi_id', sort=False).sum()
            od_grouped = od_grouped.T.sort_index().groupby('voronoi_id', sort=False).sum().T
            od_grouped = od_grouped.drop(columns='commune_id', errors='ignore')

            arr = od_grouped.to_numpy().copy()
            np.fill_diagonal(arr, 0)
            od_grouped = pd.DataFrame(arr, index=od_grouped.index, columns=od_grouped.columns)

            od_grouped.to_csv(fr"data/traffic_flow/od/developments/cycling_od_matrix_dev{xx}_{scen}.csv")
            del od_grouped, od_matrix_reset, od_matrix_temp, scaled
        del od_matrix, pop_empl, od_values, multi_index
        import gc; gc.collect()

    return


def link_traffic_to_map():
    # Import travel flows from matrix to df, no index, set column name to flow
    # flow = pd.read_csv(r"data/traffic_flow/Xi_sum.csv", header=None, index_col=False)
    flow = pd.read_csv(r"data/traffic_flow/developments/D_i/Xi_sum_status_quo_20.csv", header=None, index_col=False)
    flow.columns = ['flow']
    print(flow.head(10).to_string())

    # Import data with links
    edges = gpd.read_file(r"data/Network/processed/edges_with_attribute.gpkg")
    print(edges.head(10).to_string())

    # Compare lenght of dataframes
    print(f"Length of edges df: {len(edges)}")
    print(f"Length of flow df: {len(flow)}")

    # Sort edges by edge_ID
    edges["ID_edge"] = edges["ID_edge"].astype(int)
    edges = edges.sort_values(by=['ID_edge'])

    # Add flow column to edges df
    edges['flow'] = flow['flow']

    print(edges.head(10).to_string())

    # Only keep column capacity, flow and geometry
    edges = edges[['ID_edge', 'geometry', 'flow']]
    # Safe file
    edges.to_file(r"data/Network/processed/edges_only_flow.gpkg")

    # Compare values to calibrate to tau value when creating the OD matrix
    # Edge ID 94 -> Tagesverkehr 3028 (DTV 54014)
    # Edge ID 95 -> Tagesverkehr  3034 (DTV 53867)
    # Edge ID 88 -> Tagesverkehr  1103 (DTV 18852)
    # Edge ID 90 -> Tagesverkehr 1087 (DTV 18547)
    # Print a table comparing the flow (edges["flow"] values in edges for ID mentioned above and the Tagesverkehr values

    # print(f"Link 94 - modelled flow: {edges.loc[edges['ID_edge'] == 94, 'flow'].iloc[0]} and measured flow: 3028")
    # print(f"Link 95 - modelled flow: {edges.loc[edges['ID_edge'] == 95, 'flow'].iloc[0]} and measured flow: 3034")
    # print(f"Link 88 - modelled flow: {edges.loc[edges['ID_edge'] == 88, 'flow'].iloc[0]} and measured flow: 1103")
    # print(f"Link 90 - modelled flow: {edges.loc[edges['ID_edge'] == 90, 'flow'].iloc[0]} and measured flow: 1087")


def convert_data_to_input(points, edges):
    # Set crs for points and edges to epsg:2056
    points = points.set_crs("epsg:2056", allow_override=True)
    edges = edges.set_crs("epsg:2056", allow_override=True)
    # Print crs of points and edges
    # print(f"Points crs: {points.crs}")
    # print(f"Edges crs: {edges.crs}")

    # Change "corridor_border" to False if "within_corridor" is True
    if "within_corridor" in points.columns and "on_corridor_border" in points.columns:
        points.loc[points["within_corridor"] == True, "on_corridor_border"] = False

    # Define which nodes generate traffic in the model.
    # Restrict to the thinned access points so the OD matrix stays small.
    _ap_ids = _get_ap_ids()
    if _ap_ids:
        pt_ids = points["ID_point"].astype(int) if "ID_point" in points.columns else points.index.astype(int)
        points["generate_traffic"] = pt_ids.isin(_ap_ids)
    else:
        # Fallback: all corridor / border nodes
        points["generate_traffic"] = (
            points['within_corridor'].astype(str).isin(['1', 'True', 'true']) |
            points['on_corridor_border'].astype(str).isin(['1', 'True', 'true'])
        )

    #######################################################################################################################
    # Store values as needed for the model

    # Assert nodes and edges are sorted by ID
    # points["ID_point"] = points["ID_point"].astype(int)
    # points = points.sort_values(by=["ID_point"])
    points.index = points.index.astype(int)
    points = points.sort_index()
    edges["ID_edge"] = edges["ID_edge"].astype(int)
    edges = edges.sort_values(by=["ID_edge"])

    # Nodes: store dict of coordinates of all nodes nodes = [[x1, y1], [x2, y2], ...] <class 'numpy.ndarray'>
    nodes_lv95 = points[["geometry"]].to_numpy()
    # Same with coordinates converted to wgs84
    nodes_wgs84 = points[["geometry"]].to_crs("epsg:4326").to_numpy()

    # Edges: store dict of coordinates of all edges links = [[id_start_node_1, id_end_node_1], [id_start_node_2, id_end_node_2], ...] <class 'numpy.ndarray'>
    links = edges[["start", "end"]].to_numpy(int)

    # Length of edges stored a array link_length_i = [[length_1], [length_2], ...] <class 'numpy.ndarray'>
    link_length_i = edges["geometry"].length.to_numpy(float)

    # Number of edges
    nlinks = len(edges)

    # Travel time on each link assuming free flow speed
    # Get edge length
    edges["length"] = edges["geometry"].length / 1000  # in kilometers
    # Calculate free flow travel time on all edges
    edges["fftt_i"] = edges["length"] / pd.to_numeric(edges["ffs"])  # in hours
    # Store these values in a dict as fftt_i = [[fftt_1], [fftt_2], ...] <class 'numpy.ndarray'>
    fftt_i = edges[["fftt_i"]].to_numpy(float)

    # Capacity on each link
    # Store these values in a dict as capacity_i = [[capacity_1], [capacity_2], ...] <class 'numpy.ndarray'>
    Xmax_i = edges[["capacity"]].to_numpy(float)

    # Store same alpha and gamma for all links as alpha_i = [[alpha_1], [alpha_2], ...] <class 'numpy.ndarray'>
    alpha = 0.25
    gamma = 2.4
    alpha_i = np.tile(alpha, Xmax_i.shape)
    gamma_i = np.tile(gamma, Xmax_i.shape)

    par = {"fftt_i": fftt_i, "Xmax_i": Xmax_i, "alpha_i": alpha_i, "gamma_i": gamma_i}

    return nodes_lv95, nodes_wgs84, links, link_length_i, nlinks, par


def get_nw_data(OD_matrix, points, voronoi_gdf, edges):
    # Normalise: ensure ID_point is always a column (it may be the index when
    # loaded directly from points_with_attribute.gpkg)
    if "ID_point" not in points.columns:
        points = points.copy()
        points["ID_point"] = points.index.astype(int)

    # Adapt OD matrix
    # nodes within perimeter and on border
    ####################################################
    ### Define zones
    ### Check for 2050

    # Filter points to only keep those within the corridor or on border
    # points_in = points[(points["within_corridor"] == True) | (points["on_corridor_border"] == True)]
    points_in = points[
        points['within_corridor'].astype(str).isin(['1', 'True', 'true']) |
        points['on_corridor_border'].astype(str).isin(['1', 'True', 'true'])]

    # Further restrict to thinned access points if the file exists
    _ap_ids = _get_ap_ids()
    if _ap_ids:
        points_in = points_in[points_in["ID_point"].astype(int).isin(_ap_ids)]

    # Get common ID_point and voronoi_ID as list
    common_ID = list(set(pd.to_numeric(points_in["ID_point"])) & set(voronoi_gdf["ID_point"]))

    # print(f"\n\n\n max value in ID_point: {max(voronoi_gdf['ID_point'])}")
    # print(f"Point in polygon and with voronoi: {len(common_ID)}")

    # new column "generate_demand" where ID_point is in common_ID
    points["generate_demand"] = points["ID_point"].astype(int).isin(common_ID)

    # Filter OD matrix to only keep common ID in rows and columns
    # Convert ID elements to the appropriate type if necessary
    common_ID = [int(id) for id in common_ID]
    OD_matrix.index = OD_matrix.index.map(lambda x: int(float(x)))
    OD_matrix.columns = OD_matrix.columns.map(lambda x: int(float(x)))

    od_ids = set(OD_matrix.index) & set(OD_matrix.columns)
    common_ID = [id for id in common_ID if id in od_ids]
    OD_matrix = OD_matrix.loc[common_ID, common_ID]
    # print(f"Shape OD matrix: {OD_matrix.shape}")

    # Re-sync generate_demand after OD filter so demand_nodes matches nOD dimensions
    points["generate_demand"] = points["ID_point"].astype(int).isin(common_ID)

    # flatten OD matrix to 1D array as D_od
    D_od = OD_matrix.to_numpy().flatten()

    # Get the amount of values in the OD matrix as nOD
    nOD = len(D_od)
    # print(nOD)

    # Map the single zones of the OD to actual nodes in the network
    # nodes within perimeter and on border
    # Get nodes in zones -> then build network with these nodes

    # Build network using networkx
    # Map the coordinates to the edges DataFrame
    # Set the index of the nodes DataFrame to be the 'ID_point' column
    points.set_index('ID_point', inplace=True)

    # Build graph vectorized (much faster than row-by-row iteration)
    edges['fftt'] = pd.to_numeric(edges['ffs']) / edges.geometry.length
    G = nx.from_pandas_edgelist(
        edges, source='start', target='end',
        edge_attr=['ID_edge', 'fftt'],
        create_using=nx.MultiGraph()
    )
    nx.set_node_attributes(G, points['geometry'].apply(lambda g: (g.x, g.y)).to_dict(), 'pos')
    nx.set_node_attributes(G, points['generate_demand'].to_dict(), 'demand')
    """
    # Plot graph small points with coordinates as position and color based on demand attribute
    nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), node_size=3, node_color=list(nx.get_node_attributes(G, 'demand').values()), edge_color='black', width=0.5)
    #nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), node_size=1, node_color='black', edge_color='black', width=0.5)
    plt.show()
    """

    # Compute the route for each OD pair (maybe limit to max 5)
    # Get routes for all points with demand = True in the network G
    # Get all nodes with demand = True
    # nx.all_simple_edge_paths() # (u,v,k) -> u,v are nodes, k is the key of the edge

    # Step 1: Identify nodes with demand
    demand_nodes = [n for n, attr in G.nodes(data=True) if attr.get('demand') == True]
    if len(demand_nodes) < 29:
        print(f"Number of points considered: {len(demand_nodes)} ({len(demand_nodes) * len(demand_nodes)})")
    """
    # Find connected components
    connected_components = list(nx.connected_components(G))
    # Filter components to include only those with at least one node having 'demand' == True
    #connected_components_with_demand = [comp for comp in connected_components if any(G.nodes[node]['demand'] for node in comp)]

    print(connected_components)

    # Convert the generator to a list to get its length
    connected_components_list = list(connected_components)

    # To plot each connected component in a different color, we can use a color map
    colors = plt.cm.rainbow(np.linspace(0, 1, len(connected_components_list)))

    # Create a plot
    plt.figure(figsize=(8, 6))

    # For each connected component, using a different color for each
    for comp, color in zip(connected_components_list, colors):
        # Separate nodes with demand and without demand
        nodes_with_demand = [node for node in comp if G.nodes[node]['demand']]
        nodes_without_demand = [node for node in comp if not G.nodes[node]['demand']]

        # Get positions for all nodes in the component
        pos = nx.get_node_attributes(G, 'pos')  # Use existing positions if available

        # Draw nodes with demand
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_with_demand, node_color=[color], node_size=20)
        # Draw nodes without demand
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_without_demand, node_color=[color], node_size=3)
        # Draw edges
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v) for u, v in G.edges() if u in comp and v in comp])

        # Draw node labels in small size
        nx.draw_networkx_labels(G, pos, font_size=2)

    # Show the plot
    plt.title("Connected Components with Individual Colors")
    plt.axis('off')  # Turn off axis
    # safe plot
    plt.savefig(fr"plot/results/connected_components.png", dpi=500)
    plt.show()
    """

    index_routes = 0
    index_OD_pair = 0
    delta_odr = np.zeros((nOD, 1000000))
    routelinks_list = []
    # delta_ir = np.zero((nlinks, 10000)

    # Convert MultiDiGraph or MultiGraph to DiGraph or Graph
    G_simple = nx.Graph(G)

    def k_shortest_paths_edge_ids(G, source, target, k, weight=None):
        return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

    for i in range(len(demand_nodes)):
        for j in range(len(demand_nodes)):
            source = demand_nodes[i]
            target = demand_nodes[j]
            unique_paths_ij = []

            try:
                paths = k_shortest_paths_edge_ids(G_simple, source, target, 2, weight='fftt')
            except nx.NetworkXNoPath:
                index_OD_pair += 1
                continue

            for path in paths:
                edge_ids = [list(G[u][v])[0] for u, v in zip(path[:-1], path[1:])]  # Assuming G is a DiGraph or Graph

                # Check if this path is already in unique_paths
                if edge_ids not in unique_paths_ij:
                    unique_paths_ij.append(edge_ids)
            # print(unique_paths_ij)
            for edge_ids in unique_paths_ij:
                routelinks_list.append(edge_ids)
                delta_odr[index_OD_pair][index_routes] = 1
                index_routes += 1

            index_OD_pair += 1
            # print(f"Number of routes: {index_routes} and OD pairs: {index_OD_pair}")

    # print(f"Number of routes: {index_routes}")
    # print(f"Number of OD pairs: {index_OD_pair}")

    max_length = max(len(arr) for arr in routelinks_list)
    # Replace all 0s with -1
    routelinks_list = [[-1 if element == 0 else element for element in sublist] for sublist in routelinks_list]

    # Create a 2D array, padding shorter arrays with zeros
    routelinks_same_len = [arr + [0] * (max_length - len(arr)) for arr in routelinks_list]
    routelinks = np.array(routelinks_same_len)

    # Get the number of rows in array1
    n_routes = routelinks.shape[0]

    ## Assert the algorithm works well
    column_sums = np.sum(delta_odr, axis=0)
    # print(f"The number of routes {np.sum(column_sums >= 1)} and {n_routes}")
    if np.sum(column_sums >= 1) == n_routes:
        delta_odr = delta_odr[:, :n_routes]
    else:
        print("not same amount of routes in computation - check code for errors")

    # Example edge keys and 2D numpy array
    edge_keys = [k for u, v, k in G.edges(keys=True)]  # Replace with your actual edge keys
    # Replace all 0s with -1
    edge_keys = [-1 if element == 0 else element for element in edge_keys]

    # Initialize DataFrame with edge IDs as columns and zeros
    delta_ir_df = pd.DataFrame(0, index=np.arange(len(routelinks)), columns=edge_keys)

    # Fill the DataFrame
    ####################################################################################################################
    ######## This is not filling correctly

    # From a matrix with all edge IDs for each route (routelinks) route ID (x) and edge ID (y) just in a list
    # to a matrix with all routes (x) and all edges (y) as delta_ir (binary if edge in route)
    for row_idx, path in enumerate(routelinks):
        # print(f"row_idx: {row_idx} and path: {path}")
        for edge in path:
            if edge != 0:
                delta_ir_df.at[row_idx, edge] = 1
        # print entire row
        # print(delta_ir_df.iloc[row_idx])

    # Sort in ascending edge_id and store it as array
    # print(delta_ir_df.sort_index(axis=1).transpose().head(10).to_string())
    delta_ir = delta_ir_df.sort_index(axis=1).transpose().to_numpy()
    """
    print(f"Amount of links used: {np.sum(delta_ir > 0)} (delta_ir) and {np.sum(routelinks > 0)} (routelinks)")

    print(f"Shape delta_odr {delta_odr.shape} (OD pairs x #routes)")
    print(f"Shape delta_ir {delta_ir.shape} (#links (edge ID sorted ascending) x #routes)")
    print(f"Shape routelinks {routelinks.shape} (#routes x amount of links of longest route)")
    print(f"Shape D_od {D_od.shape[0]} ({math.sqrt(D_od.shape[0])} OD zones)")
    print(f"nOD number of OD pairs {nOD}")
    print(f"number of routes (n_routes) {n_routes}")
    """
    # paths = list(k_shortest_paths(G, source, target, 3))

    # Store a matrix with edge_ID for each route (routelinks) route ID (x) and edge ID (y)

    # Store a matrix with all OD pairs (x) and all route (y) as delta_odr (binary if route serves OD pair)

    # Store a matrix with all links (x) and all edges (y) as delta_ir (binary if edge in route)
    return delta_ir, delta_odr, routelinks, D_od, nOD, n_routes


def CostFun(Xi, par):
    # Computes the cost function for the flow Xi, with the parameters 'par', Xi has the adequate size

    # BPR
    s1 = np.power(np.divide(Xi, par['Xmax_i']), par['gamma_i'])
    s2 = np.multiply(par['alpha_i'], s1)
    Ci = np.multiply(par['fftt_i'], np.add(1, s2))
    # Ci=par.fftt_i.*(1+par.alpha_i.*(Xi./par.Xmax_i).^par.gamma_i);
    return Ci


def IntCostFun(Xi, par):
    # Computes the integral of the cost function for the flow Xi, with the
    # parameters 'par'. Xi has the adequate size
    # BPR
    # s1 = (Xi./par.Xmax_i).^par.gamma_i            (Flow/MaxCapacity)^gamma
    # s2 = Xi*par.alpha_i.*par.fftt_i.*s1           Flow*alpha*fftt*s1
    # Ci = Xi.*par.fftt_i + s2./(par.gamma_i + 1)   Flow*fftt + s2/(gamma+1)

    s1 = np.power(np.divide(Xi, par['Xmax_i']), par['gamma_i'])  # (Xi./par.Xmax_i).^par.gamma_i
    s2 = np.multiply(Xi,
                     np.multiply(par['alpha_i'], np.multiply(par['fftt_i'], s1)))  # (Xi.*par.alpha_i.*par.fftt_i.*s1)
    Ci = np.multiply(Xi, par['fftt_i']) + np.divide(s2, (
        np.add(par['gamma_i'], 1)))  # Xi.*par.fftt_i + s2./(par.gamma_i + 1);
    # Ci=Xi.*par.fftt_i + (Xi.*par.alpha_i.*par.fftt_i.*(Xi./par.Xmax_i).^par.gamma_i)./(par.gamma_i + 1);

    # returns travel cost for each link
    return Ci


def SUE_C_Logit(nroutes, D_od, par, delta_ir, delta_odr, cf_r, theta):
    # De acuerdo con C logit SUE_Zhou (2010)

    # --- Optimizacion NO lineal
    # objfun = @(D_r)( sum(IntLinksTimes(D_r)) + 1/theta*sum(D_r.*log(D_r)) + sum(D_r.*cf_r) );
    # A=[];b=[];  # A x <= b
    # Aeq=delta_odr;beq=D_od; # Aeq x = beq
    # lb=zeros(nroutes,1);ub=max(D_od)*ones(nroutes,1);
    # D_r0=delta_odr.H*(D_od./sum(delta_odr,2)); #estimacion: reparto equitativamente #zeros(nroutes,1);#
    ##options=optimset('Algorithm','interior-point','MaxFunEvals',1e5);#,'Display','off'); #matlab viejo
    # options=optimoptions('fmincon','Algorithm','sqp','MaxFunEvals',1e5,'Display','off');
    # [D_r,fval,exitflag,output]=fmincon(objfun,D_r0,A,b,Aeq,beq,lb,ub,[],options);
    # x_i=delta_ir*D_r

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- Non-linear optimisation using Sequential least squares programming

    def IntLinksTimes(D_r):
        # Demand on each link from demand on routes
        x_i = np.matmul(delta_ir, D_r)
        intTrec_i = IntCostFun(x_i, par)  # integral de la funcion de coste
        return intTrec_i

    # def bpr(x):

    def fun(x):
        x = np.where(np.isnan(x) | (x <= 0), 0.0001, x)
        temp_log = np.log(x)
        temp_log[np.isinf(temp_log)] = 0.1
        temp_log[np.isnan(temp_log)] = 0.1
        # Compute IntLinksTimes once — it's the most expensive call (matmul)
        ilt = IntLinksTimes(x)
        result = np.sum(ilt) + np.sum(x * temp_log) + np.sum(x * cf_r)
        return result

        # s=np.squeeze(eqval)
        # val=float(s)
        # val=float(eqval)
        # return eqval

    # def fun_der(x):
    #     der = np.zeros_like(x)
    #     s0 = np.matmul(delta_ir,x)
    #     s1 = np.power(np.divide(s0,par['Xmax_i']),par['gamma_i'])
    #     s2 = np.multiply(par['alpha_i'],s1)
    #     s3 = np.multiply(par['fftt_i'],s2)
    #     s4 = np.matmul(delta_ir.transpose(),s3)
    #     a1 = np.matmul(delta_ir.transpose(),par['fftt_i'])
    #     der = np.divide((np.log(x)+1),theta)+cf_r+a1+s4
    #     return der.flatten()

    def callback_function(*args):
        xk = args[0]  # The first argument is the current solution vector
        objective_value = fun(xk)
        print(f"Iteration, Objective Function Value: {objective_value}")

    ineq_cons = {'type': 'ineq',
                 'fun': lambda x: x}  # ,
    # 'jac' : lambda x: np.array([])}
    eq_cons = {'type': 'eq',
               'fun': lambda x: (np.matmul(delta_odr, x) - (D_od)).flatten()}  # ,
    # 'jac' : lambda x: -delta_odr.flatten()}

    ###################################################################################################################
    # Check if there are nan values in the matrix
    # replace nan values with 0
    with np.errstate(divide='ignore', invalid='ignore'):
        tt = (np.divide(D_od.transpose(), np.sum(delta_odr, axis=1))).transpose()
    tt[np.isinf(tt)] = tt.max() * 10
    tt[np.isnan(tt)] = 0

    D_r0 = np.matmul(delta_odr.transpose(), tt)

    # D_r0=np.matmul(delta_odr.transpose(),(np.divide(D_od.transpose(),np.sum(delta_odr,axis=1))).transpose())
    lb = np.zeros((np.shape(D_r0))).flatten()
    # Add a very small value to lb to avoid 0 in x
    lb = lb + 0.01
    ub = (max(D_od) * np.ones(np.shape(D_r0))).flatten()
    ub = ub * 5
    bounds = Bounds(lb, ub)
    # print(f"lb: {lb}")
    # print(f"ub: {ub}")
    # res=least_squares(fun, D_r0.flatten(),jac=fun_der,bounds=bounds)

    # D_r to be optimized -> demand on each route.
    # trust-constr is preferred but fails with a LinAlgError when the equality-
    # constraint Jacobian (delta_odr) is numerically singular.  SLSQP is used
    # as a fallback: it is less sensitive to rank deficiency and handles the
    # bounds directly without an SVD factorisation step.
    x0 = np.clip(D_r0.flatten(), lb, ub)
    try:
        res = minimize(fun, x0,
                       method='trust-constr',
                       constraints=[eq_cons, ineq_cons],
                       options={
                           'maxiter': 3,
                           'gtol': 5.0,
                           'xtol': 1e-3,
                           'verbose': 0,
                           'disp': False},
                       bounds=bounds)
    except (np.linalg.LinAlgError, ValueError):
        res = minimize(fun, x0,
                       method='SLSQP',
                       constraints=[eq_cons],
                       options={'maxiter': 50, 'ftol': 1.0, 'disp': False},
                       bounds=bounds)
    # callback=callback_function
    # )
    # Describe variables
    # D_r is the demand on each route
    D_r = res.x
    D_r[D_r <= 0] = 0.001
    # check if values in D_r are negative if so print the amount
    if np.sum(D_r < 0) > 0:
        print(f"Negative values in D_r: {np.sum(D_r < 0)}")
    # Same for 0
    if np.sum(D_r == 0) > 0:
        print(f"0 in D_r: {np.sum(D_r == 0)}")

    # x_i is the demand on each link
    x_i = delta_ir * D_r
    # Get travel time on each route
    intTrec_i = IntCostFun(x_i, par)

    thetavec = np.ones_like(D_r)

    # fval is the objective function value
    fval = np.sum(IntLinksTimes(D_r)) + np.sum(np.divide(np.multiply(D_r, np.log(D_r)), thetavec)) + np.sum(
        (np.multiply(D_r, cf_r)))
    # fval = sum(IntLinksTimes(D_r)) + 1/theta*sum(np.multiply(D_r,np.log(D_r))) + sum(np.multiply(D_r,cf_r))
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    return [x_i, D_r, intTrec_i, fval]


def Commonality(betaCom, delta_ir, delta_odr, fftt_i, fftt_r):
    [nOD, nroutes] = delta_odr.shape
    # print(f"nOD {nOD} and nroutes {nroutes}")

    # Initialize commonality factor (per route)
    cf_r = np.zeros((nroutes, 1))

    # Iterate over all pairs of OD
    for od in range(0, nOD):
        # Find routes for OD pair
        routes = np.argwhere(np.ravel(delta_odr[od, :]) == 1)
        # print(routes)

        # Check if there is more than  route for the OD pair
        if len(routes) <= 1:
            # In this case there is no commonality factor
            continue
        else:
            # routes=find(delta_odr[od,:]==1)
            for r1 in routes:
                # Freeflow travel time for route
                t0_1 = fftt_r[r1]
                for r2 in routes:
                    # Freeflow travel time for route
                    t0_2 = fftt_r[r2]
                    aa = np.argwhere(np.ravel(delta_ir[:, r1]) == 1)
                    bb = np.argwhere(np.ravel(delta_ir[:, r2]) == 1)
                    # Get common links among routes routes compared
                    common = np.array(list(set(aa.flatten()).intersection(bb.flatten())))
                    # common=intersect(find(delta_ir(:,r1)==1),find(delta_ir(:,r2)==1));

                    # Check if common is empty
                    if len(common) == 0:
                        t0_1_2 = 0

                    else:
                        # Sum of freeflow travel time for common links
                        t0_1_2 = sum(fftt_i[common])

                    # Update commonality factor
                    cf_r[r1] = cf_r[r1] + t0_1_2 / (t0_1 ** .5 * t0_2 ** .5)
                    """
                    if (t0_1**.5*t0_2**.5) == 0:
                        print(f"t0_1 or t0_2 is 0 for OD pair {od} and routes {r1} and {r2}")
                    """
    cf_r[cf_r <= 0] = 0.0001
    cf_r = betaCom * np.log(cf_r)
    cf_r[np.isnan(cf_r)] = 0  # for routes with just one link -->cf_r=0;
    cf_r[np.isinf(cf_r)] = 0  # for routes with just one link -->cf_r=0;
    return cf_r


def travel_flow_optimization(OD_matrix, points, edges, voronoi, dev, scen, _topo=None):
    """Run C-Logit SUE assignment for one (dev, scenario) combination.

    _topo: optional dict returned by a previous call on the same network.
           When provided, the expensive route-enumeration step is skipped and
           only the demand vector is recomputed from OD_matrix.
           Pass the second return value from a prior call to reuse topology.

    Returns: (travel_time_array, topo_dict)
    """
    if _topo is None:
        _, _, _, _, _, par = convert_data_to_input(points=points, edges=edges)
    else:
        par = _topo['par']

    if _topo is None:
        # Full topology build: route enumeration (expensive k-shortest paths)
        delta_ir, delta_odr, routelinks, D_od_full, nOD, nroutes = get_nw_data(
            OD_matrix=OD_matrix, points=points, voronoi_gdf=voronoi, edges=edges)

        OD_single = int(math.sqrt(nOD))
        idx_diag  = np.arange(0, nOD, OD_single + 1)
        D_od      = np.delete(D_od_full, idx_diag)
        delta_odr = np.delete(delta_odr, idx_diag, axis=0)

        # Precompute topology-only quantities (shared across scenarios)
        fftt_r = (np.matmul(par['fftt_i'].transpose(), delta_ir)).transpose()
        cf_r   = Commonality(1, delta_ir, delta_odr, par['fftt_i'], fftt_r)

        # Identify which OD IDs were used (to filter other scenarios' matrices)
        _od_ids = list(OD_matrix.index.map(lambda x: int(float(x))))

        _topo = {
            'delta_ir':   delta_ir,
            'delta_odr':  delta_odr,   # already has diagonals removed
            'nroutes':    nroutes,
            'cf_r':       cf_r,
            'par':        par,
            'od_ids':     _od_ids,
            'OD_single':  OD_single,
            'idx_diag':   idx_diag,
        }
    else:
        # Reuse precomputed topology; only recompute demand vector D_od
        delta_ir  = _topo['delta_ir']
        delta_odr = _topo['delta_odr']
        nroutes   = _topo['nroutes']
        cf_r      = _topo['cf_r']
        par       = _topo['par']
        od_ids    = _topo['od_ids']
        idx_diag  = _topo['idx_diag']

        OD_matrix.index   = OD_matrix.index.map(lambda x: int(float(x)))
        OD_matrix.columns = OD_matrix.columns.map(lambda x: int(float(x)))
        common = [i for i in od_ids if i in OD_matrix.index and i in OD_matrix.columns]
        D_od_full = OD_matrix.loc[common, common].to_numpy().flatten()
        D_od = np.delete(D_od_full, idx_diag)
    # print amount of nan in cf_r
    # print(f"Amount of nan in cf_r: {np.sum(np.isnan(cf_r))}")
    # print(f"Amount of inf in cf_r: {np.sum(np.isinf(cf_r))}")
    theta = 1.2

    iteration_count = 0

    done = 0  # calculate iterations
    factor = 0.01  # fftt and capacity will be multiplied bu this factor

    ## CALCULATION OF INCREASE OF TOTAL TRAVEL COST
    if not done:
        t = timeit.default_timer()
        # t = time.process_time()

        # --- Reference value: network with no damage

        [Xi, D_r1, intTrec_i, ref] = SUE_C_Logit(nroutes, D_od, par, delta_ir, delta_odr, cf_r, theta)
        Results = {'Xi': Xi, "D_r1": D_r1, 'ref': ref}

        # Sum values of Xi for each row
        Xi_sum = np.sum(Xi, axis=1)

        # Compute total travel time
        # Multiplying travel time on each link with demand on each link
        travel_time = np.matmul(intTrec_i.transpose(), Xi_sum).flatten()

        # --- Damaging link by link
        # Results=zeros(nlinks,1) #sol for each i
        # tic = time.time()
        # parfor i=1:nlinks
        #    #Reduction of capacity and ffftt of the affected link
        #    par2=par;
        #    par2.fftt_i(i)=par2.fftt_i(i)/factor;
        #    par2.Xmax_i(i)=par2.Xmax_i(i)*factor;
        #    # cost evaluation
        #    [~,~,fval]=SUE_C_Logit(nroutes,D_od,par2,delta_ir,delta_odr,cf_r,theta);
        #    fprintf('Evaluation i=#-3.0f  -> #10.2f (#10.2f ##)\n',i,fval, (fval-ref)/ref*100)
        #    Results(i)=fval;
        # end
        # toc = time.time()
        # print(toc-tic, ' sec elapsed')
        comptime = timeit.default_timer() - t;
        # print(f'CPU time (seconds): {comptime}')

        return travel_time, _topo  # .item(0)


def _run_one_dev(dev, cand_row, links_base, points_base):
    """Run SUE assignment for one development across all 3 scenarios.

    Isolated function so joblib can run it in a subprocess.
    Returns dict {"development": dev, "low": tt, "medium": tt, "high": tt}.
    """
    scenario = ["low", "medium", "high"]

    edges = links_base.copy()
    points = points_base.copy()

    if cand_row["dev_type"] == "netzluecke":
        edge_ID_max = edges["ID_edge"].max()
        new_edge_row = {
            "start":    int(cand_row["start"]),
            "end":      int(cand_row["end"]),
            "geometry": cand_row["geometry"],
            "ffs":      20,
            "capacity": 1000,
            "tt_min":   float(cand_row["length_m"]) / 1000.0 / 20.0 * 60.0,
            "ID_edge":  edge_ID_max + 1,
        }
        edges = gpd.GeoDataFrame(
            pd.concat([edges, pd.DataFrame(pd.Series(new_edge_row)).T], ignore_index=True))
    else:
        mask = (
            ((edges["start"] == int(cand_row["start"])) & (edges["end"] == int(cand_row["end"]))) |
            ((edges["start"] == int(cand_row["end"])) & (edges["end"] == int(cand_row["start"])))
        )
        if mask.any():
            edges.loc[mask, "ffs"] = 22

    edges["ID_edge"] = edges["ID_edge"].astype(int)
    edges = edges.sort_values(by=["ID_edge"])

    voronoi_df = gpd.read_file(fr"data/Network/travel_time/developments/dev{dev}_Voronoi.gpkg")

    results = {"development": dev}
    _topo = None  # topology built on first scenario, reused for the rest
    for scen in scenario:
        print(f"  dev {dev} / {scen}")
        OD_matrix = pd.read_csv(
            fr"data/traffic_flow/od/developments/cycling_od_matrix_dev{dev}_{scen}.csv",
            sep=",", index_col=0)
        tt, _topo = travel_flow_optimization(
            OD_matrix=OD_matrix, points=points, edges=edges, voronoi=voronoi_df,
            dev=dev, scen=scen, _topo=_topo)
        results[scen] = tt

    return results


def tt_optimization_all_developments(n_jobs=1, dev_type_filter=None):
    """n_jobs: number of parallel workers(1 = sequential / safe default, 2 + = parallel).
    Set n_jobs > 1 only if peak memory is well below half of available RAM.
    dev_type_filter: None(all), 'netzluecke', or 'schwachstelle'
    """
    scenario = ["low", "medium", "high"]
    dev_candidates = gpd.read_file(r"data/Network/processed/development_candidates.gpkg")
    in_corridor = dev_candidates["within_corridor"] | dev_candidates["on_border"]
    if dev_type_filter is not None:
        in_corridor = in_corridor & (dev_candidates["dev_type"] == dev_type_filter)
        print(f"  dev_type_filter='{dev_type_filter}': {in_corridor.sum()} candidates")
    _corridor_ids = set(dev_candidates[in_corridor]["ID_new"].tolist())
    developments = list(_corridor_ids)

    # Base network loaded once and shared (read-only) across workers
    points_base = gpd.read_file(r"data/Network/processed/points_with_attribute.gpkg")
    points_base['id_dummy'] = points_base.index.values
    points_base.index = points_base.index.astype(int)
    points_base = points_base.sort_index()

    links_base = gpd.read_file(r"data/Network/processed/edges_with_attribute.gpkg")
    links_base["ID_edge"] = links_base["ID_edge"].astype(int)
    links_base = links_base.sort_values(by=["ID_edge"])

    valid_devs = [d for d in developments if d in dev_candidates['ID_new'].values]
    print(f"Running TT assignment for {len(valid_devs)} developments "
          f"({'parallel' if n_jobs != 1 else 'sequential'}, n_jobs={n_jobs})")

    rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_run_one_dev)(
            dev,
            dev_candidates[dev_candidates["ID_new"] == dev].iloc[0],
            links_base,
            points_base,
        )
        for dev in valid_devs
    )

    costs_travel_time = pd.DataFrame(rows, columns=["development", "low", "medium", "high"])
    costs_travel_time.to_csv(r"data/traffic_flow/travel_time.csv", index=False)
    costs_travel_time.to_csv(r"data/traffic_flow/travel_time_2.csv", index=False)


def tt_optimization_status_quo():
    # Run travel time optimization for current infrastructure and all scenarios
    scenario = ["low", "medium", "high"]  # "20" unused by monetize_tts and takes hours
    dev = "status_quo"

    # Load network once — topology is identical for all status-quo scenarios
    voronoi_df = gpd.read_file(r"data/Network/travel_time/Voronoi_statusquo.gpkg")
    points = gpd.read_file(r"data/Network/processed/points_with_attribute.gpkg")
    points.index = points.index.astype(int)
    points = points.sort_index()
    edges = gpd.read_file(r"data/Network/processed/edges_with_attribute.gpkg")
    edges["ID_edge"] = edges["ID_edge"].astype(int)
    edges = edges.sort_values(by=["ID_edge"])

    results_status_quo = {}
    for scen in scenario:
        OD_matrix = pd.read_csv(r"data/traffic_flow/od/od_matrix_" + scen + ".csv", sep=",", index_col=0)
        # No _topo caching here: each status-quo scenario can have a different OD
        # matrix size (e.g. "20" uses all nodes; "low/medium/high" use only access
        # points), so the cached delta_odr dimensions would not match.
        tt, _ = travel_flow_optimization(
            OD_matrix=OD_matrix, points=points, edges=edges, voronoi=voronoi_df,
            dev=dev, scen=scen, _topo=None)
        results_status_quo[scen] = tt
    pd.DataFrame(results_status_quo).to_csv(r"data/traffic_flow/travel_time_status_quo.csv", index=False)


def monetize_tts(VTTS, duration):
   # Import total travel time for each scenario and each development
   tt_total = pd.read_csv(r"data/traffic_flow/travel_time.csv")
   # tt_total_low = pd.read_csv(r"data/traffic_flow/travel_time_low.csv")
   # tt_total["low"] = tt_total_low["low"]
   # Some values are stored in list format, convert them to float

   # convert columns object to float
   tt_total["low"] = tt_total["low"].apply(lambda x: float(x[1:-1]))
   tt_total["medium"] = tt_total["medium"].apply(lambda x: float(x[1:-1]))
   tt_total["high"] = tt_total["high"].apply(lambda x: float(x[1:-1]))

   # Import reference travel time for each scenario and current infrastructure
   tt_status_quo = pd.read_csv(fr"data/traffic_flow/travel_time_status_quo.csv")

   # monetization factor of travel time (peak hour * CHF/h * 365 d/a * 30 a)
   #mon_factor = VTTS * 365 * duration
   mon_factor = VTTS * 2.5 * 250 * duration
   # Compute difference in travel time for each scenario and each development

   tt_total["tt_low"] = (tt_status_quo["low"].iloc[0] - tt_total["low"]) * mon_factor
   tt_total["tt_medium"] = (tt_status_quo["medium"].iloc[0] - tt_total["medium"]) * mon_factor
   tt_total["tt_high"] = (tt_status_quo["high"].iloc[0] - tt_total["high"]) * mon_factor

   # Change presign of all psitive values to negative
   columns_to_negate = ['tt_low', 'tt_medium', 'tt_high']
   for col in columns_to_negate:
       tt_total[col] = tt_total[col].apply(lambda x: -abs(x))

   # drop useless columns
   tt_total = tt_total.drop(columns=["low", "medium", "high"])
   tt_total.to_csv(r"data/costs/traveltime_savings.csv")


def monetize_dijkstra_tts(VTTS, duration):
    """Convert Dijkstra-based node-hour improvements to CHF travel time savings.

    Replaces the SUE-based monetize_tts when SKIP_TT=True.
    Travel time is purely length/speed (no congestion, no capacity constraints).

    Formula matches monetize_tts:
        savings_chf = improvement_h * VTTS * 2.5 * 250 * duration
    Applied equally to all three scenarios (no scenario-specific demand modelling).
    Savings are stored as negative values (cost convention: negative = saved cost).
    """
    import pandas as pd, os

    src = r'data/costs/dijkstra_improvements.csv'
    if not os.path.exists(src):
        print(f"[monetize_dijkstra_tts] {src} not found — skipping")
        return

    df = pd.read_csv(src)
    mon_factor = VTTS * 2.5 * 250 * duration
    savings = -(df['tt_improvement_h'] * mon_factor)  # negative = saved cost

    out = pd.DataFrame({
        'ID_new':     df['ID_new'].astype(int),
        'tt_low':     savings,
        'tt_medium':  savings,
        'tt_high':    savings,
    })
    os.makedirs('data/costs', exist_ok=True)
    out.to_csv(r'data/costs/traveltime_savings.csv', index=False)
    print(f"  Dijkstra TT savings monetized → data/costs/traveltime_savings.csv  "
          f"({len(out)} rows, factor={mon_factor:.1f} CHF/h)")


