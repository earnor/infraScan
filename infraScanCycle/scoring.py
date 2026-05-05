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
import osmnx as ox
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

# Loaded once per process; avoids repeated gpkg reads inside the hot assignment loop.
_AP_IDS_CACHE: "frozenset | None" = None

def _get_ap_ids() -> frozenset:
    global _AP_IDS_CACHE
    if _AP_IDS_CACHE is None:
        _ap_path = r"data/Network/processed/access_points_corridor.gpkg"
        if os.path.exists(_ap_path):
            _ap_gdf = gpd.read_file(_ap_path)
            if "ID_point" in _ap_gdf.columns:
                _AP_IDS_CACHE = frozenset(_ap_gdf["ID_point"].astype(int).tolist())
            else:
                _AP_IDS_CACHE = frozenset(_ap_gdf.index.astype(int).tolist())
        else:
            _AP_IDS_CACHE = frozenset()
    return _AP_IDS_CACHE


def construction_costs(cycle_path, upgrade_factor=0.40):
    """
    cycle_path     = cost per metre of new cycle path [CHF/m]
    upgrade_factor = fraction of cycle_path cost for Schwachstellen upgrades (default 40%)
    """
    candidates = gpd.read_file(r"data/Network/processed/development_candidates.gpkg")
    candidates = candidates[candidates["within_corridor"] | candidates["on_border"]].fillna(0)

    # Each candidate is already one edge → no groupby needed
    candidates["path_len"] = candidates.geometry.length

    # Differentiate build cost: Netzlücke = full build, Schwachstelle = upgrade
    candidates["unit_cost"] = candidates["dev_type"].apply(
        lambda t: cycle_path if t == "netzluecke" else cycle_path * upgrade_factor
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
    print(generated_links_gdf.head(10).to_string())
    return







def land_tb_reallocated(links, buffer_distance):
    zones = gpd.read_file(r"data/landuse_landcover/processed/partly_protected.gpkg")
    print("Zones", zones.name.unique())

    buffer = links.copy()
    buffer = buffer[buffer.geometry.notna() & buffer.geometry.is_valid]
    # Create a buffer around each line

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
        print(path)

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
    print(grouped_sum.head().to_string())

    grouped_sum["local_s1"] = costs["s1_pop"] - grouped_sum["s1_pop"]
    grouped_sum["local_s2"] = costs["s2_pop"] - grouped_sum["s2_pop"]
    grouped_sum["local_s3"] = costs["s3_pop"] - grouped_sum["s3_pop"]
    print(grouped_sum.head().to_string())
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
            print(trip_tif.shape)
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
    points = gpd.read_file(r"data/Network/processed/generated_nodes.gpkg")
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

def route_comfort(duration, comfort_value_chf_m_year=2.0):
    """
    Compute route comfort score and monetised comfort benefit per development.

    For each candidate edge (Netzlücke or Schwachstelle):
      cli_candidate  — Comfort Level Index from its ROUTENTYP (0–1)
      comfort_delta  — improvement over no cycling infrastructure (NO_CYCLING_CLI = 0.10)
      slope_factor   — penalty for steep gradient:
                         < 4 %  → 1.0 (good for everyday cycling)
                         4–8 %  → 0.6 (hard)
                         > 8 %  → 0.3 (very bad)
                       weighted average over the edge length
      comfort_benefit = comfort_delta × length_m × slope_factor
                        × comfort_value_chf_m_year × duration  [CHF]

    Saves: data/costs/route_comfort.csv
    """
    ROUTENTYP_CLI = {
        'Velobahn':                        1.00,
        'Veloschnellroute':                0.90,
        'Hauptverbindung':                 0.70,
        'Nebenverbindung':                 0.50,
        'Zusätzliche Freizeitverbindung':  0.25,
    }
    DEFAULT_CLI    = 0.40
    NO_CYCLING_CLI = 0.10

    def cli(routentyp):
        return ROUTENTYP_CLI.get(str(routentyp).strip(), DEFAULT_CLI)

    def _slope_factor(geom, elev_data, elev_transform, elev_nodata, n_samples=20):
        """Return a slope comfort factor (0–1) by sampling elevation along geom."""
        length = geom.length
        if length == 0:
            return 1.0
        fracs = np.linspace(0, 1, n_samples)
        pts   = [geom.interpolate(f, normalized=True) for f in fracs]
        rows, cols = rasterio.transform.rowcol(
            elev_transform,
            [p.x for p in pts],
            [p.y for p in pts],
        )
        h, w = elev_data.shape
        elevations = []
        for r, c in zip(rows, cols):
            if 0 <= r < h and 0 <= c < w:
                v = float(elev_data[r, c])
                elevations.append(np.nan if (elev_nodata is not None and v == elev_nodata) else v)
            else:
                elevations.append(np.nan)
        elevations = np.array(elevations, dtype=float)
        valid_mask = ~np.isnan(elevations)
        if valid_mask.sum() < 2:
            return 1.0
        seg_len = length / (n_samples - 1)
        elev_valid = elevations[valid_mask]
        grad = np.abs(np.diff(elev_valid)) / seg_len
        good = np.sum(grad <  0.04) * seg_len
        hard = np.sum((grad >= 0.04) & (grad < 0.08)) * seg_len
        bad  = np.sum(grad >= 0.08) * seg_len
        total = good + hard + bad
        if total == 0:
            return 1.0
        return (good * 1.0 + hard * 0.6 + bad * 0.3) / total

    elev_path = r"data/elevation_model/elevation.tif"
    elev_data      = None
    elev_transform = None
    elev_nodata    = None
    if os.path.exists(elev_path):
        with rasterio.open(elev_path) as src:
            elev_data      = src.read(1)
            elev_transform = src.transform
            elev_nodata    = src.nodata

    candidates = gpd.read_file(r"data/Network/processed/development_candidates.gpkg")
    candidates = candidates[candidates["within_corridor"] | candidates["on_border"]]
    edges      = gpd.read_file(r"data/Network/processed/edges_corridor.gpkg")

    rt_col = next((c for c in edges.columns if c.upper().startswith('ROUTENTYP')), None)
    edges['_cli'] = edges[rt_col].apply(cli) if rt_col else DEFAULT_CLI
    edges['_len'] = edges['length_m'].where(
        edges['length_m'].notna() & (edges['length_m'] > 0),
        other=edges.geometry.length
    )
    total_len = edges['_len'].sum()
    comfort_corridor = (edges['_cli'] * edges['_len']).sum() / total_len if total_len > 0 else DEFAULT_CLI

    records = []
    for _, row in candidates.iterrows():
        id_new    = row['ID_new']
        routentyp = row.get('ROUTENTYP', '')
        cli_cand  = cli(routentyp)
        length_m  = row['length_m'] if row.get('length_m', 0) > 0 else row.geometry.length

        if elev_data is not None:
            slope_fac = _slope_factor(row.geometry, elev_data, elev_transform, elev_nodata)
        else:
            slope_fac = 1.0

        comfort_delta   = cli_cand - NO_CYCLING_CLI
        comfort_benefit = comfort_delta * length_m * slope_fac * comfort_value_chf_m_year * duration

        records.append({
            'ID_new':            id_new,
            'comfort_candidate': round(cli_cand,        3),
            'comfort_corridor':  round(comfort_corridor, 3),
            'comfort_delta':     round(comfort_delta,    3),
            'slope_factor':      round(slope_fac,        3),
            'comfort_benefit':   round(comfort_benefit,  2),
            'link_length_m':     round(length_m,         1),
            'dev_type':          row.get('dev_type', ''),
        })

    result = pd.DataFrame(records)
    os.makedirs(r"data/costs", exist_ok=True)
    result.to_csv(r"data/costs/route_comfort.csv", index=False)

    print(f"  Route comfort saved → data/costs/route_comfort.csv  ({len(result)} developments)")
    print(f"  Corridor CLI (status quo): {comfort_corridor:.3f}")
    if len(result):
        print(f"  Candidate CLI range: {result['comfort_candidate'].min():.3f} – {result['comfort_candidate'].max():.3f}")
        print(f"  Comfort benefit range: {result['comfort_benefit'].min():,.0f} – {result['comfort_benefit'].max():,.0f} CHF")
    return result


def safety_benefits(value_of_safety, duration):
    """
    Compute safety benefits for each development vs. the status-quo cycling network.

    For every OD pair (between access-point Voronoi zones) the function finds
    the shortest path by travel time on both the current and the augmented
    network (current + new development link) and computes the difference in
    cumulative route-safety cost.  The result is monetised via value_of_safety.

    Logic
    -----
    1. Assign a risk weight to every edge based on its ROUTENTYP.
       Higher weight = less safe / more exposed to car traffic.
    2. Build a NetworkX graph for the status-quo network.
    3. For each OD pair (origin, destination access point), find the
       travel-time-optimal route and sum up  risk_weight × length_m
       along the route → route_risk_sq.
    4. For each development:
         a. Add the new node and link to the graph.
         b. Re-compute route_risk_dev for each OD pair.
         c. safety_benefit = Σ_OD [ trips_OD × (route_risk_sq - route_risk_dev) ]
    5. Monetise:  CHF = safety_benefit × value_of_safety × duration

    Parameters
    ----------
    value_of_safety : float
        Willingness-to-pay to avoid one unit of risk-weighted route length,
        per trip, per year  [CHF / (risk_unit · trip · year)].
        A reasonable proxy:  safety literature values accident risk at
        ~0.10–0.30 CHF per person per km on unsegregated vs. segregated
        cycling infrastructure.
    duration : int
        Appraisal horizon in years.

    Returns
    -------
    pd.DataFrame  columns: ID_new, safety_sq, safety_dev,
                           safety_s1, safety_s2, safety_s3  [CHF]
        Saved to  data/costs/safety_benefits.csv
        and       data/costs/safety_benefits.gpkg
    """
    # ------------------------------------------------------------------ #
    # 1.  Safety risk weights per ROUTENTYP
    #     (risk per metre, relative; higher = more dangerous)
    # ------------------------------------------------------------------ #
    ROUTENTYP_RISK = {
        "Veloschnellroute":              1.0,   # dedicated express route – safest
        "Hauptverbindung":               2.0,   # main cycling connection
        "Nebenverbindung":               3.5,   # secondary – often shared road
        "Zusätzliche Freizeitverbindung": 2.5,  # recreational – usually off-road
    }
    DEFAULT_RISK  = 5.0   # unclassified / mixed traffic
    NEW_LINK_RISK = 1.5   # new dedicated cycling infrastructure

    # ------------------------------------------------------------------ #
    # 2.  Load network
    # ------------------------------------------------------------------ #
    points = gpd.read_file("data/Network/processed/points_with_attribute.gpkg")
    edges  = gpd.read_file("data/Network/processed/edges_with_attribute.gpkg")
    edges  = edges.set_crs("epsg:2056", allow_override=True)

    edges["length_m"] = edges.geometry.length
    edges["risk_w"]   = edges["ROUTENTYP"].map(ROUTENTYP_RISK).fillna(DEFAULT_RISK)
    edges["risk_len"] = edges["risk_w"] * edges["length_m"]

    # ------------------------------------------------------------------ #
    # 3.  Build status-quo NetworkX graph
    #     Node IDs  = ID_point (int)
    #     Edge attr = tt   [min]  – used for routing
    #                 risk_len    – accumulated along route for safety score
    # ------------------------------------------------------------------ #
    G_sq = nx.Graph()

    for _, row in points.iterrows():
        G_sq.add_node(int(row["ID_point"]),
                      x=row.geometry.x, y=row.geometry.y)

    for _, row in edges.iterrows():
        u = int(row["start"])
        v = int(row["end"])
        tt       = float(row["tt_min"])
        risk_len = float(row["risk_len"])
        length_m = float(row["length_m"])

        # For parallel edges keep the one with the lower travel time
        if G_sq.has_edge(u, v):
            if tt < G_sq[u][v].get("tt", float("inf")):
                G_sq[u][v].update({"tt": tt, "risk_len": risk_len, "length_m": length_m})
        else:
            G_sq.add_edge(u, v, tt=tt, risk_len=risk_len, length_m=length_m)

    # ------------------------------------------------------------------ #
    # 4.  Load generated links and nodes for developments
    # ------------------------------------------------------------------ #
    new_links = gpd.read_file("data/Network/processed/development_candidates.gpkg")
    nodes_path = "data/Network/processed/generated_nodes.gpkg"
    new_nodes = gpd.read_file(nodes_path)

    voronoi_vals_path = "data/Voronoi/voronoi_developments_tt_values.shp"
    voronoi_vals = gpd.read_file(voronoi_vals_path) if os.path.exists(voronoi_vals_path) else None

    dev_ids = sorted(new_links[new_links["within_corridor"] | new_links["on_border"]]["ID_new"].unique())
    results = []

    # ------------------------------------------------------------------ #
    # 5.  Helper: compute total safety cost for an OD matrix on a graph
    #     Returns a dict  (origin, dest) → (trips, route_risk)
    # ------------------------------------------------------------------ #
    def od_safety_cost(G, od_matrix):
        nodes   = list(od_matrix.index)
        results = {}
        for o in nodes:
            if o not in G:
                continue
            try:
                _, paths = nx.single_source_dijkstra(G, o, weight="tt")
            except Exception:
                continue
            for d in nodes:
                if d == o or d not in G:
                    continue
                trips = float(od_matrix.loc[o, d])
                if trips <= 0:
                    continue
                if d not in paths:
                    print(f"  [DEBUG] Disconnected OD pair: {o} → {d}, trips={trips:.1f}")
                    # No cycling route exists in status-quo: user reroutes via
                    # mixed traffic. Penalty = Euclidean distance × detour
                    # factor × DEFAULT_RISK (worst-case risk weight).
                    ox = G.nodes[o].get("x", 0); oy = G.nodes[o].get("y", 0)
                    dx = G.nodes[d].get("x", 0); dy = G.nodes[d].get("y", 0)
                    eucl_m = ((ox - dx) ** 2 + (oy - dy) ** 2) ** 0.5
                    route_risk = eucl_m * 1.4 * DEFAULT_RISK  # 1.4 detour factor
                else:
                    path = paths[d]
                    route_risk = sum(
                        G[u][v].get("risk_len", 0.0)
                        for u, v in zip(path[:-1], path[1:])
                    )
                results[(o, d)] = (trips, route_risk)
        return results





    # ------------------------------------------------------------------ #
    # 8.  Loop over developments
    # ------------------------------------------------------------------ #
    results = []

    for dev_id in tqdm(dev_ids, desc="Safety benefits"):
        dev_link_rows = new_links[new_links["ID_new"] == dev_id]


        # Load per-development OD matrix (required — skip if missing)
        od_dev_path = f"data/traffic_flow/od/developments/cycling_od_matrix_dev{dev_id}_medium.csv"
        if not os.path.exists(od_dev_path):
            continue
        od_dev = pd.read_csv(od_dev_path, index_col=0)
        od_dev.index = od_dev.index.map(lambda x: int(float(x)))
        od_dev.columns = od_dev.columns.map(lambda x: int(float(x)))

        # Restrict OD to nodes present in G_sq
        valid_nodes = [n for n in od_dev.index if n in G_sq]
        od_f = od_dev.loc[valid_nodes, valid_nodes]
        vals = od_f.values.copy()
        np.fill_diagonal(vals, 0)
        od_f = pd.DataFrame(vals, index=od_f.index, columns=od_f.columns)

        # Compute safety on status-quo before applying development
        sq_costs_dev = od_safety_cost(G_sq, od_f)

        # Mutate G_sq in-place for this development; save original state to restore after
        saved_edges = {}
        for _, lr in dev_link_rows.iterrows():
            u, v = int(lr["start"]), int(lr["end"])
            tt = float(lr["tt_min"])
            length_m = float(lr["length_m"])
            risk_len = NEW_LINK_RISK * length_m
            if G_sq.has_edge(u, v):
                saved_edges[(u, v)] = dict(G_sq[u][v])
                G_sq[u][v].update({"risk_len": risk_len})
            else:
                saved_edges[(u, v)] = None
                G_sq.add_edge(u, v, tt=tt, risk_len=risk_len, length_m=length_m)

        dev_costs = od_safety_cost(G_sq, od_f)

        # Restore G_sq to status-quo state (no copy needed)
        for (u, v), orig in saved_edges.items():
            if orig is None:
                G_sq.remove_edge(u, v)
            else:
                G_sq[u][v].update(orig)

        sq_risk = sum(t * r for t, r in sq_costs_dev.values())
        dev_risk = sum(t * r for t, r in dev_costs.values())
        del sq_costs_dev, dev_costs, od_f, saved_edges
        delta = sq_risk - dev_risk  # positive = safer with development

        if value_of_safety is not None:
            base_benefit = delta * value_of_safety * 365 * duration
        else:
            base_benefit = delta  # unmonetised risk delta
        safety_s1 = base_benefit
        safety_s2 = base_benefit
        safety_s3 = base_benefit

        if voronoi_vals is not None:
            dv = voronoi_vals[voronoi_vals["ID_develop"] == dev_id]
            if not dv.empty:
                s1 = dv["s1_pop"].sum()
                s2 = dv["s2_pop"].sum()
                s3 = dv["s3_pop"].sum()
                if s2 > 0:
                    safety_s1 = base_benefit * (s1 / s2)
                    safety_s3 = base_benefit * (s3 / s2)

        results.append({
            "ID_new": int(dev_id),
            "safety_sq": sq_risk,
            "safety_dev": dev_risk,
            "safety_s1": safety_s1,
            "safety_s2": safety_s2,
            "safety_s3": safety_s3,
        })

    # ------------------------------------------------------------------ #
    # 9.  Save results
    # ------------------------------------------------------------------ #
    out_df = pd.DataFrame(results)
    os.makedirs("data/costs", exist_ok=True)
    out_df.to_csv("data/costs/safety_benefits.csv", index=False)

    gen_nodes = gpd.read_file(nodes_path)[["ID_new", "geometry"]]
    out_gdf   = gen_nodes.merge(out_df, on="ID_new", how="right")
    if out_gdf.geometry.notna().any():
        out_gdf = gpd.GeoDataFrame(out_gdf, geometry="geometry", crs="epsg:2056")
        out_gdf.to_file("data/costs/safety_benefits.gpkg", driver="GPKG")

    print(f"[safety_benefits] Done – {len(out_df)} developments, "
          f"saved to data/costs/safety_benefits.csv / .gpkg")
    return out_df

def net_benefits():
    """
    Compute net benefit per development:  NB = C + M + T + R + S

    C  = construction cost       [CHF, negative]
    M  = maintenance cost        [CHF, negative]
    T  = travel time savings     [CHF, negative costs → positive NB contribution]
    R  = route comfort benefit   [CHF, positive]
    S  = safety benefit          [CHF, positive when value_of_safety is set]

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
    c_comfort = pd.read_csv(r"data/costs/route_comfort.csv")[["ID_new", "comfort_benefit"]]
    c_comfort["ID_new"] = c_comfort["ID_new"].astype(int)

    # ── Safety benefits ───────────────────────────────────────────────────────
    c_safety = pd.read_csv(r"data/costs/safety_benefits.csv")[["ID_new", "safety_s1", "safety_s2", "safety_s3"]]
    c_safety["ID_new"] = c_safety["ID_new"].astype(int)

    # ── Merge all components on ID_new ────────────────────────────────────────
    nb = c_constr.copy()
    for df in [c_maint, c_tt, c_comfort, c_safety]:
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

    nb["R"]    = nb["comfort_benefit"]      # positive

    nb["S_s1"] = nb["safety_s1"]            # positive when monetised, else raw delta
    nb["S_s2"] = nb["safety_s2"]
    nb["S_s3"] = nb["safety_s3"]

    # ── Net benefit per scenario ──────────────────────────────────────────────
    nb["NB_s1"] = nb["C"] + nb["M"] + nb["T_s1"] + nb["R"] + nb["S_s1"]
    nb["NB_s2"] = nb["C"] + nb["M"] + nb["T_s2"] + nb["R"] + nb["S_s2"]
    nb["NB_s3"] = nb["C"] + nb["M"] + nb["T_s3"] + nb["R"] + nb["S_s3"]

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_cols = [
        "ID_new",
        "C", "M",
        "T_s1", "T_s2", "T_s3",
        "R",
        "S_s1", "S_s2", "S_s3",
        "NB_s1", "NB_s2", "NB_s3",
    ]
    out = nb[out_cols].copy()
    os.makedirs("data/costs", exist_ok=True)
    out.to_csv(r"data/costs/net_benefits.csv", index=False)

    # ── Save GPKG (attach point geometry) ────────────────────────────────────
    nodes = gpd.read_file(r"data/Network/processed/generated_nodes.gpkg")[["ID_new", "geometry"]]
    nodes["ID_new"] = nodes["ID_new"].astype(int)
    out_gdf = nodes.merge(out, on="ID_new", how="right")
    out_gdf = gpd.GeoDataFrame(out_gdf, geometry="geometry", crs="EPSG:2056")
    out_gdf.to_file(r"data/costs/net_benefits.gpkg", driver="GPKG")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[net_benefits] {len(out)} developments")
    for label, c_col, t_col, s_col, nb_col in [
        ("s1 (low)",    "C", "T_s1", "S_s1", "NB_s1"),
        ("s2 (medium)", "C", "T_s2", "S_s2", "NB_s2"),
        ("s3 (high)",   "C", "T_s3", "S_s3", "NB_s3"),
    ]:
        print(f"\n  Scenario {label}:")
        print(f"    C+M  : {(out['C']+out['M']).mean():>15,.0f} CHF (mean)")
        print(f"    T    : {out[t_col].mean():>15,.0f} CHF (mean)")
        print(f"    R    : {out['R'].mean():>15,.0f} CHF (mean)")
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
    print(total_costs.head(10).to_string())
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
    # Map point geometries
    points = gpd.read_file(r"data/Network/processed/generated_nodes.gpkg")
    total_costs = total_costs.merge(right=points, how="left", on="ID_new")
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

    for xx in tqdm(xx_values, desc='Processing Voronoi IDs'):
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


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################


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

    # D_r to be optimized -> demand on each route
    res = minimize(fun, D_r0.flatten(),
                   method='trust-constr',
                   constraints=[eq_cons, ineq_cons],
                   options={
                       'maxiter': 3,
                       'gtol': 5.0,   # loose — exit as soon as gradient norm < 5
                       'xtol': 1e-3,
                       'verbose': 0,
                       'disp': False},
                   bounds=bounds
                   )
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


def tt_optimization_all_developments(n_jobs=1):
    """Run SUE travel-time assignment for all developments.

    n_jobs: number of parallel workers (1 = sequential/safe default, 2+ = parallel).
    Set n_jobs > 1 only if peak memory is well below half of available RAM.
    """
    scenario = ["low", "medium", "high"]

    developments = [
        int(re.match(r'cycling_od_matrix_dev([0-9]+)_medium\.csv', f).group(1))
        for f in os.listdir(r"data/traffic_flow/od/developments")
        if re.match(r'cycling_od_matrix_dev([0-9]+)_medium\.csv', f)
    ]

    dev_candidates = gpd.read_file(r"data/Network/processed/development_candidates.gpkg")
    _corridor_ids = set(
        dev_candidates[dev_candidates["within_corridor"] | dev_candidates["on_border"]]["ID_new"].tolist())
    developments = [d for d in developments if d in _corridor_ids]

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




