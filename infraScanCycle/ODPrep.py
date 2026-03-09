import math
import sys
import os
import zipfile
import timeit
import time

import shapely.geometry

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
import scipy.io
import numpy as np
from scipy.optimize import minimize, Bounds, least_squares
import timeit
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import networkx as nx
from itertools import islice
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components

from shapely.ops import voronoi_diagram
from scipy.spatial import Voronoi

#os.chdir(r"C:\Users\Fabrice\Desktop\HS23\Thesis\Code")
os.chdir(r"C:\Users\spadmin\PycharmProjects\infraScan\infraScanRoad")

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


def GetDemandPerCommune(tau=0.013, mode='miv'):
    # now we extract an od matrix for private motrised vehicle traffic from year 2019
    # we then modify the OD matrix to fit our needs of expressing peak hour highway travel demand
    y0 = 2019
    rawod = pd.read_excel('data/_basic_data/KTZH_00001982_00003903.xlsx')
    communalOD = rawod.loc[
        (rawod['jahr'] == 2018) & (rawod['kategorie'] == 'Verkehrsaufkommen') & (rawod['verkehrsmittel'] == mode)]
    # communalOD = data.drop(['jahr','quelle_name','quelle_gebietart','ziel_name','ziel_gebietart',"kategorie","verkehrsmittel","einheit","gebietsstand_jahr","zeit_dimension"],axis=1)
    # sum(communalOD['wert'])
    # if motorised traffic, we remove intrazonal travel... removes about 50% of trips
    if (mode == 'miv'):
        communalOD['wert'].loc[(communalOD['quelle_code'] == communalOD['ziel_code'])] = 0
    # sum(communalOD['wert'])
    # # Take share of OD
    # todo adapt this value
    #tau = 0.013  # Data is in trips per OD combination per day. Now we assume the number of trips gone in peak hour
    # This ratio explains the interzonal trips made in peak hour as a ratio of total interzonal trips made per day.
    # communalOD['wert'] = (communalOD['wert']*tau)
    communalOD.loc[:, 'wert'] = communalOD['wert'] * tau
    # # # Not those who travel < 15 min ?  Not yet implemented.
    return communalOD

def GetODMatrix(od):
    #od_ext = od.loc[(od['quelle_code'] > 9999) | (od['ziel_code'] > 9999)]  # here we separate the parts of the od matrix that are outside the canton. We can add them later.
    #od_int = od.loc[(od['quelle_code'] < 9999) & (od['ziel_code'] < 9999)]
    #odmat = od_int.pivot(index='quelle_code', columns='ziel_code', values='wert')
    odmat = od.pivot(index='quelle_code', columns='ziel_code', values='wert')
    return odmat

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

    geometries = []
    #sub_polygons = []
    for i in range(num_splits):
        for j in range(num_splits):
            # Calculate the corners of the sub-polygon
            sub_min_x = min_x + i * width
            sub_max_x = sub_min_x + width
            sub_min_y = min_y + j * height
            sub_max_y = sub_min_y + height

            # Create the sub-polygon and add it to the list
            sub_polygon = box(sub_min_x, sub_min_y, sub_max_x, sub_max_y)
            geometries.append(sub_polygon)
            #sub_polygons = gpd.GeoDataFrame(pd.concat(sub_polygons, gpd.GeoDataFrame(geometry=sub_polygon).T, ignore_index=True))
            #sub_polygons = gpd.GeoDataFrame(pd.concat([pd.DataFrame(sub_polygons), pd.DataFrame(sub_polygon).T], ignore_index=True))
    sub_polygons = gpd.GeoDataFrame(geometry=geometries)
    return sub_polygons

def nw_from_osm(limits):

    # Split the area into smaller polygons
    num_splits = 10  # Adjust this to get 1/10th of the area (e.g., 3 for a 1/9th split)
    sub_polygons = split_area(limits, num_splits)
    polylist = [sub_polygons.geometry]
    # Initialize the transformer between LV95 and WGS 84
    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    #for i, lv95_sub_polygon in enumerate(polylist):
    for i in range(len(sub_polygons)):
        lv95_sub_polygon = sub_polygons.geometry[i]
        # Convert the coordinates of the sub-polygon to lat/lon
        lat_lon_frame = Polygon([transformer.transform(*point) for point in lv95_sub_polygon.exterior.coords])

        try:
            # Attempt to process the OSM data for the sub-polygon
            print(f"Processing sub-polygon {i + 1}/{len(sub_polygons)}", end='\r')
            #G = ox.graph_from_polygon(lat_lon_frame, network_type="drive", simplify=True, truncate_by_edge=True)
            # Define a custom filter to exclude highways
            # This example excludes motorways, motorway_links, trunks, and trunk_links
            #custom_filter = '["highway"!~"motorway|motorway_link|trunk|trunk_link"]'
            # Create the graph using the custom filter
            G = ox.graph_from_polygon(lat_lon_frame, network_type="drive", simplify=True, truncate_by_edge=True) # custom_filter=custom_filter,
            G = ox.add_edge_speeds(G)

            # Convert the graph to a GeoDataFrame
            gdf_edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
            gdf_edges = gdf_edges[["geometry", "speed_kph"]]
            gdf_edges = gdf_edges[gdf_edges["speed_kph"] <= 80]

            # Project the edges GeoDataFrame to the desired CRS (if necessary)
            gdf_edges = gdf_edges.to_crs("EPSG:2056")

            # Save only the edges GeoDataFrame to a GeoPackage
            output_filename = f"data/Network/OSM_road/sub_area_edges_{i + 1}.gpkg"
            gdf_edges.to_file(output_filename, driver="GPKG")

        except ValueError as e:
            # Handle areas with no nodes by logging or printing an error message
            print(f"Skipping graph in sub-polygon {i + 1} due to error: {e}")
            # Optionally, continue with the next sub-polygon or perform other error handling
            continue

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
    #gdf_combined.dropna(subset=['speed_kph'], inplace=True)
    gdf_combined['speed_kph'].fillna(30, inplace=True)
    # print(gdf_combined.crs)
    # print(gdf_combined.head(10).to_string())
    gdf_combined.to_file('data/Network/OSM_tif/nw_speed_limit.gpkg')
    print("file stored")


    gdf_combined = gpd.read_file('data/Network/OSM_tif/nw_speed_limit.gpkg')

    # Define the resolution
    resolution = 100

    # Define the bounds of the raster (aligned with your initial limits)
    minx, miny, maxx, maxy = limits
    print(limits)

    # Compute the number of rows and columns
    num_cols = int((maxx - minx) / resolution)
    num_rows = int((maxy - miny) / resolution)

    # Initialize the raster with 4 = minimal travel speed (or np.nan for no-data value)
    #raster = np.zeros((num_rows, num_cols), dtype=np.float32)
    raster = np.full((num_rows, num_cols), 4, dtype=np.float32)

    # Define the transform
    transform = from_origin(west=minx, north=maxy, xsize=resolution, ysize=resolution)


    #lake = gpd.read_file(r"data\landuse_landcover\landcover\water_ch\Typisierung_LV95\typisierung.gpkg")
    ###############################################################################################################

    print("ready to fill")

    tot_num = num_cols * num_cols
    count=0

    for row in range(num_rows):
        for col in range(num_cols):

            #print(row, " - ", col)
            # Find the bounds of the cell
            cell_bounds = box(minx + col * resolution,
                              maxy - row * resolution,
                              minx + (col + 1) * resolution,
                              maxy - (row + 1) * resolution)

            # Find the roads that intersect with this cell
            #print(gdf_combined.head(10).to_string())
            intersecting_roads = gdf_combined[gdf_combined.intersects(cell_bounds)]

            # Debugging print
            #print(f"Cell {row},{col} intersects with {len(intersecting_roads)} roads")

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
    with rasterio.open(r"data\landuse_landcover\processed\unproductive_area.tif") as src2:
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

def match_access_point_on_highway(idx, raster):
    # get value of all idx in raster cell
    # initialise dict
    # for i in idx
    #   if value of i < 120
    #       if there is a cell A with raster value == 120 in 8 neighbors of i
    #           replace idx of i = idx of cell A
    #           dict.add(idx of A: idx of i)
    #       elif value of i < 100
    #           if there is a cell B with raster value == 100 in 8 neighbors of i
    #               replace idx of i = idx of cell B
    #               dict.add(idx of B: idx of i)
    #       elif value of i < 80
    #           if there is a cell C with raster value == 80 in 8 neighbors of i
    #               replace idx of i = idx of cell C
    #               dict.add(idx of C: idx of i)
    # return idx, dict

    matched_dict = {}
    updated_idx = []

    for i in idx:
        y, x = i
        value = raster[y, x]
        match_found = False  # Flag to indicate if a match is found

        if value < 80:
            # First search in the immediate neighborhood
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < raster.shape[0] and 0 <= nx < raster.shape[1]:
                        if raster[ny, nx] >= 100:
                            matched_dict[(ny, nx)] = i
                            i = (ny, nx)
                            match_found = True
                            break
                        elif raster[ny, nx] >= 80:
                            matched_dict[(ny, nx)] = i
                            i = (ny, nx)
                            match_found = True
                            break
                        elif raster[ny, nx] >= 50:
                            matched_dict[(ny, nx)] = i
                            i = (ny, nx)
                            match_found = True
                            break
                        elif raster[ny, nx] >= 30:
                            matched_dict[(ny, nx)] = i
                            i = (ny, nx)
                            match_found = True
                            break
                if match_found:
                    break

            # If no match found, expand search to wider range
            if not match_found:
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dy == 0 and dx == 0:  # Skip the cell itself
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < raster.shape[0] and 0 <= nx < raster.shape[1]:
                            if raster[ny, nx] >= 100:
                                matched_dict[(ny, nx)] = i
                                i = (ny, nx)
                                break
                            elif raster[ny, nx] >= 80:
                                matched_dict[(ny, nx)] = i
                                i = (ny, nx)
                                break
                            elif raster[ny, nx] >= 50:
                                matched_dict[(ny, nx)] = i
                                i = (ny, nx)
                                match_found = True
                                break
                            elif raster[ny, nx] >= 30:
                                matched_dict[(ny, nx)] = i
                                i = (ny, nx)
                                match_found = True
                                break
                    if match_found:
                        print("No point found to match on network")
                        break

        updated_idx.append(i)

    return updated_idx, matched_dict


def raster_to_graph(raster_data):
    # high_weight = 90 # sec  is the time required to cross 100 with 4 km/h
    raster_cell = 100  # m

    # convert travel speed from km/h to m/s
    raster_data = raster_data * 1000 / 3600

    rows, cols = raster_data.shape
    graph = nx.grid_2d_graph(rows, cols)

    nodes_to_remove = []
    for node in graph.nodes:
        y, x = node
        if raster_data[y, x] == 0:
            nodes_to_remove.append(node)

    graph.remove_nodes_from(nodes_to_remove)

    # Add weights for existing edges in the grid_2d_graph
    for (node1, node2) in graph.edges:
        y1, x1 = node1
        y2, x2 = node2
        if raster_data[y1, x1] == 0 or raster_data[y2, x2] == 0:
            # Assign a high weight to this edge
            weight = None
        else:
            # Calculate weight normally
            weight = (raster_cell / raster_data[y1, x1] + raster_cell / raster_data[y2, x2]) / 2

        # weight = (0.1 / raster_data[y1, x1] + 0.1 / raster_data[y2, x2]) / 2 * 3600
        graph[node1][node2]['weight'] = weight

    # Add diagonal edges (from 4 to 8 neighbors)
    new_edges = []
    for x in range(cols - 1):
        for y in range(rows - 1):
            # Check for zero values in raster data for diagonal neighbors
            if raster_data[y, x] == 0 or raster_data[y + 1, x + 1] == 0:
                weight = None
            else:
                weight = 1.4 * (raster_cell / raster_data[y, x] + raster_cell / raster_data[y + 1, x + 1]) / 2

            new_edges.append(((y, x), (y + 1, x + 1), {'weight': weight}))

            if raster_data[y, x + 1] == 0 or raster_data[y + 1, x] == 0:
                weight = None
            else:
                weight = 1.4 * (raster_cell / raster_data[y, x + 1] + raster_cell / raster_data[y + 1, x]) / 2

            new_edges.append(((y, x + 1), (y + 1, x), {'weight': weight}))

    # Add new diagonal edges with calculated weights
    graph.add_edges_from(new_edges)

    # iterate over all options
    # get the closest point
    return graph


def travel_cost_polygon(frame):

    points_all = gpd.read_file(r"data\Network\processed\points_attribute.gpkg")
    # Need the node id as ID_point
    points_all = points_all[points_all["intersection"] == 0]
    points_all_frame = points_all.cx[frame[0]:frame[2], frame[1]:frame[3]]
    # print(points_all_frame.head(10).to_string())

    # travel speed
    raster_file = r"data\Network\OSM_tif\speed_limit_raster.tif"
    # should change lake speed to 0
    # and other area to slightly higher speed to other land covers
    with rasterio.open(raster_file) as dataset:
        raster_data = dataset.read(1)  # Assumes forbidden cells are marked with 1 or another distinct value
        transform = dataset.transform

        # Convert real-world coordinates to raster indices
        sources_indices = [~transform * (x, y) for x, y in zip(points_all_frame.geometry.x, points_all_frame.geometry.y)]
        sources_indices = [(int(y), int(x)) for x, y in sources_indices]
        """
        # Calculate path lengths using Dijkstra's algorithm
        start = time.time()
        path_lengths, source_index = nx.multi_source_dijkstra_path_length(graph, sources_indices, weight='weight')
        end = time.time()
        print(f"Time dijkstra: {end-start} sec.")

        # Initialize an empty raster for path lengths
        path_length_raster = np.full(raster_data.shape, np.nan)
        source_raster = np.full(raster_data.shape, np.nan)

        # Populate the raster with path lengths
        for node, length in path_lengths.items():
            y, x = node
            path_length_raster[y, x] = length
        """

        sources_indices, idx_correct = match_access_point_on_highway(sources_indices, raster_data)
        # Remove all cells that contain highway
        #raster_data[raster_data > 90] = 50


        start = time.time()
        # Convert raster to graph
        graph = raster_to_graph(raster_data)
        end = time.time()
        print(f"Time to initialize graph: {end-start} sec.")

        start = time.time()
        # Get both path lengths and paths
        distances, paths = nx.multi_source_dijkstra(G=graph, sources=sources_indices, weight='weight')
        end = time.time()
        print(f"Time dijkstra: {end - start} sec.")

        # Initialize empty rasters for path lengths and source coordinates
        path_length_raster = np.full(raster_data.shape, np.nan)

        # Initialize an empty raster with np.nan and dtype float
        temp_raster = np.full(raster_data.shape, np.nan, dtype=float)
        # Change the dtype to object
        source_coord_raster = temp_raster.astype(object)

        # Populate the rasters
        for node, path in paths.items():
            y, x = node
            path_length_raster[y, x] = distances[node]

            if path:  # Check if path is not empty
                source_y, source_x = path[0]  # First element of the path is the source
                source_coord_raster[y, x] = (source_y, source_x)


    # Save the path length raster
    with rasterio.open(
            r'data\Network\travel_time\travel_time_raster.tif', 'w',
            driver='GTiff',
            height=path_length_raster.shape[0],
            width=path_length_raster.shape[1],
            count=1,
            dtype=path_length_raster.dtype,
            crs=dataset.crs,
            transform=transform
    ) as new_dataset:
        new_dataset.write(path_length_raster, 1)

    # Inverse transform to convert CRS coordinates to raster indices
    inv_transform = ~transform

    # Convert the geometry coordinates to raster indices
    points_all_frame['raster_x'], points_all_frame['raster_y'] = zip(*points_all_frame['geometry'].apply(lambda geom: inv_transform * (geom.x, geom.y)))
    points_all_frame["raster_y"] = points_all_frame["raster_y"].apply(lambda x: int(np.floor(x)))
    points_all_frame["raster_x"] = points_all_frame["raster_x"].apply(lambda x: int(np.floor(x)))

    # Create a dictionary to map raster indices to ID_point
    index_to_id = {(row['raster_y'], row['raster_x']): row['ID_point'] for _, row in points_all_frame.iterrows()}

    # Iterate over the source_coord_raster and replace coordinates with ID_point
    # Assuming new_array is your 2D array of coordinates and matched_dict is your dictionary


    for (y, x), coord in np.ndenumerate(source_coord_raster):
        if coord in idx_correct:
            source_coord_raster[y, x] = idx_correct[coord]

    for (y, x), source_coord in np.ndenumerate(source_coord_raster):
        if source_coord in index_to_id:
            source_coord_raster[y, x] = index_to_id[source_coord]

    # Convert the array to a float data type
    source_coord_raster = source_coord_raster.astype(float)
    # Set NaN values to a specific NoData value, e.g., -1
    source_coord_raster[np.isnan(source_coord_raster)] = -1

    path_id_raster = r'data\Network\travel_time\source_id_raster.tif'
    with rasterio.open(path_id_raster, 'w',
        driver='GTiff',
        height=source_coord_raster.shape[0],
        width=source_coord_raster.shape[1],
        count=1,
        dtype=source_coord_raster.dtype,
        crs=dataset.crs,
        transform=transform
        ) as new_dataset:
            new_dataset.write(source_coord_raster, 1)

    # get Voronoi polygons in vector data as gpd df
    gdf_polygon = raster_to_polygons(path_id_raster)
    #print(gdf_polygon.head(10).to_string())
    gdf_polygon.to_file(r"data\Network\travel_time\Voronoi_statusquo.gpkg")

        # how to get the inputs? nodes in which reference system, weights automatically?
        # how to get the coordinates of the closest point?

        # tif with travel time
        # tif with closest point


    return

def raster_to_polygons(tif_path):
    # Read the raster data
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        data = data.astype('int32')
        transform = src.transform

    # Find unique positive values in the raster
    unique_values = np.unique(data[data >= 0])

    # Create a mask for negative values (holes)
    negative_mask = data < 0

    # Initialize list to store polygons and their values
    polygons = []

    # Iterate over unique values and create polygons
    for val in unique_values:
        # Create mask for the current value
        positive_mask = data == val

        # Generate shapes (polygons) for positive values
        positive_shapes = rasterio.features.shapes(data, mask=positive_mask, transform=transform)

        # Generate shapes for negative values (holes)
        hole_shapes = rasterio.features.shapes(data, mask=negative_mask, transform=transform)

        # Combine positive shapes and holes
        combined_polygons = []
        for shape, value in positive_shapes:
            if value == val:
                outer_polygon = Polygon(shape['coordinates'][0])

                # Create holes
                holes = [Polygon(hole_shape['coordinates'][0]) for hole_shape, hole_value in hole_shapes if hole_value < 0]
                holes_union = unary_union(holes)

                # Combine outer polygon with holes
                if holes_union.is_empty:
                    combined_polygons.append(outer_polygon)
                else:
                    combined_polygon = outer_polygon.difference(holes_union)
                    combined_polygons.append(combined_polygon)

        # Add combined polygons to list
        polygons.extend([{'geometry': poly, 'ID_point': val} for poly in combined_polygons if not poly.is_empty])

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(polygons, crs=src.crs)
    gdf_dissolved = gdf.dissolve(by='ID_point')

    return gdf_dissolved

def GetRasterNetwork(outer):
    nw_from_osm(outer)
    osm_nw_to_raster(outer)
    return
def GetVoronoiCells(limits,outer,inner):
    travel_cost_polygon(limits)
    return

def GetNetworkNodes():
    node_table = pd.read_csv(r"data\Network\Road_Node.csv", sep=";")
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


def load_nw():
    """
    This function reads the data of the network. The data are given as table of nodes, edges and edges attributes. By
    merging these datasets the topological relationships of the network are created. It is then stored as shapefile.

    Parameters
    ----------
    :param lim: List of coordinated defining the limits of the plot [min east coordinate,
    max east coordinate, min north coordinate, max north coordinate]
    :return:
    """

    # Read csv files of node, links and link attributes to a Pandas DataFrame
    edge_table = pd.read_csv(r"data\Network\Road_Link.csv", sep=";")
    node_table = pd.read_csv(r"data\Network\Road_Node.csv", sep=";")
    link_attribute = pd.read_csv(r"data\Network\Road_LinkType.csv", sep=";")

    # Add coordinates of the origin node of each link by merging nodes and links through the node ID
    edge_table = pd.merge(edge_table, node_table, how="left", left_on="From Node", right_on="Node NR").rename(
        {'XKOORD': 'E_KOORD_O', 'YKOORD': 'N_KOORD_O'}, axis=1)
    # Add coordinates of the destination node of each link by merging nodes and links through the node ID
    edge_table = pd.merge(edge_table, node_table, how="left", left_on="To Node", right_on="Node NR").rename(
        {'XKOORD': 'E_KOORD_D', 'YKOORD': 'N_KOORD_D'}, axis=1)
    # Keep only relevant attributes for the edges
    edge_table = edge_table[['Link NR', 'From Node', 'To Node', 'Link Typ', 'Length (meter)', 'Number of Lanes',
                             'Capacity per day', 'V0IVFreeflow speed', 'Opening Year', 'E_KOORD_O', 'N_KOORD_O',
                             'E_KOORD_D', 'N_KOORD_D']]
    # Add the link attributes to the table of edges
    edge_table = pd.merge(edge_table, link_attribute[['Link Typ', 'NAME', 'Rank', 'Lanes', 'Capacity',
                                                      'Free Flow Speed']], how="left", on="Link Typ")

    # Convert single x and y coordinates to point geometries
    edge_table["point_O"] = [Point(xy) for xy in zip(edge_table["E_KOORD_O"], edge_table["N_KOORD_O"])]
    edge_table["point_D"] = [Point(xy) for xy in zip(edge_table["E_KOORD_D"], edge_table["N_KOORD_D"])]

    # Create LineString geometries for each edge based on origin and destination points
    edge_table['line'] = edge_table.apply(lambda row: LineString([row['point_O'], row['point_D']]), axis=1)

    # Filter infrastructure which was not built before 2023
    # edge_table = edge_table[edge_table['Opening Year']< 2023]
    edge_table = edge_table[(edge_table["Rank"] == 1) & (edge_table["Opening Year"] < 2023) & (edge_table["NAME"] != 'Freeway Tunnel planned') & (
                edge_table["NAME"] != 'Freeway planned')]

    # Initialize a Geopandas DataFrame based on the table of edges
    nw_gdf = gpd.GeoDataFrame(edge_table, geometry=edge_table.line, crs='epsg:21781')

    # Define and convert the coordinate reference system of the network form LV03 to LV95
    nw_gdf = nw_gdf.set_crs('epsg:21781')
    nw_gdf = nw_gdf.to_crs(2056)

    #boundline =
    #intersect_gdf = nw_gdf[(nw_gdf.touches(boundline))]

    # Drop unwanted columns and store the network DataFrame as shapefile
    nw_gdf = nw_gdf.drop(['point_O','point_D', "line"], axis=1)
    nw_gdf.to_file(r"data/temp/network_highway.gpkg")

    return


def GetTAZBounds(boundary,margin=2000):
    minx = boundary.bounds.minx.values.astype(float)
    miny = boundary.bounds.miny.values.astype(float)
    maxx = boundary.bounds.maxx.values.astype(float)
    maxy = boundary.bounds.maxy.values.astype(float)
    bbox = box(minx - margin, miny - margin, maxx + margin, maxy + margin)
    return bbox



def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : scipy.spatial.Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : ndarray
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    Source: https://stackoverflow.com/a/20678647
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges.get(p1, [])
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge
            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            # Append to the new region
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # store region
        new_regions.append(new_region.tolist())

    # Remove infinite vertices
    # finite_vertices = [v for i, v in enumerate(new_vertices) if i not in vor.vertices]

    return new_regions, np.asarray(new_vertices)
def GetTAZBasis(bbox):
    pointlist = [44,46,9,16,17,146,104,102,81]
    pointsdf = points_all[points_all["ID_point"].isin(pointlist)].copy()
    points = pointsdf.geometry
    coords = np.array([[p.x, p.y] for p in points])
    vor = Voronoi(coords)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    df_voronoi = gpd.GeoDataFrame(pd.DataFrame({"ID_point":pointlist}),geometry=gpd.GeoSeries([Polygon(vertices[region]) for region in regions]),crs="epsg:2056")
    df_voronoi = gpd.GeoDataFrame(df_voronoi.intersection(bbox))
    df_voronoi["code"] = [10001,10002,10003,10004,10005,10006,10007,10008,10009]
    df_voronoi["ID_point"] = pointlist
    df_voronoi["geometry"] = df_voronoi[0]
    df_voronoi = df_voronoi.iloc[:, 1:]
    df_voronoi.to_file(r"data\Voronoi\voronoi_basis.gpkg")
    return df_voronoi

def GetTAZwithCommunes(basis):
    boundary_geom = Polygon(cantonshape.geometry.iloc[0].exterior)
    # Create an empty list to collect new geometries
    new_geoms = []

    # Optional: keep original attributes
    attributes = []

    for idx, row in basis.iterrows():
        original_geom = row.geometry
        diff_geom = original_geom.difference(boundary_geom)

        # Only keep if the result is not empty
        if not diff_geom.is_empty:
            new_geoms.append(diff_geom)
            attributes.append(row.drop('geometry'))  # Keep all other columns

    # Rebuild GeoDataFrame with geometries outside the boundary
    result_df = pd.DataFrame(attributes)
    result_geom = pd.DataFrame(new_geoms)
    result_geom.columns=["geometry"]
    result_gdf = gpd.GeoDataFrame(pd.concat([result_df,result_geom],axis=1), crs=basis.crs)

    outsidetazdf = pd.DataFrame({"code":result_gdf["code"],"name":0,"ID_point":result_gdf["ID_point"],"geometry":result_gdf["geometry"]})
    insidetazdf = pd.DataFrame({"code":communedf["BFS"],"name":communedf["GEMEINDENA"],"ID_point":-99,"geometry":communedf["geometry"]})
    #tazgdf = gpd.GeoDataFrame(columns=tazgdf.columns, crs=tazgdf.crs)
    tazgdf = gpd.GeoDataFrame(pd.concat([outsidetazdf, insidetazdf], ignore_index=True))

    plot_this = 1
    if plot_this == 1:
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        # 1. Original GeoDataFrame
        basis.plot(ax=ax[0], color='lightblue', edgecolor='black')
        ax[0].set_title('Tessellation based on GVM\'s outside zones \n and their closest Highway access point ')
        # 3. Result after difference (outside boundary only)
        result_gdf.plot(ax=ax[1], color='lightgreen', edgecolor='black')
        ax[1].set_title('Polygons Outside Boundary')
        # 4. Result after communes
        tazgdf.plot(ax=ax[2], color='red', edgecolor='black')
        ax[2].set_title('Outside Polygons and Communes')
        for a in ax:
            a.set_axis_off()
        plt.tight_layout()
        plt.show()
    return tazgdf
def getTAZs():
    bbox = GetTAZBounds(cantonshape.boundary)
    basis = GetTAZBasis(bbox)
    taz = GetTAZwithCommunes(basis)
    return taz

#0. Define extent
# Define spatial limits of the research corridor
# The coordinates must end with 000 in order to match the coordinates of the input raster data
e_min, e_max = 2687000, 2708000     # 2688000, 2704000 - 2688000, 2705000
n_min, n_max = 1237000, 1254000     # 1238000, 1252000 - 1237000, 1252000
limits_corridor = [e_min, n_min, e_max, n_max]

# Boudary for plot
boundary_plot = polygon_from_points(e_min=e_min+1000, e_max=e_max-500, n_min=n_min+1000, n_max=n_max-2000)

# Get a polygon as limits for teh corridor
inner = polygon_from_points(e_min=e_min, e_max=e_max, n_min=n_min, n_max=n_max)

# For global operation a margin is added to the boundary
margin = 7000 # meters
outer = polygon_from_points(e_min=e_min, e_max=e_max, n_min=n_min, n_max=n_max, margin=margin)
limits = [e_min-margin, n_min-margin, e_max+margin, n_max+margin]

#boundary = LineString([[e_min,n_min],[e_max,n_min],[e_max,n_max],[e_min,n_max],[e_min,n_min]])

#1.	define TAZs defined in GVM

raster_path = r"data/Network/travel_time/source_id_raster.tif"
points_all = gpd.read_file(r"data\Network\processed\points.gpkg")

commune_raster, communedf = GetCommuneShapes(raster_path)
cantonshape = communedf.dissolve()
tazgdf = getTAZs()

tazgdf["scalecategory"] = 3
tazgdf.loc[tazgdf["code"]>9999,"scalecategory"] = 4
tazgdf.loc[tazgdf.within(outer),"scalecategory"] = 2
tazgdf.loc[tazgdf.within(inner),"scalecategory"] = 1


#2.	categorise TAZs as
##a.	within extent
##b.	within outer bounary
##c.	within canton zurich
##d.	within GVM
#3.	Define trips between TAZs(use bfs)
od = GetDemandPerCommune(tau=1, mode='miv')
#4.	Set up OD matrix (using bfs)
odmat = GetODMatrix(od)
#5.	Identify ‚access points‘ to the extent network
GetRasterNetwork(limits)
GetVoronoiCells(limits,outer,inner)

### todo: start from here to go from communal OD to newly tesselated OD for scoring
#6.	Identify ‘access points‘ to the canton‘s network from the external tazs
points_in_outer = points_all[points_all.within(outer)]
points_outside_inner = points_in_outer[~points_in_outer.within(inner)]

#7.	For each taz fully outside extent, identify shortest path to an extent access point and allocte the taz to that point.
##a.	Sum the trips in each of the tazs
#8.	For all taz fully outside the extent Join all tazs that correspond to each access point.
#9.	Make sure future grwoth is only distributed pauschal to TAZ outside the canton.

popvec = GetCommunePopulation(y0="2021")
jobvec = GetCommuneEmployment(y0=2021)




### Define zones
### Check for 2050

