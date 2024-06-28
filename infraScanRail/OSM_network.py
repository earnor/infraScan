import os
import glob
import time
import geopandas as gpd
import pandas as pd
import networkx as nx
import rasterio
import rasterio.features
import numpy as np
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon


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


def raster_to_polygons___(tif_path):
    # Read the raster data
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        data = data.astype('int32')
        transform = src.transform

    # Find unique values in the raster
    unique_values = np.unique(data[data >= 0])  # Assuming negative values are no-data

    # Initialize list to store polygons and their values
    polygons = []

    # Iterate over unique values and create polygons
    for val in unique_values:
        # Create mask for the current value
        mask = data == val

        # Generate shapes (polygons) from the mask
        shapes = rasterio.features.shapes(data, mask=mask, transform=transform)
        for shape, value in shapes:
            if value == val:
                # Convert shape to a Shapely Polygon and add to list
                polygons.append({
                    'geometry': Polygon(shape['coordinates'][0]),
                    'ID_point': val
                })

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(polygons, crs=src.crs)
    #gdf_dissolved = gdf.dissolve(by='ID_point')
    gdf_dissolved = groupby_multipoly(gdf, by="ID_point")

    return gdf_dissolved


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


def groupby_multipoly(df, by, aggfunc="first"):
    data = df.drop(labels=df.geometry.name, axis=1)
    aggregated_data = data.groupby(by=by).agg(aggfunc)

    # Process spatial component
    def merge_geometries(block):
        return MultiPolygon(block.values)

    g = df.groupby(by=by, group_keys=False)[df.geometry.name].agg(
        merge_geometries
    )

    # Aggregate
    aggregated_geometry = gpd.GeoDataFrame(g, geometry=df.geometry.name, crs=df.crs)
    # Recombine
    aggregated = aggregated_geometry.join(aggregated_data)
    return aggregated


def raster_to_graph(raster_data):
    #high_weight = 90 # sec  is the time required to cross 100 with 4 km/h
    raster_cell = 100 # m

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

        #weight = (0.1 / raster_data[y1, x1] + 0.1 / raster_data[y2, x2]) / 2 * 3600
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


def travel_cost_developments(frame):
    # First delete all elements that are in the folder where the files are stored to avoid doubling

    files = glob.glob(r'data\Network\travel_time\developments/*')
    for f in files:
        os.remove(f)

    points = gpd.read_file(r"data\Network\processed\points_attribute.gpkg")
    # Need the node id as ID_point
    #points = points[points["intersection"] == 0]
    points = points.cx[frame[0]:frame[2], frame[1]:frame[3]]

    generated_points = gpd.read_file(r"data\Network\processed\generated_nodeset.gpkg")

    # travel speed
    raster_file = r"data\Network\OSM_tif\speed_limit_raster.tif"
    # should change lake speed to 0
    # and other area to slightly higher speed to other land covers
    with rasterio.open(raster_file) as dataset:
        raster_data = dataset.read(1)  # Assumes forbidden cells are marked with 1 or another distinct value
        transform = dataset.transform

        # Iterate over all developments
        for index, row in generated_points.iterrows():
            geometry = row.geometry  # Access geometry
            id_new = row['ID_new']  # Access value in ID_new column
            print(f"Development {id_new}")

            # Create a new row to add to the target GeoDataFrame
            new_row = {'geometry': geometry, 'intersection': 0, 'ID_point': 9999} # , 'ID_new': id_new

            # Append new row to target_gdf
            temp_points = points.copy()
            #temp_points = temp_points.append(new_row, ignore_index=True)
            # geometries.append(tempgeom)
            temp_points = gpd.GeoDataFrame(pd.concat([temp_points, pd.DataFrame(pd.Series(new_row)).T], ignore_index=True))

            # Convert real-world coordinates to raster indices
            sources_indices = [~transform * (x, y) for x, y in zip(temp_points.geometry.x, temp_points.geometry.y)]
            sources_indices = [(int(y), int(x)) for x, y in sources_indices]

            sources_indices, idx_correct = match_access_point_on_highway(sources_indices, raster_data)
            # Remove all cells that contain highway
            # raster_data[raster_data > 90] = 50

            start = time.time()
            # Convert raster to graph
            graph = raster_to_graph(raster_data)

            # Get both path lengths and paths
            distances, paths = nx.multi_source_dijkstra(G=graph, sources=sources_indices, weight='weight')
            end = time.time()
            print(f"Initialize graph and running dijkstra: {end - start} sec.")

            # Initialize empty rasters for path lengths and source coordinates
            path_length_raster = np.full(raster_data.shape, np.nan)

            # Initialize an empty raster with np.nan and dtype float
            temp_raster = np.full(raster_data.shape, np.nan, dtype=float)
            # Change the dtype to object
            source_coord_raster = temp_raster.astype(object)

            # Populate the raster
            for node, path in paths.items():
                y, x = node
                path_length_raster[y, x] = distances[node]

                if path:  # Check if path is not empty
                    source_y, source_x = path[0]  # First element of the path is the source
                    source_coord_raster[y, x] = (source_y, source_x)

            # Save the path length raster
            with rasterio.open(
                    fr'data\Network\travel_time\developments\dev{id_new}_travel_time_raster.tif', 'w',
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
            temp_points['raster_x'], temp_points['raster_y'] = zip(*temp_points['geometry'].apply(lambda geom: inv_transform * (geom.x, geom.y)))
            temp_points["raster_y"] = temp_points["raster_y"].apply(lambda x: int(np.floor(x)))
            temp_points["raster_x"] = temp_points["raster_x"].apply(lambda x: int(np.floor(x)))

            # Create a dictionary to map raster indices to ID_point
            index_to_id = {(row['raster_y'], row['raster_x']): row['ID_point'] for _, row in temp_points.iterrows()}

            # Iterate over the source_coord_raster and replace coordinates with ID_point
            # Assuming new_array is your 2D array of coordinates and matched_dict is your dictionary
            for (y, x), coord in np.ndenumerate(source_coord_raster):
                if coord in idx_correct:
                    source_coord_raster[y, x] = idx_correct[coord]

            for (y, x), source_coord in np.ndenumerate(source_coord_raster):
                if source_coord in index_to_id:
                    source_coord_raster[y, x] = index_to_id[source_coord]

            # Make sure there are no tuples as point ID
            for (y, x), value in np.ndenumerate(source_coord_raster):
                # Check if the value is a tuple or an array (or another iterable except strings)
                if isinstance(value, (tuple, list, np.ndarray)):
                    # Keep only the first value of the tuple/array
                    source_coord_raster[y, x] = value[0]
                    print(f"Index ({value}) replace by {value[0]}")
                elif np.isnan(value):
                    source_coord_raster[y, x] = -1
                    pass

            # Convert the array to a float data type
            source_coord_raster = source_coord_raster.astype(float)
            # Set NaN values to a specific NoData value, e.g., -1
            source_coord_raster[np.isnan(source_coord_raster)] = -1

            path_id_raster = fr'data\Network\travel_time\developments\dev{id_new}_source_id_raster.tif'
            with rasterio.open(
                path_id_raster, 'w',
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
            # print(gdf_polygon.head(10).to_string())
            gdf_polygon.to_file(fr"data\Network\travel_time\developments\dev{id_new}_Voronoi.gpkg")
                # how to get the inputs? nodes in which reference system, weights automatically?
                # how to get the coordinates of the closest point?

                # tif with travel time
                # tif with closest point
    return


def get_voronoi_frame(polygons_gdf):
    margin = 100
    points_gdf = gpd.read_file(r"data\Network\processed\points_corridor_attribute.gpkg")
    points_gdf = points_gdf[points_gdf["intersection"] == 0]

    points_all = gpd.read_file(r"data\Network\processed\points.gpkg")
    points_all.crs = "epsg:2056"
    points_all = points_all[points_all["intersection"] == 0]

    # union of all polygons from points
    # get all polygons touching it
    # get its extrem values

    # Step 1: Identify polygons containing points
    points_gdf = points_gdf.drop(columns=["index_right"])
    polygons_with_points = gpd.sjoin(polygons_gdf, points_gdf, predicate='contains').drop_duplicates(
        subset=polygons_gdf.index.name)
    polygons_with_points = polygons_with_points[["ID_point", "geometry"]]
    polygons_with_points = polygons_with_points.drop_duplicates()
    # Use unary_union to union all geometries into a single geometry
    #polygons_with_points = unary_union(polygons_with_points['geometry'])
    #polygons_with_points = gpd.GeoDataFrame(geometry=[polygons_with_points], crs="epsg:2056")
    #polygons_with_points.to_file(r"data\Network\processed\ppg.gpkg")

    # Step 2: Find polygons touching the identified set
    # Add custom suffixes to avoid naming conflicts
    touching_polygons = gpd.sjoin(polygons_gdf, polygons_with_points, how='inner', predicate='touches', lsuffix='left',
                                  rsuffix='_right')

    # Combine the identified polygons and the ones touching them
    #combined_polygons = pd.concat([polygons_with_points, touching_polygons]).drop_duplicates(subset=polygons_gdf.index.name)

    # Step 3: Extract points contained in the combined set of polygons

    points_in_polygons = gpd.sjoin(points_all, touching_polygons, predicate='within', lsuffix='_l',
                                  rsuffix='r')
    points_in_polygons = points_in_polygons[["geometry", "index_r"]]
    points_in_polygons = points_in_polygons.drop_duplicates()

    # Step 4: Calculate extreme values
    xmin, ymin, xmax, ymax = points_in_polygons.total_bounds

    return [xmin-margin, ymin-margin, xmax+margin, ymax+margin]



