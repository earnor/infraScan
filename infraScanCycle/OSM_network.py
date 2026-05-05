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
from shapely.validation import make_valid


def make_cycling_speed_raster(cycling_speed_kmh=15):
    """
    Creates a uniform cycling speed raster from the existing speed_limit_raster.
    All passable cells (speed > 0) are replaced with cycling_speed_kmh.
    Impassable cells (speed == 0, e.g. lakes) remain 0.
    Result saved to data/Network/OSM_tif/cycling_speed_raster.tif
    """
    raster_file = r"data/Network/OSM_tif/speed_limit_raster.tif"
    output_file = r"data/Network/OSM_tif/cycling_speed_raster.tif"

    with rasterio.open(raster_file) as src:
        raster_data = src.read(1).astype(float)
        profile = src.profile

    # Replace all passable cells with the cycling speed
    cycling_raster = np.where(raster_data > 0, cycling_speed_kmh, 0).astype(float)

    with rasterio.open(output_file, 'w', **profile) as dst:
        dst.write(cycling_raster, 1)

    print(f"Cycling speed raster created at {output_file} ({cycling_speed_kmh} km/h)")
    return output_file


def travel_cost_polygon(frame, raster_file=r"data/Network/OSM_tif/cycling_speed_raster.tif"):
    points_all = gpd.read_file(r"data/Network/processed/access_points_corridor.gpkg")
    # Need the node id as ID_point
    points_all_frame = points_all.cx[frame[0]:frame[2], frame[1]:frame[3]]


    # travel speed
    # should change lake speed to 0
    # and other area to slightly higher speed to other land covers
    with rasterio.open(raster_file) as dataset:
        raster_data = dataset.read(1)  # Assumes forbidden cells are marked with 1 or another distinct value
        transform = dataset.transform

        # Convert real-world coordinates to raster indices
        sources_indices = [~transform * (x, y) for x, y in zip(points_all_frame.geometry.x, points_all_frame.geometry.y)]
        sources_indices = [(int(y), int(x)) for x, y in sources_indices]

        def snap_to_nonzero(idx, raster):
            """Snap source points on zero-speed cells to nearest valid neighbour."""
            corrected = {}
            updated = []
            for (y, x) in idx:
                if raster[y, x] > 0:
                    updated.append((y, x))
                else:
                    # Search expanding neighbourhood
                    found = False
                    for r in range(1, 5):
                        for dy in range(-r, r + 1):
                            for dx in range(-r, r + 1):
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < raster.shape[0] and 0 <= nx < raster.shape[1]:
                                    if raster[ny, nx] > 0:
                                        corrected[(ny, nx)] = (y, x)
                                        updated.append((ny, nx))
                                        found = True
                                        break
                            if found:
                                break
                        if found:
                            break
                    if not found:
                        updated.append((y, x))  # leave in place, Dijkstra will skip
            return updated, corrected

        sources_indices, idx_correct = snap_to_nonzero(sources_indices, raster_data)



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
            r'data/Network/travel_time/travel_time_raster.tif', 'w',
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

    path_id_raster = r'data/Network/travel_time/source_id_raster.tif'
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
    gdf_polygon.to_file(r"data/Network/travel_time/Voronoi_statusquo.gpkg")


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
                outer_polygon = make_valid(Polygon(shape['coordinates'][0]))
                holes = [make_valid(Polygon(hole_shape['coordinates'][0])) for hole_shape, hole_value in hole_shapes if
                         hole_value < 0]
                holes = [h for h in holes if not h.is_empty]
                holes_union = unary_union(holes) if holes else outer_polygon.__class__()

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


def raster_to_graph(raster_data, raster_cell=50):


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





def travel_cost_developments(frame, raster_file=r"data/Network/OSM_tif/cycling_speed_raster.tif"):
    os.makedirs('data/Network/travel_time/developments', exist_ok=True)

    files = glob.glob(r'data/Network/travel_time/developments/*')
    for f in files:
        os.remove(f)

    points = gpd.read_file(r"data/Network/processed/points_with_attribute.gpkg")
    points = points[points["is_intersection"] == False] if "is_intersection" in points.columns else points
    points = points.cx[frame[0]:frame[2], frame[1]:frame[3]]

    generated_points = gpd.read_file(r"data/Network/processed/generated_nodes.gpkg")
    generated_points = generated_points[generated_points["within_corridor"] | generated_points["on_border"]]

    with rasterio.open(raster_file) as dataset:
        raster_data = dataset.read(1)
        transform = dataset.transform
        inv_transform = ~transform
        cell_size = abs(dataset.transform.a)

        # ── Build graph ONCE ────────────────────────────────────────────────
        t0 = time.time()
        graph = raster_to_graph(raster_data, raster_cell=cell_size)
        print(f"Graph built in {time.time() - t0:.1f}s")

        # ── Status-quo Dijkstra ONCE (all existing access points) ───────────
        sq_raw = [inv_transform * (geom.x, geom.y) for geom in points.geometry]
        sq_indices = [(int(y), int(x)) for x, y in sq_raw]
        sq_indices = [(y, x) for y, x in sq_indices
                      if 0 <= y < raster_data.shape[0] and 0 <= x < raster_data.shape[1]]
        sq_indices, sq_idx_correct = match_access_point_on_cycling_network(sq_indices, raster_data)

        t0 = time.time()
        sq_distances, sq_paths = nx.multi_source_dijkstra(G=graph, sources=sq_indices, weight='weight')
        print(f"Status-quo Dijkstra done in {time.time() - t0:.1f}s  ({len(sq_distances)} reachable cells)")

        # Build sq distance and source-index arrays (shape = raster)
        sq_dist_arr = np.full(raster_data.shape, np.inf)
        sq_src_arr  = np.full(raster_data.shape, None, dtype=object)
        for node, dist in sq_distances.items():
            y, x = node
            sq_dist_arr[y, x] = dist
            path = sq_paths[node]
            sq_src_arr[y, x] = path[0] if path else node

        # Apply sq snapping corrections to source array
        for (y, x), src in np.ndenumerate(sq_src_arr):
            if src in sq_idx_correct:
                sq_src_arr[y, x] = sq_idx_correct[src]

        # Map raster indices → ID_point for status-quo sources
        points_copy = points.copy()
        points_copy['ry'], points_copy['rx'] = zip(*points_copy.geometry.apply(
            lambda g: (int(np.floor((inv_transform * (g.x, g.y))[1])),
                       int(np.floor((inv_transform * (g.x, g.y))[0])))))
        sq_index_to_id = {(r['ry'], r['rx']): r['ID_point'] for _, r in points_copy.iterrows()}
        # Also register snapped positions
        for snapped, orig in sq_idx_correct.items():
            if orig in sq_index_to_id:
                sq_index_to_id[snapped] = sq_index_to_id[orig]

        # Pre-build sq source-ID array (float) for fast per-dev reuse
        sq_src_id_arr = np.full(raster_data.shape, -1.0)
        for (y, x), src in np.ndenumerate(sq_src_arr):
            if src is not None and src in sq_index_to_id:
                sq_src_id_arr[y, x] = float(sq_index_to_id[src])

        # ── Per-development loop ─────────────────────────────────────────────
        for _, row in generated_points.iterrows():
            geometry = row.geometry
            id_new   = row['ID_new']
            print(f"Development {id_new}")

            # Snap new node to raster
            dx, dy = inv_transform * (geometry.x, geometry.y)
            dev_idx = (int(dy), int(dx))
            if not (0 <= dev_idx[0] < raster_data.shape[0] and 0 <= dev_idx[1] < raster_data.shape[1]):
                print(f"  Dev {id_new} outside raster — skipped")
                continue

            dev_snapped_list, dev_idx_correct = match_access_point_on_cycling_network([dev_idx], raster_data)
            dev_idx_snapped = dev_snapped_list[0]

            # Single-source Dijkstra from only the new development node
            t0 = time.time()
            dev_distances, dev_paths = nx.single_source_dijkstra(G=graph, source=dev_idx_snapped, weight='weight')
            print(f"  Dev {id_new} Dijkstra: {time.time() - t0:.1f}s")

            # Build dev distance array
            dev_dist_arr = np.full(raster_data.shape, np.inf)
            for node, dist in dev_distances.items():
                y, x = node
                dev_dist_arr[y, x] = dist

            # ── Merge: take element-wise minimum ──────────────────────────
            dev_wins = dev_dist_arr < sq_dist_arr

            path_length_raster = np.where(dev_wins, dev_dist_arr, sq_dist_arr)
            path_length_raster[np.isinf(path_length_raster)] = np.nan

            # Source-ID raster: dev cells get 9999, sq cells keep their ID
            source_id_raster = np.where(dev_wins, 9999.0, sq_src_id_arr)
            # Cells unreachable by both → -1
            both_inf = np.isinf(dev_dist_arr) & np.isinf(sq_dist_arr)
            source_id_raster[both_inf] = -1.0

            # Save travel time raster
            with rasterio.open(
                fr'data/Network/travel_time/developments/dev{id_new}_travel_time_raster.tif', 'w',
                driver='GTiff', height=path_length_raster.shape[0], width=path_length_raster.shape[1],
                count=1, dtype=path_length_raster.dtype, crs=dataset.crs, transform=transform
            ) as dst:
                dst.write(path_length_raster, 1)

            # Save source-ID raster
            path_id_raster = fr'data/Network/travel_time/developments/dev{id_new}_source_id_raster.tif'
            with rasterio.open(
                path_id_raster, 'w', driver='GTiff',
                height=source_id_raster.shape[0], width=source_id_raster.shape[1],
                count=1, dtype=source_id_raster.dtype, crs=dataset.crs, transform=transform
            ) as dst:
                dst.write(source_id_raster, 1)

            gdf_polygon = raster_to_polygons(path_id_raster)
            gdf_polygon.to_file(fr"data/Network/travel_time/developments/dev{id_new}_Voronoi.gpkg")

    return

def match_access_point_on_cycling_network(idx, raster):
    matched_dict = {}
    updated_idx = []
    for i in idx:
        y, x = i
        match_found = raster[y, x] > 0
        for radius in range(1, 4):
            if match_found:
                break
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < raster.shape[0] and 0 <= nx < raster.shape[1]:
                        if raster[ny, nx] > 0:
                            matched_dict[(ny, nx)] = i
                            i = (ny, nx)
                            match_found = True
                            break
                if match_found:
                    break
        updated_idx.append(i)
    return updated_idx, matched_dict


def get_voronoi_frame(polygons_gdf):
    margin = 100
    points_gdf = gpd.read_file(r"data/Network/processed/points_with_attribute.gpkg")
    points_gdf = points_gdf[points_gdf["intersection"] == 0]

    points_all = gpd.read_file(r"data/Network/processed/points.gpkg")
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
    #polygons_with_points.to_file(r"data/Network/processed/ppg.gpkg")

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



