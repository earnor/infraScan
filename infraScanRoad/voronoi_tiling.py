import pandas as pd
from scipy.spatial import Voronoi
import osmnx as ox
from pyproj import Transformer
import sys

from data_import import *
from plots import *

# Get the parent directory of GUI (i.e., InfraScan)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)  # Add InfraScan to Python's module search path
from logging_config import logger  # Import central logger


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


def get_voronoi_status_quo():
    existing_nodes = gpd.read_file(r"data\Network\processed\points.gpkg")
    existing_nodes = existing_nodes.set_crs("epsg:2056")

    existing_nodes = existing_nodes[existing_nodes["intersection"] == 0]

    existing_temp = existing_nodes["geometry"]
    coordinates_array = np.array(existing_temp.apply(lambda geom: (geom.x, geom.y)).tolist())

    # generate the Voronoi diagram
    vor = Voronoi(coordinates_array)

    regions, vertices = voronoi_finite_polygons_2d(vor)
    df_voronoi = gpd.GeoDataFrame(geometry=gpd.GeoSeries([Polygon(vertices[region]) for region in regions]),
                                  crs="epsg:2056")
    # df_voronoi["ID"] = 1
    print(df_voronoi.head(10).to_string())

    df_voronoi.to_file(r"data\Voronoi\voronoi_status_quo_euclidian.gpkg")

    return


def get_voronoi_all_developments():
    existing_nodes = gpd.read_file(r"data\Network\processed\points.gpkg")
    existing_nodes = existing_nodes.set_crs("epsg:2056")
    new_nodes = gpd.read_file(r"data\Network\processed\generated_nodes.gpkg")

    voronoi_developments = pd.DataFrame(columns=['ID', 'geometry'])
    voronoi_developments = gpd.GeoDataFrame(voronoi_developments, geometry="geometry", crs="epsg:2056")

    neighboring_points = pd.DataFrame(columns=['ID', 'geometry'])
    neighboring_points = gpd.GeoDataFrame(neighboring_points, geometry="geometry", crs="epsg:2056")

    for i in new_nodes["ID_new"].unique():
        existing_temp = existing_nodes["geometry"]
        new_temp = new_nodes[new_nodes["ID_new"] == i]
        new_temp['geom']=new_temp['geometry']
        one_development = gpd.GeoDataFrame(pd.concat([pd.DataFrame(existing_temp), pd.DataFrame(new_temp["geometry"])],ignore_index=True))
        #one_development = existing_temp.append(new_temp["geometry"])
        # coordinates_array = np.array(one_development.apply(lambda geom: (geom.x, geom.y))).tolist()
        #coordinates_array = np.array(one_development["geometry"])#.tolist()
        xx = one_development.geometry.x
        yy = one_development.geometry.y
        x = xx.tolist()
        y = yy.tolist()
        coordinates_array = np.array(tuple(zip(x, y)))

        # generate the Voronoi diagram
        vor = Voronoi(coordinates_array)

        regions, vertices = voronoi_finite_polygons_2d(vor)
        df_voronoi = gpd.GeoDataFrame(geometry=gpd.GeoSeries([Polygon(vertices[region]) for region in regions]),
                                      crs="epsg:2056")
        df_voronoi["ID"] = int(i)

        access_within_corridor = gpd.read_file(r"data\Network\processed\points_corridor.gpkg")
        #access_within_corridor = access_within_corridor["geometry"].append(new_temp["geometry"])
        access_within_corridor = gpd.GeoDataFrame(pd.concat([pd.DataFrame(access_within_corridor["geometry"]),
                                                            pd.DataFrame(new_temp["geometry"])], ignore_index=True))
        access_within_corridor = gpd.GeoDataFrame(access_within_corridor)
        access_within_corridor = access_within_corridor.set_crs("epsg:2056")

        # Perform a spatial join to identify polygons containing points
        polygons_containing_points = gpd.sjoin(df_voronoi, access_within_corridor, how="inner", predicate="contains")
        polygons_containing_points = polygons_containing_points[df_voronoi.columns]

        voronoi_developments = pd.concat([voronoi_developments, polygons_containing_points])

        # Filter out polygons from df_voronoi that are not in polygons_containing_points
        common_polygons = gpd.sjoin(df_voronoi, polygons_containing_points, how='inner', predicate='within')
        df_voronoi_filtered = df_voronoi[~df_voronoi.index.isin(common_polygons.index)]

        # Find neighboring polygons
        # Iterate over polygons_containing_points and check for neighboring polygons in df_voronoi_filtered
        neighbors = []
        for _, poly in polygons_containing_points.iterrows():
            for _, candidate in df_voronoi_filtered.iterrows():
                if poly['geometry'].touches(candidate['geometry']):
                    neighbors.append(candidate)
                    #neighbors = gpd.GeoDataFrame(pd.concat([pd.DataFrame(neighbors), pd.DataFrame(candidate)], ignore_index=True),geometry='geometry')
        #neighbors.set_geometry('geometry')

        # Convert list of neighbors to a GeoDataFrame and remove duplicates if any
        neighboring_polygons = gpd.GeoDataFrame(neighbors, columns=df_voronoi_filtered.columns, crs="epsg:2056",geometry='geometry')
        neighboring_polygons['ID'] = neighboring_polygons.index
        neighboring_polygons = neighboring_polygons.drop_duplicates(subset=['geometry'])

        # Find points within neighboring_polygons
        points_in_neighbors = gpd.sjoin(existing_nodes, neighboring_polygons, how='inner', predicate='within')
        points_in_neighbors = points_in_neighbors[["geometry", "ID"]]
        neighboring_points = pd.concat([neighboring_points, points_in_neighbors])

    # Store the polygons as gpkg file
    voronoi_developments.to_file(r"data\Voronoi\voronoi_developments_euclidian.gpkg")

    # For the next steps of the work the perimeter must be adapted, thus we keep the bound of the points tha lie in the
    # neighbooring polygons of the polygons in the corridor
    bounds = neighboring_points['geometry'].total_bounds
    bounds_rounded = [round(math.floor(bounds[0] - 100), -2), round(math.floor(bounds[1] - 100), -2),
                        round(math.ceil(bounds[2] + 100), -2), round(math.ceil(bounds[3] + 100), -2)]

    return bounds_rounded

    # convert the geopandas dataframe into a numpy array of points
    #points = node_filter_gdf[["XKOORD", "YKOORD"]].to_numpy()


def nw_from_osm(limits):

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
            #sub_polygons.append(sub_polygon)
            sub_polygons = gpd.GeoDataFrame(pd.concat([pd.DataFrame(sub_polygons), pd.DataFrame(sub_polygon).T], ignore_index=True))

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