import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import math
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point, MultiPoint, shape, box
from shapely.ops import split, snap, linemerge, cascaded_union
from rasterio import crs
from rasterio.transform import from_origin
from rasterio.features import shapes, rasterize
from shapely.geometry import shape, Polygon
from geopandas.tools import sjoin

from plots import *

# Get the parent directory of GUI (i.e., InfraScan)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)  # Add InfraScan to Python's module search path
from logging_config import logger  # Import central logger


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

    # Read the CSV file into a Pandas DataFrame
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
    csv_to_tiff(empl20_ch, attribute="empl20", path=r"data\independent_variable\processed\raw\empl20_ch.tif")

    empl20 = empl20_ch[(empl20_ch["E_COORD"] >= limits[0]) & (empl20_ch["E_COORD"] <= limits[2] - 100) & (empl20_ch["N_COORD"] >= limits[1]) & (empl20_ch["N_COORD"] <= limits[3] - 100)]
    print(empl20.head(10).to_string())
    csv_to_tiff(empl20, attribute="empl20", path=r"data\independent_variable\processed\raw\empl20.tif")

    population20 = pd.read_csv(r"data\independent_variable\statpop\ag-b-00.03-vz2020statpop\STATPOP2020.csv", sep=";")
    population20 = population20[["B20BTOT", "E_KOORD", "N_KOORD"]].rename({"B20BTOT": "pop20"}, axis=1)
    pop20_ch = fill_raster_dataframe(population20)
    csv_to_tiff(pop20_ch, attribute="pop20", path=r"data\independent_variable\processed\raw\pop20_ch.tif")
    pop20 = pop20_ch[
        (pop20_ch["E_COORD"] >= limits[0]) & (pop20_ch["E_COORD"] <= limits[2] - 100) & (pop20_ch["N_COORD"] >= limits[1]) & (
                    pop20_ch["N_COORD"] <= limits[3] - 100)]
    csv_to_tiff(pop20, attribute="pop20", path=r"data\independent_variable\processed\raw\pop20.tif")

    # Store the restructured dataset as csv file
    #population20.to_csv(r"data/temp/pop_filtered.csv")
    #betriebzaehlung20.to_csv(r"data/temp/empl_filtered.csv")
    return


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

    print(minX, ", ", maxX, ", ", minY, ", ", maxY)

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
    print(x_min, ", ", x_max, ", ", y_min,", ", y_max)
    print(data_table_sorted.shape)

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
    df_cities = pd.read_csv(r"data\manually_gathered_data\City_map.csv", sep=";")

    # Convert single values into coordinates of geopandas DataFrame and initialize the coordinate reference system
    gdf_cities = gpd.GeoDataFrame(df_cities, geometry=gpd.points_from_xy(df_cities["x"], df_cities["y"]),
                                  crs="epsg:2056")

    gdf_cities.to_file('data\manually_gathered_data\cities.shp', crs="epsg:2056")
    return


def get_lake_data():
    gdf = gpd.read_file(r"data\landuse_landcover\landcover\lake\WB_STEHGEWAESSER_F.shp")
    gdf = gdf[gdf["GEWAESSERN"].isin(["Zürichsee", "Greifensee", "Pfäffikersee"])]
    gdf.to_file('data\landuse_landcover\processed\lake_data_zh.gpkg', crs="epsg:2056")
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

    # Drop unwanted columns and store the network DataFrame as shapefile
    nw_gdf = nw_gdf.drop(['point_O','point_D', "line"], axis=1)
    nw_gdf.to_file(r"data/temp/network_highway.gpkg")

    return


def load_nw_zh():
    all_roads = gpd.read_file(r"data\Network\Strassennetz\TBA_STR_ACHS_L.shp")
    print(all_roads.columns)


def map_access_points_on_network(current_points, network):
    # As the access points were collected manually they do not match the infrastructure perfectly all the time. Thus,
    # each point is projected to the closest network segment
    # Create an empty GeoPandas dataframe to store the adjusted points
    current_points = gpd.GeoDataFrame(current_points, geometry=gpd.points_from_xy(current_points["x"], current_points["y"]),
                                  crs="epsg:2056")
    adjusted_points_gdf = gpd.GeoDataFrame(columns=current_points.columns)

    # Iterate over each point
    for index, point in current_points.iterrows():
        # Find the nearest line to the point
        nearest_line = network.distance(point.geometry).idxmin()

        # Get the nearest line geometry
        line_geometry = network.loc[nearest_line, 'geometry']

        # Project the point onto the line
        projected_point = line_geometry.interpolate(line_geometry.project(point.geometry))

        # Add the adjusted point to the new dataframe
        if network.distance(point.geometry).min() < 300:
            #adjusted_points_gdf = adjusted_points_gdf.append(point, ignore_index=True)
            adjusted_points_gdf = gpd.GeoDataFrame(pd.concat([pd.DataFrame(adjusted_points_gdf), pd.DataFrame(point).T],ignore_index=True))
                #pd.concat([adjusted_points_gdf, point.T], ignore_index=True))
            # Update the geometry of the adjusted point
            adjusted_points_gdf.at[index, 'geometry'] = projected_point


    #adjusted_points_gdf["point_name"] = gdf_access["name"]
    adjusted_points_gdf['x'] = adjusted_points_gdf.geometry.x
    adjusted_points_gdf['y'] = adjusted_points_gdf.geometry.y
    #adjusted_points_gdf.columns = adjusted_points_gdf.columns.astype(str)
    #adjusted_points_gdf = adjusted_points_gdf.drop('0')
    #adjusted_points_gdf = adjusted_points_gdf.dropna(subset=['geometry'])
    adjusted_points_gdf.to_file(r"data\Network\processed\access_highway_matched.gpkg")
    return


def reformat_network():
    # Import points and nodes
    #points_gdf = gpd.read_file(r"data\access_highway_correct.shp")
    current_points = pd.read_csv(r"data\manually_gathered_data\highway_access.csv", sep=";")
    current_points = gpd.GeoDataFrame(current_points, geometry=gpd.points_from_xy(current_points["x"], current_points["y"]),
                                  crs="epsg:2056")

    edges_gdf = gpd.read_file(r"data/temp/network_highway.gpkg")

    for index, row in edges_gdf.iterrows():
        coords = [(coords) for coords in list(row['geometry'].coords)]
        first_coord, last_coord = [coords[i] for i in (0, -1)]
        edges_gdf.at[index, 'first'] = Point(first_coord)
        edges_gdf.at[index, 'last'] = Point(last_coord)

    edges_gdf['x_origin'] = edges_gdf["first"].apply(lambda p: p.x)
    edges_gdf['y_origin'] = edges_gdf["first"].apply(lambda p: p.y)
    edges_gdf['x_dest'] = edges_gdf["last"].apply(lambda p: p.x)
    edges_gdf['y_dest'] = edges_gdf["last"].apply(lambda p: p.y)

    #print(edges_gdf.head(10).to_string())
    print(edges_gdf.shape)

    edges_gdf['E_KOORD_O'], edges_gdf['N_KOORD_O'], edges_gdf['E_KOORD_D'], edges_gdf['N_KOORD_D'] = \
        np.where(edges_gdf['x_origin'] < edges_gdf['x_dest'],
                 (edges_gdf['x_origin'], edges_gdf['y_origin'], edges_gdf['x_dest'], edges_gdf['y_dest']),
                 (edges_gdf['x_dest'], edges_gdf['y_dest'], edges_gdf['x_origin'], edges_gdf['y_origin']))
    edges_gdf = edges_gdf.drop_duplicates(subset=['E_KOORD_O', 'N_KOORD_O', 'E_KOORD_D', 'N_KOORD_D'], keep="first")

    #print(edges_gdf.head(10).to_string())
    print(edges_gdf.shape)

    """
    edges_gdf['E_KOORD_O'], edges_gdf['N_KOORD_O'], edges_gdf['E_KOORD_D'], edges_gdf['N_KOORD_D'] = \
        np.where(edges_gdf['E_KOORD_O'] < edges_gdf['E_KOORD_D'],
                 (edges_gdf['E_KOORD_O'], edges_gdf['N_KOORD_O'], edges_gdf['E_KOORD_D'], edges_gdf['N_KOORD_D']),
                 (edges_gdf['E_KOORD_D'], edges_gdf['N_KOORD_D'], edges_gdf['E_KOORD_O'], edges_gdf['N_KOORD_O']))
    edges_gdf = edges_gdf.drop_duplicates(subset=['E_KOORD_O', 'E_KOORD_D', 'N_KOORD_O', 'N_KOORD_D'], keep="first")
    """

    # Create a list of unique coordinates (x, y) for both source and target
    # Create an empty set to store unique coordinates
    unique_coords = set()

    # Iterate through the DataFrame and add unique source coordinates to the set
    for index, row in edges_gdf.iterrows():
        source_coord = (row['x_origin'], row['y_origin'])
        unique_coords.add(source_coord)

    # Iterate through the DataFrame and add unique target coordinates to the set
    for index, row in edges_gdf.iterrows():
        target_coord = (row['x_dest'], row['y_dest'])
        unique_coords.add(target_coord)
    #unique_coords = pd.concat([edges_gdf[['E_KOORD_O', 'N_KOORD_O']], edges_gdf[['E_KOORD_D', 'N_KOORD_D']]]).values

    # Create a dictionary to store the count of edges for each coordinate
    coord_count = {}

    # Iterate through the unique coordinates and count their appearances in the DataFrame
    for coord in unique_coords:
        count = ((edges_gdf['x_origin'] == coord[0]) & (edges_gdf['y_origin'] == coord[1])).sum() + \
                ((edges_gdf['x_dest'] == coord[0]) & (edges_gdf['y_dest'] == coord[1])).sum()
        coord_count[tuple(coord)] = count

    # Find coordinates where three edges connect
    result_coords = [coord for coord, count in coord_count.items() if count >= 3]

    geometry = [Point(x, y) for x, y in result_coords]
    crossing_nodes = gpd.GeoDataFrame(geometry=geometry, columns=['geometry'], crs="epsg:2056")
    #crossing_nodes = crossing_nodes.to_crs("epsg:2056")
    # print(crossing_nodes.head(20).to_string())

    ### Here I have all points that connect more than 3 links, these are junction in the network


    ### Now I want to group these point to a new one in order to replace the coordiantes
    buffered = crossing_nodes.copy()
    buffered['geometry'] = crossing_nodes.buffer(1000)
    joined = gpd.sjoin(buffered, crossing_nodes, how='left', predicate='intersects')
    mean_coords = joined.groupby('index_right')['geometry'].apply(lambda x: cascaded_union(x).centroid)
    crossing_nodes['new_geometry'] = crossing_nodes.apply(lambda row: mean_coords.get(row.name, row.geometry), axis=1)
    #print(crossing_nodes.head(20).to_string())
    #print("Number of nodes at highway junctions: ", crossing_nodes.shape[0])
    #print(crossing_nodes.shape)

    # dataframe showing the nodes that are highway junctions and store their new coordinates
    crossing_nodes_simple = gpd.GeoDataFrame(crossing_nodes["new_geometry"], geometry="new_geometry")
    # delete if multiple times the same point
    crossing_nodes_simple = crossing_nodes_simple.drop_duplicates(subset='new_geometry')


    #edges_gdf["origin"], edges_gdf["destination"] = edges_gdf["geometry"][0], edges_gdf["geometry"][1]
    #edges_gdf["origin"] = Point(edges_gdf["geometry"].coords[0])


    ## if both geometries then delete (both geometries -> new_geometry)
    df_first = edges_gdf.set_geometry("first")
    overlay_f = gpd.sjoin(df_first, crossing_nodes, how='left', predicate='intersects', lsuffix="first", rsuffix="geometry")
    overlay_f['first'] = np.where(overlay_f['new_geometry'].notna(), overlay_f['new_geometry'], overlay_f['first'])
    overlay_f['one'] = np.where(overlay_f['new_geometry'].notna(), True, False)
    new_edges = overlay_f.drop(columns=["new_geometry", "index_geometry"])

    df_last = new_edges.set_geometry("last")
    overlay_l = gpd.sjoin(df_last, crossing_nodes, how='left', predicate='intersects', lsuffix="last", rsuffix="geometry")
    overlay_l['two'] = np.where(overlay_l['new_geometry'].notna(), True, False)
    overlay_l['last'] = np.where(overlay_l['new_geometry'].notna(), overlay_l['new_geometry'], overlay_l['last'])

    new_edges = overlay_l.drop(columns=["new_geometry", "index_geometry"])
    print(new_edges.shape)
    new_edges = new_edges[~(new_edges['one'] & new_edges['two'])]
    print(new_edges.shape)
    new_edges['geometry'] = new_edges.apply(lambda row: LineString([row["first"], row['last']]), axis=1)
    new_edges = new_edges.set_geometry("geometry")

    #print(new_edges.head(20).to_string())
    #overlay_f = overlay_f[["Link NR", "From Node", "To Node", "geometry", "first", "new_geometry"]]
    #overlay_l = overlay_l[["Link NR", "From Node", "To Node", "geometry", "last", "new_geometry"]]

    #print(edges_gdf.head(20).to_string())
    new_edges = new_edges.set_crs("epsg:2056")
    new_edges = new_edges.drop(columns=["first", "last"])
    new_edges.to_file(r"data\temp\edges_simple.gpkg")

    # delete some edges that cannot deleted automatically
    # task made in Qgis


    # Convert the MultiLineString to a single LineString
    single_line = new_edges['geometry'].unary_union
    merged_line = linemerge(single_line)
    gpd.GeoDataFrame({'geometry': [single_line]}).to_file(r"data\temp\single_lines_n.gpkg")


    # Define the locations where you want to split the MultiLineString

    ##################################################################################################
    # Add connect points (new geometries)
    split_points = list(current_points['geometry'])
    multi_point = MultiPoint(current_points['geometry'])


    # project points on the network, to have exact position
    projected_points = []
    print(len(projected_points))
    for point in multi_point.geoms:
        if not merged_line.contains(point):
            # If the point is not on the LineString, project it onto the LineString
            projected_point = merged_line.interpolate(merged_line.project(point))
            projected_points.append(projected_point)
        else:
            # The point is already on the LineString
            projected_points.append(point)
            print("Point on line")
    print(len(projected_points))

    points_intersection = crossing_nodes_simple['new_geometry'].tolist()
    #projected_points_cross = projected_points + crossing_nodes_simple['new_geometry'].tolist()

    # Calculate the bounding box for points in list1
    multipoint_access = MultiPoint(projected_points)
    bounding_box = multipoint_access.envelope
    # Filter points in list2 to keep only those within the bounding box
    points_intersection = [point for point in points_intersection if bounding_box.contains(point)]
    projected_points_extended = projected_points + points_intersection

    pointzz = MultiPoint(projected_points_extended)
    # for i in multipoint split single_line
    splitted_line = split(snap(merged_line, pointzz, 0.1), pointzz)
    #splitted_line = split(merged_line, MultiPoint(projected_points))

    print(splitted_line)

    """
    def split_line_by_point(line, point, tolerance: float = 1.0e-12):
        return split(line, point)

    for point in split_points:
        single_line_gdf = (single_line_gdf.assign(geometry=single_line_gdf.apply(lambda x: split_line_by_point(
                x.geometry, point), axis=1))
        .explode()
        .reset_index(drop=True))

    print("Results", single_line_gdf)
    """
    # Create an empty list to store individual geometries
    individual_geometries = []
    # Iterate through the geometries in the GeometryCollection
    for geom in splitted_line.geoms:
        individual_geometries.append(geom)

    # Create a GeoSeries from the list of projected points
    geometry = gpd.GeoSeries(projected_points_extended)
    # Create a GeoDataFrame with the GeoSeries as the geometry column
    points_gdf = gpd.GeoDataFrame(geometry, columns=['geometry'])
    points_gdf["intersection"] = 0
    print(crossing_nodes_simple.head(20).to_string())

    crossing_nodes_simple['geometry'] = crossing_nodes_simple['new_geometry'].buffer(0.01)
    # Perform a spatial join
    temp = gpd.sjoin(points_gdf, crossing_nodes_simple, how='left')

    # Update "False" to "True" for matched points
    points_gdf.loc[~temp['index_right'].isna(), 'intersection'] = 1
    points_gdf = points_gdf.rename(columns={"geometry_left":"geometry"})
    # Remove unnecessary columns
    points_gdf = points_gdf[['geometry', 'intersection']]
    print(points_gdf['intersection'].sum())
    points_gdf = points_gdf.set_crs("epsg:2056")
    points_gdf["intersection"] = points_gdf["intersection"].astype(int)
    #points_gdf.to_file(r"data\Network\processed\points.gpkg")

    # Create a GeoDataFrame with one geometry object per row
    gdf = gpd.GeoDataFrame({'geometry': individual_geometries})
    #gdf.to_file(r"data\temp\splited_lines.shp")

    # Create two new columns for start and end points
    gdf['start'] = gdf['geometry'].apply(lambda line: Point(line.coords[0]))
    gdf['end'] = gdf['geometry'].apply(lambda line: Point(line.coords[-1]))

    print(gdf.shape)

    # Create 100-meter buffers around the points in `nodes_A`
    points_gdf['buffered_geometry'] = points_gdf['geometry'].buffer(100)

    # Check if both the start and end points are within the 100-meter buffer of any point in `nodes_A`
    gdf['start_access'] = gdf['start'].apply(
        lambda point: any(point.within(buffer) for buffer in points_gdf['buffered_geometry']))
    gdf['end_access'] = gdf['end'].apply(
        lambda point: any(point.within(buffer) for buffer in points_gdf['buffered_geometry']))

    # Filter the GeoDataFrame to include only the lines that start and end with points from nodes_A
    #filtered_gdf = gdf[(gdf['start_access'] or gdf['end_access'])]
    filtered_edges = gdf[gdf[['start_access', 'end_access']].any(axis=1)]
    filtered_edges = filtered_edges.set_geometry("geometry")
    #print(filtered_gdf.head(20).to_string())

    #filtered_gdf  = filtered_gdf.drop(columns=["start", "end"])
    #filtered_gdf.crs = "epsg:2056"

    #filtered_gdf.plot()
    #plt.show()
    filtered_edges = filtered_edges.set_crs("epsg:2056")
    #filtered_gdf.to_file(r"data\Network\processed\splited_edges_filtered.gpkg")
    #filtered_gdf.to_file(r"data\Network\processed\edges.gpkg")



    ## SOme more operations
    points_gdf["ID_point"] = points_gdf.index

    print(filtered_edges.head(10).to_string())

    # Create a temporary GeoDataFrame with buffered points this is to avoid rounding errors when checking if a point is
    # within the polygon
    points_temp = points_gdf.copy()
    points_temp['buffered_points'] = points_temp['geometry'].buffer(1e-6)
    points_temp = points_temp.set_geometry("buffered_points")

    # Check if all endpoints of the edges are points in points gdf. If not add them to the points dataframe
    # Extract endpoints from edges
    endpoints = [Point(edge.coords[0]) for edge in filtered_edges.geometry] + [Point(edge.coords[-1]) for edge in filtered_edges.geometry]
    # Convert endpoints to a GeoDataFrame
    endpoints_gdf = gpd.GeoDataFrame(geometry=endpoints, crs=filtered_edges.crs)

    # Perform a spatial join to find endpoints not in the buffered points
    joined = gpd.sjoin(endpoints_gdf, points_temp, how='left', predicate='within')
    missing_points = joined[joined['index_right'].isnull()]

    # Drop duplicates and unnecessary columns
    missing_points = missing_points.rename(columns={'geometry_left': 'geometry'}).drop(
        columns=['index_right', 'geometry_right'])
    missing_points = missing_points.drop_duplicates(subset=['geometry'])

    # Append missing points to points_gdf
    # Generate new IDs for missing points
    max_existing_id = points_gdf['ID_point'].max()
    missing_points['ID_point'] = range(max_existing_id + 1, max_existing_id + 1 + len(missing_points))
    missing_points["intersection"] = 0

    # Initialize the 'open_ends' column in points_gdf as False
    points_gdf['open_ends'] = False
    # Set 'open_ends' as True for missing points
    missing_points['open_ends'] = True
    # Append missing points to points_gdf
    #points_completed = points_gdf.append(missing_points, ignore_index=True)
    points_completed = gpd.GeoDataFrame(pd.concat([pd.DataFrame(points_gdf), pd.DataFrame(missing_points)], ignore_index=True))
    points_completed = points_completed.drop(columns=['buffered_geometry']).set_geometry('geometry')
    # Drop points with ID_point = 97 and 98
    points_completed = points_completed[~points_completed['ID_point'].isin([96, 97])]
    points_completed["ID_point"] = points_completed.index

    points_completed.to_file(r"data\Network\processed\points.gpkg")

    # Create a temporary GeoDataFrame with buffered points this is to avoid rounding errors
    points_temp = points_completed.copy()
    points_temp['buffered_points'] = points_temp['geometry'].buffer(1e-6)
    points_temp = points_temp.set_geometry("buffered_points")

    # Replace the point values in edges "start" and "end" by the ID_point of the point gdf based on geometry (ensure there are no errors from roundign)
    # Create temporary GeoDataFrames for 'start' and 'end' points in edges
    start_points_gdf = gpd.GeoDataFrame(geometry=filtered_edges['start'], crs=filtered_edges.crs)
    end_points_gdf = gpd.GeoDataFrame(geometry=filtered_edges['end'], crs=filtered_edges.crs)

    # Perform spatial joins to map the 'ID_point' from points to start_points_gdf and end_points_gdf
    start_joined = sjoin(start_points_gdf, points_temp, how='left', predicate='intersects')
    end_joined = sjoin(end_points_gdf, points_temp, how='left', predicate='intersects')

    # Replace the 'start' and 'end' points in edges with the 'ID_point' from points
    filtered_edges['start'] = start_joined['ID_point']
    filtered_edges['end'] = end_joined['ID_point']

    print(filtered_edges.head(100).to_string())

    filtered_edges.to_file(r"data\Network\processed\edges.gpkg")


def required_manipulations_on_network():
    # This function is introduced to add/remoce some parts of the network enable some computations
    # Import the network
    edges = gpd.read_file(r"data/Network/processed/edges_with_attribute.gpkg")
    points = gpd.read_file(r"data/Network/processed/points_with_attribute.gpkg")
    print(edges.head(10).to_string())
    print(points.head(10).to_string())



    # Add one row ro points with ID_point max_id+1
    max_id = points["ID_point"].max()
    #intersection = 0, ID_point = max_id+1, within_corridor = False, on_corridor_border = False, generate_traffic = False, geometry = Point(2676958, 1243990)
    new_row_data = {
        'intersection': 0,
        'ID_point': max_id + 1,
        'within_corridor': False,
        'on_corridor_border': False,
        'generate_traffic': False,
        'geometry': Point(2676958, 1243990)
    }
    # Append the new row to the DataFrame
    #points = points.append(new_row_data, ignore_index=True)
    points = gpd.GeoDataFrame(pd.concat([pd.DataFrame(points), pd.DataFrame(pd.Series(new_row_data)).T], ignore_index=True))

    # Some changes to the network are needed
    # Point with id 131 should be deleted, while all edges to it should go to id 7
    # Step 1: Get the coordinates of the point with ID_point = 7
    point_coordinates = points.loc[points['ID_point'] == 7, 'geometry'].iloc[0].coords[0]

    # Step 2: Update edges where "start" = 131
    for index, edge in edges[edges['start'] == 131].iterrows():
        # Update the "start" value
        edges.at[index, 'start'] = 7

        # Update the first point of the LineString geometry
        new_line_coords = [point_coordinates] + list(edge['geometry'].coords[1:])
        edges.at[index, 'geometry'] = LineString(new_line_coords)

    # Step 3: Update edges where "end" = 131
    for index, edge in edges[edges['end'] == 131].iterrows():
        # Update the "end" value
        edges.at[index, 'end'] = 7

        # Update the last point of the LineString geometry
        new_line_coords = list(edge['geometry'].coords[:-1]) + [point_coordinates]
        edges.at[index, 'geometry'] = LineString(new_line_coords)

    # Delete the point with ID_point = 131
    points = points[points['ID_point'] != 131]

    # Add a set of new edges connecting following nodes: 89 to 90, 172 to 70, 94 to max_id+1, 146 to max_id+1, 89 to max_id+1
    # Define the pairs of points to connect
    pairs_to_connect = [(152, 168), (89, 90), (172, 70), (94, max_id + 1), (146, max_id + 1), (89, max_id + 1)]

    def get_coords(point_id):
        return points.loc[points['ID_point'] == point_id, 'geometry'].iloc[0].coords[0]

    # Add new links
    for start_id, end_id in pairs_to_connect:
        start_coords = get_coords(start_id)
        end_coords = get_coords(end_id)

        # Create a new LineString
        new_line = LineString([start_coords, end_coords])

        # Create a new row with the desired attributes
        new_row = {
            'start': start_id,
            'end': end_id,
            'start_access': False,
            'end_access': False,
            'polygon_border': False,
            'capacity': 2200,
            'ffs': 100,
            'ID_edge': edges['ID_edge'].max() + 1,
            'geometry': new_line
        }

        # Append the new row to edges_df
        #edges = edges.append(new_row, ignore_index=True)
        edges = gpd.GeoDataFrame(pd.concat([pd.DataFrame(edges), pd.DataFrame(pd.Series(new_row)).T], ignore_index=True))



    # Add one edge which is not a highway but a cantonal road point 39 to 6
    # Create a new LineString
    new_line = LineString([get_coords(39), get_coords(6)])

    # Create a new row with the desired attributes
    new_row = {
        'start': 39,
        'end': 6,
        'start_access': False,
        'end_access': False,
        'polygon_border': False,
        'capacity': 1000,
        'ffs': 80,
        'ID_edge': edges['ID_edge'].max() + 1,
        'geometry': new_line
    }

    # Append the new row to edges_df
    #edges = edges.append(new_row, ignore_index=True)
    edges = gpd.GeoDataFrame(pd.concat([pd.DataFrame(edges), pd.DataFrame(pd.Series(new_row)).T], ignore_index=True))

    # Store the updated edges DataFrame
    edges.to_file(r"data/Network/processed/edges_with_attribute.gpkg")
    points.to_file(r"data/Network/processed/points_with_attribute.gpkg")



def get_edge_attributes():
    edges_raw = gpd.read_file(r"data/temp/network_highway.gpkg")
    edges_process = gpd.read_file(r"data/Network/processed/edges_with_attribute.gpkg")


    # Step 1: Create a buffer around the edges in df2
    buffer_distance = 50
    edges_process['buffered'] = edges_process.geometry.buffer(buffer_distance)

    # Step 2 & 3: Find overlapping edges from df1 for each edge in df2 and print lengths
    def get_min_attributes(edge, df1):
        #overlaps = df1[df1.geometry.within(edge.buffered)]
        overlaps = df1[df1.geometry.intersects(edge.buffered)]
        if not overlaps.empty:
            # Get the mean value of the attribute capacity and speed weighted by each link length
            mean_capacity = np.average(overlaps['Capacity'], weights=overlaps.geometry.length)
            mean_speed = np.average(overlaps['Free Flow Speed'], weights=overlaps.geometry.length)

            # divide mean capacity by 24 to get hourly capacity
            mean_capacity = mean_capacity / 24

            return mean_capacity, mean_speed
        return None, None, None, None

    # Step 4 & 5: Apply the function to each edge in df2
    edges_process['capacity'], edges_process['ffs'] = zip(*edges_process.apply(lambda x: get_min_attributes(x, edges_raw), axis=1))

    # Drop the buffered column if not needed
    edges_process.drop(columns=['buffered'], inplace=True)

    # Store values as integer
    # Drop rows with None values
    edges_process.dropna(inplace=True)
    edges_process['capacity'] = edges_process['capacity'].astype(int)
    edges_process['ffs'] = edges_process['ffs'].astype(int)

    # Check if there are None values in the capacity and ffs columns
    print(edges_process[edges_process['capacity'].isnull()])
    print(edges_process[edges_process['ffs'].isnull()])

    edges_process["ID_edge"] = edges_process.index

    print(edges_process.head(10).to_string())

    edges_process.to_file(r"data\Network\processed\edges_with_attribute.gpkg")


def network_in_corridor(polygon):
    # polygon to Geopandas dataframe
    polygon = gpd.GeoDataFrame({'geometry': [polygon]})
    polygon.crs = "epsg:2056"

    edges = gpd.read_file(r"data\Network\processed\edges.gpkg")
    points = gpd.read_file(r"data\Network\processed\points.gpkg")

    print(edges.head(10).to_string())
    print(points.head(10).to_string())
    print(points.shape)

    # connect the two edges with id 81 and 70 by their closest end points to make one new out of it


    # filter points in polygon
    points_corridor = gpd.sjoin(points, polygon, how="inner")
    points_corridor = points_corridor.drop(columns=["index_right"]) # , "FID"
    points_corridor.to_file(r"data\Network\processed\points_corridor.gpkg")

    # Check if point are in polygon if so add True as "within_corridor" attribute otherwise False
    points['within_corridor'] = points.apply(lambda row: polygon.contains(row.geometry), axis=1)

    # get edges within the polygon
    edges_corridor = gpd.sjoin(edges, polygon, how="inner")
    edges_corridor = edges_corridor.drop(columns=["start_access", "end_access", "index_right"])
    edges_corridor.to_file(r"data\Network\processed\edges_in_corridor.gpkg")

    # Get edges crossed by polygon frame
    # Only keep edges with exactly on endpoint in the polygon and the other outside the polygon (= point_corridor)
    # Define the function to check endpoints
    def is_one_endpoint_inside(edge_geom, poly):
        start_point = Point(edge_geom.coords[0])
        end_point = Point(edge_geom.coords[-1])
        return poly.contains(start_point) != poly.contains(end_point)  # True if only one point is inside

    # Apply the function directly in the apply method
    edges['polygon_border'] = edges['geometry'].apply(lambda x: is_one_endpoint_inside(x, polygon))

    # Apply the function to filter edges and save the filtered edges in seperate file
    #edges_crossing_polygon = edges[edges.apply(lambda x: is_one_endpoint_inside(x, polygon), axis=1)]
    edges_crossing_polygon = edges[edges["polygon_border"] == True]
    edges_crossing_polygon.to_file(r"data\Network\processed\edges_on_corridor_border.gpkg")

    # get nodes of edges on corridor

    #points_crossing_polygon = gpd.sjoin(edges_crossing_polygon, points_temp, how="inner", predicate="intersects", lsuffix="left", rsuffix="")
    #points_crossing_polygon = points_crossing_polygon.drop(columns=["geometry_left", "index_"])
    #print(points_crossing_polygon.head(10).to_string())
    #points_crossing_polygon = points_crossing_polygon.rename(columns={'geometry_':'geometry'}, inplace=True)

    #points_crossing_polygon = points_crossing_polygon.set_geometry("geometry_")
    #print(points_crossing_polygon.head(10).to_string())
    #points_crossing_polygon.to_file(r"data\Network\processed\points_corridor_border.gpkg")

    points_temp = points.copy()
    points_temp['buffered_points'] = points_temp['geometry'].buffer(1e-6)
    points_temp = points_temp.set_geometry("buffered_points")

    # Perform the spatial join and create a temporary DataFrame
    temp_joined = gpd.sjoin(points_temp, edges_crossing_polygon, how="left", predicate="intersects", lsuffix="left",
                            rsuffix="right")
    # Add 'on_corridor_border' attribute to points
    # A point is on the corridor border if it has a match in the spatial join (non-null index_right)
    points['on_corridor_border'] = temp_joined['index_right'].notnull()

    # Link the point ID (from points) to each endpoint of the edges
    # Create a temporary DataFrame with the endpoints of the edges


    print(points.head(10).to_string())
    print(edges.head(10).to_string())


    points.to_file(r"data\Network\processed\points_with_attribute.gpkg")
    edges.to_file(r"data\Network\processed\edges_with_attribute.gpkg")


def map_values_to_nodes():
    nodes_processed = gpd.read_file(r"data\Network\processed\points_corridor.gpkg")
    nodes_processed.set_crs("epsg:2056")

    points_raw = pd.read_csv(r"data\manually_gathered_data\highway_access.csv", sep=";")
    points_raw = gpd.GeoDataFrame(points_raw,
                                      geometry=gpd.points_from_xy(points_raw["x"], points_raw["y"]), crs="epsg:2056")

    neww = nodes_processed.sjoin_nearest(points_raw.drop(columns=["x", "y"]), how="left")
    print(neww.head(10).to_string())
    columns_to_replace = neww.columns.difference(['geometry', "index_right", "ID_point"]) # "intersection",
    neww.loc[neww['intersection'] == 1, columns_to_replace] = np.nan
    #neww.set_crs("epsg:2056")

    neww.to_file(r"data\Network\processed\points_corridor_attribute.gpkg")

    # Same for all points
    nodes_processed = gpd.read_file(r"data\Network\processed\points.gpkg")

    points_raw = pd.read_csv(r"data\manually_gathered_data\highway_access.csv", sep=";")
    points_raw = gpd.GeoDataFrame(points_raw,
                                      geometry=gpd.points_from_xy(points_raw["x"], points_raw["y"]), crs="epsg:2056")

    neww = nodes_processed.sjoin_nearest(points_raw.drop(columns=["x", "y"]), how="left")
    columns_to_replace = neww.columns.difference(['geometry', "intersecti", "index_right", "ID_point"])
    neww.loc[neww['intersection'] == 1, columns_to_replace] = np.nan
    neww = neww.rename(columns={"index_right": "ID_point"})
    neww.crs = "epsg:2056"

    # Remove duplicate columns, keeping the first occurrence
    neww = neww.loc[:, ~neww.columns.duplicated()]

    neww.to_file(r"data\Network\processed\points_attribute.gpkg")


def only_links_to_corridor():
    all_links = gpd.read_file(r"data\Network\processed\new_links.gpkg")
    all_access_points = gpd.read_file(r"data\Network\processed\points_corridor_attribute.gpkg")
    access_corridor = all_access_points[all_access_points["cor_1"] == "1"]
    print(access_corridor.head(50).to_string())

    # only keep the links that connect to a access point on the highway corridor that is considered
    links_corridor = all_links.merge(right=access_corridor[["cor_1", "ID_point"]], left_on="ID_current", right_index=True) #right_on="ID_point")
    print("Links connected to access points within the corridor: ", links_corridor.shape[0], " of ", all_links.shape[0])
    links_corridor = links_corridor.drop(columns=["ID_current"])
    links_corridor.to_file(r"data\Network\processed\developments_to_corridor_attribute.gpkg")

    # same for access points
    generated_points = gpd.read_file(r"data\Network\processed\generated_nodes.gpkg")

    temp = generated_points.merge(right=links_corridor, left_on="ID_new", right_on="ID_new")
    generated_points_corridor = temp[["ID_new", "geometry_x", "ID_point"]]
    #all_objects_corridor = temp[["ID_new", "geometry_x", "ID_point", "geometry_y"]]
    #all_objects_corridor = all_objects_corridor.rename({"geometry_x":"point_geom", "geometry_y":"edge_geom", "ID_point":"ID_current"})

    gpd.GeoDataFrame(generated_points_corridor, geometry="geometry_x").to_file(r"data\Network\processed\generated_nodes_connecting_corridor.gpkg")
    #gpd.GeoDataFrame(all_objects_corridor,geometry="geometry_x").to_file(r"data\Network\processed\generated_objects_connecting_corridor.shp")


def get_protected_area(limits):
    bln = gpd.read_file(r"data\landuse_landcover\Schutzzonen\BLN\N2017_Revision_landschaftnaturdenkmal_20170727_20221110.shp")
    wildkorridore = gpd.read_file(r"data\landuse_landcover\Schutzzonen\Wildtierkorridore\Wildtierkorridore.gpkg")
    trockenweiden = gpd.read_file(r"data\landuse_landcover\Schutzzonen\Trockenwiesen\TWW_LV95\trockenwiesenweiden.shp")
    trockenlandschaften = gpd.read_file(r"data\landuse_landcover\Schutzzonen\Moorlandschaft\Moorlandschaft_LV95\moorlandschaft.shp")
    flachmoore = gpd.read_file(r"data\landuse_landcover\Schutzzonen\Flachmoore\Flachmoor_LV95\flachmoor_20210701.shp")
    hochmoore = gpd.read_file(r"data\landuse_landcover\Schutzzonen\Hochmoor\Hochmoor_LV95\hochmoor.shp")
    bundesinventar_auen = gpd.read_file(r"data\landuse_landcover\Schutzzonen\Bundesinventar_auen\N2017_Revision_Auengebiete_20171101_20221122.shp")
    ramsar = gpd.read_file(r"data\landuse_landcover\Schutzzonen\Ramsar\Ramsar_LV95\ra.shp")
    #naturschutz = gpd.read_file(r"data\landuse_landcover\Schutzzonen\Inventar_der_Natur-_und_Landsch...uberkommunaler_Bedeutung_-OGD\INV80_NATURSCHUTZOBJEKTE_F.shp")
    wald = gpd.read_file(r"data\landuse_landcover\Schutzzonen\Waldareal_-OGD\WALD_WALDAREAL_F.shp")
    fruchtfolgeflaeche = gpd.read_file(r"data\landuse_landcover\Schutzzonen\Fruchtfolgeflachen_-OGD\FFF_F.shp")

    gdf_fully_protected = [
        bln,
        wildkorridore,
        flachmoore,
        hochmoore,
        bundesinventar_auen,
        ramsar,
        #naturschutz
    ]
    names_fully_protected = [
        'bln',
        'wildkorridore',
        'flachmoore',
        'hochmoore',
        'bundesinventar_auen',
        'ramsar',
        #'naturschutz'
    ]

    gdf_partly_protected = [
        wald,
        fruchtfolgeflaeche,
        trockenweiden,
        trockenlandschaften
    ]
    names_partly_protected = [
        "wald",
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
        dissolved = gdf.dissolve()
        # Append the dissolved geometry to the list
        dissolved_geometries.append(dissolved.geometry.unary_union)
        #dissolved_geometries = gpd.GeoDataFrame(pd.concat([pd.DataFrame(dissolved_geometries), pd.DataFrame(dissolved.geometry.unary_union)], ignore_index=True))

    # Now create a new DataFrame with the dissolved geometries
    # Use the 'unary_union' attribute to ensure that the geometry is merged into one
    combined_gdf = gpd.GeoDataFrame({'geometry': dissolved_geometries})

    combined_gdf["name"] = names_list
    combined_gdf.crs = "epsg:2056"

    combined_gdf.to_file(fr"data\landuse_landcover\Schutzzonen\{path}.gpkg", driver="GPKG")

    # Create a bounding box as a shapely object
    frame_box = box(limits[0], limits[1], limits[2], limits[3])

    # Clip the GeoDataFrame using the bounding box
    combined_gdf_frame = gpd.clip(combined_gdf, frame_box)
    #combined_gdf_frame.to_file(fr"data\landuse_landcover\Schutzzonen\{path}_frame.gpkg")
    combined_gdf_frame.to_file(fr"data\landuse_landcover\processed\{path}_frame.gpkg")


def all_protected_area_to_raster(suffix=""):
    # Load your shapefile with geopandas
    #shp_file = r"data\landuse_landcover\processed\fully_protected.gpkg"
    shp_file = r"data\landuse_landcover\Schutzzonen\fully_protected.gpkg" #correction by Arnor
    shapes = gpd.read_file(shp_file)

    # Load your raster file with rasterio
    tif_file = r"data\landuse_landcover\processed\protected_area.tif"

    try:
        # Load the CSV file with the coordinates
        csv_file = r"data\manually_gathered_data\cell_to_remove.csv"
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
        with rasterio.open(fr'data\landuse_landcover\processed\zone_no_infra\protected_area_{suffix}.tif', 'w', **meta) as dst:
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
    print(areal_stat.shape)
    protected_categories = [1, 2, 3, 4, 5, 7, 8,
                            9, 10,
                            17, 18,
                            23, 27]
    protected_area = areal_stat[areal_stat["AS18_27"].isin(protected_categories)]
    print(protected_area.shape)
    protected_area_full = fill_raster_dataframe(protected_area)
    print(protected_area_full.head(10).to_string())
    # Correction of the reference of each raster cell from bottom left to top left
    protected_area_full["N_COORD"] = protected_area_full["N_COORD"] + 100
    csv_to_tiff(protected_area_full, attribute="AS18_27", path=r"data\landuse_landcover\processed\protected_area.tif")
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
    # Raster data of unproductive area
    unproductive_zones = [23, 24, 25, 26, 27]
    unproductive_area = areal_stat[areal_stat["AS18_27"].isin(unproductive_zones)]
    #print(unproductive_area.shape)
    unproductive_area_full = fill_raster_dataframe(unproductive_area)
    #print(unproductive_area_full.head(10).to_string())
    # Correction of the reference of each raster cell from bottom left to top left
    unproductive_area_full["N_COORD"] = unproductive_area_full["N_COORD"] + 100
    csv_to_tiff(unproductive_area_full, attribute="AS18_27", path=r"data\landuse_landcover\processed\unproductive_area.tif")

    # AS85_17 17 Klassen gemäss Standardnomenklatur der Arealstatistik 1979/85
    # AS85_4  4 Hauptbereiche gemäss Standardnomenklatur der Arealstatistik 1979/85
    # LU85_10 10 Klassen der Bodennutzung der Arealstatistik 1979/85
    # LU85_4  4 Hauptbereiche der Bodennutzung der Arealstatistik 1979/85


def tif_to_shp(path_tif, path_shp):
    # Read the raster data
    # Open the raster file and read the first band
    with rasterio.open(path_tif) as src:
        image = src.read(1)  # Assuming you want the first band
        image = image.astype('float32')  # Convert to float32
        affine = src.transform
        crs = src.crs

    # Define your threshold value here
    #threshold = 0.5  # Example threshold value

    # Create a mask based on the threshold
    #mask = image > threshold

    # Extract shapes from the binary mask using rasterio's shapes function
    shape_gen = shapes(image,  transform=affine) # mask=mask,

    # Check and process valid geometries
    geometries = []
    for geom, value in shape_gen:
        if value == 1:  # Assuming '1' corresponds to the shapes you want
            tempgeom = {'geometry': shape(geom), 'properties': {'raster_val': value}}
            #geometries.append(tempgeom)
            geometries = gpd.GeoDataFrame(pd.concat([pd.DataFrame(geometries), pd.DataFrame(pd.Series(tempgeom)).T], ignore_index=True))
    # Proceed only if there are valid geometries
    if geometries:
        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(geometries)
        gdf.crs = crs  # Set the CRS
        # Save to a Shapefile
        gdf.to_file(path_shp)
    else:
        print("No valid geometries were found in the raster with the given threshold.")

