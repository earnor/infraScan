import os
os.environ['USE_PYGEOS'] = '0'
import settings
from shapely.geometry import Polygon
import paths
from plots import *
import ast
from tqdm import tqdm


def import_cities():
    """
    This functions converts a csv files of location with coordinate to a geopandas DataFrame
    :return: GeoPandas DataFrame containing the locations as points
    """
    # Read csv file into pandas DataFrame
    df_cities = pd.read_csv(r"data\manually_gathered_data\City_map.csv", sep=";")

    # Convert single values into coordinates of geopandas DataFrame and initialize the coordinate reference system
    gdf_cities = gpd.GeoDataFrame(df_cities, geometry=gpd.points_from_xy(df_cities["x"], df_cities["y"]),
                                  crs="epsg:2056")

    gdf_cities.to_file('data\manually_gathered_data\cities.shp')
    return


def get_lake_data():
    gdf = gpd.read_file(r"data\landuse_landcover\landcover\lake\WB_STEHGEWAESSER_F.shp")
    gdf = gdf[gdf["GEWAESSERN"].isin(["Zürichsee", "Greifensee", "Pfäffikersee"])].to_crs("epsg:2056")
    gdf.to_file('data\landuse_landcover\processed\lake_data_zh.gpkg')
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


def reformat_rail_edges(rail_network):
    """
    This function reformats the railway network to add the access points to the network. The railway network is
    imported from a geopackage and the access points are loaded from a csv file. The access points are projected to
    the closest railway segment. The railway segments are then split at the access points.
    """
    rail_service_path = paths.get_rail_services_path(rail_network)
    edges_gdf = gpd.read_file(rail_service_path)

    for index, row in edges_gdf.iterrows():
        coords = [(coords) for coords in list(row['geometry'].coords)]
        first_coord, last_coord = [coords[i] for i in (0, -1)]
        edges_gdf.at[index, 'first'] = Point(first_coord)
        edges_gdf.at[index, 'last'] = Point(last_coord)

    edges_gdf['x_origin'] = edges_gdf["first"].apply(lambda p: p.x)
    edges_gdf['y_origin'] = edges_gdf["first"].apply(lambda p: p.y)
    edges_gdf['x_dest'] = edges_gdf["last"].apply(lambda p: p.x)
    edges_gdf['y_dest'] = edges_gdf["last"].apply(lambda p: p.y)

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

    # Create a dictionary to store the count of edges for each coordinate
    coord_count = {}

    # Iterate through the unique coordinates and count their appearances in the DataFrame
    for coord in unique_coords:
        count = ((edges_gdf['x_origin'] == coord[0]) & (edges_gdf['y_origin'] == coord[1])).sum() + \
                ((edges_gdf['x_dest'] == coord[0]) & (edges_gdf['y_dest'] == coord[1])).sum()
        coord_count[tuple(coord)] = count


    ##################################################################################################
    # Add connect points (new geometries)



    edges_gdf = edges_gdf.drop(['first', 'last'],axis=1)
    edges_gdf.to_file(r"data\Network\processed\edges.gpkg")


def reformat_rail_nodes():
    # Simplify the physical topology of the network
    # One distinct edge between two nodes (currently multiple edges between nodes)
    # Edges are stored in "data\Network\processed\edges.gpkg"
    # Points in simplified network can be intersections ("intersection"==1) or access points ("intersection"==0)
    # Points are stored in "data\Network\processed\points.gpkg"

    current_points = pd.read_csv(paths.RAIL_POINTS_PATH, sep=";", decimal=",", encoding="ISO-8859-1")
    current_points = gpd.GeoDataFrame(current_points,
                                      geometry=gpd.points_from_xy(current_points["XKOORD"], current_points["YKOORD"]),
                                      crs="epsg:21781")
    current_points = current_points.to_crs("epsg:2056")
    current_points = current_points.rename(columns={"NR": "ID_point"})
    current_points.to_file(r"data\Network\processed\points.gpkg")


def add_new_line(stations, frequency, service_name, travel_times, edges, points, via=[]):
    """
    Add a new line to the network with bidirectional edges and travel times.

    Parameters:
        stations (list): List of station names representing the new line.
        frequency (int): Frequency of the new line.
        service_name (str): Name of the service for the new line.
        travel_times (list): List of travel times between consecutive stations.
        edges (str): Path to the GeoPackage file containing the edges.
        points (str): GeoPackage file containing the station points.

    Returns:
        None
    """
    if not via:
        via = ['-99'] * (len(stations) - 1)


    # Iterate through consecutive station pairs
    for i in range(len(stations) - 1):

        # Get the IDs and geometries of the two stations
        from_station = stations[i]
        to_station = stations[i + 1]


        from_id = points.loc[points['NAME'] == from_station, 'ID_point'].iloc[0]
        to_id = points.loc[points['NAME'] == to_station, 'ID_point'].iloc[0]
        from_geom = points.loc[points['ID_point'] == from_id, 'geometry'].iloc[0]
        to_geom = points.loc[points['ID_point'] == to_id, 'geometry'].iloc[0]

        # Get the coordinate values from the points dataframe
        from_xkoord = points.loc[points['ID_point'] == from_id, 'XKOORD'].iloc[0]
        from_ykoord = points.loc[points['ID_point'] == from_id, 'YKOORD'].iloc[0]
        to_xkoord = points.loc[points['ID_point'] == to_id, 'XKOORD'].iloc[0]
        to_ykoord = points.loc[points['ID_point'] == to_id, 'YKOORD'].iloc[0]

        # Create LineString geometries
        line_geom_a = LineString([from_geom, to_geom])
        line_geom_b = LineString([to_geom, from_geom])

        # Determine FromEnd and ToEnd for direction "A"
        from_end_a = 1 if i == 0 else 0
        to_end_a = 1 if i == len(stations) - 2 else 0


        via_numeric = ast.literal_eval(via[i])
        # Result: [2487, 644, 1977, 1503]

        if isinstance(via_numeric, list):
            via_b = list(reversed(via_numeric))
        else:
            via_b = via[i]

        via_b = str(via_b)

        # Create a new row for the edge in direction "A"
        new_edge_a = {
            'FromStation': from_station,
            'ToStation': to_station,
            'FromNode': from_id,
            'ToNode': to_id,
            'Service': service_name,
            'Frequency': frequency,
            'Direction': 'A',
            'Via': via[i],
            'FromEnd': from_end_a,
            'ToEnd': to_end_a,
            'TravelTime': travel_times[i],
            'InVehWait': 0,
            'E_KOORD_O': from_xkoord,  # Origin coordinates
            'N_KOORD_O': from_ykoord,
            'E_KOORD_D': to_xkoord,     # Destination coordinates
            'N_KOORD_D': to_ykoord,
            'geometry': line_geom_a
        }

        # Determine FromEnd and ToEnd for direction "B"
        from_end_b = 1 if i == len(stations) - 2 else 0
        to_end_b = 1 if i == 0 else 0

        # Create a new row for the edge in direction "B"
        new_edge_b = {
            'FromStation': to_station, # to and from must be swapped
            'ToStation': from_station,
            'FromNode': to_id,
            'ToNode': from_id,
            'Service': service_name,
            'Frequency': frequency,
            'Direction': 'B',
            'Via': via_b,
            'FromEnd': from_end_b,
            'ToEnd': to_end_b,
            'TravelTime': travel_times[i],
            'InVehWait': 0,
            'E_KOORD_O': to_xkoord,    # Swapped - Origin coordinates for direction B
            'N_KOORD_O': to_ykoord,
            'E_KOORD_D': from_xkoord,  # Swapped - Destination coordinates for direction B
            'N_KOORD_D': from_ykoord,
            'geometry': line_geom_b
        }

        # Append the new edges to the edges GeoDataFrame
        new_line_gpd = gpd.GeoDataFrame([new_edge_a, new_edge_b])
        edges = gpd.GeoDataFrame(pd.concat([edges, new_line_gpd], ignore_index=True))
    # Save the updated edges GeoDataFrame
    return edges

def create_railway_services_2024_extended():
    edges_ak2024_ext = gpd.read_file(paths.RAIL_SERVICES_2024_PATH)
    points = gpd.read_file(r'data\Network\processed\points.gpkg')

    edges_ak2024_ext = add_new_line(
        stations=[
            'Zürich Stadelhofen',
            'Zürich, Kreuzplatz',
            'Zürich, Hegibachplatz',
            'Zürich, Balgrist',
            'Zürich, Rehalp',
            'Waldburg',
            'Spital Zollikerberg',
            'Zollikerberg',
            'Waltikon',
            'Zumikon',
            'Maiacher',
            'Neue Forch',
            'Forch',
            'Scheuren',
            'Neuhaus bei Hinteregg',
            'Hinteregg',
            'Egg',
            'Langwies ZH',
            'Emmat',
            'Esslingen'
        ],
        frequency=4,
        service_name='S18',
        travel_times=[
            1,  # Zürich Stadelhofen -> Zürich, Kreuzstrasse
            2,  # Zürich, Kreuzstrasse -> Zürich, Hegibachplatz
            3,  # Zürich, Hegibachplatz -> Zürich, Balgrist
            3,  # Zürich, Balgrist -> Zürich, Rehalp
            1,  # Zürich, Rehalp -> Waldburg
            1,  # Waldburg -> Spital Zollikerberg
            1,  # Spital Zollikerberg -> Zollikerberg
            1,  # Zollikerberg -> Waltikon
            2,  # Waltikon -> Zumikon
            2,  # Zumikon -> Maiacher
            1,  # Maiacher -> Neue Forch
            1,  # Neue Forch -> Forch
            3,  # Forch -> Scheuren
            2,  # Scheuren -> Neuhaus bei Hinteregg
            2,  # Neuhaus bei Hinteregg -> Hinteregg
            3,  # Hinteregg -> Egg
            2,  # Egg -> Langwies ZH
            1,  # Langwies ZH -> Emmat
            1,  # Emmat -> Esslingen
            4  # Emmat -> Esslingen (korrigiert: 06:21 → 06:25)
        ],
        edges=edges_ak2024_ext,
        points=points)

    edges_ak2024_ext = add_new_line(
        stations=[
            'Winterthur',
            'Winterthur Grüze',
            'Winterthur Seen',
            'Sennhof-Kyburg',
            'Kollbrunn',
            'Rikon',
            'Rämismühle-Zell',
            'Turbenthal',
            'Wila',
            'Saland',
            'Bauma',
            'Steg',
            'Fischenthal',
            'Gibswil',
            'Wald ZH',
            'Tann-Dürnten',
            'Rüti ZH'
        ],
        frequency=2,
        service_name='S26',
        travel_times=[
            5,  # Winterthur -> Winterthur Grüze (05:13 → 05:18)
            3,  # Winterthur Grüze -> Winterthur Seen (05:18 → 05:21)
            3,  # Winterthur Seen -> Sennhof-Kyburg (05:21 → 05:24)
            3,  # Sennhof-Kyburg -> Kollbrunn (05:24 → 05:27)
            3,  # Kollbrunn -> Rikon (05:27 → 05:30)
            2,  # Rikon -> Rämismühle-Zell (05:30 → 05:32)
            4,  # Rämismühle-Zell -> Turbenthal (05:32 → 05:36)
            2,  # Turbenthal -> Wila (05:36 → 05:38)
            6,  # Wila -> Saland (05:38 → 05:44)
            6,  # Saland -> Bauma (05:44 → 05:50)
            4,  # Bauma -> Steg (05:50 → 05:54)
            5,  # Steg -> Fischenthal (05:54 → 05:59)
            2,  # Fischenthal -> Gibswil (05:59 → 06:01)
            7,  # Gibswil -> Wald ZH (06:01 → 06:08)
            5,  # Wald ZH -> Tann-Dürnten (06:08 → 06:13)
            3  # Tann-Dürnten -> Rüti ZH (06:13 → 06:16)
        ],
        edges=edges_ak2024_ext,
        points=points)

    edges_ak2024_ext = add_new_line(
        stations=[
    'Winterthur',
    'Kemptthal',
    'Effretikon',
    'Bassersdorf',
    'Kloten',
    'Kloten Balsberg',
    'Opfikon',
    'Zürich Oerlikon',
    'Zürich Hardbrücke',
    'Zürich HB',
    'Zürich Stadelhofen',
    'Meilen',
    'Uetikon',
    'Männedorf',
    'Stäfa',
    'Uerikon',
    'Feldbach',
    'Kempraten',
    'Rapperswil SG'
    ],
        frequency=2,
        service_name='S7',
        travel_times=[
    4,   # Winterthur -> kemptthal (05:05 → 05:09)
    4,   # Kemptthal -> Effretikon (05:09 → 05:13)
    6,   # Effretikon -> Bassersdorf (05:14 → 05:20)
    5,   # Bassersdorf -> Kloten (05:20 → 05:25)
    2,   # Kloten -> Kloten Balsberg (05:25 → 05:27)
    2,   # Kloten Balsberg -> Opfikon (05:27 → 05:29)
    4,   # Opfikon -> Zürich Oerlikon (05:29 → 05:33)
    4,   # Zürich Oerlikon -> Zürich Hardbrücke (05:33 → 05:37)
    3,   # Zürich Hardbrücke -> Zürich HB (05:37 → 05:40)
    4,   # Zürich HB -> Zürich Stadelhofen (05:40 → 05:44)
    14,  # Zürich Stadelhofen -> Meilen (05:44 → 05:58)
    3,   # Meilen -> Uetikon (05:58 → 06:01)
    2,   # Uetikon -> Männedorf (06:01 → 06:03)
    5,   # Männedorf -> Stäfa (06:03 → 06:08)
    3,   # Stäfa -> Uerikon (06:08 → 06:11)
    5,   # Uerikon -> Feldbach (06:11 → 06:16)
    2,   # Feldbach -> Kempraten (06:16 → 06:18)
    4    # Kempraten -> Rapperswil SG (06:18 → 06:22)
    ],
        edges=edges_ak2024_ext,
        points=points,
        via=[
    '-99','-99', '-99', '-99', '-99', '-99', '-99', '-99', '-99', '-99',
    '[2551, 2534, 1122, 1174, 702, 2459, 999]', '-99', '-99', '-99', '-99', '-99', '-99', '-99'
            ])

    edges_ak2024_ext = add_new_line(
        stations=[
            'Zürich HB',
            'Zürich Stadelhofen',
            'Zürich Tiefenbrunnen',
            'Zollikon',
            'Küsnacht Goldbach',
            'Küsnacht ZH',
            'Erlenbach ZH',
            'Winkel am Zürichsee',
            'Herrliberg-Feldmeilen',
            'Meilen'
        ],
        frequency=4,
        service_name='pseudo S6',
        travel_times=[
            3,  # Zürich HB → Zürich Stadelhofen (06:00 → 06:03)
            3,  # Zürich Stadelhofen → Zürich Tiefenbrunnen (06:03 → 06:06)
            2,  # Zürich Tiefenbrunnen → Zollikon (06:06 → 06:08)
            2,  # Zollikon → Küsnacht Goldbach (06:08 → 06:10)
            3,  # Küsnacht Goldbach → Küsnacht ZH (06:10 → 06:13)
            2,  # Küsnacht ZH → Erlenbach ZH (06:13 → 06:15)
            2,  # Erlenbach ZH → Winkel am Zürichsee (06:15 → 06:17)
            3,  # Winkel am Zürichsee → Herrliberg-Feldmeilen (06:17 → 06:20)
            3  # Herrliberg-Feldmeilen → Meilen (06:20 → 06:23)
        ],
        edges=edges_ak2024_ext,
        points=points)
    edges_ak2024_ext = add_new_line(
        stations=['Zürich HB', 'Zürich Oerlikon', 'Wallisellen', 'Dietlikon', 'Effretikon', 'Winterthur'],
        frequency=2,
        service_name='S8',
        travel_times=[5, 4, 2, 6, 7],
        edges=edges_ak2024_ext,
        points=points,
        via=['-99', '-99', '-99', '-99', '[1119]'])

    edges_ak2024_ext = add_new_line(
        stations=['Zürich HB', 'Zürich Stadelhofen', 'Stettbach', 'Winterthur'],
        frequency=4,
        service_name='S11+S12',
        travel_times=[4, 6, 12],
        edges=edges_ak2024_ext,
        points=points,
        via=['-99', '-99', '[638, 666, 1119]'])

    edges_ak2024_ext = add_new_line(
        stations=['Zürich HB', 'Pfäffikon SZ'],
        frequency=2,
        service_name='fast_ZH_PF',
        travel_times=[26],
        edges=edges_ak2024_ext,
        points=points,
        via=['-99'])

    edges_ak2024_ext.to_file(paths.RAIL_SERVICES_AK2024_EXTENDED_PATH)
    return edges_ak2024_ext, points

def create_railway_services_AK2035():

    edges_ak2035 = gpd.read_file(paths.RAIL_SERVICES_2024_PATH)
    points = gpd.read_file(r'data\Network\processed\points.gpkg')

    # Double the frequency and capacity for rows where "Service" contains "S9"
    edges_ak2035.loc[edges_ak2035["Service"] == "S9", "Frequency"] *= 2

    #S14 no longer runs to Hinwil
    hinwil_id = get_station_id('Hinwil', points)
    # Filter out edges with Service "S14" starting or ending in Hinwil
    edges_ak2035 = edges_ak2035[~((edges_ak2035['Service'] == 'S14') &
                                  ((edges_ak2035['FromNode'] == hinwil_id) | (edges_ak2035['ToNode'] == hinwil_id)))]

    wetzikon_id = get_station_id('Wetzikon ZH', points)

    # Update ToEnd for the edge where Service is "S14" and ToNode is Wetzikon
    edges_ak2035.loc[(edges_ak2035['Service'] == 'S14') & (edges_ak2035['ToNode'] == wetzikon_id), 'ToEnd'] = True

    # Update FromEnd for the edge where Service is "S14" and FromNode is Wetzikon
    edges_ak2035.loc[(edges_ak2035['Service'] == 'S14') & (edges_ak2035['FromNode'] == wetzikon_id), 'FromEnd'] = True
    # Add new lines to the network which are introduced with AK2035
    edges_ak2035=add_new_line(
        stations=['Zürich HB', 'Zürich Oerlikon', 'Uster', 'Wetzikon ZH', 'Hinwil'],
        frequency=2,
        service_name='G',
        travel_times=[5, 10, 6, 4],  # TT Oerlikon-Uster 1 min faster
        edges=edges_ak2035,
        points=points,
        via = ['-99', '[2487,644,1977,1503]', '[6]', '-99'])

    edges_ak2035=add_new_line(
        stations=['Zürich HB', 'Zürich Stadelhofen', 'Stettbach', 'Dietlikon', 'Effretikon', 'Illnau', 'Fehraltorf', 'Pfäffikon ZH'],
        frequency=2,
        service_name='P',
        travel_times=[3, 5, 3, 6, 4, 5, 4],
        edges=edges_ak2035,
        points=points)
    edges_ak2035=add_new_line(
        stations=['Zürich HB', 'Zürich Oerlikon', 'Wallisellen', 'Dietlikon', 'Effretikon', 'Winterthur'],
        frequency=4,
        service_name='C+D',
        travel_times=[5, 4, 2, 6, 7],
        edges=edges_ak2035,
        points=points,
        via=['-99', '-99', '-99', '-99', '[1119]'])
    edges_ak2035 = edges_ak2035.fillna(0)
    edges_ak2035.loc[edges_ak2035['Via'] == 0, 'Via'] = '-99'

    edges_ak2035.to_file(paths.RAIL_SERVICES_AK2035_PATH)
    return edges_ak2035, points

def create_railway_services_AK2035_extended(edges_ak2035_ext, points):
    edges_ak2035_ext = add_new_line(
        stations=[
    'Zürich Stadelhofen',
    'Zürich, Kreuzplatz',
    'Zürich, Hegibachplatz',
    'Zürich, Balgrist',
    'Zürich, Rehalp',
    'Waldburg',
    'Spital Zollikerberg',
    'Zollikerberg',
    'Waltikon',
    'Zumikon',
    'Maiacher',
    'Neue Forch',
    'Forch',
    'Scheuren',
    'Neuhaus bei Hinteregg',
    'Hinteregg',
    'Egg',
    'Langwies ZH',
    'Emmat',
    'Esslingen'
    ],
        frequency=4,
        service_name='S18',
        travel_times=[
    1,  # Zürich Stadelhofen -> Zürich, Kreuzstrasse
    2,  # Zürich, Kreuzstrasse -> Zürich, Hegibachplatz
    3,  # Zürich, Hegibachplatz -> Zürich, Balgrist
    3,  # Zürich, Balgrist -> Zürich, Rehalp
    1,  # Zürich, Rehalp -> Waldburg
    1,  # Waldburg -> Spital Zollikerberg
    1,  # Spital Zollikerberg -> Zollikerberg
    1,  # Zollikerberg -> Waltikon
    2,  # Waltikon -> Zumikon
    2,  # Zumikon -> Maiacher
    1,  # Maiacher -> Neue Forch
    1,  # Neue Forch -> Forch
    3,  # Forch -> Scheuren
    2,  # Scheuren -> Neuhaus bei Hinteregg
    2,  # Neuhaus bei Hinteregg -> Hinteregg
    3,  # Hinteregg -> Egg
    2,  # Egg -> Langwies ZH
    1,  # Langwies ZH -> Emmat
    1,  # Emmat -> Esslingen
    4   # Emmat -> Esslingen (korrigiert: 06:21 → 06:25)
    ],
        edges=edges_ak2035_ext,
        points=points)

    edges_ak2035_ext=add_new_line(
        stations=[
    'Winterthur',
    'Winterthur Grüze',
    'Winterthur Seen',
    'Sennhof-Kyburg',
    'Kollbrunn',
    'Rikon',
    'Rämismühle-Zell',
    'Turbenthal',
    'Wila',
    'Saland',
    'Bauma',
    'Steg',
    'Fischenthal',
    'Gibswil',
    'Wald ZH',
    'Tann-Dürnten',
    'Rüti ZH'
    ],
        frequency=2,
        service_name='S26',
        travel_times=[
    5,   # Winterthur -> Winterthur Grüze (05:13 → 05:18)
    3,   # Winterthur Grüze -> Winterthur Seen (05:18 → 05:21)
    3,   # Winterthur Seen -> Sennhof-Kyburg (05:21 → 05:24)
    3,   # Sennhof-Kyburg -> Kollbrunn (05:24 → 05:27)
    3,   # Kollbrunn -> Rikon (05:27 → 05:30)
    2,   # Rikon -> Rämismühle-Zell (05:30 → 05:32)
    4,   # Rämismühle-Zell -> Turbenthal (05:32 → 05:36)
    2,   # Turbenthal -> Wila (05:36 → 05:38)
    6,   # Wila -> Saland (05:38 → 05:44)
    6,   # Saland -> Bauma (05:44 → 05:50)
    4,   # Bauma -> Steg (05:50 → 05:54)
    5,   # Steg -> Fischenthal (05:54 → 05:59)
    2,   # Fischenthal -> Gibswil (05:59 → 06:01)
    7,   # Gibswil -> Wald ZH (06:01 → 06:08)
    5,   # Wald ZH -> Tann-Dürnten (06:08 → 06:13)
    3    # Tann-Dürnten -> Rüti ZH (06:13 → 06:16)
    ],
        edges=edges_ak2035_ext,
        points=points)

    edges_ak2035_ext=add_new_line(
        stations=[
    'Winterthur',
    'Kemptthal',
    'Effretikon',
    'Bassersdorf',
    'Kloten',
    'Kloten Balsberg',
    'Opfikon',
    'Zürich Oerlikon',
    'Zürich Hardbrücke',
    'Zürich HB',
    'Zürich Stadelhofen',
    'Meilen',
    'Uetikon',
    'Männedorf',
    'Stäfa',
    'Uerikon',
    'Feldbach',
    'Kempraten',
    'Rapperswil SG'
    ],
        frequency=2,
        service_name='S7',
        travel_times=[
    4,   # Winterthur -> kemptthal (05:05 → 05:09)
    4,   # Kemptthal -> Effretikon (05:09 → 05:13)
    6,   # Effretikon -> Bassersdorf (05:14 → 05:20)
    5,   # Bassersdorf -> Kloten (05:20 → 05:25)
    2,   # Kloten -> Kloten Balsberg (05:25 → 05:27)
    2,   # Kloten Balsberg -> Opfikon (05:27 → 05:29)
    4,   # Opfikon -> Zürich Oerlikon (05:29 → 05:33)
    4,   # Zürich Oerlikon -> Zürich Hardbrücke (05:33 → 05:37)
    3,   # Zürich Hardbrücke -> Zürich HB (05:37 → 05:40)
    4,   # Zürich HB -> Zürich Stadelhofen (05:40 → 05:44)
    14,  # Zürich Stadelhofen -> Meilen (05:44 → 05:58)
    3,   # Meilen -> Uetikon (05:58 → 06:01)
    2,   # Uetikon -> Männedorf (06:01 → 06:03)
    5,   # Männedorf -> Stäfa (06:03 → 06:08)
    3,   # Stäfa -> Uerikon (06:08 → 06:11)
    5,   # Uerikon -> Feldbach (06:11 → 06:16)
    2,   # Feldbach -> Kempraten (06:16 → 06:18)
    4    # Kempraten -> Rapperswil SG (06:18 → 06:22)
    ],
        edges=edges_ak2035_ext,
        points=points,
        via=[
    '-99','-99', '-99', '-99', '-99', '-99', '-99', '-99', '-99', '-99',
    '[2551, 2534, 1122, 1174, 702, 2459, 999]', '-99', '-99', '-99', '-99', '-99', '-99', '-99'
            ])

    edges_ak2035_ext=add_new_line(
        stations=[
    'Zürich HB',
    'Zürich Stadelhofen',
    'Zürich Tiefenbrunnen',
    'Zollikon',
    'Küsnacht Goldbach',
    'Küsnacht ZH',
    'Erlenbach ZH',
    'Winkel am Zürichsee',
    'Herrliberg-Feldmeilen',
    'Meilen'
],
        frequency=4,
        service_name='I+J',
        travel_times=[
    3,  # Zürich HB → Zürich Stadelhofen (06:00 → 06:03)
    3,  # Zürich Stadelhofen → Zürich Tiefenbrunnen (06:03 → 06:06)
    2,  # Zürich Tiefenbrunnen → Zollikon (06:06 → 06:08)
    2,  # Zollikon → Küsnacht Goldbach (06:08 → 06:10)
    3,  # Küsnacht Goldbach → Küsnacht ZH (06:10 → 06:13)
    2,  # Küsnacht ZH → Erlenbach ZH (06:13 → 06:15)
    2,  # Erlenbach ZH → Winkel am Zürichsee (06:15 → 06:17)
    3,  # Winkel am Zürichsee → Herrliberg-Feldmeilen (06:17 → 06:20)
    3   # Herrliberg-Feldmeilen → Meilen (06:20 → 06:23)
],
        edges=edges_ak2035_ext,
        points=points)
    edges_ak2035_ext = add_new_line(
        stations=['Zürich HB', 'Zürich Stadelhofen', 'Stettbach', 'Winterthur'],
        frequency=4,
        service_name='S11+S12',
        travel_times=[4, 6, 12],
        edges=edges_ak2035_ext,
        points=points,
        via=['-99', '-99', '[638, 666, 1119]'])

    edges_ak2035_ext = add_new_line(
        stations=['Zürich HB', 'Pfäffikon SZ'],
        frequency=2,
        service_name='fast_ZH_PF',
        travel_times=[26],
        edges=edges_ak2035_ext,
        points=points,
        via=['-99'])

    edges_ak2035_ext = edges_ak2035_ext.fillna(0)
    edges_ak2035_ext.loc[edges_ak2035_ext['Via'] == 0, 'Via'] = '-99'

    edges_ak2035_ext.to_file(paths.RAIL_SERVICES_AK2035_EXTENDED_PATH)


def network_in_corridor(poly):
    """
    This function takes a polygon and a network and selects the parts of the network within the polygon.
    # Filter the infrastructure elements that lie within a given polygon
    # Points within the corridor are stored in "data/Network/processed/points_corridor.gpkg"
    # Edges within the corridor are stored in "data/Network/processed/edges_corridor.gpkg"
    # Edges crossing the corridor border are stored in "data/Network/processed/edges_on_corridor.gpkg"

    # In general, the final product of this function is edges_with_attributes.gpkg and points_with_attributes.gpkg

    :param poly: Polygon in which to search for network elements.
    :return: None
    """

    # Create a GeoDataFrame from the polygon
    polygon = gpd.GeoDataFrame({'geometry': [poly]})
    polygon.crs = "epsg:2056"

    edges = gpd.read_file("data/Network/processed/edges.gpkg")
    points = gpd.read_file("data/Network/processed/points.gpkg")


    # connect the two edges with id 81 and 70 by their closest end points to make one new out of it

    # filter points in polygon
    points_corridor = gpd.sjoin(points, polygon, how="inner")
    points_corridor = points_corridor.drop(columns=["index_right"]) 
    points_corridor.to_file("data/Network/processed/points_corridor.gpkg")

    # Check if point are in polygon if so add True as "within_corridor" attribute otherwise False
    points['within_corridor'] = points.apply(lambda row: polygon.contains(row.geometry), axis=1)

    # get edges within the polygon
    edges_corridor = gpd.sjoin(edges, polygon, how="inner")
    #edges_corridor = edges_corridor.drop(columns=["start_access", "end_access", "index_right"])
    edges_corridor = edges_corridor.drop(columns=["index_right"])
    edges_corridor.to_file("data/Network/processed/edges_in_corridor.gpkg")

    # Get edges crossed by polygon frame
    # Only keep edges with exactly on endpoint in the polygon and the other outside the polygon (= point_corridor)
    # Define the function to check endpoints
    def is_one_endpoint_inside(edge_geom, poly):
        start_point = Point(edge_geom.coords[0])
        end_point = Point(edge_geom.coords[-1])
        return poly.contains(start_point) != poly.contains(end_point)

    # Apply the function directly in the apply method
    edges['polygon_border'] = edges['geometry'].apply(lambda x: is_one_endpoint_inside(x, polygon))

    # Apply the function to filter edges and save the filtered edges in seperate file
    #edges_crossing_polygon = edges[edges.apply(lambda x: is_one_endpoint_inside(x, polygon), axis=1)]
    edges_crossing_polygon = edges[edges["polygon_border"] == True]
    edges_crossing_polygon.to_file("data/Network/processed/edges_on_corridor_border.gpkg")

    points_temp = points.copy()
    points_temp['buffered_points'] = points_temp['geometry'].buffer(1e-6)
    points_temp = points_temp.set_geometry("buffered_points")

    # Perform the spatial join and create a temporary DataFrame
    temp_joined = gpd.sjoin(points_temp, edges_crossing_polygon, how="left", predicate="intersects", lsuffix="left",
                            rsuffix="right")
    temp_joined = temp_joined.drop_duplicates('ID_point')
    
    # Add 'on_corridor_border' attribute to points
    # A point is on the corridor border if it has a match in the spatial join (non-null index_right)
    points['on_corridor_border'] = temp_joined['index_right'].notna()

    # Step 4 & 5: Apply the function to each edge in df2
    edges = edges.rename(columns={'Link NR': 'ID_edge', 'Capacity': 'TrainCapacity', 'TotalPeakCapacity': 'capacity',
                                  'TravelTime': 'tt'})
    edges = edges.drop(
        ['FromStation', 'ToStation', 'FromCode', 'ToCode', 'FromGde', 'ToGde', 'E_KOORD_O', 'E_KOORD_D', 'N_KOORD_O',
         'N_KOORD_D', 'Speed'], axis=1)

    # Store values as integer
    # Drop rows with None values
    edges.dropna(inplace=True)
    edges['capacity'] = edges['capacity'].astype(int)
    edges['tt'] = edges['tt'].astype(int)

    # Check if there are None values in the capacity and ffs columns

    edges["ID_edge"] = edges.index

    points.to_file(r"data\Network\processed\points_with_attribute.gpkg")
    edges.to_file(r"data\Network\processed\edges_with_attribute.gpkg")


def only_links_to_corridor():
    # Load the new_links and all access points datasets
    new_links = gpd.read_file(r"data\Network\processed\filtered_new_links.gpkg")
    all_access_points = gpd.read_file(r"data\Network\processed\points_with_attribute.gpkg")

    # Filter all_access_points to retain only those within the corridor
    access_corridor = all_access_points[all_access_points["within_corridor"] == 1]

    # Extract unique IDs of points within the corridor
    corridor_ids = access_corridor["ID_point"].unique()

    # Filter new_links to keep only rows where from_ID_new is in corridor_ids
    filtered_new_links = new_links[new_links["from_ID_new"].isin(corridor_ids)]

    # Add the new 'dev_id' column with unique values starting from 100000
    filtered_new_links = filtered_new_links.reset_index(drop=True)  # Reset index if needed
    filtered_new_links['dev_id'] = 100000 + filtered_new_links.index

    # Save the filtered new_links to a new file if needed
    filtered_new_links.to_file("data/Network/processed/filtered_new_links_in_corridor.gpkg", driver="GPKG")


def save_focus_area_shapefile(e_min, e_max, n_min, n_max, margin):
    # Function to create a polygon from given coordinates
    def polygon_from_points(e_min, e_max, n_min, n_max, margin=0):
        return Polygon([
            (e_min - margin, n_min - margin),
            (e_min - margin, n_max + margin),
            (e_max + margin, n_max + margin),
            (e_max + margin, n_min - margin),
            (e_min - margin, n_min - margin)
        ])

    # Spatial limits of the research corridor

    # Create polygons
    innerboundary = polygon_from_points(e_min=e_min, e_max=e_max, n_min=n_min, n_max=n_max)
    outerboundary = polygon_from_points(e_min=e_min, e_max=e_max, n_min=n_min, n_max=n_max, margin=margin)

    # Save innerboundary as a separate shapefile
    innerboundary_gdf = gpd.GeoDataFrame({"name": ["innerboundary"], "geometry": [innerboundary]}, crs="EPSG:2056")
    innerboundary_gdf.to_file("data/_basic_data/innerboundary.shp")

    # Save outerboundary as a separate shapefile
    outerboundary_gdf = gpd.GeoDataFrame({"name": ["outerboundary"], "geometry": [outerboundary]}, crs="EPSG:2056")
    outerboundary_gdf.to_file("data/_basic_data/outerboundary.shp")

    print("Shapefiles 'innerboundary.shp' and 'outerboundary.shp' saved successfully in 'data_basic_data'!")
    return innerboundary, outerboundary


def get_station_id(station_name, points, name_column='NAME', id_column='ID_point'):
    """ Get the station ID for a given station name.
    Parameters:
        station_name (str): The name of the station to search for.
        points (str): GPD File containing the station data.
        name_column (str): Column name containing station names. Default is 'NAME'.
        id_column (str): Column name containing station IDs. Default is 'ID_point'.

    Returns:
        int: The station ID corresponding to the given station name.

    Raises:
        ValueError: If the station name is not found in the dataset.
    """
    # Load the GeoPackage file

    # Filter the dataset for the given station name
    station_row = points.loc[points[name_column] == station_name]

    # Check if the station exists
    if station_row.empty:
        raise ValueError(f"Station '{station_name}' not found in the dataset.")

    # Return the station ID
    return station_row[id_column].iloc[0]