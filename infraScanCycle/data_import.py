import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import math
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point, MultiPoint, shape, box
from shapely.ops import split, snap, linemerge, unary_union
from rasterio import crs
from rasterio.transform import from_origin
from rasterio.features import shapes, rasterize
from shapely.geometry import shape, Polygon
from geopandas.tools import sjoin
from plots import *



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
    print(empl20.head(10).to_string())
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
    gdf.to_file('data/landuse_landcover/processed/lake_data_zh.gpkg')
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
    ##################################################################################
    # TODO: REFACTOR FOR CYCLING NETWORK (InfraScanCycle)
    # 1. CHANGE: Replace highway-specific source files (e.g., Road_Link.csv) with cycle paths.
    #    If using OSM, prioritize tags: 'cycleway', 'highway=cycleway', and 'surface'.
    # 2. REMOVE: Attributes like 'Number of Lanes' or 'Capacity per day'. Cycle bottlenecks
    #    are driven by intersection safety and infrastructure quality, not lane volume.
    # 3. ADD: 'Surface Quality' and 'Segregation' (e.g., separated vs. mixed traffic).
    #    This is critical for cost-scoring (e.g., gravel has a higher cost/friction than asphalt).
    ##################################################################################
    # Read csv files of node, links and link attributes to a Pandas DataFrame

    edge_table = pd.read_csv(r"data/Network/Cycle_Link.csv", sep=";") #TODO
    node_table = pd.read_csv(r"data/Network/Cycle_Node.csv", sep=";") #TODO
    link_attribute = pd.read_csv(r"data/Network/Cycle_LinkType.csv", sep=";") #TODO

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
    # Drop unwanted columns
    nw_gdf = nw_gdf.drop(['point_O', 'point_D', "line"], axis=1)

    # --- FIX START: MERGE NETWORKS ---
    print("Merging OSMnx feeder network with ALLTAG main network...")

    # 1. Load the ALLTAG network you saved earlier in the script
    alltag_path = r"data/converted/transport_network.gpkg"
    try:
        alltag_edges = gpd.read_file(alltag_path, layer='edges')

        # 2. Combine both GeoDataFrames into one master network
        # (Both should be in EPSG:2056 at this point)
        combined_edges = pd.concat([alltag_edges, nw_gdf], ignore_index=True)

        # 3. Save as a NEW combined file so you don't destroy the raw data
        combined_path = r"data/converted/transport_network_combined.gpkg"
        combined_edges.to_file(combined_path, layer='edges', driver="GPKG")

        print(f"Success! Merged {len(alltag_edges)} ALLTAG edges and {len(nw_gdf)} OSMnx edges.")
        print(f"Combined network saved to: {combined_path}")

    except Exception as e:
        print(f"Error merging networks: {e}")
        # Fallback just in case ALLTAG isn't generated yet
        nw_gdf.to_file(r"data/converted/transport_network_osmnx_only.gpkg", layer='edges', driver="GPKG")
    # --- FIX END ---

    return


def load_nw_zh():
    all_roads = gpd.read_file(r"data/Network/Strassennetz/TBA_STR_ACHS_L.shp")
    print(all_roads.columns)

    ##################################################################################
    # TODO: REDEFINE "ACCESS POINTS" (InfraScanCycle Adaptation)
    # 1. LOGIC: Unlike highways where access is limited to ramps, cycle "access"
    #    points are where users enter the high-quality bike network.
    # 2. CHANGE (Line 279): Replace 'highway_access.csv' with Points of Interest (POI)
    #    that drive cycling demand (e.g., S-Bahn stations, schools, bike-sharing hubs).
    # 3. BUFFER UPDATE (Line 300): Tighten the matching tolerance. While 300m works
    #    for cars, cycle matching should be accurate within 5–10m to ensure points
    #    snap to the correct side of divided roads or specific cycle tracks.
    ##################################################################################


def map_access_points_on_network(network_edges, df_access):
    import geopandas as gpd
    from shapely.geometry import Point
    from collections import Counter
    import os

    os.makedirs('data/Network/processed', exist_ok=True)

    # Ensure network_edges is a proper GeoDataFrame
    if not isinstance(network_edges, gpd.GeoDataFrame):
        raise TypeError(f"network_edges must be a GeoDataFrame, got {type(network_edges)}")

    if network_edges.crs.to_epsg() != 2056:
        network_edges = network_edges.to_crs("EPSG:2056")

    # ------------------------------------------------------------------
    # 1. DERIVE NODES from edge endpoints
    # ------------------------------------------------------------------
    coord_to_id = {}
    node_records = []
    edge_node_pairs = []

    def get_or_create_node(coord):
        coord = (round(coord[0], 6), round(coord[1], 6))
        if coord not in coord_to_id:
            nid = len(coord_to_id)
            coord_to_id[coord] = nid
            node_records.append({'node_id': nid, 'x': coord[0], 'y': coord[1],
                                  'geometry': Point(coord)})
        return coord_to_id[coord]

    for geom in network_edges.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = list(geom.geoms) if geom.geom_type == 'MultiLineString' else [geom]
        src_id = get_or_create_node(parts[0].coords[0])
        tgt_id = get_or_create_node(parts[-1].coords[-1])
        edge_node_pairs.append((src_id, tgt_id))

    access_points = gpd.GeoDataFrame(node_records, geometry='geometry', crs="EPSG:2056")
    access_points = access_points.set_index('node_id')

    # ------------------------------------------------------------------
    # 2. COMPUTE NODE DEGREE and flag intersections
    # ------------------------------------------------------------------
    degree_count = Counter()
    for src, tgt in edge_node_pairs:
        degree_count[src] += 1
        degree_count[tgt] += 1

    access_points['degree'] = access_points.index.map(degree_count).fillna(0).astype(int)
    access_points['is_intersection'] = access_points['degree'] >= 3
    access_points['is_access_point'] = True  # all nodes accessible on cycling infra

    access_points.to_file('data/Network/processed/access_points.gpkg', driver='GPKG')
    print(f"  -> {len(access_points)} access points identified "
          f"({access_points['is_intersection'].sum()} intersections, "
          f"{(~access_points['is_intersection']).sum()} endpoints)")

    # ------------------------------------------------------------------
    # 3. PREPARE HUBS from veloparking CSV
    # ------------------------------------------------------------------
    df_access = df_access.copy()
    df_access.columns = df_access.columns.str.strip()

    hubs = gpd.GeoDataFrame(
        df_access,
        geometry=gpd.points_from_xy(df_access['E'], df_access['N']),
        crs='EPSG:2056'
    )

    # ------------------------------------------------------------------
    # 4. SNAP HUBS to nearest access point node
    # ------------------------------------------------------------------
    hubs_snapped = gpd.sjoin_nearest(
        hubs,
        access_points[['geometry', 'is_intersection', 'degree']].reset_index(),
        how='left',
        distance_col='snap_distance_m'
    )

    max_snap_dist = 200  # meters
    hubs_snapped = hubs_snapped[hubs_snapped['snap_distance_m'] <= max_snap_dist].copy()

    matched_geoms = access_points.loc[hubs_snapped['node_id'], 'geometry'].values
    hubs_snapped = hubs_snapped.copy()
    hubs_snapped['geometry'] = matched_geoms
    hubs_snapped['E_snapped'] = hubs_snapped.geometry.x
    hubs_snapped['N_snapped'] = hubs_snapped.geometry.y
    hubs_snapped['is_destination'] = True

    hubs_snapped.to_file('data/Network/processed/hubs_destinations.gpkg', driver='GPKG')
    print(f"  -> {len(hubs_snapped)} hubs snapped to network "
          f"(of {len(hubs)} total, {len(hubs) - len(hubs_snapped)} outside {max_snap_dist}m)")

    return access_points, hubs_snapped

def reformat_network():
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point, MultiPoint
    from shapely.ops import unary_union, snap, split
    import os

    print("\nreformat_network: start\n")

    # ------------------------------------------------------------------
    # 1. LOAD already-processed edges and hub destinations
    # ------------------------------------------------------------------
    edges_path = 'data/Network/processed/edges.gpkg'
    hubs_path  = 'data/Network/processed/hubs_destinations.gpkg'

    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"Missing: {edges_path} — run import_network_GIS_ALLTAG() first")
    if not os.path.exists(hubs_path):
        raise FileNotFoundError(f"Missing: {hubs_path} — run map_access_points_on_network() first")

    edges_gdf = gpd.read_file(edges_path)
    hubs      = gpd.read_file(hubs_path)

    if edges_gdf.crs.to_epsg() != 2056:
        edges_gdf = edges_gdf.to_crs("EPSG:2056")
    if hubs.crs.to_epsg() != 2056:
        hubs = hubs.to_crs("EPSG:2056")

    print(f"  Loaded {len(edges_gdf)} edges, {len(hubs)} hubs")

    # ------------------------------------------------------------------
    # 2. IDENTIFY JUNCTIONS (degree >= 3) and cluster within 2m
    # ------------------------------------------------------------------
    starts = edges_gdf.geometry.apply(lambda g: Point(g.coords[0]))
    ends   = edges_gdf.geometry.apply(lambda g: Point(g.coords[-1]))

    all_coords = (
        list(zip(starts.x, starts.y)) +
        list(zip(ends.x,   ends.y))
    )
    coord_counts    = pd.Series(all_coords).value_counts()
    junction_coords = coord_counts[coord_counts >= 3].index.tolist()

    junctions = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in junction_coords],
        crs="EPSG:2056"
    ).reset_index(drop=True)

    print(f"  {len(junctions)} junction nodes identified")

    # Cluster junctions within 2m → replace with centroid
    if len(junctions) > 0:
        buffered = junctions.copy()
        buffered['geometry'] = junctions.buffer(2)
        joined = gpd.sjoin(buffered, junctions, how='left', predicate='intersects')
        mean_coords = joined.groupby('index_right')['geometry'].apply(
            lambda x: unary_union(x).centroid
        )
        junctions['geometry'] = junctions.apply(
            lambda row: mean_coords.get(row.name, row.geometry), axis=1
        )
        junctions = junctions.drop_duplicates(subset='geometry').reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. SPLIT EDGES at hub locations
    # ------------------------------------------------------------------
    hub_points = hubs.geometry.tolist()
    all_split_points = hub_points + junctions.geometry.tolist()

    if len(all_split_points) == 0:
        print("  Warning: no split points found — keeping edges as-is")
        edges_split = edges_gdf.copy()
    else:
        point_collection = MultiPoint(all_split_points)
        new_edges = []
        for _, edge in edges_gdf.iterrows():
            try:
                snapped_geom = snap(edge.geometry, point_collection, tolerance=1.0)
                split_result = split(snapped_geom, point_collection)
                if len(split_result.geoms) > 1:
                    for part in split_result.geoms:
                        new_row = edge.copy()
                        new_row['geometry'] = part
                        new_row['length_m'] = part.length
                        new_edges.append(new_row)
                else:
                    new_edges.append(edge)
            except Exception as e:
                print(f"  Warning: could not split edge {edge.name}: {e}")
                new_edges.append(edge)

        edges_split = gpd.GeoDataFrame(new_edges, crs="EPSG:2056").reset_index(drop=True)

    edges_split = edges_split[edges_split.geometry.length > 0.1].copy()
    edges_split['ID_edge'] = range(len(edges_split))

    # ------------------------------------------------------------------
    # 4. BUILD FINAL NODE TABLE
    # ------------------------------------------------------------------
    node_starts = edges_split.geometry.apply(lambda g: Point(g.coords[0]))
    node_ends   = edges_split.geometry.apply(lambda g: Point(g.coords[-1]))

    all_node_geoms = pd.concat([node_starts, node_ends], ignore_index=True).drop_duplicates()
    nodes_gdf = gpd.GeoDataFrame(
        {'geometry': all_node_geoms.values},
        crs="EPSG:2056"
    ).reset_index(drop=True)
    nodes_gdf['ID_point'] = nodes_gdf.index

    # ------------------------------------------------------------------
    # 5. FLAG intersections and destinations
    # ------------------------------------------------------------------
    nodes_buf = nodes_gdf.copy()
    nodes_buf['geometry'] = nodes_gdf.buffer(1e-6)
    nodes_buf = nodes_buf.reset_index(drop=True)
    nodes_buf['orig_idx'] = nodes_buf.index  # track original row position

    # Intersections
    if len(junctions) > 0:
        junc_join = gpd.sjoin(
            nodes_buf[['orig_idx', 'geometry']],
            junctions.reset_index(drop=True),
            how='left', predicate='intersects'
        )
        is_junc = junc_join.groupby('orig_idx')['index_right'].apply(
            lambda x: x.notnull().any()
        ).reindex(range(len(nodes_gdf)), fill_value=False)
        nodes_gdf['is_intersection'] = is_junc.astype(int).values
    else:
        nodes_gdf['is_intersection'] = 0

    # Destinations
    if len(hubs) > 0:
        hub_join = gpd.sjoin(
            nodes_buf[['orig_idx', 'geometry']],
            hubs[['geometry']].reset_index(drop=True),
            how='left', predicate='intersects'
        )
        is_dest = hub_join.groupby('orig_idx')['index_right'].apply(
            lambda x: x.notnull().any()
        ).reindex(range(len(nodes_gdf)), fill_value=False)
        nodes_gdf['is_destination'] = is_dest.astype(int).values
    else:
        nodes_gdf['is_destination'] = 0

    # ------------------------------------------------------------------
    # 6. MAP start/end node IDs onto edges
    # ------------------------------------------------------------------
    nodes_lookup = nodes_buf[['ID_point', 'geometry']].copy()

    start_gdf = gpd.GeoDataFrame(geometry=node_starts.values, crs="EPSG:2056").reset_index(drop=True)
    end_gdf   = gpd.GeoDataFrame(geometry=node_ends.values,   crs="EPSG:2056").reset_index(drop=True)

    start_join = gpd.sjoin(start_gdf, nodes_lookup, how='left', predicate='intersects')
    start_join = start_join[~start_join.index.duplicated(keep='first')]
    end_join   = gpd.sjoin(end_gdf,   nodes_lookup, how='left', predicate='intersects')
    end_join   = end_join[~end_join.index.duplicated(keep='first')]

    edges_split['start'] = start_join['ID_point'].values
    edges_split['end']   = end_join['ID_point'].values

    # ------------------------------------------------------------------
    # 7. EXPORT
    # ------------------------------------------------------------------
    os.makedirs('data/Network/processed', exist_ok=True)
    nodes_gdf.to_file('data/Network/processed/points.gpkg', driver='GPKG')
    edges_split.to_file('data/Network/processed/edges.gpkg', driver='GPKG')

    print(f"  -> {len(edges_split)} edges and {len(nodes_gdf)} nodes")
    print(f"     {nodes_gdf['is_intersection'].sum()} intersections, "
          f"{nodes_gdf['is_destination'].sum()} hub destinations")
    print("\nreformat_network: end\n")

    return nodes_gdf, edges_split  # ← was missing in the version that returned None


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
    """
    Assigns cycling-specific capacity and free-flow speed to edges
    based on ROUTENTYP from the ALLTAG dataset.

    ROUTENTYP → ffs (km/h) / capacity (bikes/h):
        Veloschnellroute          → 25 km/h / 1000 bikes/h
        Hauptverbindung           → 20 km/h /  600 bikes/h
        Nebenverbindung           → 15 km/h /  300 bikes/h
        Zusätzliche Freizeitverbindung → 15 km/h / 200 bikes/h
        default                   → 18 km/h /  400 bikes/h
    """
    ROUTENTYP_ATTRS = {
        'Veloschnellroute':               {'ffs': 25, 'capacity': 1000},
        'Hauptverbindung':                {'ffs': 20, 'capacity':  600},
        'Nebenverbindung':                {'ffs': 15, 'capacity':  300},
        'Zusätzliche Freizeitverbindung': {'ffs': 15, 'capacity':  200},
    }
    DEFAULT_ATTRS = {'ffs': 18, 'capacity': 400}

    edges = gpd.read_file('data/Network/processed/edges_with_attribute.gpkg')
    if edges.crs is None:
        edges = edges.set_crs("EPSG:2056")

    print(f"  Edge columns: {edges.columns.tolist()}")

    # ------------------------------------------------------------------
    # 1. ROUTENTYP → ffs and capacity
    #    Use startswith match to survive any column name truncation
    # ------------------------------------------------------------------
    routentyp_col = next((c for c in edges.columns if c.upper().startswith('ROUTENTYP')), None)

    def lookup(routentyp, key):
        return ROUTENTYP_ATTRS.get(str(routentyp).strip(), DEFAULT_ATTRS)[key]

    if routentyp_col:
        edges['ffs']      = edges[routentyp_col].apply(lambda r: lookup(r, 'ffs'))
        edges['capacity'] = edges[routentyp_col].apply(lambda r: lookup(r, 'capacity'))
    else:
        print("  Warning: ROUTENTYP column not found — applying defaults")
        edges['ffs']      = DEFAULT_ATTRS['ffs']
        edges['capacity'] = DEFAULT_ATTRS['capacity']

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

    edges['ID_edge'] = range(len(edges))

    edges.to_file('data/Network/processed/edges_with_attribute.gpkg', driver='GPKG')

    print(f"  -> {len(edges)} edges attributed")
    print(f"     ffs range:      {edges['ffs'].min()}–{edges['ffs'].max()} km/h")
    print(f"     capacity range: {edges['capacity'].min()}–{edges['capacity'].max()} bikes/h")
    print(f"     one-way edges:  {edges['oneway'].sum()}")
    print(f"     tt_min range:   {edges['tt_min'].min():.1f}–{edges['tt_min'].max():.1f} min")

    return edges

def network_in_corridor(polygon):
    os.makedirs('data/Network/processed', exist_ok=True)

    # ------------------------------------------------------------------
    # 1. LOAD inputs
    # ------------------------------------------------------------------
    edges  = gpd.read_file('data/Network/processed/edges.gpkg')
    points = gpd.read_file('data/Network/processed/points.gpkg')

    if edges.crs is None:
        edges  = edges.set_crs("EPSG:2056")
    if points.crs is None:
        points = points.set_crs("EPSG:2056")

    poly_gdf  = gpd.GeoDataFrame({'geometry': [polygon]}, crs="EPSG:2056")
    poly_geom = polygon

    # ------------------------------------------------------------------
    # 2. NODES inside corridor
    # ------------------------------------------------------------------
    points_corridor = gpd.sjoin(points, poly_gdf, how='inner', predicate='within') \
                         .drop(columns=['index_right'], errors='ignore') \
                         .reset_index(drop=True)
    points_corridor.to_file('data/Network/processed/points_corridor.gpkg', driver='GPKG')

    # ------------------------------------------------------------------
    # 3. EDGES strictly inside corridor
    # ------------------------------------------------------------------
    edges_corridor = gpd.sjoin(edges, poly_gdf, how='inner', predicate='within') \
                        .drop(columns=['index_right'], errors='ignore') \
                        .reset_index(drop=True)
    edges_corridor.to_file('data/Network/processed/edges_in_corridor.gpkg', driver='GPKG')

    # ------------------------------------------------------------------
    # 4. EDGES crossing the corridor border (XOR on endpoints)
    # ------------------------------------------------------------------
    def one_endpoint_inside(geom, poly):
        return poly.contains(Point(geom.coords[0])) != poly.contains(Point(geom.coords[-1]))

    edges['on_border'] = edges.geometry.apply(lambda g: one_endpoint_inside(g, poly_geom))
    edges_border = edges[edges['on_border']].copy()
    edges_border.to_file('data/Network/processed/edges_on_corridor_border.gpkg', driver='GPKG')

    # ------------------------------------------------------------------
    # 5. FLAG nodes — use index-based groupby to avoid length mismatches
    # ------------------------------------------------------------------
    # within_corridor: vectorised contains check — no sjoin needed
    points['within_corridor'] = points.geometry.apply(lambda g: poly_geom.contains(g))

    # on_corridor_border: buffer + sjoin, then GROUP BY original index
    #   groupby().any() collapses multiple matches per node to a single True/False
    points_buf = points.copy()
    points_buf['geometry'] = points.buffer(1e-6)
    points_buf = points_buf.reset_index().rename(columns={'index': 'orig_idx'})

    border_join = gpd.sjoin(
        points_buf[['orig_idx', 'geometry']],
        edges_border[['geometry']].reset_index(drop=True),
        how='left',
        predicate='intersects'
    )
    # Collapse: one row per original node, True if ANY border edge was matched
    on_border_flag = border_join.groupby('orig_idx')['index_right'].apply(
        lambda x: x.notnull().any()
    )
    points['on_corridor_border'] = on_border_flag.reindex(points.index, fill_value=False).values

    # ------------------------------------------------------------------
    # 6. FLAG edges
    # ------------------------------------------------------------------
    edges['within_corridor'] = edges.geometry.apply(lambda g: poly_geom.contains(g))

    # ------------------------------------------------------------------
    # 7. SAVE
    # ------------------------------------------------------------------
    points.to_file('data/Network/processed/points_with_attribute.gpkg', driver='GPKG')
    edges.to_file('data/Network/processed/edges_with_attribute.gpkg',   driver='GPKG')

    print(f"  -> {len(points_corridor)} nodes inside corridor (of {len(points)} total)")
    print(f"  -> {len(edges_corridor)} edges inside corridor, "
          f"{len(edges_border)} crossing border (of {len(edges)} total)")

    return points_corridor, edges_corridor, edges_border


def map_values_to_nodes():
    os.makedirs('data/Network/processed', exist_ok=True)

    # ------------------------------------------------------------------
    # 1. LOAD hubs and strip all join-artifact columns
    # ------------------------------------------------------------------
    hubs = gpd.read_file('data/Network/processed/hubs_destinations.gpkg')
    if hubs.crs.to_epsg() != 2056:
        hubs = hubs.to_crs("EPSG:2056")

    drop_cols = ['E', 'N', 'E_snapped', 'N_snapped', 'snap_distance_m',
                 'node_id', 'index_right', 'index_left']
    hub_attrs = hubs.drop(columns=drop_cols, errors='ignore').reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Helper: join hub attributes onto a nodes GeoDataFrame
    # ------------------------------------------------------------------
    def attach_hub_attrs(nodes_gdf, hub_attrs_gdf, intersection_col='is_intersection'):
        nodes_clean     = nodes_gdf.reset_index(drop=True)
        hub_attrs_clean = hub_attrs_gdf.reset_index(drop=True)

        joined = nodes_clean.sjoin_nearest(
            hub_attrs_clean,
            how='left',
            distance_col='hub_dist_m'
        ).drop(columns=['index_right'], errors='ignore')

        joined = joined[~joined.index.duplicated(keep='first')].reset_index(drop=True)

        # Rename suffixed columns BEFORE identifying hub_cols
        for suffix in ['_left', '_right']:
            for col in ['is_intersection', 'is_destination']:
                suffixed = col + suffix
                if suffixed in joined.columns and col not in joined.columns:
                    joined = joined.rename(columns={suffixed: col})
                elif suffixed in joined.columns:
                    joined = joined.drop(columns=[suffixed])

        # Identify hub attribute columns (everything added by the join)
        node_cols = list(nodes_clean.columns) + ['hub_dist_m']
        hub_cols  = [c for c in joined.columns if c not in node_cols]

        # Clear hub attributes for intersection/topology nodes
        if intersection_col in joined.columns and hub_cols:
            joined[hub_cols] = joined[hub_cols].astype(object)
            joined.loc[joined[intersection_col] == 1, hub_cols] = np.nan

        joined = joined.set_crs("EPSG:2056", allow_override=True)
        joined = joined.loc[:, ~joined.columns.duplicated()]

        return joined

    # ------------------------------------------------------------------
    # 3. ALL nodes — read from points.gpkg (has is_intersection, is_destination)
    # ------------------------------------------------------------------
    nodes_all = gpd.read_file('data/Network/processed/points.gpkg')
    if nodes_all.crs is None:
        nodes_all = nodes_all.set_crs("EPSG:2056")

    nodes_all_attr = attach_hub_attrs(nodes_all, hub_attrs)
    nodes_all_attr.to_file('data/Network/processed/points_attribute.gpkg', driver='GPKG')
    print(f"  -> points_attribute.gpkg: {len(nodes_all_attr)} nodes")

    # ------------------------------------------------------------------
    # 4. CORRIDOR nodes — filter from points.gpkg using points_corridor IDs
    #    Do NOT read points_corridor.gpkg — it lacks is_intersection
    # ------------------------------------------------------------------
    points_corridor_ids = gpd.read_file('data/Network/processed/points_corridor.gpkg')
    corridor_ids = set(points_corridor_ids['ID_point'].values)

    nodes_corridor = nodes_all[nodes_all['ID_point'].isin(corridor_ids)].copy().reset_index(drop=True)

    nodes_corridor_attr = attach_hub_attrs(nodes_corridor, hub_attrs)
    nodes_corridor_attr.to_file('data/Network/processed/points_corridor_attribute.gpkg', driver='GPKG')
    print(f"  -> points_corridor_attribute.gpkg: {len(nodes_corridor_attr)} nodes")

    return nodes_all_attr, nodes_corridor_attr


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

##################################################################################
    # TODO: REVISE LAND USE & PROTECTED AREAS (get_protected_area)
    # 1. LOGIC: Unlike 4-lane highways, 2-meter cycle paths are often permitted in
    #    zones where heavy infrastructure is banned due to their lower footprint.
    # 2. CHANGE (Lines 512–540): Review the 'fully_protected' list. While sensitive
    #    zones like 'hochmoor' (high moor) remain off-limits, cycle paths are
    #    frequently allowed in 'wald' (forests) or 'bln' (protected landscapes).
    # 3. ACTION: Reclassify 'wald' (forest) and 'fruchtfolgeflaeche' (crop rotation)
    #    from the "banned" category into a "low-impact" or "permitted" category.
    ##################################################################################
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
        dissolved = gdf.dissolve()
        # Append the dissolved geometry to the list
        dissolved_geometries.append(dissolved.geometry.unary_union)
        #dissolved_geometries = gpd.GeoDataFrame(pd.concat([pd.DataFrame(dissolved_geometries), pd.DataFrame(dissolved.geometry.unary_union)], ignore_index=True))

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
    print(areal_stat.shape)

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
    print(protected_area.shape)
    protected_area_full = fill_raster_dataframe(protected_area)
    print(protected_area_full.head(10).to_string())
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

