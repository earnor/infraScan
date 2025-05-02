import geopandas as gpd
import pandas as pd
from shapely import Point


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
