import momepy
import osmnx
import geopandas as gpd
import pyogrio
import matplotlib.pyplot as plt

from fontTools.ttLib.woff2 import bboxFormat
from shapely.geometry import Polygon


# Dübendorf - Hinwil corridor
left = 8.5800    # West (slightly west of Dübendorf)
bottom = 47.2700  # South (below Hinwil/Aathal)
right = 8.8700   # East (slightly east of Hinwil)
top = 47.4200    # North (above Dübendorf/Volketswil)

#import network from OSMnx
bbox = ([left, bottom, right, top])
G = osmnx.graph.graph_from_bbox(bbox, network_type='bike', simplify=True, retain_all=True, truncate_by_edge=True, custom_filter=None)

#covert OSMnx graph H to GeoDataFrames
gdf_nodes, gdf_edges = osmnx.graph_to_gdfs(G)
#gis data
path = '/Users/ruki/PycharmProjects/infraScan/infraScanCycle/data/raw/ALLTAG/OGD_VELO_ALLTAG_NETZ_L_M.shp'
df = gpd.read_file(path, engine='pyogrio')

#Fix: Explode MultiLineStrings into individual LineStrings
df = df.explode(index_parts=False)

#convert data frame df to primal graph
H = momepy.gdf_to_nx(df, approach= 'primal')

#convert momepy graph H to GeoDataFrames
gdf_h_edges = momepy.nx_to_gdf(H, points=False)

#align coordinate system
gdf_edges = gdf_edges.to_crs(epsg=2056)
gdf_h_edges = gdf_h_edges.to_crs(epsg=2056)

#check if they even exist in the same space
print(f"OSM Edges Bounds: {gdf_edges.total_bounds}")
print(f"GIS Data Bounds: {gdf_h_edges.total_bounds}")

#ensure that CRS is recognized
if gdf_h_edges.crs != gdf_h_edges.crs:
    gdf_h_edges = gdf_h_edges.to_crs(gdf_edges.crs)

#use small buffer to catch near-overlaps
gdf_h_buffered = gdf_h_edges.copy()
gdf_h_buffered['geometry'] = gdf_h_buffered.geometry.buffer(5)


#filter network graph G
res_intersection = gpd.sjoin(gdf_edges, gdf_h_buffered, how="inner", predicate="intersects")

#extract node
res_nodes = res_intersection.copy()
res_nodes['geometry'] = res_nodes.geometry.boundary

#plot OSMnx import
#osmnx.plot.plot_graph(G, ax=None, figsize=(8, 8), bgcolor='#111111', node_color='w', node_size=15, node_alpha=None, node_edgecolor='none', node_zorder=1, edge_color='#999999', edge_linewidth=1, edge_alpha=None, bbox=None, show=True, close=False, save=False, filepath=None, dpi=300)

#plot intersection
if res_intersection.empty:
    print("Warning: Intersection is still empty. Plotting both layers to see the gap.")
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf_edges.plot(ax=ax, color='red', alpha=0.5, label='OSM')
    gdf_h_edges.plot(ax=ax, color='blue', alpha=0.5, label='Local GIS')
    plt.legend()
else:
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#111111')
    # Plot Edges (Lines)
    res_intersection.plot(ax=ax, color='#1fb5ad', linewidth=1.5, label='Edges')
    # Plot Nodes (Endpoints)
    res_nodes.plot(ax=ax, color='white', markersize=5, zorder=3, label='Nodes')
    plt.legend()
    plt.show()

#Create a copy
df_export = res_intersection.copy()

#Convert the geometry column to WKT (Well-Known Text) string format
df_export['geometry'] = df_export['geometry'].apply(lambda x: x.wkt)

#Export to CSV
output_path = "res_intersection_network.csv"
df_export.to_csv(output_path, index=False)

print(f"Successfully exported {len(df_export)} rows to {output_path}")