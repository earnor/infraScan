from operator import truediv
from turtledemo.chaos import plot

import momepy
import osmnx
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry import LineString, MultiLineString
import os

os.chdir(r'/Users/ruki/PycharmProjects/infraScan/infraScanCycle')  # TODO: implement the same code for data_converter

#gis data
path = 'data/raw/ALLTAG/OGD_VELO_ALLTAG_NETZ_L_M.shp'
df = gpd.read_file(path, engine='pyogrio')


#store original data for visualization later
df['visual_geom'] = df['geometry']

#simplify the original graph to straight lines between nodes
def simplify_to_endpoints(geom):
    if geom is None or geom.is_empty:
        return None

    # If it's a MultiLineString, we need to grab the start of the
    # first part and the end of the last part.
    if isinstance(geom, MultiLineString):
        first_part = geom.geoms[0]
        last_part = geom.geoms[-1]
        start_point = first_part.coords[0]
        end_point = last_part.coords[-1]
    else:
        # Standard LineString behavior
        coords = list(geom.coords)
        start_point = coords[0]
        end_point = coords[-1]

    return LineString([start_point, end_point])


# Apply the updated function
df['geometry'] = df['geometry'].apply(simplify_to_endpoints)

#convert to primal graph
H = momepy.gdf_to_nx(df, approach='primal')

#convert momepy graph H to GeoDataFrames
gdf_h_nodes, gdf_h_edges = momepy.nx_to_gdf(H, points=True)

#export
df_export = gdf_h_edges.copy()

# Ensure we keep the visual_geom if it survived the momepy conversion
# If momepy dropped it, you can join it back using the index
if 'visual_geom' in df_export.columns:
    df_export['visual_geom_wkt'] = df_export['visual_geom'].apply(lambda x: x.wkt if x else None)

df_export['geometry_wkt'] = df_export['geometry'].apply(lambda x: x.wkt)

# Export to CSV
df_export.drop(columns=['geometry', 'visual_geom'], errors='ignore').to_csv("GIS_ALLTAG_SIMPLIFIED.csv", index=False)

print(f"Successfully exported {len(df_export)} simplified rows.")

# Plot
fig, ax = plt.subplots(figsize=(12,12))

#plot edges
gdf_h_edges.plot(ax=ax, color='#1fb5ad', linewidth=1, alpha=0.7, label='Edges')

#plot nodes
gdf_h_nodes.plot(ax=ax, color='#ff5733', markersize=5, zorder=3, label='Nodes')

ax.set_facecolor('#111111') # Dark background makes the graph pop
ax.set_title("Simplified Graph: Nodes & Edges", color='white', fontsize=15)
ax.axis('off') # Hide the lat/lon coordinates for a cleaner look
plt.legend()
plt.show()


#store graph as .gpkg
#Define your output path
gpkg_path = "data/converted/transport_network.gpkg"

#Save Edges to the GPKG
gdf_h_edges.to_file(gpkg_path, layer='edges', driver="GPKG")

#Save Nodes to the SAME GPKG (using the 'layer' argument)
gdf_h_nodes.to_file(gpkg_path, layer='nodes', driver="GPKG")

print(f"Graph successfully stored in {gpkg_path} with layers 'edges' and 'nodes'.")

#osmnx data import
# Dübendorf - Hinwil corridor
#left = 8.5800    # West (slightly west of Dübendorf)
#bottom = 47.2700  # South (below Hinwil/Aathal)
#right = 8.8700   # East (slightly east of Hinwil)
#top = 47.4200    # North (above Dübendorf/Volketswil)

#import network from OSMnx
#bbox = ([left, bottom, right, top])
#G = osmnx.graph.graph_from_bbox(bbox, network_type='bike', simplify=True, retain_all=True, truncate_by_edge=True, custom_filter=None)

#covert OSMnx graph H to GeoDataFrames
#gdf_g_nodes, gdf_g_edges = osmnx.graph_to_gdfs(G)

#align coordinate system
#gdf_g_edges = gdf_g_edges.to_crs(epsg=2056)
#gdf_g_nodes = gdf_g_nodes.to_crs(epsg=2056)
#gdf_h_edges = gdf_h_edges.to_crs(epsg=2056)
#gdf_h_nodes = gdf_h_nodes.to_crs(epsg=2056)

#check if they even exist in the same space
#print(f"OSM Edges Bounds: {gdf_g_edges.total_bounds}")
#print(f"GIS Data Bounds: {gdf_h_edges.total_bounds}")

#filter network
# Find GIS edges within 2 meters of an OSM edge
#gdf_g_buffered = gdf_g_edges.copy()
#gdf_g_buffered['geometry'] = gdf_g_buffered.geometry.buffer(2)
#res_intersection_edges = gpd.sjoin(gdf_h_edges, gdf_g_buffered, how="inner", predicate="intersects")

#get nodes
#start_nodes = res_intersection_edges['node_start'].unique()
#end_nodes = res_intersection_edges['node_end'].unique()
#all_intersected_node_ids = set(start_nodes) | set(end_nodes)

#Filter your node GeoDataFrame using those IDs
#res_intersection_nodes = gdf_h_nodes[gdf_h_nodes.index.isin(all_intersected_node_ids)]


#plot intersection
#if res_intersection_nodes.empty or res_intersection_edges.empty:
    #print("Warning: Intersection is still empty. Plotting both layers to see the gap.")
    #fig, ax = plt.subplots(figsize=(10, 10))
    #gdf_g_edges.plot(ax=ax, color='red', alpha=0.5, label='OSM')
    #gdf_h_edges.plot(ax=ax, color='blue', alpha=0.5, label='Local GIS')
    #plt.legend()
#else:
    #fig, ax = plt.subplots(figsize=(10, 10))
    #ax.set_facecolor('#111111')
    # Plot Edges (Lines)
    #res_intersection_edges.plot(ax=ax, color='#1fb5ad', linewidth=1.5, label='Edges')
    # Plot Nodes (Endpoints)
    #res_intersection_nodes.plot(ax=ax, color='white', markersize=5, zorder=3, label='Nodes')
    #plt.legend()
    #plt.show()

