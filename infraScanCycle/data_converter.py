import momepy
import osmnx
import geopandas as gpd
import matplotlib.pyplot as plt



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

#check if they even exist in the same space
print(f"GIS Data Bounds: {gdf_h_edges.total_bounds}")


#use small buffer to catch near-overlaps
gdf_h_buffered = gdf_h_edges.copy()
gdf_h_buffered['geometry'] = gdf_h_buffered.geometry.buffer(5)

#plot intersection
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_facecolor('#111111')
# Plot Edges (Lines)
gdf_h_edges.plot(ax=ax, color='#1fb5ad', linewidth=1.5, label='Edges')
plt.legend()
plt.show()