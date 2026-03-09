import geopandas as gpd
import momepy
import networkx as nx

gpd.options.io_engine = "fiona"
try:
    gdf = gpd.read_file("/Users/ruki/PycharmProjects/infraScan/infraScanCycle/data/raw/ALLTAG/OGD_VELO_ALLTAG_NETZ_L_M.shp")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# 2. Convert to NetworkX MultiGraph
# This generates a 'primal' graph where lines are edges and junctions are nodes
G = momepy.gdf_to_nx(gdf, approach='primal')

# 3. Extract nodes and edges back to GeoDataFrames
# momepy.nx_to_gdf returns a tuple: (nodes_gdf, edges_gdf)
nodes, edges = momepy.nx_to_gdf(G)

# 4. Set the CRS (Coordinate Reference System)
# It's vital to keep the spatial context for the GeoPackage
nodes.set_crs(gdf.crs, allow_override=True, inplace=True)
edges.set_crs(gdf.crs, allow_override=True, inplace=True)

# 5. Save to GeoPackage
output_file = "velo_network.gpkg"

# Use engine="fiona" here as well if pyogrio continues to fail
edges.to_file(output_file, layer='edges', driver="GPKG", engine="fiona")
nodes.to_file(output_file, layer='nodes', driver="GPKG", engine="fiona")

print(f"Success! Saved {len(nodes)} nodes and {len(edges)} edges to {output_file}")