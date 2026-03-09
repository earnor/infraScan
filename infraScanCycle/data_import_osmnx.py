import osmnx
import geopandas as gpd
from fontTools.ttLib.woff2 import bboxFormat



left = 8.5796
bottom = 47.2876
right = 8.8461
top = 47.4543

bbox = ([left, bottom, right, top])
G = osmnx.graph.graph_from_bbox(bbox, network_type='bike', simplify=True, retain_all=True, truncate_by_edge=True, custom_filter=None)

osmnx.plot.plot_graph(G, ax=None, figsize=(8, 8), bgcolor='#111111', node_color='w', node_size=15, node_alpha=None, node_edgecolor='none', node_zorder=1, edge_color='#999999', edge_linewidth=1, edge_alpha=None, bbox=None, show=True, close=False, save=False, filepath=None, dpi=300)

