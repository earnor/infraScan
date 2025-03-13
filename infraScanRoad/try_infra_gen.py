import os

os.environ["USE_PYGEOS"] = "0"
import sys

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Set working directory
os.chdir(os.path.dirname(sys.argv[0]))

import generate_infrastructure


# ------------------------- set variables -------------------------

e_min, e_max = 2687000, 2708000  # 2688000, 2704000 - 2688000, 2705000
n_min, n_max = 1237000, 1254000  # 1238000, 1252000 - 1237000, 1252000
limits_corridor = [e_min, n_min, e_max, n_max]

# ------------------------- Look at structure -------------------------

scenario_shp = gpd.read_file(r"data\temp\data_scenario_n.shp")
print(scenario_shp.head())
print(scenario_shp.columns)


# ------------------------- load network -------------------------
network = gpd.read_file(r"data/temp/network_highway.gpkg")

print("Network loaded")
print(network.head())
print(network.columns)

# clip network
network = network.cx[e_min:e_max, n_min:n_max]

# Load new edges
new_edges_new_access = gpd.read_file(
    r"data\Network\processed\new_links_realistic_new_access.gpkg"
)
new_edges_new_connections = gpd.read_file(
    r"data\Network/processed/new_links_realistic_new_connections.gpkg"
)
new_edges = gpd.read_file(r"data\Network/processed/new_links_realistic.gpkg")

print("New edges loaded")
print(new_edges_new_access.head())
print(new_edges_new_access.columns)

# Plot 10x10 size
fig, ax = plt.subplots(figsize=(10, 10))
network.plot(ax=ax, color="grey", linewidth=2)
new_edges_new_access.plot(ax=ax, color="black", linewidth=2)
new_edges_new_connections.plot(ax=ax, color="black", linewidth=2)
new_edges.plot(ax=ax, color="red", linestyle="dotted", linewidth=1)
plt.show()

# -------------------------  Voronoi development shp -------------------------

# Load voronoi
voronoi = gpd.read_file("data/Voronoi/voronoi_developments_tt_values.shp")
# select ID_development first unique value
voronoi = voronoi.loc[voronoi["ID_develop"] == voronoi["ID_develop"].unique()[-1]]


# print
print("Voronoi loaded")
print(voronoi.head())
print(voronoi.columns)

# plot
fig, ax = plt.subplots(figsize=(10, 10))
network.plot(ax=ax, color="grey", linewidth=2)
new_edges.plot(ax=ax, color="red", linestyle="dotted", linewidth=2)
voronoi.plot(ax=ax, color="grey", alpha=0.5, edgecolor="black", linewidth=0.5)
plt.show()
