import geopandas as gpd
from tabulate import tabulate

# Read the network dataset
network = gpd.read_file(r"data/temp/network_railway-services.gpkg")

# Print the DataFrame as a table
print(tabulate(network.head(), headers='keys', tablefmt='psql'))  # 'psql' gives a nice table format
