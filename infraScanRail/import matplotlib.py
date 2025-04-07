import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load your access points data
df_access = pd.read_csv(r"data/Network/Rail_Node.csv", sep=";", decimal=",", encoding="ISO-8859-1")

# Convert the DataFrame to a GeoDataFrame
# Assuming 'Easting' and 'Northing' are the column names for your coordinates
gdf_access = gpd.GeoDataFrame(
    df_access, 
    geometry=gpd.points_from_xy(df_access[' YKOORD'], df_access['XKOORD']),
    crs="EPSG:4326"  # Change to the appropriate CRS if needed
)

# Set plot limits
easting_limit = 1 * 1e6
northing_limit = 0.4 * 1e5

# Filter the GeoDataFrame
filtered_gdf = gdf_access[(gdf_access.geometry.x <= easting_limit) & (gdf_access.geometry.y <= northing_limit)]

# Plotting the access points
plt.figure(figsize=(10, 10))
base = filtered_gdf.plot(marker='o', color='blue', markersize=5, label='Access Points')
plt.xlim(0, easting_limit * 1.1)  # Extend the x-limits for better visualization
plt.ylim(0, northing_limit * 1.1)  # Extend the y-limits for better visualization
plt.title('Access Points')
plt.xlabel('Easting')
plt.ylabel('Northing')
plt.grid(True)
plt.legend()
plt.show()
