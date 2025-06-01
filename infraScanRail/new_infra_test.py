from generate_infrastructure import get_missing_connections, generate_new_railway_lines, \
    export_new_railway_lines, prepare_Graph
from scoring import *
from plots import plot_graph, plot_lines_for_each_missing_connection

os.chdir(paths.MAIN)

df_network = gpd.read_file(settings.infra_generation_rail_network)
df_points = gpd.read_file(r'data\Network\processed\points.gpkg')
G, pos = prepare_Graph(df_network, df_points)

# Analyze the railway network to find missing connections
print("\n=== RAILWAY NETWORK ANALYSIS ===")
print("Identifying missing connections...")
missing_connections = get_missing_connections(G, pos, print_results=True, polygon=settings.perimeter_infra_generation)
plot_graph(G, pos, highlight_centers=True, missing_links=missing_connections, directory=paths.PLOT_DIRECTORY, polygon=settings.perimeter_infra_generation)

# Generate potential new railway lines
print("\n=== GENERATING NEW RAILWAY LINES ===")
new_railway_lines = generate_new_railway_lines(G, missing_connections)

# Print detailed information about the new lines
print("\n=== NEW RAILWAY LINES DETAILS ===")
print_new_railway_lines(new_railway_lines)

# Export to GeoPackage for further analysis and visualization in GIS software
export_new_railway_lines(new_railway_lines, pos, "data/Network/processed/new_railway_lines.gpkg")
print("\nNew railway lines exported to data/Network/processed/new_railway_lines.gpkg")

# Visualize the new railway lines on the network graph
print("\n=== VISUALIZATION ===")
print("Creating visualization of the network with highlighted missing connections...")

# Create a directory for individual connection plots if it doesn't exist

plots_dir = "plots/missing_connections"
plot_lines_for_each_missing_connection(new_railway_lines, G, pos, plots_dir)

print("\nAnalysis and visualization complete!")
print(f"Individual connection plots saved to {plots_dir}/")
