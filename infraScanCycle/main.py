# import packages
import os
import math
import time
import pandas as pd
import geopandas as gpd
import sys
import ast # Added for the elevation profile fix
from plot_network import*
from data_import import *
from voronoi_tiling import *
from scenarios import *
from plots import *
from generate_infrastructure import *
from scoring import *
from OSM_network import *
from traveltime_delay import *
from data_converter import *
import warnings
import tracemalloc
warnings.filterwarnings("ignore", message="driver ESRI Shapefile does not support open option CRS")


def _mem():
    current, peak = tracemalloc.get_traced_memory()
    print(f"[MEM] current={current/1e6:.1f} MB  peak={peak/1e6:.1f} MB")



def print_hi(name):
    os.chdir(r'/Users/ruki/PycharmProjects/infraScan/infraScanCycle')
    #os.chdir(r'/Users/ninablattler/PycharmProjects/infraScan/infraScanCycle')
    sys.setrecursionlimit(2000)
    tracemalloc.start()
    runtimes = {}

    ##################################################################################
    # Initializing global variables
    print("\nINITIALIZE VARIABLES \n")
    st = time.time()

    # Filter which development type to score:
    #   None            → all (Netzlücken + Schwachstellen)
    #   'netzluecke'    → only Netzlücken
    #   'schwachstelle' → only Schwachstellen
    DEV_TYPE_FILTER = 'netzluecke'
    # Set True to skip the graph-based TTS section (slow) during quick test runs
    SKIP_TT = False
    # Minimum distance (m) between kept access points — 0 = keep all nodes.
    # Intersections are always kept; non-intersection nodes closer than this
    # to any already-kept node are dropped.  Try 300–500 to reduce OD pairs ~4×.
    ACCESS_POINT_MIN_DIST = 400

    # Define spatial limits of the research corridor
    # The coordinates must end with 000 in order to match the coordinates of the input raster data
    e_min, e_max = 2687000, 2708000     # 2688000, 2704000 - 2688000, 2705000
    n_min, n_max = 1237000, 1254000     # 1238000, 1252000 - 1237000, 1252000
    limits_corridor = [e_min, n_min, e_max, n_max]

    # Boundary for plot
    boundary_plot = polygon_from_points(e_min=e_min+1000, e_max=e_max-500, n_min=n_min+1000, n_max=n_max-2000)

    # Get a polygon as limits for the corridor
    innerboundary = polygon_from_points(e_min=e_min, e_max=e_max, n_min=n_min, n_max=n_max)

    # For global operation a margin is added to the boundary
    margin = 3000 # meters TODO: change margin
    outerboundary = polygon_from_points(e_min=e_min, e_max=e_max, n_min=n_min, n_max=n_max, margin=margin)

    # Define the size of the resolution of the raster to 25-50 meter, before 100 was too coarse
    raster_size = 50 # meters

    ##################################################################################
    # Define variables for monetisation

    # ---------------------------------------------------------------------------------
    # PLACEHOLDER VALUES — all costs below must be updated with cycling-specific values
    # Sources to consult: ARE Wegleitung Kosten-Nutzen-Analyse, VSS norms for cycling
    # ---------------------------------------------------------------------------------

    # Construction costs [CHF/m]
    c_cycle_path_new    = 1000   # new path (Netzlücke)    TODO: calibrate
    c_cycle_path_update = 300    # upgrade (Schwachstelle)  TODO: calibrate

    # Maintenance costs
    c_om_cycle_path    = 100     # operational maintenance [CHF/m/year]  TODO: calibrate
    c_structural_maint = 1.2/100 # structural maintenance [fraction of construction cost/year]

    # Value of Travel Time Savings [CHF/h]
    VTTS = 28  # CHF/h — Swiss official value ~28 CHF/h for leisure cycling
    travel_time_duration = 50  # appraisal horizon [years]

    # Route comfort monetisation [CHF / m / year per CLI unit]
    comfort_value_chf_m_year = 2.0  # TODO: calibrate

    # Safety — willingness-to-pay to avoid risk-weighted route exposure [CHF/(risk_unit·trip·year)]
    value_of_safety = None  # TODO: set once a unit value is agreed (e.g. 0.0001)

    runtimes["Initialize variables"] = time.time() - st
    st = time.time()

    ##################################################################################
    # Import and prepare raw data
    print("\nIMPORT RAW DATA \n")

    # Import shapes of lake for plots
    get_lake_data() #ok

    # Import the file containing the locations to be plotted
    import_locations() #ok

    # Define area that is protected for constructing cycling paths
    get_protected_area(limits=limits_corridor)
    get_unproductive_area(limits=limits_corridor)
    landuse(limits=limits_corridor)

    # Tif file of all unsuitable land cover and protected areas
    # File is stored to 'data/landuse_landcover/processed/zone_no_infra/protected_area_{suffix}.tif'
    all_protected_area_to_raster(suffix="corridor")

    runtimes["Import land use and land cover data"] = time.time() - st
    _mem()
    st = time.time()

    ##################################################################################
    ##################################################################################
    # INFRASTRUCTURE NETWORK
    # 1) Import network
    # 2) Process network
    # 3) Generate developments (new access points) and connection to existing infrastructure

    print("\nINFRASTRUCTURE NETWORK \n")
    ##################################################################################
    # 1) Import network
    # Import the cycling network and preprocess it ALLTAG.GIS
    # higher network layer (like highway for cycling) stored in infraScanCycle/data/raw/ALLTAG/Velonetz_Alltag_-OGD.gpkg
    # gaps in higher level network stored in infraScanCycle/data/raw/SCHWACHSTELLEN/TBA_VNP_SCHWACHSTELLEN_L.shp
    # make option to join them together

    network = import_network_GIS_ALLTAG()  # already EPSG:2056, already clean 29.04.26


    #plot network (can comment out)
    # plot_network(network)

    # CLEAN DATA
    network = network[network.geometry.notnull()]


    runtimes["Import network data"] = time.time() - st
    st = time.time()


    ##################################################################################
    # 2) Process network

    # Simplify the physical topology of the network
    # One distinct edge between two nodes (currently multiple edges between nodes)
    # Edges are stored in r"data/Network/processed/edges.gpkg"
    # Points in simplified network can be intersections ("intersection"==1) or access points ("intersection"==0)
    # Points are stored in r"data/Network/processed/points.gpkg"
    
    nodes_gdf, edges_final = reformat_network()

    # plot network (can comment out)
    #plot_network(edges_final)

    # Filter the infrastructure elements that lie within a given polygon
    # Points within the corridor are stored in r"data/Network/processed/points_corridor.gpkg"
    # Edges within the corridor are stored in r"data/Network/processed/edges_corridor.gpkg"
    # Edges crossing the corridor border are stored in r"data/Network/processed/edges_on_corridor.gpkg"
    points_corridor, edges_corridor, edges_border = network_in_corridor(
        polygon=outerboundary, access_point_min_dist=ACCESS_POINT_MIN_DIST)
    #plot network corridor
    #plot_corridor_network(outerboundary, points_corridor, edges_corridor, edges_border, points_full=nodes_gdf, edges_full=edges_final)

    # Add attributes to the edges
    edges = get_edge_attributes() #TODO
    # plot
    #plot_edge_attributes(edges)

    runtimes["Preprocess the network"] = time.time() - st
    st = time.time()

    # --- NETWORK QUALITY CHECK ---
    print("\n--- NETWORK QUALITY CHECK ---")
    print(edges['ROUTENTYP'].value_counts())
    print(
        f"Edges missing ROUTENTYP:       {edges['ROUTENTYP'].isna().sum() if 'ROUTENTYP' in edges.columns else 'col missing'}")
    print(f"One-way edges:                 {edges['oneway'].sum()}")
    print(
        f"Corridor edge coverage:        {len(edges_corridor)}/{len(edges)} ({100 * len(edges_corridor) / len(edges):.0f}%)")
    print(f"Total nodes:                   {len(nodes_gdf)}")
    print(f"Corridor nodes:                {len(points_corridor)}")
    print(
        f"Intersections in corridor:     {points_corridor['is_intersection'].sum() if 'is_intersection' in points_corridor.columns else 'col missing'}")
    print(
        f"Endpoints in corridor:         {points_corridor['is_endpoint'].sum() if 'is_endpoint' in points_corridor.columns else 'col missing'}")
    print("-----------------------------\n")


    """
    ##################################################################################
    # 3a) OPTION A: Generate developments (new access points) and connection to existing infrastructure

    # Make random points within the perimeter (extent) and filter them
    num_rand = 1000 #Todo: change to 1000
    random_gdf = generated_access_points(extent=innerboundary, number=num_rand)


    # Assign the return value of the function to the variable name
    filtered_gdf = filter_access_points(random_gdf)
    filtered_gdf.to_file(r"data/Network/processed/generated_nodes.gpkg")

    generated_points = filtered_gdf

    # Import current points as dataframe (Note: all nodes are now access points)
    current_access_points = gpd.read_file(r"data/Network/processed/access_points_corridor.gpkg")

    # Filter to corridor nodes only (spatial join with corridor polygon)
    poly_gdf = gpd.GeoDataFrame({'geometry': [outerboundary]}, crs="EPSG:2056")
    current_access_points = gpd.sjoin(
        current_access_points, poly_gdf, how='inner', predicate='within'
    ).drop(columns=['index_right'], errors='ignore').reset_index(drop=True)

    print(f"  Current access points in corridor: {len(current_access_points)}")
    print(f"  Columns: {current_access_points.columns.tolist()}")

    # Connect generated points to existing network nodes
    edges_gdf = edges
    new_links = connect_points_to_network(generated_points, current_access_points, edges_gdf=edges_gdf)

    # plot
    plot_connections(generated_points, current_access_points, new_links, polygon=outerboundary)

    # Filter by max realistic cycling link distance
    min_link_dist = 200  # don't connect points already very close to network
    max_link_dist = 10000

    new_links_realistic = new_links[
        (new_links['dist_to_node'] >= min_link_dist) &
        (new_links['dist_to_node'] <= max_link_dist)
        ].copy().reset_index(drop=True)

    print(f"  -> {len(new_links_realistic)} links within {max_link_dist}m threshold "
          f"(dropped {len(new_links) - len(new_links_realistic)})")

    new_links_realistic.to_file('data/Network/processed/new_links_corridor.gpkg', driver='GPKG')


    # Find a routing for the generated links that considers protected areas
    raster = r'data/landuse_landcover/processed/zone_no_infra/protected_area_corridor.tif'
    routing_raster(raster_path=raster)

    routed = gpd.read_file('data/Network/processed/new_links_realistic.gpkg')
    max_routed_length = 3000  # routed path shouldn't be more than 2x the straight-line max
    routed = routed[routed['length_routed_m'] <= max_routed_length].copy().reset_index(drop=True)
    routed.to_file('data/Network/processed/new_links_realistic.gpkg', driver='GPKG')
    print(f"  After routed length filter: {len(routed)} links")
    # ── Slope filter (cycling max 8%) ────────────────────────────────────
    max_slope_pct = 8.0
    elevation_raster_path = r'data/elevation_model/elevation.tif'

    with rasterio.open(elevation_raster_path) as elev_src:
        elev_data = elev_src.read(1)

        def _max_slope(geom, src=elev_src, data=elev_data, interval=50):
            n = max(2, int(geom.length / interval) + 1)
            pts = [geom.interpolate(d) for d in np.linspace(0, geom.length, n)]
            z = [data[src.index(p.x, p.y)] for p in pts]
            return float(np.max(np.abs(np.diff(z)) / interval * 100))

        routed['max_slope_pct'] = routed.geometry.apply(_max_slope)

    before_slope = len(routed)
    routed = routed[routed['max_slope_pct'] <= max_slope_pct].copy().reset_index(drop=True)
    print(f"  After slope filter (max {max_slope_pct}%): {len(routed)} links "
          f"(dropped {before_slope - len(routed)})")
    routed.to_file('data/Network/processed/new_links_realistic.gpkg', driver='GPKG')

    # Keep only generated points that have a surviving link
    connected_ids = set(routed['ID_new'].unique())
    generated_points = generated_points[
        generated_points['ID_new'].isin(connected_ids)
    ].copy().reset_index(drop=True)

    print(f"  Generated points with valid links: {len(generated_points)} "
          f"(dropped {len(filtered_gdf) - len(generated_points)})")
    generated_points.to_file(r"data/Network/processed/generated_nodes.gpkg", driver='GPKG')
    # plot
    plot_connections(generated_points, current_access_points, routed, polygon=outerboundary)

    # ─────────────────────────────────────────────────────────────────────
    # Compute the Voronoi polygons for status quo
    voronoi_sq = get_voronoi_status_quo(corridor_polygon=innerboundary)
    # plot
    plot_voronoi_status_quo(voronoi_sq, nodes_gdf, edges_final, corridor_polygon=innerboundary)
    limits_variables = [2680600, 1227700, 2724300, 1265600]


    runtimes["Generate infrastructure developments"] = time.time() - st
    st = time.time()

    #combines generated infrastructure with current network (comment out)
    
    #combined_nodes, combined_edges = build_combined_network(
        #nodes_gdf=nodes_gdf,
        #edges_gdf=edges_gdf,  # the attributed edges from get_edge_attributes()
        #generated_points=generated_points,
        #routed_links=routed  # the filtered routed links
    #)

    #plot_combined_vs_status_quo(
        #nodes_gdf, edges_gdf,
        #combined_nodes, combined_edges,
        #corridor_polygon=outerboundary
    #)
    
    import_data(limits_variables)
    runtimes["Import variable for scenario (population and employment)"] = time.time() - st
    st = time.time()

    """

    ##################################################################################
    # 3b) OPTION B: Generate Developments based on the weak points and gaps in the
    # ALLTAG network
    # Schwachstellen  — existing edges with quality issues (is_schwachstelle == 1)
    # Netzlücken      — planned but unbuilt edges          (is_development   == 1)

    developments = get_development_candidates(edges, corridor_polygon=innerboundary,
                                              dev_type_filter=DEV_TYPE_FILTER)

    runtimes["Generate infrastructure developments"] = time.time() - st
    st = time.time()

    #plot_developments(edges, corridor_polygon=outerboundary)

    # Voronoi polygons for status quo (needed by scenario aggregation below)
    voronoi_sq = get_voronoi_status_quo(corridor_polygon=innerboundary) #TODO: check points
    #plot_voronoi_status_quo(voronoi_sq, nodes_gdf, edges_final, corridor_polygon=innerboundary)

    # Wider bounding box for scenario rasters and scoring (covers full Voronoi catchment)
    limits_variables = [2680600, 1227700, 2724300, 1265600]

    import_data(limits_variables) #todo: move up the pipline or change
    runtimes["Import variable for scenario (population and employment)"] = time.time() - st
    _mem()
    st = time.time()

    ##################################################################################


    ##################################################################################
    ##################################################################################
    # SCENARIO
    print("\nSCENARIO \n")
    ##################################################################################
    # 1) Define scenario based on cantonal predictions
    # Import the predicted scenario defined by the canton of Zürich
    scenario_zh = pd.read_csv(r"data/Scenario/KTZH_00000705_00001741.csv", sep=";")

    # Define the relative growth per scenario and district
    # The growth rates are stored in "data/temp/data_scenario_n.shp"
    future_scenario_zuerich_2022(scenario_zh)
    # Plot the growth rates as computed above for population and employment and over three scenarios


    # Compute the predicted amount of population and employment in each raster cell (hectar) for each scenario
    # The resulting raster data are stored in "data/independent_variables/scenario/{col}.tif" with col being pop or empl and the scenario
    scenario_to_raster(limits_variables)

    # Aggregate the scenario data to over the voronoi polygons, here euclidian polygons
    # Store the resulting file to "data/Voronoi/voronoi_developments_euclidian_values.shp"
    polygons_gdf = gpd.read_file(r"data/Voronoi/voronoi_status_quo_euclidian.gpkg")
    scenario_to_voronoi(polygons_gdf, euclidean=True)

    # plot
    # plot_scenarios(corridor_polygon=outerboundary)


    # Convert multiple tif files to one same tif with multiple bands
    stack_tif_files(var="empl")
    stack_tif_files(var="pop")
    runtimes["Generate the scenarios"] = time.time() - st
    _mem()
    st = time.time()


    ##################################################################################
    ##################################################################################
    print("\nIMPLEMENT SCORING \n")

    ##################################################################################
    # 0) Load developments and network files
    developments = pd.read_csv('data/Network/processed/developments_list.csv')
    edges_sq  = gpd.read_file('data/Network/processed/edges_with_attribute.gpkg')
    nodes_sq  = gpd.read_file('data/Network/processed/points_with_attribute.gpkg')
    dev_nodes = gpd.read_file('data/Network/processed/generated_nodes.gpkg')
    dev_nodes = dev_nodes[dev_nodes["within_corridor"] | dev_nodes["on_border"]]
    runtimes["Load scoring inputs"] = time.time() - st
    st = time.time()

    ##################################################################################
    # 1) Raster-based travel time and OD matrices
    make_cycling_speed_raster(cycling_speed_kmh=15)
    travel_cost_polygon(limits_variables)
    voronoi_sq = gpd.read_file(r"data/Network/travel_time/Voronoi_statusquo.gpkg")
    GetCyclingOD(voronoi_gdf=voronoi_sq)
    accessib_sq = accessibility_status_quo(VTT_h=VTTS, duration=travel_time_duration)

    travel_cost_developments(limits_variables)
    single_tt_voronoi_ton_one("data/Network/travel_time/developments")
    polygon_gdf = gpd.read_file(r"data/Voronoi/combined_developments.gpkg")
    scenario_to_voronoi(polygon_gdf, euclidean=False)
    GetVoronoiOD_multi()
    accessib_devs = accessibility_developments(accessib_sq, VTT_h=VTTS, duration=travel_time_duration)
    runtimes["OD matrices and accessibility"] = time.time() - st
    _mem()
    st = time.time()
    """
    ##################################################################################
    #old
    # 2) Graph-based travel time savings → data/costs/traveltime_savings.csv
    if not SKIP_TT:
        tt_optimization_status_quo()
        tt_optimization_all_developments()
        monetize_tts(VTTS=VTTS, duration=travel_time_duration)
    runtimes["Travel time savings (graph)"] = time.time() - st
    _mem()
    st = time.time()
    """
    ##################################################################################
    # 3) Protected area for scoring perimeter (wider than corridor, needed by externalities)
    get_protected_area(limits=limits_variables)
    get_unproductive_area(limits=limits_variables)
    landuse(limits=limits_variables)
    all_protected_area_to_raster(suffix="variables")
    runtimes["Protected area (variables perimeter)"] = time.time() - st
    st = time.time()

    ##################################################################################
    # 4) Construction and maintenance costs
    #    Netzlücken: full build at c_cycle_path_new [CHF/m]
    #    Schwachstellen: upgrade at c_cycle_path_update [CHF/m]
    construction_costs(
        cycle_path=c_cycle_path_new,
        upgrade_factor=c_cycle_path_update / c_cycle_path_new,
    )
    maintenance_costs(
        duration=travel_time_duration,
        cycle_path=c_om_cycle_path,
        structural=c_structural_maint,
    )
    runtimes["Construction and maintenance costs"] = time.time() - st
    st = time.time()

    ##################################################################################
    # 5) Route comfort
    #    CLI per ROUTENTYP (Velobahn=1.0 … Nebenverbindung=0.5, baseline=0.1)
    #    Slope penalty: <4% good (×1.0), 4–8% hard (×0.6), >8% very bad (×0.3)
    route_comfort(duration=travel_time_duration,
                  comfort_value_chf_m_year=comfort_value_chf_m_year)
    runtimes["Route comfort"] = time.time() - st
    st = time.time()

    ##################################################################################
    # 6) Safety benefits
    #    Risk weights per ROUTENTYP; new infrastructure gets lower risk weight. 
    #    Set value_of_safety (CHF per risk_unit·trip·year) to monetise.
    safety_benefits(value_of_safety=value_of_safety, duration=travel_time_duration)
    runtimes["Safety benefits"] = time.time() - st
    st = time.time()

    ##################################################################################
    # 7) Net benefits: NB = C + M + T + R + S  per scenario
    nb_df = net_benefits()
    runtimes["Net benefits"] = time.time() - st
    st = time.time()

    ##################################################################################
    # VISUALIZE THE RESULTS
    print("\nVISUALIZE THE RESULTS \n")
    os.makedirs("plot/results", exist_ok=True)

    tif_path_plot = r"data/landuse_landcover/processed/zone_no_infra/protected_area_corridor.tif"
    network       = gpd.read_file(r"data/Network/processed/edges_with_attribute.gpkg")
    access_points = gpd.read_file(r"data/Network/processed/points_corridor_attribute.gpkg")

    gdf_nb_raw = gpd.read_file(r"data/costs/net_benefits.gpkg")
    money_cols = ["C", "M", "T_s1", "T_s2", "T_s3", "R",
                  "S_s1", "S_s2", "S_s3", "NB_s1", "NB_s2", "NB_s3"]
    gdf_nb = gdf_nb_raw.copy()
    for col in money_cols:
        gdf_nb[col] = gdf_nb[col] / 1e6
    gdf_nb["total_low"]    = gdf_nb["NB_s1"]
    gdf_nb["total_medium"] = gdf_nb["NB_s2"]
    gdf_nb["total_high"]   = gdf_nb["NB_s3"]

    for scen_col, scen_label, plot_tag in [
        ("NB_s1", "low growth (s1)",    "nb_map_s1"),
        ("NB_s2", "medium growth (s2)", "nb_map_s2"),
        ("NB_s3", "high growth (s3)",   "nb_map_s3"),
    ]:
        plot_cost_result(
            df_costs=gdf_nb.copy(), banned_area=tif_path_plot,
            title_bar=f"cycling net benefit — {scen_label}",
            boundary=boundary_plot, network=network,
            access_points=access_points, plot_name=plot_tag, col=scen_col,
        )

    for comp_col, comp_label, plot_tag in [
        ("C",    "construction cost",     "comp_construction"),
        ("M",    "maintenance cost",      "comp_maintenance"),
        ("T_s2", "travel time savings",   "comp_traveltime"),
        ("R",    "route comfort benefit", "comp_comfort"),
        ("S_s2", "safety benefit",        "comp_safety"),
    ]:
        plot_single_cost_result(
            df_costs=gdf_nb_raw.copy(), banned_area=tif_path_plot,
            title_bar=comp_label, boundary=boundary_plot, network=network,
            access_points=access_points, plot_name=plot_tag, col=comp_col,
        )

    gdf_nb["mean_costs"] = gdf_nb[["NB_s1", "NB_s2", "NB_s3"]].mean(axis=1)
    gdf_nb["std"]        = gdf_nb[["NB_s1", "NB_s2", "NB_s3"]].std(axis=1)
    gdf_nb["cv"]         = (gdf_nb["std"] / gdf_nb["mean_costs"].abs()
                            ).replace([float("inf"), float("-inf")], 0).fillna(0) * 1e4
    plot_cost_uncertainty(
        df_costs=gdf_nb.copy(), banned_area=tif_path_plot, boundary=boundary_plot,
        network=network, col="std",
        legend_title="Std. dev. across\nscenarios [Mio. CHF]",
        access_points=access_points, plot_name="nb_uncertainty",
    )

    boxplot(gdf_nb, nbr=15)

    plot_benefit_distribution_line_multi(
        df_costs=gdf_nb.copy(),
        columns=["NB_s1", "NB_s2", "NB_s3"],
        labels=["low growth", "medium growth", "high growth"],
        plot_name="nb_all_scenarios", legend_title="Growth scenario",
    )

    gdf_nb_comp = gdf_nb.copy()
    for c in ["C", "M", "T_s2", "R", "S_s2"]:
        gdf_nb_comp[c] = gdf_nb_comp[c].astype(int)
    plot_benefit_distribution_line_multi(
        df_costs=gdf_nb_comp,
        columns=["C", "M", "T_s2", "R", "S_s2"],
        labels=["Construction (C)", "Maintenance (M)",
                "Travel time savings (T)", "Route comfort (R)", "Safety (S)"],
        plot_name="nb_components",
        legend_title="NB component\n(medium scenario)",
    )
    runtimes["Visualization"] = time.time() - st


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('You did a good job ;)')