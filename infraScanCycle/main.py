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
    # TODO: hardcoded path — replace with pathlib.Path(__file__).parent or a
    # config variable so the script runs on any machine without editing.
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

    
    # Minimum distance (m) between kept access points
    # 0 = keep all nodes.
    # Intersections are always kept; non-intersection nodes closer than this
    # to any already-kept node are dropped.  Try 300–500 to reduce OD pairs ~4×.
    ACCESS_POINT_MIN_DIST = 0 #todo remove

    # Define spatial limits of the research corridor
    # The coordinates must end with 000 in order to match the coordinates of the input raster data
    e_min, e_max = 2687000, 2708000
    n_min, n_max = 1237000, 1254000
    limits_corridor = [e_min, n_min, e_max, n_max]


    # Get a polygon as limits for the corridor
    innerboundary = polygon_from_points(e_min=e_min, e_max=e_max, n_min=n_min, n_max=n_max)

    # For global operation a margin is added to the boundary
    margin = 3000 # meters TODO: change margin
    outerboundary = polygon_from_points(e_min=e_min, e_max=e_max, n_min=n_min, n_max=n_max, margin=margin)

    # Define the size of the resolution of the raster to 25-50 meter
    raster_size = 50 # meters

    ##################################################################################
    # Define variables for monetisation

    # Construction costs [CHF/m]
    c_cycle_path_new    = 1000   # new path (Netzlücke)    TODO: calibrate
    c_cycle_path_update = 300    # upgrade (Schwachstelle)  TODO: calibrate

    # Maintenance costs
    c_om_cycle_path    = 100     # operational maintenance [CHF/m/year]  TODO: calibrate
    c_structural_maint = 1.2/100 # structural maintenance [fraction of construction cost/year]

    # Value of Travel Time Savings [CHF/h]
    VTTS = 18.2  # CHF/h — Swiss official value ~18.2 CHF/h for leisure cycling
    travel_time_duration = 50  # appraisal horizon [years]

    # Safety — per-ROUTENTYP accident risk rates [accidents/km/trip-year]
    # Mirrors scoring.RISK_RATE; override here if you want run-specific values,
    # otherwise scoring.RISK_RATE constants are used directly.
    # ROUTENTYP of a newly built Netzlücke (sets its post-development risk rate):
    #   'Velobahn' = 0.10 | 'Veloschnellroute' = 0.15 | 'Hauptverbindung' = 0.30
    #   'Nebenverbindung' = 0.50 | 'Zusätzliche Freizeitverbindung' = 0.40
    # Monetary value per prevented accident [CHF] — scoring.VALUE_PER_ACCIDENT

    runtimes["Initialize variables"] = time.time() - st
    st = time.time()

    ##################################################################################
    # Import and prepare raw data check all good
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

    network = import_network_GIS_ALLTAG()

    # Plot 1 — raw shapefile import: edges coloured by ROUTENTYP, no node classification yet
    plot_raw_network(network, title="Step 1 — Raw imported network (ALLTAG shapefile)", save_path="data/Network/processed/plot_01_raw_import.png")

    runtimes["Import network data"] = time.time() - st
    st = time.time()

    ##################################################################################
    # 2) Process network

    nodes_gdf, edges_final = reformat_network()

    # Plot 2 — after topology resolution: intersections, dead ends, through-nodes classified
    plot_network_classified(nodes_gdf, edges_final)

    points_corridor, edges_corridor, edges_border = network_in_corridor(
        polygon=innerboundary, access_point_min_dist=ACCESS_POINT_MIN_DIST)

    # Plot 3 — corridor only, node IDs and type counts
    plot_corridor_network(innerboundary, points_corridor, edges_corridor, edges_border,
                          points_full=nodes_gdf, edges_full=edges_final)

    # Plot 4 — edge attributes: free-flow speed, average incline, travel time
    plot_edge_attributes(edges_corridor)


    ##################################################################################
    # A) Add Netzlücken to corridor graph at BAD_FFS
    # Netzlücken are planned-but-unbuilt edges (is_development == 1).  Including them
    # with a degraded speed (BAD_FFS = 15 km/h) makes the graph more connected
    # while signalling that these links are substandard in their current state.
    # They get their own ROUTENTYP so they are distinguishable from both existing
    # edges and auto-generated connectors in plots and routing.
    edges_aug = edges_corridor.copy()
    nl_mask   = edges_aug['is_development'] == 1
    edges_aug.loc[nl_mask, 'ROUTENTYP'] = 'Netzlücke'
    edges_aug.loc[nl_mask, 'ffs']       = BAD_FFS
    edges_aug.loc[nl_mask, 'tt_min']    = (
        edges_aug.loc[nl_mask, 'length_m'] / 1000) / BAD_FFS * 60
    print(f"\n  Added {nl_mask.sum()} Netzlücken at BAD_FFS ({BAD_FFS} km/h) to corridor graph")

    G_with_nl = build_graph_direct(edges_aug)
    conn_nl   = check_network_connectivity(G_with_nl, label="corridor + Netzlücken")

    # Keep only the largest connected component — drop isolated sub-graphs.
    # Note: build_graph_direct uses (round(x,1), round(y,1)) coordinate tuples
    # as node IDs, not ID_point integers.  Filter by matching edge geometry
    # endpoints against the coordinate-tuple node set.
    if not conn_nl['is_connected']:
        import networkx as nx
        largest_nodes = max(nx.connected_components(G_with_nl.to_undirected()),
                            key=len)

        def _coords_of(geom):
            c = list(geom.coords)
            return ((round(c[0][0], 1), round(c[0][1], 1)),
                    (round(c[-1][0], 1), round(c[-1][1], 1)))

        edge_mask = edges_aug.geometry.apply(
            lambda g: all(n in largest_nodes for n in _coords_of(g))
        )
        edges_aug = edges_aug[edge_mask].reset_index(drop=True)

        pt_mask = points_corridor.geometry.apply(
            lambda g: (round(g.x, 1), round(g.y, 1)) in largest_nodes
        )
        points_corridor = points_corridor[pt_mask].reset_index(drop=True)

        n_dropped = G_with_nl.number_of_nodes() - len(largest_nodes)
        print(f"  Kept largest component: {len(largest_nodes)} nodes, "
              f"{len(edges_aug)} edges  ({n_dropped} nodes dropped)")




    ##################################################################################
    # B) Plot all route types: existing / Schwachstellen / Netzlücken / Connectors
    # conn_gdf: empty since auto-connector section (B) is not active
    conn_gdf = gpd.GeoDataFrame(
        columns=['geometry', 'ROUTENTYP', 'ffs', 'tt_min', 'length_m'],
        geometry='geometry', crs="EPSG:2056"
    )
    plot_network_all_types(
        edges_gdf=edges_aug,
        conn_gdf=conn_gdf if len(conn_gdf) > 0 else None,
        corridor_polygon=innerboundary,
        save_path="data/Network/processed/network_all_types.png",
    )

    # Save the full annotated network (all route types) as GeoPackages for GIS reference.
    # Edges include ID_edge, ROUTENTYP, ffs, tt_min, is_development, is_schwachstelle.
    # Nodes include ID_point, is_intersection, is_through_point, is_endpoint.
    _full_edges = edges_aug.copy()
    _full_edges['ID_edge'] = range(len(_full_edges))
    _full_edges.to_file('data/Network/processed/network_full_annotated_edges.gpkg', driver='GPKG')
    nodes_gdf.to_file('data/Network/processed/network_full_annotated_nodes.gpkg', driver='GPKG')
    print(f"  Full annotated network saved: "
          f"{len(_full_edges)} edges, {len(nodes_gdf)} nodes "
          f"→ network_full_annotated_{{edges,nodes}}.gpkg")

    runtimes["Preprocess the network"] = time.time() - st
    st = time.time()

    # --- NETWORK QUALITY CHECK ---
    print("\n--- NETWORK QUALITY CHECK ---")
    print(edges_aug['ROUTENTYP'].value_counts() if 'ROUTENTYP' in edges_aug.columns else "ROUTENTYP column not found")
    print(f"Edges missing ROUTENTYP:       {edges_aug['ROUTENTYP'].isna().sum() if 'ROUTENTYP' in edges_aug.columns else 'col missing'}")
    print(f"Corridor edge coverage:        {len(edges_corridor)}/{len(edges_final)} ({100 * len(edges_corridor) / max(len(edges_final), 1):.0f}%)")
    print(f"Total nodes:                   {len(nodes_gdf)}")
    print(f"Corridor nodes:                {len(points_corridor)}")
    print(f"Intersections in corridor:     {points_corridor['is_intersection'].sum() if 'is_intersection' in points_corridor.columns else 'col missing'}")
    print(f"Endpoints in corridor:         {points_corridor['is_endpoint'].sum() if 'is_endpoint' in points_corridor.columns else 'col missing'}")
    print("-----------------------------\n")

    plot_network_quality(
        edges_aug=edges_aug,
        edges_corridor=edges_corridor,
        edges_final=edges_final,
        points_corridor=points_corridor,
        nodes_gdf=nodes_gdf,
        save_path="data/Network/processed/network_quality.png",
    )

    plot_network_graph(
        edges_aug=edges_aug,
        points_corridor=points_corridor,
        corridor_polygon=innerboundary,
        save_path="data/Network/processed/network_graph.png",
    )


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




    # Wider bounding box covering the full Voronoi catchment (larger than corridor)
    limits_variables = [2680600, 1227700, 2724300, 1265600]

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

    # (Re)generate Voronoi tessellation for all current corridor nodes.
    # Pass points_corridor directly so the Voronoi uses exactly the same
    # 404 nodes as the network — no spatial-predicate mismatch with disk file.
    voronoi_sq = get_voronoi_status_quo(
        corridor_polygon=innerboundary,
        nodes_gdf=points_corridor,
    )

    # Aggregate the scenario data over the voronoi polygons (euclidian).
    # Store the resulting file to "data/Voronoi/voronoi_developments_euclidian_values.shp"
    scenario_to_voronoi(voronoi_sq, euclidean=True)

    # plot
    plot_scenarios(corridor_polygon=innerboundary)


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


    ##################################################################################
    # 1) OD matrix — Voronoi-weighted cycling trips for each scenario
    #
    # A) Each node's Voronoi polygon already contains aggregated pop/empl per scenario
    #    (written by scenario_to_voronoi).  Load those values directly — no routing needed.
    # B) Origin count of node i  = population  in its Voronoi polygon (s{n}_pop)
    #    Dest   count of node j  = employment  in its Voronoi polygon (s{n}_empl)
    #    trips_ij_s = pop_i_s × (empl_j_s / Σ_j empl_j_s) × MODAL_SHARE
    # C) Save od_s1/s2/s3.csv and combined od_scenarios.csv; plot origins, dests, totals

    od_scenarios, voronoi_vals = od_cycling_weighted()

    plot_od_results(
        od_scenarios=od_scenarios,
        voronoi_vals=voronoi_vals,
        points_gdf=points_corridor,
        corridor_polygon=innerboundary,
        scenarios=['s1', 's2', 's3'],
        save_path='data/OD/od_plot.png',
    )

    runtimes["OD matrix (Voronoi-weighted)"] = time.time() - st
    _mem()
    st = time.time()


    ##################################################################################
    # 2) Node accessibility — Voronoi-catchment scoring + rasterization

    accessibility = node_accessibility(
        voronoi_path='data/Voronoi/voronoi_developments_euclidian_values.shp',
        raster_template='data/landuse_landcover/processed/zone_no_infra/protected_area_corridor.tif',
        beta=2.0,
        scenarios=['s1', 's2', 's3'],
    )

    plot_node_accessibility(
        accessibility_results=accessibility,
        corridor_polygon=innerboundary,
        scenarios=['s1', 's2', 's3'],
        save_path='data/Network/accessibility/accessibility_plot.png',
    )

    runtimes["Node accessibility (Voronoi-based)"] = time.time() - st
    _mem()
    st = time.time()


    ##################################################################################
    # 3) Travel time savings — pure Dijkstra (length/speed, no congestion/capacity)
    #
    # A) Base network: edges_aug with Netzlücken at BAD_FFS (15 km/h) — status quo.
    # B) For each Netzlücke: upgrade that single edge to good_ffs (25 km/h),
    #    re-run Dijkstra, compare every OD pair travel time to base.
    # C) TTS_s = Σ_ij trips_ij_s × max(0, tt_base_ij − tt_dev_ij) / 3600  [h/day]
    #    savings_CHF = TTS_h × VTTS × 250 days × duration years
    # D) Reroute factor = routed_dist / straight-line_dist per OD pair (base network).

    tts_df = compute_dijkstra_tts_od(
        edges_aug=edges_aug,
        od_scenarios=od_scenarios,
        points_corridor=points_corridor,
        VTTS=VTTS,
        duration=travel_time_duration,
        good_ffs=25.0,
        trips_per_year=250,
        scenarios=['s1', 's2', 's3'],
    )

    runtimes["Travel time savings (Dijkstra)"] = time.time() - st
    _mem()
    st = time.time()

    ##################################################################################
    # 4) Construction and maintenance costs
    #    Netzlücken: full build at c_cycle_path_new [CHF/m]
    #    Schwachstellen: upgrade at c_cycle_path_update [CHF/m]

    # Build development_candidates.gpkg from live Netzlücken in edges_aug
    nl_devs = edges_aug[edges_aug['ROUTENTYP'] == 'Netzlücke'].copy()
    nl_devs = nl_devs.rename(columns={'ID_edge': 'ID_new'})
    nl_devs['within_corridor'] = True
    nl_devs['on_border'] = False
    nl_devs['dev_type'] = 'netzluecke'
    os.makedirs('data/Network/processed', exist_ok=True)
    nl_devs[['ID_new', 'within_corridor', 'on_border', 'dev_type', 'geometry']].to_file(
        'data/Network/processed/development_candidates.gpkg', driver='GPKG'
    )
    print(f"  development_candidates.gpkg written: {len(nl_devs)} Netzlücken")

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
    # 5) Route comfort — Meister et al. (2021) perceived-distance slope model
    #    slope < 2 %       → +0 % perceived distance   (flat)
    #    2 % ≤ slope < 6 % → +41 % perceived distance  (moderate)
    #    slope ≥ 6 %       → +251 % perceived distance  (steep)
    #    comfort_cost = extra_perceived_length / speed × VTTS × trips × duration
    route_comfort(
        edges_gdf=edges_aug,
        duration=travel_time_duration,
        VTTS=VTTS,
        good_ffs=20.0,
        trips_per_year=250,
    )
    runtimes["Route comfort"] = time.time() - st
    st = time.time()

    ##################################################################################
    # 6) Safety benefits
    #    Per-ROUTENTYP risk rates in scoring.RISK_RATE [accidents/km/trip-year].
    #    Built ROUTENTYP in scoring.NETZLUECKE_BUILT_ROUTENTYP.
    #    Monetary value per accident in scoring.VALUE_PER_ACCIDENT [CHF].
    safety_benefits(
        edges_gdf=edges_aug,
        od_scenarios=od_scenarios,
        duration=travel_time_duration,
    )
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
            boundary=outerboundary, network=network,
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
            title_bar=comp_label, boundary=outerboundary, network=network,
            access_points=access_points, plot_name=plot_tag, col=comp_col,
        )

    gdf_nb["mean_costs"] = gdf_nb[["NB_s1", "NB_s2", "NB_s3"]].mean(axis=1)
    gdf_nb["std"]        = gdf_nb[["NB_s1", "NB_s2", "NB_s3"]].std(axis=1)
    gdf_nb["cv"]         = (gdf_nb["std"] / gdf_nb["mean_costs"].abs()
                            ).replace([float("inf"), float("-inf")], 0).fillna(0) * 1e4
    plot_cost_uncertainty(
        df_costs=gdf_nb.copy(), banned_area=tif_path_plot, boundary=outerboundary,
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

    # ── DEVELOPMENT RESULT VISUALIZATIONS ────────────────────────────────────

    # 1) Ranked priority bar: one bar per development sorted by NB_s2,
    #    error bars = scenario range s1–s3, coloured by dev_type.
    plot_priority_ranking(plot_name="priority_ranking")

    # 2) Component waterfall: C+M (left) vs T+R+S (right) stacked per development.
    #    Shows what drives each net benefit — cheap to build or high travel savings?
    plot_nb_components_waterfall(plot_name="nb_components_waterfall")

    # 3) Travel-time improvement raster: top 3 developments side-by-side,
    #    green = minutes saved vs status quo, corridor network overlaid.
    plot_tt_improvement_map(
        boundary=outerboundary, network=network,
        top_n=3, plot_name="tt_improvement_map",
    )

    # 4) NB on network: candidate edge geometries coloured by NB_s2,
    #    line width proportional to construction cost, corridor as backdrop.
    #    More accurate than the interpolated heatmap in plot_cost_result().
    plot_nb_on_network(
        boundary=outerboundary, network=network,
        access_points=access_points, plot_name="nb_network_map",
    )

    runtimes["Visualization"] = time.time() - st


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('You did a good job ;)')