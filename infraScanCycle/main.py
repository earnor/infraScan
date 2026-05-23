# import packages
import os
import math
import time
import numpy as np
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
    # Set working directory to the script's own folder so all relative paths work
    # regardless of where Python is invoked from.
    # TODO: replace with pathlib.Path(__file__).parent for a fully portable solution.
    os.chdir(r'/Users/ninablattler/PycharmProjects/infraScan/infraScanCycle')
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

    ##################################################################################
    # Define variables for monetisation

    # Construction costs [CHF/m]
    c_cycle_path_new    = 1800   # new path (Netzlücke)    TODO: calibrate
    c_cycle_path_update = 20    # upgrade (Schwachstelle)  TODO: calibrate

    # Maintenance costs
    c_om_cycle_path    = 20     # operational maintenance [CHF/m/year]  TODO: calibrate
    c_structural_maint = 1.2/100 # structural maintenance [fraction of construction cost/year]

    # Value of Travel Time Savings — ARE 2023, cycling, short-distance [CHF/h]
    VTTS = 21.1  # CHF/h
    appraisal_horizon = 50  # years  (standard Swiss infrastructure appraisal horizon)

    runtimes["Initialize variables"] = time.time() - st
    st = time.time()

    ##################################################################################
    # Import and prepare raw data
    print("\nIMPORT RAW DATA \n")

    # Import shapes of lake for plots
    get_lake_data()

    # Import the file containing the locations to be plotted
    import_locations()

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
    # 2) Process network (topology, corridor clip, Netzlücken augmentation)

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
        polygon=innerboundary)

    # Plot 3 — corridor only, node IDs and type counts
    plot_corridor_network(innerboundary, points_corridor, edges_corridor, edges_border,
                          points_full=nodes_gdf, edges_full=edges_final)

    # Plot 4 — edge attributes: free-flow speed, average incline, travel time
    plot_edge_attributes(edges_corridor)


    ##################################################################################
    # A) Add Netzlücken to corridor graph at BAD_FFS
    edges_aug, points_corridor = augment_with_netzluecken(
        edges_corridor, points_corridor,
        c_cycle_path_new=c_cycle_path_new,
        c_om_cycle_path=c_om_cycle_path,
        c_structural_maint=c_structural_maint,
    )




    ##################################################################################
    # Plot all route types: existing / Schwachstellen / Netzlücken / Connectors
    # conn_gdf passed as None — no additional auto-connectors beyond those in edges_aug
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


    # ── DISABLED SECTION (infrastructure generation via random points) ───────────
    # The block below generates candidate cycling links from random points, routes
    # them around protected areas, filters by slope/distance, and builds a combined
    # network.  It is currently disabled because the Netzlücken from the ALLTAG
    # shapefile are used directly instead.  Re-enable if you want to score
    # algorithmically generated alternatives in addition to the GIS-derived ones.
    """
    ##################################################################################
    # 3a) Generate candidate developments from random points

    num_rand = 1000
    random_gdf = generated_access_points(extent=innerboundary, number=num_rand)


    # Assign the return value of the function to the variable name
    filtered_gdf = filter_access_points(random_gdf)
    filtered_gdf.to_file(r"data/Network/processed/generated_nodes.gpkg")

    generated_points = filtered_gdf

    # Import current points as dataframe (Note: all nodes are now access points)
    current_access_points = gpd.read_file(r"data/Network/processed/points_corridor.gpkg")

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
    # 0) Build/refresh the base commune-level OD matrix from the BFS Pendlermatrix.
    #    Applies 8% cycling modal share (no distance filter — all pairs kept).
    #    Saves od_matrix_zh_cycling.csv with commuters_cycling baked in.
    print("\n--- OD BASE MATRIX (BFS Pendlermatrix, 8% modal share) ---")
    import_pendler_matrix(
        path_csv='data/OD/pendler_matrix.csv',
        canton_filter='ZH',
        cycling_mode_share=0.08,
        output_dir='data/OD',
    )

    ##################################################################################
    # 1) OD matrix — disaggregate commune-level cycling trips to Voronoi nodes
    #
    # A) Assign each Voronoi node to a Gemeinde (BFS) via centroid spatial join.
    # B) Read commune-to-commune commuters_cycling from od_matrix_zh_cycling.csv
    #    (8% modal share was applied once in step 0 — no further multiplier here).
    # C) Disaggregate to node pairs using within-commune population/employment shares:
    #      trips_ij_s = origin_share_i × dest_share_j × commuters_cycling_ij
    #    where origin_share is population-weighted, dest_share is employment-weighted.
    # D) Save od_s1/s2/s3.csv per scenario and combined od_scenarios.csv.

    od_scenarios, voronoi_vals = od_cycling_weighted()

    plot_od_results(
        od_scenarios=od_scenarios,
        voronoi_vals=voronoi_vals,
        points_gdf=points_corridor,
        corridor_polygon=innerboundary,
        scenarios=['s1', 's2', 's3'],
        edges_gdf=gpd.read_file(r"data/Network/processed/edges_with_attribute.gpkg"),
        lakes_gdf=gpd.read_file(r"data/landuse_landcover/processed/lake_data_zh.gpkg"),
        save_path='data/OD/od_plot.png',
    )

    runtimes["OD matrix (Voronoi-weighted)"] = time.time() - st
    _mem()
    st = time.time()


    ##################################################################################
    # 2) Travel time savings — pure Dijkstra (length/speed, no congestion/capacity)


    tts_df = compute_dijkstra_tts_od(
        edges_aug=edges_aug,
        od_scenarios=od_scenarios,
        points_corridor=points_corridor,
        VTTS=VTTS,
        duration=appraisal_horizon,

        trips_per_year=250,
        scenarios=['s1', 's2', 's3'],
    )

    runtimes["Travel time savings (Dijkstra)"] = time.time() - st
    _mem()
    st = time.time()

    # ── DISABLED: node accessibility + per-development accessibility benefits ────
    # Steps 3 and 4 below run gravity-based accessibility scoring and compute ΔA
    # per Netzlücke.  They are disabled because compute_accessibility_benefits()
    # is computationally expensive and accessibility is currently set to A=0 in
    # net_benefits().  Re-enable both blocks together when ready.
    """
    ##################################################################################
    # 3) Node accessibility — gravity scoring using network travel times from step 2

    accessibility = node_accessibility(
        voronoi_path='data/Voronoi/voronoi_developments_euclidian_values.shp',
        raster_template='data/landuse_landcover/processed/zone_no_infra/protected_area_corridor.tif',
        beta=2.0,
        scenarios=['s1', 's2', 's3'],
        travel_times_path='data/OD/od_base_travel_times.csv',
    )

    plot_node_accessibility(
        accessibility_results=accessibility,
        corridor_polygon=innerboundary,
        scenarios=['s1', 's2', 's3'],
        save_path='data/Network/accessibility/accessibility_plot.png',
    )

    runtimes["Node accessibility (network travel-time)"] = time.time() - st
    _mem()
    st = time.time()

    
    ##################################################################################
    # 4) Accessibility benefits per Netzlücke — population-weighted gravity ΔA
    #
    # A_base[s][i] = Σ_j empl[j,s] / tt_base[i,j]^β
    # For each Netzlücke d: G_d upgrades that edge to its built ffs, re-run Dijkstra,
    # compute A_d[s][i], then benefits[d][s] = Σ_i pop[i,s] × (A_d − A_base).
    # Saved to data/costs/accessibility_benefits.csv and included in net_benefits().

    acc_df = compute_accessibility_benefits(
        edges_aug=edges_aug,
        points_corridor=points_corridor,
        voronoi_path='data/Voronoi/voronoi_developments_euclidian_values.shp',
        beta=2.0,

        scenarios=['s1', 's2', 's3'],
    )

    runtimes["Accessibility benefits (per-development)"] = time.time() - st
    _mem()
    st = time.time()

    """
    ##################################################################################
    # 5) Construction and maintenance costs
    #    Construction: Netzlücken at c_cycle_path_new [CHF/m], Schwachstellen at c_cycle_path_update [CHF/m]
    #    Maintenance:  (c_om_cycle_path + c_structural_maint × c_cycle_path_new) × length_m × duration
    #    With current defaults: total lifecycle cost ≈ 6,600 CHF/m over 50 years.
    #    NOTE: c_om_cycle_path=100 CHF/m/year is likely too high — calibrate against
    #    Swiss ASTRA/VöV benchmarks before interpreting negative NB results.

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
        upgrade=c_cycle_path_update,
    )
    maintenance_costs(
        duration=appraisal_horizon,
        cycle_path=c_om_cycle_path,
        structural=c_structural_maint,
    )
    runtimes["Construction and maintenance costs"] = time.time() - st
    st = time.time()

    ##################################################################################
    # 6) Route comfort — Meister et al. (2021) slope discomfort model
    #    Per-edge comfort cost [h/trip]:
    #      comfort_h = length_m × (1 + slope_extra_f) × ε / (ffs_edge × 1000)
    #    Slope extra factors:  < 2% → +0%,  2–6% → +41%,  ≥ 6% → +251%
    #    Discomfort ε: Netzlücke=2.0, Nebenverbindung=1.6, Hauptverbindung=1.3, Velobahn=1.0
    #    Benefit = Σ_ij trips × (C_base − C_built) × VTTS × trips_per_year × duration
    route_comfort(
        edges_aug=edges_aug,
        od_scenarios=od_scenarios,
        points_corridor=points_corridor,
        VTTS=VTTS,
        duration=appraisal_horizon,

        trips_per_year=250,
    )
    runtimes["Route comfort"] = time.time() - st
    st = time.time()

    ##################################################################################
    # 7) Safety benefits
    #    Per-edge cost [CHF/trip] = CRASH_RATE_CHF_PKM[ROUTENTYP] × length_km
    #    Rates from KNA Limmattal: Velobahn=0.104, Hauptverbindung=0.409,
    #    Nebenverbindung=0.714, Netzlücke=1.020 CHF/Pkm.
    #    Benefit = Σ_ij trips × (S_base − S_built) × trips_per_year × duration
    #    Netzlücken not traversed by any OD path produce zero benefit.
    safety_benefits(
        edges_aug=edges_aug,
        od_scenarios=od_scenarios,
        points_corridor=points_corridor,
        duration=appraisal_horizon,

        trips_per_year=250,
    )
    runtimes["Safety benefits"] = time.time() - st
    st = time.time()

    ##################################################################################
    # 8) Net benefits: NB = C + M + T + R + S  [CHF] per scenario
    #    C = construction cost (negative), M = maintenance over appraisal_horizon (negative)
    #    T = travel time savings, R = route comfort benefit, S = safety benefit (all positive)
    #    A = accessibility benefit — currently set to 0 (re-enable in scoring.net_benefits)
    nb_df = net_benefits()
    runtimes["Net benefits"] = time.time() - st
    st = time.time()

    ##################################################################################
    # VISUALIZE THE RESULTS
    print("\nVISUALIZE THE RESULTS \n")
    os.makedirs("plot/results", exist_ok=True)

    tif_path_plot = r"data/landuse_landcover/processed/zone_no_infra/protected_area_corridor.tif"
    network       = gpd.read_file(r"data/Network/processed/edges_with_attribute.gpkg")
    access_points = gpd.read_file(r"data/Network/processed/points_corridor.gpkg")

    gdf_nb_raw = gpd.read_file(r"data/costs/net_benefits.gpkg")
    money_cols = ["C", "M", "T_s1", "T_s2", "T_s3",
                  "R_s1", "R_s2", "R_s3",
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
        ("R_s2", "route comfort benefit", "comp_comfort"),
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

    # Grouped bar: all 3 growth scenarios per development side-by-side
    plot_scenario_grouped_bar(plot_name="scenario_grouped_bar")

    # Scatter: total costs vs. total benefits — break-even diagonal shows NB=0
    plot_cost_benefit_scatter(plot_name="cost_benefit_scatter")

    # BCR bar: benefit-to-cost ratio per development, sorted, break-even at 1
    plot_bcr_bar(plot_name="bcr_bar")

    # Dübendorf detail map: zoomed view of the NW sub-area
    plot_duebendorf_zoom(
        df_costs=gdf_nb.copy(), banned_area=tif_path_plot,
        title_bar="cycling net benefit — medium growth (s2)",
        network=network, access_points=access_points,
        plot_name="duebendorf_zoom", col="NB_s2",
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

    # ── Report figures (saved to figures/) ───────────────────────────────────
    generate_report_figures()

    runtimes["Visualization"] = time.time() - st


def generate_report_figures():
    """
    Generates all figures referenced in 05_Results.tex and saves them to figures/.

    Figures produced
    ----------------
    figures/network_all_types.png   — copied from pipeline output
    figures/od_plot.png             — copied from pipeline output
    figures/detour_distribution.png — histogram of OD detour factors
    figures/accessibility_best.png  — node accessibility map, ID 800 (best NB)
    figures/accessibility_worst.png — node accessibility map, ID 385 (worst NB)
    figures/safety_index_map.png    — per-link crash-rate coloured map
    figures/elevation_map.png       — DEM hillshade + contours + network overlay
    figures/comfort_index_map.png   — per-link comfort index (alpha x epsilon) map
    """
    import shutil
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    import matplotlib.cm as mcm
    from matplotlib.colors import LinearSegmentedColormap, LightSource
    from matplotlib.patches import FancyArrowPatch
    from matplotlib_scalebar.scalebar import ScaleBar
    import rasterio
    import rasterio.plot
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    os.makedirs('figures', exist_ok=True)

    # ── shared data layers ────────────────────────────────────────────────────
    network_edges = gpd.read_file('data/Network/processed/edges_with_attribute.gpkg')
    corridor_pts  = gpd.read_file('data/Network/processed/points_corridor.gpkg')
    lakes_path    = 'data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp'
    cities_path   = 'data/manually_gathered_data/Cities.shp'

    net_bounds = network_edges.total_bounds   # [minx, miny, maxx, maxy]
    x_pad, y_pad = 500, 500

    def _add_base(ax):
        """Add lakes, grey network, city labels, scale bar, north arrow."""
        if os.path.exists(lakes_path):
            gpd.read_file(lakes_path).plot(ax=ax, color='lightblue', zorder=1)
        network_edges.plot(ax=ax, color='#bbbbbb', lw=0.8, zorder=2, alpha=0.6)
        if os.path.exists(cities_path):
            cities = gpd.read_file(cities_path, crs='epsg:2056')
            cities.plot(ax=ax, color='black', markersize=50, zorder=8)
            for _, r in cities.iterrows():
                ax.annotate(r['location'], xy=r.geometry.coords[0],
                            ha='center', va='top', xytext=(0, -5),
                            textcoords='offset points', fontsize=10, zorder=8)
        ax.add_artist(ScaleBar(1, location='lower right'))
        ax.text(0.96, 0.93, 'N', fontsize=22, weight='bold',
                ha='center', va='center', transform=ax.transAxes, zorder=100)
        ax.add_patch(FancyArrowPatch(
            (0.96, 0.90), (0.96, 0.97), color='black', lw=1.5,
            arrowstyle='->', mutation_scale=20, transform=ax.transAxes, zorder=100))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor('black'); sp.set_linewidth(1)
        ax.set_xlim(net_bounds[0] - x_pad, net_bounds[2] + x_pad)
        ax.set_ylim(net_bounds[1] - y_pad, net_bounds[3] + y_pad)

    # ── 1. Copy pipeline figures ──────────────────────────────────────────────
    for src, dst in [
        ('data/Network/processed/network_all_types.png', 'figures/network_all_types.png'),
        ('data/OD/od_plot.png',                          'figures/od_plot.png'),
    ]:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f'  copied  {src} → {dst}')
        else:
            print(f'  [WARN] pipeline figure not found: {src}')

    # ── 2. Detour factor histogram ────────────────────────────────────────────
    od_path = 'data/OD/od_base_travel_times.csv'
    if os.path.exists(od_path):
        od        = pd.read_csv(od_path)
        df_clipped = od['detour_factor'].clip(upper=20)
        median_v  = od['detour_factor'].median()
        mean_v    = od['detour_factor'].mean()
        pct95_v   = od['detour_factor'].quantile(0.95)
        max_v     = od['detour_factor'].max()

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(df_clipped, bins=80, color='steelblue', edgecolor='white',
                linewidth=0.4, zorder=3)
        ax.axvline(median_v, color='#e74c3c', lw=1.8, linestyle='--',
                   label=f'Median = {median_v:.2f}')
        ax.axvline(mean_v,   color='#e67e22', lw=1.8, linestyle=':',
                   label=f'Mean = {mean_v:.2f}')
        ax.set_xlabel('Detour factor  (routed distance / Euclidean distance)', fontsize=12)
        ax.set_ylabel('Number of OD pairs', fontsize=12)
        ax.set_title(
            f'Detour factor distribution across {len(od):,} base-graph OD pairs\n'
            f'(x-axis capped at 20; true max = {max_v:.1f})',
            fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.text(0.97, 0.97,
                f'95th pct = {pct95_v:.2f}\nMax = {max_v:.1f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        plt.tight_layout()
        plt.savefig('figures/detour_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('  saved   figures/detour_distribution.png')

    # ── 3. Accessibility maps — best (800) and worst (385) ────────────────────
    cands_path = 'data/Network/processed/development_candidates.gpkg'
    if os.path.exists(cands_path) and len(corridor_pts) > 0:
        cands = gpd.read_file(cands_path)
        cands['ID_new'] = cands['ID_new'].astype(int)

        for dev_id, fname, subtitle in [
            (800, 'figures/accessibility_best.png',
             'Best-performing development  (ID 800, 35 m Velobahn)'),
            (385, 'figures/accessibility_worst.png',
             'Worst-performing development  (ID 385, 6.8 km Nebenverbindung)'),
        ]:
            dev_row = cands[cands['ID_new'] == dev_id]
            if dev_row.empty:
                print(f'  [WARN] ID {dev_id} not in development_candidates — skipping')
                continue

            dev_geom = dev_row.iloc[0].geometry
            pts = corridor_pts.copy()

            # accessibility gain proxy: 1 / (1 + distance-to-dev [km])
            pts['dist_km']  = pts.geometry.distance(dev_geom) / 1000.0
            pts['acc_gain'] = 1.0 / (1.0 + pts['dist_km'])
            gain_min, gain_max = pts['acc_gain'].min(), pts['acc_gain'].max()
            pts['acc_norm'] = (pts['acc_gain'] - gain_min) / (gain_max - gain_min + 1e-9)

            fig, ax = plt.subplots(figsize=(13, 9))
            _add_base(ax)

            sc = ax.scatter(
                pts.geometry.x, pts.geometry.y,
                c=pts['acc_norm'], cmap='YlOrRd', s=45,
                vmin=0, vmax=1, zorder=6, edgecolors='none', alpha=0.9,
            )
            dev_row.plot(ax=ax, color='red', lw=4, zorder=10)
            # label the development
            mid = dev_geom.interpolate(0.5, normalized=True)
            ax.annotate(f'ID {dev_id}', xy=(mid.x, mid.y),
                        xytext=(6, 6), textcoords='offset points',
                        fontsize=10, fontweight='bold', color='red', zorder=11)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='2%', pad=0.4)
            cbar = plt.colorbar(sc, cax=cax)
            cbar.set_label('Normalised accessibility gain\n(1 / (1 + distance to development [km]))',
                           rotation=90, labelpad=12, fontsize=10)

            leg = [mpatches.Patch(color='red', label=f'Development ID {dev_id} (highlighted)')]
            ax.legend(handles=leg, loc='upper left', fontsize=10, framealpha=0.85)
            ax.set_title(f'Node-level accessibility — scenario S2\n{subtitle}',
                         fontsize=12, pad=8)
            plt.tight_layout()
            plt.savefig(fname, dpi=300, bbox_inches='tight')
            plt.close()
            print(f'  saved   {fname}')

    # ── 4. Safety index map ───────────────────────────────────────────────────
    CRASH_RATE = {
        'Velobahn':                       0.104,
        'Veloschnellroute':               0.104,
        'Hauptverbindung':                0.409,
        'Nebenverbindung':                0.714,
        'Zusätzliche Freizeitverbindung': 0.409,
        'Netzlücke':                      1.020,
        'connector':                      1.020,
    }

    edges_safe = network_edges.copy()
    edges_safe['crash_rate'] = edges_safe['ROUTENTYP'].map(CRASH_RATE).fillna(1.020)

    if os.path.exists(cands_path):
        nl = gpd.read_file(cands_path)[['geometry']].copy()
        nl['crash_rate'] = 1.020
        edges_safe = gpd.GeoDataFrame(
            pd.concat([edges_safe[['crash_rate', 'geometry']], nl], ignore_index=True),
            geometry='geometry', crs='epsg:2056')

    vmin_s, vmax_s = 0.104, 1.020
    norm_s  = mcolors.Normalize(vmin=vmin_s, vmax=vmax_s)
    cmap_s  = plt.cm.RdYlGn_r

    fig, ax = plt.subplots(figsize=(14, 9))
    _add_base(ax)
    edges_safe.plot(ax=ax, column='crash_rate', cmap=cmap_s, norm=norm_s,
                    lw=1.8, zorder=5, legend=False)

    sm_s = mcm.ScalarMappable(cmap=cmap_s, norm=norm_s)
    sm_s.set_array([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='2%', pad=0.4)
    cbar = plt.colorbar(sm_s, cax=cax)
    cbar.set_label('Crash-cost rate [CHF/Pkm]\n(KNA Limmattal methodology)',
                   rotation=90, labelpad=14, fontsize=11)

    leg_handles = [
        mpatches.Patch(color=cmap_s(norm_s(r)), label=f'{rt}  ({r:.3f} CHF/Pkm)')
        for rt, r in [('Velobahn', 0.104), ('Hauptverbindung / Freizeit', 0.409),
                      ('Nebenverbindung', 0.714), ('Netzlücke (unbuilt)', 1.020)]
    ]
    ax.legend(handles=leg_handles, loc='upper left', fontsize=9, framealpha=0.85,
              title='ROUTENTYP', title_fontsize=10)
    ax.set_title('Per-link safety index — crash-cost rate (CHF/Pkm)\n'
                 'Red = high risk (Netzlücke / Nebenverbindung)  ·  Green = low risk (Velobahn)',
                 fontsize=12, pad=8)
    plt.tight_layout()
    plt.savefig('figures/safety_index_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('  saved   figures/safety_index_map.png')

    # ── 5. Elevation map with hillshade + contours ────────────────────────────
    dem_path = 'data/elevation_model/elevation.tif'
    if os.path.exists(dem_path):
        with rasterio.open(dem_path) as src:
            dem     = src.read(1).astype(float)
            extent  = [src.bounds.left, src.bounds.right,
                       src.bounds.bottom, src.bounds.top]

        dem_min, dem_max = float(np.nanmin(dem)), float(np.nanmax(dem))
        norm_dem  = mcolors.Normalize(vmin=dem_min, vmax=dem_max)
        cmap_dem  = plt.cm.terrain

        ls = LightSource(azdeg=315, altdeg=45)
        hs = ls.hillshade(dem, vert_exag=2)

        ny, nx = dem.shape
        xs = np.linspace(extent[0], extent[1], nx)
        ys = np.linspace(extent[2], extent[3], ny)[::-1]

        fig, ax = plt.subplots(figsize=(14, 9))
        ax.imshow(cmap_dem(norm_dem(dem)), extent=extent, origin='upper',
                  zorder=1, alpha=0.75)
        ax.imshow(hs, extent=extent, origin='upper',
                  cmap='gray', alpha=0.35, zorder=2)

        contour_step = 20
        c_levels = np.arange(
            int(dem_min // contour_step) * contour_step,
            int(dem_max // contour_step) * contour_step + contour_step,
            contour_step)
        cs = ax.contour(xs, ys, dem, levels=c_levels,
                        colors='black', linewidths=0.3, alpha=0.4, zorder=3)
        ax.clabel(cs, inline=True, fontsize=6, fmt='%d m')

        network_edges.plot(ax=ax, color='#222222', lw=0.9, zorder=4, alpha=0.7)
        if os.path.exists(lakes_path):
            gpd.read_file(lakes_path).plot(ax=ax, color='lightblue', zorder=5, alpha=0.85)
        if os.path.exists(cities_path):
            cities = gpd.read_file(cities_path, crs='epsg:2056')
            cities.plot(ax=ax, color='black', markersize=50, zorder=7)
            for _, r in cities.iterrows():
                ax.annotate(r['location'], xy=r.geometry.coords[0],
                            ha='center', va='top', xytext=(0, -5),
                            textcoords='offset points', fontsize=10, zorder=7)

        sm_dem = mcm.ScalarMappable(cmap=cmap_dem, norm=norm_dem)
        sm_dem.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.4)
        cbar = plt.colorbar(sm_dem, cax=cax)
        cbar.set_label('Elevation [m a.s.l.]', rotation=90, labelpad=14, fontsize=11)

        ax.add_artist(ScaleBar(1, location='lower right'))
        ax.text(0.96, 0.93, 'N', fontsize=22, weight='bold',
                ha='center', va='center', transform=ax.transAxes, zorder=100)
        ax.add_patch(FancyArrowPatch(
            (0.96, 0.90), (0.96, 0.97), color='black', lw=1.5,
            arrowstyle='->', mutation_scale=20, transform=ax.transAxes, zorder=100))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor('black'); sp.set_linewidth(1)
        ax.set_title('Digital elevation model (2 m resolution) with 20 m contour lines\n'
                     'and cycling network overlay', fontsize=12, pad=8)
        ax.set_xlim(net_bounds[0] - x_pad, net_bounds[2] + x_pad)
        ax.set_ylim(net_bounds[1] - y_pad, net_bounds[3] + y_pad)
        plt.tight_layout()
        plt.savefig('figures/elevation_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('  saved   figures/elevation_map.png')

    # ── 6. Comfort index map ──────────────────────────────────────────────────
    comfort_net_path = 'data/costs/route_comfort_network.gpkg'
    if os.path.exists(comfort_net_path):
        EPSILON = {
            'Velobahn': 1.0, 'Veloschnellroute': 1.0,
            'Hauptverbindung': 1.3, 'Nebenverbindung': 1.6,
            'Zusätzliche Freizeitverbindung': 1.3,
            'Netzlücke': 2.0, 'connector': 2.0,
        }
        comfort_net = gpd.read_file(comfort_net_path)
        comfort_net['epsilon'] = comfort_net['ROUTENTYP'].map(EPSILON).fillna(2.0)
        # comfort index = (1 + alpha) * epsilon  where extra_factor = 1 + alpha
        comfort_net['comfort_index'] = comfort_net['extra_factor'] * comfort_net['epsilon']

        vmin_c = comfort_net['comfort_index'].min()
        vmax_c = comfort_net['comfort_index'].max()
        norm_c = mcolors.Normalize(vmin=vmin_c, vmax=vmax_c)
        cmap_c = plt.cm.RdYlGn_r

        fig, ax = plt.subplots(figsize=(14, 9))
        _add_base(ax)
        comfort_net.plot(ax=ax, column='comfort_index', cmap=cmap_c, norm=norm_c,
                         lw=2.0, zorder=5, legend=False)

        sm_c = mcm.ScalarMappable(cmap=cmap_c, norm=norm_c)
        sm_c.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='2%', pad=0.4)
        cbar = plt.colorbar(sm_c, cax=cax)
        cbar.set_label('Comfort index  (1 + α) × ε\nα = slope discomfort,  ε = ROUTENTYP multiplier',
                       rotation=90, labelpad=14, fontsize=10)

        ax.set_title('Per-link route comfort index — slope discomfort factor α × ROUTENTYP multiplier ε\n'
                     'Red = steep / low-quality link;  Green = flat Velobahn',
                     fontsize=12, pad=8)

        comfort_bounds = comfort_net.total_bounds
        ax.set_xlim(comfort_bounds[0] - x_pad, comfort_bounds[2] + x_pad)
        ax.set_ylim(comfort_bounds[1] - y_pad, comfort_bounds[3] + y_pad)
        plt.tight_layout()
        plt.savefig('figures/comfort_index_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        print('  saved   figures/comfort_index_map.png')

    print('\n[generate_report_figures] done — figures saved to figures/')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('You did a good job ;)')