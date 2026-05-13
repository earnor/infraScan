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
    ACCESS_POINT_MIN_DIST = 0

    # Define spatial limits of the research corridor
    # The coordinates must end with 000 in order to match the coordinates of the input raster data
    e_min, e_max = 2687000, 2708000     # 2688000, 2704000 - 2688000, 2705000
    n_min, n_max = 1237000, 1254000     # 1238000, 1252000 - 1237000, 1252000
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
    VTTS = 18.2  # CHF/h — Swiss official value ~18.2 CHF/h for leisure cycling
    travel_time_duration = 50  # appraisal horizon [years]

    # Route comfort monetisation [CHF / m / year per CLI unit]
    comfort_value_chf_m_year = 2.0  # TODO: calibrate

    # Safety — willingness-to-pay to avoid risk-weighted route exposure [CHF/(risk_unit·trip·year)]
    value_of_safety = 0.0001  # TODO: set once a unit value is agreed (e.g. 0.0001)

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
    # Reads ALLTAG shapefile, reprojects to EPSG:2056, snaps endpoints to 0.1 m grid,
    # explodes MultiLineStrings, and converts to a primal graph via momepy.
    # Outputs: data/Network/processed/{nodes,edges}.gpkg + CSV exports.
    network = import_network_GIS_ALLTAG()

    # Plot 1 — raw shapefile import: edges coloured by ROUTENTYP, no node classification yet
    plot_raw_network(network, title="Step 1 — Raw imported network (ALLTAG shapefile)", save_path="data/Network/processed/plot_01_raw_import.png")

    runtimes["Import network data"] = time.time() - st
    st = time.time()

    ##################################################################################
    # 2) Process network
    # reformat_network(): resolves topology, tags intersections/endpoints, attaches
    #   route names and Netzlücken/Schwachstellen flags, adds travel-time attributes.
    #   Outputs: data/Network/processed/{edges,points}.gpkg
    # network_in_corridor(): spatial filter to innerboundary; thins access points by
    #   ACCESS_POINT_MIN_DIST; saves corridor and border-crossing subsets.
    # get_edge_attributes(): returns the fully attributed edge GDF used by scoring.
    nodes_gdf, edges_final = reformat_network()

    # Plot 2 — after topology resolution: intersections, dead ends, through-nodes classified
    plot_network_classified(nodes_gdf, edges_final)

    points_corridor, edges_corridor, edges_border = network_in_corridor(
        polygon=innerboundary, access_point_min_dist=ACCESS_POINT_MIN_DIST)

    # Plot 3 — corridor filter: full network as context, corridor subset highlighted
    plot_corridor_network(outerboundary, points_corridor, edges_corridor, edges_border,
                         points_full=nodes_gdf, edges_full=edges_final)


    edges = get_edge_attributes()
    # TODO: edges_corridor.gpkg (written by network_in_corridor) and
    # edges_with_attribute.gpkg (written by get_edge_attributes) are the same
    # network but different files.  Later code reads both (edges here, edges_sq
    # below at line 451).  Verify that ID_edge is consistent between the two files;
    # travel_cost_developments() matches candidates to the base GDF by ID_edge.

    # Plot 4 — edge attributes: free-flow speed, capacity, travel time
    plot_edge_attributes(edges)

    ##################################################################################
    # A) Add Netzlücken to corridor graph at BAD_FFS
    # Netzlücken are planned-but-unbuilt edges (is_development == 1).  Including them
    # with a degraded speed (BAD_FFS = 15 km/h) makes the graph more connected
    # while signalling that these links are substandard in their current state.
    # They get their own ROUTENTYP so they are distinguishable from both existing
    # edges and auto-generated connectors in plots and routing.
    edges_aug = edges.copy()
    nl_mask   = edges_aug['is_development'] == 1
    edges_aug.loc[nl_mask, 'ROUTENTYP'] = 'Netzlücke'
    edges_aug.loc[nl_mask, 'ffs']       = BAD_FFS
    edges_aug.loc[nl_mask, 'tt_min']    = (
        edges_aug.loc[nl_mask, 'length_m'] / 1000) / BAD_FFS * 60
    print(f"\n  Added {nl_mask.sum()} Netzlücken at BAD_FFS ({BAD_FFS} km/h) to corridor graph")

    G_with_nl = build_graph_direct(edges_aug)
    conn_nl   = check_network_connectivity(G_with_nl, label="corridor + Netzlücken")

    ##################################################################################
    # B) Auto-generate connector paths where the network is still disconnected
    # Always delete any stale connectivity_developments.gpkg so _build_base_gdf()
    # never loads bridges from a previous run with different corridor/Netzlücken state.
    _conn_cache = 'data/Network/processed/connectivity_developments.gpkg'
    if os.path.exists(_conn_cache):
        os.remove(_conn_cache)
        print("  Cleared stale connectivity_developments.gpkg")

    conn_gdf = gpd.GeoDataFrame(
        columns=['geometry', 'ROUTENTYP', 'ffs', 'tt_min', 'length_m'],
        geometry='geometry', crs="EPSG:2056"
    )
    if not conn_nl['is_connected']:
        print(f"\n  {conn_nl['num_components']} component(s) remain after Netzlücken — "
              f"auto-generating connector paths …")
        raw_conn = generate_connectivity_developments(
            corridor_polygon=innerboundary,
            include_netzluecken=True,
        )
        if len(raw_conn) > 0:
            raw_conn['ROUTENTYP'] = 'Nebenverbindung'  # cycling path — same type as bridge edges
            raw_conn['ffs']       = WORST_FFS
            raw_conn['tt_min']    = raw_conn['length_m'] / 1000 / WORST_FFS * 60
            conn_gdf = raw_conn

            edges_full = pd.concat([edges_aug, conn_gdf], ignore_index=True)
            G_full     = build_graph_direct(edges_full)
            check_network_connectivity(G_full, label="corridor + Netzlücken + Connectors")
        else:
            print("  No connectors generated — gaps may already be bridged by Netzlücken.")
            # Write empty file so _build_base_gdf() finds no bridges to load
            gpd.GeoDataFrame(geometry=[], crs="EPSG:2056").to_file(_conn_cache, driver='GPKG')
    else:
        print("  Network fully connected after adding Netzlücken — no connectors needed.")
        # Write empty file so _build_base_gdf() finds no bridges to load
        gpd.GeoDataFrame(geometry=[], crs="EPSG:2056").to_file(_conn_cache, driver='GPKG')

    ##################################################################################
    # C) Plot all route types: existing / Schwachstellen / Netzlücken / Connectors
    plot_network_all_types(
        edges_gdf=edges_aug,
        conn_gdf=conn_gdf,
        corridor_polygon=innerboundary,
        save_path="data/Network/processed/network_all_types.png",
    )

    # Save the full annotated network (all route types) as GeoPackages for GIS reference.
    # Edges include ID_edge, ROUTENTYP, ffs, tt_min, is_development, is_schwachstelle.
    # Nodes include ID_point, is_intersection, is_through_point, is_endpoint.
    _full_edges = edges_aug.copy()
    if len(conn_gdf) > 0:
        _full_edges = pd.concat([_full_edges, conn_gdf], ignore_index=True)
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
    _rt_col = next((c for c in edges.columns if c.upper().startswith('ROUTENTYP')), None)
    print(edges[_rt_col].value_counts() if _rt_col else "ROUTENTYP column not found")
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


    # TODO: Option A (random-point pipeline) is entirely commented out below.
    # Decide: remove it permanently, or document why it is kept for reference.
    # If kept, it must be tested independently — it references functions
    # (generated_access_points, filter_access_points, connect_points_to_network,
    # routing_raster, build_combined_network) that diverged from the Option B flow.
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
    # 3b) Assemble all development candidates and prepare for scoring
    #
    # Step 1 — Official candidates: Netzlücken + Schwachstellen
    #   Extracts edges tagged is_development==1 (Netzlücken: planned-but-unbuilt
    #   connections) and is_schwachstelle==1 (Schwachstellen: existing edges below
    #   quality standard) from the attributed edge table.  Restricts them to the
    #   study corridor, assigns sequential IDs 0…N-1, and writes:
    #     • developments_list.csv      — tabular report used by the scoring loop
    #     • development_candidates.gpkg — geometries used for routing and plotting
    #   These are the ONLY edges scored individually for cost-benefit analysis.
    #
    # Step 2 — Connectivity bridges (reuse conn_gdf from step 2B above)
    #   The auto-generated connectors were already built in step 2B using
    #   generate_connectivity_developments(include_netzluecken=True), which routes
    #   around protected areas and only places a bridge where a Netzlücke does not
    #   already close the gap.  Reusing conn_gdf avoids a second raster-routing pass
    #   and keeps development_candidates.gpkg consistent with the step-2B plot.
    #   Bridges receive IDs starting after the last official candidate so the CSV
    #   and GPKG stay aligned.  Bridges are NEVER scored — they are infrastructure
    #   placeholders that ensure Dijkstra can reach every corridor node.
    #
    # Step 3 — Status-quo Voronoi
    #   Computes Euclidean catchment polygons around each access point using the
    #   existing network only (no developments active).  Used by the scenario module
    #   to aggregate population/employment demand to each access point.
    #
    # Step 4 — Import scenario variables
    #   Loads population and employment rasters for the wider bounding box that
    #   covers the full Voronoi catchment (larger than the corridor).

    # Step 1: official Netzlücken / Schwachstellen
    developments = get_development_candidates(
        edges, corridor_polygon=innerboundary, dev_type_filter=DEV_TYPE_FILTER
    )

    runtimes["Generate infrastructure developments"] = time.time() - st
    st = time.time()

    # Step 2: connectivity bridges are stored separately — they are never scored.
    # development_candidates.gpkg contains ONLY official Netzlücken/Schwachstellen.
    # Bridges are already written to connectivity_developments.gpkg in step 2B.
    connectivity_devs = conn_gdf if len(conn_gdf) > 0 else gpd.GeoDataFrame()

    developments.to_file('data/Network/processed/development_candidates.gpkg', driver='GPKG')
    print(f"  development_candidates.gpkg: {len(developments)} official candidates "
          f"({(developments['dev_type'] == 'netzluecke').sum()} Netzlücken, "
          f"{(developments['dev_type'] == 'schwachstelle').sum()} Schwachstellen)")
    print(f"  connectivity_developments.gpkg: {len(connectivity_devs)} bridge(s) "
          f"(not scored — infrastructure placeholders only)")

    # Plot 5 — official candidates: existing (grey) / Schwachstellen (orange) / Netzlücken (red)
    plot_developments(edges, corridor_polygon=innerboundary)


    # Step 3: Voronoi polygons for the status quo
    voronoi_sq = get_voronoi_status_quo(corridor_polygon=innerboundary)
    plot_voronoi_status_quo(voronoi_sq, nodes_gdf, edges_final, corridor_polygon=innerboundary)

    # Step 4: import scenario variables (population + employment) for wider bounding box
    limits_variables = [2680600, 1227700, 2724300, 1265600]
    import_data(limits_variables)
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
    # 0) Load developments and network files
    developments = pd.read_csv('data/Network/processed/developments_list.csv')
    edges_sq  = gpd.read_file('data/Network/processed/edges_with_attribute.gpkg')
    nodes_sq  = gpd.read_file('data/Network/processed/points_with_attribute.gpkg')
    # dev_nodes: representative points for each scored development (centroid of edge geometry)
    _dev_cands = gpd.read_file('data/Network/processed/development_candidates.gpkg')
    dev_nodes  = _dev_cands.copy()
    dev_nodes['geometry'] = dev_nodes.geometry.centroid
    dev_nodes  = dev_nodes[dev_nodes["within_corridor"] | dev_nodes["on_border"]]
    runtimes["Load scoring inputs"] = time.time() - st
    st = time.time()

    ##################################################################################
    # 1) OD matrix — fastest path between every pair of corridor access points,
    #    filtered to pairs whose path distance is ≤ 25 km.
    #
    #    Graph: full base network (existing edges at surveyed speeds, Netzlücken at
    #    BAD_FFS, Schwachstellen at BAD_FFS, connectivity bridges at WORST_FFS).
    #    Dijkstra weight: tt_sec (travel time — finds the fastest route, not shortest).
    #    Distance filter: cumulative edge length along the fastest-time path must not
    #    exceed OD_MAX_DIST_M.  Pairs beyond this threshold are not realistic cycling
    #    trips in this corridor and are dropped before any further analysis.
    #
    #    Output: data/OD/od_fastest_paths.csv
    #      origin_id  – ID_point of origin access point
    #      dest_id    – ID_point of destination access point
    #      tt_sec     – fastest travel time [s]
    #      dist_m     – path distance [m] along the fastest-time route
    OD_MAX_DIST_M = 25_000
    print(f"\n--- OD MATRIX (fastest paths, ≤{OD_MAX_DIST_M/1000:.0f} km) ---")
    od_df = compute_od_matrix(max_dist_m=OD_MAX_DIST_M)
    runtimes["OD matrix (fastest paths ≤25 km)"] = time.time() - st
    _mem()
    st = time.time()

    ##################################################################################
    # 2) Raster-based travel time and accessibility
    make_cycling_speed_raster(cycling_speed_kmh=15)
    travel_cost_polygon(limits_corridor)
    voronoi_sq = gpd.read_file(r"data/Network/travel_time/Voronoi_statusquo.gpkg")

    accessib_sq = accessibility_status_quo(VTT_h=VTTS, duration=travel_time_duration)

    travel_cost_developments(limits_corridor)
    single_tt_voronoi_ton_one("data/Network/travel_time/developments")
    polygon_gdf = gpd.read_file(r"data/Voronoi/combined_developments.gpkg")
    scenario_to_voronoi(polygon_gdf, euclidean=False)
    GetVoronoiOD_multi()
    accessib_devs = accessibility_developments(accessib_sq, VTT_h=VTTS, duration=travel_time_duration)
    runtimes["Raster travel time and accessibility"] = time.time() - st
    _mem()
    st = time.time()

    ##################################################################################
    # 3) Travel time savings — pure Dijkstra (length/speed, no congestion/capacity)
    monetize_dijkstra_tts(VTTS=VTTS, duration=travel_time_duration)
    runtimes["Travel time savings (Dijkstra)"] = time.time() - st
    _mem()
    st = time.time()

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