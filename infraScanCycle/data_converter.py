from operator import truediv
from turtledemo.chaos import plot

import momepy
import osmnx
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely.geometry import LineString, MultiLineString
import os
import pandas as pd
import ast
import networkx as nx
from typing import Optional, Any
from shapely.validation import make_valid
from OSM_network import build_network_from_shapefile, check_network_connectivity

# TODO: hardcoded os.chdir() — this file is imported by main.py which already
# sets the working directory.  Calling chdir() here again is redundant and
# will break if the module is imported from a different entry point.
# Remove this line and rely on main.py to set the working directory, or
# use pathlib.Path(__file__).parent for all relative paths in this module.
os.chdir(r'/Users/ruki/PycharmProjects/infraScan/infraScanCycle')
# os.chdir(r'/Users/ninablattler/PycharmProjects/infraScan/infraScanCycle')


def import_network_GIS_ALLTAG() -> Any:
    path = 'data/raw/ALLTAG/OGD_VELO_ALLTAG_NETZ_L_M.shp'

    # ── Step 1: Build routing graph + connectivity check ─────────────────────
    # build_network_from_shapefile() loads the raw ALLTAG shapefile, nodes all
    # lines at their mutual intersections via unary_union (so every crossing
    # becomes a true shared node), and returns a bidirectional DiGraph.
    print("\n[import_network_GIS_ALLTAG] Building noded routing graph from ALLTAG shapefile …")
    G_alltag, gdf_noded = build_network_from_shapefile(
        path,
        cycling_speed_kmh=15,
        snap_tolerance=1.0,
    )

    # Connectivity report — printed immediately so gaps are visible before any
    # further processing.  If num_components > 1 the corridor Dijkstra will
    # be unable to route between disconnected sub-networks.
    conn_report = check_network_connectivity(G_alltag, label="ALLTAG raw network (noded routing graph)")

    if not conn_report['is_connected']:
        print(f"  ⚠  {conn_report['num_components']} disconnected component(s) detected "
              f"in the raw ALLTAG data.\n"
              f"     Largest component covers "
              f"{conn_report['largest_component']}/{G_alltag.number_of_nodes()} nodes "
              f"({100*conn_report['largest_component']/max(G_alltag.number_of_nodes(),1):.1f}%).\n"
              f"     Connectivity bridges will be generated later to link isolated sub-graphs.")

    # ── Step 2: Load + preprocess for momepy primal-graph conversion ──────────
    df = gpd.read_file(path, engine='pyogrio')

    # Ensure CRS is metric (Swiss LV95)
    if df.crs.to_epsg() != 2056:
        df = df.to_crs("EPSG:2056")

    df['geometry'] = df.geometry.apply(lambda g: make_valid(g) if g is not None else g)
    df = df[df.geometry.notna() & ~df.geometry.is_empty].reset_index(drop=True)

    # Store original winding geometry as WKT for later visualization
    df['visual_geom'] = df['geometry'].apply(lambda g: g.wkt)

    # Save the actual winding length BEFORE any processing
    df['length_m'] = df.geometry.length

    # Explode MultiLineStrings into individual LineStrings so every sub-segment
    # becomes its own edge and intermediate shared nodes are preserved in the
    # primal graph topology.  A MultiLineString that bends through an
    # intermediate node would otherwise collapse to a single straight edge that
    # skips that node, breaking connectivity.
    df = df.explode(index_parts=False).reset_index(drop=True)

    # Snap all endpoints to 1-metre grid so topologically shared nodes get
    # exactly the same coordinate and momepy merges them into one graph node.
    # Interior vertices are kept unchanged for geometry accuracy.
    def _snap_endpoints(geom):
        if geom is None or geom.is_empty:
            return None
        coords = list(geom.coords)
        if len(coords) < 2:
            return None
        s = (round(coords[0][0], 0), round(coords[0][1], 0))
        e = (round(coords[-1][0], 0), round(coords[-1][1], 0))
        if s == e:
            return None  # degenerate (zero-length after snapping) — drop it
        interior = [(x, y) for x, y in coords[1:-1]]
        return LineString([s] + interior + [e])

    df['geometry'] = df.geometry.apply(_snap_endpoints)
    df = df[df.geometry.notna()].reset_index(drop=True)
    df['length_m'] = df.geometry.length  # recompute after explode + snapping

    # ── Step 3: Convert to primal graph via momepy ────────────────────────────
    # momepy builds a networkx MultiGraph where every LineString endpoint that
    # shares a coordinate becomes a single node — the primal topology needed by
    # reformat_network() downstream.
    H = momepy.gdf_to_nx(df, approach='primal')
    gdf_nodes, gdf_edges = momepy.nx_to_gdf(H, points=True)

    # Connectivity check on the momepy primal graph for comparison.
    # momepy returns an undirected MultiGraph — check_network_connectivity
    # handles both directed and undirected inputs.
    check_network_connectivity(H, label="ALLTAG momepy primal graph")

    # ── Step 4: Save outputs ──────────────────────────────────────────────────
    os.makedirs('data/Network/processed', exist_ok=True)
    gdf_nodes.to_file('data/Network/processed/nodes.gpkg', driver='GPKG')
    gdf_edges.to_file('data/Network/processed/edges.gpkg', driver='GPKG')

    # ── CSV Export ────────────────────────────────────────────────────────
    # Nodes: drop geometry, export all attributes
    nodes_csv = gdf_nodes.drop(columns='geometry').copy()
    # Add x/y explicitly from the Point geometry
    nodes_csv['x'] = gdf_nodes.geometry.x
    nodes_csv['y'] = gdf_nodes.geometry.y
    nodes_csv.to_csv('data/Network/processed/nodes_export.csv', index=False)
    print(f"  Nodes exported → data/Network/processed/nodes_export.csv  ({len(nodes_csv)} rows)")

    # Edges: drop geometry, export all attributes (includes length_m, ROUTENTYP, visual_geom, etc.)
    edges_csv = gdf_edges.drop(columns='geometry').copy()
    edges_csv.to_csv('data/Network/processed/edges_export.csv', index=False)
    print(f"  Edges exported → data/Network/processed/edges_export.csv  ({len(edges_csv)} rows)")

    # Return only edges GeoDataFrame — structure.py expects a single GeoDataFrame
    return gdf_edges

def get_attributes_GIS_ALLTAG():
    import geopandas as gpd
    import os

    path = 'data/raw/ALLTAG/OGD_VELO_ALLTAG_NETZ_L_M.shp'
    df = gpd.read_file(path, engine='pyogrio')

    # Ensure CRS is metric (Swiss LV95 / EPSG:2056)
    if df.crs.to_epsg() != 2056:
        df = df.to_crs("EPSG:2056")

    # --- LINK ATTRIBUTES ---
    # Select and rename columns based on Produktblatt (408.3)
    link_attributes = df[[
        'geometry',
        'RW_KEY',           # Route number with +/- suffix (unique edge identifier)
        'RW_KEY_NR',        # Route number without +/- suffix
        'VERBINDUNGSNAME',  # Route name with start and destination
        'ROUTENTYP',        # Type: Veloschnellroute / Hauptverbindung / Nebenverbindung / Zusätzliche Freizeitverbindung
        'PLANUNGSTYP',      # Status: bestehend / geplant / langfristiges Ziel / Variante / ...
        'FAHRRICHTUNGSTYP', # Direction: beidseitig / eine Richtung / eine Richtung mit + und - Achse
        'ZWECK',            # Purpose: Alltag / Freizeit
        'KMMIN',            # Start kilometer
        'KMMAX',            # End kilometer
    ]].copy()

    # Compute true winding length from original geometry (before any simplification)
    link_attributes['length_m'] = link_attributes.geometry.length

    # Store original geometry as WKT for later visualization
    link_attributes['visual_geom'] = link_attributes.geometry.apply(lambda g: g.wkt)

    # --- NODES ---
    # Extract unique endpoints from each geometry
    coord_to_id = {}
    node_records = []
    node_id_col = []

    def get_or_create_node(coord):
        coord = (round(coord[0], 6), round(coord[1], 6))
        if coord not in coord_to_id:
            nid = len(coord_to_id)
            coord_to_id[coord] = nid
            node_records.append({'node_id': nid, 'x': coord[0], 'y': coord[1]})
        return coord_to_id[coord]

    source_ids, target_ids = [], []

    for geom in link_attributes.geometry:
        if geom is None or geom.is_empty:
            source_ids.append(None)
            target_ids.append(None)
            continue
        parts = list(geom.geoms) if geom.geom_type == 'MultiLineString' else [geom]
        src = parts[0].coords[0]
        tgt = parts[-1].coords[-1]
        source_ids.append(get_or_create_node(src))
        target_ids.append(get_or_create_node(tgt))

    link_attributes['source_id'] = source_ids
    link_attributes['target_id'] = target_ids

    nodes = gpd.GeoDataFrame(
        node_records,
        geometry=gpd.points_from_xy(
            [n['x'] for n in node_records],
            [n['y'] for n in node_records]
        ),
        crs="EPSG:2056"
    ).set_index('node_id')

    # --- SAVE ---
    os.makedirs('data/Network/processed', exist_ok=True)
    nodes.to_file('data/Network/processed/nodes.gpkg', driver='GPKG')
    link_attributes.to_file('data/Network/processed/links.gpkg', driver='GPKG')

    return nodes, link_attributes

def import_osmnx_feeder_data():
    # Dübendorf - Hinwil corridor (bounding box in WGS84)
    left, bottom, right, top = 8.5800, 47.2700, 8.8700, 47.4200
    G = osmnx.graph_from_bbox(bbox=(left, bottom, right, top), network_type='bike', simplify=True)

    # Convert OSMnx graph to GeoDataFrames and reproject to Swiss LV95
    gdf_nodes, gdf_edges = osmnx.graph_to_gdfs(G)
    gdf_nodes = gdf_nodes.to_crs(epsg=2056)
    gdf_edges = gdf_edges.to_crs(epsg=2056)

    # Reset index so node IDs and edge u/v/key become regular columns
    gdf_nodes = gdf_nodes.reset_index()
    gdf_edges = gdf_edges.reset_index()

    # Keep only relevant columns for the feeder network
    node_cols = ['osmid', 'x', 'y', 'geometry']
    edge_cols = [
        'u', 'v', 'key',
        'osmid',
        'name',
        'highway',        # road/path type (cycleway, residential, etc.)
        'oneway',
        'length',         # edge length in meters (from OSMnx, WGS84-based)
        'maxspeed',
        'geometry',
    ]

    # Only keep columns that actually exist in the data (OSMnx output varies by area)
    gdf_nodes = gdf_nodes[[c for c in node_cols if c in gdf_nodes.columns]]
    gdf_edges = gdf_edges[[c for c in edge_cols if c in gdf_edges.columns]]

    # Recompute length in meters using projected CRS (more accurate than OSMnx default)
    gdf_edges['length_m'] = gdf_edges.geometry.length

    # Store original winding geometry as WKT for later visualization
    gdf_edges['visual_geom'] = gdf_edges.geometry.apply(lambda g: g.wkt)

    # Save outputs
    os.makedirs('data/Network/processed', exist_ok=True)
    gdf_nodes.to_file('data/Network/processed/osmnx_feeder_nodes.gpkg', driver='GPKG')
    gdf_edges.to_file('data/Network/processed/osmnx_feeder_edges.gpkg', driver='GPKG')

    return gdf_nodes, gdf_edges


def plot_network(
    network,
    source_col: str = None,
    target_col: str = None,
    directed: bool = False,
    node_color: str = "#4C9BE8",
    edge_color: str = "#888888",
    node_size: int = 10,          # small — there are thousands of nodes
    with_labels: bool = False,    # off by default for geo networks
    title: str = "Network Graph",
    figsize: tuple = (12, 9),
    layout: str = "spring",
) -> None:

    G = nx.DiGraph() if directed else nx.Graph()
    pos = {}

    # ── Case 1: GeoDataFrame with geometry ────────────────────────────────
    if isinstance(network, gpd.GeoDataFrame) and "geometry" in network.columns:

        if isinstance(source_col, str) and isinstance(target_col, str):
            # Explicit node-ID columns — use those
            for _, row in network.iterrows():
                G.add_edge(row[source_col], row[target_col])
            layout_fn = {
                "spring": nx.spring_layout, "circular": nx.circular_layout,
                "kamada_kawai": nx.kamada_kawai_layout, "spectral": nx.spectral_layout,
                "shell": nx.shell_layout, "random": nx.random_layout,
            }.get(layout, nx.spring_layout)
            pos = layout_fn(G, seed=42)

        else:
            # Derive nodes from LineString endpoints, snap to integer grid
            # to merge nearby points into the same node
            SNAP = 10  # meters — adjust if network has gaps

            def snap(xy):
                return (round(xy[0] / SNAP) * SNAP, round(xy[1] / SNAP) * SNAP)

            for _, row in network.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue
                lines = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
                for line in lines:
                    coords = list(line.coords)
                    u = snap(coords[0])
                    v = snap(coords[-1])
                    if u != v:
                        G.add_edge(u, v)
                        pos[u] = u  # x, y coords used directly as position
                        pos[v] = v

    # ── Case 2: plain list of tuples ──────────────────────────────────────
    elif isinstance(network, list):
        for edge in network:
            if len(edge) == 3:
                u, v, w = edge
                G.add_edge(u, v, weight=w)
            else:
                G.add_edge(*edge)
        pos = nx.spring_layout(G, seed=42)

    # ── Case 3: plain DataFrame with source/target columns ────────────────
    elif isinstance(network, pd.DataFrame) and source_col and target_col:
        for _, row in network.iterrows():
            G.add_edge(row[source_col], row[target_col])
        pos = nx.spring_layout(G, seed=42)

    else:
        raise TypeError(
            f"Cannot build graph from type {type(network)}. "
            "Pass a GeoDataFrame with geometry, a list of tuples, "
            "or a DataFrame with source_col/target_col."
        )

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Draw ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_color,
        width=0.8,
        alpha=0.6,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_color,
        node_size=node_size,
        alpha=0.85,
        ax=ax,
    )
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")   # keep geographic proportions correct
    ax.axis("off")
    plt.tight_layout()
    plt.show()