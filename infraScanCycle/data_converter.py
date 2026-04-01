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

os.chdir(r'/Users/ruki/PycharmProjects/infraScan/infraScanCycle')

def import_network_GIS_ALLTAG():
    path = 'data/raw/ALLTAG/OGD_VELO_ALLTAG_NETZ_L_M.shp'
    df = gpd.read_file(path, engine='pyogrio')

    # Ensure CRS is metric (Swiss LV95) — do this ONCE before anything else
    if df.crs.to_epsg() != 2056:
        df = df.to_crs("EPSG:2056")

    # Store original winding geometry as WKT for later visualization
    df['visual_geom'] = df['geometry'].apply(lambda g: g.wkt)

    # Save the actual winding length BEFORE simplifying to straight lines
    df['length_m'] = df.geometry.length

    # Simplify each geometry to a straight line between its endpoints
    def simplify_to_endpoints(geom):
        if geom is None or geom.is_empty:
            return None
        if isinstance(geom, MultiLineString):
            start_point = geom.geoms[0].coords[0]
            end_point = geom.geoms[-1].coords[-1]
        else:
            coords = list(geom.coords)
            start_point = coords[0]
            end_point = coords[-1]
        return LineString([start_point, end_point])

    df['geometry'] = df['geometry'].apply(simplify_to_endpoints)

    # Convert to primal graph and back to GeoDataFrames
    H = momepy.gdf_to_nx(df, approach='primal')
    gdf_nodes, gdf_edges = momepy.nx_to_gdf(H, points=True)

    # Save outputs
    os.makedirs('data/Network/processed', exist_ok=True)
    gdf_nodes.to_file('data/Network/processed/nodes.gpkg', driver='GPKG')
    gdf_edges.to_file('data/Network/processed/edges.gpkg', driver='GPKG')

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






