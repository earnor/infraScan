import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from rasterio.features import geometry_mask, rasterize as rio_rasterize
from scipy.stats.qmc import LatinHypercube
import re
import glob
import tkinter as tk
from tkinter.simpledialog import Dialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from shapely.geometry import shape, LineString, GeometryCollection
from shapely.ops import nearest_points, split
import fiona
from scipy.optimize import minimize
from tqdm import tqdm
import pulp
import os
import requests
import zipfile
import rasterio
import geopandas as gpd
from shapely.validation import make_valid
from shapely import wkt
from data_import import *


def get_development_candidates(edges_gdf, corridor_polygon=None, dev_type_filter=None):
    """Extract Netzlücken and Schwachstellen from the processed edge table,
    restrict them to the study corridor, and save development candidates.

    Each candidate keeps its edge geometry and gains:
      - ID_new        : sequential integer identifier
      - dev_type      : 'netzluecke' | 'schwachstelle'
      - within_corridor: True when the centroid lies inside corridor_polygon
      - on_border     : True when the edge crosses the corridor boundary
                        but its centroid is outside (counts as in-scope)
      - description   : human-readable action text used in logs and reports

    A representative-point file (centroids) is also written as
    generated_nodes.gpkg for downstream Voronoi / OD code.

    Parameters
    ----------
    edges_gdf        : GeoDataFrame  — attributed network edges from get_edge_attributes()
    corridor_polygon : shapely Polygon (EPSG:2056), optional
                       When provided, candidates are restricted to those that
                       intersect the polygon, and within_corridor / on_border
                       flags are set accordingly.  Pass innerboundary from main.py.
    """
    os.makedirs("data/Network/processed", exist_ok=True)

    routentyp_col   = next((c for c in edges_gdf.columns if c.upper().startswith('ROUTENTYP')),   None)
    planungstyp_col = next((c for c in edges_gdf.columns if c.upper().startswith('PLANUNGSTY')), None)

    netzluecken    = edges_gdf[edges_gdf['is_development']   == 1].copy().reset_index(drop=True)
    schwachstellen = edges_gdf[edges_gdf['is_schwachstelle'] == 1].copy().reset_index(drop=True)

    netzluecken['dev_type']    = 'netzluecke'
    schwachstellen['dev_type'] = 'schwachstelle'

    if dev_type_filter == 'netzluecke':
        candidates = netzluecken
    elif dev_type_filter == 'schwachstelle':
        candidates = schwachstellen
    else:
        candidates = pd.concat([netzluecken, schwachstellen], ignore_index=True)

    # ------------------------------------------------------------------
    # Corridor filter — keep only candidates that intersect the corridor
    # and tag within_corridor / on_border (fixes the KeyError in
    # travel_cost_developments and get_voronoi_all_developments)
    # ------------------------------------------------------------------
    if corridor_polygon is not None:
        corridor_gdf = gpd.GeoDataFrame({'geometry': [corridor_polygon]}, crs="EPSG:2056")
        n_before = len(candidates)
        candidates = gpd.sjoin(
            candidates, corridor_gdf, how='inner', predicate='intersects'
        ).drop(columns=['index_right'], errors='ignore').reset_index(drop=True)
        print(f"  Corridor filter: {n_before} → {len(candidates)} "
              f"(dropped {n_before - len(candidates)} outside corridor)")

        centroids = candidates.geometry.centroid
        candidates['within_corridor'] = centroids.within(corridor_polygon)
        candidates['on_border'] = (
            candidates.geometry.intersects(corridor_polygon.boundary)
            & ~candidates['within_corridor']
        )
    else:
        candidates['within_corridor'] = True
        candidates['on_border']       = False

    # TODO: ID_new is set from the DataFrame index after corridor filter and
    # reset_index(drop=True), so IDs are always 0…N-1.  This is fine for the
    # official candidates, but generate_connectivity_developments() derives its
    # start_id from max(ID_new)+1 here.  If this function is called twice (e.g.
    # in a re-run) without clearing the file, IDs may collide.  Consider
    # persisting a global ID counter or reading the max from disk.
    candidates['ID_new'] = candidates.index

    # ------------------------------------------------------------------
    # Human-readable description using official Velonetz Alltag attributes:
    #   Netzlücken  — VERBINDUNG (route name), RW_KEY_NR, PLANUNGSTY, ROUTENTYP
    #   Schwachstellen — NUMMER (official ID), VERBINDUNG, ROUTENTYP
    # Both are enriched by reformat_network() via spatial join with the
    # raw OGD shapefile and the Schwachstellen shapefile.
    # ------------------------------------------------------------------
    def _get(row, col, default=''):
        val = row[col] if col in row.index else default
        return str(val).strip() if str(val).strip() not in ('', 'nan', 'None') else default

    def _make_description(row):
        rtype    = _get(row, routentyp_col,   'Veloverbindung')
        ptype    = _get(row, planungstyp_col, '')
        name     = _get(row, 'verbindung',    '')
        route_nr = _get(row, 'rw_key_nr',     '')
        sw_nr    = _get(row, 'sw_nummer',     '')
        length_m = row['length_m'] if 'length_m' in row.index else row.geometry.length

        name_part  = f' "{name}"'         if name     else ''
        route_part = f" (Route {route_nr})" if route_nr else ''
        sw_part    = f" [{sw_nr}]"          if sw_nr   else ''

        if row['dev_type'] == 'netzluecke':
            return (f"BUILD {rtype}{name_part}{route_part}, {length_m:.0f} m — "
                    f"planned connection ({ptype}) to close network gap")
        else:
            return (f"UPGRADE {rtype}{sw_part}{name_part}{route_part}, {length_m:.0f} m — "
                    f"existing segment below quality standard (Schwachstelle)")

    candidates['description'] = candidates.apply(_make_description, axis=1)

    # ------------------------------------------------------------------
    # Per-candidate log + summary
    # ------------------------------------------------------------------
    print(f"\n  {'ID':>4}  {'Type':<15}  {'ROUTENTYP':<20}  {'Route':<12}  {'Length':>8}  {'Location':<10}  Name / Action")
    print(f"  {'-'*4}  {'-'*15}  {'-'*20}  {'-'*12}  {'-'*8}  {'-'*10}  {'-'*60}")
    for _, row in candidates.iterrows():
        rtype    = _get(row, routentyp_col, '?')
        route_nr = _get(row, 'rw_key_nr',  '—')
        length   = row['length_m'] if 'length_m' in row.index else row.geometry.length
        loc      = 'in corridor' if row['within_corridor'] else 'on border'
        name     = _get(row, 'verbindung', '')
        sw_nr    = _get(row, 'sw_nummer',  '')
        id_label = sw_nr if row['dev_type'] == 'schwachstelle' and sw_nr else route_nr
        print(f"  {row['ID_new']:>4}  {row['dev_type']:<15}  {rtype:<20}  {id_label:<12}  "
              f"{length:>7.0f}m  {loc:<10}  {name or row['description'][:60]}")

    nl     = (candidates['dev_type'] == 'netzluecke').sum()
    sw_cnt = (candidates['dev_type'] == 'schwachstelle').sum()
    lengths = (candidates['length_m'] if 'length_m' in candidates.columns
               else candidates.geometry.length)
    len_nl = lengths[candidates['dev_type'] == 'netzluecke'].sum()
    len_sw = lengths[candidates['dev_type'] == 'schwachstelle'].sum()

    print(f"\n  -> {nl} Netzlücken ({len_nl:.0f} m) + "
          f"{sw_cnt} Schwachstellen ({len_sw:.0f} m) = {len(candidates)} candidates in corridor\n")

    # ------------------------------------------------------------------
    # CSV report — full list of developments with all descriptive fields
    # ------------------------------------------------------------------
    os.makedirs('data/Network/processed', exist_ok=True)
    report_cols = ['ID_new', 'dev_type', routentyp_col, planungstyp_col,
                   'verbindung', 'rw_key_nr', 'sw_nummer', 'length_m',
                   'within_corridor', 'on_border', 'description']
    report_cols = [c for c in report_cols if c and c in candidates.columns]
    candidates[report_cols].to_csv(
        'data/Network/processed/developments_list.csv', index=False, encoding='utf-8-sig'
    )
    print(f"  -> Development list saved: data/Network/processed/developments_list.csv")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    candidates.to_file('data/Network/processed/development_candidates.gpkg', driver='GPKG')

    return candidates


def generate_connectivity_developments(
        corridor_polygon,
        protected_raster=r'data/landuse_landcover/processed/zone_no_infra/protected_area_corridor.tif',
        lake_shapefile=r'data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp',
        existing_candidates=None,
        max_gap_m=5000,
        include_netzluecken=False):
    """
    Find disconnected sub-networks in the existing corridor and generate new
    LineString edges — routed around protected areas — to close the gaps.

    Returns a GeoDataFrame with the same schema as get_development_candidates()
    (geometry, dev_type, ID_new, within_corridor, on_border, description,
    length_m, is_development).  Append this to the official candidates before
    scoring so the connectivity links are evaluated alongside Netzlücken.

    Parameters
    ----------
    corridor_polygon    : shapely Polygon (EPSG:2056)
    protected_raster    : path to the protected-area cost raster produced by
                          all_protected_area_to_raster(suffix='corridor')
    existing_candidates : GeoDataFrame from get_development_candidates — used
                          only to pick non-colliding ID_new values
    max_gap_m           : skip component pairs whose nearest nodes are further
                          apart than this (avoids implausible long bridges)
    """
    _EMPTY = gpd.GeoDataFrame(
        columns=['geometry', 'dev_type', 'ID_new', 'within_corridor',
                 'on_border', 'description', 'length_m', 'is_development'],
        crs="EPSG:2056"
    )

    print("\nConnectivity analysis:")

    # ── 1. Build undirected graph from the saved corridor edge files ──────────
    inside = gpd.read_file('data/Network/processed/edges_corridor.gpkg')
    border = gpd.read_file('data/Network/processed/edges_corridor_border.gpkg')
    edges_all = pd.concat([inside, border], ignore_index=True)
    # When include_netzluecken=True (called from the step-2 connectivity pass),
    # Netzlücken stay in the graph so bridges are only generated where a
    # Netzlücke does not already close the gap.
    if not include_netzluecken and 'is_development' in edges_all.columns:
        edges_all = edges_all[edges_all['is_development'] == False]

    G = nx.Graph()
    for _, row in edges_all.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        u = (round(coords[0][0], 1), round(coords[0][1], 1))
        v = (round(coords[-1][0], 1), round(coords[-1][1], 1))
        G.add_edge(u, v)

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    n_comp = len(components)
    sizes  = [len(c) for c in components]
    print(f"  Connected components: {n_comp}  "
          f"(largest: {sizes[0]} nodes, smallest: {sizes[-1]} nodes)")

    if n_comp <= 1:
        print("  Network is fully connected — no connectivity developments needed.")
        return _EMPTY

    # ── 2. Load the protected-area raster and build routing graph once ────────
    if not os.path.exists(protected_raster):
        print(f"  Warning: protected raster not found at {protected_raster} — using straight lines")
        routing_graph = None
        transform     = None
    else:
        with rasterio.open(protected_raster) as src:
            raster_data = src.read(1).astype(float)
            transform   = src.transform
            raster_crs  = src.crs
        raster_rows, raster_cols = raster_data.shape

        # Burn lake polygons into the cost raster so raster_to_graph() removes
        # those nodes — A* treats lakes as hard barriers, not just expensive cells.
        if lake_shapefile and os.path.exists(lake_shapefile):
            lake_gdf = gpd.read_file(lake_shapefile)
            if lake_gdf.crs != raster_crs:
                lake_gdf = lake_gdf.to_crs(raster_crs)
            lake_gdf['geometry'] = lake_gdf.geometry.apply(
                lambda g: make_valid(g) if g is not None else g)
            lake_gdf = lake_gdf[lake_gdf.geometry.notna() & ~lake_gdf.geometry.is_empty]
            if len(lake_gdf) > 0:
                lake_mask = rio_rasterize(
                    [(geom, 1) for geom in lake_gdf.geometry],
                    out_shape=(raster_rows, raster_cols),
                    transform=transform,
                    fill=0,
                    dtype='uint8',
                )
                n_lake = int((lake_mask > 0).sum())
                raster_data = np.where(lake_mask > 0, 1.0, raster_data)
                print(f"  Lake burned into cost raster: {n_lake} cells blocked")
        elif lake_shapefile:
            print(f"  Warning: lake shapefile not found at {lake_shapefile} — skipping lake avoidance")

        print("  Building routing graph from protected-area raster...")
        routing_graph = raster_to_graph(raster_data)

        # Build a cKDTree of all passable nodes so we can snap blocked
        # start/end pixels to the nearest passable cell before calling A*.
        from scipy.spatial import cKDTree as _cKDTree
        _passable_nodes = np.array(list(routing_graph.nodes()), dtype=np.int32)
        _passable_tree  = _cKDTree(_passable_nodes)

    # ── 3. Greedy nearest-pair connection (small components → main cluster) ───
    merged_nodes = list(components[0])
    # TODO: when existing_candidates is None or empty, start_id defaults to 100.
    # But official Netzlücken may already have ID_new values >= 100 (especially
    # if there are many candidates).  Replace the fallback with 0 and always
    # derive start_id from the actual max in development_candidates.gpkg so
    # IDs stay globally unique even if existing_candidates is not passed in.
    start_id = (
        int(existing_candidates['ID_new'].max()) + 1
        if existing_candidates is not None and len(existing_candidates) > 0
        else 100
    )

    new_rows      = []
    inaccessible  = []

    for comp_i, comp in enumerate(components[1:], start=1):
        comp_nodes = list(comp)

        # Closest node in this component to any node in the merged cluster
        best_dist = np.inf
        best_u = best_v = None
        for u_key in comp_nodes:
            ux, uy = u_key
            for v_key in merged_nodes:
                vx, vy = v_key
                d = ((ux - vx) ** 2 + (uy - vy) ** 2) ** 0.5
                if d < best_dist:
                    best_dist, best_u, best_v = d, u_key, v_key

        if best_dist > max_gap_m:
            print(f"  Component {comp_i} ({len(comp)} nodes): gap {best_dist:.0f} m "
                  f"> max_gap_m={max_gap_m} m — skipped")
            merged_nodes.extend(comp_nodes)
            continue

        print(f"  Component {comp_i} ({len(comp)} nodes): bridging gap of {best_dist:.0f} m")

        # Route from best_u → best_v avoiding protected areas
        if routing_graph is not None:
            def _clamp_px(xy):
                r, c = rasterio.transform.rowcol(transform, xs=xy[0], ys=xy[1])
                return (int(np.clip(r, 0, raster_rows - 1)),
                        int(np.clip(c, 0, raster_cols - 1)))

            def _snap_passable(px):
                """If px was removed (blocked), find the nearest passable node."""
                if routing_graph.has_node(px):
                    return px
                _, idx = _passable_tree.query(px)
                return tuple(_passable_nodes[idx])

            start_px = _snap_passable(_clamp_px(best_u))
            end_px   = _snap_passable(_clamp_px(best_v))
            if start_px is None or end_px is None or start_px == end_px:
                print(f"  Component {comp_i}: endpoints fully blocked — skipping bridge")
                merged_nodes.extend(comp_nodes)
                continue
            path, inaccessible = find_path(routing_graph, start_px, end_px,
                                           inaccessible, best_v)
            if path and len(path) >= 2:
                coords = [
                    rasterio.transform.xy(transform, rows=p[0], cols=p[1], offset='center')
                    for p in path
                ]
                coords[0]  = best_u
                coords[-1] = best_v
                coords = [c for j, c in enumerate(coords) if j == 0 or c != coords[j - 1]]
                geom = LineString(coords).simplify(25, preserve_topology=True) if len(coords) >= 2 \
                       else LineString([best_u, best_v])
            else:
                print(f"  Component {comp_i}: no obstacle-free path found — skipping bridge "
                      f"(gap would cross protected area or lake)")
                merged_nodes.extend(comp_nodes)
                continue
        else:
            # No routing graph available — skip rather than draw a straight line through obstacles
            print(f"  Component {comp_i}: no routing graph available — skipping bridge")
            merged_nodes.extend(comp_nodes)
            continue

        new_rows.append({
            'geometry':        geom,
            'ROUTENTYP':       'Nebenverbindung',  # cycling path type
            'dev_type':        'connectivity',
            'ID_new':          start_id + len(new_rows),
            'within_corridor': True,
            'on_border':       False,
            'length_m':        geom.length,
            'ffs':             15.0,
            'is_development':  1,
            'description':     (f"CONNECT sub-network ({len(comp)} nodes) to main corridor, "
                                f"{geom.length:.0f} m — auto-generated cycling connectivity link"),
        })

        merged_nodes.extend(comp_nodes)

    if not new_rows:
        print("  No connectivity developments generated.")
        return _EMPTY

    result = gpd.GeoDataFrame(new_rows, crs="EPSG:2056")
    total_m = result['length_m'].sum()
    print(f"  -> {len(result)} connectivity development(s) generated "
          f"(total {total_m:.0f} m)")

    out = 'data/Network/processed/connectivity_developments.gpkg'
    result.to_file(out, driver='GPKG')
    print(f"  -> Saved: {out}")

    return result


def generated_access_points(extent,number):
    e_min, n_min, e_max, n_max = extent.bounds
    e = int(e_max - e_min)
    n = int(n_max+100 - n_min+100)

    N = number

    engine = LatinHypercube(d=2, seed=42)  # seed=42
    sample = engine.random(n=N)

    n_sample = np.asarray(list(sample[:, 0]))
    e_sample = np.asarray(list(sample[:, 1]))

    n_gen = np.add(np.multiply(n_sample, n), int(n_min))
    e_gen = np.add(np.multiply(e_sample, e), int(e_min))

    idlist = list(range(0,N))
    gen_df = pd.DataFrame({"ID": idlist, "XKOORD": e_gen,"YKOORD":n_gen})
    gen_gdf = gpd.GeoDataFrame(gen_df,geometry=gpd.points_from_xy(gen_df.XKOORD,gen_df.YKOORD),crs="epsg:2056")

    return gen_gdf


def filter_access_points(gdf):

    newgdf = gdf.copy().reset_index(drop=True)
    print(f"  Total points generated: {len(newgdf)}")

    # Ensure CRS is EPSG:2056 before any spatial operation
    if newgdf.crs is None:
        newgdf = newgdf.set_crs("EPSG:2056")
    elif newgdf.crs.to_epsg() != 2056:
        newgdf = newgdf.to_crs("EPSG:2056")

    # ------------------------------------------------------------------
    # Helper: filter points that fall INSIDE a vector polygon layer
    # Returns the input GDF with points inside the polygon removed
    # ------------------------------------------------------------------
    def filter_by_polygon(points_gdf, shapefile_path, label):
        print(f"  Filtering: {label}")
        if not os.path.exists(shapefile_path):
            print(f"    Warning: {shapefile_path} not found — skipping")
            return points_gdf

        poly_gdf = gpd.read_file(shapefile_path)
        if poly_gdf.crs is None:
            poly_gdf = poly_gdf.set_crs("EPSG:2056")
        elif poly_gdf.crs.to_epsg() != 2056:
            poly_gdf = poly_gdf.to_crs("EPSG:2056")


        poly_gdf['geometry'] = poly_gdf.geometry.apply(lambda g: make_valid(g) if g is not None else g)
        poly_gdf = poly_gdf[poly_gdf.geometry.notna() & ~poly_gdf.geometry.is_empty].reset_index(drop=True)

        # sjoin: keep only points NOT inside the polygon
        joined = gpd.sjoin(points_gdf, poly_gdf[['geometry']], how='left', predicate='within')
        mask = joined['index_right'].isna()
        # Deduplicate in case a point touches multiple polygons
        mask = mask[~mask.index.duplicated(keep='first')]
        result = points_gdf[mask.values].copy().reset_index(drop=True)
        print(f"    Remaining: {len(result)}")
        return result

    # ------------------------------------------------------------------
    # 1. Schutzanordnung Natur und Landschaft
    # ------------------------------------------------------------------
    newgdf = filter_by_polygon(
        newgdf,
        "data/landuse_landcover/Schutzzonen/Schutzanordnungen_Natur_und_Landschaft_-SAO-_-OGD/FNS_SCHUTZZONE_F.shp",
        "Schutzanordnung Natur und Landschaft"
    )

    # ------------------------------------------------------------------
    # 2. Forest (Waldareal)
    # ------------------------------------------------------------------
    newgdf = filter_by_polygon(
        newgdf,
        "data/landuse_landcover/Schutzzonen/Waldareal_-OGD/WALD_WALDAREAL_F.shp",
        "Forest (Waldareal)"
    )

    # ------------------------------------------------------------------
    # 3. Network buffer — keep only points within 2500m of existing network
    #    Logic inverted: points OUTSIDE the buffer are dropped
    # ------------------------------------------------------------------
    print("  Filtering: Network proximity (within 2500m of existing network)")
    network_path = "data/Network/processed/edges.gpkg"
    if os.path.exists(network_path):
        network_gdf = gpd.read_file(network_path)
        if network_gdf.crs.to_epsg() != 2056:
            network_gdf = network_gdf.to_crs("EPSG:2056")

        network_buf = network_gdf.copy()
        network_buf['geometry'] = network_gdf.geometry.buffer(1000)


        network_buf['geometry'] = network_buf.geometry.apply(lambda g: make_valid(g) if g is not None else g)
        network_buf = network_buf[['geometry']].dissolve()

        # Keep points that ARE within the buffer (inside = good here)
        joined = gpd.sjoin(newgdf, network_buf, how='left', predicate='within')
        mask = joined['index_right'].notna()
        mask = mask[~mask.index.duplicated(keep='first')]
        newgdf = newgdf[mask.values].copy().reset_index(drop=True)
        print(f"    Remaining: {len(newgdf)}")
    else:
        print(f"    Warning: {network_path} not found — skipping")

    # ------------------------------------------------------------------
    # 4. Protected zones (Raster check)
    #    Drop points that land on protected raster cells (value > 0, not nodata)
    # ------------------------------------------------------------------
    print("  Filtering: Protected zones (Raster)")
    raster_path = "data/landuse_landcover/processed/zone_no_infra/protected_area_corridor.tif"

    if os.path.exists(raster_path):
        indices_to_drop = []
        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)
            nodata = src.nodata

            for idx, row in newgdf.iterrows():
                x, y = row.geometry.x, row.geometry.y
                try:
                    row_i, col_i = src.index(x, y)  # returns (row, col)
                    if 0 <= row_i < raster_data.shape[0] and 0 <= col_i < raster_data.shape[1]:
                        value = raster_data[row_i, col_i]
                        # Drop if protected (value != nodata and value > 0)
                        if nodata is not None and value == nodata:
                            pass  # nodata = not protected, keep
                        elif np.isnan(float(value)):
                            pass  # NaN = not protected, keep
                        elif value > 0:
                            indices_to_drop.append(idx)
                    else:
                        indices_to_drop.append(idx)  # outside raster extent = drop
                except Exception:
                    indices_to_drop.append(idx)

        newgdf = newgdf.drop(index=indices_to_drop).reset_index(drop=True)
        print(f"    Remaining: {len(newgdf)}")
    else:
        print(f"    Warning: {raster_path} not found — skipping")


    # ------------------------------------------------------------------
    # Cleanup and export
    # ------------------------------------------------------------------
    if 'ID' in newgdf.columns:
        newgdf = newgdf.rename(columns={"ID": "ID_new"})

    # Reassign clean sequential IDs after all filtering
    newgdf['ID_new'] = range(len(newgdf))
    newgdf = newgdf.drop(columns=['index'], errors='ignore')
    newgdf = newgdf.set_crs("EPSG:2056", allow_override=True)

    print(f"  Final count after all filters: {len(newgdf)}")
    return newgdf


def get_idx_todrop(pt, filename):
    #with fiona.open(r"data/landuse_landcover/landcover/lake/WB_STEHGEWAESSER_F.shp") as input:
    with fiona.open(filename, crs="epsg:2056") as input:
        #pt = newgdf.copy() #for testing
        idx = np.ones(len(pt))
        for feat in input:
            geom = shape(feat['geometry'])
            temptempidx = pt.within(geom)
            temptempidx = np.multiply(np.array(temptempidx), 1)
            tempidx = [i ^ 1 for i in temptempidx]
            #tempidx = np.multiply(np.array(tempidx),1)
            idx = np.multiply(idx, tempidx)
        intidx = [int(i) for i in idx]
        newidx = [i ^ 1 for i in intidx]
        #print(newidx)
    return newidx


def nearest(row, geom_union, df1, df2, geom1_col='geometry', geom2_col='geometry', src_column=None):
    """Find the nearest point and return the corresponding value from specified column."""

    # Find the geometry that is closest
    #nearest = df2[geom2_col] == nearest_points(row[geom1_col], geom_union)[1]
    nearest = df2[geom2_col] == nearest_points(geom_union,row[geom1_col])[1]

    # Get the corresponding value from df2 (matching is based on the geometry)
    value = df2[nearest][src_column].iloc[0]

    return value


def near(point, network_gdf,pts):
    # find the nearest point and return the corresponding Place value
    nearest = network_gdf.geometry == nearest_points(point, pts)[1]
    return network_gdf[nearest].geometry.iloc[0]


def connect_points_to_network(new_point_gdf, network_gdf, edges_gdf=None):
    import geopandas as gpd
    from shapely.geometry import LineString, Point
    from shapely.ops import nearest_points
    import os

    print("  Connecting candidate points to nearest cycling network nodes...")

    access_nodes = network_gdf.copy().reset_index(drop=True)
    if 'ID_point' not in access_nodes.columns:
        access_nodes['ID_point'] = access_nodes.index

    print(f"    {len(access_nodes)} access nodes available as connection targets")

    for df in [new_point_gdf, access_nodes]:
        drop_cols = [c for c in ['index_right', 'index_left'] if c in df.columns]
        df.drop(columns=drop_cols, inplace=True)

    # Nearest node join
    joined = gpd.sjoin_nearest(
        new_point_gdf.reset_index(drop=True),
        access_nodes[['ID_point', 'geometry']].reset_index(drop=True),
        how='left',
        distance_col='dist_to_node'
    ).drop(columns=['index_right'], errors='ignore')

    joined = joined[~joined.index.duplicated(keep='first')].reset_index(drop=True)

    node_lookup = access_nodes.set_index('ID_point')['geometry']
    joined['node_id']   = joined['ID_point']
    joined['node_geom'] = joined['node_id'].map(node_lookup)

    # Straight-line geometry (used for graph/routing)
    joined['geometry'] = joined.apply(
        lambda row: LineString([row.geometry, row['node_geom']])
        if row['node_geom'] is not None else None,
        axis=1
    )
    joined = joined[joined['geometry'].notnull()].copy()

    # Realistic visual path: new point → snap on nearest edge → node
    if edges_gdf is not None:
        from shapely.validation import make_valid
        edges_gdf = edges_gdf.copy()
        edges_gdf['geometry'] = edges_gdf.geometry.apply(lambda g: make_valid(g) if g is not None else g)
        edges_gdf = edges_gdf[edges_gdf.geometry.notna() & ~edges_gdf.geometry.is_empty]
        edges_union = edges_gdf.geometry.unary_union

        def make_visual_geom(orig_pt, node_geom):
            snap_pt = nearest_points(orig_pt, edges_union)[1]
            return LineString([orig_pt, snap_pt, node_geom])

        joined['visual_geom'] = joined.apply(
            lambda row: make_visual_geom(
                Point(row.geometry.coords[0]),
                row['node_geom']
            ).wkt,
            axis=1
        )
        print("    visual_geom: new point → edge snap → node (stored as WKT)")
    else:
        joined['visual_geom'] = joined['geometry'].apply(lambda g: g.wkt)
        print("    visual_geom: no edges_gdf provided — fallback to straight line")

    links = gpd.GeoDataFrame(joined, geometry='geometry', crs="EPSG:2056")
    links = links.drop(columns=['node_geom', 'ID_point'], errors='ignore')
    links['ID_link'] = range(len(links))

    os.makedirs('data/Network/processed', exist_ok=True)
    links.to_file('data/Network/processed/new_links.gpkg', driver='GPKG')

    print(f"  -> {len(links)} candidate links created")
    print(f"     distance range: {links['dist_to_node'].min():.0f}–{links['dist_to_node'].max():.0f} m")

    return links


def plot_connections(new_point_gdf, network_gdf, links, polygon=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, show_dist in zip(axes, [False, True]):
        if polygon:
            import geopandas as gpd
            gpd.GeoDataFrame({'geometry': [polygon]}, crs="EPSG:2056").boundary.plot(
                ax=ax, color='black', linewidth=1.5, linestyle='--', zorder=1)

        # Network nodes
        network_gdf.plot(ax=ax, color='steelblue', markersize=8, alpha=0.6, zorder=2)

        # Connector links
        if show_dist:
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            norm = mcolors.Normalize(vmin=links['dist_to_node'].min(),
                                     vmax=links['dist_to_node'].max())
            cmap = cm.RdYlGn_r
            for _, row in links.iterrows():
                gpd.GeoDataFrame([row], crs=links.crs).plot(
                    ax=ax, color=[cmap(norm(row['dist_to_node']))], linewidth=1, alpha=0.7, zorder=3)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label='Distance to network (m)', shrink=0.6)
            ax.set_title('Connector Links — coloured by distance', fontsize=11)
        else:
            links.plot(ax=ax, color='orange', linewidth=0.8, alpha=0.7, zorder=3)
            ax.set_title('Connector Links — new points to network', fontsize=11)

        # New development points
        new_point_gdf.plot(ax=ax, color='red', markersize=15, marker='*',
                           alpha=0.9, zorder=4)

        legend = [
            Line2D([0], [0], color='steelblue', marker='o', linestyle='None',
                   markersize=6, label=f'Network nodes ({len(network_gdf)})'),
            Line2D([0], [0], color='red', marker='*', linestyle='None',
                   markersize=10, label=f'New points ({len(new_point_gdf)})'),
            Line2D([0], [0], color='orange', linewidth=1.5,
                   label=f'Connector links ({len(links)})'),
        ]
        ax.legend(handles=legend, fontsize=8)
        ax.set_aspect('equal')

    plt.suptitle('Last-Mile Connections to Cycling Network', fontsize=13)
    plt.tight_layout()
    plt.savefig('data/Network/processed/connections_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved → data/Network/processed/connections_plot.png")


def build_combined_network(nodes_gdf, edges_gdf, generated_points, routed_links):
    """
    Saves the status-quo ALLTAG network and builds a combined network
    that merges existing nodes/edges with generated points and new routed links.

    Inputs:
        nodes_gdf        – existing network nodes (from reformat_network)
        edges_gdf        – existing edges with attributes (from get_edge_attributes)
        generated_points – filtered generated development points
        routed_links     – new routed connector links (from routing_raster, filtered)

    Outputs (saved to data/Network/):
        status_quo/nodes.gpkg + edges.gpkg   – original network
        combined/nodes.gpkg  + edges.gpkg    – merged network
    """
    import geopandas as gpd
    import pandas as pd
    import os

    os.makedirs('data/Network/status_quo', exist_ok=True)
    os.makedirs('data/Network/combined', exist_ok=True)

    # ------------------------------------------------------------------
    # 1. SAVE STATUS QUO as-is
    # ------------------------------------------------------------------
    nodes_gdf.to_file('data/Network/status_quo/nodes.gpkg', driver='GPKG')
    edges_gdf.to_file('data/Network/status_quo/edges.gpkg', driver='GPKG')
    print(f"  Status quo saved: {len(nodes_gdf)} nodes, {len(edges_gdf)} edges")

    # ------------------------------------------------------------------
    # 2. ASSIGN EDGE ATTRIBUTES TO NEW LINKS
    #    New links = Nebenverbindung equivalent (shared paths, low speed)
    # ------------------------------------------------------------------
    NEW_LINK_ATTRS = {'ffs': 12, 'capacity': 200, 'oneway': 0}

    new_edges = routed_links.copy()

    new_edges['ffs'] = NEW_LINK_ATTRS['ffs']
    new_edges['capacity'] = NEW_LINK_ATTRS['capacity']
    new_edges['oneway'] = NEW_LINK_ATTRS['oneway']
    new_edges['ROUTENTYP'] = 'New Link'

    # Travel time from routed length (prefer length_routed_m, fallback to geometry)
    if 'length_routed_m' in new_edges.columns:
        new_edges['length_m'] = new_edges['length_routed_m']
    else:
        new_edges['length_m'] = new_edges.geometry.length

    new_edges['tt_min'] = (new_edges['length_m'] / 1000) / new_edges['ffs'] * 60

    # Use visual_geom as geometry if available (routed path shape)
    if 'visual_geom' in new_edges.columns:
        from shapely import wkt
        new_edges['geometry'] = new_edges['visual_geom'].apply(
            lambda g: wkt.loads(g) if isinstance(g, str) else g
        )
        new_edges = gpd.GeoDataFrame(new_edges, geometry='geometry', crs="EPSG:2056")

    # ------------------------------------------------------------------
    # 3. COMBINE NODES: existing + generated points
    # ------------------------------------------------------------------
    gen_nodes = generated_points.copy()
    gen_nodes['x'] = gen_nodes.geometry.x
    gen_nodes['y'] = gen_nodes.geometry.y
    gen_nodes['is_intersection'] = 0
    gen_nodes['is_endpoint'] = 1
    gen_nodes['degree'] = 1

    id_offset = int(nodes_gdf['ID_point'].max()) + 1 if 'ID_point' in nodes_gdf.columns else len(nodes_gdf)
    gen_nodes['ID_point'] = range(id_offset, id_offset + len(gen_nodes))
    gen_nodes['source'] = 'generated'

    existing_nodes = nodes_gdf.copy()
    existing_nodes['source'] = 'existing'

    # Deduplicate columns before concat
    existing_nodes = existing_nodes.loc[:, ~existing_nodes.columns.duplicated()]
    gen_nodes = gen_nodes.loc[:, ~gen_nodes.columns.duplicated()]

    shared_cols = list(dict.fromkeys(
        c for c in existing_nodes.columns if c in gen_nodes.columns
    ))  # preserves order, deduped

    combined_nodes = pd.concat(
        [existing_nodes[shared_cols], gen_nodes[shared_cols]],
        ignore_index=True
    )
    combined_nodes = gpd.GeoDataFrame(combined_nodes, geometry='geometry', crs="EPSG:2056")

    # ------------------------------------------------------------------
    # 4. COMBINE EDGES: existing + new links
    # ------------------------------------------------------------------
    # Keep only columns present in both (+ fill missing with NaN)
    existing_edges = edges_gdf.copy()
    existing_edges['source'] = 'existing'
    new_edges['source'] = 'new_link'

    combined_edges = pd.concat([existing_edges, new_edges], ignore_index=True)
    combined_edges = gpd.GeoDataFrame(combined_edges, geometry='geometry', crs="EPSG:2056")
    combined_edges['ID_edge'] = range(len(combined_edges))

    # ------------------------------------------------------------------
    # 5. SAVE COMBINED NETWORK
    # ------------------------------------------------------------------
    combined_nodes.to_file('data/Network/combined/nodes.gpkg', driver='GPKG')
    combined_edges.to_file('data/Network/combined/edges.gpkg', driver='GPKG')

    combined_nodes.drop(columns='geometry').to_csv('data/Network/combined/nodes.csv', index=False)
    combined_edges.drop(columns='geometry').to_csv('data/Network/combined/edges.csv', index=False)

    print(f"  Combined network saved:")
    print(f"    Nodes: {len(existing_nodes)} existing + {len(gen_nodes)} generated = {len(combined_nodes)}")
    print(f"    Edges: {len(existing_edges)} existing + {len(new_edges)} new links = {len(combined_edges)}")
    print(f"    New link ffs: {NEW_LINK_ATTRS['ffs']} km/h | capacity: {NEW_LINK_ATTRS['capacity']} bikes/h")
    print(f"    New link tt_min range: {new_edges['tt_min'].min():.1f}–{new_edges['tt_min'].max():.1f} min")

    return combined_nodes, combined_edges


def create_nearest_gdf(filtered_rand_gdf):
    nearest_gdf = filtered_rand_gdf[["ID_new", "ID_point", "geometry_current"]].set_geometry("geometry_current")
    #nearest_gdf = nearest_gdf.rename({"ID":"PointID", "index_right":"NearestAccID"})
    #nearest_df = filtered_rand_gdf.assign(PointID=filtered_rand_gdf["ID"],NearestAccID=filtered_rand_gdf["index_right"],x=filtered_rand_gdf["x"],y=filtered_rand_gdf["y"])
    #nearest_gdf = gpd.GeoDataFrame(nearest_df,geometry=gpd.points_from_xy(nearest_df.x,nearest_df.y),crs="epsg:2056")
    return nearest_gdf


def create_lines(rand_pts_gdf, nearest_highway_pt_gdf):
    rand_pts_gdf = rand_pts_gdf.sort_values(by="ID_new")
    points = rand_pts_gdf.geometry
    nearest_highway_pt_gdf = nearest_highway_pt_gdf.sort_values(by="ID_new")
    nearest_points = nearest_highway_pt_gdf.geometry

    line_geometries = [LineString([points.iloc[i], nearest_points.iloc[i]]) for i in range(len(rand_pts_gdf))]
    line_gdf = gpd.GeoDataFrame(geometry=line_geometries)
    line_gdf["ID_new"] = rand_pts_gdf["ID_new"]
    line_gdf["ID_current"] = nearest_highway_pt_gdf["ID_point"]

    line_gdf = line_gdf.set_crs("epsg:2056")
    line_gdf.to_file(r"data/Network/processed/new_links.gpkg")
    return


def plot_lines_to_network(points_gdf,lines_gdf):
    points_gdf.plot(marker='*', color='green', markersize=5)
    base = lines_gdf.plot(edgecolor='black')
    points_gdf.plot(ax=base, marker='o', color='red', markersize=5)
    plt.savefig(r"plot/predict/230822_network-generation.png", dpi=300)
    return None


def line_scoring(lines_gdf,raster_location):
    # Load your raster file using rasterio
    raster_path = raster_location
    with rasterio.open(raster_path) as src:
        raster = src.read(1)  # Assuming it's a single-band raster

    # Create an empty list to store the sums
    sums = []

    # Iterate over each line geometry in the GeoDataFrame
    for idx, line in lines_gdf.iterrows():
        mask = geometry_mask([line['geometry']], out_shape=raster.shape, transform=src.transform, invert=False)
        line_sum = raster[mask].sum()
        sums.append(line_sum)

    # Add the sums as a new column to the GeoDataFrame
    lines_gdf['raster_sum'] = sums

    return lines_gdf


def routing_raster(raster_path, links_path='data/Network/processed/new_links_corridor.gpkg'):

    if not os.path.exists(links_path):
        raise FileNotFoundError(f"Missing: {links_path} — run connect_points_to_network() first")

    generated_links = gpd.read_file(links_path)
    print(f"  Routing {len(generated_links)} candidate links...")

    new_geometries      = []
    inaccessible_points = []

    with rasterio.open(raster_path) as dataset:
        raster_data = dataset.read(1)
        transform   = dataset.transform

        print("  Building routing graph from raster...")
        graph = raster_to_graph(raster_data)

        for i, row in generated_links.iterrows():
            line        = row.geometry
            start_point = line.coords[0]
            end_point   = line.coords[-1]

            start_idx = rasterio.transform.rowcol(
                transform, xs=start_point[0], ys=start_point[1]
            )
            end_idx = rasterio.transform.rowcol(
                transform, xs=end_point[0], ys=end_point[1]
            )

            path = None
            try:
                path, inaccessible_points = find_path(
                    graph, start_idx, end_idx, inaccessible_points, end_point
                )
            except Exception as e:
                path = None

            # ----------------------------------------------------------
            # GUARD: path must have at least 2 points for a LineString
            # A single-point path means start == end (same raster cell)
            # or find_path returned a degenerate result
            # ----------------------------------------------------------
            if path and len(path) >= 2:
                coords = [
                    rasterio.transform.xy(transform, rows=p[0], cols=p[1], offset='center')
                    for p in path
                ]
                # Snap to exact original coordinates (avoids 25m raster offset)
                coords[0]  = start_point
                coords[-1] = end_point

                # Final guard: deduplicate consecutive identical points
                coords = [c for j, c in enumerate(coords)
                          if j == 0 or c != coords[j - 1]]

                if len(coords) >= 2:
                    new_geometries.append(LineString(coords))
                else:
                    new_geometries.append(None)

            elif path and len(path) == 1:
                # Start and end in the same raster cell — use straight line
                if start_point != end_point:
                    new_geometries.append(LineString([start_point, end_point]))
                else:
                    new_geometries.append(None)
            else:
                new_geometries.append(None)

    # ------------------------------------------------------------------
    # Update geometries and clean up
    # ------------------------------------------------------------------
    generated_links = generated_links.copy()
    generated_links['geometry'] = new_geometries

    routed  = generated_links.dropna(subset=['geometry']).copy()
    routed  = gpd.GeoDataFrame(routed, geometry='geometry', crs="EPSG:2056")
    routed['length_routed_m'] = routed.geometry.length

    dropped = len(generated_links) - len(routed)
    print(f"  -> {len(routed)} routed links "
          f"({dropped} dropped — no valid path found)")
    if len(routed) > 0:
        print(f"     routed length range: "
              f"{routed['length_routed_m'].min():.0f}–"
              f"{routed['length_routed_m'].max():.0f} m")

    os.makedirs('data/Network/processed', exist_ok=True)
    routed.to_file('data/Network/processed/new_links_realistic.gpkg', driver='GPKG')

    if inaccessible_points:
        pd.DataFrame(inaccessible_points, columns=['x', 'y']) \
          .to_csv('data/Network/processed/points_inaccessible.csv', index=False)
        print(f"  -> {len(inaccessible_points)} inaccessible points logged")

    return routed


# NOTE: OSM_network.py also defines a raster_to_graph() that converts a speed
# raster (km/h) into a weighted graph for Dijkstra travel-time.  This version
# is different: it takes a binary protected-area raster and assigns a high
# penalty weight to protected cells (for routing around them).
# TODO: the two versions have diverged and serve different purposes but share
# the same name.  Rename this one (e.g. raster_to_routing_graph) to avoid
# confusion when both modules are imported with *.
def raster_to_graph(raster_data):
    rows, cols = raster_data.shape
    graph = nx.grid_2d_graph(rows, cols)

    # Add diagonal edges
    graph.add_edges_from([
        ((x, y), (x + 1, y + 1))
        for x in range(cols - 1)
        for y in range(rows - 1)
    ] + [
        ((x + 1, y), (x, y + 1))
        for x in range(cols - 1)
        for y in range(rows - 1)
    ], weight=1.4)

    # Remove nodes for blocked cells entirely — hard barrier, A* cannot pass through
    for y in range(rows):
        for x in range(cols):
            if raster_data[y, x] > 0 and graph.has_node((y, x)):
                graph.remove_node((y, x))

    return graph


def find_path(graph, start, end, list_no_path, point_end):
    try:
        def heuristic(a, b):
            return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        path = nx.astar_path(graph, start, end, heuristic=heuristic, weight='weight')
        return path, list_no_path
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        list_no_path.append(point_end)
        print(f"  No obstacle-free path: {point_end}")
        return None, list_no_path



def plot_corridor(network, limits, location, current_nodes=False, new_nodes=False, new_links=False, access_link=False):

    fig, ax = plt.subplots(figsize=(10, 10))

    network = network[(network["Rank"] == 1) & (network["Opening Ye"] < 2023) & (network["NAME"] != 'Freeway Tunnel planned') & (
                network["NAME"] != 'Freeway planned')]

    # Define square to show perimeter of investigation
    square = Polygon([(limits[0], limits[2]), (limits[1], limits[2]), (limits[1], limits[3]), (limits[0], limits[3])])
    frame = gpd.GeoDataFrame(geometry=[square], crs=network.crs)

    #df_voronoi.plot(ax=ax, facecolor='none', alpha=0.2, edgecolor='k')

    if access_link==True:
        access = network[network["NAME"] == "Freeway access"]
        access["point"] = access.representative_point()
        access.plot(ax=ax, color="red", markersize=50)

    if isinstance(new_links, gpd.GeoDataFrame):
        new_links.plot(ax=ax, color="darkgray")

    if isinstance(new_nodes, gpd.GeoDataFrame):
        new_nodes.plot(ax=ax, color="blue", markersize=50)

    network.plot(ax=ax, color="black", lw=4)

    if isinstance(current_nodes, gpd.GeoDataFrame):
        current_nodes.plot(ax=ax, color="black", markersize=50)

    # Plot the location as points
    location.plot(ax=ax, color="black", markersize=75)
    # Add city names to the plot
    for idx, row in location.iterrows():
        plt.annotate(row['location'], xy=row["geometry"].coords[0], ha='left', va="bottom", fontsize=15)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])

    #plt.title("Voronoi polygons to each highway access point")
    plt.savefig(r"plot/network_base_generated.png", dpi=300)
    plt.show()

    return


def single_tt_voronoi_ton_one(folder_path):

    # List all gpkg files in the folder
    gpkg_files = [f for f in os.listdir(folder_path) if f.endswith('Voronoi.gpkg')]

    # Initialize an empty list to store dataframes
    dataframes = []

    for file in gpkg_files:
        # Read the gpkg file
        gdf = gpd.read_file(os.path.join(folder_path, file))

        # Use regular expression to extract the XXX number from the filename
        id_development = re.search(r'dev(\d+)_Voronoi', file)
        if id_development:
            id_development = int(id_development.group(1))
        else:
            print("Error in predict >> 394")
            continue  # Skip file if no match is found

        # Add the ID_development as a new column
        gdf['ID_development'] = id_development

        # Append the dataframe to the list
        dataframes.append(gdf)

    # Concatenate all dataframes into one
    combined_gdf = pd.concat(dataframes)

    # Save the combined dataframe as a new gpkg file
    combined_gdf.to_file("data/Voronoi/combined_developments.gpkg", driver="GPKG")


def import_elevation_model(new_resolution):

    # Read CSV file containing the ZIP file links
    csv_file = r"data/elevation_model/ch.swisstopo.swissalti3d-pivq0Jb7.csv"
    df = pd.read_csv(csv_file, names=["url"], header=None)

    # Download and extract ZIP files
    for url in df["url"]:
        r = requests.get(url)
        zip_path = r"data/elevation_model/zip_files/temp.zip"
        with open(zip_path, 'wb') as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(r"data/elevation_model/extracted_xyz_files")

    # Find all XYZ files
    xyz_files = glob.glob(r"data/elevation_model/extracted_xyz_files/*.xyz")

    # Calculate the minimum coordinates based on the first file
    sample_data = pd.read_csv(xyz_files[0], sep=" ")
    min_x, min_y = sample_data['X'].min(), sample_data['Y'].min()

    # Process each file — collect into list, concat once to avoid O(n²) copies
    chunks = []
    for i, file in enumerate(xyz_files, start=1):
        chunks.append(downsample_elevation_xyz_file(file, min_x, min_y, resolution=new_resolution))
        print(f"Processed file {i}/{len(xyz_files)}: {file}")
    concatenated_data = pd.concat(chunks, ignore_index=True)
    del chunks
    print(concatenated_data.shape)

    # Convert the DataFrame to a 2D grid
    min_x, max_x = concatenated_data['X'].min(), concatenated_data['X'].max()
    min_y, max_y = concatenated_data['Y'].min(), concatenated_data['Y'].max()

    # Calculate the number of rows and columns
    cols = int((max_x - min_x) / new_resolution) + 1
    rows = int((max_y - min_y) / new_resolution) + 1

    # Create an empty grid
    raster = np.full((rows, cols), np.nan)

    # Populate the grid with Z values
    for _, row in concatenated_data.iterrows():
        col_idx = int((row['X'] - min_x) / new_resolution)
        row_idx = int((max_y - row['Y']) / new_resolution)
        raster[row_idx, col_idx] = row['Z']

    # Define the georeferencing transform
    transform = from_origin(min_x, max_y, new_resolution, new_resolution)

    # Write the data to a GeoTIFF file
    with rasterio.open(r'data/elevation_model/elevation.tif', 'w', driver='GTiff',
                       height=raster.shape[0], width=raster.shape[1],
                       count=1, dtype=str(raster.dtype),
                       crs='EPSG:2056', transform=transform) as dst:
        dst.write(raster, 1)

    return


def downsample_elevation_xyz_file(file_path, min_x, min_y, resolution):
    # Read the file
    data = pd.read_csv(file_path, sep=" ")

    # Filter the data
    filtered_data = data[((data['X'] - min_x) % resolution == 0) & ((data['Y'] - min_y) % resolution == 0)]

    return filtered_data


def get_cycling_elevation_profile():
    """
    Computes the elevation profile for each routed cycling link and derives
    slope statistics relevant for cycling infrastructure assessment.

    Key cycling-specific logic:
    - Slope > 4% significantly increases perceived travel time (TODO in travel time module)
    - No tunnel/bridge classification (rare/irrelevant for cycling)
    - Outputs slope metrics used downstream in construction cost + travel time scoring

    Output saved to: data/Network/processed/development_candidates_elevation.gpkg
    Returns: GeoDataFrame with slope attributes added
    """
    import rasterio
    import numpy as np

    links = gpd.read_file(r"data/Network/processed/development_candidates.gpkg")
    elevation_raster = r"data/elevation_model/elevation.tif"
    sampling_interval = 50  # meters — matches raster resolution

    def interpolate_linestring(linestring, interval):
        length = linestring.length
        num_points = max(2, int(np.ceil(length / interval)))
        distances = np.linspace(0, length, num_points)
        return [linestring.interpolate(d) for d in distances]

    def sample_raster_at_points(points, raster_data, transform):
        """Sample elevation values; return NaN for out-of-bounds points."""
        values = []
        for point in points:
            try:
                row, col = rasterio.transform.rowcol(transform, point.x, point.y)
                value = raster_data[row, col]
                values.append(float(value))
            except (IndexError, Exception):
                values.append(np.nan)
        return values

    with rasterio.open(elevation_raster) as raster:
        print(f"Elevation raster CRS: {raster.crs}")
        print(f"Links CRS:            {links.crs}")
        assert str(raster.crs) == str(links.crs), \
            "CRS mismatch between elevation raster and links — reproject before proceeding."

        raster_data = raster.read(1).astype(float)
        transform = raster.transform

        links['elevation_profile'] = links['geometry'].apply(
            lambda geom: sample_raster_at_points(
                interpolate_linestring(geom, sampling_interval),
                raster_data,
                transform
            )
        )

    # --- Slope computation ---
    # elevation_difference: rise between successive 50m samples [m]
    # slope_pct: gradient in percent [%] = (rise / run) * 100
    links['slope_pct'] = links['elevation_profile'].apply(
        lambda profile: (np.abs(np.diff(np.array(profile, dtype=float))) / sampling_interval * 100).tolist()
        if len(profile) >= 2 else []
    )

    # Mean slope along the link [%]
    links['slope_mean_pct'] = links['slope_pct'].apply(
        lambda s: float(np.nanmean(s)) if len(s) > 0 else np.nan
    )

    # Max slope along the link [%]
    links['slope_max_pct'] = links['slope_pct'].apply(
        lambda s: float(np.nanmax(s)) if len(s) > 0 else np.nan
    )

    # Share of segments exceeding 4% slope threshold [0–1]
    # Cycling comfort drops significantly above 4% (VSS / ARE standard)
    SLOPE_THRESHOLD_PCT = 4.0
    links['share_steep_4pct'] = links['slope_pct'].apply(
        lambda s: float(np.mean(np.array(s) > SLOPE_THRESHOLD_PCT)) if len(s) > 0 else np.nan
    )

    # Flag links that likely need closer review for cycling suitability:
    # - mean slope above threshold, OR
    # - more than 30% of segments are steep
    links['check_slope'] = (
        (links['slope_mean_pct'] > SLOPE_THRESHOLD_PCT) |
        (links['share_steep_4pct'] > 0.30)
    )

    # Serialize profile + slope lists for GeoPackage storage (no list dtype support)
    links['elevation_profile'] = links['elevation_profile'].apply(str)
    links['slope_pct'] = links['slope_pct'].apply(str)

    links.to_file(r"data/Network/processed/development_candidates_elevation.gpkg", driver="GPKG")

    flagged = links['check_slope'].sum()
    print(f"  Elevation profiles computed for {len(links)} links")
    print(f"  Links flagged for slope review (check_slope=True): {flagged} ({100*flagged/len(links):.0f}%)")

    return links


def get_tunnel_candidates(df):
    print("You will have to define the needed tunnels and bridges for ", df["check_needed"].sum() , " section.")

    df["elevation_profile"] = df["elevation_profile"].astype("object")
    # Custom dialog class for pop-up
    class CustomDialog(Dialog):
        def __init__(self, parent, row):
            self.row = row
            Dialog.__init__(self, parent)

        def body(self, master):
            # Create a figure for the plot
            self.fig, self.ax = plt.subplots()
            x_values = np.arange(0, len(self.row['elevation_profile'])) * 50
            self.ax.plot(x_values, self.row['elevation_profile'])
            self.ax.set_title('Elevation profile')
            self.ax.set_xlabel('Distance (m)')
            self.ax.set_ylabel('Elevation (m. asl.)')

            # Create labels and input fields for questions
            tk.Label(master, text="How much tunnel is required in meters:").pack()
            self.tunnel_len_entry = tk.Entry(master)
            self.tunnel_len_entry.pack()

            tk.Label(master, text="How much bridge is required in meters:").pack()
            self.bridge_len_entry = tk.Entry(master)
            self.bridge_len_entry.pack()

            # Create a canvas to display the plot
            canvas = FigureCanvasTkAgg(self.fig, master=master)
            canvas.get_tk_widget().pack()

        def apply(self):
            # Get the user's input values
            tunnel_len = int(self.tunnel_len_entry.get())
            bridge_len = int(self.bridge_len_entry.get())

            # Update DataFrame with user's input
            df.at[self.row.name, 'tunnel_len'] = tunnel_len
            df.at[self.row.name, 'bridge_len'] = bridge_len
    # Create new columns for user input
    df['tunnel_len'] = None
    df['bridge_len'] = None


    # Iterate through the DataFrame and show the custom pop-up for rows with 'check_needed' set to True
    for index, row in df.iterrows():
        if row['check_needed']:
            root = tk.Tk()
            root.withdraw()
            dlg = CustomDialog(root, row)
            #dlg.wait_window()
    df["elevation_profile"]=df["elevation_profile"].astype('string')
    print(df)
    df.to_file(r"data/Network/processed/new_links_realistic_tunnel-terminal.gpkg")



