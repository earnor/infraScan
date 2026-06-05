"""
infraScanRail - New Main Pipeline
Last modified: 2026-05-23

Orchestrates the full pipeline: Phase 1 (Initialisation), Phase 2 (Data Preparation),
Phase 3A (Infrastructure), Phase 3B (Services), Phase 3C (Capacity).
"""


import os
import sys
import time
import warnings
import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd

import paths
import settings
from catchment_base import (
    _dissolve_admin_polygon,
    _export_gpkg,
    _validate_containment,
    STUDY_AREA_DEFAULT_BUFFER_M,
    CATCHMENT_AREA_DEFAULT_BUFFER_M,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineConfig:
    """Holds resolved pipeline state set during Phase 1."""

    def __init__(self):
        self.grouping_strategy     = settings.CAPACITY_GROUPING_STRATEGY
        self.needs_projection      = False   # set True if svc version needs projection
        self.infra_version         = None    # resolved in phase_1_initialisation
        self.svc_version           = None    # resolved in phase_1_initialisation
        self._original_input       = None    # stored in infrascanrail_new for smart_input


PIPELINE_CONFIG = PipelineConfig()


def get_catchment_od_method() -> str:
    """Return the OD allocation method that matches the active catchment method.

    Municipal  -> 'municipal'       (commune-to-station lookup table)
    PT_Feeder  -> 'feeder_weighted' (communal OD re-weighted by feeder catchment shares)
    """
    if settings.CATCHMENT_METHOD == 'Municipal':
        return 'municipal'
    return 'feeder_weighted'


def get_routing_od_method() -> str:
    """Translate the settings OD method to the internal string used by
    catchment_OD_rail_network.route_od_matrices() when selecting W3 CSV files.

    'feeder_weighted' -> 'pt_feeder'   (reads the PT_Feeder station OD matrices)
    'municipal'       -> 'municipal'   (reads the Municipal station OD matrices)
    """
    return 'pt_feeder' if get_catchment_od_method() == 'feeder_weighted' else 'municipal'


# ═══════════════════════════════════════════════════════════════════════════════
# Study / Catchment area helpers  (Option X -no input() patching)
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_study_area():
    """Build study area polygon from settings without interactive prompts.

    Returns (polygon, buffered_polygon, admin_level, primary_names, buffer_m).
    """
    if settings.STUDY_AREA_METHOD == 'coordinates':
        polygon       = settings.perimeter_infra_generation
        admin_level   = 'coordinates'
        primary_names = []
    else:
        polygon = _dissolve_admin_polygon(
            settings.STUDY_AREA_ADMIN_LEVEL,
            settings.STUDY_AREA_ADMIN_NAMES,
            settings.STUDY_AREA_ADMIN_SUBDIVISIONS,
        )
        admin_level   = settings.STUDY_AREA_ADMIN_LEVEL
        primary_names = settings.STUDY_AREA_ADMIN_NAMES

    buffer_m = settings.STUDY_AREA_BUFFER_M
    return polygon, polygon.buffer(buffer_m), admin_level, primary_names, buffer_m


def _resolve_catchment_area():
    """Build catchment area polygon from settings without interactive prompts.

    Returns (polygon, buffered_polygon, admin_level, primary_names, buffer_m).
    """
    polygon = _dissolve_admin_polygon(
        settings.CATCHMENT_AREA_ADMIN_LEVEL,
        settings.CATCHMENT_AREA_ADMIN_NAMES,
        settings.CATCHMENT_AREA_ADMIN_SUBDIVISIONS,
    )
    buffer_m = settings.CATCHMENT_AREA_BUFFER_M
    return polygon, polygon.buffer(buffer_m), settings.CATCHMENT_AREA_ADMIN_LEVEL, \
           settings.CATCHMENT_AREA_ADMIN_NAMES, buffer_m


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 -Initialisation
# ═══════════════════════════════════════════════════════════════════════════════

def phase_1_initialisation(runtimes: dict) -> tuple:
    """Resolve study/catchment areas, validate version settings, print run summary.

    Args:
        runtimes: Dict tracking phase execution times.

    Returns:
        (sa_boundary, sa_buffer, ca_boundary, ca_buffer) -Shapely polygons.
    """
    print("\n" + "=" * 80)
    print("PHASE 1: INITIALISATION")
    print("=" * 80 + "\n")
    st = time.time()

    # ── Step 1.1: Study area ──────────────────────────────────────────────────
    print("--- Step 1.1: Study Area ---\n")
    sa_boundary_path = os.path.join(paths.MAIN, paths.STUDY_AREA_BOUNDARY_GPKG)
    sa_buffer_path   = os.path.join(paths.MAIN, paths.STUDY_AREA_BUFFER_GPKG)

    sa_boundary, sa_buffer, sa_admin, sa_names, sa_buf_m = _resolve_study_area()
    _export_gpkg(sa_boundary, 'study_area_boundary', sa_admin, sa_names, 0,         sa_boundary_path)
    _export_gpkg(sa_buffer,   'study_area_buffer',   sa_admin, sa_names, sa_buf_m,  sa_buffer_path)
    print(f"  Study area built from settings and saved.")
    print(f"  Bounds: {sa_boundary.bounds}\n")

    # ── Step 1.2: Catchment area ──────────────────────────────────────────────
    print("--- Step 1.2: Catchment Area ---\n")
    ca_boundary_path = os.path.join(paths.MAIN, paths.CATCHMENT_AREA_BOUNDARY_GPKG)
    ca_buffer_path   = os.path.join(paths.MAIN, paths.CATCHMENT_AREA_BUFFER_GPKG)

    ca_boundary, ca_buffer, ca_admin, ca_names, ca_buf_m = _resolve_catchment_area()

    valid = _validate_containment(sa_boundary, ca_boundary)
    if not valid:
        print("  WARNING: Study area is not fully within catchment area.")
        print("  Update CATCHMENT_AREA_* in settings.py and re-run.")

    _export_gpkg(ca_boundary, 'catchment_area_boundary', ca_admin, ca_names, 0,        ca_boundary_path)
    _export_gpkg(ca_buffer,   'catchment_area_buffer',   ca_admin, ca_names, ca_buf_m, ca_buffer_path)
    print(f"  Catchment area built from settings and saved.")
    print(f"  Bounds: {ca_boundary.bounds}\n")

    # ── Step 1.3: Infrastructure version check ────────────────────────────────
    print("--- Step 1.3: Infrastructure Version ---\n")
    infra_v = settings.INFRA_VERSION

    if infra_v == 'Build_New':
        print(f"  INFRA_VERSION = 'Build_New' → will build '{settings.INFRA_BUILD_NEW_NAME}' in Phase 3A.")
        PIPELINE_CONFIG.infra_version = settings.INFRA_BUILD_NEW_NAME
    elif paths.infra_version_exists(infra_v):
        print(f"  Infrastructure version '{infra_v}' found on disk.")
        PIPELINE_CONFIG.infra_version = infra_v
    else:
        print(f"  WARNING: Infrastructure version '{infra_v}' not found.")
        print(f"    Expected: {paths.get_infra_version_dir(infra_v)}")
        print(f"  1) Switch to Build_New")
        print(f"  2) Abort")
        ans = input(f"  Select [1]: ").strip() or '1'
        if ans == '1':
            print(f"  Switching INFRA_VERSION to Build_New.")
            settings.INFRA_VERSION = 'Build_New'
            PIPELINE_CONFIG.infra_version = settings.INFRA_BUILD_NEW_NAME
        else:
            raise SystemExit("Aborted: infrastructure version not found.")

    # ── Step 1.3b: Derived infra version from active interventions (Topic 2) ─
    print("\n--- Step 1.3b: Derived Infra Version ---\n")
    try:
        import infrabuild_version_manager as _vm
        _active_infra_ints = _vm.enumerate_active_infra_ints()
        if _active_infra_ints and PIPELINE_CONFIG.infra_version not in ('Build_New', None):
            print(f"  Active infra ints: {_active_infra_ints}")
            _derived = _vm.resolve_or_build(
                PIPELINE_CONFIG.infra_version, _active_infra_ints,
            )
            if _derived != PIPELINE_CONFIG.infra_version:
                print(f"  Pipeline infra version: '{PIPELINE_CONFIG.infra_version}'"
                      f" → '{_derived}'")
                PIPELINE_CONFIG.infra_version = _derived
        else:
            print("  No active infra ints — base infra version retained.")
    except Exception as _exc:
        print(f"  WARNING: derived version resolution failed: {_exc}")
        print("  Continuing with base infra version.")

    # ── Step 1.4: Services version check ─────────────────────────────────────
    print("\n--- Step 1.4: Services Version ---\n")
    svc_v = settings.SVC_VERSION

    if svc_v == 'Build_New':
        print(f"  SVC_VERSION = 'Build_New' → will build '{settings.SVC_BUILD_NEW_NAME}' in Phase 3B.")
        PIPELINE_CONFIG.svc_version       = settings.SVC_BUILD_NEW_NAME
        PIPELINE_CONFIG.needs_projection  = True
    elif not paths.svc_version_exists(svc_v):
        print(f"  WARNING: Services network '{svc_v}_network' not found.")
        print(f"    Expected: data/Network/Rail_Lines/{svc_v}_network/Unprojected/"
              "{rail_lines,rail_segments,rail_stops}.gpkg")
        print(f"  1) Switch to Build_New")
        print(f"  2) Abort")
        ans = input(f"  Select [1]: ").strip() or '1'
        if ans == '1':
            print(f"  Switching SVC_VERSION to Build_New.")
            settings.SVC_VERSION             = 'Build_New'
            PIPELINE_CONFIG.svc_version      = settings.SVC_BUILD_NEW_NAME
            PIPELINE_CONFIG.needs_projection = True
        else:
            raise SystemExit("Aborted: services version not found.")
    else:
        PIPELINE_CONFIG.svc_version = svc_v
        print(f"  Rail network '{svc_v}_network/Unprojected/' found.")

        if settings.CATCHMENT_METHOD == 'PT_Feeder':
            if paths.svc_feeder_exists(svc_v):
                print(f"  PT-Feeder network '{svc_v}_network/Unprojected/' found.")
            else:
                print(f"  WARNING: PT-Feeder network '{svc_v}_network/Unprojected/' incomplete.")
                print(f"    Expected: data/Network/Feeder_Lines/{svc_v}_network/Unprojected/"
                      "{pt_feeder_lines,pt_feeder_segments,pt_feeder_stops}.gpkg")
                print(f"  Re-run services_network_builder.py with mode 'all' or 'pt_feeder'.")

        resolved_infra = PIPELINE_CONFIG.infra_version
        if resolved_infra and resolved_infra != 'Build_New' and \
                not paths.svc_projected_exists(svc_v, resolved_infra):
            print(f"  Services version '{svc_v}' found but not yet projected to '{resolved_infra}'.")
            print(f"  Projection will run in Phase 3B.")
            PIPELINE_CONFIG.needs_projection = True
        else:
            print(f"  Services version '{svc_v}' found and projected.")

    # ── Step 1.5: Run configuration table ────────────────────────────────────
    infra_tag      = (f"  -->  build as '{settings.INFRA_BUILD_NEW_NAME}'"
                      if settings.INFRA_VERSION == 'Build_New' else "")
    needs_proj_tag = "  [needs projection]" if PIPELINE_CONFIG.needs_projection else ""

    config_lines = [
        "=" * 80,
        "  RUN CONFIGURATION",
        "=" * 80,
        f"  Infrastructure   : {settings.INFRA_VERSION}{infra_tag}",
        f"  Raw version      : {settings.INFRA_RAW_VERSION}",
        f"  Services         : {settings.SVC_VERSION}{needs_proj_tag}",
        f"  GTFS version     : {settings.GTFS_FILTER_VERSION}",
        f"  Canton           : {settings.CATCHMENT_CANTON_ABBREV}",
        f"  Catchment method : {settings.CATCHMENT_METHOD}  (OD: {get_catchment_od_method()})",
        f"  Travel cost      : {settings.TRAVEL_COST_METHOD}  "
        f"(transfer: {settings.TRANSFER_COST_MODEL})",
        f"  Capacity mode    : {settings.CAPACITY_MODE}",
        f"  OD type          : {settings.OD_TYPE}  (base year {settings.POPULATION_BASE_YEAR})",
        f"  Population base  : {settings.POPULATION_BASE_YEAR}",
        f"  Scenarios        : {settings.amount_of_scenarios} x "
        f"[{settings.start_year_scenario}-{settings.end_year_scenario}]",
        "-" * 80,
    ]
    print()
    for line in config_lines:
        print(line)
    print()

    # Write configuration header to report_new.txt immediately so it is
    # present even if the pipeline is interrupted before completion.
    rt_file = os.path.join(paths.MAIN, 'report_new.txt')
    with open(rt_file, 'w', encoding='utf-8') as f:
        f.write("INFRASCANRAIL NEW PIPELINE - RUN LOG\n\n")
        for line in config_lines:
            f.write(line + "\n")
        f.write("\n")

    runtimes["Phase 1: Initialisation"] = time.time() - st
    return sa_boundary, sa_buffer, ca_boundary, ca_buffer


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 -Data Preparation
# ═══════════════════════════════════════════════════════════════════════════════

def phase_2_data_preparation(
    sa_boundary,
    sa_buffer,
    ca_boundary,
    ca_buffer,
    runtimes: dict,
) -> None:
    """Import base datasets needed by all downstream phases.

    Args:
        sa_boundary: Study area polygon (Shapely).
        sa_buffer:   Study area polygon + margin (Shapely).
        ca_boundary: Catchment area polygon (Shapely).
        ca_buffer:   Catchment area polygon + GTFS buffer (Shapely).
        runtimes:    Dict tracking phase execution times.
    """
    print("\n" + "=" * 80)
    print("PHASE 2: DATA PREPARATION")
    print("=" * 80 + "\n")
    st = time.time()
    loaded = []
    skipped = []

    # ── Step 2.1: Lake clipping ───────────────────────────────────────────────
    print("--- Step 2.1: Lake Clipping ---\n")
    lakes_src_path = os.path.join(paths.MAIN, paths.LAKES_SHP)
    if os.path.isfile(lakes_src_path):
        os.makedirs(os.path.join(paths.MAIN, os.path.dirname(paths.LAKES_CA_GPKG)),
                    exist_ok=True)
        lakes_src = gpd.read_file(lakes_src_path).to_crs('EPSG:2056')

        lakes_ca = lakes_src[lakes_src.intersects(ca_buffer)].copy()
        lakes_ca = gpd.clip(lakes_ca, ca_buffer)
        lakes_ca.to_file(os.path.join(paths.MAIN, paths.LAKES_CA_GPKG), driver='GPKG')
        loaded.append("lakes (CA)")

        lakes_sa = lakes_src[lakes_src.intersects(sa_buffer)].copy()
        lakes_sa = gpd.clip(lakes_sa, sa_buffer)
        lakes_sa.to_file(os.path.join(paths.MAIN, paths.LAKES_SA_GPKG), driver='GPKG')
        loaded.append("lakes (SA)")
        print(f"  Lakes clipped to CA ({len(lakes_ca)} features) and "
              f"SA ({len(lakes_sa)} features) and saved.")
    else:
        print(f"  WARNING: Lake source not found: {paths.LAKES_SHP}")
    print()

    # ── Step 2.2: Population & employment grid (catchment_base) ──────────────
    print("--- Step 2.2: Population & Employment Grid ---\n")
    print(f"  Running catchment_base for POPULATION_BASE_YEAR={settings.POPULATION_BASE_YEAR} ...")
    import catchment_base as _cb
    scale_report = _cb.main(
        year=settings.POPULATION_BASE_YEAR,
        do_plots=settings.PLOT_DATA,
    )
    if scale_report:
        rt_file = os.path.join(paths.MAIN, 'report_new.txt')
        with open(rt_file, 'a', encoding='utf-8') as f:
            f.write("\n--- Grid Scaling (Step 2.2) ---\n")
            f.write(scale_report + "\n")
    loaded.append(f"pop/empl grid ({settings.POPULATION_BASE_YEAR})")
    print()

    # ── Step 2.3: BAV infrastructure filter ──────────────────────────────────
    print("--- Step 2.3: BAV Infrastructure Filter ---\n")
    raw_dir = paths.get_infra_raw_dir(settings.INFRA_RAW_VERSION)
    required_raw = [
        'nodes.gpkg', 'segments.gpkg',
        'segments_composition.gpkg', 'osm_maxspeed_segments.gpkg',
    ]
    missing_raw = [f for f in required_raw
                   if not os.path.isfile(os.path.join(raw_dir, f))]

    if not missing_raw:
        print(f"  {settings.INFRA_RAW_VERSION}/ complete -- skipping BAV filter.")
        skipped.append("BAV filter")
    else:
        print(f"  Missing in {settings.INFRA_RAW_VERSION}/: {missing_raw}")
        print(f"  Running infrabuild_filter_network ...")
        from infrabuild_filter_network import run_filter_network
        ca_buf_path = os.path.join(paths.MAIN, paths.CATCHMENT_AREA_BUFFER_GPKG)
        run_filter_network(
            output_dir=raw_dir,
            catchment_filepath=ca_buf_path,
        )
        loaded.append("BAV Raw network")
        print(f"  BAV filter complete.\n")

    # ── Step 2.4: GTFS filter ─────────────────────────────────────────────────
    print("--- Step 2.4: GTFS Filter ---\n")
    gtfs_dir      = os.path.join(paths.MAIN, paths.GTFS_TRANSIT_DIR,
                                 settings.GTFS_FILTER_VERSION)
    gtfs_key_file = os.path.join(gtfs_dir, 'stop_times.txt')

    if os.path.isfile(gtfs_key_file):
        print(f"  {settings.GTFS_FILTER_VERSION} found -- skipping GTFS filter.")
        skipped.append("GTFS filter")
    else:
        print(f"  {settings.GTFS_FILTER_VERSION} not found -- running services_filter_gtfs ...")
        print(f"    Input  : {settings.GTFS_RAW_VERSION}")
        print(f"    Output : {settings.GTFS_FILTER_VERSION}")
        script_path = os.path.join(paths.MAIN, 'services_filter_gtfs.py')
        result = subprocess.run(
            [sys.executable, script_path,
             '--input-folder',  settings.GTFS_RAW_VERSION,
             '--output-folder', settings.GTFS_FILTER_VERSION],
            cwd=paths.MAIN,
        )
        if result.returncode != 0:
            print(f"  WARNING: services_filter_gtfs.py exited with code {result.returncode}.")
        else:
            loaded.append("GTFS filtered data")
            print(f"  GTFS filter complete.\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("-" * 80)
    print("  DATA PREPARATION SUMMARY")
    print("-" * 80)
    if loaded:
        print(f"  Loaded  : {', '.join(loaded)}")
    if skipped:
        print(f"  Skipped : {', '.join(skipped)}  (already on disk)")
    print("-" * 80 + "\n")

    runtimes["Phase 2: Data Preparation"] = time.time() - st


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3A -Infrastructure Network Build
# ═══════════════════════════════════════════════════════════════════════════════

def phase_3a_infrastructure(runtimes: dict) -> None:
    """Build the macroscopic Base network if missing and resolve the infra version.

    For Build_New: launches infrabuild_version_manager interactively so the user
    can create the named scenario version before projection and enhancement.
    For named versions: validates the version exists on disk.

    Args:
        runtimes: Dict tracking phase execution times.
    """
    print("\n" + "=" * 80)
    print("PHASE 3A: INFRASTRUCTURE NETWORK BUILD")
    print("=" * 80 + "\n")
    st = time.time()

    base_name = f'Base_{settings.CATCHMENT_CANTON_ABBREV}'
    infra_v   = PIPELINE_CONFIG.infra_version  # resolved by Phase 1

    # ── Steps 3A.1 + 3A.2: Base network and infrastructure version ───────────
    if settings.INFRA_VERSION == 'Build_New':
        # 3A.1 — always rebuild Base, seeded from INFRA_BUILD_NEW_NAME year
        print("--- Step 3A.1: Base Network ---\n")
        from infrabuild_network_builder import run_build_base
        import re as _re_seed
        _m = _re_seed.search(r'(20\d{2})', settings.INFRA_BUILD_NEW_NAME)
        _seed_year = _m.group(1) if _m else None
        if _seed_year:
            print(f"  INFRA_VERSION=Build_New — rebuilding '{base_name}' "
                  f"with seed year '{_seed_year}' (from INFRA_BUILD_NEW_NAME).")
        else:
            print(f"  INFRA_VERSION=Build_New — rebuilding '{base_name}' "
                  f"(no year detected in INFRA_BUILD_NEW_NAME='{settings.INFRA_BUILD_NEW_NAME}').")
        run_build_base(
            raw_dir=paths.get_infra_raw_dir(settings.INFRA_RAW_VERSION),
            output_dir=paths.get_infra_version_dir(base_name),
            seed_year=_seed_year,
        )
        print(f"  Base network built → {paths.get_infra_version_dir(base_name)}\n")

        # 3A.2 — open version manager to create the named version
        print("--- Step 3A.2: Infrastructure Version ---\n")
        print(f"  Opening Infrastructure Version Manager to create '{infra_v}'.")
        print(f"  ┌─ INSTRUCTIONS ──────────────────────────────────────────────────")
        print(f"  │  Base version : {base_name}  (auto-selected)")
        print(f"  │  Version name : {infra_v}  (auto-filled)")
        print(f"  │  Edit nodes/segments as needed, then Save and close.")
        print(f"  └─────────────────────────────────────────────────────────────────\n")
        script_path = os.path.join(paths.MAIN, 'infrabuild_version_manager.py')
        result = subprocess.run(
            [sys.executable, script_path,
             '--create-from', base_name,
             '--name',        infra_v,
             '--overwrite'],
            cwd=paths.MAIN,
        )
        if result.returncode != 0:
            print(f"  WARNING: infrabuild_version_manager.py exited with code "
                  f"{result.returncode}.")
        elif not paths.infra_version_exists(infra_v):
            print(f"  WARNING: Version '{infra_v}' not found on disk after version "
                  f"manager exited.")
            print(f"  Ensure you saved before closing (Phase 3 → Save and close).")
        else:
            print(f"  Infrastructure version '{infra_v}' created successfully.")

    else:
        # Named version exists (guaranteed by Phase 1) — base check not needed
        print("--- Step 3A.1: Base Network ---\n")
        print(f"  Infrastructure version '{infra_v}' already exists — base check skipped.\n")
        print("--- Step 3A.2: Infrastructure Version ---\n")
        print(f"  Infrastructure version '{infra_v}' found on disk.")
    print()

    # ── Step 3A.3: Pre-enhancement network visualisation ──────────────────────
    # Skipped for _enhanced versions (no pre-enhancement state to capture).
    if not infra_v.endswith('_enhanced'):
        print("--- Step 3A.3: Pre-Enhancement Network Visualisation ---\n")
        if paths.infra_version_exists(infra_v):
            from infrabuild_network_builder import _build_infra_qgz
            _version_dir = Path(paths.get_infra_version_dir(infra_v))
            _build_infra_qgz(str(_version_dir / f'{infra_v}.qgz'), _version_dir)
            print(f"  QGIS project written: {_version_dir / f'{infra_v}.qgz'}")
        else:
            print(f"  Version '{infra_v}' not on disk — skipping QGIS project.")

        if not settings.PLOT_INFRA:
            print(f"  PLOT_INFRA = False — skipping infrastructure plots.")
        elif not paths.infra_version_exists(infra_v):
            print(f"  Version '{infra_v}' not on disk — skipping plots.")
        else:
            from infrabuild_network_builder import (
                load_version,
                build_networkx_graph,
                plot_infrastructure_canonical,
                plot_gauge_map,
                plot_electrification_map,
                plot_speed_map,
                NetworkData,
            )
            import matplotlib.pyplot as plt
            print(f"  Generating plots for '{infra_v}' ...")

            nodes, segments = load_version(infra_v)
            G = build_networkx_graph(nodes, segments)

            _ca_bdry_path = os.path.join(paths.MAIN, paths.CATCHMENT_AREA_BOUNDARY_GPKG)
            _sa_bdry_path = os.path.join(paths.MAIN, paths.STUDY_AREA_BOUNDARY_GPKG)
            ca_bdry_gdf = gpd.read_file(_ca_bdry_path) if os.path.isfile(_ca_bdry_path) else None
            sa_bdry_gdf = gpd.read_file(_sa_bdry_path) if os.path.isfile(_sa_bdry_path) else None

            def _extent_from_gdf(gdf, margin_m: int = 2000):
                if gdf is None:
                    return None
                b = gdf.total_bounds
                return (b[0] - margin_m, b[2] + margin_m, b[1] - margin_m, b[3] + margin_m)

            ca_ext = _extent_from_gdf(ca_bdry_gdf)
            sa_ext = _extent_from_gdf(sa_bdry_gdf)

            net_ca = NetworkData(nodes=nodes, segments=segments, graph=G,
                                 version=infra_v, boundary=ca_bdry_gdf)
            net_sa = NetworkData(nodes=nodes, segments=segments, graph=G,
                                 version=infra_v, boundary=sa_bdry_gdf)

            plot_dir = Path(paths.MAIN) / paths.INFRASTRUCTURE_PLOTS_DIR / infra_v
            plot_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Generating plots → {plot_dir}")

            _plots = [
                (plot_infrastructure_canonical, net_ca, ca_ext,
                 'ca_infrastructure.pdf', {'is_catchment': True, 'show_labels': False}),
                (plot_gauge_map,               net_ca, ca_ext,
                 'ca_gauge.pdf',          {'is_catchment': True}),
                (plot_electrification_map,     net_ca, ca_ext,
                 'ca_electrification.pdf', {'is_catchment': True}),
                (plot_speed_map,               net_ca, ca_ext,
                 'ca_speed.pdf',           {'is_catchment': True}),
                (plot_infrastructure_canonical, net_sa, sa_ext,
                 'sa_infrastructure.pdf', {'show_outside': True}),
                (plot_gauge_map,               net_sa, sa_ext,
                 'sa_gauge.pdf',          {'show_outside': True}),
                (plot_electrification_map,     net_sa, sa_ext,
                 'sa_electrification.pdf', {'show_outside': True}),
                (plot_speed_map,               net_sa, sa_ext,
                 'sa_speed.pdf',           {'show_outside': True}),
            ]
            for _fn, _net, _ext, _fname, _kw in _plots:
                print(f"    {_fname} ...")
                _fig = _fn(_net, extent=_ext, output_path=plot_dir / _fname, **_kw)
                plt.close(_fig)

            print(f"  Plots complete.\n")

    runtimes["Phase 3A: Infrastructure Network Build"] = time.time() - st


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3B -Services Network Build
# ═══════════════════════════════════════════════════════════════════════════════

def _list_svc_networks() -> list:
    """Return sorted list of network folder names found under RAIL_LINES_DIR."""
    rail_base = Path(paths.MAIN) / paths.RAIL_LINES_DIR
    if not rail_base.exists():
        return []
    return sorted([
        d.name for d in rail_base.iterdir()
        if d.is_dir() and d.name.endswith('_network')
        and (d / paths.SERVICES_UNPROJECTED_SUBDIR).exists()
    ])


def phase_3b_services(
    sa_boundary,
    ca_boundary,
    runtimes: dict,
) -> None:
    """Build services network, project onto infra, enhance, and plot.

    Handles the full integration pipeline:
      3B.1  Services network selection / build (Build_New only: interactive menu)
      3B.2  Project services onto infrastructure graph
      3B.3  Enhance infrastructure with GTFS travel times
      3B.4  Build QGIS project and infrastructure plots

    Args:
        sa_boundary: Study area polygon (Shapely) — held for future use.
        ca_boundary: Catchment area polygon (Shapely) — held for future use.
        runtimes:    Dict tracking phase execution times.
    """
    print("\n" + "=" * 80)
    print("PHASE 3B: SERVICES NETWORK BUILD")
    print("=" * 80 + "\n")
    st = time.time()

    infra_v = PIPELINE_CONFIG.infra_version  # e.g. 'AS_2026_ZH' or 'AS_2026_ZH_enhanced'
    svc_v   = PIPELINE_CONFIG.svc_version    # e.g. 'SVC2026_ZH_S18'

    already_enhanced = infra_v.endswith('_enhanced')
    base_infra_v     = infra_v.removesuffix('_enhanced')
    enhanced_v       = f'{base_infra_v}_enhanced'

    # ── Step 3B.1: Services network / version manager ────────────────────────
    print("--- Step 3B.1: Services Network ---\n")
    if settings.SVC_VERSION == 'Build_New':
        existing = _list_svc_networks()
        if existing:
            print(f"  Found {len(existing)} existing network(s): {', '.join(existing)}")
        else:
            print("  No existing networks found.")
        print(f"  SVC_VERSION = 'Build_New' — creating new network.\n")

        svc_network = svc_v + '_network'
        vm_script   = os.path.join(paths.MAIN, 'services_version_manager.py')

        print(f"  Building new network '{svc_network}' ...")
        print(f"    GTFS source : {settings.GTFS_FILTER_VERSION}")
        print(f"    Modes       : all  (rail + feeder)")
        print(f"    Periods     : all")
        builder_script = os.path.join(paths.MAIN, 'services_network_builder.py')
        build_cmd = [
            sys.executable, builder_script,
            '--gtfs-folder',     settings.GTFS_FILTER_VERSION,
            '--output-name',     svc_network,
            '--modes',           'all',
            '--all-periods',
            '--non-interactive',
        ]
        result = subprocess.run(build_cmd, cwd=paths.MAIN)
        if result.returncode != 0:
            print(f"  WARNING: services_network_builder.py exited with code "
                  f"{result.returncode}.")
        else:
            print(f"  Network build complete.\n")

        print(f"  Opening version manager for '{svc_network}' ...")
        result = subprocess.run(
            [sys.executable, vm_script,
             '--infra-version', infra_v,
             '--network',       svc_network],
            cwd=paths.MAIN,
        )
        if result.returncode != 0:
            print(f"  WARNING: services_version_manager.py exited with code "
                  f"{result.returncode}.")
        else:
            print(f"  Version manager complete.\n")

    else:
        need_rail   = not paths.svc_version_exists(svc_v)
        need_feeder = (
            settings.CATCHMENT_METHOD == 'PT_Feeder'
            and not paths.svc_feeder_exists(svc_v)
        )
        if not need_rail and not need_feeder:
            print(f"  Services network '{svc_v}' found — skipping build.")
        else:
            missing = []
            if need_rail:
                missing.append("rail network")
            if need_feeder:
                missing.append("PT-feeder network")
            print(f"  Missing: {', '.join(missing)} — running services_network_builder ...")
            script_path = os.path.join(paths.MAIN, 'services_network_builder.py')
            result = subprocess.run([sys.executable, script_path], cwd=paths.MAIN)
            if result.returncode != 0:
                print(f"  WARNING: services_network_builder.py exited with code "
                      f"{result.returncode}.")
            else:
                print(f"  Services network build complete.\n")
    print()

    # ── Step 3B.2: Service projection ─────────────────────────────────────
    # Always project onto infra_v (the active version). When already_enhanced,
    # infra_v is the enhanced network which is the correct routing base.
    # When not enhanced, infra_v == base_infra_v — same result, enhancement
    # will then read the projected data from the base_infra_v path.
    print("--- Step 3B.2: Service Projection ---\n")
    _svc_script = os.path.join(paths.MAIN, 'services_service_projection.py')
    _svc_base_flags = ['--svc-version', svc_v, '--infra-version', infra_v]
    if settings.CATCHMENT_METHOD != 'PT_Feeder':
        _svc_base_flags.append('--no-feeder-plots')

    if paths.svc_projected_exists(svc_v, infra_v):
        print(f"  Projected services for '{svc_v}' on '{infra_v}' found — skipping.")
        if settings.PLOT_SERVICES:
            print(f"  Generating service plots from existing projection ...")
            result = subprocess.run(
                [sys.executable, _svc_script] + _svc_base_flags + ['--plot-only'],
                cwd=paths.MAIN,
            )
            if result.returncode != 0:
                print(f"  WARNING: services_service_projection.py exited with code "
                      f"{result.returncode}.")
            else:
                print(f"  Service plots complete.\n")
    else:
        print(f"  Projecting '{svc_v}' onto '{infra_v}' (non-interactive) ...")
        _proj_cmd = [sys.executable, _svc_script] + _svc_base_flags
        if not settings.PLOT_SERVICES:
            _proj_cmd.append('--no-plots')
        result = subprocess.run(_proj_cmd, cwd=paths.MAIN)
        if result.returncode != 0:
            print(f"  WARNING: services_service_projection.py exited with code "
                  f"{result.returncode}.")
        else:
            print(f"  Service projection complete.\n")
    print()

    # ── Step 3B.3: Infrastructure enhancement ────────────────────────────
    print("--- Step 3B.3: Infrastructure Enhancement ---\n")
    if already_enhanced:
        print(f"  Infrastructure version '{infra_v}' is already enhanced.")
        print(f"  Skipping enhancement — network assumed complete.")
        print(f"  To re-calibrate, set INFRA_VERSION = '{base_infra_v}' and re-run.\n")
    else:
        mode_hint = "(extend — fills null slots)" if paths.infra_version_exists(enhanced_v) \
                    else "(initial — full calibration)"
        print(f"  Enhancing '{base_infra_v}' with GTFS-calibrated travel times "
              f"{mode_hint} (non-interactive) ...")
        script_path = os.path.join(
            paths.MAIN, 'infrabuild_infrastructure_enhancement.py'
        )
        result = subprocess.run(
            [sys.executable, script_path,
             '--infra-version',  base_infra_v,
             '--svc-version',    svc_v + '_network',
             '--enhanced-name',  enhanced_v],
            cwd=paths.MAIN,
        )
        if result.returncode != 0:
            print(f"  WARNING: infrabuild_infrastructure_enhancement.py exited with "
                  f"code {result.returncode}.")
        else:
            print(f"  Enhancement complete.\n")
            # ── Write enhancement stats to report ─────────────────────────────
            _enh_segs_path = os.path.join(
                paths.get_infra_version_dir(enhanced_v), 'segments.gpkg'
            )
            if os.path.isfile(_enh_segs_path):
                _enh_segs = gpd.read_file(_enh_segs_path)
                _n_total      = len(_enh_segs)
                _n_tt         = int(_enh_segs['TT_Stopping'].notna().sum()) \
                                if 'TT_Stopping' in _enh_segs.columns else 0
                _ss = _enh_segs['speed_source'].fillna('') \
                      if 'speed_source' in _enh_segs.columns \
                      else pd.Series([''] * _n_total)
                _n_gtfs     = int((_ss == 'gtfs').sum())
                _n_formula  = int((_ss == 'formula').sum())
                _n_estimate = int((_ss == 'estimate').sum())
                _n_design   = int((_ss == 'design').sum())
                _n_feeder   = int(
                    _enh_segs['Segment_ID'].astype(str).str.startswith('feeder_').sum()
                ) if 'Segment_ID' in _enh_segs.columns else 0
                _rt_file = os.path.join(paths.MAIN, 'report_new.txt')
                with open(_rt_file, 'a', encoding='utf-8') as _f:
                    _f.write(f"\n--- Enhancement Stats: {enhanced_v} ---\n")
                    _f.write(f"  Total segments         : {_n_total}\n")
                    _f.write(f"  TT_Stopping filled     : {_n_tt} "
                             f"({_n_tt / _n_total * 100:.1f}%)\n")
                    _f.write(f"  speed_source = gtfs    : {_n_gtfs}\n")
                    _f.write(f"  speed_source = formula : {_n_formula}\n")
                    _f.write(f"  speed_source = estimate: {_n_estimate}\n")
                    _f.write(f"  speed_source = design  : {_n_design}\n")
                    _f.write(f"  Feeder-derived segs    : {_n_feeder}\n")
        print()

    # ── Update resolved infra version ─────────────────────────────────────
    PIPELINE_CONFIG.infra_version = enhanced_v
    print(f"  Active infrastructure version updated to: {enhanced_v}\n")

    # ── Step 3B.3b: Project services onto enhanced version ────────────────────
    # Only needed when enhancement just ran. When already_enhanced, Step 3B.2
    # projected onto the enhanced version directly so this step is a no-op.
    if not already_enhanced:
        print("--- Step 3B.3b: Service Projection onto Enhanced Network ---\n")
        _svc_enh_flags = ['--svc-version', svc_v, '--infra-version', enhanced_v]
        if settings.CATCHMENT_METHOD != 'PT_Feeder':
            _svc_enh_flags.append('--no-feeder-plots')

        if paths.svc_projected_exists(svc_v, enhanced_v):
            print(f"  Projected services for '{svc_v}' on '{enhanced_v}' found — skipping.")
            if settings.PLOT_SERVICES:
                print(f"  Generating service plots from existing projection ...")
                result = subprocess.run(
                    [sys.executable, _svc_script] + _svc_enh_flags + ['--plot-only'],
                    cwd=paths.MAIN,
                )
                if result.returncode != 0:
                    print(f"  WARNING: services_service_projection.py exited with code "
                          f"{result.returncode}.")
                else:
                    print(f"  Service plots complete.\n")
        else:
            print(f"  Projecting '{svc_v}' onto '{enhanced_v}' (non-interactive) ...")
            _proj_cmd = [sys.executable, _svc_script] + _svc_enh_flags
            if not settings.PLOT_SERVICES:
                _proj_cmd.append('--no-plots')
            result = subprocess.run(_proj_cmd, cwd=paths.MAIN)
            if result.returncode != 0:
                print(f"  WARNING: services_service_projection.py exited with code "
                      f"{result.returncode}.")
            else:
                print(f"  Service projection onto enhanced network complete.\n")
        print()

    # ── Step 3B.4: QGIS project + diff report + plots ────────────────────────
    print("--- Step 3B.4: Network Visualisation ---\n")
    final_infra_v = PIPELINE_CONFIG.infra_version

    from infrabuild_network_builder import (
        _build_infra_qgz,
        load_version,
        build_networkx_graph,
        NetworkData,
        export_infrastructure_diff,
    )

    # QGIS project — always
    _version_dir = Path(paths.get_infra_version_dir(final_infra_v))
    _build_infra_qgz(str(_version_dir / f'{final_infra_v}.qgz'), _version_dir)
    print(f"  QGIS project written: {_version_dir / f'{final_infra_v}.qgz'}")

    # Load enhanced network — needed for Excel diff and/or plots
    nodes, segments = load_version(final_infra_v)
    G = build_networkx_graph(nodes, segments)

    _ca_bdry_path = os.path.join(paths.MAIN, paths.CATCHMENT_AREA_BOUNDARY_GPKG)
    _sa_bdry_path = os.path.join(paths.MAIN, paths.STUDY_AREA_BOUNDARY_GPKG)
    ca_bdry_gdf = gpd.read_file(_ca_bdry_path) if os.path.isfile(_ca_bdry_path) else None
    sa_bdry_gdf = gpd.read_file(_sa_bdry_path) if os.path.isfile(_sa_bdry_path) else None

    # Excel diff report — always when the pre-enhancement version exists
    _diff_base = base_infra_v if not already_enhanced else \
                 final_infra_v.removesuffix('_enhanced')
    _base_nodes = _base_segs = _base_G = None
    if paths.infra_version_exists(_diff_base):
        _base_nodes, _base_segs = load_version(_diff_base)
        _base_G = build_networkx_graph(_base_nodes, _base_segs)
        _ref_comp_path = Path(paths.get_infra_version_dir(_diff_base)) / 'segments_composition.gpkg'
        _enh_comp_path = _version_dir / 'segments_composition.gpkg'
        _ref_comp = gpd.read_file(str(_ref_comp_path)) if _ref_comp_path.exists() else gpd.GeoDataFrame()
        _enh_comp = gpd.read_file(str(_enh_comp_path)) if _enh_comp_path.exists() else gpd.GeoDataFrame()
        _diff_xlsx = _version_dir / f'diff_{final_infra_v}_vs_{_diff_base}.xlsx'
        export_infrastructure_diff(
            net_a=NetworkData(nodes=_base_nodes, segments=_base_segs,
                              graph=_base_G, version=_diff_base),
            net_b=NetworkData(nodes=nodes, segments=segments,
                              graph=G, version=final_infra_v),
            comp_a=_ref_comp,
            comp_b=_enh_comp,
            output_path=_diff_xlsx,
        )
        print(f"  Enhancement diff report → {_diff_xlsx}")

    if not settings.PLOT_INFRA:
        print(f"  PLOT_INFRA = False — skipping infrastructure plots.")
    else:
        from infrabuild_network_builder import (
            plot_infrastructure_canonical,
            plot_infrastructure_diff,
            plot_gauge_map,
            plot_electrification_map,
            plot_speed_map,
        )
        import matplotlib.pyplot as plt
        print(f"  Generating plots for '{final_infra_v}' ...")

        def _extent_from_gdf(gdf, margin_m: int = 2000):
            if gdf is None:
                return None
            b = gdf.total_bounds
            return (b[0] - margin_m, b[2] + margin_m, b[1] - margin_m, b[3] + margin_m)

        ca_ext = _extent_from_gdf(ca_bdry_gdf)
        sa_ext = _extent_from_gdf(sa_bdry_gdf)

        net_ca = NetworkData(nodes=nodes, segments=segments, graph=G,
                             version=final_infra_v, boundary=ca_bdry_gdf)
        net_sa = NetworkData(nodes=nodes, segments=segments, graph=G,
                             version=final_infra_v, boundary=sa_bdry_gdf)

        plot_dir = Path(paths.MAIN) / paths.INFRASTRUCTURE_PLOTS_DIR / final_infra_v
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Generating plots → {plot_dir}")

        _plots = [
            (plot_infrastructure_canonical, net_ca, ca_ext,
             'ca_infrastructure.pdf', {'is_catchment': True, 'show_labels': False}),
            (plot_gauge_map,               net_ca, ca_ext,
             'ca_gauge.pdf',          {'is_catchment': True}),
            (plot_electrification_map,     net_ca, ca_ext,
             'ca_electrification.pdf', {'is_catchment': True}),
            (plot_speed_map,               net_ca, ca_ext,
             'ca_speed.pdf',           {'is_catchment': True}),
            (plot_infrastructure_canonical, net_sa, sa_ext,
             'sa_infrastructure.pdf', {'show_outside': True}),
            (plot_gauge_map,               net_sa, sa_ext,
             'sa_gauge.pdf',          {'show_outside': True}),
            (plot_electrification_map,     net_sa, sa_ext,
             'sa_electrification.pdf', {'show_outside': True}),
            (plot_speed_map,               net_sa, sa_ext,
             'sa_speed.pdf',           {'show_outside': True}),
        ]
        for _fn, _net, _ext, _fname, _kw in _plots:
            print(f"    {_fname} ...")
            _fig = _fn(_net, extent=_ext, output_path=plot_dir / _fname, **_kw)
            plt.close(_fig)

        # Diff plots — reuse ref data already loaded for the Excel report
        if _base_nodes is not None:
            print(f"    diff vs '{_diff_base}' ...")
            _net_base_ca = NetworkData(nodes=_base_nodes, segments=_base_segs,
                                       graph=_base_G, version=_diff_base,
                                       boundary=ca_bdry_gdf)
            _fig = plot_infrastructure_diff(
                _net_base_ca, net_ca,
                extent=ca_ext,
                output_path=plot_dir / f'ca_diff_vs_{_diff_base}.pdf',
                is_catchment=True,
            )
            plt.close(_fig)
            _net_base_sa = NetworkData(nodes=_base_nodes, segments=_base_segs,
                                       graph=_base_G, version=_diff_base,
                                       boundary=sa_bdry_gdf)
            _fig = plot_infrastructure_diff(
                _net_base_sa, net_sa,
                extent=sa_ext,
                output_path=plot_dir / f'sa_diff_vs_{_diff_base}.pdf',
                show_outside=True,
            )
            plt.close(_fig)

        print(f"  Plots complete.\n")

    runtimes["Phase 3B: Services Network Build"] = time.time() - st


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3C -Capacity Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def phase_3c_capacity(
    sa_boundary,
    ca_boundary,
    runtimes: dict,
) -> None:
    """Capacity analysis: delegates to capacity_workflow_wrapper based on CAPACITY_SCOPE.

    Reads PIPELINE_CONFIG.infra_version and PIPELINE_CONFIG.svc_version (resolved by
    Phases 3A/3B). Branches on settings.CAPACITY_SCOPE:
      'SA' → run_study_area_workflow  (method from CAPACITY_MODE_SA)
      'CA' → run_catchment_area_workflow (methods from CAPACITY_MODE_SA + CAPACITY_MODE_CA)

    Skips entirely when settings.CAPACITY_MODE = 'None'.

    Args:
        sa_boundary: Study area polygon (Shapely) — reserved for future use.
        ca_boundary: Catchment area polygon (Shapely) — reserved for future use.
        runtimes:    Dict tracking phase execution times.
    """
    print("\n" + "=" * 80)
    print("PHASE 3C: CAPACITY ANALYSIS")
    print("=" * 80 + "\n")
    st = time.time()

    # ── Step 3C.0: Mode check ─────────────────────────────────────────────────
    print("--- Step 3C.0: Mode Check ---\n")
    if settings.CAPACITY_MODE == 'None':
        print("  CAPACITY_MODE = 'None' — skipping Phase 3C.")
        runtimes["Phase 3C: Capacity Analysis"] = time.time() - st
        return

    # ── Resolve infra and service versions ────────────────────────────────────
    infra_v = PIPELINE_CONFIG.infra_version
    svc_v   = PIPELINE_CONFIG.svc_version

    if infra_v is None:
        print("  WARNING: PIPELINE_CONFIG.infra_version not set (Phase 1 may not have run).")
        infra_v = settings.INFRA_VERSION
        if infra_v == 'Build_New':
            infra_v = settings.INFRA_BUILD_NEW_NAME
    if svc_v is None:
        print("  WARNING: PIPELINE_CONFIG.svc_version not set (Phase 1 may not have run).")
        svc_v = settings.SVC_VERSION
        if svc_v == 'Build_New':
            svc_v = settings.SVC_BUILD_NEW_NAME

    _scope     = getattr(settings, 'CAPACITY_SCOPE', 'CA')
    _mode_sa   = getattr(settings, 'CAPACITY_MODE_SA', settings.CAPACITY_MODE)
    _mode_ca   = getattr(settings, 'CAPACITY_MODE_CA', settings.CAPACITY_MODE)
    _set_val   = settings.CAPACITY_SET_VALUE
    _grp_strat = PIPELINE_CONFIG.grouping_strategy
    _visualize = settings.PLOT_CAPACITY

    print(f"  Scope         : {_scope}")
    print(f"  Infra version : {infra_v}")
    print(f"  Svc version   : {svc_v}")
    if _scope == 'SA':
        print(f"  SA mode       : {_mode_sa}")
    else:
        print(f"  SA mode       : {_mode_sa}  |  CA mode: {_mode_ca}")
    print(f"  Grouping      : {_grp_strat}")
    print(f"  Plots         : {_visualize}\n")

    # ── Delegate to workflow wrapper ──────────────────────────────────────────
    from capacity_workflow_wrapper import (
        run_study_area_workflow,
        run_catchment_area_workflow,
    )

    if _scope == 'SA':
        run_study_area_workflow(
            infra_v, svc_v,
            capacity_mode=_mode_sa,
            set_value=_set_val,
            grouping_strategy=_grp_strat,
            visualize=_visualize,
        )
    else:  # 'CA'
        run_catchment_area_workflow(
            infra_v, svc_v,
            mode_sa=_mode_sa,
            mode_ca=_mode_ca,
            set_value_sa=_set_val,
            set_value_ca=_set_val,
            grouping_strategy=_grp_strat,
            visualize=_visualize,
        )

    runtimes["Phase 3C: Capacity Analysis"] = time.time() - st


# ═══════════════════════════════════════════════════════════════════════════════
# Runtime writer
# ═══════════════════════════════════════════════════════════════════════════════

def phase_4a_catchment_allocation(
    sa_boundary,
    ca_boundary,
    runtimes: dict,
) -> None:
    """Phase 4A — Catchment Allocation.

    Runs catchment_allocate.get_catchment() hands-off, based on settings:
      - CATCHMENT_METHOD     → 'pt_feeder' or 'municipal'
      - TRAVEL_COST_METHOD   → 'calibrated' (weights from cost_parameters) or 'absolute' (all 1.0)
      - TRANSFER_COST_MODEL  → 'fixed_value' or 'explicit'
      - PLOT_CATCHMENT       → toggles the Phase 4A plot suite
      - use_cache_pt_catchment → skip when expected outputs are already on disk

    Resolves the feeder/rail base paths from PIPELINE_CONFIG.svc_version
    (set in Phase 1) and prints the active calibration inputs to terminal
    + report_new.txt before invoking the allocation.

    Args:
        sa_boundary: Study area polygon (Shapely) — reserved for future use.
        ca_boundary: Catchment area polygon (Shapely) — reserved for future use.
        runtimes:    Dict tracking phase execution times.
    """
    print("\n" + "=" * 80)
    print("PHASE 4A: CATCHMENT ALLOCATION")
    print("=" * 80 + "\n")
    st = time.time()

    # ── Step 4A.0: Calibration inputs ─────────────────────────────────────────
    print("--- Step 4A.0: Calibration Inputs ---")
    _write_calibration_inputs_to_report()

    # ── Step 4A.1: Resolve method and service version ────────────────────────
    print("\n--- Step 4A.1: Resolution ---\n")
    catchment_method = settings.CATCHMENT_METHOD
    method = 'pt_feeder' if catchment_method == 'PT_Feeder' else 'municipal'

    svc_version = PIPELINE_CONFIG.svc_version
    if svc_version is None:
        svc_version = settings.SVC_VERSION
        if svc_version == 'Build_New':
            svc_version = settings.SVC_BUILD_NEW_NAME

    svc_network = f'{svc_version}_network'
    feeder_base = os.path.join(paths.FEEDER_LINES_DIR, svc_network,
                                paths.SERVICES_UNPROJECTED_SUBDIR)
    rail_base   = os.path.join(paths.RAIL_LINES_DIR,   svc_network,
                                paths.SERVICES_UNPROJECTED_SUBDIR)

    temporal = getattr(settings, 'TEMPORAL', 'full_day')

    print(f"  Catchment method     : {catchment_method}  -> '{method}'")
    print(f"  Travel cost method   : {settings.TRAVEL_COST_METHOD}")
    print(f"  Transfer cost model  : {settings.TRANSFER_COST_MODEL}")
    print(f"  Temporal             : {temporal}")
    print(f"  Plot catchment       : {settings.PLOT_CATCHMENT}")
    print(f"  Service version      : {svc_version}")
    print(f"  Feeder base          : {feeder_base}")
    print(f"  Rail base            : {rail_base}\n")

    # ── Step 4A.2: Skip-if-cached check ──────────────────────────────────────
    print("--- Step 4A.2: Cache Check ---\n")
    if method == 'pt_feeder':
        expected = [
            os.path.join(paths.MAIN, 'data', 'Catchment_Area', svc_network,
                          'PT_Feeder', f)
            for f in ('cell_station_candidates.csv',
                      'catchment.gpkg',
                      'station_catchments.xlsx')
        ]
    else:
        expected = [
            os.path.join(paths.MAIN, 'data', 'Catchment_Area', svc_network,
                          'Municipal', f)
            for f in ('station_assignment.csv',
                      'catchment.gpkg',
                      'station_catchments.xlsx')
        ]

    if settings.use_cache_pt_catchment:
        missing = [f for f in expected if not os.path.exists(f)]
        if missing:
            print(f"  use_cache_pt_catchment = True but {len(missing)} expected file(s) missing — running allocation.")
            for f in missing:
                print(f"    missing: {f}")
        else:
            print(f"  use_cache_pt_catchment = True and all {len(expected)} expected files present — skipping Phase 4A.")
            runtimes["Phase 4A: Catchment Allocation"] = time.time() - st
            return
    else:
        print(f"  use_cache_pt_catchment = False — running allocation.")

    # ── Step 4A.3: Run catchment_allocate.get_catchment ──────────────────────
    print("\n--- Step 4A.3: Run Catchment Allocation ---\n")
    import catchment_allocate as _ca
    _ca.get_catchment(
        use_cache=settings.use_cache_pt_catchment,
        method=method,
        feeder_base=feeder_base,
        rail_base=rail_base,
        temporal=temporal,
        visualize=settings.PLOT_CATCHMENT,
        infra_projection=PIPELINE_CONFIG.infra_version,
    )

    runtimes["Phase 4A: Catchment Allocation"] = time.time() - st


def phase_4b_station_od_matrix(runtimes: dict) -> None:
    """Phase 4B — Station OD Matrix.

    Runs catchment_OD_preparation.prepare_all_od_matrices() for the active
    settings.CATCHMENT_METHOD. Communal OD is scaled forward to
    POPULATION_BASE_YEAR (per-commune geometric mean), out-of-catchment demand is
    routed to gateway (boundary) stations, and a top-5 origins/destinations Excel
    is exported for study-area stations.

    Gateway assignment behaves like the municipal station assignment: if a saved
    assignment exists the user is offered to reuse or recreate it; if none exists
    the interactive assignment runs here (the file is not required up-front).

    Args:
        runtimes: Dict tracking phase execution times.
    """
    print("\n" + "=" * 80)
    print("PHASE 4B: STATION OD MATRIX")
    print("=" * 80 + "\n")
    st = time.time()

    # ── Step 4B.1: Resolve method and service version ────────────────────────
    catchment_method = settings.CATCHMENT_METHOD
    method = 'pt_feeder' if catchment_method == 'PT_Feeder' else 'municipal'

    svc_version = PIPELINE_CONFIG.svc_version
    if svc_version is None:
        svc_version = settings.SVC_VERSION
        if svc_version == 'Build_New':
            svc_version = settings.SVC_BUILD_NEW_NAME
    svc_network = f'{svc_version}_network'

    print(f"  Catchment method     : {catchment_method}  -> '{method}'")
    print(f"  Attribution mode     : {settings.OD_ATTRIBUTION_MODE}")
    print(f"  Service version      : {svc_version}  -> '{svc_network}'")
    print(f"  Infrastructure       : {PIPELINE_CONFIG.infra_version}")
    print(f"  Population base year  : {settings.POPULATION_BASE_YEAR}\n")

    # ── Step 4B.2: Skip-if-cached check ──────────────────────────────────────
    expected = [paths.get_station_od_csv(svc_network, method, w)
                for w in ('peak', 'off_peak', 'full_day')]

    if settings.use_cache_stationsOD:
        missing = [f for f in expected if not os.path.exists(f)]
        if not missing:
            print(f"  use_cache_stationsOD = True and all {len(expected)} expected "
                  f"OD matrices present — skipping Phase 4B.")
            runtimes["Phase 4B: Station OD Matrix"] = time.time() - st
            return
        print(f"  use_cache_stationsOD = True but {len(missing)} expected file(s) "
              f"missing — running OD preparation.")
    else:
        print("  use_cache_stationsOD = False — running OD preparation.")

    # ── Step 4B.3: Run OD preparation ────────────────────────────────────────
    # Interactive for gateway assignment: reuse if a saved assignment exists,
    # otherwise prompt to create it here (mirrors the municipal assignment).
    print("\n--- Step 4B.3: Run Station OD Preparation ---\n")
    import catchment_OD_preparation as _odp
    _odp._INTERACTIVE_MODE = True
    _odp.prepare_all_od_matrices(
        use_cache=settings.use_cache_stationsOD,
        svc_version=svc_network,
        infra_version=PIPELINE_CONFIG.infra_version,
        method=method,
        attribution_mode=settings.OD_ATTRIBUTION_MODE,
    )

    _write_station_od_to_report(method, svc_network)
    runtimes["Phase 4B: Station OD Matrix"] = time.time() - st


def _write_station_od_to_report(method: str, svc_network: str) -> None:
    """Append a 'STATION OD MATRIX (Phase 4B)' block to report_new.txt."""
    lines = [
        "=" * 80,
        "  STATION OD MATRIX (Phase 4B)",
        "=" * 80,
        f"  Method               : {method}",
        f"  Attribution mode     : {settings.OD_ATTRIBUTION_MODE}",
        f"  Service version      : {svc_network}",
        f"  Population base year  : {settings.POPULATION_BASE_YEAR}",
        f"  Temporal window      : {getattr(settings, 'TEMPORAL', 'full_day')}",
        f"  OD output dir        : data/Traffic_Flow/OD/{svc_network}/",
        "=" * 80,
    ]
    for line in lines:
        print(line)
    rt_file = os.path.join(paths.MAIN, 'report_new.txt')
    with open(rt_file, 'a', encoding='utf-8') as f:
        f.write("\n")
        for line in lines:
            f.write(line + "\n")
        f.write("\n")


def _write_calibration_inputs_to_report() -> None:
    """Print and append a 'CALIBRATION INPUTS (Phase 4A)' block to report_new.txt.

    Lists all travel-cost weights, transfer-cost parameters, wait-function
    parameters, walk/cycle speeds + detour factors, and walk-buffer radii used
    by Phase 4A catchment allocation. When settings.TRAVEL_COST_METHOD =
    'absolute', the unitless weight values are shown as '1.0 (overridden)'
    and the transfer penalty falls back to the raw 7.1 min value.
    """
    import cost_parameters as cp
    import catchment_allocate as _ca

    method   = settings.TRAVEL_COST_METHOD
    tx_model = settings.TRANSFER_COST_MODEL
    is_abs   = (method == 'absolute')

    def _w(val: float) -> str:
        if is_abs:
            return "1.0 (overridden)"
        return f"{val:.3f}"

    if is_abs:
        transfer_penalty_active = float(cp.average_train_change_time)
        transfer_penalty_note   = "raw 7.1 min (no comfort weighting)"
    else:
        transfer_penalty_active = float(cp.PI_TRANSFER_MIN)
        transfer_penalty_note   = "12.1 min eq. IVT (Axhausen 2014)"

    lines = [
        "=" * 80,
        "  CALIBRATION INPUTS (Phase 4A)",
        "=" * 80,
        f"  Travel cost method     : {method}",
        f"  Transfer cost model    : {tx_model}",
        "-" * 80,
        "  Generalised-cost weights (cost_parameters.py)",
        f"    W_IVT      = {_w(cp.W_IVT)}",
        f"    W_WAIT     = {_w(cp.W_WAIT)}",
        f"    W_WALK     = {_w(cp.W_WALK)}",
        f"    W_BIKE     = {_w(cp.W_BIKE)}",
        f"    W_TRANSFER = {_w(cp.W_TRANSFER)}",
        "-" * 80,
        "  Transfer-penalty parameters",
        f"    Active transfer penalty : {transfer_penalty_active:.2f} min  ({transfer_penalty_note})",
        f"    PI_TRANSFER_MIN         : {cp.PI_TRANSFER_MIN:.2f} min  (comfort-weighted, 'fixed_value' model)",
        f"    TRANSFER_WALK_MIN       : {cp.TRANSFER_WALK_MIN:.2f} min  ('explicit' model only)",
        f"    average_train_change_time : {cp.average_train_change_time:.2f} min  (Axhausen 2014 raw)",
        f"    change_time_comfort_factor: {cp.change_time_comfort_factor:.2f}",
        "-" * 80,
        "  Wait-function parameters (piecewise; Wardman 2004 / Bates 2001)",
        f"    WAIT_THRESHOLD_MIN = {cp.WAIT_THRESHOLD_MIN:.2f} min",
        f"    WAIT_SLOPE_ABOVE   = {cp.WAIT_SLOPE_ABOVE:.3f}",
        "-" * 80,
        "  Speed and detour factors",
        f"    WALK_SPEED_KMH     = {cp.WALK_SPEED_KMH:.2f}",
        f"    WALK_DETOUR        = {cp.WALK_DETOUR:.3f}",
        f"    CYCLE_SPEED_KMH    = {cp.CYCLE_SPEED_KMH:.2f}",
        f"    CYCLE_DETOUR       = {cp.CYCLE_DETOUR:.3f}",
        f"    CYCLE_MAX_RADIUS_M = {cp.CYCLE_MAX_RADIUS_M:.0f} m",
        "-" * 80,
        "  Walk-buffer radii (catchment_allocate.py — ARE 2022)",
        f"    BUFFER_RAIL_M = {_ca.BUFFER_RAIL_M:.0f} m",
        f"    BUFFER_TRAM_M = {_ca.BUFFER_TRAM_M:.0f} m",
        f"    BUFFER_BUS_M  = {_ca.BUFFER_BUS_M:.0f} m",
        "=" * 80,
    ]

    print()
    for line in lines:
        print(line)
    print()

    rt_file = os.path.join(paths.MAIN, 'report_new.txt')
    with open(rt_file, 'a', encoding='utf-8') as f:
        f.write("\n")
        for line in lines:
            f.write(line + "\n")
        f.write("\n")


def _save_runtimes(runtimes: dict, filename: str) -> None:
    """Append phase runtimes to the run log file started by Phase 1."""
    total_time = sum(runtimes.values())
    # Use append mode: the config header was written at end of Phase 1
    with open(filename, 'a', encoding='utf-8') as f:
        f.write("PHASE RUNTIMES\n")
        f.write("=" * 80 + "\n\n")
        for part, runtime in runtimes.items():
            mins = int(runtime // 60)
            secs = int(runtime % 60)
            f.write(f"{part:.<60} {mins}m {secs}s ({runtime:.2f}s)\n")
        f.write("\n" + "=" * 80 + "\n")
        total_mins = int(total_time // 60)
        total_secs = int(total_time % 60)
        f.write(f"{'TOTAL TIME':.<60} {total_mins}m {total_secs}s ({total_time:.2f}s)\n")
        f.write("=" * 80 + "\n")
    print(f"Runtimes saved to: {filename}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def infrascanrail_new():
    """New InfraScanRail pipeline orchestrator. Runs all phases in sequence."""
    os.chdir(paths.MAIN)
    warnings.filterwarnings("ignore")
    runtimes = {}

    sa_boundary, sa_buffer, ca_boundary, ca_buffer = phase_1_initialisation(runtimes)
    phase_2_data_preparation(sa_boundary, sa_buffer, ca_boundary, ca_buffer, runtimes)
    phase_3a_infrastructure(runtimes)
    phase_3b_services(sa_boundary, ca_boundary, runtimes)
    phase_3c_capacity(sa_boundary, ca_boundary, runtimes)
    phase_4a_catchment_allocation(sa_boundary, ca_boundary, runtimes)
    phase_4b_station_od_matrix(runtimes)

    _save_runtimes(runtimes, 'report_new.txt')

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    total = sum(runtimes.values())
    print(f"\nTotal runtime: {int(total // 60)}m {int(total % 60)}s")
    print(f"Runtimes saved to: report_new.txt")
    print()


if __name__ == '__main__':
    infrascanrail_new()
