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

import cache_manifest
import cost_parameters as cp
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
# Phase 1 - Initialisation
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

    # The pre-development pipeline (Phases 3A–4) runs on the plain baseline infra
    # version. Infra interventions (connecting curves etc.) are NOT folded into the
    # baseline here — they are layered on top as Phase-5 developments via
    # infra_ints_orchestrator.compose_infra, which resolves its base from
    # settings.INFRA_VERSION independently of PIPELINE_CONFIG.infra_version.

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
        f"  Population base  : {settings.start_year_scenario}",
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
# Phase 2 - Data Preparation
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
    print(f"  Running catchment_base for start_year_scenario={settings.start_year_scenario} ...")
    import catchment_base as _cb
    scale_report = _cb.main(
        year=settings.start_year_scenario,
        do_plots=settings.PLOT_DATA,
    )
    if scale_report:
        rt_file = os.path.join(paths.MAIN, 'report_new.txt')
        with open(rt_file, 'a', encoding='utf-8') as f:
            f.write("\n--- Grid Scaling (Step 2.2) ---\n")
            f.write(scale_report + "\n")
    loaded.append(f"pop/empl grid ({settings.start_year_scenario})")
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

        # 3A.2 — create the named version (version manager opens only if toggle is on)
        print("--- Step 3A.2: Infrastructure Version ---\n")
        script_path = os.path.join(paths.MAIN, 'infrabuild_version_manager.py')
        vm_cmd = [sys.executable, script_path,
                  '--create-from', base_name,
                  '--name',        infra_v,
                  '--overwrite']
        if settings.OPEN_INFRA_VERSION_MANAGER:
            print(f"  Opening Infrastructure Version Manager to create '{infra_v}'.")
            print(f"  ┌─ INSTRUCTIONS ──────────────────────────────────────────────────")
            print(f"  │  Base version : {base_name}  (auto-selected)")
            print(f"  │  Version name : {infra_v}  (auto-filled)")
            print(f"  │  Edit nodes/segments as needed, then Save and close.")
            print(f"  └─────────────────────────────────────────────────────────────────\n")
        else:
            vm_cmd.append('--non-interactive')
            print(f"  OPEN_INFRA_VERSION_MANAGER = False — creating '{infra_v}' from "
                  f"'{base_name}' without opening the editor.")
        result = subprocess.run(vm_cmd, cwd=paths.MAIN)
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
                plot_engineering_structures,
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

            # Engineering structures (tunnels/bridges) — SA only, mirrors the standalone
            # builder's 'sa_construct' map. Needs segments_composition.gpkg (the per-piece
            # construct types); skipped with a note when composition is absent/empty.
            _comp_path = os.path.join(paths.get_infra_version_dir(infra_v),
                                      'segments_composition.gpkg')
            _composition = gpd.read_file(_comp_path) if os.path.isfile(_comp_path) \
                else gpd.GeoDataFrame()
            if _composition.empty:
                print("    sa_engineering_structures.pdf — skipped (no composition data).")
            else:
                print(f"    sa_engineering_structures.pdf ...")
                _fig = plot_engineering_structures(
                    net_sa, _composition, extent=sa_ext,
                    output_path=plot_dir / 'sa_engineering_structures.pdf',
                    show_outside=True, nodes=nodes)
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

        svc_vm_cmd = [sys.executable, vm_script,
                      '--infra-version', infra_v,
                      '--network',       svc_network]
        if settings.OPEN_SVC_VERSION_MANAGER:
            print(f"  Opening version manager for '{svc_network}' ...")
        else:
            svc_vm_cmd.append('--non-interactive')
            print(f"  OPEN_SVC_VERSION_MANAGER = False — finalising '{svc_network}' "
                  f"without opening the editor.")
        result = subprocess.run(svc_vm_cmd, cwd=paths.MAIN)
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
        elif not cache_manifest.check_manifest(
                os.path.dirname(expected[0]), 'catchment_4a',
                {'svc_network': svc_network}):
            print("  use_cache_pt_catchment = True but manifest stale — running allocation.")
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
    settings.CATCHMENT_METHOD. Communal OD is the GVM-anchored blend at
    start_year_scenario (2018 actual scaled toward the symmetrised 2040 forecast;
    od_communal), out-of-catchment demand is routed to gateway (boundary) stations,
    and a top-10 origins/destinations Excel is exported for study-area stations.

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
    print(f"  Population base year  : {settings.start_year_scenario}\n")

    # ── Step 4B.2: Skip-if-cached check ──────────────────────────────────────
    expected = [paths.get_station_od_window_xlsx(svc_network, method, w)
                for w in ('peak', 'off_peak', 'full_day')]

    if settings.use_cache_stationsOD:
        missing = [f for f in expected if not os.path.exists(f)]
        if not missing and cache_manifest.check_manifest(
                paths.get_od_version_dir(svc_network), 'station_od_4b',
                {'svc_network': svc_network,
                 'infra_version': PIPELINE_CONFIG.infra_version}):
            print(f"  use_cache_stationsOD = True and all {len(expected)} expected "
                  f"OD matrices present — skipping Phase 4B.")
            runtimes["Phase 4B: Station OD Matrix"] = time.time() - st
            return
        if missing:
            print(f"  use_cache_stationsOD = True but {len(missing)} expected "
                  f"file(s) missing — running OD preparation.")
        else:
            print("  use_cache_stationsOD = True but manifest stale — running "
                  "OD preparation.")
    else:
        print("  use_cache_stationsOD = False — running OD preparation.")

    # ── Step 4B.3: Run OD preparation ────────────────────────────────────────
    # Interactive for gateway assignment: reuse if a saved assignment exists,
    # otherwise prompt to create it here (mirrors the municipal assignment).
    print("\n--- Step 4B.3: Run Station OD Preparation ---\n")
    import catchment_OD_preparation as _odp
    _odp._INTERACTIVE_MODE = True
    try:
        _odp.prepare_all_od_matrices(
            use_cache=settings.use_cache_stationsOD,
            svc_version=svc_network,
            infra_version=PIPELINE_CONFIG.infra_version,
            method=method,
            attribution_mode=settings.OD_ATTRIBUTION_MODE,
            make_plots=settings.PLOT_STATION_OD,
        )
    finally:
        # The interactive flag is enabled only for the gateway zone-assignment step;
        # reset it so it cannot leak into later phases (e.g. the Phase-4C infra-version
        # resolve, which is now automated via the threaded infra_version).
        _odp._INTERACTIVE_MODE = False

    _write_station_od_to_report(method, svc_network)
    runtimes["Phase 4B: Station OD Matrix"] = time.time() - st


def phase_4c_network_assignment(runtimes: dict) -> None:
    """Phase 4C — Passenger Routing.

    Assigns the Phase-4B station OD onto rail services via
    catchment_OD_rail_network.passenger_routing(), honouring
    settings.ROUTING_ASSIGNMENT_METHOD and the active cost model.
    """
    print("\n" + "=" * 80)
    print("PHASE 4C: PASSENGER ROUTING")
    print("=" * 80 + "\n")
    st = time.time()

    method = get_routing_od_method()   # 'pt_feeder' | 'municipal'
    svc_version = PIPELINE_CONFIG.svc_version
    if svc_version is None:
        svc_version = settings.SVC_VERSION
        if svc_version == 'Build_New':
            svc_version = settings.SVC_BUILD_NEW_NAME
    svc_network = f'{svc_version}_network'

    assignment_method = settings.ROUTING_ASSIGNMENT_METHOD
    if assignment_method == 'both':
        print("  WARNING: ROUTING_ASSIGNMENT_METHOD='both' is standalone-only — "
              "using 'logit' for the pipeline run.")
        assignment_method = 'logit'
    method_dirs = [assignment_method]

    print(f"  OD method            : {method}")
    print(f"  Assignment method    : {assignment_method}")
    print(f"  Service version      : {svc_version}  -> '{svc_network}'")
    print(f"  Cost model           : {settings.TRAVEL_COST_METHOD} / "
          f"{settings.TRANSFER_COST_MODEL}\n")

    expected = [
        os.path.join(paths.get_assignment_method_dir(svc_network, md),
                     'path_assignment.xlsx')
        for md in method_dirs
    ]
    if settings.use_cache_railRouting:
        missing = [f for f in expected if not os.path.exists(f)]
        if not missing and cache_manifest.check_manifest(
                paths.get_assignment_method_dir(svc_network, assignment_method),
                'assignment_4c',
                {'svc_network': svc_network,
                 'infra_version': PIPELINE_CONFIG.infra_version}):
            print(f"  use_cache_railRouting = True and all {len(expected)} expected "
                  f"outputs present — skipping Phase 4C.")
            runtimes["Phase 4C: Passenger Routing"] = time.time() - st
            return
        if missing:
            print(f"  use_cache_railRouting = True but {len(missing)} output(s) "
                  f"missing — running routing.")
        else:
            print("  use_cache_railRouting = True but manifest stale — running "
                  "routing.")
    else:
        print("  use_cache_railRouting = False — running routing.")

    print("\n--- Step 4C.1: Run Passenger Routing ---\n")
    import catchment_OD_rail_network as _pr
    _pr.passenger_routing(
        svc_version=svc_network,
        use_cache=settings.use_cache_railRouting,
        od_method=method,
        assignment_method=assignment_method,
        make_plots=settings.PLOT_ASSIGNMENT,
        infra_version=PIPELINE_CONFIG.infra_version)

    _write_assignment_to_report(method, svc_network, assignment_method)
    runtimes["Phase 4C: Passenger Routing"] = time.time() - st


def _write_assignment_to_report(method: str, svc_network: str,
                                assignment_method: str) -> None:
    """Append a 'NETWORK ASSIGNMENT (Phase 4C)' block to report_new.txt."""
    lines = [
        "=" * 80,
        "  NETWORK ASSIGNMENT (Phase 4C)",
        "=" * 80,
        f"  OD method            : {method}",
        f"  Assignment method    : {assignment_method}",
        f"  Service version      : {svc_network}",
        f"  Cost model           : {settings.TRAVEL_COST_METHOD} / "
        f"{settings.TRANSFER_COST_MODEL}",
        f"  Logit (K/maxT/theta) : {settings.ROUTING_K_PATHS} / "
        f"{settings.ROUTING_MAX_TRANSFERS} / {cp.LOGIT_ROUTE_THETA}",
        f"  Output dir           : data/Traffic_Flow/Assignment/{svc_network}/",
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


def _write_station_od_to_report(method: str, svc_network: str) -> None:
    """Append a 'STATION OD MATRIX (Phase 4B)' block to report_new.txt."""
    lines = [
        "=" * 80,
        "  STATION OD MATRIX (Phase 4B)",
        "=" * 80,
        f"  Method               : {method}",
        f"  Attribution mode     : {settings.OD_ATTRIBUTION_MODE}",
        f"  Service version      : {svc_network}",
        f"  Population base year  : {settings.start_year_scenario}",
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


def _phase5_base_infra() -> str:
    """Base infra version for Phase 5 — the network the pipeline propagated.

    Phase 5 (5A/5B/5C) discovers, composes and measures interventions on the SAME
    network the rest of the run used: the enhanced version produced by Phase 3B
    (PIPELINE_CONFIG.infra_version), not the unenhanced base name. Falls back to the
    configured base when the pipeline value is unset (partial runs).
    """
    import infra_ints_orchestrator as _io
    v = PIPELINE_CONFIG.infra_version
    return v if v and v != 'Build_New' else _io._resolve_base_version()


def phase_5a_infrastructure_interventions(sa_boundary, runtimes: dict) -> None:
    """Phase 5A: generate the infra-int registry + master tagged network.

    Discovers connecting curves (CC) against the base infra + service version and
    collects the capacity interventions (CAP) registered during Phase 3C, then
    materialises the master tagged 'Dev_Full' network. Gated by settings.INFRA_INT_MODE
    ('NONE' skips). Does not alter the baseline infra version; the per-svc-int deltas
    are consumed downstream (Phase 6) via infra_ints_orchestrator.compose_infra.

    Args:
        sa_boundary: study-area polygon, used to restrict CC discovery centres.
        runtimes:    dict tracking phase execution times.
    """
    if str(getattr(settings, 'INFRA_INT_MODE', 'NONE')).upper() == 'NONE':
        return {}
    print("\n" + "=" * 80)
    print("PHASE 5A: INFRASTRUCTURE INTERVENTIONS")
    print("=" * 80 + "\n")
    st = time.time()
    result: dict = {}
    try:
        import infra_ints_orchestrator as _io
        result = _io.phase_5a_infra_interventions(
            base_version=_phase5_base_infra(),
            svc_version=PIPELINE_CONFIG.svc_version or _io._resolve_svc_version(),
            polygon=sa_boundary,
        ) or {}
    except Exception as exc:
        print(f"  WARNING: Phase 5A failed: {exc}")
    runtimes["Phase 5A: Infrastructure Interventions"] = time.time() - st
    return result


def phase_5b_service_interventions(sa_boundary, sa_buffer, runtimes: dict,
                                   ndc_candidates=None) -> dict:
    """Phase 5B: discover, register and materialise the service-intervention catalogue.

    Generates EXT (line extensions) and/or NDC (new direct connections) per
    settings.SVC_INT_MODE, materialises each one's delta network (real infra TT) for
    the downstream 5C capacity pass, and writes the catalogue + delta plots. Gated by
    settings.SVC_INT_MODE ('NONE' skips). NDC consumes the connecting-curve candidates
    handed over by Phase 5A so the CC is not re-discovered.

    Args:
        sa_boundary: study-area polygon (EXT terminus / NDC scope gate).
        sa_buffer:   study-area buffer (EXT/NDC candidate-station extent).
        runtimes:    dict tracking phase execution times.
        ndc_candidates: 5A connecting-curve candidates (branch_a/b, requires_infra).
    """
    if str(getattr(settings, 'SVC_INT_MODE', 'NONE')).upper() == 'NONE':
        return {}
    print("\n" + "=" * 80)
    print("PHASE 5B: SERVICE INTERVENTIONS")
    print("=" * 80 + "\n")
    st = time.time()
    result: dict = {}
    try:
        import infra_ints_orchestrator as _io
        import svc_ints_orchestrator as _so
        result = _so.phase_5b_service_interventions(
            base_infra=_phase5_base_infra(),
            base_svc=PIPELINE_CONFIG.svc_version or _io._resolve_svc_version(),
            sa_polygon=sa_boundary, buffer_polygon=sa_buffer,
            ndc_candidates=ndc_candidates,
        ) or {}
    except Exception as exc:
        print(f"  WARNING: Phase 5B failed: {exc}")
    runtimes["Phase 5B: Service Interventions"] = time.time() - st
    return result


def phase_5c_capacity_on_matched(runtimes: dict, svc_int_ids=None) -> dict:
    """Phase 5C: per-svc-int capacity interventions on the matched composed network.

    For each registered svc-int, designs CAP on its own composed (base+CC+merged-services)
    network — one-shot resolve over the modified sections — registers/composes/plots the
    CAP like a CC, and writes the svc-int→CAP attribution table for the CBA. Gated by
    settings.SVC_INT_MODE (no svc-ints → nothing to do) and settings.CAPACITY_MODE
    ('None' skips). Phase 5C is the sole, cost-bearing CAP generator.

    Args:
        runtimes: dict tracking phase execution times.
        svc_int_ids: optional subset of svc-int ids (None = all registered;
            mirrors phase_6_intervention_recompute). NOTE: a subset run with
            use_cache_svc_int_cap=False rewrites the attribution CSV with the
            subset's rows only.
    """
    if str(getattr(settings, 'SVC_INT_MODE', 'NONE')).upper() == 'NONE':
        return {}
    if str(getattr(settings, 'CAPACITY_MODE', 'None')) == 'None':
        return {}
    print("\n" + "=" * 80)
    print("PHASE 5C: CAPACITY ON THE MATCHED NETWORK")
    print("=" * 80 + "\n")
    st = time.time()
    result: dict = {}
    try:
        import infra_ints_orchestrator as _io
        import mixed_ints_orchestrator as _mix
        result = _mix.phase_5c_capacity_on_matched(
            base_infra=_phase5_base_infra(),
            base_svc=PIPELINE_CONFIG.svc_version or _io._resolve_svc_version(),
            make_plots=getattr(settings, 'PLOT_MIXED_INTS', False),
            svc_int_ids=svc_int_ids,
        ) or {}
    except Exception as exc:
        print(f"  WARNING: Phase 5C failed: {exc}")
    runtimes["Phase 5C: Capacity on Matched Network"] = time.time() - st
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6 — Intervention recompute (per-svc-int catchment / OD / routing / flows)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_affected_sets(combo: str, base_infra: str) -> dict:
    """Read the Hook-1 affected-set CSV into dict[int_id -> {stations, services}].

    stations = set[int] id_points; services = list[str] variant_keys. Missing
    file or missing svc-int rows degrade to empty sets (the 6C closure then
    falls back to endpoint/OD-changed pairs only, with a warning upstream).
    """
    path = os.path.join(paths.get_svc_int_catalogue_dir(combo),
                        f'svc_int_affected_set_{base_infra}.csv')
    if not os.path.exists(path):
        print(f"  WARNING: affected-set CSV missing at {path} — closures will "
              f"use endpoint pairs only. Re-run Phase 5B to produce it.")
        return {}
    df = pd.read_csv(path, encoding='utf-8-sig')
    out = {}
    for _, r in df.iterrows():
        raw_st = '' if pd.isna(r.get('affected_stations')) else str(r.get('affected_stations'))
        raw_sv = '' if pd.isna(r.get('affected_services')) else str(r.get('affected_services'))
        out[str(r['int_id'])] = {
            'stations': {int(float(t)) for t in raw_st.split(',') if t.strip()},
            'services': [t.strip() for t in raw_sv.split(',') if t.strip()],
        }
    return out


_PARITY_TOL = 1e-6


def _phase6_snapshot(int_network: str, method: str, include_pt: bool) -> dict:
    """Read the per-svc-int tables the parity check compares (routing
    primitives + the PT_Feeder long OD / allocation) from disk into memory."""
    snap = {t: pd.read_parquet(paths.get_routing_primitive_path(int_network,
                                                                method, t))
            for t in ('paths', 'segments', 'events')}
    if include_pt:
        attribution = settings.OD_ATTRIBUTION_MODE.strip().lower()
        snap['od_long'] = pd.read_csv(
            paths.get_station_od_long_csv(int_network, 'pt_feeder', attribution),
            encoding='utf-8-sig')
        import catchment_base as _cb
        _cb.setup_versioned_dirs(int_network)
        snap['allocation'] = pd.read_parquet(os.path.join(
            _cb.PT_FEEDER_DATA_DIR, 'allocation_pt_feeder.parquet'))
    return snap


def _phase6_parity(sel: dict, orc: dict) -> tuple:
    """Compare selective vs oracle tables (architecture decision C).

    Returns (summary_rows, pair_rows): per-table n_rows_compared / n_mismatch /
    max_abs_diff at tolerance 1e-6, and one row per mismatching OD pair with a
    `residual_candidate` flag — True when the pair's oracle path transfers at a
    station absent from its selective path (the documented transfer-through
    residual of the closure). Reported, not failed.
    """
    rows = []
    mismatch_pairs: dict = {}

    def _pairgrain(df):
        best = df[df['path_id'] == 0]
        return best.groupby(['origin_id', 'dest_id']).agg(
            jt=('journey_time_min', 'max'), gc=('gc_min', 'max'),
            tr=('n_transfers', 'max'), trips=('trips', 'sum'))

    j = _pairgrain(sel['paths']).join(_pairgrain(orc['paths']),
                                      lsuffix='_s', rsuffix='_o', how='outer')
    one_sided = j.isna().any(axis=1)
    for col, name in (('jt', 'skim_journey_time'), ('gc', 'skim_gc'),
                      ('tr', 'skim_transfers'), ('trips', 'pair_trips')):
        d = (j[f'{col}_s'] - j[f'{col}_o']).abs()
        bad = set(d[d > _PARITY_TOL].index) | set(j.index[one_sided])
        rows.append({'table': name, 'n_rows_compared': len(j),
                     'n_mismatch': len(bad),
                     'max_abs_diff': float(d.max()) if d.notna().any() else 0.0})
        for p in bad:
            mismatch_pairs.setdefault(p, set()).add(name)
    gc_diff = (j['gc_s'] - j['gc_o']).abs()

    def _cmp_keyed(name, s, o, keys, val='trips'):
        a = s.groupby(keys)[val].sum().rename('s')
        b = o.groupby(keys)[val].sum().rename('o')
        m = a.to_frame().join(b, how='outer').fillna(0.0)
        d = (m['s'] - m['o']).abs()
        rows.append({'table': name, 'n_rows_compared': len(m),
                     'n_mismatch': int((d > _PARITY_TOL).sum()),
                     'max_abs_diff': float(d.max()) if len(d) else 0.0})

    _cmp_keyed('segment_loads', sel['segments'], orc['segments'],
               ['from_id', 'to_id', 'variant_key'])
    _cmp_keyed('station_events', sel['events'], orc['events'],
               ['station_id', 'event', 'variant_key'])
    if 'od_long' in sel:
        _cmp_keyed('od_long', sel['od_long'], orc['od_long'],
                   ['origin_station_id', 'dest_station_id'])
    if 'allocation' in sel:
        a = sel['allocation'].set_index('RELI')['id_point']
        b = orc['allocation'].set_index('RELI')['id_point']
        m = a.rename('s').to_frame().join(b.rename('o'), how='outer')
        neq = (pd.to_numeric(m['s'], errors='coerce').fillna(-9)
               != pd.to_numeric(m['o'], errors='coerce').fillna(-9))
        rows.append({'table': 'allocation', 'n_rows_compared': len(m),
                     'n_mismatch': int(neq.sum()), 'max_abs_diff': float('nan')})

    # Residual-candidate flag for the mismatching pairs
    pair_rows = []
    if mismatch_pairs:
        mp = set(mismatch_pairs)

        def _pairsets(df, id_cols, only_transfer=False):
            d = df
            if only_transfer:
                d = d[d['event'] == 'transfer']
            k = pd.Series(list(zip(d['origin_id'].astype(int),
                                   d['dest_id'].astype(int))), index=d.index)
            d = d[k.isin(mp).values]
            out: dict = {}
            for r in d.itertuples(index=False):
                key = (int(r.origin_id), int(r.dest_id))
                st = out.setdefault(key, set())
                for c in id_cols:
                    st.add(str(getattr(r, c)).lstrip('x'))
            return out

        orc_transfers = _pairsets(orc['events'], ['station_id'],
                                  only_transfer=True)
        sel_stations = _pairsets(sel['segments'], ['from_id', 'to_id'])
        n_residual = 0
        for p in sorted(mp):
            flag = bool(orc_transfers.get(p, set())
                        - sel_stations.get(p, set()))
            n_residual += int(flag)
            pair_rows.append({
                'origin_id': p[0], 'dest_id': p[1],
                'mismatch_tables': '|'.join(sorted(mismatch_pairs[p])),
                'gc_abs_diff': float(gc_diff.get(p, float('nan'))),
                'residual_candidate': flag})
        rows.append({'table': 'residual_candidates',
                     'n_rows_compared': len(mp), 'n_mismatch': n_residual,
                     'max_abs_diff': float('nan')})
    return rows, pair_rows


def _phase6_worker(int_type: str, rec: dict, merged_dir: str,
                   composed_infra: str, aff: dict,
                   ctx: dict, capture_log: bool = True) -> dict:
    """Phase-6 body for one svc-int: 6A/6B (PT_Feeder only), 6C, 6D + diff.

    Runs in a loky child process when PHASE6_N_JOBS > 1, so every
    runtime-resolved decision arrives via ``ctx`` — children re-import settings
    from file and must never read PIPELINE_CONFIG or mutated module state.
    With ``capture_log`` the worker's stdout is buffered and returned so the
    parent prints each svc-int's log as one ordered block.

    Args:
        int_type:   'ext' | 'ndc'.
        rec:        svc-int record (read_records row).
        merged_dir: merged developed-network dir from the serial pre-pass.
        composed_infra: the svc-int's CC-only composed infra version from the
                    serial pre-pass (== base_infra for EXT) — 6D unrolls onto
                    it; the projected links stay under the base-infra dirname.
        aff:        {'stations': set[int], 'services': list[str]} (Hook-1).
        ctx:        dict(svc_network, base_infra, combo, od_method,
                    assignment_method, catchment_method, od_attribution_mode,
                    plot_int_recompute, plot_flows, use_cache_flows,
                    write_workbooks).
        capture_log: buffer stdout and return it (parallel mode).

    Returns:
        dict(iid, int_type, status 'done'|'fail', error, log).
    """
    import contextlib
    import io
    import traceback

    iid = str(rec['int_id'])
    buf = io.StringIO()
    redirect = (contextlib.redirect_stdout(buf) if capture_log
                else contextlib.nullcontext())
    status, error = 'done', ''
    with redirect:
        print(f"\n--- Phase 6 [{iid}] ({int_type.upper()}) ---")
        try:
            import catchment_OD_rail_network as _pr
            import passenger_flows as _pf

            int_network = paths.svc_int_network_name(iid, ctx['combo'])
            if not aff['stations'] and not aff['services']:
                print(f"  WARNING: no affected set for {iid} — closure falls "
                      f"back to endpoint/OD-changed pairs only.")

            od_changed_pairs = None
            od_long_dev = None
            if ctx['catchment_method'] == 'PT_Feeder':
                import catchment_allocate as _ca
                import catchment_OD_preparation as _odp
                # loky children have no stdin; the module default is interactive.
                _odp._INTERACTIVE_MODE = False
                res6a = _ca.reallocate_for_svc_int(
                    iid, aff['stations'], ctx['svc_network'],
                    dev_rail_base=merged_dir,
                    make_plots=ctx['plot_int_recompute'],
                    combo=ctx['combo'])
                res6b = _odp.prepare_svc_int_od(
                    iid, ctx['svc_network'], ctx['base_infra'],
                    res6a['affected_communes'],
                    attribution=ctx['od_attribution_mode'])
                od_changed_pairs = res6b['changed_pairs']
                od_long_dev = res6b['od_routing']
                print(f"  [{iid}] 6A/6B done: {res6a['n_cells_affected']:,} "
                      f"affected cell(s), {len(res6a['affected_communes'])} "
                      f"commune(s), {len(res6b['changed_gateways'])} changed "
                      f"gateway(s), {len(od_changed_pairs):,} changed OD "
                      f"pair(s).")

            res = _pr.route_svc_int(
                iid, ctx['svc_network'], aff['stations'], aff['services'],
                rail_base=merged_dir, infra_version=ctx['base_infra'],
                od_changed_pairs=od_changed_pairs, od_long_dev=od_long_dev,
                method=ctx['assignment_method'], od_method=ctx['od_method'],
                make_plots=ctx['plot_int_recompute'],
                write_workbooks=ctx['write_workbooks'])
            print(f"  [{iid}] 6C done: {res['n_pairs_closure']:,} of "
                  f"{res['n_pairs_total']:,} pairs re-routed; routed "
                  f"{res['routed_trips']:,.1f} / unresolved "
                  f"{res['unresolved_trips']:,.1f} trips.")

            _pf.build_passenger_flows(int_network, composed_infra,
                                      method=ctx['assignment_method'],
                                      make_plots=ctx['plot_flows'],
                                      use_cache=ctx['use_cache_flows'],
                                      links_infra_version=ctx['base_infra'])
            _pf.build_flow_diff(ctx['svc_network'], int_network,
                                ctx['base_infra'],
                                method=ctx['assignment_method'],
                                make_plots=ctx['plot_flows'],
                                dev_infra_version=composed_infra)
        except Exception as exc:
            status = 'fail'
            error = f"{exc}\n{traceback.format_exc()}"
    return {'iid': iid, 'int_type': int_type, 'status': status,
            'error': error, 'log': buf.getvalue() if capture_log else ''}


def _phase6_run_oracle(iid: str, merged_dir: str, composed_infra: str,
                       aff: dict, ctx: dict,
                       combo: str, parity_results: dict) -> None:
    """Full-recompute oracle + parity report for one svc-int (decision C).

    Serial-only: PHASE6_N_JOBS is forced to 1 when use_full_recompute_ints is
    on, so this never runs inside a worker. Final on-disk outputs = oracle's.
    """
    import catchment_OD_rail_network as _pr
    import passenger_flows as _pf

    int_network = paths.svc_int_network_name(iid, ctx['combo'])
    include_pt = ctx['catchment_method'] == 'PT_Feeder'
    sel_snap = _phase6_snapshot(int_network, ctx['assignment_method'],
                                include_pt)
    print(f"\n--- Phase 6 ORACLE [{iid}]: full recompute "
          f"(decision-C parity) ---")
    od_changed_o = od_dev_o = None
    if include_pt:
        import catchment_allocate as _ca
        import catchment_OD_preparation as _odp
        res6a_o = _ca.reallocate_for_svc_int(
            iid, aff['stations'], ctx['svc_network'],
            dev_rail_base=merged_dir, make_plots=False,
            full_recompute=True, combo=ctx['combo'])
        res6b_o = _odp.prepare_svc_int_od(
            iid, ctx['svc_network'], ctx['base_infra'],
            res6a_o['affected_communes'],
            attribution=ctx['od_attribution_mode'],
            full_recompute=True)
        od_changed_o = res6b_o['changed_pairs']
        od_dev_o = res6b_o['od_routing']
    _pr.route_svc_int(
        iid, ctx['svc_network'], aff['stations'], aff['services'],
        rail_base=merged_dir, infra_version=ctx['base_infra'],
        od_changed_pairs=od_changed_o, od_long_dev=od_dev_o,
        method=ctx['assignment_method'], od_method=ctx['od_method'],
        make_plots=False, full_recompute=True,
        write_workbooks=ctx['write_workbooks'])
    _pf.build_passenger_flows(int_network, composed_infra,
                              method=ctx['assignment_method'],
                              make_plots=False, use_cache=False,
                              links_infra_version=ctx['base_infra'])
    _pf.build_flow_diff(ctx['svc_network'], int_network, ctx['base_infra'],
                        method=ctx['assignment_method'], make_plots=False,
                        dev_infra_version=composed_infra)
    orc_snap = _phase6_snapshot(int_network, ctx['assignment_method'],
                                include_pt)
    summary, pair_rows = _phase6_parity(sel_snap, orc_snap)
    ppath = paths.get_parity_csv(combo, iid)
    os.makedirs(os.path.dirname(ppath), exist_ok=True)
    pd.DataFrame(summary).to_csv(ppath, index=False, encoding='utf-8-sig')
    if pair_rows:
        pd.DataFrame(pair_rows).to_csv(
            ppath.replace('.csv', '_pairs.csv'), index=False,
            encoding='utf-8-sig')
    parity_results[iid] = summary
    worst = max((r['max_abs_diff'] for r in summary
                 if r['max_abs_diff'] == r['max_abs_diff']),
                default=0.0)
    print(f"  [{iid}] parity -> {ppath} "
          f"(worst max_abs_diff {worst:.3g}; final on-disk "
          f"outputs = oracle's)")


def phase_6_intervention_recompute(sa_boundary, ca_boundary, runtimes: dict,
                                   svc_int_ids=None) -> None:
    """Phase 6 — per-svc-int selective recompute against the Phase-4 baseline.

    Loops the Phase-5B svc-int catalogue and recomputes only what each svc-int
    changes (architecture decision C): 6A allocation / 6B OD (PT_Feeder only —
    Municipal allocation is frequency-blind, so its OD is intervention-
    invariant), 6C routing and 6D flows (always). 6C re-routes the affected
    closure on the merged developed network and re-derives the Phase-4C outputs
    under <svc_int_id>_network paths, so Phases 7/8 consume them exactly like the
    baseline. Gated by settings.SVC_INT_MODE ('NONE' skips).

    Args:
        sa_boundary: study-area polygon (6A scope, PT_Feeder path).
        ca_boundary: catchment-area polygon (6A scope, PT_Feeder path).
        runtimes:    dict tracking phase execution times.
        svc_int_ids: optional subset of svc-int ids to process (None = all
                     registered; mirrors phase_5c_capacity_on_matched).
    """
    if str(getattr(settings, 'SVC_INT_MODE', 'NONE')).upper() == 'NONE':
        return
    print("\n" + "=" * 80)
    print("PHASE 6: INTERVENTION RECOMPUTE")
    print("=" * 80 + "\n")
    st = time.time()

    import svc_ints_orchestrator as _so

    base_infra = _phase5_base_infra()
    svc_version = PIPELINE_CONFIG.svc_version
    if svc_version is None:
        svc_version = settings.SVC_VERSION
        if svc_version == 'Build_New':
            svc_version = settings.SVC_BUILD_NEW_NAME
    svc_network = f'{svc_version}_network'
    combo = f'{base_infra}__{svc_version}'
    od_method = get_routing_od_method()
    catchment_method = settings.CATCHMENT_METHOD

    assignment_method = settings.ROUTING_ASSIGNMENT_METHOD
    if assignment_method == 'both':
        print("  WARNING: ROUTING_ASSIGNMENT_METHOD='both' is standalone-only — "
              "using 'logit' for the pipeline run.")
        assignment_method = 'logit'

    records = [(t, r) for t in ('ext', 'ndc')
               for r in _so.read_records(t, network=combo)]
    if svc_int_ids is not None:
        want = {str(i) for i in svc_int_ids}
        records = [(t, r) for t, r in records if str(r['int_id']) in want]
    if not records:
        print(f"  No svc-ints registered for combo '{combo}' — nothing to do.")
        runtimes["Phase 6: Intervention Recompute"] = time.time() - st
        return
    affected = _load_affected_sets(combo, base_infra)

    import passenger_flows as _pf
    print("\n--- Phase 6D: baseline passenger flows ---")
    try:
        _pf.build_passenger_flows(svc_network, base_infra,
                                  method=assignment_method,
                                  make_plots=settings.PLOT_FLOWS,
                                  use_cache=settings.use_cache_flows)
    except Exception as exc:
        print(f"  WARNING: baseline flows failed: {exc}")

    gate_note = ('6A/6B skipped (Municipal: allocation is frequency-blind, OD '
                 'invariant)' if catchment_method == 'Municipal'
                 else '6A/6B run (PT_Feeder)')
    print(f"  Combo               : {combo}")
    print(f"  Svc-ints            : {len(records)}")
    print(f"  Catchment method    : {catchment_method} -> {gate_note}")
    print(f"  OD / assignment     : {od_method} / {assignment_method}")
    print(f"  Oracle              : use_full_recompute_ints="
          f"{getattr(settings, 'use_full_recompute_ints', False)} "
          f"(parity check lands in plan Phase 4)")

    oracle = bool(getattr(settings, 'use_full_recompute_ints', False))
    n_jobs = max(1, int(getattr(settings, 'PHASE6_N_JOBS', 1)))
    if oracle and n_jobs > 1:
        print("  use_full_recompute_ints = True — forcing serial Phase 6 "
              "(oracle + parity stay debuggable).")
        n_jobs = 1

    # Everything runtime-resolved the workers need: loky children re-import
    # settings from file, so values mutated at runtime only reach them here.
    ctx = {'svc_network': svc_network, 'base_infra': base_infra,
           'combo': combo,
           'od_method': od_method, 'assignment_method': assignment_method,
           'catchment_method': catchment_method,
           'od_attribution_mode': settings.OD_ATTRIBUTION_MODE,
           'plot_int_recompute': settings.PLOT_INT_RECOMPUTE,
           'plot_flows': settings.PLOT_FLOWS,
           'use_cache_flows': settings.use_cache_flows,
           'write_workbooks': settings.WRITE_INT_WORKBOOKS}

    # Serial pre-pass: cache probe + delta materialisation. apply/merge can
    # compose into the SHARED Developments/Derived/<hash8>/ dirs (two svc-ints
    # requiring the same CC collide), so they never run concurrently; after
    # 5B/5C they are cache-hits.
    n_done = n_skip = n_fail = 0
    todo: list = []
    for int_type, rec in records:
        iid = str(rec['int_id'])
        try:
            int_network = paths.svc_int_network_name(iid, combo)
            probe = [paths.get_routing_primitive_path(int_network,
                                                      assignment_method, t)
                     for t in ('paths', 'segments', 'events', 'unresolved')]
            if (settings.use_cache_int_recompute
                    and all(os.path.exists(p) for p in probe)
                    and cache_manifest.check_manifest(
                        paths.get_assignment_method_dir(int_network,
                                                        assignment_method),
                        'assignment_4c',
                        {'svc_network': svc_network,
                         'infra_version': base_infra,
                         'svc_int_id': iid})):
                print(f"  use_cache_int_recompute = True and outputs present — "
                      f"skipping {iid}.")
                n_skip += 1
                continue

            _so.apply_svc_int(rec, svc_version, base_infra, use_cache=True)
            merged_dir = _so.build_merged_unprojected(rec, svc_version,
                                                      base_infra, use_cache=True)
            if not merged_dir:
                print(f"  {iid}: empty delta — network equals baseline, skipping.")
                n_skip += 1
                continue
            # CC-only composed version for the 6D unroll, resolved serially
            # (Derived compose race) — NOT from apply_svc_int's return, whose
            # cached path reports base_infra. compose_infra([]) == base_infra.
            import ints_core as _core
            composed = _core.compose_infra(
                base_infra, list(rec.get('requires_infra') or []),
                svc_version=svc_version)
            todo.append((int_type, rec, merged_dir, composed,
                         affected.get(iid, {'stations': set(),
                                            'services': []})))
        except Exception as exc:
            print(f"  WARNING: Phase 6 pre-pass failed for {iid}: {exc}")
            n_fail += 1

    parity_results: dict = {}
    if n_jobs == 1 or len(todo) <= 1:
        for int_type, rec, merged_dir, composed, aff in todo:
            r = _phase6_worker(int_type, rec, merged_dir, composed, aff, ctx,
                               capture_log=False)
            if r['status'] != 'done':
                print(f"  WARNING: Phase 6 failed for {r['iid']}: {r['error']}")
                n_fail += 1
                continue
            if oracle:
                try:
                    _phase6_run_oracle(r['iid'], merged_dir, composed, aff,
                                       ctx, combo, parity_results)
                except Exception as exc:
                    print(f"  WARNING: Phase 6 oracle failed for {r['iid']}: "
                          f"{exc}")
                    n_fail += 1
                    continue
            n_done += 1
    else:
        # loky children inherit the env — pin the non-interactive backend
        # before any worker imports matplotlib.
        os.environ.setdefault('MPLBACKEND', 'Agg')
        from joblib import Parallel, delayed
        print(f"\n  Phase 6 parallel: {len(todo)} svc-int(s) on {n_jobs} loky "
              f"workers — per-svc-int logs print as each completes.")
        try:
            results = Parallel(n_jobs=n_jobs, backend='loky',
                               return_as='generator')(
                delayed(_phase6_worker)(t, r, m, c, a, ctx)
                for t, r, m, c, a in todo)
        except TypeError:                       # joblib < 1.3: no return_as
            results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(_phase6_worker)(t, r, m, c, a, ctx)
                for t, r, m, c, a in todo)
        for r in results:
            if r['log']:
                print(r['log'], end='' if r['log'].endswith('\n') else '\n')
            if r['status'] == 'done':
                n_done += 1
            else:
                print(f"  WARNING: Phase 6 failed for {r['iid']}: {r['error']}")
                n_fail += 1

    print(f"\n  Phase 6 summary: {n_done} recomputed, {n_skip} skipped, "
          f"{n_fail} failed (of {len(records)}).")
    if parity_results:
        tables = []
        for summary in parity_results.values():
            for r in summary:
                if r['table'] not in tables:
                    tables.append(r['table'])
        print("\n  Parity summary (selective vs full-recompute oracle, "
              f"tol {_PARITY_TOL:g}):")
        print(f"    {'svc-int':<14}" + ''.join(f"{t:>22}" for t in tables))
        for iid, summary in parity_results.items():
            by_t = {r['table']: r for r in summary}
            cells = []
            for t in tables:
                r = by_t.get(t)
                if r is None:
                    cells.append(f"{'-':>22}")
                else:
                    mad = r['max_abs_diff']
                    mad_s = '-' if mad != mad else f"{mad:.2g}"
                    cells.append(f"{r['n_mismatch']:,}/{r['n_rows_compared']:,}"
                                 f" ({mad_s})".rjust(22))
            print(f"    {iid:<14}" + ''.join(cells))
    runtimes["Phase 6: Intervention Recompute"] = time.time() - st


def phase_7_scenarios(runtimes: dict, svc_int_ids=None) -> None:
    """Phase 7 — demand-growth factor store + per-svc-int overrides.

    Builds the baseline per-station growth-factor vectors (scenario x year;
    seeded LHS, so baseline and every svc-int see identical stochastic paths)
    plus the station-independent modal/distance scalars, then one override
    table per registered PT_Feeder svc-int whose 6A allocation changed.
    Phase 8A composes scenario ODs on demand via
    random_scenarios.compose_scenario_od. Only scenario_type 'GENERATED' is
    implemented ('STATIC_9' / 'dummy' are deferred).

    Args:
        runtimes:    dict tracking phase execution times.
        svc_int_ids: optional subset of svc-int ids to process (None = all
                     registered; mirrors phase_6_intervention_recompute).
    """
    print("\n" + "=" * 80)
    print("PHASE 7: SCENARIOS")
    print("=" * 80 + "\n")
    st = time.time()

    if settings.scenario_type != 'GENERATED':
        raise ValueError(
            f"scenario_type '{settings.scenario_type}' is not wired into "
            f"main_new — 'STATIC_9' and 'dummy' are deferred; use 'GENERATED'.")

    import random_scenarios as _rs

    base_infra = _phase5_base_infra()
    svc_version = PIPELINE_CONFIG.svc_version
    if svc_version is None:
        svc_version = settings.SVC_VERSION
        if svc_version == 'Build_New':
            svc_version = settings.SVC_BUILD_NEW_NAME
    od_method = get_routing_od_method()
    attribution = (settings.OD_ATTRIBUTION_MODE if od_method == 'pt_feeder'
                   else 'municipal')
    print(f"  Service version     : {svc_version}")
    print(f"  Infrastructure      : {base_infra}")
    print(f"  OD method / attrib. : {od_method} / {attribution}")
    print(f"  Scenarios           : {settings.amount_of_scenarios} "
          f"({settings.start_year_scenario}-{settings.end_year_scenario}, "
          f"seeded LHS)")

    _rs.build_scenario_factor_store(
        svc_version, base_infra, od_method, attribution,
        n_scenarios=settings.amount_of_scenarios,
        start_year=settings.start_year_scenario,
        end_year=settings.end_year_scenario,
        make_plots=settings.PLOT_SCENARIOS,
        use_cache=settings.use_cache_scenarios)

    if str(getattr(settings, 'SVC_INT_MODE', 'NONE')).upper() == 'NONE':
        print("\n  SVC_INT_MODE = NONE — baseline factor store only.")
        runtimes["Phase 7: Scenarios"] = time.time() - st
        return
    if od_method != 'pt_feeder':
        print("\n  Municipal OD is intervention-invariant — no per-svc-int "
              "overrides (compose uses the baseline factors).")
        runtimes["Phase 7: Scenarios"] = time.time() - st
        return

    import svc_ints_orchestrator as _so
    combo = f'{base_infra}__{svc_version}'
    records = [(t, r) for t in ('ext', 'ndc')
               for r in _so.read_records(t, network=combo)]
    if svc_int_ids is not None:
        want = {str(i) for i in svc_int_ids}
        records = [(t, r) for t, r in records if str(r['int_id']) in want]
    if not records:
        print(f"\n  No svc-ints registered for combo '{combo}' — baseline "
              f"factor store only.")
        runtimes["Phase 7: Scenarios"] = time.time() - st
        return

    n_done = n_skip = n_fail = 0
    for _, rec in records:
        iid = str(rec['int_id'])
        try:
            res = _rs.build_svc_int_factor_overrides(
                iid, svc_version, base_infra, od_method, attribution,
                n_scenarios=settings.amount_of_scenarios,
                start_year=settings.start_year_scenario,
                end_year=settings.end_year_scenario,
                use_cache=settings.use_cache_scenarios)
            if res.get('cached') or res.get('overrides_path') is None:
                n_skip += 1
            else:
                n_done += 1
        except Exception as exc:
            print(f"  WARNING: Phase 7 overrides failed for {iid}: {exc}")
            n_fail += 1

    print(f"\n  Phase 7 summary: {n_done} override table(s) written, "
          f"{n_skip} skipped (cached / no allocation change), {n_fail} failed "
          f"(of {len(records)} svc-int(s)).")
    runtimes["Phase 7: Scenarios"] = time.time() - st


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
    phase_4c_network_assignment(runtimes)
    infra5a = phase_5a_infrastructure_interventions(sa_boundary, runtimes)
    phase_5b_service_interventions(sa_boundary, sa_buffer, runtimes,
                                   ndc_candidates=(infra5a or {}).get('ndc_candidates'))
    phase_5c_capacity_on_matched(runtimes)
    phase_6_intervention_recompute(sa_boundary, ca_boundary, runtimes)
    phase_7_scenarios(runtimes)

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
