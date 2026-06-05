"""catchment_base.py
Last modified: 2026-05-15

Shared foundation for the catchment allocation pipeline.

Step 1 — Study and catchment area definition
    Dissolve admin boundaries into a single polygon, export study area and
    catchment area as GeoPackages, and validate containment.  Callable from
    main_new.py (phase_1_initialisation) or interactively via initialise().

Step 2 — Population and employment grid preparation
    Load the 2023 federal raster CSVs, filter to the catchment boundary, scale
    cell values to any target year using commune-level cantonal data, cumulate
    totals per municipality, and write year-tagged CSV caches.

Step 3 — Visualisation
    Choropleth and raster PDF maps for population and employment.

Both catchment methods (Municipal and PT-Feeder) — implemented in
catchment_allocate.py — import the Step 2/3 primitives so that module stays
focused on Step 4 onward.

CRS: EPSG:2056 (Swiss LV95) throughout.
"""

import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib_map_utils.core.north_arrow import north_arrow
from shapely.geometry import box
from shapely.prepared import prep

import paths
import settings


# ===============================================================================
# CONSTANTS — shared by both catchment methods
# ===============================================================================

CODEBASE_CRS = 'EPSG:2056'

# Raster cell size (population/employment grids)
CELL_SIZE_M = 100

# Output directories
CATCHMENT_DATA_DIR     = paths.CATCHMENT_AREA_DIR                   # data/Catchment_Area
CATCHMENT_PLOT_DIR     = os.path.join('plots', 'Catchment_Area')    # plots/Catchment_Area
MUNICIPAL_DATA_DIR     = os.path.join(CATCHMENT_DATA_DIR, 'Municipal')
PT_FEEDER_DATA_DIR     = os.path.join(CATCHMENT_DATA_DIR, 'PT_Feeder')
MUNICIPAL_PLOT_DIR     = os.path.join(CATCHMENT_PLOT_DIR, 'Municipal')
PT_FEEDER_PLOT_DIR     = os.path.join(CATCHMENT_PLOT_DIR, 'PT_Feeder')
GUETEKLASSEN_PLOT_DIR  = os.path.join(CATCHMENT_PLOT_DIR, 'Gueteklassen')
OD_COMPARISON_PLOT_DIR = os.path.join('plots', 'Traffic_Flow', 'OD', 'Comparison')

# Aliases used by boundary-configuration helpers below
CANTONS_GPKG              = paths.CANTON_BOUNDARIES_GPKG
BEZIRKE_GPKG              = paths.BEZIRKE_BOUNDARIES_GPKG
MUNICIPALITIES_GPKG       = paths.MUNICIPAL_BOUNDARIES_GPKG
CATCHMENT_AREA_DIR        = os.path.join(CATCHMENT_DATA_DIR, 'Boundaries')
STUDY_AREA_DIR            = os.path.join(CATCHMENT_DATA_DIR, 'Boundaries')
POP_EMPL_DATA_DIR         = os.path.join(CATCHMENT_DATA_DIR, 'Pop_Empl_Data')
POP_EMPL_PLOT_DIR         = os.path.join(CATCHMENT_PLOT_DIR, 'Pop_Empl_Data')
STUDY_AREA_DEFAULT_BUFFER_M     = 3000
CATCHMENT_AREA_DEFAULT_BUFFER_M = 5000

# Module-level buffers for the Bezirk scaling report; populated by _scale_cells_to_year,
# reset and consumed by main().
_SCALE_REPORT_LINES: list = []
_BEZIRK_SCALES: dict = {}   # label -> bezirk_scale Series
_TOTALS: dict = {}          # label -> {'base': float, 'target': float}


# ===============================================================================
# SHARED UTILITIES
# ===============================================================================

def hamilton_round(fractional: np.ndarray, total: int) -> np.ndarray:
    """Largest-remainder integer rounding that preserves the input total exactly.

    Each element of ``fractional`` is split into integer-floor + remainder. The
    `total - floor.sum()` remaining units are distributed one at a time to the
    indices with the largest fractional remainder (ties broken by lower index).
    The returned array sums to ``total`` exactly.

    Args:
        fractional: 1-D non-negative array of floats.
        total:      integer the returned array must sum to.

    Returns:
        1-D ndarray of int with the same shape as fractional and sum == total.
        Returns an all-zeros array when fractional sums to zero.
    """
    arr = np.asarray(fractional, dtype=float).clip(min=0.0)
    total = int(total)

    if arr.size == 0 or arr.sum() <= 0:
        return np.zeros(arr.shape, dtype=int)

    floors = np.floor(arr).astype(int)
    remainder = total - int(floors.sum())

    if remainder == 0:
        return floors
    if remainder > 0:
        # Distribute remaining units to largest fractional parts (ties → lower index)
        fractions = arr - floors
        order = np.argsort(-fractions, kind='stable')
        bump = order[:remainder]
        floors[bump] += 1
        return floors

    # remainder < 0: floors already overshoot total (only possible when
    # `total` is less than sum(floor) — rare, but handle gracefully by
    # subtracting one unit from indices with smallest fractional parts first).
    take = -remainder
    fractions = arr - floors
    order = np.argsort(fractions, kind='stable')
    for idx in order:
        if take == 0:
            break
        if floors[idx] > 0:
            floors[idx] -= 1
            take -= 1
    return floors


# ===============================================================================
# BOUNDARY CONFIGURATION
# ===============================================================================

def _load_boundary_names(gpkg_path, objektart_filter=None):
    """Return sorted list of unique 'name' values from a boundary GeoPackage."""
    layers = gpd.list_layers(gpkg_path)
    gdf = gpd.read_file(gpkg_path, layer=layers.iloc[0]['name'])
    if objektart_filter is not None and 'objektart' in gdf.columns:
        gdf = gdf[gdf['objektart'] == objektart_filter]
    return sorted(gdf['name'].dropna().unique().tolist())


def _select_names_interactive(available_names, entity_label):
    """Pick one or more names from available_names via partial-name search."""
    print(f"\n   Available {entity_label} ({len(available_names)} total).")
    print(f"   Type comma-separated names (or partial names) to search & select.")
    print(f"   Leave empty to finish.\n")

    selected = []
    while True:
        raw = input(f"   Search / select {entity_label}: ").strip()
        if not raw:
            break

        terms = [t.strip() for t in raw.split(',') if t.strip()]
        batch = []
        for term in terms:
            exact = [n for n in available_names if n.lower() == term.lower()]
            if exact:
                batch.extend(exact)
                continue
            partial = [n for n in available_names if term.lower() in n.lower()]
            if not partial:
                print(f"     No match for '{term}'.")
            elif len(partial) == 1:
                batch.append(partial[0])
                print(f"     Matched: {partial[0]}")
            else:
                print(f"     Multiple matches for '{term}':")
                for i, name in enumerate(partial, 1):
                    print(f"       {i}) {name}")
                pick = input("     Enter numbers (comma-separated) or 'all': ").strip()
                if pick.lower() == 'all':
                    batch.extend(partial)
                else:
                    for p in pick.split(','):
                        p = p.strip()
                        if p.isdigit() and 1 <= int(p) <= len(partial):
                            batch.append(partial[int(p) - 1])

        for name in batch:
            if name not in selected:
                selected.append(name)

        if selected:
            print(f"     Currently selected: {selected}")
            more = input(f"   Add more {entity_label}? (y/n) [n]: ").strip().lower()
            if more != 'y':
                break

    return selected


def _select_admin_boundary(q_level, q_entity, q_subdiv, study_area_polygon=None):
    """Three-part admin boundary TUI: level -> entity -> subdivisions.

    Args:
        q_level:            Question label for admin level, e.g. '1.2'.
        q_entity:           Question label for entity selection, e.g. '1.3'.
        q_subdiv:           Question label for subdivisions, e.g. '1.4'.
        study_area_polygon: If provided, admin entities intersecting the study
                            area are shown as suggestions before entity selection.

    Returns:
        tuple (admin_level: str, primary_names: list[str], subdivision_names: dict)
    """
    print(f"\n[{q_level}]  Administrative level")
    print("   1) National    — All of Switzerland")
    print("   2) Cantonal    — Choose one or more cantons")
    print("   3) Bezirke     — Choose one or more districts")
    print("   4) Municipal   — Choose one or more municipalities")

    while True:
        level_choice = input("\n   Select (1-4) [2]: ").strip() or "2"
        if level_choice in ('1', '2', '3', '4'):
            break
        print("   Invalid selection. Please enter 1, 2, 3, or 4.")

    level_map = {'1': 'national', '2': 'cantonal', '3': 'bezirke', '4': 'municipal'}
    admin_level = level_map[level_choice]

    suggestions = []
    if study_area_polygon is not None and admin_level != 'national':
        suggestions = _compute_overlapping_entities(study_area_polygon, admin_level)

    primary_names = []

    if admin_level == 'national':
        print(f"\n[{q_entity}]  Entity selection — skipped (national level selected)")
    else:
        print(f"\n[{q_entity}]  Entity selection")
        if suggestions:
            entity_type = {'cantonal': 'cantons', 'bezirke': 'Bezirke',
                           'municipal': 'municipalities'}[admin_level]
            print(f"   Suggested {entity_type} (overlap with study area): {suggestions}")

        if admin_level == 'cantonal':
            print("\n   Loading canton names ...")
            names = _load_boundary_names(os.path.join(paths.MAIN, CANTONS_GPKG))
            primary_names = _select_names_interactive(names, 'cantons')
            if not primary_names and suggestions:
                confirm = input(
                    f"\n   No selection made. Use suggested {suggestions}? (y/n) [y]: "
                ).strip().lower() or "y"
                if confirm == 'y':
                    primary_names = suggestions[:]
            if not primary_names:
                raise ValueError("No cantons selected. Aborting.")

        elif admin_level == 'bezirke':
            print("\n   Loading Bezirke names ...")
            names = _load_boundary_names(os.path.join(paths.MAIN, BEZIRKE_GPKG))
            primary_names = _select_names_interactive(names, 'Bezirke')
            if not primary_names and suggestions:
                confirm = input(
                    f"\n   No selection made. Use suggested {suggestions}? (y/n) [y]: "
                ).strip().lower() or "y"
                if confirm == 'y':
                    primary_names = suggestions[:]
            if not primary_names:
                raise ValueError("No Bezirke selected. Aborting.")

        elif admin_level == 'municipal':
            print("\n   Loading municipality names ...")
            names = _load_boundary_names(
                os.path.join(paths.MAIN, MUNICIPALITIES_GPKG),
                objektart_filter='Gemeindegebiet',
            )
            primary_names = _select_names_interactive(names, 'municipalities')
            if not primary_names and suggestions:
                confirm = input(
                    f"\n   No selection made. Use suggested {suggestions}? (y/n) [y]: "
                ).strip().lower() or "y"
                if confirm == 'y':
                    primary_names = suggestions[:]
            if not primary_names:
                raise ValueError("No municipalities selected. Aborting.")

    subdivision_names = {'bezirke': [], 'municipal': []}

    if admin_level == 'national':
        print(f"\n[{q_subdiv}]  Additional subdivisions — skipped (national level selected)")
    elif admin_level == 'municipal':
        print(f"\n[{q_subdiv}]  Additional subdivisions — skipped (no finer level below municipal)")
    else:
        print(f"\n[{q_subdiv}]  Additional subdivisions")
        print("   Union finer-grained areas with the primary boundary.")

        if admin_level == 'cantonal':
            add_bez = input(
                "\n   Add Bezirke outside the selected canton(s)? (y/n) [n]: "
            ).strip().lower()
            if add_bez == 'y':
                print("   Loading Bezirke names ...")
                bez_names = _load_boundary_names(os.path.join(paths.MAIN, BEZIRKE_GPKG))
                subdivision_names['bezirke'] = _select_names_interactive(bez_names, 'Bezirke')

            add_mun = input(
                "\n   Add municipalities outside the selected canton(s)? (y/n) [n]: "
            ).strip().lower()
            if add_mun == 'y':
                print("   Loading municipality names ...")
                mun_names = _load_boundary_names(
                    os.path.join(paths.MAIN, MUNICIPALITIES_GPKG),
                    objektart_filter='Gemeindegebiet',
                )
                subdivision_names['municipal'] = _select_names_interactive(mun_names, 'municipalities')

        elif admin_level == 'bezirke':
            add_mun = input(
                "\n   Add municipalities outside the selected Bezirke? (y/n) [n]: "
            ).strip().lower()
            if add_mun == 'y':
                print("   Loading municipality names ...")
                mun_names = _load_boundary_names(
                    os.path.join(paths.MAIN, MUNICIPALITIES_GPKG),
                    objektart_filter='Gemeindegebiet',
                )
                subdivision_names['municipal'] = _select_names_interactive(mun_names, 'municipalities')

    return admin_level, primary_names, subdivision_names


def _dissolve_admin_polygon(admin_level, primary_names, subdivision_names):
    """Dissolve selected admin units into a single polygon in CODEBASE_CRS.

    Args:
        admin_level:       'national' | 'cantonal' | 'bezirke' | 'municipal'.
        primary_names:     List of entity names to dissolve.
        subdivision_names: Dict with 'bezirke' and 'municipal' lists for extra units.

    Returns:
        Shapely polygon (dissolved, reprojected to EPSG:2056).
    """
    parts = []
    raw_crs = [None]

    def _load_and_filter(gpkg_path, names=None, objektart_filter=None):
        layers = gpd.list_layers(gpkg_path)
        gdf = gpd.read_file(gpkg_path, layer=layers.iloc[0]['name'])
        if raw_crs[0] is None:
            raw_crs[0] = gdf.crs
        if objektart_filter and 'objektart' in gdf.columns:
            gdf = gdf[gdf['objektart'] == objektart_filter]
        if names is not None:
            gdf = gdf[gdf['name'].isin(names)]
        return gdf

    if admin_level == 'national':
        parts.append(_load_and_filter(os.path.join(paths.MAIN, CANTONS_GPKG)))
    elif admin_level == 'cantonal':
        parts.append(_load_and_filter(os.path.join(paths.MAIN, CANTONS_GPKG),
                                      names=primary_names))
    elif admin_level == 'bezirke':
        parts.append(_load_and_filter(os.path.join(paths.MAIN, BEZIRKE_GPKG),
                                      names=primary_names))
    elif admin_level == 'municipal':
        parts.append(_load_and_filter(
            os.path.join(paths.MAIN, MUNICIPALITIES_GPKG),
            names=primary_names,
            objektart_filter='Gemeindegebiet',
        ))
    else:
        raise ValueError(f"Unknown admin level: {admin_level}")

    if parts[0].empty:
        raise ValueError(f"No boundary rows matched for {admin_level} / {primary_names}.")

    sub_bez = subdivision_names.get('bezirke', [])
    if sub_bez:
        bez_gdf = _load_and_filter(os.path.join(paths.MAIN, BEZIRKE_GPKG), names=sub_bez)
        if not bez_gdf.empty:
            parts.append(bez_gdf)
            print(f"   + Unioning {len(bez_gdf)} additional Bezirke")

    sub_mun = subdivision_names.get('municipal', [])
    if sub_mun:
        mun_gdf = _load_and_filter(
            os.path.join(paths.MAIN, MUNICIPALITIES_GPKG),
            names=sub_mun,
            objektart_filter='Gemeindegebiet',
        )
        if not mun_gdf.empty:
            parts.append(mun_gdf)
            print(f"   + Unioning {len(mun_gdf)} additional municipalities")

    combined  = pd.concat(parts, ignore_index=True)
    dissolved = combined.dissolve().to_crs(CODEBASE_CRS)
    return dissolved.geometry.iloc[0]


def _ask_buffer(q_label, default_m, note=None):
    """Prompt for a buffer distance in metres. Returns float."""
    print(f"\n[{q_label}]  Buffer distance (m)")
    if note:
        print(f"   {note}")
    while True:
        buf_str = input(f"   Distance in metres [default: {default_m}]: ").strip() or str(default_m)
        try:
            val = float(buf_str)
            if val < 0:
                print("   Buffer must be >= 0.")
                continue
            return val
        except ValueError:
            print("   Please enter a valid number.")


def _export_gpkg(polygon, name_attr, admin_level, primary_names, buffer_m, output_path):
    """Write a single Shapely polygon as a one-row GeoPackage.

    Args:
        polygon:       Shapely geometry to export.
        name_attr:     Value for the 'name' attribute column.
        admin_level:   Value for the 'admin_level' attribute column.
        primary_names: List of entity names (joined as comma-separated string).
        buffer_m:      Buffer distance in metres (stored as metadata).
        output_path:   Absolute path to the output .gpkg file.
    """
    gdf = gpd.GeoDataFrame(
        {
            'name':        [name_attr],
            'admin_level': [admin_level],
            'primary':     [', '.join(primary_names) if primary_names else 'coordinates'],
            'buffer_m':    [buffer_m],
        },
        geometry=[polygon],
        crs=CODEBASE_CRS,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    gdf.to_file(output_path, driver='GPKG')
    print(f"   Saved: {output_path}")


def _compute_overlapping_entities(study_area_polygon, admin_level):
    """Return names of admin entities at admin_level that intersect the study area."""
    try:
        if admin_level == 'cantonal':
            gpkg_path        = os.path.join(paths.MAIN, CANTONS_GPKG)
            objektart_filter = None
        elif admin_level == 'bezirke':
            gpkg_path        = os.path.join(paths.MAIN, BEZIRKE_GPKG)
            objektart_filter = None
        elif admin_level == 'municipal':
            gpkg_path        = os.path.join(paths.MAIN, MUNICIPALITIES_GPKG)
            objektart_filter = 'Gemeindegebiet'
        else:
            return []

        layers = gpd.list_layers(gpkg_path)
        gdf    = gpd.read_file(gpkg_path, layer=layers.iloc[0]['name']).to_crs(CODEBASE_CRS)
        if objektart_filter and 'objektart' in gdf.columns:
            gdf = gdf[gdf['objektart'] == objektart_filter]

        overlapping = gdf[gdf.geometry.intersects(study_area_polygon)]
        return sorted(overlapping['name'].dropna().tolist())
    except Exception:
        return []


def _validate_containment(study_area_boundary, catchment_boundary):
    """Check that the study area lies fully within the catchment area.

    Args:
        study_area_boundary: Shapely polygon for the study area.
        catchment_boundary:  Shapely polygon for the catchment area.

    Returns:
        True if the study area is fully contained; False otherwise.
    """
    outside     = study_area_boundary.difference(catchment_boundary)
    outside_pct = outside.area / study_area_boundary.area * 100

    if outside_pct < 0.01:
        print("   Pass — study area is fully within the catchment area.")
        return True

    print(f"   Warning — {outside_pct:.1f}% of the study area lies outside the catchment area.")
    return False


def _configure_study_area():
    """Interactive TUI to define the study area polygon and buffer.

    Returns:
        (polygon, buffered_polygon, admin_level, primary_names, buffer_m)
    """
    print("\n" + "─" * 68)
    print("[Phase 1]  Study area")
    print("─" * 68)

    print("\n[1.1]  Definition method")
    print("   1) Polygon coordinates from settings.py   [default]")
    print("   2) Administrative boundaries")

    while True:
        choice = input("\n   Select (1/2) [1]: ").strip() or "1"
        if choice in ('1', '2'):
            break
        print("   Invalid selection. Please enter 1 or 2.")

    if choice == "1":
        polygon       = settings.perimeter_infra_generation
        admin_level   = 'coordinates'
        primary_names = []
        print(f"\n   Using polygon from settings.py.")
        print(f"   Bounds: {polygon.bounds}")
        print("\n   (Questions 1.2 – 1.4 not applicable for coordinate-defined polygon)")
    else:
        admin_level, primary_names, subdivision_names = _select_admin_boundary(
            q_level='1.2', q_entity='1.3', q_subdiv='1.4',
        )
        print("\n   Dissolving selected boundaries ...")
        polygon = _dissolve_admin_polygon(admin_level, primary_names, subdivision_names)
        print(f"   Bounds: {polygon.bounds}")

    buffer_m = _ask_buffer(
        "1.5", STUDY_AREA_DEFAULT_BUFFER_M,
        note="Creates a margin for capturing feeder network elements near the study area edge.",
    )

    return polygon, polygon.buffer(buffer_m), admin_level, primary_names, buffer_m


def _configure_catchment_area(study_area_boundary):
    """Interactive TUI to define the catchment area polygon and buffer.

    Args:
        study_area_boundary: Shapely polygon for the study area (used for suggestions).

    Returns:
        (polygon, buffered_polygon, admin_level, primary_names, buffer_m)
    """
    print("\n" + "─" * 68)
    print("[Phase 2]  Catchment area")
    print("─" * 68)

    admin_level, primary_names, subdivision_names = _select_admin_boundary(
        q_level='2.1', q_entity='2.2', q_subdiv='2.3',
        study_area_polygon=study_area_boundary,
    )

    print("\n   Dissolving selected boundaries ...")
    polygon = _dissolve_admin_polygon(admin_level, primary_names, subdivision_names)
    print(f"   Bounds: {polygon.bounds}")

    buffer_m = _ask_buffer(
        "2.4", CATCHMENT_AREA_DEFAULT_BUFFER_M,
        note="Buffer around the catchment boundary for GTFS spatial filtering.",
    )

    return polygon, polygon.buffer(buffer_m), admin_level, primary_names, buffer_m


def initialise():
    """Interactive standalone entry point — define and export study/catchment areas.

    Prompts the user to configure study area and catchment area interactively and
    exports them as GeoPackages to data/Catchment_Area/.  Run once before the main
    pipeline when area boundaries need to be (re-)defined interactively.
    """
    os.chdir(paths.MAIN)

    print("=" * 68)
    print("infraScanRail — Initialisation")
    print("=" * 68)
    print("Defines the study area and catchment area and exports them as")
    print("GeoPackages for use by downstream pipeline modules.")

    sa_boundary, sa_buffer, sa_admin, sa_names, sa_buf = _configure_study_area()

    sa_boundary_path = os.path.join(paths.MAIN, STUDY_AREA_DIR, 'study_area_boundary.gpkg')
    sa_buffer_path   = os.path.join(paths.MAIN, STUDY_AREA_DIR, 'study_area_buffer.gpkg')

    _export_gpkg(sa_boundary, 'study_area_boundary', sa_admin, sa_names, 0,      sa_boundary_path)
    _export_gpkg(sa_buffer,   'study_area_buffer',   sa_admin, sa_names, sa_buf, sa_buffer_path)

    catchment_valid = False
    while not catchment_valid:
        ca_boundary, ca_buffer, ca_admin, ca_names, ca_buf = _configure_catchment_area(sa_boundary)
        catchment_valid = _validate_containment(sa_boundary, ca_boundary)
        if not catchment_valid:
            retry = input("\n   Re-define the catchment area? (y/n) [y]: ").strip().lower() or "y"
            if retry != 'y':
                print("   Proceeding with current catchment area despite warning.")
                catchment_valid = True

    ca_boundary_path = os.path.join(paths.MAIN, CATCHMENT_AREA_DIR, 'catchment_area_boundary.gpkg')
    ca_buffer_path   = os.path.join(paths.MAIN, CATCHMENT_AREA_DIR, 'catchment_area_buffer.gpkg')

    _export_gpkg(ca_boundary, 'catchment_area_boundary', ca_admin, ca_names, 0,      ca_boundary_path)
    _export_gpkg(ca_buffer,   'catchment_area_buffer',   ca_admin, ca_names, ca_buf, ca_buffer_path)

    print("\n" + "=" * 68)
    print("Summary")
    print("=" * 68)
    print(f"  Study area boundary   : {sa_boundary_path}")
    print(f"  Study area buffer     : {sa_buffer_path}  (+{sa_buf:.0f} m)")
    print(f"  Catchment boundary    : {ca_boundary_path}")
    print(f"  Catchment buffer      : {ca_buffer_path}  (+{ca_buf:.0f} m)")
    print("=" * 68 + "\n")


def setup_versioned_dirs(svc_version: str) -> None:
    """Point all method output dirs under data/Catchment_Area/{svc_version}/.

    Call from get_catchment() or prepare_all_od_matrices() before _ensure_dirs().
    Mutates module globals so every function in this module and in importing
    modules that re-sync their local names will use the versioned paths.
    Step 1 files (population/employment grids, boundary) remain at
    CATCHMENT_DATA_DIR — they are svc-independent.
    """
    global MUNICIPAL_DATA_DIR, PT_FEEDER_DATA_DIR
    global MUNICIPAL_PLOT_DIR, PT_FEEDER_PLOT_DIR
    global GUETEKLASSEN_PLOT_DIR, OD_COMPARISON_PLOT_DIR
    ver_data = os.path.join(CATCHMENT_DATA_DIR, svc_version)
    ver_plot = os.path.join(CATCHMENT_PLOT_DIR, svc_version)
    MUNICIPAL_DATA_DIR     = os.path.join(ver_data, 'Municipal')
    PT_FEEDER_DATA_DIR     = os.path.join(ver_data, 'PT_Feeder')
    MUNICIPAL_PLOT_DIR     = os.path.join(ver_plot, 'Municipal')
    PT_FEEDER_PLOT_DIR     = os.path.join(ver_plot, 'PT_Feeder')
    GUETEKLASSEN_PLOT_DIR  = os.path.join(ver_plot, 'Gueteklassen')
    OD_COMPARISON_PLOT_DIR = os.path.join('plots', 'Traffic_Flow', 'OD',
                                          svc_version, 'Comparison')


def _ensure_dirs():
    """Create all shared catchment output directories."""
    for d in [CATCHMENT_DATA_DIR, CATCHMENT_PLOT_DIR,
              MUNICIPAL_DATA_DIR, PT_FEEDER_DATA_DIR,
              MUNICIPAL_PLOT_DIR, PT_FEEDER_PLOT_DIR,
              GUETEKLASSEN_PLOT_DIR, POP_EMPL_DATA_DIR, POP_EMPL_PLOT_DIR]:
        os.makedirs(d, exist_ok=True)


# ===============================================================================
# MAP CARTOGRAPHIC ELEMENTS (North arrow & scale bar)
# ===============================================================================

_SCALE_BAR_NICE_KM = [1, 2, 5, 10, 20, 50, 100, 200, 500]


def _add_north_arrow(ax, location='upper left', scale=0.35):
    """Draw a north arrow using matplotlib_map_utils."""
    north_arrow(ax, location=location, scale=scale, rotation={"degrees": 0})


def _add_scale_bar(ax, location=(0.755, 0.012)):
    """Adaptive scale bar with alternating black/white cells in data coordinates."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    map_w = xlim[1] - xlim[0]
    map_h = ylim[1] - ylim[0]

    target_km = (map_w / 4.0) / 1000.0
    total_km  = min(_SCALE_BAR_NICE_KM, key=lambda v: abs(v - target_km))
    n_cells   = 4 if total_km >= 4 else 2
    cell_m    = (total_km * 1000.0) / n_cells

    x0    = xlim[0] + map_w * location[0]
    y0    = ylim[0] + map_h * location[1]
    bar_h = map_h * 0.008

    for i in range(n_cells):
        color = 'black' if i % 2 == 0 else 'white'
        rect = Rectangle(
            (x0 + i * cell_m, y0), cell_m, bar_h,
            facecolor=color, edgecolor='black', linewidth=0.6, zorder=7,
        )
        ax.add_patch(rect)

    for i in range(n_cells + 1):
        val_km = (i * cell_m) / 1000.0
        label  = (f'{val_km:.0f} km' if val_km == int(val_km)
                  else f'{val_km:.1f} km')
        ax.text(
            x0 + i * cell_m, y0 + bar_h * 1.6,
            label, ha='center', va='bottom', fontsize=5, zorder=7,
        )


def _add_map_elements(ax):
    """Add north arrow (upper left) and adaptive scale bar (lower right) to an axes."""
    _add_north_arrow(ax)
    _add_scale_bar(ax)


# ===============================================================================
# STEP 1: SHARED DATA LOADING (both methods)
# ===============================================================================

def _load_catchment_boundary():
    """Load the catchment area boundary polygon from the GPKG produced by
    services_filter_gtfs.py. Single source of truth for the spatial extent.

    Returns
    -------
    shapely.Polygon
        Study area boundary in EPSG:2056.
    """
    boundary_path = paths.CATCHMENT_AREA_BOUNDARY_GPKG
    print(f"  Loading catchment boundary from {boundary_path} ...")
    gdf = gpd.read_file(boundary_path)
    gdf = gdf.to_crs(CODEBASE_CRS)
    boundary = gdf.geometry.unary_union
    print(f"  Boundary loaded - area = {boundary.area / 1e6:.1f} km2")
    return boundary


def _load_population_grid(boundary, year: int = None):
    """Load the Swiss population CSV, filter to study area, scale to year."""
    print("  Loading population grid ...")
    df = pd.read_csv(paths.POPULATION_CSV_2023, sep=';')
    return _load_grid(df, boundary, 'population', year)


def _load_employment_grid(boundary, year: int = None):
    """Load the Swiss employment CSV, filter to study area, scale to year."""
    print("  Loading employment grid ...")
    df = pd.read_csv(paths.EMPLOYMENT_CSV_2023, sep=';')
    return _load_grid(df, boundary, 'employment', year)


def _load_grid(df, boundary, label, year: int = None):
    """Shared loader for population / employment CSVs.

    Loads the 2023 federal grid, filters to the catchment boundary, then
    scales cell values to `year` using commune-level cantonal data.  When
    year == 2023 (or None with POPULATION_BASE_YEAR == 2023) no scaling
    is applied.  Saves the result as data/Catchment_Area/{label}_{year}_{canton}.csv.

    Cells are included if their centroid is strictly inside the boundary OR
    if at least 50 % of the 100×100 m cell area intersects the boundary
    (edge-cell rule).  A fast centroid pre-filter limits the expensive area
    check to cells within one cell-width of the boundary.
    """
    if year is None:
        year = settings.POPULATION_BASE_YEAR
    # Swiss CSVs may use comma as decimal separator (e.g. "5,376" → 5.376).
    for col in ['E_KOORD', 'N_KOORD', 'NUMMER']:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '.', regex=False),
            errors='coerce',
        )
    if 'CLASS' in df.columns:
        df['CLASS'] = pd.to_numeric(df['CLASS'], errors='coerce')
    df = df.dropna(subset=['E_KOORD', 'N_KOORD', 'NUMMER'])
    df = df[df['NUMMER'] > 0].copy()

    # Compute cell centroids (bottom-left corner + 50 m)
    df['cx'] = df['E_KOORD'] + 50
    df['cy'] = df['N_KOORD'] + 50
    geometry = gpd.points_from_xy(df['cx'], df['cy'])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=CODEBASE_CRS)

    # Step 1: fast pre-filter — drop cells more than one cell-width outside
    prep_expanded = prep(boundary.buffer(CELL_SIZE_M))
    gdf = gdf[gdf.geometry.apply(prep_expanded.contains)].copy()

    # Step 2: centroid strictly inside → keep unconditionally
    prep_boundary = prep(boundary)
    inside_mask = gdf.geometry.apply(prep_boundary.contains)
    inside = gdf[inside_mask]

    # Step 3: edge cells — keep if ≥50 % of cell area intersects boundary
    edge_gdf = gdf[~inside_mask]
    cell_area = CELL_SIZE_M * CELL_SIZE_M
    if len(edge_gdf) > 0:
        keep_idx = [
            idx for idx, row in edge_gdf.iterrows()
            if box(row['E_KOORD'], row['N_KOORD'],
                   row['E_KOORD'] + CELL_SIZE_M,
                   row['N_KOORD'] + CELL_SIZE_M
                   ).intersection(boundary).area / cell_area >= 0.5
        ]
        gdf = pd.concat([inside, edge_gdf.loc[keep_idx]]).copy()
    else:
        gdf = inside.copy()

    print(f"    {label}: {len(gdf):,} cells within study area (>=50 % overlap)")

    # Scale 2023 cell values to target year when year != 2023.
    # Cell-level values stay float; integer rounding is reserved for
    # aggregate boundaries (commune totals, per-(commune, station) splits).
    if year != 2023:
        gdf = _scale_cells_to_year(gdf, year, label, boundary)

    # Save (include CLASS if present)
    save_cols = ['RELI', 'E_KOORD', 'N_KOORD', 'NUMMER']
    if 'CLASS' in gdf.columns:
        save_cols.append('CLASS')
    csv_path = _cache_csv_path(label, year)
    gdf[save_cols].to_csv(csv_path, index=False, sep=';')
    print(f"    Saved -> {csv_path}")

    return gdf


def _cumulate_per_municipality(pop_grid, empl_grid, boundary, year: int = None):
    """Spatial join of population/employment cells to municipalities, with
    nearest-municipality fallback for edge cells.

    Args:
        pop_grid:  Population grid GeoDataFrame (already scaled to year).
        empl_grid: Employment grid GeoDataFrame (already scaled to year).
        boundary:  Catchment boundary polygon.
        year:      Target year for the summary CSV filename; defaults to
                   settings.POPULATION_BASE_YEAR.

    Returns
    -------
    gpd.GeoDataFrame
        Summary with columns [BFS_NR, NAME, total_population, total_employment, geometry].
    """
    if year is None:
        year = settings.POPULATION_BASE_YEAR
    print("  Cumulating population/employment per municipality ...")

    muni = gpd.read_file(paths.MUNICIPAL_BOUNDARIES_GPKG)
    muni = muni.to_crs(CODEBASE_CRS)
    if 'objektart' in muni.columns:
        muni = muni[muni['objektart'] == 'Gemeindegebiet']
    muni = muni[muni.geometry.intersects(boundary)].copy()
    print(f"    {len(muni)} municipalities in study area")

    # Identify BFS number column
    bfs_col = None
    for candidate in ['BFS_NR', 'bfs_nr', 'BFS_NUMMER', 'bfs_nummer', 'GMDNR', 'gmdnr']:
        if candidate in muni.columns:
            bfs_col = candidate
            break
    if bfs_col is None:
        for c in muni.columns:
            if muni[c].dtype in ['int64', 'int32', 'float64'] and c != 'geometry':
                bfs_col = c
                break
    if bfs_col is None:
        raise ValueError("Cannot identify BFS number column in municipal boundaries")

    # Identify name column
    name_col = None
    for candidate in ['NAME', 'name', 'GMDNAME', 'gmdname', 'GEMEINDENAME']:
        if candidate in muni.columns:
            name_col = candidate
            break
    if name_col is None:
        name_col = bfs_col

    def _assign_cells_to_muni(grid, value_col='NUMMER'):
        joined = gpd.sjoin(grid, muni[[bfs_col, name_col, 'geometry']],
                           how='left', predicate='within')
        unassigned = joined[joined[bfs_col].isna()]
        if len(unassigned) > 0:
            print(f"    {len(unassigned)} cells outside all municipalities - assigning to nearest")
            for idx in unassigned.index:
                pt = grid.loc[idx, 'geometry']
                dists = muni.geometry.boundary.distance(pt)
                nearest_idx = dists.idxmin()
                joined.loc[idx, bfs_col] = muni.loc[nearest_idx, bfs_col]
                joined.loc[idx, name_col] = muni.loc[nearest_idx, name_col]
        agg = joined.groupby(bfs_col).agg({value_col: 'sum', name_col: 'first'}).reset_index()
        return agg

    pop_agg  = _assign_cells_to_muni(pop_grid)
    empl_agg = _assign_cells_to_muni(empl_grid)

    summary = pop_agg.rename(columns={'NUMMER': 'total_population'})
    empl_renamed = empl_agg.rename(columns={'NUMMER': 'total_employment'})
    summary = summary.merge(empl_renamed[[bfs_col, 'total_employment']],
                            on=bfs_col, how='outer')
    summary = summary.fillna(0)

    # Enforce integer types on per-municipality totals (cells already integer;
    # this is defensive against any floating-point summation drift)
    summary['total_population'] = summary['total_population'].round().astype(int)
    summary['total_employment'] = summary['total_employment'].round().astype(int)

    # Recover name for outer-join rows that only have employment
    missing_name = summary[name_col].isna() | (summary[name_col] == 0)
    if missing_name.any():
        name_lookup = muni.set_index(bfs_col)[name_col]
        summary.loc[missing_name, name_col] = (
            summary.loc[missing_name, bfs_col].map(name_lookup)
        )

    # Rename canonical columns before geometry merge so everything is consistent
    summary = summary.rename(columns={bfs_col: 'BFS_NR', name_col: 'NAME'})

    muni_geom = muni[[bfs_col, 'geometry']].rename(columns={bfs_col: 'BFS_NR'})
    summary = summary.merge(muni_geom, on='BFS_NR', how='left')
    summary = gpd.GeoDataFrame(summary, geometry='geometry', crs=CODEBASE_CRS)

    # Enforce column order: BFS_NR, NAME, total_population, total_employment
    csv_cols = ['BFS_NR', 'NAME', 'total_population', 'total_employment']
    csv_path = os.path.join(POP_EMPL_DATA_DIR, f'municipal_pop_empl_summary_{year}.csv')
    summary[csv_cols].to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"    Saved -> {csv_path}")

    # Also write an unsuffixed copy so legacy consumers
    # (e.g. catchment_allocate.py searching for 'municipal_pop_empl_summary.csv')
    # continue to work without a year-tag-aware path.
    legacy_path = os.path.join(POP_EMPL_DATA_DIR, 'municipal_pop_empl_summary.csv')
    summary[csv_cols].to_csv(legacy_path, index=False, encoding='utf-8-sig')
    print(f"    Saved -> {legacy_path}  (unsuffixed legacy alias)")
    print(f"    Total pop = {summary['total_population'].sum():,d}, "
          f"empl = {summary['total_employment'].sum():,d}")

    return summary


def _plot_municipal_distributions(summary_df, boundary, year: int = None):
    """Produce choropleth maps for population and employment per municipality.

    Uses the same discrete FSO class breaks as the raster plots so that the
    municipal and raster maps share a common visual language.
    Saved to plots/Catchment_Area/.
    For MultiPolygon boundaries a separate plot is produced per component.
    """
    if year is None:
        year = settings.POPULATION_BASE_YEAR
    print("  Plotting municipal distributions ...")

    pop_cfg = dict(
        bins   = [0, 2000, 5000, 10000, 20000, 50000, np.inf],
        labels = ['1–2,000', '2,001–5,000', '5,001–10,000',
                  '10,001–20,000', '20,001–50,000', 'more than 50,000'],
        colors = ['#FFFFB2', '#FECC5C', '#FD8D3C', '#F03B20', '#BD0026', '#800026'],
        col_header = 'Total inhabitants',
    )
    empl_cfg = dict(
        bins   = [0, 500, 1500, 5000, 20000, np.inf],
        labels = ['1–500', '501–1,500', '1,501–5,000',
                  '5,001–20,000', 'more than 20,000'],
        colors = ['#C7E9C0', '#74C476', '#238B45', '#2C7FB8', '#253494'],
        col_header = 'Total full-time equivalents',
    )

    # Load lakes once for all plots in this function
    lakes = None
    if os.path.exists(paths.LAKES_SHP):
        lakes = gpd.read_file(paths.LAKES_SHP).to_crs(CODEBASE_CRS)
        lakes = lakes[lakes.geometry.intersects(boundary)].copy()

    components = (list(boundary.geoms)
                  if boundary.geom_type == 'MultiPolygon' else [boundary])

    for col, title_word, cfg in [
        ('total_population', 'Population', pop_cfg),
        ('total_employment', 'Employment', empl_cfg),
    ]:
        bins   = cfg['bins']
        labels = cfg['labels']
        colors = cfg['colors']

        for part_idx, component in enumerate(components):
            part_suffix = f'_part{part_idx + 1}' if len(components) > 1 else ''

            clipped_summary = gpd.clip(summary_df, component)
            vals      = clipped_summary[col].fillna(0).values
            zero_mask = vals <= 0
            # fillna(0) before astype(int) handles the NaN pd.cut returns
            # for exact-zero values (which fall below the open lower bin edge)
            class_idx = (pd.Series(pd.cut(vals, bins=bins, labels=False, right=True))
                           .clip(0, len(labels) - 1)
                           .fillna(0)
                           .astype(int)
                           .values)

            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            ax.set_facecolor('#E8E8E8')   # grey outside the catchment

            # White fill for catchment interior (zero-value munis show as white)
            comp_gdf = gpd.GeoDataFrame(geometry=[component], crs=CODEBASE_CRS)
            comp_gdf.plot(ax=ax, color='white', edgecolor='none', zorder=0)

            for ci, color in enumerate(colors):
                mask = (class_idx == ci) & (~zero_mask)
                if not mask.any():
                    continue
                clipped_summary[mask].plot(ax=ax, color=color,
                                           edgecolor='grey', linewidth=0.4,
                                           zorder=1)

            # Lakes above fill, below boundary line
            if lakes is not None and not lakes.empty:
                lakes_clip = gpd.clip(lakes, component)
                if not lakes_clip.empty:
                    lakes_clip.plot(ax=ax, color='#A8D8EA', edgecolor='none',
                                    zorder=4)

            # Catchment area boundary
            comp_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8,
                                   linestyle='--', zorder=5)

            bx_min, by_min, bx_max, by_max = component.bounds
            pad = 200
            ax.set_xlim(bx_min - pad, bx_max + pad)
            ax.set_ylim(by_min - pad, by_max + pad)

            # Add cartographic elements
            _add_map_elements(ax)

            legend_handles = []
            for color, lbl in zip(colors, labels):
                legend_handles.append(
                    Patch(facecolor=color, edgecolor='none', label=lbl))
            legend_handles.append(
                Line2D([0], [0], color='black', linewidth=1.8,
                       linestyle='--', label='Catchment area boundary'))

            ax.legend(handles=legend_handles,
                      title=cfg['col_header'],
                      title_fontsize=8,
                      fontsize=7,
                      loc='upper right',
                      framealpha=0.9)

            ax.set_title(f'{title_word} per Municipality', fontsize=14)
            ax.set_xlabel('E [m]')
            ax.set_ylabel('N [m]')
            ax.set_aspect('equal')

            fname = f'plot_{title_word.lower()}_by_municipality_{year}{part_suffix}.pdf'
            out_path = os.path.join(POP_EMPL_PLOT_DIR, fname)
            fig.savefig(out_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            print(f"    Saved -> {out_path}")


def _plot_raster_map(gdf, label, boundary, year: int = None):
    """Render a population or employment grid as a coloured raster map,
    styled to match the corresponding map.geo.admin FSO layer.

    Population  → FSO "Population Statistics: Inhabitants"
                  6 discrete classes, yellow→red palette
    Employment  → FSO "Enterprise Statistics: Employment (FTE)"
                  5 discrete classes, yellow→green→blue palette

    Cells are drawn as 100 m × 100 m square patches.
    Grey axes background outside the catchment; white interior for zero cells.
    Saved to plots/Catchment_Area/.
    For MultiPolygon boundaries a separate plot is produced per component.
    """
    if year is None:
        year = settings.POPULATION_BASE_YEAR
    print(f"  Plotting {label} raster map ...")

    is_pop = (label == 'population')

    if is_pop:
        bins   = [0, 3, 6, 15, 40, 120, np.inf]
        labels = ['1–3', '4–6', '7–15', '16–40', '41–120', 'more than 120']
        colors = ['#FFFFB2', '#FECC5C', '#FD8D3C', '#F03B20', '#BD0026', '#800026']
        col_header = 'Inhabitants per ha'
        title      = 'Population'
    else:
        bins   = [0, 40, 75, 150, 300, np.inf]
        labels = ['0.1–40', '40.1–75', '75.1–150', '150.1–300', 'more than 300']
        colors = ['#C7E9C0', '#74C476', '#238B45', '#2C7FB8', '#253494']
        col_header = 'Full-time equivalents per ha'
        title      = 'Employment (FTE)'

    n_classes = len(labels)

    vals = gdf['NUMMER'].values
    class_idx = np.digitize(vals, bins[1:], right=False)
    class_idx = np.clip(class_idx, 0, n_classes - 1)

    e_arr = gdf['E_KOORD'].values
    n_arr = gdf['N_KOORD'].values
    squares = [box(e, n, e + CELL_SIZE_M, n + CELL_SIZE_M)
               for e, n in zip(e_arr, n_arr)]
    plot_gdf = gpd.GeoDataFrame(
        {'class_idx': class_idx},
        geometry=squares,
        crs=CODEBASE_CRS,
    )

    # Load municipal boundaries once (pre-filter to full boundary extent)
    muni = gpd.read_file(paths.MUNICIPAL_BOUNDARIES_GPKG).to_crs(CODEBASE_CRS)
    if 'objektart' in muni.columns:
        muni = muni[muni['objektart'] == 'Gemeindegebiet']
    muni = muni[muni.geometry.intersects(boundary)].copy()

    # Load lakes once
    lakes = None
    if os.path.exists(paths.LAKES_SHP):
        lakes = gpd.read_file(paths.LAKES_SHP).to_crs(CODEBASE_CRS)
        lakes = lakes[lakes.geometry.intersects(boundary)].copy()

    components = (list(boundary.geoms)
                  if boundary.geom_type == 'MultiPolygon' else [boundary])

    for part_idx, component in enumerate(components):
        part_suffix = f'_part{part_idx + 1}' if len(components) > 1 else ''

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.set_facecolor('#E8E8E8')   # grey outside the catchment

        # White fill for catchment interior (zero-value cells show as white)
        comp_gdf = gpd.GeoDataFrame(geometry=[component], crs=CODEBASE_CRS)
        comp_gdf.plot(ax=ax, color='white', edgecolor='none', zorder=0)

        # Clip and plot cell squares for this component
        plot_comp = gpd.clip(plot_gdf, component)
        for ci, (color, lbl) in enumerate(zip(colors, labels)):
            mask = plot_comp['class_idx'] == ci
            if not mask.any():
                continue
            plot_comp[mask].plot(ax=ax, color=color, edgecolor='none',
                                 linewidth=0, zorder=1)

        # Municipal boundaries (clipped to component)
        muni_comp  = gpd.clip(muni, component)
        muni_lines = muni_comp.boundary.explode(index_parts=False)
        muni_lines = muni_lines[
            ~muni_lines.geom_type.isin(['Point', 'MultiPoint'])]
        muni_lines.plot(ax=ax, color='#404040', linewidth=0.3, zorder=3)

        # Lakes above municipal lines, below catchment boundary
        if lakes is not None and not lakes.empty:
            lakes_clip = gpd.clip(lakes, component)
            if not lakes_clip.empty:
                lakes_clip.plot(ax=ax, color='#A8D8EA', edgecolor='none',
                                zorder=4)

        # Catchment boundary
        comp_gdf.boundary.plot(ax=ax, color='black', linewidth=1.8,
                               linestyle='--', zorder=5)

        legend_handles = []
        for color, lbl in zip(colors, labels):
            legend_handles.append(
                Patch(facecolor=color, edgecolor='none', label=lbl))
        legend_handles.append(
            Line2D([0], [0], color='#404040', linewidth=0.3,
                   label='Municipal boundary'))
        legend_handles.append(
            Line2D([0], [0], color='black', linewidth=1.8,
                   linestyle='--', label='Catchment area boundary'))

        leg = ax.legend(handles=legend_handles,
                        title=col_header,
                        title_fontsize=8,
                        fontsize=7,
                        loc='upper right',
                        framealpha=0.9,
                        ncol=1)
        leg._legend_box.align = 'left'

        bx_min, by_min, bx_max, by_max = component.bounds
        pad = 200
        ax.set_xlim(bx_min - pad, bx_max + pad)
        ax.set_ylim(by_min - pad, by_max + pad)

        # Add cartographic elements
        _add_map_elements(ax)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel('E [m]')
        ax.set_ylabel('N [m]')
        ax.set_aspect('equal')

        out_path = os.path.join(POP_EMPL_PLOT_DIR,
                                f'plot_{label}_raster_{year}{part_suffix}.pdf')
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    print(f"    Saved -> {out_path}")


# ===============================================================================
# YEAR SCALING — adjusts 2023 federal raster cells to any target year
# ===============================================================================

def _scale_cells_to_year(gdf: gpd.GeoDataFrame, year: int, label: str,
                          boundary) -> gpd.GeoDataFrame:
    """Scale 2023 federal raster cells to match Bezirk-level totals for target year.

    For each Bezirk: scale_factor = sum(commune_totals(year)) / sum(commune_totals(2023)).
    Every cell within municipalities of that Bezirk receives the same factor.
    Communes outside Canton ZH (no Bezirk mapping) default to factor = 1.0.

    Args:
        gdf:      GeoDataFrame of 2023 cells (NUMMER column holds values).
        year:     Target year.
        label:    'population' or 'employment'.
        boundary: Catchment boundary polygon used to pre-filter municipalities.

    Returns:
        GeoDataFrame with NUMMER column scaled to the target year.
    """
    global _SCALE_REPORT_LINES
    print(f"    Scaling {label} cells 2023 -> {year} by Bezirk ...")

    # Commune-to-Bezirk mapping (Canton ZH)
    mapping_df = _load_pop_xlsx_gebietszuordnung()
    commune_to_bezirk = mapping_df.set_index('bfs_nr')['bezirk']

    # Commune-level totals for year and 2023 reference
    if label == 'population':
        target = load_commune_pop(year)
        base23  = load_commune_pop(2023)
    else:
        target = load_commune_empl(year)
        base23  = load_commune_empl(2023)

    # Aggregate commune totals to Bezirk level
    all_bfs = commune_to_bezirk.index
    t_bfs   = target.reindex(all_bfs, fill_value=0.0)
    b_bfs   = base23.reindex(all_bfs, fill_value=0.0)
    t_bezirk = t_bfs.groupby(commune_to_bezirk).sum()
    b_bezirk = b_bfs.groupby(commune_to_bezirk).sum()
    bezirk_scale = (t_bezirk / b_bezirk.where(b_bezirk > 0, np.nan)).fillna(1.0)

    # Each commune inherits its Bezirk's factor; float index to match spatial join output
    commune_scale = commune_to_bezirk.map(bezirk_scale).fillna(1.0)
    commune_scale.index = commune_scale.index.astype(float)

    # Spatial join: assign each cell to its commune
    muni = gpd.read_file(paths.MUNICIPAL_BOUNDARIES_GPKG).to_crs(CODEBASE_CRS)
    if 'objektart' in muni.columns:
        muni = muni[muni['objektart'] == 'Gemeindegebiet']
    muni = muni[muni.geometry.intersects(boundary)].copy()

    bfs_col = next(
        (c for c in ['bfs_nummer', 'BFS_NR', 'bfs_nr', 'BFS_NUMMER', 'GMDNR', 'gmdnr']
         if c in muni.columns), None
    )
    if bfs_col is None:
        print("    WARNING: no BFS column found -- Bezirk scaling skipped")
        return gdf

    muni[bfs_col] = pd.to_numeric(muni[bfs_col], errors='coerce').astype('Int64')
    muni = muni.dropna(subset=[bfs_col])

    joined = gpd.sjoin(
        gdf[['E_KOORD', 'N_KOORD', 'NUMMER', 'geometry']].copy(),
        muni[[bfs_col, 'geometry']],
        how='left', predicate='within'
    )
    joined = joined[~joined.index.duplicated(keep='first')].copy()
    joined[bfs_col] = pd.to_numeric(joined[bfs_col], errors='coerce')
    cell_scales = joined[bfs_col].map(commune_scale).fillna(1.0)

    # Apply per-cell Bezirk scaling. Cells are kept as floats — integer
    # rounding happens only at aggregation boundaries (commune totals in
    # `_cumulate_per_municipality`; per-(commune, station) splits in
    # catchment_allocate's Phase 4 Hamilton step).
    result = gdf.copy()
    result['NUMMER'] = (result['NUMMER'].values * cell_scales.values).clip(min=0.0).round(2)

    # Terminal report
    mean_scale  = float(bezirk_scale.mean())
    total_2023  = float(gdf['NUMMER'].sum())
    total_year  = float(result['NUMMER'].sum())
    print(f"    {label.capitalize()} Bezirk scale factors (2023 -> {year}):")
    for bz in sorted(bezirk_scale.index):
        print(f"      {bz}: {bezirk_scale[bz]:.4f}")
    print(f"      Mean: {mean_scale:.4f}")
    print(f"    Total {label}: {total_2023:,.0f} (2023) -> {total_year:,.0f} ({year})")

    # Store for combined table (built by main() after both labels are done)
    _BEZIRK_SCALES[label] = bezirk_scale
    _TOTALS[label] = {'base': total_2023, 'target': total_year}

    return result


# ===============================================================================
# CACHE READERS — for downstream modules that consume Step 1 outputs
# ===============================================================================
#
# Step 1 (boundary, filtered grids, per-municipality summary, plots) is run by
# `python catchment_base.py`. Downstream modules (catchment_allocate.py and
# catchment_OD_preparation.py) read the cached outputs via these helpers and
# do NOT re-run Step 1 themselves.
#
# Stale-cache caveat: the cache files are tagged by canton (settings.CATCHMENT_CANTON),
# not by boundary geometry. If you change the catchment boundary, change the
# raw POPULATION_CSV_2023 / EMPLOYMENT_CSV_2023 source data, or change which
# 50%-overlap rule applies to edge cells, you must rerun catchment_base.py
# manually to refresh the cache.

def _cache_csv_path(label: str, year: int = None) -> str:
    """Path of the boundary-filtered, year-scaled grid CSV.

    Args:
        label: 'population' or 'employment'
        year:  Target year; defaults to settings.POPULATION_BASE_YEAR.
    """
    if year is None:
        year = settings.POPULATION_BASE_YEAR
    area_tag = '_'.join(settings.CATCHMENT_AREA_ADMIN_NAMES).replace(' ', '').replace('(', '').replace(')', '')
    return os.path.join(POP_EMPL_DATA_DIR, f'{label}_{year}_{area_tag}.csv')


def _missing_cache_msg(missing_path: str) -> str:
    return (
        f"Cached Step 1 file not found:\n  {missing_path}\n"
        f"Run `python catchment_base.py` from infraScanRail/ first to "
        f"generate the boundary-filtered grids and per-municipality summary."
    )


def _read_grid_csv(csv_path: str, label: str) -> gpd.GeoDataFrame:
    """Read a cached boundary-filtered grid CSV, rebuilding cell-centroid geometry."""
    df = pd.read_csv(csv_path, sep=';')
    df['cx'] = df['E_KOORD'] + 50
    df['cy'] = df['N_KOORD'] + 50
    geom = gpd.points_from_xy(df['cx'], df['cy'])
    gdf = gpd.GeoDataFrame(df, geometry=geom, crs=CODEBASE_CRS)
    print(f"  {label}: {len(gdf):,} cells loaded from cache ({csv_path})")
    return gdf


def load_population_grid_cached() -> gpd.GeoDataFrame:
    """Read the boundary-filtered, year-scaled population grid from the cache.

    Cache file: data/Catchment_Area/population_{POPULATION_BASE_YEAR}_<canton>.csv
    Written by catchment_base.main(). Raises FileNotFoundError if missing.
    """
    csv_path = _cache_csv_path('population')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(_missing_cache_msg(csv_path))
    return _read_grid_csv(csv_path, 'population')


def load_employment_grid_cached() -> gpd.GeoDataFrame:
    """Read the boundary-filtered, year-scaled employment grid from the cache.

    Cache file: data/Catchment_Area/employment_{POPULATION_BASE_YEAR}_<canton>.csv
    Written by catchment_base.main(). Raises FileNotFoundError if missing.
    """
    csv_path = _cache_csv_path('employment')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(_missing_cache_msg(csv_path))
    return _read_grid_csv(csv_path, 'employment')


def load_summary_cached() -> gpd.GeoDataFrame:
    """Read the per-municipality summary from the cache (year = POPULATION_BASE_YEAR).

    Cache file: data/Catchment_Area/municipal_pop_empl_summary_{year}.csv
    Returns: GeoDataFrame with [BFS_NR, NAME, total_population, total_employment, geometry].
    """
    year = settings.POPULATION_BASE_YEAR
    csv_path = os.path.join(POP_EMPL_DATA_DIR, f'municipal_pop_empl_summary_{year}.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(_missing_cache_msg(csv_path))

    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # Re-attach geometry from the municipal boundary GPKG
    muni = gpd.read_file(paths.MUNICIPAL_BOUNDARIES_GPKG).to_crs(CODEBASE_CRS)
    if 'objektart' in muni.columns:
        muni = muni[muni['objektart'] == 'Gemeindegebiet']
    bfs_col = None
    for c in ['BFS_NR', 'bfs_nr', 'BFS_NUMMER', 'bfs_nummer', 'GMDNR', 'gmdnr']:
        if c in muni.columns:
            bfs_col = c
            break
    if bfs_col is None:
        raise ValueError(
            f"Cannot identify BFS column in {paths.MUNICIPAL_BOUNDARIES_GPKG} "
            f"to re-attach geometry to the cached summary.")
    muni_geom = muni[[bfs_col, 'geometry']].rename(columns={bfs_col: 'BFS_NR'})
    df['BFS_NR'] = pd.to_numeric(df['BFS_NR'], errors='coerce')
    muni_geom['BFS_NR'] = pd.to_numeric(muni_geom['BFS_NR'], errors='coerce')
    summary = df.merge(muni_geom, on='BFS_NR', how='left')
    summary = gpd.GeoDataFrame(summary, geometry='geometry', crs=CODEBASE_CRS)
    print(f"  municipal summary: {len(summary):,} rows loaded from cache "
          f"({csv_path})")
    return summary


# ===============================================================================
# COMMUNE-LEVEL POPULATION & EMPLOYMENT LOADERS
# ===============================================================================
#
# These functions provide commune-level time series for any year, sourced from:
#   1962–2025 (pop) / 2011–2023 (empl) : actual cantonal data
#   2026–2050 (pop)                     : bezirk projections disaggregated via 2025 shares
#   2051–2100 (pop)                     : Eurostat extension with per-bezirk scaling
#   2024+     (empl)                    : Option A — scale 2023 FTE by pop(year)/pop(2023)
#
# All functions return a Series indexed by integer BFS-NR.
# Callers must have set os.chdir(paths.MAIN) or use absolute paths.


def _load_pop_xlsx_gemeinden() -> pd.DataFrame:
    """Read the Gemeinden sheet from POPULATION_CANTON_ZH_XLSX.

    Returns DataFrame with stripped column names, integer BFS-NR, and
    TOTAL_<year> columns for all years 1962-2025.
    """
    df = pd.read_excel(paths.POPULATION_CANTON_ZH_XLSX, sheet_name='Gemeinden', header=5)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[pd.to_numeric(df['BFS-NR'], errors='coerce').notna()].copy()
    df['BFS-NR'] = pd.to_numeric(df['BFS-NR'], errors='coerce').astype(int)
    return df


def _load_pop_xlsx_gebietszuordnung() -> pd.DataFrame:
    """Read the Gebietszuordnung sheet from POPULATION_CANTON_ZH_XLSX.

    Returns DataFrame with columns: bfs_nr (int), bez_nr (int), bezirk (str).
    Used to map communes to bezirke for projection disaggregation.
    """
    df = pd.read_excel(
        paths.POPULATION_CANTON_ZH_XLSX, sheet_name='Gebietszuordnung', header=4
    )
    df.columns = [str(c).strip() for c in df.columns]
    df = df[pd.to_numeric(df['BFS-NR'], errors='coerce').notna()].copy()
    df['BFS-NR'] = pd.to_numeric(df['BFS-NR'], errors='coerce').astype(int)
    df['BEZ-NR'] = pd.to_numeric(df['BEZ-NR'], errors='coerce').astype('Int64')
    return df[['BFS-NR', 'BEZ-NR', 'BEZIRK']].rename(
        columns={'BFS-NR': 'bfs_nr', 'BEZ-NR': 'bez_nr', 'BEZIRK': 'bezirk'}
    )


def _commune_pop_from_bezirk_projection(year: int) -> pd.Series:
    """Disaggregate bezirk-level projection for year to commune level.

    Method: commune share = commune_pop_2025 / bezirk_total_pop_2025.
    Applied to the 'Prognose_Trend_ZH_2022' bezirk total for the target year.
    """
    mapping = _load_pop_xlsx_gebietszuordnung()
    pop_2025 = load_commune_pop(2025)

    df = mapping.merge(
        pop_2025.rename('pop_2025').reset_index().rename(columns={pop_2025.index.name or 'BFS-NR': 'bfs_nr'}),
        on='bfs_nr', how='left'
    )
    df['pop_2025'] = df['pop_2025'].fillna(0.0)

    bezirk_total_2025 = df.groupby('bezirk')['pop_2025'].sum()
    df['share'] = df.apply(
        lambda r: r['pop_2025'] / bezirk_total_2025[r['bezirk']]
        if bezirk_total_2025[r['bezirk']] > 0 else 0.0,
        axis=1
    )

    bz = pd.read_csv(paths.POPULATION_SCENARIO_CANTON_ZH_2050, sep=';')
    proj = bz[(bz['daten'] == 'Prognose_Trend_ZH_2022') & (bz['jahr'] == year)]
    bezirk_target = proj.groupby('bezirk')['anzahl'].sum()

    df['pop_target'] = df.apply(
        lambda r: r['share'] * bezirk_target.get(r['bezirk'], 0.0), axis=1
    )
    s = df.set_index('bfs_nr')['pop_target']
    s.index = s.index.astype(int)
    return s.rename(year)


def _commune_pop_eurostat_extended(year: int) -> pd.Series:
    """Grow 2050 commune populations to year using Eurostat Swiss rates + bezirk scaling.

    Mirrors the logic in random_scenarios.get_bezirk_population_scenarios for
    the 2051-2100 extension, applied at commune level using 2050 commune shares.
    """
    pop_2050 = load_commune_pop(2050)

    eu = pd.read_excel(paths.POPULATION_SCENARIO_CH_EUROSTAT_2100)
    eu.columns = eu.columns.map(str)
    gr_row = eu[eu['unit'] == 'GROWTH_RATE'].iloc[0]

    bz = pd.read_csv(paths.POPULATION_SCENARIO_CANTON_ZH_2050, sep=';')
    df_ch = pd.read_csv(paths.POPULATION_SCENARIO_CH_BFS_2055, sep=',')
    ch_2018 = float(df_ch.loc[df_ch['Jahr'] == 2018, 'Beobachtungen'].iat[0])
    ch_2050 = float(df_ch.loc[df_ch['Jahr'] == 2050, 'Referenzszenario A-00-2025'].iat[0])
    ch_factor = ch_2050 / ch_2018

    proj = bz[bz['daten'] == 'Prognose_Trend_ZH_2022']
    b18 = proj[proj['jahr'] == 2018].groupby('bezirk')['anzahl'].sum()
    b50 = proj[proj['jahr'] == 2050].groupby('bezirk')['anzahl'].sum()

    bezirk_scaling = {}
    for bz_name in b50.index:
        start = b18.get(bz_name, 0.0)
        end   = b50.get(bz_name, 0.0)
        if start > 0 and ch_factor > 1:
            rel = (end / start - 1) / (ch_factor - 1)
            bezirk_scaling[bz_name] = rel ** (1 / 32)
        else:
            bezirk_scaling[bz_name] = 1.0

    mapping = _load_pop_xlsx_gebietszuordnung().set_index('bfs_nr')['bezirk']

    current = pop_2050.copy()
    for y in range(2051, year + 1):
        ch_rate = float(gr_row[str(y)])
        commune_rates = mapping.reindex(current.index).map(
            lambda b: ch_rate * bezirk_scaling.get(b, 1.0)  # noqa: B023
        ).fillna(ch_rate)
        current = current * (1.0 + commune_rates)

    return current.rename(year)


def load_commune_pop(year: int) -> pd.Series:
    """Return commune-level population for any year as Series[int BFS_NR -> float pop].

    Coverage:
      1962-2025: actual values from POPULATION_CANTON_ZH_XLSX (Gemeinden sheet)
      2026-2050: 'Prognose_Trend_ZH_2022' bezirk totals disaggregated to communes
                 using 2025 commune distribution shares
      2051-2100: 2050 commune values grown by Eurostat Swiss annual growth rates
                 scaled by the same per-bezirk factor used in random_scenarios.py

    Args:
        year: Target calendar year (1962-2100).

    Returns:
        Series indexed by integer BFS-NR with float population values.
    """
    if 1962 <= year <= 2025:
        df = _load_pop_xlsx_gemeinden()
        col = f'TOTAL_{year}'
        if col not in df.columns:
            raise ValueError(f"load_commune_pop: column '{col}' not found in xlsx")
        s = df.set_index('BFS-NR')[col]
        s.index.name = 'BFS-NR'
        return pd.to_numeric(s, errors='coerce').fillna(0.0).rename(year)

    if 2026 <= year <= 2050:
        return _commune_pop_from_bezirk_projection(year)

    if 2051 <= year <= 2100:
        return _commune_pop_eurostat_extended(year)

    raise ValueError(
        f"load_commune_pop: no data for year {year} (supported range: 1962-2100)"
    )


def load_commune_empl(year: int) -> pd.Series:
    """Return commune-level FTE employment for any year as Series[int BFS_NR -> float FTE].

    Coverage:
      2011-2023: actual total FTE per commune from EMPLOYMENT_CANTON_ZH_CSV
      2024+    : 2023 FTE scaled by pop(year)/pop(2023) per commune (Option A —
                 employment tracks population growth)

    Args:
        year: Target calendar year (2011+).

    Returns:
        Series indexed by integer BFS-NR with float FTE values.
    """
    if 2011 <= year <= 2023:
        df = pd.read_csv(paths.EMPLOYMENT_CANTON_ZH_CSV)
        mask = (
            (df['areatype_name'] == 'Gemeinde') &
            (df['year'] == year) &
            (~df['unit'].str.contains('Prozent', na=False))
        )
        fte = df[mask].groupby('area_code')['value'].sum()
        fte.index = fte.index.astype(int)
        return fte.rename(year)

    if year < 2011:
        raise ValueError(f"load_commune_empl: no employment data before 2011 (got {year})")

    empl_2023 = load_commune_empl(2023)
    pop_year  = load_commune_pop(year)
    pop_2023  = load_commune_pop(2023)

    all_bfs   = empl_2023.index.union(pop_year.index).union(pop_2023.index)
    empl_2023 = empl_2023.reindex(all_bfs, fill_value=0.0)
    pop_year  = pop_year.reindex(all_bfs, fill_value=0.0)
    pop_2023  = pop_2023.reindex(all_bfs, fill_value=0.0)

    scale = (pop_year / pop_2023.where(pop_2023 > 0, other=np.nan)).fillna(1.0)
    return (empl_2023 * scale).rename(year)


def get_commune_growth_factors(start_year: int, target_year: int) -> pd.Series:
    """Return per-commune growth factor from start_year to target_year.

    Blends population and employment growth using settings.OD_SCALING_POP_WEIGHT
    and settings.OD_SCALING_EMPL_WEIGHT. Because employment currently tracks
    population (Option A), the blended result equals the population factor
    regardless of weights. The weight parameters exist for future use when
    an independent employment projection is wired in.

    Args:
        start_year:  Reference year (denominator).
        target_year: Target year (numerator).

    Returns:
        Series[int BFS_NR -> float factor] where 1.0 means no change.
    """
    w_pop  = settings.OD_SCALING_POP_WEIGHT
    w_empl = settings.OD_SCALING_EMPL_WEIGHT

    pop_start  = load_commune_pop(start_year)
    pop_target = load_commune_pop(target_year)
    all_bfs    = pop_start.index.union(pop_target.index)
    pop_start  = pop_start.reindex(all_bfs, fill_value=0.0)
    pop_target = pop_target.reindex(all_bfs, fill_value=0.0)
    pop_factor = (
        pop_target / pop_start.where(pop_start > 0, other=np.nan)
    ).fillna(1.0)

    if w_empl <= 0.0:
        return pop_factor.rename(f'growth_{start_year}_{target_year}')

    empl_start  = load_commune_empl(start_year)
    empl_target = load_commune_empl(target_year)
    all_bfs     = all_bfs.union(empl_start.index).union(empl_target.index)
    pop_factor   = pop_factor.reindex(all_bfs, fill_value=1.0)
    empl_start   = empl_start.reindex(all_bfs, fill_value=0.0)
    empl_target  = empl_target.reindex(all_bfs, fill_value=0.0)
    empl_factor  = (
        empl_target / empl_start.where(empl_start > 0, other=np.nan)
    ).fillna(1.0)

    w_total = w_pop + w_empl
    blended = (w_pop * pop_factor + w_empl * empl_factor) / w_total
    return blended.rename(f'growth_{start_year}_{target_year}')


# ===============================================================================
# STANDALONE ENTRY POINT
# ===============================================================================

def _format_scale_table(year: int) -> list:
    """Build a combined Bezirk scaling table from _BEZIRK_SCALES and _TOTALS."""
    pop_s  = _BEZIRK_SCALES.get('population',  pd.Series(dtype=float))
    empl_s = _BEZIRK_SCALES.get('employment', pd.Series(dtype=float))
    all_bz = sorted(set(list(pop_s.index) + list(empl_s.index)))
    if not all_bz:
        return []

    col_w = max(len(b) for b in all_bz + ['Bezirk', 'Mean'])
    sep   = f"  {'-' * (col_w + 2)}+{'-' * 12}+{'-' * 11}"
    lines = [
        f"\n  Grid scaling 2023 → {year} by Bezirk:",
        f"  {'Bezirk':<{col_w}}  | {'Population':>10} | {'Employment':>10}",
        sep,
    ]
    for bz in all_bz:
        p = pop_s.get(bz, float('nan'))
        e = empl_s.get(bz, float('nan'))
        p_str = f"{p:.4f}" if not pd.isna(p) else "n/a"
        e_str = f"{e:.4f}" if not pd.isna(e) else "n/a"
        lines.append(f"  {bz:<{col_w}}  | {p_str:>10} | {e_str:>10}")
    lines.append(sep)
    pm     = float(pop_s.mean())  if len(pop_s)  > 0 else float('nan')
    em     = float(empl_s.mean()) if len(empl_s) > 0 else float('nan')
    pm_str = f"{pm:.4f}" if not pd.isna(pm) else "n/a"
    em_str = f"{em:.4f}" if not pd.isna(em) else "n/a"
    lines.append(f"  {'Mean':<{col_w}}  | {pm_str:>10} | {em_str:>10}")
    lines.append("")
    for lbl in ['population', 'employment']:
        t = _TOTALS.get(lbl)
        if t:
            lines.append(
                f"  Total {lbl:<12}: {t['base']:>12,.0f} (2023) → {t['target']:>12,.0f} ({year})"
            )
    return lines


def main(year: int = None, do_plots: bool = True) -> str:
    """Run Step 2 of the catchment pipeline.

    Loads the 2023 federal population and employment grids, filters them to
    the catchment boundary, scales cell values to `year` using Bezirk-level
    cantonal data, and writes the outputs.

    Args:
        year:     Target year for cell scaling and output filenames.
                  Defaults to settings.POPULATION_BASE_YEAR.
        do_plots: When True, generates choropleth and raster PDF maps.

    Returns:
        Bezirk scaling report string for appending to the pipeline report file.
        Empty string if year == 2023 (no scaling applied).

    Produces (under paths.MAIN):
      - data/Catchment_Area/Pop_Empl_Data/population_{year}_<canton>.csv
      - data/Catchment_Area/Pop_Empl_Data/employment_{year}_<canton>.csv
      - data/Catchment_Area/Pop_Empl_Data/municipal_pop_empl_summary_{year}.csv
      - data/Catchment_Area/Pop_Empl_Data/municipal_pop_empl_summary.csv   (unsuffixed legacy alias)
      - plots/Catchment_Area/Pop_Empl_Data/plot_population_by_municipality_{year}.pdf  (if do_plots)
      - plots/Catchment_Area/Pop_Empl_Data/plot_employment_by_municipality_{year}.pdf  (if do_plots)
      - plots/Catchment_Area/Pop_Empl_Data/plot_population_raster_{year}.pdf           (if do_plots)
      - plots/Catchment_Area/Pop_Empl_Data/plot_employment_raster_{year}.pdf           (if do_plots)

    All Pop/FTE values written to disk (cell-level NUMMER and per-municipality
    totals) are integers. Per-commune cell sums equal the per-municipality
    integer total exactly (Hamilton's largest-remainder rounding).
    """
    global _SCALE_REPORT_LINES, _BEZIRK_SCALES, _TOTALS
    _SCALE_REPORT_LINES = []
    _BEZIRK_SCALES = {}
    _TOTALS = {}

    if year is None:
        year = settings.POPULATION_BASE_YEAR

    print(f"\n=== catchment_base: Step 2 (year={year}) ===")
    os.chdir(paths.MAIN)
    os.makedirs(CATCHMENT_DATA_DIR, exist_ok=True)
    os.makedirs(POP_EMPL_DATA_DIR, exist_ok=True)
    os.makedirs(CATCHMENT_PLOT_DIR, exist_ok=True)
    os.makedirs(POP_EMPL_PLOT_DIR, exist_ok=True)

    boundary  = _load_catchment_boundary()
    pop_grid  = _load_population_grid(boundary, year)
    empl_grid = _load_employment_grid(boundary, year)
    _SCALE_REPORT_LINES.extend(_format_scale_table(year))

    summary   = _cumulate_per_municipality(pop_grid, empl_grid, boundary, year)

    if do_plots:
        _plot_municipal_distributions(summary, boundary, year)
        _plot_raster_map(pop_grid,  'population', boundary, year)
        _plot_raster_map(empl_grid, 'employment', boundary, year)

    print(f"\n=== catchment_base done (year={year}) ===")
    return '\n'.join(_SCALE_REPORT_LINES)


def _run_step2_cli():
    """Interactive standalone entry for Step 2 — prompts for year then calls main()."""
    print("\n" + "─" * 68)
    print("[Phase 3]  Population / employment grid")
    print("─" * 68)

    print(f"\n[3.1]  Target year")
    print(f"       Coverage: 1962–2025 (actual), 2026–2050 (bezirk projection), 2051–2100 (Eurostat)")
    print(f"       Default: {settings.POPULATION_BASE_YEAR} (settings.py)")
    while True:
        year_str = input(f"   Year [{settings.POPULATION_BASE_YEAR}]: ").strip() or str(settings.POPULATION_BASE_YEAR)
        try:
            chosen_year = int(year_str)
            if 1962 <= chosen_year <= 2100:
                break
            print("   Year must be between 1962 and 2100.")
        except ValueError:
            print("   Please enter a valid year.")

    do_plots_str = input("\n[3.2]  Generate plots? (y/n) "
                         f"[{'y' if settings.PLOT_DATA else 'n'}]: ").strip().lower()
    if do_plots_str == '':
        do_plots = settings.PLOT_DATA
    else:
        do_plots = do_plots_str != 'n'

    main(year=chosen_year, do_plots=do_plots)


if __name__ == '__main__':
    initialise()
    _run_step2_cli()
