# initialisation.py
#
# Defines the study area and catchment area for infraScanRail.
# Run this once before executing any other pipeline module.
#
# Outputs
# -------
#   data/Study_Area/study_area_boundary.gpkg  — exact study area polygon
#   data/Study_Area/study_area_buffer.gpkg    — study area + margin (feeder network edge handling)
#   data/Catchment_Area/catchment_area_boundary.gpkg — dissolved admin boundary
#   data/Catchment_Area/catchment_area_buffer.gpkg   — boundary + GTFS buffer

import os

import geopandas as gpd
import pandas as pd

import paths
import settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CODEBASE_CRS = 'EPSG:2056'

CANTONS_GPKG        = paths.CANTON_BOUNDARIES_GPKG
BEZIRKE_GPKG        = paths.BEZIRKE_BOUNDARIES_GPKG
MUNICIPALITIES_GPKG = paths.MUNICIPAL_BOUNDARIES_GPKG

STUDY_AREA_DIR     = paths.CATCHMENT_AREA_DIR
CATCHMENT_AREA_DIR = paths.CATCHMENT_AREA_DIR

STUDY_AREA_DEFAULT_BUFFER_M     = 3000
CATCHMENT_AREA_DEFAULT_BUFFER_M = 5000


# ===========================================================================
# Shared boundary helpers
# ===========================================================================

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
    """Three-part admin boundary TUI: level → entity → subdivisions.

    Parameters
    ----------
    q_level            : str             Question label for admin level,      e.g. "1.2"
    q_entity           : str             Question label for entity selection, e.g. "1.3"
    q_subdiv           : str             Question label for subdivisions,     e.g. "1.4"
    study_area_polygon : shapely geom|None
        If provided, all admin entities at the chosen level that intersect the
        study area are computed after admin level is selected and shown as
        suggestions before entity selection.

    Returns
    -------
    tuple (admin_level: str, primary_names: list[str], subdivision_names: dict)
    """
    # ── Admin level ──────────────────────────────────────────────────────────
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

    # ── Suggestions based on study area overlap ───────────────────────────────
    suggestions = []
    if study_area_polygon is not None and admin_level != 'national':
        suggestions = _compute_overlapping_entities(study_area_polygon, admin_level)

    # ── Entity selection ─────────────────────────────────────────────────────
    primary_names = []

    if admin_level == 'national':
        print(f"\n[{q_entity}]  Entity selection — skipped (national level selected)")
    else:
        print(f"\n[{q_entity}]  Entity selection")
        if suggestions:
            entity_type = {'cantonal': 'cantons', 'bezirke': 'Bezirke', 'municipal': 'municipalities'}[admin_level]
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

    # ── Subdivisions ─────────────────────────────────────────────────────────
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
    """Dissolve selected admin units into a single polygon in CODEBASE_CRS."""
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
        parts.append(_load_and_filter(os.path.join(paths.MAIN, CANTONS_GPKG), names=primary_names))
    elif admin_level == 'bezirke':
        parts.append(_load_and_filter(os.path.join(paths.MAIN, BEZIRKE_GPKG), names=primary_names))
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
    """Write a single polygon as a GeoPackage."""
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


# ===========================================================================
# Smart catchment suggestion
# ===========================================================================

def _compute_overlapping_entities(study_area_polygon, admin_level):
    """Return names of all admin entities at admin_level that intersect the study area."""
    try:
        if admin_level == 'cantonal':
            gpkg_path       = os.path.join(paths.MAIN, CANTONS_GPKG)
            objektart_filter = None
        elif admin_level == 'bezirke':
            gpkg_path       = os.path.join(paths.MAIN, BEZIRKE_GPKG)
            objektart_filter = None
        elif admin_level == 'municipal':
            gpkg_path       = os.path.join(paths.MAIN, MUNICIPALITIES_GPKG)
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


# ===========================================================================
# Phase 1: Study area
# ===========================================================================

def _configure_study_area():
    print("\n" + "─" * 68)
    print("[Phase 1]  Study area")
    print("─" * 68)

    # [1.1] Definition method
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
        # [1.2] Admin level  [1.3] Entity selection  [1.4] Subdivisions
        admin_level, primary_names, subdivision_names = _select_admin_boundary(
            q_level='1.2', q_entity='1.3', q_subdiv='1.4',
        )
        print("\n   Dissolving selected boundaries ...")
        polygon = _dissolve_admin_polygon(admin_level, primary_names, subdivision_names)
        print(f"   Bounds: {polygon.bounds}")

    # [1.5] Buffer
    buffer_m = _ask_buffer(
        "1.5", STUDY_AREA_DEFAULT_BUFFER_M,
        note="Creates a margin for capturing feeder network elements near the study area edge.",
    )

    return polygon, polygon.buffer(buffer_m), admin_level, primary_names, buffer_m


# ===========================================================================
# Phase 2: Catchment area
# ===========================================================================

def _configure_catchment_area(study_area_boundary):
    print("\n" + "─" * 68)
    print("[Phase 2]  Catchment area")
    print("─" * 68)

    # [2.1] Admin level  [2.2] Entity selection (with study area suggestions)  [2.3] Subdivisions
    admin_level, primary_names, subdivision_names = _select_admin_boundary(
        q_level='2.1', q_entity='2.2', q_subdiv='2.3',
        study_area_polygon=study_area_boundary,
    )

    print("\n   Dissolving selected boundaries ...")
    polygon = _dissolve_admin_polygon(admin_level, primary_names, subdivision_names)
    print(f"   Bounds: {polygon.bounds}")

    # [2.4] Buffer
    buffer_m = _ask_buffer(
        "2.4", CATCHMENT_AREA_DEFAULT_BUFFER_M,
        note="Buffer around the catchment boundary for GTFS spatial filtering.",
    )

    return polygon, polygon.buffer(buffer_m), admin_level, primary_names, buffer_m


# ===========================================================================
# Validation
# ===========================================================================

def _validate_containment(study_area_boundary, catchment_boundary):
    print("\n" + "─" * 68)
    print("[Check]  Validating: study area ⊆ catchment area")
    print("─" * 68)

    outside     = study_area_boundary.difference(catchment_boundary)
    outside_pct = outside.area / study_area_boundary.area * 100

    if outside_pct < 0.01:
        print("   Pass — study area is fully within the catchment area.")
        return True

    print(f"   Warning — {outside_pct:.1f}% of the study area lies outside the catchment area.")
    return False


# ===========================================================================
# Main entry point
# ===========================================================================

def initialise():
    os.chdir(paths.MAIN)

    print("=" * 68)
    print("infraScanRail — Initialisation")
    print("=" * 68)
    print("Defines the study area and catchment area and exports them as")
    print("GeoPackages for use by downstream pipeline modules.")

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    sa_boundary, sa_buffer, sa_admin, sa_names, sa_buf = _configure_study_area()

    sa_boundary_path = os.path.join(paths.MAIN, STUDY_AREA_DIR, 'study_area_boundary.gpkg')
    sa_buffer_path   = os.path.join(paths.MAIN, STUDY_AREA_DIR, 'study_area_buffer.gpkg')

    _export_gpkg(sa_boundary, 'study_area_boundary', sa_admin, sa_names, 0,      sa_boundary_path)
    _export_gpkg(sa_buffer,   'study_area_buffer',   sa_admin, sa_names, sa_buf, sa_buffer_path)

    # ── Phase 2 (with validation loop) ───────────────────────────────────────
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

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("Summary")
    print("=" * 68)
    print(f"  Study area boundary   : {sa_boundary_path}")
    print(f"  Study area buffer     : {sa_buffer_path}  (+{sa_buf:.0f} m)")
    print(f"  Catchment boundary    : {ca_boundary_path}")
    print(f"  Catchment buffer      : {ca_buffer_path}  (+{ca_buf:.0f} m)")
    print("=" * 68 + "\n")


if __name__ == '__main__':
    initialise()
