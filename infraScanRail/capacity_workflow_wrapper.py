"""Standalone runner for rail capacity analysis.
Last modified: 2026-05-20

Interactive CLI that discovers available infrastructure and service versions,
then dispatches to one of four named workflows. All outputs are written to:
  data/Network/Capacity/<infra_version>/<svc_version>/

Workflows:
  Study Area     — capacity run scoped to the study area (SA-filtered infra + services)
  Catchment Area — capacity run for the full catchment (SA Dynamic + CA Set_Value or uniform)
  Development    — capacity run for a development infra version
  Expanded       — capacity interventions on an existing Study Area or CA run

Phase 0 — interactive CLI: choose infra version + service version + workflow
Phase 1 — build capacity tables (stations + segments, peak + off-peak)
Phase 2 — build sections (Dynamic or Set_Value)
Phase 3 — plot capacity network
[Phase 4] — Expanded workflow only: capacity interventions

"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import paths
import settings

_PLOTS_ROOT = Path(paths.MAIN) / paths.CAPACITY_PLOTS_DIR
from capacity_calculator import (
    CAPACITY_ROOT,
    EXCEL_ENGINE,
    _build_sections_dataframe,
    build_capacity_tables,
    build_capacity_tables_sa,
    build_capacity_tables_ca,
)
from capacity_interventions import (
    run_phase_four,
    visualize_enhanced_network,
    cap_ints_to_spatial,
)
from infrabuild_version_manager import append_cap_intervention_rows, resolve_or_build

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_ROOT   = Path(paths.MAIN) / "data" / "Network"
_INFRA_ROOT = Path(paths.MAIN) / paths.NETWORK_INFRASTRUCTURE_DIR
_RAIL_ROOT  = Path(paths.MAIN) / paths.RAIL_LINES_DIR


# ---------------------------------------------------------------------------
# Version discovery
# ---------------------------------------------------------------------------

def _list_infra_versions() -> List[str]:
    """Return sorted infra version names that have nodes.gpkg and segments.gpkg."""
    if not _INFRA_ROOT.exists():
        return []
    return sorted(
        d.name for d in _INFRA_ROOT.iterdir()
        if d.is_dir()
        and not d.name.startswith("Raw")
        and d.name != "Developments"
        and (d / "nodes.gpkg").exists()
        and (d / "segments.gpkg").exists()
    )


def _list_svc_versions(infra_version: str) -> List[str]:
    """Return svc version names that have rail_segments.gpkg projected on infra_version."""
    if not _RAIL_ROOT.exists():
        return []
    results = []
    for entry in sorted(_RAIL_ROOT.iterdir()):
        if not entry.is_dir() or not entry.name.endswith("_network"):
            continue
        if (entry / infra_version / "rail_segments.gpkg").exists():
            results.append(entry.name.removesuffix("_network"))
    return results


def _existing_sections(infra_version: str, svc_version: str, label: str) -> Optional[Path]:
    """Return path to existing sections workbook, or None."""
    p = CAPACITY_ROOT / infra_version / svc_version / f"sections_{label}.xlsx"
    return p if p.exists() else None


def _safe(*parts: str) -> str:
    return re.sub(r"[^\w-]+", "_", "_".join(parts)).strip("_")


def _floor_capacity_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast capacity columns to nullable Int64 (floor any fractional value).

    Covers the selected Capacity_peak / Capacity_offpeak as well as the
    diagnostic per-strategy columns (capacity_single_track_tphpd, capacity_good_tphpd,
    etc.). Fractional tphpd values do not make physical sense (you cannot run
    half a train), so we round down conservatively. pd.NA is used for sections
    that have no computed capacity.
    """
    capacity_cols = [
        col for col in df.columns
        if col in ("Capacity_peak", "Capacity_offpeak")
        or (col.startswith("capacity_") and col.endswith("_tphpd"))
    ]
    for col in capacity_cols:
        df[col] = df[col].apply(
            lambda x: pd.NA if pd.isna(x) else int(np.floor(x))
        ).astype("Int64")
    return df


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _pick_one(labels: List[str], prompt: str = "Select") -> Optional[int]:
    """Display numbered list; return 0-based index or None on empty Enter."""
    for i, lbl in enumerate(labels, 1):
        print(f"    {i}) {lbl}")
    while True:
        raw = input(f"  {prompt} (number): ").strip()
        if not raw:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(labels):
            return int(raw) - 1
        print(f"  Invalid — enter 1-{len(labels)} or press Enter to cancel.")


def _pick_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"  {prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Enter a numeric value.")


def _select_versions(require_existing_sections: bool = False) -> Optional[Tuple[str, str]]:
    """Prompt for infra + svc version; return (infra_version, svc_version) or None."""
    infra_versions = _list_infra_versions()
    if not infra_versions:
        print("\n  ERROR: No infrastructure versions found.")
        print(f"  Expected folders with nodes.gpkg + segments.gpkg under {_INFRA_ROOT}")
        print("  Run infrabuild_network_builder.py first.")
        return None

    print("\n  Infrastructure version:")
    idx = _pick_one(infra_versions, "Infra version")
    if idx is None:
        return None
    infra_version = infra_versions[idx]

    svc_versions = _list_svc_versions(infra_version)
    if not svc_versions:
        print(f"\n  ERROR: No projected service versions found for '{infra_version}'.")
        print("  Run services_service_projection.py for this infrastructure version first.")
        return None

    if require_existing_sections:
        svc_versions = [
            s for s in svc_versions
            if _existing_sections(infra_version, s, _safe(s, infra_version)) is not None
        ]
        if not svc_versions:
            print(f"\n  ERROR: No completed Study Area runs found for '{infra_version}'.")
            print("  Run a Study Area workflow first (Phase 1-3).")
            return None

    print("\n  Service version:")
    idx = _pick_one(svc_versions, "Service version")
    if idx is None:
        return None
    return infra_version, svc_versions[idx]


def _select_capacity_mode() -> Tuple[str, Optional[float]]:
    """Prompt for capacity mode; return (mode_str, set_value_or_None)."""
    print("\n  Capacity mode:")
    print("    1) Dynamic   — UIC formula per section")
    print("    2) Set_Value — uniform value x track_count for all sections")
    while True:
        choice = input("  Select (1/2): ").strip()
        if choice in ("1", "2"):
            break
        print("  Enter 1 or 2.")
    if choice == "2":
        return "Set_Value", _pick_float(
            "Set value (trains/hour/direction)", default=settings.CAPACITY_SET_VALUE
        )
    return "Dynamic", None


def _select_grouping_strategy() -> str:
    """Prompt for grouping strategy (Dynamic mode only); return strategy string."""
    print("\n  Grouping strategy (for 2-track sections with multiple options):")
    print("    1) Manual      — prompt for each ambiguous section")
    print("    2) Conservative — always pick the lowest capacity option")
    print("    3) Baseline    — always pick the middle option")
    print("    4) Optimal     — always pick the highest capacity option")
    _map = {"1": "manual", "2": "conservative", "3": "baseline", "4": "optimal"}
    default = getattr(settings, "CAPACITY_GROUPING_STRATEGY", "manual")
    default_key = {v: k for k, v in _map.items()}.get(default, "1")
    while True:
        raw = input(f"  Select (1-4) [{default_key}]: ").strip()
        if not raw:
            return default
        if raw in _map:
            return _map[raw]
        print("  Enter 1-4.")


# ---------------------------------------------------------------------------
# Shared execution core
# ---------------------------------------------------------------------------

def _execute_capacity_run(
    infra_version: str,
    svc_version: str,
    label: str,
    out_dir: Path,
    capacity_mode: str = "Dynamic",
    set_value: Optional[float] = None,
    grouping_strategy: str = "manual",
    visualize: bool = True,
) -> int:
    """Core capacity pipeline: tables → sections → optional plot.

    Args:
        infra_version:     Infrastructure version name.
        svc_version:       Service version name.
        label:             Filename label (used in workbook names).
        out_dir:           Directory to write outputs.
        capacity_mode:     'Dynamic' or 'Set_Value'.
        set_value:         Fixed tphpd/track when capacity_mode == 'Set_Value'.
        grouping_strategy: How to resolve ambiguous 2-track groupings in Dynamic mode.
        visualize:         Whether to plot after writing sections.

    Returns:
        0 on success, 1 on error.
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        cap_path      = out_dir / f"capacity_{label}.xlsx"
        sections_path = out_dir / f"sections_{label}.xlsx"

        print(f"\n{'='*70}")
        print(f"Infra:    {infra_version}")
        print(f"Service:  {svc_version}")
        if capacity_mode == "Dynamic":
            print(f"Strategy: {grouping_strategy}")
        print(f"Output:   {out_dir}")
        print(f"{'='*70}\n")

        # ── Phase 1: Capacity tables ─────────────────────────────────────────
        print("Phase 1: Building capacity tables ...\n")
        stations_peak, segments_peak, stations_offpeak, segments_offpeak, junction_numbers = \
            build_capacity_tables(infra_version, svc_version, label, output_dir=out_dir)
        print(f"\n  Capacity workbook: {cap_path}")

        # ── Phase 2: Sections ────────────────────────────────────────────────
        print(f"\nPhase 2: Building sections ({capacity_mode}) ...\n")
        sections_df = _build_sections_dataframe(
            stations_peak, segments_peak,
            junction_numbers=junction_numbers,
            segments_offpeak_df=segments_offpeak,
            compute_capacity=(capacity_mode == "Dynamic"),
            grouping_strategy=grouping_strategy,
        )

        if sections_df.empty:
            print("  WARNING: No sections produced. Check infra/service data.")
            return 1

        if capacity_mode == "Set_Value":
            cap_val = set_value if set_value is not None else settings.CAPACITY_SET_VALUE
            capacity = np.floor(sections_df["track_count"] * cap_val)
            sections_df["Capacity_peak"]       = capacity
            sections_df["Capacity_offpeak"]    = capacity
            sections_df["Utilization_peak"]    = (
                sections_df["total_tphpd_peak"] / sections_df["Capacity_peak"]
            ).where(sections_df["Capacity_peak"] > 0)
            sections_df["Utilization_offpeak"] = (
                sections_df["total_tphpd_offpeak"] / sections_df["Capacity_offpeak"]
            ).where(sections_df["Capacity_offpeak"] > 0)
            print(f"  Set_Value {cap_val} tphpd/track applied to all {len(sections_df)} sections.")
        else:
            n_with = int(sections_df["Capacity_peak"].notna().sum())
            print(f"  {n_with}/{len(sections_df)} sections have calculated peak capacity.")

        _floor_capacity_columns(sections_df)
        with pd.ExcelWriter(sections_path, engine=EXCEL_ENGINE) as writer:
            stations_peak.to_excel(writer,    sheet_name="Stations_Peak",    index=False)
            segments_peak.to_excel(writer,    sheet_name="Segments_Peak",    index=False)
            stations_offpeak.to_excel(writer, sheet_name="Stations_Offpeak", index=False)
            segments_offpeak.to_excel(writer, sheet_name="Segments_Offpeak", index=False)
            sections_df.to_excel(writer,      sheet_name="Sections",         index=False)
        print(f"\n  Sections workbook: {sections_path}")

        # ── Phase 3: Plot ────────────────────────────────────────────────────
        if visualize:
            print("\nPhase 3: Plotting ...\n")
            try:
                from capacity_network_plots import plot_capacity_network, plot_service_network
                _plots_dir = _PLOTS_ROOT / infra_version / svc_version
                _plots_dir.mkdir(parents=True, exist_ok=True)
                plot_capacity_network(
                    workbook_path=str(sections_path),
                    sections_workbook_path=str(sections_path),
                    network_label=label,
                    infra_version=infra_version,
                    output_dir=_plots_dir,
                )
                print("  Capacity network plot complete.")
                plot_service_network(
                    workbook_path=str(sections_path),
                    network_label=label,
                    output_dir=_plots_dir,
                )
                print("  Service network plot complete.")
            except Exception as exc:
                print(f"  WARNING: Plot failed: {exc}")

        print(f"\n{'='*70}")
        print("COMPLETE")
        print(f"{'='*70}\n")
        return 0

    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        import traceback
        traceback.print_exc()
        return 1


def _make_phase4_prep_workbook(
    sections_workbook_path: Path,
    prep_path: Path,
) -> None:
    """Write a Phase 4-compatible prep workbook from a new-format sections workbook.

    Phase 4 (capacity_interventions.py) expects 'Stations' and 'Segments' sheets
    with old column names ('tracks', 'platforms'). This adapter converts the
    peak tables from the new format.
    """
    stations_df = pd.read_excel(sections_workbook_path, sheet_name="Stations_Peak")
    segments_df = pd.read_excel(sections_workbook_path, sheet_name="Segments_Peak")

    # Rename new column names to the old names that capacity_interventions expects
    stations_df = stations_df.rename(columns={
        "Track_Count":    "tracks",
        "Platform_Count": "platforms",
        "Name":           "NAME",
        "Code":           "CODE",
    })
    segments_df = segments_df.rename(columns={
        "Num_Tracks":   "tracks",
        "Length":       "length_m",
        "TT_Stopping":  "travel_time_stopping",
        "TT_Passing":   "travel_time_passing",
        "Average_Speed": "speed",
    })

    with pd.ExcelWriter(prep_path, engine="openpyxl") as writer:
        stations_df.to_excel(writer, sheet_name="Stations", index=False)
        segments_df.to_excel(writer, sheet_name="Segments", index=False)


# ---------------------------------------------------------------------------
# Workflow functions
# ---------------------------------------------------------------------------

def run_study_area_workflow(
    infra_version: str,
    svc_version: str,
    capacity_mode: str = None,
    set_value: Optional[float] = None,
    grouping_strategy: str = None,
    visualize: bool = True,
) -> int:
    """Capacity run scoped to the study area.

    Uses SA-filtered infrastructure (nodes from rail_segments_sa.gpkg path_nodes)
    and SA-filtered projected services (rail_segments_sa.gpkg). Skips the UIC
    formula entirely when capacity_mode is Set_Value.

    Args:
        infra_version:     Infrastructure version name.
        svc_version:       Service version name.
        capacity_mode:     'Dynamic' or 'Set_Value'; defaults to settings.CAPACITY_MODE_SA.
        set_value:         Fixed tphpd/track for Set_Value mode.
        grouping_strategy: How to resolve ambiguous 2-track groupings in Dynamic mode
                           ('manual', 'conservative', 'baseline', 'optimal').
        visualize:         Whether to generate plots.

    Returns:
        Exit code (0 success, 1 error).
    """
    print("\n" + "=" * 70)
    print("STUDY AREA WORKFLOW")
    print("=" * 70)

    cap_mode  = capacity_mode or settings.CAPACITY_MODE_SA
    cap_val   = set_value if set_value is not None else settings.CAPACITY_SET_VALUE
    grp_strat = grouping_strategy or getattr(settings, "CAPACITY_GROUPING_STRATEGY", "manual")
    label      = _safe(svc_version, infra_version)
    out_dir    = CAPACITY_ROOT / infra_version / svc_version
    plots_dir  = _PLOTS_ROOT   / infra_version / svc_version

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        sections_path = out_dir / f"sections_{label}.xlsx"

        print(f"\n{'='*70}")
        print(f"Infra:     {infra_version}")
        print(f"Service:   {svc_version}")
        print(f"Scope:     Study Area")
        print(f"Capacity:  {cap_mode}")
        if cap_mode == "Dynamic":
            print(f"Strategy:  {grp_strat}")
        print(f"Output:    {out_dir}")
        print(f"{'='*70}\n")

        print("Phase 1: Building SA capacity tables ...\n")
        stations_peak, segments_peak, stations_offpeak, segments_offpeak, junction_numbers, _ = \
            build_capacity_tables_sa(infra_version, svc_version, label, output_dir=out_dir)

        print(f"\nPhase 2: Building sections ({cap_mode}) ...\n")
        sections_df = _build_sections_dataframe(
            stations_peak, segments_peak,
            junction_numbers=junction_numbers,
            segments_offpeak_df=segments_offpeak,
            compute_capacity=(cap_mode == "Dynamic"),
            grouping_strategy=grp_strat,
        )

        if sections_df.empty:
            print("  WARNING: No sections produced. Check infra/service data.")
            return 1

        if cap_mode == "Set_Value":
            capacity = np.floor(sections_df["track_count"] * cap_val)
            sections_df["Capacity_peak"]       = capacity
            sections_df["Capacity_offpeak"]    = capacity
            sections_df["Utilization_peak"]    = (
                sections_df["total_tphpd_peak"] / sections_df["Capacity_peak"]
            ).where(sections_df["Capacity_peak"] > 0)
            sections_df["Utilization_offpeak"] = (
                sections_df["total_tphpd_offpeak"] / sections_df["Capacity_offpeak"]
            ).where(sections_df["Capacity_offpeak"] > 0)
            print(f"  Set_Value {cap_val} tphpd/track applied to all {len(sections_df)} sections.")
        else:
            n_with = int(sections_df["Capacity_peak"].notna().sum())
            print(f"  {n_with}/{len(sections_df)} sections have Dynamic capacity.")

        _floor_capacity_columns(sections_df)
        with pd.ExcelWriter(sections_path, engine=EXCEL_ENGINE) as writer:
            stations_peak.to_excel(writer,    sheet_name="Stations_Peak",    index=False)
            segments_peak.to_excel(writer,    sheet_name="Segments_Peak",    index=False)
            stations_offpeak.to_excel(writer, sheet_name="Stations_Offpeak", index=False)
            segments_offpeak.to_excel(writer, sheet_name="Segments_Offpeak", index=False)
            sections_df.to_excel(writer,      sheet_name="Sections",         index=False)
        print(f"\n  Sections workbook: {sections_path}")

        if visualize:
            print("\nPhase 3: Plotting ...\n")
            try:
                from capacity_network_plots import plot_capacity_network, plot_service_network
                plots_dir.mkdir(parents=True, exist_ok=True)
                plot_capacity_network(
                    workbook_path=str(sections_path),
                    sections_workbook_path=str(sections_path),
                    network_label=label,
                    infra_version=infra_version,
                    output_dir=plots_dir,
                    lakes_path=paths.LAKES_SA_GPKG,
                )
                print("  Capacity network plot complete.")
                plot_service_network(
                    workbook_path=str(sections_path),
                    network_label=label,
                    output_dir=plots_dir,
                    lakes_path=paths.LAKES_SA_GPKG,
                )
                print("  Service network plot complete.")
            except Exception as exc:
                print(f"  WARNING: Plot failed: {exc}")

        print(f"\n{'='*70}\nSTUDY AREA WORKFLOW COMPLETE\n{'='*70}\n")
        return 0

    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        import traceback
        traceback.print_exc()
        return 1


def run_catchment_area_workflow(
    infra_version: str,
    svc_version: str,
    mode_sa: str = None,
    mode_ca: str = None,
    set_value_sa: Optional[float] = None,
    set_value_ca: Optional[float] = None,
    grouping_strategy: str = None,
    visualize: bool = True,
) -> int:
    """Capacity run scoped to the catchment area with per-zone capacity methods.

    Infra and services are filtered to nodes within catchment_area_boundary.gpkg.
    When SA and CA methods differ (SA=Dynamic, CA=Set_Value), the UIC formula
    runs only for sections whose entire node sequence lies within the study area;
    all other sections receive Set_Value without running the formula.

    Args:
        infra_version:     Infrastructure version name.
        svc_version:       Service version name.
        mode_sa:           Capacity method for study-area sections ('Dynamic'/'Set_Value');
                           defaults to settings.CAPACITY_MODE_SA.
        mode_ca:           Capacity method for catchment-area sections outside the SA;
                           defaults to settings.CAPACITY_MODE_CA.
        set_value_sa:      Fixed tphpd/track for SA Set_Value mode.
        set_value_ca:      Fixed tphpd/track for CA Set_Value mode.
        grouping_strategy: How to resolve ambiguous 2-track groupings in Dynamic mode
                           ('manual', 'conservative', 'baseline', 'optimal').
        visualize:         Whether to generate plots.

    Returns:
        Exit code (0 success, 1 error).
    """
    print("\n" + "=" * 70)
    print("CATCHMENT AREA WORKFLOW")
    print("=" * 70)

    cap_mode_sa = mode_sa or settings.CAPACITY_MODE_SA
    cap_mode_ca = mode_ca or settings.CAPACITY_MODE_CA
    cap_val_sa  = set_value_sa if set_value_sa is not None else settings.CAPACITY_SET_VALUE
    cap_val_ca  = set_value_ca if set_value_ca is not None else settings.CAPACITY_SET_VALUE
    grp_strat   = grouping_strategy or getattr(settings, "CAPACITY_GROUPING_STRATEGY", "manual")
    label      = _safe(svc_version, infra_version) + "_ca"
    out_dir    = CAPACITY_ROOT / infra_version / svc_version
    plots_dir  = _PLOTS_ROOT   / infra_version / svc_version

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        sections_path = out_dir / f"sections_{label}.xlsx"
        modes_same    = (cap_mode_sa == cap_mode_ca)

        print(f"\n{'='*70}")
        print(f"Infra:    {infra_version}")
        print(f"Service:  {svc_version}")
        print(f"Scope:    Catchment Area")
        print(f"SA mode:  {cap_mode_sa}  |  CA mode: {cap_mode_ca}")
        if "Dynamic" in (cap_mode_sa, cap_mode_ca):
            print(f"Strategy: {grp_strat}")
        print(f"Output:   {out_dir}")
        print(f"{'='*70}\n")

        print("Phase 1: Building CA capacity tables ...\n")
        stations_peak, segments_peak, stations_offpeak, segments_offpeak, \
        junction_numbers, _, sa_node_set = \
            build_capacity_tables_ca(infra_version, svc_version, label, output_dir=out_dir)

        print(f"\nPhase 2: Building sections ...\n")

        if modes_same:
            # Single run — same method for all sections
            compute_cap = (cap_mode_sa == "Dynamic")
            sections_df = _build_sections_dataframe(
                stations_peak, segments_peak,
                junction_numbers=junction_numbers,
                segments_offpeak_df=segments_offpeak,
                compute_capacity=compute_cap,
                grouping_strategy=grp_strat,
            )
            if not sections_df.empty and cap_mode_sa == "Set_Value":
                cap_val = cap_val_sa
                capacity = np.floor(sections_df["track_count"] * cap_val)
                sections_df["Capacity_peak"]       = capacity
                sections_df["Capacity_offpeak"]    = capacity
                sections_df["Utilization_peak"]    = (
                    sections_df["total_tphpd_peak"] / sections_df["Capacity_peak"]
                ).where(sections_df["Capacity_peak"] > 0)
                sections_df["Utilization_offpeak"] = (
                    sections_df["total_tphpd_offpeak"] / sections_df["Capacity_offpeak"]
                ).where(sections_df["Capacity_offpeak"] > 0)
                print(f"  Set_Value {cap_val} tphpd/track applied to all {len(sections_df)} sections.")
            elif not sections_df.empty:
                n_with = int(sections_df["Capacity_peak"].notna().sum())
                print(f"  {n_with}/{len(sections_df)} sections have Dynamic capacity.")
        else:
            # Mixed: SA=Dynamic, CA=Set_Value
            # UIC runs only for SA sections (strict containment); CA sections return NaN.
            sections_df = _build_sections_dataframe(
                stations_peak, segments_peak,
                junction_numbers=junction_numbers,
                segments_offpeak_df=segments_offpeak,
                compute_capacity=True,
                sa_node_set=sa_node_set,
                grouping_strategy=grp_strat,
            )
            if not sections_df.empty:
                ca_mask = sections_df["Capacity_peak"].isna()
                n_sa  = int((~ca_mask).sum())
                n_ca  = int(ca_mask.sum())
                if ca_mask.any():
                    ca_capacity = np.floor(sections_df.loc[ca_mask, "track_count"] * cap_val_ca)
                    sections_df.loc[ca_mask, "Capacity_peak"]    = ca_capacity
                    sections_df.loc[ca_mask, "Capacity_offpeak"] = ca_capacity
                    sections_df.loc[ca_mask, "Utilization_peak"]    = (
                        sections_df.loc[ca_mask, "total_tphpd_peak"]
                        / sections_df.loc[ca_mask, "Capacity_peak"]
                    ).where(sections_df.loc[ca_mask, "Capacity_peak"] > 0)
                    sections_df.loc[ca_mask, "Utilization_offpeak"] = (
                        sections_df.loc[ca_mask, "total_tphpd_offpeak"]
                        / sections_df.loc[ca_mask, "Capacity_offpeak"]
                    ).where(sections_df.loc[ca_mask, "Capacity_offpeak"] > 0)
                print(f"  SA sections (Dynamic): {n_sa}  |  CA sections (Set_Value {cap_val_ca}): {n_ca}")

        if sections_df.empty:
            print("  WARNING: No sections produced. Check infra/service data.")
            return 1

        _floor_capacity_columns(sections_df)
        with pd.ExcelWriter(sections_path, engine=EXCEL_ENGINE) as writer:
            stations_peak.to_excel(writer,    sheet_name="Stations_Peak",    index=False)
            segments_peak.to_excel(writer,    sheet_name="Segments_Peak",    index=False)
            stations_offpeak.to_excel(writer, sheet_name="Stations_Offpeak", index=False)
            segments_offpeak.to_excel(writer, sheet_name="Segments_Offpeak", index=False)
            sections_df.to_excel(writer,      sheet_name="Sections",         index=False)
        print(f"\n  Sections workbook: {sections_path}")

        if visualize:
            print("\nPhase 3: CA Plotting ...\n")
            try:
                from capacity_network_plots import plot_capacity_network, plot_service_network
                plots_dir.mkdir(parents=True, exist_ok=True)
                plot_capacity_network(
                    workbook_path=str(sections_path),
                    sections_workbook_path=str(sections_path),
                    network_label=label,
                    infra_version=infra_version,
                    output_dir=plots_dir,
                    lakes_path=paths.LAKES_CA_GPKG,
                    include_labels=False,
                    marker_scale=0.25,
                    is_catchment=True,
                    network_marker_scale=2.25,
                    boundary_path=paths.CATCHMENT_AREA_BOUNDARY_GPKG,
                )
                print("  CA capacity network plot complete.")
                plot_service_network(
                    workbook_path=str(sections_path),
                    network_label=label,
                    output_dir=plots_dir,
                    lakes_path=paths.LAKES_CA_GPKG,
                    include_labels=False,
                )
                print("  CA service network plot complete.")
            except Exception as exc:
                print(f"  WARNING: CA plot failed: {exc}")

            # Also generate SA-scoped plots
            print("\nPhase 3b: SA Plotting ...\n")
            try:
                run_study_area_workflow(
                    infra_version, svc_version,
                    capacity_mode=cap_mode_sa,
                    set_value=cap_val_sa,
                    grouping_strategy=grp_strat,
                    visualize=True,
                )
                print("  SA plots complete.")
            except Exception as exc:
                print(f"  WARNING: SA plots failed: {exc}")

        print(f"\n{'='*70}\nCATCHMENT AREA WORKFLOW COMPLETE\n{'='*70}\n")
        return 0

    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        import traceback
        traceback.print_exc()
        return 1


def run_development_workflow(
    dev_id: str,
    infra_version: str,
    svc_version: str,
    capacity_mode: str = None,
    set_value: Optional[float] = None,
    grouping_strategy: str = None,
    visualize: bool = True,
) -> int:
    """Development network capacity run.

    In the new pipeline, a development is expressed as a dedicated infra version
    (built by infrabuild for that specific development scenario). The dev_id is
    used as a label in output paths and filenames for traceability.

    Args:
        dev_id:            Development identifier (e.g. '101032') — used for labelling.
        infra_version:     Development infra version name.
        svc_version:       Service version name.
        capacity_mode:     'Dynamic' or 'Set_Value'; defaults to settings.CAPACITY_MODE.
        set_value:         Fixed tphpd/track for Set_Value mode.
        grouping_strategy: How to resolve ambiguous 2-track groupings in Dynamic mode.
        visualize:         Whether to generate plots.

    Returns:
        Exit code (0 success, 1 error).
    """
    print("\n" + "=" * 70)
    print(f"DEVELOPMENT WORKFLOW — Dev ID: {dev_id}")
    print("=" * 70)

    cap_mode  = capacity_mode or settings.CAPACITY_MODE
    cap_val   = set_value if set_value is not None else settings.CAPACITY_SET_VALUE
    grp_strat = grouping_strategy or getattr(settings, "CAPACITY_GROUPING_STRATEGY", "manual")
    label    = _safe(f"dev_{dev_id}", svc_version, infra_version)
    out_dir  = CAPACITY_ROOT / infra_version / f"dev_{dev_id}_{svc_version}"

    print(f"\n  Development ID:   {dev_id}")
    print(f"  Infra version:    {infra_version}")
    print(f"  Service version:  {svc_version}")

    return _execute_capacity_run(
        infra_version, svc_version, label, out_dir,
        capacity_mode=cap_mode, set_value=cap_val,
        grouping_strategy=grp_strat, visualize=visualize,
    )


def run_enhanced_workflow(
    infra_version: str,
    svc_version: str,
    threshold: float = 2.0,
    max_iterations: int = 10,
) -> int:
    """Phase 4: apply iterative capacity interventions to an existing capacity run.

    Reads the completed sections workbook from the Study Area or Catchment Area
    output directory, runs the intervention loop, and writes enhanced outputs to
    an 'Enhanced' subdirectory.

    Args:
        infra_version:  Infrastructure version name.
        svc_version:    Service version name.
        threshold:      Minimum required available capacity in tphpd (default 2.0).
        max_iterations: Maximum intervention iterations (default 10).

    Returns:
        Exit code (0 success, 1 error).
    """
    print("\n" + "=" * 70)
    print("ENHANCED WORKFLOW — Phase 4 Capacity Interventions")
    print("=" * 70)

    label         = _safe(svc_version, infra_version)
    baseline_dir  = CAPACITY_ROOT / infra_version / svc_version
    sections_path = baseline_dir / f"sections_{label}.xlsx"
    output_dir    = baseline_dir / "Enhanced"

    print(f"\n  Baseline:   {baseline_dir}")
    print(f"  Sections:   {sections_path}")
    print(f"  Output:     {output_dir}")
    print(f"  Threshold:  >= {threshold} tphpd available")
    print(f"  Max iter:   {max_iterations}\n")

    if not sections_path.exists():
        print(f"  ERROR: Sections workbook not found at {sections_path}")
        print("  Run a Study Area or Catchment Area workflow first (Phase 1-3).")
        return 1

    try:
        print("Loading Phase 3 outputs ...")
        stations_df = pd.read_excel(sections_path, sheet_name="Stations_Peak")
        segments_df = pd.read_excel(sections_path, sheet_name="Segments_Peak")
        sections_df = pd.read_excel(sections_path, sheet_name="Sections")
        print(f"  Loaded {len(stations_df)} stations, {len(segments_df)} segments, "
              f"{len(sections_df)} sections\n")

        # Phase 4 internally reads 'Stations'/'Segments' sheets with old column names.
        # Write a compatible adapter workbook first.
        output_dir.mkdir(parents=True, exist_ok=True)
        prep_path = output_dir / f"capacity_{label}_prep.xlsx"
        _make_phase4_prep_workbook(sections_path, prep_path)

        interventions_catalog, enhanced_prep_path, final_sections_df = run_phase_four(
            original_sections_df=sections_df,
            original_segments_df=segments_df,
            original_stations_df=stations_df,
            prep_workbook_path=prep_path,
            output_dir=output_dir,
            network_label=label,
            threshold_tphpd=threshold,
            max_iterations=max_iterations,
        )

        if interventions_catalog:
            print("\nRegistering cap ints to cap_interventions.gpkg ...")
            new_nodes, new_segs, new_comp = cap_ints_to_spatial(
                interventions_catalog, infra_version
            )
            append_cap_intervention_rows(new_nodes, new_segs, new_comp)

            # Collect the int_ids just registered and build the derived infra version
            int_ids = []
            for gdf in (new_nodes, new_segs):
                if gdf is not None and 'int_id' in gdf.columns:
                    int_ids.extend(gdf['int_id'].dropna().unique().tolist())
            int_ids = sorted(set(int_ids))
            if int_ids:
                print(f"\nBuilding derived infra version ({len(int_ids)} cap int(s)) ...")
                derived_name = resolve_or_build(infra_version, int_ids)
                print(f"  Derived version: {derived_name}")

        # Reload enhanced stations/segments and write combined sections workbook
        enhanced_stations_df = pd.read_excel(enhanced_prep_path, sheet_name="Stations")
        enhanced_segments_df = pd.read_excel(enhanced_prep_path, sheet_name="Segments")

        enhanced_sections_path = output_dir / f"sections_{label}_enhanced.xlsx"
        with pd.ExcelWriter(enhanced_sections_path, engine="openpyxl") as writer:
            enhanced_stations_df.to_excel(writer, sheet_name="Stations", index=False)
            enhanced_segments_df.to_excel(writer, sheet_name="Segments", index=False)
            final_sections_df.to_excel(writer,    sheet_name="Sections", index=False)
        print(f"\n  Enhanced sections workbook: {enhanced_sections_path}")

        print("\nGenerating visualizations ...")
        visualize_enhanced_network(
            enhanced_prep_path=enhanced_prep_path,
            enhanced_sections_path=enhanced_sections_path,
            interventions_list=interventions_catalog,
            network_label=f"{label}_enhanced",
        )

        print(f"\n{'='*70}")
        print("ENHANCED WORKFLOW COMPLETE")
        print(f"{'='*70}")
        print(f"\n  Outputs:")
        print(f"    {enhanced_prep_path}")
        print(f"    {enhanced_sections_path}")
        print(f"    {output_dir / 'capacity_interventions.csv'}\n")
        return 0

    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        import traceback
        traceback.print_exc()
        return 1


# ---------------------------------------------------------------------------
# Interactive menu
# ---------------------------------------------------------------------------

def _select_sa_ca_modes() -> Tuple[str, Optional[float], str, Optional[float]]:
    """Prompt for SA and CA capacity modes. SA must be same or more precise than CA.

    Returns:
        (mode_sa, set_value_sa, mode_ca, set_value_ca)
    """
    print("\n  Study Area capacity mode:")
    print("    1) Dynamic   — UIC formula per section")
    print("    2) Set_Value — uniform value x track_count")
    while True:
        ch = input("  Select (1/2): ").strip()
        if ch in ("1", "2"):
            break
        print("  Enter 1 or 2.")

    if ch == "2":
        mode_sa = "Set_Value"
        sv_sa   = _pick_float("SA set value (tphpd/track)", default=settings.CAPACITY_SET_VALUE)
        # CA must also be Set_Value when SA is Set_Value
        mode_ca = "Set_Value"
        sv_ca   = _pick_float("CA set value (tphpd/track)", default=sv_sa)
    else:
        mode_sa = "Dynamic"
        sv_sa   = None
        print("\n  Catchment Area capacity mode (SA=Dynamic, so CA can be Dynamic or Set_Value):")
        print("    1) Dynamic   — same as SA (single pass for all sections)")
        print("    2) Set_Value — cheaper for non-SA sections")
        while True:
            ch2 = input("  Select (1/2): ").strip()
            if ch2 in ("1", "2"):
                break
            print("  Enter 1 or 2.")
        if ch2 == "2":
            mode_ca = "Set_Value"
            sv_ca   = _pick_float("CA set value (tphpd/track)", default=settings.CAPACITY_SET_VALUE)
        else:
            mode_ca = "Dynamic"
            sv_ca   = None

    return mode_sa, sv_sa, mode_ca, sv_ca


def main_interactive() -> int:
    """Interactive workflow selection with Phase 0 version picker."""
    print("\n" + "=" * 70)
    print("RAIL NETWORK CAPACITY ANALYSIS")
    print("=" * 70)
    print("\n  Available Workflows:")
    print("    1) Study Area     — capacity for the study area (SA-filtered data)")
    print("    2) Catchment Area — capacity for the full catchment (SA+CA methods)")
    print("    3) Development    — development network capacity run")
    print("    4) Enhanced       — Phase 4 capacity interventions")
    print("    0) Exit")
    print("=" * 70)

    while True:
        choice = input("\n  Select workflow (0-4): ").strip()

        if choice == "0":
            print("  Exiting.")
            return 0

        elif choice == "1":
            result = _select_versions()
            if result is None:
                continue
            infra_version, svc_version = result
            cap_mode, cap_val = _select_capacity_mode()
            grp_strat = _select_grouping_strategy() if cap_mode == "Dynamic" else None
            vis = input("  Generate plots? (y/n) [y]: ").strip().lower() != "n"
            return run_study_area_workflow(
                infra_version, svc_version,
                capacity_mode=cap_mode, set_value=cap_val,
                grouping_strategy=grp_strat, visualize=vis,
            )

        elif choice == "2":
            result = _select_versions()
            if result is None:
                continue
            infra_version, svc_version = result
            mode_sa, sv_sa, mode_ca, sv_ca = _select_sa_ca_modes()
            grp_strat = _select_grouping_strategy() if "Dynamic" in (mode_sa, mode_ca) else None
            vis = input("  Generate plots? (y/n) [y]: ").strip().lower() != "n"
            return run_catchment_area_workflow(
                infra_version, svc_version,
                mode_sa=mode_sa, mode_ca=mode_ca,
                set_value_sa=sv_sa, set_value_ca=sv_ca,
                grouping_strategy=grp_strat, visualize=vis,
            )

        elif choice == "3":
            dev_id = input("\n  Development ID (e.g. 101032): ").strip()
            if not dev_id:
                print("  Development ID is required.")
                continue
            result = _select_versions()
            if result is None:
                continue
            infra_version, svc_version = result
            cap_mode, cap_val = _select_capacity_mode()
            grp_strat = _select_grouping_strategy() if cap_mode == "Dynamic" else None
            vis = input("  Generate plots? (y/n) [y]: ").strip().lower() != "n"
            return run_development_workflow(
                dev_id, infra_version, svc_version,
                capacity_mode=cap_mode, set_value=cap_val,
                grouping_strategy=grp_strat, visualize=vis,
            )

        elif choice == "4":
            result = _select_versions(require_existing_sections=True)
            if result is None:
                continue
            infra_version, svc_version = result
            threshold_raw = input("  Minimum available capacity threshold (tphpd) [2.0]: ").strip()
            try:
                threshold = float(threshold_raw) if threshold_raw else 2.0
            except ValueError:
                print("  Invalid value, using 2.0.")
                threshold = 2.0
            max_iter_raw = input("  Maximum iterations [10]: ").strip()
            try:
                max_iterations = int(max_iter_raw) if max_iter_raw else 10
            except ValueError:
                print("  Invalid value, using 10.")
                max_iterations = 10
            return run_enhanced_workflow(
                infra_version, svc_version,
                threshold=threshold, max_iterations=max_iterations,
            )

        else:
            print("  Invalid choice. Enter 0-4.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point — CLI mode when arguments are provided, interactive otherwise."""
    if len(sys.argv) <= 1:
        sys.exit(main_interactive())

    parser = argparse.ArgumentParser(
        description="Rail network capacity analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python capacity_workflow_wrapper.py study-area "
            "--infra AS_2026_ZH_enhanced --svc AK_2026\n"
            "  python capacity_workflow_wrapper.py catchment-area "
            "--infra AS_2026_ZH_enhanced --svc AK_2026 --mode-sa Dynamic --mode-ca Set_Value\n"
            "  python capacity_workflow_wrapper.py development "
            "--dev-id 101032 --infra AS_2026_ZH_enhanced --svc AK_2026\n"
            "  python capacity_workflow_wrapper.py enhanced "
            "--infra AS_2026_ZH_enhanced --svc AK_2026"
        ),
    )
    subparsers = parser.add_subparsers(dest="workflow", required=True)

    def _add_version_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--infra", required=True, help="Infrastructure version name")
        p.add_argument("--svc",   required=True, help="Service version name")

    _GRP_CHOICES = ["manual", "conservative", "baseline", "optimal"]
    _GRP_DEFAULT = getattr(settings, "CAPACITY_GROUPING_STRATEGY", "manual")

    def _add_single_cap_args(p: argparse.ArgumentParser, default_mode_attr: str) -> None:
        p.add_argument("--capacity-mode",
                       choices=["Dynamic", "Set_Value"],
                       default=getattr(settings, default_mode_attr, "Dynamic"),
                       help="Capacity calculation mode")
        p.add_argument("--set-value", type=float, default=settings.CAPACITY_SET_VALUE,
                       help="Fixed tphpd/track for Set_Value mode")
        p.add_argument("--grouping-strategy", choices=_GRP_CHOICES, default=_GRP_DEFAULT,
                       help="Strategy for ambiguous 2-track groupings (Dynamic mode only)")
        p.add_argument("--no-plot", action="store_true", help="Skip plot generation")

    # study-area
    p_sa = subparsers.add_parser("study-area", help="Capacity for the study area (SA-filtered)")
    _add_version_args(p_sa)
    _add_single_cap_args(p_sa, "CAPACITY_MODE_SA")

    # catchment-area
    p_ca = subparsers.add_parser("catchment-area", help="Capacity for the catchment area")
    _add_version_args(p_ca)
    p_ca.add_argument("--mode-sa", choices=["Dynamic", "Set_Value"],
                      default=settings.CAPACITY_MODE_SA,
                      help="Capacity method for study-area sections")
    p_ca.add_argument("--mode-ca", choices=["Dynamic", "Set_Value"],
                      default=settings.CAPACITY_MODE_CA,
                      help="Capacity method for catchment-area sections outside the SA")
    p_ca.add_argument("--set-value-sa", type=float, default=settings.CAPACITY_SET_VALUE,
                      help="Fixed tphpd/track for SA Set_Value mode")
    p_ca.add_argument("--set-value-ca", type=float, default=settings.CAPACITY_SET_VALUE,
                      help="Fixed tphpd/track for CA Set_Value mode")
    p_ca.add_argument("--grouping-strategy", choices=_GRP_CHOICES, default=_GRP_DEFAULT,
                      help="Strategy for ambiguous 2-track groupings (Dynamic mode only)")
    p_ca.add_argument("--no-plot", action="store_true", help="Skip plot generation")

    # development
    p_dev = subparsers.add_parser("development", help="Development network capacity run")
    p_dev.add_argument("--dev-id", required=True, help="Development identifier")
    _add_version_args(p_dev)
    _add_single_cap_args(p_dev, "CAPACITY_MODE_SA")

    # enhanced
    p_enh = subparsers.add_parser("enhanced", help="Phase 4 capacity interventions")
    _add_version_args(p_enh)
    p_enh.add_argument("--threshold", type=float, default=2.0,
                       help="Min available capacity in tphpd (default 2.0)")
    p_enh.add_argument("--max-iterations", type=int, default=10,
                       help="Maximum intervention iterations (default 10)")

    args = parser.parse_args()

    if args.workflow == "study-area":
        code = run_study_area_workflow(
            args.infra, args.svc,
            capacity_mode=args.capacity_mode, set_value=args.set_value,
            grouping_strategy=args.grouping_strategy,
            visualize=not args.no_plot,
        )
    elif args.workflow == "catchment-area":
        code = run_catchment_area_workflow(
            args.infra, args.svc,
            mode_sa=args.mode_sa, mode_ca=args.mode_ca,
            set_value_sa=args.set_value_sa, set_value_ca=args.set_value_ca,
            grouping_strategy=args.grouping_strategy,
            visualize=not args.no_plot,
        )
    elif args.workflow == "development":
        code = run_development_workflow(
            args.dev_id, args.infra, args.svc,
            capacity_mode=args.capacity_mode, set_value=args.set_value,
            grouping_strategy=args.grouping_strategy,
            visualize=not args.no_plot,
        )
    elif args.workflow == "enhanced":
        code = run_enhanced_workflow(
            args.infra, args.svc,
            threshold=args.threshold, max_iterations=args.max_iterations,
        )
    else:
        code = 1

    sys.exit(code)


if __name__ == "__main__":
    main()
