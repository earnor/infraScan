"""Utility script to regenerate the processed rail network layers only.

Usage:
    python build_network.py [--use-cache]

By default the script rebuilds the processed network based on the current
settings (`settings.rail_network`) and overwrites the GeoPackage outputs under
`data/Network/processed`. Passing `--use-cache` skips regeneration when cached
files already exist.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import geopandas as gpd

from main import import_process_network
import paths

PROCESSED_DIR = Path(paths.MAIN) / "data" / "Network" / "processed"


def build_processed_network(use_cache: bool) -> gpd.GeoDataFrame:
    """Run the network preprocessing pipeline and return processed points."""
    original_cwd = Path.cwd()
    try:
        os.chdir(paths.MAIN)
        points = import_process_network(use_cache=use_cache)
    finally:
        os.chdir(original_cwd)

    if isinstance(points, str):
        # When caching is enabled in main.py the function may return a path;
        # ensure the caller always receives a GeoDataFrame.
        points = gpd.read_file(points)
    return points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate processed rail network files.")
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Skip regeneration when cached processed files already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = build_processed_network(use_cache=args.use_cache)
    outputs = [
        PROCESSED_DIR / "points.gpkg",
        PROCESSED_DIR / "edges.gpkg",
        PROCESSED_DIR / "points_with_attribute.gpkg",
        PROCESSED_DIR / "edges_with_attribute.gpkg",
        PROCESSED_DIR / "points_corridor.gpkg",
        PROCESSED_DIR / "edges_in_corridor.gpkg",
        PROCESSED_DIR / "edges_on_corridor_border.gpkg",
    ]
    print("Processed network regenerated. Key outputs:")
    for path in outputs:
        status = "✔" if path.exists() else "✖"
        print(f"  {status} {path}")
    print(f"Loaded {len(points)} processed points.")


if __name__ == "__main__":
    main()
