import pandas as pd
import geopandas as gpd
import os

# --- SETTINGS ---
PROJECT_ROOT = r'/Users/ruki/PycharmProjects/infraScan/infraScanCycle'
os.chdir(PROJECT_ROOT)


def check_file(label, path, file_type='gpkg', layer=None):
    """Prints status and columns for a specific file."""
    print(f"\n{'=' * 20} {label} {'=' * 20}")
    print(f"Path: {path}")

    if not os.path.exists(path):
        print("STATUS: File not found.")
        return

    try:
        if file_type == 'gpkg':
            df = gpd.read_file(path, layer=layer)
        else:
            # Handle potential semicolon or comma separators in Swiss OGD data
            df = pd.read_csv(path, sep=None, engine='python')

        print(f"STATUS: Loaded successfully ({len(df)} rows)")
        print(f"Columns found: {df.columns.tolist()}")

        # Check for the specific column that caused the crash
        if 'cor_1' in df.columns:
            print("'cor_1' column: PRESENT")
        else:
            print("'cor_1' column: MISSING")

    except Exception as e:
        print(f"ERROR reading file: {e}")


if __name__ == "__main__":
    print(f"Starting Data Diagnostic in: {os.getcwd()}")

    # 1. Check the very first raw input
    check_file(
        "RAW CSV ACCESS POINTS",
        "data/raw/VELOPARKIERANLAGEN/OGD_VELOPARKIERANLAGEN_P.csv",
        file_type='csv'
    )

    # 2. Check the reformatted network points
    check_file(
        "PROCESSED NETWORK POINTS",
        "data/Network/processed/points.gpkg"
    )

    # 3. Check the corridor-filtered points
    check_file(
        "POINTS IN CORRIDOR",
        "data/Network/processed/points_corridor.gpkg"
    )

    # 4. Check the points with mapped attributes (Input A for connection)
    check_file(
        "CURRENT ACCESS POINTS (WITH ATTR)",
        "data/Network/processed/points_corridor.gpkg"
    )

    # 5. Check the generated potential sites (Input B for connection)
    check_file(
        "GENERATED POTENTIAL SITES",
        "data/Network/processed/generated_nodes.gpkg"
    )

    print("\n" + "=" * 50)
    print("DONE. Look for the first file where 'cor_1' says MISSING.")
    print("=" * 50)