"""
Sensitivity Analysis Visualization Module for InfraScanRail

This module compares three pricing/grouping scenarios:
1. Base: Old6_GenCosts_Conservative (high costs, conservative grouping)
2. Scenario 2: Old6_SBBCosts_Conservative (reduced costs, conservative grouping)
3. Scenario 3: Old6_GenCosts_Optimal (high costs, optimal grouping)

Generates:
- Individual scenario plots (top 5/10 developments)
  - CBA boxplots
  - Net benefit boxplots
  - Cost savings bar charts
  - Combined plots (chart + network map) for: boxplot_cba, boxplot_net_benefit, cost_savings
- Comparative plots (3-scenario side-by-side comparison)
- Summary statistics CSV

Usage:
    python sensitivity_analysis.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import geopandas as gpd
from pathlib import Path
import plot_parameter as pp
import networkx as nx
import importlib
from PIL import Image
import settings

# Import network plotting function from plots module
from plots import plot_railway_lines_only

# ============================================================================
# CONFIGURATION
# ============================================================================

SCENARIOS = {
    'base': {
        'name': 'Old6_GenCosts_Conservative',
        'label': 'Base\n(High Costs, Conservative)',
        'label_short': 'Base',
        'color': '#3498db',  # Blue
        'hatch': '////',     # Forward diagonal
        'alpha': 0.85,       # Alpha for individual plots
        'alpha_comparative': 1.0  # No transparency in comparative plots (base)
    },
    'sbb_costs': {
        'name': 'Old6_SBBCosts_Conservative',
        'label': 'Reduced Construction Costs\n(Low Costs, Conservative)',
        'label_short': 'Reduced Construction Costs',
        'color': '#e74c3c',  # Red
        'hatch': '\\\\\\\\',    # Backward diagonal
        'alpha': 0.85,
        'alpha_comparative': 0.8  # 20% transparency in comparative plots
    },
    'optimal': {
        'name': 'Old6_GenCosts_Optimal',
        'label': 'Optimal Grouping\n(High Costs, Optimal)',
        'label_short': 'Optimal Grouping',
        'color': '#2ecc71',  # Green
        'hatch': '....',     # Dots
        'alpha': 0.85,
        'alpha_comparative': 0.6  # 40% transparency in comparative plots
    }
}

SCENARIO_ORDER = ['base', 'sbb_costs', 'optimal']

# Cost component colors (matching plots.py)
KOSTEN_FARBEN = {
    'TotalConstructionCost': '#a6bddb',
    'TotalMaintenanceCost': '#3690c0',
    'TotalUncoveredOperatingCost': '#034e7b',
    'monetized_savings_total': '#31a354'
}

# ============================================================================
# PATH HELPER FUNCTIONS
# ============================================================================

def get_scenario_input_path(scenario_name, filename='total_costs_raw.csv'):
    """Get input data path for a scenario."""
    # Get absolute path from infraScanRail directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'data', 'costs', scenario_name, filename)

def get_scenario_output_dir(scenario_name):
    """Get output directory for individual scenario plots."""
    # Get absolute path from infraScanRail directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'plots', 'sensitivity_analysis',
                              'individual_scenarios', scenario_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_comparative_output_dir():
    """Get output directory for comparative plots."""
    # Get absolute path from infraScanRail directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'plots', 'sensitivity_analysis', 'comparative')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_scenario_data(scenario_key):
    """
    Load and prepare data for a single scenario.

    Args:
        scenario_key: Key from SCENARIOS dict ('base', 'sbb_costs', or 'optimal')

    Returns:
        DataFrame with processed scenario data
    """
    scenario_name = SCENARIOS[scenario_key]['name']
    file_path = get_scenario_input_path(scenario_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Data file not found: {file_path}\n"
            f"Please ensure the pipeline has been run for scenario: {scenario_name}"
        )

    print(f"  Loading {scenario_name}: {file_path}")

    df = pd.read_csv(file_path)

    # Rename columns to match expected format
    if 'ID_new' in df.columns:
        df.rename(columns={'ID_new': 'scenario'}, inplace=True)

    # Calculate derived metrics
    df['monetized_savings_total'] = df['monetized_savings_total'].abs()
    df['total_costs'] = (df['TotalConstructionCost'] +
                         df['TotalMaintenanceCost'] +
                         df['TotalUncoveredOperatingCost'])
    df['total_net_benefit'] = df['monetized_savings_total'] - df['total_costs']
    df['cba_ratio'] = df['monetized_savings_total'] / df['total_costs']

    # Add scenario identifier
    df['pricing_scenario'] = scenario_key

    # Load Sline data for line_name generation from combined network
    try:
        # Get absolute path from infraScanRail directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        combined_network_path = os.path.join(base_dir, "data", "Network", "processed", "combined_network_with_all_modifications.gpkg")

        sline_data = gpd.read_file(combined_network_path)[['dev_id', 'Sline']]
        sline_data = sline_data.groupby('dev_id')['Sline'].first().reset_index()
        df = df.merge(sline_data, left_on='development', right_on='dev_id', how='left')
        if 'dev_id' in df.columns:
            df = df.drop(columns=['dev_id'])
    except Exception as e:
        print(f"    ⚠ Warning: Could not load Sline data: {e}")

    # Generate line_name
    df['line_name'] = None

    # Small developments (100xxx): format as "N_Sline"
    if 'Sline' in df.columns:
        df.loc[df['development'] < 101000, 'line_name'] = \
            df.loc[df['development'] < 101000].apply(
                lambda row: f"{int(row['development'] - 100000)}_{row['Sline']}"
                if pd.notna(row['Sline']) else str(int(row['development'] - 100000)),
                axis=1)
    else:
        df.loc[df['development'] < 101000, 'line_name'] = \
            df['development'].loc[df['development'] < 101000].map(
                lambda x: str(int(x - 100000)))

    # Large developments (101xxx): format as "XN"
    df.loc[df['development'] >= 101000, 'line_name'] = \
        df['development'].loc[df['development'] >= 101000].map(
            lambda x: f"X{int(x - 101000)}")

    return df

def load_all_scenarios():
    """
    Load data for all three scenarios.

    Returns:
        dict: Dictionary with scenario keys as keys and DataFrames as values
    """
    print("\n" + "="*80)
    print("LOADING SCENARIO DATA")
    print("="*80 + "\n")

    scenario_data = {}

    for scenario_key in SCENARIO_ORDER:
        try:
            scenario_data[scenario_key] = load_scenario_data(scenario_key)
            n_devs = scenario_data[scenario_key]['development'].nunique()
            n_scenarios = scenario_data[scenario_key]['scenario'].nunique()
            print(f"    ✓ {SCENARIOS[scenario_key]['name']}: "
                  f"{n_devs} developments, {n_scenarios} scenarios")
        except FileNotFoundError as e:
            print(f"    ✗ Error loading {SCENARIOS[scenario_key]['name']}: {e}")
            raise

    print("\n" + "="*80 + "\n")

    return scenario_data

# ============================================================================
# NETWORK VISUALIZATION HELPER FUNCTIONS
# ============================================================================

def load_network_graph():
    """
    Load network graph (G) and node positions (pos) for visualization.

    Returns:
        tuple: (G, pos) where G is NetworkX graph and pos is dict of node positions
    """
    # Get absolute path from infraScanRail directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    network_path = os.path.join(base_dir, settings.infra_generation_rail_network)
    points_path = os.path.join(base_dir, r'data\Network\processed\points.gpkg')

    # Check if files exist
    if not os.path.exists(network_path):
        raise FileNotFoundError(f"Network file not found: {network_path}")
    if not os.path.exists(points_path):
        raise FileNotFoundError(f"Points file not found: {points_path}")

    df_network = gpd.read_file(network_path)
    df_points = gpd.read_file(points_path)

    generate_infrastructure = importlib.import_module('generate_infrastructure')

    # Ensure the output directory exists for the pickle file
    # prepare_Graph tries to save a pickle file, so we need the directory to exist
    pickle_dir = os.path.join(base_dir, 'data', 'Network', 'processed')
    os.makedirs(pickle_dir, exist_ok=True)

    # Prepare graph - this builds the graph from network and points data
    # We need to change to base_dir because prepare_Graph uses relative paths
    original_dir = os.getcwd()
    try:
        os.chdir(base_dir)
        G, pos = generate_infrastructure.prepare_Graph(df_network, df_points)
    finally:
        # Always restore the original directory
        os.chdir(original_dir)

    return G, pos

def build_railway_line_representations(top_dev_ids, df_combined, color_map):
    """
    Build railway line representations for visualization.

    Uses combined_network_with_all_modifications.gpkg for all developments
    (both expand 100xxx and new direct connections 101xxx).

    Args:
        top_dev_ids: List of development IDs to visualize
        df_combined: Combined DataFrame with development data
        color_map: Dictionary mapping development IDs to colors

    Returns:
        list: List of railway line dictionaries ready for plot_railway_lines_only
    """
    railway_lines_dict = []

    # Get absolute path from infraScanRail directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    combined_network_path = os.path.join(base_dir, "data", "Network", "processed", "combined_network_with_all_modifications.gpkg")

    # Load combined network for ALL developments (100xxx and 101xxx)
    combined_network = gpd.read_file(combined_network_path)

    for dev_id in top_dev_ids:
        dev_data = df_combined[df_combined['development'] == dev_id].iloc[0]
        line_name = dev_data['line_name']
        color = color_map[dev_id]
        sline = dev_data.get('Sline', '')

        # Filter links for this development using dev_id and Sline
        dev_links = combined_network[
            (combined_network['dev_id'] == dev_id) &
            (combined_network['Sline'] == sline)
        ]

        if not dev_links.empty:
            # Extract nodes from FromNode and ToNode to build path
            nodes = []
            for _, link in dev_links.iterrows():
                if link['FromNode'] not in nodes:
                    nodes.append(link['FromNode'])
                if link['ToNode'] not in nodes:
                    nodes.append(link['ToNode'])

            # Get unique start and end stations
            # Use the first and last unique stations from the path
            all_stations = []
            for _, link in dev_links.iterrows():
                if link['FromStation'] not in all_stations:
                    all_stations.append(link['FromStation'])
                if link['ToStation'] not in all_stations:
                    all_stations.append(link['ToStation'])

            # First and last unique stations
            start_station = all_stations[0] if len(all_stations) > 0 else 'Unknown'
            end_station = all_stations[-1] if len(all_stations) > 1 else all_stations[0] if len(all_stations) > 0 else 'Unknown'

            railway_lines_dict.append({
                'name': line_name,
                'start_station': start_station,
                'end_station': end_station,
                'path': [str(n) for n in nodes],  # Convert to string list
                'color': color
            })
        else:
            print(f"    ⚠ Warning: No links found for development {dev_id} (line_name: {line_name}, Sline: {sline})")

    return railway_lines_dict


def combine_chart_and_map_images(chart_path, map_path, output_path):
    """
    Combine chart and network map images side by side.

    Args:
        chart_path: Path to chart image
        map_path: Path to network map image
        output_path: Path to save combined image
    """
    try:
        if not os.path.exists(chart_path):
            print(f"       ⚠ Chart not found: {os.path.basename(chart_path)}")
            return False
        if not os.path.exists(map_path):
            print(f"       ⚠ Map not found: {os.path.basename(map_path)}")
            return False

        map_image = Image.open(map_path)
        chart_image = Image.open(chart_path)

        # Resize to same height
        target_height = max(map_image.height, chart_image.height)

        def resize_to_height(img, target_h):
            w, h = img.size
            new_w = int(w * (target_h / h))
            return img.resize((new_w, target_h), Image.LANCZOS)

        map_image_resized = resize_to_height(map_image, target_height)
        chart_image_resized = resize_to_height(chart_image, target_height)

        # Combine images horizontally (map on left, chart on right)
        total_width = map_image_resized.width + chart_image_resized.width
        combined = Image.new("RGB", (total_width, target_height), (255, 255, 255))
        combined.paste(map_image_resized, (0, 0))
        combined.paste(chart_image_resized, (map_image_resized.width, 0))
        combined.save(output_path)

        print(f"       ✓ Saved: {os.path.basename(output_path)}")
        return True
    except Exception as e:
        print(f"       ⚠ Error combining images: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# INDIVIDUAL SCENARIO PLOTS (Part a)
# ============================================================================

def plot_scenario_topN_boxplot_cba(scenario_data, scenario_key, N, zvv_colors):
    """Generate CBA boxplot for top N developments in a single scenario."""

    scenario_name = SCENARIOS[scenario_key]['name']
    output_dir = get_scenario_output_dir(scenario_name)

    df = scenario_data[scenario_key]

    # Rank by mean net benefit
    rankings = df.groupby('development')['total_net_benefit'].mean().sort_values(ascending=False)
    top_dev_ids = rankings.head(N).index.tolist()

    # Filter to top N
    df_top = df[df['development'].isin(top_dev_ids)].copy()

    # Assign colors
    color_map = {dev: zvv_colors[i % len(zvv_colors)] for i, dev in enumerate(top_dev_ids)}

    # Create figure
    fig, ax = plt.subplots(figsize=(max(7, N * 1.2), 5), dpi=300)

    # Plot boxes for each development
    for i, dev_id in enumerate(top_dev_ids):
        dev_subset = df_top[df_top['development'] == dev_id]
        line_name = dev_subset['line_name'].iloc[0]
        color = color_map[dev_id]

        bp = ax.boxplot(
            [dev_subset['cba_ratio'].values],
            positions=[i * 1.5],
            widths=0.6,
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5},
            flierprops={"markersize": 3},
            boxprops={"facecolor": color, "alpha": 0.8, "edgecolor": "black", "linewidth": 0.8},
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8}
        )

    # Set x-axis labels
    line_names = [df_top[df_top['development'] == dev]['line_name'].iloc[0] for dev in top_dev_ids]
    ax.set_xticks([i * 1.5 for i in range(len(top_dev_ids))])
    ax.set_xticklabels(line_names, rotation=90)

    # Labels and formatting
    ax.set_xlabel('Development', fontsize=12)
    ax.set_ylabel('Cost-benefit ratio', fontsize=12)
    ax.set_title(f'Top {N} Developments: CBA Distribution\n({SCENARIOS[scenario_key]["label_short"]})',
                 fontsize=13, fontweight='bold')
    ax.axhline(y=1, color='red', linestyle='-', alpha=0.5, label='Break-even (CBA = 1)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Legend
    legend_handles = [
        mlines.Line2D([0], [0], marker='o', color='black', label='Mean', markersize=5, linestyle='None'),
        mlines.Line2D([0], [0], color='red', linestyle='-', alpha=0.5, label='Break-even (CBA = 1)')
    ]
    ax.legend(handles=legend_handles, loc='best', frameon=True, fontsize=10, fancybox=True, shadow=True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"ranked_top{N}_boxplot_cba.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {output_path}")

def plot_scenario_topN_boxplot_net_benefit(scenario_data, scenario_key, N, zvv_colors):
    """Generate net benefit boxplot for top N developments in a single scenario."""

    scenario_name = SCENARIOS[scenario_key]['name']
    output_dir = get_scenario_output_dir(scenario_name)

    df = scenario_data[scenario_key]

    # Rank by mean net benefit
    rankings = df.groupby('development')['total_net_benefit'].mean().sort_values(ascending=False)
    top_dev_ids = rankings.head(N).index.tolist()

    # Filter to top N
    df_top = df[df['development'].isin(top_dev_ids)].copy()

    # Assign colors
    color_map = {dev: zvv_colors[i % len(zvv_colors)] for i, dev in enumerate(top_dev_ids)}

    # Create figure
    fig, ax = plt.subplots(figsize=(max(7, N * 1.2), 5), dpi=300)

    # Plot boxes for each development
    for i, dev_id in enumerate(top_dev_ids):
        dev_subset = df_top[df_top['development'] == dev_id]
        line_name = dev_subset['line_name'].iloc[0]
        color = color_map[dev_id]

        # Convert to millions
        net_benefits_M = dev_subset['total_net_benefit'].values / 1e6

        bp = ax.boxplot(
            [net_benefits_M],
            positions=[i * 1.5],
            widths=0.6,
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5},
            flierprops={"markersize": 3},
            boxprops={"facecolor": color, "alpha": 0.8, "edgecolor": "black", "linewidth": 0.8},
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8}
        )

    # Set x-axis labels
    line_names = [df_top[df_top['development'] == dev]['line_name'].iloc[0] for dev in top_dev_ids]
    ax.set_xticks([i * 1.5 for i in range(len(top_dev_ids))])
    ax.set_xticklabels(line_names, rotation=90)

    # Labels and formatting
    ax.set_xlabel('Development', fontsize=12)
    ax.set_ylabel('Net benefit in CHF million', fontsize=12)
    ax.set_title(f'Top {N} Developments: Net Benefit Distribution\n({SCENARIOS[scenario_key]["label_short"]})',
                 fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Legend
    legend_handles = [
        mlines.Line2D([0], [0], marker='o', color='black', label='Mean', markersize=5, linestyle='None')
    ]
    ax.legend(handles=legend_handles, loc='best', frameon=True, fontsize=10, fancybox=True, shadow=True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"ranked_top{N}_boxplot_net_benefit.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {output_path}")

def plot_scenario_topN_cost_savings(scenario_data, scenario_key, N, zvv_colors):
    """Generate cost savings comparison for top N developments in a single scenario."""

    scenario_name = SCENARIOS[scenario_key]['name']
    output_dir = get_scenario_output_dir(scenario_name)

    df = scenario_data[scenario_key]

    # Rank by mean net benefit
    rankings = df.groupby('development')['total_net_benefit'].mean().sort_values(ascending=False)
    top_dev_ids = rankings.head(N).index.tolist()

    # Filter to top N and calculate summary
    df_top = df[df['development'].isin(top_dev_ids)].copy()
    summary_df = df_top.groupby('development').agg({
        'TotalConstructionCost': 'mean',
        'TotalMaintenanceCost': 'mean',
        'TotalUncoveredOperatingCost': 'mean',
        'monetized_savings_total': 'mean',
        'line_name': 'first'
    }).reset_index()

    # Assign colors
    color_map = {dev: zvv_colors[i % len(zvv_colors)] for i, dev in enumerate(top_dev_ids)}

    # Get cost colors
    kosten_farben = KOSTEN_FARBEN

    # Create figure
    fig, ax = plt.subplots(figsize=(max(7, N * 1.2), 6), dpi=300)

    bar_width = 0.6

    # Plot stacked cost bars
    for i, dev_id in enumerate(top_dev_ids):
        row = summary_df[summary_df['development'] == dev_id].iloc[0]
        x_pos = i * 2.0

        # Construction cost (bottom layer)
        ax.bar(x_pos, -row['TotalConstructionCost'] / 1e6, width=bar_width,
               color=kosten_farben['TotalConstructionCost'], edgecolor='black', linewidth=0.5)

        # Maintenance cost (middle layer)
        ax.bar(x_pos, -row['TotalMaintenanceCost'] / 1e6, width=bar_width,
               bottom=-row['TotalConstructionCost'] / 1e6,
               color=kosten_farben['TotalMaintenanceCost'], edgecolor='black', linewidth=0.5)

        # Operating cost (top layer)
        bottom_val = -(row['TotalConstructionCost'] + row['TotalMaintenanceCost']) / 1e6
        ax.bar(x_pos, -row['TotalUncoveredOperatingCost'] / 1e6, width=bar_width,
               bottom=bottom_val,
               color=kosten_farben['TotalUncoveredOperatingCost'], edgecolor='black', linewidth=0.5)

        # Savings (positive)
        ax.bar(x_pos, row['monetized_savings_total'] / 1e6, width=bar_width,
               color=color_map[dev_id], hatch='////', edgecolor='black', linewidth=0.5, alpha=0.85)

    # Set x-axis labels
    line_names = [summary_df[summary_df['development'] == dev]['line_name'].iloc[0] for dev in top_dev_ids]
    ax.set_xticks([i * 2.0 for i in range(len(top_dev_ids))])
    ax.set_xticklabels(line_names, rotation=90)

    # Labels and formatting
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Development', fontsize=12)
    ax.set_ylabel('Value in CHF million', fontsize=12)
    ax.set_title(f'Top {N} Developments: Costs and Benefits\n({SCENARIOS[scenario_key]["label_short"]})',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Legend
    legend_handles = [
        mpatches.Patch(color=kosten_farben['TotalConstructionCost'], label='Construction costs'),
        mpatches.Patch(color=kosten_farben['TotalMaintenanceCost'], label='Uncovered maintenance costs'),
        mpatches.Patch(color=kosten_farben['TotalUncoveredOperatingCost'], label='Uncovered operating costs'),
        mpatches.Patch(facecolor='gray', hatch='////', edgecolor='black', alpha=0.85, label='Travel time savings')
    ]
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.01, 1),
             frameon=True, fontsize=10, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    output_path = os.path.join(output_dir, f"ranked_top{N}_cost_savings.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {output_path}")

def plot_scenario_topN_combined_maps(scenario_data, scenario_key, N, zvv_colors, G, pos):
    """
    Generate combined plots (chart + network map) for top N developments.

    Creates combined visualizations for:
    - boxplot_cba
    - boxplot_net_benefit
    - cost_savings

    Args:
        scenario_data: Dictionary with scenario DataFrames
        scenario_key: Scenario key ('base', 'sbb_costs', or 'optimal')
        N: Number of top developments to plot
        zvv_colors: Color palette
        G: NetworkX graph
        pos: Node positions dictionary
    """
    scenario_name = SCENARIOS[scenario_key]['name']
    output_dir = get_scenario_output_dir(scenario_name)

    df = scenario_data[scenario_key]

    # Rank by mean net benefit
    rankings = df.groupby('development')['total_net_benefit'].mean().sort_values(ascending=False)
    top_dev_ids = rankings.head(N).index.tolist()

    # Filter to top N
    df_top = df[df['development'].isin(top_dev_ids)].copy()

    # Assign colors
    color_map = {dev: zvv_colors[i % len(zvv_colors)] for i, dev in enumerate(top_dev_ids)}

    # Build railway line representations
    print(f"    → Building railway line representations for {len(top_dev_ids)} developments...")
    railway_lines_dict = build_railway_line_representations(top_dev_ids, df_top, color_map)

    if not railway_lines_dict:
        print(f"    ⚠ No railway lines found for top {N} developments in {scenario_name}")
        return

    print(f"    ✓ Built {len(railway_lines_dict)} railway line representations")

    # Generate network map
    map_filename = f"railway_lines_top{N}_{scenario_name}.png"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    map_path = os.path.join(base_dir, "plots", map_filename)

    try:
        plot_railway_lines_only(
            G, pos, railway_lines_dict, map_path,
            color_dict=color_map, selected_stations=pp.selected_stations
        )
        print(f"    ✓ Network map saved: {map_filename}")
    except Exception as e:
        print(f"    ✗ Error generating network map: {e}")
        return

    # Combine images for each chart type
    chart_types = [
        "boxplot_cba",
        "boxplot_net_benefit",
        "cost_savings"
    ]

    print(f"    → Combining charts with network map...")
    for chart_type in chart_types:
        chart_path = os.path.join(output_dir, f"ranked_top{N}_{chart_type}.png")
        combined_path = os.path.join(output_dir, f"ranked_top{N}_{chart_type}_combined.png")
        combine_chart_and_map_images(chart_path, map_path, combined_path)

def generate_individual_scenario_plots(scenario_data, top_n_list=[5, 10]):
    """Generate all individual scenario plots for each scenario."""

    print("\n" + "="*80)
    print("GENERATING INDIVIDUAL SCENARIO PLOTS")
    print("="*80 + "\n")

    zvv_colors = pp.zvv_colors

    # Load network graph once for all scenarios
    print("  Loading network graph for map visualization...")
    try:
        G, pos = load_network_graph()
        print("  ✓ Network graph loaded successfully\n")
        network_available = True
    except Exception as e:
        print(f"  ⚠ Warning: Could not load network graph: {e}")
        print(f"  → Combined map plots will be skipped\n")
        network_available = False

    for scenario_key in SCENARIO_ORDER:
        print(f"\n  {'='*70}")
        print(f"  Scenario: {SCENARIOS[scenario_key]['name']}")
        print(f"  {'='*70}")

        for N in top_n_list:
            print(f"\n  Generating plots for TOP {N} developments...")

            plot_scenario_topN_boxplot_cba(scenario_data, scenario_key, N, zvv_colors)
            plot_scenario_topN_boxplot_net_benefit(scenario_data, scenario_key, N, zvv_colors)
            plot_scenario_topN_cost_savings(scenario_data, scenario_key, N, zvv_colors)

            # Generate combined plots with network maps
            if network_available:
                print(f"\n  Generating combined plots (chart + map) for TOP {N}...")
                plot_scenario_topN_combined_maps(scenario_data, scenario_key, N, zvv_colors, G, pos)

    print(f"\n  {'='*70}")
    print(f"  ✓ All individual scenario plots complete")
    print(f"  {'='*70}\n")

# ============================================================================
# COMPARATIVE PLOTS (Part b) - 3 scenarios side-by-side
# ============================================================================

def plot_all_developments_cba_comparison_3scenarios(scenario_data, zvv_colors):
    """Generate CBA comparison plot for all developments (3 scenarios)."""

    output_dir = get_comparative_output_dir()

    # Combine all scenario data
    df_combined = pd.concat([scenario_data[key] for key in SCENARIO_ORDER], ignore_index=True)

    # Rank by base scenario mean net benefit
    base_df = scenario_data['base']
    rankings = base_df.groupby('development')['total_net_benefit'].mean().sort_values(ascending=False)
    dev_order = rankings.index.tolist()

    # Assign colors
    color_map = {dev: zvv_colors[i % len(zvv_colors)] for i, dev in enumerate(dev_order)}

    # Calculate mean CBA for each development and scenario
    summary = df_combined.groupby(['development', 'pricing_scenario'])['cba_ratio'].mean().reset_index()

    # Create figure - dynamic width
    n_devs = len(dev_order)
    fig_width = max(12, n_devs * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)

    bar_width = 0.25
    x_positions = np.arange(n_devs)

    # Prepare data for plotting
    line_names = []
    scenario_cbas = {key: [] for key in SCENARIO_ORDER}

    for dev_id in dev_order:
        dev_data = summary[summary['development'] == dev_id]
        line_name = df_combined[df_combined['development'] == dev_id]['line_name'].iloc[0]
        line_names.append(line_name)

        for scenario_key in SCENARIO_ORDER:
            scenario_cba = dev_data[dev_data['pricing_scenario'] == scenario_key]['cba_ratio'].values
            scenario_cbas[scenario_key].append(scenario_cba[0] if len(scenario_cba) > 0 else 0)

    # Plot bars for each scenario
    for j, scenario_key in enumerate(SCENARIO_ORDER):
        offset = (j - 1) * bar_width
        scenario_config = SCENARIOS[scenario_key]

        for i, dev_id in enumerate(dev_order):
            color = color_map[dev_id]

            ax.bar(x_positions[i] + offset, scenario_cbas[scenario_key][i], bar_width,
                   color=color, alpha=scenario_config['alpha_comparative'],
                   edgecolor='black', linewidth=0.5,
                   hatch=scenario_config['hatch'])

    # Formatting
    ax.set_xlabel('Development', fontsize=12)
    ax.set_ylabel('Mean Cost-Benefit Ratio', fontsize=12)
    ax.set_title('All Developments: CBA Comparison Across Scenarios', fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(line_names, rotation=90, fontsize=8)
    ax.axhline(y=1, color='red', linestyle='-', alpha=0.5, linewidth=1.5)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor='gray', hatch=SCENARIOS[key]['hatch'],
                      alpha=SCENARIOS[key]['alpha_comparative'], edgecolor='black',
                      label=SCENARIOS[key]['label_short'])
        for key in SCENARIO_ORDER
    ] + [mlines.Line2D([0], [0], color='red', linestyle='-', alpha=0.5, label='Break-even (CBA = 1)')]

    ax.legend(handles=legend_handles, loc='best', frameon=True, fontsize=10, fancybox=True, shadow=True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "all_developments_cba_comparison.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {output_path}")

def plot_all_developments_cost_change_comparison_3scenarios(scenario_data, zvv_colors):
    """Generate total cost change plot for all developments (3 scenarios vs base)."""

    output_dir = get_comparative_output_dir()

    # Combine all scenario data
    df_combined = pd.concat([scenario_data[key] for key in SCENARIO_ORDER], ignore_index=True)

    # Rank by base scenario mean net benefit
    base_df = scenario_data['base']
    rankings = base_df.groupby('development')['total_net_benefit'].mean().sort_values(ascending=False)
    dev_order = rankings.index.tolist()

    # Calculate mean total costs for each development and scenario
    summary = df_combined.groupby(['development', 'pricing_scenario'])['total_costs'].mean().reset_index()

    # Create figure - dynamic width
    n_devs = len(dev_order)
    fig_width = max(12, n_devs * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)

    bar_width = 0.25
    x_positions = np.arange(n_devs)

    # Prepare data for plotting
    line_names = []
    cost_changes_abs = {key: [] for key in SCENARIO_ORDER}

    for dev_id in dev_order:
        dev_data = summary[summary['development'] == dev_id]
        line_name = df_combined[df_combined['development'] == dev_id]['line_name'].iloc[0]
        line_names.append(line_name)

        # Get base cost for this development
        base_cost = dev_data[dev_data['pricing_scenario'] == 'base']['total_costs'].values
        base_cost = base_cost[0] if len(base_cost) > 0 else 0

        for scenario_key in SCENARIO_ORDER:
            scenario_cost = dev_data[dev_data['pricing_scenario'] == scenario_key]['total_costs'].values
            scenario_cost = scenario_cost[0] if len(scenario_cost) > 0 else 0

            # Calculate change from base
            change_abs = (scenario_cost - base_cost) / 1e6  # Convert to millions
            cost_changes_abs[scenario_key].append(change_abs)

    # Plot bars for each scenario
    for j, scenario_key in enumerate(SCENARIO_ORDER):
        offset = (j - 1) * bar_width
        scenario_config = SCENARIOS[scenario_key]

        for i in range(n_devs):
            change = cost_changes_abs[scenario_key][i]

            if change >= 0:
                # Cost increase - use red
                color = '#d62728'
            else:
                # Cost decrease - use green
                color = '#2ca02c'

            ax.bar(x_positions[i] + offset, change, bar_width,
                   color=color, alpha=scenario_config['alpha_comparative'],
                   edgecolor='black', linewidth=0.5,
                   hatch=scenario_config['hatch'])

    # Formatting
    ax.set_xlabel('Development', fontsize=12)
    ax.set_ylabel('Total Cost Change vs Base (CHF million)', fontsize=12)
    ax.set_title('All Developments: Cost Changes Relative to Base Scenario',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(line_names, rotation=90, fontsize=8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor='gray', hatch=SCENARIOS[key]['hatch'],
                      alpha=SCENARIOS[key]['alpha_comparative'], edgecolor='black',
                      label=SCENARIOS[key]['label_short'])
        for key in SCENARIO_ORDER
    ] + [
        mpatches.Patch(facecolor='#d62728', alpha=0.7, edgecolor='black', label='Cost Increase'),
        mpatches.Patch(facecolor='#2ca02c', alpha=0.7, edgecolor='black', label='Cost Decrease')
    ]

    ax.legend(handles=legend_handles, loc='best', frameon=True, fontsize=10, fancybox=True, shadow=True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "all_developments_cost_change_comparison.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {output_path}")

def plot_all_developments_viability_comparison_3scenarios(scenario_data, zvv_colors):
    """Generate scenario viability comparison plot for all developments (3 scenarios)."""

    output_dir = get_comparative_output_dir()

    # Combine all scenario data
    df_combined = pd.concat([scenario_data[key] for key in SCENARIO_ORDER], ignore_index=True)

    # Rank by base scenario mean net benefit
    base_df = scenario_data['base']
    rankings = base_df.groupby('development')['total_net_benefit'].mean().sort_values(ascending=False)
    dev_order = rankings.index.tolist()

    # Calculate viability (net benefit > 0)
    df_combined['is_viable'] = df_combined['total_net_benefit'] > 0

    # Count viable scenarios per development and pricing scenario
    viability_summary = df_combined.groupby(['development', 'pricing_scenario']).agg({
        'is_viable': 'sum',
        'scenario': 'count'
    }).reset_index()

    viability_summary.columns = ['development', 'pricing_scenario', 'viable_count', 'total_count']
    viability_summary['viable_pct'] = (viability_summary['viable_count'] /
                                       viability_summary['total_count']) * 100

    # Filter to developments with at least one viable scenario in any pricing scenario
    max_viable_per_dev = viability_summary.groupby('development')['viable_count'].max()
    devs_with_viable = max_viable_per_dev[max_viable_per_dev > 0].index.tolist()
    filtered_dev_order = [dev for dev in dev_order if dev in devs_with_viable]

    if len(filtered_dev_order) == 0:
        print(f"    ⚠ No developments with viable scenarios found - skipping viability plot")
        return

    # Assign colors
    color_map = {dev: zvv_colors[i % len(zvv_colors)] for i, dev in enumerate(filtered_dev_order)}

    # Create figure
    n_devs = len(filtered_dev_order)
    fig_width = max(12, n_devs * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)

    bar_width = 0.25
    x_positions = np.arange(n_devs)

    # Prepare data for plotting
    line_names = []
    scenario_viability_pcts = {key: [] for key in SCENARIO_ORDER}

    for dev_id in filtered_dev_order:
        dev_data = viability_summary[viability_summary['development'] == dev_id]
        line_name = df_combined[df_combined['development'] == dev_id]['line_name'].iloc[0]
        line_names.append(line_name)

        for scenario_key in SCENARIO_ORDER:
            pct = dev_data[dev_data['pricing_scenario'] == scenario_key]['viable_pct'].values
            scenario_viability_pcts[scenario_key].append(pct[0] if len(pct) > 0 else 0)

    # Plot bars for each scenario
    for j, scenario_key in enumerate(SCENARIO_ORDER):
        offset = (j - 1) * bar_width
        scenario_config = SCENARIOS[scenario_key]

        for i, dev_id in enumerate(filtered_dev_order):
            color = color_map[dev_id]

            ax.bar(x_positions[i] + offset, scenario_viability_pcts[scenario_key][i], bar_width,
                   color=color, alpha=scenario_config['alpha_comparative'],
                   edgecolor='black', linewidth=0.5,
                   hatch=scenario_config['hatch'])

    # Formatting
    ax.set_xlabel('Development', fontsize=12)
    ax.set_ylabel('Viable Scenarios (%)', fontsize=12)
    ax.set_title('All Developments: Scenario Viability Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(line_names, rotation=90, fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor='gray', hatch=SCENARIOS[key]['hatch'],
                      alpha=SCENARIOS[key]['alpha_comparative'], edgecolor='black',
                      label=SCENARIOS[key]['label_short'])
        for key in SCENARIO_ORDER
    ]

    ax.legend(handles=legend_handles, loc='best', frameon=True, fontsize=10, fancybox=True, shadow=True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "all_developments_scenario_viability_comparison.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {output_path}")
    print(f"    • Developments shown: {n_devs} (filtered from {len(dev_order)} total)")

def plot_topN_cba_comparison_3scenarios(scenario_data, N, zvv_colors):
    """Generate CBA comparison boxplot for top N developments (3 scenarios)."""

    output_dir = get_comparative_output_dir()

    # Rank by base scenario mean net benefit
    base_df = scenario_data['base']
    rankings = base_df.groupby('development')['total_net_benefit'].mean().sort_values(ascending=False)
    top_dev_ids = rankings.head(N).index.tolist()

    # Combine scenario data and filter to top N
    df_combined = pd.concat([scenario_data[key] for key in SCENARIO_ORDER], ignore_index=True)
    df_top = df_combined[df_combined['development'].isin(top_dev_ids)].copy()

    # Assign colors
    color_map = {dev: zvv_colors[i % len(zvv_colors)] for i, dev in enumerate(top_dev_ids)}

    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, N * 1.8), 5), dpi=300)

    # Plot boxes for each development and scenario
    for i, dev_id in enumerate(top_dev_ids):
        dev_subset = df_top[df_top['development'] == dev_id]
        line_name = dev_subset['line_name'].iloc[0]
        color = color_map[dev_id]

        for j, scenario_key in enumerate(SCENARIO_ORDER):
            scenario_config = SCENARIOS[scenario_key]
            scenario_subset = dev_subset[dev_subset['pricing_scenario'] == scenario_key]

            if not scenario_subset.empty:
                position = i * 4.0 + j * 0.7

                bp = ax.boxplot(
                    [scenario_subset['cba_ratio'].values],
                    positions=[position],
                    widths=0.5,
                    patch_artist=True,
                    showmeans=True,
                    meanprops={"marker": "o", "markerfacecolor": "black",
                              "markeredgecolor": "black", "markersize": 4},
                    flierprops={"markersize": 2},
                    boxprops={"facecolor": color, "alpha": scenario_config['alpha_comparative'],
                             "edgecolor": "black", "linewidth": 0.8},
                    medianprops={"color": "black", "linewidth": 1.5},
                    whiskerprops={"color": "black", "linewidth": 0.8},
                    capprops={"color": "black", "linewidth": 0.8}
                )

                # Add hatch manually
                for patch in bp['boxes']:
                    patch.set_hatch(scenario_config['hatch'])

    # Set x-axis labels
    line_names = [df_top[df_top['development'] == dev]['line_name'].iloc[0] for dev in top_dev_ids]
    ax.set_xticks([i * 4.0 + 0.7 for i in range(len(top_dev_ids))])
    ax.set_xticklabels(line_names, rotation=90)

    # Labels and formatting
    ax.set_xlabel('Development', fontsize=12)
    ax.set_ylabel('Cost-benefit ratio', fontsize=12)
    ax.set_title(f'Top {N} Developments: CBA Distribution Across Scenarios',
                 fontsize=14, fontweight='bold')
    ax.axhline(y=1, color='red', linestyle='-', alpha=0.5, label='Break-even (CBA = 1)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor='gray', hatch=SCENARIOS[key]['hatch'],
                      alpha=SCENARIOS[key]['alpha_comparative'], edgecolor='black',
                      label=SCENARIOS[key]['label_short'])
        for key in SCENARIO_ORDER
    ] + [
        mlines.Line2D([0], [0], marker='o', color='black', label='Mean',
                     markersize=4, linestyle='None'),
        mlines.Line2D([0], [0], color='red', linestyle='-', alpha=0.5,
                     label='Break-even (CBA = 1)')
    ]

    ax.legend(handles=legend_handles, loc='best', frameon=True, fontsize=10,
             fancybox=True, shadow=True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"top{N}_cba_comparison.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {output_path}")

def plot_topN_net_benefit_comparison_3scenarios(scenario_data, N, zvv_colors):
    """Generate net benefit comparison boxplot for top N developments (3 scenarios)."""

    output_dir = get_comparative_output_dir()

    # Rank by base scenario mean net benefit
    base_df = scenario_data['base']
    rankings = base_df.groupby('development')['total_net_benefit'].mean().sort_values(ascending=False)
    top_dev_ids = rankings.head(N).index.tolist()

    # Combine scenario data and filter to top N
    df_combined = pd.concat([scenario_data[key] for key in SCENARIO_ORDER], ignore_index=True)
    df_top = df_combined[df_combined['development'].isin(top_dev_ids)].copy()

    # Assign colors
    color_map = {dev: zvv_colors[i % len(zvv_colors)] for i, dev in enumerate(top_dev_ids)}

    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, N * 1.8), 5), dpi=300)

    # Plot boxes for each development and scenario
    for i, dev_id in enumerate(top_dev_ids):
        dev_subset = df_top[df_top['development'] == dev_id]
        line_name = dev_subset['line_name'].iloc[0]
        color = color_map[dev_id]

        for j, scenario_key in enumerate(SCENARIO_ORDER):
            scenario_config = SCENARIOS[scenario_key]
            scenario_subset = dev_subset[dev_subset['pricing_scenario'] == scenario_key]

            if not scenario_subset.empty:
                position = i * 4.0 + j * 0.7
                net_benefits_M = scenario_subset['total_net_benefit'].values / 1e6

                bp = ax.boxplot(
                    [net_benefits_M],
                    positions=[position],
                    widths=0.5,
                    patch_artist=True,
                    showmeans=True,
                    meanprops={"marker": "o", "markerfacecolor": "black",
                              "markeredgecolor": "black", "markersize": 4},
                    flierprops={"markersize": 2},
                    boxprops={"facecolor": color, "alpha": scenario_config['alpha_comparative'],
                             "edgecolor": "black", "linewidth": 0.8},
                    medianprops={"color": "black", "linewidth": 1.5},
                    whiskerprops={"color": "black", "linewidth": 0.8},
                    capprops={"color": "black", "linewidth": 0.8}
                )

                # Add hatch manually
                for patch in bp['boxes']:
                    patch.set_hatch(scenario_config['hatch'])

    # Set x-axis labels
    line_names = [df_top[df_top['development'] == dev]['line_name'].iloc[0] for dev in top_dev_ids]
    ax.set_xticks([i * 4.0 + 0.7 for i in range(len(top_dev_ids))])
    ax.set_xticklabels(line_names, rotation=90)

    # Labels and formatting
    ax.set_xlabel('Development', fontsize=12)
    ax.set_ylabel('Net benefit in CHF million', fontsize=12)
    ax.set_title(f'Top {N} Developments: Net Benefit Distribution Across Scenarios',
                 fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor='gray', hatch=SCENARIOS[key]['hatch'],
                      alpha=SCENARIOS[key]['alpha_comparative'], edgecolor='black',
                      label=SCENARIOS[key]['label_short'])
        for key in SCENARIO_ORDER
    ] + [
        mlines.Line2D([0], [0], marker='o', color='black', label='Mean',
                     markersize=4, linestyle='None')
    ]

    ax.legend(handles=legend_handles, loc='best', frameon=True, fontsize=10,
             fancybox=True, shadow=True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"top{N}_net_benefit_comparison.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {output_path}")

def plot_topN_cost_savings_comparison_3scenarios(scenario_data, N, zvv_colors):
    """Generate cost savings comparison for top N developments (3 scenarios)."""

    output_dir = get_comparative_output_dir()

    # Rank by base scenario mean net benefit
    base_df = scenario_data['base']
    rankings = base_df.groupby('development')['total_net_benefit'].mean().sort_values(ascending=False)
    top_dev_ids = rankings.head(N).index.tolist()

    # Combine scenario data and filter to top N
    df_combined = pd.concat([scenario_data[key] for key in SCENARIO_ORDER], ignore_index=True)
    df_top = df_combined[df_combined['development'].isin(top_dev_ids)].copy()

    # Calculate summary statistics
    summary_df = df_top.groupby(['development', 'pricing_scenario']).agg({
        'TotalConstructionCost': 'mean',
        'TotalMaintenanceCost': 'mean',
        'TotalUncoveredOperatingCost': 'mean',
        'monetized_savings_total': 'mean',
        'line_name': 'first'
    }).reset_index()

    # Assign colors
    color_map = {dev: zvv_colors[i % len(zvv_colors)] for i, dev in enumerate(top_dev_ids)}
    kosten_farben = KOSTEN_FARBEN

    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, N * 2.5), 6), dpi=300)

    bar_width = 0.5

    # Plot stacked cost and savings bars
    for i, dev_id in enumerate(top_dev_ids):
        for j, scenario_key in enumerate(SCENARIO_ORDER):
            scenario_config = SCENARIOS[scenario_key]

            # Get data for this development and scenario
            dev_scenario_data = summary_df[
                (summary_df['development'] == dev_id) &
                (summary_df['pricing_scenario'] == scenario_key)
            ]

            if dev_scenario_data.empty:
                continue

            row = dev_scenario_data.iloc[0]
            x_pos = i * 4.0 + j * 1.0

            # Construction cost (bottom layer)
            ax.bar(x_pos, -row['TotalConstructionCost'] / 1e6, width=bar_width,
                   color=kosten_farben['TotalConstructionCost'],
                   edgecolor='black', linewidth=0.5)

            # Maintenance cost (middle layer)
            ax.bar(x_pos, -row['TotalMaintenanceCost'] / 1e6, width=bar_width,
                   bottom=-row['TotalConstructionCost'] / 1e6,
                   color=kosten_farben['TotalMaintenanceCost'],
                   edgecolor='black', linewidth=0.5)

            # Operating cost (top cost layer)
            bottom_val = -(row['TotalConstructionCost'] + row['TotalMaintenanceCost']) / 1e6
            ax.bar(x_pos, -row['TotalUncoveredOperatingCost'] / 1e6, width=bar_width,
                   bottom=bottom_val,
                   color=kosten_farben['TotalUncoveredOperatingCost'],
                   edgecolor='black', linewidth=0.5)

            # Savings (positive) with scenario hatch
            ax.bar(x_pos, row['monetized_savings_total'] / 1e6, width=bar_width,
                   color=color_map[dev_id], hatch=scenario_config['hatch'],
                   edgecolor='black', linewidth=0.5, alpha=scenario_config['alpha_comparative'])

    # Set x-axis labels
    line_names = [summary_df[summary_df['development'] == dev]['line_name'].iloc[0]
                  for dev in top_dev_ids]
    ax.set_xticks([i * 4.0 + 1.0 for i in range(len(top_dev_ids))])
    ax.set_xticklabels(line_names, rotation=90)

    # Labels and formatting
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Development', fontsize=12)
    ax.set_ylabel('Value in CHF million', fontsize=12)
    ax.set_title(f'Top {N} Developments: Costs and Benefits Across Scenarios',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Legend
    legend_handles = [
        mpatches.Patch(color=kosten_farben['TotalConstructionCost'],
                      label='Construction costs'),
        mpatches.Patch(color=kosten_farben['TotalMaintenanceCost'],
                      label='Uncovered maintenance costs'),
        mpatches.Patch(color=kosten_farben['TotalUncoveredOperatingCost'],
                      label='Uncovered operating costs')
    ] + [
        mpatches.Patch(facecolor='gray', hatch=SCENARIOS[key]['hatch'],
                      edgecolor='black', alpha=SCENARIOS[key]['alpha_comparative'],
                      label=f"Savings ({SCENARIOS[key]['label_short']})")
        for key in SCENARIO_ORDER
    ]

    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.01, 1),
             frameon=True, fontsize=10, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    output_path = os.path.join(output_dir, f"top{N}_cost_savings_comparison.png")
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved: {output_path}")

def generate_comparative_plots(scenario_data, top_n_list=[5, 10]):
    """Generate all 3-scenario comparative plots."""

    print("\n" + "="*80)
    print("GENERATING COMPARATIVE PLOTS (3 SCENARIOS)")
    print("="*80 + "\n")

    zvv_colors = pp.zvv_colors

    print("  Generating all developments comparison plots...")
    plot_all_developments_cba_comparison_3scenarios(scenario_data, zvv_colors)
    plot_all_developments_cost_change_comparison_3scenarios(scenario_data, zvv_colors)
    plot_all_developments_viability_comparison_3scenarios(scenario_data, zvv_colors)

    for N in top_n_list:
        print(f"\n  Generating TOP {N} comparison plots...")
        plot_topN_cba_comparison_3scenarios(scenario_data, N, zvv_colors)
        plot_topN_net_benefit_comparison_3scenarios(scenario_data, N, zvv_colors)
        plot_topN_cost_savings_comparison_3scenarios(scenario_data, N, zvv_colors)

    print(f"\n  {'='*70}")
    print(f"  ✓ All comparative plots complete")
    print(f"  {'='*70}\n")

# ============================================================================
# SUMMARY STATISTICS EXPORT
# ============================================================================

def export_summary_statistics(scenario_data):
    """Export comprehensive summary statistics CSV comparing all scenarios."""

    print("\n" + "="*80)
    print("GENERATING SUMMARY STATISTICS CSV")
    print("="*80 + "\n")

    output_dir = get_comparative_output_dir()

    # Combine all scenario data
    df_combined = pd.concat([scenario_data[key] for key in SCENARIO_ORDER], ignore_index=True)

    # Rank by base scenario
    base_df = scenario_data['base']
    rankings = base_df.groupby('development')['total_net_benefit'].mean().sort_values(ascending=False)
    dev_order = rankings.index.tolist()

    # Calculate viability
    df_combined['is_viable'] = df_combined['total_net_benefit'] > 0

    # Build summary data
    summary_data = []

    for dev_id in dev_order:
        dev_data = df_combined[df_combined['development'] == dev_id]
        line_name = dev_data['line_name'].iloc[0]

        row = {
            'development': dev_id,
            'line_name': line_name
        }

        for scenario_key in SCENARIO_ORDER:
            scenario_subset = dev_data[dev_data['pricing_scenario'] == scenario_key]

            if len(scenario_subset) > 0:
                # CBA metrics
                mean_cba = scenario_subset['cba_ratio'].mean()

                # Cost metrics
                mean_cost = scenario_subset['total_costs'].mean()

                # Net benefit metrics
                mean_net_benefit = scenario_subset['total_net_benefit'].mean()

                # Viability metrics
                viable_count = scenario_subset['is_viable'].sum()
                total_scenarios = len(scenario_subset)
                viable_pct = (viable_count / total_scenarios * 100) if total_scenarios > 0 else 0

                scenario_label = SCENARIOS[scenario_key]['label_short']

                row[f'{scenario_label}_mean_cba'] = mean_cba
                row[f'{scenario_label}_mean_cost_chf'] = mean_cost
                row[f'{scenario_label}_mean_net_benefit_chf'] = mean_net_benefit
                row[f'{scenario_label}_viable_scenarios'] = int(viable_count)
                row[f'{scenario_label}_viable_pct'] = viable_pct
                row[f'{scenario_label}_total_scenarios'] = int(total_scenarios)
            else:
                scenario_label = SCENARIOS[scenario_key]['label_short']
                row[f'{scenario_label}_mean_cba'] = 0
                row[f'{scenario_label}_mean_cost_chf'] = 0
                row[f'{scenario_label}_mean_net_benefit_chf'] = 0
                row[f'{scenario_label}_viable_scenarios'] = 0
                row[f'{scenario_label}_viable_pct'] = 0
                row[f'{scenario_label}_total_scenarios'] = 0

        # Calculate differences from base
        base_cba = row.get('Base_mean_cba', 0)
        base_cost = row.get('Base_mean_cost_chf', 0)
        base_net_benefit = row.get('Base_mean_net_benefit_chf', 0)

        for scenario_key in ['sbb_costs', 'optimal']:
            scenario_label = SCENARIOS[scenario_key]['label_short']

            scenario_cba = row.get(f'{scenario_label}_mean_cba', 0)
            scenario_cost = row.get(f'{scenario_label}_mean_cost_chf', 0)
            scenario_net_benefit = row.get(f'{scenario_label}_mean_net_benefit_chf', 0)

            row[f'{scenario_label}_cba_change_from_base'] = scenario_cba - base_cba
            row[f'{scenario_label}_cost_change_from_base_chf'] = scenario_cost - base_cost
            row[f'{scenario_label}_cost_change_from_base_pct'] = \
                ((scenario_cost - base_cost) / base_cost * 100) if base_cost != 0 else 0
            row[f'{scenario_label}_net_benefit_change_from_base_chf'] = \
                scenario_net_benefit - base_net_benefit

        summary_data.append(row)

    # Convert to DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    output_path = os.path.join(output_dir, "sensitivity_analysis_summary_statistics.csv")
    summary_df.to_csv(output_path, index=False, float_format='%.2f')

    print(f"  ✓ Saved: {output_path}")
    print(f"  • Total developments: {len(summary_df)}")

    # Print some summary statistics
    for scenario_key in SCENARIO_ORDER:
        scenario_label = SCENARIOS[scenario_key]['label_short']
        viable_col = f'{scenario_label}_viable_scenarios'
        if viable_col in summary_df.columns:
            total_viable = (summary_df[viable_col] > 0).sum()
            print(f"  • Developments with ≥1 viable scenario ({scenario_label}): {total_viable}")

    print("\n" + "="*80 + "\n")

    return summary_df


def export_report_statistics(scenario_data):
    """
    Export comprehensive report-ready statistics for sensitivity analysis.

    Generates statistics suitable for academic/technical reports including:
    - Viability changes across scenarios
    - Cost changes (absolute and percentage)
    - BCR distribution statistics
    - Top performing developments across scenarios
    - Scenario comparison summary
    """
    print("\n" + "="*80)
    print("GENERATING REPORT STATISTICS")
    print("="*80 + "\n")

    output_dir = get_comparative_output_dir()

    # Combine all scenario data
    df_combined = pd.concat([scenario_data[key] for key in SCENARIO_ORDER], ignore_index=True)
    df_combined['is_viable'] = df_combined['total_net_benefit'] > 0

    # ===========================================================================
    # 1. VIABILITY STATISTICS
    # ===========================================================================
    viability_stats = []

    for scenario_key in SCENARIO_ORDER:
        scenario_df = scenario_data[scenario_key]
        scenario_df['is_viable'] = scenario_df['total_net_benefit'] > 0

        # Count viable scenarios per development
        viable_by_dev = scenario_df.groupby('development')['is_viable'].agg([
            ('total_scenarios', 'count'),
            ('viable_scenarios', 'sum')
        ]).reset_index()

        # Developments with at least one viable scenario
        devs_with_any_viable = (viable_by_dev['viable_scenarios'] > 0).sum()
        # Developments with all scenarios viable
        devs_all_viable = (viable_by_dev['viable_scenarios'] == viable_by_dev['total_scenarios']).sum()
        # Developments with no viable scenarios
        devs_no_viable = (viable_by_dev['viable_scenarios'] == 0).sum()

        total_devs = len(viable_by_dev)
        total_scenarios = scenario_df.shape[0]
        total_viable_scenarios = scenario_df['is_viable'].sum()

        viability_stats.append({
            'Scenario': SCENARIOS[scenario_key]['label_short'],
            'Total Developments': total_devs,
            'Developments with ≥1 Viable Scenario': devs_with_any_viable,
            '% Developments with ≥1 Viable Scenario': (devs_with_any_viable / total_devs * 100),
            'Developments with All Scenarios Viable': devs_all_viable,
            '% Developments All Viable': (devs_all_viable / total_devs * 100),
            'Developments with No Viable Scenarios': devs_no_viable,
            '% Developments No Viable': (devs_no_viable / total_devs * 100),
            'Total Scenario Runs': total_scenarios,
            'Total Viable Scenario Runs': int(total_viable_scenarios),
            '% Viable Scenario Runs': (total_viable_scenarios / total_scenarios * 100)
        })

    viability_df = pd.DataFrame(viability_stats)
    viability_path = os.path.join(output_dir, "report_viability_statistics.csv")
    viability_df.to_csv(viability_path, index=False, float_format='%.2f')
    print(f"  ✓ Saved: {viability_path}")

    # ===========================================================================
    # 2. COST STATISTICS (vs Base)
    # ===========================================================================
    cost_stats = []

    # Get base scenario costs
    base_df = scenario_data['base']
    base_costs_by_dev = base_df.groupby('development')['total_costs'].mean()

    for scenario_key in SCENARIO_ORDER:
        scenario_df = scenario_data[scenario_key]
        scenario_costs_by_dev = scenario_df.groupby('development')['total_costs'].mean()

        # Calculate changes from base
        if scenario_key == 'base':
            cost_changes = pd.Series(0, index=base_costs_by_dev.index)
            cost_changes_pct = pd.Series(0, index=base_costs_by_dev.index)
        else:
            cost_changes = scenario_costs_by_dev - base_costs_by_dev
            cost_changes_pct = (cost_changes / base_costs_by_dev * 100)

        # Statistics
        total_base_cost = base_costs_by_dev.sum()
        total_scenario_cost = scenario_costs_by_dev.sum()
        total_change = total_scenario_cost - total_base_cost
        total_change_pct = (total_change / total_base_cost * 100)

        cost_stats.append({
            'Scenario': SCENARIOS[scenario_key]['label_short'],
            'Total Costs (CHF M)': total_scenario_cost / 1e6,
            'Total Cost Change from Base (CHF M)': total_change / 1e6,
            'Total Cost Change from Base (%)': total_change_pct,
            'Mean Cost per Development (CHF M)': scenario_costs_by_dev.mean() / 1e6,
            'Mean Cost Change from Base (CHF M)': cost_changes.mean() / 1e6,
            'Mean Cost Change from Base (%)': cost_changes_pct.mean(),
            'Median Cost Change from Base (%)': cost_changes_pct.median(),
            'Min Cost Change (%)': cost_changes_pct.min(),
            'Max Cost Change (%)': cost_changes_pct.max(),
            'Std Dev Cost Change (%)': cost_changes_pct.std(),
            'Developments with Lower Costs': (cost_changes < 0).sum(),
            'Developments with Higher Costs': (cost_changes > 0).sum()
        })

    cost_df = pd.DataFrame(cost_stats)
    cost_path = os.path.join(output_dir, "report_cost_statistics.csv")
    cost_df.to_csv(cost_path, index=False, float_format='%.2f')
    print(f"  ✓ Saved: {cost_path}")

    # ===========================================================================
    # 3. BCR/CBA STATISTICS
    # ===========================================================================
    bcr_stats = []

    for scenario_key in SCENARIO_ORDER:
        scenario_df = scenario_data[scenario_key]

        # Mean CBA by development
        cba_by_dev = scenario_df.groupby('development')['cba_ratio'].mean()

        # Count developments above BCR threshold (1.0)
        devs_above_threshold = (cba_by_dev >= 1.0).sum()
        devs_below_threshold = (cba_by_dev < 1.0).sum()

        bcr_stats.append({
            'Scenario': SCENARIOS[scenario_key]['label_short'],
            'Mean CBA Ratio (all devs)': cba_by_dev.mean(),
            'Median CBA Ratio': cba_by_dev.median(),
            'Min CBA Ratio': cba_by_dev.min(),
            'Max CBA Ratio': cba_by_dev.max(),
            'Std Dev CBA Ratio': cba_by_dev.std(),
            'Developments with CBA ≥ 1.0': devs_above_threshold,
            '% Developments with CBA ≥ 1.0': (devs_above_threshold / len(cba_by_dev) * 100),
            'Developments with CBA < 1.0': devs_below_threshold,
            '% Developments with CBA < 1.0': (devs_below_threshold / len(cba_by_dev) * 100)
        })

    bcr_df = pd.DataFrame(bcr_stats)
    bcr_path = os.path.join(output_dir, "report_bcr_statistics.csv")
    bcr_df.to_csv(bcr_path, index=False, float_format='%.3f')
    print(f"  ✓ Saved: {bcr_path}")

    # ===========================================================================
    # 4. NET BENEFIT STATISTICS
    # ===========================================================================
    net_benefit_stats = []

    for scenario_key in SCENARIO_ORDER:
        scenario_df = scenario_data[scenario_key]

        # Mean net benefit by development
        net_benefit_by_dev = scenario_df.groupby('development')['total_net_benefit'].mean()

        # Positive vs negative
        devs_positive = (net_benefit_by_dev > 0).sum()
        devs_negative = (net_benefit_by_dev <= 0).sum()

        net_benefit_stats.append({
            'Scenario': SCENARIOS[scenario_key]['label_short'],
            'Total Net Benefit (CHF M)': net_benefit_by_dev.sum() / 1e6,
            'Mean Net Benefit per Dev (CHF M)': net_benefit_by_dev.mean() / 1e6,
            'Median Net Benefit (CHF M)': net_benefit_by_dev.median() / 1e6,
            'Min Net Benefit (CHF M)': net_benefit_by_dev.min() / 1e6,
            'Max Net Benefit (CHF M)': net_benefit_by_dev.max() / 1e6,
            'Std Dev Net Benefit (CHF M)': net_benefit_by_dev.std() / 1e6,
            'Developments with Positive Net Benefit': devs_positive,
            '% Positive Net Benefit': (devs_positive / len(net_benefit_by_dev) * 100),
            'Developments with Negative Net Benefit': devs_negative,
            '% Negative Net Benefit': (devs_negative / len(net_benefit_by_dev) * 100)
        })

    net_benefit_df = pd.DataFrame(net_benefit_stats)
    net_benefit_path = os.path.join(output_dir, "report_net_benefit_statistics.csv")
    net_benefit_df.to_csv(net_benefit_path, index=False, float_format='%.2f')
    print(f"  ✓ Saved: {net_benefit_path}")

    # ===========================================================================
    # 5. TOP PERFORMERS BY SCENARIO
    # ===========================================================================
    top_performers = []

    for scenario_key in SCENARIO_ORDER:
        scenario_df = scenario_data[scenario_key]

        # Rank by mean net benefit
        rankings = scenario_df.groupby('development').agg({
            'total_net_benefit': 'mean',
            'cba_ratio': 'mean',
            'total_costs': 'mean',
            'line_name': 'first'
        }).sort_values('total_net_benefit', ascending=False)

        # Top 10
        for rank, (dev_id, row) in enumerate(rankings.head(10).iterrows(), 1):
            top_performers.append({
                'Scenario': SCENARIOS[scenario_key]['label_short'],
                'Rank': rank,
                'Development': dev_id,
                'Line Name': row['line_name'],
                'Mean Net Benefit (CHF M)': row['total_net_benefit'] / 1e6,
                'Mean CBA Ratio': row['cba_ratio'],
                'Mean Total Cost (CHF M)': row['total_costs'] / 1e6
            })

    top_performers_df = pd.DataFrame(top_performers)
    top_performers_path = os.path.join(output_dir, "report_top10_performers.csv")
    top_performers_df.to_csv(top_performers_path, index=False, float_format='%.2f')
    print(f"  ✓ Saved: {top_performers_path}")

    # ===========================================================================
    # 6. SCENARIO COMPARISON SUMMARY
    # ===========================================================================
    print("\n" + "-"*80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("-"*80 + "\n")

    print("VIABILITY IMPACT:")
    for _, row in viability_df.iterrows():
        print(f"  {row['Scenario']}:")
        print(f"    - Developments with viable scenarios: {row['Developments with ≥1 Viable Scenario']}/{row['Total Developments']} ({row['% Developments with ≥1 Viable Scenario']:.1f}%)")
        print(f"    - Viable scenario runs: {row['Total Viable Scenario Runs']}/{row['Total Scenario Runs']} ({row['% Viable Scenario Runs']:.1f}%)")

    print("\nCOST IMPACT (vs Base):")
    for _, row in cost_df.iterrows():
        if row['Scenario'] != 'Base':
            print(f"  {row['Scenario']}:")
            print(f"    - Total cost change: {row['Total Cost Change from Base (CHF M)']:+,.0f} CHF M ({row['Total Cost Change from Base (%)']:+.1f}%)")
            print(f"    - Mean cost change per development: {row['Mean Cost Change from Base (CHF M)']:+,.0f} CHF M ({row['Mean Cost Change from Base (%)']:+.1f}%)")

    print("\nCBA PERFORMANCE:")
    for _, row in bcr_df.iterrows():
        print(f"  {row['Scenario']}:")
        print(f"    - Mean CBA ratio: {row['Mean CBA Ratio (all devs)']:.2f}")
        print(f"    - Developments with CBA ≥ 1.0: {row['Developments with CBA ≥ 1.0']}/{int(row['Developments with CBA ≥ 1.0'] + row['Developments with CBA < 1.0'])} ({row['% Developments with CBA ≥ 1.0']:.1f}%)")

    print("\n" + "="*80 + "\n")

    return {
        'viability': viability_df,
        'costs': cost_df,
        'bcr': bcr_df,
        'net_benefit': net_benefit_df,
        'top_performers': top_performers_df
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""

    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS: SCENARIO COMPARISON")
    print("="*80)
    print("\nComparing three pricing/grouping scenarios:")
    print(f"  1. {SCENARIOS['base']['name']} (Base)")
    print(f"  2. {SCENARIOS['sbb_costs']['name']}")
    print(f"  3. {SCENARIOS['optimal']['name']}")
    print("\n" + "="*80 + "\n")

    try:
        # Load all scenario data
        scenario_data = load_all_scenarios()

        # Generate individual scenario plots
        generate_individual_scenario_plots(scenario_data, top_n_list=[5, 10])

        # Generate comparative plots
        generate_comparative_plots(scenario_data, top_n_list=[5, 10])

        # Export summary statistics
        export_summary_statistics(scenario_data)

        # Export report statistics
        export_report_statistics(scenario_data)

        # Final summary
        print("\n" + "="*80)
        print("✓ SENSITIVITY ANALYSIS COMPLETE")
        print("="*80)
        print("\nGenerated outputs:")
        print(f"\n  Individual Scenario Plots:")
        for scenario_key in SCENARIO_ORDER:
            scenario_name = SCENARIOS[scenario_key]['name']
            output_dir = get_scenario_output_dir(scenario_name)
            print(f"    • {scenario_name}/")
            print(f"      - ranked_top5_boxplot_cba.png")
            print(f"      - ranked_top5_boxplot_net_benefit.png")
            print(f"      - ranked_top5_cost_savings.png")
            print(f"      - ranked_top5_boxplot_cba_combined.png (chart + network map)")
            print(f"      - ranked_top5_boxplot_net_benefit_combined.png (chart + network map)")
            print(f"      - ranked_top5_cost_savings_combined.png (chart + network map)")
            print(f"      - ranked_top10_boxplot_cba.png")
            print(f"      - ranked_top10_boxplot_net_benefit.png")
            print(f"      - ranked_top10_cost_savings.png")
            print(f"      - ranked_top10_boxplot_cba_combined.png (chart + network map)")
            print(f"      - ranked_top10_boxplot_net_benefit_combined.png (chart + network map)")
            print(f"      - ranked_top10_cost_savings_combined.png (chart + network map)")

        comparative_dir = get_comparative_output_dir()
        print(f"\n  Comparative Plots (3 scenarios):")
        print(f"    • all_developments_cba_comparison.png")
        print(f"    • all_developments_cost_change_comparison.png")
        print(f"    • all_developments_scenario_viability_comparison.png")
        print(f"    • top5_cba_comparison.png")
        print(f"    • top5_net_benefit_comparison.png")
        print(f"    • top5_cost_savings_comparison.png")
        print(f"    • top10_cba_comparison.png")
        print(f"    • top10_net_benefit_comparison.png")
        print(f"    • top10_cost_savings_comparison.png")

        print(f"\n  Summary Statistics (CSV):")
        print(f"    • sensitivity_analysis_summary_statistics.csv")

        print(f"\n  Report Statistics (CSV - for your thesis):")
        print(f"    • report_viability_statistics.csv")
        print(f"    • report_cost_statistics.csv")
        print(f"    • report_bcr_statistics.csv")
        print(f"    • report_net_benefit_statistics.csv")
        print(f"    • report_top10_performers.csv")

        print("\n" + "="*80 + "\n")

    except FileNotFoundError as e:
        print("\n" + "="*80)
        print("ERROR: Required data files not found")
        print("="*80)
        print(f"\n{e}")
        print("\nPlease ensure the pipeline has been run for all three scenarios:")
        for scenario_key in SCENARIO_ORDER:
            print(f"  • {SCENARIOS[scenario_key]['name']}")
        print("\n" + "="*80 + "\n")

    except Exception as e:
        print("\n" + "="*80)
        print("ERROR: Unexpected error occurred")
        print("="*80)
        print(f"\n{e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    main()
