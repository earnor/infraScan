"""
Cost visualization module for InfraScanRail capacity-enhanced pipeline.

This module provides visualization functions for Phase 10 and Phase 11:
- Infrastructure costs with capacity interventions (stacked bar charts)
- Capacity surplus percentage analysis
- Intervention details tables
- Viability assessment plots (BCR comparison)
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import cost_parameters as cp
import plot_parameter as pp


def plot_infrastructure_costs_with_capacity(infrastructure_costs, output_dir='plots/costs'):
    """
    Generate stacked bar chart showing infrastructure costs with capacity interventions.

    Parameters:
    -----------
    infrastructure_costs : dict
        Dictionary with keys: dev_id, values: dict with cost breakdown
        Required keys in value dict:
            - base_construction
            - base_maintenance_pv
            - capacity_construction
            - capacity_maintenance_pv
            - capacity_surplus_pct
    output_dir : str
        Directory to save the plot (relative to repository root)

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare data
    dev_ids = list(infrastructure_costs.keys())
    base_construction = [infrastructure_costs[dev]['base_construction'] / 1e6 for dev in dev_ids]
    base_maintenance = [infrastructure_costs[dev]['base_maintenance_pv'] / 1e6 for dev in dev_ids]
    capacity_construction = [infrastructure_costs[dev]['capacity_construction'] / 1e6 for dev in dev_ids]
    capacity_maintenance = [infrastructure_costs[dev]['capacity_maintenance_pv'] / 1e6 for dev in dev_ids]
    capacity_surplus_pct = [infrastructure_costs[dev]['capacity_surplus_pct'] for dev in dev_ids]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

    # Set positions for bars
    x = np.arange(len(dev_ids))
    width = 0.6

    # Create stacked bars
    p1 = ax.bar(x, base_construction, width, label='Base Construction',
                color='#1f77b4', edgecolor='white', linewidth=0.5)
    p2 = ax.bar(x, base_maintenance, width, bottom=base_construction,
                label='Base Maintenance (PV)', color='#aec7e8',
                edgecolor='white', linewidth=0.5)

    bottom_capacity = [bc + bm for bc, bm in zip(base_construction, base_maintenance)]
    p3 = ax.bar(x, capacity_construction, width, bottom=bottom_capacity,
                label='Capacity Construction', color='#ff7f0e',
                edgecolor='white', linewidth=0.5)

    bottom_total = [bc + bm + cc for bc, bm, cc in zip(base_construction, base_maintenance, capacity_construction)]
    p4 = ax.bar(x, capacity_maintenance, width, bottom=bottom_total,
                label='Capacity Maintenance (PV)', color='#ffbb78',
                edgecolor='white', linewidth=0.5)

    # Add capacity surplus percentage annotations above bars
    total_heights = [bc + bm + cc + cm for bc, bm, cc, cm in
                     zip(base_construction, base_maintenance, capacity_construction, capacity_maintenance)]

    for i, (height, surplus_pct) in enumerate(zip(total_heights, capacity_surplus_pct)):
        if surplus_pct > 0:
            ax.text(i, height + max(total_heights) * 0.02, f'+{surplus_pct:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold', color='#d62728')

    # Customize plot
    ax.set_xlabel('Development ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cost (CHF Millions)', fontsize=12, fontweight='bold')
    ax.set_title('Infrastructure Costs with Capacity Interventions',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(dev_ids, rotation=45, ha='right')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Format y-axis with comma separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'.replace(',', "'")))

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir) / 'infrastructure_costs_with_capacity.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    return fig, ax


def plot_capacity_surplus_by_development(infrastructure_costs, output_dir='plots/costs'):
    """
    Generate bar chart showing capacity cost surplus percentage by development.

    Parameters:
    -----------
    infrastructure_costs : dict
        Dictionary with dev_id keys and cost breakdown values
    output_dir : str
        Directory to save the plot

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare data
    dev_ids = list(infrastructure_costs.keys())
    capacity_surplus_pct = [infrastructure_costs[dev]['capacity_surplus_pct'] for dev in dev_ids]

    # Sort by surplus percentage (descending)
    sorted_data = sorted(zip(dev_ids, capacity_surplus_pct), key=lambda x: x[1], reverse=True)
    dev_ids_sorted, capacity_surplus_sorted = zip(*sorted_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # Create bars with color gradient
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(dev_ids_sorted)))
    bars = ax.bar(range(len(dev_ids_sorted)), capacity_surplus_sorted,
                  color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, capacity_surplus_sorted)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # Customize plot
    ax.set_xlabel('Development ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Capacity Cost Surplus (%)', fontsize=12, fontweight='bold')
    ax.set_title('Capacity Intervention Cost as % of Base Construction Cost',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(dev_ids_sorted)))
    ax.set_xticklabels(dev_ids_sorted, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir) / 'capacity_surplus_by_development.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    return fig, ax


def export_intervention_details(infrastructure_costs, output_dir='plots/costs'):
    """
    Export detailed intervention tables as images (one per development).

    Parameters:
    -----------
    infrastructure_costs : dict
        Dictionary with dev_id keys and cost breakdown values
        Each value must contain 'intervention_details' key with list of interventions
    output_dir : str
        Directory to save the plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for dev_id, cost_data in infrastructure_costs.items():
        interventions = cost_data.get('intervention_details', [])

        if not interventions:
            continue

        # Prepare table data
        table_data = []
        for intervention in interventions:
            row = [
                intervention.get('type', 'N/A'),
                f"{intervention.get('from_station', '')} - {intervention.get('to_station', '')}",
                f"{intervention.get('length_m', 0) / 1000:.2f} km" if 'length_m' in intervention
                    else f"{intervention.get('platforms_added', 0)} platforms",
                f"{intervention.get('cost_construction', 0) / 1e6:.1f} M",
                f"{intervention.get('cost_maintenance_annual', 0) / 1e6:.2f} M",
                intervention.get('cost_source', 'Enhanced')
            ]
            table_data.append(row)

        # Create figure for table
        fig, ax = plt.subplots(figsize=(14, max(3, len(table_data) * 0.4)), dpi=300)
        ax.axis('tight')
        ax.axis('off')

        # Create table
        col_labels = ['Intervention Type', 'Location', 'Length/Qty',
                     'Construction Cost', 'Annual Maintenance', 'Source']

        table = ax.table(cellText=table_data, colLabels=col_labels,
                        cellLoc='left', loc='center',
                        colWidths=[0.15, 0.25, 0.12, 0.15, 0.15, 0.12])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header row
        for i in range(len(col_labels)):
            cell = table[(0, i)]
            cell.set_facecolor('#2c3e50')
            cell.set_text_props(weight='bold', color='white')

        # Style data rows with alternating colors
        for i in range(1, len(table_data) + 1):
            for j in range(len(col_labels)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#ecf0f1')
                else:
                    cell.set_facecolor('white')

                # Highlight estimated costs
                if j == 5 and table_data[i-1][5] == 'estimated':
                    cell.set_facecolor('#fff3cd')

        plt.title(f'Capacity Interventions for {dev_id}',
                 fontsize=14, fontweight='bold', pad=20)

        # Save figure
        output_path = Path(output_dir) / f'intervention_details_{dev_id}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_path}")


def plot_bcr_comparison_scatter(viability_results, bcr_threshold=1.0, output_dir='plots/viability'):
    """
    Generate scatter plot comparing BCR with and without capacity interventions.

    Parameters:
    -----------
    viability_results : dict
        Dictionary with dev_id keys and viability assessment values
        Required keys: bcr_without_capacity, bcr_with_capacity,
                      viable_without_capacity, viable_with_capacity
    bcr_threshold : float
        Viability threshold (default 1.0)
    output_dir : str
        Directory to save the plot

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare data
    dev_ids = list(viability_results.keys())
    bcr_without = [viability_results[dev]['bcr_without_capacity'] for dev in dev_ids]
    bcr_with = [viability_results[dev]['bcr_with_capacity'] for dev in dev_ids]
    viable_without = [viability_results[dev]['viable_without_capacity'] for dev in dev_ids]
    viable_with = [viability_results[dev]['viable_with_capacity'] for dev in dev_ids]

    # Color coding
    colors = []
    for vw, vwo in zip(viable_with, viable_without):
        if vw and vwo:
            colors.append('green')  # Viable both
        elif not vw and vwo:
            colors.append('orange')  # Capacity kills viability
        elif not vw and not vwo:
            colors.append('red')  # Not viable either way
        else:
            colors.append('blue')  # Unusual: not viable without, viable with

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    # Scatter plot
    scatter = ax.scatter(bcr_without, bcr_with, c=colors, s=100, alpha=0.7,
                        edgecolors='black', linewidth=1)

    # Add diagonal line (y=x)
    max_bcr = max(max(bcr_without), max(bcr_with))
    ax.plot([0, max_bcr * 1.1], [0, max_bcr * 1.1], 'k--', alpha=0.3,
            linewidth=1.5, label='No Change (y=x)')

    # Add threshold lines
    ax.axhline(y=bcr_threshold, color='gray', linestyle='--', alpha=0.5,
              linewidth=1.5, label=f'Viability Threshold (BCR={bcr_threshold})')
    ax.axvline(x=bcr_threshold, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

    # Annotate points below diagonal (capacity reduces BCR significantly)
    for i, dev_id in enumerate(dev_ids):
        if bcr_with[i] < bcr_without[i] * 0.9:  # More than 10% reduction
            ax.annotate(dev_id, (bcr_without[i], bcr_with[i]),
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=8, alpha=0.7)

    # Customize plot
    ax.set_xlabel('BCR WITHOUT Capacity Interventions', fontsize=12, fontweight='bold')
    ax.set_ylabel('BCR WITH Capacity Interventions', fontsize=12, fontweight='bold')
    ax.set_title('Benefit-Cost Ratio Comparison:\nImpact of Capacity Intervention Costs',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim(0, max_bcr * 1.1)
    ax.set_ylim(0, max_bcr * 1.1)

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Viable (both)'),
        Patch(facecolor='orange', edgecolor='black', label='Capacity kills viability'),
        Patch(facecolor='red', edgecolor='black', label='Not viable (both)'),
        Patch(facecolor='blue', edgecolor='black', label='Viable only with capacity')
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95)

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir) / 'bcr_comparison_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    return fig, ax


def plot_bcr_by_development(viability_results, bcr_threshold=1.0, output_dir='plots/viability'):
    """
    Generate grouped bar chart showing BCR with/without capacity by development.

    Parameters:
    -----------
    viability_results : dict
        Dictionary with dev_id keys and viability assessment values
    bcr_threshold : float
        Viability threshold line
    output_dir : str
        Directory to save the plot

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare data
    dev_ids = list(viability_results.keys())
    bcr_without = [viability_results[dev]['bcr_without_capacity'] for dev in dev_ids]
    bcr_with = [viability_results[dev]['bcr_with_capacity'] for dev in dev_ids]

    # Sort by BCR with capacity (descending)
    sorted_data = sorted(zip(dev_ids, bcr_without, bcr_with),
                        key=lambda x: x[2], reverse=True)
    dev_ids_sorted, bcr_without_sorted, bcr_with_sorted = zip(*sorted_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

    # Set positions for grouped bars
    x = np.arange(len(dev_ids_sorted))
    width = 0.35

    # Create grouped bars
    bars1 = ax.bar(x - width/2, bcr_without_sorted, width,
                   label='WITHOUT Capacity', color='#2ecc71',
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, bcr_with_sorted, width,
                   label='WITH Capacity', color='#e74c3c',
                   edgecolor='black', linewidth=0.5)

    # Add threshold line
    ax.axhline(y=bcr_threshold, color='black', linestyle='--',
              linewidth=2, label=f'Viability Threshold (BCR={bcr_threshold})')

    # Customize plot
    ax.set_xlabel('Development ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Benefit-Cost Ratio (BCR)', fontsize=12, fontweight='bold')
    ax.set_title('Development Viability: BCR With vs Without Capacity Costs',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(dev_ids_sorted, rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save figure
    output_path = Path(output_dir) / 'bcr_by_development.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    return fig, ax


def export_cost_breakdown_table(infrastructure_costs, output_path='results/cost_breakdown.csv'):
    """
    Export comprehensive cost breakdown table to CSV.

    Parameters:
    -----------
    infrastructure_costs : dict
        Dictionary with dev_id keys and cost breakdown values
    output_path : str
        Path to save CSV file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for DataFrame
    data = []
    for dev_id, costs in infrastructure_costs.items():
        row = {
            'Development': dev_id,
            'Base_Construction_CHF_M': costs['base_construction'] / 1e6,
            'Cap_Inter_Construction_CHF_M': costs['capacity_construction'] / 1e6,
            'Capacity_Surplus_Pct': costs['capacity_surplus_pct'],
            'Base_Maintenance_PV_CHF_M': costs['base_maintenance_pv'] / 1e6,
            'Cap_Maintenance_PV_CHF_M': costs['capacity_maintenance_pv'] / 1e6,
            'Total_Lifecycle_CHF_M': costs['total_lifecycle'] / 1e6
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"✓ Saved: {output_path}")

    return df


def export_viability_results(viability_results, output_path='results/viability_assessment.csv'):
    """
    Export viability assessment results to CSV.

    Parameters:
    -----------
    viability_results : dict
        Dictionary with dev_id keys and viability assessment values
    output_path : str
        Path to save CSV file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for DataFrame
    data = []
    for dev_id, viability in viability_results.items():
        row = {
            'Development': dev_id,
            'Benefits_CHF_M': viability['benefits'] / 1e6,
            'Costs_Without_Capacity_CHF_M': viability['costs_without_capacity'] / 1e6,
            'Costs_With_Capacity_CHF_M': viability['costs_with_capacity'] / 1e6,
            'BCR_Without_Capacity': viability['bcr_without_capacity'],
            'BCR_With_Capacity': viability['bcr_with_capacity'],
            'Viable_Without': viability['viable_without_capacity'],
            'Viable_With': viability['viable_with_capacity'],
            'Capacity_Impact_on_BCR': viability['capacity_impact_on_bcr']
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"✓ Saved: {output_path}")

    return df
