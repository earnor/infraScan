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
import paths


# ============================================================================
# NEW FUNCTIONS: BCR Data Loading and Visualization
# ============================================================================

def load_viability_results_from_csv(
    summary_with_capacity=None,
    summary_without_capacity=None,
    bcr_threshold=1.0
):
    """
    Load BCR data from summary CSVs and format for visualization functions.
    
    This function reads the pipeline-generated summary CSVs and transforms them
    into the dictionary format expected by the BCR visualization functions.
    
    Parameters:
    -----------
    summary_with_capacity : str
        Path to summary CSV with capacity interventions (default: data/costs/total_costs_summary.csv)
    summary_without_capacity : str
        Path to summary CSV without capacity interventions (default: data/costs/total_costs_summary_old.csv)
    bcr_threshold : float
        Viability threshold for BCR (default 1.0)
    
    Returns:
    --------
    viability_results : dict
        Dictionary with development IDs as keys, containing:
        - bcr_without_capacity : float
        - bcr_with_capacity : float
        - viable_without_capacity : bool
        - viable_with_capacity : bool
        - benefits : float (Monetized Savings Mean in CHF millions)
        - costs_without_capacity : float (Total Costs in CHF millions)
        - costs_with_capacity : float (Total Costs in CHF millions)
        - capacity_impact_on_bcr : float (percentage change)
    
    Raises:
    -------
    FileNotFoundError
        If either summary CSV file is not found
    KeyError
        If required columns are missing from the CSVs
    """
    # Set default paths relative to MAIN directory
    if summary_with_capacity is None:
        summary_with_capacity = Path(paths.MAIN) / 'data' / 'costs' / 'total_costs_summary.csv'
    else:
        summary_with_capacity = Path(summary_with_capacity)
    
    if summary_without_capacity is None:
        summary_without_capacity = Path(paths.MAIN) / 'data' / 'costs' / 'total_costs_summary_old.csv'
    else:
        summary_without_capacity = Path(summary_without_capacity)
    
    # Check if files exist
    if not summary_with_capacity.exists():
        raise FileNotFoundError(
            f"Summary CSV with capacity interventions not found: {summary_with_capacity}\n"
            f"Please run the pipeline through Phase 12 first."
        )
    
    if not summary_without_capacity.exists():
        raise FileNotFoundError(
            f"Summary CSV without capacity interventions not found: {summary_without_capacity}\n"
            f"Please run the pipeline through Phase 12 first."
        )
    
    # Load CSVs
    print(f"Loading BCR data from summary CSVs...")
    print(f"  • WITH capacity:    {summary_with_capacity}")
    print(f"  • WITHOUT capacity: {summary_without_capacity}")
    
    df_with = pd.read_csv(summary_with_capacity)
    df_without = pd.read_csv(summary_without_capacity)
    
    # Validate required columns
    required_cols = ['development', 'CBA Ratio', 'Total Costs [in Mio. CHF]', 'Monetized Savings Mean [in Mio. CHF]']
    
    for col in required_cols:
        if col not in df_with.columns:
            raise KeyError(f"Column '{col}' not found in {summary_with_capacity}")
        if col not in df_without.columns:
            raise KeyError(f"Column '{col}' not found in {summary_without_capacity}")
    
    # Merge dataframes on development ID
    df_merged = df_with.merge(
        df_without[['development', 'CBA Ratio', 'Total Costs [in Mio. CHF]']],
        on='development',
        suffixes=('_with', '_without')
    )
    
    # Build viability results dictionary
    viability_results = {}
    
    for _, row in df_merged.iterrows():
        # Extract development ID (e.g., "Development_101025" -> "101025")
        dev_full = row['development']
        dev_id = dev_full.replace('Development_', '')
        
        # Extract BCR values
        bcr_with = row['CBA Ratio_with']
        bcr_without = row['CBA Ratio_without']
        
        # Extract costs and benefits
        costs_with = row['Total Costs [in Mio. CHF]_with']
        costs_without = row['Total Costs [in Mio. CHF]_without']
        benefits = row['Monetized Savings Mean [in Mio. CHF]']
        
        # Determine viability
        viable_with = bcr_with >= bcr_threshold
        viable_without = bcr_without >= bcr_threshold
        
        # Calculate capacity impact on BCR
        if bcr_without > 0:
            capacity_impact = ((bcr_with - bcr_without) / bcr_without) * 100
        else:
            capacity_impact = 0.0
        
        # Store results
        viability_results[dev_id] = {
            'bcr_without_capacity': bcr_without,
            'bcr_with_capacity': bcr_with,
            'viable_without_capacity': viable_without,
            'viable_with_capacity': viable_with,
            'benefits': benefits,
            'costs_without_capacity': costs_without,
            'costs_with_capacity': costs_with,
            'capacity_impact_on_bcr': capacity_impact
        }
    
    print(f"  ✓ Loaded {len(viability_results)} developments")
    print(f"  ✓ Viability threshold: BCR >= {bcr_threshold}")
    
    # Print summary statistics
    viable_without_count = sum(1 for v in viability_results.values() if v['viable_without_capacity'])
    viable_with_count = sum(1 for v in viability_results.values() if v['viable_with_capacity'])
    capacity_kills_count = sum(
        1 for v in viability_results.values() 
        if v['viable_without_capacity'] and not v['viable_with_capacity']
    )
    
    print(f"\n  Summary:")
    print(f"    • Viable WITHOUT capacity: {viable_without_count}/{len(viability_results)}")
    print(f"    • Viable WITH capacity:    {viable_with_count}/{len(viability_results)}")
    print(f"    • Capacity kills viability: {capacity_kills_count} developments")
    
    return viability_results


def plot_bcr_analysis_from_pipeline(
    bcr_threshold=1.0,
    output_dir=None,
    summary_with_capacity=None,
    summary_without_capacity=None
):
    """
    Load pipeline results and generate both BCR visualization plots.
    
    This is a convenience function that automatically loads the summary CSVs
    and generates both scatter and bar chart visualizations.
    
    Parameters:
    -----------
    bcr_threshold : float
        Viability threshold (default 1.0)
    output_dir : str or None
        Directory to save plots (default: plots/viability)
    summary_with_capacity : str or None
        Path to summary CSV with capacity interventions (default: auto-detect)
    summary_without_capacity : str or None
        Path to summary CSV without capacity interventions (default: auto-detect)
    
    Returns:
    --------
    viability_results : dict
        The loaded viability results dictionary (useful for further analysis)
    
    Raises:
    -------
    FileNotFoundError
        If summary CSV files are not found
    """
    # Set default output directory
    if output_dir is None:
        output_dir = Path(paths.MAIN) / paths.PLOT_DIRECTORY / 'Viability'
    
    print("\n" + "="*80)
    print("BCR VIABILITY ANALYSIS")
    print("="*80 + "\n")
    
    # Load data from CSVs (will use default paths if None)
    viability_results = load_viability_results_from_csv(
        summary_with_capacity=summary_with_capacity,
        summary_without_capacity=summary_without_capacity,
        bcr_threshold=bcr_threshold
    )
    
    # Generate scatter plot
    print("\n" + "-"*80)
    print("Generating BCR Comparison Scatter Plot...")
    print("-"*80)
    plot_bcr_comparison_scatter(
        viability_results=viability_results,
        bcr_threshold=bcr_threshold,
        output_dir=output_dir
    )
    
    # Generate bar chart
    print("\n" + "-"*80)
    print("Generating BCR by Development Bar Chart...")
    print("-"*80)
    plot_bcr_by_development(
        viability_results=viability_results,
        bcr_threshold=bcr_threshold,
        output_dir=output_dir
    )
    
    # Export viability results to CSV
    export_viability_results(
        viability_results=viability_results,
        output_path=str(Path(output_dir) / 'viability_assessment.csv')
    )
    
    print("\n" + "="*80)
    print("✓ BCR ANALYSIS COMPLETE")
    print("="*80)
    print(f"All plots saved to: {output_dir}/")
    print(f"  • bcr_comparison_scatter.png")
    print(f"  • bcr_by_development.png")
    print(f"  • viability_assessment.csv\n")
    
    return viability_results


def plot_cost_increase_analysis(
    summary_with_capacity=None,
    summary_without_capacity=None,
    output_dir=None
):
    """
    Analyze and visualize cost increases due to capacity interventions.
    
    Creates visualizations showing:
    1. Absolute cost increases (in CHF millions) - stacked bar chart
    2. Percentage cost increases by development - sorted bar chart  
    3. Box plot comparison by development type
    4. Summary statistics tables exported to CSV
    
    The function automatically classifies developments into:
    - EXTEND_LINES (dev_id < 101000): Extensions of existing rail lines
    - NEW_DIRECT_CONNECTIONS (dev_id >= 101000): New direct connections
    
    Parameters:
    -----------
    summary_with_capacity : str or None
        Path to summary CSV with capacity interventions (default: data/costs/total_costs_summary.csv)
    summary_without_capacity : str or None
        Path to summary CSV without capacity interventions (default: data/costs/total_costs_summary_old.csv)
    output_dir : str or None
        Directory to save plots (default: plots/costs)
    
    Returns:
    --------
    stats_dict : dict
        Dictionary containing:
        - 'all_developments': Summary statistics for all developments
        - 'by_type': Statistics broken down by development type
        - 'detailed_data': Full merged DataFrame with all calculations
    
    Raises:
    -------
    FileNotFoundError
        If summary CSV files are not found
    
    Examples:
    ---------
    >>> # Run with default paths
    >>> stats = plot_cost_increase_analysis()
    >>> 
    >>> # Access statistics programmatically
    >>> print(f"Mean cost increase: {stats['all_developments']['mean_increase_pct']:.2f}%")
    >>> print(f"EXTEND_LINES mean: {stats['by_type']['EXTEND_LINES']['mean_increase_pct']:.2f}%")
    """
    import settings
    
    # Set default paths
    if summary_with_capacity is None:
        summary_with_capacity = Path(paths.MAIN) / 'data' / 'costs' / 'total_costs_summary.csv'
    else:
        summary_with_capacity = Path(summary_with_capacity)
    
    if summary_without_capacity is None:
        summary_without_capacity = Path(paths.MAIN) / 'data' / 'costs' / 'total_costs_summary_old.csv'
    else:
        summary_without_capacity = Path(summary_without_capacity)
    
    if output_dir is None:
        output_dir = Path(paths.MAIN) / paths.PLOT_DIRECTORY / 'costs'
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if files exist
    if not summary_with_capacity.exists():
        raise FileNotFoundError(
            f"Summary CSV with capacity interventions not found: {summary_with_capacity}\n"
            f"Please run the pipeline through Phase 12 first."
        )
    
    if not summary_without_capacity.exists():
        raise FileNotFoundError(
            f"Summary CSV without capacity interventions not found: {summary_without_capacity}\n"
            f"Please run the pipeline through Phase 12 first."
        )
    
    # Load data
    print("\n" + "="*80)
    print("COST INCREASE ANALYSIS: Impact of Capacity Interventions")
    print("="*80 + "\n")
    print(f"Loading data from:")
    print(f"  • WITH capacity:    {summary_with_capacity}")
    print(f"  • WITHOUT capacity: {summary_without_capacity}\n")
    
    df_with = pd.read_csv(summary_with_capacity)
    df_without = pd.read_csv(summary_without_capacity)
    
    # Merge dataframes
    df_merged = df_with.merge(
        df_without[['development', 'Total Costs [in Mio. CHF]']],
        on='development',
        suffixes=('_with', '_without')
    )
    
    # Extract development ID and classify by type
    df_merged['dev_id'] = df_merged['development'].str.replace('Development_', '').astype(int)
    
    # Classify development types based on dev_id ranges from settings.py
    # EXTEND_LINES: 100000-100999
    # NEW_DIRECT_CONNECTIONS: 101000+
    df_merged['dev_type'] = df_merged['dev_id'].apply(
        lambda x: 'EXTEND_LINES' if x < settings.dev_id_start_new_direct_connections 
        else 'NEW_DIRECT_CONNECTIONS'
    )
    
    # Calculate cost increases
    df_merged['cost_increase_abs'] = (
        df_merged['Total Costs [in Mio. CHF]_with'] - 
        df_merged['Total Costs [in Mio. CHF]_without']
    )
    df_merged['cost_increase_pct'] = (
        (df_merged['cost_increase_abs'] / df_merged['Total Costs [in Mio. CHF]_without']) * 100
    )
    
    # Sort by development ID
    df_merged = df_merged.sort_values('dev_id')
    
    # Calculate summary statistics
    stats_all = {
        'total_devs': len(df_merged),
        'total_cost_without': df_merged['Total Costs [in Mio. CHF]_without'].sum(),
        'total_cost_with': df_merged['Total Costs [in Mio. CHF]_with'].sum(),
        'total_increase_abs': df_merged['cost_increase_abs'].sum(),
        'total_increase_pct': (df_merged['cost_increase_abs'].sum() / 
                              df_merged['Total Costs [in Mio. CHF]_without'].sum()) * 100,
        'mean_increase_abs': df_merged['cost_increase_abs'].mean(),
        'mean_increase_pct': df_merged['cost_increase_pct'].mean(),
        'median_increase_pct': df_merged['cost_increase_pct'].median(),
        'min_increase_pct': df_merged['cost_increase_pct'].min(),
        'max_increase_pct': df_merged['cost_increase_pct'].max()
    }
    
    # Calculate statistics by development type
    stats_by_type = {}
    for dev_type in ['EXTEND_LINES', 'NEW_DIRECT_CONNECTIONS']:
        df_type = df_merged[df_merged['dev_type'] == dev_type]
        if len(df_type) > 0:
            stats_by_type[dev_type] = {
                'count': len(df_type),
                'total_cost_without': df_type['Total Costs [in Mio. CHF]_without'].sum(),
                'total_cost_with': df_type['Total Costs [in Mio. CHF]_with'].sum(),
                'total_increase_abs': df_type['cost_increase_abs'].sum(),
                'total_increase_pct': (df_type['cost_increase_abs'].sum() / 
                                      df_type['Total Costs [in Mio. CHF]_without'].sum()) * 100,
                'mean_increase_abs': df_type['cost_increase_abs'].mean(),
                'mean_increase_pct': df_type['cost_increase_pct'].mean(),
                'median_increase_pct': df_type['cost_increase_pct'].median(),
                'min_increase_pct': df_type['cost_increase_pct'].min(),
                'max_increase_pct': df_type['cost_increase_pct'].max()
            }
    
    # Print summary statistics
    print("-" * 80)
    print("SUMMARY STATISTICS: ALL DEVELOPMENTS")
    print("-" * 80)
    print(f"Total Developments:                {stats_all['total_devs']}")
    print(f"Total Costs WITHOUT Capacity:      {stats_all['total_cost_without']:,.2f} CHF M")
    print(f"Total Costs WITH Capacity:         {stats_all['total_cost_with']:,.2f} CHF M")
    print(f"Total Cost Increase (Absolute):    {stats_all['total_increase_abs']:,.2f} CHF M")
    print(f"Total Cost Increase (Percentage):  {stats_all['total_increase_pct']:.2f}%")
    print(f"\nAverage Cost Increase per Dev:     {stats_all['mean_increase_abs']:,.2f} CHF M")
    print(f"Mean Percentage Increase:          {stats_all['mean_increase_pct']:.2f}%")
    print(f"Median Percentage Increase:        {stats_all['median_increase_pct']:.2f}%")
    print(f"Range (min-max):                   {stats_all['min_increase_pct']:.2f}% - {stats_all['max_increase_pct']:.2f}%")
    
    for dev_type in ['EXTEND_LINES', 'NEW_DIRECT_CONNECTIONS']:
        if dev_type in stats_by_type:
            stats = stats_by_type[dev_type]
            print(f"\n{'-' * 80}")
            print(f"STATISTICS: {dev_type}")
            print("-" * 80)
            print(f"Number of Developments:            {stats['count']}")
            print(f"Total Costs WITHOUT Capacity:      {stats['total_cost_without']:,.2f} CHF M")
            print(f"Total Costs WITH Capacity:         {stats['total_cost_with']:,.2f} CHF M")
            print(f"Total Cost Increase (Absolute):    {stats['total_increase_abs']:,.2f} CHF M")
            print(f"Total Cost Increase (Percentage):  {stats['total_increase_pct']:.2f}%")
            print(f"\nAverage Cost Increase per Dev:     {stats['mean_increase_abs']:,.2f} CHF M")
            print(f"Mean Percentage Increase:          {stats['mean_increase_pct']:.2f}%")
            print(f"Median Percentage Increase:        {stats['median_increase_pct']:.2f}%")
            print(f"Range (min-max):                   {stats['min_increase_pct']:.2f}% - {stats['max_increase_pct']:.2f}%")
    
    print("\n" + "="*80 + "\n")
    
    # ========================================================================
    # VISUALIZATION 1: Stacked Bar Chart (Absolute Costs)
    # ========================================================================
    print("Generating Visualization 1: Absolute Cost Comparison...")
    
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    
    x = np.arange(len(df_merged))
    width = 0.6
    
    # Base costs (without capacity)
    p1 = ax.bar(x, df_merged['Total Costs [in Mio. CHF]_without'], width,
                label='Base Infrastructure Costs', color='#3498db', 
                edgecolor='black', linewidth=0.5)
    
    # Additional capacity costs
    p2 = ax.bar(x, df_merged['cost_increase_abs'], width,
                bottom=df_merged['Total Costs [in Mio. CHF]_without'],
                label='Capacity Intervention Costs', color='#e74c3c',
                edgecolor='black', linewidth=0.5)
    
    # Customize plot
    ax.set_xlabel('Development ID', fontsize=13, fontweight='bold')
    ax.set_ylabel('Total Costs (CHF Millions)', fontsize=13, fontweight='bold')
    ax.set_title('Cost Impact of Capacity Interventions: Base Infrastructure vs. Capacity Costs',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df_merged['dev_id'], rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'.replace(',', "'")))
    
    # Add separator line between development types
    if 'EXTEND_LINES' in stats_by_type and 'NEW_DIRECT_CONNECTIONS' in stats_by_type:
        separator_idx = stats_by_type['EXTEND_LINES']['count'] - 0.5
        ax.axvline(x=separator_idx, color='black', linestyle='--', linewidth=2, alpha=0.5)
        
        # Add type labels
        extend_center = stats_by_type['EXTEND_LINES']['count'] / 2
        new_direct_center = stats_by_type['EXTEND_LINES']['count'] + \
                           stats_by_type['NEW_DIRECT_CONNECTIONS']['count'] / 2
        
        y_pos = ax.get_ylim()[1] * 0.95
        ax.text(extend_center, y_pos, 'EXTEND_LINES', ha='center', va='top',
                fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5',
                facecolor='lightblue', alpha=0.7))
        ax.text(new_direct_center, y_pos, 'NEW_DIRECT_CONNECTIONS', ha='center', va='top',
                fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5',
                facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    output_path_1 = Path(output_dir) / 'cost_increase_absolute.png'
    plt.savefig(output_path_1, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path_1}")
    plt.close()
    
    # ========================================================================
    # VISUALIZATION 2: Percentage Cost Increase by Development
    # ========================================================================
    print("Generating Visualization 2: Percentage Cost Increase by Development...")
    
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    
    # Sort by percentage increase for better visualization
    df_sorted = df_merged.sort_values('cost_increase_pct', ascending=False)
    x_sorted = np.arange(len(df_sorted))
    
    # Color by development type
    colors = df_sorted['dev_type'].map({
        'EXTEND_LINES': '#3498db',
        'NEW_DIRECT_CONNECTIONS': '#e74c3c'
    })
    
    bars = ax.bar(x_sorted, df_sorted['cost_increase_pct'],
                  color=colors, edgecolor='black', linewidth=0.5, width=0.7)
    
    # Customize plot
    ax.set_xlabel('Development ID (sorted by % increase)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cost Increase (%)', fontsize=13, fontweight='bold')
    ax.set_title('Percentage Cost Increase Due to Capacity Interventions',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x_sorted)
    ax.set_xticklabels(df_sorted['dev_id'], rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add mean and median lines
    ax.axhline(y=stats_all['mean_increase_pct'], color='red', linestyle='--',
              linewidth=2, label=f'Mean: {stats_all["mean_increase_pct"]:.1f}%')
    ax.axhline(y=stats_all['median_increase_pct'], color='orange', linestyle='--',
              linewidth=2, label=f'Median: {stats_all["median_increase_pct"]:.1f}%')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='EXTEND_LINES'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='NEW_DIRECT_CONNECTIONS'),
        plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {stats_all["mean_increase_pct"]:.1f}%'),
        plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {stats_all["median_increase_pct"]:.1f}%')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    output_path_2 = Path(output_dir) / 'cost_increase_percentage.png'
    plt.savefig(output_path_2, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path_2}")
    plt.close()
    
    # ========================================================================
    # VISUALIZATION 3: Box Plot Comparison by Development Type
    # ========================================================================
    print("Generating Visualization 3: Box Plot Comparison by Type...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # Prepare data for box plots
    data_by_type = [
        df_merged[df_merged['dev_type'] == 'EXTEND_LINES']['cost_increase_pct'].values,
        df_merged[df_merged['dev_type'] == 'NEW_DIRECT_CONNECTIONS']['cost_increase_pct'].values
    ]
    
    # Box plot for percentage increases
    bp1 = ax1.boxplot(data_by_type, labels=['EXTEND_LINES', 'NEW_DIRECT_CONNECTIONS'],
                      patch_artist=True, showmeans=True, meanline=True)
    
    for patch, color in zip(bp1['boxes'], ['#3498db', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Cost Increase (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Percentage Cost Increase Distribution', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_xticklabels(['EXTEND_LINES', 'NEW_DIRECT_CONNECTIONS'], rotation=15, ha='right')
    
    # Prepare absolute increase data
    data_abs_by_type = [
        df_merged[df_merged['dev_type'] == 'EXTEND_LINES']['cost_increase_abs'].values,
        df_merged[df_merged['dev_type'] == 'NEW_DIRECT_CONNECTIONS']['cost_increase_abs'].values
    ]
    
    # Box plot for absolute increases
    bp2 = ax2.boxplot(data_abs_by_type, labels=['EXTEND_LINES', 'NEW_DIRECT_CONNECTIONS'],
                      patch_artist=True, showmeans=True, meanline=True)
    
    for patch, color in zip(bp2['boxes'], ['#3498db', '#e74c3c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_ylabel('Cost Increase (CHF Millions)', fontsize=12, fontweight='bold')
    ax2.set_title('Absolute Cost Increase Distribution', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_xticklabels(['EXTEND_LINES', 'NEW_DIRECT_CONNECTIONS'], rotation=15, ha='right')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'.replace(',', "'")))
    
    plt.suptitle('Cost Increase Comparison by Development Type',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path_3 = Path(output_dir) / 'cost_increase_boxplot_comparison.png'
    plt.savefig(output_path_3, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path_3}")
    plt.close()
    
    # ========================================================================
    # EXPORT: Summary Statistics Table
    # ========================================================================
    print("Generating Summary Statistics Table...")
    
    # Create summary DataFrame
    summary_rows = []
    
    # All developments row
    summary_rows.append({
        'Category': 'ALL DEVELOPMENTS',
        'Count': stats_all['total_devs'],
        'Total Cost Without (CHF M)': f"{stats_all['total_cost_without']:.2f}",
        'Total Cost With (CHF M)': f"{stats_all['total_cost_with']:.2f}",
        'Total Increase (CHF M)': f"{stats_all['total_increase_abs']:.2f}",
        'Total Increase (%)': f"{stats_all['total_increase_pct']:.2f}%",
        'Mean Increase (CHF M)': f"{stats_all['mean_increase_abs']:.2f}",
        'Mean Increase (%)': f"{stats_all['mean_increase_pct']:.2f}%",
        'Median Increase (%)': f"{stats_all['median_increase_pct']:.2f}%",
        'Range (%)': f"{stats_all['min_increase_pct']:.2f}% - {stats_all['max_increase_pct']:.2f}%"
    })
    
    # Type-specific rows
    for dev_type in ['EXTEND_LINES', 'NEW_DIRECT_CONNECTIONS']:
        if dev_type in stats_by_type:
            stats = stats_by_type[dev_type]
            summary_rows.append({
                'Category': dev_type,
                'Count': stats['count'],
                'Total Cost Without (CHF M)': f"{stats['total_cost_without']:.2f}",
                'Total Cost With (CHF M)': f"{stats['total_cost_with']:.2f}",
                'Total Increase (CHF M)': f"{stats['total_increase_abs']:.2f}",
                'Total Increase (%)': f"{stats['total_increase_pct']:.2f}%",
                'Mean Increase (CHF M)': f"{stats['mean_increase_abs']:.2f}",
                'Mean Increase (%)': f"{stats['mean_increase_pct']:.2f}%",
                'Median Increase (%)': f"{stats['median_increase_pct']:.2f}%",
                'Range (%)': f"{stats['min_increase_pct']:.2f}% - {stats['max_increase_pct']:.2f}%"
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save to CSV
    csv_output_path = Path(output_dir) / 'cost_increase_summary_statistics.csv'
    summary_df.to_csv(csv_output_path, index=False)
    print(f"  ✓ Saved: {csv_output_path}")
    
    # Also save detailed development-level data
    detailed_output = df_merged[[
        'development', 'dev_id', 'dev_type',
        'Total Costs [in Mio. CHF]_without',
        'Total Costs [in Mio. CHF]_with',
        'cost_increase_abs',
        'cost_increase_pct'
    ]].copy()
    
    detailed_output.columns = [
        'Development', 'Dev_ID', 'Type',
        'Cost_Without_CHF_M', 'Cost_With_CHF_M',
        'Increase_Abs_CHF_M', 'Increase_Pct'
    ]
    
    detailed_csv_path = Path(output_dir) / 'cost_increase_detailed.csv'
    detailed_output.to_csv(detailed_csv_path, index=False, float_format='%.2f')
    print(f"  ✓ Saved: {detailed_csv_path}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("✓ COST INCREASE ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll visualizations saved to: {output_dir}/")
    print(f"  • cost_increase_absolute.png")
    print(f"  • cost_increase_percentage.png")
    print(f"  • cost_increase_boxplot_comparison.png")
    print(f"  • cost_increase_summary_statistics.csv")
    print(f"  • cost_increase_detailed.csv")
    print("\n" + "="*80 + "\n")
    
    # Return statistics for programmatic access
    return {
        'all_developments': stats_all,
        'by_type': stats_by_type,
        'detailed_data': df_merged
    }


# ============================================================================
# EXISTING FUNCTIONS (keep as-is)
# ============================================================================

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
            'Benefits_CHF_M': viability['benefits'],
            'Costs_Without_Capacity_CHF_M': viability['costs_without_capacity'],
            'Costs_With_Capacity_CHF_M': viability['costs_with_capacity'],
            'BCR_Without_Capacity': viability['bcr_without_capacity'],
            'BCR_With_Capacity': viability['bcr_with_capacity'],
            'Viable_Without': viability['viable_without_capacity'],
            'Viable_With': viability['viable_with_capacity'],
            'Capacity_Impact_on_BCR_Pct': viability['capacity_impact_on_bcr']
        }
        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"✓ Saved: {output_path}")

    return df


# ============================================================================
# TEST RUNNER
# ============================================================================

if __name__ == '__main__':
    """
    Test runner for cost visualization functions.
    
    This can be run standalone to test the analysis after Phase 12 completes.
    
    Usage:
        python visualization_costs.py
    """
    print("\n" + "="*80)
    print("TESTING COST VISUALIZATION FUNCTIONS")
    print("="*80 + "\n")
    
    print("This test requires the pipeline to have completed through Phase 12.")
    print("Expected files:")
    print("  • data/costs/total_costs_summary.csv")
    print("  • data/costs/total_costs_summary_old.csv")
    print("\n" + "-"*80 + "\n")
    
    # Run the analysis
    try:
        # Test 1: Cost Increase Analysis
        print("="*80)
        print("TEST 1: COST INCREASE ANALYSIS")
        print("="*80 + "\n")
        
        cost_stats = plot_cost_increase_analysis()
        
        print("\n" + "="*80)
        print("TEST 1 SUCCESSFUL - Cost Increase Analysis")
        print("="*80)
        print(f"\nAnalyzed {cost_stats['all_developments']['total_devs']} developments.")
        print(f"Mean cost increase: {cost_stats['all_developments']['mean_increase_pct']:.2f}%")
        print(f"\nYou can find the plots in: {Path(paths.PLOT_DIRECTORY) / 'costs'}/")
        
        # Test 2: BCR Viability Analysis
        print("\n\n" + "="*80)
        print("TEST 2: BCR VIABILITY ANALYSIS")
        print("="*80 + "\n")
        
        viability_results = plot_bcr_analysis_from_pipeline(
            bcr_threshold=1.0
        )
        
        print("\n" + "="*80)
        print("TEST 2 SUCCESSFUL - BCR Viability Analysis")
        print("="*80)
        print(f"\nGenerated visualizations for {len(viability_results)} developments.")
        print(f"\nYou can find the plots in: {Path(paths.PLOT_DIRECTORY) / 'viability'}/")
        
        # Final Summary
        print("\n\n" + "="*80)
        print("✓ ALL TESTS PASSED SUCCESSFULLY")
        print("="*80)
        print("\nGenerated outputs:")
        print(f"  • Cost Increase Analysis:  {Path(paths.PLOT_DIRECTORY) / 'costs'}/")
        print(f"    - cost_increase_absolute.png")
        print(f"    - cost_increase_percentage.png")
        print(f"    - cost_increase_boxplot_comparison.png")
        print(f"    - cost_increase_summary_statistics.csv")
        print(f"    - cost_increase_detailed.csv")
        print(f"\n  • BCR Viability Analysis:  {Path(paths.PLOT_DIRECTORY) / 'viability'}/")
        print(f"    - bcr_comparison_scatter.png")
        print(f"    - bcr_by_development.png")
        print(f"    - viability_assessment.csv")
        print("\n" + "="*80 + "\n")
        
    except FileNotFoundError as e:
        print("\n" + "="*80)
        print("TEST FAILED: Required files not found")
        print("="*80)
        print(f"\nError: {e}")
        print("\nPlease run the main pipeline through Phase 12 first:")
        print("  python main_cap.py")
        
    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED: Unexpected error")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()