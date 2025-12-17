"""
Standalone script to export pipeline comparison report statistics.

This script generates comprehensive report-ready statistics comparing
the old vs new pipeline without re-running the entire main pipeline.

Usage:
    python export_pipeline_reports.py

Generates 5 CSV files in plots/Benefits_Pipeline_Comparison/report_statistics/:
    1. report_pipeline_viability_statistics.csv
    2. report_pipeline_cost_statistics.csv
    3. report_pipeline_bcr_statistics.csv
    4. report_pipeline_net_benefit_statistics.csv
    5. report_pipeline_top10_performers.csv
"""

import os
import pandas as pd
import geopandas as gpd
from pathlib import Path
import paths


def export_pipeline_comparison_report_statistics():
    """
    Export comprehensive report-ready statistics for old vs new pipeline comparison.

    Generates 5 CSV files with detailed comparative statistics suitable for
    academic reporting, similar to sensitivity analysis output.

    Files created in plots/Benefits_Pipeline_Comparison/report_statistics/:
    - report_pipeline_viability_statistics.csv
    - report_pipeline_cost_statistics.csv
    - report_pipeline_bcr_statistics.csv
    - report_pipeline_net_benefit_statistics.csv
    - report_pipeline_top10_performers.csv
    """
    print("\n" + "="*80)
    print("EXPORTING PIPELINE COMPARISON REPORT STATISTICS")
    print("="*80 + "\n")

    # Load data from both pipelines
    new_data_path = "data/costs/total_costs_raw.csv"
    old_data_path = "data/costs/total_costs_raw_old.csv"

    if not os.path.exists(new_data_path) or not os.path.exists(old_data_path):
        print(f"  ⚠ Data files not found - skipping report statistics export")
        print(f"     Expected files:")
        print(f"       - {new_data_path}")
        print(f"       - {old_data_path}")
        return

    # Output directory
    output_dir = os.path.join("plots", "Benefits_Pipeline_Comparison", "report_statistics")
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare data using the helper function from plots.py
    from plots import _prepare_pipeline_data

    print(f"  Loading and preparing pipeline data...")
    df_new = _prepare_pipeline_data(new_data_path, pipeline='new')
    df_old = _prepare_pipeline_data(old_data_path, pipeline='old')

    # Combine datasets
    df_combined = pd.concat([df_new, df_old], ignore_index=True)

    # Mark viable scenarios (net benefit > 0)
    df_combined['is_viable'] = df_combined['total_net_benefit'] > 0

    # Get all developments
    all_developments = sorted(df_combined['development'].unique())

    print(f"  Processing {len(all_developments)} developments across both pipelines...\n")

    # ========================================================================
    # 1. VIABILITY STATISTICS
    # ========================================================================
    print("  [1/5] Generating viability statistics...")

    viability_stats = []

    for dev_id in all_developments:
        dev_data = df_combined[df_combined['development'] == dev_id]
        line_name = dev_data['line_name'].iloc[0] if 'line_name' in dev_data.columns else dev_id

        new_data = dev_data[dev_data['pipeline'] == 'new']
        old_data = dev_data[dev_data['pipeline'] == 'old']

        # Count viable scenarios
        viable_new = new_data['is_viable'].sum() if len(new_data) > 0 else 0
        viable_old = old_data['is_viable'].sum() if len(old_data) > 0 else 0
        total_scenarios = new_data['scenario'].nunique() if len(new_data) > 0 else 0

        viability_stats.append({
            'development': dev_id,
            'line_name': line_name,
            'total_scenarios': total_scenarios,
            'viable_scenarios_old': int(viable_old),
            'viable_scenarios_new': int(viable_new),
            'viable_scenarios_change': int(viable_new - viable_old),
            'viable_pct_old': (viable_old / total_scenarios * 100) if total_scenarios > 0 else 0,
            'viable_pct_new': (viable_new / total_scenarios * 100) if total_scenarios > 0 else 0,
            'viable_pct_change': ((viable_new - viable_old) / total_scenarios * 100) if total_scenarios > 0 else 0,
            'viability_improved': viable_new > viable_old,
            'viability_worsened': viable_new < viable_old,
            'viability_unchanged': viable_new == viable_old
        })

    viability_df = pd.DataFrame(viability_stats)
    viability_path = os.path.join(output_dir, "report_pipeline_viability_statistics.csv")
    viability_df.to_csv(viability_path, index=False)
    print(f"      ✓ Saved: {viability_path}")

    # ========================================================================
    # 2. COST STATISTICS
    # ========================================================================
    print("  [2/5] Generating cost statistics...")

    cost_stats = []

    for dev_id in all_developments:
        dev_data = df_combined[df_combined['development'] == dev_id]
        line_name = dev_data['line_name'].iloc[0] if 'line_name' in dev_data.columns else dev_id

        new_data = dev_data[dev_data['pipeline'] == 'new']
        old_data = dev_data[dev_data['pipeline'] == 'old']

        # Calculate mean costs
        mean_cost_new = new_data['total_costs'].mean() if len(new_data) > 0 else 0
        mean_cost_old = old_data['total_costs'].mean() if len(old_data) > 0 else 0
        cost_change_chf = mean_cost_new - mean_cost_old
        cost_change_pct = (cost_change_chf / mean_cost_old * 100) if mean_cost_old != 0 else 0

        # Cost component breakdown (if available)
        construction_new = new_data['TotalConstructionCost'].mean() if 'TotalConstructionCost' in new_data.columns and len(new_data) > 0 else 0
        construction_old = old_data['TotalConstructionCost'].mean() if 'TotalConstructionCost' in old_data.columns and len(old_data) > 0 else 0

        maintenance_new = new_data['TotalMaintenanceCost'].mean() if 'TotalMaintenanceCost' in new_data.columns and len(new_data) > 0 else 0
        maintenance_old = old_data['TotalMaintenanceCost'].mean() if 'TotalMaintenanceCost' in old_data.columns and len(old_data) > 0 else 0

        operating_new = new_data['TotalUncoveredOperatingCost'].mean() if 'TotalUncoveredOperatingCost' in new_data.columns and len(new_data) > 0 else 0
        operating_old = old_data['TotalUncoveredOperatingCost'].mean() if 'TotalUncoveredOperatingCost' in old_data.columns and len(old_data) > 0 else 0

        cost_stats.append({
            'development': dev_id,
            'line_name': line_name,
            'mean_total_cost_old_chf': mean_cost_old,
            'mean_total_cost_new_chf': mean_cost_new,
            'total_cost_change_chf': cost_change_chf,
            'total_cost_change_pct': cost_change_pct,
            'construction_cost_old_chf': construction_old,
            'construction_cost_new_chf': construction_new,
            'construction_cost_change_chf': construction_new - construction_old,
            'maintenance_cost_old_chf': maintenance_old,
            'maintenance_cost_new_chf': maintenance_new,
            'maintenance_cost_change_chf': maintenance_new - maintenance_old,
            'operating_cost_old_chf': operating_old,
            'operating_cost_new_chf': operating_new,
            'operating_cost_change_chf': operating_new - operating_old,
            'cost_reduced': cost_change_chf < 0,
            'cost_increased': cost_change_chf > 0,
            'cost_unchanged': cost_change_chf == 0
        })

    cost_df = pd.DataFrame(cost_stats)
    cost_path = os.path.join(output_dir, "report_pipeline_cost_statistics.csv")
    cost_df.to_csv(cost_path, index=False)
    print(f"      ✓ Saved: {cost_path}")

    # ========================================================================
    # 3. BCR/CBA STATISTICS
    # ========================================================================
    print("  [3/5] Generating BCR/CBA statistics...")

    bcr_stats = []

    for dev_id in all_developments:
        dev_data = df_combined[df_combined['development'] == dev_id]
        line_name = dev_data['line_name'].iloc[0] if 'line_name' in dev_data.columns else dev_id

        new_data = dev_data[dev_data['pipeline'] == 'new']
        old_data = dev_data[dev_data['pipeline'] == 'old']

        # BCR/CBA statistics
        if 'cba_ratio' in new_data.columns and len(new_data) > 0:
            mean_bcr_new = new_data['cba_ratio'].mean()
            median_bcr_new = new_data['cba_ratio'].median()
            min_bcr_new = new_data['cba_ratio'].min()
            max_bcr_new = new_data['cba_ratio'].max()
            std_bcr_new = new_data['cba_ratio'].std()
            bcr_above_1_new = (new_data['cba_ratio'] >= 1.0).sum()
        else:
            mean_bcr_new = median_bcr_new = min_bcr_new = max_bcr_new = std_bcr_new = bcr_above_1_new = 0

        if 'cba_ratio' in old_data.columns and len(old_data) > 0:
            mean_bcr_old = old_data['cba_ratio'].mean()
            median_bcr_old = old_data['cba_ratio'].median()
            min_bcr_old = old_data['cba_ratio'].min()
            max_bcr_old = old_data['cba_ratio'].max()
            std_bcr_old = old_data['cba_ratio'].std()
            bcr_above_1_old = (old_data['cba_ratio'] >= 1.0).sum()
        else:
            mean_bcr_old = median_bcr_old = min_bcr_old = max_bcr_old = std_bcr_old = bcr_above_1_old = 0

        bcr_stats.append({
            'development': dev_id,
            'line_name': line_name,
            'mean_bcr_old': mean_bcr_old,
            'mean_bcr_new': mean_bcr_new,
            'bcr_change': mean_bcr_new - mean_bcr_old,
            'median_bcr_old': median_bcr_old,
            'median_bcr_new': median_bcr_new,
            'min_bcr_old': min_bcr_old,
            'min_bcr_new': min_bcr_new,
            'max_bcr_old': max_bcr_old,
            'max_bcr_new': max_bcr_new,
            'std_bcr_old': std_bcr_old,
            'std_bcr_new': std_bcr_new,
            'scenarios_bcr_above_1_old': int(bcr_above_1_old),
            'scenarios_bcr_above_1_new': int(bcr_above_1_new),
            'bcr_viability_change': int(bcr_above_1_new - bcr_above_1_old),
            'bcr_improved': mean_bcr_new > mean_bcr_old,
            'bcr_worsened': mean_bcr_new < mean_bcr_old,
            'bcr_unchanged': mean_bcr_new == mean_bcr_old
        })

    bcr_df = pd.DataFrame(bcr_stats)
    bcr_path = os.path.join(output_dir, "report_pipeline_bcr_statistics.csv")
    bcr_df.to_csv(bcr_path, index=False)
    print(f"      ✓ Saved: {bcr_path}")

    # ========================================================================
    # 4. NET BENEFIT STATISTICS
    # ========================================================================
    print("  [4/5] Generating net benefit statistics...")

    net_benefit_stats = []

    for dev_id in all_developments:
        dev_data = df_combined[df_combined['development'] == dev_id]
        line_name = dev_data['line_name'].iloc[0] if 'line_name' in dev_data.columns else dev_id

        new_data = dev_data[dev_data['pipeline'] == 'new']
        old_data = dev_data[dev_data['pipeline'] == 'old']

        # Net benefit statistics
        mean_nb_new = new_data['total_net_benefit'].mean() if len(new_data) > 0 else 0
        mean_nb_old = old_data['total_net_benefit'].mean() if len(old_data) > 0 else 0
        median_nb_new = new_data['total_net_benefit'].median() if len(new_data) > 0 else 0
        median_nb_old = old_data['total_net_benefit'].median() if len(old_data) > 0 else 0

        total_nb_new = new_data['total_net_benefit'].sum() if len(new_data) > 0 else 0
        total_nb_old = old_data['total_net_benefit'].sum() if len(old_data) > 0 else 0

        positive_nb_new = (new_data['total_net_benefit'] > 0).sum() if len(new_data) > 0 else 0
        positive_nb_old = (old_data['total_net_benefit'] > 0).sum() if len(old_data) > 0 else 0

        negative_nb_new = (new_data['total_net_benefit'] < 0).sum() if len(new_data) > 0 else 0
        negative_nb_old = (old_data['total_net_benefit'] < 0).sum() if len(old_data) > 0 else 0

        net_benefit_stats.append({
            'development': dev_id,
            'line_name': line_name,
            'mean_net_benefit_old_chf': mean_nb_old,
            'mean_net_benefit_new_chf': mean_nb_new,
            'net_benefit_change_chf': mean_nb_new - mean_nb_old,
            'median_net_benefit_old_chf': median_nb_old,
            'median_net_benefit_new_chf': median_nb_new,
            'total_net_benefit_old_chf': total_nb_old,
            'total_net_benefit_new_chf': total_nb_new,
            'scenarios_positive_nb_old': int(positive_nb_old),
            'scenarios_positive_nb_new': int(positive_nb_new),
            'scenarios_negative_nb_old': int(negative_nb_old),
            'scenarios_negative_nb_new': int(negative_nb_new),
            'net_benefit_improved': mean_nb_new > mean_nb_old,
            'net_benefit_worsened': mean_nb_new < mean_nb_old,
            'net_benefit_unchanged': mean_nb_new == mean_nb_old
        })

    net_benefit_df = pd.DataFrame(net_benefit_stats)
    net_benefit_path = os.path.join(output_dir, "report_pipeline_net_benefit_statistics.csv")
    net_benefit_df.to_csv(net_benefit_path, index=False)
    print(f"      ✓ Saved: {net_benefit_path}")

    # ========================================================================
    # 5. TOP 10 PERFORMERS BY PIPELINE
    # ========================================================================
    print("  [5/5] Generating top 10 performers...")

    # Rank by mean net benefit for each pipeline
    new_rankings = df_new.groupby('development')['total_net_benefit'].mean().sort_values(ascending=False).head(10)
    old_rankings = df_old.groupby('development')['total_net_benefit'].mean().sort_values(ascending=False).head(10)

    top10_data = []

    # Add NEW pipeline top 10
    for rank, (dev_id, mean_nb) in enumerate(new_rankings.items(), 1):
        dev_data = df_new[df_new['development'] == dev_id]
        line_name = dev_data['line_name'].iloc[0] if 'line_name' in dev_data.columns else dev_id
        mean_cost = dev_data['total_costs'].mean()
        mean_bcr = dev_data['cba_ratio'].mean() if 'cba_ratio' in dev_data.columns else 0
        viable_scenarios = (dev_data['total_net_benefit'] > 0).sum()

        top10_data.append({
            'pipeline': 'new',
            'rank': rank,
            'development': dev_id,
            'line_name': line_name,
            'mean_net_benefit_chf': mean_nb,
            'mean_total_cost_chf': mean_cost,
            'mean_bcr': mean_bcr,
            'viable_scenarios': int(viable_scenarios)
        })

    # Add OLD pipeline top 10
    for rank, (dev_id, mean_nb) in enumerate(old_rankings.items(), 1):
        dev_data = df_old[df_old['development'] == dev_id]
        line_name = dev_data['line_name'].iloc[0] if 'line_name' in dev_data.columns else dev_id
        mean_cost = dev_data['total_costs'].mean()
        mean_bcr = dev_data['cba_ratio'].mean() if 'cba_ratio' in dev_data.columns else 0
        viable_scenarios = (dev_data['total_net_benefit'] > 0).sum()

        top10_data.append({
            'pipeline': 'old',
            'rank': rank,
            'development': dev_id,
            'line_name': line_name,
            'mean_net_benefit_chf': mean_nb,
            'mean_total_cost_chf': mean_cost,
            'mean_bcr': mean_bcr,
            'viable_scenarios': int(viable_scenarios)
        })

    top10_df = pd.DataFrame(top10_data)
    top10_path = os.path.join(output_dir, "report_pipeline_top10_performers.csv")
    top10_df.to_csv(top10_path, index=False)
    print(f"      ✓ Saved: {top10_path}")

    # ========================================================================
    # SUMMARY OUTPUT
    # ========================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPARISON REPORT STATISTICS - SUMMARY")
    print("="*80)
    print(f"\nGenerated 5 comprehensive report files in:")
    print(f"  {output_dir}")
    print(f"\nFiles created:")
    print(f"  1. report_pipeline_viability_statistics.csv")
    print(f"  2. report_pipeline_cost_statistics.csv")
    print(f"  3. report_pipeline_bcr_statistics.csv")
    print(f"  4. report_pipeline_net_benefit_statistics.csv")
    print(f"  5. report_pipeline_top10_performers.csv")

    # Summary statistics
    print(f"\nKey Findings:")
    print(f"  • Total developments analyzed: {len(all_developments)}")
    print(f"  • Developments with improved viability (NEW): {viability_df['viability_improved'].sum()}")
    print(f"  • Developments with reduced costs (NEW): {cost_df['cost_reduced'].sum()}")
    print(f"  • Developments with improved BCR (NEW): {bcr_df['bcr_improved'].sum()}")
    print(f"  • Developments with improved net benefit (NEW): {net_benefit_df['net_benefit_improved'].sum()}")
    print(f"  • Average cost change (NEW vs OLD): CHF {cost_df['total_cost_change_chf'].mean():,.0f}")
    print(f"  • Average BCR change (NEW vs OLD): {bcr_df['bcr_change'].mean():.3f}")
    print(f"  • Average net benefit change (NEW vs OLD): CHF {net_benefit_df['net_benefit_change_chf'].mean():,.0f}")
    print("="*80 + "\n")


if __name__ == '__main__':
    # Change to the infraScanRail directory
    os.chdir(paths.MAIN)

    # Export report statistics
    export_pipeline_comparison_report_statistics()

    print("\n✓ Report statistics export complete!\n")
