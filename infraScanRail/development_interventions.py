"""
Development-Intervention Mapping Module

Maps infrastructure developments to capacity interventions to calculate
full costs including baseline network enhancements.
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import List, Dict, Optional, Set
import re
import ast
import numpy as np


def extract_development_segments(dev_id: str, dev_gpkg_path: str) -> List[str]:
    """
    Extract all segment pairs for a development.
    
    Process:
    1. Load development .gpkg
    2. Parse Via column (handle "-99", "1,2,3", "[1,2,3]" formats)
    3. Build complete path: [FromNode] + via_nodes + [ToNode]
    4. Generate segment pairs: [(path[i], path[i+1]) for i in range(len(path)-1)]
    5. Normalize to "min-max" format for matching
    
    Args:
        dev_id: Development identifier
        dev_gpkg_path: Path to development .gpkg file
        
    Returns:
        List of segment pairs in "nodeA-nodeB" format (normalized min-max)
    """
    # Load development
    gdf = gpd.read_file(dev_gpkg_path)
    
    if gdf.empty:
        return []
    
    all_segments = []
    
    for _, row in gdf.iterrows():
        from_node = int(row['FromNode'])
        to_node = int(row['ToNode'])
        via_value = row.get('Via', '-99')
        
        # Parse via nodes
        via_nodes = _parse_via_column(via_value)
        
        # Build complete path
        if via_nodes:
            path = [from_node] + via_nodes + [to_node]
        else:
            path = [from_node, to_node]
        
        # Generate segment pairs
        for i in range(len(path) - 1):
            node_a = path[i]
            node_b = path[i + 1]
            
            # Normalize to min-max format
            segment = f"{min(node_a, node_b)}-{max(node_a, node_b)}"
            all_segments.append(segment)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_segments = []
    for seg in all_segments:
        if seg not in seen:
            seen.add(seg)
            unique_segments.append(seg)
    
    return unique_segments


def _parse_via_column(via_value) -> List[int]:
    """
    Parse Via column handling multiple formats and data types.
    
    Supported formats:
    - -99 or -1 (int/float) → empty list (direct connection)
    - "-99" or "-1" (string) → empty list
    - "1018,2298,2497" (string) → [1018, 2298, 2497]
    - "[1018, 2298, 2497]" (string) → [1018, 2298, 2497]
    - [1018, 2298, 2497] (list) → [1018, 2298, 2497]
    - "[]" or [] → empty list
    - nan/None → empty list
    
    Returns:
        List of intermediate node IDs
    """
    # Handle None/NaN/NaT
    if via_value is None or pd.isna(via_value):
        return []
    
    # Handle numpy/pandas numeric types
    if isinstance(via_value, (int, float, np.integer, np.floating)):
        if via_value in [-99, -1]:
            return []
        return [int(via_value)]
    
    # Handle list type (already parsed)
    if isinstance(via_value, list):
        if not via_value:
            return []
        return [int(n) for n in via_value if n not in [-99, -1]]
    
    # Convert to string for string-based parsing
    via_str = str(via_value).strip()
    
    # Handle empty/sentinel strings
    if via_str in ['', '-99', '-1', '[]', 'nan', 'None', 'NaN']:
        return []
    
    # Try parsing as list literal first
    if via_str.startswith('[') and via_str.endswith(']'):
        try:
            nodes = ast.literal_eval(via_str)
            if not nodes:
                return []
            return [int(n) for n in nodes if int(n) not in [-99, -1]]
        except (ValueError, SyntaxError, TypeError):
            # Strip brackets and try comma-separated
            via_str = via_str[1:-1].strip()
    
    # Parse comma-separated string
    if ',' in via_str:
        try:
            nodes = [int(n.strip()) for n in via_str.split(',') if n.strip()]
            return [n for n in nodes if n not in [-99, -1]]
        except ValueError:
            pass
    
    # Try single node
    try:
        node = int(float(via_str))  # Handle "1234.0" format
        return [node] if node not in [-99, -1] else []
    except (ValueError, TypeError):
        return []


def map_development_to_interventions(
    dev_segments: List[str],
    interventions_csv: str
) -> pd.DataFrame:
    """
    Identify which Phase 4 interventions a development uses.
    
    Process:
    1. Load interventions CSV
    2. Parse 'affected_segments' column (pipe-separated: "A-B | C-D | E-F")
    3. For each intervention, check if ANY dev_segment matches ANY affected_segment
    4. Return filtered interventions DataFrame
    
    Args:
        dev_segments: List of segment pairs from development
        interventions_csv: Path to capacity_interventions.csv
        
    Returns:
        DataFrame with matching interventions
    """
    # Load interventions
    df = pd.read_csv(interventions_csv)
    
    if df.empty or not dev_segments:
        return pd.DataFrame()
    
    # Parse affected_segments column
    def parse_affected_segments(seg_str):
        """Parse pipe-separated segments into list."""
        if pd.isna(seg_str) or seg_str == '':
            return []
        
        segments = [s.strip() for s in str(seg_str).split('|')]
        
        # Normalize each segment to min-max format
        normalized = []
        for seg in segments:
            if '-' in seg:
                parts = seg.split('-')
                if len(parts) == 2:
                    try:
                        a, b = int(parts[0]), int(parts[1])
                        normalized.append(f"{min(a, b)}-{max(a, b)}")
                    except ValueError:
                        pass
        
        return normalized
    
    df['affected_segments_list'] = df['affected_segments'].apply(parse_affected_segments)
    
    # Convert dev_segments to set for fast lookup
    dev_segments_set = set(dev_segments)
    
    # Find matches
    def has_matching_segment(segments_list):
        """Check if any segment matches development segments."""
        return any(seg in dev_segments_set for seg in segments_list)
    
    matching_mask = df['affected_segments_list'].apply(has_matching_segment)
    
    return df[matching_mask].copy()


def calculate_intervention_costs_per_development(
    dev_id_lookup: pd.DataFrame,
    baseline_interventions_path: str,
    capacity_analysis_results: Dict,
    development_directory: str
) -> pd.DataFrame:
    """
    Calculate intervention costs for all developments.
    
    Cost Attribution Rule:
    Each development independently pays FULL cost of ALL interventions it uses
    (no cost sharing between developments).
    
    Args:
        dev_id_lookup: DataFrame with dev_id column
        baseline_interventions_path: Path to Phase 4 interventions CSV
        capacity_analysis_results: Results from Phase 3.5 capacity analysis
        development_directory: Path to developments folder
        
    Returns:
        DataFrame with columns:
        - dev_id
        - intervention_construction_cost
        - intervention_maintenance_annual
        - intervention_count
        - intervention_ids (comma-separated)
    """
    results = []
    
    for _, row in dev_id_lookup.iterrows():
        dev_id = row['dev_id']
        dev_path = Path(development_directory) / f"{dev_id}.gpkg"
        
        if not dev_path.exists():
            print(f"  ⚠ Development file not found: {dev_path}")
            results.append({
                'dev_id': dev_id,
                'intervention_construction_cost': 0.0,
                'intervention_maintenance_annual': 0.0,
                'intervention_count': 0,
                'intervention_ids': ''
            })
            continue
        
        try:
            # Extract development segments
            dev_segments = extract_development_segments(dev_id, str(dev_path))
            
            # Map to interventions
            interventions_df = map_development_to_interventions(
                dev_segments,
                baseline_interventions_path
            )
            
            # Calculate costs
            if interventions_df.empty:
                construction_cost = 0.0
                maintenance_cost = 0.0
                intervention_count = 0
                intervention_ids = ''
            else:
                # Deduplicate by intervention_id
                interventions_df = interventions_df.drop_duplicates(subset=['intervention_id'])
                
                construction_cost = interventions_df['construction_cost_chf'].sum()
                maintenance_cost = interventions_df['maintenance_cost_annual_chf'].sum()
                intervention_count = len(interventions_df)
                intervention_ids = ','.join(interventions_df['intervention_id'].tolist())
            
            results.append({
                'dev_id': dev_id,
                'intervention_construction_cost': construction_cost,
                'intervention_maintenance_annual': maintenance_cost,
                'intervention_count': intervention_count,
                'intervention_ids': intervention_ids
            })
            
        except Exception as e:
            print(f"  ❌ Error processing {dev_id}: {e}")
            results.append({
                'dev_id': dev_id,
                'intervention_construction_cost': 0.0,
                'intervention_maintenance_annual': 0.0,
                'intervention_count': 0,
                'intervention_ids': ''
            })
    
    return pd.DataFrame(results)


def generate_dev_intervention_report(
    dev_id: str,
    dev_segments: List[str],
    interventions_df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Generate detailed report of interventions used by a development.
    
    Args:
        dev_id: Development identifier
        dev_segments: List of segment pairs
        interventions_df: Matching interventions DataFrame
        output_path: Path to save report CSV
    """
    if interventions_df.empty:
        print(f"  No interventions found for {dev_id}")
        return
    
    # Create report DataFrame
    report = interventions_df[[
        'intervention_id',
        'section_id',
        'type',
        'tracks_added',
        'construction_cost_chf',
        'maintenance_cost_annual_chf',
        'affected_segments'
    ]].copy()
    
    # Add development info
    report.insert(0, 'dev_id', dev_id)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    
    print(f"  ✓ Intervention report saved: {output_path}")
