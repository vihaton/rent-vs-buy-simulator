"""
Comparison Metrics Module

This module provides functions for calculating comparative statistics
between multiple mortgage scenarios, including:
- Pairwise comparison probabilities P(A > B)
- Dominance analysis
- Regime-conditional performance
- Exit timing analysis
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def calculate_comparative_probabilities(
    summary_df: pd.DataFrame,
    metric: str = 'final_wealth'
) -> pd.DataFrame:
    """
    Calculate P(scenario_i > scenario_j) for all pairs.
    
    This answers: "What is the probability that scenario A outperforms
    scenario B?" by comparing matched samples (same sample_id).
    
    Args:
        summary_df: Summary DataFrame with columns [sample_id, scenario_label, metric]
        metric: Metric to compare (default: 'final_wealth')
    
    Returns:
        DataFrame with scenarios as rows and columns, values are probabilities
        
    Example:
        >>> probs = calculate_comparative_probabilities(summary_df)
        >>> print(probs)
                  Bank1  Bank2    Bank3
        Bank1      0.50     0.53   0.48
        Bank2      0.47     0.50   0.44
        Bank3      0.52     0.56   0.50
    """
    scenarios = sorted(summary_df['scenario_label'].unique())
    n_scenarios = len(scenarios)
    prob_matrix = np.zeros((n_scenarios, n_scenarios))
    
    for i, scenario_i in enumerate(scenarios):
        for j, scenario_j in enumerate(scenarios):
            if i == j:
                prob_matrix[i, j] = 0.5
            else:
                # Get matched samples (same sample_id)
                df_i = summary_df[summary_df['scenario_label'] == scenario_i][['sample_id', metric]]
                df_j = summary_df[summary_df['scenario_label'] == scenario_j][['sample_id', metric]]
                
                merged = df_i.merge(df_j, on='sample_id', suffixes=('_i', '_j'))
                
                prob_matrix[i, j] = (merged[f'{metric}_i'] > merged[f'{metric}_j']).mean()
    
    return pd.DataFrame(
        prob_matrix,
        index=scenarios,
        columns=scenarios
    )


def calculate_pairwise_statistics(
    summary_df: pd.DataFrame,
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate detailed pairwise comparison statistics.
    
    Args:
        summary_df: Summary DataFrame
        metrics: List of metrics to compare (default: ['final_wealth'])
    
    Returns:
        DataFrame with columns:
            - scenario_a
            - scenario_b
            - metric
            - p_a_gt_b: P(metric_a > metric_b)
            - mean_diff: Mean(metric_a - metric_b)
            - median_diff: Median(metric_a - metric_b)
            - p10_diff: 10th percentile of difference
            - p90_diff: 90th percentile of difference
            - correlation: Correlation between metric_a and metric_b
    """
    if metrics is None:
        metrics = ['final_wealth']
    
    scenarios = sorted(summary_df['scenario_label'].unique())
    results = []
    
    for metric in metrics:
        for scenario_a in scenarios:
            for scenario_b in scenarios:
                if scenario_a == scenario_b:
                    continue
                
                # Get matched samples
                df_a = summary_df[summary_df['scenario_label'] == scenario_a][['sample_id', metric]]
                df_b = summary_df[summary_df['scenario_label'] == scenario_b][['sample_id', metric]]
                
                merged = df_a.merge(df_b, on='sample_id', suffixes=('_a', '_b'))
                
                diff = merged[f'{metric}_a'] - merged[f'{metric}_b']
                
                results.append({
                    'scenario_a': scenario_a,
                    'scenario_b': scenario_b,
                    'metric': metric,
                    'p_a_gt_b': (diff > 0).mean(),
                    'mean_diff': diff.mean(),
                    'median_diff': diff.median(),
                    'p10_diff': diff.quantile(0.1),
                    'p90_diff': diff.quantile(0.9),
                    'correlation': merged[f'{metric}_a'].corr(merged[f'{metric}_b'])
                })
    
    return pd.DataFrame(results)


def calculate_dominance_statistics(
    summary_df: pd.DataFrame,
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate how often each scenario dominates (best in all metrics).
    
    Args:
        summary_df: Summary DataFrame
        metrics: List of metrics to consider (default: ['final_wealth'])
    
    Returns:
        DataFrame with columns:
            - scenario_label
            - dominance_count: Number of samples where this scenario is best in all metrics
            - dominance_rate: Proportion of samples where this scenario dominates
    """
    if metrics is None:
        metrics = ['final_wealth']
    
    scenarios = sorted(summary_df['scenario_label'].unique())
    n_samples = summary_df['sample_id'].nunique()
    
    dominance_counts = {scenario: 0 for scenario in scenarios}
    
    for sample_id in summary_df['sample_id'].unique():
        sample_data = summary_df[summary_df['sample_id'] == sample_id]
        
        # Check which scenario is best in each metric
        is_dominant = {scenario: True for scenario in scenarios}
        
        for metric in metrics:
            best_value = sample_data[metric].max()
            best_scenarios = sample_data[sample_data[metric] == best_value]['scenario_label'].values
            
            # Mark non-best scenarios as not dominant
            for scenario in scenarios:
                if scenario not in best_scenarios:
                    is_dominant[scenario] = False
        
        # Count dominant scenarios
        for scenario in scenarios:
            if is_dominant[scenario]:
                dominance_counts[scenario] += 1
    
    results = []
    for scenario in scenarios:
        results.append({
            'scenario_label': scenario,
            'dominance_count': dominance_counts[scenario],
            'dominance_rate': dominance_counts[scenario] / n_samples
        })
    
    return pd.DataFrame(results)


def calculate_regime_conditional_performance(
    summary_df: pd.DataFrame,
    regime_column: str = 'years_in_crisis',
    bins: Optional[List[int]] = None,
    metric: str = 'final_wealth'
) -> pd.DataFrame:
    """
    Calculate performance statistics conditional on regime exposure.
    
    Args:
        summary_df: Summary DataFrame
        regime_column: Column to condition on (e.g., 'years_in_crisis')
        bins: Bin edges for regime exposure (default: [0, 2, 5, 8, 30])
        metric: Metric to analyze (default: 'final_wealth')
    
    Returns:
        DataFrame with columns:
            - scenario_label
            - regime_bin: Regime exposure bin (e.g., "0-2 years")
            - n_samples: Number of samples in this bin
            - mean: Mean metric value
            - p10: 10th percentile
            - p50: Median
            - p90: 90th percentile
    """
    if bins is None:
        bins = [0, 2, 5, 8, 30]
    
    # Create regime bins
    summary_df = summary_df.copy()
    summary_df['regime_bin'] = pd.cut(
        summary_df[regime_column],
        bins=bins,
        labels=[f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)],
        include_lowest=True
    )
    
    # Calculate statistics for each scenario and bin
    results = []
    for scenario in summary_df['scenario_label'].unique():
        scenario_data = summary_df[summary_df['scenario_label'] == scenario]
        
        for bin_label in scenario_data['regime_bin'].unique():
            if pd.isna(bin_label):
                continue
            
            bin_data = scenario_data[scenario_data['regime_bin'] == bin_label]
            
            results.append({
                'scenario_label': scenario,
                'regime_bin': bin_label,
                'n_samples': len(bin_data),
                'mean': bin_data[metric].mean(),
                'p10': bin_data[metric].quantile(0.1),
                'p50': bin_data[metric].median(),
                'p90': bin_data[metric].quantile(0.9)
            })
    
    return pd.DataFrame(results)


def calculate_exit_timing_analysis(
    summary_df: pd.DataFrame,
    years: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Calculate wealth statistics at different exit years.
    
    Args:
        summary_df: Summary DataFrame with wealth_year_X columns
        years: Years to analyze (default: auto-detect from available columns,
               or [5, 10, 15, 20, 25, 30] if none specified)
    
    Returns:
        DataFrame with columns:
            - scenario_label
            - year
            - mean_wealth
            - p10_wealth
            - p50_wealth
            - p90_wealth
            - prob_negative: Probability of negative wealth
        
        Returns empty DataFrame if no wealth_year_X columns are found.
    """
    # Auto-detect available years from columns
    wealth_cols = [col for col in summary_df.columns if col.startswith('wealth_year_')]
    available_years = sorted([int(col.split('_')[-1]) for col in wealth_cols])
    
    if years is None:
        if available_years:
            # Use available years, but filter to common checkpoints if possible
            common_checkpoints = [5, 10, 15, 20, 25, 30]
            years = [y for y in common_checkpoints if y in available_years]
            # If no common checkpoints match, use all available years
            if not years:
                years = available_years
        else:
            # No wealth columns found, return empty DataFrame
            return pd.DataFrame()
    
    results = []
    for scenario in summary_df['scenario_label'].unique():
        scenario_data = summary_df[summary_df['scenario_label'] == scenario]
        
        for year in years:
            col_name = f'wealth_year_{year}'
            if col_name not in scenario_data.columns:
                continue
            
            wealth = scenario_data[col_name]
            
            results.append({
                'scenario_label': scenario,
                'year': year,
                'mean_wealth': wealth.mean(),
                'p10_wealth': wealth.quantile(0.1),
                'p50_wealth': wealth.median(),
                'p90_wealth': wealth.quantile(0.9),
                'prob_negative': (wealth < 0).mean()
            })
    
    return pd.DataFrame(results)


def export_regime_statistics(
    summary_df: pd.DataFrame,
    output_path: str
) -> None:
    """
    Export regime-level summary statistics to CSV.
    
    Args:
        summary_df: Summary DataFrame
        output_path: Output file path
    """
    regime_cols = ['years_in_normal', 'years_in_stagnation', 'years_in_crisis', 'years_in_recovery']
    
    results = []
    for scenario in summary_df['scenario_label'].unique():
        scenario_data = summary_df[summary_df['scenario_label'] == scenario]
        
        for regime_col in regime_cols:
            regime_name = regime_col.replace('years_in_', '').title()
            
            results.append({
                'scenario_label': scenario,
                'regime': regime_name,
                'mean_years': scenario_data[regime_col].mean(),
                'std_years': scenario_data[regime_col].std(),
                'min_years': scenario_data[regime_col].min(),
                'max_years': scenario_data[regime_col].max()
            })
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
