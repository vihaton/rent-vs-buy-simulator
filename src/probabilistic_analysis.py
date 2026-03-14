"""
Probabilistic Analysis Module

This module provides functions for analyzing and interpreting results
from probabilistic mortgage comparisons.
"""

from scipy import stats
from scipy.stats import norm

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


def calculate_comparative_metrics(
    results_df: pd.DataFrame,
    metric: str = "wealth_end"
) -> pd.DataFrame:
    """
    Calculate P(scenario_i > scenario_j) for all pairs of scenarios.
    
    This function computes the probability that one mortgage scenario
    outperforms another based on the specified metric. The probabilities
    are calculated by comparing outcomes across the same world scenarios
    (matched by world_scenario_id).
    
    Args:
        results_df: Results DataFrame from run_multi_scenario_comparison()
            Must contain columns: mortgage_scenario_label, world_scenario_id, and metric
        metric: Name of metric to compare (default: "wealth_end")
    
    Returns:
        DataFrame with scenarios as both rows and columns, where
        cell [i, j] = P(scenario_i > scenario_j)
        Diagonal values are 0.5 (probability of tie with self)
    
    Example:
        >>> df = pd.read_csv("outputs/probabilistic/comparison.csv")
        >>> comp = calculate_comparative_metrics(df, "wealth_end")
        >>> print(comp)
                           Scenario_A  Scenario_B  Scenario_C
        Scenario_A              0.50        0.62        0.58
        Scenario_B              0.38        0.50        0.45
        Scenario_C              0.42        0.55        0.50
        >>> # Scenario A beats Scenario B 62% of the time
    
    Raises:
        ValueError: If metric column doesn't exist or data is invalid
    """
    if metric not in results_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in results. "
            f"Available columns: {list(results_df.columns)}"
        )
    
    if 'mortgage_scenario_label' not in results_df.columns:
        raise ValueError("Results must contain 'mortgage_scenario_label' column")
    
    if 'world_scenario_id' not in results_df.columns:
        raise ValueError("Results must contain 'world_scenario_id' column")
    
    scenarios = sorted(results_df['mortgage_scenario_label'].unique())
    n_scenarios = len(scenarios)
    
    # Create comparison matrix
    comparison_matrix = pd.DataFrame(
        index=scenarios,
        columns=scenarios,
        dtype=float
    )
    
    for i, scenario_i in enumerate(scenarios):
        for j, scenario_j in enumerate(scenarios):
            if i == j:
                # Probability of beating yourself is 0.5
                comparison_matrix.loc[scenario_i, scenario_j] = 0.5
                continue
            
            # Get values for each scenario, aligned by world_scenario_id
            df_i = results_df[results_df['mortgage_scenario_label'] == scenario_i].sort_values('world_scenario_id')
            df_j = results_df[results_df['mortgage_scenario_label'] == scenario_j].sort_values('world_scenario_id')
            
            # Verify same number of samples
            if len(df_i) != len(df_j):
                raise ValueError(
                    f"Scenarios have different number of samples: "
                    f"{scenario_i} ({len(df_i)}) vs {scenario_j} ({len(df_j)})"
                )
            
            # Calculate probability i > j
            prob = (df_i[metric].values > df_j[metric].values).mean()
            comparison_matrix.loc[scenario_i, scenario_j] = prob
    
    return comparison_matrix


def calculate_risk_metrics(
    results_df: pd.DataFrame,
    scenario_label: str,
    metrics: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate risk metrics (percentiles and statistics) for a scenario.
    
    Args:
        results_df: Results DataFrame from run_multi_scenario_comparison()
        scenario_label: Name of scenario to analyze
        metrics: List of metric names to analyze
    
    Returns:
        Dictionary mapping metric names to their statistics:
            - mean: Expected value
            - std: Standard deviation
            - min: Minimum value
            - max: Maximum value
            - p10: 10th percentile (downside risk)
            - p25: 25th percentile
            - p50: Median
            - p75: 75th percentile
            - p90: 90th percentile (upside potential)
    
    Example:
        >>> df = pd.read_csv("outputs/probabilistic/comparison.csv")
        >>> risk = calculate_risk_metrics(df, "Scenario_A", ["wealth_end", "avg_net_cost_per_month"])
        >>> print(f"Expected wealth: €{risk['wealth_end']['mean']:,.0f}")
        >>> print(f"Downside (P10): €{risk['wealth_end']['p10']:,.0f}")
        >>> print(f"Upside (P90): €{risk['wealth_end']['p90']:,.0f}")
    
    Raises:
        ValueError: If scenario not found or metrics don't exist
    """
    if scenario_label not in results_df['mortgage_scenario_label'].values:
        available = results_df['mortgage_scenario_label'].unique()
        raise ValueError(
            f"Scenario '{scenario_label}' not found. "
            f"Available scenarios: {list(available)}"
        )
    
    scenario_df = results_df[results_df['mortgage_scenario_label'] == scenario_label]
    
    risk_metrics = {}
    for metric in metrics:
        if metric not in scenario_df.columns:
            raise ValueError(
                f"Metric '{metric}' not found in results. "
                f"Available columns: {list(scenario_df.columns)}"
            )
        
        values = scenario_df[metric]
        risk_metrics[metric] = {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'min': float(values.min()),
            'max': float(values.max()),
            'p10': float(values.quantile(0.10)),
            'p25': float(values.quantile(0.25)),
            'p50': float(values.quantile(0.50)),
            'p75': float(values.quantile(0.75)),
            'p90': float(values.quantile(0.90)),
        }
    
    return risk_metrics


def calculate_regret_metrics(
    results_df: pd.DataFrame,
    metric: str = "wealth_end"
) -> pd.DataFrame:
    """
    Calculate regret: difference from best choice in each world scenario.
    
    Regret measures the opportunity cost of choosing a particular mortgage.
    For each world scenario, we identify the best mortgage and calculate
    how much worse each other mortgage performed.
    
    Args:
        results_df: Results DataFrame from run_multi_scenario_comparison()
        metric: Metric to use for regret calculation (default: "wealth_end")
    
    Returns:
        DataFrame with regret statistics for each scenario:
            - mean_regret: Average opportunity cost
            - max_regret: Worst-case opportunity cost
            - prob_optimal: Probability this mortgage is the best choice
            - p90_regret: 90th percentile regret (typical bad outcome)
    
    Example:
        >>> df = pd.read_csv("outputs/probabilistic/comparison.csv")
        >>> regret = calculate_regret_metrics(df, "wealth_end")
        >>> print(regret)
                           mean_regret  max_regret  prob_optimal  p90_regret
        Scenario_A             5234.2      45231.1          0.42     12456.3
        Scenario_B             8123.4      52341.2          0.28     18234.1
        Scenario_C             6234.1      48123.4          0.30     14567.2
    
    Raises:
        ValueError: If metric doesn't exist or data is invalid
    """
    if metric not in results_df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in results. "
            f"Available columns: {list(results_df.columns)}"
        )
    
    if 'world_scenario_id' not in results_df.columns:
        raise ValueError("Results must contain 'world_scenario_id' column")
    
    # For each world scenario, find the best mortgage
    best_per_world = results_df.groupby('world_scenario_id')[metric].max()
    
    # Calculate regret for each scenario
    regret_stats = []
    for scenario in sorted(results_df['mortgage_scenario_label'].unique()):
        scenario_df = results_df[results_df['mortgage_scenario_label'] == scenario].copy()
        scenario_df = scenario_df.sort_values('world_scenario_id')
        
        # Regret = best_value - actual_value
        scenario_df['regret'] = best_per_world.values - scenario_df[metric].values
        
        regret_stats.append({
            'mortgage_scenario_label': scenario,
            'mean_regret': scenario_df['regret'].mean(),
            'max_regret': scenario_df['regret'].max(),
            'prob_optimal': (scenario_df['regret'] == 0).mean(),
            'p90_regret': scenario_df['regret'].quantile(0.90)
        })
    
    return pd.DataFrame(regret_stats)


def identify_regime(
    value: float,
    components: List[Dict[str, float]]
) -> int:
    """
    Identify most likely regime for a sampled value from mixture distribution.
    
    Uses Bayes' rule: P(regime|value) ∝ P(value|regime) * P(regime)
    
    Args:
        value: Sampled value
        components: List of mixture components, each with:
            - mean: Mean of the component
            - std: Standard deviation of the component
            - weight: Weight of the component
    
    Returns:
        Index of most likely regime (0-indexed)
    
    Example:
        >>> components = [
        ...     {'mean': 0.035, 'std': 0.07, 'weight': 0.50},  # Normal growth
        ...     {'mean': 0.000, 'std': 0.06, 'weight': 0.30},  # Stagnation
        ...     {'mean': -0.050, 'std': 0.11, 'weight': 0.20}  # Crisis
        ... ]
        >>> regime = identify_regime(-0.08, components)
        >>> print(regime)  # 2 (crisis regime)
    """
    
    likelihoods = []
    for component in components:
        # P(value|regime) * P(regime)
        likelihood = norm.pdf(value, component['mean'], component['std'])
        posterior = likelihood * component['weight']
        likelihoods.append(posterior)
    
    # Return most likely regime
    return int(np.argmax(likelihoods))


def add_regime_labels(
    results_df: pd.DataFrame,
    variable_name: str,
    components: List[Dict[str, Any]],
    regime_column_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Add regime identification columns to results DataFrame.
    
    Args:
        results_df: Results DataFrame from run_multi_scenario_comparison()
        variable_name: Name of variable column to identify regimes for
        components: List of mixture components with mean, std, weight, and optional label
        regime_column_name: Name for regime column (default: f"{variable_name}_regime")
    
    Returns:
        DataFrame with added columns:
            - {variable_name}_regime: Regime index (0, 1, 2, ...)
            - {variable_name}_regime_label: Regime label (if provided in components)
    
    Example:
        >>> df = pd.read_csv("outputs/probabilistic/comparison.csv")
        >>> components = [
        ...     {'mean': 0.035, 'std': 0.07, 'weight': 0.50, 'label': 'Normal growth'},
        ...     {'mean': 0.000, 'std': 0.06, 'weight': 0.30, 'label': 'Stagnation'},
        ...     {'mean': -0.050, 'std': 0.11, 'weight': 0.20, 'label': 'Crisis'}
        ... ]
        >>> df = add_regime_labels(df, 'annual_value_growth', components)
        >>> print(df.groupby('annual_value_growth_regime_label')['wealth_end'].describe())
    """
    if variable_name not in results_df.columns:
        raise ValueError(
            f"Variable '{variable_name}' not found in results. "
            f"Available columns: {list(results_df.columns)}"
        )
    
    df = results_df.copy()
    
    # Determine column name
    if regime_column_name is None:
        regime_column_name = f"{variable_name}_regime"
    
    # Identify regime for each value
    df[regime_column_name] = df[variable_name].apply(
        lambda x: identify_regime(x, components)
    )
    
    # Add regime labels if provided
    if all('label' in c for c in components):
        label_map = {i: c['label'] for i, c in enumerate(components)}
        df[f"{regime_column_name}_label"] = df[regime_column_name].map(label_map)
    
    return df


def validate_mixture_distribution(
    results_df: pd.DataFrame,
    variable_name: str,
    components: List[Dict[str, float]],
    bins: int = 50
) -> Dict[str, Any]:
    """
    Validate that sampled values match the intended mixture distribution.
    
    Args:
        results_df: Results DataFrame with sampled values
        variable_name: Name of variable to validate
        components: Expected mixture components
        bins: Number of bins for histogram
    
    Returns:
        Dictionary with validation statistics:
            - observed_mean: Mean of sampled values
            - expected_mean: Expected mean from mixture
            - observed_std: Std of sampled values
            - expected_std: Expected std from mixture
            - ks_statistic: Kolmogorov-Smirnov test statistic
            - ks_pvalue: P-value for KS test
    
    Example:
        >>> df = pd.read_csv("outputs/probabilistic/comparison.csv")
        >>> components = [
        ...     {'mean': 0.035, 'std': 0.07, 'weight': 0.50},
        ...     {'mean': 0.000, 'std': 0.06, 'weight': 0.30},
        ...     {'mean': -0.050, 'std': 0.11, 'weight': 0.20}
        ... ]
        >>> validation = validate_mixture_distribution(df, 'annual_value_growth', components)
        >>> print(f"Observed mean: {validation['observed_mean']:.4f}")
        >>> print(f"Expected mean: {validation['expected_mean']:.4f}")
    """
    
    if variable_name not in results_df.columns:
        raise ValueError(
            f"Variable '{variable_name}' not found in results. "
            f"Available columns: {list(results_df.columns)}"
        )
    
    # Get unique values (remove duplicates from multiple scenarios)
    values = results_df[variable_name].unique()
    
    # Calculate observed statistics
    observed_mean = float(np.mean(values))
    observed_std = float(np.std(values))
    
    # Calculate expected statistics from mixture
    expected_mean = sum(c['mean'] * c['weight'] for c in components)
    expected_var = sum(
        c['weight'] * (c['std']**2 + c['mean']**2)
        for c in components
    ) - expected_mean**2
    expected_std = np.sqrt(expected_var)
    
    # Generate samples from expected mixture for KS test
    n_samples = len(values)
    rng = np.random.default_rng(42)
    
    # Sample regime indices
    weights = [c['weight'] for c in components]
    regime_indices = rng.choice(len(components), size=n_samples, p=weights)
    
    # Sample from each regime
    expected_samples = np.zeros(n_samples)
    for i, component in enumerate(components):
        mask = regime_indices == i
        n_regime = mask.sum()
        if n_regime > 0:
            expected_samples[mask] = rng.normal(
                component['mean'],
                component['std'],
                n_regime
            )
    
    # Kolmogorov-Smirnov test
    ks_statistic, ks_pvalue = stats.ks_2samp(values, expected_samples)
    
    return {
        'observed_mean': observed_mean,
        'expected_mean': expected_mean,
        'observed_std': observed_std,
        'expected_std': expected_std,
        'ks_statistic': float(ks_statistic),
        'ks_pvalue': float(ks_pvalue)
    }


def summarize_comparison(
    results_df: pd.DataFrame,
    metric: str = "wealth_end"
) -> pd.DataFrame:
    """
    Generate comprehensive summary of scenario comparison.
    
    Args:
        results_df: Results DataFrame from run_multi_scenario_comparison()
        metric: Primary metric for comparison (default: "wealth_end")
    
    Returns:
        DataFrame with summary statistics for each scenario
    
    Example:
        >>> df = pd.read_csv("outputs/probabilistic/comparison.csv")
        >>> summary = summarize_comparison(df, "wealth_end")
        >>> print(summary)
    """
    scenarios = sorted(results_df['mortgage_scenario_label'].unique())
    
    summary_data = []
    for scenario in scenarios:
        scenario_df = results_df[results_df['mortgage_scenario_label'] == scenario]
        values = scenario_df[metric]
        
        summary_data.append({
            'scenario': scenario,
            'mean': values.mean(),
            'std': values.std(),
            'min': values.min(),
            'p10': values.quantile(0.10),
            'p25': values.quantile(0.25),
            'median': values.quantile(0.50),
            'p75': values.quantile(0.75),
            'p90': values.quantile(0.90),
            'max': values.max(),
            'samples': len(values)
        })
    
    return pd.DataFrame(summary_data)
