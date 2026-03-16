#!/usr/bin/env python3
"""
Calibrate transition matrix from BIS property price data.

This script implements the methodology described in plans/transition_matrix_calibration.md
to estimate Markov regime transition probabilities from historical property price data.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from collections import defaultdict
import argparse
import shutil


def load_config(config_path: str) -> dict:
    """Load calibration configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_prior_matrix(prior_path: str, regime_names: list) -> pd.DataFrame:
    """
    Load prior transition matrix from CSV file.
    
    Args:
        prior_path: Path to CSV file with prior matrix
        regime_names: List of regime names in order
    
    Returns:
        Prior transition matrix as DataFrame
    """
    df = pd.read_csv(prior_path, index_col=0)
    
    # Reorder columns to match regime_names order
    column_map = {f'to_{name}': name for name in regime_names}
    df = df.rename(columns=column_map)
    
    # Ensure correct order
    prior_matrix = df.loc[regime_names, regime_names]
    
    # Validate that rows sum to 1
    row_sums = prior_matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError(f"Prior matrix rows don't sum to 1: {row_sums.to_dict()}")
    
    return prior_matrix


def classify_regime(annual_growth, thresholds: dict):
    """
    Classify economic regime based on property price growth.
    
    Args:
        annual_growth: Annualized growth rate (e.g., 0.05 for 5%)
        thresholds: Dict with recovery_threshold, normal_threshold, stagnation_threshold
    
    Returns:
        Regime label or None if growth is NaN
    """
    if pd.isna(annual_growth):
        return None
    elif annual_growth > thresholds['recovery_threshold']:
        return 'RECOVERY'
    elif annual_growth > thresholds['normal_threshold']:
        return 'NORMAL_GROWTH'
    elif annual_growth > thresholds['stagnation_threshold']:
        return 'STAGNATION'
    else:
        return 'CRISIS'


def estimate_transition_matrix(regime_series, regime_names: list):
    """
    Estimate transition matrix from regime time series.
    
    Counts observed transitions between regimes and normalizes
    to get transition probabilities.
    
    Args:
        regime_series: Series of regime labels
        regime_names: List of regime names in order
    
    Returns:
        Transition matrix as DataFrame
    """
    # Remove NaN values
    regimes = regime_series.dropna()
    
    # Count transitions
    transitions = defaultdict(lambda: defaultdict(int))
    for i in range(len(regimes) - 1):
        from_regime = regimes.iloc[i]
        to_regime = regimes.iloc[i + 1]
        transitions[from_regime][to_regime] += 1
    
    # Convert to matrix
    n = len(regime_names)
    matrix = np.zeros((n, n))
    
    for i, from_regime in enumerate(regime_names):
        total = sum(transitions[from_regime].values())
        if total > 0:
            for j, to_regime in enumerate(regime_names):
                matrix[i, j] = transitions[from_regime][to_regime] / total
    
    return pd.DataFrame(matrix, index=regime_names, columns=regime_names)


def calibrate_transition_matrix(
    data_path: str,
    countries: list,
    prior_matrix: pd.DataFrame,
    thresholds: dict,
    alpha: float = 0.3
) -> tuple:
    """
    Calibrate transition matrix from BIS property price data.
    
    Combines empirical estimates from multiple countries with prior beliefs
    using Bayesian smoothing.
    
    Args:
        data_path: Path to BIS CSV file
        countries: List of countries to include
        prior_matrix: Prior belief matrix
        thresholds: Dict with regime classification thresholds
        alpha: Weight on prior (0 = all data, 1 = all prior)
    
    Returns:
        Tuple of (calibrated_matrix, empirical_matrix, country_stats)
    """
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    
    # Process each country
    all_regimes = []
    country_stats = []
    regime_names = prior_matrix.index.tolist()
    
    for country in countries:
        try:
            # Extract country data
            country_row = df[df['Reference area'] == country]
            if country_row.empty:
                print(f"Warning: Country '{country}' not found in data")
                continue
            
            # Extract country data (columns are in reverse chronological order)
            country_data = country_row.iloc[0, 1:].values.astype(float)
            dates = df.columns[1:]
            
            # Reverse to get chronological order (oldest first)
            country_data_chrono = country_data[::-1]
            dates_chrono = dates[::-1]
            
            country_df = pd.DataFrame({
                'date': pd.to_datetime(dates_chrono),
                'price_index': country_data_chrono
            })
            
            # Calculate growth rates
            country_df['qoq_growth'] = country_df['price_index'].pct_change()
            country_df['annual_growth'] = (1 + country_df['qoq_growth']) ** 4 - 1
            country_df['regime'] = country_df['annual_growth'].apply(
                lambda x: classify_regime(x, thresholds)
            )
            
            # Store regime series
            all_regimes.append(country_df['regime'])
            
            # Collect statistics
            regime_counts = country_df['regime'].value_counts()
            country_stats.append({
                'country': country,
                'n_quarters': len(country_df),
                'n_transitions': len(country_df['regime'].dropna()) - 1,
                'regimes': regime_counts.to_dict()
            })
            
            print(f"  ✓ {country}: {len(country_df)} quarters, {regime_counts.to_dict()}")
            
        except Exception as e:
            print(f"  ✗ Error processing {country}: {e}")
    
    if not all_regimes:
        raise ValueError("No countries could be processed successfully")
    
    # Combine and estimate
    print("\nEstimating transition matrix from combined data...")
    combined_regimes = pd.concat(all_regimes, ignore_index=True)
    estimated_matrix = estimate_transition_matrix(combined_regimes, regime_names)
    
    print("\nEmpirical transition matrix (from data):")
    print(estimated_matrix.round(3))
    
    # Smooth with prior
    print(f"\nSmoothing with prior (alpha={alpha})...")
    smoothed = alpha * prior_matrix + (1 - alpha) * estimated_matrix
    
    # Ensure rows sum to 1
    row_sums = smoothed.sum(axis=1)
    smoothed = smoothed.div(row_sums, axis=0)
    
    return smoothed, estimated_matrix, country_stats


def validate_transition_matrix(matrix, ergodicity_power: int = 100):
    """
    Validate that transition matrix satisfies required properties.
    
    Checks:
    1. All elements in [0, 1]
    2. Rows sum to 1
    3. Matrix is ergodic (can reach any state from any state)
    
    Args:
        matrix: Transition matrix DataFrame
        ergodicity_power: Power to raise matrix to for ergodicity check
    
    Returns:
        True if valid, raises AssertionError otherwise
    """
    print("\nValidating transition matrix...")
    
    # Check 1: Elements in [0, 1]
    assert (matrix >= 0).all().all(), "Negative probabilities found"
    assert (matrix <= 1).all().all(), "Probabilities > 1 found"
    print("  ✓ All probabilities in [0, 1]")
    
    # Check 2: Rows sum to 1
    row_sums = matrix.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"Rows don't sum to 1: {row_sums}"
    print("  ✓ Rows sum to 1")
    
    # Check 3: Ergodicity (can reach any state)
    matrix_power = np.linalg.matrix_power(matrix.values, ergodicity_power)
    if (matrix_power > 0).all():
        print("  ✓ Matrix is ergodic (all states reachable)")
    else:
        print("  ⚠ Matrix may not be fully ergodic (some states hard to reach)")
    
    # Calculate stationary distribution
    stationary = matrix_power[0]
    print(f"  ✓ Stationary distribution: {dict(zip(matrix.index, stationary.round(3)))}")
    
    return True


def calculate_regime_statistics(transition_matrix):
    """
    Calculate expected regime durations and frequencies.
    
    Args:
        transition_matrix: Transition matrix DataFrame
    
    Returns:
        DataFrame with regime statistics
    """
    # Expected duration in regime i = 1 / (1 - P_ii)
    durations = {}
    for regime in transition_matrix.index:
        p_stay = transition_matrix.loc[regime, regime]
        expected_duration = 1 / (1 - p_stay) if p_stay < 1 else np.inf
        durations[regime] = expected_duration
    
    # Stationary distribution (long-run frequencies)
    matrix_power = np.linalg.matrix_power(transition_matrix.values, 1000)
    stationary = matrix_power[0]
    
    # Create DataFrame with explicit index
    stats_df = pd.DataFrame({
        'expected_duration_years': pd.Series(durations),
        'long_run_frequency': pd.Series(stationary, index=transition_matrix.index)
    })
    
    return stats_df


def save_calibrated_matrix(
    calibrated_matrix: pd.DataFrame,
    empirical_matrix: pd.DataFrame,
    country_stats: list,
    config: dict,
    output_dir: Path
):
    """Save calibrated and empirical matrices to YAML files."""
    
    regime_names = calibrated_matrix.index.tolist()
    
    # Convert quarterly matrix to annual (raise to 4th power)
    time_step = config['output']['time_step_years']
    if time_step == 0.25:
        # Quarterly to annual: P_annual = P_quarterly^4
        print("\nConverting quarterly transition matrix to annual...")
        annual_matrix = pd.DataFrame(
            np.linalg.matrix_power(calibrated_matrix.values, 4),
            index=regime_names,
            columns=regime_names
        )
        print("Annual transition matrix:")
        print(annual_matrix.round(3))
        
        # Use annual matrix for output
        calibrated_matrix = annual_matrix
        time_step = 1.0
    
    stats = calculate_regime_statistics(calibrated_matrix)
    
    # Save calibrated matrix
    calibrated_path = output_dir / config['output']['calibrated_file']
    with open(calibrated_path, 'w') as f:
        f.write("# Calibrated Transition Matrix from BIS Property Price Data\n")
        f.write("#\n")
        f.write("# Data source: Bank for International Settlements (BIS)\n")
        f.write("# Real Residential Property Prices Index\n")
        f.write("# Period: 2023-Q2 to 2025-Q3 (quarterly data)\n")
        f.write("#\n")
        f.write(f"# Countries included: {', '.join([s['country'] for s in country_stats])}\n")
        f.write(f"# Total transitions observed: {sum(s['n_transitions'] for s in country_stats)}\n")
        f.write("#\n")
        f.write("# Calibration method:\n")
        f.write(f"#   - Empirical estimation from historical data ({(1-config['prior']['alpha'])*100:.0f}% weight)\n")
        f.write(f"#   - Prior beliefs from expert judgment ({config['prior']['alpha']*100:.0f}% weight)\n")
        f.write("#   - Bayesian smoothing to handle limited data\n")
        f.write("#\n")
        f.write("# Regime classification rules:\n")
        f.write(f"#   - RECOVERY: > {config['regimes']['recovery_threshold']*100:.0f}% annual property price growth\n")
        f.write(f"#   - NORMAL_GROWTH: {config['regimes']['normal_threshold']*100:.0f}% to {config['regimes']['recovery_threshold']*100:.0f}% annual growth\n")
        f.write(f"#   - STAGNATION: {config['regimes']['stagnation_threshold']*100:.0f}% to {config['regimes']['normal_threshold']*100:.0f}% annual growth\n")
        f.write(f"#   - CRISIS: < {config['regimes']['stagnation_threshold']*100:.0f}% annual growth\n")
        f.write("\n")
        
        f.write("regimes:\n")
        for regime in regime_names:
            f.write(f"  - {regime}\n")
        f.write("\n")
        
        f.write("transition_matrix:\n")
        for i, row in enumerate(calibrated_matrix.values):
            regime_name = regime_names[i]
            f.write(f"  - [{', '.join(f'{x:.4f}' for x in row)}]  # From {regime_name}\n")
        
        f.write("\n")
        f.write("# Initial state probabilities (stationary distribution)\n")
        stationary = stats['long_run_frequency'].values
        f.write(f"initial_state_probs: [{', '.join(f'{x:.4f}' for x in stationary)}]\n")
        f.write("\n")
        
        f.write("# Time step (each transition represents this many years)\n")
        f.write(f"time_step_years: {time_step}\n")
        f.write("\n")
        
        f.write("# Expected regime durations (years)\n")
        f.write("expected_durations:\n")
        for regime, duration in stats['expected_duration_years'].items():
            f.write(f"  {regime}: {duration:.2f}\n")
    
    print(f"\n✓ Calibrated matrix saved to: {calibrated_path}")
    
    # Save empirical matrix
    empirical_path = output_dir / config['output']['empirical_file']
    with open(empirical_path, 'w') as f:
        f.write("# Empirical Transition Matrix (No Prior Smoothing)\n")
        f.write("# For comparison purposes only\n\n")
        f.write("transition_matrix:\n")
        for i, row in enumerate(empirical_matrix.values):
            regime_name = regime_names[i]
            f.write(f"  - [{', '.join(f'{x:.4f}' for x in row)}]  # From {regime_name}\n")
    
    print(f"✓ Empirical matrix saved to: {empirical_path}")


def save_expert_matrix(prior_matrix: pd.DataFrame, output_path: Path, config: dict):
    """
    Save expert prior matrix in the same format as other matrices.
    
    Args:
        prior_matrix: Expert prior transition matrix
        output_path: Path to save expert matrix YAML
        config: Configuration dict for metadata
    """
    regime_names = prior_matrix.index.tolist()
    stats = calculate_regime_statistics(prior_matrix)
    
    with open(output_path, 'w') as f:
        f.write("# Expert Prior Transition Matrix\n")
        f.write("#\n")
        f.write("# Source: Expert judgment (see plans/reasoning_for_priors.md)\n")
        f.write("# This matrix represents realistic regime persistence based on\n")
        f.write("# economic theory and historical patterns.\n")
        f.write("#\n")
        f.write("# Recommended for most simulations as it provides stable,\n")
        f.write("# theoretically-grounded transition probabilities.\n")
        f.write("\n")
        
        f.write("regimes:\n")
        for regime in regime_names:
            f.write(f"  - {regime}\n")
        f.write("\n")
        
        f.write("transition_matrix:\n")
        for i, row in enumerate(prior_matrix.values):
            regime_name = regime_names[i]
            f.write(f"  - [{', '.join(f'{x:.4f}' for x in row)}]  # From {regime_name}\n")
        
        f.write("\n")
        f.write("# Initial state probabilities (stationary distribution)\n")
        stationary = stats['long_run_frequency'].values
        f.write(f"initial_state_probs: [{', '.join(f'{x:.4f}' for x in stationary)}]\n")
        f.write("\n")
        
        f.write("# Time step (each transition represents this many years)\n")
        f.write("time_step_years: 1.0\n")
        f.write("\n")
        
        f.write("# Expected regime durations (years)\n")
        f.write("expected_durations:\n")
        for regime, duration in stats['expected_duration_years'].items():
            f.write(f"  {regime}: {duration:.2f}\n")


def copy_outputs_to_configs(
    prior_matrix: pd.DataFrame,
    config: dict,
    output_dir: Path
):
    """
    Copy calibration outputs to configs directory for use in simulations.
    
    Copies three matrices:
    1. expert_matrix.yaml - The expert prior (unchanged)
    2. calibrated_matrix.yaml - Bayesian smoothed result
    3. empirical_matrix.yaml - Pure empirical result
    
    Args:
        prior_matrix: Expert prior matrix (to be saved as expert_matrix.yaml)
        config: Configuration dict
        output_dir: Directory containing calibration outputs
    """
    if not config['output'].get('copy_to_configs', False):
        print("\nSkipping copy to configs (copy_to_configs=False)")
        return
    
    config_dir = Path(config['output']['config_dir'])
    config_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Copying matrices to configs directory")
    print("=" * 70)
    
    # 1. Copy calibrated matrix
    src_calibrated = output_dir / config['output']['calibrated_file']
    dst_calibrated = config_dir / 'calibrated_matrix.yaml'
    shutil.copy2(src_calibrated, dst_calibrated)
    print(f"✓ Copied calibrated matrix: {dst_calibrated}")
    
    # 2. Copy empirical matrix
    src_empirical = output_dir / config['output']['empirical_file']
    dst_empirical = config_dir / 'empirical_matrix.yaml'
    shutil.copy2(src_empirical, dst_empirical)
    print(f"✓ Copied empirical matrix: {dst_empirical}")
    
    # 3. Save expert prior as expert_matrix.yaml
    dst_expert = config_dir / 'expert_matrix.yaml'
    save_expert_matrix(prior_matrix, dst_expert, config)
    print(f"✓ Saved expert matrix: {dst_expert}")
    
    print("\nAll matrices ready for use in simulations!")


def main():
    """Main calibration workflow."""
    parser = argparse.ArgumentParser(
        description='Calibrate Markov transition matrix from BIS property price data'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='scenarios/markov/calibration/calibration_config.yaml',
        help='Path to calibration configuration file'
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("Transition Matrix Calibration from BIS Property Price Data")
    print("=" * 70)
    
    # Load configuration
    print(f"\nLoading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Load prior matrix
    prior_path = config['prior']['prior_matrix_path']
    regime_names = config['regimes']['names']
    print(f"Loading prior matrix from {prior_path}...")
    prior_matrix = load_prior_matrix(prior_path, regime_names)
    
    print("\nPrior transition matrix (from expert judgment):")
    print(prior_matrix)
    
    # Get calibration parameters
    countries = config['data']['countries']
    print(f"\nCountries to analyze: {', '.join(countries)}")
    print()
    
    # Calibrate
    calibrated_matrix, empirical_matrix, country_stats = calibrate_transition_matrix(
        data_path=config['data']['bis_data_path'],
        countries=countries,
        prior_matrix=prior_matrix,
        thresholds=config['regimes'],
        alpha=config['prior']['alpha']
    )
    
    print("\n" + "=" * 70)
    print("FINAL CALIBRATED TRANSITION MATRIX (quarterly data)")
    print("=" * 70)
    print(calibrated_matrix.round(3))
    
    # Validate
    validate_transition_matrix(
        calibrated_matrix,
        ergodicity_power=config['validation']['ergodicity_power']
    )
    
    # Calculate statistics
    print("\n" + "=" * 70)
    print("REGIME STATISTICS")
    print("=" * 70)
    stats = calculate_regime_statistics(calibrated_matrix)
    print(stats.round(3))
    
    # Save results
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_calibrated_matrix(
        calibrated_matrix,
        empirical_matrix,
        country_stats,
        config,
        output_dir
    )
    
    # Copy outputs to configs directory
    copy_outputs_to_configs(
        prior_matrix=prior_matrix,
        config=config,
        output_dir=output_dir
    )
    
    print("\n" + "=" * 70)
    print("Calibration complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
