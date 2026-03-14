"""
Probabilistic Mortgage Comparison Module

This module provides functions for comparing multiple mortgage scenarios
under shared uncertainty using Monte Carlo simulation.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

import pandas as pd

from src.sensitivity import generate_parameter_space, run_sensitivity_analysis
from src.estate import BuyingScenario
from src.tax import NLHomeTax2026


def load_scenario_config(config_path: str) -> Dict[str, Any]:
    """
    Load scenario configuration from YAML file.
    
    Args:
        config_path: Path to scenario YAML file
    
    Returns:
        Dictionary with scenario configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Scenario config not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def run_multi_scenario_comparison(
    scenario_configs: List[str],
    shared_variables: Dict[str, Dict],
    n_samples: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Run probabilistic comparison of multiple mortgage scenarios.
    
    This function evaluates multiple mortgage scenarios under the same
    uncertain conditions (shared parameter space) to enable fair comparison.
    Each scenario is evaluated with the same Monte Carlo samples, ensuring
    that differences in outcomes are due to the mortgage structure, not
    random variation.
    
    Args:
        scenario_configs: List of paths to scenario YAML files
        shared_variables: Uncertain variables configuration (same for all scenarios)
            Example: {
                'annual_value_growth': {
                    'strategy': 'normal',
                    'mean': 0.02,
                    'std': 0.015,
                    'samples': 5000,
                    'seed': 42
                }
            }
        n_samples: Override number of samples (if not specified in variables)
        seed: Override random seed (if not specified in variables)
        verbose: Print progress information
    
    Returns:
        Combined DataFrame with columns:
            - All input parameters (from shared_variables)
            - All evaluation results (wealth_end, avg_net_cost_per_month, etc.)
            - mortgage_scenario_label: Name of the mortgage scenario
            - scenario_file: Path to scenario config file
            - world_scenario_id: Index of the Monte Carlo sample (0 to n_samples-1)
    
    Example:
        >>> scenarios = [
        ...     'scenarios/mortgage_scenario_a.yaml',
        ...     'scenarios/mortgage_scenario_b.yaml'
        ... ]
        >>> variables = {
        ...     'annual_value_growth': {
        ...         'strategy': 'normal',
        ...         'mean': 0.02,
        ...         'std': 0.015,
        ...         'samples': 1000,
        ...         'seed': 42
        ...     }
        ... }
        >>> df = run_multi_scenario_comparison(scenarios, variables, verbose=True)
        >>> print(df.groupby('mortgage_scenario_label')['wealth_end'].describe())
    
    Raises:
        FileNotFoundError: If any scenario config file doesn't exist
        ValueError: If scenario configs are invalid
    """
    if verbose:
        print(f"Running probabilistic comparison of {len(scenario_configs)} scenarios...")
        print(f"Shared variables: {list(shared_variables.keys())}")
    
    # Override samples and seed if provided
    if n_samples is not None or seed is not None:
        shared_variables = shared_variables.copy()
        for var_name, var_config in shared_variables.items():
            var_config = var_config.copy()
            if n_samples is not None:
                var_config['samples'] = n_samples
            if seed is not None:
                var_config['seed'] = seed
            shared_variables[var_name] = var_config
    
    # Generate shared parameter space (same samples for all scenarios)
    if verbose:
        print("\nGenerating shared parameter space...")
    
    param_space = generate_parameter_space(shared_variables)
    
    if verbose:
        print(f"Generated {len(param_space):,} parameter combinations")
    
    all_results = []
    
    for i, scenario_path in enumerate(scenario_configs):
        if verbose:
            print(f"\n[{i+1}/{len(scenario_configs)}] Evaluating: {scenario_path}")
        
        # Load scenario config
        config = load_scenario_config(scenario_path)
        
        # Extract scenario name
        scenario_name = config.get('name', Path(scenario_path).stem)
        
        # Create BuyingScenario from config
        if 'buying' not in config:
            raise ValueError(f"Scenario config missing 'buying' section: {scenario_path}")
        
        base_scenario = BuyingScenario(**config['buying'])
        
        # Create tax configuration if present
        tax = None
        if 'tax' in config:
            tax = NLHomeTax2026(**config['tax'])
        
        # Run sensitivity analysis with shared parameter space
        results_df = run_sensitivity_analysis(
            base_scenario=base_scenario,
            param_space=param_space,
            tax=tax,
            verbose=verbose
        )
        
        # Add scenario metadata
        results_df['mortgage_scenario_label'] = scenario_name
        results_df['scenario_file'] = scenario_path
        
        all_results.append(results_df)
    
    # Combine all results
    if verbose:
        print("\nCombining results...")
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Add world scenario ID (row index within each parameter combination)
    # This allows matching the same "world state" across different mortgages
    combined_df['world_scenario_id'] = combined_df.groupby('mortgage_scenario_label').cumcount()
    
    if verbose:
        print(f"\nCompleted: {len(combined_df):,} total evaluations")
        print(f"  Scenarios: {combined_df['mortgage_scenario_label'].nunique()}")
        print(f"  Samples per scenario: {len(param_space):,}")
    
    return combined_df


def load_shared_variables_config(variables_path: str) -> Dict[str, Dict]:
    """
    Load shared variables configuration from YAML file.
    
    Args:
        variables_path: Path to variables YAML file
    
    Returns:
        Dictionary mapping variable names to their configuration
        
    Example YAML:
        variables:
          annual_value_growth:
            strategy: "normal"
            mean: 0.02
            std: 0.015
            samples: 5000
            seed: 42
    
    Raises:
        FileNotFoundError: If variables file doesn't exist
        ValueError: If variables file is invalid
    """
    variables_file = Path(variables_path)
    if not variables_file.exists():
        raise FileNotFoundError(f"Variables config not found: {variables_path}")
    
    with open(variables_file, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'variables' not in config:
        raise ValueError(
            f"Variables config must have 'variables' key. Found: {list(config.keys())}"
        )
    
    return config['variables']


def export_comparison_results(
    df: pd.DataFrame,
    output_dir: str = "outputs/probabilistic",
    filename_prefix: str = "comparison",
    include_timestamp: bool = True
) -> str:
    """
    Export comparison results to CSV file.
    
    Args:
        df: Results DataFrame from run_multi_scenario_comparison()
        output_dir: Output directory path
        filename_prefix: Prefix for output filename
        include_timestamp: Include timestamp in filename
    
    Returns:
        Path to saved file
    """
    from datetime import datetime
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    if include_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename_prefix}_{timestamp}.csv"
    else:
        filename = f"{filename_prefix}.csv"
    
    # Export to CSV
    filepath = output_path / filename
    df.to_csv(filepath, index=False)
    
    return str(filepath)
