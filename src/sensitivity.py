"""
Sensitivity Analysis Module

This module provides functions for performing brute-force sensitivity analysis
on buying scenarios by varying parameters across defined ranges and strategies.
"""

import itertools
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from src.estate import BuyingScenario, evaluate_buying
from src.tax import NLHomeTax2026


def _generate_values_for_strategy(
    param_name: str,
    config: Dict[str, Any]
) -> np.ndarray:
    """
    Generate values for a single parameter based on strategy.
    
    Args:
        param_name: Name of parameter (for error messages)
        config: Strategy configuration dict with keys:
            - strategy: 'linspace', 'logspace', 'random', or 'values'
            - For linspace/logspace: min, max, steps
            - For random: min, max, samples (default 100)
            - For values: values (list)
    
    Returns:
        Numpy array of values
        
    Raises:
        ValueError: If strategy is unknown or required fields are missing
    """
    strategy = config.get('strategy')
    
    if strategy == 'linspace':
        if 'min' not in config or 'max' not in config or 'steps' not in config:
            raise ValueError(f"Parameter '{param_name}': linspace requires min, max, and steps")
        if config['min'] >= config['max']:
            raise ValueError(f"Parameter '{param_name}': min must be less than max")
        return np.linspace(config['min'], config['max'], config['steps'])
    
    elif strategy == 'logspace':
        if 'min' not in config or 'max' not in config or 'steps' not in config:
            raise ValueError(f"Parameter '{param_name}': logspace requires min, max, and steps")
        if config['min'] <= 0 or config['max'] <= 0:
            raise ValueError(f"Parameter '{param_name}': logspace requires positive min and max")
        if config['min'] >= config['max']:
            raise ValueError(f"Parameter '{param_name}': min must be less than max")
        return np.logspace(np.log10(config['min']), np.log10(config['max']), config['steps'])
    
    elif strategy == 'random':
        if 'min' not in config or 'max' not in config:
            raise ValueError(f"Parameter '{param_name}': random requires min and max")
        if config['min'] >= config['max']:
            raise ValueError(f"Parameter '{param_name}': min must be less than max")
        samples = config.get('samples', 100)
        return np.random.uniform(config['min'], config['max'], samples)
    
    elif strategy == 'values':
        if 'values' not in config:
            raise ValueError(f"Parameter '{param_name}': values strategy requires 'values' list")
        return np.array(config['values'])
    
    else:
        raise ValueError(f"Parameter '{param_name}': unknown strategy '{strategy}'. "
                        f"Use 'linspace', 'logspace', 'random', or 'values'")


def generate_parameter_space(variables_config: Dict[str, Dict]) -> List[Dict[str, float]]:
    """
    Generate parameter space from variable configuration.
    
    Args:
        variables_config: Dict mapping parameter names to their config
            Example: {
                'purchase_price': {'strategy': 'linspace', 'min': 300000, 'max': 400000, 'steps': 20},
                'monthly_vve': {'strategy': 'values', 'values': [200, 300, 400]}
            }
    
    Returns:
        List of parameter dictionaries, one per combination
        Example: [
            {'purchase_price': 300000, 'monthly_vve': 200},
            {'purchase_price': 300000, 'monthly_vve': 300},
            ...
        ]
        
    Raises:
        ValueError: If no variables defined or invalid configuration
    """
    if not variables_config:
        raise ValueError("No variables defined in sensitivity config")
    
    # Generate values for each parameter
    param_names = []
    param_values = []
    
    for param_name, config in variables_config.items():
        values = _generate_values_for_strategy(param_name, config)
        param_names.append(param_name)
        param_values.append(values)
    
    # Create all combinations using Cartesian product
    combinations = list(itertools.product(*param_values))
    
    # Convert to list of dictionaries
    param_space = [
        dict(zip(param_names, combo))
        for combo in combinations
    ]
    
    # Warn if parameter space is very large
    if len(param_space) > 100000:
        print(f"WARNING: Large parameter space ({len(param_space):,} combinations). "
              f"This may take significant time and memory.")
    
    return param_space


def run_sensitivity_analysis(
    base_scenario: BuyingScenario,
    param_space: List[Dict[str, float]],
    tax: Optional[NLHomeTax2026] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Run sensitivity analysis over parameter space.
    
    Args:
        base_scenario: Base BuyingScenario with default values
        param_space: List of parameter dictionaries to vary
        tax: Tax configuration (optional, uses base_scenario.tax if not provided)
        verbose: Print progress information
    
    Returns:
        DataFrame with input parameters and all evaluation results
        
    Note:
        The 'monthly_net_cost' list is excluded from the DataFrame to reduce size.
    """
    if verbose:
        print(f"Running sensitivity analysis with {len(param_space):,} parameter combinations...")
    
    results = []
    errors = 0
    
    for i, params in enumerate(param_space):
        try:
            # Create scenario with updated parameters
            # Use dataclasses.replace() to create a new instance with updated values
            scenario = replace(base_scenario, **params)
            
            # Override tax if provided (replace doesn't handle this well)
            if tax is not None:
                scenario = replace(scenario, tax=tax)
            
            # Evaluate scenario
            evaluation = evaluate_buying(scenario)
            
            # Combine input parameters with output results
            result = {**params, **evaluation}
            
            # Remove large list fields to keep DataFrame manageable
            result.pop('monthly_net_cost', None)
            
            results.append(result)
            
        except Exception as e:
            errors += 1
            if verbose:
                print(f"Error evaluating combination {i+1}: {e}")
            continue
        
        # Progress reporting
        if verbose and (i + 1) % max(1, len(param_space) // 10) == 0:
            progress = (i + 1) / len(param_space) * 100
            print(f"Progress: {progress:.1f}% ({i+1:,}/{len(param_space):,})")
    
    if verbose:
        print(f"Completed: {len(results):,} successful evaluations, {errors} errors")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df


def apply_constraints(
    results_df: pd.DataFrame,
    constraints: List[str]
) -> pd.DataFrame:
    """
    Filter results based on constraint expressions.
    
    Args:
        results_df: DataFrame with sensitivity analysis results
        constraints: List of constraint expressions (pandas query syntax)
            Example: ["cash_required_upfront <= 50000", "avg_net_cost_per_month <= 2500"]
    
    Returns:
        Filtered DataFrame
        
    Raises:
        ValueError: If constraint expression is invalid
    """
    if not constraints:
        return results_df
    
    filtered_df = results_df.copy()
    
    for constraint in constraints:
        try:
            filtered_df = filtered_df.query(constraint)
        except Exception as e:
            raise ValueError(f"Invalid constraint '{constraint}': {e}")
    
    return filtered_df


def export_results(
    df: pd.DataFrame,
    output_config: Dict[str, Any],
    metadata: Optional[Dict] = None
) -> str:
    """
    Export results to file.
    
    Args:
        df: Results DataFrame
        output_config: Output configuration dict with keys:
            - format: 'csv', 'parquet', or 'json'
            - directory: output directory path
            - filename_prefix: prefix for filename
            - include_timestamp: bool (default True)
        metadata: Optional metadata to include (config used, runtime, etc.)
    
    Returns:
        Path to saved file
        
    Raises:
        ValueError: If format is unsupported
        ImportError: If required library is missing (e.g., pyarrow for parquet)
    """
    # Extract config with defaults
    format_type = output_config.get('format', 'csv').lower()
    directory = output_config.get('directory', 'outputs/sensitivity')
    filename_prefix = output_config.get('filename_prefix', 'sensitivity_analysis')
    include_timestamp = output_config.get('include_timestamp', True)
    
    # Create output directory
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    if include_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{filename_prefix}_{timestamp}"
    else:
        filename = filename_prefix
    
    # Export based on format
    if format_type == 'csv':
        filepath = output_dir / f"{filename}.csv"
        df.to_csv(filepath, index=False)
    
    elif format_type == 'parquet':
        try:
            filepath = output_dir / f"{filename}.parquet"
            df.to_parquet(filepath, index=False)
        except ImportError:
            raise ImportError("Parquet export requires 'pyarrow' package. "
                            "Install with: pip install pyarrow")
    
    elif format_type == 'json':
        filepath = output_dir / f"{filename}.json"
        df.to_json(filepath, orient='records', indent=2)
    
    else:
        raise ValueError(f"Unsupported format: {format_type}. Use 'csv', 'parquet', or 'json'")
    
    # Save metadata if provided
    if metadata:
        import json
        metadata_path = output_dir / f"{filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    return str(filepath)
