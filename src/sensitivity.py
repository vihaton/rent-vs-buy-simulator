"""
Sensitivity Analysis Module

This module provides functions for performing brute-force sensitivity analysis
on buying scenarios by varying parameters across defined ranges and strategies.
"""

import itertools
import re
import copy
import json
from dataclasses import replace, fields
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
            - strategy: 'linspace', 'logspace', 'random', 'values', 'normal', 'mixture', 'triangular'
            - For linspace/logspace: min, max, steps
            - For random: min, max, samples (default 100), seed (optional)
            - For values: values (list)
            - For normal: mean, std, samples (default 1000), seed (optional), min/max (optional truncation)
            - For mixture: components (list of dicts with weight, mean, std), samples, seed, min/max (optional)
            - For triangular: min, mode, max, samples (default 1000), seed (optional)
    
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
    
    elif strategy == 'random' or strategy == 'uniform':
        if 'min' not in config or 'max' not in config:
            raise ValueError(f"Parameter '{param_name}': {strategy} requires min and max")
        if config['min'] >= config['max']:
            raise ValueError(f"Parameter '{param_name}': min must be less than max")
        samples = config.get('samples', 100)
        seed = config.get('seed')
        rng = np.random.default_rng(seed)
        return rng.uniform(config['min'], config['max'], samples)
    
    elif strategy == 'normal':
        if 'mean' not in config or 'std' not in config:
            raise ValueError(f"Parameter '{param_name}': normal requires mean and std")
        samples = config.get('samples', 1000)
        seed = config.get('seed')
        rng = np.random.default_rng(seed)
        values = rng.normal(config['mean'], config['std'], samples)
        
        # Optional: truncation
        if 'min' in config:
            values = np.maximum(values, config['min'])
        if 'max' in config:
            values = np.minimum(values, config['max'])
        
        return values
    
    elif strategy == 'mixture':
        if 'components' not in config:
            raise ValueError(f"Parameter '{param_name}': mixture requires 'components' list")
        
        components = config['components']
        if not components:
            raise ValueError(f"Parameter '{param_name}': mixture requires at least one component")
        
        samples = config.get('samples', 1000)
        seed = config.get('seed')
        rng = np.random.default_rng(seed)
        
        # Validate mixture weights sum to 1
        weights = [c.get('weight', 0) for c in components]
        if not np.isclose(sum(weights), 1.0, atol=1e-6):
            raise ValueError(
                f"Parameter '{param_name}': mixture weights must sum to 1.0, got {sum(weights):.6f}"
            )
        
        # Validate each component has required fields
        for i, component in enumerate(components):
            if 'mean' not in component or 'std' not in component:
                raise ValueError(
                    f"Parameter '{param_name}': component {i} requires 'mean' and 'std'"
                )
        
        # Sample regime indices according to weights
        regime_indices = rng.choice(
            len(components),
            size=samples,
            p=weights
        )
        
        # Sample from each regime's distribution
        values = np.zeros(samples)
        for i, component in enumerate(components):
            mask = regime_indices == i
            n_samples_regime = mask.sum()
            
            if n_samples_regime > 0:
                regime_values = rng.normal(
                    component['mean'],
                    component['std'],
                    n_samples_regime
                )
                values[mask] = regime_values
        
        # Optional: truncation
        if 'min' in config:
            values = np.maximum(values, config['min'])
        if 'max' in config:
            values = np.minimum(values, config['max'])
        
        return values
    
    elif strategy == 'triangular':
        if 'min' not in config or 'mode' not in config or 'max' not in config:
            raise ValueError(f"Parameter '{param_name}': triangular requires min, mode, and max")
        if not (config['min'] <= config['mode'] <= config['max']):
            raise ValueError(
                f"Parameter '{param_name}': triangular requires min <= mode <= max"
            )
        samples = config.get('samples', 1000)
        seed = config.get('seed')
        rng = np.random.default_rng(seed)
        return rng.triangular(config['min'], config['mode'], config['max'], samples)
    
    elif strategy == 'values':
        if 'values' not in config:
            raise ValueError(f"Parameter '{param_name}': values strategy requires 'values' list")
        return np.array(config['values'])
    
    else:
        raise ValueError(
            f"Parameter '{param_name}': unknown strategy '{strategy}'. "
            f"Use 'linspace', 'logspace', 'random', 'uniform', 'normal', 'mixture', 'triangular', or 'values'"
        )


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


def _set_nested_attr(obj: Any, path: str, value: Any):
    """
    Set nested attribute using dot notation and array indexing.
    
    Handles both object attributes and dictionary keys (for YAML-loaded data).
    Supports wildcard syntax [*] to apply value to all elements in a list.
    
    Examples:
        "mortgage_loans[0].annual_rate" sets obj.mortgage_loans[0].annual_rate = value
        "mortgage_loans[*].annual_rate" sets annual_rate for ALL mortgage loans
        "mortgage_loans[*].rate_resets[0].new_rate_default" sets new_rate_default for first reset of ALL loans
    
    Args:
        obj: Object to modify
        path: Nested path (e.g., "mortgage_loans[0].annual_rate" or "mortgage_loans[*].annual_rate")
        value: Value to set
    """
    parts = re.split(r'\.|\[|\]', path)
    parts = [p for p in parts if p]  # Remove empty strings
    
    # Check if path contains wildcard
    if '*' in parts:
        # Find the position of the wildcard
        wildcard_idx = parts.index('*')
        
        # Navigate to the list that contains the wildcard
        current = obj
        for i, part in enumerate(parts[:wildcard_idx]):
            if part.isdigit():
                current = current[int(part)]
            else:
                # Handle both object attributes and dictionary keys
                if isinstance(current, dict):
                    current = current[part]
                else:
                    current = getattr(current, part)
        
        # Current should now be a list
        if not isinstance(current, list):
            raise ValueError(
                f"Wildcard [*] used on non-list object at path '{path}'. "
                f"Object type: {type(current).__name__}"
            )
        
        # Build the remaining path after the wildcard
        remaining_parts = parts[wildcard_idx + 1:]
        if remaining_parts:
            remaining_path = _rebuild_path_from_parts(remaining_parts)
            # Recursively apply to each element in the list
            for item in current:
                _set_nested_attr(item, remaining_path, value)
        else:
            # Wildcard is at the end, set value directly on each list element
            for i in range(len(current)):
                current[i] = value
    else:
        # No wildcard - standard nested attribute setting
        current = obj
        for i, part in enumerate(parts[:-1]):
            if part.isdigit():
                current = current[int(part)]
            else:
                # Handle both object attributes and dictionary keys
                if isinstance(current, dict):
                    current = current[part]
                else:
                    current = getattr(current, part)
        
        final_part = parts[-1]
        if final_part.isdigit():
            current[int(final_part)] = value
        else:
            # Handle both object attributes and dictionary keys
            if isinstance(current, dict):
                current[final_part] = value
            else:
                setattr(current, final_part, value)


def _rebuild_path_from_parts(parts: List[str]) -> str:
    """
    Rebuild a path string from parts list.
    
    Converts ['rate_resets', '0', 'new_rate_default'] back to 'rate_resets[0].new_rate_default'
    
    Args:
        parts: List of path parts
        
    Returns:
        Reconstructed path string
    """
    if not parts:
        return ""
    
    path = parts[0]
    for part in parts[1:]:
        if part.isdigit() or part == '*':
            path += f"[{part}]"
        else:
            path += f".{part}"
    
    return path


def _apply_nested_params(base_scenario: BuyingScenario, params: Dict[str, Any]) -> BuyingScenario:
    """
    Apply parameters including nested ones to scenario.
    
    Examples:
        "mortgage_loans[0].annual_rate" -> scenario.mortgage_loans[0].annual_rate = value
        "mortgage_loans[1].rate_resets[0].new_rate_default" -> ...
    
    Args:
        base_scenario: Base scenario to modify
        params: Dictionary of parameters to apply
    
    Returns:
        Modified scenario copy
        
    Raises:
        AttributeError: If parameter path is invalid (with helpful error message)
    """
    scenario = copy.deepcopy(base_scenario)
    
    simple_params = {}
    nested_params = {}
    
    for param_path, value in params.items():
        if '.' not in param_path and '[' not in param_path:
            # Simple parameter
            simple_params[param_path] = value
        else:
            # Nested parameter
            nested_params[param_path] = value
    
    # Apply simple parameters using dataclasses.replace
    if simple_params:
        try:
            scenario = replace(scenario, **simple_params)
        except TypeError as e:
            # Get valid field names from the dataclass
            valid_fields = [f.name for f in fields(BuyingScenario)]
            invalid_params = [p for p in simple_params.keys() if p not in valid_fields]
            raise ValueError(
                f"Invalid parameter(s): {', '.join(invalid_params)}. "
                f"Valid BuyingScenario fields: {', '.join(valid_fields)}"
            ) from e
    
    # Apply nested parameters
    for param_path, value in nested_params.items():
        try:
            _set_nested_attr(scenario, param_path, value)
        except (AttributeError, KeyError, IndexError) as e:
            raise AttributeError(
                f"Invalid parameter path '{param_path}': {e}. "
                f"Check for typos (e.g., 'morgage_loans' vs 'mortgage_loans') "
                f"and verify the path exists in the scenario."
            ) from e
    
    return scenario


def run_sensitivity_analysis(
    base_scenario: BuyingScenario,
    param_space: List[Dict[str, float]],
    tax: Optional[NLHomeTax2026] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Run sensitivity analysis over parameter space.
    
    Supports both simple and nested parameters:
    - Simple: "purchase_price", "monthly_vve"
    - Nested: "mortgage_loans[0].annual_rate", "mortgage_loans[1].rate_resets[0].new_rate_default"
    
    Args:
        base_scenario: Base BuyingScenario with default values
        param_space: List of parameter dictionaries to vary
        tax: Tax configuration (optional, uses base_scenario.tax if not provided)
        verbose: Print progress information
    
    Returns:
        DataFrame with input parameters and all evaluation results
        
    Note:
        The 'monthly_net_cost' list is excluded from the DataFrame to reduce size.
        
    Raises:
        ValueError: If first evaluation fails (likely due to invalid parameter names)
    """
    if verbose:
        print(f"Running sensitivity analysis with {len(param_space):,} parameter combinations...")
    
    results = []
    errors = 0
    first_error = None
    
    for i, params in enumerate(param_space):
        try:
            # Create scenario with updated parameters (supports nested)
            scenario = _apply_nested_params(base_scenario, params)
            
            # Override tax if provided
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
            
            # Capture first error for better diagnostics
            if first_error is None:
                first_error = (i, params, e)
            
            # Fail fast on first error to provide immediate feedback
            if i == 0:
                raise ValueError(
                    f"First scenario evaluation failed. This usually indicates an invalid parameter name.\n"
                    f"Parameters: {params}\n"
                    f"Error: {str(e)}\n"
                    f"Hint: Check for typos in parameter names (e.g., 'morgage_loans' vs 'mortgage_loans')"
                ) from e
            
            if verbose:
                print(f"Error evaluating combination {i+1}: {e}")
                print(f"  Parameters: {params}")
            continue
        
        # Progress reporting
        if verbose and (i + 1) % max(1, len(param_space) // 10) == 0:
            progress = (i + 1) / len(param_space) * 100
            print(f"Progress: {progress:.1f}% ({i+1:,}/{len(param_space):,})")
    
    if verbose:
        print(f"Completed: {len(results):,} successful evaluations, {errors} errors")
    
    # Warn if all evaluations failed
    if len(results) == 0 and errors > 0:
        if first_error:
            i, params, e = first_error
            raise ValueError(
                f"All {errors} scenario evaluations failed. First error:\n"
                f"Parameters: {params}\n"
                f"Error: {str(e)}\n"
                f"Hint: Check for typos in parameter names in your config file."
            )
        else:
            raise ValueError(f"All {errors} scenario evaluations failed with unknown errors.")
    
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
        ValueError: If constraint expression is invalid or DataFrame is empty
    """
    if not constraints:
        return results_df
    
    # Check if DataFrame is empty
    if len(results_df) == 0:
        raise ValueError(
            "Cannot apply constraints to empty results DataFrame. "
            "All scenario evaluations failed. Check the error messages above for details."
        )
    
    filtered_df = results_df.copy()
    
    for constraint in constraints:
        try:
            filtered_df = filtered_df.query(constraint)
        except Exception as e:
            # Provide more helpful error message
            available_columns = list(results_df.columns)
            raise ValueError(
                f"Invalid constraint '{constraint}': {e}\n"
                f"Available columns: {', '.join(available_columns)}"
            )
    
    return filtered_df


def list_available_constraint_fields() -> Dict[str, str]:
    """
    Return a dictionary of all fields available for constraints.
    
    Returns:
        Dictionary mapping field names to their descriptions
    """
    return {
        # Financial Summary
        "total_spent": "Total cash spent over the period",
        "total_income": "Total income received (rent + tax benefits)",
        "net_cashflow": "Net cash flow (income - spent)",
        "wealth_end": "Final wealth position",
        "avg_net_cost_per_month": "Average net cost per month",
        "usual_net_cost_per_month": "Usual monthly cost (excluding one-off costs)",
        "months": "Number of months in the scenario",
        
        # Upfront Costs
        "cash_required_upfront": "Total upfront cash needed",
        "down_payment": "Down payment amount",
        "one_off_costs": "One-time costs",
        "renovation_costs_once": "Renovation costs",
        
        # Mortgage Details
        "mortgage_monthly_payment": "Monthly mortgage payment",
        "mortgage_interest_paid": "Total interest paid",
        "mortgage_principal_paid": "Total principal paid",
        "mortgage_remaining_balance": "Remaining debt at end",
        
        # Property Value
        "home_value_end": "Home value at end of period",
        "equity_end_if_not_sold": "Equity if not sold",
        "selling_costs_if_sold": "Selling costs if sold",
        "net_sale_proceeds_if_sold": "Net proceeds from sale",
        
        # Tax Effects
        "tax_effect_total_over_horizon": "Total tax benefit/cost over period",
    }


def print_constraint_fields(verbose: bool = False):
    """
    Print all available constraint fields in a formatted way.
    
    Args:
        verbose: If True, include example constraints
    """
    fields = list_available_constraint_fields()
    
    print("=" * 80)
    print("AVAILABLE CONSTRAINT FIELDS")
    print("=" * 80)
    print("\nThese fields can be used in constraint expressions with pandas query syntax.")
    print("Operators: <, <=, >, >=, ==, !=, and, or, not\n")
    
    # Group by category
    categories = {
        "Financial Summary": [
            "total_spent", "total_income", "net_cashflow", "wealth_end",
            "avg_net_cost_per_month", "usual_net_cost_per_month", "months"
        ],
        "Upfront Costs": [
            "cash_required_upfront", "down_payment", "one_off_costs", "renovation_costs_once"
        ],
        "Mortgage Details": [
            "mortgage_monthly_payment", "mortgage_interest_paid",
            "mortgage_principal_paid", "mortgage_remaining_balance"
        ],
        "Property Value": [
            "home_value_end", "equity_end_if_not_sold",
            "selling_costs_if_sold", "net_sale_proceeds_if_sold"
        ],
        "Tax Effects": [
            "tax_effect_total_over_horizon"
        ]
    }
    
    for category, field_names in categories.items():
        print(f"\n{category}:")
        print("-" * 80)
        for field_name in field_names:
            if field_name in fields:
                print(f"  • {field_name:35s} - {fields[field_name]}")
    
    if verbose:
        print("\n" + "=" * 80)
        print("EXAMPLE CONSTRAINTS")
        print("=" * 80)
        examples = [
            ('Limit upfront cash', '"cash_required_upfront <= 30000"'),
            ('Limit monthly costs', '"avg_net_cost_per_month <= 2500"'),
            ('Ensure positive wealth', '"wealth_end >= 0"'),
            ('Limit mortgage payment', '"mortgage_monthly_payment <= 2000"'),
            ('Require minimum equity', '"equity_end_if_not_sold >= 50000"'),
            ('Multiple conditions', '"cash_required_upfront <= 40000 and avg_net_cost_per_month <= 2200"'),
            ('Range constraint', '"wealth_end >= -10000 and wealth_end <= 100000"'),
            ('Positive sale proceeds', '"net_sale_proceeds_if_sold > 0"'),
        ]
        
        for description, constraint in examples:
            print(f"\n  {description}:")
            print(f"    {constraint}")
        
        print("\n" + "=" * 80)
        print("USAGE IN CONFIG FILE")
        print("=" * 80)
        print("""
sensitivity:
  constraints:
    - "cash_required_upfront <= 30000"
    - "avg_net_cost_per_month <= 2500"
    - "wealth_end >= 0"
""")


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
        metadata_path = output_dir / f"{filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    return str(filepath)
