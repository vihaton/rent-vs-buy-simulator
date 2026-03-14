"""
CLI tool for sensitivity analysis on buying scenarios.

Usage:
    python -m src.cli_sensitivity --config scenarios/sensitivity_example.yaml
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, Any
import traceback

from src.sensitivity import (
    generate_parameter_space,
    run_sensitivity_analysis,
    apply_constraints,
    export_results,
    print_constraint_fields
)
from src.estate import BuyingScenario
from src.tax import NLHomeTax2026
from src.mortgage import MortgageLoan, RateReset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run sensitivity analysis on buying scenarios',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings from config
  python -m src.cli_sensitivity --config scenarios/sensitivity_example.yaml
  
  # Override output format
  python -m src.cli_sensitivity --config scenarios/sensitivity_example.yaml --format parquet
  
  # Specify custom output directory
  python -m src.cli_sensitivity --config scenarios/sensitivity_example.yaml --output outputs/my_analysis
  
  # Run with verbose output
  python -m src.cli_sensitivity --config scenarios/sensitivity_example.yaml --verbose
"""
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=False,
        help='Path to sensitivity analysis configuration YAML file'
    )
    
    parser.add_argument(
        '--list-constraints',
        action='store_true',
        help='List all available constraint fields and exit'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (overrides config)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'parquet', 'json'],
        default=None,
        help='Output format (overrides config)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress information'
    )
    
    return parser.parse_args()


def _convert_mortgage_loans(loans_data):
    """
    Convert YAML-loaded mortgage_loans dictionaries to MortgageLoan objects.
    
    Args:
        loans_data: List of dictionaries from YAML
    
    Returns:
        List of MortgageLoan objects
    """
    if not loans_data:
        return None
    
    loans = []
    for loan_dict in loans_data:
        # Convert rate_resets if present
        rate_resets = None
        if 'rate_resets' in loan_dict and loan_dict['rate_resets']:
            rate_resets = [
                RateReset(**reset_dict)
                for reset_dict in loan_dict['rate_resets']
            ]
        
        # Create MortgageLoan object
        loan_data = loan_dict.copy()
        loan_data['rate_resets'] = rate_resets
        loans.append(MortgageLoan(**loan_data))
    
    return loans


def load_sensitivity_config(path: str) -> Dict[str, Any]:
    """
    Load sensitivity configuration from YAML file.
    
    Args:
        path: Path to YAML configuration file
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    config_path = Path(path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    # Load YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    if 'buying' not in config:
        raise ValueError("Configuration must include 'buying' section with base scenario")
    
    if 'sensitivity' not in config:
        raise ValueError("Configuration must include 'sensitivity' section")
    
    if 'variables' not in config['sensitivity']:
        raise ValueError("Sensitivity section must include 'variables'")
    
    if not config['sensitivity']['variables']:
        raise ValueError("At least one variable must be defined in sensitivity.variables")
    
    # Convert mortgage_loans from dictionaries to MortgageLoan objects
    if 'mortgage_loans' in config['buying'] and config['buying']['mortgage_loans']:
        config['buying']['mortgage_loans'] = _convert_mortgage_loans(
            config['buying']['mortgage_loans']
        )
    
    return config


def main():
    """Main CLI entry point."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Handle --list-constraints flag
        if args.list_constraints:
            print_constraint_fields(verbose=args.verbose)
            return 0
        
        # Validate that config is provided for normal operation
        if not args.config:
            print("ERROR: --config is required (or use --list-constraints to see available fields)",
                  file=sys.stderr)
            return 1
        
        if args.verbose:
            print(f"Loading configuration from: {args.config}")
        
        # Load configuration
        config = load_sensitivity_config(args.config)
        
        # Extract components
        base_buying = config['buying']
        sensitivity_config = config['sensitivity']
        tax_config = config.get('tax')
        
        # Create tax object if provided
        tax = NLHomeTax2026(**tax_config) if tax_config else None
        
        # Create base scenario
        base_scenario = BuyingScenario(**base_buying, tax=tax)
        
        if args.verbose:
            print(f"Base scenario: {config.get('name', 'Unnamed')}")
            print(f"Variables to vary: {list(sensitivity_config['variables'].keys())}")
        
        # Generate parameter space
        param_space = generate_parameter_space(sensitivity_config['variables'])
        
        if args.verbose:
            print(f"Generated {len(param_space):,} parameter combinations")
        
        # Run sensitivity analysis
        results_df = run_sensitivity_analysis(
            base_scenario=base_scenario,
            param_space=param_space,
            tax=tax,
            verbose=args.verbose
        )
        
        # Apply constraints if specified
        if 'constraints' in sensitivity_config and sensitivity_config['constraints']:
            original_count = len(results_df)
            results_df = apply_constraints(results_df, sensitivity_config['constraints'])
            
            if args.verbose:
                filtered_count = len(results_df)
                print(f"Applied constraints: {original_count:,} → {filtered_count:,} scenarios "
                      f"({filtered_count/original_count*100:.1f}% passed)")
        
        # Prepare output config
        output_config = sensitivity_config.get('output', {}).copy()
        
        # Override with command-line arguments
        if args.output:
            output_config['directory'] = args.output
        if args.format:
            output_config['format'] = args.format
        
        # Extract case file name from config path (e.g., "caseLH361_3" from "scenarios/sensitivity/caseLH361_3.yaml")
        config_path = Path(args.config)
        case_file_name = config_path.stem  # Gets filename without extension
        
        # Set defaults
        output_config.setdefault('format', 'csv')
        output_config.setdefault('directory', 'outputs/sensitivity')
        output_config.setdefault('filename_prefix', 'sensitivity_analysis')
        output_config.setdefault('include_timestamp', True)
        
        # Append case file name to the prefix
        original_prefix = output_config['filename_prefix']
        output_config['filename_prefix'] = f'{original_prefix}_{case_file_name}'
        
        # Prepare metadata
        metadata = {
            'config_file': args.config,
            'scenario_name': config.get('name', 'Unnamed'),
            'total_combinations': len(param_space),
            'results_count': len(results_df),
            'variables': list(sensitivity_config['variables'].keys()),
            'constraints': sensitivity_config.get('constraints', [])
        }
        
        # Export results
        output_path = export_results(results_df, output_config, metadata=metadata)
        
        print(f"\n{'='*60}")
        print(f"Sensitivity Analysis Complete")
        print(f"{'='*60}")
        print(f"Results saved to: {output_path}")
        print(f"Total scenarios evaluated: {len(results_df):,}")
        
        # Print summary statistics
        if args.verbose and len(results_df) > 0:
            print(f"\n{'='*60}")
            print(f"Summary Statistics")
            print(f"{'='*60}")
            
            key_metrics = ['wealth_end', 'avg_net_cost_per_month', 'cash_required_upfront']
            available_metrics = [m for m in key_metrics if m in results_df.columns]
            
            if available_metrics:
                summary = results_df[available_metrics].describe()
                print(summary.to_string())
        
        return 0
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    
    except ValueError as e:
        print(f"ERROR: Invalid configuration - {e}", file=sys.stderr)
        return 1
    
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
