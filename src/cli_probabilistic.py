"""
CLI for Probabilistic Mortgage Comparison

This module provides a command-line interface for comparing multiple
mortgage scenarios under shared uncertainty.
"""

import argparse
import sys
from pathlib import Path

from src.probabilistic import (
    run_multi_scenario_comparison,
    load_shared_variables_config,
    export_comparison_results
)


def main():
    """
    Main CLI entry point for probabilistic mortgage comparison.
    
    Example usage:
        # Compare three mortgage scenarios with shared uncertainty
        python -m src.cli_probabilistic \
          --scenarios \
            scenarios/mortgage_scenario_a.yaml \
            scenarios/mortgage_scenario_b.yaml \
            scenarios/mortgage_scenario_c.yaml \
          --variables scenarios/probabilistic/shared_variables.yaml \
          --n-samples 5000 \
          --seed 42 \
          --output outputs/probabilistic \
          --verbose
        
        # Using mixture distributions
        python -m src.cli_probabilistic \
          --scenarios scenarios/mortgage_scenario_a.yaml \
          --variables scenarios/probabilistic/shared_mixture_variables.yaml \
          --verbose
    """
    parser = argparse.ArgumentParser(
        description='Compare mortgage scenarios probabilistically using Monte Carlo simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare three scenarios with 1000 samples
  python -m src.cli_probabilistic \\
    --scenarios scenarios/mortgage_scenario_a.yaml scenarios/mortgage_scenario_b.yaml \\
    --variables scenarios/probabilistic/shared_variables.yaml \\
    --n-samples 1000 \\
    --seed 42

  # Use mixture distributions for realistic uncertainty
  python -m src.cli_probabilistic \\
    --scenarios scenarios/mortgage_scenario_a.yaml \\
    --variables scenarios/probabilistic/shared_mixture_variables.yaml \\
    --verbose

  # Compare all scenarios in a directory
  python -m src.cli_probabilistic \\
    --scenarios scenarios/mortgage_*.yaml \\
    --variables scenarios/probabilistic/shared_variables.yaml

Variables file format (YAML):
  variables:
    annual_value_growth:
      strategy: "normal"
      mean: 0.02
      std: 0.015
      samples: 5000
      seed: 42
    
    "mortgage_loans[*].rate_resets[0].new_rate_default":
      strategy: "mixture"
      samples: 5000
      seed: 42
      components:
        - weight: 0.40
          mean: 0.030
          std: 0.009
        - weight: 0.35
          mean: 0.041
          std: 0.010
        - weight: 0.25
          mean: 0.060
          std: 0.014
      min: 0.005
      max: 0.09
        """
    )
    
    parser.add_argument(
        '--scenarios',
        nargs='+',
        required=True,
        help='Paths to scenario YAML files to compare (space-separated)'
    )
    
    parser.add_argument(
        '--variables',
        required=True,
        help='Path to shared variables configuration YAML file'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=None,
        help='Number of Monte Carlo samples (overrides value in variables config)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (overrides value in variables config)'
    )
    
    parser.add_argument(
        '--output',
        default='outputs/probabilistic',
        help='Output directory for results (default: outputs/probabilistic)'
    )
    
    parser.add_argument(
        '--filename-prefix',
        default='comparison',
        help='Prefix for output filename (default: comparison)'
    )
    
    parser.add_argument(
        '--no-timestamp',
        action='store_true',
        help='Do not include timestamp in output filename'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    
    args = parser.parse_args()
    
    # Validate scenario files exist
    for scenario_path in args.scenarios:
        if not Path(scenario_path).exists():
            print(f"Error: Scenario file not found: {scenario_path}", file=sys.stderr)
            sys.exit(1)
    
    # Validate variables file exists
    if not Path(args.variables).exists():
        print(f"Error: Variables file not found: {args.variables}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Load shared variables configuration
        if args.verbose:
            print(f"Loading shared variables from: {args.variables}")
        
        shared_variables = load_shared_variables_config(args.variables)
        
        if args.verbose:
            print(f"Loaded {len(shared_variables)} shared variable(s):")
            for var_name, var_config in shared_variables.items():
                strategy = var_config.get('strategy', 'unknown')
                samples = var_config.get('samples', 'N/A')
                print(f"  - {var_name}: {strategy} (samples: {samples})")
        
        # Run comparison
        print(f"\nComparing {len(args.scenarios)} scenario(s)...")
        
        results_df = run_multi_scenario_comparison(
            scenario_configs=args.scenarios,
            shared_variables=shared_variables,
            n_samples=args.n_samples,
            seed=args.seed,
            verbose=args.verbose
        )
        
        # Export results
        output_path = export_comparison_results(
            df=results_df,
            output_dir=args.output,
            filename_prefix=args.filename_prefix,
            include_timestamp=not args.no_timestamp
        )
        
        print(f"\n{'='*80}")
        print(f"RESULTS SAVED")
        print(f"{'='*80}")
        print(f"Output file: {output_path}")
        print(f"Total evaluations: {len(results_df):,}")
        print(f"Scenarios compared: {results_df['mortgage_scenario_label'].nunique()}")
        print(f"Samples per scenario: {len(results_df) // results_df['mortgage_scenario_label'].nunique():,}")
        
        # Print summary statistics
        print(f"\n{'='*80}")
        print(f"SUMMARY STATISTICS")
        print(f"{'='*80}")
        
        summary = results_df.groupby('mortgage_scenario_label')['wealth_end'].agg([
            'count', 'mean', 'std', 'min', 
            ('p10', lambda x: x.quantile(0.10)),
            ('p50', lambda x: x.quantile(0.50)),
            ('p90', lambda x: x.quantile(0.90)),
            'max'
        ])
        
        print("\nWealth at End (by scenario):")
        print(summary.to_string())
        
        print(f"\n{'='*80}")
        print("Next steps:")
        print("  1. Analyze results with src.probabilistic_analysis")
        print("  2. Calculate P(scenario_i > scenario_j) with calculate_comparative_metrics()")
        print("  3. Identify regimes with identify_regime()")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
