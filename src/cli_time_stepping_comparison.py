"""
CLI for Time-Stepping Multi-Scenario Comparison

This module provides a command-line interface for comparing multiple
mortgage scenarios using time-stepping simulation with Markov regime transitions.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import traceback
import glob

from src.markov_regime import MarkovChainConfig
from src.time_stepping_comparison import (
    run_multi_scenario_comparison,
    export_comparison_results
)
from src.analyse.comparison_metrics import (
    calculate_comparative_probabilities,
    calculate_pairwise_statistics,
    export_regime_statistics
)
from src.analyse.comparison_plots import generate_comparison_plots
from src.analyse.comparison_report import generate_comparison_report


def main():
    """
    Main CLI entry point for time-stepping comparison.
    
    Example usage:
        python -m src.cli_time_stepping_comparison \
          --markov-config scenarios/markov/default_regimes.yaml \
          --regime-distributions scenarios/markov/regime_distributions.yaml \
          --scenarios \
            scenarios/case07b10_scenario_a.yaml \
            scenarios/case07b10_scenario_b.yaml \
            scenarios/case07b10_scenario_c.yaml \
          --n-samples 5000 \
          --seed 42 \
          --output outputs/time_stepping_comparison/
    """
    parser = argparse.ArgumentParser(
        description='Compare mortgage scenarios with time-stepping Markov simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare scenarios using expert matrix (default)
  python -m src.cli_time_stepping_comparison \\
    --scenarios scenarios/case07b10_*.yaml \\
    --n-samples 5000 \\
    --seed 42

  # Compare using calibrated matrix from BIS data
  python -m src.cli_time_stepping_comparison \\
    --matrix-type calibrated \\
    --scenarios scenarios/case07b10_*.yaml \\
    --n-samples 5000 \\
    --seed 42

  # Compare using empirical matrix (no prior smoothing)
  python -m src.cli_time_stepping_comparison \\
    --matrix-type empirical \\
    --scenarios scenarios/case07b10_*.yaml \\
    --n-samples 5000 \\
    --seed 42

  # Use custom matrix configuration
  python -m src.cli_time_stepping_comparison \\
    --markov-config path/to/custom_matrix.yaml \\
    --scenarios scenarios/case07b10_*.yaml \\
    --n-samples 5000 \\
    --seed 42

  # Quick test with 100 samples
  python -m src.cli_time_stepping_comparison \\
    --scenarios scenarios/case07b10_scenario_a.yaml scenarios/case07b10_scenario_b.yaml \\
    --n-samples 100 \\
    --seed 42 \\
    --quick

Output files:
  - trajectories.parquet: Year-by-year simulation data
  - summary.parquet: Summary statistics per sample
  - comparison_report.md: Comprehensive markdown report
  - comparison_plots.pdf: Visualization plots
  - regime_statistics.csv: Regime-level statistics
  - pairwise_comparisons.csv: Detailed pairwise comparisons
        """
    )
    
    # Matrix type selection (mutually exclusive with --markov-config)
    matrix_group = parser.add_mutually_exclusive_group()
    matrix_group.add_argument(
        '--matrix-type',
        choices=['expert', 'calibrated', 'empirical'],
        default='expert',
        help='Transition matrix type: expert (default_regimes.yaml), calibrated (calibrated_regimes.yaml), or empirical (empirical_matrix.yaml). Default: expert'
    )
    matrix_group.add_argument(
        '--markov-config',
        help='Path to custom Markov chain configuration (transition matrix). Overrides --matrix-type.'
    )
    
    parser.add_argument(
        '--regime-distributions',
        default='scenarios/markov/regime_distributions.yaml',
        help='Path to regime distributions configuration (default: scenarios/markov/regime_distributions.yaml)'
    )
    
    parser.add_argument(
        '--scenarios',
        nargs='+',
        required=True,
        help='Paths to scenario YAML files to compare (space-separated)'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=1000,
        help='Number of Monte Carlo samples (default: 1000)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--output',
        default='outputs/timestepping_comparison',
        help='Output directory prefix (default: outputs/timestepping_comparison). Will be suffixed with number of scenarios and timestamp.'
    )
    
    parser.add_argument(
        '--format',
        choices=['parquet', 'csv'],
        default='csv',
        help='Output format for data files (default: csv)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: skip plots and detailed analysis'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress information'
    )
    
    args = parser.parse_args()
    
    # Expand wildcards in scenario paths
    expanded_scenarios = []
    for pattern in args.scenarios:
        # Use glob to expand wildcards
        matches = glob.glob(pattern)
        if matches:
            # Sort for consistent ordering
            expanded_scenarios.extend(sorted(matches))
        else:
            # If no matches, keep the original pattern (will fail validation below)
            expanded_scenarios.append(pattern)
    
    # Remove duplicates while preserving order
    seen = set()
    args.scenarios = []
    for scenario in expanded_scenarios:
        if scenario not in seen:
            seen.add(scenario)
            args.scenarios.append(scenario)
    
    # Determine markov config path based on matrix type or custom path
    if args.markov_config:
        markov_config_path = args.markov_config
    else:
        # Map matrix type to file
        matrix_files = {
            'expert': 'scenarios/markov/default_regimes.yaml',
            'calibrated': 'scenarios/markov/calibrated_regimes.yaml',
            'empirical': 'scenarios/markov/empirical_matrix.yaml'
        }
        markov_config_path = matrix_files[args.matrix_type]
    markov_str = Path(markov_config_path).name.split('_')[0]
    
    # Validate inputs
    if not Path(markov_config_path).exists():
        print(f"Error: Markov config not found: {markov_config_path}", file=sys.stderr)
        sys.exit(1)
    
    if not Path(args.regime_distributions).exists():
        print(f"Error: Regime distributions not found: {args.regime_distributions}", file=sys.stderr)
        sys.exit(1)
    
    for scenario_path in args.scenarios:
        if not Path(scenario_path).exists():
            print(f"Error: Scenario not found: {scenario_path}", file=sys.stderr)
            sys.exit(1)
    
    # Create output directory with number of scenarios and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_scenarios = len(args.scenarios)
    output_dir_name = f"{args.output}_{n_scenarios}s_{markov_str}_{timestamp}"
    output_dir = Path(output_dir_name)
    
    if args.verbose:
        print("=" * 80)
        print("Time-Stepping Multi-Scenario Comparison")
        print("=" * 80)
        if args.markov_config:
            print(f"Markov config: {markov_config_path} (custom)")
        else:
            print(f"Markov config: {markov_config_path} (matrix type: {args.matrix_type})")
        print(f"Regime distributions: {args.regime_distributions}")
        print(f"Scenarios: {len(args.scenarios)}")
        for i, scenario in enumerate(args.scenarios, 1):
            print(f"  {i}. {Path(scenario).name}")
        print(f"Samples: {args.n_samples:,}")
        print(f"Seed: {args.seed}")
        print(f"Output: {output_dir}")
        print("=" * 80)
        print()
    
    # Load Markov configuration
    if args.verbose:
        print("Loading Markov chain configuration...")
    
    try:
        markov_config = MarkovChainConfig.from_yaml(
            markov_config_path,
            args.regime_distributions
        )
    except Exception as e:
        print(f"Error loading Markov configuration: {e}", file=sys.stderr)
        sys.exit(1)
    
    if args.verbose:
        print(f"  Regimes: {', '.join(markov_config.regimes)}")
        print(f"  Time step: {markov_config.time_step_years} year(s)")
        print()
    
    # Run comparison
    if args.verbose:
        print("Running multi-scenario comparison...")
        print()
    
    try:
        trajectories_df, summary_df = run_multi_scenario_comparison(
            scenario_configs=args.scenarios,
            markov_config=markov_config,
            n_samples=args.n_samples,
            seed=args.seed,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"Error running comparison: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    
    # Export results
    if args.verbose:
        print()
        print("Exporting results...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export data files
    export_comparison_results(
        trajectories_df,
        summary_df,
        str(output_dir),
        format=args.format
    )
    
    if args.verbose:
        print(f"  ✓ Saved trajectories.{args.format}")
        print(f"  ✓ Saved summary.{args.format}")
    
    # Calculate and export additional metrics
    if not args.quick:
        # Pairwise comparisons
        pairwise_stats = calculate_pairwise_statistics(
            summary_df,
            metrics=['final_wealth', 'total_interest_paid']
        )
        pairwise_stats.to_csv(output_dir / 'pairwise_comparisons.csv', index=False)
        
        if args.verbose:
            print(f"  ✓ Saved pairwise_comparisons.csv")
        
        # Regime statistics
        export_regime_statistics(
            summary_df,
            output_dir / 'regime_statistics.csv'
        )
        
        if args.verbose:
            print(f"  ✓ Saved regime_statistics.csv")
        
        # Generate report
        if args.verbose:
            print()
            print("Generating comparison report...")
        
        # Determine horizon from first scenario
        horizon_years = trajectories_df['year'].max()
        
        generate_comparison_report(
            summary_df,
            trajectories_df,
            output_dir / 'comparison_report.md',
            n_samples=args.n_samples,
            horizon_years=horizon_years,
            markov_config=markov_config
        )
        
        if args.verbose:
            print(f"  ✓ Saved comparison_report.md")
        
        # Generate plots
        if args.verbose:
            print()
            print("Generating comparison plots...")
        
        prob_matrix = calculate_comparative_probabilities(summary_df, 'final_wealth')
        
        generate_comparison_plots(
            trajectories_df,
            summary_df,
            output_dir / 'comparison_plots.pdf',
            prob_matrix=prob_matrix
        )
        
        if args.verbose:
            print(f"  ✓ Saved comparison_plots.pdf")
    
    # Summary
    if args.verbose:
        print()
        print("=" * 80)
        print("Comparison complete!")
        print("=" * 80)
        print()
        print("Quick summary:")
        print()
        
        # Show mean wealth by scenario
        mean_wealth = summary_df.groupby('scenario_label')['final_wealth'].mean()
        print("Mean final wealth by scenario:")
        for scenario, wealth in mean_wealth.sort_values(ascending=False).items():
            print(f"  {scenario}: €{wealth:,.0f}")
        
        print()
        print(f"Full results saved to: {output_dir}")
        print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
