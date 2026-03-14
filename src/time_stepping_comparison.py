"""
Multi-Scenario Time-Stepping Comparison Module

This module orchestrates the comparison of multiple mortgage scenarios
under shared regime paths, ensuring fair comparison where differences
are due to mortgage structure, not random economic variation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import yaml

from src.estate import BuyingScenario
from src.markov_regime import (
    MarkovChainConfig,
    generate_shared_regime_paths,
    generate_shared_parameter_draws
)
from src.time_stepping_simulator import simulate_scenario_with_fixed_paths
from src.tax import NLHomeTax2026
from src.mortgage import MortgageLoan, AmortizationType, RateReset


def load_scenario(scenario_path: str) -> Tuple[BuyingScenario, str]:
    """
    Load a mortgage scenario from YAML file.
    
    Args:
        scenario_path: Path to scenario YAML file
    
    Returns:
        Tuple of (BuyingScenario, label)
    """
    with open(scenario_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle nested structure (buying section)
    if 'buying' in config:
        buying_config = config['buying']
        tax_config = config.get('tax', {})
        label = config.get('name', Path(scenario_path).stem)
    else:
        # Flat structure
        buying_config = config
        tax_config = config.get('tax', {})
        label = config.get('description', Path(scenario_path).stem)
    
    # Use buying_config instead of config for the rest
    config = buying_config
    
    # Parse mortgage loans if present
    mortgage_loans = None
    if 'mortgage_loans' in config:
        mortgage_loans = []
        for loan_config in config['mortgage_loans']:
            term_years = loan_config['term_years']
            fixed_period_years = loan_config.get('fixed_period_years')
            
            # Parse rate resets if present
            rate_resets = None
            if 'rate_resets' in loan_config and loan_config['rate_resets'] is not None:
                rate_resets = []
                for reset_config in loan_config['rate_resets']:
                    rate_resets.append(RateReset(
                        month=reset_config.get('month', reset_config.get('reset_month')),
                        new_rate_min=reset_config.get('new_rate_min', 0.01),
                        new_rate_max=reset_config.get('new_rate_max', 0.10),
                        new_rate_default=reset_config.get('new_rate_default')
                    ))
            elif fixed_period_years is not None and fixed_period_years < term_years:
                # Auto-generate rate resets based on fixed_period_years
                # Generate resets at each fixed period boundary until term ends
                rate_resets = []
                reset_month = fixed_period_years * 12
                
                # Get default rate bounds from first explicit reset if any, otherwise use defaults
                default_min = 0.02
                default_max = 0.08
                default_rate = loan_config['annual_rate']  # Use initial rate as default
                
                while reset_month < term_years * 12:
                    rate_resets.append(RateReset(
                        month=reset_month,
                        new_rate_min=default_min,
                        new_rate_max=default_max,
                        new_rate_default=default_rate
                    ))
                    reset_month += fixed_period_years * 12
            
            # Parse amortization type
            amort_type_str = loan_config.get('amortization_type', 'ANNUITY').upper()
            amort_type = AmortizationType[amort_type_str]
            
            mortgage_loans.append(MortgageLoan(
                principal=loan_config.get('principal'),
                percentage=loan_config.get('percentage', loan_config.get('principal_percentage')),
                annual_rate=loan_config['annual_rate'],
                term_years=term_years,
                amortization_type=amort_type,
                fixed_period_years=fixed_period_years,
                rate_resets=rate_resets,
                label=loan_config.get('label', 'Loan')
            ))
    
    # Parse tax configuration if present
    tax = None
    if tax_config:
        tax = NLHomeTax2026(
            woz_value=tax_config.get('woz_value'),
            ewf_rate_main=tax_config.get('ewf_rate_main', 0.0035),
            villa_threshold=tax_config.get('villa_threshold', 1_350_000.0),
            ewf_rate_above_threshold=tax_config.get('ewf_rate_above_threshold', 0.0235),
            deduction_cap_rate=tax_config.get('deduction_cap_rate', 0.3756),
            marginal_tax_rate=tax_config.get('marginal_tax_rate', 0.3756),
            apply_wet_hillen=tax_config.get('apply_wet_hillen', True),
            wet_hillen_factor_2026=tax_config.get('wet_hillen_factor_2026', 0.71867),
            rental_income_tax_rate=tax_config.get('rental_income_tax_rate', 0.0)
        )
    
    # Create scenario
    scenario = BuyingScenario(
        purchase_price=config['purchase_price'],
        mortgage_principal=config.get('mortgage_principal'),
        mortgage_annual_rate=config.get('mortgage_annual_rate'),
        mortgage_term_years=config.get('mortgage_term_years'),
        mortgage_loans=mortgage_loans,
        living_months=config.get('living_months', 360),
        one_off_costs=config.get('one_off_costs', 0.0),
        renovation_costs_once=config.get('renovation_costs_once', 0.0),
        monthly_vve=config.get('monthly_vve', 0.0),
        monthly_utilities=config.get('monthly_utilities', 0.0),
        monthly_rent_income=config.get('monthly_rent_income', 0.0),
        months_rented=config.get('months_rented', 0),
        annual_value_growth=config.get('annual_value_growth', 0.0),
        sold_at_end=config.get('sold_at_end', False),
        selling_cost_rate=config.get('selling_cost_rate', 0.0),
        selling_cost_fixed=config.get('selling_cost_fixed', 0.0),
        tax=tax,
        description=config.get('description'),
        link=config.get('link')
    )
    
    return scenario, label


def run_multi_scenario_comparison(
    scenario_configs: List[str],
    markov_config: MarkovChainConfig,
    n_samples: int,
    seed: int,
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run time-stepping comparison of multiple mortgage scenarios.
    
    Key feature: All scenarios experience the SAME regime path for each sample.
    This ensures fair comparison (differences are due to mortgage structure,
    not random variation in economic conditions).
    
    Args:
        scenario_configs: List of paths to scenario YAML files
        markov_config: Markov chain configuration (transition matrix + distributions)
        n_samples: Number of Monte Carlo samples
        seed: Random seed
        verbose: Print progress
    
    Returns:
        Tuple of (trajectories_df, summary_df)
            - trajectories_df: Long format (sample × year × scenario)
            - summary_df: Wide format (sample × scenario)
    """
    if verbose:
        print(f"Generating {n_samples} shared regime paths...")
    
    # Determine horizon from first scenario
    first_scenario, _ = load_scenario(scenario_configs[0])
    horizon_years = first_scenario.living_months // 12
    
    # 1. Generate shared regime paths
    regime_paths = generate_shared_regime_paths(
        markov_config,
        n_samples,
        horizon_years,
        seed=seed
    )
    
    if verbose:
        print(f"Sampling parameters for {horizon_years} years...")
    
    # 2. Generate shared parameter draws
    parameter_draws = generate_shared_parameter_draws(
        regime_paths,
        markov_config,
        seed=seed
    )
    
    # 3. Simulate each scenario with the shared paths
    all_trajectories = []
    all_summaries = []
    
    for i, scenario_path in enumerate(scenario_configs):
        if verbose:
            print(f"Simulating scenario {i+1}/{len(scenario_configs)}: {Path(scenario_path).name}")
        
        scenario, label = load_scenario(scenario_path)
        
        # Simulate with pre-determined regime paths and parameters
        traj_df, summ_df = simulate_scenario_with_fixed_paths(
            scenario,
            regime_paths,
            parameter_draws,
            scenario_label=label
        )
        
        # Add scenario file path for reference
        traj_df['scenario_file'] = scenario_path
        summ_df['scenario_file'] = scenario_path
        
        all_trajectories.append(traj_df)
        all_summaries.append(summ_df)
    
    # 4. Combine results
    if verbose:
        print("Combining results...")
    
    trajectories_df = pd.concat(all_trajectories, ignore_index=True)
    summary_df = pd.concat(all_summaries, ignore_index=True)
    
    if verbose:
        print(f"Complete! Generated {len(trajectories_df)} trajectory records")
        print(f"and {len(summary_df)} summary records")
    
    return trajectories_df, summary_df


def export_comparison_results(
    trajectories_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: str,
    format: str = 'parquet'
) -> None:
    """
    Export comparison results to files.
    
    Args:
        trajectories_df: Trajectory data
        summary_df: Summary data
        output_dir: Output directory path
        format: Output format ('parquet' or 'csv')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if format == 'parquet':
        trajectories_df.to_parquet(output_path / 'trajectories.parquet', index=False)
        summary_df.to_parquet(output_path / 'summary.parquet', index=False)
    elif format == 'csv':
        trajectories_df.to_csv(output_path / 'trajectories.csv', index=False)
        summary_df.to_csv(output_path / 'summary.csv', index=False)
    else:
        raise ValueError(f"Unknown format: {format}")
