"""
Comparison Report Module

This module generates comprehensive markdown reports summarizing
the comparison of multiple mortgage scenarios.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd

from src.analyse.comparison_metrics import (
    calculate_comparative_probabilities,
    calculate_dominance_statistics,
    calculate_regime_conditional_performance,
    calculate_exit_timing_analysis
)


def format_currency(value: float) -> str:
    """Format value as currency."""
    return f"€{value:,.0f}"


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value*100:.1f}%"


def extract_config_differences(summary_df: pd.DataFrame) -> tuple:
    """
    Extract configuration differences from scenario files.
    
    Args:
        summary_df: Summary DataFrame with scenario_file column
    
    Returns:
        Dictionary mapping scenario labels to their configurations
    """
    import yaml
    
    configs = {}
    scenario_files = summary_df[['scenario_label', 'scenario_file']].drop_duplicates()
    
    for _, row in scenario_files.iterrows():
        label = row['scenario_label']
        file_path = row['scenario_file']
        
        try:
            with open(file_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check if this is a rental scenario
            is_rental = 'rent_monthly' in config
            
            if is_rental:
                # Rental scenario configuration
                config_summary = {
                    'scenario_type': 'rental',
                    'rent_monthly': config.get('rent_monthly'),
                    'utilities_monthly': config.get('utilities_monthly'),
                    'living_months': config.get('living_months', 360),
                    'annual_rent_increase': config.get('annual_rent_increase', 0.0),
                    'description': config.get('description')
                }
            else:
                # Buying scenario configuration
                if 'buying' in config:
                    buying_config = config['buying']
                else:
                    buying_config = config
                
                config_summary = {
                    'scenario_type': 'buying',
                    'purchase_price': buying_config.get('purchase_price'),
                    'mortgage_principal': buying_config.get('mortgage_principal'),
                    'mortgage_annual_rate': buying_config.get('mortgage_annual_rate'),
                    'mortgage_term_years': buying_config.get('mortgage_term_years'),
                    'living_months': buying_config.get('living_months', 360),
                    'one_off_costs': buying_config.get('one_off_costs', 0.0),
                    'monthly_vve': buying_config.get('monthly_vve', 0.0),
                    'monthly_utilities': buying_config.get('monthly_utilities', 0.0),
                    'sold_at_end': buying_config.get('sold_at_end', False),
                    'mortgage_loans': buying_config.get('mortgage_loans', [])
                }
            
            configs[label] = config_summary
        except Exception as e:
            configs[label] = {'error': str(e)}
    
    # Identify differences
    if len(configs) > 1:
        all_keys = set()
        for config in configs.values():
            if 'error' not in config:
                all_keys.update(config.keys())
        
        differences = {}
        for key in all_keys:
            if key == 'mortgage_loans':
                continue  # Handle separately
            values = {label: config.get(key) for label, config in configs.items() if 'error' not in config}
            unique_values = set(str(v) for v in values.values())
            if len(unique_values) > 1:  # Only include if values differ
                differences[key] = values
        
        # Check mortgage loans differences
        loan_differences = {}
        max_loans = max(len(config.get('mortgage_loans', [])) for config in configs.values() if 'error' not in config)
        for loan_idx in range(max_loans):
            for loan_key in ['principal', 'percentage', 'annual_rate', 'term_years', 'amortization_type', 'fixed_period_years', 'label']:
                values = {}
                for label, config in configs.items():
                    if 'error' in config:
                        continue
                    loans = config.get('mortgage_loans', [])
                    if loan_idx < len(loans):
                        values[label] = loans[loan_idx].get(loan_key)
                
                if values:
                    unique_values = set(str(v) for v in values.values())
                    if len(unique_values) > 1:
                        loan_differences[f'loan_{loan_idx+1}_{loan_key}'] = values
        
        differences.update(loan_differences)
    else:
        differences = {}
    
    return configs, differences


def generate_comparison_report(
    summary_df: pd.DataFrame,
    trajectories_df: pd.DataFrame,
    output_path: str,
    n_samples: int,
    horizon_years: int,
    markov_config: Optional['MarkovChainConfig'] = None,
    matrix_type: Optional[str] = None
) -> None:
    """
    Generate comprehensive markdown comparison report.
    
    Args:
        summary_df: Summary DataFrame
        trajectories_df: Trajectory DataFrame
        output_path: Output file path
        n_samples: Number of Monte Carlo samples
        horizon_years: Simulation horizon in years
        markov_config: Optional Markov chain configuration for documenting simulation parameters
        matrix_type: Optional matrix type identifier ('expert', 'calibrated', 'empirical', or custom name)
    """
    scenarios = sorted(summary_df['scenario_label'].unique())
    
    # Calculate metrics
    prob_matrix = calculate_comparative_probabilities(summary_df, 'final_wealth')
    dominance_stats = calculate_dominance_statistics(summary_df, ['final_wealth'])
    regime_perf = calculate_regime_conditional_performance(summary_df)
    exit_timing = calculate_exit_timing_analysis(summary_df)
    
    # Extract configuration differences
    configs, differences = extract_config_differences(summary_df)
    
    # Start building report
    lines = []
    lines.append("# Mortgage Scenario Comparison Report")
    lines.append("")
    lines.append(f"- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Samples**: {n_samples:,}")
    lines.append(f"- **Horizon**: {horizon_years} years")
    lines.append(f"- **Scenarios**: {', '.join(scenarios)}")
    lines.append("")
    
    # Simulation Configuration
    if markov_config is not None:
        lines.append("## Simulation Configuration")
        lines.append("")
        lines.append("### Markov Chain Parameters")
        lines.append("")
        lines.append(f"- **Time Step**: {markov_config.time_step_years} year(s)")
        lines.append(f"- **Regimes**: {', '.join(markov_config.regimes)}")
        lines.append(f"- **Random Seed**: {n_samples} samples")
        lines.append("")
        
        # Transition matrix
        lines.append("### Regime Transition Matrix")
        lines.append("")
        lines.append("Probability of transitioning from row regime to column regime:")
        lines.append("")
        
        header = "| From \\ To |" + " | ".join(markov_config.regimes) + " |"
        separator = "|" + "|".join(["---"] * (len(markov_config.regimes) + 1)) + "|"
        lines.append(header)
        lines.append(separator)
        
        for i, from_regime in enumerate(markov_config.regimes):
            row = f"| **{from_regime}** |"
            for j, to_regime in enumerate(markov_config.regimes):
                prob = markov_config.transition_matrix[i, j]
                row += f" {prob:.3f} |"
            lines.append(row)
        
        lines.append("")
        
        # Regime distributions
        lines.append("### Regime Parameter Distributions")
        lines.append("")
        
        for regime in markov_config.regimes:
            macro_regime = markov_config.regime_distributions.get(regime)
            if macro_regime:
                lines.append(f"**{regime}**:")
                lines.append("")
                
                # MacroRegime is a dataclass with specific fields
                lines.append(f"- Property Growth: Normal(μ={macro_regime.property_growth_mean:.4f}, σ={macro_regime.property_growth_std:.4f})")
                lines.append(f"- Mortgage Rate: Normal(μ={macro_regime.mortgage_rate_mean:.4f}, σ={macro_regime.mortgage_rate_std:.4f})")
                
                lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Key Differences Between Scenarios (HIGHLIGHTED)
    if differences:
        lines.append("## 🔍 KEY DIFFERENCES BETWEEN SCENARIOS")
        lines.append("")
        lines.append("**This table shows ONLY the parameters that differ between scenarios.**")
        lines.append("All other parameters are identical across scenarios.")
        lines.append("")
        
        # Build comparison table
        header = "| Parameter |" + " | ".join(scenarios) + " |"
        separator = "|" + "|".join(["---"] * (len(scenarios) + 1)) + "|"
        lines.append(header)
        lines.append(separator)
        
        for param, values in sorted(differences.items()):
            param_display = param.replace('_', ' ').title()
            row = f"| **{param_display}** |"
            
            for scenario in scenarios:
                value = values.get(scenario)
                if value is None:
                    row += " - |"
                elif param.endswith('_rate') or param == 'annual_rate':
                    row += f" {float(value)*100:.2f}% |"
                elif param.endswith('_price') or param.endswith('_principal') or param.endswith('_costs') or param.endswith('_vve') or param.endswith('_utilities'):
                    row += f" {format_currency(float(value))} |"
                elif param.endswith('_years') or param.endswith('_months'):
                    row += f" {value} |"
                elif param == 'percentage':
                    row += f" {float(value)*100:.1f}% |"
                else:
                    row += f" {value} |"
            
            lines.append(row)
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Full Configuration Details
    lines.append("## Full Scenario Configuration Details")
    lines.append("")
    lines.append("Complete configuration for each scenario (for reference).")
    lines.append("")
    
    for scenario in scenarios:
        config = configs.get(scenario, {})
        lines.append(f"### {scenario}")
        lines.append("")
        
        if 'error' in config:
            lines.append(f"*Error loading configuration: {config['error']}*")
            lines.append("")
            continue
        
        # Check if this is a rental scenario
        if config.get('scenario_type') == 'rental':
            lines.append("| Parameter | Value |")
            lines.append("|-----------|-------|")
            lines.append(f"| Scenario Type | Rental |")
            lines.append(f"| Monthly Rent | {format_currency(config.get('rent_monthly', 0))} |")
            lines.append(f"| Monthly Utilities | {format_currency(config.get('utilities_monthly', 0))} |")
            lines.append(f"| Living Duration | {config.get('living_months', 360)} months ({config.get('living_months', 360)//12} years) |")
            lines.append(f"| Annual Rent Increase | {config.get('annual_rent_increase', 0)*100:.2f}% |")
            if config.get('description'):
                lines.append(f"| Description | {config.get('description')} |")
        else:
            # Buying scenario
            lines.append("| Parameter | Value |")
            lines.append("|-----------|-------|")
            lines.append(f"| Scenario Type | Buying |")
            lines.append(f"| Purchase Price | {format_currency(config.get('purchase_price', 0))} |")
            
            if config.get('mortgage_principal'):
                lines.append(f"| Mortgage Principal | {format_currency(config.get('mortgage_principal', 0))} |")
            
            if config.get('mortgage_annual_rate'):
                lines.append(f"| Mortgage Annual Rate | {config.get('mortgage_annual_rate', 0)*100:.2f}% |")
            
            if config.get('mortgage_term_years'):
                lines.append(f"| Mortgage Term | {config.get('mortgage_term_years', 0)} years |")
            
            lines.append(f"| Living Duration | {config.get('living_months', 360)} months ({config.get('living_months', 360)//12} years) |")
            lines.append(f"| One-off Costs | {format_currency(config.get('one_off_costs', 0))} |")
            lines.append(f"| Monthly VVE | {format_currency(config.get('monthly_vve', 0))} |")
            lines.append(f"| Monthly Utilities | {format_currency(config.get('monthly_utilities', 0))} |")
            lines.append(f"| Sold at End | {config.get('sold_at_end', False)} |")
            
            # Mortgage loans details
            mortgage_loans = config.get('mortgage_loans', [])
            if mortgage_loans:
                lines.append("")
                lines.append("**Mortgage Loans:**")
                lines.append("")
                for idx, loan in enumerate(mortgage_loans, 1):
                    lines.append(f"*Loan {idx}: {loan.get('label', 'Unnamed')}*")
                    lines.append("")
                    lines.append("| Parameter | Value |")
                    lines.append("|-----------|-------|")
                    
                    if loan.get('principal'):
                        lines.append(f"| Principal | {format_currency(loan.get('principal', 0))} |")
                    elif loan.get('percentage'):
                        lines.append(f"| Principal | {loan.get('percentage', 0)*100:.1f}% of purchase price |")
                    
                    lines.append(f"| Annual Rate | {loan.get('annual_rate', 0)*100:.2f}% |")
                    lines.append(f"| Term | {loan.get('term_years', 0)} years |")
                    lines.append(f"| Amortization Type | {loan.get('amortization_type', 'ANNUITY')} |")
                    
                    if loan.get('fixed_period_years'):
                        lines.append(f"| Fixed Period | {loan.get('fixed_period_years', 0)} years |")
                    
                    if loan.get('rate_resets'):
                        lines.append(f"| Rate Resets | {len(loan.get('rate_resets', []))} scheduled |")
                    
                    lines.append("")
        
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    
    # Identify rental baseline
    rental_baseline = None
    buying_scenarios = []
    for scenario in scenarios:
        if 'rental' in scenario.lower() and 'baseline' in scenario.lower():
            rental_baseline = scenario
        else:
            buying_scenarios.append(scenario)
    
    # Rental Baseline Comparison (if present)
    if rental_baseline:
        lines.append("### 🏠 Rental Baseline Comparison")
        lines.append("")
        lines.append("Comparison of buying scenarios against the rental baseline:")
        lines.append("")
        
        rental_data = summary_df[summary_df['scenario_label'] == rental_baseline]
        rental_wealth = rental_data['final_wealth'].mean()
        rental_traj = trajectories_df[trajectories_df['scenario_label'] == rental_baseline]
        rental_cost = rental_traj[rental_traj['year'] > 0]['monthly_net_housing_cost'].mean()  # Exclude Year 0
        
        lines.append("| Scenario | Final Wealth | Wealth vs Rental | Avg Net Housing Cost | Cost vs Rental |")
        lines.append("|----------|--------------|------------------|----------------------|----------------|")
        
        # Rental baseline row
        lines.append(f"| **{rental_baseline}** | {format_currency(rental_wealth)} | - | {format_currency(rental_cost)} | - |")
        
        # Buying scenarios
        for scenario in buying_scenarios:
            scenario_data = summary_df[summary_df['scenario_label'] == scenario]
            mean_wealth = scenario_data['final_wealth'].mean()
            wealth_diff = mean_wealth - rental_wealth
            
            traj_scenario = trajectories_df[trajectories_df['scenario_label'] == scenario]
            avg_cost = traj_scenario[traj_scenario['year'] > 0]['monthly_net_housing_cost'].mean()
            cost_diff = avg_cost - rental_cost
            
            wealth_sign = "+" if wealth_diff >= 0 else ""
            cost_sign = "+" if cost_diff >= 0 else ""
            
            lines.append(f"| {scenario} | {format_currency(mean_wealth)} | {wealth_sign}{format_currency(wealth_diff)} | {format_currency(avg_cost)} | {cost_sign}{format_currency(cost_diff)} |")
        
        lines.append("")
        lines.append("*Note: Positive wealth difference means buying outperforms renting. Positive cost difference means buying costs more per month.*")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Expected Outcomes
    lines.append("### Expected Outcomes (Mean)")
    lines.append("")
    lines.append("| Scenario | Final Wealth | Total Interest | Avg Net Housing Cost |")
    lines.append("|----------|--------------|----------------|----------------------|")
    
    for scenario in scenarios:
        scenario_data = summary_df[summary_df['scenario_label'] == scenario]
        mean_wealth = scenario_data['final_wealth'].mean()
        
        # Calculate average net housing cost from trajectories (excluding Year 0)
        traj_scenario = trajectories_df[trajectories_df['scenario_label'] == scenario]
        avg_cost = traj_scenario[traj_scenario['year'] > 0]['monthly_net_housing_cost'].mean()
        
        # Total interest only for buying scenarios
        if scenario == rental_baseline:
            lines.append(f"| {scenario} | {format_currency(mean_wealth)} | N/A | {format_currency(avg_cost)} |")
        else:
            mean_interest = scenario_data['total_interest_paid'].mean()
            lines.append(f"| {scenario} | {format_currency(mean_wealth)} | {format_currency(mean_interest)} | {format_currency(avg_cost)} |")
    
    lines.append("")
    
    # Find best scenario (excluding rental baseline)
    if buying_scenarios:
        buying_summary = summary_df[summary_df['scenario_label'].isin(buying_scenarios)]
        best_scenario = buying_summary.groupby('scenario_label')['final_wealth'].mean().idxmax()
        best_wealth = buying_summary.groupby('scenario_label')['final_wealth'].mean().max()
        lines.append(f"**Winner (Expected)**: {best_scenario} ({format_currency(best_wealth)})")
        
        if rental_baseline:
            rental_wealth = summary_df[summary_df['scenario_label'] == rental_baseline]['final_wealth'].mean()
            if best_wealth > rental_wealth:
                lines.append(f"   - Outperforms rental baseline by {format_currency(best_wealth - rental_wealth)}")
            else:
                lines.append(f"   - Underperforms rental baseline by {format_currency(rental_wealth - best_wealth)}")
    else:
        best_scenario = summary_df.groupby('scenario_label')['final_wealth'].mean().idxmax()
        best_wealth = summary_df.groupby('scenario_label')['final_wealth'].mean().max()
        lines.append(f"**Winner (Expected)**: {best_scenario} ({format_currency(best_wealth)})")
    
    lines.append("")
    
    # Risk Analysis
    lines.append("### Risk Analysis (10th Percentile)")
    lines.append("")
    lines.append("| Scenario | P10 Wealth | P10 Net Housing Cost |")
    lines.append("|----------|------------|----------------------|")
    
    for scenario in scenarios:
        scenario_data = summary_df[summary_df['scenario_label'] == scenario]
        p10_wealth = scenario_data['final_wealth'].quantile(0.1)
        
        # For rental baseline, all samples are identical
        if scenario == rental_baseline:
            traj_scenario = trajectories_df[trajectories_df['scenario_label'] == scenario]
            p10_cost = traj_scenario[traj_scenario['year'] > 0]['monthly_net_housing_cost'].mean()
        else:
            # Find sample_ids in the P10 wealth percentile
            p10_threshold = scenario_data['final_wealth'].quantile(0.1)
            p10_samples = scenario_data[scenario_data['final_wealth'] <= p10_threshold]['sample_id']
            
            # Calculate average net housing cost for those specific samples (excluding Year 0)
            traj_scenario = trajectories_df[
                (trajectories_df['scenario_label'] == scenario) &
                (trajectories_df['sample_id'].isin(p10_samples)) &
                (trajectories_df['year'] > 0)
            ]
            p10_cost = traj_scenario['monthly_net_housing_cost'].mean()
        
        lines.append(f"| {scenario} | {format_currency(p10_wealth)} | {format_currency(p10_cost)} |")
    
    lines.append("")
    
    # Find best downside scenario (excluding rental baseline for buying comparison)
    if buying_scenarios:
        buying_summary = summary_df[summary_df['scenario_label'].isin(buying_scenarios)]
        best_downside = buying_summary.groupby('scenario_label')['final_wealth'].quantile(0.1).idxmax()
        lines.append(f"**Winner (Downside)**: {best_downside} (best worst-case outcome among buying scenarios)")
    else:
        best_downside = summary_df.groupby('scenario_label')['final_wealth'].quantile(0.1).idxmax()
        lines.append(f"**Winner (Downside)**: {best_downside} (best worst-case outcome)")
    lines.append("")
    
    # Upside Potential
    lines.append("### Upside Potential (90th Percentile)")
    lines.append("")
    lines.append("| Scenario | P90 Wealth | P90 Net Housing Cost |")
    lines.append("|----------|------------|----------------------|")
    
    for scenario in scenarios:
        scenario_data = summary_df[summary_df['scenario_label'] == scenario]
        p90_wealth = scenario_data['final_wealth'].quantile(0.9)
        
        # For rental baseline, all samples are identical
        if scenario == rental_baseline:
            traj_scenario = trajectories_df[trajectories_df['scenario_label'] == scenario]
            p90_cost = traj_scenario[traj_scenario['year'] > 0]['monthly_net_housing_cost'].mean()
        else:
            # Find sample_ids in the P90 wealth percentile
            p90_threshold = scenario_data['final_wealth'].quantile(0.9)
            p90_samples = scenario_data[scenario_data['final_wealth'] >= p90_threshold]['sample_id']
            
            # Calculate average net housing cost for those specific samples (excluding Year 0)
            traj_scenario = trajectories_df[
                (trajectories_df['scenario_label'] == scenario) &
                (trajectories_df['sample_id'].isin(p90_samples)) &
                (trajectories_df['year'] > 0)
            ]
            p90_cost = traj_scenario['monthly_net_housing_cost'].mean()
        
        lines.append(f"| {scenario} | {format_currency(p90_wealth)} | {format_currency(p90_cost)} |")
    
    lines.append("")
    
    # Comparative Probabilities
    lines.append("## Comparative Probabilities")
    lines.append("")
    lines.append("### P(Scenario A > Scenario B)")
    lines.append("")
    lines.append("Probability that row scenario outperforms column scenario:")
    lines.append("")
    
    # Build probability table
    header = "| |" + " | ".join(scenarios) + " |"
    separator = "|" + "|".join(["---"] * (len(scenarios) + 1)) + "|"
    lines.append(header)
    lines.append(separator)
    
    for i, scenario_i in enumerate(scenarios):
        row = f"| {scenario_i} |"
        for scenario_j in scenarios:
            prob = prob_matrix.loc[scenario_i, scenario_j]
            row += f" {format_percentage(prob)} |"
        lines.append(row)
    
    lines.append("")
    
    # Dominance Analysis
    lines.append("### Dominance Analysis")
    lines.append("")
    lines.append("Frequency each scenario is best across all metrics:")
    lines.append("")
    lines.append("| Scenario | Dominance Count | Dominance Rate |")
    lines.append("|----------|-----------------|----------------|")
    
    for _, row in dominance_stats.iterrows():
        lines.append(f"| {row['scenario_label']} | {row['dominance_count']} | {format_percentage(row['dominance_rate'])} |")
    
    lines.append("")
    
    # Regime-Conditional Performance
    lines.append("## Regime-Conditional Performance")
    lines.append("")
    
    # Crisis-heavy scenarios
    crisis_heavy = regime_perf[regime_perf['regime_bin'] == '8-30']
    if not crisis_heavy.empty:
        lines.append("### Performance in Crisis-Heavy Scenarios (≥8 years in crisis)")
        lines.append("")
        lines.append("| Scenario | Mean Wealth | P10 Wealth | P90 Wealth |")
        lines.append("|----------|-------------|------------|------------|")
        
        for _, row in crisis_heavy.iterrows():
            lines.append(f"| {row['scenario_label']} | {format_currency(row['mean'])} | {format_currency(row['p10'])} | {format_currency(row['p90'])} |")
        
        lines.append("")
    
    # Crisis-light scenarios
    crisis_light = regime_perf[regime_perf['regime_bin'] == '0-2']
    if not crisis_light.empty:
        lines.append("### Performance in Crisis-Light Scenarios (≤2 years in crisis)")
        lines.append("")
        lines.append("| Scenario | Mean Wealth | P10 Wealth | P90 Wealth |")
        lines.append("|----------|-------------|------------|------------|")
        
        for _, row in crisis_light.iterrows():
            lines.append(f"| {row['scenario_label']} | {format_currency(row['mean'])} | {format_currency(row['p10'])} | {format_currency(row['p90'])} |")
        
        lines.append("")
    
    # Exit Timing Analysis (only if data is available)
    if not exit_timing.empty:
        lines.append("## Exit Timing Analysis")
        lines.append("")
        
        # Get unique years from exit_timing data
        available_years = sorted(exit_timing['year'].unique())
        
        for year in available_years:
            year_data = exit_timing[exit_timing['year'] == year]
            if not year_data.empty:
                lines.append(f"### Wealth at Year {year}")
                lines.append("")
                lines.append("| Scenario | Mean | P10 | P90 | Prob(Negative) |")
                lines.append("|----------|------|-----|-----|----------------|")
                
                for _, row in year_data.iterrows():
                    lines.append(f"| {row['scenario_label']} | {format_currency(row['mean_wealth'])} | {format_currency(row['p10_wealth'])} | {format_currency(row['p90_wealth'])} | {format_percentage(row['prob_negative'])} |")
                
                lines.append("")
    
    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    
    # Determine overall winner
    mean_wealth_by_scenario = summary_df.groupby('scenario_label')['final_wealth'].mean()
    best_overall = mean_wealth_by_scenario.idxmax()
    
    lines.append(f"1. **Best Overall**: {best_overall}")
    lines.append(f"   - Highest expected wealth: {format_currency(mean_wealth_by_scenario[best_overall])}")
    
    # Check if it also has best downside
    p10_by_scenario = summary_df.groupby('scenario_label')['final_wealth'].quantile(0.1)
    if p10_by_scenario.idxmax() == best_overall:
        lines.append(f"   - Best downside protection")
    
    lines.append("")
    
    # Risk considerations
    lines.append("2. **Risk Considerations**:")
    
    # Calculate probability of negative wealth at year 10
    if 'wealth_year_10' in summary_df.columns:
        prob_negative_10 = (summary_df['wealth_year_10'] < 0).mean()
        lines.append(f"   - {format_percentage(prob_negative_10)} probability of negative wealth at year 10")
    
    # Calculate crisis scenario frequency
    crisis_scenarios = (summary_df['years_in_crisis'] >= 8).mean()
    lines.append(f"   - Crisis-heavy scenarios (8+ years) occur in {format_percentage(crisis_scenarios)} of simulations")
    
    lines.append("")
    
    lines.append("3. **Sensitivity**:")
    
    # Determine matrix type description
    if matrix_type == 'calibrated':
        matrix_desc = "calibrated transition matrix (BIS data + smoothing with expert judgment)"
    elif matrix_type == 'empirical':
        matrix_desc = "empirical transition matrix (no prior smoothing)"
    elif matrix_type == 'expert':
        matrix_desc = "prior transition matrix (expert judgment)"
    elif matrix_type:
        matrix_desc = f"{matrix_type} transition matrix"
    else:
        matrix_desc = "transition matrix (type not specified)"
    
    lines.append(f"   - Results based on {matrix_desc}")
    lines.append("   - See alternative matrix configurations in scenarios/markov/ for sensitivity analysis")
    lines.append("")
    
    # Write report
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
