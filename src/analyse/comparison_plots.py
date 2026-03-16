"""
Comparison Plots Module

This module provides visualization functions for comparing multiple
mortgage scenarios, including wealth distributions, trajectories,
regime-conditional performance, and exit timing analysis.

Glossary:
    Maximum Drawdown: The largest peak-to-trough decline in wealth during
    the simulation period. It measures the worst loss from a previous high
    point, indicating the maximum financial stress experienced. For example,
    if wealth peaks at €100k and later drops to €60k before recovering,
    the maximum drawdown is €40k. This metric helps assess downside risk
    and liquidity needs during adverse market conditions.
"""

from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch


def plot_wealth_distributions(
    summary_df: pd.DataFrame,
    metric: str = 'final_wealth',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot wealth distribution comparison as histograms/densities.
    
    Args:
        summary_df: Summary DataFrame
        metric: Metric to plot (default: 'final_wealth')
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = sorted(summary_df['scenario_label'].unique())
    
    # Separate rental baseline from buying scenarios
    rental_baseline = None
    buying_scenarios = []
    for scenario in scenarios:
        if 'rental' in scenario.lower() and 'baseline' in scenario.lower():
            rental_baseline = scenario
        else:
            buying_scenarios.append(scenario)
    
    colors = [plt.cm.tab10(i) for i in range(len(buying_scenarios))]
    
    # Plot buying scenarios
    for i, scenario in enumerate(buying_scenarios):
        data = summary_df[summary_df['scenario_label'] == scenario][metric]
        ax.hist(data, bins=50, alpha=0.5, label=scenario, color=colors[i], density=True)
    
    # Plot rental baseline as vertical line (all samples identical)
    if rental_baseline:
        data = summary_df[summary_df['scenario_label'] == rental_baseline][metric]
        rental_value = data.iloc[0]  # All values are identical
        ax.axvline(
            rental_value,
            color='#404040',
            linestyle='--',
            linewidth=2.5,
            label=rental_baseline,
            alpha=1.0,
            zorder=10
        )
    
    ax.set_xlabel(f'{metric.replace("_", " ").title()} (€)')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution of {metric.replace("_", " ").title()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_wealth_trajectories(
    trajectories_df: pd.DataFrame,
    n_samples: int = 100,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot sample wealth trajectories over time.
    
    Args:
        trajectories_df: Trajectory DataFrame
        n_samples: Number of sample paths to plot per scenario
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    scenarios = sorted(trajectories_df['scenario_label'].unique())
    
    # Separate rental baseline from other scenarios
    rental_baseline_label = None
    buying_scenarios = []
    for scenario in scenarios:
        if 'rental' in scenario.lower() and 'baseline' in scenario.lower():
            rental_baseline_label = scenario
        else:
            buying_scenarios.append(scenario)
    
    # Assign colors (rental baseline gets dark gray)
    colors = [plt.cm.tab10(i) for i in range(len(buying_scenarios))]
    
    # Sample random paths
    sample_ids = np.random.choice(
        trajectories_df['sample_id'].unique(),
        size=min(n_samples, trajectories_df['sample_id'].nunique()),
        replace=False
    )
    
    # Plot buying scenarios with transparent sample paths
    for i, scenario in enumerate(buying_scenarios):
        scenario_data = trajectories_df[
            (trajectories_df['scenario_label'] == scenario) &
            (trajectories_df['sample_id'].isin(sample_ids))
        ]
        
        for sample_id in sample_ids:
            sample_data = scenario_data[scenario_data['sample_id'] == sample_id]
            ax.plot(
                sample_data['year'],
                sample_data['wealth'],
                color=colors[i],
                alpha=0.3,
                linewidth=0.5
            )
    
    # Plot rental baseline as a single bold dashed line
    if rental_baseline_label is not None:
        rental_data = trajectories_df[
            (trajectories_df['scenario_label'] == rental_baseline_label) &
            (trajectories_df['sample_id'] == 0)  # All samples are identical
        ]
        ax.plot(
            rental_data['year'],
            rental_data['wealth'],
            color='#404040',  # Dark gray
            linestyle='--',
            linewidth=2.5,
            alpha=1.0,
            zorder=10,
            label=rental_baseline_label
        )
    
    # Add legend with scenario colors
    legend_patches = [
        mpatches.Patch(color=colors[i], label=scenario)
        for i, scenario in enumerate(buying_scenarios)
    ]
    if rental_baseline_label is not None:
        legend_patches.append(
            mpatches.Patch(color='#404040', label=rental_baseline_label)
        )
    ax.legend(handles=legend_patches)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Wealth (€)')
    ax.set_title(f'Wealth Trajectories ({n_samples} samples per scenario)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    return ax


def plot_regime_conditional_performance(
    summary_df: pd.DataFrame,
    regime_column: str = 'years_in_crisis',
    metric: str = 'final_wealth',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot performance conditional on regime exposure as box plots.
    
    Args:
        summary_df: Summary DataFrame
        regime_column: Column to condition on
        metric: Metric to plot
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create regime bins
    bins = [0, 2, 5, 8, 30]
    summary_df = summary_df.copy()
    summary_df['regime_bin'] = pd.cut(
        summary_df[regime_column],
        bins=bins,
        labels=[f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)],
        include_lowest=True
    )
    
    scenarios = sorted(summary_df['scenario_label'].unique())
    regime_bins = sorted(summary_df['regime_bin'].dropna().unique())
    
    # Prepare data for box plot
    data_to_plot = []
    labels = []
    positions = []
    colors_list = []
    
    colors = [plt.cm.tab10(i) for i in range(len(scenarios))]
    
    pos = 0
    for bin_label in regime_bins:
        for i, scenario in enumerate(scenarios):
            data = summary_df[
                (summary_df['scenario_label'] == scenario) &
                (summary_df['regime_bin'] == bin_label)
            ][metric]
            
            if len(data) > 0:
                data_to_plot.append(data)
                labels.append(f'{scenario}\n{bin_label}')
                positions.append(pos)
                colors_list.append(colors[i])
                pos += 1
        pos += 0.5  # Gap between regime bins
    
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(f'{metric.replace("_", " ").title()} (€)')
    ax.set_title(f'Performance by {regime_column.replace("_", " ").title()}')
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_property_value_by_crisis(
    summary_df: pd.DataFrame,
    regime_column: str = 'years_in_crisis',
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot final property value conditional on years in crisis as box plots.
    
    Shows how apartment valuations at the end of the observation period
    vary based on the number of years spent in crisis regime. Since all
    scenarios share the same regime paths and property growth rates, the
    property value is the same across scenarios - this plot shows one
    boxplot per crisis exposure bin.
    
    Args:
        summary_df: Summary DataFrame
        regime_column: Column to condition on (default: 'years_in_crisis')
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create regime bins
    bins = [0, 2, 5, 8, 30]
    summary_df = summary_df.copy()
    summary_df['regime_bin'] = pd.cut(
        summary_df[regime_column],
        bins=bins,
        labels=[f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)],
        include_lowest=True
    )
    
    # Filter out rental baseline (no property ownership)
    buying_scenarios = []
    for scenario in summary_df['scenario_label'].unique():
        if not ('rental' in scenario.lower() and 'baseline' in scenario.lower()):
            buying_scenarios.append(scenario)
    
    summary_df = summary_df[summary_df['scenario_label'].isin(buying_scenarios)]
    
    if summary_df.empty:
        ax.text(0.5, 0.5, 'No buying scenarios to plot',
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Since property value is the same across all scenarios (shared regime paths),
    # we only need to take unique samples by sample_id to avoid duplicates
    summary_df_unique = summary_df.drop_duplicates(subset=['sample_id', 'regime_bin'])
    
    regime_bins = sorted(summary_df_unique['regime_bin'].dropna().unique())
    
    # Prepare data for box plot - one box per regime bin
    data_to_plot = []
    labels = []
    
    for bin_label in regime_bins:
        data = summary_df_unique[
            summary_df_unique['regime_bin'] == bin_label
        ]['final_property_value']
        
        if len(data) > 0:
            data_to_plot.append(data)
            labels.append(f'{bin_label} years\nin crisis')
    
    if not data_to_plot:
        ax.text(0.5, 0.5, 'No data to plot',
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    # Create boxplot with single color
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
    
    # Color all boxes the same
    for patch in bp['boxes']:
        patch.set_facecolor(plt.cm.tab10(0))
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Final Property Value (€)')
    ax.set_xlabel('Years in Crisis Regime')
    ax.set_title('Final Apartment Valuation by Years in Crisis')
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax


def plot_exit_timing_analysis(
    summary_df: pd.DataFrame,
    years: Optional[List[int]] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot mean wealth over time for exit timing analysis.
    
    Args:
        summary_df: Summary DataFrame with wealth_year_X columns
        years: Years to plot (default: [5, 10, 15, 20, 25, 30])
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    if years is None:
        years = [5, 10, 15, 20, 25, 30]
    
    scenarios = sorted(summary_df['scenario_label'].unique())
    
    # Separate rental baseline from buying scenarios
    rental_baseline = None
    buying_scenarios = []
    for scenario in scenarios:
        if 'rental' in scenario.lower() and 'baseline' in scenario.lower():
            rental_baseline = scenario
        else:
            buying_scenarios.append(scenario)
    
    colors = [plt.cm.tab10(i) for i in range(len(buying_scenarios))]
    
    # Plot buying scenarios
    for i, scenario in enumerate(buying_scenarios):
        scenario_data = summary_df[summary_df['scenario_label'] == scenario]
        
        mean_wealth = []
        p10_wealth = []
        p90_wealth = []
        available_years = []
        
        for year in years:
            col_name = f'wealth_year_{year}'
            if col_name in scenario_data.columns:
                wealth = scenario_data[col_name]
                mean_wealth.append(wealth.mean())
                p10_wealth.append(wealth.quantile(0.1))
                p90_wealth.append(wealth.quantile(0.9))
                available_years.append(year)
        
        if available_years:
            ax.plot(available_years, mean_wealth, marker='o', label=scenario, color=colors[i], linewidth=2)
            ax.fill_between(
                available_years,
                p10_wealth,
                p90_wealth,
                alpha=0.2,
                color=colors[i]
            )
    
    # Plot rental baseline (if present)
    if rental_baseline:
        scenario_data = summary_df[summary_df['scenario_label'] == rental_baseline]
        
        mean_wealth = []
        available_years = []
        
        for year in years:
            col_name = f'wealth_year_{year}'
            if col_name in scenario_data.columns:
                wealth = scenario_data[col_name]
                mean_wealth.append(wealth.mean())
                available_years.append(year)
        
        if available_years:
            ax.plot(
                available_years,
                mean_wealth,
                marker='s',
                label=rental_baseline,
                color='#404040',
                linestyle='--',
                linewidth=2.5,
                alpha=1.0,
                zorder=10
            )
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Wealth (€)')
    ax.set_title('Mean Wealth Over Time (with P10-P90 range)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    return ax


def plot_probability_heatmap(
    prob_matrix: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    metric_label: Optional[str] = None
) -> plt.Axes:
    """
    Plot probability matrix as heatmap.
    
    Args:
        prob_matrix: Probability matrix from calculate_comparative_probabilities
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(prob_matrix.values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(prob_matrix.columns)))
    ax.set_yticks(np.arange(len(prob_matrix.index)))
    ax.set_xticklabels(prob_matrix.columns, rotation=45, ha='right')
    ax.set_yticklabels(prob_matrix.index)
    
    # Add text annotations
    for i in range(len(prob_matrix.index)):
        for j in range(len(prob_matrix.columns)):
            _ = ax.text(j, i, f'{prob_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    title_str = 'P(Row Scenario > Column Scenario)'
    title_str += f'\nfor {metric_label}' if metric_label else ''
    ax.set_title(title_str)
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Scenario')
    
    plt.colorbar(im, ax=ax, label='Probability')
    
    return ax


def plot_drawdown_analysis(
    summary_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot maximum drawdown distribution.
    
    Args:
        summary_df: Summary DataFrame with max_drawdown column
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = sorted(summary_df['scenario_label'].unique())
    
    # Separate rental baseline from buying scenarios
    rental_baseline = None
    buying_scenarios = []
    for scenario in scenarios:
        if 'rental' in scenario.lower() and 'baseline' in scenario.lower():
            rental_baseline = scenario
        else:
            buying_scenarios.append(scenario)
    
    colors = [plt.cm.tab10(i) for i in range(len(buying_scenarios))]
    
    # Plot buying scenarios
    for i, scenario in enumerate(buying_scenarios):
        data = summary_df[summary_df['scenario_label'] == scenario]['max_drawdown']
        ax.hist(data, bins=50, alpha=0.5, label=scenario, color=colors[i], density=True)
    
    # Plot rental baseline as vertical line (all samples identical)
    if rental_baseline:
        data = summary_df[summary_df['scenario_label'] == rental_baseline]['max_drawdown']
        rental_value = data.iloc[0]  # All values are identical
        ax.axvline(
            rental_value,
            color='#404040',
            linestyle='--',
            linewidth=2.5,
            label=rental_baseline,
            alpha=1.0,
            zorder=10
        )
    
    ax.set_xlabel('Maximum Drawdown (€)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Maximum Drawdown')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_wealth_trajectory_boxplots(
    trajectories_df: pd.DataFrame,
    years: Optional[List[int]] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot boxplots of wealth at different time points (one per 5 years).
    
    Args:
        trajectories_df: Trajectory DataFrame
        years: Years to plot (default: auto-detect from data, or [5, 10, 15, 20, 25, 30])
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    
    # Auto-detect available years if not specified
    if years is None:
        available_years = sorted(trajectories_df['year'].unique())
        common_checkpoints = [5, 10, 15, 20, 25, 30]
        years = [y for y in common_checkpoints if y in available_years]
        # If no common checkpoints match, use all available years
        if not years:
            years = available_years
    
    scenarios = sorted(trajectories_df['scenario_label'].unique())
    
    # Separate rental baseline from buying scenarios
    rental_baseline = None
    buying_scenarios = []
    for scenario in scenarios:
        if 'rental' in scenario.lower() and 'baseline' in scenario.lower():
            rental_baseline = scenario
        else:
            buying_scenarios.append(scenario)
    
    colors = [plt.cm.tab10(i) for i in range(len(buying_scenarios))]
    
    # Prepare data for box plot
    data_to_plot = []
    labels = []
    positions = []
    colors_list = []
    
    # Store rental baseline values for horizontal lines
    rental_values = {}
    if rental_baseline:
        for year in years:
            year_data = trajectories_df[
                (trajectories_df['scenario_label'] == rental_baseline) &
                (trajectories_df['year'] == year)
            ]
            if len(year_data) > 0:
                rental_values[year] = year_data['wealth'].iloc[0]
    
    pos = 0
    year_positions = {}  # Track position range for each year
    for year in years:
        year_data = trajectories_df[trajectories_df['year'] == year]
        if len(year_data) == 0:
            continue
        
        year_start_pos = pos
        for i, scenario in enumerate(buying_scenarios):
            scenario_data = year_data[year_data['scenario_label'] == scenario]['wealth']
            
            if len(scenario_data) > 0:
                data_to_plot.append(scenario_data)
                labels.append(f'{scenario}\nY{year}')
                positions.append(pos)
                colors_list.append(colors[i])
                pos += 1
        
        year_end_pos = pos - 1
        year_positions[year] = (year_start_pos, year_end_pos)
        pos += 0.5  # Gap between years
    
    # Only create boxplot if we have data
    if not data_to_plot:
        ax.text(0.5, 0.5, 'No data available for selected years',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Wealth Distribution Over Time (Boxplots per 5 Years)')
        return ax
    
    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add rental baseline as horizontal dashed lines
    if rental_baseline and rental_values:
        for year, (start_pos, end_pos) in year_positions.items():
            if year in rental_values:
                ax.hlines(
                    rental_values[year],
                    start_pos - 0.3,
                    end_pos + 0.3,
                    colors='#404040',
                    linestyles='--',
                    linewidth=2.5,
                    alpha=1.0,
                    zorder=10,
                    label=rental_baseline if year == years[0] else None
                )
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Wealth (€)')
    ax.set_title('Wealth Distribution Over Time (Boxplots per 5 Years)')
    if rental_baseline:
        ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    return ax


def plot_monthly_payment_distributions(
    trajectories_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot distribution of net monthly housing costs for all scenarios.
    
    Args:
        trajectories_df: Trajectory DataFrame
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = sorted(trajectories_df['scenario_label'].unique())
    
    # Separate rental baseline from buying scenarios
    rental_baseline = None
    buying_scenarios = []
    for scenario in scenarios:
        if 'rental' in scenario.lower() and 'baseline' in scenario.lower():
            rental_baseline = scenario
        else:
            buying_scenarios.append(scenario)
    
    colors = [plt.cm.tab10(i) for i in range(len(buying_scenarios))]
    
    # Plot buying scenarios
    for i, scenario in enumerate(buying_scenarios):
        data = trajectories_df[trajectories_df['scenario_label'] == scenario]['monthly_net_housing_cost']
        ax.hist(data, bins=50, alpha=0.5, label=scenario, color=colors[i], density=True)
    
    # Plot rental baseline (if present) - show distribution over time
    if rental_baseline:
        data = trajectories_df[trajectories_df['scenario_label'] == rental_baseline]['monthly_net_housing_cost']
        # Rental payments change over time due to rent increases, so plot the distribution
        ax.hist(
            data,
            bins=50,
            alpha=0.3,
            label=rental_baseline,
            color='#D3D3D3',  # Light gray
            density=True,
            zorder=1  # Place in background
        )
    
    ax.set_xlabel('Net Monthly Housing Cost (€)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Net Monthly Housing Costs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_2d_wealth_vs_payment_scatter(
    summary_df: pd.DataFrame,
    trajectories_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot 2D scatter of final wealth vs average net monthly housing cost with P10-P90 ranges.
    
    Shows mean as a star marker with rectangles showing P10-P90 ranges for each metric.
    
    Args:
        summary_df: Summary DataFrame with final_wealth
        trajectories_df: Trajectory DataFrame with monthly_net_housing_cost
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    scenarios = sorted(summary_df['scenario_label'].unique())
    
    # Separate rental baseline from buying scenarios
    rental_baseline = None
    buying_scenarios = []
    for scenario in scenarios:
        if 'rental' in scenario.lower() and 'baseline' in scenario.lower():
            rental_baseline = scenario
        else:
            buying_scenarios.append(scenario)
    
    colors = [plt.cm.tab10(i) for i in range(len(buying_scenarios))]
    
    # Plot buying scenarios
    for i, scenario in enumerate(buying_scenarios):
        # Get final wealth for each sample
        scenario_summary = summary_df[summary_df['scenario_label'] == scenario]
        
        # Calculate average net housing cost per sample across all years (excluding Year 0)
        scenario_traj = trajectories_df[trajectories_df['scenario_label'] == scenario]
        avg_costs = scenario_traj[scenario_traj['year'] > 0].groupby('sample_id')['monthly_net_housing_cost'].mean()
        
        # Merge to get both metrics per sample
        plot_data = scenario_summary.set_index('sample_id')[['final_wealth']].join(
            avg_costs.rename('avg_monthly_cost')
        )
        
        # Calculate statistics
        cost_mean = plot_data['avg_monthly_cost'].mean()
        cost_p10 = plot_data['avg_monthly_cost'].quantile(0.1)
        cost_p90 = plot_data['avg_monthly_cost'].quantile(0.9)
        
        wealth_mean = plot_data['final_wealth'].mean()
        wealth_p10 = plot_data['final_wealth'].quantile(0.1)
        wealth_p90 = plot_data['final_wealth'].quantile(0.9)
        
        # Draw rectangle showing P10-P90 range
        rect_width = cost_p90 - cost_p10
        rect_height = wealth_p90 - wealth_p10
        rect = Rectangle(
            (cost_p10, wealth_p10),
            rect_width,
            rect_height,
            linewidth=2,
            edgecolor=colors[i],
            facecolor=colors[i],
            alpha=0.2,
            label=None
        )
        ax.add_patch(rect)
        
        # Plot as scatter with very low transparency
        ax.scatter(
            plot_data['avg_monthly_cost'],
            plot_data['final_wealth'],
            alpha=0.05,
            s=50,
            color=colors[i],
            label=None,
            edgecolors='none',
            linewidth=0
        )
        
        # Plot mean as star
        ax.scatter(
            cost_mean,
            wealth_mean,
            marker='*',
            s=400,
            color=colors[i],
            edgecolors='black',
            linewidths=1.5,
            zorder=10,
            label=scenario
        )
        
        # Add text label near the mean
        ax.annotate(
            scenario,
            (cost_mean, wealth_mean),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3)
        )
    
    # Plot rental baseline (if present) as single point
    if rental_baseline:
        scenario_summary = summary_df[summary_df['scenario_label'] == rental_baseline]
        scenario_traj = trajectories_df[trajectories_df['scenario_label'] == rental_baseline]
        
        # All samples are identical for rental, calculate average excluding Year 0
        rental_cost = scenario_traj[scenario_traj['year'] > 0]['monthly_net_housing_cost'].mean()
        rental_wealth = scenario_summary['final_wealth'].mean()
        
        ax.scatter(
            rental_cost,
            rental_wealth,
            marker='s',
            s=200,
            color='#404040',
            edgecolors='black',
            linewidths=2,
            zorder=11,
            label=rental_baseline
        )
        
        ax.annotate(
            rental_baseline,
            (rental_cost, rental_wealth),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#404040', alpha=0.3)
        )
    
    ax.set_xlabel('Average Net Monthly Housing Cost (€)', fontsize=12)
    ax.set_ylabel('Final Wealth (€)', fontsize=12)
    ax.set_title('Final Wealth vs Net Housing Cost (★ = mean, ■ = rental, shaded = P10-P90 range)', fontsize=14)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    return ax


def plot_monthly_payment_over_time(
    trajectories_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot mean net monthly housing cost over time with P10-P90 range.
    
    Args:
        trajectories_df: Trajectory DataFrame with year and monthly_net_housing_cost columns
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = sorted(trajectories_df['scenario_label'].unique())
    
    # Separate rental baseline from buying scenarios
    rental_baseline = None
    buying_scenarios = []
    for scenario in scenarios:
        if 'rental' in scenario.lower() and 'baseline' in scenario.lower():
            rental_baseline = scenario
        else:
            buying_scenarios.append(scenario)
    
    colors = [plt.cm.tab10(i) for i in range(len(buying_scenarios))]
    
    # Plot buying scenarios
    for i, scenario in enumerate(buying_scenarios):
        scenario_data = trajectories_df[trajectories_df['scenario_label'] == scenario]
        
        # Group by year and calculate statistics
        yearly_stats = scenario_data.groupby('year')['monthly_net_housing_cost'].agg([
            ('mean', 'mean'),
            ('p10', lambda x: x.quantile(0.1)),
            ('p90', lambda x: x.quantile(0.9))
        ]).reset_index()
        
        # Plot mean line
        ax.plot(
            yearly_stats['year'],
            yearly_stats['mean'],
            marker='o',
            label=scenario,
            color=colors[i],
            linewidth=2
        )
        
        # Fill between P10 and P90
        ax.fill_between(
            yearly_stats['year'],
            yearly_stats['p10'],
            yearly_stats['p90'],
            alpha=0.2,
            color=colors[i]
        )
    
    # Plot rental baseline (if present)
    if rental_baseline:
        scenario_data = trajectories_df[trajectories_df['scenario_label'] == rental_baseline]
        
        # Group by year and calculate mean (all samples identical)
        yearly_stats = scenario_data.groupby('year')['monthly_net_housing_cost'].mean().reset_index()
        yearly_stats.columns = ['year', 'mean']
        
        # Plot rental baseline
        ax.plot(
            yearly_stats['year'],
            yearly_stats['mean'],
            marker='s',
            label=rental_baseline,
            color='#404040',
            linestyle='--',
            linewidth=2.5,
            alpha=1.0,
            zorder=10
        )
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Net Monthly Housing Cost (€)')
    ax.set_title('Mean Net Monthly Housing Cost Over Time (with P10-P90 range)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_mortgage_payment_over_time(
    trajectories_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot mean monthly mortgage payment (principal + interest only) over time with P10-P90 range.
    Only includes buying scenarios (rental baseline excluded).
    
    Args:
        trajectories_df: Trajectory DataFrame with year and monthly_payment columns
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = sorted(trajectories_df['scenario_label'].unique())
    
    # Filter to buying scenarios only (exclude rental baseline)
    buying_scenarios = []
    for scenario in scenarios:
        if not ('rental' in scenario.lower() and 'baseline' in scenario.lower()):
            buying_scenarios.append(scenario)
    
    if not buying_scenarios:
        ax.text(0.5, 0.5, 'No buying scenarios to plot',
                ha='center', va='center', transform=ax.transAxes)
        return ax
    
    colors = [plt.cm.tab10(i) for i in range(len(buying_scenarios))]
    
    # Plot buying scenarios
    for i, scenario in enumerate(buying_scenarios):
        scenario_data = trajectories_df[trajectories_df['scenario_label'] == scenario]
        
        # Group by year and calculate statistics
        yearly_stats = scenario_data.groupby('year')['monthly_payment'].agg([
            ('mean', 'mean'),
            ('p10', lambda x: x.quantile(0.1)),
            ('p90', lambda x: x.quantile(0.9))
        ]).reset_index()
        
        # Plot mean line
        ax.plot(
            yearly_stats['year'],
            yearly_stats['mean'],
            marker='o',
            label=scenario,
            color=colors[i],
            linewidth=2
        )
        
        # Fill between P10 and P90
        ax.fill_between(
            yearly_stats['year'],
            yearly_stats['p10'],
            yearly_stats['p90'],
            alpha=0.2,
            color=colors[i]
        )
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Monthly Mortgage Payment (€)')
    ax.set_title('Mean Monthly Mortgage Payment Over Time (Principal + Interest, P10-P90 range)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_roi_and_cash_efficiency(
    summary_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot ROI and Cash Efficiency distributions as box plots.
    
    Args:
        summary_df: Summary DataFrame with roi and cash_efficiency columns
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    scenarios = sorted(summary_df['scenario_label'].unique())
    
    # Prepare data for grouped box plots
    roi_data = []
    ce_data = []
    labels = []
    positions_roi = []
    positions_ce = []
    
    colors = [plt.cm.tab10(i) for i in range(len(scenarios))]
    
    pos = 0
    for i, scenario in enumerate(scenarios):
        scenario_data = summary_df[summary_df['scenario_label'] == scenario]
        
        roi_data.append(scenario_data['roi'])
        ce_data.append(scenario_data['cash_efficiency'])
        labels.append(scenario)
        
        positions_roi.append(pos)
        positions_ce.append(pos + 0.4)
        pos += 1
    
    # Plot ROI boxes
    bp_roi = ax.boxplot(roi_data, positions=positions_roi, widths=0.35,
                         patch_artist=True, showfliers=False)
    for patch, color in zip(bp_roi['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Plot Cash Efficiency boxes
    bp_ce = ax.boxplot(ce_data, positions=positions_ce, widths=0.35,
                       patch_artist=True, showfliers=False)
    for patch, color in zip(bp_ce['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
        patch.set_hatch('//')
    
    # Set labels and formatting
    ax.set_xticks([p + 0.2 for p in positions_roi])
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Ratio')
    ax.set_title('ROI and Cash Efficiency Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    # Add legend
    legend_elements = [
        Patch(facecolor='gray', alpha=0.7, label='ROI (Wealth / Cash Outflow)'),
        Patch(facecolor='gray', alpha=0.4, hatch='//', label='Cash Efficiency (Equity / Cash Outflow)')
    ]
    ax.legend(handles=legend_elements, loc='best')
    
    return ax


def plot_roi_vs_cash_efficiency_2d(
    summary_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot 2D scatter of ROI vs Cash Efficiency with P10-P90 ranges.
    
    Shows mean as a star marker with rectangles showing P10-P90 ranges for each metric.
    
    Args:
        summary_df: Summary DataFrame with roi and cash_efficiency columns
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    scenarios = sorted(summary_df['scenario_label'].unique())
    colors = [plt.cm.tab10(i) for i in range(len(scenarios))]
    
    for i, scenario in enumerate(scenarios):
        scenario_data = summary_df[summary_df['scenario_label'] == scenario]
        
        # Calculate statistics
        roi_mean = scenario_data['roi'].mean()
        roi_p10 = scenario_data['roi'].quantile(0.1)
        roi_p90 = scenario_data['roi'].quantile(0.9)
        
        ce_mean = scenario_data['cash_efficiency'].mean()
        ce_p10 = scenario_data['cash_efficiency'].quantile(0.1)
        ce_p90 = scenario_data['cash_efficiency'].quantile(0.9)
        
        # Draw rectangle showing P10-P90 range
        rect_width = ce_p90 - ce_p10
        rect_height = roi_p90 - roi_p10
        rect = Rectangle(
            (ce_p10, roi_p10),
            rect_width,
            rect_height,
            linewidth=2,
            edgecolor=colors[i],
            facecolor=colors[i],
            alpha=0.2,
            label=None
        )
        ax.add_patch(rect)
        
        # Plot mean as star
        ax.scatter(
            ce_mean,
            roi_mean,
            marker='*',
            s=400,
            color=colors[i],
            edgecolors='black',
            linewidths=1.5,
            zorder=10,
            label=scenario
        )
        
        # Add text label near the mean
        ax.annotate(
            scenario,
            (ce_mean, roi_mean),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3)
        )
    
    ax.set_xlabel('Cash Efficiency (Equity / Cash Outflow)')
    ax.set_ylabel('ROI (Wealth / Cash Outflow)')
    ax.set_title('ROI vs Cash Efficiency (★ = mean, shaded = P10-P90 range)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    return ax


def plot_wealth_vs_cash_outflow(
    summary_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot final wealth vs total cash outflow with P10-P90 ranges.
    
    Shows mean as a star marker with rectangles showing P10-P90 ranges for each metric.
    This shows the relationship between how much you spent and what wealth you achieved.
    
    Args:
        summary_df: Summary DataFrame
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    scenarios = sorted(summary_df['scenario_label'].unique())
    colors = [plt.cm.tab10(i) for i in range(len(scenarios))]
    
    for i, scenario in enumerate(scenarios):
        scenario_data = summary_df[summary_df['scenario_label'] == scenario]
        
        # Calculate statistics
        outflow_mean = scenario_data['total_cash_outflow'].mean()
        outflow_p10 = scenario_data['total_cash_outflow'].quantile(0.1)
        outflow_p90 = scenario_data['total_cash_outflow'].quantile(0.9)
        
        wealth_mean = scenario_data['final_wealth'].mean()
        wealth_p10 = scenario_data['final_wealth'].quantile(0.1)
        wealth_p90 = scenario_data['final_wealth'].quantile(0.9)
        
        # Draw rectangle showing P10-P90 range
        rect_width = outflow_p90 - outflow_p10
        rect_height = wealth_p90 - wealth_p10
        rect = Rectangle(
            (outflow_p10, wealth_p10),
            rect_width,
            rect_height,
            linewidth=2,
            edgecolor=colors[i],
            facecolor=colors[i],
            alpha=0.2,
            label=None
        )
        ax.add_patch(rect)
        
        # Plot as scatter with very low transparency
        ax.scatter(
            scenario_data['total_cash_outflow'],
            scenario_data['final_wealth'],
            alpha=0.05,
            s=30,
            color=colors[i],
            label=None,
            edgecolors='none'
        )
        
        # Plot mean as star
        ax.scatter(
            outflow_mean,
            wealth_mean,
            marker='*',
            s=400,
            color=colors[i],
            edgecolors='black',
            linewidths=1.5,
            zorder=10,
            label=scenario
        )
        
        # Add text label near the mean
        ax.annotate(
            scenario,
            (outflow_mean, wealth_mean),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3)
        )
    
    ax.set_xlabel('Total Cash Outflow (€)')
    ax.set_ylabel('Final Wealth (€)')
    ax.set_title('Final Wealth vs Total Cash Spent (★ = mean, shaded = P10-P90 range)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Add diagonal line showing ROI = -1 (wealth = -outflow)
    xlim = ax.get_xlim()
    ax.plot(xlim, [-xlim[0], -xlim[1]], 'r--', alpha=0.3, linewidth=1, label='ROI = -1')
    
    # Add legend after all plot elements are added
    ax.legend(loc='best')
    
    return ax


def plot_mortgage_performance_over_time(
    trajectories_df: pd.DataFrame,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot how mortgages perform over time (remaining balance).
    
    Args:
        trajectories_df: Trajectory DataFrame
        ax: Matplotlib axes (creates new if None)
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    scenarios = sorted(trajectories_df['scenario_label'].unique())
    colors = [plt.cm.tab10(i) for i in range(len(scenarios))]
    
    for i, scenario in enumerate(scenarios):
        scenario_data = trajectories_df[trajectories_df['scenario_label'] == scenario]
        
        # Group by year and calculate statistics
        yearly_stats = scenario_data.groupby('year')['mortgage_balance'].agg(['mean', 'std', 'quantile'])
        years = yearly_stats.index
        mean_balance = yearly_stats['mean']
        
        # Calculate percentiles
        p10 = scenario_data.groupby('year')['mortgage_balance'].quantile(0.1)
        p90 = scenario_data.groupby('year')['mortgage_balance'].quantile(0.9)
        
        ax.plot(years, mean_balance, marker='o', label=scenario, color=colors[i], linewidth=2)
        ax.fill_between(years, p10, p90, alpha=0.2, color=colors[i])
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Mortgage Balance (€)')
    ax.set_title('Mortgage Balance Over Time (Mean with P10-P90 range)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    return ax


def generate_comparison_plots(
    trajectories_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: str,
    prob_matrices: Optional[Dict[str, pd.DataFrame]] = None
) -> None:
    """
    Generate comprehensive comparison plots and save to PDF.
    
    Args:
        trajectories_df: Trajectory DataFrame
        summary_df: Summary DataFrame
        output_path: Output PDF file path
        prob_matrices: Pre-calculated probability matrices (optional)
    """
    with PdfPages(output_path) as pdf:
        # Plot 1: Wealth Distribution Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_wealth_distributions(summary_df, ax=ax)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Plot 2: Wealth Trajectories
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_wealth_trajectories(trajectories_df, n_samples=100, ax=ax)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Plot 3: Regime-Conditional Performance (Final Wealth)
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_regime_conditional_performance(summary_df, ax=ax)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Plot 4: Property Value by Years in Crisis
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_property_value_by_crisis(summary_df, ax=ax)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Plot 6: Exit Timing Analysis
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_exit_timing_analysis(summary_df, ax=ax)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Plot 7: Probability Heatmap
        if prob_matrices is not None:
            for key, matrix in prob_matrices.items():
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_probability_heatmap(matrix, ax=ax, metric_label=key.replace('_', ' ').title())
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()
        
        # Plot 8: Drawdown Analysis
        if 'max_drawdown' in summary_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_drawdown_analysis(summary_df, ax=ax)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
        
        # Plot 9: Wealth Trajectory Boxplots (one per 5 years)
        fig, ax = plt.subplots(figsize=(14, 6))
        plot_wealth_trajectory_boxplots(trajectories_df, ax=ax)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Plot 10: 2D Wealth vs Payment Scatter
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_2d_wealth_vs_payment_scatter(summary_df, trajectories_df, ax=ax)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Plot 11: Net Housing Cost Over Time
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_monthly_payment_over_time(trajectories_df, ax=ax)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Plot 12: Mortgage Payment Over Time (buying scenarios only)
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_mortgage_payment_over_time(trajectories_df, ax=ax)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Plot 13: Monthly Payment Distributions
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_monthly_payment_distributions(trajectories_df, ax=ax)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Plot 14: ROI and Cash Efficiency (Box plots)
        if 'roi' in summary_df.columns and 'cash_efficiency' in summary_df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_roi_and_cash_efficiency(summary_df, ax=ax)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
        
        # Plot 15: ROI vs Cash Efficiency 2D Scatter
        if 'roi' in summary_df.columns and 'cash_efficiency' in summary_df.columns:
            fig, ax = plt.subplots(figsize=(10, 8))
            plot_roi_vs_cash_efficiency_2d(summary_df, ax=ax)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
        
        # Plot 16: Wealth vs Cash Outflow
        if 'total_cash_outflow' in summary_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_wealth_vs_cash_outflow(summary_df, ax=ax)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
        
        # Plot 17: Mortgage Performance Over Time
        if 'mortgage_balance' in trajectories_df.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            plot_mortgage_performance_over_time(trajectories_df, ax=ax)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
