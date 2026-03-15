"""
Rental Time-Stepping Simulator

This module simulates rental scenarios year-by-year with rent increases,
providing a baseline for comparison with buying scenarios.
"""

from dataclasses import dataclass
from typing import List
import pandas as pd

from src.estate import RentingScenario


@dataclass
class RentalTimeStepState:
    """
    State of the rental simulation at a specific year.
    
    Attributes:
        year: Current year (0 = initial)
        monthly_payment: Current monthly housing cost (rent + utilities)
        monthly_net_housing_cost: Same as monthly_payment for consistency
        cumulative_cashflow: Sum of all cash spent (negative)
        wealth: Same as cumulative_cashflow (no equity)
        equity: Always 0 (no property ownership)
        mortgage_balance: Always 0 (no mortgage)
        annual_rent: Rent paid this year
        annual_utilities: Utilities paid this year
        current_rent_monthly: Current monthly rent (after increases)
    """
    year: int
    monthly_payment: float
    monthly_net_housing_cost: float
    cumulative_cashflow: float
    wealth: float
    equity: float
    mortgage_balance: float
    annual_rent: float
    annual_utilities: float
    current_rent_monthly: float


class RentalTimeSteppingSimulator:
    """
    Year-by-year rental simulator with rent increases.
    """
    
    def __init__(
        self,
        scenario: RentingScenario,
        scenario_label: str = "Rental Baseline"
    ):
        """
        Initialize rental simulator.
        
        Args:
            scenario: Rental scenario configuration
            scenario_label: Human-readable label for this scenario
        """
        self.scenario = scenario
        self.scenario_label = scenario_label
        self.horizon_years = scenario.living_months // 12
    
    def simulate_single_path(self) -> List[RentalTimeStepState]:
        """
        Simulate a single rental trajectory year-by-year.
        
        Returns:
            List of RentalTimeStepState objects, one per year (including year 0)
        """
        states = []
        
        # Year 0: Initial state (no costs yet)
        state0 = RentalTimeStepState(
            year=0,
            monthly_payment=self.scenario.rent_monthly + self.scenario.utilities_monthly,
            monthly_net_housing_cost=self.scenario.rent_monthly + self.scenario.utilities_monthly,
            cumulative_cashflow=0.0,
            wealth=0.0,
            equity=0.0,
            mortgage_balance=0.0,
            annual_rent=0.0,
            annual_utilities=0.0,
            current_rent_monthly=self.scenario.rent_monthly
        )
        states.append(state0)
        
        # Simulate year by year
        cumulative_cashflow = 0.0
        current_rent_monthly = self.scenario.rent_monthly
        
        for year in range(1, self.horizon_years + 1):
            # Apply rent increase at the start of each year
            if year > 1 and self.scenario.annual_rent_increase > 0:
                current_rent_monthly = current_rent_monthly * (1 + self.scenario.annual_rent_increase)
            
            # Calculate annual costs
            annual_rent = current_rent_monthly * 12
            annual_utilities = self.scenario.utilities_monthly * 12
            annual_total = annual_rent + annual_utilities
            
            # Update cumulative cashflow (negative = spent)
            cumulative_cashflow -= annual_total
            
            # Monthly payment (average for the year)
            monthly_payment = annual_total / 12
            
            state = RentalTimeStepState(
                year=year,
                monthly_payment=monthly_payment,
                monthly_net_housing_cost=monthly_payment,
                cumulative_cashflow=cumulative_cashflow,
                wealth=cumulative_cashflow,  # No equity, so wealth = cashflow
                equity=0.0,
                mortgage_balance=0.0,
                annual_rent=annual_rent,
                annual_utilities=annual_utilities,
                current_rent_monthly=current_rent_monthly
            )
            states.append(state)
        
        return states


def simulate_rental_baseline(
    scenario: RentingScenario,
    n_samples: int,
    scenario_label: str = "Rental Baseline"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate rental baseline and replicate across all samples.
    
    Since rental baseline is deterministic (not affected by regimes),
    we calculate it once and replicate for all samples.
    
    Args:
        scenario: Rental scenario configuration
        n_samples: Number of samples to replicate
        scenario_label: Label for the scenario
    
    Returns:
        Tuple of (trajectories_df, summary_df)
    """
    # Simulate once
    simulator = RentalTimeSteppingSimulator(scenario, scenario_label)
    states = simulator.simulate_single_path()
    
    # Convert to trajectory records
    trajectory_records = []
    for sample_id in range(n_samples):
        for state in states:
            trajectory_records.append({
                'sample_id': sample_id,
                'scenario_label': scenario_label,
                'year': state.year,
                'wealth': state.wealth,
                'equity': state.equity,
                'cumulative_cashflow': state.cumulative_cashflow,
                'mortgage_balance': state.mortgage_balance,
                'monthly_payment': state.monthly_payment,
                'monthly_net_housing_cost': state.monthly_net_housing_cost,
                'regime': -1,  # Not applicable for rental
                'property_value': 0.0,
                'annual_interest_paid': 0.0,
                'annual_principal_paid': 0.0,
                'property_growth_rate': 0.0,
                'mortgage_rate': 0.0,
                'current_rent_monthly': state.current_rent_monthly,
                'annual_rent': state.annual_rent,
                'annual_utilities': state.annual_utilities
            })
    
    trajectories_df = pd.DataFrame(trajectory_records)
    
    # Create summary records (one per sample)
    summary_records = []
    final_state = states[-1]
    
    for sample_id in range(n_samples):
        summary_records.append({
            'sample_id': sample_id,
            'scenario_label': scenario_label,
            'final_wealth': final_state.wealth,
            'final_equity': 0.0,
            'final_mortgage_balance': 0.0,
            'total_interest_paid': 0.0,
            'total_principal_paid': 0.0,
            'total_cash_outflow': -final_state.cumulative_cashflow,  # Make positive
            'roi': -1.0,  # Spent everything, got nothing back
            'cash_efficiency': 0.0,  # No equity built
            'max_drawdown': 0.0,  # Not applicable
            'years_in_crisis': 0,  # Not applicable
            'years_in_boom': 0,  # Not applicable
            'years_in_normal': 0,  # Not applicable
            'years_in_recovery': 0  # Not applicable
        })
        
        # Add wealth_year_X columns for exit timing analysis
        for state in states:
            if state.year > 0:
                summary_records[-1][f'wealth_year_{state.year}'] = state.wealth
    
    summary_df = pd.DataFrame(summary_records)
    
    return trajectories_df, summary_df
