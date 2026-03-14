"""
Time-Stepping Mortgage Simulator with Markov Regime Transitions

This module simulates mortgage scenarios year-by-year with dynamic regime
transitions, allowing for realistic modeling of economic cycles and their
impact on property values and refinancing rates.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from copy import deepcopy

from src.estate import BuyingScenario, evaluate_buying
from src.markov_regime import MarkovChainConfig, REGIME_LABELS
from src.mortgage import MortgageLoan


@dataclass
class TimeStepState:
    """
    State of the simulation at a specific year.
    
    Attributes:
        year: Current year (0 = initial)
        regime: Current regime state (0-3)
        property_value: Current property value
        mortgage_balance: Remaining mortgage debt
        equity: property_value - mortgage_balance
        cumulative_cashflow: Sum of all cash in/out (negative = spent)
        wealth: equity + cumulative_cashflow
        annual_interest_paid: Interest paid this year
        annual_principal_paid: Principal paid this year
        monthly_payment: Current monthly payment
        property_growth_rate: Realized growth rate this year
        mortgage_rate: Current mortgage rate
    """
    year: int
    regime: int
    property_value: float
    mortgage_balance: float
    equity: float
    cumulative_cashflow: float
    wealth: float
    annual_interest_paid: float
    annual_principal_paid: float
    monthly_payment: float
    property_growth_rate: float
    mortgage_rate: float


class TimeSteppingSimulator:
    """
    Year-by-year mortgage simulator with Markov regime transitions.
    """
    
    def __init__(
        self,
        scenario: BuyingScenario,
        regime_path: np.ndarray,
        property_growth_rates: np.ndarray,
        mortgage_rates: np.ndarray,
        scenario_label: Optional[str] = None
    ):
        """
        Initialize simulator with fixed regime path and parameters.
        
        Args:
            scenario: Base mortgage scenario
            regime_path: Array of regime states for each year
            property_growth_rates: Array of property growth rates for each year
            mortgage_rates: Array of mortgage rates for each year
            scenario_label: Human-readable label for this scenario
        """
        self.base_scenario = scenario
        self.regime_path = regime_path
        self.property_growth_rates = property_growth_rates
        self.mortgage_rates = mortgage_rates
        self.scenario_label = scenario_label or "Unnamed"
        self.horizon_years = len(regime_path) - 1  # Exclude year 0
    
    def simulate_single_path(self) -> List[TimeStepState]:
        """
        Simulate a single trajectory year-by-year.
        
        Returns:
            List of TimeStepState objects, one per year (including year 0)
        """
        states = []
        
        # Year 0: Initial state
        scenario_year0 = deepcopy(self.base_scenario)
        scenario_year0.living_months = 0
        scenario_year0.sold_at_end = False
        
        result_year0 = evaluate_buying(scenario_year0)
        
        # Initial values
        property_value = self.base_scenario.purchase_price
        total_mortgage = sum(loan.principal for loan in self._get_loans())
        mortgage_balance = total_mortgage
        equity = property_value - mortgage_balance
        
        # Initial cash outflow (down payment + one-off costs)
        down_payment = property_value - total_mortgage
        initial_cashflow = -(down_payment + self.base_scenario.one_off_costs + 
                            self.base_scenario.renovation_costs_once)
        
        state0 = TimeStepState(
            year=0,
            regime=int(self.regime_path[0]),
            property_value=property_value,
            mortgage_balance=mortgage_balance,
            equity=equity,
            cumulative_cashflow=initial_cashflow,
            wealth=equity + initial_cashflow,
            annual_interest_paid=0.0,
            annual_principal_paid=0.0,
            monthly_payment=0.0,
            property_growth_rate=0.0,
            mortgage_rate=self.mortgage_rates[0]
        )
        states.append(state0)
        
        # Simulate year by year
        cumulative_cashflow = initial_cashflow
        
        for year in range(1, self.horizon_years + 1):
            # Get regime and parameters for this year
            regime = int(self.regime_path[year])
            growth_rate = self.property_growth_rates[year]
            mortgage_rate = self.mortgage_rates[year]
            
            # Update property value
            property_value = property_value * (1 + growth_rate)
            
            # Create scenario for this year (12 months)
            scenario_year = deepcopy(self.base_scenario)
            scenario_year.living_months = 12
            scenario_year.annual_value_growth = growth_rate
            scenario_year.sold_at_end = False
            
            # CRITICAL FIX: Update loan principals AND remaining term
            # This ensures we're continuing the existing mortgage schedule, not starting fresh
            months_elapsed = (year - 1) * 12
            
            if scenario_year.mortgage_loans is not None:
                # Update each loan's principal proportionally based on current balance
                original_total = sum(loan.principal for loan in self._get_loans())
                for loan in scenario_year.mortgage_loans:
                    if loan.principal is not None:
                        # Scale down the principal proportionally
                        loan.principal = loan.principal * (mortgage_balance / original_total)
                    elif loan.percentage is not None:
                        # For percentage-based loans, calculate from current balance
                        loan.principal = loan.percentage * mortgage_balance
                        loan.percentage = None  # Clear percentage to avoid re-resolution
                    
                    # CRITICAL: Reduce the term to reflect elapsed time
                    # If original term was 30 years and we're in year 2, remaining term is 29 years
                    original_term_months = loan.term_years * 12
                    remaining_term_months = original_term_months - months_elapsed
                    loan.term_years = remaining_term_months / 12
            
            # Also update the scenario's mortgage_principal field
            scenario_year.mortgage_principal = mortgage_balance
            
            # Update mortgage rates if refinancing occurs at the start of this year
            # Convert year to month (year 1 = month 12, year 2 = month 24, etc.)
            current_month = year * 12
            scenario_year = self._apply_mortgage_rates(scenario_year, current_month, mortgage_rate)
            
            # Evaluate this year
            result = evaluate_buying(scenario_year)
            
            # Extract annual cashflows
            annual_interest = result.get('mortgage_interest_paid', 0.0)
            annual_principal = result.get('mortgage_principal_paid', 0.0)
            annual_vve_utilities = 12 * (self.base_scenario.monthly_vve + 
                                         self.base_scenario.monthly_utilities)
            annual_rent_income = min(12, self.base_scenario.months_rented) * \
                                self.base_scenario.monthly_rent_income
            
            # Tax effects (simplified - would need more detailed calculation)
            annual_tax_effect = 0.0
            
            # Net cashflow this year (negative = outflow)
            annual_cashflow = (annual_rent_income + annual_tax_effect - 
                             annual_interest - annual_principal - annual_vve_utilities)
            
            cumulative_cashflow += annual_cashflow
            
            # Update mortgage balance
            mortgage_balance = mortgage_balance - annual_principal
            
            # Calculate equity and wealth
            equity = property_value - mortgage_balance
            wealth = equity + cumulative_cashflow
            
            # Monthly payment (average for the year)
            monthly_payment = (annual_interest + annual_principal) / 12
            
            state = TimeStepState(
                year=year,
                regime=regime,
                property_value=property_value,
                mortgage_balance=mortgage_balance,
                equity=equity,
                cumulative_cashflow=cumulative_cashflow,
                wealth=wealth,
                annual_interest_paid=annual_interest,
                annual_principal_paid=annual_principal,
                monthly_payment=monthly_payment,
                property_growth_rate=growth_rate,
                mortgage_rate=mortgage_rate
            )
            states.append(state)
        
        return states
    
    def _get_loans(self) -> List[MortgageLoan]:
        """Get mortgage loans from scenario."""
        from src.estate import _get_mortgage_config
        return _get_mortgage_config(self.base_scenario)
    
    def _apply_mortgage_rates(
        self,
        scenario: BuyingScenario,
        current_month: int,
        new_rate: float
    ) -> BuyingScenario:
        """
        Apply mortgage rate for refinancing events that occur in the current period.
        
        This method only updates rates when an actual rate reset occurs.
        During fixed-rate periods, the original loan rates are preserved.
        
        Args:
            scenario: Scenario to update
            current_month: Current month in the simulation (0-indexed)
            new_rate: New rate from Markov regime (used for rate resets)
        
        Returns:
            Updated scenario (modified in place)
        """
        # For legacy format, only update if we're past any initial fixed period
        if scenario.mortgage_annual_rate is not None:
            # Legacy format doesn't have rate resets, so keep original rate
            pass
        
        # For multi-loan format, check if any rate resets occur this month
        if scenario.mortgage_loans is not None:
            for loan in scenario.mortgage_loans:
                if loan.rate_resets is not None:
                    for reset in loan.rate_resets:
                        # Check if reset occurs exactly at this month
                        if reset.month == current_month:
                            # Apply the new rate from Markov regime
                            reset.new_rate_default = new_rate
        
        return scenario


def simulate_scenario_with_fixed_paths(
    scenario: BuyingScenario,
    regime_paths: np.ndarray,
    parameter_draws: Dict[str, np.ndarray],
    scenario_label: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate a scenario with pre-determined regime paths and parameters.
    
    Args:
        scenario: Mortgage scenario to simulate
        regime_paths: Array of shape (n_samples, n_steps)
        parameter_draws: Dict with 'property_growth' and 'mortgage_rates' arrays
        scenario_label: Label for this scenario
    
    Returns:
        Tuple of (trajectories_df, summary_df)
    """
    n_samples = regime_paths.shape[0]
    
    all_trajectories = []
    all_summaries = []
    
    for sample_id in range(n_samples):
        # Create simulator for this sample
        simulator = TimeSteppingSimulator(
            scenario=scenario,
            regime_path=regime_paths[sample_id],
            property_growth_rates=parameter_draws['property_growth'][sample_id],
            mortgage_rates=parameter_draws['mortgage_rates'][sample_id],
            scenario_label=scenario_label
        )
        
        # Simulate
        states = simulator.simulate_single_path()
        
        # Convert to trajectory records
        for state in states:
            all_trajectories.append({
                'sample_id': sample_id,
                'scenario_label': scenario_label,
                'year': state.year,
                'regime': state.regime,
                'regime_label': REGIME_LABELS[state.regime],
                'property_value': state.property_value,
                'property_growth_rate': state.property_growth_rate,
                'mortgage_balance': state.mortgage_balance,
                'mortgage_rate': state.mortgage_rate,
                'monthly_payment': state.monthly_payment,
                'equity': state.equity,
                'cumulative_cashflow': state.cumulative_cashflow,
                'wealth': state.wealth,
                'annual_interest_paid': state.annual_interest_paid,
                'annual_principal_paid': state.annual_principal_paid
            })
        
        # Create summary record
        final_state = states[-1]
        
        # Calculate regime statistics
        regime_counts = np.bincount(regime_paths[sample_id], minlength=4)
        
        # Calculate wealth at key years
        wealth_at_years = {}
        for state in states:
            if state.year in [5, 10, 15, 20, 25, 30]:
                wealth_at_years[f'wealth_year_{state.year}'] = state.wealth
        
        # Calculate risk metrics
        wealth_trajectory = [s.wealth for s in states]
        min_wealth = min(wealth_trajectory)
        min_wealth_year = wealth_trajectory.index(min_wealth)
        
        # Calculate max drawdown
        peak = wealth_trajectory[0]
        max_drawdown = 0.0
        for w in wealth_trajectory:
            if w > peak:
                peak = w
            drawdown = peak - w
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Time underwater (negative wealth)
        time_underwater = sum(1 for w in wealth_trajectory if w < 0)
        
        # Regime path as string
        regime_path_str = ','.join(str(int(r)) for r in regime_paths[sample_id])
        
        summary = {
            'sample_id': sample_id,
            'scenario_label': scenario_label,
            'final_wealth': final_state.wealth,
            'final_property_value': final_state.property_value,
            'final_equity': final_state.equity,
            'total_interest_paid': sum(s.annual_interest_paid for s in states),
            'total_principal_paid': sum(s.annual_principal_paid for s in states),
            'years_in_normal': int(regime_counts[0]),
            'years_in_stagnation': int(regime_counts[1]),
            'years_in_crisis': int(regime_counts[2]),
            'years_in_recovery': int(regime_counts[3]),
            'min_wealth': min_wealth,
            'min_wealth_year': min_wealth_year,
            'max_drawdown': max_drawdown,
            'time_underwater': time_underwater,
            'regime_path': regime_path_str
        }
        
        # Add wealth at key years
        summary.update(wealth_at_years)
        
        all_summaries.append(summary)
    
    trajectories_df = pd.DataFrame(all_trajectories)
    summary_df = pd.DataFrame(all_summaries)
    
    return trajectories_df, summary_df
