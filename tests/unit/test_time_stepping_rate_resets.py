"""
Test that time-stepping simulator properly handles interest rate resets.

This test verifies that loans with different fixed periods (5y vs 10y) produce
different monthly payments when interest rates change.
"""

import pytest
import numpy as np
from src.estate import BuyingScenario
from src.mortgage import MortgageLoan, RateReset
from src.time_stepping_simulator import TimeSteppingSimulator


def test_rate_reset_affects_monthly_payments():
    """
    Test that a 5-year fixed period loan has different payments than a 10-year
    fixed period loan when interest rates change after year 5.
    """
    # Create a simple scenario with one loan
    base_scenario = BuyingScenario(
        purchase_price=300000,
        mortgage_principal=270000,
        mortgage_loans=[
            MortgageLoan(
                percentage=1.0,
                annual_rate=0.03,  # 3% initial rate
                term_years=30,
                amortization_type="annuity",
                fixed_period_years=5,
                rate_resets=[
                    RateReset(
                        month=60,  # After 5 years
                        new_rate_min=0.04,
                        new_rate_max=0.06,
                        new_rate_default=0.05  # Will be overridden by Markov rate
                    )
                ],
                label="5-year fixed"
            )
        ],
        living_months=120,  # 10 years
        one_off_costs=10000,
        renovation_costs_once=0,
        monthly_vve=200,
        monthly_utilities=150,
        monthly_rent_income=0,
        months_rented=0,
        annual_value_growth=0.02,
        sold_at_end=True,
        selling_cost_rate=0.02
    )
    
    # Create regime path: 10 years + initial year 0
    n_years = 11
    regime_path = np.zeros(n_years, dtype=int)  # All normal growth
    
    # Property growth: constant 2%
    property_growth_rates = np.full(n_years, 0.02)
    
    # Mortgage rates: 3% for first 5 years, then jump to 5%
    mortgage_rates = np.full(n_years, 0.03)
    mortgage_rates[6:] = 0.05  # Year 6 onwards (after 5-year fixed period)
    
    # Simulate with 5-year fixed period
    simulator_5y = TimeSteppingSimulator(
        scenario=base_scenario,
        regime_path=regime_path,
        property_growth_rates=property_growth_rates,
        mortgage_rates=mortgage_rates,
        scenario_label="5-year fixed"
    )
    
    states_5y = simulator_5y.simulate_single_path()
    
    # Now create a 10-year fixed period version
    scenario_10y = BuyingScenario(
        purchase_price=300000,
        mortgage_principal=270000,
        mortgage_loans=[
            MortgageLoan(
                percentage=1.0,
                annual_rate=0.03,
                term_years=30,
                amortization_type="annuity",
                fixed_period_years=10,
                rate_resets=[
                    RateReset(
                        month=120,  # After 10 years
                        new_rate_min=0.04,
                        new_rate_max=0.06,
                        new_rate_default=0.05
                    )
                ],
                label="10-year fixed"
            )
        ],
        living_months=120,
        one_off_costs=10000,
        renovation_costs_once=0,
        monthly_vve=200,
        monthly_utilities=150,
        monthly_rent_income=0,
        months_rented=0,
        annual_value_growth=0.02,
        sold_at_end=True,
        selling_cost_rate=0.02
    )
    
    simulator_10y = TimeSteppingSimulator(
        scenario=scenario_10y,
        regime_path=regime_path,
        property_growth_rates=property_growth_rates,
        mortgage_rates=mortgage_rates,
        scenario_label="10-year fixed"
    )
    
    states_10y = simulator_10y.simulate_single_path()
    
    # Assertions:
    # 1. Years 1-5: Both should have similar monthly payments (both at 3%)
    for year in range(1, 6):
        payment_5y = states_5y[year].monthly_payment
        payment_10y = states_10y[year].monthly_payment
        # Should be very close (within 1%)
        assert abs(payment_5y - payment_10y) / payment_5y < 0.01, \
            f"Year {year}: Payments should be similar before any reset. " \
            f"5y: €{payment_5y:.2f}, 10y: €{payment_10y:.2f}"
    
    # 2. Years 6-10: 5-year fixed should have HIGHER payments (rate increased to 5%)
    #    while 10-year fixed stays at 3%
    for year in range(6, 11):
        payment_5y = states_5y[year].monthly_payment
        payment_10y = states_10y[year].monthly_payment
        
        # 5-year fixed should have higher payment after reset
        assert payment_5y > payment_10y, \
            f"Year {year}: 5-year fixed should have higher payment after rate reset. " \
            f"5y: €{payment_5y:.2f}, 10y: €{payment_10y:.2f}"
        
        # The difference should be noticeable (at least 0.5% higher)
        # Note: As balance decreases, the absolute difference decreases too
        assert (payment_5y - payment_10y) / payment_10y > 0.005, \
            f"Year {year}: Payment difference should be noticeable (>0.5%). " \
            f"5y: €{payment_5y:.2f}, 10y: €{payment_10y:.2f}, " \
            f"diff: {(payment_5y - payment_10y) / payment_10y * 100:.1f}%"
    
    # 3. Total interest paid should be higher for 5-year fixed
    total_interest_5y = sum(s.annual_interest_paid for s in states_5y)
    total_interest_10y = sum(s.annual_interest_paid for s in states_10y)
    
    assert total_interest_5y > total_interest_10y, \
        f"5-year fixed should pay more total interest due to rate increase. " \
        f"5y: €{total_interest_5y:.2f}, 10y: €{total_interest_10y:.2f}"


def test_multiple_loans_with_different_reset_schedules():
    """
    Test scenario with 2 loans that reset at different times.
    """
    scenario = BuyingScenario(
        purchase_price=300000,
        mortgage_principal=270000,
        mortgage_loans=[
            MortgageLoan(
                percentage=0.7,
                annual_rate=0.03,
                term_years=30,
                amortization_type="annuity",
                fixed_period_years=10,
                rate_resets=[
                    RateReset(month=120, new_rate_min=0.04, new_rate_max=0.06, new_rate_default=0.05)
                ],
                label="Loan A - 70% Annuity 10y"
            ),
            MortgageLoan(
                percentage=0.3,
                annual_rate=0.03,
                term_years=30,
                amortization_type="linear",
                fixed_period_years=5,
                rate_resets=[
                    RateReset(month=60, new_rate_min=0.04, new_rate_max=0.06, new_rate_default=0.05)
                ],
                label="Loan B - 30% Linear 5y"
            )
        ],
        living_months=120,
        one_off_costs=10000,
        renovation_costs_once=0,
        monthly_vve=200,
        monthly_utilities=150,
        monthly_rent_income=0,
        months_rented=0,
        annual_value_growth=0.02,
        sold_at_end=True,
        selling_cost_rate=0.02
    )
    
    n_years = 11
    regime_path = np.zeros(n_years, dtype=int)
    property_growth_rates = np.full(n_years, 0.02)
    
    # Rates jump at year 6
    mortgage_rates = np.full(n_years, 0.03)
    mortgage_rates[6:] = 0.05
    
    simulator = TimeSteppingSimulator(
        scenario=scenario,
        regime_path=regime_path,
        property_growth_rates=property_growth_rates,
        mortgage_rates=mortgage_rates,
        scenario_label="Mixed loans"
    )
    
    states = simulator.simulate_single_path()
    
    # Year 5: Both loans at 3%
    payment_year5 = states[5].monthly_payment
    
    # Year 6: Loan B (30%) resets to 5%, Loan A (70%) stays at 3%
    payment_year6 = states[6].monthly_payment
    
    # Payment should increase in year 6 (but not by full 2% since only 30% of loan resets)
    assert payment_year6 > payment_year5, \
        f"Payment should increase when Loan B resets. Year 5: €{payment_year5:.2f}, Year 6: €{payment_year6:.2f}"
    
    # The increase should be modest (less than 10% since only 30% of loan resets)
    increase_pct = (payment_year6 - payment_year5) / payment_year5
    assert 0.01 < increase_pct < 0.15, \
        f"Payment increase should be modest (1-15%) when only 30% of loan resets. Got {increase_pct*100:.1f}%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
