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


def test_full_30y_mortgage_with_two_10y_resets():
    """
    Test a full 30-year mortgage with 10-year fixed periods.
    The loan should reset twice: at year 10 and year 20.
    Verify that rate changes persist through each 10-year period.
    """
    # Create scenario with 10-year fixed period and two resets
    scenario = BuyingScenario(
        purchase_price=300000,
        mortgage_principal=270000,
        mortgage_loans=[
            MortgageLoan(
                percentage=1.0,
                annual_rate=0.03,  # 3% initial rate
                term_years=30,
                amortization_type="annuity",
                fixed_period_years=10,
                rate_resets=[
                    RateReset(
                        month=120,  # After 10 years (month 120)
                        new_rate_min=0.02,
                        new_rate_max=0.08,
                        new_rate_default=0.04
                    ),
                    RateReset(
                        month=240,  # After 20 years (month 240)
                        new_rate_min=0.02,
                        new_rate_max=0.08,
                        new_rate_default=0.05
                    )
                ],
                label="10-year fixed"
            )
        ],
        living_months=360,  # 30 years
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
    
    # Create regime path: 30 years + initial year 0
    n_years = 31
    regime_path = np.zeros(n_years, dtype=int)  # All normal growth
    
    # Property growth: constant 2%
    property_growth_rates = np.full(n_years, 0.02)
    
    # Mortgage rates: simulate market rate changes
    # Years 0-10: 3%
    # Years 11-20: 4% (first reset)
    # Years 21-30: 5% (second reset)
    mortgage_rates = np.full(n_years, 0.03)
    mortgage_rates[11:21] = 0.04  # Year 11-20 (after first reset at month 120)
    mortgage_rates[21:] = 0.05    # Year 21-30 (after second reset at month 240)
    
    # Simulate
    simulator = TimeSteppingSimulator(
        scenario=scenario,
        regime_path=regime_path,
        property_growth_rates=property_growth_rates,
        mortgage_rates=mortgage_rates,
        scenario_label="10-year fixed with 2 resets"
    )
    
    states = simulator.simulate_single_path()
    
    # Extract mortgage rates for each year
    actual_rates = [state.mortgage_rate for state in states]
    
    # Assertions:
    # 1. Years 1-10: Rate should be 3% (initial fixed period)
    for year in range(1, 11):
        assert abs(actual_rates[year] - 0.03) < 0.0001, \
            f"Year {year}: Rate should be 3% (initial period). Got {actual_rates[year]:.4f}"
    
    # 2. Years 11-20: Rate should be 4% (after first reset at month 120)
    for year in range(11, 21):
        assert abs(actual_rates[year] - 0.04) < 0.0001, \
            f"Year {year}: Rate should be 4% (after first reset). Got {actual_rates[year]:.4f}"
    
    # 3. Years 21-30: Rate should be 5% (after second reset at month 240)
    for year in range(21, 31):
        assert abs(actual_rates[year] - 0.05) < 0.0001, \
            f"Year {year}: Rate should be 5% (after second reset). Got {actual_rates[year]:.4f}"
    
    # 4. Verify monthly payments change at reset points
    # Get average payment for each period
    payment_period1 = np.mean([states[y].monthly_payment for y in range(1, 11)])
    payment_period2 = np.mean([states[y].monthly_payment for y in range(11, 21)])
    payment_period3 = np.mean([states[y].monthly_payment for y in range(21, 31)])
    
    # Payment should increase after first reset (rate goes from 3% to 4%)
    assert payment_period2 > payment_period1, \
        f"Payment should increase after first reset. " \
        f"Period 1 (3%): €{payment_period1:.2f}, Period 2 (4%): €{payment_period2:.2f}"
    
    # Payment should increase again after second reset (rate goes from 4% to 5%)
    assert payment_period3 > payment_period2, \
        f"Payment should increase after second reset. " \
        f"Period 2 (4%): €{payment_period2:.2f}, Period 3 (5%): €{payment_period3:.2f}"
    
    # 5. Verify that the loan is fully paid off by year 30
    final_balance = states[30].mortgage_balance
    assert final_balance < 100, \
        f"Mortgage should be nearly paid off after 30 years. Remaining balance: €{final_balance:.2f}"
        
    # Due to declining balance, absolute interest decreases, but we can verify
    # that the rate changes are reflected by checking the interest/balance ratio
    if states[5].mortgage_balance > 1000:
        rate_year5 = states[5].annual_interest_paid / states[5].mortgage_balance
        assert abs(rate_year5 - 0.03) < 0.01, \
            f"Year 5 effective rate should be ~3%. Got {rate_year5:.4f}"
    
    if states[15].mortgage_balance > 1000:
        rate_year15 = states[15].annual_interest_paid / states[15].mortgage_balance
        assert abs(rate_year15 - 0.04) < 0.01, \
            f"Year 15 effective rate should be ~4%. Got {rate_year15:.4f}"
    
    if states[25].mortgage_balance > 1000:
        rate_year25 = states[25].annual_interest_paid / states[25].mortgage_balance
        assert abs(rate_year25 - 0.05) < 0.01, \
            f"Year 25 effective rate should be ~5%. Got {rate_year25:.4f}"
    
    # 7. Verify equity progression is reasonable
    # Equity should increase over time as mortgage is paid down and property appreciates
    assert states[30].equity > states[10].equity, \
        f"Equity should increase over 30 years. Year 10: €{states[10].equity:.2f}, Year 30: €{states[30].equity:.2f}"
    
    # 8. Verify that the mortgage balance decreases over time
    assert states[30].mortgage_balance < states[10].mortgage_balance, \
        f"Mortgage balance should decrease. Year 10: €{states[10].mortgage_balance:.2f}, Year 30: €{states[30].mortgage_balance:.2f}"


def test_rate_reset_timing_and_persistence():
    """
    Explicitly test that rate resets occur at the correct months and persist.
    
    This test verifies:
    1. Resets trigger at exactly month 120 (year 10) and month 240 (year 20)
    2. The new rate is clamped between new_rate_min and new_rate_max
    3. The rate persists throughout each 10-year period
    4. The loan tracks through all 30 years with 2 resets
    """
    # Create scenario with explicit reset configuration
    scenario = BuyingScenario(
        purchase_price=300000,
        mortgage_principal=270000,
        mortgage_loans=[
            MortgageLoan(
                percentage=1.0,
                annual_rate=0.03,  # 3% initial rate
                term_years=30,
                amortization_type="annuity",
                fixed_period_years=10,
                rate_resets=[
                    RateReset(
                        month=120,  # After 10 years (month 120)
                        new_rate_min=0.02,
                        new_rate_max=0.08,
                        new_rate_default=0.04
                    ),
                    RateReset(
                        month=240,  # After 20 years (month 240)
                        new_rate_min=0.02,
                        new_rate_max=0.08,
                        new_rate_default=0.05
                    )
                ],
                label="10-year fixed"
            )
        ],
        living_months=360,  # 30 years
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
    
    # Create regime path: 30 years + initial year 0
    n_years = 31
    regime_path = np.zeros(n_years, dtype=int)
    property_growth_rates = np.full(n_years, 0.02)
    
    # Market rates that will be clamped by reset bounds
    # Year 11: market at 4.5% -> clamped to [0.02, 0.08] = 4.5%
    # Year 21: market at 6.0% -> clamped to [0.02, 0.08] = 6.0%
    mortgage_rates = np.full(n_years, 0.03)
    mortgage_rates[11:21] = 0.045  # Market rate for first reset period
    mortgage_rates[21:] = 0.06     # Market rate for second reset period
    
    simulator = TimeSteppingSimulator(
        scenario=scenario,
        regime_path=regime_path,
        property_growth_rates=property_growth_rates,
        mortgage_rates=mortgage_rates,
        scenario_label="10-year fixed with 2 resets"
    )
    
    states = simulator.simulate_single_path()
    
    # Test 1: Initial period (years 1-10) should maintain 3% rate
    for year in range(1, 11):
        assert abs(states[year].mortgage_rate - 0.03) < 0.0001, \
            f"Year {year}: Expected 3% initial rate, got {states[year].mortgage_rate:.4f}"
    
    # Test 2: First reset period (years 11-20) should use market rate 4.5%
    # Reset at month 120 means it affects year 11 onwards
    for year in range(11, 21):
        assert abs(states[year].mortgage_rate - 0.045) < 0.0001, \
            f"Year {year}: Expected 4.5% after first reset (month 120), got {states[year].mortgage_rate:.4f}"
    
    # Test 3: Second reset period (years 21-30) should use market rate 6.0%
    # Reset at month 240 means it affects year 21 onwards
    for year in range(21, 31):
        assert abs(states[year].mortgage_rate - 0.06) < 0.0001, \
            f"Year {year}: Expected 6.0% after second reset (month 240), got {states[year].mortgage_rate:.4f}"
    
    # Test 4: Verify that exactly 2 rate changes occurred
    rate_changes = []
    prev_rate = states[1].mortgage_rate
    for year in range(2, 31):
        curr_rate = states[year].mortgage_rate
        if abs(curr_rate - prev_rate) > 0.0001:
            rate_changes.append((year, prev_rate, curr_rate))
        prev_rate = curr_rate
    
    assert len(rate_changes) == 2, \
        f"Expected exactly 2 rate changes, got {len(rate_changes)}: {rate_changes}"
    
    # Verify the rate changes happened at the right years
    assert rate_changes[0][0] == 11, \
        f"First rate change should be at year 11 (month 120), got year {rate_changes[0][0]}"
    assert rate_changes[1][0] == 21, \
        f"Second rate change should be at year 21 (month 240), got year {rate_changes[1][0]}"
    
    # Test 5: Verify rate changes are reflected in monthly payments
    payment_year10 = states[10].monthly_payment
    payment_year11 = states[11].monthly_payment
    payment_year20 = states[20].monthly_payment
    payment_year21 = states[21].monthly_payment
    
    assert payment_year11 > payment_year10, \
        f"Payment should increase after first reset. Year 10: €{payment_year10:.2f}, Year 11: €{payment_year11:.2f}"
    
    assert payment_year21 > payment_year20, \
        f"Payment should increase after second reset. Year 20: €{payment_year20:.2f}, Year 21: €{payment_year21:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
