"""
Test that rate resets persist across years until the next reset.
"""
import pytest
import numpy as np
from src.estate import BuyingScenario
from src.mortgage import MortgageLoan, RateReset
from src.time_stepping_simulator import TimeSteppingSimulator


def test_rate_reset_persists_across_years():
    """
    Test that when a rate reset occurs, the new rate persists for all subsequent years
    until the next reset, not just the year when the reset occurs.
    """
    # Create a scenario with a 5-year fixed rate that resets at month 60
    loan = MortgageLoan(
        principal=300_000,
        annual_rate=0.03,  # Initial 3% rate
        term_years=30,
        rate_resets=[
            RateReset(
                month=60,  # Reset after 5 years (0-indexed, so month 60 = start of year 6)
                new_rate_default=0.05,
                new_rate_min=0.02,
                new_rate_max=0.08
            )
        ]
    )
    
    scenario = BuyingScenario(
        purchase_price=350_000,
        mortgage_loans=[loan],
        living_months=0,
        sold_at_end=False
    )
    
    # Create regime path and parameters for 10 years
    n_years = 11  # 0-10
    regime_path = np.zeros(n_years, dtype=int)  # All normal regime
    property_growth_rates = np.full(n_years, 0.02)  # 2% growth
    mortgage_rates = np.full(n_years, 0.05)  # 5% market rate (used for resets)
    
    # Simulate
    simulator = TimeSteppingSimulator(
        scenario=scenario,
        regime_path=regime_path,
        property_growth_rates=property_growth_rates,
        mortgage_rates=mortgage_rates,
        scenario_label="5y fixed test"
    )
    
    states = simulator.simulate_single_path()
    
    # Extract mortgage rates by year
    rates_by_year = {state.year: state.mortgage_rate for state in states}
    payments_by_year = {state.year: state.monthly_payment for state in states}
    
    print("\nYear | Mortgage Rate | Monthly Payment")
    print("-" * 45)
    for year in range(11):
        rate = rates_by_year.get(year, 0)
        payment = payments_by_year.get(year, 0)
        print(f"{year:4d} | {rate:12.4f} | €{payment:,.2f}")
    
    # CRITICAL ASSERTIONS:
    # 1. Years 1-5 should have the initial 3% rate
    for year in range(1, 6):
        assert rates_by_year[year] == pytest.approx(0.03, abs=1e-6), \
            f"Year {year} should have initial 3% rate, got {rates_by_year[year]}"
    
    # 2. Year 6 should have the reset rate (5% from mortgage_rates)
    assert rates_by_year[6] == pytest.approx(0.05, abs=1e-6), \
        f"Year 6 should have reset rate 5%, got {rates_by_year[year]}"
    
    # 3. CRITICAL: Years 7-10 should ALSO have the reset rate (5%)
    # This is the bug - currently they revert to the original rate
    for year in range(7, 11):
        assert rates_by_year[year] == pytest.approx(0.05, abs=1e-6), \
            f"Year {year} should persist reset rate 5%, got {rates_by_year[year]}"
    
    # 4. Monthly payments should be higher after the reset
    # (even though balance is lower, the rate increase should dominate initially)
    avg_payment_before_reset = np.mean([payments_by_year[y] for y in range(1, 6)])
    avg_payment_after_reset = np.mean([payments_by_year[y] for y in range(6, 11)])
    
    print(f"\nAverage payment years 1-5: €{avg_payment_before_reset:,.2f}")
    print(f"Average payment years 6-10: €{avg_payment_after_reset:,.2f}")
    print(f"Difference: €{avg_payment_after_reset - avg_payment_before_reset:,.2f}")
    
    # With a 2% rate increase (3% -> 5%), payments should increase
    # even as balance decreases
    assert avg_payment_after_reset > avg_payment_before_reset, \
        "Payments should increase after rate reset from 3% to 5%"


def test_multiple_rate_resets():
    """
    Test that multiple rate resets work correctly, with each rate persisting
    until the next reset.
    """
    # Create a scenario with resets at 5y and 10y
    loan = MortgageLoan(
        principal=300_000,
        annual_rate=0.03,  # Initial 3% rate
        term_years=30,
        rate_resets=[
            RateReset(
                month=60,  # Reset after 5 years
                new_rate_default=0.05,
                new_rate_min=0.02,
                new_rate_max=0.08
            ),
            RateReset(
                month=120,  # Reset after 10 years
                new_rate_default=0.04,
                new_rate_min=0.02,
                new_rate_max=0.08
            )
        ]
    )
    
    scenario = BuyingScenario(
        purchase_price=350_000,
        mortgage_loans=[loan],
        living_months=0,
        sold_at_end=False
    )
    
    # Create regime path and parameters for 15 years
    n_years = 16  # 0-15
    regime_path = np.zeros(n_years, dtype=int)
    property_growth_rates = np.full(n_years, 0.02)
    
    # Vary mortgage rates to test that resets pick up the current market rate
    mortgage_rates = np.full(n_years, 0.05)
    mortgage_rates[10:] = 0.04  # Market rate drops to 4% in year 10
    
    # Simulate
    simulator = TimeSteppingSimulator(
        scenario=scenario,
        regime_path=regime_path,
        property_growth_rates=property_growth_rates,
        mortgage_rates=mortgage_rates,
        scenario_label="Multiple resets test"
    )
    
    states = simulator.simulate_single_path()
    
    # Extract rates
    rates_by_year = {state.year: state.mortgage_rate for state in states}
    
    print("\nYear | Mortgage Rate")
    print("-" * 25)
    for year in range(16):
        rate = rates_by_year.get(year, 0)
        print(f"{year:4d} | {rate:12.4f}")
    
    # Years 1-5: Initial 3% rate
    for year in range(1, 6):
        assert rates_by_year[year] == pytest.approx(0.03, abs=1e-6), \
            f"Year {year} should have initial 3% rate"
    
    # Years 6-10: First reset to 5%
    for year in range(6, 11):
        assert rates_by_year[year] == pytest.approx(0.05, abs=1e-6), \
            f"Year {year} should have first reset rate 5%"
    
    # Years 11-15: Second reset to 4% (market rate at year 10)
    for year in range(11, 16):
        assert rates_by_year[year] == pytest.approx(0.04, abs=1e-6), \
            f"Year {year} should have second reset rate 4%"


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Rate reset persistence")
    print("=" * 60)
    test_rate_reset_persists_across_years()
    print("\n✓ Test 1 passed")
    
    print("\n" + "=" * 60)
    print("Test 2: Multiple rate resets")
    print("=" * 60)
    test_multiple_rate_resets()
    print("\n✓ Test 2 passed")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
