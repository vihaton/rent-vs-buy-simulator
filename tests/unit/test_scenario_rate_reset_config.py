"""
Test that scenario loading auto-generates rate resets based on fixed_period_years.

This test verifies that the load_scenario function automatically generates
rate resets when fixed_period_years is specified but rate_resets are not.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from src.time_stepping_comparison import load_scenario
from src.mortgage import RateReset


def test_auto_generate_rate_resets_for_10y_fixed_30y_term():
    """
    Test that a 30-year loan with 10-year fixed period auto-generates 2 resets.
    
    When rate_resets are not explicitly defined, the system should automatically
    generate resets at each fixed_period_years boundary:
    - Reset at month 120 (after first 10 years)
    - Reset at month 240 (after second 10 years)
    """
    # Create a test scenario without explicit rate_resets
    test_scenario = {
        'name': 'Test 30y with 10y fixed',
        'buying': {
            'purchase_price': 300000,
            'mortgage_principal': 270000,
            'mortgage_loans': [
                {
                    'percentage': 1.0,
                    'annual_rate': 0.03,
                    'term_years': 30,
                    'amortization_type': 'annuity',
                    'fixed_period_years': 10,
                    # Note: NO rate_resets defined - should be auto-generated
                    'label': 'Test Loan 10y fixed'
                }
            ],
            'living_months': 360,
            'one_off_costs': 10000,
            'monthly_vve': 200,
            'monthly_utilities': 150,
            'annual_value_growth': 0.02,
            'sold_at_end': True,
            'selling_cost_rate': 0.02
        }
    }
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_scenario, f)
        temp_file = f.name
    
    try:
        # Load the scenario
        scenario, label = load_scenario(temp_file)
        
        # Verify that mortgage loans were loaded
        assert scenario.mortgage_loans is not None, "Mortgage loans should be loaded"
        assert len(scenario.mortgage_loans) == 1, "Should have exactly one loan"
        
        loan = scenario.mortgage_loans[0]
        
        # Should have auto-generated rate resets
        assert loan.rate_resets is not None, \
            f"Loan {loan.label} should have auto-generated rate resets"
        
        # Should have exactly 2 resets for 30-year term with 10-year fixed
        assert len(loan.rate_resets) == 2, \
            f"Loan {loan.label} should have 2 resets, got {len(loan.rate_resets)}"
        
        # Check reset months
        reset_months = [reset.month for reset in loan.rate_resets]
        expected_months = [120, 240]
        
        assert reset_months == expected_months, \
            f"Loan {loan.label} should have resets at months {expected_months}, got {reset_months}"
        
        # Verify reset properties
        for reset in loan.rate_resets:
            assert isinstance(reset, RateReset), "Should be RateReset object"
            assert reset.new_rate_min > 0, "Should have positive min rate"
            assert reset.new_rate_max > reset.new_rate_min, "Max rate should be > min rate"
            assert reset.new_rate_default is not None, "Should have default rate"
    
    finally:
        # Clean up temp file
        Path(temp_file).unlink()


def test_auto_generate_rate_resets_for_5y_fixed_30y_term():
    """
    Test that a 30-year loan with 5-year fixed period auto-generates 5 resets.
    
    Resets should occur at months: 60, 120, 180, 240, 300
    """
    # Create a test scenario without explicit rate_resets
    test_scenario = {
        'name': 'Test 30y with 5y fixed',
        'buying': {
            'purchase_price': 300000,
            'mortgage_principal': 270000,
            'mortgage_loans': [
                {
                    'percentage': 1.0,
                    'annual_rate': 0.035,
                    'term_years': 30,
                    'amortization_type': 'linear',
                    'fixed_period_years': 5,
                    # Note: NO rate_resets defined - should be auto-generated
                    'label': 'Test Loan 5y fixed'
                }
            ],
            'living_months': 360,
            'one_off_costs': 10000,
            'monthly_vve': 200,
            'monthly_utilities': 150,
            'annual_value_growth': 0.02,
            'sold_at_end': True,
            'selling_cost_rate': 0.02
        }
    }
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_scenario, f)
        temp_file = f.name
    
    try:
        # Load the scenario
        scenario, label = load_scenario(temp_file)
        
        loan = scenario.mortgage_loans[0]
        
        # Should have 5 resets (at 5, 10, 15, 20, 25 years)
        assert loan.rate_resets is not None, \
            f"Loan {loan.label} should have auto-generated rate resets"
        
        assert len(loan.rate_resets) == 5, \
            f"Loan {loan.label} should have 5 resets, got {len(loan.rate_resets)}"
        
        # Check reset months: 60, 120, 180, 240, 300
        reset_months = [reset.month for reset in loan.rate_resets]
        expected_months = [60, 120, 180, 240, 300]
        
        assert reset_months == expected_months, \
            f"Loan {loan.label} should have resets at months {expected_months}, got {reset_months}"
    
    finally:
        # Clean up temp file
        Path(temp_file).unlink()


def test_explicit_rate_resets_take_precedence():
    """
    Test that explicitly defined rate_resets are used instead of auto-generation.
    
    If a scenario file has rate_resets defined, those should be used as-is,
    not auto-generated.
    """
    # Create a test scenario WITH explicit rate_resets (only 1 reset)
    test_scenario = {
        'name': 'Test with explicit reset',
        'buying': {
            'purchase_price': 300000,
            'mortgage_principal': 270000,
            'mortgage_loans': [
                {
                    'percentage': 1.0,
                    'annual_rate': 0.03,
                    'term_years': 30,
                    'amortization_type': 'annuity',
                    'fixed_period_years': 10,
                    # Explicit rate_resets - only 1 reset (not the expected 2)
                    'rate_resets': [
                        {
                            'month': 120,
                            'new_rate_min': 0.025,
                            'new_rate_max': 0.07,
                            'new_rate_default': 0.045
                        }
                    ],
                    'label': 'Test Loan with explicit reset'
                }
            ],
            'living_months': 360,
            'one_off_costs': 10000,
            'monthly_vve': 200,
            'monthly_utilities': 150,
            'annual_value_growth': 0.02,
            'sold_at_end': True,
            'selling_cost_rate': 0.02
        }
    }
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_scenario, f)
        temp_file = f.name
    
    try:
        # Load the scenario
        scenario, label = load_scenario(temp_file)
        
        loan = scenario.mortgage_loans[0]
        
        # Should have exactly 1 reset (the explicit one), not auto-generated 2
        assert loan.rate_resets is not None
        assert len(loan.rate_resets) == 1, \
            f"Should preserve explicit reset count (1), got {len(loan.rate_resets)}"
        
        # Check that it's the explicit reset we defined
        reset = loan.rate_resets[0]
        assert reset.month == 120
        assert reset.new_rate_min == 0.025
        assert reset.new_rate_max == 0.07
        assert reset.new_rate_default == 0.045
    
    finally:
        # Clean up temp file
        Path(temp_file).unlink()


def test_no_resets_when_no_fixed_period():
    """
    Test that no rate resets are generated when fixed_period_years is not set.
    """
    test_scenario = {
        'name': 'Test without fixed period',
        'buying': {
            'purchase_price': 300000,
            'mortgage_principal': 270000,
            'mortgage_loans': [
                {
                    'percentage': 1.0,
                    'annual_rate': 0.03,
                    'term_years': 30,
                    'amortization_type': 'annuity',
                    # No fixed_period_years specified
                    'label': 'Test Loan no fixed period'
                }
            ],
            'living_months': 360,
            'one_off_costs': 10000,
            'monthly_vve': 200,
            'monthly_utilities': 150,
            'annual_value_growth': 0.02,
            'sold_at_end': True,
            'selling_cost_rate': 0.02
        }
    }
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_scenario, f)
        temp_file = f.name
    
    try:
        # Load the scenario
        scenario, label = load_scenario(temp_file)
        
        loan = scenario.mortgage_loans[0]
        
        # Should NOT have rate resets
        assert loan.rate_resets is None or len(loan.rate_resets) == 0, \
            "Should not generate resets when fixed_period_years is not set"
    
    finally:
        # Clean up temp file
        Path(temp_file).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
