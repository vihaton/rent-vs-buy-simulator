"""
Unit tests for sensitivity analysis module.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from src.sensitivity import (
    _generate_values_for_strategy,
    generate_parameter_space,
    run_sensitivity_analysis,
    apply_constraints,
    export_results
)
from src.estate import BuyingScenario
from src.tax import NLHomeTax2026


class TestGenerateValuesForStrategy:
    """Tests for _generate_values_for_strategy function."""
    
    def test_linspace_strategy(self):
        """Test linspace strategy generates correct values."""
        config = {'strategy': 'linspace', 'min': 0, 'max': 100, 'steps': 11}
        values = _generate_values_for_strategy('test_param', config)
        
        assert len(values) == 11
        assert values[0] == 0
        assert values[-1] == 100
        assert np.allclose(values, np.linspace(0, 100, 11))
    
    def test_logspace_strategy(self):
        """Test logspace strategy generates correct values."""
        config = {'strategy': 'logspace', 'min': 1, 'max': 1000, 'steps': 4}
        values = _generate_values_for_strategy('test_param', config)
        
        assert len(values) == 4
        assert np.isclose(values[0], 1)
        assert np.isclose(values[-1], 1000)
    
    def test_random_strategy(self):
        """Test random strategy generates correct number of values."""
        config = {'strategy': 'random', 'min': 0, 'max': 100, 'samples': 50}
        values = _generate_values_for_strategy('test_param', config)
        
        assert len(values) == 50
        assert np.all(values >= 0)
        assert np.all(values <= 100)
    
    def test_random_strategy_default_samples(self):
        """Test random strategy uses default samples if not specified."""
        config = {'strategy': 'random', 'min': 0, 'max': 100}
        values = _generate_values_for_strategy('test_param', config)
        
        assert len(values) == 100  # default
    
    def test_values_strategy(self):
        """Test values strategy uses exact provided values."""
        config = {'strategy': 'values', 'values': [10, 20, 30, 40]}
        values = _generate_values_for_strategy('test_param', config)
        
        assert len(values) == 4
        assert np.array_equal(values, [10, 20, 30, 40])
    
    def test_unknown_strategy_raises_error(self):
        """Test unknown strategy raises ValueError."""
        config = {'strategy': 'unknown'}
        
        with pytest.raises(ValueError, match="unknown strategy"):
            _generate_values_for_strategy('test_param', config)
    
    def test_linspace_missing_fields_raises_error(self):
        """Test linspace with missing fields raises ValueError."""
        config = {'strategy': 'linspace', 'min': 0, 'max': 100}  # missing steps
        
        with pytest.raises(ValueError, match="requires min, max, and steps"):
            _generate_values_for_strategy('test_param', config)
    
    def test_linspace_invalid_range_raises_error(self):
        """Test linspace with min >= max raises ValueError."""
        config = {'strategy': 'linspace', 'min': 100, 'max': 0, 'steps': 10}
        
        with pytest.raises(ValueError, match="min must be less than max"):
            _generate_values_for_strategy('test_param', config)


class TestGenerateParameterSpace:
    """Tests for generate_parameter_space function."""
    
    def test_single_variable(self):
        """Test parameter space generation with single variable."""
        variables = {
            'purchase_price': {'strategy': 'linspace', 'min': 300000, 'max': 400000, 'steps': 3}
        }
        
        param_space = generate_parameter_space(variables)
        
        assert len(param_space) == 3
        assert all('purchase_price' in p for p in param_space)
        assert param_space[0]['purchase_price'] == 300000
        assert param_space[-1]['purchase_price'] == 400000
    
    def test_multiple_variables(self):
        """Test parameter space generation with multiple variables."""
        variables = {
            'purchase_price': {'strategy': 'values', 'values': [300000, 350000]},
            'monthly_vve': {'strategy': 'values', 'values': [200, 300, 400]}
        }
        
        param_space = generate_parameter_space(variables)
        
        # Should have 2 * 3 = 6 combinations
        assert len(param_space) == 6
        
        # Check all combinations exist
        prices = [300000, 350000]
        vves = [200, 300, 400]
        for price in prices:
            for vve in vves:
                assert any(p['purchase_price'] == price and p['monthly_vve'] == vve 
                          for p in param_space)
    
    def test_empty_variables_raises_error(self):
        """Test empty variables dict raises ValueError."""
        with pytest.raises(ValueError, match="No variables defined"):
            generate_parameter_space({})
    
    def test_three_variables_combinations(self):
        """Test parameter space with three variables."""
        variables = {
            'var1': {'strategy': 'values', 'values': [1, 2]},
            'var2': {'strategy': 'values', 'values': [10, 20]},
            'var3': {'strategy': 'values', 'values': [100, 200]}
        }
        
        param_space = generate_parameter_space(variables)
        
        # Should have 2 * 2 * 2 = 8 combinations
        assert len(param_space) == 8
        assert all(set(p.keys()) == {'var1', 'var2', 'var3'} for p in param_space)


class TestRunSensitivityAnalysis:
    """Tests for run_sensitivity_analysis function."""
    
    @pytest.fixture
    def base_scenario(self):
        """Create a base buying scenario for testing."""
        tax = NLHomeTax2026(
            woz_value=350000,
            marginal_tax_rate=0.37,
            deduction_cap_rate=0.37,
            apply_wet_hillen=True,
            rental_income_tax_rate=0.0
        )
        
        return BuyingScenario(
            purchase_price=330000,
            mortgage_principal=320000,
            mortgage_annual_rate=0.04,
            mortgage_term_years=30,
            living_months=36,
            one_off_costs=10000,
            renovation_costs_once=5000,
            monthly_vve=350,
            monthly_utilities=300,
            monthly_rent_income=0,
            months_rented=0,
            annual_value_growth=0.02,
            sold_at_end=True,
            selling_cost_rate=0.03,
            tax=tax
        )
    
    def test_run_analysis_single_combination(self, base_scenario):
        """Test analysis with single parameter combination."""
        param_space = [{'purchase_price': 340000}]
        
        results_df = run_sensitivity_analysis(base_scenario, param_space)
        
        assert len(results_df) == 1
        assert 'purchase_price' in results_df.columns
        assert 'wealth_end' in results_df.columns
        assert results_df['purchase_price'].iloc[0] == 340000
    
    def test_run_analysis_multiple_combinations(self, base_scenario):
        """Test analysis with multiple parameter combinations."""
        param_space = [
            {'purchase_price': 320000, 'monthly_vve': 300},
            {'purchase_price': 340000, 'monthly_vve': 400}
        ]
        
        results_df = run_sensitivity_analysis(base_scenario, param_space)
        
        assert len(results_df) == 2
        assert 'purchase_price' in results_df.columns
        assert 'monthly_vve' in results_df.columns
        assert 'wealth_end' in results_df.columns
    
    def test_monthly_net_cost_excluded(self, base_scenario):
        """Test that monthly_net_cost list is excluded from results."""
        param_space = [{'purchase_price': 340000}]
        
        results_df = run_sensitivity_analysis(base_scenario, param_space)
        
        assert 'monthly_net_cost' not in results_df.columns
    
    def test_verbose_output(self, base_scenario, capsys):
        """Test verbose output prints progress."""
        param_space = [{'purchase_price': 340000}]
        
        run_sensitivity_analysis(base_scenario, param_space, verbose=True)
        
        captured = capsys.readouterr()
        assert "Running sensitivity analysis" in captured.out
    
    def test_cash_required_with_correct_values(self, base_scenario):
        """Test that cash_required_upfront is correctly calculated when purchase > mortgage.
        
        This test verifies correct behavior when values are in the right order.
        """
        # Correct order: purchase_price > mortgage_principal
        param_space = [
            {'purchase_price': 365000, 'mortgage_principal': 360000},  # 5k down payment
        ]
        
        results_df = run_sensitivity_analysis(base_scenario, param_space)
        
        assert len(results_df) == 1
        row = results_df.iloc[0]
        
        # Correct calculation:
        # down_payment = 365000 - 360000 = 5000
        # cash_required = 5000 + 10000 (one_off) + 5000 (renovation) = 20000
        assert row['down_payment'] == 5000.0
        assert row['cash_required_upfront'] == 20000.0
        assert row['one_off_costs'] == 10000.0
        assert row['renovation_costs_once'] == 5000.0
    
    def test_cash_required_with_swapped_values(self, base_scenario):
        """Test that shows the bug when purchase_price and mortgage_principal are swapped.
        
        This documents what happens when the YAML has the values backwards,
        which is the issue in the user's scenario file.
        """
        # WRONG order: mortgage_principal > purchase_price (values swapped)
        param_space = [
            {'purchase_price': 360000, 'mortgage_principal': 365000},  # Swapped!
        ]
        
        results_df = run_sensitivity_analysis(base_scenario, param_space)
        
        assert len(results_df) == 1
        row = results_df.iloc[0]
        
        # When values are swapped (BUG in configuration):
        # down_payment = 360000 - 365000 = -5000
        # cash_required = -5000 + 10000 + 5000 = 10000 (WRONG!)
        assert row['down_payment'] == -5000.0
        assert row['cash_required_upfront'] == 10000.0  # Should be 20000 with correct values
        
        # This demonstrates the user's issue: they have the values swapped in their YAML
        # and get 10k instead of the expected 20k
    
    def test_cash_required_varies_correctly_in_sensitivity(self, base_scenario):
        """Test that cash_required_upfront varies when purchase_price or mortgage_principal changes."""
        param_space = [
            {'purchase_price': 300000, 'mortgage_principal': 280000},  # 20k down
            {'purchase_price': 300000, 'mortgage_principal': 290000},  # 10k down
            {'purchase_price': 350000, 'mortgage_principal': 330000},  # 20k down
        ]
        
        results_df = run_sensitivity_analysis(base_scenario, param_space)
        
        assert len(results_df) == 3
        
        # With base scenario having one_off_costs=10000 and renovation_costs_once=5000
        # Expected cash_required = down_payment + 10000 + 5000
        
        # Scenario 1: 20k down + 15k costs = 35k
        assert results_df.iloc[0]['down_payment'] == 20000.0
        assert results_df.iloc[0]['cash_required_upfront'] == 35000.0
        
        # Scenario 2: 10k down + 15k costs = 25k
        assert results_df.iloc[1]['down_payment'] == 10000.0
        assert results_df.iloc[1]['cash_required_upfront'] == 25000.0
        
        # Scenario 3: 20k down + 15k costs = 35k
        assert results_df.iloc[2]['down_payment'] == 20000.0
        assert results_df.iloc[2]['cash_required_upfront'] == 35000.0
        
        # Verify cash_required changes appropriately
        assert results_df['cash_required_upfront'].nunique() == 2  # Two unique values: 35k and 25k
    
    def test_total_spent_includes_down_payment(self, base_scenario):
        """Test that total_spent varies when purchase_price changes in sensitivity analysis.
        
        This test validates that total_spent includes the down payment, which should
        vary when purchase_price changes. This is a regression test for a bug where
        total_spent remained constant despite changes in purchase_price.
        
        Expected behavior:
        - total_spent should include: down_payment + one_off_costs + monthly_costs + mortgage_payments
        - When purchase_price increases (with constant mortgage_principal), down_payment increases
        - Therefore, total_spent should also increase
        """
        # Test with different purchase prices but same mortgage principal
        # This creates different down payments
        param_space = [
            {'purchase_price': 330000, 'mortgage_principal': 320000},  # 10k down
            {'purchase_price': 340000, 'mortgage_principal': 320000},  # 20k down
            {'purchase_price': 350000, 'mortgage_principal': 320000},  # 30k down
        ]
        
        results_df = run_sensitivity_analysis(base_scenario, param_space)
        
        assert len(results_df) == 3
        
        # Verify down payments are different
        assert results_df.iloc[0]['down_payment'] == 10000.0
        assert results_df.iloc[1]['down_payment'] == 20000.0
        assert results_df.iloc[2]['down_payment'] == 30000.0
        
        # CRITICAL: total_spent should vary with down_payment
        # This is the bug - currently total_spent stays the same
        total_spent_values = results_df['total_spent'].values
        
        # All three should be different
        assert len(set(total_spent_values)) == 3, \
            f"total_spent should vary with purchase_price, but got: {total_spent_values}"
        
        # Verify the relationship: higher down payment = higher total_spent
        assert total_spent_values[0] < total_spent_values[1] < total_spent_values[2], \
            f"total_spent should increase with down_payment: {total_spent_values}"
        
        # Verify the differences match the down payment differences
        diff_1_to_2 = total_spent_values[1] - total_spent_values[0]
        diff_2_to_3 = total_spent_values[2] - total_spent_values[1]
        
        assert abs(diff_1_to_2 - 10000.0) < 0.01, \
            f"Difference in total_spent should equal difference in down_payment (10k), got {diff_1_to_2}"
        assert abs(diff_2_to_3 - 10000.0) < 0.01, \
            f"Difference in total_spent should equal difference in down_payment (10k), got {diff_2_to_3}"
    
    def test_higher_down_payment_yields_higher_wealth(self, base_scenario):
        """Test that for the same purchase price, higher down payment leads to higher wealth.
        
        Economic principle: Paying more upfront (higher down payment) means borrowing less,
        which means paying less interest over time. This results in better wealth outcomes.
        
        The wealth calculation is complex because:
        - Higher down payment = more cash spent upfront
        - But also = less interest paid over time
        - And also = less principal to pay back (but principal isn't a "cost", it's equity)
        
        The key insight: When you pay more down payment, you're essentially pre-paying
        part of the principal, which saves you from paying interest on that amount.
        
        Expected behavior:
        - Same purchase_price, different mortgage_principal values
        - Higher down payment = lower mortgage = less interest paid = higher wealth_end
        """
        purchase_price = 350000
        
        # Test with different down payment amounts (same purchase price)
        param_space = [
            {'purchase_price': purchase_price, 'mortgage_principal': 340000},  # 10k down, 340k mortgage
            {'purchase_price': purchase_price, 'mortgage_principal': 330000},  # 20k down, 330k mortgage
            {'purchase_price': purchase_price, 'mortgage_principal': 320000},  # 30k down, 320k mortgage
        ]
        
        results_df = run_sensitivity_analysis(base_scenario, param_space)
        
        assert len(results_df) == 3
        
        # Verify down payments increase
        assert results_df.iloc[0]['down_payment'] == 10000.0
        assert results_df.iloc[1]['down_payment'] == 20000.0
        assert results_df.iloc[2]['down_payment'] == 30000.0
        
        # Verify mortgage interest paid decreases with higher down payment
        interest_paid = results_df['mortgage_interest_paid'].values
        assert interest_paid[0] > interest_paid[1] > interest_paid[2], \
            f"Interest paid should decrease with higher down payment: {interest_paid}"
        
        # CRITICAL: Higher down payment should lead to higher wealth_end
        # This is the key economic relationship we're testing
        wealth_values = results_df['wealth_end'].values
        
        assert wealth_values[0] < wealth_values[1] < wealth_values[2], \
            f"wealth_end should increase with higher down payment: {wealth_values}"
        
        # Verify the wealth improvement is positive and meaningful
        wealth_improvement_1_to_2 = wealth_values[1] - wealth_values[0]
        wealth_improvement_2_to_3 = wealth_values[2] - wealth_values[1]
        
        assert wealth_improvement_1_to_2 > 0, \
            f"Wealth should improve with higher down payment, got {wealth_improvement_1_to_2:.2f}"
        assert wealth_improvement_2_to_3 > 0, \
            f"Wealth should improve with higher down payment, got {wealth_improvement_2_to_3:.2f}"


class TestApplyConstraints:
    """Tests for apply_constraints function."""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample results DataFrame."""
        return pd.DataFrame({
            'purchase_price': [300000, 350000, 400000],
            'wealth_end': [50000, 40000, 30000],
            'avg_net_cost_per_month': [2000, 2500, 3000],
            'cash_required_upfront': [30000, 40000, 50000]
        })
    
    def test_single_constraint(self, sample_results):
        """Test filtering with single constraint."""
        constraints = ["avg_net_cost_per_month <= 2500"]
        
        filtered = apply_constraints(sample_results, constraints)
        
        assert len(filtered) == 2
        assert all(filtered['avg_net_cost_per_month'] <= 2500)
    
    def test_multiple_constraints(self, sample_results):
        """Test filtering with multiple constraints."""
        constraints = [
            "avg_net_cost_per_month <= 2500",
            "cash_required_upfront <= 40000"
        ]
        
        filtered = apply_constraints(sample_results, constraints)
        
        assert len(filtered) == 2
        assert all(filtered['avg_net_cost_per_month'] <= 2500)
        assert all(filtered['cash_required_upfront'] <= 40000)
    
    def test_no_constraints(self, sample_results):
        """Test that empty constraints list returns original DataFrame."""
        filtered = apply_constraints(sample_results, [])
        
        assert len(filtered) == len(sample_results)
        pd.testing.assert_frame_equal(filtered, sample_results)
    
    def test_invalid_constraint_raises_error(self, sample_results):
        """Test invalid constraint raises ValueError."""
        constraints = ["nonexistent_column <= 100"]
        
        with pytest.raises(ValueError, match="Invalid constraint"):
            apply_constraints(sample_results, constraints)


class TestExportResults:
    """Tests for export_results function."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for export testing."""
        return pd.DataFrame({
            'purchase_price': [300000, 350000, 400000],
            'wealth_end': [50000, 40000, 30000],
            'avg_net_cost_per_month': [2000, 2500, 3000]
        })
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    def test_export_csv(self, sample_dataframe, temp_dir):
        """Test CSV export."""
        output_config = {
            'format': 'csv',
            'directory': temp_dir,
            'filename_prefix': 'test',
            'include_timestamp': False
        }
        
        filepath = export_results(sample_dataframe, output_config)
        
        assert Path(filepath).exists()
        assert filepath.endswith('.csv')
        
        # Verify content
        loaded_df = pd.read_csv(filepath)
        pd.testing.assert_frame_equal(loaded_df, sample_dataframe)
    
    def test_export_json(self, sample_dataframe, temp_dir):
        """Test JSON export."""
        output_config = {
            'format': 'json',
            'directory': temp_dir,
            'filename_prefix': 'test',
            'include_timestamp': False
        }
        
        filepath = export_results(sample_dataframe, output_config)
        
        assert Path(filepath).exists()
        assert filepath.endswith('.json')
        
        # Verify content
        loaded_df = pd.read_json(filepath)
        pd.testing.assert_frame_equal(loaded_df, sample_dataframe)
    
    def test_export_with_timestamp(self, sample_dataframe, temp_dir):
        """Test export with timestamp in filename."""
        output_config = {
            'format': 'csv',
            'directory': temp_dir,
            'filename_prefix': 'test',
            'include_timestamp': True
        }
        
        filepath = export_results(sample_dataframe, output_config)
        
        assert Path(filepath).exists()
        assert 'test_' in filepath
        assert filepath.endswith('.csv')
    
    def test_export_with_metadata(self, sample_dataframe, temp_dir):
        """Test export with metadata file."""
        output_config = {
            'format': 'csv',
            'directory': temp_dir,
            'filename_prefix': 'test',
            'include_timestamp': False
        }
        
        metadata = {'test_key': 'test_value', 'count': 3}
        
        filepath = export_results(sample_dataframe, output_config, metadata=metadata)
        
        # Check metadata file exists
        metadata_path = filepath.replace('.csv', '_metadata.json')
        assert Path(metadata_path).exists()
        
        # Verify metadata content
        import json
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        assert loaded_metadata['test_key'] == 'test_value'
        assert loaded_metadata['count'] == 3
    
    def test_export_creates_directory(self, sample_dataframe, temp_dir):
        """Test export creates output directory if it doesn't exist."""
        nested_dir = str(Path(temp_dir) / 'nested' / 'path')
        
        output_config = {
            'format': 'csv',
            'directory': nested_dir,
            'filename_prefix': 'test',
            'include_timestamp': False
        }
        
        filepath = export_results(sample_dataframe, output_config)
        
        assert Path(filepath).exists()
        assert Path(nested_dir).exists()
    
    def test_export_unsupported_format_raises_error(self, sample_dataframe, temp_dir):
        """Test unsupported format raises ValueError."""
        output_config = {
            'format': 'xml',
            'directory': temp_dir,
            'filename_prefix': 'test'
        }
        
        with pytest.raises(ValueError, match="Unsupported format"):
            export_results(sample_dataframe, output_config)


class TestWildcardParameterPaths:
    """Tests for wildcard syntax in parameter paths."""
    
    def test_wildcard_applies_to_all_list_elements(self):
        """Test that wildcard [*] applies value to all elements in a list."""
        from src.sensitivity import _set_nested_attr
        from src.mortgage import MortgageLoan
        
        # Create a mock object with a list of mortgage loans
        class MockScenario:
            def __init__(self):
                self.mortgage_loans = [
                    MortgageLoan(
                        principal=100000,
                        annual_rate=0.03,
                        term_years=30,
                        amortization_type="annuity",
                        fixed_period_years=10,
                        rate_resets=[
                            type('RateReset', (), {'month': 120, 'new_rate_default': 0.04})()
                        ]
                    ),
                    MortgageLoan(
                        principal=50000,
                        annual_rate=0.035,
                        term_years=30,
                        amortization_type="linear",
                        fixed_period_years=10,
                        rate_resets=[
                            type('RateReset', (), {'month': 120, 'new_rate_default': 0.04})()
                        ]
                    )
                ]
        
        scenario = MockScenario()
        
        # Apply wildcard parameter
        _set_nested_attr(scenario, "mortgage_loans[*].rate_resets[0].new_rate_default", 0.05)
        
        # Verify all loans were updated
        assert scenario.mortgage_loans[0].rate_resets[0].new_rate_default == 0.05
        assert scenario.mortgage_loans[1].rate_resets[0].new_rate_default == 0.05
    
    def test_wildcard_with_simple_attribute(self):
        """Test wildcard with simple attribute (not nested)."""
        from src.sensitivity import _set_nested_attr
        from src.mortgage import MortgageLoan
        
        class MockScenario:
            def __init__(self):
                self.mortgage_loans = [
                    MortgageLoan(
                        principal=100000,
                        annual_rate=0.03,
                        term_years=30,
                        amortization_type="annuity",
                        fixed_period_years=10
                    ),
                    MortgageLoan(
                        principal=50000,
                        annual_rate=0.035,
                        term_years=30,
                        amortization_type="linear",
                        fixed_period_years=10
                    )
                ]
        
        scenario = MockScenario()
        
        # Apply wildcard to simple attribute
        _set_nested_attr(scenario, "mortgage_loans[*].annual_rate", 0.04)
        
        # Verify all loans were updated
        assert scenario.mortgage_loans[0].annual_rate == 0.04
        assert scenario.mortgage_loans[1].annual_rate == 0.04
    
    def test_wildcard_on_non_list_raises_error(self):
        """Test that wildcard on non-list object raises ValueError."""
        from src.sensitivity import _set_nested_attr
        
        class MockScenario:
            def __init__(self):
                self.single_loan = type('Loan', (), {'rate': 0.03})()
        
        scenario = MockScenario()
        
        # Should raise error when wildcard used on non-list
        with pytest.raises(ValueError, match="Wildcard.*used on non-list"):
            _set_nested_attr(scenario, "single_loan[*].rate", 0.04)
    
    def test_rebuild_path_from_parts(self):
        """Test path reconstruction from parts."""
        from src.sensitivity import _rebuild_path_from_parts
        
        # Test simple path
        parts = ['mortgage_loans', '0', 'annual_rate']
        path = _rebuild_path_from_parts(parts)
        assert path == 'mortgage_loans[0].annual_rate'
        
        # Test path with wildcard
        parts = ['mortgage_loans', '*', 'rate_resets', '0', 'new_rate_default']
        path = _rebuild_path_from_parts(parts)
        assert path == 'mortgage_loans[*].rate_resets[0].new_rate_default'
        
        # Test empty parts
        parts = []
        path = _rebuild_path_from_parts(parts)
        assert path == ''
