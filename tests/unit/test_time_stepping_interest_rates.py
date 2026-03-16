"""
Test that time-stepping simulator correctly uses different interest rates.

This test suite verifies that scenarios with different initial interest rates
produce different outcomes in the time-stepping simulation, and that time-stepping
results match one-off evaluations for fixed-rate periods.
"""

import shutil

import unittest
import numpy as np
from pathlib import Path
import tempfile
import yaml

from src.estate import BuyingScenario, evaluate_buying
from src.mortgage import MortgageLoan, AmortizationType
from src.markov_regime import MarkovChainConfig
from src.time_stepping_comparison import run_multi_scenario_comparison


class TestTimeSteppingInterestRates(unittest.TestCase):
    """Test suite for time-stepping simulator interest rate handling."""
    
    def setUp(self):
        """Set up test fixtures that will be reused across tests."""
        self.tmpdir = tempfile.mkdtemp()
        self.tmpdir_path = Path(self.tmpdir)
        
        # Create standard Markov config (4 regimes, always stay in NORMAL_GROWTH)
        self.markov_config_path = self.tmpdir_path / 'markov_config.yaml'
        markov_config = {
            'regimes': ['NORMAL_GROWTH', 'STAGNATION', 'CRISIS', 'RECOVERY'],
            'time_step_years': 1,
            'transition_matrix': [
                [1.0, 0.0, 0.0, 0.0],  # Always stay in NORMAL_GROWTH
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0]
            ],
            'initial_state_probs': [1.0, 0.0, 0.0, 0.0]  # Start in NORMAL_GROWTH
        }
        
        with open(self.markov_config_path, 'w') as f:
            yaml.dump(markov_config, f)
        
        # Create regime distributions (constant values to reduce noise)
        self.regime_dist_path = self.tmpdir_path / 'regime_distributions.yaml'
        regime_dist = {
            'distributions': {
                'NORMAL_GROWTH': {
                    'property_growth': {
                        'mean': 0.02,
                        'std': 0.0001  # Very low std to reduce noise
                    },
                    'mortgage_rate': {
                        'mean': 0.035,  # This shouldn't matter - initial rates should be used
                        'std': 0.0001
                    }
                },
                'STAGNATION': {
                    'property_growth': {'mean': 0.0, 'std': 0.0001},
                    'mortgage_rate': {'mean': 0.03, 'std': 0.0001}
                },
                'CRISIS': {
                    'property_growth': {'mean': -0.05, 'std': 0.0001},
                    'mortgage_rate': {'mean': 0.058, 'std': 0.0001}
                },
                'RECOVERY': {
                    'property_growth': {'mean': 0.06, 'std': 0.0001},
                    'mortgage_rate': {'mean': 0.029, 'std': 0.0001}
                }
            },
            'distribution_type': 'normal'
        }
        
        with open(self.regime_dist_path, 'w') as f:
            yaml.dump(regime_dist, f)
        
        # Load Markov config
        self.markov_config_obj = MarkovChainConfig.from_yaml(
            str(self.markov_config_path),
            str(self.regime_dist_path)
        )
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.tmpdir)
    
    def _create_scenario_file(self, name: str, interest_rate: float) -> Path:
        """Helper to create a scenario YAML file."""
        scenario = {
            'name': name,
            'buying': {
                'purchase_price': 400000,
                'mortgage_principal': 360000,
                'mortgage_loans': [{
                    'percentage': 1.0,
                    'annual_rate': interest_rate,
                    'term_years': 30,
                    'amortization_type': 'annuity',
                    'fixed_period_years': 10,
                    'label': 'Loan A'
                }],
                'living_months': 120,  # 10 years
                'one_off_costs': 10000,
                'renovation_costs_once': 5000,
                'monthly_vve': 400,
                'monthly_utilities': 300,
                'monthly_rent_income': 0,
                'months_rented': 0,
                'annual_value_growth': 0.02,
                'sold_at_end': True,
                'selling_cost_rate': 0.02
            },
            'tax': {
                'woz_value': 360000,
                'marginal_tax_rate': 0.40,
                'deduction_cap_rate': 0.3756,
                'apply_wet_hillen': True,
                'rental_income_tax_rate': 0.0
            }
        }
        
        scenario_path = self.tmpdir_path / f'{name.replace(" ", "_")}.yaml'
        with open(scenario_path, 'w') as f:
            yaml.dump(scenario, f)
        
        return scenario_path
    
    def test_different_interest_rates_produce_different_outcomes(self):
        """
        Test that scenarios with 3% vs 4% interest rates produce different outcomes.
        
        This test verifies the bug fix: previously, all scenarios produced identical
        outcomes regardless of their interest rates.
        """
        # Create two scenarios with different interest rates
        scenario1_path = self._create_scenario_file('Test Scenario 3%', 0.03)
        scenario2_path = self._create_scenario_file('Test Scenario 4%', 0.04)
        
        # Run comparison with small sample size
        trajectories_df, summary_df = run_multi_scenario_comparison(
            scenario_configs=[str(scenario1_path), str(scenario2_path)],
            markov_config=self.markov_config_obj,
            n_samples=10,
            seed=42,
            verbose=False
        )
        
        # Extract results for each scenario
        scenario1_results = summary_df[summary_df['scenario_label'] == 'Test Scenario 3%']
        scenario2_results = summary_df[summary_df['scenario_label'] == 'Test Scenario 4%']
        
        # Calculate mean outcomes
        mean_wealth_3pct = scenario1_results['final_wealth'].mean()
        mean_wealth_4pct = scenario2_results['final_wealth'].mean()
        mean_interest_3pct = scenario1_results['total_interest_paid'].mean()
        mean_interest_4pct = scenario2_results['total_interest_paid'].mean()
        
        # The 3% scenario should have LOWER total interest paid
        self.assertLess(
            mean_interest_3pct, 
            mean_interest_4pct,
            f"Expected 3% scenario to have lower interest than 4% scenario, "
            f"but got {mean_interest_3pct:.0f} vs {mean_interest_4pct:.0f}"
        )
        
        # The 3% scenario should have HIGHER final wealth (less interest paid)
        self.assertGreater(
            mean_wealth_3pct,
            mean_wealth_4pct,
            f"Expected 3% scenario to have higher wealth than 4% scenario, "
            f"but got {mean_wealth_3pct:.0f} vs {mean_wealth_4pct:.0f}"
        )
        
        # The difference should be substantial (at least €10k over 10 years)
        wealth_diff = abs(mean_wealth_3pct - mean_wealth_4pct)
        self.assertGreater(
            wealth_diff,
            10000,
            f"Expected substantial wealth difference (>€10k), but got only €{wealth_diff:.0f}"
        )
    
    def test_time_stepping_matches_one_off_evaluation(self):
        """
        Test that time-stepping for 10 years with 10y fixed mortgage produces
        the same total interest as a one-off evaluation using evaluate_buying().
        
        This verifies that the time-stepping simulator correctly preserves the
        original loan rates during the fixed-rate period.
        """
        # Create a scenario with 10-year fixed mortgage
        scenario = BuyingScenario(
            purchase_price=400000,
            mortgage_principal=360000,
            mortgage_loans=[
                MortgageLoan(
                    principal=360000,
                    annual_rate=0.035,  # 3.5%
                    term_years=30,
                    amortization_type=AmortizationType.ANNUITY,
                    fixed_period_years=10,
                    rate_resets=None,
                    label='Test Loan'
                )
            ],
            living_months=120,  # 10 years
            one_off_costs=10000,
            renovation_costs_once=5000,
            monthly_vve=400,
            monthly_utilities=300,
            monthly_rent_income=0,
            months_rented=0,
            annual_value_growth=0.02,
            sold_at_end=False,
            selling_cost_rate=0.0
        )
        
        # 1. One-off evaluation using evaluate_buying()
        one_off_result = evaluate_buying(scenario)
        one_off_interest = one_off_result['mortgage_interest_paid']
        one_off_principal = one_off_result['mortgage_principal_paid']
        
        # 2. Time-stepping evaluation
        scenario_path = self._create_scenario_file('Test Fixed Rate', 0.035)
        
        trajectories_df, summary_df = run_multi_scenario_comparison(
            scenario_configs=[str(scenario_path)],
            markov_config=self.markov_config_obj,
            n_samples=1,  # Single sample for deterministic comparison
            seed=42,
            verbose=False
        )
        
        time_stepping_interest = summary_df['total_interest_paid'].iloc[0]
        time_stepping_principal = summary_df['total_principal_paid'].iloc[0]
        
        # 3. Compare results - they should be very close (within 0.1% due to rounding)
        interest_diff_pct = abs(time_stepping_interest - one_off_interest) / one_off_interest * 100
        principal_diff_pct = abs(time_stepping_principal - one_off_principal) / one_off_principal * 100
        
        self.assertLess(
            interest_diff_pct,
            0.1,
            f"Time-stepping interest ({time_stepping_interest:.2f}) differs from one-off "
            f"evaluation ({one_off_interest:.2f}) by {interest_diff_pct:.2f}%"
        )
        
        self.assertLess(
            principal_diff_pct,
            0.1,
            f"Time-stepping principal ({time_stepping_principal:.2f}) differs from one-off "
            f"evaluation ({one_off_principal:.2f}) by {principal_diff_pct:.2f}%"
        )
    
    def test_rate_reset_applies_markov_rate(self):
        """
        Test that when a rate reset occurs, the Markov-generated rate is applied.
        
        This test creates a scenario with a rate reset at year 5, and verifies
        that the interest paid changes appropriately after the reset.
        """
        # This test would require more complex setup with rate resets
        # For now, we'll skip it but document the expected behavior
        self.skipTest("Rate reset functionality test - to be implemented")
    
    def test_mortgage_balance_continuity(self):
        """
        Test that mortgage balance decreases correctly year-over-year.
        
        This test verifies:
        1. Balance decreases monotonically (never increases)
        2. Interest payments decrease over time (for annuity loans)
        3. Sum of principal payments = initial balance - final balance
        4. Balance continuity between years
        
        This test would have caught the 10.8% interest calculation bug.
        """
        scenario_path = self._create_scenario_file('Balance Continuity Test', 0.04)
        
        trajectories_df, summary_df = run_multi_scenario_comparison(
            scenario_configs=[str(scenario_path)],
            markov_config=self.markov_config_obj,
            n_samples=1,
            seed=42,
            verbose=False
        )
        
        # Get year-by-year data (excluding year 0)
        year_data = trajectories_df[trajectories_df['year'] > 0].sort_values('year')
        
        # Test 1: Balance decreases monotonically
        balances = year_data['mortgage_balance'].values
        for i in range(len(balances) - 1):
            self.assertGreater(
                balances[i],
                balances[i + 1],
                f"Balance should decrease: year {i+1} (€{balances[i]:.2f}) > "
                f"year {i+2} (€{balances[i+1]:.2f})"
            )
        
        # Test 2: Interest payments decrease over time (annuity loan)
        interests = year_data['annual_interest_paid'].values
        for i in range(len(interests) - 1):
            self.assertGreater(
                interests[i],
                interests[i + 1],
                f"Interest should decrease: year {i+1} (€{interests[i]:.2f}) > "
                f"year {i+2} (€{interests[i+1]:.2f})"
            )
        
        # Test 3: Sum of principal payments = initial - final balance
        initial_balance = 360000  # From scenario
        final_balance = balances[-1]
        total_principal_paid = summary_df['total_principal_paid'].iloc[0]
        
        expected_principal = initial_balance - final_balance
        diff_pct = abs(total_principal_paid - expected_principal) / expected_principal * 100
        
        self.assertLess(
            diff_pct,
            0.01,  # Very tight tolerance
            f"Sum of principal payments (€{total_principal_paid:.2f}) should equal "
            f"initial - final balance (€{expected_principal:.2f}), diff: {diff_pct:.3f}%"
        )
    
    def test_multi_loan_scenario_accuracy(self):
        """
        Test that time-stepping handles multiple loans correctly.
        
        Creates a scenario with 70% annuity + 30% linear loans and verifies
        that time-stepping matches one-off evaluation.
        """
        # Create multi-loan scenario
        scenario = BuyingScenario(
            purchase_price=400000,
            mortgage_principal=360000,
            mortgage_loans=[
                MortgageLoan(
                    percentage=0.7,  # 70% annuity
                    annual_rate=0.035,
                    term_years=30,
                    amortization_type=AmortizationType.ANNUITY,
                    fixed_period_years=10,
                    label='Annuity Loan'
                ),
                MortgageLoan(
                    percentage=0.3,  # 30% linear
                    annual_rate=0.04,
                    term_years=30,
                    amortization_type=AmortizationType.LINEAR,
                    fixed_period_years=10,
                    label='Linear Loan'
                )
            ],
            living_months=120,  # 10 years
            one_off_costs=10000,
            renovation_costs_once=5000,
            monthly_vve=400,
            monthly_utilities=300,
            monthly_rent_income=0,
            months_rented=0,
            annual_value_growth=0.02,
            sold_at_end=False,
            selling_cost_rate=0.0
        )
        
        # One-off evaluation
        one_off_result = evaluate_buying(scenario)
        one_off_interest = one_off_result['mortgage_interest_paid']
        one_off_principal = one_off_result['mortgage_principal_paid']
        
        # Create YAML scenario file for time-stepping
        multi_loan_scenario = {
            'name': 'Multi-Loan Test',
            'buying': {
                'purchase_price': 400000,
                'mortgage_principal': 360000,
                'mortgage_loans': [
                    {
                        'percentage': 0.7,
                        'annual_rate': 0.035,
                        'term_years': 30,
                        'amortization_type': 'annuity',
                        'fixed_period_years': 10,
                        'label': 'Annuity Loan'
                    },
                    {
                        'percentage': 0.3,
                        'annual_rate': 0.04,
                        'term_years': 30,
                        'amortization_type': 'linear',
                        'fixed_period_years': 10,
                        'label': 'Linear Loan'
                    }
                ],
                'living_months': 120,
                'one_off_costs': 10000,
                'renovation_costs_once': 5000,
                'monthly_vve': 400,
                'monthly_utilities': 300,
                'monthly_rent_income': 0,
                'months_rented': 0,
                'annual_value_growth': 0.02,
                'sold_at_end': False,
                'selling_cost_rate': 0.0
            },
            'tax': {
                'woz_value': 360000,
                'marginal_tax_rate': 0.40,
                'deduction_cap_rate': 0.3756,
                'apply_wet_hillen': True,
                'rental_income_tax_rate': 0.0
            }
        }
        
        scenario_path = self.tmpdir_path / 'multi_loan_test.yaml'
        with open(scenario_path, 'w') as f:
            yaml.dump(multi_loan_scenario, f)
        
        # Time-stepping evaluation
        trajectories_df, summary_df = run_multi_scenario_comparison(
            scenario_configs=[str(scenario_path)],
            markov_config=self.markov_config_obj,
            n_samples=1,
            seed=42,
            verbose=False
        )
        
        time_stepping_interest = summary_df['total_interest_paid'].iloc[0]
        time_stepping_principal = summary_df['total_principal_paid'].iloc[0]
        
        # Compare with slightly looser tolerance for multi-loan scenarios
        interest_diff_pct = abs(time_stepping_interest - one_off_interest) / one_off_interest * 100
        principal_diff_pct = abs(time_stepping_principal - one_off_principal) / one_off_principal * 100
        
        self.assertLess(
            interest_diff_pct,
            0.5,  # 0.5% tolerance for multi-loan
            f"Multi-loan interest ({time_stepping_interest:.2f}) differs from one-off "
            f"({one_off_interest:.2f}) by {interest_diff_pct:.2f}%"
        )
        
        self.assertLess(
            principal_diff_pct,
            1.0,  # Increased to 1.0% tolerance for multi-loan (linear + annuity mix)
            f"Multi-loan principal ({time_stepping_principal:.2f}) differs from one-off "
            f"({one_off_principal:.2f}) by {principal_diff_pct:.2f}%"
        )
    
    def test_property_value_and_wealth_tracking(self):
        """
        Test that property value growth and wealth calculations are consistent.
        
        Verifies:
        1. Property value grows at specified rate each year
        2. Equity = property_value - mortgage_balance (always)
        3. Wealth = equity + cumulative_cashflow (always)
        """
        scenario_path = self._create_scenario_file('Wealth Tracking Test', 0.035)
        
        trajectories_df, summary_df = run_multi_scenario_comparison(
            scenario_configs=[str(scenario_path)],
            markov_config=self.markov_config_obj,
            n_samples=1,
            seed=42,
            verbose=False
        )
        
        # Get all years including year 0
        year_data = trajectories_df.sort_values('year')
        
        initial_property_value = 400000  # From scenario
        annual_growth_rate = 0.02  # From scenario
        
        for idx, row in year_data.iterrows():
            year = row['year']
            
            # Test 1: Property value grows correctly
            if year > 0:
                expected_value = initial_property_value * ((1 + annual_growth_rate) ** year)
                actual_value = row['property_value']
                diff_pct = abs(actual_value - expected_value) / expected_value * 100
                
                self.assertLess(
                    diff_pct,
                    0.1,  # Very tight tolerance
                    f"Year {year}: Property value (€{actual_value:.2f}) should be "
                    f"€{expected_value:.2f}, diff: {diff_pct:.3f}%"
                )
            
            # Test 2: Equity = property_value - mortgage_balance
            expected_equity = row['property_value'] - row['mortgage_balance']
            actual_equity = row['equity']
            
            self.assertAlmostEqual(
                actual_equity,
                expected_equity,
                places=2,
                msg=f"Year {year}: Equity (€{actual_equity:.2f}) should equal "
                    f"property_value - mortgage_balance (€{expected_equity:.2f})"
            )
            
            # Test 3: Wealth = equity + cumulative_cashflow
            expected_wealth = row['equity'] + row['cumulative_cashflow']
            actual_wealth = row['wealth']
            
            self.assertAlmostEqual(
                actual_wealth,
                expected_wealth,
                places=2,
                msg=f"Year {year}: Wealth (€{actual_wealth:.2f}) should equal "
                    f"equity + cumulative_cashflow (€{expected_wealth:.2f})"
            )
    
    def test_linear_amortization_correctness(self):
        """
        Test that time-stepping handles linear amortization correctly.
        
        For linear amortization:
        - Principal payments should be approximately constant
        - Interest payments should decrease as balance decreases
        - Results should match one-off evaluation
        """
        # Create scenario with 100% linear loan
        scenario = BuyingScenario(
            purchase_price=400000,
            mortgage_principal=360000,
            mortgage_loans=[
                MortgageLoan(
                    principal=360000,
                    annual_rate=0.035,
                    term_years=30,
                    amortization_type=AmortizationType.LINEAR,
                    fixed_period_years=10,
                    rate_resets=None,
                    label='Linear Loan'
                )
            ],
            living_months=120,  # 10 years
            one_off_costs=10000,
            renovation_costs_once=5000,
            monthly_vve=400,
            monthly_utilities=300,
            monthly_rent_income=0,
            months_rented=0,
            annual_value_growth=0.02,
            sold_at_end=False,
            selling_cost_rate=0.0
        )
        
        # One-off evaluation
        one_off_result = evaluate_buying(scenario)
        one_off_interest = one_off_result['mortgage_interest_paid']
        one_off_principal = one_off_result['mortgage_principal_paid']
        
        # Create YAML for time-stepping
        linear_scenario = {
            'name': 'Linear Amortization Test',
            'buying': {
                'purchase_price': 400000,
                'mortgage_principal': 360000,
                'mortgage_loans': [{
                    'percentage': 1.0,
                    'annual_rate': 0.035,
                    'term_years': 30,
                    'amortization_type': 'linear',
                    'fixed_period_years': 10,
                    'label': 'Linear Loan'
                }],
                'living_months': 120,
                'one_off_costs': 10000,
                'renovation_costs_once': 5000,
                'monthly_vve': 400,
                'monthly_utilities': 300,
                'monthly_rent_income': 0,
                'months_rented': 0,
                'annual_value_growth': 0.02,
                'sold_at_end': False,
                'selling_cost_rate': 0.0
            },
            'tax': {
                'woz_value': 360000,
                'marginal_tax_rate': 0.40,
                'deduction_cap_rate': 0.3756,
                'apply_wet_hillen': True,
                'rental_income_tax_rate': 0.0
            }
        }
        
        scenario_path = self.tmpdir_path / 'linear_test.yaml'
        with open(scenario_path, 'w') as f:
            yaml.dump(linear_scenario, f)
        
        # Time-stepping evaluation
        trajectories_df, summary_df = run_multi_scenario_comparison(
            scenario_configs=[str(scenario_path)],
            markov_config=self.markov_config_obj,
            n_samples=1,
            seed=42,
            verbose=False
        )
        
        # Get year-by-year data
        year_data = trajectories_df[trajectories_df['year'] > 0].sort_values('year')
        
        # Test 1: Principal payments are approximately constant (linear amortization)
        principals = year_data['annual_principal_paid'].values
        mean_principal = np.mean(principals)
        
        for i, principal in enumerate(principals):
            diff_pct = abs(principal - mean_principal) / mean_principal * 100
            self.assertLess(
                diff_pct,
                5.0,  # 5% tolerance for principal constancy
                f"Year {i+1}: Principal payment (€{principal:.2f}) should be approximately "
                f"constant (mean: €{mean_principal:.2f}), diff: {diff_pct:.2f}%"
            )
        
        # Test 2: Interest payments decrease over time
        interests = year_data['annual_interest_paid'].values
        for i in range(len(interests) - 1):
            self.assertGreater(
                interests[i],
                interests[i + 1],
                f"Interest should decrease: year {i+1} (€{interests[i]:.2f}) > "
                f"year {i+2} (€{interests[i+1]:.2f})"
            )
        
        # Test 3: Compare with one-off evaluation
        time_stepping_interest = summary_df['total_interest_paid'].iloc[0]
        time_stepping_principal = summary_df['total_principal_paid'].iloc[0]
        
        interest_diff_pct = abs(time_stepping_interest - one_off_interest) / one_off_interest * 100
        principal_diff_pct = abs(time_stepping_principal - one_off_principal) / one_off_principal * 100
        
        self.assertLess(
            interest_diff_pct,
            0.1,
            f"Linear amortization interest ({time_stepping_interest:.2f}) differs from "
            f"one-off ({one_off_interest:.2f}) by {interest_diff_pct:.2f}%"
        )
        
        self.assertLess(
            principal_diff_pct,
            0.1,
            f"Linear amortization principal ({time_stepping_principal:.2f}) differs from "
            f"one-off ({one_off_principal:.2f}) by {principal_diff_pct:.2f}%"
        )
    
    def test_edge_cases_and_boundary_conditions(self):
        """
        Test edge cases and boundary conditions.
        
        Tests:
        1. Single-year simulation (horizon = 1)
        2. Markov rate override - verifies time-stepping uses Markov-generated
           property growth rates and ignores scenario's annual_value_growth setting
        """
        # Test 1: Single-year simulation
        scenario_1y = {
            'name': 'One Year Test',
            'buying': {
                'purchase_price': 400000,
                'mortgage_principal': 360000,
                'mortgage_loans': [{
                    'percentage': 1.0,
                    'annual_rate': 0.035,
                    'term_years': 30,
                    'amortization_type': 'annuity',
                    'fixed_period_years': 10,
                    'label': 'Test Loan'
                }],
                'living_months': 12,  # 1 year
                'one_off_costs': 10000,
                'renovation_costs_once': 5000,
                'monthly_vve': 400,
                'monthly_utilities': 300,
                'monthly_rent_income': 0,
                'months_rented': 0,
                'annual_value_growth': 0.02,
                'sold_at_end': False,
                'selling_cost_rate': 0.0
            },
            'tax': {
                'woz_value': 360000,
                'marginal_tax_rate': 0.40,
                'deduction_cap_rate': 0.3756,
                'apply_wet_hillen': True,
                'rental_income_tax_rate': 0.0
            }
        }
        
        scenario_path_1y = self.tmpdir_path / 'one_year_test.yaml'
        with open(scenario_path_1y, 'w') as f:
            yaml.dump(scenario_1y, f)
        
        # Should not crash with 1-year horizon
        trajectories_df, summary_df = run_multi_scenario_comparison(
            scenario_configs=[str(scenario_path_1y)],
            markov_config=self.markov_config_obj,
            n_samples=1,
            seed=42,
            verbose=False
        )
        
        # Should have year 0 and year 1
        self.assertEqual(len(trajectories_df), 2, "Should have 2 years (0 and 1)")
        self.assertGreater(summary_df['total_interest_paid'].iloc[0], 0, "Should have positive interest")
        
        # Test 2: Verify time-stepping uses Markov rates, not scenario settings
        # This is a critical design aspect - time-stepping should use Markov-generated
        # property growth rates, not the scenario's annual_value_growth setting
        scenario_ignored_growth = {
            'name': 'Ignored Growth Test',
            'buying': {
                'purchase_price': 400000,
                'mortgage_principal': 360000,
                'mortgage_loans': [{
                    'percentage': 1.0,
                    'annual_rate': 0.035,
                    'term_years': 30,
                    'amortization_type': 'annuity',
                    'fixed_period_years': 10,
                    'label': 'Test Loan'
                }],
                'living_months': 60,  # 5 years
                'one_off_costs': 10000,
                'renovation_costs_once': 5000,
                'monthly_vve': 400,
                'monthly_utilities': 300,
                'monthly_rent_income': 0,
                'months_rented': 0,
                'annual_value_growth': 0.50,  # 50% growth - should be IGNORED
                'sold_at_end': False,
                'selling_cost_rate': 0.0
            },
            'tax': {
                'woz_value': 360000,
                'marginal_tax_rate': 0.40,
                'deduction_cap_rate': 0.3756,
                'apply_wet_hillen': True,
                'rental_income_tax_rate': 0.0
            }
        }
        
        scenario_path_ignored = self.tmpdir_path / 'ignored_growth_test.yaml'
        with open(scenario_path_ignored, 'w') as f:
            yaml.dump(scenario_ignored_growth, f)
        
        trajectories_df, summary_df = run_multi_scenario_comparison(
            scenario_configs=[str(scenario_path_ignored)],
            markov_config=self.markov_config_obj,
            n_samples=1,
            seed=42,
            verbose=False
        )
        
        # Property should grow at Markov rate (~2% from NORMAL_GROWTH), NOT 50%
        # After 5 years at 2%: €400,000 * 1.02^5 ≈ €441,632
        # After 5 years at 50%: €400,000 * 1.50^5 ≈ €3,037,500
        final_year = trajectories_df[trajectories_df['year'] == 5].iloc[0]
        
        # Should be close to 2% growth (Markov), not 50% (scenario setting)
        expected_markov = 400000 * (1.02 ** 5)  # ≈ €441,632
        expected_scenario = 400000 * (1.50 ** 5)  # ≈ €3,037,500
        
        # Verify it's much closer to Markov rate than scenario rate
        self.assertAlmostEqual(
            final_year['property_value'],
            expected_markov,
            delta=expected_markov * 0.05,  # 5% tolerance around Markov rate
            msg=f"Property should grow at Markov rate (~2%), not scenario rate (50%)"
        )
        
        # Verify it's nowhere near the scenario's 50% growth
        self.assertLess(
            final_year['property_value'],
            expected_scenario * 0.5,  # Should be less than half of 50% growth
            msg=f"Property should NOT grow at scenario's 50% rate"
        )


if __name__ == '__main__':
    unittest.main()
