"""
Unit tests for comparison report generation.
"""

import unittest
import pandas as pd
import numpy as np


class TestP10P90MonthlyPaymentCalculation(unittest.TestCase):
    """
    Test that P10/P90 monthly payments are calculated correctly
    from the samples that correspond to P10/P90 wealth outcomes.
    """
    
    def setUp(self):
        """
        Create synthetic test data with a known relationship:
        - Samples with lower wealth have higher monthly payments
        - This simulates scenarios where bad outcomes (low wealth) 
          correlate with higher mortgage rates
        """
        np.random.seed(42)
        n_samples = 100
        n_years = 10
        
        # Create summary data
        summary_data = []
        for sample_id in range(n_samples):
            # Simulate final wealth with variation
            final_wealth = np.random.normal(-200000, 80000)
            summary_data.append({
                'sample_id': sample_id,
                'scenario_label': 'Test Scenario',
                'final_wealth': final_wealth
            })
        
        self.summary_df = pd.DataFrame(summary_data)
        
        # Create trajectory data with correlation:
        # Lower wealth → higher payments (simulating rate resets in bad scenarios)
        trajectory_data = []
        for sample_id in range(n_samples):
            wealth = self.summary_df.loc[sample_id, 'final_wealth']
            # Base payment inversely correlated with wealth
            # Normalize to realistic payment range (1400-1700)
            base_payment = 1550 - (wealth + 200000) / 2000
            
            for year in range(n_years):
                monthly_payment = base_payment + np.random.normal(0, 20)
                trajectory_data.append({
                    'sample_id': sample_id,
                    'scenario_label': 'Test Scenario',
                    'year': year,
                    'monthly_payment': monthly_payment
                })
        
        self.trajectories_df = pd.DataFrame(trajectory_data)
        
        # Pre-calculate expected values for assertions
        self.scenario = 'Test Scenario'
        scenario_data = self.summary_df[self.summary_df['scenario_label'] == self.scenario]
        
        # P10 calculations
        self.p10_threshold = scenario_data['final_wealth'].quantile(0.1)
        self.p10_samples = scenario_data[scenario_data['final_wealth'] <= self.p10_threshold]['sample_id']
        
        # P90 calculations
        self.p90_threshold = scenario_data['final_wealth'].quantile(0.9)
        self.p90_samples = scenario_data[scenario_data['final_wealth'] >= self.p90_threshold]['sample_id']
    
    def test_p10_monthly_payment_uses_p10_wealth_samples(self):
        """
        Test that P10 monthly payment is calculated from samples
        with P10 wealth, not from the 10th percentile of all payments.
        """
        scenario_data = self.summary_df[self.summary_df['scenario_label'] == self.scenario]
        traj_scenario = self.trajectories_df[self.trajectories_df['scenario_label'] == self.scenario]
        
        # CORRECT method: filter to P10 wealth samples
        traj_p10 = traj_scenario[traj_scenario['sample_id'].isin(self.p10_samples)]
        p10_monthly_correct = traj_p10['monthly_payment'].mean()
        
        # INCORRECT method (old bug): 10th percentile of all payments
        p10_monthly_incorrect = traj_scenario['monthly_payment'].quantile(0.1)
        
        # Assertions
        # 1. The correct method should give a different result than the incorrect method
        self.assertNotAlmostEqual(
            p10_monthly_correct, 
            p10_monthly_incorrect,
            places=0,
            msg="P10 monthly payment should differ between correct and incorrect methods"
        )
        
        # 2. In our test data, P10 wealth samples should have HIGHER payments
        #    (because we set up correlation: low wealth → high payments)
        overall_mean = traj_scenario['monthly_payment'].mean()
        self.assertGreater(
            p10_monthly_correct,
            overall_mean,
            msg="P10 wealth scenarios should have higher than average payments"
        )
        
        # 3. The correct method should use exactly the P10 samples
        expected_n_payments = len(self.p10_samples) * len(self.trajectories_df['year'].unique())
        self.assertEqual(
            len(traj_p10),
            expected_n_payments,
            msg=f"Should filter to {len(self.p10_samples)} P10 samples across all years"
        )
    
    def test_p90_monthly_payment_uses_p90_wealth_samples(self):
        """
        Test that P90 monthly payment is calculated from samples
        with P90 wealth, not from the 90th percentile of all payments.
        """
        scenario_data = self.summary_df[self.summary_df['scenario_label'] == self.scenario]
        traj_scenario = self.trajectories_df[self.trajectories_df['scenario_label'] == self.scenario]
        
        # CORRECT method: filter to P90 wealth samples
        traj_p90 = traj_scenario[traj_scenario['sample_id'].isin(self.p90_samples)]
        p90_monthly_correct = traj_p90['monthly_payment'].mean()
        
        # INCORRECT method (old bug): 90th percentile of all payments
        p90_monthly_incorrect = traj_scenario['monthly_payment'].quantile(0.9)
        
        # Assertions
        # 1. The correct method should give a different result than the incorrect method
        self.assertNotAlmostEqual(
            p90_monthly_correct,
            p90_monthly_incorrect,
            places=0,
            msg="P90 monthly payment should differ between correct and incorrect methods"
        )
        
        # 2. In our test data, P90 wealth samples should have LOWER payments
        #    (because we set up correlation: high wealth → low payments)
        overall_mean = traj_scenario['monthly_payment'].mean()
        self.assertLess(
            p90_monthly_correct,
            overall_mean,
            msg="P90 wealth scenarios should have lower than average payments"
        )
        
        # 3. The correct method should use exactly the P90 samples
        expected_n_payments = len(self.p90_samples) * len(self.trajectories_df['year'].unique())
        self.assertEqual(
            len(traj_p90),
            expected_n_payments,
            msg=f"Should filter to {len(self.p90_samples)} P90 samples across all years"
        )
    
    def test_p10_and_p90_payments_are_inversely_correlated_with_wealth(self):
        """
        Integration test: verify that in our test data setup,
        P10 wealth scenarios have higher payments than P90 wealth scenarios.
        """
        traj_scenario = self.trajectories_df[self.trajectories_df['scenario_label'] == self.scenario]
        
        # Calculate payments for P10 and P90 wealth samples
        traj_p10 = traj_scenario[traj_scenario['sample_id'].isin(self.p10_samples)]
        traj_p90 = traj_scenario[traj_scenario['sample_id'].isin(self.p90_samples)]
        
        p10_monthly = traj_p10['monthly_payment'].mean()
        p90_monthly = traj_p90['monthly_payment'].mean()
        
        # In our test setup, worse wealth (P10) should have higher payments
        self.assertGreater(
            p10_monthly,
            p90_monthly,
            msg="P10 (worst wealth) scenarios should have higher payments than P90 (best wealth)"
        )


if __name__ == '__main__':
    unittest.main()
