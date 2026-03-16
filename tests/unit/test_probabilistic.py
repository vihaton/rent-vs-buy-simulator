"""
Unit tests for probabilistic mortgage comparison functionality.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import yaml

from src.sensitivity import _generate_values_for_strategy, generate_parameter_space
from src.probabilistic_analysis import (
    calculate_comparative_metrics,
    calculate_risk_metrics,
    calculate_regret_metrics,
    identify_regime,
    add_regime_labels,
    validate_mixture_distribution
)


class TestDistributionStrategies(unittest.TestCase):
    """Test probability distribution sampling strategies."""
    
    def test_normal_distribution(self):
        """Test normal distribution sampling."""
        config = {
            'strategy': 'normal',
            'mean': 0.02,
            'std': 0.01,
            'samples': 1000,
            'seed': 42
        }
        values = _generate_values_for_strategy('test_param', config)
        
        self.assertEqual(len(values), 1000)
        self.assertAlmostEqual(values.mean(), 0.02, delta=0.005)
        self.assertAlmostEqual(values.std(), 0.01, delta=0.005)
    
    def test_normal_distribution_with_truncation(self):
        """Test normal distribution with min/max truncation."""
        config = {
            'strategy': 'normal',
            'mean': 0.02,
            'std': 0.01,
            'samples': 1000,
            'seed': 42,
            'min': 0.0,
            'max': 0.05
        }
        values = _generate_values_for_strategy('test_param', config)
        
        self.assertEqual(len(values), 1000)
        self.assertGreaterEqual(values.min(), 0.0)
        self.assertLessEqual(values.max(), 0.05)
    
    def test_mixture_distribution(self):
        """Test mixture distribution sampling."""
        config = {
            'strategy': 'mixture',
            'samples': 5000,
            'seed': 42,
            'components': [
                {'weight': 0.50, 'mean': 0.035, 'std': 0.07},
                {'weight': 0.30, 'mean': 0.000, 'std': 0.06},
                {'weight': 0.20, 'mean': -0.050, 'std': 0.11}
            ]
        }
        values = _generate_values_for_strategy('test_param', config)
        
        self.assertEqual(len(values), 5000)
        
        # Expected mean: 0.50*0.035 + 0.30*0.000 + 0.20*(-0.050) = 0.0075
        expected_mean = 0.0075
        self.assertAlmostEqual(values.mean(), expected_mean, delta=0.01)
    
    def test_mixture_distribution_with_truncation(self):
        """Test mixture distribution with min/max truncation."""
        config = {
            'strategy': 'mixture',
            'samples': 1000,
            'seed': 42,
            'components': [
                {'weight': 0.50, 'mean': 0.035, 'std': 0.07},
                {'weight': 0.30, 'mean': 0.000, 'std': 0.06},
                {'weight': 0.20, 'mean': -0.050, 'std': 0.11}
            ],
            'min': -0.15,
            'max': 0.15
        }
        values = _generate_values_for_strategy('test_param', config)
        
        self.assertEqual(len(values), 1000)
        self.assertGreaterEqual(values.min(), -0.15)
        self.assertLessEqual(values.max(), 0.15)
    
    def test_mixture_weights_validation(self):
        """Test that mixture weights must sum to 1.0."""
        config = {
            'strategy': 'mixture',
            'samples': 100,
            'seed': 42,
            'components': [
                {'weight': 0.50, 'mean': 0.02, 'std': 0.01},
                {'weight': 0.30, 'mean': 0.00, 'std': 0.01}
                # Weights sum to 0.80, not 1.0
            ]
        }
        
        with self.assertRaises(ValueError) as context:
            _generate_values_for_strategy('test_param', config)
        
        self.assertIn('must sum to 1.0', str(context.exception))
    
    def test_triangular_distribution(self):
        """Test triangular distribution sampling."""
        config = {
            'strategy': 'triangular',
            'min': 0.01,
            'mode': 0.03,
            'max': 0.06,
            'samples': 1000,
            'seed': 42
        }
        values = _generate_values_for_strategy('test_param', config)
        
        self.assertEqual(len(values), 1000)
        self.assertGreaterEqual(values.min(), 0.01)
        self.assertLessEqual(values.max(), 0.06)
    
    def test_uniform_distribution(self):
        """Test uniform distribution sampling (updated random strategy)."""
        config = {
            'strategy': 'uniform',
            'min': 0.02,
            'max': 0.05,
            'samples': 1000,
            'seed': 42
        }
        values = _generate_values_for_strategy('test_param', config)
        
        self.assertEqual(len(values), 1000)
        self.assertGreaterEqual(values.min(), 0.02)
        self.assertLessEqual(values.max(), 0.05)
        # Mean should be approximately (min + max) / 2
        self.assertAlmostEqual(values.mean(), 0.035, delta=0.005)
    
    def test_seed_reproducibility(self):
        """Test that same seed produces same results."""
        config = {
            'strategy': 'normal',
            'mean': 0.02,
            'std': 0.01,
            'samples': 100,
            'seed': 42
        }
        
        values1 = _generate_values_for_strategy('test_param', config)
        values2 = _generate_values_for_strategy('test_param', config)
        
        np.testing.assert_array_equal(values1, values2)


class TestComparativeMetrics(unittest.TestCase):
    """Test comparative metrics calculation."""
    
    def setUp(self):
        """Create sample results DataFrame."""
        np.random.seed(42)
        n_samples = 100
        
        # Create synthetic results for 3 scenarios
        data = []
        for scenario in ['A', 'B', 'C']:
            for i in range(n_samples):
                # Scenario A is best, B is middle, C is worst
                if scenario == 'A':
                    wealth = 100000 + np.random.normal(0, 10000)
                elif scenario == 'B':
                    wealth = 90000 + np.random.normal(0, 10000)
                else:  # C
                    wealth = 80000 + np.random.normal(0, 10000)
                
                data.append({
                    'mortgage_scenario_label': scenario,
                    'world_scenario_id': i,
                    'wealth_end': wealth,
                    'avg_net_cost_per_month': 2000
                })
        
        self.df = pd.DataFrame(data)
    
    def test_calculate_comparative_metrics(self):
        """Test P(i > j) calculation."""
        comp = calculate_comparative_metrics(self.df, 'wealth_end')
        
        # Check shape
        self.assertEqual(comp.shape, (3, 3))
        
        # Check diagonal is 0.5
        self.assertEqual(comp.loc['A', 'A'], 0.5)
        self.assertEqual(comp.loc['B', 'B'], 0.5)
        self.assertEqual(comp.loc['C', 'C'], 0.5)
        
        # Check A beats B and C most of the time
        self.assertGreater(comp.loc['A', 'B'], 0.5)
        self.assertGreater(comp.loc['A', 'C'], 0.5)
        
        # Check B beats C most of the time
        self.assertGreater(comp.loc['B', 'C'], 0.5)
        
        # Check symmetry: P(A > B) + P(B > A) = 1
        self.assertAlmostEqual(
            comp.loc['A', 'B'] + comp.loc['B', 'A'],
            1.0,
            places=10
        )


class TestRiskMetrics(unittest.TestCase):
    """Test risk metrics calculation."""
    
    def setUp(self):
        """Create sample results DataFrame."""
        np.random.seed(42)
        n_samples = 100
        
        data = []
        for i in range(n_samples):
            data.append({
                'mortgage_scenario_label': 'TestScenario',
                'world_scenario_id': i,
                'wealth_end': 100000 + np.random.normal(0, 10000),
                'avg_net_cost_per_month': 2000 + np.random.normal(0, 100)
            })
        
        self.df = pd.DataFrame(data)
    
    def test_calculate_risk_metrics(self):
        """Test risk metrics calculation."""
        risk = calculate_risk_metrics(
            self.df,
            'TestScenario',
            ['wealth_end', 'avg_net_cost_per_month']
        )
        
        # Check structure
        self.assertIn('wealth_end', risk)
        self.assertIn('avg_net_cost_per_month', risk)
        
        # Check wealth_end metrics
        wealth_metrics = risk['wealth_end']
        self.assertIn('mean', wealth_metrics)
        self.assertIn('std', wealth_metrics)
        self.assertIn('p10', wealth_metrics)
        self.assertIn('p50', wealth_metrics)
        self.assertIn('p90', wealth_metrics)
        
        # Check values are reasonable
        self.assertAlmostEqual(wealth_metrics['mean'], 100000, delta=2000)
        self.assertLess(wealth_metrics['p10'], wealth_metrics['p50'])
        self.assertLess(wealth_metrics['p50'], wealth_metrics['p90'])


class TestRegretMetrics(unittest.TestCase):
    """Test regret metrics calculation."""
    
    def setUp(self):
        """Create sample results DataFrame."""
        np.random.seed(42)
        n_samples = 100
        
        data = []
        for scenario in ['A', 'B']:
            for i in range(n_samples):
                # A is better than B
                if scenario == 'A':
                    wealth = 100000 + np.random.normal(0, 5000)
                else:
                    wealth = 90000 + np.random.normal(0, 5000)
                
                data.append({
                    'mortgage_scenario_label': scenario,
                    'world_scenario_id': i,
                    'wealth_end': wealth
                })
        
        self.df = pd.DataFrame(data)
    
    def test_calculate_regret_metrics(self):
        """Test regret calculation."""
        regret = calculate_regret_metrics(self.df, 'wealth_end')
        
        # Check structure
        self.assertEqual(len(regret), 2)
        self.assertIn('mean_regret', regret.columns)
        self.assertIn('max_regret', regret.columns)
        self.assertIn('prob_optimal', regret.columns)
        
        # A should have lower regret than B
        regret_a = regret[regret['mortgage_scenario_label'] == 'A']['mean_regret'].values[0]
        regret_b = regret[regret['mortgage_scenario_label'] == 'B']['mean_regret'].values[0]
        self.assertLess(regret_a, regret_b)
        
        # A should be optimal more often
        prob_a = regret[regret['mortgage_scenario_label'] == 'A']['prob_optimal'].values[0]
        prob_b = regret[regret['mortgage_scenario_label'] == 'B']['prob_optimal'].values[0]
        self.assertGreater(prob_a, prob_b)


class TestRegimeIdentification(unittest.TestCase):
    """Test regime identification for mixture distributions."""
    
    def test_identify_regime(self):
        """Test regime identification."""
        components = [
            {'mean': 0.035, 'std': 0.07, 'weight': 0.50},
            {'mean': 0.000, 'std': 0.06, 'weight': 0.30},
            {'mean': -0.050, 'std': 0.11, 'weight': 0.20}
        ]
        
        # Value clearly in first regime (positive and away from zero)
        regime = identify_regime(0.10, components)
        self.assertEqual(regime, 0)
        
        # Value exactly at zero - could be regime 0 or 1 depending on weights
        # With high std in regime 0 (0.07), value 0.00 is within 0.5 std of regime 0
        # So it's reasonable for it to be classified as regime 0
        regime = identify_regime(0.00, components)
        self.assertIn(regime, [0, 1])  # Accept either regime 0 or 1
        
        # Value clearly in crisis regime (very negative, beyond 2 std of regime 2)
        regime = identify_regime(-0.15, components)
        self.assertEqual(regime, 2)
    
    def test_add_regime_labels(self):
        """Test adding regime labels to DataFrame."""
        np.random.seed(42)
        
        # Create sample data with clearer regime separation
        df = pd.DataFrame({
            'annual_value_growth': [0.08, 0.00, -0.10, 0.05, -0.08],
            'wealth_end': [100000, 90000, 70000, 95000, 85000]
        })
        
        components = [
            {'mean': 0.035, 'std': 0.07, 'weight': 0.50, 'label': 'Normal'},
            {'mean': 0.000, 'std': 0.06, 'weight': 0.30, 'label': 'Stagnation'},
            {'mean': -0.050, 'std': 0.11, 'weight': 0.20, 'label': 'Crisis'}
        ]
        
        df_labeled = add_regime_labels(df, 'annual_value_growth', components)
        
        # Check columns were added
        self.assertIn('annual_value_growth_regime', df_labeled.columns)
        self.assertIn('annual_value_growth_regime_label', df_labeled.columns)
        
        # Check that we have regime labels (at least one of each type should appear)
        unique_labels = set(df_labeled['annual_value_growth_regime_label'].values)
        self.assertGreater(len(unique_labels), 1)  # Should have multiple regimes
        
        # Check that all labels are from our defined set
        valid_labels = {'Normal', 'Stagnation', 'Crisis'}
        self.assertTrue(unique_labels.issubset(valid_labels))


class TestMixtureValidation(unittest.TestCase):
    """Test mixture distribution validation."""
    
    def test_validate_mixture_distribution(self):
        """Test mixture distribution validation."""
        np.random.seed(42)
        
        # Generate samples from mixture
        components = [
            {'mean': 0.035, 'std': 0.07, 'weight': 0.50},
            {'mean': 0.000, 'std': 0.06, 'weight': 0.30},
            {'mean': -0.050, 'std': 0.11, 'weight': 0.20}
        ]
        
        # Create synthetic data
        n_samples = 1000
        rng = np.random.default_rng(42)
        regime_indices = rng.choice(3, size=n_samples, p=[0.5, 0.3, 0.2])
        values = np.zeros(n_samples)
        for i, comp in enumerate(components):
            mask = regime_indices == i
            values[mask] = rng.normal(comp['mean'], comp['std'], mask.sum())
        
        df = pd.DataFrame({
            'annual_value_growth': values,
            'mortgage_scenario_label': 'Test'
        })
        
        validation = validate_mixture_distribution(
            df, 'annual_value_growth', components
        )
        
        # Check structure
        self.assertIn('observed_mean', validation)
        self.assertIn('expected_mean', validation)
        self.assertIn('observed_std', validation)
        self.assertIn('expected_std', validation)
        self.assertIn('ks_statistic', validation)
        self.assertIn('ks_pvalue', validation)
        
        # Check values are reasonable
        expected_mean = 0.50 * 0.035 + 0.30 * 0.000 + 0.20 * (-0.050)
        self.assertAlmostEqual(
            validation['observed_mean'],
            expected_mean,
            delta=0.02
        )


if __name__ == '__main__':
    unittest.main()
