"""
Unit tests for Markov regime start regime functionality.
"""

import unittest
import numpy as np
from src.markov_regime import MarkovChainConfig, MacroRegime


class TestMarkovStartRegime(unittest.TestCase):
    """Test the with_start_regime method of MarkovChainConfig."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.regimes = ['NORMAL_GROWTH', 'STAGNATION', 'CRISIS', 'RECOVERY']
        self.transition_matrix = np.array([
            [0.75, 0.18, 0.03, 0.04],
            [0.35, 0.45, 0.12, 0.08],
            [0.10, 0.20, 0.50, 0.20],
            [0.55, 0.20, 0.05, 0.20]
        ])
        self.initial_state_probs = np.array([0.50, 0.30, 0.10, 0.10])
        
        # Create minimal regime distributions
        self.regime_distributions = {
            regime: MacroRegime(
                property_growth_mean=0.03,
                property_growth_std=0.07,
                mortgage_rate_mean=0.04,
                mortgage_rate_std=0.01
            )
            for regime in self.regimes
        }
        
        self.config = MarkovChainConfig(
            regimes=self.regimes,
            transition_matrix=self.transition_matrix,
            initial_state_probs=self.initial_state_probs,
            time_step_years=1,
            regime_distributions=self.regime_distributions
        )
    
    def test_with_start_regime_normal_growth(self):
        """Test starting in NORMAL_GROWTH regime."""
        new_config = self.config.with_start_regime('NORMAL_GROWTH')
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(new_config.initial_state_probs, expected)
    
    def test_with_start_regime_stagnation(self):
        """Test starting in STAGNATION regime."""
        new_config = self.config.with_start_regime('STAGNATION')
        expected = np.array([0.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(new_config.initial_state_probs, expected)
    
    def test_with_start_regime_crisis(self):
        """Test starting in CRISIS regime."""
        new_config = self.config.with_start_regime('CRISIS')
        expected = np.array([0.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(new_config.initial_state_probs, expected)
    
    def test_with_start_regime_recovery(self):
        """Test starting in RECOVERY regime."""
        new_config = self.config.with_start_regime('RECOVERY')
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(new_config.initial_state_probs, expected)
    
    def test_with_start_regime_invalid(self):
        """Test that invalid regime name raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.config.with_start_regime('INVALID_REGIME')
        
        self.assertIn('Unknown regime', str(context.exception))
        self.assertIn('INVALID_REGIME', str(context.exception))
    
    def test_with_start_regime_preserves_other_fields(self):
        """Test that with_start_regime preserves other config fields."""
        new_config = self.config.with_start_regime('CRISIS')
        
        # Check that other fields are preserved
        self.assertEqual(new_config.regimes, self.config.regimes)
        self.assertEqual(new_config.time_step_years, self.config.time_step_years)
        np.testing.assert_array_equal(new_config.transition_matrix, self.config.transition_matrix)
        self.assertEqual(new_config.regime_distributions, self.config.regime_distributions)
    
    def test_with_start_regime_does_not_modify_original(self):
        """Test that with_start_regime does not modify the original config."""
        original_probs = self.config.initial_state_probs.copy()
        new_config = self.config.with_start_regime('CRISIS')
        
        # Original should be unchanged
        np.testing.assert_array_equal(self.config.initial_state_probs, original_probs)
        
        # New config should be different
        self.assertFalse(np.array_equal(new_config.initial_state_probs, original_probs))
    
    def test_with_start_regime_probabilities_sum_to_one(self):
        """Test that new initial_state_probs sum to 1.0."""
        for regime in self.regimes:
            new_config = self.config.with_start_regime(regime)
            self.assertAlmostEqual(new_config.initial_state_probs.sum(), 1.0)


if __name__ == '__main__':
    unittest.main()
