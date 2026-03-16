"""
Markov Chain Regime Simulation Module

This module provides a Markov chain framework for simulating economic regime
transitions over time. It supports:
- 4-state regime model (NORMAL_GROWTH, STAGNATION, CRISIS, RECOVERY)
- Configurable transition matrices
- Parameter sampling conditional on regime
- Shared regime paths for fair scenario comparison
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import IntEnum
import numpy as np
import yaml
from pathlib import Path


class RegimeState(IntEnum):
    """Economic regime states."""
    NORMAL_GROWTH = 0
    STAGNATION = 1
    CRISIS = 2
    RECOVERY = 3


REGIME_LABELS = {
    RegimeState.NORMAL_GROWTH: "Normal Growth",
    RegimeState.STAGNATION: "Stagnation",
    RegimeState.CRISIS: "Crisis",
    RegimeState.RECOVERY: "Recovery"
}


@dataclass
class MacroRegime:
    """
    Defines parameter distributions for a single economic regime.
    
    Attributes:
        property_growth_mean: Mean annual property value growth rate
        property_growth_std: Standard deviation of property growth
        mortgage_rate_mean: Mean mortgage interest rate
        mortgage_rate_std: Standard deviation of mortgage rate
    """
    property_growth_mean: float
    property_growth_std: float
    mortgage_rate_mean: float
    mortgage_rate_std: float


@dataclass
class MarkovChainConfig:
    """
    Configuration for Markov chain regime transitions.
    
    Attributes:
        regimes: List of regime names (must match RegimeState enum)
        transition_matrix: 4x4 matrix of transition probabilities
        initial_state_probs: Initial state distribution (length 4)
        time_step_years: Years per transition (typically 1)
        regime_distributions: Dict mapping regime names to MacroRegime objects
    """
    regimes: List[str]
    transition_matrix: np.ndarray
    initial_state_probs: np.ndarray
    time_step_years: int
    regime_distributions: Dict[str, MacroRegime]
    
    def __post_init__(self):
        """Validate configuration."""
        # Validate transition matrix
        if self.transition_matrix.shape != (4, 4):
            raise ValueError(f"Transition matrix must be 4x4, got {self.transition_matrix.shape}")
        
        # Check rows sum to 1 (with tolerance for floating-point rounding)
        row_sums = self.transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-3):
            raise ValueError(f"Transition matrix rows must sum to 1, got {row_sums}")
        
        # Validate initial state probs
        if len(self.initial_state_probs) != 4:
            raise ValueError(f"Initial state probs must have length 4, got {len(self.initial_state_probs)}")
        
        if not np.isclose(self.initial_state_probs.sum(), 1.0, atol=1e-3):
            raise ValueError(f"Initial state probs must sum to 1, got {self.initial_state_probs.sum()}")
        
        # Validate regimes
        if len(self.regimes) != 4:
            raise ValueError(f"Must have exactly 4 regimes, got {len(self.regimes)}")
    
    @classmethod
    def from_yaml(cls, config_path: str, distributions_path: str) -> 'MarkovChainConfig':
        """
        Load configuration from YAML files.
        
        Args:
            config_path: Path to transition matrix config (e.g., default_regimes.yaml)
            distributions_path: Path to regime distributions (e.g., regime_distributions.yaml)
        
        Returns:
            MarkovChainConfig instance
        """
        # Load transition matrix config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        regimes = config['regimes']
        transition_matrix = np.array(config['transition_matrix'])
        initial_state_probs = np.array(config['initial_state_probs'])
        time_step_years = config.get('time_step_years', 1)
        
        # Load regime distributions
        with open(distributions_path, 'r') as f:
            dist_config = yaml.safe_load(f)
        
        regime_distributions = {}
        for regime_name in regimes:
            dist = dist_config['distributions'][regime_name]
            regime_distributions[regime_name] = MacroRegime(
                property_growth_mean=dist['property_growth']['mean'],
                property_growth_std=dist['property_growth']['std'],
                mortgage_rate_mean=dist['mortgage_rate']['mean'],
                mortgage_rate_std=dist['mortgage_rate']['std']
            )
        
        return cls(
            regimes=regimes,
            transition_matrix=transition_matrix,
            initial_state_probs=initial_state_probs,
            time_step_years=time_step_years,
            regime_distributions=regime_distributions
        )
    
    def with_start_regime(self, regime_name: str) -> 'MarkovChainConfig':
        """
        Create a new config with deterministic starting regime.
        
        This method returns a new MarkovChainConfig instance with initial_state_probs
        set to deterministically start in the specified regime (probability = 1.0 for
        that regime, 0.0 for all others).
        
        Args:
            regime_name: Name of starting regime (e.g., "CRISIS", "NORMAL_GROWTH")
        
        Returns:
            New MarkovChainConfig with deterministic initial state
        
        Raises:
            ValueError: If regime_name is not in the list of valid regimes
        
        Example:
            >>> config = MarkovChainConfig.from_yaml(...)
            >>> crisis_config = config.with_start_regime("CRISIS")
            >>> # All simulations will now start in CRISIS regime
        """
        if regime_name not in self.regimes:
            raise ValueError(
                f"Unknown regime: '{regime_name}'. "
                f"Must be one of: {', '.join(self.regimes)}"
            )
        
        # Find index of specified regime
        regime_idx = self.regimes.index(regime_name)
        
        # Create deterministic initial state probabilities
        new_initial_probs = np.zeros(4)
        new_initial_probs[regime_idx] = 1.0
        
        # Create new instance with updated initial probs
        # (all other fields remain the same)
        return MarkovChainConfig(
            regimes=self.regimes,
            transition_matrix=self.transition_matrix.copy(),
            initial_state_probs=new_initial_probs,
            time_step_years=self.time_step_years,
            regime_distributions=self.regime_distributions
        )
    
    def get_expected_duration(self, regime: RegimeState) -> float:
        """
        Calculate expected duration in a regime.
        
        Expected duration = 1 / (1 - self_transition_prob)
        
        Args:
            regime: Regime state
        
        Returns:
            Expected number of years in regime
        """
        self_prob = self.transition_matrix[regime, regime]
        return self.time_step_years / (1 - self_prob)


class MarkovRegimeSimulator:
    """
    Simulator for Markov chain regime transitions.
    
    This class generates regime paths and samples parameters conditional
    on the current regime.
    """
    
    def __init__(self, config: MarkovChainConfig, seed: Optional[int] = None):
        """
        Initialize simulator.
        
        Args:
            config: Markov chain configuration
            seed: Random seed for reproducibility
        """
        self.config = config
        self.rng = np.random.default_rng(seed)
    
    def simulate_regime_path(self, horizon_years: int) -> np.ndarray:
        """
        Simulate a single regime path.
        
        Args:
            horizon_years: Number of years to simulate
        
        Returns:
            Array of regime states (length horizon_years + 1, includes year 0)
        """
        n_steps = int(horizon_years // self.config.time_step_years + 1)
        path = np.zeros(n_steps, dtype=int)
        
        # Sample initial state (normalize to ensure sum is exactly 1.0)
        initial_probs = self.config.initial_state_probs / self.config.initial_state_probs.sum()
        path[0] = self.rng.choice(4, p=initial_probs)
        
        # Simulate transitions
        for t in range(1, n_steps):
            current_state = path[t - 1]
            transition_probs = self.config.transition_matrix[current_state]
            # Normalize to ensure sum is exactly 1.0 for numpy.random.choice
            transition_probs = transition_probs / transition_probs.sum()
            path[t] = self.rng.choice(4, p=transition_probs)
        
        return path
    
    def simulate_regime_paths(self, n_samples: int, horizon_years: int) -> np.ndarray:
        """
        Simulate multiple regime paths.
        
        Args:
            n_samples: Number of paths to simulate
            horizon_years: Number of years per path
        
        Returns:
            Array of shape (n_samples, n_steps) with regime states
        """
        n_steps = int(horizon_years // self.config.time_step_years + 1)
        paths = np.zeros((n_samples, n_steps), dtype=int)
        
        for i in range(n_samples):
            paths[i] = self.simulate_regime_path(horizon_years)
        
        return paths
    
    def sample_parameters(
        self,
        regime_path: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample parameters conditional on regime path.
        
        Args:
            regime_path: Array of regime states (length n_steps)
        
        Returns:
            Tuple of (property_growth_rates, mortgage_rates), each of length n_steps
        """
        n_steps = len(regime_path)
        property_growth = np.zeros(n_steps)
        mortgage_rates = np.zeros(n_steps)
        
        for t in range(n_steps):
            regime_idx = regime_path[t]
            regime_name = self.config.regimes[regime_idx]
            regime_dist = self.config.regime_distributions[regime_name]
            
            # Sample property growth
            property_growth[t] = self.rng.normal(
                regime_dist.property_growth_mean,
                regime_dist.property_growth_std
            )
            
            # Sample mortgage rate
            mortgage_rates[t] = self.rng.normal(
                regime_dist.mortgage_rate_mean,
                regime_dist.mortgage_rate_std
            )
            
            # Ensure mortgage rate is positive
            mortgage_rates[t] = max(0.001, mortgage_rates[t])
        
        return property_growth, mortgage_rates
    
    def sample_parameters_batch(
        self,
        regime_paths: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample parameters for multiple regime paths.
        
        Args:
            regime_paths: Array of shape (n_samples, n_steps)
        
        Returns:
            Tuple of (property_growth_rates, mortgage_rates),
            each of shape (n_samples, n_steps)
        """
        n_samples, n_steps = regime_paths.shape
        property_growth = np.zeros((n_samples, n_steps))
        mortgage_rates = np.zeros((n_samples, n_steps))
        
        for i in range(n_samples):
            property_growth[i], mortgage_rates[i] = self.sample_parameters(regime_paths[i])
        
        return property_growth, mortgage_rates


def generate_shared_regime_paths(
    config: MarkovChainConfig,
    n_samples: int,
    horizon_years: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate regime paths to be shared across all scenarios.
    
    This ensures fair comparison: all scenarios experience the same
    economic conditions in each Monte Carlo sample.
    
    Args:
        config: Markov chain configuration
        n_samples: Number of paths to generate
        horizon_years: Number of years per path
        seed: Random seed
    
    Returns:
        Array of shape (n_samples, n_steps) with regime states
    """
    simulator = MarkovRegimeSimulator(config, seed=seed)
    return simulator.simulate_regime_paths(n_samples, horizon_years)


def generate_shared_parameter_draws(
    regime_paths: np.ndarray,
    config: MarkovChainConfig,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate parameter draws for shared regime paths.
    
    Args:
        regime_paths: Array of shape (n_samples, n_steps)
        config: Markov chain configuration
        seed: Random seed
    
    Returns:
        Dictionary with keys:
            - 'property_growth': Array of shape (n_samples, n_steps)
            - 'mortgage_rates': Array of shape (n_samples, n_steps)
            - 'regime_paths': Copy of input regime_paths
    """
    simulator = MarkovRegimeSimulator(config, seed=seed)
    property_growth, mortgage_rates = simulator.sample_parameters_batch(regime_paths)
    
    return {
        'property_growth': property_growth,
        'mortgage_rates': mortgage_rates,
        'regime_paths': regime_paths.copy()
    }
