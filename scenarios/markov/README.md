# Markov Regime Transition Matrices

This directory contains calibrated transition matrices for economic regime modeling in the rent-vs-buy simulator.

## Files

### [`calibrated_regimes.yaml`](calibrated_regimes.yaml)

The main calibrated transition matrix combining empirical data with expert priors.

**Key Features:**
- **Data-driven**: 70% weight on empirical estimates from BIS property price data
- **Prior-informed**: 30% weight on expert judgment to handle limited data
- **Validated**: Matrix is ergodic (all states reachable) with stable stationary distribution
- **Countries**: Netherlands, Germany, Belgium, France, Austria, Denmark, Sweden, Finland
- **Period**: 2023-Q2 to 2025-Q3 (64 quarterly transitions observed)

**Regime Statistics:**
- **Expected durations**: 1.5-1.8 years per regime
- **Stationary distribution**: Reflects recent European property market volatility

### [`empirical_matrix.yaml`](empirical_matrix.yaml)

Pure empirical estimates without prior smoothing (for comparison).

**Note**: This matrix has higher variance due to limited data and should not be used directly for simulations.

## Regime Definitions

Based on annualized property price growth rates:

| Regime        | Annual Growth Rate | Interpretation                          |
|---------------|--------------------|-----------------------------------------|
| RECOVERY      | > 5%               | Strong property price appreciation      |
| NORMAL_GROWTH | 2% to 5%           | Healthy, sustainable growth             |
| STAGNATION    | -2% to 2%          | Flat or minimal price movement          |
| CRISIS        | < -2%              | Property price decline                  |

## Calibration Methodology

### Data Source
- **Provider**: Bank for International Settlements (BIS)
- **Dataset**: Real Residential Property Prices Index
- **Frequency**: Quarterly
- **Base**: Index = 100 (reference period varies by country)

### Process

1. **Extract Data**: Load BIS property price indices for selected countries
2. **Calculate Growth**: Compute quarter-over-quarter growth rates
3. **Annualize**: Convert quarterly to annual growth rates: `(1 + qoq)^4 - 1`
4. **Classify Regimes**: Apply threshold rules to assign regime labels
5. **Count Transitions**: Tally observed regime changes
6. **Estimate Matrix**: Normalize transition counts to probabilities
7. **Smooth with Prior**: Bayesian combination with expert beliefs
8. **Validate**: Check ergodicity and stochastic properties

### Bayesian Smoothing

```
Calibrated = α × Prior + (1 - α) × Empirical
```

Where:
- `α = 0.3` (30% weight on prior)
- Prior matrix from [`plans/reasoning_for_priors.md`](../../plans/reasoning_for_priors.md)
- Empirical matrix from BIS data

**Rationale**: Limited historical data (9 quarters) requires prior beliefs to stabilize estimates, especially for rare transitions (e.g., CRISIS → RECOVERY).

## Usage

### In Python

```python
import yaml
import numpy as np

# Load calibrated matrix
with open('scenarios/markov/calibrated_regimes.yaml') as f:
    config = yaml.safe_load(f)

regimes = config['regimes']
transition_matrix = np.array(config['transition_matrix'])
initial_probs = config['initial_state_probs']

# Simulate regime path
current_state = np.random.choice(len(regimes), p=initial_probs)
for year in range(30):
    next_state = np.random.choice(len(regimes), p=transition_matrix[current_state])
    print(f"Year {year}: {regimes[next_state]}")
    current_state = next_state
```

### Re-calibration

To update the matrix with new data:

```bash
python scripts/calibrate_transition_matrix.py
```

This will:
1. Load the latest BIS data from [`data/bis_dp_custom_table_export_20260314-174549.csv`](../../data/bis_dp_custom_table_export_20260314-174549.csv)
2. Re-estimate transition probabilities
3. Smooth with priors
4. Validate and save updated matrices

## Limitations and Caveats

### Data Limitations

1. **Short time series**: Only 2023-2025 data (9 quarters)
   - Not enough to observe full economic cycle
   - May miss rare events (major crises)
   - **Mitigation**: Heavy reliance on prior beliefs (α = 0.3)

2. **Recent boom period**: 2023-2025 was primarily recovery/growth
   - No major crisis observed in this period
   - Crisis transition probabilities rely heavily on prior
   - **Mitigation**: Include longer historical data when available

3. **Quarterly frequency**: Transitions estimated at quarterly intervals
   - Need to adjust for annual time steps in simulations
   - **Current approach**: Use annual time steps (time_step_years = 1)

### Methodological Limitations

1. **Regime classification**: Thresholds are somewhat arbitrary
   - 2% vs 3% for NORMAL/STAGNATION boundary could be debated
   - **Mitigation**: Sensitivity analysis on thresholds (future work)

2. **Homogeneity assumption**: Assumes constant transition probabilities
   - Reality: May change with policy regime, structural changes
   - **Mitigation**: Re-calibrate periodically with new data

3. **Markov property**: Assumes memoryless transitions
   - Reality: Regimes may have momentum or memory
   - **Mitigation**: Consider higher-order Markov chains in future versions

## Validation

The calibrated matrix has been validated for:

- ✓ **Probability constraints**: All elements in [0, 1]
- ✓ **Stochastic property**: Rows sum to 1
- ✓ **Ergodicity**: All states reachable from any starting state
- ✓ **Stability**: Stationary distribution exists and is unique

## Future Improvements

1. **Longer time series**: Extend back to 2010 or 2000 to include 2008 crisis
2. **Additional indicators**: Incorporate GDP growth, unemployment, interest rates
3. **Regime-switching models**: Hamilton (1989) approach for joint estimation
4. **Country-specific matrices**: Separate calibrations for different markets
5. **Time-varying transitions**: Allow probabilities to evolve with macro conditions

## References

- Bank for International Settlements (BIS) - [Property Price Statistics](https://www.bis.org/statistics/pp.htm)
- Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle"
- [`plans/transition_matrix_calibration.md`](../../plans/transition_matrix_calibration.md) - Detailed methodology
- [`plans/reasoning_for_priors.md`](../../plans/reasoning_for_priors.md) - Prior distributions

## Contact

For questions about the calibration methodology or to suggest improvements, please refer to the implementation plan in [`plans/transition_matrix_calibration.md`](../../plans/transition_matrix_calibration.md).
