# Probabilistic Mortgage Analysis

This directory contains configuration files for probabilistic mortgage comparison using Monte Carlo simulation.

## Overview

The probabilistic analysis framework allows you to:
- Compare multiple mortgage scenarios under shared uncertainty
- Model realistic uncertainty using mixture distributions (regime shifts)
- Calculate comparative metrics like P(mortgage_A > mortgage_B)
- Identify worst-case and best-case scenarios
- Analyze performance by regime (normal/stagnation/crisis)

## Quick Start

### 1. Simple Comparison with Normal Distributions

```bash
python -m src.cli_probabilistic \
  --scenarios \
    scenarios/mortgage_scenario_a.yaml \
    scenarios/mortgage_scenario_b.yaml \
    scenarios/mortgage_scenario_c.yaml \
  --variables scenarios/probabilistic/shared_variables_simple.yaml \
  --n-samples 1000 \
  --seed 42 \
  --verbose
```

### 2. Realistic Comparison with Mixture Distributions

```bash
python -m src.cli_probabilistic \
  --scenarios \
    scenarios/mortgage_scenario_a.yaml \
    scenarios/mortgage_scenario_b.yaml \
    scenarios/mortgage_scenario_c.yaml \
  --variables scenarios/probabilistic/shared_mixture_variables.yaml \
  --verbose
```

## Configuration Files

### `shared_variables_simple.yaml`

Simple configuration using normal distributions:
- **Property growth**: Normal(mean=2%, std=1.5%)
- **Interest rate reset**: Normal(mean=4%, std=1%)
- **Samples**: 1,000

Good for quick analysis and testing.

### `shared_mixture_variables.yaml`

Realistic configuration using 3-regime mixture distributions:

**Property Growth** (3 regimes):
- Normal growth (50%): 3.5% ± 7%
- Stagnation (30%): 0% ± 6%
- Crisis (20%): -5% ± 11%

**Interest Rate Reset** (3 regimes):
- Low-rate (40%): 3.0% ± 0.9%
- Normal (35%): 4.1% ± 1.0%
- High-rate (25%): 6.0% ± 1.4%

**Samples**: 5,000

Based on reasoning in [`plans/reasoning_for_priors.md`](../../plans/reasoning_for_priors.md).

## Analysis Examples

### Load and Analyze Results

```python
import pandas as pd
from src.probabilistic_analysis import (
    calculate_comparative_metrics,
    calculate_risk_metrics,
    calculate_regret_metrics,
    add_regime_labels
)

# Load results
df = pd.read_csv("outputs/probabilistic/comparison_20260314_143000.csv")

# 1. Compare scenarios: P(i > j)
comp = calculate_comparative_metrics(df, "wealth_end")
print("\nProbability of outperformance:")
print(comp)
# Example output:
#                Scenario_A  Scenario_B  Scenario_C
# Scenario_A           0.50        0.62        0.58
# Scenario_B           0.38        0.50        0.45
# Scenario_C           0.42        0.55        0.50

# 2. Risk analysis for specific scenario
risk = calculate_risk_metrics(df, "Scenario_A", ["wealth_end", "avg_net_cost_per_month"])
print(f"\nScenario A Risk Profile:")
print(f"  Expected wealth: €{risk['wealth_end']['mean']:,.0f}")
print(f"  Downside (P10): €{risk['wealth_end']['p10']:,.0f}")
print(f"  Upside (P90): €{risk['wealth_end']['p90']:,.0f}")

# 3. Regret analysis
regret = calculate_regret_metrics(df, "wealth_end")
print("\nRegret Analysis:")
print(regret)
# Shows average opportunity cost of each mortgage choice

# 4. Identify regimes (for mixture distributions)
property_components = [
    {'mean': 0.035, 'std': 0.07, 'weight': 0.50, 'label': 'Normal growth'},
    {'mean': 0.000, 'std': 0.06, 'weight': 0.30, 'label': 'Stagnation'},
    {'mean': -0.050, 'std': 0.11, 'weight': 0.20, 'label': 'Crisis'}
]

df = add_regime_labels(df, 'annual_value_growth', property_components)

# Analyze by regime
print("\nPerformance by property regime:")
print(df.groupby(['property_regime_label', 'mortgage_scenario_label'])['wealth_end'].describe())

# Crisis scenarios only
crisis_df = df[df['annual_value_growth_regime_label'] == 'Crisis']
print(f"\nCrisis scenarios ({len(crisis_df)}):")
print(crisis_df.groupby('mortgage_scenario_label')['wealth_end'].describe())
```

### Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df = pd.read_csv("outputs/probabilistic/comparison_20260314_143000.csv")

# 1. Distribution comparison
fig, ax = plt.subplots(figsize=(12, 6))
for scenario in df['mortgage_scenario_label'].unique():
    scenario_df = df[df['mortgage_scenario_label'] == scenario]
    ax.hist(scenario_df['wealth_end'], bins=50, alpha=0.5, label=scenario)
ax.set_xlabel('Wealth at End (€)')
ax.set_ylabel('Frequency')
ax.set_title('Wealth Distribution by Mortgage Scenario')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/probabilistic/wealth_distributions.png')

# 2. Box plot comparison
fig, ax = plt.subplots(figsize=(10, 6))
df.boxplot(column='wealth_end', by='mortgage_scenario_label', ax=ax)
ax.set_xlabel('Mortgage Scenario')
ax.set_ylabel('Wealth at End (€)')
ax.set_title('Wealth Distribution Comparison')
plt.suptitle('')  # Remove default title
plt.tight_layout()
plt.savefig('outputs/probabilistic/wealth_boxplot.png')

# 3. Scatter plot: property growth vs wealth
fig, ax = plt.subplots(figsize=(12, 6))
for scenario in df['mortgage_scenario_label'].unique():
    scenario_df = df[df['mortgage_scenario_label'] == scenario]
    ax.scatter(scenario_df['annual_value_growth'], 
               scenario_df['wealth_end'], 
               alpha=0.3, label=scenario)
ax.set_xlabel('Annual Property Growth Rate')
ax.set_ylabel('Wealth at End (€)')
ax.set_title('Wealth vs Property Growth by Scenario')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/probabilistic/wealth_vs_growth.png')
```

## Creating Custom Variables

### Normal Distribution

```yaml
variables:
  your_parameter:
    strategy: "normal"
    mean: 0.02       # Expected value
    std: 0.01        # Standard deviation
    samples: 5000    # Number of samples
    seed: 42         # Random seed
    min: -0.05       # Optional: truncate lower bound
    max: 0.10        # Optional: truncate upper bound
```

### Mixture Distribution (3 regimes)

```yaml
variables:
  your_parameter:
    strategy: "mixture"
    samples: 5000
    seed: 42
    components:
      - weight: 0.50      # Probability of regime 1
        mean: 0.03
        std: 0.01
        label: "Regime 1"  # Optional
      - weight: 0.30      # Probability of regime 2
        mean: 0.00
        std: 0.01
        label: "Regime 2"
      - weight: 0.20      # Probability of regime 3
        mean: -0.02
        std: 0.02
        label: "Regime 3"
    min: -0.10       # Optional: truncate
    max: 0.10
```

### Triangular Distribution

```yaml
variables:
  your_parameter:
    strategy: "triangular"
    min: 0.01        # Minimum value
    mode: 0.03       # Most likely value
    max: 0.06        # Maximum value
    samples: 5000
    seed: 42
```

### Uniform Distribution

```yaml
variables:
  your_parameter:
    strategy: "uniform"
    min: 0.02        # Minimum value
    max: 0.05        # Maximum value
    samples: 5000
    seed: 42
```

## Parameter Paths

You can specify nested parameters using dot notation and array indexing:

```yaml
variables:
  # Simple parameter
  annual_value_growth:
    strategy: "normal"
    mean: 0.02
    std: 0.01
  
  # Nested parameter (first loan's rate reset)
  "mortgage_loans[0].rate_resets[0].new_rate_default":
    strategy: "normal"
    mean: 0.04
    std: 0.01
  
  # Wildcard (all loans' first rate reset)
  "mortgage_loans[*].rate_resets[0].new_rate_default":
    strategy: "mixture"
    components: [...]
```

## Output Files

Results are saved to `outputs/probabilistic/` with timestamp:
- `comparison_YYYYMMDD_HHMMSS.csv` - Full results with all parameters and metrics

Each row represents one evaluation with columns:
- Input parameters (e.g., `annual_value_growth`)
- Output metrics (e.g., `wealth_end`, `avg_net_cost_per_month`)
- `mortgage_scenario_label` - Name of mortgage scenario
- `scenario_file` - Path to scenario config
- `world_scenario_id` - Index of Monte Carlo sample (0 to n-1)

## Tips

1. **Start small**: Use 1,000 samples for testing, increase to 5,000+ for final analysis
2. **Use seeds**: Always specify `seed` for reproducibility
3. **Match samples**: All variables must have the same number of samples
4. **Check convergence**: Run with different sample sizes to verify stability
5. **Validate mixtures**: Use `validate_mixture_distribution()` to check sampling
6. **Analyze regimes**: Use `add_regime_labels()` to understand regime-specific performance

## See Also

- [`plans/probabilistic_architecture_v2.md`](../../plans/probabilistic_architecture_v2.md) - Architecture design
- [`plans/probabilistic_mixture_support.md`](../../plans/probabilistic_mixture_support.md) - Mixture distributions
- [`plans/reasoning_for_priors.md`](../../plans/reasoning_for_priors.md) - Prior reasoning
- [`src/probabilistic_analysis.py`](../../src/probabilistic_analysis.py) - Analysis functions
