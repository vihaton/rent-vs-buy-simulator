# Simulate Renting vs Buying a Flat

A Python toolkit for financial analysis of renting vs buying real estate, with support for Dutch tax calculations and sensitivity analysis.

## Features

- **Scenario Comparison**: Compare renting vs buying scenarios with detailed financial projections
- **Dutch Tax Support**: Built-in support for Dutch home ownership tax calculations (2026)
- **Sensitivity Analysis**: Explore parameter spaces to find optimal buying scenarios
- **Flexible Configuration**: YAML-based scenario definitions
- **CLI Tools**: Command-line interfaces for batch processing
- **Programmatic API**: Import and use functions in scripts or notebooks

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd rent-vs-buy-simulator

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Scenario Comparison

Create a scenario file (e.g., `scenarios/my_scenario.yaml`):

Use in Python:

```python
import yaml
from src.estate import RentingScenario, BuyingScenario, evaluate_renting, evaluate_buying
from src.tax import NLHomeTax2026

# Load scenario
with open('scenarios/my_scenario.yaml') as f:
    config = yaml.safe_load(f)

# Evaluate renting
renting = RentingScenario(**config['renting'])
rent_result = evaluate_renting(renting)
print(f"Renting wealth end: €{rent_result['wealth_end']:,.0f}")

# Evaluate buying
tax = NLHomeTax2026(**config['tax'])
buying = BuyingScenario(**config['buying'], tax=tax)
buy_result = evaluate_buying(buying)
print(f"Buying wealth end: €{buy_result['wealth_end']:,.0f}")
```

## Sensitivity Analysis

Sensitivity analysis allows you to explore how different parameters affect buying scenarios. This is useful for:

- **Budget Discovery**: "What purchase price can I afford with my monthly budget?"
- **Cost Impact Analysis**: "How do VVE costs affect my finances?"
- **Financing Optimization**: "What mortgage structure minimizes costs?"
- **Multi-dimensional Exploration**: Vary multiple parameters simultaneously

### CLI Usage

Create a sensitivity configuration file (e.g., `scenarios/sensitivity/case01.yaml`):

Run the analysis:

```bash
# Basic usage
python -m src.cli_sensitivity --config scenarios/sensitivity/case01.yaml

# With verbose output
python -m src.cli_sensitivity --config scenarios/sensitivity/case01.yaml --verbose

# Override output format
python -m src.cli_sensitivity --config scenarios/sensitivity/case01.yaml --format parquet

# Custom output directory
python -m src.cli_sensitivity --config scenarios/sensitivity/case01.yaml --output outputs/my_analysis
```

### Programmatic Usage

```python
from src.sensitivity import generate_parameter_space, run_sensitivity_analysis
from src.estate import BuyingScenario
from src.tax import NLHomeTax2026
import yaml

# Load base scenario
with open('scenarios/case06b3.yaml') as f:
    config = yaml.safe_load(f)

# Define parameter space
variables = {
    'purchase_price': {'strategy': 'linspace', 'min': 280000, 'max': 400000, 'steps': 50},
    'monthly_vve': {'strategy': 'linspace', 'min': 200, 'max': 500, 'steps': 30}
}

# Generate parameter combinations
param_space = generate_parameter_space(variables)
print(f"Total combinations: {len(param_space):,}")

# Run analysis
tax = NLHomeTax2026(**config['tax'])
base_scenario = BuyingScenario(**config['buying'], tax=tax)
results_df = run_sensitivity_analysis(base_scenario, param_space, tax, verbose=True)

# Filter for affordable scenarios
affordable = results_df[
    (results_df['cash_required_upfront'] <= 50000) &
    (results_df['avg_net_cost_per_month'] <= 2500)
]

print(f"Affordable scenarios: {len(affordable)} / {len(results_df)}")
print(f"Max affordable price: €{affordable['purchase_price'].max():,.0f}")
```

### Supported Strategies

| Strategy | Description | Required Fields | Example |
|----------|-------------|-----------------|---------|
| `linspace` | Evenly spaced values | min, max, steps | `{'strategy': 'linspace', 'min': 300000, 'max': 400000, 'steps': 20}` |
| `logspace` | Logarithmically spaced | min, max, steps | `{'strategy': 'logspace', 'min': 1, 'max': 1000, 'steps': 10}` |
| `random` | Random uniform sampling | min, max, samples | `{'strategy': 'random', 'min': 0.03, 'max': 0.05, 'samples': 100}` |
| `values` | Explicit value list | values | `{'strategy': 'values', 'values': [20, 25, 30]}` |

### Variable Parameters

All parameters from `BuyingScenario` can be varied:

**Property:**
- `purchase_price`
- `annual_value_growth`

**Financing:**
- `mortgage_principal`
- `mortgage_annual_rate`
- `mortgage_term_years`

**Costs:**
- `one_off_costs`
- `renovation_costs_once`
- `monthly_vve`
- `monthly_utilities`

**Room Rental:**
- `monthly_rent_income`
- `months_rented`

**Exit:**
- `sold_at_end`
- `selling_cost_rate`

### Output Columns

Results include all input parameters plus evaluation metrics:

- `total_spent`, `total_income`, `net_cashflow`
- `wealth_end`, `avg_net_cost_per_month`
- `cash_required_upfront`, `down_payment`
- `mortgage_monthly_payment`, `mortgage_interest_paid`
- `home_value_end`, `equity_end_if_not_sold`
- And more...


## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/unit/test_sensitivity.py
```

## TODO

- rent increases
- define when apartment is rented out (matters for compounding interest)
- allow renting out and living simultaneously/separately
- support subletting for the rental scenarios
