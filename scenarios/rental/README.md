# Rental Scenario Configurations

This directory contains standalone rental scenario configurations for use as baselines in rent vs. buy comparisons.

## File Format

Rental scenario YAML files should contain the following fields:

```yaml
name: "Rental Baseline - Location"
rent_monthly: 1950              # Monthly rent in euros
utilities_monthly: 250          # Monthly utilities in euros
living_months: 360              # Total duration (e.g., 360 = 30 years)
annual_rent_increase: 0.011     # Annual rent increase rate (e.g., 0.011 = 1.1%)
description: "Property description"  # Optional
```

## Usage

Use rental scenarios as baselines in time-stepping comparisons:

```bash
python -m src.cli_time_stepping_comparison \
  --scenarios scenarios/case09_*.yaml \
  --rental-baseline scenarios/rental/rental_baseline.yaml \
  --n-samples 5000 \
  --seed 42
```

## Payment Terminology

**Important**: Rental scenarios track total housing costs:
- `monthly_payment` = rent + utilities
- `monthly_net_housing_cost` = rent + utilities (same as monthly_payment)

This enables fair "apples-to-apples" comparison with buying scenarios where:
- `monthly_payment` = mortgage payment only (principal + interest)
- `monthly_net_housing_cost` = mortgage + VVE + utilities - rent_income

## Rent Increase Modeling

The `annual_rent_increase` parameter models realistic rent growth over time:
- **0.0**: No rent increases (constant rent)
- **0.011**: 1.1% annual increase (typical Dutch market)
- **0.02**: 2% annual increase (higher inflation scenario)

Rent increases are applied at the start of each year and are **deterministic** (not affected by macro-economic regimes). This is a simplification but acceptable for baseline comparison purposes.

## Example Scenarios

- `rental_baseline.yaml`: Amsterdam rental (€1,950/month + €250 utilities, 1.1% annual increase)

## Notes

- Rental baselines are calculated once and replicated across all Monte Carlo samples
- No equity is built (equity = 0 always)
- Final wealth equals negative cumulative cashflow (all money spent)
- ROI = -1.0 (spent everything, got nothing back)
- Cash efficiency = 0.0 (no equity built)
