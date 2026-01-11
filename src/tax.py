import math
from typing import List
from dataclasses import dataclass

@dataclass
class NLHomeTax2026:
    # Approximate WOZ basis (often close to market but not identical)
    woz_value: float

    # Eigenwoningforfait:
    # For most homes: 0.35% in 2026
    ewf_rate_main: float = 0.0035

    # "Villa tax" add-on above threshold (optional; you can ignore if not relevant)
    villa_threshold: float = 1_350_000.0
    ewf_rate_above_threshold: float = 0.0235  # applied only to part above threshold (simplified)

    # Mortgage interest deduction cap (max rate you effectively get back)
    # 2026: 37.56% cap
    deduction_cap_rate: float = 0.3756

    # Your marginal tax rate in Box 1 (approx):
    # - If you want simplicity: use something like 0.3756
    # - With 30% ruling etc, you may set a different effective marginal.
    marginal_tax_rate: float = 0.3756

    # Wet Hillen (optional):
    apply_wet_hillen: bool = True
    wet_hillen_factor_2026: float = 0.71867  # ~71.867% in 2026

    # If you want to tax rental income (optional; default 0 = ignore tax)
    rental_income_tax_rate: float = 0.0

def annual_eigenwoningforfait(t: NLHomeTax2026) -> float:
    woz = t.woz_value
    if woz <= t.villa_threshold:
        return woz * t.ewf_rate_main
    # Simplified: main rate up to threshold + higher rate on the excess
    return t.villa_threshold * t.ewf_rate_main + (woz - t.villa_threshold) * t.ewf_rate_above_threshold

def compute_tax_cashflow_monthly(
    interests: List[float],
    tax: NLHomeTax2026,
    months_rented: int,
    monthly_rent_income: float,
) -> float:
    """
    Returns TOTAL tax effect over the horizon as a cashflow (positive = tax benefit, negative = tax cost).
    We compute yearly buckets:
      taxable_home = EWF - deductible_interest
      - if negative: deduction benefit at min(marginal, cap)
      - if positive: tax cost at marginal
      Optional Wet Hillen reduces taxable addition when EWF > interest and debt is small (simplified trigger).
    Rental income tax: optional flat rate on rental cash income (simplified).
    """
    total_tax_effect = 0.0
    ewf_year = annual_eigenwoningforfait(tax)

    # Rental income tax (very simplified)
    rental_income_total = months_rented * monthly_rent_income
    total_tax_effect -= rental_income_total * tax.rental_income_tax_rate

    # Yearly bucketize interest by calendar-year within horizon (just groups of 12 months)
    n_months = len(interests)
    years = math.ceil(n_months / 12) if n_months > 0 else 0

    for y in range(years):
        start = y * 12
        end = min((y + 1) * 12, n_months)
        interest_year = sum(interests[start:end])

        taxable_home = ewf_year - interest_year  # positive -> added income; negative -> deduction

        # Wet Hillen: applies when taxable_home > 0 and (nearly) no interest.
        # The real rule is about (small) eigenwoningschuld; we approximate by "interest very small".
        if tax.apply_wet_hillen and taxable_home > 0:
            # Reduce the taxable addition by the Hillen factor:
            taxable_home = taxable_home * (1.0 - tax.wet_hillen_factor_2026)

        if taxable_home < 0:
            # deduction -> tax benefit
            deduction_rate = min(tax.marginal_tax_rate, tax.deduction_cap_rate)
            total_tax_effect += (-taxable_home) * deduction_rate
        else:
            # addition -> tax cost
            total_tax_effect -= taxable_home * tax.marginal_tax_rate

    return total_tax_effect

