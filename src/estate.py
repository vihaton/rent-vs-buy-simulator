import math

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from src.tax import NLHomeTax2026, compute_tax_cashflow_monthly
from src.utils import amortize_schedule, future_value_monthly_growth, invest_monthly

# ----------------------------
# Scenario A: Renting
# ----------------------------

@dataclass
class RentingScenario:
    rent_monthly: float
    utilities_monthly: float
    living_months: int

def evaluate_renting(s: RentingScenario) -> Dict[str, float]:
    x = s.living_months
    monthly_cost = s.rent_monthly + s.utilities_monthly

    spent = x * monthly_cost
    income = 0.0

    return {
        "total_spent": spent,
        "total_income": income,
        "net_cashflow": income - spent,
        "wealth_end": income - spent,
        "avg_net_cost_per_month": monthly_cost,
        "monthly_net_cost": [monthly_cost] * x,
        "months": x,
    }

# ----------------------------
# Scenario B: Buying (with tax)
# ----------------------------

@dataclass
class BuyingScenario:
    # Home + financing
    purchase_price: float
    mortgage_principal: float
    mortgage_annual_rate: float
    mortgage_term_years: int

    living_months: int

    # Costs
    one_off_costs: float
    monthly_vve: float
    monthly_utilities: float

    # Renting out
    monthly_rent_income: float
    months_rented: int

    # Value change
    annual_value_growth: float

    # Exit
    sold_at_end: bool
    selling_cost_rate: float
    selling_cost_fixed: float = 0.0

    # Taxes
    tax: Optional[NLHomeTax2026] = None

def evaluate_buying(s: BuyingScenario) -> Dict[str, float]:
    x = s.living_months
    term_months = s.mortgage_term_years * 12
    months_rented = max(0, min(s.months_rented, x))

    sched = amortize_schedule(
        principal=s.mortgage_principal,
        annual_rate=s.mortgage_annual_rate,
        term_months=term_months,
        horizon_months=x,
    )

    # Cash outflows
    spent_one_off = s.one_off_costs
    spent_monthly_non_mortgage = x * (s.monthly_vve + s.monthly_utilities)
    spent_mortgage_interest = sched["total_interest"]
    spent_mortgage_principal = sched["total_principal"]

    total_spent = spent_one_off + spent_monthly_non_mortgage + spent_mortgage_interest + spent_mortgage_principal

    # Cash inflows
    total_income = months_rented * s.monthly_rent_income

    # Taxes (cashflow effect)
    tax_effect = 0.0
    if s.tax is not None:
        tax_effect = compute_tax_cashflow_monthly(
            interests=sched["interests"],
            tax=s.tax,
            months_rented=months_rented,
            monthly_rent_income=s.monthly_rent_income,
        )
        # tax_effect > 0 means you get money back / pay less tax overall
        total_income += max(0.0, tax_effect)
        total_spent += max(0.0, -tax_effect)

    # Home value at end
    home_value_end = future_value_monthly_growth(s.purchase_price, s.annual_value_growth, x)
    remaining_debt = sched["remaining_balance"]
    equity_end = home_value_end - remaining_debt

    sale_proceeds_net = 0.0
    selling_costs = 0.0

    if s.sold_at_end:
        selling_costs = home_value_end * s.selling_cost_rate + s.selling_cost_fixed
        sale_proceeds_net = max(0.0, home_value_end - selling_costs - remaining_debt)
        wealth_from_asset = sale_proceeds_net
    else:
        wealth_from_asset = equity_end

    net_cashflow = total_income - total_spent
    wealth_end = net_cashflow + wealth_from_asset

    monthly_net_cost = []

    for m in range(x):
        mortgage_payment = sched["monthly_payment"] if m < sched["months_paid"] else 0.0
        rent_income = s.monthly_rent_income if m < months_rented else 0.0

        monthly_cost = (
            mortgage_payment
            + s.monthly_vve
            + s.monthly_utilities
            - rent_income
        )
        monthly_net_cost.append(monthly_cost)

# return dict
    return {
        "total_spent": total_spent,
        "total_income": total_income,
        "net_cashflow": net_cashflow,
        "wealth_end": wealth_end,
        "avg_net_cost_per_month": (total_spent - total_income) / x if x > 0 else 0.0,
        "months": x,

        # breakdown
        "mortgage_monthly_payment": sched["monthly_payment"],
        "mortgage_interest_paid": spent_mortgage_interest,
        "mortgage_principal_paid": spent_mortgage_principal,
        "mortgage_remaining_balance": remaining_debt,
        "monthly_net_cost": monthly_net_cost,
        "home_value_end": home_value_end,
        "equity_end_if_not_sold": equity_end,
        "selling_costs_if_sold": selling_costs,
        "net_sale_proceeds_if_sold": sale_proceeds_net,
        "tax_effect_total_over_horizon": tax_effect,
    }

def invest_difference(
    rent_result,
    buy_result,
    annual_return=0.05,
):
    # Positive means renting is cheaper that month → invest
    mon_net_r = np.array(rent_result["monthly_net_cost"])
    mon_net_b = np.array(buy_result["monthly_net_cost"])
    len_diff = len(mon_net_b) - len(mon_net_r)
    if len_diff > 0:
        append_rental = [0] * len_diff
        mon_net_r = np.concatenate((mon_net_r, append_rental))
        print("appended rental months")

    diffs = np.array([
        buy - rent
        for rent, buy in zip(
            mon_net_r,
            mon_net_b,
        )
    ])
    renting_invests = np.where(diffs > 0, diffs, 0)  # Positive values for renting, 0 otherwise
    buying_invests = np.where(diffs <= 0, -diffs, 0)  # Positive values for buying, 0 otherwise

    return {
        "renting_invests": invest_monthly(renting_invests, annual_return),
        "buying_invests": invest_monthly(buying_invests, annual_return),
        "principal_r": renting_invests.sum(),
        "principal_b": buying_invests.sum()
    }


def compare_scenarios(rent: Optional[RentingScenario], buy: Optional[BuyingScenario]) -> Dict[str, Dict[str, float]]:
    out = {}
    if rent is not None:
        out["renting"] = evaluate_renting(rent)
    if buy is not None:
        out["buying"] = evaluate_buying(buy)
    if rent is not None and buy is not None:
        out["invest"] = invest_difference(
            out["renting"],
            out["buying"],
        )
    return out
