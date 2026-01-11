from typing import Optional, Dict, List
import math


def monthly_payment_annuity(principal: float, annual_rate: float, term_months: int) -> float:
    r = annual_rate / 12.0
    if term_months <= 0:
        raise ValueError("term_months must be > 0")
    if abs(r) < 1e-12:
        return principal / term_months

    return principal * (r * (1 + r) ** term_months) / ((1 + r) ** term_months - 1)

def amortize_schedule(
    principal: float,
    annual_rate: float,
    term_months: int,
    horizon_months: int,
) -> Dict[str, object]:
    """
    Returns month-by-month schedule up to horizon_months (or until loan paid off).
    """
    bal = max(0.0, principal)
    payment = monthly_payment_annuity(bal, annual_rate, term_months) if bal > 0 else 0.0
    r = annual_rate / 12.0

    interests: List[float] = []
    principals: List[float] = []
    balances: List[float] = []

    for _ in range(min(horizon_months, term_months)):
        if bal <= 1e-9:
            break
        interest = bal * r
        principal_paid = payment - interest

        if principal_paid > bal:  # final month
            principal_paid = bal
            payment_eff = interest + principal_paid
        else:
            payment_eff = payment

        bal -= principal_paid

        interests.append(interest)
        principals.append(principal_paid)
        balances.append(bal)

    return {
        "monthly_payment": payment,
        "interests": interests,
        "principals": principals,
        "balances": balances,
        "total_interest": sum(interests),
        "total_principal": sum(principals),
        "remaining_balance": bal,
        "months_paid": len(interests),
    }

def future_value_monthly_growth(initial_value: float, annual_growth: float, months: int) -> float:
    g_m = (1 + annual_growth) ** (1/12) - 1
    return initial_value * ((1 + g_m) ** months)

def invest_monthly(cashflows, annual_return):
    """
    cashflows: list of monthly amounts invested (can be 0 or positive)
    annual_return: e.g. 0.05 for 5%
    """
    r = (1 + annual_return) ** (1/12) - 1
    balance = 0.0
    for cf in cashflows:
        balance = balance * (1 + r) + max(0.0, cf)
    return balance

