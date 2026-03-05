"""
Mortgage calculation module for complex multi-loan scenarios.

Supports:
- Multiple separate loans with different terms
- Annuity and linear amortization types
- Rate resets at specific months
- Percentage-based loan specifications
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import copy

from src.utils import monthly_payment_annuity


class AmortizationType(Enum):
    """Mortgage amortization type"""
    ANNUITY = "annuity"  # Equal monthly payments
    LINEAR = "linear"    # Equal principal payments


@dataclass
class RateReset:
    """Defines a rate reset event at a specific month"""
    month: int                              # Absolute month when reset occurs
    new_rate_min: float                     # Minimum possible new rate
    new_rate_max: float                     # Maximum possible new rate
    new_rate_default: Optional[float] = None  # Default for non-sensitivity runs


@dataclass
class MortgageLoan:
    """Individual mortgage loan configuration"""
    percentage: Optional[float] = None        # Percentage of total mortgage (0.0-1.0)
    principal: Optional[float] = None         # Absolute loan amount (calculated if percentage given)
    annual_rate: float = 0.04                 # Initial annual interest rate
    term_years: int = 30                      # Total term in years
    amortization_type: str = "annuity"        # "annuity" or "linear"
    fixed_period_years: Optional[int] = None  # Fixed rate period (None = entire term)
    rate_resets: Optional[List[RateReset]] = field(default=None)  # Future rate changes
    label: Optional[str] = None               # For identification (e.g., "Loan A")
    
    def __post_init__(self):
        """Validate and normalize fields"""
        if self.percentage is None and self.principal is None:
            raise ValueError("Either 'percentage' or 'principal' must be specified")
        if self.percentage is not None and not (0.0 <= self.percentage <= 1.0):
            raise ValueError(f"Percentage must be between 0.0 and 1.0, got {self.percentage}")
        
        # Convert string amortization type to enum if needed
        if isinstance(self.amortization_type, str):
            self.amortization_type = AmortizationType(self.amortization_type.lower())
        
        # Convert rate_resets dicts to RateReset objects if needed
        if self.rate_resets:
            self.rate_resets = [
                RateReset(**r) if isinstance(r, dict) else r
                for r in self.rate_resets
            ]


def monthly_payment_linear(principal: float, annual_rate: float, term_months: int) -> float:
    """
    Calculate first month payment for linear amortization.
    
    Linear: Equal principal payments + decreasing interest
    First payment = (principal / term_months) + (principal * monthly_rate)
    
    Args:
        principal: Loan amount
        annual_rate: Annual interest rate (e.g., 0.04 for 4%)
        term_months: Loan term in months
    
    Returns:
        First month's payment amount
    """
    if term_months <= 0:
        raise ValueError("term_months must be > 0")
    
    monthly_principal = principal / term_months
    monthly_interest = principal * (annual_rate / 12.0)
    return monthly_principal + monthly_interest


def amortize_schedule_linear(
    principal: float,
    annual_rate: float,
    term_months: int,
    horizon_months: int,
) -> Dict[str, Any]:
    """
    Generate linear amortization schedule.
    
    Linear amortization: Equal principal payments each month, with interest
    calculated on remaining balance (so total payment decreases over time).
    
    Args:
        principal: Loan amount
        annual_rate: Annual interest rate
        term_months: Loan term in months
        horizon_months: Simulation horizon
    
    Returns:
        Dictionary with schedule details (same structure as annuity schedule)
    """
    bal = max(0.0, principal)
    monthly_principal = bal / term_months if term_months > 0 else 0.0
    r = annual_rate / 12.0
    
    interests: List[float] = []
    principals: List[float] = []
    balances: List[float] = []
    payments: List[float] = []
    
    for month in range(min(horizon_months, term_months)):
        if bal <= 1e-9:
            break
        
        interest = bal * r
        principal_paid = min(monthly_principal, bal)
        payment = interest + principal_paid
        bal -= principal_paid
        
        interests.append(interest)
        principals.append(principal_paid)
        balances.append(bal)
        payments.append(payment)
    
    return {
        "monthly_payment": payments[0] if payments else 0.0,  # First payment
        "monthly_payments": payments,  # All payments (decreasing)
        "interests": interests,
        "principals": principals,
        "balances": balances,
        "total_interest": sum(interests),
        "total_principal": sum(principals),
        "remaining_balance": bal,
        "months_paid": len(interests),
    }


def amortize_single_loan_with_resets(
    loan: MortgageLoan,
    horizon_months: int,
    rate_resets: Optional[Dict[int, float]] = None
) -> Dict[str, Any]:
    """
    Amortize single loan with potential rate resets.
    
    When rate resets occur, recalculate payment based on:
    - Remaining balance
    - New rate
    - Remaining term
    
    Args:
        loan: MortgageLoan configuration
        horizon_months: Simulation horizon
        rate_resets: Optional dict mapping month -> new_rate
    
    Returns:
        Dictionary with schedule details
    """    
    rate_resets = rate_resets or {}
    current_rate = loan.annual_rate
    bal = loan.principal
    term_months = loan.term_years * 12
    
    interests = []
    principals = []
    balances = []
    payments = []
    
    for month in range(horizon_months):
        if bal <= 1e-9 or month >= term_months:
            break
        
        # Check for rate reset
        if month in rate_resets:
            current_rate = rate_resets[month]
        
        remaining_term = term_months - month
        
        # Calculate payment based on amortization type
        if loan.amortization_type == AmortizationType.ANNUITY:
            payment = monthly_payment_annuity(bal, current_rate, remaining_term)
            interest = bal * (current_rate / 12.0)
            principal_paid = payment - interest
        else:  # LINEAR
            monthly_principal = bal / remaining_term
            interest = bal * (current_rate / 12.0)
            principal_paid = min(monthly_principal, bal)
            payment = interest + principal_paid
        
        # Handle final month
        if principal_paid > bal:
            principal_paid = bal
            payment = interest + principal_paid
        
        bal -= principal_paid
        
        interests.append(interest)
        principals.append(principal_paid)
        balances.append(bal)
        payments.append(payment)
    
    return {
        "payments": payments,
        "interests": interests,
        "principals": principals,
        "balances": balances,
        "total_interest": sum(interests),
        "total_principal": sum(principals),
        "remaining_balance": bal,
        "months_paid": len(interests),
    }


def amortize_multi_loan(
    loans: List[MortgageLoan],
    horizon_months: int,
    rate_reset_values: Optional[Dict[str, Dict[int, float]]] = None
) -> Dict[str, Any]:
    """
    Calculate amortization for multiple loans with rate resets.
    
    Args:
        loans: List of MortgageLoan configurations
        horizon_months: Simulation horizon
        rate_reset_values: Optional dict mapping loan_label -> {month: new_rate}
    
    Returns:
        Aggregated schedule with per-loan breakdowns
    """
    rate_reset_values = rate_reset_values or {}
    
    # Initialize aggregated arrays
    total_interests = [0.0] * horizon_months
    total_principals = [0.0] * horizon_months
    total_payments = [0.0] * horizon_months
    
    loan_schedules = {}
    
    for loan_idx, loan in enumerate(loans):
        label = loan.label or f"Loan {loan_idx + 1}"
        
        # Determine rate resets for this loan
        resets = {}
        if loan.rate_resets:
            for reset in loan.rate_resets:
                if label in rate_reset_values and reset.month in rate_reset_values[label]:
                    resets[reset.month] = rate_reset_values[label][reset.month]
                elif reset.new_rate_default is not None:
                    resets[reset.month] = reset.new_rate_default
                else:
                    # Use midpoint of range as default
                    resets[reset.month] = (reset.new_rate_min + reset.new_rate_max) / 2
        
        # Calculate schedule with rate resets
        schedule = amortize_single_loan_with_resets(
            loan=loan,
            horizon_months=horizon_months,
            rate_resets=resets
        )
        
        loan_schedules[label] = schedule
        
        # Aggregate
        for month in range(len(schedule["interests"])):
            total_interests[month] += schedule["interests"][month]
            total_principals[month] += schedule["principals"][month]
            total_payments[month] += schedule["payments"][month]
    
    # Trim arrays to actual length
    actual_length = max(len(s["interests"]) for s in loan_schedules.values()) if loan_schedules else 0
    total_interests = total_interests[:actual_length]
    total_principals = total_principals[:actual_length]
    total_payments = total_payments[:actual_length]
    
    return {
        "monthly_payments": total_payments,
        "interests": total_interests,
        "principals": total_principals,
        "total_interest": sum(total_interests),
        "total_principal": sum(total_principals),
        "remaining_balance": sum(s["remaining_balance"] for s in loan_schedules.values()),
        "months_paid": actual_length,
        "loan_schedules": loan_schedules,  # Per-loan details
        "monthly_payment": total_payments[0] if total_payments else 0.0,  # First month
    }


def resolve_loan_percentages(loans: List[MortgageLoan], total_mortgage: float, auto_adjust: bool = True, verbose: bool = False) -> List[MortgageLoan]:
    """
    Resolve percentage-based loan specifications to absolute principals.
    
    If auto_adjust is True and percentages don't sum to 1.0, will automatically adjust
    the last percentage-based loan to make the total equal 1.0 (for 2-loan scenarios).
    
    Args:
        loans: List of MortgageLoan configurations (or dictionaries that will be converted)
        total_mortgage: Total mortgage amount
        auto_adjust: If True, automatically adjust percentages to sum to 1.0 (default: True)
    
    Returns:
        List of loans with principals calculated
    
    Raises:
        ValueError: If percentage values are invalid
    """
    import warnings
    
    # Convert dictionaries to MortgageLoan objects if needed (for YAML loading)
    converted_loans = []
    for loan in loans:
        if isinstance(loan, dict):
            # Convert rate_resets if present
            rate_resets = None
            if 'rate_resets' in loan and loan['rate_resets']:
                rate_resets = [
                    RateReset(**reset_dict) if isinstance(reset_dict, dict) else reset_dict
                    for reset_dict in loan['rate_resets']
                ]
            
            # Create MortgageLoan object
            loan_data = loan.copy()
            loan_data['rate_resets'] = rate_resets
            converted_loans.append(MortgageLoan(**loan_data))
        else:
            converted_loans.append(loan)
    
    resolved_loans = copy.deepcopy(converted_loans)
    
    # Separate loans into those with percentages and those without
    percentage_loans = []
    non_percentage_loans = []
    
    for loan in resolved_loans:
        if loan.percentage is not None and loan.principal is None:
            # Validate percentage is in correct range (0.0-1.0, not 0-100)
            if loan.percentage > 1.0:
                raise ValueError(
                    f"Loan percentage must be between 0.0 and 1.0 (e.g., 0.70 for 70%), "
                    f"but got {loan.percentage}. Did you mean {loan.percentage/100}?"
                )
            if loan.percentage < 0.0:
                raise ValueError(
                    f"Loan percentage cannot be negative, but got {loan.percentage}"
                )
            percentage_loans.append(loan)
        elif loan.principal is None:
            non_percentage_loans.append(loan)
    
    if not percentage_loans:
        return resolved_loans
    
    # Calculate total percentage
    total_percentage = sum(loan.percentage for loan in percentage_loans)
    
    # Auto-adjust if needed
    if auto_adjust and abs(total_percentage - 1.0) > 0.01:
        # For 2-loan scenarios, adjust the last loan
        if len(percentage_loans) == 2:
            adjustment_needed = 1.0 - total_percentage
            original_pct = percentage_loans[-1].percentage
            percentage_loans[-1].percentage = original_pct + adjustment_needed
            
            # Validate adjusted percentage is valid (allow small negative due to floating point)
            if percentage_loans[-1].percentage < -0.001 or percentage_loans[-1].percentage > 1.001:
                raise ValueError(
                    f"Cannot auto-adjust loan percentages: adjusting {percentage_loans[-1].label or 'last loan'} "
                    f"from {original_pct:.2%} by {adjustment_needed:.2%} would result in "
                    f"{percentage_loans[-1].percentage:.2%}, which is outside valid range [0%, 100%]"
                )
            
            # Clamp to valid range to handle floating point errors
            percentage_loans[-1].percentage = max(0.0, min(1.0, percentage_loans[-1].percentage))
            
            if verbose:
                warnings.warn(
                    f"Auto-adjusted {percentage_loans[-1].label or 'loan 2'} percentage from "
                    f"{original_pct:.2%} to {percentage_loans[-1].percentage:.2%} to ensure total equals 100%",
                    UserWarning
                )
        elif len(percentage_loans) > 2:
            # More than 2 loans - warn but don't auto-adjust
            warnings.warn(
                f"Loan percentages sum to {total_percentage:.2%} instead of 100%. "
                f"Auto-adjustment is only supported for 2-loan scenarios. "
                f"With {len(percentage_loans)} loans, please manually ensure percentages sum to 1.0.",
                UserWarning
            )
            # Still validate
            if abs(total_percentage - 1.0) > 0.01:
                loan_details = ", ".join([
                    f"{loan.label or f'Loan {i+1}'}: {loan.percentage:.2%}"
                    for i, loan in enumerate(percentage_loans)
                ])
                raise ValueError(
                    f"Loan percentages must sum to 1.0 (100%), but got {total_percentage:.2%}. "
                    f"Loans: [{loan_details}]"
                )
    
    # Calculate principals
    for loan in percentage_loans:
        loan.principal = loan.percentage * total_mortgage
    
    return resolved_loans
