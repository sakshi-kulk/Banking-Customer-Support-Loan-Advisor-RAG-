"""Deterministic EMI calculator tool."""

import math


def calculate_emi(principal: float, interest_rate: float, tenure_years: float):
    """Return the standard monthly loan installment in INR."""
    values = (principal, interest_rate, tenure_years)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Loan inputs must be finite numbers")
    if principal <= 0:
        raise ValueError("principal must be greater than zero")
    if interest_rate < 0:
        raise ValueError("interest_rate cannot be negative")
    if tenure_years <= 0:
        raise ValueError("tenure_years must be greater than zero")

    months = tenure_years * 12
    monthly_rate = interest_rate / (12 * 100)
    if monthly_rate == 0:
        amount = principal / months
    else:
        factor = (1 + monthly_rate) ** months
        amount = principal * monthly_rate * factor / (factor - 1)
    return {"monthly_emi": round(amount, 2)}


def build_emi_tool():
    from langchain_core.tools import StructuredTool

    return StructuredTool.from_function(
        func=calculate_emi,
        name="emi_calculator",
        description="Calculate monthly EMI from principal, annual interest rate in percent, and tenure in years.",
    )