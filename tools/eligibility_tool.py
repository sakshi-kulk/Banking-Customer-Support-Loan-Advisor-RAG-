"""Illustrative loan eligibility rules requested for this application."""

import math


def check_eligibility(salary: float, existing_emi: float):
    if not math.isfinite(float(salary)) or not math.isfinite(float(existing_emi)):
        raise ValueError("salary and existing_emi must be finite numbers")
    if salary <= 0:
        raise ValueError("salary must be greater than zero")
    if existing_emi < 0:
        raise ValueError("existing_emi cannot be negative")

    reasons = []
    if salary <= 30000:
        reasons.append("Monthly salary must be greater than ₹30,000.")
    debt_ratio = existing_emi / salary
    if debt_ratio > 0.5:
        reasons.append("Existing EMI must not exceed 50% of monthly salary.")
    return {
        "eligible": not reasons,
        "reason": " ".join(reasons) if reasons else "Meets the sample salary and debt-ratio rules.",
        "debt_ratio": round(debt_ratio, 4),
    }


def build_eligibility_tool():
    from langchain_core.tools import StructuredTool

    return StructuredTool.from_function(
        func=check_eligibility,
        name="loan_eligibility_check",
        description="Check sample eligibility rules using monthly salary and existing monthly EMI. This is not a bank approval.",
    )