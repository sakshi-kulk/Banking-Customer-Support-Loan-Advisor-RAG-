import pytest

from tools.eligibility_tool import check_eligibility


def test_eligibility_uses_strict_salary_and_inclusive_debt_limit():
    assert check_eligibility(60_000, 30_000)["eligible"] is True
    assert check_eligibility(30_000, 0)["eligible"] is False


def test_rejects_debt_ratio_above_half():
    result = check_eligibility(60_000, 30_001)
    assert result["eligible"] is False
    assert "50%" in result["reason"]


def test_rejects_negative_existing_emi():
    with pytest.raises(ValueError, match="cannot be negative"):
        check_eligibility(60_000, -1)