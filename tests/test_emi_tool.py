import pytest

from tools.emi_tool import calculate_emi


def test_calculates_standard_emi_for_example():
    result = calculate_emi(1_000_000, 9, 5)
    assert result["monthly_emi"] == pytest.approx(20_758.36, abs=0.01)


def test_zero_interest_uses_equal_principal_payments():
    assert calculate_emi(120_000, 0, 1) == {"monthly_emi": 10_000.0}


@pytest.mark.parametrize(
    "values",
    [(0, 9, 5), (1_000, -1, 5), (1_000, 9, 0)],
)
def test_rejects_invalid_loan_inputs(values):
    with pytest.raises(ValueError):
        calculate_emi(*values)