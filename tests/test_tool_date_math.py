import pytest

from ..tools.do_date_math import ToolDoDateMath


@pytest.mark.parametrize(
    "initial_date,delta,delta_type,expected",
    [
        ("2024-10-01", "1", "day", "2024-10-02 Wednesday"),
        ("2024-10-01", "2, 4", "day", "2024-10-03 Thursday,2024-10-05 Saturday"),
        ("2024-10-03", "-1", "day", "2024-10-02 Wednesday"),
        ("2024-10-09", "0", "day", "2024-10-09 Wednesday"),
        ("2024-10-09", "0", "month", "2024-10-09 Wednesday"),
        ("2024-10-09", "0", "year", "2024-10-09 Wednesday"),
        ("2023-10-01", "2", "month", "2023-12-01 Friday"),
        ("2023-10-01", "3,-3", "month", "2024-01-01 Monday,2023-07-01 Saturday"),
        (
            "2023-10-01",
            "1,-1,0, -3,3, 4",
            "year",
            "2024-10-01 Tuesday,2022-10-01 Saturday,2023-10-01 Sunday,2020-10-01 Thursday,2026-10-01 Thursday,2027-10-01 Friday",
        ),
    ],
)
def test_sum_periods(initial_date, delta, delta_type, expected):
    dm = ToolDoDateMath()
    computed_date = dm(initial_date, delta, delta_type)
    assert computed_date == expected


def test_unexpected_arg(unexpected_param_msg):
    dm = ToolDoDateMath()
    computed_date = dm("initial_date", "delta", "delta_type", unexpected_argument=None)
    assert computed_date == f"{unexpected_param_msg}unexpected_argument"
