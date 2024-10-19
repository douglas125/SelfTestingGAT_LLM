import pytest
from unittest.mock import patch
from ..tools.solve_numeric import ToolSolveNumeric


def test_unexpected_arg(unexpected_param_msg):
    tsn = ToolSolveNumeric()
    ans = tsn("import numpy as np", unexpected_argument=None)
    assert ans == f"{unexpected_param_msg}unexpected_argument"


def test_solve_numeric_success():
    tsn = ToolSolveNumeric()
    numpy_code = """
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
ans = np.mean(arr)
"""
    result = tsn(numpy_code)
    assert result == "3.0"


def test_solve_numeric_execution_error():
    tsn = ToolSolveNumeric()
    numpy_code = "invalid_code"
    result = tsn(numpy_code)
    assert "Code did NOT execute correctly" in result
    assert "Error description" in result


@pytest.mark.parametrize(
    "code,expected",
    [
        ("import numpy as np\nans = np.sum([1, 2, 3, 4, 5])", "15"),
        ("import numpy as np\nans = np.prod([1, 2, 3, 4, 5])", "120"),
        ("import numpy as np\nans = np.max([1, 2, 3, 4, 5])", "5"),
        ("import numpy as np\nans = np.min([1, 2, 3, 4, 5])", "1"),
    ],
)
def test_solve_numeric_various_operations(code, expected):
    tsn = ToolSolveNumeric()
    result = tsn(code)
    assert result == expected


# Add more tests for different scenarios and edge cases
