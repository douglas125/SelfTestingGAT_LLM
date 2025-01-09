import pytest
from ..tools.solve_symbolic import ToolSolveSymbolic


def test_unexpected_arg(unexpected_param_msg):
    tss = ToolSolveSymbolic()
    ans = tss("import sympy", unexpected_argument=None)
    assert ans == f"{unexpected_param_msg}unexpected_argument"


def test_solve_symbolic_success():
    tss = ToolSolveSymbolic()
    sympy_code = """
import sympy as sp
x = sp.Symbol('x')
expr = x**2 + 2*x + 1
ans = sp.solve(expr)
"""
    result = tss(sympy_code)
    assert result == "[-1]"


def test_solve_symbolic_execution_error():
    tss = ToolSolveSymbolic()
    sympy_code = "invalid_code"
    result = tss(sympy_code)
    assert "Code did NOT execute correctly" in result
    assert "Error description" in result


@pytest.mark.parametrize(
    "code,expected",
    [
        (
            "import sympy as sp\nx = sp.Symbol('x')\nans = sp.integrate(sp.sin(x), x)",
            "-cos(x)",
        ),
        (
            "import sympy as sp\nx, y = sp.symbols('x y')\nans = sp.diff(x**2 + y**2, x)",
            "2*x",
        ),
        (
            "import sympy as sp\nx = sp.Symbol('x')\nans = sp.limit(sp.sin(x)/x, x, 0)",
            "1",
        ),
    ],
)
def test_solve_symbolic_various_operations(code, expected):
    tss = ToolSolveSymbolic()
    result = tss(code)
    assert result == expected


# Add more tests for different scenarios and edge cases
