import pytest
from gat_llm.tools.solve_python_code import ToolSolvePythonCode


def test_unexpected_arg(unexpected_param_msg):
    tspc = ToolSolvePythonCode()
    ans = tspc("print('Hello, World!')", unexpected_argument=None)
    assert ans == f"{unexpected_param_msg}unexpected_argument"


def test_solve_python_code_success():
    tspc = ToolSolvePythonCode()
    python_code = """
text = "Hello, World!"
ans = text[::-1]
"""
    result = tspc(python_code)
    assert result == "!dlroW ,olleH"


def test_solve_python_code_execution_error():
    tspc = ToolSolvePythonCode()
    python_code = "invalid_code"
    result = tspc(python_code)
    assert "Code did NOT execute correctly" in result
    assert "Error description" in result


@pytest.mark.parametrize(
    "code,expected",
    [
        (
            "ans = sorted([3, 1, 4, 1, 5, 9, 2, 6, 5, 3])",
            "[1, 1, 2, 3, 3, 4, 5, 5, 6, 9]",
        ),
        ("ans = ''.join(sorted(set('mississippi')))", "imps"),
        ("ans = sum([x for x in range(1, 101) if x % 2 == 0])", "2550"),
        ("ans = {x: x**2 for x in range(1, 6)}", "{1: 1, 2: 4, 3: 9, 4: 16, 5: 25}"),
    ],
)
def test_solve_python_code_various_operations(code, expected):
    tspc = ToolSolvePythonCode()
    result = tspc(code)
    assert result == expected


# Add more tests for different scenarios and edge cases
