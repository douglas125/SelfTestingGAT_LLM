import pytest
from unittest.mock import patch, mock_open
from gat_llm.tools.make_custom_plot import ToolMakeCustomPlot


@pytest.fixture
def mock_exec():
    with patch("builtins.exec") as mock:
        yield mock


@pytest.fixture
def mock_os_path_isfile():
    with patch("os.path.isfile") as mock:
        mock.return_value = True
        yield mock


def test_unexpected_arg(unexpected_param_msg):
    tmcp = ToolMakeCustomPlot()
    ans = tmcp("import matplotlib.pyplot as plt", unexpected_argument=None)
    assert ans == f"{unexpected_param_msg}unexpected_argument"


def test_make_custom_plot_success(mock_exec, mock_os_path_isfile):
    tmcp = ToolMakeCustomPlot()
    plot_code = """
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.savefig('media/plot.jpg')
"""
    result = tmcp(plot_code)
    assert "<image>" in result
    assert "<path_to_image>" in result
    assert "</image>" in result


def test_make_custom_plot_execution_error(mock_exec):
    mock_exec.side_effect = Exception("Execution error")
    tmcp = ToolMakeCustomPlot()
    plot_code = "invalid_code"
    result = tmcp(plot_code)
    assert "Plot was NOT generated" in result
    assert "Execution error" in result


@patch("os.path.isfile", return_value=False)
def test_make_custom_plot_file_not_saved(mock_isfile, mock_exec):
    tmcp = ToolMakeCustomPlot()
    plot_code = """
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.savefig('media/plot.jpg')
"""
    result = tmcp(plot_code)
    assert "Error: Image was not saved correctly" in result


# Add more tests for different scenarios and edge cases
