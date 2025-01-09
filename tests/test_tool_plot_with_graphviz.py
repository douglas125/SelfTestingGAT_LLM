import pytest
from unittest.mock import patch, Mock
from gat_llm.tools.plot_with_graphviz import ToolPlotWithGraphviz


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
    tpwg = ToolPlotWithGraphviz()
    ans = tpwg("import pydot", unexpected_argument=None)
    assert ans == f"{unexpected_param_msg}unexpected_argument"


def test_plot_with_graphviz_success(mock_exec, mock_os_path_isfile):
    tpwg = ToolPlotWithGraphviz()
    graph_code = """
import pydot
graph = pydot.Dot(graph_type='graph')
edge = pydot.Edge("node1", "node2")
graph.add_edge(edge)
"""
    result = tpwg(graph_code)
    assert "<image>" in result
    assert "<path_to_image>" in result
    assert "</image>" in result


def test_plot_with_graphviz_execution_error(mock_exec):
    mock_exec.side_effect = Exception("Execution error")
    tpwg = ToolPlotWithGraphviz()
    graph_code = "invalid_code"
    result = tpwg(graph_code)
    assert "Graph was NOT generated" in result
    assert "Execution error" in result


@patch("os.path.isfile", return_value=False)
def test_plot_with_graphviz_file_not_saved(mock_isfile):
    tpwg = ToolPlotWithGraphviz()
    graph_code = """
import pydot
graph = pydot.Dot(graph_type='graph')
edge = pydot.Edge("node1", "node2")
graph.add_edge(edge)
"""
    result = tpwg(graph_code)
    assert "Error: Image was not saved correctly" in result


# Add more tests for different scenarios and edge cases
