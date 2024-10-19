# UNREVIEWED

import pytest
from unittest.mock import patch, mock_open
from ..tools.write_local_file import ToolWriteLocalFile


@pytest.fixture
def mock_os_path_isfile():
    with patch("os.path.isfile") as mock:
        yield mock


@pytest.fixture
def mock_os_makedirs():
    with patch("os.makedirs") as mock:
        yield mock


def test_unexpected_arg(unexpected_param_msg):
    twlf = ToolWriteLocalFile()
    ans = twlf("path/to/file.txt", "content", unexpected_argument=None)
    assert ans == f"{unexpected_param_msg}unexpected_argument"


def test_write_local_file_success(mock_os_path_isfile, mock_os_makedirs):
    mock_os_path_isfile.return_value = False
    twlf = ToolWriteLocalFile()
    with patch("builtins.open", mock_open()) as mock_file:
        result = twlf("path/to/file.txt", "Test content")
    assert "<outcome>" in result
    assert "File written successfully" in result
    assert "<path_to_file>path/to/file.txt</path_to_file>" in result
    assert "</outcome>" in result
    mock_file.assert_called_once_with("path/to/file.txt", "w", encoding="utf-8")
    mock_file().write.assert_called_once_with("Test content")


def test_write_local_file_already_exists(mock_os_path_isfile, mock_os_makedirs):
    mock_os_path_isfile.return_value = True
    twlf = ToolWriteLocalFile()
    result = twlf("path/to/existing_file.txt", "Test content")
    assert "<outcome>" in result
    assert "Error: File already exists" in result
    assert "This tool is not allowed to overwrite files" in result
    assert "</outcome>" in result


@pytest.mark.parametrize(
    "file_path,content",
    [
        ("media/new_file.txt", "Hello, World!"),
        ("path/to/document.md", "# Markdown Title\n\nContent here"),
        ("scripts/test.py", "print('Hello, Python!')"),
    ],
)
def test_write_local_file_various_types(
    file_path, content, mock_os_path_isfile, mock_os_makedirs
):
    mock_os_path_isfile.return_value = False
    twlf = ToolWriteLocalFile()
    with patch("builtins.open", mock_open()) as mock_file:
        result = twlf(file_path, content)
    assert "<outcome>" in result
    assert "File written successfully" in result
    assert f"<path_to_file>{file_path}</path_to_file>" in result
    assert "</outcome>" in result
    mock_file.assert_called_once_with(file_path, "w", encoding="utf-8")
    mock_file().write.assert_called_once_with(content)


# Add more tests for different scenarios and edge cases
