import pytest
from unittest.mock import patch
from gat_llm.tools.read_file_names_in_local_folder import ToolReadLocalFolder


@pytest.fixture
def mock_os_path_isdir():
    with patch("os.path.isdir") as mock:
        mock.return_value = True
        yield mock


@pytest.fixture
def mock_os_listdir():
    with patch("os.listdir") as mock:
        mock.return_value = ["file1.txt", "file2.py", "folder"]
        yield mock


@pytest.fixture
def mock_os_path_isfile():
    with patch("os.path.isfile") as mock:
        mock.side_effect = lambda x: not x.endswith("folder")
        yield mock


def test_unexpected_arg(unexpected_param_msg):
    trlf = ToolReadLocalFolder()
    ans = trlf("/path/to/folder", unexpected_argument=None)
    assert ans == f"{unexpected_param_msg}unexpected_argument"


def test_read_file_names_success(
    mock_os_path_isdir, mock_os_listdir, mock_os_path_isfile
):
    trlf = ToolReadLocalFolder()
    result = trlf("/path/to/folder")
    assert "<files>" in result
    assert "file1.txt</file>" in result
    assert "file2.py</file>" in result
    assert "</files>" in result


@patch("os.path.isdir", return_value=False)
def test_read_file_names_invalid_folder(mock_isdir):
    trlf = ToolReadLocalFolder()
    result = trlf("/invalid/folder")
    assert "Error: Did not find folder" in result


# Add more tests for different scenarios and edge cases
