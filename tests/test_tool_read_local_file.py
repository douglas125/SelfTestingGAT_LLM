from unittest.mock import call, patch, mock_open

from gat_llm.tools.read_local_file import ToolReadLocalFile


def test_unexpected_arg(unexpected_param_msg):
    trlf = ToolReadLocalFile()
    ans = trlf("media/file.txt", unexpected_argument=None)
    assert ans == f"{unexpected_param_msg}unexpected_argument"


@patch("builtins.open", new_callable=mock_open, read_data="file_data")
@patch("os.path.isfile", return_value=True)
def test_file_read(mock_isfile, mock_open_file):
    trlf = ToolReadLocalFile()
    ans = trlf("media/file.txt")

    mock_isfile.assert_called_with("media/file.txt")
    mock_open_file.assert_called_with("media/file.txt", "r", encoding="utf-8")
    assert "file_data" in ans


@patch("builtins.open", new_callable=mock_open, read_data="file_data")
@patch("os.path.isfile", return_value=True)
def test_files_read(mock_isfile, mock_open_file):
    trlf = ToolReadLocalFile()
    ans = trlf("media/file.txt \n media/file2.txt")

    mock_isfile.assert_has_calls([call("media/file.txt"), call("media/file2.txt")])
    mock_open_file.assert_has_calls(
        [
            call("media/file.txt", "r", encoding="utf-8"),
            call("media/file2.txt", "r", encoding="utf-8"),
        ],
        any_order=True,
    )
    assert "file_data" in ans


@patch("builtins.open", new_callable=mock_open, read_data="file_data")
@patch("os.path.isfile", return_value=False)
def test_file_not_read(mock_isfile, mock_open_file):
    trlf = ToolReadLocalFile()
    ans = trlf("media/file.txt")

    mock_isfile.assert_called_with("media/file.txt")
    mock_open_file.assert_not_called()
    assert "Error: Did not find file" in ans
