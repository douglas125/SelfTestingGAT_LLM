import pytest
from unittest.mock import patch, mock_open
from gat_llm.tools.update_user_details import ToolUpdateUserDetails


@pytest.fixture
def mock_os_path_isfile():
    with patch("os.path.isfile") as mock:
        yield mock


def test_unexpected_arg(unexpected_param_msg):
    tuud = ToolUpdateUserDetails()
    ans = tuud("READ", username="testuser", unexpected_argument=None)
    assert ans == f"{unexpected_param_msg}unexpected_argument"


def test_read_user_details_success(mock_os_path_isfile):
    mock_os_path_isfile.return_value = True
    tuud = ToolUpdateUserDetails()
    with patch(
        "builtins.open",
        mock_open(read_data="[user_info]\n[name]John Doe[/name]\n[/user_info]"),
    ):
        result = tuud("READ", username="testuser")
    assert result == "[user_info]\n[name]John Doe[/name]\n[/user_info]"


def test_read_user_details_not_found(mock_os_path_isfile):
    mock_os_path_isfile.return_value = False
    tuud = ToolUpdateUserDetails()
    result = tuud("READ", username="testuser")
    assert result == "No information available."


def test_write_user_details_success():
    tuud = ToolUpdateUserDetails()
    with patch("builtins.open", mock_open()) as mock_file:
        result = tuud(
            "WRITE",
            username="testuser",
            contents="[user_info]\n[name]Jane Doe[/name]\n[/user_info]",
        )
    assert result == "User information saved successfully."
    mock_file.assert_called_once_with("user_info/testuser.txt", "w", encoding="utf-8")
    mock_file().write.assert_called_once_with(
        "[user_info]\n[name]Jane Doe[/name]\n[/user_info]"
    )


def test_invalid_action():
    tuud = ToolUpdateUserDetails()
    result = tuud("INVALID", username="testuser")
    assert "Action must be one of" in result


# Add more tests for different scenarios and edge cases
