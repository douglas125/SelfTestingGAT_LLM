import pytest
from unittest.mock import patch, Mock
from gat_llm.tools.get_webpage_contents import ToolGetUrlContent


@pytest.fixture
def mock_requests_get():
    with patch("requests.get") as mock_get:
        mock_response = Mock()
        mock_response.content = (
            "<html><body><p><div>Test content</div></p></body></html>"
        )
        mock_response.text = "Test content"
        mock_response.url = "http://example.com"
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        yield mock_get


def test_unexpected_arg(unexpected_param_msg):
    tguc = ToolGetUrlContent(None)
    ans = tguc("http://example.com", True, unexpected_argument=None)
    assert ans == f"{unexpected_param_msg}unexpected_argument"


def test_get_url_content_all_visible_html(mock_requests_get):
    tguc = ToolGetUrlContent(None)
    result = tguc("http://example.com", return_all_visible_html="True")
    assert "Test content" in result
    assert "<source_url>http://example.com</source_url>" in result
    assert "<status_code>200</status_code>" in result


def test_get_url_content_text_and_urls(mock_requests_get):
    tguc = ToolGetUrlContent(None)
    result = tguc("http://example.com", return_all_visible_html="False")
    assert "Test content" in result
    assert "<source_url>http://example.com</source_url>" in result
    assert "<status_code>200</status_code>" in result
    assert "<urls>" in result


@patch("requests.get", side_effect=Exception("Connection error"))
def test_get_url_content_error(mock_requests_get):
    tguc = ToolGetUrlContent(None)
    result = tguc("http://example.com")
    assert "Could not retrieve page from URL" in result
    assert "Connection error" in result


# Add more tests for different scenarios and edge cases
