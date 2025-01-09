import pytest
from unittest.mock import patch, Mock
from gat_llm.tools.make_qr_code import ToolMakeQRCode


@pytest.fixture
def mock_qrcode():
    with patch("qrcode.QRCode") as mock:
        mock_qr = Mock()
        mock_qr.make_image.return_value = Mock()
        mock.return_value = mock_qr
        yield mock


@pytest.fixture
def mock_os_path_isfile():
    with patch("os.path.isfile") as mock:
        mock.return_value = True
        yield mock


def test_unexpected_arg(unexpected_param_msg):
    tmqc = ToolMakeQRCode()
    ans = tmqc("Test QR Code", unexpected_argument=None)
    assert ans == f"{unexpected_param_msg}unexpected_argument"


def test_make_qr_code_success(mock_qrcode, mock_os_path_isfile):
    tmqc = ToolMakeQRCode()
    result = tmqc("Test QR Code")
    assert "<image>" in result
    assert "<path_to_image>" in result
    assert "</image>" in result


def test_make_qr_code_with_error_correction(mock_qrcode, mock_os_path_isfile):
    tmqc = ToolMakeQRCode()
    result = tmqc("Test QR Code", error_correction="high")
    assert "<image>" in result
    assert "<path_to_image>" in result
    assert "</image>" in result


def test_make_qr_code_invalid_error_correction():
    tmqc = ToolMakeQRCode()
    result = tmqc("Test QR Code", error_correction="invalid")
    assert "error_correction must be one of" in result


@patch("os.path.isfile", return_value=False)
def test_make_qr_code_file_not_saved(mock_isfile, mock_qrcode):
    tmqc = ToolMakeQRCode()
    result = tmqc("Test QR Code")
    assert "Error: Image was not saved correctly" in result


# Add more tests for different scenarios and edge cases
