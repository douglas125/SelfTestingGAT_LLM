import pytest
from unittest.mock import patch, Mock
from ..tools.use_ffmpeg import ToolUseFFMPEG


@pytest.fixture
def mock_subprocess_run():
    with patch("subprocess.run") as mock:
        mock_process = Mock()
        mock_process.stdout = "ffmpeg output"
        mock_process.stderr = ""
        mock.return_value = mock_process
        yield mock


def test_unexpected_arg(unexpected_param_msg):
    tuf = ToolUseFFMPEG()
    ans = tuf("-i input.mp4 output.mp4", unexpected_argument=None)
    assert ans == f"{unexpected_param_msg}unexpected_argument"


def test_use_ffmpeg_success(mock_subprocess_run):
    tuf = ToolUseFFMPEG()
    result = tuf("-i input.mp4 output.mp4")
    assert "<ffmpeg_stdout>" in result
    assert "ffmpeg output" in result
    assert "</ffmpeg_stdout>" in result
    mock_subprocess_run.assert_called_once_with(
        "ffmpeg -i input.mp4 output.mp4", capture_output=True, text=True, shell=True
    )


def test_use_ffmpeg_with_error(mock_subprocess_run):
    mock_subprocess_run.return_value.stderr = "ffmpeg error"
    tuf = ToolUseFFMPEG()
    result = tuf("-i nonexistent.mp4 output.mp4")
    assert "<ffmpeg_stdout>" in result
    assert "ffmpeg error" in result
    assert "</ffmpeg_stdout>" in result


@pytest.mark.parametrize(
    "ffmpeg_args",
    [
        "-i input.mp4 -ss 10 -t 20 output.mp4",
        '-i input.mp4 -filter_complex "[0:v]setpts=0.5*PTS[v];[0:a]atempo=2.0[a]" -map "[v]" -map "[a]" output.mp4',
        "-i input.mp4 -vf scale=1280:720 output.mp4",
    ],
)
def test_use_ffmpeg_various_commands(ffmpeg_args, mock_subprocess_run):
    tuf = ToolUseFFMPEG()
    result = tuf(ffmpeg_args)
    assert "<ffmpeg_stdout>" in result
    assert "ffmpeg output" in result
    assert "</ffmpeg_stdout>" in result
    mock_subprocess_run.assert_called_once_with(
        f"ffmpeg {ffmpeg_args}", capture_output=True, text=True, shell=True
    )


# Add more tests for different scenarios and edge cases
