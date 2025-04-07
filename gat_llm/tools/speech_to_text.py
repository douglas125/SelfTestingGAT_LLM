import os

import numpy as np
from openai import OpenAI

rng = np.random.default_rng()


class ToolSpeechToText:
    def __init__(self):
        self.name = "speech_to_text"

        self.tool_description = {
            "name": self.name,
            "description": """Uses an automatic speech recognition tool to convert the provided audio into text in SRT format (subtitles).
Unless requested by the user, do not attempt to read the produced SRT file because the transcription may be too long.
Returns: path to a file containing a transcription of the audio in SRT format.
Raises ValueError: if not able to read the audio or video file.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "audio_file_path": {
                        "type": "string",
                        "description": "Path to the audio file that will be converted to text.",
                    },
                    "language": {
                        "type": "string",
                        "description": "Language of the audio in file audio_file_path. Should be provided in ISO-639-1 (e.g. en, pt, es) format.",
                    },
                },
                "required": ["audio_file_path", "language"],
            },
        }
        self.polly_client = None
        self.openai_client = None

    def __call__(
        self,
        audio_file_path,
        language,
        **kwargs,
    ):
        os.makedirs("media", exist_ok=True)
        if len(kwargs) > 0:
            return f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"

        # create client
        if self.openai_client is None:
            self.openai_client = OpenAI()

        if not os.path.isfile(audio_file_path):
            return f"Audio file not found: {audio_file_path}"

        rng_num = rng.integers(low=0, high=900000)
        target_file = f"media/transcript_{rng_num}.srt"
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1", response_format="srt", file=audio_file
                )
            with open(target_file, "w", encoding="UTF-8") as f:
                f.write(str(transcript))
        except Exception as e:
            return f"Transcription was NOT generated.\nError description: {str(e)}"

        if not os.path.isfile(target_file):
            return "Error: Transcription file was not saved correctly."

        ans = ["<transcript>"]
        ans.append(f"<path_to_file>{target_file}</path_to_file>")
        ans.append("</transcript>")
        return "\n".join(ans)
