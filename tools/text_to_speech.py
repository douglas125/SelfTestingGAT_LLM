import os

import boto3
import numpy as np
from openai import OpenAI

from contextlib import closing

rng = np.random.default_rng()

# picking neural voices
VOICE_MAP = {
    ("english", "female"): "Vitoria",
    ("english", "male"): "Gregory",
    ("portuguese", "female"): "Camila",
    ("portuguese", "male"): "Thiago",
    ("spanish", "female"): "Lucia",
    ("spanish", "male"): "Sergio",
    ("italian", "female"): "Bianca",
    ("italian", "male"): "Adriano",
}


class ToolTextToSpeech:
    def __init__(self):
        self.name = "text_to_speech"

        self.tool_description = {
            "name": self.name,
            "description": """Uses a text-to-speech tool to convert the provided text into spoken audio in the given language.
You need to specify the text that will be spoken, the language and the speaker gender. The language is used to select the appropriate text-to-speech model.

The user can listen to the audio without exposing the auto-generated file names. Do NOT include actual file names in the answer. Do NOT include the <path_to_audio></path_to_audio> tag in the answer. Only mention that the image has been generated successfully.

Raises ValueError: if not able to generate the audio.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "input_text": {
                        "type": "string",
                        "description": "Text that will be converted to speech.",
                    },
                    "language": {
                        "type": "string",
                        "enum": ["english", "portuguese", "spanish", "italian"],
                        "description": "Language of the input_text. Must be one of english, portuguese, spanish or italian",
                    },
                    "speaker_gender": {
                        "type": "string",
                        "enum": ["male", "female"],
                        "description": "Desired gender of the speaker. Default is female. Can be male or female.",
                    },
                    "tts_engine": {
                        "type": "string",
                        "enum": ["aws_polly", "openai"],
                        "description": """Must be aws_polly or openai.
Select the neural engine that will be used to generate the audio. Defaults to OpenAI text to speech. If not available, you must manually select the other.""",
                    },
                },
                "required": ["input_text", "language"],
            },
        }
        self.polly_client = None
        self.openai_client = None

    def __call__(
        self,
        input_text,
        language,
        speaker_gender="female",
        tts_engine="openai",
        **kwargs,
    ):
        os.makedirs("media", exist_ok=True)
        if len(kwargs) > 0:
            return f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"

        # pick TTS engine
        tts_engine = tts_engine.lower().strip()
        valid_tts_engines = ["aws_polly", "openai"]
        assert (
            tts_engine in valid_tts_engines
        ), f"Invalid text to speech engine: {tts_engine}. Must be one of {valid_tts_engines}."
        if tts_engine == "openai" and self.openai_client is None:
            self.openai_client = OpenAI()
        elif tts_engine == "aws_polly" and self.polly_client is None:
            self.polly_client = boto3.client(
                service_name="polly", region_name="us-west-2"
            )

        rng_num = rng.integers(low=0, high=900000)
        target_file = f"media/audio_{rng_num}.mp3"

        try:
            if tts_engine == "aws_polly":
                # Request speech synthesis
                response = self.polly_client.synthesize_speech(
                    Text=input_text,
                    OutputFormat="mp3",
                    VoiceId=VOICE_MAP[
                        (language.lower(), speaker_gender.lower())
                    ],  # "Danielle",
                    Engine="neural",
                )
                if "AudioStream" in response:
                    with closing(response["AudioStream"]) as stream:
                        with open(target_file, "wb") as file:
                            file.write(stream.read())
            elif tts_engine == "openai":
                response = self.openai_client.audio.speech.create(
                    model="tts-1",
                    voice="onyx" if speaker_gender == "male" else "nova",
                    input=input_text,
                )

                response.stream_to_file(target_file)
        except Exception as e:
            return f"Audio was NOT generated.\nError description: {str(e)}"

        if not os.path.isfile(target_file):
            return "Error: Audio was not saved correctly."

        ans = ["<audio>"]
        ans.append(f"<path_to_audio>{target_file}</path_to_audio>")
        ans.append("</audio>")
        return "\n".join(ans)
