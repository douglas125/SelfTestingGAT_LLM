import os

import numpy as np
import pandas as pd
import parselmouth
from openai import OpenAI

rng = np.random.default_rng()


def fill_zeros_with_last(arr):
    prev = np.arange(len(arr))
    prev[arr <= 0] = 0
    prev = np.maximum.accumulate(prev)
    return arr[prev]


def combine_transcription_with_speech_params(transcription_dict, speech_params):
    """
    Combines the transcription with speech parameters

    Args:
        transcription_dict: OpenAI style transcript dict with key "words" and a list of type
            'words': [{'end': 2.4, 'start': 2.3, 'word': 'Hello'},
        speech_params: dictionary with keys time_s and the various voice parameters (see analyze_voice)
    """
    # accumulate the values to make it efficient to compute averages
    accum_params = {}
    delta_t = speech_params["time_s"][1] - speech_params["time_s"][0]
    for k in speech_params:
        accum_params[f"accum_{k}"] = (
            np.cumsum(fill_zeros_with_last(np.nan_to_num(speech_params[k]))) * delta_t
        )
    accum_params["time_s"] = speech_params["time_s"]

    # extract the words and their start / stop times
    word_start_stops = np.array(
        [[x["start"], x["end"]] for x in transcription_dict["words"]]
    )
    word_durations = word_start_stops[:, 1] - word_start_stops[:, 0]

    all_words = [
        {
            "word": x["word"],
            "start_time_s": float(np.round(x["start"], decimals=1)),
            "end_time_s": float(np.round(x["end"], decimals=1)),
        }
        for x in transcription_dict["words"]
    ]

    for k in speech_params:
        if k != "time_s":
            cur_avg = np.interp(
                word_start_stops, accum_params["time_s"], accum_params[f"accum_{k}"]
            )
            cur_avg = np.divide(cur_avg[:, 1] - cur_avg[:, 0], word_durations + 1e-5)
            for i in range(len(all_words)):
                all_words[i][k] = float(np.round(cur_avg[i], decimals=1))

    df = pd.DataFrame(all_words)
    return df.to_xml(
        index=False,
        xml_declaration=False,
        root_name="speech_and_features",
        row_name="word_and_features",
    )


def analyze_voice(
    input_wav, frame_step=0.01, window=0.2, pitch_floor=75, pitch_ceiling=450
):
    """Analyzes voice using Praat.
    Note that it would be better to customize pitch floor and ceiling to gender / children
    """
    snd = parselmouth.Sound(input_wav)

    # === Fundamental frequency (F0) ===
    pitch = snd.to_pitch(
        time_step=frame_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling
    )
    f0_values = pitch.selected_array["frequency"]  # Hz, 0 = unvoiced
    f0_times = pitch.xs()

    # === Harmonicity (HNR) ===
    harmonicity = snd.to_harmonicity_cc(time_step=frame_step, minimum_pitch=pitch_floor)
    hnr_values = harmonicity.values.flatten()  # 1D array
    hnr_times = harmonicity.xs()

    # === Intensity ===
    intensity = snd.to_intensity(time_step=frame_step, minimum_pitch=pitch_floor)
    intensity_values = intensity.values.flatten()  # dB
    intensity_times = intensity.xs()

    # === Point process (for jitter & shimmer) ===
    point_process = parselmouth.praat.call(
        snd,
        "To PointProcess (periodic, peaks)",
        pitch_floor,
        pitch_ceiling,
        "yes",
        "no",
    )

    jitters = []
    shimmers = []

    for t in f0_times:
        start = max(0, t - window / 2)
        end = min(snd.xmax, t + window / 2)
        try:
            jitter_local = parselmouth.praat.call(
                point_process, "Get jitter (local)", start, end, 0.0001, 0.02, 1.3
            )
            shimmer_local = parselmouth.praat.call(
                [snd, point_process],
                "Get shimmer (local)",
                start,
                end,
                0.0001,
                0.02,
                1.3,
                1.6,
            )
        except Exception:
            jitter_local = np.nan
            shimmer_local = np.nan

        jitters.append(jitter_local)
        shimmers.append(shimmer_local)

    # === Align HNR and Intensity to F0 time grid ===
    hnr_interp = np.interp(f0_times, hnr_times, hnr_values, left=np.nan, right=np.nan)
    intensity_interp = np.interp(
        f0_times, intensity_times, intensity_values, left=np.nan, right=np.nan
    )

    ans = {
        "time_s": f0_times,
        "f0_hz": f0_values,
        "hnr_db": hnr_interp,
        "intensity_db": intensity_interp,
        "jitter_percentage": [x * 100 for x in jitters],
        "shimmer_db": shimmers,
    }
    return ans


class ToolSpeechAnalysis:
    def __init__(self):
        self.name = "speech_analysis"

        self.tool_description = {
            "name": self.name,
            "description": """Uses an automatic speech recognition tool to convert the provided audio into text.
Uses Praat to analyze frequency, intensity, SNR and other speech parameters. Each word is provided along with its start time, duration, and
Returns: path to a file containing a JSON description of the audio, along with various useful parameters.
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
                        "description": "Language of the audio in file audio_file_path. Should be provided in ISO-639-1 (e.g. en, pt, es) format. Defaults to en",
                    },
                },
                "required": ["audio_file_path"],
            },
        }
        self.openai_client = None

    def __call__(
        self,
        audio_file_path,
        language="en",
        return_path_to_file_only=True,
        **kwargs,
    ):
        """Creates a transcription of an audio file along with speech parameters.

        Arguments:
          - audio_file_path: Path to the file that should be transcribed
          - language: language in ISO format (e.g. en, pt, es, it)
          - return_path_to_file_only: if True, writes a transcription file and returns it. If false, returns the transcription itself
        Returns:
          - The audio srt file or the full transcription
        """
        os.makedirs("media", exist_ok=True)
        if len(kwargs) > 0:
            return f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"

        # create client
        if self.openai_client is None:
            self.openai_client = OpenAI()

        if not os.path.isfile(audio_file_path):
            return f"Audio file not found: {audio_file_path}"

        rng_num = rng.integers(low=0, high=900000)
        target_file = f"media/transcript_{rng_num}.xml"
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                )
            transcript_json = transcript.dict()

            # Use Praat to extract relevant information
            speech_params = analyze_voice(audio_file_path)
            xml_content = combine_transcription_with_speech_params(
                transcript_json, speech_params
            )

            if return_path_to_file_only:
                with open(target_file, "w", encoding="UTF-8") as f:
                    f.write(xml_content)
            else:
                return transcript
        except Exception as e:
            return f"Transcription was NOT generated.\nError description: {str(e)}"

        if not os.path.isfile(target_file):
            return "Error: Transcription file was not saved correctly."

        ans = ["<transcript>"]
        ans.append(f"<path_to_file>{target_file}</path_to_file>")
        ans.append("</transcript>")
        return "\n".join(ans)
