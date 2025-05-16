import os
import subprocess


class ToolSelectVideoFrames:
    def __init__(self):
        self.name = "select_video_frames"

        self.tool_summary = f"""<tool_summary>
<tool_name>{self.name}</tool_name>
<summary>Selects relevant video frames from a video file.</summary>
</tool_summary>"""

        self.tool_description = {
            "name": self.name,
            "description": """Extracts desired video frames from a video. Use this tool when the user asks for a video to be analyzed and the answer requires the analysis of frames extracted from the video. Afterwards, you are allowed to use tools that directly analyze images.

Raises ValueError: if one of the parameters is invalid.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "video_file_path": {
                        "type": "string",
                        "description": "Path to the video file",
                    },
                    "desired_frame_times": {
                        "type": "string",
                        "description": """Desired times from where to extract frames in Hours:Minutes:Seconds. Separate each frame using commas. Example: "00:23:45, 01:43:54" (this will be used as an argument to ffmpeg -ss [time])""",
                    },
                },
                "required": ["video_file_path", "desired_frame_times"],
            },
        }

        self.tool_summary = self.tool_description

    def __call__(
        self, video_file_path, desired_frame_times, output_folder="media", **kwargs
    ):
        if len(kwargs) > 0:
            return f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"

        if not os.path.isfile(video_file_path):
            ans = f"Error: Did not find file `{video_file_path}`"
            ans = f"<error>\n{ans}\n</error>"
            return ans

        list_frame_times = [x.strip() for x in desired_frame_times.split(",")]

        ans = ["Frames extracted:"]
        for t in list_frame_times:
            file_name = f"""frame_{t.replace(":", "_")}.jpg"""
            file_name = os.path.join(output_folder, file_name)

            all_args = (
                f"ffmpeg -y -ss {t} -i {video_file_path} -frames:v 1 -q:v 2 {file_name}"
            )
            subprocess.run(all_args, capture_output=True, text=True, shell=True)

            file_name = f"<frame>{file_name}</frame>"
            ans.append(file_name)

        return "\n".join(ans)
