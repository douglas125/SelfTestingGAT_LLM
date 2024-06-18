import subprocess


class ToolUseFFMPEG:
    def __init__(self):
        self.name = "use_ffmpeg"

        self.tool_description = {
            "name": self.name,
            "description": """Runs ffmpeg in the command line to manipulate videos.
If file arguments are used, specify the complete path to the file.
Save output files in the same folder as the input unless the user requests a different folder.
The command line call will be "ffmpeg [ffmpeg_arguments]".

This tool returns ffmpeg_stdout containing the stdout of ffmpeg and ffmpeg_stderr with errors returned.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "ffmpeg_arguments": {
                        "type": "string",
                        "description": """Arguments to use with ffmpeg, as in the <examples></examples>:
<examples>
<example>
<task>Extract a snippet of c:\\videos\\input.mp4 starting at 10 seconds with a duration of 20 seconds. Save the result in c:\\videos\\output.mp4</task>
<ffmpeg_arguments>-i input.mp4 -ss 10 -t 20 output.mp4</ffmpeg_arguments>
</example>
<example>
<task>Speed up input.mp4 by a factor of 2, and save the result in output.mp4</task>
<ffmpeg_arguments>-i input.mp4 -filter_complex "[0:v]setpts=0.5*PTS[v];[0:a]atempo=2.0[a]" -map "[v]" -map "[a]" output.mp4</ffmpeg_arguments>
</example>
</examples>""",
                    },
                },
                "required": ["ffmpeg_arguments"],
            },
        }

    def __call__(self, ffmpeg_arguments, **kwargs):
        if len(kwargs) > 0:
            return f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"

        all_args = "ffmpeg " + ffmpeg_arguments  # ffmpeg_arguments.split()
        p = subprocess.run(all_args, capture_output=True, text=True, shell=True)
        print(all_args)

        final_ans = ["<ffmpeg_stdout>"]
        final_ans.append(p.stdout + p.stderr)
        final_ans.append("</ffmpeg_stdout>")
        return "\n".join(final_ans)
