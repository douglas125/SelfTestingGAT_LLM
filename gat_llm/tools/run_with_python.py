import subprocess


class ToolRunWithPython:
    def __init__(self):
        self.name = "run_with_python"

        self.tool_description = {
            "name": self.name,
            "description": """Changes to target folder and runs a python file with the command line `python <file_name.py>`.
Note that the script will be run in the `test_env` environment. The issued command will be `cd && conda activate test_env && python file_name`.
If the environment does not exist or a package is missing, instruct the user about how to fix it.

This tool returns python_stdout containing the stdout of python with errors returned.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "Name of the Python file to run.",
                    },
                    "exec_folder": {
                        "type": "string",
                        "description": "Folder to change to before executing the Python script.",
                    },
                },
                "required": ["file_name", "exec_folder"],
            },
        }

    def __call__(self, file_name, exec_folder, **kwargs):
        if len(kwargs) > 0:
            return f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"

        all_args = f"cd {exec_folder} && conda activate test_env && python {file_name}"
        p = subprocess.run(all_args, capture_output=True, text=True, shell=True)

        final_ans = ["<python_stdout>"]
        final_ans.append(p.stdout + p.stderr)
        final_ans.append("</python_stdout>")
        return "\n".join(final_ans)
