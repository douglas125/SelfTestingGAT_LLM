import os


class ToolReadLocalFile:
    def __init__(self):
        self.name = "read_local_files"

        self.tool_description = {
            "name": self.name,
            "description": """Reads one or more local file and return its contents. Provide one file per line.

Raises ValueError: if the file does not exist.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path_to_files": {
                        "type": "string",
                        "description": """Local path to the files whose contents should be retrieved. Provide one file per line as in the <example></example>:
<example>
file1.txt
file2.py
subfolder/file3.docx
</example>""",
                    },
                },
                "required": ["path_to_files"],
            },
        }

    def __call__(self, path_to_files, **kwargs):
        if len(kwargs) > 0:
            return f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"

        final_ans = ["<files>"]
        all_files = [x for x in path_to_files.splitlines() if x.strip() != ""]
        for path_to_file in path_to_files.splitlines():

            if not os.path.isfile(path_to_file):
                ans = f"Error: Did not find file `{path_to_file}`"
                ans = f"<error>\n{ans}\n</error>"
            else:
                with open(path_to_file, "r", encoding="utf-8") as f:
                    ans = f.read()
                ans = f"<contents>\n{ans}\n</contents>"

            final_ans.append("<file>")
            final_ans.append(f"<file_name>{path_to_file}</file_name>")
            final_ans.append(ans)
            final_ans.append("</file>")

        final_ans.append("</files>")
        return "\n".join(final_ans)
