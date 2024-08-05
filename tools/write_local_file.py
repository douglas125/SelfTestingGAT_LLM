import os


class ToolWriteLocalFile:
    def __init__(self):
        self.name = "write_local_files"

        self.tool_description = {
            "name": self.name,
            "description": """Write a text file in the local file system.
Only write text files, like files with extensions .txt, .py, .md and others that usually contain only text.
Do not attempt to write files that are usually in binary format. If the path is not specified, write the file to the media/ folder.

Raises ValueError: if the file does not exist.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path_to_file": {
                        "type": "string",
                        "description": "Local path to the file which will be written.",
                    },
                    "text_content": {
                        "type": "string",
                        "description": "Content to write in the file (text).",
                    },
                },
                "required": ["path_to_file", "text_content"],
            },
        }

    def __call__(self, path_to_file, text_content, **kwargs):
        os.makedirs("media", exist_ok=True)
        if len(kwargs) > 0:
            return f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"

        final_ans = ["<outcome>"]
        if not os.path.isfile(path_to_file):
            with open(path_to_file, "w", encoding="latin-1") as f:
                f.write(text_content)
                final_ans.append("File written successfully")
                final_ans.append(f"<path_to_file>{path_to_file}</path_to_file>")
        else:
            final_ans.append(
                f"Error: File already exists: {path_to_file}. This tool is not allowed to overwrite files."
            )
        final_ans.append("</outcome>")
        return "\n".join(final_ans)
