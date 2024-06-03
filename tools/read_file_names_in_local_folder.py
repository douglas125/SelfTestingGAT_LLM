import os


class ToolReadLocalFolder():
    def __init__(self):
        self.name = 'read_file_names_in_local_folder'

        self.tool_description = {
            'name': self.name,
            'description': """Reads the file names contained in a local folder.

Raises ValueError: if the folder does not exist.""",
            'input_schema': {
                'type': 'object',
                'properties': {
                    'path_to_folder': {
                        'type': 'string',
                        'description': """Local path to the files whose contents should be retrieved. Provide one file per line as in the <example></example>:
<example>
file1.txt
file2.py
subfolder/file3.docx
</example>""",
                    },
                },
                'required': ['path_to_folder']
            }
        }

    def __call__(self, path_to_folder, **kwargs):
        if len(kwargs) > 0:
            return f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"

        if not os.path.isdir(path_to_folder):
            return f"Error: Did not find folder `{path_to_folder}`"

        ans = [os.path.join(path_to_folder, x) for x in os.listdir(path_to_folder)]
        ans = [x for x in ans if os.path.isfile(x)]

        return f"<files><file>{'</file><file>'.join(ans)}</file></files>"
