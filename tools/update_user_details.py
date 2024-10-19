import os


class ToolUpdateUserDetails:
    def __init__(self):
        self.requires_username = True
        self.name = "read_write_user_details"

        self.tool_description = {
            "name": self.name,
            "description": """Reads or writes relevant information about the user being assisted. Always READ before you WRITE. Before saving new data, show all the changes (before/after) to the user and ask for confirmation. Never save or replace information without first confirming with the user that the changes are correct. Use [] instead of <> when creating groups.

Always retrieve the existing information first so you don't forget to save important existing data. When saving, make sure to include all information still relevant, as shown below in  <user_example></user_example>:

<user_example>
If the current user information is:

[user_info]
[name]Julian[/name]
[friends]Karlo, Joseph, Milly[/friends]
[/user_info]

And the user says: Joseph is not my friend. After confirmation, the user information should be updated to:

[user_info]
[name]Julian[/name]
[friends]Karlo, Milly[/friends]
[/user_info]

All previous relevant information has to be sent in the contents.
</user_example>""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["READ", "WRITE"],
                        "description": """One of READ or WRITE, depending whether information about the user should be retrieved or written. Always READ before using WRITE to make sure important information isn't discarded.
Before using WRITE, make sure to show to the user the changes that will be made and that they agree with them""",
                    },
                    "contents": {
                        "type": "string",
                        "description": "If action=WRITE, content that will fully replace the relevant information about the user. Do not forget to write previous data that is still relevant",
                    },
                },
                "required": ["action"],
            },
        }

    def __call__(self, action, username="None", contents="", **kwargs):
        """
        debug = [action, username, contents]
        with open("debug.txt", "w") as f:
            f.write(str(debug))
        """

        if len(kwargs) > 0:
            return f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"

        actions = ["READ", "WRITE"]
        if action not in actions:
            return f"Action must be one of {actions}"

        os.makedirs("user_info", exist_ok=True)
        filename = f"user_info/{username}.txt"

        if action == "READ":
            if os.path.isfile(filename):
                with open(filename, "r", encoding="utf-8") as f:
                    ans = f.read()
                return ans
            else:
                return "No information available."
        elif action == "WRITE":
            with open(filename, "w", encoding="utf-8") as f:
                f.write(contents)
            return "User information saved successfully."
