import os
import json
import base64

import numpy as np
from openai import OpenAI

rng = np.random.default_rng()


class ToolImageEdit:
    def _gen_img_openai(self, image_paths, prompt, target_file):
        if self.openai_client is None:
            self.openai_client = OpenAI()
        response = self.openai_client.images.edit(
            model="gpt-image-1",
            image=[open(x, "rb") for x in image_paths],
            prompt=prompt,
            n=1,
        )
        parsed_ans = json.loads(response.json())
        revised_prompt = parsed_ans["data"][0]["revised_prompt"]
        img_content = base64.b64decode(parsed_ans["data"][0]["b64_json"])
        with open(target_file, "wb") as f:
            f.write(img_content)
        return revised_prompt

    def __init__(self):
        self.name = "edit_image"

        self.tool_description = {
            "name": self.name,
            "description": """Creates an edited or extended image given one or more source images and a prompt. The prompt should be precise and complete. In case of doubt, ask clarification questions to the user before calling this tool.

Raises ValueError: if not able to generate the image.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "image_files": {
                        "type": "string",
                        "description": """A list containing the full path of the images to be edited. Use one file name per line. Up to 16 images can be used. Note that the order matters - the Nth image should be referred to using its number and a brief description.
These files will be used as input. For example:
path/to/image1.png
path/to/image2.jpg
""",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "A complete, descriptive prompt providing precise instructions about how to edit the images and what to generate as output.",
                    },
                },
                "required": ["image_files", "prompt"],
            },
        }
        self.openai_client = None

    def __call__(
        self,
        image_files,
        prompt,
        engine=None,
        **kwargs,
    ):
        os.makedirs("media", exist_ok=True)
        if len(kwargs) > 0:
            return f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"

        rng_num = rng.integers(low=0, high=900000)
        target_file = f"media/edt_image_{rng_num}.png"

        if engine is None:
            engine = "openai-img"
        engine = engine.lower()

        try:
            if engine == "openai-img":
                image_files = [x.strip() for x in image_files.splitlines()]
                for f in image_files:
                    if not os.path.isfile(f):
                        return f"Error: image file not found: {f}. Please check the path."
                self._gen_img_openai(image_files, prompt, target_file)
        except Exception as e:
            return f"Image was NOT generated.\nError description: {str(e)}"

        if not os.path.isfile(target_file):
            return "Error: Image was not saved correctly."

        ans = ["<image>"]
        ans.append(
            f"<used_engine>{engine}</used_engine><path_to_image>{target_file}</path_to_image>"
        )
        ans.append("</image>")
        return "\n".join(ans)
