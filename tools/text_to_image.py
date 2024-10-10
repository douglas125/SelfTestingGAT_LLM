import os
import json
import base64

import numpy as np
from openai import OpenAI

rng = np.random.default_rng()


class ToolTextToImage:
    def __init__(self):
        self.name = "text_to_image"

        self.tool_description = {
            "name": self.name,
            "description": """Uses a text-to-image tool to convert the provided text into an image. Uses neural methods such as diffusion.
You need to specify the text that will be converted to the image.

Raises ValueError: if not able to generate the image.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "input_text": {
                        "type": "string",
                        "description": "Text that will be converted to image.",
                    },
                },
                "required": ["input_text"],
            },
        }
        self.openai_client = OpenAI()

    def __call__(
        self,
        input_text,
        **kwargs,
    ):
        os.makedirs("media", exist_ok=True)
        if len(kwargs) > 0:
            return f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"

        rng_num = rng.integers(low=0, high=900000)
        target_file = f"media/gen_image_{rng_num}.png"

        try:
            response = self.openai_client.images.generate(
                model="dall-e-3",
                prompt=input_text,
                n=1,
                size="1024x1024",
                response_format="b64_json",
            )
            parsed_ans = json.loads(response.json())
            revised_prompt = parsed_ans["data"][0]["revised_prompt"]
            img_content = base64.b64decode(parsed_ans["data"][0]["b64_json"])
            with open(target_file, "wb") as f:
                f.write(img_content)
        except Exception as e:
            return f"Image was NOT generated.\nError description: {str(e)}"

        if not os.path.isfile(target_file):
            return "Error: Image was not saved correctly."

        ans = ["<image>"]
        ans.append(
            f"<revised_prompt>{revised_prompt}</revised_prompt><path_to_image>{target_file}</path_to_image>"
        )
        ans.append("</image>")
        return "\n".join(ans)
