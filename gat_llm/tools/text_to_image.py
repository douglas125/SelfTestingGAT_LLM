import os
import json
import boto3
import base64

import numpy as np
from openai import OpenAI

rng = np.random.default_rng()


class ToolTextToImage:
    def _gen_img_bedrock(self, input_text, target_file, default_region="us-west-2"):
        if self.bedrock_client is None:
            self.bedrock_client = boto3.client(
                "bedrock-runtime", region_name=default_region
            )
        model_id = "stability.stable-image-ultra-v1:0"

        # seed = int(rng.integers(low=0, high=4294967295))
        native_request = {
            "prompt": input_text,
        }
        request = json.dumps(native_request)
        response = self.bedrock_client.invoke_model(modelId=model_id, body=request)
        output_body = json.loads(response["body"].read().decode("utf-8"))
        base64_output_image = output_body["images"][0]

        # Extract the image data.
        image_data = base64.b64decode(base64_output_image)
        with open(target_file, "wb") as file:
            file.write(image_data)
        return input_text

    def _gen_img_openai(self, input_text, target_file):
        if self.openai_client is None:
            self.openai_client = OpenAI()
        response = self.openai_client.images.generate(
            model="gpt-image-1.5",
            prompt=input_text,
            n=1,
            moderation="low",
            output_format="png",
        )
        parsed_ans = json.loads(response.json())
        revised_prompt = parsed_ans["data"][0]["revised_prompt"]
        img_content = base64.b64decode(parsed_ans["data"][0]["b64_json"])
        with open(target_file, "wb") as f:
            f.write(img_content)
        return revised_prompt

    def __init__(self):
        self.name = "text_to_image"

        self.valid_engines = ["openai-img", "bedrock-stablediffusion"]
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
                    "engine": {
                        "type": "string",
                        "description": f"Engine that will be used to generate the image. Valid values are: {self.valid_engines}. Default value is {self.valid_engines[0]}",
                    },
                },
                "required": ["input_text"],
            },
        }
        self.openai_client = None
        self.bedrock_client = None

    def __call__(
        self,
        input_text,
        engine=None,
        **kwargs,
    ):
        os.makedirs("media", exist_ok=True)
        if len(kwargs) > 0:
            return f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"

        rng_num = rng.integers(low=0, high=900000)
        target_file = f"media/gen_image_{rng_num}.png"

        if engine is None:
            if os.getenv("OPENAI_API_KEY"):
                engine = "openai-img"
            else:
                engine = "bedrock-stablediffusion"
        engine = engine.lower()

        if engine not in self.valid_engines:
            return f"Invalid text-to-image engine: {engine}. Valid engines are: {self.valid_engines}"

        try:
            if engine == "openai-img":
                revised_prompt = self._gen_img_openai(input_text, target_file)
            elif engine == "bedrock-stablediffusion":
                revised_prompt = self._gen_img_bedrock(input_text, target_file)
        except Exception as e:
            return f"Image was NOT generated.\nError description: {str(e)}"

        if not os.path.isfile(target_file):
            return "Error: Image was not saved correctly."

        ans = ["<image>"]
        ans.append(
            f"<revised_prompt>{revised_prompt}</revised_prompt><used_engine>{engine}</used_engine><path_to_image>{target_file}</path_to_image>"
        )
        ans.append("</image>")
        return "\n".join(ans)
