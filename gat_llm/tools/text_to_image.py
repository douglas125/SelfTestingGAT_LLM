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
        response_gen = self.openai_client.images.generate(
            model="gpt-image-1.5",
            prompt=input_text,
            n=1,
            moderation="low",
            output_format="png",
            stream=True,
            partial_images=3,
        )
        for response in response_gen:
            image_base64 = response.b64_json
            img_content = base64.b64decode(image_base64)
            with open(target_file, "wb") as f:
                f.write(img_content)
            yield f"Generating ..."

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
            yield f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"
            return

        rng_num = rng.integers(low=0, high=900000)
        target_file = f"media/gen_image_{rng_num}.png"
        generation_ans = f"<used_engine>{engine}</used_engine><path_to_image>{target_file}</path_to_image>"

        if engine is None:
            if os.getenv("OPENAI_API_KEY"):
                engine = "openai-img"
            else:
                engine = "bedrock-stablediffusion"
        engine = engine.lower()

        if engine not in self.valid_engines:
            yield f"Invalid text-to-image engine: {engine}. Valid engines are: {self.valid_engines}"
            return

        try:
            if engine == "openai-img":
                yield f"<scratchpad>Creating image: {input_text}</scratchpad>"
                img_gen = self._gen_img_openai(input_text, target_file)
                for img in img_gen:
                    yield f"<scratchpad>{generation_ans}</scratchpad>"
            elif engine == "bedrock-stablediffusion":
                self._gen_img_bedrock(input_text, target_file)
        except Exception as e:
            yield f"Image was NOT generated.\nError description: {str(e)}"
            return

        if not os.path.isfile(target_file):
            yield "Error: Image was not saved correctly."
            return

        ans = ["<image>"]
        ans.append(generation_ans)
        ans.append("</image>")
        yield "\n".join(ans)
