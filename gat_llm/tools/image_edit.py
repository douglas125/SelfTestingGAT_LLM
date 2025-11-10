import os
import json
import base64
from io import BytesIO

import numpy as np
from PIL import Image
from openai import OpenAI

rng = np.random.default_rng()


def remove_semi_transparent_pixels(image_path: str) -> bytes:
    """
    Opens an RGBA image, sets all pixels with alpha != 255 to fully transparent black (0,0,0,0)
    and returns the resulting PNG image as bytes.

    :param image_path: Path to the input image.
    :return: PNG-encoded image bytes.
    """
    # Load the image as RGBA
    img = Image.open(image_path).convert("RGBA")

    # Convert to NumPy array
    arr = np.array(img)

    # Mask for alpha not equal to 255
    alpha_mask = arr[:, :, 3] != 255
    not_alpha_mask = arr[:, :, 3] == 255

    # Set affected pixels to fully transparent black
    arr[alpha_mask] = [0, 0, 0, 0]
    arr[not_alpha_mask] = [0, 0, 0, 255]

    # Convert back to PIL Image
    result_img = Image.fromarray(arr, mode="RGBA")

    # Save to bytes
    buf = BytesIO()
    result_img.save(buf, format="PNG")
    mask_bytes = buf.getvalue()
    # Save the resulting file
    with open(image_path, "wb") as f:
        f.write(mask_bytes)


class ToolImageEdit:
    def _gen_img_openai(
        self, image_paths, prompt, mask_file, target_file, input_fidelity
    ):
        if self.openai_client is None:
            self.openai_client = OpenAI()

        args = {
            "model": "gpt-image-1",
            "image": [open(x, "rb") for x in image_paths],
            "prompt": prompt,
            "input_fidelity": input_fidelity,
            "n": 1,
            "stream": True,
            "partial_images": 3,
        }
        if mask_file is not None:
            remove_semi_transparent_pixels(mask_file)
            args["mask"] = open(mask_file, "rb")

        response_gen = self.openai_client.images.edit(**args)
        for response in response_gen:
            # parsed_ans = json.loads(response.json())
            # parsed_ans["b64_json"] = ""
            # print(parsed_ans)

            image_base64 = response.b64_json
            img_content = base64.b64decode(image_base64)
            # revised_prompt = parsed_ans["data"][0]["revised_prompt"]
            with open(target_file, "wb") as f:
                f.write(img_content)
            yield f"Generating ..."

    def __init__(self):
        self.name = "edit_image"

        self.tool_description = {
            "name": self.name,
            "description": """Creates an edited or extended image given one or more source images and a prompt. The prompt should be precise and complete. In case of doubt, ask clarification questions to the user before calling this tool. Never mention file names in the prompt because the image generation model only receives the bytes of each image,
When dealing with humans and/or faces, make sure to set input_fidelity to high.
In all cases, including when using a mask, the prompt has to describe the entire resulting image, not just the area that is masked. When in doubt, ask the user to generate or clarify the full description.

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
                    "input_fidelity": {
                        "type": "string",
                        "description": "Control how much effort the model will exert to match the style and features, especially facial features, of input images. Can be high or low. Defaults to low.",
                    },
                    "mask_file": {
                        "type": "string",
                        "description": "Full path to the image that will be used as mask. The fully transparent areas (e.g. where alpha is zero) indicate where image should be edited. If there are multiple images provided, the mask will be applied on the first image.",
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
        mask_file=None,
        engine=None,
        input_fidelity="low",
        **kwargs,
    ):
        os.makedirs("media", exist_ok=True)
        if len(kwargs) > 0:
            yield f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"
            return

        rng_num = rng.integers(low=0, high=900000)
        target_file = f"media/edt_image_{rng_num}.png"
        generation_ans = f"<used_engine>{engine}</used_engine><path_to_image>{target_file}</path_to_image>"

        if engine is None:
            engine = "openai-img"
        engine = engine.lower()

        try:
            if engine == "openai-img":
                image_files = [x.strip() for x in image_files.splitlines()]
                for f in image_files:
                    if not os.path.isfile(f):
                        yield (
                            f"</scratchpad>Error: image file not found: {f}. Please check the path.</scratchpad>"
                        )
                        return
                yield f"<scratchpad>Creating image: {prompt}</scratchpad>"
                img_gen = self._gen_img_openai(
                    image_files, prompt, mask_file, target_file, input_fidelity
                )
                for img in img_gen:
                    yield f"<scratchpad>{generation_ans}</scratchpad>"
        except Exception as e:
            yield f"<scratchpad>Image was NOT generated.\nError description: {str(e)}</scratchpad>"
            return

        if not os.path.isfile(target_file):
            yield "<scratchpad>Error: Image was not saved correctly.</scratchpad>"
            return

        ans = ["<image>"]
        ans.append(generation_ans)
        ans.append("</image>")
        yield "\n".join(ans)
