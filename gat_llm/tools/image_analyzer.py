import os
import base64
import numpy as np

from PIL import Image
from io import BytesIO


rng = np.random.default_rng()


class ToolImageAnalyzer:
    """Tool for analyzing and describing images content."""

    def __init__(self, query_llm=None):
        self.name = "analyze_images"
        self.query_llm = query_llm
        self.tool_description = {
            "name": self.name,
            "description": """Analyzes the content of images and returns a detailed description.
This tool can identify objects, scenes, people, text, activities, and other visual elements in the images.

Call this tool only once per interaction, providing all paths to the images at once.
Raises ValueError: if the images cannot be processed.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path_to_images": {
                        "type": "string",
                        "description": """Path to the images whose contents should be retrieved. Provide all the paths of the images at once. Provide one image per line as in the <example></example>:
<example>
image1.jpg
subfolder/image2.png
</example>""",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Prompt that can be used to directly query and retrieve information from the images, without analyzing their full content. Always provide in English language.",
                    },
                    "items_to_identify": {
                        "type": "string",
                        "description": "List of relevant specific items (e.g. texts, figures, tables, objects, people) to focus attention on. The list of items should be based on the user's question and image file names. Provide one enumerated item per line sorted by relevance. Always provide in English language.",
                    },
                },
                "required": ["path_to_images", "prompt", "items_to_identify"],
            },
        }
        self.system_prompt = """You will analyze the content of an image and returns a detailed description.

Highlight items in <items_to_identify></items_to_identify> within the image to understand their content and correlate it with the user's query. Prioritize high-contrast regions for precise delimitation.
<items_to_identify>
[[ITEMS]]
</items_to_identify>
Ignore irrelevant regions to direct the model's attention to areas of interest.
Link each identified items in the image with a specific ID (e.g. #1, #2).
List the identified items individually and describe them comprehensively.

Do not invent any information. Only use information that can be found in the image.
"""

    def _load_image(self, path_to_images: str):
        """Load image from path."""
        try:
            if os.path.exists(path_to_images):
                return Image.open(path_to_images)
            else:
                raise ValueError(f"Image not found at path: {path_to_images}")
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _analyze_with_llm(
        self,
        path_to_image: str,
        prompt: str,
        items_to_identify: str,
    ) -> str:
        """Analyze image using LLM model."""
        # Load and encode image for API
        image = self._load_image(path_to_image)
        base64_image = self._encode_image_to_base64(image)

        # replace strings
        system_prompt = self.system_prompt.replace("[[ITEMS]]", items_to_identify)

        # Call LLM
        llm_ans = self.query_llm(
            prompt,
            system_prompt=system_prompt,
            b64images=[base64_image],
        )

        for x in llm_ans:
            yield x

    def __call__(
        self,
        path_to_images,
        prompt,
        items_to_identify,
        **kwargs,
    ):
        if len(kwargs) > 0:
            yield f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"
            return

        final_ans = ["<image_descriptions>"]
        all_images = [x.strip() for x in path_to_images.splitlines() if x.strip() != ""]

        for path_to_image in all_images:
            ans = ""
            if not os.path.isfile(path_to_image):
                ans = f"Error: Did not find image `{path_to_image}`"
                ans = f"<error>\n{ans}\n</error>"
                yield f"<scratchpad>{ans}</scratchpad>"
            else:
                ans_gen = self._analyze_with_llm(
                    path_to_image, prompt, items_to_identify
                )
                for ans in ans_gen:
                    yield f"<scratchpad>{ans}</scratchpad>"

            final_ans.append("<image_description>")
            final_ans.append(f"<path_to_image>{path_to_image}</path_to_image>")
            final_ans.append(ans)
            final_ans.append("</image_description>")

        final_ans.append("</image_descriptions>")
        final_ans = "\n".join(final_ans)
        yield final_ans
