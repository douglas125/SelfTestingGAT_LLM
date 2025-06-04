import os

from openai import OpenAI
from .openai import LLM_GPT_OpenAI


class LLM_Grok(LLM_GPT_OpenAI):
    def __init__(self, model_size):
        """Constructor
        Arguments:
            model_size - Instance of boto3.client(service_name='bedrock-runtime')
                to use when making calls to bedrock models
        """
        if model_size == "Grok2Vision xAI":
            self.model_id = "grok-2-vision-1212"
            self.llm_description = "Grok 2 (medium-sized LLM) - directly from xAI"
            self.price_per_M_input_tokens = 2
            self.price_per_M_output_tokens = 10

        try:
            self.openai_client = OpenAI(
                api_key=os.environ.get("GROK_API_KEY"),
                base_url="https://api.x.ai/v1",
            )
        except Exception:
            self.openai_client = None

        self.config = {
            # "messages": prompt,
            # "system": sysprompt,
            "max_tokens": 4000,
            "temperature": 0.5,  # 0.5 is default,
            "stream": True,
            # "top_k": 250,
            # "top_p": 1,
            "stop": None,  # the regular is already implemented
            "model": self.model_id,
        }
        # requests and answer word count
        self.word_counts = []
