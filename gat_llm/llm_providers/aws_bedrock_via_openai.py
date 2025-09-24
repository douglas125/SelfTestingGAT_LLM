import os

from openai import OpenAI
from .openai import LLM_GPT_OpenAI


class LLM_Bedrock_OpenAI(LLM_GPT_OpenAI):
    def __init__(self, model_size):
        """Constructor
        Arguments:
            model_size - Bedrock model to use to make LLM calls
        """

        if model_size == "GPT OSS 20b Bedrock":
            self.model_id = "openai.gpt-oss-20b-1:0"
            self.llm_description = (
                "GPT OSS 20b (small-sized LLM) - directly from Bedrock using OpenAI API"
            )
            self.price_per_M_input_tokens = 0.07
            self.price_per_M_output_tokens = 0.3
        if model_size == "GPT OSS 120b Bedrock":
            self.model_id = "openai.gpt-oss-120b-1:0"
            self.llm_description = "GPT OSS 120b (medium-sized LLM) - directly from Bedrock using OpenAI API"
            self.price_per_M_input_tokens = 0.15
            self.price_per_M_output_tokens = 0.6

        try:
            self.openai_client = OpenAI(
                api_key=os.environ.get("AWS_BEDROCK_API_KEY"),
                base_url="https://bedrock-runtime.us-west-2.amazonaws.com/openai/v1",
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
