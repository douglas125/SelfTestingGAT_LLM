import os

from openai import OpenAI
from .openai import LLM_GPT_OpenAI


class LLM_Gemini(LLM_GPT_OpenAI):
    def __init__(self, model_size):
        """Constructor
        Arguments:
            model_size - Gemini model to use to make LLM calls
        """
        if model_size == "Gemini 2.5 Pro Google":
            self.model_id = "gemini-2.5-pro"
            self.llm_description = "Gemini 2.5 Pro (large LLM) - directly from Google"
            self.price_per_M_input_tokens = 1.25
            self.price_per_M_output_tokens = 10
        elif model_size == "Gemini 2.5 Flash Google":
            self.model_id = "gemini-2.5-flash"
            self.llm_description = (
                "Gemini 2.5 Flash (medium LLM) - directly from Google"
            )
            self.price_per_M_input_tokens = 0.3
            self.price_per_M_output_tokens = 2.5
        elif model_size == "Gemini 2.5 Flash Lite Google":
            self.model_id = "gemini-2.5-flash-lite"
            self.llm_description = (
                "Gemini 2.5 Flash-Lite (small LLM) - directly from Google"
            )
            self.price_per_M_input_tokens = 0.1
            self.price_per_M_output_tokens = 0.4

        try:
            self.openai_client = OpenAI(
                api_key=os.environ.get("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
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
