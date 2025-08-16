""" Note:

Serve vLLM using the command

docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest --model Qwen/Qwen3-1.7B --trust-remote-code --gpu-memory-utilization 0.8 --max_model_len 8192 --enable-auto-tool-choice --tool-call-parser hermes

Since vLLM preallocates GPU and can't offload to CPU, this is a safe command for testing using a 8Gb RAM GPU.
"""
from openai import OpenAI
from .openai import LLM_GPT_OpenAI


class LLM_VLLM(LLM_GPT_OpenAI):
    def __init__(self, model):
        """Constructor
        Arguments:
            model - Desired model
        """
        # there would be electricity cost for running a model locally
        # but for now we keep cost at 0
        self.price_per_M_input_tokens = 0
        self.price_per_M_output_tokens = 0
        if model == "Qwen 3 1.7b VLLM":
            self.model_id = "Qwen/Qwen3-1.7B"
            self.llm_description = "Qwen 3 1.7b (Tiny-size LLM) - locally from vLLM"

        try:
            self.openai_client = OpenAI(
                base_url="http://localhost:8000/v1/",
                api_key="vllm",  # required but ignored
            )
        except Exception:
            self.openai_client = None

        self.config = {
            # "messages": prompt,
            # "system": sysprompt,
            "max_tokens": 2048,
            "temperature": 0.5,  # 0.5 is default,
            "stream": True,
            # "top_k": 250,
            # "top_p": 1,
            "stop": None,  # the regular is already implemented
            "model": self.model_id,
        }
        # requests and answer word count
        self.word_counts = []
