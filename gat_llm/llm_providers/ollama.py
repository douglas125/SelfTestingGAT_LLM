from openai import OpenAI
from .openai import LLM_GPT_OpenAI


class LLM_Ollama(LLM_GPT_OpenAI):
    def __init__(self, model):
        """Constructor
        Arguments:
            model - Desired model
        """
        # there would be electricity cost for running a model locally
        # but for now we keep cost at 0
        self.price_per_M_input_tokens = 0
        self.price_per_M_output_tokens = 0
        if model == "DeepSeek R1 14b Ollama":
            self.model_id = "deepseek-r1:14b"
            self.llm_description = (
                "DeepSeek R1 14b (Tiny-size LLM) - locally from Ollama"
            )
        elif model == "GPT OSS 20b Ollama":
            self.model_id = "gpt-oss:20b"
            self.llm_description = (
                "GPT OSS 20b OpenAI (Small-size LLM) - locally from Ollama"
            )
        elif model == "GPT OSS 120b Ollama":
            self.model_id = "gpt-oss:120b"
            self.llm_description = (
                "GPT OSS 120b OpenAI (Small-size LLM) - locally from Ollama"
            )
        elif model == "Qwen 3 Coder 30b Ollama":
            self.model_id = "qwen3-coder:30b"
            self.llm_description = "Qwen 3 Coder (Small-size LLM) - locally from Ollama"
        elif model == "Qwen 3vl 8b Ollama":
            self.model_id = "qwen3-vl:8b"
            self.llm_description = "Qwen 3vl 8b (Tiny-size VLLM) - locally from Ollama"
        elif model == "Qwen 3vl 4b Ollama":
            self.model_id = "qwen3-vl:4b"
            self.llm_description = "Qwen 3vl 4b (Tiny-size VLLM) - locally from Ollama"
        elif model == "Qwen 3vl 2b Ollama":
            self.model_id = "qwen3-vl:2b"
            self.llm_description = "Qwen 3vl 2b (Tiny-size VLLM) - locally from Ollama"
        elif model == "Qwen 2.5vl 7b Ollama":
            self.model_id = "qwen2.5vl:7b"
            self.llm_description = (
                "Qwen 2.5vl 7b (Tiny-size VLLM) - locally from Ollama"
            )
        elif model == "Qwen 2.5vl 3b Ollama":
            self.model_id = "qwen2.5vl:3b"
            self.llm_description = (
                "Qwen 2.5vl 3b (Tiny-size VLLM) - locally from Ollama"
            )
        elif model == "Qwen 3 0.6b Ollama":
            self.model_id = "qwen3:0.6b"
            self.llm_description = "Qwen 3 0.6b (Tiny-size LLM) - locally from Ollama"
        elif model == "Qwen 3 1.7b Ollama":
            self.model_id = "qwen3:1.7b"
            self.llm_description = "Qwen 3 1.7b (Tiny-size LLM) - locally from Ollama"
        elif model == "Qwen 3 4b Ollama":
            self.model_id = "qwen3:4b"
            self.llm_description = "Qwen 3 4b (Small-size LLM) - locally from Ollama"
        elif model == "Qwen 3 8b Ollama":
            self.model_id = "qwen3:8b"
            self.llm_description = "Qwen 3 8b (Small-size LLM) - locally from Ollama"
        elif model == "Qwen 3 14b Ollama":
            self.model_id = "qwen3:14b"
            self.llm_description = "Qwen 3 14b (Small-size LLM) - locally from Ollama"
        elif model == "Llama4 16x17b Ollama":
            # this model may be too big for many systems
            self.model_id = "llama4:16x17b"
            self.llm_description = (
                "Llama 4 16x17b (Small-size LLM) - locally from Ollama"
            )

        try:
            self.openai_client = OpenAI(
                base_url="http://localhost:11434/v1/",
                api_key="ollama",  # required but ignored
            )
        except Exception:
            self.openai_client = None

        self.config = {
            # "messages": prompt,
            # "system": sysprompt,
            "max_tokens": 4096,
            "temperature": 0.5,  # 0.5 is default,
            "stream": True,
            # "top_k": 250,
            # "top_p": 1,
            "stop": None,  # the regular is already implemented
            "model": self.model_id,
        }
        # requests and answer word count
        self.word_counts = []
