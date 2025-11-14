""" Set of available and useful LLMs (mostly posted on AWS Bedrock)
"""
import warnings

# outdated
from .llm_providers.aws_bedrock import LLM_Llama13b
from .llm_providers.aws_bedrock import LLM_Llama70b
from .llm_providers.aws_bedrock import LLM_Llama3
from .llm_providers.aws_bedrock import LLM_Claude2_1_Bedrock
from .llm_providers.aws_bedrock import LLM_Claude_Instant_1_2_Bedrock

# current
from .llm_providers.openai import LLM_GPT_OpenAI
from .llm_providers.anthropic import LLM_Claude_Anthropic
from .llm_providers.aws_bedrock import LLM_Claude_Bedrock
from .llm_providers.aws_bedrock import LLM_Mistral_Bedrock
from .llm_providers.aws_bedrock_cohere import LLM_Command_Cohere
from .llm_providers.maritaca import LLM_Maritalk
from .llm_providers.aws_bedrock_nova import LLM_Nova_Bedrock
from .llm_providers.deepseek import LLM_Deepseek
from .llm_providers.grok import LLM_Grok
from .llm_providers.aws_bedrock_via_openai import LLM_Bedrock_OpenAI

# local
from .llm_providers.ollama import LLM_Ollama
from .llm_providers.vllm import LLM_VLLM


warnings.simplefilter("always", DeprecationWarning)


class LLM_Provider:
    outdated_llms = [
        "Claude 3.5 Sonnet - Anthropic",
        "Claude 3.5 Sonnet - Bedrock",
        "Claude 3 Opus - Anthropic",
        "Claude 3 Haiku - Anthropic",
        "Claude 3 Haiku - Bedrock",
        "Claude 3.5 Haiku - Anthropic",
        "Claude 3.5 Haiku - Bedrock",
        "Claude 3 Sonnet - Bedrock",
        "Claude 3 Opus - Bedrock",
        "Claude 4 Sonnet - Anthropic",
        "Claude 4 Sonnet - Bedrock",
        "Claude 4 Opus - Anthropic",
        "Claude 4 Opus - Bedrock",
        "Claude 3.7 Sonnet - Anthropic",
        "Claude 2.1",
        "Claude Instant 1.2",
        "Llama2 13b",
        "Llama2 70b",
        "Llama3 8b instruct",
        "Llama3 70b instruct",
        "Grok2Vision - Grok",
        # OpenAI
        "GPT 5 - OpenAI",
        "GPT 4o - OpenAI",
        "GPT 4.1 - OpenAI",
        "GPT 3.5 - OpenAI",
        "GPT 4o mini - OpenAI",
        # AWS
        "Amazon Nova Micro 1.0 - Bedrock",
        "Amazon Nova Lite 1.0 - Bedrock",
        "Amazon Nova Pro 1.0 - Bedrock",
        # Misc
        "DeepSeekV3 Chat - DeepSeek",
        "Command R - Bedrock",
        "Command RPlus - Bedrock",
        "Mistral Mixtral 8x7B",
        "Mistral Large v1",
        "Llama3_1 8b instruct",
        "Llama3_1 70b instruct",
        "Llama3_1 405b instruct",
        # Qwen
        "Qwen 2.5vl 7b - Ollama",
        "Qwen 2.5vl 3b - Ollama",
    ]

    allowed_llms = [
        # Local
        "Qwen 3 0.6b - Ollama",
        "OpenAI GPT OSS 20b - Ollama",
        "OpenAI GPT OSS 120b - Ollama",
        "Qwen 3 1.7b - Ollama",
        "Qwen 3 4b - Ollama",
        "Qwen 3 8b - Ollama",
        "Qwen 3 14b - Ollama",
        "Qwen 3 Coder 30b - Ollama",
        "Llama4 16x17b - Ollama",
        "Qwen 3vl 8b - Ollama",
        "Qwen 3vl 4b - Ollama",
        "Qwen 3vl 2b - Ollama",
        "DeepSeek R1 14b - Ollama",
        "Qwen 3 1.7b - VLLM",
        # AWS Bedrock via OpenAI API
        "OpenAI GPT OSS 20b - AWSBedrock_OpenAI",
        "OpenAI GPT OSS 120b - AWSBedrock_OpenAI",
        # Grok
        "Grok4 - Grok",
        "Grok4 Fast reasoning - Grok",
        "Grok4 Fast nonreasoning - Grok",
        # Maritaca
        "Sabia3 - Maritaca",
        # OpenAI
        "GPT 5_1 - OpenAI",
        "GPT 5 mini - OpenAI",
        "GPT 5 nano - OpenAI",
        # Anthropic
        "Claude 4.5 Sonnet - Anthropic",
        "Claude 4.5 Sonnet - Bedrock",
        "Claude 4.5 Haiku - Anthropic",
        "Claude 4.5 Haiku - Bedrock",
        "Claude 3.7 Sonnet - Bedrock",
    ]

    def get_llm(bedrock_client, llm):
        """Constructor
        Arguments:
            bedrock_client - Instance of boto3.client(service_name='bedrock-runtime')
                to use when making calls to bedrock models
            llm - which LLM to use. Check LLM_Service.allowed_llms for a list
        """
        if llm in LLM_Provider.outdated_llms:
            warn_msg = f"Selected model is outdated: {llm}. Consider switching to a newer model."
            warnings.warn(warn_msg, DeprecationWarning)

        assert (
            llm in LLM_Provider.allowed_llms + LLM_Provider.outdated_llms
        ), f"LLM has to be one of {LLM_Provider.allowed_llms}"
        if llm == "Claude 2.1":
            return LLM_Claude2_1_Bedrock(bedrock_client)
        elif llm == "Claude Instant 1.2":
            return LLM_Claude_Instant_1_2_Bedrock(bedrock_client)
        elif llm == "Llama2 13b":
            return LLM_Llama13b(bedrock_client)
        elif llm == "Llama2 70b":
            return LLM_Llama70b(bedrock_client)

        # Local - Ollama
        elif llm == "OpenAI GPT OSS 20b - Ollama":
            return LLM_Ollama(model="GPT OSS 20b Ollama")
        elif llm == "OpenAI GPT OSS 120b - Ollama":
            return LLM_Ollama(model="GPT OSS 120b Ollama")
        elif llm == "Llama4 16x17b - Ollama":
            return LLM_Ollama(model="Llama4 16x17b Ollama")
        elif llm == "Qwen 3 0.6b - Ollama":
            return LLM_Ollama(model="Qwen 3 0.6b Ollama")
        elif llm == "Qwen 3 Coder 30b - Ollama":
            return LLM_Ollama(model="Qwen 3 Coder 30b Ollama")
        elif llm == "Qwen 3 1.7b - Ollama":
            return LLM_Ollama(model="Qwen 3 1.7b Ollama")
        elif llm == "Qwen 3 4b - Ollama":
            return LLM_Ollama(model="Qwen 3 4b Ollama")
        elif llm == "Qwen 3 8b - Ollama":
            return LLM_Ollama(model="Qwen 3 8b Ollama")
        elif llm == "Qwen 3 14b - Ollama":
            return LLM_Ollama(model="Qwen 3 14b Ollama")
        elif llm == "Qwen 3vl 8b - Ollama":
            return LLM_Ollama(model="Qwen 3vl 8b Ollama")
        elif llm == "Qwen 3vl 4b - Ollama":
            return LLM_Ollama(model="Qwen 3vl 4b Ollama")
        elif llm == "Qwen 3vl 2b - Ollama":
            return LLM_Ollama(model="Qwen 3vl 2b Ollama")
        elif llm == "Qwen 2.5vl 7b - Ollama":
            return LLM_Ollama(model="Qwen 2.5vl 7b Ollama")
        elif llm == "Qwen 2.5vl 3b - Ollama":
            return LLM_Ollama(model="Qwen 2.5vl 3b Ollama")
        elif llm == "DeepSeek R1 14b - Ollama":
            return LLM_Ollama(model="DeepSeek R1 14b Ollama")

        # Local - vllm
        elif llm == "Qwen 3 1.7b - VLLM":
            return LLM_VLLM(model="Qwen 3 1.7b VLLM")

        # AWS Bedrock via OpenAI API
        elif llm == "OpenAI GPT OSS 20b - AWSBedrock_OpenAI":
            return LLM_Bedrock_OpenAI(model_size="GPT OSS 20b Bedrock")
        elif llm == "OpenAI GPT OSS 120b - AWSBedrock_OpenAI":
            return LLM_Bedrock_OpenAI(model_size="GPT OSS 120b Bedrock")

        # Amazon
        elif llm == "Amazon Nova Micro 1.0 - Bedrock":
            return LLM_Nova_Bedrock(bedrock_client, model_size="Nova_Micro")
        elif llm == "Amazon Nova Lite 1.0 - Bedrock":
            return LLM_Nova_Bedrock(bedrock_client, model_size="Nova_Lite")
        elif llm == "Amazon Nova Pro 1.0 - Bedrock":
            return LLM_Nova_Bedrock(bedrock_client, model_size="Nova_Pro")

        # Maritaca
        elif llm == "Sabia3 - Maritaca":
            return LLM_Maritalk(model_size="Sabia3 Maritaca")

        # Grok
        elif llm == "Grok2Vision - Grok":
            return LLM_Grok(model_size="Grok2Vision xAI")
        elif llm == "Grok4 - Grok":
            return LLM_Grok(model_size="Grok4")
        elif llm == "Grok4 Fast reasoning - Grok":
            return LLM_Grok(model_size="Grok4-fast-reasoning")
        elif llm == "Grok4 Fast nonreasoning - Grok":
            return LLM_Grok(model_size="Grok4-fast-non-reasoning")

        # DeepSeek
        elif llm == "DeepSeekV3 Chat - DeepSeek":
            return LLM_Deepseek(model_size="Deepseek Chat")
        elif llm == "DeepSeekR1 Reasoner - DeepSeek":
            return LLM_Deepseek(model_size="Deepseek Reasoner")

        # Llama 3.1 family
        elif llm == "Llama3_1 8b instruct":
            return LLM_Llama3(bedrock_client, model="Llama3_1 8B Instruct - Bedrock")
        elif llm == "Llama3_1 70b instruct":
            return LLM_Llama3(bedrock_client, model="Llama3_1 70B Instruct - Bedrock")
        elif llm == "Llama3_1 405b instruct":
            return LLM_Llama3(bedrock_client, model="Llama3_1 405B Instruct - Bedrock")

        elif llm == "Llama3 8b instruct":
            return LLM_Llama3(bedrock_client, model="Llama3 8B Instruct - Bedrock")
        elif llm == "Llama3 70b instruct":
            return LLM_Llama3(bedrock_client, model="Llama3 70B Instruct - Bedrock")
        elif llm == "Command R - Bedrock":
            return LLM_Command_Cohere(bedrock_client, model_size="Command R Cohere 1")
        elif llm == "Command RPlus - Bedrock":
            return LLM_Command_Cohere(
                bedrock_client, model_size="Command RPlus Cohere 1"
            )
        elif llm == "Mistral Mixtral 8x7B":
            return LLM_Mistral_Bedrock(bedrock_client, model_size="Mixtral 8x7B v0:1")
        elif llm == "Mistral Large v1":
            return LLM_Mistral_Bedrock(bedrock_client, model_size="Mistral Large v1")

        # Legacy Claude
        elif llm == "Claude 3 Opus - Bedrock":
            return LLM_Claude_Bedrock(bedrock_client, model_size="Opus")
        elif llm == "Claude 3 Sonnet - Bedrock":
            return LLM_Claude_Bedrock(bedrock_client, model_size="Sonnet")
        elif llm == "Claude 3 Haiku - Bedrock":
            return LLM_Claude_Bedrock(bedrock_client, model_size="Haiku")
        elif llm == "Claude 3 Opus - Anthropic":
            return LLM_Claude_Anthropic(model_size="Opus 3 Anthropic")
        elif llm == "Claude 3 Haiku - Anthropic":
            return LLM_Claude_Anthropic(model_size="Haiku 3 Anthropic")

        # current Claude
        elif llm == "Claude 4 Sonnet - Anthropic":
            return LLM_Claude_Anthropic(model_size="Sonnet 4 Anthropic")
        elif llm == "Claude 4.5 Sonnet - Anthropic":
            return LLM_Claude_Anthropic(model_size="Sonnet 4.5 Anthropic")
        elif llm == "Claude 4.5 Haiku - Anthropic":
            return LLM_Claude_Anthropic(model_size="Haiku 4.5 Anthropic")
        elif llm == "Claude 4 Opus - Anthropic":
            return LLM_Claude_Anthropic(model_size="Opus 4 Anthropic")
        elif llm == "Claude 4 Sonnet - Bedrock":
            return LLM_Claude_Bedrock(bedrock_client, model_size="Sonnet 4")
        elif llm == "Claude 4.5 Sonnet - Bedrock":
            return LLM_Claude_Bedrock(bedrock_client, model_size="Sonnet 4.5")
        elif llm == "Claude 4.5 Haiku - Bedrock":
            return LLM_Claude_Bedrock(bedrock_client, model_size="Haiku 4.5")
        elif llm == "Claude 4 Opus - Bedrock":
            return LLM_Claude_Bedrock(bedrock_client, model_size="Opus 4")
        elif llm == "Claude 3.7 Sonnet - Bedrock":
            return LLM_Claude_Bedrock(bedrock_client, model_size="Sonnet 3.7")
        elif llm == "Claude 3.5 Sonnet - Bedrock":
            return LLM_Claude_Bedrock(bedrock_client, model_size="Sonnet 3.5")
        elif llm == "Claude 3.5 Haiku - Bedrock":
            return LLM_Claude_Bedrock(bedrock_client, model_size="Haiku 3.5")
        elif llm == "Claude 3.5 Sonnet - Anthropic":
            return LLM_Claude_Anthropic(model_size="Sonnet 3.5 Anthropic")
        elif llm == "Claude 3.7 Sonnet - Anthropic":
            return LLM_Claude_Anthropic(model_size="Sonnet 3.7 Anthropic")
        elif llm == "Claude 3.5 Haiku - Anthropic":
            return LLM_Claude_Anthropic(model_size="Haiku 3.5 Anthropic")

        # OpenAI
        elif llm == "GPT 5 - OpenAI":
            return LLM_GPT_OpenAI(model_size="GPT5 OpenAI", reasoning_effort="low")
        elif llm == "GPT 5_1 - OpenAI":
            return LLM_GPT_OpenAI(model_size="GPT5_1 OpenAI", reasoning_effort="low")
        elif llm == "GPT 5 mini - OpenAI":
            return LLM_GPT_OpenAI(model_size="GPT5 mini OpenAI", reasoning_effort="low")
        elif llm == "GPT 5 nano - OpenAI":
            return LLM_GPT_OpenAI(model_size="GPT5 nano OpenAI", reasoning_effort="low")
        elif llm == "GPT 4.1 - OpenAI":
            return LLM_GPT_OpenAI(model_size="GPT4_1 OpenAI")
        elif llm == "GPT 3.5 - OpenAI":
            return LLM_GPT_OpenAI(model_size="GPT3_5 OpenAI")
        elif llm == "GPT 4o - OpenAI":
            return LLM_GPT_OpenAI(model_size="GPT4o OpenAI")
        elif llm == "GPT 4o mini - OpenAI":
            return LLM_GPT_OpenAI(model_size="GPT4o mini OpenAI")
