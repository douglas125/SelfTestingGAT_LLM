""" Set of available and useful LLMs (mostly posted on AWS Bedrock)
"""
# outdated
from llm_providers.aws_bedrock import LLM_Llama13b
from llm_providers.aws_bedrock import LLM_Llama70b
from llm_providers.aws_bedrock import LLM_Llama3
from llm_providers.aws_bedrock import LLM_Claude2_1_Bedrock
from llm_providers.aws_bedrock import LLM_Claude_Instant_1_2_Bedrock

# current
from llm_providers.openai import LLM_GPT_OpenAI
from llm_providers.anthropic import LLM_Claude3_Anthropic
from llm_providers.aws_bedrock import LLM_Claude3_Bedrock
from llm_providers.aws_bedrock import LLM_Mistral_Bedrock
from llm_providers.aws_bedrock_cohere import LLM_Command_Cohere


class LLM_Provider:
    allowed_llms = [
        "GPT 4o - OpenAI",
        "GPT 3.5 - OpenAI",
        "GPT 4o mini - OpenAI",
        "Claude 3.5 Sonnet - Anthropic",
        "Claude 3 Opus - Anthropic",
        "Claude 3 Haiku - Anthropic",
        "Claude 3 Haiku - Bedrock",
        "Claude 3 Sonnet - Bedrock",
        "Claude 3.5 Sonnet - Bedrock",
        "Claude 3 Opus - Bedrock",
        "Command R - Bedrock",
        "Command RPlus - Bedrock",
        "Mistral Mixtral 8x7B",
        "Mistral Large v1",
        "Llama3_1 8b instruct",
        "Llama3_1 70b instruct",
        "Llama3_1 405b instruct",
        # Legacy
        "Claude 2.1",
        "Claude Instant 1.2",
        "Llama2 13b",
        "Llama2 70b",
        "Llama3 8b instruct",
        "Llama3 70b instruct",
    ]

    def get_llm(bedrock_client, llm):
        """Constructor
        Arguments:
            bedrock_client - Instance of boto3.client(service_name='bedrock-runtime')
                to use when making calls to bedrock models
            llm - which LLM to use. Check LLM_Service.allowed_llms for a list
        """
        assert (
            llm in LLM_Provider.allowed_llms
        ), f"LLM has to be one of {LLM_Provider.allowed_llms}"
        if llm == "Claude 2.1":
            return LLM_Claude2_1_Bedrock(bedrock_client)
        elif llm == "Claude Instant 1.2":
            return LLM_Claude_Instant_1_2_Bedrock(bedrock_client)
        elif llm == "Llama2 13b":
            return LLM_Llama13b(bedrock_client)
        elif llm == "Llama2 70b":
            return LLM_Llama70b(bedrock_client)

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
        elif llm == "Claude 3 Opus - Bedrock":
            return LLM_Claude3_Bedrock(bedrock_client, model_size="Opus")
        elif llm == "Claude 3 Sonnet - Bedrock":
            return LLM_Claude3_Bedrock(bedrock_client, model_size="Sonnet")
        elif llm == "Claude 3 Haiku - Bedrock":
            return LLM_Claude3_Bedrock(bedrock_client, model_size="Haiku")
        elif llm == "Claude 3.5 Sonnet - Bedrock":
            return LLM_Claude3_Bedrock(bedrock_client, model_size="Sonnet 3.5")
        elif llm == "Claude 3.5 Sonnet - Anthropic":
            return LLM_Claude3_Anthropic(model_size="Sonnet 3.5 Anthropic")
        elif llm == "Claude 3 Opus - Anthropic":
            return LLM_Claude3_Anthropic(model_size="Opus 3 Anthropic")
        elif llm == "Claude 3 Haiku - Anthropic":
            return LLM_Claude3_Anthropic(model_size="Haiku 3 Anthropic")
        elif llm == "GPT 3.5 - OpenAI":
            return LLM_GPT_OpenAI(model_size="GPT3_5 OpenAI")
        elif llm == "GPT 4o - OpenAI":
            return LLM_GPT_OpenAI(model_size="GPT4o OpenAI")
        elif llm == "GPT 4o mini - OpenAI":
            return LLM_GPT_OpenAI(model_size="GPT4o mini OpenAI")
