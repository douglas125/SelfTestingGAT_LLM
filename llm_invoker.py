""" Set of available and useful LLMs (mostly posted on AWS Bedrock)
"""
# outdated
from llm_providers.aws_bedrock import LLM_Llama13b
from llm_providers.aws_bedrock import LLM_Llama70b
from llm_providers.aws_bedrock import LLM_Claude2_1_Bedrock
from llm_providers.aws_bedrock import LLM_Mixtral8x7b_Bedrock
from llm_providers.aws_bedrock import LLM_Claude_Instant_1_2_Bedrock

# current
from llm_providers.openai import LLM_GPT_OpenAI
from llm_providers.anthropic import LLM_Claude3_Anthropic
from llm_providers.aws_bedrock import LLM_Claude3_Bedrock


class LLM_Provider:
    allowed_llms = [
        "GPT 4o - OpenAI",
        "GPT 3.5 - OpenAI",
        "Claude 3.5 Sonnet - Anthropic",
        "Claude 3 Haiku",
        "Claude 3 Sonnet",
        "Claude 3 Opus",
        "Claude 2.1",
        "Claude Instant 1.2",
        "Llama2 13b",
        "Llama2 70b",
        "Mistral Mixtral 8x7B",
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
        elif llm == "Mistral Mixtral 8x7B":
            return LLM_Mixtral8x7b_Bedrock(bedrock_client)
        elif llm == "Claude 3 Opus":
            return LLM_Claude3_Bedrock(bedrock_client, model_size="Opus")
        elif llm == "Claude 3 Sonnet":
            return LLM_Claude3_Bedrock(bedrock_client, model_size="Sonnet")
        elif llm == "Claude 3 Haiku":
            return LLM_Claude3_Bedrock(bedrock_client, model_size="Haiku")
        elif llm == "Claude 3.5 Sonnet":
            return LLM_Claude3_Bedrock(bedrock_client, model_size="Sonnet 3.5")
        elif llm == "Claude 3.5 Sonnet - Anthropic":
            return LLM_Claude3_Anthropic(model_size="Sonnet 3.5 Anthropic")
        elif llm == "GPT 3.5 - OpenAI":
            return LLM_GPT_OpenAI(model_size="GPT3_5 OpenAI")
        elif llm == "GPT 4o - OpenAI":
            return LLM_GPT_OpenAI(model_size="GPT4o OpenAI")
