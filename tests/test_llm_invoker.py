import json

from unittest.mock import MagicMock
from unittest.mock import Mock

import pytest

from gat_llm.llm_invoker import LLM_Provider


@pytest.mark.parametrize(
    "outdated_llm",
    [
        ("Claude 3 Opus - Anthropic"),
        ("Claude 3 Haiku - Anthropic"),
        ("Claude 3 Haiku - Bedrock"),
        ("Claude 3 Sonnet - Bedrock"),
        ("Claude 3 Opus - Bedrock"),
        ("Claude 2.1"),
        ("Claude Instant 1.2"),
        ("Llama2 13b"),
        ("Llama2 70b"),
        ("Llama3 8b instruct"),
        ("Llama3 70b instruct"),
    ],
)
def test_outdated_llm_warning(outdated_llm):
    # Ensures that outdated models show warnings
    with pytest.deprecated_call():
        bedrock_client = None
        LLM_Provider.get_llm(bedrock_client, outdated_llm)


@pytest.mark.parametrize("llm", LLM_Provider.allowed_llms + LLM_Provider.outdated_llms)
def test_model_creation(llm):
    # Checks if the models are created successfully
    # (even without the API key)
    bedrock_client = None
    LLM_Provider.get_llm(bedrock_client, llm)


def dummy_response_gen(text, postpend=""):
    # just mocks the yield part of the response
    if isinstance(text, str):
        yield text
    else:
        yield "Dummy LLM generation"


@pytest.mark.parametrize(
    "llm_name", LLM_Provider.allowed_llms + LLM_Provider.outdated_llms
)
def test_if_llm_responds(llm_name):
    # Checks if the llms actually produce an answer when required
    bedrock_client = None
    llm = LLM_Provider.get_llm(bedrock_client, llm_name)

    # mocks the clients
    llm.anthropic_client = MagicMock()
    llm.bedrock_client = MagicMock()
    llm.openai_client = MagicMock()

    # mocks the invoke methods
    llm.anthropic_client.messages.create = Mock(return_value="Dummy LLM generation")
    llm.openai_client.chat.completions.create = Mock(
        return_value="Dummy LLM generation"
    )
    llm.bedrock_client.invoke_model_with_response_stream = Mock(
        return_value={
            "body": [
                {
                    "chunk": {
                        "bytes": json.dumps(
                            {
                                "generation": "Dummy LLM generation",
                                "outputs": [{"text": "Dummy LLM generation"}],
                                "completion": "Dummy LLM generation",
                                "stop": "",
                            }
                        )
                    }
                }
            ]
        }
    )

    llm.cur_tool_spec = None
    llm.stop_reason = None
    llm._response_gen = dummy_response_gen
    ans = llm("Dummy message")
    for x in ans:
        pass
    assert x == "Dummy LLM generation"
