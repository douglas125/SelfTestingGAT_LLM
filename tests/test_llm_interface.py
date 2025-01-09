import types
from unittest.mock import Mock

from gat_llm.llm_interface import LLMInterface


def test_return_any_answer():
    llm = Mock(return_value=["<scratchpad>Thoughts</scratchpad>Bot response"])
    llm.tool_use_added_msgs = []
    llm.word_counts = [2]
    rpg = None
    llm_tools = None

    li = LLMInterface("You are a helpful assistant", llm, llm_tools, rpg)
    response_ui = li.chat_with_function_caller(
        "Hello", None, ui_history=[], username=""
    )
    assert isinstance(
        response_ui, types.GeneratorType
    ), "UI Response has to be a generator"

    for x in response_ui:
        pass

    _, scratchpad_info, _, cur_history = x

    assert scratchpad_info == "Thoughts", "Scratchpad info not identified"
    assert cur_history[-1] == {
        "role": "assistant",
        "content": "Bot response\n",
    }, "Unexpected bot response"
