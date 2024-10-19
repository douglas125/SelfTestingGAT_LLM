import pytest

from ..tools.summarize_past import ToolSummarizePast


def test_unexpected_arg(unexpected_param_msg):
    tsp = ToolSummarizePast()
    ans = tsp("past", "info", "next", unexpected_argument=None)
    assert ans == f"{unexpected_param_msg}unexpected_argument"


@pytest.mark.parametrize("past_conversation_summary", ["past1", "past2", "past3"])
@pytest.mark.parametrize("information_still_relevant", ["info1", "info2", "info3"])
@pytest.mark.parametrize("next_question_to_answer", ["next1", "next2"])
def test_complete_ans(
    past_conversation_summary,
    information_still_relevant,
    next_question_to_answer,
):
    """Checks if the answer actually contains the arguments
    (expected for this function)
    """
    tsp = ToolSummarizePast()
    ans = tsp(
        past_conversation_summary,
        information_still_relevant,
        next_question_to_answer,
    )
    assert past_conversation_summary in ans
    assert information_still_relevant in ans
    assert next_question_to_answer in ans
