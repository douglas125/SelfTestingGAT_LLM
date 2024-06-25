import os

from tools.base import LLMTools
from self_tests.self_test_base import SelfTestBase


class SelfTestPerformer(SelfTestBase):
    """SelfTestPerformer class
    This class enables using specific test case files to check whether the LLMs are calling the correct tools.
    """

    def test_tool_use(self, test_cases):
        pass


if __name__ == "__main__":
    # configure tests
    n_test_cases = 2
    llms_to_test = ["Claude 3 Haiku"]  # , "Claude 3 Sonnet"]
    test_files = [
        os.path.join(
            "self_tests", "selected_with_dummies_test_cases_Claude 3 Sonnet.json"
        )
    ]

    # Note: if a function call fails, it may be the case that the LLM will call it multiple times
    # we will use the lt.invoke_log to track which tools were used. It needs to be cleared between LLM calls
