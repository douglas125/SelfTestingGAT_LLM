import os
import json

import pandas as pd
from tqdm import tqdm

from tools.base import LLMTools
from self_tests.self_test_base import SelfTestBase


class SelfTestPerformer(SelfTestBase):
    """SelfTestPerformer class
    This class enables using specific test case files to check whether the LLMs are calling the correct tools.
    """

    def test_tool_use(self, test_cases, source_file):
        print(f"Testing {self.ref_llm}. Native tools: {self.use_native_LLM_tools}")
        all_results = []
        all_tools = LLMTools.get_all_tools(self.llm)
        li = self.get_llm_interface(all_tools)
        for test_case in tqdm(test_cases):
            try:
                q = ["Consider the following <question></question>:"]
                q.append("<question>")
                q.append(test_case["question"])
                q.append("</question>")
                q.append(
                    "Plan what tool or tools are needed to answer the previous question."
                )
                q.append(
                    "Do NOT use any tools yet - only mention which ones will be necessary. Do not repeat the question."
                )
                q.append("Explain your choices before giving the final answer.")
                q.append(
                    "Your final answer should contain only the tool names separated by colon in <tool_use_plan></tool_use_plan> tags."
                )
                q.append(
                    "Example answers: <tool_use_plan>find_ticker_name</tool_use_plan>, <tool_use_plan>find_ticker_name,retrieve_price</tool_use_plan>"
                )
                q = "\n".join(q)
                ans_gen = li.chat_with_function_caller(q, image=None, ui_history=[])
                for x in ans_gen:
                    pass
                tool_plan_answer = self._extract_answer(x)
                all_results.append(
                    {
                        "model": self.ref_llm,
                        "use_native_tools": self.use_native_LLM_tools,
                        "question": q,
                        "source_file": source_file,
                        "raw_answer": tool_plan_answer,
                        "parsed_tool_names": self._extract_tool_names(tool_plan_answer),
                        "expected_answer": test_case["appropriate_tools"],
                    }
                )
            except Exception as err:
                print(f"Unexpected {err}")
        return all_results

    def _extract_tool_names(self, x):
        ans = x.split("</tool_use_plan>")[0]
        ans = ans.split("<tool_use_plan>")
        if len(ans) > 1:
            return [x.strip() for x in ans[1].split(",")]
        else:
            return []


if __name__ == "__main__":
    # configure tests
    n_test_cases = 2
    llms_to_test = [
        {"model": "Claude 3 Haiku", "native_tools": True},
        # {"model": "Claude 3 Haiku", "native_tools": False},
        # {"model": "Claude 3.5 Sonnet - Anthropic", "native_tools": True},
    ]
    test_files = [
        os.path.join(
            "self_tests", "use_all_test_cases_Claude 3.5 Sonnet - Anthropic.json"
        ),
        os.path.join(
            "self_tests",
            "selected_with_dummies_test_cases_Claude 3.5 Sonnet - Anthropic.json",
        ),
        os.path.join(
            "self_tests", "only_selected_test_cases_Claude 3.5 Sonnet - Anthropic.json"
        ),
    ]
    for llm_spec in llms_to_test:
        result_file_name = os.path.join(
            "self_tests",
            f"self_test_results_{llm_spec['model']}_{llm_spec['native_tools']}.csv",
        )

        if not os.path.isfile(result_file_name):
            test_results = []
            for test_file in test_files:
                with open(test_file, "r", encoding="utf-8") as f:
                    test_cases = json.loads(f.read())
                stp = SelfTestPerformer(llm_spec["model"], llm_spec["native_tools"])
                test_results += stp.test_tool_use(test_cases, test_file)

            df = pd.DataFrame(test_results)
            df.to_csv(result_file_name, index=False)
        else:
            print(f"Skipping {result_file_name}")
