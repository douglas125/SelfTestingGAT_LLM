# python -m self_tests.self_test_performer
import os
import json

import pandas as pd
from tqdm import tqdm

from gat_llm.tools.base import LLMTools
from self_tests.self_test_base import SelfTestBase


class SelfTestPerformer(SelfTestBase):
    """SelfTestPerformer class
    This class enables using specific test case files to check whether the LLMs are calling the correct tools.
    """

    def test_tool_use(self, test_cases, source_file):
        print(f"Testing {self.ref_llm}. Native tools: {self.use_native_LLM_tools}")
        all_results = []
        all_tools = LLMTools.get_all_tools()
        li = self.get_llm_interface(all_tools)
        progbar = tqdm(test_cases)
        for test_case in progbar:
            try:
                q = ["Consider the following <question></question>:"]
                q.append("<question>")
                q.append(test_case["question"])
                q.append("</question>")
                q.append(
                    "Plan what tool or tools are needed to answer the previous question."
                )
                q.append(
                    "Do NOT use any tools yet - only mention which ones will be necessary. Do not repeat the question. Do not make tool calls."
                )
                q.append("Explain your choices before giving the final answer.")
                q.append(
                    "Your final answer should contain only the tool names separated by colon in <tool_use_plan></tool_use_plan> tags."
                )
                q.append(
                    "Example answers: <tool_use_plan>find_ticker_name</tool_use_plan>, <tool_use_plan>find_ticker_name,retrieve_price</tool_use_plan>"
                )
                q = "\n".join(q)
                ans_gen = li.chat_with_function_caller(q, images=None, ui_history=[])
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
                progbar.set_description(f"Results so far: {len(all_results)}")
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
    llms_to_test = [
        {"model": "Nemotron 3 Nano 30b - Ollama", "native_tools": True},
        {"model": "Claude 4.5 Haiku - Anthropic", "native_tools": True},
        {"model": "Claude 4.5 Sonnet - Anthropic", "native_tools": True},
        {"model": "GPT 5 nano - OpenAI", "native_tools": True},
        {"model": "GPT 5 mini - OpenAI", "native_tools": True},
        {"model": "GPT 5 - OpenAI", "native_tools": True},
        {"model": "OpenAI GPT OSS 20b - Ollama", "native_tools": True},
        {"model": "Qwen 3 8b - Ollama", "native_tools": True},
        {"model": "Qwen 3 14b - Ollama", "native_tools": True},
        # {"model": "DeepSeek R1 14b - Ollama", "native_tools": False},
        {"model": "Amazon Nova Micro 1.0 - Bedrock", "native_tools": True},
        {"model": "Amazon Nova Lite 1.0 - Bedrock", "native_tools": True},
        {"model": "Amazon Nova Pro 1.0 - Bedrock", "native_tools": True},
        {"model": "Claude 3 Haiku - Bedrock", "native_tools": True},
        {"model": "Claude 3 Haiku - Bedrock", "native_tools": False},
        {"model": "Llama3_1 8b instruct", "native_tools": False},
        {"model": "Llama3_1 70b instruct", "native_tools": False},
        {"model": "Llama3_1 405b instruct", "native_tools": False},
        {"model": "Llama3 70b instruct", "native_tools": False},
        {"model": "Llama3 8b instruct", "native_tools": False},
        {"model": "Claude 3.7 Sonnet - Anthropic", "native_tools": True},
        {"model": "Claude 3.5 Sonnet - Anthropic", "native_tools": True},
        {"model": "Claude 3.5 Sonnet - Anthropic", "native_tools": False},
        {"model": "Claude 3.5 Haiku - Anthropic", "native_tools": True},
        {"model": "Claude 3.5 Haiku - Anthropic", "native_tools": False},
        {"model": "GPT 3.5 - OpenAI", "native_tools": True},
        {"model": "GPT 3.5 - OpenAI", "native_tools": False},
        {"model": "GPT 4o mini - OpenAI", "native_tools": True},
        {"model": "GPT 4o mini - OpenAI", "native_tools": False},
        {"model": "GPT 4.1 - OpenAI", "native_tools": True},
        {"model": "GPT 4o - OpenAI", "native_tools": True},
        {"model": "GPT 4o - OpenAI", "native_tools": False},
        {"model": "Mistral Mixtral 8x7B", "native_tools": False},
        {"model": "Mistral Large v1", "native_tools": False},
        {"model": "Command R - Bedrock", "native_tools": False},
        {"model": "Command RPlus - Bedrock", "native_tools": False},
        {"model": "Sabia3 - Maritaca", "native_tools": True},
        {"model": "Sabia3 - Maritaca", "native_tools": False},
        {"model": "DeepSeekV3 Chat - DeepSeek", "native_tools": False},
        {"model": "Grok2Vision - Grok", "native_tools": True},
    ]
    test_files = [
        os.path.join(
            "self_tests",
            "selected_with_dummies_test_cases_Claude 3.5 Sonnet - Anthropic.json",
        ),
        os.path.join(
            "self_tests",
            "selected_with_dummies_test_cases_GPT 4o - OpenAI.json",
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
