import os
import json

import boto3
import botocore

import llm_invoker as inv
from tools.base import LLMTools
from llm_interface import LLMInterface
from prompts.prompt_generator import RAGPromptGenerator


class SelfTestGenerator:
    def __init__(self, ref_llm, use_native_LLM_tools=True):
        """Constructor.
        Arguments:
            ref_llm: Reference LLM that will be used to generate the self-assessment questions
                Ideally, a LLM with a large amount of parameters that works well with function
                calling.
            use_native_LLM_tools: Use native LLM tools instead of manual specification
        """
        self.config = botocore.client.Config(
            connect_timeout=9000, read_timeout=9000, region_name="us-west-2"
        )
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime", config=self.config
        )
        self.ref_llm = ref_llm
        self.llm = inv.LLM_Bedrock.get_llm(self.bedrock_client, ref_llm)
        self.use_native_LLM_tools = use_native_LLM_tools

    def get_llm_interface(self, tool_subset):
        """Initializes the tools with a subset.
        Note that we do not want to actually let the LLM call tools
        because we just want to generate self-evaluation questions

        Arguments:
            - tool_subset: desired subset of tools to be used

        Returns: a LLM Interface to be used with
            .chat_with_function_caller(q, image=None, ui_history=[])
            where q is a query
        """
        lt = LLMTools(query_llm=self.llm, desired_tools=tool_subset)

        tool_descriptions = lt.get_tool_descriptions()
        rpg = RAGPromptGenerator(use_native_tools=self.use_native_LLM_tools)
        system_prompt = rpg.prompt.replace("{{TOOLS}}", tool_descriptions)
        li = LLMInterface(
            system_prompt=system_prompt, llm=self.llm, llm_tools=lt, rpg=rpg
        )
        return li

    def gen_test_cases(self, n_test_cases, tool_strategy):
        """Generate tool use test cases for this RAG-LLM application
        Arguments:
            tool_strategy: one of
                use_all: ask to generate use cases giving all tools in a single prompt
                only_selected: ask to generate use cases giving only one tool at a time, multiple prompts
                selected_with_dummies: ask to generate use cases giving all tools, but specifically asking in multiple prompts for specific tools
        """
        # TODO: It is interesting that despite the fact that we provide an answer format,
        # the LLM still gets creative in its output format.

        possible_strategies = ["use_all", "only_selected", "selected_with_dummies"]
        assert (
            tool_strategy in possible_strategies
        ), f"Strategy must be one of {possible_strategies}"

        # retrieve all available tools
        all_tools = LLMTools.get_all_tools(self.llm)
        answer_format_prompt = """Never invoke any tools. The format of your answer should be in valid JSON format enclosed within <answer></answer>, as in example below. The keys of each dictionary in the list should match exactly the example.

<answer>
[
    {
        "question": "What day will it be 10 days from today?",
        "appropriate_tools": [
            "do_date_math"
        ]
    },
    {
        "question": "Speed up all videos in folder C:\\Videos by a factor of 2",
        "appropriate_tools": [
            "read_file_names_in_local_folder",
            "use_ffmpeg"
        ]
    }
]
</answer>"""

        if tool_strategy == "use_all":
            q = """Analyze carefully your tools.
For each tool, generate a set of NTESTCASES questions that can be answered using exactly that tool and no other.
The total number of questions to be generated is NTESTCASES * (number of tools), which means NTESTCASES per tool.
Then, generate a set of NTESTCASES questions that can be answered using exactly two tools and no other.

Give a list of the tool names in the order they should be used. The questions should be as different as possible from each other.
""".replace(
                "NTESTCASES", str(n_test_cases)
            )
            q = q + answer_format_prompt

            li = self.get_llm_interface(all_tools)
            ans_gen = li.chat_with_function_caller(q, image=None, ui_history=[])
            for x in ans_gen:
                pass
            test_cases = json.loads(self._extract_answer(x))
            return json.dumps(test_cases)
        elif tool_strategy == "only_selected":
            test_cases = []
            q = """Analyze carefully your tool.
Generate a set of NTESTCASES questions that can be answered using exactly the tool. The questions should be as different as possible from each other.
""".replace(
                "NTESTCASES", str(n_test_cases)
            )
            q = q + answer_format_prompt

            # individual tests
            for cur_tool in all_tools:
                print(f"Test cases with {cur_tool.name}")
                li = self.get_llm_interface([cur_tool])
                ans_gen = li.chat_with_function_caller(q, image=None, ui_history=[])
                for x in ans_gen:
                    pass
                try:
                    cur_case = self._extract_answer(x)
                    cur_case = json.loads(cur_case)
                    test_cases += cur_case
                except Exception as ex:
                    print(x, str(ex))

            # pairwise tests
            q = """Analyze carefully your tools.
Generate a set of NTESTCASES questions that can be answered using exactly those tools.
All tools should be necessary to provide the answer.
The questions should be as different as possible from each other. Never generate questions that would require only one tool to be answered.
""".replace(
                "NTESTCASES", str(n_test_cases)
            )
            q = q + answer_format_prompt
            for idx1, t1 in enumerate(all_tools[:-1]):
                for t2 in all_tools[idx1 + 1 :]:
                    print(idx1, f"Test cases with {t1.name} and {t2.name}")
                    cur_tools = [t1, t2]
                    li = self.get_llm_interface(cur_tools)
                    ans_gen = li.chat_with_function_caller(q, image=None, ui_history=[])
                    for x in ans_gen:
                        pass
                    try:
                        cur_case = self._extract_answer(x)
                        cur_case = json.loads(cur_case)
                        test_cases += cur_case
                    except Exception as ex:
                        print(x, str(ex))

            return json.dumps(test_cases)

        elif tool_strategy == "selected_with_dummies":
            test_cases = []
            q = """Analyze carefully your tools.
Generate a set of NTESTCASES questions that would need to use the tool CURRENTTOOL to be answered correctly. The tool CURRENTTOOL must be sufficient to answer the question and no other tools should be necessary.
The questions should be as different as possible from each other.
""".replace(
                "NTESTCASES", str(n_test_cases)
            )
            q = q + answer_format_prompt

            # individual tests
            for cur_tool in all_tools:
                li = self.get_llm_interface(all_tools)
                cur_q = q.replace("CURRENTTOOL", cur_tool.name)
                print(f"Test cases with {cur_tool.name}")
                ans_gen = li.chat_with_function_caller(cur_q, image=None, ui_history=[])
                for x in ans_gen:
                    pass
                try:
                    cur_case = self._extract_answer(x)
                    cur_case = json.loads(cur_case)
                    test_cases += cur_case
                except Exception as ex:
                    print(x, str(ex))

            # pairwise tests
            q = """Analyze carefully your tools.
Generate a set of NTESTCASES questions that would need to use the tools CURRENTTOOL1 and CURRENTTOOL2 to be answered correctly. The tools CURRENTTOOL1 and CURRENTTOOL2 must be sufficient to answer the question and no other tools should be necessary.
The questions should be as different as possible from each other. Never generate questions that would require only one tool to be answered.
""".replace(
                "NTESTCASES", str(n_test_cases)
            )
            q = q + answer_format_prompt
            for idx1, t1 in enumerate(all_tools[:-1]):
                for t2 in all_tools[idx1 + 1 :]:
                    cur_q = q.replace("CURRENTTOOL1", t1.name)
                    cur_q = cur_q.replace("CURRENTTOOL2", t2.name)
                    print(idx1, f"Test cases with {t1.name} and {t2.name}")
                    cur_tools = [t1, t2]
                    li = self.get_llm_interface(all_tools)
                    ans_gen = li.chat_with_function_caller(
                        cur_q, image=None, ui_history=[]
                    )
                    for x in ans_gen:
                        pass
                    try:
                        cur_case = self._extract_answer(x)
                        cur_case = json.loads(cur_case)
                        test_cases += cur_case
                    except Exception as ex:
                        print(x, str(ex))

            return json.dumps(test_cases)

    def _extract_answer(self, x):
        """Extracts the final answer from chatwithfunctioncaller method"""
        ans = x[3][-1][1].split("<answer><b>")[-1].split("</answer></b>")[0]
        return ans.replace("\\", "\\\\\\\\")  # adjust backslash


if __name__ == "__main__":
    # configure tests
    n_test_cases = 2
    llms_to_test = ["Claude 3 Haiku"]  # , "Claude 3 Sonnet"]
    tool_strategies = ["use_all", "only_selected", "selected_with_dummies"]
    for cur_llm_name in llms_to_test:
        stg = SelfTestGenerator(cur_llm_name)
        for strategy in tool_strategies:
            out_file = os.path.join(
                "self_tests", f"{strategy}_test_cases_{stg.ref_llm}.json"
            )
            if not os.path.isfile(out_file):
                ans = stg.gen_test_cases(n_test_cases, strategy)
                with open(out_file, "w") as f:
                    f.write(str(ans))
            print(out_file)