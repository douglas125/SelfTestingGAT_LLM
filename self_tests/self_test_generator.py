import boto3
import botocore

import llm_invoker as inv
from tools.base import LLMTools
from llm_interface import LLMInterface
from prompts.prompt_generator import RAGPromptGenerator


class SelfTestGenerator:
    def __init__(self, ref_llm="Claude 3 Haiku", use_native_LLM_tools=True):
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
        lt = LLMTools(query_llm=self.llm)

        tool_descriptions = lt.get_tool_descriptions()
        rpg = RAGPromptGenerator(use_native_tools=self.use_native_LLM_tools)
        system_prompt = rpg.prompt.replace("{{TOOLS}}", tool_descriptions)
        li = LLMInterface(
            system_prompt=system_prompt, llm=self.llm, llm_tools=lt, rpg=rpg
        )
        return li

    def gen_test_cases(self, n_test_cases=2, tool_strategy="use_all"):
        """Generate tool use test cases for this RAG-LLM application"""
        possible_strategies = ["use_all", "only_selected", "selected_with_dummy"]
        assert (
            tool_strategy in possible_strategies
        ), f"Strategy must be one of {possible_strategies}"

        # retrieve all available tools
        all_tools = LLMTools.get_all_tools(self.llm)
        if tool_strategy == "use_all":
            li = self.get_llm_interface(all_tools)
            q = """Analyze carefully your tools.
For each tool, generate a set of NTESTCASES questions that can be answered using exactly that tool and no other.
Then, generate a set of NTESTCASES questions that can be answered using exactly two tools and no other. Give a list of the tool names in the order they should be used.
The format of your answer should be in JSON format enclosed within <answer></answer>, as in example below:

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
</answer>
""".replace(
                "NTESTCASES", str(n_test_cases)
            )
            ans_gen = li.chat_with_function_caller(q, image=None, ui_history=[])
            for x in ans_gen:
                pass
            test_cases = x[3][-1][1]
            test_cases = test_cases.split("<answer><b>")[-1].split("</answer></b>")[0]
            return test_cases
        else:
            pass


if __name__ == "__main__":
    stg = SelfTestGenerator()
    ans = stg.gen_test_cases()
    with open("test_cases.json", "w") as f:
        f.write(str(ans))
    print("test_cases.json")
