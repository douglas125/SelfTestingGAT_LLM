import boto3
import botocore

import llm_invoker as inv
from tools.base import LLMTools
from llm_interface import LLMInterface
from prompts.prompt_generator import RAGPromptGenerator


class SelfTestBase:
    """SelfTest Base class
    This class creates shared LLM call functions for test generation / check
    """

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
        self.llm = inv.LLM_Provider.get_llm(self.bedrock_client, ref_llm)
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

    def _extract_answer(self, x):
        """Extracts the final answer from chatwithfunctioncaller method"""
        ans = x[3][-1]["content"].split("<answer><b>")[-1].split("</answer></b>")[0]
        return ans.replace("\\", "\\\\\\\\")  # adjust backslash
