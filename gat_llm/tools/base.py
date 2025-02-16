import time
import xml.etree.ElementTree as ET

import numpy as np

from .make_custom_plot import ToolMakeCustomPlot
from .solve_symbolic import ToolSolveSymbolic
from .solve_numeric import ToolSolveNumeric
from .solve_python_code import ToolSolvePythonCode
from .get_webpage_contents import ToolGetUrlContent
from .make_qr_code import ToolMakeQRCode
from .read_local_file import ToolReadLocalFile
from .write_local_file import ToolWriteLocalFile
from .read_file_names_in_local_folder import ToolReadLocalFolder
from .do_date_math import ToolDoDateMath
from .update_user_details import ToolUpdateUserDetails
from .use_ffmpeg import ToolUseFFMPEG
from .plot_with_graphviz import ToolPlotWithGraphviz
from .text_to_speech import ToolTextToSpeech
from .speech_to_text import ToolSpeechToText
from .summarize_past import ToolSummarizePast
from .text_to_image import ToolTextToImage
from .query_database import ToolQueryLLMDB
from .query_database import SampleOrder_LLM_DB

rng = np.random.default_rng()


class LLMTools:
    def get_all_tools():
        """Returns a list of all tools available"""
        return [
            ToolDoDateMath(),
            ToolUpdateUserDetails(),
            ToolMakeCustomPlot(),
            ToolSolveSymbolic(),
            ToolSolveNumeric(),
            ToolGetUrlContent(None),
            ToolMakeQRCode(),
            ToolReadLocalFile(),
            ToolWriteLocalFile(),
            ToolReadLocalFolder(),
            ToolUseFFMPEG(),
            ToolSolvePythonCode(),
            ToolPlotWithGraphviz(),
            ToolTextToSpeech(),
            ToolSpeechToText(),
            ToolTextToImage(),
            ToolSummarizePast(),
            ToolQueryLLMDB(SampleOrder_LLM_DB()),
        ]

    def __init__(self, query_llm=None, desired_tools=None):
        """Constructor."""
        self.query_llm = query_llm
        if desired_tools is None:
            self.tools = [
                ToolDoDateMath(),
                ToolMakeCustomPlot(),
                ToolSolveSymbolic(),
                ToolSolveNumeric(),
                ToolGetUrlContent(self.query_llm),
                ToolMakeQRCode(),
                ToolReadLocalFile(),
                ToolWriteLocalFile(),
                ToolReadLocalFolder(),
                ToolUseFFMPEG(),
                ToolPlotWithGraphviz(),
                # Being left out for now. Just uncomment to enable
                # ToolSummarizePast(),
                # ToolUpdateUserDetails(),
                # ToolQueryLLMDB(SampleOrder_LLM_DB()),
                # ToolSolvePythonCode(),
                ToolTextToSpeech(),
                ToolSpeechToText(),
                ToolTextToImage(),
            ]
        else:
            self.tools = desired_tools

        self.tool_mapping = dict(zip([x.name for x in self.tools], self.tools))
        self.call_return_string = """
<function_results>
<result>
<tool_name>{{TOOLNAME}}</tool_name>
<stdout>
{{TOOLRESULTS}}
</stdout>
</result>
</function_results>
<scratchpad> To answer the question, I still need to:"""
        self.invoke_log = []

    def invoke_from_cmd(self, xml_cmd, username=None):
        """Invokes a tool given the xml command sent by the LLM"""
        cmd = self.parse_command(xml_cmd)
        cmd["parameters"]["username"] = str(username)
        return self.invoke_tool(cmd["tool_name"], **cmd["parameters"])

    def parse_command(self, xml_cmd):
        """Parses a XML command to retrieve arguments and tool name"""
        try:
            assert (
                "&" not in xml_cmd
            ), "Do not include the character & in any of the arguments."

            root = ET.fromstring(xml_cmd)
            ans = {}
            func_params = {}
            for child in root:
                if child.tag == "parameters":
                    for x in child:
                        func_params[x.tag] = x.text
                if child.tag == "tool_name":
                    ans["tool_name"] = child.text
            ans["parameters"] = func_params
            return ans
        except Exception as e:
            print(str(e))
            return {"tool_name": "unknown", "parameters": {}, "error": str(e)}

    def get_tool_descriptions(self):
        """Retrieves the description of all tools available"""
        desc_list = ["<tools>"]
        for x in self.tools:
            desc_list.append(self._parse_tool_description(x.tool_description))
        desc_list.append("</tools>")
        return "\n".join(desc_list)

    def _parse_tool_description(self, description_dict):
        """Given a tool description in JSON, parse its contents"""
        textual_desc = {}
        textual_desc["tool_name"] = description_dict["name"]
        textual_desc["description"] = description_dict["description"]

        # parameters
        textual_desc["parameters"] = []
        cur_desc_dict = description_dict["input_schema"]["properties"]
        for k in cur_desc_dict:
            cur_param = {"parameter": {}}
            cur_param["parameter"] = {"name": k}
            for k2 in cur_desc_dict[k]:
                cur_param["parameter"][k2] = str(cur_desc_dict[k][k2])
            textual_desc["parameters"].append(self._json2xml(cur_param))
        textual_desc["required_parameters"] = str(
            description_dict["input_schema"]["required"]
        )

        return self._json2xml({"tool_description": textual_desc})

    def _json2xml(self, json_obj, line_padding=""):
        # https://stackoverflow.com/questions/8988775/convert-json-to-xml-in-python
        result_list = list()

        json_obj_type = type(json_obj)

        if json_obj_type is list:
            for sub_elem in json_obj:
                result_list.append(self._json2xml(sub_elem, line_padding))
            return "\n".join(result_list)

        if json_obj_type is dict:
            for tag_name in json_obj:
                sub_obj = json_obj[tag_name]
                result_list.append("%s<%s>" % (line_padding, tag_name))
                result_list.append(self._json2xml(sub_obj, line_padding))
                result_list.append("%s</%s>" % (line_padding, tag_name))
            return "\n".join(result_list)

        return "%s%s" % (line_padding, json_obj)

    def invoke_tool(self, tool_name, return_results_only=False, **kwargs):
        try:
            cur_tool = self.tool_mapping.get(tool_name)
            # if 'error' in cur_tool.keys():
            #     return f'Error: {cur_tool["error"]}'
            if cur_tool is not None:
                t0 = time.time()
                if (
                    not hasattr(cur_tool, "requires_username")
                    or not cur_tool.requires_username
                ):
                    kwargs.pop("username", None)

                ans = cur_tool(**kwargs)
                self.invoke_log.append(
                    {
                        "tool_name": tool_name,
                        "execution_time": time.time() - t0,
                        "result_length": len(ans),
                    }
                )
                if return_results_only:
                    return ans
                else:
                    return self.call_return_string.replace(
                        "{{TOOLNAME}}", tool_name
                    ).replace("{{TOOLRESULTS}}", ans)
            else:
                return f"Tool {tool_name} not found. Please check tool name. Available tools: {self.tool_mapping.keys()}"
        except Exception as e:
            print(f"Tool execution failed: {str(e)}")
            return str(
                e
            )  # "Failed to invoke tool. Please try again, possibly in a different way."
