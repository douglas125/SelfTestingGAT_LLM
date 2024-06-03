import os
import time
import datetime
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from tools.make_custom_plot import ToolMakeCustomPlot
from tools.solve_symbolic import ToolSolveSymbolic
from tools.solve_numeric import ToolSolveNumeric
from tools.solve_python_code import ToolSolvePythonCode
from tools.get_webpage_contents import ToolGetUrlContent
from tools.make_qr_code import ToolMakeQRCode
from tools.read_local_file import ToolReadLocalFile
from tools.read_file_names_in_local_folder import ToolReadLocalFolder
from tools.do_date_math import ToolDoDateMath
from tools.update_user_details import ToolUpdateUserDetails
rng = np.random.default_rng()


class LLMTools:
    def __init__(self, query_llm=None):
        """ Constructor.
        """
        self.query_llm = query_llm
        self.tools = [
            ToolDoDateMath(),

            # ToolUpdateUserDetails(),
            # ToolMakeCustomPlot(),
            # ToolSolveSymbolic(),
            # ToolSolveNumeric(),
            # ToolGetUrlContent(self.query_llm),
            # ToolMakeQRCode(),

            # Being left out for now
            # ToolSolvePythonCode(),
            # ToolReadLocalFile(),
            # ToolReadLocalFolder(),
        ]

        self.tool_mapping = dict(zip(
            [x.name for x in self.tools],
            self.tools
        ))
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
        """ Invokes a tool given the xml command sent by the LLM
        """
        cmd = self.parse_command(xml_cmd)
        cmd['parameters']['username'] = str(username)
        return self.invoke_tool(cmd['tool_name'], **cmd['parameters'])

    def parse_command(self, xml_cmd):
        """ Parses a XML command to retrieve arguments and tool name
        """
        try:
            assert '&' not in xml_cmd, 'Do not include the character & in any of the arguments.'

            root = ET.fromstring(xml_cmd)
            ans = {}
            func_params = {}
            for child in root:
                if child.tag == 'parameters':
                    for x in child:
                        func_params[x.tag] = x.text
                if child.tag == 'tool_name':
                    ans['tool_name'] = child.text
            ans['parameters'] = func_params
            return ans
        except Exception as e:
            print(str(e))
            return {'tool_name': 'unknown', 'parameters': {}, 'error': str(e)}

    def get_tool_descriptions(self):
        """ Retrieves the description of all tools available
        """
        desc_list = ['<tools>']
        for x in self.tools:
            desc_list.append(self._parse_tool_description(x.tool_description))
        desc_list.append('</tools>')
        return '\n'.join(desc_list)

    def _parse_tool_description(self, description_dict):
        """ Given a tool description in JSON, parse its contents
        """
        textual_desc = {}
        textual_desc['tool_name'] = description_dict['name']
        textual_desc['description'] = description_dict['description']

        # parameters
        textual_desc['parameters'] = []
        cur_desc_dict = description_dict['input_schema']['properties']
        for k in cur_desc_dict:
            cur_param = {'parameter': {}}
            cur_param['parameter'] = {'name': k}
            for k2 in cur_desc_dict[k]:
                cur_param['parameter'][k2] = str(cur_desc_dict[k][k2])
            textual_desc['parameters'].append(self._json2xml(cur_param))
        textual_desc['required_parameters'] = str(description_dict['input_schema']['required'])

        return self._json2xml({'tool_description': textual_desc})

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

    def invoke_tool(self, tool_name, **kwargs):
        try:
            cur_tool = self.tool_mapping.get(tool_name)
            # if 'error' in cur_tool.keys():
            #     return f'Error: {cur_tool["error"]}'
            if cur_tool is not None:
                t0 = time.time()
                if not hasattr(cur_tool, 'requires_username') or not cur_tool.requires_username:
                    kwargs.pop('username', None)

                ans = cur_tool(**kwargs)
                self.invoke_log.append({
                    'tool_name': tool_name,
                    'execution_time': time.time() - t0,
                    'result_length': len(ans),
                })
                return self.call_return_string.replace(
                    '{{TOOLNAME}}', tool_name
                ).replace(
                    '{{TOOLRESULTS}}', ans
                )
            else:
                return f"Tool {tool_name} not found. Please check tool name. Available tools: {self.tool_mapping.keys()}"
        except Exception as e:
            print(f'Tool execution failed: {str(e)}')
            return str(e)  # "Failed to invoke tool. Please try again, possibly in a different way."


class RAGPromptGenerator:
    def __init__(self, use_native_tools=False):
        """ Constructor.

        Arguments:

        - use_native_tools: Use LLM native tool calling abilities. In this case, we do not
            need to "teach" the LLM about tools
        """
        self.prompt = ""
        if not use_native_tools:
            with open(os.path.join('prompts', "prompt_GAT.txt"),
                      'r', encoding='utf-8') as f:
                self.prompt += f.read()

        # base prompt
        with open(os.path.join('prompts', "prompt_base.txt"),
                  'r', encoding='utf-8') as f:
            self.prompt += f.read()

        dt0 = datetime.datetime.today()
        weekday = dt0.strftime('%A')

        date_str = f'{weekday}, {dt0.year}-{str(dt0.month).zfill(2)}-{str(dt0.day).zfill(2)} in the format YYYY-MM-DD'
        date_str = f"""
<today_date>
<day_of_week>{weekday}</day_of_week>
<day>{str(dt0.day).zfill(2)}</day>
<month>{str(dt0.month).zfill(2)}</month>
<year>{dt0.year}</year>

</today_date>
        """

        self.prompt = self.prompt.replace('{{DATE}}', date_str)

        self.post_anti_hallucination = "<scratchpad> I can only use functions that have been explicitly provided. I must follow the <tool_guidelines></tool_guidelines>. I may need the following tools:"