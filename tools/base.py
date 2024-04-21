import os
import time
import datetime
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
rng = np.random.default_rng()

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


class LLMTools:
    def __init__(self, query_llm=None):
        """ Constructor.
        """
        self.query_llm = query_llm
        self.tools = [
            ToolUpdateUserDetails(),
            ToolMakeCustomPlot(),
            ToolSolveSymbolic(),
            ToolSolveNumeric(),
            # ToolSolvePythonCode(),
            ToolGetUrlContent(self.query_llm),
            ToolMakeQRCode(),
            ToolDoDateMath(),
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
            desc_list.append(x.tool_description)
        desc_list.append('</tools>')
        return '\n'.join(desc_list)

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
    def __init__(self):
        with open("prompt_GAT.txt", 'r', encoding='utf-8') as f:
            self.prompt = f.read()
        
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

        # self.post_anti_hallucination = "<scratchpad> I understand I cannot use functions that have not been provided to me to answer this question."
        # self.post_anti_hallucination = "<scratchpad> I can only use functions that have been explicitly provided to me to answer this question. The names of the functions I can use are: "
        self.post_anti_hallucination = "<scratchpad> I can only use functions that have been explicitly provided. I must follow the <tool_guidelines></tool_guidelines>. I may need the following tools:"