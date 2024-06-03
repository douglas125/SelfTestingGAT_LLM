import os
import numpy as np


ans = None
class ToolSolvePythonCode():
    def __init__(self):
        self.name = 'solve_with_python'

        self.tool_description = {
            'name': self.name,
            'description': f"""Runs Python code to answer questions that can be easily solved with basic Python code.
Ensure that the code generates the final answer to the problem, without requiring any further analysis. Assign the final answer to the variable "ans".

Use {self.name} in any cases where a simple Python program can give the correct answer. Some examples are provided below in <use_cases></use_cases>:
<use_cases>
<use_case>String manipulation (e.g. reversing or sorting strings)<use_case>
<use_case>Sorting, ordering, filtering<use_case>
</use_cases>

Raises ValueError: if the code to be executed was invalid.""",
            'input_schema': {
                'type': 'object',
                'properties': {
                    'plot_code': {
                        'type': 'string',
                        'description': 'Python code that, when executed, will provide the final answer to the problem',
                    },
                },
                'required': ['python_code']
            }
        }

    def __call__(self, python_code, **kwargs):
        if len(kwargs) > 0:
            return f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"

        try:
            global ans
            if ans is not None:
                ans = None
            exec(python_code, globals())
        except Exception as e:
            return f"Code did NOT execute correctly.\nError description: {str(e)}"

        return str(ans)
