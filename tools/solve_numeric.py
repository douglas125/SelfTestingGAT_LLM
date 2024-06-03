import os
import numpy as np


ans = None
class ToolSolveNumeric():
    def __init__(self):
        self.name = 'solve_numeric'

        self.tool_description = {
            'name': self.name,
            'description': f"""Evaluates numerical expressions the Python library numpy. Always use {self.name} to evaluate the numerical expressions instead of doing it manually. Make sure to import numpy before using it in the numpy_code.
Ensure that the code generates the final answer to the problem, without requiring any further analysis. Assign the final answer to the variable "ans".

Use {self.name} in these <use_cases></use_cases>:
<use_cases>
<use_case>Doing direct numeric evaluation of expressions</use_case>
<use_case>There are no symbolic equations to solve</use_case>
</use_cases>

Raises ValueError: if the code to be executed was invalid.""",
            'input_schema': {
                'type': 'object',
                'properties': {
                    'numpy_code': {
                        'type': 'string',
                        'description': 'Python code that, when executed, will provide the final answer to the problem',
                    },
                },
                'required': ['numpy_code']
            }
        }

    def __call__(self, numpy_code, **kwargs):
        if len(kwargs) > 0:
            return f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"

        try:
            global ans
            if ans is not None:
                ans = None
            exec(numpy_code, globals())
        except Exception as e:
            return f"Code did NOT execute correctly.\nError description: {str(e)}"

        return str(ans)
