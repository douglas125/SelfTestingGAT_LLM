ans = None


class ToolSolveSymbolic:
    def __init__(self):
        self.name = "solve_symbolic"

        self.tool_description = {
            "name": self.name,
            "description": f"""Gets the solution to a symbolic mathematics problem using the Python library sympy. Make sure to import sympy before using it in the <sympy_code></sympy_code>.
Ensure that the code generates the final answer to the problem, without requiring any further analysis. Assign the final answer to the variable "ans".

Use {self.name} only in these <use_cases>:
<use_cases>
<use_case>There are symbolic equations or expressions to simplify or solve</use_case>
<use_case>The problem cannot be solved using purely numeric evaluations</use_case>
</use_cases>

Before calling this function, explicitly list out all problem assumptions in the <scratchpad></scratchpad>.

Raises ValueError: if the code to be executed was invalid.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sympy_code": {
                        "type": "string",
                        "description": "Python code that, when executed, will provide the final answer to the problem",
                    },
                },
                "required": ["sympy_code"],
            },
        }

    def __call__(self, sympy_code, **kwargs):
        if len(kwargs) > 0:
            return f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"

        try:
            global ans
            if ans is not None:
                ans = None
            exec(sympy_code, globals())
        except Exception as e:
            return f"Code did NOT execute correctly.\nError description: {str(e)}"

        return str(ans)
