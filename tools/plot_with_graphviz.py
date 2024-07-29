import os
import numpy as np

rng = np.random.default_rng()


class ToolPlotWithGraphviz:
    def __init__(self):
        self.name = "plot_with_graphviz"

        self.save_code = "graph.write_png('media/graph.png')"

        self.tool_description = {
            "name": self.name,
            "description": f"""Generates a graph visualization using the provided custom Python code graph_code, whose only dependencies should be pydot. This tool is useful when the user requests graph visualizations to be generated. Prefer to use a horizontal layout if not asked.

If the user didn't specifically ask for a graph visualization, confirm if the user wants to generate a graph visualization before calling this tool.

The last line of the graph code must save the figure to 'media/graph.png'. The last line of the code should be: {self.save_code}

The user can view the images without exposing the auto-generated file names. Do NOT include actual file names in the answer. Do NOT include the <path_to_image></path_to_image> tag in the answer. Only mention that the graph image has been generated successfully.

Raises ValueError: if the code to be executed was invalid.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "graph_code": {
                        "type": "string",
                        "description": "Python code that, when executed, will generate a graph visualization using pydot",
                    },
                },
                "required": ["graph_code"],
            },
        }

    def __call__(self, graph_code, **kwargs):
        os.makedirs("media", exist_ok=True)
        # fix weird save attempts
        graph_code = graph_code.splitlines()
        graph_code = [x for x in graph_code if not x.startswith("graph.write_png")]
        graph_code.append(self.save_code)
        graph_code = "\n".join(graph_code)

        if len(kwargs) > 0:
            return f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"

        rng_num = rng.integers(low=0, high=900000)
        target_file = f"'media/graph_{rng_num}.png'"
        graph_code = graph_code.replace("'media/graph.png'", target_file)

        try:
            exec(graph_code)
        except Exception as e:
            return f"Graph was NOT generated.\nError description: {str(e)}"

        target_file = target_file.replace("'", "")
        if not os.path.isfile(target_file):
            return "Error: Image was not saved correctly."

        ans = ["<image>"]
        ans.append(f"<path_to_image>{target_file}</path_to_image>")
        ans.append("</image>")
        return "\n".join(ans)
