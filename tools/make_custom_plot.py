import os
import numpy as np
from matplotlib import pyplot as plt

rng = np.random.default_rng()


class ToolMakeCustomPlot:
    def __init__(self):
        self.name = "make_custom_plot"

        self.save_code = "plt.savefig('media/plot.jpg')"

        self.tool_description = {
            "name": self.name,
            "description": f"""Generates an image or plot using the provided custom Python code plot_code, whose only dependencies should be numpy and matplotlib. This tool is useful when the user requests plots to be generated from data that was previously retrieved.

If the user didn't specifically ask for a plot, confirm is the user wants to generate a plot or visualization before calling this tool.

The last line of the plot code must save the figure to 'media/plot.jpg'. The last line of the code should be: {self.save_code}

The user can view the images without exposing the auto-generated file names. Do NOT include actual file names in the answer. Do NOT include the <path_to_image></path_to_image> tag in the answer. Only mention that the image has been generated successfully.

Raises ValueError: if the code to be executed was invalid.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "plot_code": {
                        "type": "string",
                        "description": "Python code that, when executed, will generate an image or plot requested by the user",
                    },
                    "deltas": {
                        "type": "int",
                        "description": "Interval, as defined in delta_type, to add or subtract from the base date, separated by commas",
                    },
                    "delta_type": {
                        "type": "string",
                        "enum": ["day", "week", "month", "year"],
                        "description": "Type of interval to sum or subtract from base_date. Possible values are listed in enum",
                    },
                },
                "required": ["plot_code"],
            },
        }

    def __call__(self, plot_code, **kwargs):
        os.makedirs("media", exist_ok=True)
        # fix weird save attempts
        plot_code = plot_code.splitlines()
        plot_code = [x for x in plot_code if not x.startswith("plt.savefig")]
        plot_code.append(self.save_code)
        plot_code = "\n".join(plot_code)

        if len(kwargs) > 0:
            return f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"

        rng_num = rng.integers(low=0, high=900000)
        target_file = f"'media/plot_{rng_num}.jpg'"
        plot_code = plot_code.replace("'media/plot.jpg'", target_file)

        try:
            exec(plot_code)
        except Exception as e:
            return f"Plot was NOT generated.\nError description: {str(e)}"

        target_file = target_file.replace("'", "")
        if not os.path.isfile(target_file):
            return "Error: Image was not saved correctly."

        # with open(f'plot_code_{rng_num}.txt', 'w', encoding='utf-8') as debug_file:
        #    p_str = plot_code + f'\n***{target_file} {os.path.isfile(target_file)}'
        #    debug_file.write(p_str)

        ans = ["<image>"]
        ans.append(f"<path_to_image>{target_file}</path_to_image>")
        ans.append("</image>")
        return "\n".join(ans)
