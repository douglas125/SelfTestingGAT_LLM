import datetime
import dateutil


class ToolDoDateMath:
    def __init__(self):
        self.name = "do_date_math"

        self.tool_summary = f"""<tool_summary>
<tool_name>{self.name}</tool_name>
<summary>Adds or subtracts one or more time intervals from a given date in the format YYYY-MM-DD.</summary>
</tool_summary>"""

        self.tool_description = {
            "name": self.name,
            "description": """Adds or subtracts one or more time intervals from a given date in the format YYYY-MM-DD.

The <deltas></deltas> to be added or subtracted should be separated by commas. Use negative values to subtract, as shown in the <example_deltas></example_deltas>:

<example_deltas>
<example_delta>5<example_delta>
<example_delta>-7, -14, -21<example_delta>
<example_delta>5, -6, -8<example_delta>
</example_deltas>

Raises ValueError: if one of the parameters is invalid.""",
            "input_schema": {
                "type": "object",
                "properties": {
                    "base_date": {
                        "type": "string",
                        "description": "Base date in the format YYYY-MM-DD",
                    },
                    "deltas": {
                        "type": "string",
                        "description": "Intervals, as defined in delta_type, to add or subtract from the base date, separated by commas",
                    },
                    "delta_type": {
                        "type": "string",
                        "enum": ["day", "week", "month", "year"],
                        "description": "Type of interval to sum or subtract from base_date. Possible values are listed in the string enum: ['day', 'week', 'month', 'year']",
                    },
                },
                "required": ["base_date", "deltas", "delta_type"],
            },
        }

        self.tool_summary = self.tool_description

    def __call__(self, base_date, deltas, delta_type, **kwargs):
        if len(kwargs) > 0:
            return f"Error: Unexpected parameter(s): {','.join([x for x in kwargs])}"

        allowed_delta_types = ["day", "week", "month", "year"]
        if delta_type not in allowed_delta_types:
            return f"Error: delta_type must be one of {allowed_delta_types}"

        try:
            date_object = datetime.datetime.strptime(base_date, "%Y-%m-%d").date()
        except:
            return "Error: please provide a base_date in the format YYYY-MM-DD"

        delta_periods = [int(x.strip()) for x in deltas.split(",")]

        # dateutil.relativedelta.relativedelta(dt1=None, dt2=None, years=0, months=0, days=0, leapdays=0, weeks=0

        if delta_type == "day":
            final_deltas = [
                dateutil.relativedelta.relativedelta(days=x) for x in delta_periods
            ]
        elif delta_type == "week":
            final_deltas = [
                dateutil.relativedelta.relativedelta(weeks=x) for x in delta_periods
            ]
        elif delta_type == "month":
            final_deltas = [
                dateutil.relativedelta.relativedelta(months=x) for x in delta_periods
            ]
        elif delta_type == "year":
            final_deltas = [
                dateutil.relativedelta.relativedelta(years=x) for x in delta_periods
            ]

        ans = [(date_object + x).strftime("%Y-%m-%d %A") for x in final_deltas]
        return ",".join(ans)
