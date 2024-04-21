import os
import datetime
import dateutil


class ToolDoDateMath():
    def __init__(self):
        self.name = 'do_date_math'

        self.tool_summary = f"""<tool_summary>
<tool_name>{self.name}</tool_name>
<summary>Adds or subtracts one or more time intervals from a given date in the format YYYY-MM-DD.</summary>
</tool_summary>"""

        self.tool_description = """
<tool_description>
<tool_name>{{TOOLNAME}}</tool_name>
<description>Adds or subtracts one or more time intervals from a given date in the format YYYY-MM-DD.

The <deltas> to be added or subtracted should be separated by commas. Use negative values to subtract, as shown in the <example_deltas></example_deltas>:

<example_deltas>
<example_delta>5<example_delta>
<example_delta>-7, -14, -21<example_delta>
<example_delta>5, -6, -8<example_delta>
</example_deltas>

Raises ValueError: if one of the parameters is invalid.
</description>

<parameters>
<parameter>
<name>base_date</name>
<type>string</type>
<description>Base date in the format YYYY-MM-DD.
</description>
</parameter>

<parameter>
<name>deltas</name>
<type>string</type>
<description>Interval, as defined in delta_type, to add or subtract from the base date, separated by commas.
</description>
</parameter>

<parameter>
<name>delta_type</name>
<type>string</type>
<description>Type of interval to sum or subtract from base_date. Can be one of the following <delta_options>:

<delta_options>
<option>day</option>
<option>week</option>
<option>month</option>
<option>year</option>
</delta_options>

</description>
</parameter>

</parameters>

</tool_description>
        """.replace('{{TOOLNAME}}', self.name)

        self.tool_summary = self.tool_description

    def __call__(self, base_date, deltas, delta_type, **kwargs):
        allowed_delta_types = ['day', 'week', 'month', 'year']
        if not delta_type in allowed_delta_types:
            return f'Error: delta_type must be one of {allowed_delta_types}'

        try:
            date_object = datetime.datetime.strptime(base_date, '%Y-%m-%d').date()
        except:
            return 'Error: please provide a base_date in the format YYYY-MM-DD'

        delta_periods = [int(x.strip()) for x in deltas.split(',')]

        # dateutil.relativedelta.relativedelta(dt1=None, dt2=None, years=0, months=0, days=0, leapdays=0, weeks=0

        if delta_type == 'day':
            final_deltas = [dateutil.relativedelta.relativedelta(days=x) for x in delta_periods]
        elif delta_type == 'week':
            final_deltas = [dateutil.relativedelta.relativedelta(weeks=x) for x in delta_periods]
        elif delta_type == 'month':
            final_deltas = [dateutil.relativedelta.relativedelta(months=x) for x in delta_periods]
        elif delta_type == 'year':
            final_deltas = [dateutil.relativedelta.relativedelta(years=x) for x in delta_periods]

        ans = [(date_object + x).strftime('%Y-%m-%d %A') for x in final_deltas]
        return ','.join(ans)
