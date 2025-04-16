import datetime
import importlib.resources


class RAGPromptGenerator:
    def __init__(self, use_native_tools=False):
        """Constructor.

        Arguments:

        - use_native_tools: Use LLM native tool calling abilities. In this case, we do not
            need to "teach" the LLM about tools
        """
        self.prompt = ""
        self.use_native_tools = use_native_tools
        if not use_native_tools:
            with importlib.resources.files("gat_llm.prompts").joinpath(
                "prompt_GAT.txt"
            ).open("r", encoding="utf-8") as f:
                self.prompt += f.read()

        # base prompt
        with importlib.resources.files("gat_llm.prompts").joinpath(
            "prompt_base.txt"
        ).open("r", encoding="utf-8") as f:
            self.prompt += f.read()

        dt0 = datetime.datetime.today()
        weekday = dt0.strftime("%A")

        date_str = f"{weekday}, {dt0.year}-{str(dt0.month).zfill(2)}-{str(dt0.day).zfill(2)} in the format YYYY-MM-DD"
        date_str = f"""
<today_date>
<day_of_week>{weekday}</day_of_week>
<day>{str(dt0.day).zfill(2)}</day>
<month>{str(dt0.month).zfill(2)}</month>
<year>{dt0.year}</year>

</today_date>
        """

        self.prompt = self.prompt.replace("{{DATE}}", date_str)

        self.post_anti_hallucination = "<scratchpad> I can only use functions that have been explicitly provided. I must follow the <tool_guidelines></tool_guidelines>. I need the following tools:"
