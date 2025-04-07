import os
import re
import json
import time

from openai import OpenAI
from .base_service import LLM_Service


class LLM_Maritalk(LLM_Service):
    def __init__(self, model_size):
        """Constructor
        Arguments:
            model_size - Maritaca model to use to make LLM calls
        """
        self.openai_client = None
        if model_size == "Sabia3 Maritaca":
            self.model_id = "sabia-3"
            self.llm_description = "Sabia-3 (medium-sized LLM) - directly from Maritaca"
            self.price_per_M_input_tokens = 0.95
            self.price_per_M_output_tokens = 1.9

        self.config = {
            # "messages": prompt,
            # "system": sysprompt,
            "max_tokens": 4000,
            "temperature": 0.5,  # 0.5 is default,
            "stream": True,
            # "top_k": 250,
            # "top_p": 1,
            "stop": None,  # the regular is already implemented
            "model": self.model_id,
        }
        # requests and answer word count
        self.word_counts = []

    def _prepare_prompt_from_list(self, msg_list):
        """Receives a list of dictionaries containing keys
        'role' and 'content' and produces the relevant formatting for the LLM
        In the case of OpenAI, the input is already the expected format
        """
        ans = msg_list
        return ans

    def _prepare_call_list_from_history(
        self, system_prompt, msg, b64image, chat_history
    ):
        """Prepares the prompt for the next interaction with the LLM"""
        history_list = [
            {"role": "system", "content": system_prompt},
        ]
        for x in chat_history:
            if isinstance(x, dict):
                history_list.append(x)
            else:
                history_list.append({"role": "user", "content": x[0]})
                history_list.append({"role": "assistant", "content": str(x[1])})

        if b64image is None:
            history_list.append({"role": "user", "content": msg})
        else:
            history_list.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64image}",
                            },
                        },
                        {"type": "text", "text": msg},
                    ],
                }
            )
        return history_list

    def invoke_streaming(
        self,
        prompt,
        b64image=None,
        postpend="",
        extra_stop_sequences=[],
        tools=None,
        tool_invoker_fn=None,
        max_retries=25,
        cur_fail_sleep=60,
    ):
        """
        Invokes the Maritaca model to run an inference
        using the input provided in the request body.

        :param prompt: The prompt to be answered.
            In the case of Claude 3, it is a dictionary with keys
            system -> str, messages -> list
        :param postpend: Extra text to append, to `put words in the mouth` of the LLM
        tools: description of the tools that can be used, Claude format
        tool_invoker_fn: function that invokes the tools. Arguments are:
                function name - function to call
                return_results_only - we set to True because we already use Claude format
                kwargs - arguments to the tool that will be called
        :return: Inference response from the model.
        """
        # create client if it hasn't been created already
        if self.openai_client is None:
            try:
                self.openai_client = OpenAI(
                    api_key=os.environ.get("MARITACA_API_KEY"),
                    base_url="https://chat.maritaca.ai/api",
                )
            except Exception:
                pass

        # Messages that had to be added because of function use
        self.tool_use_added_msgs = []

        # try:
        # The different model providers have individual request and response formats.
        # For the format, ranges, and default values for Meta Llama 2 Chat, refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html

        # start time
        t0 = time.time()
        body = self.config.copy()
        if len(extra_stop_sequences) > 0:
            body["stop"] = extra_stop_sequences
        else:
            body.pop("stop", None)
        body["messages"] = prompt

        if tools is None:
            body["messages"].append({"role": "assistant", "content": postpend})
        else:
            adj_tools = []
            for x in tools:
                cur_desc = x.copy()
                cur_desc["parameters"] = cur_desc["input_schema"]
                cur_desc.pop("input_schema", None)

                adj_tools.append(
                    {
                        "type": "function",
                        "function": cur_desc,
                    }
                )
            body["tools"] = adj_tools
            # assert postpend == "", "When using tools, postpend is not supported"
            assert (
                tool_invoker_fn is not None
            ), "When using tools, a tool invoker must be provided"

        for k in range(max_retries):
            try:
                self.debug_body = body
                llm_body_changed = True
                while llm_body_changed:
                    llm_body_changed = False
                    response = self.openai_client.chat.completions.create(**body)
                    word_count = len(re.findall(r"\w+", str(body["messages"])))
                    print(f"Invoking {self.llm_description}. Word count: {word_count}")

                    # stream responses
                    partial_ans = self._response_gen(response, postpend)
                    x = ""
                    for x in partial_ans:
                        yield x
                    cur_ans = x

                    if self.cur_tool_spec is not None:
                        # tool use has been required. Let's do it
                        # TODO: update upstream to reflect the inclusion of a response
                        # TODO: probably rework gradio UI to re-instantiate things every chat, or keep an instance per chat ID
                        tool_ans = tool_invoker_fn(
                            self.cur_tool_spec["tool_name"],
                            return_results_only=True,
                            **self.cur_tool_spec["input"],
                        )

                        # append assistant responses
                        assistant_msg = {
                            "role": "assistant",
                            "content": cur_ans,
                            "tool_calls": [
                                {
                                    "id": self.cur_tool_spec["id"],
                                    "type": "function",
                                    "function": {
                                        "name": self.cur_tool_spec["tool_name"],
                                        "arguments": json.dumps(
                                            self.cur_tool_spec["input"]
                                        ),
                                    },
                                }
                            ],
                        }

                        next_user_msg = {
                            "role": "tool",
                            "content": tool_ans,
                            "tool_call_id": self.cur_tool_spec["id"],
                        }
                        body["messages"].append(assistant_msg)
                        body["messages"].append(next_user_msg)

                        # keep a log of messages that had to be appended due to tool use
                        self.tool_use_added_msgs.append(assistant_msg)
                        self.tool_use_added_msgs.append(next_user_msg)
                        llm_body_changed = True

                    # TODO: Include proper token count and pricing
                    ans_word_count = len(
                        re.findall(r"\w+", postpend + cur_ans + str(self.stop_reason))
                    )
                    self.word_counts.append(
                        {
                            "request_word_count": word_count,
                            "answer_word_count": ans_word_count,
                            "price_estimate": 0.001
                            * (0.003 * word_count + 0.015 * ans_word_count),
                            "exec_time_in_s": time.time() - t0,
                        }
                    )
                return
            except Exception as e:
                yield f"Error {str(e)}. Waiting {int(cur_fail_sleep)} s. Retrying {k+1}/{max_retries}..."
                time.sleep(int(cur_fail_sleep))
                cur_fail_sleep *= 1.2
        yield "Could not invoke the AI model."

    def _response_gen(self, response_body, postpend=""):
        cur_ans = ""
        cur_tool_spec = None
        for x in response_body:
            txt = ""
            if x.choices[0].delta.tool_calls is not None and cur_tool_spec is None:
                cur_tool_spec = x.choices[0].delta.tool_calls[0].__dict__.copy()
                cur_tool_spec["arguments"] = ""

            txt = (
                x.choices[0].delta.content
                if x.choices[0].delta.content is not None
                else ""
            )
            if x.choices[0].delta.tool_calls is not None:
                cur_tool_spec["arguments"] += (
                    x.choices[0].delta.tool_calls[0].function.arguments
                )

            if txt != "":
                cur_ans += txt
                yield postpend + cur_ans

            stop_reason = x.choices[0].finish_reason
            if stop_reason is not None and stop_reason == "stop_sequence":
                stop_txt = x.delta.stop_sequence
                yield postpend + cur_ans + stop_txt
                break
        if cur_tool_spec is not None:
            cur_tool_spec["arguments"] = cur_tool_spec["arguments"].split("{")[1:]
            cur_tool_spec["arguments"] = "{" + "{".join(cur_tool_spec["arguments"])

            # print(f'*{cur_tool_spec["arguments"]}*')
            cur_tool_spec["input"] = (
                cur_tool_spec["arguments"]
                if isinstance(cur_tool_spec["arguments"], dict)
                else json.loads(cur_tool_spec["arguments"])
            )
            cur_tool_spec["tool_name"] = cur_tool_spec["function"].name
            cur_tool_spec.pop("index", None)
            cur_tool_spec.pop("type", None)
            cur_tool_spec.pop("function", None)
            cur_tool_spec.pop("arguments", None)

        self.cur_tool_spec = cur_tool_spec
        self.stop_reason = stop_reason
