import re
import json
import time

import anthropic
from llm_providers.base_service import LLM_Service


class LLM_Claude3_Anthropic(LLM_Service):
    def __init__(self, model_size):
        """Constructor
        Arguments:
            bedrock_client - Instance of boto3.client(service_name='bedrock-runtime')
                to use when making calls to bedrock models
        """
        if model_size == "Sonnet 3.5 Anthropic":
            self.model_id = "claude-3-5-sonnet-20240620"
            self.anthropic_client = anthropic.Anthropic()
            self.llm_description = "Anthropic Claude 3.5 Sonnet (Medium-size LLM) - directly from Anthropic"

        self.config = {
            # "messages": prompt,
            # "system": sysprompt,
            "max_tokens": 4000,
            "temperature": 0.5,  # 0.5 is default,
            "stream": True,
            # "top_k": 250,
            # "top_p": 1,
            "stop_sequences": [],  # the regular is already implemented
            "model": self.model_id,
        }
        # requests and answer word count
        self.word_counts = []

    def _prepare_prompt_from_list(self, msg_list):
        """Receives a list of dictionaries containing keys
        'role' and 'content' and produces the relevant formatting for the LLM
        """
        ans = {}
        if msg_list[0]["role"] == "system":
            ans["system"] = msg_list[0]["content"]
        ans["messages"] = []
        for msg in msg_list[1:]:
            ans["messages"].append(msg)
        return ans

    def invoke_streaming(
        self,
        prompt,
        b64image=None,
        postpend="",
        extra_stop_sequences=[],
        tools=None,
        tool_invoker_fn=None,
        max_retries=25,
    ):
        """
        Invokes the Claude 3 model to run an inference
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
        # Messages that had to be added because of function use
        self.tool_use_added_msgs = []

        # try:
        # The different model providers have individual request and response formats.
        # For the format, ranges, and default values for Meta Llama 2 Chat, refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html

        # start time
        t0 = time.time()
        body = self.config.copy()
        body["stop_sequences"] = body["stop_sequences"] + extra_stop_sequences

        body["system"] = prompt["system"]
        body["messages"] = prompt["messages"].copy()

        if tools is None:
            body["messages"].append({"role": "assistant", "content": postpend})
        else:
            body["tools"] = tools
            assert postpend == "", "When using tools, postpend is not supported"
            assert (
                tool_invoker_fn is not None
            ), "When using tools, a tool invoker must be provided"

        cur_fail_sleep = 60
        for k in range(max_retries):
            try:
                self.debug_body = body
                llm_body_changed = True
                while llm_body_changed:
                    llm_body_changed = False
                    response = self.anthropic_client.messages.create(**body)
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
                            self.cur_tool_spec["name"],
                            return_results_only=True,
                            **self.cur_tool_spec["input"],
                        )

                        # append assistant responses
                        assistant_msg = {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": cur_ans,
                                },
                                self.cur_tool_spec,
                            ],
                        }

                        next_user_msg = {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": self.cur_tool_spec["id"],
                                    "content": tool_ans,
                                }
                            ],
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
                print(
                    f'Error {str(e)}. Prompt length: {len(str(body["messages"]))}\n\nRetrying {k}...'
                )
                time.sleep(int(cur_fail_sleep))
                cur_fail_sleep *= 1.2

        raise

    def _response_gen(self, response_body, postpend=""):
        cur_ans = ""
        cur_tool_spec = None
        for x in response_body:
            txt = ""
            if hasattr(x, "content_block"):
                if x.content_block.type == "text":
                    txt = x.content_block.text
                elif x.content_block.type == "tool_use":
                    cur_tool_spec = x.content_block.__dict__.copy()
                    cur_tool_spec["input"] = ""

            elif hasattr(x, "delta"):
                txt = x.delta.text if hasattr(x.delta, "text") else ""
                if cur_tool_spec is not None:
                    cur_tool_spec["input"] += (
                        x.delta.partial_json if hasattr(x.delta, "partial_json") else ""
                    )

            if txt != "":
                cur_ans += txt
                yield postpend + cur_ans

            stop_reason = (
                x.delta.stop_reason
                if hasattr(x, "delta") and hasattr(x.delta, "stop_reason")
                else None
            )
            if stop_reason is not None and stop_reason == "stop_sequence":
                stop_txt = x.delta.stop_sequence
                yield postpend + cur_ans + stop_txt
                break
        if cur_tool_spec is not None:
            cur_tool_spec["input"] = json.loads(cur_tool_spec["input"])
        self.cur_tool_spec = cur_tool_spec
        self.stop_reason = stop_reason
