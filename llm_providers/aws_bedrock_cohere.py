import re
import copy
import json
import time

from llm_providers.base_service import LLM_Service


class LLM_Command_Cohere(LLM_Service):
    # TODO: Native function calling
    def __init__(self, bedrock_client, model_size):
        """Constructor
        Arguments:
            bedrock_client - Instance of boto3.client(service_name='bedrock-runtime')
                to use when making calls to bedrock models
        """
        self.bedrock_client = bedrock_client

        if model_size == "Command R Cohere 1":
            self.model_id = "cohere.command-r-v1:0"
            self.llm_description = (
                "Cohere Command R 1.0 (Medium-size LLM) - from Bedrock"
            )
            self.price_per_M_input_tokens = 0.5
            self.price_per_M_output_tokens = 1.5
        elif model_size == "Command RPlus Cohere 1":
            self.model_id = "cohere.command-r-plus-v1:0"
            self.llm_description = (
                "Cohere Command R+ 1.0 (Large-size LLM) - from Bedrock"
            )
            self.price_per_M_input_tokens = 3
            self.price_per_M_output_tokens = 15

        self.config = {
            # "messages": prompt,
            # "system": sysprompt,
            "max_tokens": 3900,
            "temperature": 0.5,  # 0.5 is default,
            "force_single_step": False,
            # "top_k": 250,
            # "top_p": 1,
            "stop_sequences": [],  # the regular is already implemented
        }
        # requests and answer word count
        self.word_counts = []

    def _prepare_prompt_from_list(self, msg_list):
        """Receives a list of dictionaries containing keys
        'role' and 'content' and produces the relevant formatting for the LLM
        """
        ans = {}
        if msg_list[0]["role"] == "system":
            ans["preamble"] = msg_list[0]["content"]

        role_map = {
            "user": "USER",
            "assistant": "CHATBOT",
        }
        ans["chat_history"] = []
        for msg in msg_list[1:-1]:
            ans["chat_history"].append(
                {
                    "role": role_map[msg["role"]],
                    "message": msg["content"],
                }
            )

        assert msg_list[-1]["role"] == "user"
        ans["message"] = msg_list[-1]["content"]
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
        Invokes the Cohere model to run an inference
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
        self.tool_results = []

        # try:
        # The different model providers have individual request and response formats.
        # For the format, ranges, and default values for Meta Llama 2 Chat, refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html

        # start time
        t0 = time.time()
        body = self.config.copy()
        body["stop_sequences"] = body["stop_sequences"] + extra_stop_sequences
        if len(body["stop_sequences"]) == 0:
            body.pop("stop_sequences", None)

        body["preamble"] = prompt["preamble"]
        body["chat_history"] = prompt["chat_history"].copy()
        body["message"] = prompt["message"]

        # TODO: Cohere does not support postpend. Let it use tools by itself.
        # assert postpend == "", "Model does not support postpend argument"
        postpend = ""

        # if tools is not None:
        #     body["messages"].append({"role": "assistant", "content": postpend})
        if tools is not None:
            assert postpend == "", "When using tools, postpend is not supported"
            assert (
                tool_invoker_fn is not None
            ), "When using tools, a tool invoker must be provided"

            adj_tools = []
            for x in tools:
                cur_desc = x.copy()
                required_params = cur_desc["input_schema"].copy().pop("required")
                cur_desc["parameter_definitions"] = copy.deepcopy(
                    cur_desc["input_schema"]["properties"]
                )
                cur_desc.pop("input_schema", None)
                # adjust parameter types
                lookup_param_type = {
                    "string": "str",
                    "integer": "int",
                    "boolean": "bool",
                }
                for k in cur_desc["parameter_definitions"]:
                    cur_desc["parameter_definitions"][k]["type"] = lookup_param_type[
                        cur_desc["parameter_definitions"][k]["type"]
                    ]
                # adjust required parameters
                for p in required_params:
                    cur_desc["parameter_definitions"][p]["required"] = True
                adj_tools.append(cur_desc)
            body["tools"] = adj_tools
            # assert postpend == "", "When using tools, postpend is not supported"
            assert (
                tool_invoker_fn is not None
            ), "When using tools, a tool invoker must be provided"

        cur_fail_sleep = 60
        for k in range(max_retries):
            # try:
            self.debug_body = body
            llm_body_changed = True
            while llm_body_changed:
                llm_body_changed = False
                response = self.bedrock_client.invoke_model_with_response_stream(
                    modelId=self.model_id, body=json.dumps(body)
                )
                word_count = len(
                    re.findall(r"\w+", body["message"] + str(body["chat_history"]))
                )
                print(f"Invoking {self.llm_description}. Word count: {word_count}")

                # stream responses
                partial_ans = self._response_gen(response["body"], postpend)
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
                    """
                        "tool_name": func_call_spec["name"],
                        "input": func_call_spec["parameters"],
                        "generation_id": func_call_spec["generation_id"],
                    """

                    print(self.cur_tool_spec)
                    last_call = {
                        "name": self.cur_tool_spec["tool_name"],
                        "parameters": self.cur_tool_spec["input"],
                        # "generation_id": self.cur_tool_spec["generation_id"],
                    }
                    self.tool_results.append(
                        {
                            "call": last_call,
                            "outputs": [
                                {self.cur_tool_spec["tool_name"] + "_output": tool_ans}
                            ],
                        }
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
                                # "generation_id": self.cur_tool_spec["generation_id"],
                                "content": tool_ans,
                            }
                        ],
                    }
                    # body["chat_history"].append(assistant_msg)
                    # body["chat_history"].append(next_user_msg)
                    body["tool_results"] = self.tool_results
                    body["chat_history"] = self.cur_tool_spec["chat_history"]
                    body["message"] = ""

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
        # except Exception as e:
        #    print(
        #        f'Error {str(e)}. Prompt length: {len(str(body["chat_history"]))}\n\nRetrying {k}...'
        #    )
        #    time.sleep(int(cur_fail_sleep))
        #    cur_fail_sleep *= 1.2

        raise

    def _response_gen(self, response_body, postpend=""):
        cur_ans = ""
        cur_tool_spec = None
        for x in response_body:
            out_dict = json.loads(x["chunk"]["bytes"])
            txt = ""
            if "event_type" in out_dict.keys():
                # if out_dict["event_type"] == "text-generation":
                #     txt = out_dict["text"]
                if (
                    out_dict["event_type"] == "tool-calls-generation"
                    and out_dict.get("out_dict", False)
                    and len(out_dict["tool_calls"]) > 0
                ):
                    func_call_spec = out_dict["tool_calls"][0]
                    cur_tool_spec = {
                        "tool_name": func_call_spec["name"],
                        "input": func_call_spec["parameters"],
                    }
                txt = out_dict.get("text", "")

            if txt != "":
                cur_ans += txt
                yield postpend + cur_ans
            stop_reason = out_dict["event_type"] if out_dict["is_finished"] else None
            if out_dict["is_finished"] and cur_tool_spec is not None:
                cur_tool_spec["generation_id"] = out_dict["response"]["generation_id"]
                cur_tool_spec["chat_history"] = out_dict["response"]["chat_history"]

        yield postpend + cur_ans
        # TODO: Make tool call work
        self.cur_tool_spec = cur_tool_spec
        self.stop_reason = stop_reason
