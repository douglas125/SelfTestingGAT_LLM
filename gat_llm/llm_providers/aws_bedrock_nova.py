# https://docs.aws.amazon.com/nova/latest/userguide/complete-request-schema.html
# https://docs.aws.amazon.com/nova/latest/userguide/tool-use-results.html
import re
import json
import time
import types

from .base_service import LLM_Service


class LLM_Nova_Bedrock(LLM_Service):
    def __init__(self, bedrock_client, model_size):
        """Constructor
        Arguments:
            bedrock_client - Instance of boto3.client(service_name='bedrock-runtime')
                to use when making calls to bedrock models
        """
        if model_size == "Nova_Micro":
            self.model_id = "us.amazon.nova-micro-v1:0"
            self.llm_description = "Amazon Nova Micro v1.0 from AWS Bedrock (Tiny LLM)"
            self.price_per_M_input_tokens = 0.035
            self.price_per_M_output_tokens = 0.14
        elif model_size == "Nova_Lite":
            self.model_id = "us.amazon.nova-lite-v1:0"
            self.llm_description = "Amazon Nova Lite v1.0 from AWS Bedrock (Small LLM)"
            self.price_per_M_input_tokens = 0.06
            self.price_per_M_output_tokens = 0.24
        elif model_size == "Nova_Pro":
            self.model_id = "us.amazon.nova-pro-v1:0"
            self.llm_description = "Amazon Nova Pro v1.0 from AWS Bedrock (Large LLM)"
            self.price_per_M_input_tokens = 0.8
            self.price_per_M_output_tokens = 3.2

        self.bedrock_client = bedrock_client
        self.config = {
            # "messages": prompt,
            # "system": sysprompt,
            "inferenceConfig": {
                "max_new_tokens": 4000,
                "temperature": 0.65,
                "stopSequences": [],
            },
        }
        # requests and answer word count
        self.word_counts = []

    def _prepare_prompt_from_list(self, msg_list):
        """Receives a list of dictionaries containing keys
        'role' and 'content' and produces the relevant formatting for the LLM
        """
        ans = {}
        if msg_list[0]["role"] == "system":
            ans["system"] = [{"text": msg_list[0]["content"]}]
        ans["messages"] = []
        for msg in msg_list[1:]:
            if isinstance(msg["content"], str):
                msg["content"] = [{"text": msg["content"]}]
            elif isinstance(msg["content"], list):
                for k in range(len(msg["content"])):
                    if msg["content"][k].get("type") == "text":
                        adj_msg = {"text": msg["content"][k]["text"]}
                        msg["content"][k] = adj_msg
                    elif msg["content"][k].get("type") == "image":
                        adj_msg = {
                            "image": {
                                "format": "jpeg",
                                "source": {
                                    "bytes": msg["content"][k]["source"]["data"],
                                },
                            }
                        }
                        msg["content"][k] = adj_msg
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
        cur_fail_sleep=6,
    ):
        """
        Invokes the Nova model to run an inference
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
        body["inferenceConfig"]["stopSequences"] = (
            body["inferenceConfig"]["stopSequences"] + extra_stop_sequences
        )

        body["messages"] = prompt["messages"].copy()

        if tools is None:
            body["messages"].append(
                {"role": "assistant", "content": [{"text": postpend}]}
            )
        else:
            adj_tools = []
            for x in tools:
                cur_desc = x.copy()
                cur_desc["inputSchema"] = {"json": cur_desc["input_schema"]}
                cur_desc.pop("input_schema", None)
                adj_tools.append({"toolSpec": cur_desc})
            body["toolConfig"] = {"tools": adj_tools}
            assert postpend == "", "When using tools, postpend is not supported"
            assert (
                tool_invoker_fn is not None
            ), "When using tools, a tool invoker must be provided"

        for k in range(max_retries):
            try:
                llm_body_changed = True
                while llm_body_changed:
                    llm_body_changed = False
                    response = self.bedrock_client.invoke_model_with_response_stream(
                        modelId=self.model_id, body=json.dumps(body)
                    )
                    word_count = len(re.findall(r"\w+", str(body["messages"])))
                    print(f"Invoking {self.llm_description}. Word count: {word_count}")

                    # stream responses
                    # return response
                    partial_ans = self._response_gen(response["body"], postpend)
                    for x in partial_ans:
                        yield x
                    cur_ans = x

                    if self.cur_tool_spec is not None:
                        # tool use has been required. Let's do it
                        # TODO: update upstream to reflect the inclusion of a response
                        # TODO: probably rework gradio UI to re-instantiate things every chat, or keep an instance per chat ID
                        tool_ans = tool_invoker_fn(
                            self.cur_tool_spec["toolUse"]["name"],
                            return_results_only=True,
                            **self.cur_tool_spec["toolUse"]["input"],
                        )
                        if isinstance(tool_ans, types.GeneratorType):
                            for partial_ans in tool_ans:
                                yield partial_ans
                            tool_ans = partial_ans

                        # append assistant responses
                        assistant_msg = {
                            "role": "assistant",
                            "content": [
                                {
                                    "text": cur_ans,
                                },
                            ],
                        }
                        assistant_msg2 = {
                            "role": "assistant",
                            "content": [
                                {
                                    "toolUse": self.cur_tool_spec["toolUse"],
                                },
                            ],
                        }

                        next_user_msg = {
                            "role": "user",
                            "content": [
                                {
                                    "toolResult": {
                                        # "type": "tool_result",
                                        "toolUseId": self.cur_tool_spec["toolUse"][
                                            "toolUseId"
                                        ],
                                        "content": [{"text": tool_ans}],
                                    }
                                }
                            ],
                        }

                        body["messages"].append(assistant_msg)
                        body["messages"].append(assistant_msg2)
                        body["messages"].append(next_user_msg)

                        # keep a log of messages that had to be appended due to tool use
                        self.tool_use_added_msgs.append(assistant_msg)
                        self.tool_use_added_msgs.append(assistant_msg2)
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
                    # """
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
            out_dict = json.loads(x["chunk"]["bytes"])
            txt = ""
            if "contentBlockDelta" in out_dict.keys():
                if (
                    out_dict["contentBlockDelta"]["delta"].get("text")
                    and cur_tool_spec is None
                ):
                    txt = out_dict["contentBlockDelta"]["delta"]["text"]
                elif out_dict["contentBlockDelta"]["delta"].get("toolUse"):
                    cur_tool_spec["toolUse"]["input"] = out_dict["contentBlockDelta"][
                        "delta"
                    ]["toolUse"]["input"]

            if "contentBlockStart" in out_dict.keys():
                if out_dict["contentBlockStart"]["start"].get("toolUse"):
                    cur_tool_spec = out_dict["contentBlockStart"]["start"].copy()

            if cur_tool_spec is None:
                cur_ans += txt
                yield postpend + cur_ans

            stop_reason = None
            if out_dict.get("messageStop"):
                stop_reason = out_dict["messageStop"].get("stopReason", None)
            if stop_reason is not None:
                break
        if cur_tool_spec is not None:
            cur_tool_spec["toolUse"]["input"] = json.loads(
                cur_tool_spec["toolUse"]["input"]
            )
        self.cur_tool_spec = cur_tool_spec
        self.stop_reason = stop_reason
