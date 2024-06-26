""" Set of available and useful LLMs (mostly posted on AWS Bedrock)
"""
import re
import json
import time

from typing import Dict, List

import anthropic


class LLM_Bedrock:
    allowed_llms = [
        "Claude 3.5 Sonnet - Anthropic",
        "Claude 3 Haiku",
        "Claude 3 Sonnet",
        "Claude 3 Opus",
        "Claude 2.1",
        "Claude Instant 1.2",
        "Llama2 13b",
        "Llama2 70b",
        "Mistral Mixtral 8x7B",
    ]

    def get_llm(bedrock_client, llm="Claude 2.1"):
        """Constructor
        Arguments:
            bedrock_client - Instance of boto3.client(service_name='bedrock-runtime')
                to use when making calls to bedrock models
            llm - which LLM to use. Check LLM_Bedrock.allowed_llms for a list
        """
        assert (
            llm in LLM_Bedrock.allowed_llms
        ), f"LLM has to be one of {LLM_Bedrock.allowed_llms}"
        if llm == "Claude 2.1":
            return LLM_Claude2_1_Bedrock(bedrock_client)
        elif llm == "Claude Instant 1.2":
            return LLM_Claude_Instant_1_2_Bedrock(bedrock_client)
        elif llm == "Llama2 13b":
            return LLM_Llama13b(bedrock_client)
        elif llm == "Llama2 70b":
            return LLM_Llama70b(bedrock_client)
        elif llm == "Mistral Mixtral 8x7B":
            return LLM_Mixtral8x7b_Bedrock(bedrock_client)
        elif llm == "Claude 3 Opus":
            return LLM_Claude3_Bedrock(bedrock_client, model_size="Opus")
        elif llm == "Claude 3 Sonnet":
            return LLM_Claude3_Bedrock(bedrock_client, model_size="Sonnet")
        elif llm == "Claude 3 Haiku":
            return LLM_Claude3_Bedrock(bedrock_client, model_size="Haiku")
        elif llm == "Claude 3.5 Sonnet":
            return LLM_Claude3_Bedrock(bedrock_client, model_size="Sonnet 3.5")
        elif llm == "Claude 3.5 Sonnet - Anthropic":
            return LLM_Claude3_Anthropic(
                bedrock_client, model_size="Sonnet 3.5 Anthropic"
            )

    def __str__(self):
        return self.llm_description

    def __repr__(self):
        return self.llm_description

    def __call__(
        self,
        msg,
        b64image=None,
        system_prompt="You are a helpful assistant. Do not use emojis in the answers.",
        chat_history=[],
        postpend="",
        extra_stop_sequences=[],
        tools=None,
        tool_invoker_fn=None,
    ):
        """Calls the LLM in streaming mode
        Arguments:
        system_prompt: prompt that should persist across questions, using specialist attention
        msg: next user message
        chat_history: list of lists. Each inner element should contain [<user msg>, <assistant msg>]
        """
        assert isinstance(
            extra_stop_sequences, list
        ), "extra_stop_sequences should be a list of strings"
        call_list = self._prepare_call_list_from_history(
            system_prompt, msg, b64image, chat_history
        )
        prompt = self._prepare_prompt_from_list(call_list)
        self.last_prompt = str(prompt) + postpend
        if tools is None:
            return self.invoke_streaming(
                prompt, postpend=postpend, extra_stop_sequences=extra_stop_sequences
            )
        else:
            return self.invoke_streaming(
                prompt,
                postpend=postpend,
                extra_stop_sequences=extra_stop_sequences,
                tools=tools,
                tool_invoker_fn=tool_invoker_fn,
            )

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
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64image,
                            },
                        },
                        {"type": "text", "text": msg},
                    ],
                }
            )
        return history_list


class LLM_Claude3_Anthropic(LLM_Bedrock):
    def __init__(self, bedrock_client, model_size):
        """Constructor
        Arguments:
            bedrock_client - Instance of boto3.client(service_name='bedrock-runtime')
                to use when making calls to bedrock models
        """
        if model_size == "Sonnet 3.5 Anthropic":
            self.model_id = "claude-3-5-sonnet-20240620"
            self.anthropic_client = anthropic.Anthropic()
            self.llm_description = (
                "Anthropic Claude 3.5 Sonnet (Small-size LLM) - directly from Anthropic"
            )

        self.config = {
            # "messages": prompt,
            # "system": sysprompt,
            "max_tokens": 3500,
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
            # try:
            self.debug_body = body
            llm_body_changed = True
            while llm_body_changed:
                llm_body_changed = False
                response = self.anthropic_client.messages.create(**body)
                word_count = len(re.findall(r"\w+", str(body["messages"])))
                print(f"Invoking {self.llm_description}. Word count: {word_count}")

                # stream responses
                partial_ans = self._response_gen(response, postpend)
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
        # except Exception as e:
        #     print(
        #         f'Error {str(e)}. Prompt length: {len(str(body["messages"]))}\n\nRetrying {k}...'
        #     )
        #     time.sleep(int(cur_fail_sleep))
        #     cur_fail_sleep *= 1.2

        raise
        # return response

        # except ClientError:
        #    logger.error("Couldn't invoke Claude")
        #    raise

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


class LLM_Claude3_Bedrock(LLM_Bedrock):
    def __init__(self, bedrock_client, model_size):
        """Constructor
        Arguments:
            bedrock_client - Instance of boto3.client(service_name='bedrock-runtime')
                to use when making calls to bedrock models
        """
        if model_size == "Opus":
            self.model_id = "anthropic.claude-3-opus-20240229-v1:0"
            self.llm_description = "Anthropic Claude 3.0 Opus (Large LLM)"
        elif model_size == "Sonnet":
            self.model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
            self.llm_description = "Anthropic Claude 3.0 Sonnet (Medium-size LLM)"
        elif model_size == "Sonnet 3.5":
            self.model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
            self.llm_description = "Anthropic Claude 3.5 Sonnet (Medium-size LLM)"
            assert False, "Model still not available"
        elif model_size == "Haiku":
            self.model_id = "anthropic.claude-3-haiku-20240307-v1:0"
            self.llm_description = "Anthropic Claude 3.0 Haiku (Small-size LLM)"

        self.bedrock_client = bedrock_client
        self.config = {
            # "messages": prompt,
            # "system": sysprompt,
            "max_tokens": 3500,
            "temperature": 0.5,  # 0.5 is default,
            # "top_k": 250,
            # "top_p": 1,
            "stop_sequences": [],  # the regular is already implemented
            "anthropic_version": "bedrock-2023-05-31",
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
                # print(body)
                llm_body_changed = True
                while llm_body_changed:
                    llm_body_changed = False
                    response = self.bedrock_client.invoke_model_with_response_stream(
                        modelId=self.model_id, body=json.dumps(body)
                    )
                    word_count = len(re.findall(r"\w+", str(body["messages"])))
                    print(f"Invoking {self.llm_description}. Word count: {word_count}")

                    # stream responses
                    partial_ans = self._response_gen(response["body"], postpend)
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
        # return response

        # except ClientError:
        #    logger.error("Couldn't invoke Claude")
        #    raise

    def _response_gen(self, response_body, postpend=""):
        cur_ans = ""
        cur_tool_spec = None
        for x in response_body:
            out_dict = json.loads(x["chunk"]["bytes"])
            txt = ""
            if "content_block" in out_dict.keys():
                if out_dict["content_block"]["type"] == "text":
                    txt = out_dict["content_block"]["text"]
                elif out_dict["content_block"]["type"] == "tool_use":
                    cur_tool_spec = out_dict["content_block"].copy()
                    cur_tool_spec["input"] = ""

            elif "delta" in out_dict.keys():
                txt = out_dict["delta"].get("text", "")
                if cur_tool_spec is not None:
                    cur_tool_spec["input"] += out_dict["delta"].get("partial_json", "")

            if txt != "":
                cur_ans += txt
                yield postpend + cur_ans
            stop_reason = (
                out_dict["delta"].get("stop_reason", None)
                if "delta" in out_dict.keys()
                else None
            )
            if stop_reason is not None and stop_reason == "stop_sequence":
                stop_txt = out_dict["delta"]["stop_sequence"]
                yield postpend + cur_ans + stop_txt
                break
        if cur_tool_spec is not None:
            cur_tool_spec["input"] = json.loads(cur_tool_spec["input"])
        self.cur_tool_spec = cur_tool_spec
        self.stop_reason = stop_reason


class LLM_Mixtral8x7b_Bedrock(LLM_Bedrock):
    # https://www.promptingguide.ai/models/mixtral
    def __init__(self, bedrock_client):
        """Constructor
        Arguments:
            bedrock_client - Instance of boto3.client(service_name='bedrock-runtime')
                to use when making calls to bedrock models
        """

        self.bedrock_client = bedrock_client
        self.llm_description = "Mistral Mixtral 8x7B LLM"
        self.config = {
            # "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.3,  # 0.5 is default,
            "top_k": 50,
            "top_p": 0.9,
        }
        # requests and answer word count
        self.word_counts = []

    def _prepare_prompt_from_list(self, msg_list):
        """Receives a list of dictionaries containing keys
        'role' and 'content' and produces the relevant formatting for the LLM
        """
        return format_messages_for_mistral(msg_list)

    def invoke_streaming(
        self, prompt, postpend="", extra_stop_sequences=[], max_retries=25
    ):
        """
        Invokes the Llama2 large-language model to run an inference
        using the input provided in the request body.

        :param prompt: The prompt to be answered
        :param postpend: Extra text to append, to `put words in the mouth` of the LLM
        :return: Inference response from the model.
        """

        # try:
        # The different model providers have individual request and response formats.
        # For the format, ranges, and default values for Meta Llama 2 Chat, refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html

        # start time
        t0 = time.time()
        body = self.config.copy()
        # body['stop_sequences'] = body['stop_sequences'] + extra_stop_sequences
        body["prompt"] = prompt + postpend

        cur_fail_sleep = 1
        for k in range(max_retries):
            # try:
            response = self.bedrock_client.invoke_model_with_response_stream(
                modelId="mistral.mixtral-8x7b-instruct-v0:1", body=json.dumps(body)
            )
            word_count = len(re.findall(r"\w+", body["prompt"]))
            print(f"Invoking Mixtral 8x7B. Word count: {word_count}")

            cur_ans = ""
            for x in response["body"]:
                partial = json.loads(x["chunk"]["bytes"])["outputs"][0]["text"]
                cur_ans += partial

                # we include those again so there's no need to leave these in the answer
                cur_ans = cur_ans.replace("<s>", "").replace("</s>", "")

                # we have to manually check for stopword generation
                extra_stop_break = False
                for x in extra_stop_sequences:
                    cur_split = cur_ans.split(x)
                    if len(cur_split) > 1:
                        cur_ans = cur_split[0] + x
                        extra_stop_break = True
                        break

                yield postpend + cur_ans

                if extra_stop_break:
                    break
                # stop_reason = json.loads(x['chunk']['bytes'])['stop_reason']

            ans_word_count = len(re.findall(r"\w+", postpend + cur_ans))
            self.word_counts.append(
                {
                    "request_word_count": word_count,
                    "answer_word_count": ans_word_count,
                    "price_estimate": 0.001
                    * (0.00195 * word_count + 0.00256 * ans_word_count),
                    "exec_time_in_s": time.time() - t0,
                }
            )

            return
        # except Exception as e:
        #    print(f'Error {str(e)}. Prompt length: {len(body["prompt"])}\n\nRetrying {k}...')
        #    time.sleep(int(cur_fail_sleep))
        #    cur_fail_sleep *= 1.2

        raise
        # return response

        # except ClientError:
        #    logger.error("Couldn't invoke Claude")
        #    raise


class LLM_Claude2_1_Bedrock(LLM_Bedrock):
    def __init__(self, bedrock_client):
        """Constructor
        Arguments:
            bedrock_client - Instance of boto3.client(service_name='bedrock-runtime')
                to use when making calls to bedrock models
        """

        self.bedrock_client = bedrock_client
        self.llm_description = "Anthropic Claude 2.1 LLM"
        self.config = {
            # "prompt": prompt,
            "max_tokens_to_sample": 2500,
            "temperature": 0.3,  # 0.5 is default,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": ["\n\nHuman:"],
            "anthropic_version": "bedrock-2023-05-31",
        }
        # requests and answer word count
        self.word_counts = []

    def _prepare_prompt_from_list(self, msg_list):
        """Receives a list of dictionaries containing keys
        'role' and 'content' and produces the relevant formatting for the LLM
        """
        return format_messages_for_claude(msg_list)

    def invoke_streaming(
        self, prompt, postpend="", extra_stop_sequences=[], max_retries=25
    ):
        """
        Invokes the Claude large-language model to run an inference
        using the input provided in the request body.

        :param prompt: The prompt to be answered
        :param postpend: Extra text to append, to `put words in the mouth` of the LLM
        :return: Inference response from the model.
        """

        # try:
        # The different model providers have individual request and response formats.
        # For the format, ranges, and default values for Meta Llama 2 Chat, refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html

        # start time
        t0 = time.time()
        body = self.config.copy()
        body["stop_sequences"] = body["stop_sequences"] + extra_stop_sequences
        body["prompt"] = prompt + postpend

        cur_fail_sleep = 1
        for k in range(max_retries):
            try:
                response = self.bedrock_client.invoke_model_with_response_stream(
                    modelId="anthropic.claude-v2:1", body=json.dumps(body)
                )
                word_count = len(re.findall(r"\w+", body["prompt"]))
                print(f"Invoking Claude. Word count: {word_count}")

                cur_ans = ""
                for x in response["body"]:
                    partial = json.loads(x["chunk"]["bytes"])["completion"]
                    cur_ans += partial
                    yield postpend + cur_ans
                    stop_reason = json.loads(x["chunk"]["bytes"])["stop"]
                    if (
                        stop_reason is not None
                        and stop_reason != body["stop_sequences"][0]
                    ):
                        yield postpend + cur_ans + stop_reason

                ans_word_count = len(
                    re.findall(r"\w+", postpend + cur_ans + stop_reason)
                )
                self.word_counts.append(
                    {
                        "request_word_count": word_count,
                        "answer_word_count": ans_word_count,
                        "price_estimate": 0.001
                        * (0.008 * word_count + 0.024 * ans_word_count),
                        "exec_time_in_s": time.time() - t0,
                    }
                )

                return
            except Exception as e:
                print(
                    f'Error {str(e)}. Prompt length: {len(body["prompt"])}\n\nRetrying {k}...'
                )
                time.sleep(int(cur_fail_sleep))
                cur_fail_sleep *= 1.2

        raise
        # return response

        # except ClientError:
        #    logger.error("Couldn't invoke Claude")
        #    raise


class LLM_Claude_Instant_1_2_Bedrock(LLM_Bedrock):
    def __init__(self, bedrock_client):
        """Constructor
        Arguments:
            bedrock_client - Instance of boto3.client(service_name='bedrock-runtime')
                to use when making calls to bedrock models
        """

        self.bedrock_client = bedrock_client
        self.llm_description = "Anthropic Claude Instant 1.0 LLM"
        self.config = {
            # "prompt": prompt,
            "max_tokens_to_sample": 600,
            "temperature": 0.3,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": ["\n\nHuman:"],
            "anthropic_version": "bedrock-2023-05-31",
        }
        # requests and answer word count
        self.word_counts = []

    def _prepare_prompt_from_list(self, msg_list):
        """Receives a list of dictionaries containing keys
        'role' and 'content' and produces the relevant formatting for the LLM
        """
        return format_messages_for_claude(msg_list)

    def invoke_streaming(
        self, prompt, postpend="", extra_stop_sequences=[], max_retries=25
    ):
        """
        Invokes the Claude large-language model to run an inference
        using the input provided in the request body.

        :param prompt: The prompt to be answered
        :param postpend: Extra text to append, to `put words in the mouth` of the LLM
        :return: Inference response from the model.
        """

        # try:
        # The different model providers have individual request and response formats.
        # For the format, ranges, and default values for Meta Llama 2 Chat, refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html
        # start time
        t0 = time.time()

        body = self.config.copy()
        body["stop_sequences"] = body["stop_sequences"] + extra_stop_sequences
        body["prompt"] = prompt + postpend

        cur_fail_sleep = 1
        for k in range(max_retries):
            try:
                response = self.bedrock_client.invoke_model_with_response_stream(
                    modelId="anthropic.claude-instant-v1", body=json.dumps(body)
                )
                word_count = len(re.findall(r"\w+", body["prompt"]))
                print(f"Invoking Claude Instant. Word count: {word_count}")

                cur_ans = ""
                for x in response["body"]:
                    partial = json.loads(x["chunk"]["bytes"])["completion"]
                    cur_ans += partial
                    yield postpend + cur_ans
                    stop_reason = json.loads(x["chunk"]["bytes"])["stop"]
                    if (
                        stop_reason is not None
                        and stop_reason != body["stop_sequences"][0]
                    ):
                        yield postpend + cur_ans + stop_reason

                ans_word_count = len(
                    re.findall(r"\w+", postpend + cur_ans + stop_reason)
                )
                self.word_counts.append(
                    {
                        "request_word_count": word_count,
                        "answer_word_count": ans_word_count,
                        "price_estimate": 0.001
                        * (0.0008 * word_count + 0.0024 * ans_word_count),
                        "exec_time_in_s": time.time() - t0,
                    }
                )

                return
            except Exception as e:
                print(
                    f'Error {str(e)}. Prompt length: {len(body["prompt"])}\n\nRetrying {k}...'
                )
                time.sleep(int(cur_fail_sleep))
                cur_fail_sleep *= 1.2

        raise
        # return response

        # except ClientError:
        #    logger.error("Couldn't invoke Claude")
        #    raise


class LLM_Llama13b(LLM_Bedrock):
    """https://huggingface.co/blog/llama2#how-to-prompt-llama-2"""

    def __init__(self, bedrock_client):
        """Constructor
        Arguments:
            bedrock_client - Instance of boto3.client(service_name='bedrock-runtime')
                to use when making calls to bedrock models
        """

        self.bedrock_client = bedrock_client
        self.llm_description = "Llama2 13b v1 LLM"

        self.config = {
            "max_gen_len": 1024,
            "top_p": 0.9,
            "temperature": 0.6,
        }

        # requests and answer word count
        self.word_counts = []

    def _prepare_prompt_from_list(self, msg_list):
        """Receives a list of dictionaries containing keys
        'role' and 'content' and produces the relevant formatting for the LLM
        """
        return format_messages_for_llama(msg_list)

    def invoke_streaming(
        self, prompt, postpend="", extra_stop_sequences=[], max_retries=25
    ):
        """
        Invokes the Llama2 large-language model to run an inference
        using the input provided in the request body.

        :param prompt: The prompt to be answered
        :param postpend: Extra text to append, to `put words in the mouth` of the LLM
        :return: Inference response from the model.
        """

        # try:
        # The different model providers have individual request and response formats.
        # For the format, ranges, and default values for Meta Llama 2 Chat, refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html

        # start time
        t0 = time.time()
        body = self.config.copy()
        # body['stop_sequences'] = body['stop_sequences'] + extra_stop_sequences
        body["prompt"] = prompt + postpend

        cur_fail_sleep = 1
        for k in range(max_retries):
            # try:
            response = self.bedrock_client.invoke_model_with_response_stream(
                modelId="meta.llama2-13b-chat-v1", body=json.dumps(body)
            )
            word_count = len(re.findall(r"\w+", body["prompt"]))
            print(f"Invoking Llama 2 chat 13b. Word count: {word_count}")

            cur_ans = ""
            for x in response["body"]:
                partial = json.loads(x["chunk"]["bytes"])["generation"]
                cur_ans += partial

                # we include those again so there's no need to leave these in the answer
                cur_ans = cur_ans.replace("<s>", "").replace("</s>", "")

                # we have to manually check for stopword generation
                extra_stop_break = False
                for x in extra_stop_sequences:
                    cur_split = cur_ans.split(x)
                    if len(cur_split) > 1:
                        cur_ans = cur_split[0] + x
                        extra_stop_break = True
                        break

                yield postpend + cur_ans

                if extra_stop_break:
                    break
                # stop_reason = json.loads(x['chunk']['bytes'])['stop_reason']

            ans_word_count = len(re.findall(r"\w+", postpend + cur_ans))
            self.word_counts.append(
                {
                    "request_word_count": word_count,
                    "answer_word_count": ans_word_count,
                    "price_estimate": 0.001
                    * (0.00075 * word_count + 0.001 * ans_word_count),
                    "exec_time_in_s": time.time() - t0,
                }
            )

            return
        # except Exception as e:
        #    print(f'Error {str(e)}. Prompt length: {len(body["prompt"])}\n\nRetrying {k}...')
        #    time.sleep(int(cur_fail_sleep))
        #    cur_fail_sleep *= 1.2

        raise
        # return response

        # except ClientError:
        #    logger.error("Couldn't invoke Claude")
        #    raise


class LLM_Llama70b(LLM_Bedrock):
    """https://huggingface.co/blog/llama2#how-to-prompt-llama-2"""

    def __init__(self, bedrock_client):
        """Constructor
        Arguments:
            bedrock_client - Instance of boto3.client(service_name='bedrock-runtime')
                to use when making calls to bedrock models
        """

        self.bedrock_client = bedrock_client
        self.llm_description = "Llama2 70b v1 LLM"

        self.config = {
            "max_gen_len": 1024,
            "top_p": 0.9,
            "temperature": 0.6,
        }

        # requests and answer word count
        self.word_counts = []

    def _prepare_prompt_from_list(self, msg_list):
        """Receives a list of dictionaries containing keys
        'role' and 'content' and produces the relevant formatting for the LLM
        """
        return format_messages_for_llama(msg_list)

    def invoke_streaming(
        self, prompt, postpend="", extra_stop_sequences=[], max_retries=25
    ):
        """
        Invokes the Llama2 large-language model to run an inference
        using the input provided in the request body.

        :param prompt: The prompt to be answered
        :param postpend: Extra text to append, to `put words in the mouth` of the LLM
        :return: Inference response from the model.
        """

        # try:
        # The different model providers have individual request and response formats.
        # For the format, ranges, and default values for Meta Llama 2 Chat, refer to:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-meta.html

        # start time
        t0 = time.time()
        body = self.config.copy()
        # body['stop_sequences'] = body['stop_sequences'] + extra_stop_sequences
        body["prompt"] = prompt + postpend

        cur_fail_sleep = 1
        for k in range(max_retries):
            # try:
            response = self.bedrock_client.invoke_model_with_response_stream(
                modelId="meta.llama2-70b-chat-v1", body=json.dumps(body)
            )
            word_count = len(re.findall(r"\w+", body["prompt"]))
            print(f"Invoking Llama 2 chat 70b. Word count: {word_count}")

            cur_ans = ""
            for x in response["body"]:
                partial = json.loads(x["chunk"]["bytes"])["generation"]
                cur_ans += partial

                # we include those again so there's no need to leave these in the answer
                cur_ans = cur_ans.replace("<s>", "").replace("</s>", "")

                # we have to manually check for stopword generation
                extra_stop_break = False
                for x in extra_stop_sequences:
                    cur_split = cur_ans.split(x)
                    if len(cur_split) > 1:
                        cur_ans = cur_split[0] + x
                        extra_stop_break = True
                        break

                yield postpend + cur_ans

                if extra_stop_break:
                    break
                # stop_reason = json.loads(x['chunk']['bytes'])['stop_reason']

            ans_word_count = len(re.findall(r"\w+", postpend + cur_ans))
            self.word_counts.append(
                {
                    "request_word_count": word_count,
                    "answer_word_count": ans_word_count,
                    "price_estimate": 0.001
                    * (0.00195 * word_count + 0.00256 * ans_word_count),
                    "exec_time_in_s": time.time() - t0,
                }
            )

            return
        # except Exception as e:
        #    print(f'Error {str(e)}. Prompt length: {len(body["prompt"])}\n\nRetrying {k}...')
        #    time.sleep(int(cur_fail_sleep))
        #    cur_fail_sleep *= 1.2

        raise
        # return response

        # except ClientError:
        #    logger.error("Couldn't invoke Claude")
        #    raise


def format_messages_for_claude(messages: List[Dict[str, str]]) -> List[str]:
    """Format messages for Claude 2.1 chat models."""
    prompt: List[str] = []

    if messages[0]["role"] == "system":
        prompt.append(messages[0]["content"])
        messages = messages[1:]

    assert messages[-1]["role"] == "user", "Last message has to be from user"

    last_msg_sender = "assistant"
    for msg in messages:
        assert (
            msg["role"] != last_msg_sender
        ), "Multiple messages from the same sender in sequence"
        assert msg["role"] in (
            "user",
            "assistant",
        ), "Role has to be user, assistant or system (first message only)"
        last_msg_sender = msg["role"]
        if msg["role"] == "assistant":
            prompt.append("\n\nAssistant: " + msg["content"])
        elif msg["role"] == "user":
            prompt.append("\n\nHuman: " + msg["content"])

    prompt.append("\n\nAssistant: ")
    return "".join(prompt)


def format_messages_for_llama(messages: List[Dict[str, str]]) -> List[str]:
    """Format messages for Llama-2 chat models.

    The model only supports 'system', 'user' and 'assistant' roles, starting with 'system', then 'user' and
    alternating (u/a/u/a/u...). The last message must be from 'user'.
    """
    prompt: List[str] = []
    assert messages[-1]["role"] == "user", "Last message has to be from user"

    if messages[0]["role"] == "system":
        content = "".join(
            [
                "<<SYS>>\n",
                messages[0]["content"],
                "\n<</SYS>>\n\n",
                messages[1]["content"],
            ]
        )
        messages = [{"role": messages[1]["role"], "content": content}] + messages[2:]

    for user, answer in zip(messages[::2], messages[1::2]):
        prompt.extend(
            [
                "<s>",
                "[INST] ",
                (user["content"]).strip(),
                " [/INST] ",
                (answer["content"]).strip(),
                "</s>",
            ]
        )

    prompt.extend(["<s>", "[INST] ", (messages[-1]["content"]).strip(), " [/INST] "])

    return "".join(prompt)


def format_messages_for_mistral(messages: List[Dict[str, str]]) -> List[str]:
    """Format messages for mistral chat models.

    The model only supports 'system', 'user' and 'assistant' roles, starting with 'system', then 'user' and
    alternating (u/a/u/a/u...). The last message must be from 'user'.
    """
    prompt: List[str] = []
    assert messages[-1]["role"] == "user", "Last message has to be from user"

    if messages[0]["role"] == "system":
        content = "".join([messages[0]["content"], "\n\n", messages[1]["content"]])
        messages = [{"role": messages[1]["role"], "content": content}] + messages[2:]

    for user, answer in zip(messages[::2], messages[1::2]):
        prompt.extend(
            [
                "<s>",
                "[INST] ",
                (user["content"]).strip(),
                " [/INST] ",
                (answer["content"]).strip(),
                "</s>",
            ]
        )

    prompt.extend(["<s>", "[INST] ", (messages[-1]["content"]).strip(), " [/INST] "])

    return "".join(prompt)
