import os
import re
import json
import time
import uuid
import base64
from io import BytesIO

import numpy as np
from PIL import Image


def _adjust_msg_for_gradio_ui(x, show_scratchpad=False, show_calls=False):
    """Adjusts a string to be displayed in the gradio UI

    <sup><sub>This is subbed-sup text</sub></sup>
    """
    # <!--This is a comment. Comments are not displayed in the browser-->
    # return x.replace('<answer>', '\n\nAnswer:').replace('</answer>', '')
    # x = x.replace('<', '[').replace('>', ']')

    if show_scratchpad:
        x = x.replace("<scratchpad>", "```\n[scratchpad]")
        x = x.replace("</scratchpad>", "[/scratchpad]\n```\n")
    else:
        # System.Text.RegularExpressions.Regex.Replace(test_str,"<a>[\S\s]*?</a>\s*", "")
        x = re.sub(r"<scratchpad>[\S\s]*?</scratchpad>\s*", "", x)
        # x = x.replace('<scratchpad>', '<!--<scratchpad>')
        # x = x.replace('</scratchpad>', '</scratchpad>-->')

    if not show_calls:
        x = re.sub(r"<function_calls>[\S\s]*?</function_calls>\s*", "", x)

    x = re.sub(r"<function_results>[\S\s]*?</function_results>\s*", "", x)
    # x = x.replace('<function_results>', '<!--<function_results>')
    # x = x.replace('</function_results>', '<function_results>-->')

    x = x.replace("<answer>", "<answer><b>").replace("</answer>", "</answer></b>")
    return x


class LLMInterface:
    def __init__(
        self,
        system_prompt,
        llm,
        llm_tools,
        rpg,
        output_mode="chat_bot",
        chat_log_folder="chat_logs",
        show_extra_info=True,
    ):
        """Constructor

        Arguments
        system_prompt: desired system prompt
        llm: LLM to use. Method invoke_streaming will be used
        llm_tools: implementation of resolutor to LLM tool calls. Has to implement invoke_from_cmd
        rpg: Instance of RAGPromptGenerator. Needs to have post_anti_hallucination property
        output_mode: chat_bot uses Gradio customized chatbot that has to
            receive the whole history. Otherwise uses gradio chatinterface
        chat_log_folder: folder to save chat to. If None, does not save chat
        show_extra_info: show extra info like tools used and scratchpad in the UI?
        """
        self.system_prompt = system_prompt
        self.llm = llm
        self.lt = llm_tools
        self.rpg = rpg
        self.chat_log_folder = chat_log_folder
        self.show_extra_info = show_extra_info

        # keep a hash of histories so we can send to the UI
        # something different than what has been generated
        self.history_log = {}

        valid_output_modes = ["chat_interface", "chat_bot"]
        assert (
            output_mode in valid_output_modes
        ), f"Output mode must be one of {valid_output_modes}"
        self.output_mode = output_mode
        # if requested by the LLM, get rid of the past history
        self.erase_past = False
        # keep some execution logs
        self.log = []

        # handle native tool use
        if self.rpg is not None and self.rpg.use_native_tools:
            self.native_tools = [x.tool_description for x in self.lt.tools]
            self.tool_invoker_fn = self.lt.invoke_tool
            self.extra_stop_sequences = []
        else:
            self.native_tools = None
            self.tool_invoker_fn = None
            self.extra_stop_sequences = ["</function_calls>"]

    def _format_msg(
        self, x, message, chat_history, show_ans_only=False, extra_info=None
    ):
        # the "<path_to_" substring from native tools has to be appended to the final answer for file display
        if self.output_mode == "chat_interface":
            return _adjust_msg_for_gradio_ui(x)
        elif self.output_mode == "chat_bot":
            cur_history = chat_history.copy()
            cur_ans = _adjust_msg_for_gradio_ui(x)

            ans_only = cur_ans.split("<answer>")
            if len(ans_only) > 1 and show_ans_only:
                cur_ans = ans_only[-1]
            elif show_ans_only:
                cur_ans = ""

            info_msg = None
            if extra_info is None or not self.show_extra_info:
                cur_history += [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": cur_ans},
                ]
            else:
                info_msg = {
                    "role": "assistant",
                    "content": extra_info.get("content", ""),
                    "metadata": extra_info["metadata"],
                }
                cur_history += [
                    {"role": "user", "content": message},
                    info_msg,
                    {"role": "assistant", "content": cur_ans},
                ]

            # find out if we should be showing an image
            image_candidates = x.split("<path_to_image>")
            shown_images = {}
            for k in range(1, len(image_candidates)):
                image_candidate = image_candidates[k].split("</path_to_image>")[0]

                if os.path.isfile(image_candidate) and not shown_images.get(
                    image_candidate, False
                ):
                    cur_history.append(
                        {
                            "role": "assistant",
                            "content": {"path": image_candidate, "alt_text": "media"},
                        }
                    )
                    shown_images[image_candidate] = True

            # find out if we should be showing an audio
            audio_candidates = x.split("<path_to_audio>")
            shown_audios = {}
            for k in range(1, len(audio_candidates)):
                audio_candidate = audio_candidates[k].split("</path_to_audio>")[0]

                if os.path.isfile(audio_candidate) and not shown_audios.get(
                    audio_candidate, False
                ):
                    cur_history.append(
                        {
                            "role": "assistant",
                            "content": {"path": audio_candidate, "alt_text": "media"},
                        }
                    )
                    shown_audios[audio_candidate] = True

            # find out if we should add a downloadable file
            file_candidates = x.split("<path_to_file>")
            shown_files = {}
            for k in range(1, len(file_candidates)):
                file_candidate = file_candidates[k].split("</path_to_file>")[0]
                if os.path.isfile(file_candidate) and not shown_files.get(
                    file_candidate, False
                ):
                    cur_history.append(
                        {
                            "role": "assistant",
                            "content": {"path": file_candidate, "alt_text": "media"},
                        }
                    )
                    shown_files[file_candidate] = True

            # figure out what should go into the scratchpad
            scratchpad_info = x.split("<scratchpad>")
            if len(scratchpad_info) > 1:
                scratchpad_info = scratchpad_info[-1].split("</scratchpad>")[0]
            else:
                scratchpad_info = ""
            if info_msg is not None:
                info_msg["content"] = (
                    extra_info.get("content", "") + "\n" + scratchpad_info
                )

            # also put function calls in there
            func_call_info = x.split("<function_calls>")
            if len(func_call_info) > 1:
                scratchpad_info = (
                    scratchpad_info
                    + "\n\nFunction call:\n\n"
                    + func_call_info[-1].split("</function_calls>")[0]
                )

            # make sure to send ChatBot history last
            return "", scratchpad_info, None, cur_history

    def _rem_none(self, history):
        """Returns a copy of history but with None messages from users removed"""
        return str([x for x in history if x[0] is not None])

    def chat_with_function_caller(self, msg, images, ui_history=[], username=""):
        """Performs conversation with the LLM agent

        Arguments:
            msg: next user message
            image: user input image
            ui_history: history in the user interface. The first is used to recover the state in memory
            username: user name of the user logged in the UI
        """
        image_strings = None
        if images is not None:
            image_strings = []
            if not isinstance(images, list):
                images = [images]
            for image in images:
                if image is None:
                    image_strings.append(None)
                else:
                    npimg = np.array(image, dtype=np.uint8)
                    pil_img = Image.fromarray(npimg)
                    buff = BytesIO()
                    pil_img.save(buff, format="JPEG")
                    image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
                    image_strings.append(image_string)

        t0 = time.time()

        if len(ui_history) > 0:
            chat_id = ui_history[0]["content"]
            history = self.history_log[chat_id]
            # with open('ui_debug.txt', 'w') as f:
            #    f.write(str([msg, history]))
        else:
            chat_id = str(uuid.uuid4())
            ui_history = [{"role": "assistant", "content": chat_id}]
            history = []

        ans2 = self.llm(
            msg,
            b64images=image_strings,
            system_prompt=self.system_prompt,
            chat_history=history,
            postpend=self.rpg.post_anti_hallucination
            if (self.rpg is not None and not self.rpg.use_native_tools)
            else "",
            extra_stop_sequences=self.extra_stop_sequences,
            tools=self.native_tools,
            tool_invoker_fn=self.lt.invoke_tool if self.lt is not None else None,
        )

        extra_info = {
            "metadata": {"title": "🧠", "status": "pending"},
        }
        x = ""
        for x in ans2:
            if self.lt is not None and hasattr(self.llm, "cur_tool_spec"):
                extra_info = {
                    "metadata": {"title": "🛠️", "status": "pending"},
                    "content": ", ".join([x["tool_name"] for x in self.lt.invoke_log]),
                }
            yield self._format_msg(x, msg, ui_history, extra_info=extra_info)
        # initial_ans = self._format_msg(x, msg, ui_history)
        # yield initial_ans

        cur_answer = x

        cur_answer_split = cur_answer.split("<function_calls>")
        while (
            len(cur_answer_split) > 1
            and self.lt is not None
            and not self.rpg.use_native_tools
        ):
            # this loop means that manual tool usage is needed
            cur_func_log = {}

            xml_to_parse = cur_answer_split[-1].split("</function_calls>")[0]
            post_prompt = self.lt.invoke_from_cmd(xml_to_parse, username=username)

            cur_func_log["Parse and exec query"] = {
                "exec_time": time.time() - t0,
            }
            t0 = time.time()

            cur_postpend = cur_answer + post_prompt
            # yield self._format_msg(x, msg, ui_history)

            # note: parameters tools and tool_invoker_fn are not used
            # because this call fulfills manual tool use requests
            # ie. if there are native tools, this loop should never happen
            ans2 = self.llm(
                msg,
                system_prompt=self.system_prompt,
                chat_history=history,
                postpend=cur_postpend if not self.rpg.use_native_tools else "",
                extra_stop_sequences=self.extra_stop_sequences,
            )

            for x in ans2:
                # pass
                yield self._format_msg(x, msg, ui_history)
            # yield self._format_msg(x, msg, ui_history)

            log_dict = self.llm.word_counts[-1].copy()
            cur_func_log["Analyze query with LLM"] = {
                "exec_time": time.time() - t0,
                "word_count": log_dict,
            }
            t0 = time.time()
            cur_answer = x
            last_update = cur_answer.replace(cur_postpend, "")
            cur_answer_split = last_update.split("<function_calls>")

        history_to_append = []
        tool_results = []
        if hasattr(self.llm, "tool_use_added_msgs"):
            history_to_append.append({"role": "user", "content": msg})
            tool_results.append("\n")
            for x in self.llm.tool_use_added_msgs:
                history_to_append.append(x)

                # enable media display in the Gradio UI - amazon Nova
                if x["role"] == "user" and x["content"][0].get("toolResult"):
                    cur_tool_result = x["content"][0]["toolResult"]["content"][0][
                        "text"
                    ]
                    tool_results.append(
                        cur_tool_result if "<path_to_" in cur_tool_result else ""
                    )
                # enable media display in the Gradio UI - anthropic
                if x["role"] == "user" and x["content"][0].get("content"):
                    cur_tool_result = x["content"][0]["content"]
                    tool_results.append(
                        cur_tool_result if "<path_to_" in cur_tool_result else ""
                    )
                # enable media display in the Gradio UI - openai
                if x["role"] == "tool":
                    cur_tool_result = x["content"]
                    tool_results.append(
                        cur_tool_result if "<path_to_" in cur_tool_result else ""
                    )
            history_to_append.append({"role": "assistant", "content": cur_answer})
        else:
            history_to_append.append([msg, cur_answer])

        tool_results = "\n".join(tool_results)
        self.history_log[chat_id] = history + history_to_append

        try:
            chat_log_dir = self.chat_log_folder
            if username is not None and username.strip() != "":
                cur_user_to_log = username
            else:
                cur_user_to_log = "unknown"
            if chat_log_dir is not None:
                os.makedirs(chat_log_dir, exist_ok=True)
                with open(
                    f"{chat_log_dir}/{cur_user_to_log}-{chat_id}.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(json.dumps(self.history_log[chat_id]))
        except Exception as ex:
            print(
                f"Could not log chat to folder `{self.chat_log_folder}`. Reason: {str(ex)}"
            )

        extra_info["metadata"]["status"] = "done"
        final_response_ui = self._format_msg(
            cur_answer + tool_results, msg, ui_history, extra_info=extra_info
        )
        if self.lt is not None:
            self.lt.invoke_log = []
        yield final_response_ui
        print("Final response sent.")
