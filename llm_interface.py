import os
import re
import time
import uuid
import base64
from io import BytesIO

import numpy as np
from PIL import Image


def _adjust_msg_for_gradio_ui(x, show_scratchpad=False, show_calls=False):
    """ Adjusts a string to be displayed in the gradio UI

    <sup><sub>This is subbed-sup text</sub></sup>
    """
    # <!--This is a comment. Comments are not displayed in the browser-->
    # return x.replace('<answer>', '\n\nAnswer:').replace('</answer>', '')
    # x = x.replace('<', '[').replace('>', ']')

    if show_scratchpad:
        x = x.replace('<scratchpad>', '```\n[scratchpad]')
        x = x.replace('</scratchpad>', '[/scratchpad]\n```\n')
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

    x = x.replace('<answer>', '<answer><b>').replace('</answer>', '</answer></b>')
    return x


class LLMInterface:
    def __init__(
        self, system_prompt, llm, llm_tools, rpg,
        output_mode='chat_bot'
    ):
        """Constructor

        Arguments
        system_prompt: desired system prompt
        llm: LLM to use. Method invoke_streaming will be used
        llm_tools: implementation of resolutor to LLM tool calls. Has to implement invoke_from_cmd
        rpg: Instance of RAGPromptGenerator. Needs to have post_anti_hallucination property
        output_mode: chat_bot uses Gradio customized chatbot that has to
            receive the whole history. Otherwise uses gradio chatinterface
        """
        self.system_prompt = system_prompt
        self.llm = llm
        self.lt = llm_tools
        self.rpg = rpg

        # keep a hash of histories so we can send to the UI
        # something different than what has been generated
        self.history_log = {}

        valid_output_modes = ['chat_interface', 'chat_bot']
        assert output_mode in valid_output_modes, f'Output mode must be one of {valid_output_modes}'
        self.output_mode = output_mode
        # if requested by the LLM, get rid of the past history
        self.erase_past = False
        # keep some execution logs
        self.log = []

    def _format_msg(self, x, message, chat_history, include_logs=False, show_ans_only=False):
        if self.output_mode == 'chat_interface':
            return _adjust_msg_for_gradio_ui(x)
        elif self.output_mode == 'chat_bot':
            cur_history = chat_history.copy()
            cur_ans = _adjust_msg_for_gradio_ui(x)

            ans_only = cur_ans.split('<answer>')
            if len(ans_only) > 1 and show_ans_only:
                cur_ans = ans_only[-1]
            elif show_ans_only:
                cur_ans = ''
            cur_history.append([message, cur_ans])

            # find out if we should be showing an image
            image_candidates = x.split('<path_to_image>')
            shown_images = {}
            for k in range(1, len(image_candidates)):
                image_candidate = image_candidates[k].split('</path_to_image>')[0]

                if os.path.isfile(image_candidate) and not shown_images.get(image_candidate, False):
                    cur_history.append([None, (image_candidate, "media")])
                    shown_images[image_candidate] = True
                    # img_html = f"<img src='/file={image_candidate}' style='width: 600px; max-width:none; max-height:none'></img>"
                    # cur_history.append([None, img_html])

            # figure out what should go into the scratchpad
            scratchpad_info =  x.split('<scratchpad>')
            if len(scratchpad_info) > 1:
                scratchpad_info = scratchpad_info[-1].split('</scratchpad>')[0]
            else:
                scratchpad_info = ''

            # also put function calls in there
            func_call_info = x.split('<function_calls>')
            if len(func_call_info) > 1:
                scratchpad_info = scratchpad_info + '\n\nFunction call:\n\n' + func_call_info[-1].split('</function_calls>')[0]

            # retrieve last log
            if include_logs:
                cur_log = self.log[-1]
                scratchpad_info = scratchpad_info + '\n' + str(cur_log)

            # make sure to send ChatBot history last
            return "", scratchpad_info, None, cur_history

    def _rem_none(self, history):
        """ Returns a copy of history but with None messages from users removed
        """
        return str([x for x in history if x[0] is not None])

    def chat_with_function_caller(self, msg, image, ui_history = [], username=''):
        """ Performs conversation with the LLM agent
        """
        image_string = None
        if image is not None:
            npimg = np.array(image, dtype=np.uint8)
            pil_img = Image.fromarray(npimg)
            buff = BytesIO()
            pil_img.save(buff, format="JPEG")
            image_string = base64.b64encode(buff.getvalue()).decode("utf-8")

        t0 = time.time()
        cur_log = {'Function calls': []}

        if len(ui_history) > 0:
            chat_id = ui_history[0][1]
            history = self.history_log[chat_id]
            # with open('ui_debug.txt', 'w') as f:
            #    f.write(str([msg, history]))
        else:
            chat_id = str(uuid.uuid4())
            ui_history = [[None, chat_id]]
            history = []

        cur_log['Prepare initial prompt'] = {'exec_time': time.time() - t0}
        t0 = time.time()

        ans2 = self.llm(
            msg,
            b64image=image_string,
            system_prompt=self.system_prompt,
            chat_history=history,
            postpend=self.rpg.post_anti_hallucination if self.rpg is not None else "",
            extra_stop_sequences=['</function_calls>']
        )
        with open('ui_debug_prompt.txt', 'w') as f:
            f.write(str(self.llm.last_prompt))

        for x in ans2:
            # pass
            yield self._format_msg(x, msg, ui_history)
        # initial_ans = self._format_msg(x, msg, ui_history)
        # yield initial_ans

        cur_answer = x
        cur_log['Compute initial response'] = {
            'exec_time': time.time() - t0,
            'word_count': self.llm.word_counts[-1],
        }
        t0 = time.time()

        cur_answer_split = cur_answer.split('<function_calls>')
        while len(cur_answer_split) > 1 and self.lt is not None:
            cur_func_log = {}

            xml_to_parse = cur_answer_split[-1].split('</function_calls>')[0]
            post_prompt = self.lt.invoke_from_cmd(xml_to_parse, username=username)

            cur_func_log['Parse and exec query'] = {
                'exec_time': time.time() - t0,
            }
            t0 = time.time()

            cur_postpend = cur_answer + post_prompt
            # yield self._format_msg(x, msg, ui_history)

            ans2 = self.llm(
                msg,
                system_prompt=self.system_prompt,
                chat_history=history,
                postpend=cur_postpend,
                extra_stop_sequences=['</function_calls>']
            )

            for x in ans2:
                # pass
                yield self._format_msg(x, msg, ui_history)
            # yield self._format_msg(x, msg, ui_history)

            log_dict = self.llm.word_counts[-1].copy()
            cur_func_log['Analyze query with LLM'] = {
                'exec_time': time.time() - t0,
                'word_count': log_dict,
            }
            t0 = time.time()
            cur_answer = x
            last_update = cur_answer.replace(cur_postpend, "")
            cur_answer_split = last_update.split('<function_calls>')

            cur_log['Function calls'].append(cur_func_log)

        self.log.append(cur_log)
        final_response_ui = self._format_msg(cur_answer, msg, ui_history)

        self.history_log[chat_id] = history + [[msg, cur_answer]]

        yield final_response_ui
        print('Final response sent.')
