""" Set of available and useful LLMs (mostly posted on AWS Bedrock)
"""


class LLM_Service:
    def __str__(self):
        return self.llm_description

    def __repr__(self):
        return self.llm_description

    def __call__(
        self,
        msg,
        b64images=None,
        system_prompt="You are a helpful assistant. Do not use emojis in the answers.",
        chat_history=[],
        postpend="",
        extra_stop_sequences=[],
        tools=None,
        tool_invoker_fn=None,
        max_retries=3,
        cur_fail_sleep=60,
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
            system_prompt, msg, b64images, chat_history
        )
        prompt = self._prepare_prompt_from_list(call_list)
        self.last_prompt = str(prompt) + postpend
        if tools is None:
            return self.invoke_streaming(
                prompt,
                postpend=postpend,
                extra_stop_sequences=extra_stop_sequences,
                max_retries=max_retries,
                cur_fail_sleep=cur_fail_sleep,
            )
        else:
            return self.invoke_streaming(
                prompt,
                postpend=postpend,
                extra_stop_sequences=extra_stop_sequences,
                tools=tools,
                tool_invoker_fn=tool_invoker_fn,
                max_retries=max_retries,
                cur_fail_sleep=cur_fail_sleep,
            )

    def _prepare_call_list_from_history(
        self,
        system_prompt,
        msg,
        b64images,
        chat_history,
        context_reset_string="[|[PAST_FORGOTTEN]|]",
    ):
        """Prepares the prompt for the next interaction with the LLM.
        This image preparation is suited for Anthropic's Claude
        """
        history_list = [
            {"role": "system", "content": system_prompt},
        ]
        for x in chat_history:
            # allows the LLM to determine that the past is no longer relevant, to free up context
            if context_reset_string in str(x):
                history_list = [
                    history_list[0],
                    {
                        "role": "user",
                        "content": str(x),
                    },
                ]
            elif isinstance(x, dict):
                history_list.append(x)
            else:
                history_list.append({"role": "user", "content": x[0]})
                history_list.append({"role": "assistant", "content": str(x[1])})

        if b64images is None:
            history_list.append({"role": "user", "content": msg})
        else:
            if not isinstance(b64images, list):
                b64images = [b64images]
            cur_content = []
            for b64img in b64images:
                if b64img is not None:
                    cur_content.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64img,
                            },
                        },
                    )
            cur_content.append({"type": "text", "text": msg})
            history_list.append({"role": "user", "content": cur_content})
        return history_list
