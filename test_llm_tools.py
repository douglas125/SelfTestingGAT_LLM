import os
import boto3
import botocore
import gradio as gr

import gat_llm.llm_invoker as inv
from gat_llm.tools.base import LLMTools
from gat_llm.llm_interface import LLMInterface
from gat_llm.prompts.prompt_generator import RAGPromptGenerator


examples = [
    "Give me a summary of your tools and what they do. Answer with a table.",
    "What is in the image?",
    "What day is today?",
    "What day was it 10 days ago?",
    "Make a simple flowchart with A -> B -> C",
    "If I invest $100 with an interest rate of 1% per month, how much will I have in 3 years?",
    "Make a plot of y=x^2",
    "What are the solutions to the equation: x^2 - 1 = 0",
    "If Mark has 3 times more apples than John and they have 40 apples in total, how many apples do each have?",
    "Evaluate the expression exp(2)+sin(4)",
    "faca um qr code estilo vcard para mim. me pergunte as informacoes que precisar",
    "I live in Florence IT. Fetch from the internet relevant news today.",
    "Summarize the economics news in https://www.economist.com/ and https://www.theguardian.com/business/economics . Check which articles show on both or only on one. Answer with a table.",
    "List all your tools. Summarize what each tool does and generate 3 sample questions that it could answer. Answer with a table.",
]

description = """# Retrieval Augmented by Tools

- Before running this demo, set the API key of the LLM you want to use in your environment
Bedrock: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
Anthropic: ANTHROPIC_API_KEY
OpenAI: OPENAI_API_KEY
DeepSeek: DEEPSEEK_API_KEY (cannot be used with OpenAI models)
Maritaca: MARITACA_API_KEY (cannot be used with OpenAI models)
- Select the LLM of your choice. You can't use Maritaca or Deepseek with OpenAI or tools that require OpenAI
- Note that some tools require non-LLM OpenAI models: text_to_image, text_to_speech, speech_to_text
- Select the tools allowed for the LLM
- If a model is unavailable, you need to set the proper API key in the environment before running this interface
"""

# Keep track of previous conversations
history_log = {}


def msg_forward_func(
    msg,
    img_input,
    history,
    system_prompt_prepend,
    selected_llm,
    use_native_LLM_tools,
    allowed_tools,
    request: gr.Request,
):
    if "unavailable" in selected_llm.lower():
        return

    config = botocore.client.Config(
        connect_timeout=9000, read_timeout=9000, region_name="us-west-2"
    )  # us-east-1  us-west-2
    bedrock_client = boto3.client(service_name="bedrock-runtime", config=config)

    # Initialize LLM
    allowed_tool_list = [
        x
        for x in LLMTools.get_all_tools()
        if x.tool_description["name"] in allowed_tools
    ]

    llm_name = selected_llm
    llm = inv.LLM_Provider.get_llm(bedrock_client, llm_name)
    query_llm = inv.LLM_Provider.get_llm(bedrock_client, llm_name)
    lt = LLMTools(query_llm=query_llm, desired_tools=allowed_tool_list)

    tool_descriptions = lt.get_tool_descriptions()
    rpg = RAGPromptGenerator(use_native_tools=use_native_LLM_tools)
    system_prompt = rpg.prompt.replace("{{TOOLS}}", tool_descriptions)
    li = LLMInterface(system_prompt=system_prompt, llm=llm, llm_tools=lt, rpg=rpg)
    li.history_log = history_log

    # Call LLM
    li.system_prompt = system_prompt + "\n" + system_prompt_prepend

    if msg is None or msg.strip() == "":
        msg = "perform task"

    ans_gen = li.chat_with_function_caller(
        msg, img_input, history, username=request.username
    )
    for x in ans_gen:
        txtbox, scratchpad_info, img_input, cur_history = x
        yield txtbox, scratchpad_info, img_input, cur_history, []

    chat_id = cur_history[0]["content"]
    raw_history = li.history_log[chat_id]
    yield txtbox, scratchpad_info, img_input, cur_history, {"raw_history": raw_history}


def main():
    with gr.Blocks() as demo:
        gr.Markdown(description)
        with gr.Column():
            with gr.Row():
                available_models = inv.LLM_Provider.allowed_llms
                if os.environ.get("OPENAI_API_KEY") is None:
                    available_models = [
                        "[UNAVAILABLE] " + x if "maritaca" in x.lower() else x
                        for x in available_models
                    ]
                if os.environ.get("MARITACA_API_KEY") is None:
                    available_models = [
                        "[UNAVAILABLE] " + x if "maritaca" in x.lower() else x
                        for x in available_models
                    ]
                if os.environ.get("DEEPSEEK_API_KEY") is None:
                    available_models = [
                        "[UNAVAILABLE] " + x if "deepseek" in x.lower() else x
                        for x in available_models
                    ]
                if (
                    os.environ.get("AWS_ACCESS_KEY_ID") is None
                    or os.environ.get("AWS_SECRET_ACCESS_KEY") is None
                ):
                    available_models = [
                        "[UNAVAILABLE] " + x if "bedrock" in x.lower() else x
                        for x in available_models
                    ]

                box_llm_model = gr.Dropdown(
                    available_models,
                    label="Select LLM",
                    allow_custom_value=False,
                    interactive=True,
                )
                chk_native_tools = gr.Checkbox(
                    label="Use LLM native tool calling",
                    value=True,
                    info="Use LLM native tool calling (uncheck to use universal implementation and when a LLM does not support native tool calling)",
                    interactive=True,
                )
            all_tools = [x.tool_description["name"] for x in LLMTools.get_all_tools()]
            chk_tools = gr.CheckboxGroup(
                all_tools,
                value=all_tools[0:1],
                label="Allowed tools",
                info="Select the external tools that the LLM will have access to",
                interactive=True,
            )

            msg2 = gr.Dropdown(
                examples,
                label="Question",
                info="Select or type a question",
                allow_custom_value=True,
            )

            with gr.Row():
                send_btn = gr.Button("Send")
                clear_btn = gr.ClearButton([msg2])

            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="Assistant",
                        elem_id="chatbot",
                        type="messages",
                        height=600,
                    )
                with gr.Column(scale=1):
                    image_input = gr.Image(label="Input Image")
            clear_btn.add([image_input, chatbot])
            scratchpad = gr.Textbox(label="Scratchpad")
            sys_prompt_txt = gr.Text(label="System prompt prepend", value="")
        raw_history = gr.JSON(label="Raw history", open=False)

        gr.on(
            triggers=[send_btn.click],
            fn=msg_forward_func,
            inputs=[
                msg2,
                image_input,
                chatbot,
                sys_prompt_txt,
                box_llm_model,
                chk_native_tools,
                chk_tools,
            ],
            outputs=[msg2, scratchpad, image_input, chatbot, raw_history],
            concurrency_limit=20,
        )

    demo.queue().launch(show_api=False, share=False, inline=False)


if __name__ == "__main__":
    main()
