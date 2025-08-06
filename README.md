# Self Testing GATs (Generation Augmented by Tools)

This project focuses on designing and self-testing GAT LLMs (Language Learning Models) that can effectively use a variety of tools to accomplish tasks.

Demonstration (will take you to YouTube):

[![GAT in action](https://img.youtube.com/vi/U1oxouaOf5g/0.jpg)](https://www.youtube.com/watch?v=U1oxouaOf5g)

**Paper pre-print**: in the folder `paper`

## Table of Contents
1. [Project Overview](#project-overview)
2. [Using this Code](#using-this-code)
3. [Inspecting the Tools and LLMs](#inspecting-the-tools-and-llms)
4. [Changing the Code](#changing-the-code)
5. [Self-assessment](#self-assessment)

## Project Overview

This project implements a flexible framework for:
- Integrating various tools with LLMs
- Generating test cases to evaluate LLM performance in tool selection and usage
- Performing self-tests on different LLM models
- Analyzing the results of these tests

The system supports multiple LLM providers (including OpenAI, Anthropic, and AWS Bedrock) and a wide range of tools for tasks such as date calculations, web scraping, plotting, file operations, and more.

### Current benchmarks

With the current prompts, tools, descriptions and native tool configuration use settings, this is the performance of LLMs in GAT tasks.

**Note: this is not a leaderboard or general evaluation of quality. It only refers to this test setting as a simulation of an industrial LLM GAT implementation.**

|                                           |   ('n_invented_tools', 'sum') |   ('accuracy', '%') |   ('score', '%') |   ('USD / 1M tokens', 'Input') |   ('USD / 1M tokens', 'Output') |
|:------------------------------------------|------------------------------:|--------------------:|-----------------:|-------------------------------:|--------------------------------:|
| ('DeepSeekV3 Chat - DeepSeek', False)     |                             1 |                79.4 |             89.6 |                          0.27  |                            1.1  |
| ('Claude 3.5 Sonnet - Anthropic', False)  |                             0 |                78   |             89.5 |                          3     |                           15    |
| ('GPT 4o - OpenAI', True)                 |                             1 |                79.9 |             89.4 |                          5     |                           15    |
| ('GPT 4.1 - OpenAI', True)                |                             1 |                78.6 |             89   |                          2     |                            8    |
| ('GPT 4o mini - OpenAI', True)            |                             3 |                79.9 |             89   |                          0.15  |                            0.6  |
| ('Claude 3.5 Haiku - Anthropic', True)    |                             2 |                76.6 |             89   |                          1     |                            5    |
| ('Amazon Nova Pro 1.0 - Bedrock', True)   |                             1 |                78   |             88.7 |                          0.8   |                            3.2  |
| ('Claude 3.5 Sonnet - Anthropic', True)   |                             0 |                76.6 |             88.7 |                          3     |                           15    |
| ('Claude 3 Haiku - Bedrock', True)        |                             2 |                77.5 |             88.6 |                          0.25  |                            1.25 |
| ('Claude 3.5 Haiku - Anthropic', False)   |                             9 |                73.9 |             87.9 |                          1     |                            5    |
| ('GPT 4o - OpenAI', False)                |                             4 |                76.6 |             87.7 |                          5     |                           15    |
| ('Llama3_1 405b instruct', False)         |                             3 |                75.5 |             87   |                          5.32  |                           16    |
| ('Claude 3.7 Sonnet - Anthropic', True)   |                             2 |                74.7 |             86.9 |                          3     |                           15    |
| ('Mistral Large v1', False)               |                             1 |                74.7 |             86.8 |                          4     |                           12    |
| ('GPT 4o mini - OpenAI', False)           |                             3 |                73.1 |             85.1 |                          0.15  |                            0.6  |
| ('Command RPlus - Bedrock', False)        |                             4 |                72.8 |             83.8 |                          3     |                           15    |
| ('Claude 3 Haiku - Bedrock', False)       |                             3 |                70.6 |             83.3 |                          0.25  |                            1.25 |
| ('Sabia3 - Maritaca', True)               |                             6 |                70.6 |             83.2 |                          0.95  |                            1.9  |
| ('Amazon Nova Lite 1.0 - Bedrock', True)  |                             2 |                66.2 |             80.2 |                          0.06  |                            0.24 |
| ('Llama3_1 70b instruct', False)          |                            11 |                70   |             79.6 |                          2.65  |                            3.5  |
| ('GPT 3.5 - OpenAI', False)               |                             2 |                65.4 |             78.6 |                          0.5   |                            1.5  |
| ('GPT 3.5 - OpenAI', True)                |                            18 |                66.4 |             76.9 |                          0.5   |                            1.5  |
| ('Sabia3 - Maritaca', False)              |                            14 |                61.8 |             75.7 |                          0.95  |                            1.9  |
| ('Mistral Mixtral 8x7B', False)           |                           156 |                50.1 |             67.5 |                          0.45  |                            0.7  |
| ('Amazon Nova Micro 1.0 - Bedrock', True) |                           145 |                52.5 |             66.5 |                          0.035 |                            0.14 |
| ('Command R - Bedrock', False)            |                           117 |                49.7 |             65.4 |                          0.5   |                            1.5  |
| ('Llama3 8b instruct', False)             |                            39 |                22.3 |             38.1 |                          0.3   |                            0.6  |
| ('Llama3 70b instruct', False)            |                            29 |                29.1 |             36.1 |                          2.65  |                            3.5  |
| ('Llama3_1 8b instruct', False)           |                            34 |                23.9 |             33.7 |                          0.3   |                            0.6  |
| ('Grok2Vision - Grok', True)              |                             1 |                25   |             29   |                          2     |                           10    |

## Using this Code

To use this code and run the implemented tools, follow these steps:

### With PIP

1. `pip install gat_llm`
2. (Optional) Install optional dependencies for MarkItDown with `pip install markitdown[all]` (this is used to open .DOCX, .XLSX, etc)
3. (Optional) Install `poppler` (this is used to convert PDF pages to images when PDF pages need OCR or to be handled as images). If using conda, `conda install pdf2image` should handle everything
4. Set up your API keys (depending on what tools and LLM providers you need):
   - For Linux:
     ```
     export AWS_ACCESS_KEY_ID=your_aws_access_key
     export AWS_SECRET_ACCESS_KEY=your_aws_secret_key
     export ANTHROPIC_API_KEY=your_anthropic_key
     export OPENAI_API_KEY=your_openai_key
	 export MARITACA_API_KEY=your_maritaca_key
     ```
   - For Windows:
     ```
     set AWS_ACCESS_KEY_ID=your_aws_access_key
     set AWS_SECRET_ACCESS_KEY=your_aws_secret_key
     set ANTHROPIC_API_KEY=your_anthropic_key
     set OPENAI_API_KEY=your_openai_key
	 set MARITACA_API_KEY=your_maritaca_key
     ```
5. Create a test file `test_gat.py` to check if the tools are being called correctly:
```
# Imports
import boto3
import botocore

import gat_llm.llm_invoker as inv
from gat_llm.tools.base import LLMTools
from gat_llm.prompts.prompt_generator import RAGPromptGenerator

use_native_LLM_tools = True

# pick one depending on which API key you want to use
llm_name = "GPT 4o - OpenAI"
llm_name = 'Claude 3.5 Sonnet - Bedrock'
llm_name = 'Claude 3.5 Sonnet - Anthropic'

config = botocore.client.Config(connect_timeout=9000, read_timeout=9000, region_name="us-west-2")  # us-east-1  us-west-2
bedrock_client = boto3.client(service_name='bedrock-runtime', config=config)

llm = inv.LLM_Provider.get_llm(bedrock_client, llm_name)
query_llm = inv.LLM_Provider.get_llm(bedrock_client, llm_name)

print("Testing LLM invoke")
ans = llm("and at night? Enclose your answer within <my_ans></my_ans> tags. Then explain further.",
          chat_history=[["What color is the sky?", "Blue"]],
          system_prompt="You are a very knowledgeable truck driver. Use a strong truck driver's language and make sure to mention your name is Jack.",
         )
prev = ""
for x in ans:
    cur_ans = x
    print('.', end='')
print('\n')
print(x)

# Test tool use
print("Testing GAT - LLM tool use")
lt = LLMTools(query_llm=query_llm)
tool_descriptions = lt.get_tool_descriptions()
rpg = RAGPromptGenerator(use_native_tools=use_native_LLM_tools)
system_prompt = rpg.prompt.replace('{{TOOLS}}', tool_descriptions)

cur_tools = [x.tool_description for x in lt.tools]

ans = llm(
    "What date will it be 10 days from now? Today is June 4, 2024. Use your tool do_date_math. Before calling any tools, explain your thoughts. Then, make a plot of y=x^2.",
    chat_history=[["I need to do some date math.", "Sure. I will help."]],
    system_prompt="You are a helpful assistant. Prefer to use tools when possible. Never mention tool names in the answer.",
    tools=cur_tools,
    tool_invoker_fn=lt.invoke_tool,
)

prev = ""
for x in ans:
    cur_ans = x
    print('.', end='')
print(cur_ans)
```
4. Run `python test_gat.py`. You should see a response like:
```
Testing LLM invoke
..................................

<my_ans>Black as the inside of my trailer, with little white dots all over it</my_ans>

Hey there, Jack here. Been drivin' rigs for over 20 years now, and let me tell ya, when you're haulin' freight through the night, that sky turns darker than a pot of truck stop coffee. You got them stars scattered all over like chrome bits on a custom Peterbilt, and sometimes that moon hangs up there like a big ol' headlight in the sky.

When you're cruisin' down them highways at 3 AM, with nothin' but your high beams and them stars above, it's one hell of a sight. Makes ya feel pretty damn small in your rig, if ya know what I mean. Course, sometimes you get them city lights polluting the view, but out in the boonies, man, that night sky is somethin' else.

Shoot, reminds me of this one haul I did through Montana - clearest dang night sky you'll ever see. But I better wrap this up, my 30-minute break is almost over, and I got another 400 miles to cover before sunrise.

Testing GAT - LLM tool use

In 10 days from June 4, 2024, it will be June 14, 2024 (Friday). I've also generated a plot showing the quadratic function y = x².
```

### From the repository

1. Clone this repository and `cd` to the repository folder.

2. Set up the environment:
   - If using conda, create the environment:
     ```
     conda env create -f environment.yml
     ```
   - Alternatively, install the requirements directly from `requirements.txt`
   - Activate the environment with `conda activate llm_gat_env`

3. Set up your API keys (depending on what tools and LLM providers you need):
   - For Linux:
     ```
     export AWS_ACCESS_KEY_ID=your_aws_access_key
     export AWS_SECRET_ACCESS_KEY=your_aws_secret_key
     export ANTHROPIC_API_KEY=your_anthropic_key
     export OPENAI_API_KEY=your_openai_key
	 export GROK_API_KEY=your_grok_key
	 export MARITACA_API_KEY=your_maritaca_key
     ```
   - For Windows:
     ```
     set AWS_ACCESS_KEY_ID=your_aws_access_key
     set AWS_SECRET_ACCESS_KEY=your_aws_secret_key
     set ANTHROPIC_API_KEY=your_anthropic_key
     set OPENAI_API_KEY=your_openai_key
	 set GROK_API_KEY=your_grok_key
	 set MARITACA_API_KEY=your_maritaca_key
     ```

4. Open and run `GAT-demo.ipynb` to launch the Gradio demo

5. Access the demo:
   - Click the `localhost` interface
   - To share the demo with a public Gradio link, set `share=True` in the launch command:
     ```python
     demo.queue().launch(show_api=False, share=True, inline=False)
     ```

## Inspecting the Tools and LLMs

The Jupyter Notebook (`GAT-demo.ipynb`) provides a convenient interface for inspecting:
- Direct tool call results
- Prompts used for LLM interactions
- Other relevant information about the system's operation

Refer to the comments in the notebook for detailed explanations of each section.

## Changing the Code

### Implementing a New Tool

To add a new tool to the system:

1. Create a new Python file in the `tools` folder (e.g., `new_tool.py`)
2. Define a new class for your tool (e.g., `ToolNewTool`)
3. Implement the following methods:
   - `__init__`: Initialize the tool, set its name and description
   - `__call__`: Implement the tool's functionality
4. Add the tool description in the `tool_description` attribute, following the format used in other tools
5. In `tools/base.py`, import your new tool and add it to the `get_all_tools` method in the `LLMTools` class

Example structure for a new tool:

```python
class ToolNewTool:
    def __init__(self):
        self.name = "new_tool_name"
        self.tool_description = {
            "name": self.name,
            "description": "Description of what the tool does",
            "input_schema": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "Description of param1"},
                    # Add more parameters as needed
                },
                "required": ["param1"]
            }
        }

    def __call__(self, param1, **kwargs):
        # Implement tool functionality here
        result = # ... your code ...
        return result
```

### Removing Tools

To remove a tool from the system:

1. Delete the tool's Python file from the `tools` folder
2. Remove the tool's import and reference from `tools/base.py`
3. Update any test cases or documentation that reference the removed tool

### Adding LLMs

To add support for a new LLM:

1. Create a new file in the `llm_providers` folder (e.g., `new_llm_provider.py`)
2. Implement a class for the new LLM, following the interface used by existing LLM classes
3. In `llm_invoker.py`, import your new LLM class and add it to the `allowed_llms` list in the `LLM_Provider` class
4. Implement the necessary logic in the `get_llm` method of `LLM_Provider` to instantiate your new LLM

## Self-assessment

The project includes a comprehensive self-assessment system for evaluating LLM performance in tool selection and usage. All test cases self-generated and the test results of each LLM are stored in the folder `self_tests`.

### Self-generating Test Cases

The `SelfTestGenerator` class in `self_tests/self_test_generator.py` is responsible for creating test cases. It supports three strategies for test case generation:

1. `use_all`: Generates test cases for all tools in a single prompt
2. `only_selected`: Generates test cases for each tool individually
3. `selected_with_dummies`: Generates test cases for specific tools while providing all tools as options

To generate test cases:

1. Instantiate a `SelfTestGenerator` with the desired LLM
2. Call the `gen_test_cases` method with the number of test cases and the desired strategy

### Using the Test Cases to Evaluate LLMs

The `SelfTestPerformer` class in `self_tests/self_test_performer.py` executes the generated test cases to evaluate LLM performance.

To run self-tests:

1. Prepare test case files (JSON format) using the `SelfTestGenerator`
2. Instantiate a `SelfTestPerformer` with the LLM you want to test
3. Call the `test_tool_use` method with the test cases

The results are saved in CSV format, allowing for easy analysis and comparison of different LLM models and configurations.

Use the utility functions in `self_tests/self_test_utils.py` to analyze the test results, including functions to detect invented tools, check for correct tool selection, and calculate performance scores.

# Changelog

## v0.1.4

- Added Grok as LLM
- Added caching to Claude Bedrock models (Haiku 3.5 and Sonnet 3.7)

## v0.1.5

- Changed the UI to show thinking / tools
- Fixed a bug in `test_llm_tools.py` when no tools were selected

## v0.1.6

- Add GPT 4.1 LLM
- Add GPT 4.1 image generator
- Add Claude 4 (Anthropic and Bedrock)

## v0.1.7

- Enable multiple images per user message

## v0.1.8

- Include Ollama as a local LLM provider
- Update `read_local_file` tool to read a much wider array of files
- Include Grok4 from xAI

## v0.1.9

- Include smaller qwen3 models

## TBD

- Include qwen3-coder:30b from Ollama
- Include GPT OSS 20b and 120b from Ollama
