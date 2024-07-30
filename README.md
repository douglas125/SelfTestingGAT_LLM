# Self Testing GATs (Generation Augmented by Tools)

This project focuses on designing and self-testing GAT LLMs (Language Learning Models) that can effectively use a variety of tools to accomplish tasks.

Demonstration (will take you to YouTube):



[![GAT in action](https://img.youtube.com/vi/zVODAIgDbc4/0.jpg)](https://www.youtube.com/watch?v=zVODAIgDbc4)



Coming soon: Paper pre-print

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

## Using this Code

To use this code and run the implemented tools, follow these steps:

1. Clone this repository

2. Set up the environment:
   - If using conda, create the environment:
     ```
     conda create --name llm_gat_env -c conda-forge boto3 gradio jupyterlab jupyterlab-lsp python-lsp-server openai anthropic tqdm pre-commit sympy pydot ffmpeg qrcode matplotlib ipywidgets
     ```
   - Alternatively, install the requirements directly from `requirements.txt`
   - Activate the environment with `conda activate llm_gat_env`

3. Set up your API keys:
   - For Linux:
     ```
     export AWS_ACCESS_KEY_ID=your_aws_access_key
     export AWS_SECRET_ACCESS_KEY=your_aws_secret_key
     export ANTHROPIC_API_KEY=your_anthropic_key
     export OPENAI_API_KEY=your_openai_key
     ```
   - For Windows:
     ```
     set AWS_ACCESS_KEY_ID=your_aws_access_key
     set AWS_SECRET_ACCESS_KEY=your_aws_secret_key
     set ANTHROPIC_API_KEY=your_anthropic_key
     set OPENAI_API_KEY=your_openai_key
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
