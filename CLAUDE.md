# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Self Testing GAT (Generation Augmented by Tools) LLM is a framework for evaluating and benchmarking LLMs in their ability to select and use tools. It supports multiple LLM providers (OpenAI, Anthropic, AWS Bedrock, DeepSeek, Grok, Ollama, vLLM) and 22+ tools for tasks like date math, web scraping, plotting, file operations, image/audio processing, and database queries.

## Commands

### Testing
```bash
pytest tests/                           # Run all tests (30s timeout per test)
pytest tests/test_llm_interface.py      # Run specific test file
pytest tests/test_tool_read_local_file.py -v  # Verbose single test
```

### Code Quality
```bash
black gat_llm/                          # Format code
pre-commit run --all-files              # Run all pre-commit hooks
```

### Installation
```bash
pip install -e .                        # Install from source
pip install -e ".[dev]"                 # With dev dependencies
pip install markitdown[all]             # Optional: DOCX/XLSX support
```

### Running the Demo
```bash
python test_llm_tools.py                # Launch Gradio UI
# Or open notebooks/GAT-demo.ipynb in Jupyter
```

## Architecture

### Core Components

```
User Input (Gradio UI)
    ↓
LLMInterface (gat_llm/llm_interface.py) - orchestrates LLM calls with tool support
    ├→ RAGPromptGenerator (gat_llm/prompts/) - builds system prompts
    ├→ LLM_Provider (gat_llm/llm_invoker.py) - factory routing to providers
    │   └→ Provider classes in gat_llm/llm_providers/
    └→ LLMTools (gat_llm/tools/base.py) - tool registry and invocation
```

### Key Design Patterns

1. **Provider Factory**: `LLM_Provider.get_llm(bedrock_client, llm_name)` routes to correct provider based on model name
2. **Tool Registry**: `LLMTools.get_all_tools()` returns all tool instances; `invoke_tool()` executes them
3. **Streaming**: All LLM responses are generators yielding incremental output
4. **Dual Tool Modes**: Native tool calling (OpenAI/Anthropic format) or prompt-based (for models without native support)

### Key Files

- `gat_llm/llm_invoker.py` - LLM factory with 80+ model registry
- `gat_llm/llm_interface.py` - Core interface handling streaming and tool orchestration
- `gat_llm/tools/base.py` - Tool registry, import new tools here
- `gat_llm/llm_providers/base_service.py` - Base class for all LLM providers
- `gat_llm/prompts/prompt_generator.py` - RAGPromptGenerator class
- `test_llm_tools.py` - Main Gradio web UI

### Adding a New Tool

1. Create `gat_llm/tools/new_tool.py` with class implementing `__init__`, `__call__`, and `tool_description`
2. Import and add to `get_all_tools()` in `gat_llm/tools/base.py`

```python
class ToolNewTool:
    def __init__(self, require_llm_postprocessing=True):
        self.name = "new_tool_name"
        self.require_llm_postprocessing = require_llm_postprocessing
        self.tool_description = {
            "name": self.name,
            "description": "Description of what the tool does",
            "input_schema": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "Description"},
                },
                "required": ["param1"]
            }
        }

    def __call__(self, param1, **kwargs):
        return result
```

### Tool LLM Postprocessing

Each tool has a `require_llm_postprocessing` property that controls whether the LLM should be called again after the tool executes:

- **`True` (default)**: After tool execution, the LLM is called again to analyze and present the results to the user. Use this for tools that return raw data needing interpretation.
- **`False`**: Tool output is returned directly to the UI without additional LLM processing. Use this for tools that generate self-contained outputs (images, audio) where the UI renders the result via `<path_to_image>` or `<path_to_audio>` tags.

Tools with `require_llm_postprocessing=False`:
- `make_qr_code`, `make_custom_plot`, `plot_with_graphviz` - generate images
- `text_to_image`, `edit_image` - image generation
- `text_to_speech` - generates audio

Tools with `require_llm_postprocessing=True`:
- All others, including tools returning `<path_to_file>` that need LLM explanation

### Adding a New LLM Provider

1. Create `gat_llm/llm_providers/new_provider.py` extending `LLM_Service`
2. Add model names to `allowed_llms` list in `llm_invoker.py`
3. Add instantiation logic to `get_llm()` method

## Environment Variables

Set API keys based on which providers you use:
- `ANTHROPIC_API_KEY` - Anthropic Claude
- `OPENAI_API_KEY` - OpenAI GPT
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` - AWS Bedrock
- `GROK_API_KEY` - xAI Grok
- `MARITACA_API_KEY` - Maritaca
- `DEEPSEEK_API_KEY` - DeepSeek

## Self-Testing Framework

Located in `self_tests/`. Three test generation strategies:
- `use_all` - All tools in one prompt
- `only_selected` - Each tool individually
- `selected_with_dummies` - Specific tools with dummy options

Use `SelfTestGenerator` to create test cases and `SelfTestPerformer` to evaluate models. Results stored as CSV files.
