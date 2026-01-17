# TODO_AI.md - Repository Improvement Checklist

This document contains a prioritized list of improvements and feature suggestions for the SelfTestingGAT_LLM repository, ordered by impact for LLM development with tools.

## High-Impact Improvements for LLM Tool Development

### Security Hardening (CRITICAL)

These vulnerabilities could allow LLM-generated code to compromise the host system:

- [ ] **Replace `exec()` with sandboxed execution** in code execution tools
  - Files: `solve_python_code.py`, `solve_numeric.py`, `make_custom_plot.py`, `plot_with_graphviz.py`, `select_video_frames.py`
  - Options: RestrictedPython, subprocess with resource limits, or container isolation
  - Impact: Prevents arbitrary code execution from LLM outputs

- [ ] **Remove `shell=True` from subprocess calls**
  - Files: `run_with_python.py`, `use_ffmpeg.py`
  - Use list arguments instead of shell string parsing
  - Impact: Prevents command injection attacks

- [ ] **Add path traversal protection to file tools**
  - Files: `read_local_file.py`, `write_local_file.py`, `read_file_names_in_local_folder.py`
  - Implement whitelist of allowed directories
  - Validate and sanitize all file paths from LLM

- [ ] **Add XXE protection to XML parsing**
  - File: `gat_llm/tools/base.py`
  - Use `defusedxml` library instead of standard XML parsers

- [ ] **Validate LLM-generated parameters before tool execution**
  - Add schema validation for all tool inputs
  - Sanitize strings that will be used in file operations, SQL, or commands

### Tool Interface Consistency

- [ ] **Create enforced base class for all tools**
  - Define abstract methods: `__init__`, `__call__`, `validate_input`
  - Enforce `tool_description` schema at definition time
  - Standardize return types (string vs generator)

- [ ] **Standardize tool return formats**
  - Currently: some return strings, others yield generators
  - Decide: always return string, always yield, or explicit interface for each
  - Document expected output format in tool_description

- [ ] **Unify XML tag naming conventions**
  - Current: `<path_to_file>`, `<files>`, `<result>` used inconsistently
  - Define standard tags for common data types (file paths, code, results)

- [ ] **Remove global state in tools**
  - Files: `solve_numeric.py`, `solve_python_code.py` use global `ans` variable
  - Replace with instance variables to support concurrent execution

### Tool Execution Framework

- [ ] **Add execution timeouts to all tools**
  - Currently: no timeout on code execution, API calls may hang
  - Implement configurable per-tool timeouts
  - Add graceful timeout handling with meaningful error messages

- [ ] **Implement tool result validation**
  - Define output schemas for tools
  - Validate tool outputs before returning to LLM
  - Catch malformed results early

- [ ] **Add tool execution recording/replay**
  - Record tool inputs, outputs, and timing for debugging
  - Enable replay of tool chains for testing
  - Useful for debugging complex multi-tool interactions

- [ ] **Support tool dependencies and ordering**
  - Allow specifying "tool X requires tool Y to run first"
  - Implement automatic dependency resolution
  - Prevent invalid tool sequences

### LLM Provider Improvements

- [ ] **Consolidate LLM provider code**
  - Currently: 4 separate Bedrock implementations with duplicated code
  - Create shared base class for response parsing, retry logic, error handling
  - Estimate: 30-40% code reduction possible

- [ ] **Standardize response format across providers**
  - Each provider returns slightly different structures
  - Create unified response object with: content, tool_calls, usage, metadata

- [ ] **Add provider-specific rate limit handling**
  - Detect rate limit errors and back off appropriately
  - Implement request queuing for high-throughput scenarios
  - Track and report token usage per provider

- [ ] **Validate API keys on initialization**
  - Currently: API errors only occur when first call is made
  - Check key existence and format at startup
  - Provide clear error messages for missing/invalid keys

---

## Medium-Impact Improvements

### Testing Enhancements

- [ ] **Add security-focused tests**
  - Test code injection attempts in exec() tools
  - Test path traversal attempts in file tools
  - Test SQL injection in query_database tool

- [ ] **Add integration tests**
  - Test tool chains (multiple tools in sequence)
  - Test LLM + tool integration with mock LLM
  - Test error recovery scenarios

- [ ] **Create mock LLM for testing**
  - Deterministic responses for unit tests
  - Configurable error injection
  - Tool call simulation without API costs

- [ ] **Add concurrent execution tests**
  - Test multiple simultaneous tool executions
  - Verify no state corruption with global variables
  - Test rate limiting behavior

- [ ] **Increase test coverage for edge cases**
  - Empty inputs, very large inputs, special characters
  - Network failures, timeout scenarios
  - Malformed API responses

### Code Quality

- [ ] **Add comprehensive type hints**
  - Currently: only ~12 functions have return type annotations
  - Add type hints to all public methods and classes
  - Consider using `mypy` for static type checking

- [ ] **Replace bare exception handling**
  - Files with `except Exception:` or `except:` should catch specific exceptions
  - Log original exception before re-raising or returning error
  - Especially in: `connector_mcp.py`, tool implementations

- [ ] **Implement proper logging**
  - Replace `print()` statements with Python logging
  - Add log levels (DEBUG, INFO, WARNING, ERROR)
  - Include request IDs for tracing multi-step operations

- [ ] **Add docstrings to all public methods**
  - Currently: sparse documentation in code
  - Include: description, parameters, return type, exceptions, examples
  - Generate API documentation from docstrings

### Architecture Refinements

- [ ] **Use registry pattern for tool discovery**
  - Currently: tools hardcoded in `base.py` get_all_tools()
  - Implement dynamic discovery (decorators or entry points)
  - Allow external tool registration without modifying core code

- [ ] **Reduce coupling between LLMInterface and LLMTools**
  - Currently: tightly coupled, hard to use one without the other
  - Define clear interfaces between components
  - Enable using LLMInterface with custom tool implementations

- [ ] **Add circuit breaker pattern for LLM providers**
  - Currently: retries with simple exponential backoff
  - Implement circuit breaker to fail fast when provider is down
  - Add health check endpoints where available

---

## Lower-Impact / Future Features

### Observability

- [ ] **Add metrics collection**
  - Token usage per request/session
  - Tool execution times and success rates
  - LLM latency and error rates
  - Export to Prometheus/StatsD compatible format

- [ ] **Add distributed tracing**
  - Trace requests across LLM calls and tool executions
  - Support OpenTelemetry for integration with tracing systems
  - Correlate logs with traces

### Advanced Tool Features

- [ ] **Support composite tools**
  - Define tools that orchestrate multiple sub-tools
  - Useful for common patterns (e.g., "research" = search + read + summarize)

- [ ] **Add tool rollback/undo capability**
  - Track side effects of tool execution
  - Enable reverting changes (especially for file operations)
  - Implement transaction-like semantics for multi-tool operations

- [ ] **Support multi-modal tool results**
  - Currently: tools return strings with file path references
  - Allow returning structured data (images, tables, charts) directly
  - Define standard formats for rich results

### Performance Optimizations

- [ ] **Optimize streaming implementation**
  - Replace string concatenation in loops with StringIO
  - Use deque for bounded history instead of list slicing

- [ ] **Add cleanup for history_log memory**
  - Currently: `LLMInterface.history_log` grows unbounded
  - Implement max size limit or TTL-based cleanup
  - Consider weak references for temporary data

- [ ] **Cache tool descriptions**
  - Currently: tool descriptions parsed every request
  - Cache after first access, invalidate on tool changes

- [ ] **Add jitter to retry backoff**
  - Current 1.2x multiplier can cause thundering herd
  - Add random jitter to spread out retries

### Developer Experience

- [ ] **Add cost estimation utility**
  - Estimate API costs before execution
  - Track actual costs per session
  - Set budget limits with warnings

- [ ] **Create tool development CLI**
  - Generate tool boilerplate from template
  - Validate tool_description schema
  - Test tool in isolation

- [ ] **Add configuration file support**
  - Currently: all config via environment variables
  - Support YAML/JSON config files
  - Enable per-environment configurations

---

## Notes

- Security items should be addressed before deploying in any environment where untrusted LLM outputs might be executed
- Tool interface consistency improvements will make adding new tools easier and reduce bugs
- Testing improvements provide the foundation for safe refactoring of other items
- Performance optimizations are lower priority unless handling high request volumes
