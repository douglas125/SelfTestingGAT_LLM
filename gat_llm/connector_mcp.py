""" Handles the connection with MCP servers

More details about the config parameters:
https://gofastmcp.com/clients/transports#mcp-json-configuration-transport

## TODO:
[ ] Call resources
[ ] Call prompts

The expected format is as follows:

config = {
    "mcpServers": {
        "server_name": {
            # Remote HTTP/SSE server
            "transport": "http",  # or "sse"
            "url": "https://api.example.com/mcp",
            "headers": {"Authorization": "Bearer token"},
            "auth": "oauth"  # or bearer token string
        },
        "local_server": {
            # Local stdio server
            "transport": "stdio",
            "command": "python",
            "args": ["./server.py", "--verbose"],
            "env": {"DEBUG": "true"},
            "cwd": "/path/to/server",
        }
    }
}
"""


import asyncio
from fastmcp import Client
from fastmcp.prompts.prompt import TextContent


class MCPTool:
    def __init__(self, name, description, call_fn):
        self.name = name
        self.tool_description = description
        self.tool_summary = description
        self.call_fn = call_fn

    def __call__(self, **args):
        return asyncio.run(self.call_fn(args))


class MCPConnector:
    @classmethod
    async def create_from_cfg(cls, config):
        client = Client(config)
        async with client:
            # Basic server interaction
            await client.ping()

            # List available operations
            try:
                tools = await client.list_tools()
            except:
                tools = None

            try:
                resources = await client.list_resources()
            except:
                resources = None

            try:
                prompts = await client.list_prompts()
            except:
                prompts = None

        mcpc = cls(config, client, tools, resources, prompts)
        return mcpc

    def __init__(
        self,
        config,
        client,
        mcp_tools,
        resources,
        prompts,
        return_only_text_content=True,
    ):
        """Constructor.
        Test connections, list available tools and instantiates their call process

        If return_only_text_content = True, only returns the textual part of the tool calls
        """
        self.config = config
        self.client = client
        self.mcp_tools = mcp_tools
        self.resources = resources
        self.prompts = prompts
        self.return_only_text_content = return_only_text_content
        self.tools = []
        for mcp_tool in self.mcp_tools:
            cur_tool_desc = self._convert_mcp_tool_to_std(mcp_tool)

            self.tools.append(
                MCPTool(
                    cur_tool_desc["name"],
                    cur_tool_desc,
                    self.call_tool(cur_tool_desc["name"]),
                )
            )

    def call_tool(self, name):
        async def call_tool_fn(args):
            ans = await self._call_tool_async(name, args)
            return ans

        return call_tool_fn

    async def _call_tool_async(self, name, args):
        async with self.client:
            # Simple tool call
            result = await self.client.call_tool(name, args)
        if self.return_only_text_content:
            ans = [str(x.text) for x in result.content if isinstance(x, TextContent)]
            if len(ans) == 1:
                return str(ans[0])
            elif len(ans) == 0:
                return ""
            else:
                return str(ans)
        else:
            return str(result.content)

    def _convert_mcp_tool_to_std(self, mcp_tool_description):
        """Converts a MCP tool description to Claude's standard format"""
        tool_desc = {
            "name": mcp_tool_description.name,
            "description": mcp_tool_description.description,
            "input_schema": mcp_tool_description.inputSchema,
        }
        if tool_desc["description"] is None:
            # handle more gracefully
            tool_desc["description"] = "(no description provided)"
        tool_desc["input_schema"]["type"] = "object"
        tool_desc["input_schema"]["required"] = tool_desc["input_schema"].get(
            "required", []
        )
        return tool_desc


def main():
    config = {
        "mcpServers": {
            "calc2": {
                # Remote HTTP/SSE server
                "transport": "http",  # or "sse"
                "url": "http://127.0.0.1:8000/mcp",
                # "headers": {"Authorization": "Bearer token"},
                # "auth": "oauth"  # or bearer token string
            },
            # "huggingface": {
            #    "transport": "http",
            #    "url": "https://huggingface.co/mcp",
            # }
        }
    }
    config = {
        "mcpServers": {
            "huggingface": {
                "url": "https://huggingface.co/mcp",
                # "headers": {"Authorization": "Bearer TOKEN"}
            }
        }
    }
    mcpc = asyncio.run(MCPConnector.create_from_cfg(config))
    for t in mcpc.tools:
        print(t.name, t.tool_description)
        if t.name == "hf_doc_search":
            break
    # print(mcpc.tools[3].name, mcpc.tools[3].tool_description)
    # args = {"a": 5, "b": 4}
    args = {"query": "text to speech"}
    tool_ans = t(**args)
    print(t.name, tool_ans)


if __name__ == "__main__":
    # python -m gat_llm.connector_mcp
    main()
