""" Handles the connection with MCP servers

## TODO:
[ ] Filter allowed tools
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


class MCPTool:
    def __init__(self, name, description, call_fn):
        self.name = name
        self.tool_description = description
        self.tool_summary = description
        self.call_fn = call_fn

    def __call__(self, args):
        return asyncio.run(self.call_fn(args))


class MCPConnector:
    @classmethod
    async def create_from_cfg(cls, config):
        client = Client(config)
        async with client:
            # Basic server interaction
            await client.ping()

            # List available operations
            tools = await client.list_tools()
            resources = await client.list_resources()
            prompts = await client.list_prompts()

        mcpc = cls(client, tools, resources, prompts)
        return mcpc

    def __init__(self, client, mcp_tools, resources, prompts):
        """Constructor.
        Test connections, list available tools and instantiates their call process
        """
        self.client = client
        self.mcp_tools = mcp_tools
        self.resources = resources
        self.prompts = prompts
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
        return str(result.data)

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
    mcpc = asyncio.run(MCPConnector.create_from_cfg(config))
    args = {"a": 5, "b": 4}
    tool_ans = mcpc.tools[0](args)
    print(mcpc.tools[0].name, tool_ans)


if __name__ == "__main__":
    # python -m gat_llm.connector_mcp
    main()
