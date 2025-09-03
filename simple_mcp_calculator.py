"""
Local MCP Server

Launch this server with `python calc_local.py`
Use this server by placing this JSON into the MCP Servers textbox:
{
	"mcpServers": {
		"calc": {
			"transport": "http",
			"url": "http://127.0.0.1:8000/mcp"
		}
	}
}
"""

# basic import
from fastmcp import FastMCP
from typing import Annotated
from pydantic import Field
import math

# instantiate an MCP server client
mcp = FastMCP("Hello World")

# DEFINE TOOLS

# addition tool
@mcp.tool()
def add(
    a: Annotated[float, "First number"],
    b: float = Field(description="Second number. Must be 1, 4 or 5", enum=[1, 4, 5]),
) -> float:
    return float(a + b)


# subtraction tool
@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract two numbers"""
    return float(a - b)


# multiplication tool
@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return float(a * b)


# division tool
@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a/b"""
    return float(a / b)


# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"


# execute and return the stdio output
if __name__ == "__main__":
    print("Starting...")
    # mcp.run(transport="stdio")
    mcp.run(transport="http", host="127.0.0.1", port=8000)

    """
    # Create a proxy to a remote server
    proxy = FastMCP.as_proxy(
        "http://127.0.0.1:8000/mcp",
        name="Calculator Proxy"
    )
    """
