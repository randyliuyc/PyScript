from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("TotalLINK")

# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
  """Get a personalized greeting"""
  return f"Hello, {name}!"
