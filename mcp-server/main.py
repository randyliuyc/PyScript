# server.py
from mcp.server.fastmcp import FastMCP
import asyncio

# 导入工具函数 - 数据分析
from tools_data_ana import register_data_ana_tools
# 导入工具函数 - 设备管理
from tools_dev import register_dev_tools

# Create an MCP server
mcp = FastMCP("TotalLINK")

# 注册所有数据分析工具函数
register_data_ana_tools(mcp)
# 注册所有设备管理工具函数
register_dev_tools(mcp)

async def test_get_dev_list():
    result = await mcp.call_tool("get_dev_list", {"username": "DINA", "dept": "AC", "devdes": ""})
    print(result)

if __name__ == "__main__":
  # 测试工具函数
  asyncio.run(test_get_dev_list())

  print("Starting server...")
  mcp.settings.host='0.0.0.0'
  mcp.settings.port = 7077

  try:
    mcp.run(transport="sse")
  except KeyboardInterrupt:
    print("Received KeyboardInterrupt — shutting down gracefully.")
  except Exception as e:
    # 记录并退出
    print("Server stopped with error:", e)
  finally:
    print("Server process exiting.")
