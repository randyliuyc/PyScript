# server.py
from mcp.server.fastmcp import FastMCP

# 导入基础工具函数 - 获取通用模型数据
from utils import register_ai_tools
# 导入工具函数 - 数据分析

# Create an MCP server
mcp = FastMCP("TotalLINK")

# 注册所有通用模型工具函数
register_ai_tools(mcp)

if __name__ == "__main__":
  # 测试工具函数
  # asyncio.run(test_get_dev_list())

  print("Starting server...")
  mcp.settings.host = '0.0.0.0'
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
