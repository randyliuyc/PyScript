from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("TotalLINK")

@mcp.prompt()
def inventory_analysis_prompt(user_question: str) -> str:
  """
  针对库存分析类问题，自动规划调用 get_inv_data、get_inv_detail、get_item_trx 的顺序和参数依赖。
  """
  return (
    "你是企业信息系统的智能助手，擅长库存数据分析。"
    "当用户提出如 分析某地点各仓库、各产品近几年变动情况 这类问题时，"
    "请自动按如下业务链路分步调用 MCP 工具：\n"
    "1. 先用 get_inv_data 查询地点的各仓库库存汇总信息。\n"
    "2. 分析返回结果，获取所有仓库编码。\n"
    "3. 对每个仓库，调用 get_inv_detail 获取产品明细。\n"
    "4. 对每个产品，结合地点和产品编码及时间范围，调用 get_item_trx 获取交易记录。\n"
    "每一步都要自动提取上一步的参数作为下一步的输入，最终汇总分析结果，给用户详细解答。"
    "如果用户没有给出地点、时间等参数，请主动询问。"
    f"\n用户问题：{user_question}"
  )

@mcp.prompt()
def greet_user(name: str, style: str = "friendly") -> str:
  """Generate a greeting prompt"""
  styles = {
    "friendly": "Please write a warm, friendly greeting",
    "formal": "Please write a formal, professional greeting",
    "casual": "Please write a casual, relaxed greeting",
  }

  return f"{styles.get(style, styles['friendly'])} for someone named {name}."

@mcp.prompt()
def friendly_prompt(user_question: str) -> str:
  """
  生成一个友好、积极、鼓励用户提问的提示词
  如果用户没有自我介绍，你先要问一下用户的名字，后续回答问题时用用户的名字来称呼用户，并且使用MCP Server进行功能调用时，对于需要linkuser参数的功能，要使用用户名
  Args:
    user_question: 用户输入的问题
  Returns:
    适合AI模型的友好提示词
  """
  if not hasattr(mcp.context, 'user_name'):
    return (
      "很高兴见到你！我是你的助手。请问怎么称呼你呢？"
      "这样我可以更好地为你服务。"
    )
  return (
    f"{mcp.context.user_name}，请用温暖、耐心、积极的语气，详细解答以下用户问题，"
    "如有建议请主动补充，避免生硬和冷漠：\n"
    f"用户提问：{user_question}"
  )
