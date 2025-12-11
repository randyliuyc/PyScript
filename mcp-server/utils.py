# utils.py
from mcp.server.fastmcp import FastMCP
import datetime
from typing import Dict, Any, List

# 定义基础 URL 常量，对应 TotalLINK IIS 基本服务地址
BASE_URL = "http://124.71.144.80:8088"

import httpx
from loguru import logger

# 创建全局 HTTP 客户端（连接池，可复用）
client = httpx.AsyncClient(
  timeout = httpx.Timeout(
    connect = 5.0,
    read = 30.0,
    write = 30.0,
    pool = 30.0
  ),
    headers = {
        "User-Agent": "MCP-Model-Client/1.0",
        "Accept": "application/json",
        "X-Token": ""
    }
)

# 初始化日志记录器
logger.add("server.log")

def register_ai_tools(mcp):
  @mcp.tool()
  async def get_linkdata(
    username: str,
    dmCode: str,
    dmNum: int,
    para: List[str]
  ) -> Dict[str, Any]:
    """
    获取车间的设备列表

    Args:
      username: 用户名，必填，用于权限验证。

    Returns:
      JSON 格式的模型结果，其中 isSucess 为 True 表示成功，否则为 False, result 包含返回的实际数据
    """
    return await get_ai_result(dmCode, dmNum, para, username)

# 定义异步函数，获取指定模型的结果
async def get_ai_result(
  code: str,          # 模型代码（必填）
  num: int,           # 模型编号（必填）
  para: List[str],    # 字符串数组参数（必填）
  username: str       # TotalLINK用户名
) -> Dict[str, Any]:
  """
  获取指定模型的结果
  Args:
    code: 模型代码（如 "TMES10"、"TMES20"）
    num: 模型编号（如 10, 20, 30）
    para: 数组列表（如 ["p1","p2","p3"]）
  Returns:
    JSON 格式的模型结果
  """
  try:
    # 1. 构造请求参数
    # 判断 username 是否为32位字符（由字母和数字组成）
    if len(username) == 32 and username.isalnum():
      linktoken = username
    else:
      linktoken = username + " " + calc_value()
      
    payload = {
      "loginID": linktoken,
      "par": {
        "dmCode": code,
        "dmNum": num,
        "Para": para
      }
    }

    logger.info(payload)

    # 2. 发送POST请求
    response = await client.post(
      f"{BASE_URL}/api/DataModel/linkDMAIResult",
      json = payload
    )

    # 3. 检查响应状态
    response.raise_for_status()
    return response.json()

  except httpx.RequestError as e:
    return {"status": "error", "error": str(e)}

# 定义异步函数，执行模型的附加模型动作
async def ai_action(
  code: str,                # 模型代码（必填）
  num: int,                 # 模型编号（必填）
  action: int,              # 模型动作（必填）
  para: List[str],          # 字符串数组参数（必填）
  rowdata: Dict[str, str],  # 字符串数组参数（必填）
  username: str             # TotalLINK用户名
) -> Dict[str, Any]:
  """
  执行AI模型动作操作
  Args:
    code: 模型代码（如 "TMES10"、"TMES20"）
    num: 模型编号（如 10, 20, 30）
    action: 模型动作（如 501, 502）
    para: 数组列表（如 ["p1","p2","p3"]）
    rowdata: 行数据字典，包含具体的操作数据
      - MNTTYP: 维保类型代码
      - MNTDES: 维保描述信息
    username: 用户名，默认为 "DINA"
  """
  try:
    # 1. 构造请求参数
    linktoken = username + " " + calc_value()

    payload = {
      "loginID": linktoken,
      "par": {
        "dm": {
          "dmCode": code,
          "dmNum": num,
          "Para": para,
        },
        "contextMenuNo": action,
        "rowData": rowdata
      }
    }

    logger.info(payload)

    # 2. 发送POST请求
    response = await client.post(
      f"{BASE_URL}/api/DataModel/linkDMAIAction",
      json = payload
    )

    # 3. 检查响应状态
    response.raise_for_status()
    return response.json()

  except httpx.RequestError as e:
    return {"status": "error", "error": str(e)}

# 定义异步函数，执行模型的行数据提交处理
async def ai_row_submit(
  code: str,                # 模型代码（必填）
  num: int,                 # 模型编号（必填）
  scriptType: int,          # 模型动作（必填）
  para: List[str],          # 字符串数组参数（必填）
  rowdata: Dict[str, str],  # 字符串数组参数（必填）
  username: str             # TotalLINK用户名
) -> Dict[str, Any]:
  """
  执行AI模型动作操作
  Args:
    code: 模型代码（如 "TMES10"、"TMES20"）
    num: 模型编号（如 10, 20, 30）
    scriptType: 模型脚本类型（如 1-添加, 2-编辑, 3-删除, 4-数据处理）
    para: 数组列表（如 ["p1","p2","p3"]）
    rowdata: 行数据字典，包含具体的操作数据
      - MNTTYP: 维保类型代码
      - MNTDES: 维保描述信息
    username: 用户名，默认为 "DINA"
  """
  try:
    # 1. 构造请求参数
    linktoken = username + " " + calc_value()

    payload = {
      "loginID": linktoken,
      "par": {
        "dm": {
          "dmCode": code,
          "dmNum": num,
          "Para": para,
        },
        "scriptType": scriptType,
        "rowData": rowdata
      }
    }

    logger.info(payload)

    # 2. 发送POST请求
    response = await client.post(
      f"{BASE_URL}/api/DataModel/linkDMAIRowSubmit",
      json = payload
    )

    # 3. 检查响应状态
    response.raise_for_status()
    return response.json()

  except httpx.RequestError as e:
    return {"status": "error", "error": str(e)}

# 获取 TotalLINK 的AI调用令牌
def calc_value():
  now = datetime.datetime.now()
  I = int(now.strftime("%S%M%H%y%m%d"))
  result = (I - 12251) * 12253 - 31321
  return str(result)
