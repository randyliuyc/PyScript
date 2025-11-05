"""
FastMCP quickstart example.

cd to the `examples/snippets/clients` directory and run:
    uv run server fastmcp_quickstart stdio
"""

from mcp.server.fastmcp import FastMCP
from typing import Dict, Any, List

import requests
import datetime

# Create an MCP server
mcp = FastMCP("TotalLINK")

# 销售数据
@mcp.tool()
def get_sales_data(
    startdate: str, 
    enddate: str, 
    company: str = "",  
    cus: str = "", 
    item: str = "",
    username: str = "DINA"
) -> Dict[str, Any]:
    """
    获取销售数据，提问包含具体的公司、客户、产品信息时，明确使用对应的参数值，否则保持参数值为空白
    Args:
        startdate: 起始日期，格式为 "YYYY-MM-DD"，必填
        enddate: 结束日期，格式为 "YYYY-MM-DD"，必填
        company: 公司编码或公司名称，如果未明确参数值，则使用空字符串
        cus: 客户编码或客户名称，如果未明确参数值，则使用空字符串
        item: 产品编码或产品名称，如果未明确参数值，则使用空字符串
        username: 用户名，默认为 "DINA"
    Returns:
        JSON 格式的模型结果，其中 isSucess 为 True 表示成功，否则为 False, data 包含返回的实际数据
    """
    # 1. 构造请求参数
    company = company or ""
    cus = cus or ""
    item = item or ""

    payload = {
        "dmCode": "LINKAIMCP10X.SALES", # 模型代码
        "dmNum": 10,                    # 模型编号
        "para": [startdate, enddate, company, cus, item]    # 字符串数组参数
    }

    result = get_ai_result(payload["dmCode"], payload["dmNum"], payload["para"], username)
    return result

# 采购数据
@mcp.tool()
def get_pur_data(
    startdate: str, 
    enddate: str, 
    company: str, 
    sup: str, 
    item: str,
    username: str = "DINA"
) -> Dict[str, Any]:
    """
    获取采购数据，提问包含具体的公司、供应商、产品信息时，明确使用对应的参数值，否则保持参数值为空白
    Args:
        startdate: 起始日期，格式为 "YYYY-MM-DD"
        enddate: 结束日期，格式为 "YYYY-MM-DD"
        company: 公司编码或公司名称，如果未明确参数值，则使用空字符串
        sup: 供应商编码或供应商名称，如果未明确参数值，则使用空字符串
        item: 产品编码或产品名称，如果未明确参数值，则使用空字符串
    Returns:
        JSON 格式的模型结果，其中 isSucess 为 True 表示成功，否则为 False, data 包含返回的实际数据
    """
    # 1. 构造请求参数
    payload = {
        "dmCode": "LINKAIMCP10X.PUR", # 模型代码
        "dmNum": 10,           # 模型编号
        "para": [startdate, enddate, company, sup, item]    # 字符串数组参数
    }

    result = get_ai_result(payload["dmCode"], payload["dmNum"], payload["para"], username)
    return result

# 库存数据
@mcp.tool()
def get_inv_data(
    site: str, 
    loc: str, 
    item: str,
    username: str = "DINA"
) -> Dict[str, Any]:
    """
    获取库存数据，提问包含具体的地点、仓库、产品信息时，明确使用对应的参数值，否则保持参数值为空白
    Args:
        site: 地点编码或地点名称，如果未明确参数值，则使用空字符串
        loc: 仓库，如果未明确参数值，则使用空字符串
        item: 产品编码、产品类别或产品名称，如果未明确参数值，则使用空字符串
    Returns:
        JSON 格式的模型结果，其中 isSucess 为 True 表示成功，否则为 False, data 包含返回的实际数据
    """
    # 1. 构造请求参数
    payload = {
        "dmCode": "LINKAIMCP10X.INV",           # 模型代码
        "dmNum": 10,                            # 模型编号
        "para": [site, loc, item]     # 字符串数组参数
    }

    result = get_ai_result(payload["dmCode"], payload["dmNum"], payload["para"], username)
    return result

# 库存明细数据
@mcp.tool()
def get_inv_detail(
    site: str, 
    warehouse: str,
    username: str = "DINA"
) -> Dict[str, Any]:
    """
    按地点、仓库获取产品库存明细数据
    如果没有明确的warehouse参数，则从get_inv_data的返回结果中查找warehouse参数的具体值
    Args:
        site: 地点编码，必填
        warehouse: 仓库编码，必填
    Returns:
        JSON 格式的模型结果，其中 isSucess 为 True 表示成功，否则为 False, data 包含返回的实际数据
    """
    # 1. 构造请求参数
    payload = {
        "dmCode": "LINKAIMCP10X.INV",           # 模型代码
        "dmNum": 20,                            # 模型编号
        "para": [site, warehouse]               # 字符串数组参数
    }

    result = get_ai_result(payload["dmCode"], payload["dmNum"], payload["para"], username)
    return result

# 库存明细变动记录
@mcp.tool()
def get_item_trx(
    site: str, 
    item: str,
    startdate: str,
    enddate: str,
    username: str = "DINA"
) -> Dict[str, Any]:
    """
    按地点、产品编码和时间范围获取产品库存交易变动记录
    如果没有明确的产品编码参数，则从get_inv_detail的返回结果中查找产品编码参数的具体值
    Args:
        site: 地点，必填
        item: 产品编码，必填
        startdate: 起始日期，格式为 "YYYY-MM-DD"，必填
        enddate: 结束日期，格式为 "YYYY-MM-DD"，必填
    Returns:
        JSON 格式的模型结果，其中 isSucess 为 True 表示成功，否则为 False, data 包含返回的实际数据
    """
    # 1. 构造请求参数
    payload = {
        "dmCode": "LINKAIMCP10X.INV",                # 模型代码
        "dmNum": 30,                                 # 模型编号
        "para": [site, item, startdate, enddate]     # 字符串数组参数
    }

    result = get_ai_result(payload["dmCode"], payload["dmNum"], payload["para"], username)
    return result

# 车间的设备列表
@mcp.tool()
def get_dev_list(
    dept: str, 
    devdes: str,
    username: str = "DINA"
) -> Dict[str, Any]:
    """
    获取车间的设备列表
    Args:
        dept: 车间编码，必填
        devdes: 设备名称，如果未明确参数值，则使用空字符串
        username: 用户名，默认为 "DINA"
    Returns:
        JSON 格式的模型结果，其中 isSucess 为 True 表示成功，否则为 False, data 包含返回的实际数据
    """
    # 1. 构造请求参数
    devdes = devdes or ""

    payload = {
        "dmCode": "LINKAIMCP20X.DEVLIST", # 模型代码
        "dmNum": 10,                      # 模型编号
        "para": [dept, devdes]            # 字符串数组参数
    }

    result = get_ai_result(payload["dmCode"], payload["dmNum"], payload["para"], username)
    return result

# 设备维保记录
@mcp.tool()
def get_dev_mnt_list(
    dept: str, 
    devdes: str,
    startdate: str,
    enddate: str,
    username: str = "DINA"
) -> Dict[str, Any]:
    """
    按照车间、设备名称、日期范围获取设备维保记录
    Args:
        dept: 车间编码，必填
        devdes: 设备名称，如果未明确参数值，则使用空字符串
        startdate: 起始日期，格式为 "YYYY-MM-DD"，必填
        enddate: 结束日期，格式为 "YYYY-MM-DD"，必填
    Returns:
        JSON 格式的模型结果，其中 isSucess 为 True 表示成功，否则为 False, data 包含返回的实际数据
    """
    # 1. 构造请求参数
    payload = {
        "dmCode": "LINKAIMCP20X.DEVMNT",                # 模型代码
        "dmNum": 10,                                 # 模型编号
        "para": [dept, devdes, startdate, enddate]     # 字符串数组参数
    }

    result = get_ai_result(payload["dmCode"], payload["dmNum"], payload["para"], username)
    return result

# 待处理设备维保记录
@mcp.tool()
def get_dev_mnt_remain(
    dept: str, 
    devdes: str,
    username: str = "DINA"
) -> Dict[str, Any]:
    """
    等待处理的设备维保记录，已经创建维保记录，尚未完成处理的记录
    Args:
        dept: 车间编码，必填
        devdes: 设备名称，如果未明确参数值，则使用空字符串
    Returns:
        JSON 格式的模型结果，其中 isSucess 为 True 表示成功，否则为 False, data 包含返回的实际数据
    """
    # 1. 构造请求参数
    payload = {
        "dmCode": "LINKAIMCP20X.DEVMNT",              # 模型代码
        "dmNum": 20,                                  # 模型编号
        "para": [dept, devdes]                        # 字符串数组参数
    }

    result = get_ai_result(payload["dmCode"], payload["dmNum"], payload["para"], username)
    return result

# 登记设备维保记录
@mcp.tool()
def dev_mnt_reg(
    dept: str, 
    devno: str,
    mnttyp: str,
    mntdes: str,
    username: str = "DINA"
) -> Dict[str, Any]:
    """
    等待处理的设备维保记录，已经创建维保记录，尚未完成处理的记录
    Args:
        dept: 车间编码，必填
        devno: 设备编码，必填
        mnttyp: 维保类型，必填
        mntdes: 维保描述，必填
    Returns:
        JSON 格式的模型结果，其中 isSucess 为 True 表示成功，否则为 False
    """
    # 1. 构造请求参数
    payload = {
        "dmCode": "LINKAIMCP20X.DEVLIST",               # 模型代码
        "dmNum": 10,                                    # 模型编号
        "action": 501,                                  # 附加模型
        "para": [mnttyp, mntdes],                       # 字符串数组参数
        "rowdata": {"车间": dept, "设备编号": devno}    # 引用记录的数据
    }

    actionresult = get_ai_action(payload["dmCode"], payload["dmNum"],payload["action"], payload["para"], payload["rowdata"], username)

    if actionresult["status"] == "success":
      result = get_dev_mnt_remain(dept, "", username)
      return result
    else:
      return actionresult 

# ===============================================================================
def get_ai_result(
    code: str,          # 模型代码（必填）
    num: int,           # 模型编号（必填）
    para: List[str],    # 字符串数组参数（必填）
    username: str = "DINA"
) -> Dict[str, Any]:
    """
    获取指定模型的结果
    Args:
        code: 模型代码（如 "TMES10"、"TMES20"）
        num: 模型编号（如 8, 9, 10, 20, 30）
        para: 数组列表（如 ["p1","p2","p3"]）
    Returns:
        JSON 格式的模型结果
    """
    try:
        # print(para)
        # 1. 构造请求参数
        linktoken = username + " " + calc_value()

        payload = {
            "loginID": linktoken,
            "par": {
                "dmCode": code,
                "dmNum": num,
                "Para": para
            }
        }
        
        print(payload)
        
        # 2. 发送POST请求（替换为你的模型API地址）
        response = requests.post(
            "http://124.71.144.80:8088/api/DataModel/linkDMAIResult",  # 替换为实际地址
            json = payload,
            headers = {
                "User-Agent": "MCP-Model-Client/1.0",
                "Accept": "application/json",
                "X-Token": ""
            },
            timeout = 60  # 超时时间（秒）
        )
        
        # 3. 检查响应状态
        response.raise_for_status()
        result = response.json()
        
        # 4. 返回标准化结果
        return {
            # "para": payload,
            "status": "success",
            "json": result
        }
        
    except requests.exceptions.RequestException as e:
        # 错误处理
        return {
            "status": "error",
            "error": str(e)
        }

def get_ai_action(
    code: str,                # 模型代码（必填）
    num: int,                 # 模型编号（必填）
    action: int,              # 模型动作（必填）
    para: List[str],          # 字符串数组参数（必填）
    rowdata: Dict[str, str],  # 字符串数组参数（必填）
    username: str = "DINA"
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
        # print(para)
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
        
        print(payload)
        
        # 2. 发送POST请求（替换为你的模型API地址）
        response = requests.post(
            "http://124.71.144.80:8088/api/DataModel/linkDMContextFunBack",  # 替换为实际地址
            json = payload,
            headers = {
                "User-Agent": "MCP-Model-Client/1.0",
                "Accept": "application/json",
                "X-Token": ""
            },
            timeout = 60  # 超时时间（秒）
        )
        
        # 3. 检查响应状态
        response.raise_for_status()
        result = response.json()
        
        # 4. 返回标准化结果
        return {
            # "para": payload,
            "status": "success",
            "json": result
        }
        
    except requests.exceptions.RequestException as e:
        # 错误处理
        return {
            "status": "error",
            "error": str(e)
        }

# 获取TotalLINK的AI调用令牌
def calc_value():
    now = datetime.datetime.now()
    I = int(now.strftime("%S%M%H%y%m%d"))
    result = (I - 12251) * 12253 - 31321
    return str(result)

# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello, {name}!"

@mcp.prompt()
def inventory_analysis_prompt(user_question: str) -> str:
    """
    针对库存分析类问题，自动规划调用 get_inv_data、get_inv_detail、get_item_trx 的顺序和参数依赖。
    """
    return (
        "你是企业信息系统的智能助手，擅长库存数据分析。"
        "当用户提出如“分析某地点各仓库、各产品近几年变动情况”这类问题时，"
        "请自动按如下业务链路分步调用 MCP 工具：\n"
        "1. 先用 get_inv_data 查询地点的各仓库库存汇总信息。\n"
        "2. 分析返回结果，获取所有仓库编码。\n"
        "3. 对每个仓库，调用 get_inv_detail 获取产品明细。\n"
        "4. 对每个产品，结合地点和产品编码及时间范围，调用 get_item_trx 获取交易记录。\n"
        "每一步都要自动提取上一步的参数作为下一步的输入，最终汇总分析结果，给用户详细解答。"
        "如果用户没有给出地点、时间等参数，请主动询问。"
        f"\n用户问题：{user_question}"
    )

# Add a prompt
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

def test_get_sales_data():
    """
    启动时测试 get_sales_data 是否正常运行
    """
    print("\n[TEST] get_sales_data 测试开始...")
    # 示例参数，可根据实际情况调整
    startdate = "2021-01-01"
    enddate = "2021-01-07"
    company = ""
    cus = ""
    item = ""
    username = "RANDY"
    try:
        result = get_sales_data(startdate, enddate, company, cus, item, username)
        print("[TEST] get_sales_data 返回:", result)
    except Exception as e:
        print("[TEST] get_sales_data 异常:", e)
    print("[TEST] get_sales_data 测试结束\n")

if __name__ == "__main__":
    # test_get_sales_data()
    # result = get_dev_list("AC", "")
    result = dev_mnt_reg("AC", "CS02", "维修", "皮带断裂")
    print(result)

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