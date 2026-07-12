# utils.py - TotalLINK MCP 工具集（精简版）
from typing import Dict, Any, List, Optional
import copy
import datetime
import httpx
from loguru import logger

# ============ 常量定义 ============
BASE_URL = "http://124.71.144.80:8088"
DEFAULT_PAGE_SIZE = 20  # 每页默认 20 条，避免 token 超限

# ============ 用户工具缓存 ============
_user_tool_cache: Dict[str, Dict] = {}  # userid -> {tools, timestamp}

# ============ HTTP 客户端 ============
client = httpx.AsyncClient(
    timeout=httpx.Timeout(
        connect=5.0,
        read=30.0,
        write=30.0,
        pool=30.0
    ),
    headers={
        "User-Agent": "MCP-Model-Client/1.0",
        "Accept": "application/json",
        "X-Token": ""
    }
)

logger.add("server.log")


# ============ 内部工具函数 ============
def _build_linktoken(userid: str) -> str:
    """构建认证令牌"""
    return "tlk_a69a494fc83e97f0424366cf382e467263b92a20"


def paginate_data(data: Dict[str, Any], page: int = 1, page_size: int = DEFAULT_PAGE_SIZE) -> Dict[str, Any]:
    """对数据进行分页处理，兼容 {schema+data} 数组格式和传统 Table 数组格式"""
    # 取出 Table
    table = data.get("Table")
    inner_data = data.get("data", {})
    if isinstance(inner_data, dict):
        table = table or inner_data.get("Table")

    # 判断格式：{schema: [...], data: [[...]]} 还是 [{...}, {...}]
    schema = []
    if isinstance(table, dict) and "data" in table:
        # 新格式：{schema, data}
        schema = table.get("schema", [])
        rows = table.get("data", [])
    elif isinstance(table, list):
        # 旧格式：[{...}, {...}]
        rows = table
    else:
        rows = data.get("Table", []) or data.get("Rows", [])
        # 再次尝试 dict 格式
        if isinstance(rows, dict) and "data" in rows:
            schema = rows.get("schema", [])
            rows = rows.get("data", [])

    if not rows:
        return {
            "isSuccess": "true",
            "data": data,
            "pagination": {
                "current_page": 1,
                "total_pages": 0,
                "total_items": 0
            },
            "message": "无数据"
        }

    total = len(rows)
    total_pages = (total + page_size - 1) // page_size
    page = max(1, min(page, total_pages))

    start = (page - 1) * page_size
    end = min(start + page_size, total)
    page_rows = rows[start:end]

    # 有 schema → 转成字典格式，AI 才能看懂
    if schema:
        formatted_rows = [dict(zip(schema, row)) for row in page_rows]
    else:
        formatted_rows = page_rows

    # 构建返回数据（保持原结构，只替换数据部分）
    result_data = copy.deepcopy(data)
    if isinstance(result_data.get("data"), dict) and "Table" in result_data["data"]:
        result_data["data"]["Table"]["data"] = formatted_rows
    elif "Table" in result_data:
        result_data["Table"] = formatted_rows

    return {
        "isSuccess": "true",
        "data": result_data,
        "pagination": {
            "current_page": page,
            "total_pages": total_pages,
            "total_items": total,
            "page_size": len(page_rows),
            # "has_more": end < total,
            # "next_page": page + 1 if end < total else None
        },
        # "message": f"第 {page}/{total_pages} 页，共 {total} 条"
        "message": (
            f"第 {page}/{total_pages} 页，共 {total} 条。"
            f"当前显示 {len(page_rows)} 条，还有 {total - end} 条未显示。"
            f"用户需要时再调用本工具传入 page={page + 1} 获取下一页。"
            if end < total
            else f"第 {page}/{total_pages} 页，共 {total} 条。已显示全部数据。"
        )
    }



# ============ 核心 API 调用 ============
async def call_link_api(endpoint, payload, linkurl=BASE_URL):
    try:
        response = await client.post(
            f"{linkurl}/api/DataModel/{endpoint}",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP {e.response.status_code}: {e.response.text[:300]}")
        return {"isSuccess": "false", "message": f"HTTP {e.response.status_code}: 请求失败"}
    except httpx.RequestError as e:
        logger.error(f"API call failed: {str(e)}")
        return {"isSuccess": "false", "message": str(e)}



async def get_ai_result(
    code: str,
    num: int,
    para: List[str],
    userid: str,
    linkurl: str = BASE_URL,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    endpoint: str = "linkDMAIResult"  # ← 新增：默认执行工具查询
) -> Dict[str, Any]:
    """获取模型查询结果（默认分页，保护 token 不超限）"""
    payload = {
        "loginID": _build_linktoken(userid),
        "par": {
            "dmCode": code,
            "dmNum": num,
            "Para": para
        }
    }

    logger.info(f"[AIResult] {code}/{num}, para={para}")
    logger.debug(f"[AIResult] payload keys={list(payload.keys())}")

    result = await call_link_api(endpoint, payload, linkurl)  # ← 使用参数
    is_success = str(result.get("isSuccess", "")).lower()
    if is_success == "false":
        logger.warning(f"[AIResult] {code}/{num} failed: {result.get('message', '')[:100]}")
        return result

    # 始终分页，保护 token 不超限
    data = result.get("data", result)
    return paginate_data(data, page, page_size)

async def ai_action(
    code: str,
    num: int,
    action_no: int,
    para: List[str],
    row_data: Dict[str, Any] = None,
    userid: str = "",
    linkurl: str = BASE_URL
) -> Dict[str, Any]:
    """执行模型动作（增删改等操作），不同于查询接口"""
    if row_data is None:
        row_data = {}

    payload = {
        "loginID": _build_linktoken(userid),
        "par": {
            "dm": {
                "dmCode": code,
                "dmNum": num,
                "Para": para
            },
            "contextMenuNo": action_no,
            "rowData": row_data
        }
    }

    logger.info(f"Action: {code}/{num}, action_no: {action_no}, params: {row_data}")

    return await call_link_api("linkDMAIAction", payload, linkurl)

async def ai_row_submit(
    code: str,
    num: int,
    script_type: int,
    para: List[str],
    row_data: Dict[str, Any] = None,
    userid: str = "",
    linkurl: str = BASE_URL
) -> Dict[str, Any]:
    """行数据提交操作"""
    if row_data is None:
        row_data = {}

    payload = {
        "loginID": _build_linktoken(userid),
        "par": {
            "dm": {
                "dmCode": code,
                "dmNum": num,
                "Para": para
            },
            "scriptType": script_type,
            "rowData": row_data
        }
    }

    logger.info(f"[AIRowSubmit] {code}/{num}, script_type={script_type}")
    logger.debug(f"[AIRowSubmit] row_data={row_data}")
    # print(f"RowSubmit: {code}/{num}, scriptType: {script_type}")
    return await call_link_api("linkDMAIRowSubmit", payload, linkurl)

async def ai_data_submit(
    code: str,
    num: int,
    script_type: int,
    para: List[str],
    row_data: Dict[str, Any] = None,
    table_data: Dict[str, Any] = None,
    userid: str = "",
    linkurl: str = BASE_URL
) -> Dict[str, Any]:
    """数据集提交操作"""
    if row_data is None:
        row_data = {}

    payload = {
        "loginID": _build_linktoken(userid),
        "par": {
            "dm": {
                "dmCode": code,
                "dmNum": num,
                "Para": para
            },
            "scriptType": script_type,
            "rowData": row_data,
            "tableData": table_data
        }
    }

    logger.info(f"[AIDataSubmit] {code}/{num}")
    logger.debug(f"[AIDataSubmit] row_data={row_data}, table_data={table_data}")
    # print(f"DataSubmit: {code}/{num}")
    return await call_link_api("linkDMAIDataSubmit", payload, linkurl)

# ============ 动态工具管理 ============
async def fetch_tools_from_linkai(userid: str = "") -> List[Dict[str, Any]]:
    """从 SEARCHLIST-100 获取工具列表"""
    result = await get_ai_result(
        "SEARCHLIST", 100, [], userid,
        endpoint="linkDMAIResult"
    )

    tools = []
    is_success = str(result.get("isSuccess", "")).lower()
    if is_success != "false":
        # 取 Table，兼容两种格式
        table = result.get("data", {}).get("Table")
        if table is None:
            table = result.get("Table")

        if isinstance(table, dict) and "data" in table:
            # 新格式：{schema: ["DMCODE", "DMNUM", "DMDESC", ...], data: [[...], ...]}
            schema = table.get("schema", [])
            rows = table.get("data", [])
            for row in rows:
                row_dict = dict(zip(schema, row))
                tools.append({
                    "toolid": row_dict.get("TOOL_ID"),
                    "dmCode": row_dict.get("TOOL_CODE"),
                    "dmNum": row_dict.get("TOOL_NUM", 10),
                    "name": row_dict.get("TOOL_NAME", ""),
                    "description": row_dict.get("TOOL_DESC", ""),
                    "params": row_dict.get("PARAMS", []),
                    "toolType": row_dict.get("TOOL_TYPE", "AIResult"),  # ← 新增
                })
        elif isinstance(table, list):
            # 旧格式：[{DMCODE: ..., DMDESC: ...}, ...]
            for row in table:
                tools.append({
                    "toolid": row.get("TOOL_ID"),
                    "dmCode": row.get("TOOL_CODE"),
                    "dmNum": row.get("TOOL_NUM", 10),
                    "name": row.get("TOOL_NAME", ""),
                    "description": row.get("TOOL_DESC", ""),
                    "params": row.get("PARAMS", []),
                    "toolType": row.get("TOOL_TYPE", "AIResult"),
                })

    return tools



async def get_user_tools(userid: str, force_refresh: Any = False) -> List[Dict]:
    """获取用户的工具列表（带1小时缓存）"""
    cache_key = userid or "anonymous"
    now = datetime.datetime.now()

    if not force_refresh and cache_key in _user_tool_cache:
        cache = _user_tool_cache[cache_key]
        if (now - cache["timestamp"]).total_seconds() < 900:
            return cache["tools"]

    tools = await fetch_tools_from_linkai(userid)
    _user_tool_cache[cache_key] = {
        "tools": tools,
        "timestamp": now
    }
    return tools


def match_tool_by_query(query: str, tools: List[Dict], top_n: int = 5) -> List[Dict]:
    """
    根据用户问题匹配工具，返回按相关度排序的结果列表。
    
    匹配优先级：名称完全匹配 > 名称包含匹配 > 描述关键词匹配
    返回最多 top_n 个候选（默认 5 个），节省 token。
    """
    if not query or not tools:
        return []

    query_lower = query.strip().lower()
    keywords = set(query_lower.split())

    scored = []

    for t in tools:
        name_lower = t["name"].strip().lower()
        desc_lower = (t.get("description") or "").lower()

        # 描述关键词命中数
        desc_score = sum(1 for kw in keywords if kw in desc_lower or kw in name_lower)

        if name_lower == query_lower:
            # 完全匹配名称 = 基础 1000 + 描述加成
            scored.append((1000 + desc_score, t))
        elif name_lower in query_lower or query_lower in name_lower:
            # 名称包含匹配 = 基础 500 + 描述加成
            scored.append((500 + desc_score, t))
        elif desc_score > 0:
            # 纯描述匹配
            scored.append((desc_score, t))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:top_n]]

# ============ MCP 工具注册（统一入口） ============
def register_ai_tools(mcp):
    """注册 AI 工具 - 统一入口架构"""

    @mcp.tool()
    async def get_tools(userid: str = "", force_refresh: Any = False) -> Dict[str, Any]:
        """
        【工具列表】获取当前用户可用的动态工具列表

        返回所有已授权的 TotalLINK 模型工具。工具较多时优先使用 match_tool，工具少时直接从此列表选择。

        Args:
            userid: TotalLINK用户名
            force_refresh: 是否强制刷新缓存（默认 False，缓存15分钟）

        Returns:
            工具列表，每项包含 dmCode、dmNum、name、description、toolType，以及 total 数量
            ⚠️ 调用 call_dynamic_tool 时必须使用 dmCode 和 dmNum
        """
        # 类型兜底
        if isinstance(force_refresh, str):
            force_refresh = force_refresh.lower() == "true"
        tools = await get_user_tools(userid, force_refresh=force_refresh)

        return {
            "total": len(tools),
            "tools": [
                {"tool_id": t["toolid"], "dmCode": t["dmCode"], "dmNum": t["dmNum"], "name": t["name"], "description": t["description"], "toolType": t["toolType"]}
                for t in tools
            ]
        }

    @mcp.tool()
    async def match_tool(userid: str = "", query: str = "") -> Dict[str, Any]:
        """
        【工具匹配】根据用户问题语义匹配最合适的工具

        返回最佳匹配（含 tool_id、toolType），匹配成功直接调用，无需再查完整列表，节省 token。
        匹配不准时返回 top 5 候选供 AI 二次判断。工具很少时直接用 get_tools 即可。

        Args:
            userid: TotalLINK用户名
            query: 用户的自然语言需求描述

        Returns:
            matched: 是否唯一高置信度匹配
            tool: 最佳匹配（含 tool_id、name、description、toolType），matched=true 时直接用 tool_id 调用
            candidates: 候选列表（最多5个，含 tool_id、toolType），matched=false 时按描述选最合适的
            total_available: 用户可用工具总数
            message: 匹配结果提示
        """
        tools = await get_user_tools(userid)
        candidates = match_tool_by_query(query, tools, top_n=5)

        if not candidates:
            return {
                "matched": False,
                "tool": None,
                "candidates": [],
                "message": f"未匹配到相关工具，请调用 get_tools 查看全部 {len(tools)} 个可用工具"
            }

        # 第一个候选分数最高，作为推荐
        best = candidates[0]
        logger.info(f"[MatchTool] userid={userid}, query='{query}', total_tools={len(tools)},toools:{candidates}")

        high_confidence = len(candidates) == 1

        return {
            "matched": high_confidence,
            "tool": {
                "tool_id": best["toolid"],
                "dmCode": best["dmCode"],
                "dmNum": best["dmNum"],
                "name": best["name"],
                "description": best["description"],
                "toolType": best["toolType"]
            },
            "candidates": [
                {"tool_id": t["toolid"], "dmCode": t["dmCode"], "dmNum": t["dmNum"], "name": t["name"], "description": t["description"], "toolType": t["toolType"]}
                for t in candidates
            ],
            "total_available": len(tools),
            "message": (
                f"✅ 高置信度匹配「{best['name']}」，请直接使用 dmCode/dmNum 调用 call_dynamic_tool"
                if high_confidence
                else f"找到 {len(candidates)} 个候选工具，推荐「{best['name']}」，请根据描述选择对应的 dmCode/dmNum 调用"
            )
        }


    @mcp.tool()
    async def call_dynamic_tool(
        dmCode: str,
        dmNum: int = 10,
        parameters: List[str] = [],
        userid: str = "",
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
        script_type: Any = -1,
        row_data: Dict[str, Any] = None,
        table_data: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        【统一入口】调用 TotalLINK 模型工具（3种模式自动路由，默认分页）

        通过 dmCode/dmNum 定位工具，根据 toolType 自动选择 AIResult / AIRowSubmit / AIDataSubmit 模式。
        所有工具通用: dmCode（必填）+ dmNum（必填）+ userid（必填）+ parameters（按位置数组，空位传 ""）

        Args:
            dmCode: 模型编码（从 get_tools 或 match_tool 获取，必填）
            dmNum: 模型编号（从 get_tools 或 match_tool 获取，必填）
            parameters: 参数数组，按工具 description 中的顺序传入，空位传 ""。如 ["", "2026-06-14", ""]
            userid: TotalLINK用户名（必填）
            page: 页码，从1开始（仅 AIResult 有效，默认1）
            page_size: 每页条数（仅 AIResult 有效，默认 {page_size}，最大50）
            script_type: 操作类型整数（仅 AIRowSubmit/AIDataSubmit 需要，从工具 description 获取）
            row_data: 单行数据 dict（仅 AIRowSubmit/AIDataSubmit 需要，字段从工具 description 获取）
            table_data: 批量数据 list[dict]（仅 AIDataSubmit 需要）

        Returns:
            AIResult → {{ data, pagination }}  分页结果，禁止自动翻页
            AIRowSubmit/AIDataSubmit → {{ isSuccess, message }}  操作结果
        """.format(page_size=DEFAULT_PAGE_SIZE)
        # ========== 类型强制转换 ==========
        if row_data is None:
            row_data = {}
        if table_data is None:
            table_data = []
        if isinstance(parameters, str):
            import json
            try:
                parameters = json.loads(parameters)
            except (json.JSONDecodeError, TypeError):
                parameters = [parameters]
        elif isinstance(parameters, dict):
            parameters = list(parameters.values())
        elif not isinstance(parameters, list):
            parameters = []
        parameters = [str(p) if p is not None else "" for p in parameters]

        try:
            page = int(page)
        except (ValueError, TypeError):
            page = 1
        try:
            page_size = int(page_size)
        except (ValueError, TypeError):
            page_size = DEFAULT_PAGE_SIZE
        try:
            script_type = int(script_type)
        except (ValueError, TypeError):
            script_type = -1
        # =========================================

        tools = await get_user_tools(userid)

        # ===== 匹配工具：dmCode + dmNum =====
        tool_def = next(
            (t for t in tools if t.get("dmCode") == dmCode and int(t.get("dmNum", 0)) == dmNum),
            None
        )

        if not tool_def:
            return {
                "isSuccess": "false",
                "message": (
                    f"工具 dmCode='{dmCode}' dmNum={dmNum} 不存在或未授权。"
                    f"请先调用 get_tools 获取可用工具列表"
                )
            }

        tool_type = tool_def.get("toolType", "AIResult")

        logger.info(f"[CallDynamicTool] {tool_type}, dmCode={dmCode}, dmNum={dmNum}, scriptType={script_type}, para={parameters}")
        import re
        desc = tool_def.get("description", "")
        # ===== 按 ToolType 路由 =====
        if tool_type == "AIRowSubmit" or tool_type == "AIDataSubmit":
            if script_type < 0:
                match = re.search(r'script_type["\s]*[=：:]\s*(\d+)', desc)
                if match:
                    script_type = int(match.group(1))

        if tool_type == "AIRowSubmit":
            return await ai_row_submit(
                code=dmCode,
                num=dmNum,
                script_type=script_type,
                para=parameters,
                row_data=row_data,
                userid=userid
            )

        elif tool_type == "AIDataSubmit":
            return await ai_data_submit(
                code=dmCode,
                num=dmNum,
                script_type=script_type,
                para=parameters,
                row_data=row_data,
                table_data=table_data,
                userid=userid
            )

        else:
            return await get_ai_result(
                code=dmCode,
                num=dmNum,
                para=parameters,
                userid=userid,
                page=page,
                page_size=min(page_size, 50)
            )
