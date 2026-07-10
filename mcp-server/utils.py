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
def calc_value() -> str:
    """获取 TotalLINK 的 AI 调用令牌"""
    now = datetime.datetime.now()
    I = int(now.strftime("%S%M%H%y%m%d"))

    result = (I - 12251) * 12253 - 31321

    return str(result)


def _build_linktoken(userid: str) -> str:
    """构建认证令牌"""
    if len(userid) == 32 and userid.isalnum():
        return userid
    return f"{userid} {calc_value()}"


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

        返回所有可用的 TotalLINK 模型工具及其描述。
        AI 根据工具描述选择合适的工具名称，然后通过 call_dynamic_tool 调用。

        Args:
            userid: TotalLINK用户名
            force_refresh: 是否强制刷新缓存（默认 False，使用1小时缓存）

        Returns:
            工具列表，包含 name、description，以及 total 数量
        """
        # 类型兜底
        if isinstance(force_refresh, str):
            force_refresh = force_refresh.lower() == "true"
        tools = await get_user_tools(userid, force_refresh=force_refresh)

        return {
            "total": len(tools),
            "tools": [
                {"name": t["name"], "description": t["description"]}
                for t in tools
            ]
        }

    @mcp.tool()
    async def match_tool(userid: str = "", query: str = "") -> Dict[str, Any]:
        """
        【工具匹配】根据用户问题自动匹配最合适的 TotalLINK 模型工具

        当用户描述需求时，调用此工具进行关键词匹配，找到最相关的工具。
        匹配成功时只返回匹配到的工具，不返回完整列表，节省 token。
        匹配失败时返回 top 5 候选，而非全部工具。

        Args:
            userid: TotalLINK用户名
            query: 用户的自然语言问题或需求描述

        Returns:
            matched: 是否找到高置信度匹配
            tool: 最佳匹配工具（高置信度时直接使用此 name 调用 call_dynamic_tool）
            candidates: 候选工具列表（最多 5 个，供 AI 二次判断）
            hint: 如需查看全部工具，请调用 get_tools
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
        # high_confidence = len(candidates) == 1
        # 检查是否有同名但不同描述的工具
        same_name_tools = [t for t in candidates if t["name"] == best["name"]]
        has_duplicate_names = len(same_name_tools) > 1

        high_confidence = len(candidates) == 1 and not has_duplicate_names

        return {
            "matched": high_confidence,
            "tool": {
                "name": best["name"],
                "description": best["description"]
            },
            "candidates": [
                {"name": t["name"], "description": t["description"]}
                for t in candidates
            ],
            "total_available": len(tools),
            "message": (
                f"✅ 高置信度匹配「{best['name']}」，请直接使用 call_dynamic_tool 调用"
                if high_confidence
                else (
                    f"⚠️ 找到 {len(same_name_tools)} 个同名工具「{best['name']}」，"
                    f"请根据描述选择对应的候选。调用 call_dynamic_tool 时传入具体描述"
                    if has_duplicate_names
                    else f"找到 {len(candidates)} 个候选工具，推荐「{best['name']}」"
                )
            )
        }


    @mcp.tool()
    async def call_dynamic_tool(
        tool_name: str,
        toolid: str = "",
        parameters: List[str] = [],
        userid: str = "",
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
        script_type: Any = -1,
        row_data: Dict[str, Any] = None,
        table_data: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        【统一入口】调用指定的 TotalLINK 模型工具（4种模式自动路由）（默认分页）

        通过工具名称调用 SEARCHLIST-100 中配置的模型工具。
        根据工具在 SEARCHLIST-100 中配置的 ToolType 自动选择调用方式。
        始终分页返回，避免 token 超限。每页默认 {page_size} 条。

        Args:
            tool_name: 工具名称（从 get_tools 或 match_tool 返回的 name 字段）
            toolid: 工具唯一ID（从 match_tool 返回的 toolid 字段）。当存在同名工具时，优先使用 toolid 精确匹配，避免歧义
            parameters: 工具参数数组，按顺序传入模型，如 ["", "2026-06-14", ""]，工具参数数组，按顺序传入。空字符串 "" 表示该位置不传参
            userid: TotalLINK用户名
            page: 页码（从1开始，默认第1页）
            page_size: 每页数据量（默认 {page_size}，最大建议不超过 50）AIResult 模式有效
            script_type: 脚本类型，整数，AIRowSubmit 模式有效
            row_data: AIRowSubmit/AIDataSubmit 模式有效
            table_data: 仅 AIDataSubmit 模式有效

        Returns:
            根据 ToolType 返回对应结果，包含 data 和 pagination 分页信息。
            当前页数据，包含 data 和 pagination 分页信息。
            ⚠️ 禁止自动翻页！仅返回当前页数据。
            只有当用户明确要求"下一页"、"更多"时，才继续翻页。
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

        # ===== 匹配工具：优先 toolid，其次 name =====
        tool_def = None

        if toolid:
            # 精确 toolid 匹配
            tool_def = next((t for t in tools if str(t.get("toolid", "")) == toolid), None)

        if not tool_def:
            # 按 name 匹配
            same_name_list = [t for t in tools if t["name"] == tool_name]

            if len(same_name_list) == 0:
                return {
                    "isSuccess": "false",
                    "message": f"工具 '{tool_name}' 不存在或未授权，请先调用 get_tools 查看可用工具"
                }
            elif len(same_name_list) > 1:
                # 同名工具存在，但没传 toolid → 无法精确区分，返回提示
                return {
                    "isSuccess": "false",
                    "message": (
                        f"工具 '{tool_name}' 存在 {len(same_name_list)} 个同名工具，无法确定调用哪一个。"
                        f"请先调用 match_tool 匹配具体工具，然后使用返回的 toolid 调用本方法。"
                        f"候选工具: {[t.get('description','') for t in same_name_list]}"
                    )
                }
            else:
                tool_def = same_name_list[0]

        tool_type = tool_def.get("toolType", "AIResult")

        logger.info(f"[CallDynamicTool] {tool_type}, toolid={tool_def.get('toolid')}, scriptType={script_type}, para={parameters}")
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
                code=tool_def["dmCode"],
                num=tool_def["dmNum"],
                script_type=script_type,
                para=parameters,
                row_data=row_data,
                userid=userid
            )

        elif tool_type == "AIDataSubmit":
            return await ai_data_submit(
                code=tool_def["dmCode"],
                num=tool_def["dmNum"],
                script_type=script_type,
                para=parameters,
                row_data=row_data,
                table_data=table_data,
                userid=userid
            )

        else:
            return await get_ai_result(
                code=tool_def["dmCode"],
                num=tool_def["dmNum"],
                para=parameters,
                userid=userid,
                page=page,
                page_size=min(page_size, 50)
            )
