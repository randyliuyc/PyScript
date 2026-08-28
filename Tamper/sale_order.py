# -*- coding: utf-8 -*-
"""
销售订单新增接口（用友 T+ `saleOrder/Create`）LinkPython 程序。

输入（TotalLINK 通过 #LINKLEVELDATA# 传入的真实模型输出，字段为 TotalLINK 惯例）：
[
  {
    "id": "SOH260800001",            # TotalLINK 单据号 -> dto.ExternalCode（幂等键）
    "DOCDAT": "2026-08-03",          # 单据日期 -> dto.VoucherDate
    "BPCNO": "01024001",             # 客户编码 -> dto.Customer.Code
    "REMARK": "",                    # 备注 -> dto.Memo（可选）
    "children": [                    # 明细 -> dto.SaleOrderDetails
      {
        "ITMNO": "1504126",          # 存货编码 -> Inventory.Code
        "STU": "盒",                 # 计量单位 -> Unit.Name
        "QTYSAU": 1000.0,            # 销售数量 -> Quantity
        "GROPRI": 20.5               # 含税单价 -> OrigTaxPrice
      }
    ]
  }
]

说明：采用白名单映射，只挑上表列出的字段转换，其余 TotalLINK 内部字段
（pid/leaf/LVL/审计/SITE/BPCNAM/金额汇总等）全部丢弃，不发给用友。

输出（统一返回契约）：
{
  "isSuccess": true/false,
  "message": "推送完成：成功 N 单，失败 M 单",
  "results": [
    { "ExternalCode": "...", "status": "success", "yonyouCode": "...", "yonyouId": 123 },
    { "ExternalCode": "...", "status": "failed", "error": "..." }
  ]
}
"""
import json

import chanjet_client
from chanjet_client import ChanjetError

SALE_ORDER_CREATE = "/saleOrder/Create"

# ========== TotalLINK -> 用友 字段映射（白名单） ==========

# 主表原样透传字段：(TotalLINK字段名, 用友dto字段名)
_MAIN_PASSTHROUGH = (
    ("id", "ExternalCode"),     # 单据号 -> 幂等键
    ("DOCDAT", "VoucherDate"),  # 单据日期
    ("REMARK", "Memo"),         # 备注（可选）
)

# 主表客户编码字段（单独映射为嵌套对象 dto.Customer.Code）
_CUSTOMER_CODE_KEY = "BPCNO"

# 明细原样透传字段：(TotalLINK字段名, 用友明细字段名)
_DETAIL_PASSTHROUGH = (
    ("QTYSAU", "Quantity"),     # 销售数量
    ("GROPRI", "OrigTaxPrice"), # 含税单价
)

# 明细里需要在转换时包一层嵌套对象的字段及其目标键
# { 输入键: (目标父键, 目标子键) }
_DETAIL_INVENTORY = ("ITMNO", "Inventory", "Code")
_DETAIL_UNIT = ("STU", "Unit", "Name")


def _build_dto(order):
    """
    把 TotalLINK 销售订单转换为用友 saleOrder/Create 的 dto 请求体（白名单映射）。

    规则：
    - id/DOCDAT/REMARK 映射为主表字段
    - BPCNO -> dto.Customer.Code
    - children -> dto.SaleOrderDetails，其中：
        ITMNO -> SaleOrderDetails[].Inventory.Code
        STU   -> SaleOrderDetails[].Unit.Name
        QTYSAU -> Quantity、GROPRI -> OrigTaxPrice
    - 未列出的字段一律丢弃
    """
    dto = {}
    children = order.pop("children", []) or []

    for src, dst in _MAIN_PASSTHROUGH:
        value = order.get(src)
        if value is None or value == "":
            continue  # 跳过空值，避免传空字段
        dto[dst] = value

    customer_code = order.get(_CUSTOMER_CODE_KEY)
    if customer_code:
        dto["Customer"] = {"Code": customer_code}

    details = []
    for row in children:
        detail = {}
        for src, dst in _DETAIL_PASSTHROUGH:
            value = row.get(src)
            if value is None or value == "":
                continue
            detail[dst] = value
        inv = row.get(_DETAIL_INVENTORY[0])
        if inv:
            detail.setdefault(_DETAIL_INVENTORY[1], {})[_DETAIL_INVENTORY[2]] = inv
        unit = row.get(_DETAIL_UNIT[0])
        if unit:
            detail.setdefault(_DETAIL_UNIT[1], {})[_DETAIL_UNIT[2]] = unit
        if detail:
            details.append(detail)

    if details:
        dto["SaleOrderDetails"] = details

    # 手工指定单据号：用友单据号 = ExternalCode（TotalLINK 单号）
    # 原因：T+ 13.0 旧版 saleOrder/Create 不返回用友自动生成的单号（result: null），
    # 公共网关也无查询接口（GetVoucherDTO/FindVoucherList 需 16.0+）。
    # 通过 Code + IsModifiedCode 让用友直接使用我们的单号，保证单号一一对应、始终可知。
    external_code = dto.get("ExternalCode")
    if external_code:
        dto["Code"] = external_code
        dto["IsModifiedCode"] = True

    # 创建后自动审核（T+ 动态属性 isautoaudit，官方推荐的创建即审核方式）
    # 若账套开启审批流，需另加 isneedwfsubmit=1（见 chanjet 销售订单文档）
    dto.setdefault("DynamicPropertyKeys", []).append("isautoaudit")
    dto.setdefault("DynamicPropertyValues", []).append(True)

    return dto


def _push_one_order(order):
    """
    推送单张销售订单，返回 (status, 用友信息或错误)。
    """
    try:
        dto = _build_dto(dict(order))
        payload = {"dto": dto}
        resp = chanjet_client.call_api(SALE_ORDER_CREATE, payload)
    except ChanjetError as e:
        return {"status": "failed", "error": str(e)}

    # 解析响应：创建成功时 T+ 返回 HTTP 200 + null，用友单号回退到 ExternalCode
    data = resp.get("data") or resp.get("result") or {}
    yonyou_code = data.get("code") or data.get("Code") or dto.get("ExternalCode")
    yonyou_id = data.get("id") or data.get("Id")

    entry = {"ExternalCode": dto.get("ExternalCode", "")}
    if yonyou_id is not None:
        entry["yonyouId"] = yonyou_id
    if yonyou_code:
        entry["yonyouCode"] = yonyou_code
    entry["status"] = "success"
    return entry


def linkrun(json_str):
    """
    LinkPython 程序统一入口。接收销售订单数组 JSON，逐单推送，返回统一结果契约。
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return json.dumps(
            {
                "isSuccess": False,
                "message": "输入不是合法 JSON",
                "results": [],
            },
            ensure_ascii=False,
        )

    # 兼容：支持 {"orders": [...]} 或直接 [...] 两种形态
    orders = data.get("orders") if isinstance(data, dict) else data
    if not isinstance(orders, list):
        return json.dumps(
            {
                "isSuccess": False,
                "message": "输入应为订单数组或包含 orders 数组",
                "results": [],
            },
            ensure_ascii=False,
        )

    results = []
    for order in orders:
        if not isinstance(order, dict):
            results.append({"ExternalCode": "", "status": "failed", "error": "订单非对象"})
            continue
        results.append(_push_one_order(order))

    success_cnt = sum(1 for r in results if r["status"] == "success")
    fail_cnt = len(results) - success_cnt
    is_success = fail_cnt == 0

    return json.dumps(
        {
            "isSuccess": is_success,
            "message": f"推送完成：成功 {success_cnt} 单，失败 {fail_cnt} 单",
            "results": results,
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":

    sample = """{
    "orders": 
[
  {
    "id": "SOH260800001",
    "DOCDAT": "2026-08-03",
    "BPCNO": "01024001",
    "BPCNAM": "烟台靖众商贸有限公司（老佛爷）",
    "SITE": "YT",
    "REMARK": "",
    "children": [
      {"ITMNO": "1504126", "STU": "盒", "QTYSAU": 1000.0, "GROPRI": 20.5},
      {"ITMNO": "1504200", "STU": "盒", "QTYSAU": 500.0, "GROPRI": 6.0}
    ]
  }
]
}
"""
    print(linkrun(sample))
