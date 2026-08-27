# -*- coding: utf-8 -*-
"""
销售订单新增接口（用友 T+ `saleOrder/Create`）LinkPython 程序。

输入（TotalLINK 通过 #LINKLEVELDATA# 传入，字段名与用友接口一致但结构扁平）：
[
  {
    "ExternalCode": "SOH26080001",     # TotalLINK 业务订单号（幂等键）
    "VoucherDate": "2026-08-27",
    "CustomerCode": "102311037",       # 客户编码 -> dto.Customer.Code
    "Memo": "测试OpenAPI",
    "children": [                       # 明细 -> dto.SaleOrderDetails
      { "InventoryCode": "...", "UnitName": "...", "Quantity": 1, "OrigTaxPrice": "188.87" }
    ]
  }
]

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

# 输入明细行里，需要在转换时包一层嵌套对象的字段及其目标键
# { 输入键: (目标父键, 目标子键) }
_DETAIL_INVENTORY = ("Inventory", "Code")
_DETAIL_UNIT = ("Unit", "Name")


def _build_dto(order):
    """
    把 TotalLINK 扁平订单转换为用友 saleOrder/Create 的 dto 请求体。

    规则：
    - ExternalCode / VoucherDate / Memo 等主表字段原样透传进 dto
    - CustomerCode -> dto.Customer.Code
    - children -> dto.SaleOrderDetails，其中：
        InventoryCode -> SaleOrderDetails[].Inventory.Code
        UnitName      -> SaleOrderDetails[].Unit.Name
        Quantity / OrigTaxPrice 等原样透传
    """
    dto = {}
    children = order.pop("children", []) or []

    for key, value in order.items():
        if value is None or value == "":
            continue  # 跳过空值，避免传空字段
        if key == "CustomerCode":
            dto["Customer"] = {"Code": value}
        else:
            dto[key] = value

    details = []
    for row in children:
        detail = {}
        for key, value in row.items():
            if value is None or value == "":
                continue
            if key == "InventoryCode":
                detail.setdefault(_DETAIL_INVENTORY[0], {})[_DETAIL_INVENTORY[1]] = value
            elif key == "UnitName":
                detail.setdefault(_DETAIL_UNIT[0], {})[_DETAIL_UNIT[1]] = value
            else:
                detail[key] = value
        details.append(detail)

    if details:
        dto["SaleOrderDetails"] = details

    return dto


def _push_one_order(order):
    """
    推送单张订单，返回 (status, 用友信息或错误)。
    """
    try:
        dto = _build_dto(dict(order))
        payload = {"dto": dto}
        resp = chanjet_client.call_api(SALE_ORDER_CREATE, payload)
    except ChanjetError as e:
        return {"status": "failed", "error": str(e)}

    # 解析响应：取用友单据号/ID（字段名以实际响应为准，这里做兼容读取）
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
    LinkPython 程序统一入口。接收订单数组 JSON，逐单推送，返回统一结果契约。
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
    sample = """
[
  {
    "ExternalCode": "SOH26080001",
    "VoucherDate": "2026-08-27",
    "CustomerCode": "102311037",
    "Memo": "测试OpenAPI",
    "children": [
      {"InventoryCode": "06020502", "UnitName": "台", "Quantity": 1, "OrigTaxPrice": "188.87"},
      {"InventoryCode": "0501GX01003", "UnitName": "1000g装", "Quantity": 5, "OrigTaxPrice": "20.5"}
    ]
  }
]
"""
    print(linkrun(sample))
