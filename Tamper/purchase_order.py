# -*- coding: utf-8 -*-
"""
采购订单新增接口（用友 T+ `purchaseOrder/Create`）LinkPython 程序。

输入（TotalLINK 通过 #LINKLEVELDATA# 传入，字段名与用友接口一致但结构扁平）：
[
  {
    "ExternalCode": "POH26080001",     # TotalLINK 业务订单号（幂等键）
    "VoucherDate": "2026-08-28",
    "SupplierCode": "102311024",       # 供应商编码 -> dto.Partner.Code（兼容 PartnerCode/VendorCode）
    "Memo": "测试采购订单",
    "children": [                       # 明细 -> dto.PurchaseOrderDetails
      {
        "InventoryCode": "01M56",
        "UnitName": "kg",
        "Quantity": 1,
        "OrigTaxPrice": "100.00",       # 含税单价
        "OrigDiscountPrice": "100.00"   # 折后含税单价（供应商报价不含税/价外税时必填）
      }
    ]
  }
]

输出（统一返回契约，与 sale_order.py 一致）：
{
  "isSuccess": true/false,
  "message": "推送完成：成功 N 单，失败 M 单",
  "results": [
    { "ExternalCode": "...", "status": "success", "yonyouCode": "...", "yonyouId": 123 },
    { "ExternalCode": "...", "status": "failed", "error": "..." }
  ]
}

已验证的请求体（2026-08-28 真实账套调通）：
POST /tplus/api/v2/purchaseOrder/Create
{
  "dto": {
    "ExternalCode": "...",
    "VoucherDate": "2026-08-28",
    "Partner": {"Code": "102311024"},
    "BusinessType": {"Code": "PO"},
    "PurchaseOrderDetails": [
      {"Inventory": {"Code": "01M56"}, "Quantity": 1,
       "OrigTaxPrice": "100.00", "OrigDiscountPrice": "100.00"}
    ]
  }
}
成功时返回 HTTP 200 + null body（与 saleOrder/Create 一致）。
"""
import json

import chanjet_client
from chanjet_client import ChanjetError

PURCHASE_ORDER_CREATE = "/purchaseOrder/Create"

# 采购订单固定业务类型（T+ 采购订单类型编码）
PURCHASE_BUSINESS_TYPE = "PO"

# 输入明细行里，需要在转换时包一层嵌套对象的字段及其目标键
# { 输入键: (目标父键, 目标子键) }
_DETAIL_INVENTORY = ("Inventory", "Code")
_DETAIL_UNIT = ("Unit", "Name")

# 主表供应商编码的候选输入键（按优先级匹配，兼容不同 TotalLINK 模型字段名）
_SUPPLIER_CODE_KEYS = ("SupplierCode", "PartnerCode", "VendorCode")


def _build_dto(order):
    """
    把 TotalLINK 扁平采购订单转换为用友 purchaseOrder/Create 的 dto 请求体。

    规则（与 sale_order.py 同构，差异点见注释）：
    - ExternalCode / VoucherDate / Memo 等主表字段原样透传进 dto
    - SupplierCode（或 PartnerCode/VendorCode）-> dto.Partner.Code
    - 固定 dto.BusinessType.Code = "PO"（采购订单业务类型）
    - children -> dto.PurchaseOrderDetails，其中：
        InventoryCode -> PurchaseOrderDetails[].Inventory.Code
        UnitName      -> PurchaseOrderDetails[].Unit.Name
        Quantity / OrigTaxPrice / OrigDiscountPrice 等原样透传
    """
    dto = {}
    children = order.pop("children", []) or []

    for key, value in order.items():
        if value is None or value == "":
            continue  # 跳过空值，避免传空字段
        if key in _SUPPLIER_CODE_KEYS:
            dto["Partner"] = {"Code": value}
        else:
            dto[key] = value

    # 采购订单业务类型固定为采购订单（与销售订单 SaleOrder/BusiType 类似的必填项）
    dto.setdefault("BusinessType", {"Code": PURCHASE_BUSINESS_TYPE})

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
        dto["PurchaseOrderDetails"] = details

    return dto


def _push_one_order(order):
    """
    推送单张采购订单，返回 (status, 用友信息或错误)。
    """
    try:
        dto = _build_dto(dict(order))
        payload = {"dto": dto}
        resp = chanjet_client.call_api(PURCHASE_ORDER_CREATE, payload)
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
    LinkPython 程序统一入口。接收采购订单数组 JSON，逐单推送，返回统一结果契约。
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
    "ExternalCode": "POH26080001",
    "VoucherDate": "2026-08-28",
    "SupplierCode": "102311024",
    "Memo": "测试采购订单",
    "children": [
      {"InventoryCode": "01M56", "UnitName": "kg", "Quantity": 1, "OrigTaxPrice": "100.00", "OrigDiscountPrice": "100.00"},
      {"InventoryCode": "06020502", "UnitName": "台", "Quantity": 2, "OrigTaxPrice": "188.87", "OrigDiscountPrice": "188.87"}
    ]
  }
]
"""
    print(linkrun(sample))
