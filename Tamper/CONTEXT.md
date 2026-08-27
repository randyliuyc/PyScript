# Tamper

TotalLINK 业务系统与用友 T+（进销存/财务）系统的接口对接域。TotalLINK 作为业务单据来源，通过 LinkPython 程序将订单等业务数据转换并推送到用友 T+，同时抓取用友维护的基础数据（往来单位、员工、科目等）。

## Language

**ExternalCode**:
TotalLINK 侧的业务单据号，作为推送到用友 T+ 的幂等键（对应用友 `saleOrder/Create` 的 `ExternalCode`）。
_Avoid_: 用友自动生成的单据号

**CustomerCode**:
客户在 TotalLINK 中的编码，与用友往来单位编码一致；推送时映射为用友 `dto.Customer.Code`。
_Avoid_: CustomerId

**InventoryCode**:
存货在 TotalLINK 中的编码，与用友存货档案编码一致；推送时映射为用友明细的 `Inventory.Code`。

**children**:
TotalLINK 输入中一张订单的明细行数组；推送时映射为用友 `SaleOrderDetails`。
_Avoid_: details, rows, lines

**UnitName**:
计量单位名称（如 "台"、"1000g装"）；推送时映射为用友明细的 `Unit.Name`。

**OrigTaxPrice**:
含税单价。TotalLINK 输入与用友 `saleOrder/Create` 明细字段同名，直接透传。

**LinkPython 程序**:
TotalLINK 通过 `CALLFUNCTION~LINKPYTHON~` 调用的自包含 Python 程序；接收订单数组 JSON，内部完成认证、数据转换、HTTP 调用，返回统一结果契约 JSON。

**统一返回契约**:
LinkPython 程序返回的标准 JSON：`isSuccess` / `message` 供界面展示，`results[]` 逐单记录推送结果（成功含用友单号，失败含错误原因）。
_Avoid_: 裸 results，不带 isSuccess/message
