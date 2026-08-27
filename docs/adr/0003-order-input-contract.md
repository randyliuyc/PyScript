# ADR-0003: 订单输入契约——字段名与用友一致，但扁平结构映射为嵌套结构

TotalLINK 传给 LinkPython 的订单输入 JSON，字段名尽量与用友接口一致（`ExternalCode`、`VoucherDate`、`Quantity`、`OrigTaxPrice` 等直接透传），但结构上是扁平的（`CustomerCode`、`children[]`、`children[].InventoryCode`、`children[].UnitName`），LinkPython 负责把它组装成用友接口的嵌套结构（`dto.Customer.Code`、`dto.SaleOrderDetails[].Inventory.Code`、`Unit.Name`）。

选择"字段名一致 + 结构扁平"而非让 TotalLINK 直接拼好用友的完整嵌套 dto，是因为 TotalLINK 侧的 SQL 更自然地按业务字段产出扁平结果，而字段名保持一致可最大限度减少命名映射、避免歧义。结构重组是 LinkPython 里唯一必要的转换工作。
