# ADR-0002: LinkPython 程序自包含（转换 + HTTP 调用 + 内部认证）

每个对接用友的 LinkPython 程序自包含地负责：接收 TotalLINK 订单 JSON、完成认证、数据转换、HTTP 调用，并返回统一结果契约 JSON。认证 Token 由程序内部管理（当前先用占位值，后续另开任务接入自动获取/刷新）。

选择自包含而非"认证外置"或"程序内直接落库"，因为这样程序可独立测试、可复用共享模块，且对 TotalLINK 侧透明——调用方只需传入业务 JSON，无需关心认证细节。落库统一由 TotalLINK 侧的 SQLNew（`--LINKEXECOUTPUT`）通过返回 JSON 完成，避免程序直接依赖 TotalLINK 数据库。
