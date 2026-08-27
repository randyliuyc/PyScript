# ADR-0001: 每个用友接口一个 LinkPython 程序文件 + 共享 chanjet_client 模块

为对接用友 T+，每个业务接口实现为一个独立的 LinkPython 程序文件（如 `sale_order.py`），把认证与 HTTP 调用抽成共享的 `chanjet_client.py`。

选择每接口一文件而非单文件接口路由（用一个 `apiName` 参数分发），因为接口各自业务语义差异大，独立文件便于单独测试、部署和错误隔离；共享模块避免认证/HTTP 代码重复。
