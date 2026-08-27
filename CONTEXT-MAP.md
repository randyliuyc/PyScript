# Context Map

## Contexts

- [BROTEX](./BROTEX/CONTEXT.md) - 牵伸计算、混纺配色等纺织工艺算法程序
- [Tamper](./Tamper/CONTEXT.md) - TotalLINK 业务系统与用友 T+ 财务/进销存系统的接口对接（LinkPython 程序）
- [mcp-server](./mcp-server/CONTEXT.md) - MCP Server（工具调用）

## Relationships

- **Tamper -> BROTEX**: Tamper 的 TotalLINK 模型可通过 `CALLFUNCTION~LINKPYTHON~` 调用 BROTEX 下的算法程序（如 bt8.py）完成本地计算
- **Tamper -> mcp-server**: 两者独立；Tamper 面向用友 T+ 业务单据对接，mcp-server 面向通用工具调用
