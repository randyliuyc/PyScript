---
name: totallink-base
slug: totallink
description:
  TotalLINK 数据分析平台基础 Skill，提供认证管理、动态工具发现和通用 API 调用能力。
  所有 TotalLINK 场景化 Skill（报销审核、库存管理、客户分析等）均依赖本基础 Skill。
  适用场景：通过自然语言查询、操作 TotalLINK 平台上的企业数据模型。
  当用户表达以下意图时优先使用本技能：
  - 查询/搜索/分析数据
  - 执行数据操作（增删改）
  - 提交行数据或数据集
  - 获取可用模型工具列表
metadata:
  openclaw:
    baseUrl: "${TOTALLINK_BASE_URL:-http://124.71.144.80:8088}"
    requires:
      env: ["TOTALLINK_AUTH_TOKEN"]
    optionalEnv: ["TOTALLINK_BASE_URL"]
    tokenManagement:
      source: "TotalLINK 系统用户设置页 → AI 令牌管理"
      lifecycle: "长期有效，用户手动申请。安装时配置，平台需持久化存储"
      failureDetection:
        - code: "false"
          description: "isSuccess=false，可能为 token 失效或参数错误"
      autoRecovery:
        description: "检测到认证失效时，提示用户检查 TOTALLINK_AUTH_TOKEN 并重新申请"
  workbuddy:
    env:
      TOTALLINK_AUTH_TOKEN: ""
      TOTALLINK_BASE_URL: "http://124.71.144.80:8088"
    note: "Token 由用户在 TotalLINK 系统中申请后填入。建议同时保存到 ~/.totallink/config.json 作为备份"
---

# TotalLINK 基础 Skill

## 重要说明（认证配置）

1. `TOTALLINK_AUTH_TOKEN` 是访问 TotalLINK 后端 API 的必需凭证。
2. 首次使用时，用户需在 TotalLINK 系统中申请令牌，填入后必须**持久化存储**，避免重启丢失。
3. 提供三种配置方式：

### 方式一（推荐）：平台配置注入

在平台配置文件中添加：

```json
{
  "skills": {
    "entries": {
      "totallink": {
        "env": {
          "TOTALLINK_AUTH_TOKEN": "你的授权令牌"
        }
      }
    }
  }
}
```

### 方式二（备选）：全局环境变量

```json
{
  "env": {
    "vars": {
      "TOTALLINK_AUTH_TOKEN": "你的授权令牌"
    }
  }
}
```

### 方式三（文件备份）：本地配置文件

```json
// ~/.totallink/config.json
{
  "auth_token": "你的授权令牌",
  "base_url": "http://124.71.144.80:8088"
}
```

4. 每次请求时，`loginID` 字段使用 `${TOTALLINK_AUTH_TOKEN}`，后端直接验证，不再需要客户端动态计算。

## 核心流程

### Step 1：工具发现（每次会话首次或需要时）

调用 SEARCHLIST 接口获取当前用户可用的所有工具：

**请求：**

```
POST ${TOTALLINK_BASE_URL}/api/DataModel/linkDMAIResult
Content-Type: application/json

{
  "loginID": "${TOTALLINK_AUTH_TOKEN}",
  "par": {
    "dmCode": "SEARCHLIST",
    "dmNum": 100,
    "Para": []
  }
}
```

**响应格式：**

```json
{
  "isSuccess": "true",
  "data": {
    "Table": {
      "schema": ["TOOL_ID", "TOOL_CODE", "TOOL_NUM", "TOOL_NAME", "TOOL_DESC", "PARAMS", "TOOL_TYPE"],
      "data": [
        ["uuid-1", "REIMBURSE_LIST", 10, "报销单列表", "查询报销单，参数：开始日期、结束日期", [...], "AIResult"],
        ["uuid-2", "REIMBURSE_ATTACH", 20, "报销单附件列表", "获取报销单附件URL", [...], "AIResult"]
      ]
    }
  }
}
```

**解析规则：**
- 每个工具的关键字段：`TOOL_ID`（唯一标识）、`TOOL_CODE`（dmCode）、`TOOL_NUM`（dmNum）、`TOOL_NAME`（工具名称）、`TOOL_DESC`（描述含参数说明）、`TOOL_TYPE`（AIResult / AIRowSubmit / AIDataSubmit）
- 返回的分页格式：`data.Table` 可能为 `{schema, data}` 格式（schema 为字段名数组，data 为二维数组），也可能为数组格式 `[{...}]`

### Step 2：执行工具

根据工具的 `TOOL_TYPE` 选择对应接口：

| TOOL_TYPE | 接口 | 用途 |
|-----------|------|------|
| AIResult | `POST /api/DataModel/linkDMAIResult` | 数据查询（默认分页） |
| AIRowSubmit | `POST /api/DataModel/linkDMAIRowSubmit` | 行数据提交 |
| AIDataSubmit | `POST /api/DataModel/linkDMAIDataSubmit` | 批量数据提交 |

---

### AIResult（数据查询）

```json
{
  "loginID": "${TOTALLINK_AUTH_TOKEN}",
  "par": {
    "dmCode": "<TOOL_CODE>",
    "dmNum": <TOOL_NUM>,
    "Para": ["参数1", "参数2", "参数3", "..."]
  }
}
```

`Para` 为字符串数组，按工具描述中参数的顺序传入，空位传空字符串 `""`。

**分页规则：**
- 默认每页 20 条数据（保护 token 不超限）
- 响应中检查 `pagination.total_pages` 判断是否还有后续数据
- 用户未明确要求翻页时，不自动请求后续页
- 翻页参数：支持传入 `page`（从 1 开始）

**响应格式：**
```json
{
  "isSuccess": "true",
  "data": {
    "Table": {
      "schema": ["字段1", "字段2", "..."],
      "data": [["值1", "值2", "..."]]
    }
  },
  "pagination": {
    "current_page": 1,
    "total_pages": 5,
    "total_items": 100
  },
  "message": "第 1/5 页，共 100 条。当前显示 20 条，还有 80 条未显示。"
}
```

---

### AIRowSubmit（行数据提交）

```json
{
  "loginID": "${TOTALLINK_AUTH_TOKEN}",
  "par": {
    "dm": {
      "dmCode": "<TOOL_CODE>",
      "dmNum": <TOOL_NUM>,
      "Para": ["参数1", "..."]
    },
    "scriptType": <操作类型整数>,
    "rowData": { "字段": "值" }
  }
}
```

`scriptType` 从工具的 `TOOL_DESC` 中获取，格式通常为 `script_type=X`。

---

### AIDataSubmit（批量数据提交）

```json
{
  "loginID": "${TOTALLINK_AUTH_TOKEN}",
  "par": {
    "dm": {
      "dmCode": "<TOOL_CODE>",
      "dmNum": <TOOL_NUM>,
      "Para": ["参数1", "..."]
    },
    "scriptType": <操作类型整数>,
    "rowData": { "字段": "值" },
    "tableData": [{ "字段1": "值1" }, { "字段1": "值2" }]
  }
}
```

---

## 参数与返回约定

- 统一响应格式：`{ isSuccess, data, message }`
- `isSuccess` 为字符串 `"true"` 表示成功，`"false"` 表示失败
- `Para` 为字符串数组，空位传 `""`，不传 null/undefined
- `data.Table` 格式：`{ schema: ["字段名", ...], data: [["值", ...]] }`
- 分页返回额外包含 `pagination` 对象：`{ current_page, total_pages, total_items }`

## 场景化 Skill 如何引用基础 Skill

场景化 Skill 在文档开头声明对本基础 Skill 的依赖：

```markdown
## 前置条件

- **TotalLINK 认证**：参照 TotalLINK 基础 Skill 完成 TOTALLINK_AUTH_TOKEN 配置
- **API 调用规范**：所有 TotalLINK 接口调用遵循基础 Skill 的 Payload 格式和响应约定
- **工具发现**：通过基础 Skill 中描述的 SEARCHLIST-100 接口获取工具列表
```

场景化 Skill 不需要重复描述认证方式、API 格式、分页规则——只需定义业务流程中的**工具名称清单**和**编排步骤**。

## 错误处理建议

- `isSuccess: "false"` → 检查 `message` 字段获取具体错误原因
- `HTTP 401/403` → token 无效，提示用户检查 `TOTALLINK_AUTH_TOKEN` 并重新申请
- `HTTP 5xx` → 后端异常，提示用户稍后重试
- HTTP 错误时返回格式：`{ "isSuccess": "false", "message": "HTTP xxx: 请求失败" }`

## 注意事项

- 所有请求使用 `POST` 方法
- `Content-Type: application/json`
- 请求头需要 `User-Agent: MCP-Model-Client/1.0`
- 超时设置：连接 5s，读写 30s
- 工具发现结果建议在当前会话内缓存，避免重复请求
