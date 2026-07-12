---
name: totallink-base
slug: totallink
description:
  TotalLINK 数据分析平台基础 Skill，提供认证管理、动态工具发现和通用 API 调用能力。
  所有 TotalLINK 场景化 Skill（报销审核、库存管理、客户分析等）均依赖本基础 Skill。
metadata:
  workbuddy:
    env:
      TOTALLINK_AUTH_TOKEN: ""
      TOTALLINK_BASE_URL: "http://124.71.144.80:8088"
    note: "Token 首次使用时由用户提供，持久化到 ~/.totallink/config.json，后续自动读取"
---

# TotalLINK 基础 Skill

## 认证管理

### 首次使用

用户提供 TotalLINK 授权令牌，持久化到 `~/.totallink/config.json`：

```json
{
  "auth_token": "tlk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "base_url": "http://124.71.144.80:8088"
}
```

### 后续使用

每次调用时从配置文件读取令牌，注入为环境变量 `${TOTALLINK_AUTH_TOKEN}`：

```bash
export TOTALLINK_AUTH_TOKEN=$(cat ~/.totallink/config.json | python3 -c "import sys,json; print(json.load(sys.stdin)['auth_token'])")
export TOTALLINK_BASE_URL="http://124.71.144.80:8088"
```

### 令牌续期

检测到认证失败（`HTTP 401/403` 或 `isSuccess: "false"`）时，提示用户检查令牌并重新提供。

---

## 工具发现（可选）

仅在需要探索未知工具时使用。已知工具应直接硬编码 `dmCode`/`dmNum`，跳过此步。

```
POST ${TOTALLINK_BASE_URL}/api/DataModel/linkDMAIResult

{
  "loginID": "${TOTALLINK_AUTH_TOKEN}",
  "par": {
    "dmCode": "SEARCHLIST",
    "dmNum": 100,
    "Para": []
  }
}
```

返回的 `data.Table` 包含工具列表，关键字段：
- `TOOL_CODE` → dmCode
- `TOOL_NUM` → dmNum
- `TOOL_NAME` → 工具名称
- `TOOL_DESC` → 描述（含参数说明）
- `TOOL_TYPE` → AIResult / AIRowSubmit / AIDataSubmit

---

## API 调用模板

### 通用约定

- 统一响应：`{ isSuccess, data, message }`，`isSuccess` 为字符串 `"true"`/`"false"`
- `Para` 为字符串数组，空位传 `""`，不传 `null`/`undefined`
- `data.Table` 格式：`{ schema: ["字段名", ...], data: [["值", ...]] }`
- 超时：连接 5s，读写 30s
- Content-Type: `application/json`

---

### AIResult — 数据查询（默认分页）

```
POST ${TOTALLINK_BASE_URL}/api/DataModel/linkDMAIResult

{
  "loginID": "${TOTALLINK_AUTH_TOKEN}",
  "par": {
    "dmCode": "<dmCode>",
    "dmNum": <dmNum>,
    "Para": ["参数1", "参数2", "..."]
  }
}
```

**分页规则：**
- 默认 20 条/页，自动分页，返回 `pagination` 对象
- 用户未明确要求翻页时，不自动请求后续页
- 翻页参数 `page`（从 1 开始）、`page_size`（最大 50）

---

### AIRowSubmit — 行数据提交

```
POST ${TOTALLINK_BASE_URL}/api/DataModel/linkDMAIRowSubmit

{
  "loginID": "${TOTALLINK_AUTH_TOKEN}",
  "par": {
    "dm": {
      "dmCode": "<dmCode>",
      "dmNum": <dmNum>,
      "Para": ["参数1", "..."]
    },
    "scriptType": <操作类型>,
    "rowData": { "字段": "值" }
  }
}
```

`scriptType` 从工具描述中获取（格式 `script_type: N`）。

---

### AIDataSubmit — 批量数据提交

```
POST ${TOTALLINK_BASE_URL}/api/DataModel/linkDMAIDataSubmit

{
  "loginID": "${TOTALLINK_AUTH_TOKEN}",
  "par": {
    "dm": {
      "dmCode": "<dmCode>",
      "dmNum": <dmNum>,
      "Para": ["参数1", "..."]
    },
    "scriptType": <操作类型>,
    "rowData": { "字段": "值" },
    "tableData": [{ "字段1": "值1" }, { "字段1": "值2" }]
  }
}
```

---

## 场景化 Skill 引用方式

场景化 Skill 在文档开头声明依赖，不重复描述认证和 API 格式：

```markdown
## 前置条件

- **TotalLINK 认证**：参照基础 Skill 完成 `${TOTALLINK_AUTH_TOKEN}` 配置
- **API 调用**：遵循基础 Skill 的 Payload 格式和分页约定
```

---

## 错误处理

- `isSuccess: "false"` → 检查 `message` 字段
- `HTTP 401/403` → 令牌无效，提示用户重新提供
- `HTTP 5xx` → 后端异常，稍后重试
