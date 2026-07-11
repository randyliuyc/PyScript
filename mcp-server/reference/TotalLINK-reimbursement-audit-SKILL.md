---
name: TotalLINK-reimbursement-audit
description: TotalLINK 报销单全流程审计。自动完成：查询报销单 → 下载附件识别发票 → 生成审计报告(Markdown) → 生成PDF → 发送邮件。适用于用户要求"审计报销单"、"检查报销单"、"报销单审核"、"报销单分析"等场景。
agent_created: true
---

# TotalLINK 报销单审计

## Overview

从 TotalLINK MCP 查询 TotalLINK 系统的报销单数据，逐单下载附件、识别发票内容，生成合规审计报告（Markdown 和 PDF），并通过 SMTP 自动发送到 randy.liu@sagesoft.cn（无需确认）。

## 前置条件

- TotalLINK MCP 连接器已连接（需要时通过 WorkBuddy UI 连接）
- TotalLINK MCP 调用时 userid 固定为 `randy.liu`
- 邮箱 SMTP 授权码需用户提供一次，保存后复用（保存在 ~/.workbuddy/MEMORY.md）
- Python 环境（venv，安装有 pdfplumber, requests, weasyprint）
- Pandoc CLI、weasyprint（均已安装）

## MCP 工具调用规范（重要）

所有 TotalLINK 工具通过 `mcp__TotalLINK__call_dynamic_tool` 调用。**必须严格遵守以下类型要求**，否则会反复报参数验证错误：

| 参数 | 类型 | 说明 |
|------|------|------|
| `tool_name` | string | 工具名称，如 `"报销单列表"`、`"报销单附件列表"` |
| `parameters` | **array of string** | 参数数组，如 `["", "2026-06-14", "2026-06-21", ""]`。**不是对象，不是 JSON 字符串** |
| `userid` | string | 固定为 `"randy.liu"` |
| `page` | **integer** | 页码，如 `1`。**不是字符串 `"1"`** |
| `page_size` | **integer** | 每页条数，如 `20`。**不是字符串 `"20"`** |

**正确调用示例：**

```json
{
  "tool_name": "报销单列表",
  "parameters": ["", "2026-06-14", "2026-06-21", ""],
  "userid": "randy.liu",
  "page": 1,
  "page_size": 20
}
```

**常见错误（务必避免）：**
- `parameters` 传成对象 `{"item": [...]}` → 必须是数组 `["", "2026-06-14", ...]`
- `parameters` 传成 JSON 字符串 `"[\"\", \"2026-06-14\"]"` → 必须是原生数组
- `page` / `page_size` 传成字符串 `"1"` / `"20"` → 必须是整数 `1` / `20`

**调用流程：**
1. 先用 `mcp__TotalLINK__get_tools` 获取可用工具列表（可选，首次调用时确认工具名）
2. 用 `mcp__TotalLINK__call_dynamic_tool` 调用具体工具，按上述格式传参

## Workflow

### Step 1: 查询报销单

使用 `call_dynamic_tool` 调用 `报销单列表` 工具：

```json
{
  "tool_name": "报销单列表",
  "parameters": ["", "开始日期", "结束日期", ""],
  "userid": "randy.liu",
  "page": 1,
  "page_size": 20
}
```

根据用户指定的时间范围（本周/本月/自定义）设置起止日期。注意默认开始日期为上个月，结束日期为今日。

### Step 2: 获取附件列表

对每张报销单，调用 `报销单附件列表` 工具获取附件地址（可并行调用多张）：

```json
{
  "tool_name": "报销单附件列表",
  "parameters": ["单据编号"],
  "userid": "randy.liu",
  "page": 1,
  "page_size": 20
}
```

同理，`报销单内容` 和 `报销单表头信息` 工具也通过 `call_dynamic_tool` 调用，`parameters` 传单据编号数组，`page`/`page_size` 传整数。

### Step 3: 下载并识别发票

下载附件（PDF 或 JPG），使用 Python 脚本提取发票关键信息：

**下载附件（requests）：**
```python
import requests
from urllib.parse import quote
parts = url.split('/', 3)
encoded = quote(parts[3])
encoded_url = parts[0] + '//' + parts[2] + '/' + encoded
r = requests.get(encoded_url, timeout=30)
with open(local_path, 'wb') as f:
    f.write(r.content)
```

**PDF 发票 → 读取文本（使用 venv 下的 pdfplumber）：**
```bash
/Users/liuyongchao/.workbuddy/binaries/python/envs/default/bin/python3 -c "
import pdfplumber
with pdfplumber.open('path.pdf') as pdf:
    for page in pdf.pages:
        print(page.extract_text())
"
```

**JPG 图片发票 → 使用 Read 工具（多模态）直接读取图片。**

提取的关键字段：
- 发票号码、开票日期
- 购买方名称（核对报销公司抬头）
- 项目名称、金额、税额、价税合计
- 出行人信息（如适用）

### Step 4: 核对分析

逐单对比报销单与发票信息：

| 核对项 | 检查点 |
|-------|--------|
| 金额一致性 | 发票价税合计 vs 报销单金额 |
| 发票时效 | 发票日期是否在有效期内（一般 3~6 个月）|
| 费用归类 | 发票实际内容与报销单费用类型是否匹配 |
| 购买方抬头 | 发票上的购买方是否属于可报销公司 |
| 附件合规 | 是否有附件、附件是否真实发票 |

### Step 5: 生成审计报告（Markdown）

按以下结构生成 Markdown 审计报告：

```markdown
# 本周报销单审计分析报告

## 一、总体概览
（汇总表：单号、金额、类型、状态）

## 二、各单明细审计
### 1. 单号 — 费用类型 ¥金额
（详细信息表 + 问题列表）

## 三、共性问题汇总
（发票日期问题/金额问题/归类问题/抬头问题）

## 四、风险评级
（风险表 + 建议处理）

## 五、审计结论
```

**注意：**
- 日期用 `YYYY-MM-DD` 格式
- 金额用 `¥` 前缀
- 不要使用 emoji（❌⚠️✅🔴🟡）
- 使用 `---` 分隔章节
- 报告末尾必须加上声明：`本报告由 TotalLINK AI 助手自动生成，仅供参考，最终审批以人工审核为准。`

**关键：避免使用 pandoc 会误解析的标记：**
- ~~`[X]`~~ → 使用 `X`（不要加方括号，否则 pandoc 会转成任务列表 checkbox！）
- ~~`[!]`~~ → 使用 `!`（不要加方括号）
- ~~`[OK]`~~ → 使用 `OK`（不要加方括号）
- ~~`\*文本\*`~~ → 使用 `*文本*`（反斜杠转义星号在表格内会干扰 pandoc 解析）
- 每个表格**前后必须有空行**（pandoc pipe table 要求前有空行才识别为表格）

### Step 6: 生成 PDF（pandoc + weasyprint）

将 Step 5 生成的 Markdown 报告转换为 PDF，流程如下：

**Step 6.1: 准备 CSS 样式文件**

写入 CSS 文件为 PDF 排版（需写入临时文件，因为 weasyprint Python API 的 `stylesheets` 参数接受的是文件路径，不是 CSS 字符串）：

```css
/* temp-style.css */
@page {
  size: A4;
  margin: 2cm 2.5cm;
}
body {
  font-family: -apple-system, 'PingFang SC', 'STHeiti', 'Microsoft YaHei', sans-serif;
  font-size: 11pt;
  line-height: 1.7;
  color: #222;
}
h1 { font-size: 18pt; color: #1a1a2e; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px; margin-top: 1.2em; }
h2 { font-size: 14pt; color: #16213e; margin-top: 1em; }
h3 { font-size: 12pt; color: #0f3460; margin-top: 0.8em; }
table { border-collapse: collapse; width: 100%; margin: 0.8em 0; font-size: 9.5pt; }
th { background-color: #e8e8e8; border: 1px solid #999; padding: 5px 8px; text-align: center; font-weight: bold; }
td { border: 1px solid #999; padding: 4px 8px; }
td strong { color: #c0392b; }
code { font-family: 'SF Mono', 'Menlo', monospace; font-size: 9pt; background: #f0f0f5; padding: 1px 4px; border-radius: 3px; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }
```

**Step 6.2: 转换并生成 PDF**

```bash
cd /path/to/working/dir

# Step 6.2a: Pandoc 将 Markdown 转换为 HTML
# --embed-resources --standalone 使 HTML 自包含（嵌入图片/CSS/JS）
pandoc 审计报告.md -o temp_report.html --embed-resources --standalone

# Step 6.2b: WeasyPrint 将 HTML 渲染为 PDF
/Users/liuyongchao/.workbuddy/binaries/python/envs/default/bin/python3 -c "
from weasyprint import HTML
HTML('temp_report.html').write_pdf('审计报告.pdf', stylesheets=['temp-style.css'])
print('PDF generated successfully')
"

# 清理临时文件
rm -f temp_report.html temp-style.css
```

**转换流程说明：**
```
审计报告.md  ──[pandoc]──>  temp_report.html  ──[weasyprint + CSS]──>  审计报告.pdf
```

**已知问题与规避：**
- Pandoc 解析 markdown 表格时，若表格前无空行，会当做普通文本而非表格 → **每个表格前必须有空行**
- Pandoc 会将 `[X]`（即使是表格内）转为任务列表 checkbox → **禁止在表格中使用 `[X]`**
- WeasyPrint Python API 的 `stylesheets` 参数接受文件路径而非 CSS 字符串 → **CSS 必须写入文件后再传入**

### Step 7: 发送邮件（仅 PDF 附件，自动发送无需确认）

使用 SMTP 通过网易邮箱自动发送，收件人固定为 `randy.liu@sagesoft.cn`，无需询问用户，直接发送。

**配置（保存在 ~/.workbuddy/MEMORY.md 中复用）：**
- SMTP 服务器：`smtp.163.com`
- SSL 端口：465
- 发件人：`lycurgus@163.com`
- 收件人：`randy.liu@sagesoft.cn`（固定，不询问）
- 授权码：首次使用时向用户索取，然后保存到 ~/.workbuddy/MEMORY.md

**发送脚本（smtplib，仅 PDF 附件）：**
```python
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

from_addr = "lycurgus@163.com"
to_addr = "randy.liu@sagesoft.cn"
password = "<from_memory>"

msg = MIMEMultipart()
msg["From"] = from_addr
msg["To"] = to_addr
msg["Subject"] = "本周报销单审计分析报告"
msg.attach(MIMEText(body, "plain", "utf-8"))

# 仅附加 PDF 文件，不附带 Markdown
with open(pdf_path, "rb") as f:
    attachment = MIMEBase("application", "octet-stream")
    attachment.set_payload(f.read())
    encoders.encode_base64(attachment)
    filename = "审计报告.pdf"
    attachment.add_header("Content-Disposition", "attachment",
                          filename=("utf-8", "", filename))
    msg.attach(attachment)

with smtplib.SMTP_SSL("smtp.163.com", 465, timeout=30) as server:
    server.login(from_addr, password)
    server.send_message(msg)
print("邮件已自动发送至 randy.liu@sagesoft.cn")
```

## 注意事项

- TotalLINK MCP 调用时必须带 `userid: "randy.liu"`，否则返回空列表
- **MCP 调用类型要求**：`parameters` 必须是数组（array），`page`/`page_size` 必须是整数（integer），详见上方"MCP 工具调用规范"
- 附件 URL 含中文，下载前需对路径部分做 URL-encode
- JPG 发票附件直接用 Read 工具查看（多模态识别），PDF 附件用 pdfplumber 提取文本
- 所有 Python 命令通过 venv 执行：`/Users/liuyongchao/.workbuddy/binaries/python/envs/default/bin/python3`
- 临时文件写在当前工作目录下；最终产出（Markdown + PDF）放在当前工作目录
- **PDF 转换关键陷阱**：详见 Step 6.3 的"已知问题与规避"章节
- SMTP 授权码首次使用时向用户索取，保存到 `~/.workbuddy/MEMORY.md` 供后续复用

## Resources

### scripts/
（无）

### references/
（无）

### assets/
（无）
