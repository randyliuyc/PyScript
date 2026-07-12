---
name: TotalLINK-reimbursement-audit
slug: totallink-reimbursement
description:
  TotalLINK 报销单全流程审计 Skill。自动完成：查询报销单 → 下载附件识别发票 → 生成审计报告（Markdown）→ 生成 PDF → 发送邮件。
metadata:
  dependencies:
    - totallink-base
    - totallink-email
    - totallink-pdf
  workbuddy:
    env:
      TOTALLINK_AUTH_TOKEN: ""
      TOTALLINK_BASE_URL: "http://124.71.144.80:8088"
      SMTP_HOST: "smtp.163.com"
      SMTP_PORT: "465"
      SMTP_FROM: "lycurgus@163.com"
      SMTP_TO: "randy.liu@sagesoft.cn"
    note: "Token + SMTP 授权码需首次配置后持久化"
---

# TotalLINK 报销单审计

## 概述

从 TotalLINK 后端直接查询报销单数据，逐单下载附件并识别发票内容，生成合规审计报告（Markdown + PDF），自动发送邮件。所有 API 调用直连 TotalLINK 后端。

## 前置条件

- **TotalLINK 认证**：参照 [基础 Skill](../totallink-base/SKILL.md) 完成 `${TOTALLINK_AUTH_TOKEN}` 配置
- **API 调用**：遵循基础 Skill 的 Payload 格式和分页约定
- 邮件 SMTP 授权码：参照 [邮件发送 Skill](../shared/email-sender/SKILL.md)，首次使用时向用户索取，保存到 `~/.workbuddy/MEMORY.md`
- PDF 生成工具链：Pandoc CLI + WeasyPrint，参照 [PDF 生成 Skill](../shared/pdf-generator/SKILL.md)
- Python 环境：venv 下安装有 pdfplumber（`/Users/liuyongchao/.workbuddy/binaries/python/envs/default/bin/python3`）

---

## 本次场景所需工具

以下 4 个工具的 dmCode/dmNum 已固定，调用时直接使用：

| 工具 | dmCode | dmNum | 类型 | 用途 |
|------|--------|-------|------|------|
| 报销单列表 | LINKEXP01 | 9 | AIResult | 按日期范围查询 |
| 报销单表头信息 | LINKEXP01 | 10 | AIResult | 获取单张表头 |
| 报销单内容 | LINKEXP01 | 20 | AIResult | 获取费用明细 |
| 报销单附件列表 | LINKAI60 | 10 | AIResult | 获取附件 URL |

---

## Workflow

### Step 1：查询报销单列表

```
POST ${TOTALLINK_BASE_URL}/api/DataModel/linkDMAIResult

{
  "loginID": "${TOTALLINK_AUTH_TOKEN}",
  "par": {
    "dmCode": "LINKEXP01",
    "dmNum": 9,
    "Para": ["", "2026-06-01", "2026-07-11", ""]
  }
}
```

`Para` 按位置传入：`["搜索内容(可空)", "开始日期", "结束日期", "状态(可空)"]`。
默认时间范围：上月至今（用户可指定起止日期）。

**返回数据：**

```json
{
  "isSuccess": "true",
  "data": {
    "Table": {
      "schema": ["DOCNUM", "AMTTOT", "DOCTYP", "EXPDAT", "DOCSTA", "RESNO", "REMARK", "..."],
      "data": [
        ["EXP260600009", "300.00", "招待费", "2026-06-23", "已驳回", "RANDY.LIU", "..."],
        ["EXP260600010", "606.00", "招待费", "2026-06-26", "待处理", "Oscar", "..."]
      ]
    }
  },
  "pagination": { "current_page": 1, "total_pages": 1, "total_items": 4 }
}
```

翻页：`pagination.total_pages > 1` 时，用户按需传入 `page` 参数。

---

### Step 2：获取每单详细信息

对每张报销单并行调用（提升效率）：

**表头信息：**

```
POST ${TOTALLINK_BASE_URL}/api/DataModel/linkDMAIResult

{
  "loginID": "${TOTALLINK_AUTH_TOKEN}",
  "par": { "dmCode": "LINKEXP01", "dmNum": 10, "Para": ["EXP260600009"] }
}
```

**费用明细：**

```
POST ${TOTALLINK_BASE_URL}/api/DataModel/linkDMAIResult

{
  "loginID": "${TOTALLINK_AUTH_TOKEN}",
  "par": { "dmCode": "LINKEXP01", "dmNum": 20, "Para": ["EXP260600009"] }
}
```

**附件列表：**

```
POST ${TOTALLINK_BASE_URL}/api/DataModel/linkDMAIResult

{
  "loginID": "${TOTALLINK_AUTH_TOKEN}",
  "par": { "dmCode": "LINKAI60", "dmNum": 10, "Para": ["EXP260600009"] }
}
```

---

### Step 3：下载并识别发票

下载附件（PDF 或 JPG），提取发票关键信息：

**下载附件：**

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

**PDF 发票 → 读取文本（pdfplumber）：**

```bash
/Users/liuyongchao/.workbuddy/binaries/python/envs/default/bin/python3 -c "
import pdfplumber
with pdfplumber.open('path.pdf') as pdf:
    for page in pdf.pages:
        print(page.extract_text())
"
```

**JPG 图片发票 → 使用 Read 工具（多模态）直接读取。**

提取关键字段：发票号码、开票日期、购买方名称、项目名称、金额、税额、价税合计、出行人信息。

---

### Step 4：核对分析

逐单对比报销单与发票信息：

| 核对项 | 检查点 |
|-------|--------|
| 金额一致性 | 发票价税合计 vs 报销单金额 |
| 发票时效 | 发票日期是否在有效期内（一般 3~6 个月）|
| 费用归类 | 发票实际内容与报销单费用类型是否匹配 |
| 购买方抬头 | 发票上的购买方是否属于可报销公司 |
| 附件合规 | 是否有附件、附件是否真实发票 |

---

### Step 5：生成审计报告（Markdown）

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

**重要规则：**
- 日期用 `YYYY-MM-DD` 格式
- 金额用 `¥` 前缀
- 不要使用 emoji
- 使用 `---` 分隔章节
- 报告末尾必须加上声明：`本报告由 TotalLINK AI 助手自动生成，仅供参考，最终审批以人工审核为准。`

**Pandoc 安全规则（参照 PDF 生成 Skill）：**
- 禁止 `[X]` `[!]` `[OK]` → 使用 `X` `!` `OK`
- 禁止 `\*文本\*` → 使用 `*文本*`
- 每个表格前后必须有空行

---

### Step 6：生成 PDF + 发送邮件

参照 [PDF 生成 Skill](../shared/pdf-generator/SKILL.md) 和 [邮件发送 Skill](../shared/email-sender/SKILL.md)。

**生成 PDF：**

```bash
cd /path/to/working/dir

cat > temp-style.css << 'CSS_EOF'
@page { size: A4; margin: 2cm 2.5cm; }
body { font-family: -apple-system, 'PingFang SC', 'STHeiti', 'Microsoft YaHei', sans-serif; font-size: 11pt; line-height: 1.7; color: #222; }
h1 { font-size: 18pt; color: #1a1a2e; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px; margin-top: 1.2em; }
h2 { font-size: 14pt; color: #16213e; margin-top: 1em; }
h3 { font-size: 12pt; color: #0f3460; margin-top: 0.8em; }
table { border-collapse: collapse; width: 100%; margin: 0.8em 0; font-size: 9.5pt; }
th { background-color: #e8e8e8; border: 1px solid #999; padding: 5px 8px; text-align: center; font-weight: bold; }
td { border: 1px solid #999; padding: 4px 8px; }
td strong { color: #c0392b; }
code { font-family: 'SF Mono', 'Menlo', monospace; font-size: 9pt; background: #f0f0f5; padding: 1px 4px; border-radius: 3px; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }
CSS_EOF

pandoc 审计报告.md -o temp_report.html --embed-resources --standalone

/Users/liuyongchao/.workbuddy/binaries/python/envs/default/bin/python3 -c "
from weasyprint import HTML
HTML('temp_report.html').write_pdf('审计报告.pdf', stylesheets=['temp-style.css'])
print('PDF generated successfully')
"

rm -f temp_report.html temp-style.css
```

**发送邮件（仅 PDF 附件，自动发送无需确认）：**

```python
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

from_addr = "lycurgus@163.com"
to_addr = "randy.liu@sagesoft.cn"
password = "<从 ~/.workbuddy/MEMORY.md 读取>"

msg = MIMEMultipart()
msg["From"] = from_addr
msg["To"] = to_addr
msg["Subject"] = "本周报销单审计分析报告"
msg.attach(MIMEText("报销单审计报告见附件，请查收。", "plain", "utf-8"))

with open("审计报告.pdf", "rb") as f:
    attachment = MIMEBase("application", "octet-stream")
    attachment.set_payload(f.read())
    encoders.encode_base64(attachment)
    attachment.add_header("Content-Disposition", "attachment", filename=("utf-8", "", "审计报告.pdf"))
    msg.attach(attachment)

with smtplib.SMTP_SSL("smtp.163.com", 465, timeout=30) as server:
    server.login(from_addr, password)
    server.send_message(msg)

print("邮件已自动发送至 randy.liu@sagesoft.cn")
```

---

## 关键注意事项

1. **认证**：所有请求的 `loginID` 使用 `${TOTALLINK_AUTH_TOKEN}`
2. **dmCode/dmNum**：已硬编码，无需工具发现步骤
3. **Para 数组**：空位传 `""`（空字符串），不是 `null`/`undefined`
4. **并行调用**：Step 2 中多张报销单的详情可并行请求
5. **附件 URL**：含中文时需对路径部分做 URL-encode
6. **JPG vs PDF**：JPG 用多模态直接读取，PDF 用 pdfplumber
7. **所有 Python 命令**：通过 venv 执行
8. **翻页不自动**：用户未明确要求时，不自动翻页
9. **SMTP 授权码**：首次向用户索取，保存到 `~/.workbuddy/MEMORY.md` 复用
10. **收件人固定**：`randy.liu@sagesoft.cn`，无需询问用户，直接发送

## Resources

### references/
- [TotalLINK 基础 Skill](../totallink-base/SKILL.md) — 认证管理、API 格式
- [邮件发送 Skill](../shared/email-sender/SKILL.md) — SMTP 发送规范
- [PDF 生成 Skill](../shared/pdf-generator/SKILL.md) — Pandoc + WeasyPrint 工具链
