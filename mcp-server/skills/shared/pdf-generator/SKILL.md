---
name: totallink-pdf-generator
slug: totallink-pdf
description:
  TotalLINK PDF 生成公共 Skill，提供 Markdown → HTML → PDF 的通用转换能力。
  使用 Pandoc + WeasyPrint 工具链，支持中文排版。被报销审核等场景化 Skill 引用。
metadata:
  workbuddy:
    note: "依赖 Pandoc CLI 和 WeasyPrint Python 库。Python 使用 workbuddy 的 venv"
---

# PDF 生成 Skill

## 概述

将 Markdown 报告转换为排版精美的 A4 PDF 文件。使用两步转换流程：

```
报告.md ──[pandoc]──→ temp_report.html ──[weasyprint + CSS]──→ 报告.pdf
```

## 前置条件

- Pandoc CLI 已安装
- WeasyPrint 已安装：`/Users/liuyongchao/.workbuddy/binaries/python/envs/default/bin/python3 -c "import weasyprint"`
- 中文字体可用：PingFang SC / STHeiti / Microsoft YaHei

## CSS 样式模板

写入临时文件 `temp-style.css`（因为 WeasyPrint Python API 的 stylesheets 参数接受文件路径）：

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

## 转换命令

```bash
cd /path/to/working/dir

# Step 1: Pandoc 将 Markdown 转换为 HTML
# --embed-resources --standalone 使 HTML 自包含
pandoc 报告.md -o temp_report.html --embed-resources --standalone

# Step 2: WeasyPrint 将 HTML 渲染为 PDF
/path/to/venv/bin/python3 -c "
from weasyprint import HTML
HTML('temp_report.html').write_pdf('报告.pdf', stylesheets=['temp-style.css'])
print('PDF 生成成功')
"

# Step 3: 清理临时文件
rm -f temp_report.html temp-style.css
```

## Markdown 编写规范（防止 Pandoc 解析异常）

编写 Markdown 报告时务必遵守以下规则，否则 Pandoc 转换时会出错：

| 规则 | 禁止 | 使用 |
|------|------|------|
| 任务列表 | ~~`[X]` `[!]` `[OK]`~~ | `X` `!` `OK`（不加方括号） |
| 星号转义 | ~~`\*文本\*`~~ | `*文本*`（表格内不要转义星号） |
| 表格前空行 | 表格紧接上文 | **表格前后必须有空行** |

## 调用方式

场景化 Skill 中直接执行 Shell 命令：

```bash
# 1. 写入 CSS 文件
cat > temp-style.css << 'CSS_EOF'
...CSS内容...
CSS_EOF

# 2. Pandoc 转换
pandoc 审计报告.md -o temp_report.html --embed-resources --standalone

# 3. WeasyPrint 生成 PDF
/Users/liuyongchao/.workbuddy/binaries/python/envs/default/bin/python3 -c "
from weasyprint import HTML
HTML('temp_report.html').write_pdf('审计报告.pdf', stylesheets=['temp-style.css'])
print('PDF generated')
"

# 4. 清理
rm -f temp_report.html temp-style.css
```

## 注意事项

- WeasyPrint 的 `stylesheets` 参数接受文件路径，不是 CSS 字符串，必须先写入文件
- 中文需要系统安装中文字体，否则会显示为方块
- Pandoc pipe table 要求表格前后有空行才识别为表格
- 临时文件写在当前工作目录下，转换完成后清理
