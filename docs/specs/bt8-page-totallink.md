# BT8 页面：参数传入与方案确认回传

- 状态：已确认，待实现
- 范围：`BROTEX/bt8_page.py`、`BROTEX/bt8_page.html`、`BROTEX/test_bt8_page.py`，新增环境级 `settings.json` 与共享服务入口
- 不在范围：`bt8.py` 算法本身、TotalLINK 服务端模型（`TMESMFGPLN10X727X`）的处理逻辑

页面从"本地手工计算工具"改为"由 TotalLINK 打开、计算方案、确认后回传"的操作页。以下每条都是已确认的决定，实现时不要重新发挥。

---

## 1. 背景

TotalLINK 的模型行里放一个链接按钮，点开本机 `page_server` 上的 BT8 页面。页面接收该行带来的输入数据与登录凭据，算完由操作员选定一个方案，确认后把**全部方案**以 JSON 回传给 TotalLINK，由服务端模型解析落库。

当前实现（独立服务器 8765 + 单页 HTML）已经具备计算、环形排布交互与本地确认；本规格要把它改造成带路由、带配置、带回传的环境级服务。

---

## 2. 地址与路由

功能不再挂在根路径，每个功能有自己的页面地址：

| 方法 | 路径 | 行为 |
|---|---|---|
| GET | `/` | 302 到唯一已注册功能的页面；注册了多个功能时返回一个简单的功能列表页 |
| GET | `/bt8` | BT8 页面（等价于现有的 `bt8_page.html`） |
| GET | `/bt8/` | 同上 |
| GET | `/api/bt8/sample` | 内置示例输入（本地调试用） |
| POST | `/api/bt8/calc` | 调用 `bt8.linkrun`，原样返回其 JSON |
| POST | `/api/bt8/input` | 用 `MFGNUM/MFBLIN/YARNGRP` 查模型 700（`linkDMDatasetResult`），返回颜色行 |
| POST | `/api/bt8/confirm` | 组装并转发确认报文到 TotalLINK |
| GET | `/favicon.ico` | 204 |
| 其他 | — | 404，JSON `{"isSuccess":0,"message":"页面不存在"}` |

从 TotalLINK 打开的完整地址：

```
http://127.0.0.1:8765/bt8?data=<URL 编码 JSON>&loginID=<token>&para=<URL 编码 JSON 数组>&rowData=<URL 编码 JSON 对象>
```

旧的 `/`（直接返回页面）与 `/api/calc` 不再保留；页面与接口都按功能名命名，新增功能时不需要动已有路由。

---

## 3. 配置：`settings.json`

放在仓库根目录，**入库，不 gitignore**（内含内网地址与模型编号，属于环境配置）。

```json
{
  "server": { "host": "127.0.0.1", "port": 8765 },
  "totallink": {
    "base_url": "https://link.bros.cn",
    "timeout": 30
  },
  "features": {
    "bt8": {
      "api": "/api/DataModel/linkDMGridRowSubmit",
      "dmCode": "TMESMFGPLN10X727X",
      "dmNum": 801,
      "scriptType": 4,
      "input": {
        "api": "/api/DataModel/linkDMDatasetResult",
        "dmCode": "TMESMFGPLN10X727X",
        "dmNum": 700
      }
    }
  }
}
```

约定：

- `totallink.base_url` 只放服务根地址，`features.<name>.api` 放具体接口路径，两者拼接才是完整 URL。
- `timeout` 单次转发的超时秒数，默认 30。TotalLINK 迁移版在配额预检时可能让服务端等待 10–30 秒，10 秒偏紧。
- `server` 节可选，命令行 `--host/--port` 优先于它。
- 配置文件缺失或某功能缺 `base_url` 时，服务照常启动、计算照常可用，但该功能的「确认」按钮禁用并说明原因。

---

## 4. URL 入参契约

| 参数 | 必填 | 内容 | 说明 |
|---|---|---|---|
| `data` | 否 | 输入 JSON | 结构与本仓库测试用例一致：`{"xmin","xmax","seed","maxtime","data":[{"MFMLIN","MFMDES","MFMSHO","MATRATCALC","PRIORITY","POSITION"}]}`；也接受只给颜色数组。**不传时**页面用 `MFGNUM/MFBLIN/YARNGRP` 调本地 `/api/bt8/input`，由服务端查模型 700（`linkDMDatasetResult`）取颜色 |
| `loginID` | 是 | LinkToken | 回传时原样放进请求体顶层同名字段；页面不显示、不写日志 |
| `para` | 是 | 模型参数数组的 JSON 字符串 | 对应 TotalLINK 的 `par.dm.Para`，没有参数传 `[]` |
| `rowData` | 否 | 业务列对象的 JSON 字符串 | 例：`{"PLANJSON":""}`；省略时页面默认 `{"PLANJSON": ""}`，回传 801 时由服务端填入方案 JSON |
| `MFGNUM` / `MFBLIN` / `YARNGRP` | 否 | 模型行直接给的三个业务参数 | 牵伸结果处理模型 801 的 Para 顺序；三者齐全且没有 `para` 时组成 `Para = [MFGNUM, MFBLIN, YARNGRP]`，显式 `para` 优先 |
| `lan` | 否 | 界面语言：`zh` 中文，`vie` 越南语 | 优先级高于浏览器本地记住的选择，只作用于本次打开；不带或无法识别时沿用本地记住的语言，首次打开默认越南语 |

> BROSDEV 环境的 `page_server` 超时配置为 120 秒：`linkDMGridRowSubmit` 在迁移版上实测约 50 秒返回（含配额预检等待），30 秒会超时。
>
> `data` / `para` / `rowData` 三个 JSON 参数也接受 **URL-safe Base64**（`-` `_`，可省略 `=` 补位）：先按 JSON 解析，失败再按 Base64 解码。URL 里带中文或 `&`/`#`/`+` 时用 Base64 比百分号转义更短更稳。

`rowData` 的填充约定：**值为空字符串的那个键由程序填入方案 JSON**。空值键必须恰好一个；零个或多个都按配置错误处理，在页面上直接报错，不猜。

`xmin / xmax / maxtime / seed` **永不出现在 URL 里**。页面上保留默认值 `1.03 / 3.5 / 10 / 0`，允许操作员修改。

`data` 传入后自动计算一次；不再需要 `autocalc` 参数。

---

## 5. 页面行为

### 5.1 只读与可改

带 `data` 打开时：

- 行号（`MFMLIN`）、简称（`MFMSHO`）、色比（`MATRATCALC`）**只读**，用 `readonly` 而不是 `disabled`，方便操作员复制核对。
- 隐藏「添加颜色」与每行的删除按钮，颜色行数固定为传入条数。
- `PRIORITY`（优先）与 `POSITION`（固定桶位）**可改**——这是操作员唯一能调的两个输入。
- 搜索设置（`xmin/xmax/maxtime/seed`）可改，默认折叠，改动后需重新点「开始计算」。
- 改过 `PRIORITY/POSITION` 或搜索参数后，不自动重算；「开始计算」按钮始终可用。

不带 `data` 打开（本地调试）：维持现有行为，所有字段可编辑、可增删颜色行。

### 5.2 计算

计算逻辑不变：`POST /api/bt8/calc`，服务端在锁内调用 `bt8.linkrun` 并原样返回。

---

## 6. 全部方案表格

列定义（从左到右）：

| 列 | 内容 |
|---|---|
| `#` | 行号，两位补零；已确认的方案后跟一个 `✓` |
| `X1` `X2` `X3` | 两组小数 |
| `总牵伸 D` | 六位有效数字 |
| `累计误差` | `cum_error_pct` + `%` |
| `优先色误差` | **新增**。所有优先色的最大实际误差；没有优先色时显示 `—`。阈值 0.05%，超过用错误色 |
| `告警` | **新增**。`warnings.length`；0 显示 `—`；有条数时用错误色，悬停显示全部告警文本（换行分隔） |
| `排布` | **新增**。四组简写，`FWP 特白条+HY 条子 / … / 白棉条+白棉条`；每个名称超过 4 个字符截断为前 3 字加 `…`，`title` 显示全称 |

其他：

- 删除原有的「优先色：达标 / 未达标」列——达标与否由「优先色误差」直接读出，避免没设优先色时整列都是 `—`。
- 列宽固定、单元格 `white-space: nowrap`，超长不换行，避免表格被撑开；排布列允许内部省略号。
- 排序沿用算法返回顺序 `(优先色未达标, 累计误差)`，前端不二次排序。
- 表格行点击/回车仍切换详情区。

已知限制：`优先色误差` 来自对 `colors[]` 字符串的解析，若颜色名称本身含 `: ` / ` -> ` / ` ( ` 会切错。实现时按现有 `parseColorLine` 的容错方式处理；遇到真实名称触发时再考虑让 `bt8.py` 额外返回结构化字段，不在本次范围。

---

## 7. 确认与回传

### 7.1 交互

- 全页**只允许一个最终确认**。
- 未确认时：方案详情下方是「确认方案 N」按钮。
- 已有确认方案时：
  - 该方案显示「已于 HH:MM 提交 TotalLINK（HTTP 200）」，圆环锁定、不能再调排布；
  - 其他方案的按钮变成「改为确认此方案」，点击即替换原确认并重新回传全量（报文里只带一个 `confirmedRank`）；
  - 顶部出现状态条「方案 03 已确认，可以关闭页面」（`Q4-D`），切换方案时状态条仍显示已确认的那个。
- 回传失败：显示原因与「重试回传」，**不锁定**方案，可以继续比较其他方案。

### 7.2 页面 → 本地服务

`POST /api/bt8/confirm`：

```json
{
  "loginID": "<token>",
  "para": ["..."],
  "rowData": { "PLANJSON": "" },
  "payload": {
    "isSuccess": 1,
    "confirmedRank": 3,
    "results": [
      { "rank": 1, "X1": 1.97, "X2": 1.29, "X3": 2.11,
        "cum_error": 0.00002, "cum_error_pct": 0.0002,
        "total_feed_speed_D": 5.513483, "stage_label": "scipy",
        "priority_ok": true, "warnings": [],
        "assign": [ …8 项… ], "colors": [ … ] },
      …
    ]
  }
}
```

- `results` 是**全部方案**，按界面顺序；每个方案是 `bt8.linkrun` 原始字段，只额外补一个 `rank`（1 起）。
- 不再为确认的方案加 `confirmed` 布尔——`rank` + `confirmedRank` 已经足够，避免两处真相。
- 方案对象里保留 `assign` 与 `colors`，服务端需要完整排布与配色明细。

### 7.3 本地服务 → TotalLINK

拼成 `linkDMGridRowSubmit` 的请求体，POST 到 `base_url + api`：

```json
{
  "loginID": "<token>",
  "par": {
    "dm": { "dmCode": "TMESMFGPLN10X727X", "dmNum": 801, "Para": ["..."] },
    "scriptType": 4,
    "rowData": { "PLANJSON": "<7.2 的 payload 序列化成字符串>" }
  }
}
```

- 空值键由服务端替换为 JSON 字符串；`rowData` 的其余键原样提交。
- 请求头：`Content-Type: application/json`，另带 `X-Bt8-Event: plan.confirmed` 便于排查。
- 不跟随重定向。

### 7.4 结果判定

TotalLINK 的返回是 `linkResult`，**`isSuccess` 是字符串**：

1. `str(isSuccess).lower() == "true"` 才算成功；
2. `action` 为 `prevent` / `confirm` 一律按失败处理并显示 `message`（`confirm` 需要人工确认，自动化流程应终止）；
3. `data` 为字符串时是错误码：
   - `InvalidLinkToken` → 提示「登录已失效，请让 TotalLINK 重新打开此页面」，不自动重新登录；
   - `AccessFail` → 「当前用户没有该模型的提交权限」；
   - `ModelNotExist` → 「模型 dmCode/dmNum 配置不正确」；
   - `NoConnection` → 「TotalLINK 服务端连接异常」；
4. `data` 为数字（`dataType=Int32`）时是影响行数，正常显示在成功提示里。

### 7.5 安全

- 服务只监听 `127.0.0.1`。
- `/api/<feature>/confirm` 校验 `Origin`：带了 `Origin` 且主机不是 `127.0.0.1 / localhost / ::1` 的请求直接 403，防止别的网站借浏览器转发。
- `loginID` 不写入任何日志或页面元素；日志里最多记录"已带 token"。

---

## 8. 代码结构

```
Pyscript/
├── settings.json            # 新增，入库
├── page_server.py           # 新增：环境级入口（路由、配置、转发）
└── BROTEX/
    ├── bt8.py               # 不动
    ├── bt8_page.py          # 改为「功能模块」：不再自带 main
    ├── bt8_page.html        # 改造
    └── test_bt8_page.py     # 改为测 page_server + bt8 功能
```

### 8.1 功能注册

`page_server.py` 里维护一个注册表，新增功能只加一行：

```python
FEATURES = {
    'bt8': importlib.import_module('BROTEX.bt8_page'),
}
```

每个功能模块暴露：

| 名字 | 用途 |
|---|---|
| `PAGE_FILE` | 页面 HTML 路径 |
| `calc(raw_text) -> str` | 接收请求体原文，返回 `linkrun` 的结果 JSON 文本 |
| `build_submit(feature_conf, login_id, para, row_data, payload) -> dict` | 组装发往 TotalLINK 的请求体 |
| `SAMPLE_PAYLOAD` | 可选，`/api/<feature>/sample` 用 |

路由层的转发、超时、`Origin` 校验、错误映射都是公共逻辑，写在 `page_server.py`，不放进功能模块。

### 8.2 启动

```bash
python page_server.py                 # 默认 127.0.0.1:8765，启动后打开 /bt8
python page_server.py --port 9000
python page_server.py --no-open
```

`BROTEX/bt8_page.py` 不再作为入口；相关说明与测试同步更新。

---

## 9. 错误处理清单

| 场景 | 页面表现 |
|---|---|
| 缺 `settings.json` 或 `totallink.base_url` 为空 | 计算正常；确认按钮禁用，提示「未配置 TotalLINK 地址」 |
| 缺 `loginID` | 确认按钮禁用，提示「缺少登录凭据」 |
| `para` 不是 JSON 数组 | 打开页面时表单区报错，不进入只读模式 |
| `rowData` 空值键数量 ≠ 1 | 同上 |
| `data` 解析失败 | 提示并回退到本地保存的输入（现有行为） |
| 计算服务不可达 | 现有提示不变 |
| 输入查询失败（网络/无权限/任务单无配色） | 表单区显示原因与「重新读取任务单」，不退回本地计算；连接层抖动由服务端自动重试一次 |
| 回传网络失败 / 超时 | 「无法连接 TotalLINK：…」+「重试回传」，不锁定 |
| `InvalidLinkToken` / `AccessFail` / `ModelNotExist` / `NoConnection` | 按 7.4 的文案显示 |
| `action=confirm/prevent` | 显示 `message`，按失败处理 |

---

## 10. 测试与验收

### 10.1 自动化

沿用 `unittest`，起本地服务（端口 0）与一个假的 TotalLINK 上游：

- 路由：`/` 302 到 `/bt8`；`/bt8` 200 且含页面标题；`/api/bt8/calc` 返回契约；未知路径 404。
- 只读：带 `data` 时 HTML/脚本进入只读态（可用断言检查页面里的只读开关），不带时可编辑。
- 确认：`rowData` 空值键被填入 JSON；`para` 进入 `par.dm.Para`；`scriptType` 与 `dmNum` 来自配置。
- 判定：假上游返回 `{"isSuccess":"true"}` 判成功；`"false"` + `InvalidLinkToken` 判失败；`action=prevent` 判失败。
- 安全：外站 `Origin` 403；缺 `loginID` 400/拒绝。
- 配置：`settings.json` 缺失时计算仍可用、确认被禁用。

### 10.2 手工验收

1. 按 `http://127.0.0.1:8765/bt8?data=…&loginID=…&para=[]&rowData={"PLANJSON":""}` 打开：自动算出方案，行号/简称/色比灰化只读，优先与桶位可点。
2. 改一个 `POSITION` 后点计算，结果随之变化。
3. 选一个方案点「确认方案 N」：TotalLINK 侧收到 `PLANJSON`，内容是全部方案 JSON，`confirmedRank` 指向所选方案，`results[i].rank` 与界面行号一致。
4. 再点另一个方案的「改为确认此方案」：重新回传，`confirmedRank` 变为新方案，旧方案不再带确认语义。
5. 断开网络点确认：报错并可重试，方案未锁定。
6. 手机宽度（375px）下表头不换行、排布列省略号正常。

---

## 11. 术语

| 词 | 含义 |
|---|---|
| 方案 | 算法返回的一条解，含 `X1/X2/X3`、`total_feed_speed_D`、8 桶位 `assign`、各色误差 `colors` |
| 排布 | 8 个桶位（A–H）上各放哪种颜色；由 `assign` 派生 |
| 组 | 两个速度相同的相邻桶位：AB、CD、EF、GH；CD 恒为 1.00 |
| 桶位 | A–H 共 8 个，输入里用 `POSITION` 固定某些颜色 |
| 优先色 | `PRIORITY=1` 的颜色，误差阈值 0.05%（普通色 0.5%） |
| 告警 | 单色误差超过对应阈值的提示，来自 `warnings[]` |
| 确认 | 操作员选定唯一最终方案并提交 TotalLINK 的动作 |
| 回传 | 本地服务把全部方案 JSON 转发到 TotalLINK 模型行的过程 |

---

## 12. 风险与未决

1. **报文体积**：`results` 含全部方案的 `assign` 与 `colors`，方案多时 JSON 可达数百 KB。服务端模型的存储列需要 `TEXT / NVARCHAR(MAX)`；实现侧不对方案数做截断。
2. **颜色名称含分隔符**会让 `colors[]` 解析错位，影响「优先色误差」。真实数据触发再议。
3. **服务端模型尚未设计**：本规格只固定请求体形状与判定规则；服务端如何解析 `PLANJSON`、如何保留全部方案并对确认方案特殊处理，由 TotalLINK 侧决定。
4. **Token 来源**：由 TotalLINK 打开页面时下发，页面与本地服务都不负责登录与续期。
