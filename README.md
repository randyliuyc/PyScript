# TotalLINK 页面服务（page_server）

`page_server.py` 是本仓库的页面服务入口：它把 `BROTEX/` 下的页面功能模块（目前是
`bt8` 牵伸配色）用 HTTP 暴露出来，配色计算在本地跑，操作员在页面上确认方案后由本服务
转发 `linkDMGridRowSubmit` 回传 TotalLINK。

- 页面：`http://<地址>:8765/bt8`
- 配置：仓库根目录 `settings.json`
- 功能模块：`BROTEX/bt8_page.py` + `BROTEX/bt8_page.html`

## 1. 安装依赖（Windows）

需要 Python 3 和 `numpy`、`scipy`：

```
python -m pip install numpy scipy
```

如果使用项目自带的虚拟环境，把 `python` 换成 `venv\Scripts\python.exe`：

```
venv\Scripts\python.exe -m pip install numpy scipy
```

> 服务不依赖数据库：计算在本机完成，只有确认回传和按任务单取数需要能访问 `totallink.base_url`。

## 2. 启动与停止

| 方式 | 命令 |
| --- | --- |
| 双击启动（会打开浏览器） | `start_page_server.bat` |
| 命令行 | `python page_server.py` |
| 换端口 | `python page_server.py --port 8766` |
| 指定监听地址 | `python page_server.py --host 0.0.0.0` |
| 不自动打开浏览器 | `python page_server.py --no-open` |
| 后台运行（计划任务用） | `start_page_server_task.bat` |

启动后终端会打印可访问的地址；停止用 `Ctrl+C`（计划任务里见第 5 节）。

监听地址和端口以 `settings.json` 的 `server` 段为准，命令行参数只是临时覆盖。

## 3. settings.json

```json
{
  "server": { "host": "0.0.0.0", "port": 8765 },
  "totallink": { "base_url": "https://linkdev.bros.cn", "timeout": 120 },
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

| 字段 | 说明 |
| --- | --- |
| `server.host` | 监听地址。`0.0.0.0` = 本机和局域网都能访问；`127.0.0.1` = 仅本机 |
| `server.port` | 监听端口，默认 8765；被占用时改成 8766 等 |
| `totallink.base_url` | TotalLINK 地址（生产 `https://link.bros.cn`，开发 `https://linkdev.bros.cn`），修改后需重启服务 |
| `totallink.timeout` | 回传超时（秒）。BROSDEV 上 `linkDMGridRowSubmit` 实测约 50 秒，不要低于 60 |
| `features.bt8.api` | 确认回传接口路径 |
| `features.bt8.dmCode/dmNum/scriptType` | 回传模型编号，由 TotalLINK 分配 |
| `features.bt8.input.*` | 页面按任务单取数用的查询模型（默认 700） |

改完 `settings.json` 需要重启服务；页面代码（`.html`）改完刷新浏览器即可，服务每次请求都重新读文件。

## 4. 局域网访问（不只是 127.0.0.1）

按以下三步，其他电脑（或平板）就能用浏览器打开这个页面：

1. **让服务监听所有网卡**。`settings.json` 里 `server.host` 设为 `0.0.0.0`
   （仓库自带的配置已经是这个值）。只想本机访问就改回 `127.0.0.1`。
2. **查本机内网 IP**。命令行执行 `ipconfig`，取当前网卡（通常是"以太网"或"WLAN"）的
   IPv4 地址，例如 `192.168.1.20`。
3. **放行 Windows 防火墙**。以管理员身份执行一次（只放行本网段）：

```
netsh advfirewall firewall add rule name="TotalLINK page_server 8765" dir=in action=allow protocol=TCP localport=8765 remoteip=localsubnet profile=private,domain
```

   也可以在第一次启动时让 Windows 弹出"防火墙已阻止部分功能"对话框，勾选"专用网络"后允许。

然后在其他电脑上打开：

```
http://192.168.1.20:8765/bt8
```

说明与注意事项：

- 服务启动时会打印所有可用地址（本机 + 局域网），照抄其中一条即可。
- 页面和接口都在同一个地址下，页面里的请求是相对路径，换 IP 不用改页面。
- 防火墙规则只放行了 8765；改端口时要同步改规则的 `localport`。
- 内网 IP 由 DHCP 分配时可能变化，建议在路由器上给这台机器绑定固定 IP，或直接用机器名
  （`http://机器名:8765/bt8`）。
- **这个页面没有登录校验**：能访问到这个端口的人都能看到页面、触发计算和提交确认。
  页面 URL 里的 `loginID` 是 TotalLINK 发的登录凭据，请只在可信内网开放，
  不要把端口映射到公网。

## 5. Windows 计划任务（开机自启）

计划任务里用 `start_page_server_task.bat`：它不打开浏览器、不 `pause`，
把输出写进 `logs\page_server.log`，退出后不会卡住任务。

### 方式 A：命令行（推荐）

以管理员身份打开 CMD，把路径换成实际路径（`start_page_server_task.bat` 可以放在仓库根目录）：

开机自启、无需登录、以 SYSTEM 身份运行：

```
schtasks /Create /TN "TotalLINK page_server" /TR "\"C:\Pyscript\start_page_server_task.bat\"" /SC ONSTART /RU SYSTEM /RL HIGHEST /F
```

只在某个用户登录后启动（需要访问该用户的网络盘、用户配置时用这个）：

```
schtasks /Create /TN "TotalLINK page_server" /TR "\"C:\Pyscript\start_page_server_task.bat\"" /SC ONLOGON /RU "%USERNAME%" /RL HIGHEST /F
```

常用操作：

```
schtasks /Run /TN "TotalLINK page_server"
schtasks /Query /TN "TotalLINK page_server" /V /FO LIST
schtasks /End /TN "TotalLINK page_server"
schtasks /Delete /TN "TotalLINK page_server" /F
```

需要换端口或监听地址时，在 `/TR` 的脚本路径后面加参数（注意整体仍要加引号）：

```
schtasks /Create /TN "TotalLINK page_server" /TR "\"C:\Pyscript\start_page_server_task.bat\" --port 8766" /SC ONSTART /RU SYSTEM /RL HIGHEST /F
```

### 方式 B：任务计划程序图形界面

1. `Win+R` → 输入 `taskschd.msc` → 右侧"创建任务"（不要用"创建基本任务"，它没有"不管用户是否登录"选项）。
2. **常规**：填名称；按需要选"不管用户是否登录都要运行"（SYSTEM，开机即起）或
   "只在用户登录时运行"；勾选"使用最高权限运行"。
3. **触发器**：新建 → "开始任务"选"启动时"（或"登录时"）→ 勾选"延迟任务" 30 秒，等网络就绪。
4. **操作**：新建 → 启动程序 → "程序或脚本"填 `start_page_server_task.bat` 的完整路径；
   "起始于"填它所在目录（例如 `C:\Pyscript`）。
5. **条件**：笔记本请取消"只有在计算机使用交流电源时才启动此任务"。
6. **设置**：勾选"如果任务失败，按以下频率重新启动"（1 分钟 / 3 次）；
   取消"如果任务运行时间超过以下时间，停止任务"（服务要一直跑）。
7. 保存后右键任务 →"运行"，浏览器打开 `http://<本机 IP>:8765/bt8` 验证。

### 计划任务相关注意

- **日志**：`logs\page_server.log`，每次启动都会追加一行时间戳；请求日志里只有方法和路径，
  不包含 `loginID`。
- **目录位置**：放在同步盘（坚果云、OneDrive）里时，SYSTEM 身份可能读到"占位文件"而报错；
  遇到这种情况把仓库放到本地磁盘，或改用"只在用户登录时运行"。
- **端口占用**：任务启动失败先看日志，`OSError: [WinError 10048]` 表示端口被占用，
  改 `settings.json` 的 `port` 或换一台机器。
- **改了代码或配置**：`schtasks /End` 再 `schtasks /Run`，或在任务计划程序里右键"结束"后"运行"。
- 计划任务不会自动清理上一次的进程，重复"运行"前先确认没有残留的 `python.exe`。

## 6. 测试

```
cd BROTEX
python -m unittest test_bt8_page -v
```

测试会起一个假的 TotalLINK 上游，覆盖路由、计算契约、确认请求体、`linkResult` 判定、
Origin 校验和配置缺失等场景，不会真的连 TotalLINK。
