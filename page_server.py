#!/usr/bin/env python3
"""
TotalLINK 本地操作页面服务 —— 环境级入口

启动：
    python page_server.py                  # 127.0.0.1:8765，启动后打开 /bt8
    python page_server.py --port 9000
    python page_server.py --no-open

职责：
- 路由：/ 跳到唯一功能页，/bt8 是 BT8 页面，/api/<功能>/... 是该功能的接口
- 配置：仓库根 settings.json（server / totallink / features）
- 转发：把页面确认的方案组装成 linkDMGridRowSubmit 请求体提交 TotalLINK

每个功能是一个模块（如 BROTEX/bt8_page.py），只提供
PAGE_FILE / calc / build_submit / SAMPLE_PAYLOAD；
超时、Origin 校验、linkResult 判定等公共逻辑都在这里。
"""

import argparse
import html
import importlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SETTINGS_FILE = os.path.join(ROOT, 'settings.json')

# 新增功能只加一行
FEATURES = {
    'bt8': importlib.import_module('BROTEX.bt8_page'),
}

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8765
DEFAULT_TIMEOUT = 30
MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_UPSTREAM_BYTES = 256 * 1024
LOCAL_HOSTS = ('127.0.0.1', 'localhost', '::1')

# 规格 7.4：linkResult 的 data 是错误码时的页面文案
LINK_ERROR_MESSAGES = {
    'InvalidLinkToken': '登录已失效，请让 TotalLINK 重新打开此页面',
    'AccessFail': '当前用户没有该模型的提交权限',
    'ModelNotExist': '模型 dmCode/dmNum 配置不正确',
    'NoConnection': 'TotalLINK 服务端连接异常',
}


# ---------- 配置 ----------

def load_settings(path=SETTINGS_FILE):
    """读取 settings.json；文件缺失或损坏时返回 {}，服务照常启动。"""
    try:
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _section(container, key):
    value = container.get(key) if isinstance(container, dict) else None
    return value if isinstance(value, dict) else {}


def feature_config(settings, name):
    return _section(_section(settings, 'features'), name)


def totallink_config(settings):
    """返回 (base_url, timeout)；base_url 未配置时为空串。"""
    conf = _section(settings, 'totallink')
    base = str(conf.get('base_url') or '').strip().rstrip('/')
    try:
        timeout = float(conf.get('timeout') or DEFAULT_TIMEOUT)
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT
    if timeout <= 0:
        timeout = DEFAULT_TIMEOUT
    return base, timeout


def confirm_target(settings, name):
    """返回 (完整提交地址, 超时)；地址不完整时第一项为 None。"""
    base, timeout = totallink_config(settings)
    api = str(feature_config(settings, name).get('api') or '').strip()
    if not base or not api:
        return None, timeout
    return base + (api if api.startswith('/') else '/' + api), timeout


def input_target(settings, name):
    """输入查询模型的完整地址（linkDMDatasetResult）；未配置时第一项为 None。"""
    base, timeout = totallink_config(settings)
    conf = _section(feature_config(settings, name), 'input')
    api = str(conf.get('api') or '').strip()
    if not base or not api:
        return None, timeout
    return base + (api if api.startswith('/') else '/' + api), timeout


def input_status(settings, name):
    """输入查询能否使用，返回 (可用, 不可用原因)。"""
    base, _ = totallink_config(settings)
    if not base:
        return False, '未配置 TotalLINK 地址'
    conf = _section(feature_config(settings, name), 'input')
    if not str(conf.get('api') or '').strip():
        return False, '未配置输入查询接口'
    if conf.get('dmCode') in (None, '') or conf.get('dmNum') in (None, ''):
        return False, '输入查询模型编号配置不完整'
    return True, ''


def build_input_body(feature_conf, login_id, para):
    """组装 linkDMDatasetResult 的请求体（模型查询）。"""
    conf = _section(feature_conf, 'input')
    return {
        'loginID': login_id,
        'par': {
            'dmCode': conf.get('dmCode'),
            'dmNum': conf.get('dmNum'),
            'Para': list(para),
            'SimpleFlag': False,
        },
        'RDs': [],
        'rowData': None,
    }


def extract_rows(response):
    """从 linkDMDatasetResult 的 data 里取第一张表的行对象列表。"""
    payload = response.get('data') if isinstance(response, dict) else None
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def confirm_status(settings, name=None):
    """确认能否提交，返回 (可用, 不可用原因)。"""
    base, _ = totallink_config(settings)
    if not base:
        return False, '未配置 TotalLINK 地址'
    if name and not str(feature_config(settings, name).get('api') or '').strip():
        return False, '未配置 TotalLINK 提交接口'
    return True, ''


def server_address(settings):
    conf = _section(settings, 'server')
    host = str(conf.get('host') or DEFAULT_HOST)
    try:
        port = int(conf.get('port') or DEFAULT_PORT)
    except (TypeError, ValueError):
        port = DEFAULT_PORT
    return host, port


def page_config(settings, name):
    """注入页面的配置：确认按钮是否可用。"""
    available, reason = confirm_status(settings, name)
    return {'confirmEnabled': available, 'confirmReason': '' if available else reason}


def inject_config(html_text, config):
    """把服务端配置作为 window.SERVER_CONFIG 注入页面。

    页面据此决定确认按钮是否可用；直接用浏览器打开 HTML 文件时没有这段配置，
    页面按「未配置 TotalLINK 地址」处理。
    """
    script = '<script>window.SERVER_CONFIG = %s;</script>\n' % json.dumps(config, ensure_ascii=False)
    marker = '<body>'
    at = html_text.find(marker)
    if at < 0:
        return script + html_text
    at += len(marker)
    return html_text[:at] + '\n' + script + html_text[at:]


# ---------- 转发 ----------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """确认只看 TotalLINK 的直接响应，不跟随重定向。"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def post_json(url, body, timeout, headers=None):
    """POST JSON，不跟随重定向；返回 (HTTP 状态码, 响应字节)。"""
    all_headers = {
        'Content-Type': 'application/json; charset=utf-8',
        'User-Agent': 'page-server/1.0',
    }
    all_headers.update(headers or {})
    req = urllib.request.Request(url, data=body, method='POST', headers=all_headers)
    try:
        with urllib.request.build_opener(_NoRedirect).open(req, timeout=timeout) as resp:
            return resp.status, resp.read(MAX_UPSTREAM_BYTES)
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read(MAX_UPSTREAM_BYTES)
        except OSError:
            raw = b''
        return exc.code, raw


def friendly_net_error(exc):
    """把 socket 层的英文错误换成页面上能看懂的说明。"""
    reason = getattr(exc, 'reason', exc)
    text = str(reason)
    low = text.lower()
    if 'connection refused' in low or 'errno 61' in low or 'errno 111' in low:
        return '连接被拒绝，TotalLINK 服务可能没有启动'
    if 'timed out' in low or 'timeout' in low:
        return '连接超时'
    if ('name or service not known' in low or 'nodename nor servname' in low
            or 'temporary failure in name resolution' in low):
        return '主机名无法解析'
    return text


def origin_allowed(origin):
    """Origin 只放行本机页面，防止别的网站借浏览器转发确认。"""
    try:
        host = urllib.parse.urlsplit(origin).hostname
    except ValueError:
        return False
    return host in LOCAL_HOSTS


def interpret_response(status, raw):
    """解析 TotalLINK 的响应并判定结果（规格 7.4）。"""
    text = raw.decode('utf-8', 'replace').strip()
    if not text:
        return {'isSuccess': 0, 'status': status,
                'message': f'TotalLINK 返回 HTTP {status}，响应为空'}
    try:
        data = json.loads(text)
    except ValueError:
        return {'isSuccess': 0, 'status': status,
                'message': f'TotalLINK 返回了非 JSON 内容（HTTP {status}）：{text[:200]}'}
    if not isinstance(data, dict):
        return {'isSuccess': 0, 'status': status,
                'message': f'TotalLINK 返回内容无法识别（HTTP {status}）'}
    return judge_link_result(data, status)


def judge_link_result(data, status):
    """按 linkResult 字段判定成功与否：isSuccess 是字符串。"""
    message = str(data.get('message') or '').strip()
    action = str(data.get('action') or '').strip().lower()
    ok = str(data.get('isSuccess')).strip().lower() == 'true'
    payload = data.get('data')

    # 需要人工处理的 action 一律视为失败
    if action in ('prevent', 'confirm'):
        return {'isSuccess': 0, 'status': status, 'action': action,
                'message': message or f'TotalLINK 要求人工处理（action={action}）'}

    code = payload.strip() if isinstance(payload, str) else ''
    if code in LINK_ERROR_MESSAGES:
        return {'isSuccess': 0, 'status': status, 'code': code,
                'message': LINK_ERROR_MESSAGES[code]}
    if not ok:
        detail = message or f'TotalLINK 拒绝提交（HTTP {status}）'
        if code:
            detail = f'{detail}：{code}'
        return {'isSuccess': 0, 'status': status, 'code': code, 'message': detail}

    result = {'isSuccess': 1, 'status': status, 'message': message or '已提交 TotalLINK'}
    rows = None
    if isinstance(payload, int) and not isinstance(payload, bool):
        rows = payload
    elif isinstance(payload, str) and payload.strip().isdigit():
        rows = int(payload.strip())
    # TotalLINK 的脚本类动作常回 data=0（实际已写入），0 当作「未报告行数」，
    # 只有明确大于 0 才在提示里说影响行数，避免页面显示「影响 0 行」误导操作员。
    if rows:
        result['rowsAffected'] = rows
        result['message'] = f'已提交 TotalLINK，影响 {rows} 行'
    return result


# ---------- 服务 ----------

class PageServer(ThreadingHTTPServer):
    """带计算互斥锁的 HTTP 服务：同一时间只跑一个搜索，其余请求排队。"""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, handler_class, settings=None, features=None):
        super().__init__(server_address, handler_class)
        self.settings = {} if settings is None else settings
        self.features = dict(FEATURES if features is None else features)
        self.calc_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = 'page-server/1.0'
    protocol_version = 'HTTP/1.1'

    # ---------- 路由 ----------

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == '/':
            return self._serve_index()
        if path == '/favicon.ico':
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header('Content-Length', '0')
            self.end_headers()
            return

        name = self._feature_page(path)
        if name:
            return self._serve_page(name)

        name, action = self._feature_api(path)
        if name and action == 'sample':
            sample = getattr(self.server.features[name], 'SAMPLE_PAYLOAD', None)
            if sample is None:
                return self._send_json({'isSuccess': 0, 'message': '该功能没有示例'},
                                       HTTPStatus.NOT_FOUND)
            return self._send_json(sample)
        self._not_found()

    def do_POST(self):
        path = urllib.parse.urlsplit(self.path).path
        name, action = self._feature_api(path)
        if name and action == 'calc':
            return self._handle_calc(name)
        if name and action == 'input':
            return self._handle_input(name)
        if name and action == 'confirm':
            return self._handle_confirm(name)
        self._not_found()

    def _feature_page(self, path):
        for name in self.server.features:
            if path in ('/' + name, '/' + name + '/'):
                return name
        return None

    def _feature_api(self, path):
        parts = path.split('/')
        if len(parts) == 4 and parts[0] == '' and parts[1] == 'api' and parts[2] in self.server.features:
            return parts[2], parts[3]
        return None, None

    # ---------- 页面 ----------

    def _serve_index(self):
        names = list(self.server.features)
        if not names:
            return self._not_found()
        if len(names) == 1:
            self.send_response(HTTPStatus.FOUND)
            self.send_header('Location', '/' + names[0])
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        items = ''.join(
            f'<li><a href="/{urllib.parse.quote(n)}">{html.escape(n)}</a></li>' for n in names)
        body = ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
                '<title>页面列表</title><h1>页面列表</h1><ul>' + items + '</ul>')
        self._send_bytes(body.encode('utf-8'), 'text/html; charset=utf-8')

    def _serve_page(self, name):
        page_file = getattr(self.server.features[name], 'PAGE_FILE', '')
        try:
            with open(page_file, 'rb') as fh:
                text = fh.read().decode('utf-8')
        except (OSError, UnicodeDecodeError) as exc:
            return self._send_json({'isSuccess': 0, 'message': f'找不到页面文件：{exc}'},
                                   HTTPStatus.INTERNAL_SERVER_ERROR)
        body = inject_config(text, page_config(self.server.settings, name)).encode('utf-8')
        self._send_bytes(body, 'text/html; charset=utf-8')

    # ---------- 计算 ----------

    def _handle_calc(self, name):
        raw = self._read_body()
        if raw is None:
            return
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            return self._send_json({'isSuccess': 0, 'message': '请求体必须是 UTF-8 文本'})

        module = self.server.features[name]
        with self.server.calc_lock:
            try:
                result = module.calc(text)
            except Exception as exc:  # 防御：算法异常也返回 JSON，页面不至于空白
                result = json.dumps(
                    {'isSuccess': 0, 'message': f'计算失败：{type(exc).__name__}: {exc}'},
                    ensure_ascii=False)
        self._send_json_text(result)

    # ---------- 输入查询 ----------

    def _handle_input(self, name):
        """用 URL 带来的 LinkToken 查询模型（默认 700），返回颜色行。"""
        origin = self.headers.get('Origin')
        if origin and not origin_allowed(origin):
            self.close_connection = True
            return self._send_json({'isSuccess': 0, 'message': '拒绝来自其他网站的请求'},
                                   HTTPStatus.FORBIDDEN)

        raw = self._read_body()
        if raw is None:
            return
        try:
            request = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return self._bad_request(f'输入查询请求不是合法 JSON：{exc}')
        if not isinstance(request, dict):
            return self._bad_request('输入查询请求必须是 JSON 对象')

        login_id = str(request.get('loginID') or '').strip()
        if not login_id:
            return self._bad_request('缺少登录凭据')

        params = [str(request.get(key) or '').strip() for key in ('MFGNUM', 'MFBLIN', 'YARNGRP')]
        if not params[0] or not params[1]:
            return self._bad_request('缺少 MFGNUM 或 MFBLIN')

        settings = self.server.settings
        available, reason = input_status(settings, name)
        if not available:
            return self._send_json({'isSuccess': 0, 'message': reason})
        target, timeout = input_target(settings, name)

        body_obj = build_input_body(feature_config(settings, name), login_id, params)
        body = json.dumps(body_obj, ensure_ascii=False).encode('utf-8')
        started = time.time()
        try:
            status, raw_resp = post_json(target, body, timeout,
                                         headers={'X-%s-Event' % name.capitalize(): 'input.query'})
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return self._send_json({'isSuccess': 0,
                                    'message': f'无法连接 TotalLINK：{friendly_net_error(exc)}'})

        text = raw_resp.decode('utf-8', 'replace').strip()
        try:
            data = json.loads(text)
        except ValueError:
            return self._send_json({'isSuccess': 0, 'status': status,
                                    'message': f'TotalLINK 返回了非 JSON 内容（HTTP {status}）'})
        if not isinstance(data, dict):
            return self._send_json({'isSuccess': 0, 'status': status,
                                    'message': f'TotalLINK 返回内容无法识别（HTTP {status}）'})

        if str(data.get('isSuccess')).strip().lower() != 'true':
            code = data.get('data') if isinstance(data.get('data'), str) else ''
            message = LINK_ERROR_MESSAGES.get(code) or str(data.get('message') or '').strip() \
                or f'查询任务单失败（HTTP {status}）'
            return self._send_json({'isSuccess': 0, 'status': status, 'code': code,
                                    'message': message})

        rows = extract_rows(data)
        if not rows:
            return self._send_json({'isSuccess': 0, 'status': status,
                                    'message': '该任务单没有可计算的配色数据'})
        self._send_json({'isSuccess': 1, 'status': status, 'rows': rows,
                         'count': len(rows), 'runtime': round(time.time() - started, 1)})

    # ---------- 确认回传 ----------

    def _handle_confirm(self, name):
        origin = self.headers.get('Origin')
        if origin and not origin_allowed(origin):
            self.close_connection = True   # 未读取请求体，不复用这条连接
            return self._send_json({'isSuccess': 0, 'message': '拒绝来自其他网站的确认请求'},
                                   HTTPStatus.FORBIDDEN)

        raw = self._read_body()
        if raw is None:
            return
        try:
            request = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            return self._bad_request(f'确认请求不是合法 JSON：{exc}')
        if not isinstance(request, dict):
            return self._bad_request('确认请求必须是 JSON 对象')

        login_id = str(request.get('loginID') or '').strip()
        if not login_id:
            return self._bad_request('缺少登录凭据')

        para = request.get('para')
        if para is None:
            para = []
        if not isinstance(para, list):
            return self._bad_request('para 必须是 JSON 数组')

        row_data = request.get('rowData')
        if not isinstance(row_data, dict):
            return self._bad_request('缺少 rowData（模型行的业务列）')

        payload = request.get('payload')
        if not isinstance(payload, dict):
            return self._bad_request('缺少 payload（确认的方案内容）')

        settings = self.server.settings
        available, reason = confirm_status(settings, name)
        if not available:
            return self._send_json({'isSuccess': 0, 'message': reason})
        target, timeout = confirm_target(settings, name)

        conf = feature_config(settings, name)
        missing = [k for k in ('dmCode', 'dmNum', 'scriptType') if conf.get(k) is None]
        if missing:
            return self._send_json({'isSuccess': 0,
                                    'message': f'功能 {name} 的配置缺少 {"/".join(missing)}'})

        module = self.server.features[name]
        try:
            body_obj = module.build_submit(conf, login_id, para, row_data, payload)
            body = json.dumps(body_obj, ensure_ascii=False, allow_nan=False).encode('utf-8')
        except (ValueError, TypeError) as exc:
            return self._bad_request(str(exc))

        # bt8 → X-Bt8-Event，便于 TotalLINK 侧排查
        event_header = 'X-%s-Event' % name.capitalize()
        started = time.time()
        try:
            status, raw_resp = post_json(target, body, timeout,
                                         headers={event_header: 'plan.confirmed'})
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return self._send_json({'isSuccess': 0,
                                    'message': f'无法连接 TotalLINK：{friendly_net_error(exc)}'})

        result = interpret_response(status, raw_resp)
        result['runtime'] = round(time.time() - started, 1)
        self._send_json(result)

    # ---------- 请求/响应 ----------

    def _read_body(self):
        """读取请求体；出错时已写好响应并返回 None。"""
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            self.close_connection = True
            self._send_json({'isSuccess': 0, 'message': 'Content-Length 不合法'},
                            HTTPStatus.BAD_REQUEST)
            return None

        if length <= 0:
            self._send_json({'isSuccess': 0, 'message': '请求体为空，需要输入 JSON'})
            return None
        if length > MAX_BODY_BYTES:
            self.close_connection = True
            self._send_json({'isSuccess': 0, 'message': '请求体过大'},
                            HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return None
        return self.rfile.read(length)

    def _send_bytes(self, body, content_type):
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, status=HTTPStatus.OK):
        self._send_json_text(json.dumps(obj, ensure_ascii=False), status)

    def _send_json_text(self, text, status=HTTPStatus.OK):
        body = text.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def _bad_request(self, message):
        self._send_json({'isSuccess': 0, 'message': message}, HTTPStatus.BAD_REQUEST)

    def _not_found(self):
        self.close_connection = True
        self._send_json({'isSuccess': 0, 'message': '页面不存在'}, HTTPStatus.NOT_FOUND)

    def log_request(self, code='-', size='-'):
        # 查询串里有 loginID，日志只保留方法与路径
        path = urllib.parse.urlsplit(self.path).path
        sys.stderr.write('[page-server] %s "%s %s" %s %s\n'
                         % (self.address_string(), self.command, path, code, size))

    def log_message(self, fmt, *args):
        # 请求体与查询串含 loginID，不写日志
        sys.stderr.write('[page-server] %s %s\n' % (self.address_string(), fmt % args))


def make_server(host=DEFAULT_HOST, port=DEFAULT_PORT, settings=None, features=None):
    """创建服务实例；port=0 时由系统分配（测试用）。"""
    if settings is None:
        settings = load_settings()
    return PageServer((host, port), Handler, settings=settings, features=features)


def main(argv=None):
    settings = load_settings()
    default_host, default_port = server_address(settings)

    parser = argparse.ArgumentParser(description='TotalLINK 本地操作页面服务')
    parser.add_argument('--host', default=None, help=f'监听地址（默认 {default_host}）')
    parser.add_argument('--port', type=int, default=None, help=f'监听端口（默认 {default_port}）')
    parser.add_argument('--no-open', action='store_true', help='启动后不自动打开浏览器')
    args = parser.parse_args(argv)

    host = args.host or default_host
    port = default_port if args.port is None else args.port
    try:
        server = make_server(host, port, settings=settings)
    except OSError as exc:
        print(f'无法在 {host}:{port} 启动服务：{exc}', file=sys.stderr)
        print('端口可能被占用，请换一个：python page_server.py --port 8766', file=sys.stderr)
        return 1

    actual_host, actual_port = server.server_address[:2]
    url = f'http://{actual_host}:{actual_port}/bt8'
    print(f'页面服务已启动：{url}    (Ctrl+C 停止)')
    if not settings:
        print('提示：未读到 settings.json，计算可用，确认回传已禁用。', file=sys.stderr)
    if not args.no_open:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n已停止')
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
