#!/usr/bin/env python3
"""
bt8 牵伸配色计算 —— 本地操作页面

启动：
    python BROTEX/bt8_page.py                    # http://127.0.0.1:8765
    python BROTEX/bt8_page.py --port 9000        # 换端口
    python BROTEX/bt8_page.py --no-open          # 不自动打开浏览器

页面是一个单文件 HTML（bt8_page.html），通过 POST /api/calc 提交与
linkrun 完全相同的输入 JSON；服务端直接调用 bt8.linkrun 并原样返回
它的结果 JSON。计算在本机完成，服务只监听 127.0.0.1。
"""

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bt8  # noqa: E402

PAGE_FILE = os.path.join(HERE, 'bt8_page.html')
MAX_BODY_BYTES = 512 * 1024
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8765

# 页面首次打开时预填的示例
SAMPLE_PAYLOAD = {
    'xmin': 1.03,
    'xmax': 3.5,
    'seed': 0,
    'maxtime': 10,
    'data': [
        {'MFMLIN': 20, 'MFMDES': 'FWP 特白条 UVA009BY', 'MFMSHO': 'FWP 特白条',
         'MATRATCALC': 28.0, 'PRIORITY': 0, 'POSITION': ''},
        {'MFMLIN': 112, 'MFMDES': '再生白涤纶 BROSRPET-24080102-2 1.56DTEX*38MM百隆金和泰',
         'MFMSHO': '再生白涤纶', 'MATRATCALC': 25.03, 'PRIORITY': 0, 'POSITION': ''},
        {'MFMLIN': 120, 'MFMDES': '白棉条 VA037ABY', 'MFMSHO': '白棉条',
         'MATRATCALC': 8.49, 'PRIORITY': 0, 'POSITION': ''},
        {'MFMLIN': 150, 'MFMDES': 'FW 特白条 VA008M', 'MFMSHO': 'FW 特白条',
         'MATRATCALC': 7.8, 'PRIORITY': 0, 'POSITION': ''},
        {'MFMLIN': 99999, 'MFMDES': 'HY 条子', 'MFMSHO': 'HY 条子',
         'MATRATCALC': 5.68, 'PRIORITY': 0, 'POSITION': ''},
    ],
}


def error_payload(message, runtime=0.0):
    """与 bt8.linkrun 一致的失败返回，用于服务层自身的错误"""
    return {
        'isSuccess': 0,
        'error': message,
        'message': message,
        'results': [],
        'runtime': round(runtime, 1),
    }


class Bt8CalculationServer(ThreadingHTTPServer):
    """带计算互斥锁的 HTTP 服务：同一时间只跑一个搜索，其余请求排队。"""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, handler_class):
        super().__init__(server_address, handler_class)
        self.calc_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = 'bt8-page/1.0'
    protocol_version = 'HTTP/1.1'

    # ---------- 路由 ----------

    def do_GET(self):
        path = self.path.split('?', 1)[0]
        if path in ('/', '/index.html', '/bt8_page.html'):
            self._serve_page()
        elif path == '/api/sample':
            self._send_json(json.dumps(SAMPLE_PAYLOAD, ensure_ascii=False))
        elif path == '/favicon.ico':
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_header('Content-Length', '0')
            self.end_headers()
        else:
            self._send_json(json.dumps(error_payload('页面不存在'), ensure_ascii=False),
                            status=HTTPStatus.NOT_FOUND)

    def do_POST(self):
        path = self.path.split('?', 1)[0]
        if path != '/api/calc':
            self.close_connection = True   # 未读取请求体，不复用这条连接
            self._send_json(json.dumps(error_payload('接口不存在'), ensure_ascii=False),
                            status=HTTPStatus.NOT_FOUND)
            return

        started = time.time()
        try:
            length = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            self.close_connection = True
            self._send_json(json.dumps(error_payload('Content-Length 不合法'), ensure_ascii=False),
                            status=HTTPStatus.BAD_REQUEST)
            return

        if length <= 0:
            self._send_json(json.dumps(error_payload('请求体为空，需要输入 JSON'), ensure_ascii=False))
            return
        if length > MAX_BODY_BYTES:
            self.close_connection = True
            self._send_json(json.dumps(error_payload('请求体过大'), ensure_ascii=False),
                            status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return

        raw = self.rfile.read(length)
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError:
            self._send_json(json.dumps(error_payload('请求体必须是 UTF-8 文本'), ensure_ascii=False))
            return

        # bt8.linkrun 自己会校验输入并返回 isSuccess=0 的说明
        with self.server.calc_lock:
            try:
                result = bt8.linkrun(text)
            except Exception as exc:  # 防御：算法内部异常也返回 JSON，页面不至于空白
                result = json.dumps(
                    error_payload(f'计算失败: {type(exc).__name__}: {exc}',
                                  time.time() - started),
                    ensure_ascii=False, allow_nan=False)
        self._send_json(result)

    # ---------- 响应 ----------

    def _serve_page(self):
        try:
            with open(PAGE_FILE, 'rb') as fh:
                body = fh.read()
        except OSError as exc:
            self._send_json(json.dumps(error_payload(f'找不到页面文件: {exc}'), ensure_ascii=False),
                            status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, text, status=HTTPStatus.OK):
        body = text.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stderr.write('[bt8-page] %s %s\n' % (self.address_string(), fmt % args))


def make_server(host=DEFAULT_HOST, port=DEFAULT_PORT):
    """创建服务实例；port=0 时由系统分配（测试用）。"""
    return Bt8CalculationServer((host, port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description='bt8 牵伸配色计算 - 本地操作页面')
    parser.add_argument('--host', default=DEFAULT_HOST, help='监听地址（默认 127.0.0.1）')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='监听端口（默认 8765）')
    parser.add_argument('--no-open', action='store_true', help='启动后不自动打开浏览器')
    args = parser.parse_args(argv)

    try:
        server = make_server(args.host, args.port)
    except OSError as exc:
        print(f'无法在 {args.host}:{args.port} 启动服务: {exc}', file=sys.stderr)
        print('端口可能被占用，请换一个：python BROTEX/bt8_page.py --port 8766', file=sys.stderr)
        return 1

    host, port = server.server_address[:2]
    url = f'http://{host}:{port}/'
    print(f'bt8 计算页面已启动: {url}    (Ctrl+C 停止)')
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
