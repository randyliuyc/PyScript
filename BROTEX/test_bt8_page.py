#!/usr/bin/env python3
"""
page_server + bt8 页面功能的接口测试

对照 docs/specs/bt8-page-totallink.md 第 10.1 节：
- 路由：/ 302 到 /bt8；/bt8 与 /bt8/ 返回页面；/api/bt8/calc 契约；未知路径 404
- 只读：页面里带只读开关（state.platform），平台模式下加 readonly 并隐藏增删
- 确认：rowData 空值键被填入 JSON、para 进入 par.dm.Para、dmNum/scriptType 来自配置
- 判定：isSuccess 字符串 true 判成功；false + InvalidLinkToken 判失败；action=prevent 判失败
- 安全：外站 Origin 403；缺 loginID 400
- 配置：settings.json 缺失时计算可用、确认被禁用

运行：
    python -m unittest test_bt8_page -v
依赖：numpy、scipy
"""
import contextlib
import io
import json
import os
import socket
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (ROOT, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import page_server  # noqa: E402

bt8_page = page_server.FEATURES['bt8']

FEATURE_CONF = {
    'api': '/api/DataModel/linkDMGridRowSubmit',
    'dmCode': 'TMESMFGPLN10X727X',
    'dmNum': 504,
    'scriptType': 4,
}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class UpstreamStub(BaseHTTPRequestHandler):
    """假的 TotalLINK：记录收到的提交，按 reply 返回。"""

    protocol_version = 'HTTP/1.1'
    received = []
    reply = (200, {'isSuccess': 'true', 'message': '', 'action': '',
                   'data': 1, 'dataType': 'Int32'})

    def do_POST(self):
        length = int(self.headers.get('Content-Length', '0'))
        body = self.rfile.read(length)
        type(self).received.append({
            'path': self.path,
            'content_type': self.headers.get('Content-Type', ''),
            'event': self.headers.get('X-Bt8-Event', ''),
            'body': body,
        })
        code, payload = type(self).reply
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt, *args):
        pass


def closed_port():
    """开一个临时端口再关掉，得到一个多半没人监听的本地端口。"""
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class PageServerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.servers = []

        cls.upstream = ThreadingHTTPServer(('127.0.0.1', 0), UpstreamStub)
        host, port = cls.upstream.server_address[:2]
        cls.upstream_url = f'http://{host}:{port}'
        cls.upstream_thread = threading.Thread(target=cls.upstream.serve_forever, daemon=True)
        cls.upstream_thread.start()

        _, cls.base = cls.start_server(cls.settings(cls.upstream_url))

    @classmethod
    def tearDownClass(cls):
        for server, thread in cls.servers:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        cls.upstream.shutdown()
        cls.upstream.server_close()
        cls.upstream_thread.join(timeout=5)

    @classmethod
    def settings(cls, base_url):
        return {
            'totallink': {'base_url': base_url, 'timeout': 5},
            'features': {'bt8': dict(FEATURE_CONF)},
        }

    @classmethod
    def start_server(cls, settings):
        server = page_server.make_server('127.0.0.1', 0, settings=settings)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        cls.servers.append((server, thread))
        return server, 'http://%s:%d' % server.server_address[:2]

    def setUp(self):
        UpstreamStub.received.clear()
        UpstreamStub.reply = (200, {'isSuccess': 'true', 'message': '', 'action': '',
                                    'data': 1, 'dataType': 'Int32'})

    # ---------- 请求工具 ----------

    def request(self, path, method='GET', body=None, headers=None, base=None, follow=True):
        data = None
        if body is not None:
            data = body if isinstance(body, bytes) else json.dumps(body).encode('utf-8')
        req = urllib.request.Request((base or self.base) + path, data=data,
                                     method=method, headers=headers or {})
        handlers = [] if follow else [_NoRedirect]
        opener = urllib.request.build_opener(*handlers)
        try:
            with opener.open(req, timeout=30) as resp:
                return resp.status, resp.headers, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.headers, exc.read()

    def page(self, base=None):
        status, headers, body = self.request('/bt8', base=base)
        self.assertEqual(status, 200)
        self.assertIn('text/html', headers.get('Content-Type', ''))
        return body.decode('utf-8')

    def confirm_request(self, **overrides):
        body = {
            'loginID': 'token-42',
            'para': ['p1', 'p2'],
            'rowData': {'PLANJSON': '', 'PLANID': 'A-1'},
            'payload': {'isSuccess': 1, 'confirmedRank': 2,
                        'results': [{'rank': 1}, {'rank': 2}]},
        }
        body.update(overrides)
        return body

    def post_confirm(self, body, origin=None, base=None):
        headers = {'Content-Type': 'application/json'}
        if origin:
            headers['Origin'] = origin
        status, _, raw = self.request('/api/bt8/confirm', 'POST', body, headers, base=base)
        return status, json.loads(raw.decode('utf-8'))

    def post_calc(self, payload, base=None):
        status, _, raw = self.request('/api/bt8/calc', 'POST', payload,
                                      {'Content-Type': 'application/json'}, base=base)
        return status, json.loads(raw.decode('utf-8'))

    # ---------- 路由 ----------

    def test_root_redirects_to_bt8(self):
        status, headers, _ = self.request('/', follow=False)
        self.assertEqual(status, 302)
        self.assertEqual(headers.get('Location'), '/bt8')

    def test_feature_page_served(self):
        page = self.page()
        self.assertIn('牵伸配色计算', page)
        self.assertIn('/api/bt8/calc', page)
        self.assertIn('/api/bt8/confirm', page)
        self.assertIn('readUrlContext', page)

    def test_feature_page_trailing_slash(self):
        status, _, body = self.request('/bt8/')
        self.assertEqual(status, 200)
        self.assertIn('牵伸配色计算', body.decode('utf-8'))

    def test_sample_payload(self):
        status, _, raw = self.request('/api/bt8/sample')
        self.assertEqual(status, 200)
        data = json.loads(raw.decode('utf-8'))
        self.assertIsInstance(data.get('data'), list)
        self.assertTrue(data['data'])

    def test_unknown_path_is_json_404(self):
        for path in ('/nope', '/api/nope/calc', '/api/bt8/nope'):
            status, _, raw = self.request(path)
            self.assertEqual(status, 404, path)
            self.assertEqual(json.loads(raw.decode('utf-8')),
                             {'isSuccess': 0, 'message': '页面不存在'})

    # ---------- 只读 ----------

    def test_page_has_readonly_switch(self):
        page = self.page()
        # 带 data 打开时行号/简称/色比加 readonly（不是 disabled），并隐藏增删
        self.assertIn("state.platform ? ' readonly' : ''", page)
        self.assertIn('refs.addColor.hidden = state.platform;', page)
        self.assertIn("const del = state.platform ? '' :", page)

    def test_page_reports_confirm_available(self):
        self.assertIn('"confirmEnabled": true', self.page())

    def test_login_id_is_not_logged(self):
        """规格 7.5：loginID 不写日志，请求日志只保留路径。"""
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            status, _, _ = self.request('/bt8?loginID=super-secret-token&data=%7B%7D')
        self.assertEqual(status, 200)
        logged = buf.getvalue()
        self.assertNotIn('super-secret-token', logged)
        self.assertIn('/bt8', logged)

    # ---------- 计算 ----------

    def test_calc_returns_results(self):
        status, data = self.post_calc(dict(bt8_page.SAMPLE_PAYLOAD, maxtime=5))
        self.assertEqual(status, 200)
        self.assertEqual(data.get('isSuccess'), 1, data.get('message'))
        self.assertTrue(data.get('results'))
        first = data['results'][0]
        self.assertEqual(len(first['assign']), 8)
        for key in ('X1', 'X2', 'X3', 'cum_error', 'cum_error_pct',
                    'total_feed_speed_D', 'priority_ok', 'warnings', 'assign', 'colors'):
            self.assertIn(key, first)

    def test_calc_rejects_invalid_input(self):
        _, data = self.post_calc({'data': []})
        self.assertEqual(data.get('isSuccess'), 0)
        self.assertTrue(data.get('message'))

    def test_calc_rejects_malformed_json(self):
        status, _, raw = self.request('/api/bt8/calc', 'POST', b'{not json',
                                      {'Content-Type': 'application/json'})
        data = json.loads(raw.decode('utf-8'))
        self.assertEqual(status, 200)
        self.assertEqual(data.get('isSuccess'), 0)
        self.assertIn('输入错误', data.get('message', ''))

    # ---------- 确认：组装请求体 ----------

    def test_confirm_builds_link_submit_body(self):
        status, data = self.post_confirm(self.confirm_request())
        self.assertEqual(status, 200)
        self.assertEqual(data.get('isSuccess'), 1, data.get('message'))
        self.assertEqual(data.get('status'), 200)
        self.assertEqual(data.get('rowsAffected'), 1)

        self.assertEqual(len(UpstreamStub.received), 1)
        got = UpstreamStub.received[0]
        self.assertEqual(got['path'], FEATURE_CONF['api'])
        self.assertEqual(got['event'], 'plan.confirmed')
        self.assertIn('application/json', got['content_type'])

        sent = json.loads(got['body'].decode('utf-8'))
        self.assertEqual(sent['loginID'], 'token-42')
        self.assertEqual(sent['par']['dm']['dmCode'], 'TMESMFGPLN10X727X')
        self.assertEqual(sent['par']['dm']['dmNum'], 504)
        self.assertEqual(sent['par']['dm']['Para'], ['p1', 'p2'])
        self.assertEqual(sent['par']['scriptType'], 4)
        # 空值键被填入方案 JSON，其余键原样提交
        self.assertEqual(sent['par']['rowData']['PLANID'], 'A-1')
        plan = json.loads(sent['par']['rowData']['PLANJSON'])
        self.assertEqual(plan['confirmedRank'], 2)
        self.assertEqual([r['rank'] for r in plan['results']], [1, 2])

    def test_build_submit_fills_exactly_one_empty_key(self):
        body = bt8_page.build_submit(FEATURE_CONF, 'tok', ['a'],
                                     {'PLANJSON': '', 'PLANID': 'A-1'},
                                     {'confirmedRank': 3})
        rows = body['par']['rowData']
        self.assertEqual(rows['PLANID'], 'A-1')
        self.assertEqual(json.loads(rows['PLANJSON']), {'confirmedRank': 3})

        with self.assertRaises(ValueError):
            bt8_page.build_submit(FEATURE_CONF, 'tok', [], {'PLANJSON': 'x'}, {})
        with self.assertRaises(ValueError):
            bt8_page.build_submit(FEATURE_CONF, 'tok', [], {'A': '', 'B': ''}, {})

    # ---------- 确认：结果判定 ----------

    def test_confirm_failure_with_invalid_link_token(self):
        UpstreamStub.reply = (200, {'isSuccess': 'false', 'message': '', 'action': '',
                                    'data': 'InvalidLinkToken'})
        status, data = self.post_confirm(self.confirm_request())
        self.assertEqual(status, 200)
        self.assertEqual(data.get('isSuccess'), 0)
        self.assertIn('登录已失效', data.get('message', ''))
        self.assertEqual(data.get('code'), 'InvalidLinkToken')

    def test_confirm_failure_with_action_prevent(self):
        UpstreamStub.reply = (200, {'isSuccess': 'true', 'action': 'prevent',
                                    'message': '需要人工确认'})
        _, data = self.post_confirm(self.confirm_request())
        self.assertEqual(data.get('isSuccess'), 0)
        self.assertEqual(data.get('message'), '需要人工确认')

    def test_confirm_failure_on_upstream_error(self):
        UpstreamStub.reply = (500, {'isSuccess': 'false', 'message': '上游处理失败'})
        _, data = self.post_confirm(self.confirm_request())
        self.assertEqual(data.get('isSuccess'), 0)
        self.assertIn('上游处理失败', data.get('message', ''))

    def test_confirm_network_error_is_reported(self):
        _, base = self.start_server(self.settings(f'http://127.0.0.1:{closed_port()}'))
        _, data = self.post_confirm(self.confirm_request(), base=base)
        self.assertEqual(data.get('isSuccess'), 0)
        self.assertIn('无法连接 TotalLINK', data.get('message', ''))

    # ---------- 确认：安全与入参校验 ----------

    def test_confirm_accepts_local_origin(self):
        _, data = self.post_confirm(self.confirm_request(), origin='http://127.0.0.1:8765')
        self.assertEqual(data.get('isSuccess'), 1, data.get('message'))
        self.assertEqual(len(UpstreamStub.received), 1)

    def test_confirm_rejects_foreign_origin(self):
        status, data = self.post_confirm(self.confirm_request(), origin='http://evil.example')
        self.assertEqual(status, 403)
        self.assertEqual(data.get('isSuccess'), 0)
        self.assertEqual(UpstreamStub.received, [])

    def test_confirm_requires_login_id(self):
        body = self.confirm_request()
        body.pop('loginID')
        status, data = self.post_confirm(body)
        self.assertEqual(status, 400)
        self.assertIn('登录凭据', data.get('message', ''))
        self.assertEqual(UpstreamStub.received, [])

    def test_confirm_rejects_bad_para(self):
        status, data = self.post_confirm(self.confirm_request(para={'a': 1}))
        self.assertEqual(status, 400)
        self.assertIn('para', data.get('message', ''))
        self.assertEqual(UpstreamStub.received, [])

    def test_confirm_rejects_bad_row_data(self):
        for row_data in ({'PLANJSON': 'x'}, {'A': '', 'B': ''}):
            status, data = self.post_confirm(self.confirm_request(rowData=row_data))
            self.assertEqual(status, 400)
            self.assertIn('rowData', data.get('message', ''))
        self.assertEqual(UpstreamStub.received, [])

    def test_confirm_rejects_missing_payload(self):
        body = self.confirm_request()
        body.pop('payload')
        status, data = self.post_confirm(body)
        self.assertEqual(status, 400)
        self.assertIn('payload', data.get('message', ''))

    # ---------- 配置 ----------

    def test_missing_settings_disables_confirm_only(self):
        settings = page_server.load_settings(os.path.join(ROOT, 'no-such-settings.json'))
        self.assertEqual(settings, {})
        _, base = self.start_server(settings)

        self.assertIn('"confirmEnabled": false', self.page(base=base))

        payload = dict(bt8_page.SAMPLE_PAYLOAD, maxtime=5)
        _, data = self.post_calc(payload, base=base)
        self.assertEqual(data.get('isSuccess'), 1, data.get('message'))

        _, data = self.post_confirm(self.confirm_request(), base=base)
        self.assertEqual(data.get('isSuccess'), 0)
        self.assertIn('未配置 TotalLINK 地址', data.get('message', ''))
        self.assertEqual(UpstreamStub.received, [])

    def test_shipped_settings(self):
        settings = page_server.load_settings()
        feature = settings['features']['bt8']
        self.assertEqual(feature['dmCode'], 'TMESMFGPLN10X727X')
        self.assertEqual(feature['dmNum'], 504)
        self.assertEqual(feature['scriptType'], 4)
        self.assertEqual(feature['api'], '/api/DataModel/linkDMGridRowSubmit')
        self.assertEqual(settings['totallink']['base_url'], 'https://link.bros.cn')
        self.assertEqual(settings['totallink']['timeout'], 30)

    def test_sample_payload_is_valid(self):
        """页面「载入示例」与 /api/bt8/sample 依赖的内置输入本身应能算出结果。"""
        payload = dict(bt8_page.SAMPLE_PAYLOAD, maxtime=5)
        data = json.loads(bt8_page.calc(json.dumps(payload)))
        self.assertEqual(data.get('isSuccess'), 1, data.get('message'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
