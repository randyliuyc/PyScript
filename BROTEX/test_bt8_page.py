#!/usr/bin/env python3
"""
bt8_page.py 接口测试

覆盖：
- 页面 HTML 正常返回
- POST /api/calc 调用 bt8.linkrun 并返回结果契约
- 输入错误、非法 JSON、未知路径的响应

运行：
    python BROTEX/test_bt8_page.py
依赖：numpy、scipy
"""
import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bt8_page  # noqa: E402


class Bt8PageTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.server = bt8_page.make_server('127.0.0.1', 0)
        cls.base = 'http://%s:%d' % cls.server.server_address[:2]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as resp:
            return resp.status, resp.headers.get('Content-Type', ''), resp.read().decode('utf-8')

    def post(self, payload=None, raw=None):
        body = raw if raw is not None else json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            self.base + '/api/calc', data=body,
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.read().decode('utf-8')

    # ---------- 页面 ----------

    def test_page_served(self):
        status, content_type, body = self.get('/')
        self.assertEqual(status, 200)
        self.assertIn('text/html', content_type)
        self.assertIn('牵伸配色计算', body)
        self.assertIn('/api/calc', body)

    def test_unknown_path_is_json_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.get('/nope')
        self.assertEqual(ctx.exception.code, 404)
        data = json.loads(ctx.exception.read().decode('utf-8'))
        self.assertEqual(data['isSuccess'], 0)

    # ---------- 计算接口 ----------

    def test_calc_returns_results(self):
        payload = dict(bt8_page.SAMPLE_PAYLOAD, maxtime=5)
        status, text = self.post(payload)
        self.assertEqual(status, 200)
        data = json.loads(text)
        self.assertEqual(data.get('isSuccess'), 1, data.get('message'))
        self.assertTrue(data.get('results'))
        first = data['results'][0]
        self.assertEqual(len(first['assign']), 8)
        for key in ('X1', 'X2', 'X3', 'cum_error', 'cum_error_pct',
                    'total_feed_speed_D', 'priority_ok', 'assign', 'colors'):
            self.assertIn(key, first)

    def test_calc_rejects_invalid_input(self):
        _, text = self.post({'data': []})
        data = json.loads(text)
        self.assertEqual(data.get('isSuccess'), 0)
        self.assertTrue(data.get('message'))

    def test_calc_rejects_malformed_json(self):
        _, text = self.post(raw=b'{not json')
        data = json.loads(text)
        self.assertEqual(data.get('isSuccess'), 0)
        self.assertIn('输入错误', data.get('message', ''))

    def test_calc_rejects_empty_body(self):
        _, text = self.post(raw=b'')
        data = json.loads(text)
        self.assertEqual(data.get('isSuccess'), 0)

    def test_sample_payload_is_valid(self):
        """服务端内置示例本身应能算出结果，页面「载入示例」依赖它。"""
        payload = dict(bt8_page.SAMPLE_PAYLOAD, maxtime=5)
        data = json.loads(bt8_page.bt8.linkrun(json.dumps(payload)))
        self.assertEqual(data.get('isSuccess'), 1, data.get('message'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
