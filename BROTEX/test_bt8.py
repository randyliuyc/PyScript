#!/usr/bin/env python3
"""
bt8.py 回归测试

覆盖：
- 输入/输出契约（isSuccess、严格 JSON、紧凑输出、边界输入）
- 数值自洽（speed 与 x、cum_error 与明细、种子 dev 与 X）
- 结果可复现
- POSITION 桶位约束（单桶位、多桶位）
- 多颜色与搜索框角落解（8 色等分）
- 搜索预算参数 maxtime 与 debug 开关

运行：
    python BROTEX/test_bt8.py
依赖：numpy、scipy
"""
import contextlib
import io
import json
import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bt8  # noqa: E402

DATA = [
    {"MFMLIN": 20, "MFMDES": "B024W J ", "MFMSHO": "B024W",
     "MATRATCALC": 10.0, "PRIORITY": 0, "POSITION": ""},
    {"MFMLIN": 30, "MFMDES": "G004W J ", "MFMSHO": "G004W",
     "MATRATCALC": 10.0, "PRIORITY": 0, "POSITION": ""},
    {"MFMLIN": 40, "MFMDES": "G008W QUA JC ", "MFMSHO": "G008W",
     "MATRATCALC": 80.0, "PRIORITY": 0, "POSITION": ""},
]
PAYLOAD = {"xmin": 1.03, "xmax": 3.5, "data": DATA}

# 5 色样例：与 bt8_page.SAMPLE_PAYLOAD 一致；最好方案单色相对误差约 4.4%，用于测试阈值剔除
SAMPLE_DATA = [
    {"MFMLIN": 20, "MFMDES": "FWP 特白条", "MATRATCALC": 28.0,
     "PRIORITY": 0, "POSITION": ""},
    {"MFMLIN": 112, "MFMDES": "再生白涤纶", "MATRATCALC": 25.03,
     "PRIORITY": 0, "POSITION": ""},
    {"MFMLIN": 120, "MFMDES": "白棉条", "MATRATCALC": 8.49,
     "PRIORITY": 0, "POSITION": ""},
    {"MFMLIN": 150, "MFMDES": "FW 特白条", "MATRATCALC": 7.8,
     "PRIORITY": 0, "POSITION": ""},
    {"MFMLIN": 99999, "MFMDES": "HY 条子", "MATRATCALC": 5.68,
     "PRIORITY": 0, "POSITION": ""},
]
SAMPLE_PAYLOAD = {"xmin": 1.03, "xmax": 3.5, "maxtime": 5, "data": SAMPLE_DATA}


def run(payload):
    """调用 linkrun 并把返回的 JSON 文本解析成对象"""
    return json.loads(bt8.linkrun(json.dumps(payload)))


def parse_color_line(line):
    """解析 '名称: 目标 -> 实际 (相对误差%)'，返回 (名称, 实际, 相对误差)

    最后一个括号里的数字是相对误差：|实际 - 目标| / 目标 × 100。
    """
    left, right = line.split(": ", 1)
    _, rest = right.split(" -> ")
    actual, err = rest.split(" (")
    return left.strip(), float(actual), float(err.rstrip(")%"))


class Bt8RegressionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw = bt8.linkrun(json.dumps(PAYLOAD))
        cls.out = json.loads(cls.raw)

    # ---------- 契约 ----------

    def test_success_contract(self):
        self.assertEqual(self.out.get("isSuccess"), 1)
        self.assertTrue(self.out.get("results"))

    def test_output_is_strict_json(self):
        def reject_constant(value):
            raise AssertionError(f"输出包含非法 JSON 常量: {value}")

        json.loads(self.raw, parse_constant=reject_constant)

    def test_output_is_compact(self):
        self.assertNotIn('\n  "', self.raw, "输出不应使用 indent=2")

    def test_input_validation(self):
        cases = {
            "空 data": {"data": []},
            "缺 data": {},
            "比例全 0": {"data": [{"MFMLIN": 1, "MFMDES": "X", "MATRATCALC": 0}]},
            "非法 POSITION": {"data": [{"MFMLIN": 1, "MFMDES": "X", "MATRATCALC": 1, "POSITION": "Z"}]},
            "xmin<1": {"xmin": 0.8, "data": DATA},
            "xmax<=xmin": {"xmin": 2, "xmax": 1, "data": DATA},
            "缺 MFMLIN": {"data": [{"MFMDES": "X", "MATRATCALC": 1}]},
            "maxtime<=0": {"maxtime": 0, "data": DATA},
        }
        for name, payload in cases.items():
            with self.subTest(name=name):
                result = run(payload)
                self.assertEqual(result.get("isSuccess"), 0, f"{name}: {result}")
                self.assertTrue(result.get("message"))
                self.assertEqual(result.get("errorCode"), "input_invalid")
                self.assertIn("detail", result.get("errorParams") or {})

    # ---------- 数值自洽 ----------

    def test_speed_consistent_with_displayed_x(self):
        for result in self.out["results"]:
            D = result["total_feed_speed_D"]
            for entry in result["assign"]:
                expected = round((1.0 / entry["x"]) / D * 100.0, 4)
                self.assertLess(abs(expected - entry["speed"]), 1e-4,
                                f"{entry['bucket']}: x={entry['x']} speed={entry['speed']}")

    def test_cum_error_keeps_absolute_meaning(self):
        """cum_error 仍是各颜色 |实际-目标| 百分点差之和（写入 MFGMATBTS.CUMVAR）。"""
        for result in self.out["results"]:
            detail_sum = 0.0
            for line in result["colors"]:
                target = float(line.split(": ", 1)[1].split(" -> ")[0])
                actual = float(line.split(" -> ")[1].split(" (")[0])
                detail_sum += abs(actual - target)
            # 明细里的目标/实际保留 2 位小数，每色最多带来 0.01 的舍入差
            tolerance = 0.02 * len(result["colors"]) + 0.01
            self.assertLess(abs(result["cum_error"] * 100 - detail_sum), tolerance)
            self.assertLess(abs(result["cum_error_pct"] - detail_sum), tolerance)
            self.assertEqual(result["cum_error"], round(result["cum_error"], 6))

    def test_cum_rel_error_matches_details(self):
        """cum_rel_error = 各颜色相对误差之和（括号里的数字，2 位小数）。"""
        for result in self.out["results"]:
            detail_sum = sum(
                float(line.rsplit("(", 1)[1].rstrip("%)")) for line in result["colors"]
            )
            tolerance = 0.005 * len(result["colors"]) + 0.01
            self.assertLess(abs(result["cum_rel_error"] * 100 - detail_sum), tolerance)
            self.assertLess(abs(result["cum_rel_error_pct"] - detail_sum), tolerance)
            self.assertEqual(result["cum_rel_error"], round(result["cum_rel_error"], 6))

    def test_color_error_is_relative(self):
        """colors[] 括号里是相对误差，而不是绝对百分点差。"""
        for result in self.out["results"]:
            for line in result["colors"]:
                _, actual, err = parse_color_line(line)
                target = float(line.split(": ", 1)[1].split(" -> ")[0])
                if target > 0:
                    expected = abs(actual - target) / target * 100
                    self.assertAlmostEqual(err, expected, places=1, msg=line)

    def test_no_plan_exceeds_normal_threshold(self):
        """普通色相对误差超阈值的方案不会出现在结果里。"""
        for result in self.out["results"]:
            for line in result["colors"]:
                _, _, err = parse_color_line(line)
                self.assertLessEqual(err, bt8.NORMAL_REL_THRESHOLD * 100 + 0.01, line)

    def test_seed_dev_matches_reported_x(self):
        targets = np.array([item["MATRATCALC"] for item in DATA], float)
        targets = targets / targets.sum()
        bounds = {"x1": (1.03, 3.5), "x2": (1.03, 3.5), "x3": (1.03, 3.5)}
        seeds = bt8.find_seeds_scipy(targets, len(DATA), {}, bounds, rng=random.Random(0))
        self.assertTrue(seeds)
        for seed in seeds:
            w = np.array([1 / seed["X1"], 1.0, 1 / seed["X2"], 1 / seed["X3"]])
            D = float(np.sum(w * bt8.GROUP_SIZES))
            res = bt8.fast_backtrack(targets, w, D, len(DATA), {})
            matched = [r[1] for r in res if list(r[0]) == list(seed["assign"])]
            self.assertTrue(matched, f"找不到与种子分配对应的结果: {seed['assign']}")
            self.assertAlmostEqual(matched[0], seed["dev"], places=12)

    # ---------- 可复现 ----------

    def test_deterministic(self):
        other = bt8.linkrun(json.dumps(PAYLOAD))
        self.assertEqual(self.raw, other)

    def test_seed_parameter_reproducible(self):
        payload = dict(PAYLOAD, seed=123)
        self.assertEqual(bt8.linkrun(json.dumps(payload)), bt8.linkrun(json.dumps(payload)))

    # ---------- POSITION 约束 ----------

    def test_position_constraints_respected(self):
        data = [dict(item) for item in DATA]
        data[0]["POSITION"] = "AB"
        data[1]["POSITION"] = "B"
        result = run(dict(PAYLOAD, data=data))
        self.assertEqual(result.get("isSuccess"), 1, result)
        allowed = {0: {0, 1}, 1: {1}}
        color_to_idx = {item["MFMLIN"]: i for i, item in enumerate(data)}
        for plan in result["results"]:
            for bucket_idx, entry in enumerate(plan["assign"]):
                color_idx = color_to_idx[entry["color"]]
                if color_idx in allowed:
                    self.assertIn(bucket_idx, allowed[color_idx],
                                  f"color {color_idx} 被放到 {entry['bucket']}")

    def test_position_multi_bucket(self):
        """POSITION='ACF'：该颜色必须同时出现在 A、C、F，且不能出现在别处。"""
        data = [
            {"MFMLIN": 10, "MFMDES": "A", "MATRATCALC": 30, "PRIORITY": 0, "POSITION": "ACF"},
            {"MFMLIN": 20, "MFMDES": "B", "MATRATCALC": 30, "PRIORITY": 0, "POSITION": ""},
            {"MFMLIN": 30, "MFMDES": "C", "MATRATCALC": 40, "PRIORITY": 0, "POSITION": ""},
        ]
        result = run({"data": data, "maxtime": 5})
        self.assertEqual(result.get("isSuccess"), 1, result.get("message"))
        self.assertTrue(result["results"])
        for plan in result["results"]:
            positions = {entry["bucket"]: entry["color"] for entry in plan["assign"]}
            for bucket in "ACF":
                self.assertEqual(positions.get(bucket), 10, f"颜色 10 应出现在 {bucket}")
            for bucket, color in positions.items():
                if color == 10:
                    self.assertIn(bucket, "ACF")
            for line in plan["colors"]:
                _, _, err = parse_color_line(line)
                self.assertLessEqual(err, bt8.NORMAL_REL_THRESHOLD * 100 + 0.01, line)

    def test_many_colors_corner_solution(self):
        """8 色等分：解在搜索框角落；累计误差会超过单色容差，但单色都应达标。"""
        data = [
            {"MFMLIN": i * 10, "MFMDES": f"C{i * 10}", "MATRATCALC": 100,
             "PRIORITY": 0, "POSITION": ""}
            for i in range(1, 9)
        ]
        result = run({"data": data, "maxtime": 5})
        self.assertEqual(result.get("isSuccess"), 1, result.get("message"))
        self.assertTrue(result["results"])
        for plan in result["results"]:
            for line in plan["colors"]:
                _, _, err = parse_color_line(line)
                self.assertLessEqual(err, bt8.NORMAL_REL_THRESHOLD * 100 + 0.01, line)

    def test_small_ratio_color_reports_no_solution(self):
        """0.71% 这类低于结构最小单桶比例的色比：不应返回方案。

        绝对误差下它只差 0.71 个百分点，会被当成合格；相对误差是 100%。
        """
        data = [
            {"MFMLIN": 20, "MFMDES": "FWP 特白条", "MATRATCALC": 25.68,
             "PRIORITY": 0, "POSITION": ""},
            {"MFMLIN": 102, "MFMDES": "EH007", "MATRATCALC": 0.71,
             "PRIORITY": 0, "POSITION": ""},
            {"MFMLIN": 112, "MFMDES": "再生白涤纶", "MATRATCALC": 30.0,
             "PRIORITY": 0, "POSITION": ""},
            {"MFMLIN": 120, "MFMDES": "白棉条", "MATRATCALC": 10.71,
             "PRIORITY": 0, "POSITION": ""},
            {"MFMLIN": 130, "MFMDES": "BHHH", "MATRATCALC": 25.0,
             "PRIORITY": 0, "POSITION": ""},
            {"MFMLIN": 150, "MFMDES": "FW 特白条", "MATRATCALC": 7.9,
             "PRIORITY": 0, "POSITION": ""},
        ]
        result = run({"data": data})
        self.assertEqual(result.get("isSuccess"), 0, result)
        self.assertEqual(result.get("results"), [])
        message = result.get("message", "")
        self.assertIn("EH007", message)
        self.assertIn("0.71%", message)
        self.assertIn("低于", message)
        # 页面按 errorCode/errorParams 渲染中越文案
        self.assertEqual(result.get("errorCode"), "color_below_structural_min")
        params = result.get("errorParams") or {}
        self.assertEqual(params.get("threshold_pct"), 5)
        self.assertAlmostEqual(params.get("min_pct"), 4.43, places=2)
        self.assertTrue(any("EH007" in c for c in params.get("colors", [])))

    def test_small_ratio_is_never_silently_zero(self):
        """可实现的色比都必须真的分配到桶位；不会出现「目标 >0、实际 0」的方案。"""
        for plan in self.out["results"]:
            for line in plan["colors"]:
                _, actual, _ = parse_color_line(line)
                target = float(line.split(": ", 1)[1].split(" -> ")[0])
                if target > 0:
                    self.assertGreater(actual, 0.0, line)

    def test_plans_over_normal_threshold_are_rejected_with_reason(self):
        """普通色相对误差都超标时：不返回方案，并说明最接近通过的那一条。"""
        original = bt8.NORMAL_REL_THRESHOLD
        try:
            bt8.NORMAL_REL_THRESHOLD = 0.0001   # 0.01%，现有方案必然超标
            result = run(SAMPLE_PAYLOAD)
        finally:
            bt8.NORMAL_REL_THRESHOLD = original
        self.assertEqual(result.get("isSuccess"), 0, result)
        self.assertEqual(result.get("results"), [])
        self.assertIn("没有方案满足单色相对误差阈值", result.get("message", ""))
        self.assertIn("相对误差", result.get("message", ""))
        self.assertEqual(result.get("errorCode"), "no_plan_meets_threshold")
        params = result.get("errorParams") or {}
        self.assertEqual(params.get("threshold_pct"), 0.01)
        self.assertGreater(params.get("rejected", 0), 0)
        for key in ("name", "target_pct", "actual_pct", "rel_pct"):
            self.assertIn(key, params)

    def test_priority_miss_keeps_plan_but_flags_it(self):
        """优先色未达标仍保留方案（排在后面），不像普通色那样直接剔除。"""
        original = bt8.PRIORITY_REL_THRESHOLD
        try:
            bt8.PRIORITY_REL_THRESHOLD = 0.0001
            data = [dict(item, PRIORITY=1) for item in SAMPLE_DATA]
            result = run({"xmin": 1.03, "xmax": 3.5, "maxtime": 5, "data": data})
        finally:
            bt8.PRIORITY_REL_THRESHOLD = original
        self.assertEqual(result.get("isSuccess"), 1, result.get("message"))
        self.assertTrue(result["results"])
        for plan in result["results"]:
            self.assertFalse(plan["priority_ok"])
            self.assertTrue(plan["warnings"])

    def test_infeasible_position_reports_error(self):
        data = [
            {"MFMLIN": 10, "MFMDES": "A", "MATRATCALC": 10, "POSITION": "A"},
            {"MFMLIN": 20, "MFMDES": "B", "MATRATCALC": 10, "POSITION": "A"},
            {"MFMLIN": 30, "MFMDES": "C", "MATRATCALC": 80, "POSITION": ""},
        ]
        result = run({"data": data})
        self.assertEqual(result.get("isSuccess"), 0)
        self.assertIn("POSITION", result.get("message", ""))
        self.assertEqual(result.get("errorCode"), "position_infeasible")
        self.assertGreater((result.get("errorParams") or {}).get("rejected", 0), 0)

    def test_order_in_groups(self):
        self.assertEqual(
            bt8.order_in_groups([0, 1, 2, 2, 3, 3, 4, 4], {0: [1], 1: [0]}),
            [1, 0, 2, 2, 3, 3, 4, 4],
        )
        self.assertIsNone(bt8.order_in_groups([0, 1, 2, 2, 3, 3, 4, 4], {0: [0], 1: [0]}))

    # ---------- 运行参数 ----------

    def test_maxtime_is_honored(self):
        result = run(dict(PAYLOAD, maxtime=0.3))
        self.assertEqual(result.get("isSuccess"), 1)
        self.assertTrue(result.get("results"))

    def test_stdout_silent_by_default(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            run(PAYLOAD)
        self.assertEqual(buffer.getvalue(), "")

    def test_debug_enables_diagnostics(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            run(dict(PAYLOAD, debug=True))
        self.assertTrue(buffer.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
