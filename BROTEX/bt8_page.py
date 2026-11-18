#!/usr/bin/env python3
"""
bt8 牵伸配色计算 —— 页面功能模块

本模块不再自带 HTTP 服务，由仓库根目录的 page_server.py 加载（见规格
docs/specs/bt8-page-totallink.md 第 8 节）。对外只暴露：

    PAGE_FILE        页面 HTML 路径
    calc(raw_text)   调用 bt8.linkrun，返回其结果 JSON 文本
    build_submit()   组装 TotalLINK linkDMGridRowSubmit 的请求体
    SAMPLE_PAYLOAD   /api/bt8/sample 的示例输入（本地调试用）

路由、超时、Origin 校验与 linkResult 判定都在 page_server.py。
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import bt8  # noqa: E402

PAGE_FILE = os.path.join(HERE, 'bt8_page.html')

# 页面「载入示例」与 /api/bt8/sample 用的输入
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


def calc(raw_text):
    """接收 /api/bt8/calc 的请求体原文，返回 bt8.linkrun 的结果 JSON 文本。"""
    return bt8.linkrun(raw_text)


def build_submit(feature_conf, login_id, para, row_data, payload):
    """组装 linkDMGridRowSubmit 的请求体（规格 7.3）。

    rowData 里恰好一个空字符串值，由这里填入 payload 的 JSON 文本；
    其余键原样提交。
    """
    rows = dict(row_data)
    empty = [key for key, value in rows.items() if value == '']
    if len(empty) != 1:
        raise ValueError(f'rowData 必须恰好有一个空值键，当前 {len(empty)} 个')

    rows[empty[0]] = json.dumps(payload, ensure_ascii=False,
                                separators=(',', ':'), allow_nan=False)
    return {
        'loginID': login_id,
        'par': {
            'dm': {
                'dmCode': feature_conf.get('dmCode'),
                'dmNum': feature_conf.get('dmNum'),
                'Para': list(para),
            },
            'scriptType': feature_conf.get('scriptType'),
            'rowData': rows,
        },
    }
