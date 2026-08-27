# -*- coding: utf-8 -*-
"""
用友 T+ 开放平台共享客户端。

职责：
- 组装请求头（openToken / appKey / appSecret）
- 通过 urllib.request 发起同步 POST 调用（零依赖）
- 统一解析响应，识别成功/失败
- 预留 Token 自动获取/刷新扩展点（当前从 config 读占位值）

被各接口程序（如 sale_order.py）复用，避免认证与 HTTP 逻辑重复。
"""
import json
import urllib.request
import urllib.error

import config


class ChanjetError(Exception):
    """用友接口调用失败异常。message 为可读错误信息。"""


def get_headers():
    """构造请求头。openToken 当前为占位值，后续认证任务自动获取。"""
    if not config.OPEN_TOKEN:
        raise ChanjetError("未配置 openToken，请先接入认证自动获取任务")
    return {
        "Content-Type": "application/json",
        "openToken": config.OPEN_TOKEN,
        "appKey": config.APP_KEY,
        "appSecret": config.APP_SECRET,
    }


def call_api(endpoint, payload, timeout=None):
    """
    同步调用用友 T+ API。

    参数:
        endpoint: 接口路径，如 "/saleOrder/Create"
        payload:  请求体 dict（将作为 JSON 发送）
        timeout:  超时秒数，默认取 config.TIMEOUT

    返回:
        解析后的响应 dict。

    抛异常:
        ChanjetError: 网络错误、HTTP 非 2xx、或业务 code 非成功时抛出
    """
    timeout = timeout or config.TIMEOUT
    url = config.BASE_URL.rstrip("/") + endpoint

    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=get_headers(),
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        # 非 2xx：读取错误体尽量给出可读信息
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise ChanjetError(f"HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise ChanjetError(f"网络错误: {e.reason}")
    except Exception as e:
        raise ChanjetError(f"请求异常: {e}")

    try:
        result = json.loads(body)
    except json.JSONDecodeError:
        raise ChanjetError(f"响应非合法 JSON: {body[:500]}")

    # 校验业务状态码
    code = result.get("code")
    if code is not None and code != 200:
        msg = result.get("message") or result.get("hint") or str(result)
        raise ChanjetError(f"业务失败(code={code}): {msg}")

    return result
