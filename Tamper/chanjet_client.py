# -*- coding: utf-8 -*-
"""
用友 T+ 开放平台共享客户端（token 通过回调服务 /getToken 接口获取）。

职责：
- token 自动获取：通过固定接口 config.TOKEN_URL 获取（如 http://124.71.144.80:7077/getToken）
    接口由 tplus_callback.py 提供，内部自带缓存：token 未过期零接口、过期自动刷新/换新
- 客户端本地仅做内存缓存：token 未过期直接复用（零 HTTP 调用），过期才重新请求
- 组装请求头（openToken / appKey / appSecret）
- 通过 urllib.request 发起同步 POST 调用（零依赖）
- 统一解析响应，识别成功/失败，token 失效自动重取重试

业务程序可在任意机器运行，只需能访问 config.TOKEN_URL（回调服务所在机器）。
配置以 config.py 为准（与 tplus_callback.py 同步维护）。
"""
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

import config


class ChanjetError(Exception):
    """用友接口调用失败异常。message 为可读错误信息。"""


# ========== token 获取（通过回调服务 /getToken 接口） ==========

def _build_token_key(offset_hours=0):
    """生成 /getToken 的校验 key: totalLINK + 当前日期(yyyyMMDDHH 24小时制)

    与 tplus_callback.py 的 build_token_key() 规则一致，兼容前1小时边界。
    """
    d = datetime.now() - timedelta(hours=offset_hours)
    return config.TOKEN_KEY_PREFIX + d.strftime("%Y%m%d%H")


_token_cache = {"token": None, "expire_at": None}


def get_token():
    """获取有效 openToken：内存缓存未过期直接复用（零 HTTP 调用），过期才请求接口。"""
    now = datetime.now()
    if _token_cache["token"] and _token_cache["expire_at"] and now < _token_cache["expire_at"]:
        return _token_cache["token"]

    url = config.TOKEN_URL
    sep = "&" if "?" in url else "?"
    url += sep + "key=" + _build_token_key()
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=config.TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        raise ChanjetError(f"获取 openToken 失败: {e}")

    if data.get("code") != 200:
        raise ChanjetError(f"获取 openToken 失败: {data.get('message') or data}")

    token = data.get("token")
    if not token:
        raise ChanjetError(f"获取 openToken 失败: 响应缺少 token: {data}")

    # 本地缓存到过期时间（未过期期间零 HTTP 调用）
    _token_cache["token"] = token
    _token_cache["expire_at"] = None
    expire_at = data.get("expireAt")
    if expire_at:
        try:
            _token_cache["expire_at"] = datetime.strptime(expire_at, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            _token_cache["expire_at"] = None
    if _token_cache["expire_at"] is None:
        # 兜底：未知过期时间时按 6 天缓存
        _token_cache["expire_at"] = now + timedelta(days=6)
    return token


def _invalidate_token_cache():
    """token 失效时清掉本地缓存，下次调用自动重新获取。"""
    _token_cache["token"] = None
    _token_cache["expire_at"] = None
    print("[Chanjet] 已清除本地 token 缓存，将自动重新获取")


def get_headers():
    """构造请求头。openToken 每次自动获取（未过期时零 HTTP 调用）。"""
    token = get_token()
    return {
        "Content-Type": "application/json",
        "openToken": token,
        "appKey": config.APP_KEY,
        "appSecret": config.APP_SECRET,
    }


# ========== 业务接口调用 ==========

_TOKEN_EXPIRED_HINTS = (
    "token", "过期", "失效", "expired", "invalid", "unauthorized",
    "41001", "41002", "42001", "42002",
)


def _is_token_expired(result):
    """判断响应是否为 openToken 过期/失效（据此自动重取重试）"""
    msg = str(result.get("message") or result.get("hint") or "").lower()
    code = str(result.get("code") or "")
    return any(h in msg for h in _TOKEN_EXPIRED_HINTS) or code in ("401", "41001", "41002", "42001", "42002")


def call_api(endpoint, payload, timeout=None, retries=None):
    """
    同步调用用友 T+ API（token 自动获取，过期自动重取重试）。

    参数:
        endpoint: 接口路径，如 "/saleOrder/Create"
        payload:  请求体 dict（将作为 JSON 发送）
        timeout:  超时秒数，默认取 config.TIMEOUT
        retries:  失败重试次数（含 token 过期自动重取），默认取 config.MAX_RETRY

    返回:
        解析后的响应 dict。

    抛异常:
        ChanjetError: 网络错误、HTTP 非 2xx、或业务 code 非成功时抛出
    """
    timeout = timeout or config.TIMEOUT
    if retries is None:
        retries = config.MAX_RETRY
    url = config.BASE_URL.rstrip("/") + endpoint

    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers=get_headers(),
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            # HTTP 401/403 多为 token 过期/鉴权失败：清缓存重取后重试
            if e.code in (401, 403) and attempt < retries:
                _invalidate_token_cache()
                continue
            raise ChanjetError(f"HTTP {e.code}: {body}")
        except urllib.error.URLError as e:
            raise ChanjetError(f"网络错误: {e.reason}")
        except Exception as e:
            raise ChanjetError(f"请求异常: {e}")

        try:
            result = json.loads(body)
        except json.JSONDecodeError:
            raise ChanjetError(f"响应非合法 JSON: {body[:500]}")

        # 创建类接口（如 saleOrder/Create、purchaseOrder/Create）成功时
        # T+ 返回 HTTP 200 + null body，视为成功（data 为空，无业务状态码）
        if result is None:
            return {"code": 200, "data": None}

        # 校验业务状态码
        code = result.get("code")
        if code is not None and code != 200:
            # token 过期/失效：清缓存重取后重试
            if _is_token_expired(result) and attempt < retries:
                _invalidate_token_cache()
                continue
            msg = result.get("message") or result.get("hint") or str(result)
            raise ChanjetError(f"业务失败(code={code}): {msg}")

        return result

    raise ChanjetError("重试次数用尽，仍失败")
