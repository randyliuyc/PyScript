# -*- coding: utf-8 -*-
"""
用友 T+ 对接配置（Tamper / LinkPython 共享配置）。

当前 token 为占位值：认证自动获取/刷新作为独立任务实现，后续写入真实 openToken。
所有接口程序共用本文件，避免在各程序里重复维护敏感参数。
"""
import os

# --- 应用凭证（畅捷通开放平台） ---
# AppKey / AppSecret 来自开通权限后的应用凭证（参考：TPlus 应用商店应用）
APP_KEY = os.environ.get("CHANJET_APP_KEY", "5MrbTAWT")
APP_SECRET = os.environ.get("CHANJET_APP_SECRET", "C1CD580BF686309CFE09D7F8D3A596BD")

# --- API 基础地址 ---
# 云版公共地址。若为本地部署版，替换为本地服务地址（如 http://tamper.gnway.vip/tplus/api/v2）
BASE_URL = os.environ.get("CHANJET_BASE_URL", "https://openapi.chanjet.com/tplus/api/v2")

# --- 调用凭证 openToken（占位，后续由认证任务自动获取/刷新） ---
OPEN_TOKEN = os.environ.get(
    "CHANJET_OPEN_TOKEN",
    "",  # TODO: 认证自动获取任务接入后写入真实 token
)

# --- 认证接口（后续认证任务使用） ---
AUTH_URL = os.environ.get(
    "CHANJET_AUTH_URL", "https://openapi.chanjet.com/auth/v2/getToken"
)

# --- HTTP 参数 ---
TIMEOUT = 10  # 秒
MAX_RETRY = 3  # 最大自动重试次数
