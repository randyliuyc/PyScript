# -*- coding: utf-8 -*-
"""
用友 T+ 对接配置（Tamper / LinkPython 共享配置）。

说明：
- 应用凭证、认证接口、数据文件路径等均以 tplus_callback.py 为准，两处需保持同步。
- openToken 无需手动填写：由 tplus_callback.auto_get_token() 自动获取/刷新
    ① tplus_token.json 缓存未过期 → 直接复用（零接口调用）
    ② 已过期 → refreshToken 刷新 → appTicket+certificate 换新 → 触发平台重推
- 所有接口程序共用本文件，避免在各程序里重复维护敏感参数。
"""
import os

# --- 应用凭证（畅捷通开放平台，与 tplus_callback.py 一致） ---
APP_KEY = os.environ.get("CHANJET_APP_KEY", "6VNnRGdn")
APP_SECRET = os.environ.get("CHANJET_APP_SECRET", "8BED6CFB7777F56859C6B821DD33C315")

# --- API 基础地址 ---
# 认证/消息接口域名（与 tplus_callback.py 一致）
API_BASE = os.environ.get("CHANJET_API_BASE", "https://openapi.chanjet.com")
# T+ 业务 API 基础地址（业务接口挂在 /tplus/api/v2 下）。
# 本地部署版替换为本地服务地址（如 http://tamper.gnway.vip/tplus/api/v2）
BASE_URL = os.environ.get("CHANJET_BASE_URL", "https://openapi.chanjet.com/tplus/api/v2")

# --- 数据文件（与 tplus_callback.py 一致，均放脚本同目录） ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 与 callback 共用同一目录
TOKEN_FILE = "tplus_token.json"          # openToken 缓存（未过期直接复用）
TICKET_FILE = "app_ticket.txt"           # appTicket（回调服务收到推送后自动覆盖更新）
CERT_FILE = "tplus_certificate.txt"      # 自建应用软证书
CODE_FILE = "oauth_code.txt"             # OAuth 授权码

# --- 消息接收/解密（与 tplus_callback.py 一致） ---
# 开放平台「环境密钥/消息密钥」(16位)，必须与开放平台后台配置完全一致
SECRET_KEY = "1234567890123456"
# 允许通过 HTTP 直接访问的 txt 白名单（仅限开放平台「可信域名验证」文件）
PUBLIC_TXT_FILES = ["CHANJET_CHECK.txt"]
# 调试日志文件(message_raw/decrypt_error等)大小上限(字节)，超过后自动轮转保留 .old
MAX_LOG_BYTES = 1024 * 1024  # 1MB

# --- 认证接口（与 tplus_callback.py 一致，token 获取逻辑在 callback 内） ---
# 手动触发 appTicket 重新推送（回调服务未收到推送/过期时主动触发）
RESEND_URL = API_BASE + "/auth/appTicket/resend"
RESEND_WAIT_SECONDS = 15  # 触发 resend 后等待平台重新推送的超时秒数

# --- token 获取方式：通过回调服务的 /getToken 接口（业务程序可在任意机器运行） ---
# 固定接口: http://124.71.144.80:7077/getToken  (tplus_callback.py 提供)
# 调用时会自动附上 key=totalLINK+当前日期(yyyyMMDDHH)，与回调服务校验规则一致
TOKEN_URL = os.environ.get("CHANJET_TOKEN_URL", "http://124.71.144.80:7077/getToken")
TOKEN_KEY_PREFIX = "totalLINK"  # key 前缀，+年月日小时(24小时制)

# --- HTTP 参数 ---
TIMEOUT = 10  # 秒
MAX_RETRY = 3  # 最大自动重试次数（含 token 过期自动重取重试）
