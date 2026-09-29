"""训练和检测共用的离线网址解析与特征提取模块。"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlsplit

FEATURE_SCHEMA_VERSION = 2
FEATURE_LABELS = {
    "url_length": "网址总长度",
    "host_length": "主机名长度",
    "path_length": "路径长度",
    "query_length": "查询参数长度",
    "host_label_count": "主机名层数",
    "path_depth": "路径层数",
    "query_param_count": "查询参数个数",
    "digit_count": "网址数字个数",
    "host_digit_count": "主机名数字个数",
    "dot_count": "点号个数",
    "hyphen_count": "连字符个数",
    "at_count": "@ 符号个数",
    "percent_count": "百分号个数",
    "equal_count": "等号个数",
    "ampersand_count": "& 符号个数",
    "is_ip_host": "主机名是 IP 地址",
    "is_https": "解析使用 HTTPS",
    "has_explicit_port": "显式指定端口",
    "host_hyphen_count": "主机名连字符个数",
    "host_max_label_length": "主机名最长一层的长度",
    "host_digit_ratio": "主机名数字比例",
    "host_entropy": "主机名字符熵",
    "has_punycode": "含国际化域名编码",
    "has_userinfo": "网址含用户信息段",
    "suspicious_token_count": "网址风险提示词个数",
    "host_suspicious_token_count": "主机名风险提示词个数",
    "has_fragment": "含片段标识",
    "scheme_missing": "原输入未声明协议",
}
FEATURE_NAMES = tuple(FEATURE_LABELS)
FeatureValue = int | float

_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_HOST_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_TOKENS = re.compile(
    r"login|signin|verify|verification|secure|account|update|confirm|password|wallet"
)


class URLValidationError(ValueError):
    """输入内容不是可识别的 HTTP(S) 网址。"""


@dataclass(frozen=True)
class URLFeatures:
    original: str
    hostname: str
    display_url: str
    values: dict[str, FeatureValue]
    warning: str = ""


def extract_url_features(value: object) -> URLFeatures:
    """只解析网址文本，不访问网页，也不查询 DNS。"""
    if not isinstance(value, str) or not value.strip():
        raise URLValidationError("网址不能为空")
    raw = value.strip()
    if len(raw) > 2048:
        raise URLValidationError("网址超过 2048 个字符")
    if any(char.isspace() or ord(char) < 32 for char in raw):
        raise URLValidationError("网址不能含空格或控制字符")

    bare_host_port = bool(re.match(r"^[^/:?#]+\.[^/:?#]+:\d+(?:[/?#]|$)", raw))
    scheme_missing = not bool(_SCHEME.match(raw)) or bare_host_port
    if not scheme_missing:
        scheme = raw.split(":", 1)[0].lower()
        if scheme not in {"http", "https"}:
            raise URLValidationError("仅支持 HTTP 或 HTTPS 网址")
        parseable = raw
    elif raw.startswith("//"):
        parseable = "https:" + raw
    else:
        parseable = "https://" + raw

    try:
        parts = urlsplit(parseable)
        hostname = parts.hostname
        port = parts.port
    except ValueError as exc:
        raise URLValidationError("网址主机名或端口格式无效") from exc
    if not hostname:
        raise URLValidationError("网址缺少主机名")
    try:
        host = hostname.encode("idna").decode("ascii").lower().rstrip(".")
    except UnicodeError as exc:
        raise URLValidationError("主机名编码无效") from exc
    if len(host) > 253:
        raise URLValidationError("主机名过长")
    try:
        ipaddress.ip_address(host)
        is_ip_host = 1
    except ValueError:
        is_ip_host = 0
        labels = host.split(".")
        if len(labels) < 2 or any(not _HOST_LABEL.fullmatch(label) for label in labels):
            raise URLValidationError("主机名格式无效")
        if re.fullmatch(r"[0-9.]+", host):
            raise URLValidationError("IP 地址格式无效")
    if port is not None and not 1 <= port <= 65535:
        raise URLValidationError("端口号超出有效范围")

    query_pairs = parse_qsl(parts.query, keep_blank_values=True)
    host_labels = host.split(".") if not is_ip_host else [host]
    host_digits = sum(char.isdigit() for char in host)
    entropy = -sum(
        (count / len(host)) * math.log2(count / len(host)) for count in Counter(host).values()
    )
    features: dict[str, FeatureValue] = {
        "url_length": len(raw),
        "host_length": len(host),
        "path_length": len(parts.path),
        "query_length": len(parts.query),
        "host_label_count": len(host_labels),
        "path_depth": len([segment for segment in parts.path.split("/") if segment]),
        "query_param_count": len(query_pairs),
        "digit_count": sum(char.isdigit() for char in raw),
        "host_digit_count": host_digits,
        "dot_count": raw.count("."),
        "hyphen_count": raw.count("-"),
        "at_count": raw.count("@"),
        "percent_count": raw.count("%"),
        "equal_count": raw.count("="),
        "ampersand_count": raw.count("&"),
        "is_ip_host": is_ip_host,
        "is_https": int(parts.scheme.lower() == "https"),
        "has_explicit_port": int(port is not None),
        "host_hyphen_count": host.count("-"),
        "host_max_label_length": max(map(len, host_labels)),
        "host_digit_ratio": host_digits / len(host),
        "host_entropy": entropy,
        "has_punycode": int(any(label.startswith("xn--") for label in host_labels)),
        "has_userinfo": int(parts.username is not None),
        "suspicious_token_count": len(_TOKENS.findall(raw.lower())),
        "host_suspicious_token_count": len(_TOKENS.findall(host)),
        "has_fragment": int(bool(parts.fragment)),
        "scheme_missing": int(scheme_missing),
    }
    visible_host = f"[{host}]" if ":" in host else host
    visible_port = f":{port}" if port is not None else ""
    hidden_query = "&".join(f"{key}=***" for key, _ in query_pairs)
    display = f"{parts.scheme.lower()}://{visible_host}{visible_port}{parts.path}"
    if parts.query:
        display += "?" + (hidden_query or "***")
    warning = (
        "未提供协议，按 HTTPS 解析；这只是默认假设，建议填写完整网址。" if scheme_missing else ""
    )
    return URLFeatures(raw, host, display, features, warning)
