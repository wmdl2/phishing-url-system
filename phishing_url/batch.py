"""批量 CSV 校验与导出，独立于网页框架以便测试。"""

from __future__ import annotations

import csv
import io

MAX_ROWS = 1000
MAX_BYTES = 5 * 1024 * 1024


class BatchValidationError(ValueError):
    """批量输入文件不满足约定。"""


def read_batch_csv(content: bytes) -> list[str | None]:
    if len(content) > MAX_BYTES:
        raise BatchValidationError("CSV 文件不能超过 5 MB")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeError as exc:
        raise BatchValidationError("CSV 必须使用 UTF-8 编码") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    try:
        headers = reader.fieldnames
        if headers is None or "url" not in headers:
            raise BatchValidationError("CSV 缺少 url 列")
        if len(set(headers)) != len(headers):
            raise BatchValidationError("CSV 不能含重复的列名")
        urls = []
        for row in reader:
            if None in row:
                raise BatchValidationError("CSV 某行的列数超过表头列数")
            urls.append(row.get("url"))
            if len(urls) > MAX_ROWS:
                raise BatchValidationError("CSV 最多包含 1,000 行数据")
        if not urls:
            raise BatchValidationError("CSV 至少包含 1 行数据")
        return urls
    except csv.Error as exc:
        raise BatchValidationError(f"CSV 格式无效：{exc}") from exc


def format_reasons(prediction: dict) -> str:
    reasons = prediction["reasons_for"][:2] or prediction["reasons_against"][:2]
    return "；".join(
        f"{item['feature']}={item['value']}（贡献 {item['contribution']:+.2f}）" for item in reasons
    )


def result_rows(predictions: list[dict]) -> list[dict]:
    return [
        {
            "url": item["url"],
            "status": item["status"],
            "risk_score": item.get("risk_score"),
            "reason": format_reasons(item) if item["status"] != "格式错误" else "",
            "warning": item.get("warning", ""),
            "error": item.get("error", ""),
        }
        for item in predictions
    ]


def export_csv(rows: list[dict]) -> bytes:
    """防止错误行中的公式前缀被电子表格软件执行。"""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=["url", "status", "risk_score", "reason", "warning", "error"]
    )
    writer.writeheader()
    for row in rows:
        safe = {
            key: (
                "'" + value
                if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@"))
                else value
            )
            for key, value in row.items()
        }
        writer.writerow(safe)
    return output.getvalue().encode("utf-8-sig")
