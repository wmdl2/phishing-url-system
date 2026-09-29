"""验证批量文件边界、混合错误行和下载文件安全行为。"""

import csv
import io

import pytest

from phishing_url.batch import BatchValidationError, export_csv, read_batch_csv


def test_batch_preserves_empty_rows_and_bom() -> None:
    urls = read_batch_csv(
        "\ufeffurl,note\nhttps://example.com,正常\n,空值\nftp://example.com,协议\n".encode()
    )
    assert urls == ["https://example.com", "", "ftp://example.com"]


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"name\na\n",
        b"url\n",
        b"url,url\na,b\n",
        b"url\na,b\n",
        b'url\n"unclosed\n',
        b"url\n\xff",
    ],
)
def test_invalid_batch_files(content: bytes) -> None:
    with pytest.raises(BatchValidationError):
        read_batch_csv(content)


def test_batch_row_limit() -> None:
    assert len(read_batch_csv(b"url\n" + b"https://example.com\n" * 1000)) == 1000
    with pytest.raises(BatchValidationError, match="1,000"):
        read_batch_csv(b"url\n" + b"https://example.com\n" * 1001)


def test_download_preserves_unicode_and_escapes_formula() -> None:
    rows = [
        {
            "url": "=SUM(1,2)",
            "status": "格式错误",
            "risk_score": None,
            "reason": "",
            "warning": "",
            "error": "主机无效",
        }
    ]
    content = export_csv(rows)
    decoded = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))
    assert decoded[0]["url"] == "'=SUM(1,2)"
    assert decoded[0]["error"] == "主机无效"
