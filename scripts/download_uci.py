"""下载并校验 UCI 的公开原始压缩包，不访问样本网址。"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip"
SHA256 = "0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a"


def main() -> None:
    destination = Path(__file__).resolve().parents[1] / "data/raw/phiusiil.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == SHA256:
        print("本地数据校验通过，无需重新下载。")
        return
    with urllib.request.urlopen(URL, timeout=90) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != SHA256:
        raise ValueError("UCI 文件哈希与已验证版本不符，请核对来源，未覆盖现有数据")
    destination.write_bytes(content)
    print(f"已保存并校验：{destination}")


if __name__ == "__main__":
    main()
