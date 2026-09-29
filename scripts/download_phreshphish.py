"""按列下载公开数据，只保存网址和标签，不读取目标网站。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import gettempdir

import fsspec
import pandas as pd
import pyarrow.parquet as pq

DATASET = "phreshphish/phreshphish"
REVISION = "748d45b35cca9cff94f6b1cac051f22cde5b0345"
ROOT = Path(__file__).resolve().parents[1]


def files_for_split(split: str) -> list[dict]:
    url = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/default/{split}"
    with urllib.request.urlopen(url, timeout=60) as response:
        entries = json.load(response)
    return sorted(
        [entry for entry in entries if entry["path"].endswith(".parquet")],
        key=lambda entry: entry["path"],
    )


def fetch_columns(entry: dict, cache: Path) -> tuple[pd.DataFrame, dict]:
    name = Path(entry["path"]).stem
    local = cache / f"{name}.csv.gz"
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{entry['path']}"
    frame: pd.DataFrame | None = None
    if local.exists():
        frame = pd.read_csv(local, dtype=str)
    else:
        for attempt in range(5):
            try:
                with fsspec.open(url, "rb", block_size=65536, cache_type="bytes") as source:
                    frame = pd.DataFrame(
                        pq.ParquetFile(source).read(columns=["url", "label"]).to_pandas()
                    )
                frame.to_csv(local, index=False, compression={"method": "gzip", "mtime": 0})
                break
            except (OSError, ValueError) as exc:
                if attempt == 4:
                    raise RuntimeError(f"分片读取失败：{entry['path']}") from exc
                time.sleep(min(2**attempt, 8))
    if frame is None:
        raise RuntimeError("分片读取未返回数据")
    print(f"已读取 {entry['path']}：{len(frame):,} 行", flush=True)
    return frame, {"url": url, "rows": len(frame), "parquet_size": entry["size"]}


def download(output: Path, cache: Path, test_per_class: int = 10000) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": DATASET,
        "revision": REVISION,
        "source_url": f"https://huggingface.co/datasets/{DATASET}",
        "license": "CC BY 4.0; anti-phishing research",
        "columns": ["url", "label"],
        "seed": 42,
        "test_per_class": test_per_class,
        "splits": {},
    }
    for split in ("train", "test"):
        split_cache = cache / REVISION / split
        split_cache.mkdir(parents=True, exist_ok=True)
        entries = files_for_split(split)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda item: fetch_columns(item, split_cache), entries))
        frame = pd.concat([result[0] for result in results], ignore_index=True)
        original_rows = len(frame)
        valid = pd.DataFrame(frame.loc[frame["label"].isin(["benign", "phish"])])
        if split == "test":
            valid = pd.DataFrame(
                valid.groupby("label", group_keys=False).sample(n=test_per_class, random_state=42)
            )
        destination = output / f"phreshphish_{split}.csv.gz"
        valid.to_csv(destination, index=False, compression={"method": "gzip", "mtime": 0})
        manifest["splits"][split] = {
            "original_rows": original_rows,
            "saved_rows": len(valid),
            "class_counts": {str(k): int(v) for k, v in valid["label"].value_counts().items()},
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "files": [result[1] for result in results],
        }
        print(f"已保存 {destination}：{len(valid):,} 行", flush=True)
    (output / "phreshphish_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="下载固定版本的 PhreshPhish 网址与标签")
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--cache", type=Path, default=Path(gettempdir()) / "phreshphish-url-cache")
    parser.add_argument("--test-per-class", type=int, default=10000)
    args = parser.parse_args()
    download(args.output, args.cache, args.test_per_class)


if __name__ == "__main__":
    main()
