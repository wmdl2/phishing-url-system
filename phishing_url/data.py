"""读取、审计并按主机名划分多来源网址数据。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypedDict, cast
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from phishing_url.features import (
    FEATURE_NAMES,
    FeatureValue,
    URLValidationError,
    extract_url_features,
)

CSV_NAME = "PhiUSIIL_Phishing_URL_Dataset.csv"


class CleaningAudit(TypedDict):
    """分别声明整数计数、字符串哈希与来源审计的类型。"""

    source_rows: int
    missing_or_bad_label: int
    invalid_url: int
    same_label_duplicates: int
    conflicting_urls: int
    retained_rows: int
    phishing_rows: int
    legitimate_rows: int
    dataset_sha256: str
    source_hashes: dict[str, str]
    distributions: list[dict[str, object]]


def load_clean_data(
    dataset_zip: Path, supplement: Path | None = None
) -> tuple[pd.DataFrame, CleaningAudit]:
    """只读取网址和标签，其他输入特征由共用函数重新计算。"""
    with ZipFile(dataset_zip) as archive, archive.open(CSV_NAME) as source:
        raw = pd.read_csv(source, usecols=["URL", "label"], dtype=str)
    raw = raw.rename(columns={"URL": "url"})
    raw["label"] = [{"0": "phish", "1": "benign"}.get(label) for label in raw["label"]]
    raw["data_source"] = "UCI PhiUSIIL"
    hashes = {"UCI PhiUSIIL": hashlib.sha256(dataset_zip.read_bytes()).hexdigest()}
    if supplement is not None:
        extra = pd.read_csv(supplement, usecols=["url", "label"], dtype=str)
        extra["data_source"] = "PhreshPhish train"
        raw = pd.concat([raw, extra], ignore_index=True)
        hashes["PhreshPhish train"] = hashlib.sha256(supplement.read_bytes()).hexdigest()
    identity = "|".join(f"{key}:{value}" for key, value in sorted(hashes.items()))
    audit: CleaningAudit = {
        "source_rows": len(raw),
        "missing_or_bad_label": 0,
        "invalid_url": 0,
        "same_label_duplicates": 0,
        "conflicting_urls": 0,
        "retained_rows": 0,
        "phishing_rows": 0,
        "legitimate_rows": 0,
        "dataset_sha256": hashlib.sha256(identity.encode()).hexdigest(),
        "source_hashes": hashes,
        "distributions": [],
    }
    kept: dict[str, tuple[int, str, str, dict[str, FeatureValue]]] = {}
    conflicts: set[str] = set()
    for url, label, source_name in raw[["url", "label", "data_source"]].itertuples(
        index=False, name=None
    ):
        if not isinstance(url, str) or label not in {"phish", "benign"}:
            audit["missing_or_bad_label"] += 1
            continue
        try:
            parsed = extract_url_features(url)
        except URLValidationError:
            audit["invalid_url"] += 1
            continue
        key = parsed.original
        if key in conflicts:
            continue
        target = int(label == "phish")
        old = kept.get(key)
        if old is not None:
            if old[0] == target:
                audit["same_label_duplicates"] += 1
            else:
                audit["conflicting_urls"] += 1
                conflicts.add(key)
                del kept[key]
            continue
        kept[key] = (target, parsed.hostname, source_name, parsed.values)
    rows = [
        {"url": url, "hostname": host, "target": target, "data_source": source_name, **values}
        for url, (target, host, source_name, values) in kept.items()
    ]
    clean = pd.DataFrame.from_records(
        rows, columns=["url", "hostname", "target", "data_source", *FEATURE_NAMES]
    )
    if clean.empty or clean["target"].nunique() != 2:
        raise ValueError("清理后数据不足，无法训练二分类模型")
    audit["retained_rows"] = len(clean)
    audit["phishing_rows"] = int(clean["target"].to_numpy(dtype=np.int64).sum())
    audit["legitimate_rows"] = len(clean) - audit["phishing_rows"]
    for group_key, frame in clean.groupby(["data_source", "target"]):
        source_name, target = cast(tuple[str, int], group_key)
        audit["distributions"].append(
            {
                "source": str(source_name),
                "class": "钓鱼" if target == 1 else "正常",
                "rows": len(frame),
                "nonempty_path_rate": float((frame["path_length"].to_numpy() > 0).mean()),
                "nonempty_query_rate": float((frame["query_length"].to_numpy() > 0).mean()),
                "https_rate": float(frame["is_https"].to_numpy().mean()),
            }
        )
    return clean, audit


def split_by_hostname(data: pd.DataFrame, seed: int = 42) -> tuple[dict[str, pd.DataFrame], dict]:
    """五折分组分层划分；尽量保持来源与类别比例，完整主机名互斥。"""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = np.full(len(data), -1, dtype=np.int8)
    strata = data["target"].astype(str) + "|" + data["data_source"].astype(str)
    for fold, (_, held_out) in enumerate(
        splitter.split(data[list(FEATURE_NAMES)], strata, groups=data["hostname"])
    ):
        folds[held_out] = fold
    if (folds < 0).any():
        raise AssertionError("部分样本未分配到数据集合")
    result = {
        "train": data.loc[folds >= 2].reset_index(drop=True),
        "validation": data.loc[folds == 1].reset_index(drop=True),
        "test": data.loc[folds == 0].reset_index(drop=True),
    }
    hosts = {name: set(frame["hostname"]) for name, frame in result.items()}
    names = list(result)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if hosts[left] & hosts[right]:
                raise AssertionError(f"{left} 与 {right} 中有重复主机")
    summary = {
        "seed": seed,
        "method": "五折按主机分组、按来源与标签分层；测试=0，验证=1，训练=2..4",
        "sets": {
            name: {
                "rows": len(frame),
                "hosts": len(hosts[name]),
                "phishing_rows": int(frame["target"].to_numpy(dtype=np.int64).sum()),
                "legitimate_rows": int((frame["target"].to_numpy() == 0).sum()),
                "sources": {str(k): int(v) for k, v in frame["data_source"].value_counts().items()},
            }
            for name, frame in result.items()
        },
        "hostname_overlap": 0,
    }
    return result, summary
