"""选定模型后，评价未参与训练和调参的 PhreshPhish 测试样本。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from phishing_url.features import FEATURE_NAMES, extract_url_features
from phishing_url.model import load_bundle
from phishing_url.train import PROJECT_DIR, _plot_confusion, score_predictions


def evaluate_external(source: Path, artifacts: Path, baseline_dir: Path | None = None) -> dict:
    """排除开发数据中任何主机；中期模型在同一批有效样本上作对照。"""
    data = pd.read_csv(source, usecols=["url", "label"], dtype=str)
    source_rows = len(data)
    data = pd.DataFrame(data.loc[data["label"].isin(["benign", "phish"])])
    conflicts = [key for key, group in data.groupby("url") if len(set(group["label"])) > 1]
    data = pd.DataFrame(data.loc[~data["url"].isin(conflicts)]).drop_duplicates(subset=["url"])
    hosts = set(pd.read_csv(artifacts / "development_hosts.csv.gz", dtype=str)["hostname"])
    old_extract = None
    old_bundle = None
    if baseline_dir is not None:
        feature_file = baseline_dir / "source/phishing_url/features.py"
        spec = importlib.util.spec_from_file_location("midterm_features", feature_file)
        if spec is None or spec.loader is None:
            raise ValueError("无法读取中期特征代码")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        old_extract = module.extract_url_features
        old_bundle = joblib.load(baseline_dir / "artifacts/model.joblib")
    features = []
    old_features = []
    invalid = overlap = 0
    for url, label in data.loc[:, ["url", "label"]].itertuples(index=False, name=None):
        try:
            parsed = extract_url_features(url)
            previous = old_extract(url) if old_extract is not None else None
        except ValueError:
            invalid += 1
            continue
        if parsed.hostname in hosts:
            overlap += 1
            continue
        features.append({"target": int(label == "phish"), **parsed.values})
        if previous is not None:
            old_features.append(previous.values)
    frame = pd.DataFrame(features)
    if frame.empty or frame["target"].nunique() != 2:
        raise ValueError("外部数据没有足够的两类网址")
    bundle = load_bundle(artifacts / "model.joblib")
    probabilities = np.asarray(bundle["pipeline"].predict_proba(frame[list(FEATURE_NAMES)]))[:, 1]
    labels = frame["target"].to_numpy(dtype=np.int64)
    metrics = score_predictions(labels, probabilities, bundle["threshold"])
    detail = {
        "source": "PhreshPhish 原始 test；按类各抽样 10,000 条，随机种子 42",
        "source_url": "https://huggingface.co/datasets/phreshphish/phreshphish",
        "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sampled_rows": source_rows,
        "removed_duplicates_or_conflicts": source_rows - len(data),
        "conflicting_urls": len(conflicts),
        "invalid_urls": invalid,
        "excluded_overlap_hosts_rows": overlap,
        "evaluated_rows": len(frame),
        "phishing_rows": int(labels.sum()),
        "legitimate_rows": int((labels == 0).sum()),
        "model_version": bundle["model_version"],
        "metrics": metrics,
        "note": "本数据未用于选择模型配置；同一来源的新划分也不等于未来真实流量。",
    }
    if old_bundle is not None:
        old_x = pd.DataFrame(old_features)[list(old_bundle["feature_names"])]
        old_probabilities = np.asarray(old_bundle["pipeline"].predict_proba(old_x))[:, 1]
        detail["midterm_model_version"] = old_bundle["model_version"]
        detail["midterm_on_same_samples"] = score_predictions(
            labels, old_probabilities, old_bundle["threshold"]
        )
    (artifacts / "external_validation.json").write_text(
        json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_confusion(metrics, artifacts / "external_confusion_matrix.png")
    print(json.dumps(detail, ensure_ascii=False, indent=2))
    return detail


def main() -> None:
    parser = argparse.ArgumentParser(description="在未参与开发的来源测试样本上评价模型")
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_DIR / "data/raw/phreshphish_test.csv.gz"
    )
    parser.add_argument("--artifacts", type=Path, default=PROJECT_DIR / "artifacts")
    parser.add_argument("--baseline-dir", type=Path)
    args = parser.parse_args()
    evaluate_external(args.dataset, args.artifacts, args.baseline_dir)


if __name__ == "__main__":
    main()
