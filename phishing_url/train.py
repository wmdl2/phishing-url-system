"""多来源网址数据上的可复现逻辑回归训练与参数实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from phishing_url.data import load_clean_data, split_by_hostname  # noqa: E402
from phishing_url.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION  # noqa: E402
from phishing_url.model import predict_url  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
C_VALUES = (0.01, 0.1, 1.0, 10.0)
WEIGHTS = (None, "balanced")
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def score_predictions(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    """由混淆矩阵统一计算指标；分母为零时按零处理。"""
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = (
        int(value) for value in confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / len(y_true) if len(y_true) else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _json(path: Path, value: Mapping[str, object]) -> None:
    """接收只读映射，兼容普通字典和逐字段声明的 TypedDict。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def configure_plot_font() -> None:
    """使用已有中文字体，不把系统字体打包进项目。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "Microsoft YaHei", "SimHei"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    else:
        windows_font = Path("/mnt/c/Windows/Fonts/msyh.ttc")
        if windows_font.exists():
            font_manager.fontManager.addfont(str(windows_font))
            plt.rcParams["font.sans-serif"] = [
                font_manager.FontProperties(fname=windows_font).get_name()
            ]
    plt.rcParams["axes.unicode_minus"] = False


def _plot_experiments(table: pd.DataFrame, best: dict, output: Path) -> None:
    configure_plot_font()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fields = [
        (
            "C",
            table.loc[
                (table["class_weight"] == best["class_weight"])
                & (table["threshold"] == best["threshold"])
            ],
            "正则化参数 C",
        ),
        (
            "class_weight",
            table.loc[(table["C"] == best["C"]) & (table["threshold"] == best["threshold"])],
            "类别权重",
        ),
        (
            "threshold",
            table.loc[(table["C"] == best["C"]) & (table["class_weight"] == best["class_weight"])],
            "判定阈值",
        ),
    ]
    for axis, (field, subset, title) in zip(axes, fields, strict=True):
        subset = subset.sort_values(by=field)
        x = subset[field].astype(str) if field == "class_weight" else subset[field]
        for metric, label in (("f1", "F1"), ("recall", "召回率"), ("precision", "精确率")):
            axis.plot(x, subset[metric], marker="o", label=label)
        right = axis.twinx()
        right.plot(x, subset["false_positive_rate"], color="red", linestyle="--", label="误报率")
        right.set_ylim(0, max(0.02, float(subset["false_positive_rate"].to_numpy().max()) * 1.2))
        right.set_ylabel("误报率（右轴）", color="red")
        if field == "C":
            axis.set_xscale("log")
        axis.set_title(title)
        axis.set_xlabel(field)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("验证集得分")
    axes[-1].legend(loc="lower left", fontsize=8)
    fig.suptitle("验证集参数实验：其余两个参数固定为选中配置")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_confusion(metrics: dict, output: Path) -> None:
    configure_plot_font()
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f"{matrix[row, column]:,}", ha="center", va="center")
    ax.set_xticks([0, 1], ["正常", "钓鱼"])
    ax.set_yticks([0, 1], ["正常", "钓鱼"])
    ax.set_xlabel("模型预测")
    ax.set_ylabel("真实标签")
    ax.set_title("独立测试集混淆矩阵")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run_training(
    dataset: Path, output: Path, seed: int = 42, supplement: Path | None = None
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    print("正在提取特征并清理数据……", flush=True)
    data, cleaning = load_clean_data(dataset, supplement)
    print(f"保留 {len(data):,} 条，正在按主机划分……", flush=True)
    splits, split_summary = split_by_hostname(data, seed)
    _json(output / "cleaning_summary.json", cleaning)
    _json(output / "split_summary.json", split_summary)
    x_train = splits["train"][list(FEATURE_NAMES)]
    y_train = splits["train"]["target"].to_numpy(dtype=np.int64)
    x_val = splits["validation"][list(FEATURE_NAMES)]
    y_val = splits["validation"]["target"].to_numpy(dtype=np.int64)
    x_test = splits["test"][list(FEATURE_NAMES)]
    y_test = splits["test"]["target"].to_numpy(dtype=np.int64)
    if any(len(np.unique(y)) != 2 for y in (y_train, y_val, y_test)):
        raise ValueError("有数据集合缺少某个类别，无法完成实验")
    baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    baseline_metrics = score_predictions(y_val, np.asarray(baseline.predict(x_val)), 0.5)
    rows = []
    models = {}
    for weight in WEIGHTS:
        for c_value in C_VALUES:
            print(f"训练 C={c_value}，类别权重={weight or 'none'}……", flush=True)
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=c_value,
                            class_weight=weight,
                            max_iter=1000,
                            solver="lbfgs",
                            random_state=seed,
                        ),
                    ),
                ]
            )
            pipeline.fit(x_train, y_train)
            if int(pipeline.named_steps["classifier"].n_iter_.max()) >= 1000:
                raise RuntimeError("逻辑回归未收敛，请检查特征与训练日志")
            models[(c_value, weight)] = pipeline
            probabilities = np.asarray(pipeline.predict_proba(x_val))[:, 1]
            for threshold in THRESHOLDS:
                rows.append(
                    {
                        "C": c_value,
                        "class_weight": weight or "none",
                        "threshold": threshold,
                        **score_predictions(y_val, probabilities, threshold),
                    }
                )
    experiments = pd.DataFrame(rows)
    experiments.to_csv(output / "experiment_results.csv", index=False, encoding="utf-8-sig")
    selected = (
        experiments.sort_values(
            by=["f1", "false_positive_rate", "recall", "C", "class_weight", "threshold"],
            ascending=[False, True, False, True, True, True],
            kind="stable",
        )
        .iloc[0]
        .to_dict()
    )
    chosen_weight = None if selected["class_weight"] == "none" else "balanced"
    chosen = models[(float(selected["C"]), chosen_weight)]
    threshold = float(selected["threshold"])
    test_probabilities = np.asarray(chosen.predict_proba(x_test))[:, 1]
    test_metrics = score_predictions(y_test, test_probabilities, threshold)
    parameters = {
        "C": float(selected["C"]),
        "class_weight": selected["class_weight"],
        "threshold": threshold,
    }
    material = {
        "data": cleaning["dataset_sha256"],
        "seed": seed,
        "parameters": parameters,
        "features": FEATURE_NAMES,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_code_sha256": hashlib.sha256(
            (PROJECT_DIR / "phishing_url/features.py").read_bytes()
        ).hexdigest(),
        "sklearn": sklearn.__version__,
    }
    model_version = (
        "url-lr-" + hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:12]
    )
    bundle = {
        "pipeline": chosen,
        "threshold": threshold,
        "feature_names": FEATURE_NAMES,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_version": model_version,
    }
    joblib.dump(bundle, output / "model.joblib")
    predict_url(bundle, splits["test"].iloc[0]["url"])
    latencies = []
    for sample_url in splits["test"]["url"].head(100):
        started = time.perf_counter()
        predict_url(bundle, sample_url)
        latencies.append((time.perf_counter() - started) * 1000)
    timing = {
        "single_prediction_median_ms": float(np.median(latencies)),
        "single_prediction_p95_ms": float(np.percentile(latencies, 95)),
        "measurement_repeats": len(latencies),
        "scope": "预热后特征提取、模型推断和解释；不含网页渲染与数据库写入",
    }
    source_metrics = {}
    for source_name in splits["test"]["data_source"].unique():
        mask = splits["test"]["data_source"].to_numpy() == source_name
        source_metrics[str(source_name)] = score_predictions(
            y_test[mask], test_probabilities[mask], threshold
        )
    metadata = {
        "model_version": model_version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source": "UCI PhiUSIIL + PhreshPhish train" if supplement else "UCI PhiUSIIL",
        "selected_parameters": parameters,
        "validation_metrics": {key: selected[key] for key in test_metrics},
        "test_metrics": test_metrics,
        "test_metrics_by_source": source_metrics,
        "baseline_validation_metrics": baseline_metrics,
        "timing": timing,
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_sha256": cleaning["dataset_sha256"],
        "source_hashes": cleaning["source_hashes"],
        "seed": seed,
        "environment": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    _json(output / "model_metadata.json", metadata)
    _json(output / "final_metrics.json", test_metrics)
    pd.DataFrame(
        {
            "target": y_test,
            "predicted": (test_probabilities >= threshold).astype(int),
            "risk_score": test_probabilities,
            "hostname": splits["test"]["hostname"],
            "data_source": splits["test"]["data_source"],
        }
    ).to_csv(output / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.Series(sorted(set(data["hostname"])), name="hostname").to_csv(
        output / "development_hosts.csv.gz", index=False, compression="gzip"
    )
    _plot_experiments(experiments, selected, output / "validation_curves.png")
    _plot_confusion(test_metrics, output / "test_confusion_matrix.png")
    print(
        json.dumps(
            {
                "selected": parameters,
                "test": test_metrics,
                "by_source": source_metrics,
                "timing": timing,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="训练仅依靠网址特征的钓鱼识别模型")
    parser.add_argument("--dataset", type=Path, default=PROJECT_DIR / "data/raw/phiusiil.zip")
    parser.add_argument(
        "--supplement", type=Path, default=PROJECT_DIR / "data/raw/phreshphish_train.csv.gz"
    )
    parser.add_argument("--uci-only", action="store_true", help="只用 UCI 数据进行对照实验")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_training(args.dataset, args.output, args.seed, None if args.uci_only else args.supplement)


if __name__ == "__main__":
    main()
