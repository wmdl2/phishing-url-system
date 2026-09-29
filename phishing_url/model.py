"""模型推断和线性模型的单次预测特征贡献计算。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from phishing_url.features import (
    FEATURE_LABELS,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    URLValidationError,
    extract_url_features,
)


def load_bundle(path: Path) -> dict:
    """只加载本地训练产生且可信的模型文件。"""
    bundle = joblib.load(path)
    if (
        tuple(bundle["feature_names"]) != FEATURE_NAMES
        or bundle.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
    ):
        raise ValueError("模型特征版本与当前程序不一致，请重新训练")
    return bundle


def predict_url(bundle: dict, value: object) -> dict:
    parsed = extract_url_features(value)
    x = pd.DataFrame([{name: parsed.values[name] for name in FEATURE_NAMES}])
    pipeline = bundle["pipeline"]
    classifier = pipeline.named_steps["classifier"]
    phishing_column = int(np.where(classifier.classes_ == 1)[0][0])
    risk_score = float(pipeline.predict_proba(x)[0, phishing_column])
    is_phishing = risk_score >= bundle["threshold"]
    standardized = pipeline.named_steps["scaler"].transform(x)[0]
    coefficients = classifier.coef_[0]
    if phishing_column == 0:
        coefficients = -coefficients
    contributions = standardized * coefficients
    intercept = float(classifier.intercept_[0])
    if phishing_column == 0:
        intercept = -intercept
    ranked = [
        {
            "feature": FEATURE_LABELS[name],
            "value": parsed.values[name],
            "contribution": float(contribution),
        }
        for name, contribution in zip(FEATURE_NAMES, contributions, strict=True)
    ]
    positive = sorted(
        (item for item in ranked if item["contribution"] > 0),
        key=lambda item: item["contribution"],
        reverse=True,
    )[:3]
    negative = sorted(
        (item for item in ranked if item["contribution"] < 0), key=lambda item: item["contribution"]
    )[:3]
    return {
        "url": parsed.original,
        "display_url": parsed.display_url,
        "status": "钓鱼风险" if is_phishing else "正常",
        "risk_score": risk_score,
        "reasons_for": positive,
        "reasons_against": negative,
        "model_version": bundle["model_version"],
        "warning": parsed.warning,
        "intercept": intercept,
        "total_log_odds": float(intercept + contributions.sum()),
        "all_contributions": ranked,
    }


def predict_batch(bundle: dict, urls: Sequence[object]) -> list[dict]:
    output = []
    for value in urls:
        try:
            output.append(predict_url(bundle, value))
        except URLValidationError as exc:
            output.append(
                {"url": value, "status": "格式错误", "risk_score": None, "error": str(exc)}
            )
    return output
