"""系统关键行为测试：网址解析、数据隔离、训练、批量处理与隐私保护。"""

from __future__ import annotations

import csv
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from phishing_url.data import CSV_NAME, load_clean_data, split_by_hostname
from phishing_url.features import URLValidationError, extract_url_features
from phishing_url.model import load_bundle, predict_batch, predict_url
from phishing_url.storage import (
    count_predictions,
    recent_predictions,
    register_model,
    save_prediction,
)
from phishing_url.train import run_training


def make_dataset(path: Path) -> None:
    """构造带有网页字段和重复网址的小型数据集，测试完整流程。"""
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        lines = [["URL", "label", "LineOfCode"]]
        for label in (0, 1):
            for host_number in range(15):
                for path_number in range(3):
                    url = (
                        f"https://site{host_number}-{label}.example.com/"
                        f"{'verify' if label == 0 else 'about'}/{path_number}"
                    )
                    lines.append([url, str(label), str(9999 if label == 0 else 1)])
        lines.append([lines[1][0], lines[1][1], lines[1][2]])
        lines.append(["not a valid host", "0", "0"])
        lines.append(["https://site0-0.example.com/verify/0", "1", "0"])
        import io

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(lines)
        archive.writestr(CSV_NAME, output.getvalue())


def test_feature_extraction_and_query_redaction() -> None:
    parsed = extract_url_features("https://user:pass@shop.example.com:8443/a/b?token=secret&n=2")
    assert parsed.hostname == "shop.example.com"
    assert parsed.values["is_https"] == 1
    assert parsed.values["path_depth"] == 2
    assert parsed.values["query_param_count"] == 2
    assert parsed.values["has_explicit_port"] == 1
    assert parsed.values["at_count"] == 1
    assert "secret" not in parsed.display_url
    assert "pass" not in parsed.display_url


@pytest.mark.parametrize(
    "value",
    ["", None, "ftp://example.com", "http:///missing", "https://bad..example.com", "no-host"],
)
def test_invalid_urls(value: object) -> None:
    with pytest.raises(URLValidationError):
        extract_url_features(value)


def test_training_inference_and_history(tmp_path: Path) -> None:
    dataset = tmp_path / "sample.zip"
    make_dataset(dataset)
    cleaned, audit = load_clean_data(dataset)
    assert audit["same_label_duplicates"] == 1
    assert audit["conflicting_urls"] == 1
    assert audit["invalid_url"] == 1
    assert "LineOfCode" not in cleaned.columns
    splits, summary = split_by_hostname(cleaned)
    assert summary["hostname_overlap"] == 0
    for name in splits:
        assert len(splits[name]) > 0

    artifacts = tmp_path / "artifacts"
    metadata = run_training(dataset, artifacts)
    bundle = load_bundle(artifacts / "model.joblib")
    detected = predict_url(bundle, "https://new.example.net/?token=private")
    assert detected["status"] in {"钓鱼风险", "正常"}
    assert "private" not in detected["display_url"]
    assert len(detected["reasons_for"] + detected["reasons_against"]) > 0
    results = predict_batch(bundle, ["https://example.com", "", "ftp://example.org"])
    assert results[0]["status"] in {"钓鱼风险", "正常"}
    assert results[1]["status"] == "格式错误"
    assert results[2]["status"] == "格式错误"

    database = tmp_path / "history.sqlite3"
    register_model(database, metadata)
    save_prediction(database, detected, "single")
    assert count_predictions(database) == 1
    history = recent_predictions(database)
    assert history[0]["model_version"] == metadata["model_version"]
    assert "private" not in history[0]["display_url"]


def test_features_do_not_use_network(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("离线特征提取不能发起网络操作")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    for value in ("https://example.com/account", "http://192.0.2.1/login", "example.com:8080/a"):
        parsed = extract_url_features(value)
        assert len(parsed.values) == 28
    assert extract_url_features("example.com").warning
    with pytest.raises(URLValidationError, match="IP"):
        extract_url_features("https://999.999.999.999")


def test_split_repeatability_and_exact_contributions(tmp_path: Path) -> None:
    import math

    import numpy as np
    import pandas as pd

    from phishing_url.features import FEATURE_NAMES

    dataset = tmp_path / "sample.zip"
    make_dataset(dataset)
    data, _ = load_clean_data(dataset)
    first, _ = split_by_hostname(data, 42)
    second, _ = split_by_hostname(data, 42)
    for name in first:
        assert first[name]["url"].tolist() == second[name]["url"].tolist()
    output = tmp_path / "artifacts"
    run_training(dataset, output)
    bundle = load_bundle(output / "model.joblib")
    prediction = predict_url(bundle, "https://example.com/account?key=secret")
    summed = prediction["intercept"] + sum(
        item["contribution"] for item in prediction["all_contributions"]
    )
    assert math.isclose(prediction["risk_score"], 1 / (1 + math.exp(-summed)), rel_tol=1e-10)
    np.testing.assert_allclose(
        bundle["pipeline"].named_steps["scaler"].mean_,
        first["train"][list(FEATURE_NAMES)].to_numpy().mean(axis=0),
    )
    parsed = extract_url_features("https://example.com/account?key=secret")
    direct = bundle["pipeline"].predict_proba(pd.DataFrame([parsed.values]))[0, 1]
    assert math.isclose(prediction["risk_score"], direct, rel_tol=1e-10)


def test_streamlit_pages_and_single_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from streamlit.testing.v1 import AppTest

    dataset = tmp_path / "sample.zip"
    make_dataset(dataset)
    run_training(dataset, tmp_path / "artifacts")
    shutil.copy2(Path(__file__).resolve().parents[1] / "app.py", tmp_path / "app.py")
    database = tmp_path / "history.sqlite3"
    monkeypatch.setenv("PHISHING_HISTORY_DB", str(database))
    app = AppTest.from_file(str(tmp_path / "app.py"), default_timeout=20).run()
    assert not app.exception
    app.text_input[0].input("https://example.com/account?key=private")
    app.button[0].click().run()
    assert not app.exception
    assert count_predictions(database) == 1
    app.text_input[0].input("")
    app.button[0].click().run()
    assert app.error[0].value == "网址不能为空"
    assert count_predictions(database) == 1
    for page in ["检测历史", "实验结果", "批量检测"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
    assert "private" not in recent_predictions(database)[0]["display_url"]
