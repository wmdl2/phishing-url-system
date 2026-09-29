"""用固定数据、划分和选定配置重新拟合一次，核对最终测试预测。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phishing_url.data import load_clean_data, split_by_hostname
from phishing_url.features import FEATURE_NAMES
from phishing_url.train import PROJECT_DIR, score_predictions


def main() -> None:
    artifacts = PROJECT_DIR / "artifacts"
    metadata = json.loads((artifacts / "model_metadata.json").read_text(encoding="utf-8"))
    supplement = PROJECT_DIR / "data/raw/phreshphish_train.csv.gz"
    data, audit = load_clean_data(
        PROJECT_DIR / "data/raw/phiusiil.zip",
        supplement if "PhreshPhish train" in metadata["source_hashes"] else None,
    )
    if audit["dataset_sha256"] != metadata["dataset_sha256"]:
        raise ValueError("当前数据与原训练数据不一致")
    splits, summary = split_by_hostname(data, metadata["seed"])
    selected = metadata["selected_parameters"]
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=selected["C"],
                    class_weight=None if selected["class_weight"] == "none" else "balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=metadata["seed"],
                ),
            ),
        ]
    )
    model.fit(splits["train"][list(FEATURE_NAMES)], splits["train"]["target"])
    probabilities = np.asarray(model.predict_proba(splits["test"][list(FEATURE_NAMES)]))[:, 1]
    recorded = pd.read_csv(artifacts / "test_predictions.csv")
    assert recorded["hostname"].tolist() == splits["test"]["hostname"].tolist()
    expected = recorded["risk_score"].to_numpy(dtype=float)
    np.testing.assert_allclose(probabilities, expected, rtol=1e-9, atol=1e-12)
    predicted = probabilities >= selected["threshold"]
    assert np.array_equal(predicted.astype(int), recorded["predicted"].to_numpy())
    result = {
        "model_version": metadata["model_version"],
        "seed": metadata["seed"],
        "hostname_overlap": summary["hostname_overlap"],
        "test_rows": len(recorded),
        "maximum_probability_difference": float(np.max(np.abs(probabilities - expected))),
        "identical_test_classifications": True,
        "metrics": score_predictions(
            splits["test"]["target"].to_numpy(dtype=np.int64), probabilities, selected["threshold"]
        ),
    }
    (artifacts / "reproducibility.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
