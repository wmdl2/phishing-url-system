"""钓鱼网址识别系统的本地中文网页入口。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from phishing_url.batch import BatchValidationError, export_csv, read_batch_csv, result_rows
from phishing_url.features import URLValidationError
from phishing_url.model import load_bundle, predict_batch, predict_url
from phishing_url.storage import (
    count_predictions,
    recent_predictions,
    register_model,
    save_prediction,
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DATABASE = Path(os.environ.get("PHISHING_HISTORY_DB", ROOT / "data/runtime/history.sqlite3"))


@st.cache_resource
def cached_bundle(model_path: str, modified_ns: int) -> dict:
    """模型文件变化时失效，避免重训后网页仍使用缓存旧模型。"""
    return load_bundle(Path(model_path))


def contribution_table(items: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(items).rename(
        columns={"feature": "特征", "value": "当前值", "contribution": "对数几率贡献"}
    )


def show_prediction(prediction: dict) -> None:
    st.subheader("存在钓鱼风险" if prediction["status"] == "钓鱼风险" else "模型未判为钓鱼")
    st.metric("模型风险分数", f"{prediction['risk_score']:.1%}")
    st.caption("分数由历史数据模型估计，未经真实流量概率校准；低分也不能证明网址安全。")
    st.caption(f"模型版本：{prediction['model_version']}")
    if prediction["warning"]:
        st.warning(prediction["warning"])
    left, right = st.columns(2)
    with left:
        st.write("推动钓鱼判断的特征")
        if prediction["reasons_for"]:
            st.dataframe(contribution_table(prediction["reasons_for"]), hide_index=True)
        else:
            st.write("本次没有正向贡献的特征。")
    with right:
        st.write("降低钓鱼判断的特征")
        if prediction["reasons_against"]:
            st.dataframe(contribution_table(prediction["reasons_against"]), hide_index=True)
        else:
            st.write("本次没有负向贡献的特征。")
    with st.expander("查看全部特征与计算说明"):
        st.dataframe(contribution_table(prediction["all_contributions"]), hide_index=True)
        st.write(
            f"模型截距：{prediction['intercept']:.4f}；"
            f"截距与全部贡献相加：{prediction['total_log_odds']:.4f}"
        )
        st.code(
            "标准化值 = (当前特征值 - 训练均值) / 训练标准差\n"
            "特征贡献 = 标准化值 × 模型系数\n"
            "风险分数 = 1 / (1 + exp(-(截距 + 全部贡献之和)))",
            language="text",
        )
    st.caption("贡献解释的是当前模型的计算，不代表因果关系。HTTPS、路径或提示词都不是安全结论。")


def single_page(bundle: dict) -> None:
    st.header("单条网址检测")
    st.write("输入 HTTP 或 HTTPS 网址。系统只解析文本，不访问网页或查询 DNS。")
    with st.form("single_form"):
        value = st.text_input("待检测网址", placeholder="https://example.com/account")
        submitted = st.form_submit_button("开始检测")
    if submitted:
        try:
            started = time.perf_counter()
            prediction = predict_url(bundle, value)
            elapsed_ms = (time.perf_counter() - started) * 1000
        except URLValidationError as exc:
            st.error(str(exc))
            return
        save_prediction(DATABASE, prediction, "single")
        show_prediction(prediction)
        st.caption(f"本次推断与解释耗时：{elapsed_ms:.2f} 毫秒（不含数据库和网页渲染）。")


def batch_page(bundle: dict) -> None:
    st.header("CSV 批量检测")
    st.write("UTF-8 CSV，必须有 url 列，1 至 1,000 行，文件不超过 5 MB。")
    uploaded = st.file_uploader("选择 CSV 文件", type="csv")
    if uploaded is not None and st.button("检测并生成结果"):
        try:
            urls = read_batch_csv(uploaded.getvalue())
        except BatchValidationError as exc:
            st.error(str(exc))
            return
        predictions = predict_batch(bundle, urls)
        for prediction in predictions:
            if prediction["status"] != "格式错误":
                save_prediction(DATABASE, prediction, "batch")
        st.session_state["batch_rows"] = result_rows(predictions)
    rows = st.session_state.get("batch_rows")
    if rows:
        results = pd.DataFrame(rows)
        st.success(
            f"处理 {len(rows)} 行，其中格式错误 "
            f"{int((results['status'] == '格式错误').to_numpy().sum())} 行。"
        )
        st.dataframe(results, hide_index=True, width="stretch")
        st.download_button(
            "下载结果 CSV",
            data=export_csv(rows),
            file_name="url_detection_results.csv",
            mime="text/csv",
            on_click="ignore",
        )
        st.caption("下载文件保留原始输入网址及参数；历史记录会隐藏查询参数值。请妥善保管下载文件。")


def history_page() -> None:
    st.header("检测历史")
    st.metric("已保存的检测次数", count_predictions(DATABASE))
    selected = st.selectbox("筛选结果", ["全部", "钓鱼风险", "正常"])
    rows = recent_predictions(DATABASE, status=None if selected == "全部" else selected)
    if rows:
        frame = pd.DataFrame(rows).rename(
            columns={
                "id": "序号",
                "checked_at": "检测时间（UTC）",
                "display_url": "网址（参数值已隐藏）",
                "status": "模型判断",
                "risk_score": "风险分数",
                "source": "来源",
                "model_version": "模型版本",
            }
        )
        frame["来源"] = frame["来源"].replace({"single": "单条", "batch": "批量"})
        st.dataframe(frame, hide_index=True, width="stretch")
    else:
        st.info("尚无符合条件的检测记录。")
    st.caption("展示最近 100 条。查询参数值、用户信息段与片段标识不入库；路径仍保留。")


def metric_table(metrics_by_name: dict) -> pd.DataFrame:
    labels = {
        "accuracy": "准确率",
        "precision": "钓鱼精确率",
        "recall": "钓鱼召回率",
        "f1": "钓鱼 F1",
        "false_positive_rate": "正常误报率",
    }
    return pd.DataFrame(
        [
            {"评价对象": name, **{label: f"{metrics[key]:.2%}" for key, label in labels.items()}}
            for name, metrics in metrics_by_name.items()
        ]
    )


def experiments_page(metadata: dict) -> None:
    st.header("模型实验")
    st.write("参数由验证集选择；内部测试集和来源原始测试划分均未参与调参。")
    st.json(metadata["selected_parameters"], expanded=False)
    st.subheader("内部测试集")
    st.dataframe(
        metric_table(
            {"混合数据测试": metadata["test_metrics"], **metadata.get("test_metrics_by_source", {})}
        ),
        hide_index=True,
    )
    st.image(str(ARTIFACTS / "test_confusion_matrix.png"))
    external_path = ARTIFACTS / "external_validation.json"
    if external_path.exists():
        external = json.loads(external_path.read_text(encoding="utf-8"))
        st.subheader("未参与开发的来源测试样本")
        st.write(
            f"{external['source']}；排除与开发数据重合的主机后，评价 "
            f"{external['evaluated_rows']:,} 条。"
        )
        comparison = {"当前模型": external["metrics"]}
        if "midterm_on_same_samples" in external:
            comparison["中期模型（同一批样本）"] = external["midterm_on_same_samples"]
        st.dataframe(metric_table(comparison), hide_index=True)
        st.warning("正常网址仍可能被误报，不能将本原型用于真实拦截；历史测试不能证明未来始终有效。")
    st.subheader("参数变化曲线")
    st.image(str(ARTIFACTS / "validation_curves.png"))
    st.dataframe(pd.read_csv(ARTIFACTS / "experiment_results.csv"), hide_index=True)
    with st.expander("数据清理与主机隔离审计"):
        cleaning = json.loads((ARTIFACTS / "cleaning_summary.json").read_text(encoding="utf-8"))
        splits = json.loads((ARTIFACTS / "split_summary.json").read_text(encoding="utf-8"))
        st.json({"清理统计": cleaning, "划分统计": splits}, expanded=False)


def main() -> None:
    st.set_page_config(page_title="钓鱼网址识别与分析", page_icon="🔎", layout="wide")
    st.title("钓鱼网址识别与分析系统")
    model_path = ARTIFACTS / "model.joblib"
    metadata_path = ARTIFACTS / "model_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        st.error("未找到训练结果。请先运行：python -m phishing_url.train")
        st.stop()
    try:
        bundle = cached_bundle(str(model_path), model_path.stat().st_mtime_ns)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    register_model(DATABASE, metadata)
    page = st.sidebar.radio("功能", ["单条检测", "批量检测", "检测历史", "实验结果"])
    st.sidebar.caption("仅分析网址文本。本系统是课程实践原型。")
    if page == "单条检测":
        single_page(bundle)
    elif page == "批量检测":
        batch_page(bundle)
    elif page == "检测历史":
        history_page()
    else:
        experiments_page(metadata)


if __name__ == "__main__":
    main()
