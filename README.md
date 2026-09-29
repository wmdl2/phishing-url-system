# 可解释钓鱼网址识别与分析系统

本地中文课程实践原型：单条网址检测、UTF-8 CSV 批量检测、SQLite 历史记录和模型实验展示。只解析网址文本，不访问目标网页或查询 DNS。训练与检测共用同一套 28 个网址特征。

当前模型在混合数据独立测试集上准确率约 88.36%、钓鱼 F1 约 86.50%。在 PhreshPhish 原始 test 保留样本上，正常误报率仍约 23.17%；本项目用于学习与实验，不能作为真实网站拦截依据。分数未经真实流量概率校准，低分也不能证明安全。

## 安装与启动

使用 Python 3.12，在本文件所在目录执行。GitHub 仓库仅含源码，不含原始数据、训练模型或检测历史。

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-data.txt
python scripts/download_uci.py
python scripts/download_phreshphish.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m phishing_url.train
python -m phishing_url.external_eval
python -m streamlit run app.py
~~~

Windows PowerShell 的激活命令为 .\.venv\Scripts\Activate.ps1；训练命令使用 python -m phishing_url.train 即可。所有特征均可在 Windows 与 Linux 上离线计算。

若代码放在 WSL 的 /mnt 下，虚拟环境可以建在 WSL 家目录以改善安装速度：

~~~bash
python3 -m venv "$HOME/.venvs/phishing-url-system"
source "$HOME/.venvs/phishing-url-system/bin/activate"
~~~

本地已经有 artifacts/model.joblib 时，直接启动即可。重新训练会覆盖 artifacts 内的实验文件；模型缓存会在文件变化后失效。仅使用 UCI 作对照可运行 python -m phishing_url.train --uci-only --output artifacts_uci。

数据下载脚本只从 UCI 和 Hugging Face 下载公开数据，固定校验哈希或版本；PhreshPhish 使用 Parquet 按列读取 URL 与标签，避免下载整份约 36 GB 的网页数据。下载缓存可以被删除，最终 CSV 和模型仍保存在项目 data/raw 和 artifacts 中。

## 输入输出与历史

- 支持 HTTP、HTTPS、IP 主机、国际化域名。无协议输入按 HTTPS 解析并明确显示假设；建议输入完整协议。
- 单条检测显示风险分数、正负特征贡献及完整计算说明。
- 批量 CSV 必须有 url 列，UTF-8（可带 BOM），1 至 1,000 行，最大 5 MB；有效与无效行按原顺序返回。
- 下载 CSV 含 url、status、risk_score、reason、warning、error。原始网址和查询参数会保留在下载文件中，电子表格公式前缀会转义。
- SQLite 存在 data/runtime/history.sqlite3。历史记录隐藏查询参数值、用户信息段和片段标识，路径仍保留；界面显示最近 100 条并可按类别筛选。
- 模型版本同时关联参数、数据哈希、特征版本和依赖版本；旧版历史保留其原版本标识。
- 网页默认只监听 127.0.0.1，关闭使用统计上传。

## 实验与可解释性

UCI 原始标签 0 为钓鱼、1 为正常；内部统一 1 为钓鱼。PhreshPhish 的 phish/benign 分别映射为 1/0。空值、无效标签、无效网址、同标签重复和冲突网址均有清理统计。

合并 UCI 与 PhreshPhish 的 train 后，用五折 StratifiedGroupKFold 按完整主机名分组，尽量保持来源与类别比例；三折训练、一折验证、一折测试，随机种子 42。完整主机名在三集合之间互斥，不保证不同子域所属的注册域也互斥。

模型是 StandardScaler + LogisticRegression。标准化只拟合训练集。验证集搜索 C=[0.01, 0.1, 1, 10]、类别权重=[none, balanced]、阈值=[0.3, 0.4, 0.5, 0.6, 0.7]，共 8 次拟合、40 组验证结果；以钓鱼 F1 最高、误报率最低为优先规则选择配置。内部测试与 PhreshPhish 原始 test 均不用于选参。后者每类固定抽样 10,000 条，排除全部开发数据中的重合主机后评价 13,456 条；其来源与训练补充数据相同，并非全新数据来源或未来真实流量。

每个特征的局部贡献为标准化值乘模型系数，全部贡献与截距相加得到对数几率，再经过 sigmoid 得到分数。提示词、HTTPS、路径等只解释模型计算，不证明因果关系。新增主机熵、数字比例、最长层长度等特征仍完全由网址得到。

初期 UCI 正常样本均无路径且使用 HTTPS，模型学习了数据集捷径。中期模型在同一批 PhreshPhish 保留测试样本上的正常误报率为 100%，改进后为 23.17%。本地 archive 保留中期源码和模型；它们不上传 GitHub。

## 检查与复现

~~~bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pyright --pythonpath "$(which python)"
python -m pytest -q
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m scripts.verify_reproducibility
~~~

最后一条会用同一数据和选定配置重新拟合一次，核对全部测试集预测，而不重新寻优。requirements-runtime-lock.txt 记录已验证环境的完整包版本，可用于复现。GitHub Actions 在提交和 PR 时运行静态检查、类型检查与合成样本功能测试，不下载训练数据。

## 文件结构

~~~text
app.py                       四个网页页面
phishing_url/features.py     离线解析与特征提取
phishing_url/data.py         多来源清理和分组划分
phishing_url/train.py        训练、参数曲线与最终评价
phishing_url/model.py        推断、特征贡献与版本检查
phishing_url/batch.py        CSV 文件校验和结果导出
phishing_url/storage.py      SQLite 模型版本与检测历史
phishing_url/external_eval.py 原始来源测试划分上的额外评价
scripts/                     下载与复现工具
tests/                      功能测试（含 Streamlit 页面测试）
~~~

课程报告、源代码 Word、周志、截图和实际检测记录仅保存在本地。个人信息、课程日期、成员贡献和个人学时应由提交者按实际情况填写。多人协作方式见 CONTRIBUTING.md；数据来源与引用见 DATA_SOURCE.md。
