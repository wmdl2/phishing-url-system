# 数据来源与处理说明

## 1. UCI PhiUSIIL

- 来源：[UCI PhiUSIIL Phishing URL (Website)](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)，数据集编号 967。
- 作者：Arvind Prasad、Shalini Chandra；对应论文发表于 Computers & Security（2024）。
- 许可：CC BY 4.0。原始数据 235,795 行，使用 URL、label 两列；原始 0=钓鱼、1=正常。
- 原始 ZIP SHA-256：0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a。
- 网页内容相关特征均未使用，防止训练输入与实际检测条件不同。

## 2. PhreshPhish

- 来源：[PhreshPhish 数据集卡](https://huggingface.co/datasets/phreshphish/phreshphish)。
- 作者：Thomas Dalton、Hemanth Gowda、Girish Rao、Sachin Pargi、Alireza Hadj Khodabakhshi、Joseph Rombs、Stephan Jou、Manish Marwah。
- 论文：[PhreshPhish: A Real-World, High-Quality, Large-Scale Phishing Website Dataset and Benchmark](https://arxiv.org/abs/2507.10854)。
- 许可：CC BY 4.0，来源要求用于反钓鱼研究。仅提取 url、label；不运行或展示网页 HTML。
- 固定 Parquet 版本：748d45b35cca9cff94f6b1cac051f22cde5b0345。原始 train 共 498,255 行，全量作为补充开发来源；原始 test 固定种子 42，每类抽样 10,000 行。
- 下载清单、分片来源和本地哈希保存在 data/raw/phreshphish_manifest.json。该文件和原始数据不提交 GitHub。
- benign=正常、phish=钓鱼。所有 28 个模型输入由本项目共用网址提取函数重新计算。

## 3. 清理、划分与评价

合并 UCI 和 PhreshPhish train 后清理，按完整主机名分成约 60% 训练、20% 验证、20% 测试。来源只用于审计与分层划分，不是模型特征。相同原始网址去重；冲突标签网址整体移除。此处不是对所有语义等价的网址进行完全规范化去重。

PhreshPhish test 抽样只用于最后评价，剔除与全部开发数据重合的完整主机后保留 13,456 行。它检验未参与开发的主机样本，但训练也使用了同一来源的 train，因此不能称为对全新来源的证明。抽样比例及剔除会改变正负类比例，结果不能直接替代真实流量统计。

中期还曾用 EdgePhish-5G（[Zenodo record 19371661](https://zenodo.org/records/19371661)）作诊断性评价，其原始数据未再用于本次优化训练，也不在项目或 GitHub 中分发；中期已有统计仅在本地归档。

## 4. 限制

UCI 正常样本路径、参数为空且均使用 HTTPS，存在来源偏差。补充正常网页路径和网址结构特征降低了误报，但保留测试样本仍有约 23.17% 正常误报。完整主机分组不能消除同注册域的关联，网址文本也不能识别全部内容型钓鱼、跳转和新型攻击。历史测试不能证明未来网址始终有效。

逻辑回归参数说明：[scikit-learn 官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)。Parquet 按列读取说明：[Hugging Face 官方文档](https://huggingface.co/docs/dataset-viewer/en/parquet)。
