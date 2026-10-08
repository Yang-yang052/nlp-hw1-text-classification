# 作业一 文本分类

本仓库包含实验报告 PDF、完整实验代码和独立运行说明；报告记录了六组真实实验结果。
Task 1 按更正后的要求使用**两种**词袋表示；全部方法共享 NYT 的 80% / 10% / 10% 划分。

- [实验报告 PDF](实验报告.pdf)：八页，含方法、验证记录、混淆矩阵、误分类案例、序列截断分析和复现说明。
- [代码运行说明](运行说明.md)：环境安装、数据准备、训练、指标核验、报告生成和常见问题。
- [完整训练代码](experiment.py)：两种词袋、三种 100 维平均词向量、BERT 微调。

运行前从课程材料取得 `nyt.csv` 和 `ag.csv`，放在项目根目录。
训练和核验后会生成 `results/summary.csv` 与逐样本预测；这些文件保存在本地。

## 实测结果

| 方法 | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Binary Bag of Words | 0.9896 | 0.9779 |
| Word Frequency | 0.9896 | 0.9752 |
| GloVe 100d + LR | 0.9870 | 0.9691 |
| Word2Vec AG News 100d + LR | 0.9809 | 0.9547 |
| Word2Vec NYT 100d + LR | 0.9852 | 0.9624 |
| BERT base uncased | 0.9783 | 0.9542 |

NYT 共 11,519 条，训练、验证、测试集分别为 9,215 / 1,152 / 1,152 条。
随机种子为 42。BERT 的 max_length=64，完整训练 3 epochs，按验证 Macro-F1 选中第 2 轮检查点。
结果由导出的逐样本预测重新核验；原始数据的重复文本及评价局限在报告中说明。

## 快速运行

在项目根目录使用 Python 3.12：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_glove.py
.\.venv\Scripts\python.exe experiment.py --glove assets/glove.6B.100d.txt
.\.venv\Scripts\python.exe analyze_results.py
.\.venv\Scripts\python.exe validate_results.py
.\.venv\Scripts\python.exe report.py
```

Linux/macOS 将执行文件路径改为 `.venv/bin/python`。其他运行方式、CPU 混合精度条件及字体设置见运行说明。
首次训练会下载 BERT；GloVe 和模型文件较大，保存在被 Git 忽略的 assets 或模型目录。

## 仓库文件

| 文件 | 用途 |
| --- | --- |
| `experiment.py` | 六组实验与固定划分 |
| `analyze_results.py` | 长度、截断比例、错误方向与实例分析 |
| `validate_results.py` | 从预测复算指标、核验数据哈希和划分 |
| `test_core.py` | 表示方法和数据隔离的核心检查 |
| `report.py` | 使用真实结果生成并嵌入字体的 PDF |
| `download_glove.py` | 官方 GloVe 下载 |
| `download_resumable.py` | 可选断点分块下载 |
| `run_all.ps1` | Windows 一键运行 |
| `package_submission.py` | 生成 ZIP 和 SHA256 提交清单 |
| 课程提供的 `nyt.csv` / `ag.csv` | 运行前自行放入根目录 |
| 运行后生成的 `results/` | 实测指标、原始行号、分类报告和预测 |

无需下载大模型即可阅读 PDF 中的完整结果。拥有本地预测输出后可运行 `validate_results.py` 核验。
完整重训的依赖和参数均已记录，课程数据、输出目录、预训练权重和本地认证信息保存在本地。
