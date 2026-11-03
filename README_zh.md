# NJW 谱聚类 — COMP3314 小组项目（中文说明）

本仓库是对 Ng, Jordan & Weiss (NIPS 2001) 论文

> **On Spectral Clustering: Analysis and an Algorithm**
> [官方 PDF](https://proceedings.neurips.cc/paper_files/paper/2001/file/801272ee79cfde7fa5960571fee36b9b-Paper.pdf)

的**从零实现**与一系列**自建数据上的实验研究**。

英文版概览见 [README.md](README.md)。更细的实验笔记、论文精读在 `docs/` 目录（多为中文）。

---

## 1. 这个项目在做什么

| 做了什么 | 没做什么 / 不要误解成什么 |
|---|---|
| 自己实现 NJW 算法（论文第 2 节） | **没有**调用 `sklearn.cluster.SpectralClustering` |
| 在自建 2D 玩具数据上对比 K-means / NJW / 若干 baseline | **不是**原文 Figure 1 的精确复现（原文未公布点集、n、σ） |
| 用 ARI / NMI 评价划分质量 | 原文没有报告 ARI；表里数字全是我们的实测 |
| 检查论文 Assumptions / Theorem 2 在我们数据上是否成立 | 不等于证明了论文定理 |

写作约定（贯穿 `docs/`）：

- **【原文】**：论文里写明的内容  
- **【实测】**：本仓库脚本跑出来的结果  
- **【观点 / 待验证】**：解释或猜想，写报告时不要当成事实  

---

## 2. 仓库目录总览

```
COMP3314GP/
├── README.md                 # 英文总览（安装、命令、结果表）
├── README_zh.md              # 本文件：中文详细说明
├── requirements.txt          # Python 依赖
├── .gitignore                # 忽略 .venv、缓存等
│
├── src/njw_spectral/         # ★ 核心库：算法、数据、baseline、画图
├── scripts/                  # ★ 可运行的实验入口（生成 results/）
├── tests/                    # 单元 / 健全性测试
├── docs/                     # 中文笔记与阶段性说明
└── results/
    ├── figures/              # 实验图（png + pdf）
    └── tables/               # 实验表（csv / json）
```

本地还会有 `.venv/`（虚拟环境，已在 `.gitignore` 中，**不会**上传到 GitHub）。

---

## 3. `src/njw_spectral/` — 核心代码库

对外入口在 `__init__.py`，主要导出算法相关函数。各文件职责如下。

### 3.1 `algorithm.py` — NJW 算法本体（最重要）

按论文第 2 节逐步实现：

1. `gaussian_affinity`：高斯相似度矩阵 \(A\)，对角线为 0  
2. `normalized_affinity`：度归一化，得到 \(L = D^{-1/2} A D^{-1/2}\)  
3. `top_eigenvectors`：取 \(L\) **代数上最大**的 \(k\) 个特征向量，组成 \(X\)  
4. `normalize_rows`：行归一化 \(X \to Y\)（可开关）  
5. 对 \(Y\) 的行做 K-means → 得到簇标签  

常用入口：

| 函数 / 类型 | 作用 |
|---|---|
| `njw_spectral_clustering(points, k, sigma, ...)` | 从点云一路跑完 NJW |
| `njw_from_affinity(...)` | 已有 \(A\) 时从归一化开始跑 |
| `sigma_knn_median(points, n_neighbors=7)` | 第 7 近邻距离中位数启发式选 σ |
| `KMeansConfig` | K-means 超参（`k-means++`、`n_init=10` 等） |
| `NJWResult` | 返回标签、\(A,L,X,Y\)、特征值、eigengap 等 |

特征分解用 `scipy.linalg.eigh`；最后一步 K-means 用 `sklearn.cluster.KMeans`。  
**相似度构造、归一化、特征向量选取、行归一化、输入校验** 都是我们自己写的。

### 3.2 `datasets.py` — 第一阶段及批判实验用数据

| 函数 | 用途 |
|---|---|
| `make_dataset(name, seed)` | stage1：`circles` / `moons` / `blobs` |
| `make_three_circles` | 三个同心环（错误 \(k\) 实验） |
| `make_moons_with_bridge` / `make_blobs_with_bridge` | 簇间加噪声桥 |
| `make_moons_with_gap_noise` | 缝隙处撒噪声云 |
| `make_unbalanced_blobs` | 大小不平衡的两团（测行归一化） |

### 3.3 `paper_like_data.py` — 按 Figure 1「样子」重建的数据

原文没有给坐标。这里按子图含义自建近似场景，例如：

- `make_twocircles`、`make_threecircles_joined`
- `make_lineandballs`、`make_fourclouds`、`make_squiggles`
- `make_nips`（按 N/I/P/S/2/0/0/1 笔画采样）

供 `scripts/run_paper_like.py` 使用。**不能称为复现了 Figure 1。**

### 3.4 `baselines.py` — 论文第 4 节相关对照方法

| 函数 | 含义（写报告时勿说错） |
|---|---|
| `meila_shi_clustering` | 用 \(P=D^{-1}A\) 的特征向量，**不做**行归一化，再 K-means |
| `connected_components_for_k` | 距离阈值图上找恰好 \(k\) 个连通分量 |
| `kannan_style_max_abs` | **简化对照**：按 \(\lvert X_{ij}\rvert\) 最大列归类；不是 Kannan 原文代码 |

### 3.5 `diagnostics.py` — 论文第 3 节理论量

在给定 \(A\) 与真值簇划分上，计算 Assumptions A1–A4、Theorem 2 相关量（δ、ε、条件是否成立、bound 与实测紧度等）。仅用于**分析**，不参与选参。

### 3.6 `plotting.py` — 画图工具

方法对比图、同心圆 pipeline 图（原始点 → \(X\) 的行 → \(Y\) 的行 → 最终划分）、统一存 png/pdf。展示用颜色会做 Hungarian 对齐，**不影响** ARI。

---

## 4. `scripts/` — 实验怎么跑

每个脚本跑完会覆盖/写入对应的 `results/` 文件。建议在虚拟环境里执行。

| 脚本 | 做什么 | 主要产出 |
|---|---|---|
| `run_stage1.py` | 三数据集上 K-means vs NJW；画对比图与 circles pipeline | `stage1_*` |
| `check_paper_assumptions.py` | 在同一批 stage1 跑次上检查 A1–A4 / Theorem 2 | `stage1_paper_assumptions.csv` |
| `run_sigma_sensitivity.py` | 相对 σ 网格敏感性（**不改** stage1 默认 σ） | `sigma_sensitivity_*` |
| `run_wrong_k_three_circles.py` | 三环上 \(k=2\) vs \(k=3\) | `wrong_k_three_circles*` |
| `run_noise_bridge.py` | 噪声桥 / 缝隙噪声 / 行归一化 ablation | `noise_bridge_*`、`density_imbalance_*` |
| `run_paper_like.py` | Figure-1 风格自建数据 + 多 baseline | `paper_like_*` |

常用命令：

```bash
python -m pytest tests -q
python scripts/run_stage1.py
python scripts/check_paper_assumptions.py
python scripts/run_sigma_sensitivity.py
python scripts/run_wrong_k_three_circles.py
python scripts/run_noise_bridge.py
python scripts/run_paper_like.py
```

---

## 5. `tests/` — 测试在保什么

| 文件 | 内容 |
|---|---|
| `test_njw.py` | \(A\) 对称对角为零、\(L\) 定义、特征向量正交、理想块对角情形、行归一化开关、非法输入、代数最大特征值选取等 |
| `test_baselines.py` | baseline 在简单数据上行为合理，且不依赖 `SpectralClustering` |

跑：`python -m pytest tests -q`。

---

## 6. `docs/` — 笔记（建议写 report 前先读）

| 文件 | 内容 |
|---|---|
| `paper_notes_zh.md` | 论文精读（动机、算法、理想情形、扰动理论、实验节） |
| `learning_notes_zh.md` | 符号、两种归一化、为何还要用 K-means、stage1 学习收获 |
| `sigma_sensitivity_zh.md` | σ 敏感性实验结论与读图指引 |
| `wrong_k_three_circles_zh.md` | 给错 \(k\) 时「合理但错误」的合并 |
| `noise_bridge_zh.md` | 噪声桥打垮连通分量、NJW 仍稳；行归一化小幅帮助 |
| `paper_like_experiments_zh.md` | Figure-1 风格尝试：数据对应、σ 搜索、baseline 定义、ARI 表 |
| `next_stages_zh.md` | 已完成 / 仍可选的后续（多种子、报告与 slides） |

---

## 7. `results/` — 图与表分别是什么

仓库里已提交当前跑出来的结果，方便组员直接看图；若你改了代码想更新，重新跑对应 `scripts/` 即可。

### 7.1 Stage 1

| 文件 | 内容 |
|---|---|
| `tables/stage1_results.csv` | ARI、NMI、σ、特征值、eigengap、种子与参数 |
| `tables/stage1_config.json` | 完整配置与库版本 |
| `tables/stage1_paper_assumptions.csv` | 论文假设诊断 |
| `figures/stage1_comparison.{png,pdf}` | 三数据集：真值 / K-means / NJW |
| `figures/stage1_circles_pipeline.{png,pdf}` | 同心圆四步 pipeline |

Stage1 单种子实测摘要（详见英文 README）：

| 数据 | K-means ARI | NJW ARI |
|---|---|---|
| 同心圆 | ≈0 | 1.000 |
| 双月 | 0.274 | 1.000 |
| 高斯团 | 1.000 | 1.000 |

### 7.2 Sigma 敏感性

| 文件 | 内容 |
|---|---|
| `tables/sigma_sensitivity.csv` | 相对网格上的 ARI / eigengap / Y distortion / 失败类型 |
| `figures/sigma_sensitivity_ari.*` | ARI–\(r\) 曲线（\(r=\sigma/\sigma_0\)） |
| `figures/sigma_sensitivity_aux.*` | 辅助曲线 |
| `figures/sigma_sensitivity_cases.*` | 同心圆成功/失败时 \(Y\) 长什么样 |

### 7.3 错误 \(k\)（三环）

| 文件 | 内容 |
|---|---|
| `tables/wrong_k_three_circles.csv` 等 | \(k=2/3\) 结果、合并分数、σ 探针 |
| `figures/wrong_k_three_circles.*` | 划分对比 |
| `figures/wrong_k_three_circles_embedding.*` | \(Y\) 空间解释为何合并中外环 |

### 7.4 噪声桥与密度

| 文件 | 内容 |
|---|---|
| `tables/noise_bridge_and_density.csv` | 桥 / 缝隙噪声 / 行归一化结果 |
| `figures/noise_bridge_moons.*` 等 | moons/blobs 场景图与 embedding |
| `figures/density_imbalance_row_norm.*` | 不平衡团上行归一化开/关 |

### 7.5 Paper-like（Figure 1 风格）

| 文件 | 内容 |
|---|---|
| `tables/paper_like_results.csv` | NJW 与各 baseline 的 ARI |
| `tables/paper_like_sigma_search.csv` | 按 Y 上 K-means distortion 搜 σ 的记录 |
| `figures/paper_like_njw.*` | NJW 结果拼图 |
| `figures/paper_like_baselines.*` | K-means / Meila–Shi / 连通分量 / Kannan 式 |

---

## 8. 环境安装

需要 Python 3.10+（开发时测过 3.14 等版本，见英文 README）。

```bash
git clone https://github.com/Joao-luri/COMP3314GP.git
cd COMP3314GP
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

依赖：`numpy`、`scipy`、`scikit-learn`、`matplotlib`、`pytest`。

若 joblib 提示找不到物理核数，可忽略，或设 `LOKY_MAX_CPU_COUNT=4`。

---

## 9. 实验设定要点（写报告常用）

- **数据**：stage1 为 `make_circles` / `make_moons` / `make_blobs`（参数见英文 README）；均为自建，`data_seed=0`。  
- **不做特征缩放**：K-means 与 NJW 看到同一套原始坐标、同一个 \(k\)。  
- **σ（stage1 默认）**：第 7 近邻距离中位数；**不是**原文「在网格上最小化 Y 的 K-means distortion」。paper-like 脚本才按 distortion 搜 σ。  
- **真值标签**：只用于设定 \(k\)、算 ARI/NMI、画图着色；**不用于**选默认 σ。  
- **K-means**：`k-means++`，`n_init=10`，`random_state=0`；原文脚注 3 的 90° 初始化未实现。  
- **规模**：稠密 \(O(n^2)\) 相似度 + 全特征分解，适合几千点量级的玩具实验。

---

## 10. 当前局限（批判分析可写）

- 多数实验仍是单 data seed / 单 K-means seed，尚无系统方差估计。  
- 玩具 2D、低噪声；circles/moons 上 eigengap 很小，结果可能脆弱（σ 实验已部分展示）。  
- \(k\) 必须事先给定；给错会得到另一种「看起来合理」的划分。  
- Theorem 2 条件在 circles/moons 上可失败，但 NJW 仍完美聚类——理论充分条件 ≠ 实际必要。  
- 原文 Figure 1 无法精确复现。

更多后续计划见 `docs/next_stages_zh.md`。

---

## 11. 组员协作（GitHub）

- 仓库：https://github.com/Joao-luri/COMP3314GP（Private）  
- 被邀请后：邮件 Accept → `git clone` → 建分支改 → PR 或约定好再 push `main`  
- 不要提交 `.venv/`、密钥、超大无关文件  
- 大改算法 / 改默认实验参数前先和组里对齐，避免覆盖彼此已写入报告的数字  

---

## 12. 建议阅读顺序

1. 本文件 + 英文 [README.md](README.md)（安装与命令）  
2. `docs/paper_notes_zh.md` → `docs/learning_notes_zh.md`  
3. 看图：`results/figures/stage1_comparison.png`、`stage1_circles_pipeline.png`  
4. 按兴趣读：`sigma_sensitivity_zh.md`、`wrong_k_three_circles_zh.md`、`noise_bridge_zh.md`、`paper_like_experiments_zh.md`  
5. 读代码从 `src/njw_spectral/algorithm.py` 开始  

有问题直接在仓库开 Issue，或在组群里对齐后再改默认实验。
