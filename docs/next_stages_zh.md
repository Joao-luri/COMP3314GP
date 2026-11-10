# 后续阶段计划（尚未实现）

第一阶段的结论只基于一个 data seed、一个 K-means seed 和一个 σ 启发式。以下各项都是计划，没有结果。

## 2. Sigma sensitivity — 已完成（相对网格）

见 `docs/sigma_sensitivity_zh.md` 与 `scripts/run_sigma_sensitivity.py`。  
第一阶段默认 σ 未改；ARI 未用于选默认参数。

仍可选的后续：更密的网格、多种子、原文 90° K-means 初始化、把 distortion 自动选 σ 写成单独对照实验。

## 2b. 三个环 + 错误 k — 已完成

见 `docs/wrong_k_three_circles_zh.md` 与 `scripts/run_wrong_k_three_circles.py`。  
自建三环（非原文 threecircles-joined）。NJW \(k=2\) 稳定合并为 inner vs middle+outer；默认 \(\sigma\) 下 \(k=3\) 也未完美。

仍可选：带噪声桥的 threecircles-joined 风格；连通分量 baseline。

## 3. 噪声桥 + 行归一化 — 已完成（部分）

见 `docs/noise_bridge_zh.md`。  
要点：细/强桥与缝隙噪声下 NJW ARI 仍为 1，连通分量（搜阈值使分量数=k）ARI 掉到 0；质量不平衡时 row_normalize 开/关为 1.000 / 0.956。

Meila–Shi（\(P=D^{-1}A\)，不做行归一化）和连通分量已在 `scripts/run_paper_like.py` 里作为 baseline。Kannan 只做了原文那句话对应的简化版，见 `docs/paper_like_experiments_zh.md`。原文 Figure 1 的点集没有公布，该脚本用的是我们按图意重建的数据。

## 4. 适量重复、90° 初始化、规模 — 已完成

见 `docs/limitations_experiments_zh.md`（`run_multi_seed.py`、`run_kmeans_init.py`、`run_scaling.py`）。  
10 个 data seed × 固定 K-means seed 与固定 data seed × 10 个 K-means seed 分开报告；均值、标准差、最小/最大都在 `multi_seed_summary.csv`。

仍可选：稀疏 kNN 图 + `eigsh` 的规模对照；改变噪声水平/点数而不只是 data seed。

## 5. Report 与 slides

- 3–4 页英文 LaTeX：动机（K-means 的凸性假设）→ 算法 → 理想情形和 Theorem 2 的假设 → 我们的实验 → 批判分析 → 讨论。
- 批判点候选（需要用实验支撑或明确标为观点）：σ 的选择对结果的影响；k 需要事先给定；Theorem 2 的假设难以在实际数据上检验；O(n²) 内存和 O(n³) 特征分解的规模限制；小 eigengap 下的稳定性。
- 9 分钟 slides：逐页列出讲解要点和负责的组员；proposed extension 与已实现内容分开标注。
