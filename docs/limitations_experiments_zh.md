# 局限性补充实验：规模、90° 初始化、多种子（实测）

**【实测】** 三个脚本，全部是自建数据；**不改**任何已有默认设置，已有 `results/` 里的表重跑后逐字节不变。

```bash
python scripts/run_scaling.py --max-n 8000   # 约 2.5 分钟；--replot 只重画图
python scripts/run_kmeans_init.py            # 约 10 秒
python scripts/run_multi_seed.py             # 约 1 分钟
```

| 脚本 | 图 | 表 |
|---|---|---|
| `run_scaling.py` | `figures/scaling_time_memory.*` | `tables/scaling.csv`、`scaling_meta.json`（含机器、库版本、拟合斜率） |
| `run_kmeans_init.py` | `figures/kmeans_init_comparison.*` | `tables/kmeans_init_runs.csv`、`kmeans_init_summary.csv` |
| `run_multi_seed.py` | `figures/multi_seed_headlines.*`、`multi_seed_sigma_curves.*` | `tables/multi_seed_runs.csv`、`multi_seed_summary.csv` |

代码改动只有一处：`algorithm.py` 新增 `orthogonal_init` 和 `KMeansConfig(init="orthogonal")`（可选项，默认仍是 `k-means++`、`n_init=10`），`tests/test_njw.py` 加了两条测试。

---

## 1. O(n²) 内存与 O(n³) 特征分解

**设置**：同心圆（与 stage1 相同的 `make_circles`，noise 0.05、factor 0.5），\(k=2\)，σ 用 stage1 的第 7 近邻中位数规则。\(n\) 从 250 到 8000，每个 \(n\) 在**独立子进程**里跑，内存取峰值 RSS 的增量；时间取多次重复的最小值（\(n\le2000\) 重复 3 次）。机器为 Apple Silicon 笔记本（10 核，arm64），见 `scaling_meta.json`。

**结果**（斜率是 \(n\ge1000\) 上的 log-log 拟合）：

| 量 | 实测斜率 | n = 8000 时 |
|---|---|---|
| 峰值内存增量 | 1.93（≈2） | 1551 MB |
| 第 3 步 top-(k+1) `eigh` | 3.46 | 48.2 s（占总时间 94%） |
| 第 1 步 构造 \(A\) | 2.58 | 1.1 s |
| 第 2 步 构造 \(L\) | 2.64 | 1.8 s |
| 第 4–5 步 行归一化 + K-means | 1.37 | 0.06 s |
| 参照：完整 `eigh`（只跑到 4000） | 2.86 | 4000 时 9.9 s |

**能支持的结论**

1. 内存严格按 \(n^2\) 增长，峰值约为 **3.8 个** \(n\times n\) float64 稠密矩阵（距离平方、\(A\)、\(L\) 及其对称化临时量）。按此外推，**8 GB 内存在 \(n\approx1.7\) 万点时耗尽，16 GB 在 \(n\approx2.4\) 万**。这是我们这个稠密实现的硬上限，不是某个参数没调好。
2. 时间几乎全花在特征分解上。虽然只要 \(k+1\) 个特征向量，`scipy.linalg.eigh(subset_by_index=...)` 仍要先把整个 \(L\) 化成三对角形，所以是 \(O(n^3)\)：实测指数 2.9–3.5（BLAS 多线程和缓存会让小 \(n\) 偏离理论值）。按 \(n^3\) 外推，\(n=2\) 万约需 12 分钟（**外推，未实测**）。
3. 前两步虽然是 \(O(n^2)\)，但在 8000 点时也已到秒级；`normalized_affinity` 里的对称性检查 `allclose(A, A.T)` 访问转置，缓存不友好。

**不能声称的**

- 不能说"谱聚类本身做不了大数据"。稀疏 kNN 图 + Lanczos/ARPACK（`eigsh`）、Nyström 近似等可以绕开这两个上限，**我们没有实现也没有测**，只能写成 proposed extension。
- 绝对时间依赖机器和 BLAS，只有斜率和比例有意义。

**顺带发现**：\(n=250\) 时同一个 σ 规则在同心圆上 ARI ≈ 0。点越少，第 7 近邻距离越大（σ = 0.161 vs \(n=400\) 时 0.105），两环被连起来。σ 规则在 \(n\) 变化时并不"自动适配"。

---

## 2. 原文脚注 3 的 90° 初始化

**实现**（`orthogonal_init`）：第一个中心是随机选的一行 \(Y\)；之后每次选"与已选中心最大余弦最小"的那一行，即在最差情况下最接近 90°。只跑一次 K-means（`n_init=1`），不重启。
**注意**：这一行是按 `paper_notes_zh.md` 里我们对脚注 3 的转述实现的，写报告前请对照原文 PDF 再核一次措辞（尤其是用余弦还是余弦绝对值）。

**设置**：每个任务的 \(Y\) 只算一次（\(Y\) 与 K-means 无关），然后只重复最后的 K-means，30 个初始化种子 × 4 种方案：

- `k-means++ ×10`：本项目默认
- `k-means++ ×1`
- `random ×1`
- `90° ×1`：原文做法

任务：stage1 三个数据集（σ₀）；三环 \(k=3\)（σ₀ 和 0.7σ₀）；paper-like 的 threecircles-joined（\(k=3\)）、squiggles（\(k=4\)）、nips（\(k=8\)），σ 按原文规则（\(Y\) 上 distortion 最小）选。

**结果**（ARI 均值；"best" = 达到该任务所有运行中最低 distortion 的比例）：

| 任务 | k-means++ ×10 | k-means++ ×1 | random ×1 | 90° ×1 |
|---|---|---|---|---|
| circles / moons / blobs | 1.000 | 1.000 | 1.000 | 1.000 |
| 三环 k=3, 0.7σ₀ | 0.899 | 0.899 | **0.780 ± 0.200**（best 73%） | 0.899 |
| threecircles-joined k=3 | 1.000 | 1.000 | **0.866 ± 0.209**（best 70%） | 1.000 |
| squiggles k=4 | 1.000 | 1.000 | **0.943 ± 0.129** | 1.000 |
| nips k=8 | 1.000 | 1.000 | **0.831 ± 0.129**（best 27%） | 1.000 |
| 三环 k=3, σ₀ | 0.522 | 0.538 | 0.541 ± 0.095 | 0.536 |

**能支持的结论**

1. **原文的 90° 初始化只跑一次，就和我们 10 次重启的 k-means++ 一样好**：ARI 在每个任务上都与默认相同或略高；8 个任务里有 7 个 30/30 到达最低 distortion，例外是下面第 3 点的三环 σ₀（17/30）。所以项目里"没实现 90° 初始化"这个偏离原文的地方，**不影响我们已报告的数字**。
2. 真正会出问题的是**单次随机初始化**，而且 \(k\) 越大越明显（nips \(k=8\)：只有 27% 的运行到达最优）。原文说"随机初始化加少量重启也给出相同结果"，在我们的数据上要加上"重启"这个条件才成立。90° 规则有效的原因与理论一致：理想情形下 \(Y\) 的行聚在 \(k\) 个**相互正交**的点附近，按"最接近 90°"挑点几乎必然每簇挑到一个。
3. **三环 σ₀ 是反例，说明问题不在初始化**：所有方案都卡在 ARI ≈ 0.52–0.57。更关键的是，30 次 random 中有 1 次得到 **ARI = 1.000**，但它的 distortion 是 73.9，比最优的 44.2 **差 67%**。也就是说，在这个 σ 下，"让 \(Y\) 上 distortion 最小"本身就偏向错误划分。这个准则正是 K-means 的目标，也是原文选 σ 的准则。这个失败来自 σ/embedding，不是 K-means 没找到最优。

---

## 3. 多种子：幻灯片上的数字有多稳

**设置**：两种随机性分开：

- **data**：data seed 0–9，K-means seed 固定为 0（每次重新生成点集，σ₀ 按同一规则重算）
- **kmeans**：data seed 固定为 0，K-means seed 0–9

种子 0 正好复现幻灯片上的数字（星号）。

**结果**（ARI，data seed 0–9 的均值 ± 标准差 [最小, 最大]；kmeans 那组见表）：

| 幻灯片上的说法 | 种子 0 | 10 个 data seed | 10 个 K-means seed |
|---|---|---|---|
| circles：K-means / NJW | −0.00 / 1.00 | −0.00 / 1.00 ± 0.00 | 完全不变 |
| moons：K-means / NJW | 0.27 / 1.00 | 0.25 ± 0.02 / 1.00 ± 0.00 | 不变 |
| moons 2.8σ₀ | 0.81 | **0.83 ± 0.07** [0.73, 0.98] | 0.81 ± 0 |
| moons 8σ₀ | 0.30 | 0.29 ± 0.01 | 0.30 ± 0 |
| circles 2σ₀ 全部崩到 0 | −0.00 | 10/10 为 0 | 10/10 为 0 |
| 80 点噪声桥：NJW / 连通分量 | 1.00 / 0.00 | **0.97 ± 0.05** [0.87, 1.00] / 0.00 ± 0 | 1.00 / 0.00 |
| 三环 k=3, σ₀ | 0.52 | 0.52 ± 0.14 [0.38, 0.89] | 0.52 ± 0 |
| 三环 k=3, 0.7σ₀ | 0.90 | **0.86 ± 0.17** [0.56, 1.00]，5/10 完美 | 0.90 ± 0 |

**能支持的结论**

1. **变化几乎全部来自数据，不来自 K-means 初始化**。固定数据换 10 个 K-means 种子，所有 NJW 数字标准差都是 0（因为 `n_init=10`）。这和第 2 节一致。
2. **定性结论都站得住**：K-means 在 circles/moons 上失败、NJW 在 σ₀ 上 10/10 完美；circles 在 2σ₀ 处 10/10 崩溃；连通分量在噪声桥上 10/10 为 0；三环 \(k=3\) 在 σ₀ 下 10/10 都不完美。
3. **有两个数字是"偏好运"的种子，写报告要改口径**：
   - 噪声桥上 NJW 不是"总是 1.00"，而是 **8/10 完美、最差 0.87**。"比连通分量鲁棒得多"仍然成立，"不受噪声桥影响"不成立。
   - "把 σ 调到 0.7σ₀ 就能修好三环"只在 **5/10** 个种子上成立，最差 0.56。更准确的说法：缩小 σ 平均有帮助（0.52 → 0.86），但不存在一个对所有点集都安全的固定倍数。这反而加强了"σ 没有可靠的自动选法"这一批判点。
4. circles 的 σ 曲线上，1.4σ₀ 处只有种子 0 是 1.00，其余 9 个种子都在 0.00–0.66 之间（`multi_seed_sigma_curves.png` 右上）。单种子的 σ 图**高估了**circles 的安全窗口：实际上 σ₀ 往上不到 1.4 倍就开始坏。

**不能声称的**

- 10 个种子只够给出量级和失败频率，不够做显著性检验。
- 只改了 data seed，没有改数据的噪声水平、点数或形状参数。

---

## 对 critical analysis 可写的句子（草稿）

- Our dense implementation needs ≈3.8 n×n float64 matrices at peak: memory grows as n² (measured slope 1.93), which caps a 16 GB machine at roughly 24k points. The eigendecomposition takes 94% of the time at n = 8000 and grows as ≈n³ even though only k+1 eigenvectors are needed.
- The paper's 90° initialization with a single run matched 10 restarts of k-means++ on all our tasks; a single random initialization did not (nips, k = 8: mean ARI 0.83, best distortion in only 27% of runs).
- Over 10 data seeds, all qualitative conclusions hold, but two single-seed numbers were optimistic: NJW on the 80-point bridge is perfect in 8/10 seeds (worst 0.87), and 0.7σ₀ fixes the three rings in only 5/10 seeds.
- On three rings at σ₀, a partition with ARI 1.0 has 67% higher Y-distortion than the one K-means returns (ARI 0.52), so minimizing distortion, which is also the paper's criterion for choosing σ, can prefer the wrong answer.
