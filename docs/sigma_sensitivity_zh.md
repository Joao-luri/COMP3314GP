# Sigma 敏感性实验笔记（实测）

**【实测】** 以下数字全部来自 `python scripts/run_sigma_sensitivity.py`。  
未改第一阶段默认 σ，也没有用 ARI 重选默认参数。

## 设置

- 数据与 stage1 相同：circles / moons / blobs，`data_seed=0`，`kmeans_seed=0`
- \(\sigma_0\)：第 7 近邻距离中位数（与 stage1 相同规则）
- 相对网格：\(r=\sigma/\sigma_0 \in \{0.25,0.35,0.5,0.7,1,1.4,2,2.8,4,5.6,8\}\)
- 真实标签只用于 ARI/NMI 和案例图着色

## 主结论

1. **circles：对过大的 σ 很敏感**  
   \(r\le 1.4\) 时 ARI = 1；\(r\ge 2\) 时 ARI ≈ 0（和原始 K-means 一样用直线切开）。  
   stage1 的 \(r=1\) 落在成功区间内，但离失败边界不远（下一个网格点 \(r=2\) 已失败）。

2. **moons：过大时逐渐变差，不是断崖**  
   \(r\le 2\) 仍为 ARI = 1；之后降到 0.81 → 0.47 → 0.37 → 0.30。

3. **blobs：在本网格上几乎一直成功；过小会数值失败**  
   \(r=0.25\) 报 `zero_row_norm`（图太碎，某些点在 top-k 特征空间里行范数≈0）；  
   \(r\ge 0.35\) 全部 ARI = 1。

4. **失败时 Y 长什么样（circles 案例图）**  
   - \(r=0.5\) 与 \(r=1\)：Y 上两簇收成单位圆上的两个点，ARI = 1  
   - \(r=4\)：Y 上变成两段混在一起的弧，真实两色交织；K-means 只能左右切开，ARI ≈ 0

5. **和原文“用 distortion 选 σ”的对照（本网格上的观察）**  
   - circles / moons：成功的小 σ 上，Y 的 K-means distortion 接近 0；σ 过大、ARI 崩掉时 distortion 升到几十。  
     因此在本网格上，**取 distortion 最小会落在高 ARI 区域**，与原文启发式一致。  
   - 但很小的 σ 上 eigengap 也接近 0（circles 在 \(r\le 0.35\) 时 eigengap ≈ 0），说明“distortion 很小”并不等于“理论条件很舒服”。  
   - blobs 的 \(r=0.25\) 在算 distortion 之前就数值失败了；原文搜索若碰到这类点需要单独处理。

## 看哪些文件

| 文件 | 内容 |
|---|---|
| `results/figures/sigma_sensitivity_ari.png` | ARI–\(r\) 曲线，竖虚线 = stage1 的 \(r=1\) |
| `results/figures/sigma_sensitivity_aux.png` | eigengap 与 Y distortion |
| `results/figures/sigma_sensitivity_cases.png` | circles 在 \(r=0.5,1,4\) 的原图 + Y |
| `results/tables/sigma_sensitivity.csv` | 全部数字与失败状态 |

## 对 critical analysis 可写的句子

- NJW 不是“设好 k 就稳”：σ 过大时，非凸结构（圆环）会退化成和普通 K-means 类似的错误划分。
- stage1 的近邻启发式在这三个数据上可用，但 circles 的安全裕度不大。
- 过小的 σ 可能直接触发数值失败（blobs），不能只看 ARI 曲线。

## 未做（避免和后续实验混在一起）

- 未改 \(k\)，未加噪声桥，未做密度差 / Meila–Shi
- 未实现原文脚注 3 的 90° K-means 初始化
- 只用一个 data seed；方差以后再补
