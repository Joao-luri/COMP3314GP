# 噪声桥 / 密度不均实验笔记（实测）

**【实测】** `python scripts/run_noise_bridge.py`。自建数据，不是原文数据。

## 这个实验要改什么

| 改动 | 说明 |
|---|---|
| 新增数据生成 | `datasets.py`：`make_moons_with_bridge`、`make_moons_with_gap_noise`、`make_blobs_with_bridge`、`make_unbalanced_blobs` |
| 新增脚本 | `scripts/run_noise_bridge.py` |
| 新增结果 | `results/figures/noise_bridge_*.png`、`density_imbalance_row_norm.png`、`results/tables/noise_bridge_and_density.csv` |
| **不改** | stage1 / sigma / wrong-k 的默认设置与结果 |

ARI 在有噪声点时**只在原始簇点上计算**（噪声点标签 = −1，不进 ARI）。

## 能证明什么 / 不能证明什么

### 能支持的结论

1. **细噪声桥会打垮“连通分量”式方法，但不一定打垮 NJW**  
   在干净双月 / 双团上，NJW 与连通分量（搜索阈值使分量数 = \(k\)）都是 ARI = 1。  
   一旦加上桥或缝隙噪声云：

   | 设置 | NJW ARI | 连通分量 ARI | 分量数 |
   |---|---|---|---|
   | moons clean | 1.000 | 1.000 | 2 |
   | moons thin bridge (25) | 1.000 | 0.000 | 2 |
   | moons strong bridge (80) | 1.000 | 0.000 | 2 |
   | moons gap noise (120) | 1.000 | 0.000 | 2 |
   | blobs clean | 1.000 | 1.000 | 2 |
   | blobs dense corridor (150) | 1.000 | 0.000 | 2 |

   连通分量仍然能凑出 2 个分量，但划分与真值无关（ARI = 0）——和原文 Figure 1j“分出一个单点簇”的失败模式同类。  
   NJW 用的是软相似度 + 归一化割思路，**倾向于切断稀疏桥**，所以在这些设置下仍保持 ARI = 1。

2. **这与原文对 connected components 的批评一致，并补充了我们的实测**  
   原文说阈值图方法很不鲁棒；我们在噪声桥上复现了“不鲁棒”，同时看到 NJW 更稳。

3. **行归一化在质量不平衡时有小幅帮助（本数据）**  
   350 vs 50 的两个高斯团：`row_normalize=True` → ARI = 1.000；`False` → 0.956。  
   有差异，但不剧烈；不能夸大成“没有行归一化就会崩”。

### 不能声称的

- **不能说“噪声桥一定让 NJW 失败”**——在本实验的桥/噪声云强度下，NJW 没有失败。
- **不能说已实现 Meila–Shi 全文对比**——这里只做了 NJW 自己的 row normalize 开关，没有实现 \(D^{-1}A\) 那条算法。
- **不能说复现了原文 threecircles-joined**——数据是我们生成的。

## 先看哪些图

1. `results/figures/noise_bridge_moons.png` — 干净 / 细桥 / 强桥 / 噪声云上的 NJW  
2. `results/figures/noise_bridge_blobs.png` — 双团 + 走廊  
3. `results/figures/density_imbalance_row_norm.png` — 行归一化开关  
4. 数字表：`results/tables/noise_bridge_and_density.csv`

## 对 critical analysis 可写的句子

- 谱聚类依赖的是带权图上的割，不是“有没有一条路径连上”；因此稀疏噪声桥更伤连通分量，而未必伤 NJW。
- 原文用连通分量当负面对照是有道理的；我们的桥实验支持这一点。
- 若噪声把缝隙填成厚走廊、或 \(\sigma\) 大到簇间亲和本身很高，NJW 仍可能失败——那是 \(\sigma\) 敏感性实验已经覆盖的机制，不必和“细桥”混为一谈。
