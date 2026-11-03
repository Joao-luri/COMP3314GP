# 按原文 Figure 1 的设定做的尝试（不是精确复现）

【实测】`python scripts/run_paper_like.py`。  
原文没有公布点坐标、样本量、噪声和 \(\sigma\) 网格，所以**不能**称为复现了 Figure 1。下面每一套点都是我们按子图标题和大致形状生成的，生成代码在 `src/njw_spectral/paper_like_data.py`。

## 和原文实验的对应

| 原文 | 我们的数据 | 我们做的方法 |
|---|---|---|
| (e)(i) twocircles | `make_twocircles` | NJW、K-means、Meila–Shi、连通分量、Kannan 式 |
| (f)(g)(j) threecircles-joined，\(k=2\) 和 \(k=3\) | `make_threecircles_joined`（三环 + 两段径向短桥；桥上的点标签为 −1，不进 ARI） | 同上，\(k=3\) 和 \(k=2\) 各跑一遍 |
| (b)(k) lineandballs | 稀疏上弧 + 两个密集团 | 同上 |
| (c) fourclouds，\(k=2\) | 四个团，左右各有一条竖桥；真值定义为左 vs 右 | 同上 |
| (d) squiggles | 四条不相交的波浪线（原文看起来更缠，我们没有做成交叉） | 同上 |
| (a)(l) nips，8 簇 | 沿 N,I,P,S,2,0,0,1 的笔画采样 | 同上 |
| (h) Y 的行 | 第一阶段的 circles pipeline 图，不在这次重复 | — |

## \(\sigma\) 怎么选（对齐原文，但网格是我们定的）

原文：在一组 \(\sigma^2\) 里选 **Y 上 K-means distortion 最小** 的值。没有给出网格。  
我们：\(\sigma = r\sigma_0\)，\(\sigma_0\) 为第 7 近邻中位数，\(r\) 取 `geomspace(0.35, 6, 12)`。选 distortion 最小的成功点。真实标签不参与选择。  
核方法（NJW、Meila–Shi、Kannan 式）共用这个 \(\sigma\)。K-means 不用 \(\sigma\)。连通分量自己搜距离阈值 \(T\)，取**最大的**能得到恰好 \(k\) 个分量的 \(T\)。

完整搜索记录：`results/tables/paper_like_sigma_search.csv`。

## Baseline 的定义（写进报告时不要说错）

| 方法 | 代码 | 和原文的关系 |
|---|---|---|
| NJW | `njw_spectral_clustering` | 我们的实现 |
| K-means | 直接在原始坐标上 | 原文 Figure 1i |
| Meila–Shi | `baselines.meila_shi_clustering` | 按原文：用 \(P=D^{-1}A\) 的特征向量，**不做**行归一化，再 K-means |
| 连通分量 | `baselines.connected_components_for_k` | 按原文：距离 \(\le T\) 连边 |
| Kannan 式 | `baselines.kannan_style_max_abs` | **不是** Kannan 等人的原代码。原文只说他们把簇对应到单个奇异向量。我们的做法是：在 \(P\) 的 \(k\) 个特征向量上，把每个点分给 \(\lvert X_{ij}\rvert\) 最大的那一列 |

K-means 设置与全项目一致：`k-means++`，`n_init=10`，`random_state=0`。原文脚注 3 的 90° 初始化没有用。

## 实测 ARI（相对我们自己定义的真值）

| 数据 | \(k\) | NJW | K-means | Meila–Shi | 连通分量 | Kannan 式 | 选出的 \(\sigma\) |
|---|---|---|---|---|---|---|---|
| twocircles | 2 | 1.000 | −0.003 | 1.000 | 1.000 | 1.000 | 0.048 |
| threecircles_joined | 3 | 0.336 | 0.010 | 0.336 | 0.336 | 0.336 | 0.053 |
| threecircles_joined | 2 | 0.079 | 0.006 | 0.079 | 0.079 | 0.079 | 0.053 |
| lineandballs | 3 | 1.000 | 0.679 | 1.000 | 1.000 | 1.000 | 0.039 |
| fourclouds | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.184 |
| squiggles | 4 | 1.000 | 0.328 | 1.000 | 1.000 | 0.713 | 0.093 |
| nips | 8 | 1.000 | 0.639 | 1.000 | 1.000 | 0.762 | 0.044 |

## 这些数字说明什么

1. **两圆上 K-means 失败、NJW 成功**，和 Figure 1e/1i 的定性结论一致。数据不是原文的点。
2. **我们的 threecircles-joined 没有复现出原文那种“三个环被分开”的结果。**  
   distortion 最小的 \(\sigma\) 很小。在这个 \(\sigma\) 下，内环和中环通过短桥连成一块，外环碎成两段；NJW、Meila–Shi、Kannan 式和连通分量给出**同一种**错误划分（ARI 都是 0.336）。  
   同一网格上把 \(\sigma\) 加大，ARI 最高也只有约 0.37，然后掉到接近 0。所以失败不只是“选到了最小的 \(\sigma\)”，这套桥的接法本身就没有给出原文那种干净的三环恢复。  
   这是我们的数据构造和原文图不一致，**不能**据此说原文 Figure 1g 是错的。
3. **Meila–Shi 在这批数据上和 NJW 的 ARI 相同**，没有复现原文 Figure 1k 那种 Meila–Shi 更差的现象。原文认为差别出现在各簇连接程度差很大时；我们的 lineandballs 可能还不够“不均衡”。
4. **Kannan 式在 squiggles（0.713）和 nips（0.762）上低于 NJW**，和原文“把簇绑到单个向量上经常较差”的方向一致，但实现不是他们的原算法，图也不是 Figure 1l。
5. **连通分量在分开得干净的数据上也能得到 ARI=1**（存在一个刚好切出 \(k\) 个分量的阈值）。它在三环上和 NJW 一起失败，没有复现原文那个“单点簇”的具体画面。

## 看哪里

- 总表：`results/tables/paper_like_results.csv`
- \(\sigma\) 搜索：`results/tables/paper_like_sigma_search.csv`
- NJW 总览：`results/figures/paper_like_njw.png`
- 三个设定上的五种方法：`results/figures/paper_like_baselines.png`

## 写报告时的句子

可以说：我们按 Figure 1 的问题类型自己生成了数据，用原文的 distortion 规则选 \(\sigma\)，并实现了文中提到的对照方法。  
不可以说：我们复现了 Figure 1，或 Meila–Shi / Kannan / 连通分量的结果与原文图一致。
