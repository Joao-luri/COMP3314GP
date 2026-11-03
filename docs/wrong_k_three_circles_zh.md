# 错误 \(k\) 实验：三个同心环（实测）

**【实测】** 来自 `python scripts/run_wrong_k_three_circles.py`。  
数据是我们自己生成的三个同心环，**不是**原文 Figure 1 的 threecircles-joined。

## 设置

- 半径 \(0.3, 0.65, 1.0\)，每环 150 点，噪声 0.03，`data_seed=0`
- \(\sigma\)：第 7 近邻中位数（与 stage1 同规则），\(\sigma_0\approx 0.0936\)
- 比较：K-means / NJW，各自 \(k=2\) 与 \(k=3\)
- 真实 3 环标签只用于评价与画图

## 主结论

1. **\(k\) 必须由人给定，给错会得到另一种“合理”划分**  
   NJW \(k=2\) 把结果收成 **inner vs middle+outer**。  
   相对这种自然合并，ARI = 1.000；相对 3 环真值，ARI = 0.570。  
   也就是说：算法没有“知道有三个环”，它只按你要的 \(k\) 在谱空间里切。

2. **Y embedding 解释了为什么合并的是中外环**  
   见 `wrong_k_three_circles_embedding.png`：\(k=2\) 时，内环在单位圆上单独一处，中环与外环靠得更近，所以 K-means 把后两者合成一类。

3. **K-means 在 \(k=2\) 和 \(k=3\) 上都失败**  
   ARI 相对 3 环真值约为 0，仍然是左右切开。

4. **同一 \(\sigma_0\) 下，NJW \(k=3\) 也没有完美恢复三环**  
   ARI = 0.522：内环分对了，中环与外环被切乱。  
   小范围探测（不是完整敏感性分析）：

   | \(r=\sigma/\sigma_0\) | NJW \(k=3\) ARI |
   |---|---|
   | 0.7 | 0.899 |
   | 1.0（默认） | 0.522 |
   | 1.4 | 0.294 |
   | ≥2 | ≈0 |

   说明：三环比两环更挑 \(\sigma\)；stage1 的近邻启发式在这里偏大。  
   **不要**据此改 stage1 默认值；这是 critical 证据，不是新的默认参数。

## 对 critical analysis 可写的句子

- 原文 Figure 1 里不同运行“唯一变化的参数是 \(k\)”，本实验用自建三环说明：\(k=2\) 与 \(k=3\) 回答的是不同问题。
- \(k=2\) 时 NJW 给出稳定、可解释的合并（内环 vs 中外环），不是随机乱分；但若人的意图是三个环，这就算错。
- \(k\) 与 \(\sigma\) 耦合：就算 \(k\) 给对（=3），\(\sigma\) 不合适仍会失败。

## 看哪些文件

| 文件 | 内容 |
|---|---|
| `results/figures/wrong_k_three_circles.png` | 真值 / K-means \(k=2\) / NJW \(k=2\) / NJW \(k=3\) |
| `results/figures/wrong_k_three_circles_embedding.png` | \(k=2\) 的 Y 空间 |
| `results/tables/wrong_k_three_circles.csv` | 主表 |
| `results/tables/wrong_k_three_circles_merges.csv` | \(k=2\) 与三种自然合并的 ARI |
| `results/tables/wrong_k_three_circles_k3_sigma_probe.csv` | \(k=3\) 附近 \(\sigma\) 探测 |
