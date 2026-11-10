# 口播稿 · COMP3314_Spectral_Slides_v4.pptx

正片是第 1–20 页，大约 10 分钟。第 21–26 页是 backup，被问到再翻，不计入正片。

带 “(click)” 的几页在文件里是连续几张，用 Morph 接起来，算同一个人的一段。每段前面留了讲解人，直接填名字。

稿子是英文，对着片子念。第 10–16 页的 90 度讲法就是已经说定的那版。

| 页 | 内容 | 大约 | 讲解人 |
|---|---|---|---|
| 1 | 标题 | 10 s | |
| 2 | K-means 为什么切不开月牙 | 45 s | |
| 3 | 方法改的是坐标 | 25 s | |
| 4–6 | 六个点变成矩阵 A | 65 s | |
| 7–9 | A → L → X | 70 s | |
| 10–13 | 为什么做行归一化 | 85 s | |
| 14–16 | 为什么是 90 度、为什么能分开 | 70 s | |
| 17 | 和论文 Figure 1 的对照 | 65 s | |
| 18 | σ | 75 s | |
| 19 | k | 60 s | |
| 20 | 收束 | 50 s | |
| 21–26 | backup | 被问再翻 | |

---

## 1 · On Spectral Clustering

讲解人：

[~10 s] Title. Ng, Jordan and Weiss, NIPS 2001. We start straight from the example on the right.

## 2 · Why does K-means get this wrong?

讲解人：

[45 s] Here are two moons, 400 points we generated. On the left are the groups we would expect. On the right is K-means with k = 2. K-means puts every point with its nearest centroid, the two crosses, so the boundary between the two clusters is a straight line, the dashed one. The moons interlock, so any straight line cuts through both. ARI is 0.27. ARI 1 would mean identical to the expected groups. The problem: these groups are defined by which points are connected to which, not by distance to a centre.

## 3 · What does the new method change?

讲解人：

[25 s] The method keeps K-means as its last step. What it changes is the representation. From the points we build a similarity graph: who is close to whom. From the graph we compute new coordinates, one row per point. In those coordinates each moon collapses to almost a single point, and K-means is trivial. The next three slides open up these steps on a tiny example with six points, so that every number is visible.

## 4–6 · How do points become a matrix?

讲解人：

[65 s, two clicks] To see every number, take six points, sigma 0.7. A is a 6 by 6 table: entry i, j says how similar point i and point j are. The diagonal is zero.

(click) Points 1 and 2 are 0.85 apart. Plug into the Gaussian: exp of minus 0.85 squared over 2 sigma squared gives 0.47. A is symmetric, so the same number goes in row 2, column 1.

(click) Fill in every pair. Within each group the weights are around 0.4 to 0.6. Between the groups they are almost zero, except one weak link, 0.07 between points 2 and 4. So A has two blocks. That block structure is what the rest of the algorithm turns into coordinates.

## 7–9 · How do we get new coordinates?

讲解人：

[70 s, two clicks] Step two. The degree dᵢ is the row sum of A: how connected point i is overall. Point 2 has the largest degree because it also touches the other group.

(click) L divides each weight by the square root of the two degrees. For example L₁₂ = 0.47 over the square root of 0.88 times 1.09, which is 0.49. L is still 6 by 6 and keeps the two blocks.

(click) Now the eigenvalues of L: 1.00 and 0.94, then a big drop to negative values. We keep the k = 2 largest and put their eigenvectors side by side. That is X, 6 by 2. Read it by rows: row i is the new coordinate of point i. Point 2 is now (0.42, 0.42). Look at the second column: positive on the first group, negative on the second. The groups are already separated, but the rows still have different lengths.

## 10–13 · Why normalize the rows?

讲解人：

这一页先不要解释为什么是 90 度。只讲方向和长度。

[85 s, three clicks] On the left is X. The first group's three rows are about (0.40, 0.42). The second group's rows are about (0.42, −0.40). Within a group, the rows already point the same way. On the right, the six points do the same: two directions, but at different distances from the origin.

(click) This column is the length of each row. The six lengths sit between 0.56 and 0.60, and they differ inside a group. The length does not say which group a point belongs to. Leaving it in only adds noise for K-means.

(click) Divide each row by its length. Every point slides along its own direction onto the unit circle. The first group stacks near (0.70, 0.72). The second stacks near (0.72, −0.69). Each group is now one point, and the two points are about a right angle apart. The next slide says why the angle is 90 degrees.

(click) Those two points are far apart, so K-means has nothing hard to do. The colours go back to the original six points.

## 14–16 · Why does this recover the clusters?

讲解人：

90 度是这一页才解释的。图上的两根虚线是转过角度的坐标轴，为了和上一页对齐。两组并不躺在画出来的横轴和纵轴上，夹角仍然是 90 度。

[70 s, two clicks] Look at the two off-blocks boxed in A. Delete every edge between the groups. The matrix falls into two separate blocks, and each block has an eigenvalue of exactly 1. We can choose the coordinates so the first group is (a positive number, 0) and the second group is (0, a positive number). That is the horizontal axis and the vertical axis, so the angle is 90 degrees. Divide by the length, and the first group becomes the point (1, 0), the second the point (0, 1). The dashed lines on this figure are those same two axes after a rotation, turned so they line up with the previous slide. A rotation does not change the angle. Lambda 1 and lambda 2 are both 1.00.

(click) Put the weak edge of 0.07 back. Lambda 2 drops from 1.00 to 0.94, and the six points barely leave those two positions.

(click) Multiply the edges between the groups by ten, until they are as strong as the edges inside a group. Lambda 2 drops to 0.65. Points 2 and 4, the ones that carry that edge, are pulled off their own points toward each other. A group is no longer a single point.

One sentence on the theorem, and stop. If the edges between groups are weak, and the gap between lambda 2 and lambda 3 is large, Y stays near these two perpendicular points. On the figure that gap is about 1.44 with no cross edges, about 1.37 with the weak edge, and about 0.83 after the edges are multiplied by ten.

## 17 · Did we see the same effect?

讲解人：

[65 s] Did we see the same thing? Left, the paper's Figure 1. Right, our own implementation on data we rebuilt. The paper released neither the points nor sigma, so this is a qualitative comparison, and the ARI numbers are ours. Top row, two circles: the paper shows NJW succeeding and K-means failing. We get exactly that, 1.00 against 0.00. Bottom row, three rings joined by short bridges: NJW recovers the three rings here too. One sentence, then move on. Across everything we ran: NJW beats K-means on circles and moons, ties on blobs, and gets 1.00 on the rebuilt Figure 1 settings.

## 18 · One σ, and how the paper chooses it

讲解人：

[75 s] Sigma sets the scale of every link. The formula is on the right: Aᵢⱼ is exp of minus the squared distance over 2 sigma squared. The diagonal is 0. A small sigma only connects nearby points. A large sigma also connects points that are far apart. The whole graph gets one sigma.

On the left, the moons at our default sigma: ARI 1.00. At 2.8 times that sigma, ARI is 0.81. The dashed curve is the two circles. At 2 sigma-zero it drops to 0, and the cut looks like ordinary K-means.

The paper chooses sigma by trying several values. For each one it runs K-means on the rows of Y and keeps the sigma with the smallest distortion. Distortion is the sum of squared distances from those rows to their centres. True labels are not used.

On this grid, distortion is about 0 while ARI is 1, and it rises to about 30 when ARI collapses. So the rule lands in the successful range. It returns one winner. It does not tell you that doubling sigma already fails on the circles.

## 19 · k is given by hand

讲解人：

[60 s] The algorithm does not decide how many groups there are. We tell it k.

Left: three rings, the groups we want. Middle: we ask for k = 2. It does not scatter the points at random. The inner ring is one group, and the middle and outer rings are merged. Against that merge, ARI is 1.00. Against the three rings, ARI is 0.57. So k = 2 is the right answer to a different question. Right: we ask for k = 3, the true number. At our default sigma, ARI is still only 0.52. The inner ring comes out, and the outer two are mixed.

One more number, and then stop. Same k = 3: shrink sigma to 0.7 times the default, and ARI rises to 0.90. The right count is not enough. The scale has to fit as well.

## 20 · Where this method fits

讲解人：

[50 s] Where does this method fit? When groups are connected through local neighbours, but a straight line cannot separate them: the rings and the moons. The graph needs one length scale, and k has to be known. When the clusters are already separated round clouds, K-means is enough. When the scales differ, or k is asking a different question, that is the previous two slides.

The paper matters because of the recipe. In 2001, spectral clustering was already around, but authors turned eigenvectors into clusters in different ways. Ng, Jordan and Weiss fixed one sequence that people then actually used: an affinity graph, the top k eigenvectors, each row normalized, then K-means. Normalization is the step that makes a whole connected piece into one point, which is what K-means can cut. On our moons that takes ARI from 0.27 to 1.00.

---

## Backup（不排进正片）

被问到再打开。没有写死的口播，看表回答。

**21 · Theorem 2 on our runs.** 圆环和月牙上定理条件不成立，NJW 的 ARI 仍是 1.00。Blobs 上条件成立，上界比实测大约松十个数量级。定理是充分条件，不是必要条件。

**22 · Connections to the course.** K-means、PCA、核、两种归一化分别对应课程里的哪一块。行归一化是 X → Y，和 A → L 的度归一化不是同一步。

**23 · Questions we prepared.** 五条预备问答：为什么最后还要 K-means；为什么保留第一个特征向量；特征向量旋转了怎么办；大 eigengap 是否证明分对了；σ 和 k 怎么选；到底复现了什么。

**24 · Baselines on the Figure 1 settings.** 同一批重建数据上 NJW、Meila–Shi、Kannan 式、连通分量、K-means 的 ARI。核方法共用按 distortion 选出的 σ。Kannan 式是我们按论文一句话做的简化，不是他们的原代码。

**25 · σ sensitivity on all three datasets.** 三套数据的 ARI–σ 曲线。Blobs 在 0.25σ₀ 会因为某一行长度接近 0 而算不下去。月牙在 2σ₀ 以前是 1.00，之后逐步下降。

**26 · References.** 论文出处，以及图和代码哪些是我们自己的。
