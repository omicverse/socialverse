# %% [markdown]
# # 空间数据的自相关与空间回归
#
# 社会科学的数据几乎从不「独立同分布」:失业率、房价、投票倾向、疫情传播——**邻近的地方彼此相似**。一个县失业率高,它周围的县往往也高。忽略这种空间依赖,普通 OLS 会低估标准误,把本属于「扩散/溢出」的现象误当成「个体效应」,推断因此不可靠。空间分析就是把「地理上的邻近关系」正式写进模型的一套方法。
#
# 完整的空间分析通常分两步走,本教程也照此展开。**第一步是诊断**:变量到底有没有在空间上聚集?我们先用**全局 Moran's I**——一个标量加一个置换检验的伪 p 值——回答「有没有」;再用**局部 LISA** 把这个全局数拆到每个地点,回答「聚在哪、是热点还是冷点」。**第二步是建模**:如果聚集是真的,就用**空间滞后模型(SAR)** `y = ρ·Wy + Xβ + ε` 把邻居的影响写进方程,估出空间自回归系数 `ρ`,并把自变量的效应分解成直接效应、间接效应(空间溢出)和总效应——这是普通回归给不了的。
#
# 贯穿全程的关键对象是**空间权重矩阵 `W`**:它编码「谁是谁的邻居」,是空间分析区别于普通表格分析的唯一新东西。本教程用 `socialverse` 走完这条链,它是一套面向社会科学的分析库,把每种方法登记进一张函数注册表,运行时校验前置是否就绪,并自动积累一份可复现的证据链。方法学与实现上,这里的全局/局部自相关对标 Python 的 **PySAL `esda`** 和 R 的 **`spdep`**,空间滞后回归对标 PySAL 的 **`spreg.ML_Lag`** 与 R 的 **`spatialreg::lagsarlm`**。

# %%
import matplotlib
matplotlib.use("Agg")  # 无显示环境:图直接写文件

import numpy as np
import pandas as pd

import socialverse as sv
from socialverse import datasets as ds

# %% [markdown]
# ## 载入空间数据
#
# 我们用一个内置的合成空间数据集:一个 8×8 的**车式邻接(rook)网格**,共 64 个格点(想象成 64 个县排成方阵,上下左右相邻的算邻居)。数据由一个真实的 SAR 过程生成,**真实的空间自回归系数 ρ = 0.5、自变量系数 β_x = 1.0**。有了已知的真值,这本教程就有了一个可证伪的靶子:我们估出来的 `ρ` 应当落在 0.5 附近,方法能不能复原它是一道硬检验。
#
# `ds.load_spatial` 返回一个 `(df, W)` **元组**:`df` 里每行一个格点,列有 `id, row, col`(位置)、`y`(结果变量)、`x`(自变量);`W` 是 64×64 的邻接矩阵,并且**已行标准化**——每行和为 1,于是 `Wy` 就是「邻居的平均 y」。角点有 2 个邻居、边点 3 个、内点 4 个。

# %%
df, W = ds.load_spatial(rho=0.5)   # SAR grid,真实 rho = 0.5

print(f"格点数 n = {len(df)}   ·   权重矩阵 W 形状 = {W.shape}")
df.head(6)

# %% [markdown]
# 检查 `W` 的两个基本性质:每行和恒为 1(行标准化的定义),以及邻居数随位置变化——角点少、内点多。这两点确认了 `W` 是一个合规的行标准化车式邻接矩阵。

# %%
row_sums = W.sum(axis=1)                    # 每行和
neighbours = (W > 0).sum(axis=1)            # 每个格点的邻居数
print(f"W 每行和(应恒为 1):min={row_sums.min():.3f}, max={row_sums.max():.3f}")
print("邻居数分布(角=2, 边=3, 内=4):",
      dict(zip(*np.unique(neighbours, return_counts=True))))

# %% [markdown]
# ## 建立研究态
#
# 分析的起点是把数据登记进一个研究态 `StudyState`。空间函数认得 `(df, W)` 这种打包形式:我们把整个元组写进 `sources.datasets` 一次,后续每一步——诊断、建模、出图——都从这里读数据和权重矩阵,**不必再反复传 `W`**。

# %%
st = sv.StudyState()
st.write("sources", "datasets", (df, W))   # (df, W) 元组即整条链的单一数据源
print(repr(st))

# %% [markdown]
# ## 全局 Moran's I:变量在空间上聚集吗
#
# 建模之前先诊断。**全局 Moran's I** 是一个标量,度量相邻格点的取值是否相关:正值表示高挨高、低挨低(正空间自相关),0 附近表示空间随机,负值表示棋盘式交错。它的显著性由**置换检验**给出——把 `y` 的标签在格点间随机洗牌 999 次,得到一个「若无空间结构会看到什么」的参考分布,再看真实的 I 有多极端。这正是 `esda.Moran(permutations=999)` 的做法。
#
# `spatial_autocorr` 一次调用同时算全局 Moran 和后面要用的局部 LISA。`value="y"` 指定对哪个变量做诊断,`seed=0` 固定置换的随机种子以便复现。

# %%
sv.tl.spatial_autocorr(st, value="y", permutations=999, seed=0)

moran = st.diagnostics["moran"]
print(f"Moran's I   : {moran['I']:.4f}   (n = {moran['n']})")
print(f"E[I]        : {moran['expected_I']:.4f}   (空间随机时的期望)")
print(f"z 分数      : {moran['z_score']:.3f}")
print(f"置换 p 值   : {moran['p_perm']:.4f}   ({moran['permutations']} 次置换,双侧)")
print(f"显著象限计数: {moran['cluster_counts']}   (p<0.05)")

# %% [markdown]
# `I ≈ 0.39`,远大于空间随机时的期望 `E[I] ≈ −0.016`,置换 p 值 = 0.001——**强而显著的正空间自相关**:高值格点扎堆、低值格点扎堆。这与生成数据时放进去的 ρ = 0.5 正空间依赖完全吻合。显著象限里 HH(热点)和 LL(冷点)占主导,说明这是「聚集」而非「离群」。

# %% [markdown]
# ## 局部 LISA:聚集发生在哪里
#
# 全局 I 只说了「有没有」,**LISA**(Local Indicators of Spatial Association)进一步说「在哪里」。它把全局 Moran 拆到每个格点:每点一个局部 `I_i`、一个 Moran 散点象限(HH=热点、LL=冷点、HL/LH=空间离群),以及一个条件置换 p 值。这些结果已经随上一步一起算好,存在 `st.models["lisa"]` 里,直接取用即可。
#
# 下面把每个格点的局部结果整理成一张表,并挑出 `|I_i|` 最强的 8 个格点看看——它们是空间结构最突出的地方。

# %%
lisa = st.models["lisa"]
print(f"显著局部簇的格点数(p_sim<0.05):{lisa['n_significant']} / {len(df)}")
print(f"各象限显著计数:{lisa['cluster_counts']}   (HH=热点, LL=冷点, HL/LH=离群)")

lisa_df = pd.DataFrame({
    "id": df["id"], "row": df["row"], "col": df["col"], "y": df["y"],
    "Ii": np.round(lisa["Ii"], 3),
    "quadrant": lisa["quadrant"],
    "p_sim": np.round(lisa["p_sim"], 3),
})
# 按局部 I_i 的绝对值排序,取最突出的 8 个格点
lisa_df.reindex(lisa_df["Ii"].abs().sort_values(ascending=False).index).head(8)

# %% [markdown]
# 64 个格点里有 5 个达到 `p_sim < 0.05` 的局部显著,集中在 HH 与 LL 象限——热点抱团、冷点抱团。最强的几个点(如 `id=47` 的 `I_i ≈ 2.82`)都落在 HH/LL,和全局诊断的方向一致:这是一片以正关联为主的空间场。

# %% [markdown]
# ## Moran 散点图:把诊断画出来
#
# Moran 散点图是全局 I 的可视化:横轴是标准化后的值 `z`,纵轴是它的空间滞后 `Wz`(邻居的平均),**拟合直线的斜率恰好等于 Moran's I**。四个象限一目了然——右上 HH(热点)、左下 LL(冷点)、右下 HL 与左上 LH(空间离群)。它和上面那个标量 I 是同一件事的两种呈现:一个给显著性数字,一个给空间直觉。
#
# `sv.pl.moran_scatter` 从研究态里读回诊断结果和 `(df, W)`,重建 `z` 与 `Wz` 并出图。图存成同目录下的 PNG,随后用 markdown 引用。

# %%
sv.pl.moran_scatter(st, variable="y", out="fig_moran.png")
print("图已保存:fig_moran.png")

# %% [markdown]
# ![Moran 散点图 · 拟合斜率 = Moran's I](fig_moran.png)
#
# 点云自左下向右上倾斜,拟合斜率 ≈ 报告的 `I ≈ 0.39`——**正斜率就是正空间自相关**。绝大多数点落在 HH/LL(正关联)象限,只有零星点在 HL/LH(空间离群),与前面的象限计数完全对应。

# %% [markdown]
# ## 声明结果变量
#
# 诊断确认了强空间依赖,接下来要建模。空间回归的因变量是研究设计的一部分,不该由函数替你猜——所以在跑 SAR 之前,先显式声明谁是结果变量。这一步把 `y` 写进研究态的 `variables.outcome` 槽,`spatial_regression` 会从这里读取。

# %%
st.write("variables", "outcome", "y")   # 显式声明因变量
print(repr(st))

# %% [markdown]
# ## 空间滞后回归(SAR):估计 ρ
#
# 现在把空间依赖写进模型:`y = ρ·Wy + Xβ + ε`。这里 `Wy` 是「邻居的平均结果」,系数 `ρ` 度量**一个格点的结果被邻居的结果拉动多少**。估计走**集中似然 ML**:先用 OLS 把 `β` 和 `σ²` 剖掉,只剩 `ρ` 的一维目标函数,再用 SciPy 的有界搜索最大化——目标里含 `log|I − ρW|` 雅可比项(用 `W` 的特征值计算),这正是 `spreg.ML_Lag` 的算法。
#
# 我们仍然不必传 `W`——它从 `sources.datasets` 的 `(df, W)` 元组里自动取。`outcome="y"` 与刚声明的因变量一致,`predictors=["x"]` 指定自变量。

# %%
sv.tl.spatial_regression(st, outcome="y", predictors=["x"])

sar = st.models["sar"]
print("=== 空间滞后模型 (SAR):y = rho * Wy + X beta + eps ===")
print(f"  样本量 n   : {sar['n']}")
print(f"  rho        : {sar['rho']:.4f}     ← 真实 rho = 0.5")
print(f"  beta       : const={sar['beta']['const']:.4f}, "
      f"x={sar['beta']['x']:.4f}   ← 真实 beta_x = 1.0")
print(f"  sigma^2    : {sar['sigma2']:.4f}")
print(f"  对数似然   : {sar['loglik']:.3f}")

# %% [markdown]
# **方法复原了已知参数。** 估计 `ρ ≈ 0.53`(真实 0.5)、`β_x ≈ 0.90`(真实 1.0)——在一个只有 64 个格点的小样本上,集中似然 ML 把生成数据的两个参数都稳稳打回。这正是「参数已知」的合成数据集的价值:方法的正确性不靠信任,当场可证。

# %% [markdown]
# ## 直接 / 间接 / 总效应
#
# 在 SAR 里,`β_x` **不是** `x` 对 `y` 的完整边际效应。原因是一个格点的 `x` 变化会改变它自己的 `y`,进而通过 `Wy` 溢出到邻居,再反馈回来。真正可解释的量来自约简形式 `(I − ρW)⁻¹`(LeSage & Pace 的分解):**直接效应**是自己 `x` 变对自己 `y` 的影响(含空间反馈),**间接效应**是溢出到所有邻居的影响之和,**总效应**是两者相加。

# %%
impacts = sar["impacts"]["x"]
print("=== x 的效应分解 (LeSage-Pace, 基于 (I - rho W)^-1) ===")
print(f"  直接效应 (direct)   : {impacts['direct']:.4f}   (含自我空间反馈)")
print(f"  间接效应 (indirect) : {impacts['indirect']:.4f}   (溢出到邻居)")
print(f"  总效应   (total)    : {impacts['total']:.4f}   (≈ beta_x / (1 - rho))")

# %% [markdown]
# 间接效应(空间溢出)明显非零且为正:`ρ>0` 意味着一个地方 `x` 的提升会外溢到邻居的 `y`。总效应 ≈ `β_x/(1−ρ) ≈ 0.9/0.47 ≈ 1.9`,几乎是直接效应的两倍。**如果只用普通 OLS,你只会看到 β,完全丢掉这块溢出**——这正是空间计量存在的理由。

# %% [markdown]
# ## 可复现的证据链
#
# 最后看一眼 `socialverse` 与普通空间分析脚本的关键差别。整条链从 `(df, W)` 元组出发,经过 Moran 诊断、LISA、Moran 散点图、SAR 估计,所有结果都落在**同一个 `StudyState` 的具名槽**里;每一次成功的函数调用,注册表都会自动往证据链里焊上一条「用了哪个函数、消费了什么、产出了什么」的记录。对标 PySAL / spdep,你需要在 `esda` + `spreg` + `libpysal` + `matplotlib` 之间手动拼胶水并自己记账;这里,记账是分析的副产品。

# %%
print(st.summary())

# %% [markdown]
# ## 小结
#
# 我们走完了一条标准的空间分析链:载入带 `W` 的空间数据 → 全局 Moran 诊断 → 局部 LISA 定位 → Moran 散点图 → 声明因变量 → SAR 估计 ρ → 效应分解。它对标 Python 的 **PySAL**(`esda` 做自相关、`spreg.ML_Lag` 做空间滞后回归)与 R 的 **`spdep` / `spatialreg`**。
#
# 与纯估计工具相比,这里多了两样东西:所有步骤共享**同一个研究态**(诊断产出的槽直接被出图和建模读取,不必手动搬数据),以及一份贯穿始终、由注册表自动焊上的**证据链**。下一本教程 [15_qca_demography](15_qca_demography.ipynb) 转向 fsQCA 组态分析与人口学的生命表 / 分解方法。
