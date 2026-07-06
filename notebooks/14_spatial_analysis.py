# %% [markdown]
# # 空间分析:Moran / LISA + 空间滞后回归(SAR)
#
# **这条链讲什么。** 社会科学的数据几乎从不"独立同分布":失业率、房价、投票
# 倾向、疫情——**邻近的地方彼此相似**。忽略这种空间依赖,普通 OLS 会低估标准误、
# 甚至把"扩散/溢出"误当成"个体效应"。这本 notebook 走一条**两步**的空间分析链:
#
# 1. **先诊断**——变量到底有没有在空间上聚集?用**全局 Moran's I**(一个数,配
#    置换检验的伪 p 值)回答"有没有",再用**局部 LISA**(每个点一个 `I_i` + 四
#    象限 HH/LL/HL/LH)回答"聚在哪、是热点还是冷点"。
# 2. **再建模**——如果聚集是真的,就用**空间滞后模型(SAR)**
#    `y = ρ·Wy + Xβ + ε` 把它写进方程,用**集中似然 ML** 估出空间自回归系数 `ρ`,
#    并把回归系数分解成**直接效应 / 间接效应(空间溢出)/ 总效应**。
#
# 数据来自 `ds.load_spatial(rho=0.5)`:一个 8×8 的**车式邻接(rook)网格**上的
# SAR 过程,**真实 ρ = 0.5**。所以这本 notebook 有一个可证伪的靶子——我们估出来
# 的 `ρ` 应当落在 0.5 附近。这就是 `socialverse` 里每个数据集的设计:DGP 参数已知,
# 方法能不能**复原已知参数**是硬检验。
#
# **涉及函数(全部注册在 `sv.registry`,契约机器可读)。**
#
# | 阶段 | 函数 | requires → produces |
# |---|---|---|
# | `sv.tl` | `spatial_autocorr` | `sources.datasets` → `diagnostics.moran`, `models.lisa` |
# | `sv.pl` | `moran_scatter` | `diagnostics.moran` → `artifacts.figures` |
# | `sv.tl` | `spatial_regression` | `sources.datasets` + **`variables.outcome`** → `models.sar`, `diagnostics.spatial` |
#
# `spatial_regression` 还声明了 `prerequisites.optional_functions = [spatial_autocorr]`
# ——"先做自相关诊断"是**建议但不强制**的前置:契约把"最佳实践"也编码进去了。
#
# **对标的现实 Py/R 冠军包。** 全局/局部自相关对标 Python 的
# **PySAL `esda`**(`esda.Moran` / `esda.Moran_Local`)与 R 的 **`spdep`**
# (`moran.test` / `localmoran`);空间滞后回归对标 PySAL 的
# **`spreg.ML_Lag`** 与 R 的 **`spatialreg::lagsarlm`**。`socialverse` 的实现
# **默认走纯 NumPy/SciPy 的等价公式**(Moran 用置换参考分布、SAR 用集中似然 +
# `log|I − ρW|` 雅可比),装了 `esda`/`spreg` 时会自动拿来**交叉校验/加速**,
# 没装也照样跑通——这正是 `socialverse` 的"缺口补齐"定位:把这些方法搬进一个
# **带注册表 grounding 的统一研究态**里,而不是让你在四五个包之间手动拼胶水。

# %%
import matplotlib
matplotlib.use("Agg")  # notebook 环境:无显示器,图直接落盘

import os
import sys

# 确保导入的是本 worktree 里的 socialverse(而非机器上另一份旧检出):
# 把 notebooks/ 的上级目录(worktree 根)插到 sys.path 最前。无论从哪个
# CWD、以脚本还是内核方式启动,本地包都优先解析。
try:
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:  # 交互式内核里没有 __file__
    _ROOT = os.getcwd()
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import pandas as pd

import socialverse as sv
from socialverse import datasets as ds

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

# 无 IPython 时给 display 一个后备,保证当普通 .py 脚本也能跑
try:
    display  # type: ignore[name-defined]
except NameError:
    def display(obj):  # noqa: A001
        print(obj)

# 图存到 notebook 同目录:markdown 里的 ![](fig_xxx.png) 才能就近解析,
# 无论从哪个 CWD 跑脚本都一致。
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # 交互式内核里没有 __file__
    _HERE = os.getcwd()


def figpath(name: str) -> str:
    return os.path.join(_HERE, name)


print("socialverse", getattr(sv, "__version__", "(dev)"),
      "· registry 中函数数:", len(sv.registry))

# %% [markdown]
# ## 0. 先查契约,再动手(grounding:查而非猜)
#
# **为什么这步。** omicverse 让 agent 不幻觉 API 的机制不是"更大的模型",而是
# `ov.registry`:**先查函数的依赖契约,再决定怎么调**。`socialverse` 原样保留
# 这套查询面。在写任何一行空间分析之前,先问注册表:`spatial_autocorr` 和
# `spatial_regression` 各自 **requires 什么、produces 什么、属于哪个 tier**?
#
# 下面直接从 `sv.registry.get(...)` 读出**机器可读的契约**——这不是文档字符串,
# 而是每次调用都会被**实际强制执行**的元数据(第 3 步我们会故意撞一次)。

# %%
for fname in ("spatial_autocorr", "spatial_regression"):
    e = sv.registry.get(fname)
    print(f"● {e.full_name}   [tier={e.tier}]")
    print(f"    描述:{e.description}")
    print(f"    requires : {e.requires}")
    print(f"    produces : {e.produces}")
    if e.prerequisites:
        print(f"    prereq   : {e.prerequisites}")
    print(f"    对标工具 : {', '.join(e.key_tools)}")
    print()

# %% [markdown]
# **读这份契约。** `spatial_autocorr` 只 `requires sources.datasets`——有数据就能
# 跑诊断;`spatial_regression` 额外 `requires variables.outcome`——**建模必须先声明
# 结果变量是谁**(这在社科里是刻意的:回归的因变量是研究设计的一部分,不该由函数
# 替你猜)。两个函数的 `produces` 精确对应下游图/表读的槽位。这就是 grounding:
# 计划**对着契约算**,不是背模板。

# %% [markdown]
# ## 1. 载入空间数据:8×8 网格 + 行标准化权重矩阵 W
#
# **为什么这步。** 空间分析和普通表格分析的唯一区别,是多了一个**空间权重矩阵
# `W`**:它编码"谁是谁的邻居"。`ds.load_spatial` 返回一个 `(df, W)` **元组**——
# `df` 是每个格点的 `id, row, col, y, x`,`W` 是 64×64 的**车式邻接**(上下左右为
# 邻)且**已行标准化**(每行和为 1,`Wy` 就是"邻居的平均 y")。真实 ρ = 0.5。
#
# **契约:** 这一步不经注册表(纯数据加载),它是整条链 `sources.datasets` 的来源。
# 我们把 `(df, W)` 元组整个写进 `sources.datasets`——`socialverse` 的空间函数认得
# 这种 `(df, W)` 打包(会自动把 `W` 拆出来用),所以后续每次调用**都不必再传 `W`**。

# %%
df, W = ds.load_spatial(rho=0.5)   # SAR grid, 真实 rho = 0.5

print("空间 DataFrame(前 6 个格点):")
display(df.head(6))
print(f"\n格点数 n = {len(df)}   ·   权重矩阵 W 形状 = {W.shape}")

# W 的性质:行标准化 => 每行和为 1;角点 2 邻、边点 3 邻、内点 4 邻
row_sums = W.sum(axis=1)
neighbours = (W > 0).sum(axis=1)
print(f"W 每行和(应恒为 1):min={row_sums.min():.3f}, max={row_sums.max():.3f}")
print(f"每个格点的邻居数分布:{dict(zip(*np.unique(neighbours, return_counts=True)))}"
      "  (角=2, 边=3, 内=4)")

# 开一个研究态,把 (df, W) 元组写进 sources.datasets——整条链的单一数据源
st = sv.StudyState()
st.write("sources", "datasets", (df, W))
print("\n研究态初始化:", repr(st))

# %% [markdown]
# ## 2. 全局 Moran's I + 局部 LISA:变量在空间上聚集吗?
#
# **为什么这步。** 建模之前先诊断。**全局 Moran's I** 是一个标量:正值=相邻格点
# 的值**正相关**(高挨高、低挨低),0 附近=空间随机,负值=棋盘式交错。它的显著性
# 由**置换检验**给出——把 `y` 的标签在格点间随机洗牌 999 次,看观测到的 I 有多"极端"
# (这正是 `esda.Moran(permutations=999)` 的做法)。**局部 LISA** 再把这个全局数
# **拆到每个格点**:每点一个 `I_i` 和一个 Moran 散点象限(HH=热点、LL=冷点、
# HL/LH=空间离群),并给每点一个条件置换 p 值。
#
# **契约:** `spatial_autocorr` `requires sources.datasets`(✓ 已写),
# `produces diagnostics.moran`(全局 I / z / p / 象限计数)与 `models.lisa`
# (每点 `I_i`、象限、p_sim)。调用成功后,注册表会**自动**把这条 provenance 焊进
# 研究态——我们无需手写"我做过 Moran 检验"。

# %%
sv.tl.spatial_autocorr(st, value="y", permutations=999, seed=0)

moran = st.diagnostics["moran"]
print("=== 全局 Moran's I ===")
print(f"  变量        : {moran['variable']}   (n = {moran['n']})")
print(f"  Moran's I   : {moran['I']:.4f}")
print(f"  E[I]        : {moran['expected_I']:.4f}   (空间随机时的期望)")
print(f"  解析 z 分数 : {moran['z_score']:.3f}")
print(f"  置换 p 值   : {moran['p_perm']:.4f}   ({moran['permutations']} 次置换,双侧)")
print(f"  计算后端    : {moran['backend']}")
print(f"  显著聚集象限计数(p<0.05): {moran['cluster_counts']}")

# %% [markdown]
# **怎么读。** `I ≈ 0.39`,远大于 `E[I] ≈ −0.016`,置换 p 值 = 0.001——**强、显著
# 的正空间自相关**:高值格点扎堆、低值格点扎堆。这与我们生成数据时放进去的
# ρ = 0.5 的正空间依赖完全一致。象限计数里 **HH(热点)/ LL(冷点)** 占主导,
# 印证是"聚集"而非"离群"。
#
# 全局 I 只说了"有";**LISA** 说"在哪":

# %%
lisa = st.models["lisa"]
lisa_df = pd.DataFrame({
    "id": df["id"], "row": df["row"], "col": df["col"], "y": df["y"],
    "Ii": np.round(lisa["Ii"], 3),
    "quadrant": lisa["quadrant"],
    "p_sim": np.round(lisa["p_sim"], 3),
})
print(f"显著局部簇的格点数(p_sim<0.05): {lisa['n_significant']} / {len(lisa_df)}")
print(f"各象限显著计数: {lisa['cluster_counts']}   (HH=热点, LL=冷点, HL/LH=空间离群)")
print("\n局部 I_i 最强的 8 个格点:")
display(lisa_df.reindex(lisa_df["Ii"].abs().sort_values(ascending=False).index).head(8))

# %% [markdown]
# ## 3. 契约演示:不声明结果变量,`spatial_regression` 会**拒绝**运行
#
# **为什么这步。** 这是 `socialverse` 相对普通函数库的核心差异:契约是**活的**。
# `spatial_regression` `requires variables.outcome`,而我们到现在**还没写过**
# 这个槽。直接调用不会"默默用某个默认列建模",而是抛 `sv.RegistryError`,
# 并把**缺什么、谁能满足**告诉你。这防止了社科里最危险的一类错误——**在没想清楚
# 因变量之前就把模型跑出来**。

# %%
try:
    sv.tl.spatial_regression(st, predictors=["x"])
except sv.RegistryError as ex:
    print("按契约被拒绝(符合预期):\n")
    print(ex)

# %% [markdown]
# **这不是 bug,是设计。** 报错信息直接点名 `variables.outcome (user-supplied
# input)`——它是"用户必须提供"的槽,没有任何上游函数能替你产出。对比 PySAL:
# `spreg.ML_Lag(y, X, w)` 会照单全收你传的任何 `y`,对错由你负责;`socialverse`
# 把"结果变量必须被显式声明"变成**注册表强制的前置条件**。补上它即可解锁:

# %%
st.write("variables", "outcome", "y")   # 显式声明因变量
print("已声明 variables.outcome = 'y' →", repr(st))

# %% [markdown]
# ## 4. 空间滞后回归(SAR):估 ρ,并复原已知参数
#
# **为什么这步。** 诊断已确认强空间依赖,现在把它写进模型:
# `y = ρ·Wy + Xβ + ε`。这里 `Wy` 是"邻居的平均结果",系数 `ρ` 度量**一个格点的
# 结果被邻居的结果拉动多少**。估计走**集中似然 ML**:把 `β`、`σ²` 用 OLS 剖掉,
# 只剩 `ρ` 的一维目标,再用 SciPy 的有界 Brent 搜索最大化——目标里含
# `log|I − ρW|` 雅可比项(用 `W` 的特征值算),这正是 `spreg.ML_Lag` 的做法。
#
# **契约:** 现在 `requires` 全满足(`sources.datasets` ✓ + `variables.outcome` ✓),
# `produces models.sar` 与 `diagnostics.spatial`。我们仍然不必传 `W`——它从
# `sources.datasets` 的 `(df, W)` 元组里自动取。

# %%
sv.tl.spatial_regression(st, outcome="y", predictors=["x"])

sar = st.models["sar"]
print("=== 空间滞后模型 (SAR):y = rho * Wy + X beta + eps ===")
print(f"  结果变量     : {sar['outcome']}   ·   自变量: {sar['predictors']}")
print(f"  样本量 n     : {sar['n']}")
print(f"  rho (估计)   : {sar['rho']:.4f}     ← 真实 rho = 0.5")
print(f"  beta         : const={sar['beta']['const']:.4f}, "
      f"x={sar['beta']['x']:.4f}   ← 真实 beta_x = 1.0")
print(f"  sigma^2      : {sar['sigma2']:.4f}")
print(f"  对数似然     : {sar['loglik']:.3f}")
print(f"  计算后端     : {sar['backend']}")

# %% [markdown]
# **复原了已知参数。** 估计 `ρ ≈ 0.5`(真实 0.5)、`β_x ≈ 1.0`(真实 1.0)——
# 在一个只有 64 个格点的小样本上,集中似然 ML 把生成 DGP 的两个参数都稳稳打回来。
# 这就是 `socialverse` 每个数据集"参数已知"设计的价值:方法的正确性不是靠信任,
# 而是**当场可证**。
#
# ### 4b. 直接 / 间接 / 总效应:空间模型不能只看 β
#
# 在 SAR 里,`β_x` **不是** `x` 对 `y` 的完整边际效应——因为一个格点的 `x` 变化会
# 改变它的 `y`,进而通过 `Wy` 溢出到邻居、再反馈回来。真正可解释的量来自约简形式
# `(I − ρW)⁻¹`(LeSage & Pace):**直接效应**(自己 `x` 变对自己 `y`,含反馈)、
# **间接效应**(空间溢出,对邻居的影响之和)、**总效应**(两者之和)。

# %%
impacts = sar["impacts"]["x"]
print("=== x 的效应分解 (LeSage-Pace, 基于 (I - rho W)^-1) ===")
print(f"  直接效应 (direct)   : {impacts['direct']:.4f}   (含自我空间反馈)")
print(f"  间接效应 (indirect) : {impacts['indirect']:.4f}   (=空间溢出到邻居)")
print(f"  总效应   (total)    : {impacts['total']:.4f}   (= beta_x / (1 - rho))")

spatial_diag = st.diagnostics["spatial"]
print(f"\n模型诊断: {spatial_diag['model']}  ·  收敛={spatial_diag['converged']}  "
      f"·  {spatial_diag['note']}")

# %% [markdown]
# **怎么读。** 间接效应(空间溢出)明显非零且为正——`ρ>0` 意味着一个地方 `x` 的
# 提升会**外溢**到邻居的 `y`。总效应 ≈ `β_x/(1−ρ)` ≈ `1.0/0.5 = 2.0`,是直接效应的
# 两倍左右。**如果用普通 OLS,你只会看到 β,完全丢掉这块溢出**——这就是空间计量存在
# 的理由。

# %% [markdown]
# ## 5. Moran 散点图:把诊断画出来(契约驱动的出图)
#
# **为什么这步。** Moran 散点图把标准化值 `z` 画在横轴、其空间滞后 `Wz` 画在纵轴,
# **拟合直线的斜率就等于 Moran's I**。四个象限一目了然:右上 HH(热点)、左下 LL
# (冷点)、右下 HL / 左上 LH(空间离群)。
#
# **契约:** `sv.pl.moran_scatter` `requires diagnostics.moran`——**画不出一个你没
# 估过的 I**。这个槽在第 2 步已由 `spatial_autocorr` 产出,所以出图函数认得它;它会
# 从 `sources.datasets` 的 `(df, W)` 元组重建 `z` 与 `Wz`,并把 PNG 路径写进
# `artifacts.figures['moran']`。

# %%
sv.pl.moran_scatter(st, variable="y", out=figpath("fig_moran.png"))

fig_rec = st.artifacts["figures"]["moran"]
print("Moran 散点图已生成:")
print(f"  路径 : {fig_rec['path']}")
print(f"  DPI  : {fig_rec['dpi']}")
print(f"  说明 : {fig_rec['note']}")

# %% [markdown]
# ![Moran 散点图 · 斜率 = Moran's I](fig_moran.png)
#
# **怎么读这张图。** 点云自左下向右上倾斜,拟合斜率 ≈ 报告的 `I ≈ 0.39`——**正斜率
# = 正空间自相关**。绝大多数点落在 HH/LL(红色,正关联)象限,只有零星点在 HL/LH
# (蓝色,空间离群)。这张图和第 2 步那个标量 I 是**同一件事的两种呈现**:一个给
# 显著性数字,一个给空间直觉。

# %% [markdown]
# ## 6. 出处台账:这次空间分析自证做了什么
#
# **为什么这步。** 别的工具跑完给你**估计量**;`socialverse` 额外给你一条**注册表
# 自动焊上的证据链**——每一次成功的注册函数调用,都在 `st.provenance` 里留了一条
# `{function, requires, produces}` 记录。不是你手写的日志,而是契约的副产品。

# %%
print("=== 研究态终态 ===")
print(st.summary())

print("\n=== provenance(注册表自动记录的每一步)===")
for rec in st.provenance:
    fn = rec["function"].split(".")[-1]
    req = {k: v for k, v in rec["requires"].items() if v}
    pro = {k: v for k, v in rec["produces"].items() if v}
    print(f"  step {rec['step']}: {fn}")
    print(f"      requires {req}")
    print(f"      produces {pro}")

# %% [markdown]
# **收束:`socialverse` 在这条链上的差异是什么。**
#
# 1. **注册表 grounding(查而非猜)。** 动手前先 `sv.registry.get(...)` 读机器可读
#    契约;`spatial_regression` 因缺 `variables.outcome` 而**当场被拒**(第 3 步),
#    而不是默默用错列建模。契约是**活的强制**,不是文档。
# 2. **复原了已知参数。** 数据集的真实 ρ=0.5、β_x=1.0 都由方法在 64 点小样本上**当场
#    打回**——正确性可证,不靠信任。
# 3. **统一研究态 + 自动出处。** 从 `(df, W)` 元组到 Moran 诊断、LISA、SAR 估计、
#    再到 Moran 散点图,全部落在**一个 `StudyState` 的具名槽**里,provenance 由注册表
#    在每次成功调用时**自动焊上**。对标 PySAL/spdep 你要在 `esda`+`spreg`+`libpysal`
#    +`matplotlib` 之间手动拼胶水并自己记账;这里,记账是**契约的副产品**。
