# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 准实验：断点回归 RDD + 合成控制
#
# > **这条链讲什么。** 当我们无法随机分配处理、又想识别一个**因果效应**时，
# > 社会科学有两件趁手的兵器：
# >
# > 1. **断点回归（RDD, Regression Discontinuity）** —— 当处理由一条**规则**决定
# >    （分数 ≥ 60 才录取、收入 ≤ 阈值才补贴），在断点两侧无限接近处，个体几乎
# >    "随机"地落在处理组 / 对照组，于是断点处结果变量的**跳跃**就是局部因果效应
# >    （LATE）。
# > 2. **合成控制（Synthetic Control, SCM）** —— 只有**一个**处理单元（一座城市加了
# >    税、一个国家改了法），没有天然对照。我们用一组"捐赠者"单元的**加权组合**造出
# >    一个"如果没被处理会怎样"的合成反事实，处理后 treated − synthetic 的缺口就是效应。
# >
# > **本 notebook 涉及的 socialverse 函数**
# >
# > | 环节 | 函数 | 契约（requires → produces） |
# > |---|---|---|
# > | 锐性 RDD 估计 | `sv.tl.rdd` | `sources.datasets` + `variables.outcome` + `estimand.target` → `models.rdd` + `diagnostics.bandwidth` |
# > | RDD 断点图 | `sv.pl.rdd_plot` | `models.rdd` → `artifacts.figures` |
# > | 合成控制估计 | `sv.tl.synthetic_control` | `sources.datasets` + `design.treatment/time` + `variables.outcome` + `estimand.target` → `models.synth` + `diagnostics.pre_fit` |
# > | 反事实路径图 | `sv.pl.synth_path` | `models.synth` → `artifacts.figures` |
# >
# > **对标现实冠军包**
# >
# > * `sv.tl.rdd` ≈ R 的 **rdrobust** / Python `rdrobust`
# >   （Calonico–Cattaneo–Titiunik）：三角核局部线性 WLS + 数据驱动带宽。
# > * `sv.tl.synthetic_control` ≈ R 的 **Synth** / **gsynth** / **augsynth**、Python
# >   `pysyncon` / `SparseSC`：Abadie–Diamond–Hainmueller 的非负、和为一的捐赠者权重。
# >
# > **socialverse 的差异**：每个函数都带**机器可读的依赖契约**（`requires`/`produces`），
# > 状态放在 12 槽 `StudyState`（社科版 AnnData），且我们不猜 API——**查注册表**。
# > 两个玩具数据集都由**已知参数**的真实 DGP 生成（RDD 真跳 `τ=3.0`、SCM 真 `att=-0.8`），
# > 于是本 notebook 结束时你能亲眼看到估计量**复原了已知参数**。

# %%
import matplotlib
matplotlib.use("Agg")  # 无显示环境下渲染，图直接存盘

import os
import sys

# 让本 notebook 无论从哪里启动都加载 worktree 里的 socialverse,
# 并把工作目录切到 notebook 自身目录(图存相对名 fig_xxx.png 就落在这里)。
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:  # 交互式内核里没有 __file__
    _here = os.getcwd()
_root = os.path.dirname(_here)  # worktree 根:notebooks/ 的上一级
if os.path.isdir(os.path.join(_root, "socialverse")) and _root not in sys.path:
    sys.path.insert(0, _root)
if os.path.isdir(_here):
    os.chdir(_here)

import numpy as np
import pandas as pd

import socialverse as sv
from socialverse import datasets as ds

print("socialverse", sv.__version__, "|", sv.__file__)

# %% [markdown]
# ## 0. 先查注册表，而不是猜 API
#
# socialverse 的立身之本是那张**依赖注册表**：与其凭记忆猜函数名和参数，不如
# **问注册表**。先看看它对 "断点回归" 和 "合成控制" 知道些什么——包括每个函数的
# `requires` / `produces` 契约，以及"要满足这个前置槽，可以由谁来产出"。

# %%
# 中文别名也能命中(注册表按中/英/缩写/工具名做模糊+子串检索);
# 我们要哪个函数就按名字挑出来,不依赖返回顺序。
for q, want in [("断点回归", "rdd"), ("合成控制", "synthetic_control")]:
    hits = sv.registry.find(q, limit=10)
    e = next(h for h in hits if h["name"] == want)
    print(f"[{q}] -> {e['full_name']}  (tier={e['tier']}, 对标={e['key_tools']})")
    print("     ", e["description"])

# %% [markdown]
# 再把 `rdd` 的完整契约拉出来——这就是"要跑它，状态里得先有什么、它会写出什么"。
# 这份契约是 **live** 的：稍后如果 `StudyState` 缺了 `requires` 里的槽，调用会直接
# 抛 `RegistryError`（而不是给你一个错得离谱的结果）。

# %%
import json
prereq = sv.registry.get_prerequisites("rdd")
print(json.dumps(
    {k: prereq[k] for k in ("function", "requires", "produces", "auto_fix")},
    ensure_ascii=False, indent=2,
))

# %% [markdown]
# ---
# ## 1. 锐性断点回归（Sharp RDD）
#
# **数据。** `ds.load_rdd(tau=3.0)` 生成一个锐性 RDD 玩具集：running 变量在
# `cutoff=0` 处触发处理，真实的断点跳跃 `τ = 3.0`。列为
# `running`（running 变量）、`treat`（0/1 处理）、`y`（结果）、`x`（协变量）。
# 结果的真实结构里既有斜率也有曲率（`1.5·running − 0.8·running²`），所以一条粗暴的
# 全局直线会把曲率误当成跳跃——这正是为什么要在**断点附近**做局部线性拟合。

# %%
rdd_df = ds.load_rdd(tau=3.0)
print(rdd_df.head())
print("\n形状:", rdd_df.shape,
      "| cutoff=0 左侧样本:", int((rdd_df["running"] < 0).sum()),
      "右侧样本:", int((rdd_df["running"] >= 0).sum()))

# %% [markdown]
# ### 1.1 装配 StudyState（满足 `rdd` 的 requires）
#
# 契约要求三样东西：`sources.datasets`（数据）、`variables.outcome`（哪个是结果）、
# `estimand.target`（估计目标——这里是 **LATE**，局部平均处理效应）。我们逐一 `write`
# 进状态。这一步纯粹是"把契约要求的槽填上"，不做任何计算。

# %%
st = sv.StudyState()
st.write("estimand", "target", "LATE")        # 断点回归识别的是局部平均处理效应
st.write("variables", "outcome", "y")
st.write("sources", "datasets", rdd_df)

# 用契约自检:满足了吗?
req = sv.registry.get_prerequisites("rdd")["requires"]
print("满足 rdd.requires 吗?", st.satisfies(req))
print("状态快照:", st)

# %% [markdown]
# ### 1.2 契约先行：故意漏一个槽，看 `RegistryError`
#
# 为了让你**看见**契约是活的：如果我们在一个**空**状态上调 `rdd`，注册表会在计算
# 之前就拦下来，并告诉我们缺哪个 `(slot, key)`——这就是 "grounding：查而非猜" 的另一面。

# %%
try:
    sv.tl.rdd(sv.StudyState(), running="running", cutoff=0.0)
except sv.RegistryError as err:
    print("按预期抛出 RegistryError：")
    print(err)

# %% [markdown]
# ### 1.3 估计断点跳跃
#
# 现在在**装配好的** `st` 上调用。`sv.tl.rdd` 会：
# 选一个数据驱动带宽（IK-lite 经验法则），在断点两侧各用**三角核加权**的局部线性
# WLS 拟合，读取两侧在断点处的**边界截距**，**右 − 左**即为跳跃。
# 它原地改 `st` 并返回，同时写 `models.rdd`（跳跃、SE、t、两侧截距）和
# `diagnostics.bandwidth`（带宽 h、两侧有效样本量）。

# %%
sv.tl.rdd(st, running="running", cutoff=0.0)

model = st.models["rdd"]
bw = st.diagnostics["bandwidth"]
print(f"真实跳跃 τ      = 3.0  (DGP 已知)")
print(f"估计跳跃 τ̂      = {model['jump']:.4f}   (SE={model['se']:.4f}, t={model['t']:.2f})")
print(f"  左侧边界截距  = {model['left_intercept']:.4f}")
print(f"  右侧边界截距  = {model['right_intercept']:.4f}")
print(f"估计量           = {model['estimator']}  核={model['kernel']}")
print(f"带宽 h           = {bw['h']:.4f}  ({bw['selector']}), "
      f"左侧 n={bw['n_left']}, 右侧 n={bw['n_right']}")

# %% [markdown]
# **读数。** 估计跳跃 `τ̂ ≈ 3`，落在真实 `τ = 3.0` 的一两个标准误之内，t 值很大——
# 断点处结果确有一个显著、且量级正确的跳升。局部线性 + 三角核（权重在带宽边缘衰减到 0）
# 让我们把估计聚焦在断点邻域，避免全局曲率污染，这正是 rdrobust 的做法。

# %% [markdown]
# ### 1.4 断点图（rdplot 风格）
#
# `sv.pl.rdd_plot` 读 `models.rdd`，把 running 变量在两侧做**等计数分箱**画均值散点，
# 再叠上两侧的局部线性拟合线，让它们在断点处相遇——两线在 cutoff 的落差就是估计的跳跃。
# `out=` 直接存 PNG（同目录相对名）。

# %%
sv.pl.rdd_plot(st, out="fig_rdd.png")
print("已存:", st.artifacts["figures"]["rdd"])

# %% [markdown]
# ![RDD 断点图](fig_rdd.png)
#
# 蓝（左）与绿（右）分箱均值分别拟合出两条局部线性线，它们在 `running=0` 的红色虚线
# 处的**垂直落差**≈ 3，正是我们估计的因果跳跃。

# %% [markdown]
# ---
# ## 2. 合成控制（Synthetic Control）
#
# **数据。** `ds.load_did_panel(att=-0.8)` 是一个带**平行预趋势**的面板：
# 前半数单元在 `year=2015` 被处理，真实处理效应 `att = -0.8`。列包括
# `firm_id`、`year`、`treat`、`post`、`treat_post`、`y`、`x1` 等。
# 我们把**单个** treated 单元（`firm_id=0`）挑出来，用其余"捐赠者"单元的加权组合造反事实。

# %%
panel = ds.load_did_panel(att=-0.8)
print(panel.head())
print("\n单元数:", panel["firm_id"].nunique(),
      "| 年份:", sorted(panel["year"].unique()),
      "| 处理起始年: 2015")

# %% [markdown]
# ### 2.1 装配 StudyState（满足 `synthetic_control` 的 requires）
#
# 合成控制的契约更严一点：除 `sources.datasets`、`variables.outcome`、`estimand.target`
# 外，还要 `design.treatment` 和 `design.time`——因为它得知道"哪一列标记处理"和
# "哪一列是时间轴"。注意估计目标这里是 **ATT**（处理组的平均处理效应），不是 LATE。

# %%
st2 = sv.StudyState()
st2.write("design", "treatment", "treat")   # 用于把"曾被处理"的单元踢出捐赠池
st2.write("design", "time", "year")
st2.write("variables", "outcome", "y")
st2.write("estimand", "target", "ATT")
st2.write("sources", "datasets", panel)

req2 = sv.registry.get_prerequisites("synthetic_control")["requires"]
print("满足 synthetic_control.requires 吗?", st2.satisfies(req2))
print("状态快照:", st2)

# %% [markdown]
# ### 2.2 求解捐赠者权重并估 ATT
#
# `sv.tl.synthetic_control` 把长面板 pivot 成 `时间 × 单元` 的结果矩阵，在
# `treat_time=2015` 处切成前 / 后期。它用 SLSQP 求一组**非负、和为一**的捐赠者权重，
# 使 treated 单元与其合成对照在**前期**的 MSE 最小；反事实即加权捐赠者路径，
# 后期的平均 gap（treated − synthetic）就是 ATT。
# 它会自动把**任何曾被处理**的单元排除出捐赠池（避免"处理过的捐赠者"污染反事实）。

# %%
sv.tl.synthetic_control(st2, unit="firm_id", time="year",
                        treated_unit=0, treat_time=2015)

synth = st2.models["synth"]
pre_fit = st2.diagnostics["pre_fit"]
print(f"真实 att   = -0.8  (DGP 已知)")
print(f"估计 ATT   = {synth['att']:.4f}")
print(f"捐赠者数量  = {synth['n_donors']}  (已剔除曾处理单元)")
print(f"前期拟合    = pre-RMSE {pre_fit['pre_rmse']:.4f} "
      f"(n_pre={pre_fit['n_pre']}, n_post={pre_fit['n_post']})")
print(f"估计量      = {synth['estimator']}")

# 展示前几个非零捐赠者权重(非负且和为一)
top_w = sorted(synth["weights"].items(), key=lambda kv: -kv[1])[:5]
print("\n主要捐赠者权重(top 5):")
for donor, w in top_w:
    print(f"  firm_id={donor:>2}  w={w:.3f}")
print(f"  权重之和 = {sum(synth['weights'].values()):.3f}")

# %% [markdown]
# **读数。** 前期 RMSE 很小，说明合成对照在处理前**紧贴**真实 treated 单元——这是
# 合成控制可信的前提。后期平均 gap（估计 ATT）为负、方向正确，权重非负且和为一
# （"treated 单元 ≈ 这几个捐赠者的凸组合"）。
#
# 但注意这个**单案例**点估计（`firm_id=0`)并不精确等于 `-0.8`：合成控制针对
# **单个**处理单元，只有 5 个前期点、捐赠者本身带噪声，它没有传统意义上的抽样分布，
# 单案例估计天然是**有噪的**。下一步我们用一个诚实的检验说明：真实 DGP 效应是在
# **聚合**层面被复原的。

# %% [markdown]
# ### 2.3 单案例有噪，聚合复原真值
#
# 把合成控制**逐个**套到 20 个处理单元上，收集每个单元的 ATT。单看任一单元会偏离，
# 但处理组 ATT 的**均值 / 中位数**应回到真实的 `-0.8`——这正是把 SCM 当作
# "每个单元一个反事实" 的估计器时应有的表现。

# %%
treated_ids = sorted(panel.loc[panel["treat"] == 1, "firm_id"].unique())
per_unit_att = []
for u in treated_ids:
    s = sv.StudyState()
    s.write("design", "treatment", "treat"); s.write("design", "time", "year")
    s.write("variables", "outcome", "y"); s.write("estimand", "target", "ATT")
    s.write("sources", "datasets", panel)
    sv.tl.synthetic_control(s, unit="firm_id", time="year",
                            treated_unit=int(u), treat_time=2015)
    per_unit_att.append(s.models["synth"]["att"])

per_unit_att = np.array(per_unit_att, float)
print(f"真实 att            = -0.8  (DGP 已知)")
print(f"单案例 (firm_id=0)  = {synth['att']:.3f}   <- 有噪,不精确")
print(f"处理组 ATT 均值      = {per_unit_att.mean():.3f}   <- 聚合复原真值")
print(f"处理组 ATT 中位数    = {np.median(per_unit_att):.3f}")
print(f"处理组 ATT 标准差    = {per_unit_att.std():.3f}  (n={per_unit_att.size} 个处理单元)")

# %% [markdown]
# **读数。** 单案例 `firm_id=0` 的 ATT 偏离 −0.8，但 20 个处理单元的**均值与中位数**
# 都紧贴真实的 `-0.8`——DGP 效应在聚合层面被**如实复原**。这正是合成控制该有的诚实
# 姿态：它给单个案例一条可信的反事实路径，而非一个精确的总体点估计。

# %% [markdown]
# ### 2.4 反事实路径图
#
# 回到主案例（`firm_id=0`），`sv.pl.synth_path` 读 `models.synth['path']`，画
# treated（实线）与 synthetic（虚线）两条轨迹，并在处理时点画竖线：前期两线应几乎
# 重合（前期 RMSE 小），后期的**分岔**就是效应。

# %%
sv.pl.synth_path(st2, out="fig_synth.png")
print("已存:", st2.artifacts["figures"]["synth"])

# %% [markdown]
# ![合成控制路径图](fig_synth.png)
#
# 处理时点（2015）之前，蓝色 treated 与红色 synthetic 几乎贴合（前期 RMSE 小）；
# 之后 treated 系统性地**低于**合成对照，这条持续的向下缺口正是估计出的负向效应。
# 单案例的缺口幅度带噪（本例约 −1.3），但方向确定;真值 −0.8 由上一步的
# **聚合**估计如实复原。

# %% [markdown]
# ---
# ## 3. provenance：这条分析链自带审计轨迹
#
# 每个注册函数写状态时都**追加一条 provenance 记录**，于是一条跑完的分析链自带可复现的
# "证据脊柱"。`st.summary()` 把两条链各自的槽和步数打出来。

# %%
print("=== RDD 链 ===")
print(st.summary())
print("\n=== 合成控制链 ===")
print(st2.summary())

# %%
# provenance 逐条(RDD 链):看每一步 requires 什么、produces 什么
print("RDD 链 provenance:")
for rec in st.provenance:
    print(f"  step {rec['step']}: {rec['function']}  "
          f"requires={rec['requires']}  produces={rec['produces']}")

# %% [markdown]
# ---
# ## 小结：socialverse 与冠军包的差异
#
# 我们用两件准实验兵器各识别了一个因果效应，且**都复原了已知的 DGP 参数**：
#
# * **RDD**：局部线性 + 三角核（对标 `rdrobust`）估出断点跳跃 `τ̂ ≈ 3`，真值 `3.0`——
#   单次估计就精确命中。
# * **合成控制**：非负、和为一的捐赠者权重（对标 `Synth` / `gsynth` / `augsynth`）。
#   单案例 `firm_id=0` 的 ATT 有噪（前期 RMSE 小、方向正确），但 20 个处理单元的
#   **均值/中位数** ≈ `-0.8`,真值在**聚合层面**被如实复原——这是单案例 SCM 应有的诚实姿态。
#
# **差异不在算法，而在脊柱。** 与其把数据硬塞进一个统一矩阵，socialverse 把
# 每个方法登记进一张**依赖注册表**：`requires` / `produces` 契约是机器可读且 **live**
# 的（缺槽即 `RegistryError`，而非静默地算出垃圾），状态放在 12 槽 `StudyState`，
# 分析链自带 provenance。于是 "查注册表而非猜 API" 与 "复原已知参数" 一起，构成了
# 可被 agent 直接消费、可复现、可信的社科分析基座。
