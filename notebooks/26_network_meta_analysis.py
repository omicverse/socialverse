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
# # 网络 meta 分析:多种干预放到一起比
#
# 传统 meta 分析一次只合并**两种**干预的直接对比。但现实里往往有 A、B、C、D 好几种干预,
# 而**没有任何一项试验把它们两两都比过**——有的试验比 A vs B,有的比 C vs D,有的比 A vs D……
# 想知道「D 到底比 A 好多少」,如果只用直接比过 A 和 D 的试验,就浪费了「A vs B、B vs D」这条
# 间接路径携带的信息。**网络 meta 分析(network meta-analysis, NMA)** 把整张证据网络——直接 +
# 间接——一起放进一个模型,同时排出所有干预的优劣,并给出**任意两两**的合并对比。
#
# 这本 notebook 用频率学派的**对比法(contrast-based GLS)**,它是图论 `netmeta` 的**精确等价**:
# 同样的点估计、同样的方差,纯 numpy/scipy 实现,不跑 R、不跑 MCMC。多臂试验用**精确的簇内
# 协方差**(共享基线臂)处理。我们会走通:
#
# 1. **臂层 → 对比数据**:把 study×treatment 的事件/样本表压成各非基线臂 vs 基线的 log-OR;
# 2. **网络 meta 合并 + 联赛表**:一次估出所有处理 vs 参照,以及全配对的 league table;
# 3. **排名**:P-score(频率学派 SUCRA,闭式)+ 排序概率图 rankogram/SUCRA;
# 4. **不一致性**:设计×处理全局 Q 分解 + 节点劈分(直接 vs 间接冲突);
# 5. **成分 NMA**:把复合干预拆成加性成分;
# 6. **两张图**:网络几何图 + 联赛表热图。
#
# 关键卖点:数据是**一致网络**,真值 log-odds 为 A=−0.5、B=0、C=0.5、D=1.0(D vs A = **+1.5**)——
# 我们要看网络估计能不能**还原这个真值**,并确认没有直接/间接冲突。
#
# > **对标**:R `netmeta`(图论频率学派)/ `gemtc`(贝叶斯)/ `BUGSnet`。

# %%
import os
import sys

# 确保用的是本 worktree 里的 socialverse(而不是环境里 editable 安装指向的其它 checkout)
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # 在 Jupyter cell 里没有 __file__,退回当前工作目录
    _HERE = os.path.abspath(os.getcwd())
_ROOT = os.path.dirname(_HERE) if os.path.basename(_HERE) == "notebooks" else _HERE
if os.path.isdir(os.path.join(_ROOT, "socialverse")) and _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")  # 无显示环境:图直接写文件
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
import numpy as np
import pandas as pd
from IPython.display import Image

import socialverse as sv
from socialverse import datasets as ds

# 让本 notebook 自绘的图也能显示中文标签
_CJK = ["PingFang SC", "Hiragino Sans GB", "Songti SC", "STHeiti",
        "Arial Unicode MS", "Noto Sans CJK SC", "Microsoft YaHei"]
_have = {f.name for f in _fm.fontManager.ttflist}
plt.rcParams["font.sans-serif"] = [c for c in _CJK if c in _have] + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

print("socialverse", sv.__version__)

# %% [markdown]
# ## 1. 数据:4 种干预 A/B/C/D 的臂层试验
#
# 每一行是一个**臂(arm)**:某项试验里某种干预下有多少人发生了结局(`events`)、总人数(`n`)。
# 一项两臂试验占两行(共享 `study`)。这个网络里六种两两对比(A-B、A-C、B-C、B-D、C-D、A-D)
# 各有几项试验,拼成一张**连通**的证据网络。因为结局越大越「好」(事件率高),后面 log-odds
# 越高代表干预越强,真值 D(1.0)> C(0.5)> B(0)> A(−0.5)。

# %%
net = ds.load_network_trials()
print(f"试验数 = {net['study'].nunique()},臂数 = {len(net)},处理 = {sorted(net['treat'].unique())}")
net.head(10)

# %% [markdown]
# ## 2. 臂层 → 对比数据:sv.pp.nma_pairwise
#
# **解决什么问题**:网络 meta 的引擎吃的是**对比(contrast)**——「本研究里 treat1 相对 treat2 的
# log-OR + 标准误」,而不是原始臂层事件数。**关键前提**:每项研究选一个基线臂(这里取每项试验
# 的第一臂),其余臂都对它作对比;多臂试验里各对比**共享基线臂的方差**,这就是簇内协方差的来源。
# **哪几步**:`nma_pairwise` 按 `study=`/`treatment=` 分组,二分类给 `events=`/`n=` → 每个对比算
# log-OR、seTE,并记下基线臂方差 `vbase`,写进 `models["nma_contrasts"]`。

# %%
study = sv.StudyState()
study.write("sources", "datasets", net)

sv.pp.nma_pairwise(study, study="study", treatment="treat", events="events", n="n")

contrasts = study.models["nma_contrasts"]
print(f"对比数 = {len(contrasts)}(效应量 = {contrasts['measure'].iloc[0]})")
contrasts[["studlab", "treat1", "treat2", "TE", "seTE"]].round(3).head(10)

# %% [markdown]
# ## 3. 网络 meta 合并:sv.tl.netmeta —— 一次排出所有处理
#
# **解决什么问题**:把上面所有对比(直接证据)通过共享的处理节点连起来,做**广义最小二乘(GLS)**,
# 同时估出每个处理 vs 参照的效应——间接证据自动通过网络传导进来。**关键前提**:网络连通、
# 用随机效应(`comb='random'`)吸收研究间异质。**哪几步**:构造设计矩阵 X(对比 × 基本参数)、
# 协方差 V(含多臂簇内协方差),GLS 解出 β 与 vcov,再展开成全配对**联赛表**。
# 参照默认取字典序第一的处理 A——正好是我们的真值参照。

# %%
sv.tl.netmeta(study, reference="A", comb="random")
nma = study.models["nma"]

print(f"参照 = {nma['reference']}  ·  效应量 = {nma['sm']}(log-OR)  ·  模型 = {nma['model']}")
print(f"τ² = {nma['tau2']:.4f}   Q = {nma['Q']:.2f} (df={nma['df']}, p={nma['Q_pval']:.3f})   I² = {nma['I2']:.1f}%")
print()
print("各处理 vs 参照 A(应还原真值:B≈+0.5, C≈+1.0, D≈+1.5):")
for t, e in nma["effects"].items():
    print(f"  {t} vs A:  {e['vs_ref']:+.3f}  (SE {e['se']:.3f})")

# %% [markdown]
# 三个非参照处理的估计与真值差(B 真值 −0.5−(−0.5)=+0.5,C=+1.0,D=+1.5)非常接近——网络把直接 +
# 间接证据合起来,成功**还原了 D vs A ≈ 1.5**。下面把 league table 拉成一张矩阵看全配对。

# %%
tr = nma["treatments"]
league_mat = pd.DataFrame(index=tr, columns=tr, dtype=object)
for i in tr:
    for j in tr:
        if i == j:
            league_mat.loc[i, j] = "—"
        else:
            c = nma["league"][f"{i} vs {j}"]
            league_mat.loc[i, j] = f"{c['estimate']:+.2f} [{c['ci_lb']:+.2f},{c['ci_ub']:+.2f}]"
print("联赛表:行 vs 列 的 log-OR [95% CI]")
league_mat

# %% [markdown]
# ## 4. 排名(一):sv.tl.netrank —— P-score(频率学派 SUCRA)
#
# **解决什么问题**:决策者要的是「哪种干预最好」的**排名**,而不只是两两对比。**P-score** 是
# SUCRA 的频率学派闭式版:某处理**平均优于其它处理的概率**。**关键前提**:说清「小值好还是坏」——
# 这里事件率越高越好,所以 `small_values='undesirable'`(小的 log-odds 不理想)。**哪几步**:对每对
# 处理用正态尾概率算「i 优于 k」,再对所有 k 取平均。P-score 越接近 1 越好。

# %%
sv.tl.netrank(study, small_values="undesirable")
pscore = study.diagnostics["netrank"]["pscore"]
print("P-score 排名(越大越好,真优序 D>C>B>A):")
for t, s in pscore.items():
    print(f"  {t}:  {s:.3f}")

# %% [markdown]
# ## 5. 排名(二):sv.tl.nma_rankogram —— 排序概率图 + SUCRA
#
# **解决什么问题**:P-score 给一个标量;**rankogram** 给更细的分布——每个处理**占据每个名次的概率**。
# **关键前提**:从网络估计的多元正态(β, cov)里蒙特卡洛抽样,统计每次抽样里各处理的排名。
# **哪几步**:MVN 抽样 → 对每次抽样按 log-odds 排名 → 汇总各名次频率 → SUCRA(名次分布的曲线下面积)。
# 同样声明 `small_values='undesirable'`。

# %%
sv.tl.nma_rankogram(study, small_values="undesirable", nsim=5000, seed=42)
rk = study.diagnostics["rankogram"]
print("SUCRA(越大越好):")
for t, s in rk["SUCRA"].items():
    print(f"  {t}:  {s:.3f}")
print()
print("排序概率(行=处理,列=名次1..4;名次1=最好):")
pd.DataFrame(rk["rank_probabilities"], index=[f"名次{r}" for r in range(1, len(tr) + 1)]).T.round(3)

# %% [markdown]
# ## 6. 不一致性(一):sv.tl.nma_inconsistency —— 设计×处理全局 Q 分解
#
# **解决什么问题**:网络 meta 的核心假设是**一致性**——直接证据和间接证据不冲突。总的失拟 Q 可以拆成
# 两部分:**设计内异质性**(同一种对比集合的多项试验之间的散布)和**设计间不一致性**(不同对比路径
# 给出的估计彼此矛盾)。**关键前提**:先跑过 `netmeta`(拿到总 Q)。**哪几步**:按「设计」(每项研究
# 比较的处理集合)分组池化算设计内 Q_het,不一致性 Q_inc = Q_total − Q_het,对其做 χ² 检验。
# 数据是一致网络,期望 p 不显著。

# %%
sv.tl.nma_inconsistency(study)
inc = study.diagnostics["nma_inconsistency"]
print(f"总 Q            = {inc['Q_total']:.2f}")
print(f"设计内异质性 Q  = {inc['Q_heterogeneity']:.2f}")
print(f"设计间不一致 Q  = {inc['Q_inconsistency']:.2f}  (df={inc['df_inconsistency']}, p={inc['inconsistency_pval']:.3f})")
print(f"→ 存在不一致?  {inc['inconsistent']}   (一致网络,如期不显著)")

# %% [markdown]
# ## 7. 不一致性(二):sv.tl.netsplit —— 节点劈分(直接 vs 间接)
#
# **解决什么问题**:全局 Q 说「整体一不一致」;**节点劈分(SIDE)** 逐个对比地问:这条对比的**直接**
# 估计和绕开它的**间接**估计一致吗?**关键前提**:对比要同时有直接证据和间接路径;间接估计用
# **反算法**从网络估计和直接估计倒推(1/se_net² = 1/se_dir² + 1/se_ind²)。**哪几步**:每个有直接
# 证据的对比,随机效应池化直接估计,反算间接估计,z 检验二者之差。p 都不显著 = 无局部冲突。

# %%
sv.tl.netsplit(study)
ns = study.diagnostics["netsplit"]
split_df = pd.DataFrame(ns["comparisons"])
print(f"检出不一致的对比数 = {ns['n_inconsistent']}(一致网络,期望 0)")
split_df[["comparison", "direct", "indirect", "network", "difference", "pval"]].round(3)

# %% [markdown]
# ## 8. 成分 NMA:sv.tl.netcomb —— 把复合干预拆成加性成分
#
# **解决什么问题**:当干预是**可拆的组合**(如「A+B」= 成分 A 叠加成分 B)时,成分 NMA 不再把每种组合
# 当成独立处理,而是估计**各成分的加性效应**,处理效应 = 其成分效应之和。这样能借用不同组合之间的
# 共享成分信息,还能外推没直接试过的组合。**关键前提**:处理名要能按 `sep='+'` 拆成成分。
# **哪几步**:把处理名拆成成分、构造成分设计矩阵、GLS 估各成分效应。
#
# 本数据的处理名是原子的 A/B/C/D(不含「+」),此时**每个处理自成一个成分**,成分效应即处理效应——
# 我们先在原始网络上演示这个退化情形(成分 = 各处理本身),验证它与 `netmeta` 的处理效应一致。

# %%
sv.tl.netcomb(study, sep="+")
comp = study.models["nma_components"]
print("成分效应(处理名原子 ⇒ 成分即处理;与 netmeta 的处理效应对应):")
for c, e in comp["effects"].items():
    print(f"  成分 {c}:  {e['estimate']:+.3f}  (SE {e['se']:.3f}, p={e['pval']:.3f})")

# %% [markdown]
# 为真正展示「组合分解」,我们把网络**重标签**成加性名字:把每个非参照处理写成「基线成分 X + 增量成分」
# ——A→`X`,B→`X+b`,C→`X+c`,D→`X+d`。于是 X 承担共同基线,b/c/d 是各自的**增量成分效应**,应分别
# 还原真值差 +0.5 / +1.0 / +1.5。这演示了 `netcomb` 在处理名含「+」时如何解析并估各成分。

# %%
relabel = {"A": "X", "B": "X+b", "C": "X+c", "D": "X+d"}
add_study = sv.StudyState()
add_contrasts = study.models["nma_contrasts"].copy()
add_contrasts["treat1"] = add_contrasts["treat1"].map(relabel)
add_contrasts["treat2"] = add_contrasts["treat2"].map(relabel)
add_study.write("models", "nma_contrasts", add_contrasts)

sv.tl.netcomb(add_study, sep="+")
add_comp = add_study.models["nma_components"]
print("加性重标签后的成分效应(增量成分 b/c/d 应还原 +0.5 / +1.0 / +1.5):")
for c, e in add_comp["effects"].items():
    print(f"  成分 {c!r}:  {e['estimate']:+.3f}  (SE {e['se']:.3f}, p={e['pval']:.3f})")

# %% [markdown]
# ## 9. 网络几何图:sv.pl.netgraph
#
# **解决什么问题**:在做任何合并前,先看**网络长什么样**——哪些处理直接比过、哪条对比证据多、
# 网络连不连通、有没有闭环(闭环才有间接证据可校验一致性)。**哪几步**:处理作节点(环形布局),
# 直接对比作边,边宽 ∝ 该对比的研究数。这里六种两两对比都有,形成含多个闭环的密集网络。

# %%
sv.pl.netgraph(study, out="fig26_netgraph.png", title="4 种干预的证据网络(边宽 ∝ 研究数)")
Image("fig26_netgraph.png")

# %% [markdown]
# ## 10. 联赛表热图:sv.pl.netheat
#
# **解决什么问题**:把全配对的联赛表用颜色一眼呈现——每格是「行处理 vs 列处理」的合并 log-OR,
# 颜色编码方向与大小。**哪几步**:读 `models["nma"]` 的处理效应,作差得全配对矩阵并热图化。
# 一致网络下,该矩阵应干净地反映 D>C>B>A 的阶梯(D vs A 最红/最大)。

# %%
sv.pl.netheat(study, out="fig26_netheat.png", title="联赛表热图(行 vs 列,log-OR)")
Image("fig26_netheat.png")

# %% [markdown]
# ## 小结:一条可复现的证据链
#
# 我们用一份**一致网络**走完了频率学派网络 meta 分析,每一步都落在 `StudyState` 上、每个数字都能追到
# 是哪个 `sv.*` 函数产出的,并确认网络**还原了真值**(D vs A ≈ 1.5)、**没有**直接/间接冲突:
#
# ```
# 臂层表 ──sv.pp.nma_pairwise──▶ 对比数据(log-OR + 簇内协方差)
#        ──sv.tl.netmeta(GLS, random)──▶ 全处理效应 + 联赛表(D vs A ≈ +1.5,还原真值)
#        ──sv.tl.netrank / nma_rankogram──▶ 排名 D>C>B>A(P-score / SUCRA 一致)
#        ──sv.tl.nma_inconsistency──▶ 设计间不一致 Q 不显著
#        ──sv.tl.netsplit──▶ 逐对比直接≈间接,0 处冲突
#        ──sv.tl.netcomb──▶ 成分加性效应(重标签演示 b/c/d 还原 +0.5/+1.0/+1.5)
#        ──sv.pl.netgraph / netheat──▶ 网络几何 + 联赛表热图
# ```
#
# **要点**:网络 meta 让你用上全部直接 + 间接证据、一次排出所有干预,但**一致性是前提**——排名之前
# 务必用全局 Q 分解 + 节点劈分检查直接/间接是否冲突;报告排名要用 P-score/SUCRA 而非只看点估计名次。
# 频率学派对比法(本 notebook)与贝叶斯 `gemtc` 结果通常高度一致,且**精确、无需 MCMC 调参**。
# 下一本会转向**诊断准确性 meta 分析**(双变量模型 + SROC),那是另一类结局(敏感度/特异度)的合并。
