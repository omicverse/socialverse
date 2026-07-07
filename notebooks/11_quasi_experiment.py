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
# # 没有随机分配时,如何识别因果:断点回归与合成控制
#
# 理想的因果推断靠随机对照实验,但社会科学的现实里,处理往往不是随机分配的:一项补贴按分数线发放、一座城市单方面加了税。这时我们退而求其次,寻找那些「近似随机」的缝隙,用准实验设计从观测数据里挤出因果效应。本教程走两件最常用的兵器,它们各自利用一种不同的缝隙。
#
# 第一件是**断点回归(Regression Discontinuity, RDD)**。当处理由一条清晰的规则触发——分数 ≥ 60 才录取、收入 ≤ 阈值才补贴——那么在断点两侧无限接近的地方,个体几乎是被「随机」地分到了处理组和对照组:考 59.9 分和 60.1 分的人,能力上几乎没差别,差别只在一个落在线上、一个落在线下。于是结果变量在断点处的**跳跃**,就是这条规则带来的局部因果效应(LATE)。关键的难点在于:断点邻域之外的结果曲线可能又弯又斜,一条粗暴的全局直线会把这份弯曲误当成跳跃,所以正确做法是只在断点**附近**做局部线性拟合。
#
# 第二件是**合成控制(Synthetic Control, SCM)**。有些处理天然只作用于**一个**单位:一个国家改了法、一座城市办了世博。世上没有第二个一模一样的国家能当对照。合成控制的办法是,用一组没被处理的「捐赠者」单位的**加权组合**,拼出一个「假如这个单位没被处理、本会怎样」的合成反事实;处理之后,真实轨迹与合成轨迹之间的缺口就是效应。它的难点在于反事实的可信度——只有当合成对照在处理**之前**紧贴真实单位时,处理**之后**的缺口才配叫效应。
#
# 我们在两个内置玩具数据上演示。它们都由**已知参数**的真实数据生成过程造出(RDD 的真实跳跃 τ = 3.0,面板的真实处理效应 att = −0.8),所以教程结束时,你能亲眼看到估计量是否**复原了已知的真值**。全流程用 `socialverse` 完成,它是一套面向社会科学的分析库,对标 R 的 `rdrobust`、`Synth` / `gsynth` 等专用包;它多做的一件事,我们留到最后一节再讲。

# %%
import matplotlib
matplotlib.use("Agg")  # 无显示环境:图直接写文件

import os
import sys

# 让 notebook 无论从哪启动都加载 worktree 里的 socialverse,并把图存到本目录。
try:
    _here = os.path.dirname(os.path.abspath(__file__))
except NameError:  # 交互式内核里没有 __file__
    _here = os.getcwd()
_root = os.path.dirname(_here)
if os.path.isdir(os.path.join(_root, "socialverse")) and _root not in sys.path:
    sys.path.insert(0, _root)
if os.path.isdir(_here):
    os.chdir(_here)

import numpy as np
import pandas as pd

import socialverse as sv
from socialverse import datasets as ds

# %% [markdown]
# ---
# # 第一部分:断点回归
#
# ## 载入 RDD 数据
#
# `ds.load_rdd(tau=3.0)` 生成一个锐性断点数据集:running 变量在 `cutoff=0` 处触发处理,真实跳跃 `τ = 3.0`。四列分别是 `running`(决定处理的 running 变量)、`treat`(0/1 处理指示)、`y`(结果)、`x`(一个协变量)。这里刻意在结果里埋了斜率和曲率(`1.5·running − 0.8·running²`),正是为了逼你在断点附近做局部拟合——否则全局直线会被这份曲率带偏。

# %%
rdd_df = ds.load_rdd(tau=3.0)
rdd_df.head()

# %% [markdown]
# 看一眼断点两侧的样本量:cutoff=0 左右各有几百个观测,足够在断点邻域内做局部拟合。

# %%
n_left = int((rdd_df["running"] < 0).sum())
n_right = int((rdd_df["running"] >= 0).sum())
print(f"样本量 = {len(rdd_df)}  |  cutoff=0 左侧 {n_left}  右侧 {n_right}")

# %% [markdown]
# ## 声明结果与估计目标
#
# 动手估计前,先把「哪一列是结果、我们要估的是什么」写进研究状态。断点回归识别的是断点处的**局部**平均处理效应(LATE),所以 `estimand.target` 写 `LATE`;结果变量是 `y`;再把数据本身放进状态。后续的估计函数都从这里读设定,不必反复传参。

# %%
st = sv.StudyState()
st.write("estimand", "target", "LATE")   # 断点回归识别的是局部平均处理效应
st.write("variables", "outcome", "y")    # 结果变量
st.write("sources", "datasets", rdd_df)  # 数据本身
st

# %% [markdown]
# ## 估计断点跳跃
#
# `sv.tl.rdd` 做的事很具体:先用一条数据驱动的经验法则(IK-lite)挑一个带宽 `h`,只保留断点 `±h` 范围内的观测;然后在断点左右两侧各做一次**三角核加权**的局部线性回归(权重从断点处的 1 平滑衰减到带宽边缘的 0,让估计聚焦在断点邻域);读出两侧拟合线在断点处的截距,**右侧截距 − 左侧截距**就是跳跃。它原地更新 `st`,把跳跃、标准误、两侧截距写进 `models.rdd`,把带宽和两侧有效样本量写进 `diagnostics.bandwidth`。

# %%
sv.tl.rdd(st, running="running", cutoff=0.0)  # running 变量列名 + 断点位置

model = st.models["rdd"]
bw = st.diagnostics["bandwidth"]
print(f"真实跳跃 τ  = 3.0   (数据生成过程已知)")
print(f"估计跳跃 τ̂  = {model['jump']:.4f}   SE={model['se']:.4f}  t={model['t']:.2f}")
print(f"  左侧断点截距 = {model['left_intercept']:.4f}")
print(f"  右侧断点截距 = {model['right_intercept']:.4f}")
print(f"带宽 h = {bw['h']:.4f} ({bw['selector']})  |  左侧 n={bw['n_left']}  右侧 n={bw['n_right']}")

# %% [markdown]
# 估计跳跃 `τ̂ ≈ 2.99`,与真实的 `3.0` 几乎重合,标准误只有约 0.10,t 值高达 29——断点处结果确有一个显著、且量级正确的跳升。左右两侧的断点截距(约 1.99 与 4.98)之差正好就是这个跳跃。局部线性加三角核让我们把注意力锁在断点邻域,避开了远处曲率的干扰,这正是 `rdrobust` 的核心思路。

# %% [markdown]
# ## 断点图
#
# 数字之外,断点图能一眼看出跳跃是否真实存在。`sv.pl.rdd_plot` 读 `models.rdd`,把 running 变量在两侧做等计数分箱、画出每箱的结果均值散点,再叠上两侧的局部线性拟合线;两条线在断点处的垂直落差就是估计的跳跃。`out=` 直接把图存成同目录下的 PNG。

# %%
sv.pl.rdd_plot(st, out="fig_rdd.png")
print("已存:", st.artifacts["figures"]["rdd"]["path"])

# %% [markdown]
# ![RDD 断点图](fig_rdd.png)
#
# 左侧(蓝)与右侧(绿)的分箱均值各自拟合出一条局部线性线,它们在 `running=0` 的红色虚线处的垂直落差约等于 3,正是我们估计出的因果跳跃。断点两侧曲线各自平滑延伸、只在断点处断开,说明这个落差不是曲率假象。

# %% [markdown]
# ---
# # 第二部分:合成控制
#
# ## 载入面板数据
#
# `ds.load_did_panel(att=-0.8)` 是一个带**平行预趋势**的面板:40 家企业 × 8 年(2010–2017),前一半企业在 2015 年被处理,真实处理效应 `att = −0.8`。列包括单位 `firm_id`、时间 `year`、处理组标记 `treat`、结果 `y` 等。合成控制专治「单个处理单位」的场景,所以我们只把**一个** treated 单位(`firm_id=0`)拎出来,用其余捐赠者单位的加权组合替它造反事实。

# %%
panel = ds.load_did_panel(att=-0.8)
panel.head()

# %% [markdown]
# 确认一下面板的规模和处理时点:40 个单位、8 个年份、处理从 2015 年开始。

# %%
print("单位数:", panel["firm_id"].nunique(),
      "|  年份:", f"{panel['year'].min()}–{panel['year'].max()}",
      "|  处理起始年: 2015")

# %% [markdown]
# ## 声明设计与估计目标
#
# 合成控制要多知道两件事:哪一列标记「曾被处理」(这样才能把处理过的单位踢出捐赠池,免得污染反事实),哪一列是时间轴。所以除了结果 `y` 和数据,还要把 `design.treatment` 与 `design.time` 写进状态。注意这里估计目标是 **ATT**(处理组的平均处理效应),而非断点回归的 LATE。

# %%
st2 = sv.StudyState()
st2.write("design", "treatment", "treat")   # 用来把「曾被处理」的单位踢出捐赠池
st2.write("design", "time", "year")         # 时间轴
st2.write("variables", "outcome", "y")      # 结果变量
st2.write("estimand", "target", "ATT")      # 估计处理组平均处理效应
st2.write("sources", "datasets", panel)
st2

# %% [markdown]
# ## 求解捐赠者权重并估 ATT
#
# `sv.tl.synthetic_control` 先把长面板 pivot 成「时间 × 单位」的结果矩阵,在 `treat_time=2015` 处切成处理前 / 处理后两段。然后用带约束的优化(SLSQP)求一组**非负、且和为一**的捐赠者权重,让 treated 单位与它的合成对照在**处理前**的均方误差最小;这组权重加权出的捐赠者路径就是反事实,处理后 treated 与合成对照之间的平均缺口就是 ATT。任何**曾被处理**的单位都会被自动排除出捐赠池。

# %%
sv.tl.synthetic_control(st2, unit="firm_id", time="year",
                        treated_unit=0, treat_time=2015)  # 处理 firm_id=0, 从 2015 起

synth = st2.models["synth"]
pre_fit = st2.diagnostics["pre_fit"]
print(f"真实 att  = -0.8   (数据生成过程已知)")
print(f"估计 ATT  = {synth['att']:.4f}   (单个 treated 单位: firm_id=0)")
print(f"捐赠者数量 = {synth['n_donors']}   (已剔除曾被处理的单位)")
print(f"处理前拟合 = pre-RMSE {pre_fit['pre_rmse']:.4f}"
      f"  (前期 {pre_fit['n_pre']} 点, 后期 {pre_fit['n_post']} 点)")

# %% [markdown]
# 权重非负且和为一,意味着合成对照是几个捐赠者的一个**凸组合**——"firm_id=0 ≈ 这几家企业的加权平均"。看一眼贡献最大的几个捐赠者:

# %%
top_w = sorted(synth["weights"].items(), key=lambda kv: -kv[1])[:5]
pd.DataFrame(
    [{"donor_firm_id": int(d), "weight": round(w, 3)} for d, w in top_w]
)

# %% [markdown]
# 处理前 RMSE 只有约 0.24,说明合成对照在 2015 年之前紧贴真实的 `firm_id=0`——反事实站得住。但要留意:这个**单案例**点估计(ATT ≈ −1.28)并不精确等于真值 −0.8。原因很实在:合成控制只对着**一个**处理单位、只有 5 个处理前的点、捐赠者本身还带噪声,它没有传统意义上的抽样分布,单案例估计天生就是**有噪**的。下一步我们用一个诚实的检查说明:真值是在**聚合**层面被复原的。

# %% [markdown]
# ## 单案例有噪,聚合复原真值
#
# 把同一套合成控制流程**逐个**套到全部 20 个处理单位上,收集每个单位各自的 ATT。单看任意一个单位都会偏离真值,但如果 DGP 是诚实的,这 20 个 ATT 的**均值与中位数**应当回到 −0.8 附近。

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
print(f"真实 att           = -0.8")
print(f"单案例 (firm_id=0) = {synth['att']:.3f}    <- 有噪, 不精确")
print(f"处理组 ATT 均值     = {per_unit_att.mean():.3f}    <- 聚合复原真值")
print(f"处理组 ATT 中位数   = {np.median(per_unit_att):.3f}")
print(f"处理组 ATT 标准差   = {per_unit_att.std():.3f}   (n={per_unit_att.size} 个处理单位)")

# %% [markdown]
# 单案例 `firm_id=0` 的 ATT 偏到了 −1.28,但 20 个处理单位的均值 −0.84、中位数 −0.81,都紧贴真实的 −0.8。这正是合成控制该有的诚实姿态:它给单个案例一条可信的反事实路径,而非一个精确的总体点估计;真值要在聚合层面才如实浮现。

# %% [markdown]
# ## 反事实路径图
#
# 回到主案例 `firm_id=0`,`sv.pl.synth_path` 读 `models.synth` 里的路径,画出 treated(实线)与 synthetic(虚线)两条轨迹,并在处理时点画一条竖线。处理前两线应几乎重合(处理前 RMSE 小),处理后的分岔就是效应。

# %%
sv.pl.synth_path(st2, out="fig_synth.png")
print("已存:", st2.artifacts["figures"]["synth"]["path"])

# %% [markdown]
# ![合成控制路径图](fig_synth.png)
#
# 2015 年之前,treated 与 synthetic 两条线几乎贴合;之后 treated 系统性地低于合成对照,这条持续向下的缺口就是估计出的负向效应。单案例的缺口幅度带噪(本例约 −1.3),但方向确定;真值 −0.8 由上一节的聚合估计如实复原。

# %% [markdown]
# ---
# ## 可复现的证据链
#
# 前面所有步骤都是普通的统计估计,任何一套脚本都能做。`socialverse` 额外做的一件事,是把每一步都记进研究状态的一份账本里:用了哪个函数、需要什么前置、产出了什么。于是一条跑完的分析链自带可追溯的「证据脊柱」——在社会科学里,「这个结论从哪一步、哪份数据来」往往和结论本身同等重要。`st.summary()` 把两条链各自的槽和步数打出来。

# %%
print("=== 断点回归链 ===")
print(st.summary())
print("\n=== 合成控制链 ===")
print(st2.summary())

# %% [markdown]
# 更细一层,可以把 RDD 链的 provenance 逐条摊开,看每一步 requires 什么、produces 什么。这份契约是活的:如果某一步的前置槽没填,调用会在计算之前就被拦下并报错,而不是默默算出一个错得离谱的结果。

# %%
for rec in st.provenance:
    print(f"step {rec['step']}: {rec['function']}")
    print(f"    requires={rec['requires']}")
    print(f"    produces={rec['produces']}")

# %% [markdown]
# ## 小结
#
# 我们用两件准实验兵器各识别了一个因果效应,且都复原了已知的真值:断点回归(对标 `rdrobust`)用局部线性加三角核估出断点跳跃 `τ̂ ≈ 2.99`,单次就精确命中真值 3.0;合成控制(对标 `Synth` / `gsynth` / `augsynth`)用非负、和为一的捐赠者权重造反事实,单案例带噪但 20 个处理单位聚合后的均值/中位数 ≈ −0.8,如实复原真值。
#
# 与纯估计工具相比,`socialverse` 多给的不是更快的算法,而是一条**自带证据链**的分析脊柱:每个方法的前置与产出都是机器可读且会真的拦住你的契约,状态和 provenance 贯穿始终,让结论可追溯、可复现。下一本教程 [12_psychometrics](12_psychometrics.ipynb) 转向测量:如何用项目反应理论从一堆题目里估出潜在能力。
