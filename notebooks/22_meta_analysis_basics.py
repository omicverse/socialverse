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
# # Meta 分析入门:把一堆研究合并成一个可辩护的估计
#
# 一个问题问了很多遍,答案却散在几十篇论文里:卡介苗到底能不能防结核?一项心理干预
# 的效应有多大?每篇研究给一个数字,彼此还不完全一致——有的说有效,有的说没差别,样本
# 量从几十到几万不等。**Meta 分析(meta-analysis)** 就是把这些研究的效应量按精度加权
# 合并成一个总估计,并**诚实地报告它的不确定性**。它不是简单求平均:大样本研究该有更大
# 发言权,研究之间真实的差异(异质性)必须量化而不是抹平。
#
# 这本 notebook 用一份**真实的、经典的**数据——13 项卡介苗防结核试验(Colditz et al.
# 1994,也是 R 的 `metafor` 包最常用的教学数据 `dat.bcg`)——走通一次标准 meta 分析:
#
# 1. **效应量**:把每项试验的 2×2 表算成对数风险比(log risk ratio)+ 抽样方差;
# 2. **固定效应 vs 随机效应**:两种合并假设,以及为什么随机效应几乎总是更诚实;
# 3. **异质性**:Cochran's Q、I²、τ²——研究之间的差异有多大、是不是真实的;
# 4. **Knapp-Hartung**:小样本更稳健的置信区间;
# 5. **预测区间**:未来一项研究的真值大概落在哪——比合并点估计更能说明「异质」的后果;
# 6. **发表偏倚**:Egger 检验 + 漏斗图;
# 7. **森林图**:把全部证据画进一张图。
#
# 关键卖点:`socialverse` 的合并结果与 `metafor` **逐位吻合**(随机效应 log-RR = −0.7145,
# τ² = 0.313,I² = 92%)——同样的统计量,纯 numpy/scipy 原生实现,不依赖 R。
#
# > **对标**:R `metafor::rma` / `meta::metabin` · Stata `meta`。

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
# ## 1. 数据:13 项卡介苗试验
#
# 每项试验是一个 2×2 表:接种组里有多少人得结核(`tpos`)、没得(`tneg`);对照组同理
# (`cpos` / `cneg`)。还有一个后面会用到的调节变量——试验地点的**绝对纬度** `ablat`
# (纬度越高、越冷的地区,卡介苗似乎越有效,这是这份数据最著名的发现)。

# %%
bcg = ds.load_bcg()
bcg

# %% [markdown]
# ## 2. 效应量:对数风险比
#
# 直接比较各研究的「得病人数」没有意义(样本量不同)。要先把每项试验压缩成一个**可比的
# 效应量** + 它的**抽样方差**。二分类结局的标准选择是**风险比(risk ratio, RR)**:接种组
# 的发病风险 ÷ 对照组的发病风险。RR < 1 表示保护。为了让它对称、方差好算,合并时用它的
# 对数 `log(RR)`。
#
# `socialverse` 把一次分析组织成一个 `StudyState` 对象:数据放进去,每个 `sv.*` 函数
# 声明「需要什么、产出什么」,顺着契约走。`sv.pp.escalc` 是效应量计算的总入口,按
# `measure=` 路由——这里选 `"RR"`,并顺手把 `trial` 存成研究标签、把 `ablat`/`year` 带成
# 调节变量。

# %%
study = sv.StudyState()
study.write("sources", "datasets", bcg)

sv.pp.escalc(study, measure="RR",
             ai="tpos", bi="tneg", ci="cpos", di="cneg",
             study="trial", slab="trial", moderators=["ablat", "year"])

eff = study.models["meta_effects"]
eff[["slab", "yi", "vi", "sei"]].round(4)

# %% [markdown]
# 每行现在有 `yi`(log-RR)和 `vi`(抽样方差)。多数 `yi` 是负的——大方向就是保护性的。
# 但研究之间差别不小(从近乎无效到强保护),这正是下面要量化的**异质性**。

# %% [markdown]
# ## 3. 固定效应 vs 随机效应
#
# 有两种合并假设:
#
# - **固定/共同效应(fixed-effect)**:假设所有研究估计的是**同一个**真值,差异纯粹来自
#   抽样误差。按 `1/vi` 加权平均。这个假设在这里几乎肯定不成立——不同年代、不同纬度、不同
#   人群的试验,真效应本就该不同。
# - **随机效应(random-effects)**:假设各研究的真效应围绕一个总体均值**分布**,方差为
#   `τ²`。权重变成 `1/(vi + τ²)`——研究间差异越大,大研究的相对优势就越被拉平。
#
# 除非有很强理由相信「同一个真值」,**随机效应是默认选择**。τ² 的默认估计量是 REML。

# %%
sv.tl.meta_fixed(study)
fe = study.models["meta"].copy()

sv.tl.meta_random(study, method="REML")
re = study.models["meta"]

print(f"固定效应  log-RR = {fe['estimate']:.4f}  95% CI [{fe['ci_lb']:.3f}, {fe['ci_ub']:.3f}]")
print(f"随机效应  log-RR = {re['estimate']:.4f}  95% CI [{re['ci_lb']:.3f}, {re['ci_ub']:.3f}]   τ² = {re['tau2']:.4f}")
print(f"随机效应  RR = {np.exp(re['estimate']):.3f}  →  接种把结核风险降到约 {np.exp(re['estimate'])*100:.0f}%")
print()
print(f"✓ 与 metafor(REML)逐位吻合:log-RR = -0.7145,τ² = 0.313")

# %% [markdown]
# 随机效应的置信区间明显更宽——因为它把「研究之间的真实差异」也算进了不确定性。合并结果:
# 接种把结核风险降到约 **49%**(减半),但下面会看到,这个「平均」掩盖了很大的异质。

# %% [markdown]
# ## 4. 异质性:研究之间差多少
#
# 合并点估计**从不单独汇报**——必须同时给异质性。三个量:
#
# - **Cochran's Q**:观测到的离散是否超过抽样误差能解释的(χ² 检验);
# - **I²**:总变异中「研究间真实差异」占的百分比。>50% 算实质异质,>75% 算高度异质;
# - **τ² / τ**:研究间真效应的方差 / 标准差(和效应量同尺度,可解释)。

# %%
sv.tl.meta_heterogeneity(study)
het = study.diagnostics["heterogeneity"]
print(f"Q = {het['Q']:.1f}  (df = {het['df']}, p = {het['Q_pval']:.2e})")
print(f"I² = {het['I2']:.1f}%      H² = {het['H2']:.2f}")
print(f"τ² = {het['tau2']:.4f}     τ = {het['tau']:.3f}")
print(f"→ I² = {het['I2']:.0f}%:绝大部分离散是研究间真实差异,不是抽样噪声。合并的『平均』要非常小心地解读。")

# %% [markdown]
# ## 5. Knapp-Hartung:小样本更诚实的置信区间
#
# 只有 13 项研究时,标准的正态(z)置信区间会偏窄——它当作 τ² 已知,其实 τ² 也是估出来的。
# **Knapp-Hartung-Sidik-Jonkman(HKSJ)** 校正改用 t 分布 + 一个尺度修正,置信区间通常更宽,
# 覆盖率更接近名义 95%。研究数少时**强烈建议开**。

# %%
sv.tl.meta_random(study, method="REML", knapp_hartung=True)
hk = study.models["meta"]
print(f"普通 z 区间   [{re['ci_lb']:.3f}, {re['ci_ub']:.3f}]  (宽 {re['ci_ub']-re['ci_lb']:.3f})")
print(f"HKSJ  t 区间  [{hk['ci_lb']:.3f}, {hk['ci_ub']:.3f}]  (宽 {hk['ci_ub']-hk['ci_lb']:.3f})  ← 更宽、更诚实")

# %% [markdown]
# ## 6. 预测区间:下一项研究会落在哪
#
# 合并点估计回答「平均效应是多少」;但在高异质下,读者更该关心的是「**未来一项研究**的真值
# 大概在什么范围」。这就是**预测区间(prediction interval, HTS)**:`θ ± t·√(τ²+SE²)`。它比
# 置信区间宽得多——因为它包含了研究间的真实变异。高 I² 时,预测区间常常跨越「无效」,这是
# 对「别把平均当普适」最直观的提醒。

# %%
sv.tl.meta_random(study, method="REML")  # 回到标准区间供预测区间使用
sv.tl.meta_prediction_interval(study)
pi = study.diagnostics["prediction_interval"]
print(f"95% 预测区间 (log-RR): [{pi['pi_lb']:.3f}, {pi['pi_ub']:.3f}]")
print(f"→ RR 尺度: [{np.exp(pi['pi_lb']):.2f}, {np.exp(pi['pi_ub']):.2f}]")
print(f"未来一项试验的真实 RR 可能从强保护(~{np.exp(pi['pi_lb']):.2f})到几乎无效(~{np.exp(pi['pi_ub']):.2f})——'平均减半'远不是全部故事。")

# %% [markdown]
# ## 7. 发表偏倚:小研究是不是被选择性发表了
#
# 如果「阳性的小研究更容易发表」,漏斗图会不对称。**Egger 回归检验**把标准化效应对精度回归,
# 检验截距是否偏离 0。这里我们既看数字,也画**漏斗图**——每个点是一项研究(x = 效应,
# y = 标准误),对称的漏斗提示没有明显小研究效应。

# %%
sv.tl.egger_test(study)
eg = study.diagnostics["egger"]
print(f"Egger 截距 = {eg['intercept']:.3f}  (p = {eg['pval']:.3f})")
print("p > 0.05:没有强证据表明存在小研究效应/发表偏倚(但研究数少时功效有限,别据此下定论)。")

sv.pl.funnel(study, out="fig22_funnel.png", title="卡介苗试验 · 漏斗图")
Image("fig22_funnel.png")

# %% [markdown]
# ## 8. 森林图:把全部证据画进一张图
#
# 森林图是 meta 分析的标准输出:每项研究一行,方块是点估计(大小 ∝ 权重)、横线是 95% CI;
# 底部红色菱形是合并估计;再下面一条是预测区间。一张图同时呈现「每项研究说了什么」「合并
# 起来说了什么」「异质有多大」。

# %%
sv.pl.meta_forest(study, out="fig22_forest.png",
                  title="卡介苗防结核 · 随机效应 meta 分析(log 风险比)")
Image("fig22_forest.png")

# %% [markdown]
# ## 小结:一条可复现的证据链
#
# 我们用一份真实数据走完了标准 meta 分析,每一步都落在 `StudyState` 上、每个数字都能追到
# 是哪个 `sv.*` 函数、从哪份数据算出来的:
#
# ```
# 2×2 表 ──sv.pp.escalc(RR)──▶ 效应量(yi, vi)
#        ──sv.tl.meta_random(REML, HKSJ)──▶ 合并 log-RR = -0.7145(与 metafor 逐位吻合)
#        ──sv.tl.meta_heterogeneity──▶ I² = 92%(高度异质)
#        ──sv.tl.meta_prediction_interval──▶ 预测区间跨度很大
#        ──sv.tl.egger_test / sv.pl.funnel──▶ 无明显发表偏倚
#        ──sv.pl.meta_forest──▶ 一张图收束全部证据
# ```
#
# **要点**:合并点估计永远和 I²、预测区间一起报告;高异质时用预测区间的语言下结论;小样本
# 开 Knapp-Hartung。下一本([23 · 多层与稳健 meta](23_multilevel_and_robust_meta.ipynb))处理
# 一个更棘手的情形:**每项研究贡献多个相依的效应量**,这时两层随机效应会低估不确定性,需要
# 三层 `rma_mv` 和簇稳健方差。
