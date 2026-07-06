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
# # 多层与生存:HLM 随机效应 + Cox 事件史
#
# **这条分析链讲什么。** 社科数据里有两类结构会让"跑一个普通回归"直接失效:
#
# 1. **嵌套 / 分组结构**——学生嵌在学校里、员工嵌在公司里、受访者嵌在国家里。
#    同一组内的观测**不独立**(共享组级冲击),忽略它会低估标准误、把组级方差
#    误算成个体噪声。对策是**多层线性模型 HLM**(随机截距 / 随机斜率),并用
#    **组内相关系数 ICC** 量化"有多少方差在组间"。
# 2. **删失的时间到事件数据**——我们观测的是"多久之后发生了事件",而且很多个体
#    在观测窗口结束时**还没发生**(右删失)。普通 OLS 会把删失当成"事件已发生在
#    该时点",系统性地扭曲结论。对策是**事件史 / 生存分析**:Cox 比例风险模型
#    估 log 风险比,Kaplan-Meier 估生存曲线。
#
# 这本 notebook 把这两条链都用 `socialverse` 的**注册表契约**跑一遍。两份数据都是
# **参数已知的合成数据**(HLM 真实斜率 = 2.0、真实 ICC ≈ 0.5;Cox 真实 log-HR(x) = 0.8),
# 所以我们可以直接核对"估出来的 = 埋进去的",这是教学 notebook 最硬的自检。
#
# **涉及函数(全部注册在 `sv.registry`,契约机器可读)。**
#
# | 阶段 | 函数 | requires → produces |
# |---|---|---|
# | — | `ds.load_multilevel` / `ds.load_survival` | 合成数据(真参数已知) |
# | `sv.tl` | `multilevel` | `sources.datasets + variables.outcome` → `models.mixedlm`, `diagnostics.variance_components` |
# | `sv.tl` | `survival` | `sources.datasets + variables.outcome` → `models.{cox,km}`, `diagnostics.ph_test` |
# | `sv.pl` | `km_curve` | `models.km` → `artifacts.figures` |
#
# **`StudyState` 会被填哪些槽。**
# `sources`(原始 student×school / 生存表)· `variables`(结果变量名)·
# `models`(`mixedlm` 随机效应模型 / `cox` 比例风险 + `km` 生存曲线)·
# `diagnostics`(方差成分 + ICC / PH 假设检验)· `artifacts`(KM 图)。
#
# **对标的现实 Py/R 包。** 多层这条对标 **`lme4::lmer`** 与 **`brms`**(贝叶斯多层),
# 底层用 `statsmodels.MixedLM`(REML);生存这条对标 **`survival::coxph` + `survminer::ggsurvplot`**
# 与 `lifelines`,底层用 `statsmodels.PHReg`(Cox 偏似然)与 `SurvfuncRight`(KM)。
# `socialverse` 的差异**不在算法**——算法就是这些冠军包的同一套——而在**注册表 grounding**:
# 每个函数登记 `requires/produces`,AI agent 是**查表**规划这条链,而不是**猜** API。

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
matplotlib.use("Agg")  # 无头后端:图存文件,不弹窗
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
import numpy as np

import socialverse as sv
from socialverse import datasets as ds

# 让本 notebook 自绘的图(非 sv.pl 产出的那两张)也能显示中文标签
_CJK = ["PingFang SC", "Hiragino Sans GB", "Songti SC", "STHeiti",
        "Arial Unicode MS", "Noto Sans CJK SC", "Microsoft YaHei"]
_have = {f.name for f in _fm.fontManager.ttflist}
plt.rcParams["font.sans-serif"] = [c for c in _CJK if c in _have] + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 用 ASCII 连字符,负号正常显示

print("socialverse", getattr(sv, "__version__", "(dev)"), "|", os.path.dirname(sv.__file__))

# %% [markdown]
# ---
# ## 第一部分 · 多层模型 HLM:学生嵌在学校里
#
# ### 1.1 数据:两层嵌套,真参数已埋好
#
# `ds.load_multilevel()` 生成 30 所学校 × 每校 20 名学生 = 600 条观测。生成过程是
# 一个**随机截距模型**:
#
# $$ y_{ij} = 1.0 + u_j + 2.0\,x_{ij} + \varepsilon_{ij}, \quad u_j \sim N(0, 1^2),\ \varepsilon_{ij}\sim N(0,1^2) $$
#
# 其中 $u_j$ 是**学校 $j$ 的随机截距**(组级冲击),$x_{ij}$ 是学生级预测变量。
# 真实固定斜率 = **2.0**;真实组间方差 = 残差方差 ≈ 1,所以真实
# **ICC = σ²_组间 / (σ²_组间 + σ²_残差) ≈ 0.5**——恰好一半方差在学校之间。
# 这正是"忽略嵌套结构就会出错"的场景。

# %%
mldf = ds.load_multilevel()
print("形状:", mldf.shape, "| 列:", list(mldf.columns))
print("学校数:", mldf["school"].nunique(), "| 每校学生数:", mldf.groupby("school").size().iloc[0])
print(mldf.head(6).to_string(index=False))

# %% [markdown]
# ### 1.2 装进 StudyState:先满足 `multilevel` 的契约
#
# `sv.tl.multilevel` 的 `requires = {sources: [datasets], variables: [outcome]}`。
# 我们必须先把这两个槽填上——写 `sources.datasets`(原始表)和 `variables.outcome`
# (哪一列是结果变量)。**这一步是 grounding 的关键**:函数不猜"哪列是 y",而是
# 读你在 `variables.outcome` 里声明的名字。

# %%
st = sv.StudyState()
st.write("variables", "outcome", "y")
st.write("sources", "datasets", mldf)
print("已填槽:", st.populated())

# %% [markdown]
# ### 1.3 拟合随机截距模型 → `models.mixedlm` + `diagnostics.variance_components`
#
# `multilevel(state, groups=..., predictors=...)` 用 `statsmodels.MixedLM`(REML)
# 拟合 `y ~ 1 + x` + 学校级随机截距,产出:
#
# - `models.mixedlm`:固定效应系数(含 SE)、组数、是否收敛、估计器;
# - `diagnostics.variance_components`:组间方差、残差方差、**ICC**。
#
# 契约满足后,注册表 wrapper 会自动把这一步记进 `provenance`。

# %%
sv.tl.multilevel(st, groups="school", predictors=["x"])

mm = st.models["mixedlm"]
vc = st.diagnostics["variance_components"]

print("估计器 :", mm["estimator"], "| 收敛:", mm["converged"], "| n:", mm["n"], "| 组数:", mm["n_groups"])
print("\n固定效应 (系数, SE):")
for name, (b, se) in mm["fixed_effects"].items():
    print(f"  {name:<10} {b:+.4f}  (SE {se:.4f})")

print("\n方差成分:")
print(f"  组间方差 σ²_u   = {vc['between_var']:.4f}")
print(f"  残差方差 σ²_e   = {vc['residual_var']:.4f}")
print(f"  ICC            = {vc['icc']:.4f}")

# %% [markdown]
# **核对真参数(教学自检)。** 生成时埋的是 斜率 = 2.0、ICC ≈ 0.5。我们把估计值
# 与真值并排打出来——差得越小,说明这条链算对了。

# %%
slope_hat = mm["fixed_effects"]["x"][0]
icc_hat = vc["icc"]
print(f"{'量':<14}{'真值':>10}{'估计':>12}{'绝对误差':>12}")
print(f"{'固定斜率 x':<14}{2.0:>10.3f}{slope_hat:>12.3f}{abs(slope_hat-2.0):>12.3f}")
print(f"{'ICC':<14}{0.5:>10.3f}{icc_hat:>12.3f}{abs(icc_hat-0.5):>12.3f}")
assert abs(slope_hat - 2.0) < 0.15, "斜率偏离真值太多"
assert abs(icc_hat - 0.5) < 0.15, "ICC 偏离真值太多"
print("\n✅ 斜率与 ICC 都命中真参数。多层结构下的组间方差被正确分离出来。")

# %% [markdown]
# ### 1.4 为什么必须多层:ICC ≈ 0.5 意味着什么
#
# ICC ≈ 0.5 表示**一半的结果方差在学校之间**。如果我们无视嵌套、直接 pool 成一个
# OLS,标准误会被系统性低估(等价于假装 600 个观测都独立,而其实有效样本量远小于 600)。
# 下面画一张组级图:每所学校自己的 `y~x` 拟合线。它们**共享同一个斜率(≈2)但截距上下
# 平移**——这就是"随机截距"的几何直觉,也是 HLM 相对 OLS 的价值所在。

# %%
fig, ax = plt.subplots(figsize=(7.0, 4.6))
cmap = plt.get_cmap("viridis")
schools = sorted(mldf["school"].unique())
for i, g in enumerate(schools):
    sub = mldf[mldf["school"] == g]
    # 每校自己的 OLS 拟合线(展示截距的组间漂移)
    b1, b0 = np.polyfit(sub["x"].to_numpy(), sub["y"].to_numpy(), 1)
    xs = np.linspace(sub["x"].min(), sub["x"].max(), 20)
    ax.plot(xs, b0 + b1 * xs, color=cmap(i / max(len(schools) - 1, 1)),
            alpha=0.55, linewidth=1.0, zorder=2)
# 叠加固定效应(总体)线:用 HLM 的固定截距 + 固定斜率
b0_fix = mm["fixed_effects"]["Intercept"][0]
b1_fix = mm["fixed_effects"]["x"][0]
xs = np.linspace(mldf["x"].min(), mldf["x"].max(), 50)
ax.plot(xs, b0_fix + b1_fix * xs, color="crimson", linewidth=3.0, zorder=5,
        label=f"HLM 固定效应线 (斜率 {b1_fix:.2f})")
ax.set_xlabel("学生级预测变量 x")
ax.set_ylabel("结果 y")
ax.set_title(f"30 所学校各自的回归线:随机截距平移(ICC ≈ {icc_hat:.2f})")
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig("fig_hlm_schools.png", dpi=150)
plt.close(fig)
print("已存 fig_hlm_schools.png")

# %% [markdown]
# ![学校随机截距](fig_hlm_schools.png)
#
# 细线是每所学校自己的拟合线(彩虹色按学校编号),粗红线是 HLM 估出的固定效应线。
# 学校线大体**平行**(斜率都在 2 附近)但**上下错开**——正是随机截距 $u_j$ 的体现。

# %% [markdown]
# ---
# ## 第二部分 · 生存 / 事件史:Cox 比例风险 + Kaplan-Meier
#
# ### 2.1 数据:右删失的时间到事件
#
# `ds.load_survival(beta=0.8)` 从一个指数比例风险模型生成 400 条记录:
#
# $$ \lambda_i(t) = 0.1 \cdot \exp(0.8\,x_i + 0.5\,\text{group}_i) $$
#
# 事件时间 $t^{\text{event}}$ 与独立的删失时间 $t^{\text{cens}}$ 取较小者作为观测
# `time`,`event = 1` 当且仅当事件先发生。真实 **log-HR(x) = 0.8**(即 HR ≈ 2.23:
# `x` 每升 1,风险翻一倍多)。注意有相当比例是删失的——这正是不能用 OLS 的原因。

# %%
survdf = ds.load_survival(beta=0.8)
n_evt = int(survdf["event"].sum())
print("形状:", survdf.shape, "| 列:", list(survdf.columns))
print(f"事件数 = {n_evt} / {len(survdf)}  (删失 {len(survdf)-n_evt} 条, "
      f"删失率 {1 - n_evt/len(survdf):.0%})")
print(survdf.head(6).to_string(index=False))

# %% [markdown]
# ### 2.2 契约演示:不填槽就调用 → `RegistryError`(注册表在拦截你)
#
# 这是 `socialverse` 的核心卖点,值得**故意触发一次**。`survival` 的
# `requires = {sources:[datasets], variables:[outcome]}`。如果我们对一个空
# `StudyState` 直接调用,注册表 wrapper 会在函数体运行**之前**抛 `RegistryError`,
# 并告诉你**缺哪个槽、该由谁产出**——这就是 AI agent 规划分析链时"查而非猜"的机制。

# %%
try:
    empty = sv.StudyState()
    sv.tl.survival(empty)
except sv.RegistryError as e:
    print("按预期抛出 RegistryError:\n")
    print(e)

# %% [markdown]
# 注意报错信息里 `sources.datasets (produced by: ingest)` 和
# `variables.outcome (user-supplied input)`——注册表不仅说"缺什么",还说
# "**从哪来**"。这正是 `registry.resolve_plan(...)` 能自动补链的依据。
#
# ### 2.3 正确填槽并拟合 → `models.{cox,km}` + `diagnostics.ph_test`
#
# 现在把契约补齐:`variables.outcome = "time"`(时长列)+ `sources.datasets`。
# `survival(state, time=, event=, covariates=)` 一次产出三样东西:
#
# - `models.cox`:每个协变量的 log-HR、SE、p 值,及 HR = exp(log-HR);
# - `models.km`:总体 + 分组的 Kaplan-Meier 生存函数(含中位生存时间);
# - `diagnostics.ph_test`:比例风险假设的 Grambsch-Therneau / Schoenfeld 检验。

# %%
st2 = sv.StudyState()
st2.write("variables", "outcome", "time")
st2.write("sources", "datasets", survdf)
sv.tl.survival(st2, time="time", event="event", covariates=["x", "group"])

cox = st2.models["cox"]
print("估计器:", cox["estimator"], "| n:", cox["n"], "| 事件数:", cox["n_events"])
print("\n协变量        log-HR      SE       HR        p")
for name, (b, se, p) in cox["log_hr"].items():
    print(f"  {name:<10} {b:+.4f}   {se:.4f}   {cox['hr'][name]:.3f}   {p:.2e}")

# %% [markdown]
# **核对真参数。** 埋进去的是 log-HR(x) = 0.8。看估计值离得多近:

# %%
loghr_x = cox["log_hr"]["x"][0]
print(f"{'量':<16}{'真值':>10}{'估计':>12}{'绝对误差':>12}")
print(f"{'log-HR(x)':<16}{0.8:>10.3f}{loghr_x:>12.3f}{abs(loghr_x-0.8):>12.3f}")
print(f"{'HR(x)=exp(·)':<16}{np.exp(0.8):>10.3f}{cox['hr']['x']:>12.3f}"
      f"{abs(cox['hr']['x']-np.exp(0.8)):>12.3f}")
assert abs(loghr_x - 0.8) < 0.15, "log-HR 偏离真值太多"
print("\n✅ Cox 偏似然在 27% 删失下仍然复原了真实 log-HR。")

# %% [markdown]
# ### 2.4 比例风险假设检验:Cox 到底能不能用
#
# Cox 模型的**前提**是比例风险(PH):各协变量的风险比不随时间变化。这是 Cox 的
# "平行趋势"——**前提不过,系数就不能当因果 / 稳定效应解读**。`survival` 自动跑了
# Grambsch-Therneau 检验(等价 `survival::cox.zph`):对每个协变量做 Schoenfeld 残差
# 与时间的相关性检验,再给一个全局检验。`p > 0.05` = PH 成立。

# %%
ph = st2.diagnostics["ph_test"]
print("方法:", ph["method"])
print(f"全局 PH 检验:  χ² = {ph['global_chi2']:.3f},  p = {ph['global_p']:.3f}  → 判定: {ph['verdict']}")
print("\n逐协变量:")
for name, d in ph["per_covariate"].items():
    print(f"  {name:<8} rho={d['rho']:+.4f}  χ²={d['chi2']:.3f}  p={d['p']:.3f}")
print("\n结论:", ph["note"])
assert ph["verdict"] == "pass", "PH 假设未通过,Cox 系数需谨慎"
print("✅ PH 假设成立 → Cox 的 log-HR 可以作为稳定的风险比解读。")

# %% [markdown]
# ### 2.5 Kaplan-Meier 生存曲线 → `sv.pl.km_curve`
#
# `km_curve` 的 `requires = {models: [km]}`——上一步 `survival` 刚好产出了它,契约满足。
# 它把总体与分组的 KM 生存函数画成右连续阶梯图(标准 Kaplan-Meier 呈现),对标
# `survminer::ggsurvplot` / `lifelines.KaplanMeierFitter.plot`,产出写到 `artifacts.figures`。

# %%
km = st2.models["km"]
print("KM 分组列:", km["group_col"], "| 分组:", sorted(km["by_group"].keys()))
print(f"总体中位生存时间: {km['overall']['median']:.3f}")
for g, curve in sorted(km["by_group"].items()):
    print(f"  组 {g}: 中位生存时间 = {curve['median']:.3f}  (n={curve['n']})")

sv.pl.km_curve(st2, out="fig_km.png",
               title="Kaplan-Meier 生存曲线(按 group 分层)")
print("\n图已登记到 artifacts.figures:", st2.artifacts["figures"]["km"])

# %% [markdown]
# ![KM 生存曲线](fig_km.png)
#
# 两条实线是 `group=0/1` 的分层生存曲线,灰色虚线是总体参考。`group=1` 因为多了
# `+0.5` 的线性项(更高风险),生存曲线**掉得更快**、中位生存时间更短——和数据生成
# 机制一致。

# %% [markdown]
# ### 2.6 补一张风险比森林图(直接用 matplotlib)
#
# `sv.pl.forest` 的契约要求 `models.did`(为 DID 设计),不适用于 Cox;所以这里
# 直接用 `models.cox` 的 log-HR ± 1.96·SE 手画一张 HR 森林图,在对数刻度上展示
# 每个协变量的风险比与 95% 置信区间。HR=1(虚线)是"无效应"参考。

# %%
names = list(cox["log_hr"].keys())
b = np.array([cox["log_hr"][k][0] for k in names])
se = np.array([cox["log_hr"][k][1] for k in names])
hr = np.exp(b)
lo = np.exp(b - 1.96 * se)
hi = np.exp(b + 1.96 * se)

fig, ax = plt.subplots(figsize=(6.6, 2.6))
ypos = np.arange(len(names))[::-1]
ax.errorbar(hr, ypos, xerr=[hr - lo, hi - hr], fmt="o", color="#2c3e70",
            capsize=4, markersize=7, linewidth=1.6, zorder=3)
ax.axvline(1.0, color="0.5", linestyle="--", linewidth=1.0, zorder=1)
ax.set_yticks(ypos)
ax.set_yticklabels([f"{k}" for k in names])
ax.set_xscale("log")
ax.set_xlabel("风险比 HR = exp(log-HR)  (对数刻度)")
ax.set_title("Cox 比例风险:HR 与 95% CI")
for x_, y_ in zip(hr, ypos):
    ax.annotate(f"{x_:.2f}", (x_, y_), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=9)
ax.set_ylim(-0.6, len(names) - 0.4)
fig.tight_layout()
fig.savefig("fig_cox_forest.png", dpi=150)
plt.close(fig)
print("已存 fig_cox_forest.png")

# %% [markdown]
# ![Cox 森林图](fig_cox_forest.png)
#
# `x` 的 HR ≈ 2.3(真值 exp(0.8)=2.23),置信区间整段在 1 的右侧 → 显著提升风险;
# `group` 的 HR ≈ 1.8 同样显著。两个 CI 都不跨过 HR=1 的虚线。

# %% [markdown]
# ---
# ## 收尾 · Provenance:注册表把整条链记了下来
#
# 两个 `StudyState` 各自累积了一条可审计的 provenance。它不是我们手写的日志——是
# 注册表 wrapper 在每个契约满足、函数成功执行后**自动追加**的,每一步都带
# `requires → produces`。这就是 `socialverse` 相对裸调 `statsmodels` 的差异所在。

# %%
print("=== 多层链 StudyState ===")
print(st.summary())
print("\n步骤明细:")
for p in st.provenance:
    print(f"  [{p['step']}] {p['function'].split('.')[-1]}"
          f"  requires={dict(p['requires'])}  produces={dict(p['produces'])}")

print("\n=== 生存链 StudyState ===")
print(st2.summary())
print("\n步骤明细:")
for p in st2.provenance:
    print(f"  [{p['step']}] {p['function'].split('.')[-1]}"
          f"  requires={dict(p['requires'])}  produces={dict(p['produces'])}")

# %% [markdown]
# **一句话总结 `socialverse` 的差异。** 算法和 `lme4` / `survival` / `lifelines`
# 是同一套(REML 混合模型、Cox 偏似然、Kaplan-Meier),但每个函数都登记了机器可读的
# `requires/produces` 契约:
#
# - **grounding(查而非猜):** 缺 `sources.datasets` / `variables.outcome` 时,
#   `survival` 在跑之前就抛 `RegistryError` 并指出"缺什么、由谁产出",AI agent
#   据此 `resolve_plan` 自动补链,而不是瞎编 API;
# - **可核验(复原真参数):** 合成数据的真参数(斜率 2.0、ICC 0.5、log-HR 0.8)
#   被逐一命中,证明这条链不是"看起来跑通"而是"算对了";
# - **可审计(自动 provenance):** 整条 HLM / 生存链的 `requires→produces` 被完整
#   记录,任何一步产出都能追溯到它的契约来源。
