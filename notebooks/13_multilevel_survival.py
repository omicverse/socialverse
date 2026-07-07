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
# # 嵌套数据与时间到事件:多层模型与生存分析
#
# 社会科学的数据常常有两种结构,会让「直接跑一个普通回归」给出错误的答案。第一种是**嵌套结构**:学生嵌在学校里、员工嵌在公司里、受访者嵌在国家里。同一所学校的学生共享校长、师资、生源这些组级冲击,他们的结果并不相互独立;如果无视这层结构、把 600 个学生当成 600 个独立观测直接 pool 成 OLS,标准误会被系统性低估,组间的方差还会被误当成个体噪声。对策是**多层线性模型(HLM)**——给每个组一个随机截距(必要时再给随机斜率),并用**组内相关系数 ICC** 量化「到底有多少方差落在组之间」。
#
# 第二种是**删失的时间到事件数据**:我们关心的是「多久之后发生了某件事」——失业多久后再就业、企业成立多久后倒闭、病人确诊多久后复发。麻烦在于观测窗口结束时,很多个体**事件还没发生**(右删失),我们只知道「到目前为止至少活了这么久」。把删失时点当成事件时点直接做 OLS,会系统性地压低估计。对策是**事件史 / 生存分析**:用 Cox 比例风险模型估各协变量的 log 风险比(log-HR),用 Kaplan-Meier 估分组生存曲线。Cox 有自己的关键前提——**比例风险(PH)**:各协变量的风险比不随时间漂移;这个前提要专门检验,不过关的话系数就不能当稳定效应来解读。
#
# 这本 notebook 把这两条链各走一遍。两份数据都是**参数已知的合成数据**,方便我们把「估出来的」和「埋进去的」并排核对——这是教学 notebook 最硬的一道自检。多层这一段的真实固定斜率是 2.0、真实 ICC ≈ 0.5;生存这一段的真实 log-HR(x) = 0.8(即风险比 HR ≈ 2.23,x 每升一个单位、风险翻一倍多)。我们用 `socialverse` 完成全流程,它把每种方法登记进一张函数注册表,底层算法就是 `statsmodels` 那一套冠军实现——多层对标 R 的 `lme4::lmer`,生存对标 `survival::coxph` 与 Python 的 `lifelines`。

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

import socialverse as sv
from socialverse import datasets as ds

# 让本 notebook 自绘的图也能显示中文标签
_CJK = ["PingFang SC", "Hiragino Sans GB", "Songti SC", "STHeiti",
        "Arial Unicode MS", "Noto Sans CJK SC", "Microsoft YaHei"]
_have = {f.name for f in _fm.fontManager.ttflist}
plt.rcParams["font.sans-serif"] = [c for c in _CJK if c in _have] + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False  # 用 ASCII 连字符,负号正常显示

# %% [markdown]
# # 第一部分 · 多层模型:学生嵌在学校里
#
# ## 载入嵌套数据
#
# 我们用一个内置的两层合成数据:30 所学校 × 每校 20 名学生 = 600 条观测。生成机制是一个随机截距模型 `y = 1.0 + u_school + 2.0·x + 噪声`,其中 `u_school ~ N(0, 1²)` 是每所学校的随机截距(组级冲击),`x` 是学生级预测变量。真实固定斜率是 2.0,组间方差和残差方差都约等于 1,所以真实 ICC ≈ 0.5——恰好一半的结果方差落在学校之间。这正是「无视嵌套就会出错」的典型场景。
#
# 数据是长格式,每行一名学生:`school` 是组、`student` 是组内编号、`x` 是预测变量、`y` 是结果。

# %%
mldf = ds.load_multilevel()
print("形状:", mldf.shape, "| 学校数:", mldf["school"].nunique(),
      "| 每校学生数:", mldf.groupby("school").size().iloc[0])
mldf.head()

# %% [markdown]
# ## 装进研究状态
#
# 和别的分析一样,第一步不是跑模型,而是把数据和「哪一列是结果变量」讲清楚。`sv.pp.ingest` 把这张表登记进研究状态,`variables.outcome` 声明结果列是 `y`。后面的 `multilevel` 会从这里读结果变量的名字,不用再传一遍。

# %%
st = sv.StudyState()
sv.pp.ingest(st, data=mldf, name="students_in_schools")
st.write("variables", "outcome", "y")   # 结果变量是 y
st.populated()

# %% [markdown]
# ## 拟合随机截距模型
#
# `multilevel` 用 `statsmodels.MixedLM`(REML)拟合 `y ~ 1 + x` 加上学校级的随机截距。`groups` 指定分组列,`predictors` 给固定效应协变量。它一次产出两样东西:固定效应系数(含标准误、组数、是否收敛),以及方差成分——组间方差、残差方差,还有由二者算出的 ICC。

# %%
sv.tl.multilevel(st, groups="school", predictors=["x"])

mm = st.models["mixedlm"]
vc = st.diagnostics["variance_components"]

print("估计器:", mm["estimator"], "| 收敛:", mm["converged"],
      "| n:", mm["n"], "| 组数:", mm["n_groups"])
print("\n固定效应 (系数, SE):")
for name, (b, se) in mm["fixed_effects"].items():
    print(f"  {name:<10} {b:+.4f}  (SE {se:.4f})")
print("\n方差成分:")
print(f"  组间方差 σ²_u = {vc['between_var']:.4f}")
print(f"  残差方差 σ²_e = {vc['residual_var']:.4f}")
print(f"  ICC          = {vc['icc']:.4f}")

# %% [markdown]
# 怎么读:`x` 的固定斜率估到 **+1.99**,几乎正中真值 2.0;截距约 1.2。方差成分显示组间方差(≈1.04)和残差方差(≈0.95)大体相当,ICC ≈ **0.52**,意味着约一半的结果方差在学校之间——这恰恰是数据里埋的结构。
#
# 因为这是合成数据、真参数已知,我们可以直接把估计和真值并排核对。差得越小,说明这条链算得越对。

# %%
slope_hat = mm["fixed_effects"]["x"][0]
icc_hat = vc["icc"]
pd.DataFrame([
    {"量": "固定斜率 x", "真值": 2.0, "估计": round(slope_hat, 3), "绝对误差": round(abs(slope_hat - 2.0), 3)},
    {"量": "ICC",       "真值": 0.5, "估计": round(icc_hat, 3),   "绝对误差": round(abs(icc_hat - 0.5), 3)},
])

# %% [markdown]
# 斜率和 ICC 都落在真值附近——多层结构下的组间方差被正确分离了出来。

# %% [markdown]
# ## 可视化:随机截距的几何直觉
#
# ICC ≈ 0.5 到底长什么样?下面给每所学校单独拟合一条 `y~x` 直线(细彩线),再叠上 HLM 估出的固定效应线(粗红线)。如果数据真是随机截距结构,那么各校的线应该**斜率接近、截距上下平移**——斜率共享(都在 2 附近),截距各自漂移,就是随机截距 `u_school` 的体现,也是 HLM 相对一条 pooled OLS 直线的价值所在。

# %%
fig, ax = plt.subplots(figsize=(7.0, 4.6))
cmap = plt.get_cmap("viridis")
schools = sorted(mldf["school"].unique())
for i, g in enumerate(schools):
    sub = mldf[mldf["school"] == g]
    b1, b0 = np.polyfit(sub["x"].to_numpy(), sub["y"].to_numpy(), 1)  # 每校自己的 OLS 线
    xs = np.linspace(sub["x"].min(), sub["x"].max(), 20)
    ax.plot(xs, b0 + b1 * xs, color=cmap(i / max(len(schools) - 1, 1)),
            alpha=0.55, linewidth=1.0, zorder=2)
# 叠加 HLM 固定效应线:固定截距 + 固定斜率
b0_fix = mm["fixed_effects"]["Intercept"][0]
b1_fix = mm["fixed_effects"]["x"][0]
xs = np.linspace(mldf["x"].min(), mldf["x"].max(), 50)
ax.plot(xs, b0_fix + b1_fix * xs, color="crimson", linewidth=3.0, zorder=5,
        label=f"HLM 固定效应线 (斜率 {b1_fix:.2f})")
ax.set_xlabel("学生级预测变量 x")
ax.set_ylabel("结果 y")
ax.set_title(f"30 所学校各自的回归线:随机截距上下平移(ICC ≈ {icc_hat:.2f})")
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig("fig_hlm_schools.png", dpi=150)
plt.close(fig)
print("已存 fig_hlm_schools.png")

# %% [markdown]
# ![学校随机截距](fig_hlm_schools.png)
#
# 细线是每所学校自己的拟合线(按学校编号上色),粗红线是 HLM 的固定效应线。各校的线大体**平行**(斜率都在 2 附近)却**上下错开**——这就是随机截距 `u_school` 的模样。

# %% [markdown]
# # 第二部分 · 生存分析:时间到事件与右删失
#
# ## 载入生存数据
#
# `ds.load_survival(beta=0.8)` 从一个指数比例风险模型生成 400 条记录:风险率 `λ(t) = 0.1·exp(0.8·x + 0.5·group)`。事件时间和一个独立的删失时间取较小者作为观测到的 `time`,`event = 1` 当且仅当事件先于删失发生。真实 **log-HR(x) = 0.8**(即 HR ≈ 2.23:`x` 每升一个单位,风险翻一倍多)。
#
# 数据每行一个个体:`time` 是观测到的时长、`event` 是事件指示(1=观测到、0=删失)、`x` 和 `group` 是协变量。注意有相当比例是删失的——正是这一点让 OLS 失效、非用生存模型不可。

# %%
survdf = ds.load_survival(beta=0.8)
n_evt = int(survdf["event"].sum())
print("形状:", survdf.shape,
      f"| 事件数 = {n_evt} / {len(survdf)}  "
      f"(删失 {len(survdf) - n_evt} 条,删失率 {1 - n_evt / len(survdf):.0%})")
survdf.head()

# %% [markdown]
# ## 拟合 Cox 模型
#
# 把这份数据装进一个新的研究状态,声明 `time` 是时长列,然后调 `survival`:`time` 指定时长、`event` 指定事件指示、`covariates` 指定协变量。它一次产出三样东西——Cox 模型的 log-HR(用 `statsmodels.PHReg`、Breslow 处理并列)、Kaplan-Meier 生存曲线,以及比例风险假设检验。先看 Cox 系数。

# %%
st2 = sv.StudyState()
sv.pp.ingest(st2, data=survdf, name="time_to_event")
st2.write("variables", "outcome", "time")   # 时长列
sv.tl.survival(st2, time="time", event="event", covariates=["x", "group"])

cox = st2.models["cox"]
print("估计器:", cox["estimator"], "| n:", cox["n"], "| 事件数:", cox["n_events"])
print("\n协变量        log-HR      SE       HR        p")
for name, (b, se, p) in cox["log_hr"].items():
    print(f"  {name:<10} {b:+.4f}   {se:.4f}   {cox['hr'][name]:.3f}   {p:.2e}")

# %% [markdown]
# 怎么读:`x` 的 log-HR ≈ **0.83**(真值 0.8),对应 HR ≈ **2.30**——`x` 每升一个单位,事件风险涨到约 2.3 倍;`group` 的 HR ≈ 1.79,也显著。两个 p 值都远小于 0.05。把 `x` 的估计和真值并排核对一下:

# %%
loghr_x = cox["log_hr"]["x"][0]
pd.DataFrame([
    {"量": "log-HR(x)",     "真值": round(0.8, 3),         "估计": round(loghr_x, 3),        "绝对误差": round(abs(loghr_x - 0.8), 3)},
    {"量": "HR(x)=exp(·)",  "真值": round(float(np.exp(0.8)), 3), "估计": round(cox["hr"]["x"], 3), "绝对误差": round(abs(cox["hr"]["x"] - np.exp(0.8)), 3)},
])

# %% [markdown]
# 在近 28% 的删失下,Cox 偏似然仍然复原了真实的 log-HR——这正是它相对 OLS 的价值:删失的观测没有被丢弃、也没有被当成已发生,而是以「至少活到这个时点」的方式进入风险集。

# %% [markdown]
# ## 检验比例风险假设
#
# Cox 系数能不能当作稳定的风险比来解读,取决于**比例风险(PH)**前提:各协变量的风险比不随时间漂移。这是 Cox 的「平行趋势」——前提不过,系数就不该被当成一个时间上稳定的效应。`survival` 自动跑了 Grambsch-Therneau 检验(等价于 R 的 `survival::cox.zph`):对每个协变量检验其 Schoenfeld 残差是否与事件时间相关,再汇总成一个全局检验。`p > 0.05` 表示不拒绝 PH,前提站得住。

# %%
ph = st2.diagnostics["ph_test"]
print("方法:", ph["method"])
print(f"全局 PH 检验: χ² = {ph['global_chi2']:.3f}, p = {ph['global_p']:.3f} → 判定: {ph['verdict']}")
print("\n逐协变量:")
for name, d in ph["per_covariate"].items():
    print(f"  {name:<8} rho={d['rho']:+.4f}  χ²={d['chi2']:.3f}  p={d['p']:.3f}")

# %% [markdown]
# 全局检验的 `p ≈ 0.65`,远大于 0.05——不拒绝比例风险,逐协变量也都不显著。前提成立,Cox 的 log-HR 可以作为稳定的风险比解读。

# %% [markdown]
# ## 可视化:Kaplan-Meier 生存曲线
#
# 先看 KM 估出的中位生存时间:总体的、以及按 `group` 分层的。因为 `group=1` 在风险率里多了 `+0.5` 的项(风险更高),它的生存曲线应该掉得更快、中位生存时间更短。

# %%
km = st2.models["km"]
print("KM 分组列:", km["group_col"], "| 分组:", sorted(km["by_group"].keys()))
print(f"总体中位生存时间: {km['overall']['median']:.3f}")
for g, curve in sorted(km["by_group"].items()):
    print(f"  组 {g}: 中位生存时间 = {curve['median']:.3f}  (n={curve['n']})")

# %% [markdown]
# `sv.pl.km_curve` 直接从研究状态里读 KM 曲线出图,画成右连续的阶梯图(标准 Kaplan-Meier 呈现),对标 `survminer::ggsurvplot` 与 `lifelines.KaplanMeierFitter.plot`。

# %%
sv.pl.km_curve(st2, out="fig_km.png", title="Kaplan-Meier 生存曲线(按 group 分层)")
print("图已登记:", st2.artifacts["figures"]["km"])

# %% [markdown]
# ![KM 生存曲线](fig_km.png)
#
# 两条实线是 `group=0/1` 的分层生存曲线,灰色虚线是总体参考。`group=1`(高风险)的曲线掉得更快、中位生存时间(≈3.7)明显短于 `group=0`(≈6.7)——和数据生成机制一致。

# %% [markdown]
# ## 可视化:风险比森林图
#
# 最后把两个协变量的风险比画成森林图,在对数刻度上展示 HR 与 95% 置信区间。`HR=1`(虚线)是「无效应」参考:置信区间整段落在 1 右侧,就说明该协变量显著提升风险。

# %%
names = list(cox["log_hr"].keys())
b = np.array([cox["log_hr"][k][0] for k in names])
se = np.array([cox["log_hr"][k][1] for k in names])
hr = np.exp(b)
lo = np.exp(b - 1.96 * se)   # 95% CI 下界
hi = np.exp(b + 1.96 * se)   # 95% CI 上界

fig, ax = plt.subplots(figsize=(6.6, 2.6))
ypos = np.arange(len(names))[::-1]
ax.errorbar(hr, ypos, xerr=[hr - lo, hi - hr], fmt="o", color="#2c3e70",
            capsize=4, markersize=7, linewidth=1.6, zorder=3)
ax.axvline(1.0, color="0.5", linestyle="--", linewidth=1.0, zorder=1)  # HR=1 无效应参考
ax.set_yticks(ypos)
ax.set_yticklabels(list(names))
ax.set_xscale("log")   # 风险比在对数刻度上对称
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
# `x` 的 HR ≈ 2.3(真值 exp(0.8)=2.23),`group` 的 HR ≈ 1.8,两个置信区间都整段落在 `HR=1` 虚线右侧 → 都显著提升风险。

# %% [markdown]
# ## 可复现的证据链
#
# 两条链各自跑在一个 `StudyState` 上,研究状态里自动积累了一份 provenance 账本:每一步用了哪个函数、消费了什么槽、产出了什么槽。这份账本不是我们手写的日志,而是每步成功执行后自动追加的——在社会科学里,「结论从哪一步、哪份数据来」往往和结论本身同等重要。

# %%
print("=== 多层链 ===")
print(st.summary())
print("\n=== 生存链 ===")
print(st2.summary())

# %% [markdown]
# ## 小结
#
# 我们走完了两条针对特殊数据结构的分析链。多层这条对标 R 的 `lme4::lmer` 与 Python 的 `statsmodels.MixedLM`:用随机截距吸收组级冲击,用 ICC 量化组间方差;生存这条对标 `survival::coxph` + `survminer` 与 `lifelines`:用 Cox 偏似然在删失下估风险比,用 Kaplan-Meier 画生存曲线,并用比例风险检验守住前提。
#
# 与裸调 `statsmodels` 相比,`socialverse` 多给了两样东西:一是把「哪列是结果变量、这一步需要哪些前置」写成机器可读的契约,让分析可以被查表规划而不是靠猜;二是一份贯穿始终、可追溯的证据链。下一本教程 [14_spatial_analysis](14_spatial_analysis.ipynb) 转向空间数据——当观测在地理上相邻、彼此不独立时,该怎么建模。
