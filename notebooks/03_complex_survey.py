# %% [markdown]
# # 从一份复杂抽样调查里，得到面向总体的诚实推断
#
# 几乎没有一份真实的社会调查是简单随机抽样。民调、健康调查、家户面板在设计时就带着三样东西:**抽样权重**(每个受访者代表总体里多少人)、**分层**(先按地区/年龄把总体切块再在层内抽样)、**初级抽样单元 PSU**(实际是先抽村庄/社区这样的聚类,再在其中抽人)。如果你无视这套设计、把它当独立同分布的样本直接跑 OLS,会同时踩两个坑:点估计会**偏**(不加权等于在描述"样本"而不是"目标总体"),标准误会**假小**(忽略同一 PSU 内受访者的相关,置信区间过窄、p 值虚假显著)。
#
# 应对办法叫**设计基础推断(design-based inference)**:把抽样设计显式声明出来,让权重进入点估计、让 PSU 进入方差公式,从而得到一个面向**目标总体**、且区间诚实的估计。这正是 R 的 `survey` 包和 Python 的 `samplics` 在做的事——`svydesign()` 声明设计、`svyglm()` 做加权回归、方差用 Taylor 线性化或重抽样。除了估计,一份严肃的调查在**采集之前**还要先回答两个测量问题:量表可靠吗(内部一致性信度 Cronbach's α)、样本够大吗(功效/样本量,而且聚类会让有效样本缩水,必须按设计效应 DEFF 往上抬 n)。
#
# 这条链就按调查的真实工序走一遍:载入数据 → 声明抽样设计 → 采集前的测量与功效设计 → 设计加权回归 → 加权与朴素对照 → 出图 → 小结。我们用 `socialverse` 完成全程,它把这些方法薄封在 `statsmodels` 与 NumPy 之上,方法本身照搬社科惯例,顺便在每一步留下一份可复现的证据链——最后会看到。
#
# 数据是一份内置的合成调查:300 个受访者、6 道 Likert 题项(item1..item6,由同一个潜变量驱动,所以信度天然很高),外加设计列 `weight / strata / psu`、一个二元暴露 `exposure` 和一个二元结局 `outcome`。

# %%
import matplotlib
matplotlib.use("Agg")  # 无显示环境:图直接写文件

import socialverse as sv
from socialverse import datasets as ds

# %% [markdown]
# ## 载入调查数据
#
# 先把这份表读进来看一眼。每行是一个受访者:前 6 列是量表题项(1–5 分),`weight` 是抽样权重,`strata` 是分层编号,`psu` 是初级抽样单元编号,`exposure` 与 `outcome` 是我们关心的暴露和结局。

# %%
df = ds.load_survey()
print("调查维度:", df.shape)  # (300, 11)
df.head()

# %% [markdown]
# ## 声明目标量与抽样设计
#
# 调查分析的起点不是数据,而是**你想估什么**。我们要估的是**人群占比 / 患病率(prevalence)**——某个二元结局在目标总体里的水平,以及暴露与它的关联。把这个目标写进研究状态,后续的测量设计与估计都从这里读。

# %%
st = sv.StudyState()
st.write("estimand", "target", "prevalence")  # 目标量:面向总体的占比/关联,而非样本描述

sv.pp.ingest(st, data=df)  # 把这份调查表登记进研究状态
st.sources["dataset_meta"]

# %% [markdown]
# 接着是复杂抽样与"随便跑个回归"的**分水岭**:把三个设计事实显式声明出来。`weights='weight'` 让估计面向目标总体(纠正不等概率抽样);`strata='strata'` 记录分层,方差要按层算;`psu='psu'` 是聚类单元,同一 PSU 内的观测相关,标准误必须按它聚类,否则区间假窄。这三列的名字被记进研究状态,下游的加权回归就靠读它们干活。

# %%
sv.pp.declare_design(
    st,
    weights="weight",   # 抽样权重列 → 点估计面向总体
    strata="strata",    # 分层列 → 方差按层
    psu="psu",          # 初级抽样单元(聚类)列 → 标准误按 PSU 聚类
)
dict(st.design)

# %% [markdown]
# ## 采集前:量表信度与样本量
#
# 在"估"之前先问两件事:**量表可靠吗?**(6 道题是不是在一致地测同一个东西)**样本够大吗?**(要达到目标功效需要多少人,且必须为聚类设计做 DEFF 膨胀)。`design_survey` 一步算完这两样:用题项矩阵算 Cronbach's α,用正态近似算所需样本量并按设计效应 DEFF 抬高。
#
# Cronbach's α 是内部一致性信度,取值 0–1,越高说明各题越像在测同一个潜构念;经验上 α ≥ 0.9 属"优秀"。样本量用 n ≈ ((z_{α/2}+z_β)/效应量)² × DEFF:DEFF(设计效应)大于 1 反映聚类让有效样本缩水,所以要按它把 n 往上抬。

# %%
items = df[[c for c in df.columns if c.startswith("item")]]  # 6 道 Likert 题项作响应矩阵
sv.tl.design_survey(
    st,
    items=items,
    effect_size=0.2,  # 标准化效应(小效应)
    deff=1.5,         # 设计效应:聚类使方差膨胀 50%
    alpha=0.05,
    power=0.8,
)

rel = st.diagnostics["reliability"]
pow_ = st.diagnostics["power"]
print(f"Cronbach α = {rel['alpha']:.3f}   (k = {rel['k']} 题, n = {rel['n_respondents']} 人)")
print(f"所需样本量 n = {pow_['n_required']}   (已按 DEFF = {pow_['deff']} 膨胀, 每组 {pow_['n_per_group']})")

# %% [markdown]
# 读数:**α ≈ 0.932**——6 道题内部一致性优秀,量表可用。功效计算给出**所需 n ≈ 295**(已按 DEFF=1.5 膨胀),我们手上正好 300,勉强达标。这正是聚类设计里最容易被忽视的一课:看起来 300 个样本很宽裕,一旦把聚类算进去,有效信息其实缩水到刚够用。如果把 `deff` 调回 1.0(假装简单随机),所需 n 会明显更小——**忽视聚类会让你误以为样本充足**。

# %% [markdown]
# ## 设计加权回归:整条链的心脏
#
# 现在做采集后的估计。这不是普通 OLS,而是三件事同时发生:一是**加权**,用声明的权重做 WLS,估计面向目标总体;二是**聚类稳健方差**,因为声明了 PSU,`statsmodels` 用 `cov_type="cluster"` 按 PSU 聚类算标准误,给出诚实(通常更宽)的区间;三是**朴素对照**,同时跑一遍未加权 OLS,把加权与不加权的系数并排收好,方便判断权重到底重不重要。
#
# 估计要求先声明结局变量,所以我们把二元结局 `outcome` 写进研究状态,再把暴露 `exposure` 作为自变量传入,拟合 `outcome ~ exposure`。

# %%
st.write("variables", "outcome", "outcome")  # 声明二元结局列

sv.tl.survey_estimate(st, exposure="exposure")  # WLS + PSU 聚类稳健:outcome ~ exposure

m = st.models["weighted_reg"]
print(f"exposure 系数 = {m['coef']['exposure']:.3f}")
print(f"95% CI        = [{m['ci']['exposure'][0]:.3f}, {m['ci']['exposure'][1]:.3f}]")
print(f"SE            = {m['se']['exposure']:.3f}")
print(f"方差类型      = {m['cov_type']}   (n = {m['n']})")

# %% [markdown]
# 读数:`exposure` 的设计加权系数 ≈ **0.205**,95% CI ≈ **[0.116, 0.295]**,区间不含 0,暴露与结局正相关。关键在 `cov_type` 字段——它是 **cluster**,说明这个区间是按 PSU 聚类算出来的,而不是天真地假设独立同分布。这一个字段本身就是证据:它记录了区间的诚实度来自哪里。

# %% [markdown]
# ## 加权 vs 朴素:设计声明不是走过场
#
# 怎么知道前面声明设计有没有用?把设计加权系数和未加权 OLS 系数并排看。若两者差得多,说明权重真的移动了结论;若差不多,那价值主要不在点估计,而在**方差**——聚类稳健的标准误通常比 iid 的更宽,区间更诚实。这张对照表就是你在论文里回应审稿人"你考虑抽样设计了吗"的直接证据。

# %%
import pandas as pd

sens = st.diagnostics["sensitivity"]
pd.DataFrame({
    "设计加权": sens["weighted"],
    "朴素 OLS": sens["unweighted"],
    "差异 delta": sens["delta"],
})

# %% [markdown]
# 在这份温和加权的数据上,`exposure` 的加权系数(0.205)与朴素系数(0.215)只差约 0.01——权重没有大幅移动点估计。这很正常:本例的权重分布不极端。真正的收获在方差侧,聚类稳健 SE 让区间更宽也更可信。`survey_estimate` 还把一张整洁的系数表存进了研究状态,含加权与未加权两列,可直接进论文附表。

# %%
st.artifacts["tables"]

# %% [markdown]
# ## 可视化:把加权估计画出来
#
# 交付物。`survey_dist` 直接从研究状态读回归结果,画一张森林图式的横向条形:每个非截距项一条**设计加权**点估计加 95% CI 须线,并把**朴素(未加权)**估计作为浅色菱形叠上去,让读者一眼看到权重把系数移了多少。图存成 notebook 同目录的相对路径。

# %%
sv.pl.survey_dist(st, out="fig_survey.png", title="调查加权估计 · outcome ~ exposure")
print("图已保存:fig_survey.png")

# %% [markdown]
# 蓝条是设计加权点估计加聚类稳健 95% CI,灰色菱形是朴素未加权对照。这张图加上上面的系数表,就是这条链的可交付结果:
#
# ![设计加权系数图](fig_survey.png)

# %% [markdown]
# ## 可复现的证据链
#
# 最后看一眼 `socialverse` 与普通估计脚本的关键差别。整条链跑下来,研究状态里自动积累了一份 **provenance 账本**:每一步用了哪个函数、消费了哪些槽、产出了什么。这份账本不是事后补的日志,而是每次调用的副产品——审稿人问"你怎么做的、权重和聚类考虑了吗",答案就逐步写在这里。同一套契约还会在你违规时拦你:如果没声明设计就去跑加权估计,注册表会当场报错并告诉你缺哪个前置,而不是默默给你一个错误的区间。

# %%
print(st.summary())

# %% [markdown]
# ## 小结
#
# 我们完整走了一条设计基础加权推断链:声明设计(权重/分层/PSU)→ 信度 α 与功效/样本量 → WLS + PSU 聚类稳健的加权回归 → 加权 vs 朴素敏感性 → 出图。方法上等价于 **R 的 survey 包**(Lumley)的 `svydesign()` + `svyglm()`、Python 的 **samplics**,测量部分对标 `psych::alpha` 与 `pwr`。socialverse 不重写这些方法,它薄封在 `statsmodels.WLS(cov_type="cluster")` 与纯 NumPy 的 α/功效公式之上。
#
# 与纯估计工具相比,这里多了两样东西:一是量表信度和聚类膨胀后的样本量在**采集前**就摆上台面,而不是估完才发现样本不够;二是一份贯穿始终、可审计的证据链,加上未声明设计就估计时会真的拦住你的守卫。下一本教程 [04_econometrics_replication](04_econometrics_replication.ipynb) 转向计量经济学的复现工作流。
