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
# # 复杂抽样:设计基础加权推断 (design-based weighted inference)
#
# **这条分析链讲什么。** 一份真实的社会调查(民调、健康调查、家户面板)几乎从不是简单
# 随机抽样:它带着**抽样权重**(weight)、**分层**(strata)、**初级抽样单元 / PSU**
# (primary sampling unit,聚类)。如果你把它当独立同分布的样本直接跑 OLS,你的**点估计
# 会偏**(不加权 ⇒ 面向"样本"而非"目标总体")、**标准误会假小**(忽略聚类内相关 ⇒
# 置信区间过窄、p 值虚假显著)。**设计基础推断 (design-based inference)** 就是把抽样设计
# 显式声明出来,让权重进入估计、让 PSU 进入方差,从而得到面向**目标总体**的、诚实的区间。
#
# 我们走一条 socialverse 的 **complex-survey 链**,一次端到端:
#
# ```
# ingest → declare_design → design_survey → survey_estimate → survey_dist
# ```
#
# **涉及的函数(全部是注册表里的契约函数)**
#
# | 函数 | 相 | 干什么 | requires → produces |
# |---|---|---|---|
# | `sv.pp.ingest` | prepare | 把 DataFrame 登记进 state | `∅` → `sources['datasets']` |
# | `sv.pp.declare_design` | prepare | 声明 weights/strata/psu 等设计列 | `sources['datasets']` → `design[…]` |
# | `sv.tl.design_survey` | analyze | 采集前:构念→题项、信度 α、功效/样本量(含 DEFF) | `estimand['target']` → `variables['scales'/'constructs']`, `diagnostics['reliability'/'power']` |
# | `sv.tl.survey_estimate` | analyze | 采集后:权重 + PSU cluster-robust 的 design-based 回归 | `sources['datasets']`, `design['weights']`, `variables['outcome']` → `models['weighted_reg']`, `diagnostics['sensitivity']`, `artifacts['tables']` |
# | `sv.pl.survey_dist` | plot | 加权系数条形图(设计加权 vs 朴素对比) | `models['weighted_reg']` → `artifacts['figures']` |
#
# **`StudyState` 会被填哪些槽(12 槽词汇表里的子集):**
# `sources`(原始数据)· `design`(权重/分层/PSU)· `variables`(结局/量表/构念)·
# `estimand`(用户给定的目标量:这里是 *prevalence* 患病率/占比)· `models`(加权回归)·
# `diagnostics`(信度 α、功效、加权 vs 朴素敏感性)· `artifacts`(系数表、图)。
#
# **对标的现实工具。** 这条链对标 Python 的 [**samplics**](https://github.com/samplics-org/samplics)
# 和 R 的 [**survey** 包(Thomas Lumley)](https://cran.r-project.org/package=survey) ——
# 它们是设计基础推断的事实标准:`svydesign(ids=~psu, strata=~strata, weights=~w)` 声明设计,
# `svyglm()` 做加权回归,方差用 Taylor 线性化 / 重抽样。测量部分(Cronbach α、功效/样本量)
# 对标 R 的 **psych**(`psych::alpha`)与 **pwr**。socialverse 不重写这些方法——它在
# `statsmodels.WLS(cov_type="cluster")` 之上薄薄封一层,**真正不同的是那张依赖注册表**:
# 每个函数带机器可读契约(requires→produces),调用前可以*查*而不是*猜*,调用后自动留下
# provenance 证据链。下面每一步我们都先看契约、再跑、再看真实输出。

# %% [markdown]
# ## 0. 先查注册表,别猜 API
#
# omicverse 让 agent 不幻觉 API 的机制是 `ov.registry`;socialverse 原样搬来。写代码前,
# 用 `sv.utils.registry_summary()` 看域地图,用 `registry_lookup('survey')` 看某功能的契约
# ——这就是 grounding:**查,而不是猜**。注意 `registry_lookup` 的输出格式和 omicverse 的
# 生物域完全一致,所以 OmicOS 的 `registry_lookup` 工具能直接消费这张社科注册表。

# %%
import matplotlib

matplotlib.use("Agg")  # 无显示后端:内核/CI 安全,图只写文件

import json

import pandas as pd

import socialverse as sv
from socialverse import datasets as ds

# 让 DataFrame 在纯脚本里也"像 notebook 一样"打印;在真正的 Jupyter 里会用富文本渲染。
try:
    display  # noqa: F821  (Jupyter 内建)
except NameError:
    def display(obj):
        print(obj.to_string() if isinstance(obj, pd.DataFrame) else obj)

print(sv.utils.registry_summary())

# %% [markdown]
# 域地图告诉我们 **complex survey** 链是
# `ingest → declare_design → design_survey → survey_estimate → survey_dist`。
# 现在放大看这条链的核心——`survey_estimate` 的契约:它 **requires**
# `sources['datasets']` + `design['weights']` + `variables['outcome']`,**produces**
# `models['weighted_reg']` + `diagnostics['sensitivity']`。`registry_lookup` 把这一切
# 连同后端(statsmodels/scipy)一起打印给 agent 看。

# %%
print(sv.utils.registry_lookup("complex survey", max_results=3))

# %% [markdown]
# ## 1. 目标量 (estimand):这是唯一"用户给定"的输入
#
# 社科分析的起点不是数据,而是**你想估什么**。在 12 槽词汇表里 `estimand` 是**用户给定**的
# ——没有任何函数 produce 它,所以 `resolve_plan` 会把它列为 `needs_input`。这里我们估的是
# **患病率 / 人群占比 (prevalence)**:某二元结局在**目标总体**中的水平,以及暴露与它的关联。
#
# 契约角度:`design_survey` 的 `requires={"estimand": ["target"]}`,所以我们必须先写下
# `estimand['target']`,否则采集前设计那步会被注册表拒绝。

# %%
st = sv.StudyState()
st.write("estimand", "target", "prevalence")  # 唯一手工输入:我们要估的目标量
print("estimand 槽 =", dict(st.estimand))
print("StudyState 初始:", repr(st))

# %% [markdown]
# ## 2. `ingest`:把调查表登记进 `sources`
#
# **为什么这步。** 下游每个契约都 `require` 某个已存在的槽;链的第一步就是让 `sources['datasets']`
# 存在,否则 `declare_design` / `survey_estimate` 无从 require 起。
#
# **契约。** `ingest`: `requires={}` → `produces={"sources": ["datasets"]}`。它是 community 层
# 的入口,永不抛错(coerce 到 DataFrame,失败也给空表)。
#
# 数据是 `ds.load_survey()`:300 个受访者、6 个 Likert 题项(item1..item6,由同一潜变量驱动
# ⇒ 信度会很高)、外加设计列 `weight / strata / psu`、一个二元 `exposure` 和二元 `outcome`。

# %%
df = ds.load_survey()
print("survey 数据:", df.shape, "行×列")
display(df.head(5))

sv.pp.ingest(st, data=df)
print("\ningest 后 sources['dataset_meta'] =")
print(json.dumps(st.sources["dataset_meta"], ensure_ascii=False, indent=2))

# %% [markdown]
# ## 3. `declare_design`:显式声明抽样设计 —— 这是设计基础推断的全部关键
#
# **为什么这步。** 这是复杂抽样和"随便跑个回归"的**分水岭**。我们把三个设计事实写进 `design` 槽:
#
# - **`weights='weight'`** — 抽样权重。加权 ⇒ 估计面向**目标总体**而非样本(纠正不等概率抽样)。
# - **`strata='strata'`** — 分层。分层抽样降低方差;方差公式要按层算。
# - **`psu='psu'`** — 初级抽样单元(聚类)。同一 PSU 内的观测相关,标准误必须**按 PSU 聚类**,
#   否则区间假窄。
#
# **契约。** `declare_design`: `requires={"sources": ["datasets"]}` →
# `produces={"design": [...weights, strata, psu, ...]}`。它把列名和已登记的数据**核对**:
# 不存在的列进 `design['warnings']` 而**不抛错**(声明是元数据,不是硬门)。下面我故意传一个
# `unit='row'`(数据里没有这列)来演示这个 fail-soft 警告机制。

# %%
sv.pp.declare_design(
    st,
    weights="weight",   # 抽样权重列
    strata="strata",    # 分层列
    psu="psu",          # 初级抽样单元(聚类)列
    unit="row",         # 故意写一个不存在的列 → 触发 fail-soft 警告(不抛错)
)
print("design 槽 =")
print(json.dumps(dict(st.design), ensure_ascii=False, indent=2))

# %% [markdown]
# 注意 `design['warnings']` 里出现了 `"design.unit: column 'row' not found in datasets"`
# ——声明步骤把一个可疑列名**报告**出来而非崩溃。真实工作流里这正是你想要的:错列名早暴露,
# 但一个拼错的可选字段不该让整条管道停摆。`weights/strata/psu` 三列都存在,已被记录,
# 下游的加权估计就靠读这三个槽干活。

# %% [markdown]
# ## 4. `design_survey`:采集前的测量与功效设计(信度 α + 样本量)
#
# **为什么这步。** 在"估"之前,先问两件事:**我的量表可靠吗?**(内部一致性信度)**样本够大吗?**
# (功效 / 样本量,且必须为聚类设计做 **DEFF 膨胀**)。这一步把构念变成量表、算 Cronbach α、
# 算达到目标功效所需的 n。
#
# **契约。** `design_survey`: `requires={"estimand": ["target"]}` →
# `produces={"variables": ["scales","constructs"], "diagnostics": ["reliability","power"]}`。
# 它就是我们在第 1 步写 `estimand['target']` 的原因。
#
# **两个计算(都是真算,不是占位):**
# - **Cronbach's α** = (k/(k−1))·(1 − Σ题项方差 / 总分方差),用完整个案、样本方差(ddof=1)。
#   有 `pingouin` 就用它(带 CI),没有就用这套 NumPy 公式(本环境无 pingouin,走 NumPy——
#   结果一样真实、确定)。α ≥ 0.9 属"优秀"(6 个题项由同一潜因子驱动,理应很高)。
# - **样本量** n ≈ ((z_{α/2}+z_β)/效应量)² × **DEFF**。DEFF(设计效应)>1 反映聚类使有效样本
#   缩水,所以要按它把 n 往上抬。z 分位数有 scipy 就精确算(本环境有)。

# %%
items = df[[c for c in df.columns if c.startswith("item")]]  # 6 个 Likert 题项作试点响应矩阵
sv.tl.design_survey(
    st,
    items=items,
    effect_size=0.2,  # 标准化效应(小)
    deff=1.5,         # 设计效应:聚类使方差膨胀 50%
    alpha=0.05,
    power=0.8,
)

print("信度 (Cronbach α):")
print(json.dumps(st.diagnostics["reliability"], ensure_ascii=False, indent=2))
print("\n功效 / 样本量 (含 DEFF 膨胀):")
print(json.dumps(st.diagnostics["power"], ensure_ascii=False, indent=2))
print("\n推断出的量表 variables['scales'] =", st.variables["scales"])

# %% [markdown]
# 读数:**α ≈ 0.932**(k=6 题,n=300 完整个案)——内部一致性优秀,量表可用。功效计算给出
# **n_required ≈ 295**(已按 DEFF=1.5 膨胀);我们手上正好 300,勉强达标——这正是聚类设计里
# "看起来 300 很多、有效信息其实缩水"的教训。如果把 `deff` 调回 1.0(假装简单随机),所需 n
# 会更小——**忽视聚类会让你误以为样本够用**。

# %% [markdown]
# ## 5. `survey_estimate`:设计基础加权回归(WLS + PSU cluster-robust SE)
#
# **为什么这步 —— 也是整条链的心脏。** 现在做"采集后"的估计。它不是普通 OLS,而是:
#
# 1. **加权** —— 用 `design['weights']` 做 **WLS**,估计面向目标总体(纠正不等概率抽样);
# 2. **聚类稳健方差** —— 因为声明了 `design['psu']`,`statsmodels` 用
#    `cov_type="cluster"` 按 PSU 聚类算标准误(诚实的、通常更宽的区间);
# 3. **敏感性对照** —— 同时跑一遍**未加权 OLS**,把加权 vs 朴素的系数并排放进
#    `diagnostics['sensitivity']`:两者差得多 ⇒ 权重/聚类真的重要。
#
# **契约(硬门)。** `survey_estimate`: `requires` = `sources['datasets']` +
# `design['weights']` + `variables['outcome']`。所以在跑它之前,我们必须先写下结局变量。

# %%
st.write("variables", "outcome", "outcome")  # 声明二元结局列(survey_estimate 的 requires 之一)

sv.tl.survey_estimate(st, exposure="exposure")  # y = outcome ~ exposure,WLS + PSU 聚类稳健

print("models['weighted_reg']:")
print(json.dumps(st.models["weighted_reg"], ensure_ascii=False, indent=2))

# %% [markdown]
# 读数:`exposure` 的设计加权系数 ≈ **0.205**,95% CI ≈ **[0.116, 0.295]**,`cov_type` 是
# **`cluster`**(按 PSU 聚类的标准误,而非天真的 iid 方差),n=300。这是一个面向目标总体、
# 且方差诚实的关联估计。注意 `cov_type` 字段本身就是**证据**:它记录了这个区间是聚类稳健的,
# 而不是假设独立同分布算出来的。

# %% [markdown]
# ### 加权 vs 朴素:为什么设计声明不是走过场
#
# `diagnostics['sensitivity']` 把设计加权系数和未加权 OLS 系数并排。`delta` 小 ⇒ 在这份
# (温和加权的)数据上,权重没有大幅移动点估计;但**方差**的故事不同——聚类稳健 SE 通常比
# iid SE 宽,而那一步的价值不在点估计而在**区间的诚实度**。这个对照表就是你在论文里回应
# 审稿人"你考虑抽样设计了吗"的证据。

# %%
sens = st.diagnostics["sensitivity"]
print("敏感性(设计加权 vs 朴素 OLS):")
print(json.dumps(sens, ensure_ascii=False, indent=2))

print("\nartifacts['tables'] —— 整洁系数表(含未加权列):")
display(st.artifacts["tables"])

# %% [markdown]
# ## 6. `survey_dist`:把加权估计画出来(设计加权 vs 朴素叠加)
#
# **为什么这步。** 交付物。森林图式的横向条形:每个非截距项一条**设计加权**点估计 + 95% CI
# 须线,并把**朴素(未加权)**估计作为浅色菱形叠上去,让读者一眼看到权重把系数移了多少。
#
# **契约。** `survey_dist`: `requires={"models": ["weighted_reg"]}` →
# `produces={"artifacts": ["figures"]}`。你画不出一个你没估过的系数——注册表守住这条底线。
# 图保存为 notebook 同目录的相对路径 `fig_survey.png`。

# %%
sv.pl.survey_dist(st, out="fig_survey.png", title="调查加权估计 · outcome ~ exposure")
print("figures 槽 =")
print(json.dumps(st.artifacts["figures"], ensure_ascii=False, indent=2))

# %% [markdown]
# ![设计加权系数分布图](fig_survey.png)
#
# 蓝条是设计加权点估计 + 聚类稳健 95% CI;灰色菱形是朴素未加权对照。这张图 + 上面的系数表,
# 就是这条链的可交付结果。

# %% [markdown]
# ## 7. 注册表的 grounding:契约是**活的**,不只是元数据
#
# 前面每一步都"恰好"满足了下一步的 requires——这不是巧合,是注册表在守门。下面用两个演示证明:
# **(a)** 违约会被当场拒绝(而不是给你一个悄悄错误的结果);**(b)** 你可以让注册表**自己规划**
# 整条链的顺序。这正是 omicverse 里 `leiden` 必须在 `neighbors` 之后的那道守卫,搬到了社科。

# %% [markdown]
# ### (a) 违约即拒:在声明设计之前调用 `survey_estimate`
#
# 新开一个 state,只 `ingest` 就直接跳到加权估计——`design['weights']` 和
# `variables['outcome']` 都还没有。注册表抛 `sv.RegistryError`,并**精确告诉你缺哪个槽、
# 由哪个函数产**(`design.weights` 由 `declare_design` 产;`variables.outcome` 是用户输入)。
# 这个报错是**特性**:宁可当场拒绝,也不给一个静默错误的估计。

# %%
st_bad = sv.StudyState()
st_bad.write("estimand", "target", "prevalence")
sv.pp.ingest(st_bad, data=df)  # 只 ingest,故意跳过 declare_design

try:
    sv.tl.survey_estimate(st_bad, exposure="exposure")
except sv.RegistryError as err:
    print("RegistryError(这是设计好的守卫,不是 bug):\n")
    print(err)

# %% [markdown]
# ### (b) 让注册表自己规划链:`get_prerequisites` + `resolve_plan`
#
# 与其记住调用顺序,不如问注册表。`get_prerequisites('survey_estimate')` 返回它的完整契约
# 和"每个 requires 由谁满足"(`satisfied_by`)。`resolve_plan('survey_dist')` 更进一步:
# 从目标反推,拓扑排序出一条**完整可执行的链**,把用户必须自己给的输入列进 `needs_input`
# (这里正是 `variables.outcome`——它没有生产者),把 `auto_fix=escalate` 的自动插入步骤
# 列进 `escalations`(需人确认)。这个 JSON 的形状和 omicverse 的完全一致,agent 直接消费。

# %%
print("get_prerequisites('survey_estimate'):")
print(json.dumps(sv.registry.get_prerequisites("survey_estimate"), ensure_ascii=False, indent=2))

plan = sv.registry.resolve_plan("survey_dist")
print("\nresolve_plan('survey_dist') —— 注册表自动排出的链:")
print("  plan       :", [p.split(".")[-1] for p in plan["plan"]])
print("  needs_input:", plan["needs_input"])
print("  escalations:", [f"{e['for'].split('.')[-1]} ← {e['auto_insert']}" for e in plan["escalations"]])

# %% [markdown]
# 排出的 `plan` 正是我们手动走的顺序:`ingest → declare_design → survey_estimate → survey_dist`。
# `needs_input` 精确点名 `variables.outcome`——注册表知道这是**用户必须提供**的、无法自动生成的
# 输入。这就是"查而非猜":agent 不必背 API,问一句就得到有序、可执行、且标注了人工确认点的计划。

# %% [markdown]
# ## 8. 证据链:`st.summary()` 与完整 provenance
#
# 每个注册函数在写槽的同时都自动 `record` 一条 provenance(函数全名 + 参数 + requires + produces)。
# 走完链,`StudyState` 就**自带一份可复现、可审计的账本**——这不是事后补的日志,而是每次调用的副产品。
# 在社科里,这条"证据脊柱"是一等公民:审稿人问"你怎么做的",答案就在这里,逐步、带契约。

# %%
print(st.summary())

# %%
print("完整 provenance 账本(每步:函数 / requires / produces):")
for rec in st.provenance:
    fn = rec["function"].split(".")[-1]
    req = {k: v for k, v in rec["requires"].items() if v}
    pro = {k: v for k, v in rec["produces"].items() if v}
    print(f"  [step {rec['step']}] {fn}")
    print(f"           requires={req or '∅'}")
    print(f"           produces={pro}")

# %% [markdown]
# ## 小结:这条链对标的现实工具 + socialverse 的差异
#
# **对标。** 我们完整走了一条设计基础加权推断链:声明设计(权重/分层/PSU)→ 信度 α + 功效/样本量
# → WLS + PSU cluster-robust 加权回归 + 加权 vs 朴素敏感性 → 交付图。方法上等价于
# **R survey(Lumley)** 的 `svydesign()` + `svyglm()`、Python 的 **samplics**,测量部分等价于
# **psych::alpha** 与 **pwr**。socialverse 不重写这些方法——它薄封在 `statsmodels.WLS(cov_type="cluster")`
# 与纯 NumPy 的 α/功效公式之上。
#
# **差异 = 注册表 grounding + 证据链。** R survey 给你正确的方法,但顺序、依赖、"这步能不能跑"
# 全靠你自己记。socialverse 把这层做成机器可读契约:
#
# - **grounding(查而非猜)** —— `registry_lookup` / `get_prerequisites` / `resolve_plan` 让 agent
#   在写代码前查到每个函数的 requires→produces,并自动排出有序、标注人工确认点的链;违约(如先估计
#   后声明设计)当场抛 `RegistryError` 并告诉你缺什么、谁产——**宁可拒绝,不给静默错误的估计**。
# - **证据链(built-in provenance)** —— 每步自动记账,`st.summary()` + `provenance` 就是可复现、
#   可审计的"证据脊柱",直接回应"你考虑抽样设计了吗 / 你怎么做的"这类审稿问题。
#
# 一句话:**方法照搬社科最佳工具,注册表把它们变成 agent 可查、可规划、可审计的契约。**
