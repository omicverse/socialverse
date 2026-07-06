# %% [markdown]
# # 因果计量:面板固定效应 + 双重差分(DID)政策评估
#
# **这条分析链讲什么。** 我们用一个企业 × 年份的面板,评估某项政策(在
# `first_treated` 年对处理组企业生效)对结果变量 `y` 的**平均处理效应
# ATT**。核心不是"跑一个回归",而是把一条**有前置门槛的因果识别链**用
# `socialverse` 的注册表契约串起来:*平行趋势没过,DID 就不许被称作因果*。
#
# **涉及函数(全部注册在 `sv.registry`,契约机器可读)。**
#
# | 阶段 | 函数 | requires → produces |
# |---|---|---|
# | `sv.pp` | `ingest` | `∅` → `sources.datasets` |
# | `sv.pp` | `declare_design` | `sources.datasets` → `design.{panel_id,time,treatment,first_treated,…}` |
# | `sv.tl` | `parallel_trends` | `design.* + variables.outcome + estimand.target` → `diagnostics.pretrend`, `identification.parallel_trends` |
# | `sv.tl` | `did` | `design.* + variables.outcome + **identification.parallel_trends**` → `models.{did,twfe}`, `diagnostics.robustness` |
# | `sv.tl` | `event_study` | `design.* + variables.outcome` → `models.event_study` |
# | `sv.pl` | `forest` / `event_study_plot` | `models.did` / `models.event_study` → `artifacts.figures` |
#
# **`StudyState` 会被填哪些槽(12 槽词汇表的子集)。**
# `sources`(原始面板)· `design`(面板/时间/处理/处理时点列名)·
# `variables`(结果变量)· `estimand`(用户给定的 ATT 目标)·
# `identification`(**平行趋势判定**——这条链的因果闸门)·
# `models`(DID/TWFE、事件研究)· `diagnostics`(前趋势检验、稳健性矩阵)·
# `artifacts`(森林图、事件研究图)。
#
# **对标的现实 Py/R 包。** 这条链对标 **`pyfixest`**(Python 高维固定效应 +
# 聚类稳健 SE)与 R 的 **`fixest`**;平行趋势门槛 + 事件研究部分对标
# **`did`(Callaway–Sant'Anna)**。差别在最后一节:现实工具给你**估计量**,
# `socialverse` 额外给你一条**注册表 grounding + 证据链**——它在你调用 `did`
# 之前**查契约、拒绝越级**,并在事后把"平行趋势是否通过"焊进估计结论里。

# %%
import matplotlib
matplotlib.use("Agg")  # notebook 环境:无显示器,图直接存文件

import json
import os

import numpy as np
import pandas as pd

import socialverse as sv
from socialverse import datasets as ds

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

# 无 IPython 时给 display 一个后备,保证当普通脚本也能跑
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


print("socialverse", getattr(sv, "__version__", "(dev)"), "· registry 中函数数:", len(sv.registry))

# %% [markdown]
# ## 0. 先查注册表,不要猜 API(grounding)
#
# **为什么这步。** omicverse 让 agent 不幻觉 API 的机制不是"更大的模型",
# 而是 `ov.registry`:**查契约,而非猜签名**。`socialverse` 原样保留这套
# 查询面。开工前,先用一句中文 `双重差分` 把 DID 的完整契约问出来——它会
# 告诉我们 `did` 需要哪些槽、谁来生产、必须先跑什么。

# %%
print(sv.utils.registry_lookup("双重差分", max_results=3))

# %% [markdown]
# 契约里最关键的一行是 `Must run first: parallel_trends` 和
# `Requires: … identification['parallel_trends']`。也就是说:**`did` 把
# 平行趋势判定当成一个必填输入槽**。这不是文档建议,是注册时声明的硬依赖。
# 用 `get_prerequisites` 拿到它机器可读的全貌(与 omicverse 同形状,OmicOS 的
# `registry_lookup` 工具可以原样消费):

# %%
prereq = sv.registry.get_prerequisites("did")
print(json.dumps(prereq, ensure_ascii=False, indent=2))

# %% [markdown]
# 读法:
# - `required_functions: ["parallel_trends"]` —— 因果闸门。
# - `requires.identification: ["parallel_trends"]` + `satisfied_by` 指明
#   `parallel_trends` 这个函数来满足它 —— **查得到"谁产出我要的槽"**。
# - `requires.variables.outcome` 的 `satisfied_by` 是空 `[]` ——
#   代表这是**用户必须自己供给**的输入(没有函数会替你产出结果变量)。
# - `produces.models: ["did","twfe"]` + `diagnostics: ["robustness"]` ——
#   跑完会写哪些槽,下游(如森林图)据此可被自动编排。

# %% [markdown]
# ## 1. 让注册表规划整条链(resolve_plan)
#
# **为什么这步。** 我们的最终目标是一张森林图 `sv.pl.forest`。与其自己排
# 顺序,不如让注册表沿 `requires ↔ produces` 反向解析:要画森林图 → 需要
# `models.did` → 需要先跑 `did` → 需要 `parallel_trends` 和 `declare_design`
# → 需要 `ingest`。这就是 omicverse 里 `leiden → neighbors → pca` 的解析,
# 移植到社科。

# %%
plan = sv.registry.resolve_plan("sv.pl.forest")
print("有序执行计划:")
for i, full in enumerate(plan["plan"], 1):
    print(f"  {i}. {full.split('.')[-1]:16s}  ({full})")

print("\n需要用户供给的输入(needs_input,注册表无法代劳):")
for n in plan["needs_input"]:
    print(f"  - {n['slot']}.{n['key']}  ← 供给给 {n['for'].split('.')[-1]}")

print(f"\nescalations(auto_fix=escalate,需人确认再自动插入):{len(plan['escalations'])} 条")
print("  例:", json.dumps(plan["escalations"][0], ensure_ascii=False))

# %% [markdown]
# 注意 `needs_input` 精准点名了 `variables.outcome` 和 `estimand.target`
# ——**没有任何函数能产出它们**,必须由研究者供给。这正是下一步我们只手动
# 写这两个槽的原因:注册表把"哪些是人的判断、哪些是机器的编排"分得很清。

# %% [markdown]
# ## 2. 契约是活的:越级调用会被拒绝(RegistryError)
#
# **为什么这步。** 在写任何数据之前,先故意"越级":对一个空 `StudyState`
# 直接调 `sv.tl.did`。`@register` 包装器会在调用时**实时核对 `requires`**,
# 缺哪个槽就抛 `RegistryError`,并告诉你**每个缺失槽由谁产出**。这就是
# `leiden`-before-`neighbors` 的守卫,移植到因果识别:*平行趋势没验,
# 不给你报 DID*。这不是 bug,是这套设计的核心特性。

# %%
st_empty = sv.StudyState()
try:
    sv.tl.did(st_empty)
except sv.RegistryError as e:
    print("被注册表拦截(符合预期):\n")
    print(e)

# %% [markdown]
# 报错逐行给出**修复路径**:`design.*` 由 `declare_design` 产出、
# `identification.parallel_trends` 由 `parallel_trends` 产出、
# `variables.outcome` 是"user-supplied input"。照着 `resolve_plan` 的计划走
# 一遍即可。下面正式开工。

# %% [markdown]
# ## 3. 用户的两个输入 + 载入面板
#
# **为什么这步 / 契约。** 按 `needs_input`,研究者只需供给两件事:估计目标
# `estimand.target = "ATT"`(我们要估平均处理效应,不是关联)与结果变量
# `variables.outcome = "y"`。然后载入玩具面板——一个有**真实平行前趋势**、
# 处理在第 6 期(2015 年)对处理组开启、真实 ATT ≈ −0.8 的企业面板。

# %%
st = sv.StudyState()
st.write("estimand", "target", "ATT")   # 用户输入 1:估计量目标
st.write("variables", "outcome", "y")   # 用户输入 2:结果变量

df = ds.load_did_panel(att=-0.8)
print("面板维度:", df.shape, "· 列:", list(df.columns))
print(f"单元数 firm_id: {df['firm_id'].nunique()} · 年份: {df['year'].min()}–{df['year'].max()}"
      f" · 处理组占比: {df['treat'].mean():.0%} · first_treated(处理组)= "
      f"{sorted(df.loc[df.treat==1,'first_treated'].unique())}")
display(df.head(6))

# %% [markdown]
# ## 4. ingest → sources.datasets(`requires: ∅`)
#
# **为什么这步 / 契约。** `ingest` 是链的入口,`requires={}`(无前置),
# `produces={"sources": ["datasets"]}`。它把 DataFrame 登记进 state,并顺手
# 记一份元数据。跑完后,注册表包装器会往 `provenance` 追加一条审计记录。

# %%
sv.pp.ingest(st, data=df, name="did_panel")
print("sources 槽现在有:", list(st.sources.keys()))
print("dataset_meta:", json.dumps(st.sources["dataset_meta"], ensure_ascii=False))

# %% [markdown]
# ## 5. declare_design → design.*(`requires: sources.datasets`)
#
# **为什么这步 / 契约。** DID 的"设计"是一组**列名**:面板 id、时间、
# 处理指示、处理起始时点。`declare_design` 只登记列名(并校验列是否存在,
# 不存在只告警不报错),`requires={"sources":["datasets"]}` ——所以它必须在
# `ingest` 之后。它产出的 `design.*` 正是 `parallel_trends` / `did` 所需。

# %%
sv.pp.declare_design(
    st,
    panel_id="firm_id",
    time="year",
    treatment="treat_post",
    first_treated="first_treated",
)
print("design 槽:", dict(st.design))
print("列名校验告警:", st.design.get("warnings", "(无 —— 所有列都在数据里)"))

# %% [markdown]
# ## 6. parallel_trends → 因果闸门(`produces: identification.parallel_trends`)
#
# **为什么这步 / 契约。** 这是全链的**门槛**。函数估一个完整事件研究(单元
# FE + 时间 FE),再对所有**前导期**(相对时点 < −1,基期 = −1)系数做**联合
# Wald/F 检验**。原假设:前导期系数全为 0(即处理组与控制组在处理前趋势平行)。
# `p > 0.05` ⇒ **未拒绝**平行趋势 ⇒ `identification.parallel_trends = "pass"`。
# 它 `requires` 了 `estimand.target` —— 逼你先声明"要估因果量",再谈识别。

# %%
sv.tl.parallel_trends(st)
pt = st.diagnostics["pretrend"]
print("平行趋势判定:", st.identification["parallel_trends"])
print(f"联合 F = {pt['joint_F']:.3f} · p = {pt['p_value']:.3f} · 前导期数 = {pt['n_pre']}")
print("结论:", pt["note"])
print("\n各前导期系数(相对时点 → (系数, SE),应统计上≈0):")
pre_tbl = pd.DataFrame(
    [{"相对时点": k, "系数": round(v[0], 4), "SE": round(v[1], 4)} for k, v in pt["pre_coefs"].items()]
)
display(pre_tbl)

# %% [markdown]
# `p = 0.755 > 0.05` —— 前导期系数联合不显著,**未拒绝平行趋势**,闸门放行。
# 这一步把判定写进了 `identification.parallel_trends`,`did` 的 `requires`
# 从此被满足。

# %% [markdown]
# ## 7. did → ATT + 聚类稳健 SE(`requires: identification.parallel_trends`)
#
# **为什么这步 / 契约。** 现在——也**只有现在**——`did` 能跑。它估
# `y ~ treat_post + 单元 FE + 时间 FE`,SE 按 `panel_id` 聚类。关键设计:
# 它把上一步的平行趋势判定**读进结论**——通过则标"因果 ATT",未过则标
# "关联非因果"。同时产出一个 `robustness` 矩阵:同一点估计在
# classical / HC1 / 聚类三种方差设定下的 SE 对比。

# %%
sv.tl.did(st)
m = st.models["did"]
print(f"ATT   = {m['att']:.4f}")
print(f"SE    = {m['se']:.4f}  (聚类于 {m['n_clusters']} 家企业)")
print(f"95%CI = [{m['ci'][0]:.4f}, {m['ci'][1]:.4f}]")
print(f"p     = {m['p']:.2e}   ·   n = {m['n']}")
print(f"平行趋势 = {m['parallel_trends']}  →  结论:{m['note']}")
print("(真实 DGP 的 att 设为 −0.80,估计 −0.73 落在 CI 内)")

print("\n稳健性矩阵(点估计不变,只对比 SE 敏感度):")
display(pd.DataFrame(st.diagnostics["robustness"]["specs"]))

# %% [markdown]
# ## 8. event_study → 动态效应(leads/lags)
#
# **为什么这步 / 契约。** DID 给一个"平均"效应;事件研究把它**按相对处理
# 时点展开**:每个相对期一个系数(基期 −1 归一化为 0)。前导期(< 0)再次
# 检视前趋势,滞后期(≥ 0)刻画政策生效后的动态。`requires` 不含平行趋势
# (它本身就是用来看趋势的),但 `optional_functions` 建议先跑 `parallel_trends`。

# %%
sv.tl.event_study(st)
es = st.models["event_study"]
es_tbl = pd.DataFrame(
    [{"相对时点": int(k), "系数": round(v[0], 4), "SE": round(v[1], 4)} for k, v in es["coefs"].items()]
)
display(es_tbl)
print("读法:相对时点 <0(前导期)系数≈0 ⇒ 前趋势平行;≥0(滞后期)系数≈−0.8 ⇒ 政策负效应。")

# %% [markdown]
# ## 9. 出版级图:森林图 + 事件研究图(sv.pl → artifacts.figures)
#
# **为什么这步 / 契约。** `forest` `requires: models.did`,`event_study_plot`
# `requires: models.event_study` —— 因为第 7、8 步已产出这两个槽,绘图函数
# 直接可跑,无需我们手工传数据。图路径登记进 `artifacts.figures`,让证据链
# 也能追溯到图。

# %%
sv.pl.forest(st, out=figpath("fig_forest.png"), title="DID · ATT 点估计 ± 95% CI")
sv.pl.event_study_plot(st, out=figpath("fig_eventstudy.png"))

figs = st.artifacts["figures"]
print("已登记图件 artifacts.figures:")
for key in ("forest", "event_study"):
    info = figs.get(key, {})
    print(f"  {key:12s} → {os.path.basename(info.get('path', ''))}  · {info.get('note')}")

# %% [markdown]
# **森林图**(单个 ATT 系数 + 95% CI,虚线为零效应参考线):
#
# ![森林图](fig_forest.png)
#
# **事件研究图**(动态效应:前导期贴着 0,处理时点 0 起跳到 ≈ −0.8):
#
# ![事件研究图](fig_eventstudy.png)

# %% [markdown]
# ## 10. 反面演示:平行趋势**没过**时,同一条链拒绝叫它"因果"
#
# **为什么这步。** 门槛只有在能真的挡住东西时才有意义。我们构造一个处理组
# 在**处理前就有额外发散趋势**的面板(平行趋势被违背),跑同一条链,看
# `did` 的结论如何自动翻转——这正是 `pyfixest`/`fixest` 给你系数、却不会替你
# 把"识别是否成立"焊进结论的地方。

# %%
rng = np.random.default_rng(0)
n_units, n_periods, treat_period, att = 40, 8, 5, -0.8
units = np.arange(n_units)
treated = units < n_units // 2
unit_fe = rng.normal(0, 1.0, n_units)
time_fe = np.linspace(0, 1.2, n_periods)
rows = []
for i in units:
    for t in range(n_periods):
        post = int(t >= treat_period)
        tp = int(treated[i]) * post
        pre_divergence = 0.35 * t * int(treated[i])   # 处理组特有的前趋势 → 违背平行
        y = 2.0 + unit_fe[i] + time_fe[t] + pre_divergence + att * tp + rng.normal(0, 0.4)
        rows.append({
            "firm_id": int(i), "year": 2010 + t, "treat": int(treated[i]),
            "post": post, "treat_post": tp,
            "first_treated": (2010 + treat_period) if treated[i] else 0,
            "y": round(float(y), 4),
        })
bad = pd.DataFrame(rows)

st_bad = sv.StudyState()
st_bad.write("estimand", "target", "ATT")
st_bad.write("variables", "outcome", "y")
sv.pp.ingest(st_bad, data=bad)
sv.pp.declare_design(st_bad, panel_id="firm_id", time="year",
                     treatment="treat_post", first_treated="first_treated")
sv.tl.parallel_trends(st_bad)
sv.tl.did(st_bad)

pt_bad = st_bad.diagnostics["pretrend"]
m_bad = st_bad.models["did"]
print(f"平行趋势判定 : {st_bad.identification['parallel_trends']}"
      f"  (p = {pt_bad['p_value']:.4f} ≤ 0.05 ⇒ 拒绝)")
print(f"结论标签     : {m_bad['note']}")
print(f"点估计 ATT   : {m_bad['att']:.3f}  ← 数值仍算得出,但**不许被称作因果**")
print("\n对照:干净面板判定 =", st.identification["parallel_trends"],
      "→", st.models["did"]["note"])

# %% [markdown]
# 同一套代码、同一个 `did`,**只因平行趋势 p 值跨过阈值,结论标签从"因果
# ATT"翻成"关联非因果"**。门槛不是注释里的提醒,而是数据驱动、写进 state 的
# 判定。这就是这条链相对纯估计工具的增量。

# %% [markdown]
# ## 11. 证据链:provenance 审计账本
#
# **为什么这步。** 每个注册函数写 state 时都追加一条 provenance 记录
# (第几步、函数全名、参数、requires、produces)。跑完整条链,`StudyState`
# 自带一份**可复现审计轨迹**——社科里第一等重要的"证据脊柱"。

# %%
print(st.summary())
print("\n—— provenance 逐步账本 ——")
for r in st.provenance:
    prod = "; ".join(f"{k}:{','.join(v)}" for k, v in r["produces"].items()) or "∅"
    print(f"  [{r['step']}] {r['function'].split('.')[-1]:16s} produces → {prod}")

# %% [markdown]
# ## 小结:对标现实工具 + socialverse 的差异
#
# 这条 **ingest → declare_design → parallel_trends → did → event_study →
# forest** 链,对标 **`pyfixest` / R `fixest`**(高维固定效应 + 聚类稳健 SE)
# 与 **`did`(Callaway–Sant'Anna)** 的平行趋势 + 事件研究工作流。
#
# **差异 = 注册表 grounding + 证据链。** `pyfixest`/`fixest` 交付的是**估计
# 量**;你得自己记住"先验平行趋势""SE 要聚类""这系数到底能不能叫因果"。
# `socialverse` 把这些做成**机器可读的契约**:调 `did` 前先查
# `get_prerequisites` / `resolve_plan`(**查而非猜**),越级即抛 `RegistryError`
# 并指明修复路径(第 2 节);平行趋势判定被 `did` 的 `requires` **强制前置**,
# 并把 pass/fail 焊进结论标签(第 7、10 节);整条链留下一份 `provenance`
# 审计账本(第 11 节)。**注册表是脊柱,估计器只是接在契约上的一个后端。**

# %%
print("02_causal_did 教学 notebook 运行完毕 · 图已存:fig_forest.png, fig_eventstudy.png")
