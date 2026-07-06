# %% [markdown]
# # 研究闭环:一个可复核的小型研究 + 证据链导出
#
# **这条分析链讲什么。** 前面每本 notebook 各讲一个环节(治理、因果、调查、
# 质性、文本…)。这一本是**收束**:把一次**完整而微型的研究**从头跑到尾——
# 先过**治理闸门**(伦理/可识别性),再声明估计量、由**注册表规划**因果链、
# 估计 ATT、出图,最后**把整条链的出处(provenance)导出成一份可复核的证据
# 台账**,并把**注册表清单(registry manifest)**导出成机器可读的目录。
#
# 主线只有一句话:**一次分析结束后,它能不能自证它做了什么、依赖什么、
# 产出什么。** 这正是 `socialverse` 对标传统计量/统计包的差异化收束——别的
# 工具给你**估计量**,`socialverse` 额外给你一条**注册表 grounding + 证据链**。
#
# **涉及函数(全部注册在 `sv.registry`,契约机器可读)。**
#
# | 阶段 | 函数 | requires → produces |
# |---|---|---|
# | `sv.pp` | `ingest` | `∅` → `sources.datasets` |
# | `sv.gov` | `ethics_check` | `design.unit` → `governance.ethics` |
# | `sv.pp` | `declare_design` | `sources.datasets` → `design.{panel_id,time,treatment,first_treated,…}` |
# | `sv.tl` | `parallel_trends` | `design.* + variables.outcome + estimand.target` → `diagnostics.pretrend`, `identification.parallel_trends` |
# | `sv.tl` | `did` | `design.* + variables.outcome + **identification.parallel_trends**` → `models.{did,twfe}`, `diagnostics.robustness` |
# | `sv.pl` | `forest` | `models.did` → `artifacts.figures` |
#
# 查询面(不产出、只导出):`sv.registry.resolve_plan` · `st.provenance` ·
# `st.summary()` · `sv.registry.manifest` / `export_registry`。
#
# **`StudyState` 会被填哪些槽(12 槽词汇表的子集)。**
# `sources`(原始面板)· `design`(单元 + 面板/时间/处理列名)·
# `governance`(**伦理判定**——这条链的准入闸门)· `variables`(结果变量)·
# `estimand`(用户给定的 ATT 目标)· `identification`(平行趋势判定)·
# `models`(DID/TWFE)· `diagnostics`(前趋势、稳健性矩阵)·
# `artifacts`(森林图)。跑完这些槽合起来就是一份研究的**结构化底稿**。
#
# **对标的现实 Py/R 包。** 单看每一步:治理对标 IRB 表单 + `sdcMicro`(k-匿名)
# 这类无代码/半代码流程;因果对标 **`pyfixest`** / R 的 **`fixest`** 与
# **`did`(Callaway–Sant'Anna)**;出图对标 `matplotlib` / `coefplot`。但**把它们
# 串成一次带出处的研究并导出清单**——这一整套没有对标物。最接近的是
# **`targets`(R)/ `snakemake`** 的"可复现流水线"哲学,加上 **DataLad / W3C
# PROV** 的出处追踪;`socialverse` 的差异在于:出处**不是你额外写的**,而是
# 注册表在每次调用成功时**自动焊上去的**,并且每一步的依赖都是**机器可查、
# 可拒绝越级**的契约。

# %%
import matplotlib
matplotlib.use("Agg")  # notebook 环境:无显示器,图直接存文件

import json
import os

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


print("socialverse", getattr(sv, "__version__", "(dev)"),
      "· registry 中函数数:", len(sv.registry))

# %% [markdown]
# ## 0. 开一个空的研究态,先建立"我要走到哪"
#
# **为什么这步。** 一次可复核的研究,起点不是"加载数据然后开跑",而是
# **先声明目标、再让注册表把到达目标的链路规划出来**——这样每一步为什么存在
# 都有据可查。我们的目标(target)就是估计一个 **DID 的 ATT**。
#
# `StudyState` 是社科版的 AnnData:它不是一个数据矩阵,而是 **12 个具名槽**
# 组成的共享词汇表,注册表的 `requires`/`produces` 契约就用这些槽名说话。
# 开局它是空的——我们让它一步步被填满,而**每一次填充都会留下出处记录**。

# %%
st = sv.StudyState()
print("初始状态(空研究态):")
print(st.summary())

# 一次性把「目标量」和「结果变量」声明进去:这两样通常来自研究问题,
# 没有任何函数能替你产出(下面 resolve_plan 会把它们标成 needs_input)。
st.write("estimand", "target", "ATT")     # 我们要估计的目标:平均处理效应
st.write("variables", "outcome", "y")     # 结果变量列名

print("\n声明目标后:", repr(st))

# %% [markdown]
# ## 1. 让注册表规划这条链(grounding:查而非猜)
#
# **为什么这步。** omicverse 让 agent 不幻觉 API 的机制不是"更大的模型",而是
# `ov.registry`:**查契约、按依赖把链路排出来**。`socialverse` 原样保留这套
# 查询面。在写任何一行分析代码之前,先问一句:「要得到 `did`,得先跑哪些函数、
# 还差哪些用户输入?」
#
# `resolve_plan("did")` 会走一遍依赖图,返回三样东西:
# - `plan`:到达 `did` 的**有序**函数链(`did` 的 `prerequisites` 与 `requires`
#   会把 `parallel_trends`、`declare_design`、`ingest` 逐层前插);
# - `needs_input`:**没有任何函数能产出、必须你来给**的槽(比如 `estimand.target`);
# - `escalations`:`auto_fix="escalate"` 的步骤——**能自动补,但按契约要人确认**
#   (社科的因果识别不该被静默地自动接管)。

# %%
plan = sv.registry.resolve_plan("did", state=st)

print("要得到 did,注册表规划的有序链路:")
for i, full in enumerate(plan["plan"], 1):
    print(f"  {i}. {full.split('.')[-1]:<16} ({full})")

print("\nneeds_input(无函数可产出、必须用户提供):")
for item in plan["needs_input"]:
    print(f"  - {item['slot']}.{item['key']}  ← for {item['for'].split('.')[-1]}")

print("\nescalations(可自动前插、但契约要求人工确认):", len(plan["escalations"]), "处")
for e in plan["escalations"][:4]:
    print(f"  - {e['for'].split('.')[-1]:<16} 需 {e['needs']:<28} → 拟插 {e['auto_insert']}")
print("  …(共 %d 处;auto_fix=escalate 的都会浮现出来)" % len(plan["escalations"]))

# %% [markdown]
# **读这份计划。** 注意 `needs_input` 里**没有** `estimand.target` 和
# `variables.outcome`——因为我们在第 0 步已经写进去了,注册表因此不再要求它们。
# 这就是 grounding 的价值:计划是**对着当前研究态**算出来的,不是背模板。
#
# 我们也可以对单个函数问它的完整契约——`did` 到底 requires 什么、谁能满足:

# %%
prereq = sv.registry.get_prerequisites("did")
print("did 的契约(get_prerequisites):")
print("  必须先跑(prerequisites.functions):", prereq["required_functions"])
print("  requires:", prereq["requires"])
print("  produces:", prereq["produces"])
print("  auto_fix:", prereq["auto_fix"])
print("  每个 requires 由谁满足(satisfied_by):")
for slot_key, producers in prereq["satisfied_by"].items():
    who = ", ".join(producers) if producers else "(用户输入,无函数产出)"
    print(f"    {slot_key:<28} ← {who}")

# %% [markdown]
# 关键的一行:`identification.parallel_trends ← parallel_trends`。这说明
# `did` **不接受**一个没做过平行趋势检验的世界——注册表把"因果识别的前置假设"
# 写死成了一个**机器可查的依赖**。下面我们就照这份计划一步步执行。

# %% [markdown]
# ## 2. 加载数据并登记进 sources(链的第一环)
#
# **为什么这步 + 契约。** `ingest` 的契约是 `requires=∅ → produces=sources.datasets`——
# 它是每条链的社区版入口,不需要任何前置,只负责把一张表登记进研究态,好让
# 下游的 `requires={"sources": ["datasets"]}` 有东西可依赖。
#
# 数据是一个**企业 × 年份**面板:处理组企业在 `first_treated` 年被政策命中,
# 真实 ATT = `-0.8`(数据生成时注入,后面看估计能不能把它找回来)。

# %%
df = ds.load_did_panel(att=-0.8)
sv.pp.ingest(st, data=df, name="did_panel_firm_year")

print("面板形状:", df.shape, "| 列:", list(df.columns))
display(df.head(6))
print("\ningest 后 sources 槽:", list(st.sources.keys()))
print("dataset_meta:", st.sources["dataset_meta"])

# %% [markdown]
# ## 3. 治理闸门:先过伦理审查,再谈分析(准入门槛)
#
# **为什么这步 + 契约。** 社科研究里,**能不能碰这批数据**是先于"怎么分析"的问题。
# `ethics_check` 的契约是 `requires=design.unit → produces=governance.ethics`:它要求
# 你**先声明分析单元**,然后对 IRB 分类 / 知情同意 / **可识别性(真算 k-匿名)** /
# 数据最小化跑四项检查,折成一个 `PASS / FIX / NO-GO` 判决。`auto_fix="escalate"`
# ——`FIX`/`NO-GO` 必须人来处理,不允许自动放行。
#
# 我们**故意先犯一个常见错误**:把单元写成 `"firm-year"`。看闸门怎么反应。

# %%
st.write("design", "unit", "firm-year")   # 故意的"手滑":连字符写法
sv.gov.ethics_check(st, data=df, quasi_identifiers=["treat"])

ethics = st.governance["ethics"]
print("伦理判决:", ethics["verdict"], "| 判定为人类受试者:", ethics["human_subjects"])
for c in ethics["checks"]:
    print(f"  - {c['check']:<12} {c['status']:<6} :: {c['detail']}")
print("\nk-匿名(真实计算):", ethics["k_anonymity"])

# %% [markdown]
# **闸门抓到了什么。** 判决是 **NO-GO**,而且它把单元判成了"人类受试者"——
# 因为 `"firm-year"` 不在它认识的非人类单元集合里(`firm`/`country`/`document`…
# 在,但带连字符的 `firm-year` 不在),于是它保守地当作**涉及个人数据**处理,
# 进而在没有 IRB / 知情同意记录时给出 NO-GO。这不是 bug,是**保守默认**:
# **拿不准就当最严的算**。k-匿名那项本身是 PASS(`treat` 只有两个取值,
# 每个等价类 160 条,`k=160 ≥ 5`),真正卡住的是 IRB 与知情同意。
#
# **正确的修法**是把单元如实声明为非人类的 `"firm"`,并补上治理事实(这是家
# 企业面板、公开监管数据、已最小化到所需变量)。重跑闸门——判决应翻成 PASS。

# %%
st.write("design", "unit", "firm")        # 如实声明:分析单元是"企业"
sv.gov.ethics_check(
    st,
    data=df,
    quasi_identifiers=["treat", "year"],  # 用组合准标识做真 k-匿名
    irb="exempt",                          # 企业监管面板:IRB 豁免
    consent="public",                      # 公开数据,无需个体知情同意
    minimized=True,                        # 已删除直接标识、只留所需变量
)
ethics = st.governance["ethics"]
print("修正后伦理判决:", ethics["verdict"], "| 人类受试者:", ethics["human_subjects"])
for c in ethics["checks"]:
    print(f"  - {c['check']:<12} {c['status']}")
print("k-匿名 k =", ethics["k_anonymity"]["k"],
      "(≥ 阈值", ethics["k_anonymity"]["k_threshold"], ")")

assert ethics["verdict"] == "PASS", "闸门未放行,不应继续分析"
print("\n闸门放行 ✓ —— 现在才被允许进入因果分析")

# %% [markdown]
# 这一小段就是治理槽的意义:**闸门是链的一部分,不是事后合规文档。** 出处台账
# 里会留下"伦理检查在因果分析之前就跑过、且判决为 PASS"的证据(第 8 步导出时可见)。

# %% [markdown]
# ## 4. 声明研究设计(把列名翻译成 design 词汇表)
#
# **为什么这步 + 契约。** `declare_design` 的契约是
# `requires=sources.datasets → produces=design.{panel_id,time,treatment,first_treated,…}`。
# 它把数据里的**具体列名**登记进 `design` 槽——下游的因果估计器读的是**槽**
# (`design['panel_id']`)而不是猜列名。声明时它还会对着已登记的数据校验列是否存在,
# 列不存在只是**警告**(写进 `design['warnings']`),不是异常。

# %%
sv.pp.declare_design(
    st,
    panel_id="firm_id",
    time="year",
    treatment="treat_post",       # 处理×时点交互(已在数据里预先算好)
    first_treated="first_treated",
)
print("design 槽现在登记的设计变量:")
for k in ("unit", "panel_id", "time", "treatment", "first_treated"):
    print(f"  design['{k}'] = {st.design.get(k)!r}")
print("列名校验警告:", st.design.get("warnings", "无(所有列都在数据中)"))

# %% [markdown]
# ## 5. 因果识别的前置门槛:平行趋势检验
#
# **为什么这步 + 契约。** `parallel_trends` 的契约要求
# `design.{panel_id,time,treatment,first_treated} + variables.outcome + estimand.target`,
# 产出 `diagnostics.pretrend` 与 **`identification.parallel_trends`**。后者是关键——
# 它是 `did` 的一个 `requires`。也就是说:**不先跑这一步,`did` 根本无法通过契约检查。**
#
# 检验本身是真算的:估一个含单元 + 时间固定效应的事件研究,对**所有前导期系数**
# 做联合 Wald 检验。`p > 0.05`(未拒绝)→ 判定 `"pass"`。我们的玩具数据是带
# 真实平行前趋势生成的,应当通过。

# %%
sv.tl.parallel_trends(st)

pt = st.diagnostics["pretrend"]
print("平行趋势联合 Wald 检验:")
print(f"  joint_F = {pt['joint_F']:.4f}   p_value = {pt['p_value']:.4g}   (前导期数 n_pre={pt['n_pre']})")
print(f"  结论: identification.parallel_trends = {st.identification['parallel_trends']!r}")
print(f"  note: {pt['note']}")
print("\n前导期系数(应都≈0):")
for period, (coef, se) in pt["pre_coefs"].items():
    print(f"    event_time={period:>3}:  β={coef:+.4f}  (se={se:.4f})")

# %% [markdown]
# ## 6. 越级会怎样?——注册表在 `did` 之前拒绝没做检验的世界
#
# **为什么演示这步。** 契约不是文档里的建议,是**运行时闸门**。为了看清这一点,
# 我们造一个**没跑过平行趋势**的平行研究态,直接调 `did`——它应当抛
# `sv.RegistryError`,并在报错里**告诉你缺什么、谁能补**。这是"查而非猜"的反面:
# 你猜错了链路,注册表当场拦下,而不是给你一个看似正常、实则未识别的估计。

# %%
st_bad = sv.StudyState()
sv.pp.ingest(st_bad, data=df)
st_bad.write("variables", "outcome", "y")
sv.pp.declare_design(st_bad, panel_id="firm_id", time="year",
                     treatment="treat_post", first_treated="first_treated")
# 注意:故意跳过 parallel_trends —— identification.parallel_trends 尚不存在

try:
    sv.tl.did(st_bad)                     # 越级调用
    print("(不应到这里)")
except sv.RegistryError as err:
    print("RegistryError —— 注册表按契约拒绝越级:\n")
    print(err)

# %% [markdown]
# 报错里那句 `identification.parallel_trends (produced by: parallel_trends)` 正是
# grounding 的兑现:它不只说"你缺东西",还说"这东西由 `parallel_trends` 产出"——
# 于是修复路径是唯一确定的。回到我们**合规的** `st`,它已经有了平行趋势结论,
# `did` 就能顺利通过契约。

# %% [markdown]
# ## 7. 估计 ATT 并出森林图(链的末端产出)
#
# **为什么这步 + 契约。** `did` 的契约是
# `design.* + variables.outcome + **identification.parallel_trends** →
# models.{did,twfe} + diagnostics.robustness`。它估一个双向固定效应 DID,报
# **聚类稳健 SE**,并在**多种方差设定下重估**(classical / HC1 / panel-cluster)
# 写进稳健性矩阵。最妙的是:它把"平行趋势是否通过"**焊进结论**——通过则标
# "因果 ATT",没过则降级为"关联非因果"。
#
# `forest` 的契约是 `requires=models.did → produces=artifacts.figures`——**你画不出
# 一个你从没估过的系数**。

# %%
sv.tl.did(st)

m = st.models["did"]
print("DID / TWFE 估计结果:")
print(f"  ATT     = {m['att']:+.4f}   (真值注入 = -0.8000)")
print(f"  SE      = {m['se']:.4f}   (panel-cluster, {m['n_clusters']} 个聚类)")
print(f"  95% CI  = [{m['ci'][0]:.4f}, {m['ci'][1]:.4f}]")
print(f"  p 值    = {m['p']:.3g}")
print(f"  样本量  = {m['n']}   估计量 = {m['estimator']}")
print(f"  因果标注= {m['note']}   (parallel_trends={m['parallel_trends']})")

print("\n稳健性矩阵(同一 ATT 在不同 SE 设定下):")
rob = pd.DataFrame(st.diagnostics["robustness"]["specs"])
display(rob)

# %%
# forest 需要 models.did —— 上一步已产出,契约满足;图存到同目录
sv.pl.forest(st, out=figpath("fig_study.png"),
             title="研究闭环 · DID 的 ATT 点估计 ± 95% CI")
fig_meta = st.artifacts["figures"]["forest"]
print("森林图已产出:", os.path.basename(fig_meta["path"]),
      "| dpi:", fig_meta["dpi"])
print("note:", fig_meta["note"])

# %% [markdown]
# ![研究闭环:DID 的 ATT 森林图](fig_study.png)
#
# 点估计 `-0.73`,95% CI 落在负区间、不跨 0,与注入的真值 `-0.8` 相符。因为平行
# 趋势通过,这个数**被允许**读作因果 ATT——这个"被允许"是注册表给的,不是我们
# 嘴上说的。

# %% [markdown]
# ## 8. 证据链 = 出处台账(这本 notebook 的主角)
#
# **为什么这步。** 到这里研究"做完了"。但可复核研究的真正交付物,不只是那个
# `-0.73`,而是**这条链能自证它怎么来的**。`StudyState` 有一本**只增不改的出处
# 台账** `st.provenance`:每一个注册函数**成功执行**时,`@register` 包装器会自动
# 追加一条记录——函数全名、调用参数、它 requires 了哪些槽、produces 了哪些槽。
# **你没有额外写一行日志**,台账是注册表机制的副产品。
#
# 把它打出来,就是一条从"伦理放行"到"ATT 估计"的、依赖前后咬合的证据链:

# %%
print("=== PROVENANCE(证据链 / 出处台账) ===\n")
for p in st.provenance:
    fn = p["function"].split(".")[-1]
    req = ", ".join(f"{s}.{k}" for s, ks in p["requires"].items() for k in ks) or "∅"
    pro = ", ".join(f"{s}.{k}" for s, ks in p["produces"].items() for k in ks) or "∅"
    print(f"  step {p['step']}: {fn}")
    print(f"           requires ← {req}")
    print(f"           produces → {pro}")

# %% [markdown]
# **读这条链。** 每一步的 `produces` 都恰好喂饱了后面某一步的 `requires`:
# `ingest` 产出 `sources.datasets` → `declare_design` 要它;`declare_design` 产出
# `design.*` → `parallel_trends` 要它;`parallel_trends` 产出
# `identification.parallel_trends` → **`did` 要它**。**依赖前后咬合、无缺环**——
# 这就是“可复核”的机器可读形态。注意 `ethics_check` **出现两次**(第 2 步的
# 误声明 + 第 3 步的修正)——台账**只增不改**,连“先犯错再修正”都如实留痕,
# 这恰恰是可复核性该有的样子。而且伦理检查排在因果分析之前,证明
# **治理不是事后补的**。
#
# 我们把它整理成一张表(它本身就能作为论文附录的"分析出处"):

# %%
prov_rows = []
for p in st.provenance:
    prov_rows.append({
        "step": p["step"],
        "function": p["function"].split(".")[-1],
        "requires": ", ".join(f"{s}.{k}" for s, ks in p["requires"].items() for k in ks) or "∅",
        "produces": ", ".join(f"{s}.{k}" for s, ks in p["produces"].items() for k in ks) or "∅",
    })
prov_df = pd.DataFrame(prov_rows)
display(prov_df)

# 出处台账可直接导出为 JSON,随论文一起存档(可复核的证据附件)
prov_json_path = figpath("evidence_chain_provenance.json")
with open(prov_json_path, "w", encoding="utf-8") as fh:
    json.dump(st.provenance, fh, ensure_ascii=False, indent=2)
print("出处台账已导出:", os.path.basename(prov_json_path),
      f"({len(st.provenance)} 步)")

# %% [markdown]
# ## 9. 导出注册表清单(manifest):人可读的目录 / OmicOS 可消费
#
# **为什么这步。** 证据链回答"**这次研究**做了什么";注册表清单回答"**这套工具**
# 能做什么、每个能力的契约是什么"。`sv.registry.manifest()` 把整个注册表 dump 成
# 一份 JSON:12 槽词汇表 + 每个函数的完整契约(requires/produces/prerequisites/
# tier/skill/backend)。这正是 OmicOS 的 `registry_lookup` 消费的形状——**同一份
# 清单,人能读、agent 能查**。

# %%
manifest = sv.registry.manifest()
print("registry manifest:")
print("  函数总数:", manifest["count"])
print("  类别(", len(manifest["categories"]), "):", ", ".join(manifest["categories"]))
print("  槽词汇表(", len(manifest["slots"]), "槽):")
for slot, meaning in list(manifest["slots"].items())[:4]:
    print(f"    - {slot:<15} {meaning}")
print("    …(共 12 槽)")

# 抽一条契约看清 manifest 的粒度(以本链核心的 did 为例)
did_entry = next(f for f in manifest["functions"] if f["name"] == "did")
print("\nmanifest 里 did 这一条(节选):")
for key in ("full_name", "category", "tier", "skill", "requires", "produces",
            "prerequisites", "auto_fix", "key_tools"):
    print(f"  {key:<14}: {did_entry[key]}")

# %%
# 把清单落盘成机器可读目录(与出处台账并列,一份"能力目录",一份"这次做了什么")
manifest_path = figpath("registry_manifest.json")
blob = sv.registry.export_registry(manifest_path)
print("registry manifest 已导出:", os.path.basename(manifest_path),
      f"({len(blob):,} 字节,{manifest['count']} 个函数契约)")

# 按类别数一下能力分布,给读者一张"这套工具的版图"
by_cat = sv.registry.list_functions()
print("\n各类别函数数(能力版图):")
for cat in sorted(by_cat, key=lambda c: -len(by_cat[c])):
    names = sorted(f.split(".")[-1] for f in by_cat[cat])
    print(f"  [{cat:<11}] {len(names):>2} :: {', '.join(names)}")

# %% [markdown]
# ## 10. 收束:研究态的最终快照 + 这条链的差异化
#
# `st.summary()` 把整份研究底稿一览:哪些槽被填了、填了什么键、出处几步。
# 这就是一次**可复核研究**跑完后应该留下的东西——不是一个孤零零的数字,而是
# **数据来源 → 治理放行 → 设计声明 → 识别假设检验 → 估计 → 出图**的一条闭环,
# 外加两份可存档的机器可读文件(出处台账 + 注册表清单)。

# %%
print(st.summary())

print("\n本次研究导出的可复核证据文件:")
for pth in (figpath("fig_study.png"), prov_json_path, manifest_path):
    exists = "✓" if os.path.exists(pth) else "✗"
    print(f"  [{exists}] {os.path.basename(pth)}")

# %% [markdown]
# ---
# ### 这条链对标的现实工具 + socialverse 的差异
#
# **对标物。** 拆开看,每一步都有成熟对应物:治理闸门 ≈ IRB 表单 + `sdcMicro`
# 的 k-匿名;因果估计 ≈ **`pyfixest` / R `fixest`** 与 **`did`(Callaway–
# Sant'Anna)**;出图 ≈ `matplotlib` / `coefplot`;而"把它们串成可复现流水线"这一
# 层最接近 **`targets`(R)/ `snakemake`**,出处追踪层接近 **W3C PROV / DataLad**。
#
# **差异(socialverse 的收束点)。** 现实工具给你**估计量**;把估计量变成一次
# **可复核的研究**,通常要你**额外**手写流水线定义、手写出处日志、手写"我确实先
# 做了平行趋势"的说明。`socialverse` 的不同在于这些**不是额外工作**,而是注册表
# 机制的内建产物:
#
# 1. **grounding(查而非猜)** —— `resolve_plan` / `get_prerequisites` 让你在写代码
#    前就把链路和缺口问清楚,链路是**对着当前研究态**算的,不是背模板;
# 2. **契约即闸门** —— `did` 的 `requires` 把"平行趋势已检验"写成机器可查的依赖,
#    越级调用当场抛 `RegistryError` 并告诉你**谁能补**,而不是给你一个未识别的估计;
# 3. **证据链自动生成** —— `st.provenance` 是每次成功调用的副产品,依赖前后咬合、
#    只增不改,直接导出即为论文附录级的"分析出处";
# 4. **能力清单可消费** —— `manifest()` 把整套契约 dump 成同一份 JSON,人能读、
#    OmicOS 的 `registry_lookup` 能查——**同一张注册表既驱动分析,又自证能力**。
#
# 一句话:别的工具让你**做出**一个结果,`socialverse` 让这个结果**自己带着**它
# 是怎么来的、依赖什么、是否越级——这就是"世界第一的根是注册表(而非数据容器)"
# 在社科研究闭环里的兑现。
