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
# # 08 · 研究治理:伦理闸门 + 数据合规 + AI 使用披露
#
# **这条分析链讲什么。** 在生物信息里,合规大多是外挂的清单;在社会科学里,**治理是硬需求、
# 是一等公民**——一份微数据能不能分析、能不能分享、稿件里生成式 AI 的使用怎么交代,这些不是
# 「事后补材料」,而是决定研究**能不能进行**的前置闸门。本 notebook 走完 socialverse 的三道
# 治理宏观闸门(全部是注册表里带契约的一等公民函数):
#
# 1. **伦理闸门 `ethics_check`** —— IRB 分类 / 知情同意 / **可识别性(真实 k-匿名计算)** / 数据
#    最小化,折叠成一个 `PASS / FIX / NO-GO` 判决。我们先看一个 **NO-GO** 的裸状态,再一步步
#    **补救**(记录 IRB/同意 + 粗化准标识符)把它抬到 **PASS**,并展示一个 **k=1 直接可识别** 的
#    红线案例。
# 2. **数据合规 `data_use_check`** —— 逐源版权/许可**五桶分诊**(公有域 / CC / 出版商 TDM /
#    GLAM / 平台 ToS),给出抓取与再分发决策;展示 **UNKNOWN → 最严桶** 的默认,以及多源
#    **最弱环**(weakest-link)权利求交。
# 3. **AI 使用披露 `ai_use_disclosure`** —— 审计 AI 贡献日志的**「已采纳但未核验」红线**,并按
#    目标期刊政策族(ICMJE / COPE / …)渲染一段**可直接粘贴**的披露声明。
#
# **涉及的函数(全部先查注册表,再调用)。**
#
# | 闸门 | 函数 | requires → produces |
# |---|---|---|
# | 伦理 | `sv.gov.ethics_check` | `design[unit]` → `governance[ethics]` |
# | 数据合规 | `sv.gov.data_use_check` | `sources[datasets]` → `governance[data_use]` |
# | AI 披露 | `sv.gov.ai_use_disclosure` | `∅`(无前置)→ `governance[ai_disclosure]`, `artifacts[tables]`, `evidence[provenance]` |
#
# **`StudyState` 会被填的槽:** `sources` · `design` · `governance`(`ethics` / `data_use` /
# `ai_disclosure` 三个治理键)· `artifacts`(AI 日志审计表 + 治理仪表盘图)· `evidence`
# (provenance)。每个 `requires`/`produces` 都用这 12 槽词汇表书写——正是这套契约让治理**可被
# 机器检查、可被 `resolve_plan` 反推、可自动累积证据链**。
#
# **对标的现实工具。** 这三道闸门在现实里通常分散在**不同工具/人手**里:IRB 用机构的
# eProtocol / IRBNet 表单,k-匿名靠 **ARX / sdcMicro / `pandas` 手搓**,许可分诊靠研究者读
# rightsstatements.org + 期刊 TDM 条款,AI 披露靠照抄 **ICMJE / COPE** 的模板。socialverse 的
# 差异在于:这是社会科学域**特有的一等公民治理轴**——三道闸门是注册表里**带 `requires/produces`
# 契约的可查询函数**,和 DID、复杂抽样、质性编码等分析函数**平级**,能被同一个 `resolve_plan`
# 编排、把判决与证据自动写进 `governance` / `evidence` 槽,形成可审计的合规证据链。

# %% [markdown]
# ## 0 · 环境与「查而非猜」
#
# 固定 matplotlib 为 Agg 后端(无窗口、内核/CI 安全),导入 socialverse,并提供一个极简
# `display()`(脱离 Jupyter 当普通脚本跑时也不报错)。
#
# **关键姿势:先查注册表。** socialverse 的设计论点是「让 agent 可靠的不是统一的数据容器,
# 而是带显式依赖标注的**可查询函数注册表**」。所以在写任何调用前,先看目录里治理这条**横切**
# 链长什么样——它与生物域调 `ov.utils.registry_lookup` 是同一套输出格式。

# %%
import matplotlib
matplotlib.use("Agg")  # 必须在 pyplot 被任何地方 import 之前设定

import json
import os

import pandas as pd

import socialverse as sv
from socialverse import datasets as ds

# 图保存到 notebook 同目录(无论从哪个 cwd 运行都对得上 ![](fig.png) 相对引用)。
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # 交互式 Jupyter 里没有 __file__
    _HERE = os.getcwd()

def _fig(name):
    return os.path.join(_HERE, name)

try:  # 在真正的 Jupyter 里用富显示;当普通脚本跑时回退到 print
    from IPython.display import display
except Exception:  # pragma: no cover
    def display(obj):
        print(obj)

print("socialverse version:", sv.__version__)
print()
print(sv.utils.registry_summary())

# %% [markdown]
# 目录末尾把 `governance` 单列为一条**横切(cross-cutting)** 链:
# `data_use_check · ethics_check · redact_pii · ai_use_disclosure`。这与分析链不同——治理不是
# 数据流水线上的一环,而是**贯穿始终、随时可插入**的闸门。我们逐个查它们的契约,尤其注意
# `ethics_check` 的 `Requires: design['unit']`——这条契约稍后会**真的**在运行时被强制执行。

# %%
print(sv.utils.registry_lookup("伦理", max_results=1))
print()
print(sv.utils.registry_lookup("数据合规", max_results=1))
print()
print(sv.utils.registry_lookup("AI披露", max_results=1))

# %% [markdown]
# ## 1 · 契约是活的:未满足 `requires` 会抛 `RegistryError`
#
# 在填任何数据之前,我们**故意**去调 `ethics_check`。它的契约要求 `design['unit']`(分析单位:
# 个体?家户?国家?——这决定了「是不是人类受试者」),而空的 `StudyState` 里没有——于是抛
# `RegistryError`。这不是 bug,而是**特性**:契约不是死的元数据,而是每次调用时被强制的守卫
# (omicverse 的 `valid_keys` 机制移植到社科)。错误信息还顺带告诉你**谁能生产**这个缺失槽
# (`declare_design`),这就是 grounding:报错本身指向修复路径。

# %%
empty = sv.StudyState()
try:
    sv.gov.ethics_check(empty, quasi_identifiers=["strata"])
except sv.RegistryError as err:
    print("RegistryError(如预期,ethics_check 缺 design['unit']):\n")
    print(err)

print("\n" + "─" * 60 + "\n")

# data_use_check 的契约要求 sources['datasets'] —— 没登记数据源就不能做许可分诊
try:
    sv.gov.data_use_check(sv.StudyState(), license="CC-BY-4.0")
except sv.RegistryError as err:
    print("RegistryError(如预期,data_use_check 缺 sources['datasets']):\n")
    print(err)

# %% [markdown]
# 与其猜「治理闸门前应该先做什么」,不如让注册表**反推整条计划**。
# `resolve_plan('ethics_check')` 沿依赖图回溯:要跑伦理闸门得先有 `design[unit]`(由
# `declare_design` 生产),而 `declare_design` 又要 `sources[datasets]`(由 `ingest` 生产)……
# 最终排出有序 `plan`。注意 `escalations`:因为这些自动插入步骤下游 `auto_fix=escalate`,注册表
# **不会替你静默补齐**,而是**升级给人确认**——治理链尤其需要「人在环」,这正是社科合规的姿态。

# %%
plan = sv.registry.resolve_plan("ethics_check")
print(json.dumps(plan, ensure_ascii=False, indent=2))

# %% [markdown]
# `ai_use_disclosure` 则是三道闸门里唯一 **`requires` 为空** 的——它不依赖任何数据槽,随时可跑,
# 但它 `produces` 三样东西:`governance[ai_disclosure]`(声明+审计)、`artifacts[tables]`(日志
# 审计表)、`evidence[provenance]`(证据戳)。用 `get_prerequisites` 把契约看清楚。

# %%
print(json.dumps(sv.registry.get_prerequisites("ai_use_disclosure"), ensure_ascii=False, indent=2))

# %% [markdown]
# ## 2 · 登记数据源与分析单位(满足两道闸门的 `requires`)
#
# **为什么这步。** 治理闸门要在**真实微数据**上判决,不能空谈。我们加载玩具**调查数据**
# (300 行,含 `strata`(分层)、`psu`(初级抽样单元)、态度量表 `item1..item6`、`weight`、
# `exposure`、`outcome`),把它写进 `sources['datasets']`,并声明分析单位 `design['unit']='row'`
# (受访者个人 = 人类受试者)。这两笔写入分别满足 `data_use_check` 与 `ethics_check` 的 `requires`。

# %%
df = ds.load_survey()
print("调查数据 shape:", df.shape)
display(df.head())
print("\nstrata 取值:", sorted(df["strata"].unique().tolist()),
      " · psu 取值数:", df["psu"].nunique())

st = sv.StudyState()
st.write("sources", "datasets", df)     # 满足 data_use_check.requires
st.write("design", "unit", "row")        # 满足 ethics_check.requires(row = 受访者 = 人类受试者)
print("\n已填槽:", st.populated())

# %% [markdown]
# ## 3 · 伦理闸门(第一版:裸状态 → NO-GO)→ `ethics_check`
#
# **为什么这步。** 伦理闸门跑四项检查并折叠成一个判决:
#
# 1. **IRB** —— 人类受试者分类(exempt / expedited / full),以及是否已记录裁定;
# 2. **知情同意** —— 分析单位的同意基础(informed / waiver / public / none);
# 3. **可识别性** —— 一个**真实的 k-匿名计算**:`k = df.groupby(准标识符).size().min()`,即每条
#    记录与至少 `k−1` 个他人共享同一准标识符组合,没有任何组合能把人缩到少于 `k`;
# 4. **数据最小化** —— 是否已删除直接标识符 / 只保留所需变量。
#
# **契约。** `requires design['unit']` → `produces governance['ethics']`,`auto_fix=escalate`
# (FIX/NO-GO 都要人工复核才能继续)。第一版我们**什么都不声明**,只把 `strata+psu` 当准标识符
# 交给它算 k——看它如实报出一个 **NO-GO**。

# %%
sv.gov.ethics_check(st, data=df, quasi_identifiers=["strata", "psu"])  # k_threshold 默认 5

ethics_v1 = st.governance["ethics"]
print("伦理判决 verdict:", ethics_v1["verdict"])
print("\n四项检查:")
for c in ethics_v1["checks"]:
    print(f"  [{c['status']:<5}] {c['check']:<12} — {c['detail']}")

print("\nk-匿名细节(真实计算,非占位):")
print(json.dumps(ethics_v1["k_anonymity"], ensure_ascii=False, indent=2))

# %% [markdown]
# 判决是 **NO-GO**,由 `_verdict` 规则聚合:任一 `NO-GO` → 整体 `NO-GO`。这里 IRB 与同意都是
# `NO-GO`(有人类受试者却没记录裁定/同意),k-匿名是 `FIX`(k=2 < 阈值 5:每个 `(strata,psu)`
# 组合最小只有 2 人,需要粗化)。**这正是治理作为一等公民的价值**:在你写第一行分析代码之前,
# 闸门就把「不能上」的理由**结构化地**摆出来了,而不是等审稿人或 IRB 事后打回。下一步我们把这些
# 都补救掉。

# %% [markdown]
# ## 4 · 补救伦理闸门(第二版:粗化 QI + 记录 IRB/同意 → PASS)
#
# **为什么这步。** NO-GO 不是终点,而是一张**待办清单**。我们逐项补救:
#
# - **IRB**:记录裁定 `irb="exempt"`(如「已获豁免类审查」);
# - **同意**:记录基础 `consent="informed"`;
# - **可识别性**:把准标识符从 `strata+psu` **粗化**为只保留 `strata`——这是 k-匿名的标准补救
#   (泛化/抑制准标识符),让最小等价类从 2 抬到几十;
# - **最小化**:`minimized=True`(已确认删除直接标识符)。
#
# 这四项都是研究者的**真实治理决定**,通过 kwargs 声明给闸门。闸门重算后应给出 **PASS**。

# %%
st_fixed = sv.StudyState()
st_fixed.write("sources", "datasets", df)
st_fixed.write("design", "unit", "row")

sv.gov.ethics_check(
    st_fixed,
    data=df,
    quasi_identifiers=["strata"],   # 粗化:丢掉 psu(k-匿名的泛化补救)
    irb="exempt",                   # 记录 IRB 裁定
    consent="informed",             # 记录同意基础
    minimized=True,                 # 已删除直接标识符
    k_threshold=5,
)

ethics_v2 = st_fixed.governance["ethics"]
print("补救后 verdict:", ethics_v2["verdict"])
print("\n四项检查:")
for c in ethics_v2["checks"]:
    print(f"  [{c['status']:<5}] {c['check']:<12} — {c['detail']}")

print(f"\nk-匿名:粗化前 (strata+psu) k={ethics_v1['k_anonymity']['k']}"
      f" → 粗化后 (strata) k={ethics_v2['k_anonymity']['k']}"
      f"(≥ 阈值 5,PASS)")

# %% [markdown]
# ### 4.1 红线案例:k=1 直接可识别 → 硬 NO-GO
#
# k-匿名的极端是 **k=1**:存在只属于一个人的准标识符组合——这个人**可被直接重识别**。我们人为
# 造一列唯一行号 `resp_id` 当准标识符,闸门会把它判成 **NO-GO**(而非可修的 FIX),因为
# 「300 个单例记录」是不能靠泛化轻易补救的结构性泄露。这演示了闸门对**re-identifiability 分级**
# 的判断力(k≥阈值=PASS / 1<k<阈值=FIX / k≤1=NO-GO)。

# %%
df_leak = df.copy()
df_leak["resp_id"] = range(len(df_leak))   # 唯一直接标识符

st_leak = sv.StudyState()
st_leak.write("sources", "datasets", df_leak)
st_leak.write("design", "unit", "row")
sv.gov.ethics_check(st_leak, data=df_leak, quasi_identifiers=["resp_id"],
                    irb="exempt", consent="informed", minimized=True)

leak = st_leak.governance["ethics"]
k_check = next(c for c in leak["checks"] if c["check"] == "k_anonymity")
print("verdict:", leak["verdict"], "· k =", leak["k_anonymity"]["k"],
      "· 单例记录数:", leak["k_anonymity"]["n_unique_records"])
print("k-匿名检查:", k_check["status"], "—", k_check["detail"])

# %% [markdown]
# ## 5 · 数据合规:许可五桶分诊 → `data_use_check`
#
# **为什么这步。** 「这份数据我到底能不能抓、能不能再分发」在社科里是**逐源**的法律/伦理判断。
# `data_use_check` 把每个来源分诊进**五个桶**(严→宽):`platform_tos`(平台 ToS)/
# `publisher_tdm`(出版商文本数据挖掘许可)/ `glam`(图书馆档案馆)/ `cc`(知识共享)/
# `public_domain`(公有域),从许可字符串推出 `can_scrape` / `redistribution` / `attribution`
# 决策与义务标记(NC / ND / SA)。**关键默认:许可未知 → 落到最严桶**(`platform_tos`),即
# 「无证据 = 不假定有权」。
#
# **契约。** `requires sources['datasets']` → `produces governance['data_use']`。先做单源:一个
# 干净的 `CC-BY-4.0`。

# %%
sv.gov.data_use_check(st_fixed, license="CC-BY-4.0")

du = st_fixed.governance["data_use"]
print("单源 (CC-BY-4.0) 分诊:")
print("  桶:", du["bucket"], "· 可抓取:", du["can_scrape"],
      "· 再分发:", du["redistribution"], "· 需署名:", du["attribution"])
print("  per_source[0].note:", du["per_source"][0]["note"])

# %% [markdown]
# ### 5.1 UNKNOWN → 最严桶(不假定有权)
#
# 传一个**空**许可串,看闸门把它落到 `platform_tos`(最严),`can_scrape=False`、
# `redistribution=prohibited`,并打上一条明确的 flag:先厘清权利再抓取或分享。这是治理里
# **safe-by-default** 的姿态——UNKNOWN 不等于「随便用」。

# %%
st_unknown = sv.StudyState()
st_unknown.write("sources", "datasets", df)
sv.gov.data_use_check(st_unknown, license="")   # 许可未知

du_u = st_unknown.governance["data_use"]
print("桶:", du_u["bucket"], "· 可抓取:", du_u["can_scrape"],
      "· 再分发:", du_u["redistribution"])
print("flags:", json.dumps(du_u["flags"], ensure_ascii=False, indent=2))

# %% [markdown]
# ### 5.2 多源「最弱环」求交(weakest-link)
#
# 真实项目往往**混合多个来源**。闸门对多源做**逐源分诊**,然后取**权利的交集**:只要有一个源
# 不能抓,整盘 `can_scrape=False`;再分发权利取**最严的那一档**(prohibited < derived_only <
# conditional < share_alike_or_by < unrestricted);NC/ND/SA 义务标记**并集**上浮。下面混四个源:
# 公有域普查、Twitter 平台 ToS、出版商 TDM、CC-BY-NC-SA 图像——看整盘怎么被最弱环拉低。

# %%
st_multi = sv.StudyState()
st_multi.write("sources", "datasets", df)
sv.gov.data_use_check(st_multi, license={
    "census_pums": "Public Domain (US government work)",
    "twitter_api": "Twitter Developer Policy / platform ToS",
    "journal_tdm": "Publisher text-and-data-mining licence",
    "cc_images":   "CC-BY-NC-SA 4.0",
})

du_m = st_multi.governance["data_use"]
print("逐源分诊:")
per_source_df = pd.DataFrame([
    {"source": t["source"], "bucket": t["bucket"],
     "can_scrape": t["can_scrape"], "redistribution": t["redistribution"]}
    for t in du_m["per_source"]
])
display(per_source_df)

print("\n整盘(最弱环求交):")
print("  涉及的桶:", du_m["bucket"])
print("  can_scrape(全部可抓才为真):", du_m["can_scrape"])
print("  redistribution(取最严):", du_m["redistribution"])
print("  义务标记(并集):")
for f in du_m["flags"]:
    print("   -", f)

# %% [markdown]
# ## 6 · AI 使用披露 → `ai_use_disclosure`
#
# **为什么这步。** 期刊现在普遍要求交代生成式 AI 的使用(ICMJE / COPE / Nature 各有政策)。
# 但真正的**研究诚信红线**不是「用没用 AI」,而是**「已采纳但未核验」**——把 AI 生成的内容
# 直接写进稿件却没人核对过。`ai_use_disclosure` 审计一份**逐阶段 AI 使用日志**,专门抓这条红线,
# 并按目标期刊政策族渲染一段**可直接粘贴**的披露声明。
#
# **契约。** `requires ∅`(随时可跑)→ `produces governance['ai_disclosure']` +
# `artifacts['tables']`(日志审计表)+ `evidence['provenance']`(证据戳)。第一版:一份**含红线**
# 的日志(分析阶段 accepted 但 unverified),政策族 ICMJE。

# %%
sv.gov.ai_use_disclosure(
    st_fixed,
    ai_log=[
        {"stage": "analysis", "tool": "LLM", "accepted": True,  "verified": False},  # 红线!
        {"stage": "drafting", "tool": "LLM", "accepted": True,  "verified": True},
    ],
    policy="ICMJE",
)

disc = st_fixed.governance["ai_disclosure"]
print("审计状态:", disc["audit"]["status"], "·", disc["audit"]["detail"])
print("政策族:", disc["policy"], "→", disc["policy_family"])
print("\n『已采纳但未核验』红线命中:")
for u in disc["audit"]["unverified"]:
    print("  -", u["stage"], "/", u["tool"], "→ flag:", u["flag"])

print("\nartifacts['tables'](日志审计表,可进稿件附录):")
display(st_fixed.artifacts["tables"])

print("\n可直接粘贴的披露声明(ICMJE 政策族):")
print(" ", disc["statement"])

# %% [markdown]
# ### 6.1 政策族切换 + 全核验 PASS
#
# 同一份日志换成**全部已核验**、政策族换成 **COPE**——审计从 `ESCALATE` 降为 `PASS`,声明也换成
# COPE 措辞。这演示了 `ai_use_disclosure` 的两个维度:**红线审计**(诚信)与**政策族渲染**(格式)。

# %%
st_clean = sv.StudyState()
sv.gov.ai_use_disclosure(
    st_clean,
    ai_log=[
        {"stage": "copyediting", "tool": "Claude", "accepted": True, "verified": True},
        {"stage": "code review", "tool": "Claude", "accepted": True, "verified": True},
    ],
    policy="COPE",
)
disc_c = st_clean.governance["ai_disclosure"]
print("审计状态:", disc_c["audit"]["status"], "·", disc_c["audit"]["detail"])
print("政策族:", disc_c["policy"], "→", disc_c["policy_family"])
print("\n可直接粘贴的披露声明(COPE 政策族):")
print(" ", disc_c["statement"])

# %% [markdown]
# ## 7 · 治理仪表盘(一图看三道闸门)
#
# **为什么这步。** 把三道闸门的关键判据画成一张仪表盘,让审稿人/合作者一眼看清合规态势:
# (左)k-匿名如何随准标识符**粗化**而改善、越过阈值转绿;(右)多源许可**桶**的抓取/再分发
# 权利。我们复用 socialverse 图形模块里的 `_cjk_fonts()`,让中文标签在装了 CJK 字体时正常渲染、
# 否则优雅回退(不报错)。图存成同目录 PNG,并把元信息写进 `artifacts['figures']`。

# %%
import matplotlib.pyplot as plt

# 复用包内的 CJK 字体探测(与 sv.pl 一致):有则用,无则回退 DejaVu(不报错)
try:
    from socialverse.pl._figure import _cjk_fonts
    plt.rcParams["font.sans-serif"] = _cjk_fonts() + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:  # pragma: no cover
    pass

# 左图数据:k 随 QI 粗化的阶梯(真实重算)
qi_ladder = [("strata+psu", ["strata", "psu"]), ("psu", ["psu"]), ("strata", ["strata"])]
ks = []
for _, qi in qi_ladder:
    _s = sv.StudyState(); _s.write("sources", "datasets", df); _s.write("design", "unit", "row")
    sv.gov.ethics_check(_s, data=df, quasi_identifiers=qi,
                        irb="exempt", consent="informed", minimized=True)
    ks.append(_s.governance["ethics"]["k_anonymity"]["k"])

# 右图数据:多源许可桶的再分发权利等级(0=禁止 … 4=无限制)
redist_rank = {"prohibited": 0, "derived_only": 1, "conditional": 2,
               "share_alike_or_by": 3, "unrestricted": 4}
sources = du_m["per_source"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# --- 左:k-匿名粗化阶梯 ---
labels = [lbl for lbl, _ in qi_ladder]
colors = ["#c0392b" if k < 5 else "#27ae60" for k in ks]
bars = ax1.bar(labels, ks, color=colors, edgecolor="#222", linewidth=0.6)
ax1.axhline(5, ls="--", color="#333", lw=1.2)
ax1.text(len(labels) - 0.5, 5.6, "k=5 阈值", ha="right", fontsize=9, color="#333")
for b, k in zip(bars, ks):
    ax1.text(b.get_x() + b.get_width() / 2, k + 1.5, str(k), ha="center", fontsize=11, fontweight="bold")
ax1.set_ylabel("k-匿名(最小等价类规模)")
ax1.set_xlabel("准标识符集合(自左向右逐步粗化)")
ax1.set_title("① 伦理闸门:k-匿名随 QI 粗化而改善")
ax1.set_ylim(0, max(ks) * 1.25)
ax1.grid(axis="y", alpha=0.25)

# --- 右:许可桶再分发权利 ---
names = [t["source"] for t in sources]
ranks = [redist_rank.get(t["redistribution"], 0) for t in sources]
scrape_ok = [t["can_scrape"] for t in sources]
bcolors = ["#27ae60" if ok else "#c0392b" for ok in scrape_ok]
y = range(len(names))
ax2.barh(list(y), ranks, color=bcolors, edgecolor="#222", linewidth=0.6)
ax2.set_yticks(list(y))
ax2.set_yticklabels(names)
ax2.set_xticks(list(redist_rank.values()))
ax2.set_xticklabels(["禁止", "仅衍生", "有条件", "署名/相同", "无限制"], rotation=20, fontsize=8)
ax2.set_xlabel("再分发权利(越右越宽松)")
ax2.set_title("② 数据合规:多源许可桶(绿=可抓取)")
for i, t in enumerate(sources):
    ax2.text(ranks[i] + 0.06, i, t["bucket"], va="center", fontsize=8, color="#333")
ax2.set_xlim(0, 4.8)
ax2.grid(axis="x", alpha=0.25)

fig.suptitle("研究治理仪表盘:伦理闸门 + 数据合规", fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(_fig("fig_governance.png"), dpi=150, bbox_inches="tight")  # PNG → bbox tight
plt.close(fig)

# 把图元信息也记进 StudyState 的 artifacts 槽(与 sv.pl 的产物形态一致)
st_fixed.write("artifacts", "figures", {
    "governance_dashboard": {"path": _fig("fig_governance.png"), "dpi": 150,
                             "note": "k-匿名粗化阶梯 + 多源许可桶再分发权利"}
})
print("已保存图:", st_fixed.artifacts["figures"]["governance_dashboard"]["path"])
print("k 阶梯 (strata+psu → psu → strata):", ks)

# %% [markdown]
# ![研究治理仪表盘](fig_governance.png)

# %% [markdown]
# ## 8 · 证据链:`st.summary()` 与 provenance ledger
#
# 最后展示这条治理链自带的**可复现审计轨迹**。`populated()` 显示每个被填的槽;provenance
# ledger 逐条记下每一步的 `function / requires / produces`——三道闸门的判决因此**可溯源**:
# 任一「能不能上/能不能分发/怎么披露」的结论,都能顺着 ledger 回到它写入的 `governance` 键、
# 再回到它读的 `design[unit]` / `sources[datasets]` / AI 日志。这就是社科里一等重要的
# 「治理证据脊」。

# %%
print(st_fixed.summary())

print("\ngovernance 槽的三道闸门判决(一站式合规回执):")
print("  ethics.verdict       :", st_fixed.governance["ethics"]["verdict"])
print("  data_use.bucket      :", st_fixed.governance["data_use"]["bucket"],
      "· can_scrape:", st_fixed.governance["data_use"]["can_scrape"])
print("  ai_disclosure.status :", st_fixed.governance["ai_disclosure"]["audit"]["status"])

print("\nprovenance ledger(每步的契约,按执行序):")
for r in st_fixed.provenance:
    req = ", ".join(f"{s}[{','.join(k)}]" for s, k in r["requires"].items()) or "∅"
    pro = ", ".join(f"{s}[{','.join(k)}]" for s, k in r["produces"].items()) or "∅"
    print(f"  step {r['step']}: {sv.utils._friendly(r['function'])}")
    print(f"         requires {req}")
    print(f"         produces {pro}")

# %% [markdown]
# ## 小结:对标的现实工具 + socialverse 的差异
#
# 这条链对标现实里**分散在多处**的合规实践:IRB 用机构 eProtocol/IRBNet 表单;k-匿名靠 **ARX /
# sdcMicro** 或 `pandas` 手搓;许可分诊靠人读 rightsstatements.org + 期刊 TDM 条款;AI 披露照抄
# **ICMJE / COPE** 模板。
#
# **socialverse 的差异**在两点,都源自「治理是一等公民 + 注册表是脊柱」这一设计:
#
# 1. **治理是注册表里的一等公民(与分析函数平级)。** 三道闸门是带 `requires/produces` 契约的
#    可查询函数:`ethics_check` 未满足 `design[unit]` 会抛 `RegistryError` 并指向 `declare_design`;
#    整条治理链能被 `resolve_plan` 从目标反推,且因为 `auto_fix=escalate`,注册表**不会静默替你
#    补齐**而是**升级给人确认**——这正是社科合规「人在环」的姿态。现实工具是彼此割裂的表单与脚本,
#    步骤间无机器可读契约。
# 2. **判决写进证据链(可审计、safe-by-default)。** 每道闸门把结构化判决写入 `governance` 槽、
#    把审计与声明写入 `evidence` / `artifacts`,provenance ledger 自动累积;k-匿名是**真实计算**
#    (`groupby().size().min()`,分 PASS/FIX/NO-GO 级),许可 UNKNOWN **默认落最严桶**、多源取
#    **最弱环**,AI 披露专抓**「已采纳但未核验」红线**。于是每个合规结论都能一路溯源:
#    verdict → 它读的槽 → 原始数据 / 日志——形成可放进稿件与 IRB 材料的**治理证据脊**。
