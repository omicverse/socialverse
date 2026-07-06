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
# # 05 · 质性研究:去标识 → 反身主题编码 → 引语溯源
#
# **这条分析链讲什么。** 我们拿一小批访谈片段(含真实 PII),走完一条完整的
# *计算辅助质性研究* 管道:先把语料切成**可寻址的编码单元**,再**去标识**(PII 假名化,
# 保留可逆 crosswalk),然后做 Braun & Clarke 六阶段里的**反身主题编码**(编码台账 +
# 主题地图),最后为每个主题建立**论断⇄引语双向溯源索引**(offset 溯源戳 + 原文 slice
# 回校),并把研究者的解释轨迹结构化为**反身备忘**。全程每一步都在 `StudyState` 上留下
# provenance,链末拿到一条可审计的证据链。
#
# **涉及的函数(全部先查注册表,再调用)。**
#
# | 阶段 | 函数 | requires → produces |
# |---|---|---|
# | `sv.pp` | `build_corpus` | `sources[corpora]` → `corpus[documents,units,manifest]` |
# | `sv.pp` | `redact_pii` | `corpus[documents]` → `corpus[documents]`, `governance[pii_status]` |
# | `sv.tl` | `code_themes` | `corpus[units]` → `codes[codebook,segments,themes,theme_map]`, `evidence[claim_evidence]` |
# | `sv.tl` | `trace_quotes` | `corpus[units]` + `codes[segments]` → `evidence[quote_index]` |
# | `sv.tl` | `reflexive_memo` | `codes[themes]` + `corpus[units]` → `evidence[provenance]`, `governance[ethics]` |
# | `sv.pl` | `theme_map` | `codes[theme_map]` → `artifacts[figures]` |
#
# **`StudyState` 会被填的槽:** `sources` · `corpus` · `codes` · `evidence` ·
# `governance` · `artifacts`(每一步的 `requires`/`produces` 都以这 12 槽词汇表书写,
# 这正是让依赖图可被机器检查、让 agent「查而非猜」的机制)。
#
# **对标的现实工具。** 这条链对标 **CAQDAS**(NVivo / ATLAS.ti / 开源的
# [QualCoder](https://github.com/ccbogel/QualCoder))的编码—检索—可视化工作流,
# 方法论骨架是 **Braun & Clarke (2006/2019) 反身主题分析**六阶段;去标识部分对标
# spaCy / Presidio 式 PII 检测。socialverse 的差异在于:CAQDAS 是**点选式 GUI、无
# 契约**,而这里每个动作都是**注册表里带 `requires/produces` 契约的函数**——链的顺序
# 可被 `resolve_plan` 反推,证据链由 provenance ledger 自动累积,可溯源到**字符偏移**。

# %% [markdown]
# ## 0 · 环境与「查而非猜」
#
# 先固定 matplotlib 为 Agg 后端(无窗口、CI/内核安全),再导入 socialverse。
# 我们提供一个极简 `display()`(脱离 Jupyter 当普通脚本跑时也不报错)。
#
# **关键姿势:先查注册表。** socialverse 的设计论点是「让 agent 可靠的不是统一的数据容器,
# 而是带显式依赖标注的**可查询函数注册表**」。所以在写任何调用前,我们先用
# `sv.utils.registry_summary()` 看目录,用 `registry_lookup(query)` 查某个技能的契约——
# 这与在生物域调 `ov.utils.registry_lookup` 是同一套输出格式。

# %%
import matplotlib
matplotlib.use("Agg")  # 必须在 pyplot 被任何地方 import 之前设定

import json
import os

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
# 目录里 `qualitative` 链清清楚楚写着推荐顺序:
# `build_corpus → redact_pii → code_themes → trace_quotes → reflexive_memo → theme_map`。
# 我们不臆造 API,而是逐个查它们的契约。先查「主题编码」这个技能——注意它的
# `Requires: corpus['units']`,这条契约稍后会**真的**在运行时被强制执行。

# %%
print(sv.utils.registry_lookup("主题编码", max_results=1))

# %% [markdown]
# ## 1 · 契约是活的:未满足 `requires` 会抛 `RegistryError`
#
# 在填任何数据之前,我们**故意**去调 `code_themes`。它的契约要求 `corpus['units']`
# 存在,而空的 `StudyState` 里没有——于是抛 `RegistryError`。这不是 bug,而是**特性**:
# 契约不是死的元数据,而是每次调用时被强制的守卫(omicverse 的 `valid_keys` 机制移植到
# 社科)。错误信息还顺带告诉你**谁能生产**这个缺失槽(`build_corpus`),这就是
# grounding:报错本身指向修复路径。

# %%
empty = sv.StudyState()
try:
    sv.tl.code_themes(empty, lexicon={"x": ["y"]})
except sv.RegistryError as err:
    print("RegistryError(如预期):\n")
    print(err)

# %% [markdown]
# 与其猜「应该先调什么」,不如让注册表**反推整条计划**。`resolve_plan('theme_map')`
# 沿依赖图回溯:要画主题地图得先有 `codes[theme_map]`(由 `code_themes` 生产),而
# `code_themes` 又要 `corpus[units]`(由 `build_corpus` 生产)……最终排出有序 `plan`。
# 无人能生产的 `needs_input`(这里是 `sources[corpora]`——**用户必须提供的原始语料**)
# 也被明确列出;`escalations` 提示哪些自动插入步骤 `auto_fix=escalate`、需人工确认。

# %%
plan = sv.registry.resolve_plan("theme_map")
print(json.dumps(plan, ensure_ascii=False, indent=2))

# %% [markdown]
# ## 2 · 登记原始语料 → `build_corpus`
#
# **为什么这步。** 质性分析的最小可寻址对象是「编码单元」。`build_corpus` 做两件事:
# Unicode 规范化(NFC),再按粒度(默认段落)切成携带**稳定 `unit_id` 与字符偏移**的
# units——正是这个 `(doc_id, start, end)` 偏移让稍后的引语可以**逐字回溯到原文**。
#
# **契约。** `requires sources['corpora']` → `produces corpus['documents','units','manifest']`。
# 我们先把玩具语料(3 段短「访谈」,内含姓名/邮箱/电话等 PII)写进 `sources` 槽,再调用。

# %%
st = sv.StudyState()
st.write("sources", "corpora", ds.load_corpus())

sv.pp.build_corpus(st)  # 契约满足:sources['corpora'] 已在

print("文档数:", len(st.corpus["documents"]), " · 编码单元数:", len(st.corpus["units"]))
print("\nmanifest(每文档单元数/字符数):")
display(st.corpus["manifest"])

print("\n第一个 unit(注意 unit_id 内嵌 doc_id 与字符偏移):")
print(json.dumps(st.corpus["units"][0], ensure_ascii=False, indent=2))

# %% [markdown]
# ## 3 · 去标识 → `redact_pii`(伦理是一等公民)
#
# **为什么这步。** 在任何编码/分享之前,语料里的 PII 必须去除。`redact_pii` 用确定性
# regex(邮箱 / 电话 / 身份证等长数字串)做检测(装了 spaCy 还会加人名 NER 一遍),把每个
# 实体映射到**稳定假名**(`[EMAIL_1]`、`[PHONE_2]`……),同一实体到处读作同一 token,
# 并保留一份 `crosswalk`(假名→原文)供**授权者**在治理允许时再识别。
#
# **契约。** `requires corpus['documents']` → `produces corpus['documents']`(原地擦洗)
# + `governance['pii_status']`(合规回执)。下面对照擦洗前后同一段文本。

# %%
before = st.corpus["documents"]["int01"]

sv.pp.redact_pii(st)  # 原地擦洗 documents,并写 governance 回执

after = st.corpus["documents"]["int01"]
print("擦洗前 int01:\n ", before)
print("\n擦洗后 int01:\n ", after)

print("\npii_status(合规回执):", st.governance["pii_status"])
print("crosswalk(可逆再识别映射,须治理授权):")
print(json.dumps(st.governance["pii_crosswalk"], ensure_ascii=False, indent=2))

# %% [markdown]
# > **方法学注记(一个真实的陷阱)。** 目录里的推荐顺序是
# > `build_corpus → redact_pii`,但要小心:上一步的 `units` 是从**擦洗前**的文本切出来的,
# > 而 `redact_pii` 只擦洗了 `documents`。`[EMAIL_1]` 比原邮箱短,擦洗后**文档偏移已经位移**。
# > 如果此刻直接做引语溯源,`trace_quotes` 用旧 units 的 `(start,end)` 去 slice **新** documents,
# > 逐字回校就会失败(`verified=False`)。下一格我们先看这个失败,再修好它——
# > 这正是「证据链可审计」的价值:错位会被**如实报告**,而不是被悄悄吞掉。

# %% [markdown]
# ## 4 · 反身主题编码(第一版:暴露溯源陷阱)→ `code_themes` + `trace_quotes`
#
# **为什么这步。** 这是 Braun & Clarke 的 phase 2–4:对每个 unit 应用**编码词典**
# (`lexicon`),记录命中的 `segments`,把 codes 聚合为更高阶的 `themes`,并建一张
# **code 共现图**(`theme_map`,networkx 图导出为邻接字典)。每个主题带 claim→支撑单元的
# 证据脚手架。
#
# **契约。** `code_themes` 要 `corpus['units']`;`trace_quotes` 要 `corpus['units']` +
# `codes['segments']`,且 `prerequisites: code_themes` 必须先跑。我们先用**当前(擦洗后但
# 未重切)** 的 units 跑一遍,看引语回校率。

# %%
LEXICON = {
    "burnout":     ["burnout", "burned out", "crushing"],
    "support":     ["support", "belonging", "colleagues"],
    "autonomy":    ["autonomy", "flexibility"],
    "recognition": ["recognition", "morale"],
}
# 把 codes 归入更高阶主题(Braun & Clarke 的 theming 阶段,由研究者判断)
THEME_GROUPS = {
    "工作压力": ["burnout"],
    "组织支持": ["support", "recognition"],
    "工作条件": ["autonomy"],
}

sv.tl.code_themes(st, lexicon=LEXICON, themes=THEME_GROUPS)
sv.tl.trace_quotes(st)

cov1 = st.evidence["quote_index"]["coverage"]
print("第一版引语回校:verify_rate =", cov1["verify_rate"],
      f"({cov1['n_verified']}/{cov1['n_checkable']} 逐字回校通过)")
print("原因:units 切自擦洗前文本,偏移与擦洗后 documents 不再对齐 → 回校失败(如实报告)。")

# %% [markdown]
# `verify_rate = 0.0` —— 一个**如实报告**的失败。CAQDAS 里这种偏移错位往往悄无声息;
# 这里因为溯源是逐字 slice 回校,它被抓了个正着。修法很直接:**去标识后,从擦洗过的
# documents 重切 units**,让编码单元与最终文本严格对齐。下面重来一版。

# %% [markdown]
# ## 5 · 修正顺序:去标识 → **重切** → 编码 → 溯源(第二版:全绿)
#
# **为什么重切。** `build_corpus` 接受 `data=` 覆盖入参。我们把**擦洗后的 documents**
# 喂回 `build_corpus`,units 的 `(start,end)` 就与最终文本一致了。之后再 `code_themes` +
# `trace_quotes`,逐字回校应当 100% 通过。这是一条**去标识优先**的合规链:进入编码阶段的
# 每一个单元都已不含 PII,且可逐字溯源。

# %%
# 用擦洗后的 documents 重切编码单元(覆盖旧的、偏移错位的 units)
sv.pp.build_corpus(st, data=st.corpus["documents"])

# 重新编码 + 重新溯源
sv.tl.code_themes(st, lexicon=LEXICON, themes=THEME_GROUPS)
sv.tl.trace_quotes(st)

cov2 = st.evidence["quote_index"]["coverage"]
print("第二版引语回校:verify_rate =", cov2["verify_rate"],
      f"({cov2['n_verified']}/{cov2['n_checkable']} 逐字回校通过)")
print("\n完整 coverage:")
print(json.dumps(cov2, ensure_ascii=False, indent=2))

# %% [markdown]
# ### 5.1 编码台账(codebook)——每个 code 的频次、所属主题、关键词定义
#
# 这是 CAQDAS 里「Code Book」的等价物,但它是一张真实的 `pandas.DataFrame`,按频次排序,
# 可直接进稿件附录。

# %%
display(st.codes["codebook"])

# %% [markdown]
# ### 5.2 主题(themes)与「主题 → 支撑单元」证据脚手架
#
# `themes` 把 codes 聚合成研究者命名的高阶主题,记下每个主题由哪些 code、哪些 unit 支撑。
# `evidence['claim_evidence']` 则把它写成 **claim→support** 的证据形态——「主题 X 得到 N
# 个编码单元支撑」——这就是可放进结果段、且**每条都能点回原文**的论断。

# %%
print("themes(主题 → codes / n_segments / 支撑 unit_ids):")
print(json.dumps(st.codes["themes"], ensure_ascii=False, indent=2))

print("\nclaim_evidence(每个主题的论断⇄支撑证据):")
print(json.dumps(st.evidence["claim_evidence"], ensure_ascii=False, indent=2))

# %% [markdown]
# ### 5.3 每主题可溯源:验证过的引语(带字符偏移)
#
# 这是本 notebook 的核心交付:对每个主题,列出**已逐字回校**的引语,以及它精确的溯源戳
# `(doc_id, unit_id, start, end)`。`recovered` 是用该偏移从最终 document 切回来的文本,
# `verified=True` 表示它与编码时看到的 `quote` **逐字一致**——这就是「引语溯源」。

# %%
entries = st.evidence["quote_index"]["entries"]

def quotes_for_theme(theme_name):
    """返回某主题下所有 (code, 溯源戳, 引语, 是否回校) 的行。"""
    codes = set(st.codes["themes"][theme_name]["codes"])
    return [e for e in entries if e["code"] in codes]

for theme in st.codes["themes"]:
    print(f"■ 主题「{theme}」")
    for e in quotes_for_theme(theme):
        stamp = f"{e['doc_id']}[{e['start']}:{e['end']}]"
        flag = "✓已回校" if e["verified"] else "✗未回校"
        print(f"   [{e['code']:<11}] {stamp:<16} {flag}")
        print(f"      引语: {e['quote'][:88]}")
    print()

# 孤儿审计:没有引语的 code / 没有任何编码的 unit —— 空列表 = 覆盖完整
print("孤儿审计 orphans:", json.dumps(st.evidence["quote_index"]["orphans"], ensure_ascii=False))

# %% [markdown]
# ## 6 · 反身备忘 → `reflexive_memo`(把解释轨迹结构化为可审计协议)
#
# **为什么这步。** 反身主题分析的关键区别于「机械编码」之处,是研究者要交代自己的**立场**
# 与**解释轨迹**。`reflexive_memo` 把这条通常散落在私人笔记里的轨迹,结构化成可审计的协议:
# 三轴**立场声明**(社会位置 / 田野关系 / 利害)、每个主题一条**四段日志**(观察 / 反应 /
# 偏见 / 调整),以及一份明确的 **AI vs 人类 解释归属**(哪些由生成式辅助、哪些归研究者)。
#
# **契约。** `requires codes['themes'] + corpus['units']` → `produces evidence['provenance']`
# (解释审计链)+ `governance['ethics']`(反身性声明状态)。研究者须**亲自填**立场三轴;
# 只要填了,`ethics.status` 就从 `pending-researcher-input` 变为 `declared`。

# %%
sv.tl.reflexive_memo(
    st,
    researcher="FZ",
    positionality={
        "social_location": "组织社会学训练、非该行业从业者的外部研究者。",
        "field_relation": "与受访者无雇佣或师生关系,访谈为一次性、自愿参与。",
        "stakes": "研究不涉及商业委托,发现对研究者无直接利害;对受访机构可能敏感。",
    },
    authorship={
        "ai": ["按词典的自动编码命中", "code 共现主题地图的计算"],
        "human": ["主题命名与解释", "立场声明", "最终 claim–evidence 判断"],
    },
)

memo = st.evidence["provenance"]
print("立场声明(三轴,研究者亲填):")
print(json.dumps(memo["positionality"], ensure_ascii=False, indent=2))

print("\n某主题的四段反身日志(观察/反应/偏见/调整):")
first_theme = next(iter(memo["log"]))
print(f'  主题「{first_theme}」:')
print(json.dumps(memo["log"][first_theme], ensure_ascii=False, indent=2))

print("\n解释归属 AI vs 人类(须在方法与致谢中披露):")
print(json.dumps(memo["interpretation_authorship"], ensure_ascii=False, indent=2))

print("\ngovernance['ethics'](反身性声明合规状态):")
print(json.dumps(st.governance["ethics"], ensure_ascii=False, indent=2))

# %% [markdown]
# ## 7 · 主题共现网络图 → `theme_map`
#
# **为什么这步。** 把 `codes['theme_map']` 的 code 共现结构画成网络:节点是 code(按度加权
# 大小),边是共现次数,颜色编码所属主题。这是给读者的**一眼可见的主题结构**。
#
# **契约。** `requires codes['theme_map']`(由 `code_themes` 生产)→ `produces
# artifacts['figures']`。用确定性 spring layout(`seed=0`),存成同目录 PNG。

# %%
sv.pl.theme_map(st, out=_fig("fig_thememap.png"), title="访谈主题共现网络")

fig_meta = st.artifacts["figures"]["theme_map"]
print("已保存图:", fig_meta["path"], "· dpi:", fig_meta["dpi"])
print("note:", fig_meta["note"])
print("\ntheme_map 邻接结构(共现次数):")
print(json.dumps(st.codes["theme_map"], ensure_ascii=False, indent=2))

# %% [markdown]
# ![主题共现网络](fig_thememap.png)

# %% [markdown]
# ## 8 · 证据链:`st.summary()` 与 provenance ledger
#
# 最后展示这条链自带的**可复现审计轨迹**。`populated()` 显示每个被填的槽;provenance
# ledger 逐条记下每一步的 `function / requires / produces`——这就是社科里一等重要的
# 「证据脊」:任何一个结论,都能顺着 ledger 回溯到它依赖的槽、再回到原始语料的字符偏移。

# %%
print(st.summary())

print("\nprovenance ledger(每步的契约,按执行序):")
for r in st.provenance:
    req = ", ".join(f"{s}[{','.join(k)}]" for s, k in r["requires"].items()) or "∅"
    pro = ", ".join(f"{s}[{','.join(k)}]" for s, k in r["produces"].items()) or "∅"
    print(f"  step {r['step']}: {sv.utils._friendly(r['function'])}")
    print(f"         requires {req}")
    print(f"         produces {pro}")

# %% [markdown]
# ## 小结:对标的现实工具 + socialverse 的差异
#
# 这条链对标 **CAQDAS(NVivo / ATLAS.ti / QualCoder)的编码—检索—可视化工作流 + Braun &
# Clarke 反身主题分析六阶段**;去标识部分对标 spaCy / Presidio 式 PII 检测。
#
# **socialverse 的差异**在两点,都源自「注册表是脊柱」这一设计:
#
# 1. **注册表 grounding(查而非猜)。** 每个动作都是带 `requires/produces` 契约的注册函数:
#    未满足依赖会抛 `RegistryError` 并指向能修复它的生产者;整条链的顺序能被
#    `resolve_plan` 从目标反推。CAQDAS 是点选式 GUI、步骤间无机器可读契约,顺序错了
#    (比如本例的**去标识 vs 重切**顺序陷阱)往往悄无声息。
# 2. **证据链(逐字可溯源)。** provenance ledger 自动累积每一步的契约,`quote_index` 把每个
#    主题的每条引语**逐字回校**到原文字符偏移(`verify_rate` 如实报告错位),`reflexive_memo`
#    把研究者立场与解释归属结构化进 `governance`。结论因此可以一路溯源:
#    claim → 支撑 unit → 字符偏移 → 去标识回执 → 原始语料。
