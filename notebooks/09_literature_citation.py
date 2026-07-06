# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 09 · 文献与引证:检索 → 三库核验(揪幻觉引用) → 稿件审计
#
# **这条分析链讲什么。** 一篇稿子的参考文献列表,是它与既有知识对话的凭证。可当草稿
# 出自(或经手于)一个大模型时,列表里往往混着**幻觉引用**——DOI 语法完美、标题像真的,
# 却把某篇真论文的身份**张冠李戴**焊到了错误的作者头上(chimeric),或干脆缺 DOI、无从
# 解析(suspicious)。本链走一遍**文献工作者 + 稿件审校**的完整流程:
#
# 1. **检索初筛**(`search_free`):把一批题录喂进免费无鉴权 API 的离线通道,规范化成
#    统一 schema,写进 `sources['bib']` 池 + 一份可筛选的候选台账。
# 2. **个人库扇出**(`zotero_bridge`):对 `sources['bib']` 按**五策略**(标题/作者/标签/
#    全文/批注)打分排序去重——顺带演示**去重会吞掉共 DOI 的幻觉记录**这一现实陷阱。
# 3. **三库核验**(`verify_citations`):逐条判定 `verified / suspicious / not_found /
#    chimeric`。三个真值来源(resolver > 本地已知表 > 在线 Crossref/OpenAlex)**三角**
#    核验,第一命中者胜——这是「三库核验」的字面含义,也是揪出**张冠李戴幻觉引用**的关键。
# 4. **引用管理**(`citation_manage`):把列表格式化成目标期刊风格(APA / Vancouver / Nature)。
# 5. **文献地图**(`literature_map`):共现聚类出流派 / 代表学者 / 论战轴线 / 时间脉络。
# 6. **稿件审计**(`manuscript_review`):正则抽正文引注,与 `verified_bib` 配平
#    (orphan 孤儿引注 / uncited 未被引),逐句做 **claim→evidence 支撑审计**,对**因果/
#    绝对措辞**做对冲错配检查,给出 `BLOCKER / MAJOR / MINOR / READY` 就绪裁决。
#
# **涉及的函数(全部先查注册表,再调用)。**
#
# | 阶段 | 函数 | requires → produces | Tier |
# |---|---|---|---|
# | `sv.lit` | `search_free` | `∅` → `sources[bib]`, `evidence[citations]` | community |
# | `sv.lit` | `zotero_bridge` | `sources[bib]` → `sources[bib]` | plus |
# | `sv.lit` | `verify_citations` | `sources[bib]` → `evidence[verified_bib]` | community |
# | `sv.lit` | `citation_manage` | `sources[bib]` → `evidence[citations]`, `artifacts[tables]` | community |
# | `sv.lit` | `literature_map` | `sources[bib]` → `evidence[landscape]`, `artifacts[figures]` | community |
# | `sv.lit` | `manuscript_review` | `sources[datasets]` + `evidence[verified_bib]` → `evidence[claim_evidence]`, `diagnostics[coverage]`, `artifacts[tables]` | community |
#
# **`StudyState` 会被填的槽:** `sources`(bib / datasets)· `evidence`(citations /
# verified_bib / landscape / claim_evidence)· `diagnostics`(coverage)·
# `artifacts`(tables / figures)。这 12 槽词汇表正是每一步 `requires`/`produces`
# 契约书写的语言——让依赖图**可被机器检查**、让 agent「查而非猜」的机制。
#
# **对标的现实工具。** 这条链对标 **Zotero**(个人文献库 + 引用样式引擎 CSL)+
# **CrossRef / OpenAlex**(DOI 元数据解析)+ 期刊社的稿件初审。socialverse 的差异在于:
# Zotero 只**管理**引用、不判真伪,CrossRef 只在你**主动查**某个 DOI 时才告诉你元数据;
# 而这里 `verify_citations` 把「本地已知表 / resolver / 在线库」做成**三角核验的一等公民**,
# 逐条给整数级的 `chimeric/suspicious` 判定;更关键的是**每一步都是注册表里带 `requires/
# produces` 契约的函数**——链的顺序可被 `resolve_plan` 反推,`st.provenance` 自动累积成
# 一条可审计的**证据链**。这是「Zotero + CrossRef」这类点选式工具给不了的 grounding。

# %% [markdown]
# ## 0 · 环境与「查而非猜」
#
# 先把 matplotlib 固定为 **Agg** 后端(无窗口、CI/内核安全),再导入 socialverse。
# 图保存到 notebook 同目录,这样 `![](fig.png)` 相对引用无论从哪个 cwd 跑都对得上。
# 我们提供一个极简 `display()`,脱离 Jupyter 当普通脚本跑时也不报错。
#
# **关键姿势:先查注册表。** socialverse 的设计论点是「让 agent 可靠的不是统一的数据
# 容器,而是带显式依赖标注的**可查询函数注册表**」。所以写任何调用前,我们先用
# `sv.utils.registry_summary()` 看目录、用 `registry_lookup(query)` 查某个技能的契约——
# 这与在生物域调 `ov.utils.registry_lookup` 是同一套「Found N … Requires … Produces …
# Example」输出格式。

# %%
import matplotlib
matplotlib.use("Agg")  # 必须在 pyplot 被任何地方 import 之前设定

import base64
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

import socialverse as sv
from socialverse import datasets as ds

# 图保存到 notebook 同目录(与 ![](fig.png) 相对引用对齐)。
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # 交互式 Jupyter 里没有 __file__
    _HERE = os.getcwd()


def _fig(name):
    return os.path.join(_HERE, name)


try:  # 真正的 Jupyter 里用富显示;当普通脚本跑时回退到 print
    from IPython.display import display
except Exception:  # pragma: no cover
    def display(obj):
        print(obj)


pd.set_option("display.max_colwidth", 70)
pd.set_option("display.width", 120)

print("socialverse version:", sv.__version__)
print()
print(sv.utils.registry_summary())

# %% [markdown]
# 目录里 `literature` 链清清楚楚写着推荐顺序:
# `search_free → zotero_bridge → citation_manage → verify_citations → manuscript_review`。
# 我们不臆造 API,而是先查这条链上最核心的一步——**引文核验**——的契约。注意它的
# `Requires: sources['bib']`,这条契约稍后会**真的**在运行时被强制执行;它的
# `Produces: evidence['verified_bib']`,正是下游稿件审计的输入。

# %%
print(sv.utils.registry_lookup("引文核验", max_results=1))

# %% [markdown]
# ## 1 · 契约是活的:未满足 `requires` 会抛 `RegistryError`
#
# 在把任何题录填进 `StudyState` 之前,我们**故意**去调 `verify_citations`。它的契约要求
# `sources['bib']` 存在,而空的 state 里没有——于是抛 `RegistryError`。这不是 bug,而是
# **特性**:契约不是死的元数据,而是每次调用时被强制的守卫(omicverse 的 `valid_keys`
# 机制移植到社科)。错误信息还顺带告诉你**谁能生产**这个缺失槽(`search_free` /
# `zotero_bridge`)——这就是 grounding:**报错本身指向修复路径**。

# %%
empty = sv.StudyState()
try:
    sv.lit.verify_citations(empty)
except sv.RegistryError as err:
    print("RegistryError(如预期):\n")
    print(err)

# %% [markdown]
# 与其猜「审稿要先跑什么」,不如让注册表**反推整条计划**。`resolve_plan('manuscript_review')`
# 沿依赖图回溯:审稿要 `evidence['verified_bib']`(由 `verify_citations` 生产),而
# `verify_citations` 又要 `sources['bib']`(由 `search_free` 生产);审稿还要
# `sources['datasets']`(稿件正文,由 `ingest` 生产)……最终排出有序 `plan`。
# `escalations` 里明确提示:自动插入 `search_free` 来补 `sources.bib` 这一步,因为下游
# `verify_citations` 的 `auto_fix=escalate`,**需人工确认**——不替你偷偷跑。

# %%
plan = sv.registry.resolve_plan("manuscript_review")
print(json.dumps(plan, ensure_ascii=False, indent=2))

# %% [markdown]
# 再看一眼 `manuscript_review` 自己的契约:它显式声明了 `required_functions:
# [verify_citations]`(必须先核验),`requires` 里 `evidence['verified_bib']` 由
# `verify_citations` 满足、`sources['datasets']` 由 `ingest` 满足——`satisfied_by`
# 把「缺什么、谁能补」一并列清。这就是 `resolve_plan` 能自动排链的信息基础。

# %%
print(json.dumps(sv.registry.get_prerequisites("manuscript_review"),
                 ensure_ascii=False, indent=2))

# %% [markdown]
# ## 2 · 检索初筛 `search_free`:题录 → 规范化 `sources['bib']`
#
# **契约:** `∅ → sources['bib'], evidence['citations']`。
#
# `search_free` 是**默认离线**的第一遍检索:除非显式传 `online=True` 且 `requests` 可用,
# 否则它绝不联网,而是消费调用方给的 `records=[...]`,把每条题录规范化成统一 schema
# (`id/title/authors/year/doi/venue/...`),同时写出一份**可筛选的候选台账**
# `evidence['citations']`(每条一行,`screen='pending'`)。
#
# 我们用内置玩具题录 `ds.load_bib()`:它**故意**掺了一条 `sus1`(缺 DOI)和一条 `chi1`
# (幻觉引用:把 Braun & Clarke 的真标题+真 DOI 焊到了 Foucault 头上)。先看原始 4 条:

# %%
raw_bib = ds.load_bib()
print("原始题录(4 条,含 1 条 suspicious + 1 条 chimeric):")
display(pd.DataFrame(raw_bib)[["id", "title", "authors", "year", "doi"]])

st = sv.StudyState()
sv.lit.search_free(st, records=raw_bib, query="thematic analysis reflexivity")
print("\nsources['bib'] 规范化后条数:", len(st.sources["bib"]))
print("规范化后保留的 id:", [r["id"] for r in st.sources["bib"]])

# %% [markdown]
# **注意到了吗——`chi1` 不见了。** `search_free` 会按 **DOI-优先-否则标题** 去重,而
# 幻觉记录 `chi1` 复用了 `ok2` 的真 DOI(`10.1191/1478088706qp063oa`),于是被当作重复
# 项**合并吞掉**了。这是一个**极其现实**的陷阱:去重是好事,但它会让「借真 DOI 伪装」的
# 幻觉引用**在初筛阶段就隐身**。记住这一点——第 4 步核验时我们会**绕过去重、拿原始 4 条
# 上真值表**,才能把 `chi1` 揪出来。
#
# 候选台账 `evidence['citations']` 记录了本次检索的 mode / 是否联网 / 候选数,供后续
# 人工筛选(include / exclude / pending):

# %%
cit_ledger = st.evidence["citations"]
print("检索台账:mode=%s  online=%s  n_candidates=%s  query=%r" % (
    cit_ledger["mode"], cit_ledger["online"],
    cit_ledger["n_candidates"], cit_ledger["query"]))
display(pd.DataFrame(cit_ledger["candidates"])[["id", "title", "year", "doi", "screen"]])

# %% [markdown]
# ## 3 · 个人库五策略扇出 `zotero_bridge`:打分排序
#
# **契约:** `sources['bib'] → sources['bib']`(原地精炼)。Tier = **plus**。
#
# 这是对标 **Zotero 个人库**的扇出检索:把 `sources['bib']` 里每条记录对 `query` 按
# **五策略**(title / author / tag / fulltext / annotation)分别打分,加权融合成一个相关度
# `_score`,再去重、按分降序排。无 Zotero MCP 时它退化为纯 in-process 文本匹配(完全离线、
# 确定性)。我们查一下 `thematic analysis`,看每条命中的策略与融合分:

# %%
sv.lit.zotero_bridge(st, query="thematic analysis")
ranked = st.sources["bib"]
display(pd.DataFrame([{
    "id": r["id"],
    "title": r["title"][:44],
    "_score": r["_score"],
    "matched": ",".join(r["_matched_strategies"]),
} for r in ranked]))

# %% [markdown]
# `ok2`(*Using thematic analysis in psychology*)分最高、命中 title/fulltext——符合直觉。
# 但请留意:经过 `search_free` 去重后池子里只有 3 条,**幻觉记录 `chi1` 早已被吞**,所以
# 无论怎么排序都排不出它。**排序解决不了真伪问题**——这正是下一步核验存在的理由。

# %% [markdown]
# ## 4 · 三库三角核验 `verify_citations`:揪出 chimeric / suspicious
#
# **契约:** `sources['bib'] → evidence['verified_bib']`。这是本 notebook 的**主菜**。
#
# `verify_citations` 逐条把参考文献判进四种完整性状态:
#
# - **`verified`** — DOI 语法有效 + 题录完整,且无矛盾真值。
# - **`suspicious`** — 题录尚全但 **DOI 缺失/畸形**,无法解析核验。
# - **`not_found`** — 既无有效 DOI 又无匹配真值,查无实据。
# - **`chimeric`** — 标题与已知真值**高度吻合**,但**作者集不符**(Jaccard 重叠 <
#   `author_cut`):真论文身份被焊到错作者上——**签名式的幻觉引用**。
#
# 真值三角来自三处,按优先级 **resolver > 本地 `known` 表 > 在线 Crossref/OpenAlex**,
# 第一命中者胜(这就是「三库核验」的字面含义)。全程默认离线,只有显式 `online=True` 才
# 联网,且失败**静默降级**为离线判定、绝不抛错。
#
# **为了让 `chi1` 现形,我们做两件事:**(a)**不走去重后的池子**,而是把原始 4 条
# 直接写进 `sources['bib']`;(b)提供一张**本地已知表 `known`**——即 CrossRef/一个受信
# 参考库会给出的、那两个 DOI 的**真实作者**。这张表就是「第二个库」。

# %%
# (a) 绕过去重:把原始 4 条(含 chi1)直接登记为待核验的 bib
st_verify = sv.StudyState()
st_verify.write("sources", "bib", raw_bib)

# (b) 本地已知真值表(= CrossRef/受信库对这两个 DOI 给出的真实作者身份)
known = [
    {"title": "Using thematic analysis in psychology",
     "authors": ["Braun, V.", "Clarke, V."], "doi": "10.1191/1478088706qp063oa"},
    {"title": "The Practice of Reflexivity in Qualitative Research",
     "authors": ["Finlay, L."], "doi": "10.1177/104973202129120052"},
]

sv.lit.verify_citations(st_verify, known=known)
vb = st_verify.evidence["verified_bib"]

print("核验汇总:")
print(json.dumps(vb["summary"], ensure_ascii=False, indent=2))
print("\n状态计数:", vb["tally"])

# %% [markdown]
# 汇总里 `n_chimeric=1`、`ground_truth='known'`、`flagged_ids=['sus1','chi1']`——两条问题
# 引用**全部落网**。逐条看判定理由(`reason`)和**通过哪个库解析的**(`resolved_via`):

# %%
display(pd.DataFrame([{
    "id": r["id"],
    "status": r["status"],
    "via": r["resolved_via"],
    "doi_valid": r["doi_valid"],
    "reason": r["reason"],
} for r in vb["records"]]))

# %% [markdown]
# 读一遍这张表就是一次**取证**:
#
# - `ok1` / `ok2` → **verified**(via `known`):标题 sim=1.00 且作者一致,三角命中真值。
# - `sus1` → **suspicious**(via `offline`):题录完整但**无有效 DOI**,离线规则也无法放行。
# - `chi1` → **chimeric**(via `known`):标题与 Braun & Clarke 真论文 sim=1.00,可作者集
#   `{foucault}` 与真值 `{braun, clarke}` 重叠 **0.00 < 0.34**——**张冠李戴的杜撰引文**被
#   点名。这正是一个 LLM 草稿里最难靠肉眼发现、却最伤学术诚信的错误。
#
# 对照:若**不给 `known` 表**(纯离线规则),`chi1` 因为 DOI 语法完美、题录齐全,只会被
# 判成 `verified`——**幻觉逃逸**。这就是为什么「三库三角」不是可选项:**规则能查语法,
# 只有真值能查身份**。下面把两种模式并排跑给你看:

# %%
st_rules = sv.StudyState()
st_rules.write("sources", "bib", raw_bib)
sv.lit.verify_citations(st_rules)  # 纯规则,无 known 表
tally_rules = st_rules.evidence["verified_bib"]["tally"]
chi_status_rules = next(r["status"] for r in st_rules.evidence["verified_bib"]["records"]
                        if r["id"] == "chi1")
print("纯离线规则     tally:", tally_rules, "| chi1 →", chi_status_rules, "(幻觉逃逸!)")
print("加本地 known 表 tally:", vb["tally"], "| chi1 →", "chimeric", "(被揪出)")

# %% [markdown]
# 把核验结果画成一张**状态构成条形图**,直观看到「4 条里只有一半真正过关」:

# %%
order = ["verified", "suspicious", "not_found", "chimeric"]
colors = {"verified": "#2e7d32", "suspicious": "#ef6c00",
          "not_found": "#757575", "chimeric": "#c62828"}
counts = [vb["tally"].get(s, 0) for s in order]

fig, ax = plt.subplots(figsize=(6.4, 3.6))
bars = ax.bar(order, counts, color=[colors[s] for s in order],
              edgecolor="black", linewidth=0.6)
for b, c in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03, str(c),
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_ylabel("# references")
ax.set_ylim(0, max(counts) + 0.6)
# 图内文字用英文,避免默认 DejaVu 字体缺中文字形的 tofu(叙事仍在 markdown 里用中文)
ax.set_title("verify_citations - reference integrity (n=4)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(_fig("fig_verify_status.png"), dpi=150, bbox_inches="tight")  # PNG → tight
plt.close(fig)
print("saved:", _fig("fig_verify_status.png"))

# %% [markdown]
# ![](fig_verify_status.png)

# %% [markdown]
# ## 5 · 引用管理 `citation_manage`:格式化到目标期刊风格
#
# **契约:** `sources['bib'] → evidence['citations'], artifacts['tables']`。
#
# 拿到干净的 bib,`citation_manage` 把它格式化成目标期刊风格——**APA 7 / Vancouver /
# Nature** 三套是真实模板实现(不是占位串)。APA 按作者姓氏排序,编号制保持引用顺序。
# 这一步对标 **Zotero 的 CSL 引用样式引擎**。我们用**去重后的 3 条**(核验通过 + suspicious)
# 分别出 APA 与 Vancouver:

# %%
sv.lit.citation_manage(st, style="APA")
apa = st.evidence["citations"]
print("=== APA 7 (n=%d) ===" % apa["n"])
for f in apa["formatted"]:
    print(f"  [{f['n']}] {f['reference']}")

sv.lit.citation_manage(st, style="Vancouver")
van = st.evidence["citations"]
print("\n=== Vancouver (n=%d) ===" % van["n"])
for f in van["formatted"]:
    print(f"  [{f['n']}] {f['reference']}")

# %% [markdown]
# `artifacts['tables']` 同时留了一份**逐条 DataFrame**(n / id / formatted / style),
# 方便直接落表或导出:

# %%
display(st.artifacts["tables"])

# %% [markdown]
# ## 6 · 文献地图 `literature_map`:流派 / 代表学者 / 时间脉络
#
# **契约:** `sources['bib'] → evidence['landscape'], artifacts['figures']`。
#
# `literature_map` 读 bib,建一张**关键词共现图**(作为共引的代理),用真实的 `networkx`
# 贪心模块度**社区划分**切出**流派(schools of thought)**;再算**代表作**(影响力代理
# 排序)、**论战轴线**(最大的几个流派两两对照)、**时间脉络**。这一步无代码、是方法论层
# 的知识地形图。小语料下流派会退化成单点簇,但**机制**与真实语料一致:

# %%
sv.lit.literature_map(st)
land = st.evidence["landscape"]
print("n_references=%d  n_schools=%d" % (land["n_references"], land["n_schools"]))
print("cluster_method:", land["cluster_method"])
print("\n代表作(按影响力代理排序):")
for w in land["seminal_works"]:
    print("  · %s (%s) — %s [influence=%.2f]" % (
        ", ".join(w["authors"]) or "?", w["year"], w["title"][:48], w["influence"]))
print("\n时间脉络:")
for t in land["timeline"]:
    print("  %d  n=%d  · %s" % (t["year"], t["n"], t["exemplar"][:52]))

# %% [markdown]
# `literature_map` 在 matplotlib 可用时还会渲一张**知识地形图象限散点**(x=时间中点,
# y=流派规模),以 **base64 data-URI** 挂在 `artifacts['figures']` 里(内核安全、可回传)。
# 我们把它**解码存到 notebook 同目录**,好用 `![](fig.png)` 引用:

# %%
figs = st.artifacts.get("figures") or {}
land_fig = figs.get("landscape")
if land_fig and land_fig.get("data_uri"):
    _, b64 = land_fig["data_uri"].split(",", 1)
    with open(_fig("fig_landscape.png"), "wb") as fh:
        fh.write(base64.b64decode(b64))
    print("saved:", _fig("fig_landscape.png"), "·", land_fig.get("caption"))
else:
    print("matplotlib 不可用,literature_map 已 fail-soft 跳过图(结构化 landscape 仍在)")

# %% [markdown]
# ![](fig_landscape.png)

# %% [markdown]
# ## 7 · 稿件审计 `manuscript_review`:claim→evidence 支撑审计 + 就绪裁决
#
# **契约:** `sources['datasets'] + evidence['verified_bib'] → evidence['claim_evidence'],
# diagnostics['coverage'], artifacts['tables']`。`prerequisites: [verify_citations]`。
#
# 这是全链的**收口**:拿一份稿件正文,`manuscript_review`
#
# 1. 正则抽出正文里的**在文引注**(APA 括注 / 叙述式 / 编号制);
# 2. 与 `evidence['verified_bib']` **配平**:`orphan`(引了但核验里没有)/ `uncited`
#    (核验了但正文从未引);
# 3. 逐句做 **claim→evidence 支撑审计**,对**因果/绝对措辞**(cause/prove/always/every…)
#    做**对冲错配**检查,把每句标成 `supported / unsupported / over-claim`;
# 4. 汇成 `supported_ratio` + `BLOCKER / MAJOR / MINOR / READY` **就绪裁决**。
#
# 契约里的 `requires`(尤其 `evidence['verified_bib']`)会在函数体运行**之前**被注册表
# wrapper 强制:所以我们必须先有第 4 步的核验结果。我们复用刚才 `st_verify` 的
# `verified_bib`,再喂一段**故意埋雷**的稿件(1 条正常引用、1 条叙述式引用、1 句
# **因果绝对+孤儿引注**、1 句**因果绝对+无引注**):

# %%
manuscript = (
    "Thematic analysis is a widely used qualitative method (Braun & Clarke, 2006). "
    "Reflexivity strengthens credibility, as Finlay (2002) argues. "
    "Our data prove definitively that autonomy causes engagement in every case (Smith, 2019). "
    "The results show a significant effect on burnout across all teams."
)
# 稿件正文登记到 sources['datasets'](manuscript_review 的 requires 之一)
st_verify.write("sources", "datasets", manuscript)

sv.lit.manuscript_review(st_verify, manuscript=manuscript)
ce = st_verify.evidence["claim_evidence"]

print("逐句 claim→evidence 审计:")
display(pd.DataFrame([{
    "claim": c["claim_id"],
    "status": c["status"],
    "cited": c["cited"],
    "causal": c["causal_language"],
    "hedged": c["hedged"],
    "sentence": c["sentence"][:64],
} for c in ce["claims"]]))

# %% [markdown]
# 读表:C1/C2 有**已核验引文**支撑 → `supported`;C3「prove definitively … causes … every
# case」是**因果绝对措辞**、引的 `(Smith, 2019)` 又不在 verified_bib 里 → `over-claim`;
# C4「significant effect … across all teams」同样是因果绝对、且**无引注** → `over-claim`。
#
# 引注配平也把两类问题挑了出来:`(Smith, 2019)` 是**孤儿引注**(核验里查无此人),而
# `foucault` / `nobody` 是**已核验却从未被正文引用**的参考(uncited)。这正是 Zotero
# 这类工具**不会替你做**的一致性审计:

# %%
print("引注配平 balance:")
print(json.dumps(ce["balance"], ensure_ascii=False, indent=2))

# %% [markdown]
# 最后看 `diagnostics['coverage']` 的**就绪裁决**——`supported_ratio=0.5`、2 条 over-claim、
# 1 条孤儿引注,裁决 **MAJOR**(有 over-claim 或孤儿即至少 MAJOR;超过阈值升 BLOCKER)。
# 这是一句可以直接写进审稿意见的话:**「稿件尚未就绪:半数论断缺乏已核验引文支撑,存在
# 因果过度断言与孤儿引注,需修回」**。

# %%
cov = st_verify.diagnostics["coverage"]
print("就绪裁决:")
print(json.dumps(cov, ensure_ascii=False, indent=2))

print("\n审稿问题清单 artifacts['tables']:")
display(st_verify.artifacts["tables"])

# %% [markdown]
# ## 8 · 证据链:`st.summary()` 是这条分析的可复现脊柱
#
# 每个被 `@register` 包裹的函数,**成功调用后都会自动向 `st.provenance`(只读追加账本)
# 记一笔**,带上它声明的 `requires`/`produces`。所以链末的 `st.summary()` 不只是「填了哪些
# 槽」的快照,更是一条**可审计的证据链**:谁在什么契约下写了哪个槽,一目了然。
#
# 我们打印**核验+审稿这条 state**(`st_verify`)的 summary,再把 provenance 账本逐条列出——
# 这就是社科/人文最看重的**可复现脊柱**:

# %%
print(st_verify.summary())
print()
print("provenance 账本(每一步的契约):")
for p in st_verify.provenance:
    fn = p["function"].split(".")[-1]
    req = sv.utils._fmt_slots(p["requires"]) or "∅"
    pro = sv.utils._fmt_slots(p["produces"]) or "∅"
    print("  step %d · %-18s  requires[%s]  →  produces[%s]" % (
        p["step"], fn, req, pro))

# %% [markdown]
# ---
# ### 小结:这条链对标什么,socialverse 的差异在哪
#
# 这条 **检索 → 三库核验 → 稿件审计** 链,功能上对标 **Zotero(个人库 + CSL 引用样式)+
# CrossRef / OpenAlex(DOI 元数据解析)+ 期刊社稿件初审**。但三点本质差异:
#
# 1. **判真伪,不止管理。** Zotero 只**存**引用、CrossRef 只在你**主动查**时才吐元数据;
#    `verify_citations` 把「本地已知表 / resolver / 在线库」做成**三角核验的一等公民**,
#    逐条给整数级 `chimeric/suspicious` 判定——一眼揪出 LLM 草稿里**张冠李戴的幻觉引用**
#    (`chi1`:真标题真 DOI 焊错作者),而这类错误纯规则或肉眼几乎抓不住。
# 2. **grounding:查而非猜。** 每一步都是注册表里带 `requires/produces` 契约的函数;未满足
#    依赖当场抛 `RegistryError` 并**指向修复路径**,整条链的顺序可由 `resolve_plan` 反推。
#    这不是文档约定,而是**运行时强制**的机制。
# 3. **证据脊柱自动累积。** `st.provenance` 把每一步的契约记成只读账本,`st.summary()` 即
#    可审计的复现凭证——把「文献工作」从点选式黑箱,变成一条**可机器检查、可溯源**的分析链。
