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
# # 核验一份参考文献,揪出稿件里的幻觉引用
#
# 一篇稿子的参考文献列表,是它与既有知识对话的凭证。传统上我们用 Zotero 之类的工具把引用**收集、整理、格式化**成期刊要求的样式,这套流程假设了一个前提:列表里的每一条都真实存在。可当草稿出自(或经手于)一个大语言模型时,这个前提就塌了——列表里会混进**幻觉引用**:DOI 语法完美、标题像模像样,却把某篇真论文的身份张冠李戴地焊到了错误的作者头上,或者干脆缺 DOI、无从解析。这类错误纯靠肉眼几乎抓不住,却最伤学术诚信,而 Zotero 只管理引用、从不判真伪,CrossRef 也只在你主动去查某个 DOI 时才吐元数据。
#
# 所以「文献核验」的核心不是格式化,而是**逐条判定真伪**:一条参考文献,它的 DOI 能不能解析?题录是否完整?最关键的——标题指向的那篇真论文,作者集对不对得上?判断真伪需要一个可信的**真值来源**,而单一来源常有盲区,于是我们让三处来源(本地已知表、解析器、在线 Crossref/OpenAlex)做**三角核验**,第一个命中的胜出。这就是本教程的主菜:一条从检索到核验再到稿件审计的完整链路,把幻觉引用在进入正文之前拦下来。
#
# 我们用 `socialverse` 的 `sv.lit`(literature)工具族走完这条链:检索初筛(`search_free`)→ 个人库扇出(`zotero_bridge`)→ **三库核验(`verify_citations`)** → 引用风格化(`citation_manage`)→ 文献地图(`literature_map`)→ 稿件审计(`manuscript_review`)。数据用内置的玩具题录 `ds.load_bib()`:它只有 4 条,但**故意**掺了一条缺 DOI 的可疑记录(`sus1`)和一条把 Braun & Clarke 的真论文焊到 Foucault 头上的幻觉记录(`chi1`)——正好让每一步的判断都看得清清楚楚。功能上这条链对标 Zotero(个人库 + CSL 引用样式)加 CrossRef / OpenAlex(DOI 元数据解析)加期刊社的稿件初审。

# %% [markdown]
# ## 环境准备
#
# 先把 matplotlib 固定成 Agg 后端(无窗口、内核/CI 安全),再导入 socialverse。图统一存到 notebook 同目录,这样 `![](fig.png)` 相对引用无论从哪个工作目录跑都对得上。我们还提供一个极简的 `display()` 回退,好让这份 notebook 当普通脚本跑时也不报错。

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

# %% [markdown]
# ## 载入题录
#
# `ds.load_bib()` 返回一份 4 条的原始题录。前两条(`ok1` / `ok2`)是真实的定性研究经典;`sus1` 是一条**可疑记录**——题录看着完整,却缺 DOI,无从解析;`chi1` 是一条**幻觉引用**——它复用了 `ok2`(Braun & Clarke《Using thematic analysis in psychology》)的真标题和真 DOI,却把作者写成了 Foucault、年份改成 1975。这类「真论文身份 + 错误作者」的杜撰,正是核验要抓的靶子。先看原始 4 条:

# %%
raw_bib = ds.load_bib()
pd.DataFrame(raw_bib)[["id", "title", "authors", "year", "doi"]]

# %% [markdown]
# ## 检索初筛:规范化成统一 schema
#
# 拿到一批题录,第一步是把它们规范化成统一格式并去重。`search_free` 默认离线(除非显式传 `online=True` 且 `requests` 可用,否则绝不联网),它消费我们给的 `records=[...]`,把每条题录整理成 `id/title/authors/year/doi/...` 的统一 schema,写进 `sources['bib']`,同时留一份可供人工筛选(include/exclude/pending)的候选台账。

# %%
st = sv.StudyState()
sv.lit.search_free(st, records=raw_bib, query="thematic analysis reflexivity")

print("规范化后条数:", len(st.sources["bib"]))
print("保留的 id  :", [r["id"] for r in st.sources["bib"]])

# %% [markdown]
# **注意到 `chi1` 不见了。** `search_free` 按「DOI 优先,否则标题」去重,而幻觉记录 `chi1` 复用了 `ok2` 的真 DOI(`10.1191/1478088706qp063oa`),于是被当成重复项合并吞掉了。这是一个很现实的陷阱:去重本身是好事,但它会让「借真 DOI 伪装」的幻觉引用在初筛阶段就隐身。记住这一点——真正的核验必须**绕过去重、拿原始 4 条**上真值表,才能把 `chi1` 揪出来。这也说明了为什么「整理」和「核验」是两件不同的事。
#
# 候选台账记录了本次检索的模式、是否联网、候选数,每条一行、`screen='pending'`,方便后续人工筛选:

# %%
cit_ledger = st.evidence["citations"]
print("检索台账:mode=%s  online=%s  n_candidates=%s  query=%r" % (
    cit_ledger["mode"], cit_ledger["online"],
    cit_ledger["n_candidates"], cit_ledger["query"]))
pd.DataFrame(cit_ledger["candidates"])[["id", "title", "year", "doi", "screen"]]

# %% [markdown]
# ## 个人库扇出:按相关度打分排序
#
# 这一步对标 Zotero 个人库里的检索:`zotero_bridge` 把 `sources['bib']` 里每条记录对查询词按**五种策略**(标题 / 作者 / 标签 / 全文 / 批注)分别打分,加权融合成一个相关度 `_score`,再去重、按分降序排。没有连 Zotero 时它退化成纯 in-process 文本匹配,完全离线、确定性。我们查 `thematic analysis`,看每条命中了哪些策略、融合分多少:

# %%
sv.lit.zotero_bridge(st, query="thematic analysis")
ranked = st.sources["bib"]
pd.DataFrame([{
    "id": r["id"],
    "title": r["title"][:44],
    "_score": round(r["_score"], 3),
    "matched": ",".join(r["_matched_strategies"]),
} for r in ranked])

# %% [markdown]
# `ok2`(*Using thematic analysis in psychology*)分最高、命中 title 与 fulltext,符合直觉。但请留意:经过 `search_free` 去重后池子里只剩 3 条,幻觉记录 `chi1` 早已被吞,所以无论怎么排序都排不出它。**排序解决不了真伪问题**——这正是下一步核验存在的理由。

# %% [markdown]
# ## 三库核验:揪出 chimeric 与 suspicious
#
# 这是整条链的主菜。`verify_citations` 逐条把参考文献判进四种完整性状态:
#
# - **`verified`** — DOI 语法有效、题录完整,且无矛盾真值。
# - **`suspicious`** — 题录尚全,但 DOI 缺失或畸形,无法解析核验。
# - **`not_found`** — 既无有效 DOI 又无匹配真值,查无实据。
# - **`chimeric`** — 标题与已知真值高度吻合,但作者集不符(作者 Jaccard 重叠低于 `author_cut`):真论文的身份被焊到了错误的作者头上——这就是签名式的幻觉引用。
#
# 真值来自三处,按优先级 **resolver > 本地已知表 > 在线 Crossref/OpenAlex** 做三角核验,第一命中者胜(这就是「三库核验」的字面含义)。全程默认离线,只有显式 `online=True` 才联网,且联网失败会**静默降级**为离线判定、绝不抛错。
#
# 为了让 `chi1` 现形,我们做两件事:(a) 不走去重后的池子,而是把**原始 4 条**直接写进 `sources['bib']`;(b) 提供一张**本地已知表 `known`**——也就是 CrossRef 或某个受信参考库会给出的、那两个 DOI 的真实作者身份。这张表就是三角里的「第二个库」。

# %%
# (a) 绕过去重:把原始 4 条(含 chi1)直接登记为待核验的 bib
st_verify = sv.StudyState()
st_verify.write("sources", "bib", raw_bib)

# (b) 本地已知真值表(= CrossRef/受信库对这两个 DOI 给出的真实作者)
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
# 汇总里 `n_chimeric=1`、`ground_truth='known'`、`flagged_ids=['sus1','chi1']`——两条问题引用全部落网。逐条看判定理由(`reason`)和它是通过哪个库解析的(`resolved_via`):

# %%
pd.DataFrame([{
    "id": r["id"],
    "status": r["status"],
    "via": r["resolved_via"],
    "doi_valid": r["doi_valid"],
    "reason": r["reason"],
} for r in vb["records"]])

# %% [markdown]
# 读这张表就是一次取证:
#
# - `ok1` / `ok2` → **verified**(via `known`):标题相似度 1.00 且作者一致,三角命中真值。
# - `sus1` → **suspicious**(via `offline`):题录完整但缺有效 DOI,离线规则也无法放行。
# - `chi1` → **chimeric**(via `known`):标题与 Braun & Clarke 真论文相似度 1.00,可作者集 `{foucault}` 与真值 `{braun, clarke}` 重叠 0.00,低于阈值 0.34——张冠李戴的杜撰引文被点名。这正是 LLM 草稿里最难靠肉眼发现、却最伤诚信的一类错误。

# %% [markdown]
# 为什么非要那张 `known` 表?因为**规则只能查语法,真值才能查身份**。如果不给 `known` 表、纯用离线规则,`chi1` 因为 DOI 语法完美、题录齐全,会被判成 `verified`——幻觉逃逸。把两种模式并排跑一遍,差别一目了然:

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
# 把核验结果画成一张状态构成条形图,直观看到「4 条里只有一半真正过关」:

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
# ![核验状态构成](fig_verify_status.png)

# %% [markdown]
# ## 引用风格化:格式化到目标期刊
#
# 拿到判过真伪的 bib,`citation_manage` 把它格式化成目标期刊风格——APA 7 / Vancouver / Nature 三套是真实模板实现,不是占位串。APA 按作者姓氏排序,编号制保持引用顺序。这一步对标 Zotero 的 CSL 引用样式引擎。我们用去重后的 3 条(核验通过的 + suspicious)分别出 APA 与 Vancouver:

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
# `artifacts['tables']` 里同时留了一份逐条 DataFrame(n / id / formatted / style),方便直接落表或导出:

# %%
st.artifacts["tables"]

# %% [markdown]
# ## 文献地图:流派、代表作、时间脉络
#
# `literature_map` 读 bib,建一张关键词共现图(作为共引的代理),用 `networkx` 的贪心模块度做社区划分,切出不同的**流派**;再算出**代表作**(按影响力代理排序)、论战轴线、**时间脉络**。这一步不做估计,是方法论层面的知识地形图。我们的语料只有 3 条,流派会退化成单点簇,但机制与真实大语料完全一致:

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
# matplotlib 可用时,`literature_map` 还会渲一张知识地形图象限散点(x = 时间中点,y = 流派规模),以 base64 data-URI 挂在 `artifacts['figures']` 里(内核安全、可回传)。我们把它解码存到 notebook 同目录,好用 `![](fig.png)` 引用:

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
# ![文献知识地形图](fig_landscape.png)

# %% [markdown]
# ## 稿件审计:claim→evidence 支撑 + 就绪裁决
#
# 这是全链的收口。前面把参考文献查干净了,现在拿一份稿件正文,看**正文里的每一句论断是否有已核验的引文撑腰**。`manuscript_review` 做四件事:
#
# 1. 用正则抽出正文里的在文引注(APA 括注 / 叙述式 / 编号制);
# 2. 与 `evidence['verified_bib']` 配平——找出 `orphan`(引了但核验里没有)和 `uncited`(核验了但正文从没引);
# 3. 逐句做 claim→evidence 支撑审计,对因果/绝对措辞(cause / prove / always / every…)做对冲错配检查,把每句标成 `supported / unsupported / over-claim`;
# 4. 汇成 `supported_ratio` 加 `BLOCKER / MAJOR / MINOR / READY` 就绪裁决。
#
# 这一步的输入之一是 `evidence['verified_bib']`,所以我们直接复用上面 `st_verify` 里的核验结果,再喂一段**故意埋雷**的稿件:1 句正常引用、1 句叙述式引用、1 句因果绝对措辞 + 孤儿引注、1 句因果绝对措辞 + 无引注。

# %%
manuscript = (
    "Thematic analysis is a widely used qualitative method (Braun & Clarke, 2006). "
    "Reflexivity strengthens credibility, as Finlay (2002) argues. "
    "Our data prove definitively that autonomy causes engagement in every case (Smith, 2019). "
    "The results show a significant effect on burnout across all teams."
)
# 稿件正文登记到 sources['datasets'](manuscript_review 的输入之一)
st_verify.write("sources", "datasets", manuscript)

sv.lit.manuscript_review(st_verify, manuscript=manuscript)
ce = st_verify.evidence["claim_evidence"]

pd.DataFrame([{
    "claim": c["claim_id"],
    "status": c["status"],
    "cited": c["cited"],
    "causal": c["causal_language"],
    "hedged": c["hedged"],
    "sentence": c["sentence"][:60],
} for c in ce["claims"]])

# %% [markdown]
# 读表:C1 / C2 有已核验引文支撑 → `supported`;C3「prove definitively … causes … every case」是因果绝对措辞,引的 `(Smith, 2019)` 又不在 verified_bib 里 → `over-claim`;C4「significant effect … across all teams」同样因果绝对、且无引注 → `over-claim`。
#
# 引注配平又挑出了两类问题:`(Smith, 2019)` 是**孤儿引注**(核验里查无此人),而 `foucault` / `nobody` 是**已核验却从没被正文引用**的参考(uncited)。这类一致性审计,正是 Zotero 不会替你做的:

# %%
print("引注配平 balance:")
print(json.dumps(ce["balance"], ensure_ascii=False, indent=2))

# %% [markdown]
# 最后看 `diagnostics['coverage']` 的就绪裁决:`supported_ratio=0.5`、2 条 over-claim、1 条孤儿引注,裁决 **MAJOR**(有 over-claim 或孤儿即至少 MAJOR,超阈值升 BLOCKER)。这是一句可以直接写进审稿意见的话:「稿件尚未就绪——半数论断缺乏已核验引文支撑,存在因果过度断言与孤儿引注,需修回」。

# %%
cov = st_verify.diagnostics["coverage"]
print("就绪裁决:")
print(json.dumps(cov, ensure_ascii=False, indent=2))

# %%
st_verify.artifacts["tables"]

# %% [markdown]
# ## 可复现的证据链
#
# 和一份普通的核验脚本相比,`socialverse` 多给了一样东西。上面每一步核心函数在调用前,都会检查它需要的前置槽位是否就绪——比如 `manuscript_review` 要求 `evidence['verified_bib']` 已存在,少了就当场报错并告诉你该先跑哪一步,而不是默默给你一个错误结论。调用成功后,这一步用了什么、产出了什么,会自动记进一份只读的 provenance 账本。所以链末的 `st.summary()` 不只是「填了哪些槽」的快照,更是一条可审计的证据链:谁在什么契约下写了哪个槽,一目了然。这在文献核验里尤其重要——「这条引用是通过哪个库、按什么规则判成幻觉的」和判定结果本身同等重要。

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
# ## 小结
#
# 我们走完了一条完整的文献链:检索初筛 → 个人库扇出 → **三库核验** → 引用风格化 → 文献地图 → 稿件审计。功能上它对标 Zotero(个人库 + CSL 引用样式)加 CrossRef / OpenAlex(DOI 元数据解析)加期刊社的稿件初审。
#
# 但和这些点选式工具相比,这里多了两样关键的东西:一是 `verify_citations` 把「本地已知表 / resolver / 在线库」做成三库三角核验,逐条判真伪,一眼揪出 Zotero 存不出、CrossRef 不主动告诉你的张冠李戴幻觉引用(`chi1`:真标题、真 DOI、错作者);二是每一步都自动累积进一条可审计的证据链,让「文献工作」从点选式黑箱,变成一条可溯源的分析链。下一本教程 [10_full_study_evidence_chain](10_full_study_evidence_chain.ipynb) 把这些方法串成一个可复核的完整小研究,并把整条证据链导出成可交付的凭证。
