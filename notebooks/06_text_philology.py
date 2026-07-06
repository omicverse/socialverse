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
# # 数字人文:扫描件 → TEI 编码 + 校勘异文与谱系重建
#
# **对标现实工具:** `Tesseract` / `Kraken`(OCR) · `TEI-P5` / `TEI Guidelines`(编码标准)
# · `CollateX` / `Juxta`(多见证本对勘) · `Stemmaweb` / `PHYLO` 的谱系树。
#
# 这条分析链走一遍**文本学者(textual scholar)的标准流程**:把一页手稿扫描件 OCR 成
# 文本、编码成合法的 **TEI-P5 XML**;再把同一部作品的多个**见证本(witnesses)**
# 相互对勘,分类出**异文(variant readings)**、排成**校勘记(critical apparatus)**,
# 并用**共同错误法(method of common errors)**重建抄本**谱系(stemma codicum)**。
#
# ## 涉及的 socialverse 函数(全部带注册契约)
#
# | 函数 | 阶段 | requires → produces | 后端 |
# |---|---|---|---|
# | `sv.pp.ocr_tei` | prepare | `sources['scans']` → `corpus['documents','tei']`, `artifacts['xml']` | pytesseract / TEI-P5(缺引擎自动降级为文本直通) |
# | `sv.pp.build_corpus` | prepare | `sources['corpora']` → `corpus['documents','units','manifest']` | pandas / regex / unicodedata(NFC) |
# | `sv.tl.philology_collate` | analyze(text) | `corpus['documents']` → `models['stemma']`, `artifacts['apparatus']` | difflib / networkx |
# | `sv.tl.tei_encode` | analyze(text) | `corpus['documents']` → `corpus['tei']`, `artifacts['xml']` | lxml / TEI-P5 |
#
# ## StudyState 会被填哪些槽(12 槽词汇表里的子集)
#
# - **`sources`** — 登记的原始输入:`scans`(扫描件/页面)、`corpora`(见证本文本)。
# - **`corpus`** — 文本即数据态:`documents`(规范化后的文档)、`units`(带字符偏移的可编码单元)、`manifest`、`tei`(TEI-XML)。
# - **`models`** — 拟合结果:这里是 `stemma`(谱系树:nodes / edges / adjacency)。
# - **`artifacts`** — 交付物:`apparatus`(校勘记)、`xml`(TEI-XML)。
# - **`evidence`** — 证据链:`provenance`(每步的方法学记录)。
#
# 此外,每个被 `@register` 包裹的函数**在成功调用后会自动向 `st.provenance`(只读追加账本)
# 记一笔**,带上它声明的 `requires`/`produces`——这就是社科/人文特别看重的**可复现证据脊柱**,
# `st.summary()` 结尾会把它数出来。

# %% [markdown]
# ## 0. 环境与注册表:先查,不猜
#
# socialverse 的设计命题(见 README):让分析 agent 可靠的不是一个统一数据容器(AnnData),
# 而是一张**带依赖契约的可查询函数注册表**。所以我们做任何事之前,先**查注册表**——
# 这正是 grounding 的全部意义:`registry_lookup` 让 agent「查而非猜」API。
#
# 下面先打印这条 text/philology 链在注册表里的样子,确认我们要调的四个函数、它们的
# `requires → produces` 契约,以及它们对标的现实后端。

# %%
import matplotlib
matplotlib.use("Agg")  # 无显示环境:图直接存文件

import json
import textwrap

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib import font_manager

import socialverse as sv

# 让图里的中文标题正常渲染:挑一个装了的 CJK 字体(缺则回落,不报错)
_installed = {f.name for f in font_manager.fontManager.ttflist}
for _cjk in ("Arial Unicode MS", "Songti SC", "STHeiti", "Hiragino Sans GB",
             "PingFang SC", "Noto Sans CJK SC", "Microsoft YaHei"):
    if _cjk in _installed:
        plt.rcParams["font.sans-serif"] = [_cjk, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

try:
    from IPython.display import display  # 在 notebook 里用 display() 渲染 DataFrame
except Exception:  # 当普通脚本跑时回落到 print
    def display(x):
        print(x)


def show(title, obj):
    """把一个 dict/list 结果连同小标题打印出来(教学用)。"""
    print(f"\n=== {title} ===")
    print(json.dumps(obj, ensure_ascii=False, indent=1, default=str))


print(f"socialverse v{sv.__version__} — 注册表里有 {len(sv.registry)} 个函数\n")

# 查询面 1:registry_lookup —— OmicOS kernel 打给 agent 看的同一套格式
print(sv.utils.registry_lookup("philology 校勘", max_results=1))

# %% [markdown]
# 注意上面输出里的 **Requires / Produces**——这不是文档字符串,而是**机器可读的契约**。
# 注册表还能反过来回答「要得到 TEI-XML,该按什么顺序跑?」:`resolve_plan` 会走依赖图,
# 把缺的槽用其**生产者**补齐(`corpus['documents']` 由 `build_corpus` / `ocr_tei` / `redact_pii`
# 生产),并把真正需要人给的输入列进 `needs_input`(这里是原始 `sources['corpora']`)。

# %%
plan = sv.registry.resolve_plan("tei_encode")
show("resolve_plan('tei_encode') — 到 TEI 的最短合法链", plan)

prereq = sv.registry.get_prerequisites("philology_collate")
show("get_prerequisites('philology_collate') — 谁能满足它的 requires", prereq)

# %% [markdown]
# ## 1. 契约是「活」的:未满足 requires 会抛 RegistryError
#
# 契约不是摆设。`philology_collate` 声明 `requires={'corpus': ['documents']}`;若在一个
# **空 StudyState** 上直接调它,注册表包装器会在调用前检查、抛出 `sv.RegistryError`——
# **而且错误信息会告诉你哪个函数能生产缺失的槽**。这就是 grounding 的运行时形态:
# 系统不会让你在缺前置的情况下静默产出垃圾结果。
#
# **契约(requires→produces):** requires `corpus['documents']` → (未满足,拒绝执行)。

# %%
st_empty = sv.StudyState()
try:
    sv.tl.philology_collate(st_empty)  # 缺 corpus['documents']
    print("(未预期:没有抛错)")
except sv.RegistryError as err:
    print("如预期抛出 RegistryError —— 这是特性,不是 bug:\n")
    print(textwrap.indent(str(err), "    "))

# %% [markdown]
# ## 2. OCR → TEI:把一页扫描件编码成 TEI-P5
#
# 文本学的第一步:让手稿**可计算**。`ocr_tei` 做版面感知 OCR(装了 Tesseract 时);
# 本环境没有 OCR 引擎,函数会**优雅降级**为「文本直通」——把已提供的文本当作已 OCR 的结果,
# 照样编码成合法 TEI。这一点很重要:契约保证「有没有引擎都能跑通链条」,只是把用了哪种路径
# 如实写进 `provenance`。
#
# **契约(requires→produces):** requires `sources['scans']` →
# produces `corpus['documents']`, `corpus['tei']`, `artifacts['xml']`, `evidence['provenance']`。
#
# `sources['scans']` 支持 `{doc_id: 页面}` 结构,页面既可以是图片路径(会被 OCR),
# 也可以是已有文本字符串(直接采用)。这里我们喂一页仿古拼写的转录文本。

# %%
st_tei = sv.StudyState()
st_tei.write("sources", "scans", {"folio1": "In the begynning was the Word."})

sv.pp.ocr_tei(st_tei, titles={"folio1": "Prologue, folio 1r"})

# ocr_tei 把 corpus['tei'] 写成 {doc_id: TEI字符串} 的字典(逐页一份 TEI)
tei_map = st_tei.corpus["tei"]
print("corpus['tei'] 的类型:", type(tei_map).__name__, "· 页面:", list(tei_map))
print("\n--- folio1 的 TEI-P5(前 12 行)---")
print("\n".join(tei_map["folio1"].splitlines()[:12]))

# %% [markdown]
# `ocr_tei` 用了哪条路径?看它写进证据链的这一笔:`engine` 记录了是真 Tesseract
# 还是文本直通,`ocr_available` 记录了引擎是否可用——**方法学如实留痕**,而不是假装 OCR 过。

# %%
show("evidence['provenance'] · ocr_tei 一步的方法学记录", st_tei.evidence["provenance"])

# %% [markdown]
# ## 3. 登记多个见证本,规范化成可编码单元
#
# 校勘的原料是**同一部作品的多个抄本/版本**(见证本)。真实校勘里,每个见证本先要被
# **Unicode 规范化(NFC)**、切成带**字符偏移**的可寻址单元,才能逐位对齐——`build_corpus`
# 就干这个,并产出一份 `manifest`。
#
# **契约(requires→produces):** requires `sources['corpora']` →
# produces `corpus['documents']`, `corpus['units']`, `corpus['manifest']`, `evidence['provenance']`。
#
# 我们用一部微型「传统」:四个见证本 A/B/C/D。这里**故意植入共同错误**——B 和 C 都把
# `bright` 抄成 `bryght`(还有 `nyght`),而 D 最接近底本 A。共同错误法的核心假设是:
# **两个见证本在异文上一致 ⇒ 它们有共同的来源**;所以我们预期重建出的谱系会把 B、C 聚到一起。

# %%
witnesses = {
    "A": "the moon was bright and the sea was calm that night",          # 底本(base)
    "B": "the moon was bryght and the sea was calm that nyght",          # 与 C 共享 bright→bryght
    "C": "the moon was bryght and the ocean was calm that night",        # 与 B 共享 bright→bryght;另有 sea→ocean
    "D": "the moon was bright and the sea was calm that night indeed",   # 最接近 A,仅多一词
}

st = sv.StudyState()
st.write("sources", "corpora", witnesses)
sv.pp.build_corpus(st, unit="sentence")

# manifest:每个见证本切了几个单元、多少字符
print("corpus['manifest'] —— 见证本清单:")
display(st.corpus["manifest"])

print("\ncorpus['units'][0] —— 一个带 unit_id 和字符偏移的可编码单元:")
show("unit[0]", st.corpus["units"][0])

# %% [markdown]
# ## 4. 校勘对勘:异文 → 校勘记 → 谱系
#
# 主戏。`philology_collate` 真正用 `difflib.SequenceMatcher` 把每个见证本对齐到底本,
# 把差异分类成**替换(sub)/脱漏(om)/增衍(add)**三类异文,按位置合并成一份**校勘记
# (apparatus)**;再从每个见证本的「异文签名」算两两**共同错误距离(Hamming)**,用
# `networkx` 的**最小生成树**重建谱系。全程真算,不是占位。
#
# **契约(requires→produces):** requires `corpus['documents']` →
# produces `models['stemma']`, `artifacts['apparatus']`, `evidence['provenance']`。
#
# `build_corpus` 已经把 `corpus['documents']` 填好,契约满足,可以直接跑(不会抛
# RegistryError)。

# %%
st.write("corpus", "documents", witnesses)  # 显式声明底本用的 documents(与 build_corpus 一致)
sv.tl.philology_collate(st, base="A")

apparatus = st.artifacts["apparatus"]
print(f"校勘记共 {len(apparatus)} 个 locus(异文点):\n")

# 把 apparatus 排成一张编辑会印在书页脚的批判校勘表
app_rows = []
for e in apparatus:
    readings = "; ".join(f"{w}: {r}" for w, r in e["readings"].items())
    app_rows.append({
        "locus": e["locus"],
        "base_span": tuple(e["base_span"]),
        "lemma (底本读法)": e["lemma"],
        "异文 readings": readings,
        "与底本一致": ", ".join(e["agree_with_base"]) or "—",
    })
display(pd.DataFrame(app_rows))

# %% [markdown]
# 逐条看第一个 locus 的完整结构——`witness_types` 标注了这是替换/脱漏/增衍,
# `agree_with_base` 是「阳性校勘记」里与底本一致的那些见证本。

# %%
show("apparatus[0] · 一个 locus 的完整批判记录", apparatus[0])

# %% [markdown]
# ### 谱系树(stemma codicum)
#
# `models['stemma']` 给出重建出的树:`nodes` / `edges`(带 `shared_error_distance`)/
# `adjacency`。共同错误法的预期在这里兑现——B 与 C 因共享 `bright→bryght` 而距离最近,
# 应当相邻;D 因最接近底本 A 而挂在 A 旁边。

# %%
stemma = st.models["stemma"]
show("models['stemma'] · 谱系(MST over shared-error distance)", stemma)

print("\n谱系边(共同错误距离越小 = 亲缘越近):")
for edge in sorted(stemma["edges"], key=lambda x: x["shared_error_distance"]):
    print(f"  {edge['from']} — {edge['to']}   shared_error_distance = {edge['shared_error_distance']}")

# %% [markdown]
# 把谱系画出来。边权是共同错误距离,越小越近;根名义上是底本 A。

# %%
g = nx.Graph()
for node in stemma["nodes"]:
    g.add_node(node)
for edge in stemma["edges"]:
    g.add_edge(edge["from"], edge["to"], weight=edge["shared_error_distance"])

pos = nx.spring_layout(g, seed=7, weight="weight")
fig, ax = plt.subplots(figsize=(6.2, 4.4))
root = stemma["root"]
node_colors = ["#c0392b" if n == root else "#2c3e50" for n in g.nodes()]
nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=1500, ax=ax)
nx.draw_networkx_labels(g, pos, font_color="white", font_size=13, font_weight="bold", ax=ax)
nx.draw_networkx_edges(g, pos, width=2.0, edge_color="#7f8c8d", ax=ax)
edge_labels = {(u, v): d["weight"] for u, v, d in g.edges(data=True)}
nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=11, ax=ax)
ax.set_title(
    f"Stemma codicum — 谱系树(根/底本 = {root},红)\n边权 = 共同错误距离(Hamming)",
    fontsize=11,
)
ax.axis("off")
fig.tight_layout()
fig.savefig("fig_stemma.png", dpi=130, bbox_inches="tight")  # PNG 用 tight
plt.close(fig)
print("已保存谱系图 -> fig_stemma.png")

# %% [markdown]
# ![谱系树 stemma](fig_stemma.png)
#
# 红色节点是底本 A;B 与 C 因共享错误彼此靠拢,D 紧挨 A——重建出的谱系正确复现了
# 我们在第 3 步植入的抄写关系。

# %% [markdown]
# ## 5. 把定稿见证本编码成合法 TEI-P5
#
# 校勘定稿后,最终交付一份**合法 TEI-P5 XML**。`tei_encode` 把转录文本结构化成段落
# `<p>` 与物理换行 `<lb/>` 里程碑,配一个真正的 `teiHeader`(标题/作者/责任声明),
# 并用 `lxml`(缺则 stdlib `xml.etree`)做**良构性校验**。
#
# **契约(requires→produces):** requires `corpus['documents']` →
# produces `corpus['tei']`, `artifacts['xml']`, `evidence['provenance']`。
#
# 我们把底本 A 编码为带元数据的 TEI(注意:这条链里 `corpus['tei']` 被写成**单个 XML 字符串**;
# 与第 2 步 `ocr_tei` 写的「逐页字典」形态不同——契约声明的是同一个槽,形态由具体函数决定)。

# %%
sv.tl.tei_encode(
    st,
    witness="A",
    title="On the Calm Sea",
    author="Anonymous (base witness A)",
    responsibility="socialverse tei-encoding demo",
)

tei_xml = st.corpus["tei"]
print("corpus['tei'] 的类型:", type(tei_xml).__name__, "(单份 XML 字符串)\n")
print("--- 定稿 TEI-P5(teiHeader + body)---")
print(tei_xml)

# 保存一份 XML 交付物到磁盘
with open("witness_A.tei.xml", "w", encoding="utf-8") as fh:
    fh.write(tei_xml)
print("\n已保存 -> witness_A.tei.xml")

# %% [markdown]
# 良构性由函数自己校验并写进证据链——`validation.well_formed=True`、用的哪个解析器、
# 根元素是不是 `TEI`,都留了痕:

# %%
show("evidence['provenance'] · tei_encode 的良构校验记录", st.evidence["provenance"])

# %% [markdown]
# ## 6. 证据链:provenance 就是可复现脊柱
#
# 最后看整条链的证据。`st.summary()` 把**每个被填的槽**和**追加式 provenance 账本的步数**
# 数出来。这本账本不是我们手动维护的——它由 `@register` 包装器在每次成功调用后**自动追加**,
# 每条都带该步声明的 `requires`/`produces`。换句话说,一份跑完的分析**自带审计轨迹**。

# %%
print(st.summary())

print("\n--- provenance 账本(逐步的 requires→produces 契约)---")
for rec in st.provenance:
    req = ", ".join(f"{s}[{','.join(k)}]" for s, k in rec["requires"].items()) or "∅"
    pro = ", ".join(f"{s}[{','.join(k)}]" for s, k in rec["produces"].items()) or "∅"
    fn = rec["function"].split(".")[-1]
    print(f"  step {rec['step']}: {fn}")
    print(f"           requires: {req}")
    print(f"           produces: {pro}")

# %% [markdown]
# ## 小结:对标现实工具,以及 socialverse 的差异
#
# 这条链复刻了数字人文的标准工作台:**Tesseract/Kraken** 做 OCR、**TEI-P5** 做编码标准、
# **CollateX/Juxta** 做多见证本对勘、**Stemmaweb** 一类工具做谱系重建。功能上等价:
# OCR→TEI、异文→apparatus→stemma,全程真算(`difflib` 对齐 + `networkx` 最小生成树)。
#
# **socialverse 的差异 = 注册表 grounding + 证据链。** 现实里这些是**互不相识的独立工具**,
# 拼装、传参、记录方法学全靠人。socialverse 把它们收进**一张带 `requires→produces` 契约的
# 可查询注册表**:
#
# 1. **查而非猜**——`registry_lookup` / `resolve_plan` 让 agent 在写代码前就知道该调什么、
#    按什么顺序调、缺什么该由谁补(第 0 节);
# 2. **契约是活的**——缺前置时 `philology_collate` 直接抛 `RegistryError` 并指出生产者,
#    而不是静默产垃圾(第 1 节);
# 3. **自带证据脊柱**——每步自动进 `provenance` 账本,一份跑完的校勘**自带可复现审计轨迹**
#    (第 6 节)——这正是社科/人文最看重、而散装工具链最缺的东西。
