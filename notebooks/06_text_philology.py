# %% [markdown]
# # 从扫描件到校勘本:OCR、TEI 编码与抄本谱系重建
#
# 数字人文里,「校勘(textual criticism)」要解决的问题很古老:同一部作品流传下来好几个手抄本,彼此文字不完全一样,我们既想知道哪个读法更接近原作,也想知道这些抄本之间是谁抄谁。计算机没有让这件事变简单,但让它变得**可重复、可核对**:先把手稿扫描件 OCR 成文本、按 TEI 标准编码成结构化 XML,再把多个**见证本(witnesses)**逐字对勘,把差异整理成**校勘记(critical apparatus)**,最后用抄本间「共享的错误」推断它们的亲缘关系,画出一棵**谱系树(stemma codicum)**。
#
# 这条链最关键、也最容易被误解的一步是谱系重建。它背后是一条朴素但强大的原则——**共同错误法(method of common errors)**:两个抄本如果在同一个地方犯了同样的、不太可能各自独立犯下的错误,那它们多半有共同的来源。所以校勘的重点不是「哪里不一样」,而是「哪些不一样是被一起继承下来的」。难点在于把「一致」和「有意义的一致」区分开:抄本恰好都写对同一个常见词不能说明什么,只有共享的**异文(variant)**才携带谱系信息。
#
# 我们用 `socialverse` 走完 OCR→TEI→对勘→谱系→定稿编码的全流程。它是一套面向社会科学与人文的分析库,把每种方法登记进一张函数注册表,运行时校验「这一步要的前置在不在」,并把每一步记进一份可复现的证据链——你会在最后看到它。功能上它对标数字人文里一串各自为战的工具:`Tesseract` / `Kraken` 做 OCR、`TEI-P5` 做编码标准、`CollateX` / `Juxta` 做多见证本对勘、`Stemmaweb` 一类工具做谱系可视化。
#
# 我们的语料是一部微型「传统」:同一句话的四个见证本 A/B/C/D,故意植入了抄写关系(B、C 共享同一处拼写错误),好让重建出的谱系有一个可以核对的正确答案。

# %%
import matplotlib
matplotlib.use("Agg")  # 无显示环境:图直接写文件

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

# %% [markdown]
# ## OCR 与 TEI 编码:让一页手稿变得可计算
#
# 文本学的第一步是把手稿「录入」成机器能处理的形式,而且录入本身要遵守标准,否则每个项目各录各的、无法交换。数字人文的通用标准是 **TEI-P5**——一套用 XML 描述文本的规范,`<teiHeader>` 记录元数据(标题、责任人、来源),`<body>` 里的 `<p>`、`<lb/>` 记录段落与换行。`ocr_tei` 把这两步合在一起:对图片做版面感知 OCR,再编码成合法的 TEI。
#
# 本环境没装 Tesseract,`ocr_tei` 会**优雅降级**——把我们直接提供的转录文本当作已经 OCR 好的结果,照样编码成合法 TEI。这点在教学里恰好值得强调:用了真 OCR 还是文本直通,函数会如实写进证据链,而不是假装扫描过。下面喂一页仿古拼写的转录,`sources['scans']` 用 `{doc_id: 页面}` 结构,页面可以是图片路径(会被 OCR),也可以像这里是现成文本。

# %%
st_tei = sv.StudyState()
st_tei.write("sources", "scans", {"folio1": "In the begynning was the Word."})

sv.pp.ocr_tei(st_tei, titles={"folio1": "Prologue, folio 1r"})  # titles 填进 teiHeader

# ocr_tei 把每一页各编码成一份 TEI,存成 {doc_id: TEI字符串} 的字典
tei_map = st_tei.corpus["tei"]
print("corpus['tei'] 的类型:", type(tei_map).__name__, "· 页面:", list(tei_map))
print("\n--- folio1 的 TEI-P5(前 12 行)---")
print("\n".join(tei_map["folio1"].splitlines()[:12]))

# %% [markdown]
# 输出是一份合法的 TEI 文档:`<teiHeader>` 里带上了我们给的标题 `Prologue, folio 1r`,`<body>` 里是那句转录文本。用了哪条 OCR 路径,记在证据链里——这里 `engine` 是 `text-passthrough`、`ocr_available` 是 `False`,一眼就能看出这一页没有经过真正的图像识别。

# %%
prov = st_tei.evidence["provenance"]  # 该槽存放最近一步的方法学记录
print("OCR 引擎     :", prov["engine"])
print("引擎可用     :", prov["ocr_available"])
print("识别语言     :", prov["lang"])
print("文档数       :", prov["n_documents"])

# %% [markdown]
# ## 登记多个见证本
#
# 校勘的原料是同一部作品的多个抄本。在逐字对勘之前,每个见证本要先做两件事:**Unicode 规范化**(把视觉相同但编码不同的字符统一成 NFC,否则对齐会把它们当成异文),以及切成带**字符偏移**的可寻址单元(这样校勘记才能精确指到「第几个字到第几个字」)。`build_corpus` 一次做完,并产出一份 `manifest` 清单。
#
# 我们的四个见证本是同一句话的不同抄写。这里**故意植入共同错误**:B 和 C 都把 `bright` 抄成了 `bryght`——这是一个不太可能被两人各自独立犯下的拼写变异,正是共同错误法要抓的信号;C 另外还把 `sea` 改成了 `ocean`;D 最接近底本 A,只是句尾多了一个词。按共同错误法的逻辑,我们预期重建出的谱系会反映「B、C 同源」这层关系。

# %%
witnesses = {
    "A": "the moon was bright and the sea was calm that night",          # 底本(base)
    "B": "the moon was bryght and the sea was calm that nyght",          # 与 C 共享 bright→bryght
    "C": "the moon was bryght and the ocean was calm that night",        # 与 B 共享 bright→bryght;另有 sea→ocean
    "D": "the moon was bright and the sea was calm that night indeed",   # 最接近 A,仅句尾多一词
}

st = sv.StudyState()
st.write("sources", "corpora", witnesses)
sv.pp.build_corpus(st, unit="sentence")  # 按句切单元;也可 unit="word"/"line"

st.corpus["manifest"]  # 每个见证本切了几个单元、多少字符

# %% [markdown]
# `manifest` 印证了长度差异:A/B 各 51 字符,C 因 `sea→ocean` 长了 2 字符,D 因多一词长到 58。每个单元还带着自己的 id 和字符偏移,校勘记之后就靠它定位。

# %%
unit0 = st.corpus["units"][0]
print("unit_id  :", unit0["unit_id"])      # 形如 "A:0-51",见证本 + 字符区间
print("doc_id   :", unit0["doc_id"])
print("span     :", (unit0["start"], unit0["end"]))
print("text     :", unit0["text"])

# %% [markdown]
# ## 对勘:异文与校勘记
#
# 这是校勘的主戏。`philology_collate` 把每个见证本逐字对齐到底本 A(用 `difflib.SequenceMatcher`),把每处差异归成三类——**替换(sub)**、**脱漏(om)**、**增衍(add)**,再按位置合并成一份**校勘记(apparatus)**。校勘记是校勘本书页脚那一栏批注的机读版本:每个 locus(异文点)记下底本的读法(lemma)、各见证本在此处的异读、以及哪些见证本与底本一致。
#
# 它依赖 `corpus['documents']`,`build_corpus` 已经填好,所以可以直接跑。我们显式再写一次 `documents` 只是为了让底本的来源和 `build_corpus` 完全一致。

# %%
st.write("corpus", "documents", witnesses)  # 与 build_corpus 用同一份文本作底本
sv.tl.philology_collate(st, base="A")        # 以 A 为底本对勘

apparatus = st.artifacts["apparatus"]
print(f"校勘记共 {len(apparatus)} 个 locus(异文点)")

# 排成一张编辑会印在书页脚的批判校勘表
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
pd.DataFrame(app_rows)

# %% [markdown]
# 四个 locus 把这部小传统的差异说清楚了:locus 1(`bright`)上 B、C 共读 `bryght`,而 D 与底本一致——这正是我们埋下的共同错误;locus 2(`sea`)只有 C 改成 `ocean`;locus 3(`night`)只有 B 抄成 `nyght`;locus 4 是底本没有、只有 D 增衍的一个词(lemma 为空 `∅`,类型是增衍)。
#
# 逐条看第一个 locus 的完整结构:`witness_types` 标注了每个见证本此处是替换/脱漏/增衍,`agree_with_base` 是与底本一致的那些见证本——这两项合起来,才让谱系那一步能算出「谁和谁一起偏离了底本」。

# %%
e0 = apparatus[0]
print("locus          :", e0["locus"])
print("lemma (底本)   :", e0["lemma"])
print("readings       :", e0["readings"])         # 各见证本此处的异读
print("witness_types  :", e0["witness_types"])    # sub / om / add
print("与底本一致     :", e0["agree_with_base"])

# %% [markdown]
# ## 谱系重建
#
# 有了校勘记,谱系重建就是一步顺理成章的事。`philology_collate` 为每个见证本算一个「异文签名」——它在各个 locus 上偏离底本的模式,再两两算**共同错误距离(Hamming)**:两个见证本在越多的 locus 上以相同方式偏离底本,距离就越小、亲缘越近。最后用 `networkx` 的**最小生成树**把这些距离连成一棵树,得到 `models['stemma']`。
#
# 结果里 A–D 距离最小(1),因为 D 只在句尾多一词、其余全随底本;A–B 与 A–C 距离都是 2。B 与 C 因共享 `bright→bryght` 而在谱系上彼此靠拢——共同错误法的预期兑现了。

# %%
stemma = st.models["stemma"]
print("root(底本)  :", stemma["root"])
print("method       :", stemma["method"])
print("\n谱系边(共同错误距离越小 = 亲缘越近):")
for edge in sorted(stemma["edges"], key=lambda x: x["shared_error_distance"]):
    print(f"  {edge['from']} — {edge['to']}   shared_error_distance = {edge['shared_error_distance']}")

# %% [markdown]
# 把谱系画出来更直观。边权是共同错误距离,越小越近;红色节点是名义上的底本 A。

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
# 红色节点是底本 A;B 与 C 因共享错误彼此靠拢,D 紧挨 A——重建出的谱系正确复现了我们植入的抄写关系。

# %% [markdown]
# ## 定稿:编码成合法 TEI-P5
#
# 校勘定稿后,最终交付物是一份**合法的 TEI-P5 XML**,别的项目可以直接引用、检索、再加工。`tei_encode` 把底本的转录结构化成段落 `<p>` 与换行里程碑 `<lb/>`,配一个带标题/作者/责任声明的真正 `teiHeader`,并用 `lxml` 做良构性校验(缺 lxml 时回落到标准库的 `xml.etree`)。

# %%
sv.tl.tei_encode(
    st,
    witness="A",                                   # 把底本 A 编码为定稿
    title="On the Calm Sea",
    author="Anonymous (base witness A)",
    responsibility="socialverse tei-encoding demo",
)

tei_xml = st.corpus["tei"]  # 注意:这里是单份 XML 字符串,与第一步 ocr_tei 的逐页字典形态不同
print("--- 定稿 TEI-P5(teiHeader + body)---")
print(tei_xml)

with open("witness_A.tei.xml", "w", encoding="utf-8") as fh:
    fh.write(tei_xml)
print("已保存 -> witness_A.tei.xml")

# %% [markdown]
# 良构性不是我们口头保证的,而是函数自己校验后写进证据链:`well_formed=True`、用的是哪个解析器、根元素是不是 `TEI`,都留了痕。

# %%
val = st.evidence["provenance"]["validation"]  # tei_encode 最近写入的良构校验记录
print("良构       :", val["well_formed"])
print("解析器     :", val["parser"])
print("根元素     :", val["root"])
print("错误       :", val["error"])

# %% [markdown]
# ## 可复现的证据链
#
# 最后看一眼 `socialverse` 与一堆散装脚本的关键差别。整条链跑下来,研究状态里自动积累了一份 **provenance 账本**:每一步用了哪个函数、消费了哪些槽、产出了哪些槽。这本账本不是手动维护的,而是每个函数成功调用后自动追加的——一份跑完的校勘因此**自带审计轨迹**。在人文/社科里,「这个结论从哪一步、哪份材料来」往往和结论本身一样重要。

# %%
print(st.summary())

print("\n--- provenance 账本(逐步的 requires→produces)---")
for rec in st.provenance:
    req = ", ".join(f"{s}[{','.join(k)}]" for s, k in rec["requires"].items()) or "∅"
    pro = ", ".join(f"{s}[{','.join(k)}]" for s, k in rec["produces"].items()) or "∅"
    fn = rec["function"].split(".")[-1]
    print(f"  step {rec['step']}: {fn}")
    print(f"           requires: {req}")
    print(f"           produces: {pro}")

# %% [markdown]
# ## 小结
#
# 我们走完了一条标准的数字人文校勘链:OCR→TEI 编码 → 登记见证本 → 对勘异文 → 排校勘记 → 重建谱系 → 定稿 TEI。它对标 `Tesseract`/`Kraken`(OCR)、`TEI-P5`(编码标准)、`CollateX`/`Juxta`(对勘)与 `Stemmaweb`(谱系)这一串工具,而且全程真算:`difflib` 做对齐、`networkx` 的最小生成树做谱系。
#
# 与那串各自为战的工具相比,这里多了两样东西:一是缺前置时函数会**直接拦住你**(比如在空状态上调 `philology_collate` 会抛 `RegistryError` 并告诉你该先跑谁),而不是默默给一个错的结果;二是一份贯穿始终、自动生成的证据链——这正是散装工具链最缺、而人文考据最看重的东西。下一本教程 [07_theory_lens_network](07_theory_lens_network.ipynb) 转向用网络视角审视社会理论。
