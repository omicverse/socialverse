# %% [markdown]
# # 给一批访谈做质性编码:去标识、主题分析与引语溯源
#
# 质性研究要回答的是「人们怎么讲述自己的经历」,而不是「某个数字有多大」。最常见的做法是**反身主题分析**(Braun & Clarke 2006/2019):把访谈文本切成小段,逐段贴上「编码」(code)标签,再把意思相近的编码归拢成更高一层的「主题」(theme),最后用原话引语把每个主题撑起来。它不像回归那样有一个「显著」的门槛,分析的可信度来自另一处——**每一条结论都能被追回到某个受访者的某句原话**。
#
# 所以这条链有两个绕不开的关口。第一是**伦理**:访谈里几乎一定夹着姓名、邮箱、电话这类可识别信息(PII),在编码或分享之前必须先去标识;而且去标识要留一份可逆的对照表,以便在获得授权时再识别。第二是**可溯源**:主题分析常被诟病「研究者想看到什么就编出什么」,对治的办法是给每条引语盖一个精确的溯源戳(哪份文档、哪个字符区间),事后能逐字回校原文——这一步最容易出错,因为去标识会改变文本长度、让字符偏移错位,我们会专门撞一次这个坑再修好它。
#
# 整条链要走的步骤是:**载入语料 → 去标识 → 主题编码 → 引语溯源 → 反身备忘 → 主题地图**。我们用 `socialverse` 完成,它是一套面向社会科学的分析库,每一步都从一个共享的研究状态里读数据、往里写结果,并顺手记下一份可复现的证据链——你会在最后看到它。方法学骨架是 Braun & Clarke 的反身主题分析;工作流对标 **CAQDAS** 类质性软件(NVivo / ATLAS.ti / 开源的 [QualCoder](https://github.com/ccbogel/QualCoder)),去标识部分对标 spaCy / Presidio 式 PII 检测。
#
# 我们用一小批玩具语料把每一步看清楚:3 段短「访谈」(int01–int03),每段一两句话,里面故意埋了姓名、邮箱、电话。语料小,才好一眼盯住偏移、引语、回校率这些容易被忽略的细节。

# %%
import matplotlib
matplotlib.use("Agg")  # 无显示环境:图直接写文件

import json
import socialverse as sv
from socialverse import datasets as ds

# %% [markdown]
# ## 载入语料
#
# 质性分析的最小可寻址对象是「编码单元」——通常就是一个段落。`build_corpus` 先把每份文档做 Unicode 规范化(NFC),再按粒度切成携带**稳定 `unit_id` 与字符偏移**的单元。这个 `(doc_id, start, end)` 偏移是后面引语溯源的基石:有了它,任何一条引语都能被切回原文逐字比对。
#
# 我们先把原始访谈写进研究状态的 `sources` 槽,再调用 `build_corpus`。`manifest` 汇总每份文档切出多少单元、多少字符;每个 unit 的 `unit_id` 里内嵌了文档号和字符区间。

# %%
st = sv.StudyState()
st.write("sources", "corpora", ds.load_corpus())  # 3 段短访谈,含姓名/邮箱/电话

sv.pp.build_corpus(st)

print("文档数:", len(st.corpus["documents"]), " · 编码单元数:", len(st.corpus["units"]))
st.corpus["manifest"]

# %% [markdown]
# 看一眼第一个编码单元。注意 `unit_id` 形如 `int01:0-184`——文档号加字符区间,这就是「可寻址」的含义。此刻文本里的邮箱还是明文,下一步就去掉它。

# %%
print(json.dumps(st.corpus["units"][0], ensure_ascii=False, indent=2))

# %% [markdown]
# ## 去标识
#
# 在做任何编码或分享之前,先把 PII 擦掉。`redact_pii` 用确定性正则识别邮箱、电话、长数字串等实体(若环境里装了 spaCy,还会再跑一遍人名 NER),把每个实体替换成**稳定假名**——`[EMAIL_1]`、`[PHONE_1]`……——同一个实体在全语料里读作同一个 token。它同时留下一份 `crosswalk`(假名 → 原文),供获得治理授权者在需要时再识别;并写一份合规回执 `pii_status`。
#
# 下面对照 int01 擦洗前后的同一段文本,再打印回执和对照表。

# %%
before = st.corpus["documents"]["int01"]

sv.pp.redact_pii(st)  # 原地擦洗 documents,并写 governance 回执

after = st.corpus["documents"]["int01"]
print("擦洗前:", before)
print("擦洗后:", after)

# %%
print("pii_status(合规回执):", st.governance["pii_status"])
print("\ncrosswalk(假名 → 原文,可逆再识别,须授权):")
print(json.dumps(st.governance["pii_crosswalk"], ensure_ascii=False, indent=2))

# %% [markdown]
# 这里藏着一个真实的陷阱,值得停下来看清楚。上一步的 `units` 是从**擦洗前**的文本切出来的,而 `redact_pii` 只改了 `documents`。`[EMAIL_1]` 比原邮箱 `jane.doe@example.com` 短,擦洗后文档的字符偏移整体位移了。如果此刻就去做引语溯源,系统会拿旧 units 的 `(start, end)` 去切**新** documents,逐字回校自然对不上。下一节我们先如实撞上这个失败,再把它修好——因为失败被如实报告,恰恰是这条链的价值所在。

# %% [markdown]
# ## 主题编码
#
# 这是 Braun & Clarke 六阶段里的 phase 2–4:对每个单元应用一份**编码词典**(`lexicon`,把关键词映射到 code),记录命中的片段 `segments`,再把 code 按研究者的判断聚成更高阶的**主题** `themes`。同时建一张 code 共现图,供稍后画主题地图。
#
# `code_themes` 负责编码与主题;`trace_quotes` 负责把每个命中片段回校到原文、生成引语索引。我们先用**当前(擦洗后但没重切)**的 units 跑一遍,专门看引语回校率会怎样。

# %%
LEXICON = {
    "burnout":     ["burnout", "burned out", "crushing"],
    "support":     ["support", "belonging", "colleagues"],
    "autonomy":    ["autonomy", "flexibility"],
    "recognition": ["recognition", "morale"],
}
# 把 codes 归入更高阶主题(这一步是研究者的解释判断,不是自动的)
THEME_GROUPS = {
    "工作压力": ["burnout"],
    "组织支持": ["support", "recognition"],
    "工作条件": ["autonomy"],
}

sv.tl.code_themes(st, lexicon=LEXICON, themes=THEME_GROUPS)
sv.tl.trace_quotes(st)

cov1 = st.evidence["quote_index"]["coverage"]
print("第一版引语回校:verify_rate =", cov1["verify_rate"],
      f"({cov1['n_verified']}/{cov1['n_checkable']} 条逐字回校通过)")

# %% [markdown]
# `verify_rate = 0.0`——一条引语都没对上。原因正是上一节埋的坑:units 切自擦洗前文本,偏移与擦洗后的 documents 不再对齐。在很多点选式质性软件里,这种偏移错位是悄无声息的;这里因为溯源是逐字 slice 回校,它被当场抓住。修法很直接:**去标识之后,从擦洗过的 documents 重新切一遍 units**,让编码单元与最终文本严格对齐,再重新编码、重新溯源。

# %%
# 用擦洗后的 documents 重切编码单元(覆盖旧的、偏移错位的 units)
sv.pp.build_corpus(st, data=st.corpus["documents"])

# 在对齐后的单元上重新编码 + 重新溯源
sv.tl.code_themes(st, lexicon=LEXICON, themes=THEME_GROUPS)
sv.tl.trace_quotes(st)

cov2 = st.evidence["quote_index"]["coverage"]
print("第二版引语回校:verify_rate =", cov2["verify_rate"],
      f"({cov2['n_verified']}/{cov2['n_checkable']} 条逐字回校通过)")
print("\n完整 coverage:")
print(json.dumps(cov2, ensure_ascii=False, indent=2))

# %% [markdown]
# 现在 100% 回校通过,而且每个单元都至少被编码到一次(`unit_coverage = 1.0`)。这条「去标识优先」的顺序保证了:进入编码阶段的每一个单元都已不含 PII,且都能逐字溯源。

# %% [markdown]
# ### 编码台账
#
# 编码台账是质性软件里「Code Book」的等价物:每个 code 出现了几次、归在哪个主题下、由哪些关键词定义。这里它就是一张真实的 `pandas.DataFrame`,按频次排序,可直接进稿件附录。

# %%
st.codes["codebook"]

# %% [markdown]
# ### 主题与支撑证据
#
# `themes` 把 codes 聚成研究者命名的高阶主题,记下每个主题由哪些 code、哪些 unit 支撑。`claim_evidence` 进一步把它写成「论断 → 支撑」的形态——「主题 X 得到 N 个编码单元支撑」——这是可以直接放进结果段、且每条都能点回原文的论断。

# %%
print("themes(主题 → codes / n_segments / 支撑 unit_ids):")
print(json.dumps(st.codes["themes"], ensure_ascii=False, indent=2))

# %%
print("claim_evidence(每个主题的论断 ⇄ 支撑证据):")
print(json.dumps(st.evidence["claim_evidence"], ensure_ascii=False, indent=2))

# %% [markdown]
# ## 引语溯源
#
# 这是整本 notebook 的核心交付。对每个主题,列出**已逐字回校**的引语,以及它精确的溯源戳 `(doc_id, start, end)`。`verified=True` 表示用该偏移从最终 document 切回来的文本,与编码时看到的引语逐字一致。下面按主题打印。

# %%
entries = st.evidence["quote_index"]["entries"]

def quotes_for_theme(theme_name):
    """返回某主题下所有已溯源的引语条目。"""
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

# %% [markdown]
# 再做一次孤儿审计:有没有哪个 code 没配到任何引语、或哪个 unit 没被任何 code 编到。两个都是空列表,说明编码覆盖是完整的。

# %%
print("孤儿审计:", json.dumps(st.evidence["quote_index"]["orphans"], ensure_ascii=False))

# %% [markdown]
# ## 反身备忘
#
# 反身主题分析区别于「机械编码」的关键,是研究者要交代自己的**立场**和**解释轨迹**——通常这些散落在私人笔记里,难以审计。`reflexive_memo` 把它结构化成一份可审计的协议:三轴立场声明(社会位置 / 田野关系 / 利害)、每个主题一条四段日志(观察 / 反应 / 偏见 / 调整),以及一份明确的 **AI vs 人类** 解释归属——哪些由自动编码承担、哪些归研究者判断。立场三轴须研究者亲自填写;填了之后,伦理声明状态才从待补写变为 `declared`。

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

# %% [markdown]
# 每个主题都自动起了一条四段日志的骨架(观察已填,反应 / 偏见 / 调整留给研究者补写),再加一份 AI 与人类的解释归属——这份归属应当写进论文的方法与致谢。

# %%
first_theme = next(iter(memo["log"]))
print(f"主题「{first_theme}」的四段反身日志:")
print(json.dumps(memo["log"][first_theme], ensure_ascii=False, indent=2))

# %%
print("解释归属 AI vs 人类:")
print(json.dumps(memo["interpretation_authorship"], ensure_ascii=False, indent=2))
print("\nethics(反身性声明合规状态):")
print(json.dumps(st.governance["ethics"], ensure_ascii=False, indent=2))

# %% [markdown]
# ## 主题地图
#
# 最后把 code 的共现结构画成一张网络:节点是 code(按度加权大小),边是共现次数,颜色编码所属主题。这是给读者的一眼可见的主题结构。`theme_map` 用确定性布局(`seed=0`),直接从研究状态里读图出图,存成同目录 PNG。

# %%
sv.pl.theme_map(st, out="fig_thememap.png", title="访谈主题共现网络")

fig_meta = st.artifacts["figures"]["theme_map"]
print("已保存图:", fig_meta["note"], "· dpi:", fig_meta["dpi"])
print("\ntheme_map 邻接结构(共现次数):")
print(json.dumps(st.codes["theme_map"], ensure_ascii=False, indent=2))

# %% [markdown]
# ![主题共现网络](fig_thememap.png)
#
# burnout 与 support 是共现最密的一对(同一个受访者常常在讲压力的同时提到同事支持),autonomy 与 recognition 各自连着相邻主题——这张图把「主题之间如何交织」直观地摊开了。

# %% [markdown]
# ## 可复现的证据链
#
# 和普通的质性编码脚本相比,`socialverse` 多留了一样东西:整条链跑下来,研究状态里自动积累了一份 provenance 账本,逐条记下每一步用了哪个函数、消费了什么槽、产出了什么槽。质性研究里,「这条结论从哪一步、哪份数据、哪个受访者的哪句话来」和结论本身同等重要——这份账本让它可追溯、可复现。

# %%
print(st.summary())

# %% [markdown]
# 账本里能看到我们**跑了两遍**编码与溯源:step 3–4 是撞坑的第一版,step 5–7 是重切后修好的第二版。任何一条最终引语,都能顺着账本回溯:主题 → 支撑单元 → 字符偏移 → 去标识回执 → 原始语料。

# %%
for r in st.provenance:
    req = ", ".join(f"{s}[{','.join(k)}]" for s, k in r["requires"].items()) or "∅"
    pro = ", ".join(f"{s}[{','.join(k)}]" for s, k in r["produces"].items()) or "∅"
    print(f"step {r['step']}: {sv.utils._friendly(r['function'])}")
    print(f"   requires {req}")
    print(f"   produces {pro}")

# %% [markdown]
# ## 小结
#
# 我们走完了一条完整的质性编码链:载入语料 → 去标识 → 主题编码 → 引语溯源 → 反身备忘 → 主题地图。它对标 **CAQDAS**(NVivo / ATLAS.ti / QualCoder)的编码—检索—可视化工作流,方法学骨架是 Braun & Clarke 的反身主题分析六阶段,去标识部分对标 spaCy / Presidio。
#
# 与那些点选式软件相比,这里多了两样东西:去标识是一道**默认优先、且留可逆对照表**的合规关口,引语溯源则会把每条引语**逐字回校**到原文字符偏移——回校失败会被如实报告(就像本例的偏移错位陷阱),而不是被悄悄吞掉。下一本教程 [06_text_philology](06_text_philology.ipynb) 把文本处理推向更深:OCR、异文校勘与 TEI 编码。
