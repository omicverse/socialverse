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
# # 在跑分析之前,先过研究治理这三道闸门
#
# 一份社会科学微数据要真正落地,拦在分析前面的往往不是模型,而是三个「能不能」的问题:这份数据里的人会不会被重新识别出来?这些来源我到底有没有权利去抓、去再分发?稿件里用到的生成式 AI 该怎么向期刊交代?在生物信息里这些多半是事后补的合规清单;在社会科学里,它们是决定研究**能不能开始**的前置闸门——过不了,后面的回归再漂亮也发不出去、甚至不该做。
#
# 这三个问题各自对应一套成熟的方法。**可识别性**用 **k-匿名**衡量:把若干「准标识符」(quasi-identifier,比如分层、抽样单元、年龄段)组合起来,数一数最小的那一组还剩几个人——`k` 就是这个最小等价类的规模,`k=1` 意味着有人只属于自己那一组、可被直接重识别,`k` 越大越安全。**数据合规**是逐个来源的版权/许可判断:公有域、知识共享(CC)、出版商文本数据挖掘许可(TDM)、图书馆档案馆(GLAM)、平台服务条款(ToS)——不同来源给的抓取与再分发权利天差地别,而且许可**未知时不能默认有权**。**AI 使用披露**的真正红线不是「用没用 AI」,而是**「已采纳但未核验」**:把 AI 生成的内容直接写进稿件却没人核对过。
#
# 这本教程用 `socialverse` 把这三道闸门连成一条治理链走一遍:载入微数据 → 声明分析单位 → 跑伦理闸门(先看它如实报出 NO-GO,再一步步补救到 PASS)→ 许可分诊 → AI 披露审计 → 出一张治理仪表盘 → 留下证据链。`socialverse` 是一套面向社会科学的分析库,它把治理当成和 DID、复杂抽样等分析方法**平级**的一等公民——每道闸门都是可调用的函数,判决结构化、可追溯。对标现实里的工具:k-匿名通常靠 **ARX / sdcMicro** 或 `pandas` 手搓,许可判断靠人读 rightsstatements.org 与期刊条款,AI 披露靠照抄 **ICMJE / COPE** 模板——这里把它们收进同一条链。
#
# 我们用一份内置的合成调查数据:300 位受访者,含分层 `strata`、初级抽样单元 `psu`、六道态度量表 `item1..item6`、抽样权重 `weight`、暴露 `exposure` 与结果 `outcome`。它足够小、字段清楚,方便把每道闸门的判据看清楚。

# %%
import matplotlib
matplotlib.use("Agg")  # 无显示环境:图直接写文件,必须在 import pyplot 之前设定

import json
import os

import pandas as pd

import socialverse as sv
from socialverse import datasets as ds

# 图保存到 notebook 同目录,这样无论从哪个 cwd 运行都对得上 ![](fig.png) 相对引用
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # 交互式 Jupyter 里没有 __file__
    _HERE = os.getcwd()

def _fig(name):
    return os.path.join(_HERE, name)

print("socialverse version:", sv.__version__)

# %% [markdown]
# ## 载入数据
#
# 先把调查数据读进来看一眼。我们要盯的几列是准标识符候选:`strata`(3 个分层)和 `psu`(19 个初级抽样单元)——它们的组合决定了一个人有多容易被从人群里挑出来。态度量表和权重列这本教程用不到,但它们的存在正说明真实微数据里「顺手能定位到人」的字段无处不在。

# %%
df = ds.load_survey()
print("调查数据 shape:", df.shape)
print("strata 取值:", sorted(df["strata"].unique().tolist()),
      " · psu 取值数:", df["psu"].nunique())
df.head()

# %% [markdown]
# ## 声明分析单位
#
# 伦理闸门的第一件事不是算 k,而是问清楚「分析单位是什么」——个体?家户?国家?这决定了研究**是不是涉及人类受试者**。我们建一个 `StudyState`(可以类比 AnnData:一个贯穿全程、按槽位存放研究态的容器),把数据登记进 `sources`,并声明分析单位为 `row`(一行 = 一位受访者 = 人类受试者)。这两笔写入分别是后面 `data_use_check` 与 `ethics_check` 的入口条件。

# %%
st = sv.StudyState()
st.write("sources", "datasets", df)   # 供数据合规闸门做许可分诊
st.write("design", "unit", "row")     # 分析单位:一行 = 一位受访者 = 人类受试者
print("已填槽:", st.populated())

# %% [markdown]
# ## 伦理闸门:先看裸状态如实报出 NO-GO
#
# `ethics_check` 一次跑四项检查,再折叠成一个 `PASS / FIX / NO-GO` 判决:**IRB**(有没有记录人类受试者审查裁定)、**知情同意**(有没有同意基础)、**可识别性**(对声明的准标识符做真实 k-匿名计算,默认阈值 `k_threshold=5`)、**数据最小化**(有没有删掉直接标识符)。聚合规则很直白:任一项 NO-GO,整体就 NO-GO。
#
# 第一版我们**什么都不声明**,只把 `strata+psu` 当准标识符交给它算 k,看它对一个还没做任何治理工作的裸状态给出什么判决。

# %%
sv.gov.ethics_check(st, data=df, quasi_identifiers=["strata", "psu"])  # k_threshold 默认 5

ethics_v1 = st.governance["ethics"]
print("伦理判决:", ethics_v1["verdict"])
print("\n四项检查:")
for c in ethics_v1["checks"]:
    print(f"  [{c['status']:<5}] {c['check']:<12} — {c['detail']}")

# %% [markdown]
# 判决是 **NO-GO**。IRB 和同意都是 NO-GO(有人类受试者却没记录裁定/同意),数据最小化是 FIX,而 k-匿名是 FIX——因为 `strata+psu` 组合下最小的等价类只有 **k=2** 人,低于阈值 5。这就是治理作为一等公民的价值:在你写第一行分析代码之前,闸门就把「为什么现在不能上」结构化地摆了出来,而不是等审稿人或 IRB 事后打回。下面看一眼那次 k-匿名到底算了什么。

# %%
print("k-匿名细节(真实计算 df.groupby(QI).size().min(),非占位):")
print(json.dumps(ethics_v1["k_anonymity"], ensure_ascii=False, indent=2))

# %% [markdown]
# 300 条记录被 `strata × psu` 分成 57 个等价类,最小的一类只有 2 人(`k=2`),没有单例(`n_unique_records=0`)。所以问题不是「有人被直接暴露」,而是「粒度太细、离阈值还差一点」——这类问题可以靠**粗化准标识符**来修。

# %% [markdown]
# ## 补救伦理闸门:粗化准标识符 + 记录 IRB/同意 → PASS
#
# NO-GO 不是终点,而是一张待办清单。我们逐项补救:记录 IRB 裁定为 `exempt`(已获豁免类审查)、记录同意基础为 `informed`、确认已删除直接标识符 `minimized=True`;至于 k-匿名,把准标识符从 `strata+psu` **粗化**到只留 `strata`——这是 k-匿名最标准的补救手法(泛化/抑制准标识符),用更粗的粒度换更大的最小等价类。这四项都是研究者真实的治理决定,通过关键字参数声明给闸门,它重算后应给出 PASS。

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
print("补救后判决:", ethics_v2["verdict"])
print("\n四项检查:")
for c in ethics_v2["checks"]:
    print(f"  [{c['status']:<5}] {c['check']:<12} — {c['detail']}")

# %% [markdown]
# 四项全绿,整体 **PASS**。粗化的效果很直观:准标识符从 `strata+psu` 收到只剩 `strata` 后,最小等价类从 2 人一下抬到 97 人——远超阈值 5。

# %%
print(f"k-匿名:粗化前 (strata+psu) k={ethics_v1['k_anonymity']['k']}"
      f"  →  粗化后 (strata) k={ethics_v2['k_anonymity']['k']}(≥ 阈值 5,PASS)")

# %% [markdown]
# ### 红线案例:k=1 的直接可识别,是硬 NO-GO
#
# 不是所有 k 问题都能靠粗化修好。k-匿名的极端是 **k=1**:存在只属于一个人的准标识符组合,这个人可被直接重识别。我们人为造一列唯一行号 `resp_id` 当准标识符——即便 IRB、同意、最小化全都声明齐了,闸门也会把它判成 **NO-GO** 而非可修的 FIX,因为「300 个单例记录」是结构性泄露,不是泛化能轻易补救的。这演示了闸门对可识别性的分级:`k ≥ 阈值` = PASS,`1 < k < 阈值` = FIX,`k ≤ 1` = NO-GO。

# %%
df_leak = df.copy()
df_leak["resp_id"] = range(len(df_leak))   # 唯一直接标识符,人为制造 k=1

st_leak = sv.StudyState()
st_leak.write("sources", "datasets", df_leak)
st_leak.write("design", "unit", "row")
sv.gov.ethics_check(st_leak, data=df_leak, quasi_identifiers=["resp_id"],
                    irb="exempt", consent="informed", minimized=True)

leak = st_leak.governance["ethics"]
k_check = next(c for c in leak["checks"] if c["check"] == "k_anonymity")
print("判决:", leak["verdict"], "· k =", leak["k_anonymity"]["k"],
      "· 单例记录数:", leak["k_anonymity"]["n_unique_records"])
print("k-匿名检查:", k_check["status"], "—", k_check["detail"])

# %% [markdown]
# ## 数据合规:许可五桶分诊
#
# 「这份数据我到底能不能抓、能不能再分发」在社科里是逐个来源的判断。`data_use_check` 把每个来源分诊进**五个桶**(从严到宽):`platform_tos`(平台服务条款)、`publisher_tdm`(出版商文本数据挖掘许可)、`glam`(图书馆档案馆)、`cc`(知识共享)、`public_domain`(公有域),从许可字符串推出 `can_scrape`(能不能抓)/ `redistribution`(能不能再分发)/ `attribution`(要不要署名)以及 NC/ND/SA 义务标记。先做最简单的单源:一份干净的 `CC-BY-4.0`。

# %%
sv.gov.data_use_check(st_fixed, license="CC-BY-4.0")

du = st_fixed.governance["data_use"]
print("单源 (CC-BY-4.0) 分诊:")
print("  桶:", du["bucket"], "· 可抓取:", du["can_scrape"],
      "· 再分发:", du["redistribution"], "· 需署名:", du["attribution"])
print("  说明:", du["per_source"][0]["note"])

# %% [markdown]
# CC-BY 落进 `cc` 桶:可抓取、可再分发(在 CC 条款下,`share_alike_or_by`)、需署名。这是一份「好数据」应有的样子。接下来看两种更常见、也更棘手的情形。

# %% [markdown]
# ### 许可未知就落到最严桶
#
# 治理的一条基本姿态是 safe-by-default:**没有证据 ≠ 可以随便用**。传一个空许可串,看闸门把它默认落到最严的 `platform_tos` 桶——`can_scrape=False`、再分发 `prohibited`,并打上一条明确的待办:先厘清权利,再谈抓取或分享。

# %%
st_unknown = sv.StudyState()
st_unknown.write("sources", "datasets", df)
sv.gov.data_use_check(st_unknown, license="")   # 许可未知

du_u = st_unknown.governance["data_use"]
print("桶:", du_u["bucket"], "· 可抓取:", du_u["can_scrape"],
      "· 再分发:", du_u["redistribution"])
print("flags:")
for f in du_u["flags"]:
    print("  -", f)

# %% [markdown]
# ### 多源混合取「最弱环」
#
# 真实项目往往混合多个来源。闸门先逐源分诊,再取权利的**交集**:只要有一个源不能抓,整盘就不能抓;再分发权利取最严的那一档;NC/ND/SA 义务标记则做并集上浮。下面混四个典型来源——公有域普查、Twitter 平台 ToS、出版商 TDM、CC-BY-NC-SA 图像——看整盘怎样被最弱的那一环拉低。

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
pd.DataFrame([
    {"来源": t["source"], "桶": t["bucket"],
     "可抓取": t["can_scrape"], "再分发": t["redistribution"]}
    for t in du_m["per_source"]
])

# %% [markdown]
# 逐源看,census 是公有域(最宽)、cc 图像可在 CC 条款下用,但 Twitter 落进平台 ToS(不能抓)。取交集后,整盘被这一环拉到最严:

# %%
print("整盘(最弱环求交):")
print("  涉及的桶:", du_m["bucket"])
print("  can_scrape(全部可抓才为真):", du_m["can_scrape"])
print("  redistribution(取最严):", du_m["redistribution"])
print("  义务标记(并集上浮):")
for f in du_m["flags"]:
    print("   -", f)

# %% [markdown]
# 只要盘里有一个 Twitter,整个数据集就 `can_scrape=False`、再分发 `prohibited`——哪怕另外三个源都很宽松。这正是「最弱环」的意思:合规看的是权利的下限,不是上限。

# %% [markdown]
# ## AI 使用披露:审计「已采纳但未核验」红线
#
# 期刊现在普遍要求交代生成式 AI 的使用(ICMJE、COPE、Nature 各有政策),但真正的研究诚信红线不是「用没用 AI」,而是**「已采纳但未核验」**——把 AI 生成的内容直接写进稿件却没人核对过。`ai_use_disclosure` 审计一份逐阶段的 AI 使用日志,专门抓这条红线,并按目标期刊政策族渲染一段可直接粘贴的披露声明。第一版给一份**含红线**的日志:分析阶段的 AI 产出被采纳但未核验,政策族选 ICMJE。

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
print("\n『已采纳但未核验』命中:")
for u in disc["audit"]["unverified"]:
    print("  -", u["stage"], "/", u["tool"])

# %% [markdown]
# 审计给出 `ESCALATE`:analysis 那条被抓了出来,提交前必须核验或删除。闸门同时把整份日志整理成一张表(可直接进稿件附录),红线行带着 `accepted-but-unverified` 标记:

# %%
st_fixed.artifacts["tables"]

# %% [markdown]
# 无论审计是否命中红线,闸门都会渲染一段可粘贴进稿件的披露声明——措辞跟着政策族走:

# %%
print("ICMJE 政策族的披露声明:\n")
print(disc["statement"])

# %% [markdown]
# ### 换政策族、全部核验后转 PASS
#
# 把同一类工作换成**全部已核验**、政策族换成 **COPE**——审计从 `ESCALATE` 降为 `PASS`,声明也自动换成 COPE 的措辞。这演示了这道闸门的两个维度:红线审计管**诚信**,政策族渲染管**格式**,两者互不干扰。

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
print("\nCOPE 政策族的披露声明:\n")
print(disc_c["statement"])

# %% [markdown]
# ## 治理仪表盘:一图看两道闸门
#
# 把两道数值型闸门的关键判据画成一张仪表盘,让审稿人或合作者一眼看清合规态势:左边是 k-匿名如何随准标识符逐步粗化而改善、越过阈值转绿;右边是多源许可各自的抓取/再分发权利。左图的三档 k 是**真实重算**出来的(不是硬编码),我们对三种粗化程度分别再跑一次 `ethics_check`。图存成同目录 PNG,再用 Markdown 引用。

# %%
import matplotlib.pyplot as plt

# 复用包内的 CJK 字体探测(与 sv.pl 一致):有则用,无则回退 DejaVu(不报错)
try:
    from socialverse.pl._figure import _cjk_fonts
    plt.rcParams["font.sans-serif"] = _cjk_fonts() + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:  # pragma: no cover
    pass

# 左图数据:k 随准标识符粗化的阶梯(每档都真实重算一次 ethics_check)
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
ax1.set_title("① 伦理闸门:k-匿名随粗化而改善")
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
#
# 左图把整条补救逻辑压成三根柱子:准标识符越粗,最小等价类越大,`strata+psu`(k=2,红)→ `psu`(k=10,红)→ `strata`(k=97,绿)越过阈值转绿。右图里只有 Twitter 是红的,却足以把整盘拖成「不可抓取」——这就是最弱环。

# %% [markdown]
# ## 可复现的证据链
#
# 这一节轻描淡写地看一眼 `socialverse` 与普通合规脚本的关键差别。整条治理链跑下来,`StudyState` 自动攒了一份账本:每道闸门用了哪个函数、读了哪个槽、把判决写进了哪里。`summary()` 一览被填的槽和步数,`governance` 槽则是一站式的合规回执——「能不能上 / 能不能分发 / 怎么披露」三个结论都在这里,而且每个都能顺着账本回到它读的原始数据或日志。在社会科学里,「结论从哪一步、哪份数据来」常常和结论本身同等重要。

# %%
print(st_fixed.summary())

print("\ngovernance 槽的三道闸门判决(一站式合规回执):")
print("  ethics.verdict       :", st_fixed.governance["ethics"]["verdict"])
print("  data_use.bucket      :", st_fixed.governance["data_use"]["bucket"],
      "· can_scrape:", st_fixed.governance["data_use"]["can_scrape"])
print("  ai_disclosure.status :", st_fixed.governance["ai_disclosure"]["audit"]["status"])

# %% [markdown]
# 账本里逐条记着每一步的契约——读了什么(requires)、产出了什么(produces)——所以任一合规结论都能一路溯源。这份轨迹可以直接放进稿件的方法/伦理部分与 IRB 材料。

# %%
for r in st_fixed.provenance:
    req = ", ".join(f"{s}[{','.join(k)}]" for s, k in r["requires"].items()) or "∅"
    pro = ", ".join(f"{s}[{','.join(k)}]" for s, k in r["produces"].items()) or "∅"
    print(f"step {r['step']}: {sv.utils._friendly(r['function'])}")
    print(f"        requires {req}")
    print(f"        produces {pro}")

# %% [markdown]
# ## 小结
#
# 我们走完了一条完整的治理链:声明单位 → 伦理闸门(NO-GO → 补救 → PASS,含 k=1 红线)→ 许可五桶分诊(单源 / UNKNOWN 最严 / 多源最弱环)→ AI 披露审计与声明 → 仪表盘 → 证据链。它对标现实里分散在多处的合规实践:k-匿名靠 **ARX / sdcMicro** 或 `pandas`,许可判断靠人读 rightsstatements.org 与期刊 TDM 条款,AI 披露照抄 **ICMJE / COPE** 模板。
#
# 与把这些各自散落的做法相比,`socialverse` 多给了两样东西:治理是**会真的拦住你**的一等公民闸门(k-匿名分 PASS/FIX/NO-GO、许可未知默认落最严桶、AI 披露专抓「已采纳但未核验」),而不是一份可以选择性忽略的清单;三道闸门的判决又自动汇进同一份可审计的证据链,和后续的分析结论挂在一起。下一本教程 [09_literature_citation](09_literature_citation.ipynb) 转向文献与引证:检索、三库核验揪出幻觉引用、稿件引文审计。
