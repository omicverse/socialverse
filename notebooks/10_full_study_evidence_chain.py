# %% [markdown]
# # 一次可复核的小型研究:从伦理审查到证据链
#
# 前面几本教程各自讲一个方法——双重差分、复杂抽样、文本分析。这一本把它们背后的工作方式串成一次**完整而微型的政策评估**:某项政策在 2015 年覆盖了一批企业,我们要估计它对企业结果变量的**平均处理效应(ATT)**,并且要让整个分析在做完之后能**自己交代清楚**:数据从哪来、有没有过伦理关、用了什么识别假设、结论是不是被允许读作因果。
#
# 之所以要这样走,是因为社会科学的可信度往往不在那个点估计上,而在它周围的一整套程序里。一个 `-0.73` 谁都算得出来;真正决定它能不能进论文的,是「你有没有先确认能碰这批数据」「有没有检验平行趋势这个前提」「换一种标准误设定结论还稳不稳」「别人能不能照着你的记录复现」。所以这本教程的主线不是「怎么估 DID」——那在 [02_causal_did](02_causal_did.ipynb) 里已经讲过——而是**怎么把一次分析做成一份可复核的底稿**。
#
# 全流程用 `socialverse` 完成。它是一套面向社会科学的分析库:每个方法都是一个登记在案的函数,运行时会校验「这一步需要的前置是否就绪」,并把每一步默默记进一份出处台账。数据是内置的合成面板(40 家企业 × 8 年,真实 ATT = −0.8),干净、可控,方便我们把每一环看清楚。我们要走的链是:**过伦理闸门 → 载入数据 → 声明设计 → 检验平行趋势 → 估计 ATT → 出图 → 导出证据链**。

# %%
import matplotlib
matplotlib.use("Agg")  # 无显示环境:图直接写文件

import json
import os

import pandas as pd

import socialverse as sv
from socialverse import datasets as ds

# 图存到 notebook 同目录,markdown 里的 ![](fig_xxx.png) 才能就近解析,
# 无论从哪个目录跑脚本都一致
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:  # 交互式内核里没有 __file__
    _HERE = os.getcwd()


def figpath(name: str) -> str:
    return os.path.join(_HERE, name)


# %% [markdown]
# ## 声明研究目标
#
# 一次可复核的研究,起点不是「加载数据然后开跑」,而是先把要估什么讲清楚。我们要估的是**平均处理效应 ATT**(而不是单纯的相关),结果变量是 `y`。这两样来自研究问题本身,没有任何函数能替你产出,所以我们一开始就把它们写进研究状态 `StudyState`——它就像这次分析的共享笔记本,后续每一步都从这里读上下文、往这里写结果。

# %%
st = sv.StudyState()
st.write("estimand", "target", "ATT")   # 目标量:平均处理效应,不是相关
st.write("variables", "outcome", "y")   # 结果变量列名

print("研究状态初始化:", repr(st))

# %% [markdown]
# ## 载入面板数据
#
# 数据是一个**企业 × 年份**的长格式面板:每行是一家企业在某一年的观测。处理组企业在 `first_treated` 标记的年份(2015)被政策命中,`treat_post` 标记「已经受处理」的那些观测,`y` 是结果变量。数据生成时注入了真实 ATT = −0.8,后面我们看估计能不能把它找回来。
#
# 载入后用 `sv.pp.ingest` 把这张表登记进研究状态,之后所有步骤都从状态里取数据,不再手动传来传去。

# %%
df = ds.load_did_panel(att=-0.8)
sv.pp.ingest(st, data=df, name="did_panel_firm_year")

print("面板维度:", df.shape, "| 列:", list(df.columns))
df.head(6)

# %% [markdown]
# ## 治理闸门:先过伦理审查
#
# 社科研究里,**能不能碰这批数据**是先于「怎么分析」的问题。`ethics_check` 会对四件事各跑一遍真实检查——IRB 分类、知情同意、**可识别性(真算 k-匿名)**、数据最小化——折成一个 `PASS / FIX / NO-GO` 判决。它要求你先声明**分析单元**是什么,因为「一行数据代表一个什么」直接决定了这是不是涉及个人隐私。
#
# 我们先**故意犯一个常见的手滑**:把单元写成 `"firm-year"`(带连字符),看闸门怎么反应。

# %%
st.write("design", "unit", "firm-year")   # 手滑:连字符写法
sv.gov.ethics_check(st, data=df, quasi_identifiers=["treat"])

ethics = st.governance["ethics"]
print("伦理判决:", ethics["verdict"], "| 判定为人类受试者:", ethics["human_subjects"])
for c in ethics["checks"]:
    print(f"  - {c['check']:<12} {c['status']:<6} :: {c['detail']}")

# %% [markdown]
# 判决是 **NO-GO**,而且它把单元当成了「人类受试者」。原因很直白:`"firm-year"` 不在它认识的非人类单元集合里(`firm`、`country`、`document` 都在,带连字符的 `firm-year` 不在),于是它保守地按「可能涉及个人数据」处理,在没有 IRB 记录、没有知情同意的情况下给 NO-GO。这不是 bug,是**保守默认**:拿不准就按最严的算。真正卡住的是 IRB 和知情同意两项——k-匿名那项本身是 PASS。
#
# 正确的修法是把单元如实声明为 `"firm"`,并补上治理事实:这是一份企业面板、来自公开监管数据、已经最小化到所需变量。重跑闸门。

# %%
st.write("design", "unit", "firm")        # 如实声明:分析单元是「企业」
sv.gov.ethics_check(
    st,
    data=df,
    quasi_identifiers=["treat", "year"],  # 用组合准标识做真 k-匿名
    irb="exempt",                          # 企业监管面板:IRB 豁免
    consent="public",                      # 公开数据,无需个体知情同意
    minimized=True,                        # 已删除直接标识,只留所需变量
)
ethics = st.governance["ethics"]
print("修正后判决:", ethics["verdict"], "| 人类受试者:", ethics["human_subjects"])
for c in ethics["checks"]:
    print(f"  - {c['check']:<12} {c['status']}")
print("k-匿名 k =", ethics["k_anonymity"]["k"],
      "(阈值", ethics["k_anonymity"]["k_threshold"], ")")

assert ethics["verdict"] == "PASS", "闸门未放行,不应继续分析"
print("闸门放行 ✓ —— 现在才被允许进入因果分析")

# %% [markdown]
# 判决翻成 **PASS**,分析单元被正确识别为非人类的企业,四项检查全绿,k-匿名 k = 20 也在阈值之上。闸门是链条的一部分,不是事后补的合规文档——这一点在最后导出证据链时会看得很清楚。

# %% [markdown]
# ## 声明研究设计
#
# 过了闸门,下一步是把数据里的**具体列名**翻译成因果分析读得懂的角色:谁是单位、谁是时间、哪列标记处理、哪列记首次受处理的年份。`declare_design` 把这套映射登记进研究状态,后续的估计器读的是这些角色,而不是去猜列名。它还会顺手校验这些列是否真的在数据里。

# %%
sv.pp.declare_design(
    st,
    panel_id="firm_id",
    time="year",
    treatment="treat_post",       # 处理×时点交互(数据里已预先算好)
    first_treated="first_treated",
)
for k in ("unit", "panel_id", "time", "treatment", "first_treated"):
    print(f"  design['{k}'] = {st.design.get(k)!r}")
print("列名校验警告:", st.design.get("warnings", "无(所有列都在数据中)"))

# %% [markdown]
# ## 检验平行趋势
#
# 这是整条因果链的门槛。DID 能被读作因果,靠的是**平行趋势**假设:如果没有政策,处理组和对照组的结果本会沿平行的轨迹演化。这个前提没法直接观测,但可以用处理前若干期的「前趋势」间接考察——`parallel_trends` 估一个含单位固定效应和时间固定效应的事件研究,再对**所有处理前**的相对期系数做一次联合 Wald 检验。
#
# 原假设是「前导期系数全为 0」,也就是两组处理前趋势平行。若 `p > 0.05`(不拒绝),判定为 `pass`,识别前提站得住;若 `p` 很小,前趋势已经发散,后面即便算得出系数也不该叫「因果」。我们的合成数据是带真实平行前趋势生成的,应当通过。

# %%
sv.tl.parallel_trends(st)

pt = st.diagnostics["pretrend"]
print(f"联合 F = {pt['joint_F']:.3f}   p = {pt['p_value']:.3f}   (前导期数 = {pt['n_pre']})")
print("判定:", st.identification["parallel_trends"], "—", pt["note"])
print("\n前导期系数(应都≈0):")
for period, (coef, se) in pt["pre_coefs"].items():
    print(f"  event_time={period:>3}:  β={coef:+.4f}  (se={se:.4f})")

# %% [markdown]
# `p = 0.755` 远大于 0.05——处理前各期系数联合不显著,平行趋势成立。前导期系数也都贴着零线。这个判定被写进了研究状态,成为下一步 `did` 能否运行的前置条件:没有它,`did` 会拒绝运行。

# %% [markdown]
# ## 估计 ATT
#
# 现在可以估了。`did` 拟合一个双向固定效应模型(`y ~ treat_post + 单位固定效应 + 时间固定效应`),按 `firm_id` 聚类计算稳健标准误——处理效应的推断通常要在单位层面聚类,才容许同一企业跨年的相关。它同时把上一步的平行趋势判定读进结论:通过则标注为「因果 ATT」,没通过则降级为「关联,非因果」。

# %%
sv.tl.did(st)

m = st.models["did"]
print(f"ATT    = {m['att']:+.4f}   (真值注入 = -0.8000)")
print(f"SE     = {m['se']:.4f}   (聚类于 {m['n_clusters']} 家企业)")
print(f"95% CI = [{m['ci'][0]:.4f}, {m['ci'][1]:.4f}]")
print(f"p      = {m['p']:.2e}")
print(f"n      = {m['n']}   估计量 = {m['estimator']}")
print("结论   :", m["note"])

# %% [markdown]
# 估计的 ATT ≈ **−0.73**,95% 置信区间 `[−0.93, −0.53]` 落在负区间、不跨 0,且覆盖了注入的真值 −0.8。因为平行趋势通过了,这个数**被允许**读作因果 ATT:政策使结果变量显著下降。

# %% [markdown]
# ## 稳健性:标准误对设定的敏感度
#
# 点估计定下来后,一个常规检查是看标准误在不同方差设定下稳不稳。`did` 顺带在三种设定下重估了同一个 ATT——经典(同方差)、异方差稳健(HC1)、按企业聚类。点估计三者完全一致(方差设定不改点估计),聚类 SE 通常最大也最可信,因为它容许企业内部跨年的相关。

# %%
rob = pd.DataFrame(st.diagnostics["robustness"]["specs"])
rob

# %% [markdown]
# ## 展开动态效应
#
# ATT 是一个「平均」。事件研究把它按相对处理时点展开:以处理前一期(−1)为基准,给出每个相对期的系数。处理前的系数再一次让我们检视前趋势是否贴零线,处理后的系数则刻画政策生效后效应如何随时间演化。

# %%
sv.tl.event_study(st)

es = st.models["event_study"]
pd.DataFrame(
    [{"相对时点": int(k), "系数": round(v[0], 3), "SE": round(v[1], 3)}
     for k, v in es["coefs"].items()]
)

# %% [markdown]
# ## 可视化
#
# 结果就绪后,绘图函数直接从研究状态里读数出图,不用手工整理。画两张:ATT 的森林图,以及动态效应的事件研究图。图存成同目录的 png。

# %%
sv.pl.forest(st, out=figpath("fig_study.png"),
             title="政策 ATT · 点估计 ± 95% CI")
sv.pl.event_study_plot(st, out=figpath("fig_eventstudy.png"))
print("图已保存:fig_study.png, fig_eventstudy.png")

# %% [markdown]
# **森林图**——单个 ATT 系数与 95% 置信区间,虚线为零效应参考:
#
# ![森林图](fig_study.png)
#
# **事件研究图**——前导期贴着零线(平行趋势成立),处理时点 0 之后跳到约 −0.8 并维持:
#
# ![事件研究图](fig_eventstudy.png)

# %% [markdown]
# ## 导出证据链
#
# 到这里研究「做完了」,但可复核研究的真正交付物不只是那个 −0.73,而是**这条链能自证它怎么来的**。跑到这一步,研究状态里已经自动积累了一份**只增不改的出处台账**:每个函数成功执行时,都会追加一条记录——它是第几步、消费了哪些槽、产出了哪些槽。你没有额外写一行日志,这份台账是分析过程的副产品。
#
# 打出来,就是一条从「伦理放行」到「ATT 估计」、依赖前后咬合的证据链。

# %%
prov_rows = []
for p in st.provenance:
    prov_rows.append({
        "step": p["step"],
        "function": p["function"].split(".")[-1],
        "requires": ", ".join(f"{s}.{k}" for s, ks in p["requires"].items() for k in ks) or "∅",
        "produces": ", ".join(f"{s}.{k}" for s, ks in p["produces"].items() for k in ks) or "∅",
    })
pd.DataFrame(prov_rows)

# %% [markdown]
# 读这条链:每一步的 `produces` 恰好喂饱后面某一步的 `requires`——`ingest` 产出 `sources.datasets` 给 `declare_design` 用;`declare_design` 产出 `design.*` 给 `parallel_trends` 用;`parallel_trends` 产出 `identification.parallel_trends`,而 `did` 正是要它。依赖前后咬合、无缺环,这就是「可复核」的机器可读形态。注意 `ethics_check` 出现了**两次**(先犯错的声明 + 修正后的声明)——台账只增不改,连「先犯错再改对」都如实留痕;而且伦理检查排在因果分析之前,证明治理不是事后补的。
#
# 这份台账可以直接导出成 JSON,作为论文附录里的「分析出处」随文存档。

# %%
prov_json_path = figpath("evidence_chain_provenance.json")
with open(prov_json_path, "w", encoding="utf-8") as fh:
    json.dump(st.provenance, fh, ensure_ascii=False, indent=2)
print("出处台账已导出:", os.path.basename(prov_json_path),
      f"({len(st.provenance)} 步)")

# %% [markdown]
# ## 最终快照
#
# `st.summary()` 把整份研究底稿一览:哪些槽被填了、填了什么、出处几步。这就是一次可复核研究跑完后该留下的东西——不是一个孤零零的数字,而是**数据来源 → 治理放行 → 设计声明 → 识别假设检验 → 估计 → 出图**的一条闭环,外加一份可存档的出处台账。

# %%
print(st.summary())

print("\n本次研究导出的可复核文件:")
for pth in (figpath("fig_study.png"), figpath("fig_eventstudy.png"), prov_json_path):
    exists = "✓" if os.path.exists(pth) else "✗"
    print(f"  [{exists}] {os.path.basename(pth)}")

# %% [markdown]
# ## 小结
#
# 我们走完了一次完整的政策评估闭环:过伦理闸门 → 载入 → 声明设计 → 检验平行趋势 → 估计 ATT → 出图 → 导出证据链。拆开看,每一步都有成熟的现实对应物:治理闸门 ≈ IRB 表单 + `sdcMicro` 的 k-匿名;因果估计 ≈ `pyfixest` / R 的 `fixest` 与 `did`(Callaway–Sant'Anna);出图 ≈ `matplotlib` / `coefplot`。
#
# 与把这些工具各自拼起来相比,`socialverse` 多给了两样东西,而且都不用你额外写代码:平行趋势是一道**会真的拦住你**的门槛(没通过时 `did` 直接拒绝运行,而不是默默给你一个未识别的系数),以及一份**贯穿始终、只增不改的证据链**(直接导出即为论文附录级的分析出处)。别的工具让你**做出**一个结果,`socialverse` 让这个结果自己带着它是怎么来的、依赖什么、有没有越级。
#
# 这是本系列的收束。想回看单个方法的细节,可以从因果那本 [02_causal_did](02_causal_did.ipynb) 重新进入。
