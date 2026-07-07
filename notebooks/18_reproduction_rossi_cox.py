# %% [markdown]
# # 完整复现:Rossi 累犯实验的全部生存分析
#
# 这一本用**真实公开数据**把一项已发表研究的**完整数据分析**从头到尾跑一遍——不是单个模型,而是这份数据的整套经典分析:描述统计 → 分组生存曲线与检验 → 主 Cox 模型 → 比例风险诊断 → 简约模型 → **时变协变量(就业)扩展**。并让 socialverse 的每个数字与文献发表值对上。
#
# **研究**:Rossi, Berk & Lenihan (1980) *Money, Work, and Crime* —— 432 名刚出狱的重罪犯,一半随机获得过渡期**财务援助**(`fin`),追踪出狱后一年内**是否再次被捕**(`arrest`)及**再捕周数**(`week`)。其生存分析是 Allison (2014) 与 Fox & Weisberg (*Cox Regression in R*) 的经典范例。数据取自公开的 [Rdatasets](https://vincentarelbundock.github.io/Rdatasets/)(`carData::Rossi`)。
#
# **这条链走的正是顶刊论文的通用解剖**:数据(`sv.pp`)→ 设计(`declare_design`)→ 主分析与诊断(`sv.tl`)→ 图表(`sv.pl`)→ 治理(`sv.gov`)→ 证据链。

# %%
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

import socialverse as sv

pd.set_option("display.float_format", lambda v: f"{v:.4f}")

URL = "https://vincentarelbundock.github.io/Rdatasets/csv/carData/Rossi.csv"
raw = pd.read_csv(URL)
# carData 因子 → Allison 的 0/1 数值编码
_enc = {"fin": {"yes": 1, "no": 0}, "race": {"black": 1, "other": 0},
        "wexp": {"yes": 1, "no": 0}, "mar": {"married": 1, "not married": 0},
        "paro": {"yes": 1, "no": 0}}
for c, m in _enc.items():
    if raw[c].dtype == object:
        raw[c] = raw[c].map(m)

cov = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]
df = raw[["week", "arrest"] + cov].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

# %% [markdown]
# ## 1. 数据与样本(描述统计)
#
# 先看整体:多少人、多少再捕事件、删失比例;再按是否获得财务援助分组做"表 1"。随机化良好的话,两组的协变量应大致均衡。

# %%
n, k = len(df), int(df["arrest"].sum())
print(f"样本 {n} 名释放者 · 再捕事件 {k} 起({k/n:.1%})· 删失(未再捕){n-k} 人({1-k/n:.1%})· 追踪至多 {int(df['week'].max())} 周")

tab1 = df.groupby("fin")[["age", "race", "wexp", "mar", "paro", "prio", "arrest"]].mean().T
tab1.columns = ["无援助 (fin=0)", "有援助 (fin=1)"]
tab1["全样本"] = df[["age", "race", "wexp", "mar", "paro", "prio", "arrest"]].mean()
print("\n表 1:协变量均值 / 比例(按财务援助分组)")
tab1

# %% [markdown]
# 随机化奏效:两组在年龄、种族、前科等协变量上相近。原始再捕率在有援助组略低(下节正式检验)。

# %% [markdown]
# ## 2. 分组生存曲线与 log-rank 检验(`sv.pl.km_curve`)
#
# 登记进 `StudyState`,按财务援助分组估 Kaplan-Meier 未再捕生存曲线,并用 **log-rank(Mantel-Cox)检验**两组曲线是否相同。socialverse 的 `survival` 会一并算出 KM 曲线、log-rank、Cox 模型与 PH 诊断。

# %%
st = sv.StudyState()
sv.pp.ingest(st, data=df, name="rossi")
st.write("variables", "outcome", "arrest")
st.write("design", "unit", "released_prisoner")
st.write("design", "duration", "week")
st.write("design", "treatment", "fin")

sv.tl.survival(st, time="week", event="arrest", covariates=cov, group="fin")

lr = st.models["km"]["logrank"]
print(f"log-rank(按 fin):χ² = {lr['chi2']:.3f} · p = {lr['p']:.3f} · df = {lr['df']}")
sv.pl.km_curve(st, out="fig_rossi_km.png", group="fin")
print("KM 曲线已保存:fig_rossi_km.png")

# %% [markdown]
# **KM 曲线**(未获援助 vs 获援助的未再捕生存比例):
#
# ![KM](fig_rossi_km.png)
#
# log-rank p ≈ 0.05:财务援助对再捕的边际显著保护效应——与原研究"效应存在但不强"的结论一致。

# %% [markdown]
# ## 3. 主模型:Cox 比例风险(全协变量)
#
# 论文的核心表:再捕风险 ~ 财务援助 + 年龄 + 种族 + 工作经历 + 婚姻 + 假释 + 前科数。系数是对数风险比(log-HR),`exp(β)` 是风险比。下面把 socialverse 的系数与 **Allison (2014) 发表值**逐位对照。

# %%
m = st.models["cox"]
published = {"fin": -0.379, "age": -0.057, "race": 0.314, "wexp": -0.150,
             "mar": -0.434, "paro": -0.085, "prio": 0.091}
rows = []
for c in cov:
    v = m["log_hr"][c]
    beta = v[0] if isinstance(v, (list, tuple)) else v
    rows.append({"协变量": c, "sv log-HR": beta, "Allison 2014": published[c],
                 "HR": np.exp(beta), "|偏差|": abs(beta - published[c])})
tbl = pd.DataFrame(rows).set_index("协变量")
print(f"n = {m['n']} · 事件 = {m['n_events']} · 与已发表值最大偏差 = {tbl['|偏差|'].max():.4f}")
tbl

# %% [markdown]
# **逐位吻合**(最大偏差 < 0.002)。前科数 `prio` 与种族 `race` 显著提升再捕风险;已婚 `mar`、年龄 `age`、财务援助 `fin` 降低风险。财务援助 HR ≈ 0.68(风险降约 32%,边际显著)。

# %% [markdown]
# ## 4. 比例风险假设诊断
#
# Cox 的关键假设:各协变量的效应不随时间变化。`sv.tl.survival` 附带 Grambsch-Therneau(Schoenfeld 残差)检验——全局 + 逐协变量。

# %%
ph = st.diagnostics["ph_test"]
print(f"全局 PH 检验:χ² = {ph['global_chi2']:.2f} · p = {ph['global_p']:.3f} → {ph['verdict']}")
print(f"方法:{ph['method']}")
pc = ph.get("per_covariate")
if isinstance(pc, dict):
    display(pd.DataFrame({"PH p 值": {k: (v[1] if isinstance(v, (list, tuple)) else v) for k, v in pc.items()}}))

# %% [markdown]
# 全局 p ≈ 0.08 > 0.05:整体上比例风险假设可接受(个别协变量若 p 偏小,可在稳健性里加时间交互——这也正是下一节时变模型要处理的)。

# %% [markdown]
# ## 5. 简约模型(只留稳健预测因子)
#
# 顶刊常报一个简约规格以示结果不依赖冗余控制。这里只留财务援助、年龄、前科数,重估 Cox。

# %%
st_r = sv.StudyState()
sv.pp.ingest(st_r, data=df)
st_r.write("variables", "outcome", "arrest")
sv.tl.survival(st_r, time="week", event="arrest", covariates=["fin", "age", "prio"])
mr = st_r.models["cox"]
pd.DataFrame({
    "log-HR": {c: (mr["log_hr"][c][0] if isinstance(mr["log_hr"][c], (list, tuple)) else mr["log_hr"][c]) for c in ["fin", "age", "prio"]},
    "HR": {c: mr["hr"][c] if not isinstance(mr["hr"][c], (list, tuple)) else mr["hr"][c][0] for c in ["fin", "age", "prio"]},
})

# %% [markdown]
# 简约模型下三个系数与全模型方向、量级一致,结论稳健。

# %% [markdown]
# ## 6. 扩展:时变协变量——就业状态(Andersen-Gill Cox)
#
# 原研究最重要的机制发现是**就业**:数据里 `emp1…emp52` 记录了每一周是否在业。要把"就业"作为**随时间变化**的协变量放进 Cox,需把每名释放者展开成"人-周"(person-period)长表,再用计数过程(Andersen-Gill)Cox 拟合。
#
# > ⚠️ **这一步超出 `sv.tl.survival` 当前的标准 Cox**(它按每人一行拟合,不支持时变协变量)。下面用 `statsmodels.PHReg`(左截断 `entry=` 实现区间在险)直接做,并把**时变 Cox(Andersen-Gill)标注为 socialverse survival 的下一步增强**。这正体现注册表的价值:能诚实标出"论文用了但库还没有"的那一步。

# %%
import statsmodels.api as sm

_empmap = {"yes": 1, "no": 0}
def _emp(v):
    return 0 if pd.isna(v) else _empmap.get(v, v)

pp = []
for _, r in raw.iterrows():
    T = int(r["week"])
    for wk in range(1, T + 1):
        pp.append({"start": wk - 1, "stop": wk,
                   "arrest": int(r["arrest"]) if wk == T else 0,
                   "employed": _emp(r.get(f"emp{wk}")),
                   "fin": r["fin"], "age": r["age"], "prio": r["prio"]})
pp = pd.DataFrame(pp).apply(pd.to_numeric, errors="coerce").dropna()
X = pp[["fin", "age", "prio", "employed"]]
res = sm.PHReg(pp["stop"], X, status=pp["arrest"], entry=pp["start"], ties="breslow").fit()

print(f"人-周展开:{len(pp)} 行(来自 {len(raw)} 名释放者)")
pd.DataFrame({"log-HR": dict(zip(X.columns, res.params)),
              "HR": dict(zip(X.columns, np.exp(res.params))),
              "SE": dict(zip(X.columns, res.bse))})

# %% [markdown]
# **关键发现**:`employed`(当周在业)log-HR ≈ −1.36,**HR ≈ 0.26**——在业的那些周,被捕风险仅为无业周的约四分之一。引入时变就业后,财务援助 `fin` 的效应有所减弱(从 −0.38 到约 −0.33):财务援助的部分作用是**经由促进就业**传导的。这与 Fox & Weisberg 的经典结论一致,也是这份数据最重要的实质发现。

# %% [markdown]
# ## 7. 治理与证据链
#
# 随机对照人类被试实验:伦理闸门 + 数据使用合规。最后打印 provenance 账本——从原始数据到发表级系数,每步可追溯。

# %%
sv.gov.ethics_check(st, data=df, quasi_identifiers=["age", "race", "mar"])
sv.gov.data_use_check(st, license="public-domain")
print("伦理:", (st.governance.get("ethics") or {}).get("verdict"),
      "· 数据使用:", (st.governance.get("data_use") or {}).get("bucket"), "\n")
print(st.summary())

# %% [markdown]
# ## 小结
#
# 我们用真实公开数据,把 Rossi 累犯实验的**整套生存分析**端到端复现了一遍:
#
# | 分析 | socialverse | 结果 vs 文献 |
# |---|---|---|
# | 描述统计 / 表 1 | `sv.pp.ingest` + pandas | 随机化均衡 ✓ |
# | KM + log-rank | `sv.tl.survival` + `sv.pl.km_curve` | χ²=3.84, p≈0.05 ✓ |
# | 全模型 Cox | `sv.tl.survival` | 与 Allison 2014 **逐位吻合(≤0.002)** ✓ |
# | PH 诊断 | `ph_test`(Schoenfeld) | 全局 p≈0.08 通过 ✓ |
# | 简约模型 | `sv.tl.survival` | 结论稳健 ✓ |
# | 时变就业 Cox | ⚠️ statsmodels(超出 sv 标准 Cox) | employed HR≈0.26 ✓ |
#
# 六步里五步 socialverse 原生跑通、结果与发表值一致;唯一超出的是**时变协变量 Cox**,已诚实标注为 survival 的下一步增强。整条链
# `ingest → declare_design → survival(Cox+KM+log-rank+PH) → km_curve → gov → 证据链`
# 正是顶刊论文通用解剖的实例化——**一篇论文的完整数据分析,就是一条可执行、可核验、可审查的 socialverse 函数序列**。

# %%
print("完整复现完成 · 全模型 Cox vs Allison(2014) 最大偏差:", round(tbl["|偏差|"].max(), 4),
      "· 时变就业 HR:", round(np.exp(res.params[list(X.columns).index("employed")]), 3))
