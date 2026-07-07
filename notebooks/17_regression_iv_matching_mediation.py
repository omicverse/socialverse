# %% [markdown]
# # 回归底座与因果工具箱(详解):GLM、多项/有序、边际效应、工具变量、匹配、中介
#
# 前面几本教程覆盖的是专门方法(DiD、断点、SEM、生存……)。这一本补的是社会科学里**最日常**的一批命令——广义回归、多项与有序结果、边际效应、工具变量、倾向得分匹配、因果中介。它们是 R 用户的 `glm`/`multinom`/`polr`/`margins`/`ivreg`/`MatchIt`/`mediation`,Stata 用户的 `logit`/`mlogit`/`ologit`/`margins`/`ivregress`/`psmatch2`/`mediate`,几乎每篇实证论文都会用到。
#
# 本教程是**详解版**:每个方法都完整跑一遍,展示完整的系数表(点估计、标准误、95% 置信区间、z 值、p 值、拟合统计),给出如何解读,并和内置合成数据的**已知真值**对照,证明这些实现不是占位符。每一节开头标注它对应的 R / Stata / SPSS 命令,方便迁移。
#
# 所有函数都登记进 socialverse 注册表、operate on 同一个 `StudyState`。如果你来自 R/Stata/SPSS,可直接用命令名查:`sv.registry.get("py-logit")`、`py-ivregress`、`py-psmatch2`、`py-mediate` 都能命中对应函数。

# %%
import numpy as np
import pandas as pd

import socialverse as sv
from socialverse import datasets as ds

pd.set_option("display.float_format", lambda v: f"{v:.4f}")


def coef_table(model, drop_const=False):
    """把一个存了 coef/se/ci/z/p 的模型 dict 渲染成完整系数表(DataFrame)。"""
    rows = []
    for name, b in model["coef"].items():
        if drop_const and name == "const":
            continue
        se = model.get("se", {}).get(name)
        ci = model.get("ci", {}).get(name)
        z = (model.get("z") or model.get("t") or {}).get(name)
        p = model.get("p", {}).get(name)
        rows.append({
            "变量": name, "系数": b, "标准误": se,
            "CI下": ci[0] if ci else None, "CI上": ci[1] if ci else None,
            "z": z, "p": p,
        })
    return pd.DataFrame(rows).set_index("变量")


# %% [markdown]
# ## 1. 广义线性回归 `glm` — 一个函数覆盖 OLS / logit / probit / Poisson
#
# **对标**:R `stats::glm` · Stata `regress`/`logit`/`poisson` · SPSS `REGRESSION`/`GENLIN`。
#
# 社会科学的结果变量形态各异:连续的(收入)、二值的(是否投票)、计数的(抗议次数)。广义线性模型用一个 `family`(误差分布)+ 连接函数把它们统一起来:`gaussian` 是普通最小二乘(OLS),`binomial` 是 logit/probit,`poisson` 是计数模型。socialverse 的 `glm` 就是这一个入口,支持稳健(HC1)与聚类标准误。
#
# 合成数据 `load_regression` 内嵌已知真值:连续 `y` 的系数 x1=0.5、x2=−0.4;二值 `y_bin` 的 logit 真值 x1=0.8、x2=−0.5;计数 `y_count` 的 poisson 真值 x1=0.4。

# %%
df = ds.load_regression()
print("数据形状:", df.shape, "| 列:", list(df.columns))
df.head()

# %% [markdown]
# ### 1a. OLS(`family="gaussian"`)
#
# 最基本的线性回归。系数直接就是"x 增加一单位,y 平均变化多少"。

# %%
st = sv.StudyState()
sv.pp.ingest(st, data=df)
st.write("variables", "outcome", "y")
sv.tl.glm(st, predictors=["x1", "x2"], family="gaussian", cov="robust")

m = st.models["glm"]
print(f"估计量: {m['estimator']} · 标准误: {m['cov']} · n = {m['n']}")
print(f"拟合: R² = {st.diagnostics['glm_fit'].get('r2')}, AIC = {st.diagnostics['glm_fit']['aic']:.1f}")
print("真值: x1 = 0.5, x2 = -0.4\n")
coef_table(m)

# %% [markdown]
# 读表:`x1` 的系数落在真值 0.5 附近,`x2` 在 −0.4 附近,95% 置信区间都不含 0(p 值极小)。稳健标准误(HC1)容许异方差。

# %% [markdown]
# ### 1b. Logit(`family="binomial"`)
#
# 二值结果。**注意**:logit 系数是对数几率(log-odds)尺度,不能直接当概率变化读——正负号和显著性可以看,大小要靠下一节的边际效应来解释。

# %%
st_logit = sv.StudyState()
sv.pp.ingest(st_logit, data=df)
st_logit.write("variables", "outcome", "y_bin")
sv.tl.glm(st_logit, predictors=["x1", "x2"], family="binomial")

ml = st_logit.models["glm"]
print(f"family = {ml['family']} · McFadden pseudo-R² = {st_logit.diagnostics['glm_fit']['pseudo_r2']:.3f}")
print("真值(对数几率尺度): x1 = 0.8, x2 = -0.5\n")
coef_table(ml)

# %% [markdown]
# ### 1c. Poisson(`family="poisson"`)
#
# 计数结果。系数是对数发生率(log-rate)尺度:`exp(系数)` 是发生率比(incidence-rate ratio)。真值 x1=0.4,即 x1 每升一单位、事件发生率乘以 `exp(0.4)≈1.49`。

# %%
st_pois = sv.StudyState()
sv.pp.ingest(st_pois, data=df)
st_pois.write("variables", "outcome", "y_count")
sv.tl.glm(st_pois, predictors=["x1", "x2"], family="poisson")

mp = st_pois.models["glm"]
tbl = coef_table(mp, drop_const=True)
tbl["发生率比 exp(β)"] = np.exp(tbl["系数"])
print("真值: x1 = 0.4  → exp(0.4) ≈ 1.49\n")
tbl

# %% [markdown]
# ## 2. 平均边际效应 `margins` — 把非线性系数翻译成可解释的效应
#
# **对标**:Stata `margins` · R `marginaleffects::slopes` / `emmeans`。
#
# logit/poisson 的系数不在结果的自然尺度上。`margins` 在**刚拟合的模型**上计算平均边际效应(AME):对每个观测算一次偏效应再平均。对 logit,AME 就是"x 增加一单位,结果发生概率平均变化多少个百分点"——这才是能写进论文正文的量。它读取上一步 `glm` 存在 state 里的拟合对象。

# %%
sv.tl.margins(st_logit, model="glm")   # 在 1b 的 logit 上算
mg = st_logit.diagnostics["margins"]
print(f"模型: {mg['model']} · 求值点: {mg['at']}")
pd.DataFrame({"平均边际效应(概率尺度)": mg["ame"]})

# %% [markdown]
# 读表:x1 的 AME 为正、x2 为负——与 logit 系数同号,但现在是**概率**尺度(如 +0.19 表示 x1 每升一单位、事件概率平均升约 19 个百分点)。

# %% [markdown]
# ## 3. 多项 logit `mlogit` — 无序多类别结果
#
# **对标**:Stata `mlogit` · R `nnet::multinom` · SPSS `NOMREG`。
#
# 结果是三个及以上**无序**类别(如选了 A/B/C 三种方案)。多项 logit 以一个基准类别(base)为参照,给出其余每个类别相对基准的对数几率系数。合成数据里,选 B 的倾向随 x1 上升、选 C 的倾向随 x1 下降。

# %%
st_ml = sv.StudyState()
sv.pp.ingest(st_ml, data=df)
st_ml.write("variables", "outcome", "choice")
sv.tl.mlogit(st_ml, predictors=["x1"])

mm = st_ml.models["mlogit"]
print(f"基准类别: {mm['base']} · 类别: {mm['categories']} · n = {mm['n']} · logL = {mm['llf']:.1f}\n")
rows = []
for cat, params in mm["coef"].items():
    for var, cell in params.items():
        rows.append({"类别(vs %s)" % mm["base"]: cat, "变量": var,
                     "系数(log-odds)": cell["coef"], "标准误": cell["se"]})
pd.DataFrame(rows).set_index(["类别(vs %s)" % mm["base"], "变量"])

# %% [markdown]
# 读表:B 相对 A 的 `x1` 系数为**正**(x1 越大越倾向 B),C 相对 A 的 `x1` 系数为**负**(x1 越大越不选 C)——与数据生成机制一致。

# %% [markdown]
# ## 4. 有序 logit `ologit` — 有序多类别结果
#
# **对标**:Stata `ologit`/`oprobit` · R `MASS::polr` · SPSS `PLUM`。
#
# 结果是**有序**类别(如"不同意 < 中立 < 同意")。有序 logit 假设一个潜变量被若干切点(threshold)切成有序段,只估一组斜率系数 + 切点。正系数=预测变量越大、越可能落到更高等级。

# %%
st_ol = sv.StudyState()
sv.pp.ingest(st_ol, data=df)
st_ol.write("variables", "outcome", "y_ord")
sv.tl.ologit(st_ol, predictors=["x1", "x2"], link="logit")

mo = st_ol.models["ologit"]
print(f"{mo['estimator']} · {mo['n_categories']} 个有序等级 · n = {mo['n']}")
print("切点(thresholds):", {k: round(v, 3) for k, v in mo["thresholds"].items()}, "\n")
pd.DataFrame({"系数(潜变量尺度)": mo["coef"], "标准误": mo["coef_se"]})

# %% [markdown]
# 读表:`x1` 系数为正(x1 越大越可能落到更高的有序等级),`x2` 为负。两个切点把潜变量分成 3 段。

# %% [markdown]
# ## 5. 工具变量 `iv_regress` — 当回归量内生时
#
# **对标**:Stata `ivregress`/`ivreg2` · R `AER::ivreg`/`fixest` · SPSS `2SLS`。
#
# 如果关键回归量与误差项相关(遗漏变量、反向因果、测量误差),OLS 有偏。工具变量用一个只通过该回归量影响结果的"工具"来剥离内生性:两阶段最小二乘(2SLS)先用工具预测内生变量,再用预测值估效应,并用**正确的 2SLS 协方差**算标准误。
#
# 合成数据 `load_iv` 里,`x` 被未观测混杂污染,真实因果效应 = 1.5;OLS 会高估;工具 `z` 帮我们拉回来。**弱工具检验**:一阶段 F 远大于 10 才可信。

# %%
st_iv = sv.StudyState()
sv.pp.ingest(st_iv, data=ds.load_iv())
st_iv.write("variables", "outcome", "y")
sv.tl.iv_regress(st_iv, endogenous="x", instruments=["z"], exog=["w"])

miv = st_iv.models["iv"]
print(f"{miv['estimator']} · 标准误: {miv['cov_type']} · n = {miv['n']}")
print("真值: x 的因果效应 = 1.5\n")
coef_table(miv)

# %%
fs = st_iv.diagnostics["first_stage"]
pd.DataFrame({
    "指标": ["一阶段 F(排除性工具联合显著)", "是否弱工具(F<10)",
             "OLS 内生系数(有偏)", "2SLS 内生系数(修正)"],
    "值": [round(fs["F"], 1), fs["weak_instrument"],
           round(fs["ols_endog_coef"], 3), round(fs["iv_endog_coef"], 3)],
}).set_index("指标")

# %% [markdown]
# 读表:2SLS 把 `x` 的效应估到 ≈1.37(真值 1.5),而**有偏的 OLS 高估到 ≈2.57**;一阶段 F=624 ≫ 10,工具很强,不是弱工具。这正是纯 OLS 会误导、而工具变量能纠偏的场景。

# %% [markdown]
# ## 6. 倾向得分匹配 `psm` — 从观察数据估处理效应
#
# **对标**:Stata `teffects psmatch`/`psmatch2` · R `MatchIt`/`WeightIt`。
#
# 没有随机分配时,处理组与对照组在协变量上往往不可比。倾向得分匹配先用协变量估"接受处理的概率",再把处理单元与倾向相近的对照单元配对(或按逆概率加权),从而在可比子样本上估平均处理效应(ATT)。合成数据 `load_treatment` 里,处理由 x1..x3 驱动(选择性偏差),真实 ATT=2.0,朴素的处理组减对照组差值有偏。

# %%
st_psm = sv.StudyState()
sv.pp.ingest(st_psm, data=ds.load_treatment())
st_psm.write("variables", "outcome", "y")
st_psm.write("design", "treatment", "treat")

sv.tl.psm(st_psm, covariates=["x1", "x2", "x3"], method="nn")   # 最近邻 1:1
nn = st_psm.models["psm"]
sv.tl.psm(st_psm, covariates=["x1", "x2", "x3"], method="ipw")  # 逆概率加权
ipw = st_psm.models["psm"]

pd.DataFrame({
    "估计": ["ATT(最近邻匹配)", "ATT(逆概率加权 IPW)", "朴素差值(未调整)"],
    "值": [nn["att"], ipw["att"], nn["naive_diff"]],
    "标准误": [nn["se"], ipw["se"], None],
}).set_index("估计")

# %% [markdown]
# 匹配的意义是让协变量在两组间可比。诊断给出匹配前后的**标准化均值差(SMD)**,匹配后应显著变小(经验上 |SMD|<0.1 算平衡良好):

# %%
bal = st_psm.diagnostics["balance"]
pd.DataFrame({
    "协变量": list(bal["smd_before"]),
    "匹配前 SMD": [float(bal["smd_before"][c]) for c in bal["smd_before"]],
    "匹配后 SMD": [float(bal["smd_after"][c]) for c in bal["smd_before"]],
}).set_index("协变量")

# %% [markdown]
# 读表:匹配/加权后的 ATT 都在真值 2.0 附近(最近邻略低、IPW 略高),而**朴素差值 2.67 明显偏高**(被选择偏差推高)。匹配后 x1 的 SMD 从 0.83 降到近 0——协变量被拉平了。

# %% [markdown]
# ## 7. 因果中介 `mediation` — 效应经由什么传导
#
# **对标**:Stata `mediate`/`med4way` · R `mediation::mediate` · SPSS PROCESS(Hayes)。
#
# 知道 X 影响 Y 还不够,常要问"这个影响有多少是**经由**中介 M 传导的"。中介分析把总效应拆成:经由 M 的间接效应(ACME)与绕过 M 的直接效应(ADE)。socialverse 拟合中介模型(x→m,取 a)与结果模型(y→x+m,取 b 与直接效应 c'),用系数乘积 a·b 估 ACME,并用**非参数 bootstrap** 给置信区间(还可选用 statsmodels 的 Mediation 做交叉校验)。
#
# 合成数据 `load_mediation`:a=0.6、b=0.7 → 间接 ACME=0.42、直接 ADE=0.30、总 0.72。

# %%
st_med = sv.StudyState()
sv.pp.ingest(st_med, data=ds.load_mediation())
st_med.write("variables", "outcome", "y")
sv.tl.mediation(st_med, treatment="x", mediator="m", boot=1000, seed=0)

med = st_med.models["mediation"]
print(f"路径: a(x→m)={med['a']:.3f}, b(m→y)={med['b']:.3f}, 直接 c'={med['direct']:.3f}")
print(f"bootstrap: {med['boot']} 次 · 交叉校验({med['crosscheck']['source'].split('.')[-1]}): ACME={med['crosscheck']['acme']:.3f}\n")
pd.DataFrame({
    "效应": ["间接 ACME (a·b)", "直接 ADE (c')", "总效应", "中介占比"],
    "估计": [med["acme"], med["ade"], med["total"], med["prop_mediated"]],
    "95%CI下": [med["ci_acme"][0], med["ci_ade"][0], med["ci_total"][0], None],
    "95%CI上": [med["ci_acme"][1], med["ci_ade"][1], med["ci_total"][1], None],
    "真值": [0.42, 0.30, 0.72, 0.42 / 0.72],
}).set_index("效应")

# %% [markdown]
# 读表:间接效应 ACME≈0.41(真值 0.42)、直接 ADE≈0.29(真值 0.30)、总≈0.70(真值 0.72),bootstrap 置信区间都不含 0;约 59% 的总效应经由中介 M 传导。statsmodels 的独立实现交叉校验给出几乎一致的 ACME,佐证实现正确。

# %% [markdown]
# ## 小结
#
# 这一本详解了社会科学最高频的量化底座:广义回归(`glm` 覆盖 OLS/logit/probit/poisson)、多项与有序结果(`mlogit`/`ologit`)、边际效应(`margins`)、工具变量(`iv_regress`)、倾向得分匹配(`psm`)、因果中介(`mediation`)。每个都给出完整系数表、诊断,并在带已知真值的数据上验证——不是占位实现。加上前面的专门方法,socialverse 现在覆盖从"基础回归"到"前沿准实验"的主干。
#
# 迁移提示:它们对标 R 的 `glm`/`multinom`/`polr`/`margins`/`ivreg`/`MatchIt`/`mediation` 与 Stata 的 `logit`/`mlogit`/`ologit`/`margins`/`ivregress`/`psmatch2`/`mediate`。用熟悉的命令名即可查到:

# %%
for cmd in ["py-logit", "py-mlogit", "py-polr", "py-margins", "py-ivregress", "py-psmatch2", "py-mediate"]:
    e = sv.registry.get(cmd)
    print(f"  {cmd:14s} -> sv.tl.{e.name}" if e else f"  {cmd}: (未命中)")
