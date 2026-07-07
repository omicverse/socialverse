# %% [markdown]
# # 回归底座与因果工具箱:GLM、工具变量、匹配与中介
#
# 前面几本教程覆盖的是专门方法(DiD、断点、SEM、生存……)。这一本补的是社会科学里**最日常**的一批命令——广义回归、工具变量、倾向得分匹配、中介分析。它们是 R 用户的 `glm`/`ivreg`/`MatchIt`/`mediation`、Stata 用户的 `logit`/`ivregress`/`psmatch2`/`mediate`,几乎每篇实证论文都会用到。
#
# 这些方法在 socialverse 里和别的函数一样,都登记进注册表、operate on 同一个 `StudyState`,并且每个都用带**已知真值**的合成数据实现,跑一遍就能看到它把参数还原出来。如果你来自 R/Stata/SPSS,可以直接用熟悉的命令名查:`sv.registry.get("py-logit")`、`py-ivregress`、`py-psmatch2`、`py-mediate` 都能命中。

# %%
import socialverse as sv
from socialverse import datasets as ds

# %% [markdown]
# ## 广义线性回归:一个 `glm` 覆盖 OLS / logit / poisson
#
# 社会科学的结果变量有连续的(收入)、二值的(是否投票)、计数的(抗议次数)。`glm` 用一个 `family` 参数把它们统一:`gaussian` 就是 OLS,`binomial` 就是 logit,`poisson` 就是计数模型。合成数据 `load_regression` 里,连续结果 `y` 的真实系数是 x1=0.5、x2=-0.4,二值结果 `y_bin` 的 logit 真值是 x1=0.8、x2=-0.5,计数结果 `y_count` 的 poisson 真值是 x1=0.4。

# %%
df = ds.load_regression()

st = sv.StudyState()
sv.pp.ingest(st, data=df)
st.write("variables", "outcome", "y")
sv.tl.glm(st, predictors=["x1", "x2"], family="gaussian")   # OLS
print("OLS 系数:", {k: round(v, 3) for k, v in st.models["glm"]["coef"].items()}, "  (真值 x1=0.5, x2=-0.4)")

# %%
st_logit = sv.StudyState()
sv.pp.ingest(st_logit, data=df)
st_logit.write("variables", "outcome", "y_bin")
sv.tl.glm(st_logit, predictors=["x1", "x2"], family="binomial")   # logit
print("logit 系数:", {k: round(v, 3) for k, v in st_logit.models["glm"]["coef"].items()}, "  (真值 x1=0.8, x2=-0.5)")

# %% [markdown]
# ## 非线性模型要看边际效应,不是系数
#
# logit 的系数是对数几率尺度,不能直接当"x1 增加一单位,概率变化多少"来读。`margins` 在刚拟合的模型上算**平均边际效应**——这正是 Stata 的 `margins`、R 的 `marginaleffects`。它读取上一步存下的模型,给出概率尺度上的可解释效应。

# %%
sv.tl.margins(st_logit, model="glm")
print("平均边际效应(概率尺度):", {k: round(v, 3) for k, v in st_logit.diagnostics["margins"]["ame"].items()})

# %% [markdown]
# ## 工具变量:当回归量内生时
#
# 如果关键回归量和误差项相关(遗漏变量、反向因果、测量误差),OLS 会有偏。工具变量用一个只通过该回归量影响结果的"工具"来剥离内生性。合成数据 `load_iv` 里,`x` 被未观测混杂污染,真实因果效应是 1.5,但 OLS 会高估;用工具 `z` 做两阶段最小二乘(2SLS)能把它拉回来。

# %%
st_iv = sv.StudyState()
sv.pp.ingest(st_iv, data=ds.load_iv())
st_iv.write("variables", "outcome", "y")
sv.tl.iv_regress(st_iv, endogenous="x", instruments=["z"], exog=["w"])

fs = st_iv.diagnostics["first_stage"]
print(f"IV 估计 x 的效应 = {st_iv.models['iv']['coef']['x']:.3f}   (真值 1.5)")
print(f"对比:有偏的 OLS  = {fs['ols_endog_coef']:.3f}   (被混杂推高)")
print(f"一阶段 F = {fs['F']:.0f}   (远大于 10 → 工具够强,非弱工具)")

# %% [markdown]
# ## 倾向得分匹配:从观察数据估处理效应
#
# 没有随机分配时,处理组和对照组在协变量上往往不可比。倾向得分匹配先用协变量估"接受处理的概率",再把处理单元和倾向相近的对照单元配对,从而在可比的子样本上估平均处理效应(ATT)。合成数据 `load_treatment` 里,处理由 x1..x3 驱动(选择性偏差),真实 ATT=2.0,而朴素的处理组减对照组差值是有偏的。

# %%
st_psm = sv.StudyState()
sv.pp.ingest(st_psm, data=ds.load_treatment())
st_psm.write("variables", "outcome", "y")
st_psm.write("design", "treatment", "treat")

sv.tl.psm(st_psm, covariates=["x1", "x2", "x3"], method="nn")    # 最近邻匹配
m = st_psm.models["psm"]
print(f"匹配后 ATT = {m['att']:.3f}   (真值 2.0)")
print(f"朴素差值   = {m['naive_diff']:.3f}   (有偏)")

sv.tl.psm(st_psm, covariates=["x1", "x2", "x3"], method="ipw")   # 逆概率加权
print(f"IPW 估 ATT = {st_psm.models['psm']['att']:.3f}")

# %% [markdown]
# 匹配的意义在于让协变量在两组间可比。诊断里给出匹配前后的标准化均值差(SMD),匹配后应明显变小:

# %%
bal = st_psm.diagnostics["balance"]
for cov in bal["smd_before"]:
    print(f"  {cov}: 匹配前 SMD={bal['smd_before'][cov]:+.3f}  →  匹配后 SMD={bal['smd_after'][cov]:+.3f}")

# %% [markdown]
# ## 因果中介:效应经由什么传导
#
# 知道 X 影响 Y 还不够,常常要问"这个影响有多少是**经由**中介 M 传导的"。中介分析把总效应拆成:经由 M 的间接效应(ACME)和绕过 M 的直接效应(ADE)。合成数据 `load_mediation` 里,x→m 的路径 a=0.6、m→y 的路径 b=0.7,所以间接效应 = a·b = 0.42,直接效应 = 0.30,总效应 = 0.72。

# %%
st_med = sv.StudyState()
sv.pp.ingest(st_med, data=ds.load_mediation())
st_med.write("variables", "outcome", "y")
sv.tl.mediation(st_med, treatment="x", mediator="m", boot=500, seed=0)

m = st_med.models["mediation"]
print(f"间接效应 ACME = {m['acme']:.3f}   (真值 0.42)")
print(f"直接效应 ADE  = {m['ade']:.3f}   (真值 0.30)")
print(f"总效应 total  = {m['total']:.3f}   (真值 0.72)")
print(f"中介占比      = {m['prop_mediated']:.1%}")

# %% [markdown]
# ## 小结
#
# 这一本补齐了社会科学最高频的量化底座:广义回归(`glm` 覆盖 OLS/logit/poisson)、多项与有序结果(`mlogit`/`ologit`)、边际效应(`margins`)、工具变量(`iv_regress`)、倾向得分匹配(`psm`)、因果中介(`mediation`)。加上前面的专门方法,socialverse 现在覆盖了从"基础回归"到"前沿准实验"的主干。
#
# 它们对标 R 的 `glm`/`ivreg`/`MatchIt`/`mediation`、Stata 的 `logit`/`ivregress`/`psmatch2`/`mediate`——如果你习惯那些命令名,`sv.registry.get("py-ivregress")` 之类会直接带你找到对应函数。每个方法都在带已知真值的数据上验证过,不是占位实现。

# %%
print("本 notebook 覆盖的函数已全部注册,可在注册表查到:")
for name in ["glm", "mlogit", "ologit", "margins", "iv_regress", "psm", "mediation"]:
    e = sv.registry.get(name)
    print(f"  sv.tl.{name:12s} · {e.description[:42]}")
